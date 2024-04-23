#pragma once
#include "error.cuh"
#include <cub/cub.cuh>
#include <assert.h>
#include <cuda_fp16.h>

namespace tinycudallama
{

namespace
{
    constexpr int block_size = 128;
}

template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return max(a, b); }
};

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T warpAllReduce(T val)
{
    auto func = ReductionOp<T>();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val = func(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockAllReduceSum(T val)
{
    static __shared__ T shared[32];
    __shared__ T result;
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpAllReduce<SumOp, T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = warpAllReduce<SumOp, T>(val);
    if (threadIdx.x == 0)
        result = val;
    __syncthreads();
    return result;
}

template <typename T>
__inline__ __device__ T blockAllReduceMax(T val)
{
    static __shared__ T shared[32];
    __shared__ T result;
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpAllReduce<MaxOp, T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(-1e9f);
    val = warpAllReduce<MaxOp, T>(val);
    if (threadIdx.x == 0)
        result = val;
    __syncthreads();
    return result;
}

template <typename DataType>
__global__ void resNormKernel(DataType *__restrict__ output, const DataType *__restrict__ input,
                                const DataType *__restrict__ gamma, const float eps, const int hidden_units)
{
    const int offset = blockIdx.x * hidden_units;
    float mean;
    float val = 0.0f;
    for (int i = threadIdx.x; i < hidden_units; i += blockDim.x)
    {
        val += input[offset + i] * input[offset + i];
    }
    __syncthreads();

    val = blockAllReduceSum<float>(val);
    mean = rsqrtf(val / hidden_units + eps);
    // __syncthreads();

    for (int i = threadIdx.x; i < hidden_units; i += blockDim.x)
    {
        output[offset + i] = (DataType)(mean * input[offset + i] * gamma[i]);
    }
}

template <>
__global__ void resNormKernel(half *__restrict__ output, const half *__restrict__ input,
                                const half *__restrict__ gamma, const float eps, const int hidden_units)
{
    const int offset = blockIdx.x * hidden_units;
    half2 *out_ptr = (half2 *)(output + offset);
    const half2 *inp_ptr = (const half2 *)(input + offset);
    const half2 *gamma_ptr = (const half2 *)gamma;

    float mean = 0.0f;
    float2 val;

    for (int i = threadIdx.x; i < (hidden_units >> 1); i += blockDim.x)
    {
        val = __half22float2(inp_ptr[i]);
        mean += val.x * val.x + val.y * val.y;
    }
    __syncthreads();

    mean = blockAllReduceSum<float>(mean);
    mean = rsqrtf(mean / hidden_units + eps);

    float2 scale;

    for (int i = threadIdx.x; i < (hidden_units >> 1); i += blockDim.x)
    {
        val = __half22float2(inp_ptr[i]);
        scale = __half22float2(gamma_ptr[i]);
        val.x *= (mean * scale.x);
        val.y *= (mean * scale.y);
        out_ptr[i] = __float22half2_rn(val);
    }
}

template <typename DataType>
void launchResNormKernel(DataType *output, const DataType *input, const DataType *gamma, const float eps,
                            const int m, const int n, cudaStream_t stream = 0)
{
    dim3 grid(m);
    dim3 block(block_size);
    resNormKernel<DataType><<<grid, block, 0, stream>>>(output, input, gamma, eps, n);
}

#define WARP_SIZE 32
// kernel code
static __global__ void rms_norm_f32(const float *x, float *dst, const int ncols, const float *gamma, const float eps)
{
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp
    // 一个线程求和(ncols/WARP_SIZE)个数据的x^2
    for (int col = tid; col < ncols; col += WARP_SIZE)
    {
        const float xi = x[row * ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    // 一个线程束(32个线程)内的归约求和
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const float mean = tmp / ncols;         // mean(x^2)
    const float scale = rsqrtf(mean + eps); // 1/根号mean
    // 算完之后写回原数组
    for (int col = tid; col < ncols; col += WARP_SIZE)
    {
        dst[row * ncols + col] = scale * x[row * ncols + col] * gamma[col];
    }
}

// call kernel
static void rms_norm_f32_cuda(const float *x, float *dst, const int ncols, const int nrows, const float *gamma, const float eps, cudaStream_t stream = 0)
{
    assert(ncols % WARP_SIZE == 0);
    const dim3 block_dims(WARP_SIZE, 1, 1); //(32,1,1)
    // 所以调用的cuda的gridDim =(nrows,1,1) ,blockDim = (32,1,1)
    // 也就是说一个block处理一个row的数据，即每32个线程处理一行数据 ，共计nrows行
    rms_norm_f32<<<nrows, block_dims, 0, stream>>>(x, dst, ncols, gamma, eps);
}

/**
 * grid(seq_len)  block(block_size) for size_per_head/2 >= block_size(128)
 */
__global__ void precomputeFreqsCis(float *freq_cis, const int size_per_head)
{
    int offset = blockIdx.x * size_per_head;
    for (int i = threadIdx.x; i < (size_per_head >> 1); i += blockDim.x)
    {
        float val = i * (-2.0f) / size_per_head;
        float theta = __powf(1e4f, val) * blockIdx.x;
        freq_cis[offset + 2 * i] = __cosf(theta);
        freq_cis[offset + 2 * i + 1] = __sinf(theta);
    }
}

/**
 * block(32, 4)   each warp compute one row
 */
__global__ void warpPrecomputeFreqsCis(float *freq_cis, const int size_per_head, const int seq_len)
{
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    int offset = row * size_per_head;
    if (row < seq_len)
    {
        for (int i = threadIdx.x; i < (size_per_head >> 1); i += blockDim.x)
        {
            float val = i * (-2.0f) / size_per_head;
            float theta = __powf(1e4f, val) * row;
            freq_cis[offset + 2 * i] = __cosf(theta);
            freq_cis[offset + 2 * i + 1] = __sinf(theta);
        }
    }
}

void launchPrecomputeFreqsCis(float *freq_cis, const int size_per_head, const int seq_len, cudaStream_t stream = 0)
{
    if ((size_per_head / 2) < 128)
    {
        int warp_num = block_size / 32;
        int grid_size = (seq_len + warp_num - 1) / warp_num;
        dim3 grid(grid_size);
        dim3 block(32, warp_num);
        warpPrecomputeFreqsCis<<<grid, block, 0, stream>>>(freq_cis, size_per_head, seq_len);
    }
    else
    {
        dim3 grid(seq_len);
        dim3 block(block_size);
        precomputeFreqsCis<<<grid, block, 0, stream>>>(freq_cis, size_per_head);
    }
}

/**
 * from_tensor: [batch_size, seq_len, hidden_units]
 * word_ids:    [batch_size, seq_len]
 */
template <typename DataType>
__global__ void embeddingLookingUpKernel(DataType *__restrict__ from_tensor, const DataType *__restrict__ embedding_table,
                                            const int *__restrict__ word_ids, const int hidden_units, const int seq_len)
{
    const int batch_id = blockIdx.x;
    const int seq_id = blockIdx.y;
    const int offset = batch_id * seq_len * hidden_units + seq_id * hidden_units;
    const int id_for_word = word_ids[batch_id * seq_len + seq_id];
    for (int i = threadIdx.x; i < hidden_units; i += blockDim.x)
    {
        from_tensor[offset + i] = embedding_table[id_for_word * hidden_units + i];
    }
}

template <typename DataType>
void launchEmbeddingLookingUpKernel(DataType *from_tensor, const DataType *embedding_table,
                                    const int *word_ids, const int hidden_units, const int batch_size, const int seq_len, cudaStream_t stream = 0)
{
    dim3 grid(batch_size, seq_len);
    dim3 block(block_size);
    embeddingLookingUpKernel<DataType><<<grid, block, 0, stream>>>(from_tensor, embedding_table, word_ids, hidden_units, seq_len);
}

static inline __device__ int8_t float_to_int8_rn(float x)
{
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;"
                    : "=r"(dst)
                    : "f"(x));
    return reinterpret_cast<const int8_t &>(dst);
}

template <typename DataType>
__global__ void perChannelQuantizedKernel(int8_t *__restrict__ dst, const DataType *__restrict__ src, float *__restrict__ scale_ptr,
                                            const int hidden_size)
{
    const int offset = blockIdx.x * hidden_size;
    float absmax = 0.0f;
    for (int i = (threadIdx.x << 2); i < hidden_size; i += (blockDim.x << 2))
    {
        absmax = fmaxf(absmax, fabsf(static_cast<float>(__ldg(&src[offset + i]))));
        absmax = fmaxf(absmax, fabsf(static_cast<float>(__ldg(&src[offset + i + 1]))));
        absmax = fmaxf(absmax, fabsf(static_cast<float>(__ldg(&src[offset + i + 2]))));
        absmax = fmaxf(absmax, fabsf(static_cast<float>(__ldg(&src[offset + i + 3]))));
    }
    __syncthreads();
    absmax = blockAllReduceMax<float>(absmax);
    float scale = 127.0f / absmax;

    char4 *dst_ptr4 = (char4 *)(dst + offset);
    char4 tmp;
    for (int i = (threadIdx.x << 2); i < hidden_size; i += (blockDim.x << 2))
    {
        tmp.x = float_to_int8_rn(static_cast<float>(__ldg(&src[offset + i])) * scale);
        tmp.y = float_to_int8_rn(static_cast<float>(__ldg(&src[offset + i + 1])) * scale);
        tmp.z = float_to_int8_rn(static_cast<float>(__ldg(&src[offset + i + 2])) * scale);
        tmp.w = float_to_int8_rn(static_cast<float>(__ldg(&src[offset + i + 3])) * scale);
        dst_ptr4[i >> 2] = tmp;
    }
    if (threadIdx.x == 0)
    {
        scale_ptr[blockIdx.x] = absmax / 127.0f;
    }
}

template <typename DataType>
void perChannelQuantizedKernelLauncher(int8_t *dst, const DataType *src, float *scale_ptr, const int hidden_size,
                                        const int nrows, cudaStream_t stream = 0)
{
    dim3 grid(nrows);
    dim3 block(block_size);
    perChannelQuantizedKernel<DataType><<<grid, block, 0, stream>>>(dst, src, scale_ptr, hidden_size);
}

/**
 * 反量化、rope旋转编码、量化、转置
 * Q K: [batch_size, seq_len, head_num, size_per_head]
 * grid(head_num / warp_num, seq_len, batch_size * 2) block(32, warp_num), each warp process size_per_head elements
 * q_inp_sacle k_inp_scale: [batch_size, seq_len], absmax / 127.0f
 * q_weight_scale k_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
 * freq_cis: [max_seq_len, size_per_head]
 * q_out_scale k_out_scale: [batch_size, seq_len, head_num], absmax / 127.0f
 */
__global__ void warpQKRoteEmbeddingQuantizedTransposeKernel(int8_t *q_buf, int8_t *k_buf, const int32_t *Q,
                                                            const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                            const float *q_weight_scale, const float *k_weight_scale, float *q_out_scale,
                                                            float *k_out_scale, float *freq_cis, const int batch_size, const int seq_len, const int head_num,
                                                            const int size_per_head, const int warp_num)
{
    const int qk_id = blockIdx.z / batch_size;
    const int batch_id = blockIdx.z % batch_size;
    const int seq_id = blockIdx.y;
    const int head_id = blockIdx.x * warp_num + threadIdx.y;
    const int offset = batch_id * seq_len * head_num * size_per_head + seq_id * head_num * size_per_head + head_id * size_per_head;
    const float inp_scale_val = (qk_id == 0) ? __ldg(q_inp_scale + batch_id * seq_len + seq_id) : __ldg(k_inp_scale + batch_id * seq_len + seq_id);
    const int32_t *data_ptr = (qk_id == 0) ? Q + offset : K + offset;
    const float *weight_scale_ptr = (qk_id == 0) ? q_weight_scale : k_weight_scale;
    const float *freq_cis_ptr = freq_cis + seq_id * size_per_head;
    float *out_scale_ptr = (qk_id == 0) ? q_out_scale : k_out_scale;
    char4 *out_ptr = (qk_id == 0) ? (char4 *)q_buf : (char4 *)k_buf;
    float4 val, rope_val;
    char4 out_val;
    float absmax = 0.0f;
    const int tid = (threadIdx.x << 2);
    float out_scale;
    int target_idx;
    if (tid < size_per_head)
    {
        // dequantized
        val.x = static_cast<float>(data_ptr[tid]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid);
        val.y = static_cast<float>(data_ptr[tid + 1]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 1);
        val.z = static_cast<float>(data_ptr[tid + 2]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 2);
        val.w = static_cast<float>(data_ptr[tid + 3]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 3);

        // rope embedding
        rope_val.x = val.x * freq_cis_ptr[tid] - val.y * freq_cis_ptr[tid + 1];
        rope_val.y = val.y * freq_cis_ptr[tid] + val.x * freq_cis_ptr[tid + 1];
        rope_val.z = val.z * freq_cis_ptr[tid + 2] - val.w * freq_cis_ptr[tid + 3];
        rope_val.w = val.w * freq_cis_ptr[tid + 2] + val.z * freq_cis_ptr[tid + 3];

        // quantized
        absmax = fmaxf(absmax, fmaxf(fabsf(rope_val.x), fmaxf(fabsf(rope_val.y), fmaxf(fabsf(rope_val.z), fabsf(rope_val.w)))));
        __syncwarp();
        absmax = warpAllReduce<MaxOp, float>(absmax);
        if (tid == 0)
        {
            out_scale_ptr[batch_id * head_num * seq_len + head_id * seq_len + seq_id] = absmax / 127.0f;
        }
        out_scale = 127.0f / absmax;
        out_val.x = float_to_int8_rn(static_cast<float>(rope_val.x) * out_scale);
        out_val.y = float_to_int8_rn(static_cast<float>(rope_val.y) * out_scale);
        out_val.z = float_to_int8_rn(static_cast<float>(rope_val.z) * out_scale);
        out_val.w = float_to_int8_rn(static_cast<float>(rope_val.w) * out_scale);

        // transpose
        target_idx = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head + tid;
        out_ptr[target_idx >> 2] = out_val;
    }
}

/**
 * 反量化、rope旋转编码、量化、转置
 * Q K: [batch_size, seq_len, head_num, size_per_head]
 * grid(head_num, seq_len, batch_size * 2) block(128), each block process size_per_head(256) elements
 * q_inp_sacle k_inp_scale: [batch_size, seq_len], absmax / 127.0f
 * q_weight_scale k_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
 * freq_cis: [max_seq_len, size_per_head]
 * q_out_scale k_out_scale: [batch_size, seq_len, head_num], absmax / 127.0f
 */
__global__ void blockQKRoteEmbeddingQuantizedTransposeForDim256Kernel(int8_t *q_buf, int8_t *k_buf, const int32_t *Q,
                                                                        const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                                        const float *q_weight_scale, const float *k_weight_scale, float *q_out_scale,
                                                                        float *k_out_scale, float *freq_cis, const int batch_size, const int seq_len, const int head_num,
                                                                        const int size_per_head)
{
    const int qk_id = blockIdx.z / batch_size;
    const int batch_id = blockIdx.z % batch_size;
    const int seq_id = blockIdx.y;
    const int head_id = blockIdx.x;
    const int offset = batch_id * seq_len * head_num * size_per_head + seq_id * head_num * size_per_head + head_id * size_per_head;
    const float inp_scale_val = (qk_id == 0) ? __ldg(q_inp_scale + batch_id * seq_len + seq_id) : __ldg(k_inp_scale + batch_id * seq_len + seq_id);
    const int32_t *data_ptr = (qk_id == 0) ? Q + offset : K + offset;
    const float *weight_scale_ptr = (qk_id == 0) ? q_weight_scale : k_weight_scale;
    const float *freq_cis_ptr = freq_cis + seq_id * size_per_head;
    float *out_scale_ptr = (qk_id == 0) ? q_out_scale : k_out_scale;
    char2 *out_ptr = (qk_id == 0) ? (char2 *)q_buf : (char2 *)k_buf;
    float2 val, rope_val;
    char2 out_val;
    float absmax = 0.0f;
    const int tid = (threadIdx.x << 1);
    float out_scale;
    int target_idx;
    // dequantized
    val.x = static_cast<float>(data_ptr[tid]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid);
    val.y = static_cast<float>(data_ptr[tid + 1]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 1);

    // rope embedding
    rope_val.x = val.x * freq_cis_ptr[tid] - val.y * freq_cis_ptr[tid + 1];
    rope_val.y = val.y * freq_cis_ptr[tid] + val.x * freq_cis_ptr[tid + 1];

    // quantized
    absmax = fmaxf(absmax, fmaxf(fabsf(rope_val.x), fabsf(rope_val.y)));
    __syncthreads();
    absmax = blockAllReduceMax<float>(absmax);
    if (tid == 0)
    {
        out_scale_ptr[batch_id * head_num * seq_len + head_id * seq_len + seq_id] = absmax / 127.0f;
    }
    out_scale = 127.0f / absmax;
    out_val.x = float_to_int8_rn(static_cast<float>(rope_val.x) * out_scale);
    out_val.y = float_to_int8_rn(static_cast<float>(rope_val.y) * out_scale);

    // transpose
    target_idx = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head + tid;
    out_ptr[target_idx >> 1] = out_val;
}

/**
 * 反量化、rope旋转编码、量化、转置
 * Q K: [batch_size, seq_len, head_num, size_per_head]
 * grid(head_num, seq_len, batch_size * 2) block(size_per_head / 4), each block process size_per_head elements
 * q_inp_sacle k_inp_scale: [batch_size, seq_len], absmax / 127.0f
 * q_weight_scale k_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
 * freq_cis: [max_seq_len, size_per_head]
 * q_out_scale k_out_scale: [batch_size, head_num, seq_len], absmax / 127.0f
 */
__global__ void blockQKRoteEmbeddingQuantizedTransposeKernel(int8_t *q_buf, int8_t *k_buf, const int32_t *Q,
                                                                const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                                const float *q_weight_scale, const float *k_weight_scale, float *q_out_scale,
                                                                float *k_out_scale, float *freq_cis, const int batch_size, const int seq_len, const int head_num,
                                                                const int size_per_head)
{
    const int qk_id = blockIdx.z / batch_size;
    const int batch_id = blockIdx.z % batch_size;
    const int seq_id = blockIdx.y;
    const int head_id = blockIdx.x;
    const int offset = batch_id * seq_len * head_num * size_per_head + seq_id * head_num * size_per_head + head_id * size_per_head;
    const float inp_scale_val = (qk_id == 0) ? __ldg(q_inp_scale + batch_id * seq_len + seq_id) : __ldg(k_inp_scale + batch_id * seq_len + seq_id);
    const int32_t *data_ptr = (qk_id == 0) ? Q + offset : K + offset;
    const float *weight_scale_ptr = (qk_id == 0) ? q_weight_scale : k_weight_scale;
    const float *freq_cis_ptr = freq_cis + seq_id * size_per_head;
    float *out_scale_ptr = (qk_id == 0) ? q_out_scale : k_out_scale;
    char4 *out_ptr = (qk_id == 0) ? (char4 *)q_buf : (char4 *)k_buf;
    float4 val, rope_val;
    char4 out_val;
    float absmax = 0.0f;
    const int tid = (threadIdx.x << 2);
    float out_scale;
    int target_idx;

    // dequantized
    val.x = static_cast<float>(data_ptr[tid]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid);
    val.y = static_cast<float>(data_ptr[tid + 1]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 1);
    val.z = static_cast<float>(data_ptr[tid + 2]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 2);
    val.w = static_cast<float>(data_ptr[tid + 3]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 3);

    // rope embedding
    rope_val.x = val.x * freq_cis_ptr[tid] - val.y * freq_cis_ptr[tid + 1];
    rope_val.y = val.y * freq_cis_ptr[tid] + val.x * freq_cis_ptr[tid + 1];
    rope_val.z = val.z * freq_cis_ptr[tid + 2] - val.w * freq_cis_ptr[tid + 3];
    rope_val.w = val.w * freq_cis_ptr[tid + 2] + val.z * freq_cis_ptr[tid + 3];

    // quantized
    absmax = fmaxf(absmax, fmaxf(fabsf(rope_val.x), fmaxf(fabsf(rope_val.y), fmaxf(fabsf(rope_val.z), fabsf(rope_val.w)))));
    __syncthreads();
    absmax = blockAllReduceMax<float>(absmax);
    if (tid == 0)
    {
        out_scale_ptr[batch_id * head_num * seq_len + head_id * seq_len + seq_id] = absmax / 127.0f;
    }
    out_scale = 127.0f / absmax;
    out_val.x = float_to_int8_rn(static_cast<float>(rope_val.x) * out_scale);
    out_val.y = float_to_int8_rn(static_cast<float>(rope_val.y) * out_scale);
    out_val.z = float_to_int8_rn(static_cast<float>(rope_val.z) * out_scale);
    out_val.w = float_to_int8_rn(static_cast<float>(rope_val.w) * out_scale);

    // transpose
    target_idx = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head + tid;
    out_ptr[target_idx >> 2] = out_val;
}

void launchQKRoteEmbeddingQuantizedTranspose(int8_t *q_buf, int8_t *k_buf, const int32_t *Q,
                                                const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                const float *q_weight_scale, const float *k_weight_scale, float *q_out_scale,
                                                float *k_out_scale, float *freq_cis, const int batch_size, const int seq_len, const int head_num,
                                                const int size_per_head, cudaStream_t stream = 0)
{
    assert(size_per_head <= 1024);
    if (size_per_head <= 128)
    {
        int warp_num = 4;
        dim3 grid(head_num / warp_num, seq_len, batch_size * 2);
        dim3 block(32, warp_num);
        warpQKRoteEmbeddingQuantizedTransposeKernel<<<grid, block, 0, stream>>>(q_buf, k_buf, Q, K, q_inp_scale, k_inp_scale,
                                                                                q_weight_scale, k_weight_scale, q_out_scale, k_out_scale, freq_cis, batch_size, seq_len, head_num, size_per_head, warp_num);
    }
    else if (size_per_head == 256)
    {
        dim3 grid(head_num, seq_len, batch_size * 2);
        dim3 block(128);
        blockQKRoteEmbeddingQuantizedTransposeForDim256Kernel<<<grid, block, 0, stream>>>(q_buf, k_buf, Q, K, q_inp_scale, k_inp_scale,
                                                                                            q_weight_scale, k_weight_scale, q_out_scale, k_out_scale, freq_cis, batch_size, seq_len, head_num, size_per_head);
    }
    else if (size_per_head == 512 || size_per_head == 1024)
    {
        dim3 grid(head_num, seq_len, batch_size * 2);
        dim3 block(size_per_head / 4);
        blockQKRoteEmbeddingQuantizedTransposeKernel<<<grid, block, 0, stream>>>(q_buf, k_buf, Q, K, q_inp_scale, k_inp_scale,
                                                                                    q_weight_scale, k_weight_scale, q_out_scale, k_out_scale, freq_cis, batch_size, seq_len, head_num, size_per_head);
    }
    else {
        throw "invalid size_per_head!";
    }
}

/**
 * grid: [seq_len, head_num / blockDim.y, batch_size * 2]  block(size_per_head / 4, 256 / (size_per_head / 4))
 * k_cache v_cache: [batch_size, head_num, max_seq_len, size_per_head]
 * K V : [batch_size, head_num, seq_len, size_per_head]
 */
__global__ void storeKVcacheKernel(float *__restrict__ k_cache, float *__restrict__ v_cache, const float *__restrict__ K,
                                    const float *__restrict__ V, const int start_pos, const int seq_len, const int batch_size, const int head_num,
                                    const int max_seq_len, const int size_per_head)
{
    const int kv_id = blockIdx.z / batch_size;
    const int batch_id = blockIdx.z % batch_size;
    const int head_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int seq_id = blockIdx.x;
    const int offset = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head;
    const int cache_offset = batch_id * head_num * max_seq_len * size_per_head + head_id * max_seq_len * size_per_head +
                                (start_pos + seq_id) * size_per_head;
    float4 *cache_ptr = (kv_id == 0) ? (float4 *)(k_cache + cache_offset) : (float4 *)(v_cache + cache_offset);
    const float4 *data_ptr = (kv_id == 0) ? (const float4 *)(K + offset) : (const float4 *)(V + offset);
    cache_ptr[threadIdx.x] = data_ptr[threadIdx.x];
}

void launchStoreKVcacheKernel(float *k_cache, float *v_cache, const float *K, const float *V, const int start_pos, const int seq_len,
                                const int batch_size, const int head_num, const int max_seq_len, const int size_per_head, cudaStream_t stream = 0)
{
    assert(size_per_head <= 1024);
    dim3 block, grid;
    block.x = size_per_head / 4;
    assert(block.x >= 1);
    block.y = 256 / block.x;
    grid.x = seq_len;
    grid.y = head_num / block.y;
    assert(grid.y >= 1);
    grid.z = batch_size * 2;

    storeKVcacheKernel<<<grid, block, 0, stream>>>(k_cache, v_cache, K, V, start_pos, seq_len, batch_size, head_num, max_seq_len,
                                                            size_per_head);
}

/**
 * grid: [seq_len, head_num / blockDim.y, batch_size * 2]  block(size_per_head / 4, 256 / (size_per_head / 4))
 * k_cache v_cache: [batch_size, head_num, max_seq_len, size_per_head]
 * K V : [batch_size, head_num, seq_len, size_per_head]
 */
__global__ void storeINT8KVcacheKernel(int8_t *__restrict__ k_cache, int8_t *__restrict__ v_cache, const int8_t *__restrict__ K,
                                    const int8_t *__restrict__ V, const int start_pos, const int seq_len, const int batch_size, const int head_num,
                                    const int max_seq_len, const int size_per_head)
{
    const int kv_id = blockIdx.z / batch_size;
    const int batch_id = blockIdx.z % batch_size;
    const int head_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int seq_id = blockIdx.x;
    const int offset = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head;
    const int cache_offset = batch_id * head_num * max_seq_len * size_per_head + head_id * max_seq_len * size_per_head +
                                (start_pos + seq_id) * size_per_head;
    char4 *cache_ptr = (kv_id == 0) ? (char4 *)(k_cache + cache_offset) : (char4 *)(v_cache + cache_offset);
    const char4 *data_ptr = (kv_id == 0) ? (const char4 *)(K + offset) : (const char4 *)(V + offset);
    cache_ptr[threadIdx.x] = data_ptr[threadIdx.x];
}

void launchINT8StoreKVcacheKernel(int8_t *k_cache, int8_t *v_cache, const int8_t *K, const int8_t *V, const int start_pos, const int seq_len,
                                const int batch_size, const int head_num, const int max_seq_len, const int size_per_head, cudaStream_t stream = 0)
{
    assert(size_per_head <= 1024);
    dim3 block, grid;
    block.x = size_per_head / 4;
    assert(block.x >= 1);
    block.y = 256 / block.x;
    grid.x = seq_len;
    grid.y = head_num / block.y;
    assert(grid.y >= 1);
    grid.z = batch_size * 2;

    storeINT8KVcacheKernel<<<grid, block, 0, stream>>>(k_cache, v_cache, K, V, start_pos, seq_len, batch_size, head_num, max_seq_len,
                                                            size_per_head);
}


/**
 * 反量化、softmax、量化
 * grid(seq_len_q, head_num, batch_size), block(128), each block process seq_len_k elements
 * qk score: [batch_size, head_num, seq_len_q, seq_len_k]
 * atten_mask: [max_seq_len, max_seq_len]
 * q_inp_scale: [batch_size, head_num, seq_len_q]
 * k_inp_scale: [batch_size, head_num, seq_len_k]
 * score_scale: [batch_size, head_num, seq_len_q]
 * 
*/
__global__ void blockSoftmaxKernel(int8_t * __restrict__ score, const int32_t * __restrict__ qk, const float * __restrict__ attn_mask, 
    const float * __restrict__ q_inp_scale, const float * __restrict__ k_inp_scale, float * __restrict__ score_scale,
    const float attn_scale, const int batch_size, const int head_num, const int seq_len_q, const int seq_len_k, const int max_seq_len)
{
    const int batch_id = blockIdx.z;
    const int head_id = blockIdx.y;
    const int seq_q_id = blockIdx.x;
    const int offset = batch_id * head_num * seq_len_q * seq_len_k + head_id * seq_len_q * seq_len_k + seq_q_id * seq_len_k;
    const int k_scale_offset = batch_id * head_num * seq_len_k + head_id * seq_len_k;
    const float q_inp_scale_val = q_inp_scale[batch_id * head_num * seq_len_q + head_id * seq_len_q + seq_q_id];
    const int mask_offset = seq_q_id * max_seq_len;
    extern __shared__ float s_buf[];
    float val, mask_val;
    float sum_val = 0.0f;
    float max_val = -1e9f;
    for (int i=threadIdx.x; i<seq_len_k; i+=blockDim.x) {
        // dequantized
        val = static_cast<float>(qk[offset + i]) * q_inp_scale_val * k_inp_scale[k_scale_offset + i];
        mask_val = (attn_mask) ? attn_mask[mask_offset + i] : 0.0f;
        val = val * attn_scale + mask_val;
        s_buf[i] = val;
        max_val = max(max_val, val);
    }
    __syncthreads();
    max_val = blockAllReduceMax<float>(max_val);
    
    for (int i=threadIdx.x; i<seq_len_k; i+=blockDim.x) {
        val = expf(s_buf[i] - max_val);
        sum_val += val;
        score[offset + i] = float_to_int8_rn(val * 127.0f);
    }
    __syncthreads();
    sum_val = blockAllReduceSum<float>(sum_val);

    if (threadIdx.x == 0) {
        score_scale[batch_id * head_num * seq_len_q + head_id * seq_len_q + seq_q_id] = 1.0f / (127.0f * (sum_val + 1e-7f));
    }
}


void launchBlockSoftmaxKernel(int8_t * score, const int32_t * qk, const float * attn_mask, const float * q_inp_scale, 
    const float * k_inp_scale, float * score_scale, const float attn_scale, const int batch_size, const int head_num, 
    const int seq_len_q, const int seq_len_k, const int max_seq_len, cudaStream_t stream = 0)
{
    dim3 grid(seq_len_q, head_num, batch_size);
    dim3 block(128);
    int shared_mem_size = sizeof(float) * seq_len_k;
    blockSoftmaxKernel<<<grid, block, shared_mem_size, stream>>>(score, qk, attn_mask, q_inp_scale, k_inp_scale, score_scale, attn_scale, 
        batch_size, head_num, seq_len_q, seq_len_k, max_seq_len);
}

/**
 * 反量化、量化、转置
 * grid(head_num / warp_num, seq_len, batch_size) block(32, warp_num), each warp process size_per_head elements
 * V: [batch_size, seq_len, head_num, size_per_head]
 * v_buf: [batch_size, head_num, seq_len, size_per_head]
 * v_inp_sacle: [batch_size, seq_len], absmax / 127.0f
 * v_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
 * v_out_scale k_out_scale: [batch_size, seq_len, head_num], absmax / 127.0f
*/
__global__ void warpDequantizedVTransposeQuantizedKernel(int8_t * __restrict__ v_buf, const int32_t * __restrict__ V,
    const float * __restrict__ v_inp_scale, const float * __restrict__ v_weight_scale, float * __restrict__ v_out_scale, 
    const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int warp_num)
{
    const int batch_id = blockIdx.z;
    const int seq_id = blockIdx.y;
    const int head_id = blockIdx.x * warp_num + threadIdx.y;
    const int offset = batch_id * seq_len * head_num * size_per_head + seq_id * head_num * size_per_head + head_id * size_per_head;
    const float inp_scale_val = __ldg(v_inp_scale + batch_id * seq_len + seq_id);
    const int32_t *data_ptr = V + offset;
    const float *weight_scale_ptr = v_weight_scale;
    float *out_scale_ptr = v_out_scale;
    char4 *out_ptr = (char4 *)v_buf;
    float4 val;
    char4 out_val;
    float absmax = 0.0f;
    const int tid = (threadIdx.x << 2);
    float out_scale;
    int target_idx;
    if (tid < size_per_head)
    {
        // dequantized
        val.x = static_cast<float>(data_ptr[tid]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid);
        val.y = static_cast<float>(data_ptr[tid + 1]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 1);
        val.z = static_cast<float>(data_ptr[tid + 2]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 2);
        val.w = static_cast<float>(data_ptr[tid + 3]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 3);

        // quantized
        absmax = fmaxf(absmax, fmaxf(fabsf(val.x), fmaxf(fabsf(val.y), fmaxf(fabsf(val.z), fabsf(val.w)))));
        __syncwarp();
        absmax = warpAllReduce<MaxOp, float>(absmax);
        if (tid == 0)
        {
            out_scale_ptr[batch_id * head_num * seq_len + head_id * seq_len + seq_id] = absmax / 127.0f;
        }
        out_scale = 127.0f / absmax;
        out_val.x = float_to_int8_rn(val.x * out_scale);
        out_val.y = float_to_int8_rn(val.y * out_scale);
        out_val.z = float_to_int8_rn(val.z * out_scale);
        out_val.w = float_to_int8_rn(val.w * out_scale);

        // transpose
        target_idx = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head + tid;
        out_ptr[target_idx >> 2] = out_val;
    }
}

/**
 * 反量化、量化、转置
 * grid(head_num, seq_len, batch_size) block(128), each block process size_per_head elements
 * V: [batch_size, seq_len, head_num, size_per_head]
 * v_buf: [batch_size, head_num, seq_len, size_per_head]
 * v_inp_sacle: [batch_size, seq_len], absmax / 127.0f
 * v_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
 * v_out_scale k_out_scale: [batch_size, seq_len, head_num], absmax / 127.0f
*/
__global__ void blockDequantizedVTransposeQuantizedFor256Kernel(int8_t * __restrict__ v_buf, const int32_t * __restrict__ V,
    const float * __restrict__ v_inp_scale, const float * __restrict__ v_weight_scale, float * __restrict__ v_out_scale, 
    const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    const int batch_id = blockIdx.z;
    const int seq_id = blockIdx.y;
    const int head_id = blockIdx.x;
    const int offset = batch_id * seq_len * head_num * size_per_head + seq_id * head_num * size_per_head + head_id * size_per_head;
    const float inp_scale_val = __ldg(v_inp_scale + batch_id * seq_len + seq_id);
    const int32_t *data_ptr = V + offset;
    const float *weight_scale_ptr = v_weight_scale;
    
    float *out_scale_ptr = v_out_scale;
    char2 *out_ptr = (char2 *)v_buf;
    float2 val;
    char2 out_val;
    float absmax = 0.0f;
    const int tid = (threadIdx.x << 1);
    float out_scale;
    int target_idx;
    
    // dequantized
    val.x = static_cast<float>(data_ptr[tid]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid);
    val.y = static_cast<float>(data_ptr[tid + 1]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 1);
    // quantized
    absmax = fmaxf(absmax, fmaxf(fabsf(val.x), fabsf(val.y)));
    __syncthreads();
    absmax = blockAllReduceMax<float>(absmax);
    if (tid == 0)
    {
        out_scale_ptr[batch_id * head_num * seq_len + head_id * seq_len + seq_id] = absmax / 127.0f;
    }
    out_scale = 127.0f / absmax;
    out_val.x = float_to_int8_rn(val.x * out_scale);
    out_val.y = float_to_int8_rn(val.y * out_scale);

    // transpose
    target_idx = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head + tid;
    out_ptr[target_idx >> 1] = out_val;
}

/**
 * 反量化、量化、转置
 * grid(head_num, seq_len, batch_size) block(size_per_head / 4), each block process size_per_head elements
 * V: [batch_size, seq_len, head_num, size_per_head]
 * v_buf: [batch_size, head_num, seq_len, size_per_head]
 * v_inp_sacle: [batch_size, seq_len], absmax / 127.0f
 * v_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
 * v_out_scale k_out_scale: [batch_size, seq_len, head_num], absmax / 127.0f
*/
__global__ void blockDequantizedVTransposeQuantizedKernel(int8_t * __restrict__ v_buf, const int32_t * __restrict__ V,
    const float * __restrict__ v_inp_scale, const float * __restrict__ v_weight_scale, float * __restrict__ v_out_scale, 
    const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    const int batch_id = blockIdx.z;
    const int seq_id = blockIdx.y;
    const int head_id = blockIdx.x;
    const int offset = batch_id * seq_len * head_num * size_per_head + seq_id * head_num * size_per_head + head_id * size_per_head;
    const float inp_scale_val = __ldg(v_inp_scale + batch_id * seq_len + seq_id);
    const int32_t *data_ptr = V + offset;
    const float *weight_scale_ptr = v_weight_scale;
    float *out_scale_ptr = v_out_scale;
    char4 *out_ptr = (char4 *)v_buf;
    float4 val;
    char4 out_val;
    float absmax = 0.0f;
    const int tid = (threadIdx.x << 2);
    float out_scale;
    int target_idx;
    
    // dequantized
    val.x = static_cast<float>(data_ptr[tid]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid);
    val.y = static_cast<float>(data_ptr[tid + 1]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 1);
    val.z = static_cast<float>(data_ptr[tid + 2]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 2);
    val.w = static_cast<float>(data_ptr[tid + 3]) * inp_scale_val * __ldg(weight_scale_ptr + head_id * size_per_head + tid + 3);

    // quantized
    absmax = fmaxf(absmax, fmaxf(fabsf(val.x), fmaxf(fabsf(val.y), fmaxf(fabsf(val.z), fabsf(val.w)))));
    __syncthreads();
    absmax = blockAllReduceMax<float>(absmax);
    if (tid == 0)
    {
        out_scale_ptr[batch_id * head_num * seq_len + head_id * seq_len + seq_id] = absmax / 127.0f;
    }
    out_scale = 127.0f / absmax;
    out_val.x = float_to_int8_rn(val.x * out_scale);
    out_val.y = float_to_int8_rn(val.y * out_scale);
    out_val.z = float_to_int8_rn(val.z * out_scale);
    out_val.w = float_to_int8_rn(val.w * out_scale);

    // transpose
    target_idx = batch_id * head_num * seq_len * size_per_head + head_id * seq_len * size_per_head + seq_id * size_per_head + tid;
    out_ptr[target_idx >> 2] = out_val;
}


void launchDequantizedVTransposeQuantizedKernel(int8_t * v_buf, const int32_t * V, const float * v_inp_scale, const float * v_weight_scale, 
    float * v_out_scale, const int batch_size, const int seq_len, const int head_num, const int size_per_head, cudaStream_t stream = 0)
{
    assert(size_per_head <= 1024);
    if (size_per_head <= 128) {
        int warp_num = 128 / 32;
        dim3 grid(head_num / warp_num, seq_len, batch_size);
        dim3 block(32, warp_num);
        warpDequantizedVTransposeQuantizedKernel<<<grid, block, 0, stream>>>(v_buf, V, v_inp_scale, v_weight_scale, v_out_scale, batch_size,
            seq_len, head_num, size_per_head, warp_num);
    }
    else if (size_per_head == 256) {
        dim3 grid(head_num, seq_len, batch_size);
        dim3 block(128);
        blockDequantizedVTransposeQuantizedFor256Kernel<<<grid, block, 0, stream>>>(v_buf, V, v_inp_scale, v_weight_scale, v_out_scale,
            batch_size, seq_len, head_num, size_per_head);
    }
    else if (size_per_head == 512 || size_per_head == 1024) {
        dim3 grid(head_num, seq_len, batch_size);
        dim3 block(size_per_head / 4);
        blockDequantizedVTransposeQuantizedKernel<<<grid, block, 0, stream>>>(v_buf, V, v_inp_scale, v_weight_scale, v_out_scale,
            batch_size, seq_len, head_num, size_per_head);
    }
    else {
        throw "invalid size_per_head!";
    }
}


} // tinycudallama