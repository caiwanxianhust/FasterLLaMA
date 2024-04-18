#pragma once
#include "error.cuh"
#include <cub/cub.cuh>
#include <assert.h>
#include <cuda_fp16.h>

namespace tinycudallama {

namespace {
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

template <template <typename> class ReductionOp, typename T>
__inline__ __device__
    T
    blockAllReduce(T val)
{
    static __shared__ T shared[32];
    __shared__ T result;
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpAllReduce<ReductionOp, T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = warpAllReduce<ReductionOp, T>(val);
    if (threadIdx.x == 0) result = val;
    __syncthreads();
    return result;
}


template <typename DataType>
__global__ void resNormKernel(DataType* __restrict__ output, const DataType* __restrict__ input, 
    const DataType* __restrict__ gamma, const float eps, const int hidden_units)
{
    const int offset = blockIdx.x * hidden_units;
    float mean;
    float val = 0.0f;
    for (int i=threadIdx.x; i<hidden_units; i+=blockDim.x) {
        val += input[offset + i] * input[offset + i];
    }
    __syncthreads();

    val = blockAllReduce<SumOp, float>(val);
    mean = rsqrtf(val / hidden_units + eps);
    // __syncthreads();

    for (int i=threadIdx.x; i<hidden_units; i+=blockDim.x) {
        output[offset + i] = (DataType)(mean * input[offset + i] * gamma[i]);
    }
}

template <>
__global__ void resNormKernel(half* __restrict__ output, const half* __restrict__ input, 
    const half* __restrict__ gamma, const float eps, const int hidden_units)
{
    const int offset = blockIdx.x * hidden_units;
    half2 *out_ptr = (half2 *)(output + offset);
    const half2 *inp_ptr = (const half2 *)(input + offset);
    const half2 *gamma_ptr = (const half2 *)gamma;
    
    float mean = 0.0f;
    float2 val;
    
    for (int i=threadIdx.x; i<(hidden_units >> 1); i+=blockDim.x) {
        val = __half22float2(inp_ptr[i]);
        mean += val.x * val.x + val.y * val.y;
    }
    __syncthreads();

    mean = blockAllReduce<SumOp, float>(mean);
    mean = rsqrtf(mean / hidden_units + eps);

    float2 scale;

    for (int i=threadIdx.x; i<(hidden_units >> 1); i+=blockDim.x) {
        val = __half22float2(inp_ptr[i]);
        scale = __half22float2(gamma_ptr[i]);
        val.x *= (mean * scale.x);
        val.y *= (mean * scale.y);
        out_ptr[i] = __float22half2_rn(val);
    }
}

template <typename DataType>
void launchResNormKernel(DataType* output, const DataType* input, const DataType* gamma, const float eps, 
    const int m, const int n, cudaStream_t stream = 0)
{
    dim3 grid(m);
    dim3 block(block_size);
    resNormKernel<DataType><<<grid, block, 0, stream>>>(output, input, gamma, eps, n);
}


#define WARP_SIZE 32
//kernel code
static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float * gamma, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp
    //一个线程求和(ncols/WARP_SIZE)个数据的x^2 
    for (int col = tid; col < ncols; col += WARP_SIZE) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }
    
    // sum up partial sums
    // 一个线程束(32个线程)内的归约求和
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }
    
    const float mean = tmp / ncols; // mean(x^2)
    const float scale = rsqrtf(mean + eps); // 1/根号mean
    //算完之后写回原数组
    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[row*ncols + col] = scale * x[row*ncols + col] * gamma[col];
    }
}

//call kernel
static void rms_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float * gamma, const float eps, cudaStream_t stream = 0) {
    assert(ncols % WARP_SIZE == 0);
    const dim3 block_dims(WARP_SIZE, 1, 1); //(32,1,1)
    //所以调用的cuda的gridDim =(nrows,1,1) ,blockDim = (32,1,1)
    //也就是说一个block处理一个row的数据，即每32个线程处理一行数据 ，共计nrows行
    rms_norm_f32<<<nrows, block_dims, 0, stream>>>(x, dst, ncols, gamma, eps);
}

/**
 * grid(seq_len)  block(block_size) for size_per_head/2 >= block_size(128)
*/
__global__ void precomputeFreqsCis(float *freq_cis, const int size_per_head)
{
    int offset = blockIdx.x * size_per_head;
    for (int i=threadIdx.x; i<(size_per_head >> 1); i+=blockDim.x) {
        float val = i * (-2.0f) / size_per_head;
        float theta = __powf(1e4f, val)  * blockIdx.x;
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
    if (row < seq_len) {
        for (int i=threadIdx.x; i<(size_per_head >> 1); i+=blockDim.x) {
            float val = i * (-2.0f) / size_per_head;
            float theta = __powf(1e4f, val)  * row;
            freq_cis[offset + 2 * i] = __cosf(theta);
            freq_cis[offset + 2 * i + 1] = __sinf(theta);
        }
    }
}

void launchPrecomputeFreqsCis(float *freq_cis, const int size_per_head, const int seq_len, cudaStream_t stream = 0)
{
    if ((size_per_head / 2) < 128) {
        int warp_num = block_size / 32;
        int grid_size = (seq_len + warp_num - 1) / warp_num;
        dim3 grid(grid_size);
        dim3 block(32, warp_num);
        warpPrecomputeFreqsCis<<<grid, block, 0, stream>>>(freq_cis, size_per_head, seq_len);
    }
    else {
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
__global__ void embeddingLookingUpKernel(DataType * __restrict__ from_tensor, const DataType * __restrict__ embedding_table,
    const int * __restrict__ word_ids, const int hidden_units, const int seq_len)
{
    const int batch_id = blockIdx.x;
    const int seq_id = blockIdx.y;
    const int offset = batch_id * seq_len * hidden_units + seq_id * hidden_units;
    const int id_for_word = word_ids[batch_id * seq_len + seq_id];
    for (int i=threadIdx.x; i<hidden_units; i+=blockDim.x) {
        from_tensor[offset + i] = embedding_table[id_for_word * hidden_units + i];
    }
}

template <typename DataType>
void launchEmbeddingLookingUpKernel(DataType * from_tensor, const DataType * embedding_table,
    const int * word_ids, const int hidden_units, const int batch_size, const int seq_len, cudaStream_t stream = 0)
{
    dim3 grid(batch_size, seq_len);
    dim3 block(block_size);
    embeddingLookingUpKernel<DataType><<<grid, block, 0, stream>>>(from_tensor, embedding_table, word_ids, hidden_units, seq_len);
}


    
}   // tinycudallama