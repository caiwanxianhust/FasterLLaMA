#pragma once
#include "error.cuh"
#include <cub/cub.cuh>
#include <assert.h>

namespace tinycudallama {

namespace {
    constexpr int block_size = 256;
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
    const DataType* __restrict__ gamma, const DataType eps, const int hidden_uints)
{
    const DataType *inp = input + blockIdx.x * hidden_uints;
    float mean;
    float val = 0.0f;
    for (int i=threadIdx.x; i<hidden_uints; i+=blockDim.x) {
        val += inp[i] * inp[i];
    }
    __syncthreads();

    val = blockAllReduce<DataType, SumOp>(val);
    mean = rsqrtf(val / hidden_uints + eps);
    __syncthreads();

    for (int i=threadIdx.x; i<hidden_uints; i+=blockDim.x) {
        output[blockIdx.x * hidden_uints + i] = mean * inp[i] * gamma[i];
    }
}

template <typename DataType>
void launchResNormKernel(DataType* output, const DataType* input, 
    const DataType* gamma, const DataType eps, const int m, const int n)
{
    dim3 grid(m);
    dim3 block(block_size);
    resNormKernel<DataType><<<grid, block>>>(output, input, gamma, eps, n);
}

#define WARP_SIZE 32
//call kernel
static void rms_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float * gamma, const float eps, cudaStream_t stream = 0) {
    assert(ncols % WARP_SIZE == 0);
    const dim3 block_dims(WARP_SIZE, 1, 1); //(32,1,1)
    //所以调用的cuda的gridDim =(nrows,1,1) ,blockDim = (32,1,1)
    //也就是说一个block处理一个row的数据，即每32个线程处理一行数据 ，共计nrows行
    rms_norm_f32<<<nrows, block_dims, 0, stream>>>(x, dst, ncols, gamma, eps);
}

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

    
}   // tinycudallama