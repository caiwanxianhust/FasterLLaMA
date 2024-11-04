#pragma once
#include "common.h"
#include <cub/cub.cuh>
#include <assert.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cfloat>

namespace FasterLLaMA
{

    static inline __device__ int8_t float_to_int8_rn(float x);
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
    __inline__ __device__ T warpAllReduce(T val);

    template <typename T>
    __inline__ __device__ T blockAllReduceSum(T val);

    template <typename T>
    __inline__ __device__ T blockAllReduceMax(T val);

    static inline __device__ int8_t float_to_int8_rn(float x)
    {
        uint32_t dst;
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;"
                     : "=r"(dst)
                     : "f"(x));
        return reinterpret_cast<const int8_t &>(dst);
    }

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

        val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(-1 * FLT_MAX);
        val = warpAllReduce<MaxOp, T>(val);
        if (threadIdx.x == 0)
            result = val;
        __syncthreads();
        return result;
    }

} // FasterLLaMA