#pragma once
#include "common.h"
#include <cub/cub.cuh>
#include <assert.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

namespace tinycudallama
{

    static inline __device__ int8_t float_to_int8_rn(float x)
    {
        uint32_t dst;
        asm volatile("cvt.rni.sat.s8.f32 %0, %1;"
                     : "=r"(dst)
                     : "f"(x));
        return reinterpret_cast<const int8_t &>(dst);
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
    __inline__ __device__ T warpAllReduce(T val);

    template <typename T>
    __inline__ __device__ T blockAllReduceSum(T val);

    template <typename T>
    __inline__ __device__ T blockAllReduceMax(T val);

} // tinycudallama