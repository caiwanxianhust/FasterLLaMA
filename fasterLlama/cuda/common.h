#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace FasterLLaMA
{

    enum class OperationType
    {
        FP32,
        FP16,
        INT8
    };
    enum class AllocatorType
    {
        CUDA,
        TF,
        TH
    };

    template <typename T>
    struct ResNormWeight
    {
        T *gamma = nullptr;
        float eps = 1e-5f;
    };

    template <typename T, typename WeightType>
    struct DenseWeight
    {
        WeightType *kernel = nullptr;
        T *bias = nullptr;
        float *weight_scale = nullptr;
    };

    template <typename T, typename WeightType>
    struct AttentionWeight
    {
        DenseWeight<T, WeightType> query_weight;
        DenseWeight<T, WeightType> key_weight;
        DenseWeight<T, WeightType> value_weight;
        DenseWeight<T, WeightType> attention_output_weight;
    };

    template <typename T, typename WeightType>
    struct FFNWeight
    {
        DenseWeight<T, WeightType> w1_weight;
        DenseWeight<T, WeightType> w2_weight;
        DenseWeight<T, WeightType> w3_weight;
    };

    template <OperationType OpType_>
    class TransformerTraits;

    template <>
    class TransformerTraits<OperationType::INT8>
    {
    public:
        typedef int8_t DataType;
        typedef int32_t AlphaType;
        static const OperationType OpType = OperationType::INT8;
        static cublasComputeType_t const computeType = CUBLAS_COMPUTE_32I;
        static cudaDataType_t const AType = CUDA_R_8I;
        static cudaDataType_t const BType = CUDA_R_8I;
        static cudaDataType_t const CType = CUDA_R_32I;
    };

    template <>
    class TransformerTraits<OperationType::FP32>
    {
    public:
        typedef float DataType;
        typedef float AlphaType;
        static const OperationType OpType = OperationType::FP32;
        static cublasComputeType_t const computeType = CUBLAS_COMPUTE_32F_FAST_16F;
        static cudaDataType_t const AType = CUDA_R_32F;
        static cudaDataType_t const BType = CUDA_R_32F;
        static cudaDataType_t const CType = CUDA_R_32F;
    };

    template <>
    class TransformerTraits<OperationType::FP16>
    {
    public:
        typedef half DataType;
        typedef half AlphaType;
        static const OperationType OpType = OperationType::FP16;
        static cublasComputeType_t const computeType = CUBLAS_COMPUTE_16F;
        static cudaDataType_t const AType = CUDA_R_16F;
        static cudaDataType_t const BType = CUDA_R_16F;
        static cudaDataType_t const CType = CUDA_R_16F;
    };

    template <OperationType OpType_>
    class DecoderTransformerTraits;

    template <>
    class DecoderTransformerTraits<OperationType::FP32> : public TransformerTraits<OperationType::FP32>
    {
    };

    template <>
    class DecoderTransformerTraits<OperationType::FP16> : public TransformerTraits<OperationType::FP16>
    {
    };

    template <>
    class DecoderTransformerTraits<OperationType::INT8> : public TransformerTraits<OperationType::INT8>
    {
    };

}
