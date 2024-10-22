#pragma once

#include "decoder_kernels.cuh"

namespace tinycudallama
{
    template <OperationType OpType_>
    class TransformerTraits;

    template <>
    class TransformerTraits<OperationType::FP32>
    {
    public:
        typedef float DataType;
        static const OperationType OpType = OperationType::FP32;
        static cublasComputeType_t const computeType = CUBLAS_COMPUTE_32I;
        typedef int32_t AlphaBetaType;
        static cudaDataType_t const AType = CUDA_R_8I;
        static cudaDataType_t const BType = CUDA_R_8I;
        static cudaDataType_t const CType = CUDA_R_32I;
    };

    template <>
    class TransformerTraits<OperationType::FP16>
    {
    public:
        typedef half DataType;
        static const OperationType OpType = OperationType::FP16;
        static cublasComputeType_t const computeType = CUBLAS_COMPUTE_32I;
        typedef int32_t AlphaBetaType;
        static cudaDataType_t const AType = CUDA_R_8I;
        static cudaDataType_t const BType = CUDA_R_8I;
        static cudaDataType_t const CType = CUDA_R_32I;
    };

    template <typename T>
    class DecoderInitParam
    {
    public:
        /* weights for transformer */
        ResNormWeight<T> attn_resnorm;
        AttentionWeight<T> attention;

        T *attn_mask;

        ResNormWeight<T> ffn_resnorm;
        FFNWeight<T> ffn;

        cublasHandle_t cublas_handle;
        cudaStream_t stream;
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

    template <OperationType OpType_>
    class OpenDecoder
    {
    private:
        typedef DecoderTransformerTraits<OpType_> Traits_;
        typedef typename Traits_::DataType DataType_;
        DecoderInitParam<DataType_> param_;

        typedef typename Traits_::AlphaBetaType AlphaBetaType_;
        const cublasComputeType_t computeType_ = Traits_::computeType;
        const cudaDataType_t AType_ = Traits_::AType;
        const cudaDataType_t BType_ = Traits_::BType;
        const cudaDataType_t CType_ = Traits_::CType;
        int cublasAlgo_[5];

        int batch_size_;
        int max_prompt_len_;
        int max_gen_len_;
        int total_len_;
        int head_num_;
        int size_per_head_;
        int hidden_units_;

        /*  buf_size = batch_size * max_prompt_len_ * head_num * size_per_head
            cache_size = batch_size * head_num * total_len_ * size_per_head
         */
        int8_t *from_tensor_int8_buf_; // buf_size, [batch_size * seq_len, head_num * size_per_head]
        float *from_tensor_scale_buf_; // batch_size * max_prompt_len_, [batch_size, seq_len]
        int32_t *query_buf_;           // buf_size, [batch_size * seq_len, head_num * size_per_head]
        int32_t *key_buf_;             // buf_size, [batch_size * seq_len, head_num * size_per_head]
        int32_t *value_buf_;           // buf_size, [batch_size * seq_len, head_num * size_per_head]
        float *query_out_buf_;         // buf_size, [batch_size, head_num, seq_len, size_per_head]
        float *key_out_buf_;           // buf_size, [batch_size, head_num, seq_len, size_per_head]
        float *value_out_fp_buf_;      // buf_size, [batch_size, head_num, seq_len, size_per_head]
        float *qk_buf_;                // [batch_size * head_num, seq_len, total_len_]
        float *qkv_buf_;               // buf_size, [batch_size * head_num, seq_len, size_per_head]
        DataType_ *ffn_tensor_buf_;    // buf_size, [batch_size, seq_len, head_num * size_per_head]
        float *ffn_inter_scale_buf_;   // batch_size * max_prompt_len_, [batch_size, seq_len]

    public:
        OpenDecoder(int batch_size, int max_prompt_len, int max_gen_len,
                    int head_num, int size_per_head) : batch_size_(batch_size), max_prompt_len_(max_prompt_len),
                                                       max_gen_len_(max_gen_len), head_num_(head_num),
                                                       size_per_head_(size_per_head)
        {
#ifndef NDEBUG
            PRINT_FUNC_NAME_();
#endif
            hidden_units_ = head_num_ * size_per_head_;
            total_len_ = max_prompt_len_ + max_gen_len_;
            for (int i = 0; i < 5; i++)
            {
                cublasAlgo_[i] = -1; // CUBLAS_GEMM_DEFAULT
            }
        }

        void initialize(DecoderInitParam<DataType_> param, char *buf)
        {
#ifndef NDEBUG
            PRINT_FUNC_NAME_();
#endif
            param_ = param;
            int buf_size = batch_size_ * max_prompt_len_ * head_num_ * size_per_head_;
            from_tensor_int8_buf_ = (int8_t *)(buf);
            from_tensor_scale_buf_ = (float *)(from_tensor_int8_buf_ + buf_size);
            query_buf_ = (int32_t *)(from_tensor_scale_buf_ + batch_size_ * max_prompt_len_);
            key_buf_ = (int32_t *)(query_buf_ + buf_size);
            value_buf_ = (int32_t *)(key_buf_ + buf_size);
            query_out_buf_ = (float *)(value_buf_ + buf_size);
            key_out_buf_ = (float *)(query_out_buf_ + buf_size);
            value_out_fp_buf_ = (float *)(key_out_buf_ + buf_size);
            qk_buf_ = (float *)(value_out_fp_buf_ + buf_size);
            qkv_buf_ = (float *)(qk_buf_ + batch_size_ * head_num_ * max_prompt_len_ * total_len_);
            ffn_tensor_buf_ = (DataType_ *)(qkv_buf_ + buf_size);
            ffn_inter_scale_buf_ = (float *)(ffn_tensor_buf_ + buf_size);

#ifndef NDEBUG
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(cudaGetLastError());
#endif
        }

        int getWorkspaceSize()
        {
            int buf_size = batch_size_ * max_prompt_len_ * hidden_units_;
            int work_space_size = sizeof(int8_t) * buf_size + sizeof(float) * 4 * buf_size + sizeof(int32_t) * 3 * buf_size +
                                  sizeof(float) * 2 * batch_size_ * max_prompt_len_ + sizeof(DataType_) * buf_size +
                                  sizeof(float) * batch_size_ * head_num_ * max_prompt_len_ * total_len_;
            return work_space_size;
        }

        /**
         * key_cache_ value_cache_: cache_size, [batch_size, head_num, total_len_, size_per_head]
         * freq_cis_: [max_prompt_len_, size_per_head]
         */
        void forward(const DataType_ *from_tensor, const float *freq_cis, float *key_cache_, float *value_cache_, int ffn_hidden_units,
                     DataType_ *decoder_output, const int start_pos, const int seq_len)
        {
#ifndef NDEBUG
            PRINT_FUNC_NAME_();
#endif
            const AlphaBetaType_ alpha = 1;
            const AlphaBetaType_ beta = 0;
            try
            {
                /* masked multi-head attention */
                /* ResNorm-Quantized(from_tensor) -> from_tensor_int8_buf_ and from_tensor_scale_buf_ */
                launchResNormQuantizedKernel<DataType_>(from_tensor_int8_buf_, from_tensor, param_.attn_resnorm.gamma, from_tensor_scale_buf_,
                                                        param_.attn_resnorm.eps, batch_size_ * seq_len, hidden_units_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());
#endif

                /* Q\K\V gemm(from_tensor_int8_buf_) -> query_buf_、key_buf_、value_buf_ */
                int m = batch_size_ * seq_len;
                int n = hidden_units_;
                int k = hidden_units_;

                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.attention.query_weight.kernel, AType_, n,
                                                 from_tensor_int8_buf_, BType_, k,
                                                 &beta,
                                                 query_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.attention.key_weight.kernel, AType_, n,
                                                 from_tensor_int8_buf_, BType_, k,
                                                 &beta,
                                                 key_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.attention.value_weight.kernel, AType_, n,
                                                 from_tensor_int8_buf_, BType_, k,
                                                 &beta,
                                                 value_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                /**
                 * Q\K Quantized-rope-Quantized-Transpose
                 * query_buf_, key_buf_ -> query_out_buf_, key_out_buf_
                 */
                launchQKRoteEmbeddingTranspose(query_out_buf_, key_out_buf_, query_buf_, key_buf_, from_tensor_scale_buf_, from_tensor_scale_buf_,
                                               param_.attention.query_weight.weight_scale, param_.attention.key_weight.weight_scale,
                                               freq_cis, batch_size_, seq_len, start_pos, total_len_, head_num_, size_per_head_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());
#endif

                /**
                 * Dequantized V Transpose
                 * value_buf_ -> value_out_fp_buf_
                 */
                launchDequantizedVTransposeKernel(value_out_fp_buf_, value_buf_, from_tensor_scale_buf_, param_.attention.value_weight.weight_scale,
                                                  batch_size_, seq_len, head_num_, size_per_head_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());
#endif

                /**
                 * Store K\V in cache
                 * k_cache v_cache: [batch_size, head_num, total_len_, size_per_head]
                 * store k\v [batch_size, head_num, seq_len, size_per_head] to [batch_size, head_num, start_pos:start_pos+seq_len, size_per_head]
                 */
                launchStoreKVcacheKernel(key_cache_, value_cache_, key_out_buf_, value_out_fp_buf_, start_pos, seq_len, batch_size_,
                                         head_num_, total_len_, size_per_head_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());
#endif

                // prompt 阶段，此时 qk 乘法为 gemm
                if (seq_len > 1)
                {
                    CHECK_CUBLAS_STATUS(cublasGemmStridedBatchedEx(param_.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                                   seq_len, seq_len, size_per_head_,
                                                                   &alpha,
                                                                   key_out_buf_, CUDA_R_32F, size_per_head_, seq_len * size_per_head_,
                                                                   query_out_buf_, CUDA_R_32F, size_per_head_, seq_len * size_per_head_,
                                                                   &beta,
                                                                   qk_buf_, CUDA_R_32F, seq_len, seq_len * seq_len,
                                                                   batch_size_ * head_num_,
                                                                   CUBLAS_COMPUTE_32F,
                                                                   cublasAlgo_[1]));
                }
                else
                { // generation 阶段，此时 qk 乘法为 gemv
                    CHECK_CUBLAS_STATUS(cublasSgemvStridedBatched(param_.cublas_handle, CUBLAS_OP_T,
                                                                  seq_len + start_pos, size_per_head_,
                                                                  &alpha,
                                                                  key_cache_, size_per_head_, total_len_ * size_per_head_,
                                                                  query_out_buf_, 1, size_per_head_,
                                                                  &beta,
                                                                  qk_buf_, 1, size_per_head_,
                                                                  batch_size_ * head_num_));
                }

                /**
                 * softmax
                 */
                launchBlockSoftmaxKernel(qk_buf_, param_.attn_mask, batch_size_, head_num_, seq_len,
                                         seq_len + start_pos, total_len_, rsqrtf(static_cast<float>(size_per_head_)), param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());
#endif

                // prompt 阶段，此时 qk*v 乘法为 gemm
                if (seq_len > 1)
                {
                    CHECK_CUBLAS_STATUS(cublasGemmStridedBatchedEx(param_.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                                   size_per_head_, seq_len, seq_len,
                                                                   &alpha,
                                                                   value_out_fp_buf_, CUDA_R_32F, size_per_head_, seq_len * size_per_head_,
                                                                   qk_buf_, CUDA_R_32F, seq_len, seq_len * seq_len,
                                                                   &beta,
                                                                   qkv_buf_, CUDA_R_32F, size_per_head_, seq_len * size_per_head_,
                                                                   batch_size_ * head_num_,
                                                                   CUBLAS_COMPUTE_32F,
                                                                   cublasAlgo_[1]));
                }
                else
                { // generation 阶段，此时 qk*v 乘法为 gemv
                    CHECK_CUBLAS_STATUS(cublasSgemvStridedBatched(param_.cublas_handle, CUBLAS_OP_N,
                                                                  size_per_head_, seq_len + start_pos,
                                                                  &alpha,
                                                                  value_cache_, size_per_head_, total_len_ * size_per_head_,
                                                                  qk_buf_, 1, size_per_head_,
                                                                  &beta,
                                                                  qkv_buf_, 1, size_per_head_,
                                                                  batch_size_ * head_num_));
                }

                /**
                 * quantized qkv to int8 and transpose from [batch_size, head_num, seq_len, size_per_head]
                 * to [batch_size, seq_len, hidden_units]
                 */
                launchAttnQuantizedTransposeKernel(from_tensor_int8_buf_, qkv_buf_, from_tensor_scale_buf_, batch_size_, head_num_, seq_len,
                                                   size_per_head_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());
#endif

                /**
                 * project gemm, Reuse the query_buf_ as attn_out_buf_
                 */
                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.attention.attention_output_weight.kernel, AType_, n,
                                                 from_tensor_int8_buf_, BType_, k,
                                                 &beta,
                                                 query_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                /**
                 * query_buf_ -> dequantized & add Residual -> ffn_tensor_buf_, DataType_, [batch_size, seq_len, hidden_units]
                 * ffn_tensor_buf_ -> resNorm & quantized -> from_tensor_int8_buf_, int8, [batch_size, seq_len, hidden_units]
                 */
                launchDequantizedResidualResNormQuantized<DataType_>(from_tensor_int8_buf_, ffn_tensor_buf_, from_tensor, query_buf_,
                                                                     from_tensor_scale_buf_, param_.attention.attention_output_weight.weight_scale,
                                                                     param_.ffn_resnorm.gamma, from_tensor_scale_buf_, param_.ffn_resnorm.eps,
                                                                     batch_size_ * seq_len, hidden_units_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());
#endif

                n = ffn_hidden_units;
                /**
                 * w1 gemm, Reuse the query_buf_ as w1_buf_
                 */
                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.ffn.w1_weight.kernel, AType_, n,
                                                 from_tensor_int8_buf_, BType_, k,
                                                 &beta,
                                                 query_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                /**
                 * w3 gemm, Reuse the key_buf_ as w3_buf_
                 */
                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.ffn.w3_weight.kernel, AType_, n,
                                                 from_tensor_int8_buf_, BType_, k,
                                                 &beta,
                                                 key_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                /**
                 * dequantized query_buf_ to w1_out
                 * dequantized key_buf__ & silu to w3_out
                 * pointwise-multiply (w1_out, w3_out) to w13_out
                 * quantized w13_out to from_tensor_int8_buf_, ffn_inter_scale_buf_
                 */
                launchDequantizedSiluMultifyQuantized(from_tensor_int8_buf_, query_buf_, from_tensor_scale_buf_, param_.ffn.w1_weight.weight_scale,
                                                      key_buf_, param_.ffn.w3_weight.weight_scale, ffn_inter_scale_buf_,
                                                      batch_size_ * seq_len, ffn_hidden_units, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());
#endif

                k = ffn_hidden_units;
                n = hidden_units_;
                /**
                 * w2 gemm, Reuse the value_buf_ as w2_buf_
                 */
                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.ffn.w2_weight.kernel, AType_, n,
                                                 from_tensor_int8_buf_, BType_, k,
                                                 &beta,
                                                 value_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                /**
                 * dequantized value_buf_ to w2_out used ffn_inter_scale_buf_ and weight_scale
                 * add Residual: w2_out + ffn_tensor_buf_ -> decoder_output
                 */
                launchDequantizedResidual<DataType_>(decoder_output, ffn_tensor_buf_, value_buf_, ffn_inter_scale_buf_,
                                                     param_.ffn.w2_weight.weight_scale, batch_size_ * seq_len, hidden_units_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());
#endif
            }

            catch (std::runtime_error &error)
            {
                throw error;
            }
        }

        ~OpenDecoder()
        {
            from_tensor_int8_buf_ = nullptr;
            from_tensor_scale_buf_ = nullptr;
            query_buf_ = nullptr;
            key_buf_ = nullptr;
            value_buf_ = nullptr;
            query_out_buf_ = nullptr;
            key_out_buf_ = nullptr;
            value_out_fp_buf_ = nullptr;
            qk_buf_ = nullptr;
            qkv_buf_ = nullptr;
            ffn_tensor_buf_ = nullptr;
            ffn_inter_scale_buf_ = nullptr;
        }
    };

} // namespace tinycudallama
