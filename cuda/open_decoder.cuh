#include "cuda_kernels.cuh"

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

        T attn_mask;

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
            cache_size = batch_size, head_num, total_len_, size_per_head
         */
        float *freq_cis_;            // [max_prompt_len_, size_per_head]
        int8_t *attn_norm_out_buf_;  // buf_size, [batch_size * seq_len, head_num * size_per_head]
        float *attn_norm_scale_buf_; // size: batch_size * max_prompt_len_,  [batch_size, seq_len]
        int32_t *query_buf_;         // buf_size, [batch_size * seq_len, head_num * size_per_head]
        int32_t *key_buf_;           // buf_size, [batch_size * seq_len, head_num * size_per_head]
        int32_t *value_buf_;         // buf_size, [batch_size * seq_len, head_num * size_per_head]
        float *query_out_buf_;       // buf_size, [batch_size, head_num, seq_len, size_per_head]
        float *key_out_buf_;         // buf_size, [batch_size, head_num, start_pos+seq_len, size_per_head]
        float *value_out_fp_buf_;    // buf_size, [batch_size, head_num, seq_len, size_per_head]
        int8_t *value_out_buf;       // buf_size, [batch_size, head_num, start_pos+seq_len, size_per_head]
        float *query_scale_buf_;     // [batch_size, head_num, max_prompt_len_]
        // float *key_scale_buf_;            // [batch_size, head_num, max_prompt_len_]
        float *value_scale_buf;     // [batch_size, head_num, total_len_, size_per_head]
        float *qk_buf_;             // [batch_size * head_num, seq_len, total_len_]
        float *qkv_buf_;            // [batch_size * head_num, seq_len, size_per_head]
        int8_t *qkv_trans_buf_;     // [batch_size, seq_len, head_num * size_per_head]
        float *attn_out_scale_buf_; // [batch_size, seq_len]
        int32_t *attn_out_buf_;     // [batch_size, seq_len, head_num * size_per_head]
        DataType_ *ffn_tensor_buf_; //[batch_size, seq_len, head_num * size_per_head]
        int32_t *w1_buf_;
        int32_t *w3_buf;
        int32_t *w2_buf_;
        int32_t *ffn_inter_buf_;
        float *ffn_inter_scale; // [batch_size, seq_len]

        // DataType_ *norm_from_tensor_buf_, *query_buf_, *context_buf_, *masked_output_buf_;
        // DataType_ *norm_masked_output_buf_, *cross_output_buf_, *norm_cross_output_buf_, *ffn_inner_buf_;
        // DataType_ *key_buf_, *value_buf_;

    public:
        OpenDecoder(int batch_size, int max_seq_len,
                    int head_num, int size_per_head,
                    int memory_hidden_units) : batch_size_(batch_size),
                                               max_seq_len_(max_seq_len), head_num_(head_num),
                                               size_per_head_(size_per_head),
                                               memory_hidden_units_(memory_hidden_units)
        {
            hidden_units_ = head_num_ * size_per_head_;
            total_len_ = max_prompt_len_ + max_gen_len_;
            for (int i = 0; i < 5; i++)
            {
                cublasAlgo_[i] = -1; // CUBLAS_GEMM_DEFAULT
            }
        }

        void initialize(DecoderInitParam<DataType_> param, DataType_ *buf)
        {
            param_ = param;
            // int buf_size = batch_size_ * hidden_units_;
            // norm_from_tensor_buf_ = buf;
            // query_buf_ = buf + buf_size; // store the query values (from_tensor * Q) in both masked and multi-head attention
            // key_buf_ = buf + 2 * buf_size;
            // value_buf_ = buf + 3 * buf_size;
            // context_buf_ = buf + 4 * buf_size; // store the context result (softmax(qk)v) in both masked and multi-head attention

            // masked_output_buf_ = buf + 5 * buf_size;      // masked_attention_output
            // norm_masked_output_buf_ = buf + 6 * buf_size; // norm(masked_attention_output)

            // cross_output_buf_ = buf + 7 * buf_size;      // mutli-head attention_output
            // norm_cross_output_buf_ = buf + 8 * buf_size; // norm(multi-head attention_output)
            // ffn_inner_buf_ = buf + 9 * buf_size;         // 4 buf size to store inner product

            launchPrecomputeFreqsCis(freq_cis_, size_per_head_, max_seq_len_, param_.stream);
        }
        /**
         * key_cache_ value_cache_: [batch_size, head_num, total_len_, size_per_head]
         * key_scale_cache_: [batch_size, head_num, total_len_]
         */
        void forward(const DataType_ *from_tensor, float *key_cache_, float *value_cache_,
                     float *key_scale_cache_, float *value_scale_cache_, int ffn_hidden_units,
                     DataType_ *decoder_output, const int start_pos, const int seq_len)
        {
            const AlphaBetaType_ alpha = 1;
            const AlphaBetaType_ beta = 0;
            try
            {
                /* masked multi-head attention */
                /* ResNorm-Quantized(from_tensor) -> attn_norm_out_buf_ and attn_norm_scale_buf_ */
                launchResNormQuantizedKernel<DataType_>(attn_norm_out_buf_, from_tensor, param_.attn_resnorm.gamma, attn_norm_scale_buf_,
                                                        param_.attn_resnorm.eps, batch_size_ * seq_len, hidden_units_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
#endif

                /* Q\K\V gemm(attn_norm_out_buf_) -> query_buf_、key_buf_、value_buf_ */
                int m = batch_size_ * seq_len;
                int n = hidden_units_;
                int k = hidden_units_;

                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.attention.query_weight.kernel, AType_, n,
                                                 attn_norm_out_buf_, BType_, k,
                                                 &beta,
                                                 query_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.attention.key_weight.kernel, AType_, n,
                                                 attn_norm_out_buf_, BType_, k,
                                                 &beta,
                                                 key_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.attention.value_weight.kernel, AType_, n,
                                                 attn_norm_out_buf_, BType_, k,
                                                 &beta,
                                                 value_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                /**
                 * Q\K Quantized-rope-Quantized-Transpose
                 * query_buf_, key_buf_ -> query_out_buf_, key_out_buf_, query_scale_buf_, key_scale_cache_
                 */
                launchQKRoteEmbeddingTranspose(query_out_buf_, key_out_buf_, query_buf_, key_buf_, attn_norm_scale_buf_, attn_norm_scale_buf_,
                                               param_.attention.query_weight.weight_scale, param_.attention.key_weight.weight_scale, query_scale_buf_, key_scale_cache_, freq_cis_, batch_size_, seq_len,
                                               start_pos, total_len_, head_num_, size_per_head_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
#endif

                /**
                 * Dequantized V Transpose
                 * value_buf_ -> value_out_fp_buf_
                 */
                launchDequantizedVTransposeKernel(value_out_fp_buf_, value_buf_, attn_norm_scale_buf_, param_.attention.value_weight.weight_scale,
                                                  batch_size_, seq_len, head_num_, size_per_head_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
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
                check_cuda_error(cudaGetLastError());
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
                check_cuda_error(cudaGetLastError());
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
                launchAttnQuantizedTransposeKernel(qkv_trans_buf_, qkv_buf_, attn_out_scale_buf_, batch_size_, head_num_, seq_len,
                                                   size_per_head_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
#endif

                /**
                 * project gemm
                 */
                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.attention.attention_output_weight.kernel, AType_, n,
                                                 qkv_trans_buf_, BType_, k,
                                                 &beta,
                                                 attn_out_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                /**
                 * attn_out_buf_ -> dequantized & add Residual -> ffn_tensor_buf_, DataType_, [batch_size, seq_len, hidden_units]
                 * ffn_tensor_buf_ -> resNorm & quantized -> attn_norm_out_buf_, int8, [batch_size, seq_len, hidden_units]
                 */
                launchDequantizedResidualResNormQuantized<DataType_>(attn_norm_out_buf_, ffn_tensor_buf_, from_tensor, attn_out_buf_,
                                                                     attn_out_scale_buf_, param_.attention.attention_output_weight.weight_scale,
                                                                     param_.ffn_resnorm.gamma, attn_norm_scale_buf_, param_.ffn_resnorm.eps,
                                                                     batch_size_ * seq_len, hidden_units_, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
#endif

                n = ffn_hidden_units;
                /**
                 * w1 gemm
                 */
                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.ffn.w1_weight.kernel, AType_, n,
                                                 attn_norm_out_buf_, BType_, k,
                                                 &beta,
                                                 w1_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                /**
                 * w3 gemm
                 */
                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.ffn.w3_weight.kernel, AType_, n,
                                                 attn_norm_out_buf_, BType_, k,
                                                 &beta,
                                                 w3_buf, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                /**
                 * dequantized w1_buf_ to w1_out
                 * dequantized w3_buf_ & silu to w3_out
                 * pointwise-multiply (w1_out, w3_out) to w13_out
                 * quantized w13_out to attn_norm_out_buf_, ffn_out_scale
                 */
                launchDequantizedSiluMultifyQuantized(ffn_inter_buf_, w1_buf_, attn_norm_scale_buf_, param_.ffn.w1_weight.weight_scale,
                                                      w3_buf, param_.ffn.w3_weight.weight_scale, ffn_inter_scale,
                                                      batch_size_ * seq_len, ffn_hidden_units, param_.stream);

#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
#endif

                k = ffn_hidden_units;
                n = hidden_units_;
                /**
                 * w2 gemm
                 */
                CHECK_CUBLAS_STATUS(cublasGemmEx(param_.cublas_handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k,
                                                 &alpha,
                                                 param_.ffn.w2_weight.kernel, AType_, n,
                                                 attn_norm_out_buf_, BType_, k,
                                                 &beta,
                                                 w2_buf_, CType_, n,
                                                 computeType_,
                                                 static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

                launchDequantizedResidual<DataType_>(decoder_output, ffn_tensor_buf_, w2_buf_, ffn_out_scale,
                                                     param_.ffn.w2_weight.weight_scale, batch_size_ * seq_len, hidden_units_, param_.stream);

                
#ifndef NDEBUG
                cudaDeviceSynchronize();
                check_cuda_error(cudaGetLastError());
#endif


            }

            catch (std::runtime_error &error)
            {
                throw error;
            }
        }
        void masked_multi_head_attention(const DataType_ *from_tensor, DataType_ *key_cache_,
                                         DataType_ *value_cache_, DataType_ *decoder_output, const int step);

        void cross_multi_head_attention(const DataType_ *from_tensor, const DataType_ *memory_tensor,
                                        DataType_ *key_mem_cache_, DataType_ *value_mem_cache_,
                                        DataType_ *decoder_output, const int *memory_sequence_length,
                                        const int max_seq_len, const int step);

        void ffn(const DataType_ *input, DataType_ *ffn_inner, DataType_ *output,
                 const int m, const int inner_size, const int n);

        void decoder_norm1(const DataType_ *from_tensor, const DataType_ *gamma,
                           const DataType_ *beta, DataType_ *norm_from_tensor_buf_, const int m, const int n);

        void decoder_norm2(const DataType_ *from_tensor, const DataType_ *gamma,
                           const DataType_ *beta, const DataType_ *bias,
                           DataType_ *output, DataType_ *norm_output_buf_,
                           const int m, const int n);

        void add_bias_input(DataType_ *output, const DataType_ *input, const int m, const int n);

        ~OpenDecoder()
        {
            norm_from_tensor_buf_ = nullptr;
            query_buf_ = nullptr;
            key_buf_ = nullptr;
            value_buf_ = nullptr;
            context_buf_ = nullptr;

            masked_output_buf_ = nullptr;
            norm_masked_output_buf_ = nullptr;

            cross_output_buf_ = nullptr;
            norm_cross_output_buf_ = nullptr;
            ffn_inner_buf_ = nullptr;
        }
    };

} // namespace tinycudallama
