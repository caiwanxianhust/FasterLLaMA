#pragma once

#include "common.h"

namespace FasterLLaMA
{
    
    template <typename T, typename WeightType>
    class DecoderInitParam
    {
    public:
        /* weights for transformer */
        ResNormWeight<T> attn_resnorm;
        AttentionWeight<T, WeightType> attention;

        float *attn_mask;

        ResNormWeight<T> ffn_resnorm;
        FFNWeight<T, WeightType> ffn;

        cublasHandle_t cublas_handle;
        cudaStream_t stream;
    };

    
    template <OperationType OpType_, OperationType QuantizationType>
    class OpenDecoder
    {
    private:
        typedef DecoderTransformerTraits<OpType_> Traits_;
        typedef DecoderTransformerTraits<OperationType::FP32> qkv_Traits_;
        typedef DecoderTransformerTraits<QuantizationType> weight_Traits_;

        typedef typename Traits_::DataType DataType_;
        typedef typename weight_Traits_::DataType weight_DataType_;
        DecoderInitParam<DataType_, weight_DataType_> param_;

        int cublasAlgo_[5];

        int batch_size_;
        int max_prompt_len_;
        int max_gen_len_;
        int total_len_;
        int head_num_;
        int size_per_head_;
        int hidden_units_;
        int ffn_hidden_units_;

        /*  buf_size = batch_size * max_prompt_len_ * head_num * size_per_head
            cache_size = batch_size * head_num * total_len_ * size_per_head
         */
        int8_t *from_tensor_int8_buf_; // buf_size, [batch_size * seq_len, head_num * size_per_head]
        float *from_tensor_scale_buf_; // batch_size * max_prompt_len_, [batch_size, seq_len]
        int32_t *query_buf_;           // [batch_size * seq_len, max(hidden_units, ffn_hidden_units)]
        int32_t *key_buf_;             // [batch_size * seq_len, max(hidden_units, ffn_hidden_units)]
        int32_t *value_buf_;           // buf_size, [batch_size * seq_len, head_num * size_per_head]
        float *query_out_buf_;         // buf_size, [batch_size, head_num, seq_len, size_per_head]
        float *key_out_buf_;           // buf_size, [batch_size, head_num, seq_len, size_per_head]
        float *value_out_fp_buf_;      // buf_size, [batch_size, head_num, seq_len, size_per_head]
        float *qk_buf_;                // [batch_size * head_num, seq_len, total_len_]
        float *qkv_buf_;               // buf_size, [batch_size * head_num, seq_len, size_per_head]
        DataType_ *ffn_tensor_buf_;    // buf_size, [batch_size, seq_len, head_num * size_per_head]
        float *ffn_inter_scale_buf_;   // batch_size * max_prompt_len_, [batch_size, seq_len]

    public:
        OpenDecoder(int batch_size, int max_prompt_len, int max_gen_len, int head_num, int size_per_head, int ffn_hidden_units);

        void initialize(DecoderInitParam<DataType_, weight_DataType_> param, char *buf);

        int getWorkspaceSize();

        /**
         * key_cache_ value_cache_: cache_size, [batch_size, head_num, total_len_, size_per_head]
         * freq_cis_: [max_prompt_len_, size_per_head]
         */
        void forward(const DataType_ *from_tensor, const float *freq_cis, float *key_cache_, float *value_cache_,
                     DataType_ *decoder_output, const int start_pos, const int seq_len);

        ~OpenDecoder();
    };

} // namespace FasterLLaMA
