#pragma once

#include "common.h"
#include "open_decoder.h"
#include "allocator.h"

namespace FasterLLaMA
{
    template <typename T>
    class DecodingInitParam
    {
    public:
        T *embedding_table;
        float *freq_cis;

        // the length of tokens in the batch. [batch_size, ]
        int *prompt_sequence_length;
        int min_prompt_seq_len;
        int max_prompt_seq_len;
        // [batch_size, max_prompt_seq_len]
        int *prompt_tokens;
        // [batch_size, max_prompt_seq_len], pad token is 0, otherwise 1.
        bool *prompt_tokens_mask;

        ResNormWeight<T> decodingnorm;

        DenseWeight<T, T> output_weight;

        int *output_ids;
        int *sequence_length;
        cublasHandle_t cublas_handle;
        cudaStream_t stream;
    };

    struct TransformerArguments
    {
        int batch_size_;
        int seq_len_;
        int head_num_;
        int size_per_head_;
        int hidden_units_;
        int ffn_hidden_units_;
    };

    struct DecodingArguments : public TransformerArguments
    {
        int decoder_layers_;
        int vocab_size_;
        int start_id_;
        // the eos token's index
        int end_id_;
        int max_prompt_len_;
        int max_gen_len_;
    };

    struct DecodingSamplingArguments : public DecodingArguments
    {
        // the k for top-k sampling
        int candidate_num_;
        // the p for top-p sampling
        float probability_threshold_;
        size_t temp_storage_size_;
    };

    struct DecodingBeamsearchArguments : public DecodingArguments
    {
        int beam_width_;
        size_t temp_storage_size_;
        float beam_search_diversity_rate_;
    };

    template <OperationType OpType_>
    class DecodingSampling
    {
    private:
        typedef DecoderTransformerTraits<OpType_> Traits_;
        typedef typename Traits_::DataType DataType_;
        const IAllocator &allocator_;
        struct DecodingSamplingArguments args_;

        const cublasComputeType_t computeType_ = Traits_::computeType;
        const cudaDataType_t AType_ = Traits_::AType;
        const cudaDataType_t BType_ = Traits_::BType;
        const cudaDataType_t CType_ = Traits_::CType;
        int cublasAlgo_[1] = {20};

        OpenDecoder<OpType_, OperationType::INT8> *decoder_;
        float *K_cache_;
        float *V_cache_;
        DataType_ *from_tensor_[2];
        char *decoder_buf_;
        DataType_ *decoder_normed_result_buf_;
        float *logits_buf_;
        float *step_logits_buf_;
        int *word_ids_buf_;
        bool *finished_buf_;
        int *topk_ids_buf_;
        float *topk_val_buf_;
        void *buf_;
        bool *h_finished_buf_;
        // is initialized by [[0, 1, ..., vocab_size-1], [0, 1, ..., vocab_size-1], ..., [0, 1, ..., vocab_size-1]]
        int *topp_id_vals_buf_;
        float *topp_sorted_logits_prob_buf_;
        int *topp_sorted_id_vals_buf_;
        // is initialized by [0, vocab_size, ..., batch_size * vocab_size]
        int *topp_offset_buf_;

        void *temp_storage_;

    public:
        DecodingSampling(const IAllocator &allocator, const int batch_size,
                         const int max_prompt_len, const int max_gen_len,
                         const int head_num, const int size_per_head,
                         const int vocab_size, const int decoder_layers,
                         const int end_id, const int ffn_hidden_units,
                         const int candidate_num = 0, const float probability_threshold = 0.0);

        void forward(const DecoderInitParam<DataType_, int8_t> *param,
                     DecodingInitParam<DataType_> decoding_params);
       

        virtual ~DecodingSampling();
    };

} // namespace FasterLLaMA
