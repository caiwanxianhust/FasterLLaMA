#pragma once

#include <cub/cub.cuh>
#include "cuda_kernels.cuh"

namespace tinycudallama
{
    namespace
    {
        constexpr int defalut_block_size = 256;
    }

    
     /** resNorm
      * grid(batch_size * seq_len)  block(128)
      * output: [batch_size, seq_len, hidden_units]
      * input: [batch_size, seq_len, hidden_units]
      * gamma: [hidden_units, ]
      */
     template <typename DataType>
     __global__ void resNormKernel(DataType *__restrict__ output, const DataType *__restrict__ input,
                                   const DataType *__restrict__ gamma, const float eps, const int hidden_units);

     template <>
     __global__ void resNormKernel(half *__restrict__ output, const half *__restrict__ input,
                                   const half *__restrict__ gamma, const float eps, const int hidden_units);

     template <typename DataType>
     void launchResNormKernel(DataType *output, const DataType *input, const DataType *gamma, const float eps,
                              const int m, const int n, cudaStream_t stream = 0);

    /** precomputeFreqsCis
      * grid(seq_len)  block(block_size) for size_per_head/2 >= block_size(128)
      * freq_cis: [seq_len, size_per_head]
      */
     __global__ void precomputeFreqsCis(float *freq_cis, const int size_per_head);

     /**
      * block(32, 4)   each warp compute one row
      */
     __global__ void warpPrecomputeFreqsCis(float *freq_cis, const int size_per_head, const int seq_len);

     void launchPrecomputeFreqsCis(float *freq_cis, const int size_per_head, const int seq_len, cudaStream_t stream = 0);

    /**
     * decoding_params.sequence_length is initialized by 0
     * finished_buf_ is initialized by false
     */
    __global__ void topKSamplingInitKernel(bool *__restrict__ finished, int *__restrict__ sequence_length, const int batch_size);

    void launchTopKSamplingInitKernel(bool *__restrict__ finished, int *__restrict__ sequence_length,
                                      const int batch_size, cudaStream_t stream = 0);

    /**
     * decoding_params.sequence_length is initialized by 0
     * finished_buf_ is initialized by false
     * topp_offset_buf is initialized by [0, vocab_size, ..., batch_size * vocab_size]
     * topp_id_val_buf is initialized by [[0, 1, ..., vocab_size-1], [0, 1, ..., vocab_size-1], ..., [0, 1, ..., vocab_size-1]]
     */
    __global__ void topPInitializationKernel(bool *__restrict__ finished, int *__restrict__ sequence_length,
                                             int *__restrict__ topp_id_val_buf, int *__restrict__ topp_offset_buf,
                                             const int batch_size, const int vocab_size);

    void launchTopPInitializationKernel(bool *__restrict__ finished, int *__restrict__ sequence_length,
                                        int *__restrict__ topp_id_val_buf, int *__restrict__ topp_offset_buf,
                                        const int batch_size, const int vocab_size, cudaStream_t stream = 0);

    template <typename T>
    __global__ void embeddingLookupKernel(T *__restrict__ from_tensor, const T *__restrict__ embedding_table,
                                          const int *__restrict__ word_ids, const int hidden_units);

    template <>
    __global__ void embeddingLookupKernel(half *__restrict__ from_tensor, const half *__restrict__ embedding_table,
                                          const int *__restrict__ word_ids, const int hidden_units);

    template <typename T>
    void launchEmbeddingLookupKernel(T *__restrict__ from_tensor, const T *__restrict__ embedding_table, const int *__restrict__ word_ids,
                                     const int batch_size, const int cur_seq_len, const int hidden_units,
                                     cudaStream_t stream = 0);

    /** 取 logits[:, -1, :] 存入 step_logits，并顺便进行停止符判断
     * grid(batch_size), block(min(vocab_size, 1024))
     * step_logits: [batch_size, 1, vocab_size]
     * logits: [batch_size, seq_len, vocab_size]
     * finished: [batch_size, 1]
     */
    __global__ void updateLogitsWithoutSoftmax(float *__restrict__ step_logits, const float *__restrict__ logits, const int end_id,
                                               const bool *__restrict__ finished, const int seq_len, const int vocab_size);

    void launchUpdateLogitsWithoutSoftmax(float *__restrict__ step_logits, const float *__restrict__ logits, const int end_id,
                                          const bool *__restrict__ finished, const int batch_size, const int seq_len,
                                          const int vocab_size, cudaStream_t stream = 0);

    template <typename T, int MAX_K>
    struct TopK
    {
        int p[MAX_K];
        T u[MAX_K];

        __device__ __forceinline__ void insert(T elem, int elem_id)
        {
            if (elem > u[MAX_K - 1] || (p[MAX_K - 1] == -1) || ((elem == u[MAX_K - 1]) && (elem_id < p[MAX_K - 1])))
            // if (elem > u[MAX_K-1] || ((elem == u[MAX_K-1]) && (elem_id < p[MAX_K-1])))
            {
                u[MAX_K - 1] = elem;
                p[MAX_K - 1] = elem_id;
            }

            for (int k = MAX_K - 2; k >= 0; --k)
            {
                if ((u[k + 1] > u[k]) || (p[k] == -1) || ((u[k + 1] == u[k]) && (p[k + 1] < p[k])))
                // if ((u[k+1] > u[k]) || ((u[k+1] == u[k])&&(p[k+1] < p[k])))
                {
                    T u2 = u[k];
                    int p2 = p[k];
                    u[k] = u[k + 1];
                    p[k] = p[k + 1];
                    u[k + 1] = u2;
                    p[k + 1] = p2;
                }
            }
        }

        __device__ __forceinline__ void init()
        {
#pragma unroll
            for (int i = 0; i < MAX_K; i++)
            {
                p[i] = -1;
                u[i] = -FLT_MAX;
            }
        }
    };

    template <typename T, int MAX_K>
    __device__ __forceinline__ TopK<T, MAX_K> reduce_topk_op(const TopK<T, MAX_K> &a, const TopK<T, MAX_K> &b)
    {
        TopK<T, MAX_K> res = a;
        for (int i = 0; i < MAX_K; ++i)
            res.insert(b.u[i], b.p[i]);
        return res;
    }

    /**
     * top-k Sampling kernel
     * grid(1), block(batch_size)
     */
    template <typename T>
    __global__ void topKSampling(int *__restrict__ topk_tmp_id_buf, T *__restrict__ topk_tmp_val_buf, int *__restrict__ ids,
                                 int *__restrict__ sequence_length, bool *__restrict__ finished_buf,
                                 const int *__restrict__ prompt_tokens, const bool *__restrict__ prompt_tokens_mask,
                                 const int cur_pos, const int max_prompt_seq_len, const int candidate_num,
                                 const int random_num, const int end_id, const int batch_size, const int vocab_size);

    template <typename T, int MAX_K, int THREADBLOCK_SIZE>
    __launch_bounds__(THREADBLOCK_SIZE) __global__ void beam_topK_kernel(const T *__restrict__ log_probs,
                                                                         int *__restrict__ topk_tmp_id_buf,
                                                                         T *__restrict__ topk_tmp_val_buf,
                                                                         const int vocab_size,
                                                                         T diversity_rate);

#define CASE_K(K)                                                                                                                                       \
    case K:                                                                                                                                             \
        beam_topK_kernel<T, K, defalut_block_size><<<batch_size, defalut_block_size, 0, stream>>>(log_probs,                                            \
                                                                                                  topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, 0.0f); \
        break;

    template <typename T>
    void launchTopKSamplingKernel(T *__restrict__ log_probs, int *__restrict__ topk_tmp_id_buf, T *__restrict__ topk_tmp_val_buf,
                                  int *__restrict__ ids, int *__restrict__ sequence_length, bool *__restrict__ finished_buf,
                                  const int *__restrict__ prompt_tokens, const bool *__restrict__ prompt_tokens_mask,
                                  const int cur_pos, const int max_prompt_seq_len, int random_num, const int batch_size,
                                  const int vocab_size, const int candidate_num, const int end_id, cudaStream_t stream = 0);

    __global__ void updateLogitsKernelWithoutLog(float *__restrict__ step_logits, const float *__restrict__ logits,
                                                 const bool *__restrict__ finished, const int seq_len, const int end_id,
                                                 const int vocab_size);

    void launchUpdateLogitsKernelWithoutLog(float *__restrict__ step_logits, const float *__restrict__ logits,
                                            const bool *__restrict__ finished, const int seq_len, const int end_id,
                                            const int batch_size, const int vocab_size, cudaStream_t stream = 0);

    /**
     * top-k Sampling kernel
     * grid(1), block(batch_size)
     */
    template <typename T>
    __global__ void topPSampling(const T *__restrict__ sorted_logits_probs, const int *__restrict__ sorted_id_vals,
                                 int *__restrict__ ids, int *__restrict__ sequence_length, bool *__restrict__ finished_buf,
                                 const int *__restrict__ prompt_tokens, const bool *__restrict__ prompt_tokens_mask,
                                 const int cur_pos, const int max_prompt_seq_len, const int batch_size, const int vocab_size,
                                 const int random_num, const float prob_threshold, const int end_id);

    /**
     * Get the temporary memory buffer size of topp sort by calling the function: cub::DeviceSegmentedRadixSort::SortPairsDescending
     */
    size_t getToppSortTempStorageSize(const float *__restrict__ log_probs, const int *__restrict__ id_vals,
                                      float *__restrict__ sorted_log_probs, int *__restrict__ sorted_id_vals,
                                      int *__restrict__ topp_offset_buf, const int batch_size, const int vocab_size);

    template <typename T>
    void launchTopPSamplingKernel(const T *__restrict__ logits_probs, const int *__restrict__ id_vals, T *__restrict__ sorted_logits_probs,
                                  int *__restrict__ sorted_id_vals, const int *__restrict__ topp_offset_buf, void *__restrict__ temp_storage,
                                  size_t temp_storage_size, bool *__restrict__ finished_buf, const int *__restrict__ prompt_tokens,
                                  const bool *__restrict__ prompt_tokens_mask, const int cur_pos, const int max_prompt_seq_len,
                                  const int random_num, int *__restrict__ output_ids, int *__restrict__ sequence_length, const int end_id,
                                  const int batch_size, const int vocab_size, const float probability_threshold, cudaStream_t stream = 0);

    __global__ void removePromptTokenKernel(int *__restrict__ gen_ids, const int *__restrict__ word_ids_buf,
                                            const int *__restrict__ sequence_length, const int *__restrict__ prompt_seq_lengths,
                                            const int min_prompt_seq_len, const int batch_size, const int total_len);

    void launchRemovePromptTokenKernel(int *__restrict__ gen_ids, const int *__restrict__ word_ids_buf, const int *__restrict__ sequence_length,
                                       const int *__restrict__ prompt_seq_lengths, const int min_prompt_seq_len, const int batch_size,
                                       const int total_len, cudaStream_t stream = 0);
}
