#pragma once

#include "cuda_kernels.cuh"

namespace tinycudallama
{
    /**
     * decoding_params.sequence_length is initialized by 0
     * finished_buf_ is initialized by false
     */
    __global__ void topKSamplingInitKernel(bool *__restrict__ finished, int *__restrict__ sequence_length);

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

    template <typename T>
    void launchEmbeddingLookupKernel(T *__restrict__ from_tensor, const T *__restrict__ embedding_table, const int *__restrict__ word_ids,
                                     const int batch_size, const int cur_seq_len, const int seq_len, const int hidden_units,
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

#define CASE_K(K)                                                                                                                                   \
    case K:                                                                                                                                         \
        beam_topK_kernel<T, K, local_block_size><<<batch_size, local_block_size, 0, stream>>>(log_probs,                                            \
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
                                  bool *__restrict__ finished_buf, const int *__restrict__ prompt_tokens,
                                  const bool *__restrict__ prompt_tokens_mask, const int cur_pos, const int max_prompt_seq_len,
                                  const int random_num, int *__restrict__ output_ids, int *__restrict__ sequence_length,
                                  const int batch_size, const int vocab_size, const float probability_threshold, cudaStream_t stream = 0);
}
