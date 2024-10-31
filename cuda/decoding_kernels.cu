#include "decoding_kernels.cuh"

namespace tinycudallama
{
    /**
     * decoding_params.sequence_length is initialized by 0
     * finished_buf_ is initialized by false
     */
    __global__ void topKSamplingInitKernel(bool *__restrict__ finished, int *__restrict__ sequence_length, const int batch_size)
    {
        int tid = threadIdx.x;
        if (tid < batch_size)
        {
            finished[tid] = false;
            sequence_length[tid] = 0;
        }
    }

    void launchTopKSamplingInitKernel(bool *__restrict__ finished, int *__restrict__ sequence_length,
                                      const int batch_size, cudaStream_t stream)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        dim3 grid(1);
        dim3 block(min(1024, batch_size));
        topKSamplingInitKernel<<<grid, block, 0, stream>>>(finished, sequence_length, batch_size);
    }

    /**
     * decoding_params.sequence_length is initialized by 0
     * finished_buf_ is initialized by false
     * topp_offset_buf is initialized by [0, vocab_size, ..., batch_size * vocab_size]
     * topp_id_val_buf is initialized by [[0, 1, ..., vocab_size-1], [0, 1, ..., vocab_size-1], ..., [0, 1, ..., vocab_size-1]]
     */
    __global__ void topPInitializationKernel(bool *__restrict__ finished, int *__restrict__ sequence_length,
                                             int *__restrict__ topp_id_val_buf, int *__restrict__ topp_offset_buf,
                                             const int batch_size, const int vocab_size)
    {
        int tid = threadIdx.x;
        int bid = blockIdx.x;

        if (bid == 0)
        {
            for (int i = tid; i < batch_size + 1; i += blockDim.x)
            {
                topp_offset_buf[i] = i * vocab_size;
            }

            for (int i = tid; i < batch_size; i += blockDim.x)
            {
                finished[i] = false;
                sequence_length[i] = 0;
            }
        }

        for (int idx = tid + bid * blockDim.x; idx < batch_size * vocab_size; idx += blockDim.x * gridDim.x)
        {
            topp_id_val_buf[idx] = idx % vocab_size;
        }
    }

    void launchTopPInitializationKernel(bool *__restrict__ finished, int *__restrict__ sequence_length,
                                        int *__restrict__ topp_id_val_buf, int *__restrict__ topp_offset_buf,
                                        const int batch_size, const int vocab_size, cudaStream_t stream)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        topPInitializationKernel<<<32, 512, 0, stream>>>(finished, sequence_length, topp_id_val_buf, topp_offset_buf,
                                                         batch_size, vocab_size);
    }

    template <typename T>
    __global__ void embeddingLookupKernel(T *__restrict__ from_tensor, const T *__restrict__ embedding_table,
                                          const int *__restrict__ word_ids, const int hidden_units)
    {
        const int token_id = blockIdx.x;
        const int batch_id = blockIdx.y;
        int write_pos, lookup_pos;
        for (int tid = threadIdx.x; tid < hidden_units; tid += blockDim.x)
        {
            write_pos = tid + token_id * hidden_units + batch_id * gridDim.x * hidden_units;
            lookup_pos = word_ids[batch_id * gridDim.x + token_id] * hidden_units + tid;
            // 1. lookup the table
            // 2. multiply hidden_dim**0.5
            // if (lookup_pos < 0) {
            //     printf("batch_id: %d  token_id: %d  word_id: %d\n", batch_id, token_id, word_ids[batch_id * gridDim.x + token_id]);
            // }

            from_tensor[write_pos] = embedding_table[lookup_pos] * (T)sqrtf(float(hidden_units));
        }
    }

    template <typename T>
    void launchEmbeddingLookupKernel(T *__restrict__ from_tensor, const T *__restrict__ embedding_table, const int *__restrict__ word_ids,
                                     const int batch_size, const int cur_seq_len, const int hidden_units,
                                     cudaStream_t stream)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        dim3 grid(cur_seq_len, batch_size);
        dim3 block(256);
        embeddingLookupKernel<T><<<grid, block, 0, stream>>>(from_tensor, embedding_table, word_ids, hidden_units);
    }

    /** 取 logits[:, -1, :] 存入 step_logits，并顺便进行停止符判断
     * grid(batch_size), block(min(vocab_size, 1024))
     * step_logits: [batch_size, 1, vocab_size]
     * logits: [batch_size, seq_len, vocab_size]
     * finished: [batch_size, 1]
     */
    __global__ void updateLogitsWithoutSoftmax(float *__restrict__ step_logits, const float *__restrict__ logits, const int end_id,
                                               const bool *__restrict__ finished, const int seq_len, const int vocab_size)
    {
        const bool is_finished = finished[blockIdx.x];

        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x)
        {
            int idx = blockIdx.x * seq_len * vocab_size + (seq_len - 1) * vocab_size + tid;
            if (is_finished)
            {
                step_logits[blockIdx.x * vocab_size + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
            }
            else
            {
                step_logits[blockIdx.x * vocab_size + tid] = logits[idx];
            }
        }
    }

    void launchUpdateLogitsWithoutSoftmax(float *__restrict__ step_logits, const float *__restrict__ logits, const int end_id,
                                          const bool *__restrict__ finished, const int batch_size, const int seq_len,
                                          const int vocab_size, cudaStream_t stream)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        dim3 grid(batch_size);
        dim3 block(min(vocab_size, 1024));
        /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
        updateLogitsWithoutSoftmax<<<grid, block, 0, stream>>>(step_logits, logits, end_id, finished, seq_len, vocab_size);
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
                                 const int random_num, const int end_id, const int batch_size, const int vocab_size)
    {
        if (threadIdx.x < batch_size)
        {
            // prompt phase, next_token[:] = prompt_tokens[:, cur_pos]
            if (cur_pos < max_prompt_seq_len && prompt_tokens_mask[threadIdx.x * max_prompt_seq_len + cur_pos])
            {
                ids[threadIdx.x] = prompt_tokens[threadIdx.x * max_prompt_seq_len + cur_pos];
            }
            else
            {
                // The maximum number of k logits in the current batch
                float max_val = (float)topk_tmp_val_buf[threadIdx.x * candidate_num];

                float sum = 0.0f;
                float tmp_val;
                for (int i = 0; i < candidate_num; ++i)
                {
                    tmp_val = __expf(topk_tmp_val_buf[threadIdx.x * candidate_num + i] - max_val);
                    topk_tmp_val_buf[threadIdx.x * candidate_num + i] = tmp_val;
                    sum += tmp_val;
                }

                curandState_t local_state;
                curand_init(random_num, threadIdx.x, 0, &local_state);
                float rand_num = curand_uniform(&local_state) * sum;

                ids[threadIdx.x] = topk_tmp_id_buf[threadIdx.x * candidate_num + candidate_num - 1] % vocab_size;
                for (int i = 0; i < candidate_num; i++)
                {
                    rand_num = rand_num - topk_tmp_val_buf[threadIdx.x * candidate_num + i];
                    if (rand_num <= 0.0f)
                    {
                        ids[threadIdx.x] = topk_tmp_id_buf[threadIdx.x * candidate_num + i] % vocab_size;
                        break;
                    }
                }

                sequence_length[threadIdx.x] = finished_buf[threadIdx.x] ? sequence_length[threadIdx.x] : sequence_length[threadIdx.x] + 1;
                finished_buf[threadIdx.x] = ids[threadIdx.x] == end_id ? true : false;
            }
        }
    }

    template <typename T, int MAX_K, int THREADBLOCK_SIZE>
    __launch_bounds__(THREADBLOCK_SIZE)
        __global__
        void beam_topK_kernel(const T *__restrict__ log_probs,
                              int *__restrict__ topk_tmp_id_buf,
                              T *__restrict__ topk_tmp_val_buf,
                              const int vocab_size,
                              T diversity_rate)
    {
        typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        int thread_id = threadIdx.x;
        int block_id = blockIdx.x;
        TopK<T, MAX_K> partial;

#pragma unroll
        for (int i = 0; i < MAX_K; ++i)
        {
            partial.p[i] = -1;
            partial.u[i] = -FLT_MAX;
        }

#pragma unroll
        for (int elem_id = thread_id; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE)
        {
            int index = elem_id + block_id * vocab_size;
            partial.insert(log_probs[index], index);
        }

        TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

        if (thread_id == 0)
        {
            int index = block_id * MAX_K;

#pragma unroll
            for (int i = 0; i < MAX_K; ++i)
            {
                topk_tmp_id_buf[index + i] = total.p[i];
                topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T)i;
            }
        }
    }

    template <typename T>
    void launchTopKSamplingKernel(T *__restrict__ log_probs, int *__restrict__ topk_tmp_id_buf, T *__restrict__ topk_tmp_val_buf,
                                  int *__restrict__ ids, int *__restrict__ sequence_length, bool *__restrict__ finished_buf,
                                  const int *__restrict__ prompt_tokens, const bool *__restrict__ prompt_tokens_mask,
                                  const int cur_pos, const int max_prompt_seq_len, int random_num, const int batch_size,
                                  const int vocab_size, const int candidate_num, const int end_id, cudaStream_t stream)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        int local_block_size = 256;
        switch (candidate_num)
        {
            CASE_K(1);
            CASE_K(2);
            CASE_K(4);
        default:
            printf("[ERROR] Topk kernel does not support candidate_num = %d \n", candidate_num);
            exit(0);
            break;
        }
        assert(batch_size <= 1024);
        if (batch_size <= 128)
        {
            local_block_size = 128;
        }
        else if (batch_size <= 256)
        {
            local_block_size = 256;
        }
        else if (batch_size <= 512)
        {
            local_block_size = 512;
        }
        else
        {
            local_block_size = 1024;
        }
        topKSampling<T><<<1, local_block_size, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids, sequence_length, finished_buf,
                                                            prompt_tokens, prompt_tokens_mask, cur_pos, max_prompt_seq_len, candidate_num,
                                                            random_num, end_id, batch_size, vocab_size);
    }

    __global__ void updateLogitsKernelWithoutLog(float *__restrict__ step_logits, const float *__restrict__ logits,
                                                 const bool *__restrict__ finished,
                                                 const int seq_len, const int end_id, const int vocab_size)
    {
        int bid = blockIdx.x;
        bool finish = finished[bid];
        int offset = bid * vocab_size;

        float max_val = -1 * FLT_MAX;

        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x)
        {
            int idx = bid * seq_len * vocab_size + (seq_len - 1) * vocab_size + tid;
            if (finish)
                step_logits[offset + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
            else
                step_logits[offset + tid] = logits[idx];
            max_val = max(max_val, step_logits[offset + tid]);
        }

        max_val = blockAllReduceMax<float>(max_val);

        float sum_val = 0.0f;
        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x)
        {
            step_logits[offset + tid] = __expf(step_logits[offset + tid] - max_val);
            sum_val += step_logits[offset + tid];
        }

        sum_val = blockAllReduceSum<float>(sum_val);

        for (int tid = threadIdx.x; tid < vocab_size; tid += blockDim.x)
        {
            step_logits[offset + tid] = (step_logits[offset + tid] / sum_val);
        }
    }

    void launchUpdateLogitsKernelWithoutLog(float *__restrict__ step_logits, const float *__restrict__ logits,
                                            const bool *__restrict__ finished, const int seq_len, const int end_id,
                                            const int batch_size, const int vocab_size, cudaStream_t stream)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        dim3 grid(batch_size);
        dim3 block(min(vocab_size, 1024));
        /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
        updateLogitsKernelWithoutLog<<<grid, block, 0, stream>>>(step_logits, logits, finished, seq_len, end_id, vocab_size);
    }

    /**
     * top-k Sampling kernel
     * grid(1), block(batch_size)
     */
    template <typename T>
    __global__ void topPSampling(const T *__restrict__ sorted_logits_probs, const int *__restrict__ sorted_id_vals,
                                 int *__restrict__ ids, int *__restrict__ sequence_length, bool *__restrict__ finished_buf,
                                 const int *__restrict__ prompt_tokens, const bool *__restrict__ prompt_tokens_mask,
                                 const int cur_pos, const int max_prompt_seq_len, const int batch_size, const int vocab_size,
                                 const int random_num, const float prob_threshold, const int end_id)
    {
        if (threadIdx.x < batch_size)
        {
            // prompt phase, next_token[:] = prompt_tokens[:, cur_pos]
            if (cur_pos < max_prompt_seq_len && prompt_tokens_mask[threadIdx.x * max_prompt_seq_len + cur_pos])
            {
                ids[threadIdx.x] = prompt_tokens[threadIdx.x * max_prompt_seq_len + cur_pos];
            }
            else
            {
                int tid = threadIdx.x;
                curandState_t local_state;
                curand_init(random_num, tid, 0, &local_state);
                float rand_num = curand_uniform(&local_state) * prob_threshold;
                ids[tid] = sorted_id_vals[vocab_size - 1];

                for (int i = tid * vocab_size; i < tid * vocab_size + vocab_size; i++)
                {
                    rand_num = rand_num - sorted_logits_probs[i];
                    if (rand_num <= 0)
                    {
                        ids[tid] = sorted_id_vals[i];
                        break;
                    }
                }

                sequence_length[tid] = finished_buf[tid] ? sequence_length[tid] : sequence_length[tid] + 1;
                finished_buf[tid] = ids[tid] == end_id ? true : false;
            }
        }
    }

    /**
     * Get the temporary memory buffer size of topp sort by calling the function: cub::DeviceSegmentedRadixSort::SortPairsDescending
     */
    size_t getToppSortTempStorageSize(const float *__restrict__ log_probs,
                                      const int *__restrict__ id_vals,
                                      float *__restrict__ sorted_log_probs,
                                      int *__restrict__ sorted_id_vals,
                                      int *__restrict__ topp_offset_buf,
                                      const int batch_size,
                                      const int vocab_size)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage,
                                                           temp_storage_bytes,
                                                           log_probs,
                                                           sorted_log_probs,
                                                           id_vals,
                                                           sorted_id_vals,
                                                           vocab_size * batch_size,
                                                           batch_size,
                                                           topp_offset_buf, topp_offset_buf + 1);
        return temp_storage_bytes;
    }

    template <typename T>
    void launchTopPSamplingKernel(const T *__restrict__ logits_probs, const int *__restrict__ id_vals, T *__restrict__ sorted_logits_probs,
                                  int *__restrict__ sorted_id_vals, const int *__restrict__ topp_offset_buf, void *__restrict__ temp_storage,
                                  size_t temp_storage_size, bool *__restrict__ finished_buf, const int *__restrict__ prompt_tokens,
                                  const bool *__restrict__ prompt_tokens_mask, const int cur_pos, const int max_prompt_seq_len,
                                  const int random_num, int *__restrict__ output_ids, int *__restrict__ sequence_length, const int end_id,
                                  const int batch_size, const int vocab_size, const float probability_threshold, cudaStream_t stream)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_storage,
                                                           temp_storage_size,
                                                           logits_probs,
                                                           sorted_logits_probs,
                                                           id_vals,
                                                           sorted_id_vals,
                                                           vocab_size * batch_size,
                                                           batch_size,
                                                           topp_offset_buf, topp_offset_buf + 1);

        int local_block_size;
        assert(batch_size <= 1024);
        if (batch_size <= 128)
        {
            local_block_size = 128;
        }
        else if (batch_size <= 256)
        {
            local_block_size = 256;
        }
        else if (batch_size <= 512)
        {
            local_block_size = 512;
        }
        else
        {
            local_block_size = 1024;
        }

        topPSampling<<<1, local_block_size, 0, stream>>>(sorted_logits_probs, sorted_id_vals, output_ids, sequence_length,
                                                         finished_buf, prompt_tokens, prompt_tokens_mask, cur_pos, max_prompt_seq_len,
                                                         batch_size, vocab_size, random_num, probability_threshold, end_id);
    }

    __global__ void removePromptTokenKernel(int *__restrict__ gen_ids, const int *__restrict__ word_ids_buf,
                                            const int *__restrict__ sequence_length, const int *__restrict__ prompt_seq_lengths,
                                            const int min_prompt_seq_len, const int batch_size, const int total_len)
    {
        const int offset = prompt_seq_lengths[blockIdx.x] - min_prompt_seq_len;
        for (int tid = threadIdx.x; tid < sequence_length[blockIdx.x]; tid += blockDim.x)
        {
            gen_ids[blockIdx.x * total_len + tid] = word_ids_buf[(offset + tid) * batch_size + blockIdx.x];
            // printf("batch_id: %d tid: %d  word_id: %d\n", blockIdx.x, tid, gen_ids[blockIdx.x * total_len + tid]);
        }
    }

    void launchRemovePromptTokenKernel(int *__restrict__ gen_ids, const int *__restrict__ word_ids_buf, const int *__restrict__ sequence_length,
                                       const int *__restrict__ prompt_seq_lengths, const int min_prompt_seq_len, const int batch_size, const int total_len, cudaStream_t stream)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif
        removePromptTokenKernel<<<batch_size, 256, 0, stream>>>(gen_ids, word_ids_buf, sequence_length, prompt_seq_lengths, min_prompt_seq_len, batch_size, total_len);
    }

    template void launchEmbeddingLookupKernel(float *__restrict__ from_tensor, const float *__restrict__ embedding_table,
                                              const int *__restrict__ word_ids, const int batch_size, const int cur_seq_len,
                                              const int hidden_units, cudaStream_t stream);

    template void launchTopKSamplingKernel(float *__restrict__ log_probs, int *__restrict__ topk_tmp_id_buf,
                                           float *__restrict__ topk_tmp_val_buf, int *__restrict__ ids,
                                           int *__restrict__ sequence_length, bool *__restrict__ finished_buf,
                                           const int *__restrict__ prompt_tokens, const bool *__restrict__ prompt_tokens_mask,
                                           const int cur_pos, const int max_prompt_seq_len, int random_num, const int batch_size,
                                           const int vocab_size, const int candidate_num, const int end_id, cudaStream_t stream);

    template void launchTopPSamplingKernel(const float *__restrict__ logits_probs, const int *__restrict__ id_vals,
                                           float *__restrict__ sorted_logits_probs, int *__restrict__ sorted_id_vals,
                                           const int *__restrict__ topp_offset_buf, void *__restrict__ temp_storage,
                                           size_t temp_storage_size,
                                           bool *__restrict__ finished_buf, const int *__restrict__ prompt_tokens,
                                           const bool *__restrict__ prompt_tokens_mask, const int cur_pos, const int max_prompt_seq_len,
                                           const int random_num, int *__restrict__ output_ids, int *__restrict__ sequence_length,
                                           const int end_id,
                                           const int batch_size, const int vocab_size, const float probability_threshold,
                                           cudaStream_t stream);
}