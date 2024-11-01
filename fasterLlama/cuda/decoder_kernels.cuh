#pragma once

#include <cuda_fp16.h>

namespace tinycudallama
{

     /** resNorm、量化
      * grid(batch_size * seq_len)  block(128)
      * output: [batch_size, seq_len, hidden_units]
      * input: [batch_size, seq_len, hidden_units]
      * gamma: [hidden_units, ]
      */
     template <typename DataType>
     __global__ void resNormQuantizedKernel(int8_t *__restrict__ output, const DataType *__restrict__ input, const DataType *__restrict__ gamma,
                                            float *__restrict__ norm_scale, const float eps, const int hidden_units);

     template <>
     __global__ void resNormQuantizedKernel(int8_t *__restrict__ output, const half *__restrict__ input, const half *__restrict__ gamma,
                                            float *__restrict__ norm_scale, const float eps, const int hidden_units);

     template <typename DataType>
     void launchResNormQuantizedKernel(int8_t *output, const DataType *input, const DataType *gamma,
                                       float *norm_scale, const float eps, const int nrows, const int hidden_units, cudaStream_t stream = 0);

     /** embeddingLookingUp
      * grid(batch_size, seq_len) block(128)
      * from_tensor: [batch_size, seq_len, hidden_units]
      * word_ids:    [batch_size, seq_len]
      */
     template <typename DataType>
     __global__ void embeddingLookingUpKernel(DataType *__restrict__ from_tensor, const DataType *__restrict__ embedding_table,
                                              const int *__restrict__ word_ids, const int hidden_units, const int seq_len);

     template <typename DataType>
     void launchEmbeddingLookingUpKernel(DataType *from_tensor, const DataType *embedding_table,
                                         const int *word_ids, const int hidden_units, const int batch_size, const int seq_len,
                                         cudaStream_t stream = 0);
     /** perChannel 量化
      * src: [rows, clos]
      * dst: [rows, clos]
      * scale_ptr: [rows, ]
      */
     template <typename DataType>
     __global__ void perChannelQuantizedKernel(int8_t *__restrict__ dst, const DataType *__restrict__ src, float *__restrict__ scale_ptr,
                                               const int hidden_size);

     template <typename DataType>
     void perChannelQuantizedKernelLauncher(int8_t *dst, const DataType *src, float *scale_ptr, const int hidden_size,
                                            const int nrows, cudaStream_t stream = 0);

     /**
      * 反量化、rope旋转编码、量化、转置
      * Q K: [batch_size, seq_len, head_num, size_per_head]
      * grid(head_num / warp_num, seq_len, batch_size * 2) block(32, warp_num), each warp process size_per_head elements
      * q_inp_sacle k_inp_scale: [batch_size, seq_len], absmax / 127.0f
      * q_weight_scale k_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
      * freq_cis: [max_seq_len, size_per_head]
      * q_out_scale k_out_scale: [batch_size, seq_len, head_num], absmax / 127.0f
      */
     __global__ void warpQKRoteEmbeddingQuantizedTransposeKernel(int8_t *q_buf, int8_t *k_buf, const int32_t *Q,
                                                                 const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                                 const float *q_weight_scale, const float *k_weight_scale, float *q_out_scale,
                                                                 float *k_out_scale, float *freq_cis, const int batch_size, const int seq_len,
                                                                 const int start_pos, const int total_len, const int head_num,
                                                                 const int size_per_head);

     /**
      * 反量化、rope旋转编码、量化、转置
      * Q K: [batch_size, seq_len, head_num, size_per_head]
      * grid(head_num, seq_len, batch_size * 2) block(128), each block process size_per_head(256) elements
      * q_inp_sacle k_inp_scale: [batch_size, seq_len], absmax / 127.0f
      * q_weight_scale k_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
      * freq_cis: [max_seq_len, size_per_head]
      * q_out_scale k_out_scale: [batch_size, seq_len, head_num], absmax / 127.0f
      */
     __global__ void blockQKRoteEmbeddingQuantizedTransposeForDim256Kernel(int8_t *q_buf, int8_t *k_buf, const int32_t *Q,
                                                                           const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                                           const float *q_weight_scale, const float *k_weight_scale, float *q_out_scale,
                                                                           float *k_out_scale, float *freq_cis, const int batch_size, const int seq_len,
                                                                           const int start_pos, const int total_len, const int head_num,
                                                                           const int size_per_head);

     /**
      * 反量化、rope旋转编码、量化、转置
      * Q K: [batch_size, seq_len, head_num, size_per_head]
      * grid(head_num, seq_len, batch_size * 2) block(size_per_head / 4), each block process size_per_head elements
      * q_inp_sacle k_inp_scale: [batch_size, seq_len], absmax / 127.0f
      * q_weight_scale k_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
      * freq_cis: [max_seq_len, size_per_head]
      * q_out_scale k_out_scale: [batch_size, head_num, seq_len], absmax / 127.0f
      */
     __global__ void blockQKRoteEmbeddingQuantizedTransposeKernel(int8_t *q_buf, int8_t *k_buf, const int32_t *Q,
                                                                  const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                                  const float *q_weight_scale, const float *k_weight_scale, float *q_out_scale,
                                                                  float *k_out_scale, float *freq_cis, const int batch_size, const int seq_len,
                                                                  const int start_pos, const int total_len, const int head_num,
                                                                  const int size_per_head);

     void launchQKRoteEmbeddingQuantizedTranspose(int8_t *q_buf, int8_t *k_buf, const int32_t *Q,
                                                  const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                  const float *q_weight_scale, const float *k_weight_scale, float *q_out_scale,
                                                  float *k_out_scale, float *freq_cis, const int batch_size, const int seq_len,
                                                  const int start_pos, const int total_len, const int head_num,
                                                  const int size_per_head, cudaStream_t stream = 0);

     /**
      * 反量化、rope旋转编码、转置
      * Q K: [batch_size, seq_len, head_num, size_per_head]
      * grid(head_num / warp_num, seq_len, batch_size * 2) block(32, warp_num), each warp process size_per_head elements
      * q_inp_sacle k_inp_scale: [batch_size, seq_len], absmax / 127.0f
      * q_weight_scale k_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
      * freq_cis: [max_seq_len, size_per_head]
      */
     __global__ void warpQKRoteEmbeddingTransposeKernel(float *q_buf, float *k_buf, const int32_t *Q,
                                                        const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                        const float *q_weight_scale, const float *k_weight_scale, float *freq_cis,
                                                        const int batch_size, const int seq_len,
                                                        const int start_pos, const int total_len, const int head_num,
                                                        const int size_per_head);

     /**
      * 反量化、rope旋转编码、转置
      * Q K: [batch_size, seq_len, head_num, size_per_head]
      * grid(head_num, seq_len, batch_size * 2) block(128), each block process size_per_head(256) elements
      * q_inp_sacle k_inp_scale: [batch_size, seq_len], absmax / 127.0f
      * q_weight_scale k_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
      * freq_cis: [max_seq_len, size_per_head]
      */
     __global__ void blockQKRoteEmbeddingTransposeForDim256Kernel(float *q_buf, float *k_buf, const int32_t *Q,
                                                                  const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                                  const float *q_weight_scale, const float *k_weight_scale,
                                                                  float *freq_cis, const int batch_size, const int seq_len,
                                                                  const int start_pos, const int total_len, const int head_num,
                                                                  const int size_per_head);

     /**
      * 反量化、rope旋转编码、转置
      * Q K: [batch_size, seq_len, head_num, size_per_head]
      * grid(head_num, seq_len, batch_size * 2) block(size_per_head / 4), each block process size_per_head elements
      * q_inp_sacle k_inp_scale: [batch_size, seq_len], absmax / 127.0f
      * q_weight_scale k_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
      * freq_cis: [max_seq_len, size_per_head]
      */
     __global__ void blockQKRoteEmbeddingTransposeKernel(float *q_buf, float *k_buf, const int32_t *Q,
                                                         const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                                         const float *q_weight_scale, const float *k_weight_scale,
                                                         float *freq_cis, const int batch_size, const int seq_len,
                                                         const int start_pos, const int total_len, const int head_num,
                                                         const int size_per_head);

     void launchQKRoteEmbeddingTranspose(float *q_buf, float *k_buf, const int32_t *Q,
                                         const int32_t *K, const float *q_inp_scale, const float *k_inp_scale,
                                         const float *q_weight_scale, const float *k_weight_scale,
                                         const float *freq_cis, const int batch_size, const int seq_len,
                                         const int start_pos, const int total_len, const int head_num,
                                         const int size_per_head, cudaStream_t stream = 0);

     /**
      * grid: [seq_len, head_num / blockDim.y, batch_size * 2]  block(size_per_head / 4, 256 / (size_per_head / 4))
      * k_cache v_cache: [batch_size, head_num, max_seq_len, size_per_head]
      * K V : [batch_size, head_num, seq_len, size_per_head]
      */
     __global__ void storeKVcacheKernel(float *__restrict__ k_cache, float *__restrict__ v_cache, const float *__restrict__ K,
                                        const float *__restrict__ V, const int start_pos, const int seq_len, const int batch_size, const int head_num,
                                        const int max_seq_len, const int size_per_head);

     __global__ void storeKVcacheBlockKernel(float *__restrict__ k_cache, float *__restrict__ v_cache, const float *__restrict__ K,
                                             const float *__restrict__ V, const int start_pos, const int seq_len, const int batch_size, const int head_num,
                                             const int max_seq_len, const int size_per_head);

     void launchStoreKVcacheKernel(float *k_cache, float *v_cache, const float *K, const float *V, const int start_pos, const int seq_len,
                                   const int batch_size, const int head_num, const int max_seq_len, const int size_per_head,
                                   cudaStream_t stream = 0);

     /**
      * grid: [seq_len, head_num / blockDim.y, batch_size * 2]  block(size_per_head / 4, 256 / (size_per_head / 4))
      * k_cache v_cache: [batch_size, head_num, max_seq_len, size_per_head]
      * K V : [batch_size, head_num, seq_len, size_per_head]
      * k_scale: [batch_size, head_num, seq_len]
      * k_scale_cache: [batch_size, head_num, max_seq_len]
      */
     __global__ void storeKVcacheKernel(int8_t *__restrict__ k_cache, float *__restrict__ v_cache, float *__restrict__ k_scale_cache,
                                        const int8_t *__restrict__ K, const float *__restrict__ V, const float *__restrict__ k_scale,
                                        const int start_pos, const int seq_len, const int batch_size, const int head_num,
                                        const int max_seq_len, const int size_per_head);

     void launchStoreKVcacheKernel(int8_t *k_cache, float *v_cache, float *k_scale_cache, const int8_t *K, const float *V,
                                   const float *k_scale, const int start_pos, const int seq_len, const int batch_size,
                                   const int head_num, const int max_seq_len, const int size_per_head, cudaStream_t stream = 0);

     /**
      * grid: [seq_len, head_num / blockDim.y, batch_size * 2]  block(size_per_head / 4, 256 / (size_per_head / 4))
      * k_cache v_cache: [batch_size, head_num, max_seq_len, size_per_head]
      * K V : [batch_size, head_num, seq_len, size_per_head]
      */
     __global__ void storeINT8KVcacheKernel(int8_t *__restrict__ k_cache, int8_t *__restrict__ v_cache, const int8_t *__restrict__ K,
                                            const int8_t *__restrict__ V, const int start_pos, const int seq_len, const int batch_size,
                                            const int head_num, const int max_seq_len, const int size_per_head);

     void launchINT8StoreKVcacheKernel(int8_t *k_cache, int8_t *v_cache, const int8_t *K, const int8_t *V, const int start_pos,
                                       const int seq_len, const int batch_size, const int head_num, const int max_seq_len,
                                       const int size_per_head, cudaStream_t stream = 0);

     /**从 K cache 拷贝数据用于后续 gemm
      * grid(end_seq_id, head_num * batch_size) block(128)
      * from [batch_size, head_num, total_len, size_per_head] to [batch_size, head_num, end_seq_id, size_per_head]
      */
     __global__ void copyKFromCacheKernel(int8_t *__restrict__ k_buf, const int8_t *__restrict__ k_cache,
                                          const int nrows, const int total_len, const int end_seq_id, const int size_per_head);

     void launchCopyKFromCacheKernel(int8_t *k_buf, const int8_t *k_cache, const int nrows, const int total_len,
                                     const int end_seq_id, const int size_per_head, cudaStream_t stream = 0);

     /**
      * 反量化、softmax、量化
      * grid(seq_len_q, head_num, batch_size), block(128), each block process seq_len_k elements
      * qk score: [batch_size, head_num, seq_len_q, seq_len_k]
      * atten_mask: [max_seq_len, max_seq_len]
      * q_inp_scale: [batch_size, head_num, seq_len_q]
      * k_inp_scale: [batch_size, head_num, seq_len_k]
      * score_scale: [batch_size, head_num, seq_len_q]
      *
      */
     __global__ void blockDeQuantizedSoftmaxQuantizedKernel(int8_t *__restrict__ score, const int32_t *__restrict__ qk,
                                                            const float *__restrict__ attn_mask, const float *__restrict__ q_inp_scale,
                                                            const float *__restrict__ k_inp_scale, float *__restrict__ score_scale,
                                                            const float attn_scale, const int batch_size, const int head_num,
                                                            const int seq_len_q, const int seq_len_k, const int max_seq_len);

     void launchBlockDeQuantizedSoftmaxQuantizedKernel(int8_t *score, const int32_t *qk, const float *attn_mask, const float *q_inp_scale,
                                                       const float *k_inp_scale, float *score_scale, const float attn_scale,
                                                       const int batch_size, const int head_num, const int seq_len_q, const int seq_len_k,
                                                       const int max_seq_len, cudaStream_t stream = 0);

     /**
      * softmax
      * grid(seq_len_q, head_num, batch_size), block(128), each block process seq_len_k elements
      * qk score: [batch_size, head_num, seq_len_q, seq_len_k]
      * atten_mask: [max_seq_len, max_seq_len]
      *
      */
     __global__ void blockSoftmaxKernel(float *__restrict__ qk, const float *__restrict__ attn_mask, const int batch_size,
                                        const int head_num, const int seq_len_q, const int seq_len_k, const int max_seq_len,
                                        const float scaler);

     void launchBlockSoftmaxKernel(float *qk, const float *attn_mask, const int batch_size, const int head_num, const int seq_len_q,
                                   const int seq_len_k, const int max_seq_len, const float scaler, cudaStream_t stream = 0);

     /**
      * 反量化、转置
      * grid(head_num / warp_num, seq_len, batch_size) block(32, warp_num), each warp process size_per_head elements
      * V: [batch_size, seq_len, head_num, size_per_head]
      * v_buf: [batch_size, head_num, seq_len, size_per_head]
      * v_inp_sacle: [batch_size, seq_len], absmax / 127.0f
      * v_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
      */
     __global__ void warpDequantizedVTransposeKernel(float *__restrict__ v_buf, const int32_t *__restrict__ V,
                                                     const float *__restrict__ v_inp_scale, const float *__restrict__ v_weight_scale,
                                                     const int batch_size, const int seq_len, const int head_num, const int size_per_head);

     /**
      * 反量化、转置
      * grid(head_num, seq_len, batch_size) block(128), each block process size_per_head elements
      * V: [batch_size, seq_len, head_num, size_per_head]
      * v_buf: [batch_size, head_num, seq_len, size_per_head]
      * v_inp_sacle: [batch_size, seq_len], absmax / 127.0f
      * v_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
      */
     __global__ void blockDequantizedVTransposeFor256Kernel(float *__restrict__ v_buf, const int32_t *__restrict__ V,
                                                            const float *__restrict__ v_inp_scale, const float *__restrict__ v_weight_scale,
                                                            const int batch_size, const int seq_len, const int head_num, const int size_per_head);

     /**
      * 反量化、转置
      * grid(head_num, seq_len, batch_size) block(size_per_head / 4), each block process size_per_head elements
      * V: [batch_size, seq_len, head_num, size_per_head]
      * v_buf: [batch_size, head_num, seq_len, size_per_head]
      * v_inp_sacle: [batch_size, seq_len], absmax / 127.0f
      * v_weight_scale: [head_num * size_per_head, ], absmax / 127.0f
      */
     __global__ void blockDequantizedVTransposeKernel(float *__restrict__ v_buf, const int32_t *__restrict__ V,
                                                      const float *__restrict__ v_inp_scale, const float *__restrict__ v_weight_scale,
                                                      const int batch_size, const int seq_len, const int head_num, const int size_per_head);

     void launchDequantizedVTransposeKernel(float *v_buf, const int32_t *V, const float *v_inp_scale, const float *v_weight_scale,
                                            const int batch_size, const int seq_len, const int head_num, const int size_per_head,
                                            cudaStream_t stream = 0);

     /**
      * 量化
      * grid(head_num, batch_size) block(size_per_head), each warp process seq_len elements
      * V: [batch_size, head_num, seq_len, size_per_head]
      * v_buf: [batch_size, head_num, seq_len, size_per_head]
      * v_out_scale: [batch_size, head_num, 1, size_per_head], absmax / 127.0f
      */
     __global__ void blockVQuantizedKernel(int8_t *__restrict__ v_buf, const float *__restrict__ V, float *__restrict__ v_out_scale,
                                           const int batch_size, const int seq_len, const int head_num, const int size_per_head);

     void launchBlockVQuantizedKernel(int8_t *v_buf, const float *V, float *v_out_scale, const int batch_size, const int seq_len,
                                      const int head_num, const int size_per_head, cudaStream_t stream = 0);

     /** 反量化、量化、转置
      * grid(seq_len, batch_size) block(32 * head_num)
      * attn_buf:[batch_size, seq_len, head_num, size_per_head]
      * attn:[batch_size, head_num, seq_len, size_per_head]
      * score_scale:[batch_size, head_num, seq_len]
      * v_scale:[batch_size, head_num, size_per_head]
      * attn_out_scale:[batch_size, seq_len]
      */
     __global__ void warpDequantizedAttnQuantizedTransposeKernel(int8_t *__restrict__ attn_buf, const int32_t *__restrict__ attn,
                                                                 const float *__restrict__ score_scale, const float *__restrict__ v_scale,
                                                                 float *__restrict__ attn_out_scale, const int batch_size,
                                                                 const int head_num, const int seq_len, const int size_per_head);

     /** 反量化、量化、转置
      * grid(seq_len, batch_size) block(size_per_head)
      * attn_buf:[batch_size, seq_len, head_num, size_per_head]
      * attn:[batch_size, head_num, seq_len, size_per_head]
      * score_scale:[batch_size, head_num, seq_len]
      * v_scale:[batch_size, head_num, size_per_head]
      * attn_out_scale:[batch_size, seq_len]
      */
     __global__ void blockDequantizedAttnQuantizedTransposeKernel(int8_t *__restrict__ attn_buf, const int32_t *__restrict__ attn,
                                                                  const float *__restrict__ score_scale, const float *__restrict__ v_scale,
                                                                  float *__restrict__ attn_out_scale, const int batch_size,
                                                                  const int head_num, const int seq_len, const int size_per_head);

     void launchDequantizedAttnQuantizedTransposeKernel(int8_t *__restrict__ attn_buf, const int32_t *__restrict__ attn,
                                                        const float *__restrict__ score_scale, const float *__restrict__ v_scale,
                                                        float *__restrict__ attn_out_scale, const int batch_size,
                                                        const int head_num, const int seq_len, const int size_per_head,
                                                        cudaStream_t stream = 0);

     /** 量化、转置
      * grid(seq_len, batch_size) block(32 * head_num)
      * attn_buf:[batch_size, seq_len, head_num, size_per_head]
      * attn:[batch_size, head_num, seq_len, size_per_head]
      * attn_out_scale:[batch_size, seq_len]
      */
     __global__ void warpAttnQuantizedTransposeKernel(int8_t *__restrict__ attn_buf, const float *__restrict__ attn,
                                                      float *__restrict__ attn_out_scale, const int batch_size,
                                                      const int head_num, const int seq_len, const int size_per_head);

     /** 量化、转置
      * grid(seq_len, batch_size) block(size_per_head)
      * attn_buf:[batch_size, seq_len, head_num, size_per_head]
      * attn:[batch_size, head_num, seq_len, size_per_head]
      * attn_out_scale:[batch_size, seq_len]
      */
     __global__ void blockAttnQuantizedTransposeKernel(int8_t *__restrict__ attn_buf, const float *__restrict__ attn,
                                                       float *__restrict__ attn_out_scale, const int batch_size,
                                                       const int head_num, const int seq_len, const int size_per_head);

     void launchAttnQuantizedTransposeKernel(int8_t *__restrict__ attn_buf, const float *__restrict__ attn,
                                             float *__restrict__ attn_out_scale, const int batch_size,
                                             const int head_num, const int seq_len, const int size_per_head, cudaStream_t stream = 0);

     /**反量化、残差结构、量化
      * grid(seq_len * batch_size) block(128)
      * norm_out: [batch_size, seq_len, hidden_units]
      * ffn_tensor: [batch_size, seq_len, hidden_units]
      * from_temsor: [batch_size, seq_len, hidden_units]
      * attn_out: [batch_size, seq_len, hidden_units]
      * attn_out_scale: [batch_size, seq_len]
      * attn_weight_scale: [hidden_units]
      * gamma: [hidden_units]
      * norm_scale: [batch_size, seq_len]
      */
     template <typename DataType>
     __global__ void dequantizedResidualResNormQuantizedKernel(int8_t *__restrict__ norm_out, DataType *__restrict__ ffn_tensor,
                                                               const DataType *__restrict__ from_temsor, const int32_t *__restrict__ attn_out,
                                                               const float *__restrict__ attn_out_scale, const float *__restrict__ attn_weight_scale,
                                                               const DataType *__restrict__ gamma, float *__restrict__ norm_scale,
                                                               const float eps, const int hidden_units);

     /**反量化、残差结构、量化
      * grid(seq_len * batch_size) block(128)
      * norm_out: [batch_size, seq_len, hidden_units]
      * ffn_tensor: [batch_size, seq_len, hidden_units]
      * from_temsor: [batch_size, seq_len, hidden_units]
      * attn_out: [batch_size, seq_len, hidden_units]
      * attn_out_scale: [batch_size, seq_len]
      * attn_weight_scale: [hidden_units]
      * gamma: [hidden_units]
      */
     template <>
     __global__ void dequantizedResidualResNormQuantizedKernel(int8_t *__restrict__ norm_out, half *__restrict__ ffn_tensor,
                                                               const half *__restrict__ from_temsor, const int32_t *__restrict__ attn_out,
                                                               const float *__restrict__ attn_out_scale, const float *__restrict__ attn_weight_scale,
                                                               const half *__restrict__ gamma, float *__restrict__ norm_scale,
                                                               const float eps, const int hidden_units);

     template <typename DataType>
     void launchDequantizedResidualResNormQuantized(int8_t *norm_out, DataType *__restrict__ ffn_tensor, const DataType *from_temsor,
                                                    const int32_t *attn_out, const float *attn_out_scale,
                                                    const float *attn_weight_scale, const DataType *gamma,
                                                    float *norm_scale, const float eps, const int rows, const int hidden_units,
                                                    cudaStream_t stream = 0);

     /** 反量化、silu、element-wise-multify、量化
      * grid(nrows) block(128)
      * out_buf: [nrows, hidden_units]
      * w1_ret w3_ret: [nrows, hidden_units]
      * norm_scale: [nrows, ]
      * w1_weight_scale w3_weight_scale: [hidden_units, ]
      * out_scale: [nrows, ]
      */
     __global__ void dequantizedSiluMultifyQuantizedKernel(int8_t *__restrict__ out_buf, const int32_t *__restrict__ w1_ret,
                                                           const float *__restrict__ norm_scale, const float *__restrict__ w1_weight_scale,
                                                           const int32_t *__restrict__ w3_ret, const float *__restrict__ w3_weight_scale,
                                                           float *__restrict__ out_scale, const int hidden_units);

     void launchDequantizedSiluMultifyQuantized(int8_t *out_buf, const int32_t *w1_ret, const float *norm_scale, const float *w1_weight_scale,
                                                const int32_t *w3_ret, const float *w3_weight_scale, float *out_scale, const int nrows,
                                                const int hidden_units, cudaStream_t stream = 0);

     /**反量化、残差结构
      * grid(seq_len * batch_size) block(128)
      * out: [batch_size, seq_len, hidden_units]
      * ffn_tensor: [batch_size, seq_len, hidden_units]
      * from_temsor: [batch_size, seq_len, hidden_units]
      * inp: [batch_size, seq_len, hidden_units]
      * inp_scale: [batch_size, seq_len]
      * weight_scale: [hidden_units]
      */
     template <typename DataType>
     __global__ void dequantizedResidualKernel(DataType *__restrict__ out, const DataType *__restrict__ from_temsor,
                                               const int32_t *__restrict__ inp, const float *__restrict__ inp_scale,
                                               const float *__restrict__ weight_scale, const int hidden_units);

     template <>
     __global__ void dequantizedResidualKernel(half *__restrict__ out, const half *__restrict__ from_temsor,
                                               const int32_t *__restrict__ inp, const float *__restrict__ inp_scale,
                                               const float *__restrict__ weight_scale, const int hidden_units);

     template <typename DataType>
     void launchDequantizedResidual(DataType *__restrict__ out, const DataType *__restrict__ from_temsor, const int32_t *__restrict__ inp,
                                    const float *__restrict__ inp_scale, const float *__restrict__ weight_scale, const int nrows,
                                    const int hidden_units, cudaStream_t stream = 0);

}