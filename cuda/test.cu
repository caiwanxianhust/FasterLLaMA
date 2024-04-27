#include "cuda_kernels.cuh"

#include <cstdio>
#include <cstdlib>

template <typename T>
void printVecInVec(const T *clusters, const int nrows, const int ncols, const int end_row, const int end_col, const char *str)
{
    printf("%s:\n[\n", str);
    for (int i = 0; i < end_row; ++i)
    {
        printf("[");
        for (int j = 0; j < end_col; ++j)
        {
            printf("%g  ", static_cast<float>(clusters[i * ncols + j]));
        }
        printf("]\n");
    }
    printf("]\n");
}

template <>
void printVecInVec(const half *clusters, const int nrows, const int ncols, const int end_row, const int end_col, const char *str)
{
    printf("%s:\n[\n", str);
    if (end_row >= nrows || end_col >= ncols)
        printf("invalid arguments!!!\nend_row >= nrows or end_col >= ncols\n");
    for (int i = 0; i < end_row; ++i)
    {
        printf("[");
        for (int j = 0; j < end_col; ++j)
        {
            printf("%g  ", __half2float(clusters[i * ncols + j]));
        }
        printf("]\n");
    }
    printf("]\n");
}

template <typename T>
void device_malloc(T **ptr, int size)
{
    CHECK_CUDA_ERROR(cudaMalloc((void **)ptr, sizeof(T) * size));
    T *tmp = new T[size];
    for (int i = 0; i < size; i++)
        tmp[i] = (T)((float)rand() / (RAND_MAX + 1.0) * 0.02);
    CHECK_CUDA_ERROR(cudaMemcpy(*ptr, tmp, sizeof(T) * size, cudaMemcpyHostToDevice));
    delete[] tmp;
}

__global__ void convertMatfloat2half(const float *input, half *output, const int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = offset; i < size; i += gridDim.x * blockDim.x)
    {
        output[i] = __float2half(input[i]);
    }
}

template <typename DataType>
void timingResNorm(DataType *output, const DataType *input, const DataType *gamma, const float eps, const int m, const int n,
                   DataType *h_out, const int method)
{
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    cudaEventQuery(start);

    switch (method)
    {
    case 0:
        tinycudallama::launchResNormKernel(output, input, gamma, 1e-7f, m, n);
        break;
    default:
        tinycudallama::launchResNormKernel(output, input, gamma, 1e-7f, m, n);
        break;
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    printf("Time = %g ms.\n", elapsedTime);

    CHECK_CUDA_ERROR(cudaMemcpy(h_out, output, sizeof(DataType) * (m * n), cudaMemcpyDeviceToHost));
    printf("method : %d\n", method);
    printVecInVec(h_out, m, n, 10, 10, "h_out");
}

void testResNorm()
{
    using DataType = float;
    const int m = 10000;
    const int n = 4096;

    DataType *h_out1 = new DataType[m * n * 2];
    DataType *h_out2 = h_out1 + m * n;

    std::srand(1234);

    DataType *d_in;
    DataType *d_out1;
    DataType *d_out2;
    DataType *d_gamma;
    device_malloc<DataType>(&d_in, sizeof(DataType) * (m * n * 3 + n));
    d_out1 = d_in + m * n;
    d_out2 = d_out1 + m * n;
    d_gamma = d_out2 + m * n;

    timingResNorm<DataType>(d_out1, d_in, d_gamma, 1e-7f, m, n, h_out1, 0);
    timingResNorm<DataType>(d_out2, d_in, d_gamma, 1e-7f, m, n, h_out2, 1);

    half *d_in_half;
    half *d_out_half;
    half *d_gamma_half;
    half *h_out_half = new half[m * n];

    device_malloc(&d_in_half, sizeof(half) * (m * n * 2 + n));
    d_out_half = d_in_half + m * n;
    d_gamma_half = d_out_half + m * n;

    convertMatfloat2half<<<m, 256>>>(d_in, d_in_half, m * n);

    convertMatfloat2half<<<1, 256>>>(d_gamma, d_gamma_half, n);
    timingResNorm<half>(d_out_half, d_in_half, d_gamma_half, 1e-7f, m, n, h_out_half, 0);

    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_in_half));
    delete[] h_out_half;
    delete[] h_out1;
}

void testPrecomputeFreqsCis()
{
    const int seq_len = 4096;
    const int size_per_head = 32;
    float *h_freqsCis = new float[seq_len * size_per_head];

    float *d_freqs_cis;

    device_malloc<float>(&d_freqs_cis, sizeof(float) * (seq_len * size_per_head));

    tinycudallama::launchPrecomputeFreqsCis(d_freqs_cis, size_per_head, seq_len);

    CHECK_CUDA_ERROR(cudaMemcpy(h_freqsCis, d_freqs_cis, sizeof(float) * (seq_len * size_per_head), cudaMemcpyDeviceToHost));
    printVecInVec(h_freqsCis, seq_len, size_per_head, 10, size_per_head, "freqs_cis");

    CHECK_CUDA_ERROR(cudaFree(d_freqs_cis));
    delete[] h_freqsCis;
}

void testEmbedding()
{
    using DataType = float;
    const int batch_size = 2;
    const int seq_len = 4;
    const int hidden_units = 32;
    int word_ids[batch_size * seq_len] = {0, 1, 2, 3, 3, 2, 1, 0};

    DataType *h_embedding_table = new DataType[5 * hidden_units];
    DataType *h_from_tensor = new DataType[batch_size * seq_len * hidden_units];
    DataType *d_embedding_table;
    DataType *from_tensor;
    int *d_word_ids;

    device_malloc(&d_embedding_table, sizeof(DataType) * 5 * hidden_units);
    device_malloc(&from_tensor, sizeof(DataType) * batch_size * seq_len * hidden_units);
    device_malloc(&d_word_ids, sizeof(int) * batch_size * seq_len);

    CHECK_CUDA_ERROR(cudaMemcpy(h_embedding_table, d_embedding_table, sizeof(DataType) * 5 * hidden_units, cudaMemcpyDeviceToHost));
    printVecInVec(h_embedding_table, 5, hidden_units, 5, hidden_units, "embedding_table");

    CHECK_CUDA_ERROR(cudaMemcpy(d_word_ids, word_ids, sizeof(int) * batch_size * seq_len, cudaMemcpyHostToDevice));

    tinycudallama::launchEmbeddingLookingUpKernel<DataType>(from_tensor, d_embedding_table, d_word_ids, hidden_units, batch_size, seq_len);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(h_from_tensor, from_tensor, sizeof(DataType) * batch_size * seq_len * hidden_units, cudaMemcpyDeviceToHost));
    printVecInVec(h_from_tensor, batch_size * seq_len, hidden_units, batch_size * seq_len, hidden_units, "from_tensor");

    CHECK_CUDA_ERROR(cudaFree(d_embedding_table));
    CHECK_CUDA_ERROR(cudaFree(from_tensor));
    CHECK_CUDA_ERROR(cudaFree(d_word_ids));

    delete[] h_embedding_table;
    delete[] h_from_tensor;
}

void testPerChannelQuantized()
{
    using DataType = float;
    const int nrows = 8;
    const int hidden_size = 32;
    DataType *h_src = new DataType[nrows * hidden_size];
    for (int i = 0; i < nrows * hidden_size; ++i)
    {
        h_src[i] = ((i & 1) == 0) ? i : (-1.0f) * i;
    }
    printVecInVec(h_src, nrows, hidden_size, nrows, hidden_size, "h_src");

    DataType *d_src;
    float *d_scale;
    int8_t *d_dst;

    device_malloc(&d_src, sizeof(DataType) * nrows * hidden_size);
    device_malloc(&d_scale, sizeof(float) * nrows);
    device_malloc(&d_dst, sizeof(int8_t) * nrows * hidden_size);

    CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_src, sizeof(DataType) * nrows * hidden_size, cudaMemcpyHostToDevice));

    tinycudallama::perChannelQuantizedKernelLauncher(d_dst, d_src, d_scale, hidden_size, nrows);

    float h_scale[nrows];
    int8_t *h_dst = new int8_t[nrows * hidden_size];
    CHECK_CUDA_ERROR(cudaMemcpy(h_scale, d_scale, sizeof(float) * nrows, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_dst, d_dst, sizeof(int8_t) * nrows * hidden_size, cudaMemcpyDeviceToHost));

    printVecInVec(h_scale, 1, nrows, 1, nrows, "scale");
    printVecInVec(h_dst, nrows, hidden_size, nrows, hidden_size, "int8_dst");

    CHECK_CUDA_ERROR(cudaFree(d_src));
    CHECK_CUDA_ERROR(cudaFree(d_scale));
    CHECK_CUDA_ERROR(cudaFree(d_dst));
    delete[] h_src;
    delete[] h_dst;
}

void testQKRoteEmbeddingQuantizedTranspose()
{
    const int batch_size = 4;
    const int seq_len = 4;
    const int head_num = 32;
    const int size_per_head = 1024;
    const int num_elements = batch_size * seq_len * head_num * size_per_head;
    const int hidden_units = head_num * size_per_head;

    int32_t *q = new int32_t[num_elements];
    int32_t *k = new int32_t[num_elements];
    for (int i = 0; i < num_elements; ++i)
    {
        q[i] = (i % 2 == 0) ? i : (-1 * i);
        k[i] = (i % 2 == 0) ? i : (-3 * i + 2);
    }
    printVecInVec(q, batch_size * seq_len, hidden_units, 1, 10, "q");
    printVecInVec(k, batch_size * seq_len, hidden_units, 1, 10, "k");

    float *q_inp_scale = new float[batch_size * seq_len];
    float *k_inp_scale = new float[batch_size * seq_len];
    for (int i = 0; i < batch_size * seq_len; ++i)
    {
        q_inp_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.5f) + 10.0f);
        k_inp_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.5f) + 5.0f);
    }

    float *q_weight_scale = new float[hidden_units];
    float *k_weight_scale = new float[hidden_units];
    for (int i = 0; i < hidden_units; ++i)
    {
        q_weight_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.3f) + 10.0f);
        k_weight_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.3f) + 5.0f);
    }

    int mem_size = sizeof(int32_t) * num_elements * 2 + sizeof(float) * (batch_size * seq_len * 2 + batch_size * seq_len * head_num * 2 + hidden_units * 2) +
                   sizeof(int8_t) * num_elements * 2 + sizeof(float) * seq_len * size_per_head;

    int32_t *d_q;
    int32_t *d_k;
    float *d_q_inp_scale;
    float *d_k_inp_scale;
    float *d_q_out_scale;
    float *d_k_out_scale;
    float *d_q_weight_scale;
    float *d_k_weight_scale;
    int8_t *d_q_out;
    int8_t *d_k_out;
    float *d_freq_cis;
    device_malloc(&d_q, mem_size);
    d_k = (int32_t *)(d_q + num_elements);
    d_q_inp_scale = (float *)(d_k + num_elements);
    d_k_inp_scale = (float *)(d_q_inp_scale + batch_size * seq_len);
    d_q_out_scale = (float *)(d_k_inp_scale + batch_size * seq_len);
    d_k_out_scale = (float *)(d_q_out_scale + batch_size * seq_len * head_num);
    d_q_weight_scale = (float *)(d_k_out_scale + batch_size * seq_len * head_num);
    d_k_weight_scale = (float *)(d_q_weight_scale + hidden_units);
    d_q_out = (int8_t *)(d_k_weight_scale + hidden_units);
    d_k_out = (int8_t *)(d_q_out + num_elements);
    d_freq_cis = (float *)(d_k_out + num_elements);

    float *h_freq_cis = new float[seq_len * size_per_head];
    float *h_q_out_scale = new float[batch_size * seq_len * head_num];
    float *h_k_out_scale = new float[batch_size * seq_len * head_num];
    int8_t *h_q_out = new int8_t[num_elements];
    int8_t *h_k_out = new int8_t[num_elements];

    CHECK_CUDA_ERROR(cudaMemcpy(d_q, q, sizeof(int32_t) * num_elements, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k, k, sizeof(int32_t) * num_elements, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_q_inp_scale, q_inp_scale, sizeof(float) * batch_size * seq_len, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k_inp_scale, k_inp_scale, sizeof(float) * batch_size * seq_len, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_q_weight_scale, q_weight_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k_weight_scale, k_weight_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));

    tinycudallama::launchPrecomputeFreqsCis(d_freq_cis, size_per_head, seq_len);
    CHECK_CUDA_ERROR(cudaMemcpy(h_freq_cis, d_freq_cis, sizeof(float) * seq_len * size_per_head, cudaMemcpyDeviceToHost));
    printVecInVec(h_freq_cis, seq_len, size_per_head, seq_len, 10, "freq_cis");

    CHECK_CUDA_ERROR(cudaMemcpy(q_inp_scale, d_q_inp_scale, sizeof(float) * batch_size * seq_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(q_inp_scale, d_q_inp_scale, sizeof(float) * batch_size * seq_len, cudaMemcpyDeviceToHost));
    printVecInVec(q_inp_scale, 1, batch_size * seq_len, 1, batch_size * seq_len, "q_inp_scale");
    printVecInVec(k_inp_scale, 1, batch_size * seq_len, 1, batch_size * seq_len, "k_inp_scale");

    CHECK_CUDA_ERROR(cudaMemcpy(q_weight_scale, d_q_weight_scale, sizeof(float) * hidden_units, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(k_weight_scale, d_k_weight_scale, sizeof(float) * hidden_units, cudaMemcpyDeviceToHost));
    printVecInVec(q_weight_scale, 1, hidden_units, 1, 10, "q_weight_scale");
    printVecInVec(k_weight_scale, 1, hidden_units, 1, 10, "k_weight_scale");

    tinycudallama::launchQKRoteEmbeddingQuantizedTranspose(d_q_out, d_k_out, d_q, d_k, d_q_inp_scale, d_k_inp_scale,
                                                           d_q_weight_scale, d_k_weight_scale, d_q_out_scale, d_k_out_scale,
                                                           d_freq_cis, batch_size, seq_len, head_num, size_per_head);

    CHECK_CUDA_ERROR(cudaMemcpy(h_q_out_scale, d_q_out_scale, sizeof(float) * batch_size * seq_len * head_num, cudaMemcpyDeviceToHost));
    printVecInVec(h_q_out_scale, 1, batch_size * seq_len * head_num, 1, 32, "h_q_out_scale");
    CHECK_CUDA_ERROR(cudaMemcpy(h_k_out_scale, d_k_out_scale, sizeof(float) * batch_size * seq_len * head_num, cudaMemcpyDeviceToHost));
    printVecInVec(h_k_out_scale, 1, batch_size * seq_len * head_num, 1, 32, "h_k_out_scale");

    CHECK_CUDA_ERROR(cudaMemcpy(h_q_out, d_q_out, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_q_out, 1, num_elements, 1, 128, "h_q_out");
    CHECK_CUDA_ERROR(cudaMemcpy(h_k_out, d_k_out, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_k_out, 1, num_elements, 1, 128, "h_k_out");

    delete[] q;
    delete[] k;
    delete[] q_inp_scale;
    delete[] k_inp_scale;
    delete[] q_weight_scale;
    delete[] k_weight_scale;
    delete[] h_q_out;
    delete[] h_k_out;
    delete[] h_freq_cis;
    delete[] h_k_out_scale;
    delete[] h_q_out_scale;

    CHECK_CUDA_ERROR(cudaFree(d_q));
}

void testStorecache()
{
    using DataType = int8_t;
    const int batch_size = 2;
    const int max_seq_len = 5;
    const int seq_len = 2;
    const int start_pos = 1;
    const int head_num = 32;
    const int size_per_head = 128;
    const int cache_size = batch_size * max_seq_len * head_num * size_per_head;
    const int num_elements = batch_size * seq_len * head_num * size_per_head;

    DataType *h_k_cache = new DataType[cache_size * 2]{0};
    DataType *h_v_cache = h_k_cache + cache_size;
    DataType *h_k_inp = new DataType[num_elements * 2];
    DataType *h_v_inp = h_k_inp + num_elements;
    for (int i = 0; i < num_elements * 2; ++i)
        h_k_inp[i] = (i % 128);

    DataType *d_v_inp;
    DataType *d_k_inp;
    DataType *d_v_cache;
    DataType *d_k_cache;

    device_malloc(&d_k_inp, sizeof(DataType) * num_elements * 2);
    d_v_inp = d_k_inp + num_elements;
    device_malloc(&d_k_cache, sizeof(DataType) * cache_size * 2);
    d_v_cache = d_k_cache + cache_size;

    CHECK_CUDA_ERROR(cudaMemcpy(d_k_inp, h_k_inp, sizeof(DataType) * num_elements * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k_cache, h_k_cache, sizeof(DataType) * cache_size * 2, cudaMemcpyHostToDevice));

    tinycudallama::launchINT8StoreKVcacheKernel(d_k_cache, d_v_cache, d_k_inp, d_v_inp, start_pos, seq_len, batch_size, head_num, max_seq_len, size_per_head);

    CHECK_CUDA_ERROR(cudaMemcpy(h_k_cache, d_k_cache, sizeof(DataType) * cache_size * 2, cudaMemcpyDeviceToHost));

    printVecInVec(h_k_cache, batch_size * head_num * max_seq_len, size_per_head, max_seq_len, size_per_head, "h_cache");
    printVecInVec(h_v_cache, batch_size * head_num * max_seq_len, size_per_head, max_seq_len, size_per_head, "v_cache");

    CHECK_CUDA_ERROR(cudaFree(d_k_cache));
    CHECK_CUDA_ERROR(cudaFree(d_k_inp));
    delete[] h_k_inp;
    delete[] h_k_cache;
}


void testSoftmax()
{
    const int batch_size = 4;
    const int seq_len_q = 100;
    const int head_num = 32;
    const int seq_len_k = 100;
    const int max_seq_len = 2048;
    const int num_elements = batch_size * seq_len_q * head_num * seq_len_k;

    int32_t *qk = new int32_t[num_elements];
    for (int i = 0; i < num_elements; ++i)
    {
        qk[i] = (i % 2 == 0) ? i : (-1 * i);
    }
    printVecInVec(qk, batch_size * seq_len_q * head_num, seq_len_k, 2, 16, "qk");

    float *q_inp_scale = new float[batch_size * head_num * seq_len_q];
    float *k_inp_scale = new float[batch_size * head_num * seq_len_k];
    for (int i = 0; i < batch_size * head_num * seq_len_q; ++i)
    {
        q_inp_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.5f) + 10.0f);
    }
    for (int i = 0; i < batch_size * head_num * seq_len_k; ++i)
    {
        k_inp_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.5f) + 5.0f);
    }

    printVecInVec(q_inp_scale, batch_size * head_num, seq_len_q, 1, 10, "q_inp_scale");
    printVecInVec(k_inp_scale, batch_size * head_num, seq_len_k, 1, 10, "k_inp_scale");

    float *attn_mask = new float[max_seq_len * max_seq_len];
    for (int i=0; i<max_seq_len; ++i) {
        for (int j=0; j<max_seq_len; ++j) {
            if (j > i) attn_mask[i * max_seq_len + j] = -1e5f;
            else attn_mask[i * max_seq_len + j] = 0.0f;
        }
    }
    printVecInVec(attn_mask, max_seq_len, max_seq_len, 10, 10, "attn_mask");

    int mem_size = sizeof(int32_t) * num_elements + 
                    sizeof(float) * (batch_size * head_num * seq_len_q) + 
                    sizeof(float) * (batch_size * head_num * seq_len_k) +
                    sizeof(float) * (max_seq_len * max_seq_len) + 
                    sizeof(float) * (batch_size * head_num * seq_len_q) +
                    sizeof(int8_t) * num_elements;

    int32_t *d_qk;
    float *d_q_inp_scale;
    float *d_k_inp_scale;
    float *d_attn_mask;
    float *d_score_scale;
    int8_t *d_score;

    device_malloc(&d_qk, mem_size);
    d_q_inp_scale = (float *)(d_qk + num_elements);
    d_k_inp_scale = (float *)(d_q_inp_scale + batch_size * head_num * seq_len_q);
    d_attn_mask = (float *)(d_k_inp_scale + batch_size * head_num * seq_len_k);
    d_score_scale = (float *)(d_attn_mask + max_seq_len * max_seq_len);
    d_score = (int8_t *)(d_score_scale + batch_size * head_num * seq_len_q);

    float *h_score_scale = new float[batch_size * head_num * seq_len_q];
    int8_t *h_score = new int8_t[num_elements];

    CHECK_CUDA_ERROR(cudaMemcpy(d_qk, qk, sizeof(int32_t) * num_elements, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_q_inp_scale, q_inp_scale, sizeof(float) * batch_size * head_num * seq_len_q, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k_inp_scale, k_inp_scale, sizeof(float) * batch_size * head_num * seq_len_k, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_attn_mask, attn_mask, sizeof(float) * max_seq_len * max_seq_len, cudaMemcpyHostToDevice));

    tinycudallama::launchBlockSoftmaxKernel(d_score, d_qk, d_attn_mask, d_q_inp_scale, d_k_inp_scale, d_score_scale, rsqrtf((float)seq_len_k), 
        batch_size, head_num, seq_len_q, seq_len_k, max_seq_len);

    CHECK_CUDA_ERROR(cudaMemcpy(h_score_scale, d_score_scale, sizeof(float) * batch_size * head_num * seq_len_q, cudaMemcpyDeviceToHost));
    printVecInVec(h_score_scale, batch_size * head_num, seq_len_q, 1, 10, "h_score_scale");
    CHECK_CUDA_ERROR(cudaMemcpy(h_score, d_score, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_score, batch_size * head_num * seq_len_q, seq_len_k, 10, 10, "h_score");

    delete[] qk;
    delete[] q_inp_scale;
    delete[] k_inp_scale;
    delete[] attn_mask;
    delete[] h_score;
    delete[] h_score_scale;

    CHECK_CUDA_ERROR(cudaFree(d_qk));
}



void testDequantizedVTransposeQuantized()
{
    const int batch_size = 4;
    const int seq_len = 4;
    const int head_num = 32;
    const int size_per_head = 128;
    const int num_elements = batch_size * seq_len * head_num * size_per_head;
    const int hidden_units = head_num * size_per_head;

    int32_t *v = new int32_t[num_elements];
    for (int i = 0; i < num_elements; ++i)
    {
        v[i] = (i % 2 == 0) ? i : (-1 * i);
    }
    printVecInVec(v, batch_size * seq_len * head_num, size_per_head, 2, 10, "v");

    float *v_inp_scale = new float[batch_size * seq_len];
    for (int i = 0; i < batch_size * seq_len; ++i)
    {
        v_inp_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.5f) + 10.0f);
    }

    float *v_weight_scale = new float[hidden_units];
    for (int i = 0; i < hidden_units; ++i)
    {
        v_weight_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.3f) + 10.0f);
    }

    int mem_size = sizeof(int32_t) * num_elements + sizeof(float) * (batch_size * seq_len + hidden_units) +
                   sizeof(int8_t) * num_elements + sizeof(float) * batch_size * head_num * size_per_head + sizeof(float) * num_elements;

    int32_t *d_v;
    float *d_v_inp_scale;
    float *d_v_weight_scale;
    float *d_v_fp32_buf;
    int8_t *d_v_out;
    float *d_v_out_scale;
    device_malloc(&d_v, mem_size);
    d_v_inp_scale = (float *)(d_v + num_elements);
    d_v_weight_scale = (float *)(d_v_inp_scale + batch_size * seq_len);
    d_v_out = (int8_t *)(d_v_weight_scale + hidden_units);
    d_v_out_scale = (float *)(d_v_out + num_elements);
    d_v_fp32_buf = (float *)(d_v_out_scale + batch_size * head_num * size_per_head);

    float *h_v_out_scale = new float[batch_size * size_per_head * head_num];
    float *h_v_fp32_buf = new float[num_elements];
    int8_t *h_v_out = new int8_t[num_elements];

    CHECK_CUDA_ERROR(cudaMemcpy(d_v, v, sizeof(int32_t) * num_elements, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_v_inp_scale, v_inp_scale, sizeof(float) * batch_size * seq_len, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_v_weight_scale, v_weight_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    
    tinycudallama::launchDequantizedVTransposeKernel(d_v_fp32_buf, d_v, d_v_inp_scale, d_v_weight_scale, batch_size, seq_len, head_num, size_per_head);
    tinycudallama::launchBlockVQuantizedKernel(d_v_out, d_v_fp32_buf, d_v_out_scale, batch_size, seq_len, head_num, size_per_head);

    CHECK_CUDA_ERROR(cudaMemcpy(v_inp_scale, d_v_inp_scale, sizeof(float) * batch_size * seq_len, cudaMemcpyDeviceToHost));
    printVecInVec(v_inp_scale, 1, batch_size * seq_len, 1, batch_size * seq_len, "v_inp_scale");

    CHECK_CUDA_ERROR(cudaMemcpy(v_weight_scale, d_v_weight_scale, sizeof(float) * hidden_units, cudaMemcpyDeviceToHost));
    printVecInVec(v_weight_scale, head_num, size_per_head, 8, 10, "v_weight_scale");

    CHECK_CUDA_ERROR(cudaMemcpy(h_v_fp32_buf, d_v_fp32_buf, sizeof(float) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_v_fp32_buf, batch_size * head_num * seq_len, size_per_head, 10, 10, "h_v_fp32_buf");

    CHECK_CUDA_ERROR(cudaMemcpy(h_v_out_scale, d_v_out_scale, sizeof(float) * batch_size * size_per_head * head_num, cudaMemcpyDeviceToHost));
    printVecInVec(h_v_out_scale, batch_size * head_num, size_per_head, 5, 10, "h_v_out_scale");

    CHECK_CUDA_ERROR(cudaMemcpy(h_v_out, d_v_out, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_v_out, batch_size * head_num * seq_len, size_per_head, 10, 10, "h_v_out");

    delete[] v;
    delete[] v_inp_scale;
    delete[] v_weight_scale;
    delete[] h_v_out;
    delete[] h_v_out_scale;
    delete[] h_v_fp32_buf;

    CHECK_CUDA_ERROR(cudaFree(d_v));
}

void testVQuantized()
{
    const int batch_size = 4;
    const int seq_len = 4;
    const int head_num = 32;
    const int size_per_head = 128;
    const int num_elements = batch_size * head_num * seq_len * size_per_head;

    float *v = new float[num_elements];
    for (int i = 0; i < num_elements; ++i)
    {
        v[i] = (i % 2 == 0) ? i : (-1 * i);
    }
    printVecInVec(v, batch_size * head_num * seq_len, size_per_head, 2, 10, "v");

    float *d_v;
    float *d_v_out_scale;
    int8_t *d_v_out;
    const int mem_size = sizeof(float) * num_elements + sizeof(float) * batch_size * head_num * size_per_head + sizeof(int8_t) * num_elements;

    device_malloc(&d_v, mem_size);
    d_v_out_scale = (float *)(d_v + num_elements);
    d_v_out = (int8_t *)(d_v_out_scale + batch_size * head_num * size_per_head);

    CHECK_CUDA_ERROR(cudaMemcpy(d_v, v, sizeof(float) * num_elements, cudaMemcpyHostToDevice));

    tinycudallama::launchBlockVQuantizedKernel(d_v_out, d_v, d_v_out_scale, batch_size, seq_len, head_num, size_per_head);

    float *h_v_out_scale = new float[batch_size * head_num * size_per_head];
    int8_t *h_v_out = new int8_t[num_elements];

    CHECK_CUDA_ERROR(cudaMemcpy(h_v_out_scale, d_v_out_scale, sizeof(float) * batch_size * size_per_head * head_num, cudaMemcpyDeviceToHost));
    printVecInVec(h_v_out_scale, batch_size * head_num, size_per_head, 5, 10, "h_v_out_scale");

    CHECK_CUDA_ERROR(cudaMemcpy(h_v_out, d_v_out, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_v_out, batch_size * seq_len * head_num, size_per_head, 10, 10, "h_v_out");

    delete[] v;
    delete[] h_v_out;
    delete[] h_v_out_scale;

    CHECK_CUDA_ERROR(cudaFree(d_v));
}

void testDequantizedAttnQuantizedTranspose()
{
    const int batch_size = 4;
    const int seq_len = 4;
    const int head_num = 32;
    const int size_per_head = 128;
    const int num_elements = batch_size * head_num * seq_len * size_per_head;

    int32_t *h_attn = new int32_t[num_elements];
    for (int i = 0; i < num_elements; ++i)
    {
        h_attn[i] = (i % 2 == 0) ? i : (-2 * i + 100);
    }
    printVecInVec(h_attn, batch_size * head_num * seq_len, size_per_head, 2, 10, "h_attn");

    float *h_score_scale = new float[batch_size * head_num * seq_len];
    for (int i = 0; i < batch_size * head_num * seq_len; ++i)
    {
        h_score_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.5f) + 10.0f);
    }

    float *h_v_scale = new float[batch_size * head_num * size_per_head];
    for (int i = 0; i < batch_size * head_num * size_per_head; ++i)
    {
        h_v_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.3f) + 10.0f);
    }

    int32_t *d_attn;
    float *d_score_scale;
    float *d_v_scale;
    float *d_attn_out_scale;
    int8_t *d_attn_out;

    int mem_size = sizeof(int32_t) * num_elements + sizeof(float) * batch_size * head_num * seq_len + 
        sizeof(float) * batch_size * head_num * size_per_head + sizeof(float) * batch_size * seq_len + 
        sizeof(int8_t) * num_elements;
    
    device_malloc(&d_attn, mem_size);
    d_score_scale = (float *)(d_attn + num_elements);
    d_v_scale = (float *)(d_score_scale + batch_size * head_num * seq_len);
    d_attn_out_scale = (float *)(d_v_scale + batch_size * head_num * size_per_head);
    d_attn_out = (int8_t *)(d_attn_out_scale + batch_size * seq_len);

    CHECK_CUDA_ERROR(cudaMemcpy(d_attn, h_attn, sizeof(int32_t) * num_elements, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_score_scale, h_score_scale, sizeof(float) * batch_size * head_num * seq_len, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_v_scale, h_v_scale, sizeof(float) * batch_size * head_num * size_per_head, cudaMemcpyHostToDevice));

    tinycudallama::launchDequantizedAttnQuantizedTransposeKernel(d_attn_out, d_attn, d_score_scale, d_v_scale, d_attn_out_scale, 
        batch_size, head_num, seq_len, size_per_head);

    float *h_attn_out_scale = new float[batch_size * seq_len];
    int8_t *h_attn_out = new int8_t[num_elements];

    CHECK_CUDA_ERROR(cudaMemcpy(h_attn_out_scale, d_attn_out_scale, sizeof(float) * batch_size * seq_len, cudaMemcpyDeviceToHost));
    printVecInVec(h_attn_out_scale, batch_size, seq_len, batch_size, seq_len, "h_attn_out_scale");

    CHECK_CUDA_ERROR(cudaMemcpy(h_attn_out, d_attn_out, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_attn_out, batch_size * seq_len, head_num * size_per_head, 10, 10, "h_attn_out");

    delete[] h_attn;
    delete[] h_score_scale;
    delete[] h_v_scale;
    delete[] h_attn_out;
    delete[] h_attn_out_scale;

    CHECK_CUDA_ERROR(cudaFree(d_attn));
}


void testDequantizedResidualResNormQuantized()
{
    const int rows = 16;
    const int hidden_units = 4096;
    const int num_elements = rows * hidden_units;

    int32_t *h_attn = new int32_t[num_elements];
    for (int i = 0; i < num_elements; ++i)
    {
        h_attn[i] = (i % 2 == 0) ? (i%7000) : (-2 * (i%6000) + 100);
    }
    printVecInVec(h_attn, rows, hidden_units, 2, 10, "h_attn");

    float *h_from_tensor = new float[num_elements];
    for (int i = 0; i < num_elements; ++i)
    {
        h_from_tensor[i] = (i % 2 == 0) ? (i%8000) : (-3.5 * (i%7000) + 100);
    }
    printVecInVec(h_from_tensor, rows, hidden_units, 2, 10, "h_from_tensor");

    float *h_attn_scale = new float[rows];
    for (int i = 0; i < rows; ++i)
    {
        h_attn_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.5f) + 10.0f);
    }

    float *h_weight_scale = new float[hidden_units];
    for (int i = 0; i < hidden_units; ++i)
    {
        h_weight_scale[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.3f) + 10.0f);
    }

    float *h_gamma = new float[hidden_units];
    for (int i = 0; i < hidden_units; ++i)
    {
        h_gamma[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.4f) + 10.0f);
    }

    int32_t *d_attn;
    float *d_from_tensor;
    float *d_attn_scale;
    float *d_weight_scale;
    float *d_norm_scale;
    int8_t *d_norm_out;
    float *d_gamma;

    int mem_size = sizeof(int32_t) * num_elements + sizeof(float) * num_elements + 
        sizeof(float) * rows + sizeof(float) * hidden_units + sizeof(float) * rows + sizeof(float) * hidden_units +
        sizeof(int8_t) * num_elements;
    
    device_malloc(&d_attn, mem_size);
    d_from_tensor = (float *)(d_attn + num_elements);
    d_attn_scale = (float *)(d_from_tensor + num_elements);
    d_weight_scale = (float *)(d_attn_scale + rows);
    d_norm_scale = (float *)(d_weight_scale + hidden_units);
    d_gamma = (float *)(d_norm_scale + rows);
    d_norm_out = (int8_t *)(d_gamma + hidden_units);

    CHECK_CUDA_ERROR(cudaMemcpy(d_attn, h_attn, sizeof(int32_t) * num_elements, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_from_tensor, h_from_tensor, sizeof(float) * num_elements, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_attn_scale, h_attn_scale, sizeof(float) * rows, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weight_scale, h_weight_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_gamma, h_gamma, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    
    tinycudallama::launchDequantizedResidualResNormQuantized<float>(d_norm_out, d_from_tensor, d_attn, d_attn_scale, 
        d_weight_scale, d_gamma, d_norm_scale, 1e-5f, rows, hidden_units);

    float *h_norm_scale = new float[rows];
    int8_t *h_norm_out = new int8_t[num_elements];

    CHECK_CUDA_ERROR(cudaMemcpy(h_norm_scale, d_norm_scale, sizeof(float) * rows, cudaMemcpyDeviceToHost));
    printVecInVec(h_norm_scale, 1, rows, 1, rows, "h_norm_scale");

    CHECK_CUDA_ERROR(cudaMemcpy(h_norm_out, d_norm_out, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_norm_out, rows, hidden_units, 10, 10, "h_norm_out");

    half *d_from_tensor_half;
    half *d_gamma_half;
    device_malloc(&d_from_tensor_half, sizeof(half) * num_elements);
    device_malloc(&d_gamma_half, sizeof(half) * hidden_units);

    convertMatfloat2half<<<rows, 128>>>(d_from_tensor, d_from_tensor_half, num_elements);
    convertMatfloat2half<<<1, 128>>>(d_gamma, d_gamma_half, hidden_units);

    tinycudallama::launchDequantizedResidualResNormQuantized(d_norm_out, d_from_tensor_half, d_attn, d_attn_scale, 
        d_weight_scale, d_gamma_half, d_norm_scale, 1e-5f, rows, hidden_units);

    CHECK_CUDA_ERROR(cudaMemcpy(h_norm_scale, d_norm_scale, sizeof(float) * rows, cudaMemcpyDeviceToHost));
    printVecInVec(h_norm_scale, 1, rows, 1, rows, "h_norm_scale_half");

    CHECK_CUDA_ERROR(cudaMemcpy(h_norm_out, d_norm_out, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_norm_out, rows, hidden_units, 10, 10, "h_norm_out_half");

    delete[] h_attn;
    delete[] h_from_tensor;
    delete[] h_attn_scale;
    delete[] h_weight_scale;
    delete[] h_gamma;
    delete[] h_norm_out;
    delete[] h_norm_scale;

    CHECK_CUDA_ERROR(cudaFree(d_attn));
    CHECK_CUDA_ERROR(cudaFree(d_from_tensor_half));
    CHECK_CUDA_ERROR(cudaFree(d_gamma_half));
}

void testResNormQuantized()
{
    const int rows = 16;
    const int hidden_units = 4096;
    const int num_elements = rows * hidden_units;

    float *h_inp = new float[num_elements];
    for (int i = 0; i < num_elements; ++i)
    {
        h_inp[i] = (i % 2 == 0) ? i : (-2 * i + 100);
    }
    printVecInVec(h_inp, rows, hidden_units, 2, 10, "h_inp");

    float *h_gamma = new float[hidden_units];
    for (int i = 0; i < hidden_units; ++i)
    {
        h_gamma[i] = (i % 2 == 0) ? (i - 10) : (powf((float)i, 0.4f) + 10.0f);
    }

    float *d_inp;
    int8_t *d_out;
    float *d_gamma;
    float *d_norm_scale;
    int mem_size = sizeof(float) * num_elements + sizeof(float) * hidden_units + sizeof(float) * rows + sizeof(int8_t) * num_elements;
    device_malloc<float>(&d_inp, mem_size);
    d_gamma = (float *)(d_inp + num_elements);
    d_norm_scale = (float *)(d_gamma + hidden_units);
    d_out = (int8_t *)(d_norm_scale + rows);

    CHECK_CUDA_ERROR(cudaMemcpy(d_inp, h_inp, sizeof(float) * num_elements, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_gamma, h_gamma, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));

    half *d_inp_half;
    int8_t *d_out_half;
    half *d_gamma_half;
    float *d_norm_scale_half;
    mem_size = sizeof(half) * num_elements + sizeof(half) * hidden_units + sizeof(float) * rows + sizeof(int8_t) * num_elements;

    device_malloc(&d_inp_half, mem_size);
    d_gamma_half = (half *)(d_inp + num_elements);
    d_norm_scale_half = (float *)(d_gamma + hidden_units);
    d_out_half = (int8_t *)(d_norm_scale + rows);

    convertMatfloat2half<<<rows, 256>>>(d_inp, d_inp_half, num_elements);

    convertMatfloat2half<<<1, 256>>>(d_gamma, d_gamma_half, hidden_units);

    tinycudallama::launchResNormQuantizedKernel(d_out, d_inp, d_gamma, d_norm_scale, 1e-5f, rows, hidden_units);
    tinycudallama::launchResNormQuantizedKernel(d_out_half, d_inp_half, d_gamma_half, d_norm_scale_half, 1e-5f, rows, hidden_units);

    float *h_norm_scale = new float[rows];
    int8_t *h_norm_out = new int8_t[num_elements];

    CHECK_CUDA_ERROR(cudaMemcpy(h_norm_scale, d_norm_scale, sizeof(float) * rows, cudaMemcpyDeviceToHost));
    printVecInVec(h_norm_scale, 1, rows, 1, rows, "h_norm_scale");

    CHECK_CUDA_ERROR(cudaMemcpy(h_norm_out, d_out, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_norm_out, rows, hidden_units, 10, 10, "h_norm_out");

    float *h_norm_scale_half = new float[rows];
    int8_t *h_norm_out_half = new int8_t[num_elements];

    CHECK_CUDA_ERROR(cudaMemcpy(h_norm_scale_half, d_norm_scale_half, sizeof(float) * rows, cudaMemcpyDeviceToHost));
    printVecInVec(h_norm_scale_half, 1, rows, 1, rows, "h_norm_scale_half");

    CHECK_CUDA_ERROR(cudaMemcpy(h_norm_out_half, d_out_half, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_norm_out_half, rows, hidden_units, 10, 10, "h_norm_out_half");

    
    CHECK_CUDA_ERROR(cudaFree(d_inp));
    CHECK_CUDA_ERROR(cudaFree(d_inp_half));
    
    delete[] h_inp;
    delete[] h_gamma;
    delete[] h_norm_out;
    delete[] h_norm_scale;
    delete[] h_norm_out_half;
    delete[] h_norm_scale_half;

}

void testDequantizedSiluMultifyQuantized()
{
    const int rows = 16;
    const int hidden_units = 11008;
    const int num_elements = rows * hidden_units;

    int32_t *h_w1_ret = new int32_t[num_elements];
    for (int i = 0; i < num_elements; ++i)
    {
        h_w1_ret[i] = (i % 2 == 0) ? (i%70) : (-2 * (i%60) + 1);
    }
    printVecInVec(h_w1_ret, rows, hidden_units, 2, 10, "h_w1_ret");

    int32_t *h_w3_ret = new int32_t[num_elements];
    for (int i = 0; i < num_elements; ++i)
    {
        h_w3_ret[i] = (i % 2 == 0) ? (i%80) : (-3.5 * (i%70) + 1);
    }
    printVecInVec(h_w3_ret, rows, hidden_units, 2, 10, "h_w3_ret");

    float *h_norm_scale = new float[rows];
    for (int i = 0; i < rows; ++i)
    {
        h_norm_scale[i] = (i % 2 == 0) ? (0.2f * powf((float)i, 0.1f) + 0.1f) : (powf((float)i, 0.5f) + 0.1f);
    }
    printVecInVec(h_norm_scale, 1, rows, 1, rows, "h_norm_scale");

    float *h_w1_weight_scale = new float[hidden_units];
    for (int i = 0; i < hidden_units; ++i)
    {
        h_w1_weight_scale[i] = (i % 2 == 0) ? (0.3f * powf((float)i, 0.1f) + 0.1f) : (powf((float)i, 0.3f) + 0.1f);
    }
    printVecInVec(h_w1_weight_scale, 1, hidden_units, 1, 10, "h_w1_weight_scale");

    float *h_w3_weight_scale = new float[hidden_units];
    for (int i = 0; i < hidden_units; ++i)
    {
        h_w3_weight_scale[i] = (i % 2 == 0) ? (0.4f * powf((float)i, 0.1f) + 0.1f) : (powf((float)i, 0.4f) + 0.1f);
    }
    printVecInVec(h_w3_weight_scale, 1, hidden_units, 1, 10, "h_w3_weight_scale");

    int32_t *d_w1_ret;
    int32_t *d_w3_ret;
    float *d_norm_scale;
    float *d_w1_weight_scale;
    float *d_w3_weight_scale;
    float *d_out_scale;
    int8_t *d_out;

    int mem_size = sizeof(int32_t) * num_elements * 2 + sizeof(float) * (rows + hidden_units * 2 + rows) + sizeof(int8_t) * num_elements;
    device_malloc(&d_w1_ret, mem_size);
    d_w3_ret = (int32_t *)(d_w1_ret + num_elements);
    d_norm_scale = (float *)(d_w3_ret + num_elements);
    d_w1_weight_scale = (float *)(d_norm_scale + rows);
    d_w3_weight_scale = (float *)(d_w1_weight_scale + hidden_units);
    d_out_scale = (float *)(d_w3_weight_scale + hidden_units);
    d_out = (int8_t *)(d_out_scale + rows);

    CHECK_CUDA_ERROR(cudaMemcpy(d_w1_ret, h_w1_ret, sizeof(int32_t) * num_elements, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_w3_ret, h_w3_ret, sizeof(int32_t) * num_elements, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_norm_scale, h_norm_scale, sizeof(float) * rows, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_w1_weight_scale, h_w1_weight_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_w3_weight_scale, h_w3_weight_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));

    tinycudallama::launchDequantizedSiluMultifyQuantized(d_out, d_w1_ret, d_norm_scale, d_w1_weight_scale, d_w3_ret, d_w3_weight_scale, 
        d_out_scale, rows, hidden_units);

    float *h_out_scale = new float[rows];
    int8_t *h_out = new int8_t[num_elements];

    CHECK_CUDA_ERROR(cudaMemcpy(h_out_scale, d_out_scale, sizeof(float) * rows, cudaMemcpyDeviceToHost));
    printVecInVec(h_out_scale, 1, rows, 1, rows, "h_out_scale");

    CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, sizeof(int8_t) * num_elements, cudaMemcpyDeviceToHost));
    printVecInVec(h_out, rows, hidden_units, 10, 10, "h_out");

    delete[] h_w1_ret;
    delete[] h_w3_ret;
    delete[] h_norm_scale;
    delete[] h_w1_weight_scale;
    delete[] h_w3_weight_scale;
    delete[] h_out_scale;
    delete[] h_out;

    CHECK_CUDA_ERROR(cudaFree(d_w1_ret));
}



int main()
{
    // testResNorm();

    // testPrecomputeFreqsCis();

    // testEmbedding();

    // testPerChannelQuantized();

    // testQKRoteEmbeddingQuantizedTranspose();

    // testStorecache();

    // testSoftmax();

    // testDequantizedVTransposeQuantized();

    // testVQuantized();

    // testDequantizedAttnQuantizedTranspose();

    // testDequantizedResidualResNormQuantized();

    // testResNormQuantized();

    testDequantizedSiluMultifyQuantized();

    return 0;
}