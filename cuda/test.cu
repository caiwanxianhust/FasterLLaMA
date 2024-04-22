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
    // case 1:
    //     tinycudallama::rms_norm_f32_cuda(input, output, n, m, gamma, 1e-7f);
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
    const int batch_size = 2;
    const int max_seq_len = 5;
    const int seq_len = 2;
    const int start_pos = 1;
    const int head_num = 32;
    const int size_per_head = 128;
    const int cache_size = batch_size * max_seq_len * head_num * size_per_head;
    const int num_elements = batch_size * seq_len * head_num * size_per_head;

    float *h_k_cache = new float[cache_size * 2]{0.0f};
    float *h_v_cache = h_k_cache + cache_size;
    float *h_k_inp = new float[num_elements * 2];
    float *h_v_inp = h_k_inp + num_elements;
    for (int i = 0; i < num_elements * 2; ++i)
        h_k_inp[i] = i * 0.1f + 0.09f;

    float *d_v_inp;
    float *d_k_inp;
    float *d_v_cache;
    float *d_k_cache;

    device_malloc(&d_k_inp, sizeof(float) * num_elements * 2);
    d_v_inp = d_k_inp + num_elements;
    device_malloc(&d_k_cache, sizeof(float) * cache_size * 2);
    d_v_cache = d_k_cache + cache_size;

    CHECK_CUDA_ERROR(cudaMemcpy(d_k_inp, h_k_inp, sizeof(float) * num_elements * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_k_cache, h_k_cache, sizeof(float) * cache_size * 2, cudaMemcpyHostToDevice));

    tinycudallama::launchStoreKVcacheKernel(d_k_cache, d_v_cache, d_k_inp, d_v_inp, start_pos, seq_len, batch_size, head_num, max_seq_len, size_per_head);

    CHECK_CUDA_ERROR(cudaMemcpy(h_k_cache, d_k_cache, sizeof(float) * cache_size * 2, cudaMemcpyDeviceToHost));

    printVecInVec(h_k_cache, batch_size * head_num * max_seq_len, size_per_head, max_seq_len, size_per_head, "h_cache");
    printVecInVec(h_v_cache, batch_size * head_num * max_seq_len, size_per_head, max_seq_len, size_per_head, "v_cache");

    CHECK_CUDA_ERROR(cudaFree(d_k_cache));
    CHECK_CUDA_ERROR(cudaFree(d_k_inp));
    delete[] h_k_inp;
    delete[] h_k_cache;
}

int main()
{
    // testResNorm();

    // testPrecomputeFreqsCis();

    // testEmbedding();

    // testPerChannelQuantized();

    // testQKRoteEmbeddingQuantizedTranspose();

    testStorecache();

    return 0;
}