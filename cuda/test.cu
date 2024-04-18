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
    if (end_row >= nrows || end_col >= ncols) printf("invalid arguments!!!\nend_row >= nrows or end_col >= ncols\n");
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

template<typename T>
void device_malloc(T** ptr, int size)
{
    CHECK_CUDA_ERROR(cudaMalloc((void**)ptr, sizeof(T) * size));
    T* tmp = new T[size];
    for(int i = 0; i < size; i++) tmp[i] = (T)((float) rand() / (RAND_MAX + 1.0) * 0.02);
    CHECK_CUDA_ERROR(cudaMemcpy(*ptr, tmp, sizeof(T) * size, cudaMemcpyHostToDevice));
    delete [] tmp;
}

__global__ void convertMatfloat2half(const float *input, half *output, const int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i=offset; i<size; i+=gridDim.x * blockDim.x) {
        output[i] = __float2half(input[i]);
    }
}


template <typename DataType>
void timingResNorm(DataType* output, const DataType* input, const DataType* gamma, const float eps, const int m, const int n, 
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
    delete [] h_out_half;
    delete [] h_out1;
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
    delete [] h_freqsCis;
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

    delete [] h_embedding_table;
    delete [] h_from_tensor;
}

void testPerChannelQuantized()
{
    using DataType = float;
    const int nrows = 8;
    const int hidden_size = 32;
    DataType *h_src = new DataType[nrows * hidden_size];
    for (int i=0; i<nrows*hidden_size; ++i) {
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
    delete [] h_src;
    delete [] h_dst;
}


int main()
{
    // testResNorm();

    // testPrecomputeFreqsCis();

    // testEmbedding();

    testPerChannelQuantized();

    return 0;
}