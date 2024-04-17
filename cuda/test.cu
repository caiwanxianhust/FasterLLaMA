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
            printf("%g  ", (float)clusters[i * ncols + j]);
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

template <typename DataType>
void timingResNorm(DataType* output, const DataType* input, const DataType* gamma, const DataType eps, const int m, const int n, 
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
        tinycudallama::launchResNormKernel<DataType>(output, input, gamma, 1e-7f, m, n);
        break;
    case 1:
        tinycudallama::rms_norm_f32_cuda(input, output, n, m, gamma, 1e-7f);
    default:
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

    CHECK_CUDA_ERROR(cudaFree(d_in));
    delete [] h_out1;
}



int main()
{
    testResNorm();

    return 0;
}