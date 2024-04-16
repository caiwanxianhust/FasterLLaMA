#include "cuda_kernels.cuh"

#include <cstdio>
#include <cstdlib>

template <typename T>
void printVecInVec(const T *clusters, const int n, const int m, const int end_n, const int end_m, const char *str)
{
    printf("%s:\n[\n", str);
    for (int i = 0; i < end_n; ++i)
    {
        printf("[");
        for (int j = 0; j < end_m; ++j)
        {
            printf("%g  ", (float)clusters[i * m + j]);
        }
        printf("]\n");
    }
    printf("]\n");
}

void testResNorm()
{
    using DataType = float;
    const int m = 10000;
    const int n = 4096;
    
    DataType *h_input = new DataType[m * n * 3];
    DataType *h_out1 = h_input + m * n;
    DataType *h_out2 = h_out1 + m * n;

    std::srand(1234);
    for(int i = 0; i < m * n; i++)
    {
        h_input[i] = (DataType)((rand() % 1000) / 500.0f) - 1.0f;
    }

    DataType *d_in;
    DataType *d_out1;
    DataType *d_out2;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_in, sizeof(DataType) * (m * n * 3)));
    d_out1 = d_in + m * n;
    d_out2 = d_out1 + m * n;
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_input, sizeof(DataType) * (m * n), cudaMemcpyHostToDevice));

    

}



int main()
{

    return 0;
}