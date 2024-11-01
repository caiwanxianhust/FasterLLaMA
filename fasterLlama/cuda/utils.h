#pragma once
#include <stdio.h>
#include <iostream>
#include <cublas_v2.h>
#include <cstdlib>

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

#define PRINT_FUNC_NAME_()                                              \
    do                                                                  \
    {                                                                   \
        std::cout << "[FL][CALL] " << __FUNCTION__ << " " << std::endl; \
    } while (0)

#define CHECK_CUDA_ERROR(call)                             \
    do                                                     \
    {                                                      \
        const cudaError_t errorCode = call;                \
        if (errorCode != cudaSuccess)                      \
        {                                                  \
            printf("CUDA Error:\n");                       \
            printf("    File:   %s\n", __FILE__);          \
            printf("    Line:   %d\n", __LINE__);          \
            printf("    Error code:     %d\n", errorCode); \
            printf("    Error text:     %s\n",             \
                   cudaGetErrorString(errorCode));         \
            exit(1);                                       \
        }                                                  \
    } while (0)

#define CHECK_CUBLAS_STATUS(call)                                 \
    do                                                            \
    {                                                             \
        const cublasStatus_t statusCode = call;                   \
        if (statusCode != CUBLAS_STATUS_SUCCESS)                  \
        {                                                         \
            printf("CUDA Error:\n");                              \
            printf("    File:   %s\n", __FILE__);                 \
            printf("    Line:   %d\n", __LINE__);                 \
            printf("    Status code:     %d\n", statusCode);      \
            printf("    Error text:     %s\n",                    \
                   _cudaGetErrorEnum(statusCode)); \
            exit(1);                                              \
        }                                                         \
    } while (0)
