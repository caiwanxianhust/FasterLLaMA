#pragma once

#include "common.h"
#include <vector>
#include <cuda_runtime.h>

namespace tinycudallama
{

  /**
   * Pop current cuda device and set new device
   * i_device - device ID to set
   * o_device - device ID to pop
   * ret  - return code (the same as cudaError_t)
   */

  inline cudaError_t get_set_device(int i_device, int *o_device = NULL)
  {
    int current_dev_id = 0;
    cudaError_t err = cudaSuccess;

    if (o_device != NULL)
    {
      err = cudaGetDevice(&current_dev_id);
      if (err != cudaSuccess)
        return err;
      if (current_dev_id == i_device)
      {
        *o_device = i_device;
      }
      else
      {
        err = cudaSetDevice(i_device);
        if (err != cudaSuccess)
        {
          return err;
        }
        *o_device = current_dev_id;
      }
    }
    else
    {
      err = cudaSetDevice(i_device);
      if (err != cudaSuccess)
      {
        return err;
      }
    }

    return cudaSuccess;
  }

  class IAllocator
  {
  public:
    virtual void *malloc(size_t size, const bool is_set_zero = true) const = 0;
    virtual void free(void *ptr) const = 0;
  };

  template <AllocatorType AllocType_>
  class Allocator;

  template <>
  class Allocator<AllocatorType::CUDA> : public IAllocator
  {
    const int device_id_;

  public:
    Allocator(int device_id) : device_id_(device_id) {}

    void *malloc(size_t size, const bool is_set_zero = true) const
    {
      void *ptr = nullptr;
      int o_device = 0;
      CHECK_CUDA_ERROR(get_set_device(device_id_, &o_device));
      CHECK_CUDA_ERROR(cudaMalloc(&ptr, size));
      CHECK_CUDA_ERROR(get_set_device(o_device));
      return ptr;
    }

    void free(void *ptr) const
    {
      int o_device = 0;
      CHECK_CUDA_ERROR(get_set_device(device_id_, &o_device));
      CHECK_CUDA_ERROR(cudaFree(ptr));
      CHECK_CUDA_ERROR(get_set_device(o_device));
      return;
    }
  };

} // namespace tinycudallama
