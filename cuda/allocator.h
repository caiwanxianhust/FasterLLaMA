#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include "common.h"
#include <vector>
#include <cuda_runtime.h>

namespace tinycudallama
{

class IAllocator
{
public:
  virtual void *malloc(size_t size, const bool is_set_zero=true) const = 0;
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

  void *malloc(size_t size, const bool is_set_zero=true) const
  {
    void *ptr = nullptr;
    int o_device = 0;
    check_cuda_error(get_set_device(device_id_, &o_device));
    check_cuda_error(cudaMalloc(&ptr, size));
    check_cuda_error(get_set_device(o_device));
    return ptr;
  }

  void free(void *ptr) const
  {
    int o_device = 0;
    check_cuda_error(get_set_device(device_id_, &o_device));
    check_cuda_error(cudaFree(ptr));
    check_cuda_error(get_set_device(o_device));
    return;
  }
};

} //namespace tinycudallama

#endif