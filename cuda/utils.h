#pragma once

#include "common.h"
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

} // namespace tinycudallama
