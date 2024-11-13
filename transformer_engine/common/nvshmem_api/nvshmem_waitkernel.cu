/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <nvshmem.h>
#include <cuda.h>

#include <cstdio>
#include <cuda_bf16.h>
#include <string>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include "nvshmem_waitkernel.h"

namespace transformer_engine{
__global__ void __launch_bounds__(1) wait_until_on_stream_and_reset(uint64_t* wait_flag, uint64_t wait_value, uint64_t signal_reset) {
    nvshmem_uint64_wait_until(wait_flag, NVSHMEM_CMP_EQ, wait_value);
    *wait_flag = signal_reset;
}
void nvshmem_wait_on_stream(uint64_t* sig_addr, int wait_kind, cudaStream_t stream){
  uint64_t wait_value = 1;
  uint64_t signal_reset = 0;
  cudaStream_t cur_stream = stream;

  assert(wait_kind<=2);

  if (wait_kind==0){
    wait_until_on_stream_and_reset<<<1, 1, 0, cur_stream>>>(sig_addr, wait_value, signal_reset);
  }
  else if(wait_kind==1){
    nvshmemx_uint64_wait_until_on_stream(sig_addr, NVSHMEM_CMP_EQ, wait_value, cur_stream);
    cuStreamWriteValue64((CUstream)cur_stream, (CUdeviceptr)sig_addr, (cuuint64_t)signal_reset, CU_STREAM_WRITE_VALUE_DEFAULT);
  }
  else if(wait_kind==2){
    cuStreamWaitValue64((CUstream)cur_stream, (CUdeviceptr)sig_addr, (cuuint64_t)wait_value, CU_STREAM_WAIT_VALUE_GEQ);
    // Reset local flag to 0
    cuStreamWriteValue64((CUstream)cur_stream, (CUdeviceptr)sig_addr, (cuuint64_t)signal_reset, CU_STREAM_WRITE_VALUE_DEFAULT);
  }
}

}