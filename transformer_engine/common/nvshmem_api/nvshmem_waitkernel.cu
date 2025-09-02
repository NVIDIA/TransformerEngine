/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cuda_bf16.h>
#include <nvshmem.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "../util/logging.h"
#include "nvshmem_waitkernel.h"

__global__ void __launch_bounds__(1)
    wait_until_on_stream_and_reset(uint64_t* wait_flag, uint64_t wait_value,
                                   uint64_t signal_reset) {
  nvshmem_uint64_wait_until(wait_flag, NVSHMEM_CMP_EQ, wait_value);
  *wait_flag = signal_reset;
}
void nvshmem_wait_on_stream(uint64_t* sig_addr, WaitKind wait_kind, cudaStream_t stream) {
  uint64_t wait_value = 1;
  uint64_t signal_reset = 0;
  cudaStream_t cur_stream = stream;

  NVTE_CHECK(wait_kind >= WaitKind::KERNEL_WAIT && wait_kind <= WaitKind::STREAM_WAIT,
             "Invalid wait kind: ", static_cast<int>(wait_kind));

  switch (wait_kind) {
    case WaitKind::KERNEL_WAIT:
      wait_until_on_stream_and_reset<<<1, 1, 0, cur_stream>>>(sig_addr, wait_value, signal_reset);
      NVTE_CHECK_CUDA(cudaGetLastError());
      break;
    case WaitKind::NVSHMEM_WAIT:
      nvshmemx_uint64_wait_until_on_stream(sig_addr, NVSHMEM_CMP_EQ, wait_value, cur_stream);
      NVTE_CHECK_CUDA_DRIVER(cuStreamWriteValue64((CUstream)cur_stream, (CUdeviceptr)sig_addr,
                                                  (cuuint64_t)signal_reset,
                                                  CU_STREAM_WRITE_VALUE_DEFAULT));
      break;
    case WaitKind::STREAM_WAIT:
      NVTE_CHECK_CUDA_DRIVER(cuStreamWaitValue64((CUstream)cur_stream, (CUdeviceptr)sig_addr,
                                                 (cuuint64_t)wait_value, CU_STREAM_WAIT_VALUE_GEQ));
      NVTE_CHECK_CUDA_DRIVER(cuStreamWriteValue64((CUstream)cur_stream, (CUdeviceptr)sig_addr,
                                                  (cuuint64_t)signal_reset,
                                                  CU_STREAM_WRITE_VALUE_DEFAULT));
      break;
  }
}
