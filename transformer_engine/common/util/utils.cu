/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/utils.h>

#include "../common.h"
#include "../util/logging.h"

namespace {

constexpr int64_t kMaxKernelAddresses = 256;

struct HostPointersArgs {
  uint64_t ptrs[kMaxKernelAddresses];
};

__global__ void write_pointers_kernel(HostPointersArgs args, uint64_t *out, int64_t count,
                                      int64_t offset) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[offset + idx] = args.ptrs[idx];
  }
}

}  // namespace

void nvte_convert_pointers_to_tensor(const uint64_t *host_ptrs, NVTETensor output, int64_t count,
                                     cudaStream_t stream) {
  NVTE_API_CALL(nvte_convert_pointers_to_tensor);
  using namespace transformer_engine;
  Tensor *out_tensor = convertNVTETensorCheck(output);
  uint64_t *out_ptr = static_cast<uint64_t *>(out_tensor->data.dptr);
  NVTE_CHECK(out_ptr != nullptr, "Output tensor data pointer is null.");

  int64_t offset = 0;
  while (offset < count) {
    const int64_t chunk = std::min(kMaxKernelAddresses, count - offset);
    HostPointersArgs args{};
    for (int64_t i = 0; i < chunk; ++i) {
      args.ptrs[i] = host_ptrs[offset + i];
    }
    constexpr int threads = kMaxKernelAddresses;
    write_pointers_kernel<<<1, threads, 0, stream>>>(args, out_ptr, chunk, offset);
    NVTE_CHECK_CUDA(cudaGetLastError());
    offset += chunk;
  }
}
