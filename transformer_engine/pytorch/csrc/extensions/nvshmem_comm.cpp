/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>

#include "../stable_common.h"

#ifdef NVTE_ENABLE_NVSHMEM
// Include only host headers — this is compiled as C++, not CUDA
#define NVSHMEM_HOSTLIB_ONLY
#include <nvshmem.h>
#include <nvshmemx.h>
#undef NVSHMEM_HOSTLIB_ONLY
#endif

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

// ============================================================================
// NVSHMEM create tensor in shared memory
// ============================================================================

Tensor nvshmem_create_tensor(int64_t num_elements, int64_t scalar_type_int, int64_t device_idx) {
#ifdef NVTE_ENABLE_NVSHMEM
  auto dtype = static_cast<ScalarType>(scalar_type_int);
  size_t elem_size = 0;
  switch (dtype) {
    case ScalarType::Float:
      elem_size = 4;
      break;
    case ScalarType::Half:
      elem_size = 2;
      break;
    case ScalarType::BFloat16:
      elem_size = 2;
      break;
    case ScalarType::Byte:
      elem_size = 1;
      break;
    case ScalarType::Long:
      elem_size = 8;
      break;
    case ScalarType::Int:
      elem_size = 4;
      break;
    case ScalarType::Double:
      elem_size = 8;
      break;
    default:
      STD_TORCH_CHECK(false, "Unsupported dtype for nvshmem_create_tensor");
  }
  size_t total_bytes = static_cast<size_t>(num_elements) * elem_size;
  void *ptr = nvshmem_malloc(total_bytes);
  STD_TORCH_CHECK(ptr != nullptr, "nvshmem_malloc failed for ", total_bytes, " bytes");

  // Wrap with from_blob and nvshmem_free as deleter
  std::vector<int64_t> shape = {num_elements};
  std::vector<int64_t> strides = {1};
  auto device = torch::stable::Device(torch::stable::DeviceType::CUDA, device_idx);
  return torch::stable::from_blob(ptr, shape, strides, device, dtype, nvshmem_free);
#else
  STD_TORCH_CHECK(false, "NVSHMEM not available. Build with NVTE_ENABLE_NVSHMEM=1.");
#endif
}

// ============================================================================
// NVSHMEM wait on current CUDA stream
//
// Uses CUDA driver API (cuStreamWaitValue64/cuStreamWriteValue64) which
// doesn't require NVSHMEM device code. Supports stream_wait and nvshmem_wait
// modes. kernel_wait mode requires device code and is not supported.
// ============================================================================

void nvshmem_wait_on_current_stream(Tensor signal, int64_t wait_kind_int) {
#ifdef NVTE_ENABLE_NVSHMEM
  uint64_t *sig_addr = reinterpret_cast<uint64_t *>(signal.data_ptr());
  auto stream = getCurrentCUDAStreamRaw(signal.get_device_index());
  uint64_t wait_value = 1;
  uint64_t signal_reset = 0;

  // WaitKind: 0=KERNEL_WAIT, 1=NVSHMEM_WAIT, 2=STREAM_WAIT
  switch (wait_kind_int) {
    case 0:  // KERNEL_WAIT — requires device code
      STD_TORCH_CHECK(false,
                      "KERNEL_WAIT mode requires NVSHMEM device code. "
                      "Use 'stream' or 'nvshmem' wait_kind instead.");
      break;
    case 1:  // NVSHMEM_WAIT — use nvshmemx host API + driver reset
      nvshmemx_uint64_wait_until_on_stream(sig_addr, NVSHMEM_CMP_EQ, wait_value, stream);
      {
        CUresult res = cuStreamWriteValue64(
            reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(sig_addr),
            static_cast<cuuint64_t>(signal_reset), CU_STREAM_WRITE_VALUE_DEFAULT);
        STD_TORCH_CHECK(res == CUDA_SUCCESS, "cuStreamWriteValue64 failed");
      }
      break;
    case 2:  // STREAM_WAIT — pure CUDA driver API
    default: {
      CUresult res = cuStreamWaitValue64(
          reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(sig_addr),
          static_cast<cuuint64_t>(wait_value), CU_STREAM_WAIT_VALUE_GEQ);
      STD_TORCH_CHECK(res == CUDA_SUCCESS, "cuStreamWaitValue64 failed");
      res = cuStreamWriteValue64(
          reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(sig_addr),
          static_cast<cuuint64_t>(signal_reset), CU_STREAM_WRITE_VALUE_DEFAULT);
      STD_TORCH_CHECK(res == CUDA_SUCCESS, "cuStreamWriteValue64 failed");
    } break;
  }
#else
  STD_TORCH_CHECK(false, "NVSHMEM not available. Build with NVTE_ENABLE_NVSHMEM=1.");
#endif
}

// ============================================================================
// NVSHMEM send with signal on current CUDA stream
// ============================================================================

void nvshmem_send_on_current_stream(Tensor src, Tensor dst, int64_t peer, Tensor signal) {
#ifdef NVTE_ENABLE_NVSHMEM
  void *src_ptr = src.data_ptr();
  void *dst_ptr = dst.data_ptr();
  uint64_t *sig_addr = reinterpret_cast<uint64_t *>(signal.data_ptr());
  size_t nelement = static_cast<size_t>(src.numel()) * src.element_size();
  uint64_t sigval = 1;
  auto stream = getCurrentCUDAStreamRaw(src.get_device_index());
  nvshmemx_putmem_signal_on_stream(dst_ptr, src_ptr, nelement, sig_addr, sigval, NVSHMEM_SIGNAL_SET,
                                   static_cast<int>(peer), stream);
#else
  STD_TORCH_CHECK(false, "NVSHMEM not available. Build with NVTE_ENABLE_NVSHMEM=1.");
#endif
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def("nvshmem_create_tensor(int num_elements, int scalar_type, int device_idx) -> Tensor");
  m.def("nvshmem_wait_on_current_stream(Tensor signal, int wait_kind) -> ()");
  m.def("nvshmem_send_on_current_stream(Tensor src, Tensor dst, int peer, Tensor signal) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("nvshmem_wait_on_current_stream", TORCH_BOX(nvshmem_wait_on_current_stream));
  m.impl("nvshmem_send_on_current_stream", TORCH_BOX(nvshmem_send_on_current_stream));
}

// nvshmem_create_tensor has no tensor input args, use CompositeImplicitAutograd
STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CompositeImplicitAutograd, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("nvshmem_create_tensor", TORCH_BOX(nvshmem_create_tensor));
}
