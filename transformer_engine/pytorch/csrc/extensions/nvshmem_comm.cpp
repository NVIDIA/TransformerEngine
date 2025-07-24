/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

#ifdef NVTE_ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmem_api/nvshmem_waitkernel.h>
#include <nvshmemx.h>
#endif

#include <cuda.h>
#include <cuda_fp8.h>
#include <torch/cuda.h>
#include <torch/extension.h>

namespace transformer_engine::pytorch {

void init_nvshmem_backend(c10d::ProcessGroup *process_group) {
#ifdef NVTE_ENABLE_NVSHMEM
  nvshmemx_init_attr_t attr = {};
  nvshmemx_uniqueid_t id = {};

  int my_rank = process_group->getRank();
  int num_ranks = process_group->getSize();
  if (my_rank == 0) {
    nvshmemx_get_uniqueid(&id);
  }

  auto backend_is_nccl = (process_group->getBackendType() == c10d::ProcessGroup::BackendType::NCCL);
  NVTE_CHECK(backend_is_nccl, "Currently only support NCCL boostrap for NVSHMEM");
  auto datatensor =
      torch::from_blob(reinterpret_cast<void *>(&id),
                       {static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t) / sizeof(uint8_t))},
                       at::device(torch::kCPU).dtype(torch::kUInt8));
  auto datatmp = (backend_is_nccl) ? datatensor.cuda() : datatensor;

  c10d::BroadcastOptions bcast_opts;
  bcast_opts.rootRank = 0;
  std::vector<torch::Tensor> datachunk = {datatmp};
  auto work = process_group->broadcast(datachunk, bcast_opts);
  work->wait();

  if (backend_is_nccl) {
    datatensor.copy_(datatmp.cpu());
    datatmp = torch::Tensor();
  }

  nvshmemx_set_attr_uniqueid_args(my_rank, num_ranks, &id, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

  NVTE_CHECK(my_rank == nvshmem_my_pe(), "my_rank: ", my_rank,
             " != nvshmem_my_pe(): ", nvshmem_my_pe());
  NVTE_CHECK(num_ranks == nvshmem_n_pes(), "num_ranks: ", num_ranks,
             " != nvshmem_n_pes(): ", nvshmem_n_pes());
#else
  NVTE_ERROR("Internal TE error: init_nvshmem_backend cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled with NVTE_ENABLE_NVSHMEM=1!");
#endif
}

void nvshmem_wait_on_current_stream(torch::Tensor signal, const std::string &wait_kind) {
#ifdef NVTE_ENABLE_NVSHMEM
  uint64_t *sig_addr = reinterpret_cast<uint64_t *>(signal.data_ptr());
  cudaStream_t cur_stream = (cudaStream_t)at::cuda::getCurrentCUDAStream();

  WaitKind wait_kind_enum = WaitKind::STREAM_WAIT;

  if (wait_kind == "kernel") {
    wait_kind_enum = WaitKind::KERNEL_WAIT;
  } else if (wait_kind == "nvshmem") {
    wait_kind_enum = WaitKind::NVSHMEM_WAIT;
  } else if (wait_kind == "stream") {
    wait_kind_enum = WaitKind::STREAM_WAIT;
  } else {
    NVTE_ERROR("Invalid wait kind: ", wait_kind);
  }
  nvshmem_wait_on_stream(sig_addr, wait_kind_enum, cur_stream);

#else
  NVTE_ERROR(
      "Internal TE error: nvshmem_wait_on_current_stream cannot be initialized with valid PyTorch ",
      "distributed process groups when TE is compiled with NVTE_ENABLE_NVSHMEM=1!");
#endif
}

torch::Tensor create_nvshmem_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
#ifdef NVTE_ENABLE_NVSHMEM
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  return at::from_blob(
      nvshmem_malloc(size), shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu);
#else
  NVTE_ERROR("Internal TE error: create_nvshmem_tensor cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled with NVTE_ENABLE_NVSHMEM=1!");
#endif
}

void nvshmem_send_on_current_stream(torch::Tensor src, torch::Tensor dst, int peer,
                                    torch::Tensor signal) {
#ifdef NVTE_ENABLE_NVSHMEM
  void *src_ptr = reinterpret_cast<void *>(src.data_ptr());
  void *dst_ptr = reinterpret_cast<void *>(dst.data_ptr());
  uint64_t *sig_addr = reinterpret_cast<uint64_t *>(signal.data_ptr());
  auto nelement = src.numel() * src.element_size();
  uint64_t sigval = 1;
  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();

  nvshmemx_putmem_signal_on_stream(dst_ptr, src_ptr, nelement, sig_addr, sigval, NVSHMEM_SIGNAL_SET,
                                   peer, (cudaStream_t)cur_stream);
#else
  NVTE_ERROR(
      "Internal TE error: nvshmem_send_on_current_stream cannot be initialized with valid PyTorch ",
      "distributed process groups when TE is compiled with NVTE_ENABLE_NVSHMEM=1!");
#endif
}
void nvshmem_finalize() {
#ifdef NVTE_ENABLE_NVSHMEM
  nvshmem_finalize();
#else
  NVTE_ERROR("Internal TE error: nvshmem_finalize cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled with NVTE_ENABLE_NVSHMEM=1!");
#endif
}

}  // namespace transformer_engine::pytorch
