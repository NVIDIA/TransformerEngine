/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "../../../common/util/cuda_driver.h"

#ifdef NVTE_ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmem_api/nvshmem_waitkernel.h>
#include <nvshmemx.h>
#endif

#include <cuda.h>
#include <cuda_fp8.h>
#include <torch/cuda.h>
#include <torch/extension.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>

namespace transformer_engine::pytorch {

namespace {

std::array<std::atomic<int64_t>, 4> cp_global_grad_return_epochs{};

at::Tensor cp_grad_return_slot(const at::Tensor &buffer, const at::Tensor &reference,
                               int cp_size, int writer_rank, const char *name) {
  NVTE_CHECK(buffer.defined() && buffer.dim() == reference.dim() + 1,
             name, " must have shape [CP, S, B, H, D].");
  NVTE_CHECK(buffer.size(0) == cp_size, name, " leading dimension must equal CP size.");
  for (int dim = 0; dim < reference.dim(); ++dim) {
    NVTE_CHECK(buffer.size(dim + 1) == reference.size(dim),
               name, " trailing dimensions must match local K/V.");
  }
  return buffer.select(0, writer_rank);
}

at::Tensor cp_epoch_slot(const at::Tensor &epochs, int writer_rank, const char *name) {
  NVTE_CHECK(epochs.is_cuda() && epochs.scalar_type() == torch::kInt32 &&
                 epochs.is_contiguous() && epochs.dim() == 1 &&
                 epochs.size(0) > writer_rank,
             name, " must be a contiguous CUDA int32 vector indexed by writer rank.");
  return epochs.select(0, writer_rank);
}

void cp_stream_write_epoch(const at::Tensor &epochs, int writer_rank, int64_t epoch) {
  NVTE_CHECK(epoch > 0 && epoch <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
             "CP gradient epoch must fit in int32.");
  at::Tensor slot = cp_epoch_slot(epochs, writer_rank, "peer_grad_committed_epoch");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  NVTE_CHECK_CUDA_DRIVER(cuStreamWriteValue32(
      reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(slot.data_ptr()),
      static_cast<cuuint32_t>(epoch), 0));
}

void cp_stream_wait_epochs(const at::Tensor &epochs, int cp_size, int64_t epoch) {
  NVTE_CHECK(epochs.is_cuda() && epochs.scalar_type() == torch::kInt32 &&
                 epochs.is_contiguous() && epochs.dim() == 1 && epochs.size(0) == cp_size,
             "grad_committed_epoch must be a contiguous CUDA int32 vector of CP size.");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto *base = epochs.data_ptr<int32_t>();
  for (int source = 0; source < cp_size; ++source) {
    NVTE_CHECK_CUDA_DRIVER(cuStreamWaitValue32(
        reinterpret_cast<CUstream>(stream),
        reinterpret_cast<CUdeviceptr>(base + source), static_cast<cuuint32_t>(epoch),
        CU_STREAM_WAIT_VALUE_GEQ));
  }
}

}  // namespace

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

std::vector<at::Tensor> nvshmem_cp_global_grad_return_execute(
    at::Tensor dk_global, at::Tensor dv_global, at::Tensor key, at::Tensor value,
    at::Tensor grad_key_return, at::Tensor grad_value_return,
    at::Tensor grad_committed_epoch,
    const std::vector<at::Tensor> &peer_grad_key_returns,
    const std::vector<at::Tensor> &peer_grad_value_returns,
    const std::vector<at::Tensor> &peer_grad_committed_epochs, int cp_size, int rank) {
  NVTE_CHECK(cp_size == 4 && rank >= 0 && rank < cp_size,
             "NVSHMEM global gradient return currently requires CP=4.");
  NVTE_CHECK(key.is_cuda() && value.is_cuda() && dk_global.is_cuda() && dv_global.is_cuda(),
             "NVSHMEM global gradient return requires CUDA tensors.");
  NVTE_CHECK(key.sizes() == value.sizes() && dk_global.sizes() == dv_global.sizes(),
             "K/V and global dK/dV pairs must have matching shapes.");
  NVTE_CHECK(dk_global.dim() == key.dim() && dk_global.size(0) == key.size(0) * cp_size,
             "Global dK/dV sequence length must be CP times local K/V sequence length.");
  for (int dim = 1; dim < key.dim(); ++dim) {
    NVTE_CHECK(dk_global.size(dim) == key.size(dim),
               "Global dK/dV non-sequence dimensions must match local K/V.");
  }
  NVTE_CHECK(key.size(0) % 2 == 0, "Local K/V sequence length must be even.");
  NVTE_CHECK(static_cast<int>(peer_grad_key_returns.size()) == cp_size &&
                 static_cast<int>(peer_grad_value_returns.size()) == cp_size &&
                 static_cast<int>(peer_grad_committed_epochs.size()) == cp_size,
             "Expected one symmetric gradient and epoch view per CP owner.");

  cp_grad_return_slot(grad_key_return, key, cp_size, rank, "grad_key_return");
  cp_grad_return_slot(grad_value_return, value, cp_size, rank, "grad_value_return");
  const int64_t half = key.size(0) / 2;
  for (int owner = 0; owner < cp_size; ++owner) {
    at::Tensor key_slot = cp_grad_return_slot(
        peer_grad_key_returns[owner], key, cp_size, rank, "peer_grad_key_return");
    at::Tensor value_slot = cp_grad_return_slot(
        peer_grad_value_returns[owner], value, cp_size, rank, "peer_grad_value_return");
    key_slot.narrow(0, 0, half).copy_(dk_global.narrow(0, owner * half, half));
    key_slot.narrow(0, half, half).copy_(dk_global.narrow(0, (7 - owner) * half, half));
    value_slot.narrow(0, 0, half).copy_(dv_global.narrow(0, owner * half, half));
    value_slot.narrow(0, half, half).copy_(dv_global.narrow(0, (7 - owner) * half, half));
  }

  const int64_t epoch = cp_global_grad_return_epochs[rank].fetch_add(1) + 1;
  for (int owner = 0; owner < cp_size; ++owner) {
    cp_stream_write_epoch(peer_grad_committed_epochs[owner], rank, epoch);
  }
  cp_stream_wait_epochs(grad_committed_epoch, cp_size, epoch);

  at::Tensor dk = grad_key_return.select(0, 0).clone();
  at::Tensor dv = grad_value_return.select(0, 0).clone();
  for (int source = 1; source < cp_size; ++source) {
    dk.add_(grad_key_return.select(0, source));
    dv.add_(grad_value_return.select(0, source));
  }
  return {dk.to(key.scalar_type()), dv.to(value.scalar_type())};
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
