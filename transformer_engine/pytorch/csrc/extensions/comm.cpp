/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NCCL backend headers, for borrowing the raw ncclComm_t. The NCCL2 backend
// header exists only on torch builds that ship it.
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#if __has_include(<torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>)
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>
#define NVTE_HAS_NCCL2_PG 1
#endif

#include <type_traits>
#include <utility>

#include "../common.h"
#include "../extensions.h"

namespace transformer_engine::pytorch {

#ifdef NVTE_HAS_NCCL2_PG
namespace {

// nccl2::ProcessGroupNCCL::getCommPtr() is not guaranteed on every torch release
// that ships the NCCL2 backend. Detect it so the borrow compiles regardless and
// fails only at runtime when the backend cannot provide the communicator.
template <typename PG, typename = void>
struct has_get_comm_ptr : std::false_type {};
template <typename PG>
struct has_get_comm_ptr<PG, std::void_t<decltype(std::declval<PG&>().getCommPtr())>>
    : std::true_type {};

template <typename PG>
int64_t borrow_comm_ptr(PG* pg) {
  if constexpr (has_get_comm_ptr<PG>::value) {
    return pg->getCommPtr();
  } else {
    NVTE_ERROR(
        "get_nccl_comm_ptr: the NCCL2 backend of this PyTorch version does not expose the raw "
        "NCCL communicator (getCommPtr()), which EP and cuSolverMp need to borrow. Set "
        "TORCH_DIST_USE_NCCL2=0 to use the classic NCCL backend instead.");
  }
}

}  // namespace
#endif  // NVTE_HAS_NCCL2_PG

int64_t get_nccl_comm_ptr(c10d::ProcessGroup* process_group) {
  NVTE_CHECK(process_group != nullptr, "get_nccl_comm_ptr: process_group must be non-null");
  auto backend = process_group->getBackend(c10::DeviceType::CUDA);
  NVTE_CHECK(backend, "get_nccl_comm_ptr: process group has no CUDA backend");

  // Classic backend (ProcessGroupNCCL._comm_ptr() in Python).
  if (auto* nccl_pg = dynamic_cast<c10d::ProcessGroupNCCL*>(backend.get())) {
    return nccl_pg->getCommPtr();
  }
  // NCCL2 backend, default on recent torch.
#ifdef NVTE_HAS_NCCL2_PG
  if (auto* nccl2_pg = dynamic_cast<c10d::nccl2::ProcessGroupNCCL*>(backend.get())) {
    return borrow_comm_ptr(nccl2_pg);
  }
#endif

  NVTE_ERROR("get_nccl_comm_ptr requires a NCCL-backed process group, but backend '",
             backend->getBackendName(), "' does not expose a borrowable NCCL communicator");
}

}  // namespace transformer_engine::pytorch
