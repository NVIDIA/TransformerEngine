/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_nccl_loader.h
 *  \brief Lazy dlopen-based resolver for libnccl_ep.so.
 *
 *  libtransformer_engine.so is not link-time bound to libnccl_ep.so. The first
 *  call to ep::loader::fns() opens it via dlopen and dlsyms the ncclEp*
 *  entry points the EP backend uses. If the library or any symbol cannot be
 *  resolved (e.g. libnccl_ep.so is missing, or system NCCL is older than the
 *  EP minimum so libnccl_ep.so's own DT_NEEDED chain fails), the call throws
 *  NVTE_ERROR with remediation instead of preventing libtransformer_engine.so
 *  from loading.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_EP_EP_NCCL_LOADER_H_
#define TRANSFORMER_ENGINE_COMMON_EP_EP_NCCL_LOADER_H_

#include <nccl_ep.h>

namespace transformer_engine {
namespace ep {
namespace loader {

struct NcclEpFns {
  decltype(&::ncclEpInitHandle) InitHandle;
  decltype(&::ncclEpCreateGroup) CreateGroup;
  decltype(&::ncclEpGroupDestroy) GroupDestroy;
  decltype(&::ncclEpHandleDestroy) HandleDestroy;
  decltype(&::ncclEpHandleMemSize) HandleMemSize;
  decltype(&::ncclEpUpdateHandle) UpdateHandle;
  decltype(&::ncclEpDispatch) Dispatch;
  decltype(&::ncclEpCombine) Combine;
};

/*! \brief Resolve libnccl_ep.so on first call; cache the table thereafter.
 *  Thread-safe; throws NVTE_ERROR if the library or any symbol is missing.
 */
const NcclEpFns& fns();

}  // namespace loader
}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_NCCL_LOADER_H_
