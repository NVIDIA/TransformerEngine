/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "ep_nccl_loader.h"

#include <dlfcn.h>

#include "../util/logging.h"

namespace transformer_engine {
namespace ep {
namespace loader {

namespace {

constexpr const char* kSonames[] = {"libnccl_ep.so.0", "libnccl_ep.so"};

void* try_dlopen(std::string& last_err) {
  for (const char* name : kSonames) {
    dlerror();
    void* h = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (h != nullptr) return h;
    if (const char* e = dlerror()) last_err = e;
  }
  return nullptr;
}

template <typename Fn>
Fn resolve(void* lib, const char* sym) {
  dlerror();
  void* p = dlsym(lib, sym);
  const char* err = dlerror();
  NVTE_CHECK(err == nullptr && p != nullptr, "libnccl_ep.so is loaded but symbol '", sym,
             "' could not be resolved", (err != nullptr ? std::string(": ") + err : std::string{}),
             ". The runtime libnccl_ep.so is older than the version TransformerEngine "
             "was built against; upgrade NCCL EP or rebuild TE with -DNVTE_WITH_NCCL_EP=OFF.");
  return reinterpret_cast<Fn>(p);
}

NcclEpFns load_or_throw() {
  std::string last_err;
  void* lib = try_dlopen(last_err);
  NVTE_CHECK(lib != nullptr, "Failed to load libnccl_ep.so (",
             (last_err.empty() ? "no error message" : last_err),
             "). NCCL EP requires libnccl_ep.so (>= 0.0.1) and NCCL >= 2.30.4 at runtime. "
             "Install the NCCL EP shared library, or rebuild TransformerEngine with "
             "-DNVTE_WITH_NCCL_EP=OFF to disable EP support.");
  NcclEpFns fns{};
  fns.InitHandle = resolve<decltype(&::ncclEpInitHandle)>(lib, "ncclEpInitHandle");
  fns.CreateGroup = resolve<decltype(&::ncclEpCreateGroup)>(lib, "ncclEpCreateGroup");
  fns.GroupDestroy = resolve<decltype(&::ncclEpGroupDestroy)>(lib, "ncclEpGroupDestroy");
  fns.HandleDestroy = resolve<decltype(&::ncclEpHandleDestroy)>(lib, "ncclEpHandleDestroy");
  fns.HandleMemSize = resolve<decltype(&::ncclEpHandleMemSize)>(lib, "ncclEpHandleMemSize");
  fns.UpdateHandle = resolve<decltype(&::ncclEpUpdateHandle)>(lib, "ncclEpUpdateHandle");
  fns.Dispatch = resolve<decltype(&::ncclEpDispatch)>(lib, "ncclEpDispatch");
  fns.Combine = resolve<decltype(&::ncclEpCombine)>(lib, "ncclEpCombine");
  return fns;
}

}  // namespace

const NcclEpFns& fns() {
  // Function-local static: thread-safe one-shot init; re-throws on every call
  // if initialization fails, so a missing library is surfaced consistently.
  static const NcclEpFns table = load_or_throw();
  return table;
}

}  // namespace loader
}  // namespace ep
}  // namespace transformer_engine
