/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "nccl_ep_provider.h"

#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../util/cuda_runtime.h"
#include "../util/logging.h"
#include "../util/shared_lib_wrapper.h"

namespace transformer_engine {
namespace ep {
namespace nccl_ep {
namespace {

constexpr char kNCCLEPLibraryName[] = "libnccl_ep.so";
const char kLibraryAnchor = 0;

bool env_is_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0';
}

void set_default_env(const char* name, const std::filesystem::path& value) {
  NVTE_CHECK(setenv(name, value.c_str(), 0) == 0, "Could not set ", name, ": ",
             std::strerror(errno));
}

void append_unique(std::vector<std::filesystem::path>* paths, const std::filesystem::path& path) {
  if (path.empty()) return;
  for (const auto& existing : *paths) {
    if (existing == path) return;
  }
  paths->push_back(path);
}

// Add the conventional unversioned and major-version paths first, then discover
// fully versioned files (for example, libnccl_ep.so.0.1) for installations that
// omit symlinks. Preserve lookup order while avoiding duplicate dlopen attempts.
void append_library_dir(std::vector<std::filesystem::path>* candidates,
                        const std::filesystem::path& directory, const std::string& versioned_name) {
  append_unique(candidates, directory / kNCCLEPLibraryName);
  append_unique(candidates, directory / versioned_name);

  std::error_code error;
  if (!std::filesystem::is_directory(directory, error) || error) return;

  std::vector<std::filesystem::path> fully_versioned;
  const std::string prefix = versioned_name + ".";
  for (std::filesystem::directory_iterator iterator(directory, error), end;
       !error && iterator != end; iterator.increment(error)) {
    const std::filesystem::directory_entry& entry = *iterator;
    const std::string filename = entry.path().filename().string();
    std::error_code file_error;
    if (filename.rfind(prefix, 0) == 0 && entry.is_regular_file(file_error) && !file_error) {
      fully_versioned.push_back(entry.path());
    }
  }
  std::sort(fully_versioned.rbegin(), fully_versioned.rend());
  for (const auto& path : fully_versioned) append_unique(candidates, path);
}

std::vector<std::filesystem::path> nccl_ep_library_candidates() {
  using Path = std::filesystem::path;
  std::vector<Path> candidates;
  const std::string versioned_name =
      std::string(kNCCLEPLibraryName) + "." + std::to_string(NCCL_EP_MAJOR);

  if (env_is_set("NCCL_EP_HOME")) {
    const Path home = std::getenv("NCCL_EP_HOME");
    append_library_dir(&candidates, home / "lib", versioned_name);
    append_library_dir(&candidates, home / "lib64", versioned_name);
  }

  const Path library_dir = shared_library_directory(static_cast<const void*>(&kLibraryAnchor));
  append_library_dir(&candidates, library_dir, versioned_name);
  append_unique(&candidates, versioned_name);
  append_unique(&candidates, kNCCLEPLibraryName);
  return candidates;
}

std::optional<int> nccl_header_version(const std::filesystem::path& include_dir) {
  std::ifstream header(include_dir / "nccl.h");
  if (!header) return std::nullopt;

  int major = -1;
  int minor = -1;
  int patch = -1;
  for (std::string line; std::getline(header, line);) {
    if (major < 0) std::sscanf(line.c_str(), "#define NCCL_MAJOR %d", &major);
    if (minor < 0) std::sscanf(line.c_str(), "#define NCCL_MINOR %d", &minor);
    if (patch < 0) std::sscanf(line.c_str(), "#define NCCL_PATCH %d", &patch);
  }
  if (major < 0 || minor < 0 || patch < 0) return std::nullopt;
  return major * 10000 + minor * 100 + patch;
}

void append_ancestor_include_dirs(std::vector<std::filesystem::path>* candidates,
                                  std::filesystem::path directory) {
  while (!directory.empty()) {
    const std::filesystem::path parent = directory.parent_path();
    if (parent == directory) break;
    append_unique(candidates, directory / "include");
    append_unique(candidates, directory / "nvidia" / "nccl" / "include");
    directory = parent;
  }
}

void configure_nccl_ep_source_dir(const std::filesystem::path& library_path) {
  if (env_is_set("NCCL_EP_HOME") || env_is_set("NCCL_EP_JIT_SOURCE_DIR")) {
    return;
  }

  const std::filesystem::path library_dir = library_path.parent_path();
  const std::filesystem::path homes[] = {
      library_dir / "nccl_ep",
      library_dir,
      library_dir.parent_path(),
  };
  for (const auto& home : homes) {
    std::error_code error;
    if (std::filesystem::is_directory(home / "include" / "nccl_ep", error) && !error) {
      set_default_env("NCCL_EP_HOME", home);
      return;
    }
  }
  NVTE_ERROR("Could not find NCCL EP JIT headers relative to ", library_path.string(),
             ". Set NCCL_EP_HOME or NCCL_EP_JIT_SOURCE_DIR.");
}

void configure_nccl_include_dir() {
  if (env_is_set("NCCL_EP_JIT_BUILD_INCLUDE_DIR") || env_is_set("NCCL_HOME")) {
    return;
  }

  int runtime_version = 0;
  NVTE_CHECK_NCCL(ncclGetVersion(&runtime_version));

  std::vector<std::filesystem::path> candidates;
  append_ancestor_include_dirs(
      &candidates, shared_library_directory(reinterpret_cast<const void*>(&ncclGetVersion)));
  append_ancestor_include_dirs(&candidates,
                               shared_library_directory(static_cast<const void*>(&kLibraryAnchor)));
  append_unique(&candidates, "/opt/nvidia/nccl/include");
  append_unique(&candidates, "/usr/local/nccl/include");
  append_unique(&candidates, "/usr/include");

  for (const auto& candidate : candidates) {
    if (nccl_header_version(candidate) == runtime_version) {
      set_default_env("NCCL_EP_JIT_BUILD_INCLUDE_DIR", candidate);
      return;
    }
  }
  NVTE_ERROR("Could not find NCCL headers matching runtime NCCL ", runtime_version / 10000, ".",
             (runtime_version / 100) % 100, ".", runtime_version % 100,
             ". Set NCCL_EP_JIT_BUILD_INCLUDE_DIR to the directory containing nccl.h or set "
             "NCCL_HOME.");
}

void configure_cuda_include_dir() {
  if (env_is_set("NCCL_EP_JIT_CUDA_INCLUDE_DIR") || env_is_set("CUDA_HOME") ||
      env_is_set("CUDA_PATH")) {
    return;
  }

  const std::string& include_dir = cuda::include_directory(false);
  if (!include_dir.empty()) {
    set_default_env("NCCL_EP_JIT_CUDA_INCLUDE_DIR", include_dir);
  }
}

void configure_jit_environment(const std::filesystem::path& library_path) {
  configure_nccl_ep_source_dir(library_path);
  configure_nccl_include_dir();
  configure_cuda_include_dir();
}

void* load_symbol(void* handle, const char* name) {
  dlerror();
  void* symbol = dlsym(handle, name);
  const char* error = dlerror();
  NVTE_CHECK(error == nullptr && symbol != nullptr, "Could not load ", name, " from ",
             kNCCLEPLibraryName, ": ", error == nullptr ? "symbol not found" : error);
  return symbol;
}

void* open_nccl_ep_library() {
  std::string failures;
  for (const auto& candidate : nccl_ep_library_candidates()) {
    dlerror();
    if (void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL)) {
      return handle;
    }
    const char* error = dlerror();
    if (!failures.empty()) failures += "; ";
    failures +=
        candidate.string() + ": " + (error == nullptr ? "unknown dynamic loader error" : error);
  }
  NVTE_ERROR("Could not load ", kNCCLEPLibraryName, ". Tried ", failures);
  return nullptr;  // Unreachable.
}

void* library_handle() {
  // Deliberately keep libnccl_ep loaded until process exit. EPBackend owns
  // objects whose implementation lives in this library.
  static void* handle = [] {
    void* result = open_nccl_ep_library();

    try {
      using GetVersion = decltype(&ncclEpGetVersion);
      auto get_version = reinterpret_cast<GetVersion>(load_symbol(result, "ncclEpGetVersion"));
      int runtime_version = 0;
      NVTE_CHECK_NCCL(get_version(&runtime_version));
      NVTE_CHECK(runtime_version / 10000 == NCCL_EP_MAJOR,
                 "Incompatible NCCL EP library major version ", runtime_version / 10000,
                 "; expected ", NCCL_EP_MAJOR);
      const std::filesystem::path library_path =
          shared_library_path(reinterpret_cast<const void*>(get_version));
      NVTE_CHECK(!library_path.empty(), "Could not determine the NCCL EP library path");
      configure_jit_environment(library_path);
    } catch (...) {
      dlclose(result);
      throw;
    }
    return result;
  }();
  return handle;
}

}  // namespace

void initialize() { (void)library_handle(); }

void* get_symbol(const char* symbol) { return load_symbol(library_handle(), symbol); }

namespace {

template <auto FuncPtr, typename... Args>
auto call_symbol(const char* name, Args&&... args) {
  using FuncT = decltype(FuncPtr);
  static FuncT func = reinterpret_cast<FuncT>(get_symbol(name));
  return func(std::forward<Args>(args)...);
}

}  // namespace

ncclResult_t create_group(ncclEpGroup_t* ep_group, ncclComm_t comm,
                          const ncclEpGroupConfig_t* config) {
  return call_symbol<&ncclEpCreateGroup>("ncclEpCreateGroup", ep_group, comm, config);
}

ncclResult_t group_destroy(ncclEpGroup_t ep_group) {
  return call_symbol<&ncclEpGroupDestroy>("ncclEpGroupDestroy", ep_group);
}

ncclResult_t handle_destroy(ncclEpHandle_t handle) {
  return call_symbol<&ncclEpHandleDestroy>("ncclEpHandleDestroy", handle);
}

ncclResult_t init_handle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout,
                         const ncclEpHandleConfig_t* config, int num_topk,
                         const ncclEpTensor_t* handle_mem) {
  return call_symbol<&ncclEpInitHandle>("ncclEpInitHandle", handle, ep_group, layout, config,
                                        num_topk, handle_mem);
}

ncclResult_t handle_mem_size(ncclEpGroup_t ep_group, ncclEpLayout_t layout,
                             const ncclEpHandleConfig_t* config, size_t* size_out, int num_topk) {
  return call_symbol<&ncclEpHandleMemSize>("ncclEpHandleMemSize", ep_group, layout, config,
                                           size_out, num_topk);
}

ncclResult_t update_handle(ncclEpHandle_t handle, const ncclEpTensor_t* topk_idx,
                           const ncclEpLayoutInfo_t* layout_info, cudaStream_t stream) {
  return call_symbol<&ncclEpUpdateHandle>("ncclEpUpdateHandle", handle, topk_idx, layout_info,
                                          stream);
}

ncclResult_t dispatch(ncclEpHandle_t handle, const ncclEpDispatchInputs_t* inputs,
                      const ncclEpDispatchOutputs_t* outputs, const ncclEpLayoutInfo_t* layout_info,
                      const ncclEpDispatchConfig_t* config, cudaStream_t stream) {
  return call_symbol<&ncclEpDispatch>("ncclEpDispatch", handle, inputs, outputs, layout_info,
                                      config, stream);
}

ncclResult_t combine(ncclEpHandle_t handle, const ncclEpCombineInputs_t* inputs,
                     const ncclEpCombineOutputs_t* outputs, const ncclEpCombineConfig_t* config,
                     cudaStream_t stream) {
  return call_symbol<&ncclEpCombine>("ncclEpCombine", handle, inputs, outputs, config, stream);
}

}  // namespace nccl_ep
}  // namespace ep
}  // namespace transformer_engine
