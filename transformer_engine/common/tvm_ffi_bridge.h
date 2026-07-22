/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_TVM_FFI_BRIDGE_H_
#define TRANSFORMER_ENGINE_COMMON_TVM_FFI_BRIDGE_H_

#include <dlfcn.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "transformer_engine/transformer_engine.h"
#include "util/cuda_runtime.h"
#include "util/logging.h"

namespace transformer_engine {
namespace tvm_ffi_bridge {

inline const char *te_dtype_to_str(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "fp32";
    case DType::kFloat16:
      return "fp16";
    case DType::kBFloat16:
      return "bf16";
    case DType::kFloat8E4M3:
      return "e4m3";
    case DType::kFloat8E5M2:
      return "e5m2";
    default:
      return "";
  }
}

// Fused activation token forwarded to Python. Encodes both the family and the
// forward-vs-derivative direction: "relu" is the forward activation, "drelu" its
// backward derivative (dact). This is why no separate is_act/is_dact flag is
// needed — the token carries it; only with_dbias (orthogonal) is a separate flag.
// The d-variants are slots for the not-yet-wired backward path; the forward
// tokens must match Python's SUPPORTED_ACTIVATIONS set.
enum class Activation {
  kNone,
  kReLU,
  kGeLU,
  kSiLU,
  kQGeLU,
  kSReLU,
  kDReLU,
  kDGeLU,
  kDSiLU,
  kDQGeLU,
  kDSReLU
};

inline const char *activation_to_str(Activation act) {
  switch (act) {
    case Activation::kReLU:
      return "relu";
    case Activation::kGeLU:
      return "gelu";
    case Activation::kSiLU:
      return "silu";
    case Activation::kQGeLU:
      return "qgelu";
    case Activation::kSReLU:
      return "srelu";
    case Activation::kDReLU:
      return "drelu";
    case Activation::kDGeLU:
      return "dgelu";
    case Activation::kDSiLU:
      return "dsilu";
    case Activation::kDQGeLU:
      return "dqgelu";
    case Activation::kDSReLU:
      return "dsrelu";
    case Activation::kNone:
      return "none";
  }
  return "none";
}

inline DLDataType convert_to_dltype(NVTEDType type) {
  switch (type) {
    case kNVTEFloat32:
      return DLDataType{kDLFloat, 32, 1};
    case kNVTEFloat16:
      return DLDataType{kDLFloat, 16, 1};
    case kNVTEBFloat16:
      return DLDataType{kDLBfloat, 16, 1};
    case kNVTEByte:
      return DLDataType{kDLUInt, 8, 1};
    case kNVTEInt32:
      return DLDataType{kDLInt, 32, 1};
    case kNVTEInt64:
      return DLDataType{kDLInt, 64, 1};
    // Native DLPack (>= 1.1) FP8 codes. TE's E4M3 is CUDA's finite __nv_fp8_e4m3,
    // i.e. the "fn" variant (kDLFloat8_e4m3 would be the IEEE-style type with
    // infinities); E8M0 scales are the unsigned, finite, single-NaN MX format.
    case kNVTEFloat8E4M3:
      return DLDataType{kDLFloat8_e4m3fn, 8, 1};
    case kNVTEFloat8E5M2:
      return DLDataType{kDLFloat8_e5m2, 8, 1};
    case kNVTEFloat8E8M0:
      return DLDataType{kDLFloat8_e8m0fnu, 8, 1};
    // FP4
    case kNVTEFloat4E2M1:
      return DLDataType{kDLFloat4_e2m1fn, 4, 1};
    default:
      NVTE_ERROR("unsupported NVTEDType: ", static_cast<int>(type));
  }
}

class DLTensorWrapper : public DLTensor {
 public:
  // Null wrapper (data == nullptr): packs as TVM-FFI None, no allocation.
  DLTensorWrapper() : DLTensor{} {}

  explicit DLTensorWrapper(const NVTEBasicTensor &tensor, bool flatten_2D = true) {
    const int32_t device_index = transformer_engine::cuda::current_device();
    const int n = static_cast<int>(tensor.shape.ndim);
    if (flatten_2D && n > 2) {
      int64_t flat_first = 1;
      for (int i = 0; i + 1 < n; ++i) flat_first *= static_cast<int64_t>(tensor.shape.data[i]);
      const int64_t flat_last = static_cast<int64_t>(tensor.shape.data[n - 1]);
      shape_buf_ = std::make_unique<int64_t[]>(2);
      strides_buf_ = std::make_unique<int64_t[]>(2);
      shape_buf_[0] = flat_first;
      shape_buf_[1] = flat_last;
      strides_buf_[0] = flat_last;
      strides_buf_[1] = 1;
      this->ndim = 2;
    } else {
      shape_buf_ = std::make_unique<int64_t[]>(n);
      strides_buf_ = std::make_unique<int64_t[]>(n);
      int64_t stride = 1;
      for (int i = n - 1; i >= 0; --i) {
        shape_buf_[i] = static_cast<int64_t>(tensor.shape.data[i]);
        strides_buf_[i] = stride;
        stride *= shape_buf_[i];
      }
      this->ndim = n;
    }
    this->data = tensor.data_ptr;
    this->device = DLDevice{kDLCUDA, device_index};
    this->dtype = convert_to_dltype(tensor.dtype);
    this->shape = shape_buf_.get();
    this->strides = strides_buf_.get();
    this->byte_offset = 0;
  }

  ~DLTensorWrapper() = default;
  DLTensorWrapper(const DLTensorWrapper &) = delete;
  DLTensorWrapper &operator=(const DLTensorWrapper &) = delete;
  DLTensorWrapper(DLTensorWrapper &&) = default;
  DLTensorWrapper &operator=(DLTensorWrapper &&) = default;

 private:
  std::unique_ptr<int64_t[]> shape_buf_;
  std::unique_ptr<int64_t[]> strides_buf_;
};

}  // namespace tvm_ffi_bridge
}  // namespace transformer_engine

namespace tvm {
namespace ffi {
// Make a (borrowed) DLTensorWrapper* a first-class TVM-FFI argument, so wrappers
// can be passed straight to Function::operator()(&w, ...). Like DLTensor* it is a
// non-owning DLTensorPtr view (the wrapper must outlive the call), but a null
// pointer OR a wrapper over an absent buffer (null data) packs as TVM-FFI None —
// so a kernel's optional args need no special handling at the call site. Only
// the pack-as-argument path (CopyToAnyView) is provided; reading back is unused.
// Declared after DLTensorWrapper: the specialization needs the complete type
// (it reads src->data and static_casts to its DLTensor base).
template <>
struct TypeTraits<transformer_engine::tvm_ffi_bridge::DLTensorWrapper *>
    : public TypeTraits<DLTensor *> {
  TVM_FFI_INLINE static void CopyToAnyView(transformer_engine::tvm_ffi_bridge::DLTensorWrapper *src,
                                           TVMFFIAny *result) {
    if (src == nullptr || src->data == nullptr) {
      TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);  // -> TVM-FFI None
    } else {
      TypeTraits<DLTensor *>::CopyToAnyView(static_cast<DLTensor *>(src), result);
    }
  }
};
}  // namespace ffi
}  // namespace tvm

namespace transformer_engine {
namespace tvm_ffi_bridge {

// Compile-time check that a config provides the lazy-loadable kernel API:
//   - std::string to_key() const
//   - bool retrieve_func_from_python(const std::string& key) const
//       (compiles + globally registers the kernel under `key`; returns whether
//        a kernel is now registered / the config is supported)
// Drives the static_assert in TVMFFICentral::lazyload_function so a config that
// is missing either method fails with a clear message instead of a deref-into-
// the-template error.
namespace detail {
template <typename, typename = void>
struct is_lazyloadable_config : std::false_type {};
template <typename T>
struct is_lazyloadable_config<
    T, std::void_t<decltype(std::declval<const T &>().to_key()),
                   decltype(std::declval<const T &>().retrieve_func_from_python(
                       std::declval<const std::string &>()))>> : std::true_type {};
}  // namespace detail

class TVMFFICentral {
 public:
  static TVMFFICentral &getInstance() {
    // Deliberately leaked (never deleted) because cache_ holds Python-backed
    // tvm::ffi::Function handles whose decref must NOT run at static-destruction
    // time -- Python / the tvm-ffi registry may already be finalized by then,
    // which would be a use-after-free crash at process exit.
    static TVMFFICentral *instance = new TVMFFICentral();
    return *instance;
  }

  template <typename Config>
  std::optional<tvm::ffi::Function> lazyload_function(const Config &cfg) {
    static_assert(detail::is_lazyloadable_config<Config>::value,
                  "Config must define `std::string to_key() const` and "
                  "`bool retrieve_func_from_python(const std::string&) const`.");
    if (!cutedsl_backend_enabled_.load(std::memory_order_relaxed)) {
      if (warn_cutedsl_backend_not_chosen_) {
        NVTE_WARN("CuTeDSL kernel for config `", cfg.to_key(),
                  "` is not supported because the CuTeDSL backend is disabled. "
                  "Set NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=1 to enable it.");
      }
      return std::nullopt;
    }
    // Only check if libtvm_ffi.so is loaded if user enables the CuTeDSL backend.
    // So if user disables the CuTeDSL backend, don't output this warning message.
    if (!tvm_ffi_available_) {
      NVTE_WARN(
          "Cannot dispatch to CuTeDSL kernels because libtvm_ffi.so is not successfully loaded."
          " Will fall back to the default CUDA C++ kernels.");
      return std::nullopt;
    }
    const std::string key = cfg.to_key();
    {
      std::shared_lock<std::shared_mutex> read_lock(mutex_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {
        // If the key is present, the value is either a valid tvm::ffi::Function or std::nullopt (indicating config not supported)
        return it->second;
      }
    }
    // First time we see this config since the key isn't present in the cache: ask Python to compile + register the kernel
    // under `key`, then resolve it once and cache the Function (or nullopt if unsupported)
    std::optional<tvm::ffi::Function> fn =
        cfg.retrieve_func_from_python(key) ? tvm::ffi::Function::GetGlobal(key) : std::nullopt;
    {
      std::unique_lock<std::shared_mutex> write_lock(mutex_);
      // emplace is a no-op if another thread populated this key meanwhile; the
      // resolved value is identical, so either copy is fine.
      cache_.emplace(key, fn);
    }
    if (!fn && warn_cutedsl_backend_not_chosen_) {
      NVTE_WARN("TVM-FFI kernel for config `", key, "` is not supported.");
    }
    return fn;
  }

  // Runtime override of NVTE_ENABLE_CUTEDSL_QUANT_BACKEND (exposed to Python as
  // nvte_set_cutedsl_quant_backend; used by tests to compare both backends in
  // one process). Safe to toggle at any time
  void set_cutedsl_backend_enabled(bool enabled) {
    cutedsl_backend_enabled_.store(enabled, std::memory_order_relaxed);
  }

 private:
  ~TVMFFICentral() = default;
  TVMFFICentral()
      : tvm_ffi_available_(load_tvm_ffi()),
        cutedsl_backend_enabled_(is_cutedsl_backend_enabled()),
        warn_cutedsl_backend_not_chosen_(warn_if_cutedsl_backend_not_chosen()) {}

  // Load all tvm-ffi symbols into the global namespace, which should be already loaded in common/__init__.py via ctypes.CDLL
  // if user uses TE from a python environment. Otherwise, if user stays in C++ only without python, then CuTeDSL kernels
  // will be unavailable either because we fail to load libtvm_ffi.so or CuTeDSL kernel entrypoints are not registered in Python.
  // In either case, we will fall back to the default TE CUDA C++ kernels.
  static bool load_tvm_ffi() { return dlopen("libtvm_ffi.so", RTLD_NOW | RTLD_GLOBAL) != nullptr; }
  TVMFFICentral(const TVMFFICentral &) = delete;
  TVMFFICentral &operator=(const TVMFFICentral &) = delete;
  TVMFFICentral(TVMFFICentral &&) = delete;
  TVMFFICentral &operator=(TVMFFICentral &&) = delete;

  static bool is_cutedsl_backend_enabled() {
    // Off by default; set NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=1 to enable.
    const char *flag = std::getenv("NVTE_ENABLE_CUTEDSL_QUANT_BACKEND");
    return flag != nullptr && flag[0] != '0';
  }

  static bool warn_if_cutedsl_backend_not_chosen() {
    const char *flag = std::getenv("NVTE_WARN_IF_CUTEDSL_BACKEND_NOT_CHOSEN");
    return flag != nullptr && flag[0] != '0';
  }

  const bool tvm_ffi_available_;  // libtvm_ffi.so loaded; false disables the backend
  std::atomic<bool> cutedsl_backend_enabled_;
  const bool warn_cutedsl_backend_not_chosen_;
  std::shared_mutex mutex_;
  // Per-config resolved kernel: cfg.to_key() -> GetGlobal result (std::nullopt ==
  // unsupported). Holds Python-backed tvm::ffi::Function handles; safe ONLY because
  // the singleton is deliberately leaked (see getInstance), so these are never
  // decref'd at static teardown.
  std::unordered_map<std::string, std::optional<tvm::ffi::Function>> cache_;
};

}  // namespace tvm_ffi_bridge
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_TVM_FFI_BRIDGE_H_
