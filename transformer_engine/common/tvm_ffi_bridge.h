/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_TVM_FFI_BRIDGE_H_
#define TRANSFORMER_ENGINE_COMMON_TVM_FFI_BRIDGE_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
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
    // FP8 / E8M0 → raw 1-byte uint; the kernel interprets the bits.
    case kNVTEFloat8E4M3:
      return DLDataType{kDLUInt, 8, 1};
    case kNVTEFloat8E5M2:
      return DLDataType{kDLUInt, 8, 1};
    case kNVTEFloat8E8M0:
      return DLDataType{kDLUInt, 8, 1};
    default:
      NVTE_ERROR("unsupported NVTEDType: ", static_cast<int>(type));
  }
}

class DLTensorWrapper : public DLTensor {
 public:
  // Null wrapper (data == nullptr): packs as TVM-FFI None, no allocation.
  DLTensorWrapper() : DLTensor{} {}

  DLTensorWrapper(const NVTEBasicTensor &tensor, bool flatten_2D = true) {
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
    static TVMFFICentral instance;
    return instance;
  }

  // Resolve the compiled kernel for `cfg`. The kernel itself lives in the tvm-ffi
  // global registry (registered by the Python entrypoint under cfg.to_key()),
  // which releases its Python-backed entries safely at interpreter shutdown; we
  // fetch it per call with GetGlobal(key). C++ caches only a bool per config
  // (supported or not), so Python is asked at most once per config and we never
  // hold a Python-backed handle in a static-duration object (which would crash
  // at exit, when the singleton is torn down after the interpreter is finalized).
  template <typename Config>
  std::optional<tvm::ffi::Function> lazyload_function(const Config &cfg) {
    static_assert(detail::is_lazyloadable_config<Config>::value,
                  "Config must define `std::string to_key() const` and "
                  "`bool retrieve_func_from_python(const std::string&) const`.");
    if (!cutedsl_backend_enabled_) {
      if (warn_cutedsl_backend_not_chosen_) {
        NVTE_WARN("TVM-FFI kernel for config `", cfg.to_key(),
                  "` is not supported because the CuTeDSL backend is disabled. "
                  "Set NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=1 to enable it.");
      }
      return std::nullopt;
    }
    const std::string key = cfg.to_key();
    {
      std::shared_lock<std::shared_mutex> read_lock(mutex_);
      auto it = supported_.find(key);
      if (it != supported_.end()) {
        return it->second ? tvm::ffi::Function::GetGlobal(key) : std::nullopt;
      }
    }
    // Cold miss: ask Python to compile + globally register the kernel under
    // `key`; cache only the support decision (avoids re-asking Python, and
    // negative-caches unsupported configs).
    const bool supported = cfg.retrieve_func_from_python(key);
    {
      std::unique_lock<std::shared_mutex> write_lock(mutex_);
      supported_.emplace(key, supported);
    }
    if (supported) {
      return tvm::ffi::Function::GetGlobal(key);
    }
    if (warn_cutedsl_backend_not_chosen_) {
      NVTE_WARN("TVM-FFI kernel for config `", key, "` is not supported.");
    }
    return std::nullopt;
  }

 private:
  ~TVMFFICentral() = default;
  TVMFFICentral()
      : cutedsl_backend_enabled_(is_cutedsl_backend_enabled()),
        warn_cutedsl_backend_not_chosen_(warn_if_cutedsl_backend_not_chosen()) {}
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

  const bool cutedsl_backend_enabled_;
  const bool warn_cutedsl_backend_not_chosen_;
  std::shared_mutex mutex_;
  // Per-config support decision (cfg.to_key() -> supported). Holds NO Python-
  // backed handles, so it is safe to destroy at static teardown — the kernels
  // live in the tvm-ffi registry, owned and released by tvm-ffi itself.
  std::unordered_map<std::string, bool> supported_;
};

}  // namespace tvm_ffi_bridge
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_TVM_FFI_BRIDGE_H_
