/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_TVM_FFI_BRIDGE_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_TVM_FFI_BRIDGE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <c10/core/ScalarType.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include "transformer_engine/transformer_engine.h"
#include "util/logging.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// dtype conversion helpers — overload resolution picks by argument type.
// ---------------------------------------------------------------------------

// NOTE: at::Tensor -> DLTensor goes through at::toDLPackNonOwning, which fills
// the DLDataType from the torch dtype automatically, so no c10::ScalarType
// overload is needed. Only TE's own NVTEBasicTensor (NVTEDType, no DLPack
// support) requires this manual mapping.
inline DLDataType convert_to_dltype(NVTEDType type) {
  switch (type) {
    case kNVTEFloat32:    return DLDataType{kDLFloat,  32, 1};
    case kNVTEFloat16:    return DLDataType{kDLFloat,  16, 1};
    case kNVTEBFloat16:   return DLDataType{kDLBfloat, 16, 1};
    case kNVTEByte:       return DLDataType{kDLUInt,    8, 1};
    case kNVTEInt32:      return DLDataType{kDLInt,    32, 1};
    case kNVTEInt64:      return DLDataType{kDLInt,    64, 1};
    // FP8 / E8M0 → raw 1-byte uint; the kernel interprets the bits.
    case kNVTEFloat8E4M3: return DLDataType{kDLUInt,    8, 1};
    case kNVTEFloat8E5M2: return DLDataType{kDLUInt,    8, 1};
    case kNVTEFloat8E8M0: return DLDataType{kDLUInt,    8, 1};
    default: NVTE_ERROR("unsupported NVTEDType: ", static_cast<int>(type));
  }
}

// ---------------------------------------------------------------------------
// DLTensorWrapper — DLTensor with managed shape/strides storage.
//
// Subclassing DLTensor (a POD C struct) lets the wrapper IS-A DLTensor: you
// can take its address and pass it directly to `tvm::ffi::TensorView`. The
// shape/strides arrays the base struct points at are either borrowed from a
// PyTorch tensor (zero copy) or owned by the wrapper itself (when built
// from an NVTE tensor that doesn't store them in int64_t form).
// ---------------------------------------------------------------------------
class DLTensorWrapper : public DLTensor {
 public:
  // Zero-copy borrow via torch's own non-owning DLPack export: fills our
  // base DLTensor in place (data/shape/strides/dtype/device/byte_offset)
  // using torch's canonical field extraction — no heap alloc, no deleter,
  // no refcount. shape/strides point into the at::Tensor's internal arrays,
  // so the caller must keep `tensor` alive through any use of this wrapper.
  explicit DLTensorWrapper(const at::Tensor &tensor) {
    NVTE_CHECK(tensor.defined(), "DLTensorWrapper: undefined at::Tensor");
    at::toDLPackNonOwning(tensor, static_cast<DLTensor *>(this));
    this->numel_ = static_cast<int64_t>(tensor.numel());
  }

  // NVTEBasicTensor stores shape as size_t and has no strides. We allocate
  // owned int64 buffers for both: copy the shape, synthesize row-major
  // contiguous strides (TE tensors are always contiguous).
  DLTensorWrapper(const NVTEBasicTensor &tensor, int32_t device_index) {
    const int n = static_cast<int>(tensor.shape.ndim);
    shape_buf_   = std::make_unique<int64_t[]>(n);
    strides_buf_ = std::make_unique<int64_t[]>(n);
    int64_t stride = 1;
    for (int i = n - 1; i >= 0; --i) {
      shape_buf_[i]   = static_cast<int64_t>(tensor.shape.data[i]);
      strides_buf_[i] = stride;
      stride *= shape_buf_[i];
    }
    this->numel_      = stride;  // product of all dims
    this->data        = tensor.data_ptr;
    this->device      = DLDevice{kDLCUDA, device_index};
    this->ndim        = n;
    this->dtype       = convert_to_dltype(tensor.dtype);
    this->shape       = shape_buf_.get();
    this->strides     = strides_buf_.get();
    this->byte_offset = 0;
  }

  ~DLTensorWrapper() = default;
  DLTensorWrapper(const DLTensorWrapper &) = delete;
  DLTensorWrapper &operator=(const DLTensorWrapper &) = delete;
  DLTensorWrapper(DLTensorWrapper &&) = default;
  DLTensorWrapper &operator=(DLTensorWrapper &&) = default;

  // Number of elements (product of shape), cached at construction. For 1-byte
  // dtypes (FP8 / E8M0 / E4M3) this equals the byte count.
  int64_t numel() const { return this->numel_; }

 private:
  int64_t numel_ = 0;
  std::unique_ptr<int64_t[]> shape_buf_;
  std::unique_ptr<int64_t[]> strides_buf_;
};

// ---------------------------------------------------------------------------
// Turn an optionally-present DLTensorWrapper into a tvm-ffi call argument:
//   present -> TensorView over its (borrowed) DLTensor
//   absent  -> None
// tvm-ffi's TypeTraits<Optional<T>> packs an empty Optional as None and a
// present one as the inner T (here, the TensorView). The returned Optional
// holds the TensorView by value, so when passed as a call-site argument it
// stays alive through the whole `fn(...)` expression (including CallPacked).
// The DLTensorWrapper it views must itself outlive the call (keep it as a
// named local — its synthesized shape/stride buffers back the DLTensor).
// ---------------------------------------------------------------------------
inline tvm::ffi::Optional<tvm::ffi::TensorView> to_ffi_arg(
    const std::optional<DLTensorWrapper> &wrapper) {
  if (wrapper.has_value()) {
    return tvm::ffi::TensorView(static_cast<const DLTensor *>(&wrapper.value()));
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// call_tvm_ffi — resolve a global tvm-ffi function by name and call it.
//
// Each argument is auto-packed into an AnyView by tvm-ffi's variadic
// `Function::operator()`. That operator keeps the call-site argument
// temporaries alive through its internal CallPacked (they are bound to
// forwarding references whose lifetime spans the full `fn(...)` expression),
// so passing TensorView / Optional<TensorView> temporaries here is safe — no
// need to park them in named arrays the way a manual AnyView[] + CallPacked
// loop would require.
//
// We use GetGlobal (returns std::nullopt on miss) rather than
// GetGlobalRequired so we can raise a domain-specific error. `fn_name` is the
// quantizer's cache key: it encodes every compile-time (constexpr) property the
// registered CuTeDSL kernel was specialized for (dtypes, per-direction formats,
// swizzle, baked shapes, ...). A registered kernel therefore *guarantees* that
// signature. A lookup miss means no kernel was registered for this exact
// signature — i.e. the caller is asking for a constexpr configuration the
// kernel author never compiled/registered, so the constexpr guarantee is
// broken. That is a setup bug (key mismatch), not a runtime input error, so we
// fail loudly with the offending name.
// ---------------------------------------------------------------------------
template <typename... Args>
inline tvm::ffi::Any call_tvm_ffi(const std::string &fn_name, Args &&...args) {
  std::optional<tvm::ffi::Function> fn = tvm::ffi::Function::GetGlobal(fn_name);
  NVTE_CHECK(fn.has_value(),
             "No tvm-ffi kernel registered under '", fn_name,
             "'. This name is the quantizer's cache key, which encodes the "
             "kernel's compile-time (constexpr) signature; a miss means the "
             "registered kernel's constexpr guarantee does not match what is "
             "being requested (no kernel was compiled/registered for this "
             "exact configuration).");
  return (*fn)(std::forward<Args>(args)...);
}

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_TVM_FFI_BRIDGE_H_
