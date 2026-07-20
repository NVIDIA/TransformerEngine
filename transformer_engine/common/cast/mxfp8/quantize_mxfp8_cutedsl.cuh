/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_QUANTIZE_MXFP8_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_QUANTIZE_MXFP8_CUTEDSL_CUH_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

#include <cstddef>
#include <optional>
#include <string>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"
#include "../../util/math.h"
#include "../core/common.cuh"  // dispatch::common::reduce_dbias
#include "quantize_mxfp8.cuh"  // dispatch::mxfp8::quantize_kernel::zero_scales_kernel

namespace transformer_engine {
namespace cutedsl_backend {

// Activation, te_dtype_to_str, activation_to_str, DLTensorWrapper, TVMFFICentral
// all live in transformer_engine::tvm_ffi_bridge (tvm_ffi_bridge.h).
using namespace tvm_ffi_bridge;

struct MXFP8QuantConfig {
  static constexpr const char *kEntrypointName = "get_mxfp8_quantization_function";

  DType dtype;              // The input format
  DType fp8_dtype;          // The fp8 output format
  bool rowwise;             // If quantize rowwisely
  bool colwise;             // If quantize columnwisely
  bool swizzled;            // If the scale output is used for cudnn's swizzled layout
  bool with_amax;           // If the kernel should return the amax
  bool with_dbias = false;  // If the dbias is computated (via the workspace tensor)
  bool with_dact = false;   // If an activation derivative operation is fused
  bool with_act = false;    // If an activation operation is fused
  bool with_noop = false;   // If a non-nullptr noop tensor is passed to the kernel
  Activation activation = Activation::kNone;

  std::string to_key() const {
    std::string key;
    key.reserve(56);
    key.append("cutedsl_mxfp8_")
        .append(te_dtype_to_str(dtype))
        .append("_")
        .append(te_dtype_to_str(fp8_dtype))
        .append("_")
        .append(rowwise ? "1" : "0")
        .append("_")
        .append(colwise ? "1" : "0")
        .append("_")
        .append(swizzled ? "1" : "0")
        .append("_")
        .append(with_amax ? "1" : "0")
        .append("_")
        .append(with_dbias ? "1" : "0")
        .append("_")
        .append(with_dact ? "1" : "0")
        .append("_")
        .append(with_act ? "1" : "0")
        .append("_")
        .append(with_noop ? "1" : "0")
        .append("_")
        .append(activation_to_str(activation));
    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result =
        (*entrypoint)(tvm::ffi::String(fn_name), tvm::ffi::String(te_dtype_to_str(dtype)),
                      tvm::ffi::String(te_dtype_to_str(fp8_dtype)), rowwise, colwise, swizzled,
                      with_amax, with_dbias, with_dact, with_act, with_noop,
                      tvm::ffi::String(activation_to_str(activation)));
    return result.try_cast<bool>().value_or(false);
  }
};

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
struct MXFP8QuantFused {
  static constexpr Activation activation = Activation::kNone;
  // No fused activation / activation derivative op: plain quantize
  static constexpr bool supported = (OP == nullptr) && !IS_DACT && !IS_ACT;
};
template <>
struct MXFP8QuantFused<false, false, true, Empty, relu<fp32, fp32>> {
  static constexpr Activation activation = Activation::kReLU;
  static constexpr bool supported = true;
};
template <>
struct MXFP8QuantFused<false, false, true, Empty, gelu<fp32, fp32>> {
  static constexpr Activation activation = Activation::kGeLU;
  static constexpr bool supported = true;
};
template <>
struct MXFP8QuantFused<false, false, true, Empty, silu<fp32, fp32>> {
  static constexpr Activation activation = Activation::kSiLU;
  static constexpr bool supported = true;
};
template <>
struct MXFP8QuantFused<false, false, true, Empty, qgelu<fp32, fp32>> {
  static constexpr Activation activation = Activation::kQGeLU;
  static constexpr bool supported = true;
};
template <>
struct MXFP8QuantFused<false, false, true, Empty, srelu<fp32, fp32>> {
  static constexpr Activation activation = Activation::kSReLU;
  static constexpr bool supported = true;
};
template <bool IS_DBIAS>
struct MXFP8QuantFused<IS_DBIAS, true, false, Empty, drelu<fp32, fp32>> {
  static constexpr Activation activation = Activation::kDReLU;
  static constexpr bool supported = true;
};
template <bool IS_DBIAS>
struct MXFP8QuantFused<IS_DBIAS, true, false, Empty, dgelu<fp32, fp32>> {
  static constexpr Activation activation = Activation::kDGeLU;
  static constexpr bool supported = true;
};
template <bool IS_DBIAS>
struct MXFP8QuantFused<IS_DBIAS, true, false, Empty, dsilu<fp32, fp32>> {
  static constexpr Activation activation = Activation::kDSiLU;
  static constexpr bool supported = true;
};
template <bool IS_DBIAS>
struct MXFP8QuantFused<IS_DBIAS, true, false, Empty, dqgelu<fp32, fp32>> {
  static constexpr Activation activation = Activation::kDQGeLU;
  static constexpr bool supported = true;
};
template <bool IS_DBIAS>
struct MXFP8QuantFused<IS_DBIAS, true, false, Empty, dsrelu<fp32, fp32>> {
  static constexpr Activation activation = Activation::kDSReLU;
  static constexpr bool supported = true;
};

// Signature mirrors mxfp8::quantize (input, act_input, noop, output, dbias,
// workspace, stream). Returns false to fall back to the CUDA kernel.
inline bool mxfp8_quantize_cutedsl(const MXFP8QuantConfig &config, const Tensor *input_tensor,
                                   const Tensor *act_input_tensor, const Tensor *noop_tensor,
                                   Tensor *output_tensor, Tensor *dbias_tensor,
                                   Tensor *workspace_tensor, cudaStream_t stream) {
  constexpr size_t kCuTeDSLMXFP8ShapeAlignment = 32;
  const size_t flat_m = input_tensor->flat_first_dim();
  const size_t flat_n = input_tensor->flat_last_dim();
  if (flat_m % kCuTeDSLMXFP8ShapeAlignment != 0 || flat_n % kCuTeDSLMXFP8ShapeAlignment != 0) {
    return false;
  }

  std::optional<tvm::ffi::Function> mxfp8_quant_func_opt =
      tvm_ffi_bridge::TVMFFICentral::getInstance().lazyload_function(config);
  if (!mxfp8_quant_func_opt.has_value()) {
    return false;
  }

  // When only WITH_DBIAS is true, we use a larger tile size (align with CUDA C++ implementation)
  const bool cast_dbias_only = config.with_dbias && !config.with_dact && !config.with_act;
  const size_t chunk_rows = cast_dbias_only ? 128 : 64;  // input rows reduced per CTA
  // Each CTA writes one partial-dbias row, so the workspace (and the cross-CTA
  // reduction below) has ceil(M / chunk_rows) rows.
  const size_t workspace_rows = (flat_m + chunk_rows - 1) / chunk_rows;

  // dbias workspace-size query, mirroring mxfp8::quantize: the framework first
  // calls with an unallocated workspace to learn its shape, allocates a buffer of
  // that shape, then calls again to run. The kernel writes per-row-block partial
  // dbias into this workspace; reducing it to the final dbias is a separate step.
  if (config.with_dbias && workspace_tensor != nullptr && workspace_tensor->data.dptr == nullptr) {
    workspace_tensor->data.shape = {workspace_rows, flat_n};
    workspace_tensor->data.dtype = DType::kFloat32;
    return true;
  }

  // Zero out swizzled scales if padding is needed.
  // Mirrors the CUDA C++ implementation in quantize_mxfp8.cuh. Skip when noop flag is set.
  if (config.swizzled && (flat_m % 128 != 0 || flat_n % 128 != 0)) {
    const float *noop_ptr = (noop_tensor != nullptr && noop_tensor->data.dptr != nullptr)
                                ? reinterpret_cast<const float *>(noop_tensor->data.dptr)
                                : nullptr;
    constexpr size_t zero_threads = 256;
    if (output_tensor->has_data()) {
      const size_t size_bytes = output_tensor->scale_inv.buffer_size_bytes();
      if (size_bytes > 0) {
        const size_t zero_blocks = (size_bytes + zero_threads - 1) / zero_threads;
        dispatch::mxfp8::quantize_kernel::
            zero_scales_kernel<<<zero_blocks, zero_threads, 0, stream>>>(
                reinterpret_cast<uint8_t *>(output_tensor->scale_inv.dptr), size_bytes, noop_ptr);
        NVTE_CHECK_CUDA(cudaGetLastError());
      }
    }
    if (output_tensor->has_columnwise_data()) {
      const size_t size_bytes = output_tensor->columnwise_scale_inv.buffer_size_bytes();
      if (size_bytes > 0) {
        const size_t zero_blocks = (size_bytes + zero_threads - 1) / zero_threads;
        dispatch::mxfp8::quantize_kernel::
            zero_scales_kernel<<<zero_blocks, zero_threads, 0, stream>>>(
                reinterpret_cast<uint8_t *>(output_tensor->columnwise_scale_inv.dptr), size_bytes,
                noop_ptr);
        NVTE_CHECK_CUDA(cudaGetLastError());
      }
    }
  }

  // Data tensors auto-flatten to 2D (DLTensorWrapper's default), matching the
  // kernel's flat (rows, cols) view; scale/amax/noop are rank <= 2 and pass through.
  tvm_ffi_bridge::DLTensorWrapper mX(input_tensor->data);
  tvm_ffi_bridge::DLTensorWrapper mO_row, mS_row, mO_col, mS_col, mAmax, mNoop, mActInput,
      mWorkspace;
  if (output_tensor->has_data()) {
    mO_row = tvm_ffi_bridge::DLTensorWrapper(output_tensor->data);
    mS_row = tvm_ffi_bridge::DLTensorWrapper(output_tensor->scale_inv);
  }
  if (output_tensor->has_columnwise_data()) {
    mO_col = tvm_ffi_bridge::DLTensorWrapper(output_tensor->columnwise_data);
    mS_col = tvm_ffi_bridge::DLTensorWrapper(output_tensor->columnwise_scale_inv);
  }
  if (output_tensor->amax.dptr != nullptr)
    mAmax = tvm_ffi_bridge::DLTensorWrapper(output_tensor->amax);
  if (noop_tensor != nullptr && noop_tensor->data.dptr != nullptr)
    mNoop = tvm_ffi_bridge::DLTensorWrapper(noop_tensor->data);
  if (act_input_tensor != nullptr && act_input_tensor->data.dptr != nullptr)
    mActInput = tvm_ffi_bridge::DLTensorWrapper(act_input_tensor->data);
  if (workspace_tensor != nullptr && workspace_tensor->data.dptr != nullptr)
    mWorkspace = tvm_ffi_bridge::DLTensorWrapper(workspace_tensor->data);
  // stream is a tvm-ffi opaque "handle"; pass the CUDA stream as void*.
  (*mxfp8_quant_func_opt)(&mX, &mO_row, &mS_row, &mO_col, &mS_col, &mAmax, &mNoop, &mActInput,
                          &mWorkspace, static_cast<void *>(stream));

  // If WITH_DBIAS, reduce the workspace partial dbias in CUDA C++ for now.
  // This will not be affected by the noop flag, which is aligned with the CUDA C++ implementation
  if (config.with_dbias) {
    const float *workspace_ptr = reinterpret_cast<const float *>(workspace_tensor->data.dptr);
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        input_tensor->dtype(), IType,
        dispatch::common::reduce_dbias<IType>(workspace_ptr, dbias_tensor, workspace_rows, flat_n,
                                              stream);)  // NOLINT(*)
  }
  return true;
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
bool mxfp8_quantize_cutedsl(const Tensor *input_tensor, const Tensor *act_input_tensor,
                            const Tensor *noop_tensor, Tensor *output_tensor, Tensor *dbias_tensor,
                            Tensor *workspace_tensor, cudaStream_t stream) {
  using Fused = MXFP8QuantFused<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>;
  if constexpr (!Fused::supported) {
    return false;
  } else {
    const bool with_noop = noop_tensor != nullptr && noop_tensor->data.dptr != nullptr;
    const MXFP8QuantConfig config{/*dtype=*/input_tensor->dtype(),
                                  /*fp8_dtype=*/output_tensor->dtype(),
                                  /*rowwise=*/output_tensor->has_data(),
                                  /*colwise=*/output_tensor->has_columnwise_data(),
                                  /*swizzled=*/output_tensor->with_gemm_swizzled_scales,
                                  /*with_amax=*/output_tensor->amax.dptr != nullptr,
                                  /*with_dbias=*/IS_DBIAS,
                                  /*with_dact=*/IS_DACT,
                                  /*with_act=*/IS_ACT,
                                  /*with_noop=*/with_noop,
                                  /*activation=*/Fused::activation};
    checkCuDriverContext(stream);
    // Sanity checks, mirroring mxfp8::quantize in quantize_mxfp8.cuh
    if (config.rowwise) {
      NVTE_CHECK(output_tensor->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
    }
    if (config.colwise) {
      NVTE_CHECK(output_tensor->columnwise_scale_inv.dptr != nullptr,
                 "Columnwise scaling tensor must be allocated");
    }
    if (noop_tensor != nullptr) {
      CheckNoopTensor(*noop_tensor, "cast_noop");
    }
    if constexpr (IS_DBIAS) {
      NVTE_CHECK(dbias_tensor != nullptr, "DBias must be a tensor.");
      NVTE_CHECK(dbias_tensor->data.dtype == input_tensor->dtype(),
                 "DBias must have the same type as input.");
      NVTE_CHECK(dbias_tensor->data.shape == Shape{input_tensor->flat_last_dim()},
                 "Wrong shape of DBias.");
      NVTE_CHECK(workspace_tensor != nullptr, "Workspace must be a tensor.");
    }
    return mxfp8_quantize_cutedsl(config, input_tensor, act_input_tensor, noop_tensor,
                                  output_tensor, dbias_tensor, workspace_tensor, stream);
  }
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_QUANTIZE_MXFP8_CUTEDSL_CUH_
