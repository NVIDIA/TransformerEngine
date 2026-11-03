/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file gated_mxfp8_cutedsl.cuh
 *  \brief CuTeDSL backend of gated_mxfp8.cuh.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GATED_MXFP8_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GATED_MXFP8_CUTEDSL_CUH_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"
#include "../../util/cutedsl_utils.h"

namespace transformer_engine {
namespace cutedsl_backend {

using namespace tvm_ffi_bridge;

namespace detail {

// TMA needs 16-byte aligned global addresses.
inline bool is_16B_aligned(const void *ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; }

}  // namespace detail

struct MXFP8GatedQuantConfig {
  static constexpr const char *kEntrypointName = "get_mxfp8_gated_quantization_function";

  DType dtype;      // The input format
  DType fp8_dtype;  // The fp8 output format
  bool rowwise;     // If quantize rowwisely
  bool colwise;     // If quantize columnwisely
  bool swizzled;    // If the scale output is used for cudnn's swizzled layout
  bool is_bwd;      // Backward (dgated) rather than forward
  GatedActivation activation;
  uint32_t sm_arch = static_cast<uint32_t>(cuda::sm_arch());

  // Bit layout: dtype [3:0], fp8_dtype [7:4], flags [11:8], activation [17:12] and
  // SM architecture [26:18].
  uint32_t to_id() const {
    static_assert(static_cast<uint32_t>(DType::kNumTypes) <= 16,
                  "DType no longer fits in the 4 bits to_id() gives it.");
    static_assert(static_cast<uint32_t>(GatedActivation::kNumTypes) <= 64,
                  "GatedActivation no longer fits in the 6 bits to_id() gives it.");
    NVTE_CHECK(sm_arch < 512, "SM architecture no longer fits in the 9 bits to_id() gives it.");
    return static_cast<uint32_t>(dtype) | (static_cast<uint32_t>(fp8_dtype) << 4) |
           (static_cast<uint32_t>(rowwise) << 8) | (static_cast<uint32_t>(colwise) << 9) |
           (static_cast<uint32_t>(swizzled) << 10) | (static_cast<uint32_t>(is_bwd) << 11) |
           (static_cast<uint32_t>(activation) << 12) | (sm_arch << 18);
  }

  std::optional<tvm::ffi::Function> get_kernel() const {
    static TVMFFIConfigCache &cache = TVMFFIConfigCache::create();
    return cache.get_or_load(*this);
  }

  std::string to_key() const {
    std::string key;
    key.reserve(80);
    key.append("cutedsl_mxfp8_gated_sm")
        .append(std::to_string(sm_arch))
        .append("_")
        .append(to_string(dtype))
        .append("_")
        .append(to_string(fp8_dtype))
        .append("_")
        .append(rowwise ? "1" : "0")
        .append("_")
        .append(colwise ? "1" : "0")
        .append("_")
        .append(swizzled ? "1" : "0")
        .append("_")
        .append(is_bwd ? "1" : "0")
        .append("_")
        .append(gated_activation_to_str(activation));
    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result =
        (*entrypoint)(tvm::ffi::String(fn_name), tvm::ffi::String(to_string(dtype)),
                      tvm::ffi::String(to_string(fp8_dtype)), rowwise, colwise, swizzled, is_bwd,
                      tvm::ffi::String(gated_activation_to_str(activation)));
    return result.try_cast<bool>().value_or(false);
  }
};

// Signature mirrors mxfp8::quantize_gated. Returns false to fall back to the CUDA kernel.
template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
bool mxfp8_quantize_gated_cutedsl(const Tensor &gated_input, const Tensor &grad, Tensor *output,
                                  const ParamOP &p, cudaStream_t stream) {
  if (!TVMFFICentral::getInstance().get_cutedsl_backend_enabled()) {
    maybe_warn_cutedsl_not_chosen("the CuTeDSL backend is disabled.");
    return false;
  }
  constexpr GatedActivation activation = gated_activation_func_to_enum<ParamOP, ActOP, DActOP>();
  if constexpr (activation == GatedActivation::kUnsupported) {
    maybe_warn_cutedsl_not_chosen("the gated activation is not supported.");
    return false;
  } else {
    const size_t rows = gated_input.flat_first_dim();
    const size_t cols = gated_input.flat_last_dim() / 2;
    const MXFP8GatedQuantConfig config{/*dtype=*/gated_input.dtype(),
                                       /*fp8_dtype=*/output->dtype(),
                                       /*rowwise=*/output->has_data(),
                                       /*colwise=*/output->has_columnwise_data(),
                                       /*swizzled=*/output->with_gemm_swizzled_scales,
                                       /*is_bwd=*/IS_BWD,
                                       /*activation=*/activation};
    // The kernel offsets the gate half's swizzled scales by a fixed amount, which only matches
    // the swizzled layout when the half spans whole 128-column scale tiles.
    if (IS_BWD && config.swizzled && cols % 128 != 0) {
      maybe_warn_cutedsl_not_chosen(
          "backward gated activations with GEMM-swizzled scales need the last dimension of the "
          "output halves to be divisible by 128.");
      return false;
    }
    bool aligned = detail::is_16B_aligned(gated_input.data.dptr) &&
                   (!config.rowwise || detail::is_16B_aligned(output->data.dptr)) &&
                   (!config.colwise || detail::is_16B_aligned(output->columnwise_data.dptr));
    if constexpr (IS_BWD) {
      aligned = aligned && detail::is_16B_aligned(grad.data.dptr);
    }
    if (!aligned) {
      maybe_warn_cutedsl_not_chosen("an input or output is not 16-byte aligned.");
      return false;
    }

    std::optional<tvm::ffi::Function> fn = config.get_kernel();
    if (!fn.has_value()) {
      return false;
    }
    checkCuDriverContext(stream);

    // Sanity checks and swizzled-scale padding, mirroring mxfp8::quantize_gated
    if (config.rowwise) {
      NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
    }
    if (config.colwise) {
      NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
    }
    if (config.swizzled && (cols % 128 != 0 || rows % 128 != 0)) {
      if (config.rowwise) {
        NVTE_CHECK_CUDA(cudaMemsetAsync(output->scale_inv.dptr, 0,
                                        output->scale_inv.buffer_size_bytes(), stream));
      }
      if (config.colwise) {
        NVTE_CHECK_CUDA(cudaMemsetAsync(output->columnwise_scale_inv.dptr, 0,
                                        output->columnwise_scale_inv.buffer_size_bytes(), stream));
      }
    }

    // The activation's own parameters are runtime arguments; the kernel is compiled with None in
    // the slots of the others, so they are passed as None (nullptr).
    tvm::ffi::Any limit = nullptr, alpha = nullptr, glu_linear_offset = nullptr, beta1 = nullptr,
                  beta2 = nullptr;
    if constexpr (std::is_same_v<ParamOP, ClampedSwiGLUParam>) {
      limit = p.limit;
      alpha = p.alpha;
      glu_linear_offset = p.glu_linear_offset;
    } else if constexpr (std::is_same_v<ParamOP, SiTUGLUParam>) {
      beta1 = p.beta1;
      beta2 = p.beta2;
    }

    const int32_t device_index = transformer_engine::cuda::current_device();
    DLTensorWrapper mGatedInput(gated_input.data, true, device_index);
    DLTensorWrapper mGrad, mO_row, mS_row, mO_col, mS_col;
    if constexpr (IS_BWD) {
      mGrad = DLTensorWrapper(grad.data, true, device_index);
    }
    if (config.rowwise) {
      mO_row = DLTensorWrapper(output->data, true, device_index);
      mS_row = DLTensorWrapper(output->scale_inv, true, device_index);
    }
    if (config.colwise) {
      mO_col = DLTensorWrapper(output->columnwise_data, true, device_index);
      mS_col = DLTensorWrapper(output->columnwise_scale_inv, true, device_index);
    }
    (*fn)(&mGatedInput, &mGrad, &mO_row, &mS_row, &mO_col, &mS_col, limit, alpha, glu_linear_offset,
          beta1, beta2, static_cast<void *>(stream));
    return true;
  }
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GATED_MXFP8_CUTEDSL_CUH_
