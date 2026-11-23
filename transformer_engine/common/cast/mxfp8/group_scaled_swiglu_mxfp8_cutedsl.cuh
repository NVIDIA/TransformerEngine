/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_scaled_swiglu_mxfp8_cutedsl.cuh
 *  \brief CuTeDSL backend of group_scaled_swiglu_mxfp8.cuh.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_SCALED_SWIGLU_MXFP8_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_SCALED_SWIGLU_MXFP8_CUTEDSL_CUH_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"
#include "../../util/cutedsl_utils.h"
#include "../../utils.cuh"                   // ShapeRepresentation
#include "group_quantize_mxfp8_cutedsl.cuh"  // shape_rep_to_str, kMaxGroupTensors
#include "group_scaled_swiglu_mxfp8.cuh"     // check_group_scaled_swiglu_args

namespace transformer_engine {
namespace cutedsl_backend {

using namespace tvm_ffi_bridge;

struct MXFP8GroupScaledSwiGLUConfig {
  static constexpr const char *kEntrypointName = "get_mxfp8_group_scaled_swiglu_function";

  DType dtype;                    // The input format
  DType fp8_dtype;                // The fp8 output format
  ShapeRepresentation shape_rep;  // SAME_BOTH_DIMS or VARYING_FIRST_DIM
  bool swizzled;                  // If the scales are written in the GEMM-swizzled layout
  GatedActivation activation;     // kSwiGLU or kClampedSwiGLU
  uint32_t sm_arch = static_cast<uint32_t>(cuda::sm_arch());

  // Bit layout: dtype [3:0], fp8_dtype [7:4], shape_rep [9:8], swizzled [10], activation [16:11]
  // and SM architecture [25:17].
  uint32_t to_id() const {
    static_assert(static_cast<uint32_t>(DType::kNumTypes) <= 16,
                  "DType no longer fits in the 4 bits to_id() gives it.");
    static_assert(ShapeRepresentation::VARYING_BOTH_DIMS < 4,
                  "ShapeRepresentation no longer fits in the 2 bits to_id() gives it.");
    static_assert(static_cast<uint32_t>(GatedActivation::kNumTypes) <= 64,
                  "GatedActivation no longer fits in the 6 bits to_id() gives it.");
    NVTE_CHECK(sm_arch < 512, "SM architecture no longer fits in the 9 bits to_id() gives it.");
    return static_cast<uint32_t>(dtype) | (static_cast<uint32_t>(fp8_dtype) << 4) |
           (static_cast<uint32_t>(shape_rep) << 8) | (static_cast<uint32_t>(swizzled) << 10) |
           (static_cast<uint32_t>(activation) << 11) | (sm_arch << 17);
  }

  std::optional<tvm::ffi::Function> get_kernel() const {
    static TVMFFIConfigCache &cache = TVMFFIConfigCache::create();
    return cache.get_or_load(*this);
  }

  std::string to_key() const {
    std::string key;
    // longest: cutedsl_group_mxfp8_scaled_swiglu_smXXX_BFloat16_Float8E4M3_vfd_1_clamped_swiglu
    key.reserve(96);
    key.append("cutedsl_group_mxfp8_scaled_swiglu_sm")
        .append(std::to_string(sm_arch))
        .append("_")
        .append(to_string(dtype))
        .append("_")
        .append(to_string(fp8_dtype))
        .append("_")
        .append(shape_rep_to_str(shape_rep))
        .append("_")
        .append(swizzled ? "1" : "0")
        .append("_")
        .append(gated_activation_to_str(activation));
    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result = (*entrypoint)(
        tvm::ffi::String(fn_name), tvm::ffi::String(to_string(dtype)),
        tvm::ffi::String(to_string(fp8_dtype)), tvm::ffi::String(shape_rep_to_str(shape_rep)),
        swizzled, tvm::ffi::String(gated_activation_to_str(activation)));
    return result.try_cast<bool>().value_or(false);
  }
};

// The activation the kernel fuses, identified by the CUDA kernel's OP.
template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr GatedActivation group_scaled_swiglu_activation() {
  return GatedActivation::kUnsupported;
}

template <>
constexpr GatedActivation group_scaled_swiglu_activation<Empty, silu_approx_x2<float, float>>() {
  return GatedActivation::kSwiGLU;
}

template <>
constexpr GatedActivation
group_scaled_swiglu_activation<ClampedSwiGLUParam, clamped_silu_approx_x2<float, float>>() {
  return GatedActivation::kClampedSwiGLU;
}

// Signature mirrors mxfp8::group_scaled_swiglu. Returns false to fall back to the CUDA kernel.
template <typename ParamOP, float (*OP)(float, const ParamOP &)>
bool mxfp8_group_scaled_swiglu_cutedsl(const GroupedTensor *input, const Tensor *prob,
                                       const Tensor *noop, GroupedTensor *output, const ParamOP &p,
                                       cudaStream_t stream) {
  if (!TVMFFICentral::getInstance().get_cutedsl_backend_enabled()) {
    report_cutedsl_not_chosen("the CuTeDSL backend is disabled.");
    return false;
  }
  constexpr GatedActivation activation = group_scaled_swiglu_activation<ParamOP, OP>();
  if constexpr (activation == GatedActivation::kUnsupported) {
    report_cutedsl_not_chosen("the fused activation is not supported.");
    return false;
  } else {
    // Invalid arguments raise the CUDA launcher's errors.
    const ShapeRepresentation shape_rep =
        dispatch::mxfp8::check_group_scaled_swiglu_args(input, prob, noop, output);
    const size_t num_tensors = input->num_tensors;
    const size_t N = output->logical_shape.data[0];
    const size_t H = output->logical_shape.data[1];
    const bool swizzled = output->with_gemm_swizzled_scales;

    // The kernel loads 16 bytes per row of each half and stores whole rows of 8 or 4 FP8 values.
    if (H % 16 != 0) {
      report_cutedsl_not_chosen("the last dimension of the output is not divisible by 16.");
      return false;
    }
    if (!is_aligned_ptr(input->data.dptr, 16) ||
        !is_aligned_ptr(output->columnwise_data.dptr, 16) ||
        !is_aligned_ptr(output->columnwise_scale_inv.dptr, 16)) {
      report_cutedsl_not_chosen("the input, output or scale buffer is not 16-byte aligned.");
      return false;
    }
    const bool vfd = shape_rep == ShapeRepresentation::VARYING_FIRST_DIM;
    if (vfd) {
      // The kernel reads the end of the group from the offsets, as the CUDA kernel does.
      NVTE_CHECK(output->tensor_offsets.has_data(),
                 "group_scaled_swiglu with per-expert first dims requires tensor_offsets.");
      NVTE_CHECK(output->first_dims.has_data(),
                 "group_scaled_swiglu with per-expert first dims requires first_dims.");
      // The search over the offsets for a chunk's expert is sized for this many.
      if (swizzled && num_tensors > kMaxGroupTensors) {
        report_cutedsl_not_chosen("the group size ", num_tensors, " exceeds ", kMaxGroupTensors,
                                  ".");
        return false;
      }
    }

    const MXFP8GroupScaledSwiGLUConfig config{/*dtype=*/input->dtype(),
                                              /*fp8_dtype=*/output->dtype(),
                                              /*shape_rep=*/shape_rep,
                                              /*swizzled=*/swizzled,
                                              /*activation=*/activation};
    std::optional<tvm::ffi::Function> fn = config.get_kernel();
    if (!fn.has_value()) {
      return false;
    }
    checkCuDriverContext(stream);
    if (N == 0 || H == 0) {
      return true;
    }

    const int32_t device_index = transformer_engine::cuda::current_device();
    DLTensorWrapper mX(make_basic_tensor(input->data.dptr, input->dtype(), {N, 2 * H}), true,
                       device_index);
    DLTensorWrapper mProb(
        make_basic_tensor(prob->data.dptr, prob->data.dtype, {prob->data.numel()}), false,
        device_index);
    DLTensorWrapper mO(
        make_basic_tensor(output->columnwise_data.dptr, output->columnwise_data.dtype, {N, H}),
        true, device_index);
    const SimpleTensor &scales = output->columnwise_scale_inv;
    DLTensorWrapper mS(make_basic_tensor(scales.dptr, scales.dtype, {scales.numel()}), false,
                       device_index);
    DLTensorWrapper mOffsets, mFirstDims;
    if (vfd) {
      mOffsets = DLTensorWrapper(
          make_basic_tensor(output->tensor_offsets.dptr, DType::kInt64, {num_tensors + 1}), false,
          device_index);
      mFirstDims =
          DLTensorWrapper(make_basic_tensor(output->first_dims.dptr, DType::kInt64, {num_tensors}),
                          false, device_index);
    }

    // Plain SwiGLU's kernel is compiled with None for the clamp parameters.
    tvm::ffi::Any limit = nullptr, alpha = nullptr, glu_linear_offset = nullptr;
    if constexpr (std::is_same_v<ParamOP, ClampedSwiGLUParam>) {
      limit = p.limit;
      alpha = p.alpha;
      glu_linear_offset = p.glu_linear_offset;
    }

    (*fn)(&mX, &mProb, &mO, &mS, &mOffsets, &mFirstDims, static_cast<int32_t>(num_tensors),
          noop->data.dptr, limit, alpha, glu_linear_offset, static_cast<void *>(stream));
    return true;
  }
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_SCALED_SWIGLU_MXFP8_CUTEDSL_CUH_
