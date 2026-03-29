/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/gemm.h>
#include <transformer_engine/swizzle.h>

#include "../stable_common.h"
#include "common/util/cuda_runtime.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

namespace {

bool requiresScaleSwizzle(NVTEScalingMode scaling_mode) {
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING:
    case NVTE_NVFP4_1D_SCALING:
      return true;
    case NVTE_INVALID_SCALING:
      NVTE_ERROR("Invalid scaling mode for swizzling scaling factors.");
    default:
      return false;
  }
}

// Return the DType used for scale_inv given a scaling mode.
DType getScaleInvDtype(NVTEScalingMode scaling_mode) {
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING:
      return DType::kFloat8E8M0;
    case NVTE_NVFP4_1D_SCALING:
      return DType::kFloat8E4M3;
    default:
      return DType::kFloat32;
  }
}

Tensor swizzleScaleForGemm(const Tensor& data, int64_t te_dtype, const Tensor& scale_inv,
                           int64_t scaling_mode, bool columnwise = false) {
  auto tensor_dtype = static_cast<DType>(te_dtype);
  auto tensor_scaling_mode = static_cast<NVTEScalingMode>(scaling_mode);
  auto data_shape = getStableTensorShape(data);
  // FP4 data is packed (2 elements per byte). Double last dim to report
  // logical element count. For rowwise [M, K/2] → [M, K]; for columnwise
  // [K, M/2] → [K, M]. TensorWrapper::shape() handles the transpose.
  if (is_fp4_dtype(tensor_dtype) && !data_shape.empty()) {
    data_shape.back() *= 2;
  }
  DType si_dtype = getScaleInvDtype(tensor_scaling_mode);
  auto si_shape = getStableTensorShape(scale_inv);

  // Allocate output scale tensor with the same shape as input
  std::vector<int64_t> si_shape_i64(si_shape.begin(), si_shape.end());
  auto output_scale_inv = allocateStableTensor(si_shape_i64, si_dtype, data.get_device_index());

  TensorWrapper input_nvte(tensor_scaling_mode);
  TensorWrapper output_nvte(tensor_scaling_mode);
  if (columnwise) {
    input_nvte.set_columnwise_data(nullptr, tensor_dtype, data_shape);
    input_nvte.set_columnwise_scale_inv(scale_inv.data_ptr(), si_dtype, si_shape);
    output_nvte.set_columnwise_data(nullptr, tensor_dtype, data_shape);
    output_nvte.set_columnwise_scale_inv(output_scale_inv.data_ptr(), si_dtype, si_shape);
  } else {
    input_nvte.set_rowwise_data(nullptr, tensor_dtype, data_shape);
    input_nvte.set_rowwise_scale_inv(scale_inv.data_ptr(), si_dtype, si_shape);
    output_nvte.set_rowwise_data(nullptr, tensor_dtype, data_shape);
    output_nvte.set_rowwise_scale_inv(output_scale_inv.data_ptr(), si_dtype, si_shape);
  }
  output_nvte.set_with_gemm_swizzled_scales(true);

  nvte_swizzle_scaling_factors(input_nvte.data(), output_nvte.data(),
                               getCurrentCUDAStreamRaw(data.get_device_index()));

  return output_scale_inv;
}

// Build a TensorWrapper with both rowwise and (optionally) columnwise data/scales.
// A_data always holds the rowwise buffer with the LOGICAL shape.
// When A_colwise_data is provided it is set as columnwise_data on the wrapper,
// allowing CanonicalizeGemmInput to select the right buffer based on transa/transb.
TensorWrapper buildInputTensorWrapper(const Tensor& rowwise_data, DType te_dtype,
                                      const std::optional<Tensor>& rowwise_scale_inv,
                                      const std::optional<Tensor>& colwise_data,
                                      const std::optional<Tensor>& colwise_scale_inv,
                                      NVTEScalingMode scaling_mode) {
  DType si_dtype = getScaleInvDtype(scaling_mode);

  TensorWrapper out(scaling_mode);
  // Only set rowwise data when there is actual data (numel > 0).
  // When a Float8Tensor has columnwise-only storage (_data=None), the caller
  // passes an empty tensor here; we skip set_rowwise_data so TensorWrapper
  // has_data() returns false, and NVTE's CanonicalizeGemmInput uses the
  // columnwise buffer instead.
  // FP4 data is packed (2 elements per byte). The physical tensor has K/2
  // bytes but the logical element count is K. Report logical shape so that
  // CheckScaleTensorShape (which derives expected scale shape from
  // flat_last_dim()) and the swizzle kernel get the correct K.
  const bool fp4_packed = is_fp4_dtype(te_dtype);

  // Flatten >2D shapes to 2D for GEMM [M_dims..., K] → [M, K].
  // Only for MXFP8/DELAYED scaling where TensorWrapper::shape() returns
  // data shape as-is (no transpose).
  auto flatten2D = [](std::vector<size_t>& shape, NVTEScalingMode sm) {
    if (shape.size() <= 2) return;
    if (sm != NVTE_MXFP8_1D_SCALING && sm != NVTE_DELAYED_TENSOR_SCALING) return;
    size_t K = shape.back();
    size_t M = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) M *= shape[i];
    shape = {M, K};
  };

  if (rowwise_data.numel() > 0) {
    auto shape = getStableTensorShape(rowwise_data);
    if (fp4_packed && !shape.empty()) shape.back() *= 2;
    flatten2D(shape, scaling_mode);
    out.set_rowwise_data(rowwise_data.data_ptr(), te_dtype, shape);
  }

  if (rowwise_scale_inv.has_value() && rowwise_scale_inv->numel() > 0) {
    auto si_shape = getStableTensorShape(*rowwise_scale_inv);
    out.set_rowwise_scale_inv(rowwise_scale_inv->data_ptr(), si_dtype, si_shape);
  }

  if (colwise_data.has_value() && colwise_data->numel() > 0) {
    auto cw_shape = getStableTensorShape(*colwise_data);
    // FP4 columnwise data [K, M/2]: double last dim to get [K, M].
    // TensorWrapper::shape() transposes to [M, K], producing correct
    // flat_first_dim()=M and flat_last_dim()=K for CheckScaleTensorShape.
    if (fp4_packed && !cw_shape.empty()) cw_shape.back() *= 2;
    flatten2D(cw_shape, scaling_mode);
    out.set_columnwise_data(colwise_data->data_ptr(), te_dtype, cw_shape);
    if (colwise_scale_inv.has_value() && colwise_scale_inv->numel() > 0) {
      auto csi_shape = getStableTensorShape(*colwise_scale_inv);
      out.set_columnwise_scale_inv(colwise_scale_inv->data_ptr(), si_dtype, csi_shape);
    }
  }
  return out;
}

}  // namespace

// ============================================================================
// Core GEMM (no CommOverlap)
//
// This is the stable ABI version of nvte_cublas_gemm_v2.
// CommOverlap remains in the pybind11 module since it requires
// opaque class handles that can't cross the stable ABI boundary.
//
// The Python shim:
// 1. Extracts raw buffers from quantized A, B tensors
// 2. Pre-allocates output D (quantized or unquantized)
// 3. Calls this op for the core GEMM
// 4. Wraps output in the appropriate Python tensor type
//
// For FP8 tensors with separate rowwise/columnwise storage:
// - A_data / B_data always hold the ROWWISE (logical-shape) buffer
// - A_colwise_data / B_colwise_data (optional) hold the columnwise buffer
// - Both are set on the TensorWrapper; CanonicalizeGemmInput selects the
//   right one based on transa/transb at run time
// ============================================================================

void gemm(
    // Input A (rowwise data = logical shape; colwise optional for FP8)
    Tensor A_data, int64_t A_te_dtype, std::optional<Tensor> A_scale_inv,
    std::optional<Tensor> A_colwise_data, std::optional<Tensor> A_colwise_scale_inv,
    int64_t A_scaling_mode, bool A_with_gemm_swizzled_scales, bool transa,
    // Input B (rowwise data = logical shape; colwise optional for FP8)
    Tensor B_data, int64_t B_te_dtype, std::optional<Tensor> B_scale_inv,
    std::optional<Tensor> B_colwise_data, std::optional<Tensor> B_colwise_scale_inv,
    int64_t B_scaling_mode, bool B_with_gemm_swizzled_scales, bool transb,
    // Output D (pre-allocated)
    Tensor D_data, int64_t D_te_dtype, std::optional<Tensor> D_amax, std::optional<Tensor> D_scale,
    std::optional<Tensor> D_scale_inv, int64_t D_scaling_mode,
    // Optional bias
    std::optional<Tensor> bias, int64_t bias_type,
    // Optional pre-gelu output
    std::optional<Tensor> pre_gelu_out,
    // Workspace
    Tensor workspace,
    // Config
    bool grad, bool accumulate, bool use_split_accumulator, double alpha) {
  auto A_te = static_cast<DType>(A_te_dtype);
  auto B_te = static_cast<DType>(B_te_dtype);
  auto D_te = static_cast<DType>(D_te_dtype);
  auto A_sm = static_cast<NVTEScalingMode>(A_scaling_mode);
  auto B_sm = static_cast<NVTEScalingMode>(B_scaling_mode);
  auto D_sm = static_cast<NVTEScalingMode>(D_scaling_mode);

  auto D_shape = getStableTensorShape(D_data);

  // Auto-swizzle scales if needed (MXFP8/NVFP4, not pre-swizzled).
  // Swizzle the direction that CanonicalizeGemmInput will actually use:
  //   transa=True  → CanonicalizeGemmInput uses rowwise data → swizzle rowwise scale
  //   transa=False → CanonicalizeGemmInput uses colwise data  → swizzle colwise scale
  //   transb=False → CanonicalizeGemmInput uses rowwise data → swizzle rowwise scale
  //   transb=True  → CanonicalizeGemmInput uses colwise data  → swizzle colwise scale
  // This matches pybind: swizzle_scales_for_gemm(A, transa, !transa) / (B, !transb, transb).
  std::vector<Tensor> swizzled_scale_inverses;
  if (!A_with_gemm_swizzled_scales && requiresScaleSwizzle(A_sm)) {
    if (transa) {
      // transa=True → rowwise direction → swizzle rowwise scale
      if (A_scale_inv.has_value() && A_scale_inv->numel() > 0) {
        swizzled_scale_inverses.emplace_back(
            swizzleScaleForGemm(A_data, A_te_dtype, *A_scale_inv, A_scaling_mode));
        A_scale_inv = swizzled_scale_inverses.back();
      }
    } else {
      // transa=False → colwise direction → swizzle colwise scale
      if (A_colwise_data.has_value() && A_colwise_scale_inv.has_value() &&
          A_colwise_data->numel() > 0 && A_colwise_scale_inv->numel() > 0) {
        swizzled_scale_inverses.emplace_back(
            swizzleScaleForGemm(*A_colwise_data, A_te_dtype, *A_colwise_scale_inv, A_scaling_mode,
                                /*columnwise=*/true));
        A_colwise_scale_inv = swizzled_scale_inverses.back();
      }
    }
    A_with_gemm_swizzled_scales = true;
  }
  if (!B_with_gemm_swizzled_scales && requiresScaleSwizzle(B_sm)) {
    if (!transb) {
      // transb=False → rowwise direction → swizzle rowwise scale
      if (B_scale_inv.has_value() && B_scale_inv->numel() > 0) {
        swizzled_scale_inverses.emplace_back(
            swizzleScaleForGemm(B_data, B_te_dtype, *B_scale_inv, B_scaling_mode));
        B_scale_inv = swizzled_scale_inverses.back();
      }
    } else {
      // transb=True → colwise direction → swizzle colwise scale
      if (B_colwise_data.has_value() && B_colwise_scale_inv.has_value() &&
          B_colwise_data->numel() > 0 && B_colwise_scale_inv->numel() > 0) {
        swizzled_scale_inverses.emplace_back(
            swizzleScaleForGemm(*B_colwise_data, B_te_dtype, *B_colwise_scale_inv, B_scaling_mode,
                                /*columnwise=*/true));
        B_colwise_scale_inv = swizzled_scale_inverses.back();
      }
    }
    B_with_gemm_swizzled_scales = true;
  }

  // On Blackwell (sm >= 100), cuBLAS does not natively support BLOCK_SCALING.
  // Convert to MXFP8, matching the pybind path's convert_block_scaling_to_mxfp8_tensor.
  const bool fp8_block_scaling = (A_sm == NVTE_BLOCK_SCALING_1D || A_sm == NVTE_BLOCK_SCALING_2D ||
                                  B_sm == NVTE_BLOCK_SCALING_1D || B_sm == NVTE_BLOCK_SCALING_2D);
  std::vector<Tensor> block_to_mxfp8_buffers;  // keep alive during GEMM

  auto convertOneDirection = [&](const Tensor& sel_data, DType dtype, const Tensor& sel_si,
                                 NVTEScalingMode orig_sm, int device_idx,
                                 bool is_columnwise) -> Tensor {
    auto d_shape = getStableTensorShape(sel_data);
    const bool is_fp4 = is_fp4_dtype(dtype);
    if (is_fp4 && !d_shape.empty()) d_shape.back() *= 2;
    size_t flat_first = 1, flat_last = 1;
    if (!d_shape.empty()) {
      if (is_columnwise) {
        // Columnwise data: first dim is K, rest are M → [K, M_dims...]
        flat_first = d_shape[0];
        for (size_t i = 1; i < d_shape.size(); ++i) flat_last *= d_shape[i];
      } else {
        // Rowwise data: last dim is K, rest are M → [M_dims..., K]
        for (size_t i = 0; i + 1 < d_shape.size(); ++i) flat_first *= d_shape[i];
        flat_last = d_shape.back();
      }
    }
    std::vector<size_t> data_2d = {flat_first, flat_last};
    TensorWrapper input_cu(orig_sm);
    input_cu.set_rowwise_data(sel_data.data_ptr(), dtype, data_2d);
    auto si_shape = getStableTensorShape(sel_si);
    input_cu.set_rowwise_scale_inv(sel_si.data_ptr(), DType::kFloat32, si_shape);

    TensorWrapper output_cu(NVTE_MXFP8_1D_SCALING);
    output_cu.set_rowwise_data(sel_data.data_ptr(), dtype, data_2d);
    size_t sw_first = ((flat_first + 127) / 128) * 128;
    size_t sw_last = ((flat_last + 127) / 128) * 4;
    std::vector<size_t> sw_shape = {sw_first, sw_last};
    auto sw_si = allocateStableTensor(
        {static_cast<int64_t>(sw_first), static_cast<int64_t>(sw_last)}, DType::kByte, device_idx);
    output_cu.set_rowwise_scale_inv(sw_si.data_ptr(), DType::kFloat8E8M0, sw_shape);
    output_cu.set_with_gemm_swizzled_scales(true);

    nvte_swizzle_block_scaling_to_mxfp8_scaling_factors(input_cu.data(), output_cu.data(),
                                                        getCurrentCUDAStreamRaw(device_idx));
    return sw_si;
  };

  auto convertBlockToMxfp8 = [&](Tensor& data, DType dtype, std::optional<Tensor>& scale_inv,
                                 std::optional<Tensor>& cw_data, std::optional<Tensor>& cw_si,
                                 NVTEScalingMode& sm, bool& swizzled, bool use_rowwise) {
    if (sm != NVTE_BLOCK_SCALING_1D && sm != NVTE_BLOCK_SCALING_2D) return;
    int device_idx = data.numel() > 0 ? data.get_device_index()
                                      : (cw_data.has_value() ? cw_data->get_device_index() : 0);

    // Convert the direction that will be used by CanonicalizeGemmInput
    if (use_rowwise) {
      if (scale_inv.has_value() && scale_inv->numel() > 0) {
        auto sw = convertOneDirection(data, dtype, *scale_inv, sm, device_idx,
                                      /*is_columnwise=*/false);
        scale_inv = sw;
        block_to_mxfp8_buffers.push_back(std::move(sw));
      }
    } else {
      if (cw_data.has_value() && cw_si.has_value() && cw_si->numel() > 0) {
        auto sw = convertOneDirection(*cw_data, dtype, *cw_si, sm, device_idx,
                                      /*is_columnwise=*/true);
        cw_si = sw;
        // Also convert rowwise scale if present (for CanonicalizeGemmInput
        // which may fall back to rowwise on Blackwell).
        if (scale_inv.has_value() && scale_inv->numel() > 0 && data.numel() > 0) {
          auto sw2 = convertOneDirection(data, dtype, *scale_inv, sm, device_idx,
                                         /*is_columnwise=*/false);
          scale_inv = sw2;
          block_to_mxfp8_buffers.push_back(std::move(sw2));
        }
        block_to_mxfp8_buffers.push_back(std::move(sw));
      }
    }
    sm = NVTE_MXFP8_1D_SCALING;
    swizzled = true;
  };

  // Build a MXFP8 TensorWrapper from block-scaling data, matching the pybind
  // convert_block_scaling_to_mxfp8_tensor pattern: select rowwise or colwise data,
  // flatten to 2D, set as rowwise data on MXFP8 wrapper.
  auto buildMxfp8FromBlock = [&](const Tensor& sel_data, DType dtype, const Tensor& mxfp8_si,
                                 bool is_colwise) -> TensorWrapper {
    auto d_shape = getStableTensorShape(sel_data);
    const bool is_fp4 = is_fp4_dtype(dtype);
    if (is_fp4 && !d_shape.empty()) d_shape.back() *= 2;
    size_t flat_first, flat_last;
    if (is_colwise) {
      // Colwise: [K, M_dims...] → [K, M]
      flat_first = d_shape.empty() ? 1 : d_shape[0];
      flat_last = 1;
      for (size_t i = 1; i < d_shape.size(); ++i) flat_last *= d_shape[i];
    } else {
      // Rowwise: [M_dims..., K] → [M, K]
      flat_first = 1;
      for (size_t i = 0; i + 1 < d_shape.size(); ++i) flat_first *= d_shape[i];
      flat_last = d_shape.empty() ? 1 : d_shape.back();
    }
    std::vector<size_t> data_2d = {flat_first, flat_last};
    auto si_shape = getStableTensorShape(mxfp8_si);
    TensorWrapper out(NVTE_MXFP8_1D_SCALING);
    out.set_rowwise_data(sel_data.data_ptr(), dtype, data_2d);
    out.set_rowwise_scale_inv(mxfp8_si.data_ptr(), DType::kFloat8E8M0, si_shape);
    out.set_with_gemm_swizzled_scales(true);
    return out;
  };

  TensorWrapper A_tensor, B_tensor;

  if (fp8_block_scaling && transformer_engine::cuda::sm_arch() >= 100) {
    // Convert block scaling to MXFP8 and build TensorWrappers directly.
    // Select the direction CanonicalizeGemmInput will use, convert its
    // scale format, then force TN layout (matching pybind path).
    auto convertAndBuild = [&](Tensor& data, DType dtype, std::optional<Tensor>& si,
                               std::optional<Tensor>& cw_data, std::optional<Tensor>& cw_si,
                               NVTEScalingMode sm, bool use_rowwise) -> TensorWrapper {
      if (sm != NVTE_BLOCK_SCALING_1D && sm != NVTE_BLOCK_SCALING_2D) {
        // Not block scaling — build normally
        auto t = buildInputTensorWrapper(data, dtype, si, cw_data, cw_si, sm);
        return t;
      }
      int dev = data.numel() > 0 ? data.get_device_index()
                                 : (cw_data.has_value() ? cw_data->get_device_index() : 0);
      auto& sel_data = use_rowwise ? data : *cw_data;
      auto& sel_si = use_rowwise ? si : cw_si;
      auto sw = convertOneDirection(sel_data, dtype, *sel_si, sm, dev, use_rowwise ? false : true);
      block_to_mxfp8_buffers.push_back(sw);
      return buildMxfp8FromBlock(sel_data, dtype, sw, !use_rowwise);
    };

    // Only convert and force TN when BOTH tensors are block-scaling
    const bool A_is_block = (A_sm == NVTE_BLOCK_SCALING_1D || A_sm == NVTE_BLOCK_SCALING_2D);
    const bool B_is_block = (B_sm == NVTE_BLOCK_SCALING_1D || B_sm == NVTE_BLOCK_SCALING_2D);
    if (A_is_block && B_is_block) {
      A_tensor = convertAndBuild(A_data, A_te, A_scale_inv, A_colwise_data, A_colwise_scale_inv,
                                 A_sm, transa);
      B_tensor = convertAndBuild(B_data, B_te, B_scale_inv, B_colwise_data, B_colwise_scale_inv,
                                 B_sm, !transb);
      transa = true;
      transb = false;
    } else {
      // Mixed: one is block scaling, one isn't — shouldn't happen but handle gracefully
      A_tensor = buildInputTensorWrapper(A_data, A_te, A_scale_inv, A_colwise_data,
                                         A_colwise_scale_inv, A_sm);
      B_tensor = buildInputTensorWrapper(B_data, B_te, B_scale_inv, B_colwise_data,
                                         B_colwise_scale_inv, B_sm);
    }
  } else {
    A_tensor = buildInputTensorWrapper(A_data, A_te, A_scale_inv, A_colwise_data,
                                       A_colwise_scale_inv, A_sm);
    B_tensor = buildInputTensorWrapper(B_data, B_te, B_scale_inv, B_colwise_data,
                                       B_colwise_scale_inv, B_sm);
  }

  auto D_tensor =
      makeQuantizedTensorWrapper(D_data, D_te, D_shape, D_amax, D_scale, D_scale_inv, D_sm);
  // For block-to-MXFP8 path, swizzled flag is already set by buildMxfp8FromBlock.
  // For non-block path, set from the Python-side flag.
  if (!fp8_block_scaling || transformer_engine::cuda::sm_arch() < 100) {
    A_tensor.set_with_gemm_swizzled_scales(A_with_gemm_swizzled_scales);
    B_tensor.set_with_gemm_swizzled_scales(B_with_gemm_swizzled_scales);
  }

  TensorWrapper bias_tensor;
  if (bias.has_value()) {
    auto bias_te = static_cast<DType>(bias_type);
    auto bias_shape = getStableTensorShape(bias.value());
    bias_tensor = makeTransformerEngineTensor(bias->data_ptr(), bias_shape, bias_te);
  }

  TensorWrapper pre_gelu_tensor;
  if (pre_gelu_out.has_value()) {
    pre_gelu_tensor = makeTransformerEngineTensor(pre_gelu_out.value());
  }

  auto ws_tensor = makeTransformerEngineTensor(workspace);

  auto device_idx = A_data.get_device_index();
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  float alpha_f = static_cast<float>(alpha);
  float beta_f = accumulate ? 1.0f : 0.0f;

  // Configure GEMM
  MatmulConfigWrapper config;
  bool gelu_flag = pre_gelu_out.has_value() && pre_gelu_out->numel() > 0;
  if (bias.has_value()) {
    if (grad) {
      config.set_dbias_tensor(bias_tensor.data());
    } else {
      config.set_bias_tensor(bias_tensor.data());
    }
  }
  if (grad) {
    config.set_with_dgelu_epilogue(gelu_flag);
  } else {
    config.set_with_gelu_epilogue(gelu_flag);
  }
  if (pre_gelu_out.has_value()) {
    config.set_epilogue_aux_tensor(pre_gelu_tensor.data());
  }
  config.set_use_split_accumulator(use_split_accumulator);

  nvte_cublas_gemm_v2(transa, transb, &alpha_f, A_tensor.data(), B_tensor.data(), &beta_f,
                      D_tensor.data(), D_tensor.data(), ws_tensor.data(), config, stream);
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def(
      "gemm("
      "Tensor A_data, int A_te_dtype, Tensor? A_scale_inv, "
      "Tensor? A_colwise_data, Tensor? A_colwise_scale_inv, "
      "int A_scaling_mode, bool A_with_gemm_swizzled_scales, bool transa, "
      "Tensor B_data, int B_te_dtype, Tensor? B_scale_inv, "
      "Tensor? B_colwise_data, Tensor? B_colwise_scale_inv, "
      "int B_scaling_mode, bool B_with_gemm_swizzled_scales, bool transb, "
      "Tensor D_data, int D_te_dtype, Tensor? D_amax, Tensor? D_scale, Tensor? D_scale_inv, "
      "int D_scaling_mode, Tensor? bias, int bias_type, Tensor? pre_gelu_out, "
      "Tensor workspace, bool grad, bool accumulate, bool use_split_accumulator, "
      "float alpha) -> ()");
  m.def(
      "swizzle_scale_for_gemm(Tensor data, Tensor scale_inv, int te_dtype, int scaling_mode) -> "
      "Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("gemm", TORCH_BOX(gemm));
  m.impl("swizzle_scale_for_gemm", TORCH_BOX(swizzleScaleForGemm));
}
