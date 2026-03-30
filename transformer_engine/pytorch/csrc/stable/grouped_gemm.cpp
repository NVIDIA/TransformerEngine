/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/gemm.h>
#include <transformer_engine/swizzle.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

namespace {

// ============================================================================
// GroupedTensorWrapper construction helpers
// ============================================================================

// Return the DType used for scale_inv given a scaling mode.
DType getGroupedScaleInvDtype(NVTEScalingMode scaling_mode) {
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING:
      return DType::kFloat8E8M0;
    case NVTE_NVFP4_1D_SCALING:
      return DType::kFloat8E4M3;
    default:
      return DType::kFloat32;
  }
}

// Build a GroupedTensorWrapper from the flat buffer tensors extracted from a
// Python GroupedTensor.  All optional Tensor? args may have numel()==0 to
// indicate "not present".
GroupedTensorWrapper buildGroupedTensorWrapper(
    const std::optional<Tensor>& rowwise_data, const std::optional<Tensor>& colwise_data,
    const std::optional<Tensor>& scale_inv, const std::optional<Tensor>& colwise_scale_inv,
    const std::optional<Tensor>& first_dims, const std::optional<Tensor>& last_dims,
    const std::optional<Tensor>& tensor_offsets, int64_t te_dtype, int64_t scaling_mode,
    int64_t logical_0, int64_t logical_1, int64_t num_tensors, bool with_gemm_swizzled_scales) {
  auto dtype = static_cast<DType>(te_dtype);
  auto sm = static_cast<NVTEScalingMode>(scaling_mode);
  DType si_dtype = getGroupedScaleInvDtype(sm);

  std::vector<size_t> logical_shape = {static_cast<size_t>(logical_0),
                                       static_cast<size_t>(logical_1)};
  GroupedTensorWrapper gtw(static_cast<size_t>(num_tensors), logical_shape, sm);

  if (rowwise_data.has_value() && rowwise_data->numel() > 0) {
    auto shape = getStableTensorShape(*rowwise_data);
    gtw.set_rowwise_data(rowwise_data->data_ptr(), dtype, shape);
  }
  if (colwise_data.has_value() && colwise_data->numel() > 0) {
    auto shape = getStableTensorShape(*colwise_data);
    gtw.set_columnwise_data(colwise_data->data_ptr(), dtype, shape);
  }
  if (scale_inv.has_value() && scale_inv->numel() > 0) {
    auto shape = getStableTensorShape(*scale_inv);
    gtw.set_rowwise_scale_inv(scale_inv->data_ptr(), si_dtype, shape);
  }
  if (colwise_scale_inv.has_value() && colwise_scale_inv->numel() > 0) {
    auto shape = getStableTensorShape(*colwise_scale_inv);
    gtw.set_columnwise_scale_inv(colwise_scale_inv->data_ptr(), si_dtype, shape);
  }
  if (first_dims.has_value() && first_dims->numel() > 0) {
    auto shape = getStableTensorShape(*first_dims);
    gtw.set_first_dims(first_dims->data_ptr(), DType::kInt64, shape);
  }
  if (last_dims.has_value() && last_dims->numel() > 0) {
    auto shape = getStableTensorShape(*last_dims);
    gtw.set_last_dims(last_dims->data_ptr(), DType::kInt64, shape);
  }
  if (tensor_offsets.has_value() && tensor_offsets->numel() > 0) {
    auto shape = getStableTensorShape(*tensor_offsets);
    gtw.set_tensor_offsets(tensor_offsets->data_ptr(), DType::kInt64, shape);
  }
  gtw.set_with_gemm_swizzled_scales(with_gemm_swizzled_scales);
  return gtw;
}

// Build a GroupedMatmulConfigWrapper with the given options.
GroupedMatmulConfigWrapper buildGroupedGemmConfig(bool use_split_accumulator, int64_t sm_count) {
  GroupedMatmulConfigWrapper config;
  config.set_use_split_accumulator(use_split_accumulator);
  if (sm_count > 0) {
    config.set_sm_count(static_cast<int>(sm_count));
  }
  return config;
}

}  // namespace

// ============================================================================
// grouped_gemm_for_grouped_tensor
//
// Wraps nvte_grouped_gemm (Blackwell+).
// A, B, D are Python GroupedTensors whose flat buffers are passed individually.
// Optional bias GroupedTensor is controlled by has_bias.
// ============================================================================

void grouped_gemm_for_grouped_tensor(
    // A (GroupedTensor) — 13 fields + transa
    std::optional<Tensor> A_rowwise, std::optional<Tensor> A_colwise, std::optional<Tensor> A_si,
    std::optional<Tensor> A_colwise_si, std::optional<Tensor> A_first_dims,
    std::optional<Tensor> A_last_dims, std::optional<Tensor> A_tensor_offsets, int64_t A_te_dtype,
    int64_t A_scaling_mode, int64_t A_logical_0, int64_t A_logical_1, int64_t A_num_tensors,
    bool A_swizzled, bool transa,
    // B (GroupedTensor) — 13 fields + transb
    std::optional<Tensor> B_rowwise, std::optional<Tensor> B_colwise, std::optional<Tensor> B_si,
    std::optional<Tensor> B_colwise_si, std::optional<Tensor> B_first_dims,
    std::optional<Tensor> B_last_dims, std::optional<Tensor> B_tensor_offsets, int64_t B_te_dtype,
    int64_t B_scaling_mode, int64_t B_logical_0, int64_t B_logical_1, int64_t B_num_tensors,
    bool B_swizzled, bool transb,
    // D (GroupedTensor) — 13 fields (no trans)
    std::optional<Tensor> D_rowwise, std::optional<Tensor> D_colwise, std::optional<Tensor> D_si,
    std::optional<Tensor> D_colwise_si, std::optional<Tensor> D_first_dims,
    std::optional<Tensor> D_last_dims, std::optional<Tensor> D_tensor_offsets, int64_t D_te_dtype,
    int64_t D_scaling_mode, int64_t D_logical_0, int64_t D_logical_1, int64_t D_num_tensors,
    // Config
    Tensor alpha, Tensor beta, Tensor workspace_setup, Tensor workspace_cublas,
    bool use_split_accumulator, int64_t sm_count,
    // Optional bias (GroupedTensor) — 13 fields, guarded by has_bias
    bool has_bias, std::optional<Tensor> bias_rowwise, std::optional<Tensor> bias_colwise,
    std::optional<Tensor> bias_si, std::optional<Tensor> bias_colwise_si,
    std::optional<Tensor> bias_first_dims, std::optional<Tensor> bias_last_dims,
    std::optional<Tensor> bias_tensor_offsets, int64_t bias_te_dtype, int64_t bias_scaling_mode,
    int64_t bias_logical_0, int64_t bias_logical_1, int64_t bias_num_tensors, bool bias_swizzled) {
  auto A_gt = buildGroupedTensorWrapper(A_rowwise, A_colwise, A_si, A_colwise_si, A_first_dims,
                                        A_last_dims, A_tensor_offsets, A_te_dtype, A_scaling_mode,
                                        A_logical_0, A_logical_1, A_num_tensors, A_swizzled);
  auto B_gt = buildGroupedTensorWrapper(B_rowwise, B_colwise, B_si, B_colwise_si, B_first_dims,
                                        B_last_dims, B_tensor_offsets, B_te_dtype, B_scaling_mode,
                                        B_logical_0, B_logical_1, B_num_tensors, B_swizzled);
  auto D_gt = buildGroupedTensorWrapper(D_rowwise, D_colwise, D_si, D_colwise_si, D_first_dims,
                                        D_last_dims, D_tensor_offsets, D_te_dtype, D_scaling_mode,
                                        D_logical_0, D_logical_1, D_num_tensors, false);

  auto alpha_tw = makeTransformerEngineTensor(alpha);
  auto beta_tw = makeTransformerEngineTensor(beta);
  auto ws_setup_tw = makeTransformerEngineTensor(workspace_setup);
  auto ws_cublas_tw = makeTransformerEngineTensor(workspace_cublas);

  auto config = buildGroupedGemmConfig(use_split_accumulator, sm_count);

  // Determine device from whichever data buffer is present
  int device_idx = 0;
  if (A_rowwise.has_value() && A_rowwise->numel() > 0)
    device_idx = A_rowwise->get_device_index();
  else if (B_rowwise.has_value() && B_rowwise->numel() > 0)
    device_idx = B_rowwise->get_device_index();

  nvte_grouped_gemm(A_gt.data(), static_cast<int>(transa), B_gt.data(), static_cast<int>(transb),
                    D_gt.data(), D_gt.data(), alpha_tw.data(), beta_tw.data(), ws_setup_tw.data(),
                    ws_cublas_tw.data(), static_cast<NVTEGroupedMatmulConfig>(config),
                    getCurrentCUDAStreamRaw(device_idx));

  if (has_bias) {
    auto bias_gt = buildGroupedTensorWrapper(bias_rowwise, bias_colwise, bias_si, bias_colwise_si,
                                             bias_first_dims, bias_last_dims, bias_tensor_offsets,
                                             bias_te_dtype, bias_scaling_mode, bias_logical_0,
                                             bias_logical_1, bias_num_tensors, bias_swizzled);
    nvte_grouped_bias_add(D_gt.data(), bias_gt.data(), getCurrentCUDAStreamRaw(device_idx));
  }
}

// ============================================================================
// grouped_gemm_for_discrete_in
//
// Wraps nvte_grouped_gemm_with_discrete_inputA (Blackwell+).
// A is provided as packed pointer arrays (rowwise_ptrs, colwise_ptrs, etc.)
// B and D are GroupedTensors.
// ============================================================================

void grouped_gemm_for_discrete_in(
    // A — packed pointer arrays, one entry per expert tensor
    Tensor A_rowwise_ptrs, Tensor A_colwise_ptrs, Tensor A_si_ptrs, Tensor A_csi_ptrs,
    Tensor A_shapes,         // (num_a, 2) int64: [rows, cols] per tensor
    Tensor A_te_dtypes,      // (num_a,) int32
    Tensor A_scaling_modes,  // (num_a,) int32
    int64_t num_a_tensors,
    // B (GroupedTensor) — 13 fields + transb
    std::optional<Tensor> B_rowwise, std::optional<Tensor> B_colwise, std::optional<Tensor> B_si,
    std::optional<Tensor> B_colwise_si, std::optional<Tensor> B_first_dims,
    std::optional<Tensor> B_last_dims, std::optional<Tensor> B_tensor_offsets, int64_t B_te_dtype,
    int64_t B_scaling_mode, int64_t B_logical_0, int64_t B_logical_1, int64_t B_num_tensors,
    bool B_swizzled, bool transb,
    // D (GroupedTensor) — 13 fields
    std::optional<Tensor> D_rowwise, std::optional<Tensor> D_colwise, std::optional<Tensor> D_si,
    std::optional<Tensor> D_colwise_si, std::optional<Tensor> D_first_dims,
    std::optional<Tensor> D_last_dims, std::optional<Tensor> D_tensor_offsets, int64_t D_te_dtype,
    int64_t D_scaling_mode, int64_t D_logical_0, int64_t D_logical_1, int64_t D_num_tensors,
    // Config
    Tensor alpha, Tensor beta, Tensor workspace_setup, Tensor workspace_cublas,
    bool use_split_accumulator, int64_t sm_count,
    // Optional bias
    bool has_bias, std::optional<Tensor> bias_rowwise, std::optional<Tensor> bias_colwise,
    std::optional<Tensor> bias_si, std::optional<Tensor> bias_colwise_si,
    std::optional<Tensor> bias_first_dims, std::optional<Tensor> bias_last_dims,
    std::optional<Tensor> bias_tensor_offsets, int64_t bias_te_dtype, int64_t bias_scaling_mode,
    int64_t bias_logical_0, int64_t bias_logical_1, int64_t bias_num_tensors, bool bias_swizzled) {
  // Build the individual A TensorWrappers from the packed pointer arrays
  const auto* rw_ptrs = reinterpret_cast<const uintptr_t*>(A_rowwise_ptrs.data_ptr());
  const auto* cw_ptrs = reinterpret_cast<const uintptr_t*>(A_colwise_ptrs.data_ptr());
  const auto* si_ptrs = reinterpret_cast<const uintptr_t*>(A_si_ptrs.data_ptr());
  const auto* csi_ptrs = reinterpret_cast<const uintptr_t*>(A_csi_ptrs.data_ptr());
  const auto* shapes = reinterpret_cast<const int64_t*>(A_shapes.data_ptr());
  const auto* dtypes = reinterpret_cast<const int32_t*>(A_te_dtypes.data_ptr());
  const auto* modes = reinterpret_cast<const int32_t*>(A_scaling_modes.data_ptr());

  std::vector<TensorWrapper> A_wrappers;
  std::vector<NVTETensor> A_nvte;
  A_wrappers.reserve(static_cast<size_t>(num_a_tensors));
  A_nvte.reserve(static_cast<size_t>(num_a_tensors));

  for (int64_t i = 0; i < num_a_tensors; ++i) {
    auto dtype = static_cast<DType>(dtypes[i]);
    auto sm = static_cast<NVTEScalingMode>(modes[i]);
    DType si_dtype = getGroupedScaleInvDtype(sm);
    int64_t rows = shapes[2 * i];
    int64_t cols = shapes[2 * i + 1];
    std::vector<size_t> shape = {static_cast<size_t>(rows), static_cast<size_t>(cols)};

    TensorWrapper tw(sm);
    if (rw_ptrs[i] != 0) {
      tw.set_rowwise_data(reinterpret_cast<void*>(rw_ptrs[i]), dtype, shape);
    }
    if (cw_ptrs[i] != 0) {
      tw.set_columnwise_data(reinterpret_cast<void*>(cw_ptrs[i]), dtype, shape);
    }
    if (si_ptrs[i] != 0) {
      // Scale shape for a (rows, cols) tensor: (rows, ceil(cols/block)) etc.
      // We pass a placeholder shape; NVTE uses the tensor's logical shape for scale counts.
      std::vector<size_t> si_shape = {static_cast<size_t>(rows)};
      tw.set_rowwise_scale_inv(reinterpret_cast<void*>(si_ptrs[i]), si_dtype, si_shape);
    }
    if (csi_ptrs[i] != 0) {
      std::vector<size_t> csi_shape = {static_cast<size_t>(cols)};
      tw.set_columnwise_scale_inv(reinterpret_cast<void*>(csi_ptrs[i]), si_dtype, csi_shape);
    }
    A_wrappers.emplace_back(std::move(tw));
    A_nvte.emplace_back(A_wrappers.back().data());
  }

  // NOTE: transa is not passed to this function per the NVTE API convention — it is
  // implied by the physical layout set via set_rowwise_data vs set_columnwise_data.
  // The Python side calls _extract_gemm_operand(Ai, transa) so the right buffer
  // (rowwise or columnwise) is already selected.  We always pass transa=false here
  // since CanonicalizeGemmInput in the C++ kernel handles orientation.
  int transa_int = 0;

  auto B_gt = buildGroupedTensorWrapper(B_rowwise, B_colwise, B_si, B_colwise_si, B_first_dims,
                                        B_last_dims, B_tensor_offsets, B_te_dtype, B_scaling_mode,
                                        B_logical_0, B_logical_1, B_num_tensors, B_swizzled);
  auto D_gt = buildGroupedTensorWrapper(D_rowwise, D_colwise, D_si, D_colwise_si, D_first_dims,
                                        D_last_dims, D_tensor_offsets, D_te_dtype, D_scaling_mode,
                                        D_logical_0, D_logical_1, D_num_tensors, false);

  auto alpha_tw = makeTransformerEngineTensor(alpha);
  auto beta_tw = makeTransformerEngineTensor(beta);
  auto ws_setup_tw = makeTransformerEngineTensor(workspace_setup);
  auto ws_cublas_tw = makeTransformerEngineTensor(workspace_cublas);
  auto config = buildGroupedGemmConfig(use_split_accumulator, sm_count);

  int device_idx = A_rowwise_ptrs.get_device_index();

  nvte_grouped_gemm_with_discrete_inputA(
      A_nvte.data(), static_cast<size_t>(num_a_tensors), transa_int, B_gt.data(),
      static_cast<int>(transb), D_gt.data(), D_gt.data(), alpha_tw.data(), beta_tw.data(),
      ws_setup_tw.data(), ws_cublas_tw.data(), static_cast<NVTEGroupedMatmulConfig>(config),
      getCurrentCUDAStreamRaw(device_idx));

  if (has_bias) {
    auto bias_gt = buildGroupedTensorWrapper(bias_rowwise, bias_colwise, bias_si, bias_colwise_si,
                                             bias_first_dims, bias_last_dims, bias_tensor_offsets,
                                             bias_te_dtype, bias_scaling_mode, bias_logical_0,
                                             bias_logical_1, bias_num_tensors, bias_swizzled);
    nvte_grouped_bias_add(D_gt.data(), bias_gt.data(), getCurrentCUDAStreamRaw(device_idx));
  }
}

// ============================================================================
// grouped_gemm_for_discrete_out
//
// Wraps nvte_grouped_gemm_with_discrete_out (Blackwell+).
// A and B are GroupedTensors; D is provided as packed pointer arrays.
// ============================================================================

void grouped_gemm_for_discrete_out(
    // A (GroupedTensor) — 13 fields + transa
    std::optional<Tensor> A_rowwise, std::optional<Tensor> A_colwise, std::optional<Tensor> A_si,
    std::optional<Tensor> A_colwise_si, std::optional<Tensor> A_first_dims,
    std::optional<Tensor> A_last_dims, std::optional<Tensor> A_tensor_offsets, int64_t A_te_dtype,
    int64_t A_scaling_mode, int64_t A_logical_0, int64_t A_logical_1, int64_t A_num_tensors,
    bool A_swizzled, bool transa,
    // B (GroupedTensor) — 13 fields + transb
    std::optional<Tensor> B_rowwise, std::optional<Tensor> B_colwise, std::optional<Tensor> B_si,
    std::optional<Tensor> B_colwise_si, std::optional<Tensor> B_first_dims,
    std::optional<Tensor> B_last_dims, std::optional<Tensor> B_tensor_offsets, int64_t B_te_dtype,
    int64_t B_scaling_mode, int64_t B_logical_0, int64_t B_logical_1, int64_t B_num_tensors,
    bool B_swizzled, bool transb,
    // D — packed pointer arrays
    Tensor D_rowwise_ptrs, Tensor D_si_ptrs,
    Tensor D_shapes,         // (num_d, 2) int64: [rows, cols] per tensor
    Tensor D_te_dtypes,      // (num_d,) int32
    Tensor D_scaling_modes,  // (num_d,) int32
    int64_t num_d_tensors,
    // Config
    Tensor alpha, Tensor beta, Tensor workspace_setup, Tensor workspace_cublas,
    bool use_split_accumulator, int64_t sm_count) {
  auto A_gt = buildGroupedTensorWrapper(A_rowwise, A_colwise, A_si, A_colwise_si, A_first_dims,
                                        A_last_dims, A_tensor_offsets, A_te_dtype, A_scaling_mode,
                                        A_logical_0, A_logical_1, A_num_tensors, A_swizzled);
  auto B_gt = buildGroupedTensorWrapper(B_rowwise, B_colwise, B_si, B_colwise_si, B_first_dims,
                                        B_last_dims, B_tensor_offsets, B_te_dtype, B_scaling_mode,
                                        B_logical_0, B_logical_1, B_num_tensors, B_swizzled);

  // Build D TensorWrapper array from packed pointers
  const auto* rw_ptrs = reinterpret_cast<const uintptr_t*>(D_rowwise_ptrs.data_ptr());
  const auto* si_ptrs = reinterpret_cast<const uintptr_t*>(D_si_ptrs.data_ptr());
  const auto* shapes = reinterpret_cast<const int64_t*>(D_shapes.data_ptr());
  const auto* dtypes = reinterpret_cast<const int32_t*>(D_te_dtypes.data_ptr());

  std::vector<TensorWrapper> D_wrappers;
  std::vector<NVTETensor> D_nvte;
  D_wrappers.reserve(static_cast<size_t>(num_d_tensors));
  D_nvte.reserve(static_cast<size_t>(num_d_tensors));

  for (int64_t i = 0; i < num_d_tensors; ++i) {
    auto dtype = static_cast<DType>(dtypes[i]);
    int64_t rows = shapes[2 * i];
    int64_t cols = shapes[2 * i + 1];
    std::vector<size_t> shape = {static_cast<size_t>(rows), static_cast<size_t>(cols)};

    TensorWrapper tw;
    if (rw_ptrs[i] != 0) {
      tw.set_rowwise_data(reinterpret_cast<void*>(rw_ptrs[i]), dtype, shape);
    }
    if (si_ptrs[i] != 0) {
      std::vector<size_t> si_shape = {static_cast<size_t>(rows)};
      tw.set_rowwise_scale_inv(reinterpret_cast<void*>(si_ptrs[i]), DType::kFloat32, si_shape);
    }
    D_wrappers.emplace_back(std::move(tw));
    D_nvte.emplace_back(D_wrappers.back().data());
  }

  auto alpha_tw = makeTransformerEngineTensor(alpha);
  auto beta_tw = makeTransformerEngineTensor(beta);
  auto ws_setup_tw = makeTransformerEngineTensor(workspace_setup);
  auto ws_cublas_tw = makeTransformerEngineTensor(workspace_cublas);
  auto config = buildGroupedGemmConfig(use_split_accumulator, sm_count);

  int device_idx = 0;
  if (A_rowwise.has_value() && A_rowwise->numel() > 0) device_idx = A_rowwise->get_device_index();

  nvte_grouped_gemm_with_discrete_out(
      A_gt.data(), static_cast<int>(transa), B_gt.data(), static_cast<int>(transb), D_nvte.data(),
      static_cast<size_t>(num_d_tensors), D_nvte.data(), static_cast<size_t>(num_d_tensors),
      alpha_tw.data(), beta_tw.data(), ws_setup_tw.data(), ws_cublas_tw.data(),
      static_cast<NVTEGroupedMatmulConfig>(config), getCurrentCUDAStreamRaw(device_idx));
}

}  // namespace transformer_engine::pytorch::stable

// ============================================================================
// Op registration
// ============================================================================

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  // grouped_gemm_for_grouped_tensor: A(13) + transa + B(13) + transb + D(13) +
  //   alpha + beta + ws_setup + ws_cublas + use_split_accum + sm_count +
  //   has_bias + bias(13) = 58 args total
  m.def(
      "grouped_gemm_for_grouped_tensor("
      // A
      "Tensor? A_rowwise, Tensor? A_colwise, Tensor? A_si, Tensor? A_colwise_si, "
      "Tensor? A_first_dims, Tensor? A_last_dims, Tensor? A_tensor_offsets, "
      "int A_te_dtype, int A_scaling_mode, int A_logical_0, int A_logical_1, "
      "int A_num_tensors, bool A_swizzled, bool transa, "
      // B
      "Tensor? B_rowwise, Tensor? B_colwise, Tensor? B_si, Tensor? B_colwise_si, "
      "Tensor? B_first_dims, Tensor? B_last_dims, Tensor? B_tensor_offsets, "
      "int B_te_dtype, int B_scaling_mode, int B_logical_0, int B_logical_1, "
      "int B_num_tensors, bool B_swizzled, bool transb, "
      // D
      "Tensor? D_rowwise, Tensor? D_colwise, Tensor? D_si, Tensor? D_colwise_si, "
      "Tensor? D_first_dims, Tensor? D_last_dims, Tensor? D_tensor_offsets, "
      "int D_te_dtype, int D_scaling_mode, int D_logical_0, int D_logical_1, "
      "int D_num_tensors, "
      // config
      "Tensor alpha, Tensor beta, Tensor workspace_setup, Tensor workspace_cublas, "
      "bool use_split_accumulator, int sm_count, "
      // bias
      "bool has_bias, "
      "Tensor? bias_rowwise, Tensor? bias_colwise, Tensor? bias_si, Tensor? bias_colwise_si, "
      "Tensor? bias_first_dims, Tensor? bias_last_dims, Tensor? bias_tensor_offsets, "
      "int bias_te_dtype, int bias_scaling_mode, int bias_logical_0, int bias_logical_1, "
      "int bias_num_tensors, bool bias_swizzled"
      ") -> ()");

  // grouped_gemm_for_discrete_in: A_ptrs(7) + A_meta(3) + num_a +
  //   B(13) + transb + D(13) + config(6) + has_bias + bias(13) = 57 args
  m.def(
      "grouped_gemm_for_discrete_in("
      // A packed pointers
      "Tensor A_rowwise_ptrs, Tensor A_colwise_ptrs, Tensor A_si_ptrs, Tensor A_csi_ptrs, "
      "Tensor A_shapes, Tensor A_te_dtypes, Tensor A_scaling_modes, int num_a_tensors, "
      // B
      "Tensor? B_rowwise, Tensor? B_colwise, Tensor? B_si, Tensor? B_colwise_si, "
      "Tensor? B_first_dims, Tensor? B_last_dims, Tensor? B_tensor_offsets, "
      "int B_te_dtype, int B_scaling_mode, int B_logical_0, int B_logical_1, "
      "int B_num_tensors, bool B_swizzled, bool transb, "
      // D
      "Tensor? D_rowwise, Tensor? D_colwise, Tensor? D_si, Tensor? D_colwise_si, "
      "Tensor? D_first_dims, Tensor? D_last_dims, Tensor? D_tensor_offsets, "
      "int D_te_dtype, int D_scaling_mode, int D_logical_0, int D_logical_1, "
      "int D_num_tensors, "
      // config
      "Tensor alpha, Tensor beta, Tensor workspace_setup, Tensor workspace_cublas, "
      "bool use_split_accumulator, int sm_count, "
      // bias
      "bool has_bias, "
      "Tensor? bias_rowwise, Tensor? bias_colwise, Tensor? bias_si, Tensor? bias_colwise_si, "
      "Tensor? bias_first_dims, Tensor? bias_last_dims, Tensor? bias_tensor_offsets, "
      "int bias_te_dtype, int bias_scaling_mode, int bias_logical_0, int bias_logical_1, "
      "int bias_num_tensors, bool bias_swizzled"
      ") -> ()");

  // grouped_gemm_for_discrete_out: A(13) + transa + B(13) + transb +
  //   D_ptrs(5) + num_d + config(6) = 51 args
  m.def(
      "grouped_gemm_for_discrete_out("
      // A
      "Tensor? A_rowwise, Tensor? A_colwise, Tensor? A_si, Tensor? A_colwise_si, "
      "Tensor? A_first_dims, Tensor? A_last_dims, Tensor? A_tensor_offsets, "
      "int A_te_dtype, int A_scaling_mode, int A_logical_0, int A_logical_1, "
      "int A_num_tensors, bool A_swizzled, bool transa, "
      // B
      "Tensor? B_rowwise, Tensor? B_colwise, Tensor? B_si, Tensor? B_colwise_si, "
      "Tensor? B_first_dims, Tensor? B_last_dims, Tensor? B_tensor_offsets, "
      "int B_te_dtype, int B_scaling_mode, int B_logical_0, int B_logical_1, "
      "int B_num_tensors, bool B_swizzled, bool transb, "
      // D packed pointers
      "Tensor D_rowwise_ptrs, Tensor D_si_ptrs, "
      "Tensor D_shapes, Tensor D_te_dtypes, Tensor D_scaling_modes, int num_d_tensors, "
      // config
      "Tensor alpha, Tensor beta, Tensor workspace_setup, Tensor workspace_cublas, "
      "bool use_split_accumulator, int sm_count"
      ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("grouped_gemm_for_grouped_tensor", TORCH_BOX(grouped_gemm_for_grouped_tensor));
  m.impl("grouped_gemm_for_discrete_in", TORCH_BOX(grouped_gemm_for_discrete_in));
  m.impl("grouped_gemm_for_discrete_out", TORCH_BOX(grouped_gemm_for_discrete_out));
}
