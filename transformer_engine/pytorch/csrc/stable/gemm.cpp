/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../stable_common.h"

#include <transformer_engine/gemm.h>

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

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
// ============================================================================

void gemm(
    // Input A
    Tensor A_data, int64_t A_te_dtype, std::optional<Tensor> A_scale_inv,
    int64_t A_scaling_mode,
    bool transa,
    // Input B
    Tensor B_data, int64_t B_te_dtype, std::optional<Tensor> B_scale_inv,
    int64_t B_scaling_mode,
    bool transb,
    // Output D (pre-allocated)
    Tensor D_data, int64_t D_te_dtype,
    std::optional<Tensor> D_amax, std::optional<Tensor> D_scale,
    std::optional<Tensor> D_scale_inv,
    int64_t D_scaling_mode,
    // Optional bias
    std::optional<Tensor> bias, int64_t bias_type,
    // Optional pre-gelu output
    std::optional<Tensor> pre_gelu_out,
    // Workspace
    Tensor workspace,
    // Config
    bool grad, bool accumulate, bool use_split_accumulator,
    double alpha) {
  auto A_te = static_cast<DType>(A_te_dtype);
  auto B_te = static_cast<DType>(B_te_dtype);
  auto D_te = static_cast<DType>(D_te_dtype);
  auto A_sm = static_cast<NVTEScalingMode>(A_scaling_mode);
  auto B_sm = static_cast<NVTEScalingMode>(B_scaling_mode);
  auto D_sm = static_cast<NVTEScalingMode>(D_scaling_mode);

  auto A_shape = getStableTensorShape(A_data);
  auto B_shape = getStableTensorShape(B_data);
  auto D_shape = getStableTensorShape(D_data);

  auto A_tensor = makeQuantizedTensorWrapper(
      A_data, A_te, A_shape, std::nullopt, std::nullopt, A_scale_inv, A_sm);
  auto B_tensor = makeQuantizedTensorWrapper(
      B_data, B_te, B_shape, std::nullopt, std::nullopt, B_scale_inv, B_sm);
  auto D_tensor = makeQuantizedTensorWrapper(
      D_data, D_te, D_shape, D_amax, D_scale, D_scale_inv, D_sm);

  TensorWrapper bias_tensor;
  if (bias.has_value()) {
    auto bias_te = static_cast<DType>(bias_type);
    auto bias_shape = getStableTensorShape(bias.value());
    bias_tensor = makeTransformerEngineTensor(
        bias->data_ptr(), bias_shape, bias_te);
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

  nvte_cublas_gemm_v2(transa, transb, &alpha_f,
                      A_tensor.data(), B_tensor.data(), &beta_f,
                      D_tensor.data(), D_tensor.data(),
                      ws_tensor.data(), config, stream);
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def("gemm(Tensor A_data, int A_te_dtype, Tensor? A_scale_inv, int A_scaling_mode, bool transa, Tensor B_data, int B_te_dtype, Tensor? B_scale_inv, int B_scaling_mode, bool transb, Tensor D_data, int D_te_dtype, Tensor? D_amax, Tensor? D_scale, Tensor? D_scale_inv, int D_scaling_mode, Tensor? bias, int bias_type, Tensor? pre_gelu_out, Tensor workspace, bool grad, bool accumulate, bool use_split_accumulator, float alpha) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("gemm", TORCH_BOX(gemm));
}
