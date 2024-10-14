/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>

#include "common.h"
#include "common/util/cuda_runtime.h"
#include "extensions.h"
#include "pytorch/csrc/common.h"
#include "transformer_engine/transformer_engine.h"

std::vector<at::Tensor> te_gemm2_helper(
    at::Tensor A, transformer_engine::DType A_dtype, std::optional<at::Tensor> A_scale_inv,
    bool transa, at::Tensor B, transformer_engine::DType B_dtype,
    std::optional<at::Tensor> B_scale_inv, bool transb, std::optional<at::Tensor> D,
    at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
    transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine;
  if (A.data_ptr() == nullptr || B.data_ptr() == nullptr) {
    at::Tensor out;
    if (D.has_value() && D->data_ptr() != nullptr && !accumulate) {
      D->zero_();  // TODO: Handle D without a value
      out = *D;
    }
    if (pre_gelu_out.data_ptr() != nullptr) pre_gelu_out.zero_();
    return {out, pre_gelu_out};
  }

  A = A.contiguous();
  B = B.contiguous();

  if (!D.has_value()) {
    auto type = GetATenDType(D_type);
    auto opts = at::TensorOptions().dtype(type).device(A.options().device());
    *D = at::empty({B.size(0), A.size(0)}, opts);
  }

  auto A_scale_inv_ptr = A_scale_inv.has_value() ? A_scale_inv->data_ptr() : nullptr;
  auto B_scale_inv_ptr = B_scale_inv.has_value() ? B_scale_inv->data_ptr() : nullptr;
  auto te_A = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_dtype,
      nullptr, nullptr, A_scale_inv_ptr);
  auto te_B = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_dtype,
      nullptr, nullptr, B_scale_inv_ptr);
  auto te_D = makeTransformerEngineTensor(
      D->data_ptr(), {static_cast<size_t>(D->size(0)), static_cast<size_t>(D->size(1))}, D_type,
      D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
  auto te_bias =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);

  const auto gelu_shape = pre_gelu_out.data_ptr() == nullptr
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto te_pre_gelu_out = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));
  auto te_workspace =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, DType::kByte);

  nvte_cublas_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(), te_pre_gelu_out.data(),
                   transa, transb, grad, te_workspace.data(), accumulate, use_split_accumulator,
                   math_sm_count, at::cuda::getCurrentCUDAStream());

  return {*D, pre_gelu_out};
}

std::vector<at::Tensor> te_gemm2(transformer_engine::Float8Tensor A, bool transa,
                                 transformer_engine::Float8Tensor B, bool transb,
                                 std::optional<at::Tensor> D, at::Tensor D_scale,
                                 transformer_engine::DType D_type, at::Tensor D_amax,
                                 at::Tensor bias, transformer_engine::DType bias_type,
                                 at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                                 size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                                 int math_sm_count) {
  return te_gemm2_helper(A.data, A.dtype, A.scale_inv, transa, B.data, B.dtype, B.scale_inv, transb,
                         D, D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad, workspace,
                         workspaceSize, accumulate, use_split_accumulator, math_sm_count);
}

std::vector<at::Tensor> te_gemm2(at::Tensor A, bool transa, at::Tensor B, bool transb,
                                 std::optional<at::Tensor> D, at::Tensor D_scale,
                                 transformer_engine::DType D_type, at::Tensor D_amax,
                                 at::Tensor bias, transformer_engine::DType bias_type,
                                 at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                                 size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                                 int math_sm_count) {
  transformer_engine::DType A_dtype = GetTransformerEngineDType(A.scalar_type());
  transformer_engine::DType B_dtype = GetTransformerEngineDType(B.scalar_type());
  return te_gemm2_helper(A, A_dtype, std::nullopt, transa, B, B_dtype, std::nullopt, transb, D,
                         D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad, workspace,
                         workspaceSize, accumulate, use_split_accumulator, math_sm_count);
}

void te_gemm(at::Tensor A, at::Tensor A_scale_inverse, transformer_engine::DType A_type,
             bool transa, at::Tensor B, at::Tensor B_scale_inverse,
             transformer_engine::DType B_type, bool transb, at::Tensor D, at::Tensor D_scale,
             transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
             transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
             at::Tensor workspace, size_t workspaceSize, bool accumulate,
             bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine;
  if (A.data_ptr() == nullptr || B.data_ptr() == nullptr) {
    if (D.data_ptr() != nullptr && !accumulate) D.zero_();
    if (bias.data_ptr() != nullptr) bias.zero_();
    if (pre_gelu_out.data_ptr() != nullptr) pre_gelu_out.zero_();
    return;
  }

  A = A.contiguous();
  B = B.contiguous();

  auto te_A = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inverse.data_ptr());
  auto te_B = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inverse.data_ptr());
  auto te_D = makeTransformerEngineTensor(
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
  auto te_bias =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);

  const auto gelu_shape = pre_gelu_out.data_ptr() == nullptr
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto te_pre_gelu_out = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));
  auto te_workspace =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, DType::kByte);

  nvte_cublas_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(), te_pre_gelu_out.data(),
                   transa, transb, grad, te_workspace.data(), accumulate, use_split_accumulator,
                   math_sm_count, at::cuda::getCurrentCUDAStream());
}

void te_atomic_gemm(at::Tensor A, at::Tensor A_scale_inverse, transformer_engine::DType A_type,
                    bool transa, at::Tensor B, at::Tensor B_scale_inverse,
                    transformer_engine::DType B_type, bool transb, at::Tensor D, at::Tensor D_scale,
                    transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
                    transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
                    at::Tensor workspace, size_t workspaceSize, bool accumulate,
                    bool use_split_accumulator, int math_sm_count, int m_split, int n_split,
                    bool gemm_producer, at::Tensor counter) {
  using namespace transformer_engine;
  auto te_A = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inverse.data_ptr());
  auto te_B = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inverse.data_ptr());
  auto te_D = makeTransformerEngineTensor(
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
  auto te_bias =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);
  auto te_counter = makeTransformerEngineTensor(
      counter.data_ptr(), {static_cast<size_t>(counter.size(0))}, DType::kInt32);

  const auto gelu_shape = pre_gelu_out.data_ptr() == nullptr
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto te_pre_gelu_out = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));
  auto te_workspace =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, DType::kByte);

  nvte_cublas_atomic_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(),
                          te_pre_gelu_out.data(), transa, transb, grad, te_workspace.data(),
                          accumulate, use_split_accumulator, math_sm_count, m_split, n_split,
                          gemm_producer, te_counter.data(), at::cuda::getCurrentCUDAStream());
}

void te_grouped_gemm(std::vector<at::Tensor> A, at::Tensor A_scale_inverse, int A_offset,
                     transformer_engine::DType A_type, bool transa, std::vector<at::Tensor> B,
                     at::Tensor B_scale_inverse, int B_offset, transformer_engine::DType B_type,
                     bool transb, std::vector<at::Tensor> D, int D_offset, at::Tensor D_scale,
                     transformer_engine::DType D_type, at::Tensor D_amax,
                     std::vector<at::Tensor> bias, transformer_engine::DType bias_type,
                     std::vector<at::Tensor> pre_gelu_out, bool grad,
                     std::vector<at::Tensor> workspace, size_t workspaceSize, bool accumulate,
                     bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine;
  std::vector<NVTETensor> te_A, te_B, te_D, te_bias, te_pre_gelu_out, te_workspace;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype, void* amax_dptr,
                                        void* scale_dptr, void* scale_inv_dptr) -> NVTETensor {
    tensor_wrappers.emplace_back(
        makeTransformerEngineTensor(dptr, shape, dtype, amax_dptr, scale_dptr, scale_inv_dptr));
    return tensor_wrappers.back().data();
  };
  for (size_t i = 0; i < A.size(); i++) {
    if (A[i].data_ptr() == nullptr || B[i].data_ptr() == nullptr) {
      if (D[i].data_ptr() != nullptr && !accumulate) D[i].zero_();
      if (bias[i].data_ptr() != nullptr) bias[i].zero_();
      if (pre_gelu_out[i].data_ptr() != nullptr) pre_gelu_out[i].zero_();
      continue;
    }

    NVTE_CHECK(A[i].is_contiguous(), "A[", i, "] must be contiguous.");
    NVTE_CHECK(B[i].is_contiguous(), "B[", i, "] must be contiguous.");
    NVTE_CHECK(D[i].is_contiguous(), "D[", i, "] must be contiguous.");

    te_A.emplace_back(make_tensor(
        A[i].data_ptr(), {static_cast<size_t>(A[i].size(0)), static_cast<size_t>(A[i].size(1))},
        A_type, nullptr, nullptr, getDataPtr(A_scale_inverse, A_offset + i)));
    te_B.emplace_back(make_tensor(
        B[i].data_ptr(), {static_cast<size_t>(B[i].size(0)), static_cast<size_t>(B[i].size(1))},
        B_type, nullptr, nullptr, getDataPtr(B_scale_inverse, B_offset + i)));
    te_D.emplace_back(make_tensor(
        D[i].data_ptr(), {static_cast<size_t>(D[i].size(0)), static_cast<size_t>(D[i].size(1))},
        D_type, getDataPtr(D_amax, D_offset + i), getDataPtr(D_scale, D_offset + i), nullptr));
    te_bias.emplace_back(make_tensor(bias[i].data_ptr(), {static_cast<size_t>(bias[i].size(0))},
                                     bias_type, nullptr, nullptr, nullptr));

    const auto gelu_shape = pre_gelu_out[i].data_ptr() == nullptr
                                ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out[i].size(0))}
                                : std::vector<size_t>{static_cast<size_t>(pre_gelu_out[i].size(0)),
                                                      static_cast<size_t>(pre_gelu_out[i].size(1))};
    te_pre_gelu_out.emplace_back(make_tensor(
        pre_gelu_out[i].data_ptr(), gelu_shape,
        GetTransformerEngineDType(pre_gelu_out[i].scalar_type()), nullptr, nullptr, nullptr));
  }
  for (size_t i = 0; i < workspace.size(); i++) {
    te_workspace.emplace_back(make_tensor(workspace[i].data_ptr(), {workspaceSize}, DType::kByte,
                                          nullptr, nullptr, nullptr));
  }

  // For now, we only have multi-stream cublas backend.
  nvte_multi_stream_cublas_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(),
                                te_pre_gelu_out.data(), te_A.size(), transa, transb, grad,
                                te_workspace.data(), accumulate, use_split_accumulator,
                                math_sm_count, at::cuda::getCurrentCUDAStream());
}

void te_grouped_gemm_single_output(
    std::vector<at::Tensor> A, std::vector<at::Tensor> A_scale_inverse, int A_offset,
    transformer_engine::DType A_type, bool transa, std::vector<at::Tensor> B,
    at::Tensor B_scale_inverse, int B_offset, transformer_engine::DType B_type, bool transb,
    std::vector<int64_t> m_splits, at::Tensor D, int D_offset, at::Tensor D_scale,
    transformer_engine::DType D_type, at::Tensor D_amax, std::vector<at::Tensor> bias,
    transformer_engine::DType bias_type, std::vector<at::Tensor> pre_gelu_out, bool grad,
    std::vector<at::Tensor> workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine;
  std::vector<NVTETensor> te_A, te_B, te_D, te_bias, te_pre_gelu_out, te_workspace;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype, void* amax_dptr,
                                        void* scale_dptr, void* scale_inv_dptr) -> NVTETensor {
    tensor_wrappers.emplace_back(
        makeTransformerEngineTensor(dptr, shape, dtype, amax_dptr, scale_dptr, scale_inv_dptr));
    return tensor_wrappers.back().data();
  };
  NVTE_CHECK(D.is_contiguous(), "D must be contiguous.");
  void* d_i_ptr = reinterpret_cast<void*>(D.data_ptr());
  for (size_t i = 0; i < A.size(); i++) {
    if (m_splits[i] == 0) continue;
    NVTE_CHECK(A[i].is_contiguous(), "A[", i, "] must be contiguous.");
    NVTE_CHECK(B[i].is_contiguous(), "B[", i, "] must be contiguous.");
    te_A.emplace_back(make_tensor(
        A[i].data_ptr(), {static_cast<size_t>(A[i].size(0)), static_cast<size_t>(A[i].size(1))},
        A_type, nullptr, nullptr, getDataPtr(A_scale_inverse[i], A_offset)));
    te_B.emplace_back(make_tensor(
        B[i].data_ptr(), {static_cast<size_t>(B[i].size(0)), static_cast<size_t>(B[i].size(1))},
        B_type, nullptr, nullptr, getDataPtr(B_scale_inverse, B_offset + i)));
    te_D.emplace_back(make_tensor(
        d_i_ptr, {static_cast<size_t>(m_splits[i]), static_cast<size_t>(A[i].size(0))}, D_type,
        getDataPtr(D_amax, D_offset + i), getDataPtr(D_scale, D_offset + i), nullptr));
    te_bias.emplace_back(make_tensor(bias[i].data_ptr(), {static_cast<size_t>(bias[i].size(0))},
                                     bias_type, nullptr, nullptr, nullptr));

    const auto gelu_shape = pre_gelu_out[i].data_ptr() == nullptr
                                ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out[i].size(0))}
                                : std::vector<size_t>{static_cast<size_t>(pre_gelu_out[i].size(0)),
                                                      static_cast<size_t>(pre_gelu_out[i].size(1))};
    te_pre_gelu_out.emplace_back(make_tensor(
        pre_gelu_out[i].data_ptr(), gelu_shape,
        GetTransformerEngineDType(pre_gelu_out[i].scalar_type()), nullptr, nullptr, nullptr));
    // Move the D pointer to the next split.
    char* char_ptr = reinterpret_cast<char*>(d_i_ptr);
    char_ptr += m_splits[i] * A[i].size(0) * D.element_size();
    d_i_ptr = reinterpret_cast<void*>(char_ptr);
  }
  for (size_t i = 0; i < workspace.size(); i++) {
    te_workspace.emplace_back(make_tensor(workspace[i].data_ptr(), {workspaceSize}, DType::kByte,
                                          nullptr, nullptr, nullptr));
  }

  // For now, we only have multi-stream cublas backend.
  nvte_multi_stream_cublas_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(),
                                te_pre_gelu_out.data(), te_A.size(), transa, transb, grad,
                                te_workspace.data(), accumulate, use_split_accumulator,
                                math_sm_count, at::cuda::getCurrentCUDAStream());
}
