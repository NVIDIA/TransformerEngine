/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/util/cuda_runtime.h"
#include "extensions.h"

void te_gemm(at::Tensor A, at::Tensor A_scale_inverse, transformer_engine::DType A_type,
             bool transa, at::Tensor B, at::Tensor B_scale_inverse,
             transformer_engine::DType B_type, bool transb, at::Tensor D, at::Tensor D_scale,
             transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
             transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
             at::Tensor workspace, size_t workspaceSize, bool accumulate,
             bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine;
  if (A.numel() == 0 || B.numel() == 0) {
    if (D.numel() != 0 && !accumulate) D.zero_();
    if (bias.numel() != 0 && grad) {
      if (B.numel() == 0) {
        bias.zero_();
      } else {
        bias.copy_(B.sum(0));
      }
    }
    if (pre_gelu_out.numel() != 0) pre_gelu_out.zero_();
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
    if (A[i].numel() == 0 || B[i].numel() == 0) {
      if (D[i].numel() != 0 && !accumulate) D[i].zero_();
      if (bias[i].numel() != 0 && grad) {
        if (B[i].numel() == 0) {
          bias[i].zero_();
        } else {
          bias[i].copy_(B[i].sum(0));
        }
      }
      if (pre_gelu_out[i].numel() != 0) pre_gelu_out[i].zero_();
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
    NVTE_CHECK(A[i].data_ptr() != nullptr, "A[", i, "] must not be nullptr.");
    NVTE_CHECK(B[i].data_ptr() != nullptr, "B[", i, "] must not be nullptr.");
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
