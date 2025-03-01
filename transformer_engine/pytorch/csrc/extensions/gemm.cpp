/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <Python.h>
#include <pybind11/pybind11.h>

#include <optional>
#include <string>

#include "../common.h"
#include "common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "extensions.h"
#include "pybind.h"
#include "transformer_engine/transformer_engine.h"

namespace {

void* get_data_ptr(MaybeTensor tensor) {
  if (tensor.has_value()) return tensor->data_ptr();
  return nullptr;
}

size_t get_size(MaybeTensor tensor, int dim) {
  if (tensor.has_value()) return static_cast<size_t>(tensor->size(dim));
  return 0;
}

}  // namespace

namespace transformer_engine::pytorch {

namespace detail {

std::vector<size_t> getGemmOutputShape(const NVTEShape& A_shape, const bool transa,
                                       const NVTEShape& B_shape, const bool transb) {
  // Flatten outer dims to get 2D matrices
  const size_t A0 = product(A_shape, 0, A_shape.ndim - 1);
  const size_t A1 = A_shape.data[A_shape.ndim - 1];
  const size_t B0 = product(B_shape, 0, B_shape.ndim - 1);
  const size_t B1 = B_shape.data[B_shape.ndim - 1];

  // Check matrix dims
  NVTE_CHECK((transa ? A1 : A0) == (transb ? B0 : B1), "Invalid matrix dimensions for GEMM (A=(",
             A0, ",", A1, "), transa=", transa, ", B=(", B0, ",", B1, "), transb=", transb, ")");

  // Construct output dims
  std::vector<size_t> ret;
  if (transb) {
    ret.emplace_back(B1);
  } else {
    // Unflatten B0
    for (size_t i = 0; i < B_shape.ndim - 1; ++i) {
      ret.emplace_back(B_shape.data[i]);
    }
  }
  if (transa) {
    ret.emplace_back(A0);
  } else {
    ret.emplace_back(A1);
  }
  return ret;
}

bool checkGemmShape(const std::vector<size_t>& expected, const NVTEShape& actual) {
  if (expected.size() != actual.ndim) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual.data[i]) return false;
  }
  return true;
}

}  // namespace detail

std::pair<TensorWrapper, py::object> createOutputTensor(const std::vector<size_t>& shape,
                                                        DType dtype, py::handle quantizer) {
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  return my_quantizer->create_tensor(shape, dtype);
}

std::vector<py::object> gemm(py::handle A, bool transa, py::handle B, bool transb, py::object D,
                             py::handle quantizer, std::optional<DType> out_dtype, MaybeTensor bias,
                             DType bias_type, bool gelu, MaybeTensor gelu_in, bool grad,
                             at::Tensor workspace, size_t workspaceSize, bool accumulate,
                             bool use_split_accumulator, CommOverlapCore* comm_overlap,
                             std::optional<CommOverlapType> comm_type, MaybeTensor extra_output,
                             bool bulk_overlap) {
  // Input tensors
  NVTE_CHECK(!A.is_none(), "Tensor A has not been provided");
  NVTE_CHECK(!B.is_none(), "Tensor B has not been provided");
  auto none = py::none();
  TensorWrapper A_tensor = makeTransformerEngineTensor(A, none);
  TensorWrapper B_tensor = makeTransformerEngineTensor(B, none);

  // Check tensor dimensions
  const auto& A_shape = A_tensor.shape();
  const auto& B_shape = B_tensor.shape();
  const auto& D_shape = detail::getGemmOutputShape(A_shape, transa, B_shape, transb);
  NVTE_CHECK(A_shape.ndim >= 1, "Tensor A needs to have at least 1 dimension");
  NVTE_CHECK(B_shape.ndim >= 1, "Tensor B needs to have at least 1 dimension");

  // Output tensor
  TensorWrapper D_tensor;
  if (D.is_none()) {
    DType output_dtype = out_dtype ? *out_dtype : A_tensor.dtype();
    std::tie(D_tensor, D) = createOutputTensor(D_shape, output_dtype, quantizer);
  } else {
    D_tensor = makeTransformerEngineTensor(D, quantizer);
    NVTE_CHECK(detail::checkGemmShape(D_shape, D_tensor.shape()),
               "GEMM output has invalid dims (expected ", std::to_string(D_shape), ", got ",
               std::to_string(D_tensor.shape()), ")");
    if (out_dtype) {
      NVTE_CHECK(*out_dtype == D_tensor.dtype(), "GEMM output has invalid dtype (expected ",
                 static_cast<int>(*out_dtype), ", found ", static_cast<int>(D_tensor.dtype()), ")");
    }
  }

  // Bias tensor
  TensorWrapper bias_tensor;
  MaybeTensor bias_grad = std::nullopt;
  if (bias.has_value()) {
    if (grad) {
      auto opts = torch::TensorOptions().dtype(GetATenDType(D_tensor.dtype())).device(torch::kCUDA);
      bias_grad = at::empty({static_cast<int64_t>(B_shape.data[B_shape.ndim - 1])}, opts);
      bias_tensor = makeTransformerEngineTensor(*bias_grad);
    } else {
      if (!bias->is_contiguous()) {
        bias = bias->contiguous();
      }
      bias_tensor = makeTransformerEngineTensor(*bias);
    }
  }

  // Activation input tensor
  MaybeTensor pre_gelu_out = std::nullopt;
  DType gelu_type = bias_type;
  if (gelu) {
    if (!grad) {
      auto dtype = GetATenDType(gelu_type);
      auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
      std::vector<int64_t> torch_shape;
      for (auto v : D_shape) {
        torch_shape.push_back(v);
      }
      pre_gelu_out = at::empty(torch_shape, opts);
    } else {
      if (gelu_in.has_value()) {
        pre_gelu_out = *gelu_in;
      }
    }
  }
  const auto gelu_shape = gelu ? D_shape : std::vector<size_t>{0};

  auto te_pre_gelu_out =
      makeTransformerEngineTensor(get_data_ptr(pre_gelu_out), gelu_shape, gelu_type);

  // Workspace
  auto te_workspace =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, DType::kByte);

  // Set an external SM Margin to all the GEMMs.
  // This comes in handy when DP is overlapped with GEMMs
  const int device_id = at::cuda::current_device();
  const int sm_count = transformer_engine::cuda::sm_count(device_id);
  int num_math_sms = sm_count - transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", sm_count);

  auto main_stream = at::cuda::getCurrentCUDAStream();
  if (A_tensor.numel() != 0 && B_tensor.numel() != 0) {
    if (comm_overlap) {
      // Prepare extra output tensor
      TensorWrapper extra_output_tensor;
      if (extra_output.has_value()) {
        extra_output_tensor = makeTransformerEngineTensor(*extra_output);
      } else {
        extra_output_tensor =
            makeTransformerEngineTensor(nullptr, std::vector<size_t>{0}, DType::kByte);
      }

      // Direct GEMM call to the correct overlap
      if (bulk_overlap) {
        comm_overlap->bulk_overlap(A_tensor, transa, B_tensor, transb, D_tensor, bias_tensor,
                                   te_pre_gelu_out, te_workspace, grad, accumulate,
                                   use_split_accumulator, comm_type.value(), extra_output_tensor,
                                   main_stream);
      } else if (comm_type.value() == CommOverlapType::AG) {
        if (comm_overlap->is_atomic_gemm()) {
          comm_overlap->atomic_gemm_overlap_ag(A_tensor, transa, B_tensor, transb, D_tensor,
                                               bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                               accumulate, use_split_accumulator,
                                               extra_output_tensor, main_stream);
        } else {
          comm_overlap->split_overlap_ag(A_tensor, transa, B_tensor, transb, D_tensor, bias_tensor,
                                         te_pre_gelu_out, te_workspace, grad, accumulate,
                                         use_split_accumulator, extra_output_tensor, main_stream);
        }
      } else {
        if (comm_overlap->is_atomic_gemm()) {
          comm_overlap->atomic_gemm_overlap_rs(A_tensor, transa, B_tensor, transb, D_tensor,
                                               bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                               accumulate, use_split_accumulator,
                                               extra_output_tensor, main_stream);
        } else {
          comm_overlap->split_overlap_rs(A_tensor, transa, B_tensor, transb, D_tensor, bias_tensor,
                                         te_pre_gelu_out, te_workspace, grad, accumulate,
                                         use_split_accumulator, extra_output_tensor, main_stream);
        }
      }
    } else {
      // Launch GEMM
      nvte_cublas_gemm(A_tensor.data(), B_tensor.data(), D_tensor.data(), bias_tensor.data(),
                       te_pre_gelu_out.data(), transa, transb, grad, te_workspace.data(),
                       accumulate, use_split_accumulator, num_math_sms, main_stream);
    }
  } else {
    if (D_tensor.numel() != 0 && !accumulate) {
      D_tensor.zero_(main_stream);
    }
    if (bias.has_value()) {
      if (bias->numel() != 0 && grad) {
        bias_grad->zero_();
      }
    }
  }

  // Pack outputs
  std::vector<py::object> out;
  out.emplace_back(std::move(D));
  out.emplace_back(py::cast(bias_grad));
  if (gelu && !grad) {
    out.emplace_back(py::cast(*pre_gelu_out));
  } else {
    out.emplace_back(py::none());
  }
  if (extra_output.has_value()) {
    out.emplace_back(py::cast(extra_output));
  } else {
    out.emplace_back(py::none());
  }
  return out;
}

}  // namespace transformer_engine::pytorch

void te_atomic_gemm(at::Tensor A, at::Tensor A_scale_inverse, transformer_engine::DType A_type,
                    std::vector<int64_t> A_scaling_mode, bool transa, at::Tensor B,
                    at::Tensor B_scale_inverse, transformer_engine::DType B_type,
                    std::vector<int64_t> B_scaling_mode, bool transb, at::Tensor D,
                    at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                    bool grad, at::Tensor workspace, size_t workspaceSize, bool accumulate,
                    bool use_split_accumulator, int math_sm_count, int m_split, int n_split,
                    bool gemm_producer, at::Tensor counter) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  // TODO: Handle scaling modes
  NVTEScalingMode nvte_scaling_modeA = NVTE_DELAYED_TENSOR_SCALING;
  NVTEScalingMode nvte_scaling_modeB = NVTE_DELAYED_TENSOR_SCALING;

  auto te_A = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inverse.data_ptr(), getTensorShape(A_scale_inverse),
      nvte_scaling_modeA);
  auto te_B = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inverse.data_ptr(), getTensorShape(B_scale_inverse),
      nvte_scaling_modeB);
  // TODO: D_scale_inv cannot be nullptr when D_type is FP8.
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

std::optional<std::vector<at::Tensor>> te_general_grouped_gemm(
    std::vector<py::handle> A, bool transa, std::vector<py::handle> B, bool transb,
    std::optional<std::vector<at::Tensor>> D, transformer_engine::DType D_type,
    std::vector<int64_t> m_splits, std::vector<at::Tensor> bias,
    transformer_engine::DType bias_type, bool single_output, std::vector<at::Tensor> pre_gelu_out,
    bool grad, std::vector<at::Tensor> workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;
  std::vector<NVTETensor> te_A_vector, te_B_vector, te_D_vector, te_bias_vector,
      te_pre_gelu_out_vector, te_workspace_vector;
  std::vector<TensorWrapper> wrappers;
  std::vector<at::Tensor> D_vectors;

  auto none = py::none();

  std::vector<size_t> single_output_begins;
  std::vector<size_t> single_output_ends;
  int slicing_dim;
  if (single_output && D == std::nullopt) {
    NVTE_ERROR("not implemented, D should be allocated for single output case.");
  }

  void* output_data_ptr;
  if (single_output) {
    output_data_ptr = (*D)[0].data_ptr();
  }

  for (size_t i = 0; i < A.size(); i++) {
    auto te_A = makeTransformerEngineTensor(A[i], none);
    auto te_B = makeTransformerEngineTensor(B[i], none);

    // if there is single output
    at::Tensor out_tensor;
    auto size_t_shape =
        pytorch::detail::getGemmOutputShape(te_A.shape(), transa, te_B.shape(), transb);
    std::vector<int64_t> D_shape;
    for (size_t t : size_t_shape) {
      D_shape.push_back(t);
    }
    auto dtype = GetATenDType(D_type);
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    if (single_output) {
      if (output_data_ptr == nullptr) {
        out_tensor = at::empty(D_shape, opts);
      } else {
        out_tensor = at::from_blob(output_data_ptr, D_shape, opts);
      }
      char* char_ptr = reinterpret_cast<char*>(output_data_ptr);
      char_ptr += D_shape[0] * D_shape[1] * (*D)[0].element_size();
      output_data_ptr = reinterpret_cast<void*>(char_ptr);
      D_vectors.emplace_back(out_tensor);
    } else {
      if (D == std::nullopt) {
        auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
        out_tensor = at::empty(D_shape, opts);
        D_vectors.emplace_back(out_tensor);
      } else {
        out_tensor = (*D)[i];
      }
    }

    if (te_A.numel() == 0 || te_B.numel() == 0) {
      if (out_tensor.numel() != 0 && !accumulate) out_tensor.zero_();
      if (bias[i].numel() != 0 && grad) {
        bias[i].zero_();
      }
      if (pre_gelu_out[i].numel() != 0) pre_gelu_out[i].zero_();
      continue;
    }

    auto te_D = makeTransformerEngineTensor(out_tensor);
    auto te_bias = makeTransformerEngineTensor(bias[i]);
    auto te_pre_gelu_out = makeTransformerEngineTensor(pre_gelu_out[i]);

    const auto gelu_shape = pre_gelu_out[i].data_ptr() == nullptr
                                ? std::vector<size_t>{static_cast<size_t>(te_pre_gelu_out.size(0))}
                                : std::vector<size_t>{static_cast<size_t>(te_pre_gelu_out.size(0)),
                                                      static_cast<size_t>(te_pre_gelu_out.size(1))};

    DType gelu_type = bias_type;
    te_pre_gelu_out =
        makeTransformerEngineTensor(get_data_ptr(pre_gelu_out[i]), gelu_shape, gelu_type);

    te_A_vector.emplace_back(te_A.data());
    te_B_vector.emplace_back(te_B.data());
    te_D_vector.emplace_back(te_D.data());
    te_bias_vector.emplace_back(te_bias.data());
    te_pre_gelu_out_vector.emplace_back(te_pre_gelu_out.data());

    wrappers.emplace_back(std::move(te_A));
    wrappers.emplace_back(std::move(te_B));
    wrappers.emplace_back(std::move(te_D));
    wrappers.emplace_back(std::move(te_bias));
    wrappers.emplace_back(std::move(te_pre_gelu_out));
  }
  for (size_t i = 0; i < workspace.size(); i++) {
    auto wsp = makeTransformerEngineTensor(workspace[i].data_ptr(), {workspaceSize}, DType::kByte);
    te_workspace_vector.emplace_back(wsp.data());
    wrappers.emplace_back(std::move(wsp));
  }
  // For now, we only have multi-stream cublas backend.
  nvte_multi_stream_cublas_gemm(te_A_vector.data(), te_B_vector.data(), te_D_vector.data(),
                                te_bias_vector.data(), te_pre_gelu_out_vector.data(),
                                te_A_vector.size(), transa, transb, grad,
                                te_workspace_vector.data(), accumulate, use_split_accumulator,
                                math_sm_count, at::cuda::getCurrentCUDAStream());
  return bias;
}
