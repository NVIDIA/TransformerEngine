/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind11/pybind11.h>

#include <optional>
#include <string>

#include "../common.h"
#include "../extensions.h"
#include "common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "pybind.h"
#include "transformer_engine/transformer_engine.h"
#include "util.h"

namespace {

void* get_data_ptr(transformer_engine::pytorch::MaybeTensor tensor) {
  if (tensor.has_value()) return tensor->data_ptr();
  return nullptr;
}

size_t get_size(transformer_engine::pytorch::MaybeTensor tensor, int dim) {
  if (tensor.has_value()) return static_cast<size_t>(tensor->size(dim));
  return 0;
}

}  // namespace

namespace transformer_engine::pytorch {

namespace detail {

bool is_low_precision(const DType type) {
  return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2;
}

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
                             bool bulk_overlap, float alpha, std::optional<float> beta) {
  using namespace transformer_engine::pytorch::detail;

  // Input tensors
  NVTE_CHECK(!A.is_none(), "Tensor A has not been provided");
  NVTE_CHECK(!B.is_none(), "Tensor B has not been provided");
  auto none = py::none();
  TensorWrapper A_tensor = makeTransformerEngineTensor(A, none);
  TensorWrapper B_tensor = makeTransformerEngineTensor(B, none);

  const bool low_precision =
      detail::is_low_precision(A_tensor.dtype()) || detail::is_low_precision(B_tensor.dtype());
  const bool fp8_block_scaling = A_tensor.scaling_mode() == NVTE_BLOCK_SCALING_1D ||
                                 A_tensor.scaling_mode() == NVTE_BLOCK_SCALING_2D ||
                                 B_tensor.scaling_mode() == NVTE_BLOCK_SCALING_1D ||
                                 B_tensor.scaling_mode() == NVTE_BLOCK_SCALING_2D;

  // Check tensor dimensions
  const auto& A_shape = A_tensor.shape();
  const auto& B_shape = B_tensor.shape();
  const auto& D_shape = detail::getGemmOutputShape(A_shape, transa, B_shape, transb);
  NVTE_CHECK(A_shape.ndim >= 1, "Tensor A needs to have at least 1 dimension");
  NVTE_CHECK(B_shape.ndim >= 1, "Tensor B needs to have at least 1 dimension");

  // Check scaling factors
  if (accumulate) {
    if (!beta) {
      beta = 1.0f;
    }
  } else {
    if (!beta) {
      beta = 0.0f;
    }
    NVTE_CHECK(beta == 0.0, "Trying to use non-zero beta while not accumulating ",
               "into D tensor. Beta has nothing to be applied to.");
  }

  DType output_dtype = out_dtype ? *out_dtype : A_tensor.dtype();
  // Output tensor
  TensorWrapper D_tensor;
  if (D.is_none()) {
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

  // maintain unquantized tensor in case we need unfused quantization support.
  TensorWrapper unquantized_D_tensor;
  py::object unquantized_out;
  // Unfused quantization is needed in the following cases
  // 1. Inputs: BF16, Output: FP8 (GEMM output has to be BF16, so FP8 quantization needed after that)
  // 2. Inputs: FP8, Output: FP8 (For any quantization apart from delayed scaling,
  // GEMM Output needs to be in BF16, to allow for unfused quantization)
  bool unfused_quantization_needed = !quantizer.is_none();
  if (low_precision) {
    // At the moment, only use-case for fused GEMM:
    // Delayed scaling quantizer with per-tensor scaling inputs
    bool is_per_tensor_scaling_input = IsFloat8Tensor(A.ptr()) || IsFloat8Tensor(B.ptr());
    if (IsFloat8Quantizers(quantizer.ptr()) && is_per_tensor_scaling_input)
      unfused_quantization_needed = false;
  }

  if (unfused_quantization_needed) {
    NoneQuantizer q{none};
    std::tie(unquantized_D_tensor, unquantized_out) = q.create_tensor(D_shape, output_dtype);
  }
  TensorWrapper& out_tensor = unfused_quantization_needed ? unquantized_D_tensor : D_tensor;

  // Bias tensor
  TensorWrapper bias_tensor;
  MaybeTensor bias_grad = std::nullopt;
  if (bias.has_value()) {
    if (grad) {
      auto opts =
          torch::TensorOptions().dtype(GetATenDType(out_tensor.dtype())).device(torch::kCUDA);
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
  DType gelu_type = low_precision ? bias_type : out_tensor.dtype();
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
  auto te_workspace = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  std::vector<size_t>{workspaceSize}, DType::kByte);

  // Set an external SM Margin to all the GEMMs.
  // This comes in handy when DP is overlapped with GEMMs
  const int device_id = at::cuda::current_device();
  const int sm_count = transformer_engine::cuda::sm_count(device_id);
  int num_math_sms = sm_count - transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", sm_count);

  // Construct GEMM config
  transformer_engine::MatmulConfigWrapper config;
  if (grad) {
    config.set_dbias_tensor(bias_tensor.data());
    config.set_with_dgelu_epilogue(gelu);
  } else {
    config.set_bias_tensor(bias_tensor.data());
    config.set_with_gelu_epilogue(gelu);
  }
  config.set_epilogue_aux_tensor(te_pre_gelu_out.data());
  config.set_use_split_accumulator(use_split_accumulator);
  config.set_sm_count(num_math_sms);

  // Keep the swizzled scaling factor tensors alive during the GEMM.
  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;
  auto main_stream = at::cuda::getCurrentCUDAStream();
  if (A_tensor.numel() != 0 && B_tensor.numel() != 0) {
    // Optionally swizzle the scaling factors
    swizzled_scale_inverses_list.emplace_back(std::move(swizzle_scaling_factors(A_tensor, transa)));
    swizzled_scale_inverses_list.emplace_back(
        std::move(swizzle_scaling_factors(B_tensor, !transb)));

    // Emulate the FP8 block scaling recipe with MXFP8 on Blackwell and newer
    // as it is not natively supported by cublasLt
    if (fp8_block_scaling && transformer_engine::cuda::sm_arch() >= 100) {
      // Convert tensors to mxfp8 and swizzle their scaling factors
      swizzled_scale_inverses_list.emplace_back(
          std::move(convert_block_scaling_to_mxfp8_tensor(A_tensor, transa)));
      swizzled_scale_inverses_list.emplace_back(
          std::move(convert_block_scaling_to_mxfp8_tensor(B_tensor, !transb)));
      // Use TN GEMM to avoid having to transpose data.
      transa = true;
      transb = false;
    }

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
        NVTE_SCOPED_GIL_RELEASE({
          comm_overlap->bulk_overlap(A_tensor, transa, B_tensor, transb, out_tensor, bias_tensor,
                                     te_pre_gelu_out, te_workspace, grad, accumulate,
                                     use_split_accumulator, comm_type.value(), extra_output_tensor,
                                     main_stream);
        });
      } else if (comm_type.value() == CommOverlapType::AG) {
        if (comm_overlap->is_atomic_gemm()) {
          NVTE_SCOPED_GIL_RELEASE({
            comm_overlap->atomic_gemm_overlap_ag(A_tensor, transa, B_tensor, transb, out_tensor,
                                                 bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                                 accumulate, use_split_accumulator,
                                                 extra_output_tensor, main_stream);
          });
        } else {
          NVTE_SCOPED_GIL_RELEASE({
            comm_overlap->split_overlap_ag(A_tensor, transa, B_tensor, transb, out_tensor,
                                           bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                           accumulate, use_split_accumulator, extra_output_tensor,
                                           main_stream);
          });
        }
      } else {
        if (comm_overlap->is_atomic_gemm()) {
          NVTE_SCOPED_GIL_RELEASE({
            comm_overlap->atomic_gemm_overlap_rs(A_tensor, transa, B_tensor, transb, out_tensor,
                                                 bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                                 accumulate, use_split_accumulator,
                                                 extra_output_tensor, main_stream);
          });
        } else {
          NVTE_SCOPED_GIL_RELEASE({
            comm_overlap->split_overlap_rs(A_tensor, transa, B_tensor, transb, out_tensor,
                                           bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                           accumulate, use_split_accumulator, extra_output_tensor,
                                           main_stream);
          });
        }
      }
    } else {
      // Launch GEMM
      NVTE_SCOPED_GIL_RELEASE({
        nvte_cublas_gemm_v2(transa, transb, &alpha, A_tensor.data(), B_tensor.data(), &beta.value(),
                            out_tensor.data(), out_tensor.data(), te_workspace.data(), config,
                            main_stream);
      });
    }
  } else {
    if (out_tensor.numel() != 0 && !accumulate) {
      out_tensor.zero_(main_stream);
    }
    if (bias.has_value()) {
      if (bias->numel() != 0 && grad) {
        bias_grad->zero_();
      }
    }
  }
  if (unfused_quantization_needed) {
    // Quantize the output
    std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
    my_quantizer->quantize(unquantized_D_tensor, D_tensor);
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

void te_atomic_gemm(at::Tensor A, at::Tensor A_scale_inverse, DType A_type,
                    std::vector<int64_t> A_scaling_mode, bool transa, at::Tensor B,
                    at::Tensor B_scale_inverse, DType B_type, std::vector<int64_t> B_scaling_mode,
                    bool transb, at::Tensor D, at::Tensor D_scale, DType D_type, at::Tensor D_amax,
                    at::Tensor bias, DType bias_type, at::Tensor pre_gelu_out, bool grad,
                    at::Tensor workspace, size_t workspaceSize, bool accumulate,
                    bool use_split_accumulator, int math_sm_count, int m_split, int n_split,
                    bool gemm_producer, at::Tensor counter) {
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
      D.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
  auto te_bias = makeTransformerEngineTensor(
      bias.data_ptr(), std::vector<size_t>{static_cast<size_t>(bias.size(0))}, bias_type);
  auto te_counter = makeTransformerEngineTensor(
      counter.data_ptr(), std::vector<size_t>{static_cast<size_t>(counter.size(0))}, DType::kInt32);

  const auto gelu_shape = pre_gelu_out.data_ptr() == nullptr
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto te_pre_gelu_out = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));
  auto te_workspace = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  std::vector<size_t>{workspaceSize}, DType::kByte);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_cublas_atomic_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(),
                            te_pre_gelu_out.data(), transa, transb, grad, te_workspace.data(),
                            accumulate, use_split_accumulator, math_sm_count, m_split, n_split,
                            gemm_producer, te_counter.data(), at::cuda::getCurrentCUDAStream());
  });
}

std::optional<std::vector<at::Tensor>> te_general_grouped_gemm(
    std::vector<py::handle> A, bool transa, std::vector<py::handle> B, bool transb,
    std::optional<std::vector<at::Tensor>> D, DType D_type, at::Tensor m_splits,
    std::vector<at::Tensor> bias, DType bias_type, bool single_output,
    std::vector<at::Tensor> pre_gelu_out, bool grad, std::vector<at::Tensor> workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, int math_sm_count) {
  if (single_output && D == std::nullopt) {
    NVTE_ERROR("not implemented, D should be allocated for single output case.");
  }

  void* output_data_ptr = nullptr;
  if (single_output) {
    output_data_ptr = (*D)[0].data_ptr();
  }

  const auto none = py::none();
  std::vector<TensorWrapper> te_A_wrappers, te_B_wrappers, te_D_wrappers, te_bias_wrappers,
      te_pre_gelu_out_wrappers;
  std::vector<at::Tensor> D_vectors;
  for (size_t i = 0; i < A.size(); i++) {
    auto te_A = makeTransformerEngineTensor(A[i], none);
    auto te_B = makeTransformerEngineTensor(B[i], none);

    // if there is single output
    at::Tensor out_tensor;
    auto size_t_shape =
        pytorch::detail::getGemmOutputShape(te_A.shape(), transa, te_B.shape(), transb);
    bool D_numel_is_zero = false;
    std::vector<int64_t> D_shape;
    for (size_t t : size_t_shape) {
      D_shape.push_back(t);
      if (t == 0) {
        D_numel_is_zero = true;
      }
    }
    auto dtype = GetATenDType(D_type);
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    if (single_output) {
      if (output_data_ptr == nullptr) {
        out_tensor = at::empty(D_shape, opts);
      } else {
        // We need to check !D_numel_is_zero because if the final input portion has zero elements,
        // output_data_ptr would point beyond the allocated memory of D. This would cause
        // at::from_blob to fail as it would reference memory not allocated by CUDA.
        if (!D_numel_is_zero) {
          out_tensor = at::from_blob(output_data_ptr, D_shape, opts);
        }
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

    te_A_wrappers.emplace_back(std::move(te_A));
    te_B_wrappers.emplace_back(std::move(te_B));
    te_D_wrappers.emplace_back(std::move(te_D));
    te_bias_wrappers.emplace_back(std::move(te_bias));
    te_pre_gelu_out_wrappers.emplace_back(std::move(te_pre_gelu_out));
  }

  // Keep the swizzled scaling factor tensors alive during the GEMM.
  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;

  // Optionally swizzle the scaling factors
  swizzled_scale_inverses_list.emplace_back(
      multi_tensor_swizzle_scaling_factors(te_A_wrappers, transa));
  swizzled_scale_inverses_list.emplace_back(
      multi_tensor_swizzle_scaling_factors(te_B_wrappers, !transb));

  // Emulate the FP8 block scaling recipe with MXFP8 on Blackwell and newer
  // as it is not natively supported by cublasLt
  if (transformer_engine::cuda::sm_arch() >= 100) {
    // Check if is using FP8 block scaling
    bool exists_tensor_using_fp8_block_scaling = false;
    bool exists_tensor_not_using_fp8_block_scaling = false;
    for (const auto& tensor_wrappers : {&te_A_wrappers, &te_B_wrappers}) {
      for (const TensorWrapper& tensor : *tensor_wrappers) {
        const NVTEScalingMode scaling_mode = tensor.scaling_mode();
        if (scaling_mode == NVTE_BLOCK_SCALING_1D || scaling_mode == NVTE_BLOCK_SCALING_2D)
          exists_tensor_using_fp8_block_scaling = true;
        else
          exists_tensor_not_using_fp8_block_scaling = true;
      }
    }
    if (exists_tensor_using_fp8_block_scaling) {
      NVTE_CHECK(!exists_tensor_not_using_fp8_block_scaling,
                 "Either all tensors or no tensor must be FP8 block scaling tensors");
      // Convert tensors to mxfp8 and swizzle their scaling factors
      for (TensorWrapper& A_tensor : te_A_wrappers) {
        swizzled_scale_inverses_list.emplace_back(
            convert_block_scaling_to_mxfp8_tensor(A_tensor, transa));
      }
      for (TensorWrapper& B_tensor : te_B_wrappers) {
        swizzled_scale_inverses_list.emplace_back(
            convert_block_scaling_to_mxfp8_tensor(B_tensor, !transb));
      }
      // Use TN GEMM to avoid having to transpose data.
      transa = true;
      transb = false;
    }
  }

  std::vector<NVTETensor> te_A_vector, te_B_vector, te_D_vector, te_bias_vector,
      te_pre_gelu_out_vector;
  for (size_t i = 0; i < te_A_wrappers.size(); i++) {
    te_A_vector.emplace_back(te_A_wrappers[i].data());
    te_B_vector.emplace_back(te_B_wrappers[i].data());
    te_D_vector.emplace_back(te_D_wrappers[i].data());
    te_bias_vector.emplace_back(te_bias_wrappers[i].data());
    te_pre_gelu_out_vector.emplace_back(te_pre_gelu_out_wrappers[i].data());
  }

  std::vector<NVTETensor> te_workspace_vector;
  std::vector<TensorWrapper> te_workspace_wrappers;
  for (size_t i = 0; i < workspace.size(); i++) {
    auto wsp = makeTransformerEngineTensor(workspace[i].data_ptr(),
                                           std::vector<size_t>{workspaceSize}, DType::kByte);
    te_workspace_vector.emplace_back(wsp.data());
    te_workspace_wrappers.emplace_back(std::move(wsp));
  }

  // For now, we only have multi-stream cublas backend.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_multi_tensor_gemm(te_A_vector.data(), te_B_vector.data(), te_D_vector.data(),
                           te_bias_vector.data(), te_pre_gelu_out_vector.data(), te_A_vector.size(),
                           transa, transb, grad, te_workspace_vector.data(), accumulate,
                           use_split_accumulator, math_sm_count, at::cuda::getCurrentCUDAStream());
  });
  return bias;
}

// Index of the next available slot in the pinned host buffer. Concurrency control is not implemented
// here, as the current execution model is strictly single-threaded. For CUDA Graph, we need to use a
// pinned host buffer to avoid illegal memory access during H2D copy.
// To avoid synchronization after H2D copy, each operator instance must use its own host buffer to
// prevent data races — e.g., one H2D copy may still be running while another operator attempts to
// reuse and overwrite the same buffer.
// A global variable is used because the function doesn't know how many instances there are and which
// instance is calling.
int pinned_host_buffer_index = 0;

std::optional<std::vector<at::Tensor>> te_general_device_initiated_grouped_gemm(
    std::vector<py::handle> A, bool transa, std::vector<py::handle> B, bool transb,
    std::optional<std::vector<at::Tensor>> D, DType D_type, at::Tensor m_splits,
    std::vector<at::Tensor> bias, DType bias_type, bool single_output,
    std::vector<at::Tensor> pre_gelu_out, bool grad, bool wgrad, std::vector<at::Tensor> workspace,
    size_t workspaceSize, bool accumulate, std::optional<at::Tensor> accumulate_mask,
    bool use_split_accumulator, int math_sm_count) {
  if (D == std::nullopt) {
    NVTE_ERROR("not implemented, D should be allocated.");
  }

  const auto none = py::none();
  // Keep tensors alive during the GEMM.
  std::vector<TensorWrapper> te_A_wrappers, te_B_wrappers, te_D_wrappers, te_bias_wrappers,
      te_pre_gelu_out_wrappers, te_workspace_wrappers;
  std::vector<at::Tensor> D_vectors;
  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;
  std::vector<NVTETensor> te_A_vector, te_B_vector, te_D_vector, te_bias_vector,
      te_pre_gelu_out_vector, te_workspace_vector;

  bool* accumulate_mask_ptr =
      accumulate_mask != std::nullopt ? (*accumulate_mask).data_ptr<bool>() : nullptr;

  NVTE_CHECK(m_splits.dtype() == torch::kInt64, "Data type of m_splits should be int64.");
  NVTE_CHECK(B.size() == 1,
             "Grouped GEMM input B should not be splited when m_splits is on device.");

  auto te_B = makeTransformerEngineTensor(B[0], none);

  if (!wgrad) {  // fprop or dgrad
    NVTE_CHECK(!transb,
               "Not implemented, Grouped GEMM input B should not be transposed for fprop and "
               "dgrad when when m_splits is on device.");
    NVTE_CHECK(single_output,
               "single_output=False is not supported for fprop and dgrad when when m_splits is "
               "on device.");
    if (te_B.numel() == 0) {  // skip the GEMM
      auto te_D = makeTransformerEngineTensor((*D)[0]);
      if (te_D.numel() != 0) {
        (*D)[0].zero_();
      }
      return bias;
    }

    te_B_vector.emplace_back(te_B.data());
    te_B_wrappers.emplace_back(std::move(te_B));
    const int num_gemms = A.size();
    for (size_t i = 0; i < num_gemms; i++) {
      auto te_A = makeTransformerEngineTensor(A[i], none);
      te_A_vector.emplace_back(te_A.data());
      te_A_wrappers.emplace_back(std::move(te_A));
    }

    // Optionally swizzle the scaling factors
    swizzled_scale_inverses_list.emplace_back(
        multi_tensor_swizzle_scaling_factors(te_A_wrappers, transa));
    swizzled_scale_inverses_list.emplace_back(
        multi_tensor_swizzle_scaling_factors(te_B_wrappers, !transb));

    // Prepare addresses array of input A and scaling factors.
    // To enable CUDA Graph, we need to use a pinned host buffer to avoid illegal memory access during H2D copy.
    at::Tensor inputA_and_SF_addrs;
    if (at::cuda::currentStreamCaptureStatusMayInitCtx() != at::cuda::CaptureStatus::None) {
      NVTE_CHECK(pinned_host_buffer_index + num_gemms * 2 <= workspace[1].size(0),
                 "Pinned host buffer out of bounds, please increase the capacity by setting "
                 "NVTE_CUTLASS_HOST_PINNED_U64_CAPACITY. "
                 "Current buffer size: ",
                 workspace[1].size(0));
      inputA_and_SF_addrs = workspace[1].narrow(0, pinned_host_buffer_index, num_gemms * 2);
      pinned_host_buffer_index += num_gemms * 2;
    } else {
      // For eager mode, use a temporary tensor to prevent exhausting the global workspace.
      auto options = at::TensorOptions().dtype(torch::kUInt64).pinned_memory(true);
      // Utilise torch tensor management to ensure memory is retained until the H2D copy is complete.
      inputA_and_SF_addrs = at::empty(num_gemms * 2, options);
    }
    int gemm_m;
    auto* inputA_and_SF_addr_ptr = inputA_and_SF_addrs.data_ptr<uint64_t>();
    for (size_t i = 0; i < num_gemms; i++) {
      transformer_engine::Tensor* inputA = convertNVTETensor(te_A_vector[i]);
      gemm_m = transa ? inputA->flat_first_dim() : inputA->flat_last_dim();
      if (transa) {
        NVTE_CHECK(inputA->has_data(), "Input A is missing row-wise usage");
      } else {
        NVTE_CHECK(inputA->has_columnwise_data(), "Input A is missing column-wise usage");
      }
      inputA_and_SF_addr_ptr[i] =
          transa ? static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(inputA->data.dptr))
                 : static_cast<uint64_t>(
                       reinterpret_cast<std::uintptr_t>(inputA->columnwise_data.dptr));
      inputA_and_SF_addr_ptr[num_gemms + i] =
          transa ? static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(inputA->scale_inv.dptr))
                 : static_cast<uint64_t>(
                       reinterpret_cast<std::uintptr_t>(inputA->columnwise_scale_inv.dptr));
    }
    // H2D copy
    at::Tensor inputA_and_SF_addrs_cuda = at::from_blob(workspace[0].data_ptr(), {num_gemms * 2}, torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA));
    inputA_and_SF_addrs_cuda.copy_(inputA_and_SF_addrs, /*non_blocking=*/true);

    auto te_D = makeTransformerEngineTensor((*D)[0]);
    te_D_vector.emplace_back(te_D.data());
    te_D_wrappers.emplace_back(std::move(te_D));

    auto wsp = makeTransformerEngineTensor(workspace[0].data_ptr(),
                                           std::vector<size_t>{workspaceSize}, DType::kByte);
    te_workspace_vector.emplace_back(wsp.data());
    te_workspace_wrappers.emplace_back(std::move(wsp));

    NVTE_SCOPED_GIL_RELEASE({
      nvte_device_cutlass_grouped_gemm(
          reinterpret_cast<const void**>(inputA_and_SF_addrs_cuda.data_ptr()), te_B_vector.data(),
          te_D_vector.data(), reinterpret_cast<int64_t*>(m_splits.data_ptr()), gemm_m,
          te_bias_vector.data(), te_pre_gelu_out_vector.data(), te_A_vector.size(), transa, transb,
          grad, te_workspace_vector.data(), workspaceSize, use_split_accumulator, math_sm_count,
          at::cuda::getCurrentCUDAStream());
    });
  } else {  // wgrad
    NVTE_CHECK(!transa,
               "Not implemented, Grouped GEMM input A should not be transposed for wgrad when "
               "m_splits is on device.");
    NVTE_CHECK(transb,
               "Not implemented, Grouped GEMM input B should be transposed for wgrad when "
               "m_splits is on device.");
    NVTE_CHECK(B.size() == 1,
               "Grouped GEMM input B should not be splited for wgrad when m_splits is on device.");
    NVTE_CHECK(D != std::nullopt,
               "Grouped GEMM output D should be allocated for wgrad when m_splits is on device.");
    // TODO: handle single_output case
    NVTE_CHECK(
        !single_output,
        "Not implemented, single output is not supported for wgrad when m_splits is on device.");

    auto te_A = makeTransformerEngineTensor(A[0], none);

    if (te_A.numel() == 0 || te_B.numel() == 0) {  // skip the GEMM
      for (size_t i = 0; i < (*D).size(); i++) {
        if (!accumulate || (accumulate_mask_ptr && !accumulate_mask_ptr[i])) {
          auto te_D = makeTransformerEngineTensor((*D)[i]);
          if (te_D.numel() != 0) {
            (*D)[i].zero_();
          }
        }
      }
      return bias;
    }

    te_A_vector.emplace_back(te_A.data());
    te_A_wrappers.emplace_back(std::move(te_A));
    te_B_vector.emplace_back(te_B.data());
    te_B_wrappers.emplace_back(std::move(te_B));
    // Optionally swizzle the scaling factors
    swizzled_scale_inverses_list.emplace_back(
        multi_tensor_swizzle_scaling_factors(te_A_wrappers, transa));
    swizzled_scale_inverses_list.emplace_back(
        multi_tensor_swizzle_scaling_factors(te_B_wrappers, !transb));

    const int num_gemms = (*D).size();
    for (size_t i = 0; i < num_gemms; i++) {
      auto te_D = makeTransformerEngineTensor((*D)[i]);
      te_D_vector.emplace_back(te_D.data());
      te_D_wrappers.emplace_back(std::move(te_D));
    }

    // Prepare addresses array of output D.
    // To enable CUDA Graph, we need to use a pinned host buffer to avoid illegal memory access during H2D copy.
    at::Tensor outputD_addrs;
    if (at::cuda::currentStreamCaptureStatusMayInitCtx() != at::cuda::CaptureStatus::None) {
      NVTE_CHECK(pinned_host_buffer_index + num_gemms <= workspace[1].size(0),
                 "Pinned host buffer out of bounds, please increase the capacity by setting "
                 "NVTE_CUTLASS_HOST_PINNED_U64_CAPACITY. "
                 "Current buffer size: ",
                 workspace[1].size(0));
      outputD_addrs = workspace[1].narrow(0, pinned_host_buffer_index, num_gemms);
      pinned_host_buffer_index += num_gemms;
    } else {
      // For eager mode, use a temporary tensor to prevent exhausting the global workspace.
      auto options = at::TensorOptions().dtype(torch::kUInt64).pinned_memory(true);
      // Utilise torch tensor management to ensure memory is retained until the H2D copy is complete.
      outputD_addrs = at::empty(num_gemms, options);
    }

    auto* outputD_addrs_ptr = outputD_addrs.data_ptr<uint64_t>();
    for (size_t i = 0; i < num_gemms; i++) {
      transformer_engine::Tensor* outputD = convertNVTETensor(te_D_vector[i]);
      NVTE_CHECK(outputD->has_data(), "Input D is missing row-wise usage");
      outputD_addrs_ptr[i] =
          static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(outputD->data.dptr));
    }
    // H2D copy
    at::Tensor outputD_addrs_cuda = at::from_blob(workspace[0].data_ptr(), {num_gemms}, torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA));
    outputD_addrs_cuda.copy_(outputD_addrs, /*non_blocking=*/true);

    auto wsp = makeTransformerEngineTensor(workspace[0].data_ptr(),
                                           std::vector<size_t>{workspaceSize}, DType::kByte);
    te_workspace_vector.emplace_back(wsp.data());
    te_workspace_wrappers.emplace_back(std::move(wsp));

    NVTE_SCOPED_GIL_RELEASE({
      nvte_device_cutlass_grouped_gemm_wgrad(
          te_A_vector.data(), te_B_vector.data(),
          reinterpret_cast<void**>(outputD_addrs_cuda.data_ptr()), D_type,
          reinterpret_cast<int64_t*>(m_splits.data_ptr()), te_bias_vector.data(),
          te_pre_gelu_out_vector.data(), te_D_vector.size(), transa, transb,
          te_workspace_vector.data(), workspaceSize, accumulate, accumulate_mask_ptr,
          use_split_accumulator, math_sm_count, at::cuda::getCurrentCUDAStream());
    });
  }

  return bias;
}

}  // namespace transformer_engine::pytorch
