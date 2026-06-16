/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../extensions.h"
#include "../pybind.h"
#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "transformer_engine/activation.h"
#include "transformer_engine/gemm.h"
#include "transformer_engine/transformer_engine.h"

namespace py = pybind11;

namespace transformer_engine::pytorch {
namespace {

constexpr int64_t kGroupedGemmCublasWorkspaceSize = 32 * 1024 * 1024 + 1024;

bool is_none(py::handle obj) { return obj.is_none(); }

std::vector<size_t> tensor_shape_1d(const at::Tensor &tensor) {
  return {static_cast<size_t>(tensor.numel())};
}

at::Tensor maybe_cast_dtype(const at::Tensor &tensor, at::ScalarType dtype) {
  at::Tensor out = tensor;
  if (out.scalar_type() != dtype) {
    out = out.to(out.options().dtype(dtype));
  }
  return out;
}

void check_contiguous(const at::Tensor &tensor, const std::string &name) {
  NVTE_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

size_t num_groups_from_prepared_split_sizes(const at::Tensor &split_sizes,
                                            const c10::Device &device) {
  NVTE_CHECK(split_sizes.dim() == 1, "split_sizes must be a 1D tensor.");
  NVTE_CHECK(split_sizes.device() == device, "split_sizes must be on the current CUDA device.");
  NVTE_CHECK(split_sizes.scalar_type() == at::kLong,
             "split_sizes must be the int64 CUDA tensor returned by splits_to_offsets_multi.");
  return static_cast<size_t>(split_sizes.numel());
}

GroupedTensorWrapper make_grouped_tensor(const at::Tensor &data,
                                         const at::Tensor &prepared_split_sizes,
                                         const at::Tensor &tensor_offsets,
                                         int64_t logical_last_dim) {
  const auto num_groups = static_cast<size_t>(prepared_split_sizes.numel());
  NVTE_CHECK(data.numel() % logical_last_dim == 0,
             "Grouped tensor storage is not divisible by logical last dimension.");
  const auto total_tokens = static_cast<size_t>(data.numel() / logical_last_dim);
  auto grouped = GroupedTensorWrapper(
      num_groups, std::vector<size_t>{total_tokens, static_cast<size_t>(logical_last_dim)});
  grouped.set_rowwise_data(data.data_ptr(), GetTransformerEngineDType(data.scalar_type()),
                           tensor_shape_1d(data));
  grouped.set_first_dims(prepared_split_sizes.data_ptr(), DType::kInt64,
                         std::vector<size_t>{num_groups});
  grouped.set_tensor_offsets(tensor_offsets.data_ptr(), DType::kInt64,
                             std::vector<size_t>{num_groups + 1});
  return grouped;
}

GroupedTensorWrapper make_uniform_grouped_tensor(at::Tensor data, size_t num_groups,
                                                 int64_t first_dim, int64_t last_dim) {
  auto grouped = GroupedTensorWrapper(
      num_groups, std::vector<size_t>{num_groups * static_cast<size_t>(first_dim),
                                      static_cast<size_t>(last_dim)});
  grouped.set_rowwise_data(data.data_ptr(), GetTransformerEngineDType(data.scalar_type()),
                           tensor_shape_1d(data));
  return grouped;
}

struct GroupedWeightArg {
  bool is_grouped = false;
  at::Tensor packed;
  std::vector<at::Tensor> discrete;
  // Logical per-expert weight shape. For both supported layouts:
  // - packed single grouped weight: packed has shape [G, rows, cols]
  // - discrete weights: each tensor has shape [rows, cols]
  // rows = out_features, cols = in_features.
  int64_t rows = 0;
  int64_t cols = 0;

  c10::Device device() const { return is_grouped ? packed.device() : discrete[0].device(); }
};

GroupedWeightArg weight_arg_from_py(py::handle arg, size_t num_groups, at::ScalarType dtype,
                                    const std::string &name) {
  GroupedWeightArg out;
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    auto seq = py::reinterpret_borrow<py::sequence>(arg);
    NVTE_CHECK(static_cast<size_t>(seq.size()) == num_groups, name, " must have ", num_groups,
               " tensors.");
    out.discrete.reserve(num_groups);
    for (size_t i = 0; i < num_groups; ++i) {
      auto tensor = maybe_cast_dtype(seq[i].cast<at::Tensor>(), dtype);
      check_contiguous(tensor, name);
      NVTE_CHECK(tensor.dim() == 2, name, " tensors must be rank-2.");
      if (i == 0) {
        // Discrete case: each expert owns one [out_features, in_features]
        // tensor. Cache the shared logical shape for later GEMM setup.
        out.rows = tensor.size(0);
        out.cols = tensor.size(1);
      } else {
        NVTE_CHECK(tensor.size(0) == out.rows && tensor.size(1) == out.cols, name,
                   " tensors must have a uniform shape.");
      }
      out.discrete.emplace_back(tensor);
    }
    return out;
  }

  out.packed = maybe_cast_dtype(arg.cast<at::Tensor>(), dtype);
  NVTE_CHECK(out.packed.dim() == 3, name, " must be a tensor with shape [num_groups, rows, cols].");
  NVTE_CHECK(static_cast<size_t>(out.packed.size(0)) == num_groups, name,
             " first dimension must be ", num_groups, ".");
  check_contiguous(out.packed, name);
  out.is_grouped = true;
  // Packed case: a single [G, out_features, in_features] tensor stores all
  // experts, so dimensions 1 and 2 are the same per-expert logical shape.
  out.rows = out.packed.size(1);
  out.cols = out.packed.size(2);
  return out;
}

at::Tensor packed_bias_from_arg(py::handle arg, size_t num_groups, at::ScalarType dtype,
                                int64_t out_features, const std::string &name) {
  if (is_none(arg)) {
    return at::Tensor();
  }

  auto packed = maybe_cast_dtype(arg.cast<at::Tensor>(), dtype);
  NVTE_CHECK(packed.dim() == 2, name, " must be a tensor with shape [num_groups, features].");
  NVTE_CHECK(static_cast<size_t>(packed.size(0)) == num_groups, name, " first dimension must be ",
             num_groups, ".");
  NVTE_CHECK(packed.size(1) == out_features, name, " second dimension must be ", out_features, ".");
  check_contiguous(packed, name);
  return packed;
}

std::vector<NVTETensor> nvte_tensor_list_from_tensors(const std::vector<at::Tensor> &tensors,
                                                      std::vector<TensorWrapper> *wrappers) {
  wrappers->clear();
  wrappers->reserve(tensors.size());
  std::vector<NVTETensor> out;
  out.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    wrappers->emplace_back(makeTransformerEngineTensor(tensor));
    out.emplace_back(wrappers->back().data());
  }
  return out;
}

int grouped_gemm_math_sm_count(const c10::Device &device) {
  const int device_id = static_cast<int>(device.index());
  const int sm_count = transformer_engine::cuda::sm_count(device_id);
  return sm_count - transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", sm_count);
}

std::array<at::Tensor, 4> grouped_gemm_scratch_from_arg(py::handle scratch,
                                                        const c10::Device &device,
                                                        size_t num_groups) {
  const int64_t num_groups_i64 = static_cast<int64_t>(num_groups);
  const int64_t setup_size =
      static_cast<int64_t>(nvte_get_grouped_gemm_setup_workspace_size(num_groups));

  if (is_none(scratch)) {
    return {
        at::ones({num_groups_i64}, at::device(device).dtype(at::kFloat)),
        at::zeros({num_groups_i64}, at::device(device).dtype(at::kFloat)),
        at::empty({setup_size}, at::device(device).dtype(at::kByte)),
        at::empty({kGroupedGemmCublasWorkspaceSize}, at::device(device).dtype(at::kByte)),
    };
  }

  NVTE_CHECK(py::isinstance<py::tuple>(scratch) || py::isinstance<py::list>(scratch),
             "megacpp grouped MLP GEMM scratch must be None or a 4-tensor tuple/list.");
  auto seq = py::reinterpret_borrow<py::sequence>(scratch);
  NVTE_CHECK(seq.size() == 4, "megacpp grouped MLP GEMM scratch must have 4 tensors.");

  std::array<at::Tensor, 4> tensors = {
      seq[0].cast<at::Tensor>(),
      seq[1].cast<at::Tensor>(),
      seq[2].cast<at::Tensor>(),
      seq[3].cast<at::Tensor>(),
  };
  return tensors;
}

struct GroupedGemmResources {
  c10::Device device;
  size_t num_groups;
  at::Tensor alpha;
  at::Tensor beta_zero;
  at::Tensor beta_one;
  at::Tensor setup;
  at::Tensor cublas;
  TensorWrapper te_alpha;
  TensorWrapper te_beta_zero;
  TensorWrapper te_beta_one;
  TensorWrapper te_setup;
  TensorWrapper te_cublas;
  std::optional<GroupedMatmulConfigWrapper> config;

  GroupedGemmResources(const c10::Device &device_, size_t num_groups_,
                       std::array<at::Tensor, 4> scratch)
      : device(device_),
        num_groups(num_groups_),
        alpha(std::move(scratch[0])),
        beta_zero(std::move(scratch[1])),
        beta_one(alpha),
        setup(std::move(scratch[2])),
        cublas(std::move(scratch[3])),
        te_alpha(makeTransformerEngineTensor(alpha)),
        te_beta_zero(makeTransformerEngineTensor(beta_zero)),
        te_beta_one(makeTransformerEngineTensor(beta_one)),
        te_setup(makeTransformerEngineTensor(
            setup.data_ptr(), std::vector<size_t>{static_cast<size_t>(setup.numel())},
            DType::kByte)),
        te_cublas(makeTransformerEngineTensor(
            cublas.data_ptr(), std::vector<size_t>{static_cast<size_t>(cublas.numel())},
            DType::kByte)) {
    // These scratch tensors may be cached by Python per CUDA stream. Every
    // current megacpp grouped GEMM below is enqueued on at::cuda::getCurrentCUDAStream(),
    // so same-stream ordering protects workspace reuse. If a future backend
    // uses auxiliary streams, cache keys or stream recording must be revisited.
    const int math_sm_count = grouped_gemm_math_sm_count(device);
    if (math_sm_count > 0) {
      config.emplace();
      config->set_sm_count(math_sm_count);
    }
  }

  NVTETensor beta(bool accumulate) { return accumulate ? te_beta_one.data() : te_beta_zero.data(); }

  NVTEGroupedMatmulConfig config_data() {
    return config.has_value() ? static_cast<NVTEGroupedMatmulConfig>(*config) : nullptr;
  }
};

GroupedGemmResources make_grouped_mlp_backend_resources(const c10::Device &device,
                                                        size_t num_groups, py::handle scratch) {
  // Keep the backend resource policy private to megacpp. Today this is cuBLAS
  // grouped GEMM scratch; future backends can change this helper without
  // changing the Python or pybind contract.
  return GroupedGemmResources(device, num_groups,
                              grouped_gemm_scratch_from_arg(scratch, device, num_groups));
}

void grouped_gemm(GroupedTensorWrapper *A, bool transa, GroupedTensorWrapper *B, bool transb,
                  GroupedTensorWrapper *D, GroupedGemmResources *resources, bool accumulate) {
  NVTE_SCOPED_GIL_RELEASE({
    nvte_grouped_gemm(A->data(), transa, B->data(), transb, D->data(), D->data(),
                      resources->te_alpha.data(), resources->beta(accumulate),
                      resources->te_setup.data(), resources->te_cublas.data(),
                      resources->config_data(), at::cuda::getCurrentCUDAStream());
  });
}

std::vector<at::Tensor> output_tensor_list_from_arg(py::handle arg, size_t num_groups, int64_t rows,
                                                    int64_t cols, const std::string &name) {
  std::vector<at::Tensor> out;
  if (is_none(arg)) {
    return out;
  }
  out.reserve(num_groups);
  // This helper is intentionally only for the discrete-weight external wgrad
  // path, where Megatron provides one main_grad tensor per expert. The packed
  // [G, rows, cols] external buffer used by single grouped weight is handled in
  // wgrad_output_from_arg so it can stay packed and use grouped-tensor GEMM.
  NVTE_CHECK(py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg), name,
             " must be a list or tuple of wgrad output tensors.");
  auto seq = py::reinterpret_borrow<py::sequence>(arg);
  NVTE_CHECK(static_cast<size_t>(seq.size()) == num_groups, name, " must have ", num_groups,
             " tensors.");
  for (size_t i = 0; i < num_groups; ++i) {
    auto tensor = seq[i].cast<at::Tensor>();
    NVTE_CHECK(tensor.is_cuda(), name, " tensors must be CUDA tensors.");
    // Do not require tensor.scalar_type() == dtype. Caller-owned
    // main_grad buffers are allocated by Megatron and may be FP32 even when TE
    // grouped MLP compute is BF16.
    NVTE_CHECK(tensor.dim() == 2, name, " tensors must be rank-2 wgrad buffers.");
    NVTE_CHECK(tensor.size(0) == rows && tensor.size(1) == cols, name,
               " tensors must have shape [rows, cols].");
    check_contiguous(tensor, name);
    out.emplace_back(tensor);
  }
  return out;
}

struct WgradOutput {
  std::vector<at::Tensor> tensors;
  at::Tensor packed;
  bool is_grouped = false;
  bool owns_storage = false;
};

WgradOutput wgrad_output_from_arg(py::handle arg, bool compute_wgrad, size_t num_groups,
                                  at::ScalarType dtype, const c10::Device &device, int64_t rows,
                                  int64_t cols, const std::string &name,
                                  bool prefer_grouped_output) {
  WgradOutput out;
  if (!compute_wgrad) {
    return out;
  }
  if (is_none(arg)) {
    // Cases 1 and 2: no external wgrad buffer was provided, so C++ owns the
    // allocation. Single grouped weight keeps this packed as [G, N, K];
    // discrete weights split the same packed allocation into per-expert views.
    out.packed =
        at::empty({static_cast<int64_t>(num_groups), rows, cols}, at::device(device).dtype(dtype));
    out.owns_storage = true;
    out.is_grouped = prefer_grouped_output;
    if (out.is_grouped) {
      return out;
    }
    out.tensors.reserve(num_groups);
    for (size_t i = 0; i < num_groups; ++i) {
      out.tensors.emplace_back(out.packed.select(0, static_cast<int64_t>(i)));
    }
    return out;
  }
  if (!py::isinstance<py::list>(arg) && !py::isinstance<py::tuple>(arg)) {
    // Case 3: single grouped weight with externally-owned storage, e.g.
    // Megatron main_grad viewed as [G, N, K]. GEMM writes in-place and Python
    // should not receive a newly allocated grad tensor from this helper.
    out.packed = arg.cast<at::Tensor>();
    NVTE_CHECK(out.packed.is_cuda(), name, " must be a CUDA tensor.");
    // Do not require out.packed.scalar_type() == dtype. Caller-owned
    // main_grad buffers keep the precision chosen by Megatron's grad-buffer config.
    NVTE_CHECK(out.packed.dim() == 3, name, " must have shape [num_groups, rows, cols].");
    NVTE_CHECK(static_cast<size_t>(out.packed.size(0)) == num_groups, name,
               " first dimension must be ", num_groups, ".");
    NVTE_CHECK(out.packed.size(1) == rows && out.packed.size(2) == cols, name,
               " has an unexpected shape.");
    check_contiguous(out.packed, name);
    out.is_grouped = true;
    return out;
  }
  // Case 4: discrete weights with externally-owned per-expert buffers, e.g.
  // Megatron main_grad list. GEMM writes each tensor in-place and returns no
  // allocated grad list to Python.
  out.tensors = output_tensor_list_from_arg(arg, num_groups, rows, cols, name);
  return out;
}

void grouped_gemm_fwd_dgrad(GroupedWeightArg *weights, bool trans_weight,
                            GroupedTensorWrapper *input, bool trans_input,
                            GroupedTensorWrapper *output, GroupedGemmResources *resources) {
  if (weights->is_grouped) {
    // Single grouped weight case: weights are packed as [G, N, K]. Wrap the
    // packed buffer as a uniform GroupedTensor and use the grouped-tensor GEMM.
    auto grouped_weight = make_uniform_grouped_tensor(weights->packed, input->num_tensors(),
                                                      weights->rows, weights->cols);
    grouped_gemm(&grouped_weight, trans_weight, input, trans_input, output, resources, false);
  } else {
    // Discrete weight case: weights are a list of per-expert tensors. Use the
    // discrete-input grouped GEMM variant.
    std::vector<TensorWrapper> weight_wrappers;
    auto weight_nvte = nvte_tensor_list_from_tensors(weights->discrete, &weight_wrappers);
    NVTE_SCOPED_GIL_RELEASE({
      nvte_grouped_gemm_with_discrete_inputA(
          weight_nvte.data(), weights->discrete.size(), trans_weight, input->data(), trans_input,
          output->data(), output->data(), resources->te_alpha.data(), resources->beta(false),
          resources->te_setup.data(), resources->te_cublas.data(), resources->config_data(),
          at::cuda::getCurrentCUDAStream());
    });
  }
}

std::vector<at::Tensor> grouped_gemm_wgrad(GroupedTensorWrapper *x, GroupedTensorWrapper *dy,
                                           py::handle output, bool compute_wgrad, bool accumulate,
                                           GroupedGemmResources *resources, at::ScalarType dtype,
                                           int64_t rows, int64_t cols, const std::string &name,
                                           bool prefer_grouped_output) {
  auto prepared = wgrad_output_from_arg(output, compute_wgrad, resources->num_groups, dtype,
                                        resources->device, rows, cols, name, prefer_grouped_output);
  NVTE_CHECK(!(prepared.owns_storage && accumulate), name,
             " cannot accumulate into a newly allocated wgrad buffer.");
  std::vector<at::Tensor> returned_wgrads;

  if (prepared.is_grouped) {
    // Cases 1 and 3: single grouped weight layout.
    // Case 1: C++ allocated packed [G, N, K] storage; return [packed].
    // Case 3: caller provided packed storage, e.g. main_grad; write in-place
    // and return nothing because autograd receives dummy wgrad tensors.
    auto grouped_output =
        make_uniform_grouped_tensor(prepared.packed, resources->num_groups, rows, cols);
    grouped_gemm(x, false, dy, true, &grouped_output, resources, accumulate);
    if (prepared.owns_storage) {
      returned_wgrads.emplace_back(prepared.packed);
    }
  } else if (!prepared.tensors.empty()) {
    // Cases 2 and 4: discrete per-expert weight layout.
    // Case 2: C++ allocated packed backing storage and split it into views;
    // return those views in parameter order.
    // Case 4: caller provided per-expert buffers, e.g. main_grad list; write
    // in-place and return nothing because autograd receives dummy wgrads.
    std::vector<TensorWrapper> output_wrappers;
    auto output_nvte = nvte_tensor_list_from_tensors(prepared.tensors, &output_wrappers);
    NVTE_SCOPED_GIL_RELEASE({
      nvte_grouped_gemm_with_discrete_out(
          x->data(), false, dy->data(), true, output_nvte.data(), resources->num_groups,
          output_nvte.data(), resources->num_groups, resources->te_alpha.data(),
          resources->beta(accumulate), resources->te_setup.data(), resources->te_cublas.data(),
          resources->config_data(), at::cuda::getCurrentCUDAStream());
    });
    if (prepared.owns_storage) {
      returned_wgrads = prepared.tensors;
    }
  }
  return returned_wgrads;
}

GroupedTensorWrapper make_grouped_bias(const at::Tensor &bias, size_t num_groups,
                                       at::ScalarType bias_dtype, int64_t out_features) {
  NVTE_CHECK(bias.defined(), "Bias tensor must be defined.");
  auto grouped = GroupedTensorWrapper(
      num_groups, std::vector<size_t>{num_groups, static_cast<size_t>(out_features)});
  grouped.set_rowwise_data(bias.data_ptr(), GetTransformerEngineDType(bias_dtype),
                           tensor_shape_1d(bias));
  return grouped;
}

void add_grouped_bias(GroupedTensorWrapper *output, const at::Tensor &bias, size_t num_groups,
                      at::ScalarType dtype, int64_t out_features,
                      std::optional<at::Tensor> bias_scale = std::nullopt) {
  if (!bias.defined()) {
    return;
  }
  auto grouped_bias = make_grouped_bias(bias, num_groups, dtype, out_features);
  if (bias_scale.has_value()) {
    auto scale = maybe_cast_dtype(*bias_scale, at::kFloat);
    check_contiguous(scale, "bias_scale");
    scale = scale.view({-1});
    auto te_scale = makeTransformerEngineTensor(scale);
    NVTE_SCOPED_GIL_RELEASE({
      nvte_grouped_scaled_bias_add(output->data(), grouped_bias.data(), te_scale.data(),
                                   at::cuda::getCurrentCUDAStream());
    });
  } else {
    NVTE_SCOPED_GIL_RELEASE({
      nvte_grouped_bias_add(output->data(), grouped_bias.data(), at::cuda::getCurrentCUDAStream());
    });
  }
}

bool is_gated_activation(const std::string &activation) {
  return activation == "swiglu" || activation == "clamped_swiglu" || activation == "geglu" ||
         activation == "reglu" || activation == "qgeglu" || activation == "sreglu";
}

at::Tensor maybe_deinterleave_glu(const at::Tensor &input, int64_t glu_interleave_size) {
  if (glu_interleave_size <= 0) {
    return input;
  }
  auto shape = input.sizes().vec();
  const int64_t last_dim = shape.back();
  NVTE_CHECK(last_dim % (2 * glu_interleave_size) == 0,
             "GLU interleaving requires the last dimension to be divisible by 2*interleave.");
  check_contiguous(input, "GLU input");
  // Explicit layout materialization: GLU interleave changes memory order.
  return input.view({-1, last_dim / (2 * glu_interleave_size), 2, glu_interleave_size})
      .transpose(1, 2)
      .contiguous()
      .view(shape);
}

at::Tensor maybe_reinterleave_glu_grad(const at::Tensor &input, int64_t glu_interleave_size) {
  if (glu_interleave_size <= 0) {
    return input;
  }
  auto shape = input.sizes().vec();
  const int64_t last_dim = shape.back();
  check_contiguous(input, "GLU grad input");
  // Explicit layout materialization: reverse GLU interleave changes memory order.
  return input.view({-1, 2, last_dim / (2 * glu_interleave_size), glu_interleave_size})
      .transpose(1, 2)
      .contiguous()
      .view(shape);
}

at::Tensor activation_forward_impl(const at::Tensor &input, const std::string &activation,
                                   double activation_limit, double activation_alpha,
                                   double activation_glu_linear_offset) {
  const int64_t out_features =
      is_gated_activation(activation) ? input.size(-1) / 2 : input.size(-1);
  auto output = at::empty({input.size(0), out_features}, input.options());
  auto te_input = makeTransformerEngineTensor(input);
  auto te_output = makeTransformerEngineTensor(output);
  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    if (activation == "swiglu") {
      nvte_swiglu(te_input.data(), te_output.data(), stream);
    } else if (activation == "glu") {
      nvte_glu(te_input.data(), te_output.data(), stream);
    } else if (activation == "geglu") {
      nvte_geglu(te_input.data(), te_output.data(), stream);
    } else if (activation == "qgeglu") {
      nvte_qgeglu(te_input.data(), te_output.data(), stream);
    } else if (activation == "reglu") {
      nvte_reglu(te_input.data(), te_output.data(), stream);
    } else if (activation == "sreglu") {
      nvte_sreglu(te_input.data(), te_output.data(), stream);
    } else if (activation == "clamped_swiglu") {
      nvte_clamped_swiglu_v2(te_input.data(), te_output.data(),
                             static_cast<float>(activation_limit),
                             static_cast<float>(activation_alpha),
                             static_cast<float>(activation_glu_linear_offset), stream);
    } else if (activation == "srelu") {
      nvte_srelu(te_input.data(), te_output.data(), stream);
    } else if (activation == "gelu") {
      nvte_gelu(te_input.data(), te_output.data(), stream);
    } else if (activation == "qgelu") {
      nvte_qgelu(te_input.data(), te_output.data(), stream);
    } else if (activation == "relu") {
      nvte_relu(te_input.data(), te_output.data(), stream);
    } else if (activation == "silu") {
      nvte_silu(te_input.data(), te_output.data(), stream);
    } else {
      NVTE_ERROR("Unsupported megacpp grouped MLP activation: ", activation);
    }
  });
  return output;
}

at::Tensor activation_backward_impl(const at::Tensor &grad, const at::Tensor &input,
                                    const std::string &activation, double activation_limit,
                                    double activation_alpha, double activation_glu_linear_offset) {
  auto output = at::empty_like(input);
  auto te_grad = makeTransformerEngineTensor(grad);
  auto te_input = makeTransformerEngineTensor(input);
  auto te_output = makeTransformerEngineTensor(output);
  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    if (activation == "swiglu") {
      nvte_dswiglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "glu") {
      nvte_dglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "geglu") {
      nvte_dgeglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "qgeglu") {
      nvte_dqgeglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "reglu") {
      nvte_dreglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "sreglu") {
      nvte_dsreglu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "clamped_swiglu") {
      nvte_clamped_dswiglu_v2(te_grad.data(), te_input.data(), te_output.data(),
                              static_cast<float>(activation_limit),
                              static_cast<float>(activation_alpha),
                              static_cast<float>(activation_glu_linear_offset), stream);
    } else if (activation == "srelu") {
      nvte_dsrelu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "gelu") {
      nvte_dgelu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "qgelu") {
      nvte_dqgelu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "relu") {
      nvte_drelu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else if (activation == "silu") {
      nvte_dsilu(te_grad.data(), te_input.data(), te_output.data(), stream);
    } else {
      NVTE_ERROR("Unsupported megacpp grouped MLP activation backward: ", activation);
    }
  });
  return output;
}

at::Tensor grouped_mlp_activation_forward(
    const at::Tensor &input, const std::optional<at::Tensor> &act_scales,
    const std::string &activation, int64_t glu_interleave_size, double activation_limit,
    double activation_alpha, double activation_glu_linear_offset, at::ScalarType dtype) {
  auto activation_input = maybe_deinterleave_glu(input, glu_interleave_size);
  auto activation_output = activation_forward_impl(activation_input, activation, activation_limit,
                                                   activation_alpha, activation_glu_linear_offset);
  if (!act_scales.has_value()) {
    return activation_output;
  }
  auto act_scales_for_fc2 = maybe_cast_dtype(*act_scales, dtype);
  check_contiguous(act_scales_for_fc2, "act_scales");
  return activation_output * act_scales_for_fc2.view({-1, 1});
}

struct ActivationBackwardResult {
  at::Tensor grad_input;
  at::Tensor grad_act_scales;
};

ActivationBackwardResult grouped_mlp_activation_backward(
    const at::Tensor &grad_output, const at::Tensor &input,
    const std::optional<at::Tensor> &act_scales, const std::string &activation,
    int64_t glu_interleave_size, double activation_limit, double activation_alpha,
    double activation_glu_linear_offset, at::ScalarType dtype, bool act_scales_requires_grad) {
  auto activation_input = maybe_deinterleave_glu(input, glu_interleave_size);

  at::Tensor grad_activation_output = grad_output;
  at::Tensor grad_act_scales;
  if (act_scales.has_value()) {
    if (act_scales_requires_grad) {
      // Scaled activations compute y = activation(x) * act_scales[:, None].
      // Recompute activation(x) for dact_scales to match the Python basic-op
      // path without saving another [tokens, hidden] activation tensor.
      auto activation_output =
          activation_forward_impl(activation_input, activation, activation_limit, activation_alpha,
                                  activation_glu_linear_offset);
      grad_act_scales = (activation_output * grad_output).sum(-1);
    }
    auto act_scales_for_grad = maybe_cast_dtype(*act_scales, dtype);
    check_contiguous(act_scales_for_grad, "act_scales");
    grad_activation_output = grad_output * act_scales_for_grad.view({-1, 1});
  }

  auto grad_activation_input =
      activation_backward_impl(grad_activation_output, activation_input, activation,
                               activation_limit, activation_alpha, activation_glu_linear_offset);
  return {maybe_reinterleave_glu_grad(grad_activation_input, glu_interleave_size), grad_act_scales};
}

}  // namespace

std::vector<at::Tensor> megacpp_grouped_mlp_forward(
    const at::Tensor &input, at::ScalarType act_dtype, const at::Tensor &split_sizes,
    py::handle fc1_weight, py::handle fc1_bias, py::handle fc2_weight, py::handle fc2_bias,
    const std::optional<at::Tensor> &act_scales, const std::string &activation,
    int64_t glu_interleave_size, double activation_limit, double activation_alpha,
    double activation_glu_linear_offset, py::handle gemm_scratch) {
  NVTE_CHECK(input.is_cuda(), "megacpp_grouped_mlp_forward requires CUDA input.");
  at::cuda::CUDAGuard device_guard(input.device());

  // act_dtype is the requested activation/GEMM input dtype. The incoming
  // tensor may have a different dtype, so canonicalize it once at the API
  // boundary and use this tensor for all downstream grouped GEMMs.
  const auto dtype = act_dtype;
  auto x = maybe_cast_dtype(input, dtype);
  check_contiguous(x, "input");

  const auto num_groups = static_cast<size_t>(split_sizes.numel());
  NVTE_CHECK(num_groups > 0, "megacpp grouped MLP requires at least one group.");

  NVTE_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
             "megacpp grouped MLP currently supports BF16/FP16 only.");

  auto fc1_weights = weight_arg_from_py(fc1_weight, num_groups, dtype, "fc1_weight");
  auto fc2_weights = weight_arg_from_py(fc2_weight, num_groups, dtype, "fc2_weight");
  const int64_t in_features = fc1_weights.cols;
  const int64_t fc1_out_features = fc1_weights.rows;
  const int64_t fc2_out_features = fc2_weights.rows;
  const int64_t fc2_in_features = fc2_weights.cols;
  const int64_t activation_out_features =
      is_gated_activation(activation) ? fc1_out_features / 2 : fc1_out_features;
  NVTE_CHECK(activation_out_features == fc2_in_features,
             "FC1 activation output dimension must match FC2 input dimension.");
  auto fc1_bias_tensor =
      packed_bias_from_arg(fc1_bias, num_groups, dtype, fc1_out_features, "fc1_bias");
  auto fc2_bias_tensor =
      packed_bias_from_arg(fc2_bias, num_groups, dtype, fc2_out_features, "fc2_bias");

  NVTE_CHECK(x.numel() % in_features == 0, "input last dimension is incompatible with FC1.");
  const int64_t total_tokens = x.numel() / in_features;
  auto [split_sizes_i64, split_offsets] = splits_to_offsets_multi(
      split_sizes, x.device(),
      std::vector<int64_t>{1, in_features, fc1_out_features, fc2_in_features, fc2_out_features},
      std::vector<bool>{true, true, true, true, true},
      std::vector<at::ScalarType>{at::kLong, at::kLong, at::kLong, at::kLong, at::kLong}, true);
  // splits_to_offsets_multi returns the canonical int64 CUDA split sizes and
  // offsets in the same order as the stride list above. The CuTe path also asks
  // for int32 split_points, but cuBLAS grouped GEMM does not consume them.
  NVTE_CHECK(split_offsets.size() == 5, "Expected five grouped split-offset tensors.");
  auto base_offsets = split_offsets[0];
  auto x_offsets = split_offsets[1];
  auto fc1_offsets = split_offsets[2];
  auto fc2_offsets = split_offsets[3];
  auto output_offsets = split_offsets[4];
  auto gemm_resources = make_grouped_mlp_backend_resources(x.device(), num_groups, gemm_scratch);

  auto fc1_preact = at::empty({total_tokens, fc1_out_features}, x.options());
  auto grouped_x = make_grouped_tensor(x, split_sizes_i64, x_offsets, in_features);
  auto grouped_fc1_preact =
      make_grouped_tensor(fc1_preact, split_sizes_i64, fc1_offsets, fc1_out_features);
  grouped_gemm_fwd_dgrad(&fc1_weights, true, &grouped_x, false, &grouped_fc1_preact,
                         &gemm_resources);
  add_grouped_bias(&grouped_fc1_preact, fc1_bias_tensor, num_groups, dtype, fc1_out_features);

  auto fc2_x = grouped_mlp_activation_forward(
      fc1_preact, act_scales, activation, glu_interleave_size, activation_limit, activation_alpha,
      activation_glu_linear_offset, dtype);

  std::vector<int64_t> out_shape = input.sizes().vec();
  out_shape.back() = fc2_out_features;
  auto output = at::empty(out_shape, x.options());
  auto grouped_fc2_x = make_grouped_tensor(fc2_x, split_sizes_i64, fc2_offsets, fc2_in_features);
  auto grouped_output =
      make_grouped_tensor(output, split_sizes_i64, output_offsets, fc2_out_features);
  grouped_gemm_fwd_dgrad(&fc2_weights, true, &grouped_fc2_x, false, &grouped_output,
                         &gemm_resources);
  add_grouped_bias(&grouped_output, fc2_bias_tensor, num_groups, dtype, fc2_out_features);

  return {output,      x,           split_sizes_i64, base_offsets, x_offsets,
          fc1_offsets, fc2_offsets, output_offsets,  fc1_preact,   fc2_x};
}

py::tuple megacpp_grouped_mlp_backward(
    const at::Tensor &grad_output, at::ScalarType act_dtype, const at::Tensor &split_sizes,
    const at::Tensor &x_offsets, const at::Tensor &fc1_offsets, const at::Tensor &fc2_offsets,
    const at::Tensor &fc2_dy_offsets, const at::Tensor &base_offsets, const at::Tensor &x,
    const at::Tensor &fc1_activation_input, const at::Tensor &fc2_x,
    const std::optional<at::Tensor> &act_scales, py::handle fc1_weight, py::handle fc2_weight,
    py::handle fc1_wgrad_output, bool fc1_compute_wgrad, bool fc1_accumulate_wgrad,
    py::handle fc2_wgrad_output, bool fc2_compute_wgrad, bool fc2_accumulate_wgrad,
    const std::string &activation, int64_t glu_interleave_size, double activation_limit,
    double activation_alpha, double activation_glu_linear_offset, bool act_scales_requires_grad,
    bool input_requires_grad, py::handle gemm_scratch) {
  (void)base_offsets;
  NVTE_CHECK(grad_output.is_cuda(), "megacpp_grouped_mlp_backward requires CUDA grad_output.");
  at::cuda::CUDAGuard device_guard(grad_output.device());

  // act_dtype is the requested grouped-MLP compute dtype. Backward receives
  // autograd's grad_output as-is, so canonicalize it here instead of requiring
  // a Python-side aten::to before entering C++.
  const auto dtype = act_dtype;
  auto dy = maybe_cast_dtype(grad_output, dtype);
  check_contiguous(dy, "grad_output");

  const auto num_groups = num_groups_from_prepared_split_sizes(split_sizes, grad_output.device());
  auto fc1_weights = weight_arg_from_py(fc1_weight, num_groups, dtype, "fc1_weight");
  auto fc2_weights = weight_arg_from_py(fc2_weight, num_groups, dtype, "fc2_weight");

  const int64_t in_features = fc1_weights.cols;
  const int64_t fc1_out_features = fc1_weights.rows;
  const int64_t fc2_out_features = fc2_weights.rows;
  const int64_t fc2_in_features = fc2_weights.cols;

  NVTE_CHECK(dy.numel() % fc2_out_features == 0,
             "grad_output last dimension is incompatible with FC2.");
  const int64_t total_tokens = dy.numel() / fc2_out_features;
  auto gemm_resources =
      make_grouped_mlp_backend_resources(grad_output.device(), num_groups, gemm_scratch);

  auto grouped_dy = make_grouped_tensor(dy, split_sizes, fc2_dy_offsets, fc2_out_features);
  std::vector<at::Tensor> fc2_wgrads;
  if (fc2_compute_wgrad) {
    auto fc2_x_for_wgrad = maybe_cast_dtype(fc2_x, dtype);
    check_contiguous(fc2_x_for_wgrad, "fc2_x");
    auto grouped_fc2_x_for_wgrad =
        make_grouped_tensor(fc2_x_for_wgrad, split_sizes, fc2_offsets, fc2_in_features);
    fc2_wgrads = grouped_gemm_wgrad(&grouped_fc2_x_for_wgrad, &grouped_dy, fc2_wgrad_output,
                                    fc2_compute_wgrad, fc2_accumulate_wgrad, &gemm_resources, dtype,
                                    fc2_out_features, fc2_in_features, "fc2_wgrad_output",
                                    fc2_weights.is_grouped);
  }

  auto fc2_dx = at::empty({total_tokens, fc2_in_features}, dy.options());
  auto grouped_fc2_dx = make_grouped_tensor(fc2_dx, split_sizes, fc2_offsets, fc2_in_features);
  grouped_gemm_fwd_dgrad(&fc2_weights, false, &grouped_dy, false, &grouped_fc2_dx, &gemm_resources);

  auto activation_grads = grouped_mlp_activation_backward(
      fc2_dx, fc1_activation_input, act_scales, activation, glu_interleave_size, activation_limit,
      activation_alpha, activation_glu_linear_offset, dtype, act_scales_requires_grad);
  auto fc1_dy = activation_grads.grad_input;
  auto grad_act_scales = activation_grads.grad_act_scales;
  auto grouped_fc1_dy = make_grouped_tensor(fc1_dy, split_sizes, fc1_offsets, fc1_out_features);

  std::vector<at::Tensor> fc1_wgrads;
  if (fc1_compute_wgrad) {
    auto x_for_wgrad = maybe_cast_dtype(x, dtype);
    check_contiguous(x_for_wgrad, "x");
    auto grouped_x_for_wgrad =
        make_grouped_tensor(x_for_wgrad, split_sizes, x_offsets, in_features);
    fc1_wgrads = grouped_gemm_wgrad(&grouped_x_for_wgrad, &grouped_fc1_dy, fc1_wgrad_output,
                                    fc1_compute_wgrad, fc1_accumulate_wgrad, &gemm_resources, dtype,
                                    fc1_out_features, in_features, "fc1_wgrad_output",
                                    fc1_weights.is_grouped);
  }

  at::Tensor grad_input;
  if (input_requires_grad) {
    std::vector<int64_t> grad_input_shape = grad_output.sizes().vec();
    grad_input_shape.back() = in_features;
    grad_input = at::empty(grad_input_shape, dy.options());
    auto grouped_grad_input = make_grouped_tensor(grad_input, split_sizes, x_offsets, in_features);
    grouped_gemm_fwd_dgrad(&fc1_weights, false, &grouped_fc1_dy, false, &grouped_grad_input,
                           &gemm_resources);
  } else {
    grad_input = at::empty({0}, dy.options());
  }

  auto empty_return = at::empty({0}, dy.options());
  if (!grad_act_scales.defined()) {
    grad_act_scales = empty_return;
  }
  return py::make_tuple(grad_input, fc1_dy, grad_act_scales, fc1_wgrads, fc2_wgrads);
}

}  // namespace transformer_engine::pytorch
