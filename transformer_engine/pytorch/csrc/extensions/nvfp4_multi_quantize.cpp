/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <vector>

#include "../extensions.h"
#include "common.h"
#include "common/util/system.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

namespace {

std::vector<size_t> get_tensor_shape(const TensorWrapper &tensor) {
  const auto &shape = tensor.shape();
  return std::vector<size_t>(shape.data, shape.data + shape.ndim);
}

}  // namespace

std::vector<py::object> nvfp4_quantize_4over6_multi(const std::vector<at::Tensor> &tensors,
                                                    py::handle quantizer,
                                                    const py::object &outputs_py) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  NVTE_CHECK(!tensors.empty(), "nvfp4_quantize_4over6_multi expects a non-empty tensor list.");
  auto quantizer_cpp = convert_quantizer(quantizer);
  NVTE_CHECK(IsNVFP4Quantizers(quantizer.ptr()),
             "nvfp4_quantize_4over6_multi only supports NVFP4 quantizers.");
  NVFP4Quantizer *nvfp4_quantizer_cpp = static_cast<NVFP4Quantizer *>(quantizer_cpp.get());
  NVTE_CHECK(nvfp4_quantizer_cpp->nvfp4_4over6_mode != kNVTENVFP44Over6Disabled,
             "nvfp4_quantize_4over6_multi requires a non-disabled 4over6 mode.");
  NVTE_CHECK(!nvfp4_quantizer_cpp->with_rht && !nvfp4_quantizer_cpp->stochastic_rounding &&
                 !nvfp4_quantizer_cpp->with_2d_quantization,
             "Batched 4over6 requires non-RHT, non-stochastic-rounding, 1D quantization.");
  NVTE_CHECK(nvfp4_quantizer_cpp->rowwise_usage && !nvfp4_quantizer_cpp->columnwise_usage,
             "Batched 4over6 supports rowwise-only quantization.");
  NVTE_CHECK(!nvfp4_quantizer_cpp->row_scaled_nvfp4,
             "Batched 4over6 targets per-tensor-scaled (weight) tensors; row-scaled activations "
             "should use whole-buffer quantization plus row slices.");

  const size_t num_tensors = tensors.size();
  const bool has_outputs = !outputs_py.is_none();
  py::list outs;
  if (has_outputs) {
    outs = py::cast<py::list>(outputs_py);
    NVTE_CHECK(py::len(outs) == num_tensors,
               "nvfp4_quantize_4over6_multi: outputs must match tensors in length.");
  }
  auto stream = at::cuda::getCurrentCUDAStream();

  std::vector<at::Tensor> keepalive;
  std::vector<TensorWrapper> inputs;
  std::vector<TensorWrapper> outputs;
  std::vector<py::object> out_py_list;
  keepalive.reserve(num_tensors);
  inputs.reserve(num_tensors);
  outputs.reserve(num_tensors);
  out_py_list.reserve(num_tensors);

  size_t rows = 0, cols = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    keepalive.push_back(tensors[i].contiguous());
    inputs.push_back(makeTransformerEngineTensor(keepalive.back()));
    const bool use_given_output = has_outputs && !outs[i].is_none();
    if (use_given_output) {
      // Quantize into the caller-provided workspace (weight cache path).
      auto [out_cpp, out_py] = quantizer_cpp->convert_and_update_tensor(outs[i]);
      outputs.push_back(std::move(out_cpp));
      out_py_list.push_back(std::move(out_py));
    } else {
      const auto shape = get_tensor_shape(inputs.back());
      auto [out_cpp, out_py] = quantizer_cpp->create_tensor(shape, inputs.back().dtype());
      outputs.push_back(std::move(out_cpp));
      out_py_list.push_back(std::move(out_py));
    }
    const size_t this_cols = keepalive.back().size(-1);
    const size_t this_rows = keepalive.back().numel() / this_cols;
    if (i == 0) {
      rows = this_rows;
      cols = this_cols;
    } else {
      NVTE_CHECK(this_rows == rows && this_cols == cols,
                 "nvfp4_quantize_4over6_multi requires same-shaped tensors, but tensor ", i,
                 " has shape (", this_rows, ", ", this_cols, ") vs (", rows, ", ", cols, ").");
      NVTE_CHECK(inputs.back().dtype() == inputs[0].dtype(),
                 "nvfp4_quantize_4over6_multi requires the same dtype for all tensors.");
    }
  }
  NVTE_CHECK(cols % 16 == 0, "nvfp4_quantize_4over6_multi requires columns divisible by 16.");

  std::vector<NVTETensor> in_nvte(num_tensors), out_nvte(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    in_nvte[i] = inputs[i].data();
    out_nvte[i] = outputs[i].data();
  }

  QuantizationConfigWrapper quant_config;
  quant_config.set_nvfp4_4over6_mode(nvfp4_quantizer_cpp->nvfp4_4over6_mode);
  const auto err_use_fast_math =
      transformer_engine::getenv<bool>("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH");
  if (err_use_fast_math) {
    quant_config.set_nvfp4_4over6_err_use_fast_math(true);
  }

  NVTE_SCOPED_GIL_RELEASE({
    nvte_nvfp4_quantize_4over6_multi(in_nvte.data(), out_nvte.data(), quant_config, num_tensors,
                                     stream);
  });
  return out_py_list;
}

}  // namespace transformer_engine::pytorch
