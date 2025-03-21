/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "extensions.h"
#include "pybind.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine::pytorch {

void compute_amax(const at::Tensor& tensor, at::Tensor& amax) {
  init_extension();

  auto input_tensor = tensor.contiguous();
  const TensorWrapper& te_input = makeTransformerEngineTensor(input_tensor);

  TORCH_CHECK(amax.scalar_type() == at::kFloat, "amax must be a float tensor");
  TORCH_CHECK(amax.numel() == 1, "amax can only has one element");
  TensorWrapper fake_te_output(
      nullptr, te_input.shape(),
      transformer_engine::DType::kFloat8E4M3,  // It doesn't matter because we only compute amax.
      amax.data_ptr<float>());

  nvte_compute_amax(te_input.data(), fake_te_output.data(), at::cuda::getCurrentCUDAStream());
}

py::object quantize_to_fragment(const at::Tensor& input, py::handle quantizer,
                                const py::object& output, const size_t start_offset_in_output,
                                std::optional<at::Tensor> noop) {
  init_extension();
  auto my_quantizer = convert_quantizer(quantizer);

  NVTE_CHECK(input.is_contiguous(), "Input tensor of quantize_to_fragment must be contiguous");
  const TensorWrapper& te_input = makeTransformerEngineTensor(input);

  NVTE_CHECK(!output.is_none(), "Output tensor of quantize_to_fragment must not be None");
  auto te_output = makeTransformerEngineTensor(output, quantizer);

  size_t input_numel = te_input.numel();
  size_t output_numel = te_output.numel();
  NVTE_CHECK(start_offset_in_output + input_numel <= output_numel,
             "start_offset_in_output + input numel must be less than or equal to output numel "
             "in quantize_to_fragment");

  TensorWrapper te_noop;
  if (noop.has_value()) {
    te_noop = makeTransformerEngineTensor(*noop);
  } else {
    te_noop = TensorWrapper();
  }

  if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    char* dptr = reinterpret_cast<char*>(te_output.dptr());
    char* fragment_dptr = dptr + start_offset_in_output * te_output.element_size();
    // Create a TensorWrapper for the fragment of the te_output.
    // There are three different attributes from te_output:
    //   1. dptr     : The fragment_dptr is offset by start_offset_in_output.
    //   2. shape    : Use the shape of te_input because the fragment should have the same shape as
    //                 te_input.
    //   3. amax_dptr: Use nullptr instead of amax_dptr from te_output, to avoid atomic amax updates
    //                 in kernel.
    // Other attributes are the same as te_output.
    TensorWrapper te_output_fragment(fragment_dptr, te_input.shape(), te_output.dtype(), nullptr,
                                     te_output.scale(), te_output.scale_inv(),
                                     te_output.scale_inv_shape(), te_output.scaling_mode());

    nvte_quantize_noop(te_input.data(), te_output_fragment.data(), te_noop.data(),
                       at::cuda::getCurrentCUDAStream());
  } else {
    // TODO: Add support for NV sub-channel here.
    NVTE_ERROR("Only per-tensor current scaling is supported for quantize_to_fragment now");
  }

  return output;
}

}  // namespace transformer_engine::pytorch
