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

    NVTE_CHECK(input.is_contiguous(),
               "Input tensor of quantize_to_fragment must be contiguous");
    const TensorWrapper& te_input = makeTransformerEngineTensor(input);

    NVTE_CHECK(!output.is_none(), "Output tensor of quantize_to_fragment must not be None");
    auto te_output = makeTransformerEngineTensor(output, quantizer);

    TensorWrapper te_noop;
    if (noop.has_value()) {
        te_noop = makeTransformerEngineTensor(*noop);
    } else {
        te_noop = TensorWrapper();
    }

    if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
        auto my_quantizer_cs = static_cast<Float8CurrentScalingQuantizer*>(my_quantizer.get());
        QuantizationConfigWrapper quant_config;
        quant_config.set_force_pow_2_scales(my_quantizer_cs->force_pow_2_scales);
        quant_config.set_amax_epsilon(my_quantizer_cs->amax_epsilon);
        nvte_cs_cast_to_fragment(te_input.data(), te_output.data(), start_offset_in_output,
                                 te_noop.data(), quant_config, at::cuda::getCurrentCUDAStream());
    } else {
        // TODO: Add support for NV sub-channel here.
        NVTE_ERROR("Only per-tensor current scaling is supported for quantize_to_fragment now");
    }

    return output;
}

}  // namespace transformer_engine::pytorch
