/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Python Bindings for Grouped FP8 Current Scaling Quantization
 *
 * This file provides Python bindings for the grouped FP8 quantization kernels.
 * These functions are exposed to Python via pybind11 and can be called from
 * the transformer_engine_torch module.
 */

#include <transformer_engine/grouped_fp8_current_scaling.h>

#include "../extensions.h"
#include "common.h"
#include "pybind.h"

namespace transformer_engine {
namespace pytorch {

/**
 * @brief Python binding for grouped FP8 rowwise quantization
 *
 * This function converts Python GroupedTensor objects to C API types and
 * launches the grouped FP8 quantization kernel.
 *
 * @param input Python handle to input GroupedTensor (high precision)
 * @param output Python handle to output GroupedTensor (FP8)
 * @return Python object (output tensor)
 */
py::object group_fp8_quantize_rowwise(const py::handle &input, py::handle &output) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  // Convert Python GroupedTensor to C++ NVTEGroupedTensor
  const auto &grouped_input_tensor = GroupedTensorFromPyTorchGroupedTensor(input);
  const auto &grouped_output_tensor = GroupedTensorFromPyTorchGroupedTensor(output);

  // Launch kernel (releases GIL for better Python concurrency)
  NVTE_SCOPED_GIL_RELEASE({
    nvte_grouped_fp8_quantize_rowwise(grouped_input_tensor.data(), grouped_output_tensor.data(),
                                      at::cuda::getCurrentCUDAStream());
  });

  return py::reinterpret_borrow<py::object>(output);
}

/**
 * @brief Python binding for grouped FP8 columnwise quantization
 *
 * This function quantizes and transposes multiple tensors simultaneously.
 *
 * @param input Python handle to input GroupedTensor (high precision)
 * @param output Python handle to output GroupedTensor (FP8, transposed)
 * @return Python object (output tensor)
 */
py::object group_fp8_quantize_columnwise(const py::handle &input, py::handle &output) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  const auto &grouped_input_tensor = GroupedTensorFromPyTorchGroupedTensor(input);
  const auto &grouped_output_tensor = GroupedTensorFromPyTorchGroupedTensor(output);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_grouped_fp8_quantize_columnwise(grouped_input_tensor.data(), grouped_output_tensor.data(),
                                         at::cuda::getCurrentCUDAStream());
  });

  return py::reinterpret_borrow<py::object>(output);
}

/**
 * @brief Python binding for grouped FP8 quantization (both layouts)
 *
 * This function produces both rowwise and columnwise outputs.
 *
 * @param input Python handle to input GroupedTensor (high precision)
 * @param output Python handle to output GroupedTensor (FP8, both layouts)
 * @return Python object (output tensor)
 */
py::object group_fp8_quantize_both(const py::handle &input, py::handle &output) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  const auto &grouped_input_tensor = GroupedTensorFromPyTorchGroupedTensor(input);
  const auto &grouped_output_tensor = GroupedTensorFromPyTorchGroupedTensor(output);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_grouped_fp8_quantize_both(grouped_input_tensor.data(), grouped_output_tensor.data(),
                                   at::cuda::getCurrentCUDAStream());
  });

  return py::reinterpret_borrow<py::object>(output);
}

/**
 * @brief Register Python bindings with pybind11
 *
 * This function is called during module initialization to register the
 * grouped FP8 quantization functions with the transformer_engine_torch module.
 *
 * @param m pybind11 module object
 */
void register_grouped_fp8_quantization_bindings(py::module &m) {
  m.def("group_fp8_quantize_rowwise", &group_fp8_quantize_rowwise, py::arg("input"),
        py::arg("output"),
        R"pbdoc(
        Perform grouped FP8 quantization with rowwise layout.

        Quantizes multiple tensors from high precision to FP8 using pre-computed
        scales. Processes all tensors in a single kernel launch for efficiency.

        Args:
            input: Input GroupedTensor (high precision: FP32/BF16/FP16)
            output: Output GroupedTensor (FP8, must have scales pre-computed)

        Returns:
            Output GroupedTensor with quantized data

        Example:
            >>> # After computing scales
            >>> output = tex.group_fp8_quantize_rowwise(input_grouped, output_grouped)

        Note:
            This is part of the three-step FP8 current scaling workflow:
            1. Compute amax (tex.group_amax_graph_safe)
            2. Compute scales (tex.multi_tensor_compute_scale_and_scale_inv)
            3. Quantize (this function)
        )pbdoc");

  m.def("group_fp8_quantize_columnwise", &group_fp8_quantize_columnwise, py::arg("input"),
        py::arg("output"),
        R"pbdoc(
        Perform grouped FP8 quantization with columnwise (transposed) layout.

        Quantizes and transposes multiple tensors simultaneously. Output is in
        columnwise format suitable for TN/NT GEMM layouts.

        Args:
            input: Input GroupedTensor (high precision, rowwise)
            output: Output GroupedTensor (FP8, columnwise)

        Returns:
            Output GroupedTensor with quantized and transposed data

        Example:
            >>> # Quantize and transpose for columnwise GEMM
            >>> output = tex.group_fp8_quantize_columnwise(input_grouped, output_grouped)

        Note:
            All tensors must be 2D for transpose operation.
        )pbdoc");

  m.def("group_fp8_quantize_both", &group_fp8_quantize_both, py::arg("input"), py::arg("output"),
        R"pbdoc(
        Perform grouped FP8 quantization producing both rowwise and columnwise outputs.

        Quantizes multiple tensors and produces both layouts simultaneously.
        Useful when both layouts are needed (e.g., forward and backward passes).

        Args:
            input: Input GroupedTensor (high precision)
            output: Output GroupedTensor (FP8, must have both buffers allocated)

        Returns:
            Output GroupedTensor with both rowwise and columnwise data

        Example:
            >>> # Quantize to both layouts
            >>> output = tex.group_fp8_quantize_both(input_grouped, output_grouped)
        )pbdoc");
}

}  // namespace pytorch
}  // namespace transformer_engine
