/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Header file for grouped FP8 quantization Python bindings
 * 
 * This header declares the function that registers grouped FP8 quantization
 * bindings with pybind11. Include this in pybind.cpp and call the registration
 * function during module initialization.
 */

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_PYBIND_GROUPED_FP8_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_PYBIND_GROUPED_FP8_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace transformer_engine {
namespace pytorch {

/**
 * @brief Register grouped FP8 quantization bindings with pybind11 module
 * 
 * This function should be called during PYBIND11_MODULE initialization to
 * expose the grouped FP8 quantization functions to Python.
 * 
 * Exposed functions:
 * - group_fp8_quantize_rowwise()
 * - group_fp8_quantize_columnwise()
 * - group_fp8_quantize_both()
 * 
 * @param m pybind11 module object
 */
void register_grouped_fp8_quantization_bindings(py::module &m);

} // namespace pytorch
} // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_PYBIND_GROUPED_FP8_H_
