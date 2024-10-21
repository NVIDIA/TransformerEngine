/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace transformer_engine::pytorch {

extern PyTypeObject *Float8TensorPythonClass;
extern PyTypeObject *Float8QParamsClass;

void init_extension();

}  // namespace transformer_engine::pytorch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
