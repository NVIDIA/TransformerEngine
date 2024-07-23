/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <utility>

#include "cudnn_frontend.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudnn_frontend {

void *cudnn_dlhandle = nullptr;

namespace python_bindings {

// Raise C++ exceptions corresponding to C++ FE error codes.
// Pybinds will automatically convert C++ exceptions to python exceptions.
void throw_if(bool const cond, cudnn_frontend::error_code_t const error_code,
              std::string const &error_msg) {
  if (cond == false) return;

  switch (error_code) {
    case cudnn_frontend::error_code_t::OK:
      return;
    case cudnn_frontend::error_code_t::ATTRIBUTE_NOT_SET:
      throw std::invalid_argument(error_msg);
    case cudnn_frontend::error_code_t::SHAPE_DEDUCTION_FAILED:
      throw std::invalid_argument(error_msg);
    case cudnn_frontend::error_code_t::INVALID_TENSOR_NAME:
      throw std::invalid_argument(error_msg);
    case cudnn_frontend::error_code_t::INVALID_VARIANT_PACK:
      throw std::invalid_argument(error_msg);
    case cudnn_frontend::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
      throw cudnn_frontend::cudnnGraphNotSupportedException(error_msg.c_str());
    case cudnn_frontend::error_code_t::GRAPH_EXECUTION_FAILED:
      throw std::runtime_error(error_msg);
    case cudnn_frontend::error_code_t::HEURISTIC_QUERY_FAILED:
      throw cudnn_frontend::cudnnGraphNotSupportedException(error_msg.c_str());
    case cudnn_frontend::error_code_t::CUDNN_BACKEND_API_FAILED:
      throw std::runtime_error(error_msg);
    case cudnn_frontend::error_code_t::CUDA_API_FAILED:
      throw std::runtime_error(error_msg);
    case cudnn_frontend::error_code_t::INVALID_CUDA_DEVICE:
      throw std::runtime_error(error_msg);
    case cudnn_frontend::error_code_t::UNSUPPORTED_GRAPH_FORMAT:
      throw cudnn_frontend::cudnnGraphNotSupportedException(error_msg.c_str());
    case cudnn_frontend::error_code_t::GRAPH_NOT_SUPPORTED:
      throw cudnn_frontend::cudnnGraphNotSupportedException(error_msg.c_str());
    case cudnn_frontend::error_code_t::HANDLE_ERROR:
      throw std::runtime_error(error_msg);
    case cudnn_frontend::error_code_t::INVALID_VALUE:
      throw std::runtime_error(error_msg);
  }
}

void set_dlhandle_cudnn(std::intptr_t dlhandle) {
  cudnn_dlhandle = reinterpret_cast<void *>(dlhandle);
}

PYBIND11_MODULE(_cudnn_compiled_module, m) {
  m.def("backend_version", &detail::get_backend_version);
  m.def("backend_version_string", &detail::get_backend_version_string);
  m.def("_set_dlhandle_cudnn", &set_dlhandle_cudnn);
  py::register_exception<cudnnGraphNotSupportedException>(m, "cudnnGraphNotSupportedError");
}

}  // namespace python_bindings
}  // namespace cudnn_frontend
