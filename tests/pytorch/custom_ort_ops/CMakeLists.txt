# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)
project(custom_ort_ops LANGUAGES CXX)

# Dependencies
find_package(CUDAToolkit REQUIRED)
set(ONNX_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/onnxruntime/include)
if(NOT EXISTS "${ONNX_INCLUDE_DIR}")
    message(FATAL_ERROR
            "Could not find ONNX Runtime headers. "
            "Please clone https://github.com/microsoft/onnxruntime "
            "into TransformerEngine/tests/pytorch/onnx.")
endif()
include_directories(${ONNX_INCLUDE_DIR})

# Configure library
add_library(custom_ort_ops SHARED custom_op_library.cc)
target_link_libraries(custom_ort_ops PUBLIC CUDA::cudart)
target_include_directories(custom_ort_ops PUBLIC
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(custom_ort_ops PRIVATE
                           ${ONNX_INCLUDE_DIR}/onnxruntime
                           ${ONNX_INCLUDE_DIR}/onnxruntime/core/session)

# Install library
install(TARGETS custom_ort_ops DESTINATION .)
