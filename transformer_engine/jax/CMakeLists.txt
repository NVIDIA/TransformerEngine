# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

pybind11_add_module(
    transformer_engine_jax
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extensions.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/modules.cc
)

target_link_libraries(transformer_engine_jax PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt transformer_engine)
