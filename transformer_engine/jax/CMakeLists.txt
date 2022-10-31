# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

pybind11_add_module(
    transformer_engine_jax
    ${PROJECT_SOURCE_DIR}/jax/csrc/extensions.cc
    ${PROJECT_SOURCE_DIR}/jax/csrc/modules.cc
)

target_link_libraries(transformer_engine_jax PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt transformer_engine)
