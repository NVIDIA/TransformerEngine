# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

pybind11_add_module(
    transformer_engine_jax
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extensions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/modules.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils.cu
)

target_link_libraries(transformer_engine_jax PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt transformer_engine)
install(TARGETS transformer_engine_jax DESTINATION .)
