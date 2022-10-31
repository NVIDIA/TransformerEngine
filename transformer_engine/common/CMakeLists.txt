# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CMAKE_CUDA_FLAGS "-G")
endif()

add_library(transformer_engine SHARED
                               transformer_engine.cpp
                               transpose/cast_transpose.cu
                               transpose/transpose.cu
                               transpose/cast_transpose_fusion.cu
                               activation/gelu.cu
                               gemm/cublaslt_gemm.cu
                               layer_norm/ln_api.cpp
                               layer_norm/ln_bwd_semi_cuda_kernel.cu
                               layer_norm/ln_fwd_cuda_kernel.cu
                               util/cast.cu)

target_include_directories(transformer_engine PUBLIC "${PROJECT_SOURCE_DIR}/common/include")

list(APPEND transformer_engine_LINKER_LIBS CUDA::cublas CUDA::cudart)
target_link_libraries(transformer_engine PUBLIC ${transformer_engine_LINKER_LIBS})

target_include_directories(transformer_engine PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
