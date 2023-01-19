# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(transformer_engine LANGUAGES CUDA CXX)

list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CMAKE_CUDA_FLAGS "-G")
endif()

add_library(transformer_engine SHARED
                               transformer_engine.cpp
                               transpose/cast_transpose.cu
                               transpose/transpose.cu
                               transpose/cast_transpose_fusion.cu
                               transpose/multi_cast_transpose.cu
                               activation/gelu.cu
                               gemm/cublaslt_gemm.cu
                               layer_norm/ln_api.cpp
                               layer_norm/ln_bwd_semi_cuda_kernel.cu
                               layer_norm/ln_fwd_cuda_kernel.cu
                               rmsnorm/rmsnorm_api.cpp
                               rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
                               rmsnorm/rmsnorm_fwd_cuda_kernel.cu
                               util/cast.cu
                               fused_softmax/scaled_masked_softmax.cu
                               fused_softmax/scaled_upper_triang_masked_softmax.cu)

target_include_directories(transformer_engine PUBLIC "${PROJECT_SOURCE_DIR}/include")

find_package(CUDAToolkit REQUIRED cublas nvToolsExt)

list(APPEND transformer_engine_LINKER_LIBS CUDA::cublas CUDA::cudart CUDA::nvToolsExt)
target_link_libraries(transformer_engine PUBLIC ${transformer_engine_LINKER_LIBS})

target_include_directories(transformer_engine PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                            fused_softmax/scaled_upper_triang_masked_softmax.cu
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
