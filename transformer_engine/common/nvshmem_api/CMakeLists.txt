##########################################################################
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
##########################################################################
cmake_minimum_required (VERSION 3.18)
project(nvshmemapi LANGUAGES CXX CUDA)

# Configure dependencies
find_package(CUDAToolkit REQUIRED)
# find_package(MPI REQUIRED)
set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}" CACHE STRING "Location of NVSHMEM installation")

add_library(nvshmemapi STATIC nvshmem_waitkernel.cu)
set(NVSHMEMAPI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)
target_link_directories(nvshmemapi PUBLIC ${NVSHMEM_HOME}/lib)
target_link_libraries(nvshmemapi PUBLIC -static-libstdc++ nvshmem_device nvshmem_host CUDA::nvml CUDA::cublas CUDA::cuda_driver)
target_include_directories(nvshmemapi PRIVATE
                           ${NVSHMEM_HOME}/include/)
target_include_directories(nvshmemapi PUBLIC
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           "${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(nvshmemapi PROPERTIES
                      CUDA_STANDARD 17
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON)

# This means nvshmemapi.so will be installed in TransformerEngine/, alongside libtransformer_engine.so and transformer_engine_common.cpython-310-x86_64-linux-gnu.so
# install(TARGETS nvshmemapi DESTINATION ./)