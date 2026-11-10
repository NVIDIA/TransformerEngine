##########################################################################
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# NVSHMEM Python packages only provide the versioned host library.
find_library(NVSHMEM_HOST_LIB
             NAMES nvshmem_host libnvshmem_host.so.3
             PATHS ${NVSHMEM_HOME}
             PATH_SUFFIXES lib lib64
             NO_DEFAULT_PATH
             REQUIRED)
target_link_directories(nvshmemapi PUBLIC ${NVSHMEM_HOME}/lib)
target_link_libraries(nvshmemapi PUBLIC -static-libstdc++ nvshmem_device ${NVSHMEM_HOST_LIB} CUDA::nvml CUDA::cublas CUDA::cuda_driver)
target_include_directories(nvshmemapi PRIVATE
                           ${NVSHMEM_HOME}/include/
                           ${CUDNN_INCLUDE_DIR}
                           ${NCCL_INCLUDE_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_include_directories(nvshmemapi PUBLIC
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           "${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(nvshmemapi PROPERTIES
                      CUDA_STANDARD 17
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON)
