# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(CUDAToolkit REQUIRED cublas)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(common)

option(ENABLE_JAX "Enable JAX in the building workflow." OFF)
if(ENABLE_JAX)
    add_subdirectory(jax)
endif()
