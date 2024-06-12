# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(transformer_engine_tests LANGUAGES CUDA CXX)

add_subdirectory(../../3rdparty/googletest ${PROJECT_BINARY_DIR}/googletest)

enable_testing()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

if(NOT DEFINED TE_LIB_PATH)
    execute_process(COMMAND bash -c "pip show transformer-engine | grep Location | cut -d ' ' -f 2 | tr -d '\n'"
                    OUTPUT_VARIABLE TE_LIB_PATH)
endif()

find_library(TE_LIB NAMES transformer_engine PATHS "${TE_LIB_PATH}/transformer_engine" ${TE_LIB_PATH} ENV TE_LIB_PATH REQUIRED)

message(STATUS "Found transformer_engine library: ${TE_LIB}")
include_directories(../../transformer_engine/common/include)
include_directories(../../transformer_engine/common)
include_directories(${CMAKE_SOURCE_DIR})

find_package(CUDAToolkit REQUIRED)

add_subdirectory(operator)
add_subdirectory(util)
