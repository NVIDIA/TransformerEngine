# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 89 90 100 120)
  else ()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 89 90)
  endif()
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(transformer_engine_tests LANGUAGES CUDA CXX)

add_subdirectory(../../3rdparty/googletest ${PROJECT_BINARY_DIR}/googletest)

enable_testing()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

if(NOT DEFINED TE_LIB_PATH)
    execute_process(COMMAND bash -c "python3 -c 'import transformer_engine as te; print(te.__file__)'"
                    OUTPUT_VARIABLE TE_LIB_FILE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    get_filename_component(TE_LIB_PATH ${TE_LIB_FILE} DIRECTORY)
endif()

find_library(TE_LIB NAMES transformer_engine PATHS "${TE_LIB_PATH}/.." ${TE_LIB_PATH} ENV TE_LIB_PATH REQUIRED)

message(STATUS "Found transformer_engine library: ${TE_LIB}")
include_directories(../../transformer_engine/common/include)
include_directories(../../transformer_engine/common)
include_directories(../../transformer_engine)
include_directories(${CMAKE_SOURCE_DIR})

find_package(CUDAToolkit REQUIRED)
include(${CMAKE_SOURCE_DIR}/../../transformer_engine/common/cmake/cuDNN.cmake)

# Must match how libtransformer_engine.so was built, hence the env var default.
option(NVTE_WITH_CUTEDSL
       "Link the tests against libpython, required when libtransformer_engine.so was built with NVTE_WITH_CUTEDSL=ON"
       "$ENV{NVTE_WITH_CUTEDSL}")

add_subdirectory(operator)
add_subdirectory(util)

if(NVTE_WITH_CUTEDSL)
  # Executables that link libtransformerengine.so must also link to the Python::Python to provide
  # the definitions of these CPython symbols at runtime via dynamic linking.
  # Find the interpreter too so older CMake versions use its version to locate libpython.
  find_package(Python COMPONENTS Interpreter Development.Embed REQUIRED)
  foreach(test_target test_operator test_util)
    target_link_libraries(${test_target} PRIVATE
                          "-Wl,--no-as-needed"
                          Python::Python
                          "-Wl,--as-needed")
  endforeach()
endif()
