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

project(transformer_engine_distributed_tests LANGUAGES CUDA CXX)

add_subdirectory(../../3rdparty/googletest ${PROJECT_BINARY_DIR}/googletest)

enable_testing()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

if(NOT DEFINED TE_LIB_PATH)
    execute_process(COMMAND bash -c "python3 -c 'import transformer_engine as te; print(te.__file__)'"
                    OUTPUT_VARIABLE TE_LIB_FILE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    get_filename_component(TE_LIB_PATH ${TE_LIB_FILE} DIRECTORY)
endif()

find_library(TE_LIB
             NAMES transformer_engine
             PATHS "${TE_LIB_PATH}/.." ${TE_LIB_PATH}
             ENV TE_LIB_PATH
             REQUIRED)
message(STATUS "Found transformer_engine library: ${TE_LIB}")

add_executable(test_comm_gemm
               test_comm_gemm.cu
               ../cpp/test_common.cu)

list(APPEND test_comm_gemm_INCLUDES
     ${CMAKE_SOURCE_DIR}/../../transformer_engine/common/include
     ${CMAKE_SOURCE_DIR}/../../transformer_engine/common
     ${CMAKE_SOURCE_DIR}/../../transformer_engine
     ${CMAKE_SOURCE_DIR}
     ${MPI_CXX_INCLUDE_PATH}
     $ENV{CUBLASMP_HOME}/include)
target_include_directories(test_comm_gemm PRIVATE ${test_comm_gemm_INCLUDES})

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_library(NCCL_LIB
             NAMES nccl libnccl
             PATH_SUFFIXES lib
             REQUIRED)
list(APPEND test_comm_gemm_LINKER_LIBS
     CUDA::cuda_driver
     CUDA::cudart
     GTest::gtest_main
     ${TE_LIB}
     CUDA::nvrtc
     ${NCCL_LIB}
     OpenMP::OpenMP_CXX
     MPI::MPI_CXX)
target_link_libraries(test_comm_gemm PUBLIC ${test_comm_gemm_LINKER_LIBS})

target_compile_options(test_comm_gemm PRIVATE -O2 -fopenmp)

include(GoogleTest)
gtest_discover_tests(test_comm_gemm DISCOVERY_TIMEOUT 600)
