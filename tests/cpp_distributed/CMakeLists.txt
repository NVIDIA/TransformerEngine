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

# -- NCCL core ----------------------------------------------------------------
# Anchor on libnccl and derive nccl.h from the same install prefix so the
# header and library can't drift across installs.
find_library(NCCL_LIB
             NAMES nccl libnccl
             HINTS /opt/nvidia/nccl/lib /opt/nvidia/nccl/lib64
                   /usr/local/nccl/lib /usr/local/nccl/lib64
             PATH_SUFFIXES lib lib64
             REQUIRED)
get_filename_component(_nccl_lib_dir "${NCCL_LIB}" DIRECTORY)
set(NCCL_PREFIX "${_nccl_lib_dir}")
while(NCCL_PREFIX AND NOT EXISTS "${NCCL_PREFIX}/include/nccl.h")
  get_filename_component(_nccl_parent "${NCCL_PREFIX}" DIRECTORY)
  if(_nccl_parent STREQUAL NCCL_PREFIX)
    break()
  endif()
  set(NCCL_PREFIX "${_nccl_parent}")
endwhile()
find_path(NCCL_INCLUDE_DIR nccl.h
          HINTS "${NCCL_PREFIX}/include"
          NO_DEFAULT_PATH)
if(NOT NCCL_INCLUDE_DIR)
  message(FATAL_ERROR
    "nccl.h not found under the prefix of ${NCCL_LIB}.")
endif()
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

# -- EP distributed tests ------------------------------------------------------
# Launched via mpirun; ncclUniqueId exchange uses MPI_Bcast (see test_ep_common.h).
# The test binary only uses NCCL core symbols (ncclMemAlloc, ncclCommWindow*);
# all ncclEp* calls live behind TE's public <transformer_engine/ep.h>, which is
# statically linked into libtransformer_engine.so.
message(STATUS "EP test: NCCL headers: ${NCCL_INCLUDE_DIR}")
set(EP_TEST_COMMON_INCLUDES
    ${NCCL_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_PATH}
    ../../transformer_engine/common/include
    ../../transformer_engine/common
    ${CMAKE_CURRENT_SOURCE_DIR})

# nvrtc must follow TE_LIB so symbols referenced from libtransformer_engine.so
# (loaded via dlopen in Python; not in its DT_NEEDED) resolve through nvrtc.
set(EP_TEST_COMMON_LIBS
    CUDA::cuda_driver
    CUDA::cudart
    GTest::gtest
    ${TE_LIB}
    CUDA::nvrtc
    ${NCCL_LIB}
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX)

# -- EP distributed tests (per-op + full pipeline + zero-copy symm) -----------
add_executable(test_ep test_ep.cu ../cpp/test_common.cu)
target_include_directories(test_ep PRIVATE ${EP_TEST_COMMON_INCLUDES})
target_link_libraries(test_ep PUBLIC ${EP_TEST_COMMON_LIBS})

# Do NOT use gtest_discover_tests - these binaries require multi-process
# launch via run_test_ep.sh, not direct single-process execution.
message(STATUS "EP distributed tests enabled (NCCL EP statically linked into libtransformer_engine.so)")
