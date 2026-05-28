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

add_executable(test_comm_gemm
               test_comm_gemm.cu
               ../cpp/test_common.cu)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# ── NCCL library ──────────────────────────────────────────────────────────────
# Search order: NCCL_HOME env → 3rdparty/nccl submodule build → system paths.
set(NCCL_SUBMODULE_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/nccl/build")
find_library(NCCL_LIB
             NAMES nccl libnccl
             HINTS $ENV{NCCL_HOME}/lib ${NCCL_SUBMODULE_BUILD}/lib
             PATH_SUFFIXES lib lib64
             REQUIRED)

# NCCL headers: prefer submodule build output (has the handle_init API),
# then submodule src, then system (CUDA toolkit).
set(NCCL_SUBMODULE_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/nccl/build/include")
set(NCCL_SUBMODULE_SRC_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/nccl/src/include")
if(EXISTS "${NCCL_SUBMODULE_INCLUDE}/nccl.h")
  set(NCCL_INCLUDE_DIR "${NCCL_SUBMODULE_INCLUDE}")
elseif(EXISTS "${NCCL_SUBMODULE_SRC_INCLUDE}/nccl.h")
  set(NCCL_INCLUDE_DIR "${NCCL_SUBMODULE_SRC_INCLUDE}")
elseif(DEFINED ENV{NCCL_HOME})
  set(NCCL_INCLUDE_DIR "$ENV{NCCL_HOME}/include")
endif()
target_include_directories(test_comm_gemm PRIVATE ${MPI_CXX_INCLUDE_PATH} $ENV{CUBLASMP_HOME}/include)
target_link_libraries(test_comm_gemm PUBLIC CUDA::cuda_driver CUDA::cudart GTest::gtest ${TE_LIB} CUDA::nvrtc MPI::MPI_CXX ${NCCL_LIB} OpenMP::OpenMP_CXX)

include(GoogleTest)
gtest_discover_tests(test_comm_gemm DISCOVERY_TIMEOUT 600)

# ── EP distributed tests ──────────────────────────────────────────────────────
# Launched via mpirun; ncclUniqueId exchange uses MPI_Bcast (see test_ep_common.h).
# Headers + libs come from the in-tree 3rdparty/nccl submodule build.
set(NCCL_EP_SUBMODULE_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/nccl")
find_library(NCCL_EP_LIB
             NAMES nccl_ep libnccl_ep
             HINTS ${NCCL_EP_SUBMODULE_ROOT}/build/lib
             NO_DEFAULT_PATH
             REQUIRED)

set(NCCL_EP_INCLUDE_DIR "${NCCL_EP_SUBMODULE_ROOT}/contrib/nccl_ep/include")
if(NOT EXISTS "${NCCL_EP_INCLUDE_DIR}/nccl_ep.h")
  message(FATAL_ERROR
    "NCCL EP header not found at ${NCCL_EP_INCLUDE_DIR}/nccl_ep.h. "
    "Run `git submodule update --init --recursive` to checkout 3rdparty/nccl.")
endif()
message(STATUS "EP test: NCCL EP headers: ${NCCL_EP_INCLUDE_DIR}")

# Collect NCCL include dirs shared by all EP test targets (nccl_ep.h + nccl.h).
set(EP_TEST_NCCL_INCLUDES ${NCCL_EP_INCLUDE_DIR})
if(DEFINED NCCL_INCLUDE_DIR)
  list(APPEND EP_TEST_NCCL_INCLUDES ${NCCL_INCLUDE_DIR})
  message(STATUS "EP test: NCCL headers: ${NCCL_INCLUDE_DIR}")
endif()

set(EP_TEST_COMMON_INCLUDES
    ${EP_TEST_NCCL_INCLUDES}
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
    ${NCCL_EP_LIB}
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX)

# ── EP distributed tests (per-op + full pipeline + zero-copy symm) ───────────
add_executable(test_ep test_ep.cu ../cpp/test_common.cu)
target_include_directories(test_ep PRIVATE ${EP_TEST_COMMON_INCLUDES})
target_link_libraries(test_ep PUBLIC ${EP_TEST_COMMON_LIBS})

# Do NOT use gtest_discover_tests — these binaries require multi-process
# launch via run_test_ep.sh, not direct single-process execution.
message(STATUS "EP distributed tests enabled: ${NCCL_EP_LIB}")
