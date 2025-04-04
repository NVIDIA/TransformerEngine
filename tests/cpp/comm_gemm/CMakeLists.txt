# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(test_comm_gemm
               test_comm_gemm.cu
               ../test_common.cu)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
target_include_directories(test_comm_gemm PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(test_comm_gemm PUBLIC CUDA::cudart GTest::gtest ${TE_LIB} CUDA::nvrtc CUDNN::cudnn MPI::MPI_CXX OpenMP::OpenMP_CXX)

include(GoogleTest)
gtest_discover_tests(test_comm_gemm DISCOVERY_TIMEOUT 600)
