# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(test_util
               test_nvrtc.cpp
               test_string.cpp
               ../test_common.cu)


find_package(OpenMP REQUIRED)
target_link_libraries(test_util PUBLIC CUDA::cudart GTest::gtest_main ${TE_LIB} CUDA::nvrtc CUDNN::cudnn  OpenMP::OpenMP_CXX)
target_compile_options(test_util PRIVATE -O2 -fopenmp)

include(GoogleTest)
gtest_discover_tests(test_util DISCOVERY_TIMEOUT 600)
