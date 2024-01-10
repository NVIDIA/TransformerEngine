# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(test_util
               test_nvrtc.cpp
               test_string.cpp
               ../test_common.cu)

target_link_libraries(test_util PUBLIC CUDA::cudart GTest::gtest_main ${TE_LIB})
target_compile_options(test_util PRIVATE -O2)

include(GoogleTest)
gtest_discover_tests(test_util)
