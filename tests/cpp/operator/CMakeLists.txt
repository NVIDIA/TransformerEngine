# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(test_operator
               test_qdq.cu
               test_cast_transpose.cu
               test_transpose.cu
               test_cast_transpose_dbias.cu
               test_cast_transpose_dbias_dgelu.cu
               test_gelu.cu
               test_layernorm.cu
               test_multi_cast_transpose.cu
               ../test_common.cu)

target_link_libraries(test_operator PUBLIC CUDA::cudart GTest::gtest_main ${TE_LIB})
target_compile_options(test_operator PRIVATE -O2)

include(GoogleTest)
gtest_discover_tests(test_operator)
