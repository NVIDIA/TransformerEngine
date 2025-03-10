# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(test_operator
               test_cast_transpose_dbias_dgelu.cu
               ../test_common.cu)

find_package(OpenMP REQUIRED)
list(APPEND test_operator_LINKER_LIBS CUDA::cudart GTest::gtest_main ${TE_LIB} CUDA::nvrtc CUDNN::cudnn)

target_link_libraries(test_operator PUBLIC ${test_operator_LINKER_LIBS} OpenMP::OpenMP_CXX)
target_compile_options(test_operator PRIVATE -O2 -fopenmp)

include(GoogleTest)
gtest_discover_tests(test_operator DISCOVERY_TIMEOUT 600)
