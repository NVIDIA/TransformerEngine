# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(test_util
               test_nvrtc.cpp
               test_string.cpp
               test_workspace.cpp
               test_workspace_api.cpp
               test_workspace_gpu.cu
               ../test_common.cu)


find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(test_util PUBLIC
                      CUDA::cudart
                      GTest::gtest_main
                      ${TE_LIB}
                      CUDA::nvrtc
                      CUDNN::cudnn
                      OpenMP::OpenMP_CXX
                      Threads::Threads)
target_compile_options(test_util PRIVATE -O2 -fopenmp)

include(GoogleTest)
# MALLOC_PERTURB_ makes glibc overwrite freed memory, so a destructor that reads a destroyed object
# at exit crashes the test process, which ctest reports as a failure, instead of passing silently.
gtest_discover_tests(test_util DISCOVERY_TIMEOUT 600 PROPERTIES ENVIRONMENT "MALLOC_PERTURB_=165")

# Context teardown is isolated from ordinary utility tests and their GPU state.
add_executable(test_cuda_context test_cuda_context.cu)
target_link_libraries(test_cuda_context PRIVATE GTest::gtest_main ${TE_LIB}
                      CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDNN::cudnn Threads::Threads)
gtest_discover_tests(test_cuda_context DISCOVERY_TIMEOUT 600)
