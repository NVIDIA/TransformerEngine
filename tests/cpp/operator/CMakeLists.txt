# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(test_operator
               test_cast.cu
               test_cast_current_scaling.cu
               test_cast_dbias.cu
               test_cast_dbias_dgelu.cu
               test_cast_gated_swiglu.cu
               test_cast_mxfp8_gated_swiglu.cu
               test_qdq.cu
               test_cast_mxfp8.cu
               test_cast_nvfp4_transpose.cu
               test_cast_float8blockwise.cu
               test_dequantize_mxfp8.cu
               test_transpose.cu
               test_cast_transpose.cu
               test_cast_transpose_current_scaling.cu
               test_cast_transpose_dbias.cu
               test_cast_transpose_dbias_dgelu.cu
               test_cast_transpose_dgeglu.cu
               test_act.cu
               test_normalization.cu
               test_normalization_mxfp8.cu
               test_memset.cu
               test_multi_cast_transpose.cu
               test_multi_padding.cu
               test_multi_unpadding.cu
               test_causal_softmax.cu
               test_swizzle.cu
               test_swap_first_dims.cu
               ../test_common.cu)

# Add profiling and debug flags for CUDA compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")                   # Generate line info for device code  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")                          # Add debug symbols for host code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")          # Add info about registers usage
# Note: Using -lineinfo instead of -G to avoid conflicts and get line mapping

# Find required packages
find_package(OpenMP REQUIRED)
list(APPEND test_operator_LINKER_LIBS CUDA::cudart GTest::gtest_main ${TE_LIB} CUDA::nvrtc CUDNN::cudnn)

target_link_libraries(test_operator PUBLIC ${test_operator_LINKER_LIBS} OpenMP::OpenMP_CXX)
target_compile_options(test_operator PRIVATE -O2 -fopenmp)

include(GoogleTest)
gtest_discover_tests(test_operator DISCOVERY_TIMEOUT 600)
