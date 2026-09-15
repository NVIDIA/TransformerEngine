# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(test_operator
               test_operator_environment.cpp
               test_cast.cu
               test_cast_current_scaling.cu
               test_cast_current_scaling_grouped.cu
               test_cast_dbias.cu
               test_cast_dbias_dgelu.cu
               test_cast_gated_swiglu.cu
               test_cast_mxfp8_gated_swiglu.cu
               test_qdq.cu
               test_cast_mxfp8.cu
               test_cast_mxfp8_grouped.cu
               test_cast_mxfp8_grouped_scaled_swiglu.cu
               test_cast_nvfp4_transpose.cu
               test_cast_float8blockwise.cu
               test_cast_float8blockwise_grouped.cu
               test_dequantize_mxfp8.cu
               test_dequantize_mxfp8_grouped.cu
               test_fused_group_requantize_mxfp8.cu
               test_dequantize_float8blockwise_grouped.cu
               test_dequantize_nvfp4.cu
               test_transpose.cu
               test_cast_transpose.cu
               test_cast_transpose_current_scaling.cu
               test_cast_transpose_dbias.cu
               test_cast_transpose_dbias_dgelu.cu
               test_cast_transpose_dgeglu.cu
               test_act.cu
               test_scaled_activation.cu
               test_normalization.cu
               test_normalization_mxfp8.cu
               test_memset.cu
               test_splits_to_offsets.cu
               test_multi_cast_transpose.cu
               test_multi_padding.cu
               test_multi_unpadding.cu
               test_softmax.cu
               test_causal_softmax.cu
               test_swizzle.cu
               test_multi_swizzle.cu
               test_swap_first_dims.cu
               test_grouped_gemm.cu
               ../test_common.cu)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Embed REQUIRED)
list(APPEND test_operator_LINKER_LIBS CUDA::cudart GTest::gtest_main ${TE_LIB} CUDA::nvrtc CUDNN::cudnn Python::Python)

if(NOT TVM_FFI_INCLUDE_DIR)
  execute_process(
      COMMAND ${Python_EXECUTABLE} -c
              "from importlib.metadata import distribution; print(distribution('apache-tvm-ffi').locate_file('tvm_ffi/include'))"
      OUTPUT_VARIABLE TVM_FFI_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
endif()
target_include_directories(test_operator PRIVATE ${TVM_FFI_INCLUDE_DIR})

set_source_files_properties(
    test_operator_environment.cpp PROPERTIES
    COMPILE_DEFINITIONS NVTE_SOURCE_DIR="${CMAKE_SOURCE_DIR}/../..")
set_source_files_properties(
    test_cast_mxfp8.cu PROPERTIES
    COMPILE_OPTIONS "--expt-relaxed-constexpr")

target_link_libraries(test_operator PUBLIC ${test_operator_LINKER_LIBS} OpenMP::OpenMP_CXX)
target_compile_options(test_operator PRIVATE -O2 -fopenmp)

# Keep all operator tests in one process so CUDA initialization and, when enabled,
# the embedded Python interpreter and CuTeDSL kernel cache are initialized only once.
# TODO(kainingz): This will only use 1 CPU and 1 GPU. Make it able to utilize all CPUs and GPUs later
add_test(NAME test_operator COMMAND test_operator)
