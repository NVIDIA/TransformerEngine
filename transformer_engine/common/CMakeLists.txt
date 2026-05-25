# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MUSA_DIR "/usr/local/musa")
set(MUSA_ARCH "31")

# Transformer Engine library
project(transformer_engine LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${MUSA_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${MUSA_DIR}/lib/cmake/mudnn")

find_package(MUSA REQUIRED)
string(APPEND MUSA_MCC_FLAGS " -std=c++${CMAKE_CXX_STANDARD}")
string(APPEND MUSA_MCC_FLAGS " --offload-arch=mp_${MUSA_ARCH}")

if (MUSA_ARCH STREQUAL "31")
  string(APPEND MUSA_MCC_FLAGS " -mllvm -mtgpu-tempint-prealloc=1")
endif()
set(MUSA_VERBOSE_BUILD ON)
set(MUSA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  string(APPEND MUSA_MCC_FLAGS " -g")
endif()

set(DEPENDENT_TARGETS)

find_package(MUSAToolkit REQUIRED)
list(APPEND DEPENDENT_TARGETS MUSA::toolkit)

include(mudnnTargets)
list(APPEND DEPENDENT_TARGETS mudnn)

find_package(MCCL REQUIRED)
add_library(MUSA::mccl SHARED IMPORTED)
set_target_properties(MUSA::mccl PROPERTIES
  IMPORTED_LOCATION ${MCCL_LIBRARIES}
  INTERFACE_INCLUDE_DIRECTORIES ${MCCL_INCLUDE_DIRS}
)
list(APPEND DEPENDENT_TARGETS MUSA::mccl)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
list(APPEND DEPENDENT_TARGETS Python::Module)

execute_process(
  COMMAND
  ${Python_EXECUTABLE} -c "import os, torch_musa;print(os.path.dirname(torch_musa.__file__))"
  ERROR_QUIET
  OUTPUT_VARIABLE TORCH_MUSA_PYTHONPATH
)
string(REGEX REPLACE "^(.+)\n$" "\\1" TORCH_MUSA_PYTHONPATH ${TORCH_MUSA_PYTHONPATH})

add_library(torch_musa_python SHARED IMPORTED)
set_target_properties(torch_musa_python PROPERTIES
  IMPORTED_LOCATION "${TORCH_MUSA_PYTHONPATH}/lib/libmusa_python.so"
)
set_property(TARGET torch_musa_python APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_MUSA_PYTHONPATH}/.."
)
set_property(TARGET torch_musa_python APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_MUSA_PYTHONPATH}/share/torch_musa_codegen"
)# Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
set_property(TARGET torch_musa_python APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_MUSA_PYTHONPATH}/share/generated_cuda_compatible/include"
)
set_property(TARGET torch_musa_python APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_MUSA_PYTHONPATH}/share/generated_cuda_compatible/include/torch/csrc/api/include"
)
list(APPEND DEPENDENT_TARGETS torch_musa_python)

execute_process(
  COMMAND
  ${Python_EXECUTABLE} -c "import os, torch;print(os.path.dirname(torch.__file__))"
  ERROR_QUIET
  OUTPUT_VARIABLE TORCH_PYTHONPATH
)
string(REGEX REPLACE "^(.+)\n$" "\\1" TORCH_PYTHONPATH ${TORCH_PYTHONPATH})

add_library(torch_python SHARED IMPORTED)
set_target_properties(torch_python PROPERTIES
  IMPORTED_LOCATION "${TORCH_PYTHONPATH}/lib/libtorch_python.so"
)
list(APPEND DEPENDENT_TARGETS torch_python)

# Configure Transformer Engine library
include_directories(${PROJECT_SOURCE_DIR}/..)
set(transformer_engine_SOURCES)
set(transformer_engine_cpp_sources)
set(transformer_engine_cuda_sources)
set(transformer_engine_cuda_arch_specific_sources)

# add_compile_options(-include "${CMAKE_CURRENT_SOURCE_DIR}/musify.h")
add_compile_definitions(CUDA_VERSION=0)
add_compile_definitions(CUDART_VERSION=0)
add_compile_definitions(CUDNN_VERSION=0)
add_compile_definitions(__CUDA_ARCH__=0)
add_compile_definitions(NVTE_USE_MUSA=1)
add_compile_definitions(NVTE_SKIP_MUSA_UNCOMPATIBLE)

list(APPEND transformer_engine_cpp_sources
    #  cudnn_utils.cpp
     transformer_engine.cpp
     fused_attn/fused_attn.cpp
     gemm/config.cpp
     normalization/common.cpp
     normalization/layernorm/ln_api.cpp
     normalization/rmsnorm/rmsnorm_api.cpp
     util/cuda_driver.cpp
    #  util/cuda_nvml.cpp
     util/cuda_runtime.cpp
     util/multi_stream.cpp
     util/rtc.cpp
     comm_gemm_overlap/userbuffers/ipcsocket.cc
     comm_gemm_overlap/userbuffers/userbuffers-host.cpp
     comm_gemm_overlap/comm_gemm_overlap.cpp
  )

list(APPEND transformer_engine_cuda_sources
     common.cu
     multi_tensor/adam.cu
     multi_tensor/l2norm.cu
     multi_tensor/scale.cu
     multi_tensor/sgd.cu
     transpose/cast_transpose.cu
     transpose/transpose.cu
     transpose/cast_transpose_fusion.cu
     transpose/transpose_fusion.cu
     transpose/multi_cast_transpose.cu
    #  transpose/quantize_transpose_vector_blockwise.cu
     transpose/quantize_transpose_vector_blockwise.mu
     transpose/swap_first_dims.cu
    #  dropout/dropout.cu
     fused_attn/flash_attn.cu
     fused_attn/context_parallel.cu
     fused_attn/kv_cache.cu
    #  fused_attn/fused_attn_f16_max512_seqlen.cu
    #  fused_attn/fused_attn_f16_arbitrary_seqlen.cu
    #  fused_attn/fused_attn_fp8.cu
    #  fused_attn/utils.cu
     fused_attn/thd_utils.mu
    #  gemm/cublaslt_gemm.cu
    #  gemm/cublaslt_grouped_gemm.cu
     gemm/mudnn_gemm.cpp
     normalization/layernorm/ln_bwd_semi_cuda_kernel.cu
     normalization/layernorm/ln_fwd_cuda_kernel.cu
     normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
     normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu
     permutation/permutation.cu
     permutation/permutation_mask.mu
     util/padding.cu
     swizzle/swizzle.cu
     swizzle/swizzle_block_scaling.cu
     fused_softmax/scaled_masked_softmax.cu
     fused_softmax/scaled_upper_triang_masked_softmax.cu
     fused_softmax/scaled_aligned_causal_masked_softmax.cu
     fused_rope/fused_rope.cu
     fused_router/fused_moe_aux_loss.cu
     fused_router/fused_score_for_moe_aux_loss.cu
     fused_router/fused_topk_with_score_function.cu
     recipe/current_scaling.cu
     recipe/delayed_scaling.cu
     recipe/fp8_block_scaling.cu
    #  recipe/nvfp4.cu
     comm_gemm_overlap/userbuffers/userbuffers.cu
  )

list(APPEND transformer_engine_cuda_arch_specific_sources
     activation/gelu.cu
     activation/glu.cu
     activation/relu.cu
     activation/swiglu.cu
     cast/cast.cu
    #  gemm/cutlass_grouped_gemm.cu
    #  hadamard_transform/group_hadamard_transform.cu
    #  hadamard_transform/graph_safe_group_hadamard_transform.cu
    #  hadamard_transform/hadamard_transform.cu
    #  hadamard_transform/hadamard_transform_cast_fusion.cu
    #  hadamard_transform/group_hadamard_transform_cast_fusion.cu
    #  hadamard_transform/group_row_cast_col_hadamard_transform_cast_fusion.cu
    #  hadamard_transform/graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu
    #  multi_tensor/compute_scale.cu
    #  recipe/mxfp8_scaling.cu
    #  transpose/quantize_transpose_square_blockwise.cu
      transpose/quantize_transpose_square_blockwise.mu
    #  transpose/quantize_transpose_vector_blockwise_fp4.cu
  )

# Compiling the files with the worst compilation time first to hopefully overlap
# better with the faster-compiling cpp files
list(APPEND transformer_engine_SOURCES ${transformer_engine_cuda_arch_specific_sources}
                                       ${transformer_engine_cuda_sources}
                                       ${transformer_engine_cpp_sources})


set_source_files_properties(${transformer_engine_SOURCES}
  PROPERTIES
  MUSA_SOURCE_PROPERTY_FORMAT OBJ
)

musa_add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(transformer_engine PUBLIC ${DEPENDENT_TARGETS})

# Compiling Userbuffers with native MPI bootstrapping requires linking against MPI
option(NVTE_UB_WITH_MPI "Bootstrap Userbuffers with MPI" OFF)
option(NVTE_ENABLE_NVSHMEM "Compile with NVSHMEM library" OFF)
option(NVTE_WITH_CUBLASMP "Use cuBLASMp for tensor parallel GEMMs" OFF)
option(NVTE_BUILD_ACTIVATION_WITH_FAST_MATH "Compile activation kernels with --use_fast_math option" OFF)

# Helper functions to make header files with C++ strings
function(make_string_header STRING STRING_NAME)
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()
function(make_string_header_from_file file_ STRING_NAME)
    file(READ "${file_}" STRING)
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()

# Header files with C++ strings
# list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 cuda_include_path)
make_string_header("${MUSA_DIR}/include"
                   string_path_cuda_include)
make_string_header_from_file(transpose/rtc/cast_transpose_fusion.cu
                             string_code_transpose_rtc_cast_transpose_fusion_cu)
make_string_header_from_file(transpose/rtc/cast_transpose.cu
                             string_code_transpose_rtc_cast_transpose_cu)
make_string_header_from_file(transpose/rtc/transpose.cu
                             string_code_transpose_rtc_transpose_cu)
make_string_header_from_file(transpose/rtc/swap_first_dims.cu
                             string_code_transpose_rtc_swap_first_dims_cu)
make_string_header_from_file(utils.cuh
                             string_code_utils_cuh)
make_string_header_from_file(util/math.h
                             string_code_util_math_h)
target_include_directories(transformer_engine PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/string_headers")


# Number of parallel build jobs
if($ENV{MAX_JOBS})
  set(BUILD_JOBS_STR $ENV{MAX_JOBS})
elseif($ENV{NVTE_BUILD_MAX_JOBS})
  set(BUILD_JOBS_STR $ENV{NVTE_BUILD_MAX_JOBS})
else()
  set(BUILD_JOBS_STR "max")
endif()
message(STATUS "Parallel build jobs: ${BUILD_JOBS_STR}")

# Number of threads per parallel build job
set(BUILD_THREADS_PER_JOB $ENV{NVTE_BUILD_THREADS_PER_JOB})
if (NOT BUILD_THREADS_PER_JOB)
  set(BUILD_THREADS_PER_JOB 1)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads ${BUILD_THREADS_PER_JOB}")
message(STATUS "Threads per parallel build job: ${BUILD_THREADS_PER_JOB}")

set_target_properties(transformer_engine PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

# Install library
install(TARGETS transformer_engine DESTINATION .)
