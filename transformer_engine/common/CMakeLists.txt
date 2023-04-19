# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Configure C++ library
set(transformer_engine_SOURCES)
list(APPEND transformer_engine_SOURCES
     transformer_engine.cpp
     transpose/cast_transpose.cu
     transpose/transpose.cu
     transpose/cast_transpose_fusion.cu
     transpose/transpose_fusion.cu
     transpose/multi_cast_transpose.cu
     activation/gelu.cu
     gemm/cublaslt_gemm.cu
     layer_norm/ln_api.cpp
     layer_norm/ln_bwd_semi_cuda_kernel.cu
     layer_norm/ln_fwd_cuda_kernel.cu
     rmsnorm/rmsnorm_api.cpp
     rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
     rmsnorm/rmsnorm_fwd_cuda_kernel.cu
     util/cast.cu
     util/cuda_driver.cpp
     util/cuda_runtime.cpp
     util/rtc.cpp
     util/system.cpp
     fused_softmax/scaled_masked_softmax.cu
     fused_softmax/scaled_upper_triang_masked_softmax.cu)
if(NVTE_MPI_FOUND)
    list(APPEND transformer_engine_SOURCES
         comm_gemm_overlap/userbuffers.cu
         comm_gemm_overlap/userbuffers-host.cpp)
endif()
add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Configure dependencies
target_link_libraries(transformer_engine PUBLIC
                      CUDA::cublas
                      CUDA::cuda_driver
                      CUDA::cudart
                      CUDA::nvrtc
                      CUDA::nvToolsExt)
if(NVTE_MPI_FOUND)
    target_link_libraries(transformer_engine PUBLIC gdrapi)
endif()
target_include_directories(transformer_engine PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Convert files to C++ strings
function(convert_file_to_string_header file_ string_name)
    file(READ "${file_}" STRING)
    set(STRING_NAME "${string_name}")
    configure_file(util/string_header.h.in
                   "string_headers/${string_name}.h"
                   @ONLY)
endfunction()
convert_file_to_string_header(utils.cuh
                              code_string_utils_cuh)
convert_file_to_string_header(transpose/rtc/transpose.cu
                              code_string_transpose_rtc_transpose_cu)
target_include_directories(transformer_engine PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/string_headers")

# Compiler options
set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                            fused_softmax/scaled_upper_triang_masked_softmax.cu
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")
if(NVTE_MPI_FOUND)
    set_source_files_properties(comm_gemm_overlap/userbuffers.cu
                                comm_gemm_overlap/userbuffers-host.cpp
                                PROPERTIES
                                INCLUDE_DIRECTORIES ${NVTE_MPI_INCLUDE}
                                COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=64>")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
