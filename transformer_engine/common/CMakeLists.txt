# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Find userbuffer dependencies
option(NVTE_WITH_USERBUFFERS
       "Support for communication with userbuffers (highly experimental)"
       OFF)
if(NVTE_WITH_USERBUFFERS)
    find_package(MPI REQUIRED)
    find_library(GDRCOPY_LIBRARY gdrapi)
    if(NOT GDRCOPY_LIBRARY)
        message(FATAL_ERROR
                "Attempted to build with support for userbuffers, but could not find GDRCopy")
    endif()
    message(STATUS "GDRCopy: ${GDRCOPY_LIBRARY}")
endif()
message(STATUS "NVTE_WITH_USERBUFFERS: ${NVTE_WITH_USERBUFFERS}")

# Configure Transformer Engine library
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
     fused_softmax/scaled_masked_softmax.cu
     fused_softmax/scaled_upper_triang_masked_softmax.cu)
if(NVTE_WITH_USERBUFFERS)
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
                      CUDA::cudart
                      CUDA::nvToolsExt)
target_include_directories(transformer_engine PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(NVTE_WITH_USERBUFFERS)
    target_link_libraries(transformer_engine PUBLIC
                          MPI::MPI_CXX
                          ${GDRCOPY_LIBRARY})
endif()

# Compiler options
set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                            fused_softmax/scaled_upper_triang_masked_softmax.cu
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")
if(NVTE_WITH_USERBUFFERS)
    set_source_files_properties(comm_gemm_overlap/userbuffers.cu
                                comm_gemm_overlap/userbuffers-host.cpp
                                PROPERTIES
                                COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=64>")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
