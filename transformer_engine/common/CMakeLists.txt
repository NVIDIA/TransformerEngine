# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

#######################################################################################
# Core Transformer Engines Library
#######################################################################################

set(transformer_engine_SOURCES)
list(APPEND transformer_engine_SOURCES
     transformer_engine.cpp
     transpose/cast_transpose.cu
     transpose/transpose.cu
     transpose/cast_transpose_fusion.cu
     transpose/transpose_fusion.cu
     transpose/multi_cast_transpose.cu
     activation/gelu.cu
     fused_attn/fused_attn_f16_max512_seqlen.cu
     fused_attn/fused_attn_f16_arbitrary_seqlen.cu
     activation/relu.cu
     activation/swiglu.cu
     fused_attn/fused_attn_fp8.cu
     fused_attn/fused_attn.cpp
     fused_attn/utils.cu
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
     fused_softmax/scaled_upper_triang_masked_softmax.cu
     fused_softmax/scaled_aligned_causal_masked_softmax.cu
     fused_rope/fused_rope.cu
     recipe/delayed_scaling.cu
     userbuffers/ipcsocket.cc
     userbuffers/userbuffers.cu
     userbuffers/userbuffers-host.cpp)
add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
target_include_directories(transformer_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Start configuring dependencies
set(transformer_engine_LIBS)
list(APPEND transformer_engine_LIBS
     CUDA::cublas
     CUDA::cuda_driver
     CUDA::cudart
     CUDA::nvrtc
     CUDA::nvToolsExt
     CUDNN::cudnn)
target_include_directories(transformer_engine PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Check for cuDNN frontend API
set(CUDNN_FRONTEND_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cudnn-frontend/include")
if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}")
    message(FATAL_ERROR
            "Could not find cuDNN frontend API. "
            "Try running 'git submodule update --init --recursive' "
            "within the Transformer Engine source.")
endif()
target_include_directories(transformer_engine PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")

# Check for MPI (optional)
option(UB_MPI_BOOTSTRAP "Bootstrap userbuffers with MPI." OFF)
message(STATUS "Userbuffers MPI bootstrap: ${UB_MPI_BOOTSTRAP}")
if (UB_MPI_BOOTSTRAP)
     find_package(MPI REQUIRED)
     list(APPEND transformer_engine_LIBS MPI::MPI_CXX)
     target_compile_definitions(transformer_engine PRIVATE UB_MPI_BOOTSTRAP)
endif()

# Link configured dependencies
target_link_libraries(transformer_engine PUBLIC ${transformer_engine_LIBS})

# Make header files with C++ strings
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
list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 cuda_include_path)
make_string_header("${cuda_include_path}"
                   string_path_cuda_include)
make_string_header_from_file(transpose/rtc/cast_transpose.cu
                             string_code_transpose_rtc_cast_transpose_cu)
make_string_header_from_file(transpose/rtc/transpose.cu
                             string_code_transpose_rtc_transpose_cu)
make_string_header_from_file(utils.cuh
                             string_code_utils_cuh)
target_include_directories(transformer_engine PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/string_headers")

# Compiler options
set(USERBUFFERS_CUDA_SUPPRESS
    -Xcudafe --diag_suppress=39      # possible division by zero
    -Xcudafe --diag_suppress=177     # variable declared but never referenced
    -Xcudafe --diag_suppress=179     # right operand of "%" is zero
    -Xcudafe --diag_suppress=550     # variable set but never used
    -Xcudafe --diag_suppress=949     # nonstandard default argument on declaration
    -Xcudafe --diag_suppress=1143    # arithmetic on pointer to void or function type
    -Xcudafe --diag_suppress=2464)   # deprecated conversion from a string literal to "char *"
set(USERBUFFERS_CXX_SUPPRESS
    -Wno-pointer-arith
    -Wno-return-local-addr)
set_source_files_properties(userbuffers/userbuffers.cu
                            PROPERTIES
                            COMPILE_OPTIONS "${USERBUFFERS_CUDA_SUPPRESS}")
set_source_files_properties(userbuffers/userbuffers-host.cpp
                            PROPERTIES
                            COMPILE_OPTIONS "${USERBUFFERS_CXX_SUPPRESS}")
set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                            fused_softmax/scaled_upper_triang_masked_softmax.cu
                            fused_softmax/scaled_aligned_causal_masked_softmax.cu
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Install core library
install(TARGETS transformer_engine DESTINATION .)

#######################################################################################
# PyBind11 Extensions
#######################################################################################
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(transformer_engine_common_cpp
                    gemm/cublaslt_gemm.cu
                    userbuffers/ipcsocket.cc
                    userbuffers/userbuffers.cu
                    userbuffers/userbuffers-host.cpp
                    transformer_engine.cpp
                    pybind.cpp)
target_include_directories(transformer_engine_common_cpp PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(transformer_engine_common_cpp PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(transformer_engine_common_cpp_LIBS)
list(APPEND transformer_engine_common_cpp_LIBS
     CUDA::cublas
     CUDA::cuda_driver
     CUDA::cudart
     CUDA::nvrtc
     CUDA::nvToolsExt
     transformer_engine)
target_link_libraries(transformer_engine_common_cpp PUBLIC ${transformer_engine_common_cpp_LIBS})

# Check for DLPack headers
set(DLPACK_FRONTEND_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/dlpack/include")
if(NOT EXISTS "${DLPACK_FRONTEND_INCLUDE_DIR}")
    message(FATAL_ERROR
            "Could not find DLPack headers. "
            "Try running 'git submodule update --init --recursive' "
            "within the Transformer Engine source.")
endif()
target_include_directories(transformer_engine_common_cpp PRIVATE ${DLPACK_FRONTEND_INCLUDE_DIR})

# Install
set_source_files_properties(pybind.cpp
                            PROPERTIES
                            COMPILE_OPTIONS "${USERBUFFERS_CXX_SUPPRESS}")
install(TARGETS transformer_engine_common_cpp DESTINATION .)
