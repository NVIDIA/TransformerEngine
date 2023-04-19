# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
add_library(transformer_engine SHARED
                               transformer_engine.cpp
                               transpose/cast_transpose.cu
                               transpose/transpose.cu
                               transpose/cast_transpose_fusion.cu
                               transpose/transpose_fusion.cu
                               transpose/multi_cast_transpose.cu
                               activation/gelu.cu
			       fused_attn/fused_attn_fp8.cu
			       fused_attn/utils.cu
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

target_include_directories(transformer_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

list(APPEND transformer_engine_LINKER_LIBS CUDA::cublas CUDA::cudart CUDA::nvToolsExt CUDNN::cudnn)
target_link_libraries(transformer_engine PUBLIC ${transformer_engine_LINKER_LIBS})

target_include_directories(transformer_engine PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(transformer_engine PRIVATE "${CMAKE_SOURCE_DIR}/../3rdparty/cudnn-frontend/include")

set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                            fused_softmax/scaled_upper_triang_masked_softmax.cu
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
