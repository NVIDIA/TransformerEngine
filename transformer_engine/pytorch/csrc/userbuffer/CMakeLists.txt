# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set(transformer_engine_ubuf_SOURCES)

list(APPEND transformer_engine_ubuf_SOURCES userbuffers.cu
                                            userbuffers-host.cpp)

add_library(transformer_engine_ubuf SHARED ${transformer_engine_ubuf_SOURCES})

target_include_directories(transformer_engine_ubuf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

list(APPEND transformer_engine_ubuf_LINKER_LIBS CUDA::cublas CUDA::cudart CUDA::nvToolsExt gdrapi)

target_link_libraries(transformer_engine_ubuf PUBLIC ${transformer_engine_ubuf_LINKER_LIBS})
target_include_directories(transformer_engine_ubuf PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_source_files_properties(userbuffers.cu
                            userbuffers-host.cpp
                            PROPERTIES
                            INCLUDE_DIRECTORIES ${NVTE_MPI_INCLUDE}
                            COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=64>")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

