# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Check for PyBind11
find_package(pybind11 CONFIG REQUIRED)

# Configure Transformer Engine library
set(transformer_engine_userbuffers_SOURCES)
list(APPEND transformer_engine_userbuffers_SOURCES
     userbuffers.cu
     userbuffers-host.cpp
     ipcsocket.cc
     executor.cpp
     pybind.cpp)
add_library(transformer_engine_userbuffers SHARED ${transformer_engine_userbuffers_SOURCES})
target_include_directories(transformer_engine_userbuffers PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Check for DLPack headers
set(DLPACK_FRONTEND_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/dlpack/include")
if(NOT EXISTS "${DLPACK_FRONTEND_INCLUDE_DIR}")
    message(FATAL_ERROR
            "Could not find DLPack headers. "
            "Try running 'git submodule update --init --recursive' "
            "within the Transformer Engine source.")
endif()

# Configure dependencies
set(transformer_engine_userbuffers_LIBS)
list(APPEND transformer_engine_userbuffers_LIBS
     CUDA::cublas
     CUDA::cuda_driver
     CUDA::cudart
     CUDA::nvrtc
     CUDA::nvToolsExt
     pybind11::module
     pybind11::embed)
target_link_libraries(transformer_engine_userbuffers PUBLIC ${transformer_engine_userbuffers_LIBS})
target_include_directories(transformer_engine_userbuffers PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(transformer_engine_userbuffers PRIVATE "${DLPACK_FRONTEND_INCLUDE_DIR}")
target_include_directories(transformer_engine_userbuffers PUBLIC "${pybind11_INCLUDE_DIRS}")

# Compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Install library
install(TARGETS transformer_engine_userbuffers DESTINATION .)
