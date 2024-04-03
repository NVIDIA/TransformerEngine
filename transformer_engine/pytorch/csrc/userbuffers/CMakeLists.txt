# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Configure userbuffers library
add_library(transformer_engine_userbuffers SHARED
            userbuffers.cu
            userbuffers-host.cpp)
target_include_directories(transformer_engine_userbuffers PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}")

# Configure dependencies
find_package(MPI REQUIRED)
find_library(GDRCOPY_LIBRARY gdrapi
             HINTS "${GDRCOPY_LIBRARY_DIR}" "$ENV{GDRCOPY_LIBRARY_DIR}")
if(NOT GDRCOPY_LIBRARY)
    message(FATAL_ERROR "Could not find GDRCopy, please set GDRCOPY_LIBRARY_DIR")
endif()
message(STATUS "Found GDRCopy: ${GDRCOPY_LIBRARY}")
target_link_libraries(transformer_engine_userbuffers PUBLIC
                      CUDA::cudart
                      CUDA::cuda_driver
                      MPI::MPI_CXX
                      ${GDRCOPY_LIBRARY})
target_include_directories(transformer_engine_userbuffers PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Compiler options
set_source_files_properties(userbuffers.cu
                            userbuffers-host.cpp
                            PROPERTIES
                            COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=64>")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Install library
install(TARGETS transformer_engine_userbuffers DESTINATION .)
