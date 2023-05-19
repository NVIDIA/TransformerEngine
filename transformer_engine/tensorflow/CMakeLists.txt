# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

pybind11_add_module(
    transformer_engine_tensorflow
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extensions.cu
)

add_library(
    _get_stream SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/get_stream_op.cpp
)

# Includes
execute_process(COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
                OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy as np; print(np.get_include())"
                OUTPUT_VARIABLE Numpy_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)

target_include_directories(transformer_engine_tensorflow PRIVATE
    ${Tensorflow_INCLUDE_DIRS}
    ${Tensorflow_INCLUDE_DIRS}/external/farmhash_archive/src
    ${Numpy_INCLUDE_DIRS})
target_include_directories(_get_stream PRIVATE ${Tensorflow_INCLUDE_DIRS})

# Libraries
execute_process(COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(tf.__file__)"
                OUTPUT_VARIABLE Tensorflow_LIB_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(Tensorflow_LIB_PATH ${Tensorflow_LIB_PATH} DIRECTORY)
list(APPEND TF_LINKER_LIBS "${Tensorflow_LIB_PATH}/libtensorflow_framework.so.2")
list(APPEND TF_LINKER_LIBS "${Tensorflow_LIB_PATH}/python/_pywrap_tensorflow_internal.so")

target_link_libraries(
    transformer_engine_tensorflow PRIVATE
    ${TF_LINKER_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt transformer_engine
)

target_link_libraries(_get_stream PRIVATE ${TF_LINKER_LIBS})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Install library
install(TARGETS transformer_engine_tensorflow DESTINATION .)
install(TARGETS _get_stream DESTINATION .)
