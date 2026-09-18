# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# Find TE
: ${TE_PATH:=/opt/transformerengine}
TE_LIB_PATH=$(pip3 show transformer-engine | grep -E "Location:|Editable project location:" | tail -n 1 | awk '{print $NF}')
export LD_LIBRARY_PATH=$TE_LIB_PATH:$LD_LIBRARY_PATH

# Set parallelization parameters
NUM_PHYSICAL_CORES=$(nproc)
NUM_PARALLEL_JOBS=4

# Run the CUDA implementation only in L1 as we want to migrate to CuTeDSL implementation so it should have more coverage in L0
export NVTE_ENABLE_CUTEDSL_BACKEND=0

TVM_FFI_LIBRARY=$(python3 -c 'from importlib.metadata import distribution; print(distribution("apache-tvm-ffi").locate_file("tvm_ffi/lib/libtvm_ffi.so"))')

cd $TE_PATH/tests/cpp
cmake -GNinja -Bbuild -DTVM_FFI_LIBRARY="$TVM_FFI_LIBRARY" .
cmake --build build
export OMP_NUM_THREADS=$((NUM_PHYSICAL_CORES / NUM_PARALLEL_JOBS))
ctest --test-dir build -j$NUM_PARALLEL_JOBS --output-junit $XML_LOG_DIR/ctest_cppunittest.xml
