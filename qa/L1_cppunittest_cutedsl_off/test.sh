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

# If NVTE_ENABLE_CUTEDSL_BACKEND is not set to 1, we assume this also applies to qa/L0_cppunittest,
# in which case we don't need to run this test since L0_cppunittest already covers the CUDA path.
# Otherwise, we override NVTE_ENABLE_CUTEDSL_BACKEND to 0 to force the test to take the CUDA implementation path,
# so we can cover both CUDA and CuteDSL paths on CI.
if [ "${NVTE_ENABLE_CUTEDSL_BACKEND:-0}" != "1" ]; then
    echo "NVTE_ENABLE_CUTEDSL_BACKEND is not 1; L0_cppunittest should have validated the CUDA path already."
    exit 0
fi

# Override this env var to force the test to take the CUDA implementation path
export NVTE_ENABLE_CUTEDSL_BACKEND=0

cd $TE_PATH/tests/cpp
cmake -GNinja -Bbuild .
cmake --build build
export OMP_NUM_THREADS=$((NUM_PHYSICAL_CORES / NUM_PARALLEL_JOBS))
ctest --test-dir build -j$NUM_PARALLEL_JOBS --output-junit $XML_LOG_DIR/ctest_cppunittest.xml
