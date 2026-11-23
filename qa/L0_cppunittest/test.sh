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

# If CuTeDSL backend is not enabled, this will be ignored
export NVTE_REPORT_IF_CUTEDSL_BACKEND_NOT_CHOSEN=ERROR

# Set parallelization parameters
NUM_PHYSICAL_CORES=$(nproc)
NUM_PARALLEL_JOBS=4

# Split test_operator into up to 15 processes when every GPU has at least 80 GB. All shards share
# one GPU and each has its own CUDA context, so 15 shards peak near 30 GB of GPU memory.
NUM_TEST_SHARDS=1
MIN_GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -n 1)
if [ "${MIN_GPU_MEM_MIB:-0}" -ge 80000 ] 2>/dev/null; then
    NUM_TEST_SHARDS=$(( NUM_PHYSICAL_CORES < 15 ? NUM_PHYSICAL_CORES : 15 ))
    # An odd shard count balances the load better, since skipped cases tend to repeat every 2, 4 or 8 tests
    NUM_TEST_SHARDS=$(( NUM_TEST_SHARDS - (1 - NUM_TEST_SHARDS % 2) ))
    NUM_PARALLEL_JOBS=$NUM_TEST_SHARDS
fi

cd $TE_PATH/tests/cpp
cmake -GNinja -Bbuild . -DNVTE_TEST_NUM_SHARDS=$NUM_TEST_SHARDS
cmake --build build
export OMP_NUM_THREADS=$((NUM_PHYSICAL_CORES / NUM_PARALLEL_JOBS))
ctest --test-dir build -j$NUM_PARALLEL_JOBS --output-on-failure --output-junit $XML_LOG_DIR/ctest_cppunittest.xml
