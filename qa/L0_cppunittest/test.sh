# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

# Find TE
: ${TE_PATH:=/opt/transformerengine}
TE_LIB_PATH=$(pip3 show transformer-engine | grep -E "Location:|Editable project location:" | tail -n 1 | awk '{print $NF}')
export LD_LIBRARY_PATH=$TE_LIB_PATH:$LD_LIBRARY_PATH

# Set parallelization parameters
NUM_PHYSICAL_CORES=$(nproc)
NUM_PARALLEL_JOBS=4

cd $TE_PATH/tests/cpp
cmake -GNinja -Bbuild .
cmake --build build
export OMP_NUM_THREADS=$((NUM_PHYSICAL_CORES / NUM_PARALLEL_JOBS))
ctest --test-dir build -j$NUM_PARALLEL_JOBS
