# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

# Find TE
: ${TE_PATH:=/opt/transformerengine}
TE_LIB_PATH=`pip show transformer-engine | grep Location | cut -d ' ' -f 2`
export LD_LIBRARY_PATH=$TE_LIB_PATH:$LD_LIBRARY_PATH

# Set parallelization parameters
NUM_PHYSICAL_CORES=16
NUM_PARALLEL_JOBS=4
export OMP_NUM_THREADS=$(NUM_PHYSICAL_CORES / NUM_PARALLEL_JOBS)

cd $TE_PATH/tests/cpp
cmake -GNinja -Bbuild .
cmake --build build
ctest --test-dir build -j$NUM_PARALLEL_JOBS
