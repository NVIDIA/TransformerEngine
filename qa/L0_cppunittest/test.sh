# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

# Find TE
: ${TE_PATH:=/opt/transformerengine}
TE_LIB_PATH=`pip show transformer-engine | grep Location | cut -d ' ' -f 2`
export LD_LIBRARY_PATH=$TE_LIB_PATH:$LD_LIBRARY_PATH

# Find MPI
MPI_HOME=${MPI_HOME:-/usr/local/mpi}
NVTE_MPI_INCLUDE="$MPI_HOME/lib"

cd $TE_PATH/tests/cpp
cmake -GNinja -Bbuild -DNVTE_MPI_INCLUDE=$NVTE_MPI_INCLUDE .
cmake --build build
ctest --test-dir build -j4
