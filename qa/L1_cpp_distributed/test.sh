# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

# Find TE
: ${TE_PATH:=/opt/transformerengine}
TE_LIB_PATH=$(pip3 show transformer-engine | grep -E "Location:|Editable project location:" | tail -n 1 | awk '{print $NF}')
export LD_LIBRARY_PATH=$TE_LIB_PATH:$LD_LIBRARY_PATH

cd $TE_PATH/tests/cpp
cmake -GNinja -S. -Bbuild
cmake --build build
mpirun --allow-run-as-root --np 4 --oversubscribe ./build/comm_gemm/test_comm_gemm
