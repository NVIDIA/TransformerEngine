# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

# Find TE
: ${TE_PATH:=/opt/transformerengine}
TE_LIB_PATH=`pip show transformer-engine | grep Location | cut -d ' ' -f 2`
export LD_LIBRARY_PATH=$TE_LIB_PATH:$LD_LIBRARY_PATH

cd $TE_PATH/tests/cpp
cmake -GNinja -Bbuild .
cmake --build build
ctest --test-dir build -j4
