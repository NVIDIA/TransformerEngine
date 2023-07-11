# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
#pytest -Wignore -v $TE_PATH/tests/paddle
export LD_LIBRARY_PATH=/workspace/cudnn/lib64:$LD_LIBRARY_PATH
echo LD_LIBRARY_PATH is $LD_LIBRARY_PATH
echo CUDNN_PATH is $CUDNN_PATH
pytest -s -v $TE_PATH/tests/paddle
