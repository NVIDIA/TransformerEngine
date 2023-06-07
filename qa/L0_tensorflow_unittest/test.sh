# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

: ${TE_PATH:=/opt/transformerengine}
CUDA_VISIBLE_DEVICES=0 pytest -Wignore -v $TE_PATH/tests/tensorflow
