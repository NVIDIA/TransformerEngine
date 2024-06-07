# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -xe

pip install pytest==8.2.1
: ${TE_PATH:=/opt/transformerengine}
pytest -Wignore -v $TE_PATH/tests/paddle
pytest -Wignore -v $TE_PATH/examples/paddle/mnist
