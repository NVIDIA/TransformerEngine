# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: "${TE_PATH:=/opt/transformerengine}"

cd $TE_PATH
pip uninstall -y transformer-engine
export NVTE_RELEASE_BUILD=1
python setup.py bdist_wheel
cd transformer_engine/pytorch
python setup.py sdist

export NVTE_RELEASE_BUILD=0
pip install dist/*
cd $TE_PATH
pip install dist/*
