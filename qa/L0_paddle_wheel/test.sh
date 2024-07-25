# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: "${TE_PATH:=/opt/transformerengine}"

cd $TE_PATH
pip uninstall -y transformer-engine
export NVTE_RELEASE_BUILD=1
python setup.py bdist_wheel
pip install dist/*
cd transformer_engine/paddle
python setup.py bdist_wheel

export NVTE_RELEASE_BUILD=0
cd $TE_PATH
pip install dist/*

python $TE_PATH/tests/paddle/test_sanity_import.py
