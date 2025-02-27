# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip3 install prettytable
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout b3375a0e38c10e2300ef4be031f7dcabab52b448
python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_convergence.py
python3 $TE_PATH/tests/pytorch/distributed/print_logs.py
