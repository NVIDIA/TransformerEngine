# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout f24fac4ed0dcf0522056521a93445d9a82f501a91 
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_sanity.py

