# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout bcce6f54e075e3c3374ea67adefe54f3f2da2b07
sed -i -e '1504,1505d' megatron/model/transformer.py
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_convergence.py
python $TE_PATH/tests/pytorch/distributed/print_logs.py
