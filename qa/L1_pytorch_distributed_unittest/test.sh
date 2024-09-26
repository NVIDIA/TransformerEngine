# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

# pkg_resources is deprecated in setuptools 70+ and the packaging submodule
# has been removed from it. This is a temporary fix until upstream MLM fix.
pip install setuptools==69.5.1

: ${TE_PATH:=/opt/transformerengine}
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_comm_gemm_overlap.py

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout bcce6f54e075e3c3374ea67adefe54f3f2da2b07
sed -i -e '1504,1505d' megatron/model/transformer.py
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_convergence.py
python $TE_PATH/tests/pytorch/distributed/print_logs.py
