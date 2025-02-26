# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

: ${TE_PATH:=/opt/transformerengine}

pip3 install pytest==8.2.1

FAIL=0

python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_numerics.py || FAIL=1
python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_fusible_ops.py || FAIL=1
python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py || FAIL=1
python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_comm_gemm_overlap.py || FAIL=1
# python3 -m pytest -v -s $TE_PATH/tests/pytorch/distributed/test_fusible_ops_with_userbuffers.py || FAIL=1  ### TODO Debug UB support with te.Sequential
python3 -m pytest -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn_with_cp.py || FAIL=1

exit $FAIL
