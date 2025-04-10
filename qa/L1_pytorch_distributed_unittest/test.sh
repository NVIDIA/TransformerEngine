# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

source $(dirname "$0")/../test_utils.sh
initialize_test_variables

: ${TE_PATH:=/opt/transformerengine}

pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"

python3 -m pytest -v -s --junitxml=/logs/pytest_test_numerics.xml $TE_PATH/tests/pytorch/distributed/test_numerics.py || test_fail "test_numerics.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_fusible_ops.xml $TE_PATH/tests/pytorch/distributed/test_fusible_ops.py || test_fail "test_fusible_ops.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_torch_fsdp2.xml $TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py || test_fail "test_torch_fsdp2.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_comm_gemm_overlap.xml $TE_PATH/tests/pytorch/distributed/test_comm_gemm_overlap.py || test_fail "test_comm_gemm_overlap.py"
# python3 -m pytest -v -s --junitxml=/logs/pytest_test_fusible_ops_with_userbuffers.xml $TE_PATH/tests/pytorch/distributed/test_fusible_ops_with_userbuffers.py || test_fail "test_fusible_ops_with_userbuffers.py" ### TODO Debug UB support with te.Sequential
python3 -m pytest -v -s --junitxml=/logs/pytest_test_fused_attn_with_cp.xml $TE_PATH/tests/pytorch/fused_attn/test_fused_attn_with_cp.py || test_fail "test_fused_attn_with_cp.py"
python3 -m pytest -v -s --junitxml=/logs/pytest_test_cast_master_weights_to_fp8.xml $TE_PATH/tests/pytorch/distributed/test_cast_master_weights_to_fp8.py || test_fail "test_cast_master_weights_to_fp8.py"

check_test_results
