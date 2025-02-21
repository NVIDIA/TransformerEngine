# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip install pytest==8.2.1

error_occurred=0

run_test() {
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        error_occurred=1
    fi
}

run_test pytest -v -s $TE_PATH/tests/pytorch/distributed/test_numerics.py
run_test pytest -v -s $TE_PATH/tests/pytorch/distributed/test_fusible_ops.py
run_test pytest -v -s $TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py
run_test pytest -v -s $TE_PATH/tests/pytorch/distributed/test_comm_gemm_overlap.py
# pytest -v -s $TE_PATH/tests/pytorch/distributed/test_fusible_ops_with_userbuffers.py  ### TODO Debug UB support with te.Sequential
run_test pytest -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn_with_cp.py

exit $error_occurred
