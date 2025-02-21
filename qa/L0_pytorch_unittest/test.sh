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

run_test pytest -v -s $TE_PATH/tests/pytorch/test_sanity.py
run_test pytest -v -s $TE_PATH/tests/pytorch/test_recipe.py
run_test pytest -v -s $TE_PATH/tests/pytorch/test_deferred_init.py
run_test env PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 pytest -v -s $TE_PATH/tests/pytorch/test_numerics.py
run_test env NVTE_CUDNN_MXFP8_NORM=0 PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 pytest -v -s $TE_PATH/tests/pytorch/test_cuda_graphs.py
run_test pytest -v -s $TE_PATH/tests/pytorch/test_jit.py
run_test pytest -v -s $TE_PATH/tests/pytorch/test_fused_rope.py
run_test pytest -v -s $TE_PATH/tests/pytorch/test_float8tensor.py
run_test pytest -v -s $TE_PATH/tests/pytorch/test_gqa.py
run_test pytest -v -s $TE_PATH/tests/pytorch/test_fused_optimizer.py
run_test pytest -v -s $TE_PATH/tests/pytorch/test_multi_tensor.py
run_test pytest -v -s $TE_PATH/tests/pytorch/test_fusible_ops.py
run_test pytest -v -s $TE_PATH/tests/pytorch/test_permutation.py
run_test env NVTE_TORCH_COMPILE=0 NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 pytest -o log_cli=true --log-cli-level=INFO -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn.py

exit $error_occurred
