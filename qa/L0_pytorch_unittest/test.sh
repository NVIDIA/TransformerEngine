# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


: ${TE_PATH:=/opt/transformerengine}

pip install pytest==8.2.1

FAIL=0

pytest -v -s $TE_PATH/tests/pytorch/test_sanity.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_recipe.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_deferred_init.py || FAIL=1
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 pytest -v -s $TE_PATH/tests/pytorch/test_numerics.py || FAIL=1
NVTE_CUDNN_MXFP8_NORM=0 PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 pytest -v -s $TE_PATH/tests/pytorch/test_cuda_graphs.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_jit.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_fused_rope.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_float8tensor.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_gqa.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_fused_optimizer.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_multi_tensor.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_fusible_ops.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_permutation.py || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/test_parallel_cross_entropy.py || FAIL=1
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 pytest -o log_cli=true --log-cli-level=INFO -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn.py || FAIL=1
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 pytest -o log_cli=true --log-cli-level=INFO -v -s $TE_PATH/tests/pytorch/fused_attn/test_paged_attn.py || FAIL=1

exit $FAIL
