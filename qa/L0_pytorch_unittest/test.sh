# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip3 install pytest==8.2.1
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_sanity.py
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_recipe.py
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_deferred_init.py
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_numerics.py
NVTE_CUDNN_MXFP8_NORM=0 PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_cuda_graphs.py
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_jit.py
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_fused_rope.py
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_float8tensor.py
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_gqa.py
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_fused_optimizer.py
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_multi_tensor.py
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_fusible_ops.py
python3 -m pytest -v -s $TE_PATH/tests/pytorch/test_permutation.py
NVTE_TORCH_COMPILE=0 NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python3 -m pytest -o log_cli=true --log-cli-level=INFO -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn.py
