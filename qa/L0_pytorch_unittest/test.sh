# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

pip install pytest==7.2 onnxruntime==1.13.1
pytest -v -s $TE_PATH/tests/pytorch/test_sanity.py
pytest -v -s $TE_PATH/tests/pytorch/test_deferred_init.py
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 pytest -v -s $TE_PATH/tests/pytorch/test_numerics.py
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 pytest -v -s $TE_PATH/tests/pytorch/test_cuda_graphs.py
pytest -v -s $TE_PATH/tests/pytorch/test_jit.py
NVTE_TORCH_COMPILE=0 pytest -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn.py
pytest -v -s $TE_PATH/tests/pytorch/test_fused_rope.py
NVTE_TORCH_COMPILE=0 pytest -v -s $TE_PATH/tests/pytorch/test_onnx_export.py
pytest -v -s $TE_PATH/tests/pytorch/test_float8tensor.py
