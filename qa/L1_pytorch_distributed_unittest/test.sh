# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}
: ${DUMMY_CONFIG_FILE:=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml}
: ${FEATURE_DIRS:=$TE_PATH/transformer_engine/debug/features}


pip install pytest==8.2.1
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_numerics.py
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_fusible_ops.py
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_torch_fsdp2.py
pytest -v -s $TE_PATH/tests/pytorch/distributed/test_comm_gemm_overlap.py
# pytest -v -s $TE_PATH/tests/pytorch/distributed/test_fusible_ops_with_userbuffers.py  ### TODO Debug UB support with te.Sequential
pytest -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn_with_cp.py

# debug tests
pytest -v -s $TE_PATH/tests/pytorch/debug/test_distributed.py --feature_dirs=$FEATURE_DIRS
# standard numerics tests with initialized debug
DEBUG=True CONFIG_FILE=$DUMMY_CONFIG_FILE FEATURE_DIRS=$FEATURE_DIRS pytest -v -s $TE_PATH/tests/pytorch/distributed/test_numerics.py
