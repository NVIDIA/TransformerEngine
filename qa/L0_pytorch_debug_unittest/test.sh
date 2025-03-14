# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.



: ${TE_PATH:=/opt/transformerengine}
: ${FEATURE_DIRS:=$TE_PATH/transformer_engine/debug/features}
: ${CONFIGS_DIR:=$TE_PATH/tests/pytorch/debug/test_configs/}
: ${DUMMY_CONFIG_FILE:=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml}

FAIL=0

pip install pytest==8.2.1
pytest -v -s $TE_PATH/tests/pytorch/debug/test_sanity.py  --feature_dirs=$FEATURE_DIRS || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/debug/test_config.py --feature_dirs=$FEATURE_DIRS || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/debug/test_numerics.py --feature_dirs=$FEATURE_DIRS || FAIL=1
NVTE_TORCH_COMPILE=0 pytest -v -s $TE_PATH/tests/pytorch/debug/test_api_features.py --feature_dirs=$FEATURE_DIRS --configs_dir=$CONFIGS_DIR || FAIL=1

# standard numerics tests with initialized debug
DEBUG=True CONFIG_FILE=$DUMMY_CONFIG_FILE FEATURE_DIRS=$FEATURE_DIRS PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 pytest -v -s $TE_PATH/tests/pytorch/test_numerics.py || FAIL=1

exit $FAIL
