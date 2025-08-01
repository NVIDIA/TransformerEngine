# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.



: ${TE_PATH:=/opt/transformerengine}
: ${NVTE_TEST_NVINSPECT_FEATURE_DIRS:=$TE_PATH/transformer_engine/debug/features}
: ${NVTE_TEST_NVINSPECT_CONFIGS_DIR:=$TE_PATH/tests/pytorch/debug/test_configs/}

# Config with the dummy feature which prevents nvinspect from being disabled.
# Nvinspect will be disabled if no feature is active.
: ${NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE:=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml}

FAIL=0

pip install pytest==8.2.1
pytest -v -s $TE_PATH/tests/pytorch/debug/test_sanity.py  --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/debug/test_config.py --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS || FAIL=1
pytest -v -s $TE_PATH/tests/pytorch/debug/test_numerics.py --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS || FAIL=1
NVTE_TORCH_COMPILE=0 pytest -v -s $TE_PATH/tests/pytorch/debug/test_api_features.py --feature_dirs=$NVTE_TEST_NVINSPECT_FEATURE_DIRS --configs_dir=$NVTE_TEST_NVINSPECT_CONFIGS_DIR || FAIL=1

# standard sanity and numerics tests with initialized debug
NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE NVTE_TEST_NVINSPECT_FEATURE_DIRS=$NVTE_TEST_NVINSPECT_FEATURE_DIRS PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 pytest -v -s $TE_PATH/tests/pytorch/test_sanity.py || FAIL=1
NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$NVTE_TEST_NVINSPECT_DUMMY_CONFIG_FILE NVTE_TEST_NVINSPECT_FEATURE_DIRS=$NVTE_TEST_NVINSPECT_FEATURE_DIRS PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 pytest -v -s $TE_PATH/tests/pytorch/test_numerics.py || FAIL=1

exit $FAIL
