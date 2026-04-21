# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
# NOTE: This test is duplicated from pre-commit, and could be deleted.

set -e

: "${TE_PATH:=/opt/transformerengine}"

pip3 install cpplint==1.6.0 ruff==0.11.4
if [ -z "${PYTHON_ONLY}" ]
then
  cd $TE_PATH
  echo "Checking common API headers"
  python3 -m cpplint --root transformer_engine/common/include --recursive transformer_engine/common/include
  echo "Checking C++ files"
  python3 -m cpplint --recursive --exclude=transformer_engine/common/include --exclude=transformer_engine/build_tools/build transformer_engine/common
  python3 -m cpplint --recursive transformer_engine/jax
fi
if [ -z "${CPP_ONLY}" ]
then
  cd $TE_PATH
  echo "Checking Python files"
  python3 -m ruff check transformer_engine/common transformer_engine/jax
fi
