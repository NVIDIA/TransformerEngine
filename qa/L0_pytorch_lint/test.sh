# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: "${TE_PATH:=/opt/transformerengine}"

pip install cpplint==1.6.0 pylint==3.3.1
if [ -z "${PYTHON_ONLY}" ]
then
  cd $TE_PATH
  echo "Checking common API headers"
  python -m cpplint --root transformer_engine/common/include --recursive transformer_engine/common/include
  echo "Checking C++ files"
  python -m cpplint --recursive --exclude=transformer_engine/common/include --exclude=transformer_engine/build_tools/build transformer_engine/common
  python -m cpplint --recursive transformer_engine/pytorch
fi
if [ -z "${CPP_ONLY}" ]
then
  cd $TE_PATH
  echo "Checking Python files"
  python -m pylint --recursive=y transformer_engine/common transformer_engine/pytorch
fi
