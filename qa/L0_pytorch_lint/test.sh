# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: "${TE_PATH:=/opt/transformerengine}"

pip install cpplint==1.6.0 pylint==2.13.5
if [ -z "${PYTHON_ONLY}" ]
then
  cp $TE_PATH/qa/L0_pytorch_lint/CPPLINT.cfg $TE_PATH
  cd $TE_PATH
  echo "Checking common API headers"
  cpplint --root transformer_engine/common/include --recursive transformer_engine/common/include
  echo "Checking C++ files"
  cpplint --recursive --exclude=transformer_engine/common/include transformer_engine
  cpplint --recursive transformer_engine/pytorch
fi
if [ -z "${CPP_ONLY}" ]
then
  cp $TE_PATH/qa/L0_pytorch_lint/pylintrc $TE_PATH
  cd $TE_PATH
  echo "Checking Python files"
  pylint --recursive=y transformer_engine/common transformer_engine/pytorch
fi
