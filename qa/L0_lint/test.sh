# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: "${TE_PATH:=/opt/transformerengine}"

pip install cpplint==1.6.0 pylint==2.13.5
if [ -z "${PYTHON_ONLY}" ]
then
  echo "Checking common API headers"
  cd $TE_PATH && \
  cpplint --root transformer_engine/common/include --recursive transformer_engine/common/include
  echo "Checking C++ files"
  cd $TE_PATH && \
  cpplint --recursive --exclude=transformer_engine/common/include transformer_engine
fi
if [ -z "${CPP_ONLY}" ]
then
  echo "Checking Python files"
  cd $TE_PATH && \
  pylint --recursive=y transformer_engine
fi
