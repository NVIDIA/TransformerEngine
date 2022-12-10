# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: "${TE_PATH:=/opt/transformerengine}"

pip install cpplint==1.6.0 pylint==2.13.5 clang-format==15.0.4 black==19.10b0 click==8.0.2
if [ -z "${PYTHON_ONLY}" ]
then
  echo "Checking format for C/C++/CUDA files"
  clang-format -Werror --dry-run $(find transformer_engine -name '*.c' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h')
  echo "Checking linter for C/C++/CUDA files"
  cd $TE_PATH && \
  cpplint --recursive transformer_engine
fi
if [ -z "${CPP_ONLY}" ]
then
  echo "Checking format for Python files"
  black --line-length 100 transformer_engine/ --check
  echo "Checking linter for Python files"
  cd $TE_PATH && \
  pylint --recursive=y transformer_engine
fi
