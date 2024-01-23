#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

python_files=`find transformer_engine tests setup.py examples -name '*.py'`
for f in $python_files
do
  black $f
done
