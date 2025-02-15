# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Utility file to run pre-commit hooks locally
# Usage: bash qa/format.sh

set -e

: "${TE_PATH:=.}"

cd $TE_PATH

pip3 install pre-commit
pre-commit run --all-files
