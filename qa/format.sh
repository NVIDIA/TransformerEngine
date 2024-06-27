# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Utility file to run pre-commit hooks locally
# Usage: bash qa/format.sh

set -e

: "${TE_PATH:=.}"

cd $TE_PATH

pip install pre-commit
pre-commit run --all-files
