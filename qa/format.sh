# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Utility file to run pre-commit hooks locally
# Usage: bash qa/format.sh

set -e

pip install pre-commit
pre-commit install
pre-commit run --all-files
rm .git/hooks/pre-commit*
