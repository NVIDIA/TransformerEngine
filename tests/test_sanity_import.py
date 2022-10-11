# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import importlib

te = importlib.util.find_spec('transformer_engine.pytorch')
assert te is not None, 'transformer_engine import failed'
print("OK")
