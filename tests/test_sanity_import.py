# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

try:
    import transformer_engine.pytorch
    te_imported = True
except:
    te_imported = False

assert te_imported, 'transformer_engine import failed'
print("OK")
