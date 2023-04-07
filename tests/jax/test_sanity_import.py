# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

try:
    import transformer_engine.jax
    te_imported = True
except:
    te_imported = False

assert te_imported, 'transformer_engine import failed'
print("OK")
