# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile glue for Transformer Engine.

Public API is re-exported here so callers keep importing from
``transformer_engine.pytorch.dynamo`` regardless of the internal module layout:

  * :mod:`.quantizer_opaque` -- make a tensorless quantizer a torch.compile
    *value* opaque type (:func:`register_value_opaque_quantizer`).
"""

from .quantizer_opaque import register_value_opaque_quantizer

__all__ = [
    "register_value_opaque_quantizer",
]
