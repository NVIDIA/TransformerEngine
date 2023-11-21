# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for Paddle"""

from .fp8 import fp8_autocast
from .layer import (Linear, LayerNorm, LayerNormLinear, LayerNormMLP, FusedScaleMaskSoftmax,
                    DotProductAttention, MultiHeadAttention, TransformerLayer)
from .recompute import recompute

try:
    # `pip install .` will move JAX examples to the installed TE/framework folder
    from . import examples
except ImportError as e:
    try:
        # if the examples are not here, then TE must be installed in editable/develop mode
        from ...examples import paddle as examples
    except ImportError as e:
        pass
