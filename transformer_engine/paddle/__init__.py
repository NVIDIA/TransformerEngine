# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for Paddle"""

# pylint: disable=wrong-import-position,wrong-import-order

import sys
import subprocess
import importlib
from importlib.metadata import version


def _load_library():
    """Load shared library with Transformer Engine C extensions"""
    module_name = "transformer_engine_paddle"

    if subprocess.run([sys.executable, "-m", "pip", "show", module_name]).returncode == 0:
        assert (
            importlib.util.find_spec("transformer_engine") is not None
        ), "Could not find `transformer-engine`."
        assert (
            importlib.util.find_spec("transformer_engine_cu12") is not None
        ), "Could not find `transformer-engine-cu12`."
        assert (
            version(module_name)
            == version("transformer-engine")
            == version("transformer-engine-cu12")
        ), (
            "TransformerEngine package version mismatch. Found"
            f" {module_name} v{version(module_name)}, transformer-engine"
            f" v{version('transformer-engine')}, and transformer-engine-cu12"
            f" v{version('transformer-engine-cu12')}. Install transformer-engine using 'pip install"
            " transformer-engine[paddle]==VERSION'"
        )

    if (
        subprocess.run([sys.executable, "-m", "pip", "show", "transformer-engine-cu12"]).returncode
        == 0
    ):
        assert importlib.util.find_spec(module_name) is not None, (
            f"Could not find package {module_name}. Install transformer-engine using 'pip install"
            " transformer-engine[paddle]==VERSION'"
        )

    from transformer_engine import transformer_engine_paddle  # pylint: disable=unused-import


_load_library()
from .fp8 import fp8_autocast
from .layer import (
    Linear,
    LayerNorm,
    LayerNormLinear,
    LayerNormMLP,
    FusedScaleMaskSoftmax,
    DotProductAttention,
    MultiHeadAttention,
    TransformerLayer,
    RotaryPositionEmbedding,
)
from .recompute import recompute
