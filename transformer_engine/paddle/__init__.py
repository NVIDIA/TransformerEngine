# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for Paddle"""

# pylint: disable=wrong-import-position,wrong-import-order

import logging
from importlib.metadata import version

from transformer_engine.common import is_package_installed


def _load_library():
    """Load shared library with Transformer Engine C extensions"""
    module_name = "transformer_engine_paddle"

    if is_package_installed(module_name):
        assert is_package_installed("transformer_engine"), "Could not find `transformer-engine`."
        assert is_package_installed(
            "transformer_engine_cu12"
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

    if is_package_installed("transformer-engine-cu12"):
        if not is_package_installed(module_name):
            logging.info(
                "Could not find package %s. Install transformer-engine using 'pip"
                " install transformer-engine[paddle]==VERSION'",
                module_name,
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
