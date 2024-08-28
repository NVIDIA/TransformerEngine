# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer Engine bindings for JAX"""

# pylint: disable=wrong-import-position,wrong-import-order

import sys
import subprocess
import importlib
import ctypes
from importlib.metadata import version

from transformer_engine.common import get_te_path
from transformer_engine.common import _get_sys_extension


def _load_library():
    """Load shared library with Transformer Engine C extensions"""
    module_name = "transformer_engine_jax"

    if subprocess.run([sys.executable, "-m", "pip", "show", module_name]).returncode == 0:
        assert (
            importlib.util.find_spec("transformer_engine") is not None
        ), "Could not find `transformer-engine`."
        assert (
            importlib.util.find_spec("transformer_engine_cu12") is not None
        ), "Could not find `transformer-engine-cu12`."
        assert version(module_name) == version("transformer-engine"), (
            "TransformerEngine package version mismatch. Found"
            f" {module_name} v{version(module_name)}, transformer-engine"
            f" v{version('transformer-engine')}, and transformer-engine-cu12"
            f" v{version('transformer-engine-cu12')}. Install transformer-engine using 'pip install"
            " transformer-engine[jax]==VERSION'"
        )

    if (
        subprocess.run([sys.executable, "-m", "pip", "show", "transformer-engine-cu12"]).returncode
        == 0
    ):
        assert importlib.util.find_spec(module_name) is not None, (
            f"Could not find package {module_name}. Install transformer-engine using 'pip install"
            " transformer-engine[jax]==VERSION'"
        )

    extension = _get_sys_extension()
    try:
        so_dir = get_te_path() / "transformer_engine"
        so_path = next(so_dir.glob(f"{module_name}.*.{extension}"))
    except StopIteration:
        so_dir = get_te_path()
        so_path = next(so_dir.glob(f"{module_name}.*.{extension}"))

    return ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)


_TE_JAX_LIB_CTYPES = _load_library()
from . import flax
from .fp8 import fp8_autocast, update_collections, get_delayed_scaling
from .fp8 import NVTE_FP8_COLLECTION_NAME
from .sharding import MeshResource
from .sharding import MajorShardingType, ShardingResource, ShardingType

from ..common.utils import deprecate_wrapper
from ..common.utils import DeprecatedEnum

MajorShardingType = DeprecatedEnum(
    MajorShardingType, "MajorShardingType is deprecating in the near feature."
)
ShardingType = DeprecatedEnum(ShardingType, "ShardingType is deprecating in the near feature.")
ShardingResource = deprecate_wrapper(
    ShardingResource,
    "ShardingResource is renamed to MeshResource, and will be removed in the near feature.",
)

__all__ = [
    "NVTE_FP8_COLLECTION_NAME",
    "fp8_autocast",
    "update_collections",
    "get_delayed_scaling",
    "MeshResource",
    "MajorShardingType",
    "ShardingResource",
    "ShardingType",
    "flax",
    "praxis",
]
