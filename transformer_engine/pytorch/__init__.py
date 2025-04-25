# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for pyTorch"""

# pylint: disable=wrong-import-position,wrong-import-order

import logging
import functools
import sys
import importlib
import importlib.util
from importlib.metadata import version
from packaging.version import Version as PkgVersion

import torch

from transformer_engine.common import get_te_path, is_package_installed
from transformer_engine.common import _get_sys_extension

_logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def torch_version() -> tuple[int, ...]:
    """Get PyTorch version"""
    return PkgVersion(str(torch.__version__)).release


def _load_library():
    """Load shared library with Transformer Engine C extensions"""
    module_name = "transformer_engine_torch"

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
            f" v{version('transformer-engine-cu12')}. Install transformer-engine using "
            "'pip3 install transformer-engine[pytorch]==VERSION'"
        )

    if is_package_installed("transformer-engine-cu12"):
        if not is_package_installed(module_name):
            _logger.info(
                "Could not find package %s. Install transformer-engine using "
                "'pip3 install transformer-engine[pytorch]==VERSION'",
                module_name,
            )

    extension = _get_sys_extension()
    try:
        so_dir = get_te_path() / "transformer_engine"
        so_path = next(so_dir.glob(f"{module_name}.*.{extension}"))
    except StopIteration:
        try:
            so_dir = get_te_path() / "transformer_engine" / "wheel_lib"
            so_path = next(so_dir.glob(f"{module_name}.*.{extension}"))
        except StopIteration:
            so_dir = get_te_path()
            so_path = next(so_dir.glob(f"{module_name}.*.{extension}"))

    spec = importlib.util.spec_from_file_location(module_name, so_path)
    solib = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = solib
    spec.loader.exec_module(solib)


assert torch_version() >= (2, 1), f"Minimum torch version 2.1 required. Found {torch_version()}."


_load_library()
from transformer_engine.pytorch.module import LayerNormLinear
from transformer_engine.pytorch.module import Linear
from transformer_engine.pytorch.module import LayerNormMLP
from transformer_engine.pytorch.module import LayerNorm
from transformer_engine.pytorch.module import RMSNorm
from transformer_engine.pytorch.module import GroupedLinear
from transformer_engine.pytorch.module import Fp8Padding, Fp8Unpadding
from transformer_engine.pytorch.module import initialize_ub
from transformer_engine.pytorch.module import destroy_ub
from transformer_engine.pytorch.attention import DotProductAttention
from transformer_engine.pytorch.attention import MultiheadAttention
from transformer_engine.pytorch.attention import InferenceParams
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
from transformer_engine.pytorch.transformer import TransformerLayer
from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_permute_with_probs,
    moe_unpermute,
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
)
from transformer_engine.pytorch.fp8 import fp8_autocast
from transformer_engine.pytorch.fp8 import fp8_model_init
from transformer_engine.pytorch.graph import make_graphed_callables
from transformer_engine.pytorch.distributed import checkpoint
from transformer_engine.pytorch.distributed import CudaRNGStatesTracker
from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context
from transformer_engine.pytorch import ops
from transformer_engine.pytorch import optimizers
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

try:
    torch._dynamo.config.error_on_nested_jit_trace = False
except AttributeError:
    pass  # error_on_nested_jit_trace was added in PyTorch 2.2.0
