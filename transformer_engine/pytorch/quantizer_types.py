# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Lightweight types shared by the quantization recipe machinery and quantizers.

The classes in this module are plain dataclasses with no dependencies on the
rest of ``transformer_engine.pytorch``. They live here, rather than in
``quantization.py``, so that both :mod:`transformer_engine.pytorch.quantization`
and the ``transformer_engine.pytorch.tensor`` package can import them at module
level without creating a circular import. For backward compatibility they are
re-exported from :mod:`transformer_engine.pytorch.quantization` and
:mod:`transformer_engine.pytorch`.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Union

from transformer_engine.common.recipe import Format

__all__ = [
    "QuantizerRole",
    "QuantizerRequest",
    "DelayedScalingRequest",
]


@dataclasses.dataclass(frozen=True)
class QuantizerRole:
    """Identity of a tensor slot requesting a quantizer.

    TE modules populate all fields they know about.
    User factories inspect only the fields they care about.

    .. warning::
        **EXPERIMENTAL**: QuantizerRole is experimental, still under active development,
        and the API is subject to change without notice. Use at your own risk.

    Fields
    ------
    module_type : str
        Module type that emits this role, e.g. `"linear"`, `"grouped_linear"`, `"dpa"`.
        Empty string when not provided.
    tensor_type : str
        What tensor is being quantized, in the module's own vocabulary.
        Linear modules: `"input"`, `"weight"`, `"grad_output"`, etc.
        DPA: `"qkv"`, `"s"`, etc.
        Empty string when not provided.
    name : str
        Caller-provided module instance name (e.g. set by the training
        framework), e.g.
        `"qkv"`, `"proj"`, `"fc1"`, `"fc2"`, `"linear_39"`.
        Empty string when not provided.
    """

    module_type: str = ""
    tensor_type: str = ""
    name: str = ""

    def __str__(self) -> str:
        parts = []
        if self.module_type:
            parts.append(f"module_type={self.module_type}")
        if self.tensor_type:
            parts.append(f"tensor_type={self.tensor_type}")
        if self.name:
            parts.append(f"name={self.name}")
        return "|".join(parts) if parts else "QuantizerRole()"


@dataclasses.dataclass(frozen=True)
class QuantizerRequest:
    """Base class for stateful quantizer requests.

    Custom recipe factories return ``QuantizerRequest`` subclasses (instead of
    quantizer instances) when the quantizer requires TE-managed shared state.
    TE detects these requests, allocates the required state, and replaces them
    with real quantizer instances.

    .. warning::
        **EXPERIMENTAL**: QuantizerRequest is experimental, still under active
        development, and the API is subject to change without notice.
    """


@dataclasses.dataclass(frozen=True)
class DelayedScalingRequest(QuantizerRequest):
    """Request a Float8Quantizer with TE-managed delayed scaling state.

    .. warning::
        **EXPERIMENTAL**: DelayedScalingRequest is experimental, still under active
        development, and the API is subject to change without notice.

    All ``DelayedScalingRequest`` instances within the same ``CustomRecipeState``
    must share identical parameter values.

    Parameters
    ----------
    fp8_format : Format, default = Format.HYBRID
        Controls fwd/bwd dtype (HYBRID = E4M3 fwd, E5M2 bwd).
    margin : int, default = 0
        Margin for scaling factor computation.
    amax_history_len : int, default = 1024
        Length of the amax history window.
    amax_compute_algo : str or Callable, default = "max"
        Algorithm for choosing amax from history.
    scaling_factor_compute_algo : Callable or None, default = None
        Custom scaling factor computation.
    reduce_amax : bool, default = True
        Whether to all-reduce amax across the distributed group.
    """

    fp8_format: Format = Format.HYBRID
    margin: int = 0
    amax_history_len: int = 1024
    amax_compute_algo: Union[str, Callable] = "max"
    scaling_factor_compute_algo: Optional[Callable] = None
    reduce_amax: bool = True
