# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions for C++ extensions"""
import functools
from typing import Dict, Optional, Tuple, Union

import torch

import transformer_engine_torch as tex


@functools.lru_cache(maxsize=None)
def empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor()


def canonicalize_fp8_scales(
    *,
    scale: Optional[torch.Tensor] = None,
    amax: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
    fp8_meta: Optional[tex.FP8TensorMeta] = None,
    fp8_meta_index: Union[tex.FP8FwdTensors, tex.FP8BwdTensors, None] = None,
    allow_multiple_offsets: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Canonicalize FP8 scaling factors (scale, amax, scale-inverse)

    If a scaling factor is not provided, try to access it within the
    FP8 meta tensors. Returns dict with tensors and dict with tensor
    offsets.

    """

    # Default: use provided scales with no offsets
    scale_offset = 0
    amax_offset = 0
    scale_inv_offset = 0

    # Get scales from FP8 meta tensors if needed
    if (fp8_meta is not None) and any(arg is None for arg in (scale, amax, scale_inv)):
        if fp8_meta_index is None:
            raise ValueError("Provided `fp8_meta` without corresponding `fp8_meta_index`")
        fp8_meta_index = int(fp8_meta_index)
        if scale is None:
            scale = fp8_meta.scale
            scale_offset = fp8_meta_index
        if amax is None:
            amax = fp8_meta.amax_history
            amax_offset = fp8_meta_index
        if scale_inv is None:
            scale_inv = fp8_meta.scale_inv
            scale_inv_offset = fp8_meta_index

    # Construct empty tensors if needed
    if scale is None:
        scale = empty_tensor()
        scale_offset = 0
    if amax is None:
        amax = empty_tensor()
        amax_offset = 0
    if scale_inv is None:
        scale_inv = empty_tensor()
        scale_inv_offset = 0

    # Force offsets to be the same if needed
    if not allow_multiple_offsets and not scale_offset == amax_offset == scale_inv_offset:
        if scale_offset != 0:
            scale = scale[scale_offset:]
            scale_offset = 0
        if amax_offset != 0:
            amax = amax[:, amax_offset:]
            amax_offset = 0
        if scale_inv_offset != 0:
            scale_inv = scale_inv[scale_inv_offset:]
            scale_inv_offset = 0

    # Pack tensors and offsets into dicts
    tensors = {"scale": scale, "amax": amax, "scale_inv": scale_inv}
    offsets = {
        "scale_offset": scale_offset,
        "amax_offset": amax_offset,
        "scale_inv_offset": scale_inv_offset,
    }
    return tensors, offsets
