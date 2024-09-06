# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for transpose extensions"""
from typing import List, Tuple, Union
import torch
import transformer_engine_torch as tex


__all__ = [
    "multi_padding_fused",
]


def multi_padding_fused(
    inp: torch.Tensor,
    row_list: List[int],
    padded_row_list: List[int],
    out: torch.Tensor,
) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], None]:
    """Padding"""

    tex.fused_multi_row_padding(
        inp,
        out,
        row_list,
        padded_row_list,
    )
