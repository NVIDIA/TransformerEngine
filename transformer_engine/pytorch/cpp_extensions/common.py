# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
from typing import Optional, Tuple, List, Union
import torch
import transformer_engine_extensions as tex
from ..constants import TE_DType

TORCH_DType = {
    tex.DType.kFloat8E4M3: torch.uint8,
    tex.DType.kFloat8E5M2: torch.uint8,
    tex.DType.kFloat16: torch.half,
    tex.DType.kBFloat16: torch.bfloat16,
    tex.DType.kFloat32: torch.float32,
    tex.DType.kInt32: torch.int32,
}

"""TE FP8 extensions and GEMMs"""
def check_tensor(x: torch.Tensor):
    """Check tensor properties."""
    assert (x.is_cuda and x.is_contiguous()
            ), "Tensor should be a GPU tensor and contiguous."

