# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from importlib.metadata import version as get_pkg_version
from importlib.metadata import PackageNotFoundError
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import logging # for get_attention_backend()
import functools

from dataclasses import dataclass, fields
import numpy as np
from packaging.version import Version as PkgVersion

import torch # for get_attention_backend()
import torch.nn.functional as F
import transformer_engine_torch as tex
import transformer_engine as te
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    QKVLayout,
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor # for AttentionParams
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.constants import TE_DType


from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    get_cudnn_version,
)

# ----Global constants----
# NVTE_DEBUG = 0/1 # disables/enables debug mode, default = 0
_NVTE_DEBUG = int(os.getenv("NVTE_DEBUG", "0"))
# NVTE_DEBUG_LEVEL = 0/1/2 # enables more and more verbose debug mode, default = 0
_NVTE_DEBUG_LEVEL = int(os.getenv("NVTE_DEBUG_LEVEL", "0"))


# ----Helper/Util classes-----
# --K: Used by get_attention_backend(), DPA and FA classes--
class AttentionLogging:
    _log_level = _NVTE_DEBUG * _NVTE_DEBUG_LEVEL
    _formatter = logging.Formatter("[%(levelname)-8s | %(name)-19s]: %(message)s")
    _stream_handler = logging.StreamHandler()
    # TODO: Move fa_logger to FAUtils
    fa_logger = logging.getLogger(__name__)

    @staticmethod
    def setup_logging():
        _log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        AttentionLogging._log_level = _log_levels[AttentionLogging._log_level if AttentionLogging._log_level in [0, 1, 2] else 2]
        AttentionLogging._stream_handler.setFormatter(AttentionLogging._formatter)
        AttentionLogging.fa_logger.setLevel(AttentionLogging._log_level)
        if not AttentionLogging.fa_logger.hasHandlers():
            AttentionLogging.fa_logger.addHandler(AttentionLogging._stream_handler)
#--------