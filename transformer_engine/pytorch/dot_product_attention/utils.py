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
_NVTE_FLASH_ATTN = int(os.getenv("NVTE_FLASH_ATTN", "1"))

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

@functools.lru_cache(maxsize=None)
def _get_supported_versions(version_min, version_max):
    return ">= " + str(version_min) + ", " + "<= " + str(version_max)

# --K: Used by get_attention_backend(), DPA and FA classes--
class FlashAttentionUtils:
    # Detect flash-attn v2 in the environment
    is_installed = False
    version = PkgVersion("0")
    version_required = PkgVersion("2.1.1")
    version_required_blackwell = PkgVersion("2.7.3")
    max_version = PkgVersion("2.7.4.post1")
    v2_plus = False
    v2_1_plus = False
    v2_3_plus = False
    v2_4_plus = False
    v2_4_1_plus = False
    v2_5_7_plus = False
    v2_6_0_plus = False
    v2_7_0_plus = False

    v3_is_installed = False
    fa3_version = PkgVersion("0")
    v3_0_0_beta = False
    use_v3 = False
    # TODO(cyang): update FA to 2.7.3 when its FA3 compilation issue is resolved
    # https://github.com/Dao-AILab/flash-attention/issues/1452
    v3_installation_steps = """\
    (1) pip install "git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2#egg=flashattn-hopper&subdirectory=hopper"
    (2) python_path=`python -c "import site; print(site.getsitepackages()[0])"`
    (3) mkdir -p $python_path/flashattn_hopper
    (4) wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.7.2/hopper/flash_attn_interface.py"""      

    @staticmethod
    def set_flash_attention_version():
        FlashAttentionUtils.is_installed = True
        FlashAttentionUtils.v2_plus = FlashAttentionUtils.version >= PkgVersion("2")
        FlashAttentionUtils.v2_1_plus = FlashAttentionUtils.version >= PkgVersion("2.1")
        FlashAttentionUtils.v2_3_plus = FlashAttentionUtils.version >= PkgVersion("2.3")
        FlashAttentionUtils.v2_4_plus = FlashAttentionUtils.version >= PkgVersion("2.4")
        FlashAttentionUtils.v2_4_1_plus = FlashAttentionUtils.version >= PkgVersion("2.4.1")
        FlashAttentionUtils.v2_5_7_plus = FlashAttentionUtils.version >= PkgVersion("2.5.7")
        FlashAttentionUtils.v2_6_0_plus = FlashAttentionUtils.version >= PkgVersion("2.6.0")
        FlashAttentionUtils.v2_7_0_plus = FlashAttentionUtils.version >= PkgVersion("2.7.0")

    # Detect flash-attn v3 in the environment
    # This section will be removed when FA3 is released as a regular FA package,
    # i.e. flashattn-hopper 3.0.0 as flash-attn 3.0.0
    @staticmethod
    def set_flash_attention_3_params():
        FlashAttentionUtils.v3_is_installed = True
        FlashAttentionUtils.v3_0_0_beta = PkgVersion("3.0.0b") < FlashAttentionUtils.fa3_version < PkgVersion("3.0.0")
        FlashAttentionUtils.use_v3 = True
#--------