# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Utils/Helper classes and methods for attention
"""
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import logging
import functools

from dataclasses import dataclass, fields
import numpy as np
from packaging.version import Version as PkgVersion

import torch
import torch.nn.functional as F
import transformer_engine_torch as tex
import transformer_engine as te
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    QKVLayout,
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
    META_QKV,
    META_DQKV,
    META_O,
    META_DO,
    META_S,
    META_DP,
    META_O_CP,
    META_DQKV_CP,
)
from transformer_engine.pytorch.attention.inference import InferenceParams
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.constants import TE_DType


from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    get_cudnn_version,
)

from transformer_engine.pytorch.jit import jit_fuser

# NVTE_DEBUG = 0/1 # disables/enables debug mode, default = 0
_NVTE_DEBUG = int(os.getenv("NVTE_DEBUG", "0"))
# NVTE_DEBUG_LEVEL = 0/1/2 # enables more and more verbose debug mode, default = 0
_NVTE_DEBUG_LEVEL = int(os.getenv("NVTE_DEBUG_LEVEL", "0"))
_NVTE_FLASH_ATTN = int(os.getenv("NVTE_FLASH_ATTN", "1"))

_cu_seqlens_cache = {}


class AttentionLogging:
    """
    Manage logging for attention module
    """

    _log_level = _NVTE_DEBUG * _NVTE_DEBUG_LEVEL
    _formatter = logging.Formatter("[%(levelname)-8s | %(name)-19s]: %(message)s")
    _stream_handler = logging.StreamHandler()
    fa_logger = logging.getLogger(__name__)
    _is_logging_setup = False

    @staticmethod
    def setup_logging():
        """
        Set up log levels, logger and handlers
        """
        _log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        AttentionLogging._log_level = _log_levels[
            AttentionLogging._log_level if AttentionLogging._log_level in [0, 1, 2] else 2
        ]
        AttentionLogging._stream_handler.setFormatter(AttentionLogging._formatter)
        AttentionLogging.fa_logger.setLevel(AttentionLogging._log_level)
        if not AttentionLogging.fa_logger.hasHandlers():
            AttentionLogging.fa_logger.addHandler(AttentionLogging._stream_handler)
        AttentionLogging._is_logging_setup = True


@functools.lru_cache(maxsize=None)
def _get_supported_versions(version_min, version_max):
    """
    Calculate version info based on min and max numbers
    """
    return ">= " + str(version_min) + ", " + "<= " + str(version_max)


def maybe_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Make tensor contiguous if final stride is not 1."""
    return tensor.contiguous() if tensor.stride(-1) != 1 else tensor


class FlashAttentionUtils:
    """
    Manage Flash Attention versioning information
    """

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
    v2_5_plus = False
    v2_5_7_plus = False
    v2_6_0_plus = False
    v2_7_0_plus = False
    warning_printed = False

    v3_is_installed = False
    fa3_version = PkgVersion("0")
    v3_0_0_beta = False
    use_v3 = False
    # FA3 from FA 2.7.3+/hopper has different APIs than FA3 from 2.7.2/hopper
    # Please follow these instructions to install FA3
    v3_installation_steps = """\
(1) git clone https://github.com/Dao-AILab/flash-attention.git
(2) cd flash-attention/ && git checkout 27f501d && cd hopper/ && python setup.py install
(3) python_path=`python -c "import site; print(site.getsitepackages()[0])"`
(4) mkdir -p $python_path/flash_attn_3
(5) wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.py"""
    v3_warning_printed = False

    @staticmethod
    def set_flash_attention_version():
        """
        Setup version info for FA v2.x
        """
        FlashAttentionUtils.is_installed = True
        FlashAttentionUtils.v2_plus = FlashAttentionUtils.version >= PkgVersion("2")
        FlashAttentionUtils.v2_1_plus = FlashAttentionUtils.version >= PkgVersion("2.1")
        FlashAttentionUtils.v2_3_plus = FlashAttentionUtils.version >= PkgVersion("2.3")
        FlashAttentionUtils.v2_4_plus = FlashAttentionUtils.version >= PkgVersion("2.4")
        FlashAttentionUtils.v2_4_1_plus = FlashAttentionUtils.version >= PkgVersion("2.4.1")
        FlashAttentionUtils.v2_5_plus = FlashAttentionUtils.version >= PkgVersion("2.5.0")
        FlashAttentionUtils.v2_5_7_plus = FlashAttentionUtils.version >= PkgVersion("2.5.7")
        FlashAttentionUtils.v2_6_0_plus = FlashAttentionUtils.version >= PkgVersion("2.6.0")
        FlashAttentionUtils.v2_7_0_plus = FlashAttentionUtils.version >= PkgVersion("2.7.0")

    @staticmethod
    def set_flash_attention_3_params():
        """
        Setup version info for FA v3.x
        """
        FlashAttentionUtils.v3_is_installed = True
        FlashAttentionUtils.v3_0_0_beta = (
            PkgVersion("3.0.0b") < FlashAttentionUtils.fa3_version < PkgVersion("3.0.0")
        )


@dataclass(eq=True)
class AttentionParams:
    """
    Attention parameters used to determine which backend to be used.

    Parameters
    ----------
    qkv_type: Union[torch.Tensor, Float8Tensor], default = `torch.Tensor`
        Type of query/key/value tensors, {`torch.Tensor`, `Float8Tensor`}.
    qkv_dtype: torch.dtype, default = `torch.bfloat16`
        Data type of query/key/value tensors.
    qkv_layout: str, default = "sbh3d"
        Query/key/value tensor memory layout.
    batch_size: int, default = 1
        Batch size.
    num_heads: int, default = 16
        Number of attention heads in the query tensor.
    num_gqa_groups: int, default = 16
        Number of attention heads in key and value tensors.
    max_seqlen_q: int, default = 128
        Maximum sequence length of the query tensor.
    max_seqlen_kv: int, default = 128
        Maximum sequence length of the key and value tensors.
    head_dim_qk: int, default = 64
        The size of each attention head in query and key tensors.
    head_dim_v: int, default = 64
        The size of each attention head in the value tensor.
    attn_mask_type: str, default = `no_mask`
        Attention mask type, {`no_mask`, `padding`, `causal`, `padding_causal`,
        `causal_bottom_right`, `padding_causal_bottom_right`, `arbitrary`}
    window_size: Tuple[int, int], default = None
        Sliding window attention size.
    alibi_slopes_shape: Optional[Union[torch.Size, List]], default = `None`
        Tensor shape of :attr:`alibi_slopes` in `DotProductAttention`.
    core_attention_bias_type: str, default = `no_bias`
        Attention bias type, {`no_bias`, `pre_scale_bias`, `post_scale_bias`, `alibi`}.
    core_attention_bias_shape: str, default = `1hss`
        Attention bias shape, {`1hss`, `b1ss`, `bhss`}.
    core_attention_bias_requires_grad: bool, default = `True`
        Whether attention bias requires gradient.
    pad_between_seqs: bool, default = `False`
        Whether there is padding between sequences in a batch.
        This only applies to `qkv_format=thd`.
    attention_dropout: float, default = 0.0
        Attention dropout.
    context_parallel: bool, default = `False`
        Whether context parallelism is used or not.
    deterministic: bool, default = `False`
        Whether to run `DotProductAttention` with determinism or not.
    is_training: bool, default = `True`
        Whether in training mode (`True`) or inference mode (`False`)
    fp8: bool, default = `False`
        Whether `DotProductAttention` is in an `fp8_autocast` region.
    fp8_meta: Optional[Dict[str Any]], default = `None`
        The FP8 metadata tensor of `DotProductAttention`.
    inference_params: Optional[InferenceParams], default = `None`
        Inference-related parameters. See InferenceParams for details.
    """

    qkv_type: Union[torch.Tensor, Float8Tensor] = torch.Tensor
    qkv_dtype: torch.dtype = torch.bfloat16
    qkv_layout: str = "sbh3d"
    batch_size: int = 1
    num_heads: int = 16
    num_gqa_groups: int = 16
    max_seqlen_q: int = 128
    max_seqlen_kv: int = 128
    head_dim_qk: int = 64
    head_dim_v: int = 64
    attn_mask_type: str = "no_mask"
    window_size: Union[Tuple[int, int], None] = None
    alibi_slopes_shape: Union[torch.Size, List, None] = None
    core_attention_bias_type: str = "no_bias"
    core_attention_bias_shape: str = "1hss"
    core_attention_bias_requires_grad: bool = True
    pad_between_seqs: bool = False
    attention_dropout: float = 0.0
    context_parallel: bool = False
    deterministic: bool = False
    is_training: bool = True
    fp8: bool = False
    fp8_meta: Union[Dict[str, Any], None] = None
    inference_params: Optional[InferenceParams] = None

    def __eq__(self, other):
        """
        Overwrite dataclass.__eq__ so that only fp8_meta["recipe"] is compared,
        since all other entries of fp8_meta are unused in get_attention_backend.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented
        for field in fields(self):
            fname = field.name
            sf = getattr(self, fname)
            of = getattr(other, fname)
            if fname != "fp8_meta":
                if sf != of:
                    return False
            elif sf.get("recipe", None) != of.get("recipe", None):
                return False
        return True


def get_attention_backend(
    attention_params: AttentionParams = None,
):
    """
    Select the appropriate attention backend/sub-backend based on user input and runtime environment.

    Parameters
    ----------
    See `AttentionParams`.

    Returns
    ----------
    use_flash_attention: bool
        Whether the `FlashAttention` backend has been selected.
    use_fused_attention: bool
        Whether the `FusedAttention` backend has been selected.
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
        If `use_fused_attention = True`, one of `FusedAttention` three sub-backends, else `None`.
    use_unfused_attention: bool
        Whether the `UnfusedDotProductAttention` backend has been selected.
    available_backends: List[bool]
        All available backends that could support the provided input. A list of Booleans
        in the form of [use_flash_attention, use_fused_attention, use_unfused_attention].
    """
    # NOTE: As part of refactoring attention.py, populating the _attention_backends cache in attention
    # is no longer performed at the end of get_attention_backend(), but the responsibility of doing so
    # is shifted over to the caller of this function
    qkv_type = attention_params.qkv_type
    qkv_dtype = attention_params.qkv_dtype
    qkv_layout = attention_params.qkv_layout
    batch_size = attention_params.batch_size
    num_heads = attention_params.num_heads
    num_gqa_groups = attention_params.num_gqa_groups
    max_seqlen_q = attention_params.max_seqlen_q
    max_seqlen_kv = attention_params.max_seqlen_kv
    head_dim_qk = attention_params.head_dim_qk
    head_dim_v = attention_params.head_dim_v
    attn_mask_type = attention_params.attn_mask_type
    window_size = attention_params.window_size
    alibi_slopes_shape = attention_params.alibi_slopes_shape
    core_attention_bias_type = attention_params.core_attention_bias_type
    core_attention_bias_shape = attention_params.core_attention_bias_shape
    core_attention_bias_requires_grad = attention_params.core_attention_bias_requires_grad
    pad_between_seqs = attention_params.pad_between_seqs
    attention_dropout = attention_params.attention_dropout
    context_parallel = attention_params.context_parallel
    deterministic = attention_params.deterministic
    is_training = attention_params.is_training
    fp8 = attention_params.fp8
    fp8_meta = attention_params.fp8_meta
    inference_params = attention_params.inference_params

    # Run config
    logger = logging.getLogger("DotProductAttention")
    logger.setLevel(AttentionLogging._log_level)
    if not logger.hasHandlers():
        logger.addHandler(AttentionLogging._stream_handler)
    device_compute_capability = get_device_compute_capability()
    cudnn_version = get_cudnn_version()
    run_config = {
        "transformer_engine_version": te.__version__,
        "compute_capability": "sm"
        + str(10 * device_compute_capability[0] + device_compute_capability[1]),
        "flash_attn_version": (
            str(FlashAttentionUtils.version)
            if FlashAttentionUtils.is_installed
            else "not installed"
        ),
        "flash_attn_3_version": (
            str(FlashAttentionUtils.fa3_version)
            if FlashAttentionUtils.v3_is_installed
            else "not installed"
        ),
        "cudnn_version": ".".join([str(i) for i in cudnn_version]),
    }
    attention_params_dict = {
        field.name: getattr(attention_params, field.name) for field in fields(attention_params)
    }
    run_config.update(attention_params_dict)
    if fp8:
        run_config["NVTE_FP8_DPA_BWD"] = int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
    logger.debug("Running with config=%s", run_config)

    # The following sections check if `FlashAttention` supports the provided attention params,
    # regardless of whether FA2 or FA3 is installed. If FA2 or FA3 is not installed but is
    # necessary for performance/functionality, a warning will be issued to prompt users to
    # install an appropriate FA version.
    qkv_format, q_format, _ = get_qkv_format(qkv_layout, inference_params)

    # Filter: Environment variables
    use_flash_attention = int(os.getenv("NVTE_FLASH_ATTN", "1"))
    use_flash_attention_2 = use_flash_attention
    use_flash_attention_3 = use_flash_attention
    flash_attention_backend = None
    use_fused_attention = int(os.getenv("NVTE_FUSED_ATTN", "1"))
    use_unfused_attention = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))
    if not use_flash_attention_2 and FlashAttentionUtils.is_installed:
        logger.debug("Disabling FlashAttention 2 due to NVTE_FLASH_ATTN=0")
    if not use_flash_attention_3 and FlashAttentionUtils.v3_is_installed:
        logger.debug("Disabling FlashAttention 3 due to NVTE_FLASH_ATTN=0")
    if not use_fused_attention:
        logger.debug("Disabling FusedAttention due to NVTE_FUSED_ATTN=0")
    if not use_unfused_attention:
        logger.debug("Disabling UnfusedDotProductAttention due to NVTE_UNFUSED_ATTN=0")

    # Filter: Compute capability
    if device_compute_capability < (8, 0):
        if use_flash_attention_2 and FlashAttentionUtils.is_installed:
            logger.debug("Disabling FlashAttention 2 for compute capability < sm80")
        use_flash_attention_2 = False
        if use_fused_attention:
            logger.debug("Disabling FusedAttention for compute capability < sm80")
            use_fused_attention = False
    if device_compute_capability != (9, 0):
        if use_flash_attention_3 and FlashAttentionUtils.v3_is_installed:
            logger.debug("Disabling FlashAttention 3 for compute capability != sm90")
        use_flash_attention_3 = False

    # Filter: Data type
    if qkv_dtype not in [torch.bfloat16, torch.float16]:
        if use_flash_attention_2 and FlashAttentionUtils.is_installed:
            logger.debug(
                "Disabling FlashAttention 2 for unsupported qkv_dtype = %s. "
                "Supported: qkv_dtype = {torch.bfloat16, torch.float16}. ",
                qkv_dtype,
            )
        use_flash_attention_2 = False
    if qkv_dtype not in [torch.bfloat16, torch.float16, torch.float8_e4m3fn] or qkv_type not in [
        torch.Tensor,
        Float8Tensor,
    ]:
        if use_flash_attention_3 and FlashAttentionUtils.v3_is_installed:
            logger.debug(
                "Disabling FlashAttention 3 for unsupported qkv_dtype = %s, qkv_type = %s. "
                "Supported: qkv_dtype = {torch.bfloat16, torch.float16, torch.float8_e4m3fn}, "
                "qkv_type = {torch.Tensor, Float8Tensor}. ",
                qkv_dtype,
                qkv_type,
            )
        use_flash_attention_3 = False
        if use_fused_attention:
            logger.debug(
                "Disabling FusedAttention for unsupported qkv_dtype = %s, qkv_type = %s. "
                "Supported: qkv_dtype = {torch.bfloat16, torch.float16, torch.float8_e4m3fn}, "
                "qkv_type = {torch.Tensor, Float8Tensor}. ",
                qkv_dtype,
                qkv_type,
            )
            use_fused_attention = False

    # Filter: Execution type
    if fp8 and fp8_meta["recipe"].fp8_dpa:
        if use_flash_attention_2 and FlashAttentionUtils.is_installed:
            logger.debug("Disabling FlashAttention 2 for FP8 attention")
        use_flash_attention_2 = False
        if use_flash_attention_3 and is_training:
            if FlashAttentionUtils.v3_is_installed:
                logger.debug("Disabling FlashAttention 3 for FP8 training")
            use_flash_attention_3 = False
        if use_unfused_attention:
            logger.debug("Disabling UnfusedDotProductAttention for FP8 attention")
            use_unfused_attention = False

    # Filter: KV cache
    # backend  | precision      |    KV cache     | architecture | qkv_format    | page_size
    # ---------------------------------------------------------------------------------------
    # Fused    | FP16/BF16      | non-paged/paged | sm80+        | bshd,sbhd,thd | >= 1
    # Flash v2 | FP16/BF16      | non-paged/paged | sm80+        | bshd,sbhd,thd | >= 256
    # Flash v3 | FP16/BF16      | non-paged/paged | sm90         | bshd,sbhd,thd | >= 1
    #          | FP8            | non-paged/paged | sm90         | thd           | >= 1
    # Unfused  | FP32/FP16/BF16 | non-paged/paged | all          | bshd,sbhd,thd | >= 1
    if inference_params is not None:
        if device_compute_capability == (8, 9) and cudnn_version < (9, 11, 0):
            logger.debug("Disabling FusedAttention for KV caching for sm89 and cuDNN < 9.11")
            use_fused_attention = False
        if context_parallel:
            logger.debug("Disabling all backends for KV caching with context parallelism")
            use_flash_attention = False
            use_fused_attention = False
            use_unfused_attention = False
        if fp8 and fp8_meta["recipe"].fp8_dpa:
            if fp8_meta["recipe"].fp8_mha:
                logger.debug("Disabling all backends for KV caching with FP8 MHA")
                use_flash_attention = False
                use_fused_attention = False
                use_unfused_attention = False
            if use_flash_attention_3 and q_format != "thd":
                if FlashAttentionUtils.v3_is_installed:
                    logger.debug("Disabling FlashAttention 3 for FP8 KV caching and non-THD")
                use_flash_attention_3 = False
            if use_fused_attention:
                logger.debug("Disabling FusedAttention for FP8 KV caching")
                use_fused_attention = False
        else:
            if q_format == "thd" and pad_between_seqs:
                logger.debug("Disabling all backends for pad_between_seqs = True and KV caching")
                use_flash_attention = False
                use_fused_attention = False
                use_unfused_attention = False
        if inference_params.is_paged:
            if use_flash_attention_2 and inference_params.page_size < 256:
                if FlashAttentionUtils.is_installed:
                    logger.debug("Disabling FlashAttention 2 for page size < 256")
                use_flash_attention_2 = False
            if use_flash_attention_2:
                if not FlashAttentionUtils.is_installed:
                    FlashAttentionUtils.version_required = PkgVersion("2.5")
                elif not FlashAttentionUtils.v2_5_plus:
                    logger.debug(
                        "Disabling FlashAttention 2 as paged attention requires flash-attn 2.5+"
                    )
                    use_flash_attention_2 = False

    # Filter: Head dimension
    if head_dim_qk != head_dim_v:
        if (use_flash_attention_2 and FlashAttentionUtils.is_installed) or (
            use_flash_attention_3 and FlashAttentionUtils.v3_is_installed
        ):
            logger.debug("Disabling FlashAttention as it does not support MLA.")
        use_flash_attention = False
        qkv_layout_group = qkv_layout.replace("b", "").replace("s", "").replace("t", "")
        if use_fused_attention and qkv_layout_group != "hd_hd_hd":
            logger.debug(
                "Disabling FusedAttention as MLA is not supported with qkv_layout = %s",
                qkv_layout,
            )
            use_fused_attention = False
    if use_flash_attention_2 and (
        head_dim_qk > 256
        or head_dim_qk % 8 != 0
        or (
            head_dim_qk > 192
            and device_compute_capability not in ((8, 0), (9, 0), (10, 0), (12, 0))
        )
    ):
        if FlashAttentionUtils.is_installed:
            logger.debug(
                "Disabling FlashAttention 2 due to unsupported head_dim_qk and head_dim_v. "
                "Supported: head_dim_qk = head_dim_v, head_dim_qk %%8 = 0, "
                "head_dim_qk <= 256 (>192 requires sm80/90/100+). "
                "Found: head_dim_qk = %s, head_dim_v = %s, on sm%s.",
                head_dim_qk,
                head_dim_v,
                ".".join([str(i) for i in device_compute_capability]),
            )
        use_flash_attention_2 = False
    if use_flash_attention_3 and (head_dim_qk > 128 or head_dim_v > 128):
        if FlashAttentionUtils.v3_is_installed:
            logger.debug("Disabling FlashAttention 3 for head_dim > 128")
        use_flash_attention_3 = False

    # Filter: QKV layout
    if qkv_format == "thd":
        if use_unfused_attention:
            logger.debug("Disabling UnfusedDotProductAttention for qkv_format = thd")
            use_unfused_attention = False
        if pad_between_seqs:
            if (use_flash_attention_2 and FlashAttentionUtils.is_installed) or (
                use_flash_attention_3 and FlashAttentionUtils.v3_is_installed
            ):
                logger.debug(
                    "Disabling FlashAttention for qkv_format = thd when there is "
                    "padding between sequences, i.e. [a, a, PAD, b, b, b, PAD, c, PAD]"
                )
            use_flash_attention = False

    # Filter: Dropout
    if attention_dropout != 0.0 and use_flash_attention_3:
        logger.debug("Disabling FlashAttention 3 for dropout")
        use_flash_attention_3 = False

    # Filter: Context parallelism
    # qkv_format | attn_mask_type              | attn_bias_type           | supported backends
    # ----------------------------------------------------------------------------------------------------
    # bshd, sbhd | self-attention:             | no_bias, post_scale_bias | FlashAttention, FusedAttention
    #            |     no_mask, causal         |                          |
    #            | cross-attention:            |                          |
    #            |     no_mask                 |                          |
    # thd        | self-attention:             | no_bias                  | FlashAttention, FusedAttention
    #            |     padding, padding_causal |                          | if no padding between sequences,
    #            | cross-attention:            |                          | FusedAttention
    #            |     padding                 |                          | if there is padding between sequences
    # Note: context parallelism requires seq_len % (cp_size * 2) == 0 for each sequence in q, k, v.
    if context_parallel and use_unfused_attention:
        logger.debug(
            "Disabling UnfusedDotProductAttention as it does not support context parallelism"
        )
        use_unfused_attention = False
    if context_parallel and (use_flash_attention_2 or use_flash_attention_3):
        if FlashAttentionUtils.is_installed or FlashAttentionUtils.v3_is_installed:
            if fp8 and fp8_meta["recipe"].fp8_dpa:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with FP8"
                )
                use_flash_attention = False
            if "bottom_right" in attn_mask_type:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with"
                    " causal_bottom_right masking"
                )
                use_flash_attention = False
            elif "causal" in attn_mask_type and max_seqlen_q != max_seqlen_kv:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with"
                    " causal masking for cross-attention"
                )
                use_flash_attention = False
            elif core_attention_bias_type not in ["no_bias", "post_scale_bias"]:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with bias"
                    " type of %s",
                    core_attention_bias_type,
                )
                use_flash_attention = False
            elif qkv_format == "thd" and core_attention_bias_type != "no_bias":
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with"
                    " attention bias for THD format"
                )
                use_flash_attention = False

    if context_parallel and use_fused_attention:
        if "bottom_right" in attn_mask_type:
            logger.debug(
                "Disabling FusedAttention as it does not support context parallelism with"
                " causal_bottom_right masking"
            )
            use_fused_attention = False
        elif "causal" in attn_mask_type and max_seqlen_q != max_seqlen_kv:
            logger.debug(
                "Disabling FusedAttention as it does not support context parallelism with causal"
                " masking for cross-attention"
            )
            use_fused_attention = False
        elif core_attention_bias_type not in ["no_bias", "post_scale_bias"]:
            logger.debug(
                "Disabling FusedAttention as it does not support context parallelism with bias type"
                " of %s",
                core_attention_bias_type,
            )
            use_fused_attention = False
        elif qkv_format == "thd" and core_attention_bias_type != "no_bias":
            logger.debug(
                "Disabling FusedAttention as it does not support context parallelism with attention"
                " bias for THD format"
            )
            use_fused_attention = False
        elif head_dim_qk != head_dim_v:
            logger.debug(
                "Disabling FusedAttention as it does not support context parallelism with MLA"
            )
            use_fused_attention = False

    # Filter: Attention mask
    # attn_mask_type              | attention_mask                       | supported backends
    # ----------------------------------------------------------------------------------------
    # no_mask                     | None                                 | All
    # padding                     |                                      | All
    #     self-attention          | One tensor in shape [b, 1, 1, sq]    |
    #     cross-attention         | Tuple of two tensors in shapes       |
    #                             | [b, 1, 1, sq] and [b, 1, 1, skv]     |
    # causal                      | None                                 |
    #     self-attention          |                                      | All
    #     cross-attention         |                                      | FusedAttention, UnfusedDotProductAttention
    # padding_causal              | Same as "padding"                    |
    #     self-attention          |                                      | All
    #     cross-attention         |                                      | FusedAttention, UnfusedDotProductAttention
    # causal_bottom_right         | None                                 | All
    # padding_causal_bottom_right | Same as "padding"                    | All
    # arbitrary                   | One tensor in shape broadcastable to | UnfusedDotProductAttention
    #                             | [b, h, sq, skv]                      |
    if attn_mask_type == "arbitrary":
        if (use_flash_attention_2 and FlashAttentionUtils.is_installed) or (
            use_flash_attention_3 and FlashAttentionUtils.v3_is_installed
        ):
            logger.debug("Disabling FlashAttention for arbitrary mask")
        use_flash_attention = False
        if use_fused_attention:
            logger.debug("Disabling FusedAttention for arbitrary mask")
        use_fused_attention = False
    if (
        (use_flash_attention_2 or use_flash_attention_3)
        and attn_mask_type in ["causal", "padding_causal"]
        and max_seqlen_q != max_seqlen_kv
    ):
        logger.warning(
            "Disabling FlashAttention as it only supports bottom-right-diagonal "
            "causal mask since flash-attn 2.1 (our minimum supported version). See "
            "https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag"
        )
        use_flash_attention = False

    # Filter: Sliding window attention
    #    backend                 |      window_size       | diagonal alignment
    # ---------------------------------------------------------------------------------
    # FlashAttention             | (-1, -1) or (>=0, >=0) | bottom right
    # FusedAttention             | (-1,  0) or (>=0, 0)   | top left
    # UnfusedDotProductAttention | (-1, -1) or (>=0, >=0) | both;
    #                            |                        | converts window_size to an 'arbitrary' mask
    if window_size is None:
        window_size = check_set_window_size(attn_mask_type, window_size)
    else:
        if use_fused_attention and (window_size[0] != -1 or window_size[1] not in [-1, 0]):
            if fp8 and (fp8_meta["recipe"].fp8_dpa or fp8_meta["recipe"].fp8_mha):
                logger.debug(
                    "Disabling FusedAttention as it does not support sliding window attention"
                    " for FP8"
                )
                use_fused_attention = False
            elif window_size[1] != 0 or attention_dropout != 0.0:
                logger.debug(
                    "Disabling FusedAttention as it only supports sliding window attention "
                    "with (left, 0) and no dropout"
                )
                use_fused_attention = False
            elif max_seqlen_q > max_seqlen_kv:
                logger.debug(
                    "Disabling FusedAttention as it does not support sliding window attention "
                    "with s_q > s_kv for cross-attention"
                )
                use_fused_attention = False
        if use_flash_attention_2 and (window_size[0] != -1 or window_size[1] not in [-1, 0]):
            if not FlashAttentionUtils.is_installed:
                FlashAttentionUtils.version_required = PkgVersion("2.3")
            elif not FlashAttentionUtils.v2_3_plus:
                logger.debug(
                    "Disabling FlashAttention as sliding window attention requires flash-attn 2.3+"
                )
                use_flash_attention_2 = False

    # Filter: Attention bias
    #    backend                 |      bias types              | ALiBi diagonal alignment
    # ---------------------------------------------------------------------------------
    # FlashAttention             | no_bias, alibi/alibi_slopes  | bottom right
    # FusedAttention             | no_bias, post_scale_bias     |
    #                            | alibi/alibi_slopes           | top left,
    #                            |                              | bottom_right (converts to a 'post_scale_bias' bias)
    # UnfusedDotProductAttention | no_bias, pre/post_scale_bias |
    #                            | alibi/alibi_slopes           | both; converts to a 'post_scale_bias' bias
    if core_attention_bias_type == "alibi":
        if use_flash_attention_3:
            if FlashAttentionUtils.v3_is_installed:
                logger.debug("Disabling FlashAttention 3 for ALiBi")
            use_flash_attention_3 = False
        if use_flash_attention_2:
            if not FlashAttentionUtils.is_installed:
                FlashAttentionUtils.version_required = PkgVersion("2.4")
            elif not FlashAttentionUtils.v2_4_plus:
                logger.debug("Disabling FlashAttention as ALiBi requires flash-attn 2.4+")
                use_flash_attention_2 = False

    if (
        core_attention_bias_type not in ["no_bias", "alibi"]
        or core_attention_bias_shape is not None
    ):
        if (use_flash_attention_2 and FlashAttentionUtils.is_installed) or (
            use_flash_attention_3 and FlashAttentionUtils.v3_is_installed
        ):
            logger.debug("Disabling FlashAttention for pre/post_scale_bias")
        use_flash_attention = False

    fu_core_attention_bias_type = core_attention_bias_type
    fu_core_attention_bias_shape = core_attention_bias_shape
    fu_core_attention_bias_requires_grad = core_attention_bias_requires_grad
    if (
        use_fused_attention
        and core_attention_bias_type == "alibi"
        and (alibi_slopes_shape is not None or max_seqlen_q != max_seqlen_kv)
    ):
        fu_core_attention_bias_type = "post_scale_bias"
        fu_core_attention_bias_requires_grad = False
        if alibi_slopes_shape is None:
            fu_core_attention_bias_shape = "1hss"
        elif len(alibi_slopes_shape) == 1 and alibi_slopes_shape[0] == num_heads:
            fu_core_attention_bias_shape = "1hss"
        elif (
            len(alibi_slopes_shape) == 2
            and alibi_slopes_shape[0] == batch_size
            and alibi_slopes_shape[1] == num_heads
        ):
            fu_core_attention_bias_shape = "bhss"

    if (
        use_fused_attention
        and fu_core_attention_bias_type == "post_scale_bias"
        and fu_core_attention_bias_shape != "1hss"
    ):
        if fu_core_attention_bias_requires_grad:
            # remove this line when cuDNN adds bwd support for
            # [1, 1, s, s], [b, 1, s, s] and [b, h, s, s]
            logger.debug("Disabling FusedAttention for dBias in [1, H, S, S] shape")
            use_fused_attention = False
        else:
            # max512 backend will only support [1, h, s, s]
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"

    # Filter: cuDNN support
    fused_attention_backend = None
    if use_fused_attention:
        q_type = TE_DType[qkv_dtype]
        kv_type = q_type
        if fp8 and fp8_meta["recipe"].fp8_dpa:
            q_type = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            kv_type = q_type
        fused_attention_backend = tex.get_fused_attn_backend(
            q_type,
            kv_type,
            QKVLayout[qkv_layout],
            AttnBiasType[fu_core_attention_bias_type],
            AttnMaskType[attn_mask_type],
            attention_dropout,
            num_heads,
            num_gqa_groups,
            max_seqlen_q,
            max_seqlen_kv,
            head_dim_qk,
            head_dim_v,
            window_size[0],
            window_size[1],
        )
        if fused_attention_backend == FusedAttnBackend["No_Backend"]:
            logger.debug("Disabling FusedAttention as no backend supports the provided input")
            use_fused_attention = False
            fused_attention_backend = None
        if (
            use_fused_attention
            and window_size is not None
            and window_size[0] != -1
            and fused_attention_backend != FusedAttnBackend["F16_arbitrary_seqlen"]
        ):
            logger.debug(
                "Disabling FusedAttention as only sub-backend %s does not support "
                "slidng window attention",
                int(fused_attention_backend),
            )
            use_fused_attention = False
            fused_attention_backend = None
        if (
            use_fused_attention
            and fused_attention_backend == FusedAttnBackend["F16_max512_seqlen"]
            and fu_core_attention_bias_type == "post_scale_bias"
            and fu_core_attention_bias_shape != "1hss"
        ):
            logger.debug(
                "Disabling FusedAttention as cuDNN sub-backend 0 only supports post_scale_bias in"
                " [1, H, S, S] shape"
            )
            use_fused_attention = False
            fused_attention_backend = None

    # Filter: Determinism
    # backend                      | deterministic
    # ---------------------------------------------
    # FlashAttention               |
    #     flash-attn >=2.0, <2.4.1 | no
    #     flash-attn >=2.4.1       | yes
    # FusedAttention               |
    #     sub-backend 0            | yes
    #     sub-backend 1            | workspace optimization path and sm90+: yes;
    #                              | otherwise: no
    #     sub-backend 2            | no
    # UnfusedDotProductAttention   | yes
    if use_flash_attention_2 and deterministic:
        if not FlashAttentionUtils.is_installed:
            FlashAttentionUtils.version_required = PkgVersion("2.4.1")
        elif not FlashAttentionUtils.v2_4_1_plus:
            logger.warning(
                "Disabling FlashAttention as version <2.4.1 does not support deterministic "
                "execution. To use FlashAttention with deterministic behavior, "
                "please install flash-attn >= 2.4.1."
            )
            use_flash_attention_2 = False
    if use_fused_attention and deterministic:
        if fused_attention_backend == FusedAttnBackend["FP8"] and is_training:
            logger.debug("Disabling FusedAttention for determinism reasons")
            use_fused_attention = False
        if (
            fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]
            and is_training
            and (
                device_compute_capability < (9, 0)
                or core_attention_bias_requires_grad
                or cudnn_version < (8, 9, 5)
            )
        ):
            logger.debug("Disabling FusedAttention for determinism reasons")
            use_fused_attention = False

    # use_flash_attention may have been set above
    use_flash_attention_2 = use_flash_attention and use_flash_attention_2
    use_flash_attention_3 = use_flash_attention and use_flash_attention_3

    # `FusedAttention` and `FlashAttention` are faster backends than `UnfusedDotProductAttention`.
    # When `FusedAttention` does not support the provided attention params, and `FlashAttention`
    # does, we recommend users to install flash-attn if not installed already.
    if not use_fused_attention and _NVTE_FLASH_ATTN:
        if (
            use_flash_attention_3
            and not FlashAttentionUtils.v3_is_installed
            and not FlashAttentionUtils.v3_warning_printed
            and torch.cuda.current_device() == 0
        ):
            logger.warning(
                "flash-attn v3 may provide important feature support or performance improvement."
                " Please install flash-attn v3 by \n%s",
                FlashAttentionUtils.v3_installation_steps,
            )
            FlashAttentionUtils.v3_warning_printed = True
        elif (
            use_flash_attention_2
            and not FlashAttentionUtils.is_installed
            and not FlashAttentionUtils.warning_printed
            and torch.cuda.current_device() == 0
        ):
            logger.warning(
                "flash-attn may provide important feature support or performance improvement."
                " Please install flash-attn %s by pip3 install flash-attn==<version>.",
                _get_supported_versions(
                    FlashAttentionUtils.version_required,
                    FlashAttentionUtils.max_version,
                ),
            )
            FlashAttentionUtils.warning_printed = True
    # All available backends
    if use_flash_attention_2 and not FlashAttentionUtils.is_installed:
        use_flash_attention_2 = False
    if use_flash_attention_3 and not FlashAttentionUtils.v3_is_installed:
        use_flash_attention_3 = False
    use_flash_attention = use_flash_attention_2 or use_flash_attention_3
    available_backends = [use_flash_attention, use_fused_attention, use_unfused_attention]
    if use_flash_attention_2:
        flash_attention_backend = FlashAttentionUtils.version
    if use_flash_attention_3:
        flash_attention_backend = FlashAttentionUtils.fa3_version

    logger.debug(
        "Available backends = {FlashAttention=%s%s, FusedAttention=%s%s,"
        " UnfusedDotProductAttention=%s}",
        bool(available_backends[0]),
        (f" ({str(flash_attention_backend)})" if flash_attention_backend is not None else ""),
        bool(available_backends[1]),
        (
            f" (sub-backend {int(fused_attention_backend)})"
            if fused_attention_backend is not None
            else ""
        ),
        bool(available_backends[2]),
    )

    # Select FusedAttention for performance
    if use_flash_attention and use_fused_attention and device_compute_capability >= (9, 0):
        logger.debug(
            "Disabling FlashAttention to give FusedAttention preference on Hopper+ "
            "for performance reasons"
        )
        use_flash_attention = False

    # Selected backend
    if use_flash_attention:
        use_fused_attention = False
        use_unfused_attention = False
    elif use_fused_attention:
        use_unfused_attention = False
    selected_backend = "NoBackend"
    if use_flash_attention:
        selected_backend = f"FlashAttention ({str(flash_attention_backend)})"
    elif use_fused_attention:
        selected_backend = f"FusedAttention (sub-backend {int(fused_attention_backend)})"
    elif use_unfused_attention:
        selected_backend = "UnfusedDotProductAttention"
    logger.debug("Selected backend = %s", selected_backend)

    return (
        use_flash_attention,
        flash_attention_backend,
        use_fused_attention,
        fused_attention_backend,
        use_unfused_attention,
        available_backends,
    )


@torch.no_grad()
def get_padding_mask(
    batch_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
):
    """Convert cu_seqlens to attention_mask"""
    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
    attention_mask_kv = torch.Tensor([]).to(dtype=torch.bool)
    for i in range(batch_size):
        attention_mask_q = torch.cat(
            [
                attention_mask_q,
                torch.Tensor([False] * seqlens_q[i] + [True] * (max_seqlen_q - seqlens_q[i]))
                .to(dtype=torch.bool)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0),
            ],
            dim=0,
        )
        attention_mask_kv = torch.cat(
            [
                attention_mask_kv,
                torch.Tensor([False] * seqlens_kv[i] + [True] * (max_seqlen_kv - seqlens_kv[i]))
                .to(dtype=torch.bool)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0),
            ],
            dim=0,
        )
    attention_mask = (
        attention_mask_q.to(device="cuda"),
        attention_mask_kv.to(device="cuda"),
    )
    return attention_mask


@torch.no_grad()
def get_full_mask(
    max_seqlen_q: int,
    max_seqlen_kv: int,
    attn_mask_type: str = "no_mask",
    attention_mask: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
    window_size: Tuple[int, int] = None,
    attention_type: str = "self",
    bottom_right_alignment: bool = True,
) -> torch.Tensor:
    """
    Get full attention mask in [..., max_seqlen_q, max_seqlen_kv] shape, based on `attn_mask_type`,
    `attention_mask`, and `window_size`. For sliding window attention, the diagonal alignment depends
    on both `attn_mask_type` and `bottom_right_alignment`, as detailed below.::

       attn_mask_type              output shape                                 diagonal alignment
       --------------------------------------------------------------------------------------------
       no_mask                     [1, 1, max_seqlen_q, max_seqlen_kv]          follow bottom_right_alignment
       causal                      [1, 1, max_seqlen_q, max_seqlen_kv]          always top left
       causal_bottom_right         [1, 1, max_seqlen_q, max_seqlen_kv]          always bottom right
       padding                     [batch_size, 1, max_seqlen_q, max_seqlen_kv] follow bottom_right_alignment
       padding_causal              [batch_size, 1, max_seqlen_q, max_seqlen_kv] always top left
       padding_causal_bottom_right [batch_size, 1, max_seqlen_q, max_seqlen_kv] always bottom right
       arbitrary                   same as attention_mask                       follow bottom_right_alignment

    .. note::

    For "padding_bottom_right" mask, or "padding" mask with `bottom_right_alignment` = True, the bottom right
    diagonal comes from the bottom right corner of the [actual_seqlens_q[i], actual_seqlens_kv[i]] matrix,
    i = 0,...,batch_size-1, not the [max_seqlen_q, max_seqlen_kv] matrix. For example, with max_seqlen_q = 4,
    max_seqlen_kv = 4, attn_mask_type = "padding", attention_type = "cross", and attention_mask = (
    [[False, False,  True, True], [False, False, False, False]],
    [[False, False, False, True], [False,  True,  True,  True]]), the returned full attention mask has [2, 4, 4]
    shape and is,::

      [[[False, False, False, True],
        [False, False, False, True],
        [ True,  True,  True, True],
        [ True,  True,  True, True]],
       [[False,  True,  True, True],
        [False,  True,  True, True],
        [False,  True,  True, True],
        [False,  True,  True, True]]]

    Parameters
    ----------
    max_seqlen_q: int
        Maximum sequence length for queries.
    max_seqlen_kv: int
        Maximum sequence length for keys and values.
    attn_mask_type: str, default = `no_mask`
        Attention mask type, {"`no_mask`", "`padding`", "`causal`", "`padding_causal`",
        "`causal_bottom_right`", "`padding_causal_bottom_right`", "`arbitrary`"}
    attention_mask: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        default = `None`
        Boolean tensor(s) used to mask out attention softmax input. Please see DotProductAttention
        for the requirements of `attention_mask` for different `attn_mask_type`s.
    window_size: Tuple[int, int], default = `None`
        Sliding window size for local attention, where query at position i attends to keys
        in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
        + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
        window and causal mask specifically. Both `causal` and `causal_bottom_right` masks
        map to `window_size = (-1, 0)` and Transformer Engine distinguishes them based on
        `attn_mask_type`.
    attention_type: str, default = "self"
        Attention type, {"self", "cross"}
    bottom_right_alignment: bool, default = `True`
        Whether to align the diagonal of the sliding window attention to the bottom right (`True`)
        or top left (`False`) corner of the softmax matrix. Ignored if `attn_mask_type` explicitly
        specifies "causal" or "causal_bottom_right".

    Returns
    ----------
    attn_mask_type: str
        For sliding window attention (>=0, >0), "arbitrary"; otherwise, the same as input `attn_mask_type`
    attention_mask: torch.Tensor
        The full attention mask based on `attn_mask_type`, `attention_mask` and `window_size`
    actual_seqlens_q: torch.Tensor
        For padding masks, the actual sequence lengths for queries, in shape [batch_size].
        For other masks, `None`.
    actual_seqlens_kv: Optional[torch.Tensor], default = `None`
        For padding masks, the actual sequence lengths for keys and values, in shape [batch_size].
        For other masks, `None`.
    """
    # perform basic checks
    change_type = window_size is not None and (
        window_size[0] != -1 or window_size[1] not in [-1, 0]
    )
    if window_size is None:
        window_size = (-1, -1)
    if "causal" in attn_mask_type:
        window_size = (window_size[0], 0)
    window_size = (
        max_seqlen_kv if window_size[0] == -1 else window_size[0],
        max_seqlen_q if window_size[1] == -1 else window_size[1],
    )

    # apply padding mask
    actual_seqlens_q = None
    actual_seqlens_kv = None
    if "padding" in attn_mask_type:
        if attention_type == "self":
            attention_mask = torch.logical_or(
                attention_mask.squeeze(1).unsqueeze(3), attention_mask
            )
        else:
            attention_mask = torch.logical_or(
                attention_mask[0].squeeze(1).unsqueeze(3), attention_mask[1]
            )
        m = attention_mask.logical_not()
        actual_seqlens_q = m[:, 0, :, 0].sum(dim=1)
        actual_seqlens_kv = m[:, 0, 0, :].sum(dim=1)

    # apply SWA mask
    mask = torch.arange(max_seqlen_q, dtype=torch.int32, device="cuda").view(
        1, 1, max_seqlen_q, 1
    ) - torch.arange(max_seqlen_kv, dtype=torch.int32, device="cuda").view(1, 1, 1, max_seqlen_kv)
    swa_left = None
    swa_right = None
    if attn_mask_type == "causal_bottom_right" or (
        attn_mask_type in ["no_mask", "arbitrary"] and bottom_right_alignment
    ):
        swa_left = mask + max_seqlen_kv - max_seqlen_q - window_size[0]
        swa_right = mask + max_seqlen_kv - max_seqlen_q + window_size[1]
    elif attn_mask_type in ["causal", "padding_causal"] or (
        attn_mask_type in ["no_mask", "padding", "arbitrary"] and not bottom_right_alignment
    ):
        swa_left = mask - window_size[0]
        swa_right = mask + window_size[1]
    elif attn_mask_type == "padding_causal_bottom_right" or (
        attn_mask_type == "padding" and bottom_right_alignment
    ):
        batch_size = attention_mask.shape[0]
        swa_left = mask.expand(batch_size, 1, max_seqlen_q, max_seqlen_kv) + (
            actual_seqlens_kv - actual_seqlens_q - window_size[0]
        ).view(batch_size, 1, 1, 1)
        swa_right = mask.expand(batch_size, 1, max_seqlen_q, max_seqlen_kv) + (
            actual_seqlens_kv - actual_seqlens_q + window_size[1]
        ).view(batch_size, 1, 1, 1)
    swa_mask = torch.logical_not(
        torch.where(swa_left <= 0, 1, 0) - torch.where(swa_right < 0, 1, 0)
    )
    if attention_mask is not None:
        attention_mask = torch.logical_or(swa_mask, attention_mask)
    else:
        attention_mask = swa_mask

    # change mask type
    if change_type:
        attn_mask_type = "arbitrary"

    return attn_mask_type, attention_mask, actual_seqlens_q, actual_seqlens_kv


@torch.no_grad()
def get_alibi(
    _alibi_cache: Dict[str, Any],
    num_heads: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    actual_seqlens_q: Optional[torch.Tensor] = None,
    actual_seqlens_kv: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    bias_dtype: Optional[torch.dtype] = None,
    bottom_right_alignment: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    num_heads: int
        Number of heads.
    max_seqlen_q: int
        Maximum sequence length for queries.
    max_seqlen_kv: int
        Maximum sequence length for keys and values.
    actual_seqlens_q: Optional[torch.Tensor], default = `None`
        Actual sequence lengths for queries, in shape [batch_size].
    actual_seqlens_kv: Optional[torch.Tensor], default = `None`
        Actual sequence lengths for keys and values, in shape [batch_size].
    alibi_slopes: Optional[torch.Tensor], default = `None`
        Custom ALiBi slopes, FP32, CUDA tensor, in shape [num_heads] or [batch_size, num_heads].
    bias_dtype: Optional[torch.dtype], default = `None`
        Dtype of the generated ALiBi bias. If None, use torch.float32.
    bottom_right_alignment: bool, default = `True`
        Whether to align the diagonal of the ALiBi bias to the bottom right corner of
        the matrix (`True`) or top left (`False`).

    Returns
    ----------
    alibi_slopes: torch.Tensor
        ALiBi slopes in FP32 and shape [num_heads] or [batch_size, num_heads].
    alibi_bias: torch.Tensor
        ALiBi bias in FP32 or `bias_dtype`. Its shape is
        (1) [1, num_heads, max_seqlen_q, max_seqlen_kv] if `alibi_slopes` is in [num_heads] shape,
        and `actual_seqlens_q` and `actual_seqlens_kv` are `None`; or
        (2) [batch_size, num_heads, max_seqlen_q, max_seqlen_kv] if `alibi_slopes` is in
        [batch_size, num_heads] shape, or, if `alibi_slopes` is in [num_heads] shape and
        `actual_seqlens_q` and `actual_seqlens_kv` are not `None`.
    """
    # NOTE: As part of refactoring attention.py, get_alibi() now receives the alibi cache from the caller
    # as an additional input arg
    if _alibi_cache["_alibi_slopes_require_update"]:
        if alibi_slopes is not None:
            _alibi_cache["_alibi_slopes"] = alibi_slopes
        else:
            n = 2 ** math.floor(math.log2(num_heads))
            m_0 = 2.0 ** (-8.0 / n)
            m = torch.pow(m_0, torch.arange(1, 1 + n))

            if n < num_heads:
                m_hat_0 = 2.0 ** (-4.0 / n)
                m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (num_heads - n), 2))
                m = torch.cat([m, m_hat])

            _alibi_cache["_alibi_slopes"] = m.to(dtype=torch.float32, device="cuda")
        _alibi_cache["_num_heads"] = num_heads
        _alibi_cache["_alibi_slopes_require_update"] = False

    if _alibi_cache["_alibi_bias_require_update"]:
        assert _alibi_cache["_alibi_slopes"] is not None, "ALiBi slopes can not be None!"
        if _alibi_cache["_alibi_slopes"].dim() == 1:
            slopes_shape = torch.Size([1, _alibi_cache["_alibi_slopes"].shape[0], 1, 1])
        elif _alibi_cache["_alibi_slopes"].dim() == 2:
            slopes_shape = torch.Size([*_alibi_cache["_alibi_slopes"].shape[:], 1, 1])
        else:
            raise ValueError("ALiBi slopes cannot exceed 2 dimensions.")

        bias = torch.arange(max_seqlen_q, dtype=torch.int32, device="cuda").view(
            1, 1, max_seqlen_q, 1
        ) - torch.arange(max_seqlen_kv, dtype=torch.int32, device="cuda").view(
            1, 1, 1, max_seqlen_kv
        )
        if actual_seqlens_q is None and actual_seqlens_kv is None:
            if bottom_right_alignment:
                bias = bias + max_seqlen_kv - max_seqlen_q
        elif actual_seqlens_q is not None and actual_seqlens_kv is not None:
            batch_size = actual_seqlens_q.shape[0]
            bias = bias.expand(batch_size, 1, max_seqlen_q, max_seqlen_kv)
            if bottom_right_alignment:
                bias = bias + (actual_seqlens_kv - actual_seqlens_q).view(batch_size, 1, 1, 1)
        else:
            assert (
                False
            ), "actual_seqlens_q and actual_seqlens_kv need to be both None or torch.Tensors!"
        bias = bias.abs().mul(-1)
        bias = bias * _alibi_cache["_alibi_slopes"].view(slopes_shape)
        _alibi_cache["_max_seqlen_q"], _alibi_cache["_max_seqlen_kv"] = max_seqlen_q, max_seqlen_kv
        _alibi_cache["_bottom_right_alignment"] = bottom_right_alignment
        bias_dtype = torch.float32 if bias_dtype is None else bias_dtype
        _alibi_cache["_alibi_bias"] = bias.contiguous().to(dtype=bias_dtype, device="cuda")
        _alibi_cache["_alibi_bias_require_update"] = False

    return _alibi_cache["_alibi_slopes"], _alibi_cache["_alibi_bias"]


def get_cu_seqlens(mask: torch.Tensor) -> torch.Tensor:
    """
    Given a padding mask of shape [batch_size, 1, 1, max_seqlen], returns an int32
    tensor of shape [batch_size + 1] containing the cumulative sequence lengths of
    the samples in a batch.
    """
    mask = mask.squeeze(1).squeeze(1)
    reduced_mask = mask.logical_not().sum(dim=1)
    cu_seqlens = reduced_mask.cumsum(dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat((zero, cu_seqlens))

    return cu_seqlens


def get_cu_seqlens_and_indices(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a padding mask of shape [batch_size, 1, 1, max_seqlen], returns an int32
    tensor of shape [batch_size + 1] containing the cumulative sequence lengths of
    the samples in a batch, and another int32 tensor of shape [batch_size * max_seqlen, 1, 1]
    containing the indices for the valid tokens.
    """
    mask = mask.squeeze(1).squeeze(1)
    bs, seqlen = mask.shape

    reduced_mask = mask.logical_not().sum(dim=1)
    cu_seqlens = reduced_mask.cumsum(dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat((zero, cu_seqlens))

    mask = mask.reshape(-1)
    indices = mask.logical_not().nonzero()
    indices = indices.unsqueeze(-1)

    num_nonzeros = indices.shape[0]
    pad_amount = bs * seqlen - num_nonzeros
    indices = F.pad(
        input=indices, pad=(0, 0, 0, 0, 0, pad_amount), mode="constant", value=float(bs * seqlen)
    )

    return cu_seqlens, indices


def get_indices(max_seqlen: int, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    Given max_seqlen and cu_seqlens of shape [batch_size + 1], returns an int32
    tensor of shape [batch_size * max_seqlen, 1, 1] containing the indices for
    the valid tokens in a batch.
    """
    bs = len(cu_seqlens) - 1
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = [i * max_seqlen + ii for i, j in enumerate(seqlens) for ii in range(j)]
    indices = torch.Tensor(indices).unsqueeze(1).unsqueeze(1).to(dtype=torch.int64, device="cuda")

    num_nonzeros = indices.shape[0]
    pad_amount = bs * max_seqlen - num_nonzeros
    indices = F.pad(
        input=indices,
        pad=(0, 0, 0, 0, 0, pad_amount),
        mode="constant",
        value=float(bs * max_seqlen),
    )

    return indices


def get_full_cu_seqlens(
    batch_size: int,
    max_seqlen: int,
    device: torch.device,
) -> torch.Tensor:
    """Cumulative sequence lengths in full data batch

    All sequences in batch have the maximum sequence length.

    """
    global _cu_seqlens_cache
    if (batch_size, max_seqlen) not in _cu_seqlens_cache:
        _cu_seqlens_cache[(batch_size, max_seqlen)] = torch.arange(
            0,
            (batch_size + 1) * max_seqlen,
            step=max_seqlen,
            dtype=torch.int32,
            device=device,
        )
    return _cu_seqlens_cache[(batch_size, max_seqlen)]


@jit_fuser
def _pack_tensor(
    indices: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Packs the given tensor using the `indices`.
    """
    padding_indice = torch.zeros(
        1, tensor.shape[1], tensor.shape[2], dtype=tensor.dtype, device=tensor.device
    )
    indices = indices.repeat(1, tensor.shape[1], tensor.shape[2])
    if isinstance(tensor, Float8Tensor):
        tensor_data = torch.cat((tensor._data, padding_indice), dim=0)
        gathered_data = torch.gather(tensor_data, 0, indices)

        packed = Float8Tensor.make_like(tensor, data=gathered_data, shape=gathered_data.shape)
    else:
        tensor = torch.cat((tensor, padding_indice), dim=0)

        packed = torch.gather(tensor, 0, indices)
    return packed


@jit_fuser
def _pack_2_tensors(
    indices: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs the given 2 tensors using the `indices`.
    """
    t1_packed = _pack_tensor(indices, t1)
    t2_packed = _pack_tensor(indices, t2)
    return t1_packed, t2_packed


@jit_fuser
def _pack_3_tensors(
    indices: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Packs the given 3 tensors using the `indices`.
    """
    t1_packed = _pack_tensor(indices, t1)
    t2_packed = _pack_tensor(indices, t2)
    t3_packed = _pack_tensor(indices, t3)
    return t1_packed, t2_packed, t3_packed


@jit_fuser
def _unpack_tensor(
    indices: torch.Tensor,
    dim0: int,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Inverse of `_pack_tensor`.
    """
    indices = indices.repeat(1, tensor.shape[1], tensor.shape[2])
    unpacked = torch.zeros(
        dim0 + 1, tensor.shape[1], tensor.shape[2], dtype=tensor.dtype, device=tensor.device
    )
    if isinstance(tensor, Float8Tensor):
        unpacked.scatter_(0, indices, tensor._data)
        unpacked_data = unpacked[0:-1, :, :]
        unpacked = Float8Tensor.make_like(tensor, data=unpacked_data, shape=unpacked_data.shape)
    else:
        unpacked.scatter_(0, indices, tensor)
        unpacked = unpacked[0:-1, :, :]
    return unpacked


@jit_fuser
def _unpack_2_tensors(
    indices: torch.Tensor,
    dim0: int,
    t1: torch.Tensor,
    t2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of `_pack_2_tensors`.
    """
    t1_unpacked = _unpack_tensor(indices, dim0, t1)
    t2_unpacked = _unpack_tensor(indices, dim0, t2)
    return t1_unpacked, t2_unpacked


@jit_fuser
def _unpack_3_tensors(
    indices: torch.Tensor,
    dim0: int,
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inverse of `_pack_3_tensors`.
    """
    t1_unpacked = _unpack_tensor(indices, dim0, t1)
    t2_unpacked = _unpack_tensor(indices, dim0, t2)
    t3_unpacked = _unpack_tensor(indices, dim0, t3)
    return t1_unpacked, t2_unpacked, t3_unpacked


class PackTensors(torch.autograd.Function):
    """
    Autograd function to pack a tensor.
    """

    @staticmethod
    def forward(
        ctx, indices: torch.Tensor, *tensors: Tuple[torch.Tensor, ...]
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # pylint: disable=missing-function-docstring
        assert 1 <= len(tensors) <= 3, f"Packing {len(tensors)} tensors not supported."
        ctx.save_for_backward(indices)
        ctx.dim0 = tensors[0].shape[0]
        if len(tensors) == 1:
            return _pack_tensor(indices, *tensors)
        if len(tensors) == 2:
            return _pack_2_tensors(indices, *tensors)
        return _pack_3_tensors(indices, *tensors)

    @staticmethod
    def backward(ctx, *grad_outputs: Tuple[torch.Tensor, ...]):
        # pylint: disable=missing-function-docstring
        (indices,) = ctx.saved_tensors
        if len(grad_outputs) == 1:
            return None, _unpack_tensor(indices, ctx.dim0, *grad_outputs)
        if len(grad_outputs) == 2:
            return None, *_unpack_2_tensors(indices, ctx.dim0, *grad_outputs)
        return None, *_unpack_3_tensors(indices, ctx.dim0, *grad_outputs)


class UnpackTensor(torch.autograd.Function):
    """
    Autograd function to unpack a tensor.
    """

    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        dim0: int,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        ctx.save_for_backward(indices)
        return _unpack_tensor(indices, dim0, tensor)

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        (indices,) = ctx.saved_tensors
        return None, None, _pack_tensor(indices, grad_output)


def get_qkv_format(
    qkv_layout: str = "bshd_bshd_bshd",
    inference_params: InferenceParams = None,
) -> str:
    """Get qkv format.

    Parameters
    ----------
    qkv_layout: str
       Memory layout of `q`, `k` and `v`. See get_qkv_layout() for more details.
    inference_params: InferenceParams, default = `None`
        InferenceParams related to KV caching.

    Returns
    ----------
    qkv_format: str, default = `sbhd`
        Dimension format for `q`, `k` and `v`, {`sbhd`, `bshd`, `thd`}.
    q_format: str
        Format of the `q` tensor, {`bshd`, `sbhd`, `thd`}.
    kv_format: str
        Format of the `k` and `v` tensors, {`bshd`, `sbhd`, `thd`}.
    """
    splited = qkv_layout.replace("paged_kv_", "").split("_")
    if inference_params is not None:
        q_format = "".join([i for i in splited[0] if i.isalpha()])
        kv_format = "".join([i for i in splited[1] if i.isalpha()])
        qkv_format = q_format + "_2" + kv_format if q_format != kv_format else q_format
    else:
        qkv_format = "".join([i for i in splited[0] if i.isalpha()])
        q_format = qkv_format
        kv_format = qkv_format
    return qkv_format, q_format, kv_format


def get_qkv_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qkv_format: str = "sbhd",
    inference_params: InferenceParams = None,
) -> str:
    """Get qkv layout.

    Parameters
    ----------
    q: torch.Tensor
        Query tensor.
    k: torch.Tensor
        Key tensor.
    v: torch.Tensor
        Value tensor.
    qkv_format: str, default = `sbhd`
        Dimension format for `q`, `k` and `v`, {`sbhd`, `bshd`, `thd`}. `s` stands for
        the sequence length dimension, `b` batch size, `h` the number of attention heads,
        `d` head size, and `t` the total number of tokens in a batch, i.e.
        `t = sum(s_i) for i = 0...b-1`.
    inference_params: InferenceParams, default = `None`
        InferenceParams related to KV caching.

    Returns
    ----------
    qkv_layout: str
       Memory layout of `q`, `k` and `v`. Each `qkv_layout` maps to a pair of `q_format` and
       `kv_format` in {`bshd`, `sbhd`, `thd`}. The `paged_kv_` prefix is used to indicate that
       paged KV caching is in play. A few examples of the layouts are as follows.

       (1) `sb3hd` means `q`, `k`, `v` are created as one chunk of memory and that they are
       interleaved in the `2`nd dimension. (2) `sbhd_sbh2d` means `q` and `kv` are created in
       two chunks and that `q` itself is contiguous and `k`, `v` are interleaved with each other
       in the `3`rd dimension, `k = kv[:,:,:,0,:]` and `v = kv[:,:,:,1,:]`. `q_format` and
       `kv_format` in this case are still both `sbhd`. (3) `paged_kv_thd_bshd_bshd` means `q` is
       created in `thd` and `k`, `v` are in `sbhd`. This is likely due to the cache format in
       paged KV caching.

       Mapping:
       `sbhd`: {`sb3hd`, `sbh3d`, `sbhd_sb2hd`, `sbhd_sbh2d`, `sbhd_sbhd_sbhd`, `paged_kv_sbhd_sbhd_sbhd`}
       `bshd`: {`bs3hd`, `bsh3d`, `bshd_bs2hd`, `bshd_bsh2d`, `bshd_bshd_bshd`, `paged_kv_bshd_bshd_bshd`}
       `thd` : {`t3hd`, `th3d`, `thd_t2hd`, `thd_th2d`, `thd_thd_thd`}
       `sbhd_2bshd`: {`sbhd_bshd_bshd`, `paged_kv_sbhd_bshd_bshd`}
       `bshd_2sbhd`: {`bshd_sbhd_sbhd`, `paged_kv_bshd_sbhd_sbhd`}
       `thd_2bshd`: {`thd_bshd_bshd`, `paged_kv_thd_bshd_bshd`}
       `thd_2sbhd`: {`thd_sbhd_sbhd`, `paged_kv_thd_sbhd_sbhd`}

    q: torch.Tensor
        Query tensor. It may be different from input `q` as we try to fit tensors to
        a supported layout.
    k: torch.Tensor
        Key tensor. It may be different from input `k` as we try to fit tensors to
        a supported layout.
    v: torch.Tensor
        Value tensor. It may be different from input `v` as we try to fit tensors to
        a supported layout.
    q_format: str
        Format of the query tensor, {`bshd`, `sbhd`, `thd`}.
    kv_format: str
        Format of the key and value tensors, {`bshd`, `sbhd`, `thd`}.
    """

    check_last_dim_contiguous = all(x.stride(-1) == 1 for x in [q, k, v])
    assert check_last_dim_contiguous, "q, k and v must have stride 1 in their last dimension!"
    if "_2" in qkv_format:
        q_format, kv_format = qkv_format.split("_2")
        is_same_q_kv_format = False
    else:
        q_format = qkv_format
        kv_format = qkv_format
        is_same_q_kv_format = True

    def run_iteratively(q, k, v):
        # check data pointers
        data_ptr = q.untyped_storage().data_ptr()
        check_ptrs_qkv = all(x.untyped_storage().data_ptr() == data_ptr for x in [q, k, v])
        check_ptrs_qk = all(x.untyped_storage().data_ptr() == data_ptr for x in [q, k])
        data_ptr = k.untyped_storage().data_ptr()
        check_ptrs_kv = all(x.untyped_storage().data_ptr() == data_ptr for x in [k, v])

        # check tensor shapes
        shape = q.shape
        check_shapes_qkv = all(shape == x.shape for x in [q, k, v])
        shape = k.shape
        check_shapes_kv = shape[:-1] == v.shape[:-1]

        # check tensor strides
        stride = q.stride()
        check_strides_qkv = all(stride == x.stride() for x in [q, k, v])
        check_strides_kv = tuple(sk / k.shape[-1] for sk in k.stride()[:-1]) == tuple(
            sv / v.shape[-1] for sv in v.stride()[:-1]
        )

        # check tensor offsets for h3d and 3hd layouts
        prod_h_d = q.shape[-1] * q.shape[-2]
        check_3hd_offsets = all(x.storage_offset() == i * prod_h_d for i, x in enumerate([q, k, v]))
        check_h3d_offsets = all(
            x.storage_offset() == i * q.shape[-1] for i, x in enumerate([q, k, v])
        )

        # check tensor offsets for hd_h2d and hd_2hd layouts
        prod_all_dims = [np.prod(x.shape) for x in [q, k]]
        offset = prod_all_dims[0] if check_ptrs_qkv else 0
        prod_h_d = k.shape[-1] * k.shape[-2]
        check_2hd_offsets = all(
            x.storage_offset() == (offset + i * prod_h_d) for i, x in enumerate([k, v])
        )
        check_h2d_offsets = all(
            x.storage_offset() == (offset + i * k.shape[-1]) for i, x in enumerate([k, v])
        )

        # check tensor offsets for hd_hd_hd layouts
        check_hd_offsets_qkv = (
            all(x.storage_offset() == sum(prod_all_dims[:i]) for i, x in enumerate([q, k, v]))
            if check_ptrs_qkv
            else all(x.storage_offset() == 0 for i, x in enumerate([q, k, v]))
        )
        check_hd_offsets_qk = (
            all(x.storage_offset() == sum(prod_all_dims[:i]) for i, x in enumerate([q, k]))
            if not check_ptrs_qkv and check_ptrs_qk
            else all(x.storage_offset() == 0 for i, x in enumerate([q, k]))
        )
        check_hd_offsets_kv = (
            all(x.storage_offset() == sum(prod_all_dims[1 : i + 1]) for i, x in enumerate([k, v]))
            if not check_ptrs_qkv and check_ptrs_kv
            else all(x.storage_offset() == 0 for i, x in enumerate([k, v]))
        )

        if check_ptrs_qkv and check_strides_qkv and check_shapes_qkv and check_3hd_offsets:
            # sb3hd, bs3hd, t3hd
            # one chunk of memory, qkv, with q, k, v interleaved at dim=-3 in qkv
            qkv_layout = qkv_format[:-2] + "3" + qkv_format[-2:]
        elif check_ptrs_qkv and check_strides_qkv and check_shapes_qkv and check_h3d_offsets:
            # sbh3d, bsh3d, th3d
            # one chunk of memory, qkv, with q, k, v interleaved at dim=-2 in qkv
            qkv_layout = qkv_format[:-1] + "3" + qkv_format[-1:]
        elif check_ptrs_kv and check_strides_kv and check_shapes_kv and check_2hd_offsets:
            # sbhd_sb2hd, bshd_bs2hd, thd_t2hd
            # two chunks of memory, q and kv, with k, v interleaved at dim=-3 in kv
            # q and kv may be disjoint or consecutive in memory, and when consecutive, they may
            # have the same data pointer, i.e. check_ptrs_qkv=True
            qkv_layout = qkv_format + "_" + qkv_format[:-2] + "2" + qkv_format[-2:]
        elif check_ptrs_kv and check_strides_kv and check_shapes_kv and check_h2d_offsets:
            # sbhd_sbh2d, bshd_bsh2d, thd_th2d
            # two chunks of memory, q and kv, with k, v interleaved at dim=-2 in kv
            # q and kv may be disjoint or consecutive in memory, and when consecutive, they may
            # have the same data pointer, i.e. check_ptrs_qkv=True
            qkv_layout = qkv_format + "_" + qkv_format[:-1] + "2" + qkv_format[-1:]
        elif (
            check_strides_kv
            and check_shapes_kv
            and (check_hd_offsets_qkv or check_hd_offsets_kv or check_hd_offsets_qk)
        ):
            # sbhd_sbhd_sbhd, bshd_bshd_bshd, thd_thd_thd
            # three chunks of memory, q, k and v, which may be disjoint or consecutive, and
            # when consecutive, they may have the same data pointer, i.e. check_ptrs_qkv=True or
            # check_ptrs_qk=True or check_ptrs_kv=True
            if is_same_q_kv_format:
                qkv_layout = "_".join(list([qkv_format]) * 3)
            else:
                qkv_layout = q_format + "_" + kv_format + "_" + kv_format
        else:
            qkv_layout = "not_supported"

        return qkv_layout

    qkv_layout = run_iteratively(q, k, v)
    if qkv_layout == "not_supported":
        # force q,k,v to be contiguous and run get_layout again
        q, k, v = [x.contiguous() for x in [q, k, v]]
        qkv_layout = run_iteratively(q, k, v)
    if qkv_layout == "not_supported":
        raise RuntimeError("The provided qkv memory layout is not supported!")

    if inference_params is not None and inference_params.is_paged:
        qkv_layout = "paged_kv_" + qkv_layout

    return qkv_layout, q, k, v, q_format, kv_format


def check_set_window_size(
    attn_mask_type: str,
    window_size: Tuple[int, int] = None,
):
    """Check if sliding window size is compliant with attention mask type.
    If not, set it to the appropriate size.

         attn_mask_type                              |   window_size
    -------------------------------------------------------------------------
    no_mask, padding, arbitrary                      | (-1, -1) or (>=0, >=0)
    causal, padding_causal                           | (-1,  0) or (>=0, 0)
    causal_bottom_right, padding_causal_bottom_right | (-1,  0) or (>=0, 0)
    """
    orig_window_size = window_size
    if "causal" in attn_mask_type:
        if orig_window_size is None:
            window_size = (-1, 0)
        elif orig_window_size == (-1, -1) or (
            orig_window_size[0] >= 0 and orig_window_size[1] != 0
        ):
            window_size = (orig_window_size[0], 0)
            warnings.warn(
                "window_size should be (-1, 0) or (>=0, 0) for attn_mask_type=" + attn_mask_type
            )
        elif orig_window_size != (-1, 0) and (orig_window_size[0] < 0 or orig_window_size[1] != 0):
            assert False, (
                "window_size should be (-1, 0) or (>=0, 0) for attn_mask_type=" + attn_mask_type
            )
    elif attn_mask_type in ["no_mask", "padding", "arbitrary"]:
        if orig_window_size is None:
            window_size = (-1, -1)
        elif orig_window_size == (-1, 0):
            window_size = (-1, -1)
            warnings.warn(
                "window_size should be (-1, -1) or (>=0, >=0) for attn_mask_type=" + attn_mask_type
            )
        elif orig_window_size != (-1, -1) and (orig_window_size[0] < 0 or orig_window_size[1] < 0):
            assert False, (
                "window_size should be (-1, -1) or (>=0, >=0) for attn_mask_type=" + attn_mask_type
            )
    else:
        assert False, "Invalid attn_mask_type: " + attn_mask_type
    return window_size


def get_attention_quantizers(fp8, quantizers, cp_specific_quantizers=False):
    """Get the list of quantizers used in attention from the quantizers list."""
    if not fp8:
        num_of_nones = 8 if cp_specific_quantizers else 6
        return [None] * num_of_nones
    QKV_quantizer = quantizers["scaling_fwd"][META_QKV]
    QKV_quantizer.internal = True
    QKV_quantizer.set_usage(rowwise=True, columnwise=False)
    O_quantizer = quantizers["scaling_fwd"][META_O]
    O_quantizer.set_usage(rowwise=True, columnwise=False)
    S_quantizer = quantizers["scaling_fwd"][META_S]
    S_quantizer.internal = True
    S_quantizer.set_usage(rowwise=True, columnwise=False)
    dQKV_quantizer = quantizers["scaling_bwd"][META_DQKV]
    dQKV_quantizer.interal = True
    dQKV_quantizer.set_usage(rowwise=True, columnwise=False)
    dO_quantizer = quantizers["scaling_bwd"][META_DO]
    dO_quantizer.set_usage(rowwise=True, columnwise=False)
    dO_quantizer.internal = True
    dP_quantizer = quantizers["scaling_bwd"][META_DP]
    dP_quantizer.set_usage(rowwise=True, columnwise=False)
    dP_quantizer.interal = True
    dQKV_CP_quantizer = quantizers["scaling_bwd"][META_DQKV_CP]
    dQKV_CP_quantizer.set_usage(rowwise=True, columnwise=False)
    dQKV_CP_quantizer.internal = True
    O_CP_quantizer = quantizers["scaling_fwd"][META_O_CP]
    O_CP_quantizer.set_usage(rowwise=True, columnwise=False)

    if cp_specific_quantizers:
        return (
            QKV_quantizer,
            O_quantizer,
            O_CP_quantizer,
            S_quantizer,
            dQKV_quantizer,
            dQKV_CP_quantizer,
            dO_quantizer,
            dP_quantizer,
        )

    return QKV_quantizer, O_quantizer, S_quantizer, dQKV_quantizer, dO_quantizer, dP_quantizer
