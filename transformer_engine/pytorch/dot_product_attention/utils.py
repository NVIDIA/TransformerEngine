# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from importlib.metadata import version as get_pkg_version
from importlib.metadata import PackageNotFoundError
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeAlias
import warnings
import logging  # for get_attention_backend()
import functools

from dataclasses import dataclass, fields
import numpy as np
from packaging.version import Version as PkgVersion

import torch  # for get_attention_backend()
import torch.nn.functional as F
import transformer_engine_torch as tex
import transformer_engine as te
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    QKVLayout,
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor  # for AttentionParams
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
        AttentionLogging._log_level = _log_levels[
            AttentionLogging._log_level if AttentionLogging._log_level in [0, 1, 2] else 2
        ]
        AttentionLogging._stream_handler.setFormatter(AttentionLogging._formatter)
        AttentionLogging.fa_logger.setLevel(AttentionLogging._log_level)
        if not AttentionLogging.fa_logger.hasHandlers():
            AttentionLogging.fa_logger.addHandler(AttentionLogging._stream_handler)


# --------


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
        FlashAttentionUtils.v3_0_0_beta = (
            PkgVersion("3.0.0b") < FlashAttentionUtils.fa3_version < PkgVersion("3.0.0")
        )
        FlashAttentionUtils.use_v3 = True


# Create a typedef/alias for code readibility
FAUtils: TypeAlias = FlashAttentionUtils
# --------


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


# --------


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
            str(FlashAttentionUtils.version) if FAUtils.is_installed else "not installed"
        ),
        "flash_attn_3_version": (
            str(FAUtils.fa3_version) if FAUtils.v3_is_installed else "not installed"
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

    # Filter: Environment variables
    use_flash_attention = int(os.getenv("NVTE_FLASH_ATTN", "1"))
    use_fused_attention = int(os.getenv("NVTE_FUSED_ATTN", "1"))
    use_unfused_attention = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))
    if not use_flash_attention and FAUtils.is_installed:
        logger.debug("Disabling FlashAttention due to NVTE_FLASH_ATTN=0")
    if not use_fused_attention:
        logger.debug("Disabling FusedAttention due to NVTE_FUSED_ATTN=0")
    if not use_unfused_attention:
        logger.debug("Disabling UnfusedDotProductAttention due to NVTE_UNFUSED_ATTN=0")

    # Filter: Compute capability
    if device_compute_capability < (8, 0):
        if use_flash_attention and FAUtils.is_installed:
            logger.debug("Disabling FlashAttention as it requires compute capability sm80+")
        use_flash_attention = False
        if use_fused_attention:
            logger.debug("Disabling FusedAttention as it requires compute capability sm80+")
            use_fused_attention = False
    if device_compute_capability < (9, 0):
        if use_flash_attention and FAUtils.v3_is_installed:
            logger.debug("Disabling FlashAttention 3 as it requires compute capability sm90+")
        FAUtils.use_v3 = False

    # Filter: Data type
    if qkv_dtype not in [torch.bfloat16, torch.float16] or qkv_type not in [
        torch.Tensor,
        Float8Tensor,
    ]:
        if use_flash_attention and FAUtils.is_installed:
            logger.debug(
                "Disabling FlashAttention due to unsupported QKV data type. "
                "Supported: qkv_dtype = {torch.bfloat16, torch.float16}. "
                "Found: qkv_dtype = %s.",
                qkv_dtype,
            )
        use_flash_attention = False
        if use_fused_attention:
            logger.debug(
                "Disabling FusedAttention due to unsupported QKV data type. "
                "Supported: qkv_dtype = {torch.bfloat16, torch.float16}. "
                "Found: qkv_dtype = %s.",
                qkv_dtype,
            )
            use_fused_attention = False

    # Filter: Execution type
    if fp8 and fp8_meta["recipe"].fp8_dpa:
        if use_flash_attention and not FAUtils.use_v3:
            if FAUtils.is_installed:
                logger.debug("Disabling FlashAttention as FlashAttention 2 does not support FP8")
            use_flash_attention = False
        if use_flash_attention and FAUtils.use_v3 and is_training:
            logger.debug(
                "Disabling FlashAttention as FlashAttention 3 does not support FP8 training"
            )
            use_flash_attention = False
        if use_unfused_attention:
            logger.debug("Disabling UnfusedDotProductAttention as it does not support FP8")
            use_unfused_attention = False

    # Filter: Head dimension
    if use_flash_attention and head_dim_qk != head_dim_v:
        if FAUtils.is_installed:
            logger.debug("Disabling FlashAttention as it does not support MLA.")
        use_flash_attention = False
    if use_flash_attention and (
        head_dim_qk > 256
        or head_dim_qk % 8 != 0
        or (
            head_dim_qk > 192
            and device_compute_capability not in ((8, 0), (9, 0), (10, 0), (12, 0))
        )
    ):
        if FAUtils.is_installed:
            logger.debug(
                "Disabling FlashAttention due to unsupported head_dim_qk and head_dim_v. "
                "Supported: head_dim_qk = head_dim_v, head_dim_qk %%8 = 0, "
                "head_dim_qk <= 256 (>192 requires sm80/90/100+). "
                "Found: head_dim_qk = %s, head_dim_v = %s, on sm%s.",
                head_dim_qk,
                head_dim_v,
                ".".join([str(i) for i in device_compute_capability]),
            )
        use_flash_attention = False
    qkv_layout_group = qkv_layout.replace("b", "").replace("s", "").replace("t", "")
    if use_fused_attention and head_dim_qk != head_dim_v and qkv_layout_group != "hd_hd_hd":
        logger.debug(
            "Disabling FusedAttention as MLA is not supported with qkv_layout = %s",
            qkv_layout,
        )
        use_fused_attention = False

    # Filter: QKV layout
    qkv_format = "".join([i for i in qkv_layout.split("_")[0] if i.isalpha()])
    if qkv_format == "thd":
        if use_unfused_attention:
            logger.debug("Disabling UnfusedDotProductAttention for qkv_format = thd")
            use_unfused_attention = False
        if use_flash_attention and pad_between_seqs:
            if FAUtils.is_installed:
                logger.debug(
                    "Disabling FlashAttention for qkv_format = thd when there is "
                    "padding between sequences, i.e. [a, a, PAD, b, b, b, PAD, c, PAD]"
                )
            use_flash_attention = False

    # Filter: Dropout
    if attention_dropout != 0.0 and use_flash_attention and FAUtils.use_v3:
        logger.debug("Disabling FlashAttention 3 for dropout")
        FAUtils.use_v3 = False

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
    if context_parallel and use_flash_attention:
        if fp8 and fp8_meta["recipe"].fp8_dpa:
            if FAUtils.is_installed:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with FP8"
                )
            use_flash_attention = False
        if "bottom_right" in attn_mask_type:
            if FAUtils.is_installed:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with"
                    " causal_bottom_right masking"
                )
            use_flash_attention = False
        elif "causal" in attn_mask_type and max_seqlen_q != max_seqlen_kv:
            if FAUtils.is_installed:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with"
                    " causal masking for cross-attention"
                )
            use_flash_attention = False
        elif core_attention_bias_type not in ["no_bias", "post_scale_bias"]:
            if FAUtils.is_installed:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with bias"
                    " type of %s",
                    core_attention_bias_type,
                )
            use_flash_attention = False
        elif qkv_format == "thd" and core_attention_bias_type != "no_bias":
            if FAUtils.is_installed:
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
        if use_flash_attention and FAUtils.is_installed:
            logger.debug("Disabling FlashAttention for arbitrary mask")
        use_flash_attention = False
        if use_fused_attention:
            logger.debug("Disabling FusedAttention for arbitrary mask")
        use_fused_attention = False
    if (
        use_flash_attention
        and FAUtils.use_v3
        and attn_mask_type in ["causal", "padding_causal"]
        and max_seqlen_q != max_seqlen_kv
    ):
        logger.warning(
            "Disabling FlashAttention 3 as it only supports bottom-right-diagonal "
            "causal mask since flash-attn 2.1. See "
            "https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag"
        )
        FAUtils.use_v3 = False
    if (
        use_flash_attention
        and attn_mask_type in ["causal", "padding_causal"]
        and max_seqlen_q != max_seqlen_kv
    ):
        if FAUtils.v2_1_plus:
            logger.warning(
                "Disabling FlashAttention as it only supports bottom-right-diagonal "
                "causal mask since flash-attn 2.1. See "
                "https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag"
            )
            use_flash_attention = False
        if not FAUtils.is_installed:
            FAUtils.max_version = PkgVersion("2.1")
    if (
        use_flash_attention
        and attn_mask_type in ["causal_bottom_right", "padding_causal_bottom_right"]
        and max_seqlen_q != max_seqlen_kv
    ):
        if not FAUtils.is_installed:
            FAUtils.version_required = PkgVersion("2.1")
        elif not FAUtils.v2_1_plus and not FAUtils.use_v3:
            logger.warning(
                "Disabling FlashAttention as it only supports top-left-diagonal "
                "causal mask before flash-attn 2.1. See "
                "https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag"
            )
            use_flash_attention = False
    if (
        use_flash_attention
        and FAUtils.use_v3
        and fp8
        and fp8_meta["recipe"].fp8_dpa
        and "padding" in attn_mask_type
    ):
        logger.debug("Disabling FlashAttention 3 for FP8 and padding masks")
        FAUtils.use_v3 = False

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
        if use_flash_attention and (window_size[0] != -1 or window_size[1] not in [-1, 0]):
            if FAUtils.use_v3:
                logger.debug(
                    "Disabling FlashAttention 3 as it does not support sliding window attention"
                )
                FAUtils.use_v3 = False
            if not FAUtils.is_installed:
                FAUtils.version_required = PkgVersion("2.3")
            elif not FAUtils.v2_3_plus:
                logger.debug(
                    "Disabling FlashAttention as sliding window attention requires flash-attn 2.3+"
                )
                use_flash_attention = False

    # Filter: Attention bias
    #    backend                 |      bias types              | ALiBi diagonal alignment
    # ---------------------------------------------------------------------------------
    # FlashAttention             | no_bias, alibi/alibi_slopes  | bottom right
    # FusedAttention             | no_bias, post_scale_bias     |
    #                            | alibi/alibi_slopes           | top left,
    #                            |                              | bottom_right (converts to a 'post_scale_bias' bias)
    # UnfusedDotProductAttention | no_bias, pre/post_scale_bias |
    #                            | alibi/alibi_slopes           | both; converts to a 'post_scale_bias' bias
    if use_flash_attention and core_attention_bias_type == "alibi":
        if FAUtils.use_v3:
            logger.debug("Disabling FlashAttention 3 for ALiBi")
            FAUtils.use_v3 = False
        if not FAUtils.is_installed:
            FAUtils.version_required = PkgVersion("2.4")
        elif not FAUtils.v2_4_plus:
            logger.debug("Disabling FlashAttention as ALiBi requires flash-attn 2.4+")
            use_flash_attention = False

    if use_flash_attention and (
        core_attention_bias_type not in ["no_bias", "alibi"]
        or core_attention_bias_shape is not None
    ):
        if FAUtils.is_installed:
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
    if use_flash_attention and deterministic:
        if not FAUtils.is_installed:
            FAUtils.version_required = PkgVersion("2.4.1")
        elif not FAUtils.v2_4_1_plus and not FAUtils.use_v3:
            logger.warning(
                "Disabling FlashAttention as version <2.4.1 does not support deterministic "
                "execution. To use FlashAttention with deterministic behavior, "
                "please install flash-attn >= 2.4.1."
            )
            use_flash_attention = False
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

    # All available backends
    available_backends = [use_flash_attention, use_fused_attention, use_unfused_attention]

    # `FusedAttention` and `FlashAttention` are faster backends than `UnfusedDotProductAttention`.
    # When `FusedAttention` does not support the provided attention params, and `FlashAttention`
    # does, we recommend users to install flash-attn if not installed already.
    if not use_fused_attention and use_flash_attention and not FAUtils.is_installed:
        logger.warning(
            "flash-attn may provide important feature support or performance improvement."
            " Please install flash-attn %s.",
            _get_supported_versions(
                FAUtils.version_required,
                FAUtils.max_version,
            ),
        )
    if use_flash_attention and not FAUtils.is_installed:
        use_flash_attention = False
        available_backends[0] = False

    logger.debug(
        "Available backends = {FlashAttention=%s, FusedAttention=%s%s,"
        " UnfusedDotProductAttention=%s}",
        bool(available_backends[0]),
        bool(available_backends[1]),
        (
            f" (sub-backend {int(fused_attention_backend)})"
            if fused_attention_backend is not None
            else ""
        ),
        bool(available_backends[2]),
    )

    # Select FusedAttention for performance
    if (
        use_flash_attention
        and use_fused_attention
        and fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]
    ):
        if device_compute_capability >= (9, 0):
            logger.debug(
                "Disabling FlashAttention to give FusedAttention preference on Hopper+ "
                "for performance reasons"
            )
            use_flash_attention = False
    if (
        use_flash_attention
        and use_fused_attention
        and fused_attention_backend == FusedAttnBackend["FP8"]
        and FAUtils.use_v3
    ):
        logger.debug(
            "Disabling FlashAttention 3 to give FusedAttention preference for performance reasons "
            "in FP8 execution"
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
        selected_backend = "FlashAttention"
    elif use_fused_attention:
        selected_backend = f"FusedAttention (sub-backend {int(fused_attention_backend)})"
    elif use_unfused_attention:
        selected_backend = "UnfusedDotProductAttention"
    logger.debug("Selected backend = %s", selected_backend)

    """global _attention_backends
    _attention_backends["use_flash_attention"] = use_flash_attention
    _attention_backends["use_fused_attention"] = use_fused_attention
    _attention_backends["fused_attention_backend"] = fused_attention_backend
    _attention_backends["use_unfused_attention"] = use_unfused_attention
    _attention_backends["backend_selection_requires_update"] = False"""

    return (
        use_flash_attention,
        use_fused_attention,
        fused_attention_backend,
        use_unfused_attention,
        available_backends,
    )


# --------


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
