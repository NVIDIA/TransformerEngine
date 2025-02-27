# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention."""
import collections
from contextlib import nullcontext
from importlib.metadata import version as get_pkg_version
from importlib.metadata import PackageNotFoundError
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
from transformer_engine.pytorch.utils import (
    get_cudnn_version,
    nvtx_range_pop,
    nvtx_range_push,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd,
    fused_attn_bwd,
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
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    get_fp8_te_dtype,
    get_fp8_torch_dtype,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase
from transformer_engine.pytorch.module import LayerNormLinear, Linear
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.utils import (
    divide,
    attention_mask_func,
    split_tensor_along_dim,
    get_device_compute_capability,
    get_default_init_method,
)
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    AttnBiasTypes,
    QKVLayouts,
    dist_group_type,
    TE_DType,
)
from transformer_engine.pytorch.softmax import FusedScaleMaskSoftmax
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    get_distributed_rank,
    checkpoint,
    set_all_rng_states,
    CudaRNGStatesTracker,
    graph_safe_rng_available,
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
)
from transformer_engine.pytorch.jit import jit_fuser, no_torch_dynamo
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
    prepare_for_saving,
    restore_from_saved,
)


# NVTE_DEBUG = 0/1 # disables/enables debug mode, default = 0
_NVTE_DEBUG = int(os.getenv("NVTE_DEBUG", "0"))
# NVTE_DEBUG_LEVEL = 0/1/2 # enables more and more verbose debug mode, default = 0
_NVTE_DEBUG_LEVEL = int(os.getenv("NVTE_DEBUG_LEVEL", "0"))
_log_level = _NVTE_DEBUG * _NVTE_DEBUG_LEVEL
_log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
_log_level = _log_levels[_log_level if _log_level in [0, 1, 2] else 2]
_formatter = logging.Formatter("[%(levelname)-8s | %(name)-19s]: %(message)s")
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)
fa_logger = logging.getLogger()
fa_logger.setLevel(_log_level)
if not fa_logger.hasHandlers():
    fa_logger.addHandler(_stream_handler)


@functools.lru_cache(maxsize=None)
def _get_supported_versions(version_min, version_max):
    return ">= " + str(version_min) + ", " + "<= " + str(version_max)


_NVTE_FLASH_ATTN = int(os.getenv("NVTE_FLASH_ATTN", "1"))
_NVTE_FUSED_ATTN = int(os.getenv("NVTE_FUSED_ATTN", "1"))
_NVTE_UNFUSED_ATTN = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))

# Detect flash-attn v2 in the environment
_flash_attn_is_installed = False
_flash_attn_version = PkgVersion("0")
_flash_attn_version_required = PkgVersion("2.1.1")
_flash_attn_version_required_blackwell = PkgVersion("2.7.3")
_flash_attn_max_version = PkgVersion("2.7.4.post1")
_flash_attn_2_plus = False
_flash_attn_2_1_plus = False
_flash_attn_2_3_plus = False
_flash_attn_2_4_plus = False
_flash_attn_2_4_1_plus = False
_flash_attn_2_5_7_plus = False
_flash_attn_2_6_0_plus = False
_flash_attn_2_7_0_plus = False

flash_attn_cuda_bwd = None
flash_attn_func = None
flash_attn_varlen_func = None
_flash_attn_fwd = None
_flash_attn_bwd = None
_flash_attn_varlen_fwd = None
_flash_attn_varlen_bwd = None

try:
    _flash_attn_version = PkgVersion(get_pkg_version("flash-attn"))
except PackageNotFoundError:
    if torch.cuda.is_available() and get_device_compute_capability() >= (8, 0) and _NVTE_FLASH_ATTN:
        fa_logger.debug(
            "flash-attn v2 is not installed. To use, please install it by"
            """ "pip install flash-attn".""",
        )
else:
    if torch.cuda.is_available() and get_device_compute_capability() >= (10, 0):
        if _flash_attn_version_required_blackwell <= _flash_attn_version <= _flash_attn_max_version:
            _flash_attn_is_installed = True
    elif _flash_attn_version_required <= _flash_attn_version <= _flash_attn_max_version:
        _flash_attn_is_installed = True

    if _flash_attn_is_installed:
        from flash_attn_2_cuda import varlen_bwd as flash_attn_cuda_bwd
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        from flash_attn.flash_attn_interface import _flash_attn_forward as _flash_attn_fwd
        from flash_attn.flash_attn_interface import _flash_attn_backward as _flash_attn_bwd
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_forward as _flash_attn_varlen_fwd,
        )
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_backward as _flash_attn_varlen_bwd,
        )

        _flash_attn_2_plus = _flash_attn_version >= PkgVersion("2")
        _flash_attn_2_1_plus = _flash_attn_version >= PkgVersion("2.1")
        _flash_attn_2_3_plus = _flash_attn_version >= PkgVersion("2.3")
        _flash_attn_2_4_plus = _flash_attn_version >= PkgVersion("2.4")
        _flash_attn_2_4_1_plus = _flash_attn_version >= PkgVersion("2.4.1")
        _flash_attn_2_5_7_plus = _flash_attn_version >= PkgVersion("2.5.7")
        _flash_attn_2_6_0_plus = _flash_attn_version >= PkgVersion("2.6.0")
        _flash_attn_2_7_0_plus = _flash_attn_version >= PkgVersion("2.7.0")
    elif (
        torch.cuda.is_available() and get_device_compute_capability() >= (8, 0) and _NVTE_FLASH_ATTN
    ):
        fa_logger.warning(
            "Supported flash-attn versions are %s. Found flash-attn %s.",
            _get_supported_versions(
                (
                    _flash_attn_version_required
                    if get_device_compute_capability() < (10, 0)
                    else _flash_attn_version_required_blackwell
                ),
                _flash_attn_max_version,
            ),
            _flash_attn_version,
        )

# Detect flash-attn v3 in the environment
# This section will be removed when FA3 is released as a regular FA package,
# i.e. flashattn-hopper 3.0.0 as flash-attn 3.0.0
_flash_attn_3_is_installed = False
_flash_attn_3_version = PkgVersion("0")
_flash_attn_3_0_0_beta = False
_use_flash_attn_3 = False
# TODO(cyang): update FA to 2.7.3 when its FA3 compilation issue is resolved
# https://github.com/Dao-AILab/flash-attention/issues/1452
_flash_attn_3_installation_steps = """\
(1) pip install "git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2#egg=flashattn-hopper&subdirectory=hopper"
(2) python_path=`python -c "import site; print(site.getsitepackages()[0])"`
(3) mkdir -p $python_path/flashattn_hopper
(4) wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.7.2/hopper/flash_attn_interface.py"""
try:
    _flash_attn_3_version = PkgVersion(get_pkg_version("flashattn-hopper"))
except PackageNotFoundError:
    if torch.cuda.is_available() and get_device_compute_capability() >= (9, 0) and _NVTE_FLASH_ATTN:
        fa_logger.debug(
            "flash-attn v3 is not installed. To use, please install it by \n%s",
            _flash_attn_3_installation_steps,
        )
else:
    from flashattn_hopper.flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from flashattn_hopper.flash_attn_interface import (
        flash_attn_varlen_func as flash_attn_varlen_func_v3,
    )
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward as _flash_attn_fwd_v3
    from flashattn_hopper.flash_attn_interface import _flash_attn_backward as _flash_attn_bwd_v3
    from flashattn_hopper.flash_attn_interface import (
        _flash_attn_varlen_forward as _flash_attn_varlen_fwd_v3,
    )
    from flashattn_hopper.flash_attn_interface import (
        _flash_attn_varlen_backward as _flash_attn_varlen_bwd_v3,
    )

    _flash_attn_3_is_installed = True
    _flash_attn_3_0_0_beta = PkgVersion("3.0.0b") < _flash_attn_3_version < PkgVersion("3.0.0")
    _use_flash_attn_3 = True

_attention_backends = {
    "attention_params": None,
    "use_flash_attention": None,
    "use_fused_attention": None,
    "fused_attention_backend": None,
    "use_unfused_attention": None,
    "backend_selection_requires_update": False,
}


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


_alibi_cache = {
    "_num_heads": None,
    "_alibi_slopes": None,
    "_max_seqlen_q": None,
    "_max_seqlen_kv": None,
    "_bottom_right_alignment": True,
    "_alibi_bias": None,
    "_alibi_slopes_require_update": False,
    "_alibi_bias_require_update": False,
}


__all__ = ["DotProductAttention", "InferenceParams", "MultiheadAttention"]


def maybe_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Make tensor contiguous if final stride is not 1."""
    return tensor.contiguous() if tensor.stride(-1) != 1 else tensor


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
    logger.setLevel(_log_level)
    if not logger.hasHandlers():
        logger.addHandler(_stream_handler)
    device_compute_capability = get_device_compute_capability()
    cudnn_version = get_cudnn_version()
    run_config = {
        "transformer_engine_version": te.__version__,
        "compute_capability": "sm"
        + str(10 * device_compute_capability[0] + device_compute_capability[1]),
        "flash_attn_version": (
            str(_flash_attn_version) if _flash_attn_is_installed else "not installed"
        ),
        "flash_attn_3_version": (
            str(_flash_attn_3_version) if _flash_attn_3_is_installed else "not installed"
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
    global _flash_attn_version_required, _flash_attn_max_version, _use_flash_attn_3

    # Filter: Environment variables
    use_flash_attention = int(os.getenv("NVTE_FLASH_ATTN", "1"))
    use_fused_attention = int(os.getenv("NVTE_FUSED_ATTN", "1"))
    use_unfused_attention = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))
    if not use_flash_attention and _flash_attn_is_installed:
        logger.debug("Disabling FlashAttention due to NVTE_FLASH_ATTN=0")
    if not use_fused_attention:
        logger.debug("Disabling FusedAttention due to NVTE_FUSED_ATTN=0")
    if not use_unfused_attention:
        logger.debug("Disabling UnfusedDotProductAttention due to NVTE_UNFUSED_ATTN=0")

    # Filter: Compute capability
    if device_compute_capability < (8, 0):
        if use_flash_attention and _flash_attn_is_installed:
            logger.debug("Disabling FlashAttention as it requires compute capability sm80+")
        use_flash_attention = False
        if use_fused_attention:
            logger.debug("Disabling FusedAttention as it requires compute capability sm80+")
            use_fused_attention = False
    if device_compute_capability < (9, 0):
        if use_flash_attention and _flash_attn_3_is_installed:
            logger.debug("Disabling FlashAttention 3 as it requires compute capability sm90+")
        _use_flash_attn_3 = False

    # Filter: Data type
    if qkv_dtype not in [torch.bfloat16, torch.float16] or qkv_type not in [
        torch.Tensor,
        Float8Tensor,
    ]:
        if use_flash_attention and _flash_attn_is_installed:
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
        if use_flash_attention and not _use_flash_attn_3:
            if _flash_attn_is_installed:
                logger.debug("Disabling FlashAttention as FlashAttention 2 does not support FP8")
            use_flash_attention = False
        if use_flash_attention and _use_flash_attn_3 and is_training:
            logger.debug(
                "Disabling FlashAttention as FlashAttention 3 does not support FP8 training"
            )
            use_flash_attention = False
        if use_unfused_attention:
            logger.debug("Disabling UnfusedDotProductAttention as it does not support FP8")
            use_unfused_attention = False

    # Filter: Head dimension
    if use_flash_attention and head_dim_qk != head_dim_v:
        if _flash_attn_is_installed:
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
        if _flash_attn_is_installed:
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
            if _flash_attn_is_installed:
                logger.debug(
                    "Disabling FlashAttention for qkv_format = thd when there is "
                    "padding between sequences, i.e. [a, a, PAD, b, b, b, PAD, c, PAD]"
                )
            use_flash_attention = False

    # Filter: Dropout
    if attention_dropout != 0.0 and use_flash_attention and _use_flash_attn_3:
        logger.debug("Disabling FlashAttention 3 for dropout")
        _use_flash_attn_3 = False

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
            if _flash_attn_is_installed:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with FP8"
                )
            use_flash_attention = False
        if "bottom_right" in attn_mask_type:
            if _flash_attn_is_installed:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with"
                    " causal_bottom_right masking"
                )
            use_flash_attention = False
        elif "causal" in attn_mask_type and max_seqlen_q != max_seqlen_kv:
            if _flash_attn_is_installed:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with"
                    " causal masking for cross-attention"
                )
            use_flash_attention = False
        elif core_attention_bias_type not in ["no_bias", "post_scale_bias"]:
            if _flash_attn_is_installed:
                logger.debug(
                    "Disabling FlashAttention as it does not support context parallelism with bias"
                    " type of %s",
                    core_attention_bias_type,
                )
            use_flash_attention = False
        elif qkv_format == "thd" and core_attention_bias_type != "no_bias":
            if _flash_attn_is_installed:
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
        if use_flash_attention and _flash_attn_is_installed:
            logger.debug("Disabling FlashAttention for arbitrary mask")
        use_flash_attention = False
        if use_fused_attention:
            logger.debug("Disabling FusedAttention for arbitrary mask")
        use_fused_attention = False
    if (
        use_flash_attention
        and _use_flash_attn_3
        and attn_mask_type in ["causal", "padding_causal"]
        and max_seqlen_q != max_seqlen_kv
    ):
        logger.warning(
            "Disabling FlashAttention 3 as it only supports bottom-right-diagonal "
            "causal mask since flash-attn 2.1. See "
            "https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag"
        )
        _use_flash_attn_3 = False
    if (
        use_flash_attention
        and attn_mask_type in ["causal", "padding_causal"]
        and max_seqlen_q != max_seqlen_kv
    ):
        if _flash_attn_2_1_plus:
            logger.warning(
                "Disabling FlashAttention as it only supports bottom-right-diagonal "
                "causal mask since flash-attn 2.1. See "
                "https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag"
            )
            use_flash_attention = False
        if not _flash_attn_is_installed:
            _flash_attn_max_version = PkgVersion("2.1")
    if (
        use_flash_attention
        and attn_mask_type in ["causal_bottom_right", "padding_causal_bottom_right"]
        and max_seqlen_q != max_seqlen_kv
    ):
        if not _flash_attn_is_installed:
            _flash_attn_version_required = PkgVersion("2.1")
        elif not _flash_attn_2_1_plus and not _use_flash_attn_3:
            logger.warning(
                "Disabling FlashAttention as it only supports top-left-diagonal "
                "causal mask before flash-attn 2.1. See "
                "https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag"
            )
            use_flash_attention = False
    if (
        use_flash_attention
        and _use_flash_attn_3
        and fp8
        and fp8_meta["recipe"].fp8_dpa
        and "padding" in attn_mask_type
    ):
        logger.debug("Disabling FlashAttention 3 for FP8 and padding masks")
        _use_flash_attn_3 = False

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
            if _use_flash_attn_3:
                logger.debug(
                    "Disabling FlashAttention 3 as it does not support sliding window attention"
                )
                _use_flash_attn_3 = False
            if not _flash_attn_is_installed:
                _flash_attn_version_required = PkgVersion("2.3")
            elif not _flash_attn_2_3_plus:
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
        if _use_flash_attn_3:
            logger.debug("Disabling FlashAttention 3 for ALiBi")
            _use_flash_attn_3 = False
        if not _flash_attn_is_installed:
            _flash_attn_version_required = PkgVersion("2.4")
        elif not _flash_attn_2_4_plus:
            logger.debug("Disabling FlashAttention as ALiBi requires flash-attn 2.4+")
            use_flash_attention = False

    if use_flash_attention and (
        core_attention_bias_type not in ["no_bias", "alibi"]
        or core_attention_bias_shape is not None
    ):
        if _flash_attn_is_installed:
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
        if not _flash_attn_is_installed:
            _flash_attn_version_required = PkgVersion("2.4.1")
        elif not _flash_attn_2_4_1_plus and not _use_flash_attn_3:
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
    if not use_fused_attention and use_flash_attention and not _flash_attn_is_installed:
        logger.warning(
            "flash-attn may provide important feature support or performance improvement."
            " Please install flash-attn %s.",
            _get_supported_versions(
                _flash_attn_version_required,
                _flash_attn_max_version,
            ),
        )
    if use_flash_attention and not _flash_attn_is_installed:
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
        and _use_flash_attn_3
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

    global _attention_backends
    _attention_backends["use_flash_attention"] = use_flash_attention
    _attention_backends["use_fused_attention"] = use_fused_attention
    _attention_backends["fused_attention_backend"] = fused_attention_backend
    _attention_backends["use_unfused_attention"] = use_unfused_attention
    _attention_backends["backend_selection_requires_update"] = False

    return (
        use_flash_attention,
        use_fused_attention,
        fused_attention_backend,
        use_unfused_attention,
        available_backends,
    )


class InferenceParams:  # pylint: disable=too-few-public-methods
    """
    Inference parameters that are passed to the main model in order
    to efficiently calculate and store the context during inference.

    Parameters
    ----------
    max_batch_size : int
                    maximum batch size during inference.
    max_sequence_length : int
                         maximum sequence length during inference.
    """

    def __init__(self, max_batch_size, max_sequence_length):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_indices):
        """
        Reorders the KV cache using the specified batch indices.

        Parameters
        ----------
        batch_indices : List[int]
                       Sequence of indices to reorder along the batch dimensions of
                       the KV cache. Must have a length equal to the batch size.
        """
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")

        for layer_number, inference_memory in self.key_value_memory_dict.items():
            inference_key_memory, inference_value_memory = inference_memory
            assert (
                len(batch_indices) == inference_key_memory.shape[1]
            )  # make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_indices]
            new_inference_value_memory = inference_value_memory[:, batch_indices]
            self.key_value_memory_dict[layer_number] = (
                new_inference_key_memory,
                new_inference_value_memory,
            )


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
    global _alibi_cache
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


_cu_seqlens_cache = {}


def _get_full_cu_seqlens(
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


@torch.compile
def pack_tensor(
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


@torch.compile
def pack_2_tensors(
    indices: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs the given 2 tensors using the `indices`.
    """
    t1_packed = pack_tensor(indices, t1)
    t2_packed = pack_tensor(indices, t2)
    return t1_packed, t2_packed


@torch.compile
def pack_3_tensors(
    indices: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Packs the given 3 tensors using the `indices`.
    """
    t1_packed = pack_tensor(indices, t1)
    t2_packed = pack_tensor(indices, t2)
    t3_packed = pack_tensor(indices, t3)
    return t1_packed, t2_packed, t3_packed


@torch.compile
def unpack_tensor(
    indices: torch.Tensor,
    dim0: int,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Inverse of `pack_tensor`.
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


@torch.compile
def unpack_2_tensors(
    indices: torch.Tensor,
    dim0: int,
    t1: torch.Tensor,
    t2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of `pack_2_tensors`.
    """
    t1_unpacked = unpack_tensor(indices, dim0, t1)
    t2_unpacked = unpack_tensor(indices, dim0, t2)
    return t1_unpacked, t2_unpacked


@torch.compile
def unpack_3_tensors(
    indices: torch.Tensor,
    dim0: int,
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inverse of `pack_3_tensors`.
    """
    t1_unpacked = unpack_tensor(indices, dim0, t1)
    t2_unpacked = unpack_tensor(indices, dim0, t2)
    t3_unpacked = unpack_tensor(indices, dim0, t3)
    return t1_unpacked, t2_unpacked, t3_unpacked


class PackTensors(torch.autograd.Function):
    """
    Autograd function to pack tensors.
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
            return pack_tensor(indices, *tensors)
        if len(tensors) == 2:
            return pack_2_tensors(indices, *tensors)
        return pack_3_tensors(indices, *tensors)

    @staticmethod
    def backward(ctx, *grad_outputs: Tuple[torch.Tensor, ...]):
        # pylint: disable=missing-function-docstring
        (indices,) = ctx.saved_tensors
        if len(grad_outputs) == 1:
            return None, unpack_tensor(indices, ctx.dim0, *grad_outputs)
        if len(grad_outputs) == 2:
            return None, *unpack_2_tensors(indices, ctx.dim0, *grad_outputs)
        return None, *unpack_3_tensors(indices, ctx.dim0, *grad_outputs)


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
        return unpack_tensor(indices, dim0, tensor)

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        (indices,) = ctx.saved_tensors
        return None, None, pack_tensor(indices, grad_output)


def flash_attn_p2p_communicate(
    rank, send_tensor, send_dst, recv_tensor, recv_src, cp_group, batch_p2p_comm
):
    """Point-to-point communications of KV and dKV in Attention with context parallelism"""
    send_recv_ops = []

    if batch_p2p_comm:
        if rank % 2 == 0:
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, cp_group
            )
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, cp_group
            )
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, cp_group
            )
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, cp_group
            )
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        if rank % 2 == 0:
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = send_recv_ops

    return send_recv_reqs


@jit_fuser
def flash_attn_fwd_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
    movedim_src: int,
    movedim_dst: int,
):
    """Merge partial outputs of each step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse).movedim(
        movedim_src, movedim_dst
    )
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out.add_(out_corrected)


@jit_fuser
def flash_attn_fwd_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge softmax stats of each step in Attention with context parallelism"""
    max_scale = torch.max(softmax_lse, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse, softmax_lse_per_step)
    new_scale = max_scale + torch.log1p(torch.exp(min_scale - max_scale))
    softmax_lse.copy_(new_scale)


@jit_fuser
def get_cu_seqlens_on_cp_rank(
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded_on_cp_rank: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    first_half: bool,
    second_half: bool,
):
    """Compute cu_seqlens of a context parallelism rank"""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    seqlens_padded = (cu_seqlens_padded_on_cp_rank[1:] - cu_seqlens_padded_on_cp_rank[:-1]) // 2
    zeros = torch.zeros_like(seqlens)
    cu_seqlens_on_cp_rank = torch.zeros_like(cu_seqlens)
    if first_half:
        seqlens_1 = seqlens - cp_rank * seqlens_padded
        seqlens_1 = seqlens_1.clamp(zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:].add_(seqlens_1)
    if second_half:
        seqlens_2 = seqlens - (2 * cp_size - cp_rank - 1) * seqlens_padded
        seqlens_2 = seqlens_2.clamp(zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:].add_(seqlens_2)
    cu_seqlens_on_cp_rank.cumsum_(dim=0)
    return cu_seqlens_on_cp_rank


@torch.compile
def get_seq_chunk_ids_for_reordering(cp_size, device, to_contiguous):
    """
    Context parallelism assigns two discontiguous sequence chunks to each GPU for load balancing.
    To make sure tokens are ordered correctly for compute, we need to reorder sequence chunks
    before or after CP communications (e.g., all-gather, all-to-all). This function is to compute
    sequence chunk ids for reordering.
    """
    chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
    if to_contiguous:
        for rank in range(cp_size):
            chunk_ids[rank] = 2 * rank
            chunk_ids[rank + cp_size] = 2 * cp_size - 2 * rank - 1
    else:
        for rank in range(cp_size):
            chunk_ids[2 * rank] = rank
            chunk_ids[2 * rank + 1] = 2 * cp_size - rank - 1
    return chunk_ids


@torch.compile
def reorder_seq_chunks_for_a2a(x, chunk_ids_for_a2a, seq_dim, cp_size, before_attn):
    """Reorder sequence chunk for A2A communication."""
    if before_attn:
        # [cp, b, s, np//cp, hn] -> [b, cp, s, np//cp, hn]
        # or [cp, s, b, np//cp, hn] -> [cp, s, b, np//cp, hn]
        x = x.movedim(0, seq_dim).contiguous()
        # [b, cp, s, np//cp, hn] -> [b, cp*2, s//2, np//cp, hn]
        # or [cp, s, b, np//cp, hn] -> [cp*2, s//2, b, np//cp, hn]
        x = x.view(*x.shape[:seq_dim], cp_size * 2, -1, *x.shape[(seq_dim + 2) :])
        # reorder the sequence chunks
        x = torch.index_select(x, dim=seq_dim, index=chunk_ids_for_a2a)
    else:
        # [b, cp*2, s//2, np//cp, hn] -> [cp*2, b, s//2, np//cp, hn]
        # or [cp*2, s//2, b, np//cp, hn] -> [cp*2, s//2, b, np//cp, hn]
        x = x.movedim(seq_dim, 0).contiguous()
        # reorder the sequence chunks
        x = torch.index_select(x, dim=0, index=chunk_ids_for_a2a)
        # [cp*2, b, s//2, np//cp, hn] -> [cp, 2, b, s//2, np//cp, hn]
        # or [cp*2, s//2, b, np//cp, hn] -> [cp, 2, s//2, b, np//cp, hn]
        x = x.view(cp_size, 2, *x.shape[1:])
    return x


def flash_attn_a2a_communicate(
    a2a_inputs: Union[torch.Tensor, List[torch.Tensor]],
    chunk_ids_for_a2a: torch.Tensor,
    seq_dim: int,
    cp_size: int,
    cp_group: dist_group_type,
    cp_stream: torch.cuda.Stream,
    before_attn: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """A2A communication for context parallelism."""
    a2a_inputs = [a2a_inputs] if not isinstance(a2a_inputs, list) else a2a_inputs
    a2a_outputs, a2a_reqs = [None] * len(a2a_inputs), [None] * len(a2a_inputs)
    if before_attn:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    x = a2a_outputs[i - 2]
                    # reorder the sequence chunks
                    x = reorder_seq_chunks_for_a2a(
                        x, chunk_ids_for_a2a, seq_dim, cp_size, before_attn
                    )
                    # [b, cp*2, s//2, np//cp, hn] -> [b, cp*s, np//cp, hn]
                    # or [cp*2, s//2, b, np//cp, hn] -> [cp*s, b, np//cp, hn]
                    a2a_outputs[i - 2] = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
            if i < len(a2a_inputs):
                x = a2a_inputs[i]
                # [b, s, np, hn] -> [b, s, cp, np//cp, hn]
                # or [s, b, np, hn] -> [s, b, cp, np//cp, hn]
                x = x.view(*x.shape[:-2], cp_size, x.shape[-2] // cp_size, x.shape[-1])
                # [b, s, cp, np//cp, hn] -> [cp, b, s, np//cp, hn]
                # or [s, b, cp, np//cp, hn] -> [cp, s, b, np//cp, hn]
                a2a_inputs[i] = x.movedim(-3, 0).contiguous()
    else:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
            if i < len(a2a_inputs):
                x = a2a_inputs[i]
                # [b, cp*s, np//cp, hn] -> [b, cp*2, s//2, np//cp, hn]
                # or [cp*s, b, np//cp, hn] -> [cp*2, s//2, b, np//cp, hn]
                x = x.view(*x.shape[:seq_dim], cp_size * 2, -1, *x.shape[(seq_dim + 1) :])
                # reorder the sequence chunks
                a2a_inputs[i] = reorder_seq_chunks_for_a2a(
                    x, chunk_ids_for_a2a, seq_dim, cp_size, before_attn
                )
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    x = a2a_outputs[i - 2]
                    # [cp, 2, b, s//2, np//cp, hn] -> [b, 2, s//2, cp, np//cp, hn]
                    # or [cp, 2, s//2, b, np//cp, hn] -> [2, s//2, b, cp, np//cp, hn]
                    x = x.movedim(0, -3).movedim(0, seq_dim).contiguous()
                    # [b, 2, s//2, cp, np//cp, hn] -> [b*s, np, hn]
                    # or [2, s//2, b, cp, np//cp, hn] -> [s*b, np, hn]
                    a2a_outputs[i - 2] = x.view(-1, x.shape[-3] * x.shape[-2], x.shape[-1])
    torch.cuda.current_stream().wait_stream(cp_stream)
    return a2a_outputs[0] if len(a2a_inputs) == 1 else a2a_outputs


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


class AttnFuncWithCPAndKVP2P(torch.autograd.Function):
    """
    Attention implementation with context parallelism. Exchange KV between CP ranks
    with P2P in ring topology. Split attention compute into multiple steps, and overlap
    current-step compute with next-step communication.

    This implementation also supports hierarchical CP, which parallelizes attention
    heads in low-level CP groups and parallelizes sequence dimension in high-level CP
    groups. For more details, please refer to `LongVILA <https://arxiv.org/abs/2408.10188>`_
    and `USP <https://arxiv.org/abs/2405.07719>`_.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        fp8,
        fp8_meta,
        cp_group,
        cp_global_ranks,
        cp_stream,
        quantizers,
    ):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVP2P.forward")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        if isinstance(cp_group, list):
            assert (
                qkv_format != "thd"
            ), f"{qkv_format} format is not supported with hierarchical CP implementation yet!"
            assert attn_bias_type == "no_bias", (
                f"{attn_bias_type} bias type is not supported with hierarchical CP implementation"
                " yet!"
            )
            cp_group_a2a = cp_group[0]
            cp_size_a2a = get_distributed_world_size(cp_group_a2a)
            rank_a2a = get_distributed_rank(cp_group_a2a)
            cp_group = cp_group[1]
        else:
            cp_group_a2a = None
            cp_size_a2a = 1
            rank_a2a = 0

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)
        send_dst = cp_global_ranks[(rank + 1) % cp_size * cp_size_a2a + rank_a2a]
        recv_src = cp_global_ranks[(rank - 1) % cp_size * cp_size_a2a + rank_a2a]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (cp_size == 2)

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type

        seq_dim = None
        if qkv_format in ["bshd", "sbhd"]:
            seq_dim = qkv_format.index("s")
            qkv_layout = qkv_format + "_" + qkv_format[:-2] + "2" + qkv_format[-2:]
        else:
            qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        pad_between_seqs_q = cu_seqlens_q_padded is not None and not torch.equal(
            cu_seqlens_q_padded[:-1], cu_seqlens_q[:-1]
        )
        pad_between_seqs_kv = cu_seqlens_kv_padded is not None and not torch.equal(
            cu_seqlens_kv_padded[:-1], cu_seqlens_kv[:-1]
        )
        max_seqlen_q = max_seqlen_q // cp_size
        max_seqlen_kv = max_seqlen_kv // cp_size
        cu_seqlens_q_padded = (
            None if cu_seqlens_q_padded is None else cu_seqlens_q_padded // cp_size
        )
        cu_seqlens_kv_padded = (
            None if cu_seqlens_kv_padded is None else cu_seqlens_kv_padded // cp_size
        )
        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]

        fused_attn_backend = None
        qkv_dtype = q.dtype
        amax_per_step = None
        S_quantizer_per_step = [None for _ in range(cp_size)]
        O_CP_quantizer_per_step = [None for _ in range(cp_size)]
        # "fp8_mha" decides outputs in fp8, while inputs are inferred from the real dtype
        is_input_fp8 = False
        is_output_fp8 = False

        (
            QKV_quantizer,
            O_quantizer,
            O_CP_quantizer,
            S_quantizer,
            dQKV_quantizer,
            dQKV_CP_quantizer,
            dO_quantizer,
            dP_quantizer,
        ) = get_attention_quantizers(fp8, quantizers, cp_specific_quantizers=True)

        if fp8:
            if use_fused_attention:
                fused_attn_backend = FusedAttnBackend["FP8"]

                assert isinstance(k, q.__class__) and isinstance(
                    v, q.__class__
                ), "q, k, and v must have the same type."
                is_input_fp8 = isinstance(q, Float8Tensor)
                is_output_fp8 = fp8_meta is not None and fp8_meta["recipe"].fp8_mha
                if is_input_fp8:
                    QKV_quantizer = q._quantizer
                    q, k, v = q._data, k._data, v._data
                else:
                    q_f16, k_f16, v_f16 = q, k, v
                    if cp_size_a2a == 1 or int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                        q = QKV_quantizer(q_f16)._data
                    if int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                        k, v = [QKV_quantizer(x)._data for x in [k_f16, v_f16]]
                amax_per_step = torch.zeros((2, cp_size), dtype=torch.float32, device=q.device)
                # partial result quantizer
                for i in range(cp_size):
                    S_quantizer_per_step[i] = S_quantizer.copy()
                    S_quantizer_per_step[i].amax = amax_per_step[0][i]
                    O_CP_quantizer_per_step[i] = O_CP_quantizer.copy()
                    O_CP_quantizer_per_step[i].amax = amax_per_step[1][i]
            else:
                assert False, "FP8 is only supported with Fused Attention!"
        else:
            q_f16 = q
            if use_fused_attention:
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        if cp_size_a2a > 1:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(cp_size_a2a, q.device, True)

            q, k, v = flash_attn_a2a_communicate(
                [q, k, v], chunk_ids_for_a2a, seq_dim, cp_size_a2a, cp_group_a2a, cp_stream, True
            )
            if not fp8:
                q_f16 = q
            elif not is_input_fp8 and not int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                q_f16 = q
                q = QKV_quantizer(q_f16)._data

        assert qkv_format == "thd" or (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"
        if causal:
            if qkv_format == "bshd":
                # [b, s, np, hn] -> [b, 2, s//2, np, hn]
                q, k, v = [x.view(x.shape[0], 2, x.shape[1] // 2, *x.shape[2:]) for x in [q, k, v]]
            elif qkv_format == "sbhd":
                # [s, b, np, hn] -> [2, s//2, b, np, hn]
                q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]
        if attn_bias is not None:
            assert len(attn_bias.shape) == 4, (
                "Only support bias shape of [b, h, sq, sk] for forward, "
                "and [1, h, sq, sk] for backward!"
            )
            assert (
                attn_bias.shape[-2] % 2 == 0 and attn_bias.shape[-1] % (2 * cp_size) == 0
            ), "Sequence length does not meet divisible requirements!"
            # [b, np, sq, sk] -> [b, np, 2, sq//2, 2*cp, sk//(2*cp)]
            attn_bias_ = attn_bias.view(
                *attn_bias.shape[:-2],
                2,
                attn_bias.shape[-2] // 2,
                2 * cp_size,
                attn_bias.shape[-1] // (2 * cp_size),
            )
            # [b, np, sq, sk] -> [b, np, sq, 2*cp, sk//(2*cp)]
            attn_bias = attn_bias.view(
                *attn_bias.shape[:-1], 2 * cp_size, attn_bias.shape[-1] // (2 * cp_size)
            )
        assert q.shape[-1] % 8 == 0, "hidden size per attention head should be multiple of 8"

        softmax_lse_in_packed_format = False
        if qkv_format == "thd":
            if use_fused_attention:
                softmax_lse_in_packed_format = get_cudnn_version() >= (9, 6, 0)
            else:
                softmax_lse_in_packed_format = _flash_attn_2_6_0_plus or _use_flash_attn_3

        flash_attn_fwd = None
        if not use_fused_attention:
            fa_forward_kwargs = {"softmax_scale": softmax_scale}
            if _use_flash_attn_3:
                if qkv_format == "thd":
                    flash_attn_fwd = _flash_attn_varlen_fwd_v3
                else:
                    flash_attn_fwd = _flash_attn_fwd_v3
                fa_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)
            else:
                if qkv_format == "thd":
                    flash_attn_fwd = _flash_attn_varlen_fwd
                else:
                    flash_attn_fwd = _flash_attn_fwd
                fa_forward_kwargs["dropout_p"] = dropout_p
                fa_forward_kwargs["return_softmax"] = False
                if (_flash_attn_2_3_plus and not _flash_attn_2_7_0_plus) or _use_flash_attn_3:
                    fa_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)
                elif _flash_attn_2_7_0_plus:
                    fa_forward_kwargs["window_size_left"] = -1
                    fa_forward_kwargs["window_size_right"] = 0 if causal else -1
                if _flash_attn_2_4_plus:
                    fa_forward_kwargs["alibi_slopes"] = None
                if _flash_attn_2_5_7_plus and qkv_format == "thd":
                    fa_forward_kwargs["block_table"] = None
                if _flash_attn_2_6_0_plus:
                    fa_forward_kwargs["softcap"] = 0.0

        # Flash Attn inputs
        q_inputs = [None, None]
        kv_inputs = [None, None]
        attn_bias_inputs = [None, None]
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        rng_states = [None for _ in range(cp_size)]
        attn_biases = [None for _ in range(cp_size)]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()

        p2p_comm_buffers = [None for _ in range(cp_size)]
        if qkv_format in ["bshd", "sbhd"]:
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(-3), v.unsqueeze(-3)), dim=-3)
        else:
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]

        softmax_lse_ = None
        out = None
        for i in range(cp_size + 1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i % 2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i + 1) % 2]:
                        req.wait()

                    if i < (cp_size - 1):
                        p2p_comm_buffers[i + 1] = torch.empty_like(p2p_comm_buffers[i])
                        send_recv_reqs[i % 2] = flash_attn_p2p_communicate(
                            rank,
                            p2p_comm_buffers[i],
                            send_dst,
                            p2p_comm_buffers[i + 1],
                            recv_src,
                            cp_group,
                            batch_p2p_comm,
                        )

                    if not fp8 or is_input_fp8 or int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                        kv_inputs[i % 2] = p2p_comm_buffers[i]
                    else:
                        # KV exchange is in BF16/FP16, cast received KV in each step
                        kv_inputs[i % 2] = QKV_quantizer(p2p_comm_buffers[i])._data
                    if causal:
                        if i == 0:
                            if pad_between_seqs_q:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, True, True
                                )
                            elif use_fused_attention or qkv_format == "thd":
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                            if pad_between_seqs_kv:
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv, cu_seqlens_kv_padded, cp_size, rank, True, True
                                )
                            elif use_fused_attention or qkv_format == "thd":
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size
                            if qkv_format == "bshd":
                                # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                                q_inputs[i % 2] = q.view(q.shape[0], -1, *q.shape[-2:])
                                # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                    k.shape[0], -1, 2, *k.shape[-2:]
                                )
                            elif qkv_format == "sbhd":
                                # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                                q_inputs[i % 2] = q.view(-1, *q.shape[-3:])
                                # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                    -1, k.shape[2], 2, *k.shape[-2:]
                                )
                            elif qkv_format == "thd":
                                q_inputs[i % 2] = q
                            if use_fused_attention:
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i % 2] = torch.cat(
                                        (
                                            attn_bias[..., idx, :],
                                            attn_bias[..., (2 * cp_size - idx - 1), :],
                                        ),
                                        dim=-1,
                                    ).contiguous()

                                q_part = q_inputs[i % 2]
                                k_part = (
                                    kv_inputs[i % 2][..., 0, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][0]
                                )
                                v_part = (
                                    kv_inputs[i % 2][..., 1, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][1]
                                )
                                fp8_meta_kwargs = {}
                                if fp8:
                                    q_part = QKV_quantizer.create_tensor_from_data(
                                        q_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    k_part = QKV_quantizer.create_tensor_from_data(
                                        k_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    v_part = QKV_quantizer.create_tensor_from_data(
                                        v_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    fp8_meta_kwargs["s_quantizer"] = S_quantizer_per_step[i]
                                    fp8_meta_kwargs["o_quantizer"] = O_CP_quantizer_per_step[i]

                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q,
                                    max_seqlen_kv,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_part,
                                    k_part,
                                    v_part,
                                    fake_dtype=qkv_dtype,
                                    fused_attention_backend=fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type=attn_mask_type,
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias_inputs[i % 2],
                                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                    **fp8_meta_kwargs,
                                )
                                if fp8:
                                    softmax_lse_per_step[i], _, rng_states[i] = aux_ctx_tensors
                                else:
                                    softmax_lse_per_step[i], rng_states[i], *rest = aux_ctx_tensors
                                    attn_biases[i] = rest[0] if len(rest) > 0 else None
                            else:
                                fa_forward_args_thd = []
                                if qkv_format == "thd":
                                    fa_forward_args_thd = [
                                        cu_seqlens_q_per_step[i],
                                        cu_seqlens_kv_per_step[i],
                                        max_seqlen_q,
                                        max_seqlen_kv,
                                    ]
                                fa_outputs = flash_attn_fwd(
                                    q_inputs[i % 2],
                                    (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    ),
                                    (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    ),
                                    *fa_forward_args_thd,
                                    causal=True,
                                    **fa_forward_kwargs,
                                )
                                if not _flash_attn_2_7_0_plus:
                                    out_per_step[i] = fa_outputs[4]
                                    softmax_lse_per_step[i] = fa_outputs[5]
                                    if not _use_flash_attn_3:
                                        rng_states[i] = fa_outputs[7]
                                else:
                                    out_per_step[i] = fa_outputs[0]
                                    softmax_lse_per_step[i] = fa_outputs[1]
                                    if not _use_flash_attn_3:
                                        rng_states[i] = fa_outputs[3]
                        elif i <= rank:
                            if pad_between_seqs_q:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, True, True
                                )
                            elif use_fused_attention or qkv_format == "thd":
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                            if pad_between_seqs_kv:
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv,
                                    cu_seqlens_kv_padded,
                                    cp_size,
                                    (rank - i) % cp_size,
                                    True,
                                    False,
                                )
                            elif use_fused_attention or qkv_format == "thd":
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // (cp_size * 2)
                            if qkv_format == "bshd":
                                # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                                q_inputs[i % 2] = q.view(q.shape[0], -1, *q.shape[-2:])
                                # [b, 2, sk//2, 2, np, hn] -> [b, sk//2, 2, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2][:, 0, ...]
                            elif qkv_format == "sbhd":
                                # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                                q_inputs[i % 2] = q.view(-1, *q.shape[-3:])
                                # [2, sk//2, b, 2, np, hn] -> [sk//2, b, 2, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2][0]
                            elif qkv_format == "thd":
                                q_inputs[i % 2] = q
                                # [2, t, np, hn] -> [2, t/2, np, hn]
                                kv_inputs[i % 2] = tex.thd_read_half_tensor(
                                    kv_inputs[i % 2], cu_seqlens_kv_padded, 0
                                )
                            if use_fused_attention:
                                kv_inputs[i % 2] = kv_inputs[i % 2].contiguous()
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i % 2] = attn_bias[..., idx, :].contiguous()

                                q_part = q_inputs[i % 2]
                                k_part = (
                                    kv_inputs[i % 2][..., 0, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][0]
                                )
                                v_part = (
                                    kv_inputs[i % 2][..., 1, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][1]
                                )
                                fp8_meta_kwargs = {}
                                if fp8:
                                    q_part = QKV_quantizer.create_tensor_from_data(
                                        q_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    k_part = QKV_quantizer.create_tensor_from_data(
                                        k_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    v_part = QKV_quantizer.create_tensor_from_data(
                                        v_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    fp8_meta_kwargs["s_quantizer"] = S_quantizer_per_step[i]
                                    fp8_meta_kwargs["o_quantizer"] = O_CP_quantizer_per_step[i]
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q,
                                    max_seqlen_kv // 2,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_part,
                                    k_part,
                                    v_part,
                                    qkv_dtype,
                                    fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type="padding" if padding else "no_mask",
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias_inputs[i % 2],
                                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                                    cu_seqlens_kv_padded=(
                                        None
                                        if cu_seqlens_kv_padded is None
                                        else cu_seqlens_kv_padded // 2
                                    ),
                                    **fp8_meta_kwargs,
                                )
                                if fp8:
                                    softmax_lse_per_step[i], _, rng_states[i] = aux_ctx_tensors
                                else:
                                    softmax_lse_per_step[i], rng_states[i], *rest = aux_ctx_tensors
                                    attn_biases[i] = rest[0] if len(rest) > 0 else None
                            else:
                                fa_forward_args_thd = []
                                if qkv_format == "thd":
                                    fa_forward_args_thd = [
                                        cu_seqlens_q_per_step[i],
                                        cu_seqlens_kv_per_step[i],
                                        max_seqlen_q,
                                        max_seqlen_kv // 2,
                                    ]
                                if _use_flash_attn_3 or (
                                    _flash_attn_2_3_plus and not _flash_attn_2_7_0_plus
                                ):
                                    fa_forward_kwargs["window_size"] = (-1, -1)
                                elif _flash_attn_2_7_0_plus:
                                    fa_forward_kwargs["window_size_left"] = -1
                                    fa_forward_kwargs["window_size_right"] = -1
                                fa_outputs = flash_attn_fwd(
                                    q_inputs[i % 2],
                                    (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    ),
                                    (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    ),
                                    *fa_forward_args_thd,
                                    causal=False,
                                    **fa_forward_kwargs,
                                )
                                if not _flash_attn_2_7_0_plus:
                                    out_per_step[i] = fa_outputs[4]
                                    softmax_lse_per_step[i] = fa_outputs[5]
                                    if not _use_flash_attn_3:
                                        rng_states[i] = fa_outputs[7]
                                else:
                                    out_per_step[i] = fa_outputs[0]
                                    softmax_lse_per_step[i] = fa_outputs[1]
                                    if not _use_flash_attn_3:
                                        rng_states[i] = fa_outputs[3]
                        else:
                            if pad_between_seqs_q:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, False, True
                                )
                            elif use_fused_attention or qkv_format == "thd":
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // (cp_size * 2)
                            if pad_between_seqs_kv:
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv,
                                    cu_seqlens_kv_padded,
                                    cp_size,
                                    (rank - i) % cp_size,
                                    True,
                                    True,
                                )
                            elif use_fused_attention or qkv_format == "thd":
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size
                            if qkv_format == "bshd":
                                # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                                q_inputs[i % 2] = q[:, 1, ...]
                                # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                    k.shape[0], -1, 2, *k.shape[-2:]
                                )
                            elif qkv_format == "sbhd":
                                # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                                q_inputs[i % 2] = q[1]
                                # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                    -1, k.shape[2], 2, *k.shape[-2:]
                                )
                            elif qkv_format == "thd":
                                # [t, np, hn] -> [t/2, np, hn]
                                q_inputs[i % 2] = tex.thd_read_half_tensor(
                                    q, cu_seqlens_q_padded, 1
                                )
                            if use_fused_attention:
                                q_inputs[i % 2] = q_inputs[i % 2].contiguous()
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i % 2] = torch.cat(
                                        (
                                            attn_bias_[..., 1, :, idx, :],
                                            attn_bias_[..., 1, :, (2 * cp_size - idx - 1), :],
                                        ),
                                        dim=-1,
                                    ).contiguous()

                                q_part = q_inputs[i % 2]
                                k_part = (
                                    kv_inputs[i % 2][..., 0, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][0]
                                )
                                v_part = (
                                    kv_inputs[i % 2][..., 1, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][1]
                                )
                                fp8_meta_kwargs = {}
                                if fp8:
                                    q_part = QKV_quantizer.create_tensor_from_data(
                                        q_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    k_part = QKV_quantizer.create_tensor_from_data(
                                        k_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    v_part = QKV_quantizer.create_tensor_from_data(
                                        v_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    fp8_meta_kwargs["s_quantizer"] = S_quantizer_per_step[i]
                                    fp8_meta_kwargs["o_quantizer"] = O_CP_quantizer_per_step[i]
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q // 2,
                                    max_seqlen_kv,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_part,
                                    k_part,
                                    v_part,
                                    qkv_dtype,
                                    fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type="padding" if padding else "no_mask",
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias_inputs[i % 2],
                                    cu_seqlens_q_padded=(
                                        None
                                        if cu_seqlens_q_padded is None
                                        else cu_seqlens_q_padded // 2
                                    ),
                                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                    **fp8_meta_kwargs,
                                )
                                if fp8:
                                    softmax_lse_per_step[i], _, rng_states[i] = aux_ctx_tensors
                                else:
                                    softmax_lse_per_step[i], rng_states[i], *rest = aux_ctx_tensors
                                    attn_biases[i] = rest[0] if len(rest) > 0 else None
                            else:
                                fa_forward_args_thd = []
                                if qkv_format == "thd":
                                    fa_forward_args_thd = [
                                        cu_seqlens_q_per_step[i],
                                        cu_seqlens_kv_per_step[i],
                                        max_seqlen_q // 2,
                                        max_seqlen_kv,
                                    ]
                                if _use_flash_attn_3 or (
                                    _flash_attn_2_3_plus and not _flash_attn_2_7_0_plus
                                ):
                                    fa_forward_kwargs["window_size"] = (-1, -1)
                                elif _flash_attn_2_7_0_plus:
                                    fa_forward_kwargs["window_size_left"] = -1
                                    fa_forward_kwargs["window_size_right"] = -1
                                fa_outputs = flash_attn_fwd(
                                    q_inputs[i % 2],
                                    (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    ),
                                    (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    ),
                                    *fa_forward_args_thd,
                                    causal=False,
                                    **fa_forward_kwargs,
                                )
                                if not _flash_attn_2_7_0_plus:
                                    out_per_step[i] = fa_outputs[4]
                                    softmax_lse_per_step[i] = fa_outputs[5]
                                    if not _use_flash_attn_3:
                                        rng_states[i] = fa_outputs[7]
                                else:
                                    out_per_step[i] = fa_outputs[0]
                                    softmax_lse_per_step[i] = fa_outputs[1]
                                    if not _use_flash_attn_3:
                                        rng_states[i] = fa_outputs[3]
                    else:
                        if pad_between_seqs_q:
                            cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                cu_seqlens_q, cu_seqlens_q_padded, cp_size, rank, True, True
                            )
                        elif use_fused_attention or qkv_format == "thd":
                            cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                        if pad_between_seqs_kv:
                            cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                cu_seqlens_kv,
                                cu_seqlens_kv_padded,
                                cp_size,
                                (rank - i) % cp_size,
                                True,
                                True,
                            )
                        elif use_fused_attention or qkv_format == "thd":
                            cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size
                        if use_fused_attention:
                            if attn_bias is not None:
                                idx = (rank - i) % cp_size
                                attn_bias_inputs[i % 2] = torch.cat(
                                    (
                                        attn_bias[..., idx, :],
                                        attn_bias[..., (2 * cp_size - idx - 1), :],
                                    ),
                                    dim=-1,
                                ).contiguous()

                            q_part = q
                            k_part = (
                                kv_inputs[i % 2][..., 0, :, :]
                                if qkv_format in ["bshd", "sbhd"]
                                else kv_inputs[i % 2][0]
                            )
                            v_part = (
                                kv_inputs[i % 2][..., 1, :, :]
                                if qkv_format in ["bshd", "sbhd"]
                                else kv_inputs[i % 2][1]
                            )
                            fp8_meta_kwargs = {}
                            if fp8:
                                q_part = QKV_quantizer.create_tensor_from_data(
                                    q_part, fake_dtype=qkv_dtype, internal=True
                                )
                                k_part = QKV_quantizer.create_tensor_from_data(
                                    k_part, fake_dtype=qkv_dtype, internal=True
                                )
                                v_part = QKV_quantizer.create_tensor_from_data(
                                    v_part, fake_dtype=qkv_dtype, internal=True
                                )
                                fp8_meta_kwargs["s_quantizer"] = S_quantizer_per_step[i]
                                fp8_meta_kwargs["o_quantizer"] = O_CP_quantizer_per_step[i]
                            out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                is_training,
                                max_seqlen_q,
                                max_seqlen_kv,
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_kv_per_step[i],
                                q_part,
                                k_part,
                                v_part,
                                qkv_dtype,
                                fused_attn_backend,
                                attn_scale=softmax_scale,
                                dropout=dropout_p,
                                qkv_layout=qkv_layout,
                                attn_mask_type=attn_mask_type,
                                attn_bias_type=attn_bias_type,
                                attn_bias=attn_bias_inputs[i % 2],
                                cu_seqlens_q_padded=cu_seqlens_q_padded,
                                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                **fp8_meta_kwargs,
                            )
                            if fp8:
                                softmax_lse_per_step[i], _, rng_states[i] = aux_ctx_tensors
                            else:
                                softmax_lse_per_step[i], rng_states[i], *rest = aux_ctx_tensors
                                attn_biases[i] = rest[0] if len(rest) > 0 else None
                        else:
                            fa_forward_args_thd = []
                            if qkv_format == "thd":
                                fa_forward_args_thd = [
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    max_seqlen_q,
                                    max_seqlen_kv,
                                ]
                            fa_outputs = flash_attn_fwd(
                                q,
                                (
                                    kv_inputs[i % 2][..., 0, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][0]
                                ),
                                (
                                    kv_inputs[i % 2][..., 1, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][1]
                                ),
                                *fa_forward_args_thd,
                                causal=False,
                                **fa_forward_kwargs,
                            )
                            if not _flash_attn_2_7_0_plus:
                                out_per_step[i] = fa_outputs[4]
                                softmax_lse_per_step[i] = fa_outputs[5]
                                if not _use_flash_attn_3:
                                    rng_states[i] = fa_outputs[7]
                            else:
                                out_per_step[i] = fa_outputs[0]
                                softmax_lse_per_step[i] = fa_outputs[1]
                                if not _use_flash_attn_3:
                                    rng_states[i] = fa_outputs[3]

            if i > 0:
                # wait until fwd restuls correction of last step is done
                if i > 1:
                    flash_attn_streams[(i - 1) % 2].wait_event(fwd_results_correction_done)

                if use_fused_attention:
                    # [b, np, sq, 1] -> [b, np, sq] or
                    # [t, np, 1] -> [t, np]
                    softmax_lse_per_step[i - 1].squeeze_(-1)
                    if softmax_lse_in_packed_format:
                        softmax_lse_per_step[i - 1] = (
                            softmax_lse_per_step[i - 1].transpose(0, 1).contiguous()
                        )

                with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                    if fp8:
                        out_per_step[i - 1] = out_per_step[i - 1].dequantize(dtype=torch.float32)
                    if i == 1:
                        out = torch.zeros_like(q if not fp8 else out_per_step[0]).view(q.shape)
                        softmax_lse = torch.clone(softmax_lse_per_step[0]).to(torch.double)
                        if causal and qkv_format != "thd":
                            # [b, np, sq] -> [b, np, 2, sq//2]
                            softmax_lse_ = softmax_lse.view(
                                *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1] // 2
                            )
                    elif (i - 1) <= rank or not causal:
                        flash_attn_fwd_softmax_lse_correction(
                            softmax_lse, softmax_lse_per_step[i - 1]
                        )
                    else:
                        if qkv_format == "thd":
                            tex.thd_second_half_lse_correction(
                                softmax_lse,
                                softmax_lse_per_step[i - 1],
                                cu_seqlens_q_padded,
                                softmax_lse_in_packed_format,
                            )
                        else:
                            flash_attn_fwd_softmax_lse_correction(
                                softmax_lse_[..., 1, :], softmax_lse_per_step[i - 1]
                            )

                if i < cp_size:
                    flash_attn_streams[(i - 1) % 2].record_event(fwd_results_correction_done)

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        second_half_lse_seqlen = None
        if causal and rank < (cp_size - 1):
            second_half_lse_seqlen = softmax_lse_per_step[-1].shape[-1]

        softmax_lse = softmax_lse.to(torch.float)
        for i in range(cp_size):
            if i <= rank or not causal:
                if qkv_format in ["bshd", "sbhd"]:
                    flash_attn_fwd_out_correction(
                        out.view(*out_per_step[i].shape),
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        0 if softmax_lse_in_packed_format else 2,
                        2 if softmax_lse_in_packed_format else seq_dim,
                    )
                elif qkv_format == "thd":
                    tex.thd_out_correction(
                        out,
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        cu_seqlens_q_padded,
                        False,
                        softmax_lse_in_packed_format,
                    )
            else:
                if qkv_format in ["bshd", "sbhd"]:
                    out_ = out.select(seq_dim, 1)
                    flash_attn_fwd_out_correction(
                        out_,
                        out_per_step[i],
                        softmax_lse_[..., 1, :],
                        softmax_lse_per_step[i],
                        0 if softmax_lse_in_packed_format else 2,
                        2 if softmax_lse_in_packed_format else seq_dim,
                    )
                elif qkv_format == "thd":
                    tex.thd_out_correction(
                        out,
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        cu_seqlens_q_padded,
                        True,
                        softmax_lse_in_packed_format,
                    )

        kv = p2p_comm_buffers[-1]
        if qkv_format == "bshd":
            out = out.view(out.shape[0], -1, *out.shape[-2:])
            ctx.batch_size = out.shape[0]
        elif qkv_format == "sbhd":
            out = out.view(-1, *out.shape[-3:])
            ctx.batch_size = out.shape[1]

        if cp_size_a2a > 1:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(cp_size_a2a, out.device, False)
            out = flash_attn_a2a_communicate(
                out, chunk_ids_for_a2a, seq_dim, cp_size_a2a, cp_group_a2a, cp_stream, False
            )
            if use_fused_attention:
                if qkv_format == "bshd":
                    # [b*s, np, hn] -> [b, s, np, hn]
                    out = out.view(ctx.batch_size, -1, *out.shape[-2:])
                elif qkv_format == "sbhd":
                    # [s*b, np, hn] -> [s, b, np, hn]
                    out = out.view(-1, ctx.batch_size, *out.shape[-2:])
        elif not use_fused_attention:
            out = out.view(-1, *out.shape[-2:])

        if fp8 and use_fused_attention:
            amax_cp_fwd = amax_per_step.amax(dim=1)
            S_quantizer.amax = amax_cp_fwd[0]
            O_CP_quantizer.amax = amax_cp_fwd[1]

        out_fp8 = None
        out_f16 = out.to(qkv_dtype)

        if fp8 and (is_output_fp8 or int(os.getenv("NVTE_FP8_DPA_BWD", "1"))):
            out_fp8 = O_quantizer(out_f16)  # final result

        out_ret = out_fp8 if (fp8 and is_output_fp8) else out_f16

        if fp8 and int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
            q_save, kv_save, out_save = q, kv, out_fp8._data
        elif fp8 and is_input_fp8:
            q_save, kv_save, out_save = q, kv, out_f16
        else:
            q_f16 = q_f16.view(q.shape)
            q_save, kv_save, out_save = q_f16, kv, out_f16

        tensors_to_save, tensor_objects = prepare_for_saving(
            q_save,
            kv_save,
            out_save,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *cu_seqlens_q_per_step,
            *cu_seqlens_kv_per_step,
            *rng_states,
            *attn_biases,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects

        ctx.qkv_dtype = qkv_dtype
        ctx.QKV_quantizer = QKV_quantizer
        ctx.O_quantizer = O_quantizer
        ctx.O_CP_quantizer = O_CP_quantizer
        ctx.S_quantizer = S_quantizer
        ctx.dQKV_quantizer = dQKV_quantizer
        ctx.dQKV_CP_quantizer = dQKV_CP_quantizer
        ctx.dO_quantizer = dO_quantizer
        ctx.dP_quantizer = dP_quantizer

        ctx.cp_group_a2a = cp_group_a2a
        ctx.cp_size_a2a = cp_size_a2a
        ctx.rank_a2a = rank_a2a
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_bias_shape = None if attn_bias is None else attn_bias.shape
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        ctx.softmax_lse_in_packed_format = softmax_lse_in_packed_format
        ctx.second_half_lse_seqlen = second_half_lse_seqlen
        ctx.fp8 = fp8 and int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
        ctx.fp8_meta = fp8_meta
        ctx.is_input_fp8 = is_input_fp8
        ctx.is_output_fp8 = is_output_fp8
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVP2P.forward")

        return out_ret

    @staticmethod
    def backward(ctx, dout):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVP2P.backward")
        cp_size_a2a = ctx.cp_size_a2a
        rank_a2a = ctx.rank_a2a

        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)
        send_dst = ctx.cp_global_ranks[(rank - 1) % cp_size * cp_size_a2a + rank_a2a]
        recv_src = ctx.cp_global_ranks[(rank + 1) % cp_size * cp_size_a2a + rank_a2a]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (cp_size == 2)

        q, kv, out, softmax_lse, cu_seqlens_q_padded, cu_seqlens_kv_padded, *other_tensors = (
            restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
        )
        cu_seqlens_q_per_step = other_tensors[:cp_size]
        cu_seqlens_kv_per_step = other_tensors[cp_size : cp_size * 2]
        rng_states = other_tensors[cp_size * 2 : cp_size * 3]
        attn_biases = other_tensors[cp_size * 3 : cp_size * 4]

        causal = "causal" in ctx.attn_mask_type
        padding = "padding" in ctx.attn_mask_type

        seq_dim = None
        if ctx.qkv_format in ["bshd", "sbhd"]:
            seq_dim = ctx.qkv_format.index("s")
            qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format[:-2] + "2" + ctx.qkv_format[-2:]
        else:
            qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        if attn_biases[0] is not None:
            # [b, np, sq, 2*cp, sk//(2*cp)]
            attn_dbias = torch.zeros(
                *ctx.attn_bias_shape, dtype=attn_biases[0].dtype, device=attn_biases[0].device
            )
            # [b, np, sq, 2*cp, sk//(2*cp)] -> [b, np, 2, sq//2, 2*cp, sk//(2*cp)]
            attn_dbias_ = attn_dbias.view(
                *attn_dbias.shape[:-3], 2, attn_dbias.shape[-3] // 2, *attn_dbias.shape[-2:]
            )
        else:
            attn_dbias = None
            attn_dbias_ = None

        softmax_lse_ = None
        if causal and ctx.second_half_lse_seqlen is not None:
            if ctx.qkv_format == "thd":
                softmax_lse_ = tex.thd_read_second_half_lse(
                    softmax_lse,
                    cu_seqlens_q_padded,
                    ctx.softmax_lse_in_packed_format,
                    ctx.second_half_lse_seqlen,
                )
            else:
                # [b, np, sq] -> [b, np, 2, sq//2]
                softmax_lse_ = softmax_lse.view(
                    *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1] // 2
                )
                softmax_lse_ = softmax_lse_[..., 1, :].contiguous()
            if ctx.use_fused_attention:
                if ctx.softmax_lse_in_packed_format:
                    softmax_lse_ = softmax_lse_.transpose(0, 1).contiguous()
                # [b, np, sq//2] -> [b, np, sq//2, 1] or
                # [t//2, np] -> [t//2, np, 1]
                softmax_lse_.unsqueeze_(-1)
        if ctx.use_fused_attention:
            if ctx.softmax_lse_in_packed_format:
                softmax_lse = softmax_lse.transpose(0, 1).contiguous()
            # [b, np, sq] -> [b, np, sq, 1] or
            # [t, np] -> [t, np, 1]
            softmax_lse.unsqueeze_(-1)

        dq = None
        dout_dtype = dout.dtype
        fused_attn_backend = None
        fused_attn_dqkv_dtype = None
        amax_per_step = None
        dP_quantizer_per_step = [None for _ in range(cp_size)]
        dQKV_CP_quantizer_per_step = [None for _ in range(cp_size)]
        if ctx.fp8:
            if ctx.use_fused_attention:
                fused_attn_backend = FusedAttnBackend["FP8"]

                dqkv_fp8_torch_dtype = get_fp8_torch_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=False
                )
                dq_fp8 = torch.empty(
                    (cp_size, *q.shape), dtype=dqkv_fp8_torch_dtype, device=q.device
                )
                dkv_fp8 = torch.empty(
                    (cp_size, *kv.shape), dtype=dqkv_fp8_torch_dtype, device=kv.device
                )
                dkv_fp8_ = torch.empty_like(dkv_fp8)
                if ctx.is_output_fp8:
                    assert isinstance(dout, Float8Tensor), "dout must be Float8Tensors for FP8 MHA!"
                    ctx.dO_quantizer = dout._quantizer
                else:
                    dout = ctx.dO_quantizer(dout)
                fused_attn_dqkv_dtype = dout._fp8_dtype
                dout = dout._data
                p2p_comm_buffers = [[kv, dkv_fp8], [torch.empty_like(kv), dkv_fp8_]]
                fp8_meta_kwargs = {}
                fp8_meta_kwargs["s_quantizer"] = ctx.S_quantizer
                amax_per_step = torch.zeros((2, cp_size), dtype=torch.float32, device=q.device)
                for i in range(cp_size):
                    dP_quantizer_per_step[i] = ctx.dP_quantizer.copy()
                    dP_quantizer_per_step[i].amax = amax_per_step[0][i]
                    dQKV_CP_quantizer_per_step[i] = ctx.dQKV_CP_quantizer.copy()
                    dQKV_CP_quantizer_per_step[i].amax = amax_per_step[1][i]
            else:
                assert False, "FP8 is only supported with Fused Attention!"
        else:
            if ctx.fp8_meta is not None:
                if ctx.is_input_fp8:
                    q = ctx.QKV_quantizer.create_tensor_from_data(
                        q, fake_dtype=ctx.qkv_dtype, internal=True
                    )
                    kv = ctx.QKV_quantizer.create_tensor_from_data(
                        kv, fake_dtype=ctx.qkv_dtype, internal=True
                    )
                    q = q.dequantize(dtype=ctx.qkv_dtype)
                    kv = kv.dequantize(dtype=ctx.qkv_dtype)
                if ctx.is_output_fp8:
                    assert isinstance(dout, Float8Tensor), "dout must be Float8Tensors for FP8 MHA!"
                    if cp_size_a2a == 1:
                        dout = dout.dequantize(dtype=dout_dtype)
                    else:
                        ctx.dO_quantizer = dout._quantizer
                        dout = dout._data
            dq = torch.empty_like(q)
            p2p_comm_buffers = [
                torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
                torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
            ]
            p2p_comm_buffers[0][0].copy_(kv)
            if ctx.use_fused_attention:
                fp8_meta_kwargs = {}
                fused_attn_dqkv_dtype = TE_DType[dout_dtype]
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        if cp_size_a2a > 1:
            if not ctx.use_fused_attention:
                out = out.view(ctx.batch_size, -1, *out.shape[-2:])
                dout = dout.view(*out.shape)
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(cp_size_a2a, out.device, True)
            out, dout = flash_attn_a2a_communicate(
                [out, dout],
                chunk_ids_for_a2a,
                seq_dim,
                cp_size_a2a,
                ctx.cp_group_a2a,
                ctx.cp_stream,
                True,
            )
            if not ctx.fp8 and ctx.fp8_meta is not None and ctx.is_output_fp8:
                dout = ctx.dO_quantizer.create_tensor_from_data(
                    dout, fake_dtype=dout_dtype, internal=True
                )
                dout = dout.dequantize(dtype=dout_dtype)

        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        send_recv_reqs = []

        flash_attn_bwd = None
        if not ctx.use_fused_attention:
            fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
            if _use_flash_attn_3:
                if ctx.qkv_format == "thd":
                    flash_attn_bwd = _flash_attn_varlen_bwd_v3
                else:
                    flash_attn_bwd = _flash_attn_bwd_v3
                fa_backward_kwargs["deterministic"] = ctx.deterministic
            else:
                if ctx.qkv_format == "thd":
                    flash_attn_bwd = _flash_attn_varlen_bwd
                else:
                    flash_attn_bwd = _flash_attn_bwd
                fa_backward_kwargs["dropout_p"] = ctx.dropout_p
                if _flash_attn_2_4_plus:
                    fa_backward_kwargs["alibi_slopes"] = None
                if _flash_attn_2_4_1_plus:
                    fa_backward_kwargs["deterministic"] = ctx.deterministic
                if _flash_attn_2_6_0_plus:
                    fa_backward_kwargs["softcap"] = 0.0

        for i in range(cp_size):
            # wait until KV is received
            for req in send_recv_reqs:
                req.wait()

            send_tensor = p2p_comm_buffers[i % 2]
            recv_tensor = p2p_comm_buffers[(i + 1) % 2]
            if ctx.fp8:
                if i < cp_size - 1:
                    send_recv_reqs = flash_attn_p2p_communicate(
                        rank,
                        send_tensor[0],
                        send_dst,
                        recv_tensor[0],
                        recv_src,
                        ctx.cp_group,
                        batch_p2p_comm,
                    )
                else:
                    dkv_a2a_req = torch.distributed.all_to_all_single(
                        dkv_fp8,
                        dkv_fp8_,
                        group=ctx.cp_group,
                        async_op=True,
                    )
                    send_recv_reqs = [dkv_a2a_req]
            else:
                if i == 0:
                    send_tensor = send_tensor[0]
                    recv_tensor = recv_tensor[0]
                if i == (cp_size - 1):
                    send_tensor = send_tensor[1]
                    recv_tensor = recv_tensor[1]
                send_recv_reqs = flash_attn_p2p_communicate(
                    rank, send_tensor, send_dst, recv_tensor, recv_src, ctx.cp_group, batch_p2p_comm
                )

            kv = p2p_comm_buffers[i % 2][0]
            q_, kv_, out_, dout_ = None, None, None, None
            dq_, dk_, dv_ = None, None, None
            # In reversed order of fwd
            if causal:
                if i == (cp_size - 1):
                    if ctx.qkv_format == "bshd":
                        # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                        q_, out_, dout_ = [
                            x.view(x.shape[0], -1, *x.shape[-2:]) for x in [q, out, dout]
                        ]
                        # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                        kv_ = kv.view(kv.shape[0], -1, *kv.shape[-3:])
                    elif ctx.qkv_format == "sbhd":
                        # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                        q_, out_, dout_ = [x.view(-1, *x.shape[-3:]) for x in [q, out, dout]]
                        # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                        kv_ = kv.view(-1, *kv.shape[-4:])
                    elif ctx.qkv_format == "thd":
                        q_, kv_, out_, dout_ = q, kv, out, dout
                    if ctx.use_fused_attention:
                        if ctx.fp8:
                            aux_ctx_tensors = [
                                softmax_lse,
                                softmax_lse,
                                rng_states[cp_size - i - 1],
                            ]
                        else:
                            aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size - i - 1]]
                        q_part = q_
                        k_part = kv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[0]
                        v_part = kv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[1]
                        out_part = out_
                        dout_part = dout_

                        if ctx.fp8:
                            q_part = ctx.QKV_quantizer.create_tensor_from_data(
                                q_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            k_part = ctx.QKV_quantizer.create_tensor_from_data(
                                k_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            v_part = ctx.QKV_quantizer.create_tensor_from_data(
                                v_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            out_part = ctx.O_quantizer.create_tensor_from_data(
                                out_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            dout_part = ctx.dO_quantizer.create_tensor_from_data(
                                dout_part, fake_dtype=dout_dtype, internal=True
                            )
                            fp8_meta_kwargs["dp_quantizer"] = dP_quantizer_per_step[i]
                            fp8_meta_kwargs["dqkv_quantizer"] = dQKV_CP_quantizer_per_step[i]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_part,
                            k_part,
                            v_part,
                            out_part,
                            dout_part,
                            ctx.qkv_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=ctx.attn_mask_type,
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                        if ctx.fp8:
                            dq_ = dq_._data
                            dk_ = dk_._data
                            dv_ = dv_._data
                    else:
                        dq_ = torch.empty_like(q_)
                        dkv_ = torch.empty_like(kv_)
                        fa_backward_args_thd = []
                        if ctx.qkv_format == "thd":
                            fa_backward_args_thd = [
                                cu_seqlens_q_per_step[cp_size - i - 1],
                                cu_seqlens_kv_per_step[cp_size - i - 1],
                                ctx.max_seqlen_q,
                                ctx.max_seqlen_kv,
                            ]
                        if _use_flash_attn_3 or (
                            _flash_attn_2_3_plus and not _flash_attn_2_7_0_plus
                        ):
                            fa_backward_kwargs["window_size"] = (-1, 0)
                        elif _flash_attn_2_7_0_plus:
                            fa_backward_kwargs["window_size_left"] = -1
                            fa_backward_kwargs["window_size_right"] = 0
                        if not _use_flash_attn_3:
                            fa_backward_kwargs["rng_state"] = rng_states[cp_size - i - 1]
                        flash_attn_bwd(
                            dout_,
                            q_,
                            kv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[0],
                            kv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[1],
                            out_,
                            softmax_lse,
                            dq_,
                            dkv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else dkv_[0],
                            dkv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else dkv_[1],
                            *fa_backward_args_thd,
                            causal=True,
                            **fa_backward_kwargs,
                        )
                elif i >= (cp_size - rank - 1):
                    if ctx.qkv_format == "bshd":
                        # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                        q_, out_, dout_ = [
                            x.view(x.shape[0], -1, *x.shape[-2:]) for x in [q, out, dout]
                        ]
                        # [b, 2, sk//2, 2, np, hn] -> [b, sk//2, 2, np, hn]
                        kv_ = kv[:, 0]
                    elif ctx.qkv_format == "sbhd":
                        # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                        q_, out_, dout_ = [x.view(-1, *x.shape[-3:]) for x in [q, out, dout]]
                        # [2, sk//2, b, 2, np, hn] -> [sk//2, b, 2, np, hn]
                        kv_ = kv[0]
                    elif ctx.qkv_format == "thd":
                        q_, out_, dout_ = q, out, dout
                        # [2, t, np, hn] -> [2, t/2, np, hn]
                        kv_ = tex.thd_read_half_tensor(kv, cu_seqlens_kv_padded, 0)
                    if ctx.use_fused_attention:
                        kv_ = kv_.contiguous()
                        if ctx.fp8:
                            aux_ctx_tensors = [
                                softmax_lse,
                                softmax_lse,
                                rng_states[cp_size - i - 1],
                            ]
                        else:
                            aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size - i - 1]]
                        q_part = q_
                        k_part = kv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[0]
                        v_part = kv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[1]
                        out_part = out_
                        dout_part = dout_

                        if ctx.fp8:
                            q_part = ctx.QKV_quantizer.create_tensor_from_data(
                                q_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            k_part = ctx.QKV_quantizer.create_tensor_from_data(
                                k_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            v_part = ctx.QKV_quantizer.create_tensor_from_data(
                                v_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            out_part = ctx.O_quantizer.create_tensor_from_data(
                                out_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            dout_part = ctx.dO_quantizer.create_tensor_from_data(
                                dout_part, fake_dtype=dout_dtype, internal=True
                            )
                            fp8_meta_kwargs["dp_quantizer"] = dP_quantizer_per_step[i]
                            fp8_meta_kwargs["dqkv_quantizer"] = dQKV_CP_quantizer_per_step[i]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv // 2,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_part,
                            k_part,
                            v_part,
                            out_part,
                            dout_part,
                            ctx.qkv_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=(
                                None if cu_seqlens_kv_padded is None else cu_seqlens_kv_padded // 2
                            ),
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="padding" if padding else "no_mask",
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                        if ctx.fp8:
                            dq_ = dq_._data
                            dk_ = dk_._data
                            dv_ = dv_._data
                    else:
                        dq_ = torch.empty_like(q_)
                        dkv_ = torch.empty_like(kv_)
                        fa_backward_args_thd = []
                        if ctx.qkv_format == "thd":
                            fa_backward_args_thd = [
                                cu_seqlens_q_per_step[cp_size - i - 1],
                                cu_seqlens_kv_per_step[cp_size - i - 1],
                                ctx.max_seqlen_q,
                                ctx.max_seqlen_kv // 2,
                            ]
                        if _use_flash_attn_3 or (
                            _flash_attn_2_3_plus and not _flash_attn_2_7_0_plus
                        ):
                            fa_backward_kwargs["window_size"] = (-1, -1)
                        if _flash_attn_2_7_0_plus:
                            fa_backward_kwargs["window_size_left"] = -1
                            fa_backward_kwargs["window_size_right"] = -1
                        if not _use_flash_attn_3:
                            fa_backward_kwargs["rng_state"] = rng_states[cp_size - i - 1]
                        flash_attn_bwd(
                            dout_,
                            q_,
                            kv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[0],
                            kv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[1],
                            out_,
                            softmax_lse,
                            dq_,
                            dkv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else dkv_[0],
                            dkv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else dkv_[1],
                            *fa_backward_args_thd,
                            causal=False,
                            **fa_backward_kwargs,
                        )
                else:
                    if ctx.qkv_format == "bshd":
                        # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                        q_, out_, dout_ = q[:, 1], out[:, 1], dout[:, 1]
                        # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                        kv_ = kv.view(kv.shape[0], -1, *kv.shape[-3:])
                    elif ctx.qkv_format == "sbhd":
                        # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                        q_, out_, dout_ = q[1], out[1], dout[1]
                        # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                        kv_ = kv.view(-1, *kv.shape[-4:])
                    elif ctx.qkv_format == "thd":
                        # [t, np, hn] -> [t/2, np, hn]
                        q_, out_, dout_ = [
                            tex.thd_read_half_tensor(x, cu_seqlens_q_padded, 1)
                            for x in [q, out, dout]
                        ]
                        kv_ = kv
                    if ctx.use_fused_attention:
                        q_, out_, dout_ = [x.contiguous() for x in [q_, out_, dout_]]
                        if ctx.fp8:
                            aux_ctx_tensors = [
                                softmax_lse_,
                                softmax_lse_,
                                rng_states[cp_size - i - 1],
                            ]
                        else:
                            aux_ctx_tensors = [softmax_lse_, rng_states[cp_size - i - 1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size - i - 1]]

                        q_part = q_
                        k_part = kv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[0]
                        v_part = kv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[1]
                        out_part = out_
                        dout_part = dout_

                        if ctx.fp8:
                            q_part = ctx.QKV_quantizer.create_tensor_from_data(
                                q_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            k_part = ctx.QKV_quantizer.create_tensor_from_data(
                                k_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            v_part = ctx.QKV_quantizer.create_tensor_from_data(
                                v_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            out_part = ctx.O_quantizer.create_tensor_from_data(
                                out_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            dout_part = ctx.dO_quantizer.create_tensor_from_data(
                                dout_part, fake_dtype=dout_dtype, internal=True
                            )
                            fp8_meta_kwargs["dp_quantizer"] = dP_quantizer_per_step[i]
                            fp8_meta_kwargs["dqkv_quantizer"] = dQKV_CP_quantizer_per_step[i]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q // 2,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_part,
                            k_part,
                            v_part,
                            out_part,
                            dout_part,
                            ctx.qkv_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=(
                                None if cu_seqlens_q_padded is None else cu_seqlens_q_padded // 2
                            ),
                            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="padding" if padding else "no_mask",
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                        if ctx.fp8:
                            dq_ = dq_._data
                            dk_ = dk_._data
                            dv_ = dv_._data
                    else:
                        dq_ = torch.empty_like(q_)
                        dkv_ = torch.empty_like(kv_)
                        fa_backward_args_thd = []
                        if ctx.qkv_format == "thd":
                            fa_backward_args_thd = [
                                cu_seqlens_q_per_step[cp_size - i - 1],
                                cu_seqlens_kv_per_step[cp_size - i - 1],
                                ctx.max_seqlen_q // 2,
                                ctx.max_seqlen_kv,
                            ]
                        if _use_flash_attn_3 or (
                            _flash_attn_2_3_plus and not _flash_attn_2_7_0_plus
                        ):
                            fa_backward_kwargs["window_size"] = (-1, -1)
                        elif _flash_attn_2_7_0_plus:
                            fa_backward_kwargs["window_size_left"] = -1
                            fa_backward_kwargs["window_size_right"] = -1
                        if not _use_flash_attn_3:
                            fa_backward_kwargs["rng_state"] = rng_states[cp_size - i - 1]
                        flash_attn_bwd(
                            dout_,
                            q_,
                            kv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[0],
                            kv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[1],
                            out_,
                            softmax_lse_,
                            dq_,
                            dkv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else dkv_[0],
                            dkv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else dkv_[1],
                            *fa_backward_args_thd,
                            causal=False,
                            **fa_backward_kwargs,
                        )
            else:
                if ctx.use_fused_attention:
                    if ctx.fp8:
                        aux_ctx_tensors = [softmax_lse, softmax_lse, rng_states[cp_size - i - 1]]
                    else:
                        aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                    if attn_dbias is not None:
                        aux_ctx_tensors += [attn_biases[cp_size - i - 1]]
                    q_part = q
                    k_part = kv[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv[0]
                    v_part = kv[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv[1]
                    out_part = out
                    dout_part = dout

                    if ctx.fp8:
                        q_part = ctx.QKV_quantizer.create_tensor_from_data(
                            q_part, fake_dtype=ctx.qkv_dtype, internal=True
                        )
                        k_part = ctx.QKV_quantizer.create_tensor_from_data(
                            k_part, fake_dtype=ctx.qkv_dtype, internal=True
                        )
                        v_part = ctx.QKV_quantizer.create_tensor_from_data(
                            v_part, fake_dtype=ctx.qkv_dtype, internal=True
                        )
                        out_part = ctx.O_quantizer.create_tensor_from_data(
                            out_part, fake_dtype=ctx.qkv_dtype, internal=True
                        )
                        dout_part = ctx.dO_quantizer.create_tensor_from_data(
                            dout_part, fake_dtype=dout_dtype, internal=True
                        )
                        fp8_meta_kwargs["dp_quantizer"] = dP_quantizer_per_step[i]
                        fp8_meta_kwargs["dqkv_quantizer"] = dQKV_CP_quantizer_per_step[i]
                    dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_kv,
                        cu_seqlens_q_per_step[cp_size - i - 1],
                        cu_seqlens_kv_per_step[cp_size - i - 1],
                        q_part,
                        k_part,
                        v_part,
                        out_part,
                        dout_part,
                        ctx.qkv_dtype,
                        fused_attn_dqkv_dtype,
                        aux_ctx_tensors,
                        fused_attn_backend,
                        cu_seqlens_q_padded=cu_seqlens_q_padded,
                        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                        attn_scale=ctx.softmax_scale,
                        dropout=ctx.dropout_p,
                        qkv_layout=qkv_layout,
                        attn_mask_type=ctx.attn_mask_type,
                        attn_bias_type=ctx.attn_bias_type,
                        deterministic=ctx.deterministic,
                        **fp8_meta_kwargs,
                    )

                    if ctx.fp8:
                        dq_ = dq_._data
                        dk_ = dk_._data
                        dv_ = dv_._data

                else:
                    dq_ = torch.empty_like(q)
                    dkv_ = torch.empty_like(kv)
                    fa_backward_args_thd = []
                    if ctx.qkv_format == "thd":
                        fa_backward_args_thd = [
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv,
                        ]
                    if _use_flash_attn_3 or (_flash_attn_2_3_plus and not _flash_attn_2_7_0_plus):
                        fa_backward_kwargs["window_size"] = (-1, -1)
                    elif _flash_attn_2_7_0_plus:
                        fa_backward_kwargs["window_size_left"] = -1
                        fa_backward_kwargs["window_size_right"] = -1
                    if not _use_flash_attn_3:
                        fa_backward_kwargs["rng_state"] = rng_states[cp_size - i - 1]
                    flash_attn_bwd(
                        dout,
                        q,
                        kv[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv[0],
                        kv[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv[1],
                        out,
                        softmax_lse,
                        dq_,
                        dkv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else dkv_[0],
                        dkv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else dkv_[1],
                        *fa_backward_args_thd,
                        causal=False,
                        **fa_backward_kwargs,
                    )

            if ctx.fp8:
                dq = dq_fp8[(rank + i + 1) % cp_size]
            if causal and ctx.qkv_format in ["bshd", "sbhd"] and i >= (cp_size - rank - 1):
                # [b, sq, np, hn] -> [b, 2, sq//2, np, hn] or
                # [sq, b, np, hn] -> [2, sq//2, b, np, hn]
                dq_ = dq_.view(*dq.shape)

            if ctx.fp8:
                if i >= (cp_size - rank - 1) or not causal:
                    dq.copy_(dq_)
                else:
                    if ctx.qkv_format == "bshd":
                        dq[:, 0, ...].fill_(0)
                        dq[:, 1, ...].copy_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[0].fill_(0)
                        dq[1].copy_(dq_)
            elif causal:
                if i > (cp_size - rank - 1):
                    dq.add_(dq_)
                elif i == (cp_size - rank - 1):
                    if rank == (cp_size - 1):
                        dq.copy_(dq_)
                    else:
                        if ctx.qkv_format == "bshd":
                            dq[:, 0, ...].copy_(dq_[:, 0, ...])
                            dq[:, 1, ...].add_(dq_[:, 1, ...])
                        elif ctx.qkv_format == "sbhd":
                            dq[0].copy_(dq_[0])
                            dq[1].add_(dq_[1])
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, "copy", "add")
                elif i > 0:
                    if ctx.qkv_format == "bshd":
                        dq[:, 1, ...].add_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[1].add_(dq_)
                    elif ctx.qkv_format == "thd":
                        tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, "none", "add")
                else:
                    if ctx.qkv_format == "bshd":
                        dq[:, 1, ...].copy_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[1].copy_(dq_)
                    elif ctx.qkv_format == "thd":
                        tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, "none", "copy")
            else:
                if i == 0:
                    dq.copy_(dq_)
                else:
                    dq.add_(dq_)

            if attn_dbias is not None:
                idx = (rank + i + 1) % cp_size
                if i == (cp_size - 1) or not causal:
                    # [b, np, sq, sk//cp] -> [b, np, sq, 2, sk//(2*cp)]
                    dbias_ = dbias_.view(*dbias_.shape[:-1], 2, dbias_.shape[-1] // 2)
                    attn_dbias[..., idx, :].copy_(dbias_[..., 0, :])
                    attn_dbias[..., (2 * cp_size - idx - 1), :].copy_(dbias_[..., 1, :])
                elif i >= (cp_size - rank - 1):
                    # [b, np, sq, sk//(2*cp)]
                    attn_dbias[..., idx, :].copy_(dbias_)
                else:
                    # [b, np, sq//2, sk//cp] -> [b, np, sq//2, 2, sk//(2*cp)]
                    dbias_ = dbias_.view(*dbias_.shape[:-1], 2, dbias_.shape[-1] // 2)
                    attn_dbias_[..., 1, :, idx, :].copy_(dbias_[..., 0, :])
                    attn_dbias_[..., 1, :, (2 * cp_size - idx - 1), :].copy_(dbias_[..., 1, :])

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()

            if ctx.fp8:
                if i < cp_size - 1:
                    dkv = dkv_fp8_[(rank + i + 1) % cp_size]
                else:
                    dkv = dkv_fp8[(rank + i + 1) % cp_size]
            else:
                dkv = p2p_comm_buffers[(i + 1) % 2][1]
            if ctx.use_fused_attention:
                if ctx.qkv_format in ["bshd", "sbhd"]:
                    dkv_ = _combine_tensors([dk_, dv_], -2)
                elif ctx.qkv_format == "thd":
                    dkv_ = torch.cat(
                        (dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0
                    )  # pylint: disable=used-before-assignment
            if ctx.qkv_format in ["bshd", "sbhd"]:
                # [b, 2, sk//2, 2, np, hn] -> [2, b, 2, sk//2, np, hn] or
                # [2, sk//2, b, 2, np, hn] -> [2, 2, sk//2, b, np, hn]
                dkv = dkv.view(2, *dkv.shape[0:-3], *dkv.shape[-2:])
                dkv_ = dkv_.movedim(-3, 0)
                if causal and (i < (cp_size - rank - 1) or i == (cp_size - 1)):
                    # [2, b, sk, np, hn] -> [2, b, 2, sk//2, np, hn] or
                    # [2, sk, b, np, hn] -> [2, 2, sk//2, b, np, hn]
                    dkv_ = dkv_.view(*dkv.shape)

            if ctx.fp8:
                if causal and i >= (cp_size - rank - 1) and i != (cp_size - 1):
                    if ctx.qkv_format == "bshd":
                        dkv[:, :, 0, ...].copy_(dkv_)
                        dkv[:, :, 1, ...].fill_(0)
                    elif ctx.qkv_format == "sbhd":
                        dkv[:, 0, ...].copy_(dkv_)
                        dkv[:, 1, ...].fill_(0)
                else:
                    dkv.copy_(dkv_)
            elif causal:
                if i == (cp_size - 1):
                    if rank == 0:
                        if ctx.qkv_format == "bshd":
                            dkv[:, :, 0, ...].add_(dkv_[:, :, 0, ...])
                            dkv[:, :, 1, ...].copy_(dkv_[:, :, 1, ...])
                        elif ctx.qkv_format == "sbhd":
                            dkv[:, 0, ...].add_(dkv_[:, 0, ...])
                            dkv[:, 1, ...].copy_(dkv_[:, 1, ...])
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dkv, dkv_, cu_seqlens_kv_padded, "add", "copy")
                    else:
                        dkv.add_(dkv_)
                elif i >= (cp_size - rank - 1):
                    if i == 0 and rank == (cp_size - 1):
                        if ctx.qkv_format == "bshd":
                            dkv[:, :, 0, ...].copy_(dkv_)
                        elif ctx.qkv_format == "sbhd":
                            dkv[:, 0, ...].copy_(dkv_)
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dkv, dkv_, cu_seqlens_kv_padded, "copy", "none")
                    else:
                        if ctx.qkv_format == "bshd":
                            dkv[:, :, 0, ...].add_(dkv_)
                        elif ctx.qkv_format == "sbhd":
                            dkv[:, 0, ...].add_(dkv_)
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(dkv, dkv_, cu_seqlens_kv_padded, "add", "none")
                elif i > 0:
                    dkv.add_(dkv_)
                else:
                    dkv.copy_(dkv_)
            else:
                if i == 0:
                    dkv.copy_(dkv_)
                else:
                    dkv.add_(dkv_)

        if ctx.fp8 and ctx.use_fused_attention:
            amax_cp_bwd = amax_per_step.amax(dim=1)
            ctx.dP_quantizer.amax = amax_cp_bwd[0]
            ctx.dQKV_CP_quantizer.amax = amax_cp_bwd[1]
            if ctx.qkv_format in ["bshd", "sbhd"]:
                # [cp, b, 2, sk//2, 2, np, hn] -> [cp, 2, b, 2, sk//2, np, hn] or
                # [cp, 2, sk//2, b, 2, np, hn] -> [cp, 2, 2, sk//2, b, np, hn]
                dkv_fp8 = dkv_fp8.view(cp_size, 2, *dkv_fp8.shape[1:-3], *dkv_fp8.shape[-2:])
            dq = ctx.dQKV_CP_quantizer.create_tensor_from_data(
                dq_fp8, fake_dtype=torch.float32, internal=True
            )
            dkv = ctx.dQKV_CP_quantizer.create_tensor_from_data(
                dkv_fp8, fake_dtype=torch.float32, internal=True
            )
            dq, dkv = [x.dequantize(dtype=torch.float32) for x in [dq, dkv]]
            dq, dkv = [x.sum(dim=0).to(dout_dtype) for x in [dq, dkv]]

        if causal:
            if ctx.qkv_format == "bshd":
                # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                dq = dq.view(dq.shape[0], -1, *dq.shape[-2:])
                # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                dkv = dkv.view(*dkv.shape[0:2], -1, *dkv.shape[-2:])
            elif ctx.qkv_format == "sbhd":
                # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                dq = dq.view(-1, *dq.shape[-3:])
                # [2, 2, sk//2, b, np, hn] -> [2, sk, b, np, hn]
                dkv = dkv.view(dkv.shape[0], -1, *dkv.shape[-3:])

        if ctx.qkv_format == "thd" and not ctx.use_fused_attention:
            dq[cu_seqlens_q_padded[-1] :].fill_(0)
            dkv[:, cu_seqlens_kv_padded[-1] :].fill_(0)

        if ctx.fp8 and ctx.is_input_fp8:
            assert torch.uint8 not in [dq.dtype, dkv.dtype]
            dq, dkv = [ctx.dQKV_quantizer(x)._data for x in [dq, dkv]]
        dk, dv = dkv[0], dkv[1]

        if cp_size_a2a > 1:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(cp_size_a2a, q.device, False)
            dq, dk, dv = flash_attn_a2a_communicate(
                [dq, dk, dv],
                chunk_ids_for_a2a,
                seq_dim,
                cp_size_a2a,
                ctx.cp_group_a2a,
                ctx.cp_stream,
                False,
            )
            if ctx.qkv_format == "bshd":
                dq, dk, dv = [x.view(ctx.batch_size, -1, *x.shape[-2:]) for x in [dq, dk, dv]]
            elif ctx.qkv_format == "sbhd":
                dq, dk, dv = [x.view(-1, ctx.batch_size, *x.shape[-2:]) for x in [dq, dk, dv]]

        if attn_dbias is not None:
            # [b, np, sq, 2*cp, sk//(2*cp)] -> [b, np, sq, sk]
            attn_dbias = attn_dbias.view(*attn_dbias.shape[:-2], -1)
        # converting torch.uint8 to float8tensor
        if ctx.fp8 and ctx.is_input_fp8:
            dq = ctx.dQKV_quantizer.create_tensor_from_data(dq, fake_dtype=dout_dtype)
            dk = ctx.dQKV_quantizer.create_tensor_from_data(dk, fake_dtype=dout_dtype)
            dv = ctx.dQKV_quantizer.create_tensor_from_data(dv, fake_dtype=dout_dtype)
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVP2P.backward")

        return (
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            attn_dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def get_kv_seq_info_after_all_gather(
    local_chunk_id, cp_size, max_seqlen_q, max_seqlen_kv, window_size, causal
):
    """Compute KV sequence index range and update window size after all-gather."""
    local_chunk_end_idx = (local_chunk_id + 1) * max_seqlen_kv
    full_seq_end_idx = max_seqlen_kv * cp_size * 2

    if window_size is None:
        window_size = (-1, 0) if causal else (-1, -1)

    if window_size[1] == -1:
        seq_end_idx = full_seq_end_idx
        window_size_right = -1
    else:
        seq_end_idx = min(full_seq_end_idx, local_chunk_end_idx + window_size[1])
        window_size_right = local_chunk_end_idx + window_size[1] - seq_end_idx

    if window_size[0] == -1:
        seq_start_idx = 0
        window_size_left = -1
    else:
        seq_start_idx = max(0, local_chunk_end_idx - max_seqlen_q - window_size[0])
        window_size_left = window_size[0] + seq_end_idx - local_chunk_end_idx

    return (seq_start_idx, seq_end_idx), (window_size_left, window_size_right)


class AttnFuncWithCPAndKVAllGather(torch.autograd.Function):
    """
    Attention implementation with context parallelism. KV all-gather between CP ranks is exposed.
    Refer section 3.3.2 of `The Llama 3 Herd of Models <https://arxiv.org/abs/2407.21783>`_.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        window_size,
        cp_group,
        cp_stream,
    ):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVAllGather.forward")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        qkv_dtype = q.dtype

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type
        assert not padding, f"{attn_mask_type} mask type is not supported!"
        if use_fused_attention and causal and "bottom_right" not in attn_mask_type:
            attn_mask_type = attn_mask_type + "_bottom_right"
        assert attn_bias_type == "no_bias", f"{attn_bias_type} bias type is not supported!"
        assert q.shape[-1] % 8 == 0, "Hidden size per attention head should be multiple of 8!"
        assert (
            use_fused_attention or _flash_attn_2_3_plus
        ), "Sliding window attention only can work with FusedAttention or FlashAttention >= 2.3!"

        flash_attn_fwd = None
        if not use_fused_attention:
            fa_forward_kwargs = {"softmax_scale": softmax_scale}
            if _use_flash_attn_3:
                if qkv_format == "thd":
                    flash_attn_fwd = _flash_attn_varlen_fwd_v3
                else:
                    flash_attn_fwd = _flash_attn_fwd_v3
            else:
                if qkv_format == "thd":
                    flash_attn_fwd = _flash_attn_varlen_fwd
                else:
                    flash_attn_fwd = _flash_attn_fwd
                fa_forward_kwargs["dropout_p"] = dropout_p
                fa_forward_kwargs["return_softmax"] = False
                if _flash_attn_2_4_plus:
                    fa_forward_kwargs["alibi_slopes"] = None
                if _flash_attn_2_5_7_plus and qkv_format == "thd":
                    fa_forward_kwargs["block_table"] = None
                if _flash_attn_2_6_0_plus:
                    fa_forward_kwargs["softcap"] = 0.0

        assert qkv_format != "thd", f"{qkv_format} format is not supported!"
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        seq_dim = qkv_format.index("s")
        assert (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"

        max_seqlen_q = max_seqlen_q // (2 * cp_size)
        max_seqlen_kv = max_seqlen_kv // (2 * cp_size)
        if use_fused_attention or qkv_format == "thd":
            cu_seqlens_q = cu_seqlens_q // (2 * cp_size)
        cu_seqlens_q_padded = (
            None if cu_seqlens_q_padded is None else cu_seqlens_q_padded // (2 * cp_size)
        )

        # [b, s, np, hn] -> [b, 2, s//2, np, hn] or [s, b, np, hn] -> [2, s//2, b, np, hn]
        q = q.view(*q.shape[:seq_dim], 2, q.shape[seq_dim] // 2, *q.shape[(seq_dim + 1) :])
        # [b, s, np, hn] or [s, b, np, hn] -> [s, b, np, hn]
        k, v = [x.movedim(seq_dim, 0).contiguous() for x in [k, v]]

        # [s, b, np, hn] -> [cp, s, b, np, hn]
        k_ag, _ = gather_along_first_dim(k, cp_group)
        v_ag, _ = gather_along_first_dim(v, cp_group)

        # [cp, s, b, np, hn] -> [cp*2, s//2, b, np, hn]
        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering(cp_size, k.device, True)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, np, hn] -> [cp*s, b, np, hn]
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        cp_stream.wait_stream(torch.cuda.current_stream())

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]
        kv_seq_range_per_step = [None, None]
        window_size_per_step = [None, None]
        cu_seqlens_kv_per_step = [None, None]
        out_per_step = [None, None]
        softmax_lse_per_step = [None, None]
        rng_states = [None, None]
        out = torch.empty_like(q)

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.cuda.stream(flash_attn_streams[i]):
                    # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                    # or [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                    q_ = q.select(seq_dim, i).contiguous()
                    kv_seq_range_per_step[i], window_size_per_step[i] = (
                        get_kv_seq_info_after_all_gather(
                            local_seq_chunk_ids[i],
                            cp_size,
                            max_seqlen_q,
                            max_seqlen_kv,
                            window_size,
                            causal,
                        )
                    )
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i][0],
                        kv_seq_range_per_step[i][1],
                    )
                    max_seqlen_kv_ = seq_end_idx - seq_start_idx
                    if use_fused_attention or qkv_format == "thd":
                        cu_seqlens_kv_per_step[i] = _get_full_cu_seqlens(
                            k.shape[1], max_seqlen_kv_, k.device
                        )
                    k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]
                    # [s_range, b, np, hn] -> [b, s_range, np, hn] or [s_range, b, np, hn]
                    k_, v_ = [x.movedim(0, seq_dim).contiguous() for x in [k_, v_]]
                    if use_fused_attention:
                        out_per_step[i], [softmax_lse_per_step[i], rng_states[i]] = fused_attn_fwd(
                            is_training,
                            max_seqlen_q,
                            max_seqlen_kv_,
                            cu_seqlens_q,
                            cu_seqlens_kv_per_step[i],
                            q_,
                            k_,
                            v_,
                            qkv_dtype,
                            tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                            attn_scale=softmax_scale,
                            dropout=dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=attn_mask_type,
                            attn_bias_type=attn_bias_type,
                            attn_bias=attn_bias,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=cu_seqlens_kv_per_step[i],
                            window_size=window_size_per_step[i],
                        )
                    else:
                        fa_forward_args_thd = []
                        if qkv_format == "thd":
                            fa_forward_args_thd = [
                                cu_seqlens_q,
                                cu_seqlens_kv_per_step[i],
                                max_seqlen_q,
                                max_seqlen_kv_,
                            ]
                        if _use_flash_attn_3 or (
                            _flash_attn_2_3_plus and not _flash_attn_2_7_0_plus
                        ):
                            fa_forward_kwargs["window_size"] = window_size_per_step[i]
                        elif _flash_attn_2_7_0_plus:
                            fa_forward_kwargs["window_size_left"] = window_size_per_step[i][0]
                            fa_forward_kwargs["window_size_right"] = window_size_per_step[i][1]
                        fa_outputs = flash_attn_fwd(
                            q_,
                            k_,
                            v_,
                            *fa_forward_args_thd,
                            causal=causal,
                            **fa_forward_kwargs,
                        )
                        if not _flash_attn_2_7_0_plus:
                            out_per_step[i] = fa_outputs[4]
                            softmax_lse_per_step[i] = fa_outputs[5]
                            if not _use_flash_attn_3:
                                rng_states[i] = fa_outputs[7]
                        else:
                            out_per_step[i] = fa_outputs[0]
                            softmax_lse_per_step[i] = fa_outputs[1]
                            if not _use_flash_attn_3:
                                rng_states[i] = fa_outputs[3]

            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    if qkv_format == "bshd":
                        out[:, i - 1].copy_(out_per_step[i - 1])
                    elif qkv_format == "sbhd":
                        out[i - 1].copy_(out_per_step[i - 1])

        torch.cuda.current_stream().wait_stream(cp_stream)

        if use_fused_attention:
            if qkv_format == "bshd":
                out = out.view(out.shape[0], -1, *out.shape[-2:])
            elif qkv_format == "sbhd":
                out = out.view(-1, *out.shape[-3:])
        else:
            out = out.view(-1, *out.shape[-2:])

        ctx.save_for_backward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_q_padded,
            *cu_seqlens_kv_per_step,
            *out_per_step,
            *softmax_lse_per_step,
            *rng_states,
        )

        ctx.qkv_dtype = qkv_dtype
        ctx.kv_seq_range_per_step = kv_seq_range_per_step
        ctx.window_size_per_step = window_size_per_step
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVAllGather.forward")
        return out

    @staticmethod
    def backward(ctx, dout):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVAllGather.backward")
        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)

        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v, cu_seqlens_q, cu_seqlens_q_padded) = saved_tensors[:5]
        cu_seqlens_kv_per_step = saved_tensors[5:7]
        out_per_step = saved_tensors[7:9]
        softmax_lse_per_step = saved_tensors[9:11]
        rng_states = saved_tensors[11:13]
        kv_seq_range_per_step = ctx.kv_seq_range_per_step
        window_size_per_step = ctx.window_size_per_step

        seq_dim = ctx.qkv_format.index("s")
        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        dout = dout.view(q.shape)
        dq = torch.empty_like(q)
        dk = torch.zeros((k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device)
        dv = torch.zeros_like(dk)
        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), ctx.cp_stream]
        # synchronize dkv update across steps
        dkv_update_done = torch.cuda.Event()

        # [s, b, np, hn] -> [cp, s, b, np, hn]
        k_ag, _ = gather_along_first_dim(k, ctx.cp_group)
        v_ag, _ = gather_along_first_dim(v, ctx.cp_group)

        # [cp, s, b, np, hn] -> [cp*2, s//2, b, np, hn]
        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering(cp_size, k.device, True)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, np, hn] -> [cp*s, b, np, hn]
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        ctx.cp_stream.wait_stream(torch.cuda.current_stream())

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]

        flash_attn_bwd = None
        if not ctx.use_fused_attention:
            fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
            if _use_flash_attn_3:
                if ctx.qkv_format == "thd":
                    flash_attn_bwd = _flash_attn_varlen_bwd_v3
                else:
                    flash_attn_bwd = _flash_attn_bwd_v3
                fa_backward_kwargs["deterministic"] = ctx.deterministic
            else:
                if ctx.qkv_format == "thd":
                    flash_attn_bwd = _flash_attn_varlen_bwd
                else:
                    flash_attn_bwd = _flash_attn_bwd
                fa_backward_kwargs["dropout_p"] = ctx.dropout_p
                if _flash_attn_2_4_plus:
                    fa_backward_kwargs["alibi_slopes"] = None
                if _flash_attn_2_4_1_plus:
                    fa_backward_kwargs["deterministic"] = ctx.deterministic
                if _flash_attn_2_6_0_plus:
                    fa_backward_kwargs["softcap"] = 0.0

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.cuda.stream(flash_attn_streams[i]):
                    # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                    # or [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                    q_ = q.select(seq_dim, i).contiguous()
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i][0],
                        kv_seq_range_per_step[i][1],
                    )
                    max_seqlen_kv = seq_end_idx - seq_start_idx
                    k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]
                    # [cp*s, b, np, hn] -> [b, s_range, np, hn] or [s_range, b, np, hn]
                    k_, v_ = [x.movedim(0, seq_dim).contiguous() for x in [k_, v_]]
                    out_ = out_per_step[i]
                    dout_ = dout.select(seq_dim, i).contiguous().view(out_.shape)
                    if ctx.use_fused_attention:
                        aux_ctx_tensors = [softmax_lse_per_step[i], rng_states[i]]
                        dq_per_step[i], dk_per_step[i], dv_per_step[i], _ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            max_seqlen_kv,
                            cu_seqlens_q,
                            cu_seqlens_kv_per_step[i],
                            q_,
                            k_,
                            v_,
                            out_,
                            dout_,
                            ctx.qkv_dtype,
                            TE_DType[dout.dtype],
                            aux_ctx_tensors,
                            tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=cu_seqlens_kv_per_step[i],
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=ctx.attn_mask_type,
                            attn_bias_type=ctx.attn_bias_type,
                            window_size=window_size_per_step[i],
                            deterministic=ctx.deterministic,
                        )
                    else:
                        dq_per_step[i], dk_per_step[i], dv_per_step[i] = [
                            torch.empty_like(x) for x in [q_, k_, v_]
                        ]
                        fa_backward_args_thd = []
                        if ctx.qkv_format == "thd":
                            fa_backward_args_thd = [
                                cu_seqlens_q,
                                cu_seqlens_kv_per_step[i],
                                ctx.max_seqlen_q,
                                max_seqlen_kv,
                            ]
                        if not _use_flash_attn_3:
                            fa_backward_kwargs["rng_state"] = rng_states[i]
                        if _flash_attn_2_3_plus and not _flash_attn_2_7_0_plus:
                            fa_backward_kwargs["window_size"] = window_size_per_step[i]
                        if _flash_attn_2_7_0_plus:
                            fa_backward_kwargs["window_size_left"] = window_size_per_step[i][0]
                            fa_backward_kwargs["window_size_right"] = window_size_per_step[i][1]
                        flash_attn_bwd(
                            dout_,
                            q_,
                            k_,
                            v_,
                            out_,
                            softmax_lse_per_step[i],
                            dq_per_step[i],
                            dk_per_step[i],
                            dv_per_step[i],
                            *fa_backward_args_thd,
                            causal="causal" in ctx.attn_mask_type,
                            **fa_backward_kwargs,
                        )

            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    if ctx.qkv_format == "bshd":
                        dq[:, i - 1].copy_(dq_per_step[i - 1])
                    elif ctx.qkv_format == "sbhd":
                        dq[i - 1].copy_(dq_per_step[i - 1])
                    # [b, s_range, np, hn] or [s_range, b, np, hn] -> [s_range, b, np, hn]
                    dk_per_step[i - 1], dv_per_step[i - 1] = [
                        x.movedim(seq_dim, 0).contiguous()
                        for x in [dk_per_step[i - 1], dv_per_step[i - 1]]
                    ]
                    # wait until dkv update of last step is done
                    if i > 1:
                        flash_attn_streams[i - 1].wait_event(dkv_update_done)
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i - 1][0],
                        kv_seq_range_per_step[i - 1][1],
                    )
                    dk[seq_start_idx:seq_end_idx].add_(dk_per_step[i - 1])
                    dv[seq_start_idx:seq_end_idx].add_(dv_per_step[i - 1])
                    if i < len(local_seq_chunk_ids):
                        flash_attn_streams[i - 1].record_event(dkv_update_done)

        torch.cuda.current_stream().wait_stream(ctx.cp_stream)

        # [cp*s, b, np, hn] -> [cp*2, s//2, b, np, hn]
        dk = dk.view(2 * cp_size, -1, *dk.shape[-3:])
        dv = dv.view(2 * cp_size, -1, *dv.shape[-3:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering(cp_size, dk.device, False)
        dk = torch.index_select(dk, dim=0, index=chunk_ids_for_kv_ag)
        dv = torch.index_select(dv, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, np, hn] -> [cp*s, b, np, hn]
        dk = dk.view(-1, *dk.shape[-3:])
        dv = dv.view(-1, *dv.shape[-3:])
        dk, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        dq = dq.view(*dq.shape[:seq_dim], -1, *dq.shape[(seq_dim + 2) :])
        dk = dk.movedim(0, seq_dim).contiguous()
        dv = dv.movedim(0, seq_dim).contiguous()
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVAllGather.backward")

        return (
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class AttnFuncWithCPAndQKVOA2A(torch.autograd.Function):
    """
    Attention implementation with context parallelism. Like Ulysses, applying A2A to QKVO.
    Refer the paper `DeepSpeed Ulysses <https://arxiv.org/abs/2309.14509>`_.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        window_size,
        fp8,
        fp8_meta,
        cp_group,
        cp_stream,
        quantizers,
    ):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndQKVOA2A.forward")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        qkv_dtype = q.dtype

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type
        assert not padding, f"{attn_mask_type} mask type is not supported!"
        assert attn_bias_type == "no_bias", f"{attn_bias_type} bias type is not supported!"
        assert q.shape[-1] % 8 == 0, "Hidden size per attention head should be multiple of 8!"
        assert (
            window_size == (-1, 0)
            or window_size == (-1, -1)
            or use_fused_attention
            or _flash_attn_2_3_plus
        ), "Sliding window attention only can work with FusedAttention or FlashAttention >= 2.3!"

        flash_attn_fwd = None
        if not use_fused_attention:
            fa_forward_kwargs = {"softmax_scale": softmax_scale}
            if _use_flash_attn_3:
                if qkv_format == "thd":
                    flash_attn_fwd = _flash_attn_varlen_fwd_v3
                else:
                    flash_attn_fwd = _flash_attn_fwd_v3
                fa_forward_kwargs["window_size"] = window_size
            else:
                if qkv_format == "thd":
                    flash_attn_fwd = _flash_attn_varlen_fwd
                else:
                    flash_attn_fwd = _flash_attn_fwd
                fa_forward_kwargs["dropout_p"] = dropout_p
                fa_forward_kwargs["return_softmax"] = False
                if _use_flash_attn_3 or (_flash_attn_2_3_plus and not _flash_attn_2_7_0_plus):
                    fa_forward_kwargs["window_size"] = window_size
                elif _flash_attn_2_7_0_plus:
                    fa_forward_kwargs["window_size_left"] = window_size[0]
                    fa_forward_kwargs["window_size_right"] = window_size[1]
                if _flash_attn_2_4_plus:
                    fa_forward_kwargs["alibi_slopes"] = None
                if _flash_attn_2_5_7_plus and qkv_format == "thd":
                    fa_forward_kwargs["block_table"] = None
                if _flash_attn_2_6_0_plus:
                    fa_forward_kwargs["softcap"] = 0.0

        assert (
            q.shape[-2] % cp_size == 0 and k.shape[-2] % cp_size == 0
        ), "The number of attention heads needs to be divisible by CP size!"

        assert qkv_format != "thd", f"{qkv_format} format is not supported!"
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        batch_dim = qkv_format.index("b")
        seq_dim = qkv_format.index("s")
        assert (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"

        fused_attn_backend = None
        # "fp8_mha" decides outputs in fp8, while inputs are inferred from the real dtype
        is_input_fp8 = False
        is_output_fp8 = False

        QKV_quantizer, O_quantizer, S_quantizer, dQKV_quantizer, dO_quantizer, dP_quantizer = (
            get_attention_quantizers(fp8, quantizers, cp_specific_quantizers=False)
        )
        if fp8:
            if use_fused_attention:
                fused_attn_backend = FusedAttnBackend["FP8"]
                assert isinstance(k, q.__class__) and isinstance(
                    v, q.__class__
                ), "q, k, and v must have the same type."
                is_input_fp8 = isinstance(q, Float8Tensor)
                is_output_fp8 = fp8_meta is not None and fp8_meta["recipe"].fp8_mha
                if is_input_fp8:
                    QKV_quantizer = q._quantizer
                    q_fp8, k_fp8, v_fp8 = q, k, v
                    q, k, v = q_fp8._data, k_fp8._data, v_fp8._data
                elif int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                    q_f16, k_f16, v_f16 = q, k, v
                    q, k, v = [QKV_quantizer(x)._data for x in [q_f16, k_f16, v_f16]]
                fp8_meta_kwargs = {}
                fp8_meta_kwargs["s_quantizer"] = S_quantizer
                fp8_meta_kwargs["o_quantizer"] = O_quantizer  # partial result quantizer
            else:
                assert False, "FP8 is only supported with Fused Attention!"
        else:
            if use_fused_attention:
                fp8_meta_kwargs = {}
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(cp_size, q.device, True)
        q, k, v = flash_attn_a2a_communicate(
            [q, k, v], chunk_ids_for_a2a, seq_dim, cp_size, cp_group, cp_stream, True
        )

        if fp8 and not is_input_fp8 and not int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
            q_f16, k_f16, v_f16 = q, k, v
            q, k, v = [QKV_quantizer(x)._data for x in [q_f16, k_f16, v_f16]]

        batch_size = q.shape[batch_dim]
        if use_fused_attention:
            q_part, k_part, v_part = q, k, v
            if fp8:
                q_part = QKV_quantizer.create_tensor_from_data(
                    q, fake_dtype=qkv_dtype, internal=True
                )
                k_part = QKV_quantizer.create_tensor_from_data(
                    k, fake_dtype=qkv_dtype, internal=True
                )
                v_part = QKV_quantizer.create_tensor_from_data(
                    v, fake_dtype=qkv_dtype, internal=True
                )
            out, aux_ctx_tensors = fused_attn_fwd(
                is_training,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q_part,
                k_part,
                v_part,
                qkv_dtype,
                fused_attn_backend,
                attn_scale=softmax_scale,
                dropout=dropout_p,
                qkv_layout=qkv_layout,
                attn_mask_type=attn_mask_type,
                attn_bias_type=attn_bias_type,
                attn_bias=attn_bias,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                window_size=window_size,
                **fp8_meta_kwargs,
            )
            if fp8:
                out = out._data
        else:
            fa_forward_args_thd = []
            if qkv_format == "thd":
                fa_forward_args_thd = [
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                ]
            fa_outputs = flash_attn_fwd(
                q,
                k,
                v,
                *fa_forward_args_thd,
                causal=causal,
                **fa_forward_kwargs,
            )
            if not _flash_attn_2_7_0_plus:
                out, softmax_lse = fa_outputs[4], fa_outputs[5]
                rng_state = fa_outputs[7] if not _use_flash_attn_3 else None
            else:
                out, softmax_lse = fa_outputs[0], fa_outputs[1]
                rng_state = fa_outputs[3] if not _use_flash_attn_3 else None
            aux_ctx_tensors = [softmax_lse, rng_state]

        chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(cp_size, out.device, False)
        out = flash_attn_a2a_communicate(
            out, chunk_ids_for_a2a, seq_dim, cp_size, cp_group, cp_stream, False
        )

        if use_fused_attention:
            if qkv_format == "bshd":
                # [b*s, np, hn] -> [b, s, np, hn]
                out = out.view(batch_size, -1, *out.shape[-2:])
            elif qkv_format == "sbhd":
                # [s*b, np, hn] -> [s, b, np, hn]
                out = out.view(-1, batch_size, *out.shape[-2:])

        if fp8:
            if is_output_fp8:
                out_fp8 = O_quantizer.create_tensor_from_data(
                    out, fake_dtype=qkv_dtype, internal=False
                )
                out_ret = out_fp8
                out = out_fp8._data
            else:
                out_fp8 = O_quantizer.create_tensor_from_data(
                    out, fake_dtype=qkv_dtype, internal=True
                )
                out_f16 = out_fp8.dequantize(dtype=qkv_dtype)
                out_ret = out_f16
        else:
            out_ret = out

        if not fp8 or int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
            q_save, k_save, v_save, out_save = q, k, v, out
        else:
            if is_input_fp8:
                q_save, k_save, v_save = q, k, v
            else:
                q_save, k_save, v_save = q_f16, k_f16, v_f16
            if is_output_fp8:
                out_save = out
            else:
                out_save = out_f16

        tensors_to_save, tensor_objects = prepare_for_saving(
            q_save,
            k_save,
            v_save,
            out_save,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *aux_ctx_tensors,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects

        ctx.qkv_dtype = qkv_dtype
        ctx.QKV_quantizer = QKV_quantizer
        ctx.O_quantizer = O_quantizer
        ctx.S_quantizer = S_quantizer
        ctx.dQKV_quantizer = dQKV_quantizer
        ctx.dO_quantizer = dO_quantizer
        ctx.dP_quantizer = dP_quantizer

        ctx.batch_size = batch_size
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.attn_bias_type = attn_bias_type
        ctx.deterministic = deterministic
        ctx.window_size = window_size
        ctx.use_fused_attention = use_fused_attention
        ctx.fp8 = fp8 and int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
        ctx.fp8_meta = fp8_meta
        ctx.is_input_fp8 = is_input_fp8
        ctx.is_output_fp8 = is_output_fp8
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndQKVOA2A.forward")
        return out_ret

    @staticmethod
    def backward(ctx, dout):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndQKVOA2A.backward")
        cp_size = get_distributed_world_size(ctx.cp_group)

        (
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *aux_ctx_tensors,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format
        causal = "causal" in ctx.attn_mask_type
        seq_dim = ctx.qkv_format.index("s")

        dout_dtype = dout.dtype
        fused_attn_backend = None
        fused_attn_dqkv_dtype = None
        if ctx.fp8:
            if ctx.use_fused_attention:
                fused_attn_backend = FusedAttnBackend["FP8"]
                if ctx.is_output_fp8:
                    assert isinstance(dout, Float8Tensor), "dout must be Float8Tensors for FP8 MHA!"
                    ctx.dO_quantizer = dout._quantizer
                else:
                    dout = ctx.dO_quantizer(dout)
                fused_attn_dqkv_dtype = dout._fp8_dtype
                dout = dout._data
                fp8_meta_kwargs = {}
                fp8_meta_kwargs["s_quantizer"] = ctx.S_quantizer
                fp8_meta_kwargs["dp_quantizer"] = ctx.dP_quantizer
                fp8_meta_kwargs["dqkv_quantizer"] = ctx.dQKV_quantizer

            else:
                assert False, "FP8 is only supported with Fused Attention!"
        else:
            if ctx.fp8_meta is not None:
                if ctx.is_output_fp8:
                    assert isinstance(dout, Float8Tensor), "dout must be Float8Tensors for FP8 MHA!"
                    ctx.dO_quantizer = dout._quantizer
                    dout = dout._data
                if ctx.is_input_fp8:
                    q = ctx.QKV_quantizer.create_tensor_from_data(
                        q, fake_dtype=ctx.qkv_dtype, internal=True
                    )
                    k = ctx.QKV_quantizer.create_tensor_from_data(
                        k, fake_dtype=ctx.qkv_dtype, internal=True
                    )
                    v = ctx.QKV_quantizer.create_tensor_from_data(
                        v, fake_dtype=ctx.qkv_dtype, internal=True
                    )
                    q, k, v = [x.dequantize(dtype=ctx.qkv_dtype) for x in [q, k, v]]
            if ctx.use_fused_attention:
                fp8_meta_kwargs = {}
                fused_attn_dqkv_dtype = TE_DType[dout_dtype]
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        if not ctx.use_fused_attention:
            out = out.view(ctx.batch_size, -1, *out.shape[-2:])
        dout = dout.view(*out.shape)

        chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(cp_size, out.device, True)
        out, dout = flash_attn_a2a_communicate(
            [out, dout], chunk_ids_for_a2a, seq_dim, cp_size, ctx.cp_group, ctx.cp_stream, True
        )
        if not ctx.fp8 and ctx.fp8_meta is not None and ctx.is_output_fp8:
            out = ctx.O_quantizer.create_tensor_from_data(
                out, fake_dtype=ctx.qkv_dtype, internal=True
            )
            dout = ctx.dO_quantizer.create_tensor_from_data(
                dout, fake_dtype=dout_dtype, internal=True
            )
            out = out.dequantize(dtype=ctx.qkv_dtype)
            dout = dout.dequantize(dtype=dout_dtype)

        flash_attn_bwd = None
        if not ctx.use_fused_attention:
            fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
            if _use_flash_attn_3:
                if ctx.qkv_format == "thd":
                    flash_attn_bwd = _flash_attn_varlen_bwd_v3
                else:
                    flash_attn_bwd = _flash_attn_bwd_v3
                fa_backward_kwargs["window_size"] = ctx.window_size
                fa_backward_kwargs["deterministic"] = ctx.deterministic
            else:
                if ctx.qkv_format == "thd":
                    flash_attn_bwd = _flash_attn_varlen_bwd
                else:
                    flash_attn_bwd = _flash_attn_bwd
                fa_backward_kwargs["dropout_p"] = ctx.dropout_p
                if _use_flash_attn_3 or (_flash_attn_2_3_plus and not _flash_attn_2_7_0_plus):
                    fa_backward_kwargs["window_size"] = ctx.window_size
                elif _flash_attn_2_7_0_plus:
                    fa_backward_kwargs["window_size_left"] = ctx.window_size[0]
                    fa_backward_kwargs["window_size_right"] = ctx.window_size[1]
                if _flash_attn_2_4_plus:
                    fa_backward_kwargs["alibi_slopes"] = None
                if _flash_attn_2_4_1_plus:
                    fa_backward_kwargs["deterministic"] = ctx.deterministic
                if _flash_attn_2_6_0_plus:
                    fa_backward_kwargs["softcap"] = 0.0

        if ctx.use_fused_attention:
            q_part = q
            k_part = k
            v_part = v
            out_part = out
            dout_part = dout

            if ctx.fp8:
                q_part = ctx.QKV_quantizer.create_tensor_from_data(
                    q_part, fake_dtype=ctx.qkv_dtype, internal=True
                )
                k_part = ctx.QKV_quantizer.create_tensor_from_data(
                    k_part, fake_dtype=ctx.qkv_dtype, internal=True
                )
                v_part = ctx.QKV_quantizer.create_tensor_from_data(
                    v_part, fake_dtype=ctx.qkv_dtype, internal=True
                )
                out_part = ctx.O_quantizer.create_tensor_from_data(
                    out_part, fake_dtype=ctx.qkv_dtype, internal=True
                )
                dout_part = ctx.dO_quantizer.create_tensor_from_data(
                    dout_part, fake_dtype=dout_dtype, internal=True
                )

            dq, dk, dv, _ = fused_attn_bwd(
                ctx.max_seqlen_q,
                ctx.max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q_part,
                k_part,
                v_part,
                out_part,
                dout_part,
                ctx.qkv_dtype,
                fused_attn_dqkv_dtype,
                aux_ctx_tensors,
                fused_attn_backend,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                attn_scale=ctx.softmax_scale,
                dropout=ctx.dropout_p,
                qkv_layout=qkv_layout,
                attn_mask_type=ctx.attn_mask_type,
                attn_bias_type=ctx.attn_bias_type,
                window_size=ctx.window_size,
                deterministic=ctx.deterministic,
                **fp8_meta_kwargs,
            )
            if ctx.fp8:
                dq = dq._data
                dk = dk._data
                dv = dv._data
        else:
            softmax_lse, rng_state = aux_ctx_tensors
            dq, dk, dv = [torch.empty_like(x) for x in [q, k, v]]
            fa_backward_args_thd = []
            if ctx.qkv_format == "thd":
                fa_backward_args_thd = [
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_kv,
                ]
            if not _use_flash_attn_3:
                fa_backward_kwargs["rng_state"] = rng_state
            flash_attn_bwd(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                *fa_backward_args_thd,
                causal=causal,
                **fa_backward_kwargs,
            )

        chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(cp_size, q.device, False)
        dq, dk, dv = flash_attn_a2a_communicate(
            [dq, dk, dv], chunk_ids_for_a2a, seq_dim, cp_size, ctx.cp_group, ctx.cp_stream, False
        )

        if ctx.qkv_format == "bshd":
            dq, dk, dv = [x.view(ctx.batch_size, -1, *x.shape[-2:]) for x in [dq, dk, dv]]
        elif ctx.qkv_format == "sbhd":
            dq, dk, dv = [x.view(-1, ctx.batch_size, *x.shape[-2:]) for x in [dq, dk, dv]]

        if ctx.fp8:
            dq = ctx.dQKV_quantizer.create_tensor_from_data(
                dq, fake_dtype=dout_dtype, internal=not ctx.is_input_fp8
            )
            dk = ctx.dQKV_quantizer.create_tensor_from_data(
                dk, fake_dtype=dout_dtype, internal=not ctx.is_input_fp8
            )
            dv = ctx.dQKV_quantizer.create_tensor_from_data(
                dv, fake_dtype=dout_dtype, internal=not ctx.is_input_fp8
            )
            if not ctx.is_input_fp8:
                dq, dk, dv = [x.dequantize(dtype=dout_dtype) for x in [dq, dk, dv]]
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndQKVOA2A.backward")

        return (
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def attn_forward_func_with_cp(
    is_training,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    dropout_p,
    cp_group,
    cp_global_ranks,
    cp_stream,
    cp_comm_type,
    softmax_scale=None,
    qkv_format="bshd",
    attn_mask_type="causal",
    attn_bias_type="no_bias",
    attn_bias=None,
    deterministic=False,
    use_fused_attention=False,
    window_size=None,
    fp8=False,
    fp8_meta=None,
    quantizers=None,
) -> torch.Tensor:
    """
    Attention implementation with context parallelism.
    """

    if cp_comm_type == "a2a+p2p":
        assert isinstance(
            cp_group, list
        ), "Hierarchical CP implementation needs multi-level CP groups!"
        assert len(cp_group) == 2, "Current implementation only supports two-level CP groups!"
        if get_distributed_world_size(cp_group[0]) == 1:
            cp_group = cp_group[1]
            cp_comm_type = "p2p"
        elif get_distributed_world_size(cp_group[1]) == 1:
            cp_group = cp_group[0]
            cp_comm_type = "a2a"
    else:
        assert isinstance(
            cp_group, dist_group_type
        ), f"Unsupported process group for CP communication type {cp_comm_type}!"

    assert qkv_format in [
        "bshd",
        "sbhd",
        "thd",
    ], f"QKV format of {qkv_format} is not supported with context parallelism!"
    assert (
        qkv_format != "sbhd" or use_fused_attention
    ), "FlashAttention does not support sbhd format!"
    assert attn_bias is None or (use_fused_attention and "padding" not in attn_mask_type), (
        """Attention bias is only supported with FusedAttention and "causal" """
        """or "no_mask" mask types!"""
    )
    assert qkv_format != "thd" or (
        cu_seqlens_q_padded is not None and cu_seqlens_kv_padded is not None
    ), "cu_seqlens_padded cannot be None with context parallelism + THD format!"

    sliding_window_attn = (
        window_size is not None and window_size != (-1, 0) and window_size != (-1, -1)
    )
    assert not sliding_window_attn or cp_comm_type in [
        "a2a",
        "all_gather",
    ], "The context parallel running configs cannot support sliding window attetnion!"

    args = [
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
    ]

    if cp_comm_type in ["p2p", "a2a+p2p"]:
        args += [fp8, fp8_meta, cp_group, cp_global_ranks, cp_stream, quantizers]
        out = AttnFuncWithCPAndKVP2P.apply(*args)
    elif cp_comm_type == "all_gather":
        args.pop(5)
        args.pop(8)
        args += [window_size, cp_group, cp_stream]
        out = AttnFuncWithCPAndKVAllGather.apply(*args)
    elif cp_comm_type == "a2a":
        args += [window_size, fp8, fp8_meta, cp_group, cp_stream, quantizers]
        out = AttnFuncWithCPAndQKVOA2A.apply(*args)
    else:
        raise ValueError(f"Unsupported communication type: {cp_comm_type}!")

    return out


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self,
        dim: int,
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[int] = None,
        pretrained_max_position_embeddings: Optional[int] = None,
        rotary_base: float = 10000.0,
    ):
        """
        Parameters
        ----------
        dim: int
            rotary embedding dimension
        rotary_percent: float
            Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor: int
            if not None, discrete positions will be interpolated by this factor via the trick in
            https://arxiv.org/abs/2306.15595
        pretrained_max_position_embeddings: int
            pre-trained max_position_embeddings before position interpolation
        """
        super().__init__()
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.rotary_base = rotary_base
        inv_freq = 1.0 / (
            self.rotary_base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device())
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)
        self.pretrained_max_position_embeddings = pretrained_max_position_embeddings

    def forward(self, max_seq_len: int, offset: int = 0):
        """
        Create rotary position embedding frequencies

        Parameters
        ----------
        max_seq_len: int
            sequence length of a sample
        offset: int, default = 0
            fixed offset for freqencies
        """
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

        if (
            self.pretrained_max_position_embeddings is not None
            and self.seq_len_interpolation_factor is not None
        ):
            if (
                max_seq_len
                > self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor
            ):
                # dynamic linear scaling (length > position we have learned)
                seq *= 1 / (max_seq_len / self.pretrained_max_position_embeddings)
            else:
                # fixed linear scaling
                seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return emb.reshape(emb.size(0), 1, 1, emb.size(1))


class FusedRoPEFunc(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        if tensor_format == "sbhd":
            output = tex.fused_rope_forward(t, freqs, False)
        elif tensor_format == "bshd":
            output = tex.fused_rope_forward(t.transpose(0, 1), freqs, True).transpose(0, 1)
        elif tensor_format == "thd":
            output = tex.fused_rope_thd_forward(t, cu_seqlens, freqs, cp_size, cp_rank)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs, cu_seqlens)
        ctx.tensor_format = tensor_format
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        freqs, cu_seqlens = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = tex.fused_rope_backward(grad_output, freqs, False)
        elif ctx.tensor_format == "bshd":
            grad_input = tex.fused_rope_backward(
                grad_output.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        elif ctx.tensor_format == "thd":
            grad_input = tex.fused_rope_thd_backward(
                grad_output, cu_seqlens, freqs, ctx.cp_size, ctx.cp_rank
            )
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        return grad_input, None, None, None, None, None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[b, s, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    fused: bool, default = False
        Whether to use a fused applying RoPE implementation.
    tensor_format: {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    cu_seqlens: torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
        Should be `cu_seqlens_padded` when cp_size > 1.
    cp_size: int, default = 1.
        Context parallel world size. Only valid when `tensor_format` is 'thd' and `fused` is True.
    cp_rank: int, default = 0.
        Context parallel rank. Only valid when `tensor_format` is 'thd' and `fused` is True.
    """
    if fused:
        assert (
            tensor_format != "thd" or cu_seqlens is not None
        ), "cu_seqlens must not be None when tensor_format is 'thd'."
        return FusedRoPEFunc.apply(t, freqs, tensor_format, cu_seqlens, cp_size, cp_rank)

    assert tensor_format in ("sbhd", "bshd"), (
        "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
        f"when fused is False, got {tensor_format}."
    )

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert (
        cur_seq_len <= max_seq_len
    ), f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


class _SplitAlongDim(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(
        ctx,
        mixed_x_layer: torch.Tensor,
        split_dim: int,
        split_size_or_sections: Union[int, List[int], Tuple[int]],
        squeeze=False,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        ctx.split_dim = split_dim
        ctx.split_size_or_sections = split_size_or_sections
        if isinstance(mixed_x_layer, Float8TensorBase) and not isinstance(
            mixed_x_layer, Float8Tensor
        ):
            return tuple(
                Float8TensorBase(
                    fp8_scale_inv=mixed_x_layer._scale_inv,
                    fp8_dtype=mixed_x_layer._fp8_dtype,
                    data=x.squeeze(split_dim) if squeeze else x,
                    shape=x.squeeze(split_dim).shape if squeeze else x.shape,
                    quantizer=mixed_x_layer._quantizer,
                )
                for x in torch.split(
                    mixed_x_layer._data,
                    split_size_or_sections=split_size_or_sections,
                    dim=split_dim,
                )
            )
        if isinstance(mixed_x_layer, Float8Tensor):
            return tuple(
                Float8Tensor.make_like(
                    mixed_x_layer,
                    data=x.squeeze(split_dim) if squeeze else x,
                    shape=x.squeeze(split_dim).shape if squeeze else x.shape,
                )
                for x in torch.split(
                    mixed_x_layer._data,
                    split_size_or_sections=split_size_or_sections,
                    dim=split_dim,
                )
            )
        out_list = torch.split(mixed_x_layer, split_size_or_sections, dim=split_dim)
        if squeeze:
            out_list = [x.squeeze(split_dim) for x in out_list]
        return out_list

    @staticmethod
    def backward(ctx, *grad_outputs):
        # pylint: disable=missing-function-docstring
        assert len(grad_outputs) > 0, "No gradients received for backprop!"

        if isinstance(ctx.split_size_or_sections, (list, tuple)):
            split_sizes = ctx.split_size_or_sections
            assert len(grad_outputs) == len(
                split_sizes
            ), "Unequal number of gradients vs split sections for backprop!"
        if isinstance(ctx.split_size_or_sections, int):
            split_sizes = [ctx.split_size_or_sections] * len(grad_outputs)
        dims = len(grad_outputs[0].shape)
        split_dim = (ctx.split_dim + dims) % dims

        if isinstance(grad_outputs[0], Float8Tensor):
            noop_ok = True
            strides = grad_outputs[0].stride()
            data_ptr = grad_outputs[0]._data.untyped_storage().data_ptr()
            shape = list(grad_outputs[0].shape)
            for i, tensor in enumerate(grad_outputs):
                shape_i = shape
                shape_i[split_dim] = split_sizes[i]
                offset_size = sum(split_sizes[:i]) * np.prod(shape[split_dim + 1 :])
                if (
                    tensor.stride() != strides
                    or list(tensor.shape) != shape_i
                    or tensor._data.untyped_storage().data_ptr() != data_ptr
                    or tensor.storage_offset() != offset_size
                ):
                    noop_ok = False
                    break
            if noop_ok:
                ret = torch.Tensor().to(
                    device=grad_outputs[0].device, dtype=grad_outputs[0]._data.dtype
                )
                new_shape = list(shape)
                new_shape[split_dim] = sum(split_sizes)
                ret.set_(
                    grad_outputs[0]._data.untyped_storage(),
                    grad_outputs[0]._data.storage_offset(),
                    new_shape,
                    strides,
                )
                return (
                    Float8Tensor.make_like(grad_outputs[0], data=ret, shape=ret.shape),
                    None,
                    None,
                )

            grad_outputs_data = [x._data for x in grad_outputs]
            data = torch.cat(grad_outputs_data, dim=split_dim)
            return (
                Float8Tensor.make_like(grad_outputs[0], data=data, shape=data.shape),
                None,
                None,
                None,
            )
        noop_ok = True
        strides = grad_outputs[0].stride()
        data_ptr = grad_outputs[0].untyped_storage().data_ptr()
        shape = list(grad_outputs[0].shape)
        for i, tensor in enumerate(grad_outputs):
            shape_i = shape
            shape_i[split_dim] = split_sizes[i]
            offset_size = sum(split_sizes[:i]) * np.prod(shape[split_dim + 1 :])
            if (
                tensor.stride() != strides
                or list(tensor.shape) != shape_i
                or tensor.untyped_storage().data_ptr() != data_ptr
                or tensor.storage_offset() != offset_size
            ):
                noop_ok = False
                break
        if noop_ok:
            ret = torch.Tensor().to(device=grad_outputs[0].device, dtype=grad_outputs[0].dtype)
            new_shape = list(shape)
            new_shape[split_dim] = sum(split_sizes)
            ret.set_(
                grad_outputs[0].untyped_storage(),
                grad_outputs[0].storage_offset(),
                new_shape,
                strides,
            )
            return ret, None, None

        return torch.cat(grad_outputs, dim=split_dim), None, None


class UnfusedDotProductAttention(torch.nn.Module):
    """Parallel attention w/o QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_type: str = "self",
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        layer_number: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.softmax_scale = softmax_scale
        self.attention_type = attention_type
        self.attention_dropout_ctx = attention_dropout_ctx
        self.layer_number = layer_number

        self.scale_mask_softmax = FusedScaleMaskSoftmax(attention_mask_func)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        # An FP16 training trick required for certain GPT-like models.
        self.apply_qk_layer_scaling = (
            bool(int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))) and layer_number is not None
        )

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        cu_seqlens_kv: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        attn_mask_type: str = "causal",
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        window_size: Optional[Tuple[int, int]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unfused attention fprop"""
        assert (
            qkv_layout in QKVLayouts
        ), f"UnfusedDotProductAttention does not support qkv_layout = {qkv_layout}!"
        qkv_format = "".join([i for i in qkv_layout.split("_")[0] if i.isalpha()])
        if qkv_format == "bshd":
            # convert to sbhd and use sbhd implementation for now
            query_layer, key_layer, value_layer = [
                x.transpose(0, 1) for x in [query_layer, key_layer, value_layer]
            ]
        batch_size, max_seqlen_q, max_seqlen_kv = (
            query_layer.shape[1],
            query_layer.shape[0],
            key_layer.shape[0],
        )

        attn_mask_type, attention_mask, actual_seqlens_q, actual_seqlens_kv = get_full_mask(
            max_seqlen_q,
            max_seqlen_kv,
            attn_mask_type=attn_mask_type,
            attention_mask=attention_mask,
            window_size=window_size,
            attention_type=self.attention_type,
        )

        batch_size, seqlen = query_layer.shape[1], query_layer.shape[0]
        apply_qk_layer_scaling = self.apply_qk_layer_scaling and key_layer.dtype == torch.float16

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        if key_layer.shape[2] != query_layer.shape[2]:
            assert (
                query_layer.shape[2] % key_layer.shape[2] == 0
            ), "The number of attention heads must be divisible by the number of GQA groups!"
            key_layer = key_layer.repeat_interleave(
                int(query_layer.shape[2] / key_layer.shape[2]), dim=2
            )
            value_layer = value_layer.repeat_interleave(
                int(query_layer.shape[2] / value_layer.shape[2]), dim=2
            )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.reshape(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        scale = self.softmax_scale
        if apply_qk_layer_scaling:
            scale /= self.layer_number

        # Raw attention scores. [b * np, sq, sk]
        if core_attention_bias_type == "no_bias":
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=scale,
            ).view(*output_size)

        elif core_attention_bias_type == "pre_scale_bias":
            assert core_attention_bias is not None, "core_attention_bias should not be None!"
            matmul_result = torch.bmm(
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            )
            matmul_result = matmul_result.view(*output_size) + core_attention_bias
            matmul_result *= scale

        elif core_attention_bias_type in ["post_scale_bias", "alibi"]:
            if core_attention_bias_type == "post_scale_bias":
                assert core_attention_bias is not None, "core_attention_bias should not be None!"
            if core_attention_bias_type == "alibi":
                _, core_attention_bias = get_alibi(
                    output_size[1],
                    output_size[2],
                    output_size[3],
                    actual_seqlens_q=actual_seqlens_q if "padding" in attn_mask_type else None,
                    actual_seqlens_kv=actual_seqlens_kv if "padding" in attn_mask_type else None,
                    alibi_slopes=alibi_slopes,
                    bottom_right_alignment=attn_mask_type not in ["causal", "padding_causal"],
                )
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=scale,
            )
            matmul_result = (matmul_result.view(*output_size) + core_attention_bias).to(
                dtype=query_layer.dtype
            )

        # attention scores and attention mask [b, np, sq, sk]
        softmax_scale = self.layer_number if apply_qk_layer_scaling else None
        attention_probs = self.scale_mask_softmax(
            matmul_result, attention_mask, attn_mask_type, softmax_scale
        )

        # mask out the pad positions in softmax results, mostly for the rows (pad tokens from q)
        # the columns (pad tokens from k) are already zeroed out during softmax
        if "padding" in attn_mask_type:
            attention_probs = attention_probs.masked_fill(attention_mask, 0)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with self.attention_dropout_ctx():
            attention_probs = self.attention_dropout(attention_probs)

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        if qkv_format == "sbhd":
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            context_layer = context_layer.view(seqlen, batch_size, -1)

        if qkv_format == "bshd":
            # [b, np, sq, hn] --> [b, sq, np, hn]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

            # [b, sq, np, hn] --> [b, sq, hp]
            context_layer = context_layer.view(batch_size, seqlen, -1)

        return context_layer


class _PrepareQKVForFA(torch.autograd.Function):
    """This class converts QKV from interleaved (s, b, ...) layout
    to separate contiguous q, k, v tensors in (b, s, ...) layout."""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        # All inputs received are non-contiguous tensors.
        # The `query_layer` tensor is used to access the
        # full memory region of the QKV tensor.
        qkv = tex.fa_prepare_fwd(query_layer)
        q, k, v = split_tensor_along_dim(qkv, 0, 3)
        query_layer = torch.squeeze(q, 0)
        key_layer = torch.squeeze(k, 0)
        value_layer = torch.squeeze(v, 0)
        return query_layer, key_layer, value_layer

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        dqkv = tex.fa_prepare_bwd(dq, dk, dv)
        dq, dk, dv = split_tensor_along_dim(dqkv, -1, 3)
        return dq, dk, dv


def get_qkv_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qkv_format: str = "sbhd",
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

    Returns
    ----------
    qkv_layout: str
       Memory layout of `q`, `k` and `v`. Each `qkv_format` can be mapped to one of five
       memory layouts. For example, `sb3hd` means `q`, `k`, `v` are created as one chunk
       of memory and that they are interleaved in the `2`nd dimension. `sbhd_sbh2d` means
       `q` and `kv` are created in two chunks and that `q` itself is contiguous and `k`, `v`
       are interleaved with each other in the `3`rd dimension, `k = kv[:,:,:,0,:]` and
       `v = kv[:,:,:,1,:]`.
       Mapping:
       `sbhd`: {`sb3hd`, `sbh3d`, `sbhd_sb2hd`, `sbhd_sbh2d`, `sbhd_sbhd_sbhd`}
       `bshd`: {`bs3hd`, `bsh3d`, `bshd_bs2hd`, `bshd_bsh2d`, `bshd_bshd_bshd`}
       `thd` : {`t3hd`, `th3d`, `thd_t2hd`, `thd_th2d`, `thd_thd_thd`}
    q: torch.Tensor
        Query tensor. It may be different from input `q` as we try to fit tensors to
        a supported layout.
    k: torch.Tensor
        Key tensor. It may be different from input `k` as we try to fit tensors to
        a supported layout.
    v: torch.Tensor
        Value tensor. It may be different from input `v` as we try to fit tensors to
        a supported layout.
    """

    check_last_dim_contiguous = all(x.stride(-1) == 1 for x in [q, k, v])
    assert check_last_dim_contiguous, "q, k and v must have stride 1 in their last dimension!"

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
            qkv_layout = "_".join(list([qkv_format]) * 3)
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

    return qkv_layout, q, k, v


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


class FlashAttention(torch.nn.Module):
    """Dot product attention, using HazyResearch flash-attn package:
    https://github.com/Dao-AILab/flash-attention
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        if _flash_attn_is_installed:
            assert (
                _flash_attn_version >= _flash_attn_version_required
            ), f"FlashAttention minimum version {_flash_attn_version_required} is required."
            assert (
                _flash_attn_version <= _flash_attn_max_version
            ), f"FlashAttention maximum version {_flash_attn_max_version} is supported."

        self.softmax_scale = softmax_scale
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.layer_number = 1 if layer_number is None else layer_number
        self.deterministic = deterministic
        self.logger = logging.getLogger("FlashAttention")
        self.logger.setLevel(_log_level)
        if not self.logger.hasHandlers():
            self.logger.addHandler(_stream_handler)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers=None,
    ) -> torch.Tensor:
        """flash-attn fprop"""

        assert all(
            x.dtype in [torch.float16, torch.bfloat16] or isinstance(x, Float8Tensor)
            for x in [query_layer, key_layer, value_layer]
        ), "FlashAttention only supports FP16 and BF16 data types, or Float8Tensors."
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
        ), "FlashAttention currently only supports CUDA tensors."
        assert (
            qkv_layout in QKVLayouts
        ), f"FlashAttention does not support qkv_layout = {qkv_layout}!"

        cp_size = 1
        if isinstance(cp_group, dist_group_type):
            cp_size = get_distributed_world_size(cp_group)
        elif isinstance(cp_group, list):
            for group in cp_group:
                cp_size *= get_distributed_world_size(group)
        context_parallel = cp_size > 1

        qkv_format = "".join([i for i in qkv_layout.split("_")[0] if i.isalpha()])

        if all(not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]):
            if qkv_format == "sbhd":
                # For now just 128, will make it more general in the future
                if (
                    query_layer.shape[-1] == 128
                    and query_layer.shape[0] * query_layer.shape[1] >= 512
                    and qkv_layout == "sbh3d"
                ):
                    query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(
                        query_layer, key_layer, value_layer
                    )
                else:
                    query_layer, key_layer, value_layer = [
                        x.transpose(0, 1) for x in (query_layer, key_layer, value_layer)
                    ]
            if context_parallel:
                query_layer, key_layer, value_layer = [
                    x.contiguous() for x in (query_layer, key_layer, value_layer)
                ]
        else:
            if qkv_format == "sbhd":
                query_layer._data, key_layer._data, value_layer._data = [
                    x.transpose(0, 1)
                    for x in (query_layer._data, key_layer._data, value_layer._data)
                ]
                query_layer, key_layer, value_layer = [
                    Float8Tensor.make_like(x, data=x._data, shape=x._data.shape)
                    for x in (query_layer, key_layer, value_layer)
                ]
            if context_parallel:
                query_layer._data, key_layer._data, value_layer._data = [
                    x.contiguous() for x in (query_layer._data, key_layer._data, value_layer._data)
                ]

        batch_size = query_layer.shape[0]

        if qkv_format in ["sbhd", "bshd"]:
            max_seqlen_q, max_seqlen_kv = query_layer.shape[1], key_layer.shape[1]
            max_seqlen_q *= cp_size
            max_seqlen_kv *= cp_size

            if "padding" in attn_mask_type:
                assert not context_parallel, "Padding mask not supported with context parallelism!"
                # [b * s, h, d]
                query_layer, key_layer, value_layer = [
                    x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
                    for x in [query_layer, key_layer, value_layer]
                ]

                if self.attention_type == "self":
                    assert (
                        max_seqlen_q == max_seqlen_kv
                    ), "Maximum sequence length for Q and KV should be the same."
                    if cu_seqlens_q is None:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        cu_seqlens_q, indices_q = get_cu_seqlens_and_indices(attention_mask)
                    else:
                        indices_q = get_indices(max_seqlen_q, cu_seqlens_q)
                    cu_seqlens_kv = cu_seqlens_q
                    query_layer, key_layer, value_layer = PackTensors.apply(
                        indices_q, query_layer, key_layer, value_layer
                    )
                else:
                    if cu_seqlens_q is None or cu_seqlens_kv is None:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        cu_seqlens_q, indices_q = get_cu_seqlens_and_indices(attention_mask[0])
                        cu_seqlens_kv, indices_kv = get_cu_seqlens_and_indices(attention_mask[1])
                    else:
                        indices_q = get_indices(max_seqlen_q, cu_seqlens_q)
                        indices_kv = get_indices(max_seqlen_kv, cu_seqlens_kv)
                    query_layer = PackTensors.apply(indices_q, query_layer)
                    key_layer, value_layer = PackTensors.apply(indices_kv, key_layer, value_layer)
            else:
                # Cumulative sequence lengths for unpadded data
                if cu_seqlens_q is None:
                    cu_seqlens_q = _get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_q,
                        query_layer.device,
                    )
                if cu_seqlens_kv is None:
                    cu_seqlens_kv = _get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_kv,
                        key_layer.device,
                    )
        elif qkv_format == "thd":
            assert (
                cu_seqlens_q is not None and cu_seqlens_kv is not None
            ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            if max_seqlen_q is None:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                max_seqlen_q = seqlens_q.max().item()
            if max_seqlen_kv is None:
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                max_seqlen_kv = seqlens_kv.max().item()

        if context_parallel and all(
            not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]
        ):
            assert (
                alibi_slopes is None
            ), "Alibi slope bias addition is not supported with context parallelism."
            with self.attention_dropout_ctx():
                output = attn_forward_func_with_cp(
                    self.training,
                    query_layer,
                    key_layer,
                    value_layer,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q if qkv_format == "thd" else None,
                    cu_seqlens_kv if qkv_format == "thd" else None,
                    self.attention_dropout if self.training else 0.0,
                    cp_group,
                    cp_global_ranks,
                    cp_stream,
                    cp_comm_type,
                    softmax_scale=self.softmax_scale,
                    qkv_format="bshd" if qkv_format == "sbhd" else qkv_format,
                    attn_mask_type=attn_mask_type,
                    deterministic=self.deterministic,
                    window_size=window_size,
                    quantizers=quantizers,
                )
        else:

            from .cpu_offload import CPUOffloadEnabled

            if CPUOffloadEnabled:
                tensor_list = [query_layer, key_layer, value_layer, cu_seqlens_q, cu_seqlens_kv]
                for tensor in tensor_list:
                    if tensor is not None:
                        tensor.activation_offloading = True

            with self.attention_dropout_ctx():
                fa_optional_forward_kwargs = {}
                if _flash_attn_2_3_plus:
                    fa_optional_forward_kwargs["window_size"] = window_size
                if _flash_attn_2_4_plus:
                    fa_optional_forward_kwargs["alibi_slopes"] = alibi_slopes
                if _flash_attn_2_4_1_plus:
                    fa_optional_forward_kwargs["deterministic"] = self.deterministic
                fa_optional_forward_args_thd = []
                if qkv_format in ["bshd", "sbhd"] and "padding" not in attn_mask_type:
                    func = flash_attn_func if not _use_flash_attn_3 else flash_attn_func_v3
                else:
                    if _flash_attn_2_5_7_plus:
                        fa_optional_forward_kwargs["block_table"] = None
                    func = (
                        flash_attn_varlen_func
                        if not _use_flash_attn_3
                        else flash_attn_varlen_func_v3
                    )
                    fa_optional_forward_args_thd.append(cu_seqlens_q)
                    fa_optional_forward_args_thd.append(cu_seqlens_kv)
                    fa_optional_forward_args_thd.append(max_seqlen_q)
                    fa_optional_forward_args_thd.append(max_seqlen_kv)
                if _use_flash_attn_3:
                    fa_3_optional_forward_kwargs = {}
                    fa_3_optional_forward_kwargs["window_size"] = window_size
                    fa_3_optional_forward_kwargs["deterministic"] = self.deterministic
                    if fp8:
                        QKV_quantizer = quantizers["scaling_fwd"][META_QKV]
                        torch_dtype = get_fp8_torch_dtype(fp8_meta["recipe"], fprop_tensor=True)
                        torch_orig_dtype = query_layer.dtype

                        def convert_to_torch_float8(tensor, dtype):
                            out = torch.Tensor().to(device=tensor.device, dtype=dtype)
                            out.set_(
                                tensor._data.untyped_storage(),
                                tensor._data.storage_offset(),
                                tensor._data.shape,
                                tensor._data.stride(),
                            )
                            return out

                        # "fp8_mha" decides outputs in fp8, while inputs are inferred from
                        # the real dtype
                        assert isinstance(key_layer, query_layer.__class__) and isinstance(
                            value_layer, query_layer.__class__
                        ), "q, k, and v must have the same type."
                        if not isinstance(query_layer, Float8Tensor):
                            query_layer, key_layer, value_layer = (
                                QKV_quantizer(x) for x in [query_layer, key_layer, value_layer]
                            )
                        fa_3_optional_forward_kwargs["descale_q"] = (
                            query_layer._scale_inv.unsqueeze(0)
                        )
                        fa_3_optional_forward_kwargs["descale_k"] = key_layer._scale_inv.unsqueeze(
                            0
                        )
                        fa_3_optional_forward_kwargs["descale_v"] = (
                            value_layer._scale_inv.unsqueeze(0)
                        )
                        query_layer, key_layer, value_layer = (
                            convert_to_torch_float8(x, torch_dtype)
                            for x in [query_layer, key_layer, value_layer]
                        )
                    try:
                        output, _ = func(
                            query_layer,
                            key_layer,
                            value_layer,
                            *fa_optional_forward_args_thd,
                            softmax_scale=self.softmax_scale,
                            causal="causal" in attn_mask_type,
                            **fa_3_optional_forward_kwargs,
                        )
                    except TypeError as e:
                        if _flash_attn_3_0_0_beta:
                            e.args = (
                                e.args[0]
                                + ". Please update your flash-attn v3 (beta) installation as it "
                                + "may have added more supported arguments to its API. \n"
                                + _flash_attn_3_installation_steps,
                            ) + e.args[1:]
                        raise

                    if fp8:
                        output = output.to(dtype=torch_orig_dtype)
                    if fp8 and fp8_meta["recipe"].fp8_mha:
                        O_quantizer = quantizers["scaling_fwd"][META_O]
                        output = O_quantizer(output)
                else:
                    output = func(
                        query_layer,
                        key_layer,
                        value_layer,
                        *fa_optional_forward_args_thd,
                        self.attention_dropout if self.training else 0.0,
                        softmax_scale=self.softmax_scale,
                        causal="causal" in attn_mask_type,
                        **fa_optional_forward_kwargs,
                    )

        if qkv_format in ["sbhd", "bshd"] and "padding" in attn_mask_type:
            output = UnpackTensor.apply(indices_q, batch_size * max_seqlen_q, output)

        if qkv_format == "sbhd":
            # (bs)hd -> bs(hd) -> sb(hd)
            if fp8 and fp8_meta["recipe"].fp8_mha:
                output_data = (
                    output._data.reshape(batch_size, max_seqlen_q // cp_size, -1)
                    .transpose(0, 1)
                    .contiguous()
                )
                output = Float8Tensor.make_like(
                    output,
                    data=output_data,
                    shape=output_data.shape,
                )
            else:
                output = output.view(batch_size, max_seqlen_q // cp_size, -1).transpose(0, 1)
        elif qkv_format == "bshd":
            # (bs)hd -> bs(hd)
            output = output.reshape(batch_size, max_seqlen_q // cp_size, -1)
        elif qkv_format == "thd":
            # thd -> t(hd)
            output = output.reshape(output.shape[0], -1)

        return output.contiguous()


def _combine_tensors(
    tensors: List[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    """Combine tensors along a particular dimension"""

    num_tensors = len(tensors)
    new_shape = list(tensors[0].shape)
    new_shape.insert(dim, num_tensors)
    if isinstance(tensors[0], Float8Tensor):
        new_stride = list(tensors[0]._data.stride())
        new_stride.insert(dim, int(new_stride[dim - 1] / num_tensors))
        combined_tensor = torch.Tensor().to(device=tensors[0].device, dtype=tensors[0]._data.dtype)
        combined_tensor.set_(
            tensors[0]._data.untyped_storage(),
            tensors[0]._data.storage_offset(),
            new_shape,
            new_stride,
        )
        combined_tensor = Float8Tensor.make_like(tensors[0], data=combined_tensor, shape=new_shape)
    else:
        new_stride = list(tensors[0].stride())
        new_stride.insert(dim, int(new_stride[dim - 1] / num_tensors))
        combined_tensor = torch.Tensor().to(device=tensors[0].device, dtype=tensors[0].dtype)
        combined_tensor.set_(
            tensors[0].untyped_storage(), tensors[0].storage_offset(), new_shape, new_stride
        )

    return combined_tensor


class FusedAttnFunc(torch.autograd.Function):
    """Function for FusedAttention with separate Q, K, V tensors"""

    @staticmethod
    def forward(
        ctx,
        is_training,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        q,
        k,
        v,
        qkv_dtype,
        attn_bias,
        attn_scale,
        dropout_p,
        fast_zero_fill,
        qkv_layout,
        attn_bias_type,
        attn_mask_type,
        window_size,
        rng_gen,
        fused_attention_backend,
        use_FAv2_bwd,
        fp8,
        fp8_meta,
        quantizers,
        deterministic,
    ):
        # pylint: disable=missing-function-docstring
        # "fp8_mha" decides outputs in fp8, while inputs are inferred from the real dtype
        is_input_fp8 = False
        is_output_fp8 = fp8_meta["recipe"].fp8_mha if "recipe" in fp8_meta else False
        fake_dtype = q.dtype

        QKV_quantizer, O_quantizer, S_quantizer, dQKV_quantizer, dO_quantizer, dP_quantizer = (
            get_attention_quantizers(fp8, quantizers, cp_specific_quantizers=False)
        )
        if fp8:
            fused_attention_backend = FusedAttnBackend["FP8"]
            assert isinstance(k, q.__class__) and isinstance(
                v, q.__class__
            ), "q, k, and v must have the same type."

            is_input_fp8 = isinstance(q, Float8Tensor)
            q_fp8, k_fp8, v_fp8 = None, None, None
            if is_input_fp8:
                q_fp8, k_fp8, v_fp8 = q, k, v
            else:
                # 1: qkv packed, 2: kv packed, 3: qkv separate
                qkv_group = len(qkv_layout.split("_"))
                match qkv_group:
                    case 1:
                        dim = qkv_layout.find("3")
                        qkv = _combine_tensors([q, k, v], dim)
                        qkv_c = qkv.view(-1, qkv.shape[-3] * qkv.shape[-2] * qkv.shape[-1])
                        qkv_fp8 = QKV_quantizer(qkv)
                        q_fp8, k_fp8, v_fp8 = _SplitAlongDim.apply(qkv_fp8, dim, [1, 1, 1], True)
                    case 2:
                        q_fp8 = QKV_quantizer(q)
                        dim = qkv_layout.split("_")[1].find("2")
                        kv = _combine_tensors([k, v], dim)
                        kv_c = kv.view(-1, kv.shape[-3] * kv.shape[-2] * kv.shape[-1])
                        kv_fp8 = QKV_quantizer(kv_c)
                        k_fp8, v_fp8 = _SplitAlongDim.apply(kv_fp8, dim, [1, 1], True)
                    case 3:
                        q_fp8 = QKV_quantizer(q)
                        k_fp8 = QKV_quantizer(k)
                        v_fp8 = QKV_quantizer(v)
                    case _:
                        raise "Invalid qkv_layout " + qkv_layout
            out_fp8, aux_ctx_tensors = fused_attn_fwd(
                is_training,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q_fp8,
                k_fp8,
                v_fp8,
                fake_dtype,
                fused_attention_backend,
                attn_bias,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                S_quantizer,
                O_quantizer,
                attn_scale,
                dropout_p,
                fast_zero_fill,
                qkv_layout,
                attn_bias_type,
                attn_mask_type,
                window_size,
                rng_gen,
            )
            if is_output_fp8:
                out_ret = out_fp8
            else:
                out_ret = out_fp8.dequantize().view(out_fp8.shape)
            out_save = out_ret

            if not int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                # 1: qkv packed, 2: kv packed, 3: qkv separate
                if is_input_fp8:
                    qkv_group = len(qkv_layout.split("_"))
                    if qkv_group == 1:
                        dim = qkv_layout.find("3")
                        qkv = _combine_tensors([q, k, v], dim)
                        qkv_c = qkv.view(-1, qkv.shape[-3] * qkv.shape[-2] * qkv.shape[-1])
                        qkv_no_fp8 = qkv_c.dequantize().view(qkv.shape)
                        q, k, v = _SplitAlongDim.apply(qkv_no_fp8, dim, [1, 1, 1], True)
                    if qkv_group == 2:
                        q = q.dequantize()
                        dim = qkv_layout.split("_")[1].find("2")
                        kv = _combine_tensors([k, v], dim)
                        kv_c = kv.view(-1, kv.shape[-3] * kv.shape[-2] * kv.shape[-1])
                        kv_no_fp8 = kv.dequantize()
                        k, v = _SplitAlongDim.apply(kv_no_fp8, dim, [1, 1], True)
                    if qkv_group == 3:
                        q = q.dequantize()
                        k = k.dequantize()
                        v = v.dequantize()
                if is_output_fp8:
                    out_save = out_fp8.dequantize()

            fp8_tensors = (q_fp8, k_fp8, v_fp8, out_fp8)
        else:

            out_ret, aux_ctx_tensors = fused_attn_fwd(
                is_training,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q,
                k,
                v,
                fake_dtype,
                fused_attention_backend,
                attn_bias,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                None,  # s_quantizer
                None,  # o_quantizer
                attn_scale,
                dropout_p,
                fast_zero_fill,
                qkv_layout,
                attn_bias_type,
                attn_mask_type,
                window_size,
                rng_gen,
            )
            out_save = out_ret
            fp8_tensors = (None, None, None, None)

        ctx.fp8 = fp8 and int(os.getenv("NVTE_FP8_DPA_BWD", "1"))

        from .cpu_offload import CPUOffloadEnabled

        if CPUOffloadEnabled:
            if ctx.fp8:
                tensor_list = fp8_tensors
            else:
                tensor_list = [q, k, v, out_save]

            tensor_list.extend(aux_ctx_tensors)

            qkv_layout = "sbhd_sbhd_sbhd"
            for tensor in tensor_list:
                if tensor is not None:
                    tensor.activation_offloading = True

        ctx.is_input_fp8 = is_input_fp8
        ctx.is_output_fp8 = is_output_fp8
        qkvo_tensors = (q, k, v, out_save) if not ctx.fp8 else (None, None, None, None)
        tensors_to_save, tensor_objects = prepare_for_saving(
            *fp8_tensors,
            *qkvo_tensors,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *aux_ctx_tensors,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.fp8_meta = fp8_meta

        ctx.dQKV_quantizer = dQKV_quantizer
        ctx.dO_quantizer = dO_quantizer
        ctx.dP_quantizer = dP_quantizer
        ctx.S_quantizer = S_quantizer

        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.fake_dtype = fake_dtype
        ctx.qkv_dtype = qkv_dtype
        ctx.attn_scale = attn_scale
        ctx.dropout_p = dropout_p
        ctx.fast_zero_fill = fast_zero_fill
        ctx.qkv_layout = qkv_layout
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.window_size = window_size
        ctx.fused_attention_backend = (
            fused_attention_backend if ctx.fp8 else FusedAttnBackend["F16_arbitrary_seqlen"]
        )
        ctx.use_FAv2_bwd = use_FAv2_bwd
        ctx.deterministic = deterministic

        return out_ret

    @staticmethod
    def backward(ctx, d_out):
        # pylint: disable=missing-function-docstring
        if ctx.is_output_fp8:
            assert isinstance(
                d_out, Float8Tensor
            ), "Gradient of the DPA output must be in Float8Tensor type for FP8 MHA."

        d_out = d_out.contiguous()
        (
            q_fp8,
            k_fp8,
            v_fp8,
            out_fp8,
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *other_tensors,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

        aux_ctx_tensors = other_tensors

        if not aux_ctx_tensors[0].is_contiguous():
            aux_ctx_tensors[0] = aux_ctx_tensors[0].contiguous()
        rest = [None]
        if ctx.use_FAv2_bwd:
            softmax_lse, rng_state = aux_ctx_tensors
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            d_out, q, k, v, out = [maybe_contiguous(x) for x in (d_out, q, k, v, out)]
            flash_attn_cuda_bwd(
                d_out,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_kv,
                ctx.max_seqlen_q,
                ctx.max_seqlen_kv,
                ctx.dropout_p,
                ctx.attn_scale,
                False,
                "causal" in ctx.attn_mask_type,
                None,
                rng_state,
            )
            dq = dq[..., : d_out.shape[-1]]
            dk = dk[..., : d_out.shape[-1]]
            dv = dv[..., : d_out.shape[-1]]
        else:
            with torch.cuda.nvtx.range("_FusedAttn"):
                if ctx.fp8:
                    if ctx.is_output_fp8:
                        d_out_fp8 = d_out
                    else:
                        d_out_fp8 = ctx.dO_quantizer(d_out)
                    dq_fp8, dk_fp8, dv_fp8, *rest = fused_attn_bwd(
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_kv,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        q_fp8,
                        k_fp8,
                        v_fp8,
                        out_fp8,
                        d_out_fp8,
                        ctx.fake_dtype,
                        ctx.qkv_dtype,
                        aux_ctx_tensors,
                        ctx.fused_attention_backend,
                        cu_seqlens_q_padded,
                        cu_seqlens_kv_padded,
                        ctx.S_quantizer,
                        ctx.dP_quantizer,
                        ctx.dQKV_quantizer,
                        ctx.attn_scale,
                        ctx.dropout_p,
                        ctx.fast_zero_fill,
                        ctx.qkv_layout,
                        ctx.attn_bias_type,
                        ctx.attn_mask_type,
                        ctx.window_size,
                        ctx.deterministic,
                    )

                    if not ctx.is_input_fp8:
                        qkv_group = len(ctx.qkv_layout.split("_"))
                        if qkv_group == 1:
                            dim = ctx.qkv_layout.find("3")
                            dqkv_fp8_data = _combine_tensors(
                                [dq_fp8._data, dk_fp8._data, dv_fp8._data], dim
                            )
                            dqkv_fp8 = dq_fp8.make_like(
                                tensor=dq_fp8, data=dqkv_fp8_data, shape=dqkv_fp8_data.shape
                            )
                            dqkv = dqkv_fp8.dequantize()
                            dq, dk, dv = _SplitAlongDim.apply(dqkv, dim, [1, 1, 1], True)
                        if qkv_group == 2:
                            dq = dq_fp8.dequantize()
                            dim = ctx.qkv_layout.split("_")[1].find("2")
                            dkv_fp8 = _combine_tensors([dk_fp8, dv_fp8], dim)
                            dkv_c_fp8 = dkv_fp8.view(
                                -1, dkv_fp8.shape[-3] * dkv_fp8.shape[-2] * dkv_fp8.shape[-1]
                            )
                            dkv = dkv_c_fp8.dequantize()
                            dk, dv = _SplitAlongDim.apply(dkv, dim, [1, 1], True)
                        if qkv_group == 3:
                            dq = dq_fp8.dequantize()
                            dk = dk_fp8.dequantize()
                            dv = dv_fp8.dequantize()
                    else:
                        dq, dk, dv = dq_fp8, dk_fp8, dv_fp8
                else:
                    if isinstance(d_out, QuantizedTensor):
                        d_out = d_out.dequantize()
                    dq, dk, dv, *rest = fused_attn_bwd(
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_kv,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        q,
                        k,
                        v,
                        out,
                        d_out,
                        ctx.fake_dtype,
                        ctx.qkv_dtype,
                        aux_ctx_tensors,
                        ctx.fused_attention_backend,
                        cu_seqlens_q_padded,
                        cu_seqlens_kv_padded,
                        None,
                        None,
                        None,
                        ctx.attn_scale,
                        ctx.dropout_p,
                        ctx.fast_zero_fill,
                        ctx.qkv_layout,
                        ctx.attn_bias_type,
                        ctx.attn_mask_type,
                        ctx.window_size,
                        ctx.deterministic,
                    )

        # if no_bias or alibi, return dqkv
        if ctx.attn_bias_type in ["no_bias", "alibi"]:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                dq,
                dk,
                dv,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        # else, return (dqkv, dbias)
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dq,
            dk,
            dv,
            None,
            rest[0],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FusedAttention(torch.nn.Module):
    """Dot product attention, with multiple backends:

    1. FusedAttnBackend["F16_max512_seqlen"]
       cuDNN based fused attention for FP16/BF16 and <=512 sequence length.
    2. FusedAttnBackend["F16_arbitrary_seqlen"]
       cuDNN based fused attention for FP16/BF16 and any sequence length.

    Support matrix:

    | backend       | 1                       | 2                              |
    | flash based   | no                      | yes                            |
    | cuDNN based   | yes                     | yes                            |
    | qkv dtype     | fp16/bf16               | fp16/bf16                      |
    | attn_type     | self/cross              | self/cross                     |
    | qkv_layout    |                         |                                |
    |  - (q,k,v)    | sb3hd, bs3hd            | sb3hd, bs3hd, sbh3d, bsh3d     |
    |               | sbhd_sb2hd, bshd_bs2hd  | sbhd_sb2hd, bshd_bs2hd         |
    |               | bshd_bshd_bshd          | sbhd_sbh2d, bshd_bsh2d         |
    |               |                         | sbhd_sbhd_sbhd, bshd_bshd_bshd |
    | mask_type     | causal/padding/no_mask  | causal/padding/no_mask         |
    | bias_type     | post_scale_bias/no_bias | post_scale_bias/alibi/no_bias  |
    | dropout       | yes                     | yes                            |
    | max_seqlen    | <=512, multiple of 64   | any, multiple of 64            |
    | head_dim      | 64                      | <=128, multiple of 8           |
    | output dtype  | fp16/bf16               | fp16/bf16                      |
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_type = attention_type
        self.use_FAv2_bwd = os.getenv(
            "NVTE_FUSED_ATTN_USE_FAv2_BWD", "0"
        ) == "1" and get_device_compute_capability() == (9, 0)
        self.layer_number = 1 if layer_number is None else layer_number
        self.deterministic = deterministic

        def remove_extra_states_check(self, incompatible_keys):  # pylint: disable=unused-argument
            """
            Temporarily remove fused_attention._extra_state as a missing key
            or an unexpected key when loading Transformer Engine checkpoints.
            Please store FP8 metadata as DotProductAttention's _extra_state,
            rather than FusedAttention's _extra_state. This hook will be
            phased out in Transformer Engine 2.0.
            """
            for key in incompatible_keys.missing_keys:
                if "fused_attention._extra_state" in key:
                    incompatible_keys.missing_keys.remove(key)
            for key in incompatible_keys.unexpected_keys:
                if "fused_attention._extra_state" in key:
                    incompatible_keys.unexpected_keys.remove(key)
                    warnings.warn(
                        "fused_attention._extra_state is not loaded from checkpoint. Please map "
                        "FusedAttention's _extra_state to DotProductAttention's _extra_state."
                    )

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    @no_torch_dynamo()
    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        cu_seqlens_q_padded: Optional[torch.Tensor] = None,
        cu_seqlens_kv_padded: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        window_size: Optional[Tuple[int, int]] = None,
        fused_attention_backend: tex.NVTE_Fused_Attn_Backend = tex.NVTE_Fused_Attn_Backend.NVTE_No_Backend,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers=None,
    ) -> torch.Tensor:
        """fused attention fprop"""
        assert (
            fused_attention_backend != tex.NVTE_Fused_Attn_Backend.NVTE_No_Backend
        ), "No fused attention backend supports this input combination!"
        assert all(
            x.dtype in [torch.float16, torch.bfloat16] or isinstance(x, Float8Tensor)
            for x in [query_layer, key_layer, value_layer]
        ), "FusedAttention only supports FP16 and BF16 data types, or Float8Tensors."
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
        ), "FusedAttention only supports CUDA tensors."
        assert (
            qkv_layout in QKVLayouts
        ), f"FusedAttention does not support qkv_layout = {qkv_layout}!"

        cp_size = 1
        if isinstance(cp_group, dist_group_type):
            cp_size = get_distributed_world_size(cp_group)
        elif isinstance(cp_group, list):
            for group in cp_group:
                cp_size *= get_distributed_world_size(group)
        context_parallel = cp_size > 1

        qkv_format = "".join([i for i in qkv_layout.split("_")[0] if i.isalpha()])

        if qkv_format in ["sbhd", "bshd"]:
            if qkv_format == "sbhd":
                batch_size, max_seqlen_q, max_seqlen_kv = (
                    query_layer.shape[1],
                    query_layer.shape[0],
                    key_layer.shape[0],
                )
            if qkv_format == "bshd":
                batch_size, max_seqlen_q, max_seqlen_kv = (
                    query_layer.shape[0],
                    query_layer.shape[1],
                    key_layer.shape[1],
                )
            max_seqlen_q *= cp_size
            max_seqlen_kv *= cp_size
            if "padding" in attn_mask_type:
                assert not context_parallel, "Padding mask not supported with context parallelism!"

                if cu_seqlens_q is None or cu_seqlens_kv is None:
                    if attention_mask is None:
                        raise RuntimeError(
                            "Please provide attention_mask or cu_seqlens for padding!"
                        )
                    if self.attention_type == "self":
                        cu_seqlens_q = get_cu_seqlens(attention_mask)
                        cu_seqlens_kv = cu_seqlens_q
                    else:
                        cu_seqlens_q = get_cu_seqlens(attention_mask[0])
                        cu_seqlens_kv = get_cu_seqlens(attention_mask[1])
            else:
                if cu_seqlens_q is None:
                    cu_seqlens_q = _get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_q,
                        query_layer.device,
                    )
                if cu_seqlens_kv is None:
                    cu_seqlens_kv = _get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_kv,
                        key_layer.device,
                    )
        if qkv_format == "thd":
            assert (
                max_seqlen_q is not None
                and max_seqlen_kv is not None
                and cu_seqlens_q is not None
                and cu_seqlens_kv is not None
            ), "max_seqlen_q/kv and cu_seqlens_q/kv can not be None when qkv_format is thd!"

        if qkv_format == "thd" and (cu_seqlens_q_padded is None or cu_seqlens_kv_padded is None):
            cu_seqlens_q_padded = cu_seqlens_q
            cu_seqlens_kv_padded = cu_seqlens_kv

        qkv_dtype = TE_DType[query_layer.dtype]

        use_FAv2_bwd = (
            self.use_FAv2_bwd
            and (core_attention_bias_type == "no_bias")
            and (fused_attention_backend == tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen)
        )

        if fp8:
            assert fused_attention_backend == tex.NVTE_Fused_Attn_Backend.NVTE_FP8, (
                f"cuDNN attention sub-backend {int(tex.NVTE_Fused_Attn_Backend.NVTE_FP8)}"
                " is required for FP8 attention!"
            )
            assert fp8_meta is not None, "FP8 metadata fp8_meta is required for FP8 attention!"
            assert not context_parallel or fp8_meta["recipe"].reduce_amax, (
                "Amax reduction across TP+CP group is necessary when using context parallelism with"
                " FP8!"
            )

        if context_parallel:
            assert (
                fp8
                or fused_attention_backend == tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen
            ), f"{fused_attention_backend} does not work with context parallelism!"
            assert core_attention_bias_type not in [
                "alibi"
            ], f"{core_attention_bias_type} is not supported with context parallelism!"
            query_layer, key_layer, value_layer = [
                x.contiguous() for x in (query_layer, key_layer, value_layer)
            ]
            with self.attention_dropout_ctx():
                output = attn_forward_func_with_cp(
                    self.training,
                    query_layer,
                    key_layer,
                    value_layer,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    self.attention_dropout if self.training else 0.0,
                    cp_group,
                    cp_global_ranks,
                    cp_stream,
                    cp_comm_type,
                    softmax_scale=self.softmax_scale,
                    qkv_format=qkv_format,
                    attn_mask_type=attn_mask_type,
                    attn_bias_type=core_attention_bias_type,
                    attn_bias=core_attention_bias,
                    deterministic=self.deterministic,
                    use_fused_attention=True,
                    window_size=window_size,
                    fp8=fp8,
                    fp8_meta=fp8_meta,
                    quantizers=quantizers,
                )
        else:
            with self.attention_dropout_ctx():
                output = FusedAttnFunc.apply(
                    self.training,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_dtype,
                    core_attention_bias,
                    self.softmax_scale,
                    self.attention_dropout if self.training else 0.0,
                    fast_zero_fill,
                    qkv_layout,
                    core_attention_bias_type,
                    attn_mask_type,
                    window_size,
                    None,  # rng_gen
                    fused_attention_backend,
                    use_FAv2_bwd,
                    fp8,
                    fp8_meta,
                    quantizers,
                    self.deterministic,
                )

        # ...hd -> ...(hd)
        return output.view(*output.shape[:-2], -1)


class DotProductAttention(TransformerEngineBaseModule):
    """Allows the model to jointly attend to information from different
    representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. note::

        Argument :attr:`attention_mask` in the `forward` call is only used when
        :attr:`attn_mask_type` includes '"padding"' or `"arbitrary"`.

    .. warning::

        FlashAttention uses a non-deterministic algorithm for optimal performance. To observe
        deterministic behavior at the cost of performance, use FlashAttention version >= `2.4.1`
        and set the environment variable :attr:`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`. In order
        to disable`flash-attn` entirely, set :attr:`NVTE_FLASH_ATTN=0`.

    .. note::

        Transformer Engine stores the FP8 metadata under a `._extra_state` key when checkpointing.
        As the FP8 attention support expands from one backend to multiple backends, the location
        of that key has also shifted (see `FP8 checkpoint compatibility <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/faq.html#fp8-checkpoint-compatibility>`_).


    Parameters
    ----------
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels : Union[int, Tuple[int, int]]
                the head size in key and value tensors. If the same, :attr:`kv_channels` can be
                an integer; if not, :attr:`kv_channels` should be a tuple of two integers.
    num_gqa_groups : Optional[int] = None
                    number of GQA groups in the transformer layer.
                    Grouped Query Attention is described in
                    `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                    This only affects the keys and values, not the queries.
                    GQA-1 is equivalent to Multi-Query Attention
                    (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                    is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    attention_dropout: float, default = 0.0
                      dropout probability for the dropout op during multi-head attention.
    attn_mask_type: str, default = `causal`
                   type of attention mask passed into softmax operation, options are "`no_mask`",
                   "`padding`", "`causal`", "`padding,causal`", "`causal,padding`",
                   "`padding_causal`", "`causal_bottom_right`", "`padding_causal_bottom_right`", and
                   "`arbitrary`", where "`padding,causal`", "`causal,padding`" and "`padding_causal`"
                   are equivalent. This arg can be overridden by :attr:`attn_mask_type` in the
                   `forward` method. It is useful for cases involving compilation/tracing, e.g.
                   ONNX export, and the forward arg is useful for dynamically changing mask types,
                   e.g. a different mask for training and inference.
                   1. For "`no_mask`", no attention mask is applied.
                   2. For "`causal`", "`causal_bottom_right`", or the causal mask in
                   "`padding_causal`" and "`padding_causal_bottom_right`", Transformer Engine
                   calculates and applies an upper triangular mask to the softmax input.
                   No user input is needed. Causal masks without the "`bottom_right`" appendix align
                   the diagonal line to the top left corner of the softmax matrix. With
                   "`bottom_right`", the causal mask is aligned to the bottom right corner, which is
                   often used in inference/KV caching.
                   3. For "`padding`", or the padding mask in "`padding_causal`" and
                   "`padding_causal_bottom_right`", users need to provide the locations of padded
                   tokens, either via :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv` (both in shape
                   [batch_size + 1]), or via :attr:`attention_mask` (one tensor for self-attention
                   in shape [batch_size, 1, 1, max_seqlen_q], or two tensors in a tuple for
                   cross-attention in shapes [batch_size, 1, 1, max_seqlen_q] and
                   [batch_size, 1, 1, max_seqlen_kv]).
                   4. For "`arbitrary`", users need to provide a mask that is broadcastable to
                   the shape of softmax input [batch_size, num_heads, max_seqlen_q, max_seqlen_kv].
    window_size: Optional[Tuple[int, int]], default = `None`
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically. Both `causal` and `causal_bottom_right` masks
                map to `window_size = (-1, 0)` and Transformer Engine distinguishes them based on
                `attn_mask_type`. Similar to :attr:`attn_mask_type`, `window_size` can
                be overridden by :attr:`window_size` in `forward` as well.
    attention_type: str, default = `self`
                   type of attention, either "`self`" and "`cross`".
    layer_number: int, default = `None`
                 layer number of the current `DotProductAttention` when multiple such modules
                 are concatenated, for instance in consecutive transformer blocks.
    qkv_format: str, default = `sbhd`
               dimension format for `query_layer`, `key_layer` and `value_layer`,
               {`sbhd`, `bshd`, `thd`}. `s` stands for the sequence length, `b` batch size,
               `h` the number of heads, `d` head size, and `t` the total number of tokens
               in a batch, with `t = sum(s_i), for i = 0...b-1`. `sbhd` and `bshd` formats
               are used for when sequences in a batch are of equal length or padded to
               equal length, and the `thd` format is used for when sequences in a batch
               have different lengths. Please note that these formats do not reflect how
               tensors `query_layer`, `key_layer`, `value_layer` are laid out in memory.
               For that, please use `get_qkv_layout` to gain the layout information.
    softmax_scale: Optional[float], default = `None`
                softmax scale for the attention scores. If `None`, defaults to
                `1.0/math.sqrt(kv_channels if isinstance(kv_channels, int) else kv_channels[0])`.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_size : int, default = 1
             tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    cp_group : Union[ProcessGroup, List[ProcessGroup]], default = `None`
              context parallel process group.
              ProcessGroup is for cp_comm_type of "p2p", "all_gather", and "a2a".
              List[ProcessGroup] is for cp_comm_type of "a2a+p2p", where cp_group[0]
              and cp_group[1] are for a2a and p2p communications respectively.
    cp_global_ranks : list of global rank IDs, default = `None`
                     global rank IDs of GPUs that are in cp_group.
    cp_stream : CUDA stream, default = `None`
               context parallelism splits flash attention into multiple steps for
               compute and communication overlapping. To address the wave quantization
               issue of each split step, we add an additional CUDA stream so that we
               can overlap two flash attention kernels.
    cp_comm_type : str, default = `p2p`
                  inter-gpu communication type for context parallelism.
                  Can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
                  "p2p": Exchange KV chunks with P2P communications in ring topology.
                         P2P is async and can be overlapped with attention compute.
                  "all_gather": All-gather to get full sequence of KV before attention.
                                The all-gather is not async, and cannot be overlapped.
                  "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP
                         group, and gather to get full sequence of QKV.
                  "a2a+p2p": hierarchical CP implementation. First applying a2a to QKV
                  across each CP sub-group (e.g., via NVLink), then exchanging KV with
                  p2p between sub-groups (e.g., via IBLink).
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],
        num_gqa_groups: Optional[int] = None,
        attention_dropout: float = 0.0,
        qkv_format: str = "sbhd",
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        sequence_parallel: bool = False,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        attention_type: str = "self",
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        softmax_scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.logger = logging.getLogger("DotProductAttention")
        self.logger.setLevel(_log_level)
        if not self.logger.hasHandlers():
            self.logger.addHandler(_stream_handler)
        self.qkv_format = qkv_format
        attn_mask_type = attn_mask_type.replace(",", "_")
        if attn_mask_type == "causal_padding":
            attn_mask_type = "padding_causal"
        self.attn_mask_type = attn_mask_type
        self.window_size = check_set_window_size(attn_mask_type, window_size)
        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.get_rng_state_tracker = get_rng_state_tracker
        self.num_attention_heads = num_attention_heads
        self.layer_number = 1 if layer_number is None else layer_number
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream
        self.cp_comm_type = cp_comm_type

        self.hidden_size_per_attention_head_k = (
            kv_channels if isinstance(kv_channels, int) else kv_channels[0]
        )
        self.hidden_size_per_attention_head_v = (
            kv_channels if isinstance(kv_channels, int) else kv_channels[1]
        )

        self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // self.tp_size)

        assert (
            num_attention_heads % self.num_gqa_groups == 0
        ), "The number of attention heads must be divisible by the number of GQA groups!"

        self.rng_states_tracker = None
        if sequence_parallel or get_rng_state_tracker is None:
            attention_dropout_ctx = nullcontext
        else:
            self.rng_states_tracker = get_rng_state_tracker()
            set_all_rng_states(self.rng_states_tracker.get_states())
            attention_dropout_ctx = self.rng_states_tracker.fork

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                kv_channels if isinstance(kv_channels, int) else kv_channels[0]
            )

        self.deterministic = (
            not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
            or torch.are_deterministic_algorithms_enabled()
        )
        # To use the workspace optimization path for determinism, please
        # set NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT=1 for cuDNN >=8.9.5 and <9.0.0,
        # and set NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 for cuDNN >=9.0.0.
        cudnn_version = get_cudnn_version()
        if (8, 9, 5) <= cudnn_version < (9, 0, 0):
            if self.deterministic:
                os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1"

            # CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT
            # - unset:       enables workspace optimization when required workspace is <= 256MB
            #                or when bias gradient needs to be computed
            # - n:           enables workspace optimization when required workspace is <= n bytes
            # - -1:          enables workspace optimization always
            # - 0:           disables workspace optimization always
            if "NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT" in os.environ:
                if os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] == "0":
                    os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = "0"
                if os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] == "1":
                    os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = "-1"

        assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"

        self.attention_type = attention_type
        self.attention_dropout = attention_dropout

        attn_kwargs = {
            "attention_dropout": attention_dropout,
            "attention_dropout_ctx": attention_dropout_ctx,
        }

        self.flash_attention = FlashAttention(
            softmax_scale,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=self.deterministic,
            **attn_kwargs,
        )

        # Instantiating three types since use of flash-attn and FusedAttention
        # might be ruled out due to forward inputs.
        self.fused_attention = FusedAttention(
            softmax_scale,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=self.deterministic,
            **attn_kwargs,
        )

        self.unfused_attention = UnfusedDotProductAttention(
            softmax_scale,
            attention_type=attention_type,
            **attn_kwargs,
            layer_number=layer_number,
        )

        def remove_extra_states_check(self, incompatible_keys):  # pylint: disable=unused-argument
            """
            Temporarily remove core_attention._extra_state as a missing key
            when loading older Transformer Engine checkpoints. Will phase out
            this hook in Transformer Engine 2.0.
            """
            for key in incompatible_keys.missing_keys:
                if "core_attention._extra_state" in key:
                    incompatible_keys.missing_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        This function helps to load Transformer Engine 1.6 and 1.7 checkpoints, where FP8 attention
        metadata is stored under the `core_attention.fused_attention._extra_state` key and not the
        `core_attention._extra_state` key. Please see `FP8 checkpoint compatibility
        <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/faq.html#fp8-checkpoint-compatibility>`_ for more details.
        """
        fused_attn_key = False
        dot_product_attn_key = False
        for k in state_dict.keys():
            if "core_attention.fused_attention._extra_state" in k:
                fused_attn_key = True
            if "core_attention._extra_state" in k:
                dot_product_attn_key = True
        if fused_attn_key and not dot_product_attn_key:
            prefix = prefix + "fused_attention."
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _checkpointed_attention_forward(
        self,
        attention_func: Callable,
        *forward_args: Tuple[torch.Tensor, ...],
        **forward_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Forward method with activation checkpointing."""

        def custom_forward(*input_args, **input_kwargs):
            return attention_func(*input_args, **input_kwargs)

        hidden_states = checkpoint(
            custom_forward,
            distribute_saved_activations=False,
            get_rng_state_tracker=self.get_rng_state_tracker,
            tp_group=self.tp_group,
            *forward_args,
            **forward_kwargs,
        )

        return hidden_states

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, List[dist_group_type], None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
        cp_comm_type: str = "p2p",
    ) -> None:
        """
        Set the context parallel attributes for the given
        module before executing the forward pass.

        Parameters
        ----------
        cp_group : Union[ProcessGroup, List[ProcessGroup]]
                  context parallel process group.
                  ProcessGroup is for cp_comm_type of "p2p", "all_gather", and "a2a".
                  List[ProcessGroup] is for cp_comm_type of "a2a+p2p", where cp_group[0]
                  and cp_group[1] are for a2a and p2p communications respectively.
        cp_global_ranks : List[int]
                         list of global ranks in the context group.
        cp_stream : torch.cuda.Stream
                   cuda stream for context parallel execution.
        cp_comm_type : str, default = `p2p`
                      inter-gpu communication type for context parallelism.
                      Can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
                      "p2p": Exchange KV chunks with P2P communications in ring topology.
                             P2P is async and can be overlapped with attention compute.
                      "all_gather": All-gather to get full sequence of KV before attention.
                                    The all-gather is not async, and cannot be overlapped.
                      "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP
                             group, and gather to get full sequence of QKV.
                      "a2a+p2p": hierarchical CP implementation. First applying a2a to QKV
                      across each CP sub-group (e.g., via NVLink), then exchanging KV with
                      p2p between sub-groups (e.g., via IBLink).
        """
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream
        self.cp_comm_type = cp_comm_type

    @no_torch_dynamo(recursive=False)
    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_format: Optional[str] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        cu_seqlens_q_padded: Optional[torch.Tensor] = None,
        cu_seqlens_kv_padded: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        checkpoint_core_attention: bool = False,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
        inference_params: Optional[InferenceParams] = None,
    ) -> torch.Tensor:
        """
        Dot Product Attention Layer.

        .. note::

            Argument :attr:`attention_mask` is only used when :attr:`attn_mask_type`
            includes '"padding"' or `"arbitrary"`.

        .. note::

            DotProductAttention supports three backends: 1) FlashAttention which calls
            HazyResearch/Dao-AILab's `flash-attn <https://arxiv.org/pdf/2305.13245.pdf>`_
            PyTorch API, 2) FusedAttention which has multiple fused attention implementations
            based on `cuDNN Graph API
            <https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion>`_
            (see :attr:`FusedAttention` for more details on FusedAttention backends), and 3)
            UnfusedDotProductAttention which is the native PyTorch implementation
            with fused scaled masked softmax.

        .. note::

            Users can use environment variables :attr:`NVTE_FLASH_ATTN`, :attr:`NVTE_FUSED_ATTN`,
            and :attr:`NVTE_FUSED_ATTN_BACKEND` to control which DotProductAttention backend,
            and FusedAttention backend if applicable, to use. Transformer Engine prioritizes
            FlashAttention over FusedAttention and over UnfusedDotProductAttention.
            If FusedAttention is being used, users can also choose to switch to flash-attn's
            implementation for backward by setting :attr:`NVTE_FUSED_ATTN_USE_FAv2_BWD=1`
            (default: 0), because of the performance differences between various versions of
            flash-attn and FusedAttention. Further, :attr:`NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT`
            can be used to enable (:attr:`1`) or disable (:attr:`0`) the workspace related
            optimizations in FusedAttention. When unset, Transformer Engine determines the code path
            based on its internal logic. These optimizations trade memory for performance
            and should be used with care.

        .. note::
            .. _cu_seqlens note:

            When training data has variable sequence lengths, users have two options.

            1. Manipulate the data and pad all sequences to the same length. Use
               :attr:`qkv_format` = {"bshd", "sbhd"} and
               :attr:`attn_mask_type` = {"padding", "padding_causal", "padding_causal_bottom_right"}.
               Pass in :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`, or :attr:`attention_mask`
               (which will be converted to :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`), to provide
               the real sequence length information. For example, a batch of 3 sequences
               [a a a b b c c c c] can be padded to [a a a PAD b b PAD PAD c c c c], and the cumulative
               sequence length tensors would be
               :attr:`cu_seqlens_q` = :attr:`cu_seqlens_kv` = [0, 3, 5, 9] for self-attention.

            2. Do not perform padding on training data. Use :attr:`qkv_format` = "thd" and
               :attr:`attn_mask_type` = {"padding", "padding_causal", "padding_causal_bottom_right"}.
               Pass in :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`, or :attr:`attention_mask`,
               as in option 1. For example, a batch of 3 sequences [a a a b b c c c c] can be processed
               without any padding, and the sequence length tensors would be
               :attr:`cu_seqlens_q` = :attr:`cu_seqlens_kv` = [0, 3, 5, 9] for self-attention.

               In certain use cases, a varying number of identifier tokens are inserted between
               sequences. These tokens do not participate in the attention calculation.
               :attr:`cu_seqlens_q_padded` and :attr:`cu_seqlens_kv_padded` must be specified
               in such cases to correctly identify the start and end of each sequence in a batch.
               For example, a batch of 3 sequences [a a a 1 b b 2 2 c c c c 3] would have
               :attr:`cu_seqlens_q` = :attr:`cu_seqlens_kv` = [0, 3, 5, 9], and
               :attr:`cu_seqlens_q_padded` = :attr:`cu_seqlens_kv_padded` = [0, 4, 8, 13]
               for self-attention.

        .. note::
            .. _max_seqlen note:

            When :attr:`qkv_format` = {"bshd", "sbhd"}, sequences are of equal length in a batch.
            :attr:`max_seqlen_q` and :attr:`max_seqlen_kv` should be the same as the "s" dimension of
            :attr:`query_layer` and :attr:`key_layer` tensors. When unset, Transformer Engine will
            infer them as such.

            When :attr:`qkv_format` = "thd", sequences have varying lengths. :attr:`max_seqlen_q` and
            :attr:`max_seqlen_kv` should be the maximum query and key/value sequence length in a batch.
            When unset, Transformer Engine deduces them from :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`.
            This deduction costs a small kernel and some CPU-GPU synchronization, and to avoid this
            overhead, users are recommended to obtain the maximum sequence lengths from the data loaders
            and pass them in.

            - As the maximum sequence lengths, batch size, and number of tokens change from batch to batch,
              dynamic shapes need to be supported for tensor construction. FlashAttention and
              UnfusedDotProductAttention naturally do so, while FusedAttention requires parameters to be static
              to create graphs before performance heuristics analysis. To reduce the number of graphs created
              per run, Transformer Engine 1.13+ quantizes relevant parameters: for cuDNN < 9.6, {batch size,
              :attr:`max_seqlen_q`, :attr:`max_seqlen_kv`}, and for cuDNN >= 9.6, {"t" dimension of
              :attr:`query_layer`, "t" dimension of :attr:`key_layer`}.

        Parameters
        ----------
        query_layer : torch.Tensor
                     Query tensor.
        key_layer : torch.Tensor
                   Key tensor.
        value_layer : torch.Tensor
                     Value tensor.
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
             default = `None`. Boolean tensor(s) used to mask out attention softmax input.
             It should be `None` for causal masks and "`no_mask`". For padding masks, it should be
             a single tensor of [batch_size, 1, 1, seqlen_q] for self-attention, and a tuple of
             two tensors in shapes [batch_size, 1, 1, seqlen_q] and [batch_size, 1, 1, seqlen_kv]
             for cross-attention. For "`arbitrary`" mask, it should be in a shape broadcastable
             to [batch_size, num_heads, max_seqlen_q, max_seqlen_kv]. A `True` value means
             the corresponding position is masked out and a `False` means that position
             is allowed to participate in attention.
        qkv_format: str, default = `None`
                   If provided, overrides :attr:`qkv_format` from initialization.
        cu_seqlens_q: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (without offset) in a batch for `query_layer`,
                   with shape [batch_size + 1] and dtype torch.int32.
                   See :ref:`note<cu_seqlens note>` for more details.
        cu_seqlens_kv: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (without offset) in a batch for `key_layer`
                   and `value_layer`, with shape [batch_size + 1] and dtype torch.int32.
                   See :ref:`note<cu_seqlens note>` for more details.
        cu_seqlens_q_padded: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (with offset) in a batch for
                   `query_layer`, with shape [batch_size + 1] and dtype torch.int32.
                   When there is no padding between sequences in a batch,
                   `cu_seqlens_q_padded = cu_seqlens_q`.
                   See :ref:`note<cu_seqlens note>` for more details.
        cu_seqlens_kv_padded: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (with offset) in a batch for `key_layer`
                   and `value_layer`, with shape [batch_size + 1] and dtype torch.int32.
                   When there is no padding between sequences in a batch,
                   `cu_seqlens_kv_padded = cu_seqlens_kv`.
                   See :ref:`note<cu_seqlens note>` for more details.
        max_seqlen_q: Optional[int], default = `None`
                      Maximum sequence length in `query_layer`.
                      See :ref:`note<max_seqlen note>` for more details.
        max_seqlen_kv: Optional[int], default = `None`
                       Maximum sequence length in `key_layer` and `value_layer`.
                       See :ref:`note<max_seqlen note>` for more details.
        attn_mask_type: {'no_mask', 'padding', 'causal', 'padding,causal', 'causal,padding',
                       'padding_causal', 'causal_bottom_right', 'padding_causal_bottom_right',
                       'arbitrary'}, default = `None`. Type of attention mask passed into
                       softmax operation. 'padding,causal', 'causal,padding' and 'padding_causal'
                       are equivalent. By default, causal masks are aligned to the top left corner
                       of the softmax matrix. When "`bottom_right`" is specified in the mask type,
                       causal masks are aligned to the bottom right corner.
        window_size: Optional[Tuple[int, int]], default = `None`
                    Sliding window size for local attention.
        checkpoint_core_attention : bool, default = `False`
                                   If true, forward activations for attention are recomputed
                                   during the backward pass in order to save memory that would
                                   otherwise be occupied to store the forward activations until
                                   backprop.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, `post_scale_bias`, `alibi`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T, shape [1, num_head, max_seqlen_q, max_seqlen_kv].
                    It should be 'None' for 'no_bias' and 'alibi' bias types.
        alibi_slopes: Optional[torch.Tensor], default = `None`
                     ALiBi slopes in FP32 and shape [nheads] or [batch_size, nheads].
                     It adds a bias of (-alibi_slope * (i + seqlen_k - seqlen_q - j))
                     to the attention score of query i and key j.
        fast_zero_fill: bool, default = `True`
                    Whether to use the fast path to set output tensors to 0 or not.
        inference_params: Optional[InferenceParams], default = `None`
            Optimizes execution performance during inference by caching Keys and Values of the
            current decoding iteration. These cached values are appended to the K and V values
            computed in previous iterations, eliminating the need to recalculate them for the
            entire sequence.
            Initialization of `inference_params` is required prior to use to ensure sufficient
            memory allocation.
            Adjustments of the sequence_len_offset should be done after a complete forward pass.
            If rotary positional embeddings (RoPE) are utilized, they must be prepared beforehand.
            Supports "sbhd" and "bshd" layouts, with the "sbhd" layout being more efficient.
        """

        with self.prepare_forward(
            query_layer,
            num_gemms=3,
            allow_non_contiguous=True,
        ) as query_layer:
            if self.fp8:
                if self.fp8_meta["recipe"].fp8_mha:
                    if not self.fp8_meta["recipe"].fp8_dpa:
                        self.fp8_meta["recipe"].fp8_dpa = True
                        self.logger.warning(
                            """Forcing fp8_meta["recipe"].fp8_dpa=True due to """
                            """fp8_meta["recipe"].fp8_mha=True"""
                        )

            if self.fp8 and self.fp8_meta["recipe"].fp8_dpa:
                forward_dtype = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=True)
                backward_dtype = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=False)
                assert forward_dtype in [
                    tex.DType.kFloat8E4M3,
                    tex.DType.kFloat8E5M2,
                ] and backward_dtype in [
                    tex.DType.kFloat8E4M3,
                    tex.DType.kFloat8E5M2,
                ], """DotProductAttention only supports "E4M3" and "E5M2" FP8 data types."""

            assert (
                query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
            ), "DotProductAttention only supports CUDA tensors."
            assert (
                query_layer.dtype == key_layer.dtype and query_layer.dtype == value_layer.dtype
            ), "Queries, keys and values must have the same data type!"
            assert (
                key_layer.shape[:-1] == value_layer.shape[:-1]
            ), "Keys and values must have the same batch size, sequence length and number of heads!"
            assert (
                key_layer.shape[-1] == self.hidden_size_per_attention_head_k
            ), f"Keys have head_dim = {key_layer.shape[-1]}, "
            "but expected head_dim = {self.hidden_size_per_attention_head_k}!"
            assert (
                value_layer.shape[-1] == self.hidden_size_per_attention_head_v
            ), f"Values have head_dim = {value_layer.shape[-1]}, "
            "but expected head_dim = {self.hidden_size_per_attention_head_v}!"

            if qkv_format is None:
                qkv_format = self.qkv_format

            if attn_mask_type is None:
                attn_mask_type = self.attn_mask_type
            else:
                attn_mask_type = attn_mask_type.replace(",", "_")
                if attn_mask_type == "causal_padding":
                    attn_mask_type = "padding_causal"
            assert (
                attn_mask_type in AttnMaskTypes
            ), f"Attention mask type {attn_mask_type} is not supported!"
            if qkv_format == "thd":
                assert (
                    "padding" in attn_mask_type
                ), "Attention mask type must be padding or padding_causal for qkv_format=thd!"

            if window_size is None:
                window_size = self.window_size
            window_size = check_set_window_size(attn_mask_type, window_size)

            if self.rng_states_tracker is not None and is_graph_capturing():
                assert isinstance(
                    self.rng_states_tracker, CudaRNGStatesTracker
                ), "Unsupported RNG states tracker."
                assert (
                    graph_safe_rng_available()
                ), "Upgrade PyTorch version to get RNG manipulation support for cuda graph capture."

            if inference_params is not None:
                assert self.layer_number is not None, "Layer number must be set!"

                # convert causal to causal_bottom_right in inference when KV-caching is in use
                # so users can run with the same attn_mask_type for training and inference
                if attn_mask_type in ["causal", "padding_causal"]:
                    attn_mask_type = attn_mask_type + "_bottom_right"

                if qkv_format == "bshd":
                    key_layer = key_layer.transpose(0, 1)
                    value_layer = value_layer.transpose(0, 1)

                (
                    inference_key_memory,
                    inference_value_memory,
                ) = inference_params.key_value_memory_dict[self.layer_number]

                batch_start = inference_params.batch_size_offset
                batch_end = batch_start + key_layer.size(1)
                assert batch_end <= inference_key_memory.size(1)

                sequence_start = inference_params.sequence_len_offset
                sequence_end = sequence_start + key_layer.size(0)
                assert sequence_end <= inference_key_memory.size(0)

                # Copy keys and values into KV-cache
                inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = (
                    key_layer
                )
                inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = (
                    value_layer
                )
                key_layer = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
                value_layer = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

                if qkv_format == "bshd":
                    key_layer = key_layer.transpose(0, 1)
                    value_layer = value_layer.transpose(0, 1)

                key_layer = key_layer.contiguous()
                value_layer = value_layer.contiguous()

            assert (
                key_layer.shape[-2] == self.num_gqa_groups_per_partition
                and value_layer.shape[-2] == self.num_gqa_groups_per_partition
            ), (
                "Keys and values must have num_gqa_group ="
                f" {self.num_gqa_groups_per_partition} heads!"
            )
            assert qkv_format in [
                "sbhd",
                "bshd",
                "thd",
            ], "DotProductAttention only supports qkv_format = {'sbhd', 'bshd', 'thd'}!"

            if qkv_format == "thd":
                assert all(
                    len(x.shape) == 3 for x in (query_layer, key_layer, value_layer)
                ), "Queries, keys and values must be 3D tensors when qkv_format = thd!"
                assert (
                    cu_seqlens_q is not None and cu_seqlens_kv is not None
                ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
                assert (
                    cu_seqlens_q.shape == cu_seqlens_kv.shape
                    and len(cu_seqlens_q.shape) == 1
                    and len(cu_seqlens_kv.shape) == 1
                ), "cu_seqlens_q and cu_seqlens_q must both have shape [batch_size + 1]!"
                assert (
                    cu_seqlens_q.dtype == torch.int32 and cu_seqlens_kv.dtype == torch.int32
                ), "cu_seqlens_q and cu_seqlens_q must both be in dtype torch.int32!"
                batch_size = len(cu_seqlens_q) - 1
                if max_seqlen_q is None:
                    if cu_seqlens_q_padded is not None:
                        seqlens_q = cu_seqlens_q_padded[1:] - cu_seqlens_q_padded[:-1]
                    else:
                        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                    max_seqlen_q = int((seqlens_q.max().item() + 63) // 64 * 64)
                if max_seqlen_kv is None:
                    if cu_seqlens_kv_padded is not None:
                        seqlens_kv = cu_seqlens_kv_padded[1:] - cu_seqlens_kv_padded[:-1]
                    else:
                        seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                    max_seqlen_kv = int((seqlens_kv.max().item() + 63) // 64 * 64)

            cp_size = 1
            if isinstance(self.cp_group, dist_group_type):
                cp_size = get_distributed_world_size(self.cp_group)
            elif isinstance(self.cp_group, list):
                for group in self.cp_group:
                    cp_size *= get_distributed_world_size(group)
            context_parallel = cp_size > 1

            if qkv_format in ["sbhd", "bshd"]:
                assert all(
                    len(x.shape) == 4 for x in (query_layer, key_layer, value_layer)
                ), f"Queries, keys and values must be 4D tensors when qkv_format = {qkv_format}!"
                if qkv_format == "sbhd":
                    max_seqlen_q = query_layer.shape[0] if max_seqlen_q is None else max_seqlen_q
                    max_seqlen_kv = key_layer.shape[0] if max_seqlen_kv is None else max_seqlen_kv
                    batch_size = query_layer.shape[1]
                else:
                    max_seqlen_q = query_layer.shape[1] if max_seqlen_q is None else max_seqlen_q
                    max_seqlen_kv = key_layer.shape[1] if max_seqlen_kv is None else max_seqlen_kv
                    batch_size = query_layer.shape[0]
                max_seqlen_q *= cp_size
                max_seqlen_kv *= cp_size
                if cu_seqlens_q is not None:
                    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                    assert all(
                        seqlens_q <= max_seqlen_q
                    ), """Sequence lengths indicated by cu_seqlens_q must be no greater than
                        the sequence dimension in 'query_layer'!"""
                if cu_seqlens_kv is not None:
                    seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                    assert all(
                        seqlens_kv <= max_seqlen_kv
                    ), """Sequence lengths indicated by cu_seqlens_kv must be no greater than
                        the sequence dimension in 'key_layer' and 'value_layer'!"""
                if cu_seqlens_q is None or cu_seqlens_kv is None:
                    if "padding" in attn_mask_type:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        if self.attention_type == "self":
                            cu_seqlens_q = get_cu_seqlens(attention_mask)
                            cu_seqlens_kv = cu_seqlens_q
                        else:
                            cu_seqlens_q = get_cu_seqlens(attention_mask[0])
                            cu_seqlens_kv = get_cu_seqlens(attention_mask[1])
                    else:
                        cu_seqlens_q = _get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_q,
                            query_layer.device,
                        )
                        cu_seqlens_kv = _get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_kv,
                            key_layer.device,
                        )

            if (
                isinstance(query_layer, Float8Tensor)
                and isinstance(key_layer, Float8Tensor)
                and isinstance(value_layer, Float8Tensor)
            ):
                qkv_layout, query_layer._data, key_layer._data, value_layer._data = get_qkv_layout(
                    query_layer._data, key_layer._data, value_layer._data, qkv_format=qkv_format
                )
            else:
                qkv_layout, query_layer, key_layer, value_layer = get_qkv_layout(
                    query_layer, key_layer, value_layer, qkv_format=qkv_format
                )

            global _alibi_cache
            if alibi_slopes is not None:
                assert (
                    core_attention_bias_type == "alibi"
                ), "core_attention_bias_type must be alibi in order to use alibi_slopes!"
                if self.layer_number == 1:
                    _alibi_cache["_alibi_slopes_require_update"] = True
                    _alibi_cache["_alibi_bias_require_update"] = True
            bottom_right_alignment = (attn_mask_type not in ["causal", "padding_causal"],)
            if core_attention_bias_type == "alibi":
                assert (
                    core_attention_bias is None
                ), "core_attention_bias must be None when core_attention_bias_type is alibi!"
                if (
                    _alibi_cache["_num_heads"] != query_layer.shape[-2]
                    or _alibi_cache["_max_seqlen_q"] != max_seqlen_q
                    or _alibi_cache["_max_seqlen_kv"] != max_seqlen_kv
                    or _alibi_cache["_bottom_right_alignment"] != bottom_right_alignment
                    or _alibi_cache["_alibi_slopes"] is None
                ):
                    _alibi_cache["_alibi_slopes_require_update"] = True
                    _alibi_cache["_alibi_bias_require_update"] = True

            core_attention_bias_shape = None
            if core_attention_bias is not None:
                if (
                    core_attention_bias.shape[0] == batch_size
                    and core_attention_bias.shape[1] == query_layer.shape[-2]
                ):
                    core_attention_bias_shape = "bhss"
                elif (
                    core_attention_bias.shape[0] == 1
                    and core_attention_bias.shape[1] == query_layer.shape[-2]
                ):
                    core_attention_bias_shape = "1hss"
                elif (
                    core_attention_bias.shape[0] == batch_size and core_attention_bias.shape[1] == 1
                ):
                    core_attention_bias_shape = "b1ss"
                elif core_attention_bias.shape[0] == 1 and core_attention_bias.shape[1] == 1:
                    core_attention_bias_shape = "11ss"
                else:
                    assert (
                        False
                    ), "core_attention_bias must be in one of {bhss, 1hss, b1ss, 11ss} shapes"

            pad_between_seqs = (
                cu_seqlens_q_padded is not None
                and not torch.equal(cu_seqlens_q_padded[:-1], cu_seqlens_q[:-1])
            ) or (
                cu_seqlens_kv_padded is not None
                and not torch.equal(cu_seqlens_kv_padded[:-1], cu_seqlens_kv[:-1])
            )

            attention_params = AttentionParams(
                qkv_type=type(query_layer),
                qkv_dtype=query_layer.dtype,
                qkv_layout=qkv_layout,
                batch_size=batch_size,
                num_heads=query_layer.shape[-2],
                num_gqa_groups=key_layer.shape[-2],
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                head_dim_qk=query_layer.shape[-1],
                head_dim_v=value_layer.shape[-1],
                attn_mask_type=attn_mask_type,
                window_size=window_size,
                alibi_slopes_shape=alibi_slopes.shape if alibi_slopes is not None else None,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias_shape=core_attention_bias_shape,
                core_attention_bias_requires_grad=(
                    core_attention_bias.requires_grad if core_attention_bias is not None else False
                ),
                pad_between_seqs=pad_between_seqs,
                attention_dropout=self.attention_dropout,
                context_parallel=context_parallel,
                deterministic=self.deterministic,
                is_training=self.training,
                fp8=self.fp8,
                fp8_meta=self.fp8_meta,
            )
            global _attention_backends, _use_flash_attn_3
            if (
                _attention_backends["attention_params"] is None
                or attention_params != _attention_backends["attention_params"]
            ):
                _attention_backends["attention_params"] = attention_params
                _attention_backends["backend_selection_requires_update"] = True
            if _attention_backends["backend_selection_requires_update"]:
                _use_flash_attn_3 = _flash_attn_3_is_installed
                (
                    use_flash_attention,
                    use_fused_attention,
                    fused_attention_backend,
                    use_unfused_attention,
                    _,
                ) = get_attention_backend(attention_params)
                if use_flash_attention:
                    self.logger.info(
                        "Running with FlashAttention backend (version %s)",
                        _flash_attn_version if not _use_flash_attn_3 else _flash_attn_3_version,
                    )
                elif use_fused_attention:
                    self.logger.info(
                        "Running with FusedAttention backend (sub-backend %s)",
                        int(fused_attention_backend),
                    )
                elif use_unfused_attention:
                    self.logger.info("Running with UnfusedDotProductAttention backend")
            else:
                use_flash_attention = _attention_backends["use_flash_attention"]
                use_fused_attention = _attention_backends["use_fused_attention"]
                fused_attention_backend = _attention_backends["fused_attention_backend"]
                use_unfused_attention = _attention_backends["use_unfused_attention"]

            if use_flash_attention:
                if core_attention_bias_type == "alibi":
                    alibi_slopes, _ = get_alibi(
                        query_layer.shape[-2],
                        max_seqlen_q,
                        max_seqlen_kv,
                        alibi_slopes=alibi_slopes,
                    )
                return self.flash_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attention_mask=attention_mask,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    attn_mask_type=attn_mask_type,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    cp_group=self.cp_group,
                    cp_global_ranks=self.cp_global_ranks,
                    cp_stream=self.cp_stream,
                    cp_comm_type=self.cp_comm_type,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                    fp8_meta=self.fp8_meta,
                    quantizers=self.quantizers,
                )

            if use_fused_attention:
                fu_core_attention_bias_type = core_attention_bias_type
                fu_core_attention_bias = core_attention_bias
                if core_attention_bias_type == "alibi" and (
                    alibi_slopes is not None or max_seqlen_q != max_seqlen_kv
                ):
                    fu_core_attention_bias_type = "post_scale_bias"
                    _, fu_core_attention_bias = get_alibi(
                        query_layer.shape[-2],
                        max_seqlen_q,
                        max_seqlen_kv,
                        alibi_slopes=alibi_slopes,
                        bias_dtype=query_layer.dtype,
                        bottom_right_alignment=attn_mask_type not in ["causal", "padding_causal"],
                    )
                if checkpoint_core_attention:
                    return self._checkpointed_attention_forward(
                        self.fused_attention,
                        query_layer,
                        key_layer,
                        value_layer,
                        qkv_layout=qkv_layout,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        cu_seqlens_q_padded=cu_seqlens_q_padded,
                        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        attn_mask_type=attn_mask_type,
                        attention_mask=attention_mask,
                        window_size=window_size,
                        fused_attention_backend=fused_attention_backend,
                        core_attention_bias_type=fu_core_attention_bias_type,
                        core_attention_bias=fu_core_attention_bias,
                        fast_zero_fill=fast_zero_fill,
                        cp_group=self.cp_group,
                        cp_global_ranks=self.cp_global_ranks,
                        cp_stream=self.cp_stream,
                        cp_comm_type=self.cp_comm_type,
                        fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                        fp8_meta=self.fp8_meta,
                    )
                return self.fused_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    attn_mask_type=attn_mask_type,
                    attention_mask=attention_mask,
                    window_size=window_size,
                    fused_attention_backend=fused_attention_backend,
                    core_attention_bias_type=fu_core_attention_bias_type,
                    core_attention_bias=fu_core_attention_bias,
                    fast_zero_fill=fast_zero_fill,
                    cp_group=self.cp_group,
                    cp_global_ranks=self.cp_global_ranks,
                    cp_stream=self.cp_stream,
                    cp_comm_type=self.cp_comm_type,
                    fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                    fp8_meta=self.fp8_meta,
                    quantizers=self.quantizers,
                )

            from .cpu_offload import CPUOffloadEnabled

            if CPUOffloadEnabled:
                warnings.warn(
                    "Attention activation Offloading is only implemented"
                    "with Flash Attention and Fused Attention!"
                )

            if use_unfused_attention:
                if checkpoint_core_attention:
                    return self._checkpointed_attention_forward(
                        self.unfused_attention,
                        query_layer,
                        key_layer,
                        value_layer,
                        qkv_layout=qkv_layout,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        attn_mask_type=attn_mask_type,
                        attention_mask=attention_mask,
                        window_size=window_size,
                        core_attention_bias_type=core_attention_bias_type,
                        core_attention_bias=core_attention_bias,
                        alibi_slopes=alibi_slopes,
                    )
                return self.unfused_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    attn_mask_type=attn_mask_type,
                    attention_mask=attention_mask,
                    window_size=window_size,
                    core_attention_bias_type=core_attention_bias_type,
                    core_attention_bias=core_attention_bias,
                    alibi_slopes=alibi_slopes,
                )

            raise ValueError("No dot product attention support for the provided inputs!")


class MultiheadAttention(torch.nn.Module):
    r"""
    Multi-head Attention (MHA), including Query,
    Key, Value and Output projection.

    .. note::

        Argument :attr:`attention_mask` in the `forward` call is only used when
        :attr:`attn_mask_type` includes '"padding"' or `"arbitrary"`.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels: int, default = `None`
                number of key-value channels. defaults to
                :attr:`hidden_size` / :attr:`num_attention_heads` if `None`.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    layernorm_epsilon : float, default = 1e-5
                       a value added to the denominator of layer normalization
                       for numerical stability.
    init_method : Callable, default = `None`
                 used for initializing weights of QKV and FC1 weights in the following way:
                 `init_method(weight)`. When set to `None`, defaults to
                 `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing weights of PROJ and FC2 in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    layer_number: int, default = `None`
                 layer number of the current `TransformerLayer` when multiple such modules are
                 concatenated to form a transformer block.
    attn_mask_type: {'no_mask', 'padding', 'causal', 'padding_causal', 'causal_bottom_right',
                   'padding_causal_bottom_right','arbitrary'},
                   default = `causal`
                   type of attention mask passed into softmax operation. Overridden by
                   :attr:`attn_mask_type` in the `forward` method. The forward
                   arg is useful for dynamically changing mask types, e.g. a different
                   mask for training and inference. The init arg is useful for cases
                   involving compilation/tracing, e.g. ONNX export.
    window_size: Optional[Tuple[int, int]], default = `None`
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically. Both `causal` and `causal_bottom_right` masks
                map to `window_size = (-1, 0)` and Transformer Engine distinguishes them based on
                `attn_mask_type`. Similar to :attr:`attn_mask_type`, `window_size` can
                be overridden by :attr:`window_size` in `forward` as well.
    num_gqa_groups : int, default = `None`
                         number of GQA groups in the transformer layer.
                         Grouped Query Attention is described in
                         `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                         This only affects the keys and values, not the querys.
                         GQA-1 is equivalent to Multi-Query Attention
                         (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                         is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
    input_layernorm: bool, default = `False`
                     if set to `True`, layer normalization to the input is applied.
    attention_type: { 'self', 'cross' }, default = 'self'
                   type of attention applied.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    qkv_weight_interleaved : bool, default = `True`
                            if set to `False`, the QKV weight is interpreted as a concatenation of
                            query, key, and value weights along the `0th` dimension. The default
                            interpretation is that the individual `q`, `k`, and `v` weights for each
                            attention head are interleaved. This parameter is set to `False` when
                            using :attr:`fuse_qkv_params=False`.
    bias : bool, default = `True`
          if set to `False`, the transformer layer will not learn any additive biases.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    qkv_format: str, default = `sbhd`
            dimension format for `query_layer`, `key_layer` and `value_layer`,
            {`sbhd`, `bshd`}. `s` stands for the sequence length, `b` batch size,
            `h` the number of heads and `d` head size. `sbhd` and `bshd` formats
            are used for when sequences in a batch are of equal length or padded to
            equal length. Please note that these formats do not reflect how
            tensors `query_layer`, `key_layer`, `value_layer` are laid out in memory.
            For that, please use `get_qkv_layout` to gain the layout information.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, QKV and FC1 layers are used as Column Parallel
                      whereas PROJ and FC2 is used as Row Parallel as described
                      `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    fuse_qkv_params: bool, default = 'False'
                    if set to `True`, `TransformerLayer` module exposes a single fused
                    parameter for query-key-value. This enables optimizations such as QKV
                    fusion without concatentations/splits and also enables the argument
                    `fuse_wgrad_accumulation`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: Optional[int] = None,
        attention_dropout: float = 0.1,
        layernorm_epsilon: float = 1e-5,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        layer_number: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        num_gqa_groups: Optional[int] = None,
        fuse_wgrad_accumulation: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        sequence_parallel: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_bias: bool = False,
        return_layernorm_output: bool = False,
        input_layernorm: bool = False,
        attention_type: str = "self",
        set_parallel_mode: bool = False,
        fuse_qkv_params: bool = False,
        zero_centered_gamma: bool = False,
        qkv_weight_interleaved: bool = True,
        ub_overlap_ag: bool = False,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        bias: bool = True,
        normalization: str = "LayerNorm",
        device: Union[torch.device, str] = "cuda",
        qkv_format: str = "sbhd",
    ) -> None:
        super().__init__()

        self.qkv_format = qkv_format
        self.attn_mask_type = attn_mask_type
        self.window_size = check_set_window_size(attn_mask_type, window_size)
        self.layer_number = layer_number
        self.input_layernorm = input_layernorm
        self.attention_type = attention_type
        self.get_rng_state_tracker = get_rng_state_tracker
        self.tp_group = tp_group
        self.return_layernorm_output = return_layernorm_output
        self.params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_attention_heads = num_attention_heads
        self.return_bias = return_bias
        self.cp_size = 1
        self.cp_rank = 0

        kv_channels = kv_channels if kv_channels else (hidden_size // num_attention_heads)

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        if not fuse_qkv_params:
            qkv_weight_interleaved = False
        self.qkv_weight_interleaved = qkv_weight_interleaved

        assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"
        if layer_number is not None:
            assert layer_number > 0, "layer_number must be a positive integer"

        tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.tp_size = tp_size
        self.sequence_parallel = (tp_size > 1) and sequence_parallel

        self.num_attention_heads_per_partition = divide(num_attention_heads, tp_size)
        self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
        assert (
            num_attention_heads % self.num_gqa_groups == 0
        ), "The number of attention heads must be divisible by the number of GQA groups!"
        assert (
            self.num_gqa_groups % tp_size == 0
        ), "The number of GQA groups must be divisible by tensor parallel size!"
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // tp_size)

        self.hidden_size_per_attention_head = kv_channels
        self.hidden_size_q = self.hidden_size_per_attention_head * num_attention_heads
        self.hidden_size_kv = self.hidden_size_per_attention_head * self.num_gqa_groups

        common_gemm_kwargs = {
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "get_rng_state_tracker": get_rng_state_tracker,
            "sequence_parallel": sequence_parallel,
            "params_dtype": self.params_dtype,
            "device": device,
        }

        qkv_parallel_mode = "column" if set_parallel_mode else None

        if self.attention_type == "self":
            parameters_split = None
            if not fuse_qkv_params:
                parameters_split = collections.OrderedDict(
                    [
                        ("query", self.hidden_size_q),
                        ("key", self.hidden_size_kv),
                        ("value", self.hidden_size_kv),
                    ]
                )
            if self.input_layernorm:
                self.layernorm_qkv = LayerNormLinear(
                    hidden_size,
                    self.hidden_size_q + 2 * self.hidden_size_kv,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    return_layernorm_output=return_layernorm_output,
                    parameters_split=parameters_split,
                    zero_centered_gamma=zero_centered_gamma,
                    ub_bulk_wgrad=ub_bulk_wgrad,
                    ub_bulk_dgrad=ub_bulk_dgrad,
                    ub_overlap_rs_dgrad=ub_overlap_rs_dgrad,
                    ub_overlap_ag=ub_overlap_ag,
                    normalization=normalization,
                    ub_name="qkv",
                    **common_gemm_kwargs,
                )
            else:
                self.qkv = Linear(
                    hidden_size,
                    self.hidden_size_q + 2 * self.hidden_size_kv,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    parameters_split=parameters_split,
                    **common_gemm_kwargs,
                )
        elif self.attention_type == "cross":
            if self.input_layernorm:
                self.layernorm_query = LayerNormLinear(
                    hidden_size,
                    self.hidden_size_q,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    parameters_split=("query",) if not fuse_qkv_params else None,
                    return_layernorm_output=return_layernorm_output,
                    zero_centered_gamma=zero_centered_gamma,
                    ub_bulk_wgrad=ub_bulk_wgrad,
                    ub_bulk_dgrad=ub_bulk_dgrad,
                    ub_overlap_rs_dgrad=ub_overlap_rs_dgrad,
                    ub_overlap_ag=ub_overlap_ag,
                    normalization=normalization,
                    ub_name="qkv",
                    **common_gemm_kwargs,
                )
            else:
                self.query_layer = Linear(
                    hidden_size,
                    self.hidden_size_q,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    **common_gemm_kwargs,
                )
            self.key_value = Linear(
                hidden_size,
                2 * self.hidden_size_kv,
                init_method=init_method,
                bias=bias,
                return_bias=False,
                parallel_mode=qkv_parallel_mode,
                parameters_split=("key", "value") if not fuse_qkv_params else None,
                **common_gemm_kwargs,
            )

        # Attention.
        self.core_attention = DotProductAttention(
            num_attention_heads,
            self.hidden_size_per_attention_head,
            num_gqa_groups=self.num_gqa_groups,
            attention_dropout=attention_dropout,
            qkv_format=self.qkv_format,
            tp_size=tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            layer_number=self.layer_number,
            attention_type=self.attention_type,
        )

        # Linear
        self.proj = Linear(
            self.hidden_size_q,
            hidden_size,
            init_method=output_layer_init_method,
            bias=bias,
            return_bias=return_bias,
            parallel_mode="row" if set_parallel_mode else None,
            ub_overlap_rs=ub_overlap_rs,
            ub_overlap_ag=ub_overlap_ag,
            ub_name="proj",
            **common_gemm_kwargs,
        )

    def _allocate_memory(
        self, inference_max_sequence_len: int, batch_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Allocates memory for KV cache."""
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_gqa_groups_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """
        Set the tensor parallel group for the given
        module before executing the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = `None`
                  tensor parallel process group.
        """
        self.tp_group = tp_group

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, List[dist_group_type], None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
        cp_comm_type: str = "p2p",
    ) -> None:
        """
        Set the context parallel attributes for the given
        module before executing the forward pass.

        Parameters
        ----------
        cp_group : Union[ProcessGroup, List[ProcessGroup]]
                  context parallel process group.
                  ProcessGroup is for cp_comm_type of "p2p", "all_gather", and "a2a".
                  List[ProcessGroup] is for cp_comm_type of "a2a+p2p", where cp_group[0]
                  and cp_group[1] are for a2a and p2p communications respectively.
        cp_global_ranks : List[int]
                         list of global ranks in the context group.
        cp_stream : torch.cuda.Stream
                   cuda stream for context parallel execution.
        cp_comm_type : str, default = `p2p`
                      inter-gpu communication type for context parallelism.
                      Can be "p2p" or "all_gather" or "a2a", "a2a+p2p".
                      "p2p": Exchange KV chunks with P2P communications in ring topology.
                             P2P is async and can be overlapped with attention compute.
                      "all_gather": All-gather to get full sequence of KV before attention.
                                    The all-gather is not async, and cannot be overlapped.
                      "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP
                             group, and gather to get full sequence of QKV.
                      "a2a+p2p": hierarchical CP implementation. First applying a2a to QKV
                      across each CP sub-group (e.g., via NVLink), then exchanging KV with
                      p2p between sub-groups (e.g., via IBLink).
        """
        if isinstance(cp_group, dist_group_type):
            self.cp_size = get_distributed_world_size(cp_group)
            self.cp_rank = get_distributed_rank(cp_group)
        elif isinstance(cp_group, list):
            assert len(cp_group) == 2, "Current implementation only supports two-level CP groups!"
            assert (
                cp_comm_type == "a2a+p2p"
            ), "Only cp_comm_type of a2a+p2p requires hierarchical CP groups!"
            cp_size_a2a = get_distributed_world_size(cp_group[0])
            cp_rank_a2a = get_distributed_rank(cp_group[0])
            cp_size_p2p = get_distributed_world_size(cp_group[1])
            cp_rank_p2p = get_distributed_rank(cp_group[1])
            self.cp_size = cp_size_a2a * cp_size_p2p
            self.cp_rank = cp_size_a2a * cp_rank_p2p + cp_rank_a2a

        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_context_parallel_group"):
                child.set_context_parallel_group(cp_group, cp_global_ranks, cp_stream, cp_comm_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: bool = False,
        inference_params: Optional[InferenceParams] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        fast_zero_fill: bool = True,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Forward propagation for MultiheadAttention layer.

        .. note::

            Argument :attr:`attention_mask` is only used when :attr:`attn_mask_type`
            includes `"padding"` or `"arbitrary"`.

        Parameters
        ----------
        hidden_states : torch.Tensor
             Input tensor.
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
             default = `None`. Boolean tensor(s) used to mask out attention softmax input.
             It should be `None` for causal masks and "`no_mask`". For padding masks, it should be
             a single tensor of [batch_size, 1, 1, seqlen_q] for self-attention, and a tuple of
             two tensors in shapes [batch_size, 1, 1, seqlen_q] and [batch_size, 1, 1, seqlen_kv]
             for cross-attention. For "`arbitrary`" mask, it should be in a shape broadcastable to
             [batch_size, num_heads, max_seqlen_q, max_seqlen_kv]. A `True` value means
             the corresponding position is masked out and a `False` means that position
             is allowed to participate in attention.
        attn_mask_type: {'no_mask', 'padding', 'causal', 'padding_causal', 'causal_bottom_right',
                       'padding_causal_bottom_right','arbitrary'},
                       default = `None`
                       type of attention mask passed into softmax operation. By default,
                       causal masks are aligned to the top left corner of the softmax matrix.
                       When "`bottom_right`" is specified in the mask type, causal masks are
                       aligned to the bottom right corner.
        window_size: Optional[Tuple[int, int]], default = `None`
                    sliding window size for local attention.
        encoder_output : Optional[torch.Tensor], default = `None`
             Output of the encoder block to be fed into the decoder block if using
             `layer_type="decoder"`.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        checkpoint_core_attention: bool, default = `False`
                                  If true, forward activations for core attention are recomputed
                                  during the backward pass in order to save memory that would
                                  otherwise be occupied to store the forward activations until
                                  backprop.
        rotary_pos_emb: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], default = `None`
                       Embeddings for query and key tensors for applying rotary position
                       embedding. By default no input embedding is applied.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, 'post_scale_bias`, `alibi`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T, shape [1, num_head, max_seqlen_q, max_seqlen_kv].
                    It should be 'None' for 'no_bias' and 'alibi' bias types.
        alibi_slopes: Optional[torch.Tensor], default = `None`
                     ALiBi slopes in FP32 and shape [nheads] or [batch_size, nheads].
                     It adds a bias of (-alibi_slope * (i + seqlen_k - seqlen_q - j))
                     to the attention score of query i and key j.
        cu_seqlens_q: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (without offset) in a batch for `query_layer`,
                   with shape [batch_size + 1] and dtype torch.int32.
        cu_seqlens_kv: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (without offset) in a batch for `key_layer`
                   and `value_layer`, with shape [batch_size + 1] and dtype torch.int32.
        max_seqlen_q: Optional[int], default = `None`
                      Maximum sequence length in `query_layer`.
                      Calculated from `cu_seqlens_q` if not provided.
        max_seqlen_kv: Optional[int], default = `None`
                       Maximum sequence length in `key_layer` and `value_layer`.
                       Calculated from `cu_seqlens_kv` if not provided.
        fast_zero_fill: bool, default = `True`
                    Whether to set output tensors to 0 or not before use.
        """
        # hidden_states: [sq, b, h]

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        if window_size is None:
            window_size = self.window_size
        window_size = check_set_window_size(attn_mask_type, window_size)

        if "padding" in attn_mask_type and attention_mask is not None:
            for mask in attention_mask:
                assert mask.dtype == torch.bool, "Attention mask must be in boolean type!"

        assert (
            core_attention_bias_type in AttnBiasTypes
        ), f"core_attention_bias_type {core_attention_bias_type} is not supported!"

        # =================================================
        # Pre-allocate memory for key-values for inference
        # =================================================

        if inference_params and self.layer_number is not None:
            assert (
                self.qkv_format != "thd"
            ), "qkv_format == thd is not supported for an inference with KV-cache!"
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_length
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size, hidden_states.dtype
                )
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size, hidden_states.dtype
                )
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory,
                    inference_value_memory,
                )
            else:
                (
                    inference_key_memory,
                    inference_value_memory,
                ) = inference_params.key_value_memory_dict[self.layer_number]

        # ======================
        # Query, Key, and Value
        # ======================

        fp8_mha = (
            FP8GlobalStateManager.is_fp8_enabled()
            and FP8GlobalStateManager.get_fp8_recipe().fp8_mha
        )

        layernorm_output = None
        if self.attention_type == "self":
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn]
            if self.input_layernorm:
                layernorm_qkv_outputs = self.layernorm_qkv(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                    fp8_output=fp8_mha and rotary_pos_emb is None,
                )
                if self.return_layernorm_output:
                    mixed_x_layer, layernorm_output = layernorm_qkv_outputs
                else:
                    mixed_x_layer = layernorm_qkv_outputs
            else:
                mixed_x_layer = self.qkv(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                    fp8_output=fp8_mha and rotary_pos_emb is None,
                )

            num_queries_per_key_value = (
                self.num_attention_heads_per_partition // self.num_gqa_groups_per_partition
            )
            if self.qkv_weight_interleaved:
                # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, ng, (np/ng + 2), hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    self.num_gqa_groups_per_partition,
                    (num_queries_per_key_value + 2),
                    self.hidden_size_per_attention_head,
                )
                # split along second last dimension
                split_dim = -2
            else:
                # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, (np/ng + 2), ng, hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    (num_queries_per_key_value + 2),
                    self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
                # split along third last dimension
                split_dim = -3

            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # qkv_weight_interleaved:
            #  [sq, b, ng, (np/ng + 2), hn]
            #  --> [sq, b, ng, np/ng, hn], [sq, b, ng, 1, hn], [sq, b, ng, 1, hn]
            # not qkv_weight_interleaved:
            #  [sq, b, (np/ng + 2), ng, hn]
            #  --> [sq, b, np/ng, np, hn], [sq, b, 1, ng, hn], [sq, b, 1, ng, hn]
            query_layer, key_layer, value_layer = _SplitAlongDim.apply(
                mixed_x_layer, split_dim, (num_queries_per_key_value, 1, 1)
            )

            if self.qkv_format == "thd":
                query_layer, key_layer, value_layer = (
                    x.reshape(x.size(0), -1, self.hidden_size_per_attention_head)
                    for x in (query_layer, key_layer, value_layer)
                )
            else:
                # query: -> [sq, b, np, hn]
                # key, value: -> [sq, b, ng, hn]
                query_layer, key_layer, value_layer = (
                    x.reshape(x.size(0), x.size(1), -1, self.hidden_size_per_attention_head)
                    for x in (query_layer, key_layer, value_layer)
                )
        elif self.attention_type == "cross":
            # Attention heads [sk, b, h] --> [sk, b, (ng * 2 * hn)]
            mixed_kv_layer = self.key_value(
                encoder_output,
                is_first_microbatch=is_first_microbatch,
                fp8_output=fp8_mha and rotary_pos_emb is None,
            )

            if self.qkv_weight_interleaved:
                # [sq, b, (ng * 2 * hn)] --> [sq, b, ng, 2 * hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    self.num_gqa_groups_per_partition,
                    2 * self.hidden_size_per_attention_head,
                )
                # split along last dimension
                split_dim = -1
            else:
                # [sq, b, (ng * 2 * hn)] --> [sq, b, 2 * ng, hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    2 * self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
                # split along second last dimension
                split_dim = -2

            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # mixed_kv_layer --> 2 [sk, b, ng, hn]
            key_layer, value_layer = _SplitAlongDim.apply(
                mixed_kv_layer,
                split_dim,
                mixed_kv_layer.shape[split_dim] // 2,
            )
            key_layer, value_layer = (
                x.reshape(
                    x.size(0),
                    x.size(1),
                    -1,
                    self.hidden_size_per_attention_head,
                )
                for x in (key_layer, value_layer)
            )

            # Attention head [sq, b, h] --> [sq, b, hp]
            if self.input_layernorm:
                layernorm_query_outputs = self.layernorm_query(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                    fp8_output=fp8_mha and rotary_pos_emb is None,
                )
                if self.return_layernorm_output:
                    query_layer, layernorm_output = layernorm_query_outputs
                else:
                    query_layer = layernorm_query_outputs
            else:
                query_layer = self.query_layer(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                    fp8_output=fp8_mha and rotary_pos_emb is None,
                )

            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        # ======================================================
        # Apply relative positional encoding (rotary embedding)
        # ======================================================

        if rotary_pos_emb is not None:
            assert not isinstance(query_layer, Float8Tensor) and not isinstance(
                key_layer, Float8Tensor
            ), "RoPE is not supported for Float8Tensors!"
            # duplicate the pos_emb for self attention
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2

            q_pos_emb, k_pos_emb = rotary_pos_emb

            # adjust key and value for inference
            if inference_params is not None:
                if self.qkv_format == "sbhd":
                    sequence_length = key_layer.size(0)
                elif self.qkv_format == "bshd":
                    sequence_length = key_layer.size(1)
                else:
                    raise ValueError(f"QKV format {self.qkv_format} not supported for KV caching.")

                sequence_start = inference_params.sequence_len_offset
                sequence_end = sequence_start + sequence_length

                q_pos_emb = q_pos_emb[sequence_start:sequence_end, ...]
                k_pos_emb = k_pos_emb[sequence_start:sequence_end, ...]

            query_layer = apply_rotary_pos_emb(
                query_layer,
                q_pos_emb,
                self.qkv_format,
                fused=True,
                cu_seqlens=cu_seqlens_q,
                cp_size=self.cp_size,
                cp_rank=self.cp_rank,
            )
            key_layer = apply_rotary_pos_emb(
                key_layer,
                k_pos_emb,
                self.qkv_format,
                fused=True,
                cu_seqlens=cu_seqlens_kv,
                cp_size=self.cp_size,
                cp_rank=self.cp_rank,
            )

        # ===========================
        # Core attention computation
        # ===========================

        context_layer = self.core_attention(
            query_layer,
            key_layer,
            value_layer,
            qkv_format=self.qkv_format,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            checkpoint_core_attention=checkpoint_core_attention,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
            alibi_slopes=alibi_slopes,
            fast_zero_fill=fast_zero_fill,
            inference_params=inference_params,
        )

        # ===================
        # Output. [sq, b, h]
        # ===================
        projection_output = self.proj(
            context_layer,
            is_first_microbatch=is_first_microbatch,
            fp8_grad=isinstance(context_layer, QuantizedTensor),
        )

        if self.return_bias:
            attention_output, attention_bias = projection_output
        else:
            attention_output, attention_bias = projection_output, None

        outputs = (attention_output,)
        if self.return_bias:
            outputs += (attention_bias,)
        if self.input_layernorm and self.return_layernorm_output:
            outputs += (layernorm_output,)
        return outputs if len(outputs) > 1 else outputs[0]
