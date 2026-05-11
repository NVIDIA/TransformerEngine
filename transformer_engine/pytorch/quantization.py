# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Quantization utilities for TransformerEngine"""
from __future__ import annotations

import abc
import dataclasses
import warnings
import os
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import deque
from typing import Callable, List, Optional, Dict, Any, Tuple, Union

import torch
import transformer_engine_torch as tex
from transformer_engine.common.recipe import (
    Recipe,
    DelayedScaling,
    Format,
    MXFP8BlockScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    NVFP4BlockScaling,
    CustomRecipe,
)
from .constants import dist_group_type

from .utils import get_device_compute_capability
from .jit import jit_fuser


__all__ = [
    "autocast",
    "quantized_model_init",
    "is_fp8_available",
    "is_mxfp8_available",
    "is_fp8_block_scaling_available",
    "is_nvfp4_available",
    "get_default_recipe",
    "get_align_size_for_quantization",
    "QuantizerRole",
    "QuantizerRequest",
    "DelayedScalingRequest",
]


_FP8_SUPPORT: Optional[Tuple[bool, str]] = None
_MXFP8_SUPPORT: Optional[Tuple[bool, str]] = None
_NVFP4_SUPPORT: Optional[Tuple[bool, str]] = None
_FP8_BLOCK_SCALING_SUPPORT: Optional[Tuple[bool, str]] = None


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


def _compute_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 0):  # hopper and above
        return True, ""
    if get_device_compute_capability() < (8, 9):  # pre-ada
        return False, "Device compute capability 8.9 or higher required for FP8 execution."
    if tex.get_cublasLt_version() < 120103:
        return False, "CublasLt version 12.1.3.x or higher required for FP8 execution on Ada."
    if float(torch.version.cuda) < 12.1:
        return False, "Cuda version 12.1 or higher required for FP8 execution on Ada."
    return True, ""


def _compute_mxfp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (12, 0):
        return False, "MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet."
    if get_device_compute_capability() >= (10, 0):  # blackwell and above
        return True, ""
    return False, "Device compute capability 10.0 or higher required for MXFP8 execution."


def _compute_nvfp4_support() -> Tuple[bool, str]:
    """Return if nvfp4 support is available"""
    if get_device_compute_capability() >= (10, 0):  # blackwell and above
        return True, ""
    return False, "Device compute capability 10.0 or higher required for NVFP4 execution."


def _compute_fp8_block_scaling_support() -> Tuple[bool, str]:
    """Return if fp8 block scaling support is available"""
    if get_device_compute_capability() >= (9, 0) and float(torch.version.cuda) >= 12.9:
        return True, ""
    return (
        False,
        "FP8 block scaled GEMM requires compute capability 9.0 or higher and CUDA >= 12.9.",
    )


@torch.compiler.assume_constant_result
def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available."""
    global _FP8_SUPPORT
    if _FP8_SUPPORT is None:
        _FP8_SUPPORT = _compute_fp8_support()
    return _FP8_SUPPORT


@torch.compiler.assume_constant_result
def check_mxfp8_support() -> Tuple[bool, str]:
    """Return if MXFP8 support is available."""
    global _MXFP8_SUPPORT
    if _MXFP8_SUPPORT is None:
        _MXFP8_SUPPORT = _compute_mxfp8_support()
    return _MXFP8_SUPPORT


@torch.compiler.assume_constant_result
def check_nvfp4_support() -> Tuple[bool, str]:
    """Return if NVFP4 support is available."""
    global _NVFP4_SUPPORT
    if _NVFP4_SUPPORT is None:
        _NVFP4_SUPPORT = _compute_nvfp4_support()
    return _NVFP4_SUPPORT


@torch.compiler.assume_constant_result
def check_fp8_block_scaling_support() -> Tuple[bool, str]:
    """Return if fp8 block scaling support is available."""
    global _FP8_BLOCK_SCALING_SUPPORT
    if _FP8_BLOCK_SCALING_SUPPORT is None:
        _FP8_BLOCK_SCALING_SUPPORT = _compute_fp8_block_scaling_support()
    return _FP8_BLOCK_SCALING_SUPPORT


def check_recipe_support(recipe: Recipe) -> None:
    """Check if the given recipe is supported."""
    if torch.compiler.is_compiling() and isinstance(recipe, DelayedScaling):
        raise RuntimeError(
            "DelayedScaling is not supported under torch.compile. Please use other recipes instead."
        )
    recipe_supported = True
    unsupported_reason = ""
    if isinstance(recipe, (DelayedScaling, Float8CurrentScaling)):
        recipe_supported, unsupported_reason = check_fp8_support()
    elif isinstance(recipe, Float8BlockScaling):
        recipe_supported, unsupported_reason = check_fp8_block_scaling_support()
    elif isinstance(recipe, MXFP8BlockScaling):
        recipe_supported, unsupported_reason = check_mxfp8_support()
    if not recipe_supported:
        raise RuntimeError(unsupported_reason)


def get_default_fp8_recipe() -> Recipe:
    """FP8 recipe with default args."""
    assert not torch.compiler.is_compiling(), (
        "Creating Recipe objects inside compiled regions is not supported because "
        "their construction is not traceable. "
        "Pass an explicit recipe to te.autocast() instead."
    )
    if check_mxfp8_support()[0]:
        return MXFP8BlockScaling()
    if get_device_compute_capability() >= (12, 0):
        # This is a temporary restriction until MXFP8 is supported for all gemm layouts.
        return Float8CurrentScaling()
    return DelayedScaling()


def get_default_recipe() -> Recipe:
    """Returns the default training recipe based on available device."""
    return get_default_fp8_recipe()


def get_align_size_for_quantization(recipe: Recipe) -> int:
    """Get the alignment size for quantization."""
    if recipe.mxfp8():
        return 32
    if recipe.nvfp4():
        return 128
    return 16


def get_fp8_torch_dtype(fp8_recipe: Recipe, fprop_tensor: bool = True) -> torch.dtype:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return torch.float8_e4m3fn
    return torch.float8_e5m2


def get_fp8_te_dtype(fp8_recipe: Recipe, fprop_tensor: bool = True) -> tex.DType:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return tex.DType.kFloat8E4M3
    return tex.DType.kFloat8E5M2


def get_fp4_te_dtype(fp4_recipe: Recipe) -> tex.DType:
    """Get fp4 data type according to recipe and tensor"""
    if fp4_recipe.fp4_format == Format.E2M1:
        return tex.DType.kFloat4E2M1
    raise ValueError(f"Unsupported FP4 format: {fp4_recipe.fp4_format}")


def get_fp8_max(fp8_recipe: Recipe, fprop_tensor: bool = True) -> tex.DType:
    """Get max representible FP8 value."""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return Format.E4M3.value.max_fwd
    return Format.E5M2.value.max_fwd


def is_fp8_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if FP8 support is available for the delayed
    scaling and per tensor current scaling recipe.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available. The reason
        will be an empty string if support for FP8 is available.

    """
    if return_reason:
        return check_fp8_support()
    return check_fp8_support()[0]


def is_mxfp8_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if support is available for the MXFP8 recipe.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available. The reason
        will be an empty string if support for MXFP8 is available.

    """
    if return_reason:
        return check_mxfp8_support()
    return check_mxfp8_support()[0]


def is_fp8_block_scaling_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if support is available for the FP8 block scaling recipe.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available. The reason
        will be an empty string if support for FP8 block scaling is available.

    """
    if return_reason:
        return check_fp8_block_scaling_support()
    return check_fp8_block_scaling_support()[0]


def is_nvfp4_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if support is available for the NVFP4 recipe.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available. The reason
        will be an empty string if support for NVFP4 is available.

    """
    if return_reason:
        return check_nvfp4_support()
    return check_nvfp4_support()[0]


@dataclass(slots=True)
class FP8GlobalState:
    """Mutable process-global FP8 state stored on an instance.

    Using an instance avoids class-level `setattr(type, ...)` writes, which
    `torch.compile` cannot trace in fullgraph mode.
    """

    fp8_enabled: bool = False
    fp8_calibration: bool = False
    fp8_recipe: Optional[Recipe] = None
    fp8_distributed_group: Optional[dist_group_type] = None
    fp8_parameters: bool = False
    high_precision_init_val: bool = False
    is_first_fp8_module: bool = False
    fp8_graph_capturing: bool = False
    autocast_depth: int = 0
    global_amax_buffer: Dict[str, list] = field(default_factory=dict)
    global_amax_history_buffer: Dict[str, list] = field(default_factory=dict)
    global_scale_buffer: Dict[str, list] = field(default_factory=dict)
    fp8_tensors_recompute_buffer: list = field(default_factory=list)
    autocast_arguments: Dict[Any, Tuple[Recipe, Optional[dist_group_type]]] = field(
        default_factory=dict
    )
    skip_fp8_weight_update_tensor: Optional[torch.Tensor] = None


class FP8GlobalStateManager:
    """Class to keep track of and manipulate the global
    FP8 state at different stages of execution.
    """

    quantization_state = FP8GlobalState()

    @classmethod
    def reset(cls) -> None:
        """Reset the global state"""
        cls.quantization_state = FP8GlobalState()

    @classmethod
    def is_fp8_available(cls) -> Tuple[bool, str]:
        """Return if fp8 support is available"""
        return check_fp8_support()

    @classmethod
    def is_mxfp8_available(cls) -> Tuple[bool, str]:
        """Return if MXFP8/current scaling support is available."""
        return check_mxfp8_support()

    @classmethod
    def is_fp8_block_scaling_available(cls) -> Tuple[bool, str]:
        """Return if Float8 block scaling support is available."""
        return check_fp8_block_scaling_support()

    @classmethod
    def is_nvfp4_available(cls) -> Tuple[bool, str]:
        """Return if NVFP4 support is available."""
        return check_nvfp4_support()

    @staticmethod
    def get_meta_tensor_key(forward: bool = True) -> str:
        """Returns scaling key in `fp8_meta`."""
        if forward:
            return "scaling_fwd"
        return "scaling_bwd"

    @staticmethod
    def get_fwd_bwd_key(forward: bool = True) -> str:
        """Convert bool `forward` to string."""
        return "forward" if forward else "backward"

    @classmethod
    def get_buffer_info(cls) -> str:
        """
        Returns a key for `fp8_meta` that stores the module's index
        in the global buffers along with autocast information.
        """
        return "buffer_index_and_autocast_key"

    @classmethod
    def get_key_in_buffer(
        cls,
        forward: bool,
        fp8_recipe: Recipe,
        fp8_group: dist_group_type,
    ) -> str:
        """Returns a key into the global FP8 buffers."""
        autocast_key = cls.get_unique_autocast_key(fp8_recipe, fp8_group)
        fwd_bwd_key = cls.get_fwd_bwd_key(forward)
        return f"{fwd_bwd_key}_{autocast_key}"

    @classmethod
    def split_key_in_buffer(cls, key: str) -> Tuple[bool, str]:
        """Splits buffer key into relevant parts."""
        forward, autocast_key = key.split("_", 1)
        forward = forward == "forward"
        return forward, autocast_key

    @classmethod
    def add_fp8_tensors_to_global_buffer(
        cls,
        fp8_meta: Dict[str, Any],
    ) -> None:
        """
        Delayed scaling only (built-in or custom recipe with DS requests).

        The amax reduction process happens completely outside the FP8 modules.
        To participate in the reduction, the only role played by a module is
        to call this function in order to append it's FP8 tensor into a global
        buffer. There are 5 global buffers maintained, one each for amax, amax
        history, scale, scale-inverse, and non-weight-mask. Each buffer has
        keys that hold FP8 tensors. Keys have a `forward_` or `backward_` prefix
        to indicate the type of FP8 tensor, since the forward and backward
        reductions happen separately.

        Note: For CG capture, this method is called from the graphed
        wrapper. For non CG case, it's called from within the module.
        """

        # noop unless delayed scaling state is present
        if not _has_delayed_scaling_state(fp8_meta):
            return

        # Every module must call this function exactly once since
        # the amax tensors are static. Ensures that compatibility
        # with non-graphed modules is maintained.
        index_in_buffer = cls.get_buffer_info()  # Same index for fwd/bwd fp8 tensors.
        if index_in_buffer in fp8_meta:
            return

        fp8_meta[index_in_buffer] = []
        qstate = cls.quantization_state
        for forward in (True, False):
            fp8_meta_tensor_key = cls.get_meta_tensor_key(forward=forward)
            if fp8_meta_tensor_key not in fp8_meta:
                # Handles non-parameter FP8 modules, e.g. DPA.
                continue

            state = fp8_meta[fp8_meta_tensor_key]

            # Determine recipe + buffers: built-in DS or custom with DS requests
            if isinstance(state, CustomRecipeState) and state._has_delayed_scaling:
                inner_recipe = state._inner_delayed_scaling_recipe
                key = cls.get_key_in_buffer(forward, inner_recipe, fp8_meta["fp8_group"])
                # Register inner recipe in autocast_arguments for reduction
                autocast_key = cls.get_unique_autocast_key(inner_recipe, fp8_meta["fp8_group"])
                qstate.autocast_arguments[autocast_key] = (inner_recipe, fp8_meta["fp8_group"])
            else:
                key = cls.get_key_in_buffer(forward, fp8_meta["recipe"], fp8_meta["fp8_group"])

            if key not in qstate.global_amax_buffer:
                qstate.global_amax_buffer[key] = [fp8_meta[fp8_meta_tensor_key].amax_history[0]]
                qstate.global_amax_history_buffer[key] = [
                    fp8_meta[fp8_meta_tensor_key].amax_history
                ]
                qstate.global_scale_buffer[key] = [fp8_meta[fp8_meta_tensor_key].scale]
            else:
                qstate.global_amax_buffer[key].append(fp8_meta[fp8_meta_tensor_key].amax_history[0])
                qstate.global_amax_history_buffer[key].append(
                    fp8_meta[fp8_meta_tensor_key].amax_history
                )
                qstate.global_scale_buffer[key].append(fp8_meta[fp8_meta_tensor_key].scale)
            fp8_meta[index_in_buffer].append(len(qstate.global_amax_buffer[key]) - 1)
            fp8_meta[index_in_buffer].append(key)

    @classmethod
    def is_fp8_enabled(cls) -> bool:
        """Is FP8 enabled"""
        return cls.quantization_state.fp8_enabled

    @classmethod
    def is_fp8_calibration(cls) -> bool:
        """Is FP8 calibration"""
        return cls.quantization_state.fp8_calibration

    @classmethod
    def with_fp8_parameters(cls) -> bool:
        """Should the parameters be stored as FP8"""
        return cls.quantization_state.fp8_parameters

    @classmethod
    def with_high_precision_init_val(cls) -> bool:
        """Should the high precision initial values be stored with FP8 parameters"""
        return cls.quantization_state.high_precision_init_val

    @classmethod
    def fp8_graph_capturing(cls) -> bool:
        """Is CUDA graph capture under way?"""
        if torch.compiler.is_compiling():
            assert not cls.quantization_state.fp8_graph_capturing
            return False
        return (
            cls.quantization_state.fp8_graph_capturing or torch.cuda.is_current_stream_capturing()
        )

    @classmethod
    def is_first_fp8_module(cls):
        """Returns `True` only the first time when called multiple
        times from within the same `autocast` context.
        """
        tmp = cls.quantization_state.is_first_fp8_module
        cls.quantization_state.is_first_fp8_module = False
        return tmp

    @classmethod
    def get_fp8_recipe(cls) -> Recipe:
        """Return the fp8 recipe"""
        if cls.quantization_state.fp8_recipe is not None:
            return cls.quantization_state.fp8_recipe
        return get_default_fp8_recipe()

    @classmethod
    def get_fp8_group(cls) -> Union[dist_group_type, None]:
        """Return the fp8 group for scale/amax comm"""
        return cls.quantization_state.fp8_distributed_group

    @classmethod
    def get_autocast_state(cls) -> tuple:
        """Snapshot the autocast-related fields of the quantization state."""
        qstate = cls.quantization_state
        return (
            qstate.fp8_enabled,
            qstate.fp8_calibration,
            qstate.fp8_recipe,
            qstate.fp8_distributed_group,
            qstate.is_first_fp8_module,
            qstate.fp8_graph_capturing,
        )

    @classmethod
    def set_autocast_state(cls, state: tuple) -> None:
        """Restore a previously saved autocast state snapshot."""
        qstate = cls.quantization_state
        (
            qstate.fp8_enabled,
            qstate.fp8_calibration,
            qstate.fp8_recipe,
            qstate.fp8_distributed_group,
            qstate.is_first_fp8_module,
            qstate.fp8_graph_capturing,
        ) = state

    @staticmethod
    def reduce_tensor_across_group_op_max(tensor: torch.Tensor, group: dist_group_type) -> None:
        """Reduce tensor across given group."""
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=group,
                async_op=False,
            )

    @classmethod
    def reduce_and_update_fp8_tensors(
        cls,
        forward: bool = True,
    ) -> None:
        """Delayed scaling only. Concatenate, reduce, and split amaxes in the global buffer."""
        # global_amax_buffer should only be non-empty for fp8 delayed scaling
        qstate = cls.quantization_state
        for (
            buffer_key,
            amax_buffer,
        ) in qstate.global_amax_buffer.items():
            # Check for forward or backward reduction.
            fwd_update, autocast_key = cls.split_key_in_buffer(buffer_key)
            if fwd_update != forward:
                continue
            if len(amax_buffer) == 0:
                continue

            # Retrieve autocast specific args and concat amaxes.
            recipe, group = qstate.autocast_arguments[autocast_key]
            contiguous_amax = torch.cat(amax_buffer)

            # Reduction.
            if (
                recipe.reduce_amax
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size(group=group) > 1
            ):
                cls.reduce_tensor_across_group_op_max(contiguous_amax, group)

            # Amax and scale update.
            unfused_update = (
                bool(int(os.getenv("NVTE_UNFUSED_FP8_UPDATE", "0")))
                or callable(recipe.amax_compute_algo)
                or callable(recipe.scaling_factor_compute_algo)
            )

            if not unfused_update:
                tex.fused_amax_and_scale_update_after_reduction(
                    contiguous_amax,
                    qstate.global_amax_history_buffer[buffer_key],
                    qstate.global_scale_buffer[buffer_key],
                    recipe.amax_compute_algo,
                    get_fp8_te_dtype(recipe, forward),
                    recipe.margin,
                )
            else:
                split_and_copy(contiguous_amax, amax_buffer, [x.numel() for x in amax_buffer])

                for amax_history, scale in zip(
                    qstate.global_amax_history_buffer[buffer_key],
                    qstate.global_scale_buffer[buffer_key],
                ):
                    _amax_and_scale_update(
                        amax_history, scale, get_fp8_max(recipe, forward), recipe
                    )

    @classmethod
    def get_unique_autocast_key(
        cls,
        recipe: Optional[Recipe] = None,
        group: Optional[dist_group_type] = None,
    ):
        """
        For FP8, each autocast can be uniquely identified by the recipe and fp8 group.
        Object identity is sufficient since autocast contexts never outlive a single
        training session.
        """
        return str((str(recipe), id(group) if group is not None else None))

    @classmethod
    def autocast_enter(
        cls,
        enabled: bool = False,
        calibrating: bool = False,
        fp8_recipe: Optional[Recipe] = None,
        fp8_group: Optional[dist_group_type] = None,
        _graph: bool = False,
    ) -> None:
        """Set state and tracking variables for entry into FP8 region."""

        fp8_recipe = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe
        autocast_key = cls.get_unique_autocast_key(fp8_recipe, fp8_group)
        qstate = cls.quantization_state
        qstate.autocast_arguments[autocast_key] = (
            fp8_recipe,
            fp8_group,
        )

        qstate.fp8_enabled = enabled
        qstate.fp8_calibration = calibrating
        qstate.fp8_recipe = fp8_recipe
        qstate.fp8_distributed_group = fp8_group
        qstate.fp8_graph_capturing = _graph

        if qstate.autocast_depth == 0:
            qstate.is_first_fp8_module = True
        qstate.autocast_depth += 1

        if enabled:
            fp8_available, reason_for_no_fp8 = cls.is_fp8_available()
            assert fp8_available, reason_for_no_fp8
            if isinstance(fp8_recipe, MXFP8BlockScaling):
                mxfp8_available, reason_for_no_mxfp8 = cls.is_mxfp8_available()
                assert mxfp8_available, reason_for_no_mxfp8
            if isinstance(fp8_recipe, Float8BlockScaling):
                fp8_block_available, reason_for_no_fp8_block = cls.is_fp8_block_scaling_available()
                assert fp8_block_available, reason_for_no_fp8_block
            if isinstance(fp8_recipe, NVFP4BlockScaling):
                nvfp4_available, reason_for_no_nvfp4 = cls.is_nvfp4_available()
                assert nvfp4_available, reason_for_no_nvfp4

    @classmethod
    def autocast_exit(cls, enabled: bool, _graph: bool) -> None:
        """Set state and tracking variables for exit from FP8 region."""
        qstate = cls.quantization_state
        qstate.autocast_depth -= 1
        # Reduce only the non-FP8 weight modules here.
        # FP8 weight modules are reduced at the end of the optimizer
        # step after the weight amax is populated.
        if enabled and qstate.autocast_depth == 0 and not _graph and torch.is_grad_enabled():
            # delayed scaling only function, for other recipes (current scaling with any granularity),
            # this is noop for other recipes because cls.global_amax_buffer is empty list
            cls.reduce_and_update_fp8_tensors(forward=True)

    @classmethod
    def copy_forward_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
        """Copy the scaling factors and amaxes for recompute forward phase
        to ensure both forward steps are numerically same.
        """

        # delayed scaling only function, noop for any other recipe
        if not _has_delayed_scaling_state(fp8_meta):
            return

        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"

        to_copy = [
            fp8_meta["scaling_fwd"].amax_history.clone(),
            fp8_meta["scaling_fwd"].scale.clone(),
        ]

        qstate = cls.quantization_state
        if buffer_position_key in fp8_meta:
            qstate.fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].append(to_copy)
        else:
            if len(qstate.fp8_tensors_recompute_buffer) == 0:
                qstate.fp8_tensors_recompute_buffer = [deque()]
            else:
                qstate.fp8_tensors_recompute_buffer.append(deque())
            qstate.fp8_tensors_recompute_buffer[-1].append(to_copy)
            fp8_meta[buffer_position_key] = len(qstate.fp8_tensors_recompute_buffer) - 1

    @classmethod
    def get_old_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
        """Switch to the copied scaling factors and amaxes from phase
        1 forward for indentical numerical outputs.
        """
        # delayed scaling only function, noop for any other recipe
        if not _has_delayed_scaling_state(fp8_meta):
            return

        # Store updated amaxes and scales from phase 1 post forward.
        fp8_meta["updated_amax_history_fwd"] = fp8_meta["scaling_fwd"].amax_history.clone()
        fp8_meta["updated_scale_fwd"] = fp8_meta["scaling_fwd"].scale.clone()

        # Retrieve stashed amaxes and scales from phase 1 pre forward.
        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
        stashed_fp8_meta = cls.quantization_state.fp8_tensors_recompute_buffer[
            fp8_meta[buffer_position_key]
        ].popleft()

        # Replace amaxes and scales with stashed values for phase 2 forward
        fp8_meta["scaling_fwd"].amax_history.copy_(stashed_fp8_meta[0])
        fp8_meta["scaling_fwd"].scale.copy_(stashed_fp8_meta[1])

    @staticmethod
    def restore_fp8_meta_tensors(fp8_meta: Dict[str, Any]) -> None:
        """Restore latest scaling factors and amaxes after recompute forward run."""
        # delayed scaling only function, noop for any other recipe
        if not _has_delayed_scaling_state(fp8_meta):
            return

        fp8_meta["scaling_fwd"].amax_history.copy_(fp8_meta["updated_amax_history_fwd"])
        fp8_meta["scaling_fwd"].scale.copy_(fp8_meta["updated_scale_fwd"])


@contextmanager
def fp8_model_init(
    enabled: bool = True,
    recipe: Optional[Recipe] = None,
    preserve_high_precision_init_val: bool = False,
) -> None:
    """
    .. warning::

       fp8_model_init is deprecated and will be removed in a future release. Use
       ``quantized_model_init(enabled=..., recipe=..., preserve_high_precision_init_val=...)`` instead.

    """

    warnings.warn(
        "fp8_model_init is deprecated and will be removed in a future release. "
        "Use quantized_model_init("
        "enabled=..., recipe=..., preserve_high_precision_init_val=...) instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    # Call new implementation.
    with quantized_model_init(
        enabled=enabled,
        recipe=recipe,
        preserve_high_precision_init_val=preserve_high_precision_init_val,
    ):
        yield


@contextmanager
def quantized_model_init(
    enabled: bool = True,
    recipe: Optional[Recipe] = None,
    preserve_high_precision_init_val: bool = False,
) -> None:
    """
    Context manager for initialization of quantized parameters.

    Example usage:

    .. code-block:: python

        with quantized_model_init(enabled=True):
            model = transformer_engine.pytorch.Linear(768, 768)

        # Preserving high precision initial value to initialize master weight
        with quantized_model_init(enabled=True, preserve_high_precision_init_val=True):
            model = transformer_engine.pytorch.Linear(768, 768)
        master_weight = model.weight.get_high_precision_init_val()
        model.weight.clear_high_precision_init_val()

    Parameters
    ----------
    enabled : bool, default = True
             when enabled, Transformer Engine modules created inside this `quantized_model_init`
             region will hold only quantized copies of its parameters, as opposed to the default
             behavior where both higher precision and quantized copies are present. Setting this
             option to `True` may result in lower memory consumption and is especially
             useful for scenarios like:

             * full model training using optimizer with master weights, where the high
               precision copies of weights are already present in the optimizer.
             * inference, where only the quantized copies of the parameters are used.
             * LoRA-like fine-tuning, where the main parameters of the model do not change.
    recipe : transformer_engine.common.recipe.Recipe, default = None
            Recipe used to create the parameters. If left to None, it uses the default recipe.
    preserve_high_precision_init_val : bool, default = False
             when enabled, store the high precision tensor used to initialize quantized parameters
             in CPU memory, and add two function attributes named `get_high_precision_init_val()`
             and `clear_high_precision_init_val()` to quantized parameters to get/clear this high
             precision tensor. The purpose is that users can use this high-precision copy
             to initialize master weights, avoiding the loss of precision that can occur when
             using quantized parameters directly. Note that after the master weights are initialized,
             users should call `clear_high_precision_init_val()` to release this CPU memory.

             This functionality is *EXPERIMENTAL*.
    """

    qstate = FP8GlobalStateManager.quantization_state
    _fp8_parameters = qstate.fp8_parameters
    _fp8_recipe = qstate.fp8_recipe
    _high_precision_init_val = qstate.high_precision_init_val
    qstate.fp8_parameters = enabled
    qstate.fp8_recipe = get_default_fp8_recipe() if recipe is None else recipe
    qstate.high_precision_init_val = preserve_high_precision_init_val
    try:
        yield
    finally:
        qstate.fp8_parameters = _fp8_parameters
        qstate.fp8_recipe = _fp8_recipe
        qstate.high_precision_init_val = _high_precision_init_val


@contextmanager
def fp8_autocast(
    enabled: bool = True,
    calibrating: bool = False,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    _graph: bool = False,
) -> None:
    """
    .. warning::

       ``fp8_autocast`` is deprecated and will be removed in a future release.
       Use ``autocast(enabled=..., calibrating=..., recipe=..., group=..., _graph=...)`` instead.

    """

    warnings.warn(
        "fp8_autocast is deprecated and will be removed in a future release. "
        "Use autocast(enabled=..., calibrating=..., recipe=..., group=..., _graph=...) instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    # Call new implementation.
    with autocast(
        enabled=enabled,
        calibrating=calibrating,
        recipe=fp8_recipe,
        amax_reduction_group=fp8_group,
        _graph=_graph,
    ):
        yield


@contextmanager
def autocast(
    enabled: bool = True,
    calibrating: bool = False,
    recipe: Optional["Recipe"] = None,
    amax_reduction_group: Optional["dist_group_type"] = None,
    _graph: bool = False,
) -> None:
    """
    Context manager for quantization schemes like FP8 or FP4.

    .. code-block:: python

        with autocast(enabled=True):
            out = model(inp)

    .. note::

        Support for FP8 in the Linear layer of Transformer Engine is currently limited to tensors
        with shapes where both dimensions are divisible by 16. In terms of the input to the full
        Transformer network, this typically requires padding sequence length to be multiple of 16.

    .. note::

        When :attr:`recipe.reduce_amax==True`, any module must not be invoked more than once
        inside a single `autocast` region. This is unsupported behavior because the amax
        reduction is handled during the exit of the `autocast` context. Calling the same
        module more than once inside an `autocast` region overrides the amax tensors
        before reduction can occur.

    Parameters
    ----------
    enabled : bool, default = True
             whether or not to enable low precision quantization (FP8/FP4).
    calibrating : bool, default = False
                 calibration mode allows collecting statistics such as amax and scale
                 data of quantized tensors even when executing without quantization enabled.
                 This is useful for saving an inference ready checkpoint while training
                 using a higher precision.
    recipe : recipe.Recipe, default = None
            recipe used for low precision quantization.
    amax_reduction_group : torch._C._distributed_c10d.ProcessGroup, default = None
                          distributed group over which amaxes for the quantized tensors
                          are reduced at the end of each training step.
    """

    if enabled:
        check_recipe_support(recipe)

    # Save current state so we always restore it on exit.
    fp8_state = FP8GlobalStateManager.get_autocast_state()

    FP8GlobalStateManager.autocast_enter(
        enabled=enabled,
        calibrating=calibrating,
        fp8_recipe=recipe,
        fp8_group=amax_reduction_group,
        _graph=_graph,
    )
    try:
        yield
    finally:
        FP8GlobalStateManager.set_autocast_state(fp8_state)
        FP8GlobalStateManager.autocast_exit(enabled, _graph=_graph)


def _update_amax_history(amax_history: torch.Tensor) -> torch.Tensor:
    """Update amax history and set next amax to zero."""
    if amax_history.shape[0] > 1:
        new_amax_history = torch.roll(amax_history, -1, 0)
        amax_history.copy_(new_amax_history)
    amax_history[0].fill_(0.0)
    return amax_history


@torch.jit.script
def _default_get_amax_and_update_history(
    amax_history: torch.Tensor,
    amax_compute_algo: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Default function to obtain amax from history."""
    if amax_compute_algo == "max":
        amax = torch.max(amax_history, dim=0).values
    else:  # amax_compute_algo == "most_recent"
        amax = amax_history[0].clone()

    amax_history = _update_amax_history(amax_history)
    return amax_history, amax


@jit_fuser
def _default_sf_compute(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    margin: int,
    _fp32_max: float = torch.finfo(torch.float32).max,  # finfo not available in jitter
) -> torch.Tensor:
    """Default function to convert amax to scaling factor.
    Computing the scaling factor requires consideration of the following scenarios:
    1. amax == 0:
       No action is possible, set scale to the previous scale (or 1).
    2. 0 < amax < tiny_amax
       The amax is too tiny that the scale becomes infinite in FP32.
       Set scale = FP32_max
    3. tiny_amax <= amax < FP32_max:
       Set scale = FP8_max (or scaled_max) / amax
    4. When amax == inf or amax == nan:
       No action is possible, set scale to the previous scale (or 1).
    """
    sf = (fp8_max / amax) / (2**margin)
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    sf = torch.where(torch.isinf(sf), torch.full_like(sf, _fp32_max), sf)
    scale.copy_(sf)
    return scale


def _compute_amax_and_update_history(
    amax_history: torch.Tensor,
    amax_compute_algo: Union[Callable, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the amax from the history."""

    if callable(amax_compute_algo):
        amax = amax_compute_algo(amax_history)
        amax_history = _update_amax_history(amax_history)
        return amax_history, amax
    return _default_get_amax_and_update_history(
        amax_history,
        amax_compute_algo,
    )


def _compute_scaling_factor(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    recipe: DelayedScaling,
) -> torch.Tensor:
    """Convert amax to scaling factor."""

    if recipe.scaling_factor_compute_algo is None:
        return _default_sf_compute(
            amax,
            scale,
            fp8_max,
            recipe.margin,
        )
    return recipe.scaling_factor_compute_algo(amax, scale, fp8_max, recipe)


def _amax_and_scale_update(
    amax_history: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    recipe: DelayedScaling,
) -> None:
    """Updates FP8 meta tensors."""
    new_amax_history, amax = _compute_amax_and_update_history(
        amax_history,
        recipe.amax_compute_algo,
    )
    new_scale = _compute_scaling_factor(amax, scale, fp8_max, recipe)
    scale.copy_(new_scale)
    amax_history.copy_(new_amax_history)


def split_and_copy(
    buffer: torch.Tensor,
    outputs: List[torch.Tensor],
    chunk_sizes: List[int],
) -> None:
    """Split `buffer` by `chunk_sizes` and copy into `outputs`."""
    splits = buffer.split(chunk_sizes)
    torch._foreach_copy_(outputs, splits)


class RecipeState(abc.ABC):
    """Configuration and state for a quantization recipe.

    This is a builder class for quantizers, which are in turn builder
    classes for quantized tensors.

    This class may pack together the state for multiple quantizers,
    which is helpful for applying fused kernels with less overhead.

    Subclasses that own mutable training buffers (e.g. delayed scaling's
    ``scale`` / ``amax_history``) MUST list them in
    :attr:`_persistent_state_buffers`. These buffers are preserved across
    role-driven rebuilds and post-checkpoint resume via
    :meth:`inherit_state_from`. Stateless subclasses leave the attribute
    empty.
    """

    roles: Optional[List[QuantizerRole]]
    mode: str

    # Names of mutable torch.Tensor attributes that represent persistent
    # training state (e.g. running scale, amax history). The default
    # ``inherit_state_from`` rebinds these from a predecessor RecipeState
    # so external references (e.g. ``FP8GlobalStateManager`` reduction
    # buffers) keep pointing at the same backing tensor.
    _persistent_state_buffers: Tuple[str, ...] = ()

    # Canonical tensor types that a recipe state can dispatch on.
    _KNOWN_TENSOR_TYPES = ("input", "weight", "output", "grad_output", "grad_input")
    # Positional fallback used when no role information is available: the
    # tensor type at slot ``i`` defaults to ``_FWD_DEFAULT_TENSOR_TYPES[i % len]``
    # (forward) or ``_BWD_DEFAULT_TENSOR_TYPES[i % len]`` (backward). Mirrors
    # the ``[input, weight, output, ...]`` / ``[grad_output, grad_input, ...]``
    # convention assumed by ``module/base.py::set_meta_tensor``.
    _FWD_DEFAULT_TENSOR_TYPES = ("input", "weight", "output")
    _BWD_DEFAULT_TENSOR_TYPES = ("grad_output", "grad_input")

    @staticmethod
    def _validate_roles(
        roles: Optional[List[QuantizerRole]],
        num_quantizers: int,
    ) -> None:
        """Validate that ``roles``, if provided, has length ``num_quantizers``."""
        if roles is not None and len(roles) != num_quantizers:
            raise ValueError(
                "RecipeState requires roles to match num_quantizers "
                f"({len(roles)=} vs {num_quantizers=})"
            )

    def _slot_role(self, idx: int) -> QuantizerRole:
        """Resolve slot ``idx`` to a non-``None`` :class:`QuantizerRole`.

        This is the field-agnostic primitive that role-driven recipe states
        use to dispatch on any combination of role fields (``tensor_type``,
        ``module_type``, ``name``, future fields).

        Resolution rules:

        * If a real ``QuantizerRole`` was provided for this slot, it is
          returned unchanged. Producers fill only the fields they know about;
          the rest carry the dataclass defaults (empty strings). Consumers
          should treat an empty field as "no signal" rather than as "no role
          provided".
        * Otherwise (whole ``roles`` list missing, or this slot is ``None``),
          a bare ``QuantizerRole()`` with all fields empty is returned.
          Field-specific fallback policies belong to the individual
          dispatch convenience accessors (e.g. :meth:`_slot_tensor_type`),
          not to this primitive — that way a future recipe state that
          dispatches on, say, ``module_type`` is free to define its own
          fallback policy without impacting tensor-type dispatch.

        The "real role vs bare-default role" distinction is hidden from
        dispatch logic here. Recipe states that need to *warn* on missing
        roles (as :class:`CustomRecipeState` does) should consult
        ``self.roles[idx]`` directly.
        """
        if self.roles is not None:
            role = self.roles[idx]
            if role is not None:
                return role
        return QuantizerRole()

    def _slot_tensor_type(self, idx: int) -> str:
        """Convenience accessor: tensor-type dispatch with positional fallback.

        Resolves to one of :attr:`_KNOWN_TENSOR_TYPES`. Used by recipe states
        whose dispatch only depends on the tensor's role within a GEMM
        (input / weight / output / grad_output / grad_input), e.g.
        Float8BlockScalingRecipeState, NVFP4BlockScalingRecipeState.

        Behavior:

        * If the resolved :meth:`_slot_role` carries a ``tensor_type`` in
          :attr:`_KNOWN_TENSOR_TYPES`, return it.
        * Otherwise (no role provided, a role with empty / non-canonical
          ``tensor_type`` like DPA's ``"qkv"``, or a role that intentionally
          only sets           ``module_type``/``name``), fall back to the positional
          default (forward: ``[input, weight, output, ...]``;
          backward: ``[grad_output, grad_input, ...]``) indexed by
          ``idx % len(default_tensor_types)``.

        This fallback policy is local to tensor-type dispatch; it does not
        affect :meth:`_slot_role` or any other accessor.
        """
        role = self._slot_role(idx)
        if role.tensor_type in self._KNOWN_TENSOR_TYPES:
            return role.tensor_type
        # Positional fallback: tensor_type is missing or non-canonical.
        default_tensor_types = (
            self._FWD_DEFAULT_TENSOR_TYPES
            if self.mode == "forward"
            else self._BWD_DEFAULT_TENSOR_TYPES
        )
        return default_tensor_types[idx % len(default_tensor_types)]

    @staticmethod
    def create(
        recipe: Recipe,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[QuantizerRole]] = None,
    ) -> RecipeState:
        """Factory method to create the state for a quantization recipe

        Parameters
        ----------
        recipe: Recipe
            Quantization recipe.
        mode: {"forward", "backward"}
            Training stage where quantization will be performed.
        num_quantizers: int, default = 1
            Number of quantizers to create state for.
        device: torch.device, default = default CUDA device
            Device for quantized tensors.
        roles: list of QuantizerRole, optional
            Semantic roles for each quantizer slot. When provided, must
            have length ``num_quantizers``.

        Returns
        -------
        RecipeState:
            Quantization recipe state.

        """

        cls = None
        if recipe.delayed():
            cls = DelayedScalingRecipeState
        elif recipe.mxfp8():
            cls = MXFP8BlockScalingRecipeState
        elif recipe.float8_current_scaling():
            cls = Float8CurrentScalingRecipeState
        elif recipe.float8_block_scaling():
            cls = Float8BlockScalingRecipeState
        elif recipe.nvfp4():
            cls = NVFP4BlockScalingRecipeState
        elif recipe.custom():
            cls = CustomRecipeState
        else:
            raise ValueError(f"{recipe.__class__.__name__} is not supported")
        return cls(
            recipe,
            mode=mode,
            num_quantizers=num_quantizers,
            device=device,
            roles=roles,
        )

    @abc.abstractmethod
    def make_quantizers(self) -> list:
        """Convert recipe state to quantizers.

        Quantizers are builder classes for quantized tensors. They are
        typically used to convert a high-precision tensor (e.g. in
        FP32 or BF16) into a quantized tensor (e.g. in FP8).

        """

    def inherit_state_from(self, other: "RecipeState") -> bool:
        """Take over persistent training buffers from a predecessor state.

        Used when a ``RecipeState`` is being replaced (e.g. role-driven
        rebuild, post-checkpoint resume) but its mutable buffers must
        survive. The default implementation rebinds attributes listed in
        :attr:`_persistent_state_buffers` to ``other``'s tensor objects.
        Rebinding (rather than copying values) ensures any external
        references — most importantly the
        :class:`FP8GlobalStateManager` reduction buffers — keep pointing
        at storage that is also visible to this state's quantizers, so
        amax reductions and quantization stay consistent.

        Subclasses with composed sub-states (e.g. :class:`CustomRecipeState`
        owning an inner :class:`DelayedScalingRecipeState`) override this
        to recurse / stash for later use during ``make_quantizers``.

        Returns
        -------
        bool
            ``True`` if any persistent buffer was inherited; ``False`` if
            the states are incompatible (different class, mismatched
            shapes / dtypes) and a fresh state should be used instead.
        """
        if type(self) is not type(other):
            return False
        if not self._persistent_state_buffers:
            return False
        for name in self._persistent_state_buffers:
            src = getattr(other, name)
            dst = getattr(self, name)
            if src.shape != dst.shape or src.dtype != dst.dtype:
                return False
        for name in self._persistent_state_buffers:
            setattr(self, name, getattr(other, name))
        return True


class DelayedScalingRecipeState(RecipeState):
    """State for FP8 quantization with per-tensor delayed scaling.

    Delayed scaling recipe requires a scaling factor (applied when
    casting to FP8) and a history of max-abs values ("amax") from
    recent FP8 casts for updating the scaling factor. The scale update
    is handled externally by `FP8GlobalStateManager`.

    """

    recipe: DelayedScaling
    mode: str
    dtype: tex.DType
    scale: torch.Tensor
    amax_history: torch.Tensor

    # Persistent training state inherited across role-driven rebuilds.
    # See ``RecipeState.inherit_state_from``.
    _persistent_state_buffers = ("scale", "amax_history")

    def __init__(
        self,
        recipe: DelayedScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[QuantizerRole]] = None,
    ) -> None:
        self._validate_roles(roles, num_quantizers)
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.roles = roles
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

        # Allocate buffers
        if device is None:
            device = torch.device("cuda")
        self.scale = torch.ones(num_quantizers, dtype=torch.float32, device=device)
        self.amax_history = torch.zeros(
            recipe.amax_history_len,
            num_quantizers,
            dtype=torch.float32,
            device=device,
        )

    def make_quantizers(self) -> list:
        # TODO(ksivamani); Find better design for this, adding here to avoid circular import.
        from .tensor.float8_tensor import Float8Quantizer

        return [
            Float8Quantizer(self.scale[i], self.amax_history[0][i].reshape((1,)), self.dtype)
            for i in range(self.num_quantizers)
        ]


class Float8CurrentScalingRecipeState(RecipeState):
    """Configuration for Per-tensor current scaling quantization.

    Per-tensor current quantization does not require state.

    """

    recipe: Float8CurrentScaling
    mode: str
    dtype: tex.DType
    device: torch.device

    def __init__(
        self,
        recipe: Float8CurrentScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[QuantizerRole]] = None,
    ) -> None:
        self._validate_roles(roles, num_quantizers)
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.roles = roles
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

        # Allocate buffers
        if device is None:
            device = torch.device("cuda")
        self.device = device

    def make_quantizers(self) -> list:
        from .tensor.float8_tensor import Float8CurrentScalingQuantizer

        return [
            Float8CurrentScalingQuantizer(
                self.dtype, device=self.device, force_pow_2_scales=self.recipe.use_power_2_scales
            )
            for i in range(self.num_quantizers)
        ]


class MXFP8BlockScalingRecipeState(RecipeState):
    """Configuration for MXFP8 quantization.

    MXFP8 quantization does not require state.

    """

    recipe: MXFP8BlockScaling
    mode: str
    dtype: tex.DType

    def __init__(
        self,
        recipe: MXFP8BlockScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[QuantizerRole]] = None,
    ) -> None:
        self._validate_roles(roles, num_quantizers)
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.roles = roles
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

        # Allocate buffers
        if device is None:
            device = torch.device("cuda")

    def make_quantizers(self) -> list:
        # TODO(ksivamani); Find better design for this, adding here to avoid circular import.
        from .tensor.mxfp8_tensor import MXFP8Quantizer

        return [MXFP8Quantizer(self.dtype) for i in range(self.num_quantizers)]


class Float8BlockScalingRecipeState(RecipeState):
    """Configuration for Float8BlockScaling quantization.

    Float8BlockScaling quantization does not require state,
    but different quantizers use different modes.
    """

    recipe: Float8BlockScaling
    mode: str
    qx_dtype: tex.DType
    qw_dtype: tex.DType
    qgrad_dtype: tex.DType

    def __init__(
        self,
        recipe: Float8BlockScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[QuantizerRole]] = None,
    ) -> None:
        self._validate_roles(roles, num_quantizers)
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.roles = roles
        self.qx_dtype = get_fp8_te_dtype(recipe, True)
        self.qw_dtype = get_fp8_te_dtype(recipe, True)
        self.qgrad_dtype = get_fp8_te_dtype(recipe, False)

        # Allocate buffers
        if device is None:
            device = torch.device("cuda")
        self.device = device

    def make_quantizers(self) -> list:
        """Build one ``Float8BlockQuantizer`` per slot, dispatched by tensor type.

        Per-slot behavior, resolved via :meth:`RecipeState._slot_tensor_type`:

        * ``"weight"`` uses ``recipe.fp8_quant_fwd_weight`` and
          ``recipe.w_block_scaling_dim``.
        * ``"input"`` / ``"output"`` (and any unknown forward slot) use
          ``recipe.fp8_quant_fwd_inp`` and ``recipe.x_block_scaling_dim``.
        * ``"grad_output"`` / ``"grad_input"`` (and any unknown backward slot)
          use ``recipe.fp8_quant_bwd_grad`` and ``recipe.grad_block_scaling_dim``.

        When the owning module/op provides a role list via
        ``get_quantizer_roles``, the per-slot ``tensor_type`` drives dispatch.
        Otherwise (or for boundary slots whose role is ``None``), the
        positional fallback ``[input, weight, output, ...]`` /
        ``[grad_output, grad_input, ...]`` is used. This matches the legacy
        index-based convention, so behavior is unchanged for
        modules that haven't adopted roles yet.
        """
        # TODO(ksivamani); Find better design for this, adding here to avoid circular import.
        from .tensor.float8_blockwise_tensor import Float8BlockQuantizer

        def _make(tensor_type: str) -> Float8BlockQuantizer:
            if tensor_type == "weight":
                qparams = self.recipe.fp8_quant_fwd_weight
                fp8_dtype = self.qw_dtype
                block_scaling_dim = self.recipe.w_block_scaling_dim
            elif tensor_type in ("grad_output", "grad_input"):
                qparams = self.recipe.fp8_quant_bwd_grad
                fp8_dtype = self.qgrad_dtype
                block_scaling_dim = self.recipe.grad_block_scaling_dim
            else:
                # "input", "output", or any unknown forward type fall back to
                # the input config, matching the legacy positional behavior.
                qparams = self.recipe.fp8_quant_fwd_inp
                fp8_dtype = self.qx_dtype
                block_scaling_dim = self.recipe.x_block_scaling_dim
            return Float8BlockQuantizer(
                fp8_dtype=fp8_dtype,
                rowwise=True,
                columnwise=True,
                amax_epsilon=qparams.amax_epsilon,
                force_pow_2_scales=qparams.power_2_scale,
                block_scaling_dim=block_scaling_dim,
            )

        assert self.mode in ("forward", "backward"), f"Unexpected mode {self.mode}"
        return [_make(self._slot_tensor_type(idx)) for idx in range(self.num_quantizers)]


class NVFP4BlockScalingRecipeState(RecipeState):
    """Configuration for NVFP4 quantization.

    NVFP4 quantization does not require state.

    """

    recipe: NVFP4BlockScaling
    mode: str
    dtype: tex.DType

    def __init__(
        self,
        recipe: NVFP4BlockScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[QuantizerRole]] = None,
    ) -> None:
        self._validate_roles(roles, num_quantizers)
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.roles = roles
        self.dtype = get_fp4_te_dtype(recipe)

        # Allocate buffers
        if device is None:
            device = torch.device("cuda")

    def make_quantizers(self) -> list:
        """Build one ``NVFP4Quantizer`` per slot, dispatched by tensor type.

        Per-slot behavior, resolved via :meth:`RecipeState._slot_tensor_type`:

        * Forward, ``"weight"`` -> ``recipe.fp4_quant_fwd_weight``.
        * Forward, ``"input"`` / ``"output"`` (and any unknown forward type) ->
          ``recipe.fp4_quant_fwd_inp``.
        * Backward, any slot -> ``recipe.fp4_quant_bwd_grad``.

        When the owning module/op provides a role list via
        ``get_quantizer_roles``, the per-slot ``tensor_type`` drives dispatch.
        Otherwise (or for boundary slots whose role is ``None``), the
        positional fallback ``[input, weight, output, ...]`` is used; on this
        layout slot ``idx % 3 == 1`` is always weight and the rest fall into
        the input config, matching the legacy index-based behavior.
        """
        from .tensor.nvfp4_tensor import NVFP4Quantizer

        def _qparams(tensor_type: str):
            if self.mode == "backward":
                return self.recipe.fp4_quant_bwd_grad
            if tensor_type == "weight":
                return self.recipe.fp4_quant_fwd_weight
            return self.recipe.fp4_quant_fwd_inp

        def _make(tensor_type: str) -> NVFP4Quantizer:
            qparams = _qparams(tensor_type)
            return NVFP4Quantizer(
                fp4_dtype=self.dtype,
                rowwise=True,
                columnwise=True,
                with_rht=qparams.random_hadamard_transform,
                with_post_rht_amax=qparams.random_hadamard_transform,
                with_2d_quantization=qparams.fp4_2d_quantization,
                stochastic_rounding=qparams.stochastic_rounding,
                row_scaled_nvfp4=(
                    self.mode == "forward"
                    and tensor_type != "weight"
                    and self.recipe.row_scaled_activation
                ),
            )

        if self.mode not in ("forward", "backward"):
            raise RuntimeError(f"Unexpected recipe mode ({self.mode})")

        return [_make(self._slot_tensor_type(idx)) for idx in range(self.num_quantizers)]


def _handle_delayed_scaling_requests(
    raw: list,
    device: torch.device,
    mode: str,
    *,
    existing_ds_state: Optional["DelayedScalingRecipeState"] = None,
) -> Optional["DelayedScalingRecipeState"]:
    """Detect DelayedScalingRequest items, allocate shared state, replace with real quantizers.

    All DS requests in the same RecipeState must share identical parameters.

    When ``existing_ds_state`` is provided and compatible (same dtype,
    same number of DS slots, same ``amax_history_len``), it is reused
    instead of allocating fresh buffers. Reusing preserves accumulated
    ``scale`` / ``amax_history`` across role-driven rebuilds — important
    for post-checkpoint resume and mid-training factory swaps. The
    ``Float8Quantizer`` instances built here will then view into the
    SAME tensor objects already registered with
    ``FP8GlobalStateManager``'s reduction buffers, keeping reduction
    and quantization consistent.

    Returns a ``DelayedScalingRecipeState`` owning the shared buffers, or
    ``None`` when no DS requests are present.
    """
    ds_items = [(i, r) for i, r in enumerate(raw) if isinstance(r, DelayedScalingRequest)]
    if not ds_items:
        return None

    r0 = ds_items[0][1]

    # Validate all DS requests share same params
    for idx, req in ds_items[1:]:
        for field_name in (
            "fp8_format",
            "margin",
            "amax_history_len",
            "amax_compute_algo",
            "scaling_factor_compute_algo",
            "reduce_amax",
        ):
            v0 = getattr(r0, field_name)
            vi = getattr(req, field_name)
            if v0 != vi:
                raise ValueError(
                    "All DelayedScalingRequests in one CustomRecipeState must match. "
                    f"Slot 0 has {field_name}={v0!r}, slot {idx} has {vi!r}."
                )

    # Build a real DelayedScalingRecipeState to own the shared buffers.
    inner_recipe = DelayedScaling(
        fp8_format=r0.fp8_format,
        margin=r0.margin,
        amax_history_len=r0.amax_history_len,
        amax_compute_algo=r0.amax_compute_algo,
        scaling_factor_compute_algo=r0.scaling_factor_compute_algo,
        reduce_amax=r0.reduce_amax,
    )
    n = len(ds_items)

    # Reuse a compatible existing DSRS so its scale / amax_history (and any
    # external references to them) survive the rebuild.
    expected_dtype = get_fp8_te_dtype(inner_recipe, mode == "forward")
    dsrs = None
    if existing_ds_state is not None:
        if (
            existing_ds_state.num_quantizers == n
            and existing_ds_state.dtype == expected_dtype
            and existing_ds_state.amax_history.shape[0] == r0.amax_history_len
        ):
            dsrs = existing_ds_state

    if dsrs is None:
        dsrs = DelayedScalingRecipeState(
            inner_recipe,
            mode=mode,
            num_quantizers=n,
            device=device,
        )

    # Splice Float8Quantizer instances (backed by dsrs buffers) into raw list.
    quantizers = dsrs.make_quantizers()
    for j, (idx, _req) in enumerate(ds_items):
        raw[idx] = quantizers[j]

    return dsrs


def _has_delayed_scaling_state(fp8_meta: Dict[str, Any]) -> bool:
    """Check if fp8_meta has delayed scaling state (built-in or custom)."""
    if fp8_meta["recipe"].delayed():
        return True
    if fp8_meta["recipe"].custom():
        for key in ("scaling_fwd", "scaling_bwd"):
            state = fp8_meta.get(key)
            if isinstance(state, CustomRecipeState) and state._has_delayed_scaling:
                return True
    return False


class CustomRecipeState(RecipeState):
    """State for CustomRecipe: produce quantizers per tensor.

    Stateful quantizer support:
    - Supports stateful quantizers (e.g. delayed scaling) via ``DelayedScalingRequest``.
    - The factory returns request dataclasses for stateful quantizers; TE detects them,
      allocates shared buffers, and replaces with real quantizer instances.
    - Stateful recipe state is composed via real TE recipe state objects (e.g.
      ``DelayedScalingRecipeState``), not reimplemented.
    """

    recipe: CustomRecipe
    mode: str
    num_quantizers: int
    device: Optional[torch.device]

    # -- Composed sub-states for stateful sub-recipes --
    #
    # When the qfactory returns request objects (e.g. ``DelayedScalingRequest``)
    # for a stateful built-in recipe, ``make_quantizers`` allocates a real
    # built-in ``RecipeState`` for those slots and reuses its persistent
    # buffers across role-driven rebuilds via ``inherit_state_from``. One
    # ``_<x>_state`` / ``_<x>_state_to_inherit`` pair per stateful recipe.

    # Delayed scaling (``DelayedScalingRequest`` -> ``DelayedScalingRecipeState``):
    # ``_ds_state`` owns shared ``scale`` / ``amax_history`` for DS slots in this
    # CustomRecipeState; ``_ds_state_to_inherit`` is a transient stash set by
    # ``inherit_state_from`` and consumed by the next ``make_quantizers`` call.
    _ds_state: Optional[DelayedScalingRecipeState]
    _ds_state_to_inherit: Optional[DelayedScalingRecipeState]

    def __init__(
        self,
        recipe: CustomRecipe,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[QuantizerRole]] = None,
    ) -> None:
        self._validate_roles(roles, num_quantizers)
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.roles = roles
        if device is None:
            device = torch.device("cuda")
        self.device = device

        # -- Stateful sub-state slots (initialized empty) --
        # Delayed scaling
        self._ds_state = None
        self._ds_state_to_inherit = None

        if getattr(recipe, "qfactory", None) is None:
            raise ValueError("CustomRecipe requires `qfactory`.")

    def make_quantizers(self) -> list:
        qfactory = self.recipe.qfactory

        roles = self.roles
        if roles is None:
            warnings.warn(
                "CustomRecipeState: no QuantizerRole list provided by the module/op. "
                "Falling back to bare QuantizerRole() defaults. "
                "Override get_quantizer_roles() to provide meaningful roles.",
                stacklevel=2,
            )
            roles = [QuantizerRole() for _ in range(self.num_quantizers)]

        # qfactory must return a Quantizer or QuantizerRequest for every slot.
        # None is not a valid return value — it would silently disable quantization
        # for that tensor, risking hard-to-detect performance regressions.
        # TODO(negvet): Introduce an explicit IdentityQuantizer for intentional no-op
        # quantization. Until then, None is rejected.
        raw = [qfactory(roles[i]) for i in range(self.num_quantizers)]
        for i, q in enumerate(raw):
            if q is None:
                raise ValueError(
                    f"CustomRecipe qfactory returned None for slot {i} "
                    f"(role={roles[i]}). Every slot must return a Quantizer "
                    "instance or a QuantizerRequest."
                )

        # -- Delayed scaling sub-state --
        # If a predecessor stashed a compatible inner DSRS via
        # ``inherit_state_from``, reuse it so accumulated scale / amax_history
        # survive the rebuild. Consume the stash so a subsequent
        # ``make_quantizers`` doesn't reuse it again unintentionally.
        existing_ds_state = self._ds_state_to_inherit
        self._ds_state_to_inherit = None
        self._ds_state = _handle_delayed_scaling_requests(
            raw,
            self.device,
            self.mode,
            existing_ds_state=existing_ds_state,
        )

        return raw

    def inherit_state_from(self, other: "RecipeState") -> bool:
        """Stash ``other``'s composed sub-states for reuse on next ``make_quantizers``.

        ``CustomRecipeState`` cannot inherit declaratively because its
        persistent state lives in composed sub-states (one per stateful
        sub-recipe) that are allocated only when ``make_quantizers`` runs.
        For each stateful sub-recipe we stash the predecessor's sub-state
        and let the next ``make_quantizers`` decide whether the
        predecessor's shape is compatible with the new factory output.
        """
        if not isinstance(other, CustomRecipeState):
            return False

        inherited_any = False

        # -- Delayed scaling sub-state --
        if other._ds_state is not None:
            self._ds_state_to_inherit = other._ds_state
            inherited_any = True

        return inherited_any

    # -- Delegation to composed DelayedScalingRecipeState --

    @property
    def _has_delayed_scaling(self) -> bool:
        return self._ds_state is not None

    @property
    def amax_history(self) -> Optional[torch.Tensor]:
        """Amax history from the composed delayed-scaling state, if any."""
        return self._ds_state.amax_history if self._ds_state else None

    @property
    def scale(self) -> Optional[torch.Tensor]:
        """Current scale from the composed delayed-scaling state, if any."""
        return self._ds_state.scale if self._ds_state else None

    @property
    def _inner_delayed_scaling_recipe(self) -> Optional[DelayedScaling]:
        return self._ds_state.recipe if self._ds_state else None
