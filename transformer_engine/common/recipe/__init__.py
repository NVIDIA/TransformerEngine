# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""This module provides predefined FP8 recipes."""
from __future__ import annotations
import warnings
import os
from enum import Enum
from typing import Literal, Optional, Union, Callable, NamedTuple
from pydantic.dataclasses import dataclass


class _FormatHelper(NamedTuple):
    """
    Stores max FP8 values for fprop and bprop a `Format`.
    """

    max_fwd: float
    max_bwd: float


class Format(Enum):
    """
    Supported FP8 formats.

    Values
    ------
    E4M3 :
          All FP8 tensors are in e4m3 format
    E5M2 :
          All FP8 tensors are in e5m2 format
    HYBRID :
            FP8 tensors in the forward pass are in e4m3 format,
            FP8 tensors in the backward pass are in e5m2 format
    """

    E4M3 = _FormatHelper(max_fwd=448, max_bwd=448)
    E5M2 = _FormatHelper(max_fwd=57344, max_bwd=57344)
    HYBRID = _FormatHelper(max_fwd=E4M3.max_fwd, max_bwd=E5M2.max_bwd)


@dataclass(frozen=True)
class MMParams:
    """for pytorch as an example, _scaled_mm use_fast_accum = (not use_split_accumulator)
    apply split accumulator or not, turning it on will increase accuracy but impact gemm performance,
    so only turn it on for certain gemms
    """

    use_split_accumulator: bool = True


@dataclass(frozen=True)
class QParams:
    """Quantization parameters.
    power_2_scale: use power of 2 scale parameter
    amax_epsilon: optional minimum value of abs max
    """

    power_2_scale: bool = False
    amax_epsilon: float = 0.0


class Recipe:
    """
    Base recipe class.
    """

    def mxfp8(self):
        """Whether the given recipe is MXFP8 block scaling."""
        return isinstance(self, MXFP8BlockScaling)

    def delayed(self):
        """Whether the given recipe is delayed scaling."""
        return isinstance(self, DelayedScaling)

    def float8_current_scaling(self):
        """Whether the given recipe is (per-tensor) current scaling."""
        return isinstance(self, Float8CurrentScaling)

    def float8_per_tensor_scaling(self):
        """Whether the given recipe is per-tensor scaling."""
        return isinstance(self, (DelayedScaling, Float8CurrentScaling))

    def float8_block_scaling(self):
        """Whether the given recipe is float8 blockwise scaling."""
        return isinstance(self, Float8BlockScaling)


@dataclass()
class DelayedScaling(Recipe):
    """
    Use the delayed scaling factor strategy. Use scale factor from previous
    iteration and record amax history of `amax_history_len` steps.

    Parameters
    ----------
    margin : int, default = 0
            Margin for the scaling factor computation.
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.HYBRID
                Controls the FP8 data format used during forward and backward
                pass.
    amax_history_len : int, default = 1024
                      The length of the amax history window used for
                      scaling factor computation.
    amax_compute_algo : {'max', 'most_recent', Callable}, default = 'max'
                       Algorithm used for choosing the `amax` value for the
                       scaling factor computation. There are 2 predefined
                       choices: `max` chooses the largest `amax` in the history
                       window, while `most_recent` always chooses the most recently
                       seen value. Alternatively, one may pass a function of the
                       signature:

                       .. code-block:: python

                         def amax_compute(amax_history: Tensor) -> Tensor

                       where `Tensor` is a framework tensor type.
    scaling_factor_compute_algo : Callable, default = None
                                 Algorithm used for computing the new scaling
                                 factor based on the value of `amax`. It should
                                 be a function of the signature:

                                 .. code-block:: python

                                   def scaling_factor_compute(amax: Tensor,
                                                              old_scaling_factor: Tensor,
                                                              fp8_max: Tensor,
                                                              recipe: DelayedScaling) -> Tensor

                                 where `Tensor` is a framework tensor type.
    reduce_amax: bool, default = `True`
                By default, if `torch.distributed` is initialized, the `amax` value for FP8
                tensors is reduced across the `fp8_group` (specified in the `fp8_autocast`
                call). This keeps the amaxes and scaling factors synced across the given
                distributed group. If set to `False`, this reduction is skipped and every
                GPU maintains local amaxes and scaling factors. To ensure results are
                numerically identical across checkpointing boundaries in this case, all
                ranks must checkpoint in order to store the local tensors.
    fp8_dpa: bool, default = `False`
             Whether to enable FP8 dot product attention (DPA). When the model is placed in an
             `fp8_autocast(enabled=True)` region and `fp8_dpa` is set to `True`, DPA casts the
             inputs from higher precision to FP8, performs attention in FP8, and casts tensors
             back to higher precision as outputs. FP8 DPA currently is only supported in the
             `FusedAttention` backend.
    fp8_mha: bool, default = `False`
            Whether to enable FP8 multi-head attention (MHA). When `True`, it removes the casting
            operations mentioned above at the DPA boundaries. Currently only standard MHA modules
            i.e. `LayerNormLinear/Linear + DPA + Linear`, are supported for this feature. When
            `fp8_mha = False, fp8_dpa = True`, a typical MHA module works as
            `LayerNormLinear (BF16 output) -> (cast to FP8 ) FP8 DPA (cast to BF16) -> Linear`.
            When `fp8_mha = True, fp8_dpa = True`, it becomes
            `LayerNormLinear (FP8 output) -> FP8 DPA -> Linear`.

    Notes
    -----
    * By default (when `scaling_factor_compute_algo` is left as `None`) the scaling
      factor is computed from the final `amax` value using the formula:

      .. code-block:: python

          FP8_MAX = maximum_representable_value(fp8_format)
          new_scaling_factor = (FP8_MAX / amax) / (2 ^ margin)

    * `fp8_dpa` and `fp8_mha` are Beta features, and their API and functionality are
      subject to change in future Transformer Engine releases.
    """

    margin: int = 0
    fp8_format: Format = Format.HYBRID
    amax_history_len: int = 1024
    amax_compute_algo: Union[Literal["max", "most_recent"], Callable] = "max"
    scaling_factor_compute_algo: Optional[Callable] = None
    reduce_amax: bool = True
    fp8_dpa: bool = False
    fp8_mha: bool = False

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."

    def __repr__(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"amax_history_len={self.amax_history_len}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}"
        )


@dataclass()
class Float8CurrentScaling(Recipe):
    """
    Use the per-tensor current scaling factor strategy.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.HYBRID
                Controls the FP8 data format used during forward and backward
                pass.
    """

    fp8_format: Format = Format.HYBRID
    fp8_quant_fwd_inp = QParams(power_2_scale=False, amax_epsilon=0.0)
    fp8_quant_fwd_weight = QParams(power_2_scale=False, amax_epsilon=0.0)
    fp8_quant_bwd_grad = QParams(power_2_scale=False, amax_epsilon=0.0)
    fp8_gemm_fprop: MMParams = MMParams(use_split_accumulator=False)
    fp8_gemm_dgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_gemm_wgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_dpa: bool = False
    fp8_mha: bool = False

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."
        assert (
            not self.fp8_dpa and not self.fp8_mha
        ), "FP8 attention is not supported for Float8CurrentScaling."

    def __repr__(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"fp8_quant_fwd_inp={self.fp8_quant_fwd_inp}, "
            f"fp8_quant_fwd_weight={self.fp8_quant_fwd_weight}, "
            f"fp8_quant_bwd_grad={self.fp8_quant_bwd_grad}, "
            f"fp8_gemm_fprop={self.fp8_gemm_fprop}, "
            f"fp8_gemm_dgrad={self.fp8_gemm_dgrad}, "
            f"fp8_gemm_wgrad={self.fp8_gemm_wgrad}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}"
        )


@dataclass()
class MXFP8BlockScaling(Recipe):
    """
    Use the MXFP8 scaling factor strategy.

    In this strategy, tensors are scaled in blockwise fashion. Each group
    of 32 consecutive values is scaled together using their own scaling
    factor. The type of the scaling factor is E8M0 (8 bits of exponent,
    0 bits of mantissa), equivalent to scaling by a power of 2.

    Since the scaling happens in a particular direction (either rowwise
    or columnwise), in this recipe the quantized tensor and its transpose
    are not numerically equivalent. Due to this, when Transformer Engine
    needs both the MXFP8 tensor and its transpose (e.g. to calculate both
    forward and backward pass), during the quantization both versions are
    computed from the high precision input to avoid double quantization
    errors.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.E4M3
                Controls the FP8 data format used during forward and backward
                pass.
    """

    margin: int = 0
    fp8_format: Format = Format.E4M3
    fp8_dpa: bool = False
    fp8_mha: bool = False

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."

    def __repr__(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}"
        )


@dataclass()
class Float8BlockScaling(Recipe):
    """
    Use block-wise scaling for FP8 tensors.

    In this strategy, tensors are scaled in blockwise fashion. Values within
    each block share a common scaling factor. The block dimensionality
    can be configured. The scaling factors are float32 containers. They
    will by default be constrained to powers of 2.

    Since the scaling happens in a particular direction (either rowwise
    or columnwise), the quantized tensor and its transpose are not numerically
    equivalent. Due to this, when Transformer Engine needs both the FP8 tensor
    and its transpose (e.g. to calculate both forward and backward pass),
    during the quantization both versions are computed from the high precision
    input to avoid double quantization errors.

    NOTE: To relax the default constraint that scales be powers of 2, set env variable
    NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1 to override it for the recipe defaults.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.E4M3
                Controls the FP8 data format used during forward and backward
                pass.
    """

    use_f32_scales: bool = os.getenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0") == "1"

    fp8_format: Format = Format.E4M3
    fp8_quant_fwd_inp = QParams(power_2_scale=not use_f32_scales, amax_epsilon=0.0)
    fp8_quant_fwd_weight = QParams(power_2_scale=not use_f32_scales, amax_epsilon=0.0)
    fp8_quant_bwd_grad = QParams(power_2_scale=not use_f32_scales, amax_epsilon=0.0)
    x_block_scaling_dim: int = 1
    w_block_scaling_dim: int = 2
    grad_block_scaling_dim: int = 1
    fp8_gemm_fprop: MMParams = MMParams(use_split_accumulator=True)
    fp8_gemm_dgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_gemm_wgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_dpa: bool = False
    fp8_mha: bool = False

    def __post_init__(self) -> None:
        assert self.x_block_scaling_dim in [1, 2], "Only 1D or 2D blocks supported for x"
        assert self.w_block_scaling_dim in [1, 2], "Only 1D or 2D blocks supported for w"
        assert self.grad_block_scaling_dim in [1, 2], "Only 1D or 2D blocks supported for grad"
        assert not (
            self.x_block_scaling_dim == 2 and self.w_block_scaling_dim == 2
        ), "2D by 2D block gemm not supported."
        assert not (
            self.x_block_scaling_dim == 2 and self.grad_block_scaling_dim == 2
        ), "2D by 2D block gemm not supported."
        assert not (
            self.w_block_scaling_dim == 2 and self.grad_block_scaling_dim == 2
        ), "2D by 2D block gemm not supported."
        assert self.fp8_gemm_fprop.use_split_accumulator, "Split accumulator required for fprop."
        assert self.fp8_gemm_dgrad.use_split_accumulator, "Split accumulator required for dgrad."
        assert self.fp8_gemm_wgrad.use_split_accumulator, "Split accumulator required for wgrad."
        assert (
            not self.fp8_dpa and not self.fp8_mha
        ), "FP8 attention is not supported for Float8BlockScaling."

    def __repr__(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"fp8_quant_fwd_inp={self.fp8_quant_fwd_inp}, "
            f"fp8_quant_fwd_weight={self.fp8_quant_fwd_weight}, "
            f"fp8_quant_bwd_grad={self.fp8_quant_bwd_grad}, "
            f"x_block_scaling_dim={self.x_block_scaling_dim}, "
            f"w_block_scaling_dim={self.w_block_scaling_dim}, "
            f"grad_block_scaling_dim={self.grad_block_scaling_dim}, "
            f"fp8_gemm_fprop={self.fp8_gemm_fprop}, "
            f"fp8_gemm_dgrad={self.fp8_gemm_dgrad}, "
            f"fp8_gemm_wgrad={self.fp8_gemm_wgrad}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}"
        )
