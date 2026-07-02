# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""This module provides predefined FP8 recipes."""
from __future__ import annotations
import abc
import os
from enum import Enum
from typing import Any, Literal, Optional, Union, Callable, NamedTuple
from dataclasses import field
from pydantic.dataclasses import dataclass


_BACKWARD_OVERRIDES = (None, "high_precision", "dequantized")
_NVFP4_4OVER6_SCOPES = ("none", "weights", "activations", "all")
_NVFP4_4OVER6_ERR_MODES = ("MAE", "MSE")


class _FormatHelper(NamedTuple):
    """
    Stores max FP8 values for fprop and bprop a `Format`.
    """

    max_fwd: float
    max_bwd: float


class Format(Enum):
    """
    Supported FP8 formats.
    Supported FP4 formats.

    Values
    ------
    E2M1 :
          All FP4 tensors are in e2m1 format
    E4M3 :
          All FP8 tensors are in e4m3 format
    E5M2 :
          All FP8 tensors are in e5m2 format
    HYBRID :
            FP8 tensors in the forward pass are in e4m3 format,
            FP8 tensors in the backward pass are in e5m2 format
    """

    E2M1 = _FormatHelper(max_fwd=6, max_bwd=6)
    E4M3 = _FormatHelper(max_fwd=448, max_bwd=448)
    E5M2 = _FormatHelper(max_fwd=57344, max_bwd=57344)
    HYBRID = _FormatHelper(max_fwd=E4M3.max_fwd, max_bwd=E5M2.max_bwd)


@dataclass(frozen=True)
class MMParams:
    """Matrix multiplication options.

    Parameters
    ----------
    use_split_accumulator : bool, default = True
        Use FP8 fast accumulation on Hopper or Ada. For more details,
        see CUBLASLT_MATMUL_DESC_FAST_ACCUM option for cublasLtMatmul.
    """

    use_split_accumulator: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_cached_repr",
            f"MMParams(use_split_accumulator={self.use_split_accumulator})",
        )

    def __repr__(self) -> str:
        return self._cached_repr


@dataclass(frozen=True)
class QParams:
    """Quantization parameters.
    power_2_scale: use power of 2 scale parameter
    amax_epsilon: optional minimum value of abs max
    random_hadamard_transform: whether to use random hadamard transform
    stochastic_rounding: whether to use stocastic rounding
    """

    power_2_scale: bool = False
    amax_epsilon: float = 0.0
    random_hadamard_transform: bool = False
    stochastic_rounding: bool = False
    fp4_2d_quantization: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_cached_repr",
            f"Qparams(\npower_2_scale={self.power_2_scale},\n"
            f"amax_epsilon={self.amax_epsilon},\n"
            f"random_hadamard_transform={self.random_hadamard_transform},\n"
            f"stochastic_rounding={self.stochastic_rounding},\n"
            f"fp4_2d_quantization={self.fp4_2d_quantization}\n)",
        )

    def __repr__(self) -> str:
        return self._cached_repr


class Recipe:
    """
    Base recipe class.
    """

    # Cached string representation. Lazily populated by ``__repr__`` in
    # subclasses and invalidated by ``__setattr__`` whenever any attribute
    # changes. This makes repeated ``str(recipe)`` calls much cheaper
    _cached_repr: Optional[str] = None

    def __setattr__(self, name: str, value: Any) -> None:
        # Invalidate the cached repr on any attribute mutation.
        if name != "_cached_repr":
            object.__setattr__(self, "_cached_repr", None)
        object.__setattr__(self, name, value)

    @abc.abstractmethod
    def _make_repr(self) -> str:
        """Build the string representation for this recipe.

        Subclasses must override this method. The result is cached by
        ``__repr__`` and reused until any attribute is mutated.
        """

    def __repr__(self) -> str:
        if self._cached_repr is None:
            self._cached_repr = self._make_repr()
        return self._cached_repr

    @classmethod
    def nvfp4(cls):
        """Whether the given recipe is NVFP4 1D block scaling."""
        return issubclass(cls, NVFP4BlockScaling)

    @classmethod
    def nvfp4_per_token(cls):
        """Whether the given recipe is NVFP4 per-token cast (replaces the per-tensor 1A/2A paths)."""
        # Defined as a class method so callers can switch on the recipe
        # *type* (Recipe.nvfp4_per_token() called on the class itself)
        # the same way they do for ``nvfp4()``.
        return False

    @classmethod
    def mxfp8(cls):
        """Whether the given recipe is MXFP8 block scaling."""
        return issubclass(cls, MXFP8BlockScaling)

    @classmethod
    def delayed(cls):
        """Whether the given recipe is delayed scaling."""
        return issubclass(cls, DelayedScaling)

    @classmethod
    def float8_current_scaling(cls):
        """Whether the given recipe is (per-tensor) current scaling."""
        return issubclass(cls, Float8CurrentScaling)

    @classmethod
    def float8_per_tensor_scaling(cls):
        """Whether the given recipe is per-tensor scaling."""
        return issubclass(cls, (DelayedScaling, Float8CurrentScaling))

    @classmethod
    def float8_block_scaling(cls):
        """Whether the given recipe is float8 blockwise scaling."""
        return issubclass(cls, Float8BlockScaling)

    @classmethod
    def custom(cls):
        """Whether the given recipe is custom."""
        return issubclass(cls, CustomRecipe)


@dataclass(repr=False)
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
    reduce_amax: bool, default = True
                By default, if `torch.distributed` is initialized, the `amax` value for FP8
                tensors is reduced across the `amax_reduction_group` (specified in the `autocast`
                call). This keeps the amaxes and scaling factors synced across the given
                distributed group. If set to `False`, this reduction is skipped and every
                GPU maintains local amaxes and scaling factors. To ensure results are
                numerically identical across checkpointing boundaries in this case, all
                ranks must checkpoint in order to store the local tensors.
    fp8_dpa: bool, default = False
             Whether to enable FP8 dot product attention (DPA). When the model is placed in an
             `autocast(enabled=True)` region and `fp8_dpa` is set to `True`, DPA casts the
             inputs from higher precision to FP8, performs attention in FP8, and casts tensors
             back to higher precision as outputs. FP8 DPA currently is only supported in the
             `FusedAttention` backend.
    fp8_mha: bool, default = False
            Whether to enable FP8 multi-head attention (MHA). When `True`, it removes the casting
            operations mentioned above at the DPA boundaries. Currently only standard MHA modules
            i.e. `LayerNormLinear/Linear + DPA + Linear`, are supported for this feature. When
            `fp8_mha = False, fp8_dpa = True`, a typical MHA module works as
            `LayerNormLinear (BF16 output) -> (cast to FP8 ) FP8 DPA (cast to BF16) -> Linear`.
            When `fp8_mha = True, fp8_dpa = True`, it becomes
            `LayerNormLinear (FP8 output) -> FP8 DPA -> Linear`.
    backward_override : {None, 'high_precision', 'dequantized'}, default = None
            Backward precision mode. Delayed scaling only supports None.

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
    backward_override: Optional[str] = os.getenv("NVTE_BACKWARD_OVERRIDE", None)

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."
        assert (
            self.backward_override in _BACKWARD_OVERRIDES
        ), "NVTE_BACKWARD_OVERRIDE must be unset or one of: 'high_precision', 'dequantized'."
        assert (
            self.backward_override is None
        ), "Delayed scaling only supports backward_override=None."

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"amax_history_len={self.amax_history_len}, "
            f"reduce_amax={self.reduce_amax}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}, "
            f"backward_override={self.backward_override}"
        )


@dataclass(repr=False)
class Float8CurrentScaling(Recipe):
    """
    Use the per-tensor current scaling factor strategy.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.HYBRID
                Controls the FP8 data format used during forward and backward
                pass.
    backward_override : {None, 'high_precision', 'dequantized'}, default = None
            Backward precision mode. None does not modify backward behavior,
            `high_precision` keeps original high-precision operands for backward,
            and `dequantized` dequantizes saved operands to the active high-precision
            compute dtype (e.g. BF16/FP16/FP32) for backward.
    """

    use_power_2_scales: bool = os.getenv("NVTE_FP8_CURRENT_SCALING_POWER_2_SCALES", "0") == "1"
    fp8_format: Format = Format.HYBRID
    fp8_quant_fwd_inp = QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)
    fp8_quant_fwd_weight = QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)
    fp8_quant_bwd_grad = QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)
    fp8_gemm_fprop: MMParams = MMParams(use_split_accumulator=False)
    fp8_gemm_dgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_gemm_wgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = os.getenv("NVTE_BACKWARD_OVERRIDE", None)

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."
        assert (
            self.backward_override in _BACKWARD_OVERRIDES
        ), "NVTE_BACKWARD_OVERRIDE must be unset or one of: 'high_precision', 'dequantized'."

    def _make_repr(self) -> str:
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
            f"fp8_mha={self.fp8_mha}, "
            f"backward_override={self.backward_override}"
        )


@dataclass(repr=False)
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
    backward_override : {None, 'high_precision', 'dequantized'}, default = None
            Backward precision mode. None does not modify backward behavior,
            `high_precision` keeps original high-precision operands for backward,
            and `dequantized` dequantizes saved operands to the active high-precision
            compute dtype (e.g. BF16/FP16/FP32) for backward.
    """

    margin: int = 0
    fp8_format: Format = Format.E4M3
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = os.getenv("NVTE_BACKWARD_OVERRIDE", None)

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."
        assert (
            self.backward_override in _BACKWARD_OVERRIDES
        ), "NVTE_BACKWARD_OVERRIDE must be unset or one of: 'high_precision', 'dequantized'."

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"backward_override={self.backward_override}"
        )


@dataclass(repr=False)
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
    backward_override : {None, 'high_precision', 'dequantized'}, default = None
            Backward precision mode. None does not modify backward behavior,
            `high_precision` keeps original high-precision operands for backward,
            and `dequantized` dequantizes saved operands to the active high-precision
            compute dtype (e.g. BF16/FP16/FP32) for backward.
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
    backward_override: Optional[str] = os.getenv("NVTE_BACKWARD_OVERRIDE", None)

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
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."
        assert (
            self.backward_override in _BACKWARD_OVERRIDES
        ), "NVTE_BACKWARD_OVERRIDE must be unset or one of: 'high_precision', 'dequantized'."

    def _make_repr(self) -> str:
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
            f"fp8_mha={self.fp8_mha}, "
            f"backward_override={self.backward_override}"
        )


@dataclass(repr=False)
class NVFP4BlockScaling(Recipe):
    """
    Use the NVFP4 scaling strategy.

    This is a 2-level block scaling strategy. In level 1, each group of
    16 consecutive values is scaled together using their own scaling
    factor. The type of the scaling factor is E4M3 (4 bits of exponent,
    3 bits of mantissa). In level 2, a global per tensor FP32 scaling
    factor is used to scale the entire tensor.

    Since the scaling happens in a particular direction (either rowwise
    or columnwise), in this recipe the quantized tensor and its transpose
    are not numerically equivalent. Due to this, when Transformer Engine
    needs both the tensor and its transpose (e.g. to calculate both
    forward and backward pass), during the quantization both versions are
    computed from the high precision input to avoid double quantization
    errors.

    The default NVFP4 training recipe implements 3 techniques for quantizing
    to a narrow format (4-bit):

    - For weight tensors a variant of the NVFP4 quantization is used,
      where a single scaling factor is shared by a 2D block of 16x16 elements.
    - When quantizing gradients, stochastic rounding is applied to avoid the bias
      introduced by quantization. With this, values are rounded probabilistically
      to one of their two nearest representable numbers, with probabilities
      inversely proportional to their distances.
    - When quantizing inputs and gradients, random Hadamard transforms are applied
      (16x16 Hadamard matrix) to smooth outliers in the tensor distributions
      and make them easier to represent accurately in NVFP4.

    These techniques are described more comprehensively in the NVFP4 paper titled
    'Pretraining Large Language Models with NVFP4' (https://arxiv.org/abs/2509.25149v1).

    Parameters
    ----------
    fp4_format : {Format.E2M1}, default = Format.E2M1
             FP4 data type.
    disable_rht : bool, default = False
             If set to `True`, random Hadamard transforms are not applied to any tensor.
    disable_stochastic_rounding : bool, default = False
             If set to `True`, stochastic rounding is disabled during quantization for all tensors.
    disable_2d_quantization : bool, default = False
             If set to `True`, 1D block scaling with block size 16 is used for all tensors.
    row_scaled_activation : bool, default = False
             If set to `True`, forward activation quantizers emit row-scaled
             NVFP4 tensors. In this mode, rowwise ``amax`` metadata is stored
             as a vector with one FP32 value per tensor row.
    nvfp4_4over6 : {'none', 'weights', 'activations', 'all'}, default = 'none'
             Enable 4over6 adaptive NVFP4 block scaling for selected tensor
             scopes. For each selected FP4 block, quantization compares
             map-to-4 and map-to-6 candidates and stores the candidate with
             lower configured error. Current 4over6 support targets RL and
             post-training scenarios; pre-training paths that combine 4over6
             with RHT are not yet implemented.
    nvfp4_4over6_e4m3_use_256 : {'none', 'weights', 'activations', 'all'}, default = 'all'
             Select 4over6 tensors that use 256 as the global E4M3 scale
             bound. By default, all 4over6 tensors use 256. Use ``'none'``
             to keep the standard NVFP4 448 bound for 4over6 tensors.
    nvfp4_4over6_err_mode : {'MAE', 'MSE'}, default = 'MAE'
             Error metric used by NVFP4 4over6 candidate selection.
    backward_override : {None, 'high_precision', 'dequantized'}, default = None
            Backward precision mode. None does not modify backward behavior,
            `high_precision` keeps original high-precision operands for backward,
            and `dequantized` dequantizes saved operands to the active high-precision
            compute dtype (e.g. BF16/FP16/FP32) for backward.
    """

    # Configuration envvars
    disable_rht: bool = os.getenv("NVTE_NVFP4_DISABLE_RHT", "0") == "1"
    disable_stochastic_rounding: bool = (
        os.getenv("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "0") == "1"
    )
    disable_2d_quantization: bool = os.getenv("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "0") == "1"
    # Per-token only: opt INTO RHT. Per-token otherwise force-disables RHT (its
    # per-row outer amax already mitigates the long-tail outliers RHT targets),
    # but the per-token cast kernel DOES support RHT, so this flag lets callers
    # turn it back on. Env-driven (NVTE_NVFP4_PER_TOKEN_RHT=1) so frameworks that
    # only construct a default NVFP4BlockScaling (e.g. Megatron-Core) can flip it
    # without code changes. No effect on the non-per-token path.
    per_token_rht: bool = os.getenv("NVTE_NVFP4_PER_TOKEN_RHT", "0") == "1"
    # Per-token only: opt INTO stochastic rounding. Per-token otherwise
    # force-disables SR, but the per-token K2 encode kernel DOES implement SR
    # (Philox-dithered FP4 cast), so this flag re-enables it on the bwd-grad
    # quantizer (the default non-per-token recipe also applies SR to
    # fp4_quant_bwd_grad only).
    # Env-driven (NVTE_NVFP4_PER_TOKEN_SR=1) for the same reason as per_token_rht
    # above. No effect on the non-per-token path.
    per_token_sr: bool = os.getenv("NVTE_NVFP4_PER_TOKEN_SR", "0") == "1"
    # Per-token only: quantize the forward WEIGHT with the per-tensor 2D cast
    # (16x16 inner + scalar outer amax) instead of the per-token 1D cast, emitted in
    # per-token layout so the per-token CUTLASS GEMM consumes it unchanged. 2D
    # weight quant is transposition-invariant, so fwd (rowwise) and dgrad
    # (columnwise) see the same weight, removing the 1D path's gradient bias.
    # Env-driven (NVTE_NVFP4_PER_TOKEN_WEIGHT_2D=1); default off is byte-equal.
    per_token_weight_2d: bool = os.getenv("NVTE_NVFP4_PER_TOKEN_WEIGHT_2D", "0") == "1"
    row_scaled_activation: bool = os.getenv("NVTE_NVFP4_ROW_SCALED_ACTIVATION", "0") == "1"
    nvfp4_4over6: str = os.getenv("NVTE_NVFP4_4OVER6", "none")
    nvfp4_4over6_e4m3_use_256: str = os.getenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")
    nvfp4_4over6_err_mode: str = os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").upper()

    fp4_format: Format = Format.E2M1
    fp8_format: Format = Format.E4M3

    # Not applying quantization to attention for now
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = os.getenv("NVTE_BACKWARD_OVERRIDE", None)

    @classmethod
    def nvfp4_per_token(cls) -> bool:
        """Whether this NVFP4 recipe runs the per-token cast + fused-EVT GEMM.

        Two activation paths, both land here:

        * Explicit recipe class ``NVFP4PerTokenBlockScaling`` overrides
          this to return ``True`` unconditionally.
        * Env-var ``NVTE_NVFP4_PER_TOKEN=1`` flips the *base*
          ``NVFP4BlockScaling`` recipe into per-token mode WITHOUT any
          recipe-class change. This lets frameworks that already build a
          plain ``NVFP4BlockScaling`` (e.g. Megatron-core with
          ``--fp8-recipe nvfp4``) opt into per-token purely from the
          launch environment, no framework-side code edit.

        NOTE: this only changes the *cast + GEMM dispatch* (the "routing").
        It does NOT make per-token support features it lacks
        (fuse_wgrad_accumulation / sequence-parallel single-direction
        cast / comm-overlap / output-quant). Those still raise at runtime
        regardless of how per-token was activated; the launcher must keep
        them disabled until the per-token kernels grow that support.
        """
        return os.getenv("NVTE_NVFP4_PER_TOKEN", "0") == "1"

    def _force_per_token_settings(self) -> None:
        """Force-disable knobs that are mutually exclusive with per-token.

        Shared by the explicit ``NVFP4PerTokenBlockScaling`` subclass and
        the ``NVTE_NVFP4_PER_TOKEN=1`` env-var activation so the two
        activation paths can never numerically diverge. Must run BEFORE
        QParams construction so the forced values propagate into the
        ``fp4_quant_*`` QParams below.
        """
        # RHT and SR are opt-in for per-token (both kernels support them); 2D /
        # row_scaled / 4over6 stay hard-disabled (their per-token kernels are
        # not implemented yet -- see NVFP4Quantizer ctor mutex).
        object.__setattr__(self, "disable_rht", not self.per_token_rht)
        object.__setattr__(self, "disable_stochastic_rounding", not self.per_token_sr)
        object.__setattr__(self, "disable_2d_quantization", True)
        object.__setattr__(self, "row_scaled_activation", False)
        object.__setattr__(self, "nvfp4_4over6", "none")

    def __post_init__(self) -> None:
        # Resolve per-token activation (recipe class OR env var) and force
        # the mutex-incompatible knobs off BEFORE the asserts + QParams
        # construction. ``self.nvfp4_per_token()`` dispatches
        # polymorphically: True for the subclass, env-driven for the base.
        if self.nvfp4_per_token():
            self._force_per_token_settings()

        assert self.fp4_format == Format.E2M1, "Only E2M1 is supported for NVFP4 scaling"
        assert self.fp8_format == Format.E4M3, "Only E4M3 is supported for NVFP4 scaling"
        assert (
            self.backward_override in _BACKWARD_OVERRIDES
        ), "NVTE_BACKWARD_OVERRIDE must be unset or one of: 'high_precision', 'dequantized'."
        assert (
            self.nvfp4_4over6 in _NVFP4_4OVER6_SCOPES
        ), "NVTE_NVFP4_4OVER6 must be one of: 'none', 'weights', 'activations', 'all'."
        assert (
            self.nvfp4_4over6_e4m3_use_256 in _NVFP4_4OVER6_SCOPES
        ), "NVTE_NVFP4_4OVER6_E4M3_USE_256 must be one of: 'none', 'weights', 'activations', 'all'."
        assert (
            self.nvfp4_4over6_err_mode in _NVFP4_4OVER6_ERR_MODES
        ), "NVTE_NVFP4_4OVER6_ERR_MODE must be one of: 'MAE', 'MSE'."

        # Quantization params
        # Note: RHT is currently only applied to column-wise usage so that
        # it can be used for wgrad GEMM.
        self.fp4_quant_fwd_inp = QParams(
            random_hadamard_transform=not self.disable_rht,
            stochastic_rounding=False,
            fp4_2d_quantization=False,
        )
        self.fp4_quant_fwd_weight = QParams(
            random_hadamard_transform=False,
            stochastic_rounding=False,
            fp4_2d_quantization=not self.disable_2d_quantization,
        )
        self.fp4_quant_bwd_grad = QParams(
            random_hadamard_transform=not self.disable_rht,
            stochastic_rounding=not self.disable_stochastic_rounding,
            fp4_2d_quantization=False,
        )

    def _make_repr(self) -> str:
        # Surface per-token state even when activated via env var on the
        # base recipe (otherwise a per-token run looks identical to a per-tensor
        # run in logs, which is a debugging trap).
        per_token_repr = "per_token=True, " if self.nvfp4_per_token() else ""
        if self.nvfp4_per_token() and self.per_token_weight_2d:
            per_token_repr += "per_token_weight_2d=True, "
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"{per_token_repr}"
            f"fp4_format={str(self.fp4_format).split('.')[1]}, "
            f"fp8_format={str(self.fp8_format).split('.')[1]}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}, "
            f"backward_override={self.backward_override}, "
            f"row_scaled_activation={self.row_scaled_activation}, "
            f"nvfp4_4over6={self.nvfp4_4over6}, "
            f"nvfp4_4over6_e4m3_use_256={self.nvfp4_4over6_e4m3_use_256}, "
            f"nvfp4_4over6_err_mode={self.nvfp4_4over6_err_mode}, "
            f"fp4_quant_fwd_inp={self.fp4_quant_fwd_inp}, "
            f"fp4_quant_fwd_weight={self.fp4_quant_fwd_weight}, "
            f"fp4_quant_bwd_grad={self.fp4_quant_bwd_grad}, "
        )


@dataclass(repr=False)
class NVFP4PerTokenBlockScaling(NVFP4BlockScaling):
    """
    NVFP4 per-token cast variant that replaces the default per-tensor 1A
    (single-tensor) and 2A (split_quantize) routes with the per-token
    fast-path (K1 vector amax + K2 encode+swizzle, plus a fused EVT GEMM
    that consumes per-row + per-col outer-amax vectors directly).

    The cast emits NVFP4 tensors whose ``_amax_rowwise`` is a per-row
    vector of length ``M`` and ``_amax_columnwise`` is a per-col vector
    of length ``K``. cuBLASLt cannot consume these vector amaxes, so
    downstream GEMM must dispatch to ``nvfp4_cutlass_per_token_gemm``;
    ``general_gemm`` handles this routing automatically based on the
    ``_per_token`` tensor flag.

    Mutex constraints (enforced by both this recipe and the C++
    ``NVFP4Quantizer`` constructor):
      - ``disable_rht`` defaults True (per-token's per-row outer-amax
        granularity already mitigates the long-tail outliers RHT was
        added to flatten in the per-tensor NVFP4 path), but RHT is OPT-IN: set
        ``per_token_rht=True`` to re-enable it. The per-token cast kernel
        (``nvte_nvfp4_per_token_quantize``) supports RHT, and the C++
        ``NVFP4Quantizer`` ctor does NOT mutex it (only requires
        ``with_post_rht_amax=True``, which the quantizer factory sets
        alongside ``with_rht``).
      - ``disable_stochastic_rounding`` defaults True, but SR is OPT-IN:
        set ``per_token_sr=True`` to re-enable it. The per-token K2 encode
        kernel implements SR (Philox-dithered FP4 cast); when enabled it is
        applied to the bwd-grad quantizer only (matching the default per-tensor
        recipe, which only sets SR on ``fp4_quant_bwd_grad``). The C++
        ``NVFP4Quantizer`` ctor
        does NOT mutex it.
      - ``disable_2d_quantization`` is forced True (2D quant is mutex
        with per-token amax allocation).
      - ``row_scaled_activation`` is forced False (the row-scaled
        codepath has its own amax allocation that conflicts with
        per-token vector amax).
      - ``nvfp4_4over6`` is forced ``"none"`` (4over6 candidate
        selection is mutex with per-token amax flow).

    Parameters
    ----------
    per_token_rht : bool, default = False (env ``NVTE_NVFP4_PER_TOKEN_RHT``)
        Opt into the random Hadamard transform on the forward activation
        and backward gradient quantizers. Per-token disables RHT by
        default (its per-row outer amax already mitigates the long-tail
        outliers RHT targets).
    per_token_sr : bool, default = False (env ``NVTE_NVFP4_PER_TOKEN_SR``)
        Opt into stochastic rounding on the backward gradient quantizer
        (the K2 encode kernel implements a Philox-dithered FP4 cast).
        Per-token disables SR by default.
    per_token_weight_2d : bool, default = False (env ``NVTE_NVFP4_PER_TOKEN_WEIGHT_2D``)
        Quantize the forward weight with the per-tensor 2D cast (16x16
        inner tile + scalar outer amax) emitted in per-token layout.
        2D weight quantization is transposition-invariant, removing the
        1D path's weight-gradient bias. Activations/gradients stay 1D.
    backward_override : {None, 'high_precision', 'dequantized'}, default = None
        Inherited from ``NVFP4BlockScaling``.

    Notes
    -----
    The per-token forward path currently requires the unfused norm+amax
    path. When training a model whose first GEMM consumes a fused
    norm+quantize output (e.g. ``LayerNormLinear``), also set
    ``NVTE_NORM_FWD_USE_CUDNN=1`` so the norm forward uses the unfused
    implementation; the fused norm+amax path rejects per-token
    quantizers at the C++ quantizer.

    Per-token cast covers both forward and backward:

      * fwd:  ``input`` / ``output`` / ``weight`` -> per-token cast +
        fused-EVT CUTLASS GEMM (TN layout).
      * bwd:  ``grad_output`` / ``grad_input`` -> per-token cast +
        fused-EVT CUTLASS GEMM (NN layout for dgrad, NT layout for
        wgrad). The kernel itself is layout-agnostic; the dispatch
        in ``cpp_extensions/gemm.py:_nvfp4_per_token_gemm`` selects
        rowwise vs columnwise quantized data + amax for each operand
        so the contraction dim is contiguous in both kernel inputs.

    Random Hadamard transform (``per_token_rht``) and stochastic
    rounding (``per_token_sr``) are supported on the per-token path but
    OFF by default; opt in via the constructor kwargs or env vars above.

    Currently unsupported (with future work tickets):

      * ``fuse_wgrad_accumulation=True`` -- the per-token kernel does
        not yet support ``D += A @ B``; users must disable wgrad
        accumulation when training with the per-token recipe.
      * Output quantization in fwd/bwd (``output_quantizer``,
        ``grad_input_quantizer``) -- fused EVT only emits bf16; a
        post-cast wrapper is a future enhancement.
      * Communication overlap and bulk overlap.
    """

    @classmethod
    def nvfp4_per_token(cls) -> bool:  # noqa: D401 (single-line docstring is fine)
        """Whether the given recipe class is NVFP4 per-token.

        Always ``True`` for this explicit subclass (env-var independent).
        The mutex-flag forcing + repr tagging are handled by the base
        ``NVFP4BlockScaling.__post_init__`` / ``_make_repr``, which both
        dispatch on ``self.nvfp4_per_token()``.
        """
        return True


@dataclass(repr=False)
class CustomRecipe(Recipe):
    """
    Custom recipe that allows users to provide quantizer factories.

    .. warning::
        **EXPERIMENTAL**: Custom recipe is experimental, still under active development,
        and the API is subject to change without notice. Use at your own risk.

    Parameters
    ----------
    qfactory : Callable
        Factory callable that returns a quantizer instance *or* a
        ``QuantizerRequest`` subclass for a given ``QuantizerRole``.
        The callable is invoked as::

            qfactory(
                role: QuantizerRole,
            ) -> Union[Quantizer, QuantizerRequest]

        ``QuantizerRole`` is a frozen dataclass with the following fields:

        - ``module_type`` (str): module type (empty string when not set), e.g.
          ``"linear"``, ``"grouped_linear"``, ``"dpa"``.
        - ``tensor_type`` (str): what tensor is being quantized (empty
          string when not set), e.g. ``"input"``, ``"weight"``, ``"grad_output"``.
        - ``name`` (str): caller-provided module instance name (empty
          string when not set), e.g. ``"qkv"``, ``"proj"``, ``"fc1"``, ``"fc2"``.

        For stateful quantizers (delayed scaling), return a
        ``DelayedScalingRequest`` dataclass instead of a quantizer.
        TE will allocate shared scale/amax_history buffers and create
        ``Float8Quantizer`` instances integrated with the existing
        delayed-scaling reduction infrastructure.

        See ``transformer_engine.pytorch.quantization.QuantizerRole``
        and ``transformer_engine.pytorch.quantization.DelayedScalingRequest``
        for full documentation.

    backward_override : {None, 'high_precision', 'dequantized'}, default = None
        Backward precision mode. None does not modify backward behavior,
        `high_precision` keeps original high-precision operands for backward,
        and `dequantized` dequantizes saved operands to the active high-precision
        compute dtype (e.g. BF16/FP16/FP32) for backward.
    """

    qfactory: Callable[..., Any]

    # fp8_format does not affect quantization (quantization factory controls that),
    # but TE internals (e.g. get_fp8_te_dtype, backend selection) read it
    # from the recipe.  HYBRID (E4M3 fwd, E5M2 bwd) is a safe default.
    fp8_format: Format = Format.HYBRID

    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = os.getenv("NVTE_BACKWARD_OVERRIDE", None)

    def __post_init__(self) -> None:
        assert (
            self.backward_override in _BACKWARD_OVERRIDES
        ), "NVTE_BACKWARD_OVERRIDE must be unset or one of: 'high_precision', 'dequantized'."

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"qfactory={self.qfactory}, "
            f"backward_override={self.backward_override}"
        )
