"""This module provides predefined FP8 recipes."""
from enum import Enum
from pydantic.dataclasses import dataclass
from typing import Literal, Optional, Union, Callable, NamedTuple


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


class _OverrideLinearPrecision(NamedTuple):
    """
    Whether or not the execute the `fprop`, `dgrad`, and `wgrad`
    GEMMs in higher precision when using FP8.
    """

    fprop: bool = False
    dgrad: bool = False
    wgrad: bool = False


@dataclass()
class DelayedScaling:
    """
    Use the delayed scaling factor strategy.
    Use scale factor from previous iteration,
    recompute once every `interval`, and record
    amax history of `amax_history_len` steps.

    Parameters
    ----------
    margin : int, default = 0
      Margin for the scaling factor computation.
    interval : int, default = 1
      Controls how often the scaling factor is recomputed.
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.HYBRID
      Controls the FP8 data format used during forward and backward
      pass.
    amax_history_len : int, default = 1
      The length of the amax history window used for scaling factor computation.
    amax_compute_algo : {'max', 'most_recent', Callable}, default = 'most_recent'
      Algorithm used for choosing the `amax` value for the scaling factor
      computation. There are 2 predefined choices: `max` chooses the largest
      `amax` in the history window, while `most_recent` always chooses the most
      recently seen value. Alternatively, one may pass a function of the
      signature:

      .. code-block:: python

        def amax_compute(amax_history: Tensor) -> Tensor

      where `Tensor` is a framework tensor type.
    scaling_factor_compute_algo : Callable, default = None
      Algorithm used for computing the new scaling factor based on the value of
      `amax`. It should be a function of the signature:

      .. code-block:: python

        def scaling_factor_compute(amax: Tensor,
                                   old_scaling_factor: Tensor,
                                   fp8_max: Tensor,
                                   recipe: DelayedScaling) -> Tensor

      where `Tensor` is a framework tensor type.
    override_linear_precision: Tuple(bool, bool, bool), default=(False, False,
                                                                 False)
      Whether or not the execute the `fprop`, `dgrad`, and `wgrad`
      GEMMs (respectively) in higher precision when using FP8.

    Notes
    -----
    * By default (when `scaling_factor_compute_algo` is left as `None`) the
      scaling factor is computed from the final `amax` value using the formula:

      .. code-block:: python

          FP8_MAX = maximum_representable_value(fp8_format)
          exp = get_exponent(FP8_MAX / amax) - margin
          new_scaling_factor = 2.0 ^ exp

    * The scaling factor should always be a power of 2 to not introduce numerical
      error during the conversion from FP8 to higher precision format.
    """

    margin: int = 0
    interval: int = 1
    fp8_format: Format = Format.HYBRID
    amax_history_len: int = 1
    amax_compute_algo: Union[Literal["max", "most_recent"], Callable] = "most_recent"
    override_linear_precision: _OverrideLinearPrecision = _OverrideLinearPrecision()
    scaling_factor_compute_algo: Optional[Callable] = None

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."
        assert self.override_linear_precision in (
            (False, False, False),
            (False, False, True),
        ), "Only wgrad GEMM override is currently supported."
