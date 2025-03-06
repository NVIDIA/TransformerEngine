# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNorm API"""
import warnings
from typing import Iterable, Optional, Union

import torch

from transformer_engine.pytorch.ops import LayerNorm as _LayerNormOp

__all__ = ["LayerNorm"]


class LayerNorm(_LayerNormOp):
    r"""Layer Normalization

    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable affine transform
    parameters that match the inner-most dimensions of the input
    tensor.

    Parameters
    ----------
    normalized_shape: int or iterable of int
        Inner dimensions of input tensor
    eps : float, default = 1e-5
        A value added to the denominator of layer normalization for
        numerical stability
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    zero_centered_gamma : bool, default = 'False'
        If `True`, the :math:`\gamma` parameter is initialized to zero
        and the calculation changes to

            .. math::
                y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * (1 + \gamma) + \beta

    sm_margin: int or dict, default = 0
        Number of SMs to exclude when launching CUDA kernels. This
        helps overlap with other kernels, e.g. communication kernels.
        For more fine-grained control, provide a dict with the SM
        margin at each compute stage ("forward", "backward",
        "inference").

    Legacy
    ------
    sequence_parallel: bool
        Set a bool attr named `sequence_parallel` in the parameters.
        This is custom logic for Megatron-LM integration.

    """

    def __init__(
        self,
        normalized_shape: Union[Iterable[int], int, None] = None,
        eps: float = 1e-5,
        sequence_parallel: Optional[bool] = None,  # legacy
        params_dtype: Optional[torch.dtype] = None,  # deprecated
        zero_centered_gamma: bool = False,
        hidden_size: Optional[int] = None,  # deprecated
        **kwargs,
    ) -> None:

        # Handle deprecated options
        if normalized_shape is None:
            if hidden_size is None:
                raise RuntimeError(
                    "Neither `normalized_shape` nor `hidden_size` (deprecated) args are provided"
                )
            warnings.warn(
                "`hidden_size` arg has been renamed to `normalized_shape` "
                "for compatibility with `torch.nn.LayerNorm`.",
                DeprecationWarning,
                stacklevel=2,
            )
            normalized_shape = hidden_size
        elif hidden_size is not None:
            raise RuntimeError(
                "Both `normalized_shape` and `hidden_size` (deprecated) args are provided"
            )
        if params_dtype is not None:
            if "dtype" in kwargs:
                raise RuntimeError(
                    "Both `dtype` and `params_dtype` (deprecated) kwargs are provided"
                )
            kwargs["dtype"] = params_dtype

        # Initialize layer norm operation
        super().__init__(
            normalized_shape,
            eps=eps,
            zero_centered_gamma=zero_centered_gamma,
            **kwargs,
        )

        # Flag for sequence parallelism (custom Megatron-LM integration)
        self.sequence_parallel: Optional[bool] = sequence_parallel
        if sequence_parallel is not None:
            self.weight.sequence_parallel = sequence_parallel
            self.bias.sequence_parallel = sequence_parallel

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        warnings.warn(
            "This method will be deprecated in an upcoming release. "
            "Update your code to use LayerNorm.reset_parameters() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.reset_parameters()

    def reset_parameters(self, defer_init: Optional[bool] = None) -> None:
        """Init LayerNorm parameters"""

        # Check whether to defer init (deprecated)
        if defer_init is not None:
            warnings.warn(
                "defer_init argument to reset_parameters function is deprecated. Set device to"
                ' "meta" instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            if defer_init:
                return

        # Reset parameters
        super().reset_parameters()

        # Set flag for sequence parallelism (custom Megatron-LM integration)
        if getattr(self, "sequence_parallel", None) is not None:
            self.weight.sequence_parallel = self.sequence_parallel
            self.bias.sequence_parallel = self.sequence_parallel

    @property
    def fwd_ln_sm_margin(self) -> int:
        """Shim for backward compatibility"""
        warnings.warn("fwd_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        return self._sm_margins["forward"]

    @fwd_ln_sm_margin.setter
    def fwd_ln_sm_margin(self, val: int) -> None:
        """Shim for backward compatibility"""
        warnings.warn("fwd_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        self._sm_margins["forward"] = val

    @property
    def bwd_ln_sm_margin(self) -> int:
        """Shim for backward compatibility"""
        warnings.warn("bwd_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        return self._sm_margins["backward"]

    @bwd_ln_sm_margin.setter
    def bwd_ln_sm_margin(self, val: int) -> None:
        """Shim for backward compatibility"""
        warnings.warn("bwd_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        self._sm_margins["backward"] = val

    @property
    def inf_ln_sm_margin(self) -> int:
        """Shim for backward compatibility"""
        warnings.warn("inf_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        return self._sm_margins["inference"]

    @inf_ln_sm_margin.setter
    def inf_ln_sm_margin(self, val: int) -> None:
        """Shim for backward compatibility"""
        warnings.warn("inf_ln_sm_margin attr is deprecated", DeprecationWarning, stacklevel=2)
        self._sm_margins["inference"] = val
