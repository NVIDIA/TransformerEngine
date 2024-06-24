# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""RMSNorm API"""
import os
import warnings
from typing import Union, Tuple, Optional

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

from .base import TransformerEngineBaseModule
from .. import cpp_extensions as tex
from ..jit import no_torch_dynamo
from ..utils import cast_if_needed


__all__ = ["RMSNorm"]


class _RMSNorm(torch.autograd.Function):
    """functional RMSNorm"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        rmsnorm_weight: torch.Tensor,
        eps: float,
        fwd_rmsnorm_sm_margin: int,
        bwd_rmsnorm_sm_margin: int,
        inf_rmsnorm_sm_margin: int,
        zero_centered_gamma: bool,
        is_grad_enabled: bool,
        activation_dtype: torch.dtype,
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        in_features = rmsnorm_weight.numel()
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert inp.shape[-1] == in_features, "RMSNorm not possible"
        inputmat = inp.view((-1, in_features))

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        rmsnorm_weight = cast_if_needed(rmsnorm_weight, activation_dtype)

        if is_grad_enabled:
            rmsnorm_out, rsigma = tex.rmsnorm_fwd(
                inputmat, rmsnorm_weight, eps, fwd_rmsnorm_sm_margin, zero_centered_gamma
            )
            ctx.save_for_backward(inputmat, rmsnorm_weight, rsigma)
            ctx.inp_shape = inp.shape
            ctx.bwd_rmsnorm_sm_margin = bwd_rmsnorm_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
        else:
            rmsnorm_out = tex.rmsnorm_fwd_inf(
                inputmat, rmsnorm_weight, eps, inf_rmsnorm_sm_margin, zero_centered_gamma
            )
        return rmsnorm_out.view_as(inp)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        inputmat, rmsnorm_weight, rsigma = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        d_rmsnorm_out = grad_output.view(inputmat.shape)
        dxmat, dgamma = tex.rmsnorm_bwd(
            d_rmsnorm_out,
            inputmat,
            rsigma,
            rmsnorm_weight,
            ctx.bwd_rmsnorm_sm_margin,
            ctx.zero_centered_gamma,
        )
        return (
            dxmat.view(ctx.inp_shape),
            dgamma,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class RMSNorm(torch.nn.Module):
    r"""
    Applies Root Mean Square Layer Normalization over a mini-batch of inputs as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`__

    .. math::
        y = \frac{x}{RMS_\varepsilon(x)} * \gamma

    where

    .. math::
        RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^nx_i^2 + \varepsilon}

    :math:`\gamma` is a learnable affine transform parameter of size :attr:`hidden_size`

    Parameters
    ----------
    hidden_size : int
                size of each input sample.
    eps : float, default = 1e-5
        a value added to the denominator of layer normalization for numerical stability.
    sequence_parallel : bool, default = `False`
                        if set to `True`, uses sequence parallelism.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                    it controls the type used to allocate the initial parameters. Useful when
                    the model is trained with lower precision and the original FP32 parameters
                    would not fit in GPU memory.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in RMSNorm is initialized to 0 and
                         the RMSNorm formula changes to

                         .. math::
                            y = \frac{x}{RMS_\varepsilon(x)} * (1 + \gamma)
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()
        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.weight = Parameter(
            torch.empty(
                hidden_size,
                device=device,
                dtype=params_dtype,
            )
        )
        self.sequence_parallel = sequence_parallel

        self.reset_parameters(defer_init=(device == "meta"))

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward RMSNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with RMSNorm.
        self.fwd_rmsnorm_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_rmsnorm_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))
        self.inf_rmsnorm_sm_margin = int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", "0"))

    def reset_rms_norm_parameters(self) -> None:
        """Init RMSNorm params"""
        warnings.warn(
            "This method is deprecated and will be removed in an upcoming release. "
            "Update your code to use RMSNorm.reset_parameters() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.zero_centered_gamma:
            init.ones_(self.weight)
        else:
            init.zeros_(self.weight)

    def reset_parameters(self, defer_init=False) -> None:
        """Reset RMSNorm parameters"""
        if defer_init:
            return

        if self.weight.device == torch.device("meta"):
            self.weight = torch.nn.Parameter(torch.empty_like(self.weight, device="cuda"))
        init.constant_(self.weight, float(not self.zero_centered_gamma))
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)

    @no_torch_dynamo()
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """RMSNorm FWD"""

        # Set the activation type for AMP.
        TransformerEngineBaseModule.set_activation_dtype(self, inp)

        if torch.is_grad_enabled():
            fwd_fn = _RMSNorm.apply
            args = []
        else:
            fwd_fn = _RMSNorm.forward
            args = [None]

        args += (
            inp,
            self.weight,
            self.eps,
            self.fwd_rmsnorm_sm_margin,
            self.bwd_rmsnorm_sm_margin,
            self.inf_rmsnorm_sm_margin,
            self.zero_centered_gamma,
            torch.is_grad_enabled(),
            self.activation_dtype,
        )

        return fwd_fn(*args)
