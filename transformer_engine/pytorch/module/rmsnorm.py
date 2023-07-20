# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""RMSNorm API"""
import os
from typing import Union, Tuple, Any, Mapping, Optional

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

import transformer_engine_extensions as tex


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
        zero_centered_gamma: bool,
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        in_features = rmsnorm_weight.numel()
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert inp.shape[-1] == in_features, "RMSNorm not possible"
        inputmat = inp.view((-1, in_features))

        rmsnorm_out, rsigma = tex.rmsnorm_fwd(inputmat, rmsnorm_weight,
                                              eps, fwd_ln_sm_margin,
                                              zero_centered_gamma)
        ctx.save_for_backward(inputmat, rmsnorm_weight, rsigma)
        ctx.inp_shape = inp.shape
        ctx.bwd_rmsnorm_sm_margin = bwd_rmsnorm_sm_margin
        ctx.zero_centered_gamma = zero_centered_gamma
        return rmsnorm_out.view_as(inp)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        inputmat, rmsnorm_weight, rsigma = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        d_rmsnorm_out = grad_output.view(inputmat.shape)
        dxmat, dgamma = tex.rmsnorm_bwd(
            d_ln_out, inputmat, rsigma, rmsnorm_weight,
            ctx.bwd_rmsnorm_sm_margin, ctx.zero_centered_gamma
        )
        return dxmat.view(ctx.inp_shape), dgamma, None, None, None, None


class RMSNorm(torch.nn.Module):
    r"""
    Applies Root Mean Square Layer Normalization over a mini-batch of inputs as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`__

    .. math::
        y = \frac{x}{RMS\(x\) + \varepsilon}} * \gamma

    where

    .. math::
        RMS(x) = \sqrt{\frac{1}{n}\sum_{i=0}^nx_i^2}

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
                            y = \frac{x}{RMS\(x\) + \varepsilon}} * (1 + \gamma)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
    ) -> None:
        super().__init__()
        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.weight = Parameter(
            torch.empty(
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        setattr(self.weight, "sequence_parallel", sequence_parallel)
        self.reset_rms_norm_parameters()

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward RMSNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with RMSNorm.
        self.fwd_rmsnorm_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_rmsnorm_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

    def reset_rms_norm_parameters(self) -> None:
        """Init RMSNorm params"""
        if not self.zero_centered_gamma:
            init.ones_(self.weight)
        else:
            init.zeros_(self.weight)


    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """RMSNorm FWD"""
        return _RMSNorm.apply(
            inp,
            self.weight,
            self.eps,
            self.fwd_rmsnorm_sm_margin,
            self.bwd_rmsnorm_sm_margin,
            self.zero_centered_gamma
        )
