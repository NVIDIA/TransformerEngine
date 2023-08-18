# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Linear API"""

import os
from typing import Union, Tuple

import paddle
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

from ..constants import TE_DType
from ..cpp_extensions import layernorm_fwd, layernorm_bwd

__all__ = ["LayerNorm"]


class _LayerNorm(paddle.autograd.PyLayer):
    """TE Non-FP8 LayerNorm"""

    @staticmethod
    def forward(
        ctx,
        inp: paddle.Tensor,
        ln_weight: paddle.Tensor,
        ln_bias: paddle.Tensor,
        eps: float,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
    ) -> paddle.Tensor:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "LayerNorm not possible"
        inputmat = inp.reshape((-1, in_features))

        ln_out, mu, rsigma = layernorm_fwd(inputmat, ln_weight, ln_bias, eps, TE_DType[inp.dtype],
                                           fwd_ln_sm_margin, zero_centered_gamma)

        ctx.save_for_backward(inputmat, ln_weight, mu, rsigma)
        ctx.inp_shape = inp.shape
        ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
        ctx.zero_centered_gamma = zero_centered_gamma
        ctx.requires_dx = not inp.stop_gradient
        ctx.requires_dw = not ln_weight.stop_gradient
        ctx.requires_dbias = not ln_bias.stop_gradient
        return ln_out.reshape(inp.shape)

    @staticmethod
    def backward(ctx, grad_output: paddle.Tensor) -> Tuple[Union[paddle.Tensor, None], ...]:
        inputmat, ln_weight, mu, rsigma = ctx.saved_tensor()
        d_ln_out = grad_output.reshape(inputmat.shape)
        dxmat, dgamma, dbeta = layernorm_bwd(d_ln_out, inputmat, mu, rsigma, ln_weight,
                                             ctx.bwd_ln_sm_margin, ctx.zero_centered_gamma)
        return (
            dxmat.reshape(ctx.inp_shape) if ctx.requires_dx else None,
            dgamma if ctx.requires_dw else None,
            dbeta if ctx.requires_dbias else None,
        )


class LayerNorm(paddle.nn.Layer):
    r"""
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        weight_attr: Union[paddle.ParamAttr, None] = None,
        bias_attr: Union[paddle.ParamAttr, None, bool] = None,
        zero_centered_gamma: bool = False,
        backend: str = 'transformer_engine',
    ) -> None:
        super().__init__()
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.backend = backend
        self._dtype = self._helper.get_default_dtype()

        self._weight_attr = weight_attr
        if not self._weight_attr:
            self._weight_attr = paddle.ParamAttr(initializer=Constant(
                value=0.0 if self.zero_centered_gamma else 1.0))

        self._bias_attr = bias_attr
        if self._bias_attr is False:
            self._bias_attr = paddle.ParamAttr(initializer=Constant(value=0.0), trainable=False)

        self.weight = self.create_parameter(
            shape=[hidden_size],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )

        self.bias = self.create_parameter(
            shape=[hidden_size],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

    def _te_forward(self, inp: paddle.Tensor) -> paddle.Tensor:
        """LayerNorm FWD"""
        return _LayerNorm.apply(inp, self.weight, self.bias, self.eps, self.fwd_ln_sm_margin,
                                self.bwd_ln_sm_margin, self.zero_centered_gamma)

    def _pd_forward(
        self,
        inp: paddle.Tensor,
    ) -> paddle.Tensor:
        """Calls Paddle OP"""
        if self.zero_centered_gamma:
            raise NotImplementedError(
                "Paddle backend does not support LayerNorm with zero-centered scale.")

        return F.layer_norm(x=inp,
                            normalized_shape=inp.shape[-1],
                            weight=self.weight,
                            bias=self.bias,
                            epsilon=self.eps)

    def forward(self, *args, **kwargs):
        """forward"""
        if self.backend == 'transformer_engine':
            return self._te_forward(*args, **kwargs)
        if self.backend == 'paddle':
            return self._pd_forward(*args, **kwargs)
        raise AttributeError(f"Backend {self.backend} is not supported.")
