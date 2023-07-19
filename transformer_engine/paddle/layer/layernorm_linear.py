# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""LayerNormLinear API"""

import os
from typing import Union, Tuple

import paddle
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

from ..cpp_extensions import (
    gemm,
    layernorm_fwd,
    layernorm_bwd,
)

from .base import get_workspace, TransformerEngineBaseLayer
from ..constants import TE_DType
from ..utils import cast_if_needed

__all__ = ["LayerNormLinear"]


class _LayerNormLinear(paddle.autograd.PyLayer):
    """TE implementation of non-FP8 LayerNormLinear"""

    @staticmethod
    def forward(
        ctx,
        inp: paddle.Tensor,
        ln_weight: paddle.Tensor,
        ln_bias: paddle.Tensor,
        weight: paddle.Tensor,
        bias: Union[paddle.Tensor, None],
        use_bias: bool,
        eps: float,
        activation_dtype: paddle.dtype,
        return_layernorm_output: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Union[Tuple[paddle.Tensor, ...], paddle.Tensor]:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.reshape((-1, in_features))

        ln_out, mu, rsigma = layernorm_fwd(inputmat, ln_weight, ln_bias, eps,
                                           TE_DType[activation_dtype], fwd_ln_sm_margin,
                                           zero_centered_gamma)

        out, _, _ = gemm(
            weight,
            ln_out,
            activation_dtype,
            get_workspace(),
            bias=bias,
            use_bias=use_bias,
        )

        ctx.save_for_backward(
            inputmat,
            ln_weight,
            mu,
            rsigma,
            weight,
            ln_out,
        )

        ctx.activation_dtype = activation_dtype
        ctx.use_bias = use_bias
        ctx.inp_shape = inp.shape
        ctx.return_layernorm_output = return_layernorm_output
        ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
        ctx.zero_centered_gamma = zero_centered_gamma
        ctx.requires_dgrad = not inp.stop_gradient

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        out = out.reshape((-1, *inp.shape[1:-1], out.shape[-1]))

        if return_layernorm_output:
            return out, ln_out.reshape(inp.shape)
        return out

    @staticmethod
    def backward(
            ctx, *grad_outputs: Tuple[paddle.Tensor,
                                      ...]) -> Tuple[Union[paddle.Tensor, None], ...]:
        (
            inputmat,
            ln_weight,
            mu,
            rsigma,
            weight,
            ln_out,
        ) = ctx.saved_tensor()
        grad_output = grad_outputs[0]

        # Dgrad
        dgrad, _, _ = gemm(
            weight,
            grad_output,
            ctx.activation_dtype,
            get_workspace(),
            layout="NN",
            grad=True,
        )

        # Wgrad
        if not weight.stop_gradient:
            wgrad, grad_bias, _ = gemm(
                ln_out,
                grad_output,
                ctx.activation_dtype,
                get_workspace(),
                layout="NT",
                grad=True,
                use_bias=ctx.use_bias,
            )

        # LayerNorm gradient
        d_ln_out = dgrad.reshape(inputmat.shape)
        # Residual gradient
        if ctx.return_layernorm_output:
            d_ln_out = d_ln_out + grad_outputs[1].reshape(d_ln_out.shape)

        dxmat, dgamma, dbeta = layernorm_bwd(d_ln_out, inputmat, mu, rsigma, ln_weight,
                                             ctx.bwd_ln_sm_margin, ctx.zero_centered_gamma)

        if not ctx.use_bias:
            return (
                dxmat.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
                dgamma,
                dbeta,
                wgrad if not weight.stop_gradient else None,
            )

        return (
            dxmat.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma,
            dbeta,
            wgrad if not weight.stop_gradient else None,
            grad_bias,
        )


class LayerNormLinear(TransformerEngineBaseLayer):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        weight_attr: Union[paddle.ParamAttr, None] = None,
        bias_attr: Union[paddle.ParamAttr, None, bool] = None,
        return_layernorm_output: bool = False,
        zero_centered_gamma: bool = False,
        backend: str = 'transformer_engine',
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.return_layernorm_output = return_layernorm_output
        self.zero_centered_gamma = zero_centered_gamma
        self.backend = backend

        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._dtype = self._helper.get_default_dtype()

        # LayerNorm weights
        self.ln_weight = self.create_parameter(
            shape=[in_features],
            attr=paddle.ParamAttr(initializer=Constant(
                value=0.0 if self.zero_centered_gamma else 1.0)),
            dtype=self._dtype,
            is_bias=False,
        )

        self.ln_bias = self.create_parameter(
            shape=[in_features],
            attr=paddle.ParamAttr(initializer=Constant(value=0.0)),
            dtype=self._dtype,
            is_bias=True,
        )

        # Linear weights
        self.weight = self.create_parameter(
            shape=[out_features, in_features]
            if self.backend == 'transformer_engine' else [in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )

        self.has_bias = self._bias_attr is not False
        if self.has_bias:
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=self._bias_attr if self._bias_attr is not None else paddle.ParamAttr(
                    initializer=Constant(value=0.0)),
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            self.bias = None

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

    def _te_forward(
        self,
        inp: paddle.Tensor,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a linear transformation.
        """

        with self.prepare_forward(inp) as inp:
            out = _LayerNormLinear.apply(
                cast_if_needed(inp, self.activation_dtype),
                cast_if_needed(self.ln_weight, self.activation_dtype),
                cast_if_needed(self.ln_bias, self.activation_dtype),
                cast_if_needed(self.weight, self.activation_dtype),
                cast_if_needed(self.bias, self.activation_dtype),
                self.has_bias,
                self.eps,
                self.activation_dtype,
                self.return_layernorm_output,
                self.fwd_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
            )

        if self.return_layernorm_output:
            out, ln_out = out
            return out, ln_out

        return out

    def _pd_forward(
        self,
        inp: paddle.Tensor,
    ) -> paddle.Tensor:
        """Calls Paddle OP"""
        if self.zero_centered_gamma:
            raise NotImplementedError(
                "Paddle backend does not support LayerNorm with zero-centered scale.")

        ln_out = F.layer_norm(x=inp,
                              normalized_shape=inp.shape[1:],
                              weight=self.ln_weight,
                              bias=self.ln_bias,
                              epsilon=self.eps)
        out = F.linear(ln_out, self.weight, self.bias)
        if self.return_layernorm_output:
            return out, ln_out
        return out

    def forward(self, *args, **kwargs):
        """forward"""
        if self.backend == 'transformer_engine':
            return self._te_forward(*args, **kwargs)
        if self.backend == 'paddle':
            return self._pd_forward(*args, **kwargs)
        raise AttributeError(f"Backend {self.backend} is not supported.")
