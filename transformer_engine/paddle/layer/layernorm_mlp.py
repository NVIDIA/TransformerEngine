# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""LayerNormMLP API"""

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
from ..utils import cast_if_needed, get_paddle_act_func

__all__ = ["LayerNormMLP"]


class _LayerNormMLP(paddle.autograd.PyLayer):
    """TE implementation of non-FP8 LayerNormMLP"""

    @staticmethod
    def forward(
        ctx,
        inp: paddle.Tensor,
        ln_weight: paddle.Tensor,
        ln_bias: paddle.Tensor,
        fc1_weight: paddle.Tensor,
        fc1_bias: Union[paddle.Tensor, None],
        use_fc1_bias: bool,
        fc2_weight: paddle.Tensor,
        fc2_bias: Union[paddle.Tensor, None],
        use_fc2_bias: bool,
        eps: float,
        activation_dtype: paddle.dtype,
        return_layernorm_output: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        activation: str,
    ) -> Union[Tuple[paddle.Tensor, ...], paddle.Tensor]:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.reshape((-1, in_features))

        # only support gelu for now
        assert activation == 'gelu'

        # LN FWD
        ln_out, mu, rsigma = layernorm_fwd(inputmat, ln_weight, ln_bias, eps,
                                           TE_DType[activation_dtype], fwd_ln_sm_margin,
                                           zero_centered_gamma)

        # FC1 + GeLU
        gelu_out, _, fc1_out = gemm(
            fc1_weight,
            ln_out,
            activation_dtype,
            get_workspace(),
            bias=fc1_bias,
            use_bias=use_fc1_bias,
            gelu=(activation == 'gelu'),
        )

        # FC2
        fc2_out, _, _ = gemm(
            fc2_weight,
            gelu_out,
            activation_dtype,
            get_workspace(),
            bias=fc2_bias,
            use_bias=use_fc2_bias,
        )

        ctx.save_for_backward(
            inputmat,
            ln_weight,
            mu,
            rsigma,
            ln_out,
            fc1_out,
            gelu_out,
            fc1_weight,
            fc2_weight,
        )
        ctx.activation_dtype = activation_dtype
        ctx.activation = activation
        ctx.use_fc1_bias = use_fc1_bias
        ctx.use_fc2_bias = use_fc2_bias
        ctx.inp_shape = inp.shape
        ctx.return_layernorm_output = return_layernorm_output
        ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
        ctx.zero_centered_gamma = zero_centered_gamma
        ctx.requires_dgrad = not inp.stop_gradient

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        fc2_out = fc2_out.reshape((-1, *inp.shape[1:-1], fc2_out.shape[-1]))

        if return_layernorm_output:
            return fc2_out, ln_out.reshape(inp.shape)
        return fc2_out

    @staticmethod
    def backward(
            ctx, *grad_outputs: Tuple[paddle.Tensor,
                                      ...]) -> Tuple[Union[paddle.Tensor, None], ...]:
        (
            inputmat,
            ln_weight,
            mu,
            rsigma,
            ln_out,
            fc1_out,
            gelu_out,
            fc1_weight,
            fc2_weight,
        ) = ctx.saved_tensor()
        # grad_fc2_out
        grad_output = grad_outputs[0]

        # FC2 Dgrad + dGELU
        dgelu, _, _ = gemm(
            fc2_weight,
            grad_output,
            ctx.activation_dtype,
            get_workspace(),
            layout="NN",
            gelu=(ctx.activation == 'gelu'),
            gelu_input=fc1_out,
            grad=True,
        )

        # FC2 Wgrad
        if not fc2_weight.stop_gradient:
            fc2_wgrad, fc2_bias_grad, _ = gemm(
                gelu_out,
                grad_output,
                ctx.activation_dtype,
                get_workspace(),
                layout="NT",
                grad=True,
                use_bias=ctx.use_fc2_bias,
            )

        # For non-fp8 execution, FC1 bias gradient is fused with FC1 wgrad GEMM
        # and will not be calculated in case wgrad is not required.
        if fc1_weight.stop_gradient:
            fc1_bias_grad = dgelu.sum(axis=0)

        # FC1 DGRAD
        fc1_dgrad, _, _ = gemm(
            fc1_weight,
            dgelu,
            ctx.activation_dtype,
            get_workspace(),
            layout="NN",
            grad=True,
        )

        # FC1 Wgrad
        if not fc1_weight.stop_gradient:
            fc1_wgrad, fc1_bias_grad, _ = gemm(
                ln_out,
                dgelu,
                ctx.activation_dtype,
                get_workspace(),
                layout="NT",
                grad=True,
                use_bias=ctx.use_fc1_bias,
            )

        # LayerNorm gradient
        d_ln_out = fc1_dgrad.reshape(inputmat.shape)
        # Residual gradient
        if ctx.return_layernorm_output:
            d_ln_out = d_ln_out + grad_outputs[1].reshape(d_ln_out.shape)

        dxmat, dgamma, dbeta = layernorm_bwd(d_ln_out, inputmat, mu, rsigma, ln_weight,
                                             ctx.bwd_ln_sm_margin, ctx.zero_centered_gamma)

        fc1_bias_grad_out = (fc1_bias_grad,) if ctx.use_fc1_bias else ()
        fc2_bias_grad_out = (fc2_bias_grad,) if ctx.use_fc2_bias else ()

        return (
            dxmat.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma,
            dbeta,
            fc1_wgrad if not fc1_weight.stop_gradient else None,
            *fc1_bias_grad_out,
            fc2_wgrad if not fc2_weight.stop_gradient else None,
            *fc2_bias_grad_out,
        )


class LayerNormMLP(TransformerEngineBaseLayer):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        weight_attr: Union[paddle.ParamAttr, None] = None,
        bias_attr: Union[paddle.ParamAttr, None, bool] = None,
        activation: str = "gelu",
        return_layernorm_output: bool = False,
        zero_centered_gamma: bool = False,
        backend: str = 'transformer_engine',
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.eps = eps
        self.activation = activation
        self.return_layernorm_output = return_layernorm_output
        self.zero_centered_gamma = zero_centered_gamma
        self.backend = backend

        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._dtype = self._helper.get_default_dtype()

        # LayerNorm weights
        self.ln_weight = self.create_parameter(
            shape=[self.hidden_size],
            attr=paddle.ParamAttr(initializer=Constant(
                value=0.0 if self.zero_centered_gamma else 1.0)),
            dtype=self._dtype,
            is_bias=False,
        )

        self.ln_bias = self.create_parameter(
            shape=[self.hidden_size],
            attr=paddle.ParamAttr(initializer=Constant(value=0.0)),
            dtype=self._dtype,
            is_bias=True,
        )

        # FC1 weights
        self.fc1_weight = self.create_parameter(
            shape=[self.ffn_hidden_size, self.hidden_size]
            if self.backend == 'transformer_engine' else [self.hidden_size, self.ffn_hidden_size],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )

        self.has_bias = self._bias_attr is not False
        if self._bias_attr is None:
            self._bias_attr = paddle.ParamAttr(initializer=Constant(value=0.0))

        if self.has_bias:
            self.fc1_bias = self.create_parameter(
                shape=[self.ffn_hidden_size],
                attr=self._bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            self.fc1_bias = None

        # FC2 weights
        self.fc2_weight = self.create_parameter(
            shape=[self.hidden_size, self.ffn_hidden_size]
            if self.backend == 'transformer_engine' else [self.ffn_hidden_size, self.hidden_size],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )

        if self.has_bias:
            self.fc2_bias = self.create_parameter(
                shape=[self.hidden_size],
                attr=self._bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            self.fc2_bias = None

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
            out = _LayerNormMLP.apply(
                cast_if_needed(inp, self.activation_dtype),
                cast_if_needed(self.ln_weight, self.activation_dtype),
                cast_if_needed(self.ln_bias, self.activation_dtype),
                cast_if_needed(self.fc1_weight, self.activation_dtype),
                cast_if_needed(self.fc1_bias, self.activation_dtype),
                self.has_bias,
                cast_if_needed(self.fc2_weight, self.activation_dtype),
                cast_if_needed(self.fc2_bias, self.activation_dtype),
                self.has_bias,
                self.eps,
                self.activation_dtype,
                self.return_layernorm_output,
                self.fwd_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                self.activation,
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
        fc1_out = F.linear(ln_out, self.fc1_weight, self.fc1_bias)
        act_func = get_paddle_act_func(self.activation)
        act_out = act_func(fc1_out)
        out = F.linear(act_out, self.fc2_weight, self.fc2_bias)

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
