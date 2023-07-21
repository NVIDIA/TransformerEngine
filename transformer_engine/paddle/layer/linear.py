# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Linear API"""

from typing import Union, Tuple

import paddle
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

from .base import TransformerEngineBaseLayer, get_workspace
from ..cpp_extensions import gemm
from ..utils import cast_if_needed

__all__ = ["Linear"]


class _Linear(paddle.autograd.PyLayer):
    """TE implementation of non-FP8 Linear"""

    @staticmethod
    def forward(
        ctx,
        weight: paddle.Tensor,
        inp: paddle.Tensor,
        bias: paddle.Tensor,
        use_bias: bool,
        activation_dtype: paddle.dtype,
    ) -> paddle.Tensor:
        # Make sure input dimensions are compatible
        in_features = weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.reshape((-1, in_features))

        out, _, _ = gemm(
            weight,
            inputmat,
            activation_dtype,
            get_workspace(),
            bias=bias,
            use_bias=use_bias,
        )

        ctx.save_for_backward(
            inputmat,
            weight,
        )
        ctx.activation_dtype = activation_dtype
        ctx.use_bias = use_bias
        ctx.inp_shape = inp.shape
        ctx.requires_dgrad = not inp.stop_gradient

        return out.reshape((-1, *inp.shape[1:-1], out.shape[-1]))

    @staticmethod
    def backward(ctx, grad_output: paddle.Tensor) -> Tuple[Union[paddle.Tensor, None], ...]:
        inputmat, weight = ctx.saved_tensor()
        if ctx.requires_dgrad:
            dgrad, _, _ = gemm(
                weight,
                grad_output,
                ctx.activation_dtype,
                get_workspace(),
                layout="NN",
                grad=True,
            )

        if not weight.stop_gradient:
            wgrad, grad_bias, _ = gemm(
                inputmat,
                grad_output,
                ctx.activation_dtype,
                get_workspace(),
                layout="NT",
                grad=True,
                use_bias=ctx.use_bias,
            )

        if not ctx.use_bias:
            return (
                wgrad if not weight.stop_gradient else None,
                dgrad.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
            )

        return (
            wgrad if not weight.stop_gradient else None,
            dgrad.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
        )


class Linear(TransformerEngineBaseLayer):
    """
    Applies a linear transformation to the incoming data :math:`y = xA^T + b`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_attr: Union[paddle.ParamAttr, None] = None,
        bias_attr: Union[paddle.ParamAttr, None, bool] = None,
        backend: str = 'transformer_engine',
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._dtype = self._helper.get_default_dtype()

        # TE linear weight is in column major
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

    def _te_forward(
        self,
        inp: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the linear transformation to the input.
        """

        with self.prepare_forward(inp) as inp:
            out = _Linear.apply(
                cast_if_needed(self.weight, self.activation_dtype),
                cast_if_needed(inp, self.activation_dtype),
                cast_if_needed(self.bias, self.activation_dtype),
                self.has_bias,
                self.activation_dtype,
            )

        return out

    def _pd_forward(
        self,
        inp: paddle.Tensor,
    ) -> paddle.Tensor:
        """Calls Paddle OP"""
        return F.linear(inp, self.weight, self.bias)

    def forward(self, *args, **kwargs):
        """forward"""
        if self.backend == 'transformer_engine':
            return self._te_forward(*args, **kwargs)
        if self.backend == 'paddle':
            return self._pd_forward(*args, **kwargs)
        raise AttributeError(f"Backend {self.backend} is not supported.")
