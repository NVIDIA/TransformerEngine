# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Linear API"""

from typing import Union, Tuple

import paddle

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

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        weight = cast_if_needed(weight, activation_dtype)
        bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

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
        weight_attr=None,
        has_bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        # TE linear weight is in column major
        self.weight = self.create_parameter(
            shape=[out_features, in_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )

        if self.has_bias:
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            self.bias = None

    def forward(
        self,
        inp: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the linear transformation to the input.
        """

        with self.prepare_forward(inp) as inp:
            out = _Linear.apply(
                self.weight,
                inp,
                self.bias,
                self.has_bias,
                self.activation_dtype,
            )

        return out
