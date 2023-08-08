# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Linear API"""

from typing import Union, Tuple, Dict, Any

import paddle
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

from .base import (
    TransformerEngineBaseLayer,
    get_workspace,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)

from ..fp8 import get_fp8_te_dtype
from ..constants import FP8FwdTensors, FP8BwdTensors
from ..cpp_extensions import gemm, fp8_gemm, cast_to_fp8, cast_transpose
from ..utils import (
    cast_if_needed,
    cast_if_needed_inplace,
    assert_dim_for_fp8_forward_exec,
    get_bias_dtype,
)

__all__ = ["Linear", "_linear_fwd", "_linear_fwd_fp8", "_linear_bwd", "_linear_fwd_non_fp8"]


def _linear_fwd_fp8(
    inputmat: paddle.Tensor,
    inputmat_fp8_index: FP8FwdTensors,
    weight: paddle.Tensor,
    weight_fp8_index: FP8FwdTensors,
    bias: paddle.Tensor,
    use_bias: bool,
    fp8_meta: Dict[str, Any],
    activation_dtype: paddle.dtype,
    is_grad_enabled: bool,
):
    """FP8 path of Linear Fwd"""
    fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
    bias_dtype = get_bias_dtype(activation_dtype)
    bias = cast_if_needed_inplace(bias, bias_dtype)

    if is_grad_enabled:
        weight_fp8, weight_t_fp8 = cast_transpose(
            weight,
            fp8_meta["scaling_fwd"],
            weight_fp8_index,
            fp8_dtype_forward,
        )
    else:
        weight_t_fp8 = None
        weight_fp8 = cast_to_fp8(
            weight,
            fp8_meta["scaling_fwd"],
            weight_fp8_index,
            fp8_dtype_forward,
        )

    out = fp8_gemm(
        weight_fp8,
        fp8_meta["scaling_fwd"].scale_inv,
        weight_fp8_index,
        fp8_dtype_forward,
        inputmat,
        fp8_meta["scaling_fwd"].scale_inv,
        inputmat_fp8_index,
        fp8_dtype_forward,
        activation_dtype,
        get_workspace(),
        bias=bias,
        use_bias=use_bias,
        use_split_accumulator=_2X_ACC_FPROP,
    )

    return out, weight_t_fp8


def _linear_fwd_non_fp8(
    inputmat: paddle.Tensor,
    inputmat_fp8_index: FP8FwdTensors,
    weight: paddle.Tensor,
    weight_fp8_index: FP8FwdTensors,
    bias: paddle.Tensor,
    use_bias: bool,
    fp8_calibration: bool,
    fp8_meta: Dict[str, Any],
    activation_dtype: paddle.dtype,
    activation: str = "",
):
    """Non-FP8 path of Linear Fwd"""

    # Layer parameters are initialized as float32 dtype by default.
    # Cast the parameters to activation_dtype if the current dtype
    # does not match activation_dtype. The casting is inplace, so it
    # only needs to performed once throughout the traing process.
    weight = cast_if_needed_inplace(weight, activation_dtype)
    bias = cast_if_needed_inplace(bias, activation_dtype)

    if fp8_calibration:
        # amax of input
        fp8_meta["scaling_fwd"].amax_history[0, inputmat_fp8_index.value] = \
            paddle.max(paddle.abs(inputmat)).item()
        # amax of weight
        fp8_meta["scaling_fwd"].amax_history[0, weight_fp8_index.value] = \
            paddle.max(paddle.abs(weight)).item()

    outputs = gemm(weight,
                   inputmat,
                   activation_dtype,
                   get_workspace(),
                   bias=bias,
                   use_bias=use_bias,
                   gelu=(activation == 'gelu'))

    if activation == 'gelu':
        gelu_out, _, out = outputs
        return out, gelu_out

    out, _, _ = outputs
    return out


def _linear_fwd(
    inputmat: paddle.Tensor,
    inputmat_fp8_index: FP8FwdTensors,
    weight: paddle.Tensor,
    weight_fp8_index: FP8FwdTensors,
    bias: paddle.Tensor,
    use_bias: bool,
    fp8_enabled: bool,
    fp8_calibration: bool,
    fp8_meta: Dict[str, Any],
    activation_dtype: paddle.dtype,
    is_grad_enabled: bool,
):
    if fp8_enabled:
        out, weight_t_fp8 = _linear_fwd_fp8(
            inputmat,
            inputmat_fp8_index,
            weight,
            weight_fp8_index,
            bias,
            use_bias,
            fp8_meta,
            activation_dtype,
            is_grad_enabled,
        )
    else:
        out = _linear_fwd_non_fp8(
            inputmat,
            inputmat_fp8_index,
            weight,
            weight_fp8_index,
            bias,
            use_bias,
            fp8_calibration,
            fp8_meta,
            activation_dtype,
        )
    return (
        out,
        weight_t_fp8 if fp8_enabled else None,
    )


def _linear_bwd_fp8(
    inputmat: paddle.Tensor,
    inputmat_t: paddle.Tensor,
    inputmat_fp8_index: FP8FwdTensors,
    weight_t_fp8: paddle.Tensor,
    weight_fp8_index: FP8FwdTensors,
    grad_output: paddle.Tensor,
    grad_output_c: paddle.Tensor,
    grad_output_t: paddle.Tensor,
    grad_output_fp8_index: FP8BwdTensors,
    fwd_scale_inverses: paddle.Tensor,
    fp8_meta: Dict[str, Any],
    requires_dgrad: bool,
    requires_wgrad: bool,
    activation_dtype: paddle.dtype,
):
    dgrad, wgrad = None, None
    fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
    fp8_dtype_backward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
    if requires_dgrad:
        dgrad = fp8_gemm(
            weight_t_fp8,
            fwd_scale_inverses,
            weight_fp8_index,
            fp8_dtype_forward,
            grad_output_c,
            fp8_meta["scaling_bwd"].scale_inv,
            grad_output_fp8_index,
            fp8_dtype_backward,
            activation_dtype,
            get_workspace(),
            use_split_accumulator=_2X_ACC_DGRAD,
        )
    if requires_wgrad:
        if not fp8_meta["recipe"].override_linear_precision.wgrad:
            wgrad = fp8_gemm(
                inputmat_t,
                fwd_scale_inverses,
                inputmat_fp8_index,
                fp8_dtype_forward,
                grad_output_t,
                fp8_meta["scaling_bwd"].scale_inv,
                grad_output_fp8_index,
                fp8_dtype_backward,
                activation_dtype,
                get_workspace(),
                use_split_accumulator=_2X_ACC_WGRAD,
            )
        else:
            wgrad, _, _ = gemm(
                inputmat,
                grad_output,
                activation_dtype,
                get_workspace(),
                layout="NT",
                grad=True,
            )
    return dgrad, wgrad


def _linear_bwd_non_fp8(
    inputmat: paddle.Tensor,
    weight: paddle.Tensor,
    grad_output: paddle.Tensor,
    requires_bgrad: bool,
    requires_dgrad: bool,
    activation_dtype: paddle.dtype,
    gelu_input: Union[paddle.Tensor, None] = None,
    activation: str = "",
):
    """
    Performs Linear Backward. Optionally, fuses GELU backward and dbias.
    """
    dgrad, wgrad, bgrad = None, None, None
    requires_wgrad = not weight.stop_gradient
    if requires_dgrad:
        dgrad, _, _ = gemm(
            weight,
            grad_output,
            activation_dtype,
            get_workspace(),
            layout="NN",
            gelu=(activation == 'gelu'),
            gelu_input=gelu_input,
            grad=True,
        )
    if requires_wgrad:
        wgrad, bgrad, _ = gemm(
            inputmat,
            grad_output,
            activation_dtype,
            get_workspace(),
            layout="NT",
            grad=True,
            use_bias=requires_bgrad,
        )
    elif requires_bgrad:
        bgrad = grad_output.sum(axis=0)

    return dgrad, wgrad, bgrad


def _linear_bwd(
    inputmat: paddle.Tensor,
    inputmat_t: paddle.Tensor,
    inputmat_fp8_index: FP8FwdTensors,
    weight: paddle.Tensor,
    weight_t_fp8: paddle.Tensor,
    weight_fp8_index: FP8FwdTensors,
    grad_output: paddle.Tensor,
    grad_output_c: paddle.Tensor,
    grad_output_t: paddle.Tensor,
    grad_output_fp8_index: FP8BwdTensors,
    fwd_scale_inverses: paddle.Tensor,
    requires_bgrad: bool,
    fp8_enabled: bool,
    fp8_meta: Dict[str, Any],
    requires_dgrad: bool,
    activation_dtype: paddle.dtype,
):
    dgrad, wgrad, bgrad = None, None, None
    requires_wgrad = not weight.stop_gradient
    if fp8_enabled:
        dgrad, wgrad = _linear_bwd_fp8(
            inputmat,
            inputmat_t,
            inputmat_fp8_index,
            weight_t_fp8,
            weight_fp8_index,
            grad_output,
            grad_output_c,
            grad_output_t,
            grad_output_fp8_index,
            fwd_scale_inverses,
            fp8_meta,
            requires_dgrad,
            requires_wgrad,
            activation_dtype,
        )
    else:
        dgrad, wgrad, bgrad = _linear_bwd_non_fp8(
            inputmat,
            weight,
            grad_output,
            requires_bgrad,
            requires_dgrad,
            activation_dtype,
        )
    return dgrad, wgrad, bgrad


class _Linear(paddle.autograd.PyLayer):
    """TE implementation of Linear"""

    @staticmethod
    def forward(
        ctx,
        weight: paddle.Tensor,
        inp: paddle.Tensor,
        bias: paddle.Tensor,
        use_bias: bool,
        fp8_enabled: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        activation_dtype: paddle.dtype,
        is_grad_enabled: bool,
    ) -> paddle.Tensor:
        # Make sure input dimensions are compatible
        in_features = weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.reshape((-1, in_features))
        if fp8_enabled:
            assert_dim_for_fp8_forward_exec(inputmat)
            assert_dim_for_fp8_forward_exec(weight)

        inputmat_no_fp8 = inputmat

        # FP8 casting
        if fp8_enabled:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

            if not fp8_meta["recipe"].override_linear_precision.wgrad:
                if is_grad_enabled:
                    inputmat, inputmat_t = cast_transpose(
                        inputmat,
                        fp8_meta["scaling_fwd"],
                        FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                    )
                else:
                    inputmat = cast_to_fp8(
                        inputmat,
                        fp8_meta["scaling_fwd"],
                        FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                    )
            else:
                inputmat, inputmat_t = cast_to_fp8(
                    inputmat,
                    fp8_meta["scaling_fwd"],
                    FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                ), None

        # GEMM Fwd
        out, weight_t_fp8 = _linear_fwd(
            inputmat,
            FP8FwdTensors.GEMM1_INPUT,
            weight,
            FP8FwdTensors.GEMM1_WEIGHT,
            bias,
            use_bias,
            fp8_enabled,
            fp8_calibration,
            fp8_meta,
            activation_dtype,
            is_grad_enabled,
        )

        if is_grad_enabled:
            fp8_wgrad = fp8_enabled and not fp8_meta["recipe"].override_linear_precision.wgrad
            ctx.save_for_backward(
                inputmat_no_fp8 if not weight.stop_gradient and not fp8_wgrad else None,
                inputmat_t if not weight.stop_gradient and fp8_wgrad else None,
                weight,
                weight_t_fp8 if fp8_enabled else None,
                fp8_meta["scaling_fwd"].scale_inv.clone() if fp8_enabled else None,
            )
            ctx.activation_dtype = activation_dtype
            ctx.fp8_enabled = fp8_enabled
            ctx.fp8_meta = fp8_meta
            ctx.use_bias = use_bias
            ctx.inp_shape = inp.shape
            ctx.requires_dgrad = not inp.stop_gradient
            ctx.requires_bgrad = use_bias and not bias.stop_gradient

        return out.reshape((-1, *inp.shape[1:-1], out.shape[-1]))

    @staticmethod
    def backward(ctx, grad_output: paddle.Tensor) -> Tuple[Union[paddle.Tensor, None], ...]:
        with TransformerEngineBaseLayer.prepare_backward(ctx.fp8_enabled,
                                                         ctx.fp8_meta,
                                                         name="_Linear"):
            (
                inputmat,
                inputmat_t,
                weight,
                weight_t_fp8,
                fwd_scale_inverses,
            ) = ctx.saved_tensor()

            (
                grad_output,
                grad_output_c,
                grad_output_t,
                bgrad,
            ) = TransformerEngineBaseLayer.grad_output_preprocess(ctx, grad_output)

            dgrad, wgrad, bgrad_ = _linear_bwd(
                inputmat,
                inputmat_t,
                FP8FwdTensors.GEMM1_INPUT,
                weight,
                weight_t_fp8,
                FP8FwdTensors.GEMM1_WEIGHT,
                grad_output,
                grad_output_c,
                grad_output_t,
                FP8BwdTensors.GRAD_OUTPUT1,
                fwd_scale_inverses,
                ctx.requires_bgrad,
                ctx.fp8_enabled,
                ctx.fp8_meta,
                ctx.requires_dgrad,
                ctx.activation_dtype,
            )

            if not ctx.fp8_enabled:
                # bgrad is fused with gemm for non-FP8 path
                bgrad = bgrad_

            if not ctx.use_bias:
                return (
                    wgrad if not weight.stop_gradient else None,
                    dgrad.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
                )

            return (
                wgrad if not weight.stop_gradient else None,
                dgrad.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
                bgrad if ctx.requires_bgrad else None,
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
        use_default_bias = self._bias_attr is None or self._bias_attr is True
        if self.has_bias:
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=self._bias_attr if not use_default_bias else paddle.ParamAttr(
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
            # Layer input should be casted outside PyLayer, as performing
            # inplace cast to input tensors may cause problems when used
            # together with Paddle native layers.
            inp = cast_if_needed(inp, self.activation_dtype)
            out = _Linear.apply(
                self.weight,
                inp,
                self.bias,
                self.has_bias,
                self.fp8_enabled,
                self.fp8_calibration,
                self.fp8_meta,
                self.activation_dtype,
                paddle.is_grad_enabled(),
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
