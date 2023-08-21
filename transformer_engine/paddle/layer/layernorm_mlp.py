# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""LayerNormMLP API"""

import os
from typing import Union, Tuple, Dict, Any

import paddle
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

from ..cpp_extensions import (
    cast_from_fp8,
    dgelu_cast_transpose_bgrad_fp8,
    gelu_fp8,
    transpose,
)

from .base import TransformerEngineBaseLayer
from .layernorm_linear import _layernorm_fwd_fp8_cast, _layernorm_bwd
from .linear import _linear_fwd_fp8, _linear_fwd_non_fp8, _linear_bwd_fp8, _linear_bwd_non_fp8
from ..constants import TE_DType, FP8FwdTensors, FP8BwdTensors
from ..fp8 import get_fp8_te_dtype
from ..utils import cast_if_needed, assert_dim_for_fp8_forward_exec, get_paddle_act_func

__all__ = ["LayerNormMLP"]


def _mlp_forward(
    inputmat: paddle.Tensor,
    inputmat_fp8_index: FP8FwdTensors,
    fc1_weight: paddle.Tensor,
    fc1_weight_fp8_index: FP8FwdTensors,
    fc1_bias: Union[paddle.Tensor, None],
    use_fc1_bias: bool,
    fc2_input_fp8_index: FP8FwdTensors,    # FP8FwdTensors.GEMM2_INPUT
    fc2_weight: paddle.Tensor,
    fc2_weight_fp8_index: FP8FwdTensors,
    fc2_bias: Union[paddle.Tensor, None],
    use_fc2_bias: bool,
    fp8_enabled: bool,
    fp8_calibration: bool,
    fp8_meta: Dict[str, Any],
    activation_dtype: paddle.dtype,
    is_grad_enabled: bool,
):
    if fp8_enabled:
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        fc1_out, fc1_weight_t_fp8 = _linear_fwd_fp8(
            inputmat,
            inputmat_fp8_index,
            fc1_weight,
            fc1_weight_fp8_index,
            fc1_bias,
            use_fc1_bias,
            fp8_meta,
            activation_dtype,
            is_grad_enabled,
        )

        gelu_out = gelu_fp8(
            fc1_out,
            fp8_meta["scaling_fwd"],
            fc2_input_fp8_index,
            fp8_dtype_forward,
        )

        fc2_out, fc2_weight_t_fp8 = _linear_fwd_fp8(
            gelu_out,
            fc2_input_fp8_index,
            fc2_weight,
            fc2_weight_fp8_index,
            fc2_bias,
            use_fc2_bias,
            fp8_meta,
            activation_dtype,
            is_grad_enabled,
        )
    else:
        fc1_out, gelu_out = _linear_fwd_non_fp8(
            inputmat,
            inputmat_fp8_index,
            fc1_weight,
            fc1_weight_fp8_index,
            fc1_bias,
            use_fc1_bias,
            fp8_calibration,
            fp8_meta,
            activation_dtype,
            activation='gelu',
        )

        fc2_out = _linear_fwd_non_fp8(
            gelu_out,
            fc2_input_fp8_index,
            fc2_weight,
            fc2_weight_fp8_index,
            fc2_bias,
            use_fc2_bias,
            fp8_calibration,
            fp8_meta,
            activation_dtype,
        )
    return (
        fc1_out,
        gelu_out,
        fc2_out,
        fc1_weight_t_fp8 if fp8_enabled else None,
        fc2_weight_t_fp8 if fp8_enabled else None,
    )


def _mlp_backward(
    fc1_input: paddle.Tensor,    # ln_out, BF16 / FP8
    fc1_input_fp8_index: FP8FwdTensors,
    fc1_weight: paddle.Tensor,
    fc1_weight_t_fp8: paddle.Tensor,
    fc1_weight_fp8_index: FP8FwdTensors,
    fc1_grad_output_fp8_index: FP8BwdTensors,    # FP8BwdTensors.GRAD_OUTPUT2
    requires_fc1_bgrad: bool,
    fc1_out: paddle.Tensor,
    fc2_input: paddle.Tensor,    # gelu_out
    fc2_input_fp8_index: FP8FwdTensors,    # FP8FwdTensors.GEMM2_INPUT
    fc2_weight: paddle.Tensor,
    fc2_weight_t_fp8: paddle.Tensor,
    fc2_weight_fp8_index: FP8FwdTensors,
    requires_fc2_bgrad: bool,
    grad_output: paddle.Tensor,
    grad_output_c: paddle.Tensor,
    grad_output_t: paddle.Tensor,
    grad_output_fp8_index: FP8BwdTensors,    # FP8BwdTensors.GRAD_OUTPUT1
    fwd_scale_inverses: paddle.Tensor,
    fp8_enabled: bool,
    fp8_meta: Dict[str, Any],
    requires_dgrad: bool,
    activation_dtype: paddle.dtype,
    activation: str,
):
    (
        fc1_dgrad,
        fc1_wgrad,
        fc1_bgrad,
        fc2_wgrad,
        fc2_bgrad,
    ) = None, None, None, None, None

    if fp8_enabled:
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        fp8_dtype_backward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
        # FC2 Bwd
        fc2_input_no_fp8, fc2_input_t = None, None
        fp8_wgrad = not fp8_meta["recipe"].override_linear_precision.wgrad
        requires_fc2_wgrad = not fc2_weight.stop_gradient
        if requires_fc2_wgrad:
            if fp8_wgrad:
                fc2_input_t = transpose(fc2_input, fp8_dtype_forward)
            else:
                fc2_input_no_fp8 = cast_from_fp8(
                    fc2_input,
                    fp8_meta["scaling_fwd"],
                    fc2_input_fp8_index,
                    fp8_dtype_forward,
                    TE_DType[activation_dtype],
                )

        fc2_dgrad, fc2_wgrad = _linear_bwd_fp8(
            fc2_input_no_fp8,
            fc2_input_t,
            fc2_input_fp8_index,
            fc2_weight_t_fp8,
            fc2_weight_fp8_index,
            grad_output,
            grad_output_c,
            grad_output_t,
            grad_output_fp8_index,
            fwd_scale_inverses,
            fp8_meta,
            True,
            requires_fc2_wgrad,
            activation_dtype,
        )

        # GELU Bwd
        dgelu, dgelu_t, fc1_bgrad_ = dgelu_cast_transpose_bgrad_fp8(
            fc2_dgrad,
            fc1_out,
            fp8_meta["scaling_bwd"],
            fc1_grad_output_fp8_index,
            fp8_dtype_backward,
        )

        if requires_fc1_bgrad:
            fc1_bgrad = fc1_bgrad_

        # FC2 Bwd
        requires_fc1_wgrad = not fc1_weight.stop_gradient
        dgelu_no_fp8, fc1_input_no_fp8, fc1_input_t = None, None, None
        if requires_fc1_wgrad:
            if fp8_wgrad:
                fc1_input_t = transpose(fc1_input, fp8_dtype_forward)
            else:
                # TODO(tizheng) Paddle lacks fused dgelu_bgrad OP. Cast from dgrad(fp8) instead.
                dgelu_no_fp8 = cast_from_fp8(
                    dgelu,
                    fp8_meta["scaling_bwd"],
                    fc1_grad_output_fp8_index,
                    fp8_dtype_backward,
                    TE_DType[activation_dtype],
                )
                fc1_input_no_fp8 = cast_from_fp8(
                    fc1_input,
                    fp8_meta["scaling_fwd"],
                    fc1_input_fp8_index,
                    fp8_dtype_forward,
                    TE_DType[activation_dtype],
                )

        fc1_dgrad, fc1_wgrad = _linear_bwd_fp8(
            fc1_input_no_fp8,
            fc1_input_t,
            fc1_input_fp8_index,
            fc1_weight_t_fp8,
            fc1_weight_fp8_index,
            dgelu_no_fp8,
            dgelu,
            dgelu_t,
            fc1_grad_output_fp8_index,
            fwd_scale_inverses,
            fp8_meta,
            requires_dgrad,
            requires_fc1_wgrad,
            activation_dtype,
        )
    else:
        dgelu, fc2_wgrad, fc2_bgrad = _linear_bwd_non_fp8(
            fc2_input,
            fc2_weight,
            grad_output,
            requires_fc2_bgrad,
            True,
            activation_dtype,
            gelu_input=fc1_out,
            activation=activation,
        )
        fc1_dgrad, fc1_wgrad, fc1_bgrad = _linear_bwd_non_fp8(
            fc1_input,
            fc1_weight,
            dgelu,
            requires_fc1_bgrad,
            requires_dgrad,
            activation_dtype,
        )
    return (
        fc1_dgrad,
        fc1_wgrad,
        fc1_bgrad,
        fc2_wgrad,
        fc2_bgrad,
    )


class _LayerNormMLP(paddle.autograd.PyLayer):
    """TE implementation of LayerNormMLP"""

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
        fp8_enabled: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        activation_dtype: paddle.dtype,
        return_layernorm_output: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        activation: str,
    ) -> Union[Tuple[paddle.Tensor, ...], paddle.Tensor]:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.reshape((-1, in_features))
        if fp8_enabled:
            assert_dim_for_fp8_forward_exec(inputmat)
            assert_dim_for_fp8_forward_exec(fc1_weight)
            assert_dim_for_fp8_forward_exec(fc2_weight)

        # only support gelu for now
        assert activation == 'gelu'

        # LayerNorm Fwd + FP8 Cast
        (
            ln_out_return,
            ln_out,
            mu,
            rsigma,
        ) = _layernorm_fwd_fp8_cast(
            inputmat,
            ln_weight,
            ln_bias,
            FP8FwdTensors.GEMM1_INPUT,
            eps,
            fp8_enabled,
            fp8_meta,
            activation_dtype,
            return_layernorm_output,
            fwd_ln_sm_margin,
            zero_centered_gamma,
        )

        (
            fc1_out,
            gelu_out,
            fc2_out,
            fc1_weight_t_fp8,
            fc2_weight_t_fp8,
        ) = _mlp_forward(
            ln_out,
            FP8FwdTensors.GEMM1_INPUT,
            fc1_weight,
            FP8FwdTensors.GEMM1_WEIGHT,
            fc1_bias,
            use_fc1_bias,
            FP8FwdTensors.GEMM2_INPUT,
            fc2_weight,
            FP8FwdTensors.GEMM2_WEIGHT,
            fc2_bias,
            use_fc2_bias,
            fp8_enabled,
            fp8_calibration,
            fp8_meta,
            activation_dtype,
            is_grad_enabled,
        )

        if is_grad_enabled:
            ctx.save_for_backward(
                inputmat,
                ln_weight,
                mu,
                rsigma,
                ln_out,
                fc1_out,
                gelu_out,
                fc1_weight,
                fc1_weight_t_fp8,
                fc2_weight,
                fc2_weight_t_fp8,
                fp8_meta["scaling_fwd"].scale_inv.clone() if fp8_enabled else None,
            )
            ctx.activation_dtype = activation_dtype
            ctx.activation = activation
            ctx.fp8_enabled = fp8_enabled
            ctx.fp8_meta = fp8_meta
            ctx.use_fc1_bias = use_fc1_bias
            ctx.use_fc2_bias = use_fc2_bias
            ctx.inp_shape = inp.shape
            ctx.return_layernorm_output = return_layernorm_output
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.requires_dgrad = not inp.stop_gradient
            ctx.requires_fc1_bgrad = use_fc1_bias and not fc1_bias.stop_gradient
            ctx.requires_fc2_bgrad = use_fc2_bias and not fc2_bias.stop_gradient
            ctx.requires_ln_bgrad = not ln_bias.stop_gradient

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        fc2_out = fc2_out.reshape((-1, *inp.shape[1:-1], fc2_out.shape[-1]))

        if return_layernorm_output:
            return fc2_out, ln_out_return.reshape(inp.shape)
        return fc2_out

    @staticmethod
    def backward(
            ctx, *grad_outputs: Tuple[paddle.Tensor,
                                      ...]) -> Tuple[Union[paddle.Tensor, None], ...]:
        with TransformerEngineBaseLayer.prepare_backward(ctx.fp8_enabled,
                                                         ctx.fp8_meta,
                                                         name="_LayerNormMLP"):
            (
                inputmat,
                ln_weight,
                mu,
                rsigma,
                ln_out,
                fc1_out,
                gelu_out,
                fc1_weight,
                fc1_weight_t_fp8,
                fc2_weight,
                fc2_weight_t_fp8,
                fwd_scale_inverses,
            ) = ctx.saved_tensor()

            ctx.use_bias = ctx.use_fc2_bias    # For grad_output_preprocess
            (
                grad_output,
                grad_output_c,
                grad_output_t,
                fc2_bgrad,
            ) = TransformerEngineBaseLayer.grad_output_preprocess(ctx, grad_outputs[0])

            (
                fc1_dgrad,
                fc1_wgrad,
                fc1_bgrad,
                fc2_wgrad,
                fc2_bgrad_,
            ) = _mlp_backward(
                ln_out,
                FP8FwdTensors.GEMM1_INPUT,
                fc1_weight,
                fc1_weight_t_fp8,
                FP8FwdTensors.GEMM1_WEIGHT,
                FP8BwdTensors.GRAD_OUTPUT2,
                ctx.requires_fc1_bgrad,
                fc1_out,
                gelu_out,
                FP8FwdTensors.GEMM2_INPUT,
                fc2_weight,
                fc2_weight_t_fp8,
                FP8FwdTensors.GEMM2_WEIGHT,
                ctx.requires_fc2_bgrad,
                grad_output,
                grad_output_c,
                grad_output_t,
                FP8BwdTensors.GRAD_OUTPUT1,
                fwd_scale_inverses,
                ctx.fp8_enabled,
                ctx.fp8_meta,
                True,
                ctx.activation_dtype,
                ctx.activation,
            )
            if not ctx.fp8_enabled:
                # fc2_bias is fused with gemm for non-FP8 path
                fc2_bgrad = fc2_bgrad_

            # LayerNorm Bwd
            dxmat, dgamma, dbeta = _layernorm_bwd(
                inputmat,
                fc1_dgrad,
                ln_weight,
                mu,
                rsigma,
                grad_outputs[1] if ctx.return_layernorm_output else None,
                ctx.return_layernorm_output,
                ctx.bwd_ln_sm_margin,
                ctx.zero_centered_gamma,
            )

            fc1_bgrad = fc1_bgrad if ctx.requires_fc1_bgrad else None
            fc2_bgrad = fc2_bgrad if ctx.requires_fc2_bgrad else None
            fc1_bgrad_out = (fc1_bgrad,) if ctx.use_fc1_bias else ()
            fc2_bgrad_out = (fc2_bgrad,) if ctx.use_fc2_bias else ()

            return (
                dxmat.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
                dgamma if not ln_weight.stop_gradient else None,
                dbeta if ctx.requires_ln_bgrad else None,
                fc1_wgrad if not fc1_weight.stop_gradient else None,
                *fc1_bgrad_out,
                fc2_wgrad if not fc2_weight.stop_gradient else None,
                *fc2_bgrad_out,
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
        if self._bias_attr is None or self._bias_attr is True:
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

        with self.prepare_forward(inp, num_gemms=2) as inp:
            # Layer input should be casted outside PyLayer, as performing
            # inplace cast to input tensors may cause problems when used
            # together with Paddle native layers.
            inp = cast_if_needed(inp, self.activation_dtype)
            out = _LayerNormMLP.apply(
                inp,
                self.ln_weight,
                self.ln_bias,
                self.fc1_weight,
                self.fc1_bias,
                self.has_bias,
                self.fc2_weight,
                self.fc2_bias,
                self.has_bias,
                self.eps,
                self.fp8_enabled,
                self.fp8_calibration,
                self.fp8_meta,
                self.activation_dtype,
                self.return_layernorm_output,
                paddle.is_grad_enabled(),
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
                              normalized_shape=inp.shape[-1],
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
