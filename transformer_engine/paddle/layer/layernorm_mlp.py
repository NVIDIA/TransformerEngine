# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""LayerNormMLP API"""

import os
import warnings
from typing import Union, Tuple, Dict, Any, Optional

import paddle
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

from .base import TransformerEngineBaseLayer
from .layernorm_linear import _apply_normalization_fwd, _apply_normalization_bwd
from .linear import _linear_fwd_fp8, _linear_fwd_non_fp8, _linear_bwd_fp8, _linear_bwd_non_fp8
from ..constants import TE_DType, FP8FwdTensors, FP8BwdTensors, dist_group_type
from ..cpp_extensions import (
    cast_from_fp8,
    gelu_fp8,
    swiglu_fp8,
    swiglu,
    dswiglu,
    cast_transpose_bgrad,
    dgelu_cast_transpose_bgrad_fp8,
)
from ..distributed import (
    allreduce,
    get_tp_group_and_world_size,
    identity,
    track_rng_state,
    set_tensor_dist_attr,
    set_weight_tensor_dist_attr,
    mark_as_sequence_parallel_parameter,
)
from ..fp8 import get_fp8_te_dtype
from ..utils import (
    assert_dim_for_fp8_forward_exec,
    cast_if_needed,
    cast_if_needed_inplace,
    divide,
    get_paddle_act_func,
    save_for_backward_allow_none,
    saved_tensor_allow_none,
)

__all__ = ["LayerNormMLP"]


def _mlp_forward(
    inputmat: paddle.Tensor,
    inputmat_fp8_index: FP8FwdTensors,
    fc1_weight: paddle.Tensor,
    fc1_weight_fp8: Optional[paddle.Tensor],
    fc1_weight_t_fp8: Optional[paddle.Tensor],
    fc1_weight_fp8_index: FP8FwdTensors,
    fc1_bias: Union[paddle.Tensor, None],
    use_fc1_bias: bool,
    fc2_input_fp8_index: FP8FwdTensors,  # FP8FwdTensors.GEMM2_INPUT
    fc2_weight: paddle.Tensor,
    fc2_weight_fp8: Optional[paddle.Tensor],
    fc2_weight_t_fp8: Optional[paddle.Tensor],
    fc2_weight_fp8_index: FP8FwdTensors,
    fc2_bias: Union[paddle.Tensor, None],
    use_fc2_bias: bool,
    fp8_enabled: bool,
    fp8_calibration: bool,
    fp8_meta: Dict[str, Any],
    activation_dtype: paddle.dtype,
    activation: str,
    is_grad_enabled: bool,
    set_parallel_mode: bool,
    tensor_parallel: bool,
    sequence_parallel: bool,
    tp_group: Union[dist_group_type, None],
    is_first_microbatch: bool,
):
    if fp8_enabled:
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        fc1_out, fc1_weight_t_fp8 = _linear_fwd_fp8(
            inputmat,
            inputmat_fp8_index,
            fc1_weight,
            fc1_weight_fp8,
            fc1_weight_t_fp8,
            fc1_weight_fp8_index,
            fc1_bias,
            use_fc1_bias,
            fp8_meta,
            activation_dtype,
            "column" if set_parallel_mode else None,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            is_grad_enabled,
            is_first_microbatch,
        )
        if activation == "gelu":
            gelu_out = gelu_fp8(
                fc1_out,
                fp8_meta["scaling_fwd"],
                fc2_input_fp8_index,
                fp8_dtype_forward,
            )
        elif activation == "swiglu":
            gelu_out = swiglu_fp8(
                fc1_out,
                fp8_meta["scaling_fwd"],
                fc2_input_fp8_index,
                fp8_dtype_forward,
            )
        else:
            raise NotImplementedError("Activation type " + activation + " is not supported!")

        fc2_out, fc2_weight_t_fp8 = _linear_fwd_fp8(
            gelu_out,
            fc2_input_fp8_index,
            fc2_weight,
            fc2_weight_fp8,
            fc2_weight_t_fp8,
            fc2_weight_fp8_index,
            fc2_bias,
            use_fc2_bias,
            fp8_meta,
            activation_dtype,
            "row" if set_parallel_mode else None,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            is_grad_enabled,
            is_first_microbatch,
        )
    else:
        fc1_outputs = _linear_fwd_non_fp8(
            inputmat,
            inputmat_fp8_index,
            fc1_weight,
            fc1_weight_fp8_index,
            fc1_bias,
            use_fc1_bias,
            fp8_calibration,
            fp8_meta,
            activation_dtype,
            "column" if set_parallel_mode else None,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            activation=activation,
        )

        if activation == "gelu":
            fc1_out, gelu_out = fc1_outputs
        elif activation == "swiglu":
            fc1_out = fc1_outputs
            gelu_out = swiglu(fc1_out, TE_DType[activation_dtype])
        else:
            raise NotImplementedError("Activation type " + activation + " is not supported!")

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
            "row" if set_parallel_mode else None,
            tensor_parallel,
            sequence_parallel,
            tp_group,
        )
    return (
        fc1_out,
        gelu_out,
        fc2_out,
        fc1_weight_t_fp8 if fp8_enabled else None,
        fc2_weight_t_fp8 if fp8_enabled else None,
    )


def _mlp_backward(
    fc1_input: paddle.Tensor,  # ln_out, BF16 / FP8
    fc1_input_fp8_index: FP8FwdTensors,
    fc1_weight: paddle.Tensor,
    fc1_weight_t_fp8: paddle.Tensor,
    fc1_weight_fp8_index: FP8FwdTensors,
    fc1_grad_output_fp8_index: FP8BwdTensors,  # FP8BwdTensors.GRAD_OUTPUT2
    requires_fc1_wgrad: bool,
    requires_fc1_bgrad: bool,
    fc1_out: paddle.Tensor,
    fc2_input: paddle.Tensor,  # gelu_out
    fc2_input_fp8_index: FP8FwdTensors,  # FP8FwdTensors.GEMM2_INPUT
    fc2_weight: paddle.Tensor,
    fc2_weight_t_fp8: paddle.Tensor,
    fc2_weight_fp8_index: FP8FwdTensors,
    requires_fc2_wgrad: bool,
    requires_fc2_bgrad: bool,
    grad_output: paddle.Tensor,
    grad_output_c: paddle.Tensor,
    grad_output_t: paddle.Tensor,
    grad_output_fp8_index: FP8BwdTensors,  # FP8BwdTensors.GRAD_OUTPUT1
    fwd_scale_inverses: paddle.Tensor,
    fp8_enabled: bool,
    fp8_meta: Dict[str, Any],
    requires_dgrad: bool,
    activation_dtype: paddle.dtype,
    activation: str,
    set_parallel_mode: bool,
    tensor_parallel: bool,
    sequence_parallel: bool,
    tp_group: Union[dist_group_type, None],
    fuse_wgrad_accumulation: bool,
    accumulate_wgrad_into_param_main_grad: bool,
):
    (
        fc1_dgrad,
        fc1_wgrad,
        fc1_bgrad,
        fc2_wgrad,
        fc2_bgrad,
    ) = (
        None,
        None,
        None,
        None,
        None,
    )

    if fp8_enabled:
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        fp8_dtype_backward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
        # FC2 Bwd
        fp8_wgrad = not fp8_meta["recipe"].override_linear_precision.wgrad
        if requires_fc2_wgrad and not fp8_wgrad:
            fc2_input = cast_from_fp8(
                fc2_input,
                fp8_meta["scaling_fwd"],
                fc2_input_fp8_index,
                fp8_dtype_forward,
                TE_DType[activation_dtype],
            )

        fc2_dgrad, fc2_wgrad = _linear_bwd_fp8(
            fc2_input,
            None,
            fc2_input_fp8_index,
            fc2_weight,
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
            "row" if set_parallel_mode else None,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            fuse_wgrad_accumulation,
            accumulate_wgrad_into_param_main_grad,
        )

        if activation == "gelu":
            # GELU Bwd
            dgelu, dgelu_t, fc1_bgrad_ = dgelu_cast_transpose_bgrad_fp8(
                fc2_dgrad,
                fc1_out,
                fp8_meta["scaling_bwd"],
                fc1_grad_output_fp8_index,
                fp8_dtype_backward,
            )
        elif activation == "swiglu":
            dgelu = dswiglu(fc2_dgrad, fc1_out, TE_DType[fc2_dgrad.dtype])
            fc1_bgrad_, dgelu, dgelu_t = cast_transpose_bgrad(
                dgelu,
                fp8_meta["scaling_bwd"],
                fc1_grad_output_fp8_index,
                fp8_dtype_backward,
            )

        if requires_fc1_bgrad:
            fc1_bgrad = fc1_bgrad_

        # FC1 Bwd
        dgelu_no_fp8 = None
        if requires_fc1_wgrad and not fp8_wgrad:
            # TODO(tizheng) Paddle lacks fused dgelu_bgrad OP. Cast from dgrad(fp8) instead.
            dgelu_no_fp8 = cast_from_fp8(
                dgelu,
                fp8_meta["scaling_bwd"],
                fc1_grad_output_fp8_index,
                fp8_dtype_backward,
                TE_DType[activation_dtype],
            )
            fc1_input = cast_from_fp8(
                fc1_input,
                fp8_meta["scaling_fwd"],
                fc1_input_fp8_index,
                fp8_dtype_forward,
                TE_DType[activation_dtype],
            )

        fc1_dgrad, fc1_wgrad = _linear_bwd_fp8(
            fc1_input,
            None,
            fc1_input_fp8_index,
            fc1_weight,
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
            "column" if set_parallel_mode else None,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            fuse_wgrad_accumulation,
            accumulate_wgrad_into_param_main_grad,
        )
    else:
        dgelu, fc2_wgrad, fc2_bgrad = _linear_bwd_non_fp8(
            fc2_input,
            fc2_weight,
            grad_output,
            requires_fc2_bgrad,
            True,
            requires_fc2_wgrad,
            activation_dtype,
            "row" if set_parallel_mode else None,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            accumulate_wgrad_into_param_main_grad=accumulate_wgrad_into_param_main_grad,
            gelu_input=fc1_out,
            activation=activation,
        )

        if activation == "swiglu":
            dgelu = dswiglu(dgelu, fc1_out, TE_DType[dgelu.dtype])

        fc1_dgrad, fc1_wgrad, fc1_bgrad = _linear_bwd_non_fp8(
            fc1_input,
            fc1_weight,
            dgelu,
            requires_fc1_bgrad,
            requires_dgrad,
            requires_fc1_wgrad,
            activation_dtype,
            "column" if set_parallel_mode else None,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            accumulate_wgrad_into_param_main_grad=accumulate_wgrad_into_param_main_grad,
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
        ln_bias: Union[paddle.Tensor, None],
        fc1_weight: paddle.Tensor,
        fc1_weight_fp8: Optional[paddle.Tensor],
        fc1_weight_t_fp8: Optional[paddle.Tensor],
        fc1_bias: Union[paddle.Tensor, None],
        use_fc1_bias: bool,
        fc2_weight: paddle.Tensor,
        fc2_weight_fp8: Optional[paddle.Tensor],
        fc2_weight_t_fp8: Optional[paddle.Tensor],
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
        normalization: str,
        activation: str,
        set_parallel_mode: bool,
        tensor_parallel: bool,
        sequence_parallel: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        fuse_wgrad_accumulation: bool,
        is_first_microbatch: bool,
    ) -> Union[Tuple[paddle.Tensor, ...], paddle.Tensor]:
        if normalization == "RMSNorm":
            assert ln_bias is None, "RMSNorm does not support bias!"
        else:  # LayerNorm
            assert ln_bias is not None, "LayerNorm requires bias!"
        # Make sure input dimensions are compatible
        in_features = ln_weight.shape[0]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.reshape((-1, in_features))
        if fp8_enabled:
            assert_dim_for_fp8_forward_exec(inputmat)
            assert_dim_for_fp8_forward_exec(fc1_weight)
            assert_dim_for_fp8_forward_exec(fc2_weight)

        # only support gelu for now
        assert activation in ["gelu", "swiglu"], "Only gelu and swiglu are supported for now"

        # LayerNorm Fwd + FP8 Cast
        (
            ln_out_return,
            ln_out,
            mu,
            rsigma,
        ) = _apply_normalization_fwd(
            normalization,
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
            fc1_weight_fp8,
            fc1_weight_t_fp8,
            FP8FwdTensors.GEMM1_WEIGHT,
            fc1_bias,
            use_fc1_bias,
            FP8FwdTensors.GEMM2_INPUT,
            fc2_weight,
            fc2_weight_fp8,
            fc2_weight_t_fp8,
            FP8FwdTensors.GEMM2_WEIGHT,
            fc2_bias,
            use_fc2_bias,
            fp8_enabled,
            fp8_calibration,
            fp8_meta,
            activation_dtype,
            activation,
            is_grad_enabled,
            set_parallel_mode,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            is_first_microbatch,
        )

        if is_grad_enabled:
            save_for_backward_allow_none(
                ctx,
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
            ctx.set_parallel_mode = set_parallel_mode
            ctx.tensor_parallel = tensor_parallel
            ctx.sequence_parallel = sequence_parallel
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.requires_dgrad = not inp.stop_gradient
            ctx.requires_fc1_wgrad = not fc1_weight.stop_gradient
            ctx.requires_fc2_wgrad = not fc2_weight.stop_gradient
            ctx.requires_fc1_bgrad = use_fc1_bias and not fc1_bias.stop_gradient
            ctx.requires_fc2_bgrad = use_fc2_bias and not fc2_bias.stop_gradient
            ctx.requires_ln_bgrad = ln_bias is not None and not ln_bias.stop_gradient
            ctx.requires_ln_wgrad = not ln_weight.stop_gradient
            ctx.is_first_microbatch = is_first_microbatch
            ctx.has_ln_bias = ln_bias is not None
            ctx.normalization = normalization

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        fc2_out = fc2_out.reshape((-1, *inp.shape[1:-1], fc2_out.shape[-1]))

        if return_layernorm_output:
            return fc2_out, ln_out_return.reshape(inp.shape)
        return fc2_out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[paddle.Tensor, ...]
    ) -> Tuple[Union[paddle.Tensor, None], ...]:
        with TransformerEngineBaseLayer.prepare_backward(
            ctx.fp8_enabled, ctx.fp8_meta, ctx.tp_group, ctx.tp_size, name="_LayerNormMLP"
        ):
            (  # pylint: disable=unbalanced-tuple-unpacking
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
            ) = saved_tensor_allow_none(ctx)

            ctx.use_bias = ctx.use_fc2_bias  # For grad_output_preprocess
            (
                grad_output,
                grad_output_c,
                grad_output_t,
                fc2_bgrad,
            ) = TransformerEngineBaseLayer.grad_output_preprocess(ctx, grad_outputs[0], True)

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

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
                ctx.requires_fc1_wgrad,
                ctx.requires_fc1_bgrad,
                fc1_out,
                gelu_out,
                FP8FwdTensors.GEMM2_INPUT,
                fc2_weight,
                fc2_weight_t_fp8,
                FP8FwdTensors.GEMM2_WEIGHT,
                ctx.requires_fc2_wgrad,
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
                ctx.set_parallel_mode,
                ctx.tensor_parallel,
                ctx.sequence_parallel,
                ctx.tp_group,
                ctx.fuse_wgrad_accumulation,
                accumulate_wgrad_into_param_main_grad,
            )
            if not ctx.fp8_enabled:
                # fc2_bias is fused with gemm for non-FP8 path
                fc2_bgrad = fc2_bgrad_

            # LayerNorm Bwd
            dxmat, dgamma, dbeta = _apply_normalization_bwd(
                ctx.normalization,
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
            dbeta = dbeta if ctx.requires_ln_bgrad else None
            dbeta_out = (dbeta,) if ctx.has_ln_bias else ()

            if not ctx.fp8_enabled or ctx.is_first_microbatch is None:
                fc1_weight_cache_grad = ()
                fc2_weight_cache_grad = ()
            else:
                # weight_fp8 and weight_t_fp8 are stop_gradient tensors
                fc1_weight_cache_grad = (None, None)
                fc2_weight_cache_grad = (None, None)

        if ctx.requires_fc1_wgrad and ctx.fuse_wgrad_accumulation:
            fc1_wgrad = None
        if ctx.requires_fc2_wgrad and ctx.fuse_wgrad_accumulation:
            fc2_wgrad = None

        return (
            dxmat.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma if ctx.requires_ln_wgrad else None,
            *dbeta_out,
            fc1_wgrad if ctx.requires_fc1_wgrad else None,
            *fc1_weight_cache_grad,
            *fc1_bgrad_out,
            fc2_wgrad if ctx.requires_fc2_wgrad else None,
            *fc2_weight_cache_grad,
            *fc2_bgrad_out,
        )


class LayerNormMLP(TransformerEngineBaseLayer):
    r"""
    Applies layer normalization on the input followed by the MLP module, consisting of
    2 successive linear transformations, separated by the GeLU activation.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    weight_attr: Union[paddle.ParamAttr, None], default = None
                optional `paddle.ParamAttr` for weight.
    bias_attr: Union[paddle.ParamAttr, None, bool], default = None
              optional `paddle.ParamAttr` for bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    activation : str, default = 'gelu'
          activation function used.
          Options: 'gelu', 'geglu', 'relu', 'reglu', 'squared_relu', 'swiglu'.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module
                             is taken post layernorm.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    backend: {'transformer_engine', 'paddle'}, default = 'transformer_engine'
             if set to 'paddle', a framework only no-FP8 path is executed with limited optimization.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, FC1 is used as Column Parallel and FC2 is used as Row
                      Parallel as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : paddle.distributed.collective.Group, default = `None`
               tensor parallel process group.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        weight_attr: Union[paddle.ParamAttr, None] = None,
        bias_attr: Union[paddle.ParamAttr, None, bool] = None,
        normalization: str = "LayerNorm",
        activation: str = "gelu",
        return_layernorm_output: bool = False,
        zero_centered_gamma: bool = False,
        set_parallel_mode: bool = False,
        sequence_parallel: bool = False,
        tp_group: Optional[dist_group_type] = None,
        fuse_wgrad_accumulation: bool = False,
        backend: str = "transformer_engine",
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.eps = eps
        self.normalization = normalization
        assert normalization in ["LayerNorm", "RMSNorm"], "Normalization type not supported"
        self.activation = activation
        self.return_layernorm_output = return_layernorm_output
        self.zero_centered_gamma = zero_centered_gamma
        self.backend = backend

        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._dtype = self._helper.get_default_dtype()

        # Set parallel configs
        self.tp_group, self.tp_size = get_tp_group_and_world_size(
            tp_group, enable_tp=set_parallel_mode
        )
        self.tensor_parallel = self.tp_size > 1
        self.set_parallel_mode = set_parallel_mode
        self.sequence_parallel = self.tensor_parallel and sequence_parallel

        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation

        if self.set_parallel_mode:
            self.size_per_partition = divide(self.ffn_hidden_size, self.tp_size)
        else:
            self.size_per_partition = self.ffn_hidden_size

        # LayerNorm weights
        self.ln_weight = self.create_parameter(
            shape=[self.hidden_size],
            attr=paddle.ParamAttr(
                initializer=Constant(value=0.0 if self.zero_centered_gamma else 1.0)
            ),
            dtype=self._dtype,
            is_bias=False,
        )

        if self.normalization != "RMSNorm":
            self.ln_bias = self.create_parameter(
                shape=[self.hidden_size],
                attr=paddle.ParamAttr(initializer=Constant(value=0.0)),
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            self.ln_bias = None

        if self.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.ln_weight)
            if self.ln_bias is not None:
                mark_as_sequence_parallel_parameter(self.ln_bias)

        # FC1 weights
        if self.activation in ["swiglu"]:
            fc1_output_features = self.size_per_partition * 2
        else:
            fc1_output_features = self.size_per_partition

        with track_rng_state(enable=self.tensor_parallel):
            self.fc1_weight = self.create_parameter(
                shape=(
                    [fc1_output_features, self.hidden_size]
                    if self.backend == "transformer_engine"
                    else [self.hidden_size, fc1_output_features]
                ),
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
        set_weight_tensor_dist_attr(
            self.fc1_weight, self.tensor_parallel, parallel_mode="column", backend=self.backend
        )
        self.fp8_weight_shapes.append(self.fc1_weight.shape)

        self.has_bias = self._bias_attr is not False
        use_default_bias = self._bias_attr is None or self._bias_attr is True
        if use_default_bias:
            self._bias_attr = paddle.ParamAttr(initializer=Constant(value=0.0))

        if self.has_bias:
            self.fc1_bias = self.create_parameter(
                shape=[fc1_output_features],
                attr=self._bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            set_tensor_dist_attr(self.fc1_bias, self.tensor_parallel, axis=0)
        else:
            self.fc1_bias = None

        # FC2 weights
        self.fc2_weight = self.create_parameter(
            shape=(
                [self.hidden_size, self.size_per_partition]
                if self.backend == "transformer_engine"
                else [self.size_per_partition, self.hidden_size]
            ),
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        set_weight_tensor_dist_attr(
            self.fc2_weight, self.tensor_parallel, parallel_mode="row", backend=self.backend
        )
        self.fp8_weight_shapes.append(self.fc2_weight.shape)

        if self.has_bias:
            self.fc2_bias = self.create_parameter(
                shape=[self.hidden_size],
                attr=self._bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            if self.set_parallel_mode and self.sequence_parallel:
                mark_as_sequence_parallel_parameter(self.fc2_bias)
        else:
            self.fc2_bias = None

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.set_parallel_mode and self.tensor_parallel and self.has_bias:
            self.gemm_bias_fused_add = False
        else:
            self.gemm_bias_fused_add = True

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))
        self.inf_ln_sm_margin = int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", "0"))

    def _te_forward(
        self,
        inp: paddle.Tensor,
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a linear transformation.
        """

        with self.prepare_forward(inp, num_gemms=2, is_first_microbatch=is_first_microbatch) as inp:
            # Layer input should be casted outside PyLayer, as performing
            # inplace cast to input tensors may cause problems when used
            # together with Paddle native layers.
            inp = cast_if_needed(inp, self.activation_dtype)

            # Get persistent fp8 weight buffer. None if buffer does not exist.
            fc1_weight_fp8, fc1_weight_t_fp8, fc2_weight_fp8, fc2_weight_t_fp8 = (
                self.get_fp8_weights_scratchpad(is_first_microbatch)
            )

            out = _LayerNormMLP.apply(
                inp,
                self.ln_weight,
                self.ln_bias,
                self.fc1_weight,
                fc1_weight_fp8,
                fc1_weight_t_fp8,
                self.fc1_bias,
                self.has_bias,
                self.fc2_weight,
                fc2_weight_fp8,
                fc2_weight_t_fp8,
                self.fc2_bias,
                self.has_bias,
                self.eps,
                self.fp8_enabled,
                self.fp8_calibration,
                self.fp8_meta,
                self.activation_dtype,
                self.return_layernorm_output,
                paddle.is_grad_enabled(),
                self.fwd_ln_sm_margin if paddle.is_grad_enabled() else self.inf_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                self.normalization,
                self.activation,
                self.set_parallel_mode,
                self.tensor_parallel,
                self.sequence_parallel,
                self.tp_group,
                self.tp_size,
                self.fuse_wgrad_accumulation,
                is_first_microbatch,
            )

        if self.return_layernorm_output:
            out, ln_out = out

        if not self.gemm_bias_fused_add:
            out = out + cast_if_needed_inplace(self.fc2_bias, self.activation_dtype)

        if self.return_layernorm_output:
            return out, ln_out
        return out

    def _pd_forward(
        self,
        inp: paddle.Tensor,
        is_first_microbatch: Optional[bool] = None,
    ) -> paddle.Tensor:
        """Calls Paddle OP"""
        if self.zero_centered_gamma:
            raise NotImplementedError(
                "Paddle backend does not support LayerNorm with zero-centered scale."
            )

        if is_first_microbatch is not None:
            warnings.warn(
                "`is_first_microbatch` is not supported for paddle backend and is ignored."
            )

        if self.normalization == "RMSNorm":
            norm = paddle.rsqrt(paddle.mean(inp**2, axis=-1, keepdim=True) + self.eps)
            norm_out = inp * norm * self.ln_weight
        else:  # LayerNorm
            norm_out = F.layer_norm(
                x=inp,
                normalized_shape=inp.shape[-1],
                weight=self.ln_weight,
                bias=self.ln_bias,
                epsilon=self.eps,
            )
        if self.set_parallel_mode and self.tensor_parallel:
            norm_out = identity(norm_out, self.tp_group)
        fc1_out = F.linear(norm_out, self.fc1_weight, self.fc1_bias)
        act_func = get_paddle_act_func(self.activation)
        act_out = act_func(fc1_out)
        out = F.linear(
            act_out, self.fc2_weight, self.fc2_bias if self.gemm_bias_fused_add else None
        )
        if self.set_parallel_mode and self.tensor_parallel:
            out, _ = allreduce(out, self.tp_group)
            out = out + self.fc2_bias if self.fc2_bias is not None else out
        if self.return_layernorm_output:
            return out, norm_out
        return out

    def forward(self, *args, **kwargs):
        """
        Apply layer normalization to the input followed by a feedforward network (MLP Block).

        Parameters
        ----------
        inp : paddle.Tensor
             Input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
        """
        if self.backend == "transformer_engine":
            return self._te_forward(*args, **kwargs)
        if self.backend == "paddle":
            return self._pd_forward(*args, **kwargs)
        raise AttributeError(f"Backend {self.backend} is not supported.")
