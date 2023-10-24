# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""LayerNormLinear API"""

import os
from typing import Union, Tuple, Dict, Any, Optional

import paddle
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

from ..cpp_extensions import (
    cast_to_fp8,
    cast_from_fp8,
    layernorm_fwd,
    layernorm_fwd_fp8,
    layernorm_bwd,
    transpose,
)

from .base import TransformerEngineBaseLayer
from .linear import _linear_fwd, _linear_bwd
from ..constants import TE_DType, FP8FwdTensors, FP8BwdTensors, GemmParallelModes, dist_group_type
from ..distributed import (
    allreduce,
    get_tp_group_and_world_size,
    identity,
    track_rng_state,
    set_tensor_dist_attr,
    set_weight_tensor_dist_attr,
)
from ..fp8 import get_fp8_te_dtype
from ..utils import (
    assert_dim_for_fp8_forward_exec,
    cast_if_needed,
    cast_if_needed_inplace,
    divide,
    save_for_backward_allow_none,
    saved_tensor_allow_none,
)

__all__ = ["LayerNormLinear"]


def _layernorm_fwd_fp8_cast(
    inputmat: paddle.Tensor,
    ln_weight: paddle.Tensor,
    ln_bias: paddle.Tensor,
    out_fp8_index: FP8FwdTensors,
    eps: float,
    fp8_enabled: bool,
    fp8_meta: Dict[str, Any],
    activation_dtype: paddle.dtype,
    return_layernorm_output: bool,
    fwd_ln_sm_margin: int,
    zero_centered_gamma: bool,
):
    """Performs LayerNorm + FP8_Cast for FP8 path. LayerNorm only for BF16 path"""

    ln_weight = cast_if_needed_inplace(ln_weight, activation_dtype)
    ln_bias = cast_if_needed_inplace(ln_bias, activation_dtype)

    if fp8_enabled:
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        if not return_layernorm_output:
            ln_out, mu, rsigma = layernorm_fwd_fp8(
                inputmat,
                ln_weight,
                ln_bias,
                eps,
                fp8_meta["scaling_fwd"],
                out_fp8_index,
                fp8_dtype_forward,
                fwd_ln_sm_margin,
                zero_centered_gamma,
            )
            ln_out_return = ln_out
        else:
            ln_out_return, mu, rsigma = layernorm_fwd(inputmat, ln_weight, ln_bias, eps,
                                                      TE_DType[activation_dtype], fwd_ln_sm_margin,
                                                      zero_centered_gamma)
            ln_out = cast_to_fp8(
                ln_out_return,
                fp8_meta["scaling_fwd"],
                out_fp8_index,
                fp8_dtype_forward,
            )
    else:
        ln_out, mu, rsigma = layernorm_fwd(inputmat, ln_weight, ln_bias, eps,
                                           TE_DType[activation_dtype], fwd_ln_sm_margin,
                                           zero_centered_gamma)
        ln_out_return = ln_out

    return (
        ln_out_return,
        ln_out,
        mu,
        rsigma,
    )


def _layernorm_bwd(
    inputmat: paddle.Tensor,
    dgrad: paddle.Tensor,
    ln_weight: paddle.Tensor,
    mu: paddle.Tensor,
    rsigma: paddle.Tensor,
    grad_ln_out_return: paddle.Tensor,
    return_layernorm_output: bool,
    bwd_ln_sm_margin: int,
    zero_centered_gamma: bool,
):
    # LayerNorm gradient
    d_ln_out = dgrad.reshape(inputmat.shape)
    # Residual gradient
    if return_layernorm_output:
        d_ln_out = d_ln_out + grad_ln_out_return.reshape(d_ln_out.shape)

    return layernorm_bwd(d_ln_out, inputmat, mu, rsigma, ln_weight, bwd_ln_sm_margin,
                         zero_centered_gamma)


class _LayerNormLinear(paddle.autograd.PyLayer):
    """TE implementation of LayerNormLinear"""

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
        fp8_enabled: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        activation_dtype: paddle.dtype,
        return_layernorm_output: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        parallel_mode: Union[str, None],
        tensor_parallel: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
    ) -> Union[Tuple[paddle.Tensor, ...], paddle.Tensor]:
        # Make sure input dimensions are compatible
        in_features = ln_weight.shape[0]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.reshape((-1, in_features))
        if fp8_enabled:
            assert_dim_for_fp8_forward_exec(inputmat)
            assert_dim_for_fp8_forward_exec(weight)

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

        # Linear Fwd
        out, weight_t_fp8 = _linear_fwd(
            ln_out,
            FP8FwdTensors.GEMM1_INPUT,
            weight,
            FP8FwdTensors.GEMM1_WEIGHT,
            bias,
            use_bias,
            fp8_enabled,
            fp8_calibration,
            fp8_meta,
            activation_dtype,
            parallel_mode,
            tensor_parallel,
            tp_group,
            is_grad_enabled,
        )

        if is_grad_enabled:
            save_for_backward_allow_none(
                ctx,
                inputmat,
                ln_weight,
                mu,
                rsigma,
                weight,
                weight_t_fp8 if fp8_enabled else None,
                ln_out,
                fp8_meta["scaling_fwd"].scale_inv.clone() if fp8_enabled else None,
            )

            ctx.activation_dtype = activation_dtype
            ctx.fp8_enabled = fp8_enabled
            ctx.fp8_meta = fp8_meta
            ctx.use_bias = use_bias
            ctx.inp_shape = inp.shape
            ctx.return_layernorm_output = return_layernorm_output
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.parallel_mode = parallel_mode
            ctx.tensor_parallel = tensor_parallel
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.requires_dgrad = not inp.stop_gradient
            ctx.requires_wgrad = not weight.stop_gradient
            ctx.requires_bgrad = use_bias and not bias.stop_gradient
            ctx.requires_ln_bgrad = not ln_bias.stop_gradient
            ctx.requires_ln_wgrad = not ln_weight.stop_gradient
        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        out = out.reshape((-1, *inp.shape[1:-1], out.shape[-1]))

        if return_layernorm_output:
            return out, ln_out_return.reshape(inp.shape)
        return out

    @staticmethod
    def backward(
            ctx, *grad_outputs: Tuple[paddle.Tensor,
                                      ...]) -> Tuple[Union[paddle.Tensor, None], ...]:
        with TransformerEngineBaseLayer.prepare_backward(ctx.fp8_enabled,
                                                         ctx.fp8_meta,
                                                         ctx.tp_group,
                                                         ctx.tp_size,
                                                         name="_LayerNormLinear"):
            (    # pylint: disable=unbalanced-tuple-unpacking
                inputmat,
                ln_weight,
                mu,
                rsigma,
                weight,
                weight_t_fp8,
                ln_out,
                fwd_scale_inverses,
            ) = saved_tensor_allow_none(ctx)

            (
                grad_output,
                grad_output_c,
                grad_output_t,
                bgrad,
            ) = TransformerEngineBaseLayer.grad_output_preprocess(ctx, grad_outputs[0])

            # Prepare ln_out for Linear bwd
            ln_out_no_fp8, ln_out_t = None, None
            if ctx.fp8_enabled:
                fp8_dtype_forward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=True)
                fp8_wgrad = not ctx.fp8_meta["recipe"].override_linear_precision.wgrad
                if ctx.requires_wgrad:
                    if fp8_wgrad:
                        ln_out_t = transpose(ln_out, fp8_dtype_forward)
                    else:
                        ln_out_no_fp8 = cast_from_fp8(
                            ln_out,
                            ctx.fp8_meta["scaling_fwd"],
                            FP8FwdTensors.GEMM1_INPUT,
                            fp8_dtype_forward,
                            TE_DType[ctx.activation_dtype],
                        )

            # Linear Bwd
            dgrad, wgrad, bgrad_ = _linear_bwd(
                ln_out_no_fp8 if ctx.fp8_enabled else ln_out,
                ln_out_t,
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
                True,    # Always compute dgrad to feed into LayerNorm bwd
                ctx.requires_wgrad,
                ctx.activation_dtype,
                ctx.parallel_mode,
                ctx.tensor_parallel,
                ctx.tp_group,
            )

            if not ctx.fp8_enabled:
                # bgrad is fused with gemm for non-FP8 path
                bgrad = bgrad_

            # LayerNorm Bwd
            dxmat, dgamma, dbeta = _layernorm_bwd(
                inputmat,
                dgrad,
                ln_weight,
                mu,
                rsigma,
                grad_outputs[1] if ctx.return_layernorm_output else None,
                ctx.return_layernorm_output,
                ctx.bwd_ln_sm_margin,
                ctx.zero_centered_gamma,
            )

            bgrad = bgrad if ctx.requires_bgrad else None
            bgrad_out = (bgrad,) if ctx.use_bias else ()

            return (
                dxmat.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
                dgamma if ctx.requires_ln_wgrad else None,
                dbeta if ctx.requires_ln_bgrad else None,
                wgrad if ctx.requires_wgrad else None,
                *bgrad_out,
            )


class LayerNormLinear(TransformerEngineBaseLayer):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    weight_attr: Union[paddle.ParamAttr, None], default = None
                optional `paddle.ParamAttr` for weight.
    bias_attr: Union[paddle.ParamAttr, None, bool], default = None
              optional `paddle.ParamAttr` for bias.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
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
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    parallel_mode : {None, 'Column', 'Row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.
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
        parallel_mode: Optional[str] = None,
        tp_group: Union[dist_group_type, None] = None,
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

        # Set parallel configs
        self.tp_group, self.tp_size = get_tp_group_and_world_size(tp_group,
                                                                  enable_tp=parallel_mode
                                                                  is not None)
        self.tensor_parallel = self.tp_size > 1
        self.parallel_mode = parallel_mode
        assert (self.parallel_mode
                in GemmParallelModes), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        # LayerNorm weights
        self.ln_weight = self.create_parameter(
            shape=[self.in_features],
            attr=paddle.ParamAttr(initializer=Constant(
                value=0.0 if self.zero_centered_gamma else 1.0)),
            dtype=self._dtype,
            is_bias=False,
        )

        self.ln_bias = self.create_parameter(
            shape=[self.in_features],
            attr=paddle.ParamAttr(initializer=Constant(value=0.0)),
            dtype=self._dtype,
            is_bias=True,
        )

        # Initialize Linear weight parameter
        with track_rng_state(enable=self.tensor_parallel):
            # TE linear weight is in column major
            self.weight = self.create_parameter(
                shape=[self.out_features, self.in_features]
                if self.backend == 'transformer_engine' else [self.in_features, self.out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
        set_weight_tensor_dist_attr(self.weight, self.tensor_parallel, self.parallel_mode,
                                    self.backend)

        # Initialize Linear bias parameter
        self.has_bias = self._bias_attr is not False
        use_default_bias = self._bias_attr is None or self._bias_attr is True
        if self.has_bias:
            self.bias = self.create_parameter(
                shape=[self.out_features],
                attr=self._bias_attr if not use_default_bias else paddle.ParamAttr(
                    initializer=Constant(value=0.0)),
                dtype=self._dtype,
                is_bias=True,
            )
            if parallel_mode == "column":
                set_tensor_dist_attr(self.bias, self.tensor_parallel, axis=0)
        else:
            self.bias = None

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.tensor_parallel and self.has_bias:
            self.gemm_bias_fused_add = False
        else:
            self.gemm_bias_fused_add = True

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
            # Layer input should be casted outside PyLayer, as performing
            # inplace cast to input tensors may cause problems when used
            # together with Paddle native layers.
            inp = cast_if_needed(inp, self.activation_dtype)
            out = _LayerNormLinear.apply(
                inp,
                self.ln_weight,
                self.ln_bias,
                self.weight,
                self.bias if self.gemm_bias_fused_add else None,
                self.has_bias and self.gemm_bias_fused_add,
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
                self.parallel_mode,
                self.tensor_parallel,
                self.tp_group,
                self.tp_size,
            )

        if self.return_layernorm_output:
            out, ln_out = out

        if not self.gemm_bias_fused_add:
            out = out + cast_if_needed_inplace(self.bias, self.activation_dtype)

        if self.return_layernorm_output:
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
        if self.parallel_mode == 'column' and self.tensor_parallel:
            ln_out = identity(ln_out, self.tp_group)
        out = F.linear(ln_out, self.weight, self.bias if self.gemm_bias_fused_add else None)
        if self.parallel_mode == 'row' and self.tensor_parallel:
            out = allreduce(out, self.tp_group)
            out = out + self.bias if self.bias is not None else out
        if self.return_layernorm_output:
            return out, ln_out
        return out

    def forward(self, *args, **kwargs):
        """
        Apply layer normalization to the input followed by a linear transformation.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        """
        if self.backend == 'transformer_engine':
            return self._te_forward(*args, **kwargs)
        if self.backend == 'paddle':
            return self._pd_forward(*args, **kwargs)
        raise AttributeError(f"Backend {self.backend} is not supported.")
