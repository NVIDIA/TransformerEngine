# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""LayerNormLinear API"""

import warnings
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
    rmsnorm_fwd_fp8,
    rmsnorm_fwd,
    rmsnorm_bwd,
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
    mark_as_sequence_parallel_parameter,
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


def _apply_normalization_fwd(
    normalization: str,
    inputmat: paddle.Tensor,
    norm_weight: paddle.Tensor,
    norm_bias: Union[paddle.Tensor, None],
    out_fp8_index: FP8FwdTensors,
    eps: float,
    fp8_enabled: bool,
    fp8_meta: Dict[str, Any],
    activation_dtype: paddle.dtype,
    return_norm_output: bool,
    fwd_norm_sm_margin: int,
    zero_centered_gamma: bool,
):
    """Performs LayerNorm + FP8_Cast for FP8 path. LayerNorm only for BF16 path"""
    assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"
    if normalization == "RMSNorm":
        assert norm_bias is None, "RMSNorm does not support bias!"
    norm_weight = cast_if_needed_inplace(norm_weight, activation_dtype)
    if norm_bias is not None:
        norm_bias = cast_if_needed_inplace(norm_bias, activation_dtype)

    norm_kwargs = {
        "inp": inputmat,
        "weight": norm_weight,
        "eps": eps,
        "otype": TE_DType[activation_dtype],
        "sm_margin": fwd_norm_sm_margin,
        "zero_centered_gamma": zero_centered_gamma,
    }

    fwd_normalization_funcs = {
        ("LayerNorm", True, True): layernorm_fwd,
        ("LayerNorm", True, False): layernorm_fwd_fp8,
        ("LayerNorm", False, True): layernorm_fwd,
        ("LayerNorm", False, False): layernorm_fwd,
        ("RMSNorm", True, True): rmsnorm_fwd,
        ("RMSNorm", True, False): rmsnorm_fwd_fp8,
        ("RMSNorm", False, True): rmsnorm_fwd,
        ("RMSNorm", False, False): rmsnorm_fwd,
    }

    if normalization == "LayerNorm":
        norm_kwargs["bias"] = norm_bias
    norm_fwd_func = fwd_normalization_funcs[(normalization, fp8_enabled, return_norm_output)]

    if fp8_enabled:
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        if not return_norm_output:
            fp8_kwargs = {
                "fp8_meta_tensor": fp8_meta["scaling_fwd"],
                "fp8_tensor": out_fp8_index,
                "otype": fp8_dtype_forward,
            }
            norm_kwargs.update(fp8_kwargs)

    out_tuple = norm_fwd_func(**norm_kwargs)

    if normalization == "LayerNorm":
        norm_out_return, mu, rsigma = out_tuple
    else:  # RMSNorm
        norm_out_return, rsigma = out_tuple
        mu = None

    if fp8_enabled and return_norm_output:
        norm_out = cast_to_fp8(
            norm_out_return,
            fp8_meta["scaling_fwd"],
            out_fp8_index,
            fp8_dtype_forward,
        )
    else:
        norm_out = norm_out_return

    return (
        norm_out_return,
        norm_out,
        mu,
        rsigma,
    )


def _apply_normalization_bwd(
    normalization: str,
    inputmat: paddle.Tensor,
    dgrad: paddle.Tensor,
    norm_weight: paddle.Tensor,
    mu: Union[paddle.Tensor, None],
    rsigma: paddle.Tensor,
    grad_norm_out_return: paddle.Tensor,
    return_norm_output: bool,
    bwd_norm_sm_margin: int,
    zero_centered_gamma: bool,
):
    assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"
    if normalization == "RMSNorm":
        assert mu is None, "RMSNorm does not support bias!"
    # LayerNorm gradient
    d_norm_out = dgrad.reshape(inputmat.shape)
    # Residual gradient
    if return_norm_output:
        d_norm_out = d_norm_out + grad_norm_out_return.reshape(d_norm_out.shape)

    norm_bwd_func = layernorm_bwd if normalization == "LayerNorm" else rmsnorm_bwd
    norm_bwd_kwargs = {
        "dz": d_norm_out,
        "x": inputmat,
        "rsigma": rsigma,
        "gamma": norm_weight,
        "sm_margin": bwd_norm_sm_margin,
        "zero_centered_gamma": zero_centered_gamma,
    }
    if normalization == "LayerNorm":
        norm_bwd_kwargs["mu"] = mu

    out_tuple = norm_bwd_func(**norm_bwd_kwargs)
    if normalization == "LayerNorm":
        dxmat, dgamma, dbeta = out_tuple
    else:  # RMSNorm
        dxmat, dgamma = out_tuple
        dbeta = None

    return dxmat, dgamma, dbeta


class _LayerNormLinear(paddle.autograd.PyLayer):
    """TE implementation of LayerNormLinear"""

    @staticmethod
    def forward(
        ctx,
        inp: paddle.Tensor,
        ln_weight: paddle.Tensor,
        ln_bias: Union[paddle.Tensor, None],
        weight: paddle.Tensor,
        weight_fp8: Optional[paddle.Tensor],
        weight_t_fp8: Optional[paddle.Tensor],
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
        normalization: str,
        parallel_mode: Union[str, None],
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
            assert_dim_for_fp8_forward_exec(weight)

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

        # Linear Fwd
        out, weight_t_fp8 = _linear_fwd(
            ln_out,
            FP8FwdTensors.GEMM1_INPUT,
            weight,
            weight_fp8,
            weight_t_fp8,
            FP8FwdTensors.GEMM1_WEIGHT,
            bias,
            use_bias,
            fp8_enabled,
            fp8_calibration,
            fp8_meta,
            activation_dtype,
            parallel_mode,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            is_grad_enabled,
            is_first_microbatch,
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
            ctx.sequence_parallel = sequence_parallel
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.requires_dgrad = not inp.stop_gradient
            ctx.requires_wgrad = not weight.stop_gradient
            ctx.requires_bgrad = use_bias and not bias.stop_gradient
            ctx.requires_ln_bgrad = ln_bias is not None and not ln_bias.stop_gradient
            ctx.requires_ln_wgrad = not ln_weight.stop_gradient
            ctx.is_first_microbatch = is_first_microbatch
            ctx.has_ln_bias = ln_bias is not None
            ctx.normalization = normalization

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        out = out.reshape((-1, *inp.shape[1:-1], out.shape[-1]))

        if return_layernorm_output:
            return out, ln_out_return.reshape(inp.shape)
        return out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[paddle.Tensor, ...]
    ) -> Tuple[Union[paddle.Tensor, None], ...]:
        with TransformerEngineBaseLayer.prepare_backward(
            ctx.fp8_enabled, ctx.fp8_meta, ctx.tp_group, ctx.tp_size, name="_LayerNormLinear"
        ):
            (  # pylint: disable=unbalanced-tuple-unpacking
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
            ) = TransformerEngineBaseLayer.grad_output_preprocess(
                ctx, grad_outputs[0], ctx.parallel_mode == "row"
            )

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            # Prepare ln_out for Linear bwd
            linear_inputmat = ln_out
            if ctx.fp8_enabled:
                fp8_dtype_forward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=True)
                if ctx.requires_wgrad and ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                    linear_inputmat = cast_from_fp8(
                        ln_out,
                        ctx.fp8_meta["scaling_fwd"],
                        FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                        TE_DType[ctx.activation_dtype],
                    )

            # Linear Bwd
            dgrad, wgrad, bgrad_ = _linear_bwd(
                linear_inputmat,
                None,  # inputmat_t will be automatically computed if not provided
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
                True,  # Always compute dgrad to feed into LayerNorm bwd
                ctx.requires_wgrad,
                ctx.activation_dtype,
                ctx.parallel_mode,
                ctx.tensor_parallel,
                ctx.sequence_parallel,
                ctx.tp_group,
                ctx.fuse_wgrad_accumulation,
                accumulate_wgrad_into_param_main_grad,
            )

            if not ctx.fp8_enabled:
                # bgrad is fused with gemm for non-FP8 path
                bgrad = bgrad_

            # LayerNorm Bwd
            dxmat, dgamma, dbeta = _apply_normalization_bwd(
                ctx.normalization,
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
            dbeta = dbeta if ctx.requires_ln_bgrad else None
            dbeta_out = (dbeta,) if ctx.has_ln_bias else ()

            if not ctx.fp8_enabled or ctx.is_first_microbatch is None:
                weight_cache_grad = ()
            else:
                # weight_fp8 and weight_t_fp8 are stop_gradient tensors
                weight_cache_grad = (None, None)

        if ctx.requires_wgrad and ctx.fuse_wgrad_accumulation:
            wgrad = None
        return (
            dxmat.reshape(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma if ctx.requires_ln_wgrad else None,
            *dbeta_out,
            wgrad if ctx.requires_wgrad else None,
            *weight_cache_grad,
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
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
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
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.

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
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        weight_attr: Union[paddle.ParamAttr, None] = None,
        bias_attr: Union[paddle.ParamAttr, None, bool] = None,
        normalization: str = "LayerNorm",
        return_layernorm_output: bool = False,
        zero_centered_gamma: bool = False,
        parallel_mode: Optional[str] = None,
        sequence_parallel: bool = False,
        tp_group: Union[dist_group_type, None] = None,
        fuse_wgrad_accumulation: bool = False,
        backend: str = "transformer_engine",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.normalization = normalization
        assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"
        self.return_layernorm_output = return_layernorm_output
        self.zero_centered_gamma = zero_centered_gamma
        self.backend = backend

        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._dtype = self._helper.get_default_dtype()

        # Set parallel configs
        self.tp_group, self.tp_size = get_tp_group_and_world_size(
            tp_group, enable_tp=parallel_mode is not None
        )
        self.tensor_parallel = self.tp_size > 1
        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in GemmParallelModes
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        self.sequence_parallel = self.tensor_parallel and sequence_parallel

        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation

        # LayerNorm weights
        self.ln_weight = self.create_parameter(
            shape=[self.in_features],
            attr=paddle.ParamAttr(
                initializer=Constant(value=0.0 if self.zero_centered_gamma else 1.0)
            ),
            dtype=self._dtype,
            is_bias=False,
        )
        if self.normalization != "RMSNorm":
            self.ln_bias = self.create_parameter(
                shape=[self.in_features],
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

        # Initialize Linear weight parameter
        with track_rng_state(enable=self.tensor_parallel):
            # TE linear weight is in column major
            self.weight = self.create_parameter(
                shape=(
                    [self.out_features, self.in_features]
                    if self.backend == "transformer_engine"
                    else [self.in_features, self.out_features]
                ),
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
        set_weight_tensor_dist_attr(
            self.weight, self.tensor_parallel, self.parallel_mode, self.backend
        )
        self.fp8_weight_shapes.append(self.weight.shape)

        # Initialize Linear bias parameter
        self.has_bias = self._bias_attr is not False
        use_default_bias = self._bias_attr is None or self._bias_attr is True
        if self.has_bias:
            self.bias = self.create_parameter(
                shape=[self.out_features],
                attr=(
                    self._bias_attr
                    if not use_default_bias
                    else paddle.ParamAttr(initializer=Constant(value=0.0))
                ),
                dtype=self._dtype,
                is_bias=True,
            )
            if parallel_mode == "column":
                set_tensor_dist_attr(self.bias, self.tensor_parallel, axis=0)
            if parallel_mode == "row" and self.sequence_parallel:
                mark_as_sequence_parallel_parameter(self.bias)
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
        self.inf_ln_sm_margin = int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", "0"))

    def _te_forward(
        self,
        inp: paddle.Tensor,
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a linear transformation.
        """

        with self.prepare_forward(inp, is_first_microbatch=is_first_microbatch) as inp:
            # Layer input should be casted outside PyLayer, as performing
            # inplace cast to input tensors may cause problems when used
            # together with Paddle native layers.
            inp = cast_if_needed(inp, self.activation_dtype)

            # Get persistent fp8 weight buffer. None if buffer does not exist.
            weight_fp8, weight_t_fp8 = self.get_fp8_weights_scratchpad(is_first_microbatch)

            out = _LayerNormLinear.apply(
                inp,
                self.ln_weight,
                self.ln_bias,
                self.weight,
                weight_fp8,
                weight_t_fp8,
                self.bias if self.gemm_bias_fused_add else None,
                self.has_bias and self.gemm_bias_fused_add,
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
                self.parallel_mode,
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
            out = out + cast_if_needed_inplace(self.bias, self.activation_dtype)

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

        if self.parallel_mode == "column" and self.tensor_parallel:
            norm_out = identity(norm_out, self.tp_group)
        out = F.linear(norm_out, self.weight, self.bias if self.gemm_bias_fused_add else None)
        if self.parallel_mode == "row" and self.tensor_parallel:
            out, _ = allreduce(out, self.tp_group)
            out = out + self.bias if self.bias is not None else out
        if self.return_layernorm_output:
            return out, norm_out
        return out

    def forward(self, *args, **kwargs):
        """
        Apply layer normalization to the input followed by a linear transformation.

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
