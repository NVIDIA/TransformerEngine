# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Linear API"""

import warnings
from typing import Union, Tuple, Dict, Any, Optional

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

from ..constants import FP8FwdTensors, FP8BwdTensors, GemmParallelModes, dist_group_type
from ..cpp_extensions import gemm, fp8_gemm, cast_to_fp8, cast_transpose, transpose
from ..distributed import (
    allgather,
    allreduce,
    get_tp_group_and_world_size,
    identity,
    reduce_scatter,
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
    get_bias_dtype,
    save_for_backward_allow_none,
    saved_tensor_allow_none,
    clear_tensor_data,
)

__all__ = ["Linear"]


def _linear_fwd_fp8(
    inputmat: paddle.Tensor,
    inputmat_fp8_index: FP8FwdTensors,
    weight: paddle.Tensor,
    weight_fp8: Optional[paddle.Tensor],
    weight_t_fp8: Optional[paddle.Tensor],
    weight_fp8_index: FP8FwdTensors,
    bias: paddle.Tensor,
    use_bias: bool,
    fp8_meta: Dict[str, Any],
    activation_dtype: paddle.dtype,
    parallel_mode: Union[str, None],
    tensor_parallel: bool,
    sequence_parallel: bool,
    tp_group: Union[dist_group_type, None],
    is_grad_enabled: bool,
    is_first_microbatch: bool = None,
):
    """FP8 path of Linear Fwd"""
    fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
    bias_dtype = get_bias_dtype(activation_dtype)
    bias = cast_if_needed(bias, bias_dtype)

    if parallel_mode == "column" and sequence_parallel:
        inputmat_total, _ = allgather(inputmat, tp_group)
    else:
        inputmat_total = inputmat

    update_fp8_weights = is_first_microbatch is None or is_first_microbatch
    if is_grad_enabled:
        if update_fp8_weights:
            weight_fp8, weight_t_fp8 = cast_transpose(
                weight,
                fp8_meta["scaling_fwd"],
                weight_fp8_index,
                fp8_dtype_forward,
                cast_out=weight_fp8,
                transpose_out=weight_t_fp8,
            )
    else:
        weight_t_fp8 = None
        if update_fp8_weights:
            weight_fp8 = cast_to_fp8(
                weight,
                fp8_meta["scaling_fwd"],
                weight_fp8_index,
                fp8_dtype_forward,
                out=weight_fp8,
            )

    out, _ = fp8_gemm(
        weight_fp8,
        fp8_meta["scaling_fwd"].scale_inv,
        weight_fp8_index,
        fp8_dtype_forward,
        inputmat_total,
        fp8_meta["scaling_fwd"].scale_inv,
        inputmat_fp8_index,
        fp8_dtype_forward,
        activation_dtype,
        get_workspace(),
        bias=bias,
        use_bias=use_bias,
        use_split_accumulator=_2X_ACC_FPROP,
    )

    if parallel_mode == "row" and sequence_parallel:
        out, _ = reduce_scatter(out, tp_group)
    elif parallel_mode == "row" and tensor_parallel:
        out, _ = allreduce(out, tp_group)

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
    parallel_mode: Union[str, None],
    tensor_parallel: bool,
    sequence_parallel: bool,
    tp_group: Union[dist_group_type, None],
    activation: str = "",
):
    """Non-FP8 path of Linear Fwd"""

    if parallel_mode == "column" and sequence_parallel:
        inputmat_total, _ = allgather(inputmat, tp_group)
    else:
        inputmat_total = inputmat

    # Layer parameters are initialized as float32 dtype by default.
    # Cast the parameters to activation_dtype if the current dtype
    # does not match activation_dtype. The casting is inplace, so it
    # only needs to performed once throughout the traing process.
    weight = cast_if_needed_inplace(weight, activation_dtype)
    bias = cast_if_needed_inplace(bias, activation_dtype)

    if fp8_calibration:
        # amax of input
        fp8_meta["scaling_fwd"].amax_history[0, inputmat_fp8_index.value] = paddle.max(
            paddle.abs(inputmat_total)
        ).item()
        # amax of weight
        fp8_meta["scaling_fwd"].amax_history[0, weight_fp8_index.value] = paddle.max(
            paddle.abs(weight)
        ).item()
        fp8_meta["update_amax_and_scale_fwd"] = True

    outputs = gemm(
        weight,
        inputmat_total,
        activation_dtype,
        get_workspace(),
        bias=bias,
        use_bias=use_bias,
        gelu=(activation == "gelu"),
    )

    if activation == "gelu":
        gelu_out, _, out = outputs
        return out, gelu_out

    out, _, _ = outputs

    if parallel_mode == "row" and sequence_parallel:
        out, _ = reduce_scatter(out, tp_group)
    elif parallel_mode == "row" and tensor_parallel:
        out, _ = allreduce(out, tp_group)
    return out


def _linear_fwd(
    inputmat: paddle.Tensor,
    inputmat_fp8_index: FP8FwdTensors,
    weight: paddle.Tensor,
    weight_fp8: Optional[paddle.Tensor],
    weight_t_fp8: Optional[paddle.Tensor],
    weight_fp8_index: FP8FwdTensors,
    bias: paddle.Tensor,
    use_bias: bool,
    fp8_enabled: bool,
    fp8_calibration: bool,
    fp8_meta: Dict[str, Any],
    activation_dtype: paddle.dtype,
    parallel_mode: Union[str, None],
    tensor_parallel: bool,
    sequence_parallel: bool,
    tp_group: Union[dist_group_type, None],
    is_grad_enabled: bool,
    is_first_microbatch: bool = None,
):
    if fp8_enabled:
        out, weight_t_fp8 = _linear_fwd_fp8(
            inputmat,
            inputmat_fp8_index,
            weight,
            weight_fp8,
            weight_t_fp8,
            weight_fp8_index,
            bias,
            use_bias,
            fp8_meta,
            activation_dtype,
            parallel_mode,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            is_grad_enabled,
            is_first_microbatch,
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
            parallel_mode,
            tensor_parallel,
            sequence_parallel,
            tp_group,
        )
    return (
        out,
        weight_t_fp8 if fp8_enabled else None,
    )


def _linear_bwd_fp8(
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
    fp8_meta: Dict[str, Any],
    requires_dgrad: bool,
    requires_wgrad: bool,
    activation_dtype: paddle.dtype,
    parallel_mode: Union[str, None],
    tensor_parallel: bool,
    sequence_parallel: bool,
    tp_group: Union[dist_group_type, None],
    fuse_wgrad_accumulation: bool,
    accumulate_wgrad_into_param_main_grad: bool,
):
    dgrad, wgrad, handle = None, None, None

    # Overlap input AG with dgrad
    inputmat_total = None
    inputmat_t_total = None
    if requires_wgrad and parallel_mode == "column" and sequence_parallel:
        inputmat_total, handle = allgather(inputmat, tp_group, sync_op=not requires_dgrad)
    else:
        inputmat_total = inputmat
        inputmat_t_total = inputmat_t

    fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
    fp8_dtype_backward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
    if requires_dgrad:
        dgrad, _ = fp8_gemm(
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
        clear_tensor_data(grad_output_c)

        # Overlap dgrad-RS/AR with wgrad
        if parallel_mode == "column" and sequence_parallel:
            if handle is not None:
                handle.wait()
            dgrad, handle = reduce_scatter(dgrad, tp_group, sync_op=False)
        elif parallel_mode == "column" and tensor_parallel:
            dgrad, handle = allreduce(dgrad, tp_group, sync_op=False)

    if requires_wgrad:
        if not fp8_meta["recipe"].override_linear_precision.wgrad:
            if inputmat_t_total is None:
                inputmat_t_total = transpose(inputmat_total, fp8_dtype_backward)
                clear_tensor_data(inputmat_total)

            wgrad, _ = fp8_gemm(
                inputmat_t_total,
                fwd_scale_inverses,
                inputmat_fp8_index,
                fp8_dtype_forward,
                grad_output_t,
                fp8_meta["scaling_bwd"].scale_inv,
                grad_output_fp8_index,
                fp8_dtype_backward,
                "float32" if fuse_wgrad_accumulation else activation_dtype,
                get_workspace(),
                accumulate=accumulate_wgrad_into_param_main_grad,
                out=weight.main_grad if fuse_wgrad_accumulation else None,
                use_split_accumulator=_2X_ACC_WGRAD,
            )
            clear_tensor_data(inputmat_t_total, grad_output_t)
        else:
            wgrad, _, _ = gemm(
                inputmat_total,
                grad_output,
                activation_dtype,
                get_workspace(),
                grad=True,
                accumulate=accumulate_wgrad_into_param_main_grad,
                layout="NT",
                out=weight.main_grad if fuse_wgrad_accumulation else None,
                out_dtype="float32" if fuse_wgrad_accumulation else None,
            )
            clear_tensor_data(inputmat_total)

        if fuse_wgrad_accumulation:
            weight.main_grad = wgrad

    if parallel_mode == "column" and tensor_parallel and handle is not None:
        handle.wait()

    return dgrad, wgrad


def _linear_bwd_non_fp8(
    inputmat: paddle.Tensor,
    weight: paddle.Tensor,
    grad_output: paddle.Tensor,
    requires_bgrad: bool,
    requires_dgrad: bool,
    requires_wgrad: bool,
    activation_dtype: paddle.dtype,
    parallel_mode: Union[str, None],
    tensor_parallel: bool,
    sequence_parallel: bool,
    tp_group: Union[dist_group_type, None],
    fuse_wgrad_accumulation: bool,
    accumulate_wgrad_into_param_main_grad: bool,
    gelu_input: Union[paddle.Tensor, None] = None,
    activation: str = "",
):
    """
    Performs Linear Backward. Optionally, fuses GELU backward and dbias.
    """
    dgrad, wgrad, bgrad, handle = None, None, None, None

    # Overlap input AG with dgrad
    inputmat_total = None
    if requires_wgrad and parallel_mode == "column" and sequence_parallel:
        inputmat_total, handle = allgather(inputmat, tp_group, sync_op=not requires_dgrad)
    else:
        inputmat_total = inputmat

    if requires_dgrad:
        dgrad, _, _ = gemm(
            weight,
            grad_output,
            activation_dtype,
            get_workspace(),
            layout="NN",
            gelu=(activation == "gelu"),
            gelu_input=gelu_input,
            grad=True,
        )
        # Overlap dgrad-RS/AR with wgrad
        if parallel_mode == "column" and sequence_parallel:
            if handle is not None:
                handle.wait()
            dgrad, handle = reduce_scatter(dgrad, tp_group, sync_op=False)
        elif parallel_mode == "column" and tensor_parallel:
            dgrad, handle = allreduce(dgrad, tp_group, sync_op=False)

    if requires_wgrad:
        wgrad, bgrad, _ = gemm(
            inputmat_total,
            grad_output,
            activation_dtype,
            get_workspace(),
            grad=True,
            accumulate=accumulate_wgrad_into_param_main_grad,
            layout="NT",
            out=weight.main_grad if fuse_wgrad_accumulation else None,
            out_dtype="float32" if fuse_wgrad_accumulation else None,
            use_bias=requires_bgrad,
        )
        if fuse_wgrad_accumulation:
            weight.main_grad = wgrad

    elif requires_bgrad:
        bgrad = grad_output.sum(axis=0)

    if parallel_mode == "column" and tensor_parallel and handle is not None:
        handle.wait()

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
    requires_wgrad: bool,
    activation_dtype: paddle.dtype,
    parallel_mode: Union[str, None],
    tensor_parallel: bool,
    sequence_parallel: bool,
    tp_group: Union[dist_group_type, None],
    fuse_wgrad_accumulation: bool,
    accumulate_wgrad_into_param_main_grad: bool,
):
    dgrad, wgrad, bgrad = None, None, None
    if fp8_enabled:
        dgrad, wgrad = _linear_bwd_fp8(
            inputmat,
            inputmat_t,
            inputmat_fp8_index,
            weight,
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
            parallel_mode,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            accumulate_wgrad_into_param_main_grad=accumulate_wgrad_into_param_main_grad,
        )
    else:
        dgrad, wgrad, bgrad = _linear_bwd_non_fp8(
            inputmat,
            weight,
            grad_output,
            requires_bgrad,
            requires_dgrad,
            requires_wgrad,
            activation_dtype,
            parallel_mode,
            tensor_parallel,
            sequence_parallel,
            tp_group,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            accumulate_wgrad_into_param_main_grad=accumulate_wgrad_into_param_main_grad,
        )
    return dgrad, wgrad, bgrad


class _Linear(paddle.autograd.PyLayer):
    """TE implementation of Linear"""

    @staticmethod
    def forward(
        ctx,
        weight: paddle.Tensor,
        weight_fp8: Optional[paddle.Tensor],
        weight_t_fp8: Optional[paddle.Tensor],
        inp: paddle.Tensor,
        bias: paddle.Tensor,
        use_bias: bool,
        fp8_enabled: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        activation_dtype: paddle.dtype,
        is_grad_enabled: bool,
        parallel_mode: Union[str, None],
        tensor_parallel: bool,
        sequence_parallel: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        fuse_wgrad_accumulation: bool,
        is_first_microbatch: bool,
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
        inputmat_t = None
        if fp8_enabled:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            if (
                not fp8_meta["recipe"].override_linear_precision.wgrad
                and is_grad_enabled
                and not sequence_parallel
            ):
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

        # GEMM Fwd
        out, weight_t_fp8 = _linear_fwd(
            inputmat,
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
            saved_inputmat = None
            if fp8_enabled and sequence_parallel:
                saved_inputmat = inputmat
            else:
                saved_inputmat = inputmat_no_fp8
            save_for_backward_allow_none(
                ctx,
                saved_inputmat,
                inputmat_t,
                weight,
                weight_t_fp8 if fp8_enabled else None,
                fp8_meta["scaling_fwd"].scale_inv.clone() if fp8_enabled else None,
            )
            ctx.activation_dtype = activation_dtype
            ctx.fp8_enabled = fp8_enabled
            ctx.fp8_meta = fp8_meta
            ctx.use_bias = use_bias
            ctx.inp_shape = inp.shape
            ctx.parallel_mode = parallel_mode
            ctx.tensor_parallel = tensor_parallel
            ctx.sequence_parallel = sequence_parallel
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.requires_dgrad = not inp.stop_gradient
            ctx.requires_wgrad = not weight.stop_gradient
            ctx.requires_bgrad = use_bias and not bias.stop_gradient
            ctx.is_first_microbatch = is_first_microbatch

        return out.reshape((-1, *inp.shape[1:-1], out.shape[-1]))

    @staticmethod
    def backward(ctx, grad_output: paddle.Tensor) -> Tuple[Union[paddle.Tensor, None], ...]:
        with TransformerEngineBaseLayer.prepare_backward(
            ctx.fp8_enabled, ctx.fp8_meta, ctx.tp_group, ctx.tp_size, name="_Linear"
        ):

            (  # pylint: disable=unbalanced-tuple-unpacking
                inputmat,
                inputmat_t,
                weight,
                weight_t_fp8,
                fwd_scale_inverses,
            ) = saved_tensor_allow_none(ctx)

            (
                grad_output,
                grad_output_c,
                grad_output_t,
                bgrad,
            ) = TransformerEngineBaseLayer.grad_output_preprocess(
                ctx, grad_output, ctx.parallel_mode == "row"
            )
            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

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

            if not ctx.fp8_enabled or ctx.is_first_microbatch is None:
                weight_cache_grad = ()
            else:
                # weight_fp8 and weight_t_fp8 are stop_gradient tensors
                weight_cache_grad = (None, None)

            dgrad_return = dgrad.reshape(ctx.inp_shape) if ctx.requires_dgrad else None
            if not ctx.use_bias:
                bgrad_return = ()
            elif ctx.requires_bgrad:
                bgrad_return = (bgrad,)
            else:
                bgrad_return = (None,)

        if ctx.requires_wgrad and ctx.fuse_wgrad_accumulation:
            wgrad = None

        return (
            wgrad if ctx.requires_wgrad else None,
            *weight_cache_grad,
            dgrad_return,
            *bgrad_return,
        )


class Linear(TransformerEngineBaseLayer):
    """
    Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    weight_attr: Union[paddle.ParamAttr, None], default = None
                optional `paddle.ParamAttr` for weight.
    bias_attr: Union[paddle.ParamAttr, None, bool], default = None
              optional `paddle.ParamAttr` for bias.
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
        weight_attr: Union[paddle.ParamAttr, None] = None,
        bias_attr: Union[paddle.ParamAttr, None, bool] = None,
        parallel_mode: Optional[str] = None,
        sequence_parallel: bool = False,
        tp_group: Union[dist_group_type, None] = None,
        fuse_wgrad_accumulation: bool = False,
        backend: str = "transformer_engine",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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

        # Initialize weight parameter
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

        # Initialize bias parameter
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

        self.fp8_weight_shapes.append(self.weight.shape)

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.tensor_parallel and self.has_bias:
            self.gemm_bias_fused_add = False
        else:
            self.gemm_bias_fused_add = True

    def _te_forward(
        self,
        inp: paddle.Tensor,
        is_first_microbatch: Optional[bool] = None,
    ) -> paddle.Tensor:
        """
        Apply the linear transformation to the input.
        """
        with self.prepare_forward(inp, is_first_microbatch=is_first_microbatch) as inp:
            # Layer input should be casted outside PyLayer, as performing
            # inplace cast to input tensors may cause problems when used
            # together with Paddle native layers.
            inp = cast_if_needed(inp, self.activation_dtype)

            # Get persistent fp8 weight buffer. None if buffer does not exist.
            weight_fp8, weight_t_fp8 = self.get_fp8_weights_scratchpad(is_first_microbatch)

            out = _Linear.apply(
                self.weight,
                weight_fp8,
                weight_t_fp8,
                inp,
                self.bias if self.gemm_bias_fused_add else None,
                self.has_bias and self.gemm_bias_fused_add,
                self.fp8_enabled,
                self.fp8_calibration,
                self.fp8_meta,
                self.activation_dtype,
                paddle.is_grad_enabled(),
                self.parallel_mode,
                self.tensor_parallel,
                self.sequence_parallel,
                self.tp_group,
                self.tp_size,
                self.fuse_wgrad_accumulation,
                is_first_microbatch,
            )

        if not self.gemm_bias_fused_add:
            out = out + cast_if_needed_inplace(self.bias, self.activation_dtype)

        return out

    def _pd_forward(
        self,
        inp: paddle.Tensor,
        is_first_microbatch: Optional[bool] = None,
    ) -> paddle.Tensor:
        """Calls Paddle OP"""
        if is_first_microbatch is not None:
            warnings.warn(
                "`is_first_microbatch` is not supported for paddle backend and is ignored."
            )
        if self.parallel_mode == "column" and self.tensor_parallel:
            inp = identity(inp, self.tp_group)
        out = F.linear(inp, self.weight, self.bias if self.gemm_bias_fused_add else None)
        if self.parallel_mode == "row" and self.tensor_parallel:
            out, _ = allreduce(out, self.tp_group)
            out = out + self.bias if self.bias is not None else out
        return out

    def forward(self, *args, **kwargs):
        """
        Apply the linear transformation to the input.

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
