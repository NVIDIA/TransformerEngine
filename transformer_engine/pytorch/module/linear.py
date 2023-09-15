# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import warnings
from typing import Union, Optional, Callable, Tuple, List, Dict, Any

import torch
from torch.nn.parameter import Parameter

import transformer_engine_extensions as tex

from .base import (
    get_workspace,
    _prepare_backward,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..fp8 import get_fp8_te_dtype
from ..utils import (
    divide,
    get_default_init_method,
    cast_if_needed,
    assert_dim_for_fp8_exec,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    initialize_affine_weight_gpu,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    gather_along_last_dim,
)
from ..cpp_extensions import (
    fp8_gemm,
    gemm,
    fp8_cast_transpose_fused,
    cast_to_fp8,
)
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo


__all__ = ["Linear"]


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        weight_fp8: Union[torch.Tensor, None],
        weight_t_fp8: Union[torch.Tensor, None],
        inp: torch.Tensor,
        bias: torch.Tensor,
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        is_grad_enabled: bool,
        ub_split_rs: bool,
        ub_split_ag: bool,
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        in_features = weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(weight)

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        if ub_split_rs:
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1:
                ub_split_rs = False
        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        inputmat_no_fp8 = inputmat

        if fp8:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

            if not fp8_meta["recipe"].override_linear_precision.wgrad:
                if is_grad_enabled:
                    inputmat, inputmat_t = fp8_cast_transpose_fused(
                        inputmat,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                    )
                else:
                    inputmat = cast_to_fp8(
                        inputmat,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                    )
            else:
                inputmat, inputmat_t = cast_to_fp8(
                    inputmat,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                ), None

        # Column Parallel Linear
        if parallel_mode == "column" and sequence_parallel:
            inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)
        else:
            inputmat_total = inputmat

        if fp8:
            bias_dtype = (
                torch.bfloat16
                if activation_dtype == torch.float32
                else activation_dtype
            )
            bias = cast_if_needed(bias, bias_dtype) if use_bias else bias

            if update_fp8_weights:
                if is_grad_enabled:
                    fp8_cast_transpose_fused(
                        weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        cast_out=weight_fp8,
                        transpose_out=weight_t_fp8,
                    )
                else:
                    weight_t_fp8 = None
                    weight_fp8 = cast_to_fp8(
                        weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                    )

            if ub_split_rs:
                ub_obj_projout = get_ub("proj_fprop")
                out = ub_obj_projout.get_ubuf_output(1)
                dim_size = list(inputmat_total.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = weight.size(0)
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)
            else:
                dim_size = list(inputmat_total.size())
                dim_size[1] = weight.size(0)
                out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)

            _ = fp8_gemm(
                weight_fp8,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                inputmat_total,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
                out=out,
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS if ub_split_rs else None,
                ub=ub_obj_projout if ub_split_rs else None,
                extra_output_tensor=rs_out if ub_split_rs else None,
            )
        else:
            # Cast for native AMP
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

            if fp8_calibration:
                # amax of input
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_INPUT] = \
                    torch.amax(inputmat_total).float()
                # amax of weight
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_WEIGHT] = \
                    torch.amax(weight).float()

            if ub_split_rs:
                ub_obj_projout = get_ub("proj_fprop")
                out = ub_obj_projout.get_ubuf_output(1)
                dim_size = list(inputmat_total.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = weight.size(0)
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)
            else:
                dim_size = list(inputmat_total.size())
                dim_size[1] = weight.size(0)
                out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)

            _, _, _ = gemm(
                weight,
                inputmat_total,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                out=out,
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS if ub_split_rs else None,
                ub=ub_obj_projout if ub_split_rs else None,
                extra_output_tensor=rs_out if ub_split_rs else None,
            )

        if is_grad_enabled:
            fp8_wgrad = fp8 and not fp8_meta["recipe"].override_linear_precision.wgrad
            ctx.save_for_backward(
                inputmat_no_fp8 if weight.requires_grad and not fp8_wgrad else None,
                inputmat_t if weight.requires_grad and fp8_wgrad else None,
                weight,
                weight_t_fp8 if fp8 else None,
                fp8_meta["scaling_fwd"].scale_inv.clone() if fp8 else None,
            )
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.ub_split_ag = ub_split_ag
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad

        # Row Parallel Linear
        if ub_split_rs:
            out = rs_out
        elif parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])


    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8, ctx.fp8_meta, ctx.tp_group, ctx.tp_size, name="_Linear"
        ):
            (
                inputmat,
                inputmat_t,
                weight,
                weight_t_fp8,
                fwd_scale_inverses,
            ) = ctx.saved_tensors

            if ctx.ub_split_ag:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_split_ag = False
            if ctx.ub_split_ag:
                dim_size = list(grad_output.size())
                dim_size[0] = dim_size[0] * tp_world_size
                ctx.ub_obj_gradout = get_ub("proj_dgrad")
            (
                grad_output,
                grad_output_c,
                grad_output_t,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_output, ctx.parallel_mode == "row"
            )

            # Column Parallel Linear
            # Overlap input AG with dgrad
            if weight.requires_grad and ctx.parallel_mode == "column" and ctx.sequence_parallel:
                if ctx.fp8 and not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                    inputmat_t_total, handle = gather_along_last_dim(
                        inputmat_t, ctx.tp_group, async_op=ctx.requires_dgrad
                    )
                else:
                    inputmat_total, handle = gather_along_first_dim(
                        inputmat, ctx.tp_group, async_op=ctx.requires_dgrad
                    )
            else:
                inputmat_t_total = inputmat_t
                inputmat_total = inputmat
                handle = None

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            if ctx.fp8:
                fp8_dtype_forward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=True
                )
                fp8_dtype_backward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=False
                )

            if ctx.requires_dgrad:
                if ctx.fp8:
                    dgrad = fp8_gemm(
                        weight_t_fp8,
                        fwd_scale_inverses,
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        grad_output_c,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        fp8_dtype_backward,
                        ctx.activation_dtype,
                        get_workspace(),
                        use_split_accumulator=_2X_ACC_DGRAD,
                        ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ctx.ub_split_ag else None,
                        ub=ctx.ub_obj_gradout if ctx.ub_split_ag else None,
                    )
                else:
                    dgrad, _, _ = gemm(
                        weight,
                        grad_output,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NN",
                        grad=True,
                        ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ctx.ub_split_ag else None,
                        ub=ctx.ub_obj_gradout if ctx.ub_split_ag else None,
                    )

                # Overlap dgrad-RS/AR with wgrad
                if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                    if handle is not None:
                        handle.wait()
                    dgrad, handle = reduce_scatter_along_first_dim(
                        dgrad, ctx.tp_group, async_op=True
                    )
                elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
                    dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

            if weight.requires_grad:
                if ctx.fp8:
                    # WGRAD
                    if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                        if ctx.ub_split_ag:
                            grad_output_t = tex.fp8_transpose(grad_output_c, fp8_dtype_backward)
                        wgrad = fp8_gemm(
                            inputmat_t_total,
                            fwd_scale_inverses,
                            tex.FP8FwdTensors.GEMM1_INPUT,
                            fp8_dtype_forward,
                            grad_output_t,
                            ctx.fp8_meta["scaling_bwd"].scale_inv,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            fp8_dtype_backward,
                            ctx.activation_dtype,
                            get_workspace(),
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                            use_split_accumulator=_2X_ACC_WGRAD,
                        )
                    else:
                        wgrad, _, _ = gemm(
                            inputmat_total,
                            grad_output,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                        )
                else:
                    # WGRAD
                    wgrad, grad_bias, _ = gemm(
                        inputmat_total,
                        grad_output,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NT",
                        grad=True,
                        use_bias=ctx.use_bias,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    )

            # Column Parallel Linear
            if ctx.parallel_mode == "column" and ctx.tensor_parallel and handle is not None:
                handle.wait()

            if not ctx.use_bias:
                grad_bias = None

        return (
            wgrad if weight.requires_grad else None,
            None,
            None,
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Linear(TransformerEngineBaseModule):
    """
    Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    On NVIDIA GPUs it is a drop-in replacement for `torch.nn.Linear`.

    .. warning::

        Argument :attr:`skip_weight_param_allocation` is deprecated and will
        be fully removed in future releases.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      if a tuple of strings or a dict of strings to integers is provided,
                      the weight and bias parameters of the module are exposed as `N` separate
                      `torch.nn.parameter.Parameter`s each, split along the first dimension,
                      where `N` is the length of the argument and the strings contained are the
                      names of the split parameters. In the case of a tuple, each parameter
                      has the same shape. In the case of a dict, the values give the
                      `out_features` for each projection.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.
    parallel_mode : {None, 'Column', 'Row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        skip_weight_param_allocation: bool = False,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        ub_split_rs: bool = False,
        ub_split_ag: bool = False,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        if skip_weight_param_allocation:
            warnings.warn(
                "Argument `skip_weight_param_allocation` is deprecated and"
                "will be fully removed in future releases. It has ignored"
                "starting from v0.11.",
                category=DeprecationWarning,
            )

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.parameters_split = parameters_split
        self.ub_split_rs = ub_split_rs
        self.ub_split_ag = ub_split_ag

        if ub_split_rs or ub_split_ag:
            assert (
                tex.userbuf_comm_available()
            ), "Userbuffer communication backend not available."

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in GemmParallelModes
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        if init_method is None:
            init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        self.weight_tensor = torch.empty(
            self.out_features, self.in_features,
            device=device, dtype=params_dtype)

        initialize_affine_weight_gpu(
            self.weight_tensor,
            init_method,
            get_rng_state_tracker,
            partition_dim=1 if self.parallel_mode == "row" else 0,
            stride=1,
        )

        if self.use_bias:
            self.bias_tensor = torch.empty(self.out_features, device=device, dtype=params_dtype)
        else:
            self.bias_tensor = torch.Tensor().to(dtype=params_dtype, device=device)

        with torch.no_grad():
            self.bias_tensor.zero_()

        if parameters_split is None:
            parameters_split = {"": self.out_features}
        elif isinstance(parameters_split, tuple):
            assert (
                self.out_features % len(parameters_split) == 0
            ), f"Weight and bias params cannot be split into {len(parameters_split)} parts"
            split_size = self.out_features // len(parameters_split)
            parameters_split = {key: split_size for key in parameters_split}
        elif isinstance(parameters_split, dict):
            overall_split_size = sum(parameters_split.values())
            assert(
                self.out_features == overall_split_size
            ), f"Overall sum of parameters_split (={overall_split_size}) does not match "\
               f"to out features (={self.out_features})"
        else:
            assert False, "Type of 'parameters_split' is not None, tuple or dict"
        self.updated_parameters_split = parameters_split

        self.weight_names = []
        self.bias_names = []

        slice_begin = 0
        for pname, slice_size in parameters_split.items():
            wname = pname + "weight"
            bname = pname + "bias"

            slice_end = slice_begin + slice_size

            self.register_parameter(
                wname, Parameter(self.weight_tensor[slice_begin:slice_end])
            )

            set_tensor_model_parallel_attributes(
                tensor=getattr(self, wname),
                is_parallel=True,
                dim=1 if parallel_mode == "row" else 0,
                stride=1,
            )

            if self.use_bias:
                self.register_parameter(
                    bname, Parameter(self.bias_tensor[slice_begin:slice_end])
                )
            else:
                setattr(self, bname, torch.Tensor().to(dtype=params_dtype, device=device))

            if parallel_mode == "column":
                set_tensor_model_parallel_attributes(getattr(self, bname), True, 0, 1)

            self.weight_names.append(wname)
            self.bias_names.append(bname)

            slice_begin = slice_end

        self.fp8_weight_shapes.append(torch.Size((self.out_features, self.in_features)))

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """
        Fetch the fp8 weight tensor placeholders if they exist (when
        `is_first_microbatch` is not `None`) or return empty fp8 weight
        tensors (if `is_first_microbatch is None`)
        """
        if not self.fp8:
            return [None, None]

        if is_first_microbatch is None:
            # Return empty weight placeholders for each fwd/bwd pass
            fp8_weight_tensors = self.get_fp8_weights_empty_tensors(
                is_first_microbatch
            )
        else:
            # These persistent weight placeholders should've been created in
            # `set_fp8_weights` method
            fp8_weight_tensors = [self.weight1_fp8, self.weight1_t_fp8]

        return fp8_weight_tensors

    @no_torch_dynamo
    def forward(
        self,
        inp: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        .. warning::

            Arguments :attr:`weight` and :attr:`bias` are deprecated and will
            be fully removed in future releases.

        Parameters
        ----------
        inp : torch.Tensor
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
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """

        if weight is not None or bias is not None:
            raise RuntimeError(
                "Arguments `weight` and `bias` are deprecated and "
                "will be fully removed in future releases."
            )

        with self.prepare_forward(inp, is_first_microbatch) as inp:
            bias_tensor = (
                self.bias if self.parameters_split is None
                else self.bias_tensor if not torch.is_grad_enabled()
                else self.noop_cat("bias_tensor", self.bias_names,
                    self.updated_parameters_split)
            )
            weight_tensor = (
                self.weight if self.parameters_split is None
                else self.weight_tensor if not torch.is_grad_enabled()
                else self.noop_cat("weight_tensor", self.weight_names,
                    self.updated_parameters_split)
            )

            # Fetch the fp8 weights placeholders (for linear/gemm)
            weight1_fp8, weight1_t_fp8 = self.get_fp8_weights_scratchpad(
                is_first_microbatch
            )

            if torch.is_grad_enabled():
                linear_fn = _Linear.apply
                args = []
            else:
                linear_fn = _Linear.forward
                args = [None]
            args += (
                weight_tensor,
                weight1_fp8,
                weight1_t_fp8,
                inp,
                bias_tensor,
                self.apply_bias and not self.gemm_bias_unfused_add,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                torch.is_grad_enabled(),
                self.ub_split_rs,
                self.ub_split_ag,
            )
            out = linear_fn(*args)

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out
