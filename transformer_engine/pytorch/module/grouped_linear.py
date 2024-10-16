# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinear API"""
from typing import Union, Optional, Callable, Tuple, List, Dict, Any

import torch

import transformer_engine_torch as tex

from .base import (
    get_multi_stream_cublas_workspace,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from ..utils import (
    divide,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
    init_method_constant,
    requires_grad,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
)
from ..cpp_extensions import (
    cast_to_fp8,
    fp8_cast_transpose_bgrad_fused,
    fp8_multi_cast_transpose_fused,
    fp8_grouped_gemm,
    grouped_gemm,
)
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..tensor import Float8Tensor, QuantizedTensor
from ..export import is_in_onnx_export_mode
from ..cpu_offload import is_cpu_offload_enabled

__all__ = ["GroupedLinear"]


class _GroupedLinear(torch.autograd.Function):
    """GroupedLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        sequence_parallel: bool,
        activation_dtype: torch.dtype,
        fp8_meta_offsets: Dict[str, int],
        is_grad_enabled: bool,
        weights_fp8: List[Union[Float8Tensor, None]],
        *weights_and_biases: Union[Float8Tensor, torch.Tensor, None],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        num_gemms = len(m_splits)
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]

        # Make sure input dimensions are compatible
        in_features = weights[0].shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmats = torch.split(inp.view(-1, in_features), m_splits)
        if fp8:
            for i in range(num_gemms):
                assert_dim_for_fp8_exec(inputmats[i])
                assert_dim_for_fp8_exec(weights[i])

        # Cast input to expected dtype
        inputmats_no_fp8 = [cast_if_needed(mat, activation_dtype) for mat in inputmats]
        inputmats = []
        inputmats_t = []
        inputmat_scale_inv = None

        if fp8:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            inputmat_scale_inv = torch.empty([num_gemms], dtype=torch.float32, device=inp.device)
            if (
                not fp8_meta["recipe"].override_linear_precision.wgrad
                and is_grad_enabled
                and weights[0].requires_grad
                and not sequence_parallel
            ):
                # FP8 input for forward, FP8 input transpose for backward wgrad
                indices = list(
                    range(fp8_meta_offsets["input"], fp8_meta_offsets["input"] + num_gemms)
                )
                inputmats, inputmats_t = fp8_multi_cast_transpose_fused(
                    inputmats_no_fp8,
                    fp8_meta["scaling_fwd"],
                    indices,  # scale_indices
                    indices,  # amax_indices
                    indices,  # scale_inv_indices
                    fp8_dtype_forward,
                    scale_inv=inputmat_scale_inv,
                )
            else:
                # FP8 input for forward
                inputmats = [
                    cast_to_fp8(
                        inputmats_no_fp8[i],
                        fp8_meta["scaling_fwd"],
                        fp8_meta_offsets["input"] + i,
                        fp8_dtype_forward,
                        scale_inv=inputmat_scale_inv,
                    )
                    for i in range(num_gemms)
                ]

            # Hack for ONNX export
            # Note: ONNX models are represented as a graph of tensor
            # operations, so the in-place scale-inv update doesn't fit
            # very well. We work around this by making it look like
            # the scale-inv tensor is initialized with a copy.
            # Note: ONNX export expects FP8 scales can be represented
            # with constant ops. However, copying into a buffer
            # involves an expand op for array broadcasting. We work
            # around this by filling the buffer instead.
            if is_in_onnx_export_mode():
                inputmat_scale_inv.fill_(inputmat_scale_inv.item())
        else:
            inputmats = inputmats_no_fp8

        if fp8:
            bias_dtype = torch.bfloat16 if activation_dtype == torch.float32 else activation_dtype
            biases = [cast_if_needed(bias, bias_dtype) for bias in biases] if use_bias else biases

            # Use FP8 weights
            if weights_fp8[0] is None:
                weights_fp8 = weights
            assert all(isinstance(w, Float8Tensor) for w in weights_fp8)

            out = torch.empty(
                [sum(m_splits), weights_fp8[0].size(0)],
                dtype=activation_dtype,
                device=inputmats[0].device,
            )

            _ = fp8_grouped_gemm(
                [w._data for w in weights_fp8],
                [w._scale_inv for w in weights_fp8],
                0,  # weight offset is 0 for the newly created _scale_inv
                fp8_dtype_forward,
                inputmats,
                inputmat_scale_inv,
                0,
                fp8_dtype_forward,
                [out],
                activation_dtype,
                get_multi_stream_cublas_workspace(),
                m_splits=m_splits,
                bias=biases,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
            )
        else:
            # Cast for native AMP
            weights = [cast_if_needed(w, activation_dtype) for w in weights]
            biases = (
                [cast_if_needed(bias, activation_dtype) for bias in biases] if use_bias else biases
            )

            if fp8_calibration:
                for i in range(num_gemms):
                    # amax of input
                    amin, amax = inputmats[i].aminmax()
                    fp8_meta["scaling_fwd"].amax_history[0][fp8_meta_offsets["input"] + i] = (
                        torch.max(-amin, amax).float()
                    )
                    # amax of weight
                    amin, amax = weights[i].aminmax()
                    fp8_meta["scaling_fwd"].amax_history[0][fp8_meta_offsets["weight"] + i] = (
                        torch.max(-amin, amax).float()
                    )

            out = torch.empty(
                [sum(m_splits), weights[0].size(0)],
                dtype=activation_dtype,
                device=inputmats[0].device,
            )

            _ = grouped_gemm(
                weights,
                inputmats,
                torch.split(out, m_splits),
                activation_dtype,
                get_multi_stream_cublas_workspace(),
                bias=biases,
                use_bias=use_bias,
            )

        if is_grad_enabled:
            saved_inputmats = [None] * num_gemms
            saved_inputmats_t = [None] * num_gemms
            if weights[0].requires_grad:
                if fp8 and not fp8_meta["recipe"].override_linear_precision.wgrad:
                    if not inputmats_t:
                        saved_inputmats = inputmats
                    else:
                        saved_inputmats_t = inputmats_t
                        if cpu_offloading:
                            for t in saved_inputmats_t:
                                t.activation_offloading = True
                else:
                    saved_inputmats = inputmats_no_fp8

                if cpu_offloading:
                    if fp8:
                        for w in weights_fp8:
                            if w is not None:
                                w.weight_offloading = True
                    for w in weights:
                        w.weight_offloading = True
                    for t in saved_inputmats:
                        if t is not None:
                            t.activation_offloading = True

            ctx.save_for_backward(
                inputmat_scale_inv,
                *saved_inputmats,
                *saved_inputmats_t,
                *weights,
                *weights_fp8,
                *[
                    w.main_grad if cpu_offloading and fuse_wgrad_accumulation else None
                    for w in weights
                ],
            )
            ctx.m_splits = m_splits
            ctx.num_gemms = num_gemms
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.inp_shape = inp.shape
            ctx.fp8_meta_offsets = fp8_meta_offsets
            ctx.requires_dgrad = inp.requires_grad
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, weights[0], biases[0]):
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    ctx.reduce_and_update_bwd_fp8_tensors
                    or FP8GlobalStateManager.is_first_fp8_module()
                )

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        with torch.cuda.nvtx.range("_GroupedLinear_backward"):
            (
                inputmat_scale_inv,
                *saved_tensors,
            ) = ctx.saved_tensors
            inputmats = saved_tensors[: ctx.num_gemms]
            inputmats_t = saved_tensors[ctx.num_gemms : 2 * ctx.num_gemms]
            weights = saved_tensors[2 * ctx.num_gemms : 3 * ctx.num_gemms]
            weights_fp8 = saved_tensors[3 * ctx.num_gemms : 4 * ctx.num_gemms]
            main_grads = saved_tensors[4 * ctx.num_gemms :]
            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                for i in ctx.num_gemms:
                    w = torch.nn.Parameter(weights[i], weights[i].requires_grad)
                    w.main_grad = main_grads[i]
                    weights[i] = w

            # preprocess grad_output
            grad_output = grad_output.contiguous()
            grad_output_mats = torch.split(
                grad_output.view(-1, grad_output.shape[-1]), ctx.m_splits
            )
            grad_output_c = [None] * ctx.num_gemms
            grad_output_t = [None] * ctx.num_gemms
            grad_biases = [None] * ctx.num_gemms
            if ctx.fp8:
                fp8_dtype_forward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=True)
                fp8_dtype_backward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)
                if ctx.use_bias:
                    for i in range(ctx.num_gemms):
                        grad_biases[i], grad_output_c[i], grad_output_t[i] = (
                            fp8_cast_transpose_bgrad_fused(
                                grad_output_mats[i],
                                ctx.fp8_meta["scaling_bwd"],
                                ctx.fp8_meta_offsets["grad_output"] + i,
                                fp8_dtype_backward,
                            )
                        )
                else:
                    if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                        indices = list(
                            range(
                                ctx.fp8_meta_offsets["grad_output"],
                                ctx.fp8_meta_offsets["grad_output"] + ctx.num_gemms,
                            )
                        )
                        grad_output_c, grad_output_t = fp8_multi_cast_transpose_fused(
                            grad_output_mats,
                            ctx.fp8_meta["scaling_bwd"],
                            indices,  # scale_indices
                            indices,  # amax_indices
                            indices,  # scale_inv_indices
                            fp8_dtype_backward,
                        )
                    else:
                        for i in range(ctx.num_gemms):
                            grad_output_c[i] = cast_to_fp8(
                                grad_output_mats[i],
                                ctx.fp8_meta["scaling_bwd"],
                                ctx.fp8_meta_offsets["grad_output"] + i,
                                fp8_dtype_backward,
                            )

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            if ctx.requires_dgrad:
                if ctx.fp8:
                    dgrad = torch.empty(
                        (sum(ctx.m_splits), weights_fp8[0].size(1)),
                        dtype=ctx.activation_dtype,
                        device=grad_output.device,
                    )
                    fp8_grouped_gemm(
                        [w.transpose_2d() for w in weights_fp8],
                        [w._scale_inv for w in weights_fp8],
                        0,  # weight offset is 0 for the newly created _scale_inv
                        weights_fp8[0]._fp8_dtype,
                        grad_output_c,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        ctx.fp8_meta_offsets["grad_output"],
                        fp8_dtype_backward,
                        [dgrad],
                        ctx.activation_dtype,
                        get_multi_stream_cublas_workspace(),
                        m_splits=ctx.m_splits,
                        use_split_accumulator=_2X_ACC_DGRAD,
                    )
                else:
                    dgrad = torch.empty(
                        (sum(ctx.m_splits), weights[0].size(1)),
                        dtype=ctx.activation_dtype,
                        device=grad_output.device,
                    )
                    grouped_gemm(
                        weights,
                        grad_output_mats,
                        torch.split(dgrad, ctx.m_splits),
                        ctx.activation_dtype,
                        get_multi_stream_cublas_workspace(),
                        layout="NN",
                        grad=True,
                    )

            if weights[0].requires_grad:
                if ctx.fuse_wgrad_accumulation:
                    wgrad_list = [w.main_grad for w in weights]
                else:
                    wgrad_list = [
                        torch.empty(w.size(), dtype=ctx.activation_dtype, device=w.device)
                        for w in weights
                    ]
                if ctx.fp8:
                    # WGRAD
                    if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                        if inputmats_t[0] is None:
                            for i in range(ctx.num_gemms):
                                if isinstance(inputmats[i], Float8Tensor):
                                    inputmats_t[i] = inputmats[i].transpose_2d()
                                else:
                                    inputmats_t[i] = tex.fp8_transpose(
                                        inputmats[i], fp8_dtype_backward
                                    )
                        fp8_grouped_gemm(
                            [
                                inp._data if isinstance(inp, Float8Tensor) else inp
                                for inp in inputmats_t
                            ],
                            [inputmat_scale_inv],
                            0,
                            fp8_dtype_forward,
                            grad_output_t,
                            ctx.fp8_meta["scaling_bwd"].scale_inv,
                            ctx.fp8_meta_offsets["grad_output"],
                            fp8_dtype_backward,
                            wgrad_list,
                            ctx.activation_dtype,
                            get_multi_stream_cublas_workspace(),
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            use_split_accumulator=_2X_ACC_WGRAD,
                        )
                    else:
                        grouped_gemm(
                            inputmats,
                            grad_output_mats,
                            wgrad_list,
                            ctx.activation_dtype,
                            get_multi_stream_cublas_workspace(),
                            layout="NT",
                            grad=True,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                        )
                else:
                    # WGRAD
                    _, grad_biases, _ = grouped_gemm(
                        inputmats,
                        grad_output_mats,
                        wgrad_list,
                        ctx.activation_dtype,
                        get_multi_stream_cublas_workspace(),
                        layout="NT",
                        grad=True,
                        use_bias=ctx.use_bias,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                    )

                # Deallocate input tensor
                clear_tensor_data(*inputmats)
                clear_tensor_data(*inputmats_t)

            if not ctx.use_bias:
                grad_biases = [None] * ctx.num_gemms

        def handle_custom_ddp_from_mcore(w, wgrad):
            if w.requires_grad:
                if ctx.fuse_wgrad_accumulation and hasattr(w, "grad_added_to_main_grad"):
                    w.grad_added_to_main_grad = True
                    if getattr(w, "zero_out_wgrad", False):
                        wgrad = torch.zeros(
                            w.main_grad.shape,
                            dtype=w.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    else:
                        wgrad = torch.empty(
                            w.main_grad.shape,
                            dtype=w.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                elif ctx.fuse_wgrad_accumulation:
                    wgrad = None
            else:
                wgrad = None
            return wgrad

        wgrad_list = [
            handle_custom_ddp_from_mcore(w, wgrad) for w, wgrad in zip(weights, wgrad_list)
        ]

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            None,  # m_splits
            None,  # use_bias
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # fp8_meta
            None,  # fuse_wgrad_accumulation
            None,  # cpu_offloading
            None,  # sequence_parallel
            None,  # activation_dtype
            None,  # fp8_meta_offsets
            None,  # is_grad_enabled
            None,  # weights_fp8
            *wgrad_list,
            *grad_biases,
        )


class GroupedLinear(TransformerEngineBaseModule):
    """Applies linear transformations to the incoming data list
       :math:`y_i = x_iA_i^T + b_i` in a grouped way.

    Parameters
    ----------
    num_gemms : int
                number of GEMMs to be performed simutaneously.
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    get_rng_state_tracker : Callable, default = `None`
                 used to get the random number generator state tracker for initializing weights.
    rng_tracker_name : str, default = `None`
                 the param passed to get_rng_state_tracker to get the specific rng tracker.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

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
        num_gemms: int,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_rs: bool = False,
        ub_overlap_ag: bool = False,
        ub_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.ub_overlap_rs = ub_overlap_rs
        self.ub_overlap_ag = ub_overlap_ag
        self.ub_name = ub_name
        assert (
            not ub_overlap_rs and not ub_overlap_ag
        ), "GroupedLinear doesn't support Userbuffer overlap."
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

        self._offsets = {"input": 0, "weight": num_gemms, "output": 2 * num_gemms, "grad_output": 0}

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

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        for i in range(self.num_gemms):
            # Construct weight parameter
            self.register_parameter(
                f"weight{i}",
                torch.nn.Parameter(
                    torch.empty(
                        self.out_features,
                        self.in_features,
                        device=device,
                        dtype=params_dtype,
                    ),
                ),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=self._offsets["weight"] + i,
            )

            # Construct bias parameters if needed
            if self.use_bias:
                self.register_parameter(
                    f"bias{i}",
                    torch.nn.Parameter(
                        torch.empty(
                            self.out_features,
                            device=device,
                            dtype=params_dtype,
                        ),
                    ),
                    init_fn=init_method_constant(0.0),
                )
            else:
                bias = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, f"bias{i}", bias)

        if self.primary_weights_in_fp8:
            self.init_fp8_metadata(num_gemms=self.num_gemms)

        self.reset_parameters(defer_init=device == "meta")

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for linear weights
            for i in range(self.num_gemms):
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, f"weight{i}"),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                for i in range(self.num_gemms):
                    if self.parallel_mode == "row":
                        setattr(
                            getattr(self, f"bias{i}"),
                            "sequence_parallel",
                            self.sequence_parallel,
                        )
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, f"bias{i}"), True, 0, 1)

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        m_splits : List[int]
                 List of integers representing the split of the input tensor.
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
        assert not isinstance(
            inp, Float8Tensor
        ), "GroupedLinear doesn't support input tensor in FP8."
        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."

        skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        with self.prepare_forward(inp, is_first_microbatch, num_gemms=self.num_gemms) as inp:

            weight_tensors = [self._fast_get_param(f"weight{i}") for i in range(self.num_gemms)]
            bias_tensors = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]
            if not self.fp8:
                weight_tensors = [
                    w.dequantize() if isinstance(w, QuantizedTensor) else w for w in weight_tensors
                ]

            weight_tensors_fp8 = [None] * self.num_gemms
            if self.fp8:
                for i in range(self.num_gemms):
                    if isinstance(weight_tensors[i], Float8Tensor):
                        # Make sure transpose cache is valid, if present
                        # Note: Transpose cache may have been invalidated
                        # externally, e.g. by optimizer.
                        if weight_tensors[i]._transpose is not None:
                            weight_tensors[i].transpose_2d(
                                fill_cache=True,
                                noop_flag=skip_fp8_weight_update,
                            )
                    else:
                        # FP8 cast to workspace buffer
                        update_workspace = is_first_microbatch is None or is_first_microbatch
                        weight_tensors_fp8[i] = self.get_fp8_workspace(
                            tensor=weight_tensors[i],
                            fp8_meta_forward=True,
                            fp8_meta_index=self._offsets["weight"] + i,
                            cache_name=(None if is_first_microbatch is None else f"weight{i}"),
                            update_workspace=update_workspace,
                            skip_update_flag=skip_fp8_weight_update,
                        )

            if torch.is_grad_enabled():
                linear_fn = _GroupedLinear.apply
                args = []
            else:
                linear_fn = _GroupedLinear.forward
                args = [None]
            args += (
                inp,
                m_splits,
                self.apply_bias and not self.gemm_bias_unfused_add,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                is_cpu_offload_enabled(),
                self.sequence_parallel,
                self.activation_dtype,
                self._offsets,
                torch.is_grad_enabled(),
                weight_tensors_fp8,
                *weight_tensors,
                *bias_tensors,
            )
            out = linear_fn(*args)

        if self.gemm_bias_unfused_add:
            out_shape = out.shape
            out = torch.cat(
                [
                    o + cast_if_needed(b, self.activation_dtype)
                    for o, b in zip(
                        torch.split(out.view(-1, self.out_features), m_splits), bias_tensors
                    )
                ]
            ).view(out_shape)

        if self.return_bias:
            return out, [cast_if_needed(b, self.activation_dtype) for b in bias_tensors]
        return out
