# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinear API"""
from typing import Union, Optional, Callable, Tuple, List

import torch

import transformer_engine_torch as tex

from .base import (
    get_multi_stream_cublas_workspace,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..fp8 import FP8GlobalStateManager
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
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from ..cpp_extensions import (
    general_grouped_gemm,
)
from ..constants import GemmParallelModes, dist_group_type, TE_DType
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..tensor.float8_tensor import Float8Tensor
from ..cpu_offload import is_cpu_offload_enabled

from ..tensor.quantized_tensor import (
    QuantizedTensor,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)


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
        input_quantizers: List[Quantizer],
        weight_quantizers: List[Quantizer],
        output_quantizers: List[Quantizer],
        grad_output_quantizers: List[Quantizer],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        sequence_parallel: bool,
        activation_dtype: torch.dtype,
        is_grad_enabled: bool,
        module,
        skip_fp8_weight_update,
        *weights_and_biases,
    ) -> torch.Tensor:

        # pylint: disable=missing-function-docstring
        num_gemms = len(m_splits)
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]
        device = inp.device

        # TODO Support MXFP8  # pylint: disable=fixme
        if fp8 and FP8GlobalStateManager.get_fp8_recipe().mxfp8():
            raise NotImplementedError("GroupedLinear does not yet support MXFP8")

        # Make sure input dimensions are compatible
        in_features = weights[0].shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmats = torch.split(inp.view(-1, in_features), m_splits)
        if fp8:
            assert_dim_for_fp8_exec(*inputmats, *weights)

        # Cast input to expected dtype
        inputmats_no_fp8 = [cast_if_needed(mat, activation_dtype) for mat in inputmats]
        inputmats = []

        weight_requires_grad = weights[0].requires_grad

        if input_quantizers[0] is not None:
            for input_quantizer in input_quantizers:
                input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=(is_grad_enabled and weight_requires_grad),
                )
            columnwise_usage = is_grad_enabled and inp.requires_grad
            if not columnwise_usage:
                columnwise_usage = (
                    is_fp8_activation_recompute_enabled()
                    and not in_fp8_activation_recompute_phase()
                )
            if weight_quantizers[0] is not None:
                for weight_quantizer in weight_quantizers:
                    weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        if output_quantizers[0] is not None:
            for output_quantizer in output_quantizers:
                output_quantizer.set_usage(rowwise=True, columnwise=False)

        if fp8:
            inputmats = tex.fused_multi_quantize(
                inputmats_no_fp8, None, input_quantizers, TE_DType[activation_dtype]
            )
            weights_fp8 = []
            bias_dtype = torch.bfloat16 if activation_dtype == torch.float32 else activation_dtype
            if not isinstance(weights[0], QuantizedTensor):
                # FP8 cast to workspace buffer
                update_workspace = is_first_microbatch is None or is_first_microbatch
                for i in range(num_gemms):
                    weight_fp8 = module.get_weight_workspace(
                        tensor=weights[i],
                        quantizer=weight_quantizers[i],
                        cache_name=(None if is_first_microbatch is None else f"weight{i}"),
                        update_workspace=update_workspace,
                        skip_update_flag=skip_fp8_weight_update,
                    )
                    weights_fp8.append(weight_fp8)
            else:
                weights_fp8 = weights

        else:
            inputmats = inputmats_no_fp8
            bias_dtype = activation_dtype
            weights_fp8 = [cast_if_needed(weight, activation_dtype) for weight in weights]

        biases = [cast_if_needed(bias, bias_dtype) for bias in biases] if use_bias else biases

        out = torch.empty(
            [sum(m_splits), weights_fp8[0].size(0)],
            dtype=activation_dtype,
            device=device,
        )

        _ = general_grouped_gemm(
            weights_fp8,
            inputmats,
            [out],
            activation_dtype,
            get_multi_stream_cublas_workspace(),
            single_output=True,
            m_splits=m_splits,
            bias=biases,
            use_bias=use_bias,
            use_split_accumulator=_2X_ACC_FPROP,
        )

        if fp8_calibration:
            for i in range(num_gemms):
                # amax of input
                for i in range(num_gemms):
                    input_quantizers[i].calibrate(inputmats[i])
                for i in range(num_gemms):
                    weight_quantizers[i].calibrate(weights[i])

        if is_grad_enabled:

            ctx.weights_shape_1 = weights[0].shape[1]

            tensors_to_save, tensor_objects = prepare_for_saving(*inputmats, *weights_fp8, *biases)
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.weights_requires_grad = weights[0].requires_grad
            if fuse_wgrad_accumulation and ctx.weights_requires_grad:
                ctx.main_grads = [weights[i].main_grad for i in range(num_gemms)]
            else:
                ctx.main_grads = [None] * num_gemms
            ctx.device = device
            ctx.grad_output_quantizers = grad_output_quantizers
            ctx.m_splits = m_splits
            ctx.num_gemms = num_gemms
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.inp_shape = inp.shape
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
            saved_tensors = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
            N = ctx.num_gemms
            inputmats = saved_tensors[:N]
            weights = saved_tensors[N : 2 * N]
            biases = saved_tensors[2 * N : 3 * N]
            main_grads = ctx.main_grads

            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:  # TOSO
                for i in ctx.num_gemms:
                    w = torch.nn.Parameter(weights[i], weights[i].requires_grad)
                    w.main_grad = main_grads[i]
                    weights[i] = w

            # preprocess grad_output

            grad_output = grad_output.contiguous()
            grad_output_mats = torch.split(
                grad_output.view(-1, grad_output.shape[-1]), ctx.m_splits
            )
            grad_output = [None] * ctx.num_gemms
            grad_biases = [None] * ctx.num_gemms
            if ctx.fp8:
                if ctx.use_bias:
                    for i in range(ctx.num_gemms):
                        grad_biases[i], grad_output[i] = tex.bgrad_quantize(
                            grad_output_mats[i], ctx.grad_output_quantizers[i]
                        )
                else:
                    grad_output = tex.fused_multi_quantize(
                        grad_output_mats,
                        None,
                        ctx.grad_output_quantizers,
                        TE_DType[ctx.activation_dtype],
                    )
            else:
                grad_output = grad_output_mats

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            if ctx.requires_dgrad:
                dgrad = torch.empty(
                    (sum(ctx.m_splits), ctx.weights_shape_1),
                    dtype=ctx.activation_dtype,
                    device=ctx.device,
                )

                general_grouped_gemm(
                    weights,
                    grad_output,
                    [dgrad],
                    ctx.activation_dtype,
                    get_multi_stream_cublas_workspace(),
                    single_output=True,
                    layout="NN",
                    m_splits=ctx.m_splits,
                    grad=True,
                    use_split_accumulator=_2X_ACC_DGRAD,
                )

            if ctx.weights_requires_grad:
                if ctx.fuse_wgrad_accumulation:
                    wgrad_list = main_grads
                else:
                    wgrad_list = [
                        torch.empty(w.size(), dtype=ctx.activation_dtype, device=ctx.device)
                        for w in weights
                    ]
                # WGRAD
                _, grad_biases_, _ = general_grouped_gemm(
                    inputmats,
                    grad_output,
                    wgrad_list,
                    ctx.activation_dtype,
                    get_multi_stream_cublas_workspace(),
                    layout="NT",
                    grad=True,
                    m_splits=ctx.m_splits,
                    use_bias=ctx.use_bias if grad_biases[0] is None else None,
                    bias=biases,
                    use_split_accumulator=_2X_ACC_WGRAD,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                )
                for i in range(ctx.num_gemms):
                    if grad_biases[i] is None:
                        grad_biases[i] = grad_biases_[i]
                del grad_biases_

                # Deallocate input tensor
                clear_tensor_data(*inputmats)

                def handle_custom_ddp_from_mcore(w, wgrad):
                    if ctx.weights_requires_grad:
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
            else:
                wgrad_list = [None] * ctx.num_gemms

            if not ctx.use_bias:
                grad_biases = [None] * ctx.num_gemms

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
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
            None,  # is_grad_enabled
            None,  # is_grad_enabled
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

        with self.prepare_forward(inp, num_gemms=self.num_gemms) as inp:

            weight_tensors = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
            bias_tensors = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]
            if not self.fp8:
                weight_tensors = [
                    w.dequantize() if isinstance(w, QuantizedTensor) else w for w in weight_tensors
                ]

            input_quantizers, weight_quantizers, output_quantizers = (
                [None] * self.num_gemms,
                [None] * self.num_gemms,
                [None] * self.num_gemms,
            )
            grad_output_quantizers, _ = [None] * self.num_gemms, [None] * self.num_gemms
            if self.fp8:
                input_quantizers = [
                    self.quantizers["scaling_fwd"][self._offsets["input"] + i]
                    for i in range(self.num_gemms)
                ]
                for i in range(self.num_gemms):
                    input_quantizers[i].internal = True
                weight_quantizers = [
                    self.quantizers["scaling_fwd"][self._offsets["weight"] + i]
                    for i in range(self.num_gemms)
                ]
                for i in range(self.num_gemms):
                    weight_quantizers[i].internal = True
                if torch.is_grad_enabled():
                    grad_output_quantizers = [
                        self.quantizers["scaling_bwd"][self._offsets["input"] + i]
                        for i in range(self.num_gemms)
                    ]
                    for i in range(self.num_gemms):
                        grad_output_quantizers[i].internal = True

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
                input_quantizers,
                weight_quantizers,
                output_quantizers,
                grad_output_quantizers,
                self.fuse_wgrad_accumulation,
                is_cpu_offload_enabled(),
                self.sequence_parallel,
                self.activation_dtype,
                torch.is_grad_enabled(),
                self,
                skip_fp8_weight_update,
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
