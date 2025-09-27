# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinear API"""
from typing import Union, Optional, Callable, Tuple, List
import warnings
import contextlib

import functools
import torch

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from .base import (
    get_multi_stream_cublas_workspace,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ._common import WeightGradStore
from ..fp8 import FP8GlobalStateManager
from ..utils import (
    divide,
    cast_if_needed,
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
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..cpu_offload import is_cpu_offload_enabled

from ..tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer
from ..tensor.quantized_tensor import (
    QuantizedTensorBase,
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
        wgrad_store: WeightGradStore,
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
        save_original_input,
        *weights_and_biases,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        num_gemms = len(m_splits)
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]
        device = inp.device
        weight_requires_grad = weights[0].requires_grad

        # Configure quantizers
        if save_original_input and isinstance(input_quantizers[0], Float8Quantizer):
            raise ValueError("DelayedScaling recipe is not supported with save_original_input")
        if input_quantizers[0] is not None:
            for input_quantizer in input_quantizers:
                input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=(
                        is_grad_enabled and weight_requires_grad and not save_original_input
                    ),
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

        # Initialize input tensors
        in_features = weights[0].size(-1)
        if inp.size(-1) != in_features:
            raise ValueError(
                f"Input tensor (shape={tuple(inp.size())}) is not compatible with "
                f"weight tensor (shape={tuple(weights[0].size())})"
            )
        inp_view = inp.reshape(-1, in_features)
        inputmats: list
        if fp8:
            inputmats = tex.split_quantize(inp_view, m_splits, input_quantizers)
        else:
            inputmats = torch.split(cast_if_needed(inp_view, activation_dtype), m_splits)

        # Initialize weights
        weights_fp8: list
        if fp8:
            # FP8 cast to workspace buffer
            weights_fp8 = []
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
            weights_fp8 = [cast_if_needed(weight, activation_dtype) for weight in weights]

        # Initialize biases
        bias_dtype = activation_dtype
        if fp8 and activation_dtype == torch.float32:
            bias_dtype = torch.bfloat16  # FP8 GEMM only supports BF16/FP16 bias
        biases = [cast_if_needed(bias, bias_dtype) for bias in biases] if use_bias else biases

        # Initialize output tensor
        out = torch.empty(
            [sum(m_splits), weights_fp8[0].size(0)],
            dtype=activation_dtype,
            device=device,
        )

        # Choose whether to use split accumulator
        use_split_accumulator = _2X_ACC_FPROP
        if fp8:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if hasattr(recipe, "fp8_gemm_fprop"):
                use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

        # Perform GEMM
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
            use_split_accumulator=use_split_accumulator,
        )

        if fp8_calibration:
            for i in range(num_gemms):
                # amax of input
                for i in range(num_gemms):
                    input_quantizers[i].calibrate(inputmats[i])
                for i in range(num_gemms):
                    weight_quantizers[i].calibrate(weights[i])

        if is_grad_enabled:
            ctx.weight_quantizers = weight_quantizers
            ctx.weights_shape_1 = weights[0].shape[1]

            # TODO: update after #1638 is merged. # pylint: disable=fixme
            if weight_requires_grad:
                if save_original_input:
                    inputmats = [None] * num_gemms
                    inputmats[0] = inp
                else:
                    for inputmat in inputmats:
                        if isinstance(inputmat, QuantizedTensorBase):
                            inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)
            else:
                inputmats = [None] * num_gemms
            if inp.requires_grad:
                for weight in weights_fp8:
                    if isinstance(weight, QuantizedTensorBase):
                        weight.update_usage(columnwise_usage=True)

            tensors_to_save, tensor_objects = prepare_for_saving(
                *inputmats,
                *weights_fp8,
                *weights,
                *biases,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.weights_requires_grad = weights[0].requires_grad
            if fuse_wgrad_accumulation and ctx.weights_requires_grad:
                # This check is needed to ensure that main_grad is not created
                # during the forward pass when using MCore FSDP as it creates
                # the main_grad buffer lazily before backprop
                if hasattr(weights[0], "__fsdp_param__"):
                    # MCore FSDP creates main_grad lazily before backward
                    ctx.main_grad_funcs = [weights[i].get_main_grad for i in range(num_gemms)]
                else:
                    ctx.main_grad_funcs = [
                        lambda j=i: weights[j].main_grad for i in range(num_gemms)
                    ]
            else:
                ctx.main_grad_funcs = [lambda: None for i in range(num_gemms)]
            ctx.device = device
            ctx.grad_output_quantizers = grad_output_quantizers
            ctx.m_splits = m_splits
            ctx.num_gemms = num_gemms
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
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
            ctx.wgrad_store = wgrad_store
            ctx.save_original_input = save_original_input
            ctx.input_quantizers = input_quantizers

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
            origin_weights = saved_tensors[2 * N : 3 * N]
            biases = saved_tensors[3 * N : 4 * N]
            main_grads = [main_grad_func() for main_grad_func in ctx.main_grad_funcs]

            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                for i in range(ctx.num_gemms):
                    w = torch.nn.Parameter(weights[i], weights[i].requires_grad)
                    w.main_grad = main_grads[i]
                    weights[i] = w

            # Preprocess grad output
            grad_output_view = grad_output.contiguous().view(-1, grad_output.shape[-1])
            grad_output = [None] * ctx.num_gemms
            grad_biases = [None] * ctx.num_gemms
            if ctx.fp8:
                if ctx.use_bias:
                    grad_output_mats = torch.split(grad_output_view, ctx.m_splits)
                    recipe = ctx.fp8_recipe
                    if recipe.delayed() or recipe.float8_current_scaling() or recipe.mxfp8():
                        # Fused bias grad + quantize kernel
                        for i in range(ctx.num_gemms):
                            grad_biases[i], grad_output[i] = tex.bgrad_quantize(
                                grad_output_mats[i],
                                ctx.grad_output_quantizers[i],
                            )
                    else:
                        # Unfused bias grad and multi-tensor quantize
                        for i in range(ctx.num_gemms):
                            grad_biases[i] = grad_output_mats[i].sum(dim=0)
                        grad_output = tex.split_quantize(
                            grad_output_view,
                            ctx.m_splits,
                            ctx.grad_output_quantizers,
                        )
                else:
                    # Multi-tensor quantize
                    grad_output = tex.split_quantize(
                        grad_output_view,
                        ctx.m_splits,
                        ctx.grad_output_quantizers,
                    )
            else:
                # Only split grad output. Grad bias is fused with
                # wgrad GEMM.
                grad_output = torch.split(
                    cast_if_needed(grad_output_view, ctx.activation_dtype),
                    ctx.m_splits,
                )

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            if ctx.requires_dgrad:
                dgrad_gemm_use_split_accumulator = _2X_ACC_DGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_dgrad"):
                        dgrad_gemm_use_split_accumulator = (
                            recipe.fp8_gemm_dgrad.use_split_accumulator
                        )
                dgrad = torch.empty(
                    (sum(ctx.m_splits), ctx.weights_shape_1),
                    dtype=ctx.activation_dtype,
                    device=ctx.device,
                )

                for weight, quantizer in zip(weights, ctx.weight_quantizers):
                    if quantizer is not None and isinstance(weight, QuantizedTensorBase):
                        weight.update_usage(
                            rowwise_usage=quantizer.rowwise_usage,
                            columnwise_usage=quantizer.columnwise_usage,
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
                    use_split_accumulator=dgrad_gemm_use_split_accumulator,
                )

            if ctx.weights_requires_grad:
                wgrad_gemm_use_split_accumulator = _2X_ACC_WGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_wgrad"):
                        wgrad_gemm_use_split_accumulator = (
                            recipe.fp8_gemm_wgrad.use_split_accumulator
                        )
                if ctx.fuse_wgrad_accumulation:
                    wgrad_list = main_grads
                else:
                    wgrad_list = [
                        torch.empty(w.size(), dtype=ctx.activation_dtype, device=ctx.device)
                        for w in weights
                    ]

                if ctx.save_original_input:
                    inp = inputmats[0]
                    in_features = inp.shape[-1]
                    inp_view = inp.reshape(-1, in_features)
                    if ctx.input_quantizers[0] is not None:
                        for input_quantizer in ctx.input_quantizers:
                            if isinstance(
                                input_quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)
                            ):
                                input_quantizer.set_usage(rowwise=True, columnwise=True)
                            else:
                                input_quantizer.set_usage(rowwise=False, columnwise=True)
                    inputmats: list
                    if ctx.fp8:
                        inputmats = tex.split_quantize(inp_view, ctx.m_splits, ctx.input_quantizers)
                    else:
                        inputmats = torch.split(
                            cast_if_needed(inp_view, ctx.activation_dtype), ctx.m_splits
                        )

                grouped_gemm_wgrad = functools.partial(
                    general_grouped_gemm,
                    out_dtype=ctx.activation_dtype,
                    workspaces=get_multi_stream_cublas_workspace(),
                    layout="NT",
                    grad=True,
                    m_splits=ctx.m_splits,
                    use_bias=ctx.use_bias if grad_biases[0] is None else None,
                    bias=biases,
                    use_split_accumulator=wgrad_gemm_use_split_accumulator,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                )
                # WGRAD
                if ctx.wgrad_store is not None and ctx.wgrad_store.delay_wgrad_compute():
                    ctx.wgrad_store.put([inputmats, grad_output, wgrad_list], grouped_gemm_wgrad)
                else:
                    _, grad_biases_, _ = grouped_gemm_wgrad(inputmats, grad_output, wgrad_list)

                    for i in range(ctx.num_gemms):
                        if grad_biases[i] is None:
                            grad_biases[i] = grad_biases_[i]
                    del grad_biases_

                    # Deallocate input tensor
                    clear_tensor_data(*inputmats)

                def handle_custom_ddp_from_mcore(weight, wgrad):
                    if ctx.weights_requires_grad:
                        # Handle custom DDP from mcore.
                        if ctx.fuse_wgrad_accumulation and hasattr(
                            weight, "grad_added_to_main_grad"
                        ):
                            weight.grad_added_to_main_grad = True
                            if getattr(weight, "zero_out_wgrad", False):
                                wgrad = torch.zeros(
                                    weight.main_grad.shape,
                                    dtype=weight.dtype,
                                    device=torch.cuda.current_device(),
                                    requires_grad=False,
                                )
                            else:
                                wgrad = torch.empty(
                                    weight.main_grad.shape,
                                    dtype=weight.dtype,
                                    device=torch.cuda.current_device(),
                                    requires_grad=False,
                                )
                        elif ctx.fuse_wgrad_accumulation:
                            wgrad = None
                    else:
                        wgrad = None
                    return wgrad

                wgrad_list = [
                    handle_custom_ddp_from_mcore(weight, wgrad)
                    for weight, wgrad in zip(origin_weights, wgrad_list)
                ]
            else:
                wgrad_list = [None] * ctx.num_gemms

            if not ctx.use_bias or (
                ctx.wgrad_store is not None
                and ctx.wgrad_store.delay_wgrad_compute()
                and not ctx.fp8
            ):
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
            None,
            None,
            None,
            None,
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
    delay_wgrad_compute : bool, default = `False`
                         Whether to delay weight gradient computation
    save_original_input : bool, default = `False`
                       If set to `True`, always saves the original input tensor rather than the
                       cast tensor. In some scenarios, the input tensor is used by multiple modules,
                       and saving the original input tensor may reduce the memory usage.
                       Cannot work with FP8 DelayedScaling recipe.

    Note: GroupedLinear doesn't really handle the TP communications inside. The `tp_size` and
          `parallel_mode` are used to determine the shapes of weights and biases.
          The TP communication should be handled in the dispatch and combine stages of MoE models.
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
        delay_wgrad_compute: bool = False,
        save_original_input: bool = False,
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
        self.save_original_input = save_original_input
        assert (
            not ub_overlap_rs and not ub_overlap_ag
        ), "GroupedLinear doesn't support Userbuffer overlap."
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

        self.wgrad_store = WeightGradStore(delay_wgrad_compute)

        self._offsets = {"input": 0, "weight": 1, "output": 2, "grad_output": 0, "grad_input": 1}
        self._num_fp8_tensors_per_gemm = {
            "fwd": 3,
            "bwd": 2,
        }

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        if self.tp_size > 1 and bias:
            raise ValueError(
                "GroupedLinear doesn't support bias when TP > 1. "
                "Because the TP communication is handled outside of this module."
            )

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
                fp8_meta_index=self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"],
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

        if self.wgrad_store.delay_wgrad_compute():
            for name, param in self.named_parameters():
                for i in range(self.num_gemms):
                    if name in (f"weight{i}", f"bias{i}"):
                        param.skip_backward_post_hook = True

    def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
        """Init scales and amaxes for fwd | bwd."""
        super().set_meta_tensor(fwd, recipe)

        # customize quantizers based on each recipe & layer configs
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe.float8_current_scaling():
            assert not self.tp_size > 1, (
                "GroupedLinear doesn't support TP > 1 with Float8 current scaling. "
                "Because the TP communication is handled outside of this module."
            )
            self._customize_quantizers_float8_current_scaling(fwd, recipe)

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
            inp, QuantizedTensorBase
        ), "GroupedLinear doesn't support input tensor in FP8."
        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."

        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        if is_first_microbatch is None or is_first_microbatch:
            device_ctx = torch.cuda.device(
                getattr(self, list(self.named_parameters())[0][0]).device
            )
        else:
            device_ctx = contextlib.nullcontext()

        with device_ctx, self.prepare_forward(inp, num_gemms=self.num_gemms) as inp:
            weight_tensors = self._get_weight_tensors()
            bias_tensors = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]

            weight_quantizers = self._get_weight_quantizers()
            input_quantizers, output_quantizers = (
                [None] * self.num_gemms,
                [None] * self.num_gemms,
            )
            grad_output_quantizers, _ = [None] * self.num_gemms, [None] * self.num_gemms
            if self.fp8:
                input_quantizers = [
                    self.quantizers["scaling_fwd"][
                        self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                    ]
                    for i in range(self.num_gemms)
                ]
                # TODO: use internal after #1638 is merged. # pylint: disable=fixme
                for i in range(self.num_gemms):
                    input_quantizers[i].internal = False
                if torch.is_grad_enabled():
                    grad_output_quantizers = [
                        self.quantizers["scaling_bwd"][
                            self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["bwd"]
                        ]
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
                self.apply_bias,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.wgrad_store,
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
                self.save_original_input,
                *weight_tensors,
                *bias_tensors,
            )
            out = linear_fn(*args)

        if self.return_bias:
            return out, [cast_if_needed(b, self.activation_dtype) for b in bias_tensors]
        return out

    def backward_dw(self):
        """
        Execute the delayed weight gradient computation.
        This method is called after the main backward pass to compute weight gradients.
        """
        if self.wgrad_store is None or not self.wgrad_store.delay_wgrad_compute():
            return
        with torch.cuda.nvtx.range("_GroupedLinear_wgrad"):
            (_, grad_biases_, _), tensor_list = self.wgrad_store.pop()
            wgrad_list = tensor_list[2]
            weight_params = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
            bias_params = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]
            if not self.fuse_wgrad_accumulation:
                for i in range(self.num_gemms):
                    weight_params[i].grad = wgrad_list[i].to(weight_params[i].dtype)
            if self.use_bias:
                for i in range(self.num_gemms):
                    if bias_params[i].grad is None:
                        bias_params[i].grad = grad_biases_[i].to(bias_params[i].dtype)
            del grad_biases_
            del wgrad_list
            del tensor_list
            for wgrad_accumulation_and_reduce_hook in self.wgrad_accumulation_and_reduce_hooks:
                wgrad_accumulation_and_reduce_hook()

    def _customize_quantizers_float8_current_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on current scaling recipe + linear."""
        assert (
            recipe.float8_current_scaling()
        ), "current scaling recipe quantizer customization here"
        if fwd:
            for i in range(self.num_gemms):
                # set configs about amax epsilon and power_2_scale
                self.quantizers["scaling_fwd"][
                    self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                ].force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
                self.quantizers["scaling_fwd"][
                    self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                ].amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon
                # also set weight quantizer with same amax_epsilon & power_2_scale
                self.quantizers["scaling_fwd"][
                    self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                ].force_pow_2_scales = recipe.fp8_quant_fwd_weight.power_2_scale
                self.quantizers["scaling_fwd"][
                    self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                ].amax_epsilon = recipe.fp8_quant_fwd_weight.amax_epsilon
        else:
            for i in range(self.num_gemms):
                # set grad_output_quantizer with amax epsilon and power_2_scale
                self.quantizers["scaling_bwd"][
                    self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["bwd"]
                ].force_pow_2_scales = recipe.fp8_quant_bwd_grad.power_2_scale
                self.quantizers["scaling_bwd"][
                    self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["bwd"]
                ].amax_epsilon = recipe.fp8_quant_bwd_grad.amax_epsilon

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorBase]]:
        """Get the weight tensors of the module."""
        weight_tensors = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        if not self.fp8 and any(isinstance(w, QuantizedTensorBase) for w in weight_tensors):
            warnings.warn(
                "You are using quantized weights without quantized compute. "
                "Please make sure this is intentional."
            )
            weight_tensors = [
                w.dequantize() if isinstance(w, QuantizedTensorBase) else w for w in weight_tensors
            ]
        return weight_tensors

    def _get_weight_quantizers(self) -> List[Quantizer]:
        """Get the weight quantizers of the module."""
        if not self.fp8 and not self.fp8_calibration:
            return [None] * self.num_gemms
        weight_quantizers = [
            self.quantizers["scaling_fwd"][
                self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"]
            ]
            for i in range(self.num_gemms)
        ]
        for i in range(self.num_gemms):
            weight_quantizers[i].internal = True
        return weight_quantizers
