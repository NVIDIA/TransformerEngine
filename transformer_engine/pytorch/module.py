# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Top level Transformer Engine PyTorch modules"""
import os
import warnings
from abc import ABC, abstractmethod
from typing import Union, Optional, Callable, Tuple, Dict, List, Any
from functools import partial

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

import transformer_engine_extensions as tex
from .fp8 import (
    is_fp8_enabled,
    get_fp8_recipe,
    get_fp8_group,
    get_default_fp8_recipe,
    get_fp8_te_dtype,
    is_first_fp8_module,
    new_fp8_context_id,
    get_fp8_context_id,
    set_fp8_context_id,
    add_amax_to_global_buffer,
    copy_amax_from_global_buffer,
    global_amax_reduction,
    setup_amax_forward_global_reduce_func,
    amax_and_scale_update,
    get_global_fp8_buffer,
    set_global_fp8_buffer,
    set_amax_buffer_key_deletion,
    delete_key_from_amax_buffer,
)
from .jit import (
    bias_gelu_fused,
    bgrad_dgelu_fused,
    set_jit_fusion_options,
    warmup_jit_bias_gelu_all_dtypes,
)
from .utils import (
    divide,
    get_default_init_method,
    cast_if_needed,
)
from .distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    initialize_affine_weight_gpu,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    gather_along_last_dim,
)
from .cpp_extensions import (
    fp8_gemm,
    gemm,
    fp8_cast_transpose_fused,
    fp8_cast_transpose_bgrad_fused,
    fp8_gelu,
    fp8_cast_transpose_bgrad_dgelu_fused,
    layernorm_fwd_fp8,
    cast_to_fp8,
    cast_from_fp8,
)
from .constants import GemmParallelModes, dist_group_type, TE_DType

_2X_ACC_FPROP = False
_2X_ACC_DGRAD = True
_2X_ACC_WGRAD = True
_cublas_workspace = None


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        return 33_554_432
    return 4_194_304


def get_workspace() -> torch.Tensor:
    """Returns workspace for cublas."""
    global _cublas_workspace
    if _cublas_workspace is None:
        _cublas_workspace = torch.empty(
            get_cublas_workspace_size_bytes(), dtype=torch.int8, device="cuda"
        )
    return _cublas_workspace


class TransformerEngineBaseModule(torch.nn.Module, ABC):
    """Base TE module."""

    def __init__(self) -> None:
        super().__init__()
        assert torch.cuda.is_available(), "TransformerEngine needs CUDA."
        self.fp8 = False
        self.fp8_meta = {}
        self.fp8_meta["fp8_group"] = None
        self.fp8_meta["recipe"] = get_default_fp8_recipe()
        self.fp8_meta_tensors_initialized = False
        self.tp_group = None
        self.tp_group_initialized = False
        self.tp_size = 1
        self.sequence_parallel = False
        self.fp8_weight_shapes = []

    def set_meta_tensor(self, fwd: bool) -> None:
        """Init scales and amaxes for fwd | bwd."""
        fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"
        num_fp8_tensors = (
            self.fp8_meta["num_gemms"] * 2 if fwd else self.fp8_meta["num_gemms"]
        )

        self.fp8_meta[fp8_meta_tensor_key] = tex.FP8TensorMeta()
        self.fp8_meta[fp8_meta_tensor_key].scale = torch.ones(
            num_fp8_tensors, dtype=torch.float32, device="cuda"
        )
        self.fp8_meta[fp8_meta_tensor_key].scale_inv = torch.ones(
            num_fp8_tensors, dtype=torch.float32, device="cuda"
        )
        self.fp8_meta[fp8_meta_tensor_key].amax_history = torch.zeros(
            self.fp8_meta["recipe"].amax_history_len,
            num_fp8_tensors,
            dtype=torch.float32,
            device="cuda",
        )

    def init_fp8_meta_tensors(self) -> None:
        """Init scales and amaxes."""
        # Checkpoint loaded
        if self.fp8_meta_tensors_initialized:
            return

        self.set_meta_tensor(True)
        self.set_meta_tensor(False)

    def get_extra_state(self) -> Union[List[Any], None]:
        """Save before checkpointing."""
        if self.fp8:
            state = []
            state.append(self.fp8_meta["scaling_fwd"].scale)
            state.append(self.fp8_meta["scaling_fwd"].amax_history)
            state.append(self.fp8_meta["scaling_bwd"].scale)
            state.append(self.fp8_meta["scaling_bwd"].amax_history)
            state.append(get_global_fp8_buffer())
            state.append(self.fp8_meta["update_amax_and_scale_fwd"])
            state.append(self.fp8_meta["global_fp8_buffer_pos_fwd"])
            state.append(self.fp8_meta["global_fp8_buffer_pos_bwd"])
            state.append(self.fp8_meta["autocast_id_fwd"])
            state.append(self.fp8_meta["autocast_id_bwd"])
            return state
        return None

    def set_extra_state(self, state: Union[List[Any], None]) -> None:
        """Load previous state."""
        if state is None:
            return

        # Retrieve checkpointed items.
        scale_fwd = state[0]
        amax_history_fwd = state[1]
        scale_bwd = state[2]
        amax_history_bwd = state[3]
        self.fp8_meta["recipe"].amax_history_len = amax_history_fwd.shape[0]
        self.fp8_meta["num_gemms"] = (
            amax_history_fwd.shape[1] // 2
        )  # Two FWD tensors per GEMM

        # Initialize before loading
        self.init_fp8_meta_tensors()
        self.fp8_meta["scaling_fwd"].scale.copy_(scale_fwd)
        self.fp8_meta["scaling_fwd"].amax_history.copy_(amax_history_fwd)
        self.fp8_meta["scaling_bwd"].scale.copy_(scale_bwd)
        self.fp8_meta["scaling_bwd"].amax_history.copy_(amax_history_bwd)
        self.fp8_meta_tensors_initialized = True

        # Restore global FP8 buffer state.
        set_global_fp8_buffer(state[4])
        self.fp8_meta["update_amax_and_scale_fwd"] = state[5]
        self.fp8_meta["global_fp8_buffer_pos_fwd"] = state[6]
        self.fp8_meta["global_fp8_buffer_pos_bwd"] = state[7]
        self.fp8_meta["autocast_id_fwd"] = state[8]
        self.fp8_meta["autocast_id_bwd"] = state[9]

    def set_activation_dtype(self, inp: torch.Tensor) -> None:
        """Get activation data type for AMP."""
        # Native AMP (`torch.autocast`) gets highest priority
        if torch.is_autocast_enabled():
            self.activation_dtype = torch.get_autocast_gpu_dtype()
            return

        # All checks after this have already been performed once, thus skip
        # We assume that user doesn't change input types across iterations
        if hasattr(self, "activation_dtype"):
            return

        assert all(
            (
                (inp.dtype == param.dtype) if param is not None else True
                for param in self.parameters()
            )
        ), (
            "Data type for activations and weights must "
            "match when outside of autocasted region"
        )
        assert all(
            (
                (inp.dtype == buf.dtype) if buf is not None else True
                for buf in self.buffers()
            )
        ), (
            "Data type for activations and buffers must "
            "match when outside of autocasted region"
        )
        self.activation_dtype = inp.dtype

    def set_fp8_weights(self) -> None:
        """Initializes FP8 weights for the module as class attributes. These
        are not parameters or buffers since we do not want functions such as
        `.to(dtype)` or `.to(device)` to effect them. These also do not need
        to be checkpointed. During `init` phase of the module, the attribute
        `fp8_weight_shapes` must be populated with the tensor shapes for FP8
        weights. This function will iterate over those shapes and initialize
        respective attributed named `weight1_fp8`, `weight2_fp8`, ...
        """
        for i, shape in enumerate(self.fp8_weight_shapes, start=1):
            weight_cast_attr = f"weight{i}_fp8"
            weight_transpose_attr = f"weight{i}_t_fp8"
            if self.fp8:
                if not hasattr(self, weight_cast_attr):
                    setattr(
                        self,
                        weight_cast_attr,
                        torch.empty(
                            shape,
                            device=torch.cuda.current_device(),
                            dtype=torch.int8,
                        ),
                    )
                if not hasattr(self, weight_transpose_attr):
                    setattr(
                        self,
                        weight_transpose_attr,
                        torch.empty(
                            shape[1],
                            shape[0],
                            device=torch.cuda.current_device(),
                            dtype=torch.int8,
                        ),
                    )
            else:
                setattr(self, weight_cast_attr, torch.Tensor())
                setattr(self, weight_transpose_attr, torch.Tensor())

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """Set TP group."""
        self.tp_group = tp_group
        self.tp_group_initialized = True

    def fp8_init(self, num_gemms: int = 1) -> None:
        """Initialize fp8 related metadata and tensors during fprop."""
        # If fp8 isn't enabled, turn off and return.
        if not is_fp8_enabled():
            self.fp8 = False
            return

        # FP8 is already enabled and recipe is the same, don't do anything.
        if self.fp8 and get_fp8_recipe() == self.fp8_meta["recipe"]:
            return

        # Set FP8, recipe, and other FP8 metadata
        self.fp8 = True
        self.fp8_meta["recipe"] = get_fp8_recipe()
        self.fp8_meta["num_gemms"] = num_gemms
        self.fp8_meta["fp8_group"] = get_fp8_group()

        # Set FP8_MAX per tensor according to recipe
        self.fp8_meta["fp8_max_fwd"] = self.fp8_meta["recipe"].fp8_format.value.max_fwd
        self.fp8_meta["fp8_max_bwd"] = self.fp8_meta["recipe"].fp8_format.value.max_bwd

        # Allocate scales and amaxes
        self.init_fp8_meta_tensors()

    def pre_forward(self, inp: torch.Tensor, num_gemms: int = 1) -> None:
        """Checks and prep for FWD."""

        assert inp.is_cuda, "TransformerEngine needs CUDA."

        if self.tp_size > 1:
            assert self.tp_group_initialized, "TP group not initialized."

        self.set_activation_dtype(inp)
        self.fp8_init(num_gemms=num_gemms)
        self.set_fp8_weights()

        if self.fp8_meta.get("update_amax_and_scale_fwd", False):
            # Previous iteration was grad_enabled
            copy_amax_from_global_buffer(self.fp8_meta, forward=True)
            amax_and_scale_update(self.fp8_meta, True)
            set_amax_buffer_key_deletion(self.fp8_meta, forward=True)

        if self.fp8 and torch.is_grad_enabled() and self.training:
            self.fp8_meta["first_module"] = is_first_fp8_module()

            if self.fp8_meta["first_module"]:
                self.fp8_meta["autocast_id_fwd"] = new_fp8_context_id()
                set_fp8_context_id(self.fp8_meta["autocast_id_fwd"])
            else:
                self.fp8_meta["autocast_id_fwd"] = get_fp8_context_id()

            add_amax_to_global_buffer(self.fp8_meta, forward=True)
            self.fp8_meta["update_amax_and_scale_fwd"] = True
        else:
            self.fp8_meta["update_amax_and_scale_fwd"] = False

    def post_forward(self) -> None:
        """This is needed because there isn't a way for a module to know
        if it's the last FP8 module in the forward autocast. It is useful
        to setup the forward aggregated amax reduction for every module
        just in case. The autocast exit will pick up the most recent.
        """

        if self.fp8 and torch.is_grad_enabled() and self.training:
            set_fp8_context_id(self.fp8_meta["autocast_id_fwd"])
            reduce_func = partial(
                global_amax_reduction,
                self.fp8_meta,
                self.sequence_parallel,
                self.tp_group,
                forward=True,
            )
            setup_amax_forward_global_reduce_func(reduce_func)

    @staticmethod
    def pre_backward(fp8: bool, fp8_meta: Dict[str, Any]) -> None:
        """Checks and prep for BWD."""
        if not fp8:
            return

        # From previous iteration
        copy_amax_from_global_buffer(fp8_meta, forward=False)
        amax_and_scale_update(fp8_meta, False)
        set_amax_buffer_key_deletion(fp8_meta, forward=False)

        # Get new backward key.
        if "autocast_id_bwd" not in fp8_meta:
            fp8_meta["autocast_id_bwd"] = fp8_meta["autocast_id_fwd"]
        else:
            fp8_meta["autocast_id_bwd"] += 1

        add_amax_to_global_buffer(fp8_meta, forward=False)

    @staticmethod
    def post_backward(
        fp8: bool,
        fp8_meta: Dict[str, Any],
        reduce_amax_across_tp_group: bool,
        tp_group: Union[dist_group_type, None],
    ) -> None:
        """Checks and prep for BWD."""
        if not fp8:
            return

        if fp8_meta["first_module"]:
            global_amax_reduction(
                fp8_meta, reduce_amax_across_tp_group, tp_group, forward=False
            )
            delete_key_from_amax_buffer(forward=False)

    def set_nccl_overlap_warning_if_tp(self) -> None:
        """When using TP, the NCCL communication needs to be scheduled
        before the GEMM for there to be a guaranteed overlap. From the
        host side in TE, the comm calls are always launched first, but
        to ensure that the GEMM isn't scheduled first, the environment
        variable `CUDA_DEVICE_MAX_CONNECTIONS` needs to be set to 1 to
        force a single channel.
        """
        if self.tp_size == 1:
            return
        num_cuda_work_queues = int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "0"))
        if num_cuda_work_queues != 1:
            warnings.warn(
                "To guarantee overlapping TP and SP collectives with the backward"
                "GEMMs, set environment variable CUDA_DEVICE_MAX_CONNECTIONS = 1"
            )

    @staticmethod
    def grad_output_preprocess(
        ctx, grad_output: torch.Tensor, row_parallel_mode: bool
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Utility function for backward.
        Returns tuple in order (all optional/None based on training precion/recipe):
            R1: gathered `grad_output` in higher precision.
            R2: gathered `grad_output` in FP8.
            R3: R2 transposed.
            R4: bias gradient on R1.

        """
        grad_output = grad_output.contiguous()
        grad_output_mat = grad_output.view((-1, grad_output.shape[-1]))
        gather_grad_output = row_parallel_mode and ctx.sequence_parallel

        # No-FP8 case: bgrad is fused with wgrad for this case.
        if not ctx.fp8:
            if gather_grad_output:
                grad_output_mat, _ = gather_along_first_dim(
                    grad_output_mat, ctx.tp_group
                )
            return grad_output_mat, None, None, None

        fp8_dtype_backward = get_fp8_te_dtype(
            ctx.fp8_meta["recipe"], fprop_tensor=False
        )

        # FP8 case with non-FP8 wgrad
        if (
            gather_grad_output
            and ctx.fp8_meta["recipe"].override_linear_precision.wgrad
        ):
            grad_output_mat, _ = gather_along_first_dim(grad_output_mat, ctx.tp_group)
        # FP8 case with gather: unfused bgrad, cast, transpose for efficient gather
        elif gather_grad_output:
            if ctx.use_bias:
                grad_bias = grad_output_mat.sum(dim=0)
            else:
                grad_bias = None
            grad_output_c = cast_to_fp8(
                grad_output_mat,
                ctx.fp8_meta["scaling_bwd"],
                tex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
            )
            grad_output_c, _ = gather_along_first_dim(grad_output_c, ctx.tp_group)
            grad_output_t = tex.fp8_transpose(grad_output_c, fp8_dtype_backward)

            return grad_output_mat, grad_output_c, grad_output_t, grad_bias

        # FP8 case without gather: cast, transpose, bgrad fused
        if ctx.use_bias:
            grad_bias, grad_output_c, grad_output_t = fp8_cast_transpose_bgrad_fused(
                grad_output_mat,
                ctx.fp8_meta["scaling_bwd"],
                tex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
            )
        else:
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                grad_output_c, grad_output_t = fp8_cast_transpose_fused(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                )
            else:
                grad_output_c = cast_to_fp8(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                )
                grad_output_t = None
            grad_bias = None

        return grad_output_mat, grad_output_c, grad_output_t, grad_bias

    @abstractmethod
    def forward(self):
        """Needs override."""


class _LayerNormLinear(torch.autograd.Function):
    """LayerNormLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        weight: torch.Tensor,
        weight_fp8: torch.Tensor,
        weight_t_fp8: torch.Tensor,
        bias: torch.Tensor,
        use_bias: bool,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        tp_group: Union[dist_group_type, None],
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        return_layernorm_output: bool,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        ln_bias = cast_if_needed(ln_bias, activation_dtype)

        # If residual connection is after LN, we need `ln_out`
        # tensor in higher precision, this comes at the cost
        # of an extra fp8 cast.
        if fp8:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

            if not return_layernorm_output:
                ln_out, mu, rsigma = layernorm_fwd_fp8(
                    inputmat,
                    ln_weight,
                    ln_bias,
                    eps,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                )
            else:
                ln_out_return, mu, rsigma = tex.layernorm_fwd(
                    inputmat, ln_weight, ln_bias, eps
                )
                ln_out = cast_to_fp8(
                    ln_out_return,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                )
        else:
            ln_out, mu, rsigma = tex.layernorm_fwd(inputmat, ln_weight, ln_bias, eps)
            ln_out_return = ln_out

        # Column Parallel Linear
        if parallel_mode == "column" and sequence_parallel:
            ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
        else:
            ln_out_total = ln_out

        if fp8:
            bias_dtype = (
                torch.bfloat16
                if activation_dtype == torch.float32
                else activation_dtype
            )
            bias = cast_if_needed(bias, bias_dtype) if use_bias else bias

            if update_fp8_weights:
                fp8_cast_transpose_fused(
                    weight,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_WEIGHT,
                    fp8_dtype_forward,
                    cast_out=weight_fp8,
                    transpose_out=weight_t_fp8,
                )

            out = fp8_gemm(
                weight_fp8,
                fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_WEIGHT],
                fp8_dtype_forward,
                ln_out_total,
                fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
                fp8_dtype_forward,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
            )
        else:
            # Cast for native AMP
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

            out, _, _ = gemm(
                weight,
                ln_out_total,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
            )

        ctx.save_for_backward(
            inputmat,
            ln_weight,
            mu,
            rsigma,
            weight,
            weight_t_fp8,
            ln_out,
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
        ctx.return_layernorm_output = return_layernorm_output

        # Row Parallel Linear
        if parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        out = out.view(-1, *inp.shape[1:-1], out.shape[-1])

        if return_layernorm_output:
            return out, ln_out_return.view_as(inp)
        return out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        TransformerEngineBaseModule.pre_backward(ctx.fp8, ctx.fp8_meta)

        (
            inputmat,
            ln_weight,
            mu,
            rsigma,
            weight,
            weight_t_fp8,
            ln_out,
            fwd_scale_inverses,
        ) = ctx.saved_tensors

        (
            grad_output,
            grad_output_c,
            grad_output_t,
            grad_bias,
        ) = TransformerEngineBaseModule.grad_output_preprocess(
            ctx, grad_outputs[0], ctx.parallel_mode == "row"
        )

        # Column Parallel Linear
        # Overlap input AG with dgrad
        if ctx.parallel_mode == "column" and ctx.sequence_parallel:
            ln_out_total, handle = gather_along_first_dim(
                ln_out, ctx.tp_group, async_op=True
            )
        else:
            ln_out_total = ln_out

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

            # DGRAD
            dgrad = fp8_gemm(
                weight_t_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_WEIGHT],
                fp8_dtype_forward,
                grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT1],
                fp8_dtype_backward,
                ctx.activation_dtype,
                get_workspace(),
                use_split_accumulator=_2X_ACC_DGRAD,
            )
        else:
            # DGRAD
            dgrad, _, _ = gemm(
                weight,
                grad_output,
                ctx.activation_dtype,
                get_workspace(),
                layout="NN",
                grad=True,
            )

        # Overlap dgrad-RS/AR with wgrad
        if ctx.parallel_mode == "column" and ctx.sequence_parallel:
            handle.wait()
            dgrad, handle = reduce_scatter_along_first_dim(
                dgrad, ctx.tp_group, async_op=True
            )
        elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
            dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

        if ctx.fp8:
            # WGRAD
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                ln_out_total_t = tex.fp8_transpose(ln_out_total, fp8_dtype_forward)
                wgrad = fp8_gemm(
                    ln_out_total_t,
                    fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_INPUT],
                    fp8_dtype_forward,
                    grad_output_t,
                    ctx.fp8_meta["scaling_bwd"].scale_inv[
                        tex.FP8BwdTensors.GRAD_OUTPUT1
                    ],
                    fp8_dtype_backward,
                    ctx.activation_dtype,
                    get_workspace(),
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    fp32_output=ctx.fuse_wgrad_accumulation,
                    out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    use_split_accumulator=_2X_ACC_WGRAD,
                )
            else:
                ln_out_total_c = cast_from_fp8(
                    ln_out_total,
                    ctx.fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                    TE_DType[ctx.activation_dtype],
                )
                wgrad, _, _ = gemm(
                    ln_out_total_c,
                    grad_output,
                    ctx.activation_dtype,
                    get_workspace(),
                    layout="NT",
                    grad=True,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    fp32_output=ctx.fuse_wgrad_accumulation,
                    out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                )
        else:
            # WGRAD
            wgrad, grad_bias, _ = gemm(
                ln_out_total,
                grad_output,
                ctx.activation_dtype,
                get_workspace(),
                layout="NT",
                grad=True,
                use_bias=ctx.use_bias,
                accumulate=accumulate_wgrad_into_param_main_grad,
                fp32_output=ctx.fuse_wgrad_accumulation,
                out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
            )

        # Column Parallel Linear
        if ctx.parallel_mode == "column" and ctx.tensor_parallel and handle is not None:
            handle.wait()

        # LayerNorm gradient
        d_ln_out = dgrad.view(inputmat.shape)

        # Residual gradient
        if ctx.return_layernorm_output:
            d_ln_out = d_ln_out + grad_outputs[1].view_as(d_ln_out)

        dxmat, dgamma, dbeta = tex.layernorm_bwd(
            d_ln_out, inputmat, mu, rsigma, ln_weight
        )

        if not ctx.use_bias:
            grad_bias = None

        TransformerEngineBaseModule.post_backward(
            ctx.fp8, ctx.fp8_meta, ctx.sequence_parallel, ctx.tp_group
        )

        return (
            dxmat.view(ctx.inp_shape),
            dgamma,
            dbeta,
            wgrad,
            None,
            None,
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
        )


class LayerNormLinear(TransformerEngineBaseModule):
    """
    Applies layer normalization followed by linear transformation to the incoming data.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.

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
    skip_weight_param_allocation: bool, default = `False`
                                 if set to `True`, weight parameter is not allocated and must be
                                 passed as a keyword argument `weight` during the forward pass.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.float32`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: torch.dtype = torch.float32,
        parallel_mode: Optional[str] = None,
        return_layernorm_output: bool = False,
        skip_weight_param_allocation: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.return_layernorm_output = return_layernorm_output
        self.skip_weight_param_allocation = skip_weight_param_allocation

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

        self.eps = eps
        self.layer_norm_weight = Parameter(
            torch.empty(
                in_features,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.layer_norm_bias = Parameter(
            torch.empty(
                in_features,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
        setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)
        self.reset_layer_norm_parameters()

        if not skip_weight_param_allocation:
            self.weight = Parameter(
                torch.empty(
                    self.out_features,
                    self.in_features,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )

            initialize_affine_weight_gpu(
                self.weight,
                init_method,
                get_rng_state_tracker,
                partition_dim=1 if self.parallel_mode == "row" else 0,
                stride=1,
            )

            if self.use_bias or self.return_bias:
                self.bias = Parameter(
                    torch.empty(
                        self.out_features,
                        device=torch.cuda.current_device(),
                        dtype=params_dtype,
                    )
                )
                if self.parallel_mode == "column":
                    set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
            else:
                self.register_buffer("bias", torch.Tensor(), persistent=False)

            with torch.no_grad():
                self.bias.zero_()

        self.fp8_weight_shapes.append(torch.Size((self.out_features, self.in_features)))

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.use_bias:
            self.gemm_bias_unfused_add = True
            self.use_bias = False
        else:
            self.gemm_bias_unfused_add = False

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        init.ones_(self.layer_norm_weight)
        init.zeros_(self.layer_norm_bias)

    def forward(
        self,
        inp: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a linear transformation.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        weight : torch.Tensor, default = None
                An optional weight tensor for the module. This argument is compulsory if module
                is initialized with `skip_weight_param_allocation=True`
        bias : torch.Tensor, default = None
              An optional bias tensor for the module. This argument is compulsory if module
              is initialized with `skip_weight_param_allocation=True` and one of `use_bias`
              or `return_bias`
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

        self.pre_forward(inp)

        bias_tensor = bias if bias is not None else self.bias

        out = _LayerNormLinear.apply(
            inp,
            self.layer_norm_weight,
            self.layer_norm_bias,
            weight if weight is not None else self.weight,
            self.weight1_fp8,
            self.weight1_t_fp8,
            bias_tensor,
            self.use_bias,
            self.eps,
            is_first_microbatch,
            self.fp8,
            self.fp8_meta,
            self.fuse_wgrad_accumulation,
            self.tp_group,
            self.sequence_parallel,
            self.tp_size > 1,
            self.activation_dtype,
            self.parallel_mode,
            self.return_layernorm_output,
        )

        self.post_forward()

        if self.return_layernorm_output:
            out, ln_out = out

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(bias_tensor, self.activation_dtype), ln_out
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        weight_fp8: torch.Tensor,
        weight_t_fp8: torch.Tensor,
        inp: torch.Tensor,
        bias: torch.Tensor,
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        tp_group: Union[dist_group_type, None],
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        in_features = weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        inputmat_no_fp8 = inputmat

        if fp8:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

            if not fp8_meta["recipe"].override_linear_precision.wgrad:
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
                fp8_cast_transpose_fused(
                    weight,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_WEIGHT,
                    fp8_dtype_forward,
                    cast_out=weight_fp8,
                    transpose_out=weight_t_fp8,
                )

            out = fp8_gemm(
                weight_fp8,
                fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_WEIGHT],
                fp8_dtype_forward,
                inputmat,
                fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
                fp8_dtype_forward,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
            )
        else:
            # Cast for native AMP
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

            out, _, _ = gemm(
                weight,
                inputmat_total,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
            )

        ctx.save_for_backward(
            inputmat_no_fp8
            if not fp8 or fp8_meta["recipe"].override_linear_precision.wgrad
            else None,
            inputmat_t
            if fp8 and not fp8_meta["recipe"].override_linear_precision.wgrad
            else None,
            weight,
            weight_t_fp8,
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

        # Row Parallel Linear
        if parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        TransformerEngineBaseModule.pre_backward(ctx.fp8, ctx.fp8_meta)

        (
            inputmat,
            inputmat_t,
            weight,
            weight_t_fp8,
            fwd_scale_inverses,
        ) = ctx.saved_tensors

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
        if ctx.parallel_mode == "column" and ctx.sequence_parallel:
            if ctx.fp8 and not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                inputmat_t_total, handle = gather_along_last_dim(
                    inputmat_t, ctx.tp_group, async_op=True
                )
            else:
                inputmat_total, handle = gather_along_first_dim(
                    inputmat, ctx.tp_group, async_op=True
                )
        else:
            inputmat_t_total = inputmat_t
            inputmat_total = inputmat

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

            # DGRAD
            dgrad = fp8_gemm(
                weight_t_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_WEIGHT],
                fp8_dtype_forward,
                grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT1],
                fp8_dtype_backward,
                ctx.activation_dtype,
                get_workspace(),
                use_split_accumulator=_2X_ACC_DGRAD,
            )
        else:
            # DGRAD
            dgrad, _, _ = gemm(
                weight,
                grad_output,
                ctx.activation_dtype,
                get_workspace(),
                layout="NN",
                grad=True,
            )

        # Overlap dgrad-RS/AR with wgrad
        if ctx.parallel_mode == "column" and ctx.sequence_parallel:
            handle.wait()
            dgrad, handle = reduce_scatter_along_first_dim(
                dgrad, ctx.tp_group, async_op=True
            )
        elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
            dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

        if ctx.fp8:
            # WGRAD
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                wgrad = fp8_gemm(
                    inputmat_t_total,
                    fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_INPUT],
                    fp8_dtype_forward,
                    grad_output_t,
                    ctx.fp8_meta["scaling_bwd"].scale_inv[
                        tex.FP8BwdTensors.GRAD_OUTPUT1
                    ],
                    fp8_dtype_backward,
                    ctx.activation_dtype,
                    get_workspace(),
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    fp32_output=ctx.fuse_wgrad_accumulation,
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
                    fp32_output=ctx.fuse_wgrad_accumulation,
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
                fp32_output=ctx.fuse_wgrad_accumulation,
                out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
            )

        # Column Parallel Linear
        if ctx.parallel_mode == "column" and ctx.tensor_parallel and handle is not None:
            handle.wait()

        if not ctx.use_bias:
            grad_bias = None

        TransformerEngineBaseModule.post_backward(
            ctx.fp8, ctx.fp8_meta, ctx.sequence_parallel, ctx.tp_group
        )

        return (
            wgrad,
            None,
            None,
            dgrad.view(ctx.inp_shape),
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
        )


class Linear(TransformerEngineBaseModule):
    """
    Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    On NVIDIA GPUs it is a drop-in replacement for `torch.nn.Linear`.

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
    skip_weight_param_allocation: bool, default = `False`
                                 if set to `True`, weight parameter is not allocated and must be
                                 passed as a keyword argument `weight` during the forward pass.

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
    params_dtype : torch.dtype, default = `torch.float32`
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
        params_dtype: torch.dtype = torch.float32,
        parallel_mode: Optional[str] = None,
        skip_weight_param_allocation: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.skip_weight_param_allocation = skip_weight_param_allocation

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

        if not skip_weight_param_allocation:
            self.weight = Parameter(
                torch.empty(
                    self.out_features,
                    self.in_features,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )

            initialize_affine_weight_gpu(
                self.weight,
                init_method,
                get_rng_state_tracker,
                partition_dim=1 if self.parallel_mode == "row" else 0,
                stride=1,
            )

            if self.use_bias or self.return_bias:
                self.bias = Parameter(
                    torch.empty(
                        self.out_features,
                        device=torch.cuda.current_device(),
                        dtype=params_dtype,
                    )
                )
                if self.parallel_mode == "column":
                    set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
            else:
                self.register_buffer("bias", torch.Tensor(), persistent=False)

            with torch.no_grad():
                self.bias.zero_()

        self.fp8_weight_shapes.append(torch.Size((self.out_features, self.in_features)))

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.use_bias:
            self.gemm_bias_unfused_add = True
            self.use_bias = False
        else:
            self.gemm_bias_unfused_add = False

    def forward(
        self,
        inp: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        weight : torch.Tensor, default = None
                An optional weight tensor for the module. This argument is compulsory if module
                is initialized with `skip_weight_param_allocation=True`
        bias : torch.Tensor, default = None
              An optional bias tensor for the module. This argument is compulsory if module
              is initialized with `skip_weight_param_allocation=True` and one of `use_bias`
              or `return_bias`
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

        self.pre_forward(inp)

        bias_tensor = bias if bias is not None else self.bias

        out = _Linear.apply(
            weight if weight is not None else self.weight,
            self.weight1_fp8,
            self.weight1_t_fp8,
            inp,
            bias_tensor,
            self.use_bias,
            is_first_microbatch,
            self.fp8,
            self.fp8_meta,
            self.fuse_wgrad_accumulation,
            self.tp_group,
            self.sequence_parallel,
            self.tp_size > 1,
            self.activation_dtype,
            self.parallel_mode,
        )

        self.post_forward()

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out


class _LayerNormMLP(torch.autograd.Function):
    """LayerNormMLP semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_fp8: torch.Tensor,
        fc1_weight_t_fp8: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_fp8: torch.Tensor,
        fc2_weight_t_fp8: torch.Tensor,
        fc2_bias: torch.Tensor,
        use_bias: bool,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        tp_group: Union[dist_group_type, None],
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        return_layernorm_output: bool,
        bias_gelu_nvfusion: bool,
        set_parallel_mode: bool,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        ln_bias = cast_if_needed(ln_bias, activation_dtype)

        # If residual connection is after LN, we need `ln_out`
        # tensor in higher precision, this comes at the cost
        # of an extra fp8 cast.
        if fp8:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            if not return_layernorm_output:
                ln_out, mu, rsigma = layernorm_fwd_fp8(
                    inputmat,
                    ln_weight,
                    ln_bias,
                    eps,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                )
            else:
                ln_out_return, mu, rsigma = tex.layernorm_fwd(
                    inputmat, ln_weight, ln_bias, eps
                )
                ln_out = cast_to_fp8(
                    ln_out_return,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                )
        else:
            ln_out, mu, rsigma = tex.layernorm_fwd(inputmat, ln_weight, ln_bias, eps)
            ln_out_return = ln_out

        # Column Parallel Linear
        if set_parallel_mode and sequence_parallel:
            ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
        else:
            ln_out_total = ln_out

        if fp8:
            bias_dtype = (
                torch.bfloat16
                if activation_dtype == torch.float32
                else activation_dtype
            )
            fc1_bias = cast_if_needed(fc1_bias, bias_dtype)
            fc2_bias = cast_if_needed(fc2_bias, bias_dtype) if use_bias else fc2_bias

            if update_fp8_weights:
                fp8_cast_transpose_fused(
                    fc1_weight,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_WEIGHT,
                    fp8_dtype_forward,
                    cast_out=fc1_weight_fp8,
                    transpose_out=fc1_weight_t_fp8,
                )

                fp8_cast_transpose_fused(
                    fc2_weight,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM2_WEIGHT,
                    fp8_dtype_forward,
                    cast_out=fc2_weight_fp8,
                    transpose_out=fc2_weight_t_fp8,
                )

            fc1_out = fp8_gemm(
                fc1_weight_fp8,
                fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_WEIGHT],
                fp8_dtype_forward,
                ln_out_total,
                fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
                fp8_dtype_forward,
                activation_dtype,
                get_workspace(),
                bias=fc1_bias,
                use_bias=True,
                use_split_accumulator=_2X_ACC_FPROP,
            )

            gelu_out = fp8_gelu(
                fc1_out,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM2_INPUT,
                fp8_dtype_forward,
            )

            fc2_out = fp8_gemm(
                fc2_weight_fp8,
                fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM2_WEIGHT],
                fp8_dtype_forward,
                gelu_out,
                fp8_meta["scaling_fwd"].scale_inv[tex.FP8FwdTensors.GEMM2_INPUT],
                fp8_dtype_forward,
                activation_dtype,
                get_workspace(),
                bias=fc2_bias,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
            )
        else:
            # Cast for native AMP
            fc1_weight = cast_if_needed(fc1_weight, activation_dtype)
            fc2_weight = cast_if_needed(fc2_weight, activation_dtype)
            fc1_bias = cast_if_needed(fc1_bias, activation_dtype)
            fc2_bias = (
                cast_if_needed(fc2_bias, activation_dtype) if use_bias else fc2_bias
            )

            fc1_outputs = gemm(
                fc1_weight,
                ln_out_total,
                activation_dtype,
                get_workspace(),
                bias=fc1_bias,
                use_bias=not bias_gelu_nvfusion,
                gelu=not bias_gelu_nvfusion,
            )

            if bias_gelu_nvfusion:
                fc1_out, _, _ = fc1_outputs
                gelu_out = bias_gelu_fused(fc1_out, fc1_bias)
            else:
                gelu_out, _, fc1_out = fc1_outputs

            fc2_out, _, _ = gemm(
                fc2_weight,
                gelu_out,
                activation_dtype,
                get_workspace(),
                bias=fc2_bias,
                use_bias=use_bias,
            )

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
            fc1_bias,
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
        ctx.tp_group = tp_group
        ctx.bias_gelu_nvfusion = bias_gelu_nvfusion
        ctx.return_layernorm_output = return_layernorm_output
        ctx.set_parallel_mode = set_parallel_mode

        # Row Parallel Linear
        if set_parallel_mode and sequence_parallel:
            fc2_out, _ = reduce_scatter_along_first_dim(fc2_out, tp_group)
        elif set_parallel_mode and tensor_parallel:
            fc2_out, _ = allreduce(fc2_out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        fc2_out = fc2_out.view(-1, *inp.shape[1:-1], fc2_out.shape[-1])

        if return_layernorm_output:
            return fc2_out, ln_out_return.view_as(inp)
        return fc2_out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        TransformerEngineBaseModule.pre_backward(ctx.fp8, ctx.fp8_meta)

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
            fc1_bias,
            fwd_scale_inverses,
        ) = ctx.saved_tensors

        (
            grad_output,
            grad_output_c,
            grad_output_t,
            fc2_bias_grad,
        ) = TransformerEngineBaseModule.grad_output_preprocess(
            ctx, grad_outputs[0], True
        )

        # Column Parallel Linear
        # Overlap input AG with dgrad
        if ctx.set_parallel_mode and ctx.sequence_parallel:
            ln_out_total, handle = gather_along_first_dim(
                ln_out, ctx.tp_group, async_op=True
            )
        else:
            ln_out_total = ln_out

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

            # FC2 DGRAD
            fc2_dgrad = fp8_gemm(
                fc2_weight_t_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM2_WEIGHT],
                fp8_dtype_forward,
                grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT1],
                fp8_dtype_backward,
                ctx.activation_dtype,
                get_workspace(),
                use_split_accumulator=_2X_ACC_DGRAD,
            )

            # FC2 WGRAD
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                gelu_out_t = tex.fp8_transpose(gelu_out, fp8_dtype_forward)

                fc2_wgrad = fp8_gemm(
                    gelu_out_t,
                    fwd_scale_inverses[tex.FP8FwdTensors.GEMM2_INPUT],
                    fp8_dtype_forward,
                    grad_output_t,
                    ctx.fp8_meta["scaling_bwd"].scale_inv[
                        tex.FP8BwdTensors.GRAD_OUTPUT1
                    ],
                    fp8_dtype_backward,
                    ctx.activation_dtype,
                    get_workspace(),
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    fp32_output=ctx.fuse_wgrad_accumulation,
                    out=fc2_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    use_split_accumulator=_2X_ACC_WGRAD,
                )

                fc1_bias_grad, dgelu, dgelu_t = fp8_cast_transpose_bgrad_dgelu_fused(
                    fc2_dgrad,
                    fc1_out,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    fp8_dtype_backward,
                )
            else:
                gelu_out_c = cast_from_fp8(
                    gelu_out,
                    ctx.fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM2_INPUT,
                    fp8_dtype_forward,
                    TE_DType[ctx.activation_dtype],
                )
                fc2_wgrad, _, _ = gemm(
                    gelu_out_c,
                    grad_output,
                    ctx.activation_dtype,
                    get_workspace(),
                    layout="NT",
                    grad=True,
                    use_bias=ctx.use_bias,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    fp32_output=ctx.fuse_wgrad_accumulation,
                    out=fc2_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                )

                fc1_bias_grad, dgelu_no_fp8 = bgrad_dgelu_fused(
                    fc2_dgrad, fc1_out, fc1_bias
                )
                dgelu = cast_to_fp8(
                    dgelu_no_fp8,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    fp8_dtype_backward,
                )
                dgelu_t = None

            # FC1 DGRAD
            fc1_dgrad = fp8_gemm(
                fc1_weight_t_fp8,
                fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_WEIGHT],
                fp8_dtype_forward,
                dgelu,
                ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT2],
                fp8_dtype_backward,
                ctx.activation_dtype,
                get_workspace(),
                use_split_accumulator=_2X_ACC_DGRAD,
            )
        else:
            # FC2 DGRAD
            fc2_dgrad, _, _ = gemm(
                fc2_weight,
                grad_output,
                ctx.activation_dtype,
                get_workspace(),
                layout="NN",
                gelu=not ctx.bias_gelu_nvfusion,
                grad=True,
                gelu_input=fc1_out,
            )

            # FC2 WGRAD
            fc2_wgrad, fc2_bias_grad, _ = gemm(
                gelu_out,
                grad_output,
                ctx.activation_dtype,
                get_workspace(),
                layout="NT",
                grad=True,
                use_bias=ctx.use_bias,
                accumulate=accumulate_wgrad_into_param_main_grad,
                fp32_output=ctx.fuse_wgrad_accumulation,
                out=fc2_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
            )

            if ctx.bias_gelu_nvfusion:
                fc1_bias_grad, dgelu = bgrad_dgelu_fused(fc2_dgrad, fc1_out, fc1_bias)
            else:
                dgelu = fc2_dgrad

            # FC1 DGRAD
            fc1_dgrad, _, _ = gemm(
                fc1_weight,
                dgelu,
                ctx.activation_dtype,
                get_workspace(),
                layout="NN",
                grad=True,
            )

        # Overlap dgrad-RS/AR with wgrad
        if ctx.set_parallel_mode and ctx.sequence_parallel:
            handle.wait()
            fc1_dgrad, handle = reduce_scatter_along_first_dim(
                fc1_dgrad, ctx.tp_group, async_op=True
            )
        elif ctx.set_parallel_mode and ctx.tensor_parallel:
            fc1_dgrad, handle = allreduce(fc1_dgrad, ctx.tp_group, async_op=True)

        if ctx.fp8:
            # FC1 WGRAD
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                ln_out_total_t = tex.fp8_transpose(ln_out_total, fp8_dtype_forward)
                fc1_wgrad = fp8_gemm(
                    ln_out_total_t,
                    fwd_scale_inverses[tex.FP8FwdTensors.GEMM1_INPUT],
                    fp8_dtype_forward,
                    dgelu_t,
                    ctx.fp8_meta["scaling_bwd"].scale_inv[
                        tex.FP8BwdTensors.GRAD_OUTPUT2
                    ],
                    fp8_dtype_backward,
                    ctx.activation_dtype,
                    get_workspace(),
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    fp32_output=ctx.fuse_wgrad_accumulation,
                    out=fc1_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    use_split_accumulator=_2X_ACC_WGRAD,
                )
            else:
                ln_out_total_c = cast_from_fp8(
                    ln_out_total,
                    ctx.fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                    TE_DType[ctx.activation_dtype],
                )
                fc1_wgrad, _, _ = gemm(
                    ln_out_total_c,
                    dgelu_no_fp8,
                    ctx.activation_dtype,
                    get_workspace(),
                    layout="NT",
                    grad=True,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    fp32_output=ctx.fuse_wgrad_accumulation,
                    out=fc1_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                )
        else:
            # FC1 WGRAD
            fc1_wgrad_outputs = gemm(
                ln_out_total,
                dgelu,
                ctx.activation_dtype,
                get_workspace(),
                layout="NT",
                grad=True,
                use_bias=not ctx.bias_gelu_nvfusion,
                accumulate=accumulate_wgrad_into_param_main_grad,
                fp32_output=ctx.fuse_wgrad_accumulation,
                out=fc1_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
            )

            if ctx.bias_gelu_nvfusion:
                fc1_wgrad, _, _ = fc1_wgrad_outputs
            else:
                fc1_wgrad, fc1_bias_grad, _ = fc1_wgrad_outputs

        # Column Parallel Linear
        if ctx.set_parallel_mode and ctx.tensor_parallel and handle is not None:
            handle.wait()

        # LayerNorm gradient
        d_ln_out = fc1_dgrad.view(inputmat.shape)

        # Residual gradient
        if ctx.return_layernorm_output:
            d_ln_out = d_ln_out + grad_outputs[1].view_as(d_ln_out)

        dxmat, dgamma, dbeta = tex.layernorm_bwd(
            d_ln_out, inputmat, mu, rsigma, ln_weight
        )

        if not ctx.use_bias:
            fc2_bias_grad = None

        TransformerEngineBaseModule.post_backward(
            ctx.fp8, ctx.fp8_meta, ctx.sequence_parallel, ctx.tp_group
        )

        return (
            dxmat.view(ctx.inp_shape),
            dgamma,
            dbeta,
            fc1_wgrad,
            None,
            None,
            fc1_bias_grad,
            fc2_wgrad,
            None,
            None,
            fc2_bias_grad,
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


class LayerNormMLP(TransformerEngineBaseModule):
    """
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
    bias : bool, default = `True`
          if set to `False`, the FC2 layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing FC1 weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing FC2 weights in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module
                             is taken post layernorm.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, FC1 is used as Column Parallel and FC2 is used as Row
                      Parallel as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
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

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.float32`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    seq_length: int
               sequence length of input samples. Needed for JIT Warmup, a technique where jit fused
               functions are warmed up before training to ensure same kernels are used for forward
               propogation and activation recompute phase.
    micro_batch_size: int
                     batch size per training step. Needed for JIT Warmup, a technique where jit
                     fused functions are warmed up before training to ensure same kernels are
                     used for forward propogation and activation recompute phase.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        return_bias: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        output_layer_init_method: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        params_dtype: torch.dtype = torch.float32,
        return_layernorm_output: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        set_parallel_mode: bool = False,
    ) -> None:
        super().__init__()

        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.return_layernorm_output = return_layernorm_output
        self.bias_gelu_nvfusion = bool(int(os.getenv("NVTE_BIAS_GELU_NVFUSION", "1")))
        self.set_parallel_mode = set_parallel_mode

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        self.size_per_partition = divide(ffn_hidden_size, self.tp_size)

        # LN init
        self.eps = eps
        self.layer_norm_weight = Parameter(
            torch.empty(
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.layer_norm_bias = Parameter(
            torch.empty(
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
        setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)
        self.reset_layer_norm_parameters()

        # FC1 init
        self.fc1_weight = Parameter(
            torch.empty(
                self.size_per_partition,
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.fp8_weight_shapes.append(self.fc1_weight.shape)

        initialize_affine_weight_gpu(
            self.fc1_weight,
            init_method,
            get_rng_state_tracker,
            partition_dim=0,
            stride=1,
        )

        self.fc1_bias = Parameter(
            torch.empty(
                self.size_per_partition,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        set_tensor_model_parallel_attributes(self.fc1_bias, True, 0, 1)

        with torch.no_grad():
            self.fc1_bias.zero_()

        # FC2 init
        self.fc2_weight = Parameter(
            torch.empty(
                hidden_size,
                self.size_per_partition,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.fp8_weight_shapes.append(self.fc2_weight.shape)

        initialize_affine_weight_gpu(
            self.fc2_weight,
            output_layer_init_method,
            get_rng_state_tracker,
            partition_dim=1,
            stride=1,
        )

        if self.use_bias or self.return_bias:
            self.fc2_bias = Parameter(
                torch.empty(
                    hidden_size, device=torch.cuda.current_device(), dtype=params_dtype
                )
            )
        else:
            self.register_buffer("fc2_bias", torch.Tensor(), persistent=False)

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.set_parallel_mode and self.use_bias:
            self.gemm_bias_unfused_add = True
            self.use_bias = False
        else:
            self.gemm_bias_unfused_add = False

        with torch.no_grad():
            self.fc2_bias.zero_()

        if self.bias_gelu_nvfusion:
            set_jit_fusion_options()
            if seq_length and micro_batch_size:
                warmup_jit_bias_gelu_all_dtypes(
                    self.size_per_partition, seq_length, micro_batch_size
                )

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        init.ones_(self.layer_norm_weight)
        init.zeros_(self.layer_norm_bias)

    def forward(
        self, inp: torch.Tensor, is_first_microbatch: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a feedforward network (MLP Block).

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

        self.pre_forward(inp, num_gemms=2)

        out = _LayerNormMLP.apply(
            inp,
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.fc1_weight,
            self.weight1_fp8,
            self.weight1_t_fp8,
            self.fc1_bias,
            self.fc2_weight,
            self.weight2_fp8,
            self.weight2_t_fp8,
            self.fc2_bias,
            False,  # use_bias set to False for RPL
            self.eps,
            is_first_microbatch,
            self.fp8,
            self.fp8_meta,
            self.fuse_wgrad_accumulation,
            self.tp_group,
            self.sequence_parallel,
            self.tp_size > 1,
            self.activation_dtype,
            self.return_layernorm_output,
            self.bias_gelu_nvfusion,
            self.set_parallel_mode,
        )

        self.post_forward()

        if self.return_layernorm_output:
            out, ln_out = out

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(self.fc2_bias, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(self.fc2_bias, self.activation_dtype), ln_out
            return out, cast_if_needed(self.fc2_bias, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out


class _LayerNorm(torch.autograd.Function):
    """functional LayerNorm"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert inp.shape[-1] == in_features, "LayerNorm not possible"
        inputmat = inp.view((-1, in_features))

        ln_out, mu, rsigma = tex.layernorm_fwd(inputmat, ln_weight, ln_bias, eps)
        ctx.save_for_backward(inputmat, ln_weight, mu, rsigma)
        ctx.inp_shape = inp.shape
        return ln_out.view_as(inp)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        inputmat, ln_weight, mu, rsigma = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        d_ln_out = grad_output.view(inputmat.shape)
        dxmat, dgamma, dbeta = tex.layernorm_bwd(
            d_ln_out, inputmat, mu, rsigma, ln_weight
        )
        return dxmat.view(ctx.inp_shape), dgamma, dbeta, None


class LayerNorm(torch.nn.Module):
    r"""
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    size :attr:`hidden_size`

    Parameters
    ----------
    hidden_size : int
                size of each input sample.
    eps : float, default = 1e-5
        a value added to the denominator of layer normalization for numerical stability.
    sequence_parallel : bool, default = `False`
                        if set to `True`, uses sequence parallelism.
    params_dtype : torch.dtype, default = `torch.float32`
                    it controls the type used to allocate the initial parameters. Useful when
                    the model is trained with lower precision and the original FP32 parameters
                    would not fit in GPU memory.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        params_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.layer_norm_weight = Parameter(
            torch.empty(
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.layer_norm_bias = Parameter(
            torch.empty(
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        setattr(self.layer_norm_weight, "sequence_parallel", sequence_parallel)
        setattr(self.layer_norm_bias, "sequence_parallel", sequence_parallel)
        self.reset_layer_norm_parameters()

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        init.ones_(self.layer_norm_weight)
        init.zeros_(self.layer_norm_bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """LayerNorm FWD"""
        return _LayerNorm.apply(
            inp, self.layer_norm_weight, self.layer_norm_bias, self.eps
        )
