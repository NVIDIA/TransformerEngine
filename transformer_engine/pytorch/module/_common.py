# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Internal function used by multiple modules."""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import queue

from .. import cpp_extensions as tex
from ..export import is_in_onnx_export_mode
from ..fp8 import get_fp8_te_dtype
from ..utils import (
    clear_tensor_data,
    get_default_init_method,
)

def _get_normalization_func(
    normalization: str, fp8_output: bool, is_grad_enabled: bool, forward: bool
):
    fwd_normalization_funcs = {
        ("LayerNorm", True, True): tex.layernorm_fwd_fp8,
        ("LayerNorm", True, False): tex.layernorm_fwd_fp8_inf,
        ("LayerNorm", False, True): tex.layernorm_fwd_noalloc,
        ("LayerNorm", False, False): tex.layernorm_fwd_inf,
        ("RMSNorm", True, True): tex.rmsnorm_fwd_fp8,
        ("RMSNorm", True, False): tex.rmsnorm_fwd_fp8_inf,
        ("RMSNorm", False, True): tex.rmsnorm_fwd_noalloc,
        ("RMSNorm", False, False): tex.rmsnorm_fwd_inf,
    }
    bwd_normalization_funcs = {
        "LayerNorm": tex.layernorm_bwd,
        "RMSNorm": tex.rmsnorm_bwd,
    }

    if forward:
        return fwd_normalization_funcs[(normalization, fp8_output, is_grad_enabled)]
    assert not fp8_output, "FP8 output is not supported in backward normalization!"
    assert is_grad_enabled, "Gradient has to be enabled to call backward normalization!"
    return bwd_normalization_funcs[normalization]


def _apply_normalization(
    inputmat: torch.Tensor,
    ln_out: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: Union[torch.Tensor, None],
    eps: float,
    fp8_out: bool,
    fp8_meta: Dict[str, Any],
    normalization: str,
    fwd_ln_sm_margin: int,
    zero_centered_gamma: bool,
    is_grad_enabled: bool,
    fp8_scale: Optional[torch.Tensor] = None,
    fp8_amax: Optional[torch.Tensor] = None,
    fp8_scale_inv: Optional[torch.Tensor] = None,
):
    normalization_func = _get_normalization_func(normalization, fp8_out, is_grad_enabled, True)

    inputs = (inputmat, ln_weight) if ln_bias is None else (inputmat, ln_weight, ln_bias)
    if fp8_out:
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        if is_grad_enabled:
            output_key = "ln_out" if normalization == "LayerNorm" else "rmsnorm_out"
            output_kwarg = {output_key: ln_out}
            output = normalization_func(
                *inputs,
                eps,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                fwd_ln_sm_margin,
                zero_centered_gamma,
                scale=fp8_scale,
                amax=fp8_amax,
                scale_inv=fp8_scale_inv,
                **output_kwarg,
            )
        else:
            return (
                normalization_func(
                    *inputs,
                    eps,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                    fwd_ln_sm_margin,
                    zero_centered_gamma,
                    scale=fp8_scale,
                    amax=fp8_amax,
                    scale_inv=fp8_scale_inv,
                ),
                None,
                None,
            )
    else:
        if is_grad_enabled:
            output = normalization_func(*inputs, ln_out, eps, fwd_ln_sm_margin, zero_centered_gamma)
        else:
            return (
                normalization_func(*inputs, eps, fwd_ln_sm_margin, zero_centered_gamma),
                None,
                None,
            )
    if normalization == "RMSNorm":
        output = (ln_out, None, output[1])
    elif normalization == "LayerNorm":
        output = (ln_out, output[1], output[2])
    return output


class _NoopCatFunc(torch.autograd.Function):
    """Concatenate tensors, doing a no-op if possible

    See _noop_cat.

    """

    @staticmethod
    def forward(
        ctx: Any,
        dim: int,
        *tensors: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Check first tensor
        if not tensors:
            raise ValueError("Attempted to concatenate 0 tensors")
        num_dims = tensors[0].dim()
        if not -num_dims <= dim < num_dims:
            raise ValueError(
                "Attempted to concatenate tensor "
                f"with shape {list(tensors[0].size())} along dim {dim}"
            )
        dim %= num_dims

        # Check remaining tensors
        out_shape = list(tensors[0].size())
        split_ranges = [(0, tensors[0].size(dim))]
        for tensor in tensors[1:]:
            in_shape = list(tensor.size())
            if (
                len(in_shape) != num_dims
                or in_shape[:dim] != out_shape[:dim]
                or in_shape[dim + 1 :] != out_shape[dim + 1 :]
            ):
                raise ValueError(
                    "Attempted to concatenate tensors with shapes "
                    f"{[list(tensor.size()) for tensor in tensors]} "
                    f"along dim {dim}"
                )
            split_start = out_shape[dim]
            split_end = split_start + in_shape[dim]
            out_shape[dim] = split_end
            split_ranges.append((split_start, split_end))

        # Save state for backward
        ctx.dim = dim
        ctx.split_ranges = split_ranges

        # Out-of-place concatenation if needed
        dtype = tensors[0].dtype
        device = tensors[0].device
        strides = tensors[0].stride()
        data_ptr_stride = strides[dim] * tensors[0].element_size()
        data_ptr = tensors[0].data_ptr() + tensors[0].size(dim) * data_ptr_stride
        for tensor in tensors[1:]:
            if (
                tensor.dtype != dtype
                or tensor.device != device
                or tensor.stride() != strides
                or tensor.data_ptr() != data_ptr
            ):
                return torch.cat(tensors, dim=dim)
            data_ptr += tensor.size(dim) * data_ptr_stride

        # No-op concatenation
        out = tensors[0].new()
        out.set_(
            tensors[0].untyped_storage(),
            tensors[0].storage_offset(),
            out_shape,
            strides,
        )
        out.requires_grad = any(tensor.requires_grad for tensor in tensors)
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        grad_inputs = []
        for split_start, split_end in ctx.split_ranges:
            slices = [slice(None)] * grad_output.dim()
            slices[ctx.dim] = slice(split_start, split_end)
            grad_inputs.append(grad_output[tuple(slices)])
        return None, *grad_inputs


def _noop_cat(
    tensors: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """Concatenate tensors, doing a no-op if possible

    If tensors are already concatenated in memory, a tensor view of
    that memory region will be returned. Otherwise the tensors will be
    concatenated out-of-place, as usual.

    """
    if not tensors:
        raise ValueError("Attempted to concatenate 0 tensors")
    if len(tensors) == 1:
        return tensors[0]
    if is_in_onnx_export_mode():
        return torch.cat(tensors, dim=dim)
    return _NoopCatFunc.apply(dim, *tensors)


@dataclass
class _ParameterInitMeta:
    """
    Stores essential metadata needed to support deferred parameter initialization.
    """

    init_fn: Optional[Callable] = get_default_init_method()
    get_rng_state_tracker: Optional[Callable] = None
    fp8_meta_index: Optional[int] = None

    def __post_init__(self):
        """Safeguard reference to the parameter's parent module and initialization function."""
        if self.init_fn is None:
            self.init_fn = get_default_init_method()

class WeightGradStore:

    should_split_bw = True
    cache = []
    weight_grad_queue = None  # lazy init

    @classmethod
    def lazy_init(cls):
        if cls.weight_grad_queue is not None:
            return
        # Lazy init to make sure parallel_state and get_args() have been initialized.
        # num_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        num_chunks = 1
        # chunk id => Queue
        cls.weight_grad_queue = [queue.Queue() for _ in range(num_chunks)]

    @classmethod
    def is_supported(cls):
        """If not supported, fallback to original schedule."""
        # args = get_args()
        # if args.pipeline_model_parallel_size <= 1:
        #     return False
        # # if args.virtual_pipeline_model_parallel_size is not None:
        # #     return False
        # if args.overlap_grad_reduce:
        #     # the logic of overlapping grad reduce should be changed
        #     return False
        # if not args.gradient_accumulation_fusion:
        #     return False
        # if args.transformer_impl == 'transformer_engine':
        #     # hard to capture weight gradient computation for transformer_engine
        #     return False
        return True

    @classmethod
    def split_bw(cls):
        if not cls.is_supported():
            return False
        return cls.should_split_bw

    @classmethod
    def enable_split_bw(cls):
        cls.should_split_bw = True

    @classmethod
    def disable_split_bw(cls):
        cls.should_split_bw = False

    @classmethod
    @contextmanager
    def set_split_bw(cls, enabled: bool):
        prev = cls.should_split_bw
        cls.should_split_bw = enabled
        try:
            yield
        finally:
            cls.should_split_bw = prev

    @classmethod
    def put(cls, inp, grad_out, dW, func):
        assert cls.split_bw()
        # func(*pre_func(async_op=False))
        cls.cache.append((inp, grad_out, dW, func))
        return

    @classmethod
    def queue_size(cls, chunk=0):
        cls.lazy_init()
        return WeightGradStore.weight_grad_queue[chunk].qsize()

    @classmethod
    def flush(cls, chunk=0):
        cls.lazy_init()
        # Or W later will consume empty computation and leak the non-empty computation.
        if not cls.split_bw():
            assert len(cls.cache) == 0
            return
        cls.weight_grad_queue[chunk].put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls, chunk=0):
        cls.lazy_init()
        if cls.weight_grad_queue[chunk].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk].get()
            for inp, grad_out, dW, func in stored_grads:
                func(inp, grad_out, dW)
        else:
            # rank = parallel_state.get_pipeline_model_parallel_rank()
            rank = torch.distributed.get_rank()
            raise Exception(f"Pop empty queue. rank {rank}")

    @classmethod
    def assert_empty(cls):
        rank = parallel_state.get_pipeline_model_parallel_rank()
        assert len(cls.cache) == 0, f"cache is not empty. rank {rank}"
        if cls.weight_grad_queue is None:
            return
        for chunk, chunk_q in enumerate(cls.weight_grad_queue):
            assert chunk_q.empty(), f"Queue is not empty chunk {chunk} rank {rank}. len {chunk_q.qsize()}"

    @classmethod
    def clear(cls, chunk=0):
        cls.lazy_init()
        weight_grad_tasks = []
        while cls.weight_grad_queue[chunk].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk].get()
            if len(weight_grad_tasks) == 0:
                for _ in stored_grads:
                    weight_grad_tasks.append([])
            else:
                assert len(weight_grad_tasks) == len(stored_grads)
            for i, task in enumerate(stored_grads):
                weight_grad_tasks[i].append(task)
        # timers = get_timers()
        # weight_params = []
        # handles = []
        # if get_args().overlap_grad_reduce:
        #     handles += model.async_reduce_grad()

        # config = get_model_config(model)
        # # Do async all-reduce for embedding grads firstly, so that the rank 0 won't
        # # be blocked
        # embedding_handles = _allreduce_embedding_grads([model], config, async_op=True)
        # handles += embedding_handles

        for i in range(len(weight_grad_tasks)):
            tasks = weight_grad_tasks[i]
            param = None
            for j in range(len(tasks)):
                inp, grad_out, dW, func = tasks[j]
                if param is None:
                    param = weight
                assert param.storage().data_ptr() == weight.storage().data_ptr()
                func(inp, grad_out, dW)
                tasks[j] = None  # release memory
            # weight_params.append(param)
            # if get_args().overlap_grad_reduce:
            #     # All-reduce param grad here
            #     handles += model.async_reduce_grad(param)
            weight_grad_tasks[i] = None  # release memory

        # timers('wait_all_reduce', log_level=1).start(barrier=False)
        # for handle in handles:
        #     if handle is not None:
        #         handle.wait()
        # timers('wait_all_reduce').stop()