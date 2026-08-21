# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functions for CUDA Graphs support in FP8"""

from bisect import bisect_right
from collections.abc import Iterable
import contextlib
import gc
import os
import warnings
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
from torch.utils._pytree import tree_flatten as _tree_flatten
from torch.utils._pytree import tree_unflatten as _tree_unflatten
from torch._C import _graph_pool_handle
import transformer_engine_torch as tex

from transformer_engine.common.recipe import DelayedScaling, Recipe
from transformer_engine.pytorch.constants import dist_group_type
from .quantization import (
    autocast,
    FP8GlobalStateManager,
    get_default_fp8_recipe,
)
from .distributed import get_all_rng_states, graph_safe_rng_available
from .module.base import TransformerEngineBaseModule, get_dummy_wgrad
from .ops.op import BasicOperation
from .ops import Sequential
from .ops.fuser import OperationFuser
from .utils import make_weak_ref

__all__ = ["make_graphed_callables"]


_IS_GRAPH_CAPTURING = False

_T = TypeVar("_T")
SingleOrTuple = Union[_T, Tuple[_T, ...]]


class _AllocatorSettingsGuard:
    """Restore temporary allocator settings even when graph capture fails."""

    def __init__(self) -> None:
        self._setter = None
        self._settings_to_restore = None

    def apply(self, setter: Callable[[str], None], settings: str, restore: str) -> None:
        """Apply temporary allocator settings and remember how to restore them."""
        if self._setter is not None:
            raise RuntimeError("CUDA allocator settings guard is already active.")
        self._setter = setter
        self._settings_to_restore = restore
        setter(settings)

    def restore(self) -> None:
        """Restore the allocator settings saved by :meth:`apply`."""
        if self._setter is None:
            return
        setter = self._setter
        settings = self._settings_to_restore
        assert settings is not None
        setter(settings)
        self._setter = None
        self._settings_to_restore = None


def _tensor_storage_ptr(tensor: torch.Tensor) -> int:
    """Return the base storage pointer used to recognize static graph inputs."""
    return tensor.untyped_storage().data_ptr()


def _tensor_version(tensor: torch.Tensor) -> Optional[int]:
    """Return the mutation version when the tensor tracks one."""
    try:
        return tensor._version
    except RuntimeError:
        return None


def _saved_tensor_signature(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Describe the layout needed to reproduce a tensor in a static arena."""
    if tensor.layout != torch.strided:
        raise RuntimeError(
            "CUDA graph saved-tensor arenas only support strided tensors, "
            f"but got layout={tensor.layout}."
        )
    if any(stride < 0 for stride in tensor.stride()):
        raise RuntimeError("CUDA graph saved-tensor arenas do not support negative strides.")

    if tensor.numel() == 0:
        storage_numel = 0
    else:
        storage_numel = 1 + sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.requires_grad,
        storage_numel * tensor.element_size(),
    )


def _input_staging_key(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Describe user inputs that can use one forward-only staging surface."""
    return (tensor.layout, tensor.storage_offset(), *_saved_tensor_signature(tensor))


def _align_up(value: int, alignment: int = 256) -> int:
    """Align byte offsets for typed tensor views into a uint8 arena."""
    return (value + alignment - 1) // alignment * alignment


def _io_tensor_plan(tensor: Any, kind: str) -> Optional[Tuple[Any, ...]]:
    """Return an arena plan for plain CUDA tensors exposed across graph boundaries."""
    if (
        tensor.__class__ is not torch.Tensor
        or not tensor.is_cuda
        or tensor.layout != torch.strided
        or any(stride < 0 for stride in tensor.stride())
    ):
        return None
    return (kind, None, *_saved_tensor_signature(tensor))


def _arena_view(arena: torch.Tensor, offset: int, spec: Tuple[Any, ...]) -> torch.Tensor:
    """Materialize a typed tensor view at a byte offset in an arena."""
    target = torch.empty((0,), dtype=spec[4], device=spec[5])
    return target.set_(
        arena.untyped_storage(),
        (arena.storage_offset() + offset) // target.element_size(),
        spec[2],
        spec[3],
    )


def set_capture_start() -> None:
    """Record beginning of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = True


def set_capture_end() -> None:
    """Record end of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = False


def is_graph_capturing() -> bool:
    """Return whether within `make_graphed_callables`."""
    return _IS_GRAPH_CAPTURING


def graph_pool_handle():
    """
    Returns an opaque token representing the id of a graph memory pool.
    """
    return _graph_pool_handle()


@contextlib.contextmanager
def _none_grad_context_wrapper(inputs):
    """
    Wrapper to set the gradients of the inputs to None,
    in case the backward pass makes grad accumulations.
    """
    original_input_grads = []
    try:
        for input_tensor in inputs:
            original_input_grads.append(input_tensor.grad)
            input_tensor.grad = None
        yield
    finally:
        for input_tensor, original_grad in zip(inputs, original_input_grads):
            input_tensor.grad = original_grad


@contextlib.contextmanager
def _graph_context_wrapper(*args, **kwargs):
    """Wrapper around `torch.cuda.graph`.

    This wrapper is a temporary workaround for a PyTorch bug:
    automatic garbage collection can destroy a graph while another
    graph is being captured, resulting in a CUDA error. See
    https://github.com/pytorch/pytorch/pull/161037.

    """
    gc_is_enabled = gc.isenabled()
    if gc_is_enabled:
        gc.disable()
    try:
        with torch.cuda.graph(*args, **kwargs):
            yield
    finally:
        if gc_is_enabled:
            gc.enable()


@contextlib.contextmanager
def _module_forward_hooks(modules, hook_fn):
    """Remove temporary warmup hooks even when a module raises."""
    hooks = []
    try:
        for module in modules:
            hooks.append(module.register_forward_hook(hook_fn))
        yield
    finally:
        for hook in reversed(hooks):
            hook.remove()


def _make_graphed_callables(
    callables: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    cache_quantized_params: bool = False,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    _order: Optional[List[int]] = None,
    _num_layers_per_chunk: Optional[List[int]] = None,
    pool: Optional[Tuple[int, ...]] = None,
    retain_graph_in_backward: bool = False,
    _reuse_graph_input_output_buffers: bool = False,
    _graph_memory_slots: Optional[Sequence[Tuple[int, ...]]] = None,
    _allocator_settings_guard: Optional[_AllocatorSettingsGuard] = None,
    pre_warmup_hook: Optional[Callable] = None,
    post_warmup_hook: Optional[Callable] = None,
) -> SingleOrTuple[Callable]:
    """
    Helper method for `make_graphed_callables`
    """

    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast "
            "caching. Please set `cache_enabled=False`."
        )

    # Default is to pass no kwargs to callables
    if sample_kwargs is None:
        if isinstance(callables, tuple):
            sample_kwargs = tuple({} for _ in range(len(sample_args)))
        else:
            sample_kwargs = {}

    # Canonicalize args as tuples
    just_one_callable = False
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)
        sample_kwargs = (sample_kwargs,)

    # Check training/inference
    is_training = all(c.training for c in callables)
    if not is_training and any(c.training for c in callables):
        raise RuntimeError(
            "make_graphed_callables only supports when modules are all in training or all in"
            " inference mode."
        )

    # Check sizes of args
    _order_without_wgrad = None
    delay_wgrad_compute = False
    if _order is None:
        if len(sample_args) != len(callables):
            raise ValueError(
                "Expected sample_args to have the same length as callables, "
                f"but got {len(sample_args)} sample_args for {len(callables)} callables"
            )
        if len(sample_kwargs) != len(callables):
            raise ValueError(
                "Expected sample_kwargs to have the same length as callables, "
                f"but got {len(sample_kwargs)} sample_kwargs for {len(callables)} callables"
            )
    else:
        # Custom logic for interleaved pipeline parallelism
        # Note: This is tightly coupled with the Megatron-core
        # implementation of interleaved pipeline parallelism at
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py.
        # Note: The model is assumed to consist of layers
        # (corresponding to callables) that are grouped into
        # model chunks. _num_layers_per_chunk is a list of integers
        # that indicates the number of layers in each model chunk.
        # _order is a list of chunk indices (1-indexed) that
        # indicates the order in which the layers are evaluated.
        # Positive values indicate forward passes and negative
        # values indicate backward passes. Each
        # entry in sample_args corresponds to one of the forward
        # passes.
        _order_without_wgrad = []
        for c_id in _order:
            if ceil(c_id) != c_id:
                delay_wgrad_compute = True
                continue
            _order_without_wgrad.append(c_id)
        num_model_chunks = max(_order_without_wgrad)
        num_microbatches = len(_order_without_wgrad) // num_model_chunks // 2
        if num_model_chunks * num_microbatches * 2 != len(_order_without_wgrad):
            raise ValueError(
                f"Pipeline-parallel order dimension mismatch: num_model_chunks ({num_model_chunks})"
                f" * num_microbatches ({num_microbatches}) * 2 ="
                f" {num_model_chunks * num_microbatches * 2}, but len(_order_without_wgrad) ="
                f" {len(_order_without_wgrad)}"
            )

        # When delay_wgrad_compute is enabled, each layer is treated as a model chunk, which
        # allows for fine-grained graph capture order.
        if delay_wgrad_compute:
            if _num_layers_per_chunk is None:
                raise ValueError(
                    "'_num_layers_per_chunk' must be provided when delay_wgrad_compute is True."
                )
            for num_layers in _num_layers_per_chunk:
                if num_layers != 1:
                    raise ValueError(
                        "Each model chunk must have only one layer when delay_wgrad_compute is"
                        f" True, but got {num_layers} layers."
                    )

        # Determine number of layers in each model chunk.
        if _num_layers_per_chunk is None:
            if not (
                len(sample_args) * 2 >= len(_order_without_wgrad)
                and (len(sample_args) * 2 % len(_order_without_wgrad) == 0)
            ):
                raise ValueError(
                    f"{len(sample_args)} * 2 >= {len(_order_without_wgrad)} and"
                    f" {len(sample_args)} * 2 % {len(_order_without_wgrad)} == 0"
                )
            num_layers = len(sample_args) // num_model_chunks // num_microbatches
            _num_layers_per_chunk = [num_layers] * num_model_chunks
        else:
            if not (
                isinstance(_num_layers_per_chunk, int)
                or len(_num_layers_per_chunk) == num_model_chunks
            ):
                raise ValueError(
                    "If _num_layers_per_chunk is provided, it must be an integer or a list of"
                    f" {num_model_chunks} integers, but got {_num_layers_per_chunk}."
                )
            if isinstance(_num_layers_per_chunk, int):
                _num_layers_per_chunk = [_num_layers_per_chunk] * num_model_chunks
        total_num_layers = sum(_num_layers_per_chunk)
        if len(callables) != total_num_layers:
            raise ValueError(
                f"Callables should have ({total_num_layers}) "
                + f"entries when order input is provided but got {len(callables)}."
            )
        if len(sample_args) != total_num_layers * num_microbatches:
            raise ValueError(
                f"Expected {total_num_layers * num_microbatches} "
                + f"args tuple, but got {len(sample_args)}."
            )

        # Calculate the starting index of each chunk in callables for future use.
        _prefix_num_layers = [0]
        for m_chunk in range(num_model_chunks):
            num_layers = _num_layers_per_chunk[m_chunk]
            _prefix_num_layers.append(_prefix_num_layers[-1] + num_layers)

        if len(sample_kwargs) != len(sample_args):
            raise ValueError(
                "Pipeline-parallel schedule requires sample_kwargs and sample_args to have "
                f"the same length, but got {len(sample_kwargs)} sample_kwargs "
                f"for {len(sample_args)} sample_args"
            )

    use_slot_memory = _graph_memory_slots is not None
    if use_slot_memory:
        allocator_backend = torch.cuda.memory.get_allocator_backend()
        if allocator_backend != "native":
            raise RuntimeError(
                "CUDA graph slot-branch checkpointing requires the native CUDA caching "
                f"allocator, but the active backend is {allocator_backend!r}."
            )
        required_checkpoint_apis = (
            "_cuda_getCheckpointState",
            "_cuda_setCheckpointPoolState",
            "_cuda_checkPoolLiveAllocations",
            "_free_And_Remove_DeleterFn",
        )
        missing_checkpoint_apis = [
            name for name in required_checkpoint_apis if not hasattr(torch._C, name)
        ]
        if missing_checkpoint_apis:
            raise RuntimeError(
                "CUDA graph slot-branch checkpointing requires PyTorch allocator APIs "
                f"{missing_checkpoint_apis}."
            )
        if not hasattr(tex, "_graph_checkpoint_detach_storage"):
            raise RuntimeError(
                "CUDA graph slot-branch checkpointing requires "
                "transformer_engine_torch._graph_checkpoint_detach_storage."
            )

    _reuse_graph_input_buffers = _reuse_graph_input_output_buffers and not use_slot_memory
    if _reuse_graph_input_output_buffers:
        if _order is None:
            raise ValueError(
                "`_order` must be provided when `_reuse_graph_input_output_buffers` is True."
            )
        if not is_training:
            raise RuntimeError(
                "`_reuse_graph_input_output_buffers` is only available in training mode."
            )

    saved_tensor_arena_ids = None
    slot_io_memory_alias_groups = None
    slot_io_liveness_groups = None
    warmup_plan_alias_groups = None
    user_grad_arena_ids = None
    if use_slot_memory:
        if _order is None or not is_training or not _reuse_graph_input_output_buffers:
            raise RuntimeError(
                "Graph-memory slots require a training graph with `_order` and graph buffer reuse."
            )
        if pool is not None:
            raise ValueError("Graph-memory slots create and own their CUDA graph memory pool.")
        if not hasattr(torch.cuda, "MemPool"):
            raise RuntimeError("Graph-memory slots require torch.cuda.MemPool support.")
        if len(_graph_memory_slots) != len(sample_args):
            raise ValueError(
                f"Expected {len(sample_args)} graph-memory slots, got {len(_graph_memory_slots)}."
            )
        if any(
            not isinstance(slot, tuple)
            or len(slot) != 7
            or not all(isinstance(value, int) for value in slot)
            for slot in _graph_memory_slots
        ):
            raise TypeError("Each graph-memory slot must be a tuple of seven integers.")
        saved_tensor_arena_ids = [slot[0] for slot in _graph_memory_slots]
        slot_io_memory_alias_groups = [(slot[1], slot[2]) for slot in _graph_memory_slots]
        slot_io_liveness_groups = [(slot[3], slot[4]) for slot in _graph_memory_slots]
        warmup_plan_alias_groups = [slot[5] for slot in _graph_memory_slots]
        user_grad_arena_ids = [slot[6] for slot in _graph_memory_slots]

    # Check reuse graph conditions and reorganize sample_args and sample_kwargs.
    # Note: When capturing a graph, we hold onto the args and kwargs so we have static buffers
    # when the graph is replayed. If two model chunk microbatches have no overlap between their
    # forward and backward, then we can reduce memory usage by reusing the same static buffers.
    if _reuse_graph_input_buffers:
        if isinstance(sample_args, tuple):
            sample_args = list(sample_args)
        if isinstance(sample_kwargs, tuple):
            sample_kwargs = list(sample_kwargs)

        # Reorganize args and kwargs for input tensor reuse.
        # fwd_sample_qs is keyed by model chunk index. The value is a queue of tuples.
        # Each tuple contains the sample key signature and its fwd_idx. When we finish a backward
        # chunk, we pop the corresponding fwd_idx and push to the consumed_sample_q.
        # consumed_sample_q is keyed by the sample key signature. The value is a queue of the
        # fwd_idx whose backward has been called so that we can reuse the same static buffers.
        # In this way, we can reuse the same static input buffers for the non-overlapping samples
        # with the same input signature.
        fwd_sample_qs = {}
        consumed_sample_q = {}
        fwd_idx = [0] * num_model_chunks
        for c_id in _order:
            m_chunk = abs(ceil(c_id)) - 1

            if c_id > 0:
                sample_start_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                    fwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk]
                )
                fwd_sample_idx = [
                    sample_start_idx + i for i in range(_num_layers_per_chunk[m_chunk])
                ]
                if m_chunk not in fwd_sample_qs:
                    fwd_sample_qs[m_chunk] = []
                for per_callable_fwd_idx in fwd_sample_idx:
                    sample_args_keys = tuple(
                        (t.shape, t.dtype, t.layout) for t in sample_args[per_callable_fwd_idx]
                    )
                    sample_kwargs_keys = tuple(
                        (k, v.shape, v.dtype, v.layout)
                        for k, v in sorted(sample_kwargs[per_callable_fwd_idx].items())
                    )
                    sample_keys = sample_args_keys + sample_kwargs_keys

                    fwd_sample_qs[m_chunk].append((sample_keys, per_callable_fwd_idx))
                    if consumed_sample_q.get(sample_keys, []):
                        reuse_fwd_idx = consumed_sample_q[sample_keys].pop(0)
                        sample_args[per_callable_fwd_idx] = sample_args[reuse_fwd_idx]
                        sample_kwargs[per_callable_fwd_idx] = sample_kwargs[reuse_fwd_idx]
                fwd_idx[m_chunk] += 1
            elif ceil(c_id) != c_id:
                continue
            else:
                num_consumed_samples = min(
                    len(fwd_sample_qs[m_chunk]), _num_layers_per_chunk[m_chunk]
                )
                for sample_keys, per_callable_fwd_idx in fwd_sample_qs[m_chunk][
                    :num_consumed_samples
                ]:
                    if sample_keys not in consumed_sample_q:
                        consumed_sample_q[sample_keys] = []
                    consumed_sample_q[sample_keys].append(per_callable_fwd_idx)
                fwd_sample_qs[m_chunk] = fwd_sample_qs[m_chunk][num_consumed_samples:]

    if cache_quantized_params:
        # Initialize flag that controls FP8 weight updates
        FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(False)

    # Check callables
    for c in callables:
        if isinstance(c, torch.nn.Module):
            if not (
                len(c._backward_hooks) == 0
                and len(c._forward_hooks) == 0
                and len(c._forward_pre_hooks) == 0
            ):
                raise RuntimeError(
                    "Modules must not have hooks registered at the time they are passed. "
                    + "However, registering hooks on modules after passing them "
                    + "through make_graphed_callables is allowed."
                )
            if not all(b.requires_grad is False for b in c.buffers()):
                raise RuntimeError(
                    "In any :class:`~torch.nn.Module` passed to "
                    + ":func:`~make_graphed_callables`, only parameters may be trainable. "
                    + "All buffers must have ``requires_grad=False``."
                )

    # Flatten callable arguments
    per_callable_kwargs_keys = [list(kwargs.keys()) for kwargs in sample_kwargs]
    flatten_sample_args = []
    for args, kwargs, kwargs_keys in zip(sample_args, sample_kwargs, per_callable_kwargs_keys):
        flatten_arg, _ = _tree_flatten(args)
        flatten_kwarg, _ = _tree_flatten([kwargs[key] for key in kwargs_keys])
        flatten_sample_args.append(tuple(flatten_arg + flatten_kwarg))
        if not all(isinstance(arg, torch.Tensor) for arg in flatten_arg):
            raise TypeError(
                "In the beta API, sample_args "
                + "for each callable must contain only Tensors. Other types are not allowed."
            )

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    # Note: These per_callable_* variables are not actually
    # per-callable, but per-forward-pass (see description of _order).
    # The names are kept for consistency with
    # torch.cuda.make_graphed_callables.
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    if _order is None:
        per_callable_module_params = [
            tuple(c.parameters()) if isinstance(c, torch.nn.Module) else () for c in callables
        ]
        per_callable_static_input_surfaces = [
            flatten_sample_args[i] + per_callable_module_params[i] for i in range(len(callables))
        ]
    else:
        per_callable_module_params = []
        for m_chunk in range(num_model_chunks):
            for _ in range(num_microbatches):
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    per_callable_module_params.append(
                        tuple(callables[_prefix_num_layers[m_chunk] + l_no].parameters())
                        if isinstance(
                            callables[_prefix_num_layers[m_chunk] + l_no],
                            torch.nn.Module,
                        )
                        else ()
                    )
        if len(per_callable_module_params) != len(flatten_sample_args):
            raise ValueError(
                "Pipeline-parallel dimension mismatch: "
                f"per_callable_module_params has {len(per_callable_module_params)} entries, "
                f"but flatten_sample_args has {len(flatten_sample_args)} entries"
            )
        per_callable_static_input_surfaces = [
            flatten_sample_args[i] + per_callable_module_params[i]
            for i in range(len(flatten_sample_args))
        ]

    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
    bwd_dw_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
    graph_callables = [None for _ in range(len(flatten_sample_args))]

    # For cases with multiple active RNG states, e.g. TP.
    if graph_safe_rng_available() and not bool(
        int(os.getenv("NVTE_DISABLE_GRAPH_SAFE_RNG_REGISTRATION", "0"))
    ):
        for _, state in get_all_rng_states().items():
            for fwd_graph, bwd_graph, bwd_dw_graph in zip(fwd_graphs, bwd_graphs, bwd_dw_graphs):
                fwd_graph.register_generator_state(state)
                bwd_graph.register_generator_state(state)
                bwd_dw_graph.register_generator_state(state)

    allocator_settings_to_apply = None
    allocator_settings_to_restore = None
    allocator_settings_setter = None
    if use_slot_memory:
        allocator_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF") or os.getenv("PYTORCH_ALLOC_CONF", "")
        allocator_parts = [part.strip() for part in allocator_conf.split(",") if part.strip()]
        expandable_enabled = any(
            part.split(":", 1)[0].strip() == "expandable_segments"
            and part.split(":", 1)[1].strip().lower() == "true"
            for part in allocator_parts
            if ":" in part
        )
        if expandable_enabled:
            allocator_settings_setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
            if allocator_settings_setter is None:
                raise RuntimeError(
                    "Temporarily disabling expandable segments during CUDA graph capture "
                    "requires torch._C._accelerator_setAllocatorSettings."
                )
            disabled_parts = [
                (
                    "expandable_segments:False"
                    if part.split(":", 1)[0].strip() == "expandable_segments"
                    else part
                )
                for part in allocator_parts
            ]
            allocator_settings_to_apply = ",".join(disabled_parts)
            allocator_settings_to_restore = allocator_conf

    if use_slot_memory:
        slot_allocator_pool = torch.cuda.MemPool()
        mempool = slot_allocator_pool.id
    else:
        slot_allocator_pool = None
        mempool = graph_pool_handle() if pool is None else pool

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()

    # Get warmup func and func_idx.
    warmup_func_idx = []
    warmup_func = []
    if _order is None:
        for func_idx, func in enumerate(callables):
            warmup_func_idx.append(func_idx)
            warmup_func.append(func)
    else:
        fwd_idx = [0] * num_model_chunks
        for c_id in _order:
            if c_id > 0:
                m_chunk = c_id - 1
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    func = callables[_prefix_num_layers[m_chunk] + l_no]
                    func_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        fwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    warmup_func_idx.append(func_idx)
                    warmup_func.append(func)
                fwd_idx[m_chunk] += 1
    if len(warmup_func) != len(sample_args):
        raise ValueError(f"Warmup runs {len(warmup_func)} don't match args {len(sample_args)}.")
    if len(warmup_func_idx) != len(set(warmup_func_idx)):
        raise RuntimeError(
            f"Warmup runs {len(warmup_func)} but only {len(set(warmup_func_idx))} are unique."
        )

    warmup_plan_aliases = {}
    if warmup_plan_alias_groups is not None:
        templates = {}
        unique_warmups = []
        for func_idx, func in zip(warmup_func_idx, warmup_func):
            group = warmup_plan_alias_groups[func_idx]
            template = templates.get(group)
            if template is None:
                templates[group] = (func_idx, func)
                warmup_plan_aliases[func_idx] = []
                unique_warmups.append((func_idx, func))
            else:
                template_idx, template_func = template
                if template_func is not func:
                    raise RuntimeError(
                        f"Warmup-plan alias group {group} spans different callable objects."
                    )
                warmup_plan_aliases[template_idx].append(func_idx)
        # Alias-group IDs also define the communicator warmup order. Dynamic-CP captures use
        # variant 0 for the largest CP group, even though mutually exclusive smaller branches
        # must appear first in the formal graph order for memory liveness. Warm the largest
        # group first so its P2P ring is fully initialized before switching among subgroups.
        ordered_warmups = sorted(unique_warmups, key=lambda item: warmup_plan_alias_groups[item[0]])
        warmup_func_idx = [func_idx for func_idx, _ in ordered_warmups]
        warmup_func = [func for _, func in ordered_warmups]

    # Filter the TE modules that cudagraph can access.
    visited_te_modules = {}
    need_bwd_dw_graph = {}
    per_callable_fused_wgrad_params = {}
    if use_slot_memory:
        num_graph_inputs = len(flatten_sample_args)
        per_callable_saved_tensor_plans = [None] * num_graph_inputs
        per_callable_saved_tensor_boundary_aliases = [None] * num_graph_inputs
        per_callable_output_tensor_plans = [None] * num_graph_inputs
        per_callable_user_grad_tensor_plans = [None] * num_graph_inputs
        per_callable_param_grad_tensor_targets = None
        per_callable_external_storage_ptrs = [
            {
                _tensor_storage_ptr(tensor)
                for tensor in static_input_surface
                if isinstance(tensor, torch.Tensor)
            }
            for static_input_surface in per_callable_static_input_surfaces
        ]
        per_callable_snapshot_input_storage_ptrs = []
        for func_idx, args in enumerate(sample_args):
            if not args or args[0].__class__ is not torch.Tensor or not args[0].is_cuda:
                raise RuntimeError(
                    "Slot user-input snapshots require the first positional argument for "
                    f"graph input {func_idx} to be a plain CUDA tensor."
                )
            if flatten_sample_args[func_idx][0] is not args[0]:
                raise RuntimeError(
                    "Slot user-input snapshots require the first positional tensor to be the "
                    "first flattened graph input."
                )
            per_callable_snapshot_input_storage_ptrs.append(
                {
                    _tensor_storage_ptr(tensor)
                    for tensor in flatten_sample_args[func_idx]
                    if tensor.__class__ is torch.Tensor and tensor.is_cuda
                }
            )
    else:
        per_callable_saved_tensor_plans = None
        per_callable_saved_tensor_boundary_aliases = None
        per_callable_output_tensor_plans = None
        per_callable_user_grad_tensor_plans = None
        per_callable_param_grad_tensor_targets = None
        per_callable_external_storage_ptrs = None
        per_callable_snapshot_input_storage_ptrs = None

    def clone_warmup_plan(template_idx, target_idx):
        """Clone one shape-identical warmup observation onto another static slot."""
        source_args = flatten_sample_args[template_idx]
        target_args = flatten_sample_args[target_idx]
        if len(source_args) != len(target_args):
            raise RuntimeError(
                f"Warmup-plan aliases {template_idx} and {target_idx} expose different "
                "numbers of user tensors."
            )

        external_storage_map = {}
        for source, target in zip(source_args, target_args):
            if _input_staging_key(source) != _input_staging_key(target):
                raise RuntimeError(
                    f"Warmup-plan aliases {template_idx} and {target_idx} have incompatible "
                    "user tensor surfaces."
                )
            source_ptr = _tensor_storage_ptr(source)
            target_ptr = _tensor_storage_ptr(target)
            previous_target = external_storage_map.setdefault(source_ptr, target_ptr)
            if previous_target != target_ptr:
                raise RuntimeError(
                    f"Warmup-plan alias {target_idx} changes an input storage alias from "
                    f"{previous_target} to {target_ptr}."
                )

        source_plan = per_callable_saved_tensor_plans[template_idx]
        if source_plan is None:
            raise RuntimeError(f"Warmup template {template_idx} has no saved-tensor plan.")
        per_callable_saved_tensor_plans[target_idx] = [
            (
                (spec[0], external_storage_map.get(spec[1], spec[1]), *spec[2:])
                if spec[0] == "external"
                else spec
            )
            for spec in source_plan
        ]
        per_callable_saved_tensor_boundary_aliases[target_idx] = list(
            per_callable_saved_tensor_boundary_aliases[template_idx]
        )
        for plans in (
            per_callable_output_tensor_plans,
            per_callable_user_grad_tensor_plans,
        ):
            plans[target_idx] = list(plans[template_idx])

        per_callable_module_params[target_idx] = per_callable_module_params[template_idx]
        per_callable_static_input_surfaces[target_idx] = (
            target_args + per_callable_module_params[target_idx]
        )
        visited_te_modules[target_idx] = set(visited_te_modules.get(template_idx, set()))
        per_callable_fused_wgrad_params[target_idx] = set(
            per_callable_fused_wgrad_params.get(template_idx, set())
        )
        need_bwd_dw_graph[target_idx] = need_bwd_dw_graph.get(template_idx, False)

    def update_warmup_plan(plans, func_idx, observed_plan, phase):
        """Record a stable slot-memory plan across warmup iterations."""
        expected_plan = plans[func_idx]
        if expected_plan is None:
            plans[func_idx] = observed_plan
        elif expected_plan != observed_plan:
            raise RuntimeError(
                f"{phase} saved tensors changed across CUDA graph warmup iterations "
                f"for graph input {func_idx}."
            )

    def observe_saved_tensor_boundary_aliases(
        func_idx, saved_tensors, saved_versions, outputs, saved_plan
    ):
        """Record native saves that are byte ranges of public graph boundaries."""
        boundaries = []
        for kind, tensors in (
            (
                "input",
                per_callable_static_input_surfaces[func_idx][
                    : per_callable_len_user_args[func_idx]
                ],
            ),
            ("output", outputs),
        ):
            for tensor_idx, tensor in enumerate(tensors):
                if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                    continue
                span_bytes = _saved_tensor_signature(tensor)[-1]
                start = tensor.storage_offset() * tensor.element_size()
                boundaries.append(
                    (
                        tensor.untyped_storage()._cdata,
                        start,
                        start + span_bytes,
                        span_bytes,
                        kind,
                        tensor_idx,
                        _tensor_version(tensor),
                    )
                )

        aliases = []
        for tensor, saved_version, spec in zip(saved_tensors, saved_versions, saved_plan):
            if spec[0] != "native" or spec[7] == 0:
                aliases.append(None)
                continue
            saved_start = tensor.storage_offset() * tensor.element_size()
            saved_end = saved_start + spec[7]
            storage_id = tensor.untyped_storage()._cdata
            candidates = [
                (
                    span_bytes,
                    kind,
                    tensor_idx,
                    saved_start - boundary_start,
                    saved_version is not None and saved_version == boundary_version,
                )
                for (
                    boundary_storage_id,
                    boundary_start,
                    boundary_end,
                    span_bytes,
                    kind,
                    tensor_idx,
                    boundary_version,
                ) in boundaries
                if boundary_storage_id == storage_id
                and boundary_start <= saved_start
                and saved_end <= boundary_end
            ]
            aliases.append(
                min(candidates, key=lambda candidate: (not candidate[4], candidate[:4]))
                if candidates
                else None
            )
        return aliases

    def make_saved_tensor_recorder(
        func_idx,
        observed_saved_tensors,
        observed_saved_versions,
        copied_storages,
        observed_saved_tensor_plan,
    ):
        """Bind one warmup iteration's saved-tensor observation state."""

        def record_saved_tensor(tensor):
            observed_saved_tensors.append(tensor)
            observed_saved_versions.append(_tensor_version(tensor))
            storage_ptr = _tensor_storage_ptr(tensor)
            signature = _saved_tensor_signature(tensor)
            snapshot_user_input = (
                tensor.is_cuda and storage_ptr in per_callable_snapshot_input_storage_ptrs[func_idx]
            )
            is_external = not tensor.is_cuda or (
                storage_ptr in per_callable_external_storage_ptrs[func_idx]
                and not snapshot_user_input
            )
            if not is_external and tensor.__class__ is not torch.Tensor:
                raise RuntimeError(
                    "CUDA graph saved-tensor arenas do not yet support tensor "
                    f"subclass {type(tensor).__name__}."
                )
            storage_group = None
            storage_offset_bytes = None
            if not is_external:
                storage_identity = (storage_ptr, _tensor_version(tensor))
                storage_group = copied_storages.setdefault(storage_identity, len(copied_storages))
                storage_offset_bytes = tensor.storage_offset() * tensor.element_size()
            observed_saved_tensor_plan.append(
                (
                    "external" if is_external else "native",
                    storage_ptr if is_external else None,
                    *signature,
                    storage_group,
                    storage_offset_bytes,
                )
            )
            return tensor

        return record_saved_tensor

    # Run warmup and do the above filtering.
    with torch.cuda.stream(torch.cuda.Stream()):
        for func_idx, func in zip(warmup_func_idx, warmup_func):
            args = sample_args[func_idx]
            kwargs = sample_kwargs[func_idx]
            static_input_surface = per_callable_static_input_surfaces[func_idx]
            if per_callable_external_storage_ptrs is not None and isinstance(func, torch.nn.Module):
                per_callable_external_storage_ptrs[func_idx].update(
                    _tensor_storage_ptr(buffer) for buffer in func.buffers()
                )

            def hook_fn(
                module, inputs, outputs, func_idx=func_idx
            ):  # pylint: disable=unused-argument
                modules = set()
                if isinstance(module, TransformerEngineBaseModule):
                    modules.add(module)
                # If forward is called on a BasicOperation directly the hook will run
                elif isinstance(module, BasicOperation):
                    modules.add(module)
                # If forward is called on a te.ops.Sequential it is not called on its constituent ops
                elif isinstance(module, Sequential):
                    if module._module_groups is None:
                        raise RuntimeError(
                            "module._module_groups should have been initialized by warmup"
                        )
                    for module_group in module._module_groups:
                        if isinstance(module_group, OperationFuser):
                            for basic_op in module_group._basic_ops:
                                modules.add(basic_op)
                if modules:
                    if func_idx not in visited_te_modules:
                        visited_te_modules[func_idx] = modules
                    else:
                        visited_te_modules[func_idx].update(modules)

            if pre_warmup_hook is not None:
                pre_warmup_hook()
            for warmup_iter in range(num_warmup_iters):
                with _module_forward_hooks(func.modules(), hook_fn):
                    if use_slot_memory:
                        observed_saved_tensor_plan = []
                        observed_saved_tensors = []
                        observed_saved_versions = []
                        copied_storages = {}
                        record_saved_tensor = make_saved_tensor_recorder(
                            func_idx,
                            observed_saved_tensors,
                            observed_saved_versions,
                            copied_storages,
                            observed_saved_tensor_plan,
                        )

                        with torch.autograd.graph.saved_tensors_hooks(
                            record_saved_tensor, lambda x: x
                        ):
                            outputs, _ = _tree_flatten(func(*args, **kwargs))
                        observed_boundary_aliases = observe_saved_tensor_boundary_aliases(
                            func_idx,
                            observed_saved_tensors,
                            observed_saved_versions,
                            outputs,
                            observed_saved_tensor_plan,
                        )
                        update_warmup_plan(
                            per_callable_saved_tensor_plans,
                            func_idx,
                            observed_saved_tensor_plan,
                            "Forward",
                        )
                        update_warmup_plan(
                            per_callable_saved_tensor_boundary_aliases,
                            func_idx,
                            observed_boundary_aliases,
                            "Forward boundary alias",
                        )
                        update_warmup_plan(
                            per_callable_output_tensor_plans,
                            func_idx,
                            [_io_tensor_plan(output, "output") for output in outputs],
                            "Output",
                        )
                    else:
                        outputs, _ = _tree_flatten(func(*args, **kwargs))
                if is_training:
                    inputs = tuple(i for i in static_input_surface if i.requires_grad)
                    with _none_grad_context_wrapper(inputs):
                        outputs_requiring_grad = tuple(
                            o for o in outputs if o is not None and o.requires_grad
                        )
                        torch.autograd.backward(
                            outputs_requiring_grad,
                            grad_tensors=tuple(torch.empty_like(o) for o in outputs_requiring_grad),
                        )
                        grad_inputs = tuple(input.grad for input in inputs)
                    if use_slot_memory:
                        observed_user_grad_tensor_plan = []
                        grad_idx = 0
                        for input_idx, input_tensor in enumerate(static_input_surface):
                            grad_input = None
                            if (
                                isinstance(input_tensor, torch.Tensor)
                                and input_tensor.requires_grad
                            ):
                                grad_input = grad_inputs[grad_idx]
                                grad_idx += 1
                            if input_idx < per_callable_len_user_args[func_idx]:
                                observed_user_grad_tensor_plan.append(
                                    _io_tensor_plan(grad_input, "user_grad")
                                )
                        update_warmup_plan(
                            per_callable_user_grad_tensor_plans,
                            func_idx,
                            observed_user_grad_tensor_plan,
                            "User-gradient output",
                        )

                    # Filter module params that get None grad from grad_inputs and remove them
                    # from static_input_surface. This is to ensure that the backward hooks
                    # registered to these params are not wrongly triggered.
                    num_required_grad_sample_args = sum(
                        arg.requires_grad for arg in flatten_sample_args[func_idx]
                    )
                    required_grad_input_idx = []
                    for i, arg in enumerate(static_input_surface):
                        if arg.requires_grad:
                            required_grad_input_idx.append(i)
                    fused_wgrad_params = set()
                    if use_slot_memory:
                        for module in visited_te_modules.get(func_idx, set()):
                            if not (
                                isinstance(module, TransformerEngineBaseModule)
                                and getattr(module, "fuse_wgrad_accumulation", False)
                            ):
                                continue
                            for name in getattr(module, "weight_names", ()):
                                param = getattr(module, name, None)
                                if isinstance(param, torch.nn.Parameter) and param.requires_grad:
                                    fused_wgrad_params.add(param)
                                    get_dummy_wgrad(
                                        list(param.shape),
                                        param.dtype,
                                        zero=getattr(param, "zero_out_wgrad", False),
                                    )
                    per_callable_fused_wgrad_params[func_idx] = fused_wgrad_params
                    module_params_with_grad = []
                    for grad_inputs_idx, inputs_idx in enumerate(required_grad_input_idx):
                        input_tensor = static_input_surface[inputs_idx]
                        if (
                            grad_inputs[grad_inputs_idx] is None
                            and grad_inputs_idx < num_required_grad_sample_args
                        ):
                            if not allow_unused_input:
                                raise RuntimeError(
                                    "The input tensor requires grad, but the grad is None after"
                                    " backward pass."
                                )
                        elif grad_inputs_idx >= num_required_grad_sample_args and (
                            grad_inputs[grad_inputs_idx] is not None
                            or input_tensor in fused_wgrad_params
                        ):
                            # Fused wgrad writes directly into main_grad. Keep its parameter as
                            # an autograd input even when no ordinary param.grad was materialized,
                            # so replay can still trigger AccumulateGrad/DDP hooks.
                            module_params_with_grad.append(input_tensor)
                    if len(module_params_with_grad) != len(per_callable_module_params[func_idx]):
                        if warmup_iter != 0:
                            raise RuntimeError(
                                "no-grad params should only be used as inputs in the first warmup"
                                f" iteration, but found in iteration {warmup_iter}"
                            )
                        per_callable_module_params[func_idx] = tuple(module_params_with_grad)
                        static_input_surface = flatten_sample_args[func_idx] + tuple(
                            module_params_with_grad
                        )
                        per_callable_static_input_surfaces[func_idx] = static_input_surface

                    # Run wgrad. This is essential for some TE modules when they have
                    # delay_wgrad_compute enabled.
                    need_backward_dw = False
                    for module in visited_te_modules.get(func_idx, set()):
                        if hasattr(module, "need_backward_dw") and module.need_backward_dw():
                            need_backward_dw = True
                            module.backward_dw()
                    need_bwd_dw_graph[func_idx] = need_backward_dw
                else:
                    grad_inputs = None
                del outputs, grad_inputs
                if is_training:
                    del outputs_requiring_grad
                    if use_slot_memory:
                        grad_input = None
            if post_warmup_hook is not None:
                post_warmup_hook()
            if warmup_plan_alias_groups is not None:
                # Dynamic-CP warmup callables can replace the CP process group while TE still
                # has asynchronous CP/TP work queued on auxiliary streams. Drain every observed
                # callable before changing groups; otherwise one TP peer can enter the next
                # variant while the other is still completing the previous CP ring.
                torch.cuda.synchronize()
            for target_idx in warmup_plan_aliases.get(func_idx, ()):
                clone_warmup_plan(func_idx, target_idx)
    torch.cuda.synchronize()

    if use_slot_memory:
        per_callable_param_grad_tensor_targets = [
            [None] * len(static_input_surface)
            for static_input_surface in per_callable_static_input_surfaces
        ]

    if allocator_settings_to_apply is not None:
        if _allocator_settings_guard is None or allocator_settings_setter is None:
            raise RuntimeError("CUDA graph slot capture is missing its allocator settings guard.")
        _allocator_settings_guard.apply(
            allocator_settings_setter,
            allocator_settings_to_apply,
            allocator_settings_to_restore,
        )

    if use_slot_memory:
        if isinstance(sample_args, tuple):
            sample_args = list(sample_args)

        staging_group_by_key = {}
        staging_groups = []
        for func_idx, args in enumerate(sample_args):
            old_input = args[0]
            saved_arena_id = saved_tensor_arena_ids[func_idx]
            staging_key = (saved_arena_id, _input_staging_key(old_input))
            group_idx = staging_group_by_key.get(staging_key)
            if group_idx is None:
                group_idx = len(staging_groups)
                staging_group_by_key[staging_key] = group_idx
                staging_groups.append({"members": [], "candidates": {}})
            group = staging_groups[group_idx]
            group["members"].append(func_idx)
            group["candidates"].setdefault(_tensor_storage_ptr(old_input), old_input)

        # MCore's sample-input plan and the union liveness coloring are each safe in
        # isolation, but reusing an arbitrary representative can transitively merge two
        # conflicting colors. Match colors onto distinct existing storages first, then
        # allocate only when the original CP-variant plans do not provide enough choices.
        storage_owner = {}
        staging_targets = {}

        def match_staging_group(group_idx, seen_storages):
            for storage_id, tensor in staging_groups[group_idx]["candidates"].items():
                if storage_id in seen_storages:
                    continue
                seen_storages.add(storage_id)
                previous_group = storage_owner.get(storage_id)
                if previous_group is None or match_staging_group(previous_group, seen_storages):
                    storage_owner[storage_id] = group_idx
                    staging_targets[group_idx] = tensor
                    return True
            return False

        for group_idx in sorted(
            range(len(staging_groups)),
            key=lambda index: len(staging_groups[index]["candidates"]),
        ):
            match_staging_group(group_idx, set())

        for group_idx, group in enumerate(staging_groups):
            input_target = staging_targets.get(group_idx)
            if input_target is None:
                source = next(iter(group["candidates"].values()))
                signature = _saved_tensor_signature(source)
                storage_numel = signature[-1] // source.element_size()
                with torch.cuda.use_mem_pool(slot_allocator_pool):
                    backing = torch.empty(
                        (source.storage_offset() + storage_numel,),
                        dtype=source.dtype,
                        device=source.device,
                    )
                input_target = torch.empty((0,), dtype=source.dtype, device=source.device).set_(
                    backing.untyped_storage(),
                    source.storage_offset(),
                    source.shape,
                    source.stride(),
                )
                with torch.no_grad():
                    input_target.copy_(source)
                input_target.requires_grad_(source.requires_grad)
                staging_targets[group_idx] = input_target

            for func_idx in group["members"]:
                args = sample_args[func_idx]
                old_input = args[0]
                if input_target is old_input:
                    continue

                args = list(args)
                args[0] = input_target
                sample_args[func_idx] = tuple(args)
                flattened_args = list(flatten_sample_args[func_idx])
                flattened_args[0] = input_target
                flatten_sample_args[func_idx] = tuple(flattened_args)
                static_input_surface = list(per_callable_static_input_surfaces[func_idx])
                static_input_surface[0] = input_target
                per_callable_static_input_surfaces[func_idx] = tuple(static_input_surface)

    def prepare_native_io_targets(per_callable_plans, kind):
        """Validate same-slot CP branches and create their lazy alias targets."""
        if per_callable_plans is None:
            return None

        plans_by_family = {}
        for func_idx, plan in enumerate(per_callable_plans):
            arena_id, branch_id = slot_io_memory_alias_groups[func_idx]
            family = (arena_id, *slot_io_liveness_groups[func_idx])
            branch_plans = plans_by_family.setdefault(family, {})
            if branch_id in branch_plans:
                raise RuntimeError(
                    f"CUDA graph {kind} family {family} has duplicate branch {branch_id}."
                )
            branch_plans[branch_id] = plan

        for family, branch_plans in plans_by_family.items():
            plans = list(branch_plans.values())
            if len({len(plan) for plan in plans}) != 1:
                raise RuntimeError(
                    f"CUDA graph {kind} family {family} exposes different tensor counts."
                )
            for tensor_idx, specs in enumerate(zip(*plans)):
                if all(spec is None for spec in specs):
                    continue
                if any(spec is None for spec in specs):
                    raise RuntimeError(
                        f"CUDA graph {kind} family {family} has an incompatible tensor "
                        f"at position {tensor_idx}."
                    )
                layout_keys = {(spec[0], spec[4], spec[5], spec[6]) for spec in specs}
                if layout_keys != {(kind, specs[0][4], specs[0][5], specs[0][6])}:
                    raise RuntimeError(
                        f"CUDA graph {kind} family {family} has incompatible dtype, device, "
                        f"or autograd state at position {tensor_idx}."
                    )

        return [[None] * len(plan) for plan in per_callable_plans]

    per_callable_output_tensor_targets = prepare_native_io_targets(
        per_callable_output_tensor_plans, "output"
    )
    per_callable_user_grad_tensor_targets = prepare_native_io_targets(
        per_callable_user_grad_tensor_plans, "user_grad"
    )

    native_io_family_sizes = {}
    if use_slot_memory:
        for func_idx in range(len(flatten_sample_args)):
            arena_id, _ = slot_io_memory_alias_groups[func_idx]
            family = (arena_id, *slot_io_liveness_groups[func_idx])
            native_io_family_sizes[family] = native_io_family_sizes.get(family, 0) + 1
    native_io_anchors = {"output": {}, "user_grad": {}, "param_grad": {}}
    native_io_capture_counts = {"output": {}, "user_grad": {}, "param_grad": {}}
    release_native_io_targets = use_slot_memory

    def native_io_alias_target(func_idx, tensor_idx, tensor, spec, kind):
        """Alias one CP branch onto the first branch's graph-pool I/O storage."""
        arena_id, _ = slot_io_memory_alias_groups[func_idx]
        family = (arena_id, *slot_io_liveness_groups[func_idx])
        key = (*family, tensor_idx)
        anchors = native_io_anchors[kind]
        counts = native_io_capture_counts[kind]
        anchor = anchors.get(key)
        if anchor is None:
            target = tensor
            anchors[key] = tensor
        else:
            available_bytes = anchor.untyped_storage().nbytes() - (
                anchor.storage_offset() * anchor.element_size()
            )
            if spec[7] > available_bytes:
                raise RuntimeError(
                    f"CUDA graph native {kind} alias {key} needs {spec[7]} bytes, "
                    f"but its first CP branch exposes only {available_bytes} bytes."
                )
            target = _arena_view(anchor, 0, spec)

        captured = counts.get(key, 0) + 1
        expected = native_io_family_sizes[family]
        if captured > expected:
            raise RuntimeError(
                f"CUDA graph native {kind} alias {key} captured {captured} of "
                f"{expected} CP branches."
            )
        if captured == expected:
            anchors.pop(key)
            counts.pop(key, None)
        else:
            counts[key] = captured
        return target

    def clear_native_io_target_rows(func_indices, clear_outputs=False, clear_grads=False):
        """Drop capture-only I/O aliases after the corresponding TE value dies."""
        if not release_native_io_targets:
            return
        for func_idx in func_indices:
            if clear_outputs:
                per_callable_output_tensor_targets[func_idx] = [None] * len(
                    per_callable_output_tensor_targets[func_idx]
                )
            if clear_grads:
                per_callable_user_grad_tensor_targets[func_idx] = [None] * len(
                    per_callable_user_grad_tensor_targets[func_idx]
                )

    def copy_outputs_to_slot_arena(func_idx, flatten_outputs):
        """Copy public forward outputs to the fixed surface for their physical slot."""
        if per_callable_output_tensor_targets is None:
            return flatten_outputs
        plan = per_callable_output_tensor_plans[func_idx]
        targets = per_callable_output_tensor_targets[func_idx]
        if len(flatten_outputs) != len(plan):
            raise RuntimeError(
                f"CUDA graph input {func_idx} changed its output count during capture."
            )
        copied_outputs = []
        for tensor_idx, (output, spec, target) in enumerate(zip(flatten_outputs, plan, targets)):
            if spec != _io_tensor_plan(output, "output"):
                raise RuntimeError(
                    f"CUDA graph input {func_idx} changed its output tensor surface during capture."
                )
            if target is None:
                if spec is None:
                    copied_outputs.append(output)
                    continue
                target = native_io_alias_target(func_idx, tensor_idx, output, spec, "output")
                targets[tensor_idx] = target
            if target is not output:
                target.copy_(output)
            copied_outputs.append(target)
        return copied_outputs

    def copy_user_grads_to_slot_arena(func_idx, static_input_surface, grad_inputs):
        """Copy returned gradients to the fixed surface for their physical slot."""
        if per_callable_user_grad_tensor_targets is None:
            return grad_inputs
        plan = per_callable_user_grad_tensor_plans[func_idx]
        targets = per_callable_user_grad_tensor_targets[func_idx]
        copied_grad_inputs = []
        grad_idx = 0
        for input_idx, input_tensor in enumerate(static_input_surface):
            if not (isinstance(input_tensor, torch.Tensor) and input_tensor.requires_grad):
                continue
            grad_input = grad_inputs[grad_idx]
            grad_idx += 1
            if input_idx < per_callable_len_user_args[func_idx]:
                spec = plan[input_idx]
                target = targets[input_idx]
                if spec != _io_tensor_plan(grad_input, "user_grad"):
                    raise RuntimeError(
                        f"CUDA graph input {func_idx} changed its user-gradient tensor "
                        "surface during capture."
                    )
                if target is None and spec is not None:
                    target = native_io_alias_target(
                        func_idx, input_idx, grad_input, spec, "user_grad"
                    )
                    targets[input_idx] = target
                if target is not None:
                    if target is not grad_input:
                        with torch.no_grad():
                            target.copy_(grad_input)
                    grad_input = target
            elif per_callable_param_grad_tensor_targets is not None and grad_input is not None:
                spec = _io_tensor_plan(grad_input, "param_grad")
                if spec is None:
                    raise RuntimeError(
                        f"CUDA graph input {func_idx} produced an unsupported parameter-gradient "
                        f"tensor at input position {input_idx}."
                    )
                target = per_callable_param_grad_tensor_targets[func_idx][input_idx]
                if target is None:
                    target = native_io_alias_target(
                        func_idx, input_idx, grad_input, spec, "param_grad"
                    )
                    per_callable_param_grad_tensor_targets[func_idx][input_idx] = target
                if target is not grad_input:
                    with torch.no_grad():
                        target.copy_(grad_input)
                grad_input = target
            copied_grad_inputs.append(grad_input)
        return tuple(copied_grad_inputs)

    per_callable_native_saved_storages = [{} for _ in flatten_sample_args]

    def plan_native_saved_alias_targets(
        plan,
        canonical_targets,
        measure_spill=False,
        protected_storage_ranges=None,
        preassigned_targets=None,
    ):
        """Pack one CP branch's native saved tensors into canonical live storages."""
        protected_storage_ranges = protected_storage_ranges or {}
        if preassigned_targets is None:
            target_views = [None] * len(plan)
        else:
            if len(preassigned_targets) != len(plan):
                raise RuntimeError("Native saved preassignment does not match its plan.")
            target_views = list(preassigned_targets)
        storage_banks = []
        seen_storages = set()
        for target in canonical_targets:
            if target is None:
                continue
            storage = target.untyped_storage()
            if storage._cdata in seen_storages:
                continue
            seen_storages.add(storage._cdata)
            cursor = 0
            for protected_start, protected_end in protected_storage_ranges.get(storage._cdata, ()):
                if cursor < protected_start:
                    storage_banks.append(
                        {
                            "storage": storage,
                            "base_offset": cursor,
                            "capacity": protected_start - cursor,
                            "cursor": 0,
                        }
                    )
                cursor = max(cursor, protected_end)
            if cursor < storage.nbytes():
                storage_banks.append(
                    {
                        "storage": storage,
                        "base_offset": cursor,
                        "capacity": storage.nbytes() - cursor,
                        "cursor": 0,
                    }
                )
        records_by_storage_group = {}
        for saved_idx, spec in enumerate(plan):
            if spec[0] != "native":
                continue
            if target_views[saved_idx] is not None:
                continue
            if spec[7] == 0:
                target = torch.empty_strided(spec[2], spec[3], dtype=spec[4], device=spec[5])
                target.requires_grad_(spec[6])
                target_views[saved_idx] = target
                continue
            storage_group = spec[8]
            storage_offset_bytes = spec[9]
            if storage_group is None or storage_offset_bytes is None:
                raise RuntimeError(f"Native saved tensor {saved_idx} has no backing-storage plan.")
            records_by_storage_group.setdefault(storage_group, []).append(
                (storage_offset_bytes, storage_offset_bytes + spec[7], saved_idx)
            )

        components = []
        for records in records_by_storage_group.values():
            records.sort()
            group_components = []
            for start, end, saved_idx in records:
                if not group_components or start >= group_components[-1][1]:
                    group_components.append([start, end, [(start, saved_idx)]])
                else:
                    group_components[-1][1] = max(group_components[-1][1], end)
                    group_components[-1][2].append((start, saved_idx))
            components.extend(group_components)

        packed_components = []
        for component_start, component_end, component_records in components:
            component_alignment = max(
                plan[saved_idx][4].itemsize for _, saved_idx in component_records
            )
            component_origin = component_start // component_alignment * component_alignment
            packed_components.append(
                (
                    component_end - component_origin,
                    component_origin,
                    component_records,
                )
            )

        banks = list(storage_banks)
        spill_bank = None
        if measure_spill:
            spill_bank = {
                "storage": None,
                "base_offset": 0,
                "capacity": sum(_align_up(item[0]) for item in packed_components),
                "cursor": 0,
            }
            banks.append(spill_bank)
        for component_size, component_origin, component_records in sorted(
            packed_components, key=lambda item: item[0], reverse=True
        ):
            candidates = []
            for bank_idx, bank in enumerate(banks):
                if bank["storage"] is None:
                    continue
                offset = _align_up(bank["cursor"])
                if offset + component_size <= bank["capacity"]:
                    candidates.append(
                        (bank["capacity"] - offset - component_size, bank_idx, offset)
                    )
            if not candidates and spill_bank is not None:
                spill_bank_idx = len(banks) - 1
                spill_offset = _align_up(spill_bank["cursor"])
                if spill_offset + component_size <= spill_bank["capacity"]:
                    candidates.append((0, spill_bank_idx, spill_offset))
            if not candidates:
                raise RuntimeError(
                    "CUDA graph CP branch native saved tensors do not fit in canonical "
                    "live storages: "
                    f"component_bytes={component_size}, "
                    f"component_sizes={sorted((item[0] for item in packed_components), reverse=True)}, "
                    f"storage_capacities={sorted((bank['capacity'] for bank in banks), reverse=True)}."
                )
            _, bank_idx, component_target_offset = min(candidates)
            bank = banks[bank_idx]
            bank["cursor"] = component_target_offset + component_size
            for source_offset, saved_idx in component_records:
                if bank["storage"] is None:
                    target_views[saved_idx] = True
                    continue
                spec = plan[saved_idx]
                target_offset = (
                    bank["base_offset"] + component_target_offset + source_offset - component_origin
                )
                itemsize = spec[4].itemsize
                if target_offset % itemsize:
                    raise RuntimeError(
                        f"Native saved tensor {saved_idx} has an unaligned canonical offset."
                    )
                target = torch.empty((0,), dtype=spec[4], device=spec[5]).set_(
                    bank["storage"],
                    target_offset // itemsize,
                    spec[2],
                    spec[3],
                )
                target.requires_grad_(spec[6])
                target_views[saved_idx] = target

        if measure_spill:
            return _align_up(spill_bank["cursor"])

        missing = [
            saved_idx
            for saved_idx, spec in enumerate(plan)
            if spec[0] == "native" and target_views[saved_idx] is None
        ]
        if missing:
            raise RuntimeError(f"Native saved tensors have no canonical targets: {missing}.")
        return tuple(target_views)

    def semantic_boundary_alias_targets(func_idx, outputs):
        """Map boundary-backed saves onto the boundary address used at replay."""
        plan = per_callable_saved_tensor_plans[func_idx]
        aliases = per_callable_saved_tensor_boundary_aliases[func_idx]
        targets = [None] * len(plan)
        records_by_storage_group = {}
        for saved_idx, spec in enumerate(plan):
            if spec[0] != "native" or spec[7] == 0:
                continue
            records_by_storage_group.setdefault(spec[8], []).append(
                (spec[9], spec[9] + spec[7], saved_idx)
            )

        components = []
        for records in records_by_storage_group.values():
            records.sort()
            group_components = []
            for start, end, saved_idx in records:
                if not group_components or start >= group_components[-1][1]:
                    group_components.append([start, end, [saved_idx]])
                else:
                    group_components[-1][1] = max(group_components[-1][1], end)
                    group_components[-1][2].append(saved_idx)
            components.extend(group_components)

        for component_start, component_end, component_saved_indices in components:
            candidate_aliases = [
                (saved_idx, aliases[saved_idx])
                for saved_idx in component_saved_indices
                if aliases[saved_idx] is not None
                # Only the leading input is rebound to a union-liveness staging surface.
                # Other user inputs may share MCore capture-order buffers that overlap in a
                # different runtime schedule, so they must use the saved arena instead.
                and not (
                    aliases[saved_idx][1] == "input" and aliases[saved_idx][2] != 0
                )
            ]
            if not candidate_aliases:
                continue

            # An alias only proves that one saved view is a graph-boundary view.  Reusing
            # the boundary for its whole overlapping storage component is safe only when
            # every byte in that component is part of the same logical boundary tensor.
            component_aliases = []
            for saved_idx, alias in candidate_aliases:
                boundary_span_bytes, _, _, relative_offset, version_matches = alias
                if not version_matches:
                    continue
                source_boundary_start = plan[saved_idx][9] - relative_offset
                source_boundary_end = source_boundary_start + boundary_span_bytes
                if (
                    source_boundary_start <= component_start
                    and component_end <= source_boundary_end
                ):
                    component_aliases.append((saved_idx, alias))
            if not component_aliases:
                continue

            anchor_storage = None
            anchor_shift = None
            for saved_idx, alias in component_aliases:
                _, kind, boundary_idx, relative_offset, _ = alias
                if kind == "input":
                    boundary = per_callable_static_input_surfaces[func_idx][boundary_idx]
                else:
                    boundary = outputs[boundary_idx]
                if not isinstance(boundary, torch.Tensor) or not boundary.is_cuda:
                    raise RuntimeError(
                        f"CUDA graph {kind} boundary {boundary_idx} is not a CUDA tensor."
                    )

                spec = plan[saved_idx]
                storage = boundary.untyped_storage()
                boundary_start = boundary.storage_offset() * boundary.element_size()
                shift = boundary_start + relative_offset - spec[9]
                if anchor_storage is None:
                    anchor_storage = storage
                    anchor_shift = shift
                elif anchor_storage._cdata != storage._cdata or anchor_shift != shift:
                    raise RuntimeError(
                        "CUDA graph overlapping saved tensors have inconsistent boundary "
                        f"aliases: func={func_idx}, saved={component_saved_indices}."
                    )

            for saved_idx in component_saved_indices:
                spec = plan[saved_idx]
                target_offset = spec[9] + anchor_shift
                if target_offset < 0 or target_offset + spec[7] > anchor_storage.nbytes():
                    raise RuntimeError(
                        "CUDA graph boundary-backed saved component does not fit its replay "
                        f"storage: func={func_idx}, saved={saved_idx}, "
                        f"offset={target_offset}, bytes={spec[7]}, "
                        f"storage_bytes={anchor_storage.nbytes()}."
                    )
                itemsize = spec[4].itemsize
                if target_offset % itemsize:
                    raise RuntimeError(
                        f"CUDA graph boundary-backed saved tensor {saved_idx} is unaligned."
                    )
                target = torch.empty((0,), dtype=spec[4], device=spec[5]).set_(
                    anchor_storage,
                    target_offset // itemsize,
                    spec[2],
                    spec[3],
                )
                target.requires_grad_(spec[6])
                targets[saved_idx] = target
        return tuple(targets)

    def slot_tensor_targets(plan, arena=None):
        """Lay out graph-boundary tensors contiguously in an arena."""
        targets = []
        offset = 0
        for spec in plan:
            if spec is None:
                targets.append(None)
                continue
            offset = _align_up(offset)
            target = None
            if arena is not None:
                target = _arena_view(arena, offset, spec)
            targets.append(target)
            offset += spec[7]
        return tuple(targets), _align_up(offset)

    slot_saved_arenas = {}
    per_callable_slot_saved_targets = None
    if use_slot_memory:
        arena_sizes = {}
        for func_idx, plan in enumerate(per_callable_saved_tensor_plans):
            output_plan = per_callable_output_tensor_plans[func_idx]
            _, output_bytes = slot_tensor_targets(output_plan)
            temporary_arena = None
            if output_bytes:
                with torch.cuda.use_mem_pool(slot_allocator_pool):
                    temporary_arena = torch.empty(
                        (output_bytes,), dtype=torch.uint8, device=torch.cuda.current_device()
                    )
            temporary_outputs, _ = slot_tensor_targets(output_plan, temporary_arena)
            preassigned_targets = semantic_boundary_alias_targets(func_idx, temporary_outputs)
            spill_bytes = plan_native_saved_alias_targets(
                plan,
                (),
                measure_spill=True,
                preassigned_targets=preassigned_targets,
            )
            arena_id = saved_tensor_arena_ids[func_idx]
            arena_sizes[arena_id] = max(arena_sizes.get(arena_id, 0), output_bytes + spill_bytes)
            del temporary_outputs, temporary_arena, preassigned_targets

        with torch.cuda.use_mem_pool(slot_allocator_pool):
            slot_saved_arenas = {
                arena_id: torch.empty(
                    (required_bytes,), dtype=torch.uint8, device=torch.cuda.current_device()
                )
                for arena_id, required_bytes in arena_sizes.items()
                if required_bytes > 0
            }

        per_callable_slot_saved_targets = []
        for func_idx, plan in enumerate(per_callable_saved_tensor_plans):
            arena_id = saved_tensor_arena_ids[func_idx]
            arena = slot_saved_arenas.get(arena_id)
            output_targets, output_bytes = slot_tensor_targets(
                per_callable_output_tensor_plans[func_idx], arena
            )
            per_callable_output_tensor_targets[func_idx] = list(output_targets)
            preassigned_targets = semantic_boundary_alias_targets(func_idx, output_targets)
            protected_ranges = {}
            if arena is not None and output_bytes:
                protected_ranges[arena.untyped_storage()._cdata] = ((0, output_bytes),)
            canonical_targets = () if arena is None else (arena,)
            per_callable_slot_saved_targets.append(
                plan_native_saved_alias_targets(
                    plan,
                    canonical_targets,
                    protected_storage_ranges=protected_ranges,
                    preassigned_targets=preassigned_targets,
                )
            )

    slot_user_grad_arenas = {}
    if use_slot_memory:
        slot_sizes = {}
        for func_idx, plan in enumerate(per_callable_user_grad_tensor_plans):
            _, required_bytes = slot_tensor_targets(plan)
            arena_id = user_grad_arena_ids[func_idx]
            slot_sizes[arena_id] = max(slot_sizes.get(arena_id, 0), required_bytes)

        with torch.cuda.use_mem_pool(slot_allocator_pool):
            slot_user_grad_arenas = {
                arena_id: torch.empty(
                    (required_bytes,), dtype=torch.uint8, device=torch.cuda.current_device()
                )
                for arena_id, required_bytes in slot_sizes.items()
                if required_bytes > 0
            }

        for func_idx, plan in enumerate(per_callable_user_grad_tensor_plans):
            arena_id = user_grad_arena_ids[func_idx]
            targets, _ = slot_tensor_targets(plan, slot_user_grad_arenas.get(arena_id))
            per_callable_user_grad_tensor_targets[func_idx] = list(targets)

    @contextlib.contextmanager
    def capture_saved_tensors(func_idx, alias_targets=None):
        """Capture forward tensors that cross the graph's F/B boundary."""
        if per_callable_saved_tensor_plans is None:
            yield
            return

        plan = per_callable_saved_tensor_plans[func_idx]
        saved_idx = 0
        if alias_targets is not None and len(alias_targets) != len(plan):
            raise RuntimeError(
                f"CUDA graph input {func_idx} changed its canonical saved-target count."
            )

        def pack_saved_tensor(tensor):
            nonlocal saved_idx
            if saved_idx >= len(plan):
                raise RuntimeError(
                    f"CUDA graph input {func_idx} saved more forward tensors during capture "
                    "than warmup."
                )
            current_saved_idx = saved_idx
            spec = plan[current_saved_idx]
            saved_idx += 1
            if spec[2:8] != _saved_tensor_signature(tensor):
                raise RuntimeError(
                    f"CUDA graph input {func_idx} changed forward saved-tensor layout "
                    "during capture."
                )
            if spec[0] == "external":
                if spec[1] != _tensor_storage_ptr(tensor):
                    raise RuntimeError(
                        f"CUDA graph input {func_idx} changed an external saved tensor."
                    )
                return tensor
            if spec[0] != "native":
                raise RuntimeError(
                    f"CUDA graph input {func_idx} has unsupported saved-tensor mode {spec[0]}."
                )

            source_storage = tensor.untyped_storage()
            per_callable_native_saved_storages[func_idx].setdefault(
                source_storage._cdata, (source_storage, source_storage.data_ptr())
            )
            if alias_targets is None:
                target = torch.empty((0,), dtype=tensor.dtype, device=tensor.device).set_(
                    tensor.untyped_storage(),
                    tensor.storage_offset(),
                    tensor.shape,
                    tensor.stride(),
                )
                target.requires_grad_(tensor.requires_grad)
            else:
                target = alias_targets[current_saved_idx]
                if target is None:
                    raise RuntimeError(
                        f"CUDA graph input {func_idx} has no canonical target for native "
                        f"saved tensor {current_saved_idx}."
                    )
                same_view = (
                    target.data_ptr() == tensor.data_ptr()
                    and target.shape == tensor.shape
                    and target.stride() == tensor.stride()
                    and target.dtype == tensor.dtype
                )
                if not same_view:
                    with torch.no_grad():
                        target.copy_(tensor)
                tensor = target

            storage = tensor.untyped_storage()
            per_callable_native_saved_storages[func_idx].setdefault(
                storage._cdata, (storage, storage.data_ptr())
            )
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack_saved_tensor, lambda x: x):
            yield
        if saved_idx != len(plan):
            raise RuntimeError(
                f"CUDA graph input {func_idx} saved {saved_idx} forward tensors during "
                f"capture, but saved {len(plan)} during warmup."
            )

    def validate_captured_module_grads(func_idx, static_grad_inputs):
        """Require capture to preserve every parameter gradient observed during warmup."""
        if per_callable_saved_tensor_plans is None:
            return
        module_params = per_callable_module_params[func_idx]
        module_grad_inputs = static_grad_inputs[per_callable_len_user_args[func_idx] :]
        if len(module_grad_inputs) != len(module_params):
            raise RuntimeError(
                f"CUDA graph input {func_idx} captured {len(module_grad_inputs)} parameter "
                f"gradient slots for {len(module_params)} parameters."
            )
        missing_params = [
            param for param, grad in zip(module_params, module_grad_inputs) if grad is None
        ]
        if not missing_params:
            return

        func = graph_callables[func_idx]
        param_names = {}
        if isinstance(func, torch.nn.Module):
            param_names = {id(param): name for name, param in func.named_parameters()}
        missing_names = [
            param_names.get(id(param), f"<unnamed shape={tuple(param.shape)}>")
            for param in missing_params
        ]
        raise RuntimeError(
            f"CUDA graph input {func_idx} lost parameter gradients during capture: {missing_names}."
        )

    # All captures here share a mempool. To avoid replays corrupting each other's memory,
    # the safest approach is to capture all passes in the same order they'll run:
    # fwd 1, fwd 2, ... fwd N, then bwd N, bwd N-1, ... bwd 1.

    branch_capture_groups = None
    branch_checkpoint_state = None
    branch_checkpoint_live_blocks = None
    branch_checkpoint_storage_owners = None
    branch_pre_checkpoint_state = None
    branch_pre_checkpoint_live_blocks = None
    branch_pre_checkpoint_storage_owners = None
    current_storage_owners = None
    native_saved_alias_targets = None
    if use_slot_memory:
        branch_capture_groups = [None] * len(_order)
        group_records = []
        group_fwd_idx = [0] * num_model_chunks
        group_bwd_idx = [0] * num_model_chunks
        for order_idx, c_id in enumerate(_order):
            if ceil(c_id) != c_id:
                group_records.append(None)
                continue
            m_chunk = abs(int(c_id)) - 1
            logical_idx = group_fwd_idx[m_chunk] if c_id > 0 else group_bwd_idx[m_chunk]
            first_func_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                logical_idx * _num_layers_per_chunk[m_chunk]
            )
            slot = _graph_memory_slots[first_func_idx]
            event_key = (
                c_id > 0,
                slot[3],
                logical_idx,
                slot[1],
                _num_layers_per_chunk[m_chunk],
            )
            group_records.append((event_key, slot[2]))
            if c_id > 0:
                group_fwd_idx[m_chunk] += 1
            else:
                group_bwd_idx[m_chunk] += 1

        group_start = 0
        while group_start < len(group_records):
            record = group_records[group_start]
            group_stop = group_start + 1
            while (
                record is not None
                and group_stop < len(group_records)
                and group_records[group_stop] is not None
                and group_records[group_stop][0] == record[0]
            ):
                group_stop += 1
            group_size = group_stop - group_start
            if group_size > 1:
                branch_ids = [group_records[idx][1] for idx in range(group_start, group_stop)]
                if len(set(branch_ids)) != group_size:
                    raise RuntimeError(
                        "CUDA graph slot checkpoint group contains duplicate branch IDs: "
                        f"{branch_ids}."
                    )
                for position, order_idx in enumerate(range(group_start, group_stop)):
                    branch_capture_groups[order_idx] = (position, group_size)
            group_start = group_stop

    def slot_pool_active_blocks():
        """Return active allocation addresses and sizes in the graph-private pool."""
        snapshot = slot_allocator_pool.snapshot(include_traces=False)
        segments = snapshot["segments"] if isinstance(snapshot, dict) else snapshot
        return {
            block["address"]: block["size"]
            for segment in segments
            for block in segment["blocks"]
            if block["state"] == "active_allocated"
        }

    def slot_pool_layout():
        """Return the allocator block topology for checkpoint diagnostics."""
        snapshot = slot_allocator_pool.snapshot(include_traces=False)
        segments = snapshot["segments"] if isinstance(snapshot, dict) else snapshot
        return {
            segment["address"]: {
                "total_size": segment["total_size"],
                "blocks": [
                    (block["address"], block["size"], block["state"]) for block in segment["blocks"]
                ],
            }
            for segment in segments
        }

    def drain_slot_pool_pending_frees():
        """Poll completed cross-stream frees before restoring allocator state."""
        layout = slot_pool_layout()
        if not any(
            state == "active_pending_free"
            for segment in layout.values()
            for _, _, state in segment["blocks"]
        ):
            return

        # A non-capturing allocator request runs process_events(). The request stays in
        # the default pool, so polling cannot split or merge the slot-private pool.
        event_poll = torch.empty((1,), dtype=torch.uint8, device=torch.cuda.current_device())
        del event_poll
        remaining = [
            (address, size)
            for segment in slot_pool_layout().values()
            for address, size, state in segment["blocks"]
            if state == "active_pending_free"
        ]
        if remaining:
            raise RuntimeError(
                f"CUDA graph slot checkpoint could not drain pending allocator frees: {remaining}."
            )

    def assert_slot_pool_liveness(expected_blocks, phase):
        expected_ptrs = set(expected_blocks)
        if torch._C._cuda_checkPoolLiveAllocations(
            torch.cuda.current_device(), mempool, expected_ptrs
        ):
            return
        actual_blocks = slot_pool_active_blocks()
        added = set(actual_blocks).difference(expected_ptrs)
        removed = expected_ptrs.difference(actual_blocks)
        raise RuntimeError(
            f"CUDA graph slot checkpoint changed live allocations during {phase}: "
            f"added={len(added)} ({sum(actual_blocks[ptr] for ptr in added)} bytes), "
            f"added_sizes={sorted(actual_blocks[ptr] for ptr in added)}, "
            f"removed={len(removed)} ({sum(expected_blocks[ptr] for ptr in removed)} bytes), "
            f"removed_sizes={sorted(expected_blocks[ptr] for ptr in removed)}."
        )

    def record_replaced_io_storages(sources, targets, stale_storages):
        """Keep source storages whose graph-boundary tensors now use canonical storage."""
        for source, target in zip(sources, targets):
            if not (
                isinstance(source, torch.Tensor)
                and isinstance(target, torch.Tensor)
                and source.is_cuda
                and target.is_cuda
            ):
                continue
            source_storage = source.untyped_storage()
            target_storage = target.untyped_storage()
            if (
                source_storage.data_ptr() == target_storage.data_ptr()
                and torch._C._has_Standard_Deleter(target_storage._cdata)
            ):
                continue
            stale_storages[source_storage._cdata] = (
                source_storage,
                source_storage.data_ptr(),
            )

    def checkpoint_live_storage_owners(expected_blocks, phase, extra_values=()):
        """Resolve every checkpoint-live allocation to its owning StorageImpl."""
        owners = {}
        visited = set()
        visited_storage_impls = set()
        block_starts = sorted(expected_blocks)

        def record_storage(storage):
            if storage.device.type != "cuda" or storage._cdata in visited_storage_impls:
                return
            visited_storage_impls.add(storage._cdata)
            storage_ptr = storage.data_ptr()
            block_idx = bisect_right(block_starts, storage_ptr) - 1
            if block_idx < 0:
                return
            block_ptr = block_starts[block_idx]
            if storage_ptr >= block_ptr + expected_blocks[block_ptr]:
                return
            if torch._C._has_Standard_Deleter(storage._cdata):
                previous = owners.setdefault(block_ptr, storage)
                if previous._cdata != storage._cdata:
                    raise RuntimeError(
                        "CUDA graph slot checkpoint found multiple owning storages "
                        f"for allocation {block_ptr}."
                    )

        def visit(value):
            if value is None or id(value) in visited:
                return
            if isinstance(value, torch.UntypedStorage):
                record_storage(value)
                return
            visited.add(id(value))
            if isinstance(value, torch.Tensor):
                if not value.is_cuda:
                    return
                storage = value.untyped_storage()
                record_storage(storage)
                if value.is_leaf:
                    visit(value.grad)
                for child in vars(value).values():
                    visit(child)
                return
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
                return
            if isinstance(value, (list, tuple, set)):
                for child in value:
                    visit(child)

        for value in (
            sample_args,
            flatten_sample_args,
            per_callable_static_input_surfaces,
            per_callable_static_outputs,
            per_callable_static_grad_outputs,
            per_callable_static_grad_inputs,
            per_callable_native_saved_storages,
            per_callable_output_tensor_targets,
            per_callable_user_grad_tensor_targets,
            per_callable_param_grad_tensor_targets,
            static_grad_outputs_dict,
            native_io_anchors,
            slot_saved_arenas,
            slot_user_grad_arenas,
            *extra_values,
        ):
            visit(value)
        for func in graph_callables:
            if isinstance(func, torch.nn.Module):
                for module in func.modules():
                    visit(vars(module))

        missing = set(expected_blocks).difference(owners)
        if missing:
            raise RuntimeError(
                f"CUDA graph slot checkpoint could not resolve owning storages during {phase} "
                "for "
                f"{len(missing)} live allocations with sizes "
                f"{sorted(expected_blocks[ptr] for ptr in missing)}."
            )
        return owners

    def validate_checkpoint_live_storages(expected_blocks, owners):
        """Validate checkpoint owners before changing any StorageImpl."""
        if set(owners) != set(expected_blocks):
            raise RuntimeError("CUDA graph slot checkpoint live-storage owner set changed.")
        for block_ptr, storage in owners.items():
            storage_ptr = storage.data_ptr()
            if not block_ptr <= storage_ptr < block_ptr + expected_blocks[block_ptr]:
                raise RuntimeError(
                    "CUDA graph slot checkpoint live storage changed its allocation."
                )
            if not torch._C._has_Standard_Deleter(storage._cdata):
                raise RuntimeError(
                    "CUDA graph slot checkpoint live storage lost its allocator deleter."
                )

    def release_checkpoint_live_storages(expected_blocks, owners, phase):
        """Make prevalidated checkpoint-live blocks free for topology restoration."""
        for storage in owners.values():
            storage_ptr = storage.data_ptr()
            torch._C._free_And_Remove_DeleterFn(storage._cdata)
            tex._graph_checkpoint_detach_storage(storage._cdata)
            if storage.data_ptr() != storage_ptr or torch._C._has_Standard_Deleter(storage._cdata):
                raise RuntimeError(
                    "CUDA graph slot checkpoint could not detach live-storage ownership."
                )

        torch.cuda.synchronize()
        drain_slot_pool_pending_frees()
        remaining = slot_pool_active_blocks()
        if remaining:
            raise RuntimeError(
                f"CUDA graph slot checkpoint retained {len(remaining)} active allocations "
                f"during {phase}."
            )
        return [storage._cdata for storage in owners.values()]

    def verify_checkpoint_live_storages(expected_blocks, owners):
        """Verify allocator ownership was restored onto the original StorageImpls."""
        for block_ptr, storage in owners.items():
            if not torch._C._has_Standard_Deleter(storage._cdata):
                raise RuntimeError(
                    "CUDA graph slot checkpoint did not restore the live-storage deleter."
                )
            storage_ptr = storage.data_ptr()
            if not block_ptr <= storage_ptr < block_ptr + expected_blocks[block_ptr]:
                raise RuntimeError(
                    "CUDA graph slot checkpoint restored a live storage at the wrong address."
                )

    def restore_slot_pool_boundary(
        current_blocks,
        current_owners,
        target_state,
        target_blocks,
        target_owners,
        phase,
    ):
        """Switch between two allocator boundaries while preserving their StorageImpls."""
        device = torch.cuda.current_device()
        rollback_state = torch._C._cuda_getCheckpointState(device, mempool)
        current_release_started = False
        try:
            # Complete validation before the transaction starts. Once the first owner can be
            # mutated, every failure must restore the original allocator boundary.
            validate_checkpoint_live_storages(current_blocks, current_owners)
            current_release_started = True
            release_checkpoint_live_storages(current_blocks, current_owners, phase)
            torch._C._cuda_setCheckpointPoolState(
                device,
                target_state,
                [],
                [storage._cdata for storage in target_owners.values()],
            )
            verify_checkpoint_live_storages(target_blocks, target_owners)
            assert_slot_pool_liveness(target_blocks, phase)
        except BaseException as restore_error:
            if not current_release_started:
                raise
            try:
                rollback_blocks = slot_pool_active_blocks()
                rollback_owners = checkpoint_live_storage_owners(
                    rollback_blocks,
                    f"{phase} rollback",
                    (current_owners, target_owners),
                )
                validate_checkpoint_live_storages(rollback_blocks, rollback_owners)
                release_checkpoint_live_storages(
                    rollback_blocks, rollback_owners, f"{phase} rollback"
                )
                torch._C._cuda_setCheckpointPoolState(
                    device,
                    rollback_state,
                    [],
                    [storage._cdata for storage in current_owners.values()],
                )
                verify_checkpoint_live_storages(current_blocks, current_owners)
                assert_slot_pool_liveness(current_blocks, f"{phase} rollback")
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"CUDA graph slot checkpoint rollback failed after {restore_error!r}."
                ) from rollback_error
            raise

    if _order is not None:  # pylint: disable=too-many-nested-blocks
        per_callable_static_outputs = [None] * len(flatten_sample_args)
        per_callable_output_unflatten_spec = [None] * len(flatten_sample_args)
        per_callable_static_grad_outputs = [None] * len(flatten_sample_args)
        per_callable_static_grad_inputs = [None] * len(flatten_sample_args)
        fwd_idx = [0] * num_model_chunks
        bwd_idx = [0] * num_model_chunks
        static_grad_outputs_dict = {}
        wgrad_validation_list = [None] * len(_order)
        previous_chunk_last_callable_bwd_idx = None
        for i, c_id in enumerate(_order):
            branch_group = branch_capture_groups[i] if branch_capture_groups is not None else None
            deferred_native_output_target_releases = set()
            deferred_native_grad_target_releases = set()
            captured_branch_func_indices = []
            branch_stale_storages = {}
            branch_boundary_values = []
            branch_checkpoint_active = any(
                value is not None
                for value in (
                    branch_checkpoint_state,
                    branch_pre_checkpoint_state,
                    current_storage_owners,
                    native_saved_alias_targets,
                )
            )
            if branch_group is not None and branch_group[0] == 0 and branch_checkpoint_active:
                raise RuntimeError("CUDA graph slot checkpoint groups overlap.")
            if branch_group is not None:
                if branch_group[0] == 0:
                    gc.collect()
                    torch.cuda.synchronize()
                    drain_slot_pool_pending_frees()
                    branch_pre_checkpoint_state = torch._C._cuda_getCheckpointState(
                        torch.cuda.current_device(), mempool
                    )
                    branch_pre_checkpoint_live_blocks = slot_pool_active_blocks()
                    branch_pre_checkpoint_storage_owners = checkpoint_live_storage_owners(
                        branch_pre_checkpoint_live_blocks,
                        f"{'forward' if c_id > 0 else 'backward'} pre-branch "
                        f"boundary at order index {i}",
                    )
                elif branch_group[0] > 1 and c_id < 0:
                    restore_slot_pool_boundary(
                        branch_checkpoint_live_blocks,
                        branch_checkpoint_storage_owners,
                        branch_pre_checkpoint_state,
                        branch_pre_checkpoint_live_blocks,
                        branch_pre_checkpoint_storage_owners,
                        f"{'forward' if c_id > 0 else 'backward'} branch "
                        f"{branch_group[0] + 1}/{branch_group[1]} restore pre-boundary",
                    )
            if c_id > 0:
                if not isinstance(c_id, int):
                    raise TypeError(
                        f"Forward order value must be an integer, but got {type(c_id).__name__}."
                    )
                # Capture forward graph for model chunk c_id, microbatch fwd_idx[c_id-1]
                m_chunk = c_id - 1
                for l_no in range(_num_layers_per_chunk[m_chunk]):
                    func = callables[_prefix_num_layers[m_chunk] + l_no]
                    per_callable_fwd_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        fwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    captured_branch_func_indices.append(per_callable_fwd_idx)
                    args = sample_args[per_callable_fwd_idx]
                    kwargs = sample_kwargs[per_callable_fwd_idx]
                    fwd_graph = fwd_graphs[per_callable_fwd_idx]
                    native_saved_alias_targets = (
                        per_callable_slot_saved_targets[per_callable_fwd_idx]
                        if use_slot_memory
                        else None
                    )
                    with _graph_context_wrapper(fwd_graph, pool=mempool):
                        with capture_saved_tensors(
                            per_callable_fwd_idx, native_saved_alias_targets
                        ):
                            outputs = func(*args, **kwargs)
                        flatten_outputs, spec = _tree_flatten(outputs)
                        original_flatten_outputs = flatten_outputs
                        flatten_outputs = copy_outputs_to_slot_arena(
                            per_callable_fwd_idx, flatten_outputs
                        )
                        if branch_group is not None:
                            record_replaced_io_storages(
                                original_flatten_outputs,
                                flatten_outputs,
                                branch_stale_storages,
                            )
                            if branch_group[0] > 0:
                                branch_stale_storages.update(
                                    per_callable_native_saved_storages[per_callable_fwd_idx]
                                )
                        del original_flatten_outputs
                    native_saved_alias_targets = None
                    per_callable_static_outputs[per_callable_fwd_idx] = tuple(flatten_outputs)
                    per_callable_output_unflatten_spec[per_callable_fwd_idx] = spec
                    graph_callables[per_callable_fwd_idx] = func
                    if use_slot_memory:
                        del outputs
                    del flatten_outputs
                fwd_idx[m_chunk] += 1
            else:
                # Capture backward graph for model chunk c_id, microbatch bwd_idx[-c_id-1]
                m_chunk = -ceil(c_id) - 1
                previous_per_callable_bwd_idx = None
                for l_no in list(reversed(range(_num_layers_per_chunk[m_chunk]))):
                    per_callable_bwd_idx = (_prefix_num_layers[m_chunk] * num_microbatches) + (
                        bwd_idx[m_chunk] * _num_layers_per_chunk[m_chunk] + l_no
                    )
                    captured_branch_func_indices.append(per_callable_bwd_idx)
                    if ceil(c_id) == c_id and need_bwd_dw_graph[per_callable_bwd_idx]:
                        # Check if bwd graph has corresponding wgrad graph:
                        # Number of dgrad backward graphs should be equal to number of
                        # wgrad backward graphs.
                        # Note: For MCore, the validation rule is more strict (the next backward
                        # of dgrad graph must be corresponding wgrad graph).
                        if wgrad_validation_list[i] is None:
                            same_bwd_c_id_list = [i]
                            num_wgrad_c_id = 0
                            for idx in range(i + 1, len(_order)):
                                if _order[idx] > 0:
                                    continue
                                if _order[idx] == c_id:
                                    same_bwd_c_id_list.append(idx)
                                if _order[idx] + 0.5 == c_id:
                                    num_wgrad_c_id += 1
                                if len(same_bwd_c_id_list) == num_wgrad_c_id:
                                    for same_c_id_idx in same_bwd_c_id_list:
                                        wgrad_validation_list[same_c_id_idx] = True
                                    break
                                if len(same_bwd_c_id_list) < num_wgrad_c_id:
                                    # It's impossible to have more wgrad than dgrad.
                                    wgrad_validation_list[i] = False
                                    break
                            if wgrad_validation_list[i] is None:
                                wgrad_validation_list[i] = False
                            if not wgrad_validation_list[i]:
                                raise RuntimeError(
                                    f"Number of wgrad graph({num_wgrad_c_id}) doesn't match number "
                                    f"of dgrad graphs ({len(same_bwd_c_id_list)}) for chunk {c_id}."
                                )
                    elif ceil(c_id) != c_id:
                        per_callable_bwd_idx -= _num_layers_per_chunk[m_chunk]
                        if not is_training:
                            raise RuntimeError("Only training mode supports backward_dw.")
                        # If no one module needs the backward_dw, the bwd_dw_graph will be empty.
                        # So skip capturing it. For backward_dw, the order value is c_id - 0.5 to indicate
                        # the specific order of backward_dw.
                        if ceil(c_id) - c_id != 0.5:
                            raise ValueError(
                                "The order diff of wgrad and dgrad must be 0.5, "
                                f"get {ceil(c_id) - c_id}."
                            )
                        if not need_bwd_dw_graph[per_callable_bwd_idx]:
                            raise RuntimeError(
                                "No module needs wgrad computation but get float in order"
                            )
                        bwd_dw_graph = bwd_dw_graphs[per_callable_bwd_idx]
                        with _graph_context_wrapper(bwd_dw_graph, pool=mempool):
                            for module in visited_te_modules[per_callable_bwd_idx]:
                                if (
                                    hasattr(module, "need_backward_dw")
                                    and module.need_backward_dw()
                                ):
                                    module.backward_dw()
                        continue

                    static_input_surface = per_callable_static_input_surfaces[per_callable_bwd_idx]
                    static_outputs = per_callable_static_outputs[per_callable_bwd_idx]
                    bwd_graph = bwd_graphs[per_callable_bwd_idx]
                    # For now, assumes all static_outputs require grad
                    if _reuse_graph_input_output_buffers:
                        # Note for _reuse_graph_input_output_buffers: grad output is only used
                        # within backward, so we can reuse the same static buffers every time.
                        static_grad_outputs_keys = tuple(
                            (o.shape, o.dtype, o.layout)
                            for o in static_outputs
                            if o is not None and o.requires_grad
                        )
                        if static_grad_outputs_keys in static_grad_outputs_dict:
                            static_grad_outputs = static_grad_outputs_dict[static_grad_outputs_keys]
                        else:
                            static_grad_outputs = tuple(
                                (torch.empty_like(o) if o is not None and o.requires_grad else None)
                                for o in static_outputs
                            )
                            static_grad_outputs_dict[static_grad_outputs_keys] = static_grad_outputs
                    else:
                        static_grad_outputs = tuple(
                            (torch.empty_like(o) if o is not None and o.requires_grad else None)
                            for o in static_outputs
                        )
                    if is_training:
                        inputs = tuple(i for i in static_input_surface if i.requires_grad)
                        with _none_grad_context_wrapper(inputs), _graph_context_wrapper(
                            bwd_graph, pool=mempool
                        ):
                            torch.autograd.backward(
                                tuple(
                                    o for o in static_outputs if o is not None and o.requires_grad
                                ),
                                grad_tensors=tuple(o for o in static_grad_outputs if o is not None),
                                retain_graph=retain_graph_in_backward,
                            )
                            grad_inputs = tuple(input.grad for input in inputs)
                            original_grad_inputs = grad_inputs
                            grad_inputs = copy_user_grads_to_slot_arena(
                                per_callable_bwd_idx, static_input_surface, grad_inputs
                            )
                            if branch_group is not None and branch_group[0] > 0:
                                record_replaced_io_storages(
                                    original_grad_inputs,
                                    grad_inputs,
                                    branch_stale_storages,
                                )
                            del original_grad_inputs

                    # Constructs a tuple suitable for returning from Graphed.backward:
                    # Pads out the actually-needed grads with Nones in gradient slots for inputs
                    # that don't require grad. I couldn't think of a one-liner for this pattern.
                    static_grad_inputs = []
                    grad_idx = 0
                    fused_wgrad_params = per_callable_fused_wgrad_params.get(
                        per_callable_bwd_idx, set()
                    )
                    for arg in static_input_surface:
                        if is_training and isinstance(arg, torch.Tensor) and arg.requires_grad:
                            grad_input = grad_inputs[grad_idx]
                            grad_idx += 1
                            if grad_input is None and arg in fused_wgrad_params:
                                main_grad = getattr(arg, "main_grad", arg)
                                grad_input = get_dummy_wgrad(
                                    list(main_grad.shape),
                                    arg.dtype,
                                    zero=getattr(arg, "zero_out_wgrad", False),
                                )
                            static_grad_inputs.append(grad_input)
                        else:
                            static_grad_inputs.append(None)  # type: ignore[arg-type]
                    static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]
                    validate_captured_module_grads(per_callable_bwd_idx, static_grad_inputs)

                    per_callable_static_grad_outputs[per_callable_bwd_idx] = static_grad_outputs
                    per_callable_static_grad_inputs[per_callable_bwd_idx] = static_grad_inputs
                    if branch_group is not None:
                        branch_boundary_values.append(static_grad_inputs)

                    # Weak-ref static output and gradient objects after their capture lifetime.
                    # Their backing storage remains alive either in the graph pool or an explicit
                    # slot arena, while transient graph-pool references can be reclaimed.
                    if _reuse_graph_input_output_buffers:
                        # Weak ref the static outputs of the forward pass of this backward. It's
                        # no longer needed after the corresponding backward graph is built up.
                        per_callable_static_outputs[per_callable_bwd_idx] = make_weak_ref(
                            static_outputs
                        )
                        if branch_group is None:
                            clear_native_io_target_rows((per_callable_bwd_idx,), clear_outputs=True)
                        else:
                            deferred_native_output_target_releases.add(per_callable_bwd_idx)

                        # Weak ref the static grad inputs of the previous backward pass within the
                        # same chunk.
                        if previous_per_callable_bwd_idx is not None:
                            idx = previous_per_callable_bwd_idx
                            per_callable_static_grad_inputs[idx] = make_weak_ref(
                                per_callable_static_grad_inputs[idx]
                            )
                            if branch_group is None:
                                clear_native_io_target_rows((idx,), clear_grads=True)
                            else:
                                deferred_native_grad_target_releases.add(idx)
                        previous_per_callable_bwd_idx = per_callable_bwd_idx

                        # Weak ref the static grad inputs of the previous chunk's last backward
                        # pass.
                        # Note: After a chunk's backward pass, we assume Mcore will send the grad
                        # input to another pipeline parallel rank and that the communication is
                        # finished before the end of the next chunk's backward pass.
                        if l_no == 0:
                            if previous_chunk_last_callable_bwd_idx is not None:
                                idx = previous_chunk_last_callable_bwd_idx
                                per_callable_static_grad_inputs[idx] = make_weak_ref(
                                    per_callable_static_grad_inputs[idx]
                                )
                                if branch_group is None:
                                    clear_native_io_target_rows((idx,), clear_grads=True)
                                else:
                                    deferred_native_grad_target_releases.add(idx)
                            previous_chunk_last_callable_bwd_idx = per_callable_bwd_idx
                    if use_slot_memory and branch_group is None:
                        per_callable_native_saved_storages[per_callable_bwd_idx].clear()
                    del static_outputs
                if ceil(c_id) == c_id:
                    bwd_idx[m_chunk] += 1

            if branch_group is not None:
                if c_id < 0:
                    for captured_func_idx in captured_branch_func_indices:
                        per_callable_static_grad_inputs[captured_func_idx] = make_weak_ref(
                            per_callable_static_grad_inputs[captured_func_idx]
                        )
                gc.collect()
                torch.cuda.synchronize()
                drain_slot_pool_pending_frees()
                if branch_group[0] == 0:
                    branch_checkpoint_state = torch._C._cuda_getCheckpointState(
                        torch.cuda.current_device(), mempool
                    )
                    branch_checkpoint_live_blocks = slot_pool_active_blocks()
                    branch_checkpoint_storage_owners = checkpoint_live_storage_owners(
                        branch_checkpoint_live_blocks,
                        f"{'forward' if c_id > 0 else 'backward'} branch "
                        f"1/{branch_group[1]} at order index {i}",
                        (branch_stale_storages, branch_boundary_values),
                    )
                    if c_id < 0:
                        restore_slot_pool_boundary(
                            branch_checkpoint_live_blocks,
                            branch_checkpoint_storage_owners,
                            branch_pre_checkpoint_state,
                            branch_pre_checkpoint_live_blocks,
                            branch_pre_checkpoint_storage_owners,
                            f"backward branch 1/{branch_group[1]} restore pre-boundary",
                        )
                else:
                    current_live_blocks = slot_pool_active_blocks()
                    current_storage_owners = checkpoint_live_storage_owners(
                        current_live_blocks,
                        f"{'forward' if c_id > 0 else 'backward'} branch "
                        f"{branch_group[0] + 1}/{branch_group[1]} before post-boundary restore",
                        (
                            branch_stale_storages,
                            branch_boundary_values,
                            branch_pre_checkpoint_storage_owners,
                            branch_checkpoint_storage_owners,
                        ),
                    )
                    restore_slot_pool_boundary(
                        current_live_blocks,
                        current_storage_owners,
                        branch_checkpoint_state,
                        branch_checkpoint_live_blocks,
                        branch_checkpoint_storage_owners,
                        f"{'forward' if c_id > 0 else 'backward'} branch "
                        f"{branch_group[0] + 1}/{branch_group[1]} restore post-boundary",
                    )
                    current_storage_owners = None
                clear_native_io_target_rows(
                    deferred_native_output_target_releases, clear_outputs=True
                )
                clear_native_io_target_rows(deferred_native_grad_target_releases, clear_grads=True)
                if c_id < 0:
                    for captured_func_idx in captured_branch_func_indices:
                        per_callable_native_saved_storages[captured_func_idx].clear()
                if branch_group[0] == branch_group[1] - 1:
                    branch_checkpoint_state = None
                    branch_checkpoint_live_blocks = None
                    branch_checkpoint_storage_owners = None
                    branch_pre_checkpoint_state = None
                    branch_pre_checkpoint_live_blocks = None
                    branch_pre_checkpoint_storage_owners = None
                    gc.collect()
                    torch.cuda.synchronize()
                    drain_slot_pool_pending_frees()

    else:
        # Capture forward graphs
        per_callable_static_outputs = []
        per_callable_output_unflatten_spec = []
        graph_id = 0
        for func, args, kwargs, fwd_graph in zip(callables, sample_args, sample_kwargs, fwd_graphs):
            with _graph_context_wrapper(fwd_graph, pool=mempool):
                outputs = func(*args, **kwargs)
            graph_callables[graph_id] = func
            graph_id += 1

            flatten_outputs, spec = _tree_flatten(outputs)
            per_callable_static_outputs.append(tuple(flatten_outputs))
            per_callable_output_unflatten_spec.append(spec)

        # Capture backward graphs in reverse order
        per_callable_static_grad_outputs = []
        per_callable_static_grad_inputs = []
        for (
            static_input_surface,
            static_outputs,
            bwd_graph,
            bwd_dw_graph,
            bwd_idx,
        ) in zip(
            reversed(per_callable_static_input_surfaces),
            reversed(per_callable_static_outputs),
            reversed(bwd_graphs),
            reversed(bwd_dw_graphs),
            reversed(range(len(per_callable_static_input_surfaces))),
        ):
            # For now, assumes all static_outputs require grad
            static_grad_outputs = tuple(
                torch.empty_like(o) if o is not None and o.requires_grad else None
                for o in static_outputs
            )
            if is_training:
                inputs = tuple(i for i in static_input_surface if i.requires_grad)
                with _none_grad_context_wrapper(inputs), _graph_context_wrapper(
                    bwd_graph, pool=mempool
                ):
                    torch.autograd.backward(
                        tuple(o for o in static_outputs if o is not None and o.requires_grad),
                        grad_tensors=tuple(o for o in static_grad_outputs if o is not None),
                        retain_graph=retain_graph_in_backward,
                    )
                    grad_inputs = tuple(input.grad for input in inputs)

                if need_bwd_dw_graph[bwd_idx]:
                    with _graph_context_wrapper(bwd_dw_graph, pool=mempool):
                        for module in visited_te_modules[bwd_idx]:
                            if hasattr(module, "need_backward_dw") and module.need_backward_dw():
                                module.backward_dw()
            # Constructs a tuple suitable for returning from Graphed.backward:
            # Pads out the actually-needed grads with Nones in gradient slots for inputs that
            # don't require grad. I couldn't think of a slick one-liner for this pattern.
            static_grad_inputs = []
            grad_idx = 0
            for arg in static_input_surface:
                if is_training and isinstance(arg, torch.Tensor) and arg.requires_grad:
                    static_grad_inputs.append(grad_inputs[grad_idx])
                    grad_idx += 1
                else:
                    static_grad_inputs.append(None)  # type: ignore[arg-type]
            static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]

            per_callable_static_grad_outputs.append(static_grad_outputs)
            per_callable_static_grad_inputs.append(static_grad_inputs)

        # Reverses the most recent two lists
        per_callable_static_grad_outputs = list(reversed(per_callable_static_grad_outputs))
        per_callable_static_grad_inputs = list(reversed(per_callable_static_grad_inputs))

    if branch_checkpoint_state is not None or branch_pre_checkpoint_state is not None:
        raise RuntimeError("CUDA graph capture ended inside a slot checkpoint group.")

    if allocator_settings_to_restore is not None:
        _allocator_settings_guard.restore()

    if use_slot_memory and (
        any(native_io_anchors.values()) or any(native_io_capture_counts.values())
    ):
        raise RuntimeError(
            "CUDA graph capture ended with incomplete native I/O aliases: "
            f"anchors={native_io_anchors}, counts={native_io_capture_counts}."
        )

    # Now for every per_callable list, per_callable_*[i] holds the stuff for the ith callable.

    def make_graphed_autograd_function(
        fwd_graph,
        bwd_graph,
        module_params,
        kwargs_keys,
        len_user_args,
        output_unflatten_spec,
        static_input_surface,
        static_outputs,
        static_grad_outputs,
        static_grad_inputs,
    ):
        class Graphed(torch.autograd.Function):
            """Autograd function for graph replay."""

            @staticmethod
            def forward(
                ctx,
                skip_fp8_weight_update,
                cuda_graph_stream,
                cuda_graph_event,
                *inputs,
            ):
                # pylint: disable=missing-function-docstring

                # Set flag for whether to update FP8 weight updates
                ctx.is_first_module = FP8GlobalStateManager.is_first_fp8_module()
                if ctx.is_first_module and skip_fp8_weight_update is not None:
                    FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(skip_fp8_weight_update)
                ctx.cuda_graph_stream = cuda_graph_stream
                ctx.cuda_graph_event = cuda_graph_event
                # Copy values from new tensors into static tensors
                for i in range(len_user_args):
                    if (
                        isinstance(static_input_surface[i], torch.Tensor)
                        and static_input_surface[i].data_ptr() != inputs[i].data_ptr()
                    ):
                        static_input_surface[i].copy_(inputs[i])

                # Replay forward graph
                if cuda_graph_stream != torch.cuda.current_stream():
                    cuda_graph_stream.wait_stream(torch.cuda.current_stream())
                    with cuda_graph_stream:
                        fwd_graph.replay()
                    if cuda_graph_event is not None:
                        torch.cuda.current_stream().wait_event(cuda_graph_event)
                    else:
                        torch.cuda.current_stream().wait_stream(cuda_graph_stream)
                else:
                    fwd_graph.replay()
                if not isinstance(static_outputs, tuple):
                    raise TypeError(
                        "Expected static_outputs to be a tuple, but got"
                        f" {type(static_outputs).__name__}"
                    )
                return tuple(o.detach() if o is not None else o for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grads):
                # pylint: disable=missing-function-docstring

                # Replay backward graph
                if len(grads) != len(static_grad_outputs):
                    raise ValueError(
                        "Backward graph grad dimension mismatch: "
                        f"received {len(grads)} grads, "
                        f"but expected {len(static_grad_outputs)} static_grad_outputs"
                    )
                for g, grad in zip(static_grad_outputs, grads):
                    if g is not None:
                        # don't copy if autograd gods have been kind and the
                        # incoming grad is already in the right place
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                if ctx.cuda_graph_stream != torch.cuda.current_stream():
                    ctx.cuda_graph_stream.wait_stream(torch.cuda.current_stream())
                    with ctx.cuda_graph_stream:
                        bwd_graph.replay()
                    if ctx.cuda_graph_event is not None:
                        torch.cuda.current_stream().wait_event(ctx.cuda_graph_event)
                    else:
                        torch.cuda.current_stream().wait_stream(ctx.cuda_graph_stream)
                else:
                    bwd_graph.replay()

                # Update FP8 scale factors if needed
                if ctx.is_first_module:
                    FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

                # Input args that didn't require grad expect a None gradient.
                if not isinstance(static_grad_inputs, tuple):
                    raise TypeError(
                        "Expected static_grad_inputs to be a tuple, but got"
                        f" {type(static_grad_inputs).__name__}"
                    )
                return (None, None, None) + tuple(
                    b.detach() if b is not None else b for b in static_grad_inputs
                )

        def functionalized(*user_args, **user_kwargs):

            # Decide whether to update FP8 weights
            skip_fp8_weight_update = None
            if cache_quantized_params:
                if "is_first_microbatch" not in user_kwargs or not isinstance(
                    user_kwargs["is_first_microbatch"], bool
                ):
                    raise ValueError(
                        "`is_first_microbatch` boolean kwarg must be provided for FP8 weight"
                        " caching."
                    )

                skip_fp8_weight_update = not user_kwargs["is_first_microbatch"]

            # The cuda_graph_stream and cuda_graph_event are used in the TE CUDA graph replay.
            # When replaying the graph in the cuda graph stream, the graph replay could overlap
            # with the work on main stream.
            # When cuda_graph_event is given, it should be an external event recorded
            # in the cuda graph and is used to sync-back to the main stream.
            # If cuda_graph_event is not given, it will be None and the graph replay will block
            # the main stream until it is finished.
            if "cuda_graph_stream" in user_kwargs:
                cuda_graph_stream = user_kwargs["cuda_graph_stream"]
                user_kwargs.pop("cuda_graph_stream")
            else:
                cuda_graph_stream = torch.cuda.current_stream()
            if "cuda_graph_event" in user_kwargs:
                cuda_graph_event = user_kwargs["cuda_graph_event"]
                user_kwargs.pop("cuda_graph_event")
            else:
                cuda_graph_event = None
            # Check that required kwargs are provided
            for key in kwargs_keys:
                if key not in user_kwargs:
                    raise TypeError(
                        f"Graphed callable was initialized with kwarg {key} ,"
                        "but it was not provided in graph replay"
                    )

            # Runs the autograd function with inputs == all inputs to
            # the graph that might require grad (explicit user args +
            # module parameters)
            # Assumes module params didn't change since capture.
            flatten_user_args, _ = _tree_flatten(user_args)
            flatten_user_kwargs, _ = _tree_flatten([user_kwargs[key] for key in kwargs_keys])
            func_args = tuple(flatten_user_args) + tuple(flatten_user_kwargs) + module_params
            out = Graphed.apply(
                skip_fp8_weight_update, cuda_graph_stream, cuda_graph_event, *func_args
            )
            return _tree_unflatten(out, output_unflatten_spec)

        return functionalized

    def make_graphed_attribute_functions(graph_idx):
        # Get te modules for current graph
        te_modules = visited_te_modules.get(graph_idx, set())

        # Attach backward_dw as an attribute to the graphed callable.
        def backward_dw():
            if need_bwd_dw_graph.get(graph_idx, False):
                bwd_dw_graphs[graph_idx].replay()

                # Trigger the grad accumulation hook for wgrad graphs.
                for module in te_modules:
                    if (
                        isinstance(module, TransformerEngineBaseModule)
                        and module.need_backward_dw()
                    ):
                        module._trigger_wgrad_accumulation_and_reduce_hooks()

        # Attach reset as an attribute to the graphed callable.
        def reset():
            fwd_graphs[graph_idx].reset()
            bwd_graphs[graph_idx].reset()
            bwd_dw_graphs[graph_idx].reset()

        return backward_dw, reset

    # Put together the final graphed callables
    ret = []
    for i in range(len(sample_args)):
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            bwd_graphs[i],
            per_callable_module_params[i],
            per_callable_kwargs_keys[i],
            per_callable_len_user_args[i],
            per_callable_output_unflatten_spec[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
            per_callable_static_grad_outputs[i],
            per_callable_static_grad_inputs[i],
        )

        func = graph_callables[i]
        te_modules = visited_te_modules.get(i, set())
        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd, te_modules):
                def new_fwd(*user_args, **user_kwargs):
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        # Set the FP8 group from global amax reduction.
                        if FP8GlobalStateManager.is_fp8_enabled():
                            fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
                            for m in func.modules():
                                if m not in te_modules:
                                    # Only Set the FP8 meta for the modules included by forward
                                    continue
                                if isinstance(m, TransformerEngineBaseModule):
                                    from transformer_engine.pytorch.attention.dot_product_attention import (
                                        DotProductAttention,
                                    )

                                    if (
                                        isinstance(m, DotProductAttention)
                                        and not fp8_recipe.fp8_mha
                                        and not fp8_recipe.fp8_dpa
                                    ):
                                        # Don't need to update FP8 meta for non-FP8 DPA
                                        continue
                                    m.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
                                    m.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()
                                    FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                                        m.fp8_meta,
                                    )
                                elif isinstance(m, BasicOperation):
                                    for mode in ("forward", "backward"):
                                        if m.num_quantizers(mode):
                                            m._fp8_metas[mode][
                                                "fp8_group"
                                            ] = FP8GlobalStateManager.get_fp8_group()
                                            m._fp8_metas[mode][
                                                "recipe"
                                            ] = FP8GlobalStateManager.get_fp8_recipe()
                                            FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                                                m._fp8_metas[mode],
                                            )
                        return graphed(*user_args, **user_kwargs)
                    return orig_fwd(*user_args, **user_kwargs)

                return new_fwd

            forward = make_graphed_forward(func, func.training, graphed, func.forward, te_modules)
            if _order is None:
                func.forward = forward
                ret.append(func)
            else:
                ret.append(forward)
        else:
            ret.append(graphed)

        backward_dw_func, reset_func = make_graphed_attribute_functions(i)
        setattr(ret[-1], "backward_dw", backward_dw_func)
        setattr(ret[-1], "reset", reset_func)
        if slot_allocator_pool is not None:
            setattr(ret[-1], "_te_cuda_graph_allocator_pool", slot_allocator_pool)
            setattr(ret[-1], "_te_cuda_graph_saved_arenas", slot_saved_arenas)
            setattr(ret[-1], "_te_cuda_graph_user_grad_arenas", slot_user_grad_arenas)

    if just_one_callable:
        return ret[0]

    return tuple(ret)


def save_fp8_tensors(
    modules: Iterable[torch.nn.Module],
    recipe: Optional[Recipe],
) -> Optional[List[Any]]:
    """
    Returns the FP8 tensors for all modules
    with adjusted amax history sizes.
    """

    if not isinstance(recipe, DelayedScaling):
        return None

    fp8_tensors = []
    for module in modules:
        for m in module.modules():
            module_tensors = None
            if isinstance(m, TransformerEngineBaseModule):
                if m.primary_weights_in_fp8:
                    m.adjust_amax_history_length(recipe.amax_history_len)
                module_tensors = m.get_fp8_meta_tensors()
            elif isinstance(m, BasicOperation):
                m.reset_recipe_state(recipe=recipe)
                module_tensors = m._save_fp8_metas()
            fp8_tensors.append(module_tensors)
    return fp8_tensors


def restore_fp8_tensors(
    modules: Iterable[torch.nn.Module],
    fp8_tensors: Optional[List[Any]],
) -> None:
    """Restore FP8 tensors."""

    if fp8_tensors is None:
        return

    for module in modules:
        for m in module.modules():
            module_tensors = fp8_tensors.pop(0)
            if isinstance(m, TransformerEngineBaseModule):
                m.reset_fp8_meta_tensors(module_tensors)
            elif isinstance(m, BasicOperation):
                m._load_fp8_metas(module_tensors)
    if len(fp8_tensors) != 0:
        raise RuntimeError(
            f"Got FP8 state for {len(fp8_tensors)} more modules than expected. "
            "There is probably a discrepancy with `save_fp8_tensors`."
        )


def make_graphed_callables(
    modules: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    fp8_enabled: Optional[SingleOrTuple[bool]] = None,
    fp8_calibrating: Optional[bool] = None,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    fp8_weight_caching: Optional[bool] = None,
    enabled: Optional[SingleOrTuple[bool]] = None,
    calibrating: Optional[bool] = None,
    recipe: Optional[Recipe] = None,
    amax_reduction_group: Optional[dist_group_type] = None,
    cache_quantized_params: Optional[bool] = None,
    _order: Optional[List[int]] = None,
    _num_layers_per_chunk: Optional[List[int]] = None,
    pool: Optional[Tuple[int, ...]] = None,
    retain_graph_in_backward: bool = False,
    _reuse_graph_input_output_buffers: bool = False,
    _graph_memory_slots: Optional[Sequence[Tuple[int, ...]]] = None,
    pre_warmup_hook: Optional[Callable] = None,
    post_warmup_hook: Optional[Callable] = None,
) -> Union[Callable, Tuple[Callable, ...]]:
    """
    Make CUDA graph version of Transformer Engine modules

    A variation of PyTorch's `make_graphed_callables` utility function
    with support for Transformer Engine modules and FP8. Please see
    the
    `original PyTorch implementation <https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html>`_
    for more documentation.

    .. warning::

       Arguments 'fp8_enabled', 'fp8_calibrating', 'fp8_recipe', 'fp8_group', and 'fp8_weight_caching' are deprecated.
       Use arguments 'enabled', 'calibrating', 'recipe', 'amax_reduction_group', and 'cache_quantized_params' instead.

    Graphing parameters
    -------------------
    modules: (tuple of) callable
             Callable or callables to graph.
    sample_args: (tuple of) tuple of torch.Tensor
                 Positional arguments to callable(s).
    num_warmup_iters: int, default = 3
                      Number of warmup iterations.
    allow_unused_input: bool, default = False
                        Whether to handle case where callable inputs
                        and outputs are disconnected in compute graph.
    sample_kwargs: (tuple of) dict, optional
                   Keyword arguments to callable(s)
    pool: (tuple of) int, default = None, optional
          An instance returned from function `torch.cuda.graph_pool_handle` that hints
          this graph may share memory with the indicated pool.
    retain_graph_in_backward: bool, default = False
                              Whether to set retain_graph=True in backward graph capture.
    _reuse_graph_input_output_buffers: bool, default = False
        Reduce memory usage by reusing input/output data buffers between
        graphs. Only supported with Mcore interleaved pipeline parallelism, i.e.
        when `_order` is provided. All callables in `modules` are assumed to have
        inputs and outputs with the same dtype and shape.
    _graph_memory_slots: sequence of 7-int tuples, default = None
        Private liveness plan for mutually exclusive graph variants. Each tuple describes
        the saved-tensor arena, physical I/O slot, I/O branch, model chunk, layer, and warmup
        alias group, followed by the returned user-gradient arena for one graph input. Requires
        the first positional sample argument of every graph input to be a plain CUDA tensor. Plain
        CUDA user inputs are snapshotted into the slot arenas whenever forward saves them for
        backward, so shape-identical graph inputs can safely share staging surfaces.
    pre_warmup_hook: callable, default = None
                      A hook function that will be called before the warmup iterations.
    post_warmup_hook: callable, default = None
                      A hook function that will be called after the warmup iterations.

    Quantization parameters
    -----------------------
    enabled: (tuple of) bool, default = False
             whether or not to enable low precision quantization (FP8/FP4).
             If tuple, the length must match the number of modules.
    calibrating: bool, default = False
                 calibration mode allows collecting statistics such as amax and scale
                 data of quantized tensors even when executing without quantization enabled.
                 This is useful for saving an inference ready checkpoint while training
                 using a higher precision.
    recipe: recipe.Recipe, default = None
            recipe used for low precision quantization.
    amax_reduction_group: torch._C._distributed_c10d.ProcessGroup, default = None
                          distributed group over which amaxes for the quantized tensors
                          are reduced at the end of each training step.
    cache_quantized_params: bool, default = False
                            Whether or not to cache quantized weights across microbatches. if set to `True`,
                            the `is_first_microbatch` boolean argument must be passed into the forward
                            method for TransformerEngine modules. When storing primary weights in low precision
                            using TE's `quantized_model_init` API and using an quantization aware optimizer,
                            this arg must be set to `False` if calculating weight transposes' outside TE, e.g.,
                            in the optimizer step.

    """

    # Handle deprecated args. If old kwargs are set, they are prioritized with warning.
    if fp8_enabled is not None:
        if enabled is not None:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_enabled` kwarg "
                "in favor of `enabled`, but both kwargs are set."
            )
        warnings.warn(
            "make_graphed_callables has deprecated `fp8_enabled` kwarg in favor of `enabled`. "
            "`fp8_enabled` will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        enabled = fp8_enabled
    if enabled is None:
        enabled = False

    if fp8_calibrating is not None:
        if calibrating is not None:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_calibrating` kwarg "
                "in favor of `calibrating`, but both kwargs are set."
            )
        warnings.warn(
            "make_graphed_callables has deprecated `fp8_calibrating` kwarg in favor of "
            "`calibrating`. `fp8_calibrating` will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        calibrating = fp8_calibrating
    if calibrating is None:
        calibrating = False

    if fp8_recipe is not None:
        if recipe is None:
            warnings.warn(
                "make_graphed_callables has deprecated `fp8_recipe` kwarg in favor of "
                "`recipe`. `fp8_recipe` will be removed in a future release.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_recipe` kwarg "
                "in favor of `recipe`, but both kwargs are set."
            )
        recipe = fp8_recipe

    if fp8_group is not None:
        if amax_reduction_group is None:
            warnings.warn(
                "make_graphed_callables has deprecated `fp8_group` kwarg in favor of "
                "`amax_reduction_group`. `fp8_group` will be removed in a future release.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_group` kwarg "
                "in favor of `amax_reduction_group`, but both kwargs are set."
            )
        amax_reduction_group = fp8_group

    if fp8_weight_caching is not None:
        if cache_quantized_params is not None:
            raise ValueError(
                "make_graphed_callables has deprecated `fp8_weight_caching` kwarg "
                "in favor of `cache_quantized_params`, but both kwargs are set."
            )
        warnings.warn(
            "make_graphed_callables has deprecated `fp8_weight_caching` kwarg in favor of "
            "`cache_quantized_params`. `fp8_weight_caching` will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        cache_quantized_params = fp8_weight_caching
    if cache_quantized_params is None:
        cache_quantized_params = False

    # Handle single module.
    just_one_callable = False
    if not isinstance(modules, tuple):
        just_one_callable = True
        modules = (modules,)

    if not isinstance(enabled, tuple):
        if not isinstance(enabled, bool):
            raise TypeError(
                f"enabled must be a bool or a tuple of bools, but got {type(enabled).__name__}"
            )
        enabled = (enabled,) * len(modules)
    else:
        if len(enabled) != len(modules):
            raise ValueError(
                f"enabled length ({len(enabled)}) must match modules length ({len(modules)})"
            )
    if any(enabled) and recipe is None:
        recipe = get_default_fp8_recipe()
    elif not any(enabled):
        recipe = None
    module_uses_fp8 = dict(zip((id(m) for m in modules), enabled))

    for module in modules:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"Graphing for {type(module)} is not supported.")

    # FP8 wrapper.
    old_call_funcs = {}

    def wrap_autocast(block):
        block_cls = type(block)
        if block_cls in old_call_funcs:
            return

        old_call_funcs[block_cls] = block_cls.__call__

        # Wrap the original call function of the module class.
        def call_func(self, *args, **kwargs):
            with autocast(
                enabled=module_uses_fp8.get(id(self), False),
                calibrating=calibrating,
                recipe=recipe,
                amax_reduction_group=amax_reduction_group,
                _graph=True,
            ):
                outputs = old_call_funcs[block_cls](self, *args, **kwargs)
            return outputs

        block_cls.__call__ = call_func

    allocator_settings_guard = _AllocatorSettingsGuard()
    saved_fp8_tensors = None
    fp8_state_saved = False
    rng_restore_callbacks = []
    capture_started = False
    try:
        # Store all process-wide state before capture and register enough information to restore
        # anything that was already changed if a later preparation step raises.
        saved_fp8_tensors = save_fp8_tensors(modules, recipe=recipe)
        fp8_state_saved = True

        forward_funcs = []
        for module in modules:
            wrap_autocast(module)
            forward_funcs.append(module)

        if just_one_callable:
            forward_funcs = forward_funcs[0]
        else:
            forward_funcs = tuple(forward_funcs)

        if graph_safe_rng_available():
            generators = [
                torch.cuda.default_generators[torch.cuda.current_device()],
                *get_all_rng_states().values(),
            ]
            original_rng_states = [state.get_state() for state in generators]
            rng_restore_callbacks = [
                (generator.set_state, state)
                for generator, state in zip(generators, original_rng_states)
            ]
        else:
            original_rng_state = torch.cuda.get_rng_state()
            rng_restore_callbacks = [(torch.cuda.set_rng_state, original_rng_state)]

        set_capture_start()
        capture_started = True
        graphed_callables = _make_graphed_callables(
            forward_funcs,
            sample_args,
            num_warmup_iters=num_warmup_iters,
            allow_unused_input=allow_unused_input,
            cache_quantized_params=cache_quantized_params,
            sample_kwargs=sample_kwargs,
            _order=_order,
            _num_layers_per_chunk=_num_layers_per_chunk,
            pool=pool,
            retain_graph_in_backward=retain_graph_in_backward,
            _reuse_graph_input_output_buffers=_reuse_graph_input_output_buffers,
            _graph_memory_slots=_graph_memory_slots,
            _allocator_settings_guard=allocator_settings_guard,
            pre_warmup_hook=pre_warmup_hook,
            post_warmup_hook=post_warmup_hook,
        )
    finally:
        # ExitStack runs every callback even if an earlier restoration fails.
        with contextlib.ExitStack() as capture_cleanup:
            if capture_started:
                capture_cleanup.callback(set_capture_end)
            if fp8_state_saved:
                capture_cleanup.callback(restore_fp8_tensors, modules, saved_fp8_tensors)
            for module_cls, old_call in old_call_funcs.items():
                capture_cleanup.callback(setattr, module_cls, "__call__", old_call)
            for restore_rng_state, state in rng_restore_callbacks:
                capture_cleanup.callback(restore_rng_state, state)
            capture_cleanup.callback(allocator_settings_guard.restore)

    return graphed_callables
