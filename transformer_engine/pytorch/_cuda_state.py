# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Device-scoped ownership of Python caches used by CUDA operations."""

from collections import namedtuple
import functools
import inspect
import sys
import threading
import weakref

import torch

_CACHES = []
_MODULES = weakref.WeakSet()
_CacheInfo = namedtuple("CacheInfo", "hits misses maxsize currsize")


def register_module(module):
    """Track modules without extending their lifetime or allocating GPU memory."""
    _MODULES.add(module)


def device_cache(device_arg, maxsize=None):
    """An LRU per CUDA device, with explicit device-scoped cleanup.

    ``device_arg`` is the positional index of the ``device`` argument, or None
    for a function using the current device. CPU entries survive CUDA cleanup.
    Signature inspection happens only at decoration, not on the dispatch path.
    """

    def decorate(function):
        parameters = tuple(inspect.signature(function).parameters.values())
        parameter = parameters[device_arg] if device_arg is not None else None
        caches = {}
        lock = threading.RLock()

        @functools.wraps(function)
        def cached(*args, **kwargs):
            if parameter is None:
                device = None
            elif len(args) > device_arg:
                device = args[device_arg]
            else:
                device = kwargs.get(parameter.name, parameter.default)
            if device is None:
                key = torch.cuda.current_device()
            elif isinstance(device, int):
                key = device
            else:
                device = torch.device(device)
                key = (
                    (device.index if device.index is not None else torch.cuda.current_device())
                    if device.type == "cuda"
                    else (device.type, device.index)
                )
            cache = caches.get(key)
            if cache is None:
                with lock:
                    cache = caches.get(key)
                    if cache is None:
                        cache = functools.lru_cache(maxsize=maxsize)(function)
                        caches[key] = cache
            return cache(*args, **kwargs)

        def clear_device(device):
            with lock:
                cache = caches.pop(device, None)
                if cache is not None:
                    cache.cache_clear()

        def clear_all():
            with lock:
                for cache in caches.values():
                    cache.cache_clear()
                caches.clear()

        def cache_info():
            with lock:
                infos = [cache.cache_info() for cache in caches.values()]
                return _CacheInfo(
                    sum(info.hits for info in infos),
                    sum(info.misses for info in infos),
                    maxsize,
                    sum(info.currsize for info in infos),
                )

        cached.clear_device = clear_device
        cached.cache_clear = clear_all
        cached.cache_info = cache_info
        cached.cache_parameters = lambda: {"maxsize": maxsize, "typed": False}
        _CACHES.append(cached)
        return cached

    return decorate


def release_device_caches(device):
    """Release TE-owned tensor caches while their CUDA context is still usable."""
    for cache in _CACHES:
        cache.clear_device(device)
    for module in tuple(_MODULES):
        workspaces = getattr(module, "_fp8_workspaces", {})
        for name, value in tuple(workspaces.items()):
            tensors = value.get_data_tensors() if hasattr(value, "get_data_tensors") else (value,)
            if any(
                isinstance(tensor, torch.Tensor)
                and tensor.device.type == "cuda"
                and tensor.device.index == device
                for tensor in tensors
            ):
                del workspaces[name]
    # These are host caches, but their keys retain borrowed process groups.
    from .distributed import (  # pylint: disable=import-outside-toplevel
        get_distributed_rank,
        get_distributed_world_size,
    )

    get_distributed_rank.cache_clear()
    get_distributed_world_size.cache_clear()
    for name in ("utils", "context_parallel", "flex_attention", "dot_product_attention"):
        module = sys.modules.get(
            f"transformer_engine.pytorch.attention.dot_product_attention.{name}"
        )
        if module is not None:
            module._release_cuda_state(device)  # pylint: disable=protected-access


def clear_tensor_cache(cache, device):
    """Remove entries retaining tensors on the selected device, including tuples."""

    def contains(value):
        if isinstance(value, torch.Tensor):
            return value.device.type == "cuda" and value.device.index == device
        if isinstance(value, dict):
            return any(contains(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(contains(item) for item in value)
        return False

    for key, value in tuple(cache.items()):
        if contains(value):
            del cache[key]


def check_release_boundary():
    """Do not detach FP8 reduction registrations inside an active autocast."""
    from .quantization import FP8GlobalStateManager  # pylint: disable=import-outside-toplevel

    state = FP8GlobalStateManager.quantization_state
    if state.autocast_depth or state.fp8_graph_capturing:
        raise RuntimeError("Release CUDA state outside FP8 autocast and graph capture")


def release_fp8_registrations(device):
    """Remove device-scoped global references, preserving each model's numeric history.

    Buffer lists can contain registrations from several devices. Compact them
    together and repair surviving modules' indices, rather than resetting all
    quantization state or leaving holes in the reduction lists.
    """
    from .quantization import FP8GlobalStateManager  # pylint: disable=import-outside-toplevel

    manager = FP8GlobalStateManager
    state = manager.quantization_state
    index_key = manager.get_buffer_info()
    metas = {}
    for module in tuple(_MODULES):
        meta = getattr(module, "fp8_meta", None)
        if isinstance(meta, dict):
            metas[id(meta)] = meta
        for meta in (getattr(module, "_fp8_metas", None) or {}).values():
            if isinstance(meta, dict):
                metas[id(meta)] = meta
    target_metas = set()
    for identity, meta in metas.items():
        registration = meta.get(index_key, ())
        for index, key in zip(registration[::2], registration[1::2]):
            values = state.global_amax_buffer.get(key, ())
            if index < len(values) and values[index].device.index == device:
                target_metas.add(identity)
    for key, values in tuple(state.global_amax_buffer.items()):
        keep = [index for index, tensor in enumerate(values) if tensor.device.index != device]
        if len(keep) == len(values):
            continue
        remap = {old: new for new, old in enumerate(keep)}
        for identity, meta in metas.items():
            registration = meta.get(index_key)
            if registration is None:
                continue
            if identity in target_metas:
                meta.pop(index_key, None)
                meta["fp8_group"] = None
            else:
                for position in range(0, len(registration), 2):
                    if registration[position + 1] == key:
                        registration[position] = remap[registration[position]]
        for buffers in (
            state.global_amax_buffer,
            state.global_amax_history_buffer,
            state.global_scale_buffer,
        ):
            if keep:
                buffers[key] = [buffers[key][index] for index in keep]
            else:
                buffers.pop(key, None)
    recompute_key = "global_fp8_buffer_pos_fwd_recompute"
    removed = {
        meta[recompute_key]
        for identity, meta in metas.items()
        if identity in target_metas and recompute_key in meta
    }
    if removed:
        keep = [
            index
            for index in range(len(state.fp8_tensors_recompute_buffer))
            if index not in removed
        ]
        remap = {old: new for new, old in enumerate(keep)}
        state.fp8_tensors_recompute_buffer = [
            state.fp8_tensors_recompute_buffer[index] for index in keep
        ]
        for meta in metas.values():
            index = meta.get(recompute_key)
            if index in removed:
                meta.pop(recompute_key, None)
            elif index is not None:
                meta[recompute_key] = remap[index]
    used = {manager.split_key_in_buffer(key)[1] for key in state.global_amax_buffer}
    for key in tuple(state.autocast_arguments):
        if key not in used:
            del state.autocast_arguments[key]
    skip = state.skip_fp8_weight_update_tensor
    if skip is not None and skip.device.index == device:
        state.skip_fp8_weight_update_tensor = None
