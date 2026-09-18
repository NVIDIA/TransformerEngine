# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Top level package"""

# pylint: disable=unused-import

import os

import sys
import torch
import torch.utils
import torch.utils.data
import torch_musa


def patch_before_import_te():
    from .pytorch import attention
    from .pytorch import tensor
    from .pytorch import fp8
    from .pytorch import distributed
    from .pytorch.module import base
    from .pytorch.ops import op

    # from .pytorch.cpp_extensions import cast
    from .pytorch.module import linear
    from .pytorch.module import grouped_linear
    from .pytorch import utils


def patch_after_import_torch():
    def hook_cuda_device(device):
        if isinstance(device, str) and device.startswith("cuda"):
            return device.replace("cuda", "musa")
        if isinstance(device, torch.device) and device.type == "cuda":
            return torch.device("musa", device.index)
        return device

    def maybe_hook_cuda_args(args, kwargs):
        new_args = []
        for arg in args:
            new_args.append(hook_cuda_device(arg))
        if "device" in kwargs:
            v = kwargs["device"]
            kwargs["device"] = hook_cuda_device(v)
        return tuple(new_args), kwargs

    torch.cuda.is_available = torch.musa.is_available
    torch.cuda.current_device = torch.musa.current_device
    torch.cuda.device_count = torch.musa.device_count
    torch.cuda.set_device = torch.musa.set_device
    torch.cuda.DoubleTensor = torch.musa.DoubleTensor
    torch.cuda.FloatTensor = torch.musa.FloatTensor
    torch.cuda.LongTensor = torch.musa.LongTensor
    torch.cuda.HalfTensor = torch.musa.HalfTensor
    torch.cuda.BFloat16Tensor = torch.musa.BFloat16Tensor
    torch.cuda.IntTensor = torch.musa.IntTensor
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.get_rng_state = torch.musa.get_rng_state
    torch.cuda.set_rng_state = torch.musa.set_rng_state
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.empty_cache = torch.musa.empty_cache
    torch.Tensor.cuda = torch.Tensor.musa
    torch.cuda.manual_seed = torch.musa.manual_seed
    torch.cuda.Event = torch.musa.Event
    torch.cuda.Stream = torch.musa.Stream
    torch.cuda.current_stream = torch.musa.current_stream
    torch.cuda.set_stream = torch.musa.set_stream
    torch.cuda.get_device_properties = torch.musa.get_device_properties
    # add torch.musa.current_devce() to activate torch.musa.default_generators
    d = torch.musa.current_device()
    torch.cuda.default_generators = torch.musa.default_generators

    torch.cuda.memory_allocated = torch.musa.memory_allocated
    torch.cuda.max_memory_allocated = torch.musa.max_memory_allocated
    torch.cuda.memory_reserved = torch.musa.memory_reserved
    torch.cuda.max_memory_reserved = torch.musa.max_memory_reserved

    # (yehua.zhang) replace lazy_call to avoid cpu memory leak,
    # because failure of cuda init in lazy_call will cause endless operation of emplace back.
    torch.cuda._lazy_call = torch.musa.core._lazy_init._lazy_call
    torch.cuda._lazy_init = torch.musa.core._lazy_init._lazy_init

    original_tensor = torch.tensor

    def patched_tensor(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_tensor(*args, **kwargs)
        return result

    torch.tensor = patched_tensor

    orig_type = torch.Tensor.type

    def musa_type(*args, **kwargs):
        result = orig_type(*args, **kwargs)
        if isinstance(result, str):
            result = result.replace("musa", "cuda")
        return result

    torch.Tensor.type = musa_type

    original_zeros = torch.zeros

    def patched_zeros(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_zeros(*args, **kwargs)
        return result

    torch.zeros = patched_zeros

    original_ones = torch.ones

    def patched_ones(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_ones(*args, **kwargs)
        return result

    torch.ones = patched_ones

    original_empty = torch.empty

    def patched_empty(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_empty(*args, **kwargs)
        return result

    torch.empty = patched_empty

    original_rand = torch.rand

    def patched_rand(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_rand(*args, **kwargs)
        return result

    torch.rand = patched_rand

    original_arange = torch.arange

    def patched_arange(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_arange(*args, **kwargs)
        return result

    torch.arange = patched_arange

    original_empty_like = torch.empty_like

    def patched_empty_like(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_empty_like(*args, **kwargs)
        return result

    torch.empty_like = patched_empty_like

    original_is_cuda = torch.Tensor.is_cuda

    def always_cuda(self):
        return self.is_musa

    torch.Tensor.is_cuda = property(always_cuda)

    origin_init_process_group = torch.distributed.init_process_group

    def patched_init_process_group(*args, **kwargs):
        if "backend" in kwargs and kwargs["backend"] == "nccl":
            kwargs["backend"] = "mccl"
        result = origin_init_process_group(*args, **kwargs)
        return result

    torch.distributed.init_process_group = patched_init_process_group

    # def pin_memory(data, device=None):
    #     return data
    # torch.utils.data._utils.pin_memory.pin_memory = pin_memory

    def _pass_pvtx(*args, **kwargs):
        return

    torch.cuda.nvtx.range_push = _pass_pvtx
    torch.cuda.nvtx.range_pop = _pass_pvtx

    origin_module_to = torch.nn.Module.to

    def patched_module_to(self, *args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        return origin_module_to(self, *args, **kwargs)

    torch.nn.Module.to = patched_module_to

    origin_tensor_to = torch.Tensor.to

    def patched_tensor_to(self, *args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        return origin_tensor_to(self, *args, **kwargs)

    torch.Tensor.to = patched_tensor_to

    def get_default_device():
        device = torch.device("musa", torch.musa.current_device())
        return device

    torch.get_default_device = get_default_device

    def is_autocast_enabled(device_type=None):
        return False

    torch.is_autocast_enabled = is_autocast_enabled

    import os

    # HACK(sherry): enable torch.compile
    os.environ["NVTE_TORCH_COMPILE"] = "0"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    # HACK(sherry)


def py_patch():
    if sys.version_info >= (3.9, 0):
        return
    import math

    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)

    math.lcm = lcm
    return


py_patch()
patch_before_import_te()
patch_after_import_torch()


try:
    from . import pytorch
except ImportError:
    pass
except FileNotFoundError as e:
    if "Could not find shared object file" not in str(e):
        raise e  # Unexpected error
    else:
        if os.getenv("NVTE_FRAMEWORK"):
            frameworks = os.getenv("NVTE_FRAMEWORK").split(",")
            if "pytorch" in frameworks or "all" in frameworks:
                raise e
        else:
            # If we got here, we could import `torch` but could not load the framework extension.
            # This can happen when a user wants to work only with `transformer_engine.jax` on a system that
            # also has a PyTorch installation. In order to enable that use case, we issue a warning here
            # about the missing PyTorch extension in case the user hasn't set NVTE_FRAMEWORK.
            import warnings

            warnings.warn(
                "Detected a PyTorch installation but could not find the shared object file for the "
                "Transformer Engine PyTorch extension library. If this is not intentional, please "
                "reinstall Transformer Engine with `pip install transformer_engine[pytorch]` or "
                "build from source with `NVTE_FRAMEWORK=pytorch`.",
                category=RuntimeWarning,
            )

try:
    from . import jax
except ImportError:
    pass
except FileNotFoundError as e:
    if "Could not find shared object file" not in str(e):
        raise e  # Unexpected error
    else:
        if os.getenv("NVTE_FRAMEWORK"):
            frameworks = os.getenv("NVTE_FRAMEWORK").split(",")
            if "jax" in frameworks or "all" in frameworks:
                raise e
        else:
            # If we got here, we could import `jax` but could not load the framework extension.
            # This can happen when a user wants to work only with `transformer_engine.pytorch` on a system
            # that also has a Jax installation. In order to enable that use case, we issue a warning here
            # about the missing Jax extension in case the user hasn't set NVTE_FRAMEWORK.
            import warnings

            warnings.warn(
                "Detected a Jax installation but could not find the shared object file for the "
                "Transformer Engine Jax extension library. If this is not intentional, please "
                "reinstall Transformer Engine with `pip install transformer_engine[jax]` or "
                "build from source with `NVTE_FRAMEWORK=jax`.",
                category=RuntimeWarning,
            )

from importlib import metadata
import transformer_engine.common

__version__ = str(metadata.version("transformer_engine"))
