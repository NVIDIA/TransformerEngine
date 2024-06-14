# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import torch.distributed as dist

import transformer_engine.pytorch as te

_core_modules = [
    te.LayerNorm,
    te.RMSNorm,
    te.Linear,
    te.LayerNormLinear,
    te.LayerNormMLP,
]

_composed_modules = [
    te.MultiheadAttention,
    te.TransformerLayer,
]

batch_size = 32
seq_length = 2048
num_heads = 16
head_dim = 64
dtype = torch.bfloat16


class TestDeferredInit:

    @staticmethod
    def get_module_args(module):
        hidden_size = num_heads * head_dim
        args = (hidden_size,)
        kwargs = {"params_dtype": dtype, "device": "meta"}
        if module in [te.Linear, te.LayerNormLinear, te.LayerNormMLP]:
            ffn_hidden_size = 2 * hidden_size
            args += (ffn_hidden_size,)
            kwargs["bias"] = True
            if module == te.LayerNormMLP:
                kwargs["seq_length"] = seq_length
        elif module == te.MultiheadAttention:
            args += (num_heads,)
            kwargs["fuse_qkv_params"] = True
        elif module == te.TransformerLayer:
            args += (3 * hidden_size, num_heads)
            kwargs["fuse_qkv_params"] = True
            kwargs["seq_length"] = seq_length

        return args, kwargs

    @pytest.mark.parametrize("module_type", _core_modules + _composed_modules)
    def test_zero_memory_init(
        self,
        module_type: torch.nn.Module,
    ) -> None:
        """Test deferred initialization via device='meta'."""
        # This should not allocate any memory on CUDA device until we call reset_parameters() later.
        args, kwargs = TestDeferredInit.get_module_args(module_type)
        module = module_type(*args, **kwargs)
        assert torch.cuda.memory_allocated(device=0) == 0.0, (
            f"Initializing {module_type.__name__} with device='meta' prematurely allocated "
            "memory on CUDA device"
        )
        del module

    @pytest.mark.parametrize("module_type", _core_modules)
    def test_reset_parameters(
        self,
        module_type: torch.nn.Module,
    ) -> None:
        """Test parameter reset for core modules that have been initialized with device='meta'."""
        # Core modules own their own parameters so calling reset_parameters() here should
        # materialize them on CUDA device.
        args, kwargs = TestDeferredInit.get_module_args(module_type)
        module = module_type(*args, **kwargs)
        with torch.no_grad():
            module.reset_parameters()
        assert torch.cuda.memory_allocated(device=0) > 0.0, (
            f"{module_type.__name__}.reset_parameters() failed to materialize parameters "
            "on CUDA device"
        )
        del module
