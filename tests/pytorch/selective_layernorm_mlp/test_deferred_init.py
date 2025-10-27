
import pytest
import torch
import torch.distributed as dist
import gc
import transformer_engine.pytorch as te

_core_modules = [te.SelectiveLayerNormMLP]
_composed_modules = []

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
        ffn_hidden_size = 2 * hidden_size
        args += (ffn_hidden_size,)
        kwargs["bias"] = True
        if module == te.LayerNormMLP:
            kwargs["seq_length"] = seq_length
        return args, kwargs

    @pytest.mark.parametrize("module_type", _core_modules + _composed_modules)
    def test_zero_memory_init(
        self,
        module_type: torch.nn.Module,
    ) -> None:
        """Test deferred initialization via device='meta'. 
        rewrote this so that it tests difference in mem, not zero mem, since this might not be the first test"""
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(0)

        args, kwargs = TestDeferredInit.get_module_args(module_type)
        module = module_type(*args, **kwargs)  # device='meta'

        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated(0)
        assert after == before, (
            f"{module_type.__name__} init changed allocated CUDA memory: "
            f"before={before}, after={after}"
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
