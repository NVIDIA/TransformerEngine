# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import random
import contextlib
import pytest
import os
import torch
from typing import Optional, List
from transformer_engine.pytorch.cpu_offload import (
    get_cpu_offload_context,
    OffloadableLayerState,
    DefaultOffloadSynchronizer,
    start_offload,
    mark_not_offload,
)
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from utils import ModelConfig
import transformer_engine_torch as tex

# Check supported quantization schemes
fp8_available, _ = FP8GlobalStateManager.is_fp8_available()
fp8_block_scaling_available, _ = FP8GlobalStateManager.is_fp8_block_scaling_available()
mxfp8_available, _ = FP8GlobalStateManager.is_mxfp8_available()
nvfp4_available, _ = FP8GlobalStateManager.is_nvfp4_available()

quantization_recipes: List[Optional[recipe.Recipe]] = [None]
if fp8_available:
    quantization_recipes.extend((recipe.Float8CurrentScaling(), recipe.DelayedScaling()))
if fp8_block_scaling_available:
    quantization_recipes.append(recipe.Float8BlockScaling())
if mxfp8_available:
    quantization_recipes.append(recipe.MXFP8BlockScaling())
if nvfp4_available:
    quantization_recipes.append(recipe.NVFP4BlockScaling())


model_config = {
    "small": ModelConfig(8, 512, 8, 64, num_layers=5, eps=0.1),
}
SIZE = model_config["small"].hidden_size
NUM_HEADS = model_config["small"].num_heads
NUM_LAYERS = model_config["small"].num_layers
EPSILON = model_config["small"].eps

# Disable garbage collection to tests if there are reference cycles.
# We do not want them, because they can result in CUDA out of memory errors.
import gc

gc.disable()


class Utils:
    # Tensor used for simulating long-running GPU work in long_job()
    tensor1 = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    # Test tensor dimensions: _B x _S x _D = 128 x 512 x 256 = 16,777,216 elements
    # This exceeds the 256K element threshold for offloading (cpu_offload.py line 443).
    # For quantized tensors, scale_inv tensors (~524K elements for block scaling) also exceed threshold.
    _B = 128
    _S = 512
    _H = 4
    _D = 256

    @staticmethod
    def long_job(stream: Optional[torch.cuda.Stream] = None):
        NUM_ITERS = 6000
        if stream is None:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            for i in range(NUM_ITERS):
                Utils.tensor1.normal_()

    @staticmethod
    def measure_time(func):
        import time

        torch.cuda.synchronize()
        start = time.time()
        func()
        torch.cuda.synchronize()
        end = time.time()
        return (end - start) * 1000

    @staticmethod
    def get_cuda_memory_mb():
        return torch.cuda.memory_allocated() / (1024**2)

    @staticmethod
    def get_max_cuda_memory_mb():
        return torch.cuda.max_memory_allocated() / (1024**2)

    @staticmethod
    def get_cpu_memory_mb() -> float:
        import psutil, os

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

    @staticmethod
    def get_layer_names():
        return [
            "linear",
            "layernorm_linear",
            "layernorm_mlp",
            "grouped_linear",
            "multihead_attention",
            "transformer_layer",
            "linear_op",
            "layernorm_mlp_ops",
        ]

    @staticmethod
    def create_layer(layer_type: str):
        if layer_type == "linear":
            return te.Linear(Utils._D, Utils._D, params_dtype=torch.bfloat16)
        elif layer_type == "layernorm_linear":
            return te.LayerNormLinear(Utils._D, Utils._D, params_dtype=torch.bfloat16)
        elif layer_type == "layernorm_mlp":
            return te.LayerNormMLP(Utils._D, Utils._D, params_dtype=torch.bfloat16)
        elif layer_type == "multihead_attention":
            return te.MultiheadAttention(
                Utils._D, Utils._H, attention_dropout=0.0, params_dtype=torch.bfloat16
            )
        elif layer_type == "grouped_linear":
            return te.GroupedLinear(Utils._H, Utils._D, Utils._D, params_dtype=torch.bfloat16)
        elif layer_type == "transformer_layer":
            return te.TransformerLayer(
                Utils._D,
                Utils._D,
                Utils._H,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                params_dtype=torch.bfloat16,
            )
        elif layer_type == "linear_op":
            return te.ops.Linear(Utils._D, Utils._D, dtype=torch.bfloat16)
        elif layer_type == "layernorm_mlp_ops":
            return te.ops.Sequential(
                te.ops.LayerNorm(Utils._D, dtype=torch.bfloat16),
                te.ops.Linear(Utils._D, Utils._D, dtype=torch.bfloat16),
                te.ops.GELU(),
                te.ops.Linear(Utils._D, Utils._D, dtype=torch.bfloat16),
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    @staticmethod
    def create_tensor(recipe: Optional[recipe.Recipe], requires_grad: bool = False) -> torch.Tensor:
        shape = (Utils._B, Utils._S, Utils._D)
        tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        if recipe is None:
            tensor = tensor.requires_grad_() if requires_grad else tensor
            return tensor
        elif recipe.delayed():
            quantizer = te.tensor.float8_tensor.Float8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                scale=torch.tensor([1.0], device="cuda"),
                amax=torch.tensor([1.0], device="cuda"),
            )
            return quantizer(tensor)
        elif recipe.float8_current_scaling():
            quantizer = te.tensor.float8_tensor.Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, device="cuda"
            )
            return quantizer(tensor)
        elif recipe.float8_block_scaling():
            quantizer = te.tensor.float8_blockwise_tensor.Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True
            )
            return quantizer(tensor)
        elif recipe.mxfp8():
            quantizer = te.tensor.mxfp8_tensor.MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
            return quantizer(tensor)
        elif recipe.nvfp4():
            quantizer = te.tensor.nvfp4_tensor.NVFP4Quantizer()
            return quantizer(tensor)

    @staticmethod
    def create_recipe_ctx(recipe: Optional[recipe.Recipe]):
        if recipe is None:
            return lambda: contextlib.nullcontext()
        else:
            return lambda: te.fp8_autocast(fp8_recipe=recipe)

    @staticmethod
    def get_tensor_size_mb(tensor):
        if tensor is None:
            return 0
        if isinstance(tensor, te.quantized_tensor.QuantizedTensorStorage):
            return sum(Utils.get_tensor_size_mb(t) for t in tensor.get_data_tensors())
        else:
            return tensor.numel() * tensor.element_size() / (1024**2)

    @staticmethod
    def memory_leak_check():
        # Should be called before each test.
        # Only cublas workspaces and some global tensors are allowed to be allocated.
        # All other allocations should be released.
        # This is a simple check to catch memory leaks.
        if Utils.get_cuda_memory_mb() > 1000:
            memory_num = Utils.get_cuda_memory_mb()
            import gc

            gc.collect()  # We want next test to be run with clean state.
            gc.disable()
            raise RuntimeError(f"Memory leak: {memory_num} MB")


class TestsOffloadableLayerState:
    @pytest.mark.parametrize("random_num_tensors", [True, False])
    @pytest.mark.parametrize("recipe", quantization_recipes)
    def test_general(self, random_num_tensors, recipe):
        """
        Test general functionality of DefaultOffloadSynchronizer - offload NUM_LAYERS-1 out of NUM_LAYERS layers,
        for each layer offload random number of random tensors.
        Then do backward pass for each layer, and check if reloaded tensors are equal to original tensors.
        """
        Utils.memory_leak_check()
        NUM_ITERATIONS = 10

        stream = torch.cuda.Stream()

        offload_layer_state = OffloadableLayerState(
            offload_stream=stream,
        )

        for _ in range(NUM_ITERATIONS):
            original_tensors = []
            tensors_ids = []
            NUM_TENSORS = random.choice([1, 20]) if random_num_tensors else 1
            for _ in range(NUM_TENSORS):
                tensor = Utils.create_tensor(recipe)
                original_tensors.append(tensor)
                tensor_id = offload_layer_state.push_tensor(tensor)
                assert tensor.device.type == "cuda"
                tensors_ids.append(tensor_id)

            offload_layer_state.start_offload()
            offload_layer_state.release_activation_forward_gpu_memory()
            offload_layer_state.start_reload()

            for j in range(len(tensors_ids)):
                tensor_gpu = offload_layer_state.pop_tensor(tensors_ids[j])
                assert tensor_gpu.device.type == "cuda"
                assert tensor_gpu.shape == original_tensors[j].shape
                assert tensor_gpu.dtype == original_tensors[j].dtype
                torch.testing.assert_close(tensor_gpu, original_tensors[j])
            offload_layer_state.release_all_memory()
        torch.cuda.synchronize()

    def test_offload_base_tensor(self):
        Utils.memory_leak_check()
        stream = torch.cuda.Stream()
        offload_layer_state = OffloadableLayerState(
            offload_stream=stream,
        )
        init_cuda_memory = Utils.get_cuda_memory_mb()
        x = Utils.create_tensor(None)
        x_size = Utils.get_tensor_size_mb(x)
        x_1 = x[::2]
        x_2 = x[1::2]

        start_offload(x_1, offload_base_tensor=True)
        start_offload(x_2, offload_base_tensor=True)
        x1_id = offload_layer_state.push_tensor(x_1)
        x2_id = offload_layer_state.push_tensor(x_2)
        del x_1, x_2
        offload_layer_state.start_offload()
        offload_layer_state.release_activation_forward_gpu_memory()

        assert offload_layer_state.get_offloaded_total_size_mb() == pytest.approx(x_size, 0.1)

        offload_layer_state.start_reload()
        x_1 = offload_layer_state.pop_tensor(x1_id)
        x_2 = offload_layer_state.pop_tensor(x2_id)
        assert x_1.device.type == "cuda"
        assert x_2.device.type == "cuda"

        assert torch.allclose(x_1, x[::2])
        assert torch.allclose(x_2, x[1::2])
        del x

        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory + x_size, 0.1)


class TestsDefaultOffloadSynchronizer:
    @pytest.mark.parametrize("random_num_tensors", [True, False])
    @pytest.mark.parametrize("recipe", quantization_recipes)
    def test_general(self, random_num_tensors, recipe):
        """
        Test general functionality of DefaultOffloadSynchronizer - offload NUM_LAYERS-1 out of NUM_LAYERS layers,
        for each layer offload random number of random tensors.
        Then do backward pass for each layer, and check if reloaded tensors are equal to original tensors.
        """
        Utils.memory_leak_check()
        NUM_LAYERS = 10
        NUM_ITERATIONS = 10

        offload_synchronizer = DefaultOffloadSynchronizer(
            num_layers=NUM_LAYERS,
            num_offloaded_layers=NUM_LAYERS - 1,
        )

        for _ in range(NUM_ITERATIONS):
            original_tensors = []
            tensors_ids = []
            layer_ids = []

            for i in range(NUM_LAYERS):
                NUM_LAYER_TENSORS = random.randint(1, 10) if random_num_tensors else 1
                layer_tensors = []
                layer_tensors_ids = []
                layer_id = offload_synchronizer.fwd_step()
                for _ in range(NUM_LAYER_TENSORS):
                    tensor = Utils.create_tensor(recipe)
                    layer_tensors.append(tensor)
                    tensor_id = offload_synchronizer.push_tensor(tensor)
                    assert tensor.device.type == "cuda"
                    layer_tensors_ids.append(tensor_id)
                layer_ids.append(layer_id)
                tensors_ids.append(layer_tensors_ids)
                original_tensors.append(layer_tensors)
            for i in range(NUM_LAYERS - 1, -1, -1):
                offload_synchronizer.bwd_step(layer_ids[i])
                for j in range(len(tensors_ids[i])):
                    tensor_gpu = offload_synchronizer.pop_tensor(tensors_ids[i][j])
                    assert tensor_gpu.device.type == "cuda"
                    assert tensor_gpu.shape == original_tensors[i][j].shape
                    assert tensor_gpu.dtype == original_tensors[i][j].dtype
                    torch.testing.assert_close(tensor_gpu, original_tensors[i][j])
            offload_synchronizer.finish_part_of_bwd()
        torch.cuda.synchronize()

    @pytest.mark.parametrize("recipe", quantization_recipes)
    def test_memory(self, recipe):
        torch.cuda.synchronize()
        Utils.memory_leak_check()
        NUM_LAYERS = 10

        torch.cuda.reset_peak_memory_stats()

        offload_synchronizer = DefaultOffloadSynchronizer(
            num_layers=NUM_LAYERS,
            num_offloaded_layers=NUM_LAYERS - 1,
        )

        init_cuda_memory = Utils.get_cuda_memory_mb()

        tensor_ids = []

        torch.cuda.synchronize()
        for _ in range(NUM_LAYERS):
            offload_synchronizer.fwd_step()
            tensor = Utils.create_tensor(recipe)
            tensor_size = Utils.get_tensor_size_mb(tensor)
            tensor_id = offload_synchronizer.push_tensor(tensor)
            assert tensor.device.type == "cuda"
            tensor_ids.append(tensor_id)
            del tensor, tensor_id
        torch.cuda.synchronize()

        if recipe is None:
            assert Utils.get_max_cuda_memory_mb() == pytest.approx(
                init_cuda_memory + tensor_size, 0.1
            )
        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory + tensor_size, 0.1)

        for i in range(NUM_LAYERS - 1, -1, -1):
            offload_synchronizer.bwd_step(i)
            tensor_gpu = offload_synchronizer.pop_tensor(tensor_ids[i])
            assert tensor_gpu.device.type == "cuda"
            del tensor_gpu, tensor_ids[i]
        offload_synchronizer.finish_part_of_bwd()

        del tensor_ids
        torch.cuda.synchronize()

        if recipe is None:
            assert Utils.get_max_cuda_memory_mb() == pytest.approx(
                init_cuda_memory + tensor_size, 0.1
            )
        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory, 0.1)

    @pytest.mark.parametrize("recipe", quantization_recipes)
    def test_multiple_tensor_offload(self, recipe):
        Utils.memory_leak_check()
        init_cpu_memory = Utils.get_cpu_memory_mb()
        init_cuda_memory = Utils.get_cuda_memory_mb()
        offload_synchronizer = DefaultOffloadSynchronizer(
            num_layers=2,
            num_offloaded_layers=1,
        )
        x1 = Utils.create_tensor(recipe)
        x_size = Utils.get_tensor_size_mb(x1)
        offload_synchronizer.fwd_step()
        offload_synchronizer.push_tensor(x1)
        offload_synchronizer.push_tensor(x1)
        offload_synchronizer.push_tensor(x1)
        # Verify x1 is not corrupted after pushing (important for QuantizedTensor)
        if recipe is not None:
            x1.dequantize()  # Should not raise - tensor should still be valid
        offload_synchronizer.fwd_step()
        # Only one copy of tensor on cpu is allocated.
        assert Utils.get_cpu_memory_mb() == pytest.approx(init_cpu_memory + 1 * x_size, 0.1)
        del x1
        offload_synchronizer.bwd_step(1)
        offload_synchronizer.bwd_step(0)
        offload_synchronizer.finish_part_of_bwd()

        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory, 0.1)


class TestTELayers:
    @pytest.mark.parametrize("layer_type", Utils.get_layer_names())
    @pytest.mark.parametrize("recipe", quantization_recipes)
    def test_sanity(self, layer_type, recipe):
        Utils.memory_leak_check()

        # Skip ops-based layers with Float8BlockScaling recipe
        if (
            layer_type in ["linear_op", "layernorm_mlp_ops"]
            and recipe is not None
            and recipe.float8_block_scaling()
        ):
            pytest.skip("Fusible operations do not support FP8 block scaling recipe")

        recipe_ctx = Utils.create_recipe_ctx(recipe)
        init_cuda_memory = Utils.get_cuda_memory_mb()
        OFFLOAD_LAYERS = 6
        NUM_LAYERS = 10
        offload_ctx, sync_function = get_cpu_offload_context(
            enabled=True,
            num_layers=OFFLOAD_LAYERS,
            model_layers=NUM_LAYERS,
        )
        layers = [Utils.create_layer(layer_type) for _ in range(NUM_LAYERS)]
        inp = Utils.create_tensor(None)
        m_splits = (
            {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H}
            if layer_type == "grouped_linear"
            else {}
        )
        out = inp
        for i in range(NUM_LAYERS):
            with offload_ctx, recipe_ctx():
                # Ops-based layers don't support is_first_microbatch parameter
                if layer_type in ["linear_op", "layernorm_mlp_ops"]:
                    out = layers[i](out, **m_splits)
                else:
                    out = layers[i](out, is_first_microbatch=False, **m_splits)
            out = sync_function(out)
        out.sum().backward()
        torch.cuda.synchronize()
        del out, inp, layers

    @pytest.mark.parametrize("layer_type", Utils.get_layer_names())
    @pytest.mark.parametrize("recipe", quantization_recipes)
    def test_memory(self, layer_type, recipe):
        Utils.memory_leak_check()

        # Skip ops-based layers with Float8BlockScaling recipe
        if (
            layer_type in ["linear_op", "layernorm_mlp_ops"]
            and recipe is not None
            and recipe.float8_block_scaling()
        ):
            pytest.skip("Fusible operations do not support FP8 block scaling recipe")

        offload_ctx, sync_function = get_cpu_offload_context(
            enabled=True,
            num_layers=1,
            model_layers=2,
            offload_activations=True,
            offload_weights=False,
        )
        recipe_ctx = Utils.create_recipe_ctx(recipe)
        layer = Utils.create_layer(layer_type)
        inp = Utils.create_tensor(None)

        m_splits = (
            {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H}
            if layer_type == "grouped_linear"
            else {}
        )

        # Ops-based layers don't support is_first_microbatch parameter
        is_ops_layer = layer_type in ["linear_op", "layernorm_mlp_ops"]

        with recipe_ctx():
            if is_ops_layer:
                out = layer(inp, **m_splits)
            else:
                out = layer(inp, is_first_microbatch=True, **m_splits)
        out.sum().backward()

        del inp
        init_cuda_memory = Utils.get_cuda_memory_mb()

        # run layer without offload
        inp = Utils.create_tensor(None)
        with recipe_ctx():
            if is_ops_layer:
                out = layer(inp, **m_splits)
            else:
                out = layer(inp, is_first_microbatch=False, **m_splits)
        with recipe_ctx():
            out = out + 1
        del inp
        cuda_memory_no_offload = Utils.get_cuda_memory_mb()

        out.sum().backward()
        # run layer with offload
        inp = Utils.create_tensor(None)
        with offload_ctx, recipe_ctx():
            if is_ops_layer:
                out = layer(inp, **m_splits)
            else:
                out = layer(inp, is_first_microbatch=False, **m_splits)
        out = sync_function(out)
        with offload_ctx, recipe_ctx():
            out = out + 1
        out = sync_function(out)
        del inp
        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory, 0.1)
        offloaded_memory_cpu = offload_ctx.offload_synchronizer.get_offloaded_total_size_mb()

        # This assertion verifies that the memory used by tensors on the CPU matches the memory saved from a layer.
        # It helps catch cases where an offloaded tensor still has a live pointer, which would
        # cause an unnecessary copy to the CPU and prevent GPU memory from being released.
        assert Utils.get_cuda_memory_mb() + offloaded_memory_cpu == pytest.approx(
            cuda_memory_no_offload, 0.1
        )
        out.sum().backward()

    @pytest.mark.parametrize("layer_type", Utils.get_layer_names())
    @pytest.mark.parametrize("recipe", quantization_recipes)
    def test_manual_synchronization(self, recipe, layer_type):
        Utils.memory_leak_check()

        # Skip ops-based layers with Float8BlockScaling recipe
        if (
            layer_type in ["linear_op", "layernorm_mlp_ops"]
            and recipe is not None
            and recipe.float8_block_scaling()
        ):
            pytest.skip("Fusible operations do not support FP8 block scaling recipe")

        offload_ctx, sync_function, manual_controller = get_cpu_offload_context(
            enabled=True,
            model_layers=6,
            offload_activations=True,
            manual_synchronization=True,
        )
        layer_1 = Utils.create_layer(layer_type)
        layer_2 = Utils.create_layer(layer_type)
        inp1 = Utils.create_tensor(None)
        inp2 = Utils.create_tensor(None)

        recipe_ctx = Utils.create_recipe_ctx(recipe)

        m_splits = (
            {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H}
            if layer_type == "grouped_linear"
            else {}
        )

        init_cuda_memory = Utils.get_cuda_memory_mb()

        # 1 fwd
        with offload_ctx, recipe_ctx():
            out_1 = layer_1(inp1, **m_splits)
        out_1 = sync_function(out_1)

        with offload_ctx, recipe_ctx():
            out_2 = layer_2(inp2, **m_splits)
        out_2 = sync_function(out_2)

        mark_not_offload(out_1, out_2)

        del inp1, inp2

        memory_before_offload = Utils.get_cuda_memory_mb()
        manual_controller.start_offload_layer(0)
        manual_controller.release_activation_forward_gpu_memory(0)
        manual_controller.start_offload_layer(1)
        manual_controller.release_activation_forward_gpu_memory(1)
        memory_after_offload = Utils.get_cuda_memory_mb()
        assert memory_after_offload + EPSILON < memory_before_offload

        manual_controller.start_reload_layer(0)
        manual_controller.start_reload_layer(1)

        memory_after_reload = Utils.get_cuda_memory_mb()
        assert memory_after_reload == pytest.approx(memory_before_offload, 0.1)

        out_1.sum().backward()
        out_2.sum().backward()

    @pytest.mark.parametrize("recipe", quantization_recipes)
    @pytest.mark.parametrize("layer_type", Utils.get_layer_names())
    @pytest.mark.parametrize("use_cuda_graphs", [True, False])
    @pytest.mark.parametrize("retain_pinned_cpu_buffers", [True, False])
    @pytest.mark.parametrize("backend", ["FlashAttention", "FusedAttention", "UnfusedAttention"])
    def test_numerics(
        self,
        recipe,
        layer_type,
        use_cuda_graphs,
        backend,
        retain_pinned_cpu_buffers,
    ):
        # Skip ops-based layers with Float8BlockScaling recipe
        if (
            layer_type in ["linear_op", "layernorm_mlp_ops"]
            and recipe is not None
            and recipe.float8_block_scaling()
        ):
            pytest.skip("Fusible operations do not support FP8 block scaling recipe")

        recipe_ctx = Utils.create_recipe_ctx(recipe)

        if use_cuda_graphs and not retain_pinned_cpu_buffers:
            pytest.skip(
                "Cuda graphs are not yet supported with cpu offloading when"
                " retain_pinned_cpu_buffers is False."
            )

        if backend == "FusedAttention" and use_cuda_graphs:
            pytest.skip(
                "Fused attention + cuda graphs is temporarily broken, not because of cpu offloading"
            )

        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "0"
        os.environ["NVTE_UNFUSED_ATTN"] = "0"

        if backend == "FlashAttention":
            os.environ["NVTE_FLASH_ATTN"] = "1"
        elif backend == "FusedAttention":
            os.environ["NVTE_FUSED_ATTN"] = "1"
        elif backend == "UnfusedAttention":
            os.environ["NVTE_UNFUSED_ATTN"] = "1"

        offload_ctx, sync_function = get_cpu_offload_context(
            enabled=True,
            num_layers=1,
            model_layers=2,
            offload_activations=True,
            offload_weights=False,
            retain_pinned_cpu_buffers=retain_pinned_cpu_buffers,
        )

        class Callable(torch.nn.Module):
            def __init__(self, offload_ctx=None, sync_function=None):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [Utils.create_layer(layer_type) for _ in range(2)]
                )
                self.offload_ctx = offload_ctx
                self.sync_function = sync_function

            def forward(self, x):
                m_splits = (
                    {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H}
                    if layer_type == "grouped_linear"
                    else {}
                )
                is_ops_layer = layer_type in ["linear_op", "layernorm_mlp_ops"]
                for layer in self.layers:
                    with self.offload_ctx, recipe_ctx():
                        if is_ops_layer:
                            x = layer(x, **m_splits)
                        else:
                            x = layer(x, is_first_microbatch=False, **m_splits)
                    if self.sync_function is not None:
                        x = self.sync_function(x)
                return x

        callable_offload = Callable(offload_ctx=offload_ctx, sync_function=sync_function)
        callable_no_offload = Callable(offload_ctx=contextlib.nullcontext(), sync_function=None)

        # copy parameters
        for param_offload, param_no_offload in zip(
            callable_offload.parameters(), callable_no_offload.parameters()
        ):
            param_offload.data.copy_(param_no_offload.data)

        x = Utils.create_tensor(None)

        if use_cuda_graphs:
            callable_offload = te.make_graphed_callables(
                callable_offload,
                (x,),
                enabled=recipe is not None,
                recipe=(Utils.create_recipe_ctx(recipe) if recipe is not None else None),
            )

        # warm up (for example to compute sf for delayed scaling)
        for _ in range(4):
            out = callable_offload(x)
            out.sum().backward()
            out = callable_no_offload(x)
            out.sum().backward()

        callable_offload.zero_grad(set_to_none=True)
        out_offload = callable_offload(x)
        out_offload.sum().backward()

        # save out and gradients
        offload_outs = [out_offload]
        for param in callable_offload.parameters():
            offload_outs.append(param.detach().clone())

        torch.cuda.reset_peak_memory_stats()
        out_no_offload = callable_no_offload(x)
        out_no_offload.sum().backward()

        # collect gradients
        no_offload_outs = [out_no_offload]
        for param in callable_no_offload.parameters():
            no_offload_outs.append(param.detach().clone())

        # check if tensors are the same
        for i in range(len(offload_outs)):
            assert torch.allclose(offload_outs[i], no_offload_outs[i]), f"Error in tensor {i}."

        torch.cuda.synchronize()

    def test_example_from_doc(self):
        offload_stream = torch.cuda.Stream()
        num_layers = 10
        layers = [Utils.create_layer("transformer_layer") for _ in range(num_layers)]
        inp = [Utils.create_tensor(None) for _ in range(num_layers)]
        out = [None] * num_layers
        cpu_offload_context, sync_function, manual_controller = get_cpu_offload_context(
            enabled=True,
            model_layers=num_layers,
            manual_synchronization=True,
            offload_stream=offload_stream,
        )

        for i in range(num_layers):
            with cpu_offload_context:
                out[i] = layers[i].forward(inp[i])
            out[i] = sync_function(out[i])
            manual_controller.start_offload_layer(i)

        offload_stream.synchronize()
        for i in range(num_layers):
            manual_controller.release_activation_forward_gpu_memory(i)

        for i in range(num_layers - 1, -1, -1):
            # these calls are intended to be done in the backward pass
            manual_controller.start_reload_layer(i)

        offload_stream.synchronize()
        for i in range(num_layers):
            out[i].sum().backward()
