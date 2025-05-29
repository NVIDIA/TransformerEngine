# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import random
import contextlib
import pytest
import torch
from typing import Optional
from transformer_engine.pytorch.cpu_offload import  _CPUOffloadBackend
from transformer_engine.pytorch.cpu_offload import CPUOffload, get_cpu_offload_context
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_torch as tex
EPSILON = 0.1

# Disable garbage collection to tests if there are reference cycles.
# We do not want them, because they can result in CUDA out of memory errors.
import gc 
gc.disable()

class Utils:
    tensor1 = torch.randn((1000, 1000), device="cuda")
    _B = 16
    _S = 256
    _H = 4
    _D = 1024

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
        start = time.time()
        func()
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
        return ["linear", "layernorm_linear", "layernorm_mlp", "grouped_linear", "multihead_attention", "transformer_layer"]

    @staticmethod
    def create_layer(layer_type: str):
        if layer_type == "linear":
            return te.Linear(Utils._D, Utils._D)
        elif layer_type == "layernorm_linear":
            return te.LayerNormLinear(Utils._D, Utils._D)
        elif layer_type == "layernorm_mlp":
            return te.LayerNormMLP(Utils._D, Utils._D)
        elif layer_type == "multihead_attention":
            return te.MultiheadAttention(Utils._D, Utils._H, attention_dropout=0.0)
        elif layer_type == "grouped_linear":
            return te.GroupedLinear(Utils._H, Utils._D, Utils._D)
        elif layer_type == "transformer_layer":
            return te.TransformerLayer(Utils._D, Utils._D, Utils._H, attention_dropout=0.0, hidden_dropout=0.0)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    @staticmethod
    def get_recipe_names():
        return ["high precision", "fp8_delayed_scaling", "fp8_current_scaling", "fp8_block_scaling", "mxfp8"]

    @staticmethod
    def create_tensor(recipe_name: str, requires_grad: bool = False):
        shape = (Utils._B, Utils._S, Utils._D)
        tensor = torch.randn(shape, device="cuda")
        if recipe_name == "high precision":
            tensor = tensor.requires_grad_() if requires_grad else tensor
            return tensor
        elif recipe_name == "fp8_delayed_scaling":
            quantizer = te.tensor.float8_tensor.Float8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, 
                scale=torch.tensor([1.0], device="cuda"), 
                amax=torch.tensor([1.], device="cuda")
            )
            return quantizer(tensor)
        elif recipe_name == "fp8_current_scaling":
            quantizer = te.tensor.float8_tensor.Float8CurrentScalingQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, device="cuda")
            return quantizer(tensor)
        elif recipe_name == "fp8_block_scaling":
            quantizer = te.tensor.float8_blockwise_tensor.Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
            return quantizer(tensor)
        elif recipe_name == "mxfp8":
            quantizer = te.tensor.mxfp8_tensor.MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3)
            return quantizer(tensor)
    
    @staticmethod
    def skip_if_recipe_not_supported(recipe_name: str):
        fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
        mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
        fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
            FP8GlobalStateManager.is_fp8_block_scaling_available()
        )
        if recipe_name == "fp8_delayed_scaling" and not fp8_available:
            pytest.skip(reason_for_no_fp8)
        elif recipe_name == "fp8_current_scaling" and not fp8_available:
            pytest.skip(reason_for_no_fp8)
        elif recipe_name == "fp8_block_scaling" and not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
        elif recipe_name == "mxfp8" and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)


    @staticmethod
    def create_recipe_ctx(recipe_name: str):
        if recipe_name == "high precision":
            return lambda: contextlib.nullcontext()
        elif recipe_name == "fp8_delayed_scaling":
            return lambda: te.fp8_autocast(fp8_recipe=recipe.DelayedScaling())
        elif recipe_name == "fp8_current_scaling":
            return lambda: te.fp8_autocast(fp8_recipe=recipe.Float8CurrentScaling())
        elif recipe_name == "fp8_block_scaling":
            return lambda: te.fp8_autocast(fp8_recipe=recipe.Float8BlockScaling())
        elif recipe_name == "mxfp8":
            return lambda: te.fp8_autocast(fp8_recipe=recipe.MXFP8BlockScaling())
    
    @staticmethod
    def get_tensor_size_mb(tensor):
        if type(tensor) == torch.Tensor:
            return tensor.numel() * tensor.element_size() / (1024**2)
        else:
            # 1 byte for rowwise, 1 byte for columnwise
            return tensor.numel() * 2 / (1024**2)
    
    @staticmethod
    def memory_leak_check():
        # Should be called before each test.
        # Only cublas workspaces and some global tensors are allowed to be allocated.
        # All other allocations should be released.
        # This is a simple check to catch memory leaks.
        assert Utils.get_cuda_memory_mb() < 100, f"Memory leak: {Utils.get_cuda_memory_mb()} MB"

class AddOneLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

class TestsBackend:
    @pytest.mark.parametrize("reuse_gpu_buffers", [True, False])
    @pytest.mark.parametrize("random_num_tensors", [True, False])
    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_general(self, reuse_gpu_buffers, random_num_tensors, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        if reuse_gpu_buffers and random_num_tensors:
            pytest.skip("Cannot have random number of tensors and reuse_gpu_buffers at the same time.")
        backend = _CPUOffloadBackend(
            reuse_gpu_buffers=reuse_gpu_buffers, 
        )
        NUM_LAYERS = 10
        NUM_ITERATIONS = 10

        for _ in range(NUM_ITERATIONS):
            original_tensors = []
            tensors_cpu = []
            layer_ids = []

            for i in range(NUM_LAYERS):
                NUM_LAYER_TENSORS = random.randint(1, 10) if random_num_tensors else 1
                layer_tensors = []
                layer_tensors_cpu = []
                backend.start_offloaded_layer_fwd()
                for _ in range(NUM_LAYER_TENSORS):
                    tensor = Utils.create_tensor(recipe_name)
                    layer_tensors.append(tensor)
                    if random.randint(0, 1) == 0:
                        backend.mark_can_start_offload(tensor)
                    tensor_cpu = backend.offload(tensor)
                    assert tensor.device.type == "cuda"
                    assert tensor_cpu.device.type == "cpu"
                    layer_tensors_cpu.append(tensor_cpu)
                layer_id = backend.end_offloaded_layer_fwd()
                layer_ids.append(layer_id)
                tensors_cpu.append(layer_tensors_cpu)
                original_tensors.append(layer_tensors)
            backend.finish_fwd()
            backend.start_bwd_reloading()
            for i in range(NUM_LAYERS - 1, -1, -1):
                backend.start_offloaded_layer_bwd(layer_ids[i])
                for j in range(len(tensors_cpu[i])):
                    tensor_gpu = backend.reload(tensors_cpu[i][j])
                    assert tensor_gpu.device.type == "cuda"
                    assert tensor_gpu.shape == original_tensors[i][j].shape
                    assert tensor_gpu.dtype == original_tensors[i][j].dtype
                    torch.testing.assert_close(tensor_gpu, original_tensors[i][j])
                backend.end_offloaded_layer_bwd()
        torch.cuda.synchronize()
    
    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_memory(self, recipe_name):
        Utils.skip_if_recipe_not_supported(recipe_name)
        #reset max memory allocated
        init_cuda_memory = Utils.get_cuda_memory_mb()
        x = Utils.create_tensor(recipe_name)
        x_size = Utils.get_tensor_size_mb(x)
        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory + x_size, 0.1)

        torch.cuda.synchronize()
        backend = _CPUOffloadBackend()
        backend.start_offloaded_layer_fwd()
        x1_cpu = backend.offload(x)
        del x
        num1 = backend.end_offloaded_layer_fwd()

        torch.cuda.synchronize()

        # Memory is not released yet.
        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory + x_size, 0.1)

        backend.start_offloaded_layer_fwd()
        # Next offloaded layer, memory should be released.
        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory, 0.1)

        x = Utils.create_tensor(recipe_name)
        x2_cpu = backend.offload(x)
        del x
        num2 = backend.end_offloaded_layer_fwd()

        backend.start_offloaded_layer_fwd()
        x = Utils.create_tensor(recipe_name)
        x3_cpu = backend.offload(x)
        del x
        num3 = backend.end_offloaded_layer_fwd()

        backend.finish_fwd()
        torch.cuda.reset_max_memory_allocated()

        backend.start_bwd_reloading()
        torch.cuda.synchronize()
        
        backend.start_offloaded_layer_bwd(num3)
        backend.reload(x3_cpu)
        backend.end_offloaded_layer_bwd()

        torch.cuda.synchronize()

        backend.start_offloaded_layer_bwd(num2)
        backend.reload(x2_cpu)
        backend.end_offloaded_layer_bwd()

        torch.cuda.synchronize()

        backend.start_offloaded_layer_bwd(num1)
        backend.reload(x1_cpu)
        backend.end_offloaded_layer_bwd()


        torch.cuda.synchronize()
        # Third copy is released.
        assert Utils.get_max_cuda_memory_mb() < init_cuda_memory + 2 * x_size + 0.1

    def test_mark_can_start_offload(self):
        """
        Check that calling `mark_can_start_offload` lets the backend overlap the
        D2H copy with computation.  The runtime with the mark should therefore
        be strictly smaller than without it.
        """
        torch.cuda.synchronize()
        tensor = torch.randn((128, 512, 512), device="cuda")

        def _timed_run(use_mark: bool) -> float:
            """Run a single forward pass and return its wall-clock time (ms)."""
            def _run():
                backend = _CPUOffloadBackend()
                backend.start_offloaded_layer_fwd()
                if use_mark:
                    backend.mark_can_start_offload(tensor)

                # Simulate compute that should overlap with the offload copy.
                Utils.long_job()

                backend.offload(tensor)
                backend.end_offloaded_layer_fwd()
                backend.finish_fwd()
                backend.start_bwd_reloading()

                # Make sure all CUDA work is finished before timing stops.
                torch.cuda.current_stream().synchronize()
                torch.cuda.synchronize()

            return Utils.measure_time(_run)

        # Warm-up
        _timed_run(False)
        time_without_mark = _timed_run(False)
        time_with_mark = _timed_run(True)
        print(f"time_without_mark: {time_without_mark} ms, "
              f"time_with_mark: {time_with_mark} ms")
        assert time_with_mark < time_without_mark

    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_multiple_tensor_offload(self, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        init_cpu_memory = Utils.get_cpu_memory_mb()
        init_cuda_memory = Utils.get_cuda_memory_mb()
        backend = _CPUOffloadBackend()
        backend.start_offloaded_layer_fwd()
        x1 = Utils.create_tensor(recipe_name)
        x_size = Utils.get_tensor_size_mb(x1)
        backend.offload(x1)
        backend.offload(x1)
        backend.offload(x1)
        # Only one copy of tensor on cpu is allocated.
        assert Utils.get_cpu_memory_mb() == pytest.approx(init_cpu_memory + 1 * x_size, 0.1)
        del x1
        backend.end_offloaded_layer_fwd()
        backend.finish_fwd()

        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory, 0.1)
    
    @pytest.mark.parametrize("job_forward", [True, False])
    @pytest.mark.parametrize("job_backward", [True, False])
    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_overlap(self, job_forward, job_backward, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        if not job_forward and not job_backward:
            pytest.skip("")
        NUM_LAYERS = 10
        def _run(job_forward, job_backward, offloads):
            backend = _CPUOffloadBackend()
            offloaded_tensors = []
            layer_ids = []
            for _ in range(NUM_LAYERS):
                backend.start_offloaded_layer_fwd()
                if offloads:
                    offloaded_tensors.append(backend.offload(Utils.create_tensor(recipe_name)))
                else:
                    offloaded_tensors.append(Utils.create_tensor(recipe_name))
                if job_forward:
                    Utils.long_job()
                layer_id = backend.end_offloaded_layer_fwd()
                layer_ids.append(layer_id)
            backend.finish_fwd()
            backend.start_bwd_reloading()
            for i in range(NUM_LAYERS - 1, -1, -1):
                backend.start_offloaded_layer_bwd(layer_ids[i])
                if offloads:
                    backend.reload(offloaded_tensors[i])
                if job_backward:
                    Utils.long_job()
                backend.end_offloaded_layer_bwd()
            torch.cuda.synchronize()
        
        def _measure_time(job_forward, job_backward, offloads):
            return Utils.measure_time(lambda: _run(job_forward, job_backward, offloads))

        _run(True, True, True) # warm-up

        time_offload_only = _measure_time(False, False, True)
        time_offload_and_selected_jobs = _measure_time(job_forward, job_backward, True)
        time_selected_jobs = _measure_time(job_forward, job_backward, False)

        print(f"time_offload_only: {time_offload_only:.2f} ms, "
              f"time_offload_and_selected_jobs: {time_offload_and_selected_jobs:.2f} ms, "
              f"time_selected_jobs: {time_selected_jobs:.2f} ms")

        assert time_offload_only + time_selected_jobs > time_offload_and_selected_jobs + EPSILON


class TestTEAPI:
    @pytest.mark.parametrize("layer_type", Utils.get_layer_names())
    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_offload_one_layer(self, layer_type, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        torch.cuda.synchronize()
        cpu_offload = CPUOffload()
        recipe_ctx = Utils.create_recipe_ctx(recipe_name)
        layer = Utils.create_layer(layer_type)
        last_layer = AddOneLayer()
        inp = Utils.create_tensor("high precision")

        m_splits = {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H} \
            if layer_type == "grouped_linear" else {}
        
        with recipe_ctx():
            out = layer(inp, is_first_microbatch=True, **m_splits)
            out = last_layer(out)
        out.sum().backward()

        # run with is_first_microbatch=True to cache the fp8 casts
        del inp
        init_cuda_memory = Utils.get_cuda_memory_mb()
        inp = Utils.create_tensor("high precision")
        with recipe_ctx():
            out = layer(inp, is_first_microbatch=False, **m_splits)
            out = last_layer(out)
        del inp
        activation_size = Utils.get_cuda_memory_mb() - init_cuda_memory
        out.sum().backward()

        init_cuda_memory = Utils.get_cuda_memory_mb()

        # run layer with offload
        layer_offload = cpu_offload(layer, offload_activations=True)
        last_layer_offload = cpu_offload(last_layer, is_last_layer=True)
        inp = Utils.create_tensor("high precision")
        with recipe_ctx():
            out = layer_offload(inp, is_first_microbatch=False, **m_splits)
            out = last_layer_offload(out)
        del inp
        offloaded_size = cpu_offload.backend.get_offloaded_total_size_mb()
        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory, 0.1)
        assert offloaded_size == pytest.approx(activation_size, 0.1)

        out.sum().backward()
    
    @pytest.mark.parametrize("layer_type", Utils.get_layer_names())
    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_pipeline_parallel(self, layer_type, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        recipe_ctx = Utils.create_recipe_ctx(recipe_name)
        cpu_offload = CPUOffload()

        layer1 = Utils.create_layer(layer_type)
        layer2 = Utils.create_layer(layer_type)
        layer3 = Utils.create_layer(layer_type)

        def _run(inp):
            m_splits = {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H} \
                if layer_type == "grouped_linear" else {}
            
            with recipe_ctx():
                out = layer1(inp, **m_splits)
                out = layer2(out, **m_splits)
                out = layer3(out, **m_splits)
            return out.sum()
    

        def _run_offload(inp):
            m_splits = {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H} \
                if layer_type == "grouped_linear" else {}
            layer1_offload = cpu_offload(layer1, offload_activations=True)
            layer2_offload = cpu_offload(layer2, offload_activations=True)
            layer3_offload = cpu_offload(layer3, is_last_layer=True)
            with recipe_ctx():
                out = layer1_offload(inp, **m_splits)
                out = layer2_offload(out, **m_splits)
                out = layer3_offload(out, **m_splits)
            return out.sum()

        inps = [Utils.create_tensor("high precision", requires_grad=True) for _ in range(3)]

        outs = []
        for i in range(3):
            outs.append(_run(inps[i]))
        for out in outs:
            out.backward()
        
        inps_offload = [inps[i].clone().detach() for i in range(3)]
        for i in range(3):
            inps_offload[i] = inps_offload[i].requires_grad_()
        # run with offload
        outs_offload = []
        for i in range(3):
            outs_offload.append(_run_offload(inps_offload[i]))
        for out in outs_offload:
            out.backward()
        
        # check if inp grads are the same
        if recipe_name == "high precision":
            for i in range(3):
                assert torch.allclose(inps[i].grad, inps_offload[i].grad)
    
    def test_fake_tensor(self):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported("high precision")
        layer = Utils.create_layer("linear")
        cpu_offload = CPUOffload()
        layer_offload = cpu_offload(layer)
        layer_offload_2 = cpu_offload(layer)
        layer_offload_3 = cpu_offload(layer, is_last_layer=True)
        inp = Utils.create_tensor("high precision")
        model = torch.nn.Sequential(layer_offload, layer_offload_2, layer_offload_3)
        # torch compile model
        model = torch.compile(model)
        out = model(inp)
        out.sum().backward()
    

class TestLegacyAPI:
    @pytest.mark.parametrize("layer_type", Utils.get_layer_names())
    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_legacy_api(self, layer_type, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        offload_ctx, sync_function = get_cpu_offload_context(
            enabled=True,
            num_layers=1,
            model_layers=2,
            offload_activations=True,
            offload_weights=False
        )
        recipe_ctx = Utils.create_recipe_ctx(recipe_name)
        layer = Utils.create_layer(layer_type)
        inp = Utils.create_tensor("high precision")

        m_splits = {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H} \
            if layer_type == "grouped_linear" else {}
        
        with recipe_ctx():
            out = layer(inp, is_first_microbatch=True, **m_splits)
        out.sum().backward()

        init_cuda_memory = Utils.get_cuda_memory_mb()

        # run layer with offload
        inp = Utils.create_tensor("high precision")
        with offload_ctx, recipe_ctx():
            out = layer(inp, is_first_microbatch=False, **m_splits)
        out = sync_function(out)
        with offload_ctx, recipe_ctx():
            out = out + 1
        out = sync_function(out)
        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory, 0.1)
        out.sum().backward()
    
    @pytest.mark.parametrize("layer_type", Utils.get_layer_names())
    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_sanity_legacy_api(self, layer_type, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        OFFLOAD_LAYERS = 6
        NUM_LAYERS = 10
        offload_ctx, sync_function = get_cpu_offload_context(
            enabled=True,
            num_layers=OFFLOAD_LAYERS,
            model_layers=NUM_LAYERS,
        )
        recipe_ctx = Utils.create_recipe_ctx(recipe_name)
        layers = [Utils.create_layer(layer_type) for _ in range(NUM_LAYERS)]
        inp = Utils.create_tensor("high precision")
        m_splits = {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H} \
            if layer_type == "grouped_linear" else {}
        for i in range(NUM_LAYERS):
            with offload_ctx, recipe_ctx():
                out = layers[i](inp, is_first_microbatch=False, **m_splits)
        out = sync_function(out)
