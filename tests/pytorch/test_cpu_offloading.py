# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import random
import contextlib
import pytest
import torch
from typing import Optional
from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context, OffloadSynchronizer
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
        return [
            "linear",
            "layernorm_linear",
            "layernorm_mlp",
            "grouped_linear",
            "multihead_attention",
            "transformer_layer",
        ]

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
            return te.TransformerLayer(
                Utils._D, Utils._D, Utils._H, attention_dropout=0.0, hidden_dropout=0.0
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    @staticmethod
    def get_recipe_names():
        return [
            "high precision",
            "fp8_delayed_scaling",
            "fp8_current_scaling",
            "fp8_block_scaling",
            "mxfp8",
        ]

    @staticmethod
    def create_tensor(recipe_name: str, requires_grad: bool = False) -> torch.Tensor:
        shape = (Utils._B, Utils._S, Utils._D)
        tensor = torch.randn(shape, device="cuda")
        if recipe_name == "high precision":
            tensor = tensor.requires_grad_() if requires_grad else tensor
            return tensor
        elif recipe_name == "fp8_delayed_scaling":
            quantizer = te.tensor.float8_tensor.Float8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                scale=torch.tensor([1.0], device="cuda"),
                amax=torch.tensor([1.0], device="cuda"),
            )
            return quantizer(tensor)
        elif recipe_name == "fp8_current_scaling":
            quantizer = te.tensor.float8_tensor.Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, device="cuda"
            )
            return quantizer(tensor)
        elif recipe_name == "fp8_block_scaling":
            quantizer = te.tensor.float8_blockwise_tensor.Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True
            )
            return quantizer(tensor)
        elif recipe_name == "mxfp8":
            quantizer = te.tensor.mxfp8_tensor.MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
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
        if Utils.get_cuda_memory_mb() > 100:
            memory_num = Utils.get_cuda_memory_mb()
            import gc

            gc.collect()  # We want next test to be run with clean state.
            gc.disable()
            raise RuntimeError(f"Memory leak: {memory_num} MB")


class TestsOffloadSynchronizer:
    @pytest.mark.parametrize("random_num_tensors", [True, False])
    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_general(self, random_num_tensors, recipe_name):
        """
        Test general functionality of OffloadSynchronizer - offload NUM_LAYERS-1 out of NUM_LAYERS layers,
        for each layer offload random number of random tensors.
        Then do backward pass for each layer, and check if reloaded tensors are equal to original tensors.
        """
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        NUM_LAYERS = 10
        NUM_ITERATIONS = 10

        offload_synchronizer = OffloadSynchronizer(
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
                    tensor = Utils.create_tensor(recipe_name)
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

    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_memory(self, recipe_name):
        torch.cuda.synchronize()
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        NUM_LAYERS = 10

        torch.cuda.reset_peak_memory_stats()

        offload_synchronizer = OffloadSynchronizer(
            num_layers=NUM_LAYERS,
            num_offloaded_layers=NUM_LAYERS - 1,
        )

        init_cuda_memory = Utils.get_cuda_memory_mb()

        tensor_ids = []

        torch.cuda.synchronize()
        for _ in range(NUM_LAYERS):
            offload_synchronizer.fwd_step()
            # print max memory allocated
            tensor = Utils.create_tensor(recipe_name)
            tensor_size = Utils.get_tensor_size_mb(tensor)
            tensor_id = offload_synchronizer.push_tensor(tensor)
            assert tensor.device.type == "cuda"
            tensor_ids.append(tensor_id)
            del tensor, tensor_id
        torch.cuda.synchronize()

        if recipe_name == "high precision":
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

        if recipe_name == "high precision":
            assert Utils.get_max_cuda_memory_mb() == pytest.approx(
                init_cuda_memory + tensor_size, 0.1
            )
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
            offload_synchronizer = OffloadSynchronizer(
                num_layers=NUM_LAYERS,
                num_offloaded_layers=NUM_LAYERS - 1,
            )
            offloaded_tensors = []
            layer_ids = []
            for _ in range(NUM_LAYERS):
                layer_id = offload_synchronizer.fwd_step()
                layer_ids.append(layer_id)
                if offloads:
                    offloaded_tensors.append(
                        offload_synchronizer.push_tensor(Utils.create_tensor(recipe_name))
                    )
                else:
                    offloaded_tensors.append(Utils.create_tensor(recipe_name))
                if job_forward:
                    Utils.long_job()
            for i in range(NUM_LAYERS - 1, -1, -1):
                offload_synchronizer.bwd_step(layer_ids[i])
                if offloads:
                    offload_synchronizer.pop_tensor(offloaded_tensors[i])
                if job_backward:
                    Utils.long_job()
            torch.cuda.synchronize()

        def _measure_time(job_forward, job_backward, offloads):
            return Utils.measure_time(lambda: _run(job_forward, job_backward, offloads))

        _run(True, True, True)  # warm-up

        time_offload_only = _measure_time(False, False, True)
        time_offload_and_selected_jobs = _measure_time(job_forward, job_backward, True)
        time_selected_jobs = _measure_time(job_forward, job_backward, False)

        print(
            f"time_offload_only: {time_offload_only:.2f} ms, "
            f"time_offload_and_selected_jobs: {time_offload_and_selected_jobs:.2f} ms, "
            f"time_selected_jobs: {time_selected_jobs:.2f} ms"
        )

        assert time_offload_only + time_selected_jobs > time_offload_and_selected_jobs + EPSILON

    def test_offload_base_tensor(self):
        Utils.memory_leak_check()
        offload_synchronizer = OffloadSynchronizer(
            num_layers=2,
            num_offloaded_layers=1,
        )
        init_cuda_memory = Utils.get_cuda_memory_mb()
        x = Utils.create_tensor("high precision")
        x_size = Utils.get_tensor_size_mb(x)
        x_1 = x[::2]
        x_2 = x[1::2]

        offload_synchronizer.fwd_step()
        x1_id = offload_synchronizer.push_tensor(x_1, offload_base_tensor=True)
        x2_id = offload_synchronizer.push_tensor(x_2, offload_base_tensor=True)
        del x_1, x_2
        offload_synchronizer.fwd_step()

        assert offload_synchronizer.get_offloaded_total_size_mb() == pytest.approx(x_size, 0.1)

        offload_synchronizer.bwd_step(0)
        x_1 = offload_synchronizer.pop_tensor(x1_id)
        x_2 = offload_synchronizer.pop_tensor(x2_id)
        assert x_1.device.type == "cuda"
        assert x_2.device.type == "cuda"

        assert torch.allclose(x_1, x[::2])
        assert torch.allclose(x_2, x[1::2])
        del x

        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory + x_size, 0.1)

    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_multiple_tensor_offload(self, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        init_cpu_memory = Utils.get_cpu_memory_mb()
        init_cuda_memory = Utils.get_cuda_memory_mb()
        offload_synchronizer = OffloadSynchronizer(
            num_layers=2,
            num_offloaded_layers=1,
        )
        x1 = Utils.create_tensor(recipe_name)
        x_size = Utils.get_tensor_size_mb(x1)
        offload_synchronizer.fwd_step()
        offload_synchronizer.push_tensor(x1)
        offload_synchronizer.push_tensor(x1)
        offload_synchronizer.push_tensor(x1)
        offload_synchronizer.fwd_step()
        # Only one copy of tensor on cpu is allocated.
        assert Utils.get_cpu_memory_mb() == pytest.approx(init_cpu_memory + 1 * x_size, 0.1)
        del x1
        offload_synchronizer.bwd_step(1)
        offload_synchronizer.bwd_step(0)
        offload_synchronizer.finish_part_of_bwd()

        assert Utils.get_cuda_memory_mb() == pytest.approx(init_cuda_memory, 0.1)

    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_offload_start_synchronization(self, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        NUM_ITERATIONS = 10
        offload_synchronizer = OffloadSynchronizer(
            num_layers=2,
            num_offloaded_layers=1,
        )
        x = Utils.create_tensor(recipe_name)

        def job_first():
            for _ in range(NUM_ITERATIONS):
                Utils.long_job()
                offload_synchronizer.fwd_step()
                offload_synchronizer.push_tensor(x)
                offload_synchronizer.fwd_step()

                offload_synchronizer.bwd_step(1)
                offload_synchronizer.bwd_step(0)
                offload_synchronizer.finish_part_of_bwd()
                torch.cuda.synchronize()

        def offload_first():
            for _ in range(NUM_ITERATIONS):
                offload_synchronizer.fwd_step()
                offload_synchronizer.push_tensor(x)
                Utils.long_job()
                offload_synchronizer.fwd_step()

                offload_synchronizer.bwd_step(1)
                offload_synchronizer.bwd_step(0)
                offload_synchronizer.finish_part_of_bwd()
                torch.cuda.synchronize()

        def only_job():
            for _ in range(NUM_ITERATIONS):
                Utils.long_job()
                torch.cuda.synchronize()

        def only_offload():
            for _ in range(NUM_ITERATIONS):
                offload_synchronizer.fwd_step()
                offload_synchronizer.push_tensor(x)
                offload_synchronizer.fwd_step()
                offload_synchronizer.bwd_step(1)
                offload_synchronizer.bwd_step(0)
                offload_synchronizer.finish_part_of_bwd()
                torch.cuda.synchronize()

        # measure time of offload_first
        time_offload_first = Utils.measure_time(offload_first)
        time_job_first = Utils.measure_time(job_first)
        time_only_offload = Utils.measure_time(only_offload)
        time_only_job = Utils.measure_time(only_job)

        print(
            f"time_offload_first: {time_offload_first:.2f} ms, "
            f"time_job_first: {time_job_first:.2f} ms, "
            f"time_only_offload: {time_only_offload:.2f} ms, "
            f"time_only_job: {time_only_job:.2f} ms"
        )

        assert time_offload_first < time_job_first

    def test_synchronization_dict(self):
        Utils.memory_leak_check()
        NUM_LAYERS = 10

        torch.cuda.reset_peak_memory_stats()

        offload_synchronizer = OffloadSynchronizer(
            num_layers=NUM_LAYERS,
            synchronization_dict={
                0: (True, 5, False, 2),
                1: (True, 5, False, 2),
                5: (False, 8, False, 7),
                6: (False, 8, False, 7),
            },
        )

        tensor_size = Utils.get_tensor_size_mb(Utils.create_tensor("high precision"))

        init_cuda_memory = Utils.get_cuda_memory_mb()

        tensor_ids = []

        memory_before_fwd = []
        memory_before_bwd: list[float | None] = [None] * NUM_LAYERS

        torch.cuda.synchronize()
        for _ in range(NUM_LAYERS):
            memory_before_fwd.append(Utils.get_cuda_memory_mb())
            tensor = Utils.create_tensor("high precision")
            offload_synchronizer.fwd_step()
            tensor_id = offload_synchronizer.push_tensor(tensor)
            assert tensor.device.type == "cuda"
            tensor_ids.append(tensor_id)
            del tensor, tensor_id
        torch.cuda.synchronize()

        for i in range(NUM_LAYERS - 1, -1, -1):
            offload_synchronizer.bwd_step(i)
            memory_before_bwd[i] = Utils.get_cuda_memory_mb()
            del tensor_ids[i]
        offload_synchronizer.finish_part_of_bwd()
        torch.cuda.synchronize()

        assert memory_before_fwd[5] == pytest.approx(init_cuda_memory + 5 * tensor_size, 0.1)

        # layers 0 and 1 finish offloading before fwd of layer 5
        assert memory_before_fwd[6] == pytest.approx(init_cuda_memory + 4 * tensor_size, 0.1)

        # layers 0, 1 offloaded, layers 5, 6 not offloaded yet
        assert memory_before_bwd[9] == pytest.approx(
            init_cuda_memory + (NUM_LAYERS - 2) * tensor_size, 0.1
        )

        # 0, 1, 5, 6 offloaded, nothing started reloading yet
        assert memory_before_bwd[8] == pytest.approx(init_cuda_memory + 5 * tensor_size, 0.1)

        # 0, 1, offloaded, 5, 6 started reloading
        assert memory_before_bwd[7] == pytest.approx(init_cuda_memory + 6 * tensor_size, 0.1)

        # 2, 3 not offloaded
        assert memory_before_bwd[3] == pytest.approx(init_cuda_memory + 2 * tensor_size, 0.1)

        # 0, 1 reloading, 2 not offloaded
        assert memory_before_bwd[2] == pytest.approx(init_cuda_memory + 3 * tensor_size, 0.1)


class TestTELayers:
    @pytest.mark.parametrize("layer_type", Utils.get_layer_names())
    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_sanity(self, layer_type, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        init_cuda_memory = Utils.get_cuda_memory_mb()
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
        m_splits = (
            {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H}
            if layer_type == "grouped_linear"
            else {}
        )
        out = inp
        for i in range(NUM_LAYERS):
            with offload_ctx, recipe_ctx():
                out = layers[i](out, is_first_microbatch=False, **m_splits)
            out = sync_function(out)
        out.sum().backward()
        torch.cuda.synchronize()
        del out, inp, layers

    @pytest.mark.parametrize("layer_type", Utils.get_layer_names())
    @pytest.mark.parametrize("recipe_name", Utils.get_recipe_names())
    def test_memory(self, layer_type, recipe_name):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        offload_ctx, sync_function = get_cpu_offload_context(
            enabled=True,
            num_layers=1,
            model_layers=2,
            offload_activations=True,
            offload_weights=False,
        )
        recipe_ctx = Utils.create_recipe_ctx(recipe_name)
        layer = Utils.create_layer(layer_type)
        inp = Utils.create_tensor("high precision")

        m_splits = (
            {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H}
            if layer_type == "grouped_linear"
            else {}
        )

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
    def test_vpp_offloading_scenario(self, recipe_name, layer_type):
        Utils.memory_leak_check()
        Utils.skip_if_recipe_not_supported(recipe_name)
        offload_ctx, sync_function = get_cpu_offload_context(
            enabled=True,
            model_layers=6,
            offload_activations=True,
            synchronization_dict={
                0: (True, 1, False, 1),
                2: (True, 3, False, 3),
            },
        )
        pp1_1 = Utils.create_layer(layer_type)
        pp1_2 = Utils.create_layer(layer_type)
        pp2_1 = Utils.create_layer(layer_type)
        pp2_2 = Utils.create_layer(layer_type)
        pp3_1 = Utils.create_layer(layer_type)
        pp3_2 = Utils.create_layer(layer_type)
        inp = Utils.create_tensor("high precision")

        recipe_ctx = Utils.create_recipe_ctx(recipe_name)

        m_splits = (
            {"m_splits": [Utils._B * Utils._S // Utils._H] * Utils._H}
            if layer_type == "grouped_linear"
            else {}
        )

        # 1 fwd
        with offload_ctx, recipe_ctx():
            pp1_1_out = pp1_1(inp, is_first_microbatch=True, **m_splits)
        pp1_1_out = sync_function(pp1_1_out)
        with offload_ctx, recipe_ctx():
            pp1_2_out = pp1_2(pp1_1_out, is_first_microbatch=False, **m_splits)

        # 2 fwd
        pp1_2_out = sync_function(pp1_2_out)
        with offload_ctx, recipe_ctx():
            pp2_1_out = pp2_1(inp, is_first_microbatch=True, **m_splits)
        pp2_1_out = sync_function(pp2_1_out)
        with offload_ctx, recipe_ctx():
            pp2_2_out = pp2_2(pp2_1_out, is_first_microbatch=False, **m_splits)
        pp2_2_out = sync_function(pp2_2_out)

        # 1 bwd
        pp1_2_out.sum().backward()

        # 3 fwd
        with offload_ctx, recipe_ctx():
            pp3_1_out = pp3_1(inp, is_first_microbatch=True, **m_splits)
        pp3_1_out = sync_function(pp3_1_out)
        with offload_ctx, recipe_ctx():
            pp3_2_out = pp3_2(pp3_1_out, is_first_microbatch=False, **m_splits)
        pp3_2_out = sync_function(pp3_2_out)

        # 2 bwd
        pp2_2_out.sum().backward()

        # 3 bwd
        pp3_2_out.sum().backward()
