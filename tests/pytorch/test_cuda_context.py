# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Restart-boundary cleanup without replacing PyTorch's CUDA runtime."""

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import get_cached_ones_tensor
from transformer_engine.pytorch.cpp_extensions.gemm import (
    get_cublas_workspace,
    strided_batched_gemm,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.common.recipe import DelayedScaling
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm_for_grouped_tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_context_loss_rejects_initialized_framework_without_mutation():
    x = torch.ones(32, device="cuda")
    generation = tex.get_cuda_state_version()
    with pytest.raises(RuntimeError, match="initialized PyTorch CUDA runtime"):
        te.reset_cuda_state()
    assert tex.get_cuda_state_version() == generation
    torch.testing.assert_close(x, torch.ones_like(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_live_cleanup_rebuilds_gemm_caches_and_preserves_model():
    device = torch.cuda.current_device()
    model = te.Linear(128, 128, bias=False, params_dtype=torch.float32)
    x = torch.randn(8, 128, device="cuda")
    reference = torch.nn.functional.linear(x, model.weight)
    before = model(x)
    old_ones = get_cached_ones_tensor(32, torch.float32, x.device)
    old_scratch = get_cublas_workspace(device, False, False)
    cpu_ones = get_cached_ones_tensor(16, torch.float32, torch.device("cpu"))
    initial = tex.get_cuda_state_version()
    for _ in range(3):
        te.release_cuda_state()
        assert tex.get_cuda_state_version() > initial
        initial = tex.get_cuda_state_version()
        torch.testing.assert_close(model(x), reference)
        torch.testing.assert_close(before, reference)
        torch.testing.assert_close(old_ones, torch.ones_like(old_ones))
    assert get_cached_ones_tensor(32, torch.float32, x.device) is not old_ones
    assert get_cublas_workspace(device, False, False) is not old_scratch
    assert get_cached_ones_tensor(16, torch.float32, torch.device("cpu")) is cpu_ones


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_live_cleanup_rebuilds_strided_batched_gemm_handle():
    a = torch.ones((2, 16, 16), dtype=torch.bfloat16, device="cuda")
    b = a * 2
    b[1] *= 2
    output = torch.empty_like(a)
    expected = torch.bmm(b, a.transpose(1, 2))
    for _ in range(3):
        strided_batched_gemm(a, b, output, layout="TN")
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        te.release_cuda_state()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_global_cleanup_rebuilds_all_device_caches_and_preserves_current_device():
    a = get_cached_ones_tensor(32, torch.float32, torch.device("cuda:0"))
    b = get_cached_ones_tensor(32, torch.float32, torch.device("cuda:1"))
    cpu = get_cached_ones_tensor(32, torch.float32, torch.device("cpu"))
    scratch = [get_cublas_workspace(device, False, False) for device in (0, 1)]
    version = tex.get_cuda_state_version()
    with torch.cuda.device(1):
        te.release_cuda_state()
        assert torch.cuda.current_device() == 1
    assert tex.get_cuda_state_version() == version + 1
    for device, old in enumerate((a, b)):
        assert get_cached_ones_tensor(32, torch.float32, torch.device("cuda", device)) is not old
        assert get_cublas_workspace(device, False, False) is not scratch[device]
        torch.testing.assert_close(old, torch.ones_like(old))
    assert get_cached_ones_tensor(32, torch.float32, torch.device("cpu")) is cpu


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Blackwell required for grouped MXFP8",
)
def test_grouped_tma_storage_is_rebuilt_after_release():
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
    quantizer.set_usage(rowwise=True, columnwise=True)
    for groups in (2, 65):
        first = [128] * groups
        last = [128] * groups
        last[-1] = 256
        x = torch.randn(
            1, sum(m * n for m, n in zip(first, last)), device="cuda", dtype=torch.bfloat16
        )
        rows = torch.tensor(first, device="cuda", dtype=torch.int64)
        cols = torch.tensor(last, device="cuda", dtype=torch.int64)
        expected = tex.group_quantize(x, quantizer, groups, rows, cols)
        for _ in range(3):
            te.release_cuda_state()
            actual = tex.group_quantize(x, quantizer, groups, rows, cols)
            assert torch.equal(actual.rowwise_data, expected.rowwise_data)
            assert torch.equal(actual.columnwise_data, expected.columnwise_data)
            assert torch.equal(actual.scale_inv, expected.scale_inv)
            assert torch.equal(actual.columnwise_scale_inv, expected.columnwise_scale_inv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_release_is_rejected_during_graph_capture():
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ones(1, device="cuda").add_(1)
        with pytest.raises(RuntimeError, match="outside graph capture"):
            te.release_cuda_state()


@pytest.mark.skipif(not te.is_fp8_available(), reason="FP8 required")
def test_release_preserves_fp8_history_and_rebuilds_registrations():
    model = te.Linear(128, 128, params_dtype=torch.bfloat16)
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    recipe = DelayedScaling(amax_history_len=4)
    for _ in range(2):
        with te.autocast(recipe=recipe):
            model(x).float().square().mean().backward()
    index_key = FP8GlobalStateManager.get_buffer_info()
    assert index_key in model.fp8_meta
    history = model.fp8_meta["scaling_fwd"].amax_history.clone()
    scale = model.fp8_meta["scaling_fwd"].scale.clone()
    te.release_cuda_state()
    assert index_key not in model.fp8_meta
    torch.testing.assert_close(model.fp8_meta["scaling_fwd"].amax_history, history)
    torch.testing.assert_close(model.fp8_meta["scaling_fwd"].scale, scale)
    model.zero_grad(set_to_none=True)
    x.grad = None
    with te.autocast(recipe=recipe):
        output = model(x)
    output.float().square().mean().backward()
    assert index_key in model.fp8_meta
    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_release_rejects_active_fp8_autocast():
    generation = tex.get_cuda_state_version()
    with te.autocast():
        with pytest.raises(RuntimeError, match="outside FP8 autocast"):
            te.release_cuda_state()
    assert tex.get_cuda_state_version() == generation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_release_recreates_sequence_and_nvfp4_tensor_caches():
    from transformer_engine.pytorch.attention.dot_product_attention.utils import get_full_cu_seqlens
    from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
        get_seq_chunk_ids_for_reordering_before_attn,
    )
    from transformer_engine.pytorch.tensor.nvfp4_tensor import get_rht_matrix
    from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import _fp4_e2m1_vals

    device = torch.device("cuda", torch.cuda.current_device())
    values = (
        get_full_cu_seqlens(4, 128, device),
        get_seq_chunk_ids_for_reordering_before_attn(2, device),
        get_rht_matrix(False, device.index),
        _fp4_e2m1_vals(device, torch.float32),
    )
    te.release_cuda_state()
    recreated = (
        get_full_cu_seqlens(4, 128, device),
        get_seq_chunk_ids_for_reordering_before_attn(2, device),
        get_rht_matrix(False, device.index),
        _fp4_e2m1_vals(device, torch.float32),
    )
    for old, new in zip(values, recreated):
        assert old is not new
        torch.testing.assert_close(old, new)


@pytest.mark.skipif(
    not torch.cuda.is_available() or tex.get_cublasLt_build_version() < 130300,
    reason="cuBLASLt grouped GEMM requires CUDA 13.3+",
)
def test_live_cleanup_recreates_grouped_gemm_handle():
    device = torch.device("cuda", torch.cuda.current_device())
    groups, m, n, k = 2, 32, 64, 64

    def grouped(rows, cols):
        return GroupedTensor.make_grouped_tensor(
            num_tensors=groups,
            first_dims=None,
            last_dims=None,
            logical_first_dim=groups * rows,
            logical_last_dim=cols,
            quantizer=None,
            device=device,
            dtype=torch.bfloat16,
        )

    weight, value, output = grouped(n, k), grouped(m, k), grouped(m, n)
    weight.rowwise_data.copy_((torch.arange(groups * n * k, device=device) % 7 - 3) / 8)
    value.rowwise_data.copy_((torch.arange(groups * m * k, device=device) % 5 - 2) / 8)
    reference = torch.bmm(
        value.rowwise_data.view(groups, m, k),
        weight.rowwise_data.view(groups, n, k).transpose(1, 2),
    )
    for _ in range(3):
        general_grouped_gemm_for_grouped_tensor(weight, value, output, layout="TN")
        torch.testing.assert_close(output.rowwise_data.view_as(reference), reference)
        te.release_cuda_state()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_global_release_rejects_capture_on_another_device_before_mutation():
    cached = [get_cached_ones_tensor(32, torch.float32, torch.device("cuda", d)) for d in (0, 1)]
    version = tex.get_cuda_state_version()
    with torch.cuda.device(1):
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            torch.ones(1, device="cuda:1").add_(1)
            with torch.cuda.device(0):
                with pytest.raises(RuntimeError, match="outside graph capture"):
                    te.release_cuda_state()
    assert tex.get_cuda_state_version() == version
    for device, old in enumerate(cached):
        assert get_cached_ones_tensor(32, torch.float32, torch.device("cuda", device)) is old


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not te.is_fp8_available(),
    reason="Two FP8 CUDA devices required",
)
def test_global_release_preserves_fp8_history_on_two_devices(monkeypatch):
    # The existing fused recipe updater uses the calling device's stream even
    # when processing another device's reduction bucket. Exercise the supported
    # unfused path here; the single-device regression above covers fused update.
    monkeypatch.setenv("NVTE_UNFUSED_FP8_UPDATE", "1")
    states = []
    key = FP8GlobalStateManager.get_buffer_info()
    for device in (0, 1):
        with torch.cuda.device(device):
            # Delayed-scaling buffers are keyed by recipe/group, not by device.
            # Keep the two models in independent reduction buckets.
            recipe = DelayedScaling(amax_history_len=4 + device)
            model = te.Linear(128, 128, params_dtype=torch.bfloat16)
            x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            with te.autocast(recipe=recipe):
                model(x).float().square().mean().backward()
            assert key in model.fp8_meta
            states.append((model, x, recipe))
    # A later model's autocast also updates earlier reduction buckets. Snapshot
    # both histories at the actual restart boundary, after both warmup passes.
    histories = [model.fp8_meta["scaling_fwd"].amax_history.clone() for model, _, _ in states]
    te.release_cuda_state()
    for (model, _, _), history in zip(states, histories):
        assert key not in model.fp8_meta
        torch.testing.assert_close(model.fp8_meta["scaling_fwd"].amax_history, history)
    for device, (model, x, recipe) in enumerate(states):
        with torch.cuda.device(device):
            model.zero_grad(set_to_none=True)
            x.grad = None
            with te.autocast(recipe=recipe):
                output = model(x)
            output.float().square().mean().backward()
            assert key in model.fp8_meta
            assert torch.isfinite(x.grad).all()
            assert torch.isfinite(model.weight.grad).all()
    te.release_cuda_state()
