# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention scale snapshots must participate in saved-tensor lifetime management."""

from types import SimpleNamespace
from contextlib import nullcontext
from functools import partial
import os
import subprocess
import sys

import pytest
import torch

from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling
from transformer_engine.pytorch.attention.dot_product_attention.backends import FusedAttnBwdArgs
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    _prepare_backward_quantizers,
    _restore_backward_quantizers,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    AttentionParams,
    get_attention_backend,
)
from transformer_engine.pytorch.quantized_tensor import prepare_for_saving, restore_from_saved
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
)
import transformer_engine_torch as tex


def _quantizer(value):
    return Float8Quantizer(
        torch.full((1,), value, device="cuda", dtype=torch.float32),
        torch.zeros(1, device="cuda"),
        tex.DType.kFloat8E4M3,
    )


@pytest.mark.parametrize("fp8", (False, True))
def test_cp_backward_quantizers_restore_saved_scale_storage(fp8):
    quantizers = tuple(_quantizer(value) for value in (2.0, 4.0, 8.0))
    ctx = SimpleNamespace(fp8=fp8)
    scales = _prepare_backward_quantizers(ctx, quantizers)
    if not fp8:
        assert scales == [None, None, None]
        assert ctx.S_quantizer is quantizers[2]
        return

    # Saved-tensor hooks may restore a different storage, as the slot arena does.
    restored = [scale.clone() for scale in scales]
    for scale in scales:
        scale.zero_()
    _restore_backward_quantizers(ctx, restored)
    for name, original, scale in zip(
        ("QKV_quantizer", "O_quantizer", "S_quantizer"), quantizers, restored
    ):
        copied = getattr(ctx, name)
        assert copied is not original
        assert copied.scale is scale
        torch.testing.assert_close(copied.scale, original.scale, rtol=0, atol=0)


def test_cp_backward_quantizers_snapshot_s_scale_under_current_scaling():
    """Current scaling keeps QKV/O quantizers stateless but S still uses a delayed scale."""
    current = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device="cuda")
    s_quantizer = _quantizer(8.0)
    ctx = SimpleNamespace(fp8=True)
    scales = _prepare_backward_quantizers(ctx, (current, current, s_quantizer))
    assert scales[:2] == [None, None]
    assert ctx.S_quantizer.scale is scales[2]
    assert scales[2] is not s_quantizer.scale
    torch.testing.assert_close(scales[2], s_quantizer.scale, rtol=0, atol=0)


def test_fused_attention_backward_rebinds_saved_s_scale():
    args = FusedAttnBwdArgs(fp8=True, S_quantizer=_quantizer(8.0))
    restored_scale = args.S_quantizer.scale.clone()
    tensors, objects = prepare_for_saving(*(None,) * 18, restored_scale)
    ctx = SimpleNamespace(saved_tensors=tensors, tensor_objects=objects)
    args.S_quantizer.scale.zero_()
    args.setup_saved_tensors(ctx)
    assert args.S_quantizer.scale is restored_scale
    torch.testing.assert_close(args.S_quantizer.scale, torch.full_like(restored_scale, 8.0))


@pytest.mark.parametrize("offload", (False, True))
def test_legacy_offload_preserves_quantized_wrapper_buffers(offload, monkeypatch):
    from transformer_engine.pytorch import cpu_offload
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    monkeypatch.setattr(cpu_offload, "NVTE_CPU_OFFLOAD_V1", True)
    monkeypatch.setattr(cpu_offload.v1_code_path, "CPUOffloadEnabled", offload)
    quantized = _quantizer(2.0)(torch.randn(8, 128, dtype=torch.bfloat16, device="cuda"))
    assert isinstance(quantized, Float8Tensor)
    data, scale_inv = quantized._data, quantized._scale_inv
    expected = quantized.dequantize()
    tensors, objects = prepare_for_saving(quantized)
    assert quantized._data is data
    assert quantized._scale_inv is scale_inv
    if not offload:
        assert tensors == [quantized]
        assert objects == [None]
        return

    assert all(tensor is None or type(tensor) is torch.Tensor for tensor in tensors)
    # A nominal-dtype CPU copy of the wrapper loses its FP8 representation.
    restored = restore_from_saved(
        objects,
        [tensor.cpu().clone().cuda() if tensor is not None else None for tensor in tensors],
    )[0]
    assert isinstance(restored, Float8Tensor)
    assert restored._data.data_ptr() != data.data_ptr()
    torch.testing.assert_close(restored._data, data, rtol=0, atol=0)
    torch.testing.assert_close(restored._scale_inv, scale_inv, rtol=0, atol=0)
    torch.testing.assert_close(restored.dequantize(), expected, rtol=0, atol=0)


def test_legacy_offload_keeps_distributed_weight_by_reference(monkeypatch):
    """A quantized DistributedWeight keeps its identity instead of being decomposed."""
    from transformer_engine.pytorch import cpu_offload

    monkeypatch.setattr(cpu_offload, "NVTE_CPU_OFFLOAD_V1", True)
    monkeypatch.setattr(cpu_offload.v1_code_path, "CPUOffloadEnabled", True)
    weight = _quantizer(2.0)(torch.randn(8, 128, dtype=torch.bfloat16, device="cuda"))
    weight.is_distributed_weight = True
    tensors, objects = prepare_for_saving(weight)
    assert tensors == [weight]
    assert restore_from_saved(objects, tensors)[0] is weight


def _unit_scale(amax, scale, fp8_max, recipe):
    return torch.ones_like(scale)


@pytest.fixture(scope="session")
def cp_groups():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (2, 4):
        pytest.skip("CP1/CP2 banks require two or four distributed ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    initialized_here = not torch.distributed.is_initialized()
    if initialized_here:
        torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    groups = []
    selected = None
    try:
        for start in range(0, world_size, 2):
            ranks = [start, start + 1]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                groups.append(group)
                selected = group, ranks
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        torch.distributed.barrier(group=selected[0], device_ids=[torch.cuda.current_device()])
        yield selected
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            for group in groups:
                torch.distributed.destroy_process_group(group)
            if initialized_here:
                torch.distributed.destroy_process_group()


@pytest.fixture
def cp_pair(cp_comm_type, qkv_format, request):
    if qkv_format == "thd" and (
        torch.backends.cudnn.version() < 92300 or torch.cuda.get_device_capability()[0] < 10
    ):
        pytest.skip("FP8 THD backward requires cuDNN >= 9.23 and SM100+")
    if cp_comm_type is None:
        return None, None
    return request.getfixturevalue("cp_groups")


@pytest.mark.parametrize("cp_comm_type", [None, "p2p", "a2a", "all_gather"])
@pytest.mark.parametrize("qkv_format", ["sbhd", "thd"])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("recipe", ["delayed", "current"])
def test_fp8_attention_slot_replay_matches_eager_with_overlapping_frames(
    recipe, cp_comm_type, qkv_format, offload, cp_pair, monkeypatch
):
    from transformer_engine.pytorch import (
        DotProductAttention,
        autocast,
        is_fp8_available,
        make_graphed_callables,
    )
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    from transformer_engine.pytorch import graph as te_graph

    if not is_fp8_available():
        pytest.skip("FP8 is not supported on this device")
    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as dpa_module,
    )

    FP8GlobalStateManager.reset()
    # Unit scale updates keep every delayed scale (S and dP under current scaling too) fixed
    # between the eager reference and the replay; the test sets the forward scales itself.
    unit_delayed = partial(DelayedScaling, scaling_factor_compute_algo=_unit_scale)
    monkeypatch.setattr(dpa_module, "DelayedScaling", unit_delayed)
    if recipe == "delayed":
        fp8_recipe = unit_delayed(fp8_dpa=True, amax_history_len=1)
    else:
        fp8_recipe = Float8CurrentScaling(fp8_dpa=True)
    cp_group, cp_ranks = cp_pair
    offload_streams = (
        dict(d2h_stream=torch.cuda.Stream(), h2d_stream=torch.cuda.Stream()) if offload else {}
    )

    class Attention(torch.nn.Module):
        def __init__(self, group=None, ranks=None):
            super().__init__()
            self.attention = DotProductAttention(
                4,
                128,
                attention_dropout=0.0,
                attn_mask_type="padding_causal" if qkv_format == "thd" else "causal",
                qkv_format=qkv_format,
                cp_group=group,
                cp_global_ranks=ranks,
                cp_stream=torch.cuda.Stream() if group is not None else None,
                cp_comm_type=cp_comm_type or "p2p",
            )
            self.max_seqlen = 128 * (group.size() if group is not None else 1)

        def forward(self, inp, cu=None):
            kwargs = {}
            if qkv_format == "thd":
                kwargs = dict(
                    cu_seqlens_q=cu,
                    cu_seqlens_kv=cu,
                    cu_seqlens_q_padded=cu,
                    cu_seqlens_kv_padded=cu,
                    max_seqlen_q=self.max_seqlen,
                    max_seqlen_kv=self.max_seqlen,
                    pad_between_seqs=True,
                )
            context = (
                te_graph._graph_memory_offload_context(
                    lambda tensor: tensor.is_cuda, **offload_streams
                )
                if offload
                else nullcontext()
            )
            with context:
                return self.attention(*inp.unbind(0), **kwargs)

    modules = (Attention().cuda(), Attention(cp_group, cp_ranks).cuda())
    for module in modules:
        size = module.max_seqlen // 128
        params = AttentionParams(
            qkv_layout="_".join([qkv_format] * 3),
            batch_size=2 if qkv_format == "thd" else 1,
            num_heads=4,
            num_gqa_groups=4,
            head_dim_qk=128,
            head_dim_v=128,
            max_seqlen_q=module.max_seqlen,
            max_seqlen_kv=module.max_seqlen,
            num_tokens_q=128 if qkv_format == "thd" else 0,
            num_tokens_kv=128 if qkv_format == "thd" else 0,
            attn_mask_type=module.attention.attn_mask_type,
            window_size=(-1, 0),
            bottom_right_diagonal=False,
            core_attention_bias_requires_grad=False,
            pad_between_seqs=qkv_format == "thd",
            context_parallel=size > 1,
            cp_size=size,
            cp_comm_type=cp_comm_type or "p2p",
            fp8=True,
            fp8_meta={"recipe": fp8_recipe},
            cuda_graph=True,
            softmax_scale=128**-0.5,
        )
        if not get_attention_backend(params)[2]:
            pytest.skip(f"cuDNN does not support FP8 {qkv_format} CP{size} {cp_comm_type}")
    shape = (3, 128, 4, 128) if qkv_format == "thd" else (3, 128, 1, 4, 128)

    def input_args(inp, branch, microbatch=0):
        if qkv_format != "thd":
            return (inp,)
        total = modules[branch].max_seqlen
        split = total // (2 if microbatch % 2 == 0 else 4)
        cu = torch.tensor([0, split, total], device="cuda", dtype=torch.int32)
        return inp, cu

    samples = tuple(
        input_args(
            torch.ones(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True), i // 2
        )
        for i in range(4)
    )
    graphs = make_graphed_callables(
        modules,
        samples,
        enabled=True,
        recipe=fp8_recipe,
        num_warmup_iters=3,
        _order=[1, 1, -1, -1, 2, 2, -2, -2],
        _num_layers_per_chunk=[1, 1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=tuple(
            (slot, slot, branch, 0, 0, branch, slot) for branch in range(2) for slot in range(2)
        ),
    )
    pending = []

    def finish(frame):
        inp, output, expected_output, expected_grad = frame
        output.sum().backward()
        torch.testing.assert_close(output, expected_output, rtol=1e-2, atol=1e-4)
        torch.testing.assert_close(inp.grad, expected_grad, rtol=1e-2, atol=1e-4)

    try:
        # Slots alternate, so frames overlap within each bank (slot wrap) and across banks.
        for microbatch, branch in enumerate((1, 1, 0, 0, 1, 1)):
            if len(pending) == 2:
                finish(pending.pop(0))
            inp = torch.randn_like(samples[0][0]).requires_grad_()
            reference = inp.detach().clone().requires_grad_()
            # Distinct snapshots expose cross-frame scale aliasing even with fixed amax updates.
            metadata = modules[branch].attention.fp8_meta
            scale = (1.0, 2.0, 4.0, 0.5, 8.0, 2.0)[microbatch]
            metadata["scaling_fwd"].scale.fill_(scale)
            args = input_args(inp, branch, microbatch)
            with autocast(enabled=True, recipe=fp8_recipe):
                expected = modules[branch](reference, *args[1:])
            expected.sum().backward()
            metadata["scaling_fwd"].scale.fill_(scale)
            with autocast(enabled=True, recipe=fp8_recipe):
                output = graphs[branch * 2 + microbatch % 2](*args)
            pending.append((inp, output, expected.detach().clone(), reference.grad.clone()))
        for frame in pending:
            finish(frame)
    finally:
        for graph in graphs:
            graph.reset()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) > 1 or torch.cuda.device_count() < 2,
    reason="Launches its own two-rank run of the CP cases",
)
def test_cp_cases_with_two_ranks():
    """Run the CP cases, which a single-process session skips, under torchrun."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "pytest",
        "-q",
        __file__,
        "-k",
        "p2p or a2a or all_gather",
    ]
    assert subprocess.run(command, env=os.environ, check=False).returncode == 0
