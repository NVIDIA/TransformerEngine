# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed worker for mixed context-parallel attention tests."""

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformer_engine.pytorch.attention import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import mixed_cp

GROUP = None


@pytest.fixture(scope="module", autouse=True)
def distributed():
    """Use separate CP groups to exercise attention heads partitioned by TP."""
    global GROUP
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    tp_size = int(os.environ.get("TEST_TP_SIZE", "1"))
    cp_size = dist.get_world_size() // tp_size
    for tp in range(tp_size):
        ranks = list(range(tp * cp_size, (tp + 1) * cp_size))
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            GROUP = group
    yield
    dist.destroy_process_group()


def local_slice(tensor):
    """Select the two balanced sequence chunks owned by this CP rank."""
    chunks = tensor.chunk(2 * dist.get_world_size(GROUP), dim=0)
    rank = dist.get_rank(GROUP)
    return torch.cat((chunks[rank], chunks[-1 - rank]), dim=0).contiguous()


def make_case(
    batch=1, head_dim=192, dtype=torch.bfloat16, *, kv_heads=None, mask="causal", dropout=0
):
    """Return local attention inputs and an independent full-sequence reference."""
    torch.manual_seed(2026)
    heads = 16 // int(os.environ.get("TEST_TP_SIZE", "1"))
    full = [
        torch.randn(1024, batch, h, d, device="cuda", dtype=dtype)
        for h, d in ((heads, head_dim), (kv_heads or heads, head_dim), (kv_heads or heads, 128))
    ]
    gradient = torch.randn(1024, batch, heads, 128, device="cuda", dtype=dtype)
    inputs = [local_slice(t).requires_grad_() for t in full]
    cp_size = dist.get_world_size(GROUP)
    first_rank = dist.get_rank() // cp_size * cp_size
    module = (
        DotProductAttention(
            num_attention_heads=heads,
            num_gqa_groups=kv_heads or heads,
            kv_channels=(head_dim, 128),
            attention_dropout=dropout,
            qkv_format="sbhd",
            attn_mask_type=mask,
            softmax_scale=0.083,
            cp_group=GROUP,
            cp_global_ranks=list(range(first_rank, first_rank + cp_size)),
            cp_stream=torch.cuda.Stream(),
            cp_comm_type="p2p",
        )
        .cuda()
        .train()
    )
    return module, inputs, local_slice(gradient).flatten(-2), full, gradient


def run(module, inputs, gradient):
    for tensor in inputs:
        tensor.grad = None
    output = module(*inputs)
    output.backward(gradient)
    torch.cuda.synchronize()
    return output.detach().clone(), [t.grad.detach().clone() for t in inputs]


def check_gradients(actual, expected):
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value.float(), reference.float(), rtol=0.04, atol=0.025)
        relative_l2 = (value.float() - reference.float()).norm() / reference.float().norm()
        assert relative_l2 < 0.008, relative_l2


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("head_dim", [128, 192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_output_and_gradients(monkeypatch, batch, head_dim, dtype):
    module, inputs, gradient, full, full_gradient = make_case(batch, head_dim, dtype)
    monkeypatch.setenv("NVTE_FUSED_ATTN_CP_USE_FAv4_BWD", "0")
    baseline = run(module, inputs, gradient)
    calls = []
    backward = mixed_cp._backward

    def observed(*args):
        calls.append(True)
        return backward(*args)

    monkeypatch.setattr(mixed_cp, "_backward", observed)
    monkeypatch.setenv("NVTE_FUSED_ATTN_CP_USE_FAv4_BWD", "1")
    actual = run(module, inputs, gradient)
    assert len(calls) == 1
    torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)
    check_gradients(actual[1], baseline[1])

    # Compare all elements against full-sequence FP32 math attention as well.
    reference_inputs = [t.float().detach().requires_grad_() for t in full]
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(
            *(t.permute(1, 2, 0, 3) for t in reference_inputs),
            is_causal=True,
            scale=0.083,
        ).permute(2, 0, 1, 3)
    output.backward(full_gradient.float())
    torch.testing.assert_close(
        actual[0].float(), local_slice(output).flatten(-2), rtol=0.04, atol=0.025
    )
    check_gradients(actual[1], [local_slice(t.grad) for t in reference_inputs])


def gather_bits(tensor):
    """Gather raw storage, including dtypes not supported by NCCL collectives."""
    wire = tensor.contiguous().view(torch.uint8)
    pieces = [torch.empty_like(wire) for _ in range(dist.get_world_size(GROUP))]
    dist.all_gather(pieces, wire, group=GROUP)
    return [piece.view(tensor.dtype).reshape(tensor.shape) for piece in pieces]


def gather_reference(tensor, *, owner_shift=0):
    """Assemble global sequence order independently of the packing implementation."""
    size, rank = dist.get_world_size(GROUP), dist.get_rank(GROUP)
    pieces = gather_bits(tensor)
    chunks = [None] * (2 * size)
    for peer, piece in enumerate(pieces):
        owner = (peer + owner_shift) % size
        chunks[owner], chunks[-1 - owner] = piece.chunk(2, dim=0)
    return torch.cat(chunks).chunk(size, dim=2)[rank].contiguous()


@pytest.mark.parametrize("batch,head_dim", [(1, 128), (2, 192)])
def test_exchange_bits(batch, head_dim):
    torch.manual_seed(78 + dist.get_rank())
    sequence, heads = 128, 16
    tensors = [
        torch.randint(
            -32768, 32767, (sequence * 2, batch, heads, width), device="cuda", dtype=torch.int16
        )[::2].view(torch.bfloat16)
        for width in (head_dim, head_dim, 128, 128, 128)
    ]
    bits = torch.randint(
        -(2**31), 2**31 - 1, (batch, heads, sequence * 2), device="cuda", dtype=torch.int32
    )[..., ::2]
    bits[0, 0, :4] = torch.tensor(
        [0, -(2**31), 0x7F800000, 0x7FC12345], device="cuda", dtype=torch.int32
    )
    actual = mixed_cp._to_heads(*tensors, bits.view(torch.float32), GROUP)
    for value, tensor, shift in zip(actual[:5], tensors, (0, 1, 1, 0, 0), strict=True):
        expected = gather_reference(tensor.view(torch.int16), owner_shift=shift)
        torch.testing.assert_close(value.view(torch.int16), expected, rtol=0, atol=0)
    expected_lse = gather_reference(bits.permute(2, 0, 1).unsqueeze(-1))
    torch.testing.assert_close(
        actual[5].view(torch.int32), expected_lse.squeeze(-1).permute(1, 2, 0), rtol=0, atol=0
    )
    returned = mixed_cp._to_sequence(*actual[:3], GROUP)
    for value, tensor in zip(returned, actual[:3], strict=True):
        parts = gather_bits(tensor)
        expected = local_slice(torch.cat(parts, dim=2))
        torch.testing.assert_close(
            value.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )


def test_exchange_large_stride():
    """Exercise offsets beyond INT_MAX without exchanging a large payload."""
    torch.manual_seed(78 + dist.get_rank())
    query = torch.empty_strided(
        (4, 1, 16, 128), (2**30, 2048, 128, 1), device="cuda", dtype=torch.bfloat16
    )
    query.view(torch.int16).copy_(
        torch.randint(-32768, 32767, query.shape, device="cuda", dtype=torch.int16)
    )
    tensors = [query] + [torch.randn_like(query.contiguous()) for _ in range(4)]
    lse = torch.randn(1, 16, 4, device="cuda", dtype=torch.float32)
    actual = mixed_cp._to_heads(*tensors, lse, GROUP)
    expected = gather_reference(query.view(torch.int16))
    torch.testing.assert_close(actual[0].view(torch.int16), expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 2])
def test_changed_input_graph_replay(monkeypatch, batch):
    module, inputs, gradient, _, _ = make_case(batch=batch)
    monkeypatch.setenv("NVTE_FUSED_ATTN_CP_USE_FAv4_BWD", "1")
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            run(module, inputs, gradient)
    torch.cuda.current_stream().wait_stream(warmup)
    for tensor in inputs:
        tensor.grad = None
    dist.barrier(group=GROUP)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module(*inputs)
        output.backward(gradient)
    for factor in (0.9, 1.1):
        with torch.no_grad():
            for tensor in inputs:
                tensor.mul_(factor)
            gradient.mul_(factor)
        reference_inputs = [t.detach().clone().requires_grad_() for t in inputs]
        monkeypatch.setenv("NVTE_FUSED_ATTN_CP_USE_FAv4_BWD", "0")
        reference = run(module, reference_inputs, gradient)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, reference[0], rtol=0, atol=0)
        check_gradients([t.grad for t in inputs], reference[1])
    graph.reset()


@pytest.mark.parametrize(
    "mode", ["disabled", "missing_fa4", "gqa", "noncausal", "dropout", "custom_lengths"]
)
def test_native_fallback(monkeypatch, mode):
    kwargs = {}
    if mode == "gqa":
        kwargs["kv_heads"] = 4
    if mode == "noncausal":
        kwargs["mask"] = "no_mask"
    if mode == "dropout":
        kwargs["dropout"] = 0.1
    module, inputs, gradient, _, _ = make_case(**kwargs)
    monkeypatch.setenv("NVTE_FUSED_ATTN_CP_USE_FAv4_BWD", "0" if mode == "disabled" else "1")
    if mode == "missing_fa4":
        from transformer_engine.pytorch.attention.dot_product_attention import backends

        monkeypatch.setattr(backends, "_flash_attn_bwd_v4", None)

    def unexpected(*args):
        raise AssertionError("Unsupported configuration selected mixed backward")

    monkeypatch.setattr(mixed_cp, "_backward", unexpected)
    if mode == "custom_lengths":
        lengths = torch.tensor([0, 1024], dtype=torch.int32, device="cuda")
        module(*inputs, cu_seqlens_q=lengths, cu_seqlens_kv=lengths).backward(gradient)
    else:
        run(module, inputs, gradient)
    assert all(torch.isfinite(t.grad).all() for t in inputs)


def test_deterministic_mode(monkeypatch):
    """Exercise the optional backward with deterministic attention enabled."""
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    test_output_and_gradients(monkeypatch, 2, 192, torch.bfloat16)


if __name__ == "__main__":
    args = [__file__, "-q", "-x", "-c", "/dev/null"]
    if os.environ.get("XML_LOG_DIR"):
        path = Path(os.environ["XML_LOG_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        args.extend(
            [
                f"--junitxml={path / ('rank' + os.environ['RANK'] + '.xml')}",
                "-o",
                f"cache_dir={path / '.pytest-cache'}",
            ]
        )
    raise SystemExit(pytest.main(args))
