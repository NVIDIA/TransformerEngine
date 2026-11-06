# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed GDN CP numerical checks; launch with torch.distributed.run."""

import importlib.util
import os
import traceback
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from transformer_engine.pytorch import GatedDeltaNetAttention
from transformer_engine.pytorch.attention.linear_attention import context_parallel as cp
from test_gdn_attention import (
    _assert_rms_close,
    _gdn_reference,
    _FWD_TOL,
    _STATE_TOL,
    _BWD_TOL,
)


def _reference(inputs, cu, scale, norm):
    """Independent FP64 recurrence, including unchanged states for empty documents."""
    q, k, v, g, beta, initial = inputs
    outputs, states = [], []
    if norm:
        q, k = F.normalize(q.double(), dim=-1), F.normalize(k.double(), dim=-1)
    for doc, (start, end) in enumerate(zip(cu, cu[1:])):
        if start == end:
            states.append(initial[doc].double())
            continue
        out, state = _gdn_reference(
            q[start:end].unsqueeze(0),
            k[start:end].unsqueeze(0),
            v[start:end].unsqueeze(0),
            g[start:end].unsqueeze(0),
            beta[start:end].unsqueeze(0),
            scale=scale,
            initial_state=initial[doc : doc + 1],
            cu_seqlens=torch.tensor([0, end - start], device=q.device, dtype=torch.int32),
        )
        outputs.append(out[0])
        states.append(state[0])
    return (torch.cat(outputs) if outputs else v.double() * 0), torch.stack(states)


def _case(
    group,
    lengths,
    *,
    dtype=torch.bfloat16,
    key_dim=64,
    value_dim=64,
    layout="thd",
    norm=False,
    initial=True,
    final=True,
    checkpoint=False,
    scale=0.2,
    last_loss=False,
    fallback=False,
    tp_group=None,
    independent=False,
    dense_global=False,
    state_only=False,
):
    """Compare each local output/gradient to the corresponding full-sequence result."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if rank == 0:
        print(
            f"GDN CP case: lengths={lengths}, dtype={dtype}, K={key_dim}, V={value_dim},"
            f" layout={layout}, norm={norm}, initial={initial}, final={final},"
            f" checkpoint={checkpoint}, last_loss={last_loss}, fallback={fallback},"
            f" independent={independent}",
            flush=True,
        )
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(791 + (dist.get_rank(tp_group) if tp_group is not None else 0))
    heads = 2
    total = sum(lengths)
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    q = torch.randn(total, heads, key_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    if not norm:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    v = torch.randn(total, heads, value_dim, device=device, dtype=dtype)
    g = -torch.rand(total, heads, device=device) * 0.02
    beta = torch.rand(total, heads, device=device) * 0.7
    seed = torch.randn(len(lengths), heads, value_dim, key_dim, device=device) * 0.1
    if not initial:
        seed.zero_()
    full = tuple(t.detach().requires_grad_(True) for t in (q, k, v, g, beta, seed))
    global_cu = torch.tensor(cu, device=device, dtype=torch.int32)
    if independent or total == 0:
        ref_output, ref_states = _reference(full, cu, scale, norm)
    else:
        reference = GatedDeltaNetAttention(
            heads, (key_dim, value_dim), qkv_format="thd", scale=scale
        )
        ref_output, ref_states = reference(
            *full[:5],
            cu_seqlens=global_cu,
            initial_state=full[5] if initial else None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=norm,
        )
        ref_output = ref_output.reshape(total, heads, value_dim)
    grad_output = torch.randn_like(v)
    if last_loss:
        grad_output[: -max(1, total // size)] = 0
    grad_states = torch.randn_like(seed) * 0.1 if final else torch.zeros_like(seed)
    loss = (ref_states * grad_states).sum()
    if not state_only:
        loss = loss + (ref_output * grad_output).sum()
    ref_grads = torch.autograd.grad(loss, full, allow_unused=True)
    ref_grads = tuple(torch.zeros_like(t) if d is None else d for t, d in zip(full, ref_grads))

    def local(tensor):
        if layout == "thd":
            return tensor[rank * total // size : (rank + 1) * total // size]
        batch, length = len(lengths), lengths[0]
        tensor = tensor.reshape(batch, length, *tensor.shape[1:])
        tensor = tensor[:, rank * length // size : (rank + 1) * length // size]
        return tensor.transpose(0, 1) if layout == "sbhd" else tensor

    inputs = tuple(local(t).detach().requires_grad_(True) for t in full[:5])
    state = full[5].detach().requires_grad_(True) if initial else None
    tp_size = 1 if tp_group is None else dist.get_world_size(tp_group)
    module = GatedDeltaNetAttention(
        heads * tp_size,
        (key_dim, value_dim),
        qkv_format=layout,
        scale=scale,
        cp_group=group,
        tp_group=tp_group,
    )
    kwargs = dict(
        global_cu_seqlens=global_cu if layout == "thd" or dense_global else None,
        initial_state=state,
        output_final_state=final,
        checkpoint_core_attention=checkpoint,
        use_qk_l2norm_in_kernel=norm,
    )
    summary_patch = (
        patch.object(
            module.gdn_attention, "_summary_op", side_effect=ImportError("Missing summary")
        )
        if fallback
        else nullcontext()
    )
    with summary_patch:
        result = module(*inputs, **kwargs)
        output, states = result if final else (result, None)
        expected = local(ref_output).reshape_as(output)
        if output.numel():
            _assert_rms_close(output, expected, _FWD_TOL[dtype], "CP output")
        if final:
            _assert_rms_close(states, ref_states, _STATE_TOL[dtype], "CP final states")
        local_loss = 0 if state_only else (output * local(grad_output).reshape_as(output)).sum()
        if final:
            local_loss = local_loss + (states * grad_states).sum() / size
        local_loss.backward()
    for name, tensor, grad in zip(("q", "k", "v", "g", "beta"), inputs, ref_grads[:5]):
        if tensor.numel():
            expected = local(grad)
            if expected.count_nonzero() == 0:
                torch.testing.assert_close(tensor.grad, expected, atol=1e-6, rtol=0)
            else:
                _assert_rms_close(tensor.grad, expected, _BWD_TOL[dtype], f"CP d{name}")
    if initial:
        initial_grad = state.grad.detach().clone()
        dist.all_reduce(initial_grad, group=group)
        _assert_rms_close(initial_grad, ref_grads[5], _BWD_TOL[dtype], "CP initial-state gradient")
    dist.barrier(group=group)


def _invalid_metadata(group):
    """Reject incompatible packed metadata before entering a collective."""
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    size = dist.get_world_size(group)
    q = torch.zeros(4, 2, 64, device=device, dtype=torch.bfloat16)
    gate = torch.zeros(4, 2, device=device)
    module = GatedDeltaNetAttention(2, 64, qkv_format="thd", cp_group=group)
    invalid = (
        {},
        {"global_cu_seqlens": torch.tensor([0, 4 * size], dtype=torch.int64)},
        {"global_cu_seqlens": torch.tensor([0, 4 * size, 2 * size], dtype=torch.int32)},
        {"global_cu_seqlens": torch.tensor([0, 5 * size], dtype=torch.int32)},
        {
            "global_cu_seqlens": torch.tensor([0, 4 * size], dtype=torch.int32),
            "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32, device=device),
        },
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            module(q, q, q, g=gate, beta=gate, **kwargs)
    dist.barrier(group=group)


def _inference_and_deferred_group(group):
    """Default scale, deferred configuration, and metadata created in inference mode."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(998)
    q, k = (
        F.normalize(torch.randn(35, 2, 64, device=device, dtype=torch.bfloat16), dim=-1)
        for _ in range(2)
    )
    v = torch.randn_like(q)
    g, beta = -torch.rand(35, 2, device=device) * 0.02, torch.rand(35, 2, device=device) * 0.7
    reference = GatedDeltaNetAttention(2, 64, qkv_format="thd")
    module = GatedDeltaNetAttention(2, 64, qkv_format="thd")
    module.set_context_parallel_group(group)
    q, k, v, g, beta = (t.requires_grad_() for t in (q, k, v, g, beta))
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            global_cu = torch.tensor([0, 17, 35], device=device, dtype=torch.int32)
            expected, expected_states = reference(
                q, k, v, g, beta, cu_seqlens=global_cu, output_final_state=True
            )
            lo, hi = rank * 35 // size, (rank + 1) * 35 // size
            output, states = module(
                q[lo:hi],
                k[lo:hi],
                v[lo:hi],
                g[lo:hi],
                beta[lo:hi],
                global_cu_seqlens=global_cu,
                output_final_state=True,
            )
            assert not output.requires_grad and not states.requires_grad
            _assert_rms_close(output, expected[lo:hi], _FWD_TOL[q.dtype], "inference output")
            _assert_rms_close(states, expected_states, _STATE_TOL[q.dtype], "inference states")
    dist.barrier(group=group)


def _state_chaining(group):
    """Backpropagate through two CP calls joined by their replicated final states."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(932)
    tokens, heads, dim = 35, 2, 64
    full = []
    for _ in range(2):
        q, k = (
            F.normalize(
                torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16), dim=-1
            )
            for _ in range(2)
        )
        v = torch.randn_like(q)
        g = -torch.rand(tokens, heads, device=device) * 0.02
        beta = torch.rand(tokens, heads, device=device) * 0.7
        full.extend(t.detach().requires_grad_(True) for t in (q, k, v, g, beta))
    seed = (torch.randn(2, heads, dim, dim, device=device) * 0.1).requires_grad_(True)
    cu = torch.tensor([0, 17, 35], device=device, dtype=torch.int32)
    reference = GatedDeltaNetAttention(heads, dim, qkv_format="thd")
    first, state = reference(*full[:5], cu_seqlens=cu, initial_state=seed, output_final_state=True)
    second = reference(*full[5:], cu_seqlens=cu, initial_state=state)
    grad_first, grad_second = torch.randn_like(first), torch.randn_like(second)
    ((first * grad_first).sum() * 0.3 + (second * grad_second).sum()).backward()
    lo, hi = rank * tokens // size, (rank + 1) * tokens // size
    inputs = tuple(t[lo:hi].detach().requires_grad_(True) for t in full)
    local_seed = seed.detach().requires_grad_(True)
    module = GatedDeltaNetAttention(heads, dim, qkv_format="thd", cp_group=group)
    local_first, state = module(
        *inputs[:5], global_cu_seqlens=cu, initial_state=local_seed, output_final_state=True
    )
    local_second = module(*inputs[5:], global_cu_seqlens=cu, initial_state=state)
    loss = (local_first * grad_first[lo:hi]).sum() * 0.3
    loss = loss + (local_second * grad_second[lo:hi]).sum()
    for retain in (True, False):
        loss.backward(retain_graph=retain)
        for actual, expected in zip(inputs, full):
            _assert_rms_close(
                actual.grad,
                expected.grad[lo:hi],
                _BWD_TOL[torch.bfloat16],
                "chained token gradient",
            )
        initial_grad = local_seed.grad.clone()
        dist.all_reduce(initial_grad, group=group)
        _assert_rms_close(
            initial_grad, seed.grad, _BWD_TOL[torch.bfloat16], "chained initial-state gradient"
        )
        for tensor in (*inputs, local_seed):
            tensor.grad = None
    dist.barrier(group=group)


def main():
    """Run one persistent process group to amortize FE compilation across cases."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    size = dist.get_world_size(group)
    cases = 0
    try:
        for lengths in (
            [8, 4, 12],
            [16, 4, 4],
            [7, 2, 4, 5, 14],
            [6, 6, 6, 6],
            [0, 17, 0, 18, 0],
            [1, 0, 1],
            [0, 0],
        ):
            _case(group, lengths)
            cases += 1
        for dtype in (torch.float16, torch.bfloat16):
            for norm in (False, True):
                _case(
                    group,
                    [size * 65 + 3, 7],
                    dtype=dtype,
                    value_dim=128,
                    norm=norm,
                    independent=True,
                )
                cases += 1
        _case(
            group,
            [size * 65 + 3, 7],
            dtype=torch.float16,
            key_dim=128,
            value_dim=64,
            independent=True,
        )
        cases += 1
        for layout, dtype in (("bshd", torch.bfloat16), ("sbhd", torch.float16)):
            _case(group, [size * 65] * 2, dtype=dtype, layout=layout, norm=True)
            cases += 1
        for layout in ("bshd", "sbhd"):
            _case(group, [size * 17 + 3] * 2, layout=layout, dense_global=True)
            cases += 1
        _case(group, [1, 1], layout="bshd", dense_global=True, checkpoint=True)
        cases += 1
        for initial, final, checkpoint, last_loss in (
            (False, False, False, True),
            (True, False, False, True),
            (False, True, True, True),
            (True, True, True, False),
        ):
            _case(
                group,
                [size * 67],
                initial=initial,
                final=final,
                checkpoint=checkpoint,
                last_loss=last_loss,
            )
            cases += 1
        _case(group, [size * 65 + 1, 7], fallback=True, norm=True)
        cases += 1
        _case(group, [size * 65 + 1, 7], state_only=True)
        cases += 1
        # Irregular features are zero-extended internally because ordinary FE
        # state gradients are incorrect for some unpadded cuTile geometries.
        _case(group, [size * 65 + 1, 7], key_dim=32, value_dim=96, norm=True, independent=True)
        cases += 1
        # Larger cuTile geometries have no dedicated FROST summary engine;
        # exercise the real capability fallback, rather than a mocked error.
        if importlib.util.find_spec("cuda.tile") is not None:
            _case(group, [131, 7], key_dim=192, value_dim=160, independent=True)
            cases += 1
        _inference_and_deferred_group(group)
        cases += 1
        _state_chaining(group)
        cases += 1
        _invalid_metadata(group)
        cases += 1
        # Independent complete documents need no state exchange or prefix work.
        with patch.object(
            cp, "_gather_summaries", side_effect=AssertionError("Unexpected exchange")
        ), patch.object(
            cp.tex,
            "linear_attention_affine_prefix",
            side_effect=AssertionError("Unexpected prefix"),
        ):
            _case(group, [8] * size, initial=False, final=False)
        cases += 1
        # A size-one group follows the ordinary local path, even while WORLD
        # contains several ranks. Every process creates groups in the same order.
        singletons = [dist.new_group([r]) for r in range(size)]
        _case(singletons[dist.get_rank(group)], [17, 13])
        cases += 1
        if size == 4:
            cp_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
            tp_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
            rank = dist.get_rank()
            _case(cp_groups[rank // 2], [139, 13], tp_group=tp_groups[rank % 2])
            cases += 1
        if dist.get_rank() == 0:
            print(
                f"GDN CP: {cases} correctness cases passed with CP={size}",
                flush=True,
            )
    except BaseException:  # pylint: disable=broad-exception-caught
        # A failed rank must exit before a peer waits forever for its next
        # collective. NCCL's orderly shutdown can wait on that missing peer.
        traceback.print_exc()
        os._exit(1)
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
