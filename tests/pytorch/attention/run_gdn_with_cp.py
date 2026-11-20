# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Execute one GDN CP numerical scenario supplied by the pytest launcher."""

import argparse
import json
import os
import traceback
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

from transformer_engine.pytorch import GatedDeltaNetAttention
from test_gdn_attention import (
    _assert_rms_close,
    _gdn_reference,
    _FWD_TOL,
    _STATE_TOL,
    _BWD_TOL,
)


def _reference(inputs, cu, scale, norm):
    """Compute the independent FP64 reference for the full batch on each rank.

    The caller selects rank-local output slices. Empty documents retain their initial states.
    """
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
    name,
    dtype=torch.bfloat16,
    heads=2,
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
    dense_global=False,
    cpu_metadata=False,
    state_only=False,
):
    """Compare each local output/gradient to the corresponding full-sequence result."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if rank == 0:
        print(
            f"GDN CP case {name}: lengths={lengths}, dtype={dtype}, H={heads}, K={key_dim},"
            f" V={value_dim}, layout={layout}, norm={norm}, initial={initial}, final={final},"
            f" checkpoint={checkpoint}, last_loss={last_loss}, fallback={fallback}",
            flush=True,
        )
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(791 + (dist.get_rank(tp_group) if tp_group is not None else 0))
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
    initial_state = torch.randn(len(lengths), heads, value_dim, key_dim, device=device) * 0.1
    if not initial:
        initial_state.zero_()
    full = tuple(t.detach().requires_grad_(True) for t in (q, k, v, g, beta, initial_state))
    global_cu = torch.tensor(cu, device="cpu" if cpu_metadata else device, dtype=torch.int32)
    ref_output, ref_states = _reference(full, cu, scale, norm)
    grad_output = torch.randn_like(v)
    if last_loss:
        grad_output[: -max(1, total // size)] = 0
    grad_states = (
        torch.randn_like(initial_state) * 0.1 if final else torch.zeros_like(initial_state)
    )
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
    module_kwargs = dict(
        qkv_format=layout,
        scale=scale,
        cp_group=group,
        tp_group=tp_group,
    )
    module = GatedDeltaNetAttention(heads * tp_size, (key_dim, value_dim), **module_kwargs)
    kwargs = dict(
        cu_seqlens=global_cu if layout == "thd" or dense_global else None,
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


def _state_chaining(group, lengths, *, name, heads, key_dim, value_dim, dtype, norm, checkpoint):
    """Train two connected CP calls after inference has populated the metadata cache."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if rank == 0:
        print(
            f"GDN CP case {name}: two calls, lengths={lengths}",
            flush=True,
        )
    torch.manual_seed(932)
    tokens = sum(lengths)
    full = []
    for _ in range(2):
        q = torch.randn(tokens, heads, key_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn(tokens, heads, value_dim, device=device, dtype=dtype)
        g = -torch.rand(tokens, heads, device=device) * 0.02
        beta = torch.rand(tokens, heads, device=device) * 0.7
        full.extend(t.detach().requires_grad_(True) for t in (q, k, v, g, beta))
    initial_state = (
        torch.randn(len(lengths), heads, value_dim, key_dim, device=device) * 0.1
    ).requires_grad_(True)
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    global_cu = torch.tensor(cu, device=device, dtype=torch.int32)
    scale = key_dim**-0.5
    first_ref, first_state_ref = _reference((*full[:5], initial_state), cu, scale, norm)
    # Keep the native state connected so call 2 contributes gradients to call 1.
    second_ref, second_state_ref = _reference((*full[5:], first_state_ref), cu, scale, norm)
    # Use matching upstream gradient dtypes on both paths to avoid extra CP-only rounding.
    grad_first, grad_second = torch.randn_like(full[2]), torch.randn_like(full[7])
    grad_state = torch.randn_like(initial_state) * 0.1
    ref_loss = (first_ref * grad_first).sum() * 0.3 + (second_ref * grad_second).sum()
    ref_loss = ref_loss + (second_state_ref * grad_state).sum()
    ref_grads = torch.autograd.grad(ref_loss, (*full, initial_state))

    lo, hi = rank * tokens // size, (rank + 1) * tokens // size
    inputs = tuple(t[lo:hi].detach().requires_grad_(True) for t in full)
    local_initial_state = initial_state.detach().requires_grad_(True)
    kwargs = dict(cu_seqlens=global_cu, output_final_state=True, use_qk_l2norm_in_kernel=norm)
    with torch.inference_mode():
        warmup = GatedDeltaNetAttention(
            heads, (key_dim, value_dim), qkv_format="thd", cp_group=group
        )
        warm_output, warm_state = warmup(*inputs[:5], initial_state=local_initial_state, **kwargs)
    assert not warm_output.requires_grad and not warm_state.requires_grad
    _assert_rms_close(
        warm_output,
        first_ref[lo:hi].reshape_as(warm_output),
        _FWD_TOL[dtype],
        "inference output",
    )
    _assert_rms_close(warm_state, first_state_ref, _STATE_TOL[dtype], "inference final state")
    # Fresh layers share training metadata, but must not reuse inference-mode tensors.
    modules = [
        GatedDeltaNetAttention(heads, (key_dim, value_dim), qkv_format="thd") for _ in range(2)
    ]
    for module in modules:
        module.set_context_parallel_group(group)
    first, first_state = modules[0](*inputs[:5], initial_state=local_initial_state, **kwargs)
    second, second_state = modules[1](
        *inputs[5:], initial_state=first_state, checkpoint_core_attention=checkpoint, **kwargs
    )
    for name, actual, expected in (
        ("first output", first, first_ref[lo:hi].reshape_as(first)),
        ("second output", second, second_ref[lo:hi].reshape_as(second)),
    ):
        _assert_rms_close(actual, expected, _FWD_TOL[dtype], name)
    for actual, expected in ((first_state, first_state_ref), (second_state, second_state_ref)):
        _assert_rms_close(actual, expected, _STATE_TOL[dtype], "chained final state")
    loss = (first * grad_first[lo:hi].reshape_as(first)).sum() * 0.3
    loss = loss + (second * grad_second[lo:hi].reshape_as(second)).sum()
    # Replicated final-state losses collectively count once, as in _case.
    loss = loss + (second_state * grad_state).sum() / size
    loss.backward()
    for actual, expected in zip(inputs, ref_grads[:10]):
        _assert_rms_close(actual.grad, expected[lo:hi], _BWD_TOL[dtype], "chained token gradient")
    initial_grad = local_initial_state.grad.clone()
    dist.all_reduce(initial_grad, group=group)
    _assert_rms_close(
        initial_grad, ref_grads[10], _BWD_TOL[dtype], "chained initial-state gradient"
    )
    dist.barrier(group=group)


def run_gdn_with_cp(*, name, lengths, kind="attention", dtype="bf16", **kwargs):
    """Run the configured attention, state-chaining or TP+CP correctness checks."""
    group = dist.group.WORLD
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    if kind == "state_chaining":
        _state_chaining(group, lengths, name=name, dtype=dtype, **kwargs)
    elif kind == "tp_cp":
        size, rank = dist.get_world_size(), dist.get_rank()
        if size < 4 or size % 2:
            raise ValueError("TP+CP requires an even number of at least four GPUs")
        cp_groups = [dist.new_group([r, r + 1]) for r in range(0, size, 2)]
        tp_groups = [dist.new_group(list(range(r, size, 2))) for r in (0, 1)]
        try:
            _case(
                cp_groups[rank // 2],
                lengths,
                name=name,
                dtype=dtype,
                tp_group=tp_groups[rank % 2],
                **kwargs,
            )
        finally:
            for subgroup in (*cp_groups, *tp_groups):
                if subgroup != dist.GroupMember.NON_GROUP_MEMBER:
                    dist.destroy_process_group(subgroup)
    elif kind == "attention":
        _case(group, lengths, name=name, dtype=dtype, **kwargs)
    else:
        raise ValueError(f"Unknown GDN CP scenario kind: {kind}")


def main():
    """Initialize this scenario's workers and release them when its checks finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=json.loads, required=True)
    config = parser.parse_args().config
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        run_gdn_with_cp(**config)
    except BaseException:  # pylint: disable=broad-exception-caught
        # Exit promptly so torchrun can stop peers blocked in a collective.
        traceback.print_exc()
        os._exit(1)
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
