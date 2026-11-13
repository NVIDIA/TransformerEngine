# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Contiguous linear-attention CP: fragments, collectives and state gradients.

Flavor adapters supply local attention and affine summary operations. The shared
CUDA operation composes states in document order.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

import transformer_engine_torch as tex
from transformer_engine.common.linear_attention import (
    LinearAttentionCPPartition,
    build_linear_attention_partition,
)
from .base import _to_thd, _validate_cu_seqlens


@dataclass
class _DevicePartition:
    """Cached device metadata; the host partition has no framework dependency."""

    host: LinearAttentionCPPartition
    cu: torch.Tensor
    forward: torch.Tensor
    backward: torch.Tensor
    forward_zero: torch.Tensor
    backward_zero: torch.Tensor
    docs: torch.Tensor
    outgoing: torch.Tensor
    incoming: torch.Tensor
    terminal: torch.Tensor
    seed_fragments: torch.Tensor
    seed_docs: torch.Tensor
    empty_docs: torch.Tensor
    outgoing_cu: torch.Tensor
    incoming_cu: torch.Tensor


def _device_partition(host, device):
    """Materialize small metadata tensors once, outside the steady-state kernels."""

    def indices(values):
        return torch.tensor(values, device=device, dtype=torch.int64)

    def cumulative(selected):
        offsets = [0]
        for index in selected:
            frag = host.fragments[index]
            offsets.append(offsets[-1] + frag.end - frag.start)
        return torch.tensor(offsets, device=device, dtype=torch.int32)

    first = [i for i, f in enumerate(host.fragments) if f.first_rank == host.rank]
    forward = torch.tensor(host.forward_descriptors, device=device, dtype=torch.int32).reshape(
        -1, 4
    )
    backward = torch.tensor(host.backward_descriptors, device=device, dtype=torch.int32).reshape(
        -1, 4
    )
    forward_zero, backward_zero = forward.clone(), backward.clone()
    forward_zero[:, 0] = -1
    backward_zero[:, 0] = -1
    return _DevicePartition(
        host,
        torch.tensor(host.local_cu_seqlens, device=device, dtype=torch.int32),
        forward,
        backward,
        forward_zero,
        backward_zero,
        indices([f.document for f in host.fragments]),
        indices(host.outgoing),
        indices(host.incoming),
        indices(host.terminal),
        indices(first),
        indices([host.fragments[i].document for i in first]),
        indices(host.empty_owned),
        cumulative(host.outgoing),
        cumulative(host.incoming),
    )


def _gather_summaries(additive, transition, group):
    """One fixed-size all-gather for both summary tensors, in CP group rank order."""
    slots, heads, value_dim, key_dim = additive.shape
    size = dist.get_world_size(group)
    packed = torch.cat((additive.flatten(), transition.flatten()))
    gathered = torch.empty(size * packed.numel(), device=packed.device, dtype=packed.dtype)
    dist.all_gather_into_tensor(gathered, packed, group=group)
    gathered = gathered.view(size, -1)
    split = additive.numel()
    return (
        gathered[:, :split].contiguous().view(size * slots, heads, value_dim, key_dim),
        gathered[:, split:].contiguous().view(size * slots, heads, key_dim, key_dim),
    )


def _summary_buffers(seed, slots):
    """Unused slots are identity recipes, including ranks owning zero tokens."""
    _, heads, value_dim, key_dim = seed.shape
    additive = seed.new_zeros((slots, heads, value_dim, key_dim))
    transition = torch.eye(key_dim, device=seed.device, dtype=seed.dtype)
    return additive, transition.expand(slots, heads, key_dim, key_dim).contiguous()


def _selected_span(plan, selected):
    """Selected spans are consecutive: all dense spans, or one packed edge span."""
    cu = plan.host.local_cu_seqlens
    return slice(cu[selected[0]], cu[selected[-1] + 1])


def _slot_indices(plan, indices):
    """Dense slots are document IDs; packed partitions have only one edge slot."""
    if plan.host.slots == 1:
        return torch.zeros_like(indices)
    return plan.docs.index_select(0, indices)


def _local_attention(adapter, tensors, gate_names, cu, state, options, *, final=True):
    """Call the flavor adapter with packed local inputs."""
    return adapter._call_op(
        *tensors[:3],
        dict(zip(gate_names, tensors[3:])),
        cu,
        initial_state=state,
        output_final_state=final,
        **options,
    )


class _LinearAttentionContextParallel(torch.autograd.Function):
    """A distributed first-order VJP, with full local attention retained for FE backward.

    Summary APIs intentionally do not have autograd registrations. Differentiating
    an all-gather of summaries therefore cannot recover cross-rank gradients.
    The reverse summary exchange supplies the missing final-state adjoints to FE.
    """

    @staticmethod
    def forward(
        ctx, adapter, plan, group, options, final, gate_names, track_grad, initial_state, *tensors
    ):
        """Establish starting states, then retain the local attention graph."""
        ctx.set_materialize_grads(False)
        q = tensors[0]
        k = tensors[1]
        v = tensors[2]
        state_shape = (len(plan.host.documents) - 1, q.shape[1], v.shape[-1], k.shape[-1])
        seed = (
            q.new_zeros((1, *state_shape[1:]), dtype=torch.float32)
            if initial_state is None
            else initial_state
        )
        additive, transition = _summary_buffers(seed, plan.host.slots)
        if plan.host.outgoing:
            span = _selected_span(plan, plan.host.outgoing)
            selected = tuple(t[span] for t in tensors)
            h, t = adapter._cp_forward_summary(
                *selected[:3], dict(zip(gate_names, selected[3:])), plan.outgoing_cu, **options
            )
            slots = _slot_indices(plan, plan.outgoing)
            additive.index_copy_(0, slots, h)
            transition.index_copy_(0, slots, t)
        if plan.host.communicates:
            additive, transition = _gather_summaries(additive, transition, group)
        starting = tex.linear_attention_affine_prefix(
            seed.contiguous(),
            additive,
            transition,
            plan.forward if initial_state is not None else plan.forward_zero,
        )
        # Retain FE's local graph only for training. This Function owns the
        # distributed VJP and the outer state gradients.
        with torch.set_grad_enabled(track_grad):
            inputs = tuple(t.detach().requires_grad_(track_grad) for t in (*tensors, starting))
            if plan.host.fragments:
                output, states = _local_attention(
                    adapter, inputs[:-1], gate_names, plan.cu, inputs[-1], options
                )
            else:
                output = v.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
                states = starting
        terminal = seed.new_zeros(state_shape) if final else seed.new_empty((0,))
        if final:
            terminal.index_copy_(
                0,
                plan.docs.index_select(0, plan.terminal),
                states.detach().index_select(0, plan.terminal),
            )
            if initial_state is not None:
                terminal.index_copy_(0, plan.empty_docs, seed.index_select(0, plan.empty_docs))
            dist.all_reduce(terminal, group=group)
        if track_grad:
            ctx.save_for_backward(*inputs, output, states)
        ctx.plan, ctx.group, ctx.final = plan, group, final
        ctx.adapter, ctx.options, ctx.gate_names = adapter, options, gate_names
        ctx.has_initial_state = initial_state is not None
        return output.detach(), terminal

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_final):
        """Exchange reverse recipes and pass outgoing state adjoints to FE's VJP."""
        tensors = ctx.saved_tensors[:-3]
        starting, output, states = ctx.saved_tensors[-3:]
        q, k, v = tensors[:3]
        plan, group = ctx.plan, ctx.group
        if grad_output is None:
            grad_output = torch.zeros_like(output)
        grad_output = grad_output.contiguous()
        shape = (len(plan.host.documents) - 1, q.shape[1], v.shape[-1], k.shape[-1])
        seed = starting.new_zeros(shape if ctx.final else (1, *shape[1:]))
        if ctx.final:
            if grad_final is not None:
                seed.copy_(grad_final)
            # Final states are replicated outputs: sum losses consuming any replica.
            dist.all_reduce(seed, group=group)
        additive, transition = _summary_buffers(seed, plan.host.slots)
        if plan.host.incoming:
            span = _selected_span(plan, plan.host.incoming)
            selected = tuple(t[span] for t in tensors)
            h, t = ctx.adapter._cp_backward_summary(
                *selected[:3],
                dict(zip(ctx.gate_names, selected[3:])),
                grad_output[span],
                plan.incoming_cu,
                **ctx.options,
            )
            slots = _slot_indices(plan, plan.incoming)
            additive.index_copy_(0, slots, h)
            transition.index_copy_(0, slots, t)
        if plan.host.communicates:
            additive, transition = _gather_summaries(additive, transition, group)
        ending_grad = tex.linear_attention_affine_prefix(
            seed, additive, transition, plan.backward if ctx.final else plan.backward_zero
        )
        if plan.host.fragments:
            with torch.enable_grad():
                grads = torch.autograd.grad(
                    (output, states),
                    (*tensors, starting),
                    (grad_output, ending_grad),
                    # The outer graph controls lifetime, including retain_graph=True.
                    retain_graph=True,
                )
        else:
            grads = tuple(torch.zeros_like(t) for t in (*tensors, starting))
        initial_grad = None
        if ctx.has_initial_state:
            initial_grad = seed.new_zeros(shape)
            initial_grad.index_add_(
                0, plan.seed_docs, grads[-1].index_select(0, plan.seed_fragments)
            )
            if ctx.final:
                initial_grad.index_copy_(0, plan.empty_docs, seed.index_select(0, plan.empty_docs))
            # Each document's seed gradient belongs to its first rank. Do not
            # replicate it here: if initial_state came from a previous CP call,
            # that call's replicated-final-state VJP sums these owner gradients.
            # Replicating now would count chained state gradients CP-size times.
        return (None, None, None, None, None, None, None, initial_grad, *grads[:-1])


def context_parallel_attention(
    adapter,
    q,
    k,
    v,
    gates,
    initial_state,
    *,
    group,
    qkv_format,
    cu_seqlens,
    global_cu_seqlens,
    output_final_state,
    **kernel_kwargs,
):
    """Validate local inputs, cache a partition, and restore the public output layout."""
    adapter._validate_inputs(q, k, v, gates, initial_state, qkv_format)
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    tensors = tuple(_to_thd(t, qkv_format).contiguous() for t in (q, k, v, *gates.values()))
    dense = qkv_format != "thd"
    batch, length = None, None
    if dense:
        if cu_seqlens is not None:
            raise ValueError("Dense linear-attention CP does not accept local cu_seqlens.")
        batch, length = q.shape[:2] if qkv_format == "bshd" else (q.shape[1], q.shape[0])
    elif not isinstance(global_cu_seqlens, torch.Tensor):
        raise ValueError(
            "Linear-attention CP with THD requires global_cu_seqlens for all documents."
        )
    if cu_seqlens is not None:
        _validate_cu_seqlens(cu_seqlens, device=q.device, name="cu_seqlens")
    if global_cu_seqlens is not None:
        if not isinstance(global_cu_seqlens, torch.Tensor):
            raise TypeError("global_cu_seqlens must be a torch.Tensor.")
        if global_cu_seqlens.dtype != torch.int32 or global_cu_seqlens.dim() != 1:
            raise ValueError("global_cu_seqlens must be a one-dimensional int32 tensor.")
        metadata = (global_cu_seqlens, cu_seqlens)
        cacheable = all(t is None or not torch.is_inference(t) for t in metadata)
        key = (
            qkv_format,
            tuple(q.shape[:2]),
            id(global_cu_seqlens),
            global_cu_seqlens._version if cacheable else None,
            q.device,
            size,
            rank,
            id(cu_seqlens),
            None if cu_seqlens is None or not cacheable else cu_seqlens._version,
        )
    else:
        key, cacheable = (qkv_format, batch, length, q.device, size, rank), True
    cache = getattr(adapter, "_cp_partition_cache", None)
    if not cacheable:
        # Inference tensors have no mutation counter, so revalidate their
        # metadata instead of silently reusing a stale partition.
        cache = None
    if cache is None or cache[0] != key:
        documents = (
            tuple(global_cu_seqlens.tolist())
            if global_cu_seqlens is not None
            else tuple(i * length * size for i in range(batch + 1))
        )
        if dense and len(documents) != batch + 1:
            raise ValueError("Dense global_cu_seqlens must describe the full batch.")
        host = build_linear_attention_partition(documents, size, rank, dense=dense)
        plan = _device_partition(host, q.device)
        if cu_seqlens is not None:
            if tuple(cu_seqlens.tolist()) != host.local_cu_seqlens:
                raise ValueError("Local cu_seqlens does not match the contiguous CP fragments.")
        # Hold tensor references to prevent object-ID reuse from invalidating the cache.
        adapter._cp_partition_cache = (key, plan, global_cu_seqlens, cu_seqlens)
    else:
        plan = cache[1]
    if tensors[0].shape[0] != plan.host.local_cu_seqlens[-1]:
        raise ValueError("Local token count does not match this CP rank's token interval.")
    adapter._validate_initial_state(
        initial_state, batch_size=len(plan.host.documents) - 1, device=q.device
    )
    tensors, initial_state = adapter._prepare_cp_inputs(tensors, initial_state)
    if adapter._op is None:
        adapter._op = adapter._resolve_op()
    if (
        not plan.host.communicates
        and initial_state is None
        and not output_final_state
        and plan.host.fragments
    ):
        output, states = _local_attention(
            adapter, tensors, tuple(gates), plan.cu, None, kernel_kwargs, final=False
        )
    else:
        track_grad = torch.is_grad_enabled() and any(
            t is not None and t.requires_grad for t in (*tensors, initial_state)
        )
        output, states = _LinearAttentionContextParallel.apply(
            adapter,
            plan,
            group,
            kernel_kwargs,
            output_final_state,
            tuple(gates),
            track_grad,
            initial_state,
            *tensors,
        )
    output = output[..., : v.shape[-1]].contiguous()
    if output_final_state:
        states = states[..., : v.shape[-1], : k.shape[-1]].contiguous()
    output = (
        output.reshape(*q.shape[:-2], q.shape[-2] * v.shape[-1])
        if qkv_format == "thd"
        else output.reshape(batch, length, q.shape[-2] * v.shape[-1])
    )
    if qkv_format == "sbhd":
        output = output.transpose(0, 1).contiguous()
    return (output, states) if output_final_state else output
