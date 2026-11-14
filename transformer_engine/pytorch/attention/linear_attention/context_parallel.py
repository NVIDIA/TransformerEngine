# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Contiguous linear-attention CP: fragments, collectives and state gradients.

Flavor adapters supply local attention and affine summary operations. The shared
CUDA operation composes states in document order.
"""

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

import transformer_engine_torch as tex
from transformer_engine.common.linear_attention import (
    LinearAttentionCPPartition,
    build_linear_attention_partition,
)
from .base import _to_thd


_CP_METADATA_CACHE_SIZE = 8
_cp_metadata_cache = OrderedDict()
_cp_metadata_cache_lock = Lock()


def _get_shared_cp_metadata(key):
    """Look up a prepared partition shared by attention modules on one stream."""
    with _cp_metadata_cache_lock:
        entry = _cp_metadata_cache.get(key)
        if entry is not None:
            _cp_metadata_cache.move_to_end(key)
        return entry


def _cache_cp_metadata(entry):
    """Retain at most eight shared partitions, including their input metadata."""
    with _cp_metadata_cache_lock:
        key = entry[0]
        _cp_metadata_cache[key] = entry
        _cp_metadata_cache.move_to_end(key)
        if len(_cp_metadata_cache) > _CP_METADATA_CACHE_SIZE:
            _cp_metadata_cache.popitem(last=False)


@dataclass
class _CPMetadata:
    """Logical partition and device tensors used to execute linear-attention CP."""

    partition: LinearAttentionCPPartition
    cu_seqlens: torch.Tensor
    forward_descriptors: torch.Tensor
    backward_descriptors: torch.Tensor
    forward_zero_descriptors: torch.Tensor
    backward_zero_descriptors: torch.Tensor
    document_indices: torch.Tensor
    outgoing: torch.Tensor
    incoming: torch.Tensor
    terminal: torch.Tensor
    initial_state_fragments: torch.Tensor
    initial_state_documents: torch.Tensor
    empty_documents: torch.Tensor
    outgoing_cu_seqlens: torch.Tensor
    incoming_cu_seqlens: torch.Tensor


def _build_cp_metadata(partition, device):
    """Build device metadata for an existing logical partition."""

    def indices(values):
        return torch.tensor(values, device=device, dtype=torch.int64)

    def cumulative(selected):
        offsets = [0]
        for index in selected:
            frag = partition.fragments[index]
            offsets.append(offsets[-1] + frag.end - frag.start)
        return torch.tensor(offsets, device=device, dtype=torch.int32)

    first = [i for i, f in enumerate(partition.fragments) if f.first_rank == partition.cp_rank]
    forward_descriptors = torch.tensor(
        partition.forward_descriptors, device=device, dtype=torch.int32
    ).reshape(-1, 4)
    backward_descriptors = torch.tensor(
        partition.backward_descriptors, device=device, dtype=torch.int32
    ).reshape(-1, 4)
    forward_zero_descriptors, backward_zero_descriptors = (
        forward_descriptors.clone(),
        backward_descriptors.clone(),
    )
    forward_zero_descriptors[:, 0] = -1
    backward_zero_descriptors[:, 0] = -1
    return _CPMetadata(
        partition,
        torch.tensor(partition.local_cu_seqlens, device=device, dtype=torch.int32),
        forward_descriptors,
        backward_descriptors,
        forward_zero_descriptors,
        backward_zero_descriptors,
        indices([f.document for f in partition.fragments]),
        indices(partition.outgoing),
        indices(partition.incoming),
        indices(partition.terminal),
        indices(first),
        indices([partition.fragments[i].document for i in first]),
        indices(partition.empty_owned),
        cumulative(partition.outgoing),
        cumulative(partition.incoming),
    )


def _gather_summaries(additive, transition, group):
    """One fixed-size all-gather for both summary tensors, in CP group rank order."""
    slots, heads, value_dim, key_dim = additive.shape
    cp_size = dist.get_world_size(group)
    packed = torch.cat((additive.flatten(), transition.flatten()))
    gathered = torch.empty(cp_size * packed.numel(), device=packed.device, dtype=packed.dtype)
    dist.all_gather_into_tensor(gathered, packed, group=group)
    gathered = gathered.view(cp_size, -1)
    split = additive.numel()
    return (
        gathered[:, :split].contiguous().view(cp_size * slots, heads, value_dim, key_dim),
        gathered[:, split:].contiguous().view(cp_size * slots, heads, key_dim, key_dim),
    )


def _summary_buffers(state_template, slots):
    """Unused slots are identity recipes, including ranks owning zero tokens."""
    _, heads, value_dim, key_dim = state_template.shape
    additive = state_template.new_zeros((slots, heads, value_dim, key_dim))
    transition = torch.eye(key_dim, device=state_template.device, dtype=state_template.dtype)
    return additive, transition.expand(slots, heads, key_dim, key_dim).contiguous()


def _selected_span(metadata, selected):
    """Selected spans are consecutive: all dense spans, or one packed edge span."""
    cu_seqlens = metadata.partition.local_cu_seqlens
    return slice(cu_seqlens[selected[0]], cu_seqlens[selected[-1] + 1])


def _slot_indices(metadata, indices):
    """Dense slots are document IDs; packed partitions have only one edge slot."""
    if metadata.partition.slots == 1:
        return torch.zeros_like(indices)
    return metadata.document_indices.index_select(0, indices)


def _local_attention(
    adapter, tensors, gate_names, cu_seqlens, state, options, *, output_final_state=True
):
    """Call the flavor adapter with packed local inputs."""
    return adapter._call_op(
        *tensors[:3],
        dict(zip(gate_names, tensors[3:])),
        cu_seqlens,
        initial_state=state,
        output_final_state=output_final_state,
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
        ctx,
        adapter,
        metadata,
        group,
        options,
        output_final_state,
        gate_names,
        track_grad,
        initial_state,
        *tensors,
    ):
        """Establish starting states, then retain the local attention graph."""
        ctx.set_materialize_grads(False)
        partition = metadata.partition
        q = tensors[0]
        k = tensors[1]
        v = tensors[2]
        state_shape = (
            len(partition.cu_seqlens) - 1,
            q.shape[1],
            v.shape[-1],
            k.shape[-1],
        )
        # Zero-prefix descriptors need only one placeholder when no initial state is supplied.
        initial_state_buffer = (
            q.new_zeros((1, *state_shape[1:]), dtype=torch.float32)
            if initial_state is None
            else initial_state
        )
        additive, transition = _summary_buffers(initial_state_buffer, partition.slots)
        if partition.outgoing:
            span = _selected_span(metadata, partition.outgoing)
            selected = tuple(t[span] for t in tensors)
            h, t = adapter._cp_forward_summary(
                *selected[:3],
                dict(zip(gate_names, selected[3:])),
                metadata.outgoing_cu_seqlens,
                **options,
            )
            slots = _slot_indices(metadata, metadata.outgoing)
            additive.index_copy_(0, slots, h)
            transition.index_copy_(0, slots, t)
        if partition.communicates:
            additive, transition = _gather_summaries(additive, transition, group)
        starting_states = tex.linear_attention_affine_prefix(
            initial_state_buffer.contiguous(),
            additive,
            transition,
            (
                metadata.forward_descriptors
                if initial_state is not None
                else metadata.forward_zero_descriptors
            ),
        )
        # Retain FE's local graph only for training. This Function owns the
        # distributed VJP and the outer state gradients.
        with torch.set_grad_enabled(track_grad):
            inputs = tuple(
                t.detach().requires_grad_(track_grad) for t in (*tensors, starting_states)
            )
            if partition.fragments:
                output, states = _local_attention(
                    adapter, inputs[:-1], gate_names, metadata.cu_seqlens, inputs[-1], options
                )
            else:
                output = v.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
                states = starting_states
        terminal = (
            initial_state_buffer.new_zeros(state_shape)
            if output_final_state
            else initial_state_buffer.new_empty((0,))
        )
        if output_final_state:
            terminal.index_copy_(
                0,
                metadata.document_indices.index_select(0, metadata.terminal),
                states.detach().index_select(0, metadata.terminal),
            )
            if initial_state is not None:
                terminal.index_copy_(
                    0,
                    metadata.empty_documents,
                    initial_state_buffer.index_select(0, metadata.empty_documents),
                )
            dist.all_reduce(terminal, group=group)
        if track_grad:
            ctx.save_for_backward(*inputs, output, states)
        ctx.cp_metadata, ctx.group, ctx.output_final_state = metadata, group, output_final_state
        ctx.adapter, ctx.options, ctx.gate_names = adapter, options, gate_names
        ctx.has_initial_state = initial_state is not None
        return output.detach(), terminal

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_final):
        """Exchange reverse recipes and pass outgoing state adjoints to FE's VJP."""
        tensors = ctx.saved_tensors[:-3]
        starting_states, output, states = ctx.saved_tensors[-3:]
        q, k, v = tensors[:3]
        metadata, group = ctx.cp_metadata, ctx.group
        partition = metadata.partition
        if grad_output is None:
            grad_output = torch.zeros_like(output)
        grad_output = grad_output.contiguous()
        shape = (
            len(partition.cu_seqlens) - 1,
            q.shape[1],
            v.shape[-1],
            k.shape[-1],
        )
        final_state_grad = starting_states.new_zeros(
            shape if ctx.output_final_state else (1, *shape[1:])
        )
        if ctx.output_final_state:
            if grad_final is not None:
                final_state_grad.copy_(grad_final)
            # Final states are replicated outputs: sum losses consuming any replica.
            dist.all_reduce(final_state_grad, group=group)
        additive, transition = _summary_buffers(final_state_grad, partition.slots)
        if partition.incoming:
            span = _selected_span(metadata, partition.incoming)
            selected = tuple(t[span] for t in tensors)
            h, t = ctx.adapter._cp_backward_summary(
                *selected[:3],
                dict(zip(ctx.gate_names, selected[3:])),
                grad_output[span],
                metadata.incoming_cu_seqlens,
                **ctx.options,
            )
            slots = _slot_indices(metadata, metadata.incoming)
            additive.index_copy_(0, slots, h)
            transition.index_copy_(0, slots, t)
        if partition.communicates:
            additive, transition = _gather_summaries(additive, transition, group)
        ending_state_grad = tex.linear_attention_affine_prefix(
            final_state_grad,
            additive,
            transition,
            (
                metadata.backward_descriptors
                if ctx.output_final_state
                else metadata.backward_zero_descriptors
            ),
        )
        if partition.fragments:
            with torch.enable_grad():
                grads = torch.autograd.grad(
                    (output, states),
                    (*tensors, starting_states),
                    (grad_output, ending_state_grad),
                    # The outer graph controls lifetime, including retain_graph=True.
                    retain_graph=True,
                )
        else:
            grads = tuple(torch.zeros_like(t) for t in (*tensors, starting_states))
        initial_state_grad = None
        if ctx.has_initial_state:
            initial_state_grad = final_state_grad.new_zeros(shape)
            initial_state_grad.index_add_(
                0,
                metadata.initial_state_documents,
                grads[-1].index_select(0, metadata.initial_state_fragments),
            )
            if ctx.output_final_state:
                initial_state_grad.index_copy_(
                    0,
                    metadata.empty_documents,
                    final_state_grad.index_select(0, metadata.empty_documents),
                )
            # Each document's initial-state gradient belongs to its first rank. Do not
            # replicate it here: if initial_state came from a previous CP call,
            # that call's replicated-final-state VJP sums these owner gradients.
            # Replicating now would count chained state gradients CP-size times.
        return (None, None, None, None, None, None, None, initial_state_grad, *grads[:-1])


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
    output_final_state,
    **kernel_kwargs,
):
    """Derive local fragment boundaries from full-document metadata and run CP."""
    adapter._validate_inputs(q, k, v, gates, initial_state, qkv_format)
    cp_size, cp_rank = dist.get_world_size(group), dist.get_rank(group)
    stream = torch.cuda.current_stream(q.device).cuda_stream
    tensors = tuple(_to_thd(t, qkv_format).contiguous() for t in (q, k, v, *gates.values()))
    dense = qkv_format != "thd"
    batch, length = None, None
    if dense:
        batch, length = q.shape[:2] if qkv_format == "bshd" else (q.shape[1], q.shape[0])
    elif cu_seqlens is None:
        raise ValueError("Linear-attention CP with THD requires cu_seqlens for all documents.")
    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor.")
        if cu_seqlens.dtype != torch.int32 or cu_seqlens.dim() != 1:
            raise ValueError("cu_seqlens must be a one-dimensional int32 tensor.")
        cacheable = not torch.is_inference(cu_seqlens)
        key = (
            qkv_format,
            tuple(q.shape[:2]),
            id(cu_seqlens),
            cu_seqlens._version if cacheable else None,
            q.device,
            cp_size,
            cp_rank,
            stream,
        )
    else:
        key, cacheable = (qkv_format, batch, length, q.device, cp_size, cp_rank, stream), True
    cache = getattr(adapter, "_cp_partition_cache", None)
    if not cacheable:
        # Inference tensors have no mutation counter, so revalidate their
        # metadata instead of silently reusing a stale partition.
        cache = None
    elif cache is None or cache[0] != key:
        # Layers using the same batch can share metadata without another host round trip.
        cache = _get_shared_cp_metadata(key)
    # Build and upload metadata only when the cached partition no longer matches.
    if cache is None or cache[0] != key:
        document_cu_seqlens = (
            tuple(cu_seqlens.tolist())
            if cu_seqlens is not None
            else tuple(i * length * cp_size for i in range(batch + 1))
        )
        if dense and len(document_cu_seqlens) != batch + 1:
            raise ValueError("Dense cu_seqlens must describe the full batch.")
        partition = build_linear_attention_partition(
            document_cu_seqlens, cp_size=cp_size, cp_rank=cp_rank, dense=dense
        )
        metadata = _build_cp_metadata(partition, q.device)
        # Hold the metadata tensor to prevent object-ID reuse in the cache key.
        adapter._cp_partition_cache = (key, metadata, cu_seqlens)
        if cacheable:
            _cache_cp_metadata(adapter._cp_partition_cache)
    else:
        metadata = cache[1]
        partition = metadata.partition
        adapter._cp_partition_cache = cache
    if tensors[0].shape[0] != partition.local_cu_seqlens[-1]:
        raise ValueError("Local token count does not match this CP rank's token interval.")
    adapter._validate_initial_state(
        initial_state,
        batch_size=len(partition.cu_seqlens) - 1,
        device=q.device,
    )
    tensors, initial_state = adapter._prepare_cp_inputs(tensors, initial_state)
    # Resolve the cuDNN operation once and reuse it on later calls.
    if adapter._op is None:
        adapter._op = adapter._resolve_op()
    # Without shared documents or explicit states, use ordinary local autograd.
    if (
        not partition.communicates
        and initial_state is None
        and not output_final_state
        and partition.fragments
    ):
        output, states = _local_attention(
            adapter,
            tensors,
            tuple(gates),
            metadata.cu_seqlens,
            None,
            kernel_kwargs,
            output_final_state=False,
        )
    else:
        track_grad = torch.is_grad_enabled() and any(
            t is not None and t.requires_grad for t in (*tensors, initial_state)
        )
        output, states = _LinearAttentionContextParallel.apply(
            adapter,
            metadata,
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
