# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention."""
import os
import warnings
import math
from importlib.metadata import version
from contextlib import nullcontext
from typing import Any, Callable, Optional, Tuple, Union, Dict
from pkg_resources import packaging

import torch

import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd_qkvpacked,
    fused_attn_bwd_qkvpacked,
    fused_attn_fwd_kvpacked,
    fused_attn_bwd_kvpacked,
    QKVLayout,
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
)
from transformer_engine.pytorch.module import LayerNormLinear, Linear
from transformer_engine.pytorch.utils import (
    divide,
    attention_mask_func,
    split_tensor_along_dim,
    get_device_compute_capability,
    get_default_init_method,
)
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    AttnBiasTypes,
    dist_group_type,
    TE_DType,
)
from transformer_engine.pytorch.softmax import FusedScaleMaskSoftmax
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    get_distributed_rank,
    checkpoint,
)
from transformer_engine.pytorch.export import is_in_onnx_export_mode

_flash_attn_version = packaging.version.Version(version("flash-attn"))
_flash_attn_version_required = packaging.version.Version("1.0.6")
_flash_attn_2_available = _flash_attn_version >= packaging.version.Version("2")

if _flash_attn_2_available:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_forward_func # pylint: disable=no-name-in-module
    from flash_attn_2_cuda import varlen_bwd as flash_attn_cuda_bwd # pylint: disable=no-name-in-module
else:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func as flash_attn_forward_func # pylint: disable=no-name-in-module,ungrouped-imports
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward


__all__ = ["DotProductAttention", "MultiheadAttention"]


def flash_attn_p2p_communicate(rank, send_tensor, send_dst,
                               recv_tensor, recv_src,
                               cp_group, batch_p2p_comm):
    """Point-to-point communications of KV and dKV in Flash Attention with context parallelism"""
    send_recv_ops = []

    if batch_p2p_comm:
        if rank % 2 == 0:
            send_op = torch.distributed.P2POp(torch.distributed.isend,
                                              send_tensor,
                                              send_dst,
                                              cp_group)
            recv_op = torch.distributed.P2POp(torch.distributed.irecv,
                                              recv_tensor,
                                              recv_src,
                                              cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.P2POp(torch.distributed.irecv,
                                              recv_tensor,
                                              recv_src,
                                              cp_group)
            send_op = torch.distributed.P2POp(torch.distributed.isend,
                                              send_tensor,
                                              send_dst,
                                              cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        if rank % 2 == 0:
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = send_recv_ops

    return send_recv_reqs


@torch.jit.script
def flash_attn_fwd_out_correction(out, out_per_step, softmax_lse, softmax_lse_per_step):
    """Merge partial outputs of each step in Flash Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse).transpose(1, 2)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step*softmax_lse_corrected_exp
    out.add_(out_corrected)


@torch.jit.script
def flash_attn_fwd_softmax_lse_correction(softmax_lse, softmax_lse_per_step):
    """Merge softmax stats of each step in Flash Attention with context parallelism"""
    softmax_lse.exp_()
    softmax_lse.add_(softmax_lse_per_step.to(torch.double).exp())
    softmax_lse.log_()


class FlashAttnUnpaddedFuncWithCP(torch.autograd.Function):
    """
    Flash Attention implementation with context parallelism.
    Split flash attention compute into multiple steps, and overlap current-step
    compute with next-step communication.
    """

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                cp_group, cp_global_ranks, cp_stream, softmax_scale, causal, deterministic):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)
        send_dst = cp_global_ranks[(rank + 1) % cp_size]
        recv_src = cp_global_ranks[(rank + cp_size - 1) % cp_size]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (cp_size == 2)

        # [b, s, np, hn] -> [b, 2, s//2, np, hn]
        q, k, v = [x.view(x.shape[0], 2, x.shape[1]//2, *x.shape[2:]) for x in [q, k, v]]
        # Flash Attn inputs
        q_inputs = [None, None]
        kv_inputs = [None, None]
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        rng_states = [None for _ in range(cp_size)]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()

        p2p_comm_buffers = [None for _ in range(cp_size)]
        p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]

        for i in range(cp_size+1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i%2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i+1)%2]:
                        req.wait()

                    if i < (cp_size-1):
                        p2p_comm_buffers[i+1] = torch.empty_like(p2p_comm_buffers[i])
                        send_recv_reqs[i%2] = flash_attn_p2p_communicate(rank,
                                                                         p2p_comm_buffers[i],
                                                                         send_dst,
                                                                         p2p_comm_buffers[i+1],
                                                                         recv_src,
                                                                         cp_group,
                                                                         batch_p2p_comm)

                    kv_inputs[i%2] = p2p_comm_buffers[i]
                    if causal:
                        if i == 0:
                            # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                            q_inputs[i%2] = q.view(-1, *q.shape[-2:])
                            out_per_step[i] = torch.empty_like(q_inputs[i%2])
                            # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                            kv_inputs[i%2] = kv_inputs[i%2].view(2, -1, *k.shape[-2:])
                            _, softmax_lse_per_step[i], rng_states[i], _ = _flash_attn_forward(
                                q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                out_per_step[i], cu_seqlens_q, cu_seqlens_k,
                                max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale,
                                causal=True, return_softmax=False,
                            )
                        elif i <= rank:
                            # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                            q_inputs[i%2] = q.view(-1, *q.shape[-2:])
                            out_per_step[i] = torch.empty_like(q_inputs[i%2])
                            # [2, b, sk//2, np, hn] -> [2, b*sk//2, np, hn]
                            kv_inputs[i%2] = kv_inputs[i%2][:, :, 0, ...].contiguous()
                            kv_inputs[i%2] = kv_inputs[i%2].view(2, -1, *k.shape[-2:])
                            _, softmax_lse_per_step[i], rng_states[i], _ = _flash_attn_forward(
                                q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                out_per_step[i], cu_seqlens_q, cu_seqlens_k//2,
                                max_seqlen_q, max_seqlen_k//2, dropout_p, softmax_scale,
                                causal=False, return_softmax=False,
                            )
                        else:
                            # [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                            q_inputs[i%2] = q[:, 1, ...].contiguous().view(-1, *q.shape[-2:])
                            out_per_step[i] = torch.empty_like(q_inputs[i%2])
                            # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                            kv_inputs[i%2] = kv_inputs[i%2].view(2, -1, *k.shape[-2:])
                            _, softmax_lse_per_step[i], rng_states[i], _ = _flash_attn_forward(
                                q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                out_per_step[i], cu_seqlens_q//2, cu_seqlens_k,
                                max_seqlen_q//2, max_seqlen_k, dropout_p, softmax_scale,
                                causal=False, return_softmax=False,
                            )
                    else:
                        assert False, "Not implemented yet!"

            if i > 0:
                # wait until fwd restuls correction of last step is done
                if i > 1:
                    flash_attn_streams[(i-1)%2].wait_event(fwd_results_correction_done)

                with torch.cuda.stream(flash_attn_streams[(i-1)%2]):
                    if causal:
                        if i == 1:
                            out = torch.empty_like(q).zero_()
                            softmax_lse = torch.clone(softmax_lse_per_step[0]).to(torch.double)
                            # [b, np, sq] -> [b, np, 2, sq//2]
                            softmax_lse_ = softmax_lse.view(
                                *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1]//2
                            )
                        elif (i-1) <= rank:
                            flash_attn_fwd_softmax_lse_correction(softmax_lse,
                                                                  softmax_lse_per_step[i-1])
                        else:
                            flash_attn_fwd_softmax_lse_correction(softmax_lse_[..., 1, :],
                                                                  softmax_lse_per_step[i-1])
                    else:
                        assert False, "Not implemented yet!"

                if i < cp_size:
                    flash_attn_streams[(i-1)%2].record_event(fwd_results_correction_done)

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        softmax_lse = softmax_lse.to(torch.float)
        for i in range(cp_size):
            # [b*sq, np, hn] -> [b, sq, np, hn] or [b*sq//2, np, hn] -> [b, sq//2, np, hn]
            out_ = out_per_step[i].view(out.shape[0], -1, *out.shape[-2:])
            if i <= rank:
                flash_attn_fwd_out_correction(out.view(*out_.shape),
                                                       out_,
                                                       softmax_lse,
                                                       softmax_lse_per_step[i])
            else:
                flash_attn_fwd_out_correction(out[:, 1, ...],
                                              out_,
                                              softmax_lse_[..., 1, :],
                                              softmax_lse_per_step[i])

        kv = p2p_comm_buffers[-1]
        out = out.view(-1, *out.shape[-2:])
        ctx.save_for_backward(q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.rng_states = rng_states
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        return out

    @staticmethod
    def backward(ctx, dout):
        q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors

        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)
        send_dst = ctx.cp_global_ranks[(rank + cp_size - 1) % cp_size]
        recv_src = ctx.cp_global_ranks[(rank + 1) % cp_size]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (cp_size == 2)

        # [b, np, sq] -> [b, np, 2, sq//2]
        softmax_lse_ = softmax_lse.view(*softmax_lse.shape[:-1], 2, softmax_lse.shape[-1]//2)
        # [b*sq, np, hn] -> [b, 2, sq//2, np, hn]
        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        # Flash Attn outputs
        dq = torch.empty_like(q)

        p2p_comm_buffers = [torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device), \
                            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device)]
        p2p_comm_buffers[0][0].copy_(kv)
        send_recv_reqs = []

        for i in range(cp_size):
            # wait until KV is received
            for req in send_recv_reqs:
                req.wait()

            send_tensor = p2p_comm_buffers[i%2]
            recv_tensor = p2p_comm_buffers[(i+1)%2]
            if i == 0:
                send_tensor = send_tensor[0]
                recv_tensor = recv_tensor[0]
            if i == (cp_size-1):
                send_tensor = send_tensor[1]
                recv_tensor = recv_tensor[1]

            send_recv_reqs = flash_attn_p2p_communicate(rank,
                                                        send_tensor,
                                                        send_dst,
                                                        recv_tensor,
                                                        recv_src,
                                                        ctx.cp_group,
                                                        batch_p2p_comm)

            kv = p2p_comm_buffers[i%2][0]
            # In reversed order of fwd
            if ctx.causal:
                if i == (cp_size-1):
                    # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                    q_ = q.view(-1, *q.shape[-2:])
                    dq_ = torch.empty_like(q_)
                    # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                    kv_ = kv.view(2, -1, *kv.shape[-2:])
                    dkv_ = torch.empty_like(kv_)
                    # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                    out_ = out.view(-1, *out.shape[-2:])
                    dout_ = dout.view(-1, *dout.shape[-2:])
                    _flash_attn_backward(
                        dout_, q_, kv_[0], kv_[1], out_, softmax_lse,
                        dq_, dkv_[0], dkv_[1], cu_seqlens_q, cu_seqlens_k,
                        ctx.max_seqlen_q, ctx.max_seqlen_k,
                        ctx.dropout_p, ctx.softmax_scale, True,
                        rng_state=ctx.rng_states[cp_size-i-1],
                        num_splits=1 if ctx.deterministic else 0,
                    )
                elif i >= (cp_size-rank-1):
                    # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                    q_ = q.view(-1, *q.shape[-2:])
                    dq_ = torch.empty_like(q_)
                    # [2, b, sk//2, np, hn] -> [2, b*sk//2, np, hn]
                    kv_ = kv[:, :, 0, ...].contiguous().view(2, -1, *kv.shape[-2:])
                    dkv_ = torch.empty_like(kv_)
                    # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                    out_ = out.view(-1, *out.shape[-2:])
                    dout_ = dout.view(-1, *dout.shape[-2:])
                    _flash_attn_backward(
                        dout_, q_, kv_[0], kv_[1], out_, softmax_lse,
                        dq_, dkv_[0], dkv_[1], cu_seqlens_q, cu_seqlens_k//2,
                        ctx.max_seqlen_q, ctx.max_seqlen_k//2,
                        ctx.dropout_p, ctx.softmax_scale, False,
                        rng_state=ctx.rng_states[cp_size-i-1],
                        num_splits=1 if ctx.deterministic else 0,
                    )
                else:
                    # [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                    q_ = q[:, 1, ...].contiguous().view(-1, *q.shape[-2:])
                    dq_ = torch.empty_like(q_)
                    # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                    kv_ = kv.view(2, -1, *kv.shape[-2:])
                    dkv_ = torch.empty_like(kv_)
                    # [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                    out_ = out[:, 1, ...].contiguous().view(-1, *out.shape[-2:])
                    dout_ = dout[:, 1, ...].contiguous().view(-1, *dout.shape[-2:])
                    _flash_attn_backward(
                        dout_, q_, kv_[0], kv_[1], out_, softmax_lse_[..., 1, :],
                        dq_, dkv_[0], dkv_[1], cu_seqlens_q//2, cu_seqlens_k,
                        ctx.max_seqlen_q//2, ctx.max_seqlen_k,
                        ctx.dropout_p, ctx.softmax_scale, False,
                        rng_state=ctx.rng_states[cp_size-i-1],
                        num_splits=1 if ctx.deterministic else 0,
                    )

                if i >= (cp_size-rank-1):
                    # [b*sq, np, hn] -> [b, 2, sq//2, np, hn]
                    dq_ = dq_.view(*dq.shape)
                else:
                    # [b*sq//2, np, hn] -> [b, sq//2, np, hn]
                    dq_ = dq_.view(dq.shape[0], *dq.shape[2:])

                if i > (cp_size-rank-1):
                    dq.add_(dq_)
                elif i == (cp_size-rank-1):
                    if rank == (cp_size-1):
                        dq.copy_(dq_)
                    else:
                        dq[:, 0, ...].copy_(dq_[:, 0, ...])
                        dq[:, 1, ...].add_(dq_[:, 1, ...])
                elif i > 0:
                    dq[:, 1, ...].add_(dq_)
                else:
                    dq[:, 1, ...].copy_(dq_)

                # wait until dKV is received
                for req in send_recv_reqs:
                    req.wait()

                dkv = p2p_comm_buffers[(i+1)%2][1]
                if i >= (cp_size-rank-1) and i != (cp_size-1):
                    # [2, b*sk//2, np, hn] -> [2, b, sk//2, np, hn]
                    dkv_ = dkv_.view(*dkv.shape[0:2], *dkv.shape[3:])
                else:
                    # [2, b*sk, np, hn] -> [2, b, 2, sk//2, np, hn]
                    dkv_ = dkv_.view(*dkv.shape)

                if i == (cp_size-1):
                    if rank == 0:
                        dkv[:, :, 0, ...].add_(dkv_[:, :, 0, ...])
                        dkv[:, :, 1, ...].copy_(dkv_[:, :, 1, ...])
                    else:
                        dkv.add_(dkv_)
                elif i >= (cp_size-rank-1):
                    if i == 0 and rank == (cp_size-1):
                        dkv[:, :, 0, ...].copy_(dkv_)
                    else:
                        dkv[:, :, 0, ...].add_(dkv_)
                elif i > 0:
                    dkv.add_(dkv_)
                else:
                    dkv.copy_(dkv_)
            else:
                assert False, "Not implemented yet!"

        # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
        dq = dq.view(q.shape[0], -1, *q.shape[-2:])
        # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
        dkv = dkv.view(*kv.shape[0:2], -1, *kv.shape[-2:])
        return dq, dkv[0], dkv[1], None, None, None, None, None, None, None, None, None, None, None


def flash_attn_forward_func_with_cp(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                    max_seqlen_q, max_seqlen_k, dropout_p,
                                    cp_group, cp_global_ranks, cp_stream,
                                    softmax_scale=None, causal=False,
                                    deterministic=False):
    """Flash Attention implementation with context parallelism"""
    out = FlashAttnUnpaddedFuncWithCP.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
        cp_group, cp_global_ranks, cp_stream, softmax_scale, causal, deterministic
    )
    return out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor `freqs` is of shape [seq_length, ..., dim]
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


class _SplitLastDim(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx,
                mixed_x_layer: torch.Tensor,
                num_parts: int
    ) -> Tuple[torch.Tensor, ...]:
        return split_tensor_along_dim(mixed_x_layer, -1, num_parts)

    @staticmethod
    def backward(ctx,
                 *grad_outputs):
        assert len(grad_outputs) > 0, "No gradients received for backprop!"

        noop_ok = True
        strides = grad_outputs[0].stride()
        data_ptr = grad_outputs[0].storage().data_ptr()
        shape = grad_outputs[0].shape
        last_dim_size = grad_outputs[0].shape[-1]
        for i, tensor in enumerate(grad_outputs):
            if (tensor.stride() != strides or
                tensor.shape != shape or
                tensor.storage().data_ptr() != data_ptr or
                tensor.storage_offset() != i * last_dim_size):
                noop_ok = False
                break

        if noop_ok:
            ret = torch.Tensor().to(grad_outputs[0].dtype)
            ret = torch.Tensor().to(device=grad_outputs[0].device,
                                    dtype=grad_outputs[0].dtype)
            new_shape = list(shape)
            new_shape[-1] = new_shape[-1] * len(grad_outputs)
            ret.set_(grad_outputs[0].storage(),
                     grad_outputs[0].storage_offset(),
                     new_shape,
                     grad_outputs[0].stride()
            )
            return ret, None

        return torch.cat(grad_outputs, dim = -1), None

class _CombineQKV(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx,
                query_layer: torch.Tensor,
                key_layer: torch.Tensor, # pylint: disable=unused-argument
                value_layer: torch.Tensor, # pylint: disable=unused-argument
                dim: int,
    ) -> torch.Tensor:

        mixed_layer = torch.Tensor().to(device=query_layer.device,
                                dtype=query_layer.dtype)
        new_shape = list(query_layer.shape)
        new_shape[dim] = new_shape[dim] * 3
        mixed_layer.set_(query_layer.untyped_storage(),
                 query_layer.storage_offset(),
                 new_shape,
                 query_layer.stride())
        ctx.dim = dim
        return mixed_layer

    @staticmethod
    def backward(ctx,
                 *grad_outputs,
    ) -> Tuple[torch.Tensor, ...]:
        assert len(grad_outputs) > 0, "No gradients received for backprop!"
        tensors = split_tensor_along_dim(grad_outputs[0], ctx.dim, 3)
        return tensors[0], tensors[1], tensors[2], None

class _CombineKV(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx,
                key_layer: torch.Tensor,
                value_layer: torch.Tensor, # pylint: disable=unused-argument
                dim: int,
    ) -> torch.Tensor:

        mixed_layer = torch.Tensor().to(device=key_layer.device,
                                dtype=key_layer.dtype)
        new_shape = list(key_layer.shape)
        new_shape[dim] = new_shape[dim] * 2
        mixed_layer.set_(key_layer.untyped_storage(),
                 key_layer.storage_offset(),
                 new_shape,
                 key_layer.stride())
        ctx.dim = dim
        return mixed_layer

    @staticmethod
    def backward(ctx,
                 *grad_outputs,
    ) -> Tuple[torch.Tensor, ...]:
        assert len(grad_outputs) > 0, "No gradients received for backprop!"
        tensors = split_tensor_along_dim(grad_outputs[0], ctx.dim, 2)
        return tensors[0], tensors[1], None

class UnfusedDotProductAttention(torch.nn.Module):
    """Parallel attention w/o QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        norm_factor: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attn_mask_type: str = "causal",
        layer_number: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert (
            attn_mask_type in AttnMaskTypes
        ), f"attn_mask_type {attn_mask_type} not supported"

        self.norm_factor = norm_factor
        self.attention_dropout_ctx = attention_dropout_ctx
        self.layer_number = layer_number

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            attn_mask_type,
            attention_mask_func,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        # An FP16 training trick required for certain GPT-like models.
        self.apply_qk_layer_scaling = (
            bool(int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))) and layer_number is not None)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """core attention fprop"""
        batch_size, seqlen = query_layer.shape[1], query_layer.shape[0]
        apply_qk_layer_scaling = self.apply_qk_layer_scaling and key_layer.dtype == torch.float16

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        assert key_layer.shape == value_layer.shape, "Keys and values must have the same shape!"
        if key_layer.shape[2] != query_layer.shape[2]:
            assert (query_layer.shape[2]%key_layer.shape[2]==0
                ),"The number of attention heads must be divisible by the number of GQA groups!"
            key_layer = key_layer.repeat_interleave(
                    int(query_layer.shape[2]/key_layer.shape[2]), dim = 2)
            value_layer = value_layer.repeat_interleave(
                    int(query_layer.shape[2]/value_layer.shape[2]), dim = 2)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(
            output_size[2], output_size[0] * output_size[1], -1
        )
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.reshape(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        # WAR to set dtype to FP32 as ONNX lacks BF16 support for ConstantOfShape operator
        is_bf16 = query_layer.dtype == torch.bfloat16
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=torch.float32 if is_in_onnx_export_mode() and is_bf16 else query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        if is_in_onnx_export_mode() and is_bf16:
            matmul_result = matmul_result.bfloat16()

        scale = self.norm_factor
        if apply_qk_layer_scaling:
            scale *= self.layer_number

        # Raw attention scores. [b * np, sq, sk]
        if core_attention_bias_type == "no_bias":
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / scale),
            )

        elif core_attention_bias_type == "pre_scale_bias":
            assert core_attention_bias is not None, "core_attention_bias should not be None!"
            assert (core_attention_bias.shape == torch.Size(1, *output_size[1:])
                    ), "core_attention_bias must be in [1, h, sq, skv] shape!"
            matmul_result = torch.bmm(
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            )
            matmul_result = (matmul_result.view(
                output_size[0], output_size[1], output_size[2], output_size[3])
                + core_attention_bias).view(-1, output_size[2], output_size[3])
            matmul_result /= scale

        elif core_attention_bias_type == "post_scale_bias":
            assert core_attention_bias is not None, "core_attention_bias should not be None!"
            assert (core_attention_bias.shape == torch.Size([1, *output_size[1:]])
                    ), "core_attention_bias must be in [1, h, sq, skv] shape!"
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / scale),
            )
            matmul_result = (matmul_result.view(
                output_size[0], output_size[1], output_size[2], output_size[3])
                + core_attention_bias).view(-1, output_size[2], output_size[3])

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]
        softmax_scale = self.layer_number if apply_qk_layer_scaling else None
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask, softmax_scale)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with self.attention_dropout_ctx():
            attention_probs = self.attention_dropout(attention_probs)

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        context_layer = context_layer.view(seqlen, batch_size, -1)

        return context_layer


class _PrepareQKVForFA(torch.autograd.Function):
    """This class converts QKV from interleaved (s, b, ...) layout
       to separate contiguous q, k, v tensors in (b, s, ...) layout."""

    @staticmethod
    def forward(ctx,
                query_layer: torch.Tensor,
                key_layer: torch.Tensor,
                value_layer: torch.Tensor
    ) -> torch.Tensor:
        # All inputs received are non-contiguous tensors.
        # The `query_layer` tensor is used to access the
        # full memory region of the QKV tensor.
        qkv = tex.fa_prepare_fwd(query_layer)
        q, k, v = split_tensor_along_dim(qkv, 0, 3)
        query_layer = torch.squeeze(q, 0)
        key_layer = torch.squeeze(k, 0)
        value_layer = torch.squeeze(v, 0)
        return query_layer, key_layer, value_layer

    @staticmethod
    def backward(ctx,
                 dq: torch.Tensor,
                 dk: torch.Tensor,
                 dv: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        dqkv = tex.fa_prepare_bwd(dq, dk, dv)
        dq, dk, dv = split_tensor_along_dim(dqkv, -1, 3)
        return dq, dk, dv


def _check_qkv_layout(q, k, v):
    data_ptr = q.untyped_storage().data_ptr()
    check_ptrs = all(x.untyped_storage().data_ptr() == data_ptr for x in [q, k, v])
    if not check_ptrs:
        return False

    stride = q.stride()
    check_strides = all(stride == x.stride() for x in [q, k, v])
    if not check_strides:
        return False

    shape = q.shape
    check_shapes = all(shape == x.shape for x in [q, k, v])
    if not check_shapes:
        return False

    last_dim_size = shape[-1]
    check_offsets = all(i * last_dim_size == x.storage_offset()
                        for i, x in enumerate([q, k, v]))
    if check_offsets:
        return "sbh3d"

    last_dims_size = shape[-1] * shape[-2]
    check_offsets = all(i * last_dims_size == x.storage_offset()
                        for i, x in enumerate([q, k, v]))
    if check_offsets:
        return "sb3hd"

    return "other"

def _check_kv_layout(k, v):
    data_ptr = k.untyped_storage().data_ptr()
    check_ptrs = all(x.untyped_storage().data_ptr() == data_ptr for x in [k, v])
    if not check_ptrs:
        return False

    stride = k.stride()
    check_strides = all(stride == x.stride() for x in [k, v])
    if not check_strides:
        return False

    shape = k.shape
    check_shapes = all(shape == x.shape for x in [k, v])
    if not check_shapes:
        return False

    last_dim_size = shape[-1]
    check_offsets = all(i * last_dim_size == x.storage_offset()
                        for i, x in enumerate([k, v]))
    if check_offsets:
        return "sbh2d"

    last_dims_size = shape[-1] * shape[-2]
    check_offsets = all(i * last_dims_size == x.storage_offset()
                        for i, x in enumerate([k, v]))
    if check_offsets:
        return "sb2hd"

    return "other"


class FlashAttention(torch.nn.Module):
    """Dot product attention, using HazyResearch flash-attn package:
    https://github.com/HazyResearch/flash-attention
    """

    def __init__(
        self,
        norm_factor: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attn_mask_type: str = "causal",
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        assert (
            _flash_attn_version >= _flash_attn_version_required
        ), f"FlashAttention minimum version {_flash_attn_version_required} is required."

        self.attn_causal_mask = attn_mask_type == "causal"
        self.norm_factor = norm_factor
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_dropout = attention_dropout
        self.deterministic = deterministic

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        cp_group: Optional[dist_group_type] = None,
        cp_global_ranks: Union[int] = None,
        cp_stream: torch.cuda.Stream = None,
    ) -> torch.Tensor:
        """flash-attn fprop"""

        assert (
            query_layer.dtype in [torch.float16, torch.bfloat16]
            and key_layer.dtype in [torch.float16, torch.bfloat16]
            and value_layer.dtype in [torch.float16, torch.bfloat16]
            ), 'FlashAttention currently only supports FP16 and BF16.'
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
            ), 'FlashAttention currently only supports CUDA tensors.'

        # For now just 128, will make it more general in the future

        if (query_layer.shape[-1] == 128 and
            query_layer.shape[0] * query_layer.shape[1] >= 512 and
            _check_qkv_layout(query_layer, key_layer, value_layer) == "sbh3d"):
            query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(query_layer,
                                                                         key_layer,
                                                                         value_layer)
        else:
            query_layer, key_layer, value_layer = [x.transpose(0,1).contiguous()
                           for x in (query_layer, key_layer, value_layer)]

        batch_size, seqlen = query_layer.shape[0], query_layer.shape[1]

        max_seqlen = seqlen
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seqlen,
            step=seqlen,
            dtype=torch.int32,
            device=query_layer.device)

        if cp_group is None or get_distributed_world_size(cp_group) == 1:
            # [b, sq, np, hn]
            query_layer, key_layer, value_layer = [
                x.view(x.shape[0] * x.shape[1], *x.shape[2:])
                for x in [query_layer, key_layer, value_layer]
            ]

            with self.attention_dropout_ctx():
                fa_optional_forward_kwargs = {}
                if not _flash_attn_2_available:
                    fa_optional_forward_kwargs["deterministic"] = self.deterministic
                output = flash_attn_forward_func(
                    query_layer, key_layer, value_layer,
                    cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                    self.attention_dropout if self.training else 0.0,
                    softmax_scale=1.0/self.norm_factor,
                    causal=self.attn_causal_mask,
                    **fa_optional_forward_kwargs
                )
        else:
            if _flash_attn_2_available:
                assert False, "Context parallelism is only implemented with Flash Attention v1!"
            with self.attention_dropout_ctx():
                output = flash_attn_forward_func_with_cp(
                    query_layer, key_layer, value_layer,
                    cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                    self.attention_dropout if self.training else 0.0,
                    cp_group, cp_global_ranks, cp_stream,
                    softmax_scale=1.0/self.norm_factor,
                    causal=self.attn_causal_mask,
                    deterministic=self.deterministic
                )

        # [(b sq), np, hn] -> [sq, b, (np hn)]
        return output.view(batch_size, seqlen, -1).transpose(0, 1).contiguous()


class FusedAttnFunc_qkvpacked(torch.autograd.Function):
    """Function for FusedAttention with packed QKV input"""

    @staticmethod
    def forward(ctx, is_training, max_seqlen, cu_seqlens, qkv, qkv_dtype, attn_bias, attn_scale,
                dropout_p, fast_zero_fill, qkv_layout, attn_bias_type, attn_mask_type,
                rng_gen, fused_attention_backend, use_FAv2_bwd):
        out, aux_ctx_tensors = fused_attn_fwd_qkvpacked(
            is_training, max_seqlen, cu_seqlens, qkv, qkv_dtype,
            fused_attention_backend, attn_bias,
            None, None, None, None, None,
            attn_scale, dropout_p, fast_zero_fill, qkv_layout, attn_bias_type, attn_mask_type,
            rng_gen)

        ctx.save_for_backward(qkv, out, cu_seqlens)
        ctx.aux_ctx_tensors = aux_ctx_tensors
        ctx.max_seqlen = max_seqlen
        ctx.qkv_dtype = qkv_dtype
        ctx.attn_scale = attn_scale
        ctx.dropout_p = dropout_p
        ctx.fast_zero_fill = fast_zero_fill
        ctx.qkv_layout = qkv_layout
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.fused_attention_backend = fused_attention_backend
        ctx.use_FAv2_bwd = use_FAv2_bwd

        return out

    @staticmethod
    def backward(ctx, d_out):
        qkv, out, cu_seqlens = ctx.saved_tensors
        if ctx.use_FAv2_bwd:
            softmax_lse, rng_state = ctx.aux_ctx_tensors
            dqkv = torch.empty_like(qkv)
            maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
            d_out, q, k, v, out = [maybe_contiguous(x)
                for x in (d_out, qkv[:,0], qkv[:,1], qkv[:,2], out)]
            flash_attn_cuda_bwd(
                d_out, q, k, v, out, softmax_lse, dqkv[:,0], dqkv[:,1], dqkv[:,2],
                cu_seqlens, cu_seqlens, ctx.max_seqlen, ctx.max_seqlen,
                ctx.dropout_p, ctx.attn_scale, False,
                ctx.attn_mask_type == "causal", None, rng_state
            )
            dqkv = dqkv[..., :d_out.shape[-1]]
        else:
            dqkv, *rest = fused_attn_bwd_qkvpacked(
                ctx.max_seqlen, cu_seqlens, qkv, out, d_out,
                ctx.qkv_dtype, ctx.aux_ctx_tensors,
                ctx.fused_attention_backend,
                None, None, None, None, None, None, None, None, None,
                ctx.attn_scale, ctx.dropout_p, ctx.fast_zero_fill,
                ctx.qkv_layout, ctx.attn_bias_type, ctx.attn_mask_type)

        # if no_bias, return dqkv
        if ctx.attn_bias_type == "no_bias":
            return (None, None, None, dqkv, None, None, None,
                    None, None, None, None, None, None,
                    None, None, None, None, None, None)
        # else, return (dqkv, dbias)
        return (None, None, None, dqkv, None, rest[0], None,
                None, None, None, None, None, None,
                None, None, None, None, None, None)

class FusedAttnFunc_kvpacked(torch.autograd.Function):
    """Function for FusedAttention with packed KV input"""

    @staticmethod
    def forward(ctx, is_training, max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv,
                q, kv, qkv_dtype, attn_bias, attn_scale, dropout_p, fast_zero_fill,
                qkv_layout, attn_bias_type, attn_mask_type,
                rng_gen, fused_attention_backend, use_FAv2_bwd):
        out, aux_ctx_tensors = fused_attn_fwd_kvpacked(
            is_training, max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv,
            q, kv, qkv_dtype, fused_attention_backend, attn_bias,
            None, None, None, None, None,
            attn_scale, dropout_p, fast_zero_fill, qkv_layout, attn_bias_type, attn_mask_type,
            rng_gen)

        ctx.save_for_backward(q, kv, out, cu_seqlens_q, cu_seqlens_kv)
        ctx.aux_ctx_tensors = aux_ctx_tensors
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.qkv_dtype = qkv_dtype
        ctx.attn_scale = attn_scale
        ctx.dropout_p = dropout_p
        ctx.fast_zero_fill = fast_zero_fill
        ctx.qkv_layout = qkv_layout
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.fused_attention_backend = fused_attention_backend
        ctx.use_FAv2_bwd = use_FAv2_bwd

        return out

    @staticmethod
    def backward(ctx, d_out):
        q, kv, out, cu_seqlens_q, cu_seqlens_kv = ctx.saved_tensors
        if ctx.use_FAv2_bwd:
            softmax_lse, rng_state = ctx.aux_ctx_tensors
            dq = torch.empty_like(q)
            dkv = torch.empty_like(kv)
            maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
            d_out, q, k, v, out = [maybe_contiguous(x)
                for x in (d_out, q, kv[:,0], kv[:,1], out)]
            flash_attn_cuda_bwd(
                d_out, q, k, v, out, softmax_lse, dq, dkv[:,0], dkv[:,1],
                cu_seqlens_q, cu_seqlens_kv, ctx.max_seqlen_q, ctx.max_seqlen_kv,
                ctx.dropout_p, ctx.attn_scale, False,
                ctx.attn_mask_type == "causal", None, rng_state
            )
            dq = dq[..., :d_out.shape[-1]]
            dkv = dkv[..., :d_out.shape[-1]]
        else:
            dq, dkv, *rest = fused_attn_bwd_kvpacked(
                ctx.max_seqlen_q, ctx.max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv,
                q, kv, out, d_out,
                ctx.qkv_dtype, ctx.aux_ctx_tensors,
                ctx.fused_attention_backend,
                None, None, None, None, None, None, None, None, None,
                ctx.attn_scale, ctx.dropout_p, ctx.fast_zero_fill,
                ctx.qkv_layout, ctx.attn_bias_type, ctx.attn_mask_type)

        # if no_bias, return dqkv
        if ctx.attn_bias_type == "no_bias":
            return (None, None, None, None, None, dq, dkv, None, None, None,
                    None, None, None, None, None, None,
                    None, None, None, None, None, None)
        # else, return (dqkv, dbias)
        return (None, None, None, None, None, dq, dkv, None, rest[0], None,
                None, None, None, None, None, None,
                None, None, None, None, None, None)

class FusedAttention(torch.nn.Module):
    """Dot product attention, with multiple backends:

    1. FusedAttnBackend["F16_max512_seqlen"]
       cuDNN based fused attention for FP16/BF16 and <=512 sequence length.
    2. FusedAttnBackend["F16_arbitrary_seqlen"]
       cuDNN based fused attention for FP16/BF16 and any sequence length.

    Support matrix:

    | backend       | 1                       | 2               |
    | flash based   | no                      | yes             |
    | cuDNN based   | yes                     | yes             |
    | qkv dtype     | fp16/bf16               | fp16/bf16       |
    | attn_type     | self/cross              | self            |
    | qkv_layout    |                         |                 |
    |  - qkv        | qkv_interleaved         | qkv_interleaved |
    |  - (q,kv)     | kv_interleaved          |                 |
    | mask_type     | causal/no_mask          | causal          |
    | bias_type     | no_bias/post_scale_bias | no_bias         |
    | dropout       | yes                     | yes             |
    | max_seqlen    | <=512                   | any             |
    | head_dim      | 64                      | 64,128          |
    | output dtype  | fp16/bf16               | fp16/bf16       |
    """

    def __init__(
        self,
        norm_factor: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attn_mask_type: str = "causal",
        attention_type: str = "self",
    ) -> None:
        super().__init__()

        self.norm_factor = norm_factor
        self.attention_dropout = attention_dropout
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.use_FAv2_bwd = (os.getenv("NVTE_FUSED_ATTN_USE_FAv2_BWD", "1") == "1"
                        and _flash_attn_2_available
                        and get_device_compute_capability() == 9.0)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        fused_attention_backend:
            tex.NVTE_Fused_Attn_Backend = tex.NVTE_Fused_Attn_Backend.NVTE_No_Backend,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
    ) -> torch.Tensor:
        """fused attention fprop"""

        assert (fused_attention_backend
                != tex.NVTE_Fused_Attn_Backend.NVTE_No_Backend
                ), 'No fused attention backend supports this input combination!'
        assert (
            (query_layer.dtype in [torch.float16, torch.bfloat16])
            and (key_layer.dtype in [torch.float16, torch.bfloat16])
            and (value_layer.dtype in [torch.float16, torch.bfloat16])
            ), 'FusedAttention only supports FP16 and BF16 data types.'
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
            ), 'FusedAttention only supports CUDA tensors.'

        qkv_dtype = TE_DType[query_layer.dtype]
        seqlen_q, batch_size = query_layer.shape[0], query_layer.shape[1]
        seqlen_kv = key_layer.shape[0]
        max_seqlen_q = seqlen_q
        max_seqlen_kv = seqlen_kv

        if self.attention_type == "self":
            qkv_layout = _check_qkv_layout(query_layer, key_layer, value_layer)
            if qkv_layout == "sbh3d":
                mixed_layer = _CombineQKV.apply(query_layer, key_layer, value_layer, 3)
                # [s, b, h, 3, d]
                mixed_layer = mixed_layer.view(
                        *mixed_layer.shape[0:3], 3, query_layer.shape[-1])
                # [b, s, 3, h, d]
                mixed_layer = mixed_layer.transpose(2, 3).transpose(0, 1).contiguous()
            elif qkv_layout == "sb3hd":
                mixed_layer = _CombineQKV.apply(query_layer, key_layer, value_layer, 2)
                # [s, b, 3, h, d]
                mixed_layer = mixed_layer.view(
                        *mixed_layer.shape[0:2], 3, *query_layer.shape[2:])
                # [b, s, 3, h, d]
                mixed_layer = mixed_layer.transpose(0, 1).contiguous()
            else:
                raise Exception("FusedAttention only supports qkv layout sbh3d or sb3hd!")

            # [total_seqs, 3, h, d]
            mixed_layer = mixed_layer.view(
                mixed_layer.shape[0] * mixed_layer.shape[1], *mixed_layer.shape[2:])

            qkv_layout = "qkv_interleaved"
            max_seqlen = seqlen_q
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seqlen_q,
                step=seqlen_q,
                dtype=torch.int32,
                device=query_layer.device)
            use_FAv2_bwd = (self.use_FAv2_bwd
                        and (fused_attention_backend
                            == tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen)
                        and core_attention_bias_type == "no_bias")

            with self.attention_dropout_ctx():
                output = FusedAttnFunc_qkvpacked.apply(
                    self.training,
                    max_seqlen,
                    cu_seqlens,
                    mixed_layer,
                    qkv_dtype,
                    core_attention_bias,
                    1.0/self.norm_factor,
                    self.attention_dropout if self.training else 0.0,
                    fast_zero_fill,
                    qkv_layout,
                    core_attention_bias_type,
                    self.attn_mask_type,
                    None, # rng_gen
                    fused_attention_backend,
                    use_FAv2_bwd
                )
            output = output.view(batch_size, seqlen_q, -1).transpose(0, 1).contiguous()

        if self.attention_type == "cross":
            kv_layout = _check_kv_layout(key_layer, value_layer)
            if kv_layout == "sbh2d":
                key_value = _CombineKV.apply(key_layer, value_layer, 3)
                # [s, b, h, 2, d]
                key_value = key_value.view(
                        *key_value.shape[0:3], 2, key_layer.shape[-1])
                # [b, s, 2, h, d]
                key_value = key_value.transpose(2, 3).transpose(0, 1).contiguous()
            elif qkv_layout == "sb2hd":
                key_value = _CombineKV.apply(key_layer, value_layer, 2)
                # [s, b, 2, h, d]
                key_value = key_value.view(
                        *key_value.shape[0:2], 2, *key_layer.shape[2:])
                # [b, s, 2, h, d]
                key_value = key_value.transpose(0, 1).contiguous()
            else:
                raise Exception("FusedAttention only supports kv layout sbh2d or sb2hd!")

            # [total_seqs, h, d]
            query_layer = query_layer.transpose(0, 1).contiguous()
            query_layer = query_layer.view(
                    query_layer.shape[0] * query_layer.shape[1], *query_layer.shape[2:])
            # [total_seqs, 2, h, d]
            key_value = key_value.view([key_value.shape[0] * key_value.shape[1]]
                + key_value.shape[2:])

            qkv_layout = "kv_interleaved"
            cu_seqlens_q = torch.arange(
                0,
                (batch_size + 1) * seqlen_q,
                step=seqlen_q,
                dtype=torch.int32,
                device=query_layer.device)
            cu_seqlens_kv = torch.arange(
                0,
                (batch_size + 1) * seqlen_kv,
                step=seqlen_kv,
                dtype=torch.int32,
                device=key_layer.device)

            with self.attention_dropout_ctx():
                outputs = FusedAttnFunc_kvpacked.apply(
                    self.training,
                    max_seqlen_q, max_seqlen_kv,
                    cu_seqlens_q, cu_seqlens_kv,
                    query_layer, key_value,
                    qkv_dtype,
                    core_attention_bias,
                    1.0/self.norm_factor,
                    self.attention_dropout if self.training else 0.0,
                    fast_zero_fill,
                    qkv_layout,
                    core_attention_bias_type,
                    self.attn_mask_type,
                    None, # rng_gen
                    fused_attention_backend,
                    use_FAv2_bwd
                )

            output = (outputs[0].view(batch_size, seqlen_q, -1).transpose(0, 1).contiguous(),
                    outputs[1].view(batch_size, seqlen_q, -1).transpose(0, 1).contiguous())
        return output


class DotProductAttention(torch.nn.Module):
    """Allows the model to jointly attend to information from different
    representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. note::

        Argument :attr:`attention_mask` will be ignored in the `forward` call when
        :attr:`attn_mask_type` is set to `"causal"`.

    .. warning::

        FlashAttention uses a non-deterministic algorithm for optimal performance. To observe
        deterministic behavior at the cost of performance, use FlashAttention version < `2.0.0`
        and set the environment variable :attr:`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`. In order
        to disable`flash-attn` entirely, set :attr:`NVTE_FLASH_ATTN=0`.

    Parameters
    ----------
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels : int
                number of key-value channels.
    num_gqa_groups : Optional[int] = None
                    number of GQA groups in the transformer layer.
                    Grouped Query Attention is described in
                    `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                    This only affects the keys and values, not the queries.
                    GQA-1 is equivalent to Multi-Query Attention
                    (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                    is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    attention_dropout: float, default = 0.0
                      dropout probability for the dropout op during multi-head attention.
    attn_mask_type: {'causal', 'padding', 'no_mask'}, default = `causal`
                   type of attention mask passed into softmax operation.
    layer_number: int, default = `None`
                 layer number of the current `DotProductAttention` when multiple such modules
                 are concatenated, for instance in consecutive transformer blocks.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_size : int, default = 1
             tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    cp_group : ProcessGroup, default = `None`
             context parallel process group.
    cp_global_ranks: list of global rank IDs, default = `None`
             global rank IDs of GPUs that are in cp_group.
    cp_stream: CUDA stream, default = `None`
              context parallelism splits flash attention into multiple steps for compute and communication
              overlapping. To address the wave quantization issue of each split step, we add an additional
              CUDA stream so that we can overlap two flash attention kernels.
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        num_gqa_groups: Optional[int] = None,
        attention_dropout: float = 0.0,
        attn_mask_type: str = "causal",
        sequence_parallel: bool = False,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        attention_type: str = "self",
        cp_group: Optional[dist_group_type] = None,
        cp_global_ranks: Union[int] = None,
        cp_stream: torch.cuda.Stream = None,
    ) -> None:
        super().__init__()

        self.tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.tp_group = tp_group
        self.get_rng_state_tracker = get_rng_state_tracker
        self.num_attention_heads = num_attention_heads
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream

        self.hidden_size_per_attention_head = kv_channels
        self.num_gqa_groups = (
            num_attention_heads if num_gqa_groups is None else num_gqa_groups
        )
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // tp_size)

        assert (num_attention_heads % self.num_gqa_groups == 0
                ), "The number of attention heads must be divisible by the number of GQA groups!"

        if sequence_parallel or get_rng_state_tracker is None:
            attention_dropout_ctx = nullcontext
        else:
            attention_dropout_ctx = get_rng_state_tracker().fork

        norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        self.device_compute_capability = get_device_compute_capability()
        self.deterministic = not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))

        self.use_flash_attention = (
            int(os.getenv("NVTE_FLASH_ATTN", "1"))
            and self.device_compute_capability >= 8.0
        )
        if _flash_attn_2_available and self.deterministic:
            self.use_flash_attention = False
            warnings.warn(
                "Disabling usage of FlashAttention since version 2 does not support deterministic"
                "execution. In order to use FA with deterministic behavior, please install"
                "FlashAttention version 1."
            )

        self.use_fused_attention = (
            int(os.getenv("NVTE_FUSED_ATTN", "1"))
            and self.device_compute_capability >= 8.0
        )

        attn_kwargs = {
            "attention_dropout": attention_dropout,
            "attention_dropout_ctx": attention_dropout_ctx,
            "attn_mask_type": attn_mask_type,
        }
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.attention_dropout = attention_dropout

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(
                norm_factor, **attn_kwargs,
                deterministic=self.deterministic)
        # Instantiating three types since use of flash-attn and FusedAttention
        # might be ruled out due to forward inputs.
        if self.use_fused_attention:
            self.fused_attention = FusedAttention(
                norm_factor, **attn_kwargs,
                attention_type = attention_type)
        self.unfused_attention = UnfusedDotProductAttention(
            norm_factor, **attn_kwargs, layer_number=layer_number)

    def _checkpointed_attention_forward(
        self,
        attention_func: Callable,
        *forward_args: Tuple[torch.Tensor, ...],
        **forward_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Forward method with activation checkpointing."""

        def custom_forward(*input_args, **input_kwargs):
            return attention_func(*input_args, **input_kwargs)

        hidden_states = checkpoint(
            custom_forward,
            False,
            self.get_rng_state_tracker,
            self.tp_group,
            *forward_args,
            **forward_kwargs,
        )

        return hidden_states

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        checkpoint_core_attention: bool = False,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
    ) -> torch.Tensor:
        """
        Dot Product Attention Layer.

        .. note::

            Argument :attr:`attention_mask` will be ignored when :attr:`attn_mask_type`
            is set to `"causal"`.

        .. note::

            Input tensors :attr:`query_layer`, :attr:`key_layer`, and :attr:`value_layer`
            must each be of shape (:attr:`sequence_length`, :attr:`batch_size`,
            :attr:`num_attention_heads`, :attr:`kv_channels`). Output of shape
            (:attr:`sequence_length`, :attr:`batch_size`, :attr:`num_attention_heads`
            * :attr:`kv_channels`) is returned.

        .. note::

            `DotProductAttention` supports three backends: 1) `FlashAttention` which calls
            HazyResearch's FlashAttention PyTorch API, 2) `FusedAttention` which has multiple
            fused attention implementations as its backends (see `FusedAttention` for
            more details), and 3) `UnfusedDotProductAttention` which is the native PyTorch
            implementation with fused scaled masked softmax. Users can use environment variables
            `NVTE_FLASH_ATTN`, `NVTE_FUSED_ATTN`, and `NVTE_FUSED_ATTN_BACKEND` to control
            which DotProductAttention backend, and FusedAttention backend if applicable, to use.
            The default DotProductAttention backend is 1.

        Parameters
        ----------
        query_layer : torch.Tensor
                     Query tensor.
        key_layer : torch.Tensor
                   Key tensor.
        value_layer : torch.Tensor
                     Value tensor.
        attention_mask : Optional[torch.Tensor], default = `None`
                        Boolean tensor used to mask out softmax input when not using flash-attn.
        checkpoint_core_attention : bool, default = `False`
                                   If true, forward activations for attention are recomputed
                                   during the backward pass in order to save memory that would
                                   otherwise be occupied to store the forward activations until
                                   backprop.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, 'post_scale_bias`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T
        fast_zero_fill: bool, defautl = `True`
                    Whether to use the fast path to set output tensors to 0 or not.
        """

        assert (key_layer.shape[-2] == self.num_gqa_groups_per_partition
                and value_layer.shape[-2] == self.num_gqa_groups_per_partition
                ), f"Keys and values must have {self.num_gqa_groups} heads!"

        use_flash_attention = self.use_flash_attention
        use_fused_attention = self.use_fused_attention

        if (query_layer.dtype not in [torch.bfloat16, torch.float16]
            or key_layer.dtype not in [torch.bfloat16, torch.float16]
            or value_layer.dtype not in [torch.bfloat16, torch.float16]
        ):
            use_flash_attention = False

        if key_layer.shape[-1] > 64:
            if self.device_compute_capability in (8.6, 8.7):
                use_flash_attention = False
            elif not _flash_attn_2_available and self.device_compute_capability == 8.9:
                use_flash_attention = False

        if not _flash_attn_2_available and self.num_gqa_groups != self.num_attention_heads:
            use_flash_attention = False

        if self.attn_mask_type == "padding" and attention_mask is not None:
            use_flash_attention = False
            use_fused_attention = False

        if core_attention_bias_type != "no_bias" or core_attention_bias is not None:
            use_flash_attention = False

        if is_in_onnx_export_mode():
            use_flash_attention = False
            use_fused_attention = False

        qkv_layout = "qkv_interleaved" if self.attention_type == "self" else "kv_interleaved"

        if use_fused_attention:
            fused_attention_backend = tex.get_fused_attn_backend(
                TE_DType[query_layer.dtype],
                TE_DType[key_layer.dtype],
                QKVLayout[qkv_layout],
                AttnBiasType[core_attention_bias_type],
                AttnMaskType[self.attn_mask_type],
                self.attention_dropout,
                query_layer.shape[0], key_layer.shape[0],
                query_layer.shape[-1])
            # DPA does not support FP8; for FP8, use cpp_extensions modules directly
            is_backend_avail = (fused_attention_backend in
                [FusedAttnBackend["F16_max512_seqlen"], FusedAttnBackend["F16_arbitrary_seqlen"]])
            use_fused_attention = (use_fused_attention
                                  and is_backend_avail
                                  and self.num_gqa_groups == self.num_attention_heads)
            if (self.deterministic
                and fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]):
                use_fused_attention = False
                warnings.warn(
                    "Disabling usage of FusedAttention since the FusedAttention"
                    "backend does not support deterministic exection."
                )

        if use_flash_attention:
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(self.flash_attention,
                                                            query_layer,
                                                            key_layer,
                                                            value_layer,
                                                            self.cp_group,
                                                            self.cp_global_ranks,
                                                            self.cp_stream)
            return self.flash_attention(query_layer,
                                        key_layer,
                                        value_layer,
                                        self.cp_group,
                                        self.cp_global_ranks,
                                        self.cp_stream)

        assert (
            self.cp_group is None or get_distributed_world_size(self.cp_group) == 1
        ), "Context parallelism is only implemented with Flash Attention!"

        if use_fused_attention:
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(self.fused_attention,
                              query_layer,
                              key_layer,
                              value_layer,
                              fused_attention_backend = fused_attention_backend,
                              core_attention_bias_type = core_attention_bias_type,
                              core_attention_bias = core_attention_bias,
                              fast_zero_fill = fast_zero_fill)
            return self.fused_attention(query_layer, key_layer, value_layer,
                              fused_attention_backend = fused_attention_backend,
                              core_attention_bias_type = core_attention_bias_type,
                              core_attention_bias = core_attention_bias,
                              fast_zero_fill = fast_zero_fill)

        if checkpoint_core_attention:
            return self._checkpointed_attention_forward(
                self.unfused_attention,
                query_layer,
                key_layer,
                value_layer,
                attention_mask = attention_mask,
                core_attention_bias_type = core_attention_bias_type,
                core_attention_bias = core_attention_bias,
            )
        return self.unfused_attention(query_layer,
                key_layer,
                value_layer,
                attention_mask = attention_mask,
                core_attention_bias_type = core_attention_bias_type,
                core_attention_bias = core_attention_bias,
        )


class MultiheadAttention(torch.nn.Module):
    r"""
    Multi-head Attention (MHA), including Query,
    Key, Value and Output projection.

    .. note::

        Argument :attr:`attention_mask` will be ignored in the `forward` call when
        :attr:`self_attn_mask_type` is set to `"causal"`.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels: int, default = `None`
                number of key-value channels. defaults to
                :attr:`hidden_size` / :attr:`num_attention_heads` if `None`.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    layernorm_epsilon : float, default = 1e-5
                       a value added to the denominator of layer normalization
                       for numerical stability.
    init_method : Callable, default = `None`
                 used for initializing weights of QKV and FC1 weights in the following way:
                 `init_method(weight)`. When set to `None`, defaults to
                 `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing weights of PROJ and FC2 in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    layer_number: int, default = `None`
                 layer number of the current `TransformerLayer` when multiple such modules are
                 concatenated to form a transformer block.
    attn_mask_type: {'causal', 'padding', 'no_mask'}, default = `causal`
                   type of attention mask passed into softmax operation.
    num_gqa_groups : int, default = `None`
                         number of GQA groups in the transformer layer.
                         Grouped Query Attention is described in
                         `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                         This only affects the keys and values, not the querys.
                         GQA-1 is equivalent to Multi-Query Attention
                         (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                         is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
    input_layernorm: bool, default = `True`
                     if set to `False`, layer normalization to the input is not applied.
    attention_type: { 'self', 'cross' }, default = 'self'
                   type of attention applied.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    qkv_weight_interleaved : bool, default = `True`
                            if set to `False`, the QKV weight is interpreted as a concatenation of
                            query, key, and value weights along the `0th` dimension. The default
                            interpretation is that the individual `q`, `k`, and `v` weights for each
                            attention head are interleaved. This parameter is set to `False` when
                            using :attr:`fuse_qkv_params=False`.
    bias : bool, default = `True`
          if set to `False`, the transformer layer will not learn any additive biases.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, QKV and FC1 layers are used as Column Parallel
                      whereas PROJ and FC2 is used as Row Parallel as described
                      `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    fuse_qkv_params: bool, default = 'False'
                    if set to `True`, `TransformerLayer` module exposes a single fused
                    parameter for query-key-value. This enables optimizations such as QKV
                    fusion without concatentations/splits and also enables the argument
                    `fuse_wgrad_accumulation`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: Optional[int] = None,
        attention_dropout: float = 0.1,
        layernorm_epsilon: float = 1e-5,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        layer_number: Optional[int] = None,
        attn_mask_type: str = "causal",
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        num_gqa_groups: Optional[int] = None,
        fuse_wgrad_accumulation: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        sequence_parallel: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_bias: bool = False,
        return_layernorm_output: bool = False,
        input_layernorm: bool = False,
        attention_type: str = "self",
        set_parallel_mode: bool = False,
        fuse_qkv_params: bool = False,
        zero_centered_gamma: bool = False,
        qkv_weight_interleaved: bool = True,
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_split_rs: bool = False,
        ub_split_ag: bool = False,
        bias: bool = True,
        normalization: str = "LayerNorm",
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()
        self.layer_number = layer_number
        self.input_layernorm = input_layernorm
        self.attention_type = attention_type
        self.get_rng_state_tracker = get_rng_state_tracker
        self.tp_group = tp_group
        self.return_layernorm_output = return_layernorm_output
        self.params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.attn_mask_type = attn_mask_type
        self.num_attention_heads = num_attention_heads
        self.return_bias = return_bias

        kv_channels = kv_channels if kv_channels else (hidden_size // num_attention_heads)

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        if not fuse_qkv_params:
            qkv_weight_interleaved = False
        self.qkv_weight_interleaved = qkv_weight_interleaved

        assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"
        if layer_number is not None:
            assert layer_number > 0, "layer_number must be a positive integer"

        tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.tp_size = tp_size
        self.sequence_parallel = (tp_size > 1) and sequence_parallel

        self.hidden_size_per_attention_head = kv_channels
        self.num_attention_heads_per_partition = divide(num_attention_heads, tp_size)
        self.num_gqa_groups = (
            num_attention_heads if num_gqa_groups is None else num_gqa_groups
        )
        assert (num_attention_heads % self.num_gqa_groups == 0
                ), "The number of GQA groups must be divisible by the number of attention heads!"
        assert (num_attention_heads % tp_size == 0
                ), "The number of GQA groups must be divisible by tensor parallel size!"
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // tp_size)
        self.hidden_size_kv = int(hidden_size * self.num_gqa_groups // num_attention_heads)

        common_gemm_kwargs = {
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "get_rng_state_tracker": get_rng_state_tracker,
            "sequence_parallel": sequence_parallel,
            "params_dtype": self.params_dtype,
            "device": device,
        }

        qkv_parallel_mode = "column" if set_parallel_mode else None

        if self.attention_type == "self" and self.num_gqa_groups == self.num_attention_heads:
            if self.input_layernorm:
                self.layernorm_qkv = LayerNormLinear(
                    hidden_size,
                    3 * hidden_size,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    return_layernorm_output=return_layernorm_output,
                    parameters_split=("query_", "key_", "value_") if not fuse_qkv_params else None,
                    zero_centered_gamma=zero_centered_gamma,
                    ub_bulk_wgrad=ub_bulk_wgrad,
                    ub_bulk_dgrad=ub_bulk_dgrad,
                    ub_split_ag=ub_split_ag,
                    normalization=normalization,
                    **common_gemm_kwargs,
                )
            else:
                self.qkv = Linear(
                    hidden_size,
                    3 * hidden_size,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    parameters_split=("query_", "key_", "value_") if not fuse_qkv_params else None,
                    **common_gemm_kwargs,
                )
        elif ((self.attention_type == "cross")
                or (self.attention_type == "self"
                    and self.num_gqa_groups != self.num_attention_heads)):
            if self.input_layernorm:
                self.layernorm_query = LayerNormLinear(
                    hidden_size,
                    hidden_size,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    return_layernorm_output=return_layernorm_output,
                    zero_centered_gamma=zero_centered_gamma,
                    ub_bulk_wgrad=ub_bulk_wgrad,
                    ub_bulk_dgrad=ub_bulk_dgrad,
                    ub_split_ag=ub_split_ag,
                    normalization=normalization,
                    **common_gemm_kwargs,
                )
            else:
                self.query_layer = Linear(
                    hidden_size,
                    hidden_size,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    **common_gemm_kwargs,
                )
            self.key_value = Linear(
                hidden_size,
                2 * self.hidden_size_kv,
                init_method=init_method,
                bias=bias,
                return_bias=False,
                parallel_mode=qkv_parallel_mode,
                parameters_split=("key_", "value_") if not fuse_qkv_params else None,
                **common_gemm_kwargs,
            )

        # Attention.
        self.core_attention = DotProductAttention(
            num_attention_heads,
            kv_channels,
            num_gqa_groups=self.num_gqa_groups,
            attention_dropout=attention_dropout,
            tp_size=tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            attn_mask_type=attn_mask_type,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            layer_number=self.layer_number,
        )

        # Linear
        self.proj = Linear(
            hidden_size,
            hidden_size,
            init_method=output_layer_init_method,
            bias=bias,
            return_bias=return_bias,
            parallel_mode="row" if set_parallel_mode else None,
            ub_split_rs=ub_split_rs,
            ub_split_ag=ub_split_ag,
            **common_gemm_kwargs,
        )


    def _allocate_memory(
        self, inference_max_sequence_len: int, batch_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_gqa_groups_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """Set TP group"""
        self.tp_group = tp_group

    def set_context_parallel_running(
        self,
        cp_group: Union[dist_group_type, None],
        cp_global_ranks: Union[int],
        cp_stream: torch.cuda.Stream,
    ) -> None:
        """Set CP group and CP dual-stream running"""
        self.core_attention.cp_group = cp_group
        self.core_attention.cp_global_ranks = cp_global_ranks
        self.core_attention.cp_stream = cp_stream

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: bool = False,
        inference_params: Optional[Any] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Forward propagation for MultiheadAttention layer.

        .. note::

            Argument :attr:`attention_mask` will be ignored when :attr:`self_attn_mask_type`
            is set to `"causal"`.

        Parameters
        ----------
        hidden_states : torch.Tensor
             Input tensor.
        attention_mask : Optional[torch.Tensor], default = `None`
             Boolean tensor used to mask out self-attention softmax input.
        encoder_output : Optional[torch.Tensor], default = `None`
             Output of the encoder block to be fed into the decoder block if using
             `layer_type="decoder"`.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        checkpoint_core_attention: bool, default = `False`
                                  If true, forward activations for core attention are recomputed
                                  during the backward pass in order to save memory that would
                                  otherwise be occupied to store the forward activations until
                                  backprop.
        rotary_pos_emb: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], default = `None`
                       Embeddings for query and key tensors for applying rotary position
                       embedding. By default no input embedding is applied.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, 'post_scale_bias`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T
        fast_zero_fill: bool, default = `True`
                    Whether to set output tensors to 0 or not before use.
        """
        # hidden_states: [sq, b, h]

        if self.attn_mask_type == "padding" and attention_mask is not None:
            assert (
                attention_mask.dtype == torch.bool
            ), "Attention mask must be a boolean tensor"

        assert (core_attention_bias_type in AttnBiasTypes
                ), f"core_attention_bias_type {core_attention_bias_type} is not supported!"
        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================

        is_first_step = False
        if inference_params and self.layer_number is not None:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size, hidden_states.dtype
                )
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size, hidden_states.dtype
                )
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory,
                    inference_value_memory,
                )
                is_first_step = True
            else:
                (
                    inference_key_memory,
                    inference_value_memory,
                ) = inference_params.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == "self" and self.num_gqa_groups == self.num_attention_heads:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            if self.input_layernorm:
                layernorm_qkv_outputs = self.layernorm_qkv(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )
                if self.return_layernorm_output:
                    mixed_x_layer, layernorm_output = layernorm_qkv_outputs
                else:
                    mixed_x_layer = layernorm_qkv_outputs
            else:
                mixed_x_layer = self.qkv(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )

            if self.qkv_weight_interleaved:
                # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    self.num_attention_heads_per_partition,
                    3 * self.hidden_size_per_attention_head,
                )
                # split along last dimension
                split_dim = -1
            else:
                # [sq, b, (np * 3 * hn)] --> [sq, b, 3 * np, hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    3 * self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
                # split along second last dimension
                split_dim = -2

            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # mixed_x_layer --> 3 [sq, b, np, hn]
            if split_dim == -1 and not is_in_onnx_export_mode():
                query_layer, key_layer, value_layer = _SplitLastDim.apply(mixed_x_layer, 3)
            else:
                query_layer, key_layer, value_layer = split_tensor_along_dim(
                    mixed_x_layer, split_dim, 3
                )
        elif ((self.attention_type == "cross")
                or (self.attention_type == "self"
                    and self.num_gqa_groups != self.num_attention_heads)):

            if self.attention_type == "cross":
                input_tensor = encoder_output
            else:
                input_tensor = hidden_states

            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer = self.key_value(
                input_tensor,
                is_first_microbatch=is_first_microbatch,
            )

            if self.qkv_weight_interleaved:
                # [sq, b, (np * 2 * hn)] --> [sq, b, np, 2 * hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    self.num_gqa_groups_per_partition,
                    2 * self.hidden_size_per_attention_head,
                )
                # split along last dimension
                split_dim = -1
            else:
                # [sq, b, (np * 2 * hn)] --> [sq, b, 2 * np, hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    2 * self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
                # split along second last dimension
                split_dim = -2

            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # mixed_kv_layer --> 2 [sk, b, np, hn]
            if split_dim == -1 and not is_in_onnx_export_mode():
                key_layer, value_layer = _SplitLastDim.apply(mixed_kv_layer, 2)
            else:
                key_layer, value_layer = split_tensor_along_dim(mixed_kv_layer, split_dim, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            if self.input_layernorm:
                layernorm_query_outputs = self.layernorm_query(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )
                if self.return_layernorm_output:
                    query_layer, layernorm_output = layernorm_query_outputs
                else:
                    query_layer = layernorm_query_outputs
            else:
                query_layer = self.query_layer(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )

            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = ((rotary_pos_emb,) * 2)

        if inference_params and self.layer_number is not None:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[
                sequence_start:sequence_end, batch_start:batch_end, ...
            ] = key_layer
            inference_value_memory[
                sequence_start:sequence_end, batch_start:batch_end, ...
            ] = value_layer
            key_layer = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                :sequence_end, batch_start:batch_end, ...
            ]

            # adjust the key rotary positional embedding
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # need to cross check this condition during inference
                # if not set_inference_key_value_memory:
                if not is_first_step:
                    # In inference, we compute one token at a time.
                    # Select the correct positional embedding
                    # (only the last token in the sequence)
                    q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
                else:
                    # In the first forward pass of inference,
                    # we use the entire provided prefix.
                    # q_pos_emb here has the rope embeddings of the entire
                    # prefix + to-be-generated output so
                    # we slice to just the prefix.
                    q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
                k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
                rotary_pos_emb = (q_pos_emb, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)

        context_layer = self.core_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            checkpoint_core_attention=checkpoint_core_attention,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
            fast_zero_fill=fast_zero_fill,
        )

        # =================
        # Output. [sq, b, h]
        # =================

        projection_output = self.proj(
            context_layer, is_first_microbatch=is_first_microbatch
        )

        if self.return_bias:
            attention_output, attention_bias = projection_output
        else:
            attention_output, attention_bias = projection_output, None

        outputs = (attention_output,)
        if self.return_bias:
            outputs += (attention_bias,)
        if self.input_layernorm and self.return_layernorm_output:
            outputs += (layernorm_output,)
        return outputs if len(outputs) > 1 else outputs[0]
