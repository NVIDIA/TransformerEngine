# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention."""
import os
import warnings
import math
from importlib.metadata import version
from contextlib import nullcontext
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from pkg_resources import packaging
import numpy as np

import torch
import torch.nn.functional as F

import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd_qkvpacked,
    fused_attn_bwd_qkvpacked,
    fused_attn_fwd_kvpacked,
    fused_attn_bwd_kvpacked,
    fused_attn_fwd,
    fused_attn_bwd,
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
    QKVLayouts,
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
from transformer_engine.pytorch.jit import jit_fuser

_flash_attn_version = packaging.version.Version(version("flash-attn"))
_flash_attn_version_required = packaging.version.Version("1.0.6")
_flash_attn_2_available = _flash_attn_version >= packaging.version.Version("2")

if _flash_attn_2_available:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_forward_func # pylint: disable=no-name-in-module
    from flash_attn_2_cuda import varlen_bwd as flash_attn_cuda_bwd # pylint: disable=no-name-in-module
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward as _flash_attn_forward # pylint: disable=no-name-in-module,ungrouped-imports
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward as _flash_attn_backward # pylint: disable=no-name-in-module
else:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func as flash_attn_forward_func # pylint: disable=no-name-in-module,ungrouped-imports
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward


_cu_seqlens_q, _cu_seqlens_kv, _indices_q, _indices_kv = None, None, None, None


__all__ = ["DotProductAttention", "InferenceParams", "MultiheadAttention"]


class InferenceParams: # pylint: disable=too-few-public-methods
    """
    Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference.

    Parameters
    ----------
    max_batch_size : int
                    maximum batch size during inference.
    max_sequence_length : int
                         maximum sequence length during inference.
    """

    def __init__(self, max_batch_size, max_sequence_length):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_indices):
        """
        Reorders the KV cache using the specified batch indices.

        Parameters
        ----------
        batch_indices : List[int]
                       Sequence of indices to reorder along the batch dimensions of
                       the KV cache. Must have a length equal to the batch size.
        """
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")

        for layer_number, inference_memory in self.key_value_memory_dict.items():
            inference_key_memory, inference_value_memory = inference_memory
            assert (
                len(batch_indices) == inference_key_memory.shape[1]
            )  # make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_indices]
            new_inference_value_memory = inference_value_memory[:, batch_indices]
            self.key_value_memory_dict[layer_number] = (
                new_inference_key_memory,
                new_inference_value_memory,
            )


def get_cu_seqlens_and_indices(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a padding mask of shape [batch_size, 1, 1, max_seqlen], returns an int32
    tensor of shape [batch_size + 1,] containing the cumulative sequence
    lengths of every sample in the batch and the indices containing valid
    samples.
    """
    mask = mask.squeeze(1).squeeze(1)
    bs, seqlen = mask.shape

    reduced_mask = mask.sum(dim=1)
    cu_seqlens = reduced_mask.cumsum(dim=0).to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat((zero, cu_seqlens))

    mask = mask.reshape(-1)
    indices = mask.nonzero()
    indices = indices.unsqueeze(-1)

    num_nonzeros = indices.shape[0]
    pad_amount = bs * seqlen - num_nonzeros
    indices = F.pad(input=indices, pad=(0, 0, 0, 0, 0, pad_amount),
                    mode="constant", value=float(bs * seqlen))

    return cu_seqlens, indices


@jit_fuser
def pack_tensor(
    indices: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Packs the given tensor using the `indices`.
    """
    padding_indice = torch.zeros(
        1, tensor.shape[1], tensor.shape[2], dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat((tensor, padding_indice), dim=0)

    indices = indices.repeat(1, tensor.shape[1], tensor.shape[2])
    packed = torch.gather(tensor, 0, indices)
    return packed


@jit_fuser
def pack_2_tensors(
    indices: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs the given 2 tensors using the `indices`.
    """
    t1_packed = pack_tensor(indices, t1)
    t2_packed = pack_tensor(indices, t2)
    return t1_packed, t2_packed


@jit_fuser
def pack_3_tensors(
    indices: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Packs the given 3 tensors using the `indices`.
    """
    t1_packed = pack_tensor(indices, t1)
    t2_packed = pack_tensor(indices, t2)
    t3_packed = pack_tensor(indices, t3)
    return t1_packed, t2_packed, t3_packed


@jit_fuser
def unpack_tensor(
    indices: torch.Tensor,
    dim0: int,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Inverse of `pack_tensor`.
    """
    indices = indices.repeat(1, tensor.shape[1], tensor.shape[2])
    unpacked = torch.zeros(
        dim0 + 1, tensor.shape[1], tensor.shape[2], dtype=tensor.dtype, device=tensor.device)
    unpacked.scatter_(0, indices, tensor)
    unpacked = unpacked[0:-1,:,:]
    return unpacked


@jit_fuser
def unpack_2_tensors(
    indices: torch.Tensor,
    dim0: int,
    t1: torch.Tensor,
    t2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of `pack_2_tensors`.
    """
    t1_unpacked = unpack_tensor(indices, dim0, t1)
    t2_unpacked = unpack_tensor(indices, dim0, t2)
    return t1_unpacked, t2_unpacked


@jit_fuser
def unpack_3_tensors(
    indices: torch.Tensor,
    dim0: int,
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inverse of `pack_3_tensors`.
    """
    t1_unpacked = unpack_tensor(indices, dim0, t1)
    t2_unpacked = unpack_tensor(indices, dim0, t2)
    t3_unpacked = unpack_tensor(indices, dim0, t3)
    return t1_unpacked, t2_unpacked, t3_unpacked


class PackTensors(torch.autograd.Function):
    """
    Autograd function to pack tensors.
    """
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        *tensors: Tuple[torch.Tensor, ...]
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        assert 1 <= len(tensors) <= 3, f"Packing {len(tensors)} tensors not supported."
        ctx.indices = indices
        ctx.dim0 = tensors[0].shape[0]
        if len(tensors) == 1:
            return pack_tensor(indices, *tensors)
        if len(tensors) == 2:
            return pack_2_tensors(indices, *tensors)
        return pack_3_tensors(indices, *tensors)

    @staticmethod
    def backward(ctx, *grad_outputs: Tuple[torch.Tensor, ...]):
        if len(grad_outputs) == 1:
            return None, unpack_tensor(ctx.indices, ctx.dim0, *grad_outputs)
        if len(grad_outputs) == 2:
            return None, *unpack_2_tensors(ctx.indices, ctx.dim0, *grad_outputs)
        return None, *unpack_3_tensors(ctx.indices, ctx.dim0, *grad_outputs)


class UnpackTensor(torch.autograd.Function):
    """
    Autograd function to unpack a tensor.
    """
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        dim0: int,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        ctx.indices = indices
        return unpack_tensor(indices, dim0, tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, pack_tensor(ctx.indices, grad_output)


def _unpack_attn_mask_type(attn_mask_type: str) -> Tuple[str, bool]:
    """
    Unpacks the attention mask type string and returns a single mask type
    and a boolean for whether to apply causal mask. Also ensures that the
    combination of masks passed in is supported by one of the attention
    backends available.
    """
    mask_types = attn_mask_type.split(',')
    assert (
        all(mask_type in AttnMaskTypes for mask_type in mask_types)
    ), f"Mask type {attn_mask_type} is not supported."

    # Whether or not to apply causal mask toggle.
    causal_mask = False
    if "causal" in mask_types:
        mask_types.remove("causal")
        causal_mask = True

    if len(mask_types) == 0:  # Only apply causal mask.
        return "causal", True
    if len(mask_types) == 1 and causal_mask:  # Causal + padding masks
        assert mask_types[0] == "padding", f"Causal + {mask_types[0]} masking not supported."
        return "padding", True
    if len(mask_types) == 1:  # Arbitrary or padding or no_mask
        return mask_types[0], False
    raise RuntimeError("Unsupported combination of mask types.")


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
        if _flash_attn_2_available:
            assert(q.shape[-1] % 8 == 0), "hidden size per attention head should be multiple of 8"
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
                            # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                            kv_inputs[i%2] = kv_inputs[i%2].view(2, -1, *k.shape[-2:])
                            if _flash_attn_2_available:
                                _, _, _, _, out_per_step[i], \
                                softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                                    q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                    dropout_p, softmax_scale, causal=True, return_softmax=False,
                                )
                            else:
                                out_per_step[i] = torch.empty_like(q_inputs[i%2])
                                _, softmax_lse_per_step[i], rng_states[i], _ = _flash_attn_forward( # pylint: disable=unbalanced-tuple-unpacking
                                    q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                    out_per_step[i], cu_seqlens_q, cu_seqlens_k,
                                    max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale,
                                    causal=True, return_softmax=False,
                                )
                        elif i <= rank:
                            # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                            q_inputs[i%2] = q.view(-1, *q.shape[-2:])
                            # [2, b, sk//2, np, hn] -> [2, b*sk//2, np, hn]
                            kv_inputs[i%2] = kv_inputs[i%2][:, :, 0, ...].contiguous()
                            kv_inputs[i%2] = kv_inputs[i%2].view(2, -1, *k.shape[-2:])
                            if _flash_attn_2_available:
                                _, _, _, _, out_per_step[i], \
                                softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                                    q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                    cu_seqlens_q, cu_seqlens_k//2, max_seqlen_q, max_seqlen_k//2,
                                    dropout_p, softmax_scale, causal=False, return_softmax=False,
                                )
                            else:
                                out_per_step[i] = torch.empty_like(q_inputs[i%2])
                                _, softmax_lse_per_step[i], rng_states[i], _ = _flash_attn_forward( # pylint: disable=unbalanced-tuple-unpacking
                                    q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                    out_per_step[i], cu_seqlens_q, cu_seqlens_k//2,
                                    max_seqlen_q, max_seqlen_k//2, dropout_p, softmax_scale,
                                    causal=False, return_softmax=False,
                                )
                        else:
                            # [b, sq//2, np, hn] -> [b*sq//2, np, hn]
                            q_inputs[i%2] = q[:, 1, ...].contiguous().view(-1, *q.shape[-2:])
                            # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                            kv_inputs[i%2] = kv_inputs[i%2].view(2, -1, *k.shape[-2:])
                            if _flash_attn_2_available:
                                _, _, _, _, out_per_step[i], \
                                softmax_lse_per_step[i], _, rng_states[i] = _flash_attn_forward(
                                    q_inputs[i%2], kv_inputs[i%2][0], kv_inputs[i%2][1],
                                    cu_seqlens_q//2, cu_seqlens_k, max_seqlen_q//2, max_seqlen_k,
                                    dropout_p, softmax_scale, causal=False, return_softmax=False,
                                )
                            else:
                                out_per_step[i] = torch.empty_like(q_inputs[i%2])
                                _, softmax_lse_per_step[i], rng_states[i], _ = _flash_attn_forward( # pylint: disable=unbalanced-tuple-unpacking
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
        softmax_lse_ = softmax_lse_[..., 1, :].contiguous()
        # [b*sq, np, hn] -> [b, 2, sq//2, np, hn]
        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        # Flash Attn outputs
        dq = torch.empty_like(q)

        p2p_comm_buffers = [torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device), \
                            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device)]
        p2p_comm_buffers[0][0].copy_(kv)
        send_recv_reqs = []

        fa_optional_backward_kwargs = {}
        if not _flash_attn_2_available:
            fa_optional_backward_kwargs["num_splits"] = 1 if ctx.deterministic else 0

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
                        **fa_optional_backward_kwargs
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
                        **fa_optional_backward_kwargs
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
                        dout_, q_, kv_[0], kv_[1], out_, softmax_lse_,
                        dq_, dkv_[0], dkv_[1], cu_seqlens_q//2, cu_seqlens_k,
                        ctx.max_seqlen_q//2, ctx.max_seqlen_k,
                        ctx.dropout_p, ctx.softmax_scale, False,
                        rng_state=ctx.rng_states[cp_size-i-1],
                        **fa_optional_backward_kwargs
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


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """
    def __init__(
        self,
        dim: int,
        seq_len_interpolation_factor: Optional[int] = None,
        pretrained_max_position_embeddings: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        dim: int
            rotary embedding dimension
        seq_len_interpolation_factor: int
            if not None, discrete positions will be interpolated by this factor via the trick in
            https://arxiv.org/abs/2306.15595
        pretrained_max_position_embeddings: int
            pre-trained max_position_embeddings before position interpolation
        """
        super().__init__()
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.pretrained_max_position_embeddings = pretrained_max_position_embeddings

    def forward(self, max_seq_len: int, offset: int = 0):
        """
        Create rotary position embedding frequencies

        Parameters
        ----------
        max_seq_len: int
            sequence length of a sample
        offset: int, default = 0
            fixed offset for freqencies
        """
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        seq = seq.type_as(self.inv_freq)

        if (self.pretrained_max_position_embeddings is not None
            and self.seq_len_interpolation_factor is not None):
            if (max_seq_len >
                self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor):
                # dynamic linear scaling (length > position we have learned)
                seq *= 1 / (max_seq_len / self.pretrained_max_position_embeddings)
            else:
                # fixed linear scaling
                seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.einsum('i , j -> i j', seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return emb.reshape(emb.size(0), 1, 1, emb.size(1))

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


class _SplitAlongDim(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx,
                mixed_x_layer: torch.Tensor,
                split_dim: int,
                split_size_or_sections: Union[int, List[int], Tuple[int]],
    ) -> Tuple[torch.Tensor, ...]:
        ctx.split_dim = split_dim
        ctx.split_size_or_sections = split_size_or_sections
        return torch.split(mixed_x_layer, split_size_or_sections, dim = split_dim)

    @staticmethod
    def backward(ctx,
                 *grad_outputs):
        assert len(grad_outputs) > 0, "No gradients received for backprop!"

        if isinstance(ctx.split_size_or_sections, (list, tuple)):
            split_sizes = ctx.split_size_or_sections
            assert (len(grad_outputs) == len(split_sizes)
                ), "Unequal number of gradients vs split sections for backprop!"
        if isinstance(ctx.split_size_or_sections, int):
            split_sizes = [ctx.split_size_or_sections] * len(grad_outputs)
        dims = len(grad_outputs[0].shape)
        split_dim = (ctx.split_dim + dims) % dims

        noop_ok = True
        strides = grad_outputs[0].stride()
        data_ptr = grad_outputs[0].storage().data_ptr()
        shape = list(grad_outputs[0].shape)
        for i, tensor in enumerate(grad_outputs):
            shape_i = shape
            shape_i[split_dim] = split_sizes[i]
            offset_size = sum(split_sizes[:i]) * np.prod(shape[split_dim+1:])
            if (tensor.stride() != strides or
                list(tensor.shape) != shape_i or
                tensor.storage().data_ptr() != data_ptr or
                tensor.storage_offset() != offset_size):
                noop_ok = False
                break

        if noop_ok:
            ret = torch.Tensor().to(device=grad_outputs[0].device,
                                    dtype=grad_outputs[0].dtype)
            new_shape = list(shape)
            new_shape[split_dim] = sum(split_sizes)
            ret.set_(grad_outputs[0].untyped_storage(),
                     grad_outputs[0].storage_offset(),
                     new_shape,
                     strides
            )
            return ret, None, None

        return torch.cat(grad_outputs, dim = split_dim), None, None



class UnfusedDotProductAttention(torch.nn.Module):
    """Parallel attention w/o QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        norm_factor: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        layer_number: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.norm_factor = norm_factor
        self.attention_dropout_ctx = attention_dropout_ctx
        self.layer_number = layer_number

        self.scale_mask_softmax = FusedScaleMaskSoftmax(attention_mask_func)

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
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None, # pylint: disable=unused-argument
        cu_seqlens_kv: Optional[torch.Tensor] = None, # pylint: disable=unused-argument
        attn_mask_type: str = "causal",
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """core attention fprop"""

        assert (qkv_layout in QKVLayouts
            ), f"UnfusedDotProductAttention does not support qkv_layout = {qkv_layout}!"
        qkv_format = ''.join([i for i in qkv_layout.split('_')[0] if i.isalpha()])
        assert (qkv_format != 'thd'
            ), """UnfusedDotProductAttention does not support variable sequence lengths!"""
        if qkv_format == 'bshd':
            # convert to sbhd and use sbhd implementation for now
            query_layer, key_layer, value_layer = [x.transpose(0, 1)
                for x in [query_layer, key_layer, value_layer]]
        assert (
            attn_mask_type in AttnMaskTypes
        ), f"attn_mask_type {attn_mask_type} not supported"

        batch_size, seqlen = query_layer.shape[1], query_layer.shape[0]
        apply_qk_layer_scaling = self.apply_qk_layer_scaling and key_layer.dtype == torch.float16

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

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
        attention_probs = self.scale_mask_softmax(
            attention_scores, attention_mask, attn_mask_type, softmax_scale)

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

        if qkv_format == 'sbhd':
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            context_layer = context_layer.view(seqlen, batch_size, -1)

        if qkv_format == 'bshd':
            # [b, np, sq, hn] --> [b, sq, np, hn]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

            # [b, sq, np, hn] --> [b, sq, hp]
            context_layer = context_layer.view(batch_size, seqlen, -1)

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

def _get_qkv_layout(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qkv_format: str = 'sbhd',
    ) -> str:
    """Get qkv layout.

    Parameters
    ----------
    q: torch.Tensor
        Query tensor.
    k: torch.Tensor
        Key tensor.
    v: torch.Tensor
        Value tensor.
    qkv_format: str, default = `sbhd`
        Dimension format for `q`, `k` and `v`, {`sbhd`, `bshd`, `thd`}. `s` stands for
        the sequence length dimension, `b` batch size, `h` the number of attention heads,
        `d` head size, and `t` the total number of sequences in a batch, i.e.
        `t = sum(s_i) for i = 0...b-1`.

    Returns
    ----------
    qkv_layout: str
       Memory layout of `q`, `k` and `v`. Each `qkv_format` can be mapped to one of five
       memory layouts. For example, `sb3hd` means `q`, `k`, `v` are created as one chunk
       of memory and that they are interleaved in the `2`nd dimension. `sbhd_sbh2d` means
       `q` and `kv` are created in two chunks and that `q` itself is contiguous and `k`, `v`
       are interleaved with each other in the `3`rd dimension, `k = kv[:,:,:,0,:]` and
       `v = kv[:,:,:,1,:]`.
       Mapping:
       `sbhd`: {`sb3hd`, `sbh3d`, `sbhd_sb2hd`, `sbhd_sbh2d`, `sbhd_sbhd_sbhd`}
       `bshd`: {`bs3hd`, `bsh3d`, `bshd_bs2hd`, `bshd_bsh2d`, `bshd_bshd_bshd`}
       `thd` : {`t3hd`, `th3d`, `thd_t2hd`, `thd_th2d`, `thd_thd_thd`}
    """

    check_last_dim_contiguous = all(x.stride(-1) == 1 for x in [q, k, v])
    assert check_last_dim_contiguous, "q, k and v must have stride 1 in their last dimension!"

    data_ptr = q.untyped_storage().data_ptr()
    check_ptrs_qkv = all(x.untyped_storage().data_ptr() == data_ptr for x in [q, k, v])
    data_ptr = k.untyped_storage().data_ptr()
    check_ptrs_kv = all(x.untyped_storage().data_ptr() == data_ptr for x in [k, v])

    stride = q.stride()
    check_strides_qkv = all(stride == x.stride() for x in [q, k, v])
    stride = k.stride()
    check_strides_kv = all(stride == x.stride() for x in [k, v])

    shape = q.shape
    check_shapes_qkv = all(shape == x.shape for x in [q, k, v])
    shape = k.shape
    check_shapes_kv = all(shape == x.shape for x in [k, v])

    last_dim_size = q.shape[-1]
    check_last_dim_offsets_qkv = all(i * last_dim_size == x.storage_offset()
                        for i, x in enumerate([q, k, v]))
    last_dim_size = k.shape[-1]
    check_last_dim_offsets_kv = all(i * last_dim_size == x.storage_offset()
                        for i, x in enumerate([k, v]))

    last_two_dims_size = q.shape[-1] * q.shape[-2]
    check_last_two_dims_offsets_qkv = all(i * last_two_dims_size == x.storage_offset()
                        for i, x in enumerate([q, k, v]))
    last_two_dims_size = k.shape[-1] * k.shape[-2]
    check_last_two_dims_offsets_kv = all(i * last_two_dims_size == x.storage_offset()
                        for i, x in enumerate([k, v]))

    qkv_layout = None
    if (check_ptrs_qkv and check_strides_qkv and check_shapes_qkv
        and check_last_two_dims_offsets_qkv
        and not check_last_dim_offsets_qkv):
        # sb3hd, bs3hd, t3hd
        qkv_layout = qkv_format[:-2] + '3' + qkv_format[-2:]
    elif check_ptrs_qkv and check_strides_qkv and check_shapes_qkv and check_last_dim_offsets_qkv:
        # sbh3d, bsh3d, th3d
        qkv_layout = qkv_format[:-1] + '3' + qkv_format[-1:]
    elif (check_ptrs_kv and check_strides_kv and check_shapes_kv
        and check_last_two_dims_offsets_kv
        and not check_last_dim_offsets_kv):
        # sbhd_sb2hd, bshd_bs2hd, thd_t2hd
        qkv_layout = qkv_format + '_' + qkv_format[:-2] + '2' + qkv_format[-2:]
    elif (check_ptrs_kv and check_strides_kv and check_shapes_kv
        and check_last_dim_offsets_kv):
        # sbhd_sbh2d, bshd_bsh2d, thd_th2d
        qkv_layout = qkv_format + '_' + qkv_format[:-1] + '2' + qkv_format[-1:]
    elif check_strides_kv and check_shapes_kv:
        # sbhd_sbhd_sbhd, bshd_bshd_bshd, thd_thd_thd
        qkv_layout = '_'.join(list([qkv_format])*3)
    else:
        raise Exception("The provided qkv memory layout is not supported!")

    return qkv_layout


class FlashAttention(torch.nn.Module):
    """Dot product attention, using HazyResearch flash-attn package:
    https://github.com/Dao-AILab/flash-attention
    """

    def __init__(
        self,
        norm_factor: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        assert (
            _flash_attn_version >= _flash_attn_version_required
        ), f"FlashAttention minimum version {_flash_attn_version_required} is required."

        self.norm_factor = norm_factor
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.layer_number = 1 if layer_number is None else layer_number
        self.deterministic = deterministic

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        attn_mask_type: str = "causal",
        cp_group: Optional[dist_group_type] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
    ) -> torch.Tensor:
        """flash-attn fprop"""

        assert (
            query_layer.dtype in [torch.float16, torch.bfloat16]
            and key_layer.dtype in [torch.float16, torch.bfloat16]
            and value_layer.dtype in [torch.float16, torch.bfloat16]
            ), "FlashAttention currently only supports FP16 and BF16."
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
            ), "FlashAttention currently only supports CUDA tensors."
        assert (
            qkv_layout in QKVLayouts
            ), f"FlashAttention does not support qkv_layout = {qkv_layout}!"

        context_parallel = (cp_group is not None) and (get_distributed_world_size(cp_group) != 1)

        qkv_format = ''.join([i for i in qkv_layout.split('_')[0] if i.isalpha()])

        if qkv_format == 'sbhd':
            # For now just 128, will make it more general in the future
            if (query_layer.shape[-1] == 128 and
                query_layer.shape[0] * query_layer.shape[1] >= 512 and
                qkv_layout == "sbh3d"):
                query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(query_layer,
                                                                             key_layer,
                                                                             value_layer)
            else:
                query_layer, key_layer, value_layer = [x.transpose(0,1).contiguous()
                    for x in (query_layer, key_layer, value_layer)]
        elif qkv_format == 'bshd':
            query_layer, key_layer, value_layer = [x.contiguous()
                for x in (query_layer, key_layer, value_layer)]

        global _cu_seqlens_q, _cu_seqlens_kv, _indices_q, _indices_kv
        batch_size, max_seqlen_q, max_seqlen_kv = (
                query_layer.shape[0], query_layer.shape[1], key_layer.shape[1])

        if qkv_format in ['sbhd', 'bshd']:
            if not context_parallel:
                # [b * s, h, d]
                query_layer, key_layer, value_layer = [
                    x.view(x.shape[0] * x.shape[1], *x.shape[2:])
                    for x in [query_layer, key_layer, value_layer]
                ]

            if attn_mask_type == 'padding':
                assert not context_parallel, "Padding mask not supported with context parallelism."

                if self.attention_type == "self":
                    assert (
                        max_seqlen_q == max_seqlen_kv
                    ), "Maximum sequence length for Q and KV should be the same."
                    if self.layer_number == 1:
                        _cu_seqlens_q, _indices_q = get_cu_seqlens_and_indices(attention_mask)
                    _cu_seqlens_kv = _cu_seqlens_q
                    query_layer_packed, key_layer_packed, value_layer_packed = PackTensors.apply(
                        _indices_q, query_layer, key_layer, value_layer
                    )
                else:
                    if self.layer_number == 1:
                        _cu_seqlens_q, _indices_q = get_cu_seqlens_and_indices(attention_mask[0])
                        _cu_seqlens_kv, _indices_kv = get_cu_seqlens_and_indices(attention_mask[1])
                    query_layer_packed = PackTensors.apply(_indices_q, query_layer)
                    key_layer_packed, value_layer_packed = PackTensors.apply(
                        _indices_kv, key_layer, value_layer
                    )
                query_layer, key_layer, value_layer = (
                    query_layer_packed, key_layer_packed, value_layer_packed)
                cu_seqlens_q, cu_seqlens_kv = _cu_seqlens_q, _cu_seqlens_kv
            else:
                if cu_seqlens_q is None:
                    cu_seqlens_q = torch.arange(
                            0,
                            (batch_size + 1) * max_seqlen_q,
                            step=max_seqlen_q,
                            dtype=torch.int32,
                            device=query_layer.device)
                if cu_seqlens_kv is None:
                    cu_seqlens_kv = torch.arange(
                            0,
                            (batch_size + 1) * max_seqlen_kv,
                            step=max_seqlen_kv,
                            dtype=torch.int32,
                            device=key_layer.device)
        elif qkv_format == 'thd':
            assert not context_parallel, "thd format is not supported for context parallelism!"
            assert (_flash_attn_2_available
                ), "flash-attn v2 is required for variable sequence length support!"
            assert (cu_seqlens_q is not None and cu_seqlens_kv is not None
                ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            max_seqlen_q = seqlens_q.max().item()
            max_seqlen_kv = seqlens_kv.max().item()

        if context_parallel:
            with self.attention_dropout_ctx():
                output = flash_attn_forward_func_with_cp(
                    query_layer, key_layer, value_layer,
                    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
                    self.attention_dropout if self.training else 0.0,
                    cp_group, cp_global_ranks, cp_stream,
                    softmax_scale=1.0/self.norm_factor,
                    causal=attn_mask_type=="causal",
                    deterministic=self.deterministic
                )
        else:
            with self.attention_dropout_ctx():
                fa_optional_forward_kwargs = {}
                if not _flash_attn_2_available:
                    fa_optional_forward_kwargs["deterministic"] = self.deterministic
                output = flash_attn_forward_func(
                    query_layer, key_layer, value_layer,
                    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
                    self.attention_dropout if self.training else 0.0,
                    softmax_scale=1.0/self.norm_factor, causal=attn_mask_type=="causal",
                    **fa_optional_forward_kwargs
                )

        if attn_mask_type == 'padding':
            output = UnpackTensor.apply(_indices_q, batch_size * max_seqlen_q, output)

        if qkv_format == 'sbhd':
            # (bs)hd -> bs(hd) -> sb(hd)
            output = output.view(batch_size, max_seqlen_q, -1).transpose(0, 1).contiguous()
        elif qkv_format == 'bshd':
            # (bs)hd -> bs(hd)
            output = output.view(batch_size, max_seqlen_q, -1).contiguous()

        return output


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

class FusedAttnFunc(torch.autograd.Function):
    """Function for FusedAttention with separate Q, K, V tensors"""

    @staticmethod
    def forward(ctx, is_training, max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv,
                q, k, v, qkv_dtype, attn_bias, attn_scale, dropout_p, fast_zero_fill,
                qkv_layout, attn_bias_type, attn_mask_type,
                rng_gen, fused_attention_backend, use_FAv2_bwd):
        out, aux_ctx_tensors = fused_attn_fwd(
            is_training, max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv,
            q, k, v, qkv_dtype, fused_attention_backend, attn_bias,
            None, None, None, None, None,
            attn_scale, dropout_p, fast_zero_fill, qkv_layout, attn_bias_type, attn_mask_type,
            rng_gen)

        ctx.save_for_backward(q, k, v, out, cu_seqlens_q, cu_seqlens_kv)
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
        q, k, v, out, cu_seqlens_q, cu_seqlens_kv = ctx.saved_tensors
        if ctx.use_FAv2_bwd:
            softmax_lse, rng_state = ctx.aux_ctx_tensors
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
            d_out, q, k, v, out = [maybe_contiguous(x)
                for x in (d_out, q, k, v, out)]
            flash_attn_cuda_bwd(
                d_out, q, k, v, out, softmax_lse, dq, dk, dv,
                cu_seqlens_q, cu_seqlens_kv, ctx.max_seqlen_q, ctx.max_seqlen_kv,
                ctx.dropout_p, ctx.attn_scale, False,
                ctx.attn_mask_type == "causal", None, rng_state
            )
            dq = dq[..., :d_out.shape[-1]]
            dk = dk[..., :d_out.shape[-1]]
            dv = dv[..., :d_out.shape[-1]]
        else:
            dq, dk, dv, *rest = fused_attn_bwd(
                ctx.max_seqlen_q, ctx.max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv,
                q, k, v, out, d_out,
                ctx.qkv_dtype, ctx.aux_ctx_tensors,
                ctx.fused_attention_backend,
                None, None, None, None, None, None, None, None, None,
                ctx.attn_scale, ctx.dropout_p, ctx.fast_zero_fill,
                ctx.qkv_layout, ctx.attn_bias_type, ctx.attn_mask_type)

        # if no_bias, return dqkv
        if ctx.attn_bias_type == "no_bias":
            return (None, None, None, None, None, dq, dk, dv, None, None, None,
                    None, None, None, None, None, None,
                    None, None, None, None, None, None)
        # else, return (dqkv, dbias)
        return (None, None, None, None, None, dq, dk, dv, None, rest[0], None,
                None, None, None, None, None, None,
                None, None, None, None, None, None)

class FusedAttention(torch.nn.Module):
    """Dot product attention, with multiple backends:

    1. FusedAttnBackend["F16_max512_seqlen"]
       cuDNN based fused attention for FP16/BF16 and <=512 sequence length.
    2. FusedAttnBackend["F16_arbitrary_seqlen"]
       cuDNN based fused attention for FP16/BF16 and any sequence length.

    Support matrix:

    | backend       | 1                       | 2                              |
    | flash based   | no                      | yes                            |
    | cuDNN based   | yes                     | yes                            |
    | qkv dtype     | fp16/bf16               | fp16/bf16                      |
    | attn_type     | self/cross              | self                           |
    | qkv_layout    |                         |                                |
    |  - qkv        | qkv_interleaved         | qkv_interleaved                |
    |  - (q,kv)     | kv_interleaved          |                                |
    |  - (q,k,v)    | sb3hd, bs3hd            | sb3hd, bs3hd, sbh3d, bsh3d     |
    |               | sbhd_sb2hd, bshd_bs2hd  | sbhd_sb2hd, bshd_bs2hd         |
    |               | bshd_bshd_bshd          | sbhd_sbh2d, bshd_bsh2d         |
    |               |                         | sbhd_sbhd_sbhd, bshd_bshd_bshd |
    | mask_type     | causal/no_mask          | causal                         |
    | bias_type     | no_bias/post_scale_bias | no_bias                        |
    | dropout       | yes                     | yes                            |
    | max_seqlen    | <=512                   | any                            |
    | head_dim      | 64                      | 64,128                         |
    | output dtype  | fp16/bf16               | fp16/bf16                      |
    """

    def __init__(
        self,
        norm_factor: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attention_type: str = "self",
    ) -> None:
        super().__init__()

        self.norm_factor = norm_factor
        self.attention_dropout = attention_dropout
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_type = attention_type
        self.use_FAv2_bwd = (os.getenv("NVTE_FUSED_ATTN_USE_FAv2_BWD", "0") == "1"
                        and _flash_attn_2_available
                        and get_device_compute_capability() == (9, 0))

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        attn_mask_type: str = "causal",
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
        assert (
            qkv_layout in QKVLayouts
            ), f"FusedAttention does not support qkv_layout = {qkv_layout}!"

        qkv_format = ''.join([i for i in qkv_layout.split('_')[0] if i.isalpha()])
        if qkv_format in ['sbhd', 'bshd']:
            if qkv_format == 'sbhd':
                batch_size, max_seqlen_q, max_seqlen_kv = (
                    query_layer.shape[1], query_layer.shape[0], key_layer.shape[0])
            if qkv_format == 'bshd':
                batch_size, max_seqlen_q, max_seqlen_kv = (
                    query_layer.shape[0], query_layer.shape[1], key_layer.shape[1])
            if cu_seqlens_q is None:
                cu_seqlens_q = torch.arange(
                        0,
                        (batch_size + 1) * max_seqlen_q,
                        step=max_seqlen_q,
                        dtype=torch.int32,
                        device=query_layer.device)
            if cu_seqlens_kv is None:
                cu_seqlens_kv = torch.arange(
                        0,
                        (batch_size + 1) * max_seqlen_kv,
                        step=max_seqlen_kv,
                        dtype=torch.int32,
                        device=key_layer.device)
        if qkv_format == 'thd':
            assert (cu_seqlens_q is not None and cu_seqlens_kv is not None
                ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            max_seqlen_q = seqlens_q.max().item()
            max_seqlen_kv = seqlens_kv.max().item()

        qkv_dtype = TE_DType[query_layer.dtype]

        use_FAv2_bwd = (self.use_FAv2_bwd
                and (fused_attention_backend
                    == tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen))
        with self.attention_dropout_ctx():
            output = FusedAttnFunc.apply(
                self.training,
                max_seqlen_q, max_seqlen_kv,
                cu_seqlens_q, cu_seqlens_kv,
                query_layer, key_layer, value_layer,
                qkv_dtype,
                core_attention_bias,
                1.0/self.norm_factor,
                self.attention_dropout if self.training else 0.0,
                fast_zero_fill,
                qkv_layout,
                core_attention_bias_type,
                attn_mask_type,
                None, # rng_gen
                fused_attention_backend,
                use_FAv2_bwd,
            )

        # ...hd -> ...(hd)
        return output.view(*output.shape[:-2], -1)


class DotProductAttention(torch.nn.Module):
    """Allows the model to jointly attend to information from different
    representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. note::

        Argument :attr:`attention_mask` in the `forward` call is only used when
        :attr:`self_attn_mask_type` includes `"padding"` or `"arbitrary"`.

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
    attn_mask_type: str, default = `causal`
                   type of attention mask passed into softmax operation, options are "`causal`",
                   "`padding`", "`arbitrary`", "`no_mask`". For the "`causal`" mask,
                   TransformerEngine calculates and applies an upper triangular mask to
                   the softmax input. An "`arbitrary`" mask is an arbitrary user defined mask
                   broadcastable to the shape of softmax input. The "`padding`" mask is used
                   for providing locations of padded tokens in the batch, which should be of
                   the shape [batch_size, 1, 1, seq_len]. No mask is applied for the "`no_mask`"
                   option. For the `"arbitrary"` and `"padding"` mask types, the argument
                   :attr:`attention_mask` must be passed into `forward` call. The "`causal`"
                   mask can also be applied in conjunction with "`padding`" mask by passing
                   in multiple mask type as a comma separated string, for example,
                   `attn_mask_type="causal,padding"`.
    attention_type: str, default = `self`
                   type of attention, either "`self`" and "`cross`".
    layer_number: int, default = `None`
                 layer number of the current `DotProductAttention` when multiple such modules
                 are concatenated, for instance in consecutive transformer blocks.
    qkv_format: str, default = `sbhd`
               dimension format for `query_layer`, `key_layer` and `value_layer`,
               {`sbhd`, `bshd`, `thd`}. `s` stands for the sequence length, `b` batch size,
               `h` the number of heads, `d` head size, and `t` the total number of sequences
               in a batch, with `t = sum(s_i), for i = 0...b-1`. `sbhd` and `bshd` formats
               are used for when sequences in a batch are of equal length or padded to
               equal length, and the `thd` format is used for when sequences in a batch
               have different lengths. Please note that these formats do not reflect how
               tensors `query_layer`, `key_layer`, `value_layer` are laid out in memory.
               For that, please use `_get_qkv_layout` to gain the layout information.
    attn_mask_type: {'causal', 'padding', 'no_mask', 'arbitrary'}, default = `causal`
                   type of attention mask passed into softmax operation. Overridden by
                   :attr:`attn_mask_type` in the `forward` method. The forward
                   arg is useful for dynamically changing mask types, e.g. a different
                   mask for training and inference. The init arg is useful for cases
                   involving compilation/tracing, e.g. ONNX export.

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
    cp_global_ranks : list of global rank IDs, default = `None`
                     global rank IDs of GPUs that are in cp_group.
    cp_stream : CUDA stream, default = `None`
               context parallelism splits flash attention into multiple steps for
               compute and communication overlapping. To address the wave quantization
               issue of each split step, we add an additional CUDA stream so that we
               can overlap two flash attention kernels.
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        num_gqa_groups: Optional[int] = None,
        attention_dropout: float = 0.0,
        qkv_format: str = "sbhd",
        attn_mask_type: str = "causal",
        sequence_parallel: bool = False,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        attention_type: str = "self",
        cp_group: Optional[dist_group_type] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
    ) -> None:
        super().__init__()

        self.qkv_format = qkv_format
        self.attn_mask_type = attn_mask_type
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
            and self.device_compute_capability >= (8, 0)
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
            and self.device_compute_capability >= (8, 0)
        )

        assert (
            attention_type in AttnTypes
        ), f"attention_type {attention_type} not supported"

        self.attention_type = attention_type
        self.attention_dropout = attention_dropout

        attn_kwargs = {
            "attention_dropout": attention_dropout,
            "attention_dropout_ctx": attention_dropout_ctx,
        }

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(norm_factor,
                                                  attention_type=attention_type,
                                                  layer_number=layer_number,
                                                  deterministic=self.deterministic,
                                                  **attn_kwargs)

        # Instantiating three types since use of flash-attn and FusedAttention
        # might be ruled out due to forward inputs.
        if self.use_fused_attention:
            self.fused_attention = FusedAttention(
                norm_factor, **attn_kwargs,
                attention_type=attention_type)
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

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
    ) -> None:
        """
        Set the context parallel attributes for the given
        module before executing the forward pass.

        Parameters
        ----------
        cp_group : ProcessGroup
                  context parallel process group.
        cp_global_ranks : List[int]
                         list of global ranks in the context group.
        cp_stream : torch.cuda.Stream
                   cuda stream for context parallel execution.
        """
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_format: Optional[str] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        attn_mask_type: Optional[str] = None,
        checkpoint_core_attention: bool = False,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
    ) -> torch.Tensor:
        """
        Dot Product Attention Layer.

        .. note::

            Argument :attr:`attention_mask` is only used when :attr:`attn_mask_type`
            includes '"padding"' or `"arbitrary"`.

        .. note::

            Input tensors :attr:`query_layer`, :attr:`key_layer`, and :attr:`value_layer`
            must each be of shape (:attr:`sequence_length`, :attr:`batch_size`,
            :attr:`num_attention_heads`, :attr:`kv_channels`). Output of shape
            (:attr:`sequence_length`, :attr:`batch_size`, :attr:`num_attention_heads`
            * :attr:`kv_channels`) is returned.

        .. note::

            DotProductAttention supports three backends: 1) FlashAttention which calls
            HazyResearch/Dao-AILab's `flash-attn <https://arxiv.org/pdf/2305.13245.pdf>`_
            PyTorch API, 2) FusedAttention which has multiple fused attention implementations
            based on `cuDNN Graph API
            <https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion>`_
            (see :attr:`FusedAttention` for more details on FusedAttention backends), and 3)
            UnfusedDotProductAttention which is the native PyTorch implementation
            with fused scaled masked softmax.

        .. note::

            Users can use environment variables :attr:`NVTE_FLASH_ATTN`, :attr:`NVTE_FUSED_ATTN`,
            and :attr:`NVTE_FUSED_ATTN_BACKEND` to control which DotProductAttention backend,
            and FusedAttention backend if applicable, to use. TransformerEngine prioritizes
            FlashAttention over FusedAttention and over UnfusedDotProductAttention.
            If FusedAttention is being used, users can also choose to switch to flash-attn's
            implementation for backward by setting :attr:`NVTE_FUSED_ATTN_USE_FAv2_BWD=1`
            (default: 0), because of the performance differences between various versions of
            flash-attn and FusedAttention. Further, :attr:`NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT`
            can be used to enable (:attr:`1`) or disable (:attr:`0`) the workspace related
            optimizations in FusedAttention. When unset, TransformerEngine determines the code path
            based on its internal logic. These optimizations trade memory for performance
            and should be used with care.

        Parameters
        ----------
        query_layer : torch.Tensor
                     Query tensor.
        key_layer : torch.Tensor
                   Key tensor.
        value_layer : torch.Tensor
                     Value tensor.
        attention_mask : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], default = `None`
                        Boolean tensor used to mask out softmax input when not using flash-attn.
                        Can be a tuple of 2 masks for cross attention with padding masks.
        qkv_format: str, default = `None`
                   If provided, overrides :attr:`qkv_format` from initialization.
        cu_seqlens_q: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths in a batch for `query_layer`,
                   with shape [batch_size + 1] and dtype torch.int32.
        cu_seqlens_kv: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths in a batch for `key_layer` and `value_layer`,
                   with shape [batch_size + 1] and dtype torch.int32.
        attn_mask_type: {'causal', 'padding', 'no_mask', 'arbitrary'}, default = `None`
                       type of attention mask passed into softmax operation.
        checkpoint_core_attention : bool, default = `False`
                                   If true, forward activations for attention are recomputed
                                   during the backward pass in order to save memory that would
                                   otherwise be occupied to store the forward activations until
                                   backprop.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, 'post_scale_bias`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T
        fast_zero_fill: bool, default = `True`
                    Whether to use the fast path to set output tensors to 0 or not.
        """

        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
            ), 'DotProductAttention only supports CUDA tensors.'

        assert (key_layer.shape == value_layer.shape
            ), "Keys and values must have the same shape!"

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        if qkv_format is None:
            qkv_format = self.qkv_format
        attn_mask_type, causal_mask = _unpack_attn_mask_type(attn_mask_type)

        assert (key_layer.shape[-2] == self.num_gqa_groups_per_partition
            and value_layer.shape[-2] == self.num_gqa_groups_per_partition
            ), f"Keys and values must have num_gqa_group = {self.num_gqa_groups} heads!"
        assert (qkv_format in ['sbhd', 'bshd', 'thd']
            ), "DotProductAttention only supports qkv_format = {'sbhd', 'bshd', 'thd'}!"

        if qkv_format == 'thd':
            assert (all(len(x.shape) == 3 for x in (query_layer, key_layer, value_layer))
                ), "Queries, keys and values must be 3D tensors when qkv_format = thd!"
            assert (cu_seqlens_q is not None and cu_seqlens_kv is not None
                ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            assert (cu_seqlens_q.shape == cu_seqlens_kv.shape
                and len(cu_seqlens_q.shape) == 1
                and len(cu_seqlens_kv.shape) == 1
                ), "cu_seqlens_q and cu_seqlens_q must both have shape [batch_size + 1]!"
            assert (cu_seqlens_q.dtype == torch.int32
                and cu_seqlens_kv.dtype == torch.int32
                ), "cu_seqlens_q and cu_seqlens_q must both be in dtype torch.int32!"
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            max_seqlen_q = seqlens_q.max().item()
            max_seqlen_kv = seqlens_kv.max().item()

        if qkv_format in ['sbhd', 'bshd']:
            assert (all(len(x.shape) == 4 for x in (query_layer, key_layer, value_layer))
                ), f"Queries, keys and values must be 4D tensors when qkv_format = {qkv_format}!"
            if qkv_format == 'sbhd':
                max_seqlen_q, max_seqlen_kv = (query_layer.shape[0], key_layer.shape[0])
            if qkv_format == 'bshd':
                max_seqlen_q, max_seqlen_kv = (query_layer.shape[1], key_layer.shape[1])
            if cu_seqlens_q is not None:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                assert (all(seqlens_q <= max_seqlen_q)
                    ), """Sequence lengths indicated by cu_seqlens_q must be no greater than
                    the sequence dimention in 'query_layer'!"""
            if cu_seqlens_kv is not None:
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                assert (all(seqlens_kv <= max_seqlen_kv)
                    ), """Sequence lengths indicated by cu_seqlens_kv must be no greater than
                    the sequence dimention in 'key_layer' and 'value_layer'!"""

        qkv_layout = _get_qkv_layout(query_layer, key_layer, value_layer,
            qkv_format = qkv_format)

        # The priority for attention backends (subject to availability and clearing the filters)
        # is: FlashAttention > FusedAttention (cuDNN) > UnfusedDotProductAttention.
        use_flash_attention = self.use_flash_attention
        use_fused_attention = self.use_fused_attention

        # The following section filters out some backends based on
        # certain asserts before executing the forward pass.

        # Filter: Input type.
        if (query_layer.dtype not in [torch.bfloat16, torch.float16]
            or key_layer.dtype not in [torch.bfloat16, torch.float16]
            or value_layer.dtype not in [torch.bfloat16, torch.float16]
        ):
            use_flash_attention = False
            use_fused_attention = False

        # Filter: Device and dimensions.
        if key_layer.shape[-1] > 64:
            if self.device_compute_capability in ((8, 6), (8, 7)):
                use_flash_attention = False
            elif (
                not _flash_attn_2_available
                and self.device_compute_capability == (8, 9)
            ):
                use_flash_attention = False

        if not _flash_attn_2_available and self.num_gqa_groups != self.num_attention_heads:
            use_flash_attention = False

        if core_attention_bias_type != "no_bias" or core_attention_bias is not None:
            use_flash_attention = False

        # Filter: ONNX export.
        if is_in_onnx_export_mode():
            use_flash_attention = False
            use_fused_attention = False

        # Filter: Attention mask type.
        #    attn_mask_type(s)   |     supported backends
        # ------------------------------------------------
        #   causal               |     All
        #   padding              |     UnfusedDotProductAttention, FlashAttention
        #   arbitrary            |     UnfusedDotProductAttention
        #   no_mask              |     All
        #   causal + padding     |     FlashAttention
        #
        if attn_mask_type == "arbitrary":
            use_flash_attention = False
            use_fused_attention = False
        elif attn_mask_type == "padding" and causal_mask:
            assert use_flash_attention, "No attention backend available for causal + padding masks."
        elif attn_mask_type == "padding":
            use_fused_attention = False

        if use_fused_attention:
            fused_attention_backend = tex.get_fused_attn_backend(
                TE_DType[query_layer.dtype],
                TE_DType[key_layer.dtype],
                QKVLayout[qkv_layout],
                AttnBiasType[core_attention_bias_type],
                AttnMaskType[attn_mask_type],
                self.attention_dropout,
                max_seqlen_q, max_seqlen_kv,
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
            return self.flash_attention(query_layer,
                                        key_layer,
                                        value_layer,
                                        attention_mask=attention_mask,
                                        qkv_layout=qkv_layout,
                                        cu_seqlens_q=cu_seqlens_q,
                                        cu_seqlens_kv=cu_seqlens_kv,
                                        attn_mask_type=attn_mask_type,
                                        cp_group=self.cp_group,
                                        cp_global_ranks=self.cp_global_ranks,
                                        cp_stream=self.cp_stream)

        assert (
            self.cp_group is None or get_distributed_world_size(self.cp_group) == 1
        ), "Context parallelism is only implemented with Flash Attention!"

        if use_fused_attention:
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(self.fused_attention,
                              query_layer,
                              key_layer,
                              value_layer,
                              qkv_layout = qkv_layout,
                              cu_seqlens_q = cu_seqlens_q,
                              cu_seqlens_kv = cu_seqlens_kv,
                              attn_mask_type = attn_mask_type,
                              fused_attention_backend = fused_attention_backend,
                              core_attention_bias_type = core_attention_bias_type,
                              core_attention_bias = core_attention_bias,
                              fast_zero_fill = fast_zero_fill)
            return self.fused_attention(query_layer, key_layer, value_layer,
                              qkv_layout = qkv_layout,
                              cu_seqlens_q = cu_seqlens_q,
                              cu_seqlens_kv = cu_seqlens_kv,
                              attn_mask_type = attn_mask_type,
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
                qkv_layout = qkv_layout,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_kv = cu_seqlens_kv,
                attn_mask_type = attn_mask_type,
                attention_mask = attention_mask,
                core_attention_bias_type = core_attention_bias_type,
                core_attention_bias = core_attention_bias)
        return self.unfused_attention(query_layer,
                key_layer,
                value_layer,
                qkv_layout = qkv_layout,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_kv = cu_seqlens_kv,
                attn_mask_type = attn_mask_type,
                attention_mask = attention_mask,
                core_attention_bias_type = core_attention_bias_type,
                core_attention_bias = core_attention_bias)


class MultiheadAttention(torch.nn.Module):
    r"""
    Multi-head Attention (MHA), including Query,
    Key, Value and Output projection.

    .. note::

        Argument :attr:`attention_mask` will be ignored in the `forward` call when
        :attr:`attn_mask_type` is set to `"causal"`.

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
    attn_mask_type: {'causal', 'padding', 'no_mask', 'arbitrary'}, default = `causal`
                   type of attention mask passed into softmax operation. Overridden by
                   :attr:`attn_mask_type` in the `forward` method. The forward
                   arg is useful for dynamically changing mask types, e.g. a different
                   mask for training and inference. The init arg is useful for cases
                   involving compilation/tracing, e.g. ONNX export.
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
        ub_atomic_gemm_rs: bool = False,
        ub_atomic_gemm_ag: bool = False,
        bias: bool = True,
        normalization: str = "LayerNorm",
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        self.attn_mask_type = attn_mask_type
        self.layer_number = layer_number
        self.input_layernorm = input_layernorm
        self.attention_type = attention_type
        self.get_rng_state_tracker = get_rng_state_tracker
        self.tp_group = tp_group
        self.return_layernorm_output = return_layernorm_output
        self.params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
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
                ), "The number of attention heads must be divisible by the number of GQA groups!"
        assert (self.num_gqa_groups % tp_size == 0
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

        if self.attention_type == "self":
            parameters_split = {"query_": hidden_size,
                                "key_": self.hidden_size_kv,
                                "value_": self.hidden_size_kv} if not fuse_qkv_params else None
            if self.input_layernorm:
                self.layernorm_qkv = LayerNormLinear(
                    hidden_size,
                    hidden_size + 2 * self.hidden_size_kv,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    return_layernorm_output=return_layernorm_output,
                    parameters_split=parameters_split,
                    zero_centered_gamma=zero_centered_gamma,
                    ub_bulk_wgrad=ub_bulk_wgrad,
                    ub_bulk_dgrad=ub_bulk_dgrad,
                    ub_split_ag=ub_split_ag,
                    normalization=normalization,
                    ub_atomic_gemm_ag=ub_atomic_gemm_ag,
                    **common_gemm_kwargs,
                )
            else:
                self.qkv = Linear(
                    hidden_size,
                    hidden_size + 2 * self.hidden_size_kv,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    parameters_split=parameters_split,
                    **common_gemm_kwargs,
                )
        elif self.attention_type == "cross":
            if self.input_layernorm:
                self.layernorm_query = LayerNormLinear(
                    hidden_size,
                    hidden_size,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    parameters_split=("query_",) if not fuse_qkv_params else None,
                    return_layernorm_output=return_layernorm_output,
                    zero_centered_gamma=zero_centered_gamma,
                    ub_bulk_wgrad=ub_bulk_wgrad,
                    ub_bulk_dgrad=ub_bulk_dgrad,
                    ub_split_ag=ub_split_ag,
                    normalization=normalization,
                    ub_atomic_gemm_ag=ub_atomic_gemm_ag,
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
            ub_atomic_gemm_rs=ub_atomic_gemm_rs,
            ub_atomic_gemm_ag=ub_atomic_gemm_ag,
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
        """
        Set the tensor parallel group for the given
        module before executing the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = `None`
                  tensor parallel process group.
        """
        self.tp_group = tp_group

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
    ) -> None:
        """
        Set the context parallel attributes for the given
        module before executing the forward pass.

        Parameters
        ----------
        cp_group : ProcessGroup
                  context parallel process group.
        cp_global_ranks : List[int]
                         list of global ranks in the context group.
        cp_stream : torch.cuda.Stream
                   cuda stream for context parallel execution.
        """
        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_context_parallel_group"):
                child.set_context_parallel_group(cp_group, cp_global_ranks, cp_stream)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attn_mask_type: Optional[str] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: bool = False,
        inference_params: Optional[InferenceParams] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Forward propagation for MultiheadAttention layer.

        .. note::

            Argument :attr:`attention_mask` will be ignored when :attr:`attn_mask_type`
            is set to `"causal"`.

        Parameters
        ----------
        hidden_states : torch.Tensor
             Input tensor.
        attention_mask : Optional[torch.Tensor], default = `None`
             Boolean tensor used to mask out self-attention softmax input.
        attn_mask_type: {'causal', 'padding', 'no_mask', arbitrary}, default = `None`
                       type of attention mask passed into softmax operation.
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

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type

        if attn_mask_type == "padding" and attention_mask is not None:
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

        if self.attention_type == "self":
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn]
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

            num_queries_per_key_value = (self.num_attention_heads_per_partition //
                                         self.num_gqa_groups_per_partition)
            if self.qkv_weight_interleaved:
                # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, ng, (np/ng + 2), hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    self.num_gqa_groups_per_partition,
                    (num_queries_per_key_value + 2),
                    self.hidden_size_per_attention_head,
                )
                # split along second last dimension
                split_dim = -2
            else:
                # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, (np/ng + 2), ng, hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    (num_queries_per_key_value + 2),
                    self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head
                )
                # split along third last dimension
                split_dim = -3

            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # qkv_weight_interleaved:
            #  [sq, b, ng, (np/ng + 2), hn]
            #  --> [sq, b, ng, np/ng, hn], [sq, b, ng, 1, hn], [sq, b, ng, 1, hn]
            # not qkv_weight_interleaved:
            #  [sq, b, (np/ng + 2), ng, hn]
            #  --> [sq, b, np/ng, np, hn], [sq, b, 1, ng, hn], [sq, b, 1, ng, hn]
            if not is_in_onnx_export_mode():
                query_layer, key_layer, value_layer = _SplitAlongDim.apply(
                    mixed_x_layer, split_dim, (num_queries_per_key_value, 1, 1)
                )
            else:
                query_layer, key_layer, value_layer = torch.split(
                    mixed_x_layer, (num_queries_per_key_value, 1, 1), dim = split_dim,
                 )

            # query: -> [sq, b, np, hn]
            # key, value: -> [sq, b, ng, hn]
            query_layer, key_layer, value_layer = (x.reshape(x.size(0), x.size(1), -1,
                                                             self.hidden_size_per_attention_head)
                                                   for x in (query_layer, key_layer, value_layer))

        elif self.attention_type == "cross":
            # Attention heads [sk, b, h] --> [sk, b, (ng * 2 * hn)]
            mixed_kv_layer = self.key_value(
                encoder_output,
                is_first_microbatch=is_first_microbatch,
            )

            if self.qkv_weight_interleaved:
                # [sq, b, (ng * 2 * hn)] --> [sq, b, ng, 2 * hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    self.num_gqa_groups_per_partition,
                    2 * self.hidden_size_per_attention_head,
                )
                # split along last dimension
                split_dim = -1
            else:
                # [sq, b, (ng * 2 * hn)] --> [sq, b, 2 * ng, hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    2 * self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
                # split along second last dimension
                split_dim = -2

            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # mixed_kv_layer --> 2 [sk, b, ng, hn]
            if not is_in_onnx_export_mode():
                key_layer, value_layer = _SplitAlongDim.apply(
                    mixed_kv_layer, split_dim, mixed_kv_layer.shape[split_dim] // 2,
                )
            else:
                key_layer, value_layer = torch.split(
                    mixed_kv_layer, mixed_kv_layer.shape[split_dim] // 2, dim = split_dim,
                )

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
            value_layer = value_layer.contiguous()

        context_layer = self.core_attention(
            query_layer,
            key_layer,
            value_layer,
            qkv_format='sbhd',
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
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
