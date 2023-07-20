# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention."""
import os
import math
from importlib.metadata import version
from contextlib import nullcontext
from typing import Any, Callable, Optional, Tuple, Union
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
)
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    dist_group_type,
    TE_DType,
)
from transformer_engine.pytorch.softmax import FusedScaleMaskSoftmax
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    checkpoint,
)
from transformer_engine.pytorch.export import is_in_onnx_export_mode

_flash_attn_version = packaging.version.Version(version("flash-attn"))
_flash_attn_version_required = packaging.version.Version("1.0.6")
_flash_attn_2_available = _flash_attn_version >= packaging.version.Version("2")

if _flash_attn_2_available:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_forward_func # pylint: disable=no-name-in-module
else:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func as flash_attn_forward_func # pylint: disable=no-name-in-module


__all__ = ["DotProductAttention"]


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

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """core attention fprop"""
        batch_size, seqlen = query_layer.shape[1], query_layer.shape[0]
        apply_qk_layer_scaling = self.layer_number is not None and key_layer.dtype == torch.float16

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

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
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / scale),
        )

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


def _check_if_interleaved_qkv(q, k, v):
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
    return check_offsets

def _check_if_interleaved_kv(k, v):
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
    return check_offsets



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
    ) -> None:
        super().__init__()

        assert (
            _flash_attn_version >= _flash_attn_version_required
        ), f"FlashAttention minimum version {_flash_attn_version_required} is required."

        self.attn_causal_mask = attn_mask_type == "causal"
        self.norm_factor = norm_factor
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_dropout = attention_dropout
        self.deterministic = not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
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
            _check_if_interleaved_qkv(query_layer, key_layer, value_layer)):
            query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(query_layer,
                                                                         key_layer,
                                                                         value_layer)
        else:
            query_layer, key_layer, value_layer = [x.transpose(0,1).contiguous()
                           for x in (query_layer, key_layer, value_layer)]

        batch_size, seqlen = query_layer.shape[0], query_layer.shape[1]

        # [b, sq, np, hn]
        query_layer, key_layer, value_layer = [
            x.view(x.shape[0] * x.shape[1], *x.shape[2:])
            for x in [query_layer, key_layer, value_layer]
        ]

        max_seqlen = seqlen
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seqlen,
            step=seqlen,
            dtype=torch.int32,
            device=query_layer.device)

        with self.attention_dropout_ctx():
            fa_optional_forward_kwargs = {}
            if not _flash_attn_2_available:
                fa_optional_forward_kwargs["deterministic"] = self.deterministic
            output = flash_attn_forward_func(
                query_layer, key_layer, value_layer, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                self.attention_dropout if self.training else 0.0,
                softmax_scale=1.0/self.norm_factor, causal=self.attn_causal_mask,
                **fa_optional_forward_kwargs
            )

        # [(b sq), np, hn] -> [sq, b, (np hn)]
        return output.view(batch_size, seqlen, -1).transpose(0, 1).contiguous()


class FusedAttnFunc_qkvpacked(torch.autograd.Function):
    """Function for FusedAttention with packed QKV input"""

    @staticmethod
    def forward(ctx, is_training, max_seqlen, cu_seqlens, qkv, qkv_dtype, attn_bias, attn_scale,
                dropout_p, fast_zero_fill, qkv_layout, attn_bias_type, attn_mask_type,
                rng_gen, fused_attention_backend):
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

        return out

    @staticmethod
    def backward(ctx, d_out):
        qkv, out, cu_seqlens = ctx.saved_tensors
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
                rng_gen, fused_attention_backend):
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

        return out

    @staticmethod
    def backward(ctx, d_out):
        q, kv, out, cu_seqlens_q, cu_seqlens_kv = ctx.saved_tensors
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

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
    ) -> torch.Tensor:
        """fused attention fprop"""

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
            if _check_if_interleaved_qkv(query_layer, key_layer, value_layer):
                query_layer = query_layer.unsqueeze(3)
                key_layer = key_layer.unsqueeze(3)
                value_layer = value_layer.unsqueeze(3)
                # [s, b, h, 3, d]
                mixed_layer = torch.cat([query_layer, key_layer, value_layer], dim = 3)
                # [b, s, 3, h, d]
                mixed_layer = mixed_layer.transpose(2, 3).transpose(0, 1).contiguous()
            else:
                query_layer = query_layer.unsqueeze(2)
                key_layer = key_layer.unsqueeze(2)
                value_layer = value_layer.unsqueeze(2)
                # [s, b, 3, h, d]
                mixed_layer = torch.cat([query_layer, key_layer, value_layer], dim = 2)
                # [b, s, 3, h, d]
                mixed_layer = mixed_layer.transpose(0, 1).contiguous()

            # [total_seqs, 3, h, d]
            mixed_layer = mixed_layer.view(
                mixed_layer.shape[0] * mixed_layer.shape[1], *mixed_layer.shape[2:]).contiguous()

            qkv_layout = "qkv_interleaved"
            max_seqlen = seqlen_q
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seqlen_q,
                step=seqlen_q,
                dtype=torch.int32,
                device=query_layer.device)

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
                )
            output = output.view(batch_size, seqlen_q, -1).transpose(0, 1).contiguous()

        if self.attention_type == "cross":
            if _check_if_interleaved_kv(key_layer, value_layer):
                # [s, b, h, 2, d]
                key_layer = key_layer.unsqueeze(3)
                value_layer = value_layer.unsqueeze(3)
                key_value = torch.cat([key_layer, value_layer], dim = 3)
                # [b, s, 2, h, d]
                key_value = key_value.transpose(2, 3).transpose(0, 1).contiguous()
            else:
                # [s, b, 2, h, d]
                key_layer = key_layer.unsqueeze(2)
                value_layer = value_layer.unsqueeze(2)
                key_value = torch.cat([key_layer, value_layer], dim = 2)
                # [b, s, 2, h, d]
                key_value = key_value.transpose(0, 1).contiguous()

            # [total_seqs, 2, h, d]
            query_layer = query_layer.transpose(0, 1).contiguous()
            query_layer = query_layer.view(
                    query_layer.shape[0] * query_layer.shape[1], *query_layer.shape[2:])
            key_value = key_value.view([key_value.shape[0] * key_value.shape[1]]
                + key_value.shape[2:]).contiguous()

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
    attention_dropout: float, default = 0.0
                      dropout probability for the dropout op during multi-head attention.
    attn_mask_type: {'causal', 'padding'}, default = `causal`
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
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float = 0.0,
        attn_mask_type: str = "causal",
        sequence_parallel: bool = False,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        attention_type: str = "self",
    ) -> None:
        super().__init__()

        self.tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.tp_group = tp_group
        self.get_rng_state_tracker = get_rng_state_tracker

        projection_size = kv_channels * num_attention_heads
        self.hidden_size_per_partition = divide(projection_size, self.tp_size)
        self.hidden_size_per_attention_head = divide(
            projection_size, num_attention_heads
        )

        if sequence_parallel or get_rng_state_tracker is None:
            attention_dropout_ctx = nullcontext
        else:
            attention_dropout_ctx = get_rng_state_tracker().fork

        norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        self.device_compute_capability = get_device_compute_capability()
        self.use_flash_attention = (
            int(os.getenv("NVTE_FLASH_ATTN", "1"))
            and self.device_compute_capability >= 8.0
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
            self.flash_attention = FlashAttention(norm_factor, **attn_kwargs)
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
    ) -> torch.Tensor:
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            return attention_func(*inputs)

        hidden_states = checkpoint(
            custom_forward,
            False,
            self.get_rng_state_tracker,
            self.tp_group,
            *forward_args,
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

        if self.attn_mask_type == "padding" and attention_mask is not None:
            use_flash_attention = False
            use_fused_attention = False

        if is_in_onnx_export_mode():
            use_flash_attention = False
            use_fused_attention = False

        qkv_layout = "qkv_interleaved" if self.attention_type == "self" else "kv_interleaved"
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
        use_fused_attention = use_fused_attention and is_backend_avail

        if use_flash_attention:
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(self.flash_attention,
                                                            query_layer,
                                                            key_layer,
                                                            value_layer)
            return self.flash_attention(query_layer, key_layer, value_layer)

        if use_fused_attention:
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(self.fused_attention,
                                                            query_layer,
                                                            key_layer,
                                                            value_layer,
                                                            fused_attention_backend,
                                                            core_attention_bias_type,
                                                            core_attention_bias,
                                                            fast_zero_fill)
            return self.fused_attention(query_layer, key_layer, value_layer,
                                                            fused_attention_backend,
                                                            core_attention_bias_type,
                                                            core_attention_bias,
                                                            fast_zero_fill)

        if checkpoint_core_attention:
            return self._checkpointed_attention_forward(
                self.unfused_attention,
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
            )
        return self.unfused_attention(query_layer, key_layer, value_layer, attention_mask)


class MultiHeadAttention(torch.nn.Module):
    """Parallel attention w/o QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
        layernorm_epsilon: float,
        init_method: Callable,
        output_layer_init_method: Callable,
        layer_number: Optional[int] = None,
        attn_mask_type: str = "causal",
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        fuse_wgrad_accumulation: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        sequence_parallel: bool = False,
        params_dtype: Optional[torch.dtype] = None,
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
    ) -> None:
        super().__init__()
        self.layer_number = layer_number
        self.input_layernorm = input_layernorm
        self.attention_type = attention_type
        self.get_rng_state_tracker = get_rng_state_tracker
        self.tp_group = tp_group
        self.return_layernorm_output = return_layernorm_output
        self.params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.init_method = init_method
        self.attn_mask_type = attn_mask_type

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

        common_gemm_kwargs = {
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "get_rng_state_tracker": get_rng_state_tracker,
            "sequence_parallel": sequence_parallel,
            "params_dtype": self.params_dtype,
        }

        qkv_parallel_mode = "column" if set_parallel_mode else None

        if self.attention_type == "self":
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
        else:
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
                2 * hidden_size,
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
            attention_dropout,
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
            return_bias=True,
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
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """Set TP group"""
        self.tp_group = tp_group

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
        """MultiHeadAttention FWD"""
        # hidden_states: [sq, b, h]

        if self.attn_mask_type != "causal" and attention_mask is not None:
            assert (
                attention_mask.dtype == torch.bool
            ), "Attention mask must be a boolean tensor"

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
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer = self.key_value(
                encoder_output,
                is_first_microbatch=is_first_microbatch,
            )

            if self.qkv_weight_interleaved:
                # [sq, b, (np * 2 * hn)] --> [sq, b, np, 2 * hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    self.num_attention_heads_per_partition,
                    2 * self.hidden_size_per_attention_head,
                )
                # split along last dimension
                split_dim = -1
            else:
                # [sq, b, (np * 2 * hn)] --> [sq, b, 2 * np, hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    2 * self.num_attention_heads_per_partition,
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
            checkpoint_core_attention = checkpoint_core_attention,
            core_attention_bias_type = core_attention_bias_type,
            core_attention_bias = core_attention_bias,
            fast_zero_fill = fast_zero_fill,
        )

        # =================
        # Output. [sq, b, h]
        # =================

        attention_output, attention_bias = self.proj(
            context_layer, is_first_microbatch=is_first_microbatch
        )

        if self.input_layernorm and self.return_layernorm_output:
            return attention_output, attention_bias, layernorm_output
        return attention_output, attention_bias
