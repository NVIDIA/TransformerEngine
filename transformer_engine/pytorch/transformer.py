# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer."""
import os
import math
import warnings
from importlib.metadata import version
from contextlib import nullcontext
from typing import Any, Callable, Optional, Tuple, Union

import torch

from flash_attn.flash_attn_interface import flash_attn_unpadded_func

from transformer_engine.pytorch.module import LayerNormLinear, Linear, LayerNormMLP, LayerNorm
from transformer_engine.pytorch.jit import (
    set_jit_fusion_options,
    warmup_jit_bias_dropout_add_all_dtypes,
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from transformer_engine.pytorch.utils import (
    divide,
    attention_mask_func,
    split_tensor_along_dim,
    cast_if_needed,
    get_default_init_method,
    get_device_compute_capability,
)
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    LayerTypes,
    dist_group_type,
)
from transformer_engine.pytorch.softmax import FusedScaleMaskSoftmax
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    checkpoint,
)

_flash_attn_version = version("flash-attn")
warnings.filterwarnings("module", category=DeprecationWarning, module="transformer")


__all__ = ["DotProductAttention", "TransformerLayer"]


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """DropPath FWD"""

        if self.drop_prob == 0.0 or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=hidden_state.dtype, device=hidden_state.device
        )
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output

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
        data_ptr = grad_outputs[0].untyped_storage().data_ptr()
        shape = grad_outputs[0].shape
        last_dim_size = grad_outputs[0].shape[-1]
        for i, tensor in enumerate(grad_outputs):
            if (tensor.stride() != strides or
                tensor.shape != shape or
                tensor.untyped_storage().data_ptr() != data_ptr or
                tensor.storage_offset() != i * last_dim_size):
                noop_ok = False
                break

        if noop_ok:
            ret = torch.Tensor().to(grad_outputs[0].dtype)
            ret = torch.Tensor().to(device=grad_outputs[0].device,
                                    dtype=grad_outputs[0].dtype)
            new_shape = list(shape)
            new_shape[-1] = new_shape[-1] * len(grad_outputs)
            ret.set_(grad_outputs[0].untyped_storage(),
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
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

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

def _check_if_interleaved(q, k, v):
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

class FlashAttention(torch.nn.Module):
    """Dot product attention implementation by using the flash-attn package.
    """

    def __init__(
        self,
        norm_factor: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attn_mask_type: str = "causal",
    ) -> None:
        super().__init__()

        if "dev" not in _flash_attn_version:
            raise ImportError(
                'Please install correct version of flash-attn with ' \
                'pip install git+https://github.com/ksivaman/flash-attention.git@hopper. ' \
                'If running on Hopper, ' \
                'please install from source with compute capability 9.0.')
        assert (
            attn_mask_type == "causal"
            ), 'FlashAttention currently only supports causal attention mask.'

        self.attn_causal_mask = attn_mask_type == "causal"
        self.norm_factor = norm_factor
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_dropout = attention_dropout

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """flash-attn fprop"""

        assert (
            (query_layer.dtype in [torch.float16, torch.bfloat16])
            and (key_layer.dtype in [torch.float16, torch.bfloat16])
            and (value_layer.dtype in [torch.float16, torch.bfloat16])
            ), 'FlashAttention currently only supports FP16 and BF16.'
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
            ), 'FlashAttention currently only supports CUDA tensors.'
        assert (
            attention_mask is None
        ), 'FlashAttention currently does not support external attention mask.'

        # For now just 128, will make it more general in the future

        if (query_layer.shape[-1] == 128 and
            query_layer.shape[0] * query_layer.shape[1] >= 512 and
            _check_if_interleaved(query_layer, key_layer, value_layer)):
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
            output = flash_attn_unpadded_func(
                query_layer, key_layer, value_layer, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                self.attention_dropout if self.training else 0.0,
                softmax_scale=1.0/self.norm_factor, causal=self.attn_causal_mask
            )

        # [(b sq), np, hn] -> [sq, b, (np hn)]
        return output.view(batch_size, seqlen, -1).transpose(0, 1).contiguous()


class DotProductAttention(torch.nn.Module):
    """Allows the model to jointly attend to information from different
    representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. note::

        Argument :attr:`attention_mask` will be ignored in the `forward` call when
        :attr:`attn_mask_type` is set to `"causal"`.

    .. warning::

        For the default attention mechanism, this module executes a non-deterministic version of
        `flash-attn <https://github.com/ksivaman/flash-attention>`_ whenever possible in order to
        achieve optimal performance. To observe deterministic behavior, set the environment
        variable :attr:`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`. In order to disable
        `flash-attn` entirely, set :attr:`NVTE_FLASH_ATTN=0`.

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
    ) -> None:
        super().__init__()

        tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.tp_group = tp_group
        self.get_rng_state_tracker = get_rng_state_tracker

        projection_size = kv_channels * num_attention_heads
        self.hidden_size_per_partition = divide(projection_size, tp_size)
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
            and attn_mask_type == "causal"
            and self.device_compute_capability >= 8.0
        )

        attn_kwargs = {
            "attention_dropout": attention_dropout,
            "attention_dropout_ctx": attention_dropout_ctx,
            "attn_mask_type": attn_mask_type,
        }

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(norm_factor, **attn_kwargs)
        # Instantiating both types since use of flash-attn
        # might be ruled out due to forward inputs.
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
        """

        use_flash_attention = self.use_flash_attention
        if (query_layer.dtype not in [torch.bfloat16, torch.float16]
            or key_layer.dtype not in [torch.bfloat16, torch.float16]
            or value_layer.dtype not in [torch.bfloat16, torch.float16]
            or (self.device_compute_capability == 8.6 and key_layer.shape[-1] > 64)
        ):
            use_flash_attention = False

        if use_flash_attention:
            if checkpoint_core_attention:
                return self._checkpointed_attention_forward(self.flash_attention,
                                                            query_layer,
                                                            key_layer,
                                                            value_layer)
            return self.flash_attention(query_layer, key_layer, value_layer)

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
        params_dtype: torch.dtype = torch.float32,
        return_layernorm_output: bool = False,
        input_layernorm: bool = False,
        attention_type: str = "self",
        set_parallel_mode: bool = False,
        fuse_qkv_params: bool = False,
        zero_centered_gamma: bool = False,
        qkv_weight_interleaved: bool = True,
    ) -> None:
        super().__init__()
        self.layer_number = (layer_number,)
        self.input_layernorm = input_layernorm
        self.attention_type = attention_type
        self.get_rng_state_tracker = get_rng_state_tracker
        self.tp_group = tp_group
        self.return_layernorm_output = return_layernorm_output
        self.params_dtype = params_dtype
        self.init_method = init_method
        self.attn_mask_type = attn_mask_type

        if not fuse_qkv_params:
            qkv_weight_interleaved = False
        self.qkv_weight_interleaved = qkv_weight_interleaved

        assert (
            attention_type in AttnTypes
        ), f"attention_type {attention_type} not supported"

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
            "params_dtype": params_dtype,
        }

        qkv_parallel_mode = "column" if set_parallel_mode else None

        if self.attention_type == "self":
            if self.input_layernorm:
                self.layernorm_qkv = LayerNormLinear(
                    hidden_size,
                    3 * hidden_size,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=True,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    return_layernorm_output=return_layernorm_output,
                    parameters_split=("query_", "key_", "value_") if not fuse_qkv_params else None,
                    zero_centered_gamma=zero_centered_gamma,
                    **common_gemm_kwargs,
                )
            else:
                self.qkv = Linear(
                    hidden_size,
                    3 * hidden_size,
                    init_method=init_method,
                    bias=True,
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
                    bias=True,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    return_layernorm_output=return_layernorm_output,
                    zero_centered_gamma=zero_centered_gamma,
                    **common_gemm_kwargs,
                )
            else:
                self.query_layer = Linear(
                    hidden_size,
                    hidden_size,
                    init_method=init_method,
                    bias=True,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    **common_gemm_kwargs,
                )
            self.key_value = Linear(
                hidden_size,
                2 * hidden_size,
                init_method=init_method,
                bias=True,
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
            layer_number=layer_number,
        )

        # Linear
        self.proj = Linear(
            hidden_size,
            hidden_size,
            init_method=output_layer_init_method,
            bias=True,
            return_bias=True,
            parallel_mode="row" if set_parallel_mode else None,
            **common_gemm_kwargs,
        )


    def _allocate_memory(
        self, inference_max_sequence_len: int, batch_size: int
    ) -> torch.Tensor:
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
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

        if inference_params and self.layer_number is not None:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size
                )
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size
                )
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory,
                    inference_value_memory,
                )
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

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            checkpoint_core_attention=checkpoint_core_attention,
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


class TransformerLayer(torch.nn.Module):
    r"""
    TransformerLayer is made up of an attention block and a feedforward network (MLP).
    This standard layer is based on the paper "Attention Is All You Need".

    .. warning::

        Arguments :attr:`attention_softmax_in_fp32` and :attr:`apply_query_key_layer_scaling`
        are deprecated and will be fully removed in future releases.

    .. note::

        Argument :attr:`attention_mask` will be ignored in the `forward` call when
        :attr:`self_attn_mask_type` is set to `"causal"`.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    layernorm_epsilon : float, default = 1e-5
                       a value added to the denominator of layer normalization
                       for numerical stability.
    hidden_dropout: float, default = 0.1
                   dropout probability for the dropout op after FC2 layer.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    init_method : Callable, default = `None`
                 used for initializing weights of QKV and FC1 weights in the following way:
                 `init_method(weight)`. When set to `None`, defaults to
                 `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing weights of PROJ and FC2 in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    apply_residual_connection_post_layernorm : bool, default = `False`
                                              if set to `True`, residual connections are taken
                                              from the output of layer norm (default is taken
                                              from input of layer norm)
    layer_number: int, default = `None`
                 layer number of the current `TransformerLayer` when multiple such modules are
                 concatenated to form a transformer block.
    output_layernorm: bool, default = `False`
                     if set to `True`, layer normalization is applied on the output side,
                     after the final dropout-add. default behavior is to apply layer
                     normalization on the input side, before the QKV transformation.
    layer_type: {'encoder', 'decoder'}, default = `encoder`
               if set to `decoder`, an additional cross-attn block is added after self-attn.
               This can be used for structures like `T5` Transformer in conjunction with the
               `encoder` option.
    kv_channels: int, default = `None`
                number of key-value channels. defaults to
                :attr:`hidden_size` / :attr:`num_attention_heads` if `None`.
    self_attn_mask_type: {'causal', 'padding'}, default = `causal`
                        type of attention mask passed into softmax operation.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    qkv_weight_interleaved : bool, default = `True`
                            if set to `False`, the QKV weight is interpreted as a concatenation of
                            query, key, and value weights along the `0th` dimension. The default
                            interpretation is that the individual `q`, `k`, and `v` weights for each
                            attention head are interleaved. This parameter is set to `False` when
                            using :attr:`fuse_qkv_params=False`.
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
                             the weight gradient.
    params_dtype : torch.dtype, default = `torch.float32`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    seq_length: int
               sequence length of input samples. Needed for JIT Warmup, a technique where jit
               fused functions are warmed up before training to ensure same kernels are used for
               forward propogation and activation recompute phase.
    micro_batch_size: int
                     batch size per training step. Needed for JIT Warmup, a technique where jit
                     fused functions are warmed up before training to ensure same kernels are
                     used for forward propogation and activation recompute phase.
    drop_path_rate: float, default = 0.0
                   when > 0.0, applies stochastic depth per sample in
                   the main path of the residual block.
    fuse_qkv_params: bool, default = 'False'
                    if set to `True`, `TransformerLayer` module exposes a single fused
                    parameter for query-key-value. This enables optimizations such as QKV
                    fusion without concatentations/splits and also enables the argument
                    `fuse_wgrad_accumulation`.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        layer_number: Optional[int] = None,
        kv_channels: Optional[int] = None,
        self_attn_mask_type: str = "causal",
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        params_dtype: torch.dtype = torch.float32,
        get_rng_state_tracker: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        apply_query_key_layer_scaling: bool = False, # pylint: disable=unused-argument
        attention_softmax_in_fp32: bool = True, # pylint: disable=unused-argument
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        sequence_parallel: bool = False,
        apply_residual_connection_post_layernorm: bool = False,
        output_layernorm: bool = False,
        layer_type: str = "encoder",
        drop_path_rate: float = 0.0,
        set_parallel_mode: bool = False,
        fuse_qkv_params: bool = False,
        zero_centered_gamma: bool = False,
        qkv_weight_interleaved: bool = True,
    ) -> None:
        super().__init__()

        warnings.warn(
            "Arguments `attention_softmax_in_fp32` and `apply_query_key_layer_scaling`"
            "are deprecated and will be fully removed in future releases.",
            category=DeprecationWarning,
        )

        bias_dropout_fusion = bool(int(os.getenv("NVTE_BIAS_DROPOUT_FUSION", "1")))
        self.layer_number = layer_number
        self.output_layernorm = output_layernorm
        self.layer_type = layer_type
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm
        )
        self.self_attn_mask_type = self_attn_mask_type
        assert (
            self_attn_mask_type in AttnMaskTypes
        ), f"self_attn_mask_type {self_attn_mask_type} not supported"
        assert layer_type in LayerTypes, f"layer_type {layer_type} not supported"

        if not fuse_qkv_params:
            assert (
                not fuse_wgrad_accumulation
            ), "Gradient accumulation fusion requires single QKV parameter."

        if not fuse_qkv_params:
            qkv_weight_interleaved = False

        self.kv_channels = (
            kv_channels if kv_channels else (hidden_size // num_attention_heads)
        )

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.sequence_parallel = (tp_size > 1) and sequence_parallel

        self.get_rng_state_tracker = get_rng_state_tracker

        attention_args = (
            hidden_size,
            num_attention_heads,
            self.kv_channels,
            attention_dropout,
            layernorm_epsilon,
            init_method,
            output_layer_init_method,
        )
        common_attention_kwargs = {
            "layer_number": layer_number,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "get_rng_state_tracker": get_rng_state_tracker,
            "sequence_parallel": self.sequence_parallel,
            "params_dtype": params_dtype,
            "return_layernorm_output": apply_residual_connection_post_layernorm,
            "set_parallel_mode": set_parallel_mode,
            "fuse_qkv_params": fuse_qkv_params,
            "zero_centered_gamma": zero_centered_gamma,
            "qkv_weight_interleaved" : qkv_weight_interleaved,
        }

        self.self_attention = MultiHeadAttention(
            *attention_args,
            **common_attention_kwargs,
            attn_mask_type=self_attn_mask_type,
            input_layernorm=not output_layernorm,
            attention_type="self",
        )

        if layer_type == "decoder":
            self.inter_attention = MultiHeadAttention(
                *attention_args,
                **common_attention_kwargs,
                attn_mask_type="padding",
                input_layernorm=True,
                attention_type="cross",
            )

        # LayerNorm -> gelu(Linear + Bias) -> Linear
        # parallel_mode not supported for LayerNormMLP,
        # FC1 is CPL and FC2 is RPL
        self.layernorm_mlp = LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            eps=layernorm_epsilon,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            tp_group=tp_group,
            tp_size=tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            bias=True,
            return_bias=True,
            sequence_parallel=self.sequence_parallel,
            params_dtype=params_dtype,
            return_layernorm_output=apply_residual_connection_post_layernorm,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            set_parallel_mode=set_parallel_mode,
            zero_centered_gamma=zero_centered_gamma,
        )

        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = (
            nullcontext if use_nvfuser else torch.enable_grad
        )

        if self.bias_dropout_fusion:
            set_jit_fusion_options()
            if seq_length and micro_batch_size:
                if self.sequence_parallel:
                    seq_length = seq_length // tp_size
                warmup_jit_bias_dropout_add_all_dtypes(
                    hidden_size, seq_length, micro_batch_size
                )

        if self.output_layernorm:
            self.layernorm = LayerNorm(
                hidden_size,
                eps=layernorm_epsilon,
                sequence_parallel=self.sequence_parallel,
                params_dtype=params_dtype,
                zero_centered_gamma=zero_centered_gamma
            )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """Set TP group"""
        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_tensor_parallel_group"):
                child.set_tensor_parallel_group(tp_group)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        enc_dec_attn_mask: Optional[torch.Tensor] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: bool = False,
        inference_params: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Transformer Layer: attention block and a feedforward network (MLP)

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
        enc_dec_attn_mask : Optional[torch.Tensor], default = `None`
             Boolean tensor used to mask out inter-attention softmax input if using
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
        """

        hidden_states = hidden_states.contiguous()

        if self.self_attn_mask_type != "causal" and attention_mask is not None:
            assert (
                attention_mask.dtype == torch.bool
            ), "Attention mask must be a boolean tensor"

        # For AMP
        if torch.is_autocast_enabled():
            hidden_states = cast_if_needed(
                hidden_states, torch.get_autocast_gpu_dtype()
            )

        # Self attention.
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            inference_params=inference_params,
            is_first_microbatch=is_first_microbatch,
            checkpoint_core_attention=checkpoint_core_attention,
        )
        if self.apply_residual_connection_post_layernorm and not self.output_layernorm:
            attention_output, attention_bias, residual = self_attention_outputs
        else:
            attention_output, attention_bias = self_attention_outputs
            residual = hidden_states

        # Set BDA func.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # Bias dropoout add.
        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                bda_output = bias_dropout_add_func(
                    attention_output, attention_bias, residual, self.hidden_dropout
                )
        else:
            out = torch.nn.functional.dropout(
                attention_output + attention_bias,
                p=self.hidden_dropout,
                training=self.training,
            )
            bda_output = residual + self.drop_path(out)

        # Cross attention.
        if self.layer_type == "decoder":
            inter_attention_outputs = self.inter_attention(
                bda_output,
                enc_dec_attn_mask,
                encoder_output=encoder_output,
                is_first_microbatch=is_first_microbatch,
                checkpoint_core_attention=checkpoint_core_attention,
            )
            if self.apply_residual_connection_post_layernorm:
                attention_output, attention_bias, residual = inter_attention_outputs
            else:
                attention_output, attention_bias = inter_attention_outputs
                residual = bda_output

            with self.bias_dropout_add_exec_handler():
                bda_output = bias_dropout_add_func(
                    attention_output, attention_bias, residual, self.hidden_dropout
                )

        # MLP.
        mlp_outputs = self.layernorm_mlp(
            bda_output, is_first_microbatch=is_first_microbatch
        )
        if self.apply_residual_connection_post_layernorm:
            mlp_output, mlp_bias, residual = mlp_outputs
        else:
            mlp_output, mlp_bias = mlp_outputs
            residual = bda_output

        # Bias dropoout add.
        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output, mlp_bias, residual, self.hidden_dropout
                )
        else:
            out = torch.nn.functional.dropout(
                mlp_output + mlp_bias, p=self.hidden_dropout, training=self.training
            )
            output = residual + self.drop_path(out)

        # For BERT like architectures.
        if self.output_layernorm:
            output = self.layernorm(output)

        # output: [b, s, h]
        return output
