# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer."""
import os
import math
from contextlib import nullcontext
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch.nn.parameter import Parameter

from transformer_engine.pytorch import LayerNormLinear, Linear, LayerNormMLP, LayerNorm
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
    split_tensor_along_last_dim,
    cast_if_needed,
    get_default_init_method,
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
    initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)


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


class CoreAttention(torch.nn.Module):
    """Parallel attention w/o QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
        layer_number: Optional[int] = None,
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = False,
        attn_mask_type: str = "causal",
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32

        if layer_number is None:
            self.apply_query_key_layer_scaling = False
        else:
            self.layer_number = max(1, layer_number)

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        self.attn_mask_type = attn_mask_type
        projection_size = kv_channels * num_attention_heads
        assert (
            attn_mask_type in AttnMaskTypes
        ), f"attn_mask_type {attn_mask_type} not supported"

        # Per attention head and per partition values.
        self.hidden_size_per_partition = divide(projection_size, tp_size)
        self.hidden_size_per_attention_head = divide(
            projection_size, num_attention_heads
        )

        self.sequence_parallel = sequence_parallel
        if self.sequence_parallel or get_rng_state_tracker is None:
            self.attention_dropout_ctx = nullcontext
        else:
            self.attention_dropout_ctx = get_rng_state_tracker().fork

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.attn_mask_type,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
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
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """core attention fprop"""
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

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
        value_layer = value_layer.view(
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
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


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
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = False,
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
        self.fuse_qkv_params = fuse_qkv_params

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
        if not fuse_qkv_params:
            self.set_qkv_params(
                hidden_size,
                3 * hidden_size,
                parallel_mode=qkv_parallel_mode,
                bias=True,
            )

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
                    skip_weight_param_allocation=not fuse_qkv_params,
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
                    skip_weight_param_allocation=not fuse_qkv_params,
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
                    skip_weight_param_allocation=not fuse_qkv_params,
                    **common_gemm_kwargs,
                )
            else:
                self.query = Linear(
                    hidden_size,
                    hidden_size,
                    init_method=init_method,
                    bias=True,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    skip_weight_param_allocation=not fuse_qkv_params,
                    **common_gemm_kwargs,
                )
            self.key_value = Linear(
                hidden_size,
                2 * hidden_size,
                init_method=init_method,
                bias=True,
                return_bias=False,
                parallel_mode=qkv_parallel_mode,
                skip_weight_param_allocation=not fuse_qkv_params,
                **common_gemm_kwargs,
            )

        # Core Self attention.
        self.core_attention = CoreAttention(
            num_attention_heads,
            kv_channels,
            attention_dropout,
            layer_number=layer_number,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            attention_softmax_in_fp32=attention_softmax_in_fp32,
            tp_size=tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            attn_mask_type=attn_mask_type,
            sequence_parallel=sequence_parallel,
        )

        # Linear
        self.proj = Linear(
            hidden_size,
            hidden_size,
            init_method=output_layer_init_method,
            bias=False,
            return_bias=True,
            parallel_mode="row" if set_parallel_mode else None,
            **common_gemm_kwargs,
        )

    def set_qkv_params(
        self,
        in_features: torch.Tensor,
        out_features: torch.Tensor,
        parallel_mode: Optional[bool] = None,
        bias: bool = False,
    ) -> None:
        """Initialize separate Parameters for query, key, and value tensors."""

        if parallel_mode == "column":
            out_features = divide(out_features, self.tp_size)
        elif parallel_mode == "row":
            in_features = divide(in_features, self.tp_size)

        assert (
            out_features % 3 == 0
        ), f"3 way QKV split with dimension {out_features} not possible."

        weight_tensor = torch.empty(
            out_features,
            in_features,
            device=torch.cuda.current_device(),
            dtype=self.params_dtype,
        )

        initialize_affine_weight_gpu(
            weight_tensor,
            self.init_method,
            self.get_rng_state_tracker,
            partition_dim=1 if parallel_mode == "row" else 0,
            stride=1,
        )

        qkv_first_dim = out_features // 3
        self.query = Parameter(weight_tensor[0:qkv_first_dim, :])
        self.key = Parameter(weight_tensor[qkv_first_dim : 2 * qkv_first_dim, :])
        self.value = Parameter(weight_tensor[2 * qkv_first_dim : 3 * qkv_first_dim, :])
        set_tensor_model_parallel_attributes(
            tensor=self.query,
            is_parallel=True,
            dim=1 if parallel_mode == "row" else 0,
            stride=1,
        )
        set_tensor_model_parallel_attributes(
            tensor=self.key,
            is_parallel=True,
            dim=1 if parallel_mode == "row" else 0,
            stride=1,
        )
        set_tensor_model_parallel_attributes(
            tensor=self.value,
            is_parallel=True,
            dim=1 if parallel_mode == "row" else 0,
            stride=1,
        )

        if bias:
            bias_tensor = torch.empty(
                out_features,
                device=torch.cuda.current_device(),
                dtype=self.params_dtype,
            )
            self.query_bias = Parameter(bias_tensor[0:qkv_first_dim])
            self.key_bias = Parameter(bias_tensor[qkv_first_dim : 2 * qkv_first_dim])
            self.value_bias = Parameter(
                bias_tensor[2 * qkv_first_dim : 3 * qkv_first_dim]
            )

            if parallel_mode == "column":
                set_tensor_model_parallel_attributes(self.query_bias, True, 0, 1)
                set_tensor_model_parallel_attributes(self.key_bias, True, 0, 1)
                set_tensor_model_parallel_attributes(self.value_bias, True, 0, 1)
        else:
            self.register_buffer("query_bias", torch.Tensor(), persistent=False)
            self.register_buffer("key_bias", torch.Tensor(), persistent=False)
            self.register_buffer("value_bias", torch.Tensor(), persistent=False)

        with torch.no_grad():
            self.query_bias.zero_()
            self.key_bias.zero_()
            self.value_bias.zero_()

    def _checkpointed_core_attention_forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask
            )
            return output_

        hidden_states = checkpoint(
            custom_forward,
            False,
            self.get_rng_state_tracker,
            self.tp_group,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
        )

        return hidden_states

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
        attention_mask: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: Optional[bool] = None,
        inference_params: Optional[Any] = None,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """MultiHeadAttention FWD"""
        # hidden_states: [sq, b, h]

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
            qkv_weight = (
                torch.cat((self.query, self.key, self.value))
                if not self.fuse_qkv_params
                else None
            )
            qkv_bias = (
                torch.cat((self.query_bias, self.key_bias, self.value_bias))
                if not self.fuse_qkv_params
                else None
            )

            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            if self.input_layernorm:
                layernorm_qkv_outputs = self.layernorm_qkv(
                    hidden_states,
                    weight=qkv_weight,
                    bias=qkv_bias,
                    is_first_microbatch=is_first_microbatch,
                )
                if self.return_layernorm_output:
                    mixed_x_layer, layernorm_output = layernorm_qkv_outputs
                else:
                    mixed_x_layer = layernorm_qkv_outputs
            else:
                mixed_x_layer = self.qkv(
                    hidden_states,
                    weight=qkv_weight,
                    bias=qkv_bias,
                    is_first_microbatch=is_first_microbatch,
                )

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            query_layer, key_layer, value_layer = split_tensor_along_last_dim(
                mixed_x_layer, 3
            )
        else:
            kv_weight = (
                torch.cat((self.key, self.value)) if not self.fuse_qkv_params else None
            )
            kv_bias = (
                torch.cat((self.key_bias, self.value_bias))
                if not self.fuse_qkv_params
                else None
            )

            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer = self.key_value(
                encoder_output,
                weight=kv_weight,
                bias=kv_bias,
                is_first_microbatch=is_first_microbatch,
            )

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            if self.input_layernorm:
                layernorm_query_outputs = self.layernorm_query(
                    hidden_states,
                    weight=self.query,
                    bias=self.query_bias,
                    is_first_microbatch=is_first_microbatch,
                )
                if self.return_layernorm_output:
                    query_layer, layernorm_output = layernorm_query_outputs
                else:
                    query_layer = layernorm_query_outputs
            else:
                query_layer = self.query(
                    hidden_states,
                    weight=self.query,
                    bias=self.query_bias,
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

        if checkpoint_core_attention:
            context_layer = self._checkpointed_core_attention_forward(
                query_layer, key_layer, value_layer, attention_mask
            )
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask
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
    """
    TransformerLayer is made up of an attention block and a feedforward network (MLP).
    This standard layer is based on the paper "Attention Is All You Need".

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
    apply_query_key_layer_scaling: bool, default = `True`
                                  apply query-key layer scaling during BMM1
                                  by a factor of `layer_number`
    output_layernorm: bool, default = `False`
                     if set to `True`, layer normalization is applied on the output side,
                     after the final dropout-add. default behavior is to apply layer
                     normalization on the input side, before the QKV transformation.
    attention_softmax_in_fp32: bool, default = `False`
                              if set to `True`, softmax is executed in
                              torch.float32 dtype (single precision)
    layer_type: {'encoder', 'decoder'}, default = `encoder`
               if set to `decoder`, an additional cross-attn block is added after self-attn.
               This can be used for structures like `T5` Transformer in conjunction with the
               `encoder` option.
    kv_channels: int, default = `None`
                number of key-value channels. defaults to
                :attr:`hidden_size` / :attr:`num_attention_heads` if `None`.
    self_attn_mask_type: {'causal', 'padding'}, default = `causal`
                        type of attention mask passed into softmax operation.

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
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        sequence_parallel: bool = False,
        apply_residual_connection_post_layernorm: bool = False,
        output_layernorm: bool = False,
        layer_type: str = "encoder",
        drop_path_rate: float = 0.0,
        set_parallel_mode: bool = False,
        fuse_qkv_params: bool = False,
    ) -> None:
        super().__init__()

        bias_dropout_fusion = bool(int(os.getenv("NVTE_BIAS_DROPOUT_FUSION", "1")))
        self.layer_number = layer_number
        self.output_layernorm = output_layernorm
        self.layer_type = layer_type
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm
        )
        assert (
            self_attn_mask_type in AttnMaskTypes
        ), f"self_attn_mask_type {self_attn_mask_type} not supported"
        assert layer_type in LayerTypes, f"layer_type {layer_type} not supported"

        if not fuse_qkv_params:
            assert (
                not fuse_wgrad_accumulation
            ), "Gradient accumulation fusion requires single QKV parameter."

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
            "apply_query_key_layer_scaling": apply_query_key_layer_scaling,
            "attention_softmax_in_fp32": attention_softmax_in_fp32,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "get_rng_state_tracker": get_rng_state_tracker,
            "sequence_parallel": self.sequence_parallel,
            "params_dtype": params_dtype,
            "return_layernorm_output": apply_residual_connection_post_layernorm,
            "set_parallel_mode": set_parallel_mode,
            "fuse_qkv_params": fuse_qkv_params,
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
            bias=False,
            return_bias=True,
            sequence_parallel=self.sequence_parallel,
            params_dtype=params_dtype,
            return_layernorm_output=apply_residual_connection_post_layernorm,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            set_parallel_mode=set_parallel_mode,
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
        attention_mask: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        enc_dec_attn_mask: Optional[torch.Tensor] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: Optional[bool] = False,
        inference_params: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Transformer Layer: attention block and a feedforward network (MLP)

        Parameters
        ----------
        hidden_states : torch.Tensor
             Input tensor.
        attention_mask : torch.Tensor
             Boolean tensor used to mask out self-attention softmax input.
        encoder_output : torch.Tensor
             Output of the encoder block to be fed into the decoder block if using
             `layer_type="decoder"`.
        enc_dec_attn_mask : torch.Tensor
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
        checkpoint_core_attention: bool, default = `True`
                                  If true, forward activations for core attention are recomputed
                                  during the backward pass in order to save memory that would
                                  otherwise be occupied to store the forward activations until
                                  backprop.
        """

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
