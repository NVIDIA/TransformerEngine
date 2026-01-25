# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformer_engine.plugin.core.ops import FlashAttentionBase


class FlashAttentionTorch(FlashAttentionBase):
    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = None,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            softmax_scale=softmax_scale,
            attention_dropout=attention_dropout,
            attention_dropout_ctx=attention_dropout_ctx,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=deterministic,
        )

    @property
    def backend_name(self) -> str:
        return "torch_sdpa"

    def _convert_layout_to_bhsd(
        self,
        tensor: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        """Convert tensor from various layouts to [batch, heads, seq, dim] format."""
        layout = layout.lower()

        # Handle combined layouts like "sbhd_sbhd_sbhd" - extract the first part
        if "_" in layout:
            layout = layout.split("_")[0]

        if layout in ("sbhd", "sbh3d", "sb3hd"):
            return tensor.permute(1, 2, 0, 3)
        elif layout in ("bshd", "bsh3d", "bs3hd"):
            return tensor.permute(0, 2, 1, 3)
        elif layout in ("bhsd",):
            return tensor
        elif layout in ("thd",):
            # thd is packed format, should not reach here for 4D tensors
            raise ValueError(f"thd layout requires 3D tensor, got {tensor.dim()}D")
        else:
            raise ValueError(f"Unsupported qkv_layout: {layout}")

    def _convert_bhsd_to_layout(
        self,
        tensor: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        """Convert tensor from [batch, heads, seq, dim] back to original layout."""
        layout = layout.lower()

        # Handle combined layouts like "sbhd_sbhd_sbhd" - extract the first part
        if "_" in layout:
            layout = layout.split("_")[0]

        if layout in ("sbhd", "sbh3d", "sb3hd"):
            return tensor.permute(2, 0, 1, 3)
        elif layout in ("bshd", "bsh3d", "bs3hd"):
            return tensor.permute(0, 2, 1, 3)
        elif layout in ("bhsd",):
            return tensor
        elif layout in ("thd",):
            raise ValueError(f"thd layout requires 3D tensor, got {tensor.dim()}D")
        else:
            raise ValueError(f"Unsupported qkv_layout: {layout}")

    def _create_sliding_window_mask(
        self,
        seq_len_q: int,
        seq_len_kv: int,
        window_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a sliding window attention mask."""
        left_window, right_window = window_size

        if left_window == -1 and right_window == -1:
            return torch.zeros(seq_len_q, seq_len_kv, dtype=dtype, device=device)

        q_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
        kv_idx = torch.arange(seq_len_kv, device=device).unsqueeze(0)

        mask_bool = torch.zeros(seq_len_q, seq_len_kv, dtype=torch.bool, device=device)

        if left_window >= 0:
            mask_bool = mask_bool | (kv_idx < q_idx - left_window)

        if right_window >= 0:
            mask_bool = mask_bool | (kv_idx > q_idx + right_window)

        mask = torch.zeros(seq_len_q, seq_len_kv, dtype=dtype, device=device)
        mask.masked_fill_(mask_bool, float('-inf'))

        return mask

    def _unpack_tensor(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert packed tensor to padded tensor format."""
        batch_size = cu_seqlens.shape[0] - 1
        device = tensor.device
        original_shape = tensor.shape

        if tensor.dim() == 4:
            if tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            else:
                raise ValueError(
                    f"Unexpected 4D tensor shape {original_shape}. "
                    f"Expected [total_tokens, 1, num_heads, head_dim]"
                )

        if tensor.dim() != 3:
            raise ValueError(
                f"Expected tensor to be 3D or 4D after processing, got shape {original_shape}"
            )

        total_tokens, num_heads, head_dim = tensor.shape

        expected_total = cu_seqlens[-1].item()
        if total_tokens != expected_total:
            raise ValueError(
                f"Tensor has {total_tokens} tokens but cu_seqlens indicates {expected_total} tokens"
            )

        padded_tensor = torch.zeros(
            batch_size, num_heads, max_seqlen, head_dim,
            dtype=tensor.dtype, device=device
        )

        padding_mask = torch.ones(batch_size, max_seqlen, dtype=torch.bool, device=device)

        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start

            seq_data = tensor[start:end].permute(1, 0, 2)
            padded_tensor[i, :, :seq_len, :] = seq_data
            padding_mask[i, :seq_len] = False

        return padded_tensor, padding_mask

    def _pack_tensor(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Convert padded tensor back to packed tensor format."""
        batch_size = tensor.shape[0]
        num_heads = tensor.shape[1]
        head_dim = tensor.shape[3]
        total_tokens = cu_seqlens[-1].item()
        device = tensor.device

        packed_tensor = torch.zeros(
            total_tokens, num_heads, head_dim,
            dtype=tensor.dtype, device=device
        )

        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start

            seq_data = tensor[i, :, :seq_len, :].permute(1, 0, 2)
            packed_tensor[start:end, :, :] = seq_data

        return packed_tensor

    def _forward_impl(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cp_group: Optional[Any] = None,
        cp_global_ranks: Optional[List[int]] = None,
        cp_stream: Optional[torch.cuda.Stream] = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers: Optional[Any] = None,
        inference_params: Optional[Any] = None,
        flash_attention_backend: Optional[Any] = None,
        fp8_output: bool = False,
    ) -> torch.Tensor:
        """Flash Attention implementation using PyTorch's scaled_dot_product_attention."""
        if fp8:
            raise NotImplementedError("FP8 is not supported in PyTorch SDPA backend")
        if cp_group is not None:
            raise NotImplementedError("Context parallelism is not supported in PyTorch SDPA backend")
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes are not supported in PyTorch SDPA backend")

        query_original_shape = query_layer.shape

        # Check if input is in standard 4D format - same as flagos backend
        # If tensor is 4D, treat it as standard format and just do layout conversion
        # Only use unpack logic for true packed format (3D tensors with thd layout)
        is_standard_4d = query_layer.dim() == 4

        if is_standard_4d:
            # Standard 4D tensor format - just convert layout like flagos does
            query = self._convert_layout_to_bhsd(query_layer, qkv_layout)
            key = self._convert_layout_to_bhsd(key_layer, qkv_layout)
            value = self._convert_layout_to_bhsd(value_layer, qkv_layout)
            use_packed_format = False
            padding_mask_q = None
            padding_mask_kv = None
        else:
            # True packed format (thd layout, 3D tensor) - use unpack logic
            use_packed_format = cu_seqlens_q is not None or cu_seqlens_kv is not None
            padding_mask_q = None
            padding_mask_kv = None

            if use_packed_format:
                if cu_seqlens_q is not None:
                    query, padding_mask_q = self._unpack_tensor(query_layer, cu_seqlens_q, max_seqlen_q)
                else:
                    query = self._convert_layout_to_bhsd(query_layer, qkv_layout)

                if cu_seqlens_kv is not None:
                    key, padding_mask_kv = self._unpack_tensor(key_layer, cu_seqlens_kv, max_seqlen_kv)
                    value, _ = self._unpack_tensor(value_layer, cu_seqlens_kv, max_seqlen_kv)
                else:
                    key = self._convert_layout_to_bhsd(key_layer, qkv_layout)
                    value = self._convert_layout_to_bhsd(value_layer, qkv_layout)
            else:
                query = self._convert_layout_to_bhsd(query_layer, qkv_layout)
                key = self._convert_layout_to_bhsd(key_layer, qkv_layout)
                value = self._convert_layout_to_bhsd(value_layer, qkv_layout)

        batch_size, num_heads_q, seq_len_q, head_dim = query.shape
        num_heads_kv = key.shape[1]
        seq_len_kv = key.shape[2]

        if num_heads_q != num_heads_kv:
            num_groups = num_heads_q // num_heads_kv
            if num_heads_q % num_heads_kv != 0:
                raise ValueError(
                    f"num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
                )
            key = key.repeat_interleave(num_groups, dim=1)
            value = value.repeat_interleave(num_groups, dim=1)

        attn_mask = None
        is_causal = False

        if use_packed_format and padding_mask_kv is not None:
            attn_mask = torch.zeros(
                batch_size, seq_len_q, seq_len_kv,
                dtype=query.dtype, device=query.device
            )
            padding_broadcast = padding_mask_kv.unsqueeze(1)
            attn_mask.masked_fill_(padding_broadcast, float('-inf'))

        if attn_mask_type == "causal":
            is_causal = True
            attn_mask = None
            # if window_size is None and not use_packed_format:
            #     is_causal = True
            # else:
            #     causal_mask = torch.zeros(
            #         seq_len_q, seq_len_kv,
            #         dtype=query.dtype, device=query.device
            #     )
            #     causal_mask.masked_fill_(
            #         torch.triu(torch.ones(seq_len_q, seq_len_kv, device=query.device, dtype=torch.bool), diagonal=1),
            #         float('-inf')
            #     )

            #     if attn_mask is not None:
            #         if attn_mask.dim() == 2:
            #             attn_mask = attn_mask + causal_mask
            #         else:
            #             attn_mask = attn_mask + causal_mask.unsqueeze(0)
            #     else:
            #         attn_mask = causal_mask

        if window_size is not None and not is_causal:
            window_mask = self._create_sliding_window_mask(
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                window_size=window_size,
                device=query.device,
                dtype=query.dtype,
            )

            if attn_mask is not None:
                attn_mask = attn_mask + window_mask.unsqueeze(0)
            else:
                attn_mask = window_mask

        if attention_mask is not None and attn_mask_type != "causal":
            if isinstance(attention_mask, tuple):
                explicit_mask = attention_mask[0]
            else:
                explicit_mask = attention_mask

            if explicit_mask.dtype == torch.bool:
                float_mask = torch.zeros_like(explicit_mask, dtype=query.dtype)
                float_mask.masked_fill_(~explicit_mask, float('-inf'))
                explicit_mask = float_mask

            if explicit_mask.dim() == 2:
                explicit_mask = explicit_mask.unsqueeze(0).unsqueeze(0)
            elif explicit_mask.dim() == 3:
                explicit_mask = explicit_mask.unsqueeze(1)

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)
                attn_mask = attn_mask + explicit_mask
            else:
                attn_mask = explicit_mask
        elif attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        with self.attention_dropout_ctx():
            dropout_p = self.attention_dropout if self.training else 0.0

            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=self.softmax_scale,
            )

        if use_packed_format and padding_mask_q is not None:
            mask_expanded = padding_mask_q.unsqueeze(1).unsqueeze(3)
            output = output.masked_fill(mask_expanded, 0.0)

        if use_packed_format and cu_seqlens_q is not None:
            output = self._pack_tensor(output, cu_seqlens_q)

            if len(query_original_shape) == 4:
                total_tokens = output.shape[0]
                hidden_size = output.shape[1] * output.shape[2]
                output = output.contiguous().view(total_tokens, 1, hidden_size)
        else:
            output = self._convert_bhsd_to_layout(output, qkv_layout)
            # Flatten the last two dimensions (heads, dim) -> (heads * dim)
            # to match the output format of other backends
            output = output.contiguous().view(*output.shape[:-2], -1)

        return output
