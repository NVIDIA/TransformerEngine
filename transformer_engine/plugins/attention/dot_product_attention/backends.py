# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Attention Backends."""
from contextlib import nullcontext
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
from packaging.version import Version as PkgVersion

import torch
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
)
from transformer_engine.pytorch.utils import nvtx_range_push, nvtx_range_pop

from transformer_engine.pytorch.tensor.quantized_tensor import (
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.constants import (
    TE_DType,
    QKVLayouts,
    dist_group_type,
)

from transformer_engine.pytorch.distributed import get_distributed_world_size
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.attention.inference import InferenceParams

# Import attention utils
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils

from ...import_utils import have_flag_gems

HAVE_FLAG_GEMS = have_flag_gems()

if HAVE_FLAG_GEMS:
    import flag_gems


class AttnFuncFL(torch.autograd.Function):
    """FusedAttention forward and backward implementation"""

    @staticmethod
    def forward(
        ctx,
        is_training,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        page_table_k,
        page_table_v,
        q,
        k,
        v,
        attn_scale,
        dropout_p,
        qkv_layout,
        attn_mask_type,
        window_size,
        rng_gen,
        deterministic,
        layer_number,
    ):
        # pylint: disable=missing-function-docstring
        # add NVTX range
        nvtx_label = "transformer_engine.AttnFuncFL.forward"
        nvtx_range_push(f"{nvtx_label}")

        # input types are inferred from the real data while output types are controlled by fp8_output
        # fp8_output should be set upstream as (DPA.fp8 and DPA.fp8_meta["recipe"].fp8_mha)
        assert isinstance(k, q.__class__) and isinstance(
            v, q.__class__
        ), "q, k, v must be of the same class, e.g. torch.Tensor or Float8Tensor."

        # get nominal data type for out
        # FP16/BF16 attention: torch.float16 or torch.bfloat16
        # FP8 attention:       torch.float16 or torch.bfloat16
        out_nominal_dtype = q.dtype

        max_logit = None

        is_causal = attn_mask_type == 'causal'
        q_permuted = q.permute(1, 2, 0, 3) #[s, b, n_h, h] -> [b, n_h, s, h]
        k_permuted = k.permute(1, 2, 0, 3)
        v_permuted = v.permute(1, 2, 0, 3)
        (out_permuted, m) = flag_gems.scaled_dot_product_attention_forward(
            q_permuted,
            k_permuted,
            v_permuted,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=attn_scale,
            enable_gqa=True,
        )
        out = out_permuted.permute(2, 0, 1, 3) # [b, n_h, s, h] -> [s, b, n_h, h]
        aux_ctx_tensors = [out_permuted, m]
        out_ret = out
        qkvo_tensors = (q_permuted, k_permuted, v_permuted, out_permuted)

        nvtx_range_pop(f"{nvtx_label}")

        # assume fwd and bwd always use the same high precision, i.e. torch.float16 or torch.bfloat16
        # used when some tensors are base tensors and loose the "dtype" attribute
        ctx.nominal_dtype = out_nominal_dtype

        from transformer_engine.pytorch.cpu_offload import (
            CPUOffloadEnabled,
            mark_activation_offload,
        )

        if CPUOffloadEnabled:
            tensor_list = [q, k, v, out]

            mark_activation_offload(*tensor_list)
            mark_activation_offload(*aux_ctx_tensors)

        tensors_to_save, tensor_objects = prepare_for_saving(
            *qkvo_tensors,
            cu_seqlens_q,
            cu_seqlens_kv,
            *aux_ctx_tensors,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects

        ctx.layer_number = layer_number

        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.attn_scale = attn_scale
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal

        ctx.qkv_layout = qkv_layout
        ctx.attn_mask_type = attn_mask_type
        ctx.window_size = window_size
        ctx.deterministic = deterministic

        return out_ret

    @staticmethod
    def backward(ctx, d_out, *_args):
        # pylint: disable=missing-function-docstring

        # d_out is expected to be in FP8 if is_output_fp8=True,
        # but in the case it's not, convert it to FP8 before any operation
        d_out = d_out.contiguous()
        (
            q_permuted,
            k_permuted,
            v_permuted,
            out_permuted,
            cu_seqlens_q,
            cu_seqlens_kv,
            *other_tensors,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

        aux_ctx_tensors = other_tensors

        if not aux_ctx_tensors[0].is_contiguous():
            aux_ctx_tensors[0] = aux_ctx_tensors[0].contiguous()
        if not aux_ctx_tensors[1].is_contiguous():
            aux_ctx_tensors[1] = aux_ctx_tensors[1].contiguous()
        out_permuted, m = aux_ctx_tensors
        rest = [None]

        with torch.cuda.nvtx.range("AttnFuncFL.backward"):
            # get nominal data type of dq, dk, dv
            # FP16/BF16 attention: torch.float16 or torch.bfloat16
            # FP8 attention:       torch.float16 or torch.bfloat16
            dqkv_nominal_dtype = ctx.nominal_dtype

            dqkv_te_dtype = TE_DType[d_out.dtype]

            q_permuted, k_permuted, v_permuted, m = map(lambda x: x.contiguous() if not x.is_contiguous() else x, (q_permuted, k_permuted, v_permuted, m))
            d_out_permuted = d_out.permute(1, 2, 0, 3).contiguous() # [s, b, n_h, h] -> [b, n_h, s, h]
            dq_permuted, dk_permuted, dv_permuted = flag_gems.scaled_dot_product_attention_backward(
                d_out_permuted,
                q_permuted,
                k_permuted,
                v_permuted,
                out_permuted,
                m,
                attn_mask=None,
                dropout_p=ctx.dropout_p,
                is_causal=ctx.is_causal,
                scale=ctx.attn_scale,
                enable_gqa=True,
            )
            dq = dq_permuted.permute(2, 0, 1, 3)
            dk = dk_permuted.permute(2, 0, 1, 3)
            dv = dv_permuted.permute(2, 0, 1, 3)
            rest = None

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FlashAttentionFL(torch.nn.Module):
    """Dot product attention
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_type = attention_type
        self.use_FAv2_bwd = os.getenv(
            "NVTE_FUSED_ATTN_USE_FAv2_BWD", "0"
        ) == "1" and get_device_compute_capability() == (9, 0)
        self.layer_number = 1 if layer_number is None else layer_number
        self.deterministic = deterministic

        def remove_extra_states_check(self, incompatible_keys):  # pylint: disable=unused-argument
            """
            Temporarily remove fused_attention._extra_state as a missing key
            or an unexpected key when loading Transformer Engine checkpoints.
            Please store FP8 metadata as DotProductAttention's _extra_state,
            rather than FusedAttention's _extra_state. This hook will be
            phased out in Transformer Engine 2.0.
            """
            for key in incompatible_keys.missing_keys:
                if "fused_attention._extra_state" in key:
                    incompatible_keys.missing_keys.remove(key)
            for key in incompatible_keys.unexpected_keys:
                if "fused_attention._extra_state" in key:
                    incompatible_keys.unexpected_keys.remove(key)
                    warnings.warn(
                        "fused_attention._extra_state is not loaded from checkpoint. Please map "
                        "FusedAttention's _extra_state to DotProductAttention's _extra_state."
                    )

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    @no_torch_dynamo()
    def forward(
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
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers=None,
        inference_params: Optional[InferenceParams] = None,
        flash_attention_backend: Optional[PkgVersion] = PkgVersion("0"),
        fp8_output: bool = False,
    ) -> torch.Tensor:
        assert HAVE_FLAG_GEMS, "FlagGems is not installed"
        assert window_size == (-1, 0), "Triton-Based FlashAttention do not support sliding windows now"
        assert not fp8, "Triton-Based FlashAttention do not support fp8 now"
        assert attn_mask_type == "causal", "Triton-Based FlashAttention do not support padding mask now"

        assert all(
            x.dtype in [torch.float16, torch.bfloat16] or isinstance(x, Float8Tensor)
            for x in [query_layer, key_layer, value_layer]
        ), "FLAttention only supports FP16 and BF16 data types, or Float8Tensors."
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
        ), "FLAttention only supports CUDA tensors."
        assert (
            qkv_layout in QKVLayouts
        ), f"FLAttention does not support qkv_layout = {qkv_layout}!"

        cp_size = 1
        if isinstance(cp_group, dist_group_type):
            cp_size = get_distributed_world_size(cp_group)
        elif isinstance(cp_group, list):
            for group in cp_group:
                cp_size *= get_distributed_world_size(group)
        context_parallel = cp_size > 1
        assert not context_parallel, "FLAttention do not support context parallel now"

        # get q_format and kv_format for training and inference
        qkv_format, q_format, kv_format = dpa_utils.get_qkv_format(qkv_layout, inference_params)

        # cuDNN can work with 0-length sequences in the batch for both bshd/sbhd and thd formats
        # however, for bshd/sbhd, q/k/v tensors need to have the same batch size as indicated by
        # cu_seqlens, whereas thd does not have this requirement
        # e.g. if q_format = bshd, and q.shape = [3, 1, 16, 64], we should have k.shape[0] =
        # v.shape[0] = q.shape[0], and cu_seqlens_q.shape = cu_seqlens_kv.shape = [4]
        if q_format in ["bshd", "sbhd"] or kv_format in ["bshd", "sbhd"]:
            batch_size = query_layer.shape[0] if q_format == "bshd" else query_layer.shape[1]
            cu_seqlens_q = cu_seqlens_q[: batch_size + 1]
            cu_seqlens_kv = cu_seqlens_kv[: batch_size + 1]

        page_table = None
        if inference_params is None:
            if qkv_format in ["sbhd", "bshd"]:
                if qkv_format == "sbhd":
                    batch_size = query_layer.shape[1]
                    max_seqlen_q = query_layer.shape[0]
                    max_seqlen_kv = key_layer.shape[0]
                if qkv_format == "bshd":
                    batch_size = query_layer.shape[0]
                    max_seqlen_q = query_layer.shape[1]
                    max_seqlen_kv = key_layer.shape[1]
                max_seqlen_q *= cp_size
                max_seqlen_kv *= cp_size
                if "padding" in attn_mask_type:
                    assert (
                        not context_parallel
                    ), "Padding mask not supported with context parallelism!"
                    if cu_seqlens_q is None or cu_seqlens_kv is None:
                        if attention_mask is None:
                            raise RuntimeError(
                                "Please provide attention_mask or cu_seqlens for padding!"
                            )
                        if self.attention_type == "self":
                            cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask)
                            cu_seqlens_kv = cu_seqlens_q
                        else:
                            cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask[0])
                            cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask[1])
                else:
                    if cu_seqlens_q is None:
                        cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_q,
                            query_layer.device,
                        )
                    if cu_seqlens_kv is None:
                        cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_kv,
                            key_layer.device,
                        )
            if qkv_format == "thd":
                assert (
                    max_seqlen_q is not None
                    and max_seqlen_kv is not None
                    and cu_seqlens_q is not None
                    and cu_seqlens_kv is not None
                ), "max_seqlen_q/kv and cu_seqlens_q/kv can not be None when qkv_format is thd!"
        elif inference_params.is_paged:
            page_table = inference_params.cache_manager.page_table

        with self.attention_dropout_ctx():
            _attn_impl = AttnFuncFL
            output = _attn_impl.apply(
                self.training,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                page_table,
                page_table,
                query_layer,
                key_layer,
                value_layer,
                self.softmax_scale,
                self.attention_dropout if self.training else 0.0,
                qkv_layout,
                attn_mask_type,
                window_size,
                None,  # rng_gen
                self.deterministic,
                self.layer_number,
            )

        # ...hd -> ...(hd)
        return output.view(*output.shape[:-2], -1)
