# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch implementation of cuDNN-backed scaled dot-product attention.

All cuDNN graph construction and execution in this module goes through the
``nvidia-cudnn-frontend`` Python API.  TE common retains its C++ implementation
for other framework frontends, but PyTorch does not call it.
"""

from __future__ import annotations

from enum import IntEnum
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
# The extension is used only to reserve PyTorch's graph-safe Philox state;
# graph construction, backend selection, and execution do not call TE common.
import transformer_engine_torch as tex

from transformer_engine.pytorch.constants import (
    DType,
    FP8BwdTensorIdx,
    FP8FwdTensorIdx,
    TE_DType_To_Torch,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import (
    Float8TensorStorage,
)
from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import (
    MXFP8TensorStorage,
)

from ._cudnn_graph import (
    GraphEntry,
    finalize_graph,
    get_graph_entry,
    import_cudnn_frontend,
    make_graph,
    put_graph_entry,
    torch_to_cudnn_dtype,
)

__all__ = [
    "FusedAttnBackend",
    "fused_attn_fwd",
    "fused_attn_bwd",
    "META_QKV",
    "META_DQKV",
    "META_O",
    "META_DO",
    "META_S",
    "META_DP",
]


class FusedAttnBackend(IntEnum):
    """PyTorch cuDNN attention implementation families.

    The numeric values intentionally preserve the historical TE-common ABI so
    cached/backend-selection state and external Python callers remain source
    compatible after removal of the pybind enum.
    """

    No_Backend = -1
    F16_arbitrary_seqlen = 1
    FP8 = 2

    @classmethod
    def cast(cls, backend: Union["FusedAttnBackend", int, Any]) -> "FusedAttnBackend":
        if isinstance(backend, cls):
            return backend
        return cls(int(backend))


META_QKV = FP8FwdTensorIdx.GEMM1_OUTPUT
META_DQKV = FP8BwdTensorIdx.GRAD_OUTPUT1
META_O = FP8FwdTensorIdx.GEMM2_INPUT
META_DO = FP8BwdTensorIdx.GRAD_INPUT2
META_S = FP8FwdTensorIdx.GEMM3_OUTPUT
META_DP = FP8BwdTensorIdx.GRAD_INPUT3

_F16_RNG_ELTS_PER_THREAD = 16
_FP8_THREADS_PER_CTA = 128


def _is_float8_tensor(tensor: Any) -> bool:
    return isinstance(tensor, Float8TensorStorage)


def _is_mxfp8_tensor(tensor: Any) -> bool:
    return isinstance(tensor, MXFP8TensorStorage)


def _quantized_data(tensor: Any, *, columnwise: bool = False) -> torch.Tensor:
    if _is_float8_tensor(tensor):
        data = tensor._transpose if columnwise else tensor._data
    elif _is_mxfp8_tensor(tensor):
        data = tensor._columnwise_data if columnwise else tensor._rowwise_data
    else:
        data = tensor
    if data is None:
        orientation = "columnwise" if columnwise else "rowwise"
        raise ValueError(f"Attention input has no {orientation} data buffer.")
    return data


def _quantized_scale_inv(tensor: Any, *, columnwise: bool = False) -> torch.Tensor:
    if _is_float8_tensor(tensor):
        return tensor._scale_inv
    if _is_mxfp8_tensor(tensor):
        scale = (
            tensor._columnwise_scale_inv if columnwise else tensor._rowwise_scale_inv
        )
        if scale is None:
            orientation = "columnwise" if columnwise else "rowwise"
            raise ValueError(
                f"MXFP8 attention input has no {orientation} scale-inverse buffer."
            )
        return scale
    raise TypeError(f"Expected an FP8 attention tensor, got {type(tensor).__name__}.")


def _fp8_cudnn_dtype(tensor: Any):
    return torch_to_cudnn_dtype(TE_DType_To_Torch[DType.cast(tensor._fp8_dtype)])


def _scalar_graph_tensor(graph, cudnn, name: str):
    return graph.tensor(
        name=name,
        dim=(1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.FLOAT,
    )


def _constant_graph_tensor(graph, cudnn, name: str):
    """Create a scalar graph input that callers bind to a constant value."""
    return _scalar_graph_tensor(graph, cudnn, name)


def _format_stride(batch: int, heads: int, seqlen: int, dim: int, tensor_format: str):
    if tensor_format in ("bshd", "thd"):
        return (seqlen * heads * dim, dim, heads * dim, 1)
    if tensor_format == "sbhd":
        return (heads * dim, dim, batch * heads * dim, 1)
    if tensor_format == "bhsd":
        return (heads * seqlen * dim, seqlen * dim, dim, 1)
    raise ValueError(f"Unsupported FP8 tensor format {tensor_format!r}.")


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _max_ragged_tokens(num_tokens: int) -> int:
    """Quantize THD token counts to the buckets used by TE's cuDNN path."""

    if num_tokens <= 1024:
        return 1024
    if num_tokens <= 32768:
        return 1 << (num_tokens - 1).bit_length()
    return _round_up(num_tokens, 32768)


def _max_ragged_batch(batch: int) -> int:
    """Quantize THD batch sizes to the buckets used by TE's cuDNN path."""

    if batch <= 32:
        return 32
    if batch <= 512:
        return 1 << (batch - 1).bit_length()
    return _round_up(batch, 512)


def _padded_sequence_lengths(cu_seqlens: torch.Tensor, batch: int) -> torch.Tensor:
    lengths = _sequence_lengths(cu_seqlens)
    if lengths.numel() == batch:
        return lengths
    padding = torch.zeros(
        batch - lengths.numel(), dtype=lengths.dtype, device=lengths.device
    )
    return torch.cat((lengths, padding))


def _element_ragged_offsets(
    cu_seqlens_padded: torch.Tensor,
    batch: int,
    multiplier: int,
) -> torch.Tensor:
    """Convert token offsets to padded int64 element offsets for legacy SDPA graphs."""

    offsets = cu_seqlens_padded.to(dtype=torch.int64)
    if offsets.numel() < batch + 1:
        tail = offsets[-1:].expand(batch + 1 - offsets.numel())
        offsets = torch.cat((offsets, tail))
    return offsets * multiplier


def _mxfp8_padded_sizes(s_q: int, s_kv: int, d_qk: int, d_v: int) -> Dict[str, int]:
    return {
        "s_q_padded": _round_up(s_q, 128),
        "s_kv_padded": _round_up(s_kv, 128),
        "s_q_scale_padded": _round_up((s_q + 31) // 32, 4),
        "s_kv_scale_padded": _round_up((s_kv + 31) // 32, 4),
        "d_qk_padded": _round_up(d_qk, 128),
        "d_v_padded": _round_up(d_v, 128),
        "d_qk_scale_padded": _round_up((d_qk + 31) // 32, 4),
        "d_v_scale_padded": _round_up((d_v + 31) // 32, 4),
    }


def _make_mxfp8_scale_tensor(
    graph,
    cudnn,
    *,
    name: str,
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    tensor_format: str,
):
    return graph.tensor(
        name=name,
        dim=(batch, heads, seqlen, dim),
        stride=_format_stride(batch, heads, seqlen, dim, tensor_format),
        data_type=cudnn.data_type.FP8_E8M0,
    ).set_reordering_type(cudnn.tensor_reordering.F8_128x4)


def _make_float8_output(quantizer, shape, fake_dtype, device):
    data = torch.empty(shape, dtype=torch.uint8, device=device)
    return quantizer.create_tensor_from_data(
        data,
        fake_dtype=fake_dtype,
        internal=bool(getattr(quantizer, "internal", False)),
    )


def _allocate_fp8_kernel_output(quantizer, shape, fake_dtype, device):
    """Allocate the FP8 SDPA output and any hidden amax buffer.

    Delayed scaling asks cuDNN to produce FP8 directly. Current scaling and
    MXFP8 preserve the historical attention contract and ask cuDNN for a
    high-precision output, which the caller may quantize afterwards.
    """
    if isinstance(quantizer, Float8Quantizer):
        return _make_float8_output(quantizer, shape, fake_dtype, device), quantizer.amax
    if isinstance(quantizer, Float8CurrentScalingQuantizer):
        return torch.empty(shape, dtype=fake_dtype, device=device), torch.zeros(
            1, dtype=torch.float32, device=device
        )
    if isinstance(quantizer, MXFP8Quantizer):
        return torch.empty(shape, dtype=fake_dtype, device=device), None
    raise TypeError(
        f"Unsupported FP8 attention output quantizer {type(quantizer).__name__}."
    )


def _format_from_layout_component(component: str) -> str:
    return "".join(char for char in component if char.isalpha())


def _q_kv_formats(qkv_layout: str) -> Tuple[str, str]:
    layout = qkv_layout.removeprefix("paged_kv_")
    components = layout.split("_")
    q_format = _format_from_layout_component(components[0])
    kv_format = (
        _format_from_layout_component(components[-1])
        if len(components) > 1
        else q_format
    )
    return q_format, kv_format


def _is_paged_layout(qkv_layout: str) -> bool:
    return qkv_layout.startswith("paged_kv_")


def _tensor_metadata(tensor: Optional[torch.Tensor]) -> Optional[Tuple[Any, ...]]:
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )


def _logical_bhsd_desc(
    tensor: torch.Tensor,
    tensor_format: str,
    *,
    batch: int,
    max_seqlen: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Describe a physical TE attention tensor as logical cuDNN BHSD."""

    shape = tuple(tensor.shape)
    stride = tuple(tensor.stride())
    if tensor_format == "sbhd":
        return (
            (batch, shape[2], shape[0], shape[3]),
            (stride[1], stride[2], stride[0], stride[3]),
        )
    if tensor_format == "bshd":
        return (
            (batch, shape[2], shape[1], shape[3]),
            (stride[0], stride[2], stride[1], stride[3]),
        )
    if tensor_format == "bhsd":
        return ((batch, shape[1], shape[2], shape[3]), stride)
    if tensor_format == "thd":
        # The ragged offset selects each batch's first token.  The synthetic
        # batch stride is only graph metadata; S/H/D use the real packed view.
        return (
            (batch, shape[1], max_seqlen, shape[2]),
            (max_seqlen * stride[0], stride[1], stride[0], stride[2]),
        )
    raise ValueError(f"Unsupported attention tensor format {tensor_format!r}.")


def _make_bhsd_graph_tensor(
    graph,
    tensor: torch.Tensor,
    tensor_format: str,
    *,
    batch: int,
    max_seqlen: int,
    data_type=None,
    ragged_offset=None,
    ragged_offset_multiplier: int = 1,
    name: str,
):
    dim, stride = _logical_bhsd_desc(
        tensor,
        tensor_format,
        batch=batch,
        max_seqlen=max_seqlen,
    )
    graph_tensor = graph.tensor(
        name=name,
        dim=dim,
        stride=stride,
        data_type=data_type if data_type is not None else tensor.dtype,
        ragged_offset=ragged_offset,
        ragged_offset_multiplier=ragged_offset_multiplier,
    )
    return graph_tensor


def _allocate_output(
    q: torch.Tensor,
    value_head_dim: int,
    fake_dtype: torch.dtype,
    fast_zero_fill: bool,
) -> torch.Tensor:
    shape = (*q.shape[:-1], value_head_dim)
    factory = torch.zeros if fast_zero_fill else torch.empty
    return factory(shape, dtype=fake_dtype, device=q.device)


def _storage_span(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    return 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )


def _allocate_grad_views(
    inputs: Sequence[torch.Tensor],
    *,
    fast_zero_fill: bool,
) -> Tuple[torch.Tensor, ...]:
    """Allocate gradients while preserving packed-QKV storage relationships."""

    groups: Dict[Tuple[str, int, int], List[int]] = {}
    for index, tensor in enumerate(inputs):
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index or 0, storage.data_ptr())
        groups.setdefault(key, []).append(index)

    outputs: List[Optional[torch.Tensor]] = [None] * len(inputs)
    for indices in groups.values():
        if len(indices) == 1:
            inp = inputs[indices[0]]
            out = torch.empty_strided(
                inp.shape, inp.stride(), dtype=inp.dtype, device=inp.device
            )
            if fast_zero_fill:
                out.zero_()
            outputs[indices[0]] = out
            continue

        min_offset = min(inputs[index].storage_offset() for index in indices)
        max_end = max(
            inputs[index].storage_offset() + _storage_span(inputs[index])
            for index in indices
        )
        exemplar = inputs[indices[0]]
        base = torch.empty(
            max_end - min_offset, dtype=exemplar.dtype, device=exemplar.device
        )
        if fast_zero_fill:
            base.zero_()
        for index in indices:
            inp = inputs[index]
            outputs[index] = torch.as_strided(
                base,
                size=inp.shape,
                stride=inp.stride(),
                storage_offset=inp.storage_offset() - min_offset,
            )

    return tuple(output for output in outputs if output is not None)


def _reserve_philox_state(
    device: torch.device,
    rng_gen: Optional[torch.Generator],
    increment: int,
) -> torch.Tensor:
    """Reserve a Philox counter range and return CUDA ``[seed, offset]``."""

    device = torch.device(device)
    helper = getattr(tex, "get_cudnn_attention_rng_state", None)
    if helper is not None:
        with torch.cuda.device(device):
            return helper(rng_gen, increment)
    # Development-tree compatibility before the local extension has been
    # rebuilt. Installed packages always expose the graph-safe helper above.
    if rng_gen is None:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        rng_gen = torch.cuda.default_generators[index]
    seed = rng_gen.initial_seed()
    offset = rng_gen.get_offset()
    rng_gen.set_offset(offset + increment)
    return torch.tensor((seed, offset), dtype=torch.int64, device=device)


def _mask_options(
    cudnn,
    attn_mask_type: str,
    window_size: Tuple[int, int],
    bottom_right_diagonal: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> Dict[str, Any]:
    is_causal = attn_mask_type in ("causal", "padding_causal")
    is_bottom_right = attn_mask_type in (
        "causal_bottom_right",
        "padding_causal_bottom_right",
    )
    is_padding = attn_mask_type in (
        "padding",
        "padding_causal",
        "padding_causal_bottom_right",
    )
    if is_bottom_right and max_seqlen_q == max_seqlen_kv and not is_padding:
        is_causal = True
        is_bottom_right = False
        bottom_right_diagonal = False

    options: Dict[str, Any] = {
        "use_causal_mask": is_causal,
        "use_causal_mask_bottom_right": is_bottom_right,
        "diagonal_alignment": (
            cudnn.diagonal_alignment.BOTTOM_RIGHT
            if bottom_right_diagonal
            else cudnn.diagonal_alignment.TOP_LEFT
        ),
    }
    left, right = window_size
    if left != -1:
        options["diagonal_band_left_bound"] = left + 1
    # ``use_causal_mask`` already imposes a right bound of zero. The Python
    # frontend rejects specifying that same bound through both attributes.
    if right != -1 and not ((is_causal or is_bottom_right) and right == 0):
        options["diagonal_band_right_bound"] = right
    options["is_padding"] = is_padding
    return options


def _sequence_lengths(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _ragged_offset_tensor(
    graph,
    cu_seqlens_padded: torch.Tensor,
    *,
    multiplier: int,
    name: str,
    length: Optional[int] = None,
    data_type=None,
):
    return (
        graph.tensor(
            name=name,
            dim=(cu_seqlens_padded.numel() if length is None else length, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cu_seqlens_padded.dtype if data_type is None else data_type,
            is_pass_by_value=False,
        ),
        multiplier,
    )


def _stats_layout(
    *,
    batch: int,
    heads: int,
    max_seqlen_q: int,
    total_tokens_q: int,
    ragged: bool,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    if ragged:
        physical_shape = (total_tokens_q, heads, 1)
        logical_dim = (batch, heads, max_seqlen_q, 1)
        logical_stride = (heads * max_seqlen_q, 1, heads, 1)
    else:
        physical_shape = logical_dim = (batch, heads, max_seqlen_q, 1)
        logical_stride = (heads * max_seqlen_q, max_seqlen_q, 1, 1)
    return physical_shape, logical_dim, logical_stride


def _f16_fwd_key(**kwargs) -> Tuple[Any, ...]:
    return ("f16_fwd",) + tuple(kwargs.items())


def _build_f16_fwd_graph(
    *,
    is_training: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    stats: Optional[torch.Tensor],
    max_scores: Optional[torch.Tensor],
    attn_bias: Optional[torch.Tensor],
    cu_seqlens_q_padded: torch.Tensor,
    cu_seqlens_kv_padded: torch.Tensor,
    page_table_k: Optional[torch.Tensor],
    page_table_v: Optional[torch.Tensor],
    rng_state: torch.Tensor,
    softmax_offset: Optional[torch.Tensor],
    attn_scale: float,
    dropout: float,
    qkv_layout: str,
    o_format: str,
    attn_bias_type: str,
    attn_mask_type: str,
    softmax_type: str,
    window_size: Tuple[int, int],
    bottom_right_diagonal: bool,
) -> GraphEntry:
    cudnn = import_cudnn_frontend()
    graph = make_graph(
        torch_to_cudnn_dtype(q.dtype), q.device, name="te_fused_attention_fwd"
    )
    q_format, kv_format = _q_kv_formats(qkv_layout)
    batch = cu_seqlens_q.numel() - 1
    is_ragged_q = q_format == "thd"
    is_ragged_kv = kv_format == "thd"
    use_ragged_stats = is_ragged_q and cudnn.backend_version() >= 90600
    use_token_buckets = (
        cudnn.backend_version() >= 90600
        and torch.cuda.get_device_capability(q.device) != (12, 0)
    )
    use_direct_offsets = cudnn.backend_version() >= 92400 and dropout == 0.0
    use_legacy_offsets = (is_ragged_q or is_ragged_kv) and not use_direct_offsets
    graph_batch = (
        _max_ragged_batch(batch) if use_legacy_offsets and use_token_buckets else batch
    )
    if not use_token_buckets:
        use_ragged_stats = False
    graph_seqlen_q = (
        _max_ragged_tokens(q.shape[0])
        if is_ragged_q and use_token_buckets
        else max_seqlen_q
    )
    graph_seqlen_kv = (
        _max_ragged_tokens(k.shape[0])
        if is_ragged_kv and use_token_buckets
        else max_seqlen_kv
    )

    tensors: Dict[str, Any] = {}
    tensors["_legacy_offsets"] = use_legacy_offsets
    tensors["_graph_batch"] = graph_batch
    offset_q = offset_o = offset_k = offset_v = offset_stats = None
    if is_ragged_q:
        offset_q, q_mult = _ragged_offset_tensor(
            graph,
            cu_seqlens_q_padded,
            multiplier=1 if use_legacy_offsets else q.stride(0),
            name="offset_q",
            length=graph_batch + 1,
            data_type=torch.int64 if use_legacy_offsets else None,
        )
        offset_o, o_mult = _ragged_offset_tensor(
            graph,
            cu_seqlens_q_padded,
            multiplier=1 if use_legacy_offsets else output.stride(0),
            name="offset_o",
            length=graph_batch + 1,
            data_type=torch.int64 if use_legacy_offsets else None,
        )
        tensors["offset_q"] = offset_q
        tensors["offset_o"] = offset_o
    else:
        q_mult = o_mult = 1
    if is_ragged_kv:
        offset_k, k_mult = _ragged_offset_tensor(
            graph,
            cu_seqlens_kv_padded,
            multiplier=1 if use_legacy_offsets else k.stride(0),
            name="offset_k",
            length=graph_batch + 1,
            data_type=torch.int64 if use_legacy_offsets else None,
        )
        offset_v, v_mult = _ragged_offset_tensor(
            graph,
            cu_seqlens_kv_padded,
            multiplier=1 if use_legacy_offsets else v.stride(0),
            name="offset_v",
            length=graph_batch + 1,
            data_type=torch.int64 if use_legacy_offsets else None,
        )
        tensors["offset_k"] = offset_k
        tensors["offset_v"] = offset_v
    else:
        k_mult = v_mult = 1

    q_t = _make_bhsd_graph_tensor(
        graph,
        q,
        q_format,
        batch=graph_batch,
        max_seqlen=graph_seqlen_q,
        ragged_offset=offset_q,
        ragged_offset_multiplier=q_mult,
        name="Q",
    )

    if _is_paged_layout(qkv_layout):
        if kv_format == "bshd":
            num_pages_k, page_size_k = k.shape[0], k.shape[1]
            num_pages_v, page_size_v = v.shape[0], v.shape[1]
        elif kv_format == "sbhd":
            page_size_k, num_pages_k = k.shape[0], k.shape[1]
            page_size_v, num_pages_v = v.shape[0], v.shape[1]
        else:
            raise ValueError(f"Paged attention does not support KV format {kv_format}.")
        k_batch, k_seqlen = num_pages_k, page_size_k
        v_batch, v_seqlen = num_pages_v, page_size_v
    else:
        k_batch = v_batch = graph_batch
        k_seqlen = v_seqlen = max_seqlen_kv

    k_t = _make_bhsd_graph_tensor(
        graph,
        k,
        kv_format,
        batch=k_batch,
        max_seqlen=graph_seqlen_kv if is_ragged_kv else k_seqlen,
        ragged_offset=offset_k,
        ragged_offset_multiplier=k_mult,
        name="K",
    )
    v_t = _make_bhsd_graph_tensor(
        graph,
        v,
        kv_format,
        batch=v_batch,
        max_seqlen=graph_seqlen_kv if is_ragged_kv else v_seqlen,
        ragged_offset=offset_v,
        ragged_offset_multiplier=v_mult,
        name="V",
    )
    tensors.update(Q=q_t, K=k_t, V=v_t)

    options = _mask_options(
        cudnn,
        attn_mask_type,
        window_size,
        bottom_right_diagonal,
        max_seqlen_q,
        max_seqlen_kv,
    )
    is_padding = options.pop("is_padding")
    options.update(
        generate_stats=True,
        attn_scale=float(attn_scale),
        use_padding_mask=is_padding,
        use_alibi_mask=attn_bias_type == "alibi",
    )

    if attn_bias_type == "post_scale_bias":
        bias_t = graph.tensor_like(attn_bias, name="Bias")
        tensors["Bias"] = bias_t
        options["bias"] = bias_t

    if is_padding:
        seq_q = _padded_sequence_lengths(cu_seqlens_q, graph_batch)
        seq_kv = _padded_sequence_lengths(cu_seqlens_kv, graph_batch)
        seq_q_t = graph.tensor_like(seq_q, name="seq_len_q")
        seq_kv_t = graph.tensor_like(seq_kv, name="seq_len_kv")
        tensors["seq_len_q"] = seq_q_t
        tensors["seq_len_kv"] = seq_kv_t
        options["seq_len_q"] = seq_q_t
        options["seq_len_kv"] = seq_kv_t

    if page_table_k is not None:
        page_k_t = graph.tensor_like(page_table_k, name="page_table_k")
        page_v_t = graph.tensor_like(page_table_v, name="page_table_v")
        tensors["page_table_k"] = page_k_t
        tensors["page_table_v"] = page_v_t
        options["paged_attention_k_table"] = page_k_t
        options["paged_attention_v_table"] = page_v_t
        options["paged_attention_max_seq_len_kv"] = max_seqlen_kv

    if is_training and dropout != 0.0:
        seed_t = graph.tensor(
            name="dropout_seed",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT64,
        )
        offset_t = graph.tensor(
            name="dropout_offset",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT64,
        )
        tensors["dropout_seed"] = seed_t
        tensors["dropout_offset"] = offset_t
        options["dropout"] = (float(dropout), seed_t, offset_t)

    if softmax_type != "vanilla":
        if softmax_offset is None:
            raise ValueError(f"softmax_type={softmax_type!r} requires softmax_offset.")
        softmax_offset_t = graph.tensor_like(softmax_offset, name="softmax_offset")
        tensors["softmax_offset"] = softmax_offset_t
        options["sink_token"] = softmax_offset_t

    if max_scores is not None:
        _, max_dim, max_stride = _stats_layout(
            batch=graph_batch,
            heads=q_t.get_dim()[1],
            max_seqlen_q=graph_seqlen_q,
            total_tokens_q=q.shape[0] if is_ragged_q else batch * max_seqlen_q,
            ragged=use_ragged_stats,
        )
        if use_ragged_stats:
            offset_stats, stats_mult = _ragged_offset_tensor(
                graph,
                cu_seqlens_q_padded,
                multiplier=1 if use_legacy_offsets else q_t.get_dim()[1],
                name="offset_stats",
                length=graph_batch + 1,
                data_type=torch.int64 if use_legacy_offsets else None,
            )
            tensors["offset_stats"] = offset_stats
        else:
            stats_mult = 1
        max_t = graph.tensor(
            name="Max",
            dim=max_dim,
            stride=max_stride,
            data_type=cudnn.data_type.FLOAT,
            ragged_offset=offset_stats,
            ragged_offset_multiplier=stats_mult,
        ).set_output(True)
        tensors["Max"] = max_t
        options["score_max"] = max_t

    output_t, stats_t = graph.sdpa(name="te_sdpa", q=q_t, k=k_t, v=v_t, **options)
    output_dim, output_stride = _logical_bhsd_desc(
        output,
        o_format,
        batch=graph_batch,
        max_seqlen=graph_seqlen_q,
    )
    output_t.set_output(True).set_dim(output_dim).set_stride(output_stride)
    if is_ragged_q:
        output_t.set_ragged_offset(offset_o).set_ragged_offset_multiplier(o_mult)
    tensors["O"] = output_t

    assert stats is not None
    _, stats_dim, stats_stride = _stats_layout(
        batch=graph_batch,
        heads=output_dim[1],
        max_seqlen_q=graph_seqlen_q,
        total_tokens_q=q.shape[0] if is_ragged_q else batch * max_seqlen_q,
        ragged=use_ragged_stats,
    )
    if use_ragged_stats and offset_stats is None:
        offset_stats, stats_mult = _ragged_offset_tensor(
            graph,
            cu_seqlens_q_padded,
            multiplier=1 if use_legacy_offsets else output_dim[1],
            name="offset_stats",
            length=graph_batch + 1,
            data_type=torch.int64 if use_legacy_offsets else None,
        )
        tensors["offset_stats"] = offset_stats
    stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(
        stats_dim
    ).set_stride(stats_stride)
    if use_ragged_stats:
        stats_t.set_ragged_offset(offset_stats).set_ragged_offset_multiplier(stats_mult)
    tensors["Stats"] = stats_t

    return GraphEntry(
        graph=graph, tensors=tensors, workspace_size=finalize_graph(graph)
    )


def _f16_forward(
    is_training: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fake_dtype: torch.dtype,
    attn_bias: Optional[torch.Tensor],
    cu_seqlens_q_padded: Optional[torch.Tensor],
    cu_seqlens_kv_padded: Optional[torch.Tensor],
    page_table_k: Optional[torch.Tensor],
    page_table_v: Optional[torch.Tensor],
    attn_scale: float,
    dropout: float,
    fast_zero_fill: bool,
    qkv_layout: str,
    o_format: str,
    attn_bias_type: str,
    attn_mask_type: str,
    softmax_type: str,
    window_size: Tuple[int, int],
    bottom_right_diagonal: bool,
    rng_gen: Optional[torch.Generator],
    softmax_offset: Optional[torch.Tensor],
    return_max_logit: bool,
) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
    q_format, kv_format = _q_kv_formats(qkv_layout)
    batch = cu_seqlens_q.numel() - 1
    heads = q.shape[1] if q_format == "bhsd" else q.shape[-2]
    total_tokens_q = q.shape[0] if q_format == "thd" else batch * max_seqlen_q
    cudnn = import_cudnn_frontend()
    use_token_buckets = (
        cudnn.backend_version() >= 90600
        and torch.cuda.get_device_capability(q.device) != (12, 0)
    )
    use_direct_offsets = cudnn.backend_version() >= 92400 and dropout == 0.0
    use_legacy_offsets = (
        q_format == "thd" or kv_format == "thd"
    ) and not use_direct_offsets
    graph_batch = (
        _max_ragged_batch(batch) if use_legacy_offsets and use_token_buckets else batch
    )
    ragged_stats = (
        q_format == "thd"
        and cudnn.backend_version() >= 90600
        and torch.cuda.get_device_capability(q.device) != (12, 0)
    )
    output_shape = (
        (q.shape[0], heads, v.shape[-1])
        if o_format == "thd"
        else _fp8_output_shape(batch, max_seqlen_q, heads, v.shape[-1], o_format)
    )
    output_factory = torch.zeros if fast_zero_fill else torch.empty
    output = output_factory(output_shape, dtype=fake_dtype, device=q.device)
    stats_shape, _, _ = _stats_layout(
        batch=batch,
        heads=heads,
        max_seqlen_q=max_seqlen_q,
        total_tokens_q=total_tokens_q,
        ragged=ragged_stats,
    )
    stats = torch.empty(stats_shape, dtype=torch.float32, device=q.device)
    max_scores = (
        torch.empty(stats_shape, dtype=torch.float32, device=q.device)
        if return_max_logit
        else None
    )
    rng_state = _reserve_philox_state(q.device, rng_gen, _F16_RNG_ELTS_PER_THREAD)
    cu_seqlens_q_padded = (
        cu_seqlens_q if cu_seqlens_q_padded is None else cu_seqlens_q_padded
    )
    cu_seqlens_kv_padded = (
        cu_seqlens_kv if cu_seqlens_kv_padded is None else cu_seqlens_kv_padded
    )

    key = _f16_fwd_key(
        is_training=is_training,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        graph_batch=graph_batch,
        q=_tensor_metadata(q),
        k=_tensor_metadata(k),
        v=_tensor_metadata(v),
        output=_tensor_metadata(output),
        stats=_tensor_metadata(stats),
        bias=_tensor_metadata(attn_bias),
        qkv_layout=qkv_layout,
        o_format=o_format,
        attn_scale=float(attn_scale),
        dropout=float(dropout),
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        softmax_type=softmax_type,
        window_size=tuple(window_size),
        bottom_right_diagonal=bottom_right_diagonal,
        return_max_logit=return_max_logit,
        paged=page_table_k is not None,
    )
    entry = get_graph_entry(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cuDNN attention graph must be built before CUDA graph capture."
            )
        entry = _build_f16_fwd_graph(
            is_training=is_training,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            q=q,
            k=k,
            v=v,
            output=output,
            stats=stats,
            max_scores=max_scores,
            attn_bias=attn_bias,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            page_table_k=page_table_k,
            page_table_v=page_table_v,
            rng_state=rng_state,
            softmax_offset=softmax_offset,
            attn_scale=attn_scale,
            dropout=dropout,
            qkv_layout=qkv_layout,
            o_format=o_format,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            softmax_type=softmax_type,
            window_size=window_size,
            bottom_right_diagonal=bottom_right_diagonal,
        )
        put_graph_entry(key, entry)

    tensors = entry.tensors
    variant_pack: Dict[Any, Any] = {
        tensors["Q"]: q,
        tensors["K"]: k,
        tensors["V"]: v,
        tensors["O"]: output,
    }
    variant_pack[tensors["Stats"]] = stats
    if attn_bias_type == "post_scale_bias":
        variant_pack[tensors["Bias"]] = attn_bias
    legacy_offsets = tensors["_legacy_offsets"]
    graph_batch = tensors["_graph_batch"]
    if "seq_len_q" in tensors:
        variant_pack[tensors["seq_len_q"]] = _padded_sequence_lengths(
            cu_seqlens_q, graph_batch
        )
        variant_pack[tensors["seq_len_kv"]] = _padded_sequence_lengths(
            cu_seqlens_kv, graph_batch
        )
    if "offset_q" in tensors:
        if legacy_offsets:
            variant_pack[tensors["offset_q"]] = _element_ragged_offsets(
                cu_seqlens_q_padded, graph_batch, q.stride(0)
            )
            variant_pack[tensors["offset_o"]] = _element_ragged_offsets(
                cu_seqlens_q_padded, graph_batch, output.stride(0)
            )
        else:
            variant_pack[tensors["offset_q"]] = cu_seqlens_q_padded
            variant_pack[tensors["offset_o"]] = cu_seqlens_q_padded
    if "offset_k" in tensors:
        if legacy_offsets:
            variant_pack[tensors["offset_k"]] = _element_ragged_offsets(
                cu_seqlens_kv_padded, graph_batch, k.stride(0)
            )
            variant_pack[tensors["offset_v"]] = _element_ragged_offsets(
                cu_seqlens_kv_padded, graph_batch, v.stride(0)
            )
        else:
            variant_pack[tensors["offset_k"]] = cu_seqlens_kv_padded
            variant_pack[tensors["offset_v"]] = cu_seqlens_kv_padded
    if "offset_stats" in tensors:
        variant_pack[tensors["offset_stats"]] = (
            _element_ragged_offsets(cu_seqlens_q_padded, graph_batch, heads)
            if legacy_offsets
            else cu_seqlens_q_padded
        )
    if "page_table_k" in tensors:
        variant_pack[tensors["page_table_k"]] = page_table_k
        variant_pack[tensors["page_table_v"]] = page_table_v
    if "dropout_seed" in tensors:
        variant_pack[tensors["dropout_seed"]] = rng_state[:1]
        variant_pack[tensors["dropout_offset"]] = rng_state[1:]
    if "softmax_offset" in tensors:
        variant_pack[tensors["softmax_offset"]] = softmax_offset
    if "Max" in tensors:
        variant_pack[tensors["Max"]] = max_scores
    entry.execute(variant_pack, q.device)

    aux: List[torch.Tensor] = [stats, rng_state]
    if is_training:
        if attn_bias_type not in ("no_bias", "alibi"):
            aux.append(attn_bias)
        if softmax_type != "vanilla":
            aux.append(softmax_offset)

    max_logit = None
    if return_max_logit:
        if q_format == "thd" and max_scores.ndim == 4:
            seqlens_q = _sequence_lengths(cu_seqlens_q).to(device=max_scores.device)
            sq_idx = torch.arange(max_scores.shape[2], device=max_scores.device).view(
                1, 1, -1, 1
            )
            valid = sq_idx < seqlens_q.view(-1, 1, 1, 1)
            max_scores_for_reduce = max_scores.masked_fill(~valid, float("-inf"))
        else:
            max_scores_for_reduce = max_scores
        reduce_dims = (0, 2) if max_scores_for_reduce.ndim == 3 else (0, 2, 3)
        max_logit = torch.amax(max_scores_for_reduce, dim=reduce_dims).to(output.dtype)
    return output, aux, max_logit


def _fp8_output_shape(
    batch: int, seqlen: int, heads: int, dim: int, tensor_format: str
):
    if tensor_format == "bshd":
        return (batch, seqlen, heads, dim)
    if tensor_format == "sbhd":
        return (seqlen, batch, heads, dim)
    if tensor_format == "bhsd":
        return (batch, heads, seqlen, dim)
    raise ValueError(f"FP8 attention does not support output format {tensor_format!r}.")


def _allocate_attention_grad_data(
    *,
    batch: int,
    heads: int,
    kv_heads: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    head_dim_qk: int,
    head_dim_v: int,
    dqkv_layout: str,
    dtype: torch.dtype,
    device: torch.device,
    zero: bool,
):
    """Allocate dQ/dK/dV buffers with the requested packed storage layout."""

    layout = dqkv_layout.removeprefix("paged_kv_")
    components = layout.split("_")
    q_format, kv_format = _q_kv_formats(layout)
    q_shape = _fp8_output_shape(batch, max_seqlen_q, heads, head_dim_qk, q_format)
    k_shape = _fp8_output_shape(batch, max_seqlen_kv, kv_heads, head_dim_qk, kv_format)
    v_shape = _fp8_output_shape(batch, max_seqlen_kv, kv_heads, head_dim_v, kv_format)
    factory = torch.zeros if zero else torch.empty

    if len(components) == 1:
        if (
            head_dim_qk != head_dim_v
            or heads != kv_heads
            or max_seqlen_q != max_seqlen_kv
        ):
            raise ValueError(
                f"Packed QKV gradient layout {layout!r} requires matching Q/K/V."
            )
        packed_dim = components[0].index("3")
        packed_shape = list(q_shape)
        packed_shape.insert(packed_dim, 3)
        packed = factory(packed_shape, dtype=dtype, device=device)
        return tuple(packed.select(packed_dim, index) for index in range(3))
    if len(components) == 2:
        if head_dim_qk != head_dim_v:
            raise ValueError(
                f"Packed KV gradient layout {layout!r} requires dQK == dV."
            )
        q_out = factory(q_shape, dtype=dtype, device=device)
        packed_dim = components[1].index("2")
        packed_shape = list(k_shape)
        packed_shape.insert(packed_dim, 2)
        packed = factory(packed_shape, dtype=dtype, device=device)
        return q_out, packed.select(packed_dim, 0), packed.select(packed_dim, 1)
    return tuple(
        factory(shape, dtype=dtype, device=device)
        for shape in (q_shape, k_shape, v_shape)
    )


def _wrap_float8_grad_outputs(quantizer, tensors, fake_dtype):
    return tuple(
        quantizer.create_tensor_from_data(
            tensor,
            fake_dtype=fake_dtype,
            internal=bool(getattr(quantizer, "internal", False)),
        )
        for tensor in tensors
    )


def _build_fp8_fwd_graph(
    *,
    max_seqlen_q,
    max_seqlen_kv,
    q,
    k,
    v,
    output,
    stats,
    amax_s,
    amax_o,
    s_quantizer,
    o_quantizer,
    qkv_layout,
    o_format,
    qkv_scale_inv_format,
    attn_scale,
    dropout,
    attn_mask_type,
    softmax_type,
    window_size,
    bottom_right_diagonal,
    softmax_offset,
    cu_seqlens_q,
    cu_seqlens_kv,
):
    cudnn = import_cudnn_frontend()
    q_data = _quantized_data(q)
    k_data = _quantized_data(k)
    v_data = _quantized_data(v, columnwise=_is_mxfp8_tensor(v))
    q_format, kv_format = _q_kv_formats(qkv_layout)
    batch = cu_seqlens_q.numel() - 1
    heads = q.shape[-2] if q_format != "bhsd" else q.shape[1]
    kv_heads = k.shape[-2] if kv_format != "bhsd" else k.shape[1]
    d_qk = q.shape[-1]
    d_v = v.shape[-1]
    graph = make_graph(_fp8_cudnn_dtype(q), q.device, name="te_fp8_sdpa_fwd")
    tensors: Dict[str, Any] = {}

    q_t = _make_bhsd_graph_tensor(
        graph,
        q_data,
        q_format,
        batch=batch,
        max_seqlen=max_seqlen_q,
        data_type=_fp8_cudnn_dtype(q),
        name="Q",
    )
    k_t = _make_bhsd_graph_tensor(
        graph,
        k_data,
        kv_format,
        batch=batch,
        max_seqlen=max_seqlen_kv,
        data_type=_fp8_cudnn_dtype(k),
        name="K",
    )
    v_t = _make_bhsd_graph_tensor(
        graph,
        v_data,
        kv_format,
        batch=batch,
        max_seqlen=max_seqlen_kv,
        data_type=_fp8_cudnn_dtype(v),
        name="V",
    )
    tensors.update(Q=q_t, K=k_t, V=v_t)

    options = _mask_options(
        cudnn,
        attn_mask_type,
        window_size,
        bottom_right_diagonal,
        max_seqlen_q,
        max_seqlen_kv,
    )
    is_padding = options.pop("is_padding")
    options.update(
        generate_stats=True,
        attn_scale=float(attn_scale),
        use_padding_mask=is_padding,
    )
    if is_padding:
        seq_q = _sequence_lengths(cu_seqlens_q)
        seq_kv = _sequence_lengths(cu_seqlens_kv)
        seq_q_t = graph.tensor_like(seq_q, name="seq_len_q")
        seq_kv_t = graph.tensor_like(seq_kv, name="seq_len_kv")
        tensors.update(seq_len_q=seq_q_t, seq_len_kv=seq_kv_t)
        options.update(seq_len_q=seq_q_t, seq_len_kv=seq_kv_t)
    if dropout != 0.0:
        seed_t = graph.tensor(
            name="dropout_seed",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT64,
        )
        offset_t = graph.tensor(
            name="dropout_offset",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT64,
        )
        tensors.update(dropout_seed=seed_t, dropout_offset=offset_t)
        options["dropout"] = (float(dropout), seed_t, offset_t)
    if softmax_type != "vanilla":
        sink_t = graph.tensor_like(softmax_offset, name="softmax_offset")
        tensors["softmax_offset"] = sink_t
        options["sink_token"] = sink_t

    if _is_mxfp8_tensor(q):
        # The MXFP8 binding has no padding-mask keyword. Avoid forwarding the
        # false default through the generic graph capture layer.
        options.pop("use_padding_mask", None)
        if is_padding:
            raise RuntimeError(
                "The installed cuDNN Frontend Python MXFP8 graph API does not expose "
                "padding sequence lengths."
            )
        scale_format_q = qkv_scale_inv_format or q_format
        scale_format_kv = qkv_scale_inv_format or kv_format
        padded = _mxfp8_padded_sizes(max_seqlen_q, max_seqlen_kv, d_qk, d_v)
        descale_q = _make_mxfp8_scale_tensor(
            graph,
            cudnn,
            name="Descale_Q",
            batch=batch,
            heads=heads,
            seqlen=padded["s_q_padded"],
            dim=padded["d_qk_scale_padded"],
            tensor_format=scale_format_q,
        )
        descale_k = _make_mxfp8_scale_tensor(
            graph,
            cudnn,
            name="Descale_K",
            batch=batch,
            heads=kv_heads,
            seqlen=padded["s_kv_padded"],
            dim=padded["d_qk_scale_padded"],
            tensor_format=scale_format_kv,
        )
        descale_v = _make_mxfp8_scale_tensor(
            graph,
            cudnn,
            name="Descale_V",
            batch=batch,
            heads=kv_heads,
            seqlen=padded["s_kv_scale_padded"],
            dim=padded["d_v_padded"],
            tensor_format=scale_format_kv,
        )
        tensors.update(descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)
        output_t, stats_t, amax_o_t = graph.sdpa_mxfp8(
            q_t,
            k_t,
            v_t,
            descale_q,
            descale_k,
            descale_v,
            name="te_sdpa_mxfp8",
            **options,
        )
        amax_o_t.set_output(False).set_data_type(cudnn.data_type.FLOAT).set_dim(
            (1, 1, 1, 1)
        ).set_stride((1, 1, 1, 1))
    else:
        if "diagonal_band_left_bound" in options:
            options["left_bound"] = options.pop("diagonal_band_left_bound")
        if "diagonal_band_right_bound" in options:
            options["right_bound"] = options.pop("diagonal_band_right_bound")
        descale_q = _scalar_graph_tensor(graph, cudnn, "Descale_Q")
        descale_k = _scalar_graph_tensor(graph, cudnn, "Descale_K")
        descale_v = _scalar_graph_tensor(graph, cudnn, "Descale_V")
        tensors.update(descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)
        if isinstance(s_quantizer, Float8Quantizer):
            descale_s = _scalar_graph_tensor(graph, cudnn, "Descale_S")
            scale_s = _scalar_graph_tensor(graph, cudnn, "Scale_S")
            tensors.update(descale_s=descale_s, scale_s=scale_s)
        else:
            descale_s = _constant_graph_tensor(graph, cudnn, "Current_Descale_S")
            scale_s = _constant_graph_tensor(graph, cudnn, "Current_Scale_S")
            tensors.update(constant_descale_s=descale_s, constant_scale_s=scale_s)
        if isinstance(o_quantizer, Float8Quantizer):
            scale_o = _scalar_graph_tensor(graph, cudnn, "Scale_O")
            tensors["scale_o"] = scale_o
        else:
            scale_o = _constant_graph_tensor(graph, cudnn, "Current_Scale_O")
            tensors["constant_scale_o"] = scale_o
        output_t, stats_t, amax_s_t, amax_o_t = graph.sdpa_fp8(
            q_t,
            k_t,
            v_t,
            descale_q,
            descale_k,
            descale_v,
            descale_s,
            scale_s,
            scale_o,
            name="te_sdpa_fp8",
            **options,
        )
        amax_s_t.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(
            (1, 1, 1, 1)
        ).set_stride((1, 1, 1, 1))
        amax_o_t.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(
            (1, 1, 1, 1)
        ).set_stride((1, 1, 1, 1))
        tensors.update(amax_s=amax_s_t, amax_o=amax_o_t)

    output_t.set_output(True).set_dim((batch, heads, max_seqlen_q, d_v)).set_stride(
        _format_stride(batch, heads, max_seqlen_q, d_v, o_format)
    )
    output_dtype = (
        _fp8_cudnn_dtype(output) if _is_float8_tensor(output) else output.dtype
    )
    output_t.set_data_type(output_dtype)
    stats_t.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(
        (batch, heads, max_seqlen_q, 1)
    ).set_stride((heads * max_seqlen_q, max_seqlen_q, 1, 1))
    tensors.update(O=output_t, Stats=stats_t)
    return GraphEntry(
        graph=graph, tensors=tensors, workspace_size=finalize_graph(graph)
    )


def _fp8_forward(
    is_training,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    q,
    k,
    v,
    fake_dtype,
    s_quantizer,
    o_quantizer,
    attn_scale,
    dropout,
    fast_zero_fill,
    qkv_layout,
    o_format,
    qkv_scale_inv_format,
    attn_mask_type,
    softmax_type,
    window_size,
    bottom_right_diagonal,
    rng_gen,
    softmax_offset,
):
    if not isinstance(q, QuantizedTensorStorage):
        raise TypeError(
            "The FP8 cuDNN attention backend requires quantized Q/K/V tensors."
        )
    if _is_mxfp8_tensor(q) and "padding" in attn_mask_type:
        # cuDNN Frontend 1.27's Python sdpa_mxfp8 forward binding omits the
        # seq_len inputs that its C++ graph API exposes. Preserve functional
        # padding semantics through the same Python graph API by dequantizing
        # MXFP8 inputs and using the BF16/FP16 SDPA node for this configuration.
        # Remove this fallback once the Python MXFP8 forward signature exposes
        # padding sequence lengths.
        q_hp, k_hp, v_hp = (tensor.dequantize(dtype=fake_dtype) for tensor in (q, k, v))
        output, aux, _ = _f16_forward(
            is_training,
            max_seqlen_q,
            max_seqlen_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            q_hp,
            k_hp,
            v_hp,
            fake_dtype,
            None,
            None,
            None,
            None,
            None,
            attn_scale,
            dropout,
            fast_zero_fill,
            qkv_layout,
            o_format,
            "no_bias",
            attn_mask_type,
            softmax_type,
            window_size,
            bottom_right_diagonal,
            rng_gen,
            softmax_offset,
            False,
        )
        return output, aux
    del is_training, fast_zero_fill
    q_format, _ = _q_kv_formats(qkv_layout)
    batch = cu_seqlens_q.numel() - 1
    heads = q.shape[-2] if q_format != "bhsd" else q.shape[1]
    d_v = v.shape[-1]
    output_shape = _fp8_output_shape(batch, max_seqlen_q, heads, d_v, o_format)
    output, amax_o = _allocate_fp8_kernel_output(
        o_quantizer, output_shape, fake_dtype, q.device
    )
    stats = torch.empty(
        (batch, heads, max_seqlen_q, 1), dtype=torch.float32, device=q.device
    )
    amax_s = (
        s_quantizer.amax
        if isinstance(s_quantizer, Float8Quantizer)
        else (
            torch.zeros(1, dtype=torch.float32, device=q.device)
            if not _is_mxfp8_tensor(q)
            else None
        )
    )
    rng_elts = (
        max_seqlen_q * max_seqlen_q + _FP8_THREADS_PER_CTA - 1
    ) // _FP8_THREADS_PER_CTA
    rng_state = _reserve_philox_state(q.device, rng_gen, rng_elts)

    q_data = _quantized_data(q)
    k_data = _quantized_data(k)
    v_data = _quantized_data(v, columnwise=_is_mxfp8_tensor(v))
    output_data = _quantized_data(output) if _is_float8_tensor(output) else output
    key = (
        "fp8_fwd",
        max_seqlen_q,
        max_seqlen_kv,
        _tensor_metadata(q_data),
        _tensor_metadata(k_data),
        _tensor_metadata(v_data),
        _tensor_metadata(output_data),
        type(q).__name__,
        type(o_quantizer).__name__,
        qkv_layout,
        o_format,
        qkv_scale_inv_format,
        float(attn_scale),
        float(dropout),
        attn_mask_type,
        softmax_type,
        tuple(window_size),
        bottom_right_diagonal,
    )
    entry = get_graph_entry(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cuDNN FP8 attention graph must be built before CUDA graph capture."
            )
        entry = _build_fp8_fwd_graph(
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            q=q,
            k=k,
            v=v,
            output=output,
            stats=stats,
            amax_s=amax_s,
            amax_o=amax_o,
            s_quantizer=s_quantizer,
            o_quantizer=o_quantizer,
            qkv_layout=qkv_layout,
            o_format=o_format,
            qkv_scale_inv_format=qkv_scale_inv_format,
            attn_scale=attn_scale,
            dropout=dropout,
            attn_mask_type=attn_mask_type,
            softmax_type=softmax_type,
            window_size=window_size,
            bottom_right_diagonal=bottom_right_diagonal,
            softmax_offset=softmax_offset,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
        )
        put_graph_entry(key, entry)

    t = entry.tensors
    variant_pack = {
        t["Q"]: q_data,
        t["K"]: k_data,
        t["V"]: v_data,
        t["O"]: output_data,
        t["Stats"]: stats,
        t["descale_q"]: _quantized_scale_inv(q),
        t["descale_k"]: _quantized_scale_inv(k),
        t["descale_v"]: _quantized_scale_inv(v, columnwise=_is_mxfp8_tensor(v)),
    }
    if "descale_s" in t:
        variant_pack[t["descale_s"]] = torch.reciprocal(s_quantizer.scale)
        variant_pack[t["scale_s"]] = s_quantizer.scale
    if "scale_o" in t:
        variant_pack[t["scale_o"]] = o_quantizer.scale
    one = None
    for name in ("constant_descale_s", "constant_scale_s", "constant_scale_o"):
        if name in t:
            if one is None:
                one = torch.ones(1, dtype=torch.float32, device=q.device)
            variant_pack[t[name]] = one
    if "amax_s" in t:
        variant_pack[t["amax_s"]] = amax_s
        variant_pack[t["amax_o"]] = amax_o
    if "seq_len_q" in t:
        variant_pack[t["seq_len_q"]] = _sequence_lengths(cu_seqlens_q)
        variant_pack[t["seq_len_kv"]] = _sequence_lengths(cu_seqlens_kv)
    if "dropout_seed" in t:
        variant_pack[t["dropout_seed"]] = rng_state[:1]
        variant_pack[t["dropout_offset"]] = rng_state[1:]
    if "softmax_offset" in t:
        variant_pack[t["softmax_offset"]] = softmax_offset
    entry.execute(variant_pack, q.device)
    aux = [stats, rng_state]
    if softmax_type != "vanilla":
        aux.append(softmax_offset)
    return output, aux


def fused_attn_fwd(
    is_training: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fake_dtype: torch.dtype,
    fused_attention_backend: FusedAttnBackend,
    attn_bias: torch.Tensor = None,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    page_table_k: torch.Tensor = None,
    page_table_v: torch.Tensor = None,
    s_quantizer=None,
    o_quantizer=None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
    o_format: str = "sbhd",
    qkv_scale_inv_format: str = None,
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    softmax_type: str = "vanilla",
    window_size: Tuple[int, int] = (-1, -1),
    bottom_right_diagonal: bool = None,
    rng_gen: torch.Generator = None,
    softmax_offset: torch.Tensor = None,
    return_max_logit: bool = False,
    cuda_graph: bool = False,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Execute fused attention through cuDNN Frontend's Python graph API."""

    del cuda_graph
    backend = FusedAttnBackend.cast(fused_attention_backend)
    if backend == FusedAttnBackend.No_Backend:
        raise ValueError(
            "No cuDNN fused-attention backend supports this configuration."
        )
    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(q.size(-1))
    if bottom_right_diagonal is None:
        bottom_right_diagonal = attn_mask_type in (
            "causal_bottom_right",
            "padding_causal_bottom_right",
        )
    if backend == FusedAttnBackend.FP8:
        if page_table_k is not None or page_table_v is not None:
            raise ValueError("FP8 fused attention does not support paged KV cache.")
        if attn_bias_type != "no_bias" or attn_bias is not None:
            raise ValueError("FP8 fused attention does not support attention bias.")
        if return_max_logit:
            raise ValueError(
                "FP8 fused attention does not support returning maximum logits."
            )
        return _fp8_forward(
            is_training,
            max_seqlen_q,
            max_seqlen_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            q,
            k,
            v,
            fake_dtype,
            s_quantizer,
            o_quantizer,
            attn_scale,
            dropout,
            fast_zero_fill,
            qkv_layout,
            o_format,
            qkv_scale_inv_format,
            attn_mask_type,
            softmax_type,
            window_size,
            bottom_right_diagonal,
            rng_gen,
            softmax_offset,
        )

    output, aux, max_logit = _f16_forward(
        is_training,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        fake_dtype,
        attn_bias,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        page_table_k,
        page_table_v,
        attn_scale,
        dropout,
        fast_zero_fill,
        qkv_layout,
        o_format,
        attn_bias_type,
        attn_mask_type,
        softmax_type,
        window_size,
        bottom_right_diagonal,
        rng_gen,
        softmax_offset,
        return_max_logit,
    )
    if return_max_logit:
        return output, aux, max_logit
    return output, aux


def _build_f16_bwd_graph(
    *,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    d_o: torch.Tensor,
    stats: torch.Tensor,
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    d_bias: Optional[torch.Tensor],
    softmax_offset: Optional[torch.Tensor],
    d_softmax_offset: Optional[torch.Tensor],
    cu_seqlens_q_padded: torch.Tensor,
    cu_seqlens_kv_padded: torch.Tensor,
    attn_scale: float,
    dropout: float,
    qkv_layout: str,
    o_format: str,
    do_format: str,
    dqkv_layout: str,
    attn_bias_type: str,
    attn_mask_type: str,
    softmax_type: str,
    window_size: Tuple[int, int],
    bottom_right_diagonal: bool,
    deterministic: bool,
) -> GraphEntry:
    cudnn = import_cudnn_frontend()
    graph = make_graph(
        torch_to_cudnn_dtype(q.dtype), q.device, name="te_fused_attention_bwd"
    )
    q_format, kv_format = _q_kv_formats(qkv_layout)
    dq_format, dkv_format = _q_kv_formats(dqkv_layout)
    batch = cu_seqlens_q.numel() - 1
    is_ragged_q = q_format == "thd"
    is_ragged_kv = kv_format == "thd"
    use_ragged_stats = is_ragged_q and cudnn.backend_version() >= 90600
    use_token_buckets = (
        cudnn.backend_version() >= 90600
        and torch.cuda.get_device_capability(q.device) != (12, 0)
    )
    use_legacy_offsets = is_ragged_q or is_ragged_kv
    graph_batch = (
        _max_ragged_batch(batch) if use_legacy_offsets and use_token_buckets else batch
    )
    if not use_token_buckets:
        use_ragged_stats = False
    graph_seqlen_q = (
        _max_ragged_tokens(q.shape[0])
        if is_ragged_q and use_token_buckets
        else max_seqlen_q
    )
    graph_seqlen_kv = (
        _max_ragged_tokens(k.shape[0])
        if is_ragged_kv and use_token_buckets
        else max_seqlen_kv
    )

    tensors: Dict[str, Any] = {
        "_legacy_offsets": use_legacy_offsets,
        "_graph_batch": graph_batch,
    }
    offset_q = offset_o = offset_k = offset_v = offset_stats = None
    if is_ragged_q:
        offset_q, q_mult = _ragged_offset_tensor(
            graph,
            cu_seqlens_q_padded,
            multiplier=1,
            name="offset_q",
            length=graph_batch + 1,
            data_type=torch.int64,
        )
        offset_o, o_mult = _ragged_offset_tensor(
            graph,
            cu_seqlens_q_padded,
            multiplier=1,
            name="offset_o",
            length=graph_batch + 1,
            data_type=torch.int64,
        )
        tensors.update(offset_q=offset_q, offset_o=offset_o)
    else:
        q_mult = o_mult = 1
    if is_ragged_kv:
        offset_k, k_mult = _ragged_offset_tensor(
            graph,
            cu_seqlens_kv_padded,
            multiplier=1,
            name="offset_k",
            length=graph_batch + 1,
            data_type=torch.int64,
        )
        offset_v, v_mult = _ragged_offset_tensor(
            graph,
            cu_seqlens_kv_padded,
            multiplier=1,
            name="offset_v",
            length=graph_batch + 1,
            data_type=torch.int64,
        )
        tensors.update(offset_k=offset_k, offset_v=offset_v)
    else:
        k_mult = v_mult = 1

    q_t = _make_bhsd_graph_tensor(
        graph,
        q,
        q_format,
        batch=graph_batch,
        max_seqlen=graph_seqlen_q,
        ragged_offset=offset_q,
        ragged_offset_multiplier=q_mult,
        name="Q",
    )
    k_t = _make_bhsd_graph_tensor(
        graph,
        k,
        kv_format,
        batch=graph_batch,
        max_seqlen=graph_seqlen_kv,
        ragged_offset=offset_k,
        ragged_offset_multiplier=k_mult,
        name="K",
    )
    v_t = _make_bhsd_graph_tensor(
        graph,
        v,
        kv_format,
        batch=graph_batch,
        max_seqlen=graph_seqlen_kv,
        ragged_offset=offset_v,
        ragged_offset_multiplier=v_mult,
        name="V",
    )
    o_t = _make_bhsd_graph_tensor(
        graph,
        o,
        o_format,
        batch=graph_batch,
        max_seqlen=graph_seqlen_q,
        ragged_offset=offset_o,
        ragged_offset_multiplier=o_mult,
        name="O",
    )
    do_t = _make_bhsd_graph_tensor(
        graph,
        d_o,
        do_format,
        batch=graph_batch,
        max_seqlen=graph_seqlen_q,
        ragged_offset=offset_o,
        ragged_offset_multiplier=1,
        name="dO",
    )
    tensors.update(Q=q_t, K=k_t, V=v_t, O=o_t, dO=do_t)

    stats_physical, stats_dim, stats_stride = _stats_layout(
        batch=graph_batch,
        heads=q_t.get_dim()[1],
        max_seqlen_q=graph_seqlen_q,
        total_tokens_q=q.shape[0] if is_ragged_q else batch * max_seqlen_q,
        ragged=use_ragged_stats,
    )
    del stats_physical
    if use_ragged_stats:
        offset_stats, stats_mult = _ragged_offset_tensor(
            graph,
            cu_seqlens_q_padded,
            multiplier=1,
            name="offset_stats",
            length=graph_batch + 1,
            data_type=torch.int64,
        )
        tensors["offset_stats"] = offset_stats
    else:
        stats_mult = 1
    stats_t = graph.tensor(
        name="Stats",
        dim=stats_dim,
        stride=stats_stride,
        data_type=cudnn.data_type.FLOAT,
        ragged_offset=offset_stats,
        ragged_offset_multiplier=stats_mult,
    )
    tensors["Stats"] = stats_t

    options = _mask_options(
        cudnn,
        attn_mask_type,
        window_size,
        bottom_right_diagonal,
        max_seqlen_q,
        max_seqlen_kv,
    )
    is_padding = options.pop("is_padding")
    options.update(
        attn_scale=float(attn_scale),
        use_padding_mask=is_padding,
        use_alibi_mask=attn_bias_type == "alibi",
        use_deterministic_algorithm=deterministic,
    )
    if use_ragged_stats:
        options["max_total_seq_len_q"] = graph_seqlen_q
    if (
        is_ragged_kv
        and cudnn.backend_version() >= 90600
        and torch.cuda.get_device_capability(q.device) != (12, 0)
    ):
        options["max_total_seq_len_kv"] = graph_seqlen_kv

    if attn_bias_type == "post_scale_bias":
        bias_t = graph.tensor_like(attn_bias, name="Bias")
        tensors["Bias"] = bias_t
        options["bias"] = bias_t
        if d_bias is not None:
            d_bias_t = graph.tensor_like(d_bias, name="dBias").set_output(True)
            tensors["dBias"] = d_bias_t
            options["dBias"] = d_bias_t

    if is_padding:
        seq_q = _padded_sequence_lengths(cu_seqlens_q, graph_batch)
        seq_kv = _padded_sequence_lengths(cu_seqlens_kv, graph_batch)
        seq_q_t = graph.tensor_like(seq_q, name="seq_len_q")
        seq_kv_t = graph.tensor_like(seq_kv, name="seq_len_kv")
        tensors.update(seq_len_q=seq_q_t, seq_len_kv=seq_kv_t)
        options.update(seq_len_q=seq_q_t, seq_len_kv=seq_kv_t)

    if dropout != 0.0:
        seed_t = graph.tensor(
            name="dropout_seed",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT64,
        )
        offset_t = graph.tensor(
            name="dropout_offset",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT64,
        )
        tensors.update(dropout_seed=seed_t, dropout_offset=offset_t)
        options["dropout"] = (float(dropout), seed_t, offset_t)

    if softmax_type != "vanilla":
        if softmax_offset is None or d_softmax_offset is None:
            raise ValueError(f"softmax_type={softmax_type!r} requires sink tensors.")
        sink_t = graph.tensor_like(softmax_offset, name="softmax_offset")
        dsink_t = graph.tensor_like(
            d_softmax_offset, name="d_softmax_offset"
        ).set_output(True)
        tensors.update(softmax_offset=sink_t, d_softmax_offset=dsink_t)
        options.update(sink_token=sink_t, dSink_token=dsink_t)

    dq_t, dk_t, dv_t = graph.sdpa_backward(
        name="te_sdpa_backward",
        q=q_t,
        k=k_t,
        v=v_t,
        o=o_t,
        dO=do_t,
        stats=stats_t,
        **options,
    )
    dq_dim, dq_stride = _logical_bhsd_desc(
        d_q, dq_format, batch=graph_batch, max_seqlen=graph_seqlen_q
    )
    dk_dim, dk_stride = _logical_bhsd_desc(
        d_k, dkv_format, batch=graph_batch, max_seqlen=graph_seqlen_kv
    )
    dv_dim, dv_stride = _logical_bhsd_desc(
        d_v, dkv_format, batch=graph_batch, max_seqlen=graph_seqlen_kv
    )
    dq_t.set_output(True).set_dim(dq_dim).set_stride(dq_stride)
    dk_t.set_output(True).set_dim(dk_dim).set_stride(dk_stride)
    dv_t.set_output(True).set_dim(dv_dim).set_stride(dv_stride)
    if is_ragged_q:
        dq_t.set_ragged_offset(offset_q).set_ragged_offset_multiplier(1)
    if is_ragged_kv:
        dk_t.set_ragged_offset(offset_k).set_ragged_offset_multiplier(1)
        dv_t.set_ragged_offset(offset_v).set_ragged_offset_multiplier(1)
    tensors.update(dQ=dq_t, dK=dk_t, dV=dv_t)

    return GraphEntry(
        graph=graph, tensors=tensors, workspace_size=finalize_graph(graph)
    )


def _build_fp8_bwd_graph(
    *,
    max_seqlen_q,
    max_seqlen_kv,
    q,
    k,
    v,
    o,
    d_o,
    d_o_f16,
    stats,
    d_q,
    d_k,
    d_v,
    s_quantizer,
    dp_quantizer,
    dqkv_quantizer,
    qkv_layout,
    o_format,
    do_format,
    dqkv_layout,
    qkv_scale_inv_format,
    do_scale_inv_format,
    attn_scale,
    dropout,
    attn_mask_type,
    softmax_type,
    window_size,
    bottom_right_diagonal,
    deterministic,
    softmax_offset,
    d_softmax_offset,
    cu_seqlens_q,
    cu_seqlens_kv,
):
    cudnn = import_cudnn_frontend()
    q_format, kv_format = _q_kv_formats(qkv_layout)
    dq_format, dkv_format = _q_kv_formats(dqkv_layout)
    batch = cu_seqlens_q.numel() - 1
    heads = q.shape[-2] if q_format != "bhsd" else q.shape[1]
    kv_heads = k.shape[-2] if kv_format != "bhsd" else k.shape[1]
    d_qk, d_value = q.shape[-1], v.shape[-1]
    q_data = _quantized_data(q)
    k_data = _quantized_data(k)
    v_data = _quantized_data(v)
    o_data = _quantized_data(o) if _is_float8_tensor(o) else o
    do_data = _quantized_data(d_o)
    dq_data = _quantized_data(d_q) if _is_float8_tensor(d_q) else d_q
    dk_data = _quantized_data(d_k) if _is_float8_tensor(d_k) else d_k
    dv_data = _quantized_data(d_v) if _is_float8_tensor(d_v) else d_v
    graph = make_graph(_fp8_cudnn_dtype(q), q.device, name="te_fp8_sdpa_bwd")
    tensors: Dict[str, Any] = {}

    q_t = _make_bhsd_graph_tensor(
        graph,
        q_data,
        q_format,
        batch=batch,
        max_seqlen=max_seqlen_q,
        data_type=_fp8_cudnn_dtype(q),
        name="Q",
    )
    k_t = _make_bhsd_graph_tensor(
        graph,
        k_data,
        kv_format,
        batch=batch,
        max_seqlen=max_seqlen_kv,
        data_type=_fp8_cudnn_dtype(k),
        name="K",
    )
    v_t = _make_bhsd_graph_tensor(
        graph,
        v_data,
        kv_format,
        batch=batch,
        max_seqlen=max_seqlen_kv,
        data_type=_fp8_cudnn_dtype(v),
        name="V",
    )
    o_dtype = _fp8_cudnn_dtype(o) if _is_float8_tensor(o) else o.dtype
    o_t = _make_bhsd_graph_tensor(
        graph,
        o_data,
        o_format,
        batch=batch,
        max_seqlen=max_seqlen_q,
        data_type=o_dtype,
        name="O",
    )
    do_t = _make_bhsd_graph_tensor(
        graph,
        do_data,
        do_format,
        batch=batch,
        max_seqlen=max_seqlen_q,
        data_type=_fp8_cudnn_dtype(d_o),
        name="dO",
    )
    stats_t = graph.tensor_like(stats, name="Stats")
    tensors.update(Q=q_t, K=k_t, V=v_t, O=o_t, dO=do_t, Stats=stats_t)

    options = _mask_options(
        cudnn,
        attn_mask_type,
        window_size,
        bottom_right_diagonal,
        max_seqlen_q,
        max_seqlen_kv,
    )
    is_padding = options.pop("is_padding")
    if "diagonal_band_left_bound" in options:
        options["left_bound"] = options.pop("diagonal_band_left_bound")
    if "diagonal_band_right_bound" in options:
        options["right_bound"] = options.pop("diagonal_band_right_bound")
    options.update(
        attn_scale=float(attn_scale),
        use_padding_mask=is_padding,
        use_deterministic_algorithm=deterministic,
    )
    if is_padding:
        seq_q = _sequence_lengths(cu_seqlens_q)
        seq_kv = _sequence_lengths(cu_seqlens_kv)
        seq_q_t = graph.tensor_like(seq_q, name="seq_len_q")
        seq_kv_t = graph.tensor_like(seq_kv, name="seq_len_kv")
        tensors.update(seq_len_q=seq_q_t, seq_len_kv=seq_kv_t)
        options.update(seq_len_q=seq_q_t, seq_len_kv=seq_kv_t)
    if dropout != 0.0:
        seed_t = graph.tensor(
            name="dropout_seed",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT64,
        )
        offset_t = graph.tensor(
            name="dropout_offset",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.INT64,
        )
        tensors.update(dropout_seed=seed_t, dropout_offset=offset_t)
        options["dropout"] = (float(dropout), seed_t, offset_t)
    if softmax_type != "vanilla":
        sink_t = graph.tensor_like(softmax_offset, name="softmax_offset")
        dsink_t = graph.tensor_like(
            d_softmax_offset, name="d_softmax_offset"
        ).set_output(True)
        tensors.update(softmax_offset=sink_t, d_softmax_offset=dsink_t)
        options.update(sink_token=sink_t, dSink_token=dsink_t)

    if _is_mxfp8_tensor(q):
        if d_o_f16 is None:
            raise ValueError(
                "MXFP8 attention backward requires the high-precision dO tensor."
            )
        q_col = _quantized_data(q, columnwise=True)
        k_col = _quantized_data(k, columnwise=True)
        do_col = _quantized_data(d_o, columnwise=True)
        q_col_t = _make_bhsd_graph_tensor(
            graph,
            q_col,
            q_format,
            batch=batch,
            max_seqlen=max_seqlen_q,
            data_type=_fp8_cudnn_dtype(q),
            name="Q_T",
        )
        k_col_t = _make_bhsd_graph_tensor(
            graph,
            k_col,
            kv_format,
            batch=batch,
            max_seqlen=max_seqlen_kv,
            data_type=_fp8_cudnn_dtype(k),
            name="K_T",
        )
        do_col_t = _make_bhsd_graph_tensor(
            graph,
            do_col,
            do_format,
            batch=batch,
            max_seqlen=max_seqlen_q,
            data_type=_fp8_cudnn_dtype(d_o),
            name="dO_T",
        )
        do_f16_t = _make_bhsd_graph_tensor(
            graph,
            d_o_f16,
            do_format,
            batch=batch,
            max_seqlen=max_seqlen_q,
            data_type=d_o_f16.dtype,
            name="dO_f16",
        )
        scale_format_q = qkv_scale_inv_format or q_format
        scale_format_kv = qkv_scale_inv_format or kv_format
        scale_format_do = do_scale_inv_format or do_format
        padded = _mxfp8_padded_sizes(max_seqlen_q, max_seqlen_kv, d_qk, d_value)

        def mx_scale(name, h, s, d, fmt):
            tensor = _make_mxfp8_scale_tensor(
                graph,
                cudnn,
                name=name,
                batch=batch,
                heads=h,
                seqlen=padded[s],
                dim=padded[d],
                tensor_format=fmt,
            )
            tensors[name] = tensor
            return tensor

        descale_q = mx_scale(
            "descale_q", heads, "s_q_padded", "d_qk_scale_padded", scale_format_q
        )
        descale_q_t = mx_scale(
            "descale_q_t", heads, "s_q_scale_padded", "d_qk_padded", scale_format_q
        )
        descale_k = mx_scale(
            "descale_k", kv_heads, "s_kv_padded", "d_qk_scale_padded", scale_format_kv
        )
        descale_k_t = mx_scale(
            "descale_k_t", kv_heads, "s_kv_scale_padded", "d_qk_padded", scale_format_kv
        )
        descale_v = mx_scale(
            "descale_v", kv_heads, "s_kv_padded", "d_v_scale_padded", scale_format_kv
        )
        descale_do = mx_scale(
            "descale_do", heads, "s_q_padded", "d_v_scale_padded", scale_format_do
        )
        descale_do_t = mx_scale(
            "descale_do_t", heads, "s_q_scale_padded", "d_v_padded", scale_format_do
        )
        outputs = graph.sdpa_mxfp8_backward(
            q_t,
            q_col_t,
            k_t,
            k_col_t,
            v_t,
            o_t,
            do_f16_t,
            do_t,
            do_col_t,
            stats_t,
            descale_q,
            descale_q_t,
            descale_k,
            descale_k_t,
            descale_v,
            descale_do,
            descale_do_t,
            name="te_sdpa_mxfp8_backward",
            **options,
        )
        dq_t, dk_t, dv_t, *amax_outputs = outputs
        for amax_t in amax_outputs:
            amax_t.set_output(False).set_data_type(cudnn.data_type.FLOAT).set_dim(
                (1, 1, 1, 1)
            ).set_stride((1, 1, 1, 1))
        tensors.update(Q_T=q_col_t, K_T=k_col_t, dO_T=do_col_t, dO_f16=do_f16_t)
    else:
        scalar_names = (
            "descale_q",
            "descale_k",
            "descale_v",
            "descale_o",
            "descale_do",
        )
        scalars = {
            name: _scalar_graph_tensor(graph, cudnn, name) for name in scalar_names
        }
        tensors.update(scalars)
        delayed = isinstance(dqkv_quantizer, Float8Quantizer)
        if isinstance(s_quantizer, Float8Quantizer):
            for name in ("descale_s", "scale_s"):
                tensors[name] = _scalar_graph_tensor(graph, cudnn, name)
        else:
            for name in ("descale_s", "scale_s"):
                tensors[f"constant_{name}"] = _constant_graph_tensor(graph, cudnn, name)
        if isinstance(dp_quantizer, Float8Quantizer):
            for name in ("descale_dp", "scale_dp"):
                tensors[name] = _scalar_graph_tensor(graph, cudnn, name)
        else:
            for name in ("descale_dp", "scale_dp"):
                tensors[f"constant_{name}"] = _constant_graph_tensor(graph, cudnn, name)
        for name in ("scale_dq", "scale_dk", "scale_dv"):
            key = name if delayed else f"constant_{name}"
            tensors[key] = _scalar_graph_tensor(graph, cudnn, name)
        descale_o_arg = tensors["descale_o"]
        descale_s_arg = (
            tensors["descale_s"]
            if "descale_s" in tensors
            else tensors["constant_descale_s"]
        )
        descale_dp_arg = (
            tensors["descale_dp"]
            if "descale_dp" in tensors
            else tensors["constant_descale_dp"]
        )
        scale_s_arg = (
            tensors["scale_s"] if "scale_s" in tensors else tensors["constant_scale_s"]
        )
        scale_dp_arg = (
            tensors["scale_dp"]
            if "scale_dp" in tensors
            else tensors["constant_scale_dp"]
        )
        scale_dq_arg = (
            tensors["scale_dq"]
            if "scale_dq" in tensors
            else tensors["constant_scale_dq"]
        )
        scale_dk_arg = (
            tensors["scale_dk"]
            if "scale_dk" in tensors
            else tensors["constant_scale_dk"]
        )
        scale_dv_arg = (
            tensors["scale_dv"]
            if "scale_dv" in tensors
            else tensors["constant_scale_dv"]
        )
        outputs = graph.sdpa_fp8_backward(
            q_t,
            k_t,
            v_t,
            o_t,
            do_t,
            stats_t,
            tensors["descale_q"],
            tensors["descale_k"],
            tensors["descale_v"],
            descale_o_arg,
            tensors["descale_do"],
            descale_s_arg,
            descale_dp_arg,
            scale_s_arg,
            scale_dq_arg,
            scale_dk_arg,
            scale_dv_arg,
            scale_dp_arg,
            name="te_sdpa_fp8_backward",
            **options,
        )
        dq_t, dk_t, dv_t, amax_dq_t, amax_dk_t, amax_dv_t, amax_dp_t = outputs
        for name, amax_t in (
            ("amax_dq", amax_dq_t),
            ("amax_dk", amax_dk_t),
            ("amax_dv", amax_dv_t),
            ("amax_dp", amax_dp_t),
        ):
            amax_t.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(
                (1, 1, 1, 1)
            ).set_stride((1, 1, 1, 1))
            tensors[name] = amax_t

    dq_t.set_output(True).set_data_type(
        _fp8_cudnn_dtype(d_q) if _is_float8_tensor(d_q) else d_q.dtype
    ).set_dim((batch, heads, max_seqlen_q, d_qk)).set_stride(
        _format_stride(batch, heads, max_seqlen_q, d_qk, dq_format)
    )
    dk_t.set_output(True).set_data_type(
        _fp8_cudnn_dtype(d_k) if _is_float8_tensor(d_k) else d_k.dtype
    ).set_dim((batch, kv_heads, max_seqlen_kv, d_qk)).set_stride(
        _format_stride(batch, kv_heads, max_seqlen_kv, d_qk, dkv_format)
    )
    dv_t.set_output(True).set_data_type(
        _fp8_cudnn_dtype(d_v) if _is_float8_tensor(d_v) else d_v.dtype
    ).set_dim((batch, kv_heads, max_seqlen_kv, d_value)).set_stride(
        _format_stride(batch, kv_heads, max_seqlen_kv, d_value, dkv_format)
    )
    tensors.update(dQ=dq_t, dK=dk_t, dV=dv_t)
    return GraphEntry(
        graph=graph, tensors=tensors, workspace_size=finalize_graph(graph)
    )


def _fp8_backward(
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    q,
    k,
    v,
    o,
    d_o,
    fake_dtype,
    aux_ctx_tensors,
    s_quantizer,
    dp_quantizer,
    dqkv_quantizer,
    attn_scale,
    dropout,
    fast_zero_fill,
    qkv_layout,
    o_format,
    do_format,
    dqkv_layout,
    qkv_scale_inv_format,
    do_scale_inv_format,
    attn_mask_type,
    softmax_type,
    window_size,
    bottom_right_diagonal,
    deterministic,
):
    if _is_mxfp8_tensor(q) and "padding" in attn_mask_type:
        q_hp, k_hp, v_hp = (tensor.dequantize(dtype=fake_dtype) for tensor in (q, k, v))
        d_o_f16 = aux_ctx_tensors[-1]
        aux_count = 3 if softmax_type != "vanilla" else 2
        return fused_attn_bwd(
            max_seqlen_q,
            max_seqlen_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            q_hp,
            k_hp,
            v_hp,
            o,
            d_o_f16,
            fake_dtype,
            aux_ctx_tensors[:aux_count],
            FusedAttnBackend.F16_arbitrary_seqlen,
            s_quantizer=None,
            dp_quantizer=None,
            dqkv_quantizer=None,
            attn_scale=attn_scale,
            dropout=dropout,
            fast_zero_fill=fast_zero_fill,
            qkv_layout=qkv_layout,
            o_format=o_format,
            do_format=do_format,
            dqkv_layout=dqkv_layout,
            attn_bias_type="no_bias",
            attn_mask_type=attn_mask_type,
            softmax_type=softmax_type,
            window_size=window_size,
            bottom_right_diagonal=bottom_right_diagonal,
            deterministic=deterministic,
        )
    q_format, kv_format = _q_kv_formats(qkv_layout)
    batch = cu_seqlens_q.numel() - 1
    heads = q.shape[-2] if q_format != "bhsd" else q.shape[1]
    kv_heads = k.shape[-2] if kv_format != "bhsd" else k.shape[1]
    output_dtype = (
        torch.uint8 if isinstance(dqkv_quantizer, Float8Quantizer) else fake_dtype
    )
    grad_data = _allocate_attention_grad_data(
        batch=batch,
        heads=heads,
        kv_heads=kv_heads,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_qk=q.shape[-1],
        head_dim_v=v.shape[-1],
        dqkv_layout=dqkv_layout,
        dtype=output_dtype,
        device=q.device,
        zero=fast_zero_fill,
    )
    if isinstance(dqkv_quantizer, Float8Quantizer):
        d_q, d_k, d_v = _wrap_float8_grad_outputs(dqkv_quantizer, grad_data, fake_dtype)
    else:
        d_q, d_k, d_v = grad_data
    stats, rng_state = aux_ctx_tensors[:2]
    softmax_offset = aux_ctx_tensors[2] if softmax_type != "vanilla" else None
    d_o_f16 = aux_ctx_tensors[-1] if _is_mxfp8_tensor(q) else None
    d_softmax_offset = (
        torch.empty_like(softmax_offset) if softmax_offset is not None else None
    )
    hidden_amax = [
        torch.zeros(1, dtype=torch.float32, device=q.device) for _ in range(4)
    ]

    key = (
        "fp8_bwd",
        max_seqlen_q,
        max_seqlen_kv,
        *(
            _tensor_metadata(_quantized_data(x) if _is_float8_tensor(x) else x)
            for x in (q, k, v, o, d_o)
        ),
        type(q).__name__,
        type(dqkv_quantizer).__name__,
        qkv_layout,
        o_format,
        do_format,
        dqkv_layout,
        qkv_scale_inv_format,
        do_scale_inv_format,
        float(attn_scale),
        float(dropout),
        attn_mask_type,
        softmax_type,
        tuple(window_size),
        bottom_right_diagonal,
        deterministic,
    )
    entry = get_graph_entry(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cuDNN FP8 attention graph must be built before CUDA graph capture."
            )
        entry = _build_fp8_bwd_graph(
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            q=q,
            k=k,
            v=v,
            o=o,
            d_o=d_o,
            d_o_f16=d_o_f16,
            stats=stats,
            d_q=d_q,
            d_k=d_k,
            d_v=d_v,
            s_quantizer=s_quantizer,
            dp_quantizer=dp_quantizer,
            dqkv_quantizer=dqkv_quantizer,
            qkv_layout=qkv_layout,
            o_format=o_format,
            do_format=do_format,
            dqkv_layout=dqkv_layout,
            qkv_scale_inv_format=qkv_scale_inv_format,
            do_scale_inv_format=do_scale_inv_format,
            attn_scale=attn_scale,
            dropout=dropout,
            attn_mask_type=attn_mask_type,
            softmax_type=softmax_type,
            window_size=window_size,
            bottom_right_diagonal=bottom_right_diagonal,
            deterministic=deterministic,
            softmax_offset=softmax_offset,
            d_softmax_offset=d_softmax_offset,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
        )
        put_graph_entry(key, entry)

    t = entry.tensors
    variant_pack = {
        t["Q"]: _quantized_data(q),
        t["K"]: _quantized_data(k),
        t["V"]: _quantized_data(v),
        t["O"]: _quantized_data(o) if _is_float8_tensor(o) else o,
        t["dO"]: _quantized_data(d_o),
        t["Stats"]: stats,
        t["dQ"]: _quantized_data(d_q) if _is_float8_tensor(d_q) else d_q,
        t["dK"]: _quantized_data(d_k) if _is_float8_tensor(d_k) else d_k,
        t["dV"]: _quantized_data(d_v) if _is_float8_tensor(d_v) else d_v,
    }
    if _is_mxfp8_tensor(q):
        variant_pack.update(
            {
                t["Q_T"]: _quantized_data(q, columnwise=True),
                t["K_T"]: _quantized_data(k, columnwise=True),
                t["dO_T"]: _quantized_data(d_o, columnwise=True),
                t["dO_f16"]: d_o_f16,
                t["descale_q"]: _quantized_scale_inv(q),
                t["descale_q_t"]: _quantized_scale_inv(q, columnwise=True),
                t["descale_k"]: _quantized_scale_inv(k),
                t["descale_k_t"]: _quantized_scale_inv(k, columnwise=True),
                t["descale_v"]: _quantized_scale_inv(v),
                t["descale_do"]: _quantized_scale_inv(d_o),
                t["descale_do_t"]: _quantized_scale_inv(d_o, columnwise=True),
            }
        )
    else:
        variant_pack.update(
            {
                t["descale_q"]: _quantized_scale_inv(q),
                t["descale_k"]: _quantized_scale_inv(k),
                t["descale_v"]: _quantized_scale_inv(v),
                t["descale_o"]: (
                    _quantized_scale_inv(o)
                    if _is_float8_tensor(o)
                    else torch.ones(1, dtype=torch.float32, device=q.device)
                ),
                t["descale_do"]: _quantized_scale_inv(d_o),
            }
        )
        if "descale_s" in t:
            variant_pack[t["descale_s"]] = torch.reciprocal(s_quantizer.scale)
            variant_pack[t["scale_s"]] = s_quantizer.scale
        if "descale_dp" in t:
            variant_pack[t["descale_dp"]] = torch.reciprocal(dp_quantizer.scale)
            variant_pack[t["scale_dp"]] = dp_quantizer.scale
        for name in ("scale_dq", "scale_dk", "scale_dv"):
            if name in t:
                variant_pack[t[name]] = dqkv_quantizer.scale
        one = torch.ones(1, dtype=torch.float32, device=q.device)
        for name, graph_tensor in t.items():
            if name.startswith("constant_"):
                variant_pack[graph_tensor] = one
        amax_values = (
            (
                dqkv_quantizer.amax,
                dqkv_quantizer.amax,
                dqkv_quantizer.amax,
                dp_quantizer.amax,
            )
            if isinstance(dqkv_quantizer, Float8Quantizer)
            else hidden_amax
        )
        for name, value in zip(
            ("amax_dq", "amax_dk", "amax_dv", "amax_dp"), amax_values
        ):
            variant_pack[t[name]] = value
    if "seq_len_q" in t:
        variant_pack[t["seq_len_q"]] = _sequence_lengths(cu_seqlens_q)
        variant_pack[t["seq_len_kv"]] = _sequence_lengths(cu_seqlens_kv)
    if "dropout_seed" in t:
        variant_pack[t["dropout_seed"]] = rng_state[:1]
        variant_pack[t["dropout_offset"]] = rng_state[1:]
    if "softmax_offset" in t:
        variant_pack[t["softmax_offset"]] = softmax_offset
        variant_pack[t["d_softmax_offset"]] = d_softmax_offset
    entry.execute(variant_pack, q.device)
    return d_q, d_k, d_v, None, d_softmax_offset


def fused_attn_bwd(
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    d_o: torch.Tensor,
    fake_dtype: torch.dtype,
    aux_ctx_tensors: List[torch.Tensor],
    fused_attention_backend: FusedAttnBackend,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    s_quantizer=None,
    dp_quantizer=None,
    dqkv_quantizer=None,
    attn_scale: Optional[float] = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
    o_format: str = "sbhd",
    do_format: str = "sbhd",
    dqkv_layout: str = "sbh3d",
    qkv_scale_inv_format: str = None,
    do_scale_inv_format: str = None,
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    softmax_type: str = "vanilla",
    window_size: Tuple[int, int] = (-1, -1),
    bottom_right_diagonal: bool = None,
    deterministic: bool = False,
    cuda_graph: bool = False,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Execute fused-attention backward through the Python graph API."""

    del cuda_graph
    backend = FusedAttnBackend.cast(fused_attention_backend)
    if not aux_ctx_tensors:
        raise ValueError("Fused-attention backward requires forward auxiliary tensors.")
    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(q.size(-1))
    if bottom_right_diagonal is None:
        bottom_right_diagonal = attn_mask_type in (
            "causal_bottom_right",
            "padding_causal_bottom_right",
        )
    if backend == FusedAttnBackend.FP8:
        if attn_bias_type != "no_bias":
            raise ValueError(
                "FP8 fused attention backward does not support attention bias."
            )
        return _fp8_backward(
            max_seqlen_q,
            max_seqlen_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            q,
            k,
            v,
            o,
            d_o,
            fake_dtype,
            aux_ctx_tensors,
            s_quantizer,
            dp_quantizer,
            dqkv_quantizer,
            attn_scale,
            dropout,
            fast_zero_fill,
            qkv_layout,
            o_format,
            do_format,
            dqkv_layout,
            qkv_scale_inv_format,
            do_scale_inv_format,
            attn_mask_type,
            softmax_type,
            window_size,
            bottom_right_diagonal,
            deterministic,
        )
    if backend != FusedAttnBackend.F16_arbitrary_seqlen:
        raise ValueError(
            "No cuDNN fused-attention backend supports this backward configuration."
        )

    stats = aux_ctx_tensors[0]
    rng_state = aux_ctx_tensors[1]
    aux_index = 2
    attn_bias = None
    if attn_bias_type not in ("no_bias", "alibi"):
        attn_bias = aux_ctx_tensors[aux_index]
        aux_index += 1
    softmax_offset = None
    if softmax_type != "vanilla":
        softmax_offset = aux_ctx_tensors[aux_index]

    q_format, kv_format = _q_kv_formats(qkv_layout)
    batch = cu_seqlens_q.numel() - 1
    if q_format == "thd" or kv_format == "thd":
        d_q, d_k, d_v = _allocate_grad_views((q, k, v), fast_zero_fill=fast_zero_fill)
    else:
        heads = q.shape[1] if q_format == "bhsd" else q.shape[-2]
        kv_heads = k.shape[1] if kv_format == "bhsd" else k.shape[-2]
        d_q, d_k, d_v = _allocate_attention_grad_data(
            batch=batch,
            heads=heads,
            kv_heads=kv_heads,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            head_dim_qk=q.shape[-1],
            head_dim_v=v.shape[-1],
            dqkv_layout=dqkv_layout,
            dtype=q.dtype,
            device=q.device,
            zero=fast_zero_fill,
        )
    d_bias = None
    if attn_bias_type == "post_scale_bias":
        # cuDNN does not support the [1,1,1,S] reduction form.
        if not tuple(attn_bias.shape[:3]) == (1, 1, 1):
            d_bias = torch.empty_like(attn_bias)
    d_softmax_offset = (
        torch.empty_like(softmax_offset) if softmax_type != "vanilla" else None
    )
    cu_seqlens_q_padded = (
        cu_seqlens_q if cu_seqlens_q_padded is None else cu_seqlens_q_padded
    )
    cu_seqlens_kv_padded = (
        cu_seqlens_kv if cu_seqlens_kv_padded is None else cu_seqlens_kv_padded
    )
    cudnn = import_cudnn_frontend()
    use_token_buckets = (
        cudnn.backend_version() >= 90600
        and torch.cuda.get_device_capability(q.device) != (12, 0)
    )
    use_legacy_offsets = q_format == "thd" or kv_format == "thd"
    graph_batch = (
        _max_ragged_batch(batch) if use_legacy_offsets and use_token_buckets else batch
    )

    key = (
        "f16_bwd",
        max_seqlen_q,
        max_seqlen_kv,
        graph_batch,
        _tensor_metadata(q),
        _tensor_metadata(k),
        _tensor_metadata(v),
        _tensor_metadata(o),
        _tensor_metadata(d_o),
        _tensor_metadata(stats),
        _tensor_metadata(d_q),
        _tensor_metadata(d_k),
        _tensor_metadata(d_v),
        _tensor_metadata(attn_bias),
        qkv_layout,
        o_format,
        do_format,
        dqkv_layout,
        float(attn_scale),
        float(dropout),
        attn_bias_type,
        attn_mask_type,
        softmax_type,
        tuple(window_size),
        bottom_right_diagonal,
        deterministic,
    )
    entry = get_graph_entry(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cuDNN attention graph must be built before CUDA graph capture."
            )
        entry = _build_f16_bwd_graph(
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            q=q,
            k=k,
            v=v,
            o=o,
            d_o=d_o,
            stats=stats,
            d_q=d_q,
            d_k=d_k,
            d_v=d_v,
            attn_bias=attn_bias,
            d_bias=d_bias,
            softmax_offset=softmax_offset,
            d_softmax_offset=d_softmax_offset,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            attn_scale=attn_scale,
            dropout=dropout,
            qkv_layout=qkv_layout,
            o_format=o_format,
            do_format=do_format,
            dqkv_layout=dqkv_layout,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            softmax_type=softmax_type,
            window_size=window_size,
            bottom_right_diagonal=bottom_right_diagonal,
            deterministic=deterministic,
        )
        put_graph_entry(key, entry)

    tensors = entry.tensors
    variant_pack: Dict[Any, Any] = {
        tensors["Q"]: q,
        tensors["K"]: k,
        tensors["V"]: v,
        tensors["O"]: o,
        tensors["dO"]: d_o,
        tensors["Stats"]: stats,
        tensors["dQ"]: d_q,
        tensors["dK"]: d_k,
        tensors["dV"]: d_v,
    }
    if "Bias" in tensors:
        variant_pack[tensors["Bias"]] = attn_bias
    if "dBias" in tensors:
        variant_pack[tensors["dBias"]] = d_bias
    graph_batch = tensors["_graph_batch"]
    if "seq_len_q" in tensors:
        variant_pack[tensors["seq_len_q"]] = _padded_sequence_lengths(
            cu_seqlens_q, graph_batch
        )
        variant_pack[tensors["seq_len_kv"]] = _padded_sequence_lengths(
            cu_seqlens_kv, graph_batch
        )
    if "offset_q" in tensors:
        variant_pack[tensors["offset_q"]] = _element_ragged_offsets(
            cu_seqlens_q_padded, graph_batch, q.stride(0)
        )
        variant_pack[tensors["offset_o"]] = _element_ragged_offsets(
            cu_seqlens_q_padded, graph_batch, o.stride(0)
        )
    if "offset_k" in tensors:
        variant_pack[tensors["offset_k"]] = _element_ragged_offsets(
            cu_seqlens_kv_padded, graph_batch, k.stride(0)
        )
        variant_pack[tensors["offset_v"]] = _element_ragged_offsets(
            cu_seqlens_kv_padded, graph_batch, v.stride(0)
        )
    if "offset_stats" in tensors:
        stats_multiplier = stats.stride(0)
        variant_pack[tensors["offset_stats"]] = _element_ragged_offsets(
            cu_seqlens_q_padded, graph_batch, stats_multiplier
        )
    if "dropout_seed" in tensors:
        variant_pack[tensors["dropout_seed"]] = rng_state[:1]
        variant_pack[tensors["dropout_offset"]] = rng_state[1:]
    if "softmax_offset" in tensors:
        variant_pack[tensors["softmax_offset"]] = softmax_offset
        variant_pack[tensors["d_softmax_offset"]] = d_softmax_offset
    entry.execute(variant_pack, q.device)
    return d_q, d_k, d_v, d_bias, d_softmax_offset
