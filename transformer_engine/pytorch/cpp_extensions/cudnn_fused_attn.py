# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN attention graphs constructed and executed through the Python frontend.

The extension is used only to obtain capture-safe Philox state. Tensor allocation,
graph construction, plan caching, and execution do not use the common attention API.
"""

from dataclasses import dataclass, replace
import os
import threading

import torch
import transformer_engine_torch as tex

from transformer_engine.common.cudnn.attention import (
    AttentionOptions,
    AttentionGraphSpec,
    build_attention_graph,
)
from transformer_engine.common.cudnn.graph import (
    TensorSpec as _TensorSpec,
    CacheDiagnostics,
    attention_matrix_spec,
    attention_stats_spec,
    mxfp8_scale_spec,
    check_graph_support,
    compile_graph,
    bucket,
    uses_ragged_attention_graph,
)

from .cudnn import (
    _import_cudnn_frontend,
    _build_cudnn_pygraph,
    _execute_cudnn_graph,
)
from ..tensor.float8_tensor import Float8Quantizer
from ..tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ..constants import DType

_plans = {}
_cache_lock = threading.RLock()
_diagnostics = CacheDiagnostics("PyTorch")


def _event(attn, event, key=None):
    _diagnostics.event(
        "fp8" if attn.fp8 else "f16",
        "bwd" if attn.backward else "fwd",
        attn.device.index,
        event,
        key,
    )


def _build_stage(attn, name, function):
    return _diagnostics.run(
        "fp8" if attn.fp8 else "f16",
        "bwd" if attn.backward else "fwd",
        name,
        function,
    )


def _formats(layout):
    parts = layout.removeprefix("paged_kv_").replace("2", "").replace("3", "").split("_")
    return parts[0], parts[1] if len(parts) > 1 else parts[0]


def _shape(dim, fmt, tokens=0):
    b, h, s, d = dim
    if fmt == "bshd":
        return (b, s, h, d)
    if fmt == "sbhd":
        return (s, b, h, d)
    if fmt == "bhsd":
        return dim
    if fmt == "thd":
        return (tokens, h, d)
    raise ValueError(f"Unsupported attention format: {fmt}")


def _raw(tensor, columnwise=False):
    if isinstance(tensor, MXFP8TensorStorage):
        return tensor._columnwise_data if columnwise else tensor._rowwise_data
    return getattr(tensor, "_data", tensor)


def _dtype(tensor):
    fp8_dtype = getattr(tensor, "_fp8_dtype", None)
    if fp8_dtype is not None:
        return "FP8_E4M3" if fp8_dtype == DType.kFloat8E4M3 else "FP8_E5M2"
    return {
        torch.float16: "HALF",
        torch.bfloat16: "BFLOAT16",
        torch.float32: "FLOAT",
        torch.int32: "INT32",
        torch.int64: "INT64",
    }[tensor.dtype]


def _descale(tensor, columnwise=False):
    if isinstance(tensor, MXFP8TensorStorage):
        return tensor._columnwise_scale_inv if columnwise else tensor._rowwise_scale_inv
    return tensor._scale_inv


@dataclass
class _PlanEntry:
    """A supported graph, compiled lazily under the cache lock."""

    graph: object
    tensors: dict
    workspace_size: int | None = None


class _AttentionGraph:
    """Runtime tensors and data-free descriptors for one attention invocation."""

    def __init__(
        self, params, backward=False, *, dimensions=None, device=None, fp8=None, mxfp8=None
    ):
        self.params = params
        self.backward = backward
        self.device = params["q"].device if device is None else device
        self.q_format, self.kv_format = _formats(params["qkv_layout"])
        if dimensions is None:
            dimensions = (
                params["cu_seqlens_q"].numel() - 1,
                params["q"].shape[1] if self.q_format == "bhsd" else params["q"].shape[-2],
                params["k"].shape[1] if self.kv_format == "bhsd" else params["k"].shape[-2],
                params["q"].shape[-1],
                params["v"].shape[-1],
                params["q"].shape[0] if self.q_format == "thd" else 0,
                params["k"].shape[0] if self.kv_format == "thd" else 0,
            )
        self.b, self.h, self.hg, self.dq, self.dv, self.tq, self.tk = dimensions
        capability = torch.cuda.get_device_capability(self.device)
        self.ragged_graph = uses_ragged_attention_graph(10 * capability[0] + capability[1])
        self.ragged_stats = self.q_format == "thd" and self.ragged_graph
        self.sq = bucket(self.tq, 1024, 32768) if self.ragged_stats else params["max_seqlen_q"]
        self.skv = (
            bucket(self.tk, 1024, 32768)
            if self.kv_format == "thd" and self.ragged_graph
            else params["max_seqlen_kv"]
        )
        self.fp8 = hasattr(params["q"], "_fp8_dtype") if fp8 is None else fp8
        self.mxfp8 = isinstance(params["q"], MXFP8TensorStorage) if mxfp8 is None else mxfp8
        self.specs = {}
        self.tensors = {}

    def add(self, name, tensor, dim=None, stride=None, dtype=None, **kwargs):
        """Bind a runtime tensor and record its address-independent descriptor."""
        self.tensors[name] = tensor
        self.specs[name] = _TensorSpec(
            tuple(dim if dim is not None else tensor.shape),
            tuple(stride if stride is not None else tensor.stride()),
            dtype or _dtype(tensor),
            **kwargs,
        )
        return name

    def scalar(self, name, tensor, output=False, by_value=False):
        """Describe a scalar using cuDNN's four-dimensional convention."""
        return self.add(name, tensor, (1, 1, 1, 1), (1, 1, 1, 1), output=output, by_value=by_value)

    def matrix(
        self, name, tensor, fmt, heads, seqlen, cu_padded=None, columnwise=False, output=False
    ):
        """Bind storage using the same matrix descriptors as support queries."""
        data = _raw(tensor, columnwise)
        dim = (self.b, heads, seqlen, tensor.shape[-1])
        offset_name = name + "_offset" if fmt == "thd" else ""
        if offset_name:
            offsets = cu_padded.to(torch.int64) * data.stride(0)
            self.add(offset_name, offsets, (self.b + 1, 1, 1, 1), (1, 1, 1, 1))
        elif name in ("k", "v") and self.params.get("page_table_k") is not None:
            order = (0, 2, 1, 3) if fmt == "bshd" else (0, 1, 2, 3)
            dim = tuple(data.shape[index] for index in order)
        self.tensors[name] = data
        self.specs[name] = attention_matrix_spec(
            dim,
            fmt,
            _dtype(tensor),
            storage_stride=data.stride(),
            output=output,
            ragged_offset=offset_name,
        )
        return name

    def stats(self, name, tensor, output=False):
        """Bind dense or packed softmax statistics."""
        if self.ragged_stats:
            offsets = self.params["cu_seqlens_q_padded"].to(torch.int64) * self.h
            self.add("stats_offset", offsets, (self.b + 1, 1, 1, 1), (1, 1, 1, 1))
        self.tensors[name] = tensor
        self.specs[name] = attention_stats_spec(
            self.b, self.h, self.sq, ragged=self.ragged_stats, output=output
        )
        return name

    def scale(self, name, tensor, matrix_name, fmt, columnwise=False):
        """Bind tensor scales or reordered MXFP8 block scales."""
        if not self.mxfp8:
            return self.scalar(name, tensor)
        self.tensors[name] = tensor
        self.specs[name] = mxfp8_scale_spec(self.specs[matrix_name].dim, fmt, columnwise)
        return name

    def options(self):
        """Return graph attributes that affect support and plan caching."""
        p = self.params
        mask = p["attn_mask_type"]
        causal = "causal" in mask
        bottom_right = p["bottom_right_diagonal"]
        if causal:
            bottom_right = "bottom_right" in mask
        left, right = p["window_size"] or (-1, -1)
        if not causal and (left, right) == (-1, -1):
            bottom_right = False
        bounded_backward = self.backward and not self.fp8 and self.ragged_graph
        return AttentionOptions(
            backward=self.backward,
            fp8=self.fp8,
            mxfp8=self.mxfp8,
            bottom_right=bottom_right,
            left_bound=left + 1 if left != -1 else None,
            right_bound=0 if causal else (right if right != -1 else None),
            padding="padding" in mask,
            alibi=p["attn_bias_type"] == "alibi",
            dropout=p["dropout"],
            deterministic=p.get("deterministic", False),
            paged_max_seqlen_kv=p["max_seqlen_kv"] if p.get("page_table_k") is not None else 0,
            # Ragged sequence dimensions are bucketed total-token capacities.
            # Bound cuDNN's backward scratch buffers by those same capacities.
            max_total_seqlen_q=self.sq if bounded_backward and self.q_format == "thd" else 0,
            max_total_seqlen_kv=self.skv if bounded_backward and self.kv_format == "thd" else 0,
        )

    def description(self):
        """Describe the graph independently of buffer addresses and scalar values."""
        return AttentionGraphSpec.from_dict(self.options(), self.specs)

    def entry(self):
        """Share support-query graphs with execution without compiling on queries."""
        key = (self.device, self.description())
        # Autograd may run backward on another thread. Share device-specific plans;
        # each execution supplies its thread-local handle on the current stream.
        with _cache_lock:
            entry = _plans.get(key)
            if entry is None:
                _event(self, "MISS", key)
                entry = self.build()
                _plans[key] = entry
                _event(self, "CACHE_GRAPH")
            else:
                _event(self, "HIT", key)
        return entry

    def execute(self):
        """Compile once, then bind this invocation's runtime tensor addresses."""
        entry = self.entry()
        with _cache_lock:
            if entry.workspace_size is None:
                entry.workspace_size = compile_graph(
                    _import_cudnn_frontend(),
                    entry.graph,
                    lambda name, function: _build_stage(self, name, function),
                )
                _event(self, "BUILD_PLANS")
        _execute_cudnn_graph(
            entry.graph,
            {entry.tensors[n]: t for n, t in self.tensors.items()},
            entry.workspace_size,
            self.device,
        )
        _event(self, "EXECUTE")

    def build(self):
        """Construct and check the common graph description without compiling it."""
        cudnn = _import_cudnn_frontend()
        graph = _build_cudnn_pygraph(getattr(cudnn.data_type, self.specs["q"].dtype), self.device)
        built = build_attention_graph(cudnn, graph, self.description())
        _event(self, "CREATE_GRAPH")
        check_graph_support(cudnn, graph, lambda name, function: _build_stage(self, name, function))
        return _PlanEntry(graph, built.tensors)


def _prepare_inputs(attn):
    p = attn.params
    for name, fmt, h, s in (
        ("q", attn.q_format, attn.h, attn.sq),
        ("k", attn.kv_format, attn.hg, attn.skv),
        ("v", attn.kv_format, attn.hg, attn.skv),
    ):
        cu = p["cu_seqlens_q_padded"] if name == "q" else p["cu_seqlens_kv_padded"]
        col = attn.mxfp8 and name == "v" and not attn.backward
        attn.matrix(name, p[name], fmt, h, s, cu, col)
        if attn.fp8:
            scale_fmt = p.get("qkv_scale_inv_format") or fmt
            attn.scale("descale_" + name, _descale(p[name], col), name, scale_fmt, col)
    attn.scalar("attn_scale", torch.tensor([p["attn_scale"]], dtype=torch.float32), by_value=True)
    if "padding" in p["attn_mask_type"]:
        for name, cu in (("seq_q", p["cu_seqlens_q"]), ("seq_kv", p["cu_seqlens_kv"])):
            attn.add(name, cu[1:] - cu[:-1], (attn.b, 1, 1, 1), (1, 1, 1, 1))
    if p["attn_bias_type"] not in ("no_bias", "alibi"):
        attn.add("bias", p["attn_bias"])
    if p["softmax_type"] != "vanilla":
        attn.add("sink", p["softmax_offset"])
    if p["dropout"]:
        rng = p["rng_state"]
        attn.scalar("seed", rng[:1])
        attn.scalar("offset", rng[1:])
    if p.get("page_table_k") is not None:
        for name in ("page_table_k", "page_table_v"):
            table = p[name]
            attn.add(
                name,
                table,
                (attn.b, 1, table.shape[-1], 1),
                (table.shape[-1], table.shape[-1], 1, 1),
            )


def _allocate(shape, dtype, device, quantizer, data=None):
    """Allocate an output, preserving delayed-scaling metadata and packed views."""
    delayed = isinstance(quantizer, Float8Quantizer)
    if data is None:
        data = torch.empty(shape, dtype=torch.uint8 if delayed else dtype, device=device)
    if delayed:
        return quantizer.create_tensor_from_data(
            data, fake_dtype=dtype, internal=quantizer.internal
        )
    return data


def _output_shape(attn, name, fmt):
    is_q = name == "q"
    return _shape(
        (
            attn.b,
            attn.h if is_q else attn.hg,
            attn.params["max_seqlen_q"] if is_q else attn.params["max_seqlen_kv"],
            attn.dq if name in ("q", "k") else attn.dv,
        ),
        fmt,
        attn.tq if is_q else attn.tk,
    )


def _allocate_gradients(attn):
    p = attn.params
    qfmt, kfmt = _formats(p["dqkv_layout"])
    shapes = [_output_shape(attn, n, qfmt if n == "q" else kfmt) for n in ("q", "k", "v")]
    delayed = isinstance(p["dqkv_quantizer"], Float8Quantizer)
    dtype = torch.uint8 if delayed else p["fake_dtype"]
    layout = p["dqkv_layout"]
    if "3" in layout:
        axis = -2 if "h3d" in layout else -3
        packed_shape = list(shapes[0])
        packed_shape.insert(len(packed_shape) + axis + 1, 3)
        data = torch.empty(packed_shape, dtype=dtype, device=attn.device).unbind(axis)
    elif "2" in layout:
        axis = -2 if "h2d" in layout else -3
        packed_shape = list(shapes[1])
        packed_shape.insert(len(packed_shape) + axis + 1, 2)
        kv = torch.empty(packed_shape, dtype=dtype, device=attn.device).unbind(axis)
        data = (torch.empty(shapes[0], dtype=dtype, device=attn.device), *kv)
    else:
        data = [torch.empty(s, dtype=dtype, device=attn.device) for s in shapes]
    return [
        _allocate(s, p["fake_dtype"], attn.device, p["dqkv_quantizer"], t)
        for s, t in zip(shapes, data)
    ]


def _normalize(params):
    p = dict(params)
    p["cu_seqlens_q_padded"] = p.get("cu_seqlens_q_padded")
    if p["cu_seqlens_q_padded"] is None:
        p["cu_seqlens_q_padded"] = p["cu_seqlens_q"]
    p["cu_seqlens_kv_padded"] = p.get("cu_seqlens_kv_padded")
    if p["cu_seqlens_kv_padded"] is None:
        p["cu_seqlens_kv_padded"] = p["cu_seqlens_kv"]
    return p


def _query_graph(config, backward):
    """Describe a support query without allocating tensors or touching RNG state."""
    from .fused_attn import QKVLayout, QKVFormat, AttnMaskType, AttnBiasType, SoftmaxType

    if (
        backward
        and DType.cast(config.qkv_dtype) in (DType.kFloat8E4M3, DType.kFloat8E5M2)
        and not int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
    ):
        # FusedAttnFunc saves model-precision Q/K/V and O for this backward.
        # The gradient layout records the inputs' layout before FP8 quantization.
        dtype = config.nominal_dtype
        if dtype is None:
            dtype = next(
                (
                    t
                    for t in (config.dqkv_dtype, config.o_dtype)
                    if t in (DType.kFloat16, DType.kBFloat16)
                ),
                DType.kBFloat16,
            )
        config = replace(
            config,
            qkv_dtype=dtype,
            o_dtype=dtype,
            do_dtype=dtype,
            dqkv_dtype=dtype,
            qkv_layout=(
                config.dqkv_layout
                if config.dqkv_layout != tex.NVTE_QKV_Layout.NVTE_QKV_Layout_NOT_SET
                else config.qkv_layout
            ),
        )

    def name(mapping, value):
        return next(k for k, v in mapping.items() if v == value)

    layout = name(QKVLayout, config.qkv_layout)
    grad_layout = (
        name(QKVLayout, config.dqkv_layout)
        if (config.dqkv_layout != tex.NVTE_QKV_Layout.NVTE_QKV_Layout_NOT_SET)
        else layout
    )
    types = {
        DType.kFloat16: "HALF",
        DType.kBFloat16: "BFLOAT16",
        DType.kFloat32: "FLOAT",
        DType.kFloat8E4M3: "FP8_E4M3",
        DType.kFloat8E5M2: "FP8_E5M2",
    }
    fp8 = types[DType.cast(config.qkv_dtype)].startswith("FP8")
    mxfp8 = fp8 and config.scaling_mode == tex.NVTEScalingMode.NVTE_MXFP8_1D_SCALING
    params = {
        "attn_mask_type": name(AttnMaskType, config.attn_mask_type),
        "attn_bias_type": name(AttnBiasType, config.bias_type),
        "softmax_type": name(SoftmaxType, config.softmax_type),
        "bottom_right_diagonal": config.bottom_right_diagonal,
        "window_size": (config.window_size_left, config.window_size_right),
        "dropout": config.dropout if config.is_training else 0.0,
        "deterministic": config.deterministic if backward else False,
        "return_max_logit": config.return_max_logit if not backward else False,
        "max_seqlen_kv": config.max_seqlen_kv,
        "page_table_k": True if layout.startswith("paged_kv_") else None,
    }
    params.update(qkv_layout=layout, max_seqlen_q=config.max_seqlen_q)
    attn = _AttentionGraph(
        params,
        backward,
        dimensions=(
            config.batch_size,
            config.num_attn_heads,
            config.num_gqa_groups,
            config.head_dim_qk,
            config.head_dim_v,
            config.num_tokens_q,
            config.num_tokens_kv,
        ),
        device=torch.device("cuda", torch.cuda.current_device()),
        fp8=fp8,
        mxfp8=mxfp8,
    )

    def spec(n, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), dtype="FLOAT", **kwargs):
        attn.specs[n] = _TensorSpec(tuple(dim), tuple(stride), dtype, **kwargs)

    def matrix(n, fmt, h, s, d, dtype, packed="", output=False):
        dim = (attn.b, h, s, d)
        if n in ("k", "v") and layout.startswith("paged_kv_"):
            dim = (getattr(config, "num_pages_" + n), h, getattr(config, "page_size_" + n), d)
        offset = n + "_offset" if fmt == "thd" else ""
        if offset:
            spec(offset, (attn.b + 1, 1, 1, 1), dtype="INT64")
        attn.specs[n] = attention_matrix_spec(
            dim, fmt, dtype, packed=packed, output=output, ragged_offset=offset
        )

    def stats(n, output=False):
        if attn.ragged_stats:
            spec("stats_offset", (attn.b + 1, 1, 1, 1), dtype="INT64")
        attn.specs[n] = attention_stats_spec(
            attn.b, attn.h, attn.sq, ragged=attn.ragged_stats, output=output
        )

    def scale(n, matrix_name, fmt, columnwise=False):
        if not attn.mxfp8:
            spec(n)
        else:
            attn.specs[n] = mxfp8_scale_spec(attn.specs[matrix_name].dim, fmt, columnwise)

    parts = layout.removeprefix("paged_kv_").split("_")
    for n, fmt, h, s, d, part in (
        ("q", attn.q_format, attn.h, attn.sq, attn.dq, parts[0]),
        ("k", attn.kv_format, attn.hg, attn.skv, attn.dq, parts[1] if len(parts) > 1 else parts[0]),
        ("v", attn.kv_format, attn.hg, attn.skv, attn.dv, parts[-1]),
    ):
        matrix(n, fmt, h, s, d, types[DType.cast(config.qkv_dtype)], part)
        if attn.fp8:
            scale_fmt = name(QKVFormat, config.qkv_scale_inv_format) or fmt
            scale("descale_" + n, n, scale_fmt, attn.mxfp8 and n == "v" and not backward)
    spec("attn_scale", by_value=True)
    if "padding" in attn.params["attn_mask_type"]:
        for n in ("seq_q", "seq_kv"):
            spec(n, (attn.b, 1, 1, 1), dtype="INT32")
    if attn.params["dropout"]:
        spec("seed", dtype="INT64")
        spec("offset", dtype="INT64")
    if attn.params["attn_bias_type"] not in ("no_bias", "alibi"):
        dim = (
            config.bias_batch_size,
            config.bias_num_heads,
            config.bias_seqlen_q,
            config.bias_seqlen_kv,
        )
        b, h, sq, skv = dim
        stride = (h * sq * skv, sq * skv, skv, 1)
        spec("bias", dim, stride, types[DType.cast(config.qkv_dtype)])
        if backward and (b, h, sq) != (1, 1, 1):
            spec("dbias", dim, stride, types[DType.cast(config.qkv_dtype)], output=True)
    if attn.params["softmax_type"] != "vanilla":
        spec("sink", (1, attn.h, 1, 1), (attn.h, 1, 1, 1))
        if backward:
            spec("dsink", (1, attn.h, 1, 1), (attn.h, 1, 1, 1), output=True)
    if layout.startswith("paged_kv_"):
        for n in ("k", "v"):
            pages = getattr(config, "max_pages_per_seq_" + n)
            spec("page_table_" + n, (attn.b, 1, pages, 1), (pages, pages, 1, 1), "INT32")
    ofmt = name(QKVFormat, config.o_format) or attn.q_format
    matrix(
        "o", ofmt, attn.h, attn.sq, attn.dv, types[DType.cast(config.o_dtype)], output=not backward
    )
    stats("stats", output=not backward)
    if not backward and config.return_max_logit:
        stats("max", output=True)
    if backward:
        dofmt = name(QKVFormat, config.do_format) or ofmt
        matrix("do", dofmt, attn.h, attn.sq, attn.dv, types[DType.cast(config.do_dtype)])
        qfmt, kvfmt = _formats(grad_layout)
        parts = grad_layout.split("_")
        for n, fmt, h, s, d, part in (
            ("dq", qfmt, attn.h, attn.sq, attn.dq, parts[0]),
            ("dk", kvfmt, attn.hg, attn.skv, attn.dq, parts[1] if len(parts) > 1 else parts[0]),
            ("dv", kvfmt, attn.hg, attn.skv, attn.dv, parts[-1]),
        ):
            matrix(n, fmt, h, s, d, types[DType.cast(config.dqkv_dtype)], part, output=True)
        if attn.fp8:
            do_scale_fmt = name(QKVFormat, config.do_scale_inv_format) or dofmt
            scale("descale_do", "do", do_scale_fmt)
            if attn.mxfp8:
                matrix("do_f16", dofmt, attn.h, attn.sq, attn.dv, types[DType.cast(config.o_dtype)])
                for n, fmt, h, s, d, dtype in (
                    ("q_t", attn.q_format, attn.h, attn.sq, attn.dq, config.qkv_dtype),
                    ("k_t", attn.kv_format, attn.hg, attn.skv, attn.dq, config.qkv_dtype),
                    ("do_t", dofmt, attn.h, attn.sq, attn.dv, config.do_dtype),
                ):
                    matrix(n, fmt, h, s, d, types[DType.cast(dtype)])
                    sfmt = (
                        do_scale_fmt
                        if n == "do_t"
                        else name(QKVFormat, config.qkv_scale_inv_format) or fmt
                    )
                    scale("descale_" + n, n, sfmt, True)
            else:
                for n in (
                    "descale_o",
                    "descale_s",
                    "descale_dp",
                    "scale_s",
                    "scale_dp",
                    "scale_dq",
                    "scale_dk",
                    "scale_dv",
                ):
                    spec(n)
                for n in ("amax_dq", "amax_dk", "amax_dv", "amax_dp"):
                    spec(n, output=True)
    elif attn.fp8 and not attn.mxfp8:
        for n in ("descale_s", "scale_s", "scale_o"):
            spec(n)
        for n in ("amax_s", "amax_o"):
            spec(n, output=True)
    return attn


def _support_reason(config, attn, version, capability):
    """Preserve TE's hardware, recipe, and known cuDNN defect exclusions."""
    p = attn.params
    ragged = attn.q_format == "thd" or attn.kv_format == "thd"
    padding = "padding" in p["attn_mask_type"]
    if ragged and not padding:
        return "THD format requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask."
    if (attn.q_format == "thd" and not attn.tq) or (attn.kv_format == "thd" and not attn.tk):
        return "THD format requires num_tokens_q / num_tokens_kv to be set for the ragged inputs."
    if p["page_table_k"] is not None:
        if not padding:
            return "Paged KV requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask."
        if not all(
            getattr(config, field + "_" + name)
            for field in ("num_pages", "page_size", "max_pages_per_seq")
            for name in ("k", "v")
        ):
            return "Paged KV requires num_pages, page_size and max_pages_per_seq for both K and V."
    if not attn.fp8:
        if attn.q_format == attn.kv_format == "thd" and capability < (9, 0):
            return "F16/BF16 fused attention with THD format requires sm90 or later."
        affected_cuda_graph = version <= 91500 and config.cuda_graph and config.max_seqlen_kv % 128
        if (
            affected_cuda_graph
            and config.is_training
            and attn.q_format == attn.kv_format
            and attn.q_format in ("bshd", "sbhd")
            and not padding
        ):
            return "Known cuDNN <= 9.15 issue with CUDA graph. Please upgrade cuDNN."
        if (
            config.is_training
            and attn.ragged_stats
            and p["softmax_type"] == "learnable"
            and version < 92600
            and attn.dv not in (64, 128, 256)
        ):
            return (
                "Known cuDNN < 9.26.0 issue with THD learnable softmax backward. Please upgrade"
                " cuDNN."
            )
        return ""
    if config.return_max_logit:
        return "FP8 fused attention does not support return_max_logit=True."
    if attn.q_format != attn.kv_format:
        return "FP8 fused attention supports BSHD/SBHD/BHSD/THD formats."
    if config.is_training and config.dqkv_layout != tex.NVTE_QKV_Layout.NVTE_QKV_Layout_NOT_SET:
        from .fused_attn import QKVLayout

        grad_layout = next(k for k, v in QKVLayout.items() if v == config.dqkv_layout)
        qfmt, kvfmt = _formats(grad_layout)
        if qfmt != kvfmt:
            return "FP8 fused attention supports BSHD/SBHD/BHSD/THD gradient formats."
    if attn.q_format == "thd":
        if version < 92300:
            return "FP8 fused attention with THD format requires cuDNN 9.23.0 or later!"
        if config.is_training and capability < (10, 0):
            return "FP8 fused attention with THD format supports backward on sm100+ only!"
        if config.is_training and p["softmax_type"] != "vanilla" and version < 92600:
            return (
                "FP8 THD attention with a sink token requires cuDNN 9.26.0 or later for backward!"
            )
        if capability >= (10, 0) and max(attn.dq, attn.dv) > 128:
            return "FP8 THD attention supports head dimensions up to 128 on sm100+ only!"
    if p["attn_bias_type"] != "no_bias":
        return "FP8 fused attention does not support attention bias."
    half_types = (DType.kFloat16, DType.kBFloat16)
    fp8_types = (DType.kFloat8E4M3, DType.kFloat8E5M2)
    allowed_types = half_types if attn.mxfp8 else half_types + fp8_types
    tensor_scaling = config.scaling_mode == tex.NVTEScalingMode.NVTE_DELAYED_TENSOR_SCALING
    if (
        not (tensor_scaling or attn.mxfp8)
        or DType.cast(config.o_dtype) not in allowed_types
        or (config.is_training and DType.cast(config.dqkv_dtype) not in allowed_types)
    ):
        return (
            "FP8 fused attention only supports FP8DelayedScaling, FP8CurrentScaling, or MXFP8"
            " recipes!"
        )
    if attn.mxfp8 and version < 92100:
        return "MXFP8 fused attention requires cuDNN 9.21.0 or later!"
    return ""


def get_fused_attn_backend(config):
    """Select an attention backend by validating the Python frontend graphs."""
    from .fused_attn import FusedAttnBackend

    cudnn = _import_cudnn_frontend()
    if config.bias_type == tex.NVTE_Bias_Type.NVTE_PRE_SCALE_BIAS:
        return FusedAttnBackend.No_Backend, "Fused attention does not support pre-scale bias."
    if DType.cast(config.qkv_dtype) not in (
        DType.kFloat16,
        DType.kBFloat16,
        DType.kFloat8E4M3,
        DType.kFloat8E5M2,
    ):
        return FusedAttnBackend.No_Backend, f"Unsupported QKV dtype qkv_dtype={config.qkv_dtype}."
    try:
        fwd = _query_graph(config, False)
        reason = _support_reason(
            config, fwd, cudnn.backend_version(), torch.cuda.get_device_capability()
        )
        if reason:
            return FusedAttnBackend.No_Backend, reason
        fwd.entry()
        if config.is_training:
            _query_graph(config, True).entry()
        return FusedAttnBackend.FP8 if fwd.fp8 else FusedAttnBackend.F16_arbitrary_seqlen, ""
    except (cudnn.cudnnGraphNotSupportedError, ValueError) as exc:
        return FusedAttnBackend.No_Backend, str(exc)


def fused_attn_fwd(params):
    """Execute forward; preserve the extension's output and auxiliary tensor order."""
    p = _normalize(params)
    if not p["is_training"]:
        p["dropout"] = 0.0
    with torch.cuda.device(p["q"].device):
        p["rng_state"] = tex.fused_attn_rng_state(p["rng_gen"], p["rng_elts_per_thread"])
        attn = _AttentionGraph(p)
        _prepare_inputs(attn)
        out_shape = list(_output_shape(attn, "q", p["o_format"]))
        out_shape[-1] = attn.dv
        out = _allocate(out_shape, p["fake_dtype"], attn.device, p["o_quantizer"])
        if p["o_format"] == "thd":
            _raw(out).zero_()
        attn.matrix("o", out, p["o_format"], attn.h, attn.sq, p["cu_seqlens_q_padded"], output=True)
        stats_shape = (attn.tq, attn.h, 1) if attn.ragged_stats else (attn.b, attn.h, attn.sq, 1)
        stats = torch.empty(stats_shape, dtype=torch.float32, device=attn.device)
        attn.stats("stats", stats, output=True)
        aux_ctx_tensors = [stats, p["rng_state"]]
        maximum = None
        if p["return_max_logit"]:
            maximum = torch.empty_like(stats)
            attn.stats("max", maximum, output=True)
        if p["attn_bias_type"] not in ("no_bias", "alibi"):
            aux_ctx_tensors.append(p["attn_bias"])
        if p["softmax_type"] != "vanilla":
            aux_ctx_tensors.append(p["softmax_offset"])
        if attn.fp8 and not attn.mxfp8:
            sq = p["s_quantizer"]
            attn.scalar("scale_s", sq.scale)
            attn.scalar("descale_s", sq.scale.reciprocal())
            attn.scalar("amax_s", sq.amax, output=True)
            if isinstance(p["o_quantizer"], Float8Quantizer):
                scale_o, amax_o = p["o_quantizer"].scale, p["o_quantizer"].amax
            else:
                scale_o = torch.ones(1, dtype=torch.float32, device=attn.device)
                amax_o = torch.empty_like(scale_o)
            attn.scalar("scale_o", scale_o)
            attn.scalar("amax_o", amax_o, output=True)
        attn.execute()
        return out, aux_ctx_tensors, maximum


def fused_attn_bwd(params):
    """Execute backward, including FP8/MXFP8 and shared packed gradient storage."""
    p = _normalize(params)
    aux = iter(p["aux_ctx_tensors"])
    stats, p["rng_state"] = next(aux), next(aux)
    p["attn_bias"] = next(aux) if p["attn_bias_type"] not in ("no_bias", "alibi") else None
    p["softmax_offset"] = next(aux) if p["softmax_type"] != "vanilla" else None
    with torch.cuda.device(p["q"].device):
        attn = _AttentionGraph(p, backward=True)
        _prepare_inputs(attn)
        attn.matrix("o", p["o"], p["o_format"], attn.h, attn.sq, p["cu_seqlens_q_padded"])
        attn.matrix("do", p["d_o"], p["do_format"], attn.h, attn.sq, p["cu_seqlens_q_padded"])
        attn.stats("stats", stats)
        dq, dk, dv = _allocate_gradients(attn)
        qfmt, kfmt = _formats(p["dqkv_layout"])
        for name, t, fmt, h, s, cu in (
            ("dq", dq, qfmt, attn.h, attn.sq, p["cu_seqlens_q_padded"]),
            ("dk", dk, kfmt, attn.hg, attn.skv, p["cu_seqlens_kv_padded"]),
            ("dv", dv, kfmt, attn.hg, attn.skv, p["cu_seqlens_kv_padded"]),
        ):
            if fmt == "thd":
                _raw(t).zero_()
            attn.matrix(name, t, fmt, h, s, cu, output=True)
        dbias = torch.empty(0, device=attn.device, dtype=p["fake_dtype"])
        if p["attn_bias"] is not None:
            dbias = torch.empty_like(p["attn_bias"])
            if p["attn_bias"].shape[-2] != 1 or p["attn_bias"].shape[:2] != (1, 1):
                attn.add("dbias", dbias, output=True)
        dsoftmax = torch.empty(0, device=attn.device, dtype=torch.float32)
        if p["softmax_type"] != "vanilla":
            dsoftmax = torch.empty_like(p["softmax_offset"])
            attn.add("dsink", dsoftmax, output=True)
        if attn.fp8:
            do_scale_fmt = p.get("do_scale_inv_format") or p["do_format"]
            attn.scale("descale_do", _descale(p["d_o"]), "do", do_scale_fmt)
            if attn.mxfp8:
                do_f16 = next(aux)
                attn.matrix(
                    "do_f16", do_f16, p["do_format"], attn.h, attn.sq, p["cu_seqlens_q_padded"]
                )
                for name, t, fmt, h, s, cu in (
                    ("q_t", p["q"], attn.q_format, attn.h, attn.sq, p["cu_seqlens_q_padded"]),
                    ("k_t", p["k"], attn.kv_format, attn.hg, attn.skv, p["cu_seqlens_kv_padded"]),
                    ("do_t", p["d_o"], p["do_format"], attn.h, attn.sq, p["cu_seqlens_q_padded"]),
                ):
                    attn.matrix(name, t, fmt, h, s, cu, columnwise=True)
                    scale_fmt = (
                        do_scale_fmt if name == "do_t" else p.get("qkv_scale_inv_format") or fmt
                    )
                    attn.scale("descale_" + name, _descale(t, True), name, scale_fmt, True)
            else:
                attn.scalar(
                    "descale_o",
                    (
                        _descale(p["o"])
                        if hasattr(p["o"], "_fp8_dtype")
                        else torch.ones(1, dtype=torch.float32, device=attn.device)
                    ),
                )
                for short, quantizer in (("s", p["s_quantizer"]), ("dp", p["dp_quantizer"])):
                    attn.scalar("scale_" + short, quantizer.scale)
                    attn.scalar("descale_" + short, quantizer.scale.reciprocal())
                attn.scalar("amax_dp", p["dp_quantizer"].amax, output=True)
                amax_grads = []
                for name in ("dq", "dk", "dv"):
                    quantizer = p["dqkv_quantizer"]
                    scale = (
                        quantizer.scale
                        if isinstance(quantizer, Float8Quantizer)
                        else (torch.ones(1, dtype=torch.float32, device=attn.device))
                    )
                    attn.scalar("scale_" + name, scale)
                    amax = torch.empty(1, dtype=torch.float32, device=attn.device)
                    attn.scalar("amax_" + name, amax, output=True)
                    amax_grads.append(amax)
        attn.execute()
        if attn.fp8 and not attn.mxfp8 and isinstance(p["dqkv_quantizer"], Float8Quantizer):
            p["dqkv_quantizer"].amax.copy_(torch.stack(amax_grads).amax().reshape(1))
        return dq, dk, dv, dbias, dsoftmax
