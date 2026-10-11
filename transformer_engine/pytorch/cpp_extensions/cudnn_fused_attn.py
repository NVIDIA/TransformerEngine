# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN attention graphs constructed and executed through the Python frontend.

The extension is used only to obtain capture-safe Philox state. Tensor allocation,
graph construction, plan caching, and execution do not use the common attention API.
"""

from dataclasses import dataclass
import atexit
from collections import Counter
import os
import sys
import threading
import time

import torch
import transformer_engine_torch as tex

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
_diagnostics = Counter()
_build_times = Counter()


def _debug_enabled():
    setting = os.getenv("NVTE_FUSED_ATTN_CACHE_DEBUG", "0")
    level, _, ranks = setting.partition(":")
    rank = next(
        (
            os.environ[n]
            for n in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID")
            if n in os.environ
        ),
        None,
    )
    enabled = rank is None or ranks == "all" or rank in (ranks or "0").split(",")
    return int(level or 0) if enabled else 0


def _event(attn, event, key=None):
    level = _debug_enabled()
    if level == 0:
        return
    site = ("fp8" if attn.fp8 else "f16", "bwd" if attn.backward else "fwd")
    _diagnostics[(*site, event)] += 1
    if level >= 2:
        print(
            f"[FUSED-ATTN-CACHE] tid={threading.get_ident()} dev={attn.device.index} | "
            f"{site[0]} {site[1]} {event}"
            + (f" | {key!r}" if key is not None else ""),
            file=sys.stderr,
            flush=True,
        )


def _build_stage(attn, name, function):
    if not _debug_enabled():
        return function()
    start = time.perf_counter()
    try:
        return function()
    finally:
        site = ("fp8" if attn.fp8 else "f16", "bwd" if attn.backward else "fwd", name)
        _build_times[(*site, "seconds")] += time.perf_counter() - start
        _build_times[(*site, "calls")] += 1


def _print_diagnostics():
    if not _diagnostics:
        return
    print("[FUSED-ATTN-CACHE] Python frontend summary", file=sys.stderr)
    for (backend, direction, event), count in sorted(_diagnostics.items()):
        print(f"  {backend} {direction} {event}={count}", file=sys.stderr)
    for backend, direction, stage, kind in sorted(_build_times):
        if kind == "calls":
            calls = _build_times[(backend, direction, stage, kind)]
            seconds = _build_times[(backend, direction, stage, "seconds")]
            print(
                f"  {backend} {direction} {stage}: {calls} calls, "
                f"{1000 * seconds / calls:.3f} ms average",
                file=sys.stderr,
            )


atexit.register(_print_diagnostics)


def _formats(layout):
    parts = layout.removeprefix("paged_kv_").replace("2", "").replace("3", "").split("_")
    return parts[0], parts[1] if len(parts) > 1 else parts[0]


def _strides(dim, fmt):
    b, h, s, d = dim
    if fmt in ("bshd", "thd"):
        return (s * h * d, d, h * d, 1)
    if fmt == "sbhd":
        return (h * d, d, b * h * d, 1)
    if fmt == "bhsd":
        return (h * s * d, s * d, d, 1)
    raise ValueError(f"Unsupported attention format: {fmt}")


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


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _token_bucket(tokens):
    if tokens == 0:
        return 0
    if tokens <= 1024:
        return 1024
    if tokens <= 32768:
        return 1 << (tokens - 1).bit_length()
    return _round_up(tokens, 32768)


@dataclass(frozen=True)
class _TensorSpec:
    dim: tuple
    stride: tuple
    dtype: str
    output: bool = False
    ragged_offset: str = ""
    reordered: bool = False
    by_value: bool = False


class _AttentionGraph:
    """Runtime tensors and data-free descriptors for one attention invocation."""

    def __init__(self, params, backward=False):
        self.params = params
        self.backward = backward
        self.device = params["q"].device
        self.q_format, self.kv_format = _formats(params["qkv_layout"])
        self.b = params["cu_seqlens_q"].numel() - 1
        self.h = params["q"].shape[1] if self.q_format == "bhsd" else params["q"].shape[-2]
        self.hg = params["k"].shape[1] if self.kv_format == "bhsd" else params["k"].shape[-2]
        self.dq = params["q"].shape[-1]
        self.dv = params["v"].shape[-1]
        self.tq = params["q"].shape[0] if self.q_format == "thd" else 0
        self.tk = params["k"].shape[0] if self.kv_format == "thd" else 0
        capability = torch.cuda.get_device_capability(self.device)
        ragged_graph = capability >= (9, 0) and capability != (12, 0)
        self.ragged_stats = self.q_format == "thd" and ragged_graph
        self.sq = _token_bucket(self.tq) if self.ragged_stats else params["max_seqlen_q"]
        self.skv = (
            _token_bucket(self.tk)
            if self.kv_format == "thd" and ragged_graph
            else params["max_seqlen_kv"]
        )
        self.fp8 = hasattr(params["q"], "_fp8_dtype")
        self.mxfp8 = isinstance(params["q"], MXFP8TensorStorage)
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
        """Describe an attention matrix in logical BHSD order, retaining storage strides."""
        data = _raw(tensor, columnwise)
        dim = (self.b, heads, seqlen, tensor.shape[-1])
        if fmt == "bshd":
            order = (0, 2, 1, 3)
        elif fmt == "sbhd":
            order = (1, 2, 0, 3)
        else:
            order = (0, 1, 2, 3)
        offset_name = ""
        if fmt == "thd":
            token_stride, head_stride, last_stride = data.stride()
            stride = (seqlen * token_stride, head_stride, token_stride, last_stride)
            offset_name = name + "_offset"
            offsets = cu_padded.to(torch.int64) * token_stride
            self.add(offset_name, offsets, (self.b + 1, 1, 1, 1), (1, 1, 1, 1))
        elif data.ndim == 4:
            stride = tuple(data.stride(i) for i in order)
            # Paged K/V have a page axis rather than a batch axis.
            if name in ("k", "v") and self.params.get("page_table_k") is not None:
                dim = tuple(data.shape[i] for i in order)
        else:
            stride = _strides(dim, fmt)
        return self.add(
            name, data, dim, stride, _dtype(tensor), output=output, ragged_offset=offset_name
        )

    def stats(self, name, tensor, output=False):
        """Describe softmax statistics, including offsets for packed THD storage."""
        dim = (self.b, self.h, self.sq, 1)
        if self.ragged_stats:
            offsets = self.params["cu_seqlens_q_padded"].to(torch.int64) * self.h
            self.add("stats_offset", offsets, (self.b + 1, 1, 1, 1), (1, 1, 1, 1))
            return self.add(
                name,
                tensor,
                dim,
                (self.h * self.sq, 1, self.h, 1),
                output=output,
                ragged_offset="stats_offset",
            )
        return self.add(name, tensor, dim, (self.h * self.sq, self.sq, 1, 1), output=output)

    def scale(self, name, tensor, matrix_name, fmt, columnwise=False):
        """Describe per-tensor FP8 scales or reordered MXFP8 block scales."""
        if not self.mxfp8:
            return self.scalar(name, tensor)
        b, h, s, d = self.specs[matrix_name].dim
        if columnwise:
            dim = (b, h, _round_up((s + 31) // 32, 4), _round_up(d, 128))
        else:
            dim = (b, h, _round_up(s, 128), _round_up((d + 31) // 32, 4))
        return self.add(name, tensor, dim, _strides(dim, fmt), "FP8_E8M0", reordered=True)

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
        return (
            mask,
            p["attn_bias_type"],
            p["softmax_type"],
            left + 1 if left != -1 else None,
            0 if causal else (right if right != -1 else None),
            bottom_right,
            p["dropout"],
            p.get("deterministic", False),
            p.get("return_max_logit", False),
            self.fp8,
            self.mxfp8,
            self.params["max_seqlen_kv"] if self.params.get("page_table_k") is not None else 0,
        )

    def entry(self):
        """Share support-query graphs with execution without compiling on queries."""
        key = (self.device, self.backward, self.options(), tuple(sorted(self.specs.items())))
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
        """Look up the graph by metadata, then bind this call's tensor addresses."""
        entry = self.entry()
        with _cache_lock:
            if entry[2] is None:
                cudnn = _import_cudnn_frontend()
                _build_stage(
                    self,
                    "build_plans",
                    lambda: entry[0].build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE),
                )
                entry[2] = max(entry[0].get_workspace_size(), 1)
                _event(self, "BUILD_PLANS")
        graph, tensors, workspace = entry
        _execute_cudnn_graph(
            graph, {tensors[n]: t for n, t in self.tensors.items()}, workspace, self.device
        )
        _event(self, "EXECUTE")

    def build(self):
        """Construct and check a frontend graph without compiling its execution plan."""
        cudnn = _import_cudnn_frontend()
        graph = _build_cudnn_pygraph(getattr(cudnn.data_type, self.specs["q"].dtype), self.device)
        tensors = {}

        def tensor(name):
            if name not in tensors:
                spec = self.specs[name]
                desc = graph.tensor(
                    name=name,
                    dim=spec.dim,
                    stride=spec.stride,
                    data_type=getattr(cudnn.data_type, spec.dtype),
                    is_pass_by_value=spec.by_value,
                )
                if spec.ragged_offset:
                    desc.set_ragged_offset(tensor(spec.ragged_offset))
                if spec.reordered:
                    desc.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
                if spec.output:
                    desc.set_output(True)
                tensors[name] = desc
            return tensors[name]

        for name, spec in self.specs.items():
            if not spec.output:
                tensor(name)

        mask, bias, softmax, left, right, bottom_right, dropout, deterministic, _, _, _, pages = (
            self.options()
        )
        kwargs = {
            "attn_scale": tensor("attn_scale"),
            "diagonal_alignment": (
                cudnn.diagonal_alignment.BOTTOM_RIGHT
                if bottom_right
                else cudnn.diagonal_alignment.TOP_LEFT
            ),
        }
        bound_names = (
            ("left_bound", "right_bound")
            if (self.fp8 and (not self.mxfp8 or self.backward))
            else ("diagonal_band_left_bound", "diagonal_band_right_bound")
        )
        kwargs.update(zip(bound_names, (left, right)))
        if "padding" in mask:
            kwargs.update(
                use_padding_mask=True, seq_len_q=tensor("seq_q"), seq_len_kv=tensor("seq_kv")
            )
        if bias == "alibi":
            kwargs["use_alibi_mask"] = True
        elif bias != "no_bias":
            kwargs["bias"] = tensor("bias")
            if self.backward and "dbias" in self.specs:
                kwargs["dBias"] = tensor("dbias")
        if dropout:
            kwargs["dropout"] = (dropout, tensor("seed"), tensor("offset"))
        if softmax != "vanilla":
            kwargs["sink_token"] = tensor("softmax_offset")
            if self.backward:
                kwargs["dSink_token"] = tensor("dsoftmax_offset")
        if self.backward:
            kwargs["use_deterministic_algorithm"] = deterministic
        else:
            kwargs["generate_stats"] = True
            if "max" in self.specs:
                kwargs["score_max"] = tensor("max")
            if pages:
                kwargs.update(
                    paged_attention_k_table=tensor("page_table_k"),
                    paged_attention_v_table=tensor("page_table_v"),
                    paged_attention_max_seq_len_kv=pages,
                )

        q, k, v = (tensor(n) for n in ("q", "k", "v"))
        if not self.fp8:
            if self.backward:
                outputs = graph.sdpa_backward(
                    q, k, v, tensor("o"), tensor("do"), tensor("stats"), **kwargs
                )
                names = ("dq", "dk", "dv")
            else:
                outputs = graph.sdpa(q, k, v, **kwargs)
                names = ("o", "stats")
        elif not self.backward:
            scale_args = [tensor(n) for n in ("descale_q", "descale_k", "descale_v")]
            if self.mxfp8:
                outputs = graph.sdpa_mxfp8(q, k, v, *scale_args, **kwargs)
                names = ("o", "stats", "amax_o")
            else:
                outputs = graph.sdpa_fp8(
                    q,
                    k,
                    v,
                    *scale_args,
                    *(tensor(n) for n in ("descale_s", "scale_s", "scale_o")),
                    **kwargs,
                )
                names = ("o", "stats", "amax_s", "amax_o")
        elif self.mxfp8:
            outputs = graph.sdpa_mxfp8_backward(
                q,
                tensor("q_t"),
                k,
                tensor("k_t"),
                v,
                tensor("o"),
                tensor("do_f16"),
                tensor("do"),
                tensor("do_t"),
                tensor("stats"),
                *(
                    tensor(n)
                    for n in (
                        "descale_q",
                        "descale_q_t",
                        "descale_k",
                        "descale_k_t",
                        "descale_v",
                        "descale_do",
                        "descale_do_t",
                    )
                ),
                **kwargs,
            )
            names = ("dq", "dk", "dv", "amax_dq", "amax_dk", "amax_dv")
        else:
            outputs = graph.sdpa_fp8_backward(
                q,
                k,
                v,
                tensor("o"),
                tensor("do"),
                tensor("stats"),
                *(
                    tensor(n)
                    for n in (
                        "descale_q",
                        "descale_k",
                        "descale_v",
                        "descale_o",
                        "descale_do",
                        "descale_s",
                        "descale_dp",
                        "scale_s",
                        "scale_dq",
                        "scale_dk",
                        "scale_dv",
                        "scale_dp",
                    )
                ),
                **kwargs,
            )
            names = ("dq", "dk", "dv", "amax_dq", "amax_dk", "amax_dv", "amax_dp")

        for name, desc in zip(names, outputs):
            if name not in self.specs:
                # MXFP8 amax outputs stay virtual, matching the C++ graph.
                desc.set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(
                    cudnn.data_type.FLOAT
                )
                continue
            spec = self.specs[name]
            desc.set_output(True).set_dim(spec.dim).set_stride(spec.stride).set_data_type(
                getattr(cudnn.data_type, spec.dtype)
            )
            if spec.ragged_offset:
                desc.set_ragged_offset(tensor(spec.ragged_offset))
            tensors[name] = desc
        _event(self, "CREATE_GRAPH")
        _build_stage(self, "validate", graph.validate)
        _build_stage(self, "build_operation_graph", graph.build_operation_graph)
        _build_stage(
            self,
            "create_execution_plans",
            lambda: graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]),
        )
        _build_stage(self, "check_support", graph.check_support)
        return [graph, tensors, None]


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
        attn.add("softmax_offset", p["softmax_offset"])
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

    def name(mapping, value):
        return next(k for k, v in mapping.items() if v == value)

    layout = name(QKVLayout, config.qkv_layout)
    grad_layout = (
        name(QKVLayout, config.dqkv_layout)
        if (config.dqkv_layout != tex.NVTE_QKV_Layout.NVTE_QKV_Layout_NOT_SET)
        else layout
    )
    attn = _AttentionGraph.__new__(_AttentionGraph)
    attn.backward = backward
    attn.device = torch.device("cuda", torch.cuda.current_device())
    attn.q_format, attn.kv_format = _formats(layout)
    attn.b, attn.h, attn.hg = config.batch_size, config.num_attn_heads, config.num_gqa_groups
    attn.dq, attn.dv = config.head_dim_qk, config.head_dim_v
    attn.tq, attn.tk = config.num_tokens_q, config.num_tokens_kv
    ragged_graph = torch.cuda.get_device_capability() >= (9, 0) and (
        torch.cuda.get_device_capability() != (12, 0)
    )
    attn.ragged_stats = attn.q_format == "thd" and ragged_graph
    attn.sq = _token_bucket(attn.tq) if attn.ragged_stats else config.max_seqlen_q
    attn.skv = (
        _token_bucket(attn.tk) if attn.kv_format == "thd" and ragged_graph else config.max_seqlen_kv
    )
    types = {
        DType.kFloat16: "HALF",
        DType.kBFloat16: "BFLOAT16",
        DType.kFloat32: "FLOAT",
        DType.kFloat8E4M3: "FP8_E4M3",
        DType.kFloat8E5M2: "FP8_E5M2",
    }
    attn.fp8 = types[DType.cast(config.qkv_dtype)].startswith("FP8")
    attn.mxfp8 = attn.fp8 and config.scaling_mode == tex.NVTEScalingMode.NVTE_MXFP8_1D_SCALING
    attn.params = {
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
    attn.specs, attn.tensors = {}, {}

    def spec(n, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), dtype="FLOAT", **kwargs):
        attn.specs[n] = _TensorSpec(tuple(dim), tuple(stride), dtype, **kwargs)

    def matrix(n, fmt, h, s, d, dtype, packed="", output=False):
        dim = (attn.b, h, s, d)
        if n in ("k", "v") and layout.startswith("paged_kv_"):
            dim = (getattr(config, "num_pages_" + n), h, getattr(config, "page_size_" + n), d)
        stride = list(_strides(dim, fmt))
        if "3" in packed or "2" in packed:
            count = 3 if "3" in packed else 2
            stride[0] *= count
            stride[2] *= count
            if "h3d" in packed or "h2d" in packed:
                stride[1] *= count
        offset = ""
        if fmt == "thd":
            offset = n + "_offset"
            spec(offset, (attn.b + 1, 1, 1, 1), dtype="INT64")
        spec(n, dim, stride, dtype, output=output, ragged_offset=offset)

    def stats(n, output=False):
        stride = (
            (attn.h * attn.sq, 1, attn.h, 1)
            if attn.ragged_stats
            else (attn.h * attn.sq, attn.sq, 1, 1)
        )
        offset = "stats_offset" if attn.ragged_stats else ""
        if offset:
            spec(offset, (attn.b + 1, 1, 1, 1), dtype="INT64")
        spec(n, (attn.b, attn.h, attn.sq, 1), stride, output=output, ragged_offset=offset)

    def scale(n, matrix_name, fmt, columnwise=False):
        if not attn.mxfp8:
            spec(n)
            return
        b, h, s, d = attn.specs[matrix_name].dim
        dim = (
            (b, h, _round_up((s + 31) // 32, 4), _round_up(d, 128))
            if columnwise
            else (b, h, _round_up(s, 128), _round_up((d + 31) // 32, 4))
        )
        spec(n, dim, _strides(dim, fmt), "FP8_E8M0", reordered=True)

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
        spec("softmax_offset", (1, attn.h, 1, 1), (attn.h, 1, 1, 1))
        if backward:
            spec("dsoftmax_offset", (1, attn.h, 1, 1), (attn.h, 1, 1, 1), output=True)
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
            attn.add("dsoftmax_offset", dsoftmax, output=True)
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
