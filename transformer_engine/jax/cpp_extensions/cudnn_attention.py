# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX cuDNN attention graphs, metadata preparation and cached plan dispatch."""

import math
import os
import threading
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi

from transformer_engine.common.cudnn.attention import (
    AttentionOptions,
    AttentionGraphSpec,
    BuiltAttentionGraph,
    build_attention_graph,
)
from transformer_engine.common.cudnn.graph import (
    TensorSpec,
    CacheDiagnostics,
    bucket as _bucket,
    row_major_stride as _stride,
    create_graph,
    check_graph_support,
    compile_graph,
    uses_ragged_attention_graph,
)

from .cudnn_graph import import_cudnn, check_frontend_version, serialize_graph, graph_attributes


def device_capabilities():
    """Query the JAX devices without entering TE common."""
    return tuple(int(float(device.compute_capability) * 10) for device in jax.local_devices())


def cudnn_version():
    """Return the backend used by the Python frontend."""
    return int(import_cudnn().backend_version())


def _dtype(dtype):
    return {
        jnp.dtype(jnp.float16): "HALF",
        jnp.dtype(jnp.bfloat16): "BFLOAT16",
        jnp.dtype(jnp.float32): "FLOAT",
        jnp.dtype(jnp.float8_e4m3fn): "FP8_E4M3",
        jnp.dtype(jnp.float8_e5m2): "FP8_E5M2",
    }[jnp.dtype(dtype)]


def _dims(avals, config):
    q, k, v = avals[:3]
    layout = config.qkv_layout
    if layout.is_qkvpacked():
        *batch, sq, _, h, dq = q.shape
        sk, hg, dv = sq, h, dq
    elif layout.is_kvpacked():
        *batch, sq, h, dq = q.shape
        *_, sk, _, hg, dv = k.shape
    else:
        *batch, sq, h, dq = q.shape
        sk, hg, dv = v.shape[-3:]
    return math.prod(batch), sq, sk, h, hg, dq, dv


@dataclass(frozen=True)
class AttentionGraph:
    """Plan and bindings, including views of packed QKV storage."""

    graph: object
    names: tuple
    input_indices: tuple
    input_offsets: tuple
    output_indices: tuple
    output_offsets: tuple
    metadata_batch: int
    ragged_stats: bool
    multipliers: tuple
    fp8: bool


@dataclass(frozen=True)
class _AttentionDescription:
    """Static graph metadata and JAX's packed-storage buffer bindings."""

    spec: AttentionGraphSpec
    inputs: tuple
    outputs: tuple
    scalars: tuple
    metadata_batch: int
    ragged_stats: bool
    multipliers: tuple
    fp8: bool


@dataclass
class _SupportedGraph:
    """A supported live graph and its lazily compiled serialization."""

    built: BuiltAttentionGraph
    workspace_size: int | None = None
    serialized_graph: bytes | None = None


_diagnostics = CacheDiagnostics("JAX")
_plan_lock = threading.RLock()


def _site(spec):
    return "fp8" if spec.options.fp8 else "f16", "bwd" if spec.options.backward else "fwd"


@lru_cache(maxsize=256)
def _supported_graph(spec, capabilities, frontend_version, backend_version):
    # Versions and target architecture are part of the plan identity, even when
    # two targets happen to accept identical logical tensor descriptors.
    del frontend_version, backend_version
    cudnn = import_cudnn()
    backend, direction = _site(spec)
    _diagnostics.event(backend, direction, capabilities, "MISS", spec)
    graph = create_graph(cudnn, dict(spec.tensors)["o"].dtype)
    built = build_attention_graph(cudnn, graph, spec)
    _diagnostics.event(backend, direction, capabilities, "CREATE_GRAPH")
    check_graph_support(
        cudnn, graph, lambda stage, function: _diagnostics.run(backend, direction, stage, function)
    )
    _diagnostics.event(backend, direction, capabilities, "CACHE_GRAPH")
    return _SupportedGraph(built)


def _request(direction, avals, config):
    shapes = tuple((tuple(aval.shape), jnp.dtype(aval.dtype)) for aval in avals[:5])
    return (
        direction,
        shapes,
        config,
        device_capabilities(),
        bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"))),
        bool(int(os.getenv("NVTE_FP8_DPA_BWD", "1"))),
    )


def _get_supported_graph(spec, capabilities, frontend_version, backend_version):
    """Share support-query graphs with compilation, and report cache reuse."""
    # pylint infers the wrapped function's arguments for lru_cache.cache_info.
    # pylint: disable=no-value-for-parameter
    with _plan_lock:
        hits = _supported_graph.cache_info().hits
        entry = _supported_graph(spec, capabilities, frontend_version, backend_version)
        if _supported_graph.cache_info().hits > hits:
            _diagnostics.event(*_site(spec), capabilities, "HIT")
        return entry


def check_attention_support(direction, avals, config):
    """Check the common graph without compiling kernels or serializing a plan."""
    description = _describe_graph(*_request(direction, avals, config))
    cudnn = import_cudnn()
    _get_supported_graph(
        description.spec,
        device_capabilities(),
        check_frontend_version(cudnn),
        int(cudnn.backend_version()),
    )


def get_graph(direction, avals, config):
    """Compile when tracing requires workspace sizes and serialized FFI attributes."""
    return _build_graph(*_request(direction, avals, config))


@lru_cache(maxsize=256)
def _build_graph(direction, shapes, config, capabilities, nondeterministic, fp8_backward):
    description = _describe_graph(
        direction, shapes, config, capabilities, nondeterministic, fp8_backward
    )
    cudnn = import_cudnn()
    version = check_frontend_version(cudnn)
    entry = _get_supported_graph(
        description.spec, capabilities, version, int(cudnn.backend_version())
    )
    backend, direction = _site(description.spec)
    with _plan_lock:
        if entry.workspace_size is None:
            workspace_size = compile_graph(
                cudnn,
                entry.built.graph,
                lambda stage, function: _diagnostics.run(backend, direction, stage, function),
            )
            serialized_graph = bytes(entry.built.graph.serialize())
            entry.workspace_size = workspace_size
            entry.serialized_graph = serialized_graph
            _diagnostics.event(backend, direction, capabilities, "BUILD_PLANS")
    uids = entry.built.uids
    serialized = serialize_graph(
        serialized_graph=entry.serialized_graph,
        cudnn_frontend_version=version,
        workspace_size=entry.workspace_size,
        input_uids=[uids[name] for name, _, _ in description.inputs],
        output_uids=[uids[name] for name, _, _ in description.outputs],
        scalar_uids=[uids[name] for name, _ in description.scalars],
        scalar_values=[value for _, value in description.scalars],
    )
    return AttentionGraph(
        serialized,
        tuple(source for _, source, _ in description.inputs),
        tuple(range(len(description.inputs))),
        tuple(offset for _, _, offset in description.inputs),
        tuple(slot for _, slot, _ in description.outputs),
        tuple(offset for _, _, offset in description.outputs),
        description.metadata_batch,
        description.ragged_stats,
        description.multipliers,
        description.fp8,
    )


def _describe_graph(direction, shapes, config, capabilities, nondeterministic, fp8_backward):
    """Translate JAX's API and storage conventions into a common graph description."""
    avals = tuple(jax.ShapeDtypeStruct(*shape) for shape in shapes)
    batch, sq, sk, h, hg, dq, dv = _dims(avals, config)
    backward = direction == "bwd"
    ragged = config.qkv_layout.is_thd()
    ragged_stats = ragged and uses_ragged_attention_graph(capabilities[0])
    segments = batch * config.max_segments_per_seq if ragged else batch
    # JAX uses fixed metadata capacities, including for empty ragged storage.
    # Keep positive graph capacities while runtime segment lengths may be zero.
    graph_batch = _bucket(max(segments, 1), 32, 512) if ragged_stats else segments
    graph_sq = _bucket(max(batch * sq, 1), 1024, 32768) if ragged_stats else sq
    graph_sk = _bucket(max(batch * sk, 1), 1024, 32768) if ragged_stats else sk
    fp8 = config.fp8 and (not backward or fp8_backward)
    version = cudnn_version()
    if ragged and capabilities[0] < 90:
        raise ValueError("F16/BF16 fused attention with THD format requires sm90 or later.")
    if (
        backward
        and ragged_stats
        and config.softmax_type.name == "LEARNABLE_SOFTMAX"
        and version < 92600
    ):
        raise ValueError("THD learnable softmax backward requires cuDNN 9.26.0 or later.")
    if fp8 and ragged:
        if version < 92300:
            raise ValueError("FP8 THD attention requires cuDNN 9.23.0 or later.")
        if backward and capabilities[0] < 100:
            raise ValueError("FP8 THD backward requires sm100 or later.")
    if fp8 and backward and config.attn_bias_type.name != "NO_BIAS":
        raise ValueError("cuDNN FP8 backward does not support attention bias or ALiBi.")
    io_dtype = _dtype(avals[0].dtype)
    qkv_dtype = "FP8_E4M3" if fp8 else io_dtype
    specs, inputs, outputs, scalars = {}, [], [], []

    def tensor(name, dim, stride, dtype=io_dtype, source=None, offset=0, ragged_offset=""):
        specs[name] = TensorSpec(tuple(dim), tuple(stride), dtype, ragged_offset=ragged_offset)
        inputs.append((name, name if source is None else source, offset))

    def constant(name, value):
        specs[name] = TensorSpec((1, 1, 1, 1), (1, 1, 1, 1), "FLOAT", by_value=True)
        scalars.append((name, np.asarray(value, np.float32).tobytes()))

    def output(name, dim, stride, slot, offset=0, dtype=io_dtype, ragged_offset=""):
        specs[name] = TensorSpec(
            tuple(dim), tuple(stride), dtype, output=True, ragged_offset=ragged_offset
        )
        outputs.append((name, slot, offset))

    qpack = 3 if config.qkv_layout.is_qkvpacked() else 1
    kvpack = 3 if qpack == 3 else (2 if config.qkv_layout.is_kvpacked() else 1)
    itemsize = 1 if fp8 else avals[0].dtype.itemsize
    qstride = (graph_sq * qpack * h * dq, dq, qpack * h * dq, 1)
    kstride = (graph_sk * kvpack * hg * dq, dq, kvpack * hg * dq, 1)
    vstride = (graph_sk * kvpack * hg * dv, dv, kvpack * hg * dv, 1)
    ostride = (graph_sq * h * dv, dv, h * dv, 1)
    qdim = (graph_batch, h, graph_sq, dq)
    kdim = (graph_batch, hg, graph_sk, dq)
    vdim = (graph_batch, hg, graph_sk, dv)
    odim = (graph_batch, h, graph_sq, dv)
    tensor("q", qdim, qstride, qkv_dtype, ragged_offset="offset_q" if ragged else "")
    tensor(
        "k",
        kdim,
        kstride,
        qkv_dtype,
        source="q" if qpack == 3 else "k",
        offset=h * dq * itemsize if qpack == 3 else 0,
        ragged_offset="offset_k" if ragged else "",
    )
    tensor(
        "v",
        vdim,
        vstride,
        qkv_dtype,
        source="q" if qpack == 3 else ("k" if kvpack == 2 else "v"),
        offset=2 * h * dq * itemsize if qpack == 3 else (hg * dq * itemsize if kvpack == 2 else 0),
        ragged_offset="offset_v" if ragged else "",
    )
    multipliers = (qpack * h * dq, kvpack * hg * dq, kvpack * hg * dv, h * dv, h)
    if ragged:
        for name in ("offset_q", "offset_k", "offset_v", "offset_o", "offset_stats"):
            tensor(name, (graph_batch + 1, 1, 1, 1), (1, 1, 1, 1), "INT64")
    stats_dim = (graph_batch, h, graph_sq, 1)
    stats_stride = (h * graph_sq, 1, h, 1) if ragged_stats else (h * graph_sq, graph_sq, 1, 1)
    stats_offset = "offset_stats" if ragged_stats else ""
    o_offset = "offset_o" if ragged else ""
    left, right = config.effective_window_size
    if config.attn_mask_type.is_causal():
        right = 0
    padding = config.attn_mask_type.is_padding() or ragged
    if padding:
        for name in ("seq_q", "seq_kv"):
            tensor(name, (graph_batch, 1, 1, 1), (1, 1, 1, 1), "INT32")
    dropout = float(config.dropout_probability) if config.is_training else 0.0
    if dropout:
        tensor("seed", (1, 1, 1, 1), (1, 1, 1, 1), "INT64", source="rng")
        tensor("offset", (1, 1, 1, 1), (1, 1, 1, 1), "INT64", source="rng", offset=8)
    bias_type = config.attn_bias_type.name
    if bias_type not in ("ALIBI", "NO_BIAS"):
        bias_shape = (math.prod(avals[3].shape[:-3]), *avals[3].shape[-3:])
        tensor("bias", bias_shape, _stride(bias_shape))
        if backward and bias_shape[:3] != (1, 1, 1):
            output("dbias", bias_shape, _stride(bias_shape), 3)
    if config.softmax_type.name != "VANILLA_SOFTMAX":
        tensor("sink", (1, h, 1, 1), (h, 1, 1, 1), "FLOAT")
        if backward:
            output("dsink", (1, h, 1, 1), (h, 1, 1, 1), 4, dtype="FLOAT")
    constant("attn_scale", config.scaling_factor)
    if fp8:
        for name in ("descale_q", "descale_k", "descale_v"):
            tensor(name, (1, 1, 1, 1), (1, 1, 1, 1), "FLOAT")
        constant("scale_s", 448.0)
        constant("descale_s", 1.0 / 448.0)
    if backward:
        tensor("o", odim, ostride, ragged_offset=o_offset)
        tensor("do", odim, ostride, "FP8_E5M2" if fp8 else io_dtype, ragged_offset=o_offset)
        tensor("stats", stats_dim, stats_stride, "FLOAT", ragged_offset=stats_offset)
        grad_slots = (0, 0 if qpack == 3 else 1, 0 if qpack == 3 else (1 if kvpack == 2 else 2))
        grad_offsets = (
            0,
            h * dq * avals[0].dtype.itemsize if qpack == 3 else 0,
            (
                2 * h * dq * avals[0].dtype.itemsize
                if qpack == 3
                else (hg * dq * avals[0].dtype.itemsize if kvpack == 2 else 0)
            ),
        )
        for name, dim, stride, slot, offset, ragged_offset in zip(
            ("dq", "dk", "dv"),
            (qdim, kdim, vdim),
            (qstride, kstride, vstride),
            grad_slots,
            grad_offsets,
            ("offset_q", "offset_k", "offset_v"),
        ):
            output(name, dim, stride, slot, offset, ragged_offset=ragged_offset if ragged else "")
        if fp8:
            tensor("descale_do", (1, 1, 1, 1), (1, 1, 1, 1), "FLOAT")
            for name in ("descale_o", "scale_dp", "descale_dp", "scale_dq", "scale_dk", "scale_dv"):
                constant(name, 1.0)
            for slot, name in enumerate(("amax_dq", "amax_dk", "amax_dv", "amax_dp"), 5):
                output(name, (1, 1, 1, 1), (1, 1, 1, 1), slot, dtype="FLOAT")
    else:
        output("o", odim, ostride, 0, ragged_offset=o_offset)
        output("stats", stats_dim, stats_stride, 1, dtype="FLOAT", ragged_offset=stats_offset)
        if config.return_max_logit:
            output("max", stats_dim, stats_stride, 2, dtype="FLOAT", ragged_offset=stats_offset)
        if fp8:
            constant("scale_o", 1.0)
            output("amax_s", (1, 1, 1, 1), (1, 1, 1, 1), 3, dtype="FLOAT")
            output("amax_o", (1, 1, 1, 1), (1, 1, 1, 1), 4, dtype="FLOAT")
    options = AttentionOptions(
        backward=backward,
        fp8=fp8,
        bottom_right=config.bottom_right_diagonal,
        left_bound=left + 1 if left >= 0 else None,
        right_bound=right if right >= 0 else None,
        padding=padding,
        alibi=bias_type == "ALIBI",
        dropout=dropout,
        deterministic=backward and not nondeterministic,
        max_total_seqlen_q=graph_sq if backward and ragged_stats and not fp8 else 0,
        max_total_seqlen_kv=graph_sk if backward and ragged_stats and not fp8 else 0,
    )
    return _AttentionDescription(
        AttentionGraphSpec.from_dict(options, specs),
        tuple(inputs),
        tuple(outputs),
        tuple(scalars),
        graph_batch,
        ragged_stats,
        multipliers,
        fp8,
    )


def _offset64(offsets, multiplier):
    """Compute exact uint64 element offsets without enabling JAX x64 globally."""
    value = offsets.astype(jnp.uint32)
    low, high = value & 65535, value >> 16
    ml, mh = multiplier & 65535, multiplier >> 16
    w0 = low * ml
    carry = high * ml + (w0 >> 16)
    w1 = (carry & 65535) + low * mh
    upper = high * mh + (carry >> 16) + (w1 >> 16)
    lower = (w1 << 16) | (w0 & 65535)
    return jnp.stack((lower, upper), axis=-1)


def _quantize(value, dtype):
    """Current tensor scaling implemented in JAX, independently of TE common."""
    maximum = jnp.max(jnp.abs(value.astype(jnp.float32)))
    scale = jnp.where(maximum > 0, float(jnp.finfo(dtype).max) / maximum, 1.0)
    scale = jnp.minimum(scale, jnp.finfo(jnp.float32).max)
    return (value.astype(jnp.float32) * scale).astype(dtype), jnp.reciprocal(scale)


def _metadata(value, size, fill=0):
    """Bind only the configured segment capacity from fixed-size JAX metadata."""
    value = value.reshape(-1)[:size]
    return jnp.pad(value, (0, size - value.size), constant_values=fill)


def _dense_stats_to_packed(value, offsets, lengths, shape):
    """Convert SM12x's per-segment BHS1 statistics to JAX's packed TH1 layout."""
    tokens, heads = math.prod(shape[:-2]), shape[-2]
    rows = value.shape[-2]
    indices = offsets[: lengths.size, None] + jnp.arange(rows)[None, :]
    indices = jnp.where(jnp.arange(rows)[None, :] < lengths[:, None], indices, tokens)
    values = value.transpose(0, 2, 1, 3).reshape(-1, heads, 1)
    packed = jnp.full((tokens, heads, 1), -1e30, value.dtype)
    return packed.at[indices.reshape(-1)].set(values, mode="drop").reshape(shape)


def _packed_stats_to_dense(value, offsets, lengths, rows):
    """Reconstruct dense per-segment statistics for the cuDNN backward graph."""
    heads = value.shape[-2]
    indices = offsets[: lengths.size, None] + jnp.arange(rows)[None, :]
    valid = jnp.arange(rows)[None, :] < lengths[:, None]
    dense = jnp.take(value.reshape(-1, heads, 1), indices, axis=0, mode="fill", fill_value=0)
    dense = jnp.where(valid[:, :, None, None], dense, 0)
    return dense.transpose(0, 2, 1, 3)


def execute(direction, args, config, result_avals):
    """Prepare device metadata with JAX and dispatch a cached graph through FFI."""
    plan = get_graph(direction, args, config)
    q, k, v, bias, sink = args[:5]
    backward = direction == "bwd"
    if backward:
        stats, rng, o, do, q_cu, kv_cu, q_offsets, kv_offsets = args[5:13]
    else:
        seed, q_cu, kv_cu, q_offsets, kv_offsets = args[5:10]
        # JAX PRNG keys are explicit: replaying a key replays the dropout mask.
        rng = jnp.broadcast_to(
            jnp.concatenate((seed.reshape(-1)[:2], jnp.zeros(2, jnp.uint32))), result_avals[3].shape
        )
    values = {"q": q, "k": k, "v": v, "bias": bias, "sink": sink, "rng": rng}
    values["seq_q"] = _metadata(jnp.diff(q_cu), plan.metadata_batch)
    values["seq_kv"] = _metadata(jnp.diff(kv_cu), plan.metadata_batch)
    dense_ragged_stats = config.qkv_layout.is_thd() and not plan.ragged_stats
    if config.qkv_layout.is_thd():
        batch, sq, sk, *_ = _dims(args, config)
        q_offsets = _metadata(q_offsets, plan.metadata_batch + 1, batch * sq)
        kv_offsets = _metadata(kv_offsets, plan.metadata_batch + 1, batch * sk)
        for name, offsets, multiplier in zip(
            ("offset_q", "offset_k", "offset_v", "offset_o", "offset_stats"),
            (q_offsets, kv_offsets, kv_offsets, q_offsets, q_offsets),
            plan.multipliers,
        ):
            values[name] = _offset64(offsets, multiplier)
    if config.fp8:
        for name in ("q", "k", "v"):
            # Empty dummy operands are not quantized.
            if values[name].size:
                values[name], values["descale_" + name] = _quantize(values[name], jnp.float8_e4m3fn)
        if config.qkv_layout.is_qkvpacked():
            values["descale_k"] = values["descale_v"] = values["descale_q"]
        elif config.qkv_layout.is_kvpacked():
            values["descale_v"] = values["descale_k"]
        if not plan.fp8:
            for name in ("q", "k", "v"):
                if values[name].size:
                    values[name] = (
                        values[name].astype(jnp.float32) * values["descale_" + name]
                    ).astype(q.dtype)
    if backward:
        values.update(stats=stats, o=o, do=do)
        if dense_ragged_stats:
            values["stats"] = _packed_stats_to_dense(
                stats,
                q_offsets,
                values["seq_q"],
                q.shape[-4] if config.qkv_layout.is_qkvpacked() else q.shape[-3],
            )
        if plan.fp8:
            values["do"], values["descale_do"] = _quantize(do, jnp.float8_e5m2)
        specs = list(result_avals[:-1])
        if plan.fp8:
            specs += [jax.ShapeDtypeStruct((1,), jnp.float32)] * 4
    else:
        specs = list(result_avals[:3])
        if dense_ragged_stats:
            _, sq, _, heads, *_ = _dims(args, config)
            dense_shape = (plan.metadata_batch, heads, sq, 1)
            specs[1] = jax.ShapeDtypeStruct(dense_shape, jnp.float32)
            if config.return_max_logit:
                specs[2] = jax.ShapeDtypeStruct(dense_shape, jnp.float32)
        if plan.fp8:
            specs += [jax.ShapeDtypeStruct((1,), jnp.float32)] * 2
    specs.append(jax.ShapeDtypeStruct((plan.graph.workspace_size,), jnp.uint8))
    attributes = graph_attributes(plan.graph)
    attributes.update(
        input_indices=np.asarray(plan.input_indices, np.int64),
        input_offsets=np.asarray(plan.input_offsets, np.int64),
        output_indices=np.asarray(plan.output_indices, np.int64),
        output_offsets=np.asarray(plan.output_offsets, np.int64),
    )
    if not backward and config.qkv_layout.is_thd():
        # Match the neutral finite statistics value for empty ragged ring blocks.
        # Filling FP32 bytes with 0xF0 gives a large negative number without NaNs
        # when the ring correction subtracts statistics for two empty blocks.
        fills = np.zeros(len(specs) - 1, np.int64)
        fills[1:3] = 0xF0
        attributes["output_fill_values"] = fills
    results = ffi.ffi_call(
        "te_cudnn_graph_ffi", tuple(jax.ShapeDtypeStruct(spec.shape, spec.dtype) for spec in specs)
    )(*(values[name] for name in plan.names), **attributes)
    if backward:
        return (*results[:5], results[-1])
    output, stats, score_max = results[:3]
    if dense_ragged_stats:
        packed_shape = result_avals[1].shape
        stats = _dense_stats_to_packed(stats, q_offsets, values["seq_q"], packed_shape)
        if config.return_max_logit:
            score_max = _dense_stats_to_packed(score_max, q_offsets, values["seq_q"], packed_shape)
            score_max = jnp.swapaxes(score_max, -3, -2)
    return output, stats, score_max, rng, results[-1]
