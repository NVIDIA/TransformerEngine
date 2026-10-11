# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX cuDNN attention graphs, metadata preparation and cached plan dispatch."""

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi

from .cudnn_graph import import_cudnn, finalize_graph, serialize_graph, graph_attributes


def device_capabilities():
    """Query the JAX devices without entering TE common."""
    return tuple(int(float(device.compute_capability) * 10) for device in jax.local_devices())


def cudnn_version():
    """Return the backend used by the Python frontend."""
    return int(import_cudnn().backend_version())


def _dtype(cudnn, dtype):
    return {
        jnp.dtype(jnp.float16): cudnn.data_type.HALF,
        jnp.dtype(jnp.bfloat16): cudnn.data_type.BFLOAT16,
        jnp.dtype(jnp.float32): cudnn.data_type.FLOAT,
        jnp.dtype(jnp.float8_e4m3fn): cudnn.data_type.FP8_E4M3,
        jnp.dtype(jnp.float8_e5m2): cudnn.data_type.FP8_E5M2,
    }[jnp.dtype(dtype)]


def _bucket(value, minimum, maximum):
    if value <= maximum:
        return max(minimum, 1 << (value - 1).bit_length())
    return (value + maximum - 1) // maximum * maximum


def _stride(shape):
    return tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))


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


def get_graph(direction, avals, config):
    """Cache by configuration, operand metadata and GPU architecture."""
    shapes = tuple((tuple(aval.shape), jnp.dtype(aval.dtype)) for aval in avals[:5])
    return _build_graph(
        direction,
        shapes,
        config,
        device_capabilities(),
        bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"))),
        bool(int(os.getenv("NVTE_FP8_DPA_BWD", "1"))),
    )


@lru_cache(maxsize=256)
def _build_graph(direction, shapes, config, capabilities, nondeterministic, fp8_backward):
    cudnn = import_cudnn()
    avals = tuple(jax.ShapeDtypeStruct(*shape) for shape in shapes)
    batch, sq, sk, h, hg, dq, dv = _dims(avals, config)
    backward = direction == "bwd"
    ragged = config.qkv_layout.is_thd()
    ragged_stats = ragged and capabilities[0] >= 90 and capabilities[0] != 120
    segments = batch * config.max_segments_per_seq if ragged else batch
    graph_batch = _bucket(segments, 32, 512) if ragged_stats else segments
    graph_sq = _bucket(batch * sq, 1024, 32768) if ragged_stats else sq
    graph_sk = _bucket(batch * sk, 1024, 32768) if ragged_stats else sk
    fp8 = config.fp8 and (not backward or fp8_backward)
    version = int(cudnn.backend_version())
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
    io_dtype = _dtype(cudnn, avals[0].dtype)
    qkv_dtype = cudnn.data_type.FP8_E4M3 if fp8 else io_dtype
    graph = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    names, input_uids, input_indices, input_offsets = [], [], [], []
    scalar_uids, scalar_values = [], []
    uid = 0

    def tensor(name, dim, stride, dtype=None, offset=0):
        nonlocal uid
        uid += 1
        names.append(name)
        input_uids.append(uid)
        input_indices.append(len(names) - 1)
        input_offsets.append(offset)
        return graph.tensor(
            name=name,
            dim=dim,
            stride=stride,
            data_type=io_dtype if dtype is None else dtype,
            uid=uid,
        )

    def constant(name, value):
        nonlocal uid
        uid += 1
        scalar_uids.append(uid)
        scalar_values.append(np.asarray(value, np.float32).tobytes())
        return graph.tensor(
            name=name,
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
            uid=uid,
            is_pass_by_value=True,
        )

    def output(value, dim, stride, slot, offset=0, dtype=None):
        nonlocal uid
        uid += 1
        value.set_output(True).set_uid(uid).set_dim(dim).set_stride(stride)
        value.set_data_type(io_dtype if dtype is None else dtype)
        outputs.append((uid, slot, offset))
        return value

    outputs = []
    qpack = 3 if config.qkv_layout.is_qkvpacked() else 1
    kvpack = 3 if qpack == 3 else (2 if config.qkv_layout.is_kvpacked() else 1)
    # FP8 conversion preserves the packed storage layout.
    itemsize = 1 if fp8 else avals[0].dtype.itemsize
    qstride = (graph_sq * qpack * h * dq, dq, qpack * h * dq, 1)
    kstride = (graph_sk * kvpack * hg * dq, dq, kvpack * hg * dq, 1)
    vstride = (graph_sk * kvpack * hg * dv, dv, kvpack * hg * dv, 1)
    ostride = (graph_sq * h * dv, dv, h * dv, 1)
    qdim, kdim, vdim = (
        (graph_batch, h, graph_sq, dq),
        (graph_batch, hg, graph_sk, dq),
        (graph_batch, hg, graph_sk, dv),
    )
    odim = (graph_batch, h, graph_sq, dv)
    q = tensor("q", qdim, qstride, qkv_dtype)
    k = tensor(
        "q" if qpack == 3 else "k", kdim, kstride, qkv_dtype, h * dq * itemsize if qpack == 3 else 0
    )
    v = tensor(
        "q" if qpack == 3 else ("k" if kvpack == 2 else "v"),
        vdim,
        vstride,
        qkv_dtype,
        2 * h * dq * itemsize if qpack == 3 else (hg * dq * itemsize if kvpack == 2 else 0),
    )

    multipliers = (qpack * h * dq, kvpack * hg * dq, kvpack * hg * dv, h * dv, h)
    offsets = []
    if ragged:
        for name in ("offset_q", "offset_k", "offset_v", "offset_o", "offset_stats"):
            offsets.append(
                tensor(name, (graph_batch + 1, 1, 1, 1), (1, 1, 1, 1), cudnn.data_type.INT64)
            )
        q.set_ragged_offset(offsets[0])
        k.set_ragged_offset(offsets[1])
        v.set_ragged_offset(offsets[2])
    stats_dim = (graph_batch, h, graph_sq, 1)
    stats_stride = (h * graph_sq, 1, h, 1) if ragged_stats else (h * graph_sq, graph_sq, 1, 1)
    options = {
        "attn_scale": float(config.scaling_factor),
        "diagonal_alignment": (
            cudnn.diagonal_alignment.BOTTOM_RIGHT
            if config.bottom_right_diagonal
            else cudnn.diagonal_alignment.TOP_LEFT
        ),
    }
    left, right = config.effective_window_size
    if left >= 0:
        options["left_bound" if fp8 else "diagonal_band_left_bound"] = left + 1
    if config.attn_mask_type.is_causal():
        right = 0
    if right >= 0:
        options["right_bound" if fp8 else "diagonal_band_right_bound"] = right
    if config.attn_mask_type.is_padding() or ragged:
        options.update(
            use_padding_mask=True,
            seq_len_q=tensor("seq_q", (graph_batch, 1, 1, 1), (1, 1, 1, 1), cudnn.data_type.INT32),
            seq_len_kv=tensor(
                "seq_kv", (graph_batch, 1, 1, 1), (1, 1, 1, 1), cudnn.data_type.INT32
            ),
        )
    if config.dropout_probability and config.is_training:
        options["dropout"] = (
            float(config.dropout_probability),
            tensor("rng", (1, 1, 1, 1), (1, 1, 1, 1), cudnn.data_type.INT64),
            tensor("rng", (1, 1, 1, 1), (1, 1, 1, 1), cudnn.data_type.INT64, 8),
        )
    bias_type = config.attn_bias_type.name
    if bias_type == "ALIBI":
        if fp8 and backward:
            raise ValueError("cuDNN FP8 backward does not support ALiBi.")
        options["use_alibi_mask"] = True
    elif bias_type != "NO_BIAS":
        if fp8 and backward:
            raise ValueError("cuDNN FP8 backward does not support attention bias.")
        bias_shape = (*avals[3].shape[-3:],)
        bias_shape = (math.prod(avals[3].shape[:-3]), *bias_shape)
        options["bias"] = tensor("bias", bias_shape, _stride(bias_shape))
        if backward and bias_shape[:3] != (1, 1, 1):
            dbias = graph.tensor(
                name="dbias", dim=bias_shape, stride=_stride(bias_shape), data_type=io_dtype
            )
            options["dBias"] = output(dbias, bias_shape, _stride(bias_shape), 3)
    if config.softmax_type.name != "VANILLA_SOFTMAX":
        options["sink_token"] = tensor("sink", (1, h, 1, 1), (h, 1, 1, 1), cudnn.data_type.FLOAT)
        if backward:
            dsink = graph.tensor(
                name="dsink", dim=(1, h, 1, 1), stride=(h, 1, 1, 1), data_type=cudnn.data_type.FLOAT
            )
            options["dSink_token"] = output(
                dsink, (1, h, 1, 1), (h, 1, 1, 1), 4, dtype=cudnn.data_type.FLOAT
            )

    if fp8:
        for name in ("descale_q", "descale_k", "descale_v"):
            options[name] = tensor(name, (1, 1, 1, 1), (1, 1, 1, 1), cudnn.data_type.FLOAT)
        # Softmax probabilities are in [0, 1]; use the full E4M3 range.
        options["scale_s"] = constant("scale_s", 448.0)
        options["descale_s"] = constant("descale_s", 1.0 / 448.0)
    if backward:
        options["use_deterministic_algorithm"] = not nondeterministic
        o = tensor("o", odim, ostride)
        do = tensor("do", odim, ostride, cudnn.data_type.FP8_E5M2 if fp8 else io_dtype)
        stats = tensor("stats", stats_dim, stats_stride, cudnn.data_type.FLOAT)
        if ragged:
            o.set_ragged_offset(offsets[3])
            do.set_ragged_offset(offsets[3])
            if ragged_stats:
                stats.set_ragged_offset(offsets[4])
        if fp8:
            options["descale_o"] = constant("descale_o", 1.0)
            options["descale_dO"] = tensor(
                "descale_do", (1, 1, 1, 1), (1, 1, 1, 1), cudnn.data_type.FLOAT
            )
            options["scale_dP"] = constant("scale_dp", 1.0)
            options["descale_dP"] = constant("descale_dp", 1.0)
            for name in ("scale_dQ", "scale_dK", "scale_dV"):
                options[name] = constant(name, 1.0)
            gradients = graph.sdpa_fp8_backward(q=q, k=k, v=v, o=o, dO=do, stats=stats, **options)
        else:
            if ragged_stats:
                options["max_total_seq_len_q"] = graph_sq
                options["max_total_seq_len_kv"] = graph_sk
            gradients = graph.sdpa_backward(q=q, k=k, v=v, o=o, dO=do, stats=stats, **options)
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
        for index, (grad, dim, stride, slot, offset) in enumerate(
            zip(
                gradients[:3],
                (qdim, kdim, vdim),
                (qstride, kstride, vstride),
                grad_slots,
                grad_offsets,
            )
        ):
            output(grad, dim, stride, slot, offset)
            if ragged:
                grad.set_ragged_offset(offsets[index])
        if fp8:
            for slot, amax in enumerate(gradients[3:7], 5):
                output(amax, (1, 1, 1, 1), (1, 1, 1, 1), slot, dtype=cudnn.data_type.FLOAT)
    else:
        if config.return_max_logit:
            score_max = graph.tensor(
                name="max", dim=stats_dim, stride=stats_stride, data_type=cudnn.data_type.FLOAT
            )
            if ragged_stats:
                score_max.set_ragged_offset(offsets[4])
            options["score_max"] = output(
                score_max, stats_dim, stats_stride, 2, dtype=cudnn.data_type.FLOAT
            )
        # The public JAX operation remains differentiable with dropout disabled
        # by is_training=False, just as the previous common-backed path did.
        options["generate_stats"] = True
        if fp8:
            options["scale_o"] = constant("scale_o", 1.0)
            o, stats, amax_s, amax_o = graph.sdpa_fp8(q=q, k=k, v=v, **options)
            output(amax_s, (1, 1, 1, 1), (1, 1, 1, 1), 3, dtype=cudnn.data_type.FLOAT)
            output(amax_o, (1, 1, 1, 1), (1, 1, 1, 1), 4, dtype=cudnn.data_type.FLOAT)
        else:
            o, stats = graph.sdpa(q=q, k=k, v=v, **options)
        output(o, odim, ostride, 0)
        if ragged:
            o.set_ragged_offset(offsets[3])
        output(stats, stats_dim, stats_stride, 1, dtype=cudnn.data_type.FLOAT)
        if ragged_stats:
            stats.set_ragged_offset(offsets[4])

    workspace, data, version = finalize_graph(cudnn, graph)
    serialized = serialize_graph(
        serialized_graph=data,
        cudnn_frontend_version=version,
        workspace_size=workspace,
        input_uids=input_uids,
        output_uids=[value[0] for value in outputs],
        scalar_uids=scalar_uids,
        scalar_values=scalar_values,
    )
    return AttentionGraph(
        serialized,
        tuple(names),
        tuple(input_indices),
        tuple(input_offsets),
        tuple(value[1] for value in outputs),
        tuple(value[2] for value in outputs),
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
    """Convert SM120's per-segment BHS1 statistics to JAX's packed TH1 layout."""
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
