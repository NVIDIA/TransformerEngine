# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Prepare cuDNN OSS backward plans for TE's native dispatcher.

Only preparation enters Python. The registry entry is the compiled TVM-FFI
function for the complete backward chain, including its scratch initialization.
The private FE prepared-plan adapter is deliberately confined to this module.
"""

from functools import partial
from importlib.metadata import PackageNotFoundError, version
import logging

_LOG = logging.getLogger("transformer_engine.frost")
_OWNERS = {}
_ROLES = ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv")
_ORIGIN_ROLES = ("q", "k", "v", "o", "do", "dq", "dk", "dv", "stats")
_POINTER_ROLES = _ROLES + ("seq_q", "seq_kv") + tuple("ragged_" + r for r in _ORIGIN_ROLES)
_ENGINES = {128: 20609, 256: 20607}


def _declarations(batch, heads, kv_heads, sq, sk, dim, tq, thd):
    """Logical BHSD declarations over TE's BSHD or packed token-major storage."""
    result = {}
    for role in _ROLES:
        kv = role in ("k", "v", "dk", "dv")
        h, s = (kv_heads, sk) if kv else (heads, sq)
        if role == "stats":
            shape = (batch, heads, sq, 1)
            stride = (tq * heads, 1, heads, 1) if thd else (heads * sq, sq, 1, 1)
        else:
            shape, stride = (batch, h, s, dim), (s * h * dim, dim, h * dim, 1)
        result[role] = (shape, stride)
    return result


def _native_metadata(spec, dim, thd, expected_prefix):
    """Refuse an unfamiliar prepared ABI instead of dropping optional operands."""
    from cudnn.frost.compiled_cache import positional_entry
    import tvm_ffi

    if (
        spec.name != f"sdpa_bwd_sm100_d{dim}"
        or spec.scale_log2
        or spec.packed_tile_groups
        or bool(spec.length_form) != thd
    ):
        raise RuntimeError("Unsupported cuDNN OSS prepared backward contract")
    roles = tuple(spec.roles)
    expected = _POINTER_ROLES if thd else _ROLES + (("seq_kv", "delta") if dim == 256 else ())
    if roles != expected or len(spec.operands) != len(roles):
        raise RuntimeError(f"Unsupported cuDNN OSS operand order: {roles}")
    native = positional_entry(spec.artifact.entry)
    if not isinstance(native, tvm_ffi.Function):
        raise RuntimeError("cuDNN OSS must expose a compiled native TVM-FFI entry")
    if thd:
        bound = getattr(spec.fn, "entry", None)
        if not isinstance(bound, partial) or tuple(bound.args) != expected_prefix:
            raise RuntimeError("Unsupported cuDNN OSS packed backward prefix")
    elif expected_prefix:
        raise RuntimeError("Dense backward cannot have a packed prefix")
    metadata = [1, int(spec.workspace_bytes), int(thd), len(expected_prefix), len(roles)]
    metadata.extend(expected_prefix)
    for role, operand in zip(roles, spec.operands):
        if operand is None:
            metadata.extend((-1, 0, 1))
            continue
        if role not in _POINTER_ROLES or operand.opaque_bytes or operand.packed_tile_bytes:
            raise RuntimeError(f"Unsupported cuDNN OSS operand: {role}")
        span = int(operand.span) * int(operand.itemsize)
        if role in ("seq_q", "seq_kv"):
            span += 4  # TE supplies B+1 cumulative lengths, lens_form=3.
        if span <= 0 or span > 2**63 - 1:
            raise RuntimeError("cuDNN OSS operand byte span is not representable")
        metadata.extend((_POINTER_ROLES.index(role), span, int(operand.alignment)))
    return native, metadata


def get_frost_attention_backward(
    key, batch, heads, kv_heads, sq, sk, dim, tq, tk, thd, causal, scale, device, bottom_right=False
):
    """Compile a graph declaration and register its native chain; allocate no data."""
    try:
        import cudnn
        from cuda.bindings import runtime
        import tvm_ffi
    except ImportError as error:
        _LOG.warning("cuDNN OSS attention dependency unavailable: %s", error)
        return False

    from packaging.version import Version

    try:
        dsl_version = Version(version("nvidia-cutlass-dsl"))
    except PackageNotFoundError:
        _LOG.warning("cuDNN OSS attention requires nvidia-cutlass-dsl >= 4.8.0")
        return False
    if dsl_version < Version("4.8.0"):
        _LOG.warning("cuDNN OSS attention requires nvidia-cutlass-dsl >= 4.8.0")
        return False
    if dim not in _ENGINES or (thd and dim != 128):
        return False
    if bottom_right and (dim != 256 or not causal):
        return False
    from cutlass.base_dsl.version_info import CUDA_VERSION

    if CUDA_VERSION.major < 13:
        _LOG.warning("cuDNN OSS backward requires the CUDA 13 flavor: nvidia-cutlass-dsl[cu13]")
        return False
    error, current = runtime.cudaGetDevice()
    if int(error) != 0 or int(current) != device:
        raise RuntimeError("cuDNN OSS preparation must use the calling CUDA device")
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    declarations = _declarations(batch, heads, kv_heads, sq, sk, dim, tq, thd)
    refs = {}

    def attach(role, tensor):
        if not thd:
            return
        kv = role in ("k", "v", "dk", "dv")
        h = kv_heads if kv else heads
        offset = graph.tensor(
            name=role + "_origin",
            dim=[batch + 1, 1, 1, 1],
            stride=[1, 1, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        tensor.set_ragged_offset(offset)
        tensor.set_ragged_offset_multiplier(h if role == "stats" else h * dim)

    for role in _ROLES[:6]:
        shape, stride = declarations[role]
        refs[role] = graph.tensor(
            name=role,
            dim=shape,
            stride=stride,
            data_type=cudnn.data_type.FLOAT if role == "stats" else cudnn.data_type.BFLOAT16,
        )
        attach(role, refs[role])
    lengths = {}
    if thd:
        for role in ("q", "kv"):
            lengths[role] = graph.tensor(
                name="length_" + role,
                dim=[batch, 1, 1, 1],
                stride=[1, 1, 1, 1],
                data_type=cudnn.data_type.INT32,
            )
    outputs = graph.sdpa_backward(
        q=refs["q"],
        k=refs["k"],
        v=refs["v"],
        o=refs["o"],
        dO=refs["do"],
        stats=refs["stats"],
        attn_scale=scale,
        use_causal_mask=causal and not bottom_right,
        use_causal_mask_bottom_right=causal and bottom_right,
        use_padding_mask=thd,
        seq_len_q=lengths.get("q"),
        seq_len_kv=lengths.get("kv"),
        max_total_seq_len_q=tq if thd else None,
        max_total_seq_len_kv=tk if thd else None,
        use_deterministic_algorithm=False,
    )
    for role, tensor in zip(_ROLES[6:], outputs):
        shape, stride = declarations[role]
        tensor.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        tensor.set_dim(shape).set_stride(stride)
        attach(role, tensor)
    try:
        graph.validate()
        graph.build_operation_graph()
        try:
            graph.create_execution_plan(_ENGINES[dim], {})
        except ValueError as error:
            if "no python engine on this graph owns engine_id" not in str(error):
                raise
            _LOG.warning("Installed cuDNN Frontend lacks the requested OSS engine: %s", error)
            return False
        graph.build_plans()
    except (NotImplementedError, cudnn.cudnnGraphNotSupportedError) as error:
        _LOG.warning("cuDNN OSS attention declined this declaration: %s", error)
        return False
    spec = graph._compiled_plans[graph._plan_index]._prepared.spec
    prefix = (min(batch * sq, tq), min(batch * sk, tk), tq, tk, sq, sk, 0) if thd else ()
    native, metadata = _native_metadata(spec, dim, thd, prefix)
    if spec.device_index != device or float(spec.scale) != float(scale):
        raise RuntimeError("cuDNN OSS plan device/scale differs from its TE declaration")
    # Keep the graph, compiled owner and shared library alive through graph replay.
    _OWNERS[key] = graph
    tvm_ffi.register_global_func(key, native, override=False)
    return " ".join(str(value) for value in metadata)
