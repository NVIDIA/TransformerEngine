# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cold preparation for TE's native cuDNN OSS THD forward dispatcher.

The complete compiled host entry launches metadata/remap, descriptors and SDPA.
Only immutable declaration arguments cross Python during preparation; C++ binds
the current pointers, workspace and stream on every execution.
"""

from importlib.metadata import PackageNotFoundError, version
import logging
import math

_LOG = logging.getLogger("transformer_engine.frost")
_OWNERS = {}
_ORDER = tuple(
    "q_ptr k_ptr v_ptr o_ptr lse_ptr sinks_ptr meta_ptr o_desc_ptr problem_size "
    "q_strides k_strides v_strides o_strides lse_strides lse_ext scale_softmax_log2 "
    "n_thd_units seq_q_lens_addr thd_q_lens_ptr thd_kv_lens_ptr thd_lens_form "
    "o_partial_ptr block_table_ptr block_table_v_ptr table_strides n_pages stream".split()
)
_POINTERS = dict(q_ptr=0, k_ptr=1, v_ptr=2, o_ptr=3, lse_ptr=4, thd_q_lens_ptr=5, thd_kv_lens_ptr=6)


def get_frost_attention_forward_constant(key, index):
    """Return one retained declaration value to the C++ cache, only at plan time."""
    return _OWNERS[key][1][index]


def _export(graph, batch, heads, kv_heads, sq, sk, tq, tk, scale, device):
    import tvm_ffi

    spec = graph._compiled_plans[graph._plan_index]._prepared.spec
    if (
        tuple(spec.order) != _ORDER
        or not isinstance(spec.fn, tvm_ffi.Function)
        or spec.quant is not None
        or spec.paged
        or spec.has_sink
        or spec.gate_expect is not None
        or not spec.has_lse
        or spec.lse_padded
        or spec.lse_head_major
        or spec.lse_stride_override
        or spec.lse_fill_plan is not None
        or spec.split_workspace is not None
        or spec.lens_form != 3
        or spec.n_q_lens != batch + 1
        or spec.n_kv_lens != batch + 1
        or spec.device_index != device
        or (spec.b, spec.qh, spec.kh, spec.d_qk, spec.d_v) != (batch, heads, kv_heads, 128, 128)
        or spec.total_q != tq
        or spec.total_kv != tk
        or spec.s_q_max != sq
    ):
        raise RuntimeError("Unsupported cuDNN OSS prepared forward contract")
    args = dict(zip(spec.order, spec.template))
    for role, h in (("q", heads), ("k", kv_heads), ("v", kv_heads), ("o", heads)):
        if spec.expect[role] not in ("bf16", "bfloat16"):
            raise RuntimeError("cuDNN OSS forward requires BF16 operands")
        if tuple(args[role + "_strides"]) != (h * 128, h * 128, 128):
            raise RuntimeError("cuDNN OSS forward strides differ from the TE declaration")
    if (
        args["lse_strides"] != (0, 0, 0)
        or args["lse_ext"] != 0
        or args["seq_q_lens_addr"] != 0
        or args["thd_lens_form"] != 3
        or args["table_strides"] != (0, 0)
        or args["n_pages"] != 0
        or not math.isclose(args["scale_softmax_log2"], scale * math.log2(math.e), rel_tol=1e-12)
        or any(
            args[name] is not None
            for name in ("sinks_ptr", "o_partial_ptr", "block_table_ptr", "block_table_v_ptr")
        )
    ):
        raise RuntimeError("Unsupported cuDNN OSS forward constant arguments")
    args["problem_size"] = (batch, heads, kv_heads, tq, tk, 0)
    args["sinks_ptr"] = 0  # Required pointer slot; FE binds absent sinks as address zero.
    # Match FE's native binder: capacity is fixed by the TE declaration; lengths
    # remain device data and may change on a retained graph.
    args["n_thd_units"] = min(args["n_thd_units"], ((tq - 1) // spec.cga_tile_m + batch) * heads)
    workspace = int(graph.get_workspace_size())
    alignment = int(spec.workspace_alignment)
    if not (0 <= spec.off_o_desc < spec.scratch_bytes <= workspace < 2**63):
        raise RuntimeError("Invalid cuDNN OSS forward workspace contract")
    if alignment < 16 or alignment & (alignment - 1):
        raise RuntimeError("Invalid cuDNN OSS forward workspace alignment")
    spans = (
        tq * heads * 128 * 2,
        tk * kv_heads * 128 * 2,
        tk * kv_heads * 128 * 2,
        tq * heads * 128 * 2,
        tq * heads * 4,
        *((batch + 1) * 4 for _ in range(4)),
    )
    if any(n <= 0 or n >= 2**63 for n in spans):
        raise RuntimeError("cuDNN OSS forward operand span is not representable")
    metadata = [1, workspace, alignment, len(_ORDER)]
    for i, span in enumerate(spans):
        metadata.extend((span, 16 if i < 4 else 4))
    constants = []
    for i, name in enumerate(_ORDER):
        # kind 0 = immutable value; 1 = current operand; 2 = workspace offset;
        # kind 3 = current stream. The C++ bridge never constructs FFI arrays.
        kind, value = 0, i
        if name in _POINTERS:
            kind, value = 1, _POINTERS[name]
        elif name in ("meta_ptr", "o_desc_ptr"):
            kind, value = 2, 0 if name == "meta_ptr" else int(spec.off_o_desc)
        elif name == "stream":
            kind, value = 3, 0
        metadata.extend((kind, value))
        constants.append(tvm_ffi.convert(args[name]) if kind == 0 else None)
    return spec.fn, tuple(constants), metadata


def get_frost_attention_forward(key, batch, heads, kv_heads, sq, sk, tq, tk, scale, device):
    """Compile a compact BF16 THD training declaration without allocating data."""
    try:
        import cudnn
        import tvm_ffi
        from cuda.bindings import runtime
        from packaging.version import Version

        if Version(version("nvidia-cutlass-dsl")) < Version("4.8.0"):
            raise ImportError("cuDNN OSS forward requires CuTeDSL >= 4.8.0")
        from cutlass.base_dsl.version_info import CUDA_VERSION

        if CUDA_VERSION.major < 13:
            raise ImportError("cuDNN OSS forward requires nvidia-cutlass-dsl[cu13]")
    except (ImportError, PackageNotFoundError) as error:
        _LOG.warning("cuDNN OSS forward dependency unavailable: %s", error)
        return False
    error, current = runtime.cudaGetDevice()
    if int(error) != 0 or int(current) != device:
        raise RuntimeError("cuDNN OSS preparation must use the calling CUDA device")
    io = cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(
        io_data_type=io,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    def attach(tensor, name, factor):
        origin = graph.tensor(
            name=name + "_origin",
            dim=[batch + 1, 1, 1, 1],
            stride=[1, 1, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        tensor.set_ragged_offset(origin).set_ragged_offset_multiplier(factor)

    refs = {}
    for role, h, s in (("q", heads, sq), ("k", kv_heads, sk), ("v", kv_heads, sk)):
        refs[role] = graph.tensor(
            name=role, dim=[batch, h, s, 128], stride=[s * h * 128, 128, h * 128, 1], data_type=io
        )
        attach(refs[role], role, h * 128)
    lengths = [
        graph.tensor(
            name="cu_" + role,
            dim=[batch + 1, 1, 1, 1],
            stride=[1, 1, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        for role in ("q", "kv")
    ]
    o, stats = graph.sdpa(
        **refs,
        generate_stats=True,
        attn_scale=scale,
        use_padding_mask=True,
        use_causal_mask=True,
        cu_seq_len_q=lengths[0],
        cu_seq_len_kv=lengths[1],
        max_total_seq_len_q=tq,
        max_total_seq_len_kv=tk
    )
    o.set_output(True).set_dim([batch, heads, sq, 128]).set_stride(
        [sq * heads * 128, 128, heads * 128, 1]
    )
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(
        [batch, heads, sq, 1]
    ).set_stride([sq * heads, 1, heads, 1])
    attach(o, "o", heads * 128)
    attach(stats, "stats", heads)
    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        index = next(
            (
                i
                for i in range(len(graph.plans))
                if graph.get_engine_and_knobs_at_index(i)[0] == 20511
            ),
            None,
        )
        if index is None:
            return False
        graph.select_plan(index)
        graph.build_plans()
    except (NotImplementedError, cudnn.cudnnGraphNotSupportedError) as error:
        _LOG.warning("cuDNN OSS forward declined this declaration: %s", error)
        return False
    native, constants, metadata = _export(
        graph, batch, heads, kv_heads, sq, sk, tq, tk, scale, device
    )
    _OWNERS[key] = (graph, constants)
    tvm_ffi.register_global_func(key, native, override=False)
    return " ".join(str(value) for value in metadata)
