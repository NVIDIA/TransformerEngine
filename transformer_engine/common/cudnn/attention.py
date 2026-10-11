# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Construct SDPA graphs from immutable descriptors, without framework tensors."""

from dataclasses import dataclass

from .graph import TensorSpec, apply_tensor_spec, create_tensor


@dataclass(frozen=True)
class AttentionOptions:
    """Normalized attributes that affect graph construction and support."""

    backward: bool = False
    fp8: bool = False
    mxfp8: bool = False
    attn_scale: str | float = "attn_scale"
    bottom_right: bool = False
    left_bound: int | None = None
    right_bound: int | None = None
    padding: bool = False
    alibi: bool = False
    dropout: float = 0.0
    deterministic: bool = False
    generate_stats: bool = True
    paged_max_seqlen_kv: int = 0
    max_total_seqlen_q: int = 0
    max_total_seqlen_kv: int = 0


@dataclass(frozen=True)
class AttentionGraphSpec:
    """A cacheable graph description; bindings and scalar values live in adapters."""

    options: AttentionOptions
    tensors: tuple

    @classmethod
    def from_dict(cls, options, tensors):
        """Canonicalize tensor order for both UIDs and graph cache keys."""
        return cls(options, tuple(sorted(tensors.items())))


@dataclass
class BuiltAttentionGraph:
    """A live graph and its tensor handles, without any runtime buffers."""

    graph: object
    tensors: dict
    uids: dict


def build_attention_graph(cudnn, graph, spec, *, score_mod=None, score_mod_bprop=None):
    """Build ordinary, FP8, MXFP8, or score-modified SDPA using logical bindings."""
    specs = dict(spec.tensors)
    tensors = {}
    uids = {name: index + 1 for index, (name, _) in enumerate(spec.tensors)}

    def tensor(name):
        if name not in tensors:
            desc = create_tensor(cudnn, graph, name, specs[name], uids[name])
            if specs[name].ragged_offset:
                desc.set_ragged_offset(tensor(specs[name].ragged_offset))
            tensors[name] = desc
        return tensors[name]

    for name, desc in spec.tensors:
        if not desc.output:
            tensor(name)

    options = spec.options
    kwargs = {
        "attn_scale": (
            tensor(options.attn_scale)
            if isinstance(options.attn_scale, str)
            else options.attn_scale
        ),
        "diagonal_alignment": (
            cudnn.diagonal_alignment.BOTTOM_RIGHT
            if options.bottom_right
            else cudnn.diagonal_alignment.TOP_LEFT
        ),
    }
    bound_names = (
        ("left_bound", "right_bound")
        if options.fp8 and (not options.mxfp8 or options.backward)
        else ("diagonal_band_left_bound", "diagonal_band_right_bound")
    )
    for name, bound in zip(bound_names, (options.left_bound, options.right_bound)):
        if bound is not None:
            kwargs[name] = bound
    if options.padding:
        kwargs.update(use_padding_mask=True, seq_len_q=tensor("seq_q"), seq_len_kv=tensor("seq_kv"))
    if options.alibi:
        kwargs["use_alibi_mask"] = True
    if "bias" in specs:
        kwargs["bias"] = tensor("bias")
        if options.backward and "dbias" in specs:
            kwargs["dBias"] = tensor("dbias")
    if options.dropout:
        kwargs["dropout"] = (options.dropout, tensor("seed"), tensor("offset"))
    if "sink" in specs:
        kwargs["sink_token"] = tensor("sink")
        if options.backward:
            kwargs["dSink_token"] = tensor("dsink")
    if score_mod is not None:
        kwargs["score_mod"] = score_mod
    if score_mod_bprop is not None:
        kwargs["score_mod_bprop"] = score_mod_bprop
    if options.backward:
        kwargs["use_deterministic_algorithm"] = options.deterministic
        if options.max_total_seqlen_q:
            kwargs["max_total_seq_len_q"] = options.max_total_seqlen_q
            kwargs["max_total_seq_len_kv"] = options.max_total_seqlen_kv
    else:
        kwargs["generate_stats"] = options.generate_stats
        if "max" in specs:
            kwargs["score_max"] = tensor("max")
        if options.paged_max_seqlen_kv:
            kwargs.update(
                paged_attention_k_table=tensor("page_table_k"),
                paged_attention_v_table=tensor("page_table_v"),
                paged_attention_max_seq_len_kv=options.paged_max_seqlen_kv,
            )

    q, k, v = (tensor(name) for name in ("q", "k", "v"))
    if not options.fp8:
        if options.backward:
            outputs = graph.sdpa_backward(
                q, k, v, tensor("o"), tensor("do"), tensor("stats"), **kwargs
            )
            names = ("dq", "dk", "dv")
        else:
            outputs = graph.sdpa(q, k, v, **kwargs)
            names = ("o", "stats")
    elif not options.backward:
        scales = [tensor(name) for name in ("descale_q", "descale_k", "descale_v")]
        if options.mxfp8:
            outputs = graph.sdpa_mxfp8(q, k, v, *scales, **kwargs)
            names = ("o", "stats", "amax_o")
        else:
            outputs = graph.sdpa_fp8(
                q,
                k,
                v,
                *scales,
                *(tensor(name) for name in ("descale_s", "scale_s", "scale_o")),
                **kwargs
            )
            names = ("o", "stats", "amax_s", "amax_o")
    elif options.mxfp8:
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
                tensor(name)
                for name in (
                    "descale_q",
                    "descale_q_t",
                    "descale_k",
                    "descale_k_t",
                    "descale_v",
                    "descale_do",
                    "descale_do_t",
                )
            ),
            **kwargs
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
                tensor(name)
                for name in (
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
            **kwargs
        )
        names = ("dq", "dk", "dv", "amax_dq", "amax_dk", "amax_dv", "amax_dp")

    for name, desc in zip(names, outputs):
        if name not in specs:
            # Unbound MXFP8 amax outputs stay virtual. Their descriptors are still required.
            if desc is not None:
                apply_tensor_spec(cudnn, desc, TensorSpec((1, 1, 1, 1), (1, 1, 1, 1), "FLOAT"))
            continue
        apply_tensor_spec(cudnn, desc, specs[name], uids[name])
        if specs[name].ragged_offset:
            desc.set_ragged_offset(tensor(specs[name].ragged_offset))
        tensors[name] = desc
    return BuiltAttentionGraph(graph, tensors, uids)
