# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Framework-neutral helpers for cuDNN FP8 attention graph construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .cudnn import round_up


@dataclass(frozen=True)
class FP8AttentionGraphConfig:
    """Static choices that select a cuDNN FP8 attention graph family."""

    mode: str
    name: str

    def __post_init__(self):
        if self.mode not in ("delayed", "current", "mxfp8"):
            raise ValueError(f"Unknown FP8 attention scaling mode {self.mode!r}.")

    @property
    def is_mxfp8(self) -> bool:
        """Return whether the graph uses microscaling FP8 nodes."""

        return self.mode == "mxfp8"


def attention_format_stride(
    batch: int, heads: int, seqlen: int, dim: int, tensor_format: str
) -> tuple[int, int, int, int]:
    """Describe a contiguous TE attention tensor as logical BHSD."""

    if tensor_format in ("bshd", "thd"):
        return (seqlen * heads * dim, dim, heads * dim, 1)
    if tensor_format == "sbhd":
        return (heads * dim, dim, batch * heads * dim, 1)
    if tensor_format == "bhsd":
        return (heads * seqlen * dim, seqlen * dim, dim, 1)
    raise ValueError(f"Unsupported FP8 tensor format {tensor_format!r}.")


def mxfp8_padded_sizes(s_q: int, s_kv: int, d_qk: int, d_v: int) -> dict[str, int]:
    """Return the padded data and E8M0-scale dimensions required by cuDNN MXFP8."""

    return {
        "s_q_padded": round_up(s_q, 128),
        "s_kv_padded": round_up(s_kv, 128),
        "s_q_scale_padded": round_up((s_q + 31) // 32, 4),
        "s_kv_scale_padded": round_up((s_kv + 31) // 32, 4),
        "d_qk_padded": round_up(d_qk, 128),
        "d_v_padded": round_up(d_v, 128),
        "d_qk_scale_padded": round_up((d_qk + 31) // 32, 4),
        "d_v_scale_padded": round_up((d_v + 31) // 32, 4),
    }


def build_fp8_forward_operation(
    graph: Any,
    tensors: Mapping[str, Any],
    options: Mapping[str, Any],
    config: FP8AttentionGraphConfig,
) -> dict[str, Any]:
    """Add the selected FP8 SDPA forward operation to a cuDNN frontend graph."""

    kwargs = dict(options)
    if config.is_mxfp8:
        output, stats, amax_o = graph.sdpa_mxfp8(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            tensors["descale_q"],
            tensors["descale_k"],
            tensors["descale_v"],
            name=config.name,
            **kwargs,
        )
        return {"output": output, "stats": stats, "amax_o": amax_o}

    output, stats, amax_s, amax_o = graph.sdpa_fp8(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["descale_q"],
        tensors["descale_k"],
        tensors["descale_v"],
        tensors["descale_s"],
        tensors["scale_s"],
        tensors["scale_o"],
        name=config.name,
        **kwargs,
    )
    return {
        "output": output,
        "stats": stats,
        "amax_s": amax_s,
        "amax_o": amax_o,
    }


def build_fp8_backward_operation(
    graph: Any,
    tensors: Mapping[str, Any],
    options: Mapping[str, Any],
    config: FP8AttentionGraphConfig,
) -> dict[str, Any]:
    """Add the selected FP8 SDPA backward operation to a cuDNN frontend graph."""

    kwargs = dict(options)
    if config.is_mxfp8:
        outputs = graph.sdpa_mxfp8_backward(
            tensors["q"],
            tensors["q_t"],
            tensors["k"],
            tensors["k_t"],
            tensors["v"],
            tensors["o"],
            tensors["do_f16"],
            tensors["do"],
            tensors["do_t"],
            tensors["stats"],
            tensors["descale_q"],
            tensors["descale_q_t"],
            tensors["descale_k"],
            tensors["descale_k_t"],
            tensors["descale_v"],
            tensors["descale_do"],
            tensors["descale_do_t"],
            name=config.name,
            **kwargs,
        )
        dq, dk, dv, *amax = outputs
        return {"dq": dq, "dk": dk, "dv": dv, "amax": tuple(amax)}

    outputs = graph.sdpa_fp8_backward(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["o"],
        tensors["do"],
        tensors["stats"],
        tensors["descale_q"],
        tensors["descale_k"],
        tensors["descale_v"],
        tensors["descale_o"],
        tensors["descale_do"],
        tensors["descale_s"],
        tensors["descale_dp"],
        tensors["scale_s"],
        tensors["scale_dq"],
        tensors["scale_dk"],
        tensors["scale_dv"],
        tensors["scale_dp"],
        name=config.name,
        **kwargs,
    )
    dq, dk, dv, amax_dq, amax_dk, amax_dv, amax_dp = outputs
    return {
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "amax_dq": amax_dq,
        "amax_dk": amax_dk,
        "amax_dv": amax_dv,
        "amax_dp": amax_dp,
    }
