# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Data-free tensor descriptions and explicit cuDNN graph build stages."""

import atexit
from collections import Counter
from dataclasses import dataclass
import importlib
import math
import os
import sys
import threading
import time


class UnsupportedGraph(ValueError):
    """A configuration rejected by cuDNN, rather than an execution/setup failure."""


def uses_ragged_attention_graph(sm_arch):
    """Use packed statistics on SM90+ except the SM12x family (encoded as 120-129)."""
    return sm_arch >= 90 and not 120 <= sm_arch < 130


def import_cudnn():
    """Import the optional graph frontend without importing either framework."""
    try:
        return importlib.import_module("cudnn")
    except ImportError as exc:
        raise ImportError(
            "cuDNN frontend Python package not found. "
            "Install it with: pip install nvidia-cudnn-frontend"
        ) from exc


def create_graph(cudnn, dtype, handle=None):
    """Create a graph with FP32 intermediate values and accumulation."""
    kwargs = {} if handle is None else {"handle": handle}
    return cudnn.pygraph(
        io_data_type=getattr(cudnn.data_type, dtype) if isinstance(dtype, str) else dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        **kwargs,
    )


def check_graph_support(cudnn, graph, run_stage=None):
    """Validate and select supported plans, without compiling their kernels."""
    run = run_stage or (lambda _name, function: function())
    try:
        run("validate", graph.validate)
        run("build_operation_graph", graph.build_operation_graph)
        run(
            "create_execution_plans",
            lambda: graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]),
        )
        run("check_support", graph.check_support)
    except cudnn.cudnnGraphNotSupportedError as exc:
        raise UnsupportedGraph(f"cuDNN SDPA graph is not supported: {exc}") from exc


def compile_graph(cudnn, graph, run_stage=None):
    """Compile the selected plan and return its nonempty workspace size."""
    run = run_stage or (lambda _name, function: function())
    run("build_plans", lambda: graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE))
    return max(int(graph.get_workspace_size()), 1)


@dataclass(frozen=True)
class TensorSpec:
    """A logical graph tensor, independent of its runtime buffer and framework."""

    dim: tuple
    stride: tuple
    dtype: str
    output: bool = False
    ragged_offset: str = ""
    reordered: bool = False
    by_value: bool = False


def apply_tensor_spec(cudnn, tensor, spec, uid=None):
    """Apply the same descriptor to an input or an operation-produced output."""
    tensor.set_dim(spec.dim).set_stride(spec.stride).set_data_type(
        getattr(cudnn.data_type, spec.dtype)
    )
    if uid is not None:
        tensor.set_uid(uid)
    if spec.output:
        tensor.set_output(True)
    return tensor


def create_tensor(cudnn, graph, name, spec, uid=None):
    """Create an input or an explicitly supplied output with a stable UID."""
    kwargs = {} if uid is None else {"uid": uid}
    tensor = graph.tensor(
        name=name,
        dim=spec.dim,
        stride=spec.stride,
        data_type=getattr(cudnn.data_type, spec.dtype),
        is_pass_by_value=spec.by_value,
        **kwargs,
    )
    if spec.reordered:
        tensor.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
    return apply_tensor_spec(cudnn, tensor, spec, uid)


def row_major_stride(shape):
    """Return contiguous storage strides without consulting framework objects."""
    return tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))


def round_up(value, multiple):
    """Round an integer up to a multiple."""
    return (value + multiple - 1) // multiple * multiple


def bucket(value, minimum, maximum):
    """Bucket positive capacities by powers of two, then multiples of maximum."""
    if value == 0:
        return 0
    if value <= maximum:
        return max(minimum, 1 << (value - 1).bit_length())
    return round_up(value, maximum)


def attention_stride(dim, fmt):
    """Describe contiguous attention storage in logical BHSD order."""
    b, h, s, d = dim
    if fmt in ("bshd", "thd"):
        return (s * h * d, d, h * d, 1)
    if fmt == "sbhd":
        return (h * d, d, b * h * d, 1)
    if fmt == "bhsd":
        return (h * s * d, s * d, d, 1)
    raise ValueError(f"Unsupported attention format: {fmt}")


def attention_matrix_spec(
    dim, fmt, dtype, *, storage_stride=None, packed="", output=False, ragged_offset=""
):
    """Describe either actual storage strides or a hypothetical support-query tensor."""
    if storage_stride is not None and fmt == "thd":
        token, head, last = storage_stride
        stride = (dim[2] * token, head, token, last)
    elif storage_stride is not None and len(storage_stride) == 4:
        order = {"bshd": (0, 2, 1, 3), "sbhd": (1, 2, 0, 3), "bhsd": (0, 1, 2, 3)}[fmt]
        stride = tuple(storage_stride[index] for index in order)
    else:
        stride = list(attention_stride(dim, fmt))
        if "3" in packed or "2" in packed:
            count = 3 if "3" in packed else 2
            stride[0] *= count
            stride[2] *= count
            if "h3d" in packed or "h2d" in packed:
                stride[1] *= count
        stride = tuple(stride)
    return TensorSpec(tuple(dim), stride, dtype, output=output, ragged_offset=ragged_offset)


def attention_stats_spec(batch, heads, seqlen, *, ragged=False, output=False):
    """Describe dense BHS1 or packed TH1 softmax statistics."""
    stride = (heads * seqlen, 1, heads, 1) if ragged else (heads * seqlen, seqlen, 1, 1)
    return TensorSpec(
        (batch, heads, seqlen, 1),
        stride,
        "FLOAT",
        output=output,
        ragged_offset="stats_offset" if ragged else "",
    )


def mxfp8_scale_spec(dim, fmt, columnwise=False):
    """Describe the reordered block-scale storage used by MXFP8 SDPA."""
    b, h, s, d = dim
    shape = (
        (b, h, round_up((s + 31) // 32, 4), round_up(d, 128))
        if columnwise
        else (b, h, round_up(s, 128), round_up((d + 31) // 32, 4))
    )
    return TensorSpec(shape, attention_stride(shape, fmt), "FP8_E8M0", reordered=True)


class CacheDiagnostics:
    """Optional cache counters and build timings shared by Python frontends."""

    def __init__(self, framework):
        self.framework = framework
        self.events = Counter()
        self.times = Counter()
        self._level = None
        atexit.register(self.summary)

    def level(self):
        """Return the debug level, resolved from the environment on first use."""
        if self._level is None:
            self._level = self._resolve_level()
        return self._level

    @staticmethod
    def _resolve_level():
        """Resolve the existing debug level and rank filter."""
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

    def event(self, backend, direction, device, event, key=None):
        """Record an event without retaining runtime tensors."""
        level = self.level()
        if not level:
            return
        self.events[(backend, direction, event)] += 1
        if level >= 2:
            print(
                f"[FUSED-ATTN-CACHE] tid={threading.get_ident()} dev={device} | "
                f"{backend} {direction} {event}"
                + (f" | {key!r}" if key is not None else ""),
                file=sys.stderr,
                flush=True,
            )

    def run(self, backend, direction, stage, function):
        """Time a build stage only when diagnostics are enabled."""
        if not self.level():
            return function()
        start = time.perf_counter()
        try:
            return function()
        finally:
            self.times[(backend, direction, stage, "seconds")] += time.perf_counter() - start
            self.times[(backend, direction, stage, "calls")] += 1

    def summary(self):
        """Print this frontend's collected counters and mean build times."""
        if not self.events:
            return
        print(f"[FUSED-ATTN-CACHE] {self.framework} Python frontend summary", file=sys.stderr)
        for (backend, direction, event), count in sorted(self.events.items()):
            print(f"  {backend} {direction} {event}={count}", file=sys.stderr)
        for backend, direction, stage, kind in sorted(self.times):
            if kind == "calls":
                calls = self.times[(backend, direction, stage, kind)]
                seconds = self.times[(backend, direction, stage, "seconds")]
                print(
                    f"  {backend} {direction} {stage}: {calls} calls, "
                    f"{1000 * seconds / calls:.3f} ms average",
                    file=sys.stderr,
                )
