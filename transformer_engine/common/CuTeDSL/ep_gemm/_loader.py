# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Loading of the EP grouped GEMMs: kernel tree, fake operands and TVM-FFI registration.

The kernels are loaded from ``NVTE_MOE_OVERLAP`` (a ``gemm/`` tree with ``blackwell/``,
``rubin/`` and ``utils/``), compiled once per configuration and registered under a name chosen by
C++.
"""

import functools
import importlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import tvm_ffi

from ..utils import device_compute_capability, tvm_ffi_function

logger = logging.getLogger(__name__)

GEMM_PATH_ENV = "NVTE_MOE_OVERLAP"
MMA_TILER_MN = (256, 256)
CLUSTER_MN = (2, 1)
# Rows per M tile. A smaller expert alignment ends an expert in a tail tile.
ROWS_PER_TILE = MMA_TILER_MN[0] // 2 * CLUSTER_MN[0]
# Values per MXFP8 E8M0 scale.
SF_VEC_SIZE = 32


@dataclass(frozen=True)
class GemmArch:
    """The kernels of one architecture: their subdirectories of the GEMM path, searched in order,
    and the BF16 kernels as ``module.Class``."""

    subdirs: tuple
    dispatch_fwd: str  # FC1 + SwiGLU
    dispatch_bwd: str  # FC2 data gradient + SwiGLU backward
    combine: str  # FC2, FC1 data gradient
    mxfp8_tail_tiles: bool  # MXFP8 kernels for experts padded to fewer rows than a tile
    mxfp8_modules: tuple  # modules of the MXFP8 kernels


_SM100 = GemmArch(
    subdirs=("blackwell",),
    dispatch_fwd="moe_grouped_gemm_glu_sm100.MoEGroupedGemmGluBiasBf16Kernel",
    dispatch_bwd="moe_grouped_gemm_dglu_dbias.MoEGroupedGemmDgluDbiasBf16Kernel",
    combine="moe_grouped_gemm_sm100.MoEGroupedGemmBf16Kernel",
    mxfp8_tail_tiles=True,
    mxfp8_modules=(
        "moe_blockscaled_grouped_gemm_quant_sm100",
        "moe_blockscaled_grouped_gemm_dglu_dbias",
        "moe_blockscaled_grouped_gemm_wgrad",
        "moe_blockscaled_grouped_gemm_wgrad_dense",
        "moe_blockscaled_grouped_gemm_wgrad_reuse",
    ),
)
# The SM 10.7 kernels build on the SM 10.0 ones.
_SM107 = GemmArch(
    subdirs=("rubin", "blackwell"),
    dispatch_fwd="moe_grouped_gemm_glu_sm107.MoEGroupedGemmGluBiasBf16Kernel",
    dispatch_bwd="moe_grouped_gemm_sm107.MoEGroupedGemmDgluDbiasBf16Kernel",
    combine="moe_grouped_gemm_sm107.MoEGroupedGemmBf16Kernel",
    mxfp8_tail_tiles=False,
    mxfp8_modules=(
        "moe_blockscaled_grouped_gemm_quant_rubin",
        "moe_blockscaled_grouped_gemm_dglu_rubin",
        "moe_blockscaled_grouped_gemm_glu_rubin",
        "moe_blockscaled_grouped_gemm_wgrad_rubin",
        "moe_blockscaled_grouped_gemm_wgrad_reuse_rubin",
    ),
)
# Kernels by compute capability; other architectures run the unfused ops.
GEMM_ARCHS = {(10, 0): _SM100, (10, 3): _SM100, (10, 7): _SM107}


def gemm_arch():
    """The kernels of the current device (the SM 10.0 ones on an unsupported device)."""
    return GEMM_ARCHS.get(device_compute_capability(), _SM100)


def is_sm107():
    """Whether the current device runs the SM 10.7 kernels."""
    return gemm_arch() is _SM107


_LOADED_ROOT: list = []

# What TE reads from the kernel tree, module by module (dotted names are nested attributes).
_SHARED_INTERFACE = {
    "moe_utils": ("MoEWeightMode.DENSE", "WGradInputOrder.Tensor2D", "WgradTileClaim.COUNTER"),
    "moe_tail_tiles": ("FUSED_TAIL_MAX_ROWS",),
    "moe_signal_helpers": ("signal_workspace_offsets",),
    "moe_streaming_output_chunk_tile_scheduler": (
        "OutputNotifyConfig",
        "OutputNotifyMode.N_CHUNK",
        "output_notify_capacity",
        "output_notify_offsets",
    ),
}
_WGRAD_CLASSES = ("MoEGroupedGemmWgradBF16Kernel", "MoEGroupedGemmWgradWideBF16Kernel")


def _missing(obj, dotted_names):
    """The names of ``dotted_names`` that ``obj`` lacks."""
    missing = []
    for dotted in dotted_names:
        target = obj
        for part in dotted.split("."):
            target = getattr(target, part, None)
        if target is None:
            missing.append(dotted)
    return missing


@functools.lru_cache(maxsize=None)
def _gemm_interface_reason(root, capability, mxfp8):
    del root, capability  # The cache key: the tree and the architecture the checks ran on.
    problems = []
    try:
        for name, attributes in _SHARED_INTERFACE.items():
            missing = _missing(import_gemm_modules(name)[0], attributes)
            if missing:
                problems.append(f"{name} lacks {', '.join(missing)}")
        arch = gemm_arch()
        for role in ("dispatch_fwd", "dispatch_bwd", "combine"):
            module_name, class_name = getattr(arch, role).split(".")
            kernel = getattr(import_gemm_modules(module_name)[0], class_name, None)
            if kernel is None:
                problems.append(f"{module_name} lacks {class_name}")
            elif role == "dispatch_fwd" and not hasattr(kernel, "FC1_TAIL_AT_SLOTS_READY"):
                problems.append(f"{class_name} lacks FC1_TAIL_AT_SLOTS_READY")
        if mxfp8:
            for module_name in arch.mxfp8_modules:
                import_gemm_modules(module_name)
        names = (
            ("moe_grouped_gemm_sm107",) * 2
            if is_sm107()
            else ("moe_grouped_gemm_wgrad", "moe_grouped_gemm_wgrad_wide")
        )
        for module_name, class_name in zip(names, _WGRAD_CLASSES):
            if getattr(import_gemm_modules(module_name)[0], class_name, None) is None:
                problems.append(f"{module_name} lacks {class_name}")
    except ImportError as e:
        problems.append(f"cannot import {e.name or e}")
    if problems:
        return "the EP grouped GEMM kernel tree does not match Transformer Engine: " + "; ".join(
            problems
        )
    return None


# TODO(phuonguyen): check the kernel tree's version (cudnn-frontend release or EPCompOverlap
# commit) against a minimum once the kernels ship in a versioned package; for now the tree is
# only checked for the names TE reads.
def gemm_interface_reason(mxfp8=False) -> Optional[str]:
    """Why the kernel tree at ``NVTE_MOE_OVERLAP`` lacks what the EP grouped GEMMs read, or None;
    ``mxfp8`` also checks the MXFP8 kernels. Checked once per tree, architecture and recipe; the
    tree's version is not otherwise identified."""
    return _gemm_interface_reason(os.environ.get(GEMM_PATH_ENV), device_compute_capability(), mxfp8)


def import_gemm_modules(*names):
    """Import the named modules of the kernel tree (and ``moe_utils`` last) from the GEMM path."""
    root = os.environ.get(GEMM_PATH_ENV)
    if not root:
        raise ModuleNotFoundError(
            f"{GEMM_PATH_ENV} is not set: the EP grouped GEMMs are not available"
        )
    if _LOADED_ROOT and _LOADED_ROOT[0] != root:
        # The modules are imported by name, so those of the first tree stay in sys.modules.
        raise RuntimeError(
            f"{GEMM_PATH_ENV} changed from {_LOADED_ROOT[0]} to {root} after the EP grouped GEMMs"
            " were loaded; restart the process to use the other kernel tree."
        )
    _LOADED_ROOT[:] = [root]
    for sub in reversed(("", *gemm_arch().subdirs, "utils")):
        path = os.path.join(root, sub) if sub else root
        if path not in sys.path:
            sys.path.insert(0, path)
    importlib.invalidate_caches()
    return tuple(importlib.import_module(name) for name in (*names, "moe_utils"))


def import_bf16_kernel(role):
    """The current device's BF16 kernel class of ``role`` (a ``GemmArch`` field) and
    ``moe_utils``."""
    module_name, class_name = getattr(gemm_arch(), role).split(".")
    module, moe_utils = import_gemm_modules(module_name)
    return getattr(module, class_name), moe_utils


def unsupported_device_reason(what):
    """Why ``what`` cannot run on the current device, or None."""
    if device_compute_capability() in GEMM_ARCHS:
        return None
    supported = ", ".join(f"{major}.{minor}" for major, minor in GEMM_ARCHS)
    major, minor = device_compute_capability()
    return f"The {what} needs SM {supported}, got SM {major}.{minor}."


def device_smem_optin():
    """Opt-in shared memory per block of the current CUDA device, in bytes."""
    from cuda.bindings import runtime as cudart  # pylint: disable=import-outside-toplevel

    err, device = cudart.cudaGetDevice()
    assert err == cudart.cudaError_t.cudaSuccess, err
    err, smem = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, device
    )
    assert err == cudart.cudaError_t.cudaSuccess, err
    return smem


def fake_tensor(dtype, shape, order, align):
    """A compact fake gmem tensor to compile against."""
    from cutlass import cute  # pylint: disable=import-outside-toplevel

    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=order, memspace=cute.AddressSpace.gmem, assumed_align=align
    )


def rows_view(t, rows, cols):
    """Row-major [rows, cols] operand as the kernel's (rows, cols, 1) tensor."""
    from cutlass import cute  # pylint: disable=import-outside-toplevel

    return cute.make_tensor(
        t.iterator, cute.make_layout((rows, cols, 1), stride=(cols, 1, rows * cols))
    )


def weight_view(b, n, k, experts, n_major):
    """Weight [experts, n, k], or N-major [experts, k, n], as the kernel's (n, k, experts) tensor."""
    from cutlass import cute  # pylint: disable=import-outside-toplevel

    stride = (1, n, k * n) if n_major else (k, 1, n * k)
    return cute.make_tensor(b.iterator, cute.make_layout((n, k, experts), stride=stride))


def compile_launch(launch, *fakes):
    """Compile ``launch`` with TVM-FFI against fake operands and a fake stream."""
    from cutlass import cute  # pylint: disable=import-outside-toplevel

    return cute.compile(launch, *fakes, cute.runtime.make_fake_stream(), options="--enable-tvm-ffi")


def prob_view(prob, rows):
    """[rows] probabilities as the kernel's (rows, 1, 1) tensor."""
    from cutlass import cute  # pylint: disable=import-outside-toplevel

    return cute.make_tensor(prob.iterator, cute.make_layout((rows, 1, 1), stride=(1, 1, 1)))


def fp8_fake(shape, order):
    """A fake E4M3 operand."""
    import cutlass  # pylint: disable=import-outside-toplevel

    return fake_tensor(cutlass.Float8E4M3FN, shape, order, 16)


def scale_fake(rows, cols):
    """Fake E8M0 scales of a [rows, cols] MXFP8 operand, compact or GEMM-swizzled."""
    import cutlass  # pylint: disable=import-outside-toplevel

    return fake_tensor(cutlass.Float8E8M0FNU, (rows * cols // SF_VEC_SIZE,), (0,), 16)


def blockscaled_kwargs(num_experts):
    """Constructor arguments shared by the MXFP8 kernels."""
    import cutlass  # pylint: disable=import-outside-toplevel

    moe_utils = import_gemm_modules()[0]
    return {
        "sf_vec_size": SF_VEC_SIZE,
        "acc_dtype": cutlass.Float32,
        "use_2cta_instrs": True,
        "mma_tiler_mn": MMA_TILER_MN,
        "cluster_shape_mn": CLUSTER_MN,
        "vectorized_f32": False,
        "expert_cnt": num_experts,
        "weight_mode": moe_utils.MoEWeightMode.DENSE,
        "use_dynamic_sched": True,
    }


def tail_tile_kwargs(rows_alignment, fused=False):
    """Constructor arguments for experts padded to ``rows_alignment`` rows; ``fused`` runs the
    smallest tails inside their expert's last full tile (BF16 forward GEMMs)."""
    if rows_alignment >= ROWS_PER_TILE:
        return {}
    kwargs = {"tail_rows_alignment": rows_alignment}
    if fused:
        tail_tiles = import_gemm_modules("moe_tail_tiles")[0]
        kwargs["fused_tail_max_rows"] = tail_tiles.FUSED_TAIL_MAX_ROWS
    return kwargs


def mxfp8_tail_tile_kwargs(rows_alignment):
    """``tail_tile_kwargs`` of the MXFP8 kernels."""
    if gemm_arch().mxfp8_tail_tiles:
        return tail_tile_kwargs(rows_alignment)
    if rows_alignment < ROWS_PER_TILE:
        raise ValueError(
            f"The MXFP8 kernels of this device need experts padded to {ROWS_PER_TILE} rows."
        )
    return {}


def tail_fakes(rows, experts, ws_bytes):
    """Fake row ends, alpha, prob and workspace operands shared by every launch."""
    import cutlass  # pylint: disable=import-outside-toplevel

    return (
        fake_tensor(cutlass.Int32, (experts,), (0,), 16),
        fake_tensor(cutlass.Float32, (experts,), (0,), 16),
        fake_tensor(cutlass.Float32, (rows,), (0,), 16),
        fake_tensor(cutlass.Uint8, (ws_bytes,), (0,), 128),
    )


def register_function(fn_name, compile_fn, what) -> bool:
    """Compile and register ``fn_name`` unless registered; return whether it is."""
    try:
        if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
            return True
        if (reason := unsupported_device_reason(what)) is not None:
            logger.warning("%s", reason)
            return False
        tvm_ffi.register_global_func(fn_name, tvm_ffi_function(compile_fn()), override=True)
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("The %s is not available: %s", what, e)
        return False
