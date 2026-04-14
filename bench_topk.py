"""
Benchmark jax.lax.top_k, torch.topk, CUB TopK (TransformerEngine PR #2784),
AIR TopK, and topk_per_row for bfloat16 inputs across the (N, K)
configurations from the reference sheet.

Note: AIR TopK and topk_per_row are float32-only.  Inputs are cast to float32
for those kernels; the cast cost is NOT included in the reported time.

The two custom kernels live in topk/ inside this repo.  They are built
automatically on first run if the .so files are not present.
"""

import os
import subprocess
import sys

# Ensure the repo root is on the path for the TE import.
_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)

# Build topk/ extensions if not already built.
_topk_dir = os.path.join(_repo_root, "topk")
_so_names = ["air_topk_wrapper", "topk_per_row"]
_missing = [
    n
    for n in _so_names
    if not any(f.startswith(n) and f.endswith(".so") for f in os.listdir(_topk_dir))
]
if _missing:
    print(f"Building topk extensions ({', '.join(_missing)}) ...")
    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=_topk_dir,
    )
sys.path.insert(0, _topk_dir)

import time
import torch
import jax
import jax.numpy as jnp
import numpy as np

from transformer_engine.jax.cpp_extensions.cub import topk as te_cub_topk
import topk_per_row
import air_topk_wrapper

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DTYPE_JAX = jnp.bfloat16
DTYPE_TORCH = torch.bfloat16

WARMUP_ITERS = 50
BENCH_ITERS = 200

CONFIGS = [
    # (N, K)
    (100_000, 100),
    (100_000, 500),
    (100_000, 1_000),
    (500_000, 500),
    (500_000, 1_000),
    (500_000, 2_000),
    (1_000_000, 1_000),
    (1_000_000, 2_000),
    (1_000_000, 5_000),
    (2_000_000, 1_000),
    (2_000_000, 2_000),
    (2_000_000, 5_000),
    (5_000_000, 1_000),
    (5_000_000, 2_000),
    (5_000_000, 5_000),
]

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def bench_torch(x_torch: torch.Tensor, k: int) -> float:
    """Median latency in ms for torch.topk (bfloat16)."""
    for _ in range(WARMUP_ITERS):
        torch.topk(x_torch, k)
    torch.cuda.synchronize()

    times = []
    for _ in range(BENCH_ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.topk(x_torch, k)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


def bench_jax_lax(x_jax: jnp.ndarray, k: int) -> float:
    """Median latency in ms for jax.lax.top_k (bfloat16)."""
    f = jax.jit(jax.lax.top_k, static_argnums=(1,))
    for _ in range(WARMUP_ITERS):
        f(x_jax, k)[0].block_until_ready()

    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        f(x_jax, k)[0].block_until_ready()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def bench_cub(x_jax: jnp.ndarray, k: int) -> float:
    """Median latency in ms for CUB TopK via TransformerEngine JAX FFI (bfloat16)."""
    f = jax.jit(te_cub_topk, static_argnums=(1,))
    for _ in range(WARMUP_ITERS):
        f(x_jax, k)[0].block_until_ready()

    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        f(x_jax, k)[0].block_until_ready()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def bench_air_topk(
    x_f32: torch.Tensor,
    lengths: torch.Tensor,
    buffer: torch.Tensor,
    out_indices: torch.Tensor,
    k: int,
) -> float:
    """Median latency in ms for AIR TopK (float32)."""
    for _ in range(WARMUP_ITERS):
        air_topk_wrapper.topk_kernel(x_f32, lengths, k, buffer, out_indices, False)
    torch.cuda.synchronize()

    times = []
    for _ in range(BENCH_ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        air_topk_wrapper.topk_kernel(x_f32, lengths, k, buffer, out_indices, False)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


def bench_topk_per_row(
    x_f32: torch.Tensor,
    lengths: torch.Tensor,
    out_aux: torch.Tensor,
    logits_aux: torch.Tensor,
    out_indices: torch.Tensor,
    k: int,
) -> float | None:
    """Median latency in ms for topk_per_row (float32).

    Buffers are pre-allocated outside the timed loop.  The input is already
    float32 so no dtype conversion cost is measured.

    Returns None when the kernel launch fails (e.g. shared-memory limit
    exceeded for large K on the multi-block code path).
    """
    # Single probe: catch smem-limit / invalid-argument errors before the
    # timed loop so CUDA state stays clean.
    try:
        topk_per_row.topk_kernel(x_f32, lengths, k, out_aux, logits_aux, out_indices, False)
        torch.cuda.synchronize()
    except RuntimeError:
        torch.cuda.synchronize()  # drain any pending CUDA work
        return None

    for _ in range(WARMUP_ITERS - 1):
        topk_per_row.topk_kernel(x_f32, lengths, k, out_aux, logits_aux, out_indices, False)
    torch.cuda.synchronize()

    times = []
    for _ in range(BENCH_ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        topk_per_row.topk_kernel(x_f32, lengths, k, out_aux, logits_aux, out_indices, False)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  |  JAX: {jax.__version__}")
    print(f"Warmup: {WARMUP_ITERS}  |  Bench: {BENCH_ITERS} iters  |  Metric: median ms")
    print(f"topk_per_row dtype: float32 (kernel is float32-only; input cast pre-benchmark)\n")

    col = [10, 10, 8, 16, 12, 12, 14, 18]
    header = (
        f"{'dtype':<{col[0]}} {'N':>{col[1]}} {'K':>{col[2]}}"
        f" {'jax.lax.top_k':>{col[3]}} {'torch.topk':>{col[4]}}"
        f" {'CUB TopK':>{col[5]}} {'AIR TopK':>{col[6]}} {'topk_per_row':>{col[7]}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    rng = jax.random.PRNGKey(42)

    for N, K in CONFIGS:
        # --- shared input data ---
        x_jax = jax.random.uniform(rng, shape=(N,), dtype=DTYPE_JAX)
        x_jax.block_until_ready()
        x_torch = torch.from_dlpack(x_jax).clone()  # bfloat16, 1-D

        # float32 view for float32-only kernels: [1, N]
        x_f32 = x_torch.float().unsqueeze(0).contiguous()
        lengths = torch.tensor([N], dtype=torch.int32, device="cuda")
        out_aux, logits_aux, out_indices_pr = topk_per_row.allocate_buffers(x_f32, K)
        air_buf, out_indices_air = air_topk_wrapper.allocate_buffers(x_f32, K, False)

        t_jax = bench_jax_lax(x_jax, K)
        t_tor = bench_torch(x_torch, K)
        t_cub = bench_cub(x_jax, K)
        t_air = bench_air_topk(x_f32, lengths, air_buf, out_indices_air, K)
        t_pr = bench_topk_per_row(x_f32, lengths, out_aux, logits_aux, out_indices_pr, K)
        pr_str = f"{t_pr:>{col[7]}.4f}" if t_pr is not None else f"{'N/A':>{col[7]}}"

        print(
            f"{'bfloat16':<{col[0]}} {N:>{col[1]},} {K:>{col[2]},}"
            f" {t_jax:>{col[3]}.4f} {t_tor:>{col[4]}.4f}"
            f" {t_cub:>{col[5]}.4f} {t_air:>{col[6]}.4f} {pr_str}"
        )

    print(sep)
    print("All times in milliseconds (median).")
    print("AIR TopK and topk_per_row columns use float32 input; all others use bfloat16.")


if __name__ == "__main__":
    main()
