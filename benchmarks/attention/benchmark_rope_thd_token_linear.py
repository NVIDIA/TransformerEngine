"""Microbenchmark for the token-linear THD fused RoPE path.

Holds the local packed-token count fixed and sweeps the number of packed
sequences. For each point, measures forward and backward latency of the fused
RoPE kernel under three regimes:

  * forced-old:  ``NVTE_FUSED_ROPE_THD_TOKEN_LINEAR=0``
  * forced-new:  ``NVTE_FUSED_ROPE_THD_TOKEN_LINEAR=1``
  * heuristic:   variable unset

Outputs a CSV and a PNG. Intended to be run on a single GPU; not distributed.
"""

from __future__ import annotations

import argparse
import csv
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch

from transformer_engine.pytorch.attention.rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)


@contextmanager
def env(name: str, value: str | None):
    prev = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def build_cu_seqlens(total_tokens: int, n_seqs: int, cp_size: int = 1) -> torch.Tensor:
    """Build a cu_seqlens whose local packed length equals ``total_tokens``.

    Per-sequence lengths are equal to ``total_tokens / n_seqs`` rounded down to
    a multiple of ``2 * cp_size``; any leftover tokens are tacked onto the last
    span so that the local total is exact.
    """
    pad = 2 * cp_size
    per = (total_tokens // n_seqs // pad) * pad
    if per <= 0:
        raise ValueError(
            f"n_seqs={n_seqs} is too large for total_tokens={total_tokens} with cp_size={cp_size}"
        )
    lengths = [per] * n_seqs
    deficit = total_tokens - per * n_seqs
    lengths[-1] += (deficit // pad) * pad
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32)


def time_fwd_bwd(
    fn,
    iters: int,
    warmup: int,
) -> tuple[float, float]:
    """Return (fwd_ms, fwd_plus_bwd_ms) averaged across ``iters`` iterations."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        out = fn()
        out.sum().backward()
    torch.cuda.synchronize()

    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)
    end_bwd = torch.cuda.Event(enable_timing=True)

    fwd_total = 0.0
    full_total = 0.0
    for _ in range(iters):
        start_fwd.record()
        out = fn()
        end_fwd.record()
        out.sum().backward()
        end_bwd.record()
        torch.cuda.synchronize()
        fwd_total += start_fwd.elapsed_time(end_fwd)
        full_total += start_fwd.elapsed_time(end_bwd)
    return fwd_total / iters, full_total / iters


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tokens", type=int, default=65536)
    parser.add_argument("--freqs-len", type=int, default=65536)
    parser.add_argument("--head-num", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--rotary-percent", type=float, default=1.0)
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument(
        "--n-seqs",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512, 1024, 2401],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("rope_thd_bench"))
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this benchmark")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda:0")

    rotary = RotaryPositionEmbedding(args.hidden, args.rotary_percent, interleaved=args.interleaved)
    freqs = rotary(args.freqs_len).to(device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "rope_thd_bench.csv"
    fields = [
        "n_seqs",
        "regime",
        "fwd_ms",
        "fwd_bwd_ms",
        "bwd_ms",
        "blocks_old",
        "blocks_new",
        "speedup_fwd_bwd_vs_old",
    ]
    rows = []

    print(
        f"# total_tokens={args.total_tokens} freqs_len={args.freqs_len} h={args.head_num} "
        f"d={args.hidden} dtype={args.dtype} cp={args.cp_size}"
    )
    print("# n_seqs regime fwd_ms fwd_bwd_ms bwd_ms blocks_old blocks_new speedup")

    by_nseq_old: dict[int, float] = {}

    for n_seqs in args.n_seqs:
        cu = build_cu_seqlens(args.total_tokens, n_seqs, cp_size=args.cp_size).to(device)
        actual_total = int(cu[-1].item())
        t = torch.rand(
            (actual_total // args.cp_size, args.head_num, args.hidden),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        def runner() -> torch.Tensor:
            # Reset grad in-place to keep the autograd graph fresh.
            if t.grad is not None:
                t.grad = None
            return apply_rotary_pos_emb(
                t,
                freqs,
                tensor_format="thd",
                fused=True,
                cu_seqlens=cu,
                interleaved=args.interleaved,
                cp_size=args.cp_size,
                cp_rank=0,
            )

        blocks_old = args.freqs_len * n_seqs
        blocks_new = actual_total // args.cp_size

        for regime, value in [("old", "0"), ("new", "1"), ("heuristic", None)]:
            with env("NVTE_FUSED_ROPE_THD_TOKEN_LINEAR", value):
                fwd_ms, full_ms = time_fwd_bwd(runner, iters=args.iters, warmup=args.warmup)
            bwd_ms = full_ms - fwd_ms
            if regime == "old":
                by_nseq_old[n_seqs] = full_ms
            speedup = (by_nseq_old[n_seqs] / full_ms) if regime != "old" else 1.0
            rows.append(
                {
                    "n_seqs": n_seqs,
                    "regime": regime,
                    "fwd_ms": f"{fwd_ms:.4f}",
                    "fwd_bwd_ms": f"{full_ms:.4f}",
                    "bwd_ms": f"{bwd_ms:.4f}",
                    "blocks_old": blocks_old,
                    "blocks_new": blocks_new,
                    "speedup_fwd_bwd_vs_old": f"{speedup:.3f}",
                }
            )
            print(
                f"{n_seqs:>6} {regime:>10}  fwd={fwd_ms:7.3f}  fwd_bwd={full_ms:7.3f}  "
                f"bwd={bwd_ms:7.3f}  blocks_old={blocks_old}  blocks_new={blocks_new}  "
                f"speedup={speedup:.2f}x"
            )

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    # Aggregate per regime.
    nseqs = sorted({int(r["n_seqs"]) for r in rows})
    by_regime = {regime: [] for regime in ("old", "new", "heuristic")}
    for n in nseqs:
        for regime in by_regime:
            for r in rows:
                if int(r["n_seqs"]) == n and r["regime"] == regime:
                    by_regime[regime].append(float(r["fwd_bwd_ms"]))
                    break

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for regime in ("old", "new", "heuristic"):
        ax.plot(nseqs, by_regime[regime], marker="o", label=regime)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_seqs (packed spans, log)")
    ax.set_ylabel("fwd + bwd latency (ms, log)")
    ax.set_title(
        f"Fused THD RoPE latency vs n_seqs\nT_local={args.total_tokens}, "
        f"freqs_len={args.freqs_len}, h={args.head_num}, d={args.hidden}, "
        f"{args.dtype}, cp={args.cp_size}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    speedup_new = [by_regime["old"][i] / by_regime["new"][i] for i in range(len(nseqs))]
    ax.plot(nseqs, speedup_new, marker="o", color="tab:green", label="new vs old")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("n_seqs (log)")
    ax.set_ylabel("speedup (old / new)")
    ax.set_title("Token-linear speedup over (s × b) launch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    png_path = args.out_dir / "rope_thd_bench.png"
    fig.savefig(png_path, dpi=120)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
