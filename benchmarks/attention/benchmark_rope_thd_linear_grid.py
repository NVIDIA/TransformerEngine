"""Microbenchmark for the THD linear-grid fused RoPE path.

Holds the local packed-token count fixed and sweeps the number of packed
sequences. For each point, measures forward and backward latency of the fused
RoPE kernel under three regimes:

  * forced-old:  ``NVTE_FUSED_ROPE_THD_LINEAR_GRID=0``
  * forced-new:  ``NVTE_FUSED_ROPE_THD_LINEAR_GRID=1``
  * heuristic:   variable unset

Outputs a CSV. Intended to be run on a single GPU; not distributed.
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


def build_cu_seqlens(local_total_tokens: int, n_seqs: int, cp_size: int = 1) -> torch.Tensor:
    """Build cu_seqlens whose local packed length is ``local_total_tokens``.

    `cu_seqlens` stores global cumulative positions. Each sequence's local
    length after integer division by `cp_size` is even, matching the fused RoPE
    CP path requirement.
    """
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if local_total_tokens <= 0:
        raise ValueError(f"local_total_tokens must be positive, got {local_total_tokens}")
    if cp_size > 1 and local_total_tokens % 2 != 0:
        raise ValueError("local_total_tokens must be even when cp_size > 1")

    per_local = (local_total_tokens // n_seqs // 2) * 2
    if per_local <= 0:
        raise ValueError(
            f"n_seqs={n_seqs} is too large for local_total_tokens={local_total_tokens}"
        )
    local_lengths = [per_local] * n_seqs
    local_lengths[-1] += local_total_tokens - per_local * n_seqs

    cu = [0]
    for local_length in local_lengths:
        cu.append(cu[-1] + local_length * cp_size)
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
    parser.add_argument("--local-total-tokens", type=int, default=65536)
    parser.add_argument("--freqs-len", type=int, default=65536)
    parser.add_argument("--head-num", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--rotary-percent", type=float, default=1.0)
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument(
        "--cp-ranks",
        type=int,
        nargs="+",
        default=None,
        help="Context-parallel ranks to sweep. Defaults to all ranks in [0, cp_size).",
    )
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
    if args.cp_ranks is None:
        args.cp_ranks = list(range(args.cp_size))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this benchmark")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda:0")

    rotary = RotaryPositionEmbedding(args.hidden, args.rotary_percent, interleaved=args.interleaved)
    freqs = rotary(args.freqs_len).to(device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "rope_thd_bench.csv"
    fields = [
        "mode",
        "local_total_tokens",
        "global_total_tokens",
        "n_seqs",
        "cp_size",
        "cp_rank",
        "freqs_len",
        "max_seq",
        "legacy_blocks",
        "compact_blocks",
        "overlaunch_ratio",
        "fwd_ms",
        "bwd_ms",
        "env_override",
    ]
    rows = []

    print(
        f"# local_total_tokens={args.local_total_tokens} freqs_len={args.freqs_len} "
        f"h={args.head_num} d={args.hidden} dtype={args.dtype} cp_size={args.cp_size} "
        f"cp_ranks={args.cp_ranks}"
    )
    print("# mode local_total global_total n_seqs cp_size cp_rank fwd_ms bwd_ms overlaunch env")

    for n_seqs in args.n_seqs:
        cu = build_cu_seqlens(args.local_total_tokens, n_seqs, cp_size=args.cp_size).to(device)
        global_total_tokens = int(cu[-1].item())
        local_total_tokens = global_total_tokens // args.cp_size
        max_seq = int((cu[1:] - cu[:-1]).max().item())
        if args.freqs_len < max_seq:
            raise ValueError(
                f"freqs_len={args.freqs_len} is smaller than max global sequence length {max_seq}"
            )
        t = torch.rand(
            (local_total_tokens, args.head_num, args.hidden),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        legacy_blocks = args.freqs_len * n_seqs
        compact_blocks = local_total_tokens
        overlaunch_ratio = legacy_blocks / compact_blocks

        for cp_rank in args.cp_ranks:
            if cp_rank < 0 or cp_rank >= args.cp_size:
                raise ValueError(f"cp_rank={cp_rank} is invalid for cp_size={args.cp_size}")

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
                    cp_rank=cp_rank,
                )

            for mode, value in [("old", "0"), ("new", "1"), ("heuristic", None)]:
                with env("NVTE_FUSED_ROPE_THD_LINEAR_GRID", value):
                    fwd_ms, full_ms = time_fwd_bwd(runner, iters=args.iters, warmup=args.warmup)
                bwd_ms = full_ms - fwd_ms
                rows.append(
                    {
                        "mode": mode,
                        "local_total_tokens": local_total_tokens,
                        "global_total_tokens": global_total_tokens,
                        "n_seqs": n_seqs,
                        "cp_size": args.cp_size,
                        "cp_rank": cp_rank,
                        "freqs_len": args.freqs_len,
                        "max_seq": max_seq,
                        "legacy_blocks": legacy_blocks,
                        "compact_blocks": compact_blocks,
                        "overlaunch_ratio": f"{overlaunch_ratio:.6f}",
                        "fwd_ms": f"{fwd_ms:.4f}",
                        "bwd_ms": f"{bwd_ms:.4f}",
                        "env_override": "unset" if value is None else value,
                    }
                )
                print(
                    f"{mode:>9} local={local_total_tokens} global={global_total_tokens} "
                    f"n_seqs={n_seqs} cp={args.cp_size}/{cp_rank} fwd={fwd_ms:7.3f} "
                    f"bwd={bwd_ms:7.3f} overlaunch={overlaunch_ratio:.2f} env={value or 'unset'}"
                )

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
