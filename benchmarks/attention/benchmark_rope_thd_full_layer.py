# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Full TransformerLayer benchmark for token-linear THD fused RoPE.

This benchmark keeps the local packed-token count and RoPE table length fixed
while varying the number of packed THD spans. It compares the old fused RoPE
launch, the new token-linear launch, and the heuristic path on a TE
TransformerLayer using THD input and rotary embeddings. It also measures a
paired RoPE-only operation with the same tensor shape, so the output table can
report both end-to-end layer speedup and the fraction of layer time attributable
to fused RoPE.
"""

from __future__ import annotations

import argparse
import csv
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable

import torch

import transformer_engine.pytorch as te
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


def build_cu_seqlens(total_tokens: int, n_seqs: int) -> tuple[torch.Tensor, int]:
    """Build balanced packed THD cu_seqlens with an exact total token count."""
    per = total_tokens // n_seqs
    if per <= 0:
        raise ValueError(
            f"n_seqs={n_seqs} is too large for total_tokens={total_tokens}"
        )
    rem = total_tokens - per * n_seqs
    lengths = [per + (1 if i < rem else 0) for i in range(n_seqs)]
    cu = [0]
    max_seqlen = 0
    for length in lengths:
        cu.append(cu[-1] + length)
        max_seqlen = max(max_seqlen, length)
    return torch.tensor(cu, dtype=torch.int32), max_seqlen


def zero_grads(params: Iterable[torch.Tensor], x: torch.Tensor) -> None:
    if x.grad is not None:
        x.grad = None
    for p in params:
        if p.grad is not None:
            p.grad = None


def time_fwd_bwd(
    fn: Callable[[], torch.Tensor], warmup: int, iters: int
) -> tuple[float, float]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        out = fn()
        out.sum().backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)
    fwd_total = 0.0
    full_total = 0.0
    for _ in range(iters):
        start.record()
        out = fn()
        fwd_end.record()
        out.sum().backward()
        bwd_end.record()
        torch.cuda.synchronize()
        fwd_total += start.elapsed_time(fwd_end)
        full_total += start.elapsed_time(bwd_end)
    return fwd_total / iters, full_total / iters


def make_layer(args: argparse.Namespace, dtype: torch.dtype) -> te.TransformerLayer:
    sigma = 0.02

    def init_method(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return te.TransformerLayer(
        args.hidden_size,
        args.ffn_hidden_size,
        args.num_heads,
        layernorm_epsilon=1e-5,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method=init_method,
        output_layer_init_method=init_method,
        layer_number=1,
        kv_channels=args.head_dim,
        self_attn_mask_type="padding_causal",
        tp_group=None,
        tp_size=1,
        params_dtype=dtype,
        get_rng_state_tracker=None,
        fuse_wgrad_accumulation=False,
        seq_length=args.freqs_len,
        micro_batch_size=1,
        sequence_parallel=False,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        layer_type="encoder",
        set_parallel_mode=True,
        fuse_qkv_params=True,
        zero_centered_gamma=False,
        qkv_weight_interleaved=True,
        bias=True,
        attn_input_format="thd",
        rotary_pos_interleaved=args.interleaved,
        device="cuda",
    ).to(dtype=dtype, device="cuda")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tokens", type=int, default=65536)
    parser.add_argument("--freqs-len", type=int, default=65536)
    parser.add_argument("--hidden-size", type=int, default=1536)
    parser.add_argument("--ffn-hidden-size", type=int, default=6144)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    # n_seqs=50 is intentionally omitted from the default sweep because the
    # balanced-span shape has max_seqlen~=1311 and can hit a cuDNN fused-attn
    # execution failure unrelated to RoPE on the tested H100 stack. The high-span
    # cases below are the issue-relevant regime where RoPE launch waste dominates.
    parser.add_argument("--n-seqs", type=int, nargs="+", default=[128, 512, 1024, 2401])
    parser.add_argument(
        "--out-dir", type=Path, default=Path("rope_thd_full_layer_bench")
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.hidden_size % args.num_heads != 0:
        raise SystemExit("--hidden-size must be divisible by --num-heads")
    args.head_dim = args.hidden_size // args.num_heads
    if args.freqs_len < args.total_tokens:
        raise SystemExit(
            "--freqs-len should be >= --total-tokens for this long-context benchmark"
        )

    torch.manual_seed(1234)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device("cuda")

    rotary = RotaryPositionEmbedding(args.head_dim, interleaved=args.interleaved)
    freqs = rotary(args.freqs_len).to(device=device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "rope_thd_full_layer_bench.csv"
    fields = [
        "n_seqs",
        "regime",
        "max_seqlen",
        "layer_fwd_ms",
        "layer_fwd_bwd_ms",
        "layer_bwd_ms",
        "rope_pair_fwd_ms",
        "rope_pair_fwd_bwd_ms",
        "rope_pair_bwd_ms",
        "rope_pair_pct_layer",
        "layer_speedup_vs_old",
        "rope_pair_speedup_vs_old",
    ]
    rows: list[dict[str, str | int]] = []

    print(
        "# full-layer THD RoPE benchmark: "
        f"T={args.total_tokens} freqs_len={args.freqs_len} hidden={args.hidden_size} "
        f"ffn={args.ffn_hidden_size} heads={args.num_heads} dtype={args.dtype}",
        flush=True,
    )
    print(
        "# n_seqs regime max_seqlen layer_fwd layer_fwd_bwd rope_pair_fwd_bwd "
        "rope_pct layer_speedup",
        flush=True,
    )

    for n_seqs in args.n_seqs:
        cu_cpu, max_seqlen = build_cu_seqlens(args.total_tokens, n_seqs)
        cu = cu_cpu.to(device=device)
        x = torch.randn(
            args.total_tokens,
            args.hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        q = torch.randn(
            args.total_tokens,
            args.num_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        k = torch.randn_like(q, requires_grad=True)
        layer = make_layer(args, dtype)
        layer.train()
        params = tuple(layer.parameters())

        layer_old = None
        rope_old = None

        for regime, override in (("old", "0"), ("new", "1"), ("heuristic", None)):
            with env("NVTE_FUSED_ROPE_THD_TOKEN_LINEAR", override):

                def layer_fn() -> torch.Tensor:
                    zero_grads(params, x)
                    return layer(
                        x,
                        rotary_pos_emb=freqs,
                        cu_seqlens_q=cu,
                        cu_seqlens_kv=cu,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_kv=max_seqlen,
                    )

                def rope_pair_fn() -> torch.Tensor:
                    if q.grad is not None:
                        q.grad = None
                    if k.grad is not None:
                        k.grad = None
                    q_out = apply_rotary_pos_emb(
                        q,
                        freqs,
                        tensor_format="thd",
                        fused=True,
                        cu_seqlens=cu,
                        interleaved=args.interleaved,
                    )
                    k_out = apply_rotary_pos_emb(
                        k,
                        freqs,
                        tensor_format="thd",
                        fused=True,
                        cu_seqlens=cu,
                        interleaved=args.interleaved,
                    )
                    return q_out + k_out

                layer_fwd, layer_full = time_fwd_bwd(layer_fn, args.warmup, args.iters)
                rope_fwd, rope_full = time_fwd_bwd(
                    rope_pair_fn, args.warmup, args.iters
                )

            if regime == "old":
                layer_old = layer_full
                rope_old = rope_full
            assert layer_old is not None and rope_old is not None
            layer_speedup = layer_old / layer_full
            rope_speedup = rope_old / rope_full
            rope_pct = 100.0 * rope_full / layer_full
            rows.append(
                {
                    "n_seqs": n_seqs,
                    "regime": regime,
                    "max_seqlen": max_seqlen,
                    "layer_fwd_ms": f"{layer_fwd:.4f}",
                    "layer_fwd_bwd_ms": f"{layer_full:.4f}",
                    "layer_bwd_ms": f"{layer_full - layer_fwd:.4f}",
                    "rope_pair_fwd_ms": f"{rope_fwd:.4f}",
                    "rope_pair_fwd_bwd_ms": f"{rope_full:.4f}",
                    "rope_pair_bwd_ms": f"{rope_full - rope_fwd:.4f}",
                    "rope_pair_pct_layer": f"{rope_pct:.2f}",
                    "layer_speedup_vs_old": f"{layer_speedup:.3f}",
                    "rope_pair_speedup_vs_old": f"{rope_speedup:.3f}",
                }
            )
            print(
                f"{n_seqs:>6} {regime:>10} {max_seqlen:>10} "
                f"layer_fwd={layer_fwd:8.3f} layer_fwd_bwd={layer_full:8.3f} "
                f"rope_pair_fwd_bwd={rope_full:8.3f} rope_pct={rope_pct:6.2f}% "
                f"layer_speedup={layer_speedup:6.3f}x",
                flush=True,
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

    nseqs = sorted({int(r["n_seqs"]) for r in rows})
    by_regime = {regime: [] for regime in ("old", "new", "heuristic")}
    pct_by_regime = {regime: [] for regime in ("old", "new", "heuristic")}
    for n in nseqs:
        for regime in by_regime:
            row = next(
                r for r in rows if int(r["n_seqs"]) == n and r["regime"] == regime
            )
            by_regime[regime].append(float(row["layer_fwd_bwd_ms"]))
            pct_by_regime[regime].append(float(row["rope_pair_pct_layer"]))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    for regime, values in by_regime.items():
        ax.plot(nseqs, values, marker="o", label=regime)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_seqs")
    ax.set_ylabel("TransformerLayer fwd+bwd (ms)")
    ax.set_title("Full THD TransformerLayer")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    speedups = [by_regime["old"][i] / by_regime["new"][i] for i in range(len(nseqs))]
    ax.plot(nseqs, speedups, marker="o", color="tab:green")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("n_seqs")
    ax.set_ylabel("Layer speedup (old / new)")
    ax.set_title("End-to-end layer speedup")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    for regime, values in pct_by_regime.items():
        ax.plot(nseqs, values, marker="o", label=regime)
    ax.set_xscale("log")
    ax.set_xlabel("n_seqs")
    ax.set_ylabel("paired RoPE fwd+bwd / layer fwd+bwd (%)")
    ax.set_title("RoPE share estimate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    png_path = args.out_dir / "rope_thd_full_layer_bench.png"
    fig.savefig(png_path, dpi=120)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
