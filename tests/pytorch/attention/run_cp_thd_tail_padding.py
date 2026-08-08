# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression test for THD + context-parallel (p2p) tail padding.

GitHub issue: https://github.com/NVIDIA/TransformerEngine/issues/3331

A single packed sequence whose REAL length is not divisible by 2*cp_size is
tail-padded to a multiple of 2*cp_size (the common case for packed THD
training). The pad_between_seqs=False CP path approximates per-ring-step
seqlens as cu_seqlens // cp_size, which mislabels chunk-boundary rows in
this case, producing (a) nondeterministic forward outputs (uninitialized
softmax-LSE aux rows merged into a real row's scale factor), (b) silently
wrong attention (real keys dropped from every query's attention, real query
rows receiving exact-zero output), and (c) nondeterministic gradients on all
CP ranks.

This test pins the exact-path semantics — reached via explicit
pad_between_seqs=True, or via auto-detect with padded cu_seqlens supplied as
tensors distinct from their unpadded counterparts — along two axes:

  1. determinism: forward and fwd+bwd outputs are bitwise-stable across
     iterations on identical inputs;
  2. correctness: the reassembled CP output and gradients match the no-CP
     reference on all real (non-padding) rows within bf16 tolerance.

Launch: torchrun --standalone --nproc-per-node={2,4} run_cp_thd_tail_padding.py
"""

from __future__ import annotations

import os
import sys

# Backend selection must be pinned before transformer_engine is imported: the
# defect lives in the FusedAttention (cuDNN) CP path.
os.environ["NVTE_FLASH_ATTN"] = "0"
os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te

SEQ_REAL = 698  # real tokens; 698 % (2*cp) != 0 for cp in {2, 4} -> tail padding
HEADS_Q = 8
HEADS_KV = 1
HEAD_DIM = 128
SEED = 16
FWD_ITERS = 25  # forward determinism iterations
BWD_ITERS = 10  # fwd+bwd determinism iterations
# bf16 CP-vs-reference rounding tolerance. The defect produces mean |delta|
# 0.04 / max 0.36 plus exact-zero real rows — an order of magnitude outside
# this band, so the check cannot pass under the bug by luck.
ATOL = 2.5e-2
RTOL = 2.5e-2


def shard_indices(cp: int, rank: int, chunk: int) -> torch.Tensor:
    """Load-balanced CP shard: chunks [rank, 2*cp-1-rank] of the padded buffer."""
    return torch.cat(
        [
            torch.arange(rank * chunk, (rank + 1) * chunk),
            torch.arange((2 * cp - 1 - rank) * chunk, (2 * cp - rank) * chunk),
        ]
    )


def make_attn(dev: torch.device, with_cp: bool) -> te.DotProductAttention:
    attn = te.DotProductAttention(
        num_attention_heads=HEADS_Q,
        kv_channels=HEAD_DIM,
        num_gqa_groups=HEADS_KV,
        attention_dropout=0.0,
        attn_mask_type="padding_causal",
        qkv_format="thd",
        softmax_scale=None,
    ).to(dev)
    if with_cp:
        cp_group = dist.new_group(ranks=list(range(dist.get_world_size())), backend="nccl")
        attn.set_context_parallel_group(
            cp_group,
            list(range(dist.get_world_size())),
            torch.cuda.Stream(device=dev),
            cp_comm_type="p2p",
        )
    return attn


def run_fwd(attn, q, k, v, cu, cu_padded, t_total, pad_mode):
    kwargs = dict(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu_padded,
        cu_seqlens_kv_padded=cu_padded,
        max_seqlen_q=t_total,
        max_seqlen_kv=t_total,
    )
    # "explicit" pins the exact per-rank path; "auto" exercises the auto-detect
    # (distinct padded/unpadded tensor objects must select pad_between_seqs=True).
    if pad_mode == "explicit":
        kwargs["pad_between_seqs"] = True
    return attn(q, k, v, **kwargs)


def check_close(name, actual, expected, failures):
    diff = (actual.float() - expected.float()).abs()
    tol = ATOL + RTOL * expected.float().abs()
    bad = diff > tol
    if bad.any():
        failures.append(
            f"{name}: {int(bad.sum())}/{bad.numel()} elems exceed tolerance, "
            f"max|delta|={diff.max().item():.4f}"
        )


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    cp = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    assert SEQ_REAL % (2 * cp) != 0, "test requires real length not divisible by 2*cp"
    chunk = -(-SEQ_REAL // (2 * cp))  # ceil division
    t_total = 2 * cp * chunk  # padded total; SEQ_REAL < t_total (tail padding)

    idx = shard_indices(cp, rank, chunk)
    real_mask = idx < SEQ_REAL
    local_real = torch.nonzero(real_mask).flatten()  # real rows in the local shard
    global_real = idx[real_mask]  # their positions in the full padded buffer

    cu = torch.tensor([0, SEQ_REAL], dtype=torch.int32, device=dev)
    cu_padded = torch.tensor([0, t_total], dtype=torch.int32, device=dev)

    # Identical full-buffer inputs on every rank (CPU generator is device-independent).
    g = torch.Generator(device="cpu").manual_seed(SEED)
    q_full = torch.randn(t_total, HEADS_Q, HEAD_DIM, generator=g).bfloat16()
    k_full = torch.randn(t_total, HEADS_KV, HEAD_DIM, generator=g).bfloat16()
    v_full = torch.randn(t_total, HEADS_KV, HEAD_DIM, generator=g).bfloat16()
    dout_full = torch.randn(t_total, HEADS_Q * HEAD_DIM, generator=g).bfloat16()

    q_loc, k_loc, v_loc = (x[idx].to(dev) for x in (q_full, k_full, v_full))
    dout_loc = dout_full[idx].to(dev)

    # No-CP reference on the UNPADDED buffer (ground truth for the real tokens):
    # no padding anywhere, so no pad_between_seqs involvement at all. Real token i
    # occupies row i in both the padded and unpadded layouts, so global_real indexes
    # the reference directly.
    attn_ref = make_attn(dev, with_cp=False)
    qr, kr, vr = (x[:SEQ_REAL].to(dev).requires_grad_(True) for x in (q_full, k_full, v_full))
    out_ref = attn_ref(
        qr,
        kr,
        vr,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=SEQ_REAL,
        max_seqlen_kv=SEQ_REAL,
    )
    out_ref.backward(dout_full[:SEQ_REAL].to(dev))

    failures = []
    for pad_mode in ["explicit", "auto"]:
        attn = make_attn(dev, with_cp=True)

        # Arm 1: forward determinism — bitwise-stable across iterations.
        with torch.inference_mode():
            outs = [
                run_fwd(attn, q_loc, k_loc, v_loc, cu, cu_padded, t_total, pad_mode).clone()
                for _ in range(FWD_ITERS)
            ]
        n_mismatch = sum(not torch.equal(outs[0], o) for o in outs[1:])
        if n_mismatch:
            failures.append(
                f"{pad_mode}: forward nondeterministic "
                f"({n_mismatch}/{FWD_ITERS - 1} iterations differ bitwise)"
            )

        # Arm 2: fwd+bwd determinism — outputs and gradients bitwise-stable.
        iter_results = []
        for _ in range(BWD_ITERS):
            qg, kg, vg = (x.clone().requires_grad_(True) for x in (q_loc, k_loc, v_loc))
            out = run_fwd(attn, qg, kg, vg, cu, cu_padded, t_total, pad_mode)
            out.backward(dout_loc)
            iter_results.append(
                (out.detach().clone(), qg.grad.clone(), kg.grad.clone(), vg.grad.clone())
            )
        base = iter_results[0]
        n_mismatch = sum(
            not all(torch.equal(b, r) for b, r in zip(base, res)) for res in iter_results[1:]
        )
        if n_mismatch:
            failures.append(
                f"{pad_mode}: fwd+bwd nondeterministic "
                f"({n_mismatch}/{BWD_ITERS - 1} iterations differ bitwise)"
            )

        # Arm 3: correctness vs the no-CP reference on real rows.
        out_cp, dq_cp, dk_cp, dv_cp = base
        check_close(f"{pad_mode} out", out_cp[local_real], out_ref.detach()[global_real], failures)
        check_close(f"{pad_mode} dq", dq_cp[local_real], qr.grad[global_real], failures)
        check_close(f"{pad_mode} dk", dk_cp[local_real], kr.grad[global_real], failures)
        check_close(f"{pad_mode} dv", dv_cp[local_real], vr.grad[global_real], failures)

    gathered = [None] * cp
    dist.all_gather_object(gathered, failures)
    if rank == 0:
        all_failures = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        if all_failures:
            print("FAIL (issue #3331 regression):", flush=True)
            for f in all_failures:
                print(f"  {f}", flush=True)
        else:
            print(
                f"PASS: THD+CP(p2p) tail padding deterministic and correct "
                f"(cp={cp}, real={SEQ_REAL}, padded={t_total}, modes=explicit,auto)",
                flush=True,
            )
    dist.destroy_process_group()
    return 1 if any(any(fs) for fs in gathered) else 0


if __name__ == "__main__":
    sys.exit(main())
