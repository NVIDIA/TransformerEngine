# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVTE_NVFP4_FUSED_MASTER_WEIGHT_CAST=1 must give the bytes of the default NVFP4 2D path.

Each rank owns an uneven, contiguous slice of the concatenated weights, as a distributed
optimizer does. The slice boundaries cut tensors in the middle of rows and of 16x16 tiles.
The FP32 masters are not BF16-representable and include all-zero tiles and outliers. For
several update steps, quantize_master_weights with the option must write the same rowwise FP4
data, scale-inverses and amaxes as without it, on every rank. With an odd slice boundary,
the rank whose shard starts at the odd offset runs the per-tensor kernels for its own shards
while the others run the multi-tensor ones, under the same collectives (no hang).
"""

import argparse
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from utils import run_distributed

import pytest
import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch import is_nvfp4_available
import transformer_engine.pytorch.tensor.utils as te_utils

ENV = "NVTE_NVFP4_FUSED_MASTER_WEIGHT_CAST"

SHAPES = [
    (4096, 7168),
    (7168, 2048),
    (512, 512),
    (256, 1536),
    (48, 64),  # scale rows padded beyond the tile rows
    (1024, 2048),
    (128, 7168),
    (7168, 128),
    (64, 4096),
    (40, 96),
    (2112, 7168),
] * 3  # more tensors than one launch takes


def _make_weights(shapes, params_dtype):
    """NVFP4 weights; params_dtype "mixed" alternates BF16 and FP32."""
    torch.manual_seed(1234)
    dtypes = [params_dtype] * len(shapes)
    if params_dtype == "mixed":
        dtypes = [(torch.bfloat16, torch.float32)[i % 2] for i in range(len(shapes))]
    with te.quantized_model_init(enabled=True, recipe=NVFP4BlockScaling()):
        layers = [
            te.Linear(w, h, bias=False, params_dtype=dtype, device="cuda")
            for (h, w), dtype in zip(shapes, dtypes)
        ]
    return [layer.weight for layer in layers]


def _masters(shapes):
    torch.manual_seed(999)  # same full masters on every rank
    masters = []
    for i, (h, w) in enumerate(shapes):
        m = torch.randn(h, w, device="cuda", dtype=torch.float32) * (0.02 * (1 + i % 3))
        if i % 4 == 1:
            m[: h // 3] *= 1e-3
        if i % 5 == 2:
            m[:16, :16] = 0  # all-zero tile
        if i == 0:
            m[5, 7] = 3.7  # outlier
        masters.append(m)
    return masters


def _rank_slices(numels, rank, world, seed, odd_cut=False):
    """Uneven contiguous slices of the flat space, cut at even offsets (the first cut at an odd
    offset with odd_cut)."""
    total = sum(numels)
    g = torch.Generator().manual_seed(seed)
    cuts = sorted({int(x) // 2 * 2 for x in (torch.rand(world - 1, generator=g) * total).tolist()})
    if odd_cut and cuts:
        cuts[0] += 1
    bounds = [0] + cuts + [total] * (world + 1 - len(cuts) - 1)
    return bounds[rank], bounds[rank + 1]


def _shards(masters, lo, hi, dtype):
    shards, offsets = [], []
    base = 0
    for m in masters:
        n = m.numel()
        s, e = max(lo, base), min(hi, base + n)
        if s < e:
            shards.append(m.view(-1)[s - base : e - base].to(dtype).clone())
            offsets.append(s - base)
        else:
            shards.append(None)
            offsets.append(0)
        base += n
    return shards, offsets


def _quantize(weights, shards, offsets, group, fused):
    os.environ[ENV] = "1" if fused else "0"
    try:
        te_utils.quantize_master_weights(
            weights, shards, offsets, group, manual_post_all_gather_processing=True
        )
    finally:
        os.environ.pop(ENV, None)


def _compare(ref, new, tag):
    rank = dist.get_rank()
    bad = []
    for i, (a, b) in enumerate(zip(ref, new)):
        for name in ("_rowwise_data", "_rowwise_scale_inv", "_amax_rowwise"):
            ta = getattr(a, name).view(torch.uint8)
            tb = getattr(b, name).view(torch.uint8)
            if not torch.equal(ta, tb):
                bad.append(f"[rank {rank}] {tag} tensor {i} {tuple(a.shape)} {name}")
    flag = torch.tensor([len(bad)], device="cuda")
    dist.all_reduce(flag)
    assert flag.item() == 0, "\n".join(bad) or "mismatch on another rank"


def _check_bit_identity(params_dtype, master_dtype, expect_fused, seed, odd_cut=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.new_group(ranks=list(range(world)))
    shapes = SHAPES
    ref = _make_weights(shapes, params_dtype)
    new = [w.detach().clone() for w in ref]
    masters = _masters(shapes)
    lo, hi = _rank_slices([h * w for h, w in shapes], rank, world, seed, odd_cut)
    shards_ref, offsets = _shards(masters, lo, hi, master_dtype)
    shards_new, _ = _shards(masters, lo, hi, master_dtype)

    taken = []
    fused_fn = te_utils._cast_master_weights_to_nvfp4_2d_fused

    def spy(*args, **kwargs):
        taken.append(fused_fn(*args, **kwargs))
        return taken[-1]

    te_utils._cast_master_weights_to_nvfp4_2d_fused = spy
    try:
        for step in range(3):
            if step > 0:  # an optimizer update between steps
                for a, b in zip(shards_ref, shards_new):
                    if a is not None:
                        d = (torch.randn_like(a, dtype=torch.float32) * 1e-3).to(a.dtype)
                        a.add_(d)
                        b.add_(d)
            _quantize(ref, shards_ref, offsets, group, fused=False)
            _quantize(new, shards_new, offsets, group, fused=True)
            torch.cuda.synchronize()
            _compare(ref, new, f"{params_dtype} {master_dtype} step {step}")
    finally:
        te_utils._cast_master_weights_to_nvfp4_2d_fused = fused_fn
    assert taken and all(t == expect_fused for t in taken), taken


def _check_cuda_graph(seed):
    """Capture the fused cast once and replay it after in-place master updates."""
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.new_group(ranks=list(range(world)))
    shapes = SHAPES[:11]
    ref = _make_weights(shapes, torch.bfloat16)
    new = [w.detach().clone() for w in ref]
    masters = _masters(shapes)
    lo, hi = _rank_slices([h * w for h, w in shapes], rank, world, seed)
    shards_ref, offsets = _shards(masters, lo, hi, torch.float32)
    shards_new, _ = _shards(masters, lo, hi, torch.float32)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):  # warm up NCCL and the allocator outside the capture
            _quantize(new, shards_new, offsets, group, fused=True)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    os.environ[ENV] = "1"
    try:
        with torch.cuda.graph(graph, stream=stream):
            te_utils.quantize_master_weights(
                new, shards_new, offsets, group, manual_post_all_gather_processing=True
            )
    finally:
        os.environ.pop(ENV, None)

    for step in range(3):
        for a, b in zip(shards_ref, shards_new):
            if a is not None:
                d = torch.randn_like(a) * 1e-3
                a.add_(d)
                b.add_(d)
        _quantize(ref, shards_ref, offsets, group, fused=False)
        graph.replay()
        torch.cuda.synchronize()
        _compare(ref, new, f"cuda graph replay {step}")


def _worker():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    try:
        # BF16 model weights with FP32 masters: the fused path rounds in registers.
        _check_bit_identity(torch.bfloat16, torch.float32, expect_fused=True, seed=77)
        _check_bit_identity(torch.bfloat16, torch.float32, expect_fused=True, seed=78)
        # FP32 model weights: no rounding.
        _check_bit_identity(torch.float32, torch.float32, expect_fused=True, seed=79)
        # BF16 masters.
        _check_bit_identity(torch.bfloat16, torch.bfloat16, expect_fused=True, seed=80)
        # An odd slice boundary: one rank's shard starts at an odd offset, which only the
        # per-tensor kernels take. Every rank must still take the fused path's collectives.
        _check_bit_identity(torch.bfloat16, torch.float32, expect_fused=True, seed=83, odd_cut=True)
        # Mixed model dtypes in one call are left to the default path.
        _check_bit_identity("mixed", torch.float32, expect_fused=False, seed=82)
        _check_cuda_graph(seed=81)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_nvfp4_fused_master_weight_cast(world_size: int) -> None:
    """Launch the distributed bit-identity test."""
    available, reason = is_nvfp4_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        str(pathlib.Path(__file__).resolve()),
        "--worker",
    ]
    run_distributed(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    if parser.parse_args().worker:
        _worker()
