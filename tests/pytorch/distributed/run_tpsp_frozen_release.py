# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-GPU TP/SP smoke for NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE.

Launch: torchrun --nproc_per_node=2 run_tpsp_frozen_release.py

Frozen column-parallel + row-parallel te.Linear pair (Float8BlockScaling,
quantized_model_init) with sequence_parallel=True. One fwd+bwd per step,
3 steps: must not raise, frozen weights must have columnwise released,
and dgrad must be bitwise identical to a flag-off run.
"""

import os
import sys

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

SEED = 1234
FEATURES = 256
TOKENS_PER_RANK = 128


def run(flag: str, device, tp_group, world):
    os.environ["NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE"] = flag
    fp8_recipe = recipe.Float8BlockScaling(fp8_format=recipe.Format.E4M3)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    with torch.no_grad(), te.quantized_model_init(enabled=True, recipe=fp8_recipe):
        col = te.Linear(
            FEATURES,
            FEATURES,
            bias=False,
            params_dtype=torch.bfloat16,
            parallel_mode="column",
            tp_group=tp_group,
            tp_size=world,
            sequence_parallel=True,
        )
        row = te.Linear(
            FEATURES,
            FEATURES,
            bias=False,
            params_dtype=torch.bfloat16,
            parallel_mode="row",
            tp_group=tp_group,
            tp_size=world,
            sequence_parallel=True,
        )
    weights = []
    for module in (col, row):
        for p in module.parameters():
            p.requires_grad_(False)
            if isinstance(p, QuantizedTensorStorage):
                weights.append(p)

    grads = []
    for step in range(3):
        torch.manual_seed(SEED + step)  # same across ranks: SP gathers along sequence
        inp = torch.randn(
            TOKENS_PER_RANK, FEATURES, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        with te.autocast(enabled=True, recipe=fp8_recipe):
            out = row(col(inp))
        out.float().pow(2).mean().backward()
        grads.append(inp.grad.detach().clone())
    released = [not w.get_usages()["columnwise"] for w in weights]
    return grads, released


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    world = dist.get_world_size()
    tp_group = dist.new_group(ranks=list(range(world)))

    grads_off, _ = run("0", device, tp_group, world)
    grads_on, released_on = run("1", device, tp_group, world)

    assert released_on, "expected quantized frozen weights"
    assert all(released_on), f"frozen weights not released: {released_on}"
    for ref, rel in zip(grads_off, grads_on):
        assert torch.equal(ref, rel), "dgrad mismatch between flag off/on under TP/SP"

    dist.barrier()
    if dist.get_rank() == 0:
        print(
            f"TP/SP OK world={world} sequence_parallel=True steps=3"
            f" released={sum(released_on)}/{len(released_on)} dgrad=bitwise-equal",
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
