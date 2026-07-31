# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-GPU FSDP2 validation for NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE.

Launch: torchrun --nproc_per_node=2 run_fsdp2_frozen_release.py --reshard-after-forward {0,1}

Model: frozen TE Linear stack (bf16 Parameters, Float8BlockScaling autocast)
+ trainable bf16 head, wrapped with FSDP2 fully_shard. TE intentionally
disables FP8 weight-workspace caching under FSDP2 (linear.py:
``cache_name = None`` when ``is_fsdp2``), so there is no persistent cache to
inspect after the step; transient backward workspaces may still be handled by
the release helper and the existing FSDP2 cleanup, without forming resident
state. This path validates safety and numerics. For both
reshard_after_forward settings this script verifies:
  1. 3 training steps run without error with the flag on;
  2. input grads are bitwise identical to a flag-off run;
  3. no cached FP8 weight workspaces exist (cache-free expectation).

Note: quantized_model_init (primary FP8 params) + FSDP2 + Float8BlockScaling is
currently broken upstream (scale-inv padding in all-gather slice ops, see
fsdp2_tests/run_fsdp2_model.py xfail), so that combination cannot be run here;
the columnwise-only all-gather guard is covered at unit level in
tests/pytorch/test_frozen_weight_columnwise_release.py.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

SEED = 1234
FEATURES = 256
LAYERS = 4
TOKENS = 256


def _columnwise_present(ws) -> bool:
    return ws.get_usages()["columnwise"]


def build_model(device):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    layers = [
        te.Linear(FEATURES, FEATURES, bias=False, params_dtype=torch.bfloat16)
        for _ in range(LAYERS)
    ]
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad_(False)
    head = torch.nn.Linear(FEATURES, FEATURES, bias=False, dtype=torch.bfloat16, device=device)
    return torch.nn.Sequential(*layers, head)


def run(flag: str, reshard: bool, device):
    os.environ["NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE"] = flag
    fp8_recipe = recipe.Float8BlockScaling(fp8_format=recipe.Format.E4M3)
    model = build_model(device)
    for layer in model:
        fully_shard(layer, reshard_after_forward=reshard)
    fully_shard(model, reshard_after_forward=reshard)

    grads = []
    for step in range(3):
        torch.manual_seed(SEED + step + dist.get_rank())
        inp = torch.randn(TOKENS, FEATURES, device=device, dtype=torch.bfloat16, requires_grad=True)
        with te.autocast(enabled=True, recipe=fp8_recipe):
            # is_first_microbatch is passed, but under FSDP2 TE disables the
            # weight-workspace cache (cache_name=None when is_fsdp2), so no
            # workspace is retained across the step; transient backward
            # workspaces may still be processed by the release helper.
            out = inp
            for layer in model[:-1]:
                out = layer(out, is_first_microbatch=(step == 0))
            out = model[-1](out)
        out.float().pow(2).mean().backward()
        grads.append(inp.grad.detach().clone())

    cached_workspaces = []
    for module in model.modules():
        workspaces = getattr(module, "_fp8_workspaces", None)
        if not workspaces:
            continue
        for ws in workspaces.values():
            if isinstance(ws, QuantizedTensorStorage):
                cached_workspaces.append(ws)
    return grads, cached_workspaces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reshard-after-forward", type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    reshard = bool(args.reshard_after_forward)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)

    grads_off, workspaces_off = run("0", reshard, device)
    grads_on, workspaces_on = run("1", reshard, device)

    # TE intentionally disables weight-workspace caching under FSDP2
    # (linear.py: cache_name=None when is_fsdp2). Assert that cache-free
    # expectation explicitly (no empty-`all()` vacuous pass): there is no
    # persistent cache to inspect; this path validates safety and numerics.
    assert len(workspaces_off) == 0, f"unexpected cached workspaces: {len(workspaces_off)}"
    assert len(workspaces_on) == 0, f"unexpected cached workspaces: {len(workspaces_on)}"
    for ref, rel in zip(grads_off, grads_on):
        assert torch.equal(ref, rel), "dgrad mismatch between flag off/on under FSDP2"

    dist.barrier()
    if dist.get_rank() == 0:
        print(
            f"FSDP2 OK reshard_after_forward={reshard} world={dist.get_world_size()}"
            " steps=3 cached_workspaces=0 (explicit cache-free expectation)"
            " dgrad=bitwise-equal",
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
