# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Check FSDP2 safety and dgrad equivalence with BF16 parameters and FP8 workspaces.

Launch: torchrun --nproc_per_node=2 run_fsdp2_frozen_release.py --reshard-after-forward {0,1}

FSDP2 does not cache these workspaces across steps. Primary FP8 parameters
with Float8BlockScaling are marked xfail upstream in fsdp2_tests/run_fsdp2_model.py;
the columnwise-only guard is covered separately in the single-GPU tests.
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
            # FSDP2 disables workspace caching even when is_first_microbatch is set.
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

    # FSDP2 workspaces are transient; neither run should retain a cached copy.
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
