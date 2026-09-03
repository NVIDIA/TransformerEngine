# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Real-NCCL validation for single-process multi-device FP8 reduction.

Run with four visible GPUs::

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=2 \
        tests/pytorch/distributed/run_multi_device_fp8.py

Each rank owns two distinct local GPUs. Modules are registered in an
interleaved device order (first local GPU -> second local GPU), and the
DelayedScaling amax reduction uses one real NCCL all-reduce per direction.
"""

import hashlib

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2 or torch.cuda.device_count() < 4:
        raise RuntimeError("This check requires two ranks and four visible CUDA devices")

    first_device = 2 * rank
    second_device = first_device + 1
    torch.cuda.set_device(first_device)

    torch.manual_seed(1234)
    first = te.Linear(
        512, 1024, bias=False, params_dtype=torch.bfloat16, device=f"cuda:{first_device}"
    )
    torch.manual_seed(5678)
    second = te.Linear(
        1024, 512, bias=False, params_dtype=torch.bfloat16, device=f"cuda:{second_device}"
    )
    recipe = DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=16,
        reduce_amax=True,
    )

    outputs = []
    for iteration in range(10):
        torch.manual_seed(100 + iteration)
        inp = torch.randn(
            128,
            512,
            device=f"cuda:{first_device}",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        with te.autocast(
            enabled=True,
            recipe=recipe,
            amax_reduction_group=dist.group.WORLD,
        ):
            hidden = first(inp)
            out = second(hidden.to(f"cuda:{second_device}"))
        out.sum().backward()
        torch.cuda.synchronize(first_device)
        torch.cuda.synchronize(second_device)
        outputs.append(out.detach().float().cpu())

    values = list(outputs)
    for module in (first, second):
        for key in ("scaling_fwd", "scaling_bwd"):
            values.extend(
                (
                    module.fp8_meta[key].scale.cpu(),
                    module.fp8_meta[key].amax_history.cpu(),
                )
            )
    flat = torch.cat([value.reshape(-1) for value in values])
    digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, digest)
    if rank == 0:
        print(f"NCCL multi-device FP8 hashes: {gathered}", flush=True)
    if len(set(gathered)) != 1:
        raise RuntimeError(f"Ranks produced different results: {gathered}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
