# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed Muon optimizer test worker.

Launched via torchrun from test_muon_optimizer.py.
"""

import argparse
import sys

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

import transformer_engine.pytorch as te
from transformer_engine.pytorch.newton_schulz import get_coefficients
from transformer_engine.pytorch.optimizers.muon import get_muon_scale_factor


def _reference_orthogonalize(
    grad: torch.Tensor,
    *,
    partition_dim: int,
    world_size: int,
    coefficients: list[tuple[float, float, float]],
    scale_mode: str,
    extra_scale_factor: float,
    eps: float,
) -> torch.Tensor:
    global_shape = [grad.size(0), grad.size(1)]
    global_shape[partition_dim] *= world_size

    x = grad.clone()
    if partition_dim == 0:
        x = x.mT.contiguous()

    x = x / torch.sqrt((x.float() * x.float()).sum()).clamp_min(eps).to(dtype=x.dtype)

    for a, b, c in coefficients:
        xxt = x @ x.mT
        x = a * x + b * (xxt @ x) + c * ((xxt @ xxt) @ x)

    if partition_dim == 0:
        x = x.mT.contiguous()

    scale = get_muon_scale_factor(global_shape[0], global_shape[1], mode=scale_mode)
    return x * (scale * extra_scale_factor)


def _reference_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    lr: float,
    momentum: float,
    nesterov: bool,
    weight_decay: float,
    use_decoupled_weight_decay: bool,
    partition_dim: int,
    world_size: int,
    coefficients: list[tuple[float, float, float]],
    scale_mode: str,
    extra_scale_factor: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    param = param.clone()
    grad = grad.clone()
    momentum_buffer = momentum_buffer.clone()

    if use_decoupled_weight_decay:
        param = param * (1.0 - lr * weight_decay)
    elif weight_decay != 0:
        grad = grad + weight_decay * param

    momentum_buffer = momentum * momentum_buffer + (1.0 - momentum) * grad
    if nesterov:
        update = (1.0 - momentum) * grad + momentum * momentum_buffer
    else:
        update = momentum_buffer

    orth_update = _reference_orthogonalize(
        update,
        partition_dim=partition_dim,
        world_size=world_size,
        coefficients=coefficients,
        scale_mode=scale_mode,
        extra_scale_factor=extra_scale_factor,
        eps=eps,
    )
    param = param - lr * orth_update
    return param, momentum_buffer


@record
def main():
    parser = argparse.ArgumentParser(description="Distributed Muon optimizer test")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--partition-dim", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--weight-decay-mode", type=str, default="decoupled", choices=["decoupled", "l2"]
    )
    parser.add_argument("--num-steps", type=int, default=2)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if args.partition_dim == 0:
        full_shape = (world_size * 64, 96)
    else:
        full_shape = (96, world_size * 64)

    lr = 3e-4
    momentum = 0.95
    nesterov = True
    weight_decay = 0.01
    use_decoupled_weight_decay = args.weight_decay_mode == "decoupled"
    coefficient_type = "quintic"
    num_ns_steps = 5
    scale_mode = "spectral"
    extra_scale_factor = 1.0
    eps = 1e-7
    coefficients = get_coefficients(num_ns_steps, coefficient_type)

    if rank == 0:
        torch.manual_seed(1234)
        full_param = torch.randn(full_shape, device="cuda", dtype=dtype)
        full_grads = [
            torch.randn(full_shape, device="cuda", dtype=dtype) for _ in range(args.num_steps)
        ]
    else:
        full_param = torch.empty(full_shape, device="cuda", dtype=dtype)
        full_grads = [
            torch.empty(full_shape, device="cuda", dtype=dtype) for _ in range(args.num_steps)
        ]

    dist.broadcast(full_param, src=0)
    for grad in full_grads:
        dist.broadcast(grad, src=0)

    shard_size = full_shape[args.partition_dim] // world_size
    shard_slice = slice(rank * shard_size, (rank + 1) * shard_size)
    if args.partition_dim == 0:
        local_param_init = full_param[shard_slice, :].contiguous()
    else:
        local_param_init = full_param[:, shard_slice].contiguous()

    param = torch.nn.Parameter(local_param_init.clone())
    optimizer = te.optimizers.MuonOptimizer(
        [param],
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay,
        use_decoupled_weight_decay=use_decoupled_weight_decay,
        coefficient_type=coefficient_type,
        num_ns_steps=num_ns_steps,
        scale_mode=scale_mode,
        extra_scale_factor=extra_scale_factor,
        process_group=dist.group.WORLD,
        partition_dim=args.partition_dim,
        eps=eps,
    )

    ref_param = full_param.float()
    ref_momentum = torch.zeros_like(ref_param)
    for full_grad in full_grads:
        if args.partition_dim == 0:
            param.grad = full_grad[shard_slice, :].contiguous()
        else:
            param.grad = full_grad[:, shard_slice].contiguous()
        optimizer.step()

        ref_param, ref_momentum = _reference_step(
            ref_param,
            full_grad.float(),
            ref_momentum,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            partition_dim=args.partition_dim,
            world_size=world_size,
            coefficients=coefficients,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            eps=eps,
        )

    gathered = [torch.empty_like(param) for _ in range(world_size)]
    dist.all_gather(gathered, param)
    if args.partition_dim == 0:
        test_param = torch.cat(gathered, dim=0)
    else:
        test_param = torch.cat(gathered, dim=1)

    if rank == 0:
        expected = ref_param.to(dtype)
        atol, rtol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (2e-3, 2e-3)
        if torch.allclose(test_param, expected, atol=atol, rtol=rtol):
            print("MUON OPTIMIZER CHECK PASSED", flush=True)
        else:
            max_diff = (test_param - expected).abs().max().item()
            print(f"Max |optimizer - reference|: {max_diff:.6e}", flush=True)
            print("MUON OPTIMIZER CHECK FAILED", flush=True, file=sys.stderr)
            sys.exit(1)

    optimizer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
