# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Muon optimizer backed by distributed Newton-Schulz orthogonalization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from transformer_engine.pytorch.newton_schulz import (
    CusolverMpCtx,
    NSCoeffT,
    get_coefficients,
    newton_schulz,
)


MuonScaleT = Literal["shape_scaling", "spectral", "unit_rms_norm"]


def get_muon_scale_factor(size_out: int, size_in: int, mode: MuonScaleT = "spectral") -> float:
    """Return the Muon update scale factor for the logical matrix shape."""
    if mode == "shape_scaling":
        return max(1, size_out / size_in) ** 0.5
    if mode == "spectral":
        return max(size_out, size_in) ** 0.5
    if mode == "unit_rms_norm":
        return (size_out / size_in) ** 0.5
    raise ValueError(f"Invalid mode for Muon update scale factor: {mode}")


class MuonOptimizer(Optimizer):
    """Distributed Muon optimizer for 2D CUDA parameters.

    This optimizer applies SGD-momentum followed by Newton-Schulz orthogonalization
    on tensor-parallel parameter shards. The local parameter shard must represent a
    partition of a logical 2D matrix across the provided NCCL process group.

    Args:
        params: Iterable of parameters or parameter group dicts.
        lr: Learning rate.
        momentum: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: Weight decay coefficient.
        use_decoupled_weight_decay: Whether to apply decoupled weight decay.
        coefficient_type: Newton-Schulz coefficient schedule.
        num_ns_steps: Number of Newton-Schulz iterations.
        scale_mode: Muon update scale mode.
        extra_scale_factor: Extra multiplicative scale applied after orthogonalization.
        process_group: NCCL process group for distributed Newton-Schulz. Defaults to world.
        partition_dim: Dimension along which each logical 2D parameter is partitioned.
            Must be 0 or 1.
        eps: Lower bound for the distributed normalization denominator.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter | dict],
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        *,
        use_decoupled_weight_decay: bool = True,
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        scale_mode: MuonScaleT = "spectral",
        extra_scale_factor: float = 1.0,
        process_group: Optional[dist.ProcessGroup] = None,
        partition_dim: int = 1,
        eps: float = 1e-7,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if partition_dim not in (0, 1):
            raise ValueError(f"partition_dim must be 0 or 1, got {partition_dim}")
        get_coefficients(num_ns_steps, coefficient_type)

        if process_group is None:
            if not dist.is_initialized():
                raise RuntimeError("MuonOptimizer requires torch.distributed to be initialized.")
            process_group = dist.group.WORLD
        if dist.get_backend(process_group) != "nccl":
            raise RuntimeError("MuonOptimizer requires an NCCL process group.")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
            "use_decoupled_weight_decay": use_decoupled_weight_decay,
            "coefficient_type": coefficient_type,
            "num_ns_steps": num_ns_steps,
            "scale_mode": scale_mode,
            "extra_scale_factor": extra_scale_factor,
            "partition_dim": partition_dim,
            "eps": eps,
        }
        super().__init__(params, defaults)
        self.process_group = process_group
        self._ns_ctx: CusolverMpCtx | None = None

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        """Release the underlying cuSolverMp context."""
        if self._ns_ctx is not None:
            self._ns_ctx.destroy()
            self._ns_ctx = None

    def _get_ctx(self) -> CusolverMpCtx:
        if self._ns_ctx is None:
            self._ns_ctx = CusolverMpCtx(self.process_group)
        return self._ns_ctx

    @staticmethod
    def _validate_param(param: torch.Tensor, partition_dim: int) -> None:
        if param.ndim != 2:
            raise ValueError("MuonOptimizer only supports 2D parameters.")
        if not param.is_cuda:
            raise ValueError("MuonOptimizer only supports CUDA parameters.")
        if param.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                f"MuonOptimizer requires float32 or bfloat16 parameters, got {param.dtype}."
            )
        if param.size(partition_dim) == 0:
            raise ValueError("MuonOptimizer does not support empty tensor-parallel shards.")

    def _distributed_normalize_p2_(
        self,
        x: torch.Tensor,
        eps: float,
    ) -> None:
        norm_sq = (x.float() * x.float()).sum()
        dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=self.process_group)
        x.div_(torch.sqrt(norm_sq).clamp_min(eps).to(dtype=x.dtype))

    def _orthogonalize(
        self,
        grad: torch.Tensor,
        *,
        partition_dim: int,
        coefficient_type: NSCoeffT,
        num_ns_steps: int,
        scale_mode: MuonScaleT,
        extra_scale_factor: float,
        eps: float,
    ) -> torch.Tensor:
        self._validate_param(grad, partition_dim)
        world_size = dist.get_world_size(self.process_group)
        global_shape = [grad.size(0), grad.size(1)]
        global_shape[partition_dim] *= world_size

        orth_grad = grad.clone()
        transposed = partition_dim == 0
        if transposed:
            orth_grad = orth_grad.mT.contiguous()
        else:
            orth_grad = orth_grad.contiguous()

        self._distributed_normalize_p2_(orth_grad, eps)
        coefficients = get_coefficients(num_ns_steps, coefficient_type)
        newton_schulz(orth_grad, self._get_ctx(), num_ns_steps, coefficients=coefficients)

        if transposed:
            orth_grad = orth_grad.mT.contiguous()

        scale_factor = get_muon_scale_factor(global_shape[0], global_shape[1], mode=scale_mode)
        orth_grad.mul_(scale_factor * extra_scale_factor)
        return orth_grad

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                self._validate_param(p, group["partition_dim"])
                grad = p.grad
                if grad.dtype != p.dtype:
                    raise ValueError(
                        f"Gradient dtype {grad.dtype} must match parameter dtype {p.dtype}."
                    )
                if grad.shape != p.shape:
                    raise ValueError("Gradient shape must match parameter shape.")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                if group["use_decoupled_weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                elif group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.lerp_(grad, 1.0 - group["momentum"])

                if group["nesterov"]:
                    update = grad.lerp(momentum_buffer, group["momentum"])
                else:
                    update = momentum_buffer

                orth_update = self._orthogonalize(
                    update,
                    partition_dim=group["partition_dim"],
                    coefficient_type=group["coefficient_type"],
                    num_ns_steps=group["num_ns_steps"],
                    scale_mode=group["scale_mode"],
                    extra_scale_factor=group["extra_scale_factor"],
                    eps=group["eps"],
                )
                p.add_(orth_update, alpha=-group["lr"])

        return loss
