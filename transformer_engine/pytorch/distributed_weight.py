# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GTP-agnostic weight-parallelism extension point.

TE owns this contract but ships no implementation; the caller (e.g. Megatron GTP) implements the
protocol on the weight and injects it at construction. Dispatchers are list-shaped (Linear -> 1,
GroupedLinear -> N; leader is ``weights[0]``) and no-op on plain tensors.
"""

import math
from typing import Any, List, Protocol, runtime_checkable

import torch

__all__ = [
    "DistributedWeight",
    "is_distributed_weight",
    "materialize_weight_for_forward",
    "materialize_weight_for_backward",
    "finalize_weight_grads",
    "get_weight_gradient_scale",
    "mark_weight_gradient_scale",
]


@runtime_checkable
class DistributedWeight(Protocol):
    """Structural interface for a custom-weight-parallel weight (AG for the GEMM, reduce/RS the
    grad, re-materialize in backward). Duck-typed ``typing.Protocol``: implementers need not
    subclass it, and all state (shards, group, async handles) lives outside TE on the implementer.

    Implementers MUST be ``torch.Tensor`` subclasses (needed by ``ctx.save_for_backward``, DDP
    backward hooks, and ``torch.compile``); enforced at runtime by :func:`is_distributed_weight`.
    """

    # Capability marker: True on an implementer, absent on plain tensors; TE's fwd/bwd gate on it.
    is_distributed_weight: bool

    def materialize_group_for_forward(self) -> Any:
        """Return the tensor(s) to feed the forward GEMM (may all-gather shards)."""

    def materialize_group_for_backward(self) -> Any:
        """Re-materialize the full weight(s) for the backward GEMMs."""

    def finalize_group_grads(self, wgrads: Any) -> Any:
        """Post-process freshly computed weight grad(s) (e.g. reduce-scatter).

        May consume ``wgrads`` in-place -- reduce-scatter into ``main_grad`` and set
        ``grad_added_to_main_grad`` -- returning a dummy grad (or ``None`` for an async collective)
        that callers use as the parameter grad(s) or discard.
        """

    def grad_buffer(self) -> torch.Tensor:
        """The gradient accumulation buffer for this weight."""


def is_distributed_weight(weight: Any) -> bool:
    """True if ``weight`` participates in custom weight parallelism (False on plain tensors).

    Enforces the :class:`DistributedWeight` requirement that an implementer be a ``torch.Tensor``
    subclass, failing loudly here rather than silently breaking autograd downstream.
    """
    flag = bool(getattr(weight, "is_distributed_weight", False))
    if flag and not isinstance(weight, torch.Tensor):
        raise TypeError(
            "DistributedWeight implementers must be torch.Tensor subclasses; got "
            f"{type(weight).__name__}."
        )
    return flag


def get_weight_gradient_scale(weight: Any, args: Any) -> float:
    """Request an optional distributed-weight mean scale for an MXFP8 wgrad GEMM.

    The weight opts in through ``fuse_wgrad_mean_scaling``. Its optional
    ``get_weight_gradient_scale()`` callback owns the reduction
    semantics. Its finalizer must consume the gradient's ``_te_wgrad_scale`` marker
    exactly once. Existing implementers without the callback retain a scale of one.
    """
    if not getattr(weight, "fuse_wgrad_mean_scaling", False):
        return 1.0
    callback = getattr(weight, "get_weight_gradient_scale", None)
    if callback is None or not is_distributed_weight(weight):
        return 1.0
    if not args.fp8 or args.debug or args.use_bias:
        return 1.0
    if args.backward_override is not None or args.activation_dtype != torch.bfloat16:
        return 1.0
    if args.grad_weight_quantizer is not None or args.cpu_offloading or args.ub_bulk_wgrad:
        return 1.0
    if args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute():
        return 1.0
    from .tensor.mxfp8_tensor import MXFP8Quantizer

    if not isinstance(args.input_quantizer, MXFP8Quantizer):
        return 1.0
    if torch.cuda.is_current_stream_capturing():
        return 1.0
    scale = float(callback())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Distributed weight gradient scale must be finite and positive")
    return scale


def mark_weight_gradient_scale(gradient: torch.Tensor, scale: float) -> None:
    """Mark a freshly produced wgrad so its finalizer can avoid duplicate scaling."""
    if gradient.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Scaled weight gradients must use FP32 or BF16")
    if hasattr(gradient, "_te_wgrad_scale"):
        raise ValueError("Weight gradient already carries a scaling marker")
    gradient._te_wgrad_scale = scale


def materialize_weight_for_forward(weights: Any) -> List[Any]:
    """Prepare the weight(s) fed to the forward GEMM, always returned as a list.

    Args:
        weights: the module's weight(s) -- a single weight (Linear) or the full per-expert list
            (GroupedLinear). A bare weight is treated as a one-element list.

    Returns:
        - Distributed group: the leader ``weights[0]`` all-gathers/coalesces the whole group and
          returns all N materialized weights; the follower entries ``weights[1:]`` are ignored
          here (the leader already holds references to its group).
        - Otherwise: the input weights, unchanged.
    """
    if not isinstance(weights, (list, tuple)):
        weights = [weights]
    leader = weights[0]
    if is_distributed_weight(leader):
        out = leader.materialize_group_for_forward()
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return list(weights)


def materialize_weight_for_backward(weights: Any) -> List[Any]:
    """Backward-GEMM mirror of :func:`materialize_weight_for_forward` (same contract)."""
    if not isinstance(weights, (list, tuple)):
        weights = [weights]
    leader = weights[0]
    if is_distributed_weight(leader):
        out = leader.materialize_group_for_backward()
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return list(weights)


def finalize_weight_grads(weights: Any, wgrads: List[Any]) -> List[Any]:
    """Finalize a weight group's grad(s), mirroring :func:`materialize_weight_for_backward`.

    Delegates to the leader's :meth:`DistributedWeight.finalize_group_grads` (which defines the
    in-place / dummy / async-``None`` return contract); returns ``wgrads`` unchanged when not
    distributed.
    """
    if not isinstance(weights, (list, tuple)):
        weights = [weights]
    leader = weights[0]
    if is_distributed_weight(leader):
        out = leader.finalize_group_grads(wgrads if len(wgrads) > 1 else wgrads[0])
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return list(wgrads)
