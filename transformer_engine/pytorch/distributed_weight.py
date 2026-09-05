# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GTP-agnostic weight-parallelism extension point.

TE owns this contract but ships no implementation; the caller (e.g. Megatron GTP) implements the
protocol on the weight and injects it at construction. Dispatchers are list-shaped (Linear -> 1,
GroupedLinear -> N; leader is ``weights[0]``) and no-op on plain tensors.
"""

from typing import Any, List, Protocol, Sequence, runtime_checkable

import torch

__all__ = [
    "DistributedWeight",
    "is_distributed_weight",
    "materialize_weight_for_forward",
    "materialize_weight_for_backward",
    "finalize_weight_grads",
    "weight_grad_buffers",
    "weight_grad_dtype",
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
        """Where the wgrad GEMM writes this weight's gradient.

        The GEMM overwrites it and :meth:`finalize_group_grads` then reduces it, so it needs the
        full unsharded weight shape and the dtype that reduction should use. Called on every
        member of a group, unlike the group hooks above.
        """


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


def weight_grad_buffers(
    weights: Any, weight_shape: Sequence[int], compute_dtype: torch.dtype, device: torch.device
) -> List[torch.Tensor]:
    """Per-weight buffers for the wgrad GEMM to write into.

    A distributed weight brings its own, which skips this allocation and carries ``main_grad``'s
    dtype by construction; anything else gets fresh scratch in the compute dtype.
    """
    if not isinstance(weights, (list, tuple)):
        weights = [weights]
    if is_distributed_weight(weights[0]):
        buffers = [w.grad_buffer() for w in weights]
        # Spot-check the leader: a shard-shaped buffer would let the GEMM write past the end.
        if tuple(buffers[0].shape) != tuple(weight_shape):
            raise RuntimeError(
                f"grad_buffer() returned shape {tuple(buffers[0].shape)}; "
                f"the wgrad GEMM needs {tuple(weight_shape)}."
            )
        return buffers
    packed = torch.empty((len(weights), *weight_shape), dtype=compute_dtype, device=device)
    return list(packed)


def weight_grad_dtype(weights: Any, compute_dtype: torch.dtype) -> torch.dtype:
    """Dtype for a wgrad buffer the caller allocates itself.

    A distributed weight reduces its wgrad before accumulating, so the GEMM must already emit
    ``main_grad``'s dtype -- otherwise the reduction rounds on every rank. Falls back to
    ``compute_dtype`` for plain weights, and for an implementer whose ``main_grad`` the framework
    has not attached yet.
    """
    if not isinstance(weights, (list, tuple)):
        weights = [weights]
    leader = weights[0]
    if is_distributed_weight(leader):
        main_grad = getattr(leader, "main_grad", None)
        if main_grad is not None:
            return main_grad.dtype
    return compute_dtype


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
