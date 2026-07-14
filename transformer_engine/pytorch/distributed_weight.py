# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GTP-agnostic weight-parallelism extension point.

TE owns this contract but ships no implementation; the caller (e.g. Megatron GTP) implements the
protocol on the weight and injects it at construction. Dispatchers are list-shaped (Linear -> 1,
GroupedLinear -> N; leader is ``weights[0]``) and no-op on plain tensors.
"""

from typing import Any, List, Protocol, runtime_checkable

import torch

__all__ = [
    "DistributedWeight",
    "is_distributed_weight",
    "materialize_weight_for_forward",
    "materialize_weight_for_backward",
    "finalize_weight_grads",
]


@runtime_checkable
class DistributedWeight(Protocol):
    """Structural interface for a custom-weight-parallel weight (AG for the GEMM, reduce/RS the
    grad, re-materialize in backward). Duck-typed ``typing.Protocol``: implementers need not
    subclass it, and all state (shards, group, async handles) lives outside TE on the implementer.
    """

    # Capability marker: True on an implementer, absent on plain tensors; TE's fwd/bwd gate on it.
    is_distributed_weight: bool

    def materialize_group_for_forward(self) -> Any:
        """Return the tensor(s) to feed the forward GEMM (may all-gather shards)."""

    def materialize_group_for_backward(self) -> Any:
        """Re-materialize the full weight(s) for the backward GEMMs."""

    def finalize_group_grads(self, wgrads: Any) -> Any:
        """Post-process freshly computed weight grad(s) (e.g. reduce-scatter)."""

    def grad_buffer(self) -> torch.Tensor:
        """The gradient accumulation buffer for this weight."""


def is_distributed_weight(weight: Any) -> bool:
    """True if ``weight`` participates in custom weight parallelism (False on plain tensors)."""
    return bool(getattr(weight, "is_distributed_weight", False))


def materialize_weight_for_forward(weight: Any) -> List[Any]:
    """Materialize a (leader) weight for the forward GEMM. If distributed, delegate to it (it may
    coalesce the whole group and return several tensors); else return it unchanged. Always a list.
    """
    if is_distributed_weight(weight):
        out = weight.materialize_group_for_forward()
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return [weight]


def materialize_weight_for_backward(weight: Any) -> List[Any]:
    """Backward counterpart of :func:`materialize_weight_for_forward`."""
    if is_distributed_weight(weight):
        out = weight.materialize_group_for_backward()
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return [weight]


def finalize_weight_grads(weight: Any, wgrads: List[Any]) -> List[Any]:
    """Post-process the weight grad(s) of a (leader) weight's group.

    Delegates to the weight when distributed (e.g. reduce-scatter); otherwise returns ``wgrads``
    unchanged.
    """
    if is_distributed_weight(weight):
        out = weight.finalize_group_grads(wgrads if len(wgrads) > 1 else wgrads[0])
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return list(wgrads)
