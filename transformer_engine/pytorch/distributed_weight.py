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
    "materialize_weights_for_forward",
    "materialize_weights_for_backward",
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
        ...

    def materialize_group_for_backward(self) -> Any:
        """Re-materialize the full weight(s) for the backward GEMMs."""
        ...

    def finalize_group_grads(self, wgrads: Any) -> Any:
        """Post-process freshly computed weight grad(s) (e.g. reduce-scatter)."""
        ...

    def grad_buffer(self) -> torch.Tensor:
        """The gradient accumulation buffer for this weight."""
        ...


def is_distributed_weight(weight: Any) -> bool:
    """True if ``weight`` participates in custom weight parallelism (False on plain tensors)."""
    return bool(getattr(weight, "is_distributed_weight", False))


def materialize_weights_for_forward(weights: List[Any]) -> List[Any]:
    """Materialize a weight group for the forward GEMM. Delegates once to the leader (it may
    coalesce) if distributed, else returns the weights unchanged. Always returns a list.
    """
    lead = weights[0]
    if is_distributed_weight(lead):
        out = lead.materialize_group_for_forward()
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return list(weights)


def materialize_weights_for_backward(weights: List[Any]) -> List[Any]:
    """Backward counterpart of :func:`materialize_weights_for_forward`."""
    lead = weights[0]
    if is_distributed_weight(lead):
        out = lead.materialize_group_for_backward()
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return list(weights)


def finalize_weight_grads(weights: List[Any], wgrads: List[Any]) -> List[Any]:
    """Post-process the weight grads of a homogeneous group.

    Delegates to the group leader when distributed (e.g. reduce-scatter);
    otherwise returns ``wgrads`` unchanged.
    """
    lead = weights[0]
    if is_distributed_weight(lead):
        out = lead.finalize_group_grads(wgrads if len(wgrads) > 1 else wgrads[0])
        return list(out) if isinstance(out, (list, tuple)) else [out]
    return list(wgrads)
