# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import abc
from collections.abc import Iterable
import dataclasses
from typing import Optional

import torch

from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    get_default_fp8_recipe,
)
import transformer_engine_extensions as tex
from ._common import canonicalize_device, is_float8_tensor


@dataclasses.dataclass
class OperationContext:
    """State needed to apply an operation

    Coordinates interactions between ops in compute pipeline, and
    between forward and backward passes.

    """

    # Corresponding operation
    op: UnfusedOperation
    # Next operation in pipeline
    next_op: Optional[UnfusedOperation] = None
    # Previous operation in pipeline
    prev_op: Optional[UnfusedOperation] = None

    # Tensors that have been saved from forward function
    # Note: Available in the backward function, matching tensors from
    # to_save.
    saved_tensors: Optional[tuple[Optional[torch.Tensor], ...]] = None
    # Tensors to save for backward function
    # Note: Expected to be set in the forward function, either
    # directly or with save_for_backward.
    to_save: Optional[tuple[Optional[torch.Tensor], ...]] = None

    # Corresponding range in pipeline's list of saved tensors
    _saved_tensors_range: Optional[tuple[int, int]] = None

    # Whether backward pass is required
    _requires_grad: bool = False

    def save_for_backward(self, *tensors: Optional[torch.Tensor]) -> None:
        """Register tensors to be saved for the backward function

        Expected to be called in the forward function.

        """
        self.to_save = tensors


class FusableOperation(torch.nn.Module, metaclass=abc.ABCMeta):
    """Tensor operation supported by the operation fuser"""

    @property
    @abc.abstractmethod
    def is_fused_op(self) -> bool:
        """Whether this op has been fused out of smaller, unfused ops"""
        ...

    def pre_forward(self) -> None:
        """Preprocessing before forward pass"""
        pass

    def fuser_forward(
        self,
        unfused_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        unfused_op_kwargs: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Forward pass

        Called by `OperationFuser`.

        Parameters
        ----------
        unfused_op_ctxs: list of OperationContext
            Contexts for corresponding unfused operations
        input_: torch.Tensor
            Input tensor
        unfused_op_kwargs: list of dict
            Keyword arguments to forward functions of corresponding
            unfused operations

        Returns
        -------
        torch.Tensor: Output tensor.

        """
        raise NotImplementedError(
            "Forward pass is not implemented for operation "
            f"({self.__class__.__name__})"
        )

    def fuser_backward(
        self,
        unfused_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Iterable[Optional[torch.Tensor]]]]:
        """Backward pass

        Called by `OperationFuser`.

        Parameters
        ----------
        unfused_op_ctxs: list of OperationContext
            Contexts for corresponding unfused operations.
        grad_output: torch.Tensor
            Loss gradient w.r.t. operation output.

        Returns
        -------
        torch.Tensor:
            Loss gradient w.r.t. operation input
        Iterable of iterable of torch.Tensor:
            Loss gradients w.r.t. parameters for corresponding unfused
            operations

        """
        raise NotImplementedError(
            "Backward pass is not implemented for operation "
            f"({self.__class__.__name__})"
        )


class UnfusedOperation(FusableOperation, metaclass=abc.ABCMeta):
    """Single tensor operation supported by the operation fuser

    This class holds parameters and state, even if the actual forward
    and backward passes are performed by a fused operation.

    """

    def __init__(self) -> None:
        super().__init__()

        # FP8 metadata objects
        self._fp8_metas: Optional[dict[str, dict[str, Any]]] = None

    @property
    def is_fused_op(self) -> bool:
        return False

    def num_fp8_scales(self, mode: str) -> int:
        """Number of FP8 scaling factors

        Parameters
        ----------
        mode: {"input", "param", "grad_output"}
            Type of FP8 scaling factor

        """

        return 0

    def _make_fp8_metas(self) -> dict[str, Any]:
        """Construct FP8 metadata"""

        # Shared objects for FP8 metadata
        dtype = torch.float32
        device = canonicalize_device(None)
        recipe = get_default_fp8_recipe()

        def _make_meta(
            num_scales: int,
            is_forward: bool,
        ) -> dict[str, Any]:
            """Construct FP8 metadata for one tensor type"""
            key = FP8GlobalStateManager.get_meta_tensor_key(forward=is_forward)
            meta = tex.FP8TensorMeta()
            meta.scale = torch.ones(num_scales, dtype=dtype, device=device)
            meta.scale_inv = torch.ones(num_scales, dtype=dtype, device=device)
            meta.amax_history = torch.zeros(
                (recipe.amax_history_len, num_scales),
                dtype=dtype,
                device=device,
            )
            return {
                key: meta,
                "recipe": recipe,
                f"{key}_non_weight_mask": None,
            }

        # Construct FP8 metadata for all tensor types
        return dict(
            input=_make_meta(self.num_fp8_scales("input"), True),
            param=_make_meta(self.num_fp8_scales("param"), True),
            grad_output=_make_meta(self.num_fp8_scales("grad_output"), False),
        )

    def get_fp8_meta(self, mode: str) -> dict:
        """FP8 metadata

        Parameters
        ----------
        mode: {"input", "param", "grad_output"}
            Type of FP8 scaling factor

        """
        if self._fp8_metas is None:
            self._fp8_metas = self._make_fp8_metas()
        return self._fp8_metas[mode]

    def pre_forward(self) -> None:
        """Preprocessing before forward pass"""

        # Initialize FP8 metadata if needed
        fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
        if self._fp8_metas is None and fp8_enabled:
            self._fp8_metas = self._make_fp8_metas()

        # Update FP8 metadata if needed
        ### TODO Fix
        ### TODO Fused kernel
        ### TODO amax reductions
        # if fp8_enabled:
        #     if self.num_fp8_scales("input"):
        #         amax_and_scale_update(self.get_fp8_meta("input"), True)
        #     if self.num_fp8_scales("param"):
        #         amax_and_scale_update(self.get_fp8_meta("param"), True)
        #     if self.num_fp8_scales("grad_output"):
        #         amax_and_scale_update(self.get_fp8_meta("grad_output"), False)

    @abc.abstractmethod
    def op_forward(
        self,
        ctx: OperationContext,
        input: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        ctx: OperationContext
            Context to coordinate between forward and backward passes
        input: torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor:
            Output tensor

        """
        ...

    @abc.abstractmethod
    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        """Backward pass

        Parameters
        ----------
        ctx: OperationContext
            Context to coordinate between forward and backward passes
        grad_output: torch.Tensor
            Loss gradient w.r.t. operation output

        Returns
        -------
        torch.Tensor
            Loss gradient w.r.t. operation input
        Iterable of torch.Tensor:
            Loss gradients w.r.t. parameters

        """
        ...

    def fuser_forward(
        self,
        unfused_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        unfused_op_kwargs: list[dict[str, Any]],
    ) -> torch.Tensor:
        return self.op_forward(
            unfused_op_ctxs[0],
            input_,
            **unfused_op_kwargs[0],
        )

    def fuser_backward(
        self,
        unfused_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Iterable[Optional[torch.Tensor]]]]:
        grad_input, grad_params = self.op_backward(
            unfused_op_ctxs[0],
            grad_output,
        )
        return grad_input, [grad_params]

    def forward(self, input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply operation"""
        from ..fuser import OperationFuser
        return OperationFuser([self], fuse_ops=False)(input, [kwargs])


class FusedOperation(FusableOperation):
    """Compound tensor operation supported by the operation fuser

    If the forward or backward passes are defined, they must be
    functionally equivalent to the forward/backward passes of the
    corresponding unfused ops. This class should hold no parameters or
    other state, but should access them from the unfused ops.

    Parameters
    ----------
    unfused_ops: iterable of FusableOperation
        Unfused ops that are interchangeable with this op

    """

    def __init__(
        self,
        unfused_ops: Iterable[FusableOperation],
    ) -> None:
        super().__init__()

        # Unfused operations that comprise this fused operation
        self.unfused_ops: torch.nn.ModuleList = torch.nn.ModuleList(unfused_ops)
        if len(self.unfused_ops) == 0:
            raise ValueError(
                "Attempted to construct a fused operation "
                "without specifying its corresponding unfused operations"
            )

    @property
    def is_fused_op(self) -> bool:
        return True

    def pre_forward(self) -> None:
        """Preprocessing before forward pass"""
        for op in self.unfused_ops:
            op.pre_forward()

    def forward(
        self,
        input: torch.Tensor,
        unfused_op_kwargs: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Apply operation"""
        if unfused_op_kwargs is None:
            unfused_op_kwargs = [dict() for _ in range(len(self.unfused_ops))]
        from ..fuser import OperationFuser
        return OperationFuser([self], fuse_ops=False)(input, unfused_op_kwargs)
