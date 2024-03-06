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

    to_save: Optional[tuple[Optional[torch.Tensor], ...]] = None
    saved_tensors: Optional[tuple[Optional[torch.Tensor], ...]] = None
    saved_tensors_range: Optional[tuple[int, int]] = None

    def save_for_backward(self, *tensors):
        self.to_save = tensors


class FusableOperation(torch.nn.Module, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def is_fused_op(self) -> bool:
        ...

    @abc.abstractmethod
    def pre_forward(self) -> None:
        ...

    @abc.abstractmethod
    def pipeline_forward(
        self,
        unfused_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        unfused_op_kwargs: list[dict[str, Any]],
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def pipeline_backward(
        self,
        unfused_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Iterable[Optional[torch.Tensor]]]]:
        ...


class UnfusedOperation(FusableOperation, metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        super().__init__()

        # FP8 metadata objects
        self._fp8_metas: Optional[dict[str, dict[str, Any]]] = None

    @property
    def is_fused_op(self) -> bool:
        return False

    def num_fp8_scales(self, mode: str) -> int:
        return 0

    def _make_fp8_metas(self) -> dict[str, Any]:

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
            }

        # Construct FP8 metadata for all tensor types
        return dict(
            input=_make_meta(self.num_fp8_scales("input"), True),
            param=_make_meta(self.num_fp8_scales("param"), True),
            grad_output=_make_meta(self.num_fp8_scales("grad_output"), False),
        )

    def get_fp8_meta(self, mode: str) -> dict:
        if self._fp8_metas is None:
            self._fp8_metas = self._make_fp8_metas()
        return self._fp8_metas[mode]

    def pre_forward(self) -> None:

        # Initialize FP8 metadata if needed
        fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
        if self._fp8_metas is None and fp8_enabled:
            self._fp8_metas = self._make_fp8_metas()

        # Update FP8 metadata if needed
        if fp8_enabled:
            pass
            ### TODO Update fp8_metas

    @abc.abstractmethod
    def op_forward(
        self,
        ctx: OperationContext,
        input: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        ...

    def pipeline_forward(
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

    def pipeline_backward(
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
        from ..pipeline import Pipeline
        return Pipeline([self], fuse_ops=False)(input, [kwargs])


class FusedOperation(FusableOperation):

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
        for op in self.unfused_ops:
            op.pre_forward()

    def pipeline_forward(
        self,
        unfused_op_ctxs: list[OperationContext],
        input: torch.Tensor,
        unfused_op_kwargs: list[dict[str, Any]],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Forward pass is not implemented for fused operation "
            f"({self.__class__.__name__})"
        )

    def pipeline_backward(
        self,
        unfused_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Iterable[Optional[torch.Tensor]]]]:
        raise NotImplementedError(
            "Backward pass is not implemented for fused operation "
            f"({self.__class__.__name__})"
        )

    def forward(
        self,
        input: torch.Tensor,
        unfused_op_kwargs: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if unfused_op_kwargs is None:
            unfused_op_kwargs = [dict() for _ in range(len(self.unfused_ops))]
        from ..pipeline import Pipeline
        return Pipeline([self], fuse_ops=False)(input, unfused_op_kwargs)
