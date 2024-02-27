# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import torch

import transformer_engine_extensions as tex
from ...fp8 import (
    FP8GlobalStateManager,
    get_default_fp8_recipe,
)
from ._common import canonicalize_device, is_float8_tensor

@dataclass
class OperationContext:

    to_save: Optional[tuple[Optional[torch.Tensor], ...]] = None
    saved_tensors: Optional[tuple[Optional[torch.Tensor], ...]] = None
    saved_tensors_range: Optional[tuple[int, int]] = None

    def save_for_backward(self, *tensors):
        self.to_save = tensors


class FusableOperation(torch.nn.Module):

    def __init__(self, *unfused_ops: FusableOperation) -> None:
        super().__init__()

        # Unfused operations that comprise this operation
        if any(op.is_fused_op for op in unfused_ops):
            raise ValueError("Attempted to fuse an already-fused operation")
        self._unfused_ops: tuple[FusableOperation] = unfused_ops

        # FP8 metadata objects
        self._fp8_metas: Optional[dict[str, dict[str, Any]]] = None

    @property
    def is_fused_op(self):
        return len(self._unfused_ops) > 0

    def num_fp8_scales(self, mode: str) -> int:
        if self.is_fused_op:
            raise RuntimeError(
                "Attempted to get number of FP8 scaling factors "
                "for fused operation. "
                "Only unfused operations support logic for FP8 meta tensors."
            )
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

    def _get_fp8_meta(self, mode: str) -> dict:
        if self.is_fused_op:
            raise RuntimeError(
                "Attempted to get FP8 meta tensors for fused operation."
                "Only unfused operations support logic for FP8 meta tensors."
            )
        return self._fp8_metas[mode]

    def _pre_forward(self) -> None:

        # Make sure unfused ops are initialized
        for op in self._unfused_ops:
            op._pre_forward()

        if not self.is_fused_op:
            fp8_enabled = (
                FP8GlobalStateManager.is_fp8_enabled()
                or any(is_float8_tensor(param) for param in self.parameters())
            )
            if fp8_enabled and self._fp8_metas is None:
                self._fp8_metas = self._make_fp8_metas()

            ### TODO Update fp8_metas

    def _unfused_op_forward(
        self,
        ctx: OperationContext,
        input: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Forward pass is not implemented for unfused operation"
        )

    def _fused_op_forward(
        self,
        unfused_op_ctxs: list[OperationContext],
        input: torch.Tensor,
        unfused_op_kwargs: list[dict[str, Any]],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Forward pass is not implemented for fused operation"
        )

    def _unfused_op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        raise NotImplementedError(
            "Backward pass is not implemented for unfused operation"
        )

    def _fused_op_backward(
        self,
        unfused_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Iterable[Optional[torch.Tensor]]]]:
        raise NotImplementedError(
            "Backward pass is not implemented for fused operation"
        )

    def _pipeline_forward(
        self,
        unfused_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        unfused_op_kwargs: list[dict[str, Any]],
    ) -> torch.Tensor:
        if self.is_fused_op:
            return self._fused_op_forward(
                unfused_op_ctxs,
                input_,
                unfused_op_kwargs,
            )
        else:
            return self._unfused_op_forward(
                unfused_op_ctxs[0],
                input_,
                **unfused_op_kwargs[0],
            )

    def _pipeline_backward(
        self,
        unfused_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Iterable[Optional[torch.Tensor]]]]:
        if self.is_fused_op:
            return self._fused_op_backward(unfused_op_ctxs, grad_output)
        else:
            grad_input, grad_params = self._unfused_op_backward(
                unfused_op_ctxs[0],
                grad_output,
            )
            return grad_input, [grad_params]

    def forward(self, input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        from ..pipeline import Pipeline
        return Pipeline([self], fuse_ops=False)(input, [kwargs])
