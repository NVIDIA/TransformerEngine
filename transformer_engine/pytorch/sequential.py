from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, Optional

import torch

class BaseOp(torch.nn.Module):

    is_fused_op = False

    def __init__(self) -> None:
        ...

    def unfused_op_forward(self, input: torch.Tensor) -> tuple[torch.Tensor, Any, list[Optional[torch.Tensor]]]:
        raise NotImplementedError("Unfused operation forward pass is not implemented")

    def fused_op_forward(self, input: torch.Tensor) -> tuple[torch.Tensor, list[Any], list[list[Optional[torch.Tensor]]]]:
        if self.is_fused_op:
            raise NotImplementedError("Fused operation forward pass is not implemented")
        out, ctx, saved_tensors = self.unfused_op_forward(input)
        return out, [ctx], [saved_tensors]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out, _, _ = fused_op_forward(input)  ### TODO Handle differentiability
        return out

    def make_backward_op(self) -> BaseBackwardOp:
        raise NotImplementedError("Operation backward pass is not implemented")


class BackwardOp:

    is_fused_op = False

    def __init__(
        self,
        unfused_ops: Optional[list[BaseOp]] = None,
    ) -> None:
        self._unfused_ops = unfused_ops

    def unfused_op_backward(
        self,
        grad_output: torch.Tensor,
        ctx: Any,
        saved_tensors: list[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        raise NotImplementedError("Unfused operation backward pass is not implemented")

    def fused_op_backward(
        self,
        grad_output: torch.Tensor,
        ctx: list[Any],
        saved_tensors: list[list[Optional[torch.Tensor]]],
    ) -> torch.Tensor:
        if self._is_fused_op:
            raise NotImplementedError("Fused operation backward pass is not implemented")
        return self.unfused_op_backward(grad_output, ctx[0], saved_tensors[0])


class WrapperOp(BaseOp):

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module: torch.nn.Module = module

    def unfused_op_forward(
        self,
        input: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], None]:
        with torch.set_grad_enabled(True):
            x = input.detach().requires_grad_()
            y = self.module(x)
            return y.detach().requires_grad_(), (x, y), None

    def make_backward_op(self) -> None:
        return BackwardWrapperOp()


class BackwardWrapperOp(BackwardOp):

    def __init__(self) -> None:
        super().__init__()

    def unfused_op_backward(
        self,
        grad_output: torch.Tensor,
        ctx: tuple[torch.Tensor, torch.Tensor],
        saved_tensors: Any,
    ) -> torch.Tensor:
        x, y = ctx
        y.backward(grad_output)
        return x.grad


class Sequential(torch.nn.Module):

    class _AutogradFunction(torch.autograd.Function):

        @staticmethod
        def forward(
            ctx: Any,
            input_: torch.Tensor,
            forward_fused_ops: Iterable[BaseOp],
            backward_fused_ops: Iterable[BackwardOp],
        ) -> torch.Tensor:

            # Apply forward ops
            op_ctx_map = dict()
            op_saved_tensors_map = dict()
            for op, op_ids in forward_fused_ops:
                x, op_ctx, op_saved_tensors = op.forward_and_save(x)
                for i, op_id in enumerate(op_ids):
                    op_ctx_map[op_id] = op_ctx[i]
                    op_saved_tensors_map[op_id] = op_saved_tensors[i]

            # Flatten list of saved tensors
            op_saved_tensors_ranges = dict()
            saved_tensors = []
            for op_id, op_saved_tensors in op_saved_tensors_map.items():
                range_start = len(saved_tensors)
                if op_saved_tensors_map.get(op_id, None) is not None:
                    saved_tensors.extend(op_saved_tensors_map[op_id])
                range_end = len(saved_tensors)
                op_saved_tensor_ranges[op_id] = (range_start, range_end)
            ctx.op_saved_tensor_ranges = op_saved_tensor_ranges
            ctx.save_for_backward(*saved_tensors)

            # Other context for backward pass
            ctx.backward_fused_ops = backward_fused_ops
            ctx.op_ctx_map = op_ctx_map

            return x

        @staticmethod
        @torch.autograd.function.once_differentiable()
        def backward(
            ctx: Any,
            grad_output: torch.Tensor,
        ):

            # Unflatten list of saved tensors
            op_saved_tensors_map = {
                op_id: ctx.saved_tensors[slice(range_)]
                for op_id, range_ in ctx.op_saved_tensor_ranges.items()
            }
            ctx.saved_tensors = None

            # Apply backward ops
            dx = grad_output
            for op, op_ids in ctx.backward_fused_ops:
                dx = op.backward(
                    dx,
                    [ctx.op_ctx_map.get(op_id, None) for op_id in op_ids],
                    [op_saved_tensors_map.get(op_id, None) for op_id in op_ids],
                )
                for op_id in op_ids:
                    ctx.op_ctx_map[op_id] = None
                    op_saved_tensors_map[op_id] = None

            return (
                dx,    # input_
                None,  # forward_fused_ops
                None,  # backward_fused_ops
            )

    def __init__(
        self,
        *args: BaseOp | torch.nn.Module,
    ) -> None:

        super().__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

        self._ops_cache: dict[Any, tuple[list[BaseOp], list[BaseOp]]] = dict()

    def add_module(
        self,
        name: str,
        module: Optional[torch.nn.Module],
    ) -> None:
        self._ops_cache.clear()
        super().add_module(name, module)

    def _ops(self) -> tuple[list[BaseOp], list[BaseOp]]:

        # Forward pass
        forward_ops = []
        for module in self._modules.values():
            if isinstance(module, BaseOp):
                ops.append(module)
            else:
                ops.append(WrapperOp(module))

        # Backward pass
        backward_ops = []
        for op in reversed(forward_ops):
            backward_ops.append(op.make_backward_op())

        # Attempt to fuse operations
        forward_ops = self._fuse_ops(forward_ops)
        backward_ops = self._fuse_ops(backward_ops)

        return forward_ops, backward_ops

    def _fuse_ops(self, ops) -> list[BaseOp]:
        ### TODO
        return ops

    def forward(self, input) -> torch.Tensor:
        ops_cache_key = None  ### TODO
        if ops_cache_key not in self._ops_cache_key:
            self._ops_cache[ops_cache_key] = self._ops()
        forward_ops, backward_ops = self._ops_cache[ops_cache_key]
        ### TODO ctx keys
        return Sequential._AutogradFunction.apply(
            input,
            forward_ops,
            backward_ops,
        )

    ### TODO Dunder functions

    ### TODO Handle no_grads
