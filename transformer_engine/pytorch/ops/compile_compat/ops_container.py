# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""OpsContainer class for torch.compile compatibility."""

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch._library.opaque_object import register_opaque_type, MemberType

from transformer_engine.common.recipe import Recipe
from ...quantization import FP8GlobalStateManager
from ..op import BasicOperation, OperationContext
from ..fused import (
    fuse_backward_activation_bias,
    fuse_backward_add_rmsnorm,
    fuse_backward_linear_add,
    fuse_backward_linear_scale,
    fuse_forward_linear_bias_activation,
    fuse_forward_linear_bias_add,
    fuse_forward_linear_scale_add,
    fuse_userbuffers_backward_linear,
    fuse_userbuffers_forward_linear,
)
from ...quantized_tensor import prepare_for_saving, restore_from_saved

from .tensor_info import TensorInfo, TensorInfoList, PseudoForwardResult
from .opaque_kwargs import OpaqueKwargs

if TYPE_CHECKING:
    from ..op import FusibleOperation


def _split_tuple(t: tuple, idx: int) -> tuple[tuple, tuple]:
    """Split tuple at index."""
    return t[:idx], t[idx:]


class OpsContainer:
    """Container holding the list of BasicOperations for torch.compile compatibility.

    This class encapsulates the operation pipeline and provides methods that
    are invisible to torch.compile (fusion happens inside custom ops).

    Key methods:
    - run_pseudo_forward: Shape inference (compile time) and ctx reconstruction (backward time)
    - run_forward: Actual forward pass with fusion (analogous to OperationFuser.forward)
    - run_backward: Actual backward pass with fusion (analogous to OperationFuser.backward)
    """

    def __init__(self, basic_ops: list[BasicOperation]) -> None:
        self.basic_ops = basic_ops
        self._num_ops = len(basic_ops)

        # Flatten list of parameters
        self._basic_op_params = [list(op.parameters()) for op in basic_ops]
        self._basic_op_num_params = list(map(len, self._basic_op_params))

        # Number of extra inputs/outputs per op
        self._basic_op_num_extra_inputs = [op.num_extra_inputs for op in basic_ops]
        self._basic_op_num_extra_outputs = [op.num_extra_outputs for op in basic_ops]
        self.num_extra_inputs = sum(self._basic_op_num_extra_inputs)
        self.num_extra_outputs = sum(self._basic_op_num_extra_outputs)

        # Cached fusion state
        self._forward_ops: Optional[list[tuple[Any, list[int]]]] = None
        self._backward_ops: Optional[list[tuple[Any, list[int]]]] = None
        self._cached_recipe_type: Optional[type] = None
        self._cached_first_op_requiring_backward: int = 0
        self._last_amax_history_len: int = 0

    def _fuse_forward_ops(
        self,
        ops: list[tuple[Any, list[int]]],
        recipe: Optional[Recipe],
    ) -> list[tuple[Any, list[int]]]:
        """Attempt to fuse operations in forward pass."""
        ops = fuse_userbuffers_forward_linear(ops)
        ops = fuse_forward_linear_bias_add(ops)
        ops = fuse_forward_linear_bias_activation(ops)
        ops = fuse_forward_linear_scale_add(ops)
        return ops

    def _fuse_backward_ops(
        self,
        ops: list[tuple[Any, list[int]]],
        recipe: Optional[Recipe],
    ) -> list[tuple[Any, list[int]]]:
        """Attempt to fuse operations in backward pass."""
        ops = fuse_userbuffers_backward_linear(ops)
        ops = fuse_backward_linear_add(ops)
        ops = fuse_backward_linear_scale(ops)
        ops = fuse_backward_activation_bias(ops, recipe)
        ops = fuse_backward_add_rmsnorm(ops)
        return ops

    def _maybe_fuse_ops(
        self,
        recipe: Optional[Recipe],
        first_op_requiring_backward: int,
    ) -> None:
        """Attempt to fuse operations if necessary."""
        from transformer_engine.common.recipe import DelayedScaling

        # Check if fusion parameters changed
        need_reset = False
        recipe_type = type(recipe) if recipe is not None else None

        # Always need reset if not yet initialized
        if self._forward_ops is None:
            need_reset = True
        elif (recipe_type, first_op_requiring_backward) != (
            self._cached_recipe_type,
            self._cached_first_op_requiring_backward,
        ):
            need_reset = True
        elif (
            recipe is not None
            and recipe.delayed()
            and self._last_amax_history_len != recipe.amax_history_len
        ):
            need_reset = True

        if not need_reset:
            return

        # Reset recipe state on all ops
        for op in self.basic_ops:
            op.reset_recipe_state(recipe=recipe)

        # Check if first iteration
        if self._cached_recipe_type is None:
            for op in self.basic_ops:
                op.pre_first_fuser_forward()

        # Prepare op lists for fusion
        forward_ops = [(op, [idx]) for idx, op in enumerate(self.basic_ops)]
        backward_ops = list(reversed(forward_ops[first_op_requiring_backward:]))

        # Fuse ops
        self._forward_ops = self._fuse_forward_ops(forward_ops, recipe)
        self._backward_ops = self._fuse_backward_ops(backward_ops, recipe)

        # Cache fusion params
        self._cached_recipe_type = recipe_type
        self._cached_first_op_requiring_backward = first_op_requiring_backward

        # Save amax history length for delayed scaling
        if isinstance(recipe, DelayedScaling):
            self._last_amax_history_len = recipe.amax_history_len
        else:
            self._last_amax_history_len = 0

    def run_pseudo_forward(
        self,
        input_info: TensorInfo,
        extra_inputs_info: TensorInfoList,
        kwargs_opaque: OpaqueKwargs,
    ) -> PseudoForwardResult:
        """Shape inference and ctx reconstruction - no actual computation.

        Used for:
        1. Compile-time shape inference (register_fake)
        2. Backward ctx reconstruction (avoids storing ctx in opaque container)

        Args:
            input_info: TensorInfo describing input tensor
            extra_inputs_info: TensorInfo for extra inputs
            kwargs_opaque: Per-op keyword arguments

        Returns:
            PseudoForwardResult with output shape, tensors_to_save shapes, and ctx data
        """
        kwargs_list = kwargs_opaque.to_dicts()

        current_info = input_info
        all_tensors_to_save_info: list[TensorInfo] = []
        all_extra_outputs_info: list[TensorInfo] = []
        all_tensor_sources: list[int] = []
        all_ctx_data: dict[str, Any] = {}

        # Get items from TensorInfoList for slicing
        extra_inputs_info_list = extra_inputs_info.items

        # Track cumulative offsets for params and extra_inputs
        # Source encoding: 0 = x, 1..num_total_params = params, num_total_params+1.. = extra_inputs
        num_total_params = sum(len(list(op.parameters())) for op in self.basic_ops)
        param_offset = 0  # Cumulative param count before current op
        extra_inputs_offset = 0  # Cumulative extra inputs count before current op

        for op_idx, op in enumerate(self.basic_ops):
            # Get extra inputs for this op
            num_extra = op.num_extra_inputs
            op_extra_inputs_info = tuple(
                extra_inputs_info_list[extra_inputs_offset : extra_inputs_offset + num_extra]
            )

            # Get kwargs for this op
            op_kwargs = kwargs_list[op_idx] if op_idx < len(kwargs_list) else {}

            # Add internal flag to indicate if this is the first op
            # (affects tensor_sources for input - first op's input aliases original x)
            op_kwargs = dict(op_kwargs)  # Copy to avoid modifying original
            op_kwargs["_is_first_op"] = op_idx == 0

            # Call pseudo_forward on the operation
            op_result = op.pseudo_forward(
                current_info,
                op_extra_inputs_info,
                **op_kwargs,
            )

            # Accumulate results
            current_info = op_result.output_info
            all_tensors_to_save_info.extend(op_result.tensors_to_save_info)
            all_extra_outputs_info.extend(op_result.extra_outputs_info)

            # Adjust tensor_sources for global offsets
            # Op-local sources: 0=x, 1..num_op_params=params, num_op_params+1..=extra_inputs
            # Global sources: 0=x, 1..num_total_params=params, num_total_params+1..=extra_inputs
            num_op_params = len(list(op.parameters()))
            for src in op_result.tensor_sources:
                if src == -1:
                    # New tensor, no adjustment
                    all_tensor_sources.append(-1)
                elif src == 0:
                    # Input x, no adjustment
                    all_tensor_sources.append(0)
                elif src <= num_op_params:
                    # Op-local param index -> global param index
                    # src=1 means op's params[0], global is params[param_offset + 0]
                    global_param_idx = param_offset + (src - 1)
                    all_tensor_sources.append(1 + global_param_idx)
                else:
                    # Op-local extra_input index -> global extra_input index
                    # src=num_op_params+1 means op's extra_inputs[0]
                    local_extra_idx = src - num_op_params - 1
                    global_extra_idx = extra_inputs_offset + local_extra_idx
                    all_tensor_sources.append(1 + num_total_params + global_extra_idx)

            # Store ctx_data with op index prefix
            for key, value in op_result.ctx_data.items():
                all_ctx_data[f"op_{op_idx}_{key}"] = value

            # Update offsets
            param_offset += num_op_params
            extra_inputs_offset += num_extra

        return PseudoForwardResult(
            output_info=current_info,
            tensors_to_save_info=all_tensors_to_save_info,
            extra_outputs_info=all_extra_outputs_info,
            ctx_data=all_ctx_data,
            tensor_sources=all_tensor_sources,
        )

    def run_forward(
        self,
        x: torch.Tensor,
        recipe: Optional[Recipe],
        kwargs_opaque: OpaqueKwargs,
        params: list[torch.Tensor],
        extra_inputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Actual forward pass - analogous to OperationFuser forward.

        Fusion happens HERE (inside custom op, invisible to torch.compile):
        - Fuses operations based on recipe
        - Handles quantizers (prev_op_grad_output_quantizer, next_op_input_quantizer)
        - Calls pre_fuser_forward on each op

        Args:
            x: Input tensor
            recipe: FP8 recipe (or None)
            kwargs_opaque: Per-op keyword arguments
            params: Flattened list of all parameters
            extra_inputs: Flattened list of extra inputs

        Returns:
            (output, tensors_to_save, extra_outputs)
        """
        kwargs_list = kwargs_opaque.to_dicts()

        # Determine which ops require backward
        # NOTE: Custom ops run in no_grad context, so we check x.requires_grad instead of is_grad_enabled()
        if x.requires_grad:
            first_op_requiring_backward = 0
        else:
            first_op_requiring_backward = self._num_ops
            param_idx = 0
            extra_idx = 0
            for op_idx, op in enumerate(self.basic_ops):
                num_params = self._basic_op_num_params[op_idx]
                num_extra = op.num_extra_inputs
                op_params = params[param_idx : param_idx + num_params]
                op_extra = extra_inputs[extra_idx : extra_idx + num_extra]
                param_idx += num_params
                extra_idx += num_extra

                if any(t.requires_grad for t in op_params) or any(
                    t.requires_grad for t in op_extra
                ):
                    first_op_requiring_backward = op_idx
                    break

        # Fuse operations if needed
        self._maybe_fuse_ops(recipe, first_op_requiring_backward)

        # Unflatten extra inputs
        extra_inputs_copy = list(extra_inputs)
        basic_op_extra_inputs = []
        for op in self.basic_ops:
            xs, extra_inputs_copy = _split_tuple(tuple(extra_inputs_copy), op.num_extra_inputs)
            basic_op_extra_inputs.append(xs)

        # Create operation contexts
        basic_op_ctxs = [OperationContext() for _ in range(self._num_ops)]

        # Pre-forward initialization
        for idx, op in enumerate(self.basic_ops):
            op.pre_fuser_forward(requires_grad=idx >= first_op_requiring_backward)

        # Run forward ops
        extra_outputs = [None] * self._num_ops
        for op, basic_op_idxs in self._forward_ops:
            # Set requires_grad on contexts
            for idx in basic_op_idxs:
                basic_op_ctxs[idx].requires_grad = idx >= first_op_requiring_backward

            # Get quantizers
            prev_op_idx = basic_op_idxs[0] - 1
            prev_op = self.basic_ops[prev_op_idx] if prev_op_idx >= 0 else None
            prev_op_grad_output_quantizer = None
            if prev_op is not None:
                prev_op_grad_output_quantizer = prev_op.get_grad_output_quantizer()

            next_op_idx = basic_op_idxs[-1] + 1
            next_op = self.basic_ops[next_op_idx] if next_op_idx < self._num_ops else None
            next_op_input_quantizer = None
            if next_op is not None:
                next_op_input_quantizer = next_op.get_input_quantizer()

            # Forward op
            x, fused_op_extra_outputs = op.fuser_forward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                x,
                basic_op_extra_inputs=[basic_op_extra_inputs[idx] for idx in basic_op_idxs],
                prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
                next_op_input_quantizer=next_op_input_quantizer,
                basic_op_kwargs=[
                    kwargs_list[idx] if idx < len(kwargs_list) else {} for idx in basic_op_idxs
                ],
            )

            for idx, ys in zip(basic_op_idxs, fused_op_extra_outputs):
                for y in ys:
                    y.requires_grad_(idx >= first_op_requiring_backward)
                extra_outputs[idx] = ys

        # Flatten saved tensors
        tensors_to_save = []
        for ctx in basic_op_ctxs:
            if ctx.to_save is not None:
                tensors_to_save.extend(ctx.to_save)
                ctx.to_save = None

        # Flatten extra outputs
        extra_outputs_flat = []
        for idx, ys in enumerate(extra_outputs):
            if ys is not None:
                extra_outputs_flat.extend(ys)

        return x, tensors_to_save, extra_outputs_flat

    def run_backward(
        self,
        grad_output: torch.Tensor,
        grad_extra_outputs: list[torch.Tensor],
        ctx_data: dict[str, Any],
        tensors_saved: list[torch.Tensor],
        params: list[torch.Tensor],
        recipe: Optional[Recipe],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Actual backward pass - analogous to OperationFuser backward.

        Fusion happens HERE (inside custom op, invisible to torch.compile):
        - Uses fused backward ops
        - Handles quantizers

        Args:
            grad_output: Gradient w.r.t. output
            grad_extra_outputs: Gradients w.r.t. extra outputs
            ctx_data: Non-tensor context data from pseudo_forward
            tensors_saved: Saved tensors from forward
            params: Flattened list of all parameters
            recipe: FP8 recipe (or None)

        Returns:
            (grad_input, grad_params, grad_extra_inputs)
        """

        # Restore saved tensors to contexts
        basic_op_ctxs = [OperationContext() for _ in range(self._num_ops)]

        # Distribute ctx_data back to individual contexts
        for key, value in ctx_data.items():
            # Parse "op_{idx}_{attr}" format
            parts = key.split("_", 2)
            if len(parts) >= 3 and parts[0] == "op":
                op_idx = int(parts[1])
                attr_name = parts[2]
                setattr(basic_op_ctxs[op_idx], attr_name, value)

        # Distribute saved tensors (need to track ranges per op)
        # This is a simplified version - in practice we'd need to track ranges
        tensor_idx = 0
        for op_idx, ctx in enumerate(basic_op_ctxs):
            # Get number of saved tensors for this op from ctx_data
            num_saved_key = f"op_{op_idx}_num_saved_tensors"
            num_saved = ctx_data.get(num_saved_key, 0)
            if num_saved > 0:
                ctx.saved_tensors = tuple(tensors_saved[tensor_idx : tensor_idx + num_saved])
                tensor_idx += num_saved
            else:
                ctx.saved_tensors = ()
            ctx.requires_grad = True

        # Set quantizers from operations (they weren't stored in ctx_data for picklability)
        for op_idx, (op, ctx) in enumerate(zip(self.basic_ops, basic_op_ctxs)):
            # Check if this op has quantizers (e.g., BasicLinear has them, Bias doesn't)
            quantizers = getattr(op, "_quantizers", None)
            if quantizers is not None and len(quantizers.get("forward", [])) > 0:
                ctx.input_quantizer = op.get_quantizer("forward", 0)
                if len(quantizers.get("forward", [])) > 1:
                    ctx.weight_quantizer = op.get_quantizer("forward", 1)
                if len(quantizers.get("backward", [])) > 0:
                    ctx.grad_output_quantizer = op.get_quantizer("backward", 0)
            # grad_input_quantizer comes from prev op
            if op_idx > 0:
                prev_op = self.basic_ops[op_idx - 1]
                if hasattr(prev_op, "get_grad_output_quantizer"):
                    ctx.grad_input_quantizer = prev_op.get_grad_output_quantizer()

        # Unflatten grad_extra_outputs
        grad_extra_outputs_copy = list(grad_extra_outputs)
        basic_op_grad_extra_outputs = []
        for op in self.basic_ops:
            dys, grad_extra_outputs_copy = _split_tuple(
                tuple(grad_extra_outputs_copy), op.num_extra_outputs
            )
            basic_op_grad_extra_outputs.append(dys)

        # Run backward ops
        dx = grad_output
        grad_params = [None] * self._num_ops
        grad_extra_inputs = [None] * self._num_ops

        # DEBUG: Return fixed shapes
        for op, basic_op_idxs in self._backward_ops:
            # Check if gradients required
            if all(not basic_op_ctxs[idx].requires_grad for idx in basic_op_idxs):
                dx = None
                break

            # Backward op
            dx, fused_op_grad_params, fused_op_grad_extra_inputs = op.fuser_backward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                dx,
                basic_op_grad_extra_outputs=[
                    basic_op_grad_extra_outputs[idx] for idx in basic_op_idxs
                ],
            )

            for idx, dparams in zip(basic_op_idxs, fused_op_grad_params):
                grad_params[idx] = dparams
                basic_op_ctxs[idx].saved_tensors = None

            for idx, dxs in zip(basic_op_idxs, fused_op_grad_extra_inputs):
                grad_extra_inputs[idx] = dxs

        # Flatten grad_params
        grad_params_flat = []
        for idx, dparams in enumerate(grad_params):
            num_params = self._basic_op_num_params[idx]
            if dparams is None:
                dparams = [None] * num_params
            else:
                dparams = list(dparams)
            grad_params_flat.extend(dparams)

        # Flatten grad_extra_inputs
        grad_extra_inputs_flat = []
        for idx, dxs in enumerate(grad_extra_inputs):
            num_extra = self.basic_ops[idx].num_extra_inputs
            if dxs is None:
                dxs = [None] * num_extra
            else:
                dxs = list(dxs)
            grad_extra_inputs_flat.extend(dxs)

        return dx, grad_params_flat, grad_extra_inputs_flat


# Register as reference type (mutable, stateful)
register_opaque_type(
    OpsContainer,
    typ="reference",
    guard_fn=lambda c: (c._num_ops,),
    members={
        "run_pseudo_forward": MemberType.USE_REAL,
        "num_extra_outputs": MemberType.USE_REAL,
        "num_extra_inputs": MemberType.USE_REAL,
    },
)
