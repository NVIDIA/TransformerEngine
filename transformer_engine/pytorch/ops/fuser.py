# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Manager class for a pipeline of fusible operations."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
import itertools
from typing import Any, Optional, TypeAlias

import torch

from ..quantization import FP8GlobalStateManager, Recipe, DelayedScaling
from ..quantized_tensor import prepare_for_saving, restore_from_func_ctx
from .op import (
    BasicOperation,
    FusibleOperation,
    FusedOperation,
    OperationContext,
)


def _split_tuple(t: tuple, idx: int) -> tuple[tuple, tuple]:
    """Split tuple at index"""
    return t[:idx], t[idx:]


# Lazily imported function used in _is_graph_capturing
_is_graph_capturing_function: Optional[Callable[[], bool]] = None


def _is_graph_capturing() -> bool:
    """Whether function is called within ``make_graphed_callables``

    Avoid circular import with lazy import.

    """
    global _is_graph_capturing_function
    if _is_graph_capturing_function is None:
        from ..graph import is_graph_capturing

        _is_graph_capturing_function = is_graph_capturing
    return _is_graph_capturing_function()


# Type alias for a function that may perform operation fusion
OperationFusionFunction: TypeAlias = (
    "Callable[tuple[list[FusibleOperation], ...], list[FusibleOperation]]"
)


class _OperationFuserAutogradFunction(torch.autograd.Function):
    """Autograd function for a pipeline of operations

    Autograd must be done at the pipeline level since we may apply
    different fusions in the forward and backward passes.

    """

    # pylint: disable=unused-argument
    @staticmethod
    def forward(
        func_ctx: Optional[torch.autograd.function.FunctionCtx],
        input_: torch.Tensor,
        fuser: OperationFuser,
        basic_op_kwargs: list[dict[str, Any]],
        set_output_requires_grad: bool,
        *params_and_extra_inputs: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass

        Parameters
        ----------
        func_ctx: torch.autograd.function.FunctionCtx
            Context for PyTorch autograd function
        input_: torch.Tensor
            Input to first operation in pipeline
        fuser: OperationFuser
            Container for the pipeline of operations to run
        basic_op_kwargs: list of dict
            Keyword arguments to BasicOperation
        set_output_requires_grad: bool
            Whether to set ``requires_grad`` flags on returned tensors
        *params_and_extra_inputs: torch.Tensor
            Other tensor inputs to include in autograd graph. Consists
            of parameter tensors, followed by extra operation inputs.

        Returns
        -------
        Output tensor(s). If none of the operations have any extra
        tensor outputs, then the pipeline's output tensor is returned.
        Otherwise, a tuple with the pipeline's output tensor and extra
        tensor outputs is returned.

        """

        # Operation autograd contexts
        basic_op_ctxs = [OperationContext() for _ in range(fuser._num_basic_ops)]

        # Mark input tensors as not deletable in backward
        for tensor in (input_,) + params_and_extra_inputs:
            tensor._do_not_clear = True

        # Place user provided extra inputs into their basic-op slots. Slots bound to
        # internal channels are filled lazily as their producers execute.
        extra_inputs = params_and_extra_inputs[len(fuser._flat_basic_op_params) :]
        basic_op_extra_inputs: list[list[Optional[torch.Tensor]]] = [
            [None] * op.num_extra_inputs for op in fuser._basic_ops
        ]
        for tensor, (op_idx, input_idx) in zip(
            extra_inputs,
            fuser._external_extra_input_slots,
        ):
            basic_op_extra_inputs[op_idx][input_idx] = tensor

        # Apply forward ops
        x = input_
        extra_outputs: list[Optional[Sequence[Optional[torch.Tensor]]]] = [
            None
        ] * fuser._num_basic_ops
        for op, basic_op_idxs in fuser._forward_ops:

            # Set if backward op is required
            for idx in basic_op_idxs:
                basic_op_ctxs[idx].requires_grad = idx >= fuser.first_op_requiring_backward

            # Resolve internal channel inputs from outputs of
            # earlier basic ops. When a fusion contains both producer and
            # consumer, leave the consumer slot unset so the fused op can
            # wire the channel itself
            for idx in basic_op_idxs:
                for input_idx, source in enumerate(fuser._basic_op_extra_input_sources[idx]):
                    if source is None:
                        continue
                    producer_idx, output_idx = source
                    if producer_idx in basic_op_idxs:
                        # fused op will wire the channel itself internally
                        continue
                    producer_outputs = extra_outputs[producer_idx]
                    basic_op_extra_inputs[idx][input_idx] = producer_outputs[output_idx]

            # Prepare args for op forward
            extra_inputs = [basic_op_extra_inputs[idx] for idx in basic_op_idxs]
            prev_op_idx = basic_op_idxs[0] - 1
            prev_op = fuser._basic_ops[prev_op_idx] if prev_op_idx >= 0 else None
            prev_op_grad_output_quantizer = None
            if prev_op is not None:
                prev_op_grad_output_quantizer = prev_op.get_grad_output_quantizer()
            next_op_idx = basic_op_idxs[-1] + 1
            next_op = fuser._basic_ops[next_op_idx] if next_op_idx < fuser._num_basic_ops else None
            next_op_input_quantizer = None
            if next_op is not None:
                next_op_input_quantizer = next_op.get_input_quantizer()

            x, fused_op_extra_outputs = op.fuser_forward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                x,
                basic_op_extra_inputs=extra_inputs,
                prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
                next_op_input_quantizer=next_op_input_quantizer,
                basic_op_kwargs=[basic_op_kwargs[idx] for idx in basic_op_idxs],
            )
            if len(fused_op_extra_outputs) != len(basic_op_idxs):
                raise RuntimeError(
                    f"Expected {type(op).__name__} to generate extra outputs for "
                    f"{len(basic_op_idxs)} basic operations, "
                    f"but got {len(fused_op_extra_outputs)}"
                )
            for idx, ys in zip(basic_op_idxs, fused_op_extra_outputs):
                num_extra_outputs = fuser._basic_ops[idx].num_extra_outputs
                if len(ys) != num_extra_outputs:
                    raise RuntimeError(
                        f"Expected op {idx} to generate {num_extra_outputs} extra outputs, "
                        f"but got {len(ys)}"
                    )
                for output_idx, y in enumerate(ys):
                    if y is None:
                        # Extra output can be None if it is not required by any operations outside the fusion
                        # and is not required to be outputted to the caller.
                        output_to_caller = fuser._basic_op_extra_output_to_caller[idx][output_idx]
                        consumers = fuser._basic_op_extra_output_consumers[idx][output_idx]
                        needed_outside_fusion = any(
                            consumer_idx not in basic_op_idxs for consumer_idx in consumers
                        )
                        if output_to_caller:
                            raise RuntimeError(
                                f"Op {idx} extra output {output_idx} is public, "
                                f"but {type(op).__name__} returned None"
                            )
                        if needed_outside_fusion:
                            raise RuntimeError(
                                f"Op {idx} extra output {output_idx} is required by an "
                                "operation outside its forward fusion, "
                                f"but {type(op).__name__} returned None"
                            )
                        continue
                    if (
                        set_output_requires_grad
                        and idx >= fuser.first_op_requiring_backward
                        and y.is_floating_point()
                    ):
                        y.requires_grad_(True)
                extra_outputs[idx] = ys

        # Collect caller-visible extra outputs in basic-op and slot order.
        extra_outputs_flat = [
            extra_outputs[op_idx][output_idx]
            for op_idx, output_idx in fuser._public_extra_output_slots
        ]

        # Save context for backward pass
        if func_ctx is not None:

            # Flatten list of saved tensors
            to_save = []
            for ctx in basic_op_ctxs:
                range_start = len(to_save)
                if ctx.to_save is not None:
                    to_save.extend(ctx.to_save)
                range_end = len(to_save)
                ctx.to_save = None
                ctx._saved_tensors_range = (range_start, range_end)

            # Save tensors for backward
            tensors_to_save, tensor_objects = prepare_for_saving(*to_save)
            func_ctx.save_for_backward(*tensors_to_save)
            func_ctx.tensor_objects = tensor_objects

            # Whether to perform recipe update in backward pass
            is_first_module = False
            if fuser.first_op_requiring_backward < fuser._num_basic_ops:
                is_first_module = FP8GlobalStateManager.is_first_fp8_module()

            # Other context
            func_ctx.backward_ops = fuser._backward_ops
            func_ctx.basic_ops = fuser._basic_ops
            func_ctx.basic_op_ctxs = basic_op_ctxs
            func_ctx.basic_op_num_params = fuser._basic_op_num_params
            func_ctx.num_extra_outputs = len(extra_outputs_flat)
            func_ctx.external_extra_input_slots = fuser._external_extra_input_slots
            func_ctx.public_extra_output_slots = fuser._public_extra_output_slots
            func_ctx.basic_op_extra_output_channels = fuser._basic_op_extra_output_channels
            func_ctx.basic_op_extra_output_consumers = fuser._basic_op_extra_output_consumers
            func_ctx.basic_op_extra_input_sources = fuser._basic_op_extra_input_sources
            func_ctx.is_first_module = is_first_module

        # Mark output tensors as not deletable in backward
        for tensor in itertools.chain(
            (x,),
            (y for ys in extra_outputs for y in ys if y is not None),
        ):
            tensor._do_not_clear = True

        if set_output_requires_grad:
            x.requires_grad_(fuser.first_op_requiring_backward < fuser._num_basic_ops)

        if extra_outputs_flat:
            return x, *extra_outputs_flat

        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        func_ctx: Any,
        grad_output: torch.Tensor,
        *grad_extra_outputs: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Backward pass"""

        # Operations and autograd state
        backward_ops = func_ctx.backward_ops
        basic_ops = func_ctx.basic_ops
        basic_op_ctxs = func_ctx.basic_op_ctxs

        # Restore saved tensors
        saved_tensors = restore_from_func_ctx(func_ctx)

        # Unflatten list of saved tensors
        for ctx in basic_op_ctxs:
            ctx.saved_tensors = saved_tensors[slice(*ctx._saved_tensors_range)]
            ctx._saved_tensors_range = None

        # Channel wiring saved from forward
        basic_op_extra_output_channels = func_ctx.basic_op_extra_output_channels
        basic_op_extra_output_consumers = func_ctx.basic_op_extra_output_consumers
        basic_op_extra_input_sources = func_ctx.basic_op_extra_input_sources

        # Place caller-provided extra-output grads into their basic-op slots.
        # Gradients from internal channel consumers are added during backward.
        if len(grad_extra_outputs) != func_ctx.num_extra_outputs:
            raise ValueError(
                f"Expected grads for {func_ctx.num_extra_outputs} extra tensor outputs, "
                f"but got {len(grad_extra_outputs)}"
            )
        basic_op_grad_extra_outputs: list[list[Optional[torch.Tensor]]] = [
            [None] * op.num_extra_outputs for op in basic_ops
        ]
        for grad, (op_idx, output_idx) in zip(
            grad_extra_outputs,
            func_ctx.public_extra_output_slots,
        ):
            basic_op_grad_extra_outputs[op_idx][output_idx] = grad

        # Apply backward ops
        dx = grad_output
        grad_params = [None for _ in range(len(basic_ops))]
        grad_extra_inputs = [None for _ in range(len(basic_ops))]
        channel_grads: dict[str, torch.Tensor] = {}
        for op, basic_op_idxs in reversed(backward_ops):

            # Stop if no more gradients are required
            if all(not basic_op_ctxs[idx].requires_grad for idx in basic_op_idxs):
                dx = None
                break

            # Backward op. Supply gradients accumulated from every consumer of
            # each internal channel.
            for idx in basic_op_idxs:
                for output_idx, channel in enumerate(basic_op_extra_output_channels[idx]):
                    if basic_op_extra_output_consumers[idx][output_idx]:
                        channel_grad = channel_grads.get(channel)
                        if channel_grad is not None:
                            output_grad = basic_op_grad_extra_outputs[idx][output_idx]
                            basic_op_grad_extra_outputs[idx][output_idx] = (
                                channel_grad if output_grad is None else output_grad + channel_grad
                            )
            grad_extra_outputs = [basic_op_grad_extra_outputs[idx] for idx in basic_op_idxs]
            dx, fused_op_grad_params, fused_op_grad_extra_inputs = op.fuser_backward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                dx,
                basic_op_grad_extra_outputs=grad_extra_outputs,
            )
            for idx, dparams in zip(basic_op_idxs, fused_op_grad_params):
                grad_params[idx] = dparams
                basic_op_ctxs[idx].saved_tensors = None
            for idx, dxs in zip(basic_op_idxs, fused_op_grad_extra_inputs):
                grad_extra_inputs[idx] = dxs
                for input_idx, grad in enumerate(dxs):
                    source = basic_op_extra_input_sources[idx][input_idx]
                    if source is None or grad is None:
                        continue
                    producer_idx, output_idx = source
                    # Producer already ran inside this fusion; the fused op
                    # must apply these grads itself rather than via channel_grads.
                    if producer_idx in basic_op_idxs:
                        continue
                    channel = basic_op_extra_output_channels[producer_idx][output_idx]
                    previous_grad = channel_grads.get(channel)
                    channel_grads[channel] = grad if previous_grad is None else previous_grad + grad

        # Flatten list of parameter gradients
        grad_params_flat = []
        for idx, dparams in enumerate(grad_params):
            num_params = func_ctx.basic_op_num_params[idx]
            if dparams is None:
                dparams = [None for _ in range(num_params)]
            else:
                dparams = list(dparams)
            if len(dparams) != num_params:
                raise RuntimeError(
                    f"Expected op {idx} to generate {num_params} param grads, "
                    f"but got {len(dparams)}"
                )
            grad_params_flat.extend(dparams)

        # Flatten list of parameter gradients
        for idx, dxs in enumerate(grad_extra_inputs):
            num_extra_inputs = basic_ops[idx].num_extra_inputs
            if dxs is None:
                grad_extra_inputs[idx] = (None,) * num_extra_inputs
            elif len(dxs) != num_extra_inputs:
                raise RuntimeError(
                    f"Expected op {idx} to generate grads "
                    f"for {num_extra_inputs} extra inputs, "
                    f"but got {len(dxs)}"
                )

        # Collect the gradient for each public extra input.
        grad_extra_inputs_flat = [
            grad_extra_inputs[op_idx][input_idx]
            for op_idx, input_idx in func_ctx.external_extra_input_slots
        ]

        # Update FP8 scaling factors
        if func_ctx.is_first_module and not _is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            dx,  # input_
            None,  # fuser
            None,  # basic_op_kwargs
            None,  # set_output_requires_grad
            *grad_params_flat,
            *grad_extra_inputs_flat,
        )


class OperationFuser:
    """Manages forward and backward passes for a pipeline of operations

    Operations are fused with three passes (see ``register_*_fusion``):

    1. Joint forward-backward fusions.
    2. Forward-only fusions.
    3. Backward-only fusions.

    Parameters
    ----------
    ops : list of FusibleOperation
        Pipeline of operations

    """

    # Functions to perform operation fusion
    forward_backward_fusion_functions: list[OperationFusionFunction] = []
    forward_fusion_functions: list[OperationFusionFunction] = []
    backward_fusion_functions: list[OperationFusionFunction] = []

    def __init__(
        self,
        ops: list[FusibleOperation],
    ) -> None:

        # Get list of basic operations
        basic_ops = []
        for op in ops:
            if op.is_fused_op:
                basic_ops.extend(op.basic_ops)
            else:
                basic_ops.append(op)
        self._num_basic_ops: int = len(basic_ops)
        self._basic_ops: list[BasicOperation] = basic_ops
        # Capture channel routing from each basic op. If any op later rebinds a
        # channel it flips this flag, essentially invalidating this fuser's forward pass.
        self._channels_stale: bool = False
        for op in self._basic_ops:
            op._capturing_op_fusers.add(self)

        # Number of extra tensor inputs
        self._basic_op_num_extra_inputs: list[int] = list(op.num_extra_inputs for op in basic_ops)
        self._basic_op_extra_input_sources: list[list[Optional[tuple[int, int]]]] = [
            [None] * op.num_extra_inputs for op in basic_ops
        ]
        self._basic_op_extra_output_channels: list[list[Optional[str]]] = [
            list(op._extra_output_channels) for op in basic_ops
        ]
        self._basic_op_extra_output_to_caller: list[list[bool]] = [
            list(op._extra_output_to_caller) for op in basic_ops
        ]
        self._basic_op_extra_output_consumers: list[list[list[int]]] = [
            [[] for _ in range(op.num_extra_outputs)] for op in basic_ops
        ]
        self._external_extra_input_slots: list[tuple[int, int]] = []
        self._public_extra_output_slots: list[tuple[int, int]] = []

        # Find channel producers and reject ambiguous names.
        channel_producers: dict[str, tuple[int, int]] = {}
        for op_idx, op in enumerate(basic_ops):
            for output_idx, channel in enumerate(self._basic_op_extra_output_channels[op_idx]):
                if channel is None:
                    continue
                if channel in channel_producers:
                    producer_idx, _ = channel_producers[channel]
                    raise ValueError(
                        f"Extra tensor channel {channel!r} has multiple producers "
                        f"(ops {producer_idx} and {op_idx})"
                    )
                channel_producers[channel] = (op_idx, output_idx)

        # Resolve inputs. Named inputs must have an earlier producer;
        # unnamed inputs remain public.
        for op_idx, op in enumerate(basic_ops):
            for input_idx, channel in enumerate(op._extra_input_channels):
                if channel is None:
                    self._external_extra_input_slots.append((op_idx, input_idx))
                    continue
                producer = channel_producers.get(channel)
                if producer is None:
                    raise ValueError(
                        f"Extra tensor channel {channel!r} consumed by op {op_idx} "
                        f"({type(op).__name__}) has no producer"
                    )
                producer_idx, _ = producer
                if producer_idx >= op_idx:
                    raise ValueError(
                        f"Extra tensor channel {channel!r} consumed by op {op_idx} "
                        f"({type(op).__name__}) has no earlier producer"
                    )
                self._basic_op_extra_input_sources[op_idx][input_idx] = producer
                producer_idx, output_idx = producer
                self._basic_op_extra_output_consumers[producer_idx][output_idx].append(op_idx)

        # Record caller-visible outputs in stable basic-op and slot order.
        for op_idx, op in enumerate(basic_ops):
            for output_idx in range(op.num_extra_outputs):
                if self._basic_op_extra_output_to_caller[op_idx][output_idx]:
                    self._public_extra_output_slots.append((op_idx, output_idx))

        # Every channel-bound extra input must be wired to a matching producer
        # extra output. External slots remain unbound (source is None).
        for op_idx, sources in enumerate(self._basic_op_extra_input_sources):
            op = basic_ops[op_idx]
            for input_idx, source in enumerate(sources):
                channel = op._extra_input_channels[input_idx]
                if channel is None:
                    if source is not None:
                        raise RuntimeError(
                            f"Extra input {input_idx} of op {op_idx} "
                            f"({type(op).__name__}) is external but has a "
                            f"producer source {source}"
                        )
                    continue
                if source is None:
                    raise RuntimeError(
                        f"Extra input {input_idx} of op {op_idx} "
                        f"({type(op).__name__}) is bound to channel {channel!r} "
                        "without a producer source"
                    )
                producer_idx, output_idx = source
                producer_channel = self._basic_op_extra_output_channels[producer_idx][output_idx]
                if producer_channel != channel:
                    raise ValueError(
                        f"Extra input {input_idx} of op {op_idx} "
                        f"({type(op).__name__}) is bound to channel {channel!r}, "
                        f"but producer op {producer_idx} extra output {output_idx} "
                        f"is bound to {producer_channel!r}"
                    )
        # Used by Sequential to determine the number of extra inputs
        # needed for each OperationFuser module in the sequence.
        self.num_extra_inputs = len(self._external_extra_input_slots)

        # Ops for forward and backward pass, will be populated in maybe_fuse_ops
        self._forward_ops: list[tuple[FusibleOperation, list[int]]]
        self._backward_ops: list[tuple[FusibleOperation, list[int]]]

        # Cache and detect change of state relevant for fusing operations
        self.recipe_type = None
        self.first_op_requiring_backward = 0
        self.backward_override = None
        self._last_amax_history_len = 0

        # Flatten list of parameters
        self._basic_op_params = [list(op.parameters()) for op in self._basic_ops]
        self._basic_op_num_params = list(map(len, self._basic_op_params))
        self._flat_basic_op_params = sum(self._basic_op_params, [])

    @staticmethod
    def _apply_fusions(
        ops: Iterable[FusibleOperation],
        fusion_funcs: Iterable[OperationFusionFunction],
        recipe: Optional[Recipe],
    ) -> list[FusibleOperation]:
        """Apply a sequence of fusion functions to a list of ops"""
        fused_ops = list(ops)
        for func in fusion_funcs:
            fused_ops = func(fused_ops, recipe=recipe)
        return fused_ops

    @staticmethod
    def _map_to_basic_ops(
        fused_ops: Sequence[FusibleOperation],
        basic_ops: Sequence[BasicOperation],
    ) -> list[tuple[FusibleOperation, list[int]]]:
        """Map a fused op list back to basic op indices

        Verifies that the fused ops expand to exactly ``basic_ops`` in
        order, and annotates each (possibly fused) op with the indices
        of the basic ops it covers.

        """

        def raise_mismatch_error() -> None:
            """Throw error indicating invalid op fusion"""
            raise RuntimeError(
                "Found mismatch after fusing operations "
                f"(basic_ops={[o.__class__.__name__ for o in basic_ops]}, "
                f"fused_ops={[o.__class__.__name__ for o in fused_ops]})"
            )

        # Determine basic op indices corresponding to each op
        out = []
        idx = 0
        for op in fused_ops:
            if isinstance(op, FusedOperation):
                idxs = []
                for basic_op in op.basic_ops:
                    if idx >= len(basic_ops) or basic_op is not basic_ops[idx]:
                        raise_mismatch_error()
                    idxs.append(idx)
                    idx += 1
                out.append((op, idxs))
            else:
                if idx >= len(basic_ops) or op is not basic_ops[idx]:
                    raise_mismatch_error()
                out.append((op, [idx]))
                idx += 1
        if idx != len(basic_ops):
            raise_mismatch_error()

        return out

    def maybe_fuse_ops(
        self,
        is_grad_enabled: bool,
        recipe: Optional[Recipe],
        input_: torch.Tensor,
        extra_inputs: list[Iterable[torch.Tensor]],
    ):
        """Attempt to fuse operations if neccesary"""

        # Determine which basic ops require backward
        if not is_grad_enabled:
            first_op_requiring_backward = self._num_basic_ops
        elif input_.requires_grad:
            first_op_requiring_backward = 0
        else:
            first_op_requiring_backward = self._num_basic_ops
            for op_idx in range(self._num_basic_ops):
                op_inputs = itertools.chain(self._basic_op_params[op_idx], extra_inputs[op_idx])
                if any(tensor is not None and tensor.requires_grad for tensor in op_inputs):
                    first_op_requiring_backward = op_idx
                    break

        # Early exit if fusion parameters haven't changed
        need_reset = False
        recipe_type = type(recipe)
        backward_override = recipe.backward_override if recipe is not None else None
        fusion_params = (recipe_type, first_op_requiring_backward, backward_override)
        if fusion_params != (
            self.recipe_type,
            self.first_op_requiring_backward,
            self.backward_override,
        ):
            # Recipe type, backward override, or grad requirements have changed
            need_reset = True
        elif (
            recipe is not None
            and recipe.delayed()
            and self._last_amax_history_len != recipe.amax_history_len
        ):
            # FP8 delayed scaling has changed amax history length
            need_reset = True
        if not need_reset:
            return

        # Reset recipe state
        for op in self._basic_ops:
            op.reset_recipe_state(recipe=recipe)

        # Check if this is the first iteration
        if self.recipe_type is None:
            for op in self._basic_ops:
                op.pre_first_fuser_forward()

        # Apply joint forward-backward fusions first
        joint_ops = OperationFuser._apply_fusions(
            self._basic_ops,
            OperationFuser.forward_backward_fusion_functions,
            recipe=recipe,
        )

        # Apply forward-only and backward-only fusions
        self._forward_ops = OperationFuser._map_to_basic_ops(
            OperationFuser._apply_fusions(
                joint_ops,
                OperationFuser.forward_fusion_functions,
                recipe=recipe,
            ),
            self._basic_ops,
        )
        self._backward_ops = OperationFuser._map_to_basic_ops(
            OperationFuser._apply_fusions(
                joint_ops,
                OperationFuser.backward_fusion_functions,
                recipe=recipe,
            ),
            self._basic_ops,
        )

        # Save current fusion params
        self.recipe_type, self.first_op_requiring_backward, self.backward_override = fusion_params

        # Save amax history length
        if isinstance(recipe, DelayedScaling):
            self._last_amax_history_len = recipe.amax_history_len
        else:
            self._last_amax_history_len = 0

    def __call__(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        *extra_inputs: torch.Tensor,
        basic_op_kwargs: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self._channels_stale:
            raise RuntimeError(
                "Extra tensor channels changed after this OperationFuser captured "
                "its routing. Construct a new OperationFuser."
            )

        # Verify extra input count
        if len(extra_inputs) != self.num_extra_inputs:
            raise ValueError(
                f"Expected {self.num_extra_inputs} extra inputs but got {len(extra_inputs)}"
            )

        # Canonicalize op kwargs
        if basic_op_kwargs is None:
            basic_op_kwargs = [{}] * self._num_basic_ops

        # Place public extra inputs into their basic-op slots. Internal slots
        # are not available until forward executes their producers.
        basic_op_extra_inputs: list[list[Optional[torch.Tensor]]] = [
            [None] * op.num_extra_inputs for op in self._basic_ops
        ]
        for tensor, (op_idx, input_idx) in zip(extra_inputs, self._external_extra_input_slots):
            basic_op_extra_inputs[op_idx][input_idx] = tensor

        # Get environment state
        recipe = None
        if FP8GlobalStateManager.is_fp8_enabled():
            recipe = FP8GlobalStateManager.get_fp8_recipe()
        is_grad_enabled = torch.is_grad_enabled()

        # Attempt to fuse operations if neccesary
        self.maybe_fuse_ops(is_grad_enabled, recipe, input, basic_op_extra_inputs)

        # Initialization before forward
        for idx, op in enumerate(self._basic_ops):
            op.pre_fuser_forward(requires_grad=idx >= self.first_op_requiring_backward)

        # Fuser forward pass
        # Note: We call forward directly when is_grad_enabled=False,
        # which can expose non-leaf tensors to the inner ops. Avoid
        # problems in this case by passing set_output_requires_grad=False.
        args = (
            input,
            self,
            basic_op_kwargs,
            is_grad_enabled,  # set_output_requires_grad
            *self._flat_basic_op_params,
            *extra_inputs,
        )

        if not is_grad_enabled:
            return _OperationFuserAutogradFunction.forward(None, *args)

        return _OperationFuserAutogradFunction.apply(*args)


def register_forward_backward_fusion(
    op_fusion_func: OperationFusionFunction,
    prepend: bool = False,
) -> None:
    """Register a joint forward-backward operation fusion.

    A joint fusion replaces a run of basic ops with a single fused op
    that implements *both* ``fuser_forward`` and ``fuser_backward``.
    Unlike forward-only or backward-only fusions (see
    ``register_forward_fusion`` / ``register_backward_fusion``), the two
    halves need not be individually interchangeable with the unfused
    ops; only the forward/backward pair must be jointly equivalent. This
    lets the forward pass cooperate with its own backward, e.g. saving
    state that only its backward knows how to handle.

    Joint fusions are applied before the forward-only and backward-only
    fusion passes, so a joint fused op is seen by both passes. The
    forward-only and backward-only passes then fuse the remaining ops
    independently.

    The fusion function should have the following signature:

    .. code-block:: python

        func(ops, *, recipe) -> updated ops

    Parameters
    ----------
    op_fusion_func: function
        Function that takes a list of operations and may substitute
        them with fused operations.
    prepend: bool, default = ``False``
        Whether the operation fuser should apply this fusion function
        first within the joint fusion pass. The default is to apply it
        last.

    """
    if prepend:
        OperationFuser.forward_backward_fusion_functions.insert(0, op_fusion_func)
    else:
        OperationFuser.forward_backward_fusion_functions.append(op_fusion_func)


def register_forward_fusion(
    op_fusion_func: OperationFusionFunction,
    prepend: bool = False,
) -> None:
    """Register a forward-only operation fusion.

    A forward-only fusion replaces a run of basic ops with a single
    fused op that implements ``fuser_forward``. Because the backward
    pass is fused independently (see ``register_backward_fusion``), the
    fused op's forward must be interchangeable with the corresponding
    basic ops' forward: it must produce the same output and save state in
    each basic op's context that the unfused backward can consume. If the
    forward and backward need to cooperate (e.g. the forward saving
    reduced state that only a matching backward can handle), use
    ``register_forward_backward_fusion`` instead.

    The fusion function should have the following signature:

    .. code-block:: python

        func(ops, *, recipe) -> updated ops

    Parameters
    ----------
    op_fusion_func: function
        Function that takes a list of operations and may substitute
        them with fused operations.
    prepend: bool, default = ``False``
        Whether the operation fuser should apply this fusion function
        first within the forward fusion pass. The default is to apply it
        last.

    """
    if prepend:
        OperationFuser.forward_fusion_functions.insert(0, op_fusion_func)
    else:
        OperationFuser.forward_fusion_functions.append(op_fusion_func)


def register_backward_fusion(
    op_fusion_func: OperationFusionFunction,
    prepend: bool = False,
) -> None:
    """Register a backward-only operation fusion.

    A backward-only fusion replaces a run of basic ops with a single
    fused op that implements ``fuser_backward``. Because the forward
    pass is fused independently (see ``register_forward_fusion``), the
    fused op's backward must be interchangeable with the corresponding
    basic ops' backward: it must consume the state saved in each basic
    op's context by the unfused forward and produce the same gradients.
    If the forward and backward need to cooperate (e.g. the forward
    saving reduced state that only a matching backward can handle), use
    ``register_forward_backward_fusion`` instead.

    The fusion function should have the following signature:

    .. code-block:: python

        func(ops, *, recipe) -> updated ops

    Parameters
    ----------
    op_fusion_func: function
        Function that takes a list of operations and may substitute
        them with fused operations.
    prepend: bool, default = ``False``
        Whether the operation fuser should apply this fusion function
        first within the backward fusion pass. The default is to apply it
        last.

    """
    if prepend:
        OperationFuser.backward_fusion_functions.insert(0, op_fusion_func)
    else:
        OperationFuser.backward_fusion_functions.append(op_fusion_func)
