# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Base classes for fusible operations."""

from __future__ import annotations
import abc
from collections.abc import Iterable
import dataclasses
import pickle
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.fp8 import (
    DelayedScaling,
    FP8GlobalStateManager,
    get_default_fp8_recipe,
)
from ._common import canonicalize_device


@dataclasses.dataclass
class OperationContext:
    """State needed to apply an operation

    Saves state from forward pass for use in backward pass.

    """

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
    requires_grad: bool = True

    def save_for_backward(self, *tensors: Optional[torch.Tensor]) -> None:
        """Register tensors to be saved for the backward function

        Expected to be called in the forward function.

        """
        self.to_save = tensors


class FusibleOperation(torch.nn.Module, metaclass=abc.ABCMeta):
    """Tensor operation supported by the operation fuser"""

    @property
    @abc.abstractmethod
    def is_fused_op(self) -> bool:
        """Whether this op is the fusion of one or more basic ops"""

    def pre_forward(self) -> None:
        """Preprocessing before forward pass"""

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        """Forward pass

        This op is either a basic op or the fusion of basic ops, so
        several of this function's arguments are lists of arguments to
        forward functions of corresponding basic ops.

        Called by `OperationFuser`.

        Parameters
        ----------
        basic_op_ctxs: list of OperationContext
            Contexts for basic operations
        input_: torch.Tensor
            Input tensor
        basic_op_extra_inputs: list of torch.Tensor
            Extra tensor inputs to basic operations
        basic_op_prev_ops: list of BasicOperation
            Basic operations that preceed this operation's basic
            operations
        basic_op_next_ops: list of BasicOperation
            Basic operations that follow this operation's basic
            operations
        basic_op_kwargs: list of dict
            Keyword arguments to forward functions of basic
            operations.

        Returns
        -------
        torch.Tensor:
            Output tensor.
        Iterable of torch.Tensor:
            Extra tensor outputs from basic operations.

        """
        raise NotImplementedError(
            f"Forward pass is not implemented for operation ({self.__class__.__name__})"
        )

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        """Backward pass

        This op is either a basic op or the fusion of basic ops, so
        several of this function's arguments are lists of arguments to
        backward functions of corresponding basic ops.

        Called by `OperationFuser`.

        Parameters
        ----------
        basic_op_ctxs: list of OperationContext
            Contexts for basic operations
        grad_output: torch.Tensor
            Loss gradient w.r.t. operation output
        basic_op_grad_extra_outputs: list of tuple of torch.Tensor
            Loss gradients w.r.t. extra tensor outputs from basic
            operations.

        Returns
        -------
        torch.Tensor:
            Loss gradient w.r.t. operation input
        Iterable of iterable of torch.Tensor:
            Loss gradients w.r.t. parameters for basic operations
        Iterable of iterable of torch.Tensor:
            Loss gradients w.r.t. extra tensor inputs to basic
            operations

        """
        raise NotImplementedError(
            f"Backward pass is not implemented for operation ({self.__class__.__name__})"
        )


class BasicOperation(FusibleOperation, metaclass=abc.ABCMeta):
    """Single tensor operation supported by the operation fuser

    This class holds parameters and state, even if the actual forward
    and backward passes are performed by a fused operation.

    """

    # Number of extra tensor inputs
    num_extra_inputs: int = 0
    # Number of extra tensor outputs
    num_extra_outputs: int = 0

    def __init__(self) -> None:
        super().__init__()

        # FP8 metadata objects
        self._fp8_metas: Optional[dict[str, dict[str, Any]]] = None

    @property
    def is_fused_op(self) -> bool:
        return False

    def num_fp8_scales(
        self,
        mode: str,  # pylint: disable=unused-argument
    ) -> int:
        """Number of FP8 scaling factors

        Parameters
        ----------
        mode: {"input", "param", "grad_output"}
            Type of FP8 scaling factor

        """
        return 0

    def _make_fp8_metas(self) -> dict[str, Optional[dict[str, Any]]]:
        """Construct FP8 metadata"""

        # Shared objects for FP8 metadata
        dtype = torch.float32
        device = canonicalize_device(None)
        recipe = get_default_fp8_recipe()

        def _make_meta(
            num_scales: int,
            is_forward: bool,
        ) -> Optional[dict[str, Any]]:
            """Construct FP8 metadata for one tensor type"""
            if num_scales == 0:
                return None
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
                "fp8_group": None,
            }

        # Construct FP8 metadata for all tensor types
        return {
            "input": _make_meta(self.num_fp8_scales("input"), True),
            "param": _make_meta(self.num_fp8_scales("param"), True),
            "grad_output": _make_meta(self.num_fp8_scales("grad_output"), False),
        }

    @classmethod
    def _maybe_update_fp8_meta(
        cls,
        fp8_meta: Optional[dict[str, Any]],
        *,
        fp8_recipe: Optional[DelayedScaling] = None,
    ) -> None:
        if fp8_meta is None:
            return

        # Update FP8 recipe
        if fp8_recipe is None:
            fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
        fp8_meta["recipe"] = fp8_recipe

        # Update FP8 communication group
        fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()

        # Adjust amax history length if needed
        amax_history_len = fp8_recipe.amax_history_len
        for is_forward in (True, False):
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=is_forward)
            if fp8_meta_key not in fp8_meta:
                continue
            meta = fp8_meta[fp8_meta_key]
            curr_len = meta.amax_history.size(0)

            # Nothing to be done if amax history is already correct
            if curr_len == amax_history_len:
                continue

            # Reallocate amax history
            with torch.no_grad():
                if curr_len > amax_history_len:
                    meta.amax_history = meta.amax_history[:amax_history_len].clone()
                else:
                    meta.amax_history = torch.nn.functional.pad(
                        meta.amax_history,
                        pad=(0, 0, 0, amax_history_len - curr_len),
                    )

            # Update global buffers for amax reductions
            buffer_info_key = FP8GlobalStateManager.get_buffer_info()
            if buffer_info_key in fp8_meta:
                fwd_pos, fwd_key, bwd_pos, bwd_key = fp8_meta[buffer_info_key]
                for pos, buffer_key in zip((fwd_pos, bwd_pos), (fwd_key, bwd_key)):
                    assert (
                        buffer_key in FP8GlobalStateManager.global_amax_history_buffer
                    ), "TE internal error during amax history change."
                    FP8GlobalStateManager.global_amax_buffer[buffer_key][pos] = fp8_meta[
                        fp8_meta_key
                    ].amax_history[0]
                    FP8GlobalStateManager.global_amax_history_buffer[buffer_key][pos] = fp8_meta[
                        fp8_meta_key
                    ].amax_history

    def get_fp8_meta(self, mode: str) -> Optional[dict[str, Any]]:
        """FP8 metadata

        Parameters
        ----------
        mode: {"input", "param", "grad_output"}
            Type of FP8 scaling factor

        """
        if self._fp8_metas is None:
            self._fp8_metas = self._make_fp8_metas()
        return self._fp8_metas[mode]

    @torch.no_grad()
    def _save_fp8_metas(self) -> Optional[dict[str, Any]]:
        """Create copies of tensors in FP8 metadata

        Tensor copies can be loaded with _load_fp8_metas.

        """
        if self._fp8_metas is None:
            return None
        out = {}
        for mode, fp8_meta in self._fp8_metas.items():
            if fp8_meta is None:
                continue
            out[mode] = {}
            for is_forward in (True, False):
                fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=is_forward)
                if fp8_meta_key not in fp8_meta:
                    continue
                out[mode][fp8_meta_key] = (
                    fp8_meta[fp8_meta_key].scale.clone(),
                    fp8_meta[fp8_meta_key].scale_inv.clone(),
                    fp8_meta[fp8_meta_key].amax_history.clone(),
                )
        return out

    @torch.no_grad()
    def _load_fp8_metas(self, fp8_metas: Optional[dict[str, Any]]) -> None:
        """Update FP8 metadata with saved tensor copies

        Tensor copies should be generated with _save_fp8_metas.

        """
        assert (self._fp8_metas is None) == (
            fp8_metas is None
        ), "Saved FP8 metadata does not match operation's FP8 metadata"
        if fp8_metas is None:
            return
        for mode, fp8_meta in fp8_metas.items():
            assert (
                mode in self._fp8_metas
            ), f"Found an unexpected key ({mode=}) in saved FP8 metadata"
            for fp8_meta_key, tensors in fp8_meta.items():
                assert (
                    fp8_meta_key in self._fp8_metas[mode]
                ), f"Found an unexpected key ({mode=}, {fp8_meta_key=}) in saved FP8 metadata"
                scale, scale_inv, amax_history = tensors
                self._fp8_metas[mode][fp8_meta_key].scale.copy_(scale)
                self._fp8_metas[mode][fp8_meta_key].scale_inv.copy_(scale_inv)
                self._fp8_metas[mode][fp8_meta_key].amax_history.copy_(amax_history)

    def pre_forward(
        self,
        *,
        fp8_enabled: Optional[bool] = None,
        fp8_recipe: Optional[DelayedScaling] = None,
    ) -> None:
        """Preprocessing before forward pass"""

        # Initialize FP8 metadata if needed
        if fp8_enabled is None:
            fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
        if fp8_enabled:

            # Construct FP8 metadata if needed
            if self._fp8_metas is None:
                self._fp8_metas = self._make_fp8_metas()

            # Make sure FP8 metadata matches FP8 autocast context
            for fp8_meta in self._fp8_metas.values():
                self._maybe_update_fp8_meta(fp8_meta, fp8_recipe=fp8_recipe)

            # Register FP8 metadata for amax and scale update
            if not FP8GlobalStateManager.fp8_graph_capturing():
                if self.num_fp8_scales("input"):
                    FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                        self.get_fp8_meta("input"),
                    )
                if self.num_fp8_scales("param"):
                    FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                        self.get_fp8_meta("param"),
                    )
                if self.num_fp8_scales("grad_output"):
                    FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                        self.get_fp8_meta("grad_output"),
                    )

    @abc.abstractmethod
    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        ctx: OperationContext
            Context to coordinate between forward and backward passes
        input_: torch.Tensor
            Input tensor
        prev_op: BasicOperation, optional
            Basic operation that preceeds this operation
        next_op: BasicOperation, optional
            Basic operation that follows this operation

        Returns
        -------
        torch.Tensor:
            Output tensor

        """

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

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[tuple[()]]]:
        if self.num_extra_inputs > 0 or self.num_extra_outputs > 0:
            raise RuntimeError(
                "{self.__class__.__name__} operation has "
                f"{self.num_extra_inputs} extra tensor inputs "
                f"and {self.num_extra_outputs} extra tensor outputs. "
                "It should override `fuser_forward` instead of `op_forward`."
            )
        output = self.op_forward(
            basic_op_ctxs[0],
            input_,
            prev_op=basic_op_prev_ops[0],
            next_op=basic_op_next_ops[0],
            **basic_op_kwargs[0],
        )
        return output, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[Iterable[Optional[torch.Tensor]]],
        list[tuple[()]],
    ]:
        if self.num_extra_inputs > 0 or self.num_extra_outputs > 0:
            raise RuntimeError(
                "{self.__class__.__name__} operation has "
                f"{self.num_extra_inputs} extra tensor inputs "
                f"and {self.num_extra_outputs} extra tensor outputs. "
                "It should override `fuser_backward` instead of `op_backward`."
            )
        grad_input, grad_params = self.op_backward(basic_op_ctxs[0], grad_output)
        return grad_input, [grad_params], [()]

    def forward(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        *extra_inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply operation"""
        from .fuser import OperationFuser

        return OperationFuser([self], fuse_ops=False)(
            input,
            *extra_inputs,
            basic_op_kwargs=[kwargs],
        )

    def get_extra_state(self) -> torch.Tensor:
        """Serialize extra state

        Contains metadata for FP8 casting.

        """

        # This implementation is working around a few issues:
        #
        # (1) PyTorch's "extra state" infrastructure might be able to
        #     support any picklable type, but they make no guarantees.
        #     It seems that ONNX export experiences issues with
        #     non-tensor extra state.
        # (2) PyTorch's checkpointing infrastructure does not remap
        #     devices for "extra state" like it does for "state dict".
        #     Thus, we want to avoid putting extra state on the GPU
        #     since it may be loaded on the wrong device.
        # (3) The extra state consists of many small tensors. If we
        #     want to copy them all to CPU, then we need to avoid the
        #     overhead of many GPU-CPU memory transfers.
        #
        # See: https://github.com/NVIDIA/TransformerEngine/pull/351
        # See: https://github.com/NVIDIA/TransformerEngine/pull/363

        # Return immediately if op has no FP8 state
        has_fp8_state = any(
            self.num_fp8_scales(mode) > 0 for mode in ("input", "param", "grad_output")
        )
        if not has_fp8_state:
            return torch.Tensor()

        def to_cpu(src: torch.Tensor) -> torch.Tensor:
            """Helper function to make CPU copy of tensor

            Memory transfer is asynchronous w.r.t. host, so GPU should
            be synchronized before using result.

            """
            dst = torch.empty_like(src, device="cpu")
            dst.copy_(src, non_blocking=True)
            return dst

        # Store FP8 state
        state = {}
        for mode in ("input", "param", "grad_output"):

            # Get state for a given FP8 tensor
            if self.num_fp8_scales(mode) == 0:
                state[mode] = None
                continue
            fp8_meta = self.get_fp8_meta(mode)
            if fp8_meta is None:
                continue
            state[mode] = {}

            # Store tensors
            if "scaling_fwd" in fp8_meta:
                state[mode]["scale_fwd"] = to_cpu(fp8_meta["scaling_fwd"].scale)
                state[mode]["scale_inv_fwd"] = to_cpu(fp8_meta["scaling_fwd"].scale_inv)
                state[mode]["amax_history_fwd"] = to_cpu(fp8_meta["scaling_fwd"].amax_history)
            if "scaling_bwd" in fp8_meta:
                state[mode]["scale_bwd"] = to_cpu(fp8_meta["scaling_bwd"].scale)
                state[mode]["scale_inv_bwd"] = to_cpu(fp8_meta["scaling_bwd"].scale_inv)
                state[mode]["amax_history_bwd"] = to_cpu(fp8_meta["scaling_bwd"].amax_history)

            # Store other picklable items
            extra = {}
            for key, val in fp8_meta.items():
                if key == "buffer_index_and_autocast_key":
                    continue
                if not isinstance(val, (bool, int, float, str, tuple, list)):
                    continue
                extra[key] = val
            state[mode]["extra_fp8_variables"] = extra

        # Serialize state into byte tensor
        torch.cuda.synchronize()
        state_serialized = bytearray(pickle.dumps(state))
        state_serialized = torch.frombuffer(state_serialized, dtype=torch.uint8)
        return state_serialized

    def set_extra_state(self, state: Optional[torch.Tensor]) -> None:
        """Load extra state"""
        if state is None or state.numel() == 0:
            return

        # Deserialize state from byte tensor
        state = pickle.loads(state.detach().numpy(force=True).tobytes())
        if state is None:
            return

        def copy_tensor(src: torch.Tensor, dst: torch.Tensor) -> None:
            """Helper function to copy tensor from CPU

            Memory transfer is asynchronous w.r.t. host, so GPU should
            be synchronized before using result.

            """
            if src.size() != dst.size():
                dst.data = torch.empty(src.size(), dtype=dst.dtype, device=dst.device)
            dst.copy_(src, non_blocking=True)

        # Load FP8 state
        for mode in ("input", "param", "grad_output"):

            # Get state for a given FP8 tensor
            if mode not in state:
                continue
            if self.num_fp8_scales(mode) == 0:
                continue
            fp8_meta = self.get_fp8_meta(mode)
            if fp8_meta is None:
                continue

            # Load extra state
            fp8_meta.update(state[mode]["extra_fp8_variables"])
            if "amax_history_fwd" in state[mode]:
                fp8_meta["recipe"].amax_history_len = state[mode]["amax_history_fwd"].size(0)
            elif "amax_history_bwd" in state[mode]:
                fp8_meta["recipe"].amax_history_len = state[mode]["amax_history_bwd"].size(0)
            if "global_fp8_buffer_pos_fwd_recompute" in fp8_meta:
                del fp8_meta["global_fp8_buffer_pos_fwd_recompute"]

            # Load tensors
            fp8_meta = self.get_fp8_meta(mode)
            if "scaling_fwd" in fp8_meta:
                fp8_meta_fwd = fp8_meta["scaling_fwd"]
                copy_tensor(state[mode]["scale_fwd"], fp8_meta_fwd.scale)
                copy_tensor(state[mode]["scale_inv_fwd"], fp8_meta_fwd.scale_inv)
                copy_tensor(state[mode]["amax_history_fwd"], fp8_meta_fwd.amax_history)
            if "scaling_bwd" in fp8_meta:
                fp8_meta_bwd = fp8_meta["scaling_bwd"]
                copy_tensor(state[mode]["scale_bwd"], fp8_meta_bwd.scale)
                copy_tensor(state[mode]["scale_inv_bwd"], fp8_meta_bwd.scale_inv)
                copy_tensor(state[mode]["amax_history_bwd"], fp8_meta_bwd.amax_history)

        # Finish CPU-GPU memory transfers
        torch.cuda.synchronize()

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        """Load state"""

        # In the base PyTorch module class, the extra state is loaded
        # _after_ the parameters. However, copying values into FP8
        # parameters requires an FP8 cast, which uses a scaling factor
        # from the operation's FP8 metadata. The FP8 metadata is
        # included in the operation's extra state, so we need to
        # manually load the extra state before loading parameters.

        state_dict, prefix = args[0], args[1]
        extra_state_key = prefix + torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX
        if extra_state_key in state_dict:
            self.set_extra_state(state_dict[extra_state_key])
        super()._load_from_state_dict(*args, **kwargs)


class FusedOperation(FusibleOperation):
    """Compound tensor operation supported by the operation fuser

    If the forward or backward passes are defined, they must be
    functionally equivalent to the forward/backward passes of the
    corresponding basic ops. This class should hold no parameters or
    other state, but should access them from the basic ops.

    Parameters
    ----------
    basic_ops: iterable of FusibleOperation
        Basic ops that are interchangeable with this op

    """

    def __init__(
        self,
        basic_ops: Iterable[FusibleOperation],
    ) -> None:
        super().__init__()

        # Basic operations that comprise this fused operation
        self.basic_ops: torch.nn.ModuleList = torch.nn.ModuleList(basic_ops)
        if len(self.basic_ops) == 0:
            raise ValueError(
                "Attempted to construct a fused operation "
                "without specifying its corresponding basic operations"
            )

    @property
    def is_fused_op(self) -> bool:
        return True

    def pre_forward(self) -> None:
        """Preprocessing before forward pass"""
        for op in self.basic_ops:
            op.pre_forward()

    def forward(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        *extra_inputs: torch.Tensor,
        basic_op_kwargs: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Apply operation"""
        if basic_op_kwargs is None:
            basic_op_kwargs = [{} for _ in range(len(self.basic_ops))]
        from .fuser import OperationFuser

        return OperationFuser([self], fuse_ops=False)(
            input,
            *extra_inputs,
            basic_op_kwargs=basic_op_kwargs,
        )
