# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from transformer_engine.common.recipe import Recipe
from ..fp8 import (
    MXFP8BlockScalingRecipeState,
    DelayedScalingRecipeState,
    FP8GlobalStateManager,
    RecipeState,
)
from ..tensor import Quantizer


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

        # Objects for quantization
        self._quantizers: Optional[dict[str, list[Quantizer]]] = None
        self._fp8_metas: Optional[dict[str, dict[str, Any]]] = None

    @property
    def is_fused_op(self) -> bool:
        return False

    def num_quantizers(
        self,
        mode: str,  # pylint: disable=unused-argument
    ) -> int:
        """Number of quantizers

        Matches number of quantized tensors used in operation.

        Parameters
        ----------
        mode: {"forward", "backward"}
            Quantizer type

        """
        return 0

    def _reset_quantization_recipe_state(
        self,
        *,
        recipe: Optional[Recipe] = None,
    ) -> None:
        """Construct state for quantization recipe"""

        # Quantization recipe
        if recipe is None:
            recipe = FP8GlobalStateManager.get_fp8_recipe()

        # Quantization recipe state for forward and backward pass
        self._fp8_metas = {"forward": None, "backward": None}
        self._quantizers = {"forward": [], "backward": []}
        for mode in ("forward", "backward"):
            num_quantizers = self.num_quantizers(mode)
            if num_quantizers == 0:
                continue

            # Construct quantization recipe state
            recipe_state = RecipeState.create(
                recipe,
                mode=mode,
                num_quantizers=num_quantizers,
            )
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                forward=(mode == "forward"),
            )
            self._fp8_metas[mode] = {
                fp8_meta_key: recipe_state,
                "recipe": recipe,
                "fp8_group": FP8GlobalStateManager.get_fp8_group(),
            }

            # Construct builder class for quantized tensors
            self._quantizers[mode] = recipe_state.make_quantizers()

    def _update_quantization_recipe_state(
        self,
        *,
        recipe: Optional[Recipe] = None,
    ) -> None:
        """Make sure quantizer state matches quantization recipe"""

        # Quantization recipe
        if recipe is None:
            recipe = FP8GlobalStateManager.get_fp8_recipe()

        # Reset quantization state if needed
        if self._fp8_metas is None or self._quantizers is None:
            self._reset_quantization_recipe_state(recipe=recipe)
            return
        for mode in ("forward", "backward"):
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                forward=(mode == "forward"),
            )
            if self._fp8_metas[mode] is None or fp8_meta_key not in self._fp8_metas[mode]:
                continue
            recipe_state = self._fp8_metas[mode][fp8_meta_key]
            need_to_reset_recipe_state = (
                recipe.delayed() and not isinstance(recipe_state, DelayedScalingRecipeState)
            ) or (recipe.mxfp8() and not isinstance(recipe_state, MXFP8BlockScalingRecipeState))
            if need_to_reset_recipe_state:
                self._reset_quantization_recipe_state(recipe=recipe)
                return

        # Quantization recipe state for forward and backward pass
        for mode in ("forward", "backward"):
            num_quantizers = self.num_quantizers(mode)
            if num_quantizers == 0:
                continue

            # Update FP8 metadata
            fp8_meta = self._fp8_metas[mode]
            fp8_meta["recipe"] = recipe
            fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()

            # Get recipe state
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                forward=(mode == "forward"),
            )
            recipe_state = fp8_meta[fp8_meta_key]

            # Reallocate amax history if needed
            if recipe.mxfp8():
                continue

            current_length = recipe_state.amax_history.size(0)
            target_length = recipe.amax_history_len
            if current_length != target_length:
                with torch.no_grad():
                    if target_length < current_length:
                        recipe_state.amax_history = recipe_state.amax_history[
                            :target_length
                        ].clone()
                    else:
                        recipe_state.amax_history = torch.nn.functional.pad(
                            recipe_state.amax_history,
                            pad=(0, 0, 0, target_length - current_length),
                        )
                self._quantizers[mode] = recipe_state.make_quantizers()

    def get_quantizer(
        self,
        mode: str,
        index: int,
    ) -> Quantizer:
        """Get builder class for quantized tensor

        Parameters
        ----------
        mode: {"forward", "backward"}
            Quantizer type

        """
        if self._quantizers is None:
            self._reset_quantization_recipe_state()
        return self._quantizers[mode][index]

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
                scale, amax_history = tensors
                self._fp8_metas[mode][fp8_meta_key].scale.copy_(scale)
                self._fp8_metas[mode][fp8_meta_key].amax_history.copy_(amax_history)

    def pre_forward(
        self,
        *,
        fp8_enabled: Optional[bool] = None,
        fp8_recipe: Optional[Recipe] = None,
    ) -> None:
        """Preprocessing before forward pass"""

        # Initialize FP8 metadata if needed
        if fp8_enabled is None:
            fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
        if fp8_enabled:
            self._update_quantization_recipe_state(recipe=fp8_recipe)
            if not FP8GlobalStateManager.fp8_graph_capturing():
                if self.num_quantizers("forward"):
                    FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                        self._fp8_metas["forward"],
                    )
                if self.num_quantizers("backward"):
                    FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                        self._fp8_metas["backward"],
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
        #     We have experienced problems (e.g. in ONNX export) with
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
        for mode in ("forward", "backward"):

            # Get state for a given FP8 tensor
            if self.num_quantizers(mode) == 0:
                continue
            fp8_meta = self.get_fp8_meta(mode)
            state[mode] = {}

            # Store tensors
            if "scaling_fwd" in fp8_meta:
                state[mode]["scale_fwd"] = to_cpu(fp8_meta["scaling_fwd"].scale)
                state[mode]["amax_history_fwd"] = to_cpu(fp8_meta["scaling_fwd"].amax_history)
            if "scaling_bwd" in fp8_meta:
                state[mode]["scale_bwd"] = to_cpu(fp8_meta["scaling_bwd"].scale)
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
        if state is None or len(state) == 0:
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
        for mode in ("forward", "backward"):

            # Get state for a given FP8 tensor
            if mode not in state:
                continue
            if self.num_quantizers(mode) == 0:
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
                copy_tensor(state[mode]["amax_history_fwd"], fp8_meta_fwd.amax_history)
            if "scaling_bwd" in fp8_meta:
                fp8_meta_bwd = fp8_meta["scaling_bwd"]
                copy_tensor(state[mode]["scale_bwd"], fp8_meta_bwd.scale)
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
