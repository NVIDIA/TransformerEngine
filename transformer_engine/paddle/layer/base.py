# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Base modules and utilities for TransformerEngine Paddle API"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
import os
import pickle
from typing import Generator, Dict, Tuple, Union, Any, List, Optional

import numpy as np

import paddle

try:
    from paddle.base import core
    from paddle.base.framework import _dygraph_tracer
except ImportError:
    from paddle.fluid import core
    from paddle.fluid.framework import _dygraph_tracer

from ..constants import FP8BwdTensors, dist_group_type
from ..cpp_extensions import cast_transpose, cast_transpose_bgrad, cast_to_fp8, transpose
from ..fp8 import (
    FP8State,
    FP8TensorMeta,
    amax_and_scale_update,
    get_global_fp8_state,
    get_fp8_te_dtype,
)
from ..distributed import allgather
from ..profile import nvtx_range
from ..recompute import is_in_recompute_phase
from ..fp8_buffer import FP8RecomputeBuffer

_2X_ACC_FPROP = False
_2X_ACC_DGRAD = True
_2X_ACC_WGRAD = True
_cublas_workspace = None


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if paddle.device.cuda.get_device_capability()[0] >= 9:
        return 33_554_432
    return 4_194_304


def get_workspace() -> paddle.Tensor:
    """Returns workspace for cublas."""
    global _cublas_workspace
    if _cublas_workspace is None:
        _cublas_workspace = paddle.empty(
            [get_cublas_workspace_size_bytes()],
            dtype="uint8",
        )
    return _cublas_workspace


class TransformerEngineBaseLayer(paddle.nn.Layer, ABC):
    """Base TE Layer."""

    def __init__(self) -> None:
        super().__init__()
        assert "gpu" in paddle.device.get_device(), "TransformerEngine needs CUDA."
        self.fp8_initialized = False
        self.fp8_enabled = False
        self.fp8_calibration = False
        self.fp8_meta = {}
        self.fp8_meta["fp8_checkpoint"] = False
        self.fp8_meta["fp8_group"] = None
        self.fp8_meta["recipe"] = FP8State.get_default_fp8_recipe()
        self.fp8_meta["scaling_fwd"] = FP8TensorMeta(is_forward=True)
        self.fp8_meta["scaling_bwd"] = FP8TensorMeta(is_forward=False)
        self.tp_group = None
        self.tp_size = 1
        self.sequence_parallel = False
        self.fp8_meta["autocast_id_fwd_stack"] = []
        self.fp8_meta["async_amax_reduction"] = bool(
            int(os.getenv("NVTE_ASYNC_AMAX_REDUCTION", "0"))
        )
        self.fp8_weight_shapes = []
        self.fp8_weight_cache = {}

    def set_activation_dtype(self, inp: paddle.Tensor) -> None:
        """Get activation data type for AMP."""
        tracer = _dygraph_tracer()
        if tracer and tracer._amp_level != core.AmpLevel.O0:
            # Set activation_dtype to the Paddle AMP dtype if under 'paddle.amp.auto_cast' context
            if tracer._amp_dtype == "float32":
                self.activation_dtype = paddle.float32
            elif tracer._amp_dtype == "bfloat16":
                self.activation_dtype = paddle.bfloat16
            elif tracer._amp_dtype == "float16":
                self.activation_dtype = paddle.float16
            else:
                raise RuntimeError(f"AMP format {tracer._amp_dtype} is not supported.")
        else:
            # If not under paddle.amp.auto_cast, set activation_dtype to the input dtype.
            # Also, make sure the parameters match the input dtype.

            # Skip the check if activation_dtype is already set and if activation_dtype
            # matches input dtype. If they do not match, e.g, when user switch from AMP
            # training to normal training, activation_dtype will still be updated.
            if hasattr(self, "activation_dtype") and self.activation_dtype == inp.dtype:
                return

            dtype = inp.dtype

            for name, param in self.named_parameters():
                if param is not None:
                    assert dtype == param.dtype, (
                        "Data types for parameters must match when outside of autocasted region. "
                        f" Found input dtype: {dtype} and {name!r} dtype: {param.dtype}"
                    )

            self.activation_dtype = dtype

    # This routine is shared across FP8 and FP8_calibration paths so should not actually
    # assume FP8 execution.
    def fp8_init(self, num_gemms: int = 1) -> None:
        """Initialize fp8 related metadata and tensors during fprop."""
        global_fp8_state = get_global_fp8_state()
        self.fp8_enabled = global_fp8_state.is_fp8_enabled()
        self.fp8_calibration = global_fp8_state.is_fp8_calibration()
        self.fp8_meta["fp8_checkpoint"] = self.fp8_enabled or self.fp8_calibration

        if self.fp8_enabled or self.fp8_calibration:
            # FP8 init has already been run and recipe is the same, don't do anything.
            if (
                self.fp8_initialized
                and global_fp8_state.get_fp8_recipe() == self.fp8_meta["recipe"]
            ):
                return

            # Set FP8, recipe, and other FP8 metadata
            self.fp8_meta["recipe"] = global_fp8_state.get_fp8_recipe()
            self.fp8_meta["fp8_group"] = global_fp8_state.get_fp8_group()

            # Set FP8_MAX per tensor according to recipe
            self.fp8_meta["fp8_max_fwd"] = self.fp8_meta["recipe"].fp8_format.value.max_fwd
            self.fp8_meta["fp8_max_bwd"] = self.fp8_meta["recipe"].fp8_format.value.max_bwd

            # Allocate scales and amaxes
            amax_history_len = self.fp8_meta["recipe"].amax_history_len
            self.fp8_meta["scaling_fwd"].prepare(num_gemms, amax_history_len)
            self.fp8_meta["scaling_bwd"].prepare(num_gemms, amax_history_len)
            self.fp8_initialized = True
        else:
            # If fp8 isn't enabled, turn off and return.
            self.fp8_initialized = False
            return

    def set_fp8_weights(self) -> None:
        """Initializes FP8 weights for the module"""
        if not self.fp8_enabled:
            return

        for i, shape in enumerate(self.fp8_weight_shapes, start=1):
            weight_cast_key = f"weight{i}_fp8"
            weight_transpose_key = f"weight{i}_t_fp8"

            if (
                weight_cast_key in self.fp8_weight_cache
                and self.fp8_weight_cache[weight_cast_key].shape == shape
            ):
                return

            self.fp8_weight_cache[weight_cast_key] = paddle.empty(
                shape=shape,
                dtype=paddle.uint8,
            )

            self.fp8_weight_cache[weight_transpose_key] = paddle.empty(
                shape=[shape[1], shape[0]],
                dtype=paddle.uint8,
            )

    def _get_fp8_state(self) -> paddle.Tensor:
        """Dump FP8 state to paddle.Tensor."""
        state = None
        if self.fp8_meta["fp8_checkpoint"]:
            state = {}
            state["scaling_fwd"] = self.fp8_meta["scaling_fwd"].to_numpy()
            state["scaling_bwd"] = self.fp8_meta["scaling_bwd"].to_numpy()
            state["global_fp8_fwd_buffer"] = get_global_fp8_state().get_fp8_fwd_buffer().to_numpy()
            state["global_fp8_bwd_buffer"] = get_global_fp8_state().get_fp8_bwd_buffer().to_numpy()
            # Store other pickelable values.
            extra = {}
            for k, v in self.fp8_meta.items():
                if isinstance(v, (bool, int, float, str)):
                    extra[k] = v
            state["extra_fp8_variables"] = extra

        state_serialized = pickle.dumps(state)
        state_tensor = paddle.to_tensor(np.frombuffer(state_serialized, dtype=np.uint8))

        return state_tensor

    @paddle.no_grad()
    def state_dict(
        self,
        destination=None,
        include_sublayers=True,
        structured_name_prefix="",
        use_hook=True,
    ):
        """Save FP8 State when checkpointing."""
        st = super().state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
            use_hook=use_hook,
        )
        st["fp8_state"] = self._get_fp8_state()
        return st

    def _set_fp8_state(self, state: paddle.Tensor) -> None:
        """Load previous state."""
        if state is None:
            return

        state = pickle.loads(state.numpy().tobytes())
        if state is None:
            return

        # Load fp8 meta tensors.
        self.fp8_meta["scaling_fwd"].from_numpy(state["scaling_fwd"])
        self.fp8_meta["scaling_bwd"].from_numpy(state["scaling_bwd"])

        # Restore global FP8 buffer states.
        global_fp8_fwd_buffer = get_global_fp8_state().get_fp8_fwd_buffer()
        global_fp8_bwd_buffer = get_global_fp8_state().get_fp8_bwd_buffer()
        global_fp8_fwd_buffer.from_numpy(state["global_fp8_fwd_buffer"])
        global_fp8_bwd_buffer.from_numpy(state["global_fp8_bwd_buffer"])

        # Load extra items.
        self.fp8_meta.update(state["extra_fp8_variables"])
        self.fp8_meta["recipe"].amax_history_len = self.fp8_meta["scaling_fwd"].amax_history.shape[
            0
        ]
        recompute_buffer_pos_key = FP8RecomputeBuffer.get_buffer_position_key()
        if recompute_buffer_pos_key in self.fp8_meta:
            del self.fp8_meta[recompute_buffer_pos_key]

    @paddle.no_grad()
    def set_state_dict(self, state_dict, use_structured_name=True):
        """Restore FP8 State from checkpoint."""
        fp8_state_tensor = state_dict.pop("fp8_state")
        self._set_fp8_state(fp8_state_tensor)

        return super().set_state_dict(state_dict)

    @contextmanager
    def prepare_forward(
        self,
        inp: paddle.Tensor,
        is_first_microbatch: Union[bool, None],
        num_gemms: int = 1,
    ) -> Generator[paddle.Tensor, None, None]:
        """Checks and prep for FWD.
        The context manager is needed because there isn't a way for a module to know
        if it's the last FP8 module in the forward autocast. It is useful
        to setup the forward aggregated amax reduction for every module
        just in case. The autocast exit will pick up the most recent one.
        """

        if self.fp8_enabled and is_in_recompute_phase():
            global_recompute_buffer = get_global_fp8_state().get_fp8_recompute_buffer()
            global_recompute_buffer.retrieve_fp8_meta_tensors(self.fp8_meta)
        else:
            self.set_activation_dtype(inp)
            self.fp8_init(num_gemms=num_gemms)

            # Create persistent tensors for fp8 weights and their transposes
            # only when fp8 weight caching is used.
            if is_first_microbatch is not None:
                self.set_fp8_weights()

            if self.fp8_enabled and self.sequence_parallel:
                assert self.fp8_meta["recipe"].reduce_amax, (
                    "Amax reduction across tensor parallel group is "
                    "necessary when using sequence parallelism with FP8."
                )

            update_weight_scale_inv = is_first_microbatch is None or is_first_microbatch

            # Previous iteration was grad_enabled
            if self.fp8_meta.get("update_amax_and_scale_fwd", False):
                global_fp8_fwd_buffer = get_global_fp8_state().get_fp8_fwd_buffer()
                global_fp8_fwd_buffer.wait()
                if self.fp8_meta["recipe"].reduce_amax:
                    global_fp8_fwd_buffer.copy_amax_from_buffer(self.fp8_meta)
                    amax_and_scale_update(
                        self.fp8_meta, True, update_weight_scale_inv=update_weight_scale_inv
                    )
                    global_fp8_fwd_buffer.set_for_deletion(self.fp8_meta)
                else:
                    amax_and_scale_update(
                        self.fp8_meta, True, update_weight_scale_inv=update_weight_scale_inv
                    )

            if self.fp8_enabled and self.training:
                # Setup for amax reduction
                if self.fp8_meta["recipe"].reduce_amax:
                    global_fp8_state = get_global_fp8_state()
                    self.fp8_meta["first_module"] = global_fp8_state.is_first_fp8_module()
                    self.fp8_meta["autocast_id_fwd"] = global_fp8_state.get_autocast_id()
                    self.fp8_meta["autocast_id_fwd_stack"].append(self.fp8_meta["autocast_id_fwd"])
                self.fp8_meta["update_amax_and_scale_fwd"] = True
            else:
                self.fp8_meta["update_amax_and_scale_fwd"] = False

            # Activation recomputation is used and this is the first forward phase.
            if (
                self.fp8_enabled
                and self.training
                and get_global_fp8_state().is_fp8_recompute_enabled()
            ):
                global_recompute_buffer = get_global_fp8_state().get_fp8_recompute_buffer()
                global_recompute_buffer.stash_fp8_meta_tensors(self.fp8_meta)

        with nvtx_range(self.__class__.__name__ + " forward"):
            yield inp

        if self.fp8_enabled and is_in_recompute_phase():
            FP8RecomputeBuffer.restore_fp8_meta_tensors(self.fp8_meta)
            return

        if self.fp8_enabled and self.training and self.fp8_meta["recipe"].reduce_amax:
            global_fp8_state = get_global_fp8_state()
            global_fp8_fwd_buffer = global_fp8_state.get_fp8_fwd_buffer()
            global_fp8_fwd_buffer.add_amax(self.fp8_meta)
            global_fp8_fwd_buffer.set_for_amax_reduction(
                self.fp8_meta,
                self.tp_group,
                self.tp_size,
            )

    @staticmethod
    @contextmanager
    def prepare_backward(
        fp8_enabled: bool,
        fp8_meta: Dict[str, Any],
        tp_group: dist_group_type,
        tp_size: int,
        name: str = "",
    ) -> Generator[None, None, None]:
        """Checks and prep for BWD."""
        if fp8_enabled:
            global_fp8_state = get_global_fp8_state()
            global_fp8_bwd_buffer = global_fp8_state.get_fp8_bwd_buffer()
            global_fp8_bwd_buffer.wait()

            if fp8_meta["recipe"].reduce_amax:
                global_fp8_bwd_buffer.copy_amax_from_buffer(fp8_meta)
                amax_and_scale_update(fp8_meta, False)
                global_fp8_bwd_buffer.set_for_deletion(fp8_meta)

                # Get new backward key.
                fp8_meta["autocast_id_bwd"] = fp8_meta["autocast_id_fwd_stack"].pop(0)
            else:
                amax_and_scale_update(fp8_meta, False)

        with nvtx_range(name + " backward"):
            yield

        if fp8_enabled and fp8_meta["recipe"].reduce_amax:
            global_fp8_bwd_buffer.add_amax(fp8_meta)
            if fp8_meta["first_module"]:
                global_fp8_bwd_buffer.finalize(fp8_meta, tp_group, tp_size)

    @staticmethod
    def grad_output_preprocess(
        ctx, grad_output: paddle.Tensor, row_parallel_mode: bool
    ) -> Tuple[Union[paddle.Tensor, None], ...]:
        """Utility function for backward.
        Returns tuple in order (all optional/None based on training precion/recipe):
            R1: gathered `grad_output` in higher precision.
            R2: gathered `grad_output` in FP8.
            R3: R2 transposed.
            R4: bias gradient on R1.
        """
        grad_output_mat = grad_output.reshape((-1, grad_output.shape[-1]))
        gather_grad_output = row_parallel_mode and ctx.sequence_parallel

        # No-FP8 case: bgrad is fused with wgrad for this case.
        if not ctx.fp8_enabled:
            if gather_grad_output:
                grad_output_mat, _ = allgather(grad_output_mat, ctx.tp_group)
            return grad_output_mat, None, None, None

        fp8_dtype_backward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)

        if gather_grad_output:
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                # FP8 case with gather: unfused bgrad, cast, transpose for efficient gather
                if ctx.use_bias:
                    bgrad = grad_output_mat.sum(axis=0)
                else:
                    bgrad = None
                grad_output_c = cast_to_fp8(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                )
                grad_output_c, _ = allgather(grad_output_c, ctx.tp_group)
                grad_output_t = transpose(grad_output_c, fp8_dtype_backward)

                return grad_output_mat, grad_output_c, grad_output_t, bgrad

            # FP8 case with gather and non-FP8 wgrad
            grad_output_mat, _ = allgather(grad_output_mat, ctx.tp_group)

        # FP8 case without gather: cast, transpose, bgrad fused
        if ctx.use_bias:
            bgrad, grad_output_c, grad_output_t = cast_transpose_bgrad(
                grad_output_mat,
                ctx.fp8_meta["scaling_bwd"],
                FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
            )
        else:
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                grad_output_c, grad_output_t = cast_transpose(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                )
            else:
                grad_output_t = None
                grad_output_c = cast_to_fp8(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                )
            bgrad = None

        return grad_output_mat, grad_output_c, grad_output_t, bgrad

    @abstractmethod
    def forward(self):
        """Needs override."""

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[Optional[paddle.Tensor]]:
        """
        Fetch the fp8 weight tensor placeholders if they exist (when
        `is_first_microbatch` is not `None`)
        """
        if not self.fp8_enabled or is_first_microbatch is None:
            return [None, None] * len(self.fp8_weight_shapes)

        out_list = []
        for i, _ in enumerate(self.fp8_weight_shapes, start=1):
            weight_cast_key = f"weight{i}_fp8"
            weight_transpose_key = f"weight{i}_t_fp8"

            assert (
                weight_cast_key in self.fp8_weight_cache
            ), "TE internal error: fp8 weight buffer is not found"

            out_list.extend(
                [
                    self.fp8_weight_cache[weight_cast_key],
                    self.fp8_weight_cache[weight_transpose_key],
                ]
            )
        return out_list
