# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Base modules and utilities for TransformerEngine Paddle API"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
import pickle
from typing import Generator, Dict, Tuple, Union, Any

import numpy as np

import paddle
from paddle.fluid import core
from paddle.fluid.framework import _dygraph_tracer

import transformer_engine_paddle as tex

from ..cpp_extensions import cast_transpose, cast_transpose_bgrad, cast_to_fp8
from ..fp8 import (
    get_fp8_recipe,
    get_default_fp8_recipe,
    is_fp8_enabled,
    is_fp8_calibration,
    amax_and_scale_update,
    get_fp8_te_dtype,
)
from ..profile import nvtx_range
from ..utils import get_bias_dtype, cast_if_needed

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
            dtype='uint8',
        )
    return _cublas_workspace


@contextmanager
def _prepare_backward(fp8: bool,
                      fp8_meta: Dict[str, Any],
                      name: str = "") -> Generator[None, None, None]:
    """Checks and prep for BWD."""
    if fp8:
        amax_and_scale_update(fp8_meta, False)

    with nvtx_range(name + " backward"):
        yield


class TransformerEngineBaseLayer(paddle.nn.Layer, ABC):
    """Base TE Layer."""

    def __init__(self) -> None:
        super().__init__()
        assert 'gpu' in paddle.device.get_device(), "TransformerEngine needs CUDA."
        self.fp8_initialized = False
        self.fp8 = False
        self.fp8_calibration = False
        self.fp8_meta = {}
        self.fp8_meta["fp8_group"] = None
        self.fp8_meta["recipe"] = get_default_fp8_recipe()
        self.fp8_meta_tensors_initialized = False

    def set_activation_dtype(self, inp: paddle.Tensor) -> None:
        """Get activation data type for AMP."""
        # Native AMP (`paddle.amp.auto_cast`) gets highest priority
        tracer = _dygraph_tracer()
        if tracer and tracer._amp_level != core.AmpLevel.O0:
            if tracer._amp_dtype == 'float32':
                self.activation_dtype = paddle.float32
            elif tracer._amp_dtype == 'bfloat16':
                self.activation_dtype = paddle.bfloat16
            elif tracer._amp_dtype == 'float16':
                self.activation_dtype = paddle.float16
            else:
                raise RuntimeError(f"AMP format {tracer._amp_dtype} is not supported.")
            return

        # All checks after this have already been performed once, thus skip
        # We assume that user doesn't change input types across iterations
        if hasattr(self, "activation_dtype") and self.activation_dtype == inp.dtype:
            return

        dtype = inp.dtype

        for name, param in self.named_parameters():
            if param is not None:
                assert dtype == param.dtype, (
                    "Data types for parameters must match when outside of autocasted region. "
                    f" Found input dtype: {dtype} and {name!r} dtype: {param.dtype}")

        self.activation_dtype = dtype

    def set_meta_tensor(self, fwd: bool) -> None:
        """Init scales and amaxes for fwd | bwd."""
        fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"

        if self.fp8_meta_tensors_initialized:
            # Handle changed amax history size.
            curr_len = self.fp8_meta[fp8_meta_tensor_key].amax_history.shape[0]
            num_fp8_tensors = self.fp8_meta[fp8_meta_tensor_key].amax_history.shape[1]
            need_len = self.fp8_meta["recipe"].amax_history_len
            if need_len < curr_len:
                self.fp8_meta[fp8_meta_tensor_key].amax_history = (
                    self.fp8_meta[fp8_meta_tensor_key].amax_history[:self.fp8_meta["recipe"].
                                                                    amax_history_len])
            elif need_len > curr_len:
                extra_rows = need_len - curr_len
                self.fp8_meta[fp8_meta_tensor_key].amax_history = paddle.concat([
                    self.fp8_meta[fp8_meta_tensor_key].amax_history,
                    paddle.zeros((extra_rows, num_fp8_tensors), dtype='float32')
                ],
                                                                                axis=0)
            return

        # Max. number of fp8 tensors per GEMM = 3 (input, weight, output) for fwd and
        # 2 (grad_output and grad_input) for bwd
        num_fp8_tensors = (self.fp8_meta["num_gemms"] * 3 if fwd else self.fp8_meta["num_gemms"] *
                           2)

        self.fp8_meta[fp8_meta_tensor_key] = tex.FP8TensorMeta()
        self.fp8_meta[fp8_meta_tensor_key].scale = paddle.ones(num_fp8_tensors, dtype='float32')
        self.fp8_meta[fp8_meta_tensor_key].scale_inv = paddle.ones(num_fp8_tensors, dtype='float32')
        self.fp8_meta[fp8_meta_tensor_key].amax_history = paddle.zeros(
            (self.fp8_meta["recipe"].amax_history_len, num_fp8_tensors), dtype='float32')

        # Needed for calculation of scale inverses to
        # preserve scale_inv when caching FP8 weights
        if fwd:
            # [True, False, True]: -> [input, weight, output]
            self.fp8_meta[fp8_meta_tensor_key + "_non_weight_mask"] = \
                paddle.to_tensor([True, False, True] * self.fp8_meta["num_gemms"],
                                stop_gradient=True, dtype='bool')
        else:
            # [True, True]: -> [grad_output, grad_input]
            self.fp8_meta[fp8_meta_tensor_key + "_non_weight_mask"] = \
                paddle.to_tensor([True, True] * self.fp8_meta["num_gemms"],
                                stop_gradient=True, dtype='bool')

    def init_fp8_meta_tensors(self) -> None:
        """Init scales and amaxes."""
        self.set_meta_tensor(True)
        self.set_meta_tensor(False)
        self.fp8_meta_tensors_initialized = True

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

    @paddle.no_grad()
    def set_state_dict(self, state_dict, use_structured_name=True):
        """Restore FP8 State from checkpoint."""
        fp8_state_tensor = state_dict.pop("fp8_state")
        self._set_fp8_state(fp8_state_tensor)

        return super().set_state_dict(state_dict)

    def _get_fp8_state(self) -> paddle.Tensor:
        """Dump FP8 state to paddle.Tensor."""
        state = None
        if self.fp8 or self.fp8_calibration:
            state = {}
            state["scale_fwd"] = self.fp8_meta["scaling_fwd"].scale.numpy()
            state["scale_inv_fwd"] = self.fp8_meta["scaling_fwd"].scale_inv.numpy()
            state["amax_history_fwd"] = self.fp8_meta["scaling_fwd"].amax_history.numpy()
            state["scale_bwd"] = self.fp8_meta["scaling_bwd"].scale.numpy()
            state["scale_inv_bwd"] = self.fp8_meta["scaling_bwd"].scale_inv.numpy()
            state["amax_history_bwd"] = self.fp8_meta["scaling_bwd"].amax_history.numpy()

            # Store other pickelable values.
            extra = {}
            for k, v in self.fp8_meta.items():
                if isinstance(v, (bool, int, float, str)):
                    extra[k] = v
            state["extra_fp8_variables"] = extra

        state_serialized = pickle.dumps(state)
        state_tensor = paddle.to_tensor(np.frombuffer(state_serialized, dtype=np.uint8))

        return state_tensor

    def _set_fp8_state(self, state: paddle.Tensor) -> None:
        """Load previous state."""
        if state is None:
            return

        state = pickle.loads(state.numpy().tobytes())
        if state is None:
            return

        # Load extra items.
        self.fp8_meta.update(state["extra_fp8_variables"])
        self.fp8_meta["recipe"].amax_history_len = state["amax_history_fwd"].shape[0]

        # Initialize before loading.
        self.init_fp8_meta_tensors()
        self.fp8_meta["scaling_fwd"].scale.copy_(paddle.to_tensor(state["scale_fwd"]), True)
        self.fp8_meta["scaling_fwd"].amax_history.copy_(paddle.to_tensor(state["amax_history_fwd"]),
                                                        True)
        self.fp8_meta["scaling_fwd"].scale_inv.copy_(paddle.to_tensor(state["scale_inv_fwd"]), True)
        self.fp8_meta["scaling_bwd"].scale.copy_(paddle.to_tensor(state["scale_bwd"]), True)
        self.fp8_meta["scaling_bwd"].amax_history.copy_(paddle.to_tensor(state["amax_history_bwd"]),
                                                        True)
        self.fp8_meta["scaling_bwd"].scale_inv.copy_(paddle.to_tensor(state["scale_inv_bwd"]), True)

    # This routine is shared across FP8 and FP8_calibration paths so should not actually
    # assume FP8 execution.
    def fp8_init(self, num_gemms: int = 1) -> None:
        """Initialize fp8 related metadata and tensors during fprop."""
        self.fp8 = is_fp8_enabled()
        self.fp8_calibration = is_fp8_calibration()

        if self.fp8 or self.fp8_calibration:
            # FP8 init has already been run and recipe is the same, don't do anything.
            if self.fp8_initialized and get_fp8_recipe() == self.fp8_meta["recipe"]:
                return

            # Set FP8, recipe, and other FP8 metadata
            self.fp8_meta["recipe"] = get_fp8_recipe()
            self.fp8_meta["num_gemms"] = num_gemms

            # Set FP8_MAX per tensor according to recipe
            self.fp8_meta["fp8_max_fwd"] = self.fp8_meta["recipe"].fp8_format.value.max_fwd
            self.fp8_meta["fp8_max_bwd"] = self.fp8_meta["recipe"].fp8_format.value.max_bwd

            # Allocate scales and amaxes
            self.init_fp8_meta_tensors()
            self.fp8_initialized = True
        else:
            # If fp8 isn't enabled, turn off and return.
            self.fp8_initialized = False
            return

    @contextmanager
    def prepare_forward(
        self,
        inp: paddle.Tensor,
        num_gemms: int = 1,
    ) -> Generator[paddle.Tensor, None, None]:
        """Checks and prep for FWD.
        The context manager is needed because there isn't a way for a module to know
        if it's the last FP8 module in the forward autocast. It is useful
        to setup the forward aggregated amax reduction for every module
        just in case. The autocast exit will pick up the most recent one.
        """

        self.set_activation_dtype(inp)
        self.fp8_init(num_gemms=num_gemms)

        # Previous iteration was grad_enabled
        if self.fp8_meta.get("update_amax_and_scale_fwd", False):
            amax_and_scale_update(self.fp8_meta, True)

        if self.fp8 and self.training:
            self.fp8_meta["update_amax_and_scale_fwd"] = True
        else:
            self.fp8_meta["update_amax_and_scale_fwd"] = False

        with nvtx_range(self.__class__.__name__ + " forward"):
            yield inp

    @staticmethod
    def grad_output_preprocess(
            ctx, grad_output: paddle.Tensor) -> Tuple[Union[paddle.Tensor, None], ...]:
        """Utility function for backward.
        Returns tuple in order (all optional/None based on training precion/recipe):
            R1: gathered `grad_output` in higher precision.
            R2: gathered `grad_output` in FP8.
            R3: R2 transposed.
            R4: bias gradient on R1.
        """
        grad_output_mat = grad_output.reshape((-1, grad_output.shape[-1]))

        # No-FP8 case: bgrad is fused with wgrad for this case.
        if not ctx.fp8:
            return grad_output_mat, None, None, None

        fp8_dtype_backward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)

        # FP8 case without gather: cast, transpose, bgrad fused
        if ctx.use_bias:
            grad_bias, grad_output_c, grad_output_t = cast_transpose_bgrad(
                grad_output_mat,
                ctx.fp8_meta["scaling_bwd"],
                tex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
            )
            bias_dtype = get_bias_dtype(ctx.activation_dtype)
            grad_bias = cast_if_needed(grad_bias, bias_dtype)
        else:
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                grad_output_c, grad_output_t = cast_transpose(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                )
            else:
                grad_output_t = None
                grad_output_c = cast_to_fp8(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                )
            grad_bias = None

        return grad_output_mat, grad_output_c, grad_output_t, grad_bias

    @abstractmethod
    def forward(self):
        """Needs override."""
