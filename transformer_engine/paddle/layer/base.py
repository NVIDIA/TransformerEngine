# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Base modules and utilities for TransformerEngine Paddle API"""

from abc import ABC, abstractmethod
from contextlib import contextmanager

import paddle
from paddle.fluid import core
from paddle.fluid.framework import _dygraph_tracer

from ..profile import nvtx_range

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


class TransformerEngineBaseLayer(paddle.nn.Layer, ABC):
    """Base TE Layer."""

    def __init__(self) -> None:
        super().__init__()
        assert 'gpu' in paddle.device.get_device(), "TransformerEngine needs CUDA."

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

    @contextmanager
    def prepare_forward(
        self,
        inp: paddle.Tensor,
    ) -> None:
        """
        Checks and prep for FWD.
        """

        self.set_activation_dtype(inp)

        with nvtx_range(self.__class__.__name__ + " forward"):
            yield inp

    @abstractmethod
    def forward(self):
        """Needs override."""
