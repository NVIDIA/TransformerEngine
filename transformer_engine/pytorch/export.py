# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Export utilities for TransformerEngine"""
from contextlib import contextmanager

_IN_ONNX_EXPORT_MODE = False


@contextmanager
def onnx_export(
    enabled: bool = False,
) -> None:
    """
    Context manager for exporting to ONNX.

    .. code-block:: python

        with onnx_export(enabled=True):
            torch.onnx.export(model)

    Parameters
    ----------
    enabled: bool, default = `False`
             whether or not to enable export
    """

    global _IN_ONNX_EXPORT_MODE
    onnx_export_state = _IN_ONNX_EXPORT_MODE
    try:
        _IN_ONNX_EXPORT_MODE = enabled
        yield
    finally:
        _IN_ONNX_EXPORT_MODE = onnx_export_state


def is_in_onnx_export_mode() -> bool:
    """Returns True if onnx export mode is enabled, False otherwise."""
    return _IN_ONNX_EXPORT_MODE
