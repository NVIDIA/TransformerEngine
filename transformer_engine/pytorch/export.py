# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Export utilities for TransformerEngine"""

from contextlib import contextmanager
from typing import Generator
import torch


_IN_ONNX_EXPORT_MODE = False
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])


@contextmanager
def onnx_export(enabled: bool = False) -> Generator[None, None, None]:
    """
    Context manager for exporting to ONNX.

    .. code-block:: python

        from transformer_engine.pytorch.export import onnx_export, te_translation_table

        with onnx_export(enabled=True):
            torch.onnx.export(model, dynamo=True, custom_translation_table=te_translation_table)

    Parameters
    ----------
    enabled : bool, default = False
             whether or not to enable export
    """

    global _IN_ONNX_EXPORT_MODE
    onnx_export_state = _IN_ONNX_EXPORT_MODE
    if (TORCH_MAJOR, TORCH_MINOR) < (2, 4):
        raise RuntimeError("ONNX export is not supported for PyTorch versions less than 2.4")
    try:
        _IN_ONNX_EXPORT_MODE = enabled
        yield
    finally:
        _IN_ONNX_EXPORT_MODE = onnx_export_state


def is_in_onnx_export_mode() -> bool:
    """Returns True if onnx export mode is enabled, False otherwise."""
    return _IN_ONNX_EXPORT_MODE


def assert_warmed_up(module: torch.nn.Module) -> None:
    """Assert that the model has been warmed up before exporting to ONNX."""
    assert hasattr(module, "forwarded_at_least_once"), (
        "Model must be warmed up before exporting to ONNX, please run model with the"
        " same recipe before exporting."
    )


if TORCH_MAJOR == 2 and TORCH_MINOR >= 4 or TORCH_MAJOR > 2:
    # pylint: disable=unused-import
    from .onnx_extensions import (
        torch_onnx_gemm_inf_op,
        onnx_quantize_fp8_op,
        onnx_dequantize_fp8_op,
        onnx_quantize_mxfp8_op,
        onnx_dequantize_mxfp8_op,
        onnx_layernorm,
        onnx_attention_mask_func,
        onnx_gemm,
        te_translation_table,
    )
