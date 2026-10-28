# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Device type and platform used by Transformer Engine"""
import torch
import transformer_engine

# Device type and platform globals — live here because torch cannot be imported
# at the top-level transformer_engine package (shared with JAX).
# Use plugin-set values if available, otherwise fall back to CUDA defaults.
transformer_engine.TE_DEVICE_TYPE = getattr(transformer_engine, "TE_DEVICE_TYPE", "cuda")
transformer_engine.TE_PLATFORM = getattr(transformer_engine, "TE_PLATFORM", torch.cuda)


def te_device_type() -> str:
    """Torch device type used by Transformer Engine, e.g. `cuda`."""
    return transformer_engine.TE_DEVICE_TYPE


def te_platform():
    """Torch device module used by Transformer Engine, e.g. `torch.cuda`."""
    return transformer_engine.TE_PLATFORM
