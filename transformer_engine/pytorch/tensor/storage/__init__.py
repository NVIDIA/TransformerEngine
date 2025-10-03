# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Storage for quantized tensors."""

from .float8_tensor_storage import Float8TensorStorage  # noqa: F401
from .mxfp8_tensor_storage import MXFP8TensorStorage  # noqa: F401
from .float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage  # noqa: F401
from .nvfp4_tensor_storage import NVFP4TensorStorage  # noqa: F401
