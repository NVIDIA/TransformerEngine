# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile / Dynamo integration: opaque type registrations for quantizers.

Non-delayed quantizers are registered as value-typed opaque objects so that
torch.compile can bake them as constants and guard on __eq__.
Float8Quantizer (delayed scaling) is not registered here as delayed scaling
is being deprecated and slated for removal.

Requires PyTorch >= 2.14 (opaque object API). On older versions this module
is a no-op.
"""

try:
    from torch._library.opaque_object import is_opaque_type, register_opaque_type, MemberType
except ImportError:
    # PyTorch too old -- opaque object API not available.
    pass
else:
    from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    _quantizer_value_members = {
        "__setattr__": MemberType.USE_REAL,
        "__bool__": MemberType.USE_REAL,
        "set_usage": MemberType.USE_REAL,
    }

    if not is_opaque_type(Float8CurrentScalingQuantizer):
        register_opaque_type(
            Float8CurrentScalingQuantizer, typ="value", members=_quantizer_value_members
        )

    if not is_opaque_type(MXFP8Quantizer):
        register_opaque_type(MXFP8Quantizer, typ="value", members=_quantizer_value_members)

    if not is_opaque_type(Float8BlockQuantizer):
        register_opaque_type(Float8BlockQuantizer, typ="value", members=_quantizer_value_members)

    if not is_opaque_type(NVFP4Quantizer):
        register_opaque_type(NVFP4Quantizer, typ="value", members=_quantizer_value_members)
