# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX softmax modules"""
from enum import Enum
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from .cpp_extensions import scaled_softmax_fwd
from .cpp_extensions import scaled_softmax_bwd
from .cpp_extensions import scaled_masked_softmax_fwd
from .cpp_extensions import scaled_masked_softmax_bwd
from .cpp_extensions import scaled_upper_triang_masked_softmax_fwd
from .cpp_extensions import scaled_upper_triang_masked_softmax_bwd
from .cpp_extensions import ScaledSoftmaxFwdPrimitive
from .cpp_extensions import ScaledMaskedSoftmaxFwdPrimitive
from .cpp_extensions import ScaledUpperTriangMaskedSoftmaxFwdPrimitive


class SoftmaxType(Enum):
    """SoftmaxType."""
    SCALED = "scaled"
    SCALED_MASKED = "scaled_masked"
    SCALED_UPPER_TRIANG_MASKED = "scaled_upper_triang_masked"


def is_softmax_kernel_available(softmax_type: SoftmaxType, batch: int, heads: int, q_seqlen: int,
                                k_seqlen: int, dtype: jnp.dtype):
    """check softmax available"""
    if softmax_type is SoftmaxType.SCALED:
        return ScaledSoftmaxFwdPrimitive.is_kernel_available(batch, heads, q_seqlen, k_seqlen,
                                                             dtype)
    if softmax_type is SoftmaxType.SCALED_MASKED:
        return ScaledMaskedSoftmaxFwdPrimitive.is_kernel_available(batch, heads, q_seqlen, k_seqlen,
                                                                   dtype)
    if softmax_type is SoftmaxType.SCALED_UPPER_TRIANG_MASKED:
        return ScaledUpperTriangMaskedSoftmaxFwdPrimitive.is_kernel_available(
            batch, heads, q_seqlen, k_seqlen, dtype)

    raise NotImplementedError


def softmax(logits: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None,
            scale_factor: Optional[float] = 1.0,
            softmax_type: Optional[SoftmaxType] = SoftmaxType.SCALED):
    """
    Softmax wrapper
    """
    output = _softmax(logits, mask, scale_factor, softmax_type)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _softmax(logits, mask, scale_factor, softmax_type):

    output, _ = _softmax_fwd_rule(logits, mask, scale_factor, softmax_type)
    return output


def _softmax_fwd_rule(logits, mask, scale_factor, softmax_type):
    if softmax_type is SoftmaxType.SCALED_MASKED:
        assert mask is not None
        output = scaled_masked_softmax_fwd(logits, mask, scale_factor)
    elif softmax_type is SoftmaxType.SCALED_UPPER_TRIANG_MASKED:
        output = scaled_upper_triang_masked_softmax_fwd(logits, scale_factor)
    else:
        output = scaled_softmax_fwd(logits, scale_factor)

    return output, (output,)


def _softmax_bwd_rule(scale_factor, softmax_type, ctx, dz):
    softmax_output, = ctx

    if softmax_type is SoftmaxType.SCALED_MASKED:
        dgrad = scaled_masked_softmax_bwd(dz, softmax_output, scale_factor)
    elif softmax_type is SoftmaxType.SCALED_UPPER_TRIANG_MASKED:
        dgrad = scaled_upper_triang_masked_softmax_bwd(dz, softmax_output, scale_factor)
    else:
        dgrad = scaled_softmax_bwd(dz, softmax_output, scale_factor)

    return (dgrad, None)


_softmax.defvjp(_softmax_fwd_rule, _softmax_bwd_rule)
