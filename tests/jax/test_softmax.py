# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for the softmax primitives"""
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax import nn
from jax import value_and_grad, jit
from jax.typing import DTypeLike

from utils import assert_allclose

from transformer_engine.jax.cpp_extensions import is_softmax_kernel_available
from transformer_engine.jax.softmax import SoftmaxType, softmax


def catch_unsupported(method):
    """
    The unsupported case should raise error instead of running it incorrectly.
    This helper function is to check if the unsupported case raises the assertion error.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self._is_support():
            assertion_checker = pytest.raises(AssertionError)
        else:
            assertion_checker = nullcontext()
        with assertion_checker:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class SoftmaxRunner:
    """
    Softmax runner
    """

    batch_size: int
    max_seqlen_q: int
    max_seqlen_kv: int
    num_heads: int
    scale_factor: float
    softmax_type: SoftmaxType
    dtype: DTypeLike

    @staticmethod
    def reference_softmax(logits, mask, scale_factor, **_):
        """
        Jax softmax as the reference
        """
        if mask is not None:
            logits += lax.select(
                mask > 0,
                jnp.full(mask.shape, -1e10).astype(logits.dtype),
                jnp.full(mask.shape, 0.0).astype(logits.dtype),
            )
        return nn.softmax(logits * scale_factor)

    def _is_support(self):
        return is_softmax_kernel_available(
            self.softmax_type,
            self.batch_size,
            self.num_heads,
            self.max_seqlen_q,
            self.max_seqlen_kv,
            self.dtype,
        )

    def _setup_inputs(self):
        key = jax.random.PRNGKey(0)
        logits_key, mask_key = jax.random.split(key, 2)

        logits_shape = (self.batch_size, self.num_heads, self.max_seqlen_q, self.max_seqlen_kv)
        mask_shape = (self.batch_size, 1, self.max_seqlen_q, self.max_seqlen_kv)

        self.logits = jax.random.uniform(logits_key, logits_shape, self.dtype, -1.0)

        match self.softmax_type:
            case SoftmaxType.SCALED:
                self.mask = None
            case SoftmaxType.SCALED_MASKED:
                self.mask = jax.random.bernoulli(mask_key, shape=mask_shape).astype(jnp.uint8)
            case SoftmaxType.SCALED_UPPER_TRIANG_MASKED:
                self.mask = (1.0 - jnp.tril(jnp.ones_like(self.logits))).astype(jnp.uint8)
            case _:
                raise ValueError(f"Unknown {self.softmax_type=}")

    @catch_unsupported
    def test_forward(self):
        """
        Test transformer_engine.jax.softmax.softmax fwd rule
        """
        self._setup_inputs()
        primitive_out = softmax(self.logits, self.mask, self.scale_factor, self.softmax_type)
        reference_out = __class__.reference_softmax(self.logits, self.mask, self.scale_factor)
        assert_allclose(primitive_out, reference_out, dtype=self.dtype)

    @catch_unsupported
    def test_backward(self):
        """
        Test transformer_engine.jax.softmax.softmax bwd rule
        """
        self._setup_inputs()

        def grad_func(func, *args, **kwargs):
            fwd_out = func(*args, **kwargs)
            return jnp.mean(fwd_out, dtype=jnp.float32).astype(self.dtype)

        args = [self.logits, self.mask]
        kwargs = {
            "scale_factor": self.scale_factor,
            "softmax_type": self.softmax_type,
        }

        # Use FP16/BF16 to sum the results may cause overflow, use FP32 for the summation
        jitted_primitive = jit(
            value_and_grad(
                lambda logits, *args: grad_func(softmax, self.logits, *args, **kwargs), (0,)
            )
        )
        jitted_reference = jit(
            value_and_grad(
                lambda logits, *args: grad_func(
                    __class__.reference_softmax, self.logits, *args, **kwargs
                ),
                (0,),
            )
        )

        primitive_out, (primitive_grad_logits,) = jitted_primitive(*args)
        reference_out, (reference_grad_logits,) = jitted_reference(*args)

        assert_allclose(primitive_out, reference_out, dtype=self.dtype)
        assert_allclose(primitive_grad_logits, reference_grad_logits, dtype=self.dtype)


@pytest.mark.parametrize(
    "b, s_q, s_kv, h",
    [
        pytest.param(8, 16, 16, 16, id="8-16-16-16"),
        pytest.param(8, 512, 512, 16, id="8-512-512-16"),
        pytest.param(2, 8, 16384, 8, id="2-8-16384-8"),
    ],
)
@pytest.mark.parametrize("scale_factor", [0.125])
@pytest.mark.parametrize(
    "softmax_type",
    [
        pytest.param(SoftmaxType.SCALED, id="SCALED"),
        pytest.param(SoftmaxType.SCALED_MASKED, id="SCALED_MASKED"),
        pytest.param(SoftmaxType.SCALED_UPPER_TRIANG_MASKED, id="SCALED_UPPER_TRIANG_MASKED"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(jnp.bfloat16, id="BF16"),
        pytest.param(jnp.float16, id="FP16"),
    ],
)
class TestSoftmax:
    """
    Test transformer_engine.jax.softmax.softmax
    """

    @staticmethod
    def test_forward(b, s_q, s_kv, h, scale_factor, softmax_type, dtype):
        """
        Test forward with parameterized configs
        """
        runner = SoftmaxRunner(b, s_q, s_kv, h, scale_factor, softmax_type, dtype)
        runner.test_forward()

    @staticmethod
    def test_backward(b, s_q, s_kv, h, scale_factor, softmax_type, dtype):
        """
        Test forward with parameterized configs
        """
        runner = SoftmaxRunner(b, s_q, s_kv, h, scale_factor, softmax_type, dtype)
        runner.test_backward()
