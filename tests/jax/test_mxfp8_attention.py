# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 attention preparation, numerical, and integration tests."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformer_engine.common.cudnn import mxfp8
from transformer_engine.common.cudnn.graph import mxfp8_scale_spec
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    QKVLayout,
    SequenceDescriptor,
    fused_attn,
)
from transformer_engine.jax.cpp_extensions import cudnn_attention
from transformer_engine.jax.cpp_extensions import mxfp8_attention


@pytest.mark.parametrize("columnwise", (False, True))
@pytest.mark.parametrize("dim", ((2, 3, 128, 64), (1, 4, 256, 96), (2, 1, 384, 160)))
def test_scale_reordering_matches_graph(dim, columnwise):
    """Preparation and cuDNN agree on padding each batch/head scale matrix."""
    batch, heads, seqlen, head_dim = dim
    scale_shape = (
        (batch, heads, seqlen // 32, head_dim)
        if columnwise
        else (batch, heads, seqlen, head_dim // 32)
    )
    padded, tiled, permutation = mxfp8.scale_reordering(scale_shape, columnwise)
    assert padded == mxfp8_scale_spec(dim, "bhsd", columnwise).dim
    assert math.prod(tiled) == math.prod(padded)
    assert sorted(permutation) == list(range(len(tiled)))


@pytest.mark.parametrize(
    "sm,version,fmt,error",
    (
        (90, 92600, "bshd", "SM100"),
        (100, 92000, "bshd", "9.21"),
        (100, 92300, "bshd", "defects"),
        (100, 92301, "bshd", "defects"),
        (100, 92600, "thd", "THD"),
        (100, 92100, "bshd", ""),
        (100, 92600, "sbhd", ""),
    ),
)
def test_shared_mxfp8_support(sm, version, fmt, error):
    """Both frontends retain the same hardware and known-defect exclusions."""
    reason = mxfp8.support_error(sm, version, fmt)
    assert error in reason if error else not reason


def test_quantization_directions():
    """Forward V contracts over sequence; backward Q/K need both directions."""
    assert mxfp8.quantization_layouts() == ((True, False), (True, False), (False, True))
    assert mxfp8.quantization_layouts(True) == ((True, True), (True, True), (True, False))
    assert not mxfp8.shape_error(128, 256, 64, 96)
    assert mxfp8.shape_error(129, 256, 64, 96)
    assert mxfp8.shape_error(128, 256, 64, 80)


def _require_mxfp8():
    reason = mxfp8.support_error(
        cudnn_attention.device_capabilities()[0], cudnn_attention.cudnn_version(), "bshd"
    )
    if reason:
        pytest.skip(reason)


def _attention(qkv, **kwargs):
    options = dict(
        attn_bias_type=AttnBiasType.NO_BIAS,
        attn_mask_type=AttnMaskType.NO_MASK,
        softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
        qkv_layout=QKVLayout.BSHD_BSHD_BSHD,
        scaling_factor=0.125,
        dropout_probability=0.0,
        is_training=True,
        fp8="mxfp8",
    )
    options.update(kwargs)
    return fused_attn(qkv, None, SequenceDescriptor(), None, **options)


def _reference(qkv):
    q, k, v = (x.astype(jnp.float32) for x in qkv)
    repeats = q.shape[-2] // k.shape[-2]
    k, v = (jnp.repeat(x, repeats, axis=-2) for x in (k, v))
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * 0.125
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, -1), v)


@pytest.mark.parametrize("dtype", (jnp.float16, jnp.bfloat16))
@pytest.mark.parametrize("fp8_backward", (False, True))
@pytest.mark.parametrize("is_training", (False, True))
def test_dense_forward_backward(dtype, fp8_backward, is_training, monkeypatch):
    """Exactly representable inputs/probabilities retain model-precision accuracy."""
    _require_mxfp8()
    monkeypatch.setenv("NVTE_FP8_DPA_BWD", "1" if fp8_backward else "0")
    q = jnp.full((1, 128, 2, 64), 0.5, dtype)
    k = jnp.full((1, 128, 1, 64), 0.5, dtype)
    v = jnp.zeros_like(k).at[:, ::2, :, 0].set(1.0)
    inputs = (q, k, v)
    attention = lambda xs: _attention(xs, is_training=is_training)
    actual = jax.jit(attention)(inputs)
    expected = jax.jit(_reference)(inputs)
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-4)
    assert actual.dtype == dtype
    grads = jax.jit(jax.grad(lambda xs: attention(xs).sum()))(inputs)
    reference_grads = jax.jit(jax.grad(lambda xs: _reference(xs).sum()))(inputs)
    for actual_grad, expected_grad in zip(grads, reference_grads):
        assert actual_grad.dtype == dtype
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("columnwise", (False, True))
@pytest.mark.parametrize("head_dim", (64, 96, 160))
def test_quantization_and_scale_bytes(columnwise, head_dim):
    """Native quantization and cuDNN scale packing agree with an independent reference."""
    _require_mxfp8()
    rng = np.random.default_rng(17)
    value = jnp.asarray(rng.normal(size=(2, 3, 128, head_dim)), jnp.bfloat16)
    value *= jnp.asarray([0.01, 1.0, 100.0], jnp.bfloat16)[None, :, None, None]
    row, col = jax.jit(lambda x: mxfp8_attention.quantize_operand(x, not columnwise, columnwise))(
        value
    )
    tensor = col if columnwise else row
    source = np.asarray(value, np.float32)
    oriented = source.swapaxes(-1, -2) if columnwise else source
    blocks = oriented.reshape(*oriented.shape[:-1], -1, 32)
    maximum = np.max(np.abs(blocks), axis=-1)
    exponent = np.maximum(np.ceil(np.log2(np.maximum(maximum / 448.0, 2.0**-127))), -127)
    scales = np.exp2(exponent)
    encoded = (blocks / scales[..., None]).astype(jnp.float8_e4m3fn).reshape(oriented.shape)
    if columnwise:
        encoded = encoded.swapaxes(-1, -2)
    np.testing.assert_array_equal(np.asarray(tensor.data).view(np.uint8), encoded.view(np.uint8))
    scales = (exponent + 127).astype(np.uint8)
    rows, cols = scales.shape[-2:]
    padded_rows = ((rows + 127) // 128) * 128
    padded_cols = ((cols + 3) // 4) * 4
    expected = np.zeros((*scales.shape[:2], padded_rows * padded_cols), np.uint8)
    # Physical offset within each 128x4 tile, derived independently of the
    # reshape/transpose used by the implementation.
    for i in range(rows):
        for j in range(cols):
            index = ((i // 128) * (padded_cols // 4) + j // 4) * 512
            index += (i % 32) * 16 + ((i % 128) // 32) * 4 + j % 4
            expected[..., index] = scales[..., i, j]
    actual = jax.jit(lambda s: mxfp8_attention.reorder_scale(s, columnwise))(tensor.scale_inv)
    np.testing.assert_array_equal(np.asarray(actual).reshape(expected.shape), expected)


@pytest.mark.parametrize("deterministic", (False, True))
def test_flax_mxfp8_mode(deterministic):
    """Flax forwards the requested block-scaling mode without coercing it to bool."""
    from transformer_engine.jax.flax import DotProductAttention

    _require_mxfp8()
    module = DotProductAttention(
        head_dim=64,
        num_attention_heads=2,
        attn_mask_type="no_mask",
        transpose_batch_sequence=False,
        fp8="mxfp8",
    )
    q = jnp.full((1, 128, 2, 64), 0.5, jnp.bfloat16)
    value = jnp.zeros_like(q).at[:, ::2, :, 0].set(1.0)
    variables = module.init(jax.random.key(0), q, q, value, deterministic=deterministic)
    forward = lambda v: module.apply(variables, q, q, v, deterministic=deterministic)
    actual = jax.jit(forward)(value)
    np.testing.assert_allclose(actual[..., 0], 0.5, rtol=5e-3, atol=5e-4)
    np.testing.assert_allclose(jax.jit(jax.grad(lambda v: forward(v).sum()))(value), 1.0)
