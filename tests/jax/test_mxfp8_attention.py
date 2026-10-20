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
    quantize_attention_qkv,
)
from transformer_engine.jax.cpp_extensions import cudnn_attention
from transformer_engine.jax.cpp_extensions import mxfp8_attention
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


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
        (120, 92600, "bshd", "SM12x"),
        (121, 92600, "bshd", "SM12x"),
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
    descriptor = options.pop("sequence_descriptor", SequenceDescriptor())
    seed = options.pop("seed", None)
    return fused_attn(qkv, None, descriptor, seed, **options)


def _reference(qkv, mask=None, sink=None):
    q, k, v = (x.astype(jnp.float32) for x in qkv)
    repeats = q.shape[-2] // k.shape[-2]
    k, v = (jnp.repeat(x, repeats, axis=-2) for x in (k, v))
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * 0.125
    if mask is not None:
        scores = jnp.where(mask[:, None], scores, -1e30)
    if sink is None:
        probability = jax.nn.softmax(scores, -1)
    else:
        sink = jnp.broadcast_to(sink, (*scores.shape[:-1], 1))
        probability = jax.nn.softmax(jnp.concatenate((scores, sink), axis=-1), -1)[..., :-1]
    output = jnp.einsum("bhqk,bkhd->bqhd", probability, v)
    if mask is not None:
        output = jnp.where(jnp.any(mask, axis=-1)[..., None, None], output, 0.0)
    return output


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


def _pack(qkv, layout):
    q, k, v = qkv
    if layout.is_qkvpacked():
        return (jnp.stack((q, k, v), axis=-3),)
    if layout.is_kvpacked():
        return q, jnp.stack((k, v), axis=-3)
    return qkv


def _unpack(qkv, layout):
    if layout.is_qkvpacked():
        return tuple(qkv[0][..., i, :, :] for i in range(3))
    if layout.is_kvpacked():
        return qkv[0], qkv[1][..., 0, :, :], qkv[1][..., 1, :, :]
    return qkv


@pytest.mark.parametrize("layout", (QKVLayout.BS3HD, QKVLayout.BSHD_BS2HD))
@pytest.mark.parametrize("dtype", (jnp.float16, jnp.bfloat16))
@pytest.mark.parametrize("fp8_backward", (False, True))
def test_packed_forward_backward(layout, dtype, fp8_backward, monkeypatch):
    """Packed result storage preserves every gradient slice and original layout."""
    _require_mxfp8()
    monkeypatch.setenv("NVTE_FP8_DPA_BWD", "1" if fp8_backward else "0")
    q = jnp.full((2, 128, 2, 64), 0.5, dtype)
    kv = q if layout.is_qkvpacked() else q[:, :, :1]
    value = jnp.zeros_like(kv).at[:, ::2, :, 0].set(1.0)
    inputs = _pack((q, kv, value), layout)
    attention = lambda xs: _attention(xs, qkv_layout=layout)
    reference = lambda xs: _reference(_unpack(xs, layout))
    np.testing.assert_allclose(jax.jit(attention)(inputs), reference(inputs), rtol=5e-3, atol=5e-4)
    actual = jax.jit(jax.grad(lambda xs: attention(xs).sum()))(inputs)
    expected = jax.jit(jax.grad(lambda xs: reference(xs).sum()))(inputs)
    for grad, ref_grad, operand in zip(actual, expected, inputs):
        assert grad.shape == operand.shape
        assert grad.dtype == dtype
        np.testing.assert_allclose(grad, ref_grad, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("kind", ("dense", "causal", "bottom_right", "padding", "window"))
@pytest.mark.parametrize("fp8_backward", (False, True))
def test_gqa_masks(kind, fp8_backward, monkeypatch):
    """Bound the error from cuDNN's internal FP8 probabilities and dP on nonuniform data."""
    _require_mxfp8()
    monkeypatch.setenv("NVTE_FP8_DPA_BWD", "1" if fp8_backward else "0")
    keys = jax.random.split(jax.random.key(123), 4)
    sq, sk = (128, 256) if kind == "bottom_right" else (128, 128)
    q = jax.random.choice(keys[0], jnp.array([-0.5, 0.5], jnp.bfloat16), (2, sq, 4, 64))
    k = jax.random.choice(keys[1], jnp.array([-0.5, 0.5], jnp.bfloat16), (2, sk, 2, 64))
    v = jax.random.choice(keys[2], jnp.array([-1.0, 1.0], jnp.bfloat16), k.shape)
    weight = jax.random.choice(keys[3], jnp.array([-1.0, 1.0], jnp.bfloat16), q.shape)
    qpos, kpos = jnp.arange(sq)[:, None], jnp.arange(sk)[None, :]
    mask = jnp.ones((2, sq, sk), jnp.bool_)
    options = {}
    if kind in ("causal", "bottom_right"):
        diagonal = sk - sq if kind == "bottom_right" else 0
        mask &= kpos <= qpos + diagonal
        options["attn_mask_type"] = (
            AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK if diagonal else AttnMaskType.CAUSAL_MASK
        )
    elif kind == "padding":
        qlength, klength = jnp.array([112, 96]), jnp.array([96, 80])
        mask &= (qpos[None] < qlength[:, None, None]) & (kpos[None] < klength[:, None, None])
        options.update(
            attn_mask_type=AttnMaskType.PADDING_MASK,
            sequence_descriptor=SequenceDescriptor.from_seqlens((qlength, klength)),
        )
    elif kind == "window":
        mask &= (kpos >= qpos - 31) & (kpos <= qpos + 15)
        options["window_size"] = (31, 15)
    inputs = (q, k, v)
    attention = lambda xs: _attention(xs, **options)
    reference = lambda xs: _reference(xs, mask)
    actual = jax.jit(attention)(inputs)
    expected = jax.jit(reference)(inputs)
    relative_error = lambda a, b: float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    assert relative_error(actual, expected) < 0.04
    actual_grads = jax.jit(jax.grad(lambda xs: (attention(xs) * weight).sum()))(inputs)
    expected_grads = jax.jit(jax.grad(lambda xs: (reference(xs) * weight).sum()))(inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert relative_error(actual_grad, expected_grad) < 0.08


@pytest.mark.parametrize("softmax", tuple(AttnSoftmaxType))
@pytest.mark.parametrize("deterministic", (False, True))
def test_softmax_variants(softmax, deterministic, monkeypatch):
    """MXFP8 attention preserves attention-sink outputs and learnable-sink gradients."""
    _require_mxfp8()
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0" if deterministic else "1")
    q = jnp.zeros((1, 128, 2, 64), jnp.bfloat16)
    v = jnp.zeros_like(q).at[:, ::2, :, 0].set(1.0)
    sink = jnp.zeros((1, 2, 1, 1), jnp.float32)
    options = dict(softmax_type=softmax)
    if softmax == AttnSoftmaxType.LEARNABLE_SOFTMAX:
        options["softmax_offset"] = sink
    actual = jax.jit(lambda xs: _attention(xs, **options))((q, q, v))
    denominator = 128 if softmax == AttnSoftmaxType.VANILLA_SOFTMAX else 129
    expected = jnp.zeros_like(actual).at[..., 0].set(64 / denominator)
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=5e-4)
    gradients = jax.jit(jax.grad(lambda xs: _attention(xs, **options).sum()))((q, q, v))
    assert all(bool(jnp.all(jnp.isfinite(grad))) for grad in gradients)
    np.testing.assert_allclose(gradients[2], 128 / denominator, rtol=1e-2, atol=5e-4)
    if softmax == AttnSoftmaxType.LEARNABLE_SOFTMAX:
        dsink = jax.jit(
            jax.grad(lambda s: _attention((q, q, v), softmax_type=softmax, softmax_offset=s).sum())
        )(sink)
        np.testing.assert_allclose(dsink, -128 * 64 / denominator**2, rtol=2e-2, atol=1e-3)


@pytest.mark.parametrize("head_dims", ((64, 128), (128, 64), (128, 128)))
def test_different_head_dimensions(head_dims):
    """Per-head scales retain their direction when QK and V dimensions differ."""
    _require_mxfp8()
    dq, dv = head_dims
    q = jnp.full((1, 128, 2, dq), 0.5, jnp.bfloat16)
    k = jnp.full((1, 256, 1, dq), 0.5, jnp.bfloat16)
    v = jnp.zeros((1, 256, 1, dv), jnp.bfloat16).at[:, ::2, :, 0].set(1.0)
    inputs = (q, k, v)
    np.testing.assert_allclose(jax.jit(_attention)(inputs), _reference(inputs), atol=5e-4)
    actual = jax.jit(jax.grad(lambda xs: _attention(xs).sum()))(inputs)
    expected = jax.jit(jax.grad(lambda xs: _reference(xs).sum()))(inputs)
    for grad, ref_grad in zip(actual, expected):
        np.testing.assert_allclose(grad, ref_grad, rtol=1e-2, atol=1e-3)


def test_checkpoint_and_dropout_guard(monkeypatch):
    """Recomputation preserves gradients and unsupported dropout has an actionable error."""
    _require_mxfp8()
    monkeypatch.setenv("NVTE_FP8_DPA_BWD", "0")
    q = jnp.zeros((1, 128, 2, 64), jnp.bfloat16)
    v = jnp.zeros_like(q).at[..., 0].set(1.0)
    inputs = (q, q, v)
    forward = lambda xs, key: _attention(xs, seed=key)
    key = jax.random.PRNGKey(41)
    actual = jax.jit(forward)(inputs, key)
    np.testing.assert_array_equal(actual, jax.jit(forward)(inputs, key))
    np.testing.assert_allclose(actual[..., 0], 1.0)
    plain = jax.jit(jax.value_and_grad(lambda xs: forward(xs, key).sum()))(inputs)
    recomputed = jax.jit(jax.value_and_grad(lambda xs: jax.checkpoint(forward)(xs, key).sum()))(
        inputs
    )
    for value, reference in zip(jax.tree.leaves(plain), jax.tree.leaves(recomputed)):
        assert bool(jnp.all(jnp.isfinite(value)))
        np.testing.assert_array_equal(value, reference)
    with pytest.raises(ValueError, match="MXFP8 attention does not support dropout"):
        jax.jit(lambda xs: _attention(xs, seed=key, dropout_probability=0.25))(inputs)
    # Inference ignores the configured training dropout probability.
    np.testing.assert_array_equal(
        actual,
        jax.jit(lambda xs: _attention(xs, dropout_probability=0.25, is_training=False))(inputs),
    )


@pytest.mark.parametrize(
    "layout", (QKVLayout.BS3HD, QKVLayout.BSHD_BS2HD, QKVLayout.BSHD_BSHD_BSHD)
)
@pytest.mark.parametrize("fp8_backward", (False, True))
def test_prequantized_inputs(layout, fp8_backward, monkeypatch):
    """Reuse supplied quantizations and return gradients to the model-precision arrays."""
    _require_mxfp8()
    monkeypatch.setenv("NVTE_FP8_DPA_BWD", "1" if fp8_backward else "0")
    q = jnp.full((1, 128, 2, 64), 0.5, jnp.bfloat16)
    v = jnp.zeros_like(q).at[:, ::2, :, 0].set(1.0)
    inputs = _pack((q, q, v), layout)
    prepared = jax.jit(quantize_attention_qkv)((q, q, v))
    reference = lambda xs: _attention(xs, qkv_layout=layout)
    expected = jax.jit(jax.value_and_grad(lambda xs: reference(xs).sum()))(inputs)

    def fail_requantization(*args, **kwargs):
        raise AssertionError("Supplied QKV must be consumed without requantization")

    monkeypatch.setattr(mxfp8_attention, "prepare_qkv", fail_requantization)
    forward = lambda xs, buffers: _attention(xs, qkv_layout=layout, quantized_qkv=buffers)
    actual = jax.jit(jax.value_and_grad(lambda xs, ys: forward(xs, ys).sum()))(inputs, prepared)
    for value, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert value.dtype == ref.dtype
        np.testing.assert_array_equal(value, ref)
    # In one compiled training step, buffer preparation is non-differentiable
    # while the attention VJP still supplies model-precision QKV gradients.
    integrated = jax.jit(
        jax.value_and_grad(
            lambda xs: forward(xs, quantize_attention_qkv(_unpack(xs, layout))).sum()
        )
    )(inputs)
    for value, ref in zip(jax.tree.leaves(integrated), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(value, ref)


@pytest.mark.parametrize("axis", ("dp", "tp"))
def test_prequantized_sharding(axis):
    """Per-head scale packing happens after data/tensor parallel partitioning."""
    _require_mxfp8()
    if jax.local_device_count() < 2:
        pytest.skip("Requires two GPUs for attention sharding")
    mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:2]), (axis,))
    spec = (axis, None, None, None) if axis == "dp" else (None, None, axis, None)
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
    q = jnp.full((2, 128, 4, 64), 0.5, jnp.bfloat16)
    kv = q[:, :, :2]
    v = jnp.zeros_like(kv).at[:, ::2, :, 0].set(1.0)
    inputs = (q, kv, v)
    loss = lambda xs: _attention(xs, quantized_qkv=quantize_attention_qkv(xs)).sum()
    expected = jax.jit(jax.value_and_grad(loss))(inputs)
    with jax.set_mesh(mesh), global_shard_guard(MeshResource(**{axis + "_resource": axis})):
        actual = jax.jit(jax.value_and_grad(loss), in_shardings=((sharding,) * 3,))(inputs)
    for value, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(value, ref)


def test_prequantized_values_and_validation():
    """Supplied FP8 values determine the output; incompatible modes fail at the API boundary."""
    _require_mxfp8()
    q = jnp.zeros((1, 128, 2, 64), jnp.bfloat16)
    v = q.at[..., 0].set(1.0)
    prepared = jax.jit(quantize_attention_qkv)((q, q, v))
    output = jax.jit(lambda xs, ys: _attention(xs, quantized_qkv=ys))((q, q, q), prepared)
    np.testing.assert_allclose(output, v)
    with pytest.raises(ValueError, match="requires fp8"):
        _attention((q, q, v), quantized_qkv=prepared, fp8=True)
    with pytest.raises(ValueError, match="three separate BSHD"):
        quantize_attention_qkv((q,))
    with pytest.raises(ValueError, match="FP16/BF16"):
        quantize_attention_qkv((q.astype(jnp.float32), q, v))
    with pytest.raises(ValueError, match="unswizzled E4M3"):
        swapped = prepared[0].__class__(prepared[2].colwise_tensor, prepared[2].rowwise_tensor)
        _attention((q, q, v), quantized_qkv=(prepared[0], prepared[0], swapped))
