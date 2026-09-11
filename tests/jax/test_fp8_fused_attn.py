# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Coverage for JAX FP8 DPA and attention features shared with PyTorch."""

from math import sqrt
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformer_engine_jax import get_cudnn_version, get_device_compute_capability

from transformer_engine.common import recipe
from transformer_engine.jax import autocast
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    QKVLayout,
    SequenceDescriptor,
    fused_attn,
)
from transformer_engine.jax.cpp_extensions import FusedAttnHelper
from transformer_engine.jax.cpp_extensions.fp8_attention import (
    _mx_scale,
    _mxfp8_scale_inv,
    _validate_quantizer_modes,
)
from transformer_engine.jax.flax import DotProductAttention
from transformer_engine.jax.quantize import (
    AttentionQuantizerSet,
    BlockScaleQuantizer,
    QuantizeLayout,
    ScalingMode,
)
from transformer_engine.jax.sharding import MeshResource


def _require_gpu(min_arch=90, min_cudnn=90700, max_arch=None):
    try:
        if not any(device.platform == "gpu" for device in jax.devices()):
            pytest.skip("A CUDA device is required.")
        arch = get_device_compute_capability(0)
    except RuntimeError as exc:
        pytest.skip(f"A usable CUDA device is required: {exc}")
    if arch < min_arch:
        pytest.skip(f"This test requires SM{min_arch} or newer, found SM{arch}.")
    if max_arch is not None and arch >= max_arch:
        pytest.skip(
            f"This test requires an architecture older than SM{max_arch}, found SM{arch}."
        )
    cudnn_version = get_cudnn_version()
    if cudnn_version < min_cudnn:
        pytest.skip(f"This test requires cuDNN {min_cudnn}, found {cudnn_version}.")
    if cudnn_version == 91000:
        pytest.skip("cuDNN 9.10.0 has known FP8 SDPA issues.")
    return arch


def _reference_attention(q, k, v, *, bottom_right=False, alibi=False):
    q_seqlen, kv_seqlen = q.shape[1], k.shape[1]
    scores = jnp.einsum(
        "bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)
    ) / sqrt(q.shape[-1])
    q_pos = jnp.arange(q_seqlen)[:, None]
    kv_pos = jnp.arange(kv_seqlen)[None, :]
    shift = kv_seqlen - q_seqlen if bottom_right else 0
    if alibi:
        heads = q.shape[-2]
        power_of_two_heads = 2 ** int(np.floor(np.log2(heads)))
        base = 2.0 ** (-8.0 / power_of_two_heads)
        slopes = base ** jnp.arange(1, power_of_two_heads + 1, dtype=jnp.float32)
        if power_of_two_heads < heads:
            extra_base = 2.0 ** (-4.0 / power_of_two_heads)
            extra = extra_base ** jnp.arange(
                1, 1 + 2 * (heads - power_of_two_heads), 2, dtype=jnp.float32
            )
            slopes = jnp.concatenate((slopes, extra))
        distance = jnp.abs(q_pos + shift - kv_pos).astype(jnp.float32)
        scores -= slopes[None, :, None, None] * distance[None, None, :, :]
    allowed = kv_pos <= q_pos + shift
    scores = jnp.where(allowed[None, None, :, :], scores, -jnp.inf)
    probabilities = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", probabilities, v.astype(jnp.float32)).astype(
        q.dtype
    )


def _assert_fp8_close(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    np.testing.assert_allclose(actual, expected, atol=0.5, rtol=0.05)
    assert np.sqrt(np.mean(np.square(actual - expected))) < 0.11


def _mode_quantizer(mode):
    return SimpleNamespace(scaling_mode=mode)


def test_fp8_attention_quantizer_mode_assignments():
    """Current-scaling DPA uses delayed scaling for its internal S and dP tensors."""

    current = _mode_quantizer(ScalingMode.CURRENT_TENSOR_SCALING)
    delayed = _mode_quantizer(ScalingMode.DELAYED_TENSOR_SCALING)
    quantizers = AttentionQuantizerSet(
        qkv=current,
        s=delayed,
        o=current,
        do=current,
        dp=delayed,
        dqkv=current,
    )
    assert _validate_quantizer_modes(quantizers) == "current"

    quantizers.s = current
    with pytest.raises(ValueError, match=r"s=current \(expected delayed\)"):
        _validate_quantizer_modes(quantizers)


@pytest.mark.parametrize(
    "fp8_recipe,min_arch,min_cudnn",
    (
        pytest.param(
            recipe.DelayedScaling(amax_history_len=1, fp8_dpa=True),
            90,
            90700,
            id="delayed",
        ),
        pytest.param(
            recipe.Float8CurrentScaling(fp8_dpa=True),
            100,
            91400,
            id="current",
        ),
        pytest.param(
            recipe.MXFP8BlockScaling(fp8_dpa=True),
            100,
            92100,
            id="mxfp8",
        ),
    ),
)
@pytest.mark.parametrize(
    "input_dtype", (jnp.float16, jnp.bfloat16), ids=("float16", "bfloat16")
)
def test_fp8_dpa_forward_backward(fp8_recipe, min_arch, min_cudnn, input_dtype):
    """Each supported recipe executes FP8 DPA behind FP16/BF16 module boundaries."""

    _require_gpu(min_arch, min_cudnn, max_arch=120)
    if fp8_recipe.mxfp8() and get_cudnn_version() in (92300, 92301):
        pytest.skip("cuDNN 9.23.0 and 9.23.1 have known MXFP8 SDPA correctness issues.")
    batch, seqlen, heads, dim = 2, 128, 8, 128
    q_key, k_key, v_key, do_key = jax.random.split(jax.random.PRNGKey(1234), 4)
    shape = (batch, seqlen, heads, dim)
    q = jax.random.uniform(q_key, shape, input_dtype, minval=-0.5, maxval=0.5)
    k = jax.random.uniform(k_key, shape, input_dtype, minval=-0.5, maxval=0.5)
    v = jax.random.uniform(v_key, shape, input_dtype, minval=-0.5, maxval=0.5)
    doutput = jax.random.uniform(do_key, shape, input_dtype, minval=-0.5, maxval=0.5)
    seqlens = jnp.full((batch,), seqlen, dtype=jnp.int32)
    descriptor = SequenceDescriptor.from_seqlens((seqlens, seqlens))
    module = DotProductAttention(
        head_dim=dim,
        num_attention_heads=heads,
        num_gqa_groups=heads,
        attn_mask_type="causal",
        qkv_layout="bshd_bshd_bshd",
        transpose_batch_sequence=False,
    )

    def loss_fn(variables, query, key, value):
        output = module.apply(
            variables, query, key, value, descriptor, deterministic=False
        )
        loss = jnp.sum(output.astype(jnp.float32) * doutput.astype(jnp.float32))
        return loss, output

    def reference_loss(query, key, value):
        output = _reference_attention(query, key, value)
        loss = jnp.sum(output.astype(jnp.float32) * doutput.astype(jnp.float32))
        return loss, output

    with autocast(enabled=True, recipe=fp8_recipe, mesh_resource=MeshResource()):
        variables = module.init(
            jax.random.PRNGKey(0), q, k, v, descriptor, deterministic=False
        )
        (_, output), (_, dq, dk, dv) = jax.value_and_grad(
            loss_fn, argnums=(0, 1, 2, 3), has_aux=True
        )(variables, q, k, v)
    (_, reference), (dq_ref, dk_ref, dv_ref) = jax.value_and_grad(
        reference_loss, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)

    assert output.dtype == q.dtype
    _assert_fp8_close(output, reference)
    for actual, expected in zip((dq, dk, dv), (dq_ref, dk_ref, dv_ref)):
        _assert_fp8_close(actual, expected)


def test_mxfp8_attention_scale_layout():
    """Attention pads, permutes, and swizzles compact JAX MXFP8 scales for cuDNN."""

    quantizer = BlockScaleQuantizer(
        q_dtype=jnp.float8_e4m3fn,
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        q_layout=QuantizeLayout.ROWWISE_COLWISE,
        data_layout="NN",
    )
    tensor = quantizer.quantize(
        jnp.ones((2, 64, 8, 64), dtype=jnp.bfloat16), flatten_axis=-2
    )
    assert tensor.rowwise_tensor.scale_inv.shape == (2, 64, 8, 2)
    assert tensor.colwise_tensor.scale_inv.shape == (2, 2, 8, 64)
    assert _mxfp8_scale_inv(tensor).shape == (2, 8, 128, 4)
    assert _mxfp8_scale_inv(tensor, colwise=True).shape == (2, 8, 4, 128)


def test_mxfp8_attention_scale_graph_stride():
    """The cuDNN scale descriptor matches the contiguous BHSD swizzle buffer."""

    class FakeTensor:
        def set_reordering_type(self, reordering):
            self.reordering = reordering
            return self

    class FakeGraph:
        def tensor(self, **kwargs):
            self.kwargs = kwargs
            return FakeTensor()

    class FakeCudnn:
        class data_type:
            FP8_E8M0 = "fp8_e8m0"

        class tensor_reordering:
            F8_128x4 = "f8_128x4"

    graph = FakeGraph()
    tensor = _mx_scale(
        graph,
        FakeCudnn,
        name="descale_q",
        uid=101,
        batch=2,
        heads=8,
        seqlen=128,
        dim=4,
    )

    assert graph.kwargs["dim"] == (2, 8, 128, 4)
    assert graph.kwargs["stride"] == (4096, 512, 4, 1)
    assert tensor.reordering == FakeCudnn.tensor_reordering.F8_128x4


@pytest.mark.parametrize("feature", ("alibi", "bottom_right"))
def test_fused_attention_parity_features(feature):
    """JAX executes ALiBi and explicit bottom-right diagonal attention."""

    _require_gpu(90, 90700)
    batch, heads, dim = 2, 8, 64
    q_seqlen, kv_seqlen = (128, 128) if feature == "alibi" else (64, 128)
    q_key, k_key, v_key, do_key = jax.random.split(jax.random.PRNGKey(4321), 4)
    q = jax.random.normal(q_key, (batch, q_seqlen, heads, dim), jnp.bfloat16) * 0.25
    k = jax.random.normal(k_key, (batch, kv_seqlen, heads, dim), jnp.bfloat16) * 0.25
    v = jax.random.normal(v_key, (batch, kv_seqlen, heads, dim), jnp.bfloat16) * 0.25
    doutput = (
        jax.random.normal(do_key, (batch, q_seqlen, heads, dim), jnp.bfloat16) * 0.25
    )
    q_lengths = jnp.full((batch,), q_seqlen, dtype=jnp.int32)
    kv_lengths = jnp.full((batch,), kv_seqlen, dtype=jnp.int32)
    descriptor = SequenceDescriptor.from_seqlens((q_lengths, kv_lengths))
    bias_type = AttnBiasType.ALIBI if feature == "alibi" else AttnBiasType.NO_BIAS
    helper = FusedAttnHelper(
        True,
        q.dtype,
        k.dtype,
        QKVLayout.BSHD_BSHD_BSHD,
        bias_type,
        AttnMaskType.CAUSAL_MASK,
        AttnSoftmaxType.VANILLA_SOFTMAX,
        0.0,
        heads,
        heads,
        q_seqlen,
        kv_seqlen,
        dim,
        dim,
        (-1, -1),
    )
    if not helper.is_fused_attn_kernel_available():
        pytest.skip("No fused-attention kernel supports this configuration.")

    def te_loss(query, key, value):
        output = fused_attn(
            (query, key, value),
            None,
            descriptor,
            None,
            bias_type,
            AttnMaskType.CAUSAL_MASK,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnSoftmaxType.VANILLA_SOFTMAX,
            1.0 / sqrt(dim),
            0.0,
            True,
            bottom_right_diagonal=feature == "bottom_right",
        )
        return jnp.sum(output.astype(jnp.float32) * doutput.astype(jnp.float32)), output

    def reference_loss(query, key, value):
        output = _reference_attention(
            query,
            key,
            value,
            bottom_right=feature == "bottom_right",
            alibi=feature == "alibi",
        )
        return jnp.sum(output.astype(jnp.float32) * doutput.astype(jnp.float32)), output

    (_, output), grads = jax.value_and_grad(te_loss, argnums=(0, 1, 2), has_aux=True)(
        q, k, v
    )
    (_, reference), reference_grads = jax.value_and_grad(
        reference_loss, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)
    np.testing.assert_allclose(output, reference, atol=0.02, rtol=0.02)
    for actual, expected in zip(grads, reference_grads):
        np.testing.assert_allclose(actual, expected, atol=0.03, rtol=0.03)
