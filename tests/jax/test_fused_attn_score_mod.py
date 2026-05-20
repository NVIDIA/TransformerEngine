# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for cuDNN frontend score_mod fused attention."""
from math import sqrt

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import value_and_grad

import transformer_engine.jax.cpp_extensions.attention as tex_attention
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    QKVLayout,
    fused_attn,
)
from transformer_engine.jax.cpp_extensions import make_fused_attn_score_mod_config
from transformer_engine.jax.flax import transformer as flax_transformer
from utils import assert_allclose


def _has_cudnn_frontend_python():
    try:
        tex_attention._import_cudnn_for_score_mod()
    except ImportError:
        return False
    return True


def _score_mod_causal(graph, score, tensors):
    cudnn = tex_attention._import_cudnn_for_score_mod()

    row_index = graph.gen_index(
        input=score,
        axis=2,
        compute_data_type=cudnn.data_type.INT32,
    )
    row_index.set_data_type(cudnn.data_type.INT32)
    col_index = graph.gen_index(
        input=score,
        axis=3,
        compute_data_type=cudnn.data_type.INT32,
    )
    col_index.set_data_type(cudnn.data_type.INT32)
    keep = graph.cmp_ge(
        input=row_index,
        comparison=col_index,
        compute_data_type=cudnn.data_type.BOOLEAN,
    )
    keep.set_data_type(cudnn.data_type.BOOLEAN)
    return graph.binary_select(input0=score, input1=tensors["neg_inf"], mask=keep)


def _score_mod_causal_bprop(graph, dscore, tensors):
    cudnn = tex_attention._import_cudnn_for_score_mod()

    row_index = graph.gen_index(
        input=dscore,
        axis=2,
        compute_data_type=cudnn.data_type.INT32,
    )
    row_index.set_data_type(cudnn.data_type.INT32)
    col_index = graph.gen_index(
        input=dscore,
        axis=3,
        compute_data_type=cudnn.data_type.INT32,
    )
    col_index.set_data_type(cudnn.data_type.INT32)
    keep = graph.cmp_ge(
        input=row_index,
        comparison=col_index,
        compute_data_type=cudnn.data_type.BOOLEAN,
    )
    keep.set_data_type(cudnn.data_type.BOOLEAN)
    return graph.binary_select(input0=dscore, input1=tensors["zero"], mask=keep)


def _score_mod_post_scale_bias(graph, score, tensors):
    cudnn = tex_attention._import_cudnn_for_score_mod()

    row_index = graph.gen_index(
        input=score,
        axis=2,
        compute_data_type=cudnn.data_type.INT32,
    )
    row_index.set_data_type(cudnn.data_type.INT32)
    col_index = graph.gen_index(
        input=score,
        axis=3,
        compute_data_type=cudnn.data_type.INT32,
    )
    col_index.set_data_type(cudnn.data_type.INT32)
    post_scale_bias = graph.sub(
        a=row_index,
        b=col_index,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    post_scale_bias.set_data_type(cudnn.data_type.FLOAT)
    return graph.add(
        a=score,
        b=post_scale_bias,
        compute_data_type=cudnn.data_type.FLOAT,
    )


class _ScoreModSoftcap:
    """cuDNN frontend score_mod implementing softcapping."""

    def __init__(self):
        self.before_tanh_activation = None

    def score_mod_graph_cache_key(self):
        """Graph topology key for softcap score_mod."""
        return ("softcap",)

    def forward(self, graph, score, tensors):
        cudnn = tex_attention._import_cudnn_for_score_mod()

        self.before_tanh_activation = graph.div(
            a=score,
            b=tensors["softcap"],
            compute_data_type=cudnn.data_type.FLOAT,
        )
        self.before_tanh_activation.set_data_type(cudnn.data_type.FLOAT)
        tanh_out = graph.tanh(input=self.before_tanh_activation)
        tanh_out.set_data_type(cudnn.data_type.FLOAT)
        return graph.mul(
            a=tanh_out,
            b=tensors["softcap"],
            compute_data_type=cudnn.data_type.FLOAT,
        )

    def backward(self, graph, dscore, tensors):
        cudnn = tex_attention._import_cudnn_for_score_mod()

        d_tanh_out = graph.mul(
            a=dscore,
            b=tensors["softcap"],
            compute_data_type=cudnn.data_type.FLOAT,
        )
        d_tanh_out.set_data_type(cudnn.data_type.FLOAT)
        d_before_tanh_activation = graph.tanh_backward(
            loss=d_tanh_out,
            input=self.before_tanh_activation,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        d_before_tanh_activation.set_data_type(cudnn.data_type.FLOAT)
        return graph.div(
            a=d_before_tanh_activation,
            b=tensors["softcap"],
            compute_data_type=cudnn.data_type.FLOAT,
        )


def _reference_attention(
    query, key, value, scale, *, causal=False, post_scale_bias=False, softcap=None
):
    scores = jnp.einsum("bqhd,bkhd->bhqk", query, key).astype(jnp.float32) * scale
    if causal:
        q_pos = jnp.arange(query.shape[1])[:, None]
        kv_pos = jnp.arange(key.shape[1])[None, :]
        scores = jnp.where(q_pos >= kv_pos, scores, -1e9)
    if post_scale_bias:
        q_pos = jnp.arange(query.shape[1], dtype=jnp.float32)[:, None]
        kv_pos = jnp.arange(key.shape[1], dtype=jnp.float32)[None, :]
        scores = scores + q_pos - kv_pos
    if softcap is not None:
        scores = softcap * jnp.tanh(scores / softcap)
    probs = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", probs, value).astype(query.dtype)


@pytest.fixture(autouse=True, scope="module")
def init():
    """
    WAR for CUDA uninitialize error
    """
    # Calling customcalls before jax may cause CUDA uninitialize error
    _ = jnp.zeros(0)
    yield


def _require_cudnn_frontend_score_mod():
    try:
        cudnn = tex_attention._import_cudnn_for_score_mod()
    except ImportError:
        pytest.skip("cuDNN Python frontend is required for score_mod")
    version = tuple(int(part) for part in cudnn.backend_version_string().split(".")[:2])
    if version < (9, 6):
        pytest.skip("cuDNN score_mod SDPA requires cuDNN frontend 9.6 or newer")


def _identity_score_mod(_graph, score, _tensors):
    return score


def _install_fake_flax_fused_attn(monkeypatch):
    captured = {}

    def fused_attn_kernel_check_should_not_run(*_args, **_kwargs):
        raise AssertionError("score_mod path should not use the standard fused-attn kernel check")

    def fake_fused_attn(
        qkv,
        bias,
        sequence_descriptor,
        seed,
        *,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        softmax_type,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq=1,
        window_size=None,
        context_parallel_strategy=None,
        context_parallel_causal_load_balanced=False,
        context_parallel_axis="",
        context_checkpoint_name="context",
        softmax_offset=None,
        stripe_size=None,
        score_mod=None,
        score_mod_bprop=None,
        score_mod_tensors=None,
        score_mod_bprop_tensors=None,
    ):
        captured.update(
            qkv=qkv,
            bias=bias,
            sequence_descriptor=sequence_descriptor,
            seed=seed,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            softmax_type=softmax_type,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
            max_segments_per_seq=max_segments_per_seq,
            window_size=window_size,
            context_parallel_strategy=context_parallel_strategy,
            context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
            context_parallel_axis=context_parallel_axis,
            context_checkpoint_name=context_checkpoint_name,
            softmax_offset=softmax_offset,
            stripe_size=stripe_size,
            score_mod=score_mod,
            score_mod_bprop=score_mod_bprop,
            score_mod_tensors=score_mod_tensors,
            score_mod_bprop_tensors=score_mod_bprop_tensors,
        )
        return qkv[0]

    monkeypatch.setattr(
        flax_transformer,
        "is_fused_attn_kernel_available",
        fused_attn_kernel_check_should_not_run,
    )
    monkeypatch.setattr(flax_transformer, "fused_attn", fake_fused_attn)
    return captured


def test_dot_product_attention_score_mod_requires_fused_attn(monkeypatch):
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    query = jnp.ones((1, 8, 1, 16), dtype=jnp.float16)
    key = jnp.ones((1, 8, 1, 16), dtype=jnp.float16)
    value = jnp.ones((1, 8, 1, 16), dtype=jnp.float16)

    dpa = flax_transformer.DotProductAttention(
        head_dim=16,
        num_attention_heads=1,
        num_gqa_groups=1,
        attn_mask_type="no_mask",
        qkv_layout="bshd_bshd_bshd",
        transpose_batch_sequence=False,
        score_mod=_identity_score_mod,
    )

    with pytest.raises(ValueError, match="score_mod requires fused attention"):
        dpa.apply({}, query, key, value, deterministic=True)


def test_dot_product_attention_plumbs_score_mod_to_fused_attn(monkeypatch):
    captured = _install_fake_flax_fused_attn(monkeypatch)
    query = jnp.ones((1, 8, 1, 16), dtype=jnp.float16)
    key = jnp.ones((1, 8, 1, 16), dtype=jnp.float16)
    value = jnp.ones((1, 8, 1, 16), dtype=jnp.float16)
    aux = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)

    dpa = flax_transformer.DotProductAttention(
        head_dim=16,
        num_attention_heads=1,
        num_gqa_groups=1,
        attn_mask_type="no_mask",
        qkv_layout="bshd_bshd_bshd",
        transpose_batch_sequence=False,
        score_mod=_identity_score_mod,
    )
    out = dpa.apply({}, query, key, value, deterministic=True, score_mod_tensors={"aux": aux})

    np.testing.assert_array_equal(out, query)
    assert captured["score_mod"] is _identity_score_mod
    assert captured["score_mod_tensors"]["aux"].shape == aux.shape
    assert captured["score_mod_bprop"] is None
    assert captured["score_mod_bprop_tensors"] is None
    assert captured["attn_mask_type"] is AttnMaskType.NO_MASK
    assert captured["attn_bias_type"] is AttnBiasType.NO_BIAS
    assert captured["qkv_layout"] is QKVLayout.BSHD_BSHD_BSHD
    assert captured["softmax_type"] is AttnSoftmaxType.VANILLA_SOFTMAX


def test_dot_product_attention_unpacks_packed_score_mod_to_separate_layout(monkeypatch):
    captured = _install_fake_flax_fused_attn(monkeypatch)
    qkv = jnp.ones((1, 8, 3, 1, 16), dtype=jnp.float16)

    dpa = flax_transformer.DotProductAttention(
        head_dim=16,
        num_attention_heads=1,
        num_gqa_groups=1,
        attn_mask_type="no_mask",
        qkv_layout="bs3hd",
        transpose_batch_sequence=False,
        score_mod=_identity_score_mod,
    )
    out = dpa.apply({}, qkv, None, None, deterministic=True)

    assert out.shape == (1, 8, 1, 16)
    assert len(captured["qkv"]) == 3
    assert captured["qkv"][0].shape == (1, 8, 1, 16)
    assert captured["qkv_layout"] is QKVLayout.BSHD_BSHD_BSHD
    assert captured["score_mod"] is _identity_score_mod


def test_multi_head_attention_plumbs_score_mod_to_dot_product_attention(monkeypatch):
    captured = _install_fake_flax_fused_attn(monkeypatch)

    class FakeLayerNormDenseGeneral:
        def __init__(self, *, features, return_layernorm_output=False, **_kwargs):
            self.features = features
            self.return_layernorm_output = return_layernorm_output

        def __call__(self, inputs):
            features = self.features if isinstance(self.features, tuple) else (self.features,)
            output = jnp.ones((*inputs.shape[:-1], *features), dtype=inputs.dtype)
            ln_out = inputs if self.return_layernorm_output else None
            return output, ln_out

    class FakeDenseGeneral:
        def __init__(self, *, features, **_kwargs):
            self.features = features

        def __call__(self, inputs):
            features = self.features if isinstance(self.features, tuple) else (self.features,)
            return jnp.ones((*inputs.shape[:-1], *features), dtype=inputs.dtype)

    monkeypatch.setattr(flax_transformer, "LayerNormDenseGeneral", FakeLayerNormDenseGeneral)
    monkeypatch.setattr(flax_transformer, "DenseGeneral", FakeDenseGeneral)

    inputs = jnp.ones((1, 8, 16), dtype=jnp.float16)
    aux = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)

    mha = flax_transformer.MultiHeadAttention(
        head_dim=16,
        num_attention_heads=1,
        num_gqa_groups=1,
        input_layernorm=False,
        attention_dropout=0.0,
        attn_mask_type="no_mask",
        fuse_qkv_params=True,
        transpose_batch_sequence=False,
        score_mod=_identity_score_mod,
    )

    variables = mha.init(
        jax.random.key(0),
        inputs,
        inputs,
        deterministic=True,
        score_mod_tensors={"aux": aux},
    )
    out, ln_out = mha.apply(
        variables,
        inputs,
        inputs,
        deterministic=True,
        score_mod_tensors={"aux": aux},
    )

    assert out.shape == inputs.shape
    assert ln_out is None
    assert len(captured["qkv"]) == 3
    assert captured["qkv_layout"] is QKVLayout.BSHD_BSHD_BSHD
    assert captured["score_mod"] is _identity_score_mod
    assert captured["score_mod_tensors"]["aux"].shape == aux.shape


def test_fused_attn_score_mod_validation_rejects_masks_without_cudnn_frontend():
    q = jax.ShapeDtypeStruct((1, 16, 1, 128), jnp.float16)
    k = jax.ShapeDtypeStruct((1, 16, 1, 128), jnp.float16)
    v = jax.ShapeDtypeStruct((1, 16, 1, 128), jnp.float16)

    with pytest.raises(ValueError, match="mutually exclusive with attention masks"):
        fused_attn(
            (q, k, v),
            None,
            None,
            None,
            AttnBiasType.NO_BIAS,
            AttnMaskType.CAUSAL_MASK,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnSoftmaxType.VANILLA_SOFTMAX,
            1.0,
            0.0,
            True,
            score_mod=_identity_score_mod,
        )


def test_fused_attn_score_mod_config_splits_tensors_and_pass_by_value_scalars():
    tensor = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)

    config, tensor_operands, bprop_tensor_operands = make_fused_attn_score_mod_config(
        _identity_score_mod,
        None,
        {"tensor": tensor, "neg_inf": -1e9},
        None,
        0.125,
        True,
    )

    assert config.score_mod_tensor_names == ("tensor",)
    assert len(tensor_operands) == 1
    assert tensor_operands[0].shape == (1, 1, 1, 1)
    assert len(bprop_tensor_operands) == 0
    assert len(config.score_mod_scalars) == 1
    assert config.score_mod_scalars[0].name == "neg_inf"
    assert config.score_mod_scalars[0].dtype == "float32"
    assert len(config.score_mod_scalars[0].value) == np.dtype(np.float32).itemsize


def test_fused_attn_score_mod_cudnn_frontend_version_check(monkeypatch):
    class FakeCudnn:
        __version__ = "1.22.0"

    monkeypatch.setattr(
        tex_attention.transformer_engine_jax,
        "get_cudnn_frontend_version",
        lambda: 12200,
    )
    assert tex_attention._check_cudnn_frontend_version_match(FakeCudnn) == 12200

    monkeypatch.setattr(
        tex_attention.transformer_engine_jax,
        "get_cudnn_frontend_version",
        lambda: 12100,
    )
    with pytest.raises(RuntimeError, match="Python/C\\+\\+ version mismatch"):
        tex_attention._check_cudnn_frontend_version_match(FakeCudnn)


def test_fused_attn_score_mod_config_stabilizes_bound_method_cache_keys():
    softcap_score_mod = _ScoreModSoftcap()
    first_forward = softcap_score_mod.forward
    second_forward = softcap_score_mod.forward
    first_backward = softcap_score_mod.backward
    second_backward = softcap_score_mod.backward

    assert first_forward is not second_forward
    assert first_backward is not second_backward

    config_1, _, _ = make_fused_attn_score_mod_config(
        first_forward,
        first_backward,
        {"softcap": 0.8},
        {"softcap": 0.8},
        0.125,
        True,
    )
    config_2, _, _ = make_fused_attn_score_mod_config(
        second_forward,
        second_backward,
        {"softcap": 0.8},
        {"softcap": 0.8},
        0.125,
        True,
    )
    other_softcap_score_mod = _ScoreModSoftcap()
    config_3, _, _ = make_fused_attn_score_mod_config(
        other_softcap_score_mod.forward,
        other_softcap_score_mod.backward,
        {"softcap": 0.8},
        {"softcap": 0.8},
        0.125,
        True,
    )

    assert config_1 == config_2
    assert hash(config_1) == hash(config_2)
    assert config_1 == config_3


def test_fused_attn_score_mod_config_leaves_unkeyed_bound_methods_uncached():
    class UnkeyedScoreMod:
        def forward(self, _graph, score, _tensors):
            return score

    score_mod = UnkeyedScoreMod()
    config_1, _, _ = make_fused_attn_score_mod_config(
        score_mod.forward, None, None, None, 0.125, True
    )
    config_2, _, _ = make_fused_attn_score_mod_config(
        score_mod.forward, None, None, None, 0.125, True
    )

    assert config_1 != config_2
    assert tex_attention._graph_cache_key("fwd", config_1, ()) is None


@pytest.mark.skipif(not _has_cudnn_frontend_python(), reason="cuDNN Python frontend is required")
def test_fused_attn_score_mod_post_scale_bias_optional_bprop():
    _require_cudnn_frontend_score_mod()

    key = jax.random.key(0)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = (0.125 * jax.random.normal(q_key, (1, 64, 2, 128), dtype=jnp.float16)).astype(jnp.float16)
    k = (0.125 * jax.random.normal(k_key, (1, 64, 2, 128), dtype=jnp.float16)).astype(jnp.float16)
    v = (0.125 * jax.random.normal(v_key, (1, 64, 2, 128), dtype=jnp.float16)).astype(jnp.float16)
    scale = 1.0 / sqrt(q.shape[-1])

    def score_mod_loss(query, key_, value):
        out = fused_attn(
            (query, key_, value),
            None,
            None,
            None,
            AttnBiasType.NO_BIAS,
            AttnMaskType.NO_MASK,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnSoftmaxType.VANILLA_SOFTMAX,
            scale,
            0.0,
            True,
            score_mod=_score_mod_post_scale_bias,
        )
        return jnp.sum(out.astype(jnp.float32)), out

    def ref_loss(query, key_, value):
        out = _reference_attention(query, key_, value, scale, post_scale_bias=True)
        return jnp.sum(out.astype(jnp.float32)), out

    (score_mod_value, score_mod_out), score_mod_grads = value_and_grad(
        score_mod_loss, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)
    (ref_value, ref_out), ref_grads = value_and_grad(ref_loss, argnums=(0, 1, 2), has_aux=True)(
        q, k, v
    )

    assert_allclose(score_mod_out, ref_out, rtol=5e-2, atol=5e-2)
    assert_allclose(score_mod_value, ref_value, rtol=5e-2, atol=5e-2)
    for grad, ref_grad in zip(score_mod_grads, ref_grads):
        assert_allclose(grad, ref_grad, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not _has_cudnn_frontend_python(), reason="cuDNN Python frontend is required")
def test_fused_attn_score_mod_causal_with_bprop():
    _require_cudnn_frontend_score_mod()

    key = jax.random.key(1)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = (0.125 * jax.random.normal(q_key, (1, 64, 2, 128), dtype=jnp.float16)).astype(jnp.float16)
    k = (0.125 * jax.random.normal(k_key, (1, 64, 2, 128), dtype=jnp.float16)).astype(jnp.float16)
    v = (0.125 * jax.random.normal(v_key, (1, 64, 2, 128), dtype=jnp.float16)).astype(jnp.float16)
    scale = 1.0 / sqrt(q.shape[-1])

    def score_mod_loss(query, key_, value):
        out = fused_attn(
            (query, key_, value),
            None,
            None,
            None,
            AttnBiasType.NO_BIAS,
            AttnMaskType.NO_MASK,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnSoftmaxType.VANILLA_SOFTMAX,
            scale,
            0.0,
            True,
            score_mod=_score_mod_causal,
            score_mod_bprop=_score_mod_causal_bprop,
            score_mod_tensors={"neg_inf": -1e9},
            score_mod_bprop_tensors={"zero": 0.0},
        )
        return jnp.sum(out.astype(jnp.float32)), out

    def ref_loss(query, key_, value):
        out = _reference_attention(query, key_, value, scale, causal=True)
        return jnp.sum(out.astype(jnp.float32)), out

    (score_mod_value, score_mod_out), score_mod_grads = value_and_grad(
        score_mod_loss, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)
    (ref_value, ref_out), ref_grads = value_and_grad(ref_loss, argnums=(0, 1, 2), has_aux=True)(
        q, k, v
    )

    assert_allclose(score_mod_out, ref_out, rtol=5e-2, atol=5e-2)
    assert_allclose(score_mod_value, ref_value, rtol=5e-2, atol=5e-2)
    for grad, ref_grad in zip(score_mod_grads, ref_grads):
        assert_allclose(grad, ref_grad, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not _has_cudnn_frontend_python(), reason="cuDNN Python frontend is required")
def test_fused_attn_score_mod_softcap_with_bprop():
    _require_cudnn_frontend_score_mod()

    key = jax.random.key(2)
    q_key, k_key, v_key, d_out_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (1, 16, 2, 64), dtype=jnp.float16)
    k = jax.random.normal(k_key, (1, 16, 2, 64), dtype=jnp.float16)
    v = (0.1 * jax.random.normal(v_key, (1, 16, 2, 64), dtype=jnp.float16)).astype(jnp.float16)
    d_out = jax.random.normal(d_out_key, (1, 16, 2, 64), dtype=jnp.float16)
    scale = 1.0 / sqrt(q.shape[-1])
    softcap = 0.8
    softcap_score_mod = _ScoreModSoftcap()

    def score_mod_loss(query, key_, value):
        out = fused_attn(
            (query, key_, value),
            None,
            None,
            None,
            AttnBiasType.NO_BIAS,
            AttnMaskType.NO_MASK,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnSoftmaxType.VANILLA_SOFTMAX,
            scale,
            0.0,
            True,
            score_mod=softcap_score_mod.forward,
            score_mod_bprop=softcap_score_mod.backward,
            score_mod_tensors={"softcap": softcap},
            score_mod_bprop_tensors={"softcap": softcap},
        )
        return jnp.sum(out.astype(jnp.float32) * d_out.astype(jnp.float32)), out

    def ref_loss(query, key_, value):
        out = _reference_attention(query, key_, value, scale, softcap=softcap)
        return jnp.sum(out.astype(jnp.float32) * d_out.astype(jnp.float32)), out

    (score_mod_value, score_mod_out), score_mod_grads = value_and_grad(
        score_mod_loss, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)
    (ref_value, ref_out), ref_grads = value_and_grad(ref_loss, argnums=(0, 1, 2), has_aux=True)(
        q, k, v
    )

    assert_allclose(score_mod_out, ref_out, rtol=7e-2, atol=7e-2)
    assert_allclose(score_mod_value, ref_value, rtol=7e-2, atol=7e-2)
    for grad, ref_grad in zip(score_mod_grads, ref_grads):
        assert_allclose(grad, ref_grad, rtol=7e-2, atol=7e-2)
