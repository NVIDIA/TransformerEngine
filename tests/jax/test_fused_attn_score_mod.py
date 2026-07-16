# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for cuDNN frontend score_mod fused attention."""
from math import sqrt

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import transformer_engine.jax.cpp_extensions.flex_attention as tex_attention
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    QKVLayout,
)
from transformer_engine.jax.cpp_extensions import make_fused_attn_score_mod_config
from transformer_engine.jax.flax import transformer as flax_transformer
from transformer_engine_jax import get_device_compute_capability, NVTE_Fused_Attn_Backend
from test_fused_attn import FusedAttnRunner, SeqDescFormat


_CONFIG_TEST_HEAD_DIM = 128
_CONFIG_TEST_SCALING_FACTOR = 1.0 / sqrt(_CONFIG_TEST_HEAD_DIM)
_SCORE_MOD_MIN_CUDNN_VERSION = (9, 23)
_SCORE_MOD_MIN_CUDNN_VERSION_STRING = ".".join(map(str, _SCORE_MOD_MIN_CUDNN_VERSION))


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


def _reference_score_mod_causal(scores):
    q_pos = jnp.arange(scores.shape[-2])[:, None]
    kv_pos = jnp.arange(scores.shape[-1])[None, :]
    return jnp.where(q_pos >= kv_pos, scores, -1e9)


def _reference_score_mod_post_scale_bias(scores):
    q_pos = jnp.arange(scores.shape[-2], dtype=jnp.float32)[:, None]
    kv_pos = jnp.arange(scores.shape[-1], dtype=jnp.float32)[None, :]
    return scores + q_pos - kv_pos


def _reference_score_mod_softcap(softcap):
    def score_mod(scores):
        return softcap * jnp.tanh(scores / softcap)

    return score_mod


class ScoreModFusedAttnRunner(FusedAttnRunner):
    """FusedAttnRunner configured for score_mod tests."""

    ATTN_BIAS_TYPE = AttnBiasType.NO_BIAS
    ATTN_MASK_TYPE = AttnMaskType.NO_MASK
    SOFTMAX_TYPE = AttnSoftmaxType.VANILLA_SOFTMAX
    DROPOUT_PROBABILITY = 0.0
    IS_TRAINING = True
    QKV_LAYOUT = QKVLayout.BSHD_BSHD_BSHD
    BIAS_SHAPE = None
    WINDOW_SIZE = None
    SEQ_DESC_FORMAT = SeqDescFormat.Mask
    CP_LOAD_BALANCED = False
    DOUTPUT_SEED = None
    RTOL = 5e-2
    ATOL = 5e-2

    @staticmethod
    def require_cudnn_frontend():
        """Skip unless cuDNN frontend supports score_mod SDPA."""
        try:
            cudnn = tex_attention._import_cudnn_for_score_mod()
        except ImportError:
            pytest.skip("cuDNN Python frontend is required for score_mod")
        version = tuple(int(part) for part in cudnn.backend_version_string().split(".")[:2])
        if version < _SCORE_MOD_MIN_CUDNN_VERSION:
            pytest.skip(
                "cuDNN frontend score_mod SDPA requires "
                f"cuDNN {_SCORE_MOD_MIN_CUDNN_VERSION_STRING} or newer"
            )

    @staticmethod
    def _compute_input_scale(head_dim):
        return 1.0 / sqrt(head_dim)

    @classmethod
    def generic(
        cls,
        batch,
        seqlen,
        num_heads,
        head_dim,
        dtype,
        *,
        score_mod,
        score_mod_reference,
        score_mod_bprop=None,
        score_mod_tensors=None,
        score_mod_bprop_tensors=None,
        doutput_seed=None,
        rtol=None,
        atol=None,
        number_of_devices=1,
        mesh_shape=(1, 1, 1),
        mesh_axes=("dp", "cp", "tp"),
        mesh_resource=None,
    ):
        """Build a runner for a separate-Q/K/V score_mod fused-attention case."""
        kwargs = {}
        if mesh_resource is not None:
            kwargs["mesh_resource"] = mesh_resource
        if doutput_seed is None:
            doutput_seed = cls.DOUTPUT_SEED
        if rtol is None:
            rtol = cls.RTOL
        if atol is None:
            atol = cls.ATOL
        return cls(
            batch,
            seqlen,
            seqlen,
            num_heads,
            num_heads,
            head_dim,
            head_dim,
            cls.ATTN_BIAS_TYPE,
            cls.ATTN_MASK_TYPE,
            cls.SOFTMAX_TYPE,
            cls.DROPOUT_PROBABILITY,
            dtype,
            cls.IS_TRAINING,
            cls.QKV_LAYOUT,
            cls.BIAS_SHAPE,
            cls.WINDOW_SIZE,
            cls.SEQ_DESC_FORMAT,
            number_of_devices=number_of_devices,
            mesh_shape=mesh_shape,
            mesh_axes=mesh_axes,
            cp_load_balanced=cls.CP_LOAD_BALANCED,
            score_mod=score_mod,
            score_mod_bprop=score_mod_bprop,
            score_mod_tensors=score_mod_tensors,
            score_mod_bprop_tensors=score_mod_bprop_tensors,
            score_mod_reference=score_mod_reference,
            input_scale=cls._compute_input_scale(head_dim),
            doutput_seed=doutput_seed,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    @classmethod
    def softcap(
        cls,
        batch,
        seqlen,
        num_heads,
        head_dim,
        dtype,
        *,
        number_of_devices=1,
        mesh_shape=(1, 1, 1),
        mesh_axes=("dp", "cp", "tp"),
        mesh_resource=None,
    ):
        """Build a runner for softcap score modification with explicit bprop."""
        softcap = 0.8
        softcap_score_mod = _ScoreModSoftcap()
        return cls.generic(
            batch,
            seqlen,
            num_heads,
            head_dim,
            dtype,
            number_of_devices=number_of_devices,
            mesh_shape=mesh_shape,
            mesh_axes=mesh_axes,
            mesh_resource=mesh_resource,
            score_mod=softcap_score_mod.forward,
            score_mod_bprop=softcap_score_mod.backward,
            score_mod_tensors={"softcap": softcap},
            score_mod_bprop_tensors={"softcap": softcap},
            score_mod_reference=_reference_score_mod_softcap(softcap),
            doutput_seed=2025,
            rtol=7e-2,
            atol=7e-2,
        )

    @classmethod
    def post_scale_bias(
        cls,
        batch,
        seqlen,
        num_heads,
        head_dim,
        dtype,
        *,
        number_of_devices=1,
        mesh_shape=(1, 1, 1),
        mesh_axes=("dp", "cp", "tp"),
        mesh_resource=None,
    ):
        """Build a runner for post-scale-bias score modification without explicit bprop."""
        return cls.generic(
            batch,
            seqlen,
            num_heads,
            head_dim,
            dtype,
            number_of_devices=number_of_devices,
            mesh_shape=mesh_shape,
            mesh_axes=mesh_axes,
            mesh_resource=mesh_resource,
            score_mod=_score_mod_post_scale_bias,
            score_mod_reference=_reference_score_mod_post_scale_bias,
        )

    @classmethod
    def causal(
        cls,
        batch,
        seqlen,
        num_heads,
        head_dim,
        dtype,
        *,
        number_of_devices=1,
        mesh_shape=(1, 1, 1),
        mesh_axes=("dp", "cp", "tp"),
        mesh_resource=None,
    ):
        """Build a runner for causal score modification with explicit bprop."""
        return cls.generic(
            batch,
            seqlen,
            num_heads,
            head_dim,
            dtype,
            number_of_devices=number_of_devices,
            mesh_shape=mesh_shape,
            mesh_axes=mesh_axes,
            mesh_resource=mesh_resource,
            score_mod=_score_mod_causal,
            score_mod_bprop=_score_mod_causal_bprop,
            score_mod_tensors={"neg_inf": -1e9},
            score_mod_bprop_tensors={"zero": 0.0},
            score_mod_reference=_reference_score_mod_causal,
        )


@pytest.fixture(autouse=True, scope="module")
def init():
    """
    WAR for CUDA uninitialize error
    """
    # Calling customcalls before jax may cause CUDA uninitialize error
    _ = jnp.zeros(0)
    yield


def _identity_score_mod(_graph, score, _tensors):
    return score


def _install_fake_flax_fused_attn(monkeypatch, *, kernel_available=True):
    captured = {}

    class FakeFusedAttnHelper:
        def __init__(self, *args, **kwargs):
            captured.setdefault("kernel_checks", []).append((args, kwargs))

        def get_fused_attn_backend(self):
            if kernel_available:
                return NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen, ""
            return NVTE_Fused_Attn_Backend.NVTE_No_Backend, "fake: no backend"

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

    monkeypatch.setattr(flax_transformer, "FusedAttnHelper", FakeFusedAttnHelper)
    monkeypatch.setattr(flax_transformer, "fused_attn", fake_fused_attn)
    return captured


def test_dot_product_attention_score_mod_requires_fused_attn(monkeypatch):
    """DotProductAttention rejects score_mod when fused attention is disabled."""
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


def test_dot_product_attention_score_mod_requires_available_fused_attn_kernel(monkeypatch):
    """DotProductAttention rejects score_mod when no fused attention kernel is available."""
    _install_fake_flax_fused_attn(monkeypatch, kernel_available=False)
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

    with pytest.raises(ValueError, match="no fused attention kernel is available"):
        dpa.apply({}, query, key, value, deterministic=True)


def test_dot_product_attention_plumbs_score_mod_to_fused_attn(monkeypatch):
    """DotProductAttention forwards score_mod operands to fused_attn."""
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
    assert captured["kernel_checks"][0][0][3] is QKVLayout.BSHD_BSHD_BSHD


def test_dot_product_attention_unpacks_packed_score_mod_to_separate_layout(monkeypatch):
    """Packed QKV inputs are unpacked because score_mod requires separate Q/K/V."""
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
    assert captured["kernel_checks"][0][0][3] is QKVLayout.BSHD_BSHD_BSHD


def test_multi_head_attention_plumbs_score_mod_to_dot_product_attention(monkeypatch):
    """MultiHeadAttention passes score_mod tensors through its attention stack."""
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


def test_fused_attn_score_mod_config_splits_tensors_and_pass_by_value_scalars():
    """Score-mod config separates array operands from pass-by-value scalars."""
    tensor = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)

    config, tensor_operands, bprop_tensor_operands = make_fused_attn_score_mod_config(
        _identity_score_mod,
        None,
        {"tensor": tensor, "neg_inf": -1e9},
        None,
        _CONFIG_TEST_SCALING_FACTOR,
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
    """cuDNN frontend Python and C++ versions must match."""

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
    """Bound score_mod methods with graph keys share stable cache keys."""
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
        _CONFIG_TEST_SCALING_FACTOR,
        True,
    )
    config_2, _, _ = make_fused_attn_score_mod_config(
        second_forward,
        second_backward,
        {"softcap": 0.8},
        {"softcap": 0.8},
        _CONFIG_TEST_SCALING_FACTOR,
        True,
    )
    other_softcap_score_mod = _ScoreModSoftcap()
    config_3, _, _ = make_fused_attn_score_mod_config(
        other_softcap_score_mod.forward,
        other_softcap_score_mod.backward,
        {"softcap": 0.8},
        {"softcap": 0.8},
        _CONFIG_TEST_SCALING_FACTOR,
        True,
    )

    assert config_1 == config_2
    assert hash(config_1) == hash(config_2)
    assert config_1 == config_3


def test_fused_attn_score_mod_config_leaves_unkeyed_bound_methods_uncached():
    """Unkeyed bound methods stay uncached to avoid object-id reuse collisions."""

    class UnkeyedScoreMod:
        def forward(self, _graph, score, _tensors):
            return score

    score_mod = UnkeyedScoreMod()
    config_1, _, _ = make_fused_attn_score_mod_config(
        score_mod.forward, None, None, None, _CONFIG_TEST_SCALING_FACTOR, True
    )
    config_2, _, _ = make_fused_attn_score_mod_config(
        score_mod.forward, None, None, None, _CONFIG_TEST_SCALING_FACTOR, True
    )

    assert config_1 != config_2
    assert tex_attention._graph_cache_key("fwd", config_1, ()) is None


@pytest.mark.skipif(not _has_cudnn_frontend_python(), reason="cuDNN Python frontend is required")
def test_fused_attn_score_mod_post_scale_bias_optional_bprop():
    """Post-scale-bias score_mod matches the JAX reference without explicit bprop."""
    ScoreModFusedAttnRunner.require_cudnn_frontend()
    runner = ScoreModFusedAttnRunner.post_scale_bias(1, 64, 2, 128, jnp.float16)
    runner.test_forward()
    runner.test_backward()


@pytest.mark.skipif(not _has_cudnn_frontend_python(), reason="cuDNN Python frontend is required")
def test_fused_attn_score_mod_causal_with_bprop():
    """Causal score_mod matches the JAX reference with explicit bprop."""
    ScoreModFusedAttnRunner.require_cudnn_frontend()
    runner = ScoreModFusedAttnRunner.causal(1, 64, 2, 128, jnp.float16)
    runner.test_forward()
    runner.test_backward()


@pytest.mark.skipif(not _has_cudnn_frontend_python(), reason="cuDNN Python frontend is required")
@pytest.mark.skipif(
    get_device_compute_capability(0) < 90,
    reason="Softcap score_mod tests require sm90+",
)
def test_fused_attn_score_mod_softcap_with_bprop():
    """Softcap score_mod matches the JAX reference with explicit bprop."""
    ScoreModFusedAttnRunner.require_cudnn_frontend()
    runner = ScoreModFusedAttnRunner.softcap(1, 16, 2, 64, jnp.float16)
    runner.test_backward()
