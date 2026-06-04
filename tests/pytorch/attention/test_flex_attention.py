# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import pathlib

import pytest
import torch

from transformer_engine.pytorch import DotProductAttention, is_bf16_available
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
import transformer_engine.pytorch.attention.dot_product_attention.flex_attention as flex_attention
from transformer_engine.pytorch.utils import get_cudnn_version, get_device_compute_capability

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent.parent)] + sys.path
from utils import (  # pylint: disable=wrong-import-position
    reset_rng_states,
    ModelConfig,
    get_available_attention_backends,
)

param_types = [torch.float16]
if torch.cuda.is_available() and is_bf16_available():
    param_types.append(torch.bfloat16)


def _score_mod_causal(score_mod_graph, score_tensor, tensors):
    """cuDNN frontend score_mod implementing top-left causal masking."""
    cudnn = flex_attention._import_cudnn_frontend()

    row_index = score_mod_graph.gen_index(input=score_tensor, axis=2)
    row_index.set_data_type(cudnn.data_type.INT32)
    col_index = score_mod_graph.gen_index(input=score_tensor, axis=3)
    col_index.set_data_type(cudnn.data_type.INT32)
    keep = score_mod_graph.cmp_ge(
        input=row_index,
        comparison=col_index,
        compute_data_type=cudnn.data_type.BOOLEAN,
    )
    keep.set_data_type(cudnn.data_type.BOOLEAN)
    return score_mod_graph.binary_select(
        input0=score_tensor,
        input1=tensors["neg_inf"],
        mask=keep,
    )


def _score_mod_causal_bprop(score_mod_graph, dP_tensor, tensors):
    """cuDNN frontend score_mod_bprop implementing top-left causal masking."""
    cudnn = flex_attention._import_cudnn_frontend()

    row_index = score_mod_graph.gen_index(input=dP_tensor, axis=2)
    row_index.set_data_type(cudnn.data_type.INT32)
    col_index = score_mod_graph.gen_index(input=dP_tensor, axis=3)
    col_index.set_data_type(cudnn.data_type.INT32)
    keep = score_mod_graph.cmp_ge(
        input=row_index,
        comparison=col_index,
        compute_data_type=cudnn.data_type.BOOLEAN,
    )
    keep.set_data_type(cudnn.data_type.BOOLEAN)
    return score_mod_graph.binary_select(
        input0=dP_tensor,
        input1=tensors["zero"],
        mask=keep,
    )


def _score_mod_post_scale_bias(score_mod_graph, score_tensor, _tensors):
    """cuDNN frontend score_mod adding post-scale bias."""
    cudnn = flex_attention._import_cudnn_frontend()

    row_index = score_mod_graph.gen_index(input=score_tensor, axis=2)
    row_index.set_data_type(cudnn.data_type.INT32)
    col_index = score_mod_graph.gen_index(input=score_tensor, axis=3)
    col_index.set_data_type(cudnn.data_type.INT32)
    post_scale_bias = score_mod_graph.sub(
        a=row_index,
        b=col_index,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    post_scale_bias.set_data_type(cudnn.data_type.FLOAT)
    return score_mod_graph.add(
        a=score_tensor,
        b=post_scale_bias,
        compute_data_type=cudnn.data_type.FLOAT,
    )


def _score_mod_identity_bprop(_score_mod_graph, dP_tensor, _tensors):
    """cuDNN frontend score_mod_bprop for score_mods with unit score derivative."""
    return dP_tensor


class _ScoreModSoftcap:
    """cuDNN frontend score_mod implementing softcapping."""

    def __init__(self):
        self.before_tanh_activation = None

    def score_mod_graph_cache_key(self):
        """Graph topology key for softcap score_mod."""
        return ("softcap",)

    def forward(self, score_mod_graph, score_tensor, tensors):
        """Apply softcap * tanh(score / softcap)."""
        cudnn = flex_attention._import_cudnn_frontend()

        self.before_tanh_activation = score_mod_graph.div(
            a=score_tensor,
            b=tensors["softcap"],
            compute_data_type=cudnn.data_type.FLOAT,
        )
        self.before_tanh_activation.set_data_type(cudnn.data_type.FLOAT)
        tanh_out = score_mod_graph.tanh(input=self.before_tanh_activation)
        tanh_out.set_data_type(cudnn.data_type.FLOAT)
        return score_mod_graph.mul(
            a=tanh_out,
            b=tensors["softcap"],
            compute_data_type=cudnn.data_type.FLOAT,
        )

    def backward(self, score_mod_graph, dP_tensor, tensors):
        """Apply softcap derivative to dP."""
        cudnn = flex_attention._import_cudnn_frontend()

        d_tanh_out = score_mod_graph.mul(
            a=dP_tensor,
            b=tensors["softcap"],
            compute_data_type=cudnn.data_type.FLOAT,
        )
        d_tanh_out.set_data_type(cudnn.data_type.FLOAT)
        d_before_tanh_activation = score_mod_graph.tanh_backward(
            loss=d_tanh_out,
            input=self.before_tanh_activation,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        d_before_tanh_activation.set_data_type(cudnn.data_type.FLOAT)
        return score_mod_graph.div(
            a=d_before_tanh_activation,
            b=tensors["softcap"],
            compute_data_type=cudnn.data_type.FLOAT,
        )


def _score_mod_cache_cpu_inputs():
    """Small CPU tensors for score_mod cache-key tests."""
    q = torch.empty((2, 4, 3, 8), dtype=torch.float16)
    k = torch.empty((2, 4, 3, 8), dtype=torch.float16)
    v = torch.empty((2, 4, 3, 8), dtype=torch.float16)
    o = torch.empty((2, 4, 3, 8), dtype=torch.float16)
    stats = torch.empty((2, 3, 4, 1), dtype=torch.float32)
    return q, k, v, o, stats


def test_score_mod_cache_bound_method_requires_explicit_key():
    """Unkeyed bound methods should be uncached instead of keyed by object id."""

    class UnkeyedScoreMod:
        def forward(self, _score_mod_graph, score_tensor, _tensors):
            return score_tensor

    key = flex_attention._score_mod_callback_cache_key(UnkeyedScoreMod().forward)

    assert key is flex_attention._SCORE_MOD_UNCACHEABLE


def test_score_mod_cache_bound_method_explicit_key_stable():
    """Bound method keys should be stable when a structural graph key is provided."""
    softcap = _ScoreModSoftcap()
    key_0 = flex_attention._score_mod_callback_cache_key(softcap.forward)
    key_1 = flex_attention._score_mod_callback_cache_key(softcap.forward)
    other_key = flex_attention._score_mod_callback_cache_key(_ScoreModSoftcap().forward)

    assert key_0 == key_1
    assert key_0 == other_key


def test_score_mod_cache_explicit_key_distinguishes_topology():
    """Stateful score_mods can opt into caching with topology-specific keys."""

    class LayeredScoreMod:
        def __init__(self, num_layers):
            self.num_layers = num_layers

        def score_mod_graph_cache_key(self):
            return {"num_layers": self.num_layers}

        def forward(self, _score_mod_graph, score_tensor, _tensors):
            return score_tensor

    key_0 = flex_attention._score_mod_callback_cache_key(LayeredScoreMod(1).forward)
    key_1 = flex_attention._score_mod_callback_cache_key(LayeredScoreMod(1).forward)
    key_2 = flex_attention._score_mod_callback_cache_key(LayeredScoreMod(2).forward)

    assert key_0 == key_1
    assert key_0 != key_2


def test_score_mod_cache_module_lambda_keys_do_not_collide():
    """Module-level lambdas should not reuse graphs only because qualnames match."""
    score_mod_0 = lambda _graph, score_tensor, _tensors: score_tensor  # noqa: E731
    score_mod_1 = lambda _graph, score_tensor, _tensors: score_tensor  # noqa: E731
    score_mod_0.__module__ = __name__
    score_mod_1.__module__ = __name__
    score_mod_0.__qualname__ = "<lambda>"
    score_mod_1.__qualname__ = "<lambda>"

    key_0 = flex_attention._score_mod_callback_cache_key(score_mod_0)
    key_1 = flex_attention._score_mod_callback_cache_key(score_mod_1)

    assert key_0 is not flex_attention._SCORE_MOD_UNCACHEABLE
    assert key_1 is not flex_attention._SCORE_MOD_UNCACHEABLE
    assert key_0 != key_1


def test_score_mod_cache_key_ignores_pass_by_value_values():
    """Scalar CPU tensor values are runtime inputs, not execution-plan metadata."""
    q, k, v, o, stats = _score_mod_cache_cpu_inputs()
    key_0 = flex_attention._cudnn_score_mod_fwd_cache_key(
        True,
        q,
        k,
        v,
        "bshd",
        "bshd",
        1.0,
        _score_mod_causal,
        {"softcap": torch.tensor(0.8, dtype=torch.float32)},
        o,
        stats,
    )
    key_1 = flex_attention._cudnn_score_mod_fwd_cache_key(
        True,
        q,
        k,
        v,
        "bshd",
        "bshd",
        1.0,
        _score_mod_causal,
        {"softcap": torch.tensor(1.2, dtype=torch.float32)},
        o,
        stats,
    )
    key_2 = flex_attention._cudnn_score_mod_fwd_cache_key(
        True,
        q,
        k,
        v,
        "bshd",
        "bshd",
        1.0,
        _score_mod_causal,
        {"softcap": torch.tensor([0.8], dtype=torch.float32)},
        o,
        stats,
    )

    assert key_0 == key_1
    assert key_0 != key_2


def test_score_mod_cache_fwd_reuses_graph_for_pass_by_value_changes(monkeypatch):
    """Fprop graph cache should reuse entries when only scalar CPU tensor values change."""
    q, k, v, o, stats = _score_mod_cache_cpu_inputs()
    cache = flex_attention._cudnn_score_mod_graph_cache
    saved_cache = dict(cache)
    build_entries = []

    def fake_build(
        is_training,
        query_layer,
        key_layer,
        value_layer,
        q_format,
        kv_format,
        attn_scale,
        score_mod,
        score_mod_tensors,
        output_layer,
        stats,
    ):
        del (
            is_training,
            query_layer,
            key_layer,
            value_layer,
            q_format,
            kv_format,
            attn_scale,
            score_mod,
            score_mod_tensors,
            output_layer,
            stats,
        )
        entry = object()
        build_entries.append(entry)
        return entry

    monkeypatch.setattr(flex_attention, "_build_cudnn_score_mod_fwd_graph", fake_build)
    try:
        cache.clear()
        entry_0 = flex_attention._get_cudnn_score_mod_fwd_graph(
            True,
            q,
            k,
            v,
            "bshd",
            "bshd",
            1.0,
            _score_mod_causal,
            {"softcap": torch.tensor(0.8, dtype=torch.float32)},
            o,
            stats,
        )
        entry_1 = flex_attention._get_cudnn_score_mod_fwd_graph(
            True,
            q,
            k,
            v,
            "bshd",
            "bshd",
            1.0,
            _score_mod_causal,
            {"softcap": torch.tensor(1.2, dtype=torch.float32)},
            o,
            stats,
        )
        entry_2 = flex_attention._get_cudnn_score_mod_fwd_graph(
            True,
            q,
            k,
            v,
            "bshd",
            "bshd",
            1.0,
            _score_mod_causal,
            {"softcap": torch.tensor([0.8], dtype=torch.float32)},
            o,
            stats,
        )
    finally:
        cache.clear()
        cache.update(saved_cache)

    assert entry_0 is entry_1
    assert entry_2 is not entry_0
    assert len(build_entries) == 2


def test_score_mod_cache_fwd_skips_cache_for_unkeyed_bound_method(monkeypatch):
    """Unkeyed bound methods should build fresh graphs instead of using an id-based key."""

    class UnkeyedScoreMod:
        def forward(self, _score_mod_graph, score_tensor, _tensors):
            return score_tensor

    q, k, v, o, stats = _score_mod_cache_cpu_inputs()
    score_mod = UnkeyedScoreMod()
    cache = flex_attention._cudnn_score_mod_graph_cache
    saved_cache = dict(cache)
    build_entries = []

    def fake_build(
        is_training,
        query_layer,
        key_layer,
        value_layer,
        q_format,
        kv_format,
        attn_scale,
        score_mod,
        score_mod_tensors,
        output_layer,
        stats,
    ):
        del (
            is_training,
            query_layer,
            key_layer,
            value_layer,
            q_format,
            kv_format,
            attn_scale,
            score_mod,
            score_mod_tensors,
            output_layer,
            stats,
        )
        entry = object()
        build_entries.append(entry)
        return entry

    monkeypatch.setattr(flex_attention, "_build_cudnn_score_mod_fwd_graph", fake_build)
    try:
        cache.clear()
        entry_0 = flex_attention._get_cudnn_score_mod_fwd_graph(
            True,
            q,
            k,
            v,
            "bshd",
            "bshd",
            1.0,
            score_mod.forward,
            None,
            o,
            stats,
        )
        entry_1 = flex_attention._get_cudnn_score_mod_fwd_graph(
            True,
            q,
            k,
            v,
            "bshd",
            "bshd",
            1.0,
            score_mod.forward,
            None,
            o,
            stats,
        )
        assert len(cache) == 0
    finally:
        cache.clear()
        cache.update(saved_cache)

    assert entry_0 is not entry_1
    assert len(build_entries) == 2


def test_score_mod_tensors_are_version_checked_for_backward(monkeypatch):
    """In-place score_mod tensor updates before backward should be rejected."""

    class FakeEntry:
        graph = object()
        q = object()
        k = object()
        v = object()
        output = object()
        stats = object()
        score_mod_graph_tensors = {"softcap": object()}
        workspace_size = 1

    def fake_execute(graph, variant_pack, workspace_size, device):
        del graph, variant_pack, workspace_size, device

    q, k, v, _, _ = _score_mod_cache_cpu_inputs()
    q = q.requires_grad_()
    k = k.requires_grad_()
    v = v.requires_grad_()
    softcap = torch.tensor(0.8, dtype=torch.float32)

    monkeypatch.setattr(flex_attention, "_get_cudnn_score_mod_fwd_graph", lambda *args: FakeEntry())
    monkeypatch.setattr(flex_attention, "_execute_cudnn_graph", fake_execute)

    out = flex_attention.FusedAttentionWithScoreModFunc.apply(
        True,
        q,
        k,
        v,
        "bshd",
        "bshd",
        1.0,
        _score_mod_causal,
        None,
        {"softcap": softcap},
        None,
        False,
    )
    softcap.add_(1.0)

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_score_mod_bprop_tensors_require_score_mod_bprop():
    """score_mod_bprop_tensors should not be silently ignored."""

    q, k, v, _, _ = (tensor.cuda() for tensor in _score_mod_cache_cpu_inputs())
    attention = DotProductAttention(
        num_attention_heads=3,
        kv_channels=8,
        qkv_format="bshd",
        attn_mask_type="no_mask",
    ).cuda()

    with pytest.raises(AssertionError, match="score_mod_bprop_tensors requires score_mod_bprop"):
        attention(
            q,
            k,
            v,
            qkv_format="bshd",
            attn_mask_type="no_mask",
            score_mod=_score_mod_causal,
            score_mod_bprop_tensors={"zero": torch.zeros((1, 1, 1, 1), device="cuda")},
        )


def _post_scale_bias(config, dtype):
    """Materialize score + (q_idx - kv_idx) as post-scale attention bias."""
    q_idx = torch.arange(config.max_seqlen_q, dtype=torch.float32, device="cuda").view(1, 1, -1, 1)
    kv_idx = torch.arange(config.max_seqlen_kv, dtype=torch.float32, device="cuda").view(
        1, 1, 1, -1
    )
    return (q_idx - kv_idx).to(dtype).expand(1, config.num_heads, -1, -1).contiguous()


def _to_bhsd(tensor, qkv_format):
    """Convert SBHD/BSHD test tensors to logical BHSD."""
    if qkv_format == "sbhd":
        return tensor.permute(1, 2, 0, 3)
    return tensor.permute(0, 2, 1, 3)


def _from_bhsd(tensor, qkv_format):
    """Convert logical BHSD test tensors to SBHD/BSHD."""
    if qkv_format == "sbhd":
        return tensor.permute(2, 0, 1, 3).contiguous()
    return tensor.permute(0, 2, 1, 3).contiguous()


def _pytorch_softcap_attention(q, k, v, qkv_format, softmax_scale, softcap):
    """PyTorch reference for softcapped scaled dot-product attention."""
    q_bhsd = _to_bhsd(q, qkv_format).float()
    k_bhsd = _to_bhsd(k, qkv_format).float()
    v_bhsd = _to_bhsd(v, qkv_format).float()
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * softmax_scale
    scores = softcap * torch.tanh(scores / softcap)
    probs = torch.softmax(scores, dim=-1)
    out = _from_bhsd(torch.matmul(probs, v_bhsd), qkv_format).to(v.dtype)
    return out.reshape(*out.shape[:-2], out.shape[-2] * out.shape[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.skipif(get_cudnn_version() < (9, 6, 0), reason="cuDNN 9.6.0+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("qkv_format", ["sbhd", "bshd"])
@pytest.mark.parametrize(
    "score_mod_case, scalar_loss",
    [
        ("causal", False),
        ("causal", True),
        ("softcap", False),
        ("post_scale_bias", False),
    ],
)
def test_dot_product_attention_score_mod(dtype, qkv_format, score_mod_case, scalar_loss):
    """Compare score_mod attention against equivalent reference implementations."""
    try:
        flex_attention._import_cudnn_frontend()
    except ImportError:
        pytest.skip("cuDNN frontend Python package is required for score_mod attention.")
    if score_mod_case == "softcap" and get_device_compute_capability() < (9, 0):
        pytest.skip("Softcap score_mod tests require sm90+.")

    reset_rng_states()

    config = ModelConfig(
        2,
        64 if score_mod_case == "causal" else 16,
        4,
        64,
        attn_mask_type="no_mask",
    )
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout=f"{qkv_format}_{qkv_format}_{qkv_format}",
        score_mod=True,
        score_mod_bprop=True,
    )
    if not available_backends[1] or not fused_attn_backends:
        pytest.skip("FusedAttention is not available for this score_mod configuration.")

    if score_mod_case == "post_scale_bias":
        bias_config = ModelConfig(
            config.batch_size,
            config.max_seqlen_q,
            config.num_heads,
            config.head_dim_qk,
            attn_mask_type="no_mask",
            attn_bias_type="post_scale_bias",
            bias_shape="1hss",
        )
        bias_available_backends, _, bias_fused_attn_backends = get_available_attention_backends(
            bias_config,
            qkv_dtype=dtype,
            qkv_layout=f"{qkv_format}_{qkv_format}_{qkv_format}",
        )
        if not bias_available_backends[1] or not bias_fused_attn_backends:
            pytest.skip("FusedAttention is not available for post_scale_bias reference.")

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    _attention_backends["backend_selection_requires_update"] = True

    if qkv_format == "sbhd":
        q_shape = (config.max_seqlen_q, config.batch_size, config.num_heads, config.head_dim_qk)
        kv_shape = q_shape
    else:
        q_shape = (config.batch_size, config.max_seqlen_q, config.num_heads, config.head_dim_qk)
        kv_shape = q_shape

    if score_mod_case == "softcap":
        q = torch.randn(q_shape, dtype=dtype, device="cuda").requires_grad_()
        k = torch.randn(kv_shape, dtype=dtype, device="cuda").requires_grad_()
        v = (0.1 * torch.randn(kv_shape, dtype=dtype, device="cuda")).requires_grad_()
    else:
        q = (0.1 * torch.randn(q_shape, dtype=dtype, device="cuda")).requires_grad_()
        k = (0.1 * torch.randn(kv_shape, dtype=dtype, device="cuda")).requires_grad_()
        v = (0.1 * torch.randn(kv_shape, dtype=dtype, device="cuda")).requires_grad_()
    q_ref, k_ref, v_ref = [x.detach().clone().requires_grad_() for x in (q, k, v)]

    flex_attn = DotProductAttention(
        config.num_heads,
        config.head_dim_qk,
        qkv_format=qkv_format,
        attn_mask_type="no_mask",
        layer_number=1,
    ).to(dtype=dtype, device="cuda")

    if score_mod_case == "causal":
        score_mod_kwargs = {
            "score_mod": _score_mod_causal,
            "score_mod_bprop": _score_mod_causal_bprop,
            "score_mod_tensors": {"neg_inf": torch.full((1, 1, 1, 1), -1e9)},
            "score_mod_bprop_tensors": {"zero": torch.full((1, 1, 1, 1), 0.0)},
        }
        ref_attn = DotProductAttention(
            config.num_heads,
            config.head_dim_qk,
            qkv_format=qkv_format,
            attn_mask_type="causal",
            layer_number=1,
        ).to(dtype=dtype, device="cuda")
        out_ref = ref_attn(q_ref, k_ref, v_ref, qkv_format=qkv_format, attn_mask_type="causal")
        tols = dict(atol=5e-2, rtol=5e-2)
    elif score_mod_case == "softcap":
        softcap = 0.8
        softcap_tensor = torch.full((1, 1, 1, 1), softcap)
        softcap_score_mod = _ScoreModSoftcap()
        score_mod_kwargs = {
            "score_mod": softcap_score_mod.forward,
            "score_mod_bprop": softcap_score_mod.backward,
            "score_mod_tensors": {"softcap": softcap_tensor},
            "score_mod_bprop_tensors": {"softcap": softcap_tensor},
        }
        out_ref = _pytorch_softcap_attention(
            q_ref,
            k_ref,
            v_ref,
            qkv_format,
            1.0 / config.head_dim_qk**0.5,
            softcap,
        )
        tols = dict(atol=7e-2, rtol=7e-2)
    else:
        assert score_mod_case == "post_scale_bias"
        score_mod_kwargs = {
            "score_mod": _score_mod_post_scale_bias,
            "score_mod_bprop": _score_mod_identity_bprop,
        }
        ref_attn = DotProductAttention(
            config.num_heads,
            config.head_dim_qk,
            qkv_format=qkv_format,
            attn_mask_type="no_mask",
            layer_number=1,
        ).to(dtype=dtype, device="cuda")
        out_ref = ref_attn(
            q_ref,
            k_ref,
            v_ref,
            qkv_format=qkv_format,
            attn_mask_type="no_mask",
            core_attention_bias_type="post_scale_bias",
            core_attention_bias=_post_scale_bias(config, dtype),
        )
        tols = dict(atol=5e-2, rtol=5e-2)

    out = flex_attn(
        q,
        k,
        v,
        qkv_format=qkv_format,
        attn_mask_type="no_mask",
        **score_mod_kwargs,
    )

    if scalar_loss:
        out.sum().backward()
        out_ref.sum().backward()
    else:
        d_out = torch.randn_like(out)
        out.backward(d_out)
        out_ref.backward(d_out)

    torch.testing.assert_close(out, out_ref, **tols)
    torch.testing.assert_close(q.grad, q_ref.grad, **tols)
    torch.testing.assert_close(k.grad, k_ref.grad, **tols)
    torch.testing.assert_close(v.grad, v_ref.grad, **tols)
