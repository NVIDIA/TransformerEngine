# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import pathlib
from types import FunctionType, SimpleNamespace
from unittest.mock import Mock, call
from concurrent.futures import ThreadPoolExecutor
import time

import pytest
import torch

from transformer_engine.common.cudnn.attention import (
    AttentionOptions,
    AttentionGraphSpec,
    BuiltAttentionGraph,
)
from transformer_engine.common.cudnn.graph import GraphCache, GraphCacheEntry
from transformer_engine.pytorch import DotProductAttention, is_bf16_available
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
import transformer_engine.pytorch.attention.dot_product_attention.flex_attention as flex_attention
from transformer_engine.pytorch.cpp_extensions import cudnn as cudnn_graph
from transformer_engine.pytorch.utils import get_device_compute_capability

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


@pytest.mark.parametrize("has_score_mod", (False, True))
@pytest.mark.parametrize("selected_engine", (None, "sdpa_fwd", "sdpa_bwd", "missing"))
def test_cudnn_score_mod_backend_guard(monkeypatch, has_score_mod, selected_engine):
    """Exclude Python engines before planning and reject them if they are still selected."""
    cudnn = Mock(__version__="1.31.0")
    graph = Mock(selected_engine=selected_engine)
    if selected_engine == "missing":
        del graph.selected_engine
    graph.get_workspace_size.return_value = 0
    monkeypatch.setattr(cudnn_graph, "_import_cudnn_frontend", lambda: cudnn)

    if has_score_mod and selected_engine in ("sdpa_fwd", "sdpa_bwd"):
        with pytest.raises(RuntimeError, match="Only the cuDNN backend implements score_mod"):
            cudnn_graph._finalize_cudnn_graph(graph, has_score_mod=has_score_mod)
        graph.get_workspace_size.assert_not_called()
    else:
        assert cudnn_graph._finalize_cudnn_graph(graph, has_score_mod=has_score_mod) == 1

    if has_score_mod:
        exclusion = call.deselect_engines(["sdpa_fwd", "sdpa_bwd"])
        graph.deselect_engines.assert_called_once_with(["sdpa_fwd", "sdpa_bwd"])
        assert (
            graph.mock_calls.index(call.build_operation_graph())
            < graph.mock_calls.index(exclusion)
            < graph.mock_calls.index(call.create_execution_plans([cudnn.heur_mode.A]))
        )
    else:
        graph.deselect_engines.assert_not_called()


@pytest.mark.parametrize("serialize", (False, True))
def test_cudnn_graph_cache_compiles_once_across_threads(serialize):
    """Concurrent requests share one supported graph and one completed compilation."""
    cache = GraphCache()
    graph = Mock()
    graph.serialize.return_value = b"graph"

    def build():
        time.sleep(0.01)
        return GraphCacheEntry(BuiltAttentionGraph(graph, {}, {}))

    def compile_plan():
        time.sleep(0.01)
        return 64

    build = Mock(side_effect=build)
    compile_plan = Mock(side_effect=compile_plan)

    def request(_index):
        entry = cache.get("graph", build)
        assert cache.compile(entry, compile_plan, serialize=serialize) == 64
        return entry

    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(request, range(8)))
    assert all(entry is entries[0] for entry in entries)
    assert build.call_count == compile_plan.call_count == 1
    assert graph.serialize.call_count == int(serialize)


@pytest.mark.parametrize("failure", ("support", "compile", "serialize"))
def test_cudnn_graph_cache_retries_failed_build(failure):
    """Failures must not publish an incomplete cache entry or compilation state."""
    cache = GraphCache()
    graph = Mock()
    graph.serialize.return_value = b"graph"
    entry = GraphCacheEntry(BuiltAttentionGraph(graph, {}, {}))
    build = Mock(side_effect=[RuntimeError("failure"), entry] if failure == "support" else [entry])
    compile_plan = Mock(
        side_effect=[RuntimeError("failure"), 64] if failure == "compile" else [64, 64]
    )
    if failure == "serialize":
        graph.serialize.side_effect = [RuntimeError("failure"), b"graph"]
    with pytest.raises(RuntimeError, match="failure"):
        cached = cache.get("graph", build)
        cache.compile(cached, compile_plan, serialize=True)
    assert entry.workspace_size is entry.serialized_graph is None
    assert bool(cache.entries) == (failure != "support")
    cached = cache.get("graph", build)
    assert cache.compile(cached, compile_plan, serialize=True) == 64
    assert cached.serialized_graph == b"graph"


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


def _score_mod_with_defaults(graph, score, _tensors, mode=False, *, options=None):
    if options is not None:
        mode = options["modes"][0]
    return graph.tanh(input=score) if mode else score


def _clone_score_mod(default=False, options=None):
    callback = FunctionType(_score_mod_with_defaults.__code__, globals(), argdefs=(default,))
    callback.__kwdefaults__ = {"options": options}
    return callback


def _mock_score_mod_build(monkeypatch):
    """Replace cuDNN construction while exercising the real cache and bindings."""
    monkeypatch.setattr(cudnn_graph._graph_cache, "entries", {})

    def build(spec, _device, _callback, _bprop, extra, bprop_extra):
        handles = flex_attention._score_mod_bindings(
            {name: object() for name, _ in spec.tensors},
            {name: object() for name in extra},
            {name: object() for name in bprop_extra},
        )
        return GraphCacheEntry(BuiltAttentionGraph(object(), handles, {}))

    build = Mock(side_effect=build)
    monkeypatch.setattr(flex_attention, "_build_cudnn_score_mod_graph", build)
    monkeypatch.setattr(flex_attention, "compile_graph", Mock(return_value=1))
    return build


def _get_score_mod_test_graph(callback, *, backward=False, extra=None):
    q, k, v, out, stats = _score_mod_cache_cpu_inputs()
    tensors = {"q": q, "k": k, "v": v, "o": out, "stats": stats}
    if backward:
        tensors.update(do=out, dq=q, dk=k, dv=v)
    entry, _ = flex_attention._get_cudnn_score_mod_graph(
        AttentionOptions(backward=backward, attn_scale=1.0),
        tensors,
        "bshd",
        "bshd",
        _score_mod_identity_bprop if backward else callback,
        callback if backward else None,
        score_mod_tensors=extra,
    )
    return entry


@pytest.mark.parametrize("direction", ("fwd", "bwd"))
@pytest.mark.parametrize("difference", ("positional", "keyword", "code", "type"))
def test_score_mod_cache_separates_function_defaults(monkeypatch, direction, difference):
    """Different code or defaults must build separate forward and backward graphs."""
    first = _clone_score_mod()
    second = _clone_score_mod(True)
    if difference == "keyword":
        second = _clone_score_mod(options={"modes": [True]})
    elif difference == "code":
        second = FunctionType(_score_mod_identity_bprop.__code__, globals())
        second.__qualname__ = first.__qualname__
    elif difference == "type":
        second = _clone_score_mod(0)
    build = _mock_score_mod_build(monkeypatch)

    def get(callback):
        return _get_score_mod_test_graph(callback, backward=direction == "bwd")

    entry = get(first)
    assert get(first) is entry
    other = get(second)
    assert other is not entry
    assert get(second) is other
    assert build.call_count == 2

    # Keys snapshot nested defaults rather than retaining mutable references.
    first.__kwdefaults__["options"] = {"modes": [False]}
    changed = get(first)
    assert changed is not entry
    first.__kwdefaults__["options"]["modes"].append(True)
    assert get(first) is not changed
    assert build.call_count == 4


@pytest.mark.parametrize("kind", ("tensor", "opaque", "cycle"))
def test_score_mod_cache_unsafe_defaults_are_uncached(kind):
    """Opaque state, runtime tensors and recursive defaults cannot imply a stable graph."""
    if kind == "tensor":
        default = torch.tensor(1.0)
    elif kind == "opaque":
        default = object()
    else:
        default = []
        default.append(default)
    callback = _clone_score_mod(default)
    key = flex_attention._score_mod_callback_cache_key(callback)
    assert key is flex_attention._SCORE_MOD_UNCACHEABLE

    # An explicit structural key remains the caller's opt-in caching contract.
    callback.score_mod_graph_cache_key = lambda: ("explicit-mode", True)
    assert flex_attention._score_mod_callback_cache_key(callback) is not key


def test_score_mod_cache_snapshots_positional_defaults():
    """Changing positional defaults invalidates the old key while unchanged NaNs stay stable."""
    callback = _clone_score_mod(False)
    before = flex_attention._score_mod_callback_cache_key(callback)
    callback.__defaults__ = (True,)
    assert flex_attention._score_mod_callback_cache_key(callback) != before
    callback.__defaults__ = (float("nan"),)
    assert flex_attention._score_mod_callback_cache_key(callback) == (
        flex_attention._score_mod_callback_cache_key(callback)
    )
    callback.__defaults__ = (0.0,)
    positive_zero = flex_attention._score_mod_callback_cache_key(callback)
    callback.__defaults__ = (-0.0,)
    assert flex_attention._score_mod_callback_cache_key(callback) != positive_zero


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
    spec = AttentionGraphSpec.from_dict(AttentionOptions(), {})

    def key(tensor):
        return flex_attention._cudnn_score_mod_cache_key(
            spec, torch.device("cpu"), _score_mod_causal, None, {"softcap": tensor}, {}
        )

    assert key(torch.tensor(0.8)) == key(torch.tensor(1.2))
    assert key(torch.tensor(0.8)) != key(torch.tensor([0.8]))


def test_score_mod_cache_fwd_reuses_graph_for_pass_by_value_changes(monkeypatch):
    """Fprop graph cache should reuse entries when only scalar CPU tensor values change."""
    build = _mock_score_mod_build(monkeypatch)
    entries = [
        _get_score_mod_test_graph(_score_mod_causal, extra={"softcap": torch.tensor(value)})
        for value in (0.8, 1.2, [0.8])
    ]
    assert entries[0] is entries[1]
    assert entries[2] is not entries[0]
    assert build.call_count == 2
    assert flex_attention.compile_graph.call_count == 2


def test_score_mod_cache_fwd_skips_cache_for_unkeyed_bound_method(monkeypatch):
    """Unkeyed bound methods should build fresh graphs instead of using an id-based key."""

    class UnkeyedScoreMod:
        def forward(self, _score_mod_graph, score_tensor, _tensors):
            return score_tensor

    build = _mock_score_mod_build(monkeypatch)
    callback = UnkeyedScoreMod().forward
    assert _get_score_mod_test_graph(callback) is not _get_score_mod_test_graph(callback)
    assert not cudnn_graph._graph_cache.entries
    assert build.call_count == 2


def test_score_mod_tensors_are_version_checked_for_backward(monkeypatch):
    """In-place score_mod tensor updates before backward should be rejected."""

    class FakeEntry:
        built = SimpleNamespace(graph=object())
        workspace_size = 1

    def fake_execute(graph, variant_pack, workspace_size, device):
        del graph, variant_pack, workspace_size, device

    q, k, v, _, _ = _score_mod_cache_cpu_inputs()
    q = q.requires_grad_()
    k = k.requires_grad_()
    v = v.requires_grad_()
    softcap = torch.tensor(0.8, dtype=torch.float32)

    monkeypatch.setattr(
        flex_attention, "_get_cudnn_score_mod_graph", lambda *_args, **_kwargs: (FakeEntry(), {})
    )
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
def test_dot_product_attention_score_mod(
    dtype, qkv_format, score_mod_case, scalar_loss, monkeypatch
):
    """Compare score_mod attention against equivalent reference implementations."""
    monkeypatch.setattr(cudnn_graph._graph_cache, "entries", {})
    compile_plan = Mock(wraps=flex_attention.compile_graph)
    monkeypatch.setattr(flex_attention, "compile_graph", compile_plan)
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

    # New buffers with identical metadata must reuse both plans and rebind all operands.
    repeated = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    repeated_kwargs = dict(score_mod_kwargs)
    for name in ("score_mod_tensors", "score_mod_bprop_tensors"):
        if name in repeated_kwargs:
            repeated_kwargs[name] = {
                key: value.clone() for key, value in repeated_kwargs[name].items()
            }
    repeated_out = flex_attn(
        *repeated, qkv_format=qkv_format, attn_mask_type="no_mask", **repeated_kwargs
    )
    repeated_out.backward(torch.ones_like(repeated_out) if scalar_loss else d_out)
    torch.testing.assert_close(repeated_out, out_ref, **tols)
    for actual, ref in zip(repeated, (q_ref, k_ref, v_ref)):
        torch.testing.assert_close(actual.grad, ref.grad, **tols)
    assert compile_plan.call_count == 2
    assert sum(key[0] == "score_mod" for key in cudnn_graph._graph_cache.entries) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_score_mod_inference_reuses_graph(monkeypatch):
    """Inference omits Stats and reuses its plan on new runtime buffers."""
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    monkeypatch.setattr(cudnn_graph._graph_cache, "entries", {})
    compile_plan = Mock(wraps=flex_attention.compile_graph)
    monkeypatch.setattr(flex_attention, "compile_graph", compile_plan)
    _attention_backends["backend_selection_requires_update"] = True
    attention = (
        DotProductAttention(4, 64, qkv_format="bshd", attn_mask_type="no_mask").cuda().eval()
    )
    for _ in range(2):
        q, k, v = (torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.float16) for _ in range(3))
        qr, kr, vr = (x.transpose(1, 2).float() for x in (q, k, v))
        indices = torch.arange(128, device="cuda", dtype=torch.float32)
        scores = qr @ kr.transpose(-1, -2) / 8 + indices[:, None] - indices[None, :]
        reference = (scores.softmax(-1) @ vr).transpose(1, 2)
        with torch.no_grad():
            out = attention(q, k, v, score_mod=_score_mod_post_scale_bias)
        torch.testing.assert_close(
            out.reshape(reference.shape).float(), reference, rtol=0.003, atol=0.002
        )
    assert compile_plan.call_count == 1
