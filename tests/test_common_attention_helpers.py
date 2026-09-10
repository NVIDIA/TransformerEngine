# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GPU-independent tests for shared cuDNN attention helpers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from transformer_engine.common.attention.cudnn import (
    AttentionLayout,
    FusedAttentionConfig,
    check_f16_fused_attention_support,
    encode_cudnn_version,
    normalize_attention_mask,
    ragged_batch_bucket,
    ragged_token_bucket,
)
from transformer_engine.common.attention.score_mod import (
    UNCACHEABLE_SCORE_MOD,
    freeze_score_mod_cache_key,
    score_mod_callback_cache_key,
)
from transformer_engine.common.cudnn_frontend import build_cudnn_graph, make_cudnn_graph


def _attention_config(**kwargs):
    config = FusedAttentionConfig(
        is_training=False,
        q_dtype="float16",
        kv_dtype="float16",
        layout=AttentionLayout("bshd", "bshd", "bshd", "separate"),
        bias_type="no_bias",
        mask_type="no_mask",
        softmax_type="vanilla",
        dropout=0.0,
        num_attn_heads=16,
        num_gqa_groups=16,
        max_seqlen_q=128,
        max_seqlen_kv=256,
        head_dim_qk=128,
        head_dim_v=128,
        window_size=(-1, -1),
        return_max_logit=False,
        cuda_graph=False,
        deterministic=False,
        cudnn_version=(9, 25, 0),
        sm_arch=90,
    )
    return replace(config, **kwargs)


def test_cudnn_version_encoding():
    assert encode_cudnn_version((8, 9, 7)) == 8907
    assert encode_cudnn_version((9, 25, 1)) == 92501


@pytest.mark.parametrize(
    "value, expected",
    [(1, 32), (32, 32), (33, 64), (513, 1024), (1025, 1536)],
)
def test_ragged_batch_bucket(value, expected):
    assert ragged_batch_bucket(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [(1, 1024), (1024, 1024), (1025, 2048), (32769, 65536), (65537, 98304)],
)
def test_ragged_token_bucket(value, expected):
    assert ragged_token_bucket(value) == expected


def test_bottom_right_self_attention_normalizes_to_causal():
    mask = normalize_attention_mask(
        causal=False,
        bottom_right=True,
        padding=False,
        bottom_right_diagonal=True,
        window_size=(-1, 0),
        max_seqlen_q=128,
        max_seqlen_kv=128,
    )
    assert mask.causal
    assert not mask.bottom_right
    assert not mask.bottom_right_diagonal


def test_shared_f16_policy_basic_support_and_rejection():
    assert check_f16_fused_attention_support(_attention_config()).supported
    unsupported = check_f16_fused_attention_support(_attention_config(sm_arch=75))
    assert not unsupported.supported
    assert "architecture" in unsupported.reason


def test_shared_f16_policy_explicit_frontend_capabilities():
    alibi = _attention_config(bias_type="alibi", mask_type="causal")
    assert not check_f16_fused_attention_support(alibi).supported
    assert check_f16_fused_attention_support(replace(alibi, allow_alibi=True)).supported

    extended_causal = _attention_config(
        mask_type="causal",
        window_size=(128, 64),
        sm_arch=100,
    )
    assert not check_f16_fused_attention_support(extended_causal).supported
    assert check_f16_fused_attention_support(
        replace(extended_causal, allow_extended_causal_window=True)
    ).supported

    modern_padding = _attention_config(
        mask_type="padding",
        dropout=0.1,
        cudnn_version=(9, 7, 0),
    )
    assert check_f16_fused_attention_support(modern_padding).supported
    assert not check_f16_fused_attention_support(
        replace(modern_padding, modern_mask_rules_override=True)
    ).supported


def _not_an_array(_value):
    return False


def test_score_mod_module_lambda_keys_do_not_collide():
    score_mod_0 = lambda _graph, score, _tensors: score
    score_mod_1 = lambda _graph, score, _tensors: score
    score_mod_0.__module__ = __name__
    score_mod_1.__module__ = __name__
    score_mod_0.__qualname__ = "<lambda>"
    score_mod_1.__qualname__ = "<lambda>"

    key_0 = score_mod_callback_cache_key(score_mod_0, is_array=_not_an_array)
    key_1 = score_mod_callback_cache_key(score_mod_1, is_array=_not_an_array)
    assert key_0 is not UNCACHEABLE_SCORE_MOD
    assert key_1 is not UNCACHEABLE_SCORE_MOD
    assert key_0 != key_1


def test_score_mod_bound_method_cache_policy():
    class Unkeyed:
        def forward(self, _graph, score, _tensors):
            return score

    class Keyed:
        def score_mod_graph_cache_key(self):
            return {"layers": [1, 2]}

        def forward(self, _graph, score, _tensors):
            return score

    assert (
        score_mod_callback_cache_key(Unkeyed().forward, is_array=_not_an_array)
        is UNCACHEABLE_SCORE_MOD
    )
    assert score_mod_callback_cache_key(
        Keyed().forward, is_array=_not_an_array
    ) == score_mod_callback_cache_key(Keyed().forward, is_array=_not_an_array)


def test_score_mod_key_rejects_runtime_arrays():
    marker = object()
    with pytest.raises(TypeError, match="must not include tensors"):
        freeze_score_mod_cache_key(
            {"nested": [marker]},
            is_array=lambda value: value is marker,
        )


class _FakeGraph:
    def __init__(self, workspace_size=0, unsupported=False):
        self.workspace_size = workspace_size
        self.unsupported = unsupported
        self.calls = []

    def validate(self):
        self.calls.append("validate")

    def build_operation_graph(self):
        self.calls.append("build_operation_graph")

    def create_execution_plans(self, modes):
        self.calls.append(("create_execution_plans", modes))

    def check_support(self):
        self.calls.append("check_support")
        if self.unsupported:
            raise _FakeCudnn.cudnnGraphNotSupportedError("unsupported")

    def build_plans(self, policy):
        self.calls.append(("build_plans", policy))

    def get_workspace_size(self):
        return self.workspace_size


class _FakeCudnn:
    class cudnnGraphNotSupportedError(Exception):
        pass

    data_type = SimpleNamespace(FLOAT="float")
    heur_mode = SimpleNamespace(A="a", FALLBACK="fallback")
    build_plan_policy = SimpleNamespace(HEURISTICS_CHOICE="heuristics")

    def __init__(self):
        self.graph_kwargs = None

    def pygraph(self, **kwargs):
        self.graph_kwargs = kwargs
        return "graph"


def test_shared_cudnn_graph_creation_and_finalization():
    cudnn = _FakeCudnn()
    assert make_cudnn_graph(cudnn, "half", name="attention", handle=7) == "graph"
    assert cudnn.graph_kwargs == {
        "io_data_type": "half",
        "intermediate_data_type": "float",
        "compute_data_type": "float",
        "name": "attention",
        "handle": 7,
    }

    graph = _FakeGraph(workspace_size=0)
    assert build_cudnn_graph(cudnn, graph, description="attention") == 1
    assert graph.calls == [
        "validate",
        "build_operation_graph",
        ("create_execution_plans", ["a", "fallback"]),
        "check_support",
        ("build_plans", "heuristics"),
    ]


def test_shared_cudnn_graph_support_error_has_context():
    with pytest.raises(RuntimeError, match="cuDNN test graph is not supported"):
        build_cudnn_graph(
            _FakeCudnn(), _FakeGraph(unsupported=True), description="test"
        )
