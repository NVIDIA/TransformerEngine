# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GPU-independent tests for shared cuDNN attention helpers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from transformer_engine.common.attention import cache_debug
from transformer_engine.common.attention.cudnn import (
    AttentionLayout,
    FusedAttentionConfig,
    check_f16_fused_attention_support,
    check_fp8_fused_attention_support,
    cudnn_mask_options,
    encode_cudnn_version,
    normalize_attention_mask,
    parse_attention_layout,
    ragged_batch_bucket,
    ragged_token_bucket,
)
from transformer_engine.common.attention.fp8 import (
    FP8AttentionGraphConfig,
    attention_format_stride,
    build_fp8_backward_operation,
    build_fp8_forward_operation,
    mxfp8_padded_sizes,
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


def test_shared_f16_policy_framework_parity_features():
    alibi = _attention_config(bias_type="alibi", mask_type="causal")
    assert check_f16_fused_attention_support(alibi).supported

    extended_causal = _attention_config(
        mask_type="causal",
        window_size=(128, 64),
        sm_arch=100,
    )
    assert check_f16_fused_attention_support(extended_causal).supported

    modern_padding = _attention_config(
        mask_type="padding",
        dropout=0.1,
        cudnn_version=(9, 7, 0),
    )
    assert check_f16_fused_attention_support(modern_padding).supported


def test_shared_fp8_policy():
    fp8 = _attention_config(q_dtype="float8_e4m3", kv_dtype="float8_e4m3", sm_arch=100)
    assert check_fp8_fused_attention_support(fp8).supported
    assert check_fp8_fused_attention_support(replace(fp8, cudnn_version=(9, 10, 1))).supported
    assert not check_fp8_fused_attention_support(replace(fp8, cudnn_version=(9, 10, 0))).supported
    assert not check_fp8_fused_attention_support(replace(fp8, bias_type="alibi")).supported
    assert not check_fp8_fused_attention_support(replace(fp8, return_max_logit=True)).supported
    assert not check_fp8_fused_attention_support(replace(fp8, head_dim_qk=200)).supported


def test_shared_fp8_thd_policy():
    thd = _attention_config(
        q_dtype="float8_e4m3",
        kv_dtype="float8_e4m3",
        layout=AttentionLayout("thd", "thd", "thd", "separate"),
        mask_type="padding",
        cudnn_version=(9, 23, 0),
        sm_arch=100,
    )
    assert check_fp8_fused_attention_support(thd).supported
    assert check_fp8_fused_attention_support(replace(thd, sm_arch=90)).supported
    assert check_fp8_fused_attention_support(
        replace(thd, mask_type="padding_causal_bottom_right")
    ).supported
    assert not check_fp8_fused_attention_support(
        replace(thd, cudnn_version=(9, 22, 9))
    ).supported
    assert not check_fp8_fused_attention_support(replace(thd, mask_type="no_mask")).supported
    assert not check_fp8_fused_attention_support(
        replace(thd, is_training=True, sm_arch=90)
    ).supported
    assert not check_fp8_fused_attention_support(replace(thd, head_dim_qk=144)).supported

    sink_backward = replace(thd, is_training=True, softmax_type="learnable")
    assert not check_fp8_fused_attention_support(
        replace(sink_backward, cudnn_version=(9, 25, 1))
    ).supported
    assert check_fp8_fused_attention_support(
        replace(sink_backward, cudnn_version=(9, 26, 0))
    ).supported


def test_shared_fp8_thd_policy_allows_64bit_offsets():
    thd = _attention_config(
        q_dtype="float8_e4m3",
        kv_dtype="float8_e4m3",
        layout=AttentionLayout("thd", "thd", "thd", "separate"),
        mask_type="padding",
        max_seqlen_q=1_048_577,
        max_seqlen_kv=1_048_577,
        cudnn_version=(9, 23, 0),
        sm_arch=100,
    )
    assert check_fp8_fused_attention_support(thd).supported


@pytest.mark.parametrize(
    "layout, expected",
    [
        ("bs3hd", ("bshd", "bshd", "3hd")),
        ("bshd_bs2hd", ("bshd", "bshd", "hd_2hd")),
        ("bhsd_bhsd_bhsd", ("bhsd", "bhsd", "sd_sd_sd")),
        ("paged_kv_bshd_bshd_bshd", ("bshd", "bshd", "paged_separate")),
    ],
)
def test_parse_attention_layout(layout, expected):
    parsed = parse_attention_layout(layout)
    assert (parsed.q_format, parsed.kv_format, parsed.layout_group) == expected


def test_shared_mask_options_use_modern_band_api():
    options = cudnn_mask_options(
        causal=True,
        bottom_right=False,
        padding=False,
        bottom_right_diagonal=True,
        window_size=(32, -1),
        max_seqlen_q=64,
        max_seqlen_kv=128,
        cudnn_version=(9, 6, 0),
    )
    assert options == {
        "diagonal_alignment": "bottom_right",
        "is_padding": False,
        "diagonal_band_left_bound": 33,
        "diagonal_band_right_bound": 0,
    }


def test_shared_fp8_shape_helpers():
    assert attention_format_stride(2, 8, 128, 64, "bshd") == (65536, 64, 512, 1)
    assert attention_format_stride(2, 8, 128, 64, "bhsd") == (65536, 8192, 64, 1)
    assert mxfp8_padded_sizes(129, 33, 160, 96) == {
        "s_q_padded": 256,
        "s_kv_padded": 128,
        "s_q_scale_padded": 8,
        "s_kv_scale_padded": 4,
        "d_qk_padded": 256,
        "d_v_padded": 128,
        "d_qk_scale_padded": 8,
        "d_v_scale_padded": 4,
    }


class _FakeFP8Graph:
    def __init__(self):
        self.call = None

    def sdpa_fp8(self, *args, **kwargs):
        self.call = ("fp8_fwd", args, kwargs)
        return "o", "stats", "amax_s", "amax_o"

    def sdpa_mxfp8(self, *args, **kwargs):
        self.call = ("mx_fwd", args, kwargs)
        return "o", "stats", "amax_o"

    def sdpa_fp8_backward(self, *args, **kwargs):
        self.call = ("fp8_bwd", args, kwargs)
        return "dq", "dk", "dv", "aq", "ak", "av", "ap"

    def sdpa_mxfp8_backward(self, *args, **kwargs):
        self.call = ("mx_bwd", args, kwargs)
        return "dq", "dk", "dv", "aq", "ak", "av"


def test_shared_fp8_graph_operation_dispatch():
    graph = _FakeFP8Graph()
    forward = build_fp8_forward_operation(
        graph,
        {
            "q": 1,
            "k": 2,
            "v": 3,
            "descale_q": 4,
            "descale_k": 5,
            "descale_v": 6,
            "descale_s": 7,
            "scale_s": 8,
            "scale_o": 9,
        },
        {"attn_scale": 0.125},
        FP8AttentionGraphConfig("delayed", "forward"),
    )
    assert forward == {
        "output": "o",
        "stats": "stats",
        "amax_s": "amax_s",
        "amax_o": "amax_o",
    }
    assert graph.call[0] == "fp8_fwd"

    backward = build_fp8_backward_operation(
        graph,
        {
            **{name: name for name in ("q", "k", "v", "o", "do", "stats")},
            **{
                name: name
                for name in (
                    "descale_q",
                    "descale_k",
                    "descale_v",
                    "descale_o",
                    "descale_do",
                    "descale_s",
                    "descale_dp",
                    "scale_s",
                    "scale_dq",
                    "scale_dk",
                    "scale_dv",
                    "scale_dp",
                )
            },
        },
        {"attn_scale": 0.125},
        FP8AttentionGraphConfig("current", "backward"),
    )
    assert backward["amax_dp"] == "ap"
    assert graph.call[0] == "fp8_bwd"


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


def test_shared_cudnn_graph_reports_build_diagnostics():
    events = []
    graph = _FakeGraph()
    build_cudnn_graph(
        _FakeCudnn(),
        graph,
        description="attention",
        debug_callback=lambda event, elapsed_ns: events.append((event, elapsed_ns)),
    )
    assert [event for event, _ in events] == [
        "CREATE_GRAPH",
        "validate",
        "build_operation_graph",
        "create_execution_plans",
        "check_support",
        "build_plans",
        "BUILD_PLANS",
    ]
    assert all(elapsed_ns >= 0 for _, elapsed_ns in events)


def test_shared_cudnn_graph_support_error_has_context():
    with pytest.raises(RuntimeError, match="cuDNN test graph is not supported"):
        build_cudnn_graph(_FakeCudnn(), _FakeGraph(unsupported=True), description="test")


@pytest.fixture
def cache_debug_environment(monkeypatch):
    for variable in (
        "NVTE_FUSED_ATTN_CACHE_DEBUG",
        "RANK",
        "LOCAL_RANK",
        "OMPI_COMM_WORLD_RANK",
        "SLURM_PROCID",
    ):
        monkeypatch.delenv(variable, raising=False)
    cache_debug._reset_for_tests()
    yield monkeypatch
    cache_debug._reset_for_tests()


def test_cache_debug_rank_selection(cache_debug_environment):
    monkeypatch = cache_debug_environment
    monkeypatch.setenv("NVTE_FUSED_ATTN_CACHE_DEBUG", "2")
    monkeypatch.setenv("RANK", "1")
    cache_debug._reset_for_tests()
    assert not cache_debug.enabled()

    monkeypatch.setenv("NVTE_FUSED_ATTN_CACHE_DEBUG", "2:1,3")
    cache_debug._reset_for_tests()
    assert cache_debug.enabled(trace=True)

    monkeypatch.setenv("NVTE_FUSED_ATTN_CACHE_DEBUG", "1:all")
    cache_debug._reset_for_tests()
    assert cache_debug.enabled()
    assert not cache_debug.enabled(trace=True)


def test_cache_debug_summary(cache_debug_environment):
    cache_debug_environment.setenv("NVTE_FUSED_ATTN_CACHE_DEBUG", "1")
    cache_debug.record_lookup("f16", "fwd", hit=False, key=("graph", 1))
    cache_debug.record_event("f16", "fwd", "create_graph")
    cache_debug.record_event("f16", "fwd", "cache_graph")
    cache_debug.record_event("f16", "fwd", "execute", device=0)
    cache_debug.record_lookup("f16", "fwd", hit=True, key=("graph", 1))
    cache_debug.record_build_time("f16", "fwd", "validate", 2_000_000)

    summary = cache_debug.render_summary()
    assert "summary begin" in summary
    assert "f16 fwd" in summary
    assert "hit=   1" in summary
    assert "miss=   1" in summary
    assert "execute=   1" in summary
    assert "validate" in summary
    assert "2.000 ms/call" in summary


def test_cache_debug_level_two_traces_lookup_key(cache_debug_environment, capsys):
    cache_debug_environment.setenv("NVTE_FUSED_ATTN_CACHE_DEBUG", "2")
    cache_debug.record_lookup("fp8", "bwd", hit=False, key=("shape", 128))
    trace = capsys.readouterr().err
    assert "[FUSED-ATTN-CACHE]" in trace
    assert "fp8 bwd MISS" in trace
    assert "('shape', 128)" in trace
