# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Known FE host signatures and fail-closed forward export; no GPU required."""

import math
from types import SimpleNamespace

import pytest
import tvm_ffi

from transformer_engine.common.CuTeDSL import frost_attn_fwd

_BASE_ORDER = tuple(
    "q_ptr k_ptr v_ptr o_ptr lse_ptr sinks_ptr meta_ptr o_desc_ptr problem_size "
    "q_strides k_strides v_strides o_strides lse_strides lse_ext scale_softmax_log2 "
    "n_thd_units seq_q_lens_addr thd_q_lens_ptr thd_kv_lens_ptr thd_lens_form "
    "o_partial_ptr block_table_ptr block_table_v_ptr table_strides n_pages stream".split()
)
_SCORE = dict(
    score_max_ptr=None,
    score_sum_exp_ptr=None,
    score_max_strides=(0, 0, 0),
    score_sum_exp_strides=(0, 0, 0),
)
_BIAS = dict(
    bias_ptr=None,
    bias_strides=(0, 0, 0),
    bias_extents=(1, 1),
    kv_mask_ptr=None,
    kv_mask_batch_stride=0,
    inv_scale_softmax=1.0,
)


def _graph(extra):
    order = _BASE_ORDER[:-1] + tuple(extra) + ("stream",)
    args = dict.fromkeys(order)
    for role, h in [("q", 4), ("k", 1), ("v", 1), ("o", 4)]:
        args[role + "_strides"] = (h * 128, h * 128, 128)
    args.update(
        lse_strides=(0, 0, 0),
        lse_ext=0,
        seq_q_lens_addr=0,
        thd_lens_form=3,
        table_strides=(0, 0),
        n_pages=0,
        scale_softmax_log2=128**-0.5 * math.log2(math.e),
        n_thd_units=64,
        **extra,
    )
    spec = SimpleNamespace(
        order=order,
        template=[args[n] for n in order],
        fn=tvm_ffi.convert(lambda: None),
        quant=None,
        paged=False,
        has_sink=False,
        gate_expect=None,
        has_lse=True,
        lse_padded=False,
        lse_head_major=False,
        lse_stride_override=False,
        lse_fill_plan=None,
        split_workspace=None,
        lens_form=3,
        n_q_lens=3,
        n_kv_lens=3,
        device_index=0,
        b=2,
        qh=4,
        kh=1,
        d_qk=128,
        d_v=128,
        total_q=512,
        total_kv=512,
        s_q_max=256,
        expect={r: "bf16" for r in ("q", "k", "v", "o")},
        cga_tile_m=128,
        workspace_alignment=256,
        off_o_desc=256,
        scratch_bytes=512,
    )
    return SimpleNamespace(
        _compiled_plans=[SimpleNamespace(_prepared=SimpleNamespace(spec=spec))],
        _plan_index=0,
        get_workspace_size=lambda: 1024,
    )


def _export(module, g):
    return module._export(g, 2, 4, 1, 256, 256, 512, 512, 128**-0.5, 0)


@pytest.mark.parametrize("extra", [{}, _SCORE, {**_SCORE, **_BIAS}])
def test_frost_forward_known_contract(extra):
    """Legacy and current FE declarations keep the stream and operands bound."""
    g = _graph(extra)
    native, constants, metadata = _export(frost_attn_fwd, g)
    spec = g._compiled_plans[0]._prepared.spec
    count = len(spec.order)
    assert native is spec.fn
    assert len(constants) == metadata[3] == count
    assert len(metadata) == 22 + 2 * count
    assert metadata[-2:] == [3, 0]
    for index in range(26, count - 1):
        assert metadata[22 + 2 * index : 24 + 2 * index] == [0, index]


@pytest.mark.parametrize("name,value", list({**_SCORE, **_BIAS}.items()))
def test_frost_forward_declines_optional_features(name, value):
    """New slots never silently enable score outputs, bias or KV masks."""
    g = _graph({**_SCORE, **_BIAS})
    spec = g._compiled_plans[0]._prepared.spec
    changed = 8 if value is None else ((8,) * len(value) if isinstance(value, tuple) else 8)
    spec.template[spec.order.index(name)] = changed
    with pytest.raises(RuntimeError, match="Unsupported cuDNN OSS forward optional argument"):
        _export(frost_attn_fwd, g)


@pytest.mark.parametrize(
    "kind", ["unknown", "missing", "reordered", "duplicate", "truncated_template"]
)
def test_frost_forward_declines_unknown_contract(kind):
    """A changed positional ABI must fail before any FFI invocation."""
    g = _graph({**_SCORE, **_BIAS})
    spec = g._compiled_plans[0]._prepared.spec
    order = list(spec.order)
    if kind == "unknown":
        order[-2] = "new_slot"
    elif kind == "missing":
        order.pop(-2)
        spec.template.pop(-2)
    elif kind == "reordered":
        order[0], order[1] = order[1], order[0]
    elif kind == "duplicate":
        order[1] = order[0]
    else:
        spec.template.pop()
    spec.order = order
    with pytest.raises(RuntimeError, match="Unsupported cuDNN OSS prepared forward contract"):
        _export(frost_attn_fwd, g)


@pytest.mark.parametrize(
    "name,value",
    [
        ("gate_expect", "bf16"),
        ("has_sink", True),
        ("paged", True),
        ("lse_padded", True),
        ("quant", object()),
        ("split_workspace", object()),
        ("lens_form", 2),
    ],
)
def test_frost_forward_retains_feature_guards(name, value):
    """ABI compatibility does not broaden the supported operation contract."""
    g = _graph({**_SCORE, **_BIAS})
    setattr(g._compiled_plans[0]._prepared.spec, name, value)
    with pytest.raises(RuntimeError, match="Unsupported cuDNN OSS prepared forward contract"):
        _export(frost_attn_fwd, g)
