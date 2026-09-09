# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
import runpy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from transformer_engine.pytorch.utils import deinterleave_glu_tensor
from transformer_engine.pytorch.models import DeepSeekV3MoE, MultiLatentAttention

SEQ_LEN = 128
BATCH = 2
HIDDEN = 256
HEADS = 4
DTYPE = torch.bfloat16

MLA_KWARGS = dict(
    q_lora_rank=96,
    kv_lora_rank=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
)


def _input(requires_grad=True):
    torch.manual_seed(1234)
    return torch.randn(
        SEQ_LEN, BATCH, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=requires_grad
    )


@pytest.mark.parametrize("nope,rope,vdim", [(64, 32, 64), (48, 32, 64), (64, 48, 64), (64, 32, 48)])
def test_mla_rope_matches_pytorch(nope, rope, vdim):
    from transformer_engine.pytorch.models.deepseek_v3 import mla_rope

    if not mla_rope.HAVE_TRITON:
        pytest.skip("Triton unavailable")
    s, b, h = 64, 2, 4
    cos, sin = mla_rope.build_rope_tables(s, rope, device="cuda")

    torch.manual_seed(0)
    q_leaf = torch.randn(s, b, h, nope + rope, device="cuda", requires_grad=True)
    kv_leaf = torch.randn(s, b, h, nope + vdim, device="cuda", requires_grad=True)
    pos_leaf = torch.randn(s, b, 1, rope, device="cuda", requires_grad=True)
    grad_q = torch.randn(s, b, h, nope + rope, device="cuda")
    grad_k = torch.randn(s, b, h, nope + rope, device="cuda")
    grad_v = torch.randn(s, b, h, vdim, device="cuda")

    def run(fmt):
        # non-leaf copies: the Triton q kernel rotates in place
        q, kv, pos = q_leaf * 1.0, kv_leaf * 1.0, pos_leaf * 1.0
        q_out = mla_rope.apply_mla_rope_q(q, cos, sin, nope, rope, fmt)
        k_out, v_out = mla_rope.apply_mla_rope_kv(kv, pos, cos, sin, nope, rope, vdim, fmt)
        # fresh grad clones: the Triton q backward modifies its input grad in place
        torch.autograd.backward(
            [q_out, k_out, v_out], [grad_q.clone(), grad_k.clone(), grad_v.clone()]
        )
        grads = (q_leaf.grad.clone(), kv_leaf.grad.clone(), pos_leaf.grad.clone())
        q_leaf.grad = kv_leaf.grad = pos_leaf.grad = None
        return (q_out.clone(), k_out, v_out), grads

    (q_t, k_t, v_t), grads_t = run("sbhd")

    seq_dim = 0
    q_ref = torch.cat(
        (
            (q_leaf * 1.0)[..., :nope],
            mla_rope._rotate_interleaved_to_neox((q_leaf * 1.0)[..., nope:], cos, sin, seq_dim),
        ),
        dim=-1,
    )
    k_ref = torch.cat(
        (
            (kv_leaf * 1.0)[..., :nope],
            mla_rope._rotate_interleaved_to_neox(pos_leaf * 1.0, cos, sin, seq_dim).expand(
                s, b, h, rope
            ),
        ),
        dim=-1,
    )
    v_ref = (kv_leaf * 1.0)[..., nope:]
    torch.autograd.backward([q_ref, k_ref, v_ref], [grad_q.clone(), grad_k.clone(), grad_v.clone()])

    torch.testing.assert_close(q_t, q_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_t, k_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_t, v_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grads_t[0], q_leaf.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grads_t[1], kv_leaf.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grads_t[2], pos_leaf.grad, rtol=1e-5, atol=1e-5)


def test_rope_tables_yarn():
    from transformer_engine.pytorch.models.deepseek_v3 import mla_rope

    s, rope = 8192, 64
    cos, sin = mla_rope.build_rope_tables(s, rope, device="cuda")
    cos_none, sin_none = mla_rope.build_rope_tables(s, rope, device="cuda", scaling_factor=None)
    assert torch.equal(cos, cos_none) and torch.equal(sin, sin_none)

    yarn = dict(scaling_factor=40.0, original_max_position_embeddings=4096)
    cos_y, sin_y = mla_rope.build_rope_tables(s, rope, device="cuda", **yarn)
    factor = mla_rope.yarn_concentration_factor(40.0, 1.0, 0.0)
    assert factor == pytest.approx(0.1 * math.log(40.0) + 1.0)
    # amplitude scaled by the concentration factor
    torch.testing.assert_close(cos_y**2 + sin_y**2, torch.full_like(cos_y, factor**2))
    # high-frequency dims untouched, low-frequency dims interpolated by 1/scaling_factor
    torch.testing.assert_close(cos_y[:, 0] / factor, cos[:, 0])
    angle_y = torch.atan2(sin_y[:, rope // 2 - 1], cos_y[:, rope // 2 - 1])
    angle = torch.atan2(sin[:, rope // 2 - 1], cos[:, rope // 2 - 1])
    torch.testing.assert_close(angle_y[:64], angle[:64] / 40.0, atol=1e-4, rtol=0)


@pytest.mark.parametrize("mscale_all_dim", [0.0, 1.0])
def test_mla_yarn_softmax_scale(mscale_all_dim):
    mla = MultiLatentAttention(
        HIDDEN,
        HEADS,
        params_dtype=DTYPE,
        rope_scaling_factor=40.0,
        original_max_position_embeddings=64,
        mscale_all_dim=mscale_all_dim,
        **MLA_KWARGS,
    )
    m = 0.1 * mscale_all_dim * math.log(40.0) + 1.0
    qk_head_dim = MLA_KWARGS["qk_nope_head_dim"] + MLA_KWARGS["qk_rope_head_dim"]
    assert mla.softmax_scale == pytest.approx(m * m / math.sqrt(qk_head_dim))


@pytest.mark.parametrize("shared", [False, True], ids=["no_shared", "shared"])
@pytest.mark.parametrize("grouped", [False, True], ids=["ungrouped", "grouped"])
@pytest.mark.parametrize("topk", [2, 4])
def test_moe_matches_dense_reference(shared, grouped, topk):
    """Routed output must equal the prob-weighted sum of the selected expert MLPs."""
    torch.manual_seed(0)
    num_experts = 4
    moe = DeepSeekV3MoE(
        HIDDEN,
        moe_ffn_hidden_size=128,
        num_experts=num_experts,
        topk=topk,
        num_groups=2 if grouped else None,
        group_topk=topk // 2 if grouped else None,
        shared_expert_ffn_hidden_size=128 if shared else None,
        params_dtype=DTYPE,
    )
    x = _input()
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert torch.isfinite(x.grad).all()

    tokens = x.detach().reshape(-1, HIDDEN)
    probs, _ = moe._route(moe.gate(tokens).float())
    assert (probs > 0).sum(dim=1).eq(topk).all()
    assert moe._last_tokens_per_expert.sum().item() == tokens.shape[0] * topk

    fc1, _, fc2 = moe.experts
    ref = torch.zeros_like(tokens)
    for e in range(num_experts):
        w1 = deinterleave_glu_tensor(getattr(fc1, f"weight{e}"), 32)
        w2 = getattr(fc2, f"weight{e}")
        gate_part, lin_part = (tokens @ w1.t()).chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate_part.float()) * lin_part.float()
        ref += (act.to(DTYPE) * probs[:, e : e + 1].to(DTYPE)) @ w2.t()
    if shared:
        ref += moe.shared_expert(tokens)
    torch.testing.assert_close(out.reshape(-1, HIDDEN), ref, rtol=0.05, atol=0.05)

    bias_before = moe.expert_bias.clone()
    moe.update_expert_bias()
    assert torch.isfinite(moe.expert_bias).all()
    if topk < num_experts:
        assert not torch.equal(bias_before, moe.expert_bias)


@pytest.fixture(scope="module")
def deepseek_example():
    path = (
        Path(__file__).resolve().parents[2] / "examples/pytorch/deepseek_v3/deepseek_v3_layer_ep.py"
    )
    return runpy.run_path(str(path))


@pytest.fixture
def single_rank_group(tmp_path):
    if dist.is_initialized():
        pytest.skip("Requires an isolated process group")
    dist.init_process_group("nccl", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("grouped", [False, True])
def test_naive_moe_gradients(deepseek_example, single_rank_group, grouped):
    torch.manual_seed(123)
    moe = deepseek_example["NaiveMoE"](
        HIDDEN, 128, 4, 2, single_rank_group, 128, torch.float32, grouped
    )
    x = torch.randn(64, HIDDEN, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    out = moe(x)

    scores = torch.sigmoid(moe.gate(x_ref))
    idx = torch.topk(scores + moe.expert_bias, moe.topk, dim=-1).indices
    selected = scores.gather(1, idx)
    selected = selected / selected.sum(-1, keepdim=True) * 2.5
    probs = torch.zeros_like(scores).scatter(1, idx, selected)
    ref = torch.zeros_like(x_ref)
    for e in range(moe.local):
        if grouped:
            fc1, _, fc2 = moe.experts
            w1 = deinterleave_glu_tensor(getattr(fc1, f"weight{e}"), 32)
            w2 = getattr(fc2, f"weight{e}")
        else:
            w1, w2 = moe.w1[e], moe.w2[e]
        act = moe._swiglu(torch.nn.functional.linear(x_ref, w1))
        ref = ref + torch.nn.functional.linear(act * probs[:, e : e + 1], w2)
    ref = ref + moe.shared_w2(moe._swiglu(moe.shared_w1(x_ref)))

    grad = torch.randn_like(out)
    actual_grads = torch.autograd.grad(out, (x, moe.gate.weight), grad)
    ref_grads = torch.autograd.grad(ref, (x_ref, moe.gate.weight), grad)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    for actual, expected in zip(actual_grads, ref_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert actual_grads[1].abs().max() > 0


@pytest.mark.parametrize("value", [1.0, float("nan"), float("inf"), None])
def test_example_finite_check(deepseek_example, single_rank_group, value):
    tensor = None if value is None else torch.tensor(value, device="cuda")
    assert deepseek_example["_check_finite"]([tensor], torch.device("cuda")) == (value == 1.0)
