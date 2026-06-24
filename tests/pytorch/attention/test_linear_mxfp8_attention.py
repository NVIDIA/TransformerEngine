# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 end-to-end attention unit test - DSv3 671B MLA dimensions.

Path: LayerNormLinear(Q/KV, MXFP8) -> MLA-RoPE (Triton)
      -> DotProductAttention(MXFP8) -> Linear(out, MXFP8).
Tensor layout: sbhd (seq-first) throughout.

Run:
    python3 -m pytest tests/pytorch/attention/test_linear_mxfp8_attention.py -v -s

Expected benchmark output (GB200, b=1, s=4096):
    [PERF] b=1 s=4096:
      BF16 fprop:   8.582 ms  (477274 tok/s)
      BF16 bprop:  14.006 ms  (292445 tok/s)
      MXFP8 fprop:  5.210 ms  (786180 tok/s)
      MXFP8 bprop:  8.763 ms  (467428 tok/s)
      Fprop speedup: 1.65x
      Bprop speedup: 1.60x
"""

import os
import pathlib
import sys

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.utils import get_cudnn_version

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent.parent)] + sys.path
from utils import ModelConfig, compare_and_assert, get_available_attention_backends
from mla_rope_utils import apply_mla_rope, build_rope_tables


try:
    from transformer_engine.common.recipe import MXFP8BlockScaling

    mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
except (ImportError, AttributeError):
    mxfp8_available = False
    reason_for_no_mxfp8 = "MXFP8BlockScaling not available in this build"

# DSv3 671B MLA dims (micro_batch=1, seq_len=4096)
NUM_HEADS = 128
HEAD_DIM_ROPE = 64
HEAD_DIM_NOPE = 128
HEAD_DIM_QK = HEAD_DIM_NOPE + HEAD_DIM_ROPE  # 192
HEAD_DIM_V = 128
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM_V  # 16384
Q_SIZE = NUM_HEADS * HEAD_DIM_QK  # 24576
KV_SIZE = NUM_HEADS * (HEAD_DIM_NOPE + HEAD_DIM_V)  # 32768
KV_PROJ_SIZE = KV_SIZE + HEAD_DIM_ROPE  # [K_NOPE | V] per head, plus shared K RoPE
SEED = 42

WARMUP_ITERS = 10
TIMED_ITERS = 100
ATOL = 5e-1
RTOL = 5e-2
RMSE_TOL = 0.11
_DETERMINISTIC = (
    not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
    or torch.are_deterministic_algorithms_enabled()
)


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


def _set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _build_modules(dtype: torch.dtype = torch.bfloat16):
    def _make_modules():
        q_proj = te.LayerNormLinear(
            HIDDEN_SIZE,
            Q_SIZE,
            bias=True,
            params_dtype=dtype,
            device="cuda",
        )
        kv_proj = te.LayerNormLinear(
            HIDDEN_SIZE,
            KV_PROJ_SIZE,
            bias=True,
            params_dtype=dtype,
            device="cuda",
        )
        dpa = te.DotProductAttention(
            num_attention_heads=NUM_HEADS,
            kv_channels=(HEAD_DIM_QK, HEAD_DIM_V),
            attention_dropout=0.0,
            qkv_format="sbhd",
        ).to(device="cuda")
        out = te.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True).to(dtype=dtype, device="cuda")
        return q_proj, kv_proj, dpa, out

    base = _make_modules()
    mxfp8 = _make_modules()

    with torch.no_grad():
        for dst_module, src_module in zip(mxfp8, base):
            for p_dst, p_src in zip(dst_module.parameters(), src_module.parameters()):
                p_dst.copy_(p_src)

    for m in base + mxfp8:
        m.train()

    return base, mxfp8


def _require_attention_backends(
    batch_size: int,
    seq_len: int,
    fp8_recipe,
) -> None:
    if get_cudnn_version() < (9, 2, 1):
        pytest.skip("cuDNN 9.2.1+ is required for FP8 fused attention.")

    config = ModelConfig(
        batch_size,
        seq_len,
        NUM_HEADS,
        HEAD_DIM_QK,
        head_dim_v=HEAD_DIM_V,
    )
    fp8_meta = {"recipe": fp8_recipe}
    fp8_backends, _, _ = get_available_attention_backends(
        config,
        qkv_dtype=torch.float8_e4m3fn,
        qkv_layout="sbhd_sbhd_sbhd",
        fp8=True,
        fp8_meta=fp8_meta,
        is_training=True,
        deterministic=_DETERMINISTIC,
    )
    _, fused_attn_supported_fp8, _ = fp8_backends
    if not fused_attn_supported_fp8:
        pytest.skip("No fused FP8 attention backend available for DSv3 MLA shape.")

    bf16_backends, _, _ = get_available_attention_backends(
        config,
        qkv_dtype=torch.bfloat16,
        qkv_layout="sbhd_sbhd_sbhd",
        is_training=True,
        deterministic=_DETERMINISTIC,
    )
    if sum(bf16_backends) < 1:
        pytest.skip("No BF16 attention backend available for DSv3 MLA shape.")

    _attention_backends["backend_selection_requires_update"] = True


def _run_projections(
    q_proj: te.LayerNormLinear,
    kv_proj: te.LayerNormLinear,
    x: torch.Tensor,
    is_first_microbatch: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Q and workload-shaped KV, then expose metadata-only RoPE views."""
    if is_first_microbatch is None:
        q_flat = q_proj(x)
        kv_flat = kv_proj(x)
    else:
        q_flat = q_proj(x, is_first_microbatch=is_first_microbatch)
        kv_flat = kv_proj(x, is_first_microbatch=is_first_microbatch)

    s, b, _ = x.shape
    q = q_flat.view(s, b, NUM_HEADS, HEAD_DIM_QK)
    kv = kv_flat[:, :, :KV_SIZE].view(s, b, NUM_HEADS, HEAD_DIM_NOPE + HEAD_DIM_V)
    k_pos_emb = kv_flat[:, :, KV_SIZE:].view(s, b, 1, HEAD_DIM_ROPE)
    return q_flat, kv_flat, q, kv, k_pos_emb


def _run_forward_bf16(
    modules: tuple,
    x: torch.Tensor,
    rope_tables: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_proj, kv_proj, dpa, out_linear = modules
    _, _, q, kv, k_pos_emb = _run_projections(q_proj, kv_proj, x)
    q, k, v = apply_mla_rope(q, kv, k_pos_emb, cos_table=rope_tables[0], sin_table=rope_tables[1])
    attn_out = dpa(q, k, v, qkv_format="sbhd")
    return q, k, v, out_linear(attn_out.view(x.shape[0], x.shape[1], HIDDEN_SIZE))


def _run_forward_mxfp8(
    modules: tuple,
    x: torch.Tensor,
    recipe,
    rope_tables: tuple[torch.Tensor, torch.Tensor],
    is_first_microbatch: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """is_first_microbatch=True caches quantized weights; False reuses cache; None re-quantizes."""
    q_proj, kv_proj, dpa, out_linear = modules

    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        _, _, q, kv, k_pos_emb = _run_projections(
            q_proj,
            kv_proj,
            x,
            is_first_microbatch,
        )
        q, k, v = apply_mla_rope(
            q,
            kv,
            k_pos_emb,
            cos_table=rope_tables[0],
            sin_table=rope_tables[1],
        )
        attn_out = dpa(q, k, v, qkv_format="sbhd")
        out = out_linear(
            attn_out.view(x.shape[0], x.shape[1], HIDDEN_SIZE),
            is_first_microbatch=is_first_microbatch,
        )

    return q, k, v, out


def _compute_errors(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    return diff.max().item(), diff.pow(2).mean().sqrt().item()


def _clear_training_step_grads(modules: tuple, x: torch.Tensor) -> None:
    x.grad = None
    for module in modules:
        for param in module.parameters():
            param.grad = None


def _benchmark_training_step(
    forward_fn,
    modules: tuple,
    x: torch.Tensor,
    *forward_args,
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
) -> tuple[float, float]:
    for _ in range(warmup):
        _clear_training_step_grads(modules, x)
        *_, out = forward_fn(modules, x, *forward_args)
        out.sum().backward()
    torch.cuda.synchronize()

    fprop_ms = 0.0
    bprop_ms = 0.0
    for _ in range(iters):
        _clear_training_step_grads(modules, x)

        fprop_start = torch.cuda.Event(enable_timing=True)
        fprop_end = torch.cuda.Event(enable_timing=True)
        bprop_start = torch.cuda.Event(enable_timing=True)
        bprop_end = torch.cuda.Event(enable_timing=True)

        fprop_start.record()
        *_, out = forward_fn(modules, x, *forward_args)
        fprop_end.record()

        bprop_start.record()
        out.sum().backward()
        bprop_end.record()

        torch.cuda.synchronize()
        fprop_ms += fprop_start.elapsed_time(fprop_end)
        bprop_ms += bprop_start.elapsed_time(bprop_end)

    return fprop_ms / iters, bprop_ms / iters


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [4096])
class TestLinearMXFP8Attention:

    def test_accuracy(self, batch_size: int, seq_len: int) -> None:
        """Validate MXFP8 against BF16 using loose tolerances."""
        fp8_recipe = MXFP8BlockScaling(fp8_dpa=True)
        _require_attention_backends(batch_size, seq_len, fp8_recipe)
        _set_seed()
        baseline_modules, mxfp8_modules = _build_modules()
        x = torch.randn(seq_len, batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        rope_tables = build_rope_tables(seq_len, device=x.device)

        q_bf16, k_bf16, v_bf16, out_bf16 = _run_forward_bf16(baseline_modules, x, rope_tables)
        q_mxfp8, k_mxfp8, v_mxfp8, out_mxfp8 = _run_forward_mxfp8(
            mxfp8_modules,
            x,
            fp8_recipe,
            rope_tables,
        )

        for name, tensor in [("Q", q_mxfp8), ("K", k_mxfp8), ("V", v_mxfp8)]:
            assert not torch.isnan(tensor).any(), f"MXFP8 {name} contains NaN"
            assert not torch.isinf(tensor).any(), f"MXFP8 {name} contains Inf"
            assert tensor.float().abs().max() > 0, f"MXFP8 {name} is all zeros"

        assert not torch.isnan(out_mxfp8).any(), "MXFP8 output contains NaN"
        assert not torch.isinf(out_mxfp8).any(), "MXFP8 output contains Inf"
        assert out_mxfp8.float().abs().max() > 0, "MXFP8 output is all zeros"

        max_abs_q, rms_q = _compute_errors(q_bf16, q_mxfp8)
        print(f"\n[Q] b={batch_size} s={seq_len}: max_abs={max_abs_q:.6f}  rms={rms_q:.6f}")
        compare_and_assert(
            q_mxfp8,
            q_bf16,
            "q_mxfp8",
            "q_bf16",
            ATOL,
            RTOL,
            RMSE_TOL,
            True,
        )

        max_abs_k, rms_k = _compute_errors(k_bf16, k_mxfp8)
        print(f"[K] b={batch_size} s={seq_len}: max_abs={max_abs_k:.6f}  rms={rms_k:.6f}")
        compare_and_assert(
            k_mxfp8,
            k_bf16,
            "k_mxfp8",
            "k_bf16",
            ATOL,
            RTOL,
            RMSE_TOL,
            True,
        )

        max_abs_v, rms_v = _compute_errors(v_bf16, v_mxfp8)
        print(f"[V] b={batch_size} s={seq_len}: max_abs={max_abs_v:.6f}  rms={rms_v:.6f}")
        compare_and_assert(
            v_mxfp8,
            v_bf16,
            "v_mxfp8",
            "v_bf16",
            ATOL,
            RTOL,
            RMSE_TOL,
            True,
        )

        max_abs_out, rms_out = _compute_errors(out_bf16, out_mxfp8)
        print(f"[OUT] b={batch_size} s={seq_len}: max_abs={max_abs_out:.6f}  rms={rms_out:.6f}")
        compare_and_assert(
            out_mxfp8,
            out_bf16,
            "out_mxfp8",
            "out_bf16",
            ATOL,
            RTOL,
            RMSE_TOL,
            True,
        )

    def test_backward(self, batch_size: int, seq_len: int) -> None:
        """Gradients must flow end-to-end without NaN/Inf."""
        fp8_recipe = MXFP8BlockScaling(fp8_dpa=True)
        _require_attention_backends(batch_size, seq_len, fp8_recipe)
        _set_seed()
        _, mxfp8_modules = _build_modules()

        x = torch.randn(
            seq_len,
            batch_size,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        rope_tables = build_rope_tables(seq_len, device=x.device)

        *_, out_mxfp8 = _run_forward_mxfp8(mxfp8_modules, x, fp8_recipe, rope_tables)
        out_mxfp8.sum().backward()

        assert x.grad is not None, "MXFP8 path: input grad is None"
        assert not torch.isnan(x.grad).any(), "MXFP8 path: input grad NaN"
        assert not torch.isinf(x.grad).any(), "MXFP8 path: input grad Inf"

        q_fp8, kv_fp8, _, out_fp8 = mxfp8_modules
        for name, mod in [("q_proj", q_fp8), ("kv_proj", kv_fp8), ("out_linear", out_fp8)]:
            for p in mod.parameters():
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any(), f"MXFP8 {name} param grad NaN"
                    assert not torch.isinf(p.grad).any(), f"MXFP8 {name} param grad Inf"

        dx_rms = x.grad.float().pow(2).mean().sqrt().item()
        print(f"\n[BPROP] b={batch_size} s={seq_len}: dx rms={dx_rms:.6f}")
        assert dx_rms > 0.0, "MXFP8 path: input grad is all zeros (no gradient flow)"

    def test_performance(self, batch_size: int, seq_len: int) -> None:
        """Benchmark the normal MXFP8 training step against BF16."""
        fp8_recipe = MXFP8BlockScaling(fp8_dpa=True)
        _require_attention_backends(batch_size, seq_len, fp8_recipe)
        _set_seed()
        baseline_modules, mxfp8_modules = _build_modules()
        x = torch.randn(
            seq_len,
            batch_size,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        rope_tables = build_rope_tables(seq_len, device=x.device)

        mxfp8_fprop_ms, mxfp8_bprop_ms = _benchmark_training_step(
            _run_forward_mxfp8, mxfp8_modules, x, fp8_recipe, rope_tables
        )
        mxfp8_fprop_tok = (batch_size * seq_len) / (mxfp8_fprop_ms / 1000.0)
        mxfp8_bprop_tok = (batch_size * seq_len) / (mxfp8_bprop_ms / 1000.0)

        bf16_fprop_ms, bf16_bprop_ms = _benchmark_training_step(
            _run_forward_bf16, baseline_modules, x, rope_tables
        )
        bf16_fprop_tok = (batch_size * seq_len) / (bf16_fprop_ms / 1000.0)
        bf16_bprop_tok = (batch_size * seq_len) / (bf16_bprop_ms / 1000.0)
        fprop_speedup = bf16_fprop_ms / mxfp8_fprop_ms
        bprop_speedup = bf16_bprop_ms / mxfp8_bprop_ms
        print(
            f"\n[PERF] b={batch_size} s={seq_len}:"
            f"\n  BF16 fprop:  {bf16_fprop_ms:.3f} ms  ({bf16_fprop_tok:.0f} tok/s)"
            f"\n  BF16 bprop:  {bf16_bprop_ms:.3f} ms  ({bf16_bprop_tok:.0f} tok/s)"
            f"\n  MXFP8 fprop: {mxfp8_fprop_ms:.3f} ms  ({mxfp8_fprop_tok:.0f} tok/s)"
            f"\n  MXFP8 bprop: {mxfp8_bprop_ms:.3f} ms  ({mxfp8_bprop_tok:.0f} tok/s)"
            f"\n  Fprop speedup: {fprop_speedup:.2f}x"
            f"\n  Bprop speedup: {bprop_speedup:.2f}x"
        )

        assert fprop_speedup > 1.0, (
            "MXFP8 fprop should be faster than BF16: "
            f"got {mxfp8_fprop_ms:.3f} ms vs BF16 {bf16_fprop_ms:.3f} ms "
            f"(speedup={fprop_speedup:.2f}x)"
        )
        assert bprop_speedup > 1.0, (
            "MXFP8 bprop should be faster than BF16: "
            f"got {mxfp8_bprop_ms:.3f} ms vs BF16 {bf16_bprop_ms:.3f} ms "
            f"(speedup={bprop_speedup:.2f}x)"
        )
