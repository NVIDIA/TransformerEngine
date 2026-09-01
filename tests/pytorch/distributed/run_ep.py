# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process PyTorch EP tests, launched via torchrun (one process per GPU)."""

from contextlib import nullcontext
from dataclasses import replace
import os
import sys
import unittest

import numpy as np
import torch
import torch.distributed as dist

from ep_reference import BlockScaledTensor, MoeEpReference, MoeFormat, quantize_blockwise
import transformer_engine.pytorch as te
from transformer_engine.pytorch import ops as te_ops
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.ep import (
    EpBuffer,
    EpConfig,
    ep_bootstrap,
    ep_finalize,
    get_ep_drop_on_overflow,
    get_ep_group,
    ep_prepare,
    ep_dispatch,
    ep_combine,
    symm_mem_alloc,
    release_symm_mem_pool,
    is_symm_backed,
    _ep_combine_raw,
    _ep_dispatch_raw,
)
from transformer_engine.pytorch.ops.fused.moe_ep import (
    FusedMoeEp,
    _cudnn_megamoe_supported,
    _get_megamoe_combine_format,
    _pack_cudnn_weights,
    is_moe_fusion_supported,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

ZERO_COPY = os.environ.get("NVTE_EP_ZERO_COPY", "0") == "1"
EAGER = os.environ.get("NVTE_EP_EAGER", "0") == "1"
OVERFLOW = os.environ.get("NVTE_EP_OVERFLOW", "0") == "1"

# Must come after the transformer_engine import so libtransformer_engine.so is loaded.
import transformer_engine_torch as tex  # noqa: F401

NUM_LOCAL_EXPERTS = 2
# MXFP8 dispatch needs HIDDEN_DIM % 512 == 0 and TOKENS_PER_RANK % 32 == 0. Defaults
# satisfy both so the MXFP8 tests run by default; override via NVTE_EP_HIDDEN_DIM /
# NVTE_EP_TOKENS_PER_RANK.
HIDDEN_DIM = int(os.environ.get("NVTE_EP_HIDDEN_DIM", "512"))
TOP_K = 2
TOKENS_PER_RANK = int(os.environ.get("NVTE_EP_TOKENS_PER_RANK", "32"))


def _zero_copy_test_include(fn):
    """Mark a test to also run in the zero-copy pass; others skip there."""
    fn._zero_copy_test_include = True
    return fn


def _eager_test_include(fn):
    """Mark a test to run in the eager pass; others skip there."""
    fn._eager_test_include = True
    return fn


def _overflow_test_include(fn):
    """Mark a test to run in the overflow (drop-on-overflow) pass; others skip there."""
    fn._overflow_test_include = True
    return fn


# MXFP8 grouped dispatch needs a per-expert alignment of 128, but the EP backend caches a single
# alignment per process, so alignment=128 tests cannot share a process with the alignment=0 tests.
# They run in a dedicated pass (NVTE_EP_MXFP8_PASS=1) instead.
MXFP8_PASS = os.environ.get("NVTE_EP_MXFP8_PASS", "0") == "1"


def _mxfp8_align_test(fn):
    """Mark a test that dispatches with alignment=128; runs only in the MXFP8 pass."""
    fn._mxfp8_align_test = True
    return fn


class _StageToSymm(torch.autograd.Function):
    """Identity op that stages ``src`` into a symm-mem buffer; grad passes through.
    Lets a test feed a symm-mem-backed, autograd-tracked tensor into ep_combine.
    """

    @staticmethod
    def forward(ctx, src, symm_buf):  # type: ignore[override]
        symm_buf.copy_(src)
        return symm_buf

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        return g, None


class _GradToSymm(torch.autograd.Function):
    """Identity fwd; bwd stages the upstream grad into a symm-mem buffer and
    returns it, so the next backward (dispatch_bwd) receives a symm-window grad
    input — which zero-copy ncclEpCombine requires.
    """

    @staticmethod
    def forward(ctx, x, symm_buf):  # type: ignore[override]
        ctx.symm_buf = symm_buf
        return x

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        ctx.symm_buf.copy_(g)
        return ctx.symm_buf, None


def _device_sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _build_ep_group():
    """EP group spanning all ranks of the default PG."""
    world_pg = dist.distributed_c10d._get_default_group()
    ranks = list(range(world_pg.size()))
    return dist.new_group(ranks=ranks, backend="nccl")


def _make_identity_inputs(rank, ep_size, device="cuda"):
    """Per-rank identity routing + uniform weights so combine matches tokens."""
    T = TOKENS_PER_RANK
    E = ep_size * NUM_LOCAL_EXPERTS
    topk_idx = np.empty((T, TOP_K), dtype=np.int64)
    base = rank * T
    for t in range(T):
        for k in range(TOP_K):
            topk_idx[t, k] = ((base + t) * TOP_K + k) % E
    tokens_np = np.linspace(
        0.1 + rank * 0.01, 0.9 + rank * 0.01, T * HIDDEN_DIM, dtype=np.float32
    ).reshape(T, HIDDEN_DIM)
    topk_weights = np.full((T, TOP_K), 1.0 / TOP_K, dtype=np.float32)
    return (
        torch.from_numpy(topk_idx).to(device),
        torch.from_numpy(tokens_np).to(device=device, dtype=torch.bfloat16),
        torch.from_numpy(topk_weights).to(device),
    )


def _make_moe_inputs(rank, ep_size, device="cuda"):
    """Deterministic BF16 activations and FP32 top-k router weights."""
    generator = torch.Generator(device=device)
    generator.manual_seed(2026 + rank)
    tokens = (
        torch.randn(
            TOKENS_PER_RANK,
            HIDDEN_DIM,
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        * 0.25
    ).to(torch.bfloat16)
    router_logits = torch.randn(
        TOKENS_PER_RANK,
        ep_size * NUM_LOCAL_EXPERTS,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    topk_logits, topk_idx = torch.topk(router_logits, TOP_K, dim=-1)
    return topk_idx, tokens, torch.softmax(topk_logits, dim=-1)


def _degroup_mxfp8(recv_grouped, valid_counts=None):
    """Dequantize a per-expert MXFP8 GroupedTensor to a dense tensor in expert-major order.
    With ``valid_counts`` keep only the first ``valid_counts[e]`` rows of each padded expert
    slot; otherwise return every (padded) row."""
    parts = recv_grouped.split_into_quantized_tensors()
    if valid_counts is None:
        return torch.cat([p.dequantize() for p in parts], dim=0)
    return torch.cat([p.dequantize()[:v] for p, v in zip(parts, valid_counts)], dim=0)


def _reference_weights(op):
    """Pack GroupedLinear weights into MoeEpReference ``(E, in, out)`` layout."""
    packed, _ = _pack_cudnn_weights(op, block_scaled_cls=BlockScaledTensor)
    if isinstance(packed, torch.Tensor):
        return packed.detach()
    return BlockScaledTensor(
        data=packed.data.detach(),
        scale=packed.scale.detach(),
        format=packed.format,
        logical_shape=packed.logical_shape,
        axis=packed.axis,
    )


class _Cfg:
    rank: int
    world_size: int
    ep_size: int
    num_experts: int
    recv_capacity_per_rank: int
    device: torch.device


def _make_cfg() -> _Cfg:
    cfg = _Cfg()
    cfg.rank = dist.get_rank()
    cfg.world_size = dist.get_world_size()
    cfg.ep_size = cfg.world_size
    cfg.num_experts = NUM_LOCAL_EXPERTS * cfg.ep_size
    T = TOKENS_PER_RANK
    active = min(cfg.num_experts, T * cfg.ep_size * TOP_K)
    overconc = cfg.num_experts // active
    cfg.recv_capacity_per_rank = NUM_LOCAL_EXPERTS * max(T * cfg.ep_size * TOP_K, 16) * overconc * 2
    if OVERFLOW:
        # Undersize recv capacity so identity routing overflows a rank's budget;
        # HT requires capacity >= max_tokens_per_rank.
        cfg.recv_capacity_per_rank = TOKENS_PER_RANK
    cfg.device = torch.device("cuda", torch.cuda.current_device())
    return cfg


class _EpTestCase(unittest.TestCase):
    """Shared NCCL EP process state and pass selection."""

    cfg: _Cfg
    ep_group: dist.ProcessGroup

    @classmethod
    def setUpClass(cls):
        if hasattr(_EpTestCase, "cfg"):
            cls.cfg = _EpTestCase.cfg
            cls.ep_group = _EpTestCase.ep_group
            return
        if _device_sm() < 90:
            raise unittest.SkipTest(f"NCCL EP requires SM>=90 (got SM{_device_sm()})")
        _EpTestCase.cfg = _make_cfg()
        _EpTestCase.ep_group = _build_ep_group()
        cls.cfg = _EpTestCase.cfg
        cls.ep_group = _EpTestCase.ep_group
        ep_bootstrap(
            cls.ep_group,
            num_experts=cls.cfg.num_experts,
            max_tokens_per_rank=TOKENS_PER_RANK,
            hidden_dim=HIDDEN_DIM,
            num_topk=TOP_K,
            # Omit recv_capacity_per_rank to select eager mode.
            recv_capacity_per_rank=None if EAGER else cls.cfg.recv_capacity_per_rank,
            zero_copy=ZERO_COPY,
            drop_on_overflow=OVERFLOW,
        )

    def setUp(self):
        # alignment=128 MXFP8 tests run only in the dedicated MXFP8 pass; everything else skips
        # there (and the MXFP8 tests skip outside it) since the backend pins one alignment/process.
        is_mxfp8_align = getattr(getattr(self, self._testMethodName), "_mxfp8_align_test", False)
        if MXFP8_PASS and not is_mxfp8_align:
            self.skipTest("only alignment=128 MXFP8 tests run in the MXFP8 pass")
        if not MXFP8_PASS and is_mxfp8_align:
            self.skipTest("alignment=128 MXFP8 tests run in the dedicated MXFP8 pass")
        # MXFP8 quantization requires Blackwell (SM 10.0) or newer.
        if is_mxfp8_align and torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("MXFP8 EP tests require Blackwell (SM 10.0) or newer")
        # Only the zero-copy-capable tests run in the zero-copy pass.
        if ZERO_COPY and not getattr(
            getattr(self, self._testMethodName), "_zero_copy_test_include", False
        ):
            self.skipTest("not exercised in zero-copy mode")
        # Only the eager-capable tests run in the eager pass.
        if EAGER and not getattr(getattr(self, self._testMethodName), "_eager_test_include", False):
            self.skipTest("not exercised in eager mode")
        # Only the overflow-capable tests run in the overflow pass.
        if OVERFLOW and not getattr(
            getattr(self, self._testMethodName), "_overflow_test_include", False
        ):
            self.skipTest("not exercised in overflow mode")

    def _make_config(
        self,
        alignment=0,
        top_k=TOP_K,
    ):
        return EpConfig(
            top_k=top_k,
            max_tokens_per_rank=TOKENS_PER_RANK,
            hidden_dim=HIDDEN_DIM,
            num_local_experts=NUM_LOCAL_EXPERTS,
            recv_capacity_per_rank=None if EAGER else self.cfg.recv_capacity_per_rank,
            ep_group=self.ep_group,
            alignment=alignment,
            zero_copy=ZERO_COPY,
            drop_on_overflow=OVERFLOW,
        )

    def _make_buffer_from_config(
        self,
        config,
        *,
        dispatch_fwd_quant_recipe=None,
        combine_bwd_quant_recipe=None,
    ):
        return EpBuffer(
            top_k=config.top_k,
            max_tokens_per_rank=config.max_tokens_per_rank,
            hidden_dim=config.hidden_dim,
            num_local_experts=config.num_local_experts,
            recv_capacity_per_rank=config.recv_capacity_per_rank,
            alignment=config.alignment,
            payload_dtype=config.payload_dtype,
            dispatch_fwd_quant_recipe=dispatch_fwd_quant_recipe,
            combine_bwd_quant_recipe=combine_bwd_quant_recipe,
        )

    def _make_buffer(
        self,
        alignment=0,
        top_k=TOP_K,
        dispatch_fwd_quant_recipe=None,
        combine_bwd_quant_recipe=None,
    ):
        config = self._make_config(alignment=alignment, top_k=top_k)
        return self._make_buffer_from_config(
            config,
            dispatch_fwd_quant_recipe=dispatch_fwd_quant_recipe,
            combine_bwd_quant_recipe=combine_bwd_quant_recipe,
        )


class TestEP(_EpTestCase):
    """NCCL EP tests inherited from nvidia_origin/main."""

    def test_bootstrap_accessors(self):
        self.assertIs(get_ep_group(), self.ep_group)
        self.assertEqual(get_ep_drop_on_overflow(), OVERFLOW)

    def _expert_out(self, expert_out):
        """Stage the combine input into symm-mem under zero-copy (combine requires it)."""
        if not ZERO_COPY:
            return expert_out
        symm_buf = symm_mem_alloc(
            tuple(expert_out.shape), expert_out.dtype, self.ep_group, use_pool=True
        )
        return _StageToSymm.apply(expert_out, symm_buf)

    def _stage_grad_symm(self, x, symm_buf=None):
        """Route x's upstream grad through a symm-mem buffer so dispatch_bwd gets
        a symm-window grad input under zero-copy; passthrough otherwise. Pass a
        pre-allocated symm_buf to avoid allocating during an interleaved schedule."""
        if not ZERO_COPY:
            return x
        if symm_buf is None:
            symm_buf = symm_mem_alloc(tuple(x.shape), x.dtype, self.ep_group)
        return _GradToSymm.apply(x, symm_buf)

    def _make_raw_recv(self, dtype=torch.bfloat16):
        """Raw recv tensors + tokens_per_expert for the primitive tests."""
        rc = self.cfg.recv_capacity_per_rank
        return (
            torch.empty(rc, HIDDEN_DIM, dtype=dtype, device=self.cfg.device),
            torch.empty(rc, dtype=torch.float32, device=self.cfg.device),
            torch.empty(NUM_LOCAL_EXPERTS, dtype=torch.int64, device=self.cfg.device),
        )

    @staticmethod
    def _weighted(recv_tokens, recv_w):
        """fp32 per-slot weighting + cast back; matches the upstream combine input."""
        mask = (recv_w != 0).to(torch.float32).unsqueeze(-1)
        return (recv_tokens.float() * recv_w.unsqueeze(-1).float() * mask).to(recv_tokens.dtype)

    def _moe_step(self, buffer, topk_idx, tokens, w):
        recv_t, recv_w_out, _tc = ep_dispatch(buffer, tokens, topk_idx, w)
        expert_out = self._weighted(recv_t, recv_w_out)
        return ep_combine(buffer, expert_out)

    # Prepare

    @_eager_test_include
    def test_primitive_prepare(self):
        buf = self._make_buffer()
        topk_idx, _toks, _w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_per_expert = ep_prepare(buf, topk_idx)
        torch.cuda.synchronize()
        self.assertEqual(tokens_per_expert.shape, (NUM_LOCAL_EXPERTS,))
        local = int(tokens_per_expert.sum().item())
        total = torch.tensor([local], dtype=torch.int64, device=self.cfg.device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=self.ep_group)
        self.assertEqual(int(total.item()), self.cfg.world_size * TOKENS_PER_RANK * TOP_K)

    @_eager_test_include
    def test_eager_recv_sizing(self):
        """Eager mode sizes dispatch outputs to the exact per-step recv-token total."""
        if not EAGER:
            self.skipTest("eager-only assertions")
        buf = self._make_buffer()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_t, recv_w, tokens_per_expert = ep_dispatch(buf, tokens, topk_idx, w)
        torch.cuda.synchronize()
        # The per-step recv-token total is exposed on the buffer (int64 [1]).
        self.assertEqual(buf.total_recv_tokens.dtype, torch.int64)
        total = int(buf.total_recv_tokens.item())
        # recv outputs are sized to the recv total, not recv_capacity_per_rank.
        self.assertEqual(recv_t.shape[0], total)
        self.assertEqual(recv_w.shape[0], total)
        # padded total is at least the unpadded per-expert sum and within capacity.
        self.assertGreaterEqual(total, int(tokens_per_expert.sum().item()))
        self.assertLessEqual(total, self.cfg.recv_capacity_per_rank)

    @_eager_test_include
    def test_eager_rank_with_zero_recv_tokens(self):
        """Empty recv tensors remain valid through the forward and backward pipeline."""
        if not EAGER:
            self.skipTest("eager-only assertions")
        buf = self._make_buffer()
        _topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        # Experts [0, TOP_K) are local to rank 0, so every other rank receives
        # no tokens and PyTorch gives its eager recv tensors null data pointers.
        topk_idx = torch.arange(TOP_K, dtype=torch.int64, device=self.cfg.device).repeat(
            TOKENS_PER_RANK, 1
        )
        tokens_p = tokens.detach().clone().requires_grad_(True)

        recv_t, recv_w, tokens_per_expert = ep_dispatch(buf, tokens_p, topk_idx, w)
        recv_rows = int(buf.total_recv_tokens.item())
        self.assertEqual(recv_t.shape, (recv_rows, HIDDEN_DIM))
        self.assertEqual(recv_w.shape, (recv_rows,))
        if self.cfg.rank == 0:
            self.assertGreater(recv_rows, 0)
            self.assertEqual(
                int(tokens_per_expert.sum().item()),
                self.cfg.world_size * TOKENS_PER_RANK * TOP_K,
            )
        else:
            self.assertEqual(recv_rows, 0)
            self.assertEqual(int(tokens_per_expert.sum().item()), 0)
            self.assertEqual(recv_t.data_ptr(), 0)
            self.assertEqual(recv_w.data_ptr(), 0)

        expert_out = self._weighted(recv_t, recv_w)
        result = ep_combine(buf, expert_out, num_local_tokens=TOKENS_PER_RANK)
        (0.5 * (result.float() ** 2).sum()).backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), tokens.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(tokens_p.grad.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    @_overflow_test_include
    def test_overflow_drop(self):
        """drop_on_overflow: recv past capacity is dropped and dispatch continues
        instead of trapping; the pre-drop recv total exceeds recv_capacity."""
        if not OVERFLOW:
            self.skipTest("overflow-only assertions")
        buf = self._make_buffer()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        # Identity routing sends TOKENS_PER_RANK * TOP_K tokens to each rank, which
        # overflows the deliberately undersized capacity.
        expected_recv = TOKENS_PER_RANK * TOP_K
        self.assertGreater(expected_recv, self.cfg.recv_capacity_per_rank)
        # total_recv_tokens reports the true (pre-drop) recv total, counting the
        # tokens that will be dropped; the per-expert counts exclude them and sum
        # to the kept tokens (capped at recv_capacity_per_rank).
        tokens_per_expert = ep_prepare(buf, topk_idx)
        torch.cuda.synchronize()
        self.assertEqual(int(buf.total_recv_tokens.item()), expected_recv)
        self.assertEqual(int(tokens_per_expert.sum().item()), self.cfg.recv_capacity_per_rank)
        # Dispatch drops overflowing tokens and completes (no trap); recv outputs
        # stay capped at recv_capacity_per_rank.
        recv_t, recv_w, _ = ep_dispatch(buf, tokens, topk_idx, w)
        torch.cuda.synchronize()
        self.assertEqual(recv_t.shape[0], self.cfg.recv_capacity_per_rank)
        self.assertEqual(recv_w.shape[0], self.cfg.recv_capacity_per_rank)

    # Identity round-trip via raw primitives

    def test_primitive_dispatch_combine_identity(self):
        buf = self._make_buffer()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_tokens, recv_w, _ = self._make_raw_recv()
        ep_prepare(buf, topk_idx)
        _ep_dispatch_raw(buf, topk_idx, tokens, w, recv_tokens, recv_w)
        result = torch.empty_like(tokens)
        _ep_combine_raw(buf, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    # Autograd

    @_zero_copy_test_include
    def test_dispatch_autograd(self):
        """0.5*||recv_tokens||^2 ; grad_tokens equals TOP_K * tokens. Covers the
        EpBuffer-owned recv tokens (symm-mem under zero-copy) and, in normal
        mode, a caller-supplied recv_tokens buffer."""
        if ZERO_COPY:
            cases = [("buffer_owned", None)]
        else:
            rt_buf, _rw_buf, _ = self._make_raw_recv()
            cases = [
                ("default_alloc", None),
                ("caller_recv", rt_buf),
            ]
        for label, recv_tokens in cases:
            with self.subTest(case=label):
                buf = self._make_buffer()
                topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
                tokens_p = tokens.detach().clone().requires_grad_(True)
                rt, rw, _tc = ep_dispatch(buf, tokens_p, topk_idx, w, recv_tokens=recv_tokens)
                if recv_tokens is not None:  # caller-supplied recv_tokens must be used in place
                    self.assertEqual(rt.data_ptr(), recv_tokens.data_ptr())
                rt = self._stage_grad_symm(rt)
                rw = self._stage_grad_symm(rw)
                (0.5 * (rt.float() ** 2).sum() + 0.0 * rw.float().sum()).backward()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    tokens_p.grad.float(), tokens.float() * float(TOP_K), atol=5e-2, rtol=5e-2
                )

    # MXFP8 dispatch

    def _mxfp8_quantizer(self):
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)

    def _require_mxfp8_shapes(self):
        if HIDDEN_DIM % 512 != 0 or TOKENS_PER_RANK % 32 != 0:
            self.skipTest(
                "MXFP8 needs HIDDEN_DIM % 512 == 0 and TOKENS_PER_RANK % 32 == 0 "
                "(set NVTE_EP_HIDDEN_DIM / NVTE_EP_TOKENS_PER_RANK)"
            )

    def _assert_mxfp8_matches_bf16(self, recv_mx, tokens, topk_idx, w, tc):
        """Dequantized MXFP8 recv matches a bf16 dispatch of the same tokens. Both share the
        alignment=128 padded expert-major layout, so compare the full prefix [0:sum(padded)]."""
        ref_tokens = self._mxfp8_quantizer().quantize(tokens).dequantize()
        ref_recv, _rw, _tc = ep_dispatch(self._make_buffer(alignment=128), ref_tokens, topk_idx, w)
        torch.cuda.synchronize()
        n = int(tc.sum())
        torch.testing.assert_close(
            _degroup_mxfp8(recv_mx).float(), ref_recv.float()[:n], atol=1e-2, rtol=1e-2
        )

    @_eager_test_include
    @_zero_copy_test_include
    @_mxfp8_align_test
    def test_dispatch_mxfp8(self):
        """MXFP8 dispatch quantizes bf16 tokens internally; recv (a per-expert GroupedTensor)
        dequantized matches a bf16 dispatch of the same tokens. Under zero-copy the recv data and
        scales are symm-mem backed."""
        self._require_mxfp8_shapes()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        buf = self._make_buffer(dispatch_fwd_quant_recipe=MXFP8BlockScaling(), alignment=128)
        recv_mx, _rw, tc = ep_dispatch(buf, tokens, topk_idx, w)
        if ZERO_COPY:
            self.assertTrue(is_symm_backed(recv_mx.rowwise_data))
            self.assertTrue(is_symm_backed(recv_mx.scale_inv))
        self._assert_mxfp8_matches_bf16(recv_mx, tokens, topk_idx, w, tc)

    @_eager_test_include
    @_mxfp8_align_test
    def test_dispatch_mxfp8_autograd(self):
        """MXFP8 dispatch fwd+bwd. Seeding the recv grad with ones scatters TOP_K back to each token
        under identity routing, so grad_tokens equals TOP_K through the dispatch backward and the
        input quantizer STE. Exercises the fused eager MXFP8 dispatch backward."""
        self._require_mxfp8_shapes()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        buf = self._make_buffer(dispatch_fwd_quant_recipe=MXFP8BlockScaling(), alignment=128)
        recv_mx, _rw, _tc = ep_dispatch(buf, tokens_p, topk_idx, w)
        g_recv = torch.ones(recv_mx.shape, dtype=torch.bfloat16, device=self.cfg.device)
        torch.autograd.backward(recv_mx, grad_tensors=g_recv)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            tokens_p.grad.float(),
            torch.full_like(tokens_p, float(TOP_K)).float(),
            atol=5e-2,
            rtol=5e-2,
        )

    @_zero_copy_test_include
    @_mxfp8_align_test
    def test_caller_provides_dispatch_recv_mxfp8(self):
        """One caller-supplied buffer holds the recv data followed by the e8m0 scales; ep_dispatch
        slices it and the returned GroupedTensor views the data and scale regions of that buffer."""
        self._require_mxfp8_shapes()
        from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE

        rc = self.cfg.recv_capacity_per_rank
        cols = HIDDEN_DIM // MXFP8_BLOCK_SCALING_SIZE
        nbytes = rc * (HIDDEN_DIM + cols)  # fp8 data + e8m0 scales, one byte per element
        if ZERO_COPY:
            recv_buf = symm_mem_alloc((nbytes,), torch.uint8, self.ep_group)
        else:
            recv_buf = torch.empty(nbytes, dtype=torch.uint8, device=self.cfg.device)
        buf = self._make_buffer(dispatch_fwd_quant_recipe=MXFP8BlockScaling(), alignment=128)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_mx, _rw, tc = ep_dispatch(buf, tokens, topk_idx, w, recv_tokens=recv_buf)
        # the returned GroupedTensor views the caller buffer's data then scale regions
        self.assertEqual(recv_mx.rowwise_data.data_ptr(), recv_buf.data_ptr())
        self.assertEqual(recv_mx.scale_inv.data_ptr(), recv_buf.data_ptr() + rc * HIDDEN_DIM)
        self._assert_mxfp8_matches_bf16(recv_mx, tokens, topk_idx, w, tc)

    @_zero_copy_test_include
    def test_caller_provides_dispatch_recv_tokens(self):
        """Caller-supplied recv_tokens (symm-mem-backed in zero-copy): ep_dispatch
        writes into it and returns a view of the caller's buffer."""
        if ZERO_COPY:
            rc = self.cfg.recv_capacity_per_rank
            rt_buf = symm_mem_alloc((rc, HIDDEN_DIM), torch.bfloat16, self.ep_group)
        else:
            rt_buf, _rw_buf, _ = self._make_raw_recv()
        buf = self._make_buffer()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        rt, rw, _ = ep_dispatch(buf, tokens_p, topk_idx, w, recv_tokens=rt_buf)
        self.assertEqual(rt.data_ptr(), rt_buf.data_ptr())
        rt = self._stage_grad_symm(rt)
        rw = self._stage_grad_symm(rw)
        (0.5 * (rt.float() ** 2).sum() + 0.0 * rw.float().sum()).backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            tokens_p.grad.float(), tokens.float() * float(TOP_K), atol=5e-2, rtol=5e-2
        )

    @_zero_copy_test_include
    def test_caller_provides_grad_expert_out(self):
        """Caller-supplied grad_out (symm-mem-backed in zero-copy): ep_combine's
        backward scatters the expert-out grad into it."""
        rc = self.cfg.recv_capacity_per_rank
        if ZERO_COPY:
            gbuf = symm_mem_alloc((rc, HIDDEN_DIM), torch.bfloat16, self.ep_group)
        else:
            gbuf = torch.empty(rc, HIDDEN_DIM, dtype=torch.bfloat16, device=self.cfg.device)
        gbuf.zero_()
        buf = self._make_buffer()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        recv_t, recv_w, _ = ep_dispatch(buf, tokens_p, topk_idx, w)
        recv_t = self._stage_grad_symm(recv_t)
        recv_w = self._stage_grad_symm(recv_w)
        expert_out = self._expert_out(self._weighted(recv_t, recv_w))
        out = ep_combine(buf, expert_out, grad_out=gbuf)
        (0.5 * (out.float() ** 2).sum()).backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), tokens.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(tokens_p.grad.float(), tokens.float(), atol=5e-2, rtol=5e-2)
        # the caller-owned buffer was used as the combine-bwd scatter target
        self.assertGreater(gbuf.abs().sum().item(), 0.0)

    @_zero_copy_test_include
    @_mxfp8_align_test
    def test_combine_bwd_mxfp8_caller_grad_out(self):
        """MXFP8 combine backward into a single caller buffer sliced into data + e8m0 scales: the
        returned per-expert GroupedTensor views those regions and, dequantized, matches a bf16
        combine backward reference on the same routing. Under zero-copy the caller buffer and combine
        input are symm-mem backed."""
        self._require_mxfp8_shapes()
        from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE

        rc = self.cfg.recv_capacity_per_rank
        cols = HIDDEN_DIM // MXFP8_BLOCK_SCALING_SIZE
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        eo_vals = (
            torch.linspace(-0.5, 0.5, rc * HIDDEN_DIM, device=self.cfg.device)
            .reshape(rc, HIDDEN_DIM)
            .to(torch.bfloat16)
        )
        # MXFP8 combine backward writes into one caller buffer (data then e8m0 scales)
        buf_mx = self._make_buffer(combine_bwd_quant_recipe=MXFP8BlockScaling(), alignment=128)
        _recv, _rw, tc = ep_dispatch(buf_mx, tokens, topk_idx, w)  # seeds the routing
        nbytes = rc * (HIDDEN_DIM + cols)
        if ZERO_COPY:
            grad_buf = symm_mem_alloc((nbytes,), torch.uint8, self.ep_group)
        else:
            grad_buf = torch.empty(nbytes, dtype=torch.uint8, device=self.cfg.device)
        src_mx = eo_vals.detach().clone().requires_grad_(True)
        out_mx = ep_combine(buf_mx, self._expert_out(src_mx), grad_out=grad_buf)
        (0.5 * (out_mx.float() ** 2).sum()).backward()
        g_mx = src_mx.grad  # per-expert GroupedTensor viewing grad_buf
        self.assertEqual(g_mx.rowwise_data.data_ptr(), grad_buf.data_ptr())
        self.assertEqual(g_mx.scale_inv.data_ptr(), grad_buf.data_ptr() + rc * HIDDEN_DIM)
        # bf16 reference combine backward on the same routing
        buf_bf = self._make_buffer(alignment=128)
        ep_dispatch(buf_bf, tokens, topk_idx, w)
        src_bf = eo_vals.detach().clone().requires_grad_(True)
        out_bf = ep_combine(buf_bf, self._expert_out(src_bf))
        (0.5 * (out_bf.float() ** 2).sum()).backward()
        torch.cuda.synchronize()
        n = int(tc.sum())
        torch.testing.assert_close(
            _degroup_mxfp8(g_mx).float(), src_bf.grad.float()[:n], atol=5e-2, rtol=5e-2
        )

    @_eager_test_include
    @_zero_copy_test_include
    @_mxfp8_align_test
    def test_combine_bwd_mxfp8(self):
        """MXFP8 combine backward with an internally allocated grad target: the returned per-expert
        GroupedTensor, dequantized, matches a bf16 combine backward reference on the same routing.
        Under zero-copy the combine input is symm-mem backed.
        """
        self._require_mxfp8_shapes()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        buf_mx = self._make_buffer(combine_bwd_quant_recipe=MXFP8BlockScaling(), alignment=128)
        _recv, _rw, tc = ep_dispatch(buf_mx, tokens, topk_idx, w)  # seeds the routing
        # Combine input rows match the recv total (per-step in eager, capacity otherwise).
        rows = int(buf_mx.total_recv_tokens.item()) if EAGER else self.cfg.recv_capacity_per_rank
        eo_vals = (
            torch.linspace(-0.5, 0.5, rows * HIDDEN_DIM, device=self.cfg.device)
            .reshape(rows, HIDDEN_DIM)
            .to(torch.bfloat16)
        )
        src_mx = eo_vals.detach().clone().requires_grad_(True)
        out_mx = ep_combine(buf_mx, self._expert_out(src_mx))
        (0.5 * (out_mx.float() ** 2).sum()).backward()
        g_mx = src_mx.grad  # per-expert GroupedTensor
        # bf16 reference combine backward on the same routing
        buf_bf = self._make_buffer(alignment=128)
        ep_dispatch(buf_bf, tokens, topk_idx, w)
        src_bf = eo_vals.detach().clone().requires_grad_(True)
        out_bf = ep_combine(buf_bf, self._expert_out(src_bf))
        (0.5 * (out_bf.float() ** 2).sum()).backward()
        torch.cuda.synchronize()
        n = int(tc.sum())
        torch.testing.assert_close(
            _degroup_mxfp8(g_mx).float(), src_bf.grad.float()[:n], atol=5e-2, rtol=5e-2
        )

    @_zero_copy_test_include
    def test_zero_copy_pool_auto_alloc(self):
        """Zero-copy with recv/grad left None: ep_dispatch/ep_combine allocate their IO
        tensors from the symm-mem pool (is_symm_backed). This is the primary mcore
        path — mcore hands no caller buffers, TE pools them on the fly."""
        if not ZERO_COPY:
            self.skipTest("zero-copy pool auto-alloc only")
        buf = self._make_buffer()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        recv_t, recv_w, _ = ep_dispatch(buf, tokens_p, topk_idx, w)  # recv_tokens=None -> pool
        self.assertTrue(is_symm_backed(recv_t))  # dispatch recv came from the symm-mem pool
        recv_t = self._stage_grad_symm(recv_t)
        recv_w = self._stage_grad_symm(recv_w)
        expert_out = self._expert_out(self._weighted(recv_t, recv_w))
        out = ep_combine(buf, expert_out)  # grad_out=None -> bwd allocs the grad from the pool
        (0.5 * (out.float() ** 2).sum()).backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), tokens.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(tokens_p.grad.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    # Multi-iter stability

    @_eager_test_include
    def test_dispatch_autograd_multiple_iterations(self):
        """5 fwd+bwd iters on the same EpBuffer must be bit-stable."""
        buf = self._make_buffer()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)

        def one_step():
            tokens_p = tokens.detach().clone().requires_grad_(True)
            out = self._moe_step(buf, topk_idx, tokens_p, w)
            loss = 0.5 * (out.float() ** 2).sum()
            loss.backward()
            return out.detach().clone(), tokens_p.grad.detach().clone()

        out_ref, grad_ref = one_step()
        torch.cuda.synchronize()
        for _ in range(4):
            out_i, grad_i = one_step()
            torch.cuda.synchronize()
            torch.testing.assert_close(out_i, out_ref, atol=0, rtol=0)
            torch.testing.assert_close(grad_i, grad_ref, atol=0, rtol=0)

    # CUDA graph

    def test_cuda_graph_capture(self):
        """Capture raw dispatch+combine into a CUDA graph; replay must be bit-stable."""
        buf = self._make_buffer()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_tokens, recv_w, _ = self._make_raw_recv()
        result = torch.empty_like(tokens)

        def step():
            ep_prepare(buf, topk_idx)
            _ep_dispatch_raw(buf, topk_idx, tokens, w, recv_tokens, recv_w)
            _ep_combine_raw(buf, self._weighted(recv_tokens, recv_w), result)

        for _ in range(3):
            step()
        torch.cuda.synchronize()

        # Routing is fixed per layer; prepare runs once before capture.
        ep_prepare(buf, topk_idx)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(graph):
                _ep_dispatch_raw(buf, topk_idx, tokens, w, recv_tokens, recv_w)
                _ep_combine_raw(buf, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        ref = result.clone()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), ref.float(), atol=0, rtol=0)

    # PP-1F1B handle isolation

    @_zero_copy_test_include
    def test_pp_1f1b_two_handles(self):
        """PP-1F1B interleave (F0 F1 B0 F2 B1 B2) over 3 per-microbatch buffers,
        run eagerly and replayed from a CUDA graph capturing the full fwd+bwd
        schedule (prepare included; routing is fixed so replay reproduces it)."""
        for capture in (False, True):
            with self.subTest(capture=capture):
                self._run_1f1b(capture)

    def _run_1f1b(self, capture):
        T, H = TOKENS_PER_RANK, HIDDEN_DIM
        idx, _toks, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        scales = (0.13, 0.41, 0.77)
        buffers, tokens, tokens_p = [], [], []
        for s in scales:
            buffers.append(self._make_buffer())
            t = torch.full(
                (T, H), s + self.cfg.rank * 0.01, dtype=torch.bfloat16, device=self.cfg.device
            )
            tokens.append(t)
            tokens_p.append(t.detach().clone().requires_grad_(True))

        recv = [None, None, None]
        # Per-microbatch grad-staging buffers, symm-mem under zero-copy and
        # pre-allocated so nothing is allocated/freed mid-interleave.
        recv_w = [None, None, None]
        rc = self.cfg.recv_capacity_per_rank
        if ZERO_COPY:
            gbuf_t = [symm_mem_alloc((rc, H), torch.bfloat16, self.ep_group) for _ in scales]
            gbuf_w = [symm_mem_alloc((rc,), torch.float32, self.ep_group) for _ in scales]
            # Persistent symm-mem recv buffers per microbatch: leaving recv None
            # pool-allocates, which is not CUDA-graph capturable.
            rbuf_t = [symm_mem_alloc((rc, H), torch.bfloat16, self.ep_group) for _ in scales]
            rbuf_w = [symm_mem_alloc((rc,), torch.float32, self.ep_group) for _ in scales]
        else:
            gbuf_t = gbuf_w = [None, None, None]
            rbuf_t = rbuf_w = [None, None, None]

        def fwd(k):
            rt, rw, _ = ep_dispatch(
                buffers[k],
                tokens_p[k],
                idx,
                w,
                recv_tokens=rbuf_t[k],
                recv_topk_weights=rbuf_w[k],
            )
            recv[k] = self._stage_grad_symm(rt, gbuf_t[k])
            recv_w[k] = self._stage_grad_symm(rw, gbuf_w[k])

        def bwd(k):
            (0.5 * (recv[k].float() ** 2).sum() + 0.0 * recv_w[k].float().sum()).backward()
            recv[k] = None
            recv_w[k] = None

        def interleave():
            fwd(0)
            fwd(1)
            bwd(0)
            fwd(2)
            bwd(1)
            bwd(2)

        def zero_grads():
            for tp in tokens_p:
                if tp.grad is not None:
                    tp.grad.zero_()

        if not capture:
            interleave()
        else:
            # Warmup on a side stream, then capture the full schedule and replay.
            # Grads stay pre-allocated (zeroed, not None) so backward accumulates
            # in place during both capture and replay.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    zero_grads()
                    interleave()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            zero_grads()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                interleave()
            zero_grads()
            graph.replay()

        torch.cuda.synchronize()
        for k in range(3):
            torch.testing.assert_close(
                tokens_p[k].grad.float(),
                tokens[k].float() * float(TOP_K),
                atol=5e-2,
                rtol=5e-2,
            )

    @_zero_copy_test_include
    @_eager_test_include
    def test_combine_autograd(self):
        """ep_combine fwd+bwd; bwd grad target is the EpBuffer symm buffer (zc) or in-flight."""
        buf = self._make_buffer()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        recv_t, recv_w, _ = ep_dispatch(buf, tokens_p, topk_idx, w)
        recv_t = self._stage_grad_symm(recv_t)
        recv_w = self._stage_grad_symm(recv_w)
        expert_out = self._expert_out(self._weighted(recv_t, recv_w))
        out = ep_combine(buf, expert_out)
        (0.5 * (out.float() ** 2).sum()).backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), tokens.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(tokens_p.grad.float(), tokens.float(), atol=5e-2, rtol=5e-2)


class TestMoeEpSequential(_EpTestCase):
    """Integration tests for Dispatch -> expert MLP -> Combine sequences."""

    def _mxfp8_quantizer(self):
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)

    def _require_mxfp8_shapes(self):
        if HIDDEN_DIM % 512 != 0 or TOKENS_PER_RANK % 32 != 0:
            self.skipTest(
                "MXFP8 needs HIDDEN_DIM % 512 == 0 and TOKENS_PER_RANK % 32 == 0 "
                "(set NVTE_EP_HIDDEN_DIM / NVTE_EP_TOKENS_PER_RANK)"
            )

    def test_runtime_buffer_config_mismatch(self):
        config = self._make_config()
        buffer = self._make_buffer_from_config(config)
        topk_idx, tokens, topk_weights = _make_identity_inputs(
            self.cfg.rank,
            self.cfg.ep_size,
        )
        dispatch = te_ops.MoeDispatch(config, buffer)
        replacements = {
            "top_k": config.top_k + 1,
            "hidden_dim": config.hidden_dim + 1,
            "num_local_experts": config.num_local_experts + 1,
            "max_tokens_per_rank": config.max_tokens_per_rank + 1,
            "recv_capacity_per_rank": config.recv_capacity_per_rank + 1,
            "alignment": 1,
            "payload_dtype": torch.float16,
            "zero_copy": not config.zero_copy,
        }
        for field_name, wrong_value in replacements.items():
            with self.subTest(field_name=field_name):
                original = getattr(buffer, field_name)
                setattr(buffer, field_name, wrong_value)
                try:
                    with self.assertRaisesRegex(ValueError, field_name):
                        dispatch(tokens, topk_idx, topk_weights)
                finally:
                    setattr(buffer, field_name, original)

        wrong_config = replace(config, drop_on_overflow=not config.drop_on_overflow)
        with self.assertRaisesRegex(ValueError, "drop_on_overflow"):
            te_ops.MoeDispatch(wrong_config, buffer)(tokens, topk_idx, topk_weights)
        with self.assertRaisesRegex(ValueError, "ep_group"):
            te_ops.MoeDispatch(replace(config, ep_group=None), buffer)(
                tokens, topk_idx, topk_weights
            )
        expert_out = torch.empty(
            self.cfg.recv_capacity_per_rank,
            HIDDEN_DIM,
            dtype=torch.bfloat16,
            device=self.cfg.device,
        )
        with self.assertRaisesRegex(ValueError, "buffer config"):
            original = buffer.hidden_dim
            buffer.hidden_dim += 1
            try:
                te_ops.MoeCombine(config, buffer)(expert_out)
            finally:
                buffer.hidden_dim = original

    def test_megamoe_combine_format_env(self):
        old_value = os.environ.get("NVTE_MEGAMOE_MXFP8_COMBINE")
        try:
            os.environ["NVTE_MEGAMOE_MXFP8_COMBINE"] = "0"
            self.assertEqual(_get_megamoe_combine_format(), "bf16")
            os.environ["NVTE_MEGAMOE_MXFP8_COMBINE"] = "1"
            self.assertEqual(_get_megamoe_combine_format(), "mxfp8")
        finally:
            if old_value is None:
                os.environ.pop("NVTE_MEGAMOE_MXFP8_COMBINE", None)
            else:
                os.environ["NVTE_MEGAMOE_MXFP8_COMBINE"] = old_value

    @_mxfp8_align_test
    def test_role_quantizer_requires_matching_buffer_recipe(self):
        self._require_mxfp8_shapes()
        config = self._make_config(alignment=128)
        buffer = self._make_buffer_from_config(config)
        dispatch = te_ops.MoeDispatch(config, buffer)
        combine = te_ops.MoeCombine(config, buffer)
        topk_idx, tokens, topk_weights = _make_identity_inputs(
            self.cfg.rank,
            self.cfg.ep_size,
        )
        recipe = MXFP8BlockScaling()
        with te.autocast(enabled=True, recipe=recipe):
            with self.assertRaisesRegex(ValueError, "does not have an MXFP8BlockScaling recipe"):
                dispatch(tokens, topk_idx, topk_weights)
            expert_out = torch.empty(
                self.cfg.recv_capacity_per_rank,
                HIDDEN_DIM,
                dtype=torch.bfloat16,
                device=self.cfg.device,
                requires_grad=True,
            )
            with self.assertRaisesRegex(ValueError, "does not have an MXFP8BlockScaling recipe"):
                combine(expert_out)

    def _make_dispatch_combine_ops(self, *, mxfp8):
        recipe = MXFP8BlockScaling() if mxfp8 else None
        config = self._make_config(alignment=128 if mxfp8 else 0)
        buffer = self._make_buffer_from_config(
            config,
            dispatch_fwd_quant_recipe=recipe,
            combine_bwd_quant_recipe=recipe,
        )
        return (
            buffer,
            te_ops.MoeDispatch(config, buffer),
            te_ops.MoeCombine(config, buffer),
        )

    def _run_dispatch_combine_identity(self, *, mxfp8):
        """Route, apply top-k weights, and combine back to local token order."""
        if mxfp8:
            self._require_mxfp8_shapes()
        buffer, dispatch, combine = self._make_dispatch_combine_ops(mxfp8=mxfp8)
        topk_idx, tokens, topk_weights = _make_identity_inputs(
            self.cfg.rank,
            self.cfg.ep_size,
        )
        recipe = MXFP8BlockScaling() if mxfp8 else None
        with te.autocast(enabled=mxfp8, recipe=recipe):
            recv_tokens, tokens_per_expert, recv_weights = dispatch(
                tokens,
                topk_idx,
                topk_weights,
            )
            if mxfp8:
                recv_tokens = _degroup_mxfp8(recv_tokens)
                recv_weights = recv_weights[: recv_tokens.shape[0]]
            weighted_expert_output = (recv_tokens.float() * recv_weights.float().unsqueeze(-1)).to(
                torch.bfloat16
            )
            output = combine(weighted_expert_output)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, tokens, atol=5e-2, rtol=5e-2)
        self.assertEqual(tokens_per_expert.data_ptr(), buffer.tokens_per_expert.data_ptr())

    @_eager_test_include
    def test_dispatch_combine_identity_bf16(self):
        """MoeDispatch and MoeCombine basic ops form an identity in BF16."""
        self._run_dispatch_combine_identity(mxfp8=False)

    @_eager_test_include
    @_mxfp8_align_test
    def test_dispatch_combine_identity_mxfp8(self):
        """MoeDispatch and MoeCombine basic ops form an identity with MXFP8 transport."""
        self._run_dispatch_combine_identity(mxfp8=True)

    def _make_megamoe_model(
        self,
        *,
        recipe,
        accumulate_into_main_grad=False,
        delay_wgrad_compute=False,
        glu_interleave_size=None,
    ):
        """Build the exact five-op sequence recognized by MegaMoE fusion."""
        config = self._make_config(alignment=128 if recipe is not None else 0)
        buffer = self._make_buffer_from_config(
            config,
            dispatch_fwd_quant_recipe=recipe,
            combine_bwd_quant_recipe=recipe,
        )
        dispatch = te_ops.MoeDispatch(config, buffer)
        init_ctx = (
            te.quantized_model_init(enabled=True, recipe=recipe)
            if recipe is not None
            else nullcontext()
        )
        previous_single_param = os.environ.get("NVTE_GROUPED_LINEAR_SINGLE_PARAM")
        os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
        try:
            with init_ctx:
                fc1 = te_ops.GroupedLinear(
                    NUM_LOCAL_EXPERTS,
                    HIDDEN_DIM,
                    2 * 256,
                    bias=False,
                    device=self.cfg.device,
                    dtype=torch.bfloat16,
                    single_grouped_weight=True,
                    accumulate_into_main_grad=accumulate_into_main_grad,
                    delay_wgrad_compute=delay_wgrad_compute,
                )
                activation = te_ops.ScaledSwiGLU(
                    glu_interleave_size=glu_interleave_size,
                )
                fc2 = te_ops.GroupedLinear(
                    NUM_LOCAL_EXPERTS,
                    256,
                    HIDDEN_DIM,
                    bias=False,
                    device=self.cfg.device,
                    dtype=torch.bfloat16,
                    single_grouped_weight=True,
                    accumulate_into_main_grad=accumulate_into_main_grad,
                    delay_wgrad_compute=delay_wgrad_compute,
                )
        finally:
            if previous_single_param is None:
                del os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"]
            else:
                os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = previous_single_param
        combine = te_ops.MoeCombine(config, buffer)

        dispatch.set_extra_output_channel(0, "tokens_per_expert", output_to_caller=False)
        dispatch.set_extra_output_channel(1, "routing_weights", output_to_caller=False)
        fc1.set_extra_input_channel(0, "tokens_per_expert")
        activation.set_extra_input_channel(0, "routing_weights")
        fc2.set_extra_input_channel(0, "tokens_per_expert")
        model = te_ops.Sequential(dispatch, fc1, activation, fc2, combine)
        return model, fc1, fc2, buffer

    @_eager_test_include
    def test_megamoe_bf16_numerics(self):
        self._run_megamoe_vs_reference(quantization="bf16")

    @_eager_test_include
    @_mxfp8_align_test
    def test_megamoe_mxfp8_numerics(self):
        self._run_megamoe_vs_reference(quantization="mxfp8")

    @_mxfp8_align_test
    def test_megamoe_mxfp8_cuda_graph_matches_eager(self):
        """Fused Sequential forward/backward graph replay matches eager execution."""
        if torch.cuda.get_device_capability() != (10, 7):
            self.skipTest("FusedMoeEp CUDA graph test requires SM107")
        if not _cudnn_megamoe_supported():
            self.skipTest("installed cuDNN frontend does not provide fixed training resources")
        self._run_megamoe_mxfp8_cuda_graph_matches_eager(
            glu_interleave_size=32,
            expect_fused=True,
        )

    @_mxfp8_align_test
    def test_unfused_megamoe_mxfp8_cuda_graph_matches_eager(self):
        """Unfused Sequential forward/backward graph replay matches eager execution."""
        self._run_megamoe_mxfp8_cuda_graph_matches_eager(
            glu_interleave_size=None,
            expect_fused=False,
        )

    def _run_megamoe_mxfp8_cuda_graph_matches_eager(
        self,
        *,
        glu_interleave_size,
        expect_fused,
    ):
        recipe = MXFP8BlockScaling()
        graph_model, graph_fc1, graph_fc2, _ = self._make_megamoe_model(
            recipe=recipe,
            glu_interleave_size=glu_interleave_size,
        )
        fusion_supported = is_moe_fusion_supported(tuple(graph_model), recipe)
        if expect_fused and not fusion_supported:
            self.skipTest("current configuration does not support FusedMoeEp")
        self.assertEqual(fusion_supported, expect_fused)
        eager_model, eager_fc1, eager_fc2, _ = self._make_megamoe_model(
            recipe=recipe,
            glu_interleave_size=glu_interleave_size,
        )
        eager_model.load_state_dict(graph_model.state_dict())

        topk_idx, tokens, topk_weights = _make_moe_inputs(
            self.cfg.rank,
            self.cfg.ep_size,
            self.cfg.device,
        )

        static_tokens = tokens.detach().clone().requires_grad_(True)
        static_topk_idx = topk_idx.detach().clone()
        static_topk_weights = topk_weights.detach().clone().requires_grad_(True)
        static_dy = torch.randn_like(static_tokens)
        graphed_model = te.make_graphed_callables(
            graph_model,
            (static_tokens, static_topk_idx, static_topk_weights),
            num_warmup_iters=3,
            enabled=True,
            recipe=recipe,
        )

        forward_ops = graph_model._module_groups[0]._forward_ops
        backward_ops = graph_model._module_groups[0]._backward_ops
        fused_ops = [op for op, _ in forward_ops if isinstance(op, FusedMoeEp)]
        if expect_fused:
            self.assertEqual(len(forward_ops), 1)
            self.assertEqual(len(backward_ops), 1)
            self.assertEqual(len(fused_ops), 1)
            self.assertIs(backward_ops[0][0], fused_ops[0])
        else:
            self.assertFalse(fused_ops)

        # Replace the capture-time contents while retaining captured addresses.
        with torch.no_grad():
            static_tokens.copy_(torch.randn_like(static_tokens))
            static_topk_weights.copy_(torch.rand_like(static_topk_weights))
            static_dy.copy_(torch.randn_like(static_dy))

        for parameter in graph_model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        if static_tokens.grad is not None:
            static_tokens.grad.zero_()
        if static_topk_weights.grad is not None:
            static_topk_weights.grad.zero_()
        with te.autocast(enabled=True, recipe=recipe):
            graph_out = graphed_model(
                static_tokens,
                static_topk_idx,
                static_topk_weights,
            )
        graph_out_snapshot = graph_out.detach().clone()
        graph_out.backward(static_dy)
        torch.cuda.synchronize()
        graph_grad_results = (
            static_tokens.grad.detach().clone(),
            static_topk_weights.grad.detach().clone(),
            graph_fc1.weight.grad.detach().clone(),
            graph_fc2.weight.grad.detach().clone(),
        )

        eager_tokens = static_tokens.detach().clone().requires_grad_(True)
        eager_topk_weights = static_topk_weights.detach().clone().requires_grad_(True)
        for parameter in eager_model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        with te.autocast(enabled=True, recipe=recipe):
            eager_out = eager_model(
                eager_tokens,
                static_topk_idx,
                eager_topk_weights,
            )
        tolerances = {"rtol": 0.125, "atol": 0.25}
        torch.testing.assert_close(graph_out_snapshot, eager_out, **tolerances)

        eager_out.backward(static_dy)
        torch.cuda.synchronize()
        eager_grad_results = (
            eager_tokens.grad,
            eager_topk_weights.grad,
            eager_fc1.weight.grad,
            eager_fc2.weight.grad,
        )

        for graph_result, eager_result in zip(
            graph_grad_results,
            eager_grad_results,
        ):
            torch.testing.assert_close(graph_result, eager_result, **tolerances)

    @_eager_test_include
    def test_megamoe_main_grad_accumulation_bf16(self):
        self._run_megamoe_vs_reference(
            quantization="bf16",
            accumulate_into_main_grad=True,
        )

    @_eager_test_include
    @_mxfp8_align_test
    def test_megamoe_main_grad_accumulation(self):
        self._run_megamoe_vs_reference(
            quantization="mxfp8",
            accumulate_into_main_grad=True,
        )

    @_eager_test_include
    def test_megamoe_main_grad_overwrite_bf16(self):
        self._run_megamoe_vs_reference(
            quantization="bf16",
            accumulate_into_main_grad=True,
            overwrite_main_grad=True,
        )

    @_eager_test_include
    @_mxfp8_align_test
    def test_megamoe_main_grad_overwrite(self):
        self._run_megamoe_vs_reference(
            quantization="mxfp8",
            accumulate_into_main_grad=True,
            overwrite_main_grad=True,
        )

    @_eager_test_include
    def test_megamoe_delayed_wgrad_bf16(self):
        self._run_megamoe_vs_reference(
            quantization="bf16",
            delay_wgrad_compute=True,
        )

    @_eager_test_include
    @_mxfp8_align_test
    def test_megamoe_delayed_wgrad(self):
        self._run_megamoe_vs_reference(
            quantization="mxfp8",
            delay_wgrad_compute=True,
        )

    @_eager_test_include
    def test_megamoe_delayed_main_grad_bf16(self):
        self._run_megamoe_vs_reference(
            quantization="bf16",
            accumulate_into_main_grad=True,
            delay_wgrad_compute=True,
        )

    @_eager_test_include
    @_mxfp8_align_test
    def test_megamoe_delayed_main_grad(self):
        self._run_megamoe_vs_reference(
            quantization="mxfp8",
            accumulate_into_main_grad=True,
            delay_wgrad_compute=True,
        )

    def _run_megamoe_vs_reference(
        self,
        *,
        quantization,
        accumulate_into_main_grad=False,
        overwrite_main_grad=False,
        delay_wgrad_compute=False,
    ):
        """Compare the five-op MoE sequence with the PyTorch EP reference.

        The fuser selects MegaMoE when its runtime gates pass. Otherwise this
        exercises the same sequence as separate NCCL EP and grouped-MLP ops.
        """
        recipe = MXFP8BlockScaling() if quantization == "mxfp8" else None
        model, fc1, fc2, _ = self._make_megamoe_model(
            recipe=recipe,
            accumulate_into_main_grad=accumulate_into_main_grad,
            delay_wgrad_compute=delay_wgrad_compute,
            glu_interleave_size=32 if quantization == "mxfp8" else None,
        )
        generator = torch.Generator(device=self.cfg.device)
        generator.manual_seed(3100 + self.cfg.rank)
        with torch.no_grad():
            for op in (fc1, fc2):
                weights = op.weight.quantized_tensors
                if weights is None:
                    weights = op.weight.split_into_quantized_tensors()
                for expert in range(NUM_LOCAL_EXPERTS):
                    weight = (
                        torch.randn(
                            weights[expert].shape,
                            generator=generator,
                            dtype=torch.float32,
                            device=self.cfg.device,
                        )
                        * 0.1
                    ).to(torch.bfloat16)
                    weights[expert].copy_(weight)
                    if quantization == "mxfp8":
                        self.assertIsInstance(weights[expert], MXFP8Tensor)

        topk_idx, tokens, topk_weights = _make_moe_inputs(
            self.cfg.rank,
            self.cfg.ep_size,
            self.cfg.device,
        )
        seq_tokens = tokens.detach().clone().requires_grad_(True)
        seq_topk_weights = topk_weights.detach().clone().requires_grad_(True)
        main_grad_sentinel = 0.5
        if accumulate_into_main_grad:
            for op in (fc1, fc2):
                op.weight.main_grad = torch.full(
                    op.weight.size(),
                    main_grad_sentinel,
                    dtype=torch.float32,
                    device=op.weight.device,
                )
                op.weight.overwrite_main_grad = overwrite_main_grad
                op.weight.zero_out_wgrad = False
                op.weight.grad_added_to_main_grad = False
        autocast_ctx = (
            te.autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext()
        )
        with autocast_ctx:
            seq_out = model(
                seq_tokens,
                topk_idx,
                seq_topk_weights,
            )

        forward_ops = model._module_groups[0]._forward_ops
        fused = len(forward_ops) == 1 and isinstance(forward_ops[0][0], FusedMoeEp)
        if fused:
            self.assertTrue(_cudnn_megamoe_supported())
        else:
            self.assertFalse(any(isinstance(op, FusedMoeEp) for op, _ in forward_ops))
        self.assertEqual(seq_out.dtype, torch.bfloat16)

        fc1_weight = _reference_weights(fc1)
        fc2_weight = _reference_weights(fc2)
        emulate_mxfp8 = fused or quantization == "mxfp8"
        reference = MoeEpReference(
            num_experts=self.cfg.num_experts,
            hidden_size=HIDDEN_DIM,
            intermediate_size=256,
            top_k=TOP_K,
            ep_group=self.ep_group,
            max_tokens_per_rank=TOKENS_PER_RANK,
            output_format=MoeFormat.BF16,
            combine_format=(
                MoeFormat.MXFP8
                if fused and os.environ.get("NVTE_MEGAMOE_MXFP8_COMBINE", "0") == "1"
                else MoeFormat.BF16
            ),
            apply_topk_in_fc1=True,
            generate_c=True,
            intermediate_format=MoeFormat.MXFP8 if emulate_mxfp8 else None,
            backward_operand_format=MoeFormat.MXFP8 if emulate_mxfp8 and not fused else None,
            backward_wgrad_mode="operands" if fused else "none",
            token_padding_size=256,
            weight_interleave_size=32 if quantization == "mxfp8" else None,
        )
        if emulate_mxfp8 and not fused:
            reference_activation = quantize_blockwise(
                tokens.detach(),
                MoeFormat.MXFP8,
                axis=1,
            )
        else:
            reference_activation = tokens.detach()
        if emulate_mxfp8 and not isinstance(fc1_weight, BlockScaledTensor):
            fc1_weight = quantize_blockwise(fc1_weight, MoeFormat.MXFP8, axis=1)
        if emulate_mxfp8 and not isinstance(fc2_weight, BlockScaledTensor):
            fc2_weight = quantize_blockwise(fc2_weight, MoeFormat.MXFP8, axis=1)
        reference_outputs = reference(
            reference_activation,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights.detach(),
        )
        if fused:
            ref_out, fc1_c, route_metadata, wgrad_forward_stash = reference_outputs
        else:
            ref_out, fc1_c, route_metadata = reference_outputs
            wgrad_forward_stash = None

        tolerances = {"rtol": 0.125, "atol": 0.25}
        torch.testing.assert_close(seq_out, ref_out, **tolerances)

        dy = (
            torch.randn(
                seq_out.shape,
                generator=generator,
                dtype=torch.float32,
                device=self.cfg.device,
            )
            * 0.1
        ).to(torch.bfloat16)
        seq_out.backward(dy)
        if delay_wgrad_compute:
            fc1.backward_dw()
            fc2.backward_dw()
        reference_grads = reference.backward(
            dy,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights.detach(),
            fc1_c,
            route_metadata,
            wgrad_forward_stash=wgrad_forward_stash,
        )
        if fused:
            grad_tokens, grad_topk_weights, wgrad_operands = reference_grads
            grad_fc1, grad_fc2 = wgrad_operands.dense_wgrads()
            reference_wgrads = (grad_fc1, grad_fc2)
        else:
            grad_tokens, grad_topk_weights = reference_grads
            reference_wgrads = (None, None)

        torch.cuda.synchronize()
        torch.testing.assert_close(
            seq_tokens.grad,
            grad_tokens.to(dtype=seq_tokens.dtype),
            **tolerances,
        )
        torch.testing.assert_close(seq_topk_weights.grad, grad_topk_weights.float(), **tolerances)
        for op, ref_grad in zip((fc1, fc2), reference_wgrads):
            expected_grad = None if ref_grad is None else ref_grad.transpose(1, 2)
            if accumulate_into_main_grad:
                if expected_grad is not None and not overwrite_main_grad:
                    expected_grad = expected_grad + main_grad_sentinel
                if expected_grad is not None:
                    torch.testing.assert_close(
                        op.weight.main_grad,
                        expected_grad.to(dtype=op.weight.main_grad.dtype),
                        **tolerances,
                    )
                else:
                    self.assertTrue(torch.isfinite(op.weight.main_grad).all())
                    self.assertFalse(torch.all(op.weight.main_grad == main_grad_sentinel).item())
                self.assertTrue(op.weight.grad_added_to_main_grad)
                self.assertIsNotNone(op.weight.grad)
                continue
            seq_grad = op.weight.grad
            self.assertEqual(seq_grad.dtype, torch.bfloat16)
            self.assertEqual(
                tuple(seq_grad.shape),
                (NUM_LOCAL_EXPERTS, op.out_features, op.in_features),
            )
            self.assertTrue(seq_grad.is_contiguous())
            self.assertTrue(torch.isfinite(seq_grad).all())
            if expected_grad is not None:
                torch.testing.assert_close(
                    seq_grad,
                    expected_grad.to(dtype=seq_grad.dtype),
                    **tolerances,
                )


def _init_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    try:
        from torch.distributed import _symmetric_memory as _symm_mem

        _symm_mem.set_backend("NCCL")
    except (ImportError, RuntimeError):
        pass


if __name__ == "__main__":
    _init_distributed()
    loader = unittest.TestLoader()
    name_filter = os.environ.get("NVTE_EP_TEST_FILTER")
    if name_filter:
        loader.testMethodPrefix = name_filter
    suite = unittest.TestSuite(
        (
            loader.loadTestsFromTestCase(TestEP),
            loader.loadTestsFromTestCase(TestMoeEpSequential),
        )
    )
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    dist.barrier()
    ep_finalize()
    # Deregister symm-mem windows while the comm is still valid.
    release_symm_mem_pool()
    dist.destroy_process_group()
    sys.exit(0 if result.wasSuccessful() else 1)
