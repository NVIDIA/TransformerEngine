# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process PyTorch EP tests, launched via torchrun (one process per GPU)."""

import os
import sys
import unittest

import numpy as np
import torch
import torch.distributed as dist

from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.ep import (
    EpBuffer,
    ep_bootstrap,
    ep_finalize,
    ep_prepare,
    ep_dispatch,
    ep_combine,
    symm_mem_alloc,
    release_symm_mem_pool,
    is_symm_backed,
    _ep_combine_raw,
    _ep_dispatch_raw,
)

ZERO_COPY = os.environ.get("NVTE_EP_ZERO_COPY", "0") == "1"
EAGER = os.environ.get("NVTE_EP_EAGER", "0") == "1"
OVERFLOW = os.environ.get("NVTE_EP_OVERFLOW", "0") == "1"
# Fused prepare+dispatch (caller-supplied recv buffers) is opt-in on the C++ side; the tests
# that exercise it only run when the same env var is set so they match the active dispatch path.
FUSED_COUNT = os.environ.get("NVTE_EP_FUSED_PREPARE_DISPATCH", "0") == "1"

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


def _make_skewed_routing(rank, ep_size, device="cuda"):
    """Routing whose per-expert and per-rank totals differ from the identity routing's uniform
    distribution: every token's first top-k slot lands on this rank's own expert. A replay that
    reuses stale counts from an identity-routed capture would then fail the count comparison."""
    T = TOKENS_PER_RANK
    E = ep_size * NUM_LOCAL_EXPERTS
    topk_idx = np.empty((T, TOP_K), dtype=np.int64)
    for t in range(T):
        topk_idx[t, 0] = rank % E
        for k in range(1, TOP_K):
            topk_idx[t, k] = (rank % E + t * TOP_K + k) % E
    return torch.from_numpy(topk_idx).to(device)


def _degroup_mxfp8(recv_grouped, valid_counts=None):
    """Dequantize a per-expert MXFP8 GroupedTensor to a dense tensor in expert-major order.
    With ``valid_counts`` keep only the first ``valid_counts[e]`` rows of each padded expert
    slot; otherwise return every (padded) row."""
    parts = recv_grouped.split_into_quantized_tensors()
    if valid_counts is None:
        return torch.cat([p.dequantize() for p in parts], dim=0)
    return torch.cat([p.dequantize()[:v] for p, v in zip(parts, valid_counts)], dim=0)


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


class TestEP(unittest.TestCase):
    cfg: _Cfg
    ep_group: dist.ProcessGroup

    @classmethod
    def setUpClass(cls):
        if _device_sm() < 90:
            raise unittest.SkipTest(f"NCCL EP requires SM>=90 (got SM{_device_sm()})")
        cls.cfg = _make_cfg()
        cls.ep_group = _build_ep_group()
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

    def _make_buffer(
        self,
        alignment=0,
        top_k=TOP_K,
        dispatch_fwd_quant_recipe=None,
        combine_bwd_quant_recipe=None,
    ):
        return EpBuffer(
            top_k=top_k,
            max_tokens_per_rank=TOKENS_PER_RANK,
            hidden_dim=HIDDEN_DIM,
            num_local_experts=NUM_LOCAL_EXPERTS,
            recv_capacity_per_rank=None if EAGER else self.cfg.recv_capacity_per_rank,
            alignment=alignment,
            dispatch_fwd_quant_recipe=dispatch_fwd_quant_recipe,
            combine_bwd_quant_recipe=combine_bwd_quant_recipe,
        )

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
        """Dequantized MXFP8 recv matches a bf16 dispatch of the same tokens, per expert. Row
        order within an expert's block can differ between the two dispatch kernels, so compare
        by sorted row sums (order-tolerant) instead of position."""
        ref_tokens = self._mxfp8_quantizer().quantize(tokens).dequantize()
        ref_recv, _rw, _tc = ep_dispatch(self._make_buffer(alignment=128), ref_tokens, topk_idx, w)
        torch.cuda.synchronize()
        got = _degroup_mxfp8(recv_mx).float()
        cum = [0] + tc.cumsum(0).tolist()
        for lo, hi in zip(cum[:-1], cum[1:]):
            torch.testing.assert_close(
                got[lo:hi].sum(dim=1).sort().values,
                ref_recv[lo:hi].float().sum(dim=1).sort().values,
                atol=1e-1,
                rtol=1e-1,
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

    def _capture(self, step):
        """Warm up ``step`` on a side stream then capture it into a CUDA graph. Returns
        the graph; the caller replays it. With NVTE_EP_FUSED_PREPARE_DISPATCH set, dispatch
        under capture takes the fused count-mode path that derives the counts from the
        dispatch scan."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        return graph

    def _caller_recv(self):
        """A persistent recv token + weight buffer pair sized to recv_capacity_per_rank, for
        capturing a caller-provided-recv dispatch."""
        rc = self.cfg.recv_capacity_per_rank
        return (
            torch.empty(rc, HIDDEN_DIM, dtype=torch.bfloat16, device=self.cfg.device),
            torch.empty(rc, dtype=torch.float32, device=self.cfg.device),
        )

    def test_fused_count_mode_parity(self):
        """Under CUDA graph capture with a caller recv buffer, dispatch derives
        tokens_per_expert / total_recv_tokens from the fused count scan instead of the
        AllGather prepare. Zeroing the counts before replay forces the replayed graph to
        repopulate them; the result must match the AllGather counts for the same routing."""
        if not FUSED_COUNT:
            self.skipTest("fused count-mode dispatch not enabled")
        if EAGER:
            self.skipTest("fused count mode requires non-eager static recv capacity")
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)

        # Reference counts via the AllGather prepare path.
        ref_buf = self._make_buffer()
        ref_tokens_per_expert = ep_prepare(ref_buf, topk_idx).clone()
        torch.cuda.synchronize()
        ref_total = int(ref_buf.total_recv_tokens.item())

        # Fused path: capture a caller-recv dispatch, clear the counts, then replay so the
        # replayed graph is the sole source of the counts.
        buf = self._make_buffer()
        rbuf_t, rbuf_w = self._caller_recv()
        graph = self._capture(
            lambda: ep_dispatch(
                buf, tokens, topk_idx, w, recv_tokens=rbuf_t, recv_topk_weights=rbuf_w
            )
        )
        buf.tokens_per_expert.zero_()
        buf.total_recv_tokens.zero_()
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(buf.tokens_per_expert, ref_tokens_per_expert, atol=0, rtol=0)
        self.assertEqual(int(buf.total_recv_tokens.item()), ref_total)

    def test_fused_count_mode_changing_routing(self):
        """Routing may differ between graph replays. Each replay must re-derive its counts from
        its own routing rather than reuse the captured step's values (stale-count guard)."""
        if not FUSED_COUNT:
            self.skipTest("fused count-mode dispatch not enabled")
        if EAGER:
            self.skipTest("fused count mode requires non-eager static recv capacity")
        idx_a, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        idx_b = _make_skewed_routing(self.cfg.rank, self.cfg.ep_size)  # skews expert/rank totals

        buf = self._make_buffer()
        topk_idx, rbuf_t, rbuf_w = idx_a.clone(), *self._caller_recv()
        graph = self._capture(
            lambda: ep_dispatch(
                buf, tokens, topk_idx, w, recv_tokens=rbuf_t, recv_topk_weights=rbuf_w
            )
        )

        ref_tpes = []
        for routing in (idx_b, idx_a):
            # Reference counts for this routing via the AllGather prepare path.
            ref_buf = self._make_buffer()
            ref_tpe = ep_prepare(ref_buf, routing).clone()
            torch.cuda.synchronize()
            ref_tpes.append(ref_tpe)

            topk_idx.copy_(routing)
            buf.tokens_per_expert.zero_()
            buf.total_recv_tokens.zero_()
            graph.replay()
            torch.cuda.synchronize()

            torch.testing.assert_close(buf.tokens_per_expert, ref_tpe, atol=0, rtol=0)
            self.assertEqual(
                int(buf.total_recv_tokens.item()), int(ref_buf.total_recv_tokens.item())
            )

        # Guard the guard: if the two routings happen to produce identical counts, reusing a
        # stale copy would pass unnoticed and this test would stop detecting the bug.
        self.assertFalse(torch.equal(ref_tpes[0], ref_tpes[1]))

    @_overflow_test_include
    def test_fused_count_mode_overflow(self):
        """Fused count-mode dispatch under capture reports the pre-drop recv total and caps the
        per-expert counts at recv_capacity, matching the AllGather prepare overflow semantics."""
        if not FUSED_COUNT:
            self.skipTest("fused count-mode dispatch not enabled")
        if not OVERFLOW:
            self.skipTest("overflow-only assertions")
        if EAGER:
            self.skipTest("fused count mode requires non-eager static recv capacity")
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        expected_recv = TOKENS_PER_RANK * TOP_K
        self.assertGreater(expected_recv, self.cfg.recv_capacity_per_rank)

        buf = self._make_buffer()
        rbuf_t, rbuf_w = self._caller_recv()
        graph = self._capture(
            lambda: ep_dispatch(
                buf, tokens, topk_idx, w, recv_tokens=rbuf_t, recv_topk_weights=rbuf_w
            )
        )
        buf.tokens_per_expert.zero_()
        buf.total_recv_tokens.zero_()
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(int(buf.total_recv_tokens.item()), expected_recv)
        self.assertEqual(int(buf.tokens_per_expert.sum().item()), self.cfg.recv_capacity_per_rank)

    @_mxfp8_align_test
    def test_dispatch_mxfp8_fused_capture(self):
        """Fused count-mode MXFP8 dispatch with a caller recv buffer. The returned GroupedTensor
        views the caller buffer's data then scale regions; counts, payload/weights (per-expert,
        order-tolerant), and total must match the AllGather prepare."""
        if not FUSED_COUNT:
            self.skipTest("fused count-mode dispatch not enabled")
        if EAGER:
            self.skipTest("fused count mode requires non-eager static recv capacity")
        self._require_mxfp8_shapes()
        from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE

        rc = self.cfg.recv_capacity_per_rank
        cols = HIDDEN_DIM // MXFP8_BLOCK_SCALING_SIZE
        nbytes = rc * (HIDDEN_DIM + cols)  # fp8 data + e8m0 scales, one byte per element
        recv_buf = torch.empty(nbytes, dtype=torch.uint8, device=self.cfg.device)
        rbuf_w = torch.empty(rc, dtype=torch.float32, device=self.cfg.device)
        buf = self._make_buffer(dispatch_fwd_quant_recipe=MXFP8BlockScaling(), alignment=128)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)

        # Reference counts via the AllGather prepare path (alignment=128, no quant).
        ref_tokens_per_expert = ep_prepare(self._make_buffer(alignment=128), topk_idx).clone()
        torch.cuda.synchronize()

        out = {}

        def step():
            out["recv_mx"], out["rw"], _tc = ep_dispatch(
                buf, tokens, topk_idx, w, recv_tokens=recv_buf, recv_topk_weights=rbuf_w
            )

        graph = self._capture(step)
        buf.tokens_per_expert.zero_()
        buf.total_recv_tokens.zero_()
        graph.replay()
        torch.cuda.synchronize()

        # The returned GroupedTensor views the caller buffer's data then scale regions, and the
        # replayed fused scan repopulates the per-expert counts to match the AllGather path.
        recv_mx = out["recv_mx"]
        self.assertEqual(recv_mx.rowwise_data.data_ptr(), recv_buf.data_ptr())
        self.assertEqual(recv_mx.scale_inv.data_ptr(), recv_buf.data_ptr() + rc * HIDDEN_DIM)
        torch.testing.assert_close(buf.tokens_per_expert, ref_tokens_per_expert, atol=0, rtol=0)

        # Replayed payload correctness: received tokens/scales, weights, and total count must
        # come from the replay's own dispatch, not a stale copy from the captured step.
        self._assert_mxfp8_matches_bf16(recv_mx, tokens, topk_idx, w, buf.tokens_per_expert)
        ref_buf = self._make_buffer(alignment=128)
        _ref_recv, ref_rw, _ref_tc = ep_dispatch(ref_buf, tokens, topk_idx, w)
        torch.cuda.synchronize()
        cum = [0] + buf.tokens_per_expert.cumsum(0).tolist()
        for lo, hi in zip(cum[:-1], cum[1:]):
            torch.testing.assert_close(
                out["rw"][lo:hi].sort().values, ref_rw[lo:hi].sort().values, atol=0, rtol=0
            )
        self.assertEqual(int(buf.total_recv_tokens.item()), int(ref_buf.total_recv_tokens.item()))

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
    suite = loader.loadTestsFromTestCase(TestEP)
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    dist.barrier()
    ep_finalize()
    # Deregister symm-mem windows while the comm is still valid.
    release_symm_mem_pool()
    dist.destroy_process_group()
    sys.exit(0 if result.wasSuccessful() else 1)
