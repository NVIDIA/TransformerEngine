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

from transformer_engine.pytorch.ep import (
    EpBuffer,
    ep_bootstrap,
    ep_finalize,
    ep_prepare,
    ep_dispatch,
    ep_combine,
    symm_mem_alloc,
    is_symm_backed,
    _ep_combine_raw,
    _ep_dispatch_raw,
)


ZERO_COPY = os.environ.get("NVTE_EP_ZERO_COPY", "0") == "1"
EAGER = os.environ.get("NVTE_EP_EAGER", "0") == "1"
OVERFLOW = os.environ.get("NVTE_EP_OVERFLOW", "0") == "1"

# Must come after the transformer_engine import so libtransformer_engine.so is loaded.
import transformer_engine_torch as tex  # noqa: F401


NUM_LOCAL_EXPERTS = 2
HIDDEN_DIM = 32
TOP_K = 2
TOKENS_PER_RANK = 4


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
            recv_capacity_per_rank=cls.cfg.recv_capacity_per_rank,
            hidden_dim=HIDDEN_DIM,
            zero_copy=ZERO_COPY,
            eager=EAGER,
            max_num_topk=TOP_K,
            drop_on_overflow=OVERFLOW,
        )

    def setUp(self):
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

    def _make_buffer(self, alignment=0, top_k=TOP_K):
        return EpBuffer(
            top_k=top_k,
            max_tokens_per_rank=TOKENS_PER_RANK,
            recv_capacity_per_rank=self.cfg.recv_capacity_per_rank,
            hidden_dim=HIDDEN_DIM,
            num_local_experts=NUM_LOCAL_EXPERTS,
            alignment=alignment,
        )

    def _expert_out(self, expert_out):
        """Stage the combine input into symm-mem under zero-copy (combine requires it)."""
        if not ZERO_COPY:
            return expert_out
        symm_buf = symm_mem_alloc(tuple(expert_out.shape), expert_out.dtype, self.ep_group)
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
        # pre-allocated so nothing is allocated/freed mid-interleave. The recv
        # outputs are owned by each EpBuffer (symm-mem under zero-copy).
        recv_w = [None, None, None]
        rc = self.cfg.recv_capacity_per_rank
        if ZERO_COPY:
            gbuf_t = [symm_mem_alloc((rc, H), torch.bfloat16, self.ep_group) for _ in scales]
            gbuf_w = [symm_mem_alloc((rc,), torch.float32, self.ep_group) for _ in scales]
        else:
            gbuf_t = gbuf_w = [None, None, None]

        def fwd(k):
            rt, rw, _ = ep_dispatch(buffers[k], tokens_p[k], idx, w)
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
    dist.destroy_process_group()
    sys.exit(0 if result.wasSuccessful() else 1)
