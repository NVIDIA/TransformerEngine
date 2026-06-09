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
    EpHandle,
    EpBuffer,
    ep_bootstrap,
    ep_finalize,
    ep_prepare,
    ep_dispatch,
    ep_combine,
    _ep_combine_raw,
    _ep_dispatch_raw,
)

# Must come after the transformer_engine import so libtransformer_engine.so is loaded.
import transformer_engine_torch as tex  # noqa: F401


NUM_LOCAL_EXPERTS = 2
HIDDEN_DIM = 32
TOP_K = 2
TOKENS_PER_RANK = 4


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
            zero_copy=True,
        )

    def _make_handle(self, alignment=0, top_k=TOP_K):
        return EpHandle(
            top_k=top_k,
            max_tokens_per_rank=TOKENS_PER_RANK,
            recv_capacity_per_rank=self.cfg.recv_capacity_per_rank,
            hidden_dim=HIDDEN_DIM,
            num_local_experts=NUM_LOCAL_EXPERTS,
            alignment=alignment,
        )

    def _make_buffers(self, dtype=torch.bfloat16):
        """Allocate raw recv buffers + token_counts for the primitive tests."""
        rc = self.cfg.recv_capacity_per_rank
        return (
            torch.empty(rc, HIDDEN_DIM, dtype=dtype, device=self.cfg.device),
            torch.empty(rc, dtype=torch.float32, device=self.cfg.device),
            torch.empty(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=self.cfg.device),
        )

    def _make_ep_buffer(self, handle):
        return EpBuffer(handle)

    @staticmethod
    def _weighted(recv_tokens, recv_w):
        """fp32 per-slot weighting + cast back; matches the upstream combine input."""
        mask = (recv_w != 0).to(torch.float32).unsqueeze(-1)
        return (recv_tokens.float() * recv_w.unsqueeze(-1).float() * mask).to(recv_tokens.dtype)

    def _moe_step(self, handle, buffer, topk_idx, tokens, w):
        recv_t, recv_w_out, _tc = ep_dispatch(handle, buffer, tokens, topk_idx, w)
        eo = self._weighted(recv_t, recv_w_out)
        return ep_combine(handle, buffer, eo)

    # Prepare

    def test_primitive_prepare(self):
        handle = self._make_handle()
        topk_idx, _toks, _w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        token_counts = ep_prepare(handle, topk_idx)
        torch.cuda.synchronize()
        self.assertEqual(token_counts.shape, (NUM_LOCAL_EXPERTS,))
        local = int(token_counts.sum().item())
        total = torch.tensor([local], dtype=torch.int64, device=self.cfg.device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=self.ep_group)
        self.assertEqual(int(total.item()), self.cfg.world_size * TOKENS_PER_RANK * TOP_K)

    # Identity round-trip via raw primitives

    def test_primitive_dispatch_combine_identity(self):
        handle = self._make_handle()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_tokens, recv_w, _ = self._make_buffers()
        ep_prepare(handle, topk_idx)
        _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
        result = torch.empty_like(tokens)
        _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    # Autograd

    def test_dispatch_fwd_bwd(self):
        """0.5*||recv_tokens||^2 ; grad_tokens equals TOP_K * tokens."""
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        recv_t, _recv_w, _tc = ep_dispatch(handle, buffer, tokens_p, topk_idx, w)
        loss = 0.5 * (recv_t.float() ** 2).sum()
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            tokens_p.grad.float(), tokens.float() * float(TOP_K), atol=5e-2, rtol=5e-2
        )

    def test_combine_fwd_bwd(self):
        """Full dispatch + combine fwd+bwd; identity inputs round-trip."""
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        tokens_p = tokens.detach().clone().requires_grad_(True)
        out = self._moe_step(handle, buffer, topk_idx, tokens_p, w)
        loss = 0.5 * (out.float() ** 2).sum()
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), tokens.float(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(tokens_p.grad.float(), tokens.float(), atol=5e-2, rtol=5e-2)

    # Multi-iter stability

    def test_dispatch_fwd_bwd_multiple_iterations(self):
        """5 fwd+bwd iters on the same EpHandle + EpBuffer must be bit-stable."""
        handle = self._make_handle()
        buffer = self._make_ep_buffer(handle)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)

        def one_step():
            tokens_p = tokens.detach().clone().requires_grad_(True)
            out = self._moe_step(handle, buffer, topk_idx, tokens_p, w)
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
        handle = self._make_handle()
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        recv_tokens, recv_w, _ = self._make_buffers()
        result = torch.empty_like(tokens)

        def step():
            ep_prepare(handle, topk_idx)
            _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
            _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)

        for _ in range(3):
            step()
        torch.cuda.synchronize()

        # Routing is fixed per layer; prepare runs once before capture.
        ep_prepare(handle, topk_idx)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(graph):
                _ep_dispatch_raw(handle, topk_idx, tokens, w, recv_tokens, recv_w)
                _ep_combine_raw(handle, self._weighted(recv_tokens, recv_w), result)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        ref = result.clone()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result.float(), ref.float(), atol=0, rtol=0)

    # PP-1F1B handle isolation

    def test_pp_1f1b_two_handles(self):
        """PP-1F1B interleave (F0 F1 B0 F2 B1 B2) over 3 per-microbatch handles."""
        T, H = TOKENS_PER_RANK, HIDDEN_DIM
        idx, _toks, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        scales = (0.13, 0.41, 0.77)
        handles, buffers, tokens, tokens_p = [], [], [], []
        for s in scales:
            h = self._make_handle()
            handles.append(h)
            buffers.append(self._make_ep_buffer(h))
            t = torch.full(
                (T, H), s + self.cfg.rank * 0.01, dtype=torch.bfloat16, device=self.cfg.device
            )
            tokens.append(t)
            tokens_p.append(t.detach().clone().requires_grad_(True))

        recv = [None, None, None]

        def fwd(k):
            recv[k], _, _ = ep_dispatch(handles[k], buffers[k], tokens_p[k], idx, w)

        def bwd(k):
            (0.5 * (recv[k].float() ** 2).sum()).backward()
            recv[k] = None

        fwd(0)
        fwd(1)
        bwd(0)
        fwd(2)
        bwd(1)
        bwd(2)
        torch.cuda.synchronize()
        for k in range(3):
            torch.testing.assert_close(
                tokens_p[k].grad.float(),
                tokens[k].float() * float(TOP_K),
                atol=5e-2,
                rtol=5e-2,
            )

    # Input validation

    def test_topk_int32_raises_clear_error(self):
        handle = self._make_handle()
        topk_idx_int32 = torch.zeros(
            TOKENS_PER_RANK, TOP_K, dtype=torch.int32, device=self.cfg.device
        )
        with self.assertRaises(RuntimeError) as cm:
            ep_prepare(handle, topk_idx_int32)
        msg = str(cm.exception)
        self.assertIn("topk_idx", msg)
        self.assertIn(".long()", msg)


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
