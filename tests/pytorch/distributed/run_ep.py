# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process PyTorch EP tests, launched via torchrun (one process per GPU)."""

from contextlib import nullcontext
from dataclasses import replace
import contextlib
import itertools
import os
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from ep_reference import BlockScaledTensor, MoeEpReference, MoeFormat, quantize_blockwise
import transformer_engine.pytorch as te
from transformer_engine.pytorch import ops as te_ops
from transformer_engine.common.recipe import DelayedScaling, MXFP8BlockScaling
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
    _moe_overlap_available,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

ZERO_COPY = os.environ.get("NVTE_EP_ZERO_COPY", "0") == "1"
EAGER = os.environ.get("NVTE_EP_EAGER", "0") == "1"
OVERFLOW = os.environ.get("NVTE_EP_OVERFLOW", "0") == "1"
# Fused prepare+dispatch (caller-supplied recv buffers) is opt-in on the C++ side; the tests
# that exercise it only run when the same env var is set so they match the active dispatch path.
FUSED_COUNT = os.environ.get("NVTE_EP_FUSED_PREPARE_DISPATCH", "0") == "1"
# Overlapped MoE ops (ep_gemm.h): only the tests marked _overlap_test run, with NVTE_EP_MXFP8_PASS.
OVERLAP = os.environ.get("NVTE_EP_OVERLAP", "0") == "1"
# Per-expert alignment of the overlap tests: 256, 128 or 64 (BF16 only; MXFP8 cases skip).
OVERLAP_ALIGNMENT = int(os.environ.get("NVTE_EP_OVERLAP_ALIGNMENT", "256"))
# Below alignment 256 the unfused ops do not fuse into the grouped MLP, so single results of the
# fused op may round to other values than theirs.
UNFUSED_MLP_REFERENCE = OVERLAP_ALIGNMENT % 256 != 0

# Must come after the transformer_engine import so libtransformer_engine.so is loaded.
import transformer_engine_torch as tex  # noqa: F401

# SM 10.7 has no BF16 grouped MLP fusion, so there the BF16 unfused ops never fuse into it.
UNFUSED_BF16_MLP_REFERENCE = UNFUSED_MLP_REFERENCE or (
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
)

NUM_LOCAL_EXPERTS = 2
# MXFP8 dispatch needs HIDDEN_DIM % 512 == 0 and TOKENS_PER_RANK % 32 == 0. Defaults
# satisfy both so the MXFP8 tests run by default; override via NVTE_EP_HIDDEN_DIM /
# NVTE_EP_TOKENS_PER_RANK.
HIDDEN_DIM = int(os.environ.get("NVTE_EP_HIDDEN_DIM", "512"))
TOP_K = 2
TOKENS_PER_RANK = int(os.environ.get("NVTE_EP_TOKENS_PER_RANK", "32"))
OVERLAP_COMM_SMS = 16  # SMs the dispatch keeps; the overlapped GEMM runs on the rest
# SMs of the forward dispatch, forward combine, backward dispatch and backward combine
# (NVTE_EP_COMM_SMS=12,20,24,16); 0 inherits as ep_bootstrap documents.
COMM_SMS = tuple(int(sms) for sms in os.environ.get("NVTE_EP_COMM_SMS", "0,0,0,0").split(","))
if len(COMM_SMS) != 4:
    raise ValueError(f"NVTE_EP_COMM_SMS needs 4 comma-separated counts, got {COMM_SMS}")


def _comm_sms():
    """The four SM counts of the overlap pass, inherited from OVERLAP_COMM_SMS."""
    fwd = [sms or OVERLAP_COMM_SMS for sms in COMM_SMS[:2]]
    return (*fwd, *(sms or f for sms, f in zip(COMM_SMS[2:], fwd)))


@contextlib.contextmanager
def _nccl_ep_report(lines):
    """Run with NCCL_EP_ENV_VERBOSE=1 and add the lines written to file descriptor 2, where NCCL
    EP reports its settings, to ``lines``; they are written through."""
    sys.stderr.flush()
    saved = os.dup(2)
    verbose = os.environ.get("NCCL_EP_ENV_VERBOSE")
    os.environ["NCCL_EP_ENV_VERBOSE"] = "1"
    with tempfile.TemporaryFile() as captured:
        os.dup2(captured.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            if verbose is None:
                del os.environ["NCCL_EP_ENV_VERBOSE"]
            else:
                os.environ["NCCL_EP_ENV_VERBOSE"] = verbose
            captured.seek(0)
            text = captured.read().decode(errors="replace")
            sys.stderr.write(text)
            lines.extend(text.splitlines())


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


# Grouped MLP fusion needs a per-expert alignment of 256, but the EP backend caches a single
# alignment per process, so alignment=256 tests cannot share a process with the alignment=0 tests.
# They run in a dedicated pass (NVTE_EP_MXFP8_PASS=1) instead.
MXFP8_PASS = os.environ.get("NVTE_EP_MXFP8_PASS", "0") == "1"


def _mxfp8_align_test(fn):
    """Mark a test that dispatches with alignment=256; runs only in the MXFP8 pass."""
    fn._mxfp8_align_test = True
    return fn


def _overlap_test(fn):
    """Mark a test of the overlapped MoE ops; it runs only in the overlap pass."""
    fn._overlap_test = True
    return fn


def _mxfp8_overlap_min_alignment() -> int:
    """Smallest expert alignment of FusedMoE under MXFP8: SM 10.7 has no MXFP8 tail tiles."""
    return 256 if torch.cuda.get_device_capability() == (10, 7) else 128


def _mxfp8_overlap() -> bool:
    """Whether the overlap tests run their MXFP8 cases: shapes and alignment FusedMoE takes."""
    return (
        HIDDEN_DIM % 512 == 0
        and TOKENS_PER_RANK % 32 == 0
        and OVERLAP_ALIGNMENT >= _mxfp8_overlap_min_alignment()
    )


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


def _gate_up_rows(t):
    """Per-expert FC1 rows [experts, 2 * ffn, ...] of the 32-wide interleaved GLU layout in the
    gate-then-up layout (glu_interleave_size=None)."""
    return t.reshape(t.shape[0], -1, 2, 32, *t.shape[2:]).transpose(1, 2).reshape(t.shape)


def _assert_close_in_norm(actual, expected, rtol):
    """``actual`` is within ``rtol`` of ``expected`` in norm, per expert of a grouped weight."""
    rows = expected.shape[0] if expected.dim() == 3 else 1
    actual = actual.float().reshape(rows, -1)
    expected = expected.float().reshape(rows, -1)
    err, ref = (actual - expected).norm(dim=1), expected.norm(dim=1)
    if not bool((err <= rtol * ref).all()):
        raise AssertionError(f"error norms {err.tolist()} exceed {rtol} x {ref.tolist()}")


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
        # HT requires capacity >= max_tokens_per_rank. Each expert pads to 256 rows, so one slab
        # fits and the second overflows.
        cfg.recv_capacity_per_rank = 256 if OVERLAP else TOKENS_PER_RANK
    cfg.device = torch.device("cuda", torch.cuda.current_device())
    return cfg


class _EpTestCase(unittest.TestCase):
    """Shared NCCL EP process state and pass selection."""

    cfg: _Cfg
    ep_group: dist.ProcessGroup
    # Why this pass's EP group could not be created, if so; every test class then skips.
    bootstrap_skip = None

    @classmethod
    def setUpClass(cls):
        # unittest invokes this fixture for each concrete subclass, but EP is process-global.
        if hasattr(_EpTestCase, "cfg"):
            if _EpTestCase.bootstrap_skip:
                raise unittest.SkipTest(_EpTestCase.bootstrap_skip)
            return
        if _device_sm() < 90:
            raise unittest.SkipTest(f"NCCL EP requires SM>=90 (got SM{_device_sm()})")
        _EpTestCase.cfg = _make_cfg()
        _EpTestCase.ep_group = _build_ep_group()
        # The overlap pass keeps NCCL EP's report of the group it creates.
        _EpTestCase.nccl_ep_report = []
        report = _nccl_ep_report(cls.nccl_ep_report) if OVERLAP else nullcontext()
        with report:
            try:
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
                    max_num_sms=OVERLAP_COMM_SMS if OVERLAP else 0,
                    dispatch_fwd_num_sms=COMM_SMS[0],
                    combine_fwd_num_sms=COMM_SMS[1],
                    dispatch_bwd_num_sms=COMM_SMS[2],
                    combine_bwd_num_sms=COMM_SMS[3],
                )
            except RuntimeError as err:
                # A backward count other than the forward one needs NCCL EP backward budgets.
                if "built against an NCCL EP with dispatch_bwd_num_sms" not in str(err):
                    raise
                _EpTestCase.bootstrap_skip = "NCCL EP has no backward SM budgets"
                raise unittest.SkipTest(_EpTestCase.bootstrap_skip) from err

    def setUp(self):
        # alignment=256 MXFP8 tests run only in the dedicated MXFP8 pass; everything else skips
        # there (and the MXFP8 tests skip outside it) since the backend pins one alignment/process.
        is_mxfp8_align = getattr(getattr(self, self._testMethodName), "_mxfp8_align_test", False)
        is_overlap = getattr(getattr(self, self._testMethodName), "_overlap_test", False)
        if OVERLAP != is_overlap:
            self.skipTest("overlapped MoE tests run only in the overlap pass")
        if MXFP8_PASS and not is_mxfp8_align:
            self.skipTest("only alignment=256 MXFP8 tests run in the MXFP8 pass")
        if not MXFP8_PASS and is_mxfp8_align:
            self.skipTest("alignment=256 MXFP8 tests run in the dedicated MXFP8 pass")
        # MXFP8 quantization requires Blackwell (SM 10.0) or newer; the BF16 overlap tests do not.
        if is_mxfp8_align and not is_overlap and torch.cuda.get_device_capability() < (10, 0):
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
        dispatch_fwd_quant_recipe=None,
        combine_bwd_quant_recipe=None,
    ):
        # Primitive/count/overflow tests explicitly cover the unpadded path.
        # The MXFP8 pass overrides this with alignment=256 for grouped MLP fusion.
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
            dispatch_fwd_quant_recipe=dispatch_fwd_quant_recipe,
            combine_bwd_quant_recipe=combine_bwd_quant_recipe,
        )

    def _overlap_config(self, recipe=None):
        """EpConfig of the overlapped MoE tests; ``recipe`` is the EP transport recipe."""
        return self._make_config(
            alignment=OVERLAP_ALIGNMENT,
            dispatch_fwd_quant_recipe=recipe,
            combine_bwd_quant_recipe=recipe,
        )

    def _make_buffer_from_config(self, config):
        """Build the NCCL EP buffer that a config describes, recipes included."""
        return EpBuffer(
            top_k=config.top_k,
            max_tokens_per_rank=config.max_tokens_per_rank,
            hidden_dim=config.hidden_dim,
            num_local_experts=config.num_local_experts,
            recv_capacity_per_rank=config.recv_capacity_per_rank,
            alignment=config.alignment,
            payload_dtype=config.payload_dtype,
            dispatch_fwd_quant_recipe=config.dispatch_fwd_quant_recipe,
            combine_bwd_quant_recipe=config.combine_bwd_quant_recipe,
        )

    def _make_buffer(
        self,
        alignment=0,
        top_k=TOP_K,
        dispatch_fwd_quant_recipe=None,
        combine_bwd_quant_recipe=None,
    ):
        config = self._make_config(
            alignment=alignment,
            top_k=top_k,
            dispatch_fwd_quant_recipe=dispatch_fwd_quant_recipe,
            combine_bwd_quant_recipe=combine_bwd_quant_recipe,
        )
        return self._make_buffer_from_config(config)

    def _require_mxfp8_shapes(self):
        if HIDDEN_DIM % 512 != 0 or TOKENS_PER_RANK % 32 != 0:
            self.skipTest(
                "MXFP8 needs HIDDEN_DIM % 512 == 0 and TOKENS_PER_RANK % 32 == 0 "
                "(set NVTE_EP_HIDDEN_DIM / NVTE_EP_TOKENS_PER_RANK)"
            )
        if OVERLAP and OVERLAP_ALIGNMENT < _mxfp8_overlap_min_alignment():
            self.skipTest("FusedMoE runs MXFP8 with alignment 128 or 256 (256 on SM 10.7)")


class TestEP(_EpTestCase):
    """NCCL EP tests inherited from nvidia_origin/main."""

    def test_bootstrap_accessors(self):
        self.assertIs(get_ep_group(), self.ep_group)
        self.assertEqual(get_ep_drop_on_overflow(), OVERFLOW)

    @_eager_test_include
    @_overflow_test_include
    @_mxfp8_align_test
    @_overlap_test
    def test_comm_sms_reach_nccl_ep(self):
        """NCCL EP runs the dispatch and combine, forward and backward, on the SM counts given to
        ep_bootstrap, a count of 0 inherited from max_num_sms or the forward count."""
        if self.cfg.rank != 0:
            self.skipTest("NCCL EP reports its settings on rank 0")
        report = [line for line in self.nccl_ep_report if "SM budgets fwd/bwd:" in line]
        if not report:
            self.skipTest("this NCCL EP does not report its SM budgets")
        dispatch_fwd, combine_fwd, dispatch_bwd, combine_bwd = _comm_sms()
        self.assertIn(
            f"dispatch {dispatch_fwd}/{dispatch_bwd} combine {combine_fwd}/{combine_bwd};",
            report[-1],
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

    def _require_overlap_kernels(self):
        if not _moe_overlap_available():
            self.skipTest(
                "the EP grouped GEMMs need SM 10.0, 10.3 or 10.7, NVTE_ENABLE_CUTEDSL_BACKEND=1 and"
                " NVTE_MOE_OVERLAP"
            )

    def _moe_ops(
        self,
        config,
        buffer,
        glu_interleave_size=32,
        accumulate_into_main_grad=False,
        *,
        ffn=HIDDEN_DIM,
        activation=None,
        fc_kwargs=({}, {}),
        **linear_kwargs,
    ):
        """MoeDispatch, GroupedLinear, ``activation`` (default ScaledSwiGLU), GroupedLinear and
        MoeCombine over ``buffer``, with the routing channels connected and fixed weights (BF16
        unless ``linear_kwargs`` change the GroupedLinear settings; ``fc_kwargs`` adds settings
        of the first and of the second GroupedLinear)."""
        device = self.cfg.device
        gen = torch.Generator(device=device).manual_seed(5)
        linear_kwargs = {
            "bias": False,
            "dtype": torch.bfloat16,
            "single_grouped_weight": True,
            "split_alignment": OVERLAP_ALIGNMENT,
            "accumulate_into_main_grad": accumulate_into_main_grad,
            **linear_kwargs,
        }
        dispatch = te_ops.MoeDispatch(config, buffer)
        fc1 = te_ops.GroupedLinear(
            NUM_LOCAL_EXPERTS, HIDDEN_DIM, 2 * ffn, device=device, **linear_kwargs, **fc_kwargs[0]
        )
        fc2 = te_ops.GroupedLinear(
            NUM_LOCAL_EXPERTS, ffn, HIDDEN_DIM, device=device, **linear_kwargs, **fc_kwargs[1]
        )
        act = activation or te_ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size)
        combine = te_ops.MoeCombine(config, buffer)
        with torch.no_grad():
            for fc in (fc1, fc2):
                for w in fc.parameters():
                    w.copy_((torch.rand(w.shape, device=device, generator=gen) - 0.5) * 0.1)
        dispatch.set_extra_output_channel(0, "tokens_per_expert", output_to_caller=False)
        dispatch.set_extra_output_channel(1, "routing_weights", output_to_caller=False)
        fc1.set_extra_input_channel(0, "tokens_per_expert")
        act.set_extra_input_channel(0, "routing_weights")
        fc2.set_extra_input_channel(0, "tokens_per_expert")
        return dispatch, fc1, act, fc2, combine

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

    # Overlapped MoE

    @_eager_test_include
    @_overflow_test_include
    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_op(self):
        """The fused MoE op matches the same ops run unfused: output, every gradient and the
        receive peak, also when tokens overflow the capacity and are dropped. Where the
        overlapped GEMMs cannot run, the op falls back to the unfused ops."""
        self._check_moe_overlap_op(None)

    @_eager_test_include
    @_overflow_test_include
    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_op_mxfp8(self):
        """Under MXFP8 with MXFP8 EP transport the fused op matches the unfused ops, with weight
        scales that differ across blocks in both directions, and its GEMM-swizzled weight scales
        equal TE's per-tensor swizzle of each expert, rowwise and columnwise. The fused op
        dispatches x and dY in place from the EP dispatch window. An unset config transport runs
        MXFP8, and a non-MXFP8 config transport recipe is rejected."""
        self._require_mxfp8_shapes()
        recipe = MXFP8BlockScaling()
        with mock.patch.object(
            tex, "ep_dispatch_gemm_fwd", wraps=tex.ep_dispatch_gemm_fwd
        ) as fwd, mock.patch.object(
            tex, "ep_combine_gemm_bwd", wraps=tex.ep_combine_gemm_bwd
        ) as bwd:
            self._check_moe_overlap_op(recipe, recipe)
        for call in fwd.call_args_list + bwd.call_args_list:
            self.assertEqual(call.kwargs["symm_input"], tex.ep_zero_copy_supported())
        # MXFP8 FusedMoE quantizes the EP transport itself, so an unset config transport fuses.
        config = self._overlap_config()
        model = te_ops.Sequential(*self._moe_ops(config, self._make_buffer_from_config(config)))
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        with te.autocast(recipe=recipe):
            model(tokens, topk_idx, w, topk_idx)
        self.assertTrue(self._moe_overlap_selected(model))
        # A non-MXFP8 config transport recipe is rejected by the support query.
        bad = self._overlap_config(DelayedScaling())
        self.assertIn(
            "must be MXFP8BlockScaling or unset",
            te_ops.moe_overlap_unsupported_reason(
                bad, HIDDEN_DIM, params_dtype=torch.bfloat16, recipe=recipe
            ),
        )

    @_eager_test_include
    @_overflow_test_include
    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_op_variants(self):
        """The fused op matches the unfused ops, in BF16 and, where the overlap tests run MXFP8, in
        MXFP8, also: without input and routing-weight gradients (no dispatch backward), with
        MXFP8 weight parameters, with accumulate_into_main_grad and both GLU layouts, with an FFN
        size other than the hidden size, and with FP32 weight parameters cast by
        ``Module.to(torch.bfloat16)``."""
        # (recipe, _check_moe_overlap_op settings)
        cases = [
            (None, {"input_grad": False}),
            # ffn != hidden, so a transposed weight layout cannot pass.
            (None, {"ffn": HIDDEN_DIM // 2}),
            (None, {"main_grad": True}),
            (None, {"main_grad": True, "glu_interleave_size": None}),
            (None, {"params_to_bf16": True}),
            (None, {"params_to_bf16": True, "glu_interleave_size": None}),
            (None, {"params_to_bf16": True, "main_grad": True}),
        ]
        if _mxfp8_overlap():
            recipe = MXFP8BlockScaling()
            cases += [
                (recipe, {"input_grad": False}),
                (recipe, {"params_to_bf16": True}),
                (recipe, {"quantized_weights": True}),
            ]
            cases += [
                (recipe, {"quantized_weights": q, "glu_interleave_size": glu, "main_grad": True})
                for q, glu in itertools.product((False, True), (32, None))
            ]
        for recipe, settings in cases:
            with self.subTest(recipe=recipe, **settings):
                self._check_moe_overlap_op(recipe, recipe, **settings)

    def _assert_glu_none_matches(self, got, want, got_grads, want_grads):
        """The outputs and FC1, FC2 weight gradients of the fused op with glu_interleave_size=None
        match those of the 32-wide layer, the FC1 rows mapped between the layouts."""
        for g, w in zip(got[:2], want[:2]):
            self.assertTrue(torch.equal(g, w))
        if OVERLAP_ALIGNMENT >= 256 and HIDDEN_DIM <= 512:
            self.assertTrue(torch.equal(got[2], want[2]))
        else:
            # The routing-weight gradient adds more than two column tiles, or tail tiles, in
            # arrival order, which changes the FP32 sums by up to about 1e-6 of the largest one.
            torch.testing.assert_close(
                got[2], want[2], rtol=1.3e-6, atol=1e-6 * float(want[2].abs().amax())
            )
        fc1_shape = (NUM_LOCAL_EXPERTS, -1, HIDDEN_DIM)
        # The rows of an expert may arrive in another order, which changes the FP32 wgrad sums by
        # up to about 1e-6 of the expert's largest one, then by a BF16 rounding.
        for g, w in (
            (got_grads[0].reshape(fc1_shape), _gate_up_rows(want_grads[0].reshape(fc1_shape))),
            (got_grads[1], want_grads[1]),
        ):
            for ge, we in zip(g.reshape(NUM_LOCAL_EXPERTS, -1), w.reshape(NUM_LOCAL_EXPERTS, -1)):
                torch.testing.assert_close(ge, we, rtol=1e-2, atol=1e-6 * float(we.abs().amax()))

    def _check_moe_overlap_op(
        self, recipe, transport=None, quantized_weights=False, glu_interleave_size=32,
        main_grad=False, input_grad=True, params_to_bf16=False, ffn=HIDDEN_DIM,
    ):  # fmt: skip
        from transformer_engine.pytorch.ops.fused import moe_overlap

        device = self.cfg.device
        config = self._overlap_config(transport)
        with te.quantized_model_init(enabled=quantized_weights, recipe=recipe):
            buffer = self._make_buffer_from_config(config)
            ops = self._moe_ops(
                config,
                buffer,
                glu_interleave_size,
                main_grad,
                ffn=ffn,
                dtype=torch.float32 if params_to_bf16 else torch.bfloat16,
            )
        dispatch = ops[0]
        gen = torch.Generator(device=device).manual_seed(6)
        if recipe is not None:
            # Power-of-two factors per row and column give every scale block its own exponent.
            with torch.no_grad():
                for fc in (ops[1], ops[3]):
                    for w in fc.parameters():
                        rows = torch.randint(
                            -3, 4, (*w.shape[:-1], 1), device=device, generator=gen
                        )
                        cols = torch.randint(-3, 4, w.shape[-1:], device=device, generator=gen)
                        values = torch.rand(w.shape, device=device, generator=gen) - 0.5
                        w.copy_(values * 0.1 * torch.exp2((rows + cols).float()))
        if glu_interleave_size is None:
            # The layer of glu_interleave_size=32 with its FC1 rows mapped to gate then up.
            with torch.no_grad():
                ops[1].weight.copy_(_gate_up_rows(ops[1].weight * 1))
        if params_to_bf16:
            for op in ops:
                op.to(torch.bfloat16)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        grad_out = torch.randn(tokens.shape, device=device, generator=gen).to(torch.bfloat16)

        def run(fused):
            if not fused and params_to_bf16:
                # The BF16 grouped MLP fusion reads the weight data as stored, so this reference
                # runs the GroupedLinear ops, which cast it, and not FusedMoE.
                ctx = contextlib.ExitStack()
                for fusion in (moe_overlap.GroupedMLP_CuTeGEMMGLUBf16, moe_overlap.FusedMoE):
                    ctx.enter_context(mock.patch.object(fusion, "is_supported", return_value=False))
            elif not fused:
                ctx = mock.patch.object(moe_overlap.FusedMoE, "is_supported", return_value=False)
            elif input_grad:
                ctx = contextlib.nullcontext()
            else:
                ctx = mock.patch.object(
                    moe_overlap.tex, "ep_dispatch_gemm_bwd", side_effect=AssertionError("called")
                )
            dispatch.recv_tokens_peak = torch.zeros(1, dtype=torch.int64, device=device)
            if fused:
                # The fused op uses its own counts and recv total, not the reference's.
                buffer.overlap_tokens_per_expert.fill_(-1)
                buffer.overlap_total_recv_tokens.fill_(-1)
            # Paged stashing pages the tensors that carry a token count.
            stashable = []
            num_tokens = []

            def pack(tensor):
                if hasattr(tensor, "grouped_tensor_num_tokens"):
                    stashable.append((tensor.dtype, tensor.shape, tensor.nbytes))
                    n = tensor.grouped_tensor_num_tokens
                    num_tokens.append(n.clone() if torch.is_tensor(n) else n)
                return tensor

            with ctx, te.autocast(
                enabled=recipe is not None, recipe=recipe
            ), torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                model = te_ops.Sequential(*ops)
                x = tokens.clone().requires_grad_(input_grad)
                topk_w = w.clone().requires_grad_(input_grad)
                for p in model.parameters():
                    p.grad = None
                    if main_grad:
                        p.main_grad = torch.full(p.shape, 0.5, dtype=torch.float32, device=device)
                        p.grad_added_to_main_grad = False
                y = model(x, topk_idx, topk_w, topk_idx)
                y.backward(grad_out)
            torch.cuda.synchronize()
            fused_ops = self._moe_overlap_selected(model)
            if main_grad:
                self.assertTrue(all(p.grad_added_to_main_grad for p in model.parameters()))
            grads = [(p.main_grad if main_grad else p.grad).clone() for p in model.parameters()]
            peak = int(dispatch.recv_tokens_peak.item())
            counts = [int(n) for n in num_tokens]
            return y.detach(), x.grad, topk_w.grad, grads, bool(fused_ops), peak, stashable, counts

        ref = run(False)
        out = run(True)
        self.assertFalse(ref[4])
        expected = moe_overlap.FusedMoE.is_supported() and (
            recipe is None
            or (moe_overlap.GroupedMLP_CuTeGEMMGLU.is_supported() and transport is not None)
        )
        # SM 10.7 runs the MXFP8 weight gradients into main_grad only above 256 receive rows.
        one_tile_main_grad = (
            main_grad
            and recipe is not None
            and torch.cuda.get_device_capability() == (10, 7)
            and config.recv_capacity_per_rank == 256
        )
        self.assertEqual(out[4], expected and not one_tile_main_grad, "fusion selection")
        if out[4]:
            # The launcher requires this line from every rank of an overlap pass.
            print(f"OVERLAP_EXECUTED rank={self.cfg.rank}", flush=True)
        supported = te_ops.moe_overlap_supported(
            config, ffn, params_dtype=torch.bfloat16,
            glu_interleave_size=glu_interleave_size, single_grouped_weight=True, recipe=recipe,
            primary_weights_in_fp8=quantized_weights,
        )  # fmt: skip
        self.assertEqual(
            supported, out[4] or one_tile_main_grad, "moe_overlap_supported matches the fuser"
        )
        self.assertEqual(out[5], ref[5])
        if out[4]:
            self.assertEqual(len(out[6]), 4, "the fused op saves four stashable tensors")
            # Their token count is the number of received rows kept.
            kept = int(buffer.overlap_tokens_per_expert.sum())
            self.assertEqual(out[7], [kept] * 4, "stashed token counts")
            if recipe is not None:
                # The FC2 dgrad recomputes the SwiGLU output, so no [rows, ffn] tensor is saved:
                # the recv tokens and scales, the FC1 output and the recv routing weights.
                rows = next(shape[0] for dtype, shape, _ in out[6] if dtype == torch.float32)
                saved = sum(nbytes for _, _, nbytes in out[6])
                ffn = ops[3].in_features
                self.assertEqual(
                    saved, rows * (HIDDEN_DIM + HIDDEN_DIM // 32 + 2 * ffn * 2 + 4), "saved bytes"
                )
        if OVERFLOW:
            self.assertGreater(out[5], self.cfg.recv_capacity_per_rank)
        # Where the unfused ops compute the SwiGLU output in their own kernel, single elements may
        # round to another value, so those results are compared in norm. The unfused MXFP8 ops also
        # round the FC2 dgrad and the SwiGLU backward output to BF16 before quantizing them, so
        # outputs and weight gradients may differ by up to the E4M3 epsilon.
        norm_rtol = 2e-2 if recipe is None else 1 / 32
        unfused_reference = glu_interleave_size is None or (
            UNFUSED_BF16_MLP_REFERENCE if recipe is None else UNFUSED_MLP_REFERENCE
        )
        outputs_in_norm = unfused_reference or params_to_bf16
        outputs_rtol = 1 / 16 if recipe is not None and unfused_reference else norm_rtol
        if not input_grad:
            self.assertTrue(all(g is None for g in out[1:3] + ref[1:3]))
        for got, want in zip(out[: 3 if input_grad else 1], ref[:3]):
            if outputs_in_norm:
                _assert_close_in_norm(got, want, rtol=outputs_rtol)
            else:
                torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
        for g, g_ref in zip(out[3], ref[3]):
            if (
                recipe is None
                and glu_interleave_size is not None
                and not (UNFUSED_BF16_MLP_REFERENCE or params_to_bf16)
            ):
                torch.testing.assert_close(g, g_ref, rtol=2e-2, atol=2e-2)
            else:
                _assert_close_in_norm(g, g_ref, rtol=outputs_rtol)

    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_inference_mode(self):
        """Weights created in inference mode, which have no version counter, run in the fused op
        with the FC1 rows reordered at each call."""
        self._require_overlap_kernels()
        config = self._overlap_config()
        with torch.inference_mode():
            ops = self._moe_ops(config, self._make_buffer_from_config(config), None)
            model = te_ops.Sequential(*ops)
            topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
            model(tokens, topk_idx, w, topk_idx)
            torch.cuda.synchronize()
            self.assertTrue(self._moe_overlap_selected(model))

    @_eager_test_include
    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_cpu_offload(self):
        """Under CPU offload the tensors the fused op saves come back bitwise, and its output and
        gradients match those without offload."""
        from transformer_engine.pytorch import cpu_offload

        self._require_moe_overlap()
        tokens, topk_idx, w, grad_out = self._moe_calls(1)[0]
        for recipe in [None] + ([MXFP8BlockScaling()] if _mxfp8_overlap() else []):
            config = self._overlap_config(recipe)
            ops = self._moe_ops(config, self._make_buffer_from_config(config))
            results = []
            for offload in (False, True):
                saved = []
                for p in ops[1].parameters():
                    p.grad = None
                for p in ops[3].parameters():
                    p.grad = None
                with te.autocast(enabled=recipe is not None, recipe=recipe):
                    model = te_ops.Sequential(*ops)
                    x = tokens.clone().requires_grad_(True)
                    topk_w = w.clone().requires_grad_(True)
                    if offload:
                        ctx, sync, controller = cpu_offload.get_cpu_offload_context(
                            enabled=True, model_layers=1, manual_synchronization=True
                        )
                        sync_state = ctx.offload_synchronizer
                        push, pop = sync_state.push_tensor, sync_state.pop_tensor

                        def push_tensor(tensor, push=push):
                            saved.append([tensor.detach().reshape(-1).view(torch.uint8).cpu()])
                            return len(saved) - 1, push(tensor)

                        def pop_tensor(packed, pop=pop):
                            tensor = pop(packed[1])
                            saved[packed[0]].append(tensor.detach().reshape(-1).view(torch.uint8))
                            return tensor

                        with mock.patch.object(
                            sync_state, "push_tensor", push_tensor
                        ), mock.patch.object(sync_state, "pop_tensor", pop_tensor):
                            with ctx:
                                y = model(x, topk_idx, topk_w, topk_idx)
                            y = sync(y)
                        controller.start_offload_layer(0)
                        controller.release_activation_forward_gpu_memory(0)
                        controller.start_reload_layer(0)
                    else:
                        y = model(x, topk_idx, topk_w, topk_idx)
                    y.backward(grad_out)
                torch.cuda.synchronize()
                self.assertTrue(self._moe_overlap_selected(model))
                self.assertEqual(bool(saved), offload, "offloaded tensors")
                for before, after in saved:
                    self.assertTrue(torch.equal(before, after.cpu()), "a saved tensor changed")
                grads = [p.grad.clone() for p in model.parameters()]
                results.append([y.detach(), x.grad, topk_w.grad, *grads])
            with self.subTest(recipe=recipe):
                for i, (got, want) in enumerate(zip(*results)):
                    if recipe is not None and i >= 3:
                        _assert_close_in_norm(got, want, rtol=1 / 32)
                    else:
                        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)

    def _require_moe_overlap(self):
        from transformer_engine.pytorch.ops.fused import moe_overlap

        if not moe_overlap.FusedMoE.is_supported():
            self.skipTest(
                "FusedMoE needs SM 10.0, 10.3 or 10.7, NVTE_ENABLE_CUTEDSL_BACKEND=1,"
                " NVTE_MOE_OVERLAP and the BF16 cuDNN grouped GEMM kernels"
            )

    @staticmethod
    def _moe_fusion(fused):
        """Context in which the fuser may select FusedMoE, or keeps the MoE ops unfused."""
        from transformer_engine.pytorch.ops.fused import moe_overlap

        if fused:
            return contextlib.nullcontext()
        return mock.patch.object(moe_overlap.FusedMoE, "is_supported", return_value=False)

    @staticmethod
    def _moe_overlap_selected(model):
        """The FusedMoE ops of the plan ``model`` ran last."""
        from transformer_engine.pytorch.ops.fused import moe_overlap

        return [
            op
            for group in model._module_groups
            for op, _ in getattr(group, "_forward_ops", [])
            if isinstance(op, moe_overlap.FusedMoE)
        ]

    def _moe_calls(self, num_calls):
        """(tokens, topk_idx, topk_weights, grad_output) of calls that each route differently."""
        gen = torch.Generator(device=self.cfg.device).manual_seed(8)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        return [
            (
                tokens.roll(i, 0),
                (topk_idx + i) % self.cfg.num_experts,
                w,
                torch.randn(tokens.shape, device=self.cfg.device, generator=gen).to(torch.bfloat16),
            )
            for i in range(num_calls)
        ]

    @staticmethod
    def _moe_call(model, tokens, topk_idx, w, grad, *, tokens_grad=True, weights_grad=True):
        """Forward, then backward when ``grad`` is given: (output, tokens grad, topk_weights
        grad). Parameter gradients accumulate on the parameters."""
        x = tokens.clone().requires_grad_(tokens_grad)
        topk_w = w.clone().requires_grad_(weights_grad)
        y = model(x, topk_idx, topk_w, topk_idx)
        if grad is not None:
            y.backward(grad)
        return y.detach(), x.grad, topk_w.grad

    @staticmethod
    def _param_grads(model):
        return [None if p.grad is None else p.grad.clone() for p in model.parameters()]

    def _assert_moe_close(self, got, ref, in_norm=UNFUSED_BF16_MLP_REFERENCE, outputs_in_norm=True):
        """Fused and unfused MoE results, nested in lists, agree within BF16 GEMM tolerances;
        with in_norm the grouped weight gradients in norm, and with outputs_in_norm as well the
        outputs and input gradients."""
        if isinstance(ref, (list, tuple)):
            self.assertEqual(len(got), len(ref))
            for g, r in zip(got, ref):
                self._assert_moe_close(g, r, in_norm, outputs_in_norm)
        elif ref is None:
            self.assertIsNone(got)
        elif in_norm and (outputs_in_norm or ref.dim() == 3):
            _assert_close_in_norm(got, ref, rtol=2e-2)
        else:
            torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_main_grad(self):
        """With accumulate_into_main_grad, the fused op writes the weight gradients of two calls
        into an FP32 or BF16 main_grad, accumulated or overwritten, as the unfused ops do."""
        self._require_moe_overlap()
        config = self._overlap_config()
        ops = self._moe_ops(
            config, self._make_buffer_from_config(config), accumulate_into_main_grad=True
        )
        weights = (ops[1].weight, ops[3].weight)
        calls = self._moe_calls(2)
        for dtype, overwrite in itertools.product((torch.float32, torch.bfloat16), (False, True)):
            results = []
            for fused in (False, True):
                for p in weights:
                    p.grad = None
                    p.main_grad = torch.full(p.size(), 0.5, dtype=dtype, device=self.cfg.device)
                    p.overwrite_main_grad = overwrite
                    p.grad_added_to_main_grad = False
                with self._moe_fusion(fused):
                    model = te_ops.Sequential(*ops)
                    outs = [self._moe_call(model, *call) for call in calls]
                torch.cuda.synchronize()
                self.assertEqual(bool(self._moe_overlap_selected(model)), fused)
                for p in weights:
                    self.assertTrue(p.grad_added_to_main_grad)
                results.append((outs, [p.main_grad.clone() for p in weights]))
            with self.subTest(dtype=dtype, overwrite=overwrite):
                self._assert_moe_close(results[1], results[0])

    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_delay_wgrad(self):
        """With delay_wgrad_compute on one or both GroupedLinear ops, FusedMoE is not selected
        and the unfused ops run with a warning: the delayed weight gradient waits for backward_dw,
        the other is computed in the backward, and the results match the fusion disabled, into
        main_grad or not."""
        self._require_moe_overlap()
        config = self._overlap_config()
        calls = self._moe_calls(2)
        # delayed (FC1, FC2), accumulate_into_main_grad
        cases = [
            ((True, True), False),
            ((True, True), True),
            ((True, False), False),
            ((False, True), False),
        ]
        for delayed, main_grad in cases:
            ops = self._moe_ops(
                config, self._make_buffer_from_config(config),
                accumulate_into_main_grad=main_grad,
                fc_kwargs=tuple({"delay_wgrad_compute": d} for d in delayed),
            )  # fmt: skip
            fcs = (ops[1], ops[3])
            results = []
            for fused in (False, True):
                for fc in fcs:
                    fc.weight.grad = None
                    if main_grad:
                        fc.weight.main_grad = torch.zeros(
                            fc.weight.size(), dtype=torch.float32, device=self.cfg.device
                        )
                        fc.weight.grad_added_to_main_grad = False
                with self._moe_fusion(fused), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = te_ops.Sequential(*ops)
                    outs = []
                    for i, call in enumerate(calls):
                        outs.append(self._moe_call(model, *call))
                        if i == 0:
                            torch.cuda.synchronize()
                            for fc, is_delayed in zip(fcs, delayed):
                                if not is_delayed:
                                    self.assertIsNotNone(fc.weight.grad)
                                elif main_grad:
                                    self.assertEqual(int(fc.weight.main_grad.count_nonzero()), 0)
                                else:
                                    self.assertIsNone(fc.weight.grad)
                        for fc, is_delayed in zip(fcs, delayed):
                            if is_delayed:
                                fc.backward_dw()
                torch.cuda.synchronize()
                self.assertFalse(self._moe_overlap_selected(model))
                if fused:
                    self.assertTrue(any("delay_wgrad_compute" in str(w.message) for w in caught))
                wgrads = [fc.weight.main_grad if main_grad else fc.weight.grad for fc in fcs]
                results.append((outs, [g.clone() for g in wgrads]))
            with self.subTest(delayed=delayed, main_grad=main_grad):
                self._assert_moe_close(results[1], results[0])

    @_eager_test_include
    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_rank_without_tokens(self):
        """Every token picks the first expert of ranks 0 and 1, so the other ranks and the other
        experts of ranks 0 and 1 receive no tokens. The fused op matches the unfused ops, and
        the weight gradients of the experts without tokens are zero. The fused op with
        glu_interleave_size=None and the FC1 rows mapped matches it."""
        self._require_moe_overlap()
        if self.cfg.ep_size < 3:
            self.skipTest("needs a rank that receives no tokens")
        config = self._overlap_config()
        ops = self._moe_ops(config, self._make_buffer_from_config(config))
        tokens, _, w, grad = self._moe_calls(1)[0]
        topk_idx = torch.tensor([0, NUM_LOCAL_EXPERTS], device=self.cfg.device).repeat(
            TOKENS_PER_RANK, 1
        )
        results = []
        for fused in (False, True):
            for p in ops[1].parameters():
                p.grad = None
            for p in ops[3].parameters():
                p.grad = None
            with self._moe_fusion(fused):
                model = te_ops.Sequential(*ops)
                out = self._moe_call(model, tokens, topk_idx, w, grad)
            torch.cuda.synchronize()
            self.assertEqual(bool(self._moe_overlap_selected(model)), fused)
            results.append((out, self._param_grads(model)))
        self._assert_moe_close(results[1], results[0])
        has_tokens = self.cfg.rank < 2
        for g in results[1][1]:
            per_expert = g.view(NUM_LOCAL_EXPERTS, -1)
            self.assertEqual(int(per_expert[int(has_tokens) :].count_nonzero()), 0)
            if has_tokens:
                self.assertGreater(int(per_expert[0].count_nonzero()), 0)

        ops_none = self._moe_ops(config, self._make_buffer_from_config(config), None)
        shape = (NUM_LOCAL_EXPERTS, ops[1].out_features, ops[1].in_features)
        with torch.no_grad():
            ops_none[1].weight.rowwise_data.view(shape).copy_(
                _gate_up_rows(ops[1].weight.rowwise_data.view(shape))
            )
        model = te_ops.Sequential(*ops_none)
        out = self._moe_call(model, tokens, topk_idx, w, grad)
        torch.cuda.synchronize()
        self.assertTrue(self._moe_overlap_selected(model))
        self._assert_glu_none_matches(out, results[1][0], self._param_grads(model), results[1][1])

    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_cuda_graph(self):
        """The fused forward and backward, captured in a CUDA graph after warmup and replayed
        with new routing, tokens, topk_weights and output gradient, match the unfused ops."""
        self._require_moe_overlap()
        config = self._overlap_config()
        ops = self._moe_ops(config, self._make_buffer_from_config(config))
        ref_ops = self._moe_ops(config, self._make_buffer_from_config(config))
        gen = torch.Generator(device=self.cfg.device).manual_seed(9)
        random_idx, random_tokens, random_w = _make_moe_inputs(self.cfg.rank, self.cfg.ep_size)
        replays = [
            (
                random_tokens,
                random_idx,
                random_w,
                torch.randn(random_tokens.shape, device=self.cfg.device, generator=gen).to(
                    torch.bfloat16
                ),
            ),
            self._moe_calls(1)[0],
        ]
        # Static inputs, holding the captured call's values first.
        tokens, topk_idx, w, grad = (t.clone() for t in self._moe_calls(2)[1])
        x = tokens.requires_grad_(True)
        topk_w = w.requires_grad_(True)
        model = te_ops.Sequential(*ops)
        params = list(model.parameters())
        out = {}

        def step():
            for t in (x, topk_w, *params):
                t.grad = None
            out["y"] = model(x, topk_idx, topk_w, topk_idx)
            out["y"].backward(grad)

        graph = self._capture(step)
        self.assertTrue(self._moe_overlap_selected(model))
        results = []
        for call in replays:
            with torch.no_grad():
                for static, value in zip((x, topk_idx, topk_w, grad), call):
                    static.copy_(value)
            graph.replay()
            got = (
                (out["y"].detach().clone(), x.grad.clone(), topk_w.grad.clone()),
                [p.grad.clone() for p in params],
            )
            ref_model = te_ops.Sequential(*ref_ops)
            for p in ref_model.parameters():
                p.grad = None
            with self._moe_fusion(False):
                ref = (self._moe_call(ref_model, *call), self._param_grads(ref_model))
            torch.cuda.synchronize()
            results.append((got, ref))
        # Compare once every rank has run every replay, so a mismatch on one rank leaves no rank
        # waiting in a later replay.
        for got, ref in results:
            self._assert_moe_close(got, ref)

    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_fallback(self):
        """For each setting moe_overlap_supported rejects, the fuser warns with the query's
        reason, keeps the ops unfused and gives the results of the unfused ops."""
        from transformer_engine.pytorch.ops.fused import moe_overlap

        self._require_moe_overlap()
        config = self._overlap_config()
        call = self._moe_calls(1)[0]
        supported = {
            "ffn_hidden_size": HIDDEN_DIM,
            "params_dtype": torch.bfloat16,
            "glu_interleave_size": 32,
            "bias": False,
            "single_grouped_weight": True,
        }
        cases = [
            {"single_grouped_weight": False},
            {"bias": True},
            {"params_dtype": torch.float32},
            {"ffn_hidden_size": HIDDEN_DIM + 32},
        ]
        for change in cases:
            settings = {**supported, **change}
            ops = self._moe_ops(
                config, self._make_buffer_from_config(config), ffn=settings["ffn_hidden_size"],
                glu_interleave_size=settings["glu_interleave_size"], bias=settings["bias"],
                dtype=settings["params_dtype"],
                single_grouped_weight=settings["single_grouped_weight"],
            )  # fmt: skip
            # FP32 weights compute in BF16 under autocast.
            use_autocast = settings["params_dtype"] != torch.bfloat16
            with self.subTest(change=change):
                self.assertFalse(te_ops.moe_overlap_supported(config, **settings))
                reason = te_ops.moe_overlap_unsupported_reason(config, **settings)
                self.assertIsNotNone(reason)
                with warnings.catch_warnings(record=True) as caught, torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=use_autocast
                ):
                    warnings.simplefilter("always")
                    model = te_ops.Sequential(*ops)
                    out = (self._moe_call(model, *call), self._param_grads(model))
                self.assertFalse(self._moe_overlap_selected(model))
                expected = f"FusedMoE fusion skipped, running the unfused ops: {reason}."
                self.assertIn(expected, [str(warning.message) for warning in caught])
                skipped = [w for w in caught if str(w.message) == expected]
                self.assertEqual(skipped[0].filename, __file__, "warned at the caller's frame")
                for p in model.parameters():
                    p.grad = None
                with self._moe_fusion(False), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=use_autocast
                ):
                    ref_model = te_ops.Sequential(*ops)
                    ref = (self._moe_call(ref_model, *call), self._param_grads(ref_model))
                torch.cuda.synchronize()
                self._assert_moe_close(out, ref, UNFUSED_MLP_REFERENCE, outputs_in_norm=False)
        # Without the overlap enabled the unfused ops are expected, and nothing warns.
        ops = self._moe_ops(config, self._make_buffer_from_config(config), bias=True)
        with mock.patch.object(
            moe_overlap, "_moe_overlap_available", return_value=False
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._moe_call(te_ops.Sequential(*ops), *call)
        self.assertFalse([w for w in caught if "fusion skipped" in str(w.message)])

    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_execution_fallback(self):
        """Deterministic execution, an FP16 autocast and GLU activations other than SwiGLU leave
        the unfused ops in place with a warning that names the reason, and moe_overlap_supported
        gives the same answer."""
        from transformer_engine.pytorch.ops.fused import moe_overlap

        self._require_overlap_kernels()
        config = self._overlap_config()
        buffer = self._make_buffer_from_config(config)
        settings = {
            "params_dtype": torch.bfloat16,
            "glu_interleave_size": 32,
            "single_grouped_weight": True,
        }
        no_state = contextlib.nullcontext
        deterministic = mock.patch.dict(os.environ, NVTE_ALLOW_NONDETERMINISTIC_ALGO="0")
        fp16 = torch.autocast("cuda", dtype=torch.float16)
        geglu = te_ops.ScaledClampedQGeGLU(32)
        situ = te_ops.ScaledSiTUGLU(32)
        cases = [
            # (name, activation, state, query settings, reason)
            ("swiglu", None, no_state(), {}, None),
            ("deterministic", None, deterministic, {}, "deterministic"),
            ("fp16 autocast", None, fp16, {}, "BF16 autocast"),
            ("clamped geglu", geglu, no_state(), {"activation": "clamped_geglu"}, "SwiGLU"),
            ("situ", situ, no_state(), {"activation": "situ"}, "SwiGLU"),
        ]
        for name, activation, state, query, reason in cases:
            with self.subTest(case=name), state, warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ops = list(self._moe_ops(config, buffer, activation=activation))
                fused = moe_overlap.fuse_moe_overlap_ops(ops)
                if reason is None:
                    self.assertEqual(len(fused), 1)
                    self.assertIsInstance(fused[0], moe_overlap.FusedMoE)
                else:
                    self.assertEqual(fused, ops)
                    messages = [str(w.message) for w in caught if "FusedMoE" in str(w.message)]
                    self.assertEqual(len(messages), 1)
                    self.assertIn(reason, messages[0])
                self.assertEqual(
                    te_ops.moe_overlap_supported(config, HIDDEN_DIM, **settings, **query),
                    reason is None,
                )

    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_shared_buffer(self):
        """Two FusedMoE instances, each with its own EpBuffer, interleave safely: the second
        instance's forward and backward between the first instance's forward and backward leave
        the first instance's output and gradients identical to running it alone. One buffer and op
        instance per outstanding call."""
        self._require_moe_overlap()
        config = self._overlap_config()
        # Separate instances and buffers; _moe_ops seeds identical weights.
        model_a = te_ops.Sequential(*self._moe_ops(config, self._make_buffer_from_config(config)))
        model_b = te_ops.Sequential(*self._moe_ops(config, self._make_buffer_from_config(config)))
        call_a, call_b = self._moe_calls(2)

        def forward(model, call):
            for p in model.parameters():
                p.grad = None
            x = call[0].clone().requires_grad_(True)
            topk_w = call[2].clone().requires_grad_(True)
            return x, topk_w, model(x, call[1], topk_w, call[1])

        def backward(model, x, topk_w, y, grad):
            y.backward(grad)
            torch.cuda.synchronize()
            return [y.detach(), x.grad, topk_w.grad] + self._param_grads(model)

        # Each call run alone, as the serial reference.
        xa, wa, ya = forward(model_a, call_a)
        alone_a = backward(model_a, xa, wa, ya, call_a[3])
        self.assertTrue(self._moe_overlap_selected(model_a))
        xb, wb, yb = forward(model_b, call_b)
        alone_b = backward(model_b, xb, wb, yb, call_b[3])
        self.assertTrue(self._moe_overlap_selected(model_b))

        # Interleaved: the second instance's whole call lands between the first's forward and
        # backward; separate buffers keep each call's routing state intact.
        xa, wa, ya = forward(model_a, call_a)
        xb, wb, yb = forward(model_b, call_b)
        inter_b = backward(model_b, xb, wb, yb, call_b[3])
        inter_a = backward(model_a, xa, wa, ya, call_a[3])
        self._assert_moe_close(inter_a, alone_a, UNFUSED_MLP_REFERENCE, outputs_in_norm=False)
        self._assert_moe_close(inter_b, alone_b, UNFUSED_MLP_REFERENCE, outputs_in_norm=False)

    @_mxfp8_align_test
    @_overlap_test
    def test_moe_overlap_policy_change(self):
        """One Sequential reselects its plan when deterministic mode or the autocast dtype
        changes: FusedMoE runs, falls back while the setting holds and runs again after."""
        self._require_moe_overlap()
        config = self._overlap_config()
        ops = self._moe_ops(config, self._make_buffer_from_config(config))
        call = self._moe_calls(1)[0]
        model = te_ops.Sequential(*ops)
        deterministic = mock.patch.dict(os.environ, NVTE_ALLOW_NONDETERMINISTIC_ALGO="0")
        fp16 = torch.autocast("cuda", dtype=torch.float16)
        results = []
        for state, selected in (
            (contextlib.nullcontext(), True),
            (deterministic, False),
            (fp16, False),
            (contextlib.nullcontext(), True),
        ):
            with state, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if state is deterministic:
                    # The unfused grouped MLP, which only runs at alignment 256 or above, rejects
                    # the nondeterministic backward as well.
                    try:
                        out = self._moe_call(model, *call)
                    except RuntimeError as err:
                        self.assertIn("Deterministic execution", str(err))
                        out = None
                else:
                    out = self._moe_call(model, *call)
            torch.cuda.synchronize()
            self.assertEqual(bool(self._moe_overlap_selected(model)), selected)
            if out is not None and state is not fp16:
                results.append(out)
        for got in results[1:]:
            self._assert_moe_close(got, results[0])

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

    def _assert_mxfp8_matches_bf16(self, recv_mx, tokens, topk_idx, w, tc):
        """Dequantized MXFP8 recv matches a bf16 dispatch of the same tokens, per expert. Row
        order within an expert's block can differ between the two dispatch kernels, so compare
        by sorted row sums (order-tolerant) instead of position."""
        ref_tokens = self._mxfp8_quantizer().quantize(tokens).dequantize()
        ref_recv, _rw, _tc = ep_dispatch(self._make_buffer(alignment=256), ref_tokens, topk_idx, w)
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
        buf = self._make_buffer(dispatch_fwd_quant_recipe=MXFP8BlockScaling(), alignment=256)
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
        buf = self._make_buffer(dispatch_fwd_quant_recipe=MXFP8BlockScaling(), alignment=256)
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
        buf = self._make_buffer(dispatch_fwd_quant_recipe=MXFP8BlockScaling(), alignment=256)
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
        buf_mx = self._make_buffer(combine_bwd_quant_recipe=MXFP8BlockScaling(), alignment=256)
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
        buf_bf = self._make_buffer(alignment=256)
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
        buf_mx = self._make_buffer(combine_bwd_quant_recipe=MXFP8BlockScaling(), alignment=256)
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
        buf_bf = self._make_buffer(alignment=256)
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
        buf = self._make_buffer(dispatch_fwd_quant_recipe=MXFP8BlockScaling(), alignment=256)
        topk_idx, tokens, w = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)

        # Reference counts via the AllGather prepare path (alignment=256, no quant).
        ref_tokens_per_expert = ep_prepare(self._make_buffer(alignment=256), topk_idx).clone()
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
        ref_buf = self._make_buffer(alignment=256)
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


class TestMoeEpSequential(_EpTestCase):
    """Integration tests for Dispatch -> expert MLP -> Combine sequences."""

    @staticmethod
    def _reference_weights(op):
        """Pack GroupedLinear weights into MoeEpReference ``(E, in, out)`` layout."""
        weight = op.weight
        num_groups = op.num_groups
        out_features = op.out_features
        in_features = op.in_features
        data = weight.rowwise_data.view(
            num_groups,
            out_features,
            in_features,
        ).permute(0, 2, 1)
        if weight.quantizer is None:
            return data.detach()
        scale = (
            weight.scale_inv.view(
                num_groups,
                out_features,
                in_features // 32,
            )
            .view(torch.float8_e8m0fnu)
            .permute(0, 2, 1)
        )
        return BlockScaledTensor(
            data=data.view(torch.float8_e4m3fn).detach(),
            scale=scale.detach(),
            format="mxfp8",
            logical_shape=tuple(data.shape),
            axis=1,
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
                te_ops.MoeCombine(config, buffer)(expert_out, topk_idx)
            finally:
                buffer.hidden_dim = original

        # The communication recipes are backend configuration, so a config that
        # disagrees with the buffer about the transport format is rejected.
        mxfp8_config = replace(
            config,
            dispatch_fwd_quant_recipe=MXFP8BlockScaling(),
            combine_bwd_quant_recipe=MXFP8BlockScaling(),
        )
        mxfp8_buffer = self._make_buffer_from_config(mxfp8_config)
        with self.assertRaisesRegex(ValueError, "dispatch_fwd_quant_recipe"):
            te_ops.MoeDispatch(config, mxfp8_buffer)(tokens, topk_idx, topk_weights)
        with self.assertRaisesRegex(ValueError, "combine_bwd_quant_recipe"):
            te_ops.MoeCombine(config, mxfp8_buffer)(expert_out, topk_idx)

    def _run_dispatch_combine_identity(self, *, mxfp8):
        """Route, apply top-k weights, and combine back to local token order."""
        recipe = MXFP8BlockScaling() if mxfp8 else None
        if mxfp8:
            self._require_mxfp8_shapes()
        config = self._make_config(
            alignment=256 if mxfp8 else 0,
            dispatch_fwd_quant_recipe=recipe,
            combine_bwd_quant_recipe=recipe,
        )
        buffer = self._make_buffer_from_config(config)
        dispatch = te_ops.MoeDispatch(config, buffer)
        combine = te_ops.MoeCombine(config, buffer)
        topk_idx, tokens, topk_weights = _make_identity_inputs(
            self.cfg.rank,
            self.cfg.ep_size,
        )
        recv_tokens, tokens_per_expert, recv_weights = dispatch(
            tokens,
            topk_idx,
            topk_weights,
        )
        if mxfp8:
            # Convert to BF16 since combine only supports BF16 in NCCL EP
            # for now. dequantizing to mxfp8 below changes its overall shape
            # to sum(tokens_per_expert).
            recv_tokens = _degroup_mxfp8(recv_tokens)
            recv_weights = recv_weights[: recv_tokens.shape[0]]
        weighted_expert_output = (recv_tokens.float() * recv_weights.float().unsqueeze(-1)).to(
            torch.bfloat16
        )
        output = combine(weighted_expert_output, topk_idx)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, tokens, atol=5e-2, rtol=5e-2)
        self.assertEqual(tokens_per_expert.data_ptr(), buffer.tokens_per_expert.data_ptr())

    @_eager_test_include
    @_mxfp8_align_test
    @_overlap_test
    def test_unfused_ops_on_overlap_group(self):
        """On an overlap-pass group, MoeDispatch and MoeCombine run unfused and form an identity,
        forward and backward."""
        config = self._overlap_config()
        buffer = self._make_buffer_from_config(config)
        dispatch = te_ops.MoeDispatch(config, buffer)
        combine = te_ops.MoeCombine(config, buffer)
        topk_idx, tokens, topk_weights = _make_identity_inputs(self.cfg.rank, self.cfg.ep_size)
        x = tokens.clone().requires_grad_(True)
        w = topk_weights.clone().requires_grad_(True)
        recv_tokens, _tokens_per_expert, recv_weights = dispatch(x, topk_idx, w)
        output = combine(recv_tokens * recv_weights.unsqueeze(-1).to(torch.bfloat16), topk_idx)
        gen = torch.Generator(device=self.cfg.device).manual_seed(8)
        grad_out = torch.randn(tokens.shape, device=self.cfg.device, generator=gen)
        output.backward(grad_out.to(torch.bfloat16))
        torch.cuda.synchronize()
        torch.testing.assert_close(output, tokens, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(x.grad.float(), grad_out, atol=5e-2, rtol=5e-2)
        grad_w_ref = (tokens.float() * grad_out).sum(-1, keepdim=True).expand_as(w)
        torch.testing.assert_close(w.grad.float(), grad_w_ref, atol=5e-1, rtol=5e-2)

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
        config = self._make_config(
            alignment=256 if recipe is not None else 0,
            dispatch_fwd_quant_recipe=recipe,
            combine_bwd_quant_recipe=recipe,
        )
        buffer = self._make_buffer_from_config(config)
        dispatch = te_ops.MoeDispatch(config, buffer)
        init_ctx = (
            te.quantized_model_init(enabled=True, recipe=recipe)
            if recipe is not None
            else nullcontext()
        )
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
        """Unfused Sequential forward/backward graph replay matches eager execution."""
        self._run_megamoe_mxfp8_cuda_graph_matches_eager()

    def _run_megamoe_mxfp8_cuda_graph_matches_eager(self):
        recipe = MXFP8BlockScaling()
        graph_model, graph_fc1, graph_fc2, _ = self._make_megamoe_model(
            recipe=recipe,
        )
        eager_model, eager_fc1, eager_fc2, _ = self._make_megamoe_model(
            recipe=recipe,
        )
        eager_model.load_state_dict(graph_model.state_dict())

        topk_idx, tokens, topk_weights = _make_moe_inputs(
            self.cfg.rank,
            self.cfg.ep_size,
            self.cfg.device,
        )

        static_tokens = tokens.requires_grad_(True)
        static_topk_idx = topk_idx
        static_topk_weights = topk_weights.requires_grad_(True)
        static_dy = torch.randn_like(static_tokens)
        graphed_model = te.make_graphed_callables(
            graph_model,
            (static_tokens, static_topk_idx, static_topk_weights, static_topk_idx),
            num_warmup_iters=3,
            enabled=True,
            recipe=recipe,
        )

        def reset_graphed_model():
            # Release captured NCCL work before the process group is destroyed.
            graphed_model.reset()
            torch.cuda.synchronize()

        self.addCleanup(reset_graphed_model)

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
                static_topk_idx,
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
                static_topk_idx,
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
        seq_tokens = tokens.requires_grad_(True)
        seq_topk_weights = topk_weights.requires_grad_(True)
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
                topk_idx,
            )

        self.assertEqual(seq_out.dtype, torch.bfloat16)

        fc1_weight = self._reference_weights(fc1)
        fc2_weight = self._reference_weights(fc2)
        emulate_mxfp8 = quantization == "mxfp8"
        reference = MoeEpReference(
            num_experts=self.cfg.num_experts,
            hidden_size=HIDDEN_DIM,
            intermediate_size=256,
            top_k=TOP_K,
            ep_group=self.ep_group,
            max_tokens_per_rank=TOKENS_PER_RANK,
            output_format=MoeFormat.BF16,
            combine_format=MoeFormat.BF16,
            apply_topk_in_fc1=True,
            generate_c=True,
            intermediate_format=MoeFormat.MXFP8 if emulate_mxfp8 else None,
            backward_operand_format=MoeFormat.MXFP8 if emulate_mxfp8 else None,
            backward_wgrad_mode="operands",
            token_padding_size=256,
            weight_interleave_size=32 if quantization == "mxfp8" else None,
        )
        if emulate_mxfp8:
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
        ref_out, fc1_c, route_metadata, wgrad_forward_stash = reference_outputs

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
        grad_tokens, grad_topk_weights, wgrad_operands = reference.backward(
            dy,
            fc1_weight,
            fc2_weight,
            topk_idx,
            topk_weights.detach(),
            fc1_c,
            route_metadata,
            wgrad_forward_stash=wgrad_forward_stash,
        )
        grad_fc1, grad_fc2 = wgrad_operands.dense_wgrads()
        reference_wgrads = (grad_fc1, grad_fc2)

        torch.cuda.synchronize()
        torch.testing.assert_close(
            seq_tokens.grad,
            grad_tokens.to(dtype=seq_tokens.dtype),
            **tolerances,
        )
        torch.testing.assert_close(seq_topk_weights.grad, grad_topk_weights.float(), **tolerances)
        for op, ref_grad in zip((fc1, fc2), reference_wgrads):
            expected_grad = ref_grad.transpose(1, 2)
            if accumulate_into_main_grad:
                if not overwrite_main_grad:
                    expected_grad = expected_grad + main_grad_sentinel
                torch.testing.assert_close(
                    op.weight.main_grad,
                    expected_grad.to(dtype=op.weight.main_grad.dtype),
                    **tolerances,
                )
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
