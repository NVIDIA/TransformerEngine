# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process unit tests for the TE-JAX Expert Parallelism (EP) primitives.

Default mesh is (dp=2, ep=2); override via ``NVTE_TEST_EP_MESH=DPxEP``.
Coverage:

  - ``ep_bootstrap`` rejects when ``ep_resource`` is unset.
  - Individual primitives (``ep_prepare``, ``ep_dispatch_fwd``, ``ep_combine_fwd``)
    round-trip an identity expert → output ≈ tokens.
  - ``ep_dispatch`` custom_vjp: ``grad_tokens ≈ TOP_K · tokens`` (closed form).
  - ``ep_combine`` custom_vjp: ``max|grad_eo| ≈ eo_const / TOP_K`` (closed form).
  - ``ep_dispatch`` custom_vjp: exact per-(t, k) ``grad_topk_weights`` under
    skewed upstream gradients (no k-axis averaging).
  - HLO reshard guard: compile-only, no XLA collectives outside the EP FFI.

Launch via tests/jax/multi_process_launch_ep.sh (one process per GPU).
"""

import os
import sys
import unittest

import jax
import jax.experimental.multihost_utils as jmu
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from utils import is_devices_enough
from transformer_engine.jax.sharding import MeshResource, global_shard_guard
from transformer_engine.jax.ep import (
    EpLayerConfig,
    ep_bootstrap,
    ep_dispatch,
    ep_combine,
    _ep_domain_for_rank,
)
from transformer_engine.jax.cpp_extensions.ep import (
    ep_prepare,
    ep_dispatch_fwd,
    ep_combine_fwd,
    get_ep_config,
)


# ── Test config ─────────────────────────────────────────────────────────────
# NCCL EP requires NUM_LOCAL_EXPERTS*ep % 4 == 0 (TMA alignment in
# device/hybridep_adapter.cu:511). With NUM_LOCAL_EXPERTS=2, ep must be even.

NUM_LOCAL_EXPERTS = 2  # per-rank → num_experts = NLE * EP
HIDDEN_DIM = 32
TOP_K = 2
TOKENS_PER_DP_SHARD = 4  # per device along dp


def _factor_dp_ep(num_procs):
    """Default to a (2, 2) mesh. Override via ``NVTE_TEST_EP_MESH=DPxEP``.

    NUM_LOCAL_EXPERTS*ep must be a multiple of 4 for NCCL EP's TMA alignment.
    """
    override = os.environ.get("NVTE_TEST_EP_MESH")
    if override:
        dp_str, ep_str = override.lower().split("x")
        dp, ep = int(dp_str), int(ep_str)
        if dp * ep != num_procs:
            raise ValueError(
                f"NVTE_TEST_EP_MESH={override!r} does not multiply to num_procs={num_procs}"
            )
        if (NUM_LOCAL_EXPERTS * ep) % 4 != 0:
            raise ValueError(
                f"NUM_LOCAL_EXPERTS*ep ({NUM_LOCAL_EXPERTS}*{ep}) must be a multiple of 4 "
                "for NCCL EP TMA alignment"
            )
        return dp, ep
    if num_procs != 4:
        raise ValueError(
            f"default mesh expects exactly 4 ranks (got {num_procs}); set "
            "NVTE_TEST_EP_MESH=DPxEP to override"
        )
    return 2, 2


def _build_mesh(dp, ep):
    devs = np.asarray(jax.devices()).reshape(dp, ep)
    return Mesh(devs, ("dp", "ep"))


def _local_device_sm():
    """Return SM major*10+minor of the first local CUDA device, or None."""
    try:
        dev = jax.local_devices()[0]
        cap = getattr(dev, "compute_capability", None)
        if cap is None:
            return None
        major, minor = (int(x) for x in str(cap).split("."))
        return major * 10 + minor
    except Exception:
        return None


class TestEP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sm = _local_device_sm()
        if sm is not None and sm < 90:
            raise unittest.SkipTest(f"NCCL EP requires SM>=90 (got SM{sm})")
        cls.num_procs = jax.process_count()
        cls.rank = jax.process_index()
        cls.dp, cls.ep = _factor_dp_ep(cls.num_procs)
        cls.num_experts = NUM_LOCAL_EXPERTS * cls.ep
        # recv_capacity is per-DP-group (NCCL EP comms isolated per DP color).
        # Under PartitionSpec(("dp","ep"), None) each EP group sees
        # T_global/dp = TOKENS_PER_DP_SHARD tokens total; pad for routing skew.
        T_per_ep_group = TOKENS_PER_DP_SHARD
        active_experts = min(cls.num_experts, T_per_ep_group * TOP_K)
        overconc = cls.num_experts // active_experts
        cls.recv_capacity_per_rank = (
            NUM_LOCAL_EXPERTS * max(T_per_ep_group * TOP_K, 16) * overconc * 2
        )
        cls.mesh = _build_mesh(cls.dp, cls.ep)
        cls.mr = MeshResource(dp_resource="dp", ep_resource="ep")
        with cls.mesh, global_shard_guard(cls.mr):
            ep_bootstrap(
                world_size=cls.num_procs,
                rank=cls.rank,
                num_experts=cls.num_experts,
                max_tokens_per_rank=TOKENS_PER_DP_SHARD,
                recv_capacity_per_rank=cls.recv_capacity_per_rank,
                hidden_dim=HIDDEN_DIM,
            )
        # Bootstrap must snapshot ep_size and num_ep_groups onto EpConfig so
        # abstract-eval never needs the active mesh.
        assert get_ep_config().ep_size == cls.ep
        assert get_ep_config().num_ep_groups == cls.dp
        # One layer config shared by all single-layer tests below; non-zero
        # alignment exercises dispatch_output_per_expert_alignment end-to-end.
        cls.hk = EpLayerConfig(top_k=TOP_K, dispatch_output_per_expert_alignment=16)

    # ── Bootstrap precondition ────────────────────────────────────────────

    def test_bootstrap_rejects_missing_ep_axis(self):
        """ep_bootstrap raises when MeshResource has no ep_resource."""
        with self.mesh, global_shard_guard(MeshResource()):
            with self.assertRaisesRegex(ValueError, "ep_resource"):
                ep_bootstrap(
                    world_size=self.num_procs,
                    rank=self.rank,
                    num_experts=self.num_experts,
                    max_tokens_per_rank=TOKENS_PER_DP_SHARD,
                    recv_capacity_per_rank=self.recv_capacity_per_rank,
                    hidden_dim=HIDDEN_DIM,
                )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _make_identity_inputs(self, nonuniform=False):
        """Identity routing + uniform weights — combined output ≈ tokens.

        ``nonuniform=False``: ``(t*TOP_K+k) % E`` (round-robin, near-balanced).
        ``nonuniform=True``: ``top1=0`` for every token, ``top2=1+(t%(E-1))`` —
        expert 0 absorbs the entire batch while the others split the second
        slot evenly. Exercises a skewed per-expert load.
        """
        T_global = TOKENS_PER_DP_SHARD * self.dp
        E = self.num_experts
        topk_idx = np.empty((T_global, TOP_K), dtype=np.int32)
        if nonuniform:
            assert TOP_K == 2, "non-uniform pattern assumes top_k=2"
            for t in range(T_global):
                topk_idx[t, 0] = 0
                topk_idx[t, 1] = 1 + (t % (E - 1))
        else:
            for t in range(T_global):
                for k in range(TOP_K):
                    topk_idx[t, k] = (t * TOP_K + k) % E
        topk_idx = jnp.asarray(topk_idx)
        topk_weights = jnp.full((T_global, TOP_K), 1.0 / TOP_K, dtype=jnp.float32)
        tokens = jnp.asarray(
            np.linspace(0.1, 0.9, T_global * HIDDEN_DIM, dtype=np.float32).reshape(
                T_global, HIDDEN_DIM
            ),
            dtype=jnp.bfloat16,
        )
        return T_global, topk_idx, tokens, topk_weights

    def _make_random_inputs(self, seed=42, nonuniform=True):
        """Random tokens + skewed top-2 routing (top1=0 always; top2 varies).

        Non-uniform load by default — guarantees expert 0 receives every token
        while the rest of the experts split the second slot. Use
        ``nonuniform=False`` for a balanced (t%E, (t+1)%E) pattern.
        """
        T_dp = TOKENS_PER_DP_SHARD * self.dp
        E = self.num_experts
        rng = np.random.default_rng(seed=seed)
        tokens = jnp.asarray(
            rng.standard_normal((T_dp, HIDDEN_DIM), dtype=np.float32) * 0.5,
            dtype=jnp.bfloat16,
        )
        topk_idx_np = np.empty((T_dp, TOP_K), dtype=np.int32)
        if nonuniform:
            assert TOP_K == 2, "non-uniform pattern assumes top_k=2"
            for t in range(T_dp):
                topk_idx_np[t, 0] = 0
                topk_idx_np[t, 1] = 1 + (t % (E - 1))
        else:
            for t in range(T_dp):
                a, b = t % E, (t + 1) % E
                topk_idx_np[t, 0], topk_idx_np[t, 1] = (a, b) if a < b else (b, a)
        topk_idx = jnp.asarray(topk_idx_np)
        topk_weights = jnp.asarray(np.full((T_dp, TOP_K), 1.0 / TOP_K, dtype=np.float32))
        return T_dp, tokens, topk_idx, topk_weights

    @staticmethod
    def _preweight_expert_out(expert_out, recv_topk_weights):
        """ep_combine is unweighted; mirror the caller-side weighting + mask."""
        mask = (recv_topk_weights != 0).astype(jnp.float32)[..., None]
        w = recv_topk_weights[..., None]
        return (expert_out.astype(jnp.float32) * w * mask).astype(expert_out.dtype)

    # ── Individual primitives (cpp_extensions level) ──────────────────────

    def test_two_handle_mems_no_aliasing(self):
        """Two ``ep_prepare`` calls in one jit must produce distinct handle_mem
        buffers; the pointer-keyed C++ cache must not alias HandleEntries
        across distinct logical layers."""
        _T, topk_idx, _tokens, _w = self._make_identity_inputs()
        ka, kb = (
            EpLayerConfig(top_k=TOP_K, dispatch_output_per_expert_alignment=16),
            EpLayerConfig(top_k=TOP_K, dispatch_output_per_expert_alignment=16),
        )
        dp_spec = PartitionSpec(("dp", "ep"), None)
        with self.mesh, global_shard_guard(self.mr):
            idx_s = jax.lax.with_sharding_constraint(topk_idx, NamedSharding(self.mesh, dp_spec))

            @jax.jit
            def run(idx):
                _tc_a, ha = ep_prepare(ka, idx)
                _tc_b, hb = ep_prepare(kb, idx)
                return ha, hb

            hm_a, hm_b = run(idx_s)
            hm_a.block_until_ready()
            hm_b.block_until_ready()
        self.assertNotEqual(hm_a.unsafe_buffer_pointer(), hm_b.unsafe_buffer_pointer())

    def test_two_layer_dispatch_no_handle_aliasing(self):
        """Two ep_dispatch calls in one jit must not clobber each other's routing
        state. A->B data edge forces XLA to order the collectives sequentially."""
        T_global, topk_idx, tokens, topk_w = self._make_identity_inputs(nonuniform=False)
        tokens_b = (tokens.astype(jnp.float32) * -1.0 + 0.25).astype(tokens.dtype)
        ka, kb = (
            EpLayerConfig(top_k=TOP_K, dispatch_output_per_expert_alignment=16),
            EpLayerConfig(top_k=TOP_K, dispatch_output_per_expert_alignment=16),
        )
        dp_spec = PartitionSpec(("dp", "ep"), None)
        ep_spec_3d = PartitionSpec(("dp", "ep"), None, None)
        ep_spec_2d = PartitionSpec(("dp", "ep"), None)
        with self.mesh, global_shard_guard(self.mr):
            idx_s = jax.lax.with_sharding_constraint(topk_idx, NamedSharding(self.mesh, dp_spec))
            ta = jax.lax.with_sharding_constraint(tokens, NamedSharding(self.mesh, dp_spec))
            tb = jax.lax.with_sharding_constraint(tokens_b, NamedSharding(self.mesh, dp_spec))
            w = jax.lax.with_sharding_constraint(topk_w, NamedSharding(self.mesh, dp_spec))

            def one_layer(hk, idx, toks, w_):
                recv_t, recv_w, hm, tc = ep_dispatch(hk, idx, toks, w_, self.recv_capacity_per_rank)
                recv_t = jax.lax.with_sharding_constraint(
                    recv_t, NamedSharding(self.mesh, ep_spec_3d)
                )
                recv_w = jax.lax.with_sharding_constraint(
                    recv_w, NamedSharding(self.mesh, ep_spec_2d)
                )
                weighted = self._preweight_expert_out(recv_t, recv_w)
                return ep_combine(hk, hm, tc, weighted, T_global, out_sharding=(("dp", "ep"), None))

            @jax.jit
            def run(idx, ta_, tb_, w_):
                out_a = one_layer(ka, idx, ta_, w_)
                # Give XLA a data edge so it cannot schedule B's ep_prepare before A's completes.
                tb_dep = tb_ + (out_a * 0).astype(tb_.dtype)
                return out_a, one_layer(kb, idx, tb_dep, w_)

            out_a, out_b = run(idx_s, ta, tb, w)
            out_a.block_until_ready()
            out_b.block_until_ready()
            out_a_g = jmu.process_allgather(out_a, tiled=True)
            out_b_g = jmu.process_allgather(out_b, tiled=True)

        if self.rank == 0:
            np.testing.assert_allclose(
                np.asarray(out_a_g.astype(jnp.float32)),
                np.asarray(tokens.astype(jnp.float32)),
                atol=5e-2,
                rtol=5e-2,
            )
            np.testing.assert_allclose(
                np.asarray(out_b_g.astype(jnp.float32)),
                np.asarray(tokens_b.astype(jnp.float32)),
                atol=5e-2,
                rtol=5e-2,
            )

    def test_primitive_prepare(self):
        """ep_prepare returns token_counts and handle_mem of the expected shapes."""
        T_global, topk_idx, _tokens, _w = self._make_identity_inputs()
        del T_global
        dp_spec = PartitionSpec(("dp", "ep"), None)
        with self.mesh, global_shard_guard(self.mr):
            idx_s = jax.lax.with_sharding_constraint(topk_idx, NamedSharding(self.mesh, dp_spec))

            @jax.jit
            def run(idx):
                tc, hm = ep_prepare(self.hk, idx)
                return tc, hm

            tc, hm = run(idx_s)
            tc.block_until_ready()
        self.assertEqual(tc.shape, (self.dp * self.ep, NUM_LOCAL_EXPERTS))
        self.assertEqual(hm.shape[0], self.dp * self.ep)
        self.assertGreater(hm.shape[1], 0)

    def _run_identity_round_trip(self, nonuniform):
        T_global, topk_idx, tokens, topk_w = self._make_identity_inputs(nonuniform=nonuniform)
        dp_spec = PartitionSpec(("dp", "ep"), None)
        with self.mesh, global_shard_guard(self.mr):
            idx_s = jax.lax.with_sharding_constraint(topk_idx, NamedSharding(self.mesh, dp_spec))
            tok_s = jax.lax.with_sharding_constraint(tokens, NamedSharding(self.mesh, dp_spec))
            w_s = jax.lax.with_sharding_constraint(topk_w, NamedSharding(self.mesh, dp_spec))

            ep_spec_3d = PartitionSpec(("dp", "ep"), None, None)
            ep_spec_2d = PartitionSpec(("dp", "ep"), None)

            @jax.jit
            def run(idx, toks, w):
                _tc, hm = ep_prepare(self.hk, idx)
                recv_t, recv_w = ep_dispatch_fwd(
                    self.hk, hm, idx, toks, w, self.recv_capacity_per_rank
                )
                recv_t = jax.lax.with_sharding_constraint(
                    recv_t, NamedSharding(self.mesh, ep_spec_3d)
                )
                recv_w = jax.lax.with_sharding_constraint(
                    recv_w, NamedSharding(self.mesh, ep_spec_2d)
                )
                # Apply the weighted hadamard inline (combine FFI is unweighted).
                mask = (recv_w != 0).astype(jnp.float32)[..., None]
                weighted = (recv_t.astype(jnp.float32) * recv_w[..., None] * mask).astype(
                    recv_t.dtype
                )
                weighted = jax.lax.with_sharding_constraint(
                    weighted, NamedSharding(self.mesh, ep_spec_3d)
                )
                out = ep_combine_fwd(
                    self.hk,
                    hm,
                    weighted,
                    T_global,
                    out_partition_spec=(("dp", "ep"), None),
                )
                return jax.lax.with_sharding_constraint(out, NamedSharding(self.mesh, dp_spec))

            out = run(idx_s, tok_s, w_s)
            out.block_until_ready()
            # Allgather so the rank-0 numpy comparison sees the full global tensor.
            out_global = jmu.process_allgather(out, tiled=True)

        # Identity expert + uniform weights → out ≈ tokens (rank-0 check).
        if self.rank == 0:
            np.testing.assert_allclose(
                np.asarray(out_global.astype(jnp.float32)),
                np.asarray(tokens.astype(jnp.float32)),
                atol=5e-2,
                rtol=5e-2,
            )

    def test_primitive_dispatch_combine_identity_uniform(self):
        """Round-robin routing → identity round-trip via the primitive layer."""
        self._run_identity_round_trip(nonuniform=False)

    def test_primitive_dispatch_combine_identity_nonuniform(self):
        """Skewed routing (top1=0 always) → identity round-trip via the primitive layer."""
        self._run_identity_round_trip(nonuniform=True)

    def test_primitive_dispatch_combine_identity_bwd_uniform(self):
        """Bwd through identity round-trip: ∇(0.5 ||out||²) w.r.t. tokens ≈ tokens.

        Identity routing + uniform top-k weights ⇒ dispatch∘combine is the
        identity, so loss = 0.5||tokens||² and ∇_tokens loss = tokens.
        """
        T_global, topk_idx, tokens, topk_w = self._make_identity_inputs(nonuniform=False)
        dp_spec = PartitionSpec(("dp", "ep"), None)
        ep_spec_3d = PartitionSpec(("dp", "ep"), None, None)
        ep_spec_2d = PartitionSpec(("dp", "ep"), None)

        with self.mesh, global_shard_guard(self.mr):

            def loss_fn(toks):
                toks = jax.lax.with_sharding_constraint(toks, NamedSharding(self.mesh, dp_spec))
                idx = jax.lax.with_sharding_constraint(topk_idx, NamedSharding(self.mesh, dp_spec))
                w = jax.lax.with_sharding_constraint(topk_w, NamedSharding(self.mesh, dp_spec))
                recv_t, recv_w, hm, tc = ep_dispatch(
                    self.hk, idx, toks, w, self.recv_capacity_per_rank
                )
                recv_t = jax.lax.with_sharding_constraint(
                    recv_t, NamedSharding(self.mesh, ep_spec_3d)
                )
                recv_w = jax.lax.with_sharding_constraint(
                    recv_w, NamedSharding(self.mesh, ep_spec_2d)
                )
                weighted = self._preweight_expert_out(recv_t, recv_w)
                out = ep_combine(
                    self.hk, hm, tc, weighted, T_global, out_sharding=(("dp", "ep"), None)
                )
                return 0.5 * (out.astype(jnp.float32) ** 2).sum()

            grad = jax.jit(jax.grad(loss_fn))(tokens)
            grad.block_until_ready()
            grad_global = jmu.process_allgather(grad, tiled=True)

        if self.rank == 0:
            np.testing.assert_allclose(
                np.asarray(grad_global.astype(jnp.float32)),
                np.asarray(tokens.astype(jnp.float32)),
                atol=5e-2,
                rtol=5e-2,
            )

    def test_dispatch_combine_3d_input_output(self):
        """3D input ``[B, S, H]`` sharded on the first dim only —
        ``(("dp","ep"), None, None)`` here — dispatch accepts the rank-3 shape
        and combine returns a matching 3D ``[B, S, H]`` output. End-to-end
        round trip recovers the original tokens under identity routing +
        uniform top-k weights."""
        T_global, topk_idx, tokens, topk_w = self._make_identity_inputs(nonuniform=False)
        # B is sharded across all (dp*ep) ranks; S held in one piece per rank.
        B, S, H = T_global, 1, tokens.shape[-1]
        tokens_3d = tokens.reshape(B, S, H)
        topk_idx_3d = topk_idx.reshape(B, S, -1)
        topk_w_3d = topk_w.reshape(B, S, -1)
        spec_3d = PartitionSpec(("dp", "ep"), None, None)
        out_spec_3d = (("dp", "ep"), None, None)
        with self.mesh, global_shard_guard(self.mr):
            idx_s = jax.lax.with_sharding_constraint(topk_idx_3d, NamedSharding(self.mesh, spec_3d))
            tok_s = jax.lax.with_sharding_constraint(tokens_3d, NamedSharding(self.mesh, spec_3d))
            w_s = jax.lax.with_sharding_constraint(topk_w_3d, NamedSharding(self.mesh, spec_3d))

            ep_t = PartitionSpec(("dp", "ep"), None, None)
            ep_w = PartitionSpec(("dp", "ep"), None)

            @jax.jit
            def run(idx, toks, w):
                recv_t, recv_w, hm, _tc = ep_dispatch(
                    self.hk, idx, toks, w, self.recv_capacity_per_rank
                )
                recv_t = jax.lax.with_sharding_constraint(recv_t, NamedSharding(self.mesh, ep_t))
                recv_w = jax.lax.with_sharding_constraint(recv_w, NamedSharding(self.mesh, ep_w))
                weighted = self._preweight_expert_out(recv_t, recv_w)
                out = ep_combine(
                    self.hk,
                    hm,
                    _tc,
                    weighted,
                    num_local_tokens=(B, S),
                    out_sharding=out_spec_3d,
                )
                return out

            out = run(idx_s, tok_s, w_s)
            out.block_until_ready()
            out_global = jmu.process_allgather(out, tiled=True)

        if self.rank == 0:
            self.assertEqual(out_global.shape, (B, S, H))
            np.testing.assert_allclose(
                np.asarray(out_global.astype(jnp.float32)),
                np.asarray(tokens_3d.astype(jnp.float32)),
                atol=5e-2,
                rtol=5e-2,
            )

    # ── Custom-VJP tests ─────────────────────────────────────────────────

    def test_dispatch_vjp_fwd_bwd(self):
        """ep_dispatch fwd + jax.grad w.r.t. tokens.

        Identity routing + loss = 0.5||recv_tokens||² ⇒ each token appears
        TOP_K times in recv_tokens (all routes fit recv_capacity), so
        grad_tokens = TOP_K * tokens (closed form).
        """
        T_global, topk_idx, tokens, topk_w = self._make_identity_inputs()
        del T_global
        dp_spec = PartitionSpec(("dp", "ep"), None)
        ep_spec_3d = PartitionSpec(("dp", "ep"), None, None)

        with self.mesh, global_shard_guard(self.mr):

            align = max(int(self.hk.dispatch_output_per_expert_alignment), 1)

            def loss_fn(toks):
                toks = jax.lax.with_sharding_constraint(toks, NamedSharding(self.mesh, dp_spec))
                idx = jax.lax.with_sharding_constraint(topk_idx, NamedSharding(self.mesh, dp_spec))
                w = jax.lax.with_sharding_constraint(topk_w, NamedSharding(self.mesh, dp_spec))
                recv_tokens, _recv_w, _hm, tc = ep_dispatch(
                    self.hk, idx, toks, w, self.recv_capacity_per_rank
                )
                recv_tokens = jax.lax.with_sharding_constraint(
                    recv_tokens, NamedSharding(self.mesh, ep_spec_3d)
                )
                # ep_dispatch fills only slots [0, sum(padded_per_expert));
                # the tail is uninitialized. Mask with jnp.where (NaN-safe;
                # multiply would propagate NaN*0=NaN).
                padded = ((tc + align - 1) // align) * align
                total_recv = jnp.sum(padded, axis=-1, keepdims=True).astype(jnp.int32)
                slot_idx = jnp.arange(self.recv_capacity_per_rank, dtype=jnp.int32)
                mask = slot_idx[None, :] < total_recv
                rt32 = jnp.where(mask[..., None], recv_tokens.astype(jnp.float32), 0.0)
                return 0.5 * (rt32**2).sum()

            loss, grad_tokens = jax.jit(jax.value_and_grad(loss_fn))(tokens)
            grad_tokens.block_until_ready()
            grad_global = jmu.process_allgather(grad_tokens, tiled=True)

        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(grad_tokens.shape, tokens.shape)
        if self.rank == 0:
            np.testing.assert_allclose(
                np.asarray(grad_global.astype(jnp.float32)),
                np.asarray(tokens.astype(jnp.float32)) * float(TOP_K),
                atol=5e-2,
                rtol=5e-2,
            )

    def test_combine_vjp_fwd_bwd(self):
        """ep_combine fwd + jax.grad w.r.t. expert_out.

        Identity routing + constant eo=c + uniform topk_w ⇒ combined[t] = c
        (sum_k topk_w = 1) and grad_eo[e, s, h] = recv_w[e, s] * c at filled
        slots — so max|grad_eo| ≈ c / TOP_K.
        """
        T_global, topk_idx, tokens, topk_w = self._make_identity_inputs()
        eo_const = 0.5
        expert_out = jnp.full(
            (self.dp * self.ep, self.recv_capacity_per_rank, HIDDEN_DIM),
            eo_const,
            dtype=jnp.bfloat16,
        )
        dp_spec = PartitionSpec(("dp", "ep"), None)
        ep_spec_3d = PartitionSpec(("dp", "ep"), None, None)

        with self.mesh, global_shard_guard(self.mr):

            def loss_fn(eo):
                eo = jax.lax.with_sharding_constraint(eo, NamedSharding(self.mesh, ep_spec_3d))
                toks = jax.lax.with_sharding_constraint(tokens, NamedSharding(self.mesh, dp_spec))
                idx = jax.lax.with_sharding_constraint(topk_idx, NamedSharding(self.mesh, dp_spec))
                w = jax.lax.with_sharding_constraint(topk_w, NamedSharding(self.mesh, dp_spec))
                _recv_tokens, recv_w, hm, tc = ep_dispatch(
                    self.hk, idx, toks, w, self.recv_capacity_per_rank
                )
                recv_w = jax.lax.with_sharding_constraint(
                    recv_w, NamedSharding(self.mesh, PartitionSpec(("dp", "ep"), None))
                )
                weighted = self._preweight_expert_out(eo, recv_w)
                combined = ep_combine(self.hk, hm, tc, weighted, T_global)
                # Pin combined to dp-sharded so autodiff transpose feeds
                # ep_combine_bwd a per-shard cotangent.
                combined = jax.lax.with_sharding_constraint(
                    combined, NamedSharding(self.mesh, dp_spec)
                )
                return 0.5 * (combined.astype(jnp.float32) ** 2).sum()

            loss, grad_eo = jax.jit(jax.value_and_grad(loss_fn))(expert_out)
            grad_eo.block_until_ready()

        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(grad_eo.shape, expert_out.shape)
        for shard in grad_eo.addressable_shards:
            arr = np.asarray(shard.data.astype(jnp.float32))
            self.assertTrue(np.all(np.isfinite(arr)))
            self.assertGreater(arr.max(), 0.0, "grad_eo has no positive entry on filled slots")
            np.testing.assert_allclose(
                arr.max(),
                eo_const / float(TOP_K),
                atol=5e-2,
                rtol=5e-2,
            )

    def test_dispatch_bwd_exact_per_k_topk_weights(self):
        """Distinct per-(t, k) upstream grads ⇒ grad[t, 0] != grad[t, 1] for all t.

        Guards against a regression where the bwd would average across the k
        axis (per-token mean instead of per-slot exact recovery).
        """
        T_dp, tokens, topk_idx, topk_w = self._make_random_inputs()
        dp_spec = PartitionSpec(("dp", "ep"), None)

        with self.mesh, global_shard_guard(self.mr):

            def loss_fn(idx_in, tok_in, w_in):
                idx_in = jax.lax.with_sharding_constraint(idx_in, NamedSharding(self.mesh, dp_spec))
                tok_in = jax.lax.with_sharding_constraint(tok_in, NamedSharding(self.mesh, dp_spec))
                w_in = jax.lax.with_sharding_constraint(w_in, NamedSharding(self.mesh, dp_spec))
                _recv_t, recv_w, _h, _tc = ep_dispatch(
                    self.hk, idx_in, tok_in, w_in, self.recv_capacity_per_rank
                )
                # Per-slot index scale ⇒ each slot's contribution differs.
                scale = jnp.asarray(
                    np.arange(recv_w.size, dtype=np.float32).reshape(recv_w.shape) + 1.0
                )
                return jnp.sum(recv_w * scale)

            grad_topk_w = jax.jit(jax.grad(loss_fn, argnums=2))(topk_idx, tokens, topk_w)
            grad_topk_w.block_until_ready()
            grad_global = jmu.process_allgather(grad_topk_w, tiled=True)

        if self.rank == 0:
            grad_np = np.asarray(grad_global).astype(np.float32)
            mismatch = sum(int(abs(grad_np[t, 0] - grad_np[t, 1]) < 1e-6) for t in range(T_dp))
            self.assertEqual(
                mismatch,
                0,
                f"Expected grad[t, 0] != grad[t, 1] for all {T_dp} tokens under skewed "
                f"upstream scaling; got {mismatch} tokens with grad[t, 0] == grad[t, 1].",
            )

    # ── HLO reshard guard ────────────────────────────────────────────────
    # Compile-only: assert XLA inserts no cross-device collectives outside
    # the EP FFI. EP-axis flux is carried by the FFI itself.

    def test_z_no_unexpected_reshard_in_hlo_fwd(self):
        """Compiled fwd HLO must not insert XLA collectives outside the EP FFI."""
        T_dp, tokens, topk_idx, topk_w = self._make_random_inputs()
        dp_spec = PartitionSpec(("dp", "ep"), None)
        ep_spec_3d = PartitionSpec(("dp", "ep"), None, None)
        ep_spec_2d = PartitionSpec(("dp", "ep"), None)

        with self.mesh, global_shard_guard(self.mr):

            @jax.jit
            def run(idx, toks, w):
                idx = jax.lax.with_sharding_constraint(idx, NamedSharding(self.mesh, dp_spec))
                toks = jax.lax.with_sharding_constraint(toks, NamedSharding(self.mesh, dp_spec))
                w = jax.lax.with_sharding_constraint(w, NamedSharding(self.mesh, dp_spec))
                recv_t, recv_w, hm, tc = ep_dispatch(
                    self.hk, idx, toks, w, self.recv_capacity_per_rank
                )
                recv_t = jax.lax.with_sharding_constraint(
                    recv_t, NamedSharding(self.mesh, ep_spec_3d)
                )
                recv_w = jax.lax.with_sharding_constraint(
                    recv_w, NamedSharding(self.mesh, ep_spec_2d)
                )
                weighted = self._preweight_expert_out(recv_t, recv_w)
                out = ep_combine(self.hk, hm, tc, weighted, T_dp, out_sharding=(("dp", "ep"), None))
                return jax.lax.with_sharding_constraint(out, NamedSharding(self.mesh, dp_spec))

            compiled = run.lower(topk_idx, tokens, topk_w).compile()
            hlo = compiled.as_text()
            # Match instruction names; "all-gather-start" and "all-gather-done"
            # bracket a single async all-gather.
            for op in ("all-gather-start", "all-to-all", "collective-permute"):
                self.assertEqual(hlo.count(op), 0, f"unexpected XLA {op} in fwd HLO:\n{hlo}")
            # XLA drops trailing-None entries from the spec; compare as a tuple.
            # JAX collapses size-1 mesh axes, so dp=1 reduces ("dp","ep") to "ep".
            expected = (("dp", "ep"),) if self.dp > 1 else ("ep",)
            self.assertEqual(tuple(compiled.output_shardings.spec), expected)

    def test_z_no_unexpected_reshard_in_hlo_bwd(self):
        """Compiled bwd HLO must not insert XLA collectives outside the EP FFI."""
        T_dp, tokens, topk_idx, topk_w = self._make_random_inputs()
        rng = np.random.default_rng(seed=44)
        expert_out = jnp.asarray(
            rng.standard_normal(
                (self.dp * self.ep, self.recv_capacity_per_rank, HIDDEN_DIM), dtype=np.float32
            )
            * 0.5,
            dtype=jnp.bfloat16,
        )
        dp_spec = PartitionSpec(("dp", "ep"), None)
        ep_spec_3d = PartitionSpec(("dp", "ep"), None, None)
        ep_spec_2d = PartitionSpec(("dp", "ep"), None)

        with self.mesh, global_shard_guard(self.mr):

            def fwd(eo, toks, idx, w):
                eo = jax.lax.with_sharding_constraint(eo, NamedSharding(self.mesh, ep_spec_3d))
                toks = jax.lax.with_sharding_constraint(toks, NamedSharding(self.mesh, dp_spec))
                idx = jax.lax.with_sharding_constraint(idx, NamedSharding(self.mesh, dp_spec))
                w = jax.lax.with_sharding_constraint(w, NamedSharding(self.mesh, dp_spec))
                _rt, rw, hm, tc = ep_dispatch(self.hk, idx, toks, w, self.recv_capacity_per_rank)
                rw = jax.lax.with_sharding_constraint(rw, NamedSharding(self.mesh, ep_spec_2d))
                weighted = self._preweight_expert_out(eo, rw)
                combined = ep_combine(
                    self.hk, hm, tc, weighted, T_dp, out_sharding=(("dp", "ep"), None)
                )
                return jax.lax.with_sharding_constraint(combined, NamedSharding(self.mesh, dp_spec))

            # jax.vjp + pinned cotangent feeds ep_combine_bwd/ep_dispatch_bwd
            # the expected sharding without relying on XLA-transpose propagation.
            def bwd_only(eo, toks, idx, w, g):
                _y, vjp_fn = jax.vjp(fwd, eo, toks, idx, w)
                g = jax.lax.with_sharding_constraint(g, NamedSharding(self.mesh, dp_spec))
                grads = vjp_fn(g)
                return (
                    jax.lax.with_sharding_constraint(
                        grads[0], NamedSharding(self.mesh, ep_spec_3d)
                    ),
                    jax.lax.with_sharding_constraint(grads[1], NamedSharding(self.mesh, dp_spec)),
                )

            g_seed = jnp.ones((T_dp, HIDDEN_DIM), dtype=jnp.bfloat16)
            compiled = (
                jax.jit(bwd_only).lower(expert_out, tokens, topk_idx, topk_w, g_seed).compile()
            )
            hlo = compiled.as_text()
            for op in ("all-gather-start", "all-to-all", "collective-permute"):
                self.assertEqual(hlo.count(op), 0, f"unexpected XLA {op} in bwd HLO:\n{hlo}")


# ── EP domain grouping (single-process; runs under plain pytest) ─────────────


class TestEpDomainGrouping(unittest.TestCase):
    """EP domains group ranks sharing all non-ep coords, so an orthogonal tp
    axis splits the world into one EP domain per tp coordinate."""

    def test_ep_tp_splits_domains(self):
        # Gate on device count inside the test: calling jax.devices() at
        # class-definition time would initialize the XLA backend before
        # jax.distributed.initialize().
        if not is_devices_enough(8):
            self.skipTest("requires 8 devices")
        # ep=4, tp=2: tp must yield 2 EP domains, each a fixed tp coordinate.
        mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(4, 2), ("expert", "tensor"))
        # Single host shares one process_index; inject row-major ranks to mimic
        # one device per process.
        order = {int(d.id): i for i, d in enumerate(mesh.devices.reshape(-1))}
        d2r = lambda d: order[int(d.id)]

        domains = {}
        for rank in range(8):
            root, col, ndom = _ep_domain_for_rank(mesh, "expert", rank, device_to_rank=d2r)
            self.assertEqual(ndom, 2)  # every rank must agree on the domain count
            domains.setdefault(root, {})[col] = rank
        domains = {root: [m[c] for c in sorted(m)] for root, m in domains.items()}

        self.assertEqual(domains, {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]})


# ── Entry point ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test_multi_process_ep.py <coord_addr> <proc_id> <num_procs>")
        sys.exit(1)

    coord_addr = sys.argv[1]
    proc_id = int(sys.argv[2])
    num_procs = int(sys.argv[3])

    jax.distributed.initialize(
        coordinator_address=coord_addr,
        num_processes=num_procs,
        process_id=proc_id,
        local_device_ids=[proc_id],
    )

    loader = unittest.TestLoader()
    test_cases = (TestEP, TestEpDomainGrouping)
    target = os.environ.get("TARGET_TEST")
    if target:
        name = target.split(".")[-1]
        cls = next((c for c in test_cases if hasattr(c, name)), TestEP)
        suite = loader.loadTestsFromName(name, cls)
    else:
        suite = unittest.TestSuite(loader.loadTestsFromTestCase(c) for c in test_cases)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
