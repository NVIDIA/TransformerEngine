# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process DeepSeekV3 MoE/layer EP tests, launched via torchrun."""

import os
import sys
import unittest

import torch
import torch.distributed as dist

from transformer_engine.pytorch.ep import ep_bootstrap, ep_finalize, release_symm_mem_pool
from transformer_engine.pytorch.models import DeepSeekV3Layer, DeepSeekV3MoE

HIDDEN = 256
MOE_FFN = 128
SHARED_FFN = 128
NUM_LOCAL_EXPERTS = 2
TOP_K = 2
TOKENS_PER_RANK = 64
HEADS = 4
DTYPE = torch.bfloat16

MLA_KWARGS = dict(
    q_lora_rank=96,
    kv_lora_rank=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
)


def _device_sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _recv_capacity(ep_size: int) -> int:
    cap = ep_size * TOKENS_PER_RANK * TOP_K + NUM_LOCAL_EXPERTS * 128
    return -(-cap // 128) * 128


def _broadcast_params(module: torch.nn.Module) -> None:
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.detach(), src=0)


class TestDeepSeekEP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _device_sm() < 90:
            raise unittest.SkipTest(f"NCCL EP requires SM>=90 (got SM{_device_sm()})")
        cls.rank = dist.get_rank()
        cls.ep_size = dist.get_world_size()
        cls.num_experts = NUM_LOCAL_EXPERTS * cls.ep_size
        world_pg = dist.distributed_c10d._get_default_group()
        cls.ep_group = dist.new_group(ranks=list(range(world_pg.size())), backend="nccl")
        ep_bootstrap(
            cls.ep_group,
            num_experts=cls.num_experts,
            max_tokens_per_rank=TOKENS_PER_RANK,
            hidden_dim=HIDDEN,
            num_topk=TOP_K,
            recv_capacity_per_rank=_recv_capacity(cls.ep_size),
        )

    def _make_moe(self, ep: bool, shared: bool = True) -> DeepSeekV3MoE:
        return DeepSeekV3MoE(
            HIDDEN,
            moe_ffn_hidden_size=MOE_FFN,
            num_experts=self.num_experts,
            topk=TOP_K,
            shared_expert_ffn_hidden_size=SHARED_FFN if shared else None,
            params_dtype=DTYPE,
            ep_group=self.ep_group if ep else None,
            ep_max_tokens_per_rank=TOKENS_PER_RANK if ep else None,
            ep_recv_capacity_per_rank=_recv_capacity(self.ep_size) if ep else None,
        )

    def _copy_local_expert_weights(self, ep_moe: DeepSeekV3MoE, ref: DeepSeekV3MoE) -> None:
        with torch.no_grad():
            ep_moe.gate.weight.copy_(ref.gate.weight)
            if ref.shared_expert is not None:
                for dst, src in zip(
                    ep_moe.shared_expert.parameters(), ref.shared_expert.parameters()
                ):
                    dst.copy_(src)
            ep_fc1, _, ep_fc2 = ep_moe.experts
            ref_fc1, _, ref_fc2 = ref.experts
            for local_e in range(NUM_LOCAL_EXPERTS):
                global_e = self.rank * NUM_LOCAL_EXPERTS + local_e
                getattr(ep_fc1, f"weight{local_e}").copy_(getattr(ref_fc1, f"weight{global_e}"))
                getattr(ep_fc2, f"weight{local_e}").copy_(getattr(ref_fc2, f"weight{global_e}"))

    def test_moe_ep_matches_local(self):
        """EP MoE must match the single-GPU (all-experts-local) path numerically."""
        torch.manual_seed(0)
        ref = self._make_moe(ep=False)
        _broadcast_params(ref)
        ep_moe = self._make_moe(ep=True)
        self._copy_local_expert_weights(ep_moe, ref)

        torch.manual_seed(1234 + self.rank)
        x = torch.randn(TOKENS_PER_RANK, HIDDEN, dtype=DTYPE, device="cuda")
        x_ep = x.clone().requires_grad_(True)
        x_ref = x.clone().requires_grad_(True)

        out_ep = ep_moe(x_ep)
        out_ref = ref(x_ref)
        torch.testing.assert_close(out_ep, out_ref, rtol=0.05, atol=0.05)

        grad_out = torch.randn_like(out_ep)
        out_ep.backward(grad_out)
        out_ref.backward(grad_out)
        torch.testing.assert_close(x_ep.grad, x_ref.grad, rtol=0.05, atol=0.05)
        torch.testing.assert_close(
            ep_moe.gate.weight.grad, ref.gate.weight.grad, rtol=0.1, atol=0.1
        )

        # A local expert's wgrad on its owner rank equals the sum of the
        # reference wgrads over all ranks. all_reduce is collective, so every
        # rank must reduce every expert's grad (in the same order).
        ep_fc1, _, ep_fc2 = ep_moe.experts
        ref_fc1, _, ref_fc2 = ref.experts
        for ep_fc, ref_fc in ((ep_fc1, ref_fc1), (ep_fc2, ref_fc2)):
            ref_grads = [
                getattr(ref_fc, f"weight{e}").grad.float().clone() for e in range(self.num_experts)
            ]
            for g in ref_grads:
                dist.all_reduce(g)
            for local_e in range(NUM_LOCAL_EXPERTS):
                global_e = self.rank * NUM_LOCAL_EXPERTS + local_e
                ep_grad = getattr(ep_fc, f"weight{local_e}").grad.float()
                torch.testing.assert_close(ep_grad, ref_grads[global_e], rtol=0.1, atol=0.1)

        counts = ep_moe._last_tokens_per_expert.clone()
        dist.all_reduce(counts)
        self.assertEqual(counts.sum().item(), self.ep_size * TOKENS_PER_RANK * TOP_K)

    def test_layer_ep_forward_backward(self):
        """Full DeepSeekV3Layer smoke test with an EP MoE block."""
        torch.manual_seed(10 + self.rank)
        layer = DeepSeekV3Layer(
            HIDDEN,
            HEADS,
            num_experts=self.num_experts,
            moe_ffn_hidden_size=MOE_FFN,
            topk=TOP_K,
            shared_expert_ffn_hidden_size=SHARED_FFN,
            params_dtype=DTYPE,
            ep_group=self.ep_group,
            ep_max_tokens_per_rank=TOKENS_PER_RANK,
            ep_recv_capacity_per_rank=_recv_capacity(self.ep_size),
            **MLA_KWARGS,
        )
        x = torch.randn(
            TOKENS_PER_RANK // 2, 2, HIDDEN, dtype=DTYPE, device="cuda", requires_grad=True
        )
        out = layer(x)
        self.assertEqual(out.shape, x.shape)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

        layer.mlp.update_expert_bias()
        self.assertTrue(torch.isfinite(layer.mlp.expert_bias).all())


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
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDeepSeekEP)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    dist.barrier()
    ep_finalize()
    release_symm_mem_pool()
    dist.destroy_process_group()
    sys.exit(0 if result.wasSuccessful() else 1)
