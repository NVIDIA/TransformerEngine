# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multi-process tests for model-specific layers (te.models), launched via torchrun."""

import os
import sys

import torch
import torch.distributed as dist

from transformer_engine.pytorch.ep import ep_bootstrap, ep_finalize, release_symm_mem_pool
from transformer_engine.pytorch.models import DeepSeekV3Layer, DeepSeekV3MoE, MultiLatentAttention

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
    return DeepSeekV3MoE.ep_recv_capacity(ep_size, TOKENS_PER_RANK, TOP_K, NUM_LOCAL_EXPERTS)


def _broadcast_params(module: torch.nn.Module) -> None:
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.detach(), src=0)


def _make_layer(ep_group, ep_size: int, num_experts: int) -> DeepSeekV3Layer:
    ep = ep_group is not None
    return DeepSeekV3Layer(
        HIDDEN,
        HEADS,
        num_experts=num_experts,
        moe_ffn_hidden_size=MOE_FFN,
        topk=TOP_K,
        shared_expert_ffn_hidden_size=SHARED_FFN,
        params_dtype=DTYPE,
        ep_group=ep_group,
        ep_max_tokens_per_rank=TOKENS_PER_RANK if ep else None,
        **MLA_KWARGS,
    )


def _copy_weights(ep_layer: DeepSeekV3Layer, ref: DeepSeekV3Layer, rank: int) -> None:
    ref_params = dict(ref.named_parameters())
    ref_bufs = dict(ref.named_buffers())
    with torch.no_grad():
        for name, p in ep_layer.named_parameters():
            if not name.startswith("mlp.experts."):
                p.copy_(ref_params[name])
        for name, b in ep_layer.named_buffers():
            if name in ref_bufs and b.shape == ref_bufs[name].shape:
                b.copy_(ref_bufs[name])
        ep_fc1, _, ep_fc2 = ep_layer.mlp.experts
        ref_fc1, _, ref_fc2 = ref.mlp.experts
        for local_e in range(NUM_LOCAL_EXPERTS):
            global_e = rank * NUM_LOCAL_EXPERTS + local_e
            getattr(ep_fc1, f"weight{local_e}").copy_(getattr(ref_fc1, f"weight{global_e}"))
            getattr(ep_fc2, f"weight{local_e}").copy_(getattr(ref_fc2, f"weight{global_e}"))


def test_layer_ep_matches_local(
    rank: int, ep_size: int, ep_group, num_microbatches: int = 1
) -> None:
    """Full DeepSeekV3Layer with EP must match the all-experts-local layer numerically."""
    num_experts = NUM_LOCAL_EXPERTS * ep_size
    torch.manual_seed(0)
    ref = _make_layer(None, ep_size, num_experts)
    _broadcast_params(ref)
    ep_layer = _make_layer(ep_group, ep_size, num_experts)
    _copy_weights(ep_layer, ref, rank)

    torch.manual_seed(1234 + rank)
    microbatches = []
    for mb in range(num_microbatches):
        seq_len = TOKENS_PER_RANK // 2 - 8 * mb
        x = torch.randn(seq_len, 2, HIDDEN, dtype=DTYPE, device="cuda")
        x_ep = x.clone().requires_grad_(True)
        x_ref = x.clone().requires_grad_(True)
        out_ep = ep_layer(x_ep)
        out_ref = ref(x_ref)
        assert out_ep.shape == x.shape
        torch.testing.assert_close(out_ep, out_ref, rtol=0.05, atol=0.05)
        microbatches.append((x_ep, x_ref, out_ep, out_ref))

    # All microbatches must retain their routing until their own backward.
    for x_ep, x_ref, out_ep, out_ref in microbatches:
        grad_out = torch.randn_like(out_ep)
        out_ep.backward(grad_out.clone())
        out_ref.backward(grad_out.clone())
        torch.testing.assert_close(x_ep.grad, x_ref.grad, rtol=0.05, atol=0.05)

    ref_params = dict(ref.named_parameters())
    for name, p in ep_layer.named_parameters():
        if name.startswith("mlp.experts.") or p.grad is None:
            continue
        torch.testing.assert_close(p.grad, ref_params[name].grad, rtol=0.1, atol=0.1, msg=name)

    # A local expert's wgrad on its owner rank equals the sum of the
    # reference wgrads over all ranks. all_reduce is collective, so every
    # rank must reduce every expert's grad (in the same order).
    ep_fc1, _, ep_fc2 = ep_layer.mlp.experts
    ref_fc1, _, ref_fc2 = ref.mlp.experts
    for ep_fc, ref_fc in ((ep_fc1, ref_fc1), (ep_fc2, ref_fc2)):
        ref_grads = [getattr(ref_fc, f"weight{e}").grad.float().clone() for e in range(num_experts)]
        for g in ref_grads:
            dist.all_reduce(g)
        for local_e in range(NUM_LOCAL_EXPERTS):
            global_e = rank * NUM_LOCAL_EXPERTS + local_e
            ep_grad = getattr(ep_fc, f"weight{local_e}").grad.float()
            torch.testing.assert_close(ep_grad, ref_grads[global_e], rtol=0.1, atol=0.1)

    counts = ep_layer.mlp._last_tokens_per_expert.clone()
    dist.all_reduce(counts)
    last_num_tokens = microbatches[-1][0].numel() // HIDDEN
    assert counts.sum().item() == ep_size * last_num_tokens * TOP_K

    ep_layer.mlp.update_expert_bias()
    assert torch.isfinite(ep_layer.mlp.expert_bias).all()


def test_mla_tp_matches_local(rank: int, tp_size: int, tp_group) -> None:
    """Compare sharded MLA outputs and gradients with an unsharded reference."""
    torch.backends.cuda.matmul.allow_tf32 = False
    for fmt in ("sbhd", "bshd"):
        for explicit_size in (False, True):
            torch.manual_seed(0)
            common = dict(params_dtype=torch.float32, qkv_format=fmt, **MLA_KWARGS)
            ref = MultiLatentAttention(HIDDEN, 2 * tp_size, **common)
            _broadcast_params(ref)
            tp = MultiLatentAttention(
                HIDDEN,
                2 * tp_size,
                tp_group=tp_group,
                **({"tp_size": tp_size} if explicit_size else {}),
                **common,
            )
            ref_params = dict(ref.named_parameters())

            def shard(name, tensor):
                if name in ("q_up_proj.weight", "kv_up_proj.weight"):
                    return tensor.chunk(tp_size, dim=0)[rank]
                if name == "out_proj.weight":
                    return tensor.chunk(tp_size, dim=1)[rank]
                return tensor

            with torch.no_grad():
                for name, param in tp.named_parameters():
                    param.copy_(shard(name, ref_params[name]))

            torch.manual_seed(1234)
            shape = (16, 2, HIDDEN) if fmt == "sbhd" else (2, 16, HIDDEN)
            x = torch.randn(shape, device="cuda", requires_grad=True)
            x_ref = x.detach().clone().requires_grad_()
            out = tp(x)
            out_ref = ref(x_ref)
            torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)
            grad = torch.randn_like(out)
            out.backward(grad.clone())
            out_ref.backward(grad.clone())
            torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-3, atol=1e-3)
            for name, param in tp.named_parameters():
                torch.testing.assert_close(
                    param.grad,
                    shard(name, ref_params[name].grad),
                    rtol=1e-3,
                    atol=1e-3,
                    msg=name,
                )


def main() -> int:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if "--tp" in sys.argv:
        test_mla_tp_matches_local(dist.get_rank(), dist.get_world_size(), dist.group.WORLD)
        print(f"[rank {dist.get_rank()}] TP PASSED")
        dist.destroy_process_group()
        return 0
    from torch.distributed import _symmetric_memory as _symm_mem

    _symm_mem.set_backend("NCCL")

    rank = dist.get_rank()
    ep_size = dist.get_world_size()
    if _device_sm() < 90:
        if rank == 0:
            print(f"NCCL EP requires SM>=90 (got SM{_device_sm()}); skipping.")
        dist.destroy_process_group()
        return 0

    ep_group = dist.new_group(ranks=list(range(ep_size)), backend="nccl")
    ep_bootstrap(
        ep_group,
        num_experts=NUM_LOCAL_EXPERTS * ep_size,
        max_tokens_per_rank=TOKENS_PER_RANK,
        hidden_dim=HIDDEN,
        num_topk=TOP_K,
        recv_capacity_per_rank=_recv_capacity(ep_size),
    )
    for num_microbatches in (1, 3):
        test_layer_ep_matches_local(rank, ep_size, ep_group, num_microbatches)
    print(f"[rank {rank}] PASSED")

    dist.barrier()
    ep_finalize()
    release_symm_mem_pool()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
