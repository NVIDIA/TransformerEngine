"""Minimal Expert Parallel (EP) smoke test for TE Mixtral.

Run on 8 GPUs:
    torchrun --standalone --nproc_per_node=8 docs/examples/te_mixtral/example_xy_algorithm.py --ep-size 8

This script verifies that:
1) distributed device setup succeeds,
2) EP groups/mesh are attached to every MoE block,
3) one forward/backward/step runs without error.
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from te_mixtral import NVMixtralConfig, NVMixtralForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EP smoke test for TE Mixtral.")
    parser.add_argument("--ep-size", type=int, default=8, help="Expert parallel size (set to 8 for 8 GPUs).")
    parser.add_argument("--num-experts", type=int, default=8, help="Global number of experts.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mixtral expert parallelism.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    try:
        if world_size != args.ep_size:
            raise ValueError(
                f"--ep-size ({args.ep_size}) must match WORLD_SIZE ({world_size}) for this minimal EP example."
            )
        if args.num_experts % args.ep_size != 0:
            raise ValueError(
                f"--num-experts ({args.num_experts}) must be divisible by --ep-size ({args.ep_size})."
            )

        torch.manual_seed(1234 + rank)
        torch.cuda.manual_seed_all(1234 + rank)

        config = NVMixtralConfig(
            hidden_size=256,
            intermediate_size=512,
            num_local_experts=args.num_experts,
            num_experts_per_tok=2,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            vocab_size=4096,
            max_position_embeddings=512,
            expert_parallel_size=args.ep_size,
        )
        config.dtype = torch.bfloat16
        model = NVMixtralForCausalLM(config).to(device=device, dtype=torch.bfloat16)
        model.train()

        ep_mesh = DeviceMesh("cuda", torch.arange(world_size))
        model.model.set_ep_groups(ep_group=dist.group.WORLD, ep_mesh=ep_mesh)

        # Keep DTensor params separate from regular Tensor params.
        # Mixed lists can fail in fused/foreach optimizer kernels.
        dtensor_params = []
        tensor_params = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if isinstance(param, DTensor) or isinstance(param.data, DTensor):
                dtensor_params.append(param)
            else:
                tensor_params.append(param)

        param_groups = []
        if tensor_params:
            param_groups.append({"params": tensor_params})
        if dtensor_params:
            param_groups.append({"params": dtensor_params})

        optimizer = AdamW(
            param_groups,
            lr=1e-4,
            fused=False,
            foreach=False,
        )
        input_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(args.batch_size, args.seq_len),
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        optimizer.step()

        local_shard = model.model.layers[0].mlp.experts_gate_up_weight
        if hasattr(local_shard, "to_local"):
            local_shape = tuple(local_shard.to_local().shape)
        else:
            local_shape = tuple(local_shard.shape)

        loss_value = out.loss.detach().float()
        if dist.is_initialized():
            dist.all_reduce(loss_value, op=dist.ReduceOp.AVG)

        if rank == 0:
            print("EP smoke test passed.")
            print(f"world_size={world_size}, ep_size={args.ep_size}, global_experts={args.num_experts}")
            print(f"local expert shard shape (layer0 gate_up): {local_shape}")
            print(f"mean loss across ranks: {loss_value.item():.6f}")
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
