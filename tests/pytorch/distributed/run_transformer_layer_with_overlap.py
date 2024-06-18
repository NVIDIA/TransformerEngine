#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import subprocess
import argparse

import torch
import torch.distributed as dist
from torch import nn

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


class TorchLayerNorm(nn.Module):
    def __init__(self, in_features: int, eps: float, zero_centered_gamma: bool):
        super().__init__()
        self.eps = eps
        self.in_features = in_features
        self.zero_centered_gamma = zero_centered_gamma

        initial_value = torch.ones(in_features) if zero_centered_gamma else torch.zeros(in_features)
        self.weight = nn.Parameter(initial_value)
        self.bias = nn.Parameter(torch.zeros(in_features))
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight if not self.zero_centered_gamma else 1 + self.weight
        w = w.to(torch.float32)
        b = self.bias.to(torch.float32)
        inp = x.to(torch.float32)
        out = torch.nn.functional.layer_norm(
            inp, (self.in_features,), weight=w, bias=b, eps=self.eps
        )
        return out.to(x.dtype)


class TorchMHA(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=0.1,
            bias=True,
            batch_first=False,
        )

    def forward(self, x):
        output = self.mhsa(x, x, x, need_weights=False)
        if isinstance(output, tuple):
            output = output[0]
        return output


class TorchLayerNormMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.ln = TorchLayerNorm(hidden_size, eps=eps, zero_centered_gamma=False)
        self.gelu = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(self.ln(x))))


class TorchGPT(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float, num_attention_heads: int, parallel_attention_mlp: bool
    ):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=eps)
        self.causal_attn = TorchMHA(hidden_size, num_attention_heads)
        self.ln_mlp = TorchLayerNormMLP(hidden_size, 4 * hidden_size, eps)
        self.parallel_attention_mlp = parallel_attention_mlp

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        a = self.ln(x)
        b = self.causal_attn(a)
        x = x + nn.functional.dropout(b, p=0.1, training=self.training)
        n = self.ln_mlp(x)
        x = x + nn.functional.dropout(n, p=0.1, training=self.training)


def parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Test a TE layer module with GEMM+comm overlap."
    )
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=64, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=128, help="Dimension of each attention head."
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enables the te.fp8_autocast() context."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print out debug info.")
    return parser.parse_args(argv, namespace)

def train(opts):
    WORLD_RANK = int(os.getenv("RANK"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE"))

    # Seed RNG
    torch.cuda.set_device(WORLD_RANK)
    torch.manual_seed(opts.seed + WORLD_RANK)
    torch.cuda.manual_seed(opts.seed + WORLD_RANK)

    # Initialize torch.distributed global process group and get TP group
    dist.init_process_group(
        backend="nccl",
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
        device_id=torch.device(f"cuda:{WORLD_RANK}"),
    )
    tp_group = dist.new_group(backend="nccl")
    tp_size = dist.get_world_size(tp_group)

    # Intialize userbuffers
    hidden_size = opts.num_heads * opts.head_dim
    batched_size = opts.seq_length * opts.batch_size
    te.initialize_ub(
        [batched_size, hidden_size],
        tp_group,
        use_fp8=opts.fp8,
        dtype=torch.bfloat16,
    )

    # Initialize test and reference model, and share parameters
    te_gpt = te.TransformerLayer(
        hidden_size,
        4 * hidden_size,
        opts.num_heads,
        layernorm_epsilon=1e-5,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        fuse_qkv_params=True,
        qkv_weight_interleaved=False,
        params_dtype=torch.bfloat16,
        seq_length=opts.seq_length,
        set_parallel_mode=True,
        tp_group=tp_group,
        sequence_parallel=True,
        ub_tp_comm_overlap=True,
        parallel_attention_mlp=False,
    )

    torch_gpt = TorchGPT(hidden_size, 1e-5, opts.num_heads, False)
    with torch.no_grad():
        # Clone input layernorm params
        print(f"[rank:{WORLD_RANK}] TE input LN weight: {te_gpt.self_attention.layernorm_qkv.layer_norm_weight.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch input LN weight: {torch_gpt.ln.weight.size()}\n", end='', flush=True)
        torch_gpt.ln.weight = nn.Parameter(
            te_gpt.self_attention.layernorm_qkv.layer_norm_weight.clone()
        )

        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE input LN bias: {te_gpt.self_attention.layernorm_qkv.layer_norm_bias.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch input LN bias: {torch_gpt.ln.bias.size()}\n", end='', flush=True)
        torch_gpt.ln.bias = nn.Parameter(te_gpt.self_attention.layernorm_qkv.layer_norm_bias.clone())

        # Clone QKV projection params
        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE QKV weight: {te_gpt.self_attention.layernorm_qkv.weight.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch QKV weight: {torch_gpt.causal_attn.mhsa.in_proj_weight.size()}\n", end='', flush=True)
        torch_gpt.causal_attn.mhsa.in_proj_weight = nn.Parameter(
            te.distributed.gather_along_first_dim(
                te_gpt.self_attention.layernorm_qkv.weight, tp_group
            )[0]
        )

        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE QKV bias: {te_gpt.self_attention.layernorm_qkv.bias.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch QKV bias: {torch_gpt.causal_attn.mhsa.in_proj_bias.size()}\n", end='', flush=True)
        torch_gpt.causal_attn.mhsa.in_proj_bias = nn.Parameter(
            te.distributed.gather_along_first_dim(
                te_gpt.self_attention.layernorm_qkv.bias, tp_group
            )[0]
        )

        # Clone MHA projection params
        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE proj weights: {te_gpt.self_attention.proj.weight.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch proj weights: {torch_gpt.causal_attn.mhsa.out_proj.weight.size()}\n", end='', flush=True)
        torch_gpt.causal_attn.mhsa.out_proj.weight = nn.Parameter(
            torch.transpose(
                te.distributed.gather_along_first_dim(
                    torch.transpose(te_gpt.self_attention.proj.weight, 0, 1), tp_group
                )[0], 0, 1
            )
        )

        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE proj bias: {te_gpt.self_attention.proj.bias.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch proj bias: {torch_gpt.causal_attn.mhsa.out_proj.bias.size()}\n", end='', flush=True)
        torch_gpt.causal_attn.mhsa.out_proj.bias = nn.Parameter(
            te_gpt.self_attention.proj.bias.clone()
        )

        # Clone LayerNormMLP params
        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE output LN weights: {te_gpt.layernorm_mlp.layer_norm_weight.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch output LN weights: {torch_gpt.ln_mlp.ln.weight.size()}\n", end='', flush=True)
        torch_gpt.ln_mlp.ln.weight = nn.Parameter(te_gpt.layernorm_mlp.layer_norm_weight.clone())

        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE output LN bias: {te_gpt.layernorm_mlp.layer_norm_bias.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch output LN bias: {torch_gpt.ln_mlp.ln.bias.size()}\n", end='', flush=True)
        torch_gpt.ln_mlp.ln.bias = nn.Parameter(te_gpt.layernorm_mlp.layer_norm_bias.clone())

        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE MLP FC1 weights: {te_gpt.layernorm_mlp.fc1_weight.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch MLP FC1 weights: {torch_gpt.ln_mlp.fc1.weight.size()}\n", end='', flush=True)
        torch_gpt.ln_mlp.fc1.weight = nn.Parameter(
            te.distributed.gather_along_first_dim(
                te_gpt.layernorm_mlp.fc1_weight, tp_group)[0]
        )

        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE MLP FC1 bias: {te_gpt.layernorm_mlp.fc1_bias.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch MLP FC1 bias: {torch_gpt.ln_mlp.fc1.bias.size()}\n", end='', flush=True)
        torch_gpt.ln_mlp.fc1.bias = nn.Parameter(
            te.distributed.gather_along_first_dim(
                te_gpt.layernorm_mlp.fc1_bias, tp_group)[0]
        )

        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE MLP FC2 weights: {te_gpt.layernorm_mlp.fc2_weight.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch MLP FC2 weights: {torch_gpt.ln_mlp.fc2.weight.size()}\n", end='', flush=True)
        torch_gpt.ln_mlp.fc2.weight = nn.Parameter(
            torch.transpose(
                te.distributed.gather_along_first_dim(
                    torch.transpose(te_gpt.layernorm_mlp.fc2_weight, 0, 1), tp_group
                )[0], 0, 1
            )
        )

        dist.barrier()
        print(f"[rank:{WORLD_RANK}] TE MLP FC2 bias: {te_gpt.layernorm_mlp.fc2_bias.size()}\n", end='', flush=True)
        if WORLD_RANK == 0: print(f"[GLOBAL] Torch MLP FC2 bias: {torch_gpt.ln_mlp.fc2.bias.size()}\n", end='', flush=True)
        torch_gpt.ln_mlp.fc2.bias = nn.Parameter(te_gpt.layernorm_mlp.fc2_bias.clone())

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

    # Generate random input in SBHD format
    x = torch.rand(
        (opts.seq_length // tp_size, opts.batch_size, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )

    # Forward + backward passes
    with te.fp8_autocast(enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=tp_group):
        y = te_gpt(x)
        loss = y.flatten().sum()
    loss.backward()

    # Globalize input and compute reference results
    xr = te.distributed.gather_along_first_dim(x, tp_group)[0].requires_grad_(True)
    yr = torch_gpt(xr)
    lr = yr.flatten().sum()
    lr.backward()
    references = [ yr, xr.grad ]
    names = [ "output", "input_grad" ]
    for n, p in ref_model.named_parameters():
        if p.requires_grad:
            references.append(p.grad)
            names.append(n)

    # Globalize outputs and compare against reference
    torch.cuda.synchronize()
    dist.barrier()
    outputs = [
        te.distributed.gather_along_first_dim(y, tp_group)[0],
        te.distributed.gather_along_first_dim(x.grad, tp_group)[0],
    ]
    for p in te_model.parameters():
        if p.requires_grad:
            outputs.append(
                te.distributed.gather_along_first_dim(p.grad, tp_group)[0]
            )

    length_failed = len(outputs) != len(references)
    numerics_failed = False
    fail_msg = None
    if not length_failed:
        for i, (out, ref) in enumerate(zip(outputs, references)):
            torch.cuda.synchronize()
            dist.barrier()
            numerics_failed = not torch.allclose(out, ref,
                                                 rtol=0.125 if opts.fp8 else 1.6e-2,
                                                 atol=0.0675 if opts.fp8 else 1e-5)
            if numerics_failed:
                diff = torch.abs(out - ref).flatten()
                m = torch.argmax(diff)
                fail_msg = (
                    "NUMERICAL CHECK FAILED: "
                    + f"{names[i]} tensor not close enough at index {m.item()} "
                    + f"with {out.flatten()[m].item()} vs {ref.flatten()[m].item()} "
                    + f"(diff {diff[m].item()})."
                )
                break
    else:
        fail_msg = (
            "INCORRECT NUMBER OF OUTPUTS: "
            + f"Expected {len(references)} tensors but got {len(outputs)}."
        )

    if fail_msg is not None and WORLD_RANK == 0:
        print(fail_msg + '\n', end='', flush=True)

    te.destroy_ub()
    dist.destroy_process_group()

    return int(length_failed or numerics_failed)


if __name__ == "__main__":
    if "TORCHELASTIC_RUN_ID" in os.environ.keys():
        args = parse_args()
        train(args)
    else:
        subprocess.run(
            ["torchrun", f"--nproc-per-node={torch.cuda.device_count()}", *sys.argv],
            env=os.environ,
            check=True,
        )
    os._exit(0)
