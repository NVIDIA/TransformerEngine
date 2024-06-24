#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import subprocess
import argparse
import warnings
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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

    def forward(self, x, attention_mask=None):
        output = self.mhsa(x, x, x, attn_mask=attention_mask, need_weights=False)
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
        fc1_output_features = ffn_hidden_size
        self.gelu = nn.GELU(approximate="tanh")

        self.fc1 = nn.Linear(hidden_size, fc1_output_features)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(self.ln(x))))


class TorchGPT(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float, num_attention_heads: int,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=eps)
        self.causal_attn = TorchMHA(hidden_size, num_attention_heads)
        self.ln_mlp = TorchLayerNormMLP(hidden_size, 4 * hidden_size, eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        a = self.ln(x)
        b = self.causal_attn(a, attention_mask)
        x = x + b
        n = self.ln_mlp(x)
        x = x + n
        return x

torch_dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

atol_map = {
    torch.float32 : 5e-3,
    torch.float16 : 5e-2,
    torch.bfloat16 : 5e-2,
}

def mapped_argtype(opt, typemap={}):
    if str(opt).lower() not in typemap.keys():
        raise TypeError(f"Unrecognized option! Please choose from: {typemap.keys()}")
    return typemap[str(opt).lower()]

def parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Test te.TransformerLayer with GEMM+comm overlap."
    )
    parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Input sequence length.")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=12, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=64, help="Dimension of each attention head."
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enables the te.fp8_autocast() context."
    )
    parser.add_argument(
        "--dtype", type=partial(mapped_argtype, typemap=torch_dtypes), default=torch.bfloat16,
    )
    parser.add_argument(
        "--no-mask", action="store_true", default=False, help="Disable the causal attention mask."
    )
    parser.add_argument(
        "--no-overlap", action="store_true", default=False, help="Disable comm+GEMM overlap."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Print out debug info."
    )
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

    # Info printout
    def dist_print(msg, src=None, debug=False, section=False):
        if debug or opts.verbose:
            if section:
                dist.barrier()
                if WORLD_RANK == (0 if src is None else src):
                    print("\n", end="", flush=True)
            dist.barrier()
            if src is None or WORLD_RANK == src:
                prefix = "[GLOBAL] " if src is not None else f"[rank:{WORLD_RANK}] "
                lines = msg.splitlines()
                msg = "\n".join(
                    [prefix + lines[0]] + [(" " * len(prefix)) + line for line in lines[1:]]
                )
                print(msg + "\n", end="", flush=True)

    # Intialize userbuffers
    hidden_size = opts.num_heads * opts.head_dim
    batched_size = opts.seq_length * opts.batch_size
    if not opts.no_overlap:
        te.initialize_ub(
            [batched_size, hidden_size],
            tp_group,
            use_fp8=opts.fp8,
            dtype=opts.dtype,
        )

    # Initialize test and reference model, and share parameters
    te_kwargs = {
        'layernorm_epsilon' : 1e-5,
        'hidden_dropout' : 0.0,
        'attention_dropout' : 0.0,
        'fuse_qkv_params' : True,
        'qkv_weight_interleaved' : False,
        'device' : 'cuda',
        'params_dtype' : opts.dtype,
        'seq_length' : opts.seq_length,
        'set_parallel_mode' : WORLD_SIZE > 1,
        'tp_group' : tp_group if WORLD_SIZE > 1 else None,
        'sequence_parallel' : WORLD_SIZE > 1,
        'ub_tp_comm_overlap' : not opts.no_overlap,
        'parallel_attention_mlp' : False,
    }
    te_gpt = te.TransformerLayer(
        hidden_size,
        4 * hidden_size,
        opts.num_heads,
        **te_kwargs
    ).eval()

    torch_gpt = TorchGPT(hidden_size, 1e-5, opts.num_heads).cuda().to(opts.dtype).eval()
    with torch.no_grad():
        # Clone input layernorm params
        # TransformerLayer.MultiheadAttention.LayerNormLinear --> TorchGPT.LayerNorm
        dist_print(
            f"TE input LN weight: {te_gpt.self_attention.layernorm_qkv.layer_norm_weight.size()}",
            section=True)
        torch_gpt.ln.weight = nn.Parameter(
            te_gpt.self_attention.layernorm_qkv.layer_norm_weight.clone())
        dist_print(f"Torch input LN weight: {torch_gpt.ln.weight.size()}", src=0)

        dist_print(
            f"TE input LN bias: {te_gpt.self_attention.layernorm_qkv.layer_norm_bias.size()}")
        torch_gpt.ln.bias = nn.Parameter(
            te_gpt.self_attention.layernorm_qkv.layer_norm_bias.clone())
        dist_print(f"Torch input LN bias: {torch_gpt.ln.bias.size()}", src=0)

        # Clone QKV projection params
        # TransformerLayer.MultiheadAttention.LayerNormLinear --> TorchGPT.MultiheadAttention
        dist_print(f"TE QKV proj weight: {te_gpt.self_attention.layernorm_qkv.weight.size()}",
                   section=True)
        torch_gpt.causal_attn.mhsa.in_proj_weight = nn.Parameter(
            te.distributed.gather_along_first_dim(
                te_gpt.self_attention.layernorm_qkv.weight.clone(), tp_group)[0]
            if WORLD_SIZE > 1
            else te_gpt.self_attention.layernorm_qkv.weight.clone())
        dist_print(f"Torch QKV proj weight: {torch_gpt.causal_attn.mhsa.in_proj_weight.size()}",
                   src=0)

        dist_print(f"TE QKV proj bias: {te_gpt.self_attention.layernorm_qkv.bias.size()}")
        torch_gpt.causal_attn.mhsa.in_proj_bias = nn.Parameter(
            te.distributed.gather_along_first_dim(
                te_gpt.self_attention.layernorm_qkv.bias.clone(), tp_group)[0]
            if WORLD_SIZE > 1
            else te_gpt.self_attention.layernorm_qkv.bias.clone())
        dist_print(f"Torch QKV proj bias: {torch_gpt.causal_attn.mhsa.in_proj_bias.size()}", src=0)

        # Clone MHA projection params
        # TransformerLayer.MultiheadAttention.Linear --> TorchGPT.MultiheadAttention
        dist_print(f"TE MHA out proj weights: {te_gpt.self_attention.proj.weight.size()}",
                   section=True)
        torch_gpt.causal_attn.mhsa.out_proj.weight = nn.Parameter(
            torch.transpose(
                te.distributed.gather_along_first_dim(
                    torch.transpose(te_gpt.self_attention.proj.weight.clone(), 0, 1), tp_group)[0],
                0, 1)
            if WORLD_SIZE > 1
            else te_gpt.self_attention.proj.weight.clone())
        dist_print(
            f"Torch MHA out proj weights: {torch_gpt.causal_attn.mhsa.out_proj.weight.size()}",
            src=0)

        dist_print(f"TE MHA out proj bias: {te_gpt.self_attention.proj.bias.size()}")
        torch_gpt.causal_attn.mhsa.out_proj.bias = nn.Parameter(
            te_gpt.self_attention.proj.bias.clone())
        dist_print(f"Torch MHA out proj bias: {torch_gpt.causal_attn.mhsa.out_proj.bias.size()}",
                   src=0)

        # Clone LayerNormMLP params
        # TransformerLayer.LayerNormMLP --> TorchGPT.LayerNormMLP.Linear
        dist_print(f"TE LN-MLP LN weights: {te_gpt.layernorm_mlp.layer_norm_weight.size()}",
                   section=True)
        torch_gpt.ln_mlp.ln.weight = nn.Parameter(te_gpt.layernorm_mlp.layer_norm_weight.clone())
        dist_print(f"Torch LN-MLP LN weights: {torch_gpt.ln_mlp.ln.weight.size()}", src=0)

        dist_print(f"TE LN-MLP LN bias: {te_gpt.layernorm_mlp.layer_norm_bias.size()}")
        torch_gpt.ln_mlp.ln.bias = nn.Parameter(te_gpt.layernorm_mlp.layer_norm_bias.clone())
        dist_print(f"Torch LN-MLP LN bias: {torch_gpt.ln_mlp.ln.bias.size()}", src=0)

        dist_print(f"TE LN-MLP FC1 weights: {te_gpt.layernorm_mlp.fc1_weight.size()}")
        torch_gpt.ln_mlp.fc1.weight = nn.Parameter(
            te.distributed.gather_along_first_dim(
                te_gpt.layernorm_mlp.fc1_weight.clone(), tp_group)[0]
            if WORLD_SIZE > 1
            else te_gpt.layernorm_mlp.fc1_weight.clone())
        dist_print(f"Torch LN-MLP FC1 weights: {torch_gpt.ln_mlp.fc1.weight.size()}", src=0)

        dist_print(f"TE LN-MLP FC1 bias: {te_gpt.layernorm_mlp.fc1_bias.size()}",
                   section=True)
        torch_gpt.ln_mlp.fc1.bias = nn.Parameter(
            te.distributed.gather_along_first_dim(
                te_gpt.layernorm_mlp.fc1_bias.clone(), tp_group)[0]
            if WORLD_SIZE > 1
            else te_gpt.layernorm_mlp.fc1_bias.clone())
        dist_print(f"Torch LN-MLP FC1 bias: {torch_gpt.ln_mlp.fc1.bias.size()}", src=0)

        dist_print(f"TE LN-MLP FC2 weights: {te_gpt.layernorm_mlp.fc2_weight.size()}")
        torch_gpt.ln_mlp.fc2.weight = nn.Parameter(
            torch.transpose(
                te.distributed.gather_along_first_dim(
                    torch.transpose(te_gpt.layernorm_mlp.fc2_weight.clone(), 0, 1), tp_group
                )[0], 0, 1)
            if WORLD_SIZE > 1
            else te_gpt.layernorm_mlp.fc2_weight.clone())
        dist_print(f"Torch LN-MLP FC2 weights: {torch_gpt.ln_mlp.fc2.weight.size()}", src=0)

        dist_print(f"TE LN-MLP FC2 bias: {te_gpt.layernorm_mlp.fc2_bias.size()}")
        torch_gpt.ln_mlp.fc2.bias = nn.Parameter(te_gpt.layernorm_mlp.fc2_bias.clone())
        dist_print(f"Torch LN-MLP FC2 bias: {torch_gpt.ln_mlp.fc2.bias.size()}", src=0)

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

    # Generate causal attention mask and random input in SBHD format
    causal_mask = torch.triu(
        torch.ones(opts.seq_length, opts.seq_length, device="cuda"), diagonal=1
    ).bool() if not opts.no_mask else None
    x = torch.rand(
        (opts.seq_length // tp_size, opts.batch_size, hidden_size),
        dtype=opts.dtype,
        device="cuda",
        requires_grad=True,
    )
    x.retain_grad()
    dist_print(f"Distributed input: {x.size()}", section=True)

    # Forward + backward passes + globalize output for numerical check
    torch.cuda.synchronize()
    dist.barrier()
    with te.fp8_autocast(enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=tp_group):
        y = te_gpt(x, attention_mask=causal_mask)
        loss = y.sum()
    loss.backward()
    dist_print(f"Distributed output: {y.size()}")
    yg = te.distributed.gather_along_first_dim(y, tp_group)[0]
    dist_print(f"Global output: {yg.size()}", src=0)

    # Globalize input and compute reference results
    xr = te.distributed.gather_along_first_dim(x.clone(), tp_group)[0].requires_grad_(True)
    dist_print(f"Reference input: {xr.size()}", src=0, section=True)
    yr = torch_gpt(xr, attention_mask=causal_mask)
    dist_print(f"Reference output: {yr.size()}", src=0)

    # Compare against reference
    atol = 0.0675 if opts.fp8 else atol_map[opts.dtype]
    diff = torch.abs(yg - yr).flatten()
    max_idx = torch.argmax(diff)
    min_idx = torch.argmin(diff)
    numerics_failed = not torch.allclose(yg, yr, atol=atol)
    if numerics_failed:
        max_idx = torch.argmax(diff)
        min_idx = torch.argmin(diff)
        failed_idx = diff > atol
        num_fails = failed_idx.int().sum()
        fail_ratio = failed_idx.float().mean()
        result_info = (
            "NUMERICAL CHECK FAILED! "
            + f"\nError above tolerance ({atol}) for {num_fails} elements ({fail_ratio*100}%). "
            + f"\nMax diff {diff[max_idx].item()} at index {max_idx.item()} "
            + f"({yg.flatten()[max_idx].item()} vs {yr.flatten()[max_idx].item()})."
        )
        if yg.flatten()[min_idx].item() != yr.flatten()[min_idx].item() != 0.0:
            result_info += (
                f"\nMin diff {diff[min_idx].item()} at index {min_idx.item()} "
                + f"({yg.flatten()[min_idx].item()} vs {yr.flatten()[min_idx].item()})."
            )
        dist_print(result_info, src=0, section=True)
    else:
        result_info = (
            "NUMERICAL CHECK PASSED: "
            + f"\nMax diff {diff[max_idx].item()} at index {max_idx.item()}"
            + f"and min diff {diff[min_idx].item()} at index {min_idx.item()}."
        )
        dist_print(result_info, src=0, section=True)

    # Now compare overlapped layer against no-overlap
    if not numerics_failed and not opts.no_overlap:
        # Create new TransformerLayer without comm overlap
        te_kwargs['ub_tp_comm_overlap'] = False
        te_gpt_no_overlap = te.TransformerLayer(
            hidden_size,
            4 * hidden_size,
            opts.num_heads,
            **te_kwargs,
        )
        # Clone parameters from original layer
        with torch.no_grad():
            for p1, p2 in zip(te_gpt.parameters(), te_gpt_no_overlap.parameters()):
                p2.copy_(p1)

        # Run forward and backward passes
        xn = x.detach().clone().requires_grad_(True)
        xn.retain_grad()
        with te.fp8_autocast(enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=tp_group):
            yn = te_gpt_no_overlap(xn, attention_mask=causal_mask)
            ln = yn.sum()
        ln.backward()

        # Assemble list of tensors to be compared
        overlap = [ y, x.grad ]
        names = ['output', 'input_grad']
        for name, param in te_gpt.named_parameters():
            if param.requires_grad:
                overlap.append(param)
                names.append(name)
        no_overlap = [ yn, xn.grad ]
        for param in te_gpt_no_overlap.parameters():
            if param.requires_grad:
                no_overlap.append(param)

        if len(overlap) == len(no_overlap):
            for i, (o, n) in enumerate(zip(overlap, no_overlap)):
                numerics_failed = not torch.allclose(o, n, atol=atol)
                if numerics_failed:
                    diff = torch.abs(o - n).flatten()
                    max_idx = torch.argmax(diff)
                    failed_idx = diff > atol
                    num_fails = failed_idx.int().sum()
                    fail_ratio = failed_idx.float().mean()
                    mismatch_info = (
                        f"OVERLAP MISMATCH in '{names[i]}'!"
                        + f"\nDiff above tolerance ({atol}) for {num_fails} elements "
                        + f"({fail_ratio.item()*100.}%)."
                        + f"\nMax diff {diff[max_idx].item()} at index {max_idx} "
                        + f"({o.flatten()[max_idx].item()} vs {n.flatten()[max_idx].item()})."
                    )
                    dist_print(mismatch_info, section=i == 0)
                    break

        torch.cuda.synchronize()
        dist.barrier()
        te.destroy_ub()

    dist.destroy_process_group()

    return int(numerics_failed)


if __name__ == "__main__":
    if "TORCHELASTIC_RUN_ID" in os.environ.keys():
        args = parse_args()
        os._exit(train(args))
    else:
        subprocess.run(
            ["torchrun", f"--nproc-per-node={torch.cuda.device_count()}", *sys.argv],
            env=os.environ,
            check=True,
        )
        os._exit(0)
