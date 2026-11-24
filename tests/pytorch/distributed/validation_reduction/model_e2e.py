# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Full Qwen3 inference with two-rank TE attention projections and RMSNorm/MLPs."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as ops


class RowProjection(torch.nn.Module):
    """Shard checkpoint columns; gather sequence-parallel results for HF attention."""

    def __init__(self, original, kind, sp, reduction_dtype, rank):
        super().__init__()
        self.sp, self.reduction_dtype, self.rank = sp, reduction_dtype, rank
        self.kind = kind
        in_features, out_features = original.in_features, original.out_features
        if kind == "ops.Linear":
            option = {} if reduction_dtype is None else {"reduction_dtype": reduction_dtype}
            self.linear = ops.Linear(
                in_features,
                out_features,
                bias=False,
                dtype=original.weight.dtype,
                tensor_parallel_mode="row",
                tensor_parallel_group=dist.group.WORLD,
                sequence_parallel=sp,
                **option,
            )
        else:
            self.linear = te.Linear(
                in_features,
                out_features,
                bias=False,
                params_dtype=original.weight.dtype,
                parallel_mode="row",
                tp_group=dist.group.WORLD,
                tp_size=2,
                sequence_parallel=sp,
            )
        with torch.no_grad():
            self.linear.weight.copy_(original.weight.chunk(2, dim=1)[rank])

    def forward(self, inp):
        shape = inp.shape
        x = inp.reshape(-1, shape[-1])
        rows = x.shape[0]
        if self.sp and rows % 2:
            x = torch.cat([x, torch.zeros_like(x[:1])])
        x = x.chunk(2, dim=-1)[self.rank].contiguous()
        option = {} if self.reduction_dtype is None else {"reduction_dtype": self.reduction_dtype}
        out = self.linear(x) if self.kind == "ops.Linear" else self.linear(x, **option)
        if self.sp:
            parts = [torch.empty_like(out) for _ in range(2)]
            dist.all_gather(parts, out)
            out = torch.cat(parts)
        return out[:rows].reshape(*shape[:-1], out.shape[-1])


class NormalizedMLP(torch.nn.Module):
    """Fuse the checkpoint post-attention RMSNorm and SwiGLU MLP."""

    def __init__(self, norm, original, sp, reduction_dtype, rank):
        super().__init__()
        self.sp, self.reduction_dtype, self.rank = sp, reduction_dtype, rank
        hidden, intermediate = original.gate_proj.in_features, original.gate_proj.out_features
        self.mlp = te.LayerNormMLP(
            hidden,
            intermediate,
            eps=norm.variance_epsilon,
            bias=False,
            normalization="RMSNorm",
            activation="swiglu",
            params_dtype=original.gate_proj.weight.dtype,
            set_parallel_mode=True,
            tp_group=dist.group.WORLD,
            tp_size=2,
            sequence_parallel=sp,
        )
        with torch.no_grad():
            self.mlp.layer_norm_weight.copy_(norm.weight)
            self.mlp.fc1_weight.copy_(
                torch.cat(
                    [
                        original.gate_proj.weight.chunk(2)[rank],
                        original.up_proj.weight.chunk(2)[rank],
                    ]
                )
            )
            self.mlp.fc2_weight.copy_(original.down_proj.weight.chunk(2, dim=1)[rank])

    def forward(self, inp):
        shape = inp.shape
        x = inp.reshape(-1, shape[-1])
        rows = x.shape[0]
        if self.sp:
            if rows % 2:
                x = torch.cat([x, torch.zeros_like(x[:1])])
            x = x.chunk(2)[self.rank].contiguous()
        option = {} if self.reduction_dtype is None else {"reduction_dtype": self.reduction_dtype}
        out = self.mlp(x, **option)
        if self.sp:
            parts = [torch.empty_like(out) for _ in range(2)]
            dist.all_gather(parts, out)
            out = torch.cat(parts)
        return out[:rows].reshape(shape)


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@torch.inference_mode()
def generate(model, input_ids, diagnostic=False):
    past = None
    current = input_ids
    tokens = []
    first_logits = None
    for step in range(32):
        output = model(input_ids=current, past_key_values=past, use_cache=True)
        logits = output.logits[:, -1]
        if step == 0 and diagnostic:
            first_logits = logits.clone()
        current = logits.argmax(dim=-1, keepdim=True)
        tokens.append(current)
        past = output.past_key_values
    return torch.cat(tokens, dim=1), first_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reduction", choices=["default", "bf16", "fp32"], default="default")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method="file://" + os.environ["NVTE_TEST_RDZV_PATH"], rank=rank, world_size=2
    )
    reduction_dtype = {"default": None, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.reduction
    ]
    source = Path(te.__file__).parents[2]
    result = {
        "rank": rank,
        "pid": os.getpid(),
        "source": str(source),
        "source_sha": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"])
        ).hexdigest(),
        "torch": torch.__version__,
        "transformers_model": str(args.model),
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "reduction": args.reduction,
        "output_tokens": 32,
        "rounds": args.rounds,
        "cases": [],
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt = tokenizer.encode(
        "Explain how tensor parallel inference combines partial matrix products. " * 100,
        add_special_tokens=False,
    )
    counts = Counter()
    original_ar, original_rs = dist.all_reduce, dist.reduce_scatter_tensor

    def all_reduce(tensor, *positional, **kwargs):
        counts["all_reduce:" + str(tensor.dtype)] += 1
        return original_ar(tensor, *positional, **kwargs)

    def reduce_scatter(output, tensor, *positional, **kwargs):
        counts["reduce_scatter:" + str(tensor.dtype)] += 1
        return original_rs(output, tensor, *positional, **kwargs)

    dist.all_reduce, dist.reduce_scatter_tensor = all_reduce, reduce_scatter
    for kind in ["Linear", "ops.Linear"]:
        for sp in [False, True]:
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    local_files_only=True,
                )
                .cuda()
                .eval()
            )
            assert len(model.model.layers) == 28
            for layer in model.model.layers:
                layer.self_attn.o_proj = RowProjection(
                    layer.self_attn.o_proj, kind, sp, reduction_dtype, rank
                )
                layer.mlp = NormalizedMLP(
                    layer.post_attention_layernorm, layer.mlp, sp, reduction_dtype, rank
                )
                layer.post_attention_layernorm = torch.nn.Identity()
            model.requires_grad_(False)
            addresses = [parameter.data_ptr() for parameter in model.parameters()]
            for length in [128, 512]:
                name = kind.replace(".", "-") + f"-sp{int(sp)}-n{length}"
                ids = torch.tensor([prompt[:length]], device="cuda")
                tokens, logits = generate(model, ids, diagnostic=True)
                reference_path = (
                    args.reference
                    if args.reference
                    else args.output.parent / (args.output.stem + "-reference")
                ) / (name + ".pt")
                exact = True
                if args.reference:
                    reference = torch.load(reference_path, weights_only=True, map_location="cuda")
                    torch.testing.assert_close(logits, reference["logits"], rtol=0, atol=0)
                    torch.testing.assert_close(tokens, reference["tokens"], rtol=0, atol=0)
                elif rank == 0:
                    reference_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"logits": logits.cpu(), "tokens": tokens.cpu()}, reference_path)
                peers = [torch.empty_like(tokens) for _ in range(2)]
                dist.all_gather(peers, tokens)
                torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
                for _ in range(2):
                    generate(model, ids)
                samples = []
                before = counts.copy()
                for _ in range(args.rounds):
                    dist.barrier()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    measured_tokens, _ = generate(model, ids)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - started) * 1000)
                    torch.testing.assert_close(measured_tokens, tokens, rtol=0, atol=0)
                measured_calls = dict(counts - before)
                expected_dtype = torch.bfloat16 if reduction_dtype is None else reduction_dtype
                expected_op = "reduce_scatter:" if sp else "all_reduce:"
                assert measured_calls == {
                    expected_op + str(expected_dtype): 28 * 2 * 32 * args.rounds
                }
                assert addresses == [parameter.data_ptr() for parameter in model.parameters()]
                row = {
                    "name": name,
                    "kind": kind,
                    "sequence_parallel": sp,
                    "input_tokens": length,
                    "input_sha256": tensor_hash(ids),
                    "logits_sha256": tensor_hash(logits),
                    "tokens_sha256": tensor_hash(tokens),
                    "tokens": tokens[0].tolist(),
                    "reference_exact": exact,
                    "calls": measured_calls,
                    "samples_ms": samples,
                    "median_ms": statistics.median(samples),
                    "weight_addresses_stable": True,
                }
                result["cases"].append(row)
                args.output.with_name(args.output.stem + f"-rank{rank}.json").write_text(
                    json.dumps(result, indent=2)
                )
                print(json.dumps(row), flush=True)
            del model
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
