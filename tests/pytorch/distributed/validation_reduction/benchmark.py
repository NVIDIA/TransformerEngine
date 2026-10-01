# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-rank graph measurements of complete forward and backward passes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as ops


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reduction", choices=["default", "bf16", "fp32"], default="default")
args = parser.parse_args()
rank = int(os.environ["RANK"])
torch.cuda.set_device(rank)
dist.init_process_group(
    "nccl", init_method="file://" + os.environ["NVTE_TEST_RDZV_PATH"], rank=rank, world_size=2
)
reduction_dtype = {"default": None, "bf16": torch.bfloat16, "fp32": torch.float32}[args.reduction]
forward_kwargs = {} if reduction_dtype is None else {"reduction_dtype": reduction_dtype}
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
    "reduction": args.reduction,
    "rounds": 15,
    "replays": 20,
    "calls_per_graph": 3,
    "cases": [],
}
for kind in ["Linear", "LayerNormLinear", "LayerNormMLP", "ops.Linear"]:
    for sp in [False, True]:
        for rows in [2, 512]:
            torch.manual_seed(42 + rank)
            kwargs = dict(
                bias=False,
                tp_group=dist.group.WORLD,
                tp_size=2,
                sequence_parallel=sp,
                params_dtype=torch.bfloat16,
            )
            if kind == "ops.Linear":
                layer = ops.Linear(
                    1024,
                    1024,
                    bias=False,
                    dtype=torch.bfloat16,
                    tensor_parallel_mode="row",
                    tensor_parallel_group=dist.group.WORLD,
                    sequence_parallel=sp,
                    **forward_kwargs,
                )
            elif kind == "Linear":
                layer = te.Linear(1024, 1024, parallel_mode="row", **kwargs)
            elif kind == "LayerNormLinear":
                layer = te.LayerNormLinear(1024, 1024, parallel_mode="column", **kwargs)
            else:
                layer = te.LayerNormMLP(1024, 4096, set_parallel_mode=True, **kwargs)
            width = 1024 if kind in ("LayerNormLinear", "LayerNormMLP") else 512
            local_rows = rows // 2 if sp and width == 1024 else rows
            inp = torch.randn(
                local_rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            parameters = (inp, *layer.parameters())

            def forward():
                return layer(inp) if kind == "ops.Linear" else layer(inp, **forward_kwargs)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                out = forward()
                dout = torch.ones_like(out)

            def step():
                output = forward()
                return torch.autograd.grad(output, parameters, dout)

            with torch.cuda.stream(stream):
                for _ in range(5):
                    step()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(3):
                    gradients = step()
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            assert all(torch.isfinite(grad).all() for grad in gradients)
            samples = []
            for _ in range(15):
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / 60)
            row = {
                "kind": kind,
                "sequence_parallel": sp,
                "rows": rows,
                "input_shape": list(inp.shape),
                "output_shape": list(out.shape),
                "input_dtype": str(inp.dtype),
                "output_dtype": str(out.dtype),
                "median_us": statistics.median(samples),
                "samples_us": samples,
            }
            result["cases"].append(row)
            args.output.with_name(args.output.stem + f"-rank{rank}.json").write_text(
                json.dumps(result, indent=2)
            )
            print(json.dumps(row), flush=True)
            graph.reset()
dist.destroy_process_group()
