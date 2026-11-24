# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Row-parallel te.Linear as a benchmarkable distributed Case."""

import datetime
import math
import pathlib
import sys

import pytest
import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common.testing import Case, CaseSkip, benchmark

# Prepend so installed packages with a top-level utils module cannot shadow the test helpers.
sys.path = [str(pathlib.Path(__file__).resolve().parent.parent)] + sys.path
from utils import dtype_tols

NUM_GPUS = min(4, torch.cuda.device_count())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
@benchmark("hidden_size", [4096])
@pytest.mark.parametrize("hidden_size", [1024, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_row_parallel_linear(hidden_size, dtype):
    """Compare row-parallel te.Linear against the same matmul computed whole."""
    batch = 2048

    def dist_init(rank, world, coordinator_addr, coordinator_port):
        """Build the process group this Case's ranks share."""
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{coordinator_addr}:{coordinator_port}",
            rank=rank,
            world_size=world,
            # Shorter than the harness budget, so a wedged collective surfaces as a rank
            # error rather than as a killed process.
            timeout=datetime.timedelta(seconds=120),
            device_id=torch.device(f"cuda:{rank}"),
        )
        return {"pg": dist.group.WORLD, "rank": rank, "world": world}

    def dist_clean(state):
        """Release the process group."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def barrier(state):
        """Block the host until every rank arrives."""
        dist.barrier(group=state["pg"])

    def setup(state):
        """Allocate the sharded layer and its whole-tensor counterpart."""
        if hidden_size % state["world"] != 0:
            raise CaseSkip(f"hidden_size {hidden_size} is not divisible by {state['world']} ranks")
        rank, world = state["rank"], state["world"]
        shard = hidden_size // world
        torch.manual_seed(1234)
        x_full = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
        # Scaled so outputs are O(1) in bf16.
        w_full = torch.randn(hidden_size, hidden_size, device="cuda", dtype=dtype) / math.sqrt(
            hidden_size
        )

        layer = te.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            parallel_mode="row",
            tp_group=state["pg"],
            tp_size=world,
            params_dtype=dtype,
            device="cuda",
        )
        with torch.no_grad():
            layer.weight.copy_(w_full[:, rank * shard : (rank + 1) * shard])

        state.update(
            {
                "layer": layer,
                "x_shard": x_full[:, rank * shard : (rank + 1) * shard].contiguous(),
                "x_full": x_full,
                "w_full": w_full,
            }
        )
        return state

    def evaluate(state):
        """Row-parallel forward: local matmul on this rank's shard, then all-reduce."""
        with torch.no_grad():
            return state["layer"](state["x_shard"])

    def reference(state):
        """The same product computed whole on every rank."""
        with torch.no_grad():
            return state["x_full"] @ state["w_full"].transpose(0, 1)

    def verify(actual, expected):
        tols = dtype_tols(dtype)
        # Wider than the shared atol: the parallel path sums one partial product per rank
        # while the reference reduces all hidden_size terms at once.
        tols["atol"] = 2e-2
        torch.testing.assert_close(actual, expected, **tols)

    return Case(
        setup=setup,
        evaluate=evaluate,
        reference=reference,
        verify=verify,
        dist_init=dist_init,
        dist_clean=dist_clean,
        barrier=barrier,
        num_gpus=NUM_GPUS,
        timeout=300.0,
    )
