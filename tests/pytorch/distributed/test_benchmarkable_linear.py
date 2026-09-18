# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Row-parallel te.Linear as a benchmarkable distributed Case.

The harness launches the ranks: this file only says how many it wants and what each one
does. ``dist_init`` builds the process group, ``setup`` allocates against it, and
``evaluate`` runs the forward whose all-reduce is the collective being measured.
"""

import datetime
import math
import os

import pytest
import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common.testing import Case, CaseSkip, benchmark
from transformer_engine.common.testing.distributed import RANK_ENV, RENDEZVOUS_ENV, WORLD_SIZE_ENV

# Defined here rather than imported: tests/pytorch is not on sys.path from this
# subdirectory, which is why the other distributed tests carry their own tolerances.
# Loose because the two paths reduce differently: the parallel one sums four partial
# products across ranks, the reference reduces all hidden_size terms at once.
_TOLS = {torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2}}

NUM_GPUS = min(4, torch.cuda.device_count())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 GPUs")
@benchmark("hidden_size", [4096])
@pytest.mark.parametrize("hidden_size", [1024, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_row_parallel_linear(hidden_size, dtype):
    """Compare row-parallel te.Linear against the same matmul computed whole."""
    batch = 2048

    def dist_init():
        """Build the process group this Case's ranks share."""
        rank = int(os.environ[RANK_ENV])
        world = int(os.environ[WORLD_SIZE_ENV])
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{os.environ[RENDEZVOUS_ENV]}",
            rank=rank,
            world_size=world,
            # The harness enforces its own budget too; this one lets a wedged collective
            # surface as an error in the rank rather than as a killed process.
            timeout=datetime.timedelta(seconds=120),
            device_id=torch.device(f"cuda:{rank}"),
        )
        return {"pg": dist.group.WORLD, "rank": rank, "world": world}

    def dist_clean(state):
        """Release the process group. The harness calls this once, on success or failure.

        It does not run when the harness has to stop a rank outright; process death
        releases the communicator in that case.
        """
        if dist.is_initialized():
            dist.destroy_process_group()

    def barrier(state):
        """Align ranks before a timed sample, outside the measured interval."""
        dist.barrier(group=state["pg"])

    def setup(state):
        """Allocate the sharded layer and its whole-tensor counterpart."""
        if hidden_size % state["world"] != 0:
            raise CaseSkip(f"hidden_size {hidden_size} is not divisible by {state['world']} ranks")
        rank, world = state["rank"], state["world"]
        shard = hidden_size // world
        torch.manual_seed(1234)
        x_full = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
        # Scaled so outputs are O(1); unscaled randn would make a bf16 reduction over
        # hidden_size terms produce values whose absolute error dwarfs any sane atol.
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
        torch.testing.assert_close(actual, expected, **_TOLS[dtype])

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
