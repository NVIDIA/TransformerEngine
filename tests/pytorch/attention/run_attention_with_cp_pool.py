# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Persistent worker for batched CP attention tests.

Launched ONCE per (pytest session, world_size) by torchrun. All ranks init
NCCL, then enter a dispatch loop:

    rank 0:
        read one JSON request line from stdin
        broadcast it to all ranks
    all ranks:
        call run_dpa_with_cp(**kwargs) — the same work function the
        per-case subprocess design uses, with NVTE_CP_POOL_PG=1 so the
        function reuses our PG instead of re-initing it
        torch.cuda.empty_cache() per case
    all ranks gather (ok, error_msg) to rank 0
    rank 0:
        write one JSON response line to stdout

Protocol (line-delimited JSON over rank-0 stdio):
    request : {"op": "run", "kwargs": {...}}
              {"op": "shutdown"}
    response: {"ok": true}
              {"ok": false, "error": "first failing rank's traceback"}
"""
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

# Make sibling modules importable when launched directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_attention_with_cp import run_dpa_with_cp
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


def _recv_request(rank: int) -> dict:
    box = [None]
    if rank == 0:
        line = sys.stdin.readline()
        box[0] = {"op": "shutdown"} if not line else json.loads(line)
    dist.broadcast_object_list(box, src=0)
    return box[0]


def _send_response(rank: int, payload: dict) -> None:
    if rank == 0:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def _silence_non_rank0_stdout(rank: int) -> None:
    """Redirect non-rank-0 stdout to /dev/null at fd level.

    All ranks share rank 0's stdout fd (torchrun inherits it from the launcher),
    so Python/library writes on rank>0 would interleave with rank 0's JSON
    protocol on the parent's pipe. Closing fd 1 at the OS level on rank>0
    catches both Python (``print``) and C-level (NCCL, etc.) writes.
    """
    if rank == 0:
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.stdout = open(1, "w", closefd=False)


def _reset_between_cases() -> None:
    """Drop state that would otherwise cascade across cases.

    Matches the per-case startup of the single-shot worker
    (``_run_single_config`` on the per-case-subprocess branch): identical RNG
    seed at the start of every case, FP8 state cleared, allocator clean.
    ``run_dpa_with_cp`` re-sets ``NVTE_FUSED_ATTN``/``NVTE_FLASH_ATTN``
    unconditionally and pops the other transient env vars itself, so no
    explicit pop is needed here.
    """
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    FP8GlobalStateManager.reset()
    torch.cuda.empty_cache()
    # Invalidate DPA's module-level backend cache so the per-case
    # NVTE_FLASH_ATTN/NVTE_FUSED_ATTN env-var toggle actually takes effect
    # instead of reusing the previous case's resolved backend.
    try:
        from transformer_engine.pytorch.attention.dot_product_attention import dot_product_attention

        dot_product_attention._attention_backends["backend_selection_requires_update"] = True
    except (ImportError, AttributeError, KeyError):
        pass


_case_counter = 0


def _run_one(req: dict, rank: int) -> tuple[bool, str]:
    global _case_counter
    op = req["op"]
    if op != "run":
        return False, f"unknown op: {op}"
    # Reset BEFORE the case so the first case also starts from a known RNG seed
    # and clean FP8 state — same as the single-shot worker's per-process startup.
    _reset_between_cases()
    t0 = time.monotonic()
    ok = True
    err = ""
    try:
        run_dpa_with_cp(**req.get("kwargs", {}))
    except Exception:
        ok = False
        err = f"[Rank {rank}] {traceback.format_exc()}"
    wall = time.monotonic() - t0
    # Per-case wall time on rank 0, opt-in via NVTE_CP_POOL_TIMING=1.
    # Used to tune POOL_SUBMIT_TIMEOUT_SEC against the observed distribution.
    if rank == 0 and int(os.environ.get("NVTE_CP_POOL_TIMING", "0")):
        _case_counter += 1
        sys.stderr.write(
            f"[POOL-TIMING] case_idx={_case_counter} "
            f"world_size={int(os.environ.get('WORLD_SIZE', 0))} "
            f"wall_s={wall:.3f} ok={ok}\n"
        )
        sys.stderr.flush()
    return ok, err


def _create_cp_comm_groups(rank: int, world_size: int) -> tuple:
    """Pre-create the CP collective groups for this pool.

    world_size and the rank set are constant for the lifetime of one pool, so
    the world group and the a2a+p2p sub-groups are deterministic. Creating
    them once here and reusing them across every case eliminates ~50-100 ms
    of NCCL setup per case (cyanguwa's review feedback on PR #2993).

    Returns ``(world_group, a2a_p2p_sub_groups)``. ``a2a_p2p_sub_groups`` is
    empty when world_size is too small to support a2a+p2p (needs an even
    world_size ≥ 4); cases with cp_comm_type='a2a+p2p' wouldn't be routed to
    such a pool anyway.
    """
    world_group = dist.new_group(range(world_size), backend="nccl")
    sub_groups: list = []
    if world_size >= 4 and world_size % 2 == 0:
        # Mirror the layout in run_attention_with_cp.py: cp_size/2 pairs along
        # axis 0, plus 2 stride-2 groups along axis 1.
        cp_comm_sub_ranks = [range(i * 2, (i + 1) * 2) for i in range(world_size // 2)]
        cp_comm_sub_ranks += [range(i, world_size, 2) for i in range(2)]
        for sub_ranks in cp_comm_sub_ranks:
            sub_group = dist.new_group(sub_ranks, backend="nccl")
            if rank in sub_ranks:
                sub_groups.append(sub_group)
    return world_group, sub_groups


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _silence_non_rank0_stdout(rank)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    os.environ["NVTE_CP_POOL_PG"] = "1"

    # Stash pool-shared CP groups on the run_attention_with_cp module so
    # run_dpa_with_cp can read them per case. Imported here (after the env var
    # is set) to keep import-time side effects minimal.
    import run_attention_with_cp as _rac

    _rac._pool_cp_comm_group, _rac._pool_cp_comm_sub_groups = _create_cp_comm_groups(
        rank, world_size
    )

    try:
        while True:
            req = _recv_request(rank)
            if req.get("op") == "shutdown":
                break

            ok, msg = _run_one(req, rank)

            gathered: list[tuple[bool, str]] = [None] * world_size  # type: ignore[list-item]
            # gather_object is itself a collective synchronization point — if
            # every rank reached it, none is ahead. No extra barrier needed.
            dist.gather_object((ok, msg), gathered if rank == 0 else None, dst=0)

            if rank == 0:
                all_ok = all(o for o, _ in gathered)
                if all_ok:
                    _send_response(rank, {"ok": True})
                else:
                    first_err = next(m for o, m in gathered if not o)
                    _send_response(rank, {"ok": False, "error": first_err})
            # Release the allocator cache so this pool doesn't squat on
            # GPUs that an overlapping different-world-size pool needs.
            torch.cuda.empty_cache()
    finally:
        # Tear down pool-shared CP groups before the main PG (NCCL requires
        # sub-groups to be destroyed first). Each destroy is independently
        # guarded so a wedged communicator on one group doesn't leak the rest.
        if _rac._pool_cp_comm_group is not None:
            try:
                dist.destroy_process_group(_rac._pool_cp_comm_group)
            except Exception:
                pass
        for g in _rac._pool_cp_comm_sub_groups:
            try:
                dist.destroy_process_group(g)
            except Exception:
                pass
        _rac._pool_cp_comm_group = None
        _rac._pool_cp_comm_sub_groups = []
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
