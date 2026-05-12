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
        torch.cuda.empty_cache() + gc.collect() per case
    all ranks gather (ok, error_msg) to rank 0
    rank 0:
        write one JSON response line to stdout

Protocol (line-delimited JSON over rank-0 stdio):
    request : {"op": "run", "kwargs": {...}}
              {"op": "shutdown"}
    response: {"ok": true}
              {"ok": false, "error": "first failing rank's traceback"}
"""
import gc
import json
import os
import sys
import traceback

import torch
import torch.distributed as dist

# Make sibling modules importable when launched directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_attention_with_cp import run_dpa_with_cp
from transformer_engine.pytorch.quantization import FP8GlobalStateManager


# Env vars run_dpa_with_cp re-sets at the top of every call. We pop them
# defensively between cases so a future caller that *doesn't* re-set them
# can't inherit a leftover value from a previous case in the same worker.
_TRANSIENT_ENV_KEYS = (
    "NVTE_FP8_DPA_BWD",
    "NVTE_DPA_FP8CS_O_in_F16",
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
)


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


def _reset_between_cases() -> None:
    """Drop state that would otherwise cascade across cases."""
    FP8GlobalStateManager.reset()
    for env_key in _TRANSIENT_ENV_KEYS:
        os.environ.pop(env_key, None)
    torch.cuda.empty_cache()
    gc.collect()


def _run_one(req: dict, rank: int) -> tuple[bool, str]:
    op = req["op"]
    if op != "run":
        return False, f"unknown op: {op}"
    try:
        run_dpa_with_cp(**req.get("kwargs", {}))
        return True, ""
    except Exception:
        return False, f"[Rank {rank}] {traceback.format_exc()}"
    finally:
        _reset_between_cases()


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    os.environ["NVTE_CP_POOL_PG"] = "1"

    try:
        while True:
            req = _recv_request(rank)
            if req.get("op") == "shutdown":
                break

            ok, msg = _run_one(req, rank)

            gathered: list[tuple[bool, str]] = [None] * world_size  # type: ignore[list-item]
            dist.gather_object((ok, msg), gathered if rank == 0 else None, dst=0)

            # Surface a wedged communicator here, before the next case's
            # collectives can inherit the corruption.
            dist.barrier()

            if rank == 0:
                all_ok = all(o for o, _ in gathered)
                if all_ok:
                    _send_response(rank, {"ok": True})
                else:
                    first_err = next(m for o, m in gathered if not o)
                    _send_response(rank, {"ok": False, "error": first_err})
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
