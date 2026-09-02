#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Delayed-scaling backward update when some ranks skip backward.

Every rank runs forward for every module so the amax buffers match across
ranks, but only some ranks run backward. The update must still happen once
per step on every rank or the amax all-reduce hangs.
"""

import argparse
import datetime
import os

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

BATCH, HIDDEN, STEPS = 32, 128, 3


def _make_model(seed):
    torch.manual_seed(seed)
    return torch.nn.ModuleList([te.Linear(HIDDEN, HIDDEN, bias=True) for _ in range(2)]).cuda()


def _step_skipped_module_backward(model, recipe, rank):
    """Odd ranks feed the first module an empty batch and drop its output."""
    rows = BATCH if rank % 2 == 0 else 0
    x_a = torch.randn(rows, HIDDEN, device="cuda", requires_grad=True)
    x_b = torch.randn(BATCH, HIDDEN, device="cuda", requires_grad=True)
    with te.autocast(enabled=True, recipe=recipe):
        y_a = model[0](x_a)
        y_b = model[1](x_b)
    loss = y_b.float().sum()
    if y_a.numel() > 0:
        loss = loss + y_a.float().sum()
    loss.backward()


def _step_no_backward_in_scope(model, recipe, rank):
    """Odd ranks run no backward at all; the scope still triggers the update."""
    x = torch.randn(BATCH, HIDDEN, device="cuda", requires_grad=True)
    with te.quantization_backward_scope():
        with te.autocast(enabled=True, recipe=recipe):
            y = model[1](model[0](x))
        if rank % 2 == 0:
            y.float().sum().backward()


_CASES = {
    "skipped_module_backward": _step_skipped_module_backward,
    "no_backward_in_scope": _step_no_backward_in_scope,
}


def _bwd_state(model):
    tensors = []
    for module in model:
        state = module.fp8_meta["scaling_bwd"]
        tensors += [state.amax_history.clone(), state.scale.clone()]
    return tensors


def _assert_same_on_all_ranks(tensors, world_size):
    for t in tensors:
        gathered = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(gathered, t)
        for other in gathered[1:]:
            assert torch.equal(other, gathered[0]), f"{gathered[0]} vs {other}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=_CASES.keys(), required=True)
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args()

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    dist.init_process_group(backend=args.backend, timeout=datetime.timedelta(seconds=120))
    rank, world_size = dist.get_rank(), dist.get_world_size()

    model = _make_model(seed=1234)
    recipe = DelayedScaling(reduce_amax=True)
    step = _CASES[args.case]
    qstate = FP8GlobalStateManager.quantization_state

    for _ in range(STEPS):
        model.zero_grad(set_to_none=True)
        step(model, recipe, rank)
        assert not qstate.pending_backward_quantization_update
        assert qstate.backward_quantization_update_callback_task_id is None
        _assert_same_on_all_ranks(_bwd_state(model), world_size)

    # Ranks that skipped backward must have received the other ranks' amaxes.
    for module in model:
        assert module.fp8_meta["scaling_bwd"].amax_history.abs().sum() > 0

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
