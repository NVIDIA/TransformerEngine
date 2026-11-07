# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Exercise CPU optimizer export, checkpoint restore, and continued updates.

Run on one GPU with ``python tests/pytorch/run_fused_adam_cpu_export.py``.
Use ``torchrun --standalone --nproc-per-node=2`` to also cover sharded DTensors.
"""

import io
import os
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DTensor, Shard, distribute_tensor

from transformer_engine.pytorch.optimizers import FusedAdam


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


def check_equal(actual, expected, name):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise RuntimeError(f"{name}: dtype or shape changed")
    if not torch.equal(local_tensor(actual).cpu(), local_tensor(expected).cpu()):
        raise RuntimeError(f"{name}: values changed")
    if isinstance(actual, DTensor) != isinstance(expected, DTensor):
        raise RuntimeError(f"{name}: distributed tensor type changed")
    if isinstance(actual, DTensor):
        if actual.device_mesh != expected.device_mesh or actual.placements != expected.placements:
            raise RuntimeError(f"{name}: distributed placement changed")
        if actual.stride() != expected.stride():
            raise RuntimeError(f"{name}: distributed stride changed")


def make_model(mesh=None):
    model = torch.nn.Module()
    for index, rows in enumerate((17, 8)):
        value = torch.arange(rows * 16, device="cuda", dtype=torch.float32).reshape(rows, 16)
        value = (value / 1024).to(torch.bfloat16)
        if mesh is not None:
            value = distribute_tensor(value, mesh, [Shard(0)])
        model.register_parameter(f"weight_{index}", torch.nn.Parameter(value))
    return model


def advance(model, optimizer, step):
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, (index + step + 1) / 32)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def compare_exports(actual, expected):
    if actual["param_groups"] != expected["param_groups"]:
        raise RuntimeError("parameter-group metadata changed")
    if actual["state"].keys() != expected["state"].keys():
        raise RuntimeError("state keys changed")
    for key, states in actual["state"].items():
        if states.keys() != expected["state"][key].keys():
            raise RuntimeError(f"state {key}: names changed")
        for name, value in states.items():
            if local_tensor(value).device.type != "cpu":
                raise RuntimeError(f"state {key}.{name}: export is not on CPU")
            check_equal(value, expected["state"][key][name], f"state {key}.{name}")


def round_trip(moment_dtype, master_dtype, remainders, mesh=None, capturable=False):
    model = make_model(mesh)
    reference = make_model(mesh)
    options = dict(
        master_weights=True,
        master_weight_dtype=master_dtype,
        exp_avg_dtype=moment_dtype,
        exp_avg_sq_dtype=moment_dtype,
        store_param_remainders=remainders,
        capturable=capturable,
    )
    optimizer = FusedAdam(model.parameters(), state_dict_cpu_offload=True, **options)
    reference_optimizer = FusedAdam(reference.parameters(), **options)
    if optimizer.state_dict()["state"]:
        raise RuntimeError("uninitialized optimizer exported state")
    for step in range(3):
        advance(model, optimizer, step)
        advance(reference, reference_optimizer, step)

    live_states = [
        (value, local_tensor(value).data_ptr(), local_tensor(value).clone())
        for states in optimizer.state.values()
        for value in states.values()
    ]
    saved = optimizer.state_dict()
    compare_exports(saved, reference_optimizer.state_dict())
    compare_exports(optimizer.state_dict(), saved)
    for value, pointer, snapshot in live_states:
        if local_tensor(value).device.type != "cuda" or local_tensor(value).data_ptr() != pointer:
            raise RuntimeError("live optimizer state moved or changed storage")
        check_equal(local_tensor(value), snapshot, "live optimizer state")

    restored = make_model(mesh)
    restored.load_state_dict(model.state_dict())
    restored_optimizer = FusedAdam(restored.parameters(), state_dict_cpu_offload=True, **options)
    checkpoint = io.BytesIO()
    torch.save(saved, checkpoint)
    checkpoint.seek(0)
    restored_optimizer.load_state_dict(torch.load(checkpoint, weights_only=False))
    compare_exports(restored_optimizer.state_dict(), saved)

    for step in range(3, 6):
        advance(model, optimizer, step)
        advance(reference, reference_optimizer, step)
        advance(restored, restored_optimizer, step)
        for actual, expected, loaded in zip(
            model.parameters(), reference.parameters(), restored.parameters()
        ):
            check_equal(actual, expected, "continued parameter")
            check_equal(loaded, expected, "restored parameter")
    compare_exports(optimizer.state_dict(), reference_optimizer.state_dict())
    compare_exports(restored_optimizer.state_dict(), reference_optimizer.state_dict())
    print(
        f"PASS dtype={moment_dtype} master={master_dtype} remainders={remainders}"
        f" mesh={mesh} capturable={capturable}"
    )


def dcp_round_trip(mesh, asynchronous, flatten):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    model = make_model(mesh)
    optimizer = FusedAdam(
        model.parameters(),
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
        state_dict_cpu_offload=True,
    )
    advance(model, optimizer, 0)
    options = StateDictOptions(cpu_offload=True, flatten_optimizer_state_dict=flatten)
    state = get_optimizer_state_dict(model, optimizer, options=options)
    with tempfile.TemporaryDirectory() as directory:
        # All ranks use rank zero's shared checkpoint directory.
        locations = [directory]
        dist.broadcast_object_list(locations, src=0)
        directory = locations[0]
        if asynchronous:
            dcp.async_save({"optimizer": state}, checkpoint_id=directory).result()
        else:
            dcp.save({"optimizer": state}, checkpoint_id=directory)
        restored = make_model(mesh)
        restored.load_state_dict(model.state_dict())
        restored_optimizer = FusedAdam(
            restored.parameters(),
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16,
            state_dict_cpu_offload=True,
        )
        advance(restored, restored_optimizer, 0)
        restored.load_state_dict(model.state_dict())
        loaded = {
            "optimizer": get_optimizer_state_dict(restored, restored_optimizer, options=options)
        }
        dcp.load(loaded, checkpoint_id=directory)
        set_optimizer_state_dict(restored, restored_optimizer, loaded["optimizer"], options=options)
        advance(model, optimizer, 1)
        advance(restored, restored_optimizer, 1)
        for actual, expected in zip(restored.parameters(), model.parameters()):
            check_equal(actual, expected, "DCP continued parameter")
        compare_exports(restored_optimizer.state_dict(), optimizer.state_dict())
        dist.barrier()
    print(f"PASS sharded DCP save/load and continued update async={asynchronous} flatten={flatten}")


def check_export_memory():
    elements = 2**18
    for count in (1, 8):
        parameters = [
            torch.nn.Parameter(torch.zeros(elements, dtype=torch.bfloat16, device="cuda"))
            for _ in range(count)
        ]
        optimizer = FusedAdam(
            parameters,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16,
            state_dict_cpu_offload=True,
        )
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        exported = optimizer.state_dict()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        expected = elements * torch.tensor([], dtype=torch.float32).element_size()
        if peak != expected or torch.cuda.memory_allocated() != before:
            raise RuntimeError(f"export retained GPU copies: count={count}, peak={peak}")
        if len(exported["state"]) != count:
            raise RuntimeError("export omitted optimizer states")
        print(f"PASS export memory parameters={count} peak_bytes={peak}")


def main():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.uint8):
        for master_dtype, remainders in (
            (torch.float32, False),
            (torch.float32, True),
            (torch.float16, False),
        ):
            round_trip(dtype, master_dtype, remainders)
    for dtype in (torch.float32, torch.bfloat16):
        round_trip(dtype, torch.float32, False, capturable=True)
    check_export_memory()
    if "RANK" in os.environ:
        dist.init_process_group(
            "cpu:gloo,cuda:nccl", device_id=torch.device("cuda", torch.cuda.current_device())
        )
        try:
            mesh = init_device_mesh("cuda", (dist.get_world_size(),))
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                round_trip(dtype, torch.float32, False, mesh)
            for flatten in (False, True):
                for asynchronous in (False, True):
                    dcp_round_trip(mesh, asynchronous, flatten)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
