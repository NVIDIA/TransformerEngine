# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Checkpoint loading for FusedAdam."""

import copy

import pytest
import torch

from transformer_engine.pytorch.optimizers import FusedAdam


@pytest.mark.parametrize("param_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cpu_checkpoint", [False, True])
def test_load_preserves_state_and_next_update(param_dtype, state_dtype, cpu_checkpoint):
    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(torch.randn(128, device="cuda", dtype=param_dtype))
    options = dict(
        master_weights=True,
        exp_avg_dtype=state_dtype,
        exp_avg_sq_dtype=state_dtype,
        store_param_remainders=param_dtype == torch.bfloat16,
    )
    original = FusedAdam([parameter], lr=0.002, **options)
    parameter.grad = torch.ones_like(parameter)
    original.step()
    saved = copy.deepcopy(original.state_dict())
    if cpu_checkpoint:
        for state in saved["state"].values():
            for name, value in state.items():
                state[name] = value.cpu()

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = FusedAdam([restored_parameter], lr=0.1, **options)
    restored.load_state_dict(saved)
    assert restored.param_groups[0]["lr"] == original.param_groups[0]["lr"]
    assert restored.param_groups[0]["step"] == original.param_groups[0]["step"]
    for name, value in original.state[parameter].items():
        torch.testing.assert_close(restored.state[restored_parameter][name], value, rtol=0, atol=0)
        assert (
            restored.state[restored_parameter][name].data_ptr()
            != saved["state"][0][name].data_ptr()
        )

    parameter.grad = torch.full_like(parameter, 0.5)
    restored_parameter.grad = parameter.grad.clone()
    original.step()
    restored.step()
    torch.testing.assert_close(restored_parameter, parameter, rtol=0, atol=0)
    for name, value in original.state[parameter].items():
        torch.testing.assert_close(restored.state[restored_parameter][name], value, rtol=0, atol=0)


def test_load_avoids_temporary_parameter_dtype_states():
    parameter = torch.nn.Parameter(torch.ones(2**22, device="cuda", dtype=torch.bfloat16))
    optimizer = FusedAdam([parameter], master_weights=True, store_param_remainders=True)
    optimizer.initialize_state(parameter, store_param_remainders=True)
    saved = optimizer.state_dict()
    restored = FusedAdam([parameter], master_weights=True, store_param_remainders=True)
    restored.initialize_state(parameter, store_param_remainders=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    restored.load_state_dict(saved)
    torch.cuda.synchronize()

    # Replacing initialized states should not first allocate parameter-dtype copies.
    assert torch.cuda.max_memory_allocated() - before <= 1024**2


@pytest.mark.parametrize("hook_kind", ["pre", "post"])
def test_load_preserves_hook_observations(hook_kind):
    parameter = torch.nn.Parameter(torch.ones(16, device="cuda", dtype=torch.bfloat16))
    optimizer = FusedAdam([parameter], master_weights=True, store_param_remainders=True)
    optimizer.initialize_state(parameter, store_param_remainders=True)
    saved = optimizer.state_dict()
    restored = FusedAdam([parameter], master_weights=True, store_param_remainders=True)
    observed = []
    if hook_kind == "pre":
        handle = restored.register_load_state_dict_pre_hook(
            lambda _, state: observed.append(state["state"][0]["exp_avg"].dtype)
        )
    else:
        handle = restored.register_load_state_dict_post_hook(
            lambda opt: observed.append(opt.state[parameter]["exp_avg"].dtype)
        )
    try:
        restored.load_state_dict(saved)
    finally:
        handle.remove()
    assert observed == [torch.float32 if hook_kind == "pre" else torch.bfloat16]
    assert restored.state[parameter]["exp_avg"].dtype == torch.float32


def test_load_preserves_non_parameter_state_and_validates_groups():
    parameter = torch.nn.Parameter(torch.ones(16, device="cuda"))
    optimizer = FusedAdam([parameter])
    saved = optimizer.state_dict()
    saved["state"]["global_state"] = {"count": torch.tensor(7)}
    optimizer.load_state_dict(saved)
    assert optimizer.state["global_state"]["count"].item() == 7

    invalid = copy.deepcopy(saved)
    invalid["param_groups"] = []
    with pytest.raises(ValueError, match="different number of parameter groups"):
        optimizer.load_state_dict(invalid)


@pytest.mark.parametrize("param_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_load_matches_hook_path_for_state_dtypes(param_dtype, state_dtype):
    torch.manual_seed(1234)
    parameter = torch.nn.Parameter(torch.randn(128, device="cuda", dtype=param_dtype))
    options = dict(master_weights=True, exp_avg_dtype=state_dtype, exp_avg_sq_dtype=state_dtype)
    original = FusedAdam([parameter], **options)
    parameter.grad = torch.full_like(parameter, 0.5)
    original.step()
    saved = original.state_dict()
    parameters = [torch.nn.Parameter(parameter.detach().clone()) for _ in range(2)]
    optimizers = [FusedAdam([value], **options) for value in parameters]
    # A no-op hook keeps the original load path as the reference.
    handle = optimizers[0].register_load_state_dict_pre_hook(lambda _, state: None)
    try:
        for optimizer in optimizers:
            optimizer.load_state_dict(copy.deepcopy(saved))
    finally:
        handle.remove()

    for name in original.state[parameter]:
        torch.testing.assert_close(
            optimizers[0].get_unscaled_state(parameters[0], name),
            optimizers[1].get_unscaled_state(parameters[1], name),
            rtol=0,
            atol=0,
        )
    for optimizer, value in zip(optimizers, parameters):
        value.grad = torch.ones_like(value)
        optimizer.step()
    torch.testing.assert_close(parameters[0], parameters[1], rtol=0, atol=0)


def test_load_dtensor_state(tmp_path):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    owns_process_group = not torch.distributed.is_initialized()
    if owns_process_group:
        torch.distributed.init_process_group(
            "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
        )
    try:
        mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),))
        parameter = torch.nn.Parameter(
            distribute_tensor(torch.ones(32, device="cuda"), mesh, [Shard(0)])
        )
        original = FusedAdam([parameter], master_weights=True)
        original.initialize_state(parameter, store_param_remainders=False)
        saved = original.state_dict()
        restored = FusedAdam([parameter], master_weights=True)
        restored.load_state_dict(saved)
        for name, value in original.state[parameter].items():
            actual = restored.state[parameter][name]
            assert actual.placements == value.placements
            assert actual.device_mesh == value.device_mesh
            torch.testing.assert_close(actual.to_local(), value.to_local(), rtol=0, atol=0)
            assert actual.to_local().data_ptr() != value.to_local().data_ptr()
    finally:
        if owns_process_group:
            torch.distributed.destroy_process_group()


def test_load_preserves_subclass_setstate_observations():
    observed = []

    class ObservingAdam(FusedAdam):
        def __setstate__(self, state):
            super().__setstate__(state)
            observed.append(next(iter(self.state.values()))["exp_avg"].dtype)

    parameter = torch.nn.Parameter(torch.ones(16, device="cuda", dtype=torch.bfloat16))
    original = FusedAdam([parameter])
    original.initialize_state(parameter, store_param_remainders=False)
    restored = ObservingAdam([parameter])

    restored.load_state_dict(original.state_dict())

    assert observed == [torch.bfloat16]
    assert restored.state[parameter]["exp_avg"].dtype == torch.float32
