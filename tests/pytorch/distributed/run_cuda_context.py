# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Run with torchrun: live TE cleanup must retire Userbuffers on every rank."""

from datetime import timedelta
import os

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.module.base import UserBufferQuantizationMode, get_ub


def main():
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=90), device_id=torch.device("cuda", device)
    )
    world_size = dist.get_world_size()
    try:
        for _ in range(2):
            te.initialize_ub(
                (32, 128),
                tp_size=world_size,
                quantization_modes=[UserBufferQuantizationMode.NONE],
                bootstrap_backend="gloo",
            )
            owner = get_ub("qkv_fprop", False)
            generation = tex.get_cuda_state_version()
            try:
                tex.reset_cuda_state()
            except RuntimeError as error:
                assert "communication" in str(error)
            else:
                raise AssertionError("Active communication must reject post-loss reset")
            assert tex.get_cuda_state_version() == generation
            # A retained public owner must fail explicitly even after the global
            # manager has been cleared. Destruction must happen while NCCL is live.
            dist.barrier()
            try:
                te.release_cuda_state()
            except RuntimeError as error:
                assert "Release all owners" in str(error)
            else:
                raise AssertionError("A retained communication owner must reject release")
            del owner
            dist.barrier()
            te.release_cuda_state()
            value = torch.tensor(float(dist.get_rank() + 1), device="cuda")
            dist.all_reduce(value)
            torch.testing.assert_close(value, value.new_tensor(world_size * (world_size + 1) / 2))
        if dist.get_rank() == 0:
            print("CUDA STATE CLEANUP PASSED")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
