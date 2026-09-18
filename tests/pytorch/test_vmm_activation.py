# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import gc

import pytest
import torch

vmm_activation = pytest.importorskip("transformer_engine.pytorch.vmm_activation")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VMM activation reload requires CUDA")
def test_repeated_async_reload_releases_context_off_host_callback():
    """Repeated reload cleanup must not deadlock the copy-stream host callback."""
    allocation = vmm_activation.CUDAActivationVMMAllocation(
        (4096,), (1,), torch.float32, torch.device("cuda:0")
    )
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    host = torch.empty(4096, dtype=torch.float32, pin_memory=True)

    try:
        for iteration in range(10):
            expected = torch.full((4096,), iteration + 1, dtype=torch.float32, device="cuda")
            allocation.tensor.copy_(expected)
            torch.cuda.synchronize()

            with torch.cuda.stream(copy_stream):
                host.copy_(allocation.tensor, non_blocking=True)
                vmm_activation.release_hooks_after([allocation], copy_stream)

            context = vmm_activation.remap_and_copy_after([allocation], [host], copy_stream)
            vmm_activation.wait_remap_copy_on_stream(context, compute_stream)
            with torch.cuda.stream(compute_stream):
                observed = allocation.tensor.clone()
            compute_stream.synchronize()
            allocation.adopt_async_remap()

            torch.testing.assert_close(observed, expected)
            status = dict(context.status())
            assert status["error"] == 0
            assert status["releases_ready"] == 1
            assert status["remaps_done"] == 1
            assert status["copies_submitted"] == 1

            del context
            gc.collect()
    finally:
        allocation.close()
