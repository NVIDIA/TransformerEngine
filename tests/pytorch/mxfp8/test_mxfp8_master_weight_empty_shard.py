# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.tensor.utils import quantize_master_weights

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)


@pytest.fixture
def single_rank_group():
    # Only tear down a group this fixture owns; another test may have set one up.
    created = not torch.distributed.is_initialized()
    if created:
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            backend="nccl", store=torch.distributed.HashStore(), rank=0, world_size=1
        )
    try:
        yield torch.distributed.GroupMember.WORLD
    finally:
        if created:
            torch.distributed.destroy_process_group()


# multi_tensor_compute_scale_inv_e8m0 requires a bf16 amax, and the amax buffer takes the
# model weight dtype, so bf16 is the only model dtype this path supports.
MODEL_DTYPE = torch.bfloat16


def _make_weight():
    quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    weight = quantizer.make_empty((128, 128), dtype=MODEL_DTYPE, device="cuda")
    quantizer.update_quantized(torch.randn(128, 128, dtype=MODEL_DTYPE, device="cuda"), weight)
    return weight


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_empty_master_shard_agrees_with_populated_rank(monkeypatch, single_rank_group):
    """A rank owning no shard must reduce the same amax dtype as one that owns data.

    Wide FSDP sharding pads the parameter bucket, so the tail ranks can end up with an
    empty shard of every weight. Those ranks still join the amax all-reduce.

    Single-GPU counterpart of the 2-rank case in
    tests/pytorch/distributed/test_cast_master_weights_to_fp8.py, which checks the same
    agreement over a real collective.
    """
    amax_dtypes = []
    real_all_reduce = torch.distributed.all_reduce

    def spy(tensor, *args, **kwargs):
        amax_dtypes.append(tensor.dtype)
        return real_all_reduce(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.distributed, "all_reduce", spy)

    populated = _make_weight()
    master = torch.randn(populated.numel(), dtype=torch.float32, device="cuda")
    quantize_master_weights([populated], [master], [0], group=single_rank_group)

    # Used to raise UnboundLocalError instead of reaching the all-reduce.
    quantize_master_weights([_make_weight()], [None], [None], group=single_rank_group)

    assert len(amax_dtypes) == 2
    assert amax_dtypes[0] == amax_dtypes[1]
