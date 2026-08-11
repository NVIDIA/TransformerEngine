# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pathlib
import tempfile

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.tensor.utils import quantize_master_weights

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)


def _single_rank_group():
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(0)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            rendezvous_file = pathlib.Path(f.name)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=rendezvous_file.resolve().as_uri(),
            rank=0,
            world_size=1,
        )
    return torch.distributed.GroupMember.WORLD


def _make_weight(dtype):
    quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    weight = quantizer.make_empty((128, 128), dtype=dtype, device="cuda")
    quantizer.update_quantized(torch.randn(128, 128, dtype=dtype, device="cuda"), weight)
    return weight


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_empty_master_shard_agrees_with_populated_rank(monkeypatch, dtype):
    """A rank owning no shard must reduce the same amax dtype as one that owns data.

    Wide FSDP sharding pads the parameter bucket, so the tail ranks can end up with an
    empty shard of every weight. Those ranks still join the amax all-reduce.
    """
    group = _single_rank_group()

    amax_dtypes = []
    real_all_reduce = torch.distributed.all_reduce

    def spy(tensor, *args, **kwargs):
        amax_dtypes.append(tensor.dtype)
        return real_all_reduce(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.distributed, "all_reduce", spy)

    populated = _make_weight(dtype)
    master = torch.randn(populated.numel(), dtype=torch.float32, device="cuda")
    quantize_master_weights([populated], [master], [0], group=group)

    # Used to raise UnboundLocalError instead of reaching the all-reduce.
    quantize_master_weights([_make_weight(dtype)], [None], [None], group=group)

    assert len(amax_dtypes) == 2
    assert amax_dtypes[0] == amax_dtypes[1]
