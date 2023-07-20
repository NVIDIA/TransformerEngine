# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Methods needed for distributed training."""

from contextlib import contextmanager
from typing import Optional, Union, Tuple

import paddle

import paddle.distributed.fleet.base.topology as tp
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.layers.mpu import mp_ops

from .constants import dist_group_type

_weight_split_axis = {
    'transformer_engine': {
        'row': 1,
        'column': 0
    },
    'paddle': {
        'row': 0,
        'column': 1
    }
}


def get_tp_group_and_world_size(
        tp_group: Union[dist_group_type, None]) -> Tuple[Union[dist_group_type, None], int]:
    """Get TP group and world size using Fleet API"""
    if not paddle.distributed.is_initialized():
        return None, 1
    model_parallel_group = (tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
                            if tp_group is None else tp_group)
    world_size = (tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
                  if tp_group is None else tp_group.nranks)
    return model_parallel_group, world_size


@contextmanager
def track_rng_state(enable: bool) -> None:
    """
    Applies get_rng_state_tracker().rng_state() to the context.
    If not enabled, it does nothing.
    """
    if enable:
        with get_rng_state_tracker().rng_state():
            yield
    else:
        yield


def set_tensor_dist_attr(tensor: paddle.Tensor, is_parallel: bool, axis: int) -> None:
    """Set distributed attributes for the input tensor"""
    tensor.is_distributed = is_parallel
    if is_parallel:
        tensor.split_axis = axis


def set_weight_tensor_dist_attr(tensor: paddle.Tensor, is_parallel: bool, parallel_mode: str,
                                backend: str) -> None:
    """Set distributed attributes for the weight tensor"""
    if not is_parallel or parallel_mode is None:
        return
    set_tensor_dist_attr(tensor, is_parallel, axis=_weight_split_axis[backend][parallel_mode])


def allreduce(
    input_: paddle.Tensor,
    tp_group: Optional[dist_group_type] = None,
) -> paddle.Tensor:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if tp_group is None or tp_group.nranks == 1:
        return input_

    # All-reduce.
    output = mp_ops._mp_allreduce(
        input_,
        group=tp_group,
        use_calc_stream=True,
        use_model_parallel=True,
    )

    return output


def identity(
    input_: paddle.Tensor,
    tp_group: Optional[dist_group_type] = None,
) -> paddle.Tensor:
    """
    Identity when forward.
    Allreduce across model parallel group when backward.
    """
    output = mp_ops._c_identity(input_, group=tp_group)

    return output
