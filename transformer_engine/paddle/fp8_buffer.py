# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""FP8 meta buffer for FP8 amax reduction"""

from functools import partial
import os
from typing import Dict, Any, List, Union

import numpy as np
import paddle

from .constants import dist_group_type


class Fp8MetaBuffer:
    """
    A global buffer that holds FP8 meta for reduction across trainers.
    """

    def __init__(self):
        self._data = {}
        self._data = {}
        self._buffer_delete_key_fwd = None
        self._buffer_delete_key_bwd = None
        self._amax_forward_global_reduce_func = None
        self._amax_reduce_wait_func_fwd = None
        self._amax_reduce_wait_func_bwd = None
        self._dp_amax_reduce_interval = None
        self._dp_amax_reduce_forward_idx = 0
        self._dp_amax_reduce_backward_idx = 0

    @staticmethod
    def _get_meta_tensor_key(forward: bool = True) -> str:
        """Returns scaling key in `fp8_meta`."""
        if forward:
            return "scaling_fwd"
        return "scaling_bwd"

    @staticmethod
    def _get_buffer_position_key(forward: bool = True) -> str:
        """Returns module position key in `fp8_meta`."""
        if forward:
            return "global_fp8_buffer_pos_fwd"
        return "global_fp8_buffer_pos_bwd"

    @staticmethod
    def _get_autocast_key(forward: bool = True) -> str:
        """Returns module position key in `fp8_meta`."""
        if forward:
            return "autocast_id_fwd"
        return "autocast_id_bwd"

    @staticmethod
    def _get_amax_buffer_key(fp8_meta: Dict[str, Any], forward: bool = True) -> str:
        """Return a key in `_data` for the AMAX storage."""
        if forward:
            return f"FWD_AMAX_{fp8_meta['autocast_id_fwd']}"
        return f"BWD_AMAX_{fp8_meta['autocast_id_bwd']}"

    def _execute_deletion(self, forward: bool = True) -> None:
        """Delete the key from global amax buffer."""
        if forward:
            if (self._buffer_delete_key_fwd is not None
                    and self._buffer_delete_key_fwd in self._data):
                del self._data[self._buffer_delete_key_fwd]
        else:
            if (self._buffer_delete_key_bwd is not None
                    and self._buffer_delete_key_bwd in self._data):
                del self._data[self._buffer_delete_key_bwd]

    def _wait_handle_and_split(
        self,
        contiguous_amax: paddle.Tensor,
        chunk_sizes: List[int],
        amax_buffer_key: str,
        wait_handle: Union[bool, None],
    ) -> None:
        """Wait for amax reduction to finish and then copy reduced amax to buffer"""
        if wait_handle is not None:
            wait_handle.wait()
        self._data[amax_buffer_key] = list(contiguous_amax.split(chunk_sizes))

    def _global_amax_reduction(
        self,
        fp8_meta: Dict[str, Any],
        tp_group: dist_group_type,
        tp_size: int,
        forward: bool = True,
    ) -> None:
        """Concatenate, reduce, and split amaxes in the global buffer."""

        def _reduce_tensor_across_group_op_max(tensor, group, sync_op):
            if paddle.distributed.is_initialized():
                wait_handle = paddle.distributed.all_reduce(
                    tensor,
                    op=paddle.distributed.ReduceOp.MAX,
                    group=group,
                    sync_op=sync_op,
                )
                return wait_handle
            return None

        amax_buffer_key = self._get_amax_buffer_key(fp8_meta, forward=forward)
        # Key already deleted.
        if amax_buffer_key not in self._data:
            return None

        # Reduce AMAX in DP-domain at an interval.
        if self._dp_amax_reduce_interval is None:
            self._dp_amax_reduce_interval = int(os.getenv("NVTE_DP_AMAX_REDUCE_INTERVAL", "1"))

        tp_amax_reduce = False
        if forward:
            if self._dp_amax_reduce_forward_idx == 0:
                reduce_group = fp8_meta["fp8_group"]
            else:
                tp_amax_reduce = True
            self._dp_amax_reduce_forward_idx = (self._dp_amax_reduce_forward_idx +
                                                1) % self._dp_amax_reduce_interval
        else:
            if self._dp_amax_reduce_backward_idx == 0:
                reduce_group = fp8_meta["fp8_group"]
            else:
                tp_amax_reduce = True
            self._dp_amax_reduce_backward_idx = (self._dp_amax_reduce_backward_idx +
                                                 1) % self._dp_amax_reduce_interval

        if tp_amax_reduce:
            if tp_size > 1:
                reduce_group = tp_group
            else:
                return None

        chunk_sizes = [x.numel() for x in self._data[amax_buffer_key]]
        contiguous_amax = paddle.concat(self._data[amax_buffer_key])

        wait_handle = _reduce_tensor_across_group_op_max(
            contiguous_amax,
            reduce_group,
            not fp8_meta["async_amax_reduction"],
        )

        return partial(
            self._wait_handle_and_split,
            contiguous_amax,
            chunk_sizes,
            amax_buffer_key,
            wait_handle,
        )

    def add_amax(self, fp8_meta: Dict[str, Any], forward: bool = True) -> None:
        """Append `amax_history` to global buffer."""
        buffer_key = self._get_amax_buffer_key(fp8_meta, forward=forward)
        fp8_meta_tensor_key = self._get_meta_tensor_key(forward=forward)
        buffer_position_key = self._get_buffer_position_key(forward=forward)

        if buffer_key not in self._data:
            self._data[buffer_key] = [fp8_meta[fp8_meta_tensor_key].amax_history[0]]
        else:
            self._data[buffer_key].append(fp8_meta[fp8_meta_tensor_key].amax_history[0])

        if buffer_position_key not in fp8_meta:
            fp8_meta[buffer_position_key] = len(self._data[buffer_key]) - 1

        # Catch incorrect fp8_autocast usage.
        assert fp8_meta[buffer_position_key] == len(self._data[buffer_key]) - 1, \
            "Same module is being invoked more than once inside an `fp8_autocast` " \
            "region when using FP8 with amax reduction. This behavior is currently " \
            "unsupported. For more details and correct usage, please see " \
            "https://github.com/NVIDIA/TransformerEngine/pull/93."

    def get_amax(self, fp8_meta: Dict[str, Any], forward: bool = True) -> None:
        """Populate current amax with the correct location from buffer."""
        fp8_meta_tensor_key = self._get_meta_tensor_key(forward=forward)
        buffer_position_key = self._get_buffer_position_key(forward=forward)
        if buffer_position_key not in fp8_meta:
            return

        amax_buffer_key = self._get_amax_buffer_key(fp8_meta, forward=forward)
        assert amax_buffer_key in self._data, "TE internal error."

        fp8_meta[fp8_meta_tensor_key].amax_history[0] = self._data[amax_buffer_key][
            fp8_meta[buffer_position_key]]

    def set_for_deletion(self, fp8_meta: Dict[str, Any], forward: bool = True) -> None:
        """Delete this amax key from global buffer during autocast end."""
        if self._get_autocast_key(forward=forward) not in fp8_meta:
            return
        if forward:
            self._buffer_delete_key_fwd = self._get_amax_buffer_key(fp8_meta, forward=forward)
        else:
            self._buffer_delete_key_bwd = self._get_amax_buffer_key(fp8_meta, forward=forward)

    def get_amax_reduce_handle_fwd(self) -> Union[bool, None]:
        """Return AMAX reduction wait handle of forward prop."""
        return self._amax_reduce_handle_fwd

    def set_for_forward_amax_reduction(
        self,
        fp8_meta: Dict[str, Any],
        tp_group: dist_group_type,
        tp_size: int,
    ) -> None:
        """Sets up the function to call during autocast exit."""
        self._amax_forward_global_reduce_func = partial(
            self._global_amax_reduction,
            fp8_meta,
            tp_group,
            tp_size,
            forward=True,
        )

    def wait(self, forward: bool = True) -> None:
        """Wait for reduced amax to be available in buffer."""
        if forward:
            if self._amax_reduce_wait_func_fwd is not None:
                self._amax_reduce_wait_func_fwd()
                self._amax_reduce_wait_func_fwd = None
        else:
            if self._amax_reduce_wait_func_bwd is not None:
                self._amax_reduce_wait_func_bwd()
                self._amax_reduce_wait_func_bwd = None

    def finalize_fwd(self) -> None:
        """
        Called at FP8 autocast end.
        Performs AMAX reduction and delete unused buffer entries.
        """
        if callable(self._amax_forward_global_reduce_func):
            self._amax_reduce_wait_func_fwd = self._amax_forward_global_reduce_func()
        self._execute_deletion(forward=True)

    def finalize_bwd(
        self,
        fp8_meta: Dict[str, Any],
        tp_group: dist_group_type,
        tp_size: int,
    ) -> None:
        """
        Called at FP8 autocast end in backward.
        Performs AMAX reduction and delete unused buffer entries.
        """
        self._amax_reduce_wait_func_bwd = self._global_amax_reduction(fp8_meta,
                                                                      tp_group,
                                                                      tp_size,
                                                                      forward=False)
        self._execute_deletion(forward=False)

    def to_numpy(self) -> Dict[str, List[np.array]]:
        """Convert to numpy arrays"""
        out = {}
        for k, v in self._data.items():
            out[k] = [tensor.numpy() for tensor in v]
        return out

    def from_numpy(self, buffer: Dict[str, np.array]) -> None:
        """Set buffer values from numpy arrays"""
        for k, v in buffer.items():
            self._data[k] = [paddle.to_tensor(arr) for arr in v]
