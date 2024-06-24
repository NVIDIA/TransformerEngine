# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""FP8 meta buffer for FP8 amax reduction"""

from abc import ABC, abstractmethod
from collections import deque
from functools import partial
import os
from typing import Dict, Any, List, Union

import numpy as np
import paddle
from transformer_engine import transformer_engine_paddle as tex

from .constants import dist_group_type, RecomputeFunctionNames


class FP8MetaBufferBase(ABC):
    """
    A global buffer that holds FP8 meta for reduction across trainers.
    """

    def __init__(self):
        self._data = {}
        self._buffer_delete_key = None
        self._amax_reduce_wait_func = None
        self._dp_amax_reduce_interval = None
        self._dp_amax_reduce_idx = 0

    @staticmethod
    @abstractmethod
    def _get_meta_tensor_key():
        """Returns scaling key in `fp8_meta`."""

    @staticmethod
    @abstractmethod
    def _get_buffer_position_key():
        """Returns module position key in `fp8_meta`."""

    @staticmethod
    @abstractmethod
    def _get_autocast_key():
        """Returns autocast id key in `fp8_meta`."""

    def _get_amax_buffer_key(self, fp8_meta: Dict[str, Any]) -> str:
        """Return a key in `_data` for the AMAX storage."""
        return f"AMAX_{fp8_meta[self._get_autocast_key()]}"

    def _execute_deletion(self) -> None:
        """Delete the key from global amax buffer."""
        if self._buffer_delete_key is not None and self._buffer_delete_key in self._data:
            del self._data[self._buffer_delete_key]

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

        amax_buffer_key = self._get_amax_buffer_key(fp8_meta)
        # Key already deleted.
        if amax_buffer_key not in self._data:
            return None

        # Reduce AMAX in DP-domain at an interval.
        if self._dp_amax_reduce_interval is None:
            self._dp_amax_reduce_interval = int(os.getenv("NVTE_DP_AMAX_REDUCE_INTERVAL", "1"))

        tp_amax_reduce = False
        if self._dp_amax_reduce_idx == 0:
            reduce_group = fp8_meta["fp8_group"]
        else:
            tp_amax_reduce = True
        self._dp_amax_reduce_idx = (self._dp_amax_reduce_idx + 1) % self._dp_amax_reduce_interval

        if tp_amax_reduce:
            if tp_size > 1:
                reduce_group = tp_group
            else:
                return None

        chunk_sizes = [x.shape[0] for x in self._data[amax_buffer_key]]
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

    def add_amax(self, fp8_meta: Dict[str, Any]) -> None:
        """Append `amax_history` to global buffer."""
        buffer_key = self._get_amax_buffer_key(fp8_meta)
        fp8_meta_tensor_key = self._get_meta_tensor_key()
        buffer_position_key = self._get_buffer_position_key()

        if buffer_key not in self._data:
            self._data[buffer_key] = [fp8_meta[fp8_meta_tensor_key].amax_history[0]]
        else:
            self._data[buffer_key].append(fp8_meta[fp8_meta_tensor_key].amax_history[0])

        if buffer_position_key not in fp8_meta:
            fp8_meta[buffer_position_key] = len(self._data[buffer_key]) - 1

        # Catch incorrect fp8_autocast usage.
        assert fp8_meta[buffer_position_key] == len(self._data[buffer_key]) - 1, (
            "Same module is being invoked more than once inside an `fp8_autocast` "
            "region when using FP8 with amax reduction. This behavior is currently "
            "unsupported. For more details and correct usage, please see "
            "https://github.com/NVIDIA/TransformerEngine/pull/93."
        )

    def copy_amax_from_buffer(self, fp8_meta: Dict[str, Any]) -> None:
        """Populate current amax with the correct location from buffer."""
        fp8_meta_tensor_key = self._get_meta_tensor_key()
        buffer_position_key = self._get_buffer_position_key()
        if buffer_position_key not in fp8_meta:
            return

        amax_buffer_key = self._get_amax_buffer_key(fp8_meta)
        assert amax_buffer_key in self._data, "TE internal error."

        # Copy amax to amax_history[0]
        tex.update_latest_amax_history_inplace(
            _history=fp8_meta[fp8_meta_tensor_key].amax_history,
            amax=self._data[amax_buffer_key][fp8_meta[buffer_position_key]],
        )

    def set_for_deletion(self, fp8_meta: Dict[str, Any]) -> None:
        """Delete this amax key from global buffer during autocast end."""
        if self._get_autocast_key() not in fp8_meta:
            return
        self._buffer_delete_key = self._get_amax_buffer_key(fp8_meta)

    def get_amax_reduce_handle(self) -> Union[bool, None]:
        """Return AMAX reduction wait handle."""
        return self._amax_reduce_handle

    def wait(self) -> None:
        """Wait for reduced amax to be available in buffer."""
        if self._amax_reduce_wait_func is not None:
            self._amax_reduce_wait_func()  # pylint: disable=not-callable
            self._amax_reduce_wait_func = None

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


class FP8MetaFwdBuffer(FP8MetaBufferBase):
    """FP8Meta Buffer for forward"""

    @staticmethod
    def _get_meta_tensor_key() -> str:
        """Returns scaling key in `fp8_meta`."""
        return "scaling_fwd"

    @staticmethod
    def _get_buffer_position_key() -> str:
        """Returns module position key in `fp8_meta`."""
        return "global_fp8_buffer_pos_fwd"

    @staticmethod
    def _get_autocast_key() -> str:
        """Returns module position key in `fp8_meta`."""
        return "autocast_id_fwd"

    def set_for_amax_reduction(
        self,
        fp8_meta: Dict[str, Any],
        tp_group: dist_group_type,
        tp_size: int,
    ) -> None:
        """Sets up the function to call during autocast exit."""
        self._amax_global_reduce_func = partial(
            self._global_amax_reduction,
            fp8_meta,
            tp_group,
            tp_size,
        )

    def finalize(self) -> None:
        """
        Called at FP8 autocast end.
        Performs AMAX reduction and delete unused buffer entries.
        """
        if hasattr(self, "_amax_global_reduce_func") and callable(self._amax_global_reduce_func):
            self._amax_reduce_wait_func = self._amax_global_reduce_func()
        self._execute_deletion()


class FP8MetaBwdBuffer(FP8MetaBufferBase):
    """FP8Meta Buffer for backward"""

    @staticmethod
    def _get_meta_tensor_key() -> str:
        """Returns scaling key in `fp8_meta`."""
        return "scaling_bwd"

    @staticmethod
    def _get_buffer_position_key() -> str:
        """Returns module position key in `fp8_meta`."""
        return "global_fp8_buffer_pos_bwd"

    @staticmethod
    def _get_autocast_key() -> str:
        """Returns module position key in `fp8_meta`."""
        return "autocast_id_bwd"

    def finalize(
        self,
        fp8_meta: Dict[str, Any],
        tp_group: dist_group_type,
        tp_size: int,
    ) -> None:
        """
        Called at FP8 autocast end in backward.
        Performs AMAX reduction and delete unused buffer entries.
        """
        self._amax_reduce_wait_func = self._global_amax_reduction(fp8_meta, tp_group, tp_size)
        self._execute_deletion()


class FP8RecomputeBuffer:
    """Buffer used to hold FP8 meta tensors for recompute"""

    def __init__(self):
        self._data = []

    @staticmethod
    def get_buffer_position_key():
        """Returns the key (in fp8_meta) for recompute buffer position"""
        return "recompute_buffer_pos"

    def stash_fp8_meta_tensors(self, fp8_meta: Dict[str, Any]) -> None:
        """Stash the scaling factors and amaxes for recompute"""
        buffer_position_key = self.get_buffer_position_key()

        to_copy = [
            fp8_meta["scaling_fwd"].amax_history.clone(),
            fp8_meta["scaling_fwd"].scale.clone(),
            fp8_meta["scaling_fwd"].scale_inv.clone(),
        ]

        if buffer_position_key in fp8_meta:
            self._data[fp8_meta[buffer_position_key]].append(to_copy)
        else:
            self._data.append(deque())
            self._data[-1].append(to_copy)
            fp8_meta[buffer_position_key] = len(self._data) - 1

    def retrieve_fp8_meta_tensors(self, fp8_meta: Dict[str, Any]) -> None:
        """Switch to the previously saved scaling factors and amaxes"""
        # Store updated amaxes and scales from phase 1 post forward.
        fp8_meta["updated_amax_history_fwd"] = fp8_meta["scaling_fwd"].amax_history
        fp8_meta["updated_scale_fwd"] = fp8_meta["scaling_fwd"].scale
        fp8_meta["updated_scale_inv_fwd"] = fp8_meta["scaling_fwd"].scale_inv

        # Retrieve stashed amaxes and scales from phase 1 pre forward.
        buffer_position_key = self.get_buffer_position_key()
        stashed_fp8_meta = self._data[fp8_meta[buffer_position_key]].popleft()

        # Replace amaxes and scales with stashed values for phase 2 forward
        fp8_meta["scaling_fwd"].amax_history = stashed_fp8_meta[0]
        fp8_meta["scaling_fwd"].scale = stashed_fp8_meta[1]
        fp8_meta["scaling_fwd"].scale_inv = stashed_fp8_meta[2]

    @staticmethod
    def restore_fp8_meta_tensors(fp8_meta: Dict[str, Any]) -> None:
        """Restore latest scaling factors and amaxes after recompute forward run."""
        assert "updated_amax_history_fwd" in fp8_meta, (
            "Recompute internal error."
            " If you are not using recompute, please check if"
            " the forward function is called from one of these functions: "
            f"{RecomputeFunctionNames}. If so, consider change the function name "
            "or set NVTE_DISABLE_RECOMPUTE=1."
        )
        fp8_meta["scaling_fwd"].amax_history = fp8_meta["updated_amax_history_fwd"]
        fp8_meta["scaling_fwd"].scale = fp8_meta["updated_scale_fwd"]
        fp8_meta["scaling_fwd"].scale_inv = fp8_meta["updated_scale_inv_fwd"]
