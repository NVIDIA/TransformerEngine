from collections import deque
from __future__ import annotations
from typing import Any, Dict, Optional
from contextlib import contextmanager, nullcontext
import torch

from .tensor.quantized_tensor import QuantizedTensorBase
from .tensor.float8_tensor import Float8Tensor

# cpu offload for pipeline


class PipelineOffloadManager:
    OFFLOAD_MGR = None

    @classmethod
    def init_instance(cls, parallel_state):
        assert cls.OFFLOAD_MGR is None
        cls.OFFLOAD_MGR = PipelineOffloadManager(parallel_state)

    @classmethod
    def get_instance(cls):
        if cls.OFFLOAD_MGR is None:
            from megatron.core import parallel_state

            cls.init_instance(parallel_state)

        return cls.OFFLOAD_MGR

    def __init__(self, parallel_state):
        self._parallel_state = parallel_state
        self._queue = deque()
        self._vpp = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        # cache vpp - 1 stages
        self._stages = [[] for _ in range(self._vpp)]
        # allocate streams and events for synchronization
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()
        self._f_event = torch.cuda.Event()
        self._b_event = torch.cuda.Event()
        self._f_event.record(self._d2h_stream)
        self._b_event.record(self._h2d_stream)
        self.reset()

    @property
    def d2h_stream(self):
        return self._d2h_stream

    @property
    def h2d_stream(self):
        return self._h2d_stream

    def reset(self):
        self._inside_context = False
        self._cur_forward_chunk = None
        self._cur_backward_chunk = None
        self._first_last_vpp_rank = True

    def flush(self):
        # put into the queue in the backward order
        if len(self._stages[0]) == len(self._stages[-1]):
            lens = [len(e) for e in self._stages]
            assert min(lens) == max(lens)
            self._stages[-1] = []
            for chunks in reversed(self._stages):
                for chunk in chunks:
                    self.push(chunk)
            for i in range(self._vpp):
                self._stages[i] = []

    def push(self, handler):
        self._queue.append(handler)

    def pop(self):
        assert self.size()
        self._cur_backward_chunk = self._queue.popleft()

    def front(self):
        if not len(self._queue):
            return None
        f = self._queue.popleft()
        self._queue.appendleft(f)
        return f

    def size(self):
        return len(self._queue)

    def reset_chunk_handler(self, num_layer, offload_mlp_input=True):
        """
        reset state for a new micro batch, or another vpp chunk of the same micro batch
        """
        cur_vpp_rank = self._parallel_state.get_virtual_pipeline_model_parallel_rank()

        first_last_vpp_rank = self._first_last_vpp_rank
        # we do not offload last layer of the first last vpp rank chunk we ever meet, cause it comes first in backward
        first_last_vpp_rank = first_last_vpp_rank and (cur_vpp_rank == self._vpp - 1)
        cur_chunk = ChunkOffloadHandler(num_layer, first_last_vpp_rank, offload_mlp_input)
        # save for latter push
        self._stages[cur_vpp_rank].append(cur_chunk)
        if cur_vpp_rank == self._vpp - 1:
            self._first_last_vpp_rank = False
            self.push(cur_chunk)
            self.flush()
        self._cur_forward_chunk = cur_chunk
        cur_chunk.vpp_rank = cur_vpp_rank

    def cur_forward_chunk(self):
        """state for current forward  micro batch or vpp chunk"""
        return self._cur_forward_chunk

    def cur_backward_chunk(self):
        """state for current backward  micro batch or vpp chunk"""
        return self._cur_backward_chunk

    def __enter__(self):
        self.OFFLOAD_MGR
        self.inside_context = True

        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor, **kwargs) -> Any:
        """save hook"""
        assert self.inside_context
        return self.cur_forward_chunk().tensor_push(tensor)

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """get hook"""
        return self.cur_backward_chunk().tensor_pop(saved_state)


OFFLOAD_TAG = "offloading_mlp_input"


def offloading_checker(tensor):
    global OFFLOAD_TAG
    return (
        hasattr(tensor, OFFLOAD_TAG)
        and getattr(tensor, OFFLOAD_TAG)
        and not isinstance(tensor, torch.nn.Parameter)
    )


def set_offload_tag(tensor):
    global OFFLOAD_TAG
    setattr(tensor, OFFLOAD_TAG, True)


class ChunkOffloadHandler:

    @staticmethod
    def offload(src_tensor, pin_memory=True):
        """Offload."""
        fp8_offload = isinstance(src_tensor, Float8Tensor)

        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=torch.uint8 if fp8_offload else src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )

        if fp8_offload:
            cpu_backup = Float8Tensor.make_like(src_tensor, data=cpu_backup)

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None):
        """Reload."""
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    def __init__(self, num_layer, is_first_last_vpp_chunk, offload=True):
        self._num_layers = num_layer
        # Data Structure to maintain reference to activation tensors
        self._tensor_tag_to_state = {}
        # Data structure to hold the FP8/MXFP8 tensor objects
        self._fp8_tensor_object_map = {}
        self._float8_transpose_cache_valid = {}
        # Tracking the number of layers offloaded
        self._offloaded_group_count = 0
        self._is_first_last_vpp_chunk = is_first_last_vpp_chunk

        self._layer_index = 0
        self._tensor_count_current_layer = 0

        self.tensor_need_offloading_checker = None
        self.torch_tensor_count = 0
        self.d2h_stream = PipelineOffloadManager.get_instance().d2h_stream
        self.h2d_stream = PipelineOffloadManager.get_instance().h2d_stream
        self._f_event = PipelineOffloadManager.get_instance()._f_event
        self._b_event = PipelineOffloadManager.get_instance()._b_event
        self.do_offload = offload

    def is_first_last_layer(self):
        """whether is the last layer of first last vpp chunk ever meet for this batch"""
        return self._is_first_last_vpp_chunk and self.is_last_layer()

    def is_last_layer(self):
        """is the last layer for this chunk"""
        return self._layer_index == self._num_layers - 1

    def tensor_push(self, tensor):
        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )

        is_quantized_tensor = isinstance(tensor, QuantizedTensorBase)

        if not torch_stray_tensor:
            tensor_need_offload = False
            if (
                self.tensor_need_offloading_checker is not None
                and self.tensor_need_offloading_checker(tensor)
            ):
                # set_offload_tag(tensor)
                tensor_need_offload = True

            # obtain a unique tensor tag
            tensor_tag = (self._layer_index, self._tensor_count_current_layer)
            self._tensor_count_current_layer += 1
            assert tensor_tag not in self._tensor_tag_to_state
            if is_quantized_tensor and tensor_need_offload:
                tensor_list, _ = tensor.prepare_for_saving()
                self._tensor_tag_to_state[tensor_tag] = []
                self._fp8_tensor_object_map[tensor_tag] = tensor
                if isinstance(tensor, Float8Tensor):
                    self._float8_transpose_cache_valid[tensor_tag] = getattr(
                        tensor, "_transpose_invalid"
                    )
                for t in tensor_list:
                    set_offload_tag(t)
                    self._tensor_tag_to_state[tensor_tag].append(t)
                    # Need to clear the internal data reference for the quantized tensors
                    tensor.clear()
            else:
                if tensor_need_offload:
                    set_offload_tag(tensor)
                self._tensor_tag_to_state[tensor_tag] = tensor
        else:
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self._tensor_tag_to_state[tensor_tag] = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag):
        assert (
            tensor_tag in self._tensor_tag_to_state
        ), f"{tensor_tag}, {self._tensor_tag_to_state.keys()}"

        tensor = self._tensor_tag_to_state.pop(tensor_tag)
        if isinstance(tensor, list):
            self._fp8_tensor_object_map[tensor_tag].restore_from_saved(tensor)
            tensor = self._fp8_tensor_object_map.pop(tensor_tag)
        assert not isinstance(tensor, tuple)
        return tensor

    def set_offloading_checker(self, check_func):
        """check_func is a func with signature f(tensor) -> bool, check whether the tensor need offload"""
        self.tensor_need_offloading_checker = check_func

    @contextmanager
    def offload_checker_ctx(self, checker_func):
        origin_checker_func = self.tensor_need_offloading_checker
        try:
            self.tensor_need_offloading_checker = checker_func
            yield
        finally:
            self.tensor_need_offloading_checker = origin_checker_func

    def bulk_offload_group(self, group_to_offload):
        """Bulk offload group."""
        if not self.do_offload:
            return
        assert not self.is_first_last_layer()
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload:
                    assert not isinstance(state, tuple)
                    is_quantized_tensor = isinstance(state, list)
                    if is_quantized_tensor:
                        tensor_list = state
                        self.tensor_tag_to_state[tensor_tag] = []
                        for tensor_on_device in tensor_list:
                            assert offloading_checker(tensor_on_device)
                            state = self.offload(tensor_on_device)
                            tensor_on_device.record_stream(self.d2h_stream)
                            self.tensor_tag_to_state[tensor_tag].append(state)
                    else:
                        tensor_on_device = state
                        # if offload, return the reference to cpu copy
                        if offloading_checker(tensor_on_device):
                            # print(f"offload {group_to_offload}")
                            state = self.offload(tensor_on_device)
                            tensor_on_device.record_stream(self.d2h_stream)
                            self._tensor_tag_to_state[tensor_tag] = state

        self._offloaded_group_count = group_to_offload + 1
        self._f_event.record(self.d2h_stream)

    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        if not self.do_offload:
            return
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label, state in self._tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_to_reload:
                    if isinstance(state, tuple):
                        recovered_tensor = self.reload(state)
                        self._tensor_tag_to_state[tensor_label] = recovered_tensor
                    if isinstance(state, list):
                        state_list = self._tensor_tag_to_state[state]
                        self._tensor_tag_to_state[tensor_label] = []
                        for state_tuple in state_list:
                            recovered_tensor = self.reload(state_tuple)
                            self._tensor_tag_to_state[tensor_label].append(recovered_tensor)
                        _ = self._fp8_tensor_object_map[tensor_label].restore_from_saved(
                            self._tensor_tag_to_state[state]
                        )
                        if isinstance(self._fp8_tensor_object_map[tensor_label], Float8Tensor):
                            self._fp8_tensor_object_map[tensor_label]._transpose_invalid = (
                                self._float8_transpose_cache_valid.pop(tensor_label)
                            )
                        self._tensor_tag_to_state[tensor_label] = self._fp8_tensor_object_map.pop(
                            tensor_label
                        )

        self._offloaded_group_count = group_to_reload
        self._b_event.record(self.h2d_stream)

    def pre_reload_last_layer(self):
        """pre reload activation for the next layer in the backward order"""
        if not self.do_offload:
            return
        assert not self._is_first_last_vpp_chunk
        if self._num_layers == self._offloaded_group_count:
            self.bulk_reload_group(self._num_layers - 1)
        assert self._num_layers - 1 == self._offloaded_group_count

    def should_bulk_offload(self):
        if not self.do_offload:
            return False
        # first backward chunk
        if self.is_first_last_layer():
            return False

        # if next backward chunk is this chunk (for last pp stage)
        next_backward_chunk = PipelineOffloadManager.get_instance().get_instance().front()
        if next_backward_chunk is not None and next_backward_chunk is self:
            if self.is_last_layer():
                return False

        return True

    def forward_sync(self):
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        self._f_event.wait(torch.cuda.current_stream())
        # torch.cuda.empty_cache()

    def bulk_offload(self, offloaded_call_back):
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        # torch.cuda.empty_cache()
        if self.should_bulk_offload():
            self.bulk_offload_group(self._layer_index)
            if offloaded_call_back is not None:
                offloaded_call_back()

    def on_group_commit_forward(self, offloaded_call_back):
        # wait each other
        self.forward_sync()
        self.bulk_offload(offloaded_call_back)
        self._layer_index = self._layer_index + 1
        self._tensor_count_current_layer = 0

    def bulk_reload(self):
        if self.do_offload:
            assert self._layer_index == self._offloaded_group_count
        if self._layer_index:
            # load next layer
            self.bulk_reload_group(self._layer_index - 1)
        else:
            next_backward_chunk = PipelineOffloadManager.get_instance().front()
            if next_backward_chunk is not None:
                next_backward_chunk.pre_reload_last_layer()

    def backward_sync(self):
        self.h2d_stream.wait_stream(torch.cuda.current_stream())
        self._b_event.wait(torch.cuda.current_stream())

    def on_group_commit_backward(self):
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        if not cur_backward_chunk is self:
            PipelineOffloadManager.get_instance().pop()
        cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
        assert cur_backward_chunk is self
        self._layer_index = self._layer_index - 1
        self.backward_sync()
        # load previous layer
        self.bulk_reload()


class GroupCommitFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler, offloaded_call_back):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler.on_group_commit_forward(offloaded_call_back)
        ctx.cpu_offload_handler = cpu_offload_handler
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output, None, None


def group_prefetch_offload_commit_func(tensor, callback=None):
    cur_forward_chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
    return GroupCommitFunction.apply(tensor, cur_forward_chunk, callback)


def noop_func(tensor, callback=None):
    return tensor


def get_group_prefetch_offload_commit_func(config):
    if config.offload_moe_mlp_input and config.combined_1f1b:
        return group_prefetch_offload_commit_func
    else:
        return noop_func


def get_offload_context(config):
    if config.offload_moe_mlp_input and config.combined_1f1b:
        return PipelineOffloadManager.get_instance()
    else:
        return nullcontext()


def reset_chunk(config, layer_num):
    if config.offload_moe_mlp_input and config.combined_1f1b:
        # start a new forward chunk
        PipelineOffloadManager.get_instance().reset_chunk_handler(layer_num)
        PipelineOffloadManager.get_instance().cur_forward_chunk().set_offloading_checker(
            offloading_checker
        )


def reset_batch(config):
    if config.offload_moe_mlp_input and config.combined_1f1b:
        PipelineOffloadManager.get_instance().reset()


def offload_checker_ctx(config, offload_checker_func):
    if config.offload_moe_mlp_input and config.combined_1f1b:
        return (
            PipelineOffloadManager.get_instance()
            .cur_forward_chunk()
            .offload_checker_ctx(offload_checker_func)
        )
    return nullcontext()
