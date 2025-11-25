# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functionality for CPU offloading of tensors saved for backward pass."""

from __future__ import annotations
import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
import os
import warnings
from typing import Any, Optional
import torch
from torch.autograd.graph import saved_tensors_hooks
from transformer_engine.debug.pytorch.debug_state import TEDebugState
import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpu_offload_v1 as v1_code_path
from .quantized_tensor import (
    restore_from_saved,
    prepare_for_saving,
)


__all__ = ["get_cpu_offload_context", "mark_not_offload", "start_offload"]

NVTE_CPU_OFFLOAD_V1 = os.environ.get("NVTE_CPU_OFFLOAD_V1", "0") == "1"

OFFLOAD_SYNCHRONIZER = None


def is_cpu_offload_enabled():
    """Returns True if CPU offload is enabled."""
    if NVTE_CPU_OFFLOAD_V1:
        return v1_code_path.is_cpu_offload_enabled()
    return OFFLOAD_SYNCHRONIZER is not None


def mark_activation_offload(*tensors):
    """Set the type of the offloading needed for a tensor."""
    if NVTE_CPU_OFFLOAD_V1:
        v1_code_path.mark_activation_offload(*tensors)


def mark_not_offload(*tensors: torch.Tensor):
    """Marks tensors to prevent them from being offloaded."""
    if NVTE_CPU_OFFLOAD_V1:
        return

    tensors, tensor_obj = prepare_for_saving(*tensors)

    for tensor in tensors:
        if tensor is not None:
            setattr(tensor, "_TE_do_not_offload", True)

    restore_from_saved(tensor_obj, tensors)


def start_offload(*tensors: torch.Tensor, offload_base_tensor: bool = False):
    """
    Marks point in on main stream where tensors are fully computed and ready to be offloaded.
    If offload_base_tensor is True and the tensor is a view, the base tensor is offloaded
    and reloaded - the stride and storage offset of the view are saved and restored after reload.
    It is useful when multiple tensors are views of the same base tensor,
    for example in MultiHeadAttention for interleaved q, k, v tensors.
    """
    if NVTE_CPU_OFFLOAD_V1:
        return

    def _mark_tensor_for_offload(t):
        if t is None:
            return
        # Attach an event to mark when the tensor is ready for reload.
        t.start_reload_event = torch.cuda.Event()
        t.start_reload_event.record(torch.cuda.current_stream())
        if offload_base_tensor and t._base is not None:
            setattr(t, "offload_base_tensor", True)

    tensors, tensor_obj = prepare_for_saving(*tensors)

    for tensor in tensors:
        _mark_tensor_for_offload(tensor)

    restore_from_saved(tensor_obj, tensors)


@dataclass
class TensorGroup:
    """
    TensorGroup is a collection of tensors, events and auxiliary data.
    It is used multiple times in the CPU offload code.
    """

    tensor_list: list[torch.Tensor] = field(default_factory=list)
    events: list[torch.cuda.Event] = field(default_factory=list)
    aux: Any = None


class TensorGroupProcessor:
    """
    Suppose there is a tensor group T that needs to be offloaded.
    Possibly we can switch T into (T_opt, aux), where T_opt is smaller and easier to offload,
    offload T_opt, reload it and then restore T from (T_opt_reloaded, aux).

    This class contains static methods that perform these optimizations - for example
    deduplication of tensors and restoring duplicates after reload.
    """

    @staticmethod
    def tensor_group_process_before_offload(tensor_group: TensorGroup) -> tuple[TensorGroup, Any]:
        """
        Call for a tensor group, just before offloading logic.

        aux is a dictionary that contains auxiliary data, needed to restore pre-offload state.
        """
        aux = {}
        tensor_group = TensorGroupProcessor._switch_to_base_tensors(aux, tensor_group)
        tensor_group = TensorGroupProcessor._deduplicate_tensors(aux, tensor_group)
        return tensor_group, aux

    @staticmethod
    def tensor_group_process_after_reload(tensor_group: TensorGroup):
        """
        Call for a tensor group, just after reload logic.
        """
        assert tensor_group.aux is not None
        tensor_group = TensorGroupProcessor._restore_tensor_duplicates(tensor_group)
        tensor_group = TensorGroupProcessor._switch_to_views(tensor_group)
        return tensor_group

    @staticmethod
    def _switch_to_base_tensors(aux, tensor_group: TensorGroup) -> TensorGroup:
        """
        Changes tensors to base tensors and saves view options in aux.

        It we save multiple tensors which in fact are views of the same base tensor,
        this will offload only this one base tensor. It is used for example in
        MultiHeadAttention for interleaved q, k, v tensors.
        """

        def _check_if_offload_base_tensor(tensor: torch.Tensor) -> bool:
            if getattr(tensor, "offload_base_tensor", False):
                return True
            if tensor._base is not None:
                # If tensor is a view of a tensor and has the same elements,
                # but with different strides, we can safely offload the base tensor.
                # If tensor is a view on some part of a bigger tensor,
                # the decision to offload the base tensor is non-trivial and we do not do it by default.
                return tensor._base.numel() == tensor.numel()
            return False

        aux["views"] = []
        for tensor_id in range(  # pylint: disable=consider-using-enumerate
            len(tensor_group.tensor_list)
        ):
            tensor = tensor_group.tensor_list[tensor_id]
            if _check_if_offload_base_tensor(tensor):
                aux["views"].append((tensor.shape, tensor.stride(), tensor.storage_offset()))
                tensor = tensor._base
                assert (
                    tensor is not None
                ), "Cannot offload base tensor, if the tensor is not a view."
                tensor_group.tensor_list[tensor_id] = tensor
            else:
                aux["views"].append(None)
        return tensor_group

    @staticmethod
    def _deduplicate_tensors(aux, tensor_group: TensorGroup) -> TensorGroup:
        """
        Deduplicate tensors.
        """
        dedup_tensors: list[torch.Tensor] = []
        dedup_events: list[torch.cuda.Event] = []
        tensor_to_index: dict[int, int] = {}
        aux["original_tensor_ids"] = []
        # If there are several duplicates of the same tensor, with different events,
        # we keep only first event - every event is recorded when the tensor is ready to be offloaded,
        # so it is the most optimal to use the first event.
        for tensor_id, tensor in enumerate(tensor_group.tensor_list):
            if id(tensor) in tensor_to_index:
                aux["original_tensor_ids"].append(tensor_to_index[id(tensor)])
            else:
                tensor_to_index[id(tensor)] = len(dedup_tensors)
                dedup_tensors.append(tensor)

                dedup_events.append(tensor_group.events[tensor_id])
                aux["original_tensor_ids"].append(tensor_to_index[id(tensor)])

        tensor_group.tensor_list = dedup_tensors
        tensor_group.events = dedup_events
        return tensor_group

    @staticmethod
    def _restore_tensor_duplicates(tensor_group: TensorGroup) -> TensorGroup:
        """
        Restore tensor duplicates.
        """
        new_tensor_list = []
        new_events_list = []
        for tensor_id in range(len(tensor_group.aux["original_tensor_ids"])):
            original_tensor_id = tensor_group.aux["original_tensor_ids"][tensor_id]
            new_tensor_list.append(tensor_group.tensor_list[original_tensor_id])
            new_events_list.append(tensor_group.events[original_tensor_id])

        tensor_group.tensor_list = new_tensor_list
        tensor_group.events = new_events_list
        return tensor_group

    @staticmethod
    def _switch_to_views(tensor_group: TensorGroup) -> TensorGroup:
        """
        Switch to views - reverse of _switch_to_base_tensors.
        """
        for tensor_id, tensor in enumerate(tensor_group.tensor_list):
            if tensor_group.aux["views"][tensor_id] is not None:
                tensor_group.tensor_list[tensor_id] = tensor.as_strided(
                    *tensor_group.aux["views"][tensor_id]
                )
        return tensor_group


class OffloadableLayerState:
    """
    Class that manages offloading and reloading of tensors for a single layer.
    """

    def __init__(
        self,
        offload_stream: torch.cuda.Stream,
        retain_pinned_cpu_buffers: bool = False,
    ):
        self.offload_stream = offload_stream
        self.retain_pinned_cpu_buffers = retain_pinned_cpu_buffers

        # There are 3 tensor groups: tensors on gpu before offload,
        # tensors on cpu after offload, tensors on gpu after reload.
        self.fwd_gpu_tensor_group = TensorGroup()
        self.cpu_tensor_group = TensorGroup()
        self.bwd_gpu_tensor_group = TensorGroup()

        self.aux: dict[str, Any] = {}

        # State can be one of: not_offloaded, offload_started,
        # offload_finished, reload_started.
        self.state = "not_offloaded"

    def _validate_state(self, func_name: str, allowed_states: list[str]):
        assert (
            self.state in allowed_states
        ), f"Invalid state: {self.state} for {func_name}, must be one of {allowed_states}"

    def start_offload(self):
        """
        Start offloading of tensors. Puts copy from GPU to CPU tasks on offload stream.
        Before each copy event, the offload stream waits for the event signalling that the tensor is ready to be offloaded.
        This event is recorded in the start_offload or push_tensor call.
        """
        self._validate_state(func_name="start_offload", allowed_states=["not_offloaded"])
        self.state = "offload_started"

        self.fwd_gpu_tensor_group, aux = TensorGroupProcessor.tensor_group_process_before_offload(
            self.fwd_gpu_tensor_group
        )

        allocate_cpu_buffers = (
            not self.retain_pinned_cpu_buffers or len(self.cpu_tensor_group.tensor_list) == 0
        )

        for tensor_id, tensor in enumerate(self.fwd_gpu_tensor_group.tensor_list):
            assert tensor.is_contiguous()

            # Wait for the moment the tensor is ready to be offloaded.
            self.offload_stream.wait_event(self.fwd_gpu_tensor_group.events[tensor_id])  # type: ignore[arg-type]

            with torch.cuda.stream(self.offload_stream):
                if allocate_cpu_buffers:
                    # empty_like is defined also for QuantizedTensors
                    offloaded_tensor = torch.empty_like(
                        tensor, device=torch.device("cpu"), pin_memory=True
                    )
                    self.cpu_tensor_group.tensor_list.append(offloaded_tensor)
                else:
                    assert self.cpu_tensor_group.tensor_list[tensor_id].shape == tensor.shape, (
                        "CPU buffer shape does not match the offloaded tensor shape:"
                        f" {self.cpu_tensor_group.tensor_list[tensor_id].shape} != {tensor.shape}  "
                        "                       Make sure that tensor shaped do not change between"
                        " iterations if retain_pinned_cpu_buffers is True."
                    )
                    offloaded_tensor = self.cpu_tensor_group.tensor_list[tensor_id]
                offloaded_tensor.copy_(tensor, non_blocking=True)

        # aux is a dictionary that contains auxiliary data like information which tensors were deduplicated,
        # needed to restore pre-offload state after reload.
        self.aux = aux

        self.finish_offload_event = torch.cuda.Event()
        self.finish_offload_event.record(self.offload_stream)

    def release_activation_forward_gpu_memory(self):
        """
        Release GPU memory of the activations.
        Waits for offload to finish - memory needs to be kept alive when GPU->CPU copy is performed.
        """
        self._validate_state(
            func_name="release_activation_forward_gpu_memory", allowed_states=["offload_started"]
        )
        self.state = "offload_finished"

        torch.cuda.current_stream().wait_event(self.finish_offload_event)  # type: ignore[arg-type]

        # GPU memory can be released safely after the offload.
        # Notice that the memory needs to be kept alive when GPU->CPU copy is performed.
        self.fwd_gpu_tensor_group = TensorGroup()
        del self.finish_offload_event

    def start_reload(self):
        """
        Start reloading of tensors.
        It allocates new tensors on GPU and puts copy from CPU tasks on offload stream.
        """
        self._validate_state(func_name="start_reload", allowed_states=["offload_finished"])
        self.state = "reload_started"

        self.bwd_gpu_tensor_group = TensorGroup()
        for tensor in self.cpu_tensor_group.tensor_list:

            # Notice that reloaded tensor is allocated on main stream,
            # not offloaded stream. It is because PyTorch memory allocator
            # cannot move tensors from pool of one stream to another without
            # calling cudaFree and cudaMalloc again.

            # empty_like is defined also for QuantizedTensors.
            reloaded_tensor = torch.empty_like(tensor, device=torch.device("cuda"))
            self.offload_stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(self.offload_stream):
                reloaded_tensor.copy_(tensor, non_blocking=True)

            reload_tensor_event = torch.cuda.Event()
            reload_tensor_event.record(self.offload_stream)
            self.bwd_gpu_tensor_group.events.append(reload_tensor_event)
            self.bwd_gpu_tensor_group.tensor_list.append(reloaded_tensor)

        self.bwd_gpu_tensor_group.aux = self.aux
        self.bwd_gpu_tensor_group = TensorGroupProcessor.tensor_group_process_after_reload(
            self.bwd_gpu_tensor_group
        )

    def push_tensor(self, tensor: torch.Tensor) -> int | torch.Tensor:
        """
        It is called when a tensor is saved for backward pass.

        If tensor is offloaded, returns int representing the index of the tensor in the offloaded tensor group.
        If tensor is not offloaded, returns the tensor itself.
        """
        self._validate_state(func_name="push_tensor", allowed_states=["not_offloaded"])

        if self._check_if_offload(tensor):
            self.fwd_gpu_tensor_group.tensor_list.append(tensor)
            # The group is processed and offloaded at the end of the forward pass of current layer.
            # To enable offloading of tensors faster we use self.offload_stream and record
            # the events when the tensors are ready to be offloaded.
            # It means that we do not need to wait to the end of current layer to start offloading.
            if hasattr(tensor, "start_reload_event"):
                self.fwd_gpu_tensor_group.events.append(tensor.start_reload_event)
            else:
                self.fwd_gpu_tensor_group.events.append(torch.cuda.Event())
                self.fwd_gpu_tensor_group.events[-1].record(torch.cuda.current_stream())
            return len(self.fwd_gpu_tensor_group.tensor_list) - 1
        return tensor

    def pop_tensor(self, tensor_or_tensor_id: torch.Tensor | int) -> torch.Tensor:
        """
        It is called when a tensor is used in backward pass.
        Returns the tensor. If tensor was offloaded/reloaded, wait for the reload of a tensor to finish.
        """
        self._validate_state(
            func_name="pop_tensor", allowed_states=["not_offloaded", "reload_started"]
        )

        # 1. tensor not offloaded
        if isinstance(tensor_or_tensor_id, torch.Tensor):
            return tensor_or_tensor_id
        # 2. the layer was not offloaded at all
        if self.state == "not_offloaded":
            return self.fwd_gpu_tensor_group.tensor_list[tensor_or_tensor_id]

        # 3. the layer was offloaded
        assert self.state == "reload_started"
        # wait for the tensor to be reloaded
        torch.cuda.current_stream().wait_event(
            self.bwd_gpu_tensor_group.events[tensor_or_tensor_id]
        )
        return self.bwd_gpu_tensor_group.tensor_list[tensor_or_tensor_id]

    def release_all_memory(self):
        """Release all gpu and cpu memory the state stored. Is called after the backward pass."""
        self.fwd_gpu_tensor_group = TensorGroup()
        if not self.retain_pinned_cpu_buffers:
            self.cpu_tensor_group = TensorGroup()
        self.bwd_gpu_tensor_group = TensorGroup()
        self.state = "not_offloaded"

    def _check_if_offload(self, t: torch.Tensor) -> bool:
        """
        Check if tensor needs to be offloaded.
        """
        if (
            not isinstance(t, torch.nn.Parameter)
            and not getattr(t, "_TE_do_not_offload", False)
            and not isinstance(t, torch._subclasses.FakeTensor)
            and t.device.type == "cuda"
        ):
            if not t.is_contiguous() and not getattr(t, "offload_base_tensor", False):
                warnings.warn(
                    "Tried to offload non-contiguous tensor, which is not supported. Offload of"
                    " this tensor will be skipped."
                )
                return False

            return True
        return False

    def get_offloaded_total_size_mb(self) -> float:
        """
        Get total size of offloaded tensors in MB, used only for testing.
        """

        def get_tensor_size_mb(tensor):
            if tensor is None:
                return 0
            if isinstance(tensor, te.quantized_tensor.QuantizedTensorStorage):
                return sum(get_tensor_size_mb(t) for t in tensor.get_data_tensors())
            return tensor.numel() * tensor.element_size() / (1024**2)

        total_size = 0
        for tensor in self.cpu_tensor_group.tensor_list:
            total_size += get_tensor_size_mb(tensor)
        return total_size


class OffloadSynchronizer:
    """
    Base class responsible for synchronizing offloading and reloading of tensors for multiple layers.
    In base class we only track layer number and
    create OffloadableLayerState instances for all layers, but do not start offloading or reloading.
    """

    def __init__(
        self,
        num_layers: int,
        retain_pinned_cpu_buffers: bool = False,
        offload_stream: Optional[torch.cuda.Stream] = None,
    ):
        self.num_layers = num_layers
        self.offload_stream = offload_stream if offload_stream is not None else torch.cuda.Stream()

        self.layer_states = {
            i: OffloadableLayerState(self.offload_stream, retain_pinned_cpu_buffers)
            for i in range(num_layers)
        }

        self.num_of_fwds = None
        self.previous_bwd_layer_id = None
        self.current_layer_id = None

    def fwd_step(self) -> int:
        """
        Invoked before each layer forward.
        """
        if self.num_of_fwds in [None, self.num_layers - 1]:
            # reset the offload synchronizer
            for layer_id in self.layer_states:
                self.layer_states[layer_id].release_all_memory()
            self.num_of_fwds = 0
        else:
            self.num_of_fwds += 1
        self.current_layer_id = self.num_of_fwds
        return self.current_layer_id

    def bwd_step(self, layer_num: int):
        """
        Invoked before each layer backward.
        """
        if self.previous_bwd_layer_id is not None:
            self.layer_states[self.previous_bwd_layer_id].release_all_memory()
        self.previous_bwd_layer_id = layer_num
        self.current_layer_id = layer_num

    def push_tensor(self, tensor: torch.Tensor) -> int | torch.Tensor:
        """Default push tensor method"""
        return self.layer_states[self.num_of_fwds].push_tensor(tensor)

    def pop_tensor(self, tensor_or_tensor_id: torch.Tensor | int) -> torch.Tensor:
        """Default pop tensor method"""
        return self.layer_states[self.current_layer_id].pop_tensor(tensor_or_tensor_id)

    def finish_part_of_bwd(self):
        """
        We need to release memory of backward - this call does that.
        It needs to be invoked after every backward pass - there may be
        more than one in pipeline parallelism.

        It is needed, because call bwd_step is invoked before each layer backward,
        but we need to release memory after the backward pass is finished.
        """
        if self.previous_bwd_layer_id is not None:
            self.layer_states[self.previous_bwd_layer_id].release_all_memory()
        self.previous_bwd_layer_id = None

    def get_offloaded_total_size_mb(self) -> float:
        """
        Get total size of offloaded tensors in MB, used only for testing.
        """
        return sum(
            self.layer_states[layer_id].get_offloaded_total_size_mb()
            for layer_id in self.layer_states
        )


class DefaultOffloadSynchronizer(OffloadSynchronizer):
    """
    Default implementation of OffloadSynchronizer,
    intended to be used in standard training workloads - with multiple forwards
    and multiple backwards.
    """

    def __init__(
        self,
        num_layers: int,
        num_offloaded_layers: int | None = None,
        retain_pinned_cpu_buffers: bool = False,
        offload_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__(num_layers, retain_pinned_cpu_buffers, offload_stream)

        # map of layers to bool meaning if layer needs to be offloaded
        self.offload_layer_map: dict[int, bool] = {}

        # num_layer: int -> list of layers that need to finish offload by this moment
        self.finish_offload_map: defaultdict[int, list[int]] = defaultdict(list)
        # num_layer: int -> list of layers that need to start reload in this moment
        self.start_reload_map: defaultdict[int, list[int]] = defaultdict(list)

        self._init_offload_synchronization_dicts(num_offloaded_layers)

    def _init_offload_synchronization_dicts(self, num_offloaded_layers: int):
        """
        If synchronization dictionary is not provided, the number of offloaded layers is used to initialize
        offload_layer_map, finish_offload_map and start_reload_map.

        The aim is to minimize memory usage by the end of the forward pass.

        The optimal strategy for that is to offload layers 0, ..., num_offloaded_layers - 1.
        For layer i offload needs to finish before num_layers - num_offloaded_layers + i.
        For layer i reload needs to start after num_layers - num_offloaded_layers + i.

        This ensures that - if all layers have memory footprint of T - then peak memory usage of saving activations is
        (num_layers - num_offloaded_layers) * T.
        """
        for layer_id in range(self.num_layers):
            if layer_id < num_offloaded_layers:
                self.offload_layer_map[layer_id] = True
                self.finish_offload_map[self.num_layers - num_offloaded_layers + layer_id].append(
                    layer_id
                )
                self.start_reload_map[self.num_layers - 1 - num_offloaded_layers + layer_id].append(
                    layer_id
                )
            else:
                self.offload_layer_map[layer_id] = False

    def fwd_step(self) -> int:
        """
        Invoked before each layer forward.
        """
        super().fwd_step()
        if self.offload_layer_map.get(self.current_layer_id - 1, False):
            self.layer_states[self.current_layer_id - 1].start_offload()

        for layer in self.finish_offload_map[self.current_layer_id]:
            self.layer_states[layer].release_activation_forward_gpu_memory()
        return self.current_layer_id

    def bwd_step(self, layer_num: int):
        """
        Invoked before each layer backward.
        """
        super().bwd_step(layer_num)

        for layer in self.start_reload_map[layer_num]:
            self.layer_states[layer].start_reload()


class ManualOffloadSynchronizer(OffloadSynchronizer):
    """
    Manual implementation of OffloadSynchronizer,
    all synchronization is done manually by the user by using
    one of the following methods:
    - start_offload_layer
    - release_activation_forward_gpu_memory
    - start_reload_layer

    This implementation is intended to be used in more complex trainigs workflows.
    It is useful for example in pipeline parallelism.
    """

    def start_offload_layer(self, layer_id: int):
        """
        Start offloading of the layer.
        Each tensor GPU->CPU copy is done asynchronously on the offload stream.
        Start of each copy is started after tensor_push() is called on the current stream.
        """
        self.layer_states[layer_id].start_offload()

    def release_activation_forward_gpu_memory(self, layer_id: int):
        """
        Release memory of the activations of the layer.
        It waits for the offload of the layer to finish.
        """
        self.layer_states[layer_id].release_activation_forward_gpu_memory()

    def start_reload_layer(self, layer_id: int):
        """
        Start reloading of the layer.
        Each tensor reload is awaited to finish before tensor_pop() for that tensor is called on the current stream.
        """
        self.layer_states[layer_id].start_reload()


def get_cpu_offload_context(
    enabled: bool = False,
    num_layers: Optional[int] = 1,
    model_layers: int = 1,
    offload_activations: bool = True,
    offload_weights: bool = False,
    double_buffering: bool = False,  # pylint: disable=unused-argument
    manual_synchronization: bool = False,
    retain_pinned_cpu_buffers: bool = False,
    offload_stream: Optional[torch.cuda.Stream] = None,
):
    """
    CPU Offloading feature for seqeuences of layers. Can be used for arbitrary layers, not necessarily
    for these provided by the TE.

    Usage:

    .. code-block:: python

        cpu_offload_context, sync_function = get_cpu_offload_context(...)

        for _ in range(num_layers):
            with cpu_offload_context:
                x = layers[i].forward(x)
            x = sync_function(x)

    Parameters
    ----------
    enabled: bool, default = `False`
             When set to True, CPU Offloading functionality is enabled.
    num_layers: int, default = 1
            Determines the number of layers
            you want to offload activations/weights for.
    model_layers: int, default = 1
            Number of layers in the model that will be used under this context.
    offload_activations: bool, default = `True`
            Deprecated.
    offload_weights: bool, default = `True`
            Deprecated.
    double_buffering: bool, default = `False`
            Deprecated.
    retain_pinned_cpu_buffers: bool, default = `False`
            If True, the pinned CPU buffers are retained after offloading
            and reused for the next iteration. It is useful for cuda graphs capture.
    manual_synchronization: bool, default = `False`
            If True, the synchronization is done manually by the user.
            Additional argument manual_controller is returned. See more in manual control section.
    offload_stream: torch.cuda.Stream, default = `None`
            If provided, the offload stream is used for offloading and reloading.
            Otherwise, a new stream is allocated internally. It can be other than None
            only if manual_synchronization is True.

    Manual synchronization
    ----------
    By default, layers are offloaded/reloaded asynchronously
    with respect to the current forward/backward stream with predefined synchronization,
    to ensure that activation memory usage is equal to
    `(num_layers - num_offloaded_layers) * T`, where `T` is the memory footprint of a layer.

    For more control over the offloading and reloading process, you can set `manual_synchronization=True`.
    In this case, an additional argument, `manual_controller`, is returned.

    The `manual_controller` provides the following methods:
    - `start_offload_layer(layer_id: int)`
    - `release_activation_forward_gpu_memory(layer_id: int)`
    - `start_reload_layer(layer_id: int)`

    If none of these methods are invoked for a given layer, that layer will not be offloaded or reloaded.
    If `start_offload_layer()` is called for a layer, offload copies for that layer begin asynchronously on the offload stream.

    Since GPU activations must be kept in memory until the copy is finished, pointers to all activations are stored.
    To release this memory, you need to call `release_activation_forward_gpu_memory(layer_id)`.
    This method makes the current stream wait for an event recorded on the offload stream after all tensors from the layer have been offloaded.

    The `start_reload_layer()` method is used to start reloading a layer.
    Each tensor reload is awaited to finish before `tensor_pop()` for that tensor is called on the current stream.

    You can provide an `offload_stream` to be used for offload and reload operations.
    This allows for more detailed synchronization, such as delaying the start of offloading.

    Example:
    .. code-block:: python
        offload_stream = torch.cuda.Stream()
        cpu_offload_context, sync_function, manual_controller = get_cpu_offload_context(
            enabled=True, model_layers=num_layers, manual_synchronization=True, offload_stream=offload_stream)

        for i in range(num_layers):
            with cpu_offload_context:
                out[i] = layers[i].forward(inp[i])
            out[i] = sync_function(out[i])
            manual_controller.start_offload_layer(i)

        offload_stream.synchronize()
        for i in range(num_layers):
            manual_controller.release_activation_forward_gpu_memory(i)

        for i in range(num_layers - 1, -1, -1):
            manual_controller.start_reload_layer(i)

        offload_stream.synchronize()
        for i in range(num_layers):
            out[i].sum().backward()

    V1 code path
    ----------
    If you want to use the v1 code path for offloading,
    please set the environment variable NVTE_CPU_OFFLOAD_V1 to 1.

    """
    if NVTE_CPU_OFFLOAD_V1:
        return v1_code_path.get_cpu_offload_context(
            enabled=enabled,
            num_layers=num_layers,
            model_layers=model_layers,
            offload_activations=offload_activations,
            offload_weights=offload_weights,
            double_buffering=double_buffering,
        )

    if not enabled:
        if manual_synchronization:
            return contextlib.nullcontext(), lambda x: x, None
        return contextlib.nullcontext(), lambda x: x

    if not offload_weights and not offload_activations:
        raise ValueError(
            "CPU Offloading is enabled while it is not "
            "mentioned what to offload (weights/activations)"
        )

    if offload_weights:
        warnings.warn(
            "Offloading weights is deprecated. Using offload_weights=True does not have any"
            " effect.",
            DeprecationWarning,
        )

        # Weights offloading is deprecated but we maintain backward compatibility by doing nothing.
        if not offload_activations:
            if manual_synchronization:
                return contextlib.nullcontext(), lambda x: x, None
            return contextlib.nullcontext(), lambda x: x

    if TEDebugState.debug_enabled:
        raise RuntimeError("CPU offload is not supported in debug mode.")

    if not manual_synchronization:
        assert (
            num_layers <= model_layers - 1
        ), "Cannot offload all layers without manual synchronization - last layer is not offloaded."
        if num_layers == model_layers - 1:
            warnings.warn(
                "Offloading num_layers == model_layers - 1 is not recommended, it prevents"
                " overlapping of computation and offload/reload."
            )

    assert (
        offload_stream is None or manual_synchronization
    ), "offload_stream can be provided only if manual_synchronization is True"

    if manual_synchronization:
        offload_synchronizer = ManualOffloadSynchronizer(
            model_layers, retain_pinned_cpu_buffers, offload_stream
        )
    else:
        offload_synchronizer = DefaultOffloadSynchronizer(
            model_layers,
            num_layers,
            retain_pinned_cpu_buffers,
            offload_stream,
        )

    class _CpuOffloadContext(contextlib.ContextDecorator):
        def __init__(self):
            self.current_layer = None
            self.previous_offload_synchronizer = None
            self.offload_synchronizer = offload_synchronizer

            self.inside_context = False

        def __enter__(self):
            assert (
                self.inside_context is False
            ), "Offloading context was entered without synchronization function being called."
            self.inside_context = True
            self._hooks_ctx = saved_tensors_hooks(
                offload_synchronizer.push_tensor, offload_synchronizer.pop_tensor
            )
            self._hooks_ctx.__enter__()
            global OFFLOAD_SYNCHRONIZER
            self.previous_offload_synchronizer = OFFLOAD_SYNCHRONIZER
            OFFLOAD_SYNCHRONIZER = offload_synchronizer
            self.current_layer = offload_synchronizer.fwd_step()
            return self

        def __exit__(self, *args):
            self._hooks_ctx.__exit__(*args)
            global OFFLOAD_SYNCHRONIZER
            OFFLOAD_SYNCHRONIZER = self.previous_offload_synchronizer
            self.inside_context = False

        def synchronization_function(self, tensor):
            """
            This function is used to catch the backward pass of the model.
            """
            assert tensor.requires_grad is True
            assert self.current_layer is not None
            cur_layer = self.current_layer
            assert (
                self.inside_context is False
            ), "Synchronization function was called without offloading context being entered."

            def hook(_):
                # offload_synchronizer.finish_part_of_bwd needs
                # to be called after every backward pass - there may be
                # more than one in pipeline parallelism.
                torch.autograd.variable.Variable._execution_engine.queue_callback(
                    offload_synchronizer.finish_part_of_bwd
                )
                offload_synchronizer.bwd_step(cur_layer)

            tensor.grad_fn.register_prehook(hook)
            return tensor

    cpu_offload_context = _CpuOffloadContext()

    if manual_synchronization:
        return (
            cpu_offload_context,
            cpu_offload_context.synchronization_function,
            offload_synchronizer,
        )
    return (
        cpu_offload_context,
        cpu_offload_context.synchronization_function,
    )
