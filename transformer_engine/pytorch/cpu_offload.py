# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functionality for CPU offloading of tensors saved for backward pass."""

from __future__ import annotations
import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
import warnings
from typing import Any, Optional
import torch
from torch.autograd.graph import saved_tensors_hooks
from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensor
from transformer_engine.debug.pytorch.debug_state import TEDebugState

__all__ = ["get_cpu_offload_context", "mark_is_weight", "start_offload_if_offload_enabled"]

DEFAULT_MIN_TENSOR_SIZE_TO_OFFLOAD = 2**20  # 1mb
OFFLOAD_SYNCHRONIZER = None


def is_cpu_offload_enabled():
    """Returns True if CPU offload is enabled."""
    return OFFLOAD_SYNCHRONIZER is not None


def mark_is_weight(*tensors: torch.Tensor):
    """Marks tensors as weights to prevent them from being offloaded."""
    for tensor in tensors:
        if tensor is not None:
            setattr(tensor, "is_weight", True)


def start_offload_if_offload_enabled(*tensors: torch.Tensor, offload_base_tensor: bool = False):
    """Starts offload tensor. It should be used it tensors are fully computed and ready to be offloaded."""
    if not is_cpu_offload_enabled():
        return
    for tensor in tensors:
        if tensor is not None:
            OFFLOAD_SYNCHRONIZER.push_tensor(tensor, offload_base_tensor)  # type: ignore[attr-defined]


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
    def process_push_tensor(
        tensor: torch.Tensor, offload_base_tensor: bool = False
    ) -> torch.Tensor:
        """
        Public API call - for a single tensor, when this tensor is pushed to be offloaded.
        """
        if offload_base_tensor or hasattr(tensor, "offload_base_tensor"):
            # Tensor is marked as a view of a base tensor that needs to be offloaded,
            # we process this in _switch_to_base_tensors and _switch_to_views in the
            # tensor_group_process_before_offload and tensor_group_process_after_reload.
            setattr(tensor, "offload_base_tensor", True)
        else:
            # If tensor is not a view of offloaded base tensor,
            # we make it contiguous.
            # If non-contiguous tensor is offloaded, then .contiguous() needs to happen inside .copy_().
            # We call .copy_() on fully async memory stream, so we do not want to call .contiguous() there,
            # which may result in wait for a free SM needed to process .contiguous() call.
            # It's better to call .contiguous() on compute stream before it is offloaded.
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
        return tensor

    @staticmethod
    def tensor_group_process_before_offload(tensor_group: TensorGroup) -> tuple[TensorGroup, Any]:
        """
        Public API call - for a tensor group, just before offloading logic happens.

        aux is a dictionary that contains auxiliary data, needed to restore pre-offload state.
        """
        aux = {}
        tensor_group = TensorGroupProcessor._switch_to_base_tensors(aux, tensor_group)
        tensor_group = TensorGroupProcessor._deduplicate_tensors(aux, tensor_group)
        tensor_group = TensorGroupProcessor._make_contiguous(tensor_group)
        return tensor_group, aux

    @staticmethod
    def tensor_group_process_after_reload(tensor_group: TensorGroup):
        """
        Public API call - for a tensor group, just after reload logic happens.
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
        this will save unnecessary calls to .contiguous(). It is used for example in
        MultiHeadAttention for interleavedq, k, v tensors.
        """
        aux["views"] = []
        for tensor_id in range(  # pylint: disable=consider-using-enumerate
            len(tensor_group.tensor_list)
        ):
            tensor = tensor_group.tensor_list[tensor_id]
            if getattr(tensor, "offload_base_tensor", False):
                aux["views"].append((tensor.shape, tensor.stride(), tensor.storage_offset()))
                tensor = tensor._base
                assert tensor is not None
                tensor_group.tensor_list[tensor_id] = tensor
            else:
                aux["views"].append(None)
        return tensor_group

    @staticmethod
    def _make_contiguous(tensor_group: TensorGroup) -> TensorGroup:
        """
        Make tensors contiguous.
        """
        any_non_contiguous = False
        for tensor in tensor_group.tensor_list:
            if not tensor.is_contiguous():
                any_non_contiguous = True
                break
        if any_non_contiguous:
            warnings.warn(
                "Non-contiguous tensors are offloaded. Reloading will change memory layout to"
                " contiguous."
            )
            tensor_group.tensor_list = [tensor.contiguous() for tensor in tensor_group.tensor_list]
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
        for tensor_id in range(len(tensor_group.aux["original_tensor_ids"])):
            original_tensor_id = tensor_group.aux["original_tensor_ids"][tensor_id]
            new_tensor_list.append(tensor_group.tensor_list[original_tensor_id])
        tensor_group.tensor_list = new_tensor_list
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


class OffloadSynchronizer:
    """
    Class that synchronizes offloading and reloading of tensors.
    I decoupled the logic from the context manager for easier testing.
    """

    def __init__(
        self,
        num_layers: int,
        num_offloaded_layers: int | None = None,
        synchronization_dict: dict[int, tuple[bool, int, bool, int]] | None = None,
        min_tensor_size_to_offload: int = DEFAULT_MIN_TENSOR_SIZE_TO_OFFLOAD,
    ):
        self.offload_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.min_tensor_size_to_offload = min_tensor_size_to_offload

        # There are 3 types of tensor groups - for tensors before offload, cpu copies and tensors after reload.
        # fwd_gpu_tensor_groups of layer i are kept in memory until layer i is offloaded, to provide memory safety.
        self.fwd_gpu_tensor_groups: dict[int, TensorGroup] = defaultdict(TensorGroup)
        self.cpu_tensor_groups: dict[int, TensorGroup] = {}
        self.bwd_gpu_tensor_groups: dict[int, TensorGroup] = {}

        # Before bwd of layer i, remove tensor group of i-th layer from bwd_gpu_tensor_groups
        # and keep it in bwd_current_reloaded_tensor_group. It is alive until bwd of layer i is finished,
        # and is overriden by the next bwd or cleared by the next fwd.
        self.bwd_current_reloaded_tensor_group: TensorGroup | None = None

        self.current_fwd_layer = -1
        self.num_layers = num_layers

        # SYNCHRONIZATION HELPERS

        # Dictionaries of events when reload/offload of group i is finished.
        self.finish_offload_events: dict[int, torch.cuda.Event] = {}
        self.finish_reload_events: dict[int, torch.cuda.Event] = {}

        # Here the logic of when to offload/reload is stored.

        # map of layers to bool meaning if layer needs to be offloaded
        self.offload_layer_map: dict[int, bool] = {}
        # keys to dictionaries below are pairs of (is_forward: bool, num_layer: int)
        # (is_forward: bool, num_layer: int) -> list of layers that need to finish offload by this moment
        self.finish_offload_map: defaultdict[tuple[bool, int], list[int]] = defaultdict(list)
        # (is_forward: bool, num_layer: int) -> list of layers that need to start reload in this moment
        self.start_reload_map: defaultdict[tuple[bool, int], list[int]] = defaultdict(list)

        if num_offloaded_layers is not None:
            assert synchronization_dict is None
            self._default_offload_sync_init(num_offloaded_layers)
        else:
            self._process_offload_sync_dict(synchronization_dict)

    def _process_offload_sync_dict(
        self, synchronization_dict: dict[int, tuple[bool, int, bool, int]] | None = None
    ):
        """
        Process synchronization dictionary into self.offload_layer_map, self.finish_offload_map and self.start_reload_map.

        Convert dict: layer_id -> (offload_fwd, offload_num, reload_fwd, reload_num)
        into offload_layer_map, finish_offload_map and start_reload_map.
        """
        assert synchronization_dict is not None
        for layer_id in range(self.num_layers):
            if layer_id in synchronization_dict.keys():
                offload_fwd, offload_num, reload_fwd, reload_num = synchronization_dict[layer_id]

                self.offload_layer_map[layer_id] = True
                self.finish_offload_map[offload_fwd, offload_num].append(layer_id)
                self.start_reload_map[reload_fwd, reload_num].append(layer_id)
            else:
                self.offload_layer_map[layer_id] = False

    def _default_offload_sync_init(self, num_offloaded_layers: int):
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
                self.finish_offload_map[
                    True, self.num_layers - num_offloaded_layers + layer_id
                ].append(layer_id)
                self.start_reload_map[
                    False, self.num_layers - 1 - num_offloaded_layers + layer_id
                ].append(layer_id)
            else:
                self.offload_layer_map[layer_id] = False

    def fwd_step(self) -> int:
        """
        Invoked before each layer forward.
        """

        if self.current_fwd_layer in [-1, self.num_layers - 1]:
            # reset the offload synchronizer
            self.current_fwd_layer = -1

        if self.offload_layer_map.get(self.current_fwd_layer, False):
            # start offloading
            self.cpu_tensor_groups[self.current_fwd_layer] = self._offload_group(
                self.fwd_gpu_tensor_groups[self.current_fwd_layer]
            )
            self.finish_offload_events[self.current_fwd_layer] = torch.cuda.Event()
            self.finish_offload_events[self.current_fwd_layer].record(self.offload_stream)

        self.current_fwd_layer += 1
        self._start_reloads_finish_offloads(self.current_fwd_layer, True)
        return self.current_fwd_layer

    def bwd_step(self, layer_num: int):
        """
        Invoked before each layer backward.
        """

        # release memory of bwd previous layer
        self.bwd_current_reloaded_tensor_group = None

        self._start_reloads_finish_offloads(layer_num, False)

        # finish reload of current layer if necessary
        if self.offload_layer_map[layer_num]:
            assert layer_num in self.finish_reload_events, (
                f"Layer {layer_num} is not reloaded before it is needed in backward.               "
                "      Check if the synchronization dictionary is correct."
            )

            torch.cuda.current_stream().wait_event(self.finish_reload_events[layer_num])  # type: ignore[arg-type]
            self.bwd_current_reloaded_tensor_group = self.bwd_gpu_tensor_groups[layer_num]
            del self.bwd_gpu_tensor_groups[layer_num]
            del self.cpu_tensor_groups[layer_num]
            self.bwd_current_reloaded_tensor_group = (
                TensorGroupProcessor.tensor_group_process_after_reload(
                    self.bwd_current_reloaded_tensor_group
                )
            )

    def finish_part_of_bwd(self):
        """
        We need to release memory of backward - this call does that.
        It needs to be invoked after every backward pass - there may be
        more than one in pipeline parallelism.
        """
        self.bwd_current_reloaded_tensor_group = None

    def _start_reloads_finish_offloads(self, layer_num: int, forward: bool):
        """
        Start reloading layers and finish offloading layers.
        """

        # start reloading layers
        for layer in self.start_reload_map[forward, layer_num]:
            self.offload_stream.wait_stream(torch.cuda.current_stream())
            self.bwd_gpu_tensor_groups[layer] = self._reload_group(self.cpu_tensor_groups[layer])
            self.finish_reload_events[layer] = torch.cuda.Event()
            self.finish_reload_events[layer].record(self.offload_stream)

        # end offloading and release memory
        for layer in self.finish_offload_map[forward, layer_num]:
            torch.cuda.current_stream().wait_event(self.finish_offload_events[layer])  # type: ignore[arg-type]
            del self.fwd_gpu_tensor_groups[layer]
            del self.finish_offload_events[layer]

    def _reload_group(self, tensor_group: TensorGroup) -> TensorGroup:
        """
        Gets tensor group of tensors on CPU and initializes copy of tensors into GPU on self.offload_stream.
        Returns output tensor group of tensors on GPU, the event of the copy finish is recorded in self.finish_reload_events.
        """
        reloaded_tensor_group = TensorGroup()
        for tensor in tensor_group.tensor_list:
            # empty_like is defined also for QuantizedTensors.
            reloaded_tensor = torch.empty_like(tensor, device=torch.device("cuda"))
            with torch.cuda.stream(self.offload_stream):
                reloaded_tensor.copy_(tensor, non_blocking=True)
            reloaded_tensor_group.tensor_list.append(reloaded_tensor)
        reloaded_tensor_group.aux = tensor_group.aux
        return reloaded_tensor_group

    def _offload_group(self, tensor_group: TensorGroup) -> TensorGroup:
        """
        Gets tensor group of tensors on CPU and initializes copy of tensors into CPU on self.offload_stream.
        Returns output tensor group of tensors on CPU, the event of the copy finish is recorded in self.finish_offload_events.
        """
        tensor_group, aux = TensorGroupProcessor.tensor_group_process_before_offload(tensor_group)
        offloaded_tensor_group = TensorGroup()
        for tensor_id, tensor in enumerate(tensor_group.tensor_list):
            assert tensor.is_contiguous()
            # empty_like is defined also for QuantizedTensors
            offloaded_tensor = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
            self.offload_stream.wait_event(tensor_group.events[tensor_id])  # type: ignore[arg-type]

            with torch.cuda.stream(self.offload_stream):
                offloaded_tensor.copy_(tensor, non_blocking=True)
            offloaded_tensor_group.tensor_list.append(offloaded_tensor)
        offloaded_tensor_group.aux = aux
        return offloaded_tensor_group

    def _check_if_offload(self, t: torch.Tensor) -> bool:
        """
        Check if tensor needs to be offloaded.
        """
        return (
            not isinstance(t, torch.nn.Parameter)
            and not getattr(t, "is_weight", False)
            and t.numel() >= self.min_tensor_size_to_offload
            and not isinstance(t, torch._subclasses.FakeTensor)
        )

    def push_tensor(
        self, tensor: torch.Tensor, offload_base_tensor: bool = False
    ) -> int | torch.Tensor:
        """
        Push tensor to a group of tensors that will be offloaded.
        If offload_base_tensor is True, then tensor is a view of a base tensor that will be offloaded.
        This is used for example in MultiHeadAttention for interleaved q, k, v tensors.

        Returns the same tensor if it is not offloaded, otherwise returns the index of the tensor in the group.

        Offloading logic happens at the end of the forward pass of current layer.
        """
        if not self.offload_layer_map[self.current_fwd_layer]:
            return tensor

        if self._check_if_offload(tensor):
            tensor = TensorGroupProcessor.process_push_tensor(tensor, offload_base_tensor)

            current_tensor_group = self.fwd_gpu_tensor_groups[self.current_fwd_layer]
            current_tensor_group.tensor_list.append(tensor)

            # The group is processed and offloaded at the end of the forward pass of current layer.
            # To enable offloading of tensors faster we use self.offload_stream and record
            # the events when the tensors are ready to be offloaded.
            # It means that we do not need to wait to the end of current layer to start offloading.
            current_tensor_group.events.append(torch.cuda.Event())
            current_tensor_group.events[-1].record(torch.cuda.current_stream())
            return len(current_tensor_group.tensor_list) - 1
        return tensor

    def pop_tensor(self, tensor_or_tensor_id: torch.Tensor | int) -> torch.Tensor:
        """
        Pop tensor from a reloaded tensor group.

        If tensor_or_tensor_id is tensor, it means that the tensor is not offloaded. Otherwise
        the reloaded tensor is returned.
        """
        if isinstance(tensor_or_tensor_id, torch.Tensor):
            return tensor_or_tensor_id
        assert self.bwd_current_reloaded_tensor_group is not None
        return self.bwd_current_reloaded_tensor_group.tensor_list[tensor_or_tensor_id]

    def get_offloaded_total_size_mb(self) -> float:
        """
        Get total size of offloaded tensors in MB, used only for testing.
        """
        total_size = 0
        for tensor_group in self.cpu_tensor_groups.values():
            for tensor in tensor_group.tensor_list:
                total_size += tensor.numel() * tensor.element_size() / 1024 / 1024
        return total_size


def get_cpu_offload_context(
    enabled: bool = False,
    num_layers: Optional[int] = 1,
    model_layers: int = 1,
    offload_activations: bool = True,
    offload_weights: bool = False,
    double_buffering: bool = False,  # pylint: disable=unused-argument
    synchronization_dict: dict[int, tuple[bool, int, bool, int]] | None = None,
    min_tensor_size_to_offload: int = DEFAULT_MIN_TENSOR_SIZE_TO_OFFLOAD,
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

    Synchronization:

    There are 2 ways of synchronization:
    - 1. By providing num_layers. In this case layers are offloaded/reloaded in optimal order to ensure that
         activation memory usage is equal to `(num_layers - num_offloaded_layers) * T`, where `T` is the memory footprint of a layer.
    - 2. By providing synchronization_dict. In this case layers are offloaded/reloaded in the order provided in the dictionary.
         This dictionary have entries `i: (offload_fwd, offload_num, reload_fwd, reload_num)`, which means that
         layer `i` will finish offload when `offload_num` layers begins its forward/backward pass (depending on `offload_fwd` being True/False respectively).
         Layer `i` will start reload when `reload_num` layers starts its forward/backward pass (depending on `reload_fwd` being True/False respectively).
         In this way offloading will work with pipeline parallelism - when backward are interleaved with forward, assuming that synchronization dictionary is correct.


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
    synchronization_dict: dict[int, tuple[bool, int, bool, int]] | None = None
                If None, the number of offloaded layers is used to initialize the synchronization dictionary.
                Dictionary of layer ids to tuples of (offload_fwd, offload_num, reload_fwd, reload_num).
                Layer `i` is offloaded if and only if it is the key of the dictionary.
                Layer `i` will finish offload when `offload_num` layers begins its forward/backward pass (depending on `offload_fwd` being True/False respectively).
                Layer `i` will start reload when `reload_num` layers starts its forward/backward pass (depending on `reload_fwd` being True/False respectively).
    min_tensor_size_to_offload: int, default = 2 ** 20
                Minimum number of elements in a tensor to be offloaded.

    """
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
            return contextlib.nullcontext(), lambda x: x

    if TEDebugState.debug_enabled:
        raise RuntimeError("CPU offload is not supported in debug mode.")

    if num_layers is not None:
        assert (
            num_layers <= model_layers - 1
        ), "Cannot offload all layers with synchronization_dict=None - last layer is not offloaded."

    offload_synchronizer = OffloadSynchronizer(
        model_layers,
        num_layers if synchronization_dict is None else None,
        synchronization_dict,
        min_tensor_size_to_offload,
    )

    class _CpuOffloadContext(contextlib.ContextDecorator):
        def __init__(self):
            self.current_layer = None
            self.previous_offload_synchronizer = None
            self.offload_synchronizer = offload_synchronizer

        def __enter__(self):
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

        def synchronization_function(self, tensor):
            """
            This function is used to catch the backward pass of the model.
            """
            assert tensor.requires_grad is True
            assert self.current_layer is not None
            cur_layer = self.current_layer

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

    if enabled:
        return (
            cpu_offload_context,
            cpu_offload_context.synchronization_function,
        )
    return contextlib.nullcontext(), lambda x: x
