# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functionality for CPU offloading of tensors saved for backward pass."""
from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch

from .float8_tensor import Float8Tensor

__all__ = ["get_cpu_offload_context"]

CPUOffloadEnabled = False


class CpuOffloadSavedTensorHook:
    """Contex-manager that executes a pair of pack/unpack hooks for saved tensors.

    In this context, the ``on_save_for_backward`` method will be called every time
    a tensor is saved for backward (this includes intermediary results saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation).

    The ``on_get_saved_tensors`` method will be called when the backward function
    of this op attempts to retrieve the saved tensor from context (this includes
    :func: `torch.Tensor.backward()` or :func: `torch.autograd.grad()`. It takes the
    as input the return value of the ``on_save_for_backward``, and is meant to return
    an identical copy of the tensor being saved by ``on_save_for_backward`` in terms of
    size, device and element values.

    Example:

        >>> import torch
        >>> from typing import Any
        >>>
        >>> class DummyHook(CpuOffloadSavedTensorHook):
        ...
        ...     def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        ...         logging.info("On save", tensor)
        ...         return (tensor,)
        ...
        ...     def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        ...         logging.info("On get", saved_state)
        ...         tensor, = saved_state
        ...         return tensor
        ...
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with DummyHook():
        ...     y = a * b
        ...
        On save tensor([1., 1., 1., 1., 1.], requires_grad=True)
        On save tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        On get (tensor([1., 1., 1., 1., 1.], requires_grad=True),)
        On get (tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>),)

    """

    def __init__(self) -> None:
        self.inside_context = False

    def __enter__(self):
        global CPUOffloadEnabled
        CPUOffloadEnabled = True

        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        global CPUOffloadEnabled
        CPUOffloadEnabled = False

        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """On save for backward."""
        raise NotImplementedError(
            "`on_save_for_backward: Callable[[torch.Tensor], Any]`"
            "is not implemented in CpuOffloadHook class. Inherit "
            "this class and implement your custom hooks"
        )

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """On get saved tensor."""
        raise NotImplementedError(
            "`on_get_saved_tensors: Callable[[Any], torch.Tensor]`"
            "is not implemented in CpuOffloadHook class. Inherit "
            "this class and implement your custom hooks"
        )


class CpuOffloadHookWithOffloadHandler(CpuOffloadSavedTensorHook):
    """Context-manager that offloads/recovers tensors through an offload hander.

    The hook just offloads/recovers the tensor object to the handler through `tensor_push`
    and `tensor_pop` interface. How the offload-handler manages the offloading, recovering
    or prefetching timing is transparent to this hook.
    """

    def __init__(
        self,
        offload_handler: OffloadHandler,
        handler_extra_kwargs: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> None:
        if handler_extra_kwargs is None:
            handler_extra_kwargs = {}
        self.debug: bool = debug
        self.offload_handler: OffloadHandler = offload_handler
        self.handler_extra_kwargs: Dict[str, Any] = handler_extra_kwargs
        super().__init__()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        retrieve_identifier = self.offload_handler.tensor_push(tensor, **self.handler_extra_kwargs)
        return retrieve_identifier

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        tensor = self.offload_handler.tensor_pop(saved_state, **self.handler_extra_kwargs)
        return tensor


class OffloadHandler:
    """A base class for CPU offload-handler."""

    def __init__(self) -> None:
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        """Tensor push."""
        raise NotImplementedError(
            "`tensor_push is not implented in OffloadHandler class. "
            "Inherit this class and implement your custom tensor_push."
        )

    def tensor_pop(self, tensor_tag: Any, **kwargs):
        """Tensor pop."""
        raise NotImplementedError(
            "`tensor_pop is not implented in OffloadHandler class. "
            "Inherit this class and implement your custom tensor_pop."
        )


class GroupCommitFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output, None


group_prefetch_offload_commit = GroupCommitFunction.apply


class SynchronizedGroupOffloadHandler(OffloadHandler):
    """Offload Handler that offloads/reloads in a synchronized way.
    The device-to-host and host-to-device copying happen in the same stream
    as the computation kernels, thus the copying will block computation.
    """

    def __init__(
        self, num_offload_group, tensor_need_offloading_checker=(lambda _: True), debug=False
    ) -> None:
        super().__init__()

        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker
        self.debug = debug

        self.groupid_reset()

    def groupid_reset(self):
        """Groupid reset."""
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies
        # the tensor back to gpu and deletes the cpu tensor.
        # These will increment whenever `group_commit()` is invoked
        self.current_group, self.tensor_count_current_group = (0, 0)
        self.torch_tensor_count = 0
        self.tensor_tag_to_state = {}

    def on_group_commit_forward(self):
        """On group commit forward."""
        # finishing up with updating current group and tensor count
        self.current_group += 1  # increment
        self.tensor_count_current_group = 0  # reset

    def on_group_commit_backward(self):
        """On group commit backward."""
        self.current_group -= 1
        assert self.current_group >= 0

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

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        """Tensor push."""
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        self.tensor_count_current_group += 1
        assert tensor_tag not in self.tensor_tag_to_state
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(
            tensor
        ):
            state = SynchronizedGroupOffloadHandler.offload(tensor)
            self.tensor_tag_to_state[tensor_tag] = state
        else:
            # will be offloaded together after group commit
            self.tensor_tag_to_state[tensor_tag] = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        assert tensor_tag in self.tensor_tag_to_state
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            tensor = SynchronizedGroupOffloadHandler.reload(state)
        else:
            tensor = state
        return tensor


class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):
    """Compared to synchronize, this uses more memory because of the buffer but
    achieves better performance due to the overlapping. D2h and h2d copying are
    completely hidden behind computation if computation time of a layer is longer
    than host-device communication time. Bulk offloading with delay and bulk reloading
    with prefetch are implemented."""

    def __init__(
        self,
        num_offload_group,  # must be <= actual number of groups (number of commits)
        num_prefetch_group=1,
        tensor_need_offloading_checker=(lambda t: True),
        debug=False,
    ) -> None:
        super().__init__(
            num_offload_group=num_offload_group,
            tensor_need_offloading_checker=tensor_need_offloading_checker,
            debug=debug,
        )
        self.num_prefetch_group = num_prefetch_group

        # prepare for tensor buffer
        self.tensor_id_to_tensor_buf_double_bufs = []
        for _ in range(2):
            self.tensor_id_to_tensor_buf_double_bufs.append({})

        # allocate streams and events for synchronization
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()
        self.h2d_finish_events = []
        self.compute_stream_bwd_start_events = []
        for _ in range(self.num_offload_group):
            self.h2d_finish_events.append(torch.cuda.Event())
            self.compute_stream_bwd_start_events.append(torch.cuda.Event())
        self.d2h_final_event = torch.cuda.Event()

    def get_tensor_buf_for_offloaded_tensor(self, tensor, tensor_tag):
        """Get tensor buffer for offloaded tensor."""
        group_id, tensor_id = tensor_tag
        # obtain ping-pong buffer
        id_buf_map = self.tensor_id_to_tensor_buf_double_bufs[(group_id % 2)]

        if not tensor_id in id_buf_map:
            allocate_new_buf = True
        else:
            tensor_buf = id_buf_map[tensor_id]
            allocate_new_buf = (
                tensor_buf.size() != tensor.size() or tensor_buf.dtype != tensor.dtype
            )

        if allocate_new_buf:
            # supposed to only execute once
            fp8_offload = isinstance(tensor, Float8Tensor)
            buffer = torch.empty(
                tensor.size(),
                dtype=torch.uint8 if fp8_offload else tensor.dtype,
                layout=tensor.layout,
                device=tensor.device,
            )

            if isinstance(tensor, Float8Tensor):
                id_buf_map[tensor_id] = Float8Tensor.make_like(tensor, data=buffer)
            else:
                id_buf_map[tensor_id] = buffer

        return id_buf_map[tensor_id]

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:

        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )

        if not torch_stray_tensor:
            # obtain a unique tensor tag
            tensor_tag = (self.current_group, self.tensor_count_current_group)
            self.tensor_count_current_group += 1
            assert tensor_tag not in self.tensor_tag_to_state

            if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(
                tensor
            ):
                # first copy the tensor to tensorbuf,
                # so that the original tensor will not be deleted
                tensor_buf = self.get_tensor_buf_for_offloaded_tensor(tensor, tensor_tag)
                tensor_buf.copy_(tensor)
                if hasattr(tensor, "weight_offloading"):
                    tensor_buf.weight_offloading = True
                if hasattr(tensor, "activation_offloading"):
                    tensor_buf.activation_offloading = True
                # Here we just save it, and at commit, bulk_offload_group will handle it
                self.tensor_tag_to_state[tensor_tag] = tensor_buf
            else:
                self.tensor_tag_to_state[tensor_tag] = tensor
        else:
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self.tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        assert tensor_tag in self.tensor_tag_to_state
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        # the tensor should have been copied back in on_group_commit_backward()
        # which invokes bulk_reload_group.
        assert not isinstance(tensor, tuple)
        return tensor

    def bulk_offload_group(self, group_to_offload):
        """Bulk offload group."""
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload:
                    assert not isinstance(state, tuple)
                    tensor_on_device = state

                    # if offload, return the reference to cpu copy
                    if self.tensor_need_offloading_checker(tensor_on_device):
                        if hasattr(tensor_on_device, "weight_offloading"):
                            delattr(tensor_on_device, "weight_offloading")
                        if hasattr(tensor_on_device, "activation_offloading"):
                            delattr(tensor_on_device, "activation_offloading")
                        state = SynchronizedGroupOffloadHandler.offload(tensor_on_device)
                        self.tensor_tag_to_state[tensor_tag] = state

    def synchronize_on_group_commit_forward(self, current_group):
        """Synchronize on group commit forward."""
        # the host should wait for the copying of previous group
        # to avoid overwriting buffer
        previous_group = current_group - 1
        if previous_group < self.num_offload_group:
            torch.cuda.synchronize()
            # TODO (guyueh): this part is originally designed to reduce the peak memory usage. # pylint: disable=fixme
            # however, uncommenting this part will cause illegal access, have not figured out why.

            if previous_group + 2 >= self.num_offload_group:
                # this buffer is no longer required
                self.tensor_id_to_tensor_buf_double_bufs[(previous_group % 2)] = {}

        # the copying of this group should wait for the computation stream event
        if current_group < self.num_offload_group:
            # perform bulk offloading
            self.bulk_offload_group(current_group)
            if current_group == self.num_offload_group - 1:
                self.d2h_stream.record_event(self.d2h_final_event)

    def on_group_commit_forward(self):
        """This function will cause host device synchronization"""
        # handle synchronization events
        self.synchronize_on_group_commit_forward(self.current_group)

        # during forward, the next_group_to_fetch always points to the min of
        # the last commited group, and the last offloaded group
        self.next_group_to_fetch = min(self.current_group, self.num_offload_group - 1)

        super().on_group_commit_forward()

    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        assert group_to_reload < self.num_offload_group
        if group_to_reload == self.num_offload_group - 1:
            self.h2d_stream.wait_event(self.d2h_final_event)
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_to_reload:
                    if isinstance(state, tuple):
                        recovered_tensor = SynchronizedGroupOffloadHandler.reload(state)
                        self.tensor_tag_to_state[tensor_label] = recovered_tensor

    def on_group_commit_backward(self):
        # first decrement the current group.
        # after last commit in forward, the group will +1; in backward it -1.
        # Finally it should be decremented to 0.
        self.current_group -= 1
        assert self.current_group >= 0

        # decide the range of group to prefetch
        should_prefetch_until_group = self.current_group - self.num_prefetch_group
        should_prefetch_until_group = max(should_prefetch_until_group, 0)

        # do prefetch
        for group_num_to_prefetch in range(
            self.next_group_to_fetch, should_prefetch_until_group - 1, -1
        ):
            # record the event in the compute stream, for h2d to wait
            torch.cuda.current_stream().record_event(
                self.compute_stream_bwd_start_events[group_num_to_prefetch]
            )

            # start of h2d should wait for the compute and the d2h
            self.h2d_stream.wait_event(self.compute_stream_bwd_start_events[group_num_to_prefetch])

            # recover tensors (copy back from host)
            self.bulk_reload_group(group_num_to_prefetch)

            # record an event for the backward of this layer to wait
            self.h2d_stream.record_event(self.h2d_finish_events[group_num_to_prefetch])

        # always is set to -1 at the end of the backward
        self.next_group_to_fetch = min(self.num_offload_group - 1, should_prefetch_until_group - 1)

        # wait for the current group
        if self.current_group < self.num_offload_group:
            torch.cuda.current_stream().wait_event(self.h2d_finish_events[self.current_group])


def get_cpu_offload_context(
    enabled: bool = False,
    num_layers: int = 1,
    offload_activations: bool = True,
    offload_weights: bool = True,
):
    """
    This function returns the CPU Offload context and the synchronizer function that needs to be
    used after every transformer layer. Returns `nullcontext()` if offloading is not enabled.

    Usage:

    .. code-block:: python

        cpu_offload_context, cpu_offload_synchronizer = get_cpu_offload_context(enabled=True)

        with cpu_offload_context:
            te_layer.forward(inp_tensor)
        cpu_offload_synchronizer()

    Parameters
    ----------
    enabled: bool, default = `False`
             When set to True, CPU Offloading functionality is enabled.
    num_layers: int, default = 1
                Determines the number of transformer layers
                you want to offload activations/weights for.
    offload_activations: bool, default = `True`
                         When set to `True`, offloads the activations for the TE layer.
    offload_weights: bool, default = `True`
                     When set to `True`, offloads the weights for the TE layer.

    """

    def tensor_need_offloading_checker_activations(tensor):
        return hasattr(tensor, "activation_offloading")

    # This includes the Gradient Accumulation Buffer
    def tensor_need_offloading_checker_weights(tensor):
        return hasattr(tensor, "weight_offloading")

    def tensor_need_offloading_checker_all(tensor):
        return hasattr(tensor, "activation_offloading") or hasattr(tensor, "weight_offloading")

    if offload_activations and offload_weights:
        tensor_need_offloading_checker = tensor_need_offloading_checker_all
    elif offload_activations:
        tensor_need_offloading_checker = tensor_need_offloading_checker_activations
    elif offload_weights:
        tensor_need_offloading_checker = tensor_need_offloading_checker_weights
    else:
        raise ValueError(
            "CPU Offloading is enabled while it is not "
            "mentioned what to offload (weights/activations)"
        )

    cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
        num_offload_group=num_layers,
        num_prefetch_group=1,
        tensor_need_offloading_checker=tensor_need_offloading_checker,
    )

    def group_prefetch_offload_commit_async(tensor):
        return group_prefetch_offload_commit(tensor, cpu_offload_handler)

    if enabled:
        return (
            CpuOffloadHookWithOffloadHandler(offload_handler=cpu_offload_handler),
            group_prefetch_offload_commit_async,
        )
    return nullcontext(), group_prefetch_offload_commit_async
