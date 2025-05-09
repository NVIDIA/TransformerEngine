# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functionality for CPU offloading of tensors saved for backward pass."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.autograd.graph import saved_tensors_hooks
from typing import Any, Optional
import contextlib
__all__ = ["offload", "_manual_reload", "CPUOffload"]


def is_cpu_offload_enabled():
    return CURRENT_CPU_OFFLOAD_HANDLER is not None

def offload(*tensors: torch.Tensor | torch.nn.Parameter | Any, manual_reload: bool = False) -> None:
    """ 
        The provided tensors are offloaded to the CPU, with the CPU copy operation synchronized with the main PyTorch stream. 
        After invoking this function, the tensors should not be modified in place. The transition to a CPU tensor occurs within 
        save_for_backward via a PyTorch hook.

        This function can be applied multiple times to the same tensor object; however, only a single copy on the CPU is created, 
        ensuring proper synchronization.

        By default, the tensor is reloaded during restore_from_saved via a PyTorch hook.
        If manual_reload is set to True, the tensor is switched to CPU tensor in prepare_for_saving - not inside the hook.
        Then cpu tensors are saved for backward and restored in restore_from_saved. This allows to delay the reloading synchronization
        to the moment when these tensors are actually needed. To reload the tensor, use the _manual_reload function.
    """
    def _offload(tensor: torch.Tensor):
        assert CURRENT_CPU_OFFLOAD_HANDLER is not None, "offload() should be called inside CPUOffload wrapper"
        CURRENT_CPU_OFFLOAD_HANDLER.offload(tensor)
        tensor.offload_handler = CURRENT_CPU_OFFLOAD_HANDLER
        tensor.activation_offloading = True
        tensor.manual_reload = manual_reload

        return tensor

    for tensor in tensors:
        if tensor is None:
            continue
        if isinstance(tensor, torch.Tensor):
            _offload(tensor)
        else:
            data_tensors = tensor.get_data_tensors()
            for data_tensor in data_tensors:
                if data_tensor is not None:
                    _offload(data_tensor)

def _manual_reload(cpu_tensor: torch.Tensor | Any) -> torch.Tensor | Any:
    """
        This function consumes the CPU copy of the tensor and returns the reloaded GPU copy. 
        It is synchronized with the main PyTorch stream. 
        The primary advantage of this function is that it allows the reloading synchronization 
        to be delayed until the tensors are actually needed. For example:

        ```python
        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                offload(x, manual_reload=True)

                # ...
                y = ...
                # ...

                save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x = ctx.saved_tensors[0]  # x is a CPU tensor
                # Even if the reload is not finished, the following operation can start.

                # ...
                # some long operation
                # ...

                x = _manual_reload(x)  # x is a GPU tensor, reload is synchronized with the main stream

                # ...

                return grad_output
        ```

        The number of times `offload` is used with `manual_reload` for each tensor 
        object should match the number of invocations of the `_manual_reload` function. 
        This is because the CPU offloading maintains a reference count, 
        which is decremented in the `_manual_reload` function.
    """
    if cpu_tensor is None:
        return None
    if isinstance(cpu_tensor, torch.Tensor):
        return cpu_tensor.offload_handler.reload(cpu_tensor)
    else:
        data_tensors = cpu_tensor.get_data_tensors()
        new_data_tensors = []
        for data_tensor in data_tensors:
            new_data_tensors.append(_manual_reload(data_tensor))
        return cpu_tensor.set_data_tensors(*new_data_tensors)

@dataclass
class _OffloadTensorData:
    """ 
        Data of one offloaded tensor. If one tensor is offloaded multiple times,
        then all the data is stored in the same _OffloadTensorData object.
    """
    # Pointer to the gpu tensor need to be in Python until
    # the _copy function in d2h stream is finished.
    original_gpu_tensor: torch.Tensor 
    original_gpu_tensor_id: int
    cpu_tensor: torch.Tensor
    cpu_tensor_id: int
    reloaded_gpu_tensor: Optional[torch.Tensor] = None
    # Number of times the tensor object was packed.
    # Increased when tensor in in pack_hook and decreased when tensor is reloaded.
    # When refcount is 0, the _OffloadMeta object is deleted.
    refcount: int = 0

class _CPUOffloadHandler:
    """
        This is internal class handling the CPU offloading with simple and intuitive interface.
        Example of the usage of the API:

        __init__()

        start_layer_fwd()
        t1_cpu = offload(t1)
        tensor_packed(t1_cpu)
        end_layer_fwd()

        start_layer_fwd()
        t2_cpu = offload(t2)
        tensor_packed(t2_cpu)
        end_layer_fwd()

        after_fwd_before_bwd()

        start_layer_bwd()
        t1_gpu = _manual_reload(t1_cpu)
        end_layer_bwd()

        start_layer_bwd()
        t2_gpu = _manual_reload(t2_cpu)
        end_layer_bwd()
    """
    def __init__(self):
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

        # this dicts contain the same values, but with different keys
        self._by_gpu: dict[int, _OffloadTensorData] = {} # key: id(gpu tensor before offload)
        self._by_cpu: dict[int, _OffloadTensorData] = {} # key: id(cpu copy of the tensor)

        # sanity check
        self.in_layer_fwd = False
        self.in_layer_bwd = False

    def start_layer_fwd(self):
        """
            Invoked before the new offloaded layer is started.
            
            Behavior:
            - Wait for the offloading of the previous layer to finish,
            - Free the memory of the gpu activation tensors of the previous layer
                  by deleting the references to them.
        """
        self.in_layer_fwd = True
        main_stream = torch.cuda.current_stream()
        main_stream.wait_stream(self.d2h_stream)
        
        for x in self._by_gpu.values():
            x.original_gpu_tensor = None

    def end_layer_fwd(self):
        """
            Invoked after the offloaded layer is finished.
        """
        self.in_layer_fwd = False

    def after_fwd_before_bwd(self):
        """
            Invoked once between forward and backward pass.

            Behavior:
            - Wait for the remaining offloading to finish,
            - Free the memory of the gpu activation tensors of the previous layer
                  by deleting the references to them,
            - synchronize the stream for reloading the tensors - to prevent reloading during the forward pass.
        """

        assert self.in_layer_fwd == False
        assert self.in_layer_bwd == False
        
        main_stream = torch.cuda.current_stream()
        main_stream.wait_stream(self.d2h_stream)
        self.h2d_stream.wait_stream(main_stream)
        
        for meta in self._by_gpu.values():
            meta.original_gpu_tensor = None

    def start_layer_bwd(self):
        """
            Invoked when the backward pass of offloaded layer is started.

            Behavior:
            - Record the event for the start of the backward pass of the layer.
        """
        self.in_layer_bwd = True
        self.event_start_bwd_main = torch.cuda.Event()
        self.event_start_bwd_main.record(torch.cuda.current_stream())

    def end_layer_bwd(self):
        """
            Invoked when the backward pass of offloaded layer is finished.

            Behavior:
            - Stream h2d(reload) waits for the beginning of the backward pass of the layer on main.
                  This prevents starting the reload of layer n + 2 
                  before the forward pass of layer n is finished.
        """
        self.in_layer_bwd = False

        self.event_start_bwd_main.wait(self.h2d_stream)

    def offload(self, gpu_tensor: torch.Tensor) -> torch.Tensor:
        """
            Offload the tensor to the CPU. It returns the CPU copy of the tensor.
            Can be called between start_layer_fwd and end_layer_fwd.

            Behavior:
            - If the tensor is already offloaded, increment the reference count.
            - Synchronize the d2h stream with main - copy should start after the tensor is computed.
            - Otherwise, create a new empty tensor on the CPU and start async copy from GPU in the d2h stream.
            - Return the CPU copy of the tensor.
        """
        assert self.in_layer_fwd

        if id(gpu_tensor) in self._by_gpu:
            meta = self._by_gpu[id(gpu_tensor)]
            return meta.cpu_tensor
        else:
            cpu_tensor = torch.empty_like(gpu_tensor, device="cpu", pin_memory=True)
            self._by_cpu[id(cpu_tensor)] = _OffloadTensorData(gpu_tensor, id(gpu_tensor), cpu_tensor, id(cpu_tensor))
            self._by_gpu[id(gpu_tensor)] = self._by_cpu[id(cpu_tensor)]
            
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.d2h_stream):
                cpu_tensor.copy_(gpu_tensor, non_blocking=True)
                
            return cpu_tensor

    def reload(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """
            Reload the tensor from the CPU to the GPU. It returns the GPU copy of the tensor
            and is synchronized with the main PyTorch stream.
            Can be called between start_layer_bwd and end_layer_bwd.

            Behavior:
            - Decrement the reference count.
            - If the reference count is 0, delete the _OffloadTensorData object.
            - Create a new empty tensor on the GPU and start async copy from CPU in the h2d stream.
            - Synchronize the h2d stream with main - returned tensor can be used on the main stream.
            - Return the GPU copy of the tensor.
        """
        assert self.in_layer_bwd


        meta = self._by_cpu[id(cpu_tensor)]
        meta.refcount -= 1
        if meta.refcount == 0:
            self._by_cpu.pop(meta.cpu_tensor_id)
            self._by_gpu.pop(meta.original_gpu_tensor_id)

        meta.reloaded_gpu_tensor = torch.empty_like(cpu_tensor, device="cuda")
        with torch.cuda.stream(self.h2d_stream):
            meta.reloaded_gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        self.h2d_stream.wait_stream(torch.cuda.current_stream())
            
        return meta.reloaded_gpu_tensor

    def tensor_packed(self, cpu_tensor: torch.Tensor):
        """
            Called when the tensor is packed. We use the fact that number of times the tensor was packed
            is equal to the number of times the tensor is reloaded.

            Behavior:
            - Increment the reference count.
        """
        self._by_cpu[id(cpu_tensor)].refcount += 1

class SwitchOffloadHandler(contextlib.ContextDecorator):
    """
        Context manager to change the current CPU offload handler.
    """
    def __init__(self, handler: _CPUOffloadHandler):
        self.handler = handler

    def __enter__(self):
        global CURRENT_CPU_OFFLOAD_HANDLER
        self.previous_cls = CURRENT_CPU_OFFLOAD_HANDLER
        CURRENT_CPU_OFFLOAD_HANDLER = self.handler

    def __exit__(self, *args):
        global CURRENT_CPU_OFFLOAD_HANDLER
        CURRENT_CPU_OFFLOAD_HANDLER = self.previous_cls
        

CURRENT_CPU_OFFLOAD_HANDLER = None

class CPUOffload:
    """
        CPU Offloading allows to offload the activation of the layer during the forward pass,
        and reload it during the backward pass. To use it, create object of the CPU Offload class, 
        for example `cpu_offload = CPUOffload()`. Then wrap computation with `y = cpu_offload(layer, *args, **kwargs)`,
        to offload the activation of the layer. Notice that you can mix offloaded layers and non-offloaded layers.
        The function `cpu_offload.sync_before_bwd()` should be called between the forward and backward.

        Synchronization:
        ----------------
        The offloading is performed asynchronously with respect to the main stream.
        The activation offloading of one computation needs to finish before offloading of the next computation can start.
        This ensures that only one copy of the activation memory of the offloaded layer is present in the memory during forward pass.
        During the backward pass, the activation of the offloaded layer n is reloaded since the start of the backward pass of the layer n + 2,
        thus 2 portions of offloaded activations can be present in the memory at the same time.

        We encourage to profile the code using Nsight Systems to ensure full overlap of the offloading and computation.

        Example:
        --------
        ```python
        # ...
        cpu_offload = CPUOffload()

        layer1 = cpu_offload(layer1)
        layer3 = cpu_offload(layer3)

        x2 = layer1(x1)
        x3 = layer2(x2)
        y = layer3(x3)

        cpu_offload.sync_before_bwd()

        y.sum().backward()

        # ...
        
        ```
        
    """
    def __init__(self):
        self.handler = _CPUOffloadHandler()
        handler = self.handler
        self.phase = "forward"

        class _LayerHooksBefore(torch.autograd.function.Function):
            """
                This autograd function is used to call CPUOffload api calls both in forward and backward pass.
            """
            @staticmethod
            def forward(ctx, *args):
                handler.start_layer_fwd()
                self.phase = "forward"
                return args

            @staticmethod
            def backward(ctx, *args):
                handler.end_layer_bwd()
                if self.phase == "forward":
                    raise RuntimeError("sync_before_bwd should be called between forward and backward")
                return args
            
        self.layer_hooks_before = _LayerHooksBefore

        class _LayerHooksAfter(torch.autograd.Function):
            """
                This autograd function is used to call CPUOffload api calls both in forward and backward pass.
            """
            @staticmethod
            def forward(ctx, *args):
                handler.end_layer_fwd()
                self.phase = "forward"
                return args[0] if len(args) == 1 else args

            @staticmethod
            def backward(ctx, *args):
                handler.start_layer_bwd()
                if self.phase == "forward":
                    raise RuntimeError("sync_before_bwd should be called between forward and backward")
                return args[0] if len(args) == 1 else args

        self.layer_hooks_after = _LayerHooksAfter
    
    def _pack_hook(self, tensor: torch.Tensor) -> tuple[torch.Tensor, _CPUOffloadHandler, bool]:
        """
            If tensor needs to be offloaded - if it has activation_offloading attribute - offload it.

            Returns:
            - tensor: the offloaded or non-offloaded tensor.
            - handler: the offload handler or None - it is needed to restore the tensor in the backward pass.
            - manual_reload: whether the tensor should be reloaded during _unpack_hook or later manually.
        """

        if hasattr(tensor, "activation_offloading"):
            if tensor.device.type == "cpu":
                # If tensor was offloaded with manual_reload=True,
                # it is substituted with a CPU tensor inside prepare_for_saving.
                output = (tensor, CURRENT_CPU_OFFLOAD_HANDLER, True)
            else:  
                # If tensor was offloaded with manual_reload=False,
                # it is offloaded again - handler returns the same CPU tensor, so reloading does not happen twice.
                output = (self.handler.offload(tensor), CURRENT_CPU_OFFLOAD_HANDLER, False)
            
            self.handler.tensor_packed(output[0])
            return output
        else:
            return (tensor, None, False)
    
    def _unpack_hook(self, tensor_handler: tuple[torch.Tensor, _CPUOffloadHandler, bool]):
        """
            Unpacks the tensor and the offload handler.
        """
        tensor, handler, manual_reload = tensor_handler
        if handler is not None:
            tensor.offload_handler = handler
            if not manual_reload:
                return self.handler.reload(tensor)
            else:
                return tensor
        else:
            # standard, non-offloaded tensor
            return tensor

    def __call__(self, func):
        """
           Wraps the function, which activation is offloaded.

           Parameters
           ----------
           func : Callable
               The function, which activation is offloaded. All TE layers called by this function
               will be affected by the offloading. It does not impact non-TE layers.
        """
        def wrapper(*args, **kwargs):
            with saved_tensors_hooks(self._pack_hook, self._unpack_hook), \
                SwitchOffloadHandler(self.handler):
                args = self.layer_hooks_before.apply(*args)
                out = func(*args, **kwargs)
                out = self.layer_hooks_after.apply(out)
            return out
        return wrapper

    def sync_before_bwd(self):
        """
            This function should be called between the forward and backward pass to ensure proper synchronization.
            Not calling it will result in an error.
        """
        self.phase = "backward"
        self.handler.after_fwd_before_bwd()
