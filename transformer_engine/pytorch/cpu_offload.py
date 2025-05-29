# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functionality for CPU offloading of tensors saved for backward pass."""

from __future__ import annotations
import contextlib

import torch
from torch.autograd.graph import saved_tensors_hooks

__all__ = [
    "is_cpu_offload_enabled", "mark_is_weight", "mark_can_start_offload",
    "CPUOffload", "get_cpu_offload_context"
]

MIN_TENSOR_SIZE_TO_OFFLOAD = 1000
CURRENT_CPU_OFFLOAD_HANDLER = None

def is_cpu_offload_enabled():
    return CURRENT_CPU_OFFLOAD_HANDLER is not None

def mark_is_weight(*tensors: torch.Tensor):
    for tensor in tensors:
        if tensor is not None:
            tensor.is_weight = True

def mark_can_start_offload(*gpu_tensors: torch.Tensor):
    """
        Marks the moment the tensor can be offloaded. The tensor passed to the offload function,
        must be the same object as the one passed to the mark_can_start_offload function,
        to not delay the offloading.
    """
    if CURRENT_CPU_OFFLOAD_HANDLER is not None:
        for gpu_tensor in gpu_tensors:
            if gpu_tensor is not None:
                CURRENT_CPU_OFFLOAD_HANDLER.mark_can_start_offload(gpu_tensor)


class _StreamedOffloader:
    """
        _StreamedOffloader represents one stream used to offload the tensors to the CPU.
        It provides easy to use interface for offloading and synchronization.
    """
    def __init__(self): 
        self.stream = torch.cuda.Stream()
        self.cpu_tensors = {}
        self.gpu_tensors = {}
    
    def mark_can_start_offload(self, gpu_tensor: torch.Tensor):
        """
            If gpu_tensor is ready to offload on current stream, record the event.
            We may be able to start the offloading earlier than in the save_for_backward.
        """
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        gpu_tensor._offload_event = event
    
    def offload(self, gpu_tensor: torch.Tensor) -> torch.Tensor:
        """
            Offload the tensor from GPU to CPU.  Return the CPU copy of the tensor.
        """
        if id(gpu_tensor) in self.gpu_tensors.keys():
            # One tensor can be an argument to the offload function multiple times,
            # but only one cpu copy of the tensor is created.
            return self.cpu_tensors[id(gpu_tensor)]
        if hasattr(gpu_tensor, "_offload_event"):
            # Used to start early copy of the tensor
            # marked with mark_can_start_offload method.
            gpu_tensor._offload_event.wait(self.stream)
            gpu_tensor._offload_event = None
        else:
            # Synchronize with the current stream to ensure that
            # the gpu tensor is computed before the copy starts.
            self.stream.wait_stream(torch.cuda.current_stream())

        # Since we allocate the cpu tensor on the different stream,
        # we keep the copies of the cpu tensors in self._offload_cpu_tensors.
        # This tensor is released in start_offloaded_layer_fwd method of the next layer,
        # after the synchronization with the main stream. 
        # This prevents the release of the memory of the cpu tensor before the copy is finished.
        with torch.cuda.stream(self.stream):
            cpu_tensor = torch.empty_like(gpu_tensor, device="cpu", pin_memory=True)
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        self.cpu_tensors[id(gpu_tensor)] = cpu_tensor
        self.gpu_tensors[id(gpu_tensor)] = gpu_tensor
        return cpu_tensor
    
    def wait_for_offloading(self):
        """
            Wait for the offloading to finish.
        """
        torch.cuda.current_stream().wait_stream(self.stream)
    
    def get_offloaded(self, gpu_tensor: torch.Tensor):
        """
            Return the CPU copy of the tensor.
        """
        return self.cpu_tensors[id(gpu_tensor)]
    
    def get_all_offloaded_tensors(self) -> list[torch.Tensor]:
        """
            Return all the CPU copies of the tensors.
        """
        return list(self.cpu_tensors.values())
    
    def release_memory(self):
        """
            Release the memory of the CPU tensors.
        """
        self.cpu_tensors = {} 
        self.gpu_tensors = {}

class _StreamedReloader:
    """
        _StreamedReloader represents one stream used to reload the tensors from the CPU to the GPU.
        It provides easy to use interface for reloading and synchronization.

        Parameters
        ----------
        reuse_gpu_buffers : bool, default = False
            Re-use the same GPU buffers when reloading tensors.  All offloaded
            layers must therefore produce activations of identical shapes, or an
            assertion will be raised.
    """
    def __init__(self, reuse_gpu_buffers: bool):
        self.stream = torch.cuda.Stream()
        self.gpu_tensors = {}
        self._reuse_gpu_buffers = reuse_gpu_buffers
        self._gpu_buffer_pool = [] # used to re-use the same GPU buffers
    
    def wait_for_main(self):
        """
            Postpone the reloading process until this point in the main stream.
        """
        self.stream.wait_stream(torch.cuda.current_stream())

    def bulk_reload(self, cpu_tensors: list[torch.Tensor]):
        """
            Reload all provided tensors from the CPU to the GPU.
            The main stream must wait for this call to finish,
            but it can start before the main stream reaches this point.
        """
        if self._reuse_gpu_buffers:
            assert len(cpu_tensors) == len(self._gpu_buffer_pool) or len(self._gpu_buffer_pool) == 0, \
                "All offloaded layers must produce the same number of \
                    activation tensors with reuse_gpu_buffers=True"
        for tensor in cpu_tensors:
            with torch.cuda.stream(self.stream):
                if self._reuse_gpu_buffers:
                    if len(self._gpu_buffer_pool) > 0:
                        buffer = self._gpu_buffer_pool.pop()
                        assert buffer.shape == tensor.shape, \
                            "All offloaded layers must produce activations of identical \
                                shapes to run with reuse_gpu_buffers=True"
                    else:
                        # if buffers are not allocated yet - invoked during the first offloaded layer
                        # with this _StreamedReloader
                        buffer = torch.empty_like(tensor, device="cuda")
                else:
                    buffer = torch.empty_like(tensor, device="cuda")
                buffer.copy_(tensor, non_blocking=True)
                self.gpu_tensors[id(tensor)] = buffer
        torch.cuda.current_stream().wait_stream(self.stream)
    
    def get_reloaded(self, cpu_tensor: torch.Tensor):
        """
            Return the GPU copy of the tensor.
        """
        assert id(cpu_tensor) in self.gpu_tensors.keys(),\
            "The tensor that you are trying to reload was not offloaded. \
                This error should not happen - please report the bug to the Transformer Engine."
        return self.gpu_tensors[id(cpu_tensor)]

    def release_memory(self):
        """
            Release the memory of the GPU tensors.
        """
        if self._reuse_gpu_buffers:
            self._gpu_buffer_pool.extend(self.gpu_tensors.values()) # return buffers to the pool
        self.gpu_tensors.clear()

    def release_buffers(self):
        """
            Release the memory of the GPU buffers.
        """
        self._gpu_buffer_pool = []


class _CPUOffloadBackend:
    """
        Class providing unified interface for offloading and reloading tensors.
        It can be translated into different public APIs.

        The calls should be in following order, represented by the following grammar:

        CALLS -> PROGRAM*
        PROGRAM -> FWD_LAYER* finish_fwd() start_bwd_reloading() BWD_LAYER*
        FWD_LAYER -> start_offloaded_layer_fwd() (mark_can_start_offload()|offload())* end_offloaded_layer_fwd()
        BWD_LAYER -> start_offloaded_layer_bwd() reload()* end_offloaded_layer_bwd()

        The method end_offloaded_layer_bwd() returns the number of the layer, \
        which should be passed to the start_offloaded_layer_bwd(). It enables different
        order of forward and backward passes - used for example in the pipeline parallelism.


        Parameters
        ----------
        reuse_gpu_buffers : bool, default = False
            Re-use the same GPU buffers when reloading tensors.  All offloaded
            layers must produce activations of identical shapes, or an
            assertion will be raised.
    """
    def __init__(self, reuse_gpu_buffers: bool = False):
        # Two streams are used to reload the tensors -
        # we want to release the memory at the end of the each layer,
        # but we also want to start relaoding the next layer before and finish reloading it after.
        # It is hard to achieve this with a single reloading stream.
        self.streamed_reloaders = [
            _StreamedReloader(reuse_gpu_buffers) for _ in range(2)
        ]
        # we switch the streamed reloader after every layer
        # this int indicates which reloader to use
        self.reloader_parity = 0 

        self.streamed_offloader = _StreamedOffloader() # one stream for offloading

        self.cur_layer_id = 0
        self.total_num_of_layers = 0
        self.total_num_of_reloaded_layers = 0
        
        self._total_offloaded_size = 0

        self.first_layer_fwd_flag = False
        self.inside_offloaded_layer_bwd_flag = False

        # layer_num -> id of cpu tensor -> cpu tensor
        # This dictionary of dictionaries stores the cpu copies of the tensors of all the layers.
        # They are used for reloading.
        self._offload_cpu_tensors_for_each_layer: dict[int, dict[int, torch.Tensor]] = {}
        
    def start_offloaded_layer_fwd(self):
        """
            Invoked before the new offloaded layer is started.
        """
        if not self.first_layer_fwd_flag:
            self._finish_layer_offload()
        else:
            self.first_layer_fwd_flag = False
        self.cur_layer_id = self.total_num_of_layers

    def end_offloaded_layer_fwd(self) -> int:
        """
            Call right after the forward pass of an offloaded layer.
        """
        self.total_num_of_layers += 1
        return self.cur_layer_id
    
    def finish_fwd(self) -> None:
        """
            Synchronization after fwd
        """
        self._finish_layer_offload()
        self.cur_layer_id = self.total_num_of_layers

        self.first_layer_fwd_flag = True # reset the flag after the forward pass
        
    def start_bwd_reloading(self) -> None:
        """
            Start reloading the first two backward layers.
        """
        # Reload should wait for this call.
        for reloader in self.streamed_reloaders:
            reloader.wait_for_main()
        
    def start_offloaded_layer_bwd(self, layer_num: int):
        """
            Invoked when the backward pass of offloaded layer is started.
        """
        if self.inside_offloaded_layer_bwd_flag:
            raise RuntimeError(
                "Backward of one offloaded layer started before the previous one finished. "
                "This is not supported by the Transformer Engine cpu offloading. "
                "We support only offloading of subsequence of sequence of consecutive layers - "
                "such that the output of one is the input of the next one."
            )
        self.inside_offloaded_layer_bwd_flag = True
        self.cur_layer_id = layer_num
        self.reloader_parity = 1 - self.reloader_parity
        cur_reloader = self._get_current_reloader()
        cpu_tensors = self._offload_cpu_tensors_for_each_layer.pop(layer_num, {})

        self._total_offloaded_size -= self._get_size(cpu_tensors)

        cur_reloader.bulk_reload(cpu_tensors) # main waits for cur_reloader to finish here.

        next_reloader = self._get_next_reloader()
        next_reloader.wait_for_main() # Reload of next layer should not start before this call.

    def end_offloaded_layer_bwd(self):
        """
            Invoked when the backward pass of offloaded layer is finished.
        """
        self.inside_offloaded_layer_bwd_flag = False
        cur_reloader = self._get_current_reloader()
        cur_reloader.wait_for_main()
        cur_reloader.release_memory()

        # Reset the number of reloaded layers.
        self.total_num_of_reloaded_layers += 1
        if self.total_num_of_reloaded_layers == self.total_num_of_layers:
            self.total_num_of_reloaded_layers = self.total_num_of_layers = 0
    
    def mark_can_start_offload(self, gpu_tensor: torch.Tensor):
        """
            Use this helper when a tensor is produced at the very beginning of
            torch.autograd.Function.forward and you want to kick off the GPU → CPU
            copy sooner than save_for_backward would allow.  We attach a CUDA
            event to the tensor so we can later wait on it and begin the transfer
            exactly when the tensor is ready.

            Notes:
            - The tensor must eventually be passed to offload(); otherwise this
              call is a no-op.
            - Any tensor that should be transferred early must also be handed to
              offload() earlier than the rest.
            - For Float8Tensor instances we want to copy as early as
              possible, helper calls such as update_usage may discard the
              row-wise data.  By tagging the event first, then invoking
              update_usage, and finally calling offload(), the copy is initiated
              at the correct moment and correct data are copied.
        """
        self.streamed_offloader.mark_can_start_offload(gpu_tensor)

    def offload(self, gpu_tensor: torch.Tensor) -> torch.Tensor:
        """
            Starts asynchronous copy of the tensor from GPU to CPU.
            Returns the CPU copy of the tensor.
        """
        self.layer_fwd_with_unfinished_offload = True
        cpu_tensor = self.streamed_offloader.offload(gpu_tensor)
        return cpu_tensor
    
    def reload(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """
            Return the GPU copy corresponding to `cpu_tensor`.
        """
        cur_reloader = self._get_current_reloader()
        return cur_reloader.get_reloaded(cpu_tensor)

    def clear_buffers(self):
        """
            Clear the buffers for the activations. Can be used only when reuse_gpu_buffers is True.
        """
        assert self.reuse_gpu_buffers, "clear_buffers is only allowed when reuse_gpu_buffers is True"
        for reloader in self.streamed_reloaders:
            reloader.wait_for_main()
            reloader.release_buffers()
    
    def _get_next_reloader(self):
        """
            Get the next reloader.
        """
        return self.streamed_reloaders[(self.reloader_parity + 1) % 2]

    def _get_current_reloader(self):
        """
            Get the current reloader.
        """
        return self.streamed_reloaders[self.reloader_parity]

    def _finish_layer_offload(self):
        """
            Finish offloading of the previous layer.
        """
        self.layer_fwd_with_unfinished_offload = False
        self.streamed_offloader.wait_for_offloading()
        self._offload_cpu_tensors_for_each_layer[self.cur_layer_id] = \
                self.streamed_offloader.get_all_offloaded_tensors()
        self._total_offloaded_size += \
            self._get_size(self._offload_cpu_tensors_for_each_layer[self.cur_layer_id])
        self.streamed_offloader.release_memory()
    
    def get_offloaded_total_size_mb(self):
        """
            Return the size of tensors that currenlty have CPU copy,
            in megabytes.
        """
        # For debugging purposes
        return self._total_offloaded_size / 1024 / 1024

    def _get_size(self, cpu_tensors: list[torch.Tensor]):
        """
            Get the size of the cpu tensors in bytes.
        """
        total_size = 0
        for cpu_tensor in cpu_tensors:
            if type(cpu_tensor) == torch.Tensor:
                total_size += cpu_tensor.numel() * cpu_tensor.element_size()
            else:
                for tensor in cpu_tensor.get_data_tensors():
                    if tensor is not None:
                        total_size += tensor.numel() * tensor.element_size()
        return total_size


class _CPUOffloadPackHooks:
    """
        Context manager inseting hooks inside packing and unpacking the tensors.
    """
    
    def __init__(self, backend: _CPUOffloadBackend):
        self.backend = backend
        self.context = saved_tensors_hooks(self._pack_hook, self._unpack_hook)
    
    def __enter__(self):       
        self.context.__enter__()

    def __exit__(self, *args):
        self.context.__exit__()

    def _pack_hook(self, tensor: torch.Tensor) -> tuple[torch.Tensor, _CPUOffloadBackend, bool]:
        """
            If tensor needs to be offloaded - if it has activation_offloading attribute - offload it.

            Returns:
            - tensor: the offloaded or non-offloaded tensor.
            - handler: the offload handler or None - it is needed to restore the tensor in the backward pass.
        """

        if isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        ):
            return (tensor, None)

        if self._offload_checker(tensor):
            return (self.backend.offload(tensor), self.backend)
        else:
            return (tensor, None)
    
    def _unpack_hook(self, tensor_handler: tuple[torch.Tensor, _CPUOffloadBackend, bool]):
        """
            Unpacks the tensor and the offload handler.
        """
        tensor, handler = tensor_handler
        if handler is not None:
            tensor.offload_handler = handler
            tensor = self.backend.reload(tensor)
        return tensor
    
    def _offload_checker(self, tensor: torch.Tensor):
        """
            Check if the tensor should be offloaded.
        """
        # We do not offload parameters/weights.
        if isinstance(tensor, torch.nn.Parameter):
            return False

        # Sometimes weights are processed inside the TransformerEngine layer,
        # we do not want to offload them.
        if hasattr(tensor, "is_weight"):
            return False

        # We do not offload too small tensors.
        if tensor.numel() < MIN_TENSOR_SIZE_TO_OFFLOAD:
            return False

        return True

class _SwitchCPUOffloadHandler:
    """
        Context manager to switch the CPU offload handler.
    """
    def __init__(self, backend: _CPUOffloadBackend):
        self.backend = backend

    def __enter__(self):
        global CURRENT_CPU_OFFLOAD_HANDLER
        self.previous_backend = CURRENT_CPU_OFFLOAD_HANDLER
        CURRENT_CPU_OFFLOAD_HANDLER = self.backend
        
    def __exit__(self, *args):
        global CURRENT_CPU_OFFLOAD_HANDLER
        CURRENT_CPU_OFFLOAD_HANDLER = self.previous_backend

class CPUOffload:
    """
        The CPUOffload class enables asynchronous offloading of activations.
        If we have n consecutive transformer layers, we can choose some of them to be offloaded.
        The offloading of the next layer begins after the offloading of the previous layer has finished.
        The forward pass of the last layer starts after the offloading of the last layer has finished.
        During the backward pass, the reload begins after the gradients of the last layer are computed.
        Only one layer is reloaded at a time; the reload of the next layer starts after the backward pass of the previous
        offloaded layer has begun.

        This ensures that if k out of n identical layers are offloaded, then at most n - k activations are present in memory simultaneously.
        We recommend offloading 1 out of every x layers for a sufficiently large x. This will ensure that computation and offloading
        are fully overlapped, reducing memory usage.

        Each layer must be wrapped with an instance of the CPUOffload class - activation offloading
        for a particular layer is enabled/disabled by the offload_activations parameter. The last layer needs
        to be wrapped with the is_last_layer parameter set to True - activations of this layer cannot be offloaded.

        CPUOffload supports all torch.nn.Module, not only these provided by the Transformer Engine.

        The last layer must have offload_activations=False. CPUOffload supports only sequences of torch.nn.Module;
        other graph structures are not supported. CPUOffload supports multiple autograd graphs - it can be used
        for pipeline parallelism, for example.


        Example:
        --------
        ```python
        # ...
        cpu_offload_wrapper = CPUOffload()

        # Wrap all transformer layers, not just these you want to offload.
        # It enables optimal synchronization.
        layer1 = cpu_offload_wrapper(layer1, offload_activations=True)
        layer2 = cpu_offload_wrapper(layer2, offload_activations=False)
        layer3 = cpu_offload_wrapper(layer3, is_last_layer=True)

        x2 = layer1(x1)
        x3 = layer2(x2)
        y = layer3(x3)
        y.sum().backward()

        # ...

        Parameters
        ----------
        reuse_gpu_buffers : bool, default = False
            Re-use the same GPU buffers when reloading tensors.  All offloaded
            layers must produce activations of identical shapes, or an
            assertion will be raised.
        ```
    """
    def __init__(self, reuse_gpu_buffers: bool = False):
        self.backend = _CPUOffloadBackend(reuse_gpu_buffers)

        self.pack_hooks = _CPUOffloadPackHooks(self.backend)
        self.switch_cpu_offload_handler = _SwitchCPUOffloadHandler(self.backend)

    def __call__(self, module, offload_activations: bool = False, is_last_layer: bool = False):
        """
           Wraps the function, which activation is offloaded.

           Parameters
           ----------
           module : torch.nn.Module
               The module, which activation is offloaded.
           offload_activations : bool
               If True, the activation is offloaded.
            
            Returns
            -------
            torch.nn.Module
                The wrapped module.
        """
        assert not is_last_layer or not offload_activations, "Last layer activations cannot be offloaded."

        # The module is wrapped into CPUOffloadModule,
        # and the hooks are registered on the wrapped module.
        class CPUOffloadModule(torch.nn.Module):
            def __init__(self, module: torch.nn.Module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)
        
        cpu_offload_module = CPUOffloadModule(module)

        def forward_pre_hook(model, input):
            self.switch_cpu_offload_handler.__enter__()
            self.backend.start_offloaded_layer_fwd()
            self.pack_hooks.__enter__()

        def forward_hook(model, input, output):
            self.switch_cpu_offload_handler.__exit__()
            model.layer_id = self.backend.end_offloaded_layer_fwd()
            self.pack_hooks.__exit__()
        
        def backward_pre_hook(model, input):
            self.backend.start_offloaded_layer_bwd(model.layer_id)

        def backward_hook(model, grad_input, grad_output):
            if len(grad_input) == 1 and grad_input[0] is None:
                # For last layer, when input gradients are not needed,
                # we do not call the backward hook,
                # because it is called before the backward pass is even computed.

                # We will call the end_offloaded_layer_bwd after the backward pass is finished.
                torch.autograd.variable.Variable._execution_engine.queue_callback(
                    self.backend.end_offloaded_layer_bwd
                )
                return
            self.backend.end_offloaded_layer_bwd()
        
        if offload_activations:
            cpu_offload_module.register_forward_pre_hook(forward_pre_hook)
            cpu_offload_module.register_forward_hook(forward_hook)
            cpu_offload_module.register_full_backward_pre_hook(backward_pre_hook)
            cpu_offload_module.register_full_backward_hook(backward_hook)
        if is_last_layer:
            cpu_offload_module.register_forward_pre_hook(lambda *args: self.backend.finish_fwd())
            cpu_offload_module.register_full_backward_hook(lambda *args: self.backend.start_bwd_reloading())

        return cpu_offload_module

CURRENT_CPU_OFFLOAD_HANDLER = None

def get_cpu_offload_context(
    enabled: bool = False,
    num_layers: int = 1,
    model_layers: int = 1,
    offload_activations: bool = True,
    offload_weights: bool = False,
):
    """
    Legacy offloading API, will be removed in the future.
    """

    print("[WARNING] get_cpu_offload_context is deprecated. Use CPUOffload instead.")

    if not offload_weights and not offload_activations:
        raise ValueError(
            "CPU Offloading is enabled while it is not "
            "mentioned what to offload (weights/activations)"
        )

    if offload_weights:
        import warnings

        warnings.warn(
            "Offloading weights is deprecated. Using offload_weights=True does not have any"
            " effect.",
            DeprecationWarning,
        )

        # Weights offloading is deprecated but we maintain backward compatibility by doing nothing.
        if not offload_activations:
            return contextlib.nullcontext(), lambda x: x


    class _CpuOffloadContext(contextlib.ContextDecorator):
        def __init__(self, backend: _CPUOffloadBackend):
            self.backend = backend
            self.previous_backend = None

            self.current_layer = 0
            self.offload_layer = {} # int -> bool
            self.pack_hooks =  _CPUOffloadPackHooks(self.backend)
            self.switch_cpu_offload_handler = _SwitchCPUOffloadHandler(self.backend)

            self.offload_layer = self._get_layers_to_offload(num_layers, model_layers)
        
        def _get_layers_to_offload(self, num_layers_to_offload: int, model_layers: int):
            offload_layer = {}
            offload_layer[0] = True 
            for i in range(1, model_layers):
                offload_layer[i] = False
            constant = 0
            for i in range(num_layers_to_offload - 1):
                layer_to_offload = ((model_layers // num_layers_to_offload) * (i + 1)) - 1
                if i < (model_layers % num_layers_to_offload):
                    layer_to_offload += i + 1
                    constant = i + 1
                else:
                    layer_to_offload += constant

                offload_layer[layer_to_offload] = True

            return offload_layer

        def __enter__(self):
            if self.offload_layer[self.current_layer]:
                self.switch_cpu_offload_handler.__enter__()
                self.pack_hooks.__enter__()
                self.backend.start_offloaded_layer_fwd()
                

        def __exit__(self, *args):
            if self.offload_layer[self.current_layer]:
                self.switch_cpu_offload_handler.__exit__()
                self.pack_hooks.__exit__()
                self.backend.end_offloaded_layer_fwd()

            if self.current_layer == model_layers - 1:
                # finish the forward pass
                self.backend.finish_fwd()

            self.current_layer += 1

        def synchronization_function(self, tensor):
            assert tensor.requires_grad == True

            def hook(_):
                self.current_layer -= 1
                if self.current_layer < 0:
                    return 
                
                if self.current_layer == model_layers - 1:
                    # start reloading after the last layer bwd
                    self.backend.start_bwd_reloading()
                
                if self.offload_layer.get(self.current_layer + 1, False):
                    self.backend.end_offloaded_layer_bwd()

                if self.offload_layer[self.current_layer]:
                    self.backend.start_offloaded_layer_bwd(self.current_layer)
                
                if self.current_layer == 0:
                    torch.autograd.variable.Variable._execution_engine.queue_callback(
                        self.backend.end_offloaded_layer_bwd
                    )

            tensor.grad_fn.register_prehook(hook)
            return tensor
                
    backend = _CPUOffloadBackend()
    cpu_offload_context = _CpuOffloadContext(backend)

    if enabled:
        return (
            cpu_offload_context,
            cpu_offload_context.synchronization_function,
        )
    else:
        return contextlib.nullcontext(), lambda x: x
