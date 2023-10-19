# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Methods needed for distributed training (DP/TP)."""
from contextlib import contextmanager
from typing import Any, Dict, Union, Optional, Callable, Tuple

import torch
from torch.cuda import _lazy_call
from torch.utils.checkpoint import detach_variable

from .utils import safely_set_viewless_tensor_data
from .constants import dist_group_type
from .fp8 import FP8GlobalStateManager


__all__ = ["checkpoint", "CudaRNGStatesTracker"]


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}

_FP8_ACTIVATION_RECOMPUTE_ENABLED = False
_FP8_ACTIVATION_RECOMPUTE_PHASE = False


def _set_cuda_rng_state(new_state: torch.Tensor, device: Union[int, str] = -1) -> None:
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if device == -1:
        device = torch.device("cuda")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)

    def cb() -> None:
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        default_generator = torch.cuda.default_generators[idx]
        default_generator.set_state(new_state)

    _lazy_call(cb)


def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor, is_parallel: bool, dim: int, stride: int
) -> None:
    """set attributes needed for TP"""
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def get_distributed_world_size(group: Optional[dist_group_type] = None) -> int:
    """Return world size for the distributed group."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


def get_distributed_rank(group: Optional[dist_group_type] = None) -> int:
    """Return my rank for the distributed group."""
    assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
    return torch.distributed.get_rank(group=group)


def initialize_affine_weight_gpu(
    weight: torch.Tensor,
    init_method: Callable,
    get_rng_state_tracker: Callable,
    partition_dim: int = 0,
    stride: int = 1,
    set_tp_attributes: bool = True,
) -> None:
    """Initialize affine weight for model parallel on GPU."""

    if set_tp_attributes:
        set_tensor_model_parallel_attributes(
            tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
        )

    if get_rng_state_tracker is None:
        init_method(weight)
        return

    with get_rng_state_tracker().fork():
        init_method(weight)


def split_tensor_into_1d_equal_chunks(
    tensor: torch.Tensor, tp_group: dist_group_type, new_buffer: bool = False
) -> torch.Tensor:
    """Break a tensor into equal 1D chunks."""
    partition_size = torch.numel(tensor) // get_distributed_world_size(tp_group)
    start_index = partition_size * get_distributed_rank(tp_group)
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(
    tensor: torch.Tensor, tp_group: dist_group_type
) -> torch.Tensor:
    """Opposite of above function, gather values from model parallel ranks."""
    numel_gathered = torch.numel(tensor) * get_distributed_world_size(tp_group)
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    torch.distributed.all_gather_into_tensor(gathered, tensor, group=tp_group)
    return gathered


@contextmanager
def activation_recompute_forward(
    activation_recompute: bool = False,
    recompute_phase: bool = False,
) -> None:
    """Context manager used to control the forward runtime behavior when executed
    under the `CheckpointFunction` function. For running FP8, the forward pass will
    run without storing intermediate activations. Instead, the forward pass saves
    the inputs tuple and the calling function. In the backwards pass, these are
    retrieved, and the forward pass is computed again while tracking the intermediate
    activations, followed by calculation of gradients using these values.
    """
    global _FP8_ACTIVATION_RECOMPUTE_ENABLED, _FP8_ACTIVATION_RECOMPUTE_PHASE
    try:
        _FP8_ACTIVATION_RECOMPUTE_ENABLED = (
            activation_recompute and FP8GlobalStateManager.is_fp8_enabled())
        _FP8_ACTIVATION_RECOMPUTE_PHASE = recompute_phase
        yield
    finally:
        _FP8_ACTIVATION_RECOMPUTE_ENABLED = False
        _FP8_ACTIVATION_RECOMPUTE_PHASE = False


def is_fp8_activation_recompute_enabled() -> bool:
    """Return global boolean"""
    return _FP8_ACTIVATION_RECOMPUTE_ENABLED


def in_fp8_activation_recompute_phase() -> bool:
    """Return global boolean"""
    return _FP8_ACTIVATION_RECOMPUTE_PHASE


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(
        ctx,
        run_function: Callable,
        distribute_saved_activations: bool,
        get_cuda_rng_tracker: Callable,
        tp_group: dist_group_type,
        kwargs: Dict[str, Any],
        *args: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Call forward function while saving state to be able to
        redo the computation later."""
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            with activation_recompute_forward(
                activation_recompute=True, recompute_phase=False
            ):
                outputs = run_function(*args, **kwargs)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(
                    args[0].data, tp_group, new_buffer=True
                ),
            )

        # Store everything.
        ctx.save_for_backward(*args)
        ctx.get_cuda_rng_tracker = get_cuda_rng_tracker
        ctx.tp_group = tp_group
        ctx.kwargs = kwargs

        return outputs

    @staticmethod
    def backward(
        ctx, *args: Tuple[Union[torch.Tensor, None], ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Call backward function with activation recomputation."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        inputs = ctx.saved_tensors
        get_cuda_rng_tracker = ctx.get_cuda_rng_tracker

        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data, ctx.tp_group).view(
                    ctx.input_0_shape
                ),
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            with activation_recompute_forward(
                activation_recompute=True, recompute_phase=True
            ):
                outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, args)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else inp
            for inp in detached_inputs
        )
        return (None, None, None, None, None) + grads


def checkpoint(
    function: Callable,
    distribute_saved_activations: bool,
    get_cuda_rng_tracker: Callable,
    tp_group: dist_group_type,
    *args: Tuple[torch.Tensor, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    """
    Checkpoint a part of the model by trading compute for memory. This function is based on
    `torch.utils.checkpoint.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_.

    .. warning::

        It is the user's responsibility to ensure identical behavior when calling
        :attr:`function` from the forward and backward pass. If different output is
        produced (e.g. due to global state), then the checkpointed version won't
        be numerically equivalent.

    .. warning::

        The tuple :attr:`args` must contain only tensors (or :attr:`None`) in order to comply with
        PyTorch's :attr:`save_for_backward` method. :attr:`function` must be callable to produce
        valid outputs with the inputs :attr:`args` and :attr:`kwargs`.

    Parameters
    ----------
    function: Callable
            pytorch module used to run the forward and backward passes using
            the specified :attr:`args` and :attr:`kwargs`.
    distribute_saved_activations: bool
            if set to `True`, the first tensor argument is distributed across the
            specified tensor parallel group (`tp_group`) before saving it for the
            backward pass.
    get_cuda_rng_tracker: `Callable`
            python callable which returns an instance of :func:`CudaRNGStatesTracker`.
    tp_group : ProcessGroup, default = `None`
            tensor parallel process group.
    args : tuple
            tuple of torch tensors for inputs to :attr:`function`.
    kwargs : dict
            dictionary of string keys for keyword arguments to :attr:`function`.
    """

    return CheckpointFunction.apply(
        function,
        distribute_saved_activations,
        get_cuda_rng_tracker,
        tp_group,
        kwargs,
        *args,
    )


class CudaRNGStatesTracker:
    """
    For model parallelism, multiple RNG states need to simultaneously exist in order
    to execute operations in or out of the model parallel region. This class keeps
    track of the various RNG states and provides utility methods to maintain them and
    execute parts of the model under a given RNG setting. Using the `add` method, a
    cuda rng state is initialized based on the input `seed` and is assigned to `name`.
    Later, by forking the rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """
        Set to the initial state (no tracker).
        """
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self) -> Dict[str, torch.Tensor]:
        """
        Get rng states. Copy the dictionary so we have direct pointers
        to the states, not just a pointer to the dictionary.
        """
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states: Dict[str, torch.Tensor]) -> None:
        """
        Set the rng states. For efficiency purposes, we do not
        check the size of seed for compatibility.

        states: Dict[str, torch.Tensor]
               A mapping from string names to RNG states.
        """
        self.states_ = states

    def add(self, name: str, seed: int) -> None:
        """
        Adds a new RNG state.

        name: str
             string identifier for the RNG state.
        seed: int
             PyTorch seed for the RNG state.
        """
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception(f"seed {seed} already exists")
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception(f"cuda rng state {name} already exists")
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextmanager
    def fork(self, name: str = "model-parallel-rng") -> None:
        """
        Fork the cuda rng state, perform operations, and exit with
        the original state.

        name: str
             string identifier for the RNG state.
        """
        # Check if we have added the state
        if name not in self.states_:
            raise Exception(f"cuda rng state {name} is not added")
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


def reduce_scatter_along_first_dim(
    input_: torch.Tensor, tp_group: dist_group_type, async_op: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_distributed_world_size(tp_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_, None

    dim_size = list(input_.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=tp_group, async_op=async_op
    )
    return output, handle


def gather_along_first_dim(
    input_: torch.Tensor, tp_group: dist_group_type, async_op: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_distributed_world_size(tp_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_, None

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    handle = torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=tp_group, async_op=async_op
    )

    return output, handle


def gather_along_last_dim(
    input_: torch.Tensor, tp_group: dist_group_type, async_op: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_distributed_world_size(tp_group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_, None

    dim_size = list(input_.size())
    dim_size[-1] = dim_size[-1] * world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    handle = torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=tp_group, async_op=async_op
    )

    return output, handle


def allreduce(
    input_: torch.Tensor,
    tp_group: Optional[dist_group_type] = None,
    async_op: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_distributed_world_size(tp_group) == 1:
        return input_, None

    # All-reduce.
    handle = torch.distributed.all_reduce(input_, group=tp_group, async_op=async_op)

    return input_, handle
