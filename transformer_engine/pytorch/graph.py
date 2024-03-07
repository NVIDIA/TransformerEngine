# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functions for CUDA Graphs support in FP8"""
import torch
from torch.utils._pytree import tree_flatten as _tree_flatten
from torch.utils._pytree import tree_unflatten as _tree_unflatten
from torch._C import _graph_pool_handle

from .fp8 import (
    fp8_autocast,
    FP8GlobalStateManager,
    set_fp8_graph_capture_start,
    set_fp8_graph_capture_end,
    get_default_fp8_recipe,
)
from .distributed import _set_cuda_rng_state
from .module.base import TransformerEngineBaseModule


__all__ = ["make_graphed_callables"]


def graph_pool_handle():
    """
    Returns an opaque token representing the id of a graph memory pool.
    """
    return _graph_pool_handle()


def _make_graphed_callables(
    callables,
    sample_args,
    num_warmup_iters=3,
    allow_unused_input=False,
    fp8_weight_caching=False,
):
    """
    Helper method for `make_graphed_callables`
    """

    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast "
            "caching. Please set `cache_enabled=False`."
        )

    just_one_callable = False

    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)

    flatten_sample_args = []

    if fp8_weight_caching:
        modified_sample_args = []
        for args in sample_args:
            args += (torch.empty(1, device="cuda"),)
            modified_sample_args.append(args)
        sample_args = modified_sample_args

    for c, args in zip(callables, sample_args):
        if isinstance(c, torch.nn.Module):
            assert (
                len(c._backward_hooks) == 0
                and len(c._forward_hooks) == 0
                and len(c._forward_pre_hooks) == 0
            ), (
                "Modules must not have hooks registered at the time they are passed. "
                + "However, registering hooks on modules after passing them "
                + "through make_graphed_callables is allowed."
            )
            assert all(b.requires_grad is False for b in c.buffers()), (
                "In any :class:`~torch.nn.Module` passed to "
                + ":func:`~make_graphed_callables`, only parameters may be trainable. "
                + "All buffers must have ``requires_grad=False``."
            )
        flatten_arg, _ = _tree_flatten(args)
        flatten_sample_args.append(tuple(flatten_arg))
        assert all(isinstance(arg, torch.Tensor) for arg in flatten_arg), (
            "In the beta API, sample_args "
            + "for each callable must contain only Tensors. Other types are not allowed."
        )

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    per_callable_module_params = [
        tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
        for c in callables
    ]
    per_callable_static_input_surfaces = [
        flatten_sample_args[i] + per_callable_module_params[i]
        for i in range(len(callables))
    ]

    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]

    mempool = graph_pool_handle()

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        for func, args, static_input_surface in zip(
            callables, sample_args, per_callable_static_input_surfaces
        ):
            for _ in range(num_warmup_iters):
                outputs, _ = _tree_flatten(func(*args))
                grad_inputs = torch.autograd.grad(
                    outputs=tuple(o for o in outputs if o.requires_grad),
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(
                        torch.empty_like(o) for o in outputs if o.requires_grad
                    ),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )
            del outputs, grad_inputs
    torch.cuda.synchronize()

    # All captures here share a mempool. To avoid replays corrupting each other's memory,
    # the safest approach is to capture all passes in the same order they'll run:
    # fwd 1, fwd 2, ... fwd N, then bwd N, bwd N-1, ... bwd 1.

    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    for func, args, fwd_graph in zip(callables, sample_args, fwd_graphs):
        with torch.cuda.graph(fwd_graph, pool=mempool):
            outputs = func(*args)

        flatten_outputs, spec = _tree_flatten(outputs)
        per_callable_static_outputs.append(tuple(flatten_outputs))
        per_callable_output_unflatten_spec.append(spec)

    # Capture backward graphs in reverse order
    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    for static_input_surface, static_outputs, bwd_graph in zip(
        reversed(per_callable_static_input_surfaces),
        reversed(per_callable_static_outputs),
        reversed(bwd_graphs),
    ):
        # For now, assumes all static_outputs require grad
        static_grad_outputs = tuple(
            torch.empty_like(o) if o.requires_grad else None for o in static_outputs
        )

        with torch.cuda.graph(bwd_graph, pool=mempool):
            grad_inputs = torch.autograd.grad(
                outputs=tuple(o for o in static_outputs if o.requires_grad),
                inputs=tuple(i for i in static_input_surface if i.requires_grad),
                grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                only_inputs=True,
                allow_unused=allow_unused_input,
            )

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs that
        # don't require grad. I couldn't think of a slick one-liner for this pattern.
        static_grad_inputs = []
        grad_idx = 0
        for arg in static_input_surface:
            if arg.requires_grad:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)  # type: ignore[arg-type]
        static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]

        per_callable_static_grad_outputs.append(static_grad_outputs)
        per_callable_static_grad_inputs.append(static_grad_inputs)

    # Reverses the most recent two lists
    per_callable_static_grad_outputs = list(reversed(per_callable_static_grad_outputs))
    per_callable_static_grad_inputs = list(reversed(per_callable_static_grad_inputs))
    # Now for every per_callable list, per_callable_*[i] holds the stuff for the ith callable.

    def make_graphed_autograd_function(
        fwd_graph,
        bwd_graph,
        module_params,
        len_user_args,
        output_unflatten_spec,
        static_input_surface,
        static_outputs,
        static_grad_outputs,
        static_grad_inputs,
    ):
        class Graphed(torch.autograd.Function):
            """Autograd function for graph replay."""
            @staticmethod
            def forward(ctx, *inputs):
                # At this stage, only the user args may (potentially) be new tensors.
                ctx.is_first_module = FP8GlobalStateManager.is_first_fp8_module()
                for i in range(len_user_args):
                    if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                        static_input_surface[i].copy_(inputs[i])
                fwd_graph.replay()
                assert isinstance(static_outputs, tuple)
                return tuple(o.detach() for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grads):
                assert len(grads) == len(static_grad_outputs)
                for g, grad in zip(static_grad_outputs, grads):
                    if g is not None:
                        # don't copy if autograd gods have been kind and the
                        # incoming grad is already in the right place
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                bwd_graph.replay()

                if ctx.is_first_module:
                    if callable(FP8GlobalStateManager.amax_backward_global_reduce_func):
                        FP8GlobalStateManager.amax_reduce_handle_bwd = (
                            FP8GlobalStateManager.amax_backward_global_reduce_func()) # pylint: disable=not-callable

                # Input args that didn't require grad expect a None gradient.
                assert isinstance(static_grad_inputs, tuple)
                return tuple(
                    b.detach() if b is not None else b for b in static_grad_inputs
                )

        def functionalized(*user_args, **user_kwargs):
            # Runs the autograd function with inputs == all
            # inputs to the graph that might require grad
            # (explicit user args + module parameters)
            # Assumes module params didn't change since capture.
            if fp8_weight_caching:
                assert (
                    ("is_first_microbatch" in user_kwargs
                     and isinstance(user_kwargs["is_first_microbatch"], bool))
                ), "`is_first_microbatch` boolean kwarg must be provided for FP8 weight caching."
                f = torch.zeros if user_kwargs["is_first_microbatch"] else torch.ones
                user_args += (f(1, device="cuda"),)

            flatten_user_args, _ = _tree_flatten(user_args)
            out = Graphed.apply(*(tuple(flatten_user_args) + module_params))
            return _tree_unflatten(out, output_unflatten_spec)

        return functionalized

    # Put together the final graphed callables
    ret = []
    for i, func in enumerate(callables):
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            bwd_graphs[i],
            per_callable_module_params[i],
            per_callable_len_user_args[i],
            per_callable_output_unflatten_spec[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
            per_callable_static_grad_outputs[i],
            per_callable_static_grad_inputs[i],
        )

        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
                def new_fwd(*user_args, **user_kwargs):
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        # Set the FP8 group from global amax reduction.
                        for module in func.modules():
                            if isinstance(module, TransformerEngineBaseModule):
                                module.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
                        return graphed(*user_args, **user_kwargs)
                    return orig_fwd(*user_args, **user_kwargs)

                return new_fwd

            func.forward = make_graphed_forward(func, func.training, graphed, func.forward)
            ret.append(func)
        else:
            ret.append(graphed)

    if just_one_callable:
        return ret[0]

    return tuple(ret)


def save_fp8_tensors(modules, amax_history_len):
    """
    Returns the FP8 tensors for all modules
    with adjusted amax history sizes.
    """
    saved_fp8_meta_tensors = []
    for module in modules:
        for m in module.modules():
            if isinstance(m, TransformerEngineBaseModule):
                if m.primary_weights_in_fp8:
                    m.adjust_amax_history_length(amax_history_len)
                saved_fp8_meta_tensors.append(m.get_fp8_meta_tensors())
    return saved_fp8_meta_tensors


def restore_fp8_tensors(modules, fp8_tensors):
    """Restore FP8 tensors."""
    for module in modules:
        for m in module.modules():
            if isinstance(m, TransformerEngineBaseModule):
                m.reset_fp8_meta_tensors(fp8_tensors.pop(0))
    assert len(fp8_tensors) == 0, "TE internal error."


def make_graphed_callables(
    modules,
    sample_args,
    num_warmup_iters=3,
    allow_unused_input=False,
    enabled=False,
    calibrating=False,
    fp8_recipe=None,
    fp8_weight_caching=False,
):
    """
    Accepts TransformerEngine modules and returns graphed versions. This function is based
    on the `torch.cuda.make_graphed_callables` function from PyTorch. See
    `torch.utils.checkpoint.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_
    for extensive documentation.
    """

    # Set capture.
    if enabled:
        set_fp8_graph_capture_start()
        assert num_warmup_iters > 0, "Warmup is required for FP8 graph capture."

    fp8_recipe = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe

    # Handle single module.
    just_one_callable = False
    if not isinstance(modules, tuple):
        just_one_callable = True
        modules = (modules,)

    # Store FP8 tensors to reset later.
    saved_fp8_tensors = save_fp8_tensors(modules, fp8_recipe.amax_history_len)

    # FP8 wrapper.
    def wrap_autocast(block):
        old_forward = block.forward
        def forward_func(*args, **kwargs):
            with fp8_autocast(enabled=enabled,
                              calibrating=calibrating,
                              fp8_recipe=fp8_recipe):
                outputs = old_forward(*args, **kwargs)
            return outputs
        block.forward = forward_func

    forward_funcs = []
    for module in modules:
        assert isinstance(module, torch.nn.Module), f"Graphing for {type(module)} is not supported."
        wrap_autocast(module)
        forward_funcs.append(module)

        # This is not strictly necessary since adding bwd hooks to children modules
        # is okay for graph capture as long it's just for kernel launches, but it's
        # safer to remove these hooks now and re-add them post capture.
        for m in module.modules():
            if isinstance(m, TransformerEngineBaseModule):
                if m.fp8_meta["bwd_amax_reduce_hook"] is not None:
                    m.fp8_meta["bwd_amax_reduce_hook"].remove()

    if just_one_callable:
        forward_funcs = forward_funcs[0]
    else:
        forward_funcs = tuple(forward_funcs)

    # Save RNG state.
    cuda_rng_state = torch.cuda.get_rng_state()

    graphed_callables = _make_graphed_callables(
        forward_funcs, sample_args, num_warmup_iters=num_warmup_iters,
        allow_unused_input=allow_unused_input,
        fp8_weight_caching=fp8_weight_caching)

    # Ensures warmup does not affect numerics for ops such as dropout.
    _set_cuda_rng_state(cuda_rng_state)

    # Reset FP8 gradients.
    for module in modules:
        for p in module.parameters():
            p.grad = None

    # Restore FP8 state.
    restore_fp8_tensors(modules, saved_fp8_tensors)

    set_fp8_graph_capture_end()
    return graphed_callables
