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
    get_default_fp8_recipe,
)
from .distributed import get_all_rng_states, graph_safe_rng_available
from .module.base import TransformerEngineBaseModule


__all__ = ["make_graphed_callables"]


_IS_GRAPH_CAPTURING = False


def set_capture_start() -> None:
    """Record beginning of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = True


def set_capture_end() -> None:
    """Record end of `make_graphed_callables`."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = False


def is_graph_capturing() -> None:
    """Return whether within `make_graphed_callables`."""
    return _IS_GRAPH_CAPTURING


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
    _order=None,
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
    if _order is not None:
        # order is a list containing 1..model_chunk values in the order of microbatch schedule
        num_model_chunks = max(_order)
        num_microbatches = len(_order) // num_model_chunks // 2
        assert num_model_chunks * num_microbatches * 2 == len(_order)
        assert len(sample_args) * 2 >= len(_order) and (
            len(sample_args) * 2 % len(_order) == 0
        ), f"{len(sample_args)} >= {len(_order)} and {len(sample_args)} % {len(_order)} == 0"
        num_layers = len(sample_args) // num_model_chunks // num_microbatches
        assert len(callables) == num_model_chunks * num_layers, (
            f"Callables should have ({num_model_chunks * num_layers}) "
            + f"entries when order input is provided but got {len(callables)}."
        )
        assert len(sample_args) == num_model_chunks * num_microbatches * num_layers, (
            f"Expected {num_model_chunks * num_microbatches}"
            + f"args tuple, but got {len(sample_args)}."
        )

    if fp8_weight_caching:
        FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(False)

    for c in callables:
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
    for args in sample_args:
        flatten_arg, _ = _tree_flatten(args)
        flatten_sample_args.append(tuple(flatten_arg))
        assert all(isinstance(arg, torch.Tensor) for arg in flatten_arg), (
            "In the beta API, sample_args "
            + "for each callable must contain only Tensors. Other types are not allowed."
        )

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    if _order is None:
        per_callable_module_params = [
            tuple(c.parameters()) if isinstance(c, torch.nn.Module) else () for c in callables
        ]
        per_callable_static_input_surfaces = [
            flatten_sample_args[i] + per_callable_module_params[i] for i in range(len(callables))
        ]
    else:
        per_callable_module_params = []
        for c in callables:
            for i in range(num_microbatches):
                per_callable_module_params.append(
                    tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
                )
        assert len(per_callable_module_params) == len(flatten_sample_args)
        per_callable_static_input_surfaces = [
            flatten_sample_args[i] + per_callable_module_params[i]
            for i in range(len(flatten_sample_args))
        ]

    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(flatten_sample_args))]
    graph_callables = [None for _ in range(len(flatten_sample_args))]
    # For cases with multiple active RNG states, e.g. TP.
    if graph_safe_rng_available():
        for _, state in get_all_rng_states().items():
            for fwd_graph, bwd_graph in zip(fwd_graphs, bwd_graphs):
                fwd_graph.register_generator_state(state)
                bwd_graph.register_generator_state(state)

    mempool = graph_pool_handle()

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        for c_i, func in enumerate(callables):
            args = sample_args[c_i]
            static_input_surface = per_callable_static_input_surfaces[c_i]
            for _ in range(num_warmup_iters):
                outputs, _ = _tree_flatten(func(*args))
                grad_inputs = torch.autograd.grad(
                    outputs=tuple(o for o in outputs if o.requires_grad),
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(torch.empty_like(o) for o in outputs if o.requires_grad),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )
            del outputs, grad_inputs
    torch.cuda.synchronize()

    # All captures here share a mempool. To avoid replays corrupting each other's memory,
    # the safest approach is to capture all passes in the same order they'll run:
    # fwd 1, fwd 2, ... fwd N, then bwd N, bwd N-1, ... bwd 1.

    if _order is not None:  # pylint: disable=too-many-nested-blocks
        per_callable_static_outputs = [None] * len(flatten_sample_args)
        per_callable_output_unflatten_spec = [None] * len(flatten_sample_args)
        per_callable_static_grad_outputs = [None] * len(flatten_sample_args)
        per_callable_static_grad_inputs = [None] * len(flatten_sample_args)
        fwd_idx = [0] * num_model_chunks
        bwd_idx = [0] * num_model_chunks
        for c_id in _order:
            if c_id > 0:
                # Capture forward graph for model chunk c_id, microbatch fwd_idx[c_id-1]
                m_chunk = c_id - 1
                for l_no in range(num_layers):
                    func = callables[m_chunk * num_layers + l_no]
                    per_callable_fwd_idx = (m_chunk * num_microbatches * num_layers) + (
                        fwd_idx[m_chunk] * num_layers + l_no
                    )
                    args = sample_args[per_callable_fwd_idx]
                    fwd_graph = fwd_graphs[per_callable_fwd_idx]
                    with torch.cuda.graph(fwd_graph, pool=mempool):
                        outputs = func(*args)
                    flatten_outputs, spec = _tree_flatten(outputs)
                    per_callable_static_outputs[per_callable_fwd_idx] = tuple(flatten_outputs)
                    per_callable_output_unflatten_spec[per_callable_fwd_idx] = spec
                    graph_callables[per_callable_fwd_idx] = func
                fwd_idx[m_chunk] += 1
            else:
                # Capture backward graph for model chunk c_id, microbatch bwd_idx[-c_id-1]
                m_chunk = -c_id - 1
                for l_no in list(reversed(range(num_layers))):
                    per_callable_bwd_idx = (m_chunk * num_microbatches * num_layers) + (
                        bwd_idx[m_chunk] * num_layers + l_no
                    )
                    static_input_surface = per_callable_static_input_surfaces[per_callable_bwd_idx]
                    static_outputs = per_callable_static_outputs[per_callable_bwd_idx]
                    bwd_graph = bwd_graphs[per_callable_bwd_idx]
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
                    # Pads out the actually-needed grads with Nones in gradient slots for inputs
                    # that don't require grad. I couldn't think of a one-liner for this pattern.
                    static_grad_inputs = []
                    grad_idx = 0
                    for arg in static_input_surface:
                        if arg.requires_grad:
                            static_grad_inputs.append(grad_inputs[grad_idx])
                            grad_idx += 1
                        else:
                            static_grad_inputs.append(None)  # type: ignore[arg-type]
                    static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]

                    per_callable_static_grad_outputs[per_callable_bwd_idx] = static_grad_outputs
                    per_callable_static_grad_inputs[per_callable_bwd_idx] = static_grad_inputs
                bwd_idx[m_chunk] += 1
    else:
        # Capture forward graphs
        per_callable_static_outputs = []
        per_callable_output_unflatten_spec = []
        graph_id = 0
        for func, args, fwd_graph in zip(callables, sample_args, fwd_graphs):
            with torch.cuda.graph(fwd_graph, pool=mempool):
                outputs = func(*args)
            graph_callables[graph_id] = func
            graph_id += 1

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
            def forward(ctx, skip_fp8_weight_update, *inputs):
                # At this stage, only the user args may (potentially) be new tensors.
                ctx.is_first_module = FP8GlobalStateManager.is_first_fp8_module()
                if ctx.is_first_module and skip_fp8_weight_update is not None:
                    FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(skip_fp8_weight_update)

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
                    FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

                # Input args that didn't require grad expect a None gradient.
                assert isinstance(static_grad_inputs, tuple)
                return (None,) + tuple(
                    b.detach() if b is not None else b for b in static_grad_inputs
                )

        def functionalized(*user_args, **user_kwargs):
            # Runs the autograd function with inputs == all
            # inputs to the graph that might require grad
            # (explicit user args + module parameters)
            # Assumes module params didn't change since capture.
            skip_fp8_weight_update = None
            if fp8_weight_caching:
                assert "is_first_microbatch" in user_kwargs and isinstance(
                    user_kwargs["is_first_microbatch"], bool
                ), "`is_first_microbatch` boolean kwarg must be provided for FP8 weight caching."

                skip_fp8_weight_update = not user_kwargs["is_first_microbatch"]

            flatten_user_args, _ = _tree_flatten(user_args)
            out = Graphed.apply(skip_fp8_weight_update, *(tuple(flatten_user_args) + module_params))
            return _tree_unflatten(out, output_unflatten_spec)

        return functionalized

    # Put together the final graphed callables
    ret = []
    for i in range(len(sample_args)):
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

        func = graph_callables[i]
        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
                def new_fwd(*user_args, **user_kwargs):
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        # Set the FP8 group from global amax reduction.
                        for m in func.modules():
                            if (
                                isinstance(m, TransformerEngineBaseModule)
                                and FP8GlobalStateManager.is_fp8_enabled()
                            ):
                                m.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
                                m.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()
                                FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                                    m.fp8_meta, fp8_weights=m._get_fp8_params()
                                )
                        return graphed(*user_args, **user_kwargs)
                    return orig_fwd(*user_args, **user_kwargs)

                return new_fwd

            forward = make_graphed_forward(func, func.training, graphed, func.forward)
            if _order is None:
                func.forward = forward
                ret.append(func)
            else:
                ret.append(forward)
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
    fp8_enabled=False,
    fp8_calibrating=False,
    fp8_recipe=None,
    fp8_weight_caching=False,
    _order=None,
):
    """
    A version of PyTorch's `make_graphed_callables` utility function with support for
    TransformerEngine modules and FP8. Please see the original version in upstream PyTorch
    `here <https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html>`_
    for extensive documentation. The documentation for additional parameters which are
    specific to FP8 are given below.

    FP8 specific parameters
    -----------------------
    fp8_enabled: bool, default = `True`
                 whether or not to enable fp8
    fp8_calibrating: bool, default = `False`
                     calibration mode allows collecting statistics such as amax and scale
                     data of fp8 tensors even when executing without fp8 enabled. This is
                     useful for saving an inference ready fp8 checkpoint while training
                     using a higher precision.
    fp8_recipe: recipe.DelayedScaling, default = `None`
                recipe used for FP8 training.
    fp8_weight_caching: bool, default = `False`
                        Whether or not to cache FP8 weights across microbatches. if set to `True`,
                        the `is_first_microbatch` boolean argument must be passed into the forward
                        method for TransformerEngine modules. When storing primary weights in FP8
                        using TE's `fp8_model_init` API and using an FP8 aware optimizer, this arg
                        must be set to `False` if calculating weight transposes' outside TE, e.g.,
                        in the optimizer step.
    """
    set_capture_start()

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
            with fp8_autocast(
                enabled=fp8_enabled, calibrating=fp8_calibrating, fp8_recipe=fp8_recipe, _graph=True
            ):
                outputs = old_forward(*args, **kwargs)
            return outputs

        block.forward = forward_func

    forward_funcs = []
    for module in modules:
        assert isinstance(module, torch.nn.Module), f"Graphing for {type(module)} is not supported."
        wrap_autocast(module)
        forward_funcs.append(module)

    if just_one_callable:
        forward_funcs = forward_funcs[0]
    else:
        forward_funcs = tuple(forward_funcs)

    # Save RNG state.
    if graph_safe_rng_available():
        generators = [
            torch.cuda.default_generators[torch.cuda.current_device()],
            *get_all_rng_states().values(),
        ]
        original_rng_states = [state.get_state() for state in generators]
    else:
        original_rng_states = torch.cuda.get_rng_state()

    graphed_callables = _make_graphed_callables(
        forward_funcs,
        sample_args,
        num_warmup_iters=num_warmup_iters,
        allow_unused_input=allow_unused_input,
        fp8_weight_caching=fp8_weight_caching,
        _order=_order,
    )

    # Ensures warmup does not affect numerics for ops such as dropout.
    if graph_safe_rng_available():
        for gen, state in zip(generators, original_rng_states):
            gen.set_state(state)
    else:
        torch.cuda.set_rng_state(original_rng_states)

    # Restore FP8 state.
    restore_fp8_tensors(modules, saved_fp8_tensors)

    set_capture_end()
    return graphed_callables
