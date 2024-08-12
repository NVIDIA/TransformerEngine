#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import socket
import argparse
import warnings
from functools import partial

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _te_layer_argtype(name):
    te_layers = [
        te.Linear,
        te.LayerNormLinear,
        te.LayerNormMLP,
        te.MultiheadAttention,
        te.TransformerLayer,
    ]
    layer_map = dict(zip([layer.__name__.lower() for layer in te_layers], te_layers))
    if name.lower() not in layer_map.keys():
        raise argparse.ArgumentTypeError(
            f"Invalid TE layer name! Please choose from: {layer_map.keys()}"
        )
    return layer_map[name.lower()]


def _get_layer_args(config, tp_group, tp_size, reference=False):
    hidden_size = config.num_heads * config.head_dim
    input_shape = [config.seq_length, config.batch_size, hidden_size]
    args = [hidden_size]
    kwargs = {
        "params_dtype": torch.float32,
        "device": "cuda",
        "tp_group": tp_group,
        "tp_size": tp_size,
        "sequence_parallel": True,
    }
    kwargs["ub_overlap_ag"] = not reference

    if config.layer_type is te.Linear:
        input_shape[2] = hidden_size // tp_size
        args.append(hidden_size)
        kwargs["parallel_mode"] = "row"
        kwargs["ub_overlap_rs"] = not reference
        kwargs["ub_name"] = "proj"
    else:
        input_shape[0] = config.seq_length // tp_size
        kwargs["ub_bulk_wgrad"] = not reference
        kwargs["ub_bulk_dgrad"] = not reference
        if config.layer_type is te.LayerNormLinear:
            args.append(3 * hidden_size)
            kwargs["parallel_mode"] = "column"
            kwargs["ub_name"] = "qkv"
        else:
            kwargs["set_parallel_mode"] = True
            kwargs["ub_overlap_rs"] = not reference
            if config.layer_type in [te.LayerNormMLP, te.TransformerLayer]:
                args.append(4 * hidden_size)
                kwargs["seq_length"] = config.seq_length
            if config.layer_type in [te.MultiheadAttention, te.TransformerLayer]:
                args.append(config.num_heads)
                kwargs["attention_dropout"] = 0.0
                kwargs["fuse_qkv_params"] = True
                if config.layer_type is te.MultiheadAttention:
                    kwargs["input_layernorm"] = True
                else:
                    kwargs["ub_tp_comm_overlap"] = not reference
                    kwargs["hidden_dropout"] = 0.0

    return args, kwargs, input_shape


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Test a Transformer Engine layer with GEMM+comm overlap via Userbuffers."
    )
    parser.add_argument("-l", "--layer-type", type=_te_layer_argtype, default=te.LayerNormMLP)
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=12, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=64, help="Dimension of each attention head."
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enables the te.fp8_autocast() context."
    )
    parser.add_argument(
        "--fp8-init", action="store_true", default=False, help="Initialize primary weights in FP8."
    )
    parser.add_argument(
        "--tcp-init",
        action="store_true",
        default=False,
        help="Initialize torch.distributed with TcpStore.",
    )
    parser.add_argument(
        "--bind-to-device",
        action="store_true",
        default=False,
        help="Initialize torch.distributed with `device_id` to bind each rank to a single device.",
    )
    parser.add_argument(
        "--bootstrap-backend",
        type=str.lower,
        default="nccl",
        choices=["gloo", "mpi", "nccl"],
        help="Communications backend for host tensor collectives during Userbuffers bootstrapping.",
    )
    parser.add_argument(
        "--use-cuda-graphs", action="store_true", default=False, help="Use CUDA Graphs."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print out additional debug information.",
    )
    args = parser.parse_args(argv, namespace)

    if args.use_cuda_graphs and args.layer_type in [te.MultiheadAttention, te.TransformerLayer]:
        warnings.warn(f"{args.layer_type.__name__} does not support CUDA Graphs!")
        args.use_cuda_graphs = False

    return args


def _compare_tensors(name, test, ref, rtol, atol):
    # Make sure tensors aren't zero and we don't pass trivially
    if test.count_nonzero() == 0:
        if ref.count_nonzero() == 0:
            warnings.warn(
                f"WARNING: {name} is a zero-tensor for both test and reference models!",
                category=RuntimeWarning,
            )
        else:
            numerics_info = (
                f"NUMERICAL CHECK FAILED: {name} is a zero-tensor but does not match reference!"
            )
            return 1, numerics_info

    diff = torch.abs(test - ref).flatten()
    m = torch.argmax(diff)
    abs_err = diff[m].item()
    rel_err = abs_err / max(abs(ref.flatten()[m].item()), 1e-5)
    numerics_failed = 0
    if rel_err > rtol and abs_err > atol:
        numerics_failed = 1
        numerics_info = (
            "NUMERICAL CHECK FAILED: "
            + f"{name} not close enough at index {m.item()} "
            + f"with {test.flatten()[m].item()} vs {ref.flatten()[m].item()} | "
            + f"rel. error = {rel_err} (tol = {rtol}) | "
            + f"abs. error = {abs_err} (tol = {atol})"
        )
    else:
        numerics_info = f"NUMERICAL CHECK PASSED: {name} | "
        if rel_err <= rtol:
            numerics_info += f"rel. error = {rel_err} (tol = {rtol})" + (
                " | " if abs_err <= atol else "."
            )
        if abs_err <= atol:
            numerics_info += f" abs. error = {abs_err} (tol = {atol})"

    return numerics_failed, numerics_info


def _train(opts):
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        # Execution with `mpirun -np N`
        WORLD_RANK = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
        WORLD_SIZE = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
        opts.tcp_init = True
        opts.bind_to_device = True
        opts.bootstrap_backend = "mpi"
    elif "TORCHELASTIC_RUN_ID" in os.environ:
        WORLD_RANK = int(os.getenv("RANK", "0"))
        WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    else:
        raise RuntimeError(f"{__file__} must be launched with either `mpirun` or `torchrun`!")
    assert LOCAL_SIZE == WORLD_SIZE

    def dist_print(msg, src=None, end="\n", debug=False, error=False):
        if debug and not opts.debug:
            return
        stream = sys.stderr if error else sys.stdout
        if WORLD_RANK == (0 if src is None else src):
            stream.write(f"[rank{WORLD_RANK}] {msg}{end}\n")
        dist.barrier()

    # Set device and initialize RNG states
    torch.cuda.set_device(WORLD_RANK)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    # Initialize torch.distributed global process group and get DP/TP groups
    dist_init_kwargs = {
        "backend": "nccl",
        "rank": WORLD_RANK,
        "world_size": WORLD_SIZE,
    }
    if opts.tcp_init:
        MASTER_ADDR = os.getenv("MASTER_ADDR", socket.gethostbyname(socket.gethostname()))
        MASTER_PORT = os.getenv("MASTER_PORT", "1234")
        dist_init_kwargs["init_method"] = f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
    if opts.bind_to_device or opts.bootstrap_backend == "nccl":
        dist_init_kwargs["device_id"] = torch.device(f"cuda:{LOCAL_RANK}")
    assert dist.is_nccl_available()
    dist.init_process_group(**dist_init_kwargs)
    nccl_world = dist.new_group(backend="nccl")
    dist_print(f"Initialized default NCCL process group with {WORLD_SIZE} GPUs")

    # Intialize userbuffers
    te.module.base.initialize_ub(
        [opts.seq_length * opts.batch_size, opts.num_heads * opts.head_dim],
        WORLD_SIZE,
        use_fp8=opts.fp8,
        dtype=torch.bfloat16,
        bootstrap_backend=opts.bootstrap_backend,
    )

    # Initialize the Transformer Engine layer with overlap
    args, kwargs, input_shape = _get_layer_args(opts, nccl_world, WORLD_SIZE)
    with te.fp8_model_init(enabled=opts.fp8_init):
        test_model = opts.layer_type(*args, **kwargs)
    dist_print("Initialized test model...", debug=True)

    # Initialize the reference model and copy all parameters
    ref_args, ref_kwargs, _ = _get_layer_args(opts, nccl_world, WORLD_SIZE, reference=True)
    with te.fp8_model_init(enabled=opts.fp8_init):
        ref_model = opts.layer_type(*ref_args, **ref_kwargs)
    dist_print("Initialized reference model...", debug=True)
    for test_param, ref_param in zip(test_model.parameters(), ref_model.parameters()):
        with torch.no_grad():
            ref_param.copy_(test_param)
        torch.testing.assert_close(test_param, ref_param, rtol=0.0, atol=0.0)
    dist_print("Copied parameters from test model to reference model...", debug=True)

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

    # Prepare random input tensors
    test_x = torch.randn(input_shape, dtype=torch.float32, device="cuda", requires_grad=True)
    test_x.retain_grad()
    ref_x = torch.empty_like(test_x).requires_grad_(True)
    with torch.no_grad():
        ref_x.copy_(test_x)
    torch.testing.assert_close(test_x, ref_x, rtol=0.0, atol=0.0)
    ref_x.retain_grad()

    # Execute fwd/bwd and collect tensors to test
    def run_fwd_bwd(model, x):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with te.fp8_autocast(enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=nccl_world):
                y = model(x)
                if isinstance(y, tuple):
                    out, *_ = y
                else:
                    out = y
        loss = out.sum()
        loss.backward()
        return out

    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(torch.device(f"cuda:{WORLD_RANK}"))
    if opts.use_cuda_graphs:
        test_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(test_graph):
            test_out = run_fwd_bwd(test_model, test_x)
        test_graph.replay()
        del test_graph
    else:
        test_out = run_fwd_bwd(test_model, test_x)
    test_grads = [test_out, test_x.grad]
    names = ["output", "input.grad"]
    for test_name, test_param in test_model.named_parameters():
        if test_param.requires_grad and "layer_norm" not in test_name:
            test_grads.append(test_param.grad)
            names.append(test_name + ".grad")

    torch.set_rng_state(torch_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, torch.device(f"cuda:{WORLD_RANK}"))
    if opts.use_cuda_graphs:
        ref_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ref_graph):
            ref_out = run_fwd_bwd(ref_model, ref_x)
        ref_graph.replay()
        del ref_graph
    else:
        ref_out = run_fwd_bwd(ref_model, ref_x)
    ref_grads = [ref_out, ref_x.grad]
    for ref_name, ref_param in ref_model.named_parameters():
        if ref_param.requires_grad and "layer_norm" not in ref_name:
            ref_grads.append(ref_param.grad)

    # Make sure we have the same number of gradients
    numerics_failed = torch.tensor([0], dtype=torch.uint8, device="cuda")
    if len(test_grads) != len(ref_grads):
        numerics_failed[0] = 1
        numerics_info = (
            "NUMERICAL CHECK FAILED: Incorrect number of gradients, "
            + f"expected {len(ref_grads)} but got {len(test_grads)}."
        )
        dist_print(numerics_info, src=WORLD_RANK, error=True)
    dist.all_reduce(numerics_failed, dist.ReduceOp.MAX, nccl_world)

    # Now validate accuracy
    if not bool(numerics_failed.item()):
        for i, (test_g, ref_g) in enumerate(zip(test_grads, ref_grads)):
            rtol = 0.125 if opts.fp8 else 0.025
            atol = 0.0625 if opts.fp8 else 0.00125
            grad_failed, grad_info = _compare_tensors(names[i], test_g, ref_g, rtol, atol)
            dist_print(grad_info, src=WORLD_RANK, error=grad_failed)
            numerics_failed[0] = int(grad_failed)
            dist.all_reduce(numerics_failed, dist.ReduceOp.MAX, nccl_world)
            if bool(numerics_failed.item()):
                break

    te.module.base.destroy_ub()
    dist_print("Destroying Userbuffers objects...", debug=True)

    dist_print("Destroying all process groups...", debug=True)
    dist.destroy_process_group()
    if opts.debug and WORLD_RANK == 0:
        print("Exiting...\n", end="", flush=True)

    return numerics_failed[0].item()


if __name__ == "__main__":
    sys.exit(_train(_parse_args()))
