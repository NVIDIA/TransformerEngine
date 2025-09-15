#!/usr/bin/python3

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import socket
import subprocess
import argparse
import warnings
import pprint
import yaml
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
    Format,
    MXFP8BlockScaling,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class multi_module_model(torch.nn.Module):
    def __init__(self, module, num_layers, *args, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList([module(*args, **kwargs) for _ in range(num_layers)])

    def forward(self, x, layer_contexts):
        for layer, context in zip(self.layers, layer_contexts):
            with context():
                x = layer(x)
        return x


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


def _get_layer_args(config, tp_group, tp_size, num_layers, reference=False):
    hidden_size = config.num_heads * config.head_dim
    ffn_hidden_size = 4 * hidden_size
    qkv_size = 3 * hidden_size
    if num_layers > 1 and config.layer_type != te.TransformerLayer:
        raise ValueError("Stacked layers are only supported for te.TransformerLayer!")
    input_shape = [config.seq_length, config.batch_size, hidden_size]
    args = [hidden_size]
    kwargs = {
        "params_dtype": torch.float32 if not config.use_bf16_params else torch.bfloat16,
        "device": "cuda",
        "tp_group": tp_group,
        "tp_size": tp_size,
        "sequence_parallel": True,
        "ub_overlap_ag": not reference,
        "ub_overlap_rs": not reference,
    }

    if config.layer_type in [te.Linear, te.LayerNormLinear]:
        if config.linear_parallel_mode == "row":
            input_shape[-1] = ffn_hidden_size // tp_size
            args = [ffn_hidden_size, hidden_size]
            if config.in_features is not None:
                input_shape[-1] = config.in_features // tp_size
                args = [config.in_features, hidden_size]
            kwargs["ub_name"] = "proj" if config.layer_type == te.Linear else "fc2"
            kwargs["ub_name"] = kwargs["ub_name"] if config.ub_name is None else config.ub_name
        elif config.linear_parallel_mode == "column":
            input_shape[0] = config.seq_length // tp_size
            if config.out_features is not None:
                args.append(config.out_features)
            else:
                args.append(qkv_size)
            kwargs["ub_name"] = "qkv" if config.ub_name is None else config.ub_name
            kwargs["ub_overlap_rs_dgrad"] = config.overlap_rs_dgrad and not reference
            kwargs["ub_bulk_dgrad"] = not config.overlap_rs_dgrad and not reference
            kwargs["ub_bulk_wgrad"] = not config.overlap_rs_dgrad and not reference
        kwargs["parallel_mode"] = config.linear_parallel_mode
    else:
        input_shape[0] = config.seq_length // tp_size
        if config.layer_type in [te.LayerNormMLP, te.TransformerLayer]:
            args.append(ffn_hidden_size)
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
        kwargs["set_parallel_mode"] = True
        kwargs["ub_overlap_rs_dgrad"] = config.overlap_rs_dgrad and not reference
        kwargs["ub_bulk_dgrad"] = not config.overlap_rs_dgrad and not reference
        kwargs["ub_bulk_wgrad"] = not config.overlap_rs_dgrad and not reference

    if config.ub_cfg is not None and isinstance(config.ub_cfg, str):
        with open(config.ub_cfg, "r") as stream:
            config.ub_cfg = yaml.safe_load(stream)
    return args, kwargs, input_shape


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Test a Transformer Engine layer with GEMM+comm overlap via Userbuffers."
    )
    parser.add_argument("-l", "--layer-type", type=_te_layer_argtype, default=te.LayerNormMLP)
    parser.add_argument(
        "--num-layers", type=int, default=1, help="Number of identical layers to stack."
    )
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=1024, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=16, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=48, help="Dimension of each attention head."
    )
    parser.add_argument(
        "--in-features",
        type=int,
        default=None,
        help="Optional input feature size for weight. Only used for Linear layer.",
    )
    parser.add_argument(
        "--out-features",
        type=int,
        default=None,
        help="Optional output feature size for weight. Only used for LayerNormLinear layer.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Optional tensor_model_parallel_size used to initialize UB.",
    )
    parser.add_argument(
        "--use-bf16-params",
        action="store_true",
        default=False,
        help="Use BF16 params instead of FP32.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enables the te.fp8_autocast() context."
    )
    parser.add_argument(
        "--quantization",
        type=str.lower,
        default="none",
        choices=["none", "fp8_delayed_scaling", "fp8_current_scaling", "mxfp8"],
        help="Quantization recipe",
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
        "--ub-cfg", type=str, default=None, help="Optional TP config yaml file input."
    )
    parser.add_argument("--ub-name", type=str, default=None, help="Optional TP layer name.")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        default=False,
        help="Skip numerics check.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Benchmark comm-gemm overlap perf.",
    )
    parser.add_argument(
        "--benchmark-iter",
        type=int,
        default=100,
        help="Number of iterations for benchmarking perf.",
    )
    parser.add_argument(
        "--linear-parallel-mode",
        type=str.lower,
        default="row",
        choices=["row", "column"],
        help="Parallel mode for te.Linear.",
    )
    parser.add_argument(
        "--overlap-rs-dgrad",
        action="store_true",
        default=False,
        help="Replace bulk DGRAD/WGRAD overlaps with DGRAD+RS in the backward pass for AG+GEMM.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print out additional debug information.",
    )
    parser.add_argument(
        "--first-last-layers-bf16",
        action="store_true",
        default=False,
        help="Use bf16 for first and last N layers.",
    )
    parser.add_argument(
        "--num-layers-at-start-in-bf16",
        type=int,
        default=0,
        help="Number of layers at the start to run in bf16.",
    )
    parser.add_argument(
        "--num-layers-at-end-in-bf16",
        type=int,
        default=0,
        help="Number of layers at the end to run in bf16.",
    )
    args = parser.parse_args(argv, namespace)

    if args.use_cuda_graphs and args.layer_type in [te.MultiheadAttention, te.TransformerLayer]:
        warnings.warn(f"{args.layer_type.__name__} does not support CUDA Graphs!")
        args.use_cuda_graphs = False

    if not args.first_last_layers_bf16 and (
        args.num_layers_at_start_in_bf16 > 0 or args.num_layers_at_end_in_bf16 > 0
    ):
        warnings.warn(
            "num-layers-at-start-in-bf16 and num-layers-at-end-in-bf16 are only supported when"
            " first-last-layers-bf16 is enabled!"
        )
        args.num_layers_at_start_in_bf16 = 0
        args.num_layers_at_end_in_bf16 = 0

    if args.num_layers_at_start_in_bf16 + args.num_layers_at_end_in_bf16 > args.num_layers:
        raise ValueError(
            "num-layers-at-start-in-bf16 + num-layers-at-end-in-bf16 must be less than or equal to"
            " num-layers!"
        )

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

    diff = torch.abs(test.flatten() - ref.flatten())
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
    else:
        WORLD_RANK = int(os.getenv("RANK", "0"))
        WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", str(torch.cuda.device_count())))

    result = subprocess.run(
        "nvidia-smi -q | grep -m1 CliqueId | awk '{printf $3}'",
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.stdout == "0" and opts.tp is None:  # Extra checks for non-MNNVL platforms
        assert WORLD_SIZE == LOCAL_SIZE

    # Initialize torch.distributed tp process group
    new_group_kwargs = {
        "backend": "nccl",
    }
    if opts.tp is not None:
        LOCAL_SIZE = opts.tp
        tp_base_rank = (WORLD_RANK // LOCAL_SIZE) * LOCAL_SIZE
        tp_rank_list = list(range(tp_base_rank, tp_base_rank + LOCAL_SIZE))
        new_group_kwargs = {
            "backend": "nccl",
            "ranks": tp_rank_list,
            "pg_options": dist.ProcessGroupNCCL.Options(is_high_priority_stream=True),
        }
    else:
        opts.tp = WORLD_SIZE

    # Tensor dim overrides for tensors that do not require TP communication
    if opts.in_features is not None:
        assert opts.layer_type is te.Linear and opts.linear_parallel_mode == "row", (
            "--in-features is only used to configure row-tensor-parallel Linear layers. Use"
            " --num-heads or --head-dim for other cases."
        )
    if opts.out_features is not None:
        assert opts.layer_type is te.LayerNormLinear and opts.linear_parallel_mode == "column", (
            "--out-features is only used to configure column-tensor-parallel LayerNormLinear"
            " layers. Use --num-heads or --head-dim for other cases."
        )

    def dist_print(msg, src=None, end="\n", debug=False, error=False):
        if debug and not opts.debug:
            return
        stream = sys.stderr if error else sys.stdout
        if WORLD_RANK == (0 if src is None else src):
            stream.write(f"[rank{WORLD_RANK}] {msg}{end}\n")
        dist.barrier()

    # Set device and initialize RNG states
    torch.cuda.set_device(LOCAL_RANK)
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
    nccl_world = dist.new_group(**new_group_kwargs)
    dist_print(f"Initialized default NCCL process group with {WORLD_SIZE} GPUs")

    # Initialize the Transformer Engine layer with overlap
    args, kwargs, input_shape = _get_layer_args(
        opts, nccl_world, opts.tp, num_layers=opts.num_layers
    )
    # Intialize userbuffers
    ub_cfgs = None
    if opts.overlap_rs_dgrad:
        ub_cfgs = {
            "qkv_dgrad": {"method": "ring_exchange"},
            "fc1_dgrad": {"method": "ring_exchange"},
        }

    quantization_modes = [
        (
            te.module.base.UserBufferQuantizationMode.FP8
            if opts.fp8
            else te.module.base.UserBufferQuantizationMode.NONE
        )
    ]
    if opts.first_last_layers_bf16 and opts.fp8:
        quantization_modes.append(te.module.base.UserBufferQuantizationMode.NONE)

    te.module.base.initialize_ub(
        [opts.seq_length * opts.batch_size, opts.num_heads * opts.head_dim],
        opts.tp,
        quantization_modes=quantization_modes,
        dtype=torch.bfloat16,
        bootstrap_backend=opts.bootstrap_backend,
        ub_cfgs=ub_cfgs if opts.ub_cfg is None else opts.ub_cfg,
    )

    with te.fp8_model_init(enabled=opts.fp8_init):
        test_model = multi_module_model(opts.layer_type, opts.num_layers, *args, **kwargs)
    dist_print("Initialized test model...", debug=True)
    if WORLD_RANK == 0:
        pprint.pprint(kwargs)
        sys.stdout.write("\n")
    dist.barrier()

    # Initialize the reference model and copy all parameters
    ref_args, ref_kwargs, _ = _get_layer_args(
        opts, nccl_world, opts.tp, num_layers=opts.num_layers, reference=True
    )
    with te.fp8_model_init(enabled=opts.fp8_init):
        ref_model = multi_module_model(opts.layer_type, opts.num_layers, *ref_args, **ref_kwargs)
    dist_print("Initialized reference model...", debug=True)
    for test_param, ref_param in zip(test_model.parameters(), ref_model.parameters()):
        with torch.no_grad():
            ref_param.copy_(test_param)
        torch.testing.assert_close(test_param, ref_param, rtol=0.0, atol=0.0)
    dist_print("Copied parameters from test model to reference model...", debug=True)

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = None
    if opts.quantization == "fp8_delayed_scaling":
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max"
        )
    elif opts.quantization == "fp8_current_scaling":
        fp8_recipe = Float8CurrentScaling(fp8_format=fp8_format)
    elif opts.quantization == "mxfp8":
        fp8_recipe = MXFP8BlockScaling()

    layer_contexts = [
        (
            partial(te.fp8_autocast, enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=nccl_world)
            if opts.num_layers_at_start_in_bf16 <= i
            and i < (opts.num_layers - opts.num_layers_at_end_in_bf16)
            else nullcontext
        )
        for i in range(opts.num_layers)
    ]

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
            y = model(x, layer_contexts)
            if isinstance(y, tuple):
                out, *_ = y
            else:
                out = y
            loss = out.sum()
            loss.backward()
        return out

    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(torch.device(f"cuda:{LOCAL_RANK}"))
    if opts.use_cuda_graphs:
        test_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(test_graph):
            test_out = run_fwd_bwd(test_model, test_x)
        test_graph.replay()
        if not opts.benchmark:
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
    torch.cuda.set_rng_state(cuda_rng_state, torch.device(f"cuda:{LOCAL_RANK}"))
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

    numerics_failed = torch.tensor([0], dtype=torch.uint8, device="cuda")
    if not opts.skip_verify:
        # Make sure we have the same number of gradients
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
                if bool(numerics_failed.item()) and not opts.debug:
                    break

    if opts.benchmark:
        # Warmup to not profile CPU overhead
        for _ in range(100):
            if opts.use_cuda_graphs:
                test_graph.replay()
            else:
                test_out = run_fwd_bwd(test_model, test_x)
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(opts.benchmark_iter):
            if opts.use_cuda_graphs:
                test_graph.replay()
            else:
                test_out = run_fwd_bwd(test_model, test_x)
        torch.cuda.cudart().cudaProfilerStop()
        if opts.use_cuda_graphs:
            del test_graph

    torch.cuda.synchronize()
    te.module.base.destroy_ub()
    dist_print("Destroying Userbuffers objects...", debug=True)

    dist_print("Destroying all process groups...", debug=True)
    dist.destroy_process_group()
    if opts.debug and WORLD_RANK == 0:
        print("Exiting...\n", end="", flush=True)

    return numerics_failed[0].item()


if __name__ == "__main__":
    sys.exit(_train(_parse_args()))
