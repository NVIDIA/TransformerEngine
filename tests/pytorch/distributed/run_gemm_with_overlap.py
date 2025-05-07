#!/usr/bin/python3

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import socket
import warnings
import subprocess
import argparse
import operator
from functools import partial, reduce

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.module.base import (
    fill_userbuffers_buffer_for_all_gather,
    get_cublas_workspace_size_bytes,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch_dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

nvte_comm_types = {
    "rs": tex.CommOverlapType.RS,
    "ag": tex.CommOverlapType.AG,
}


def _mapped_argtype(opt, typemap):
    if str(opt).lower() not in typemap.keys():
        raise TypeError(f"Unrecognized option! Please choose from: {typemap.keys()}")
    return typemap[str(opt).lower()]


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Test comm+GEMM overlap with Userbuffers.")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=1024, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=16, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=48, help="Dimension of each attention head."
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--quantization",
        type=str.lower,
        default="none",
        choices=["none", "fp8", "mxfp8"],
        help="Quantization recipe",
    )
    parser.add_argument(
        "--fp8-output", action="store_true", default=False, help="Get FP8 output from GEMM."
    )
    parser.add_argument(
        "--p2p", action="store_true", default=False, help="Test overlap with P2P comms."
    )
    parser.add_argument(
        "--atomic", action="store_true", default=False, help="Test overlap with atomic GEMM."
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        default=False,
        help="Aggregate 2X chunks for P2P split pipelined all-gather.",
    )
    parser.add_argument(
        "--comm-type",
        type=partial(_mapped_argtype, typemap=nvte_comm_types),
        default=tex.CommOverlapType.AG,
        help="Comm type to overlap.",
    )
    parser.add_argument(
        "--bulk-overlap",
        action="store_true",
        default=False,
        help="Enable bulk AG or RS overlap for a tensor that is not involved in the GEMM compute.",
    )
    parser.add_argument(
        "--check-numerics",
        action="store_true",
        default=False,
        help="Test numerical result against torch.matmul(...)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Run some warmup iterations of the comm+GEMM overlap before " + "the timing runs.",
    )
    parser.add_argument(
        "--timing-iters",
        type=int,
        default=1,
        help="Benchmark the comm+GEMM overlap as an average of many iterations.",
    )
    parser.add_argument(
        "--clock-speed",
        type=int,
        default=-1,
        help="Set device clock speed to a fixed value via `nvidia-smi`.",
    )
    parser.add_argument(
        "--std", type=float, default=0.023, help="Standard deviation for input and weight tensors."
    )
    parser.add_argument(
        "--tcp-init",
        action="store_true",
        default=False,
        help="Initialize torch.distributed with TcpStore.",
    )
    parser.add_argument(
        "--init-method", type=str, default=None, help="Set the torch.distributed init method."
    )
    parser.add_argument(
        "--bind-to-device",
        action="store_true",
        default=False,
        help=(
            "Initialize torch.distributed with 'device_id' argument to bind each rank to 1 device."
        ),
    )
    parser.add_argument(
        "--bootstrap-backend",
        type=str.lower,
        default="nccl",
        choices=["gloo", "mpi", "nccl"],
        help=(
            "PyTorch distributed backend for host tensor collectives during comm+GEMM overlap "
            + "initialization."
        ),
    )
    parser.add_argument(
        "--use-cuda-graphs", action="store_true", default=False, help="Use CUDA graphs."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Verbose info messages."
    )
    opts = parser.parse_args(argv, namespace)

    if opts.bulk_overlap:
        if opts.p2p:
            warnings.warn("Point-2-point comms are not supported with bulk overlap.")
            opts.p2p = False
        if opts.atomic:
            warnings.warn("Atomic GEMM is not supported with bulk overlap.")
            opts.atomic = False
        if opts.quantization != "none":
            warnings.warn("Bulk overlap is supported in FP8 but only tested in BF16.")
            opts.quantization = "none"
    elif opts.comm_type == tex.CommOverlapType.AG:
        if opts.atomic:
            setattr(opts, "atomic_rs_p2p", opts.p2p)
        opts.p2p = True

    if opts.atomic:
        if not te.fp8.check_fp8_support():
            assert opts.quantization == "none", "Atomic GEMM is only supported in FP8."
        opts.quantization = "fp8"

    if opts.fp8_output:
        assert ops.quantization == "fp8", "FP8 output is only supported with FP8 compute."

    return opts


@record
def _main(opts):
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        # Execution with `mpirun -np N`
        WORLD_RANK = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
        WORLD_SIZE = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
        opts.tcp_init = True
        opts.bootstrap_backend = "mpi"
    else:  # TORCHELASTIC, SLURM, etc...
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

    if result.stdout == "0":  # Extra checks for non-MNNVL platforms
        assert WORLD_SIZE == LOCAL_SIZE
        assert LOCAL_SIZE <= torch.cuda.device_count()

    # Fix clock speed
    torch.cuda.set_device(LOCAL_RANK)
    if opts.clock_speed > 0:
        subprocess.run(
            ["nvidia-smi", "-pm", "ENABLED", "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result = subprocess.run(
            ["nvidia-smi", "-lgc", str(opts.clock_speed), "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        msg = result.stdout.decode("utf-8").splitlines()[0]
        print(f"[rank:{LOCAL_RANK}] {msg}\n", end="", flush=True)

    # Info printout
    def dist_print(msg, src=None, info=False, error=False, section=False, group=None):
        group = dist.new_group() if group is None else group
        rank = dist.get_rank(group)
        stream = sys.stderr if error else sys.stdout
        if info or opts.verbose:
            if section:
                if rank == (0 if src is None else src):
                    stream.write("\n")
                dist.barrier(group)
            if src is None or rank == src:
                prefix = "[GLOBAL] " if src is not None else f"[rank:{rank}] "
                lines = msg.splitlines()
                msg = "\n".join(
                    [prefix + lines[0]] + [(" " * len(prefix)) + line for line in lines[1:]]
                )
                stream.write(msg + "\n")
            dist.barrier(group)

    # Initialize torch.distributed global process group and get TP group
    dist_init_kwargs = {
        "backend": "nccl",
        "rank": WORLD_RANK,
        "world_size": WORLD_SIZE,
    }
    if opts.tcp_init:
        if opts.init_method is not None:
            assert opts.init_method.startswith("tcp://")
            init_method = opts.init_method
        else:
            MASTER_ADDR = os.getenv("MASTER_ADDR", socket.gethostbyname(socket.gethostname()))
            MASTER_PORT = os.getenv("MASTER_PORT", "1234")
            init_method = f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
        dist_init_kwargs["init_method"] = init_method
    elif opts.init_method is not None:
        assert (
            opts.init_method.startswith("env://")
            or opts.init_method.startswith("file://")
            or opts.init_method.startswith("tcp://")
        )
        dist_init_kwargs["init_method"] = opts.init_method
    if opts.bind_to_device or opts.bootstrap_backend == "nccl":
        dist_init_kwargs["device_id"] = torch.device(f"cuda:{LOCAL_RANK}")
    assert dist.is_nccl_available()
    dist.init_process_group(**dist_init_kwargs)
    tp_group = dist.new_group(backend="nccl")
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    dist_print(
        f"Initialized default NCCL process group with {tp_size} GPUs",
        src=0,
        section=True,
        info=True,
        group=tp_group,
    )

    # Initialize backend used in bootstrapping Userbuffers
    if opts.bootstrap_backend == "gloo":
        assert dist.is_gloo_available()
    elif opts.bootstrap_backend == "mpi":
        assert dist.is_mpi_available()
    bootstrap_pg = dist.new_group(backend=opts.bootstrap_backend)
    dist_print(
        f'Bootstrapping comm+GEMM overlap with backend="{opts.bootstrap_backend}"',
        src=0,
        section=True,
        info=True,
        group=bootstrap_pg,
    )
    if WORLD_RANK == 0:
        print("\n", end="", flush=True)

    helper = (
        tex.CommOverlapHelper()
        if tex.ubuf_built_with_mpi()
        else tex.CommOverlapHelper(bootstrap_pg)
    )

    # Initialize userbuffers with (M, N) buffer
    # M = sequence * batch
    # N = hidden size
    hidden_size = opts.num_heads * opts.head_dim
    inp_shape = (opts.seq_length, opts.batch_size, hidden_size)
    outer_size = reduce(operator.mul, inp_shape[:-1], 1)
    buffer_dtype = torch.bfloat16
    if (
        opts.quantization != "none"
        and not opts.bulk_overlap
        and opts.comm_type == tex.CommOverlapType.AG
    ):
        buffer_dtype = torch.uint8
    ub_obj = (
        tex.CommOverlapP2P(
            (outer_size, hidden_size),
            buffer_dtype,
            helper,
            tp_size,  # Tensor-parallel group size (may be different than LOCAL_SIZE)
            opts.comm_type,
            set_sm_margin=opts.comm_type == tex.CommOverlapType.RS or opts.atomic,
            atomic_gemm=opts.atomic,
            aggregate=opts.aggregate,
            use_ce=not (opts.atomic and bool(int(os.getenv("NVTE_AG_P2P_MULTI_ATOMIC", "0")))),
        )
        if opts.p2p
        else tex.CommOverlap(
            (outer_size, hidden_size),
            buffer_dtype,
            helper,
            tp_size,  # Tensor-parallel group size (may be different than LOCAL_SIZE)
            atomic_gemm=opts.atomic,
        )
    )

    # Numerical check on AG + atomic GEMM requires testing an AG+RS pair
    ub_obj2 = None
    if opts.atomic and opts.comm_type == tex.CommOverlapType.AG and opts.check_numerics:
        ub_obj2 = (
            tex.CommOverlapP2P(
                (outer_size, hidden_size),
                torch.uint8 if opts.fp8_output else torch.bfloat16,
                helper,
                tp_size,  # Tensor-parallel group size (may be different than LOCAL_SIZE)
                tex.CommOverlapType.RS,
                set_sm_margin=True,
                atomic_gemm=True,
            )
            if opts.atomic_rs_p2p
            else tex.CommOverlap(
                (outer_size, hidden_size),
                torch.uint8 if opts.fp8_output else torch.bfloat16,
                helper,
                tp_size,  # Tensor-parallel group size (may be different than LOCAL_SIZE)
                atomic_gemm=True,
            )
        )

    # Figure out problem sizing:
    # M = sequence * batch
    # N = hidden size
    # K = MLP intermediate size (usually 4x hidden size)
    # P = number of devices for sequence/tensor parallelism
    # NOTE: TE-GEMM is set up to work with a transposed kernels and  non-transposed inputs.
    ffn_hidden_size = 4 * hidden_size
    if opts.bulk_overlap:
        # Bulk overlap weight and input tensors are not relevant so they're globally sized
        local_kernel_t_shape = (ffn_hidden_size, hidden_size)
        local_inp_shape = (outer_size, hidden_size)
        # Bulk overlap comm tensor is distributed for AG overlap only
        if opts.comm_type == tex.CommOverlapType.AG:
            bulk_inp_shape = (outer_size // tp_size, hidden_size)
        else:
            bulk_inp_shape = (outer_size, hidden_size)
    else:
        if opts.comm_type == tex.CommOverlapType.AG:
            # (M/P, N) -> overlapped AG -> (M, N) x (K/P, N)^T = (M, K/P)
            local_kernel_t_shape = (ffn_hidden_size // tp_size, hidden_size)
            local_inp_shape = (outer_size // tp_size, hidden_size)
            if ub_obj2 is not None:
                local_kernel2_t_shape = (hidden_size, ffn_hidden_size // tp_size)
        else:
            # (M, K/P) x (N, K/P)^T = (M, N) -> overlapped RS -> (M/P, N)
            local_kernel_t_shape = (hidden_size, ffn_hidden_size // tp_size)
            local_inp_shape = (outer_size, ffn_hidden_size // tp_size)

    # Initialize distributed input tensor and GEMM kernels
    torch.manual_seed(opts.seed + tp_rank)
    torch.cuda.manual_seed(opts.seed + tp_rank)
    inp = torch.nn.init.normal_(
        torch.empty(local_inp_shape, dtype=torch.bfloat16, device="cuda"),
        mean=0.0,
        std=opts.std,
    )
    kernel_t = torch.nn.init.normal_(
        torch.empty(local_kernel_t_shape, dtype=torch.bfloat16, device="cuda"),
        mean=0.0,
        std=opts.std,
    )
    if ub_obj2 is not None:
        kernel2_t = torch.nn.init.normal_(
            torch.empty(local_kernel2_t_shape, dtype=torch.bfloat16, device="cuda"),
            mean=0.0,
            std=opts.std,
        )

    # Allocate cuBLAS workspace
    workspace_size = 3 * get_cublas_workspace_size_bytes()
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device="cuda")

    # Gather global tensors and calculate reference result (need these first for Fp8 scales)
    if opts.bulk_overlap:
        ker_g = torch.transpose(kernel_t, 0, 1)
        inp_g = inp
        bulk_inp = torch.nn.init.normal_(
            torch.empty(bulk_inp_shape, dtype=torch.bfloat16, device="cuda"),
            mean=0.0,
            std=opts.std,
        )
    else:
        if opts.comm_type == tex.CommOverlapType.AG:
            # AG Kernel: (K/P, N) -> gather -> (K, N) -> T -> (N, K)
            ker_g = torch.transpose(
                te.distributed.gather_along_first_dim(kernel_t, tp_group)[0], 0, 1
            ).to(dtype=torch.float32)
            # AG Input: (M/P, N) -> gather -> (M, N)
            inp_g = te.distributed.gather_along_first_dim(inp, tp_group)[0].to(dtype=torch.float32)
            if ub_obj2 is not None:
                ker2_g = te.distributed.gather_along_first_dim(
                    torch.transpose(kernel2_t, 0, 1), tp_group
                )[0].to(dtype=torch.float32)
        else:
            # RS Kernel: (N, K/P) -> T -> (K/P, N) -> gather -> (K, N)
            ker_g = te.distributed.gather_along_first_dim(
                torch.transpose(kernel_t, 0, 1), tp_group
            )[0].to(dtype=torch.float32)
            # RS Input: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
            inp_g = torch.transpose(
                te.distributed.gather_along_first_dim(torch.transpose(inp, 0, 1), tp_group)[0], 0, 1
            ).to(dtype=torch.float32)

    if opts.bulk_overlap:
        if opts.comm_type == tex.CommOverlapType.AG:
            ref_g = te.distributed.gather_along_first_dim(bulk_inp, tp_group)[0]
        else:
            # First all-gather all the bulk inputs into a list
            bulk_inp_list = [torch.zeros_like(bulk_inp) for _ in range(tp_size)]
            dist.all_gather(bulk_inp_list, bulk_inp, tp_group)
            # Sum the list together for final global result
            ref_g = torch.stack(bulk_inp_list).sum(dim=0)
    else:
        ref_g = torch.matmul(inp_g, ker_g)
        if ub_obj2 is not None:
            inp2_g = torch.nn.functional.gelu(ref_g)  # pylint: disable=not-callable
            ref2_g = torch.matmul(inp2_g, ker2_g)

    # Initialize quantizers
    with_quantized_compute = opts.quantization != "none"
    inp_quantizer = None
    ker_quantizer = None
    out_quantizer = None
    bulk_inp_quantizer = None
    inp2_quantizer = None
    ker2_quantizer = None
    out2_quantizer = None
    if opts.quantization == "fp8":
        # Structure to maintain amax and scale/scale_inv information for the kernel and input
        num_gemms = 6 if ub_obj2 is not None else 3
        fp8_dtype = tex.DType.kFloat8E4M3
        fp8_scales = torch.ones(num_gemms, dtype=torch.float, device="cuda")
        fp8_amaxes = torch.zeros(num_gemms, dtype=torch.float, device="cuda")

        # Compute initial amaxes and scales
        inp_amax = torch.max(torch.abs(inp_g))
        fp8_amaxes[0].copy_(inp_amax)
        ker_amax = torch.max(torch.abs(ker_g))
        fp8_amaxes[1].copy_(ker_amax)
        ref_amax = torch.max(torch.abs(ref_g))
        fp8_amaxes[2].copy_(ref_amax)
        if opts.bulk_overlap and opts.comm_type == tex.CommOverlapType.RS:
            bulk_amax = torch.max(torch.abs(bulk_inp))
            fp8_amaxes[5].copy_(bulk_amax)
        elif ub_obj2 is not None:
            inp2_amax = torch.max(torch.abs(inp2_g))
            fp8_amaxes[3].copy_(inp2_amax)
            ker2_amax = torch.max(torch.abs(ker2_g))
            fp8_amaxes[4].copy_(ker2_amax)
            ref2_amax = torch.max(torch.abs(ref2_g))
            fp8_amaxes[5].copy_(ref2_amax)

        inp_quantizer = Float8Quantizer(fp8_scales[0].clone(), fp8_amaxes[0].clone(), fp8_dtype)
        ker_quantizer = Float8Quantizer(fp8_scales[1].clone(), fp8_amaxes[1].clone(), fp8_dtype)
        if opts.fp8_output:
            out_quantizer = Float8Quantizer(fp8_scales[2].clone(), fp8_amaxes[2].clone(), fp8_dtype)

        if opts.bulk_overlap and opts.comm_type == tex.CommOverlapType.RS:
            bulk_inp_quantizer = Float8Quantizer(
                fp8_scales[5].clone(), fp8_amaxes[5].clone(), fp8_dtype
            )
        elif ub_obj2 is not None:
            inp2_quantizer = Float8Quantizer(
                fp8_scales[3].clone(), fp8_amaxes[3].clone(), fp8_dtype
            )
            ker2_quantizer = Float8Quantizer(
                fp8_scales[4].clone(), fp8_amaxes[4].clone(), fp8_dtype
            )
            if opts.fp8_output:
                out2_quantizer = Float8Quantizer(
                    fp8_scales[5].clone(), fp8_amaxes[5].clone(), fp8_dtype
                )
    elif opts.quantization == "mxfp8":
        fp8_dtype = tex.DType.kFloat8E4M3
        inp_quantizer = MXFP8Quantizer(fp8_dtype, columnwise=False)
        ker_quantizer = MXFP8Quantizer(fp8_dtype)
        if opts.bulk_overlap and opts.comm_type == tex.CommOverlapType.RS:
            bulk_inp_quantizer = MXFP8Quantizer(fp8_dtype, columnwise=False)
        elif ub_obj2 is not None:
            inp2_quantizer = MXFP8Quantizer(fp8_dtype, columnwise=False)
            ker2_quantizer = MXFP8Quantizer(fp8_dtype)

    # Quantize tensors
    if with_quantized_compute:

        # Quantize input tensor
        inp_fp8 = inp_quantizer(inp)

        # Quantize kernel tensor
        kernel_t_fp8 = ker_quantizer(kernel_t)
        if opts.bulk_overlap and opts.comm_type == tex.CommOverlapType.RS:
            bulk_inp_fp8 = bulk_inp_quantizer(bulk_inp)
        elif ub_obj2 is not None:
            kernel2_t_fp8 = ker2_quantizer(kernel2_t)

        # Make sure the inputs are cast correctly
        if opts.check_numerics:
            torch.allclose(
                inp.to(dtype=torch.float32),
                inp_fp8.dequantize(dtype=torch.float32),
                rtol=0.125,
                atol=0.0675,
            )
            torch.allclose(
                kernel_t.to(dtype=torch.float32),
                kernel_t_fp8.dequantize(dtype=torch.float32),
                rtol=0.125,
                atol=0.0675,
            )
            if opts.bulk_overlap and opts.comm_type == tex.CommOverlapType.RS:
                torch.allclose(
                    bulk_inp.to(dtype=torch.float32),
                    bulk_inp_fp8.dequantize(dtype=torch.float32),
                    rtol=0.125,
                    atol=0.0675,
                )
            elif ub_obj2 is not None:
                torch.allclose(
                    kernel2_t.to(dtype=torch.float32),
                    kernel2_t_fp8.dequantize(dtype=torch.float32),
                    rtol=0.125,
                    atol=0.0675,
                )

    # Set up comm/compute buffers
    ag_out = None
    rs_out = None
    rs_out2 = None
    if opts.comm_type == tex.CommOverlapType.AG:
        if opts.bulk_overlap:
            ag_out, _ = fill_userbuffers_buffer_for_all_gather(
                ub_obj,
                bulk_inp,
                bulk_inp_quantizer,
                tp_group,
            )
            gemm_inp = inp
        else:
            ag_out, _ = fill_userbuffers_buffer_for_all_gather(
                ub_obj,
                inp_fp8 if with_quantized_compute else inp,
                inp_quantizer,
                tp_group,
            )
            gemm_inp = ag_out
        if ub_obj2 is not None:
            rs_out2 = torch.empty(
                (outer_size // tp_size, hidden_size), dtype=torch.bfloat16, device="cuda"
            )
    else:
        if opts.bulk_overlap:
            if opts.quantization == "none":
                ub_obj.copy_into_buffer(bulk_inp, local_chunk=False)
            if opts.quantization == "fp8":
                ub_obj.copy_into_buffer(bulk_inp_fp8._data, local_chunk=False)
            elif opts.quantization == "mxfp8":
                ub_obj.copy_into_buffer(bulk_inp_fp8._rowwise_data, local_chunk=False)

        gemm_inp = inp_fp8 if with_quantized_compute else inp
        rs_out = torch.empty(
            (outer_size // tp_size, hidden_size), dtype=torch.bfloat16, device="cuda"
        )

    # Wrap GEMM ops in condensed functions to make CUDA Graphs easier to use
    def _fp8_gemm():
        return tex.general_gemm(
            kernel_t_fp8,
            gemm_inp,
            workspace,
            out_dtype=torch.float8_e4m3fn if opts.fp8_output else torch.bfloat16,
            quantization_params=out_quantizer,
            use_split_accumulator=te.module.base._2X_ACC_FPROP,
            ub=ub_obj,
            ub_type=opts.comm_type,
            extra_output=rs_out,
            bulk_overlap=opts.bulk_overlap,
        )

    def _fp8_gemm2(gemm1_out):
        gemm2_inp = tex.gelu(
            (gemm1_out.dequantize() if opts.fp8_output else gemm1_out),
            inp2_quantizer,
        )
        return tex.general_gemm(
            kernel2_t_fp8,
            gemm2_inp,
            workspace,
            out_dtype=torch.float8_e4m3fn if opts.fp8_output else torch.bfloat16,
            quantization_params=out2_quantizer,
            use_split_accumulator=te.module.base._2X_ACC_FPROP,
            ub=ub_obj2,
            ub_type=tex.CommOverlapType.AG,
            extra_output=rs_out2,
        )

    def _gemm():
        return tex.general_gemm(
            kernel_t,
            gemm_inp,
            workspace,
            out_dtype=torch.bfloat16,
            use_split_accumulator=te.module.base._2X_ACC_FPROP,
            ub=ub_obj,
            ub_type=opts.comm_type,
            extra_output=rs_out,
            bulk_overlap=opts.bulk_overlap,
        )

    # Trigger GEMM
    total_iters = opts.warmup_iters + opts.timing_iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    torch.cuda.synchronize()

    if opts.use_cuda_graphs:
        # Trace the CUDA graph first
        g = torch.cuda.CUDAGraph()
        if with_quantized_compute:
            if ub_obj is None:
                with torch.cuda.graph(g):
                    all_outputs = _fp8_gemm()
            else:
                with torch.cuda.graph(g):
                    all_outputs = _fp8_gemm()
                    _ = _fp8_gemm2(all_outputs[0])
        else:
            with torch.cuda.graph(g):
                all_outputs = _gemm()

        # Now replay the CUDA graph in a loop
        for i in range(total_iters):
            start_events[i].record()
            g.replay()
            end_events[i].record()

    else:
        for i in range(total_iters):
            if with_quantized_compute:
                start_events[i].record()
                all_outputs = _fp8_gemm()
                end_events[i].record()
                if ub_obj2 is not None:
                    _fp8_gemm2(all_outputs[0])
            else:
                start_events[i].record()
                all_outputs = _gemm()
                end_events[i].record()

    torch.cuda.synchronize()
    gpu_times = [
        s.elapsed_time(e)
        for s, e in zip(start_events[opts.warmup_iters :], end_events[opts.warmup_iters :])
    ]

    avg_gpu_time = sum(gpu_times) / opts.timing_iters
    gemm_name = "".join(
        [
            "p2p all-gather + " if opts.comm_type == tex.CommOverlapType.AG else "",
            "atomic " if opts.atomic else "",
            "GEMM",
            (
                f" + {'p2p ' if opts.p2p else ''}reduce-scatter"
                if opts.comm_type == tex.CommOverlapType.RS
                else ""
            ),
        ]
    )
    timing_info = (
        f"Avg. GPU time for {gemm_name}: {avg_gpu_time} ms "
        + f"({opts.warmup_iters} warmup + {opts.timing_iters} timing runs)"
    )
    dist_print(timing_info, section=True, info=True, group=tp_group)

    # Compare against standard GEMM
    numerics_failed = False
    if opts.check_numerics:
        torch.cuda.synchronize()
        dist.barrier(tp_group)
        if opts.bulk_overlap:
            output_info = ""
            if opts.comm_type == tex.CommOverlapType.AG:
                # Bulk overlap AG output is already gathered
                test_out = ag_out

                if bulk_inp_quantizer is None:
                    test_out = ub_obj.get_buffer(False)
                else:
                    test_out = Float8Tensor(
                        shape=test_out.shape,
                        dtype=torch.bfloat16,
                        data=ub_obj.get_buffer(False),
                        fp8_scale=bulk_inp_quantizer.scale,
                        fp8_dtype=bulk_inp_quantizer.dtype,
                        quantizer=bulk_inp_quantizer,
                    )
            else:
                # Bulk overlap RS output needs to be gathered
                out_local = ub_obj.get_buffer(True)
                output_info += f"rs_output: {list(out_local.shape)} | "
                test_out = te.distributed.gather_along_first_dim(out_local, tp_group)[0]

            ref_out = ref_g
            output_info += f"output: {list(test_out.shape)} | reference: {list(ref_out.shape)}"
            dist_print(
                output_info,
                src=0 if opts.comm_type == tex.CommOverlapType.RS else None,
                section=True,
            )

            test_nonzeros = torch.count_nonzero(test_out)
            ref_nonzeros = torch.count_nonzero(ref_out)
            nonzero_info = (
                f"output nonzeros = {test_nonzeros} " + f"| reference count = {ref_nonzeros}"
            )
            dist_print(nonzero_info, src=0, section=True, group=tp_group)
        else:
            if opts.comm_type == tex.CommOverlapType.AG:
                if ub_obj2 is not None:
                    # AG+RS Output: (M/P, N) -> gather -> (M, N)
                    output = rs_out2.to(dtype=torch.float32)
                    test_out = te.distributed.gather_along_first_dim(output, tp_group)[0]
                else:
                    # AG Output: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
                    output = all_outputs[0].dequantize() if opts.fp8_output else all_outputs[0]
                    test_out = torch.transpose(
                        te.distributed.gather_along_first_dim(
                            torch.transpose(output, 0, 1), tp_group
                        )[0],
                        0,
                        1,
                    )
            else:
                # RS Output: (M/P, N) -> gather -> (M, N)
                output = rs_out.to(dtype=torch.float32)
                test_out = te.distributed.gather_along_first_dim(output, tp_group)[0]

            ref_out = ref2_g if ub_obj2 is not None else ref_g
            test_nonzeros = torch.count_nonzero(test_out)
            ref_nonzeros = torch.count_nonzero(ref_out)
            nonzero_info = (
                f"output nonzeros = {test_nonzeros} " + f"| reference count = {ref_nonzeros}"
            )
            dist_print(nonzero_info, src=0, section=True, group=tp_group)

            sizing_info = (
                f"input: {list(inp.shape)} " + f"| GEMM1 weights: {list(kernel_t.shape)[::-1]} "
            )
            if ub_obj2 is not None:
                sizing_info += f"| GEMM2 weights: {list(kernel2_t.shape)[::-1]} "
            sizing_info += f"| output: {list(output.shape)}\n"
            dist_print(sizing_info, section=True, group=tp_group)

            sizing_info_g = (
                f"input: {list(inp_g.shape)} " + f"| GEMM1 weights: {list(ker_g.shape)} "
            )
            if ub_obj2 is not None:
                sizing_info_g += f"| GEMM2 weights: {list(ker2_g.shape)} "
            sizing_info_g += (
                f"| output: {list(test_out.shape)} " + f"| reference: {list(ref_out.shape)}\n"
            )
            dist_print(sizing_info_g, src=0, group=tp_group)

        torch.cuda.synchronize()
        dist.barrier(tp_group)
        diff = torch.abs(test_out - ref_out).flatten()
        m = torch.argmax(diff)
        abs_err = diff[m].item()
        rel_err = abs_err / max(abs(ref_out.flatten()[m].item()), 1e-5)
        rtol = 0.02 if opts.quantization == "none" else 0.125
        atol = 0.001 if opts.quantization == "none" else 0.0625
        if rel_err > rtol and abs_err > atol:
            numerics_failed = True
            numerics_info = (
                "NUMERICAL CHECK FAILED: "
                + f"Outputs not close enough at index {m.item()} "
                + f"with {test_out.flatten()[m].item()} vs {ref_out.flatten()[m].item()} | "
                + f"rel. error = {rel_err} (tol = {rtol}) | "
                + f"abs. error = {abs_err} (tol = {atol})"
            )
        else:
            numerics_info = "NUMERICAL CHECK PASSED: "
            if rel_err <= rtol:
                numerics_info += f"rel. error = {rel_err} (tol = {rtol})" + (
                    " | " if abs_err < atol else ""
                )
            if abs_err <= atol:
                numerics_info += f"abs. error = {abs_err} (tol = {atol})"

        dist_print(
            numerics_info, src=0, section=True, info=True, error=numerics_failed, group=tp_group
        )

    dist.barrier(tp_group)
    if LOCAL_RANK == 0:
        print("\n", end="", flush=True)

    dist.destroy_process_group()

    # Reset clock speeds
    if opts.clock_speed > 0:
        subprocess.run(
            ["nvidia-smi", "-pm", "ENABLED", "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result = subprocess.run(
            ["nvidia-smi", "-rgc", "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return int(numerics_failed)


if __name__ == "__main__":
    sys.exit(_main(_parse_args()))
