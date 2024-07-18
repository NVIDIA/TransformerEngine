#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
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
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.fp8 import _default_sf_compute

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch_dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

nvte_comm_types = {
    "ag": tex.NVTE_Comm_Overlap_Type.AG,
    "rs": tex.NVTE_Comm_Overlap_Type.RS,
}


def _mapped_argtype(opt, typemap):
    if str(opt).lower() not in typemap.keys():
        raise TypeError(f"Unrecognized option! Please choose from: {typemap.keys()}")
    return typemap[str(opt).lower()]


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Test comm+GEMM overlap with Userbuffers.")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=64, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=128, help="Dimension of each attention head."
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enable FP8 GEMM."
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
        default=tex.NVTE_Comm_Overlap_Type.AG,
        help="Comm type to overlap.",
    )
    parser.add_argument(
        "--bulk-overlap",
        action="store_true",
        default=False,
        help="Enable bulk AG or RS overlap for a tensor that is not involved in the GEMM compute."
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
        "--scale", type=float, default=1e-2, help="Set scaling factor for input and weight tensors."
    )
    parser.add_argument(
        "--tcp-init",
        action="store_true",
        default=False,
        help="Initialize torch.distributed with TcpStore."
    )
    parser.add_argument(
        "--init-method", type=str, default=None, help="Set the torch.distributed init method."
    )
    parser.add_argument(
        "--bind-to-device",
        action="store_true",
        default=False,
        help="Initialize torch.distributed with 'device_id' argument to bind each rank to 1 device."
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
    parser.add_argument("--use-cuda-graphs", action="store_true", default=False,)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    opts = parser.parse_args(argv, namespace)

    if opts.bulk_overlap:
        if opts.p2p:
            warnings.warn("Point-2-point comms are not supported with bulk overlap.")
            opts.p2p = False
        if opts.atomic:
            warnings.warn("Atomic GEMM is not supported with bulk overlap.")
            opts.atomic = False
        if opts.fp8:
            warnings.warn("Bulk overlap is supported in FP8 but only tested in BF16.")
            opts.fp8 = False
    elif opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG and not opts.p2p:
        warnings.warn("All-gather overlap is only supported with point-2-point comms.")
        opts.p2p = True

    if opts.atomic:
        if not te.fp8.check_fp8_support():
            assert not opts.fp8, "Atomic GEMM is only supported in FP8."
        elif not opts.fp8:
            warnings.warn("Atomic GEMM is only supported in FP8.")
            opts.fp8 = True

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
    elif "TORCHELASTIC_RUN_ID" in os.environ:
        WORLD_RANK = int(os.getenv("RANK", "0"))
        WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    else:
        raise RuntimeError(f"{__file__} must be launched with either `mpirun` or `torchrun`!")
    assert WORLD_SIZE == LOCAL_SIZE  # this test supports only 1 node
    assert LOCAL_SIZE <= torch.cuda.device_count()

    # Fix clock speed
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
        print(f"[rank:{WORLD_RANK}] {msg}\n", end="", flush=True)

    # Info printout
    def dist_print(msg, src=None, info=False, section=False, group=None):
        group = dist.new_group() if group is None else group
        rank = dist.get_rank(group)
        if info or opts.verbose:
            if section:
                if rank == (0 if src is None else src):
                    print("\n", end="", flush=True)
                dist.barrier(group)
            if src is None or rank == src:
                prefix = "[GLOBAL] " if src is not None else f"[rank:{rank}] "
                lines = msg.splitlines()
                msg = "\n".join(
                    [prefix + lines[0]] + [(" " * len(prefix)) + line for line in lines[1:]]
                )
                print(msg + "\n", end="", flush=True)
            dist.barrier(group)

    # Initialize torch.distributed global process group and get TP group
    torch.cuda.set_device(LOCAL_RANK)
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
    if opts.bind_to_device:
        dist_init_kwargs["device_id"] = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(**dist_init_kwargs)
    tp_group = dist.new_group(backend="nccl")
    tp_size = dist.get_world_size(tp_group)
    dist_print(
        f"Initialized default NCCL process group with {tp_size} GPUs",
        src=0,
        section=True,
        info=True,
        group=tp_group,
    )

    # torch.distributed callback wrappers for bootstrapping userbuffers
    if opts.bootstrap_backend == "gloo":
        assert dist.is_gloo_available()
    elif opts.bootstrap_backend == "mpi":
        assert dist.is_mpi_available()
    bootstrap_pg = dist.new_group(backend=opts.bootstrap_backend)
    dist_print(f"Bootstrapping comm+GEMM overlap with backend=\"{opts.bootstrap_backend}\"", src=0,
               info=True, section=True)

    # torch.distributed callback wrappers for bootstrapping userbuffers
    def allgather_callback(global_data: torch.Tensor, local_data: torch.Tensor, group: str):
        del group
        global_tmp = global_data.cuda() if opts.bootstrap_backend == "nccl" else global_data
        local_tmp = local_data.cuda() if opts.bootstrap_backend == "nccl" else local_data
        dist.all_gather_into_tensor(global_tmp, local_tmp, group=bootstrap_pg)
        if opts.bootstrap_backend == "nccl":
            global_data.copy_(global_tmp.cpu())
            global_tmp.data = torch.Tensor()
            local_tmp.data = torch.Tensor()

    def bcast_callback(data: torch.Tensor, src: int, group: str):
        del group
        data_tmp = data.cuda() if opts.bootstrap_backend == "nccl" else data
        dist.broadcast(data_tmp, src, pg=bootstrap_pg)
        if opts.bootstrap_backend == "nccl":
            data.copy_(data_tmp.cpu())
            data_tmp = torch.Tensor()

    def barrier_callback(group: str):
        del group
        dist.barrier(group=bootstrap_pg)

    tex.set_comm_overlap_callbacks(allgather_callback, bcast_callback, barrier_callback)

    if opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS:
        if opts.bulk_overlap:
            ub_algo = tex.NVTE_Comm_Overlap_Algo.BULK_OVERLAP_RS
        elif opts.p2p:
            ub_algo = (
                tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_RS_P2P
                if opts.atomic
                else tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_RS_P2P
            )
        else:
            ub_algo = (
                tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_RS
                if opts.atomic
                else tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_RS
            )
    elif opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
        if opts.bulk_overlap:
            ub_algo = tex.NVTE_Comm_Overlap_Algo.BULK_OVERLAP_AG
        else:
            ub_algo = (
                tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_AG_P2P
                if opts.atomic
                else tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_AG_P2P
            )
    else:
        raise TypeError("Invalid comm+GEMM overlap type!")

    # Initialize userbuffers with (M, N) buffer
    # M = sequence * batch
    # N = hidden size
    hidden_size = opts.num_heads * opts.head_dim
    inp_shape = (opts.seq_length, opts.batch_size, hidden_size)
    outer_size = reduce(operator.mul, inp_shape[:-1], 1)
    ubuf_dtype = (
        torch.uint8
        if opts.fp8 and opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG
        else torch.bfloat16
    )
    sample_buffer = torch.empty((outer_size, hidden_size), dtype=ubuf_dtype, device="cuda")
    ub_obj = ub_obj = (
        tex.CommGemmOverlapP2P(
            sample_buffer,
            WORLD_RANK,
            WORLD_SIZE,
            LOCAL_RANK,
            LOCAL_SIZE,
            0,
            1,
            tp_size,
            tex.NVTE_COMM_OVERLAP_MAX_STREAMS,
            1,  # cga_size
            1,  # num_comm_sms
            opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS or opts.atomic,  # set_sm_margin
            True,  # use_ce
            opts.atomic,
            opts.aggregate,
            opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS,  # is_reduce_scatter
        )
        if opts.p2p
        else tex.CommGemmOverlap(
            sample_buffer,
            WORLD_RANK,
            WORLD_SIZE,
            LOCAL_RANK,
            LOCAL_SIZE,
            0,
            1,
            tp_size,
            4,  # num_splits
            tex.NVTE_COMM_OVERLAP_MAX_STREAMS,
            2,  # cga_size
            16,  # num_comm_sms
            True,  # set_sm_margin
            False,  # use_ce
            opts.atomic,
        )
    )

    # Numerical check on AG + atomic GEMM requires testing an AG+RS pair
    ub_obj2 = None
    if opts.atomic and opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG and opts.check_numerics:
        sample_buffer2 = torch.empty((outer_size, hidden_size), dtype=ubuf_dtype, device="cuda")
        ub_obj2 = tex.CommGemmOverlapP2P(
            sample_buffer2,
            WORLD_RANK,
            WORLD_SIZE,
            LOCAL_RANK,
            LOCAL_SIZE,
            0,
            1,
            LOCAL_SIZE,
            tex.NVTE_COMM_OVERLAP_MAX_STREAMS,
            1,  # num_comm_sms
            1,  # cga_size
            True,  # set_sm_margin
            True,  # use_ce
            True,  # atomic
            False,  # aggregate
            True,  # is_reduce_scatter
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
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
            bulk_inp_shape = (outer_size // tp_size, hidden_size)
        else:
            bulk_inp_shape = (outer_size, hidden_size)
    else:
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
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
    torch.manual_seed(opts.seed + WORLD_RANK)
    torch.cuda.manual_seed(opts.seed + WORLD_RANK)
    inp = torch.mul(torch.rand(local_inp_shape, dtype=torch.bfloat16, device="cuda"), opts.scale)
    kernel_t = torch.mul(
        torch.rand(local_kernel_t_shape, dtype=torch.bfloat16, device="cuda"), opts.scale
    )
    if ub_obj2 is not None:
        kernel2_t = torch.mul(
            torch.rand(local_kernel2_t_shape, dtype=torch.bfloat16, device="cuda"), opts.scale
        )

    # Gather global tensors and calculate reference result (need these first for Fp8 scales)
    if opts.bulk_overlap:
        ker_g = torch.transpose(kernel_t, 0, 1)
        inp_g = inp
        bulk_inp = torch.mul(
            torch.rand(bulk_inp_shape, dtype=torch.bfloat16, device="cuda"), opts.scale
        )
    else:
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
            # AG Kernel: (K/P, N) -> gather -> (K, N) -> T -> (N, K)
            ker_g = torch.transpose(
                te.distributed.gather_along_first_dim(kernel_t, tp_group)[0], 0, 1
            )
            # AG Input: (M/P, N) -> gather -> (M, N)
            inp_g = te.distributed.gather_along_first_dim(inp, tp_group)[0]
            if ub_obj2 is not None:
                ker2_g = te.distributed.gather_along_first_dim(
                    torch.transpose(kernel2_t, 0, 1),
                    tp_group
                )[0]
        else:
            # RS Kernel: (N, K/P) -> T -> (K/P, N) -> gather -> (K, N)
            ker_g = te.distributed.gather_along_first_dim(
                torch.transpose(kernel_t, 0, 1), tp_group
            )[0]
            # RS Input: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
            inp_g = torch.transpose(
                te.distributed.gather_along_first_dim(torch.transpose(inp, 0, 1), tp_group)[0], 0, 1
            )

    if opts.bulk_overlap:
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
            ref_g = te.distributed.gather_along_first_dim(bulk_inp, tp_group)[0]
        else:
            # First all-gather all the bulk inputs into a list
            bulk_inp_list = [ torch.zeros_like(bulk_inp) for _ in range(LOCAL_SIZE) ]
            dist.all_gather(bulk_inp_list, bulk_inp, tp_group)
            # Sum the list together for final global result
            ref_g = torch.stack(bulk_inp_list).sum(dim=0)
    else:
        ref_g = torch.matmul(inp_g, ker_g)
        if ub_obj2 is not None:
            inp2_g = torch.mul(ref_g, opts.scale)
            ref2_g = torch.matmul(inp2_g, ker2_g)

    if opts.fp8:
        fp8_formats = {
            tex.DType.kFloat8E4M3: Format.E4M3,
            tex.DType.kFloat8E5M2: Format.E5M2,
        }

        # Structure to maintain amax and scale/scale_inv information for the kernel and input
        fp8_dtype = tex.DType.kFloat8E4M3
        fp8_meta = tex.FP8TensorMeta()
        num_gemms = 6 if ub_obj2 is not None else 3
        fp8_meta.amax_history = torch.zeros((2, num_gemms),
                                            dtype=torch.float, device="cuda")
        fp8_meta.scale = torch.ones(num_gemms, dtype=torch.float, device="cuda")
        fp8_meta.scale_inv = torch.ones(num_gemms, dtype=torch.float, device="cuda")

        # Compute initial amaxes and scales
        inp_amax = torch.max(torch.abs(inp_g))
        fp8_meta.amax_history[1][tex.FP8FwdTensors.GEMM1_INPUT].copy_(inp_amax)
        ker_amax = torch.max(torch.abs(ker_g))
        fp8_meta.amax_history[1][tex.FP8FwdTensors.GEMM1_WEIGHT].copy_(ker_amax)
        ref_amax = torch.max(torch.abs(ref_g))
        fp8_meta.amax_history[1][tex.FP8FwdTensors.GEMM1_OUTPUT].copy_(ref_amax)
        if ub_obj2 is not None:
            inp2_amax = torch.max(torch.abs(inp2_g))
            fp8_meta.amax_history[1][tex.FP8FwdTensors.GEMM2_INPUT].copy_(inp2_amax)
            ker2_amax = torch.max(torch.abs(ker2_g))
            fp8_meta.amax_history[1][tex.FP8FwdTensors.GEMM2_WEIGHT].copy_(ker2_amax)
            ref2_amax = torch.max(torch.abs(ref2_g))
            fp8_meta.amax_history[1][tex.FP8FwdTensors.GEMM2_OUTPUT].copy_(ref2_amax)
        fp8_meta.scale = _default_sf_compute(
            fp8_meta.amax_history[1], fp8_meta.scale, fp8_formats[fp8_dtype].value.max_fwd, 1
        )
        fp8_meta.scale_inv = torch.reciprocal(fp8_meta.scale)

        # Cast input to Float8Tensor
        inp_fp8 = tex.cast_to_fp8(inp, fp8_meta, tex.FP8FwdTensors.GEMM1_INPUT, fp8_dtype)

        # Cast kernel to Float8Tensor
        kernel_t_fp8 = tex.cast_to_fp8(
            kernel_t, fp8_meta, tex.FP8FwdTensors.GEMM1_WEIGHT, fp8_dtype
        )
        if ub_obj2 is not None:
            kernel2_t_fp8 = tex.cast_to_fp8(
                kernel2_t, fp8_meta, tex.FP8FwdTensors.GEMM2_WEIGHT, fp8_dtype
            )

        # Make sure the inputs are cast correctly
        if opts.check_numerics:
            torch.allclose(
                inp.to(dtype=torch.float32),
                inp_fp8 * fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
                rtol=0.125,
                atol=0.0675,
            )
            torch.allclose(
                kernel_t.to(dtype=torch.float32),
                kernel_t_fp8 * fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM1_WEIGHT],
                rtol=0.125,
                atol=0.0675,
            )
            if ub_obj2 is not None:
                torch.allclose(
                    kernel2_t.to(dtype=torch.float32),
                    kernel2_t_fp8 * fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM2_WEIGHT],
                    rtol=0.125,
                    atol=0.0675
                )

        # Set Fp8 scales for userbuffers
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
            ub_obj.set_ubuf_scale_inv(fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM1_INPUT])
            if ub_obj2 is not None:
                ub_obj2.set_ubuf_scale_inv(fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM2_OUTPUT])
        else:
            ub_obj.set_ubuf_scale_inv(fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM1_OUTPUT])

    # Set up comm/compute buffers
    ubuf_out2 = None
    rs_out2 = None
    if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
        if opts.bulk_overlap:
            ub_obj.copy_input_to_ubuf(bulk_inp, True)
            gemm_inp = inp
        else:
            ub_obj.copy_input_to_ubuf(inp_fp8 if opts.fp8 else inp, True)
            gemm_inp = ub_obj.get_ubuf_output(tex.NVTE_Comm_Overlap_Type.AG)
        ubuf_out = None
        rs_out = None
        if ub_obj2 is not None:
            ubuf_out2 = ub_obj2.get_ubuf_output(tex.NVTE_Comm_Overlap_Type.AG)
            rs_out2 = torch.empty(
                (outer_size // tp_size, hidden_size), dtype=torch.bfloat16, device="cuda"
            )
    else:
        if opts.bulk_overlap:
            ub_obj.copy_input_to_ubuf(bulk_inp, False)
            ubuf_out = None
        else:
            ubuf_out = ub_obj.get_ubuf_output(tex.NVTE_Comm_Overlap_Type.AG)
        gemm_inp = inp_fp8 if opts.fp8 else inp
        rs_out = torch.empty(
            (outer_size // tp_size, hidden_size), dtype=torch.bfloat16, device="cuda"
        )

    # Wrap GEMM ops in condensed functions to make CUDA Graphs easier to use
    def _fp8_gemm():
        return tex.fp8_gemm(
            kernel_t_fp8,
            fp8_meta.scale_inv,
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype,
            gemm_inp,
            fp8_meta.scale_inv,
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype,
            torch.bfloat16,
            te.module.base.get_workspace(),
            bias=None,
            use_bias=False,
            gelu=False,
            use_split_accumulator=te.module.base._2X_ACC_FPROP,
            ub_algo=ub_algo,
            ub=ub_obj,
            extra_output_tensor=rs_out,
            out=ubuf_out,
        )

    def _fp8_gemm2(gemm1_out):
        gemm2_inp = tex.cast_to_fp8(
            torch.mul(gemm1_out, opts.scale),
            fp8_meta,
            tex.FP8FwdTensors.GEMM2_INPUT,
            fp8_dtype,
        )
        return tex.fp8_gemm(
            kernel2_t_fp8,
            fp8_meta.scale_inv,
            tex.FP8FwdTensors.GEMM2_WEIGHT,
            fp8_dtype,
            gemm2_inp,
            fp8_meta.scale_inv,
            tex.FP8FwdTensors.GEMM2_INPUT,
            fp8_dtype,
            torch.bfloat16,
            te.module.base.get_workspace(),
            bias=None,
            use_bias=False,
            gelu=False,
            use_split_accumulator=te.module.base._2X_ACC_FPROP,
            ub_algo=tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_RS_P2P,
            ub=ub_obj2,
            extra_output_tensor=rs_out2,
            out=ubuf_out2
        )

    def _gemm():
        return tex.gemm(
            kernel_t,
            gemm_inp,
            torch.bfloat16,
            te.module.base.get_workspace(),
            bias=None,
            use_bias=False,
            gelu=False,
            ub_algo=ub_algo,
            ub=ub_obj,
            extra_output_tensor=rs_out,
            out=ubuf_out,
        )

    # Trigger GEMM
    total_iters = opts.warmup_iters + opts.timing_iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    torch.cuda.synchronize()

    if opts.use_cuda_graphs:
        # Trace the CUDA graph first
        g = torch.cuda.CUDAGraph()
        if opts.fp8:
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
            if opts.fp8:
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
            "p2p all-gather + " if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG else "",
            "atomic " if opts.atomic else "",
            "GEMM",
            (
                f" + {'p2p ' if opts.p2p else ''}reduce-scatter"
                if opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS
                else ""
            ),
        ]
    )
    timing_info = (
        f"Avg. GPU time for {gemm_name}: {avg_gpu_time} ms "
        + f"({opts.warmup_iters} warmup + {opts.timing_iters} timing runs)"
    )
    dist_print(timing_info, section=True, info=True)

    # Compare against standard GEMM
    numerics_failed = False
    if opts.check_numerics:
        torch.cuda.synchronize()
        dist.barrier()
        if opts.bulk_overlap:
            output_info = ""
            if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
                # Bulk overlap AG output is already gathered
                test_out = ub_obj.get_ubuf_output(tex.NVTE_Comm_Overlap_Type.AG)
            else:
                # Bulk overlap RS output needs to be gathered
                out_local = ub_obj.get_ubuf_output(tex.NVTE_Comm_Overlap_Type.RS)
                output_info += f"rs_output: {list(out_local.shape)} | "
                test_out = te.distributed.gather_along_first_dim(out_local, tp_group)[0]

            ref_out = ref_g
            output_info += f"output: {list(test_out.shape)} | reference: {list(ref_out.shape)}"
            dist_print(output_info, src=0 if opts.comm_type == 0 else None, section=True)

            test_nonzeros = torch.count_nonzero(test_out)
            ref_nonzeros = torch.count_nonzero(ref_out)
            nonzero_info = (
                f"output nonzeros = {test_nonzeros} " + f"| reference count = {ref_nonzeros}"
            )
            dist_print(nonzero_info, src=0, section=True)
        else:
            if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
                if ub_obj2 is not None:
                    # AG+RS Output: (M/P, N) -> gather -> (M, N)
                    output = rs_out2
                    test_out = te.distributed.gather_along_first_dim(output, tp_group)[0]
                else:
                    # AG Output: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
                    output = all_outputs[0]
                    test_out = torch.transpose(
                        te.distributed.gather_along_first_dim(torch.transpose(output, 0, 1),
                                                              tp_group)[0], 0, 1)
            else:
                # RS Output: (M/P, N) -> gather -> (M, N)
                output = rs_out
                test_out = te.distributed.gather_along_first_dim(output, tp_group)[0]

            if opts.fp8:
                dist_print("GEMM1 FP8 metas = [INPUT, WEIGHT, OUTPUT]", src=0, section=True)
                fp8_meta_info = (
                    f"amax_reference  = {fp8_meta.amax_history[1][:3].tolist()}\n"
                    + f"amax_history    = {fp8_meta.amax_history[0][:3].tolist()}\n"
                    + f"scale           = {fp8_meta.scale[:3].tolist()}\n"
                    + f"scale_inv       = {fp8_meta.scale_inv[:3].tolist()}"
                )
                dist_print(fp8_meta_info, src=0)
                if ub_obj2 is not None:
                    dist_print("GEMM2 FP8 metas = [INPUT, WEIGHT, OUTPUT]", src=0, section=True)
                    fp8_meta_info = (
                        f"amax_reference  = {fp8_meta.amax_history[1][3:].tolist()}\n"
                        + f"amax_history    = {fp8_meta.amax_history[0][3:].tolist()}\n"
                        + f"scale           = {fp8_meta.scale[3:].tolist()}\n"
                        + f"scale_inv       = {fp8_meta.scale_inv[3:].tolist()}"
                    )
                    dist_print(fp8_meta_info, src=0)

            test_nonzeros = torch.count_nonzero(test_out)
            dist.all_reduce(test_nonzeros, op=dist.ReduceOp.SUM, group=tp_group)
            ref_out = ref2_g if ub_obj2 is not None else ref_g
            ref_nonzeros = torch.count_nonzero(ref_out)
            nonzero_info = (
                f"output nonzeros = {test_nonzeros} "
                + f"| reference count = {ref_nonzeros}"
            )
            dist_print(nonzero_info, src=0, section=True)

            sizing_info = (
                f"input: {list(inp.shape)} "
                + f"| GEMM1 weights: {list(kernel_t.shape)[::-1]} "
            )
            if ub_obj2 is not None:
                sizing_info += f"| GEMM2 weights: {list(kernel2_t.shape)[::-1]} "
            sizing_info += f"| output: {list(output.shape)}\n"
            dist_print(sizing_info, section=True)

            sizing_info_g = (
                f"input: {list(inp_g.shape)} "
                + f"| GEMM1 weights: {list(ker_g.shape)} "
            )
            if ub_obj2 is not None:
                sizing_info_g += f"| GEMM2 weights: {list(ker2_g.shape)} "
            sizing_info_g += (
                f"| output: {list(test_out.shape)} "
                + f"| reference: {list(ref_out.shape)}\n"
            )
            dist_print(sizing_info_g, src=0)

        torch.cuda.synchronize()
        dist.barrier()
        test_out = test_out.to(dtype=torch.float32)
        ref_out = ref_out.to(dtype=torch.float32)
        error_below_tol = torch.allclose(
            test_out,
            ref_out,
            rtol=0.125 if opts.fp8 else 0.01,
            atol=0.0675 if opts.fp8 else 0.001,
        )
        diff = torch.abs(test_out - ref_out).flatten()
        m = torch.argmax(diff)
        abs_err = diff[m].item()
        rel_err = abs_err / (ref_out.flatten()[m].item() + 1e-5)
        if not error_below_tol:
            numerics_failed = True
            numerics_info = (
                "NUMERICAL CHECK FAILED: "
                + f"Outputs not close enough at index {m.item()} "
                + f"with {test_out.flatten()[m].item()} vs {ref_out.flatten()[m].item()} "
                + f"(abs error = {abs_err} | rel error = {rel_err})."
            )
        else:
            numerics_info = f"NUMERICAL CHECK PASSED: abs error = {abs_err} | rel error = {rel_err}"

        dist_print(numerics_info, src=0, section=True, info=True)

    dist.barrier()
    if WORLD_RANK == 0:
        print("\n", end="", flush=True)

    del ub_obj
    dist.destroy_process_group()

    # Reset clock speeds
    if opts.clock_speed > 0:
        subprocess.run(
            ["nvidia-smi", "-pm", "ENABLED", "-i", str(WORLD_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result = subprocess.run(
            ["nvidia-smi", "-rgc", "-i", str(WORLD_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return int(numerics_failed)


if __name__ == "__main__":
    os._exit(_main(_parse_args()))
