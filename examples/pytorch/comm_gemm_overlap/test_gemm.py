#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import faulthandler
import argparse
import operator
from functools import partial, reduce

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex

torch_dtypes = {
    'fp32' : torch.float32,
    'float32' : torch.float32,
    'fp16' : torch.float16,
    'float16' : torch.float16,
    'bf16' : torch.bfloat16,
    'bfloat16' : torch.bfloat16
}

nvte_comm_types = {
    'ag' : tex.NVTE_Comm_Overlap_Type.AG,
    'rs' : tex.NVTE_Comm_Overlap_Type.RS,
}

def mapped_argtype(opt, typemap={}):
    if str(opt).lower() not in typemap.keys():
        raise TypeError
    return typemap[str(opt).lower()]

def parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Test comm+GEMM overlap with Userbuffers.")
    parser.add_argument('-i', "--num-iters", type=int, default=5,
                        help="Number of dummy 'training' iterations.")
    parser.add_argument('-b', "--batch-size", type=int, default=2,
                        help="Input batch size.")
    parser.add_argument('-s', "--seq-length", type=int, default=2048,
                        help="Input sequence length.")
    parser.add_argument('-n', "--num-heads", type=int, default=64,
                        help="Number of attention heads.")
    parser.add_argument('-d', "--head-dim", type=int, default=128,
                        help="Dimension of each attention head.")
    parser.add_argument("--mlp-expansion-factor", type=int, default=4,
                        help="MLP block intermediate size as a factor of hidden dimension.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="RNG seed.")
    parser.add_argument("--fp8", action="store_true", default=False,
                        help="Enables the te.fp8_autocast() context.")
    parser.add_argument("--p2p", action="store_true", default=False,
                        help="Test overlap with P2P comms.")
    parser.add_argument("--atomic", action="store_true", default=False,
                        help="Test overlap with atomic GEMM.")
    parser.add_argument("--aggregate", action="store_true", default=False,
                        help="Aggregate 2X chunks for P2P split pipelined all-gather.")
    parser.add_argument("--comm-type", type=partial(mapped_argtype, typemap=nvte_comm_types),
                        default=tex.NVTE_Comm_Overlap_Type.RS, choices=nvte_comm_types.keys(),
                        help="Comm type to overlap.")
    parser.add_argument("--dtype", type=partial(mapped_argtype, typemap=torch_dtypes),
                        default=torch.bfloat16, choices=torch_dtypes.keys(),
                        help="Data type for input tensor and Transformer Engine module parameters.")
    parser.add_argument("--check-numerics", action="store_true", default=False,
                        help="Test numerical result against torch.matmul(...)")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument('-v', "--verbose", action="store_true", default=False)
    return parser.parse_args(argv, namespace)

def main(opts):
    WORLD_RANK = int(os.getenv("RANK"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE"))

    # Debug log
    if opts.debug:
        dbg_log = open(f'faulthandler_{WORLD_RANK}.log', 'w+')
        faulthandler.enable(dbg_log)

    # Seed RNG
    torch.cuda.set_device(WORLD_RANK)
    torch.manual_seed(opts.seed+WORLD_RANK)
    torch.cuda.manual_seed(opts.seed+WORLD_RANK)

    # Initialize torch.distributed global process group and get TP group
    dist.init_process_group(backend="nccl",
                            rank=WORLD_RANK,
                            world_size=WORLD_SIZE,
                            device_id=torch.device(f'cuda:{WORLD_RANK}'))
    tp_group = dist.new_group(backend="nccl")
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = dist.get_rank(tp_group)
    local_size = dist.get_world_size(tp_group)

    def alloc_copy_allgather_callback(local_data: torch.Tensor, group: str):
        pg = None if group == "world" else tp_group
        global_size = local_data.numel() * torch.distributed.get_world_size(pg)
        global_data = torch.zeros(global_size, dtype=local_data.dtype, device='cuda')
        torch.distributed.all_gather_into_tensor(global_data, local_data.cuda(), group=pg)
        return global_data.cpu()

    def bcast_int_callback(data: torch.Tensor, src: int, group: str):
        pg = None if group == "world" else tp_group
        data = data.cuda()
        torch.distributed.broadcast(data, src, group=pg)
        data = data.cpu()
        return data

    def barrier_callback(group: str):
        pg = None if group == "world" else tp_group
        torch.distributed.barrier(group=pg)

    def free_callback(data: torch.Tensor):
        del data

    tex.set_collective_callbacks(
        alloc_copy_allgather_callback,
        bcast_int_callback,
        barrier_callback,
        free_callback
    )

    if opts.p2p:
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS:
            if opts.atomic:
                ub_algo = tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_RS_P2P
            else:
                ub_algo = tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_RS_P2P
        else:
            if opts.atomic:
                ub_algo = tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_AG_P2P
            else:
                ub_algo = tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_AG_P2P
    else:
        assert opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS, "All-Gather overlap has to be P2P."
        if opts.atomic:
            ub_algo = tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_RS
        else:
            ub_algo = tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_RS

    # Initialize userbuffers with (M, N) buffer
    # M = sequence * batch
    # N = hidden size
    hidden_size = opts.num_heads * opts.head_dim
    inp_shape = (opts.seq_length, opts.batch_size, hidden_size)
    outer_size = reduce(operator.mul, inp_shape[:-1], 1)
    sample_buffer = torch.empty((outer_size, hidden_size),
                                dtype=torch.uint8 if opts.fp8 else torch.bfloat16, device='cuda')
    ub_obj = tex.UbufP2PCommOverlap(
        sample_buffer,
        world_rank,
        world_size,
        local_rank,
        local_size,
        tex.NVTE_MAX_USERBUFFER_STREAMS,
        opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS,  # set_sm_margin
        opts.atomic,
        opts.aggregate,
        opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS
    ) if opts.p2p else tex.UbufCommOverlap(
        sample_buffer,
        world_rank,
        world_size,
        local_rank,
        local_size,
        4 if opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS else 8,  # num_splits
        tex.NVTE_MAX_USERBUFFER_STREAMS,
        2,  # cga_size
        1,  # num_sms
        opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS,  # set_sm_margin
        opts.atomic,
    )

    # Figure out problem sizing:
    # M = sequence * batch
    # N = hidden size
    # K = MLP intermediate size (usually 4x hidden size)
    # P = number of devices for sequence/tensor parallelism
    # NOTE: TE-GEMM is set up to work with a transposed kernels and  non-transposed inputs.
    ffn_hidden_size = 4 * hidden_size
    if opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS:
        # (M, K/P) x (N, K/P)^T = (M, N) -> overlapped RS -> (M, N)
        local_kernel_t_shape = (hidden_size, ffn_hidden_size // local_size)
        local_inp_shape = (outer_size, ffn_hidden_size // local_size)
    else:
        # (M/P, N) -> overlapped AG -> (M, N) x (K/P, N)^T = (M, K/P)
        local_kernel_t_shape = (ffn_hidden_size // local_size, hidden_size)
        local_inp_shape = (outer_size // local_size, hidden_size)

    # Initialize kernel and input tensors
    kernel_t = torch.rand(local_kernel_t_shape, dtype=torch.bfloat16, device='cuda')
    inp = torch.rand(local_inp_shape, dtype=torch.bfloat16, device='cuda')
    if opts.fp8:
        # Structure to maintain amax and scale/scale_inv information for the kernel and input
        fp8_meta = tex.FP8TensorMeta()
        fp8_meta.amax_history = torch.zeros((1, 3), dtype=torch.float, device='cuda')
        fp8_meta.scale = torch.zeros(3, dtype=torch.float, device='cuda')
        fp8_meta.scale_inv = torch.zeros(3, dtype=torch.float, device='cuda')
        # Cast kernel to Float8Tensor
        kernel_t_fp8 = te.float8_tensor.Float8Tensor(
            data=torch.empty(
                kernel_t.shape,
                device=torch.cuda.current_device(),
                dtype=torch.uint8,
            ),
            fp8_dtype=tex.DType.kFloat8E4M3,
            fp8_scale_inv=1,
        )
        tex.cast_to_fp8(
            kernel_t,
            fp8_meta,
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            tex.DType.kFloat8E4M3,
            out=kernel_t_fp8._data
        )
        # Cast input to Float8Tensor
        inp_fp8 = te.float8_tensor.Float8Tensor(
            data=torch.empty(
                inp.shape,
                device=torch.cuda.current_device(),
                dtype=torch.uint8,
            ),
            fp8_dtype=tex.DType.kFloat8E4M3,
            fp8_scale_inv=1,
        )
        tex.cast_to_fp8(
            inp,
            fp8_meta,
            tex.FP8FwdTensors.GEMM1_INPUT,
            tex.DType.kFloat8E4M3,
            out=inp_fp8._data
        )
        # AG needs the input copied into userbuffers
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
            ub_obj.copy_input_to_ubuf(inp_fp8._data, 1)
            ub_obj.set_ubuf_scale_inv(fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM1_INPUT])
    else:
        # AG needs the input copied into userbuffers
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
            ub_obj.copy_input_to_ubuf(inp, 1)

    # Get userbuffers tensor
    if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
        inp_final = ub_obj.get_ubuf_output(1)
        extra_out = torch.zeros_like(inp_final)
    else:
        inp_final = inp_fp8._data if opts.fp8 else inp
        extra_out = ub_obj.get_ubuf_output(0)

    # Trigger GEMM
    if opts.fp8:
        output, *_ = tex.fp8_gemm(
            kernel_t_fp8._data,
            fp8_meta.scale_inv,
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            tex.DType.kFloat8E4M3,
            inp_final,
            fp8_meta.scale_inv,
            tex.FP8FwdTensors.GEMM1_INPUT,
            tex.DType.kFloat8E4M3,
            torch.bfloat16,
            te.module.base.get_workspace(),
            bias=None,
            use_bias=False,
            use_split_accumulator=te.module.base._2X_ACC_FPROP,
            ub_algo=ub_algo,
            ub=ub_obj,
            extra_output_tensor=extra_out,
        )
    else:
        output, *_ = tex.gemm(
            kernel_t,
            inp_final,
            torch.bfloat16,
            te.module.base.get_workspace(),
            bias=None,
            use_bias=False,
            gelu=False,
            ub_algo=ub_algo,
            ub=ub_obj,
            extra_output_tensor=extra_out,
        )

    # Compare against standard GEMM
    if opts.check_numerics:
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS:
            # Kernel: (N, K/P) -> T -> (K/P, N) -> gather -> (K, N)
            ker_g = te.distributed.gather_along_first_dim(torch.transpose(kernel_t, 0, 1),
                                                                          tp_group)[0]
            # Input: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
            inp_g = torch.transpose(
                te.distributed.gather_along_first_dim(torch.transpose(inp, 0, 1),
                                                      tp_group)[0],
                0, 1)
            # Output already (M, N)
            out_g = output
        else:
            # Kernel: (K/P, N) -> gather -> (K, N) -> T -> (N, K)
            ker_g = torch.transpose(
                te.distributed.gather_along_first_dim(kernel_t,
                                                      tp_group)[0],
                0, 1)
            # Input: (M/P, N) -> gather -> (M, N)
            inp_g = te.distributed.gather_along_first_dim(inp, tp_group)[0]
            # Output: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
            out_g = torch.transpose(
                te.distributed.gather_along_first_dim(torch.transpose(output, 0, 1),
                                                      tp_group)[0],
                0, 1)

        ref_g = torch.matmul(inp_g, ker_g)

        error_below_atol = torch.allclose(out_g, ref_g, atol=opts.atol)
        if not error_below_atol:
            diff = torch.abs(out_g - ref_g).flatten()
            m = torch.argmax(diff)
            error_msg = (
                f"Outputs not close enough at index {m.item()} "
                f"with {out_g.flatten()[m].item()} vs {ref_g.flatten()[m].item()} "
                f"(diff {diff[m].item()})."
            )
            raise AssertionError(error_msg)

    if opts.debug:
        dbg_log.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    os._exit(0)
