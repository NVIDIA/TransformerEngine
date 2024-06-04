#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import faulthandler
import warnings
import subprocess
import argparse
import operator
from functools import partial, reduce

import torch
import torch.distributed as dist


import transformer_engine.common.recipe as recipe
import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.fp8 import _default_sf_compute

torch_dtypes = {
    'fp32' : torch.float32,
    'fp16' : torch.float16,
    'bf16' : torch.bfloat16,
}

nvte_comm_types = {
    'ag' : tex.NVTE_Comm_Overlap_Type.AG,
    'rs' : tex.NVTE_Comm_Overlap_Type.RS,
}

def mapped_argtype(opt, typemap={}):
    if str(opt).lower() not in typemap.keys():
        raise TypeError(f"Unrecognized option! Please choose from: {typemap.keys()}")
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
                        default=tex.NVTE_Comm_Overlap_Type.AG,
                        help="Comm type to overlap.")
    parser.add_argument("--dtype", type=partial(mapped_argtype, typemap=torch_dtypes),
                        default=torch.bfloat16,
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

    # torch.distributed callback wrappers for bootstrapping userbuffers
    def alloc_copy_allgather_callback(local_data: torch.Tensor, group: str):
        pg = None if group == "world" else tp_group
        global_size = local_data.numel() * dist.get_world_size(pg)
        global_data = torch.zeros(global_size, dtype=local_data.dtype, device='cuda')
        dist.all_gather_into_tensor(global_data, local_data.cuda(), group=pg)
        return global_data.cpu()

    def bcast_int_callback(data: torch.Tensor, src: int, group: str):
        pg = None if group == "world" else tp_group
        data = data.cuda()
        dist.broadcast(data, src, group=pg)
        return data.cpu()

    def barrier_callback(group: str):
        pg = None if group == "world" else tp_group
        dist.barrier(group=pg)

    def free_callback(data: torch.Tensor):
        del data

    tex.set_collective_callbacks(
        alloc_copy_allgather_callback,
        bcast_int_callback,
        barrier_callback,
        free_callback
    )

    if opts.atomic and opts.check_numerics:
        assert not opts.fp8, "Atomic GEMM is only available with FP8."
        opts.comm_type = tex.NVTE_Comm_Overlap_Type.AG
        ub_algo = tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_AG_P2P
        second_overlap_is_p2p = opts.p2p
        opts.p2p = True
    elif opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS:
        if opts.p2p:
            ub_algo = tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_RS_P2P if opts.atomic else \
                      tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_RS_P2P
        else:
            ub_algo = tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_RS if opts.atomic else \
                      tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_RS
    else:
        if not opts.p2p:
            warnings.warn("All-Gather overlap is supported only via P2P ring-exchange.")
            opts.p2p = True
        ub_algo = tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_AG_P2P if opts.atomic else \
                  tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_AG_P2P

    # Initialize userbuffers with (M, N) buffer
    # M = sequence * batch
    # N = hidden size
    hidden_size = opts.num_heads * opts.head_dim
    inp_shape = (opts.seq_length, opts.batch_size, hidden_size)
    outer_size = reduce(operator.mul, inp_shape[:-1], 1)
    sample_buffer = torch.empty((outer_size, hidden_size),
                                dtype=torch.uint8 if opts.fp8 else opts.dtype, device='cuda')
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
        opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS   # is_reduce_scatter
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
    if opts.atomic and opts.check_numerics:
        # For numerical checks on atomic GEMM, we need to do an AG-RS pair of GEMMs
        # so set the first overla algo to all-gather overlap here and create two UB objects.
        ub_obj_2 = tex.UbufP2PCommOverlap(  # reduce-scatter UB object
            sample_buffer,
            world_rank,
            world_size,
            local_rank,
            local_size,
            tex.NVTE_MAX_USERBUFFER_STREAMS,
            True,  # set_sm_margin
            True,  # atomic_gemm (2nd overlap always atomic)
            opts.aggregate,
            True   # is_reduce_scatter (2nd overlap always RS)
        ) if second_overlap_is_p2p else tex.UbufCommOverlap(
            sample_buffer,
            world_rank,
            world_size,
            local_rank,
            local_size,
            4,      # num_splits
            tex.NVTE_MAX_USERBUFFER_STREAMS,
            2,      # cga_size
            1,      # num_sms
            False,  # set_sm_margin
            True,   # atomic_gemm (2nd overlap always atomic)
        )

    # Figure out problem sizing:
    # M = sequence * batch
    # N = hidden size
    # K = MLP intermediate size (usually 4x hidden size)
    # P = number of devices for sequence/tensor parallelism
    # NOTE: TE-GEMM is set up to work with a transposed kernels and  non-transposed inputs.
    ffn_hidden_size = 4 * hidden_size
    if opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS:
        # (M, K/P) x (N, K/P)^T = (M, N) -> overlapped RS -> (M/P, N)
        local_kernel_t_shape = (hidden_size, ffn_hidden_size // local_size)
        local_inp_shape = (outer_size, ffn_hidden_size // local_size)
    else:
        # (M/P, N) -> overlapped AG -> (M, N) x (K/P, N)^T = (M, K/P)
        local_kernel_t_shape = (ffn_hidden_size // local_size, hidden_size)
        local_inp_shape = (outer_size // local_size, hidden_size)
        if opts.atomic and opts.check_numerics:
            # (M, K/P) x (N, K/P)^T = (M, N) -> overlapped RS -> (M/P, N)
            local_kernel_t_shape_2 = (hidden_size, ffn_hidden_size // local_size)

    # Initialize input tensor and GEMM kernels
    inp = torch.rand(local_inp_shape, dtype=opts.dtype, device='cuda')/100.
    kernel_t = torch.rand(local_kernel_t_shape, dtype=opts.dtype, device='cuda')/100.
    if opts.atomic and opts.check_numerics:
        kernel_t_2 = torch.rand(local_kernel_t_shape_2, dtype=opts.dtype, device='cuda')/100.

    # Gather global tensors
    if opts.check_numerics:
        size_debug = \
            f"[rank:{world_rank}] input: {list(inp.shape)}  | kernel_1: {list(kernel_t.shape)}"
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS:
            # Kernel: (N, K/P) -> T -> (K/P, N) -> gather -> (K, N)
            ker_g = te.distributed.gather_along_first_dim(torch.transpose(kernel_t, 0, 1),
                                                                          tp_group)[0]
            # Input: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
            inp_g = torch.transpose(
                te.distributed.gather_along_first_dim(torch.transpose(inp, 0, 1),
                                                      tp_group)[0],
                0, 1)
        else:
            # Kernel: (K/P, N) -> gather -> (K, N) -> T -> (N, K)
            ker_g = torch.transpose(
                te.distributed.gather_along_first_dim(kernel_t,
                                                      tp_group)[0],
                0, 1)
            # Input: (M/P, N) -> gather -> (M, N)
            inp_g = te.distributed.gather_along_first_dim(inp, tp_group)[0]
            if opts.atomic:
                size_debug += f" | kernel_2: {list(kernel_t_2.shape)}"
                # Kernel 2: (N, K/P) -> T -> (K/P, N) -> gather -> (K, N)
                ker_g_2 = te.distributed.gather_along_first_dim(torch.transpose(kernel_t_2, 0, 1),
                                                                tp_group)[0]

        ref_g_1 = torch.matmul(inp_g, ker_g)
        if opts.atomic:
            ref_g_2 = torch.matmul(ref_g_1, ker_g_2)

    inp_final = inp
    if opts.fp8:
        # Structure to maintain amax and scale/scale_inv information for the kernel and input
        fp8_dtype = tex.DType.kFloat8E4M3
        fp8_meta = tex.FP8TensorMeta()
        fp8_meta.amax_history = torch.zeros((1, 9), dtype=torch.float, device='cuda')
        fp8_meta.scale = torch.ones(9, dtype=torch.float, device='cuda')
        fp8_meta.scale_inv = torch.ones(9, dtype=torch.float, device='cuda')
        # Compute amaxes and scales
        fp8_meta.amax_history[0][tex.FP8FwdTensors.GEMM1_INPUT].copy_(torch.max(torch.abs(inp_g)))
        fp8_meta.amax_history[0][tex.FP8FwdTensors.GEMM1_WEIGHT].copy_(torch.max(torch.abs(ker_g)))
        fp8_meta.amax_history[0][tex.FP8FwdTensors.GEMM1_OUTPUT].copy_(torch.max(torch.abs(ref_g_1)))
        if opts.atomic and opts.check_numerics:
            fp8_meta.amax_history[0][tex.FP8FwdTensors.GEMM2_WEIGHT].copy_(
                torch.max(torch.abs(ker_g_2)))
            fp8_meta.amax_history[0][tex.FP8FwdTensors.GEMM2_OUTPUT].copy_(
                torch.max(torch.abs(ref_g_2)))
        _default_sf_compute(fp8_meta.amax_history[0], fp8_meta.scale, 448, 1)
        fp8_meta.scale_inv = 1./fp8_meta.scale
        print(f"[rank:{local_rank}] GEMM1_OUTPUT scale={fp8_meta.scale[tex.FP8FwdTensors.GEMM1_OUTPUT].item()} scale_inv={fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM1_OUTPUT].item()}")
        # Cast input to Float8Tensor
        inp_fp8 = tex.cast_to_fp8(
            inp,
            fp8_meta,
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype
        )
        # Cast kernel to Float8Tensor
        kernel_t_fp8 = tex.cast_to_fp8(
            kernel_t,
            fp8_meta,
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype
        )
        if opts.atomic and opts.check_numerics:
            kernel_t_2_fp8 = tex.cast_to_fp8(
                kernel_t_2,
                fp8_meta,
                tex.FP8FwdTensors.GEMM2_WEIGHT,
                fp8_dtype
            )
        # Make sure the inputs are cast correctly
        if opts.check_numerics:
            torch.allclose(
                inp.to(dtype=torch.float),
                inp_fp8 * fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM1_INPUT],
                rtol=0.125, atol=0.0675)
            torch.allclose(
                kernel_t.to(dtype=torch.float),
                kernel_t_fp8 * fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM1_WEIGHT],
                rtol=0.125, atol=0.0675)
            if opts.atomic:
                torch.allclose(
                    kernel_t_2.to(dtype=torch.float),
                    kernel_t_2_fp8 * fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM2_WEIGHT],
                    rtol=0.125, atol=0.0675)

    # Set up comm/compute buffers
    if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
        ub_obj.copy_input_to_ubuf(inp_fp8 if opts.fp8 else inp, 1)
        inp_final = ub_obj.get_ubuf_output(1)
        ubuf_out = None
        extra_out = torch.empty_like(ub_obj.get_ubuf_output(0))
    else:
        inp_final = inp_fp8 if opts.fp8 else inp
        ubuf_out = ub_obj.get_ubuf_output(1)
        extra_out = torch.empty((inp.size(0) // local_size, kernel_t.size(0)),
                                dtype=opts.dtype, device='cuda')

    # Trigger GEMM
    if opts.fp8:
        fp8_output = opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS or \
                     (opts.atomic and opts.check_numerics)
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
            ub_obj.set_ubuf_scale_inv(fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM1_INPUT])
        else:
            ub_obj.set_ubuf_scale_inv(fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM1_OUTPUT])
        all_outputs = tex.fp8_gemm(
            kernel_t_fp8,
            fp8_meta.scale_inv,
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype,
            inp_final,
            fp8_meta.scale_inv,
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype,
            torch.uint8 if fp8_output else opts.dtype,
            te.module.base.get_workspace(),
            bias=None,
            use_bias=False,
            gelu=False,
            use_split_accumulator=te.module.base._2X_ACC_FPROP,
            ub_algo=ub_algo,
            ub=ub_obj,
            extra_output_tensor=extra_out,
            D_dtype=fp8_dtype if fp8_output else None,
            fp8_meta_tensor=fp8_meta if fp8_output else None,
            out_index=tex.FP8FwdTensors.GEMM1_OUTPUT if fp8_output else None,
            out=ubuf_out,
        )
        if opts.atomic and opts.check_numerics:
            ub_algo_2 = tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_RS_P2P if second_overlap_is_p2p \
                        else tex.NVTE_Comm_Overlap_Algo.ATOMIC_GEMM_RS
            ub_obj_2.set_ubuf_scale_inv(fp8_meta.scale_inv[tex.FP8FwdTensors.GEMM2_OUTPUT])
            ubuf_out_2 = ub_obj_2.get_ubuf_output(1)
            rs_out = torch.empty((all_outputs[0].size(0) // local_size, kernel_t_2.size(0)),
                                 dtype=opts.dtype, device='cuda')
            _ = tex.fp8_gemm(
                kernel_t_2_fp8,
                fp8_meta.scale_inv,
                tex.FP8FwdTensors.GEMM2_WEIGHT,
                fp8_dtype,
                all_outputs[0],
                fp8_meta.scale_inv,
                tex.FP8FwdTensors.GEMM1_OUTPUT,
                fp8_dtype,
                torch.uint8,
                te.module.base.get_workspace(),
                bias=None,
                use_bias=False,
                gelu=False,
                use_split_accumulator=te.module.base._2X_ACC_FPROP,
                ub_algo=ub_algo_2,
                ub=ub_obj_2,
                extra_output_tensor=rs_out,
                D_dtype=fp8_dtype,
                fp8_meta_tensor=fp8_meta,
                out_index=tex.FP8FwdTensors.GEMM2_OUTPUT,
                out=ubuf_out_2,
            )
            output = rs_out
        elif opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS:
            output = extra_out
        else:
            output = all_outputs[0]
    else:
        all_outputs = tex.gemm(
            kernel_t,
            inp_final,
            opts.dtype,
            te.module.base.get_workspace(),
            bias=None,
            use_bias=False,
            gelu=False,
            ub_algo=ub_algo,
            ub=ub_obj,
            extra_output_tensor=extra_out,
            out=ubuf_out,
        )
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS:
            output = extra_out
        else:
            output = all_outputs[0]

    # Compare against standard GEMM
    assert_failed = False
    error_msg = ''
    if opts.check_numerics:
        if opts.comm_type == tex.NVTE_Comm_Overlap_Type.RS:
            # Output:(M/P, N) -> gather -> (M, N)
            out_g = te.distributed.gather_along_first_dim(output, tp_group)[0]
        else:
            if opts.atomic:
                out_g = te.distributed.gather_along_first_dim(output, tp_group)[0]
            else:
                # Output: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
                out_g = torch.transpose(
                    te.distributed.gather_along_first_dim(torch.transpose(output, 0, 1),
                                                          tp_group)[0],
                    0, 1)

        ref_g = ref_g_2 if opts.atomic else ref_g_1
        size_debug += f" | output: {list(output.shape)}"
        print(size_debug)
        dist.barrier()
        if world_rank == 0:
            size_debug_g = f"[GLOBAL] inp_g: {list(inp_g.shape)}  | ker_g: {list(ker_g.shape)}"
            if opts.atomic:
                size_debug_g += f" | ker_g_2: {list(ker_g_2.shape)}"
            size_debug_g += f" | out_g: {list(out_g.shape)} | ref_g: {list(ref_g.shape)}"
            print(size_debug_g)
        dist.barrier()

        error_below_tol = torch.allclose(out_g.to(dtype=torch.float32),
                                         ref_g.to(dtype=torch.float32),
                                         rtol=0.125 if opts.fp8 else 1.6e-2,
                                         atol=0.0675 if opts.fp8 else 1e-5)
        if not error_below_tol:
            diff = torch.abs(out_g - ref_g).flatten()
            m = torch.argmax(diff)
            error_msg = (
                f"Outputs not close enough at index {m.item()} "
                f"with {out_g.flatten()[m].item()} vs {ref_g.flatten()[m].item()} "
                f"(diff {diff[m].item()})."
            )
            assert_failed = True

    if opts.debug:
        dbg_log.close()

    if assert_failed:
        raise AssertionError(error_msg)

    return 0


if __name__ == "__main__":
    if "TORCHELASTIC_RUN_ID" in os.environ.keys():
        args = parse_args()
        main(args)
    else:
        subprocess.run(
            [
                'torchrun', f'--nproc-per-node={torch.cuda.device_count()}',
                *sys.argv
            ],
            env=os.environ,
            check=True
        )
    os._exit(0)
