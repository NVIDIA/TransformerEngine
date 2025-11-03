import torch

from transformer_engine.pytorch.module.base import get_workspace
from transformer_engine.pytorch.module.linear import general_gemm
from transformer_engine.pytorch.utils import nvtx_range_push, nvtx_range_pop
# from transformer_engine.pytorch import Quantizer

def schedule_none(input_:torch.Tensor):
    return input_, 1.0

def schedule_l1_m1p5_s2(input_:torch.Tensor):
    input_[5:] *= 1.5
    return input_, 2.0

def cuda_time_call(fn, *args, **kwargs):
    """Run a callable on CUDA and measure elapsed time in milliseconds.

    Args:
        fn: callable to run.
        *args, **kwargs: forwarded to fn.

    Returns:
        A tuple (result, elapsed_ms). If fn returns multiple values, result
        is whatever fn returned.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Record start, run, record end, synchronize, compute elapsed
    start.record()
    result = fn(*args, **kwargs)
    # Ensure any CUDA kernels launched by fn are recorded before ending
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    return result, elapsed

class MetisSvdFunction():

    @staticmethod
    @torch.no_grad()
    def svd_quant_gemm(x,y,output_dtype,output_quantizer = None,layout="TN",nvtx_label=""):
        
        nvtx_range_push(f"transformer_engine._MetisLowBitLinear.svd_quant_{nvtx_label}.gemm")
        gemm_out, *_ = general_gemm(
            x,
            y,
            get_workspace(),
            accumulate=False,
            layout=layout,
            quantization_params=output_quantizer,
            out_dtype=output_dtype,
        )
        nvtx_range_pop(f"transformer_engine._MetisLowBitLinear.svd_quant_{nvtx_label}.gemm")
        return gemm_out

    @staticmethod
    @torch.no_grad()
    def svd_lowrank_quant(input_:torch.Tensor,input_quantizer: "Quantizer", rank=60, niter=0, adaptive_schedule="none", broadcast_dim=-1):
        # print("-"*20+"svd_lowrank_quant begin"+"-"*20)
        if broadcast_dim >= 0:
            cinput = input_.select(broadcast_dim, 0)
        else:
            cinput = input_

        original_shape = cinput.shape
        if len(original_shape) == 3:
            cinput = cinput.view(-1, original_shape[-1])
            input_ = input_.view(-1, original_shape[-1])

        # (ug, sg, vg),svd_time = cuda_time_call(torch.svd_lowrank,
        #     cinput.to(torch.float32), 
        #     q=rank, 
        #     niter=niter
        # )
        ug, sg, vg = torch.svd_lowrank(
            cinput.to(torch.float32), 
            q=rank, 
            niter=niter
        )
        # print("svd_time = ",svd_time)
        # start_time = torch.cuda.Event(enable_timing=True)
        # end_time = torch.cuda.Event(enable_timing=True)
        # start_time.record()
        ug = ug.to(input_.dtype)
        sg = sg.to(input_.dtype)
        vg = vg.to(input_.dtype)
        # vg = vg.T
        # ug = ug.T
        
        # sg, res_scalar = LinearLowbitContext.schedule_list[adaptive_schedule](sg)
        sg = torch.diag(sg)
        ker = (ug @ sg @ vg.T)
        if broadcast_dim >= 0:
            ker = ker.unsqueeze(broadcast_dim)

        
        input_res = input_ - ker
        # end_time.record()
        # torch.cuda.synchronize()
        # ker_time_elapsed = start_time.elapsed_time(end_time)
        # print("input_res time elapsed = ",ker_time_elapsed)
        # input_res = input_quantizer(input_res)
        # start_time.record()
        ug_nvfp4 = input_quantizer.make_empty(
            ug.shape,dtype = ug.dtype, device=ug.device, requires_grad=False
        )
        vg_nvfp4 = input_quantizer.make_empty(
            vg.shape,dtype = vg.dtype, device=vg.device, requires_grad=False
        )
        sg_nvfp4 = input_quantizer.make_empty(
            sg.shape,dtype = sg.dtype, device=sg.device, requires_grad=False
        )
        ug_native = input_quantizer.update_quantized(ug, ug_nvfp4) 
        vg_native = input_quantizer.update_quantized(vg, vg_nvfp4)
        sg_native = input_quantizer.update_quantized(sg, sg_nvfp4)
        # end_time.record()
        # torch.cuda.synchronize()
        # print("quant time elapsed = ",start_time.elapsed_time(end_time))
        # out = torch.randn(sg_native.shape,dtype = input_.dtype, device=input_.device)
        # start_time.record()
        gemm_out = MetisSvdFunction.svd_quant_gemm(sg_native, ug_native, input_.dtype, input_quantizer, layout="NN", nvtx_label="U@S")
        de_svd_gemm_out = MetisSvdFunction.svd_quant_gemm(vg_native, gemm_out, input_.dtype, None, layout="TN", nvtx_label="U@S@V")
        input_ = de_svd_gemm_out + input_res
        output_fp4 = input_quantizer.make_empty(
            input_.shape,dtype = input_.dtype, device=input_.device, requires_grad=False
        )
        output_fp4 = input_quantizer.update_quantized(input_, output_fp4)
        if len(original_shape) == 3:
            output_fp4 = output_fp4.view(original_shape[0], original_shape[1], -1)
        # end_time.record()
        # torch.cuda.synchronize()
        # print("fp4_svd and final output time elapsed = ",start_time.elapsed_time(end_time))
        # print("-"*20+"svd_lowrank_quant end"+"-"*20)
        return output_fp4

    @staticmethod
    @torch.no_grad()
    def svd_fullrank_quant(input_: torch.Tensor, quantizer:"Quantizer"):
        ### Full rank SVD quantization
        ug, sg, vg = torch.svd(input_.to(torch.float32), some=True)
        ug = ug.to(input_.dtype)
        sg = torch.diag(sg.to(input_.dtype))
        vg = vg.to(input_.dtype)
        ug_nvfp4 = quantizer.make_empty(
            ug.shape,dtype = ug.dtype, device=ug.device, requires_grad=False
        )
        vg_nvfp4 = quantizer.make_empty(
            vg.shape,dtype = vg.dtype, device=vg.device, requires_grad=False
        )
        sg_nvfp4 = quantizer.make_empty(
            sg.shape,dtype = sg.dtype, device=sg.device, requires_grad=False
        )
        ug_quant = quantizer.update_quantized(ug, ug_nvfp4) 
        vg_quant = quantizer.update_quantized(vg, vg_nvfp4)
        sg_quant = quantizer.update_quantized(sg, sg_nvfp4)
        gemm_out = MetisSvdFunction.svd_quant_gemm(sg_quant, ug_quant, input_.dtype, quantizer, layout="NN", nvtx_label="U@S")
        de_svd_gemm_out = MetisSvdFunction.svd_quant_gemm(vg_quant, gemm_out, input_.dtype, quantizer, layout="TN", nvtx_label="U@S@V")
        return de_svd_gemm_out

