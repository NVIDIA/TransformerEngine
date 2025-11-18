from typing import Optional, List
import torch

from transformer_engine.pytorch.module.base import get_workspace
from transformer_engine.pytorch.module.linear import general_gemm
from transformer_engine.pytorch.utils import nvtx_range_push, nvtx_range_pop


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
        
        nvtx_range_push(f"transformer_engine.MetisSvdFunction.svd_quant_gemm_{nvtx_label}.gemm")
        gemm_out, *_ = general_gemm(
            x,
            y,
            get_workspace(),
            accumulate=False,
            layout=layout,
            quantization_params=output_quantizer,
            out_dtype=output_dtype,
            use_split_accumulator=True,
        )
        nvtx_range_pop(f"transformer_engine.MetisSvdFunction.svd_quant_gemm_{nvtx_label}.gemm")
        return gemm_out

    @staticmethod
    @torch.no_grad()
    def svd_lowrank_quant_grad_output(grad_output:torch.Tensor,
                                      grad_output_shape,
                                      **kargs):
        assert grad_output_shape is not None
        grad_output = grad_output.view(grad_output_shape)
        return MetisSvdFunction.svd_lowrank_quant(grad_output,**kargs)    

    @staticmethod
    @torch.no_grad()
    def svd_lowrank_quant(input_:torch.Tensor,
                          input_quantizer: "Quantizer", 
                          rank=60, 
                          niter=2, 
                          broadcast_dim=-1, 
                          is_backward = False,
                          gradacc_broadcast = False,
                          load_history = False,
                          history_list=List[Optional[torch.Tensor]]):

        # for backward, input_ has already shaped into 2d tensor.
        # input_ shape [b,s,h]

        input_shape = input_.shape
        if broadcast_dim >= 0:
            cinput = input_.select(broadcast_dim, 0) #[s,h]
        else:
            cinput = input_
        original_shape = cinput.shape #[s,h]
        if load_history and gradacc_broadcast and is_backward :
            ker,de_svd_gemm_out = history_list
            # print("load")       
        else:
            cinput = cinput.view(-1, original_shape[-1]) #[s,h] or [b*s,h]
            ug, sg, vg = torch.svd_lowrank(
                cinput.to(torch.float32), 
                q=rank, 
                niter=niter
            )
            ug = ug.to(input_.dtype)
            sg = sg.to(input_.dtype)
            sg = torch.diag(sg)
            vg = vg.T.to(input_.dtype)
            ker = (ug @ sg @ vg) #[s,h] or [b*s,h]
            if broadcast_dim >= 0:
                ker = ker.unsqueeze(broadcast_dim) #[1,s,h]
            else:
                ker = ker.view(input_shape) #[b,s,h]
            ug = input_quantizer(ug)
            vg = input_quantizer(vg)
            sg = input_quantizer(sg)
            gemm_out = MetisSvdFunction.svd_quant_gemm(sg, ug, input_.dtype, input_quantizer, layout="NN", nvtx_label="U@S")
            de_svd_gemm_out = MetisSvdFunction.svd_quant_gemm(vg,gemm_out, input_.dtype, None, layout="NN", nvtx_label="U@S@V")
            #[s,h] or [b*s,h]
            if broadcast_dim >= 0:
                de_svd_gemm_out = de_svd_gemm_out.unsqueeze(broadcast_dim) #[1,s,h]
            else:
                de_svd_gemm_out = de_svd_gemm_out.view(input_shape) # [b,s,h]
            if gradacc_broadcast and is_backward:
                # print("storing history_list----")
                history_list.clear()
                history_list.extend([ker,de_svd_gemm_out])
        input_res = input_ - ker #[b,s,h]
        out_tensor = de_svd_gemm_out + input_res #[b,s,h]
        
        output_fp4 = input_quantizer(out_tensor)
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

