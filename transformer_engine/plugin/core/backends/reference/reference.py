# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ...ops import TEFLBackendBase, FP8TensorMeta, NVTE_Fused_Attn_Backend

from .impl import (
    general_gemm_torch,
    rmsnorm_fwd_torch, rmsnorm_bwd_torch,
    layernorm_fwd_torch, layernorm_bwd_torch,
    gelu_torch, geglu_torch, qgelu_torch, qgeglu_torch,
    relu_torch, reglu_torch, srelu_torch, sreglu_torch,
    silu_torch, swiglu_torch, clamped_swiglu_torch,
    dgelu_torch, dgeglu_torch, dqgelu_torch, dqgeglu_torch,
    drelu_torch, dreglu_torch, dsrelu_torch, dsreglu_torch,
    dsilu_torch, dswiglu_torch, clamped_dswiglu_torch,
    dbias_dgelu_torch, dbias_dsilu_torch, dbias_drelu_torch,
    dbias_dqgelu_torch, dbias_dsrelu_torch,
    scaled_softmax_forward_torch, scaled_softmax_backward_torch,
    scaled_masked_softmax_forward_torch, scaled_masked_softmax_backward_torch,
    scaled_upper_triang_masked_softmax_forward_torch,
    scaled_upper_triang_masked_softmax_backward_torch,
    scaled_aligned_causal_masked_softmax_forward_torch,
    scaled_aligned_causal_masked_softmax_backward_torch,
    dropout_fwd_torch, dropout_bwd_torch,
    multi_tensor_scale_torch, multi_tensor_l2norm_torch,
    multi_tensor_adam_torch, multi_tensor_sgd_torch,
)

class ReferenceBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def get_flash_attention_class(self):
        from .flash_attention import FlashAttentionTorch
        return FlashAttentionTorch

    def generic_gemm(
        self,
        A: torch.Tensor,
        transA: bool,
        B: torch.Tensor,
        transB: bool,
        D: torch.Tensor,
        quantizer: Any,
        output_dtype: torch.dtype,
        bias: Optional[torch.Tensor],
        bias_type: Any,
        gelu: bool,
        gelu_in: Optional[torch.Tensor],
        grad: bool,
        workspace: torch.Tensor,
        workspace_size: int,
        accumulate: bool,
        use_split_accumulator: bool,
        comm_overlap: Optional[Any] = None,
        comm_type: Optional[Any] = None,
        extra_output: Optional[torch.Tensor] = None,
        bulk_overlap: bool = False,
        alpha: float = 1.0,
        beta: Optional[float] = None,
    ) -> Any:
        return general_gemm_torch(
            A=A,
            transA=transA,
            B=B,
            transB=transB,
            D=D,
            quantizer=quantizer,
            output_dtype=output_dtype,
            bias=bias,
            bias_type=bias_type,
            gelu=gelu,
            gelu_in=gelu_in,
            grad=grad,
            workspace=workspace,
            workspace_size=workspace_size,
            accumulate=accumulate,
            use_split_accumulator=use_split_accumulator,
            comm_overlap=comm_overlap,
            comm_type=comm_type,
            extra_output=extra_output,
            bulk_overlap=bulk_overlap,
            alpha=alpha,
            beta=beta,
        )

    def te_general_grouped_gemm(self, *args, **kwargs) -> Any:
        raise NotImplementedError("te_general_grouped_gemm - not implemented in reference backend")

    def quantize(self, tensor: torch.Tensor, quantizer: Any, output: Optional[torch.Tensor] = None, noop: Optional[torch.Tensor] = None) -> Any:
        raise NotImplementedError("quantize - not implemented in reference backend")

    def dequantize(self, input: torch.Tensor, otype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError("dequantize - not implemented in reference backend")

    def bgrad_quantize(self, input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError("bgrad_quantize - not implemented in reference backend")

    def gelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return gelu_torch(input, quantizer)

    def geglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return geglu_torch(input, quantizer)

    def qgelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return qgelu_torch(input, quantizer)

    def qgeglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return qgeglu_torch(input, quantizer)

    def relu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return relu_torch(input, quantizer)

    def reglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return reglu_torch(input, quantizer)

    def srelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return srelu_torch(input, quantizer)

    def sreglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return sreglu_torch(input, quantizer)

    def silu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return silu_torch(input, quantizer)

    def swiglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        return swiglu_torch(input, quantizer)

    def clamped_swiglu(self, input: torch.Tensor, quantizer: Any, limit: float = 7.0, alpha: float = 1.702) -> Any:
        return clamped_swiglu_torch(input, quantizer, limit, alpha)

    def dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dgelu_torch(grad, fwd_input, quantizer)

    def dgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dgeglu_torch(grad, fwd_input, quantizer)

    def dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dqgelu_torch(grad, fwd_input, quantizer)

    def dqgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dqgeglu_torch(grad, fwd_input, quantizer)

    def drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return drelu_torch(grad, fwd_input, quantizer)

    def dreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dreglu_torch(grad, fwd_input, quantizer)

    def dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dsrelu_torch(grad, fwd_input, quantizer)

    def dsreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dsreglu_torch(grad, fwd_input, quantizer)

    def dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dsilu_torch(grad, fwd_input, quantizer)

    def dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        return dswiglu_torch(grad, fwd_input, quantizer)

    def clamped_dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any, limit: float = 7.0, alpha: float = 1.702) -> Any:
        return clamped_dswiglu_torch(grad, fwd_input, quantizer, limit, alpha)

    def dbias_dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        return dbias_dgelu_torch(grad, fwd_input, quantizer)

    def dbias_dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        return dbias_dsilu_torch(grad, fwd_input, quantizer)

    def dbias_drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        return dbias_drelu_torch(grad, fwd_input, quantizer)

    def dbias_dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        return dbias_dqgelu_torch(grad, fwd_input, quantizer)

    def dbias_dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        return dbias_dsrelu_torch(grad, fwd_input, quantizer)

    def layernorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
        ln_out: Optional[torch.Tensor],
        quantizer: Any,
        otype: torch.dtype,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return layernorm_fwd_torch(
            input=input,
            weight=weight,
            bias=bias,
            eps=eps,
            ln_out=ln_out,
            quantizer=quantizer,
            odtype=otype,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )

    def layernorm_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int = 0,
        zero_centered_gamma: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return layernorm_bwd_torch(
            dy=dy,
            x=x,
            mu=mu,
            rsigma=rsigma,
            gamma=gamma,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )

    def rmsnorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        ln_out: Optional[torch.Tensor],
        quantizer: Any,
        otype: torch.dtype,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        return rmsnorm_fwd_torch(
            input=input,
            weight=weight,
            eps=eps,
            ln_out=ln_out,
            quantizer=quantizer,
            odtype=otype,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
        )

    def rmsnorm_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int = 0,
        zero_centered_gamma: bool = False,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return rmsnorm_bwd_torch(
            dy=dy,
            x=x,
            rsigma=rsigma,
            gamma=gamma,
            sm_margin=sm_margin,
            zero_centered_gamma=zero_centered_gamma,
            eps=eps,
        )

    def rmsnorm_bwd_add(self, *args, **kwargs) -> Any:
        raise NotImplementedError("rmsnorm_bwd_add - not implemented in reference backend")

    def multi_tensor_quantize(self, tensor_list: List[torch.Tensor], quantizer_list: List[Any]) -> List[Any]:
        raise NotImplementedError("multi_tensor_quantize - not implemented in reference backend")

    def split_quantize(self, tensor: torch.Tensor, split_sections: List[int], quantizer_list: List[Any]) -> List[Any]:
        raise NotImplementedError("split_quantize - not implemented in reference backend")

    def moe_permute_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_permute_fwd - not implemented in reference backend")

    def moe_permute_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_permute_bwd - not implemented in reference backend")

    def moe_unpermute_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_unpermute_fwd - not implemented in reference backend")

    def moe_unpermute_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("moe_unpermute_bwd - not implemented in reference backend")

    def scaled_softmax_forward(self, input: torch.Tensor, scale: float) -> torch.Tensor:
        return scaled_softmax_forward_torch(input, scale)

    def scaled_softmax_backward(self, output_grad: torch.Tensor, softmax_output: torch.Tensor, scale: float) -> torch.Tensor:
        return scaled_softmax_backward_torch(output_grad, softmax_output, scale)

    def scaled_masked_softmax_forward(self, input: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
        return scaled_masked_softmax_forward_torch(input, mask, scale)

    def scaled_masked_softmax_backward(self, output_grad: torch.Tensor, softmax_output: torch.Tensor, scale: float) -> torch.Tensor:
        return scaled_masked_softmax_backward_torch(output_grad, softmax_output, scale)

    def scaled_upper_triang_masked_softmax_forward(self, input: torch.Tensor, scale: float) -> torch.Tensor:
        return scaled_upper_triang_masked_softmax_forward_torch(input, scale)

    def scaled_upper_triang_masked_softmax_backward(self, output_grad: torch.Tensor, softmax_output: torch.Tensor, scale: float) -> torch.Tensor:
        return scaled_upper_triang_masked_softmax_backward_torch(output_grad, softmax_output, scale)

    def scaled_aligned_causal_masked_softmax_forward(self, input: torch.Tensor, scale: float) -> torch.Tensor:
        return scaled_aligned_causal_masked_softmax_forward_torch(input, scale)

    def scaled_aligned_causal_masked_softmax_backward(self, output_grad: torch.Tensor, softmax_output: torch.Tensor, scale: float) -> torch.Tensor:
        return scaled_aligned_causal_masked_softmax_backward_torch(output_grad, softmax_output, scale)

    def get_fused_attn_backend(self, *args, **kwargs) -> int:
        return NVTE_Fused_Attn_Backend.NVTE_No_Backend

    def fused_attn_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_attn_fwd - not implemented in reference backend")

    def fused_attn_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_attn_bwd - not implemented in reference backend")

    def fa_prepare_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fa_prepare_fwd - not implemented in reference backend")

    def fa_prepare_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fa_prepare_bwd - not implemented in reference backend")

    def copy_to_kv_cache(self, *args, **kwargs) -> Any:
        raise NotImplementedError("copy_to_kv_cache - not implemented in reference backend")

    def convert_thd_to_bshd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("convert_thd_to_bshd - not implemented in reference backend")

    def convert_bshd_to_thd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("convert_bshd_to_thd - not implemented in reference backend")

    def fused_rope_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_rope_forward - not implemented in reference backend")

    def fused_rope_backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_rope_backward - not implemented in reference backend")

    def fused_qkv_rope_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_qkv_rope_forward - not implemented in reference backend")

    def fused_qkv_rope_backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_qkv_rope_backward - not implemented in reference backend")

    def fused_topk_with_score_function_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_topk_with_score_function_fwd - not implemented in reference backend")

    def fused_topk_with_score_function_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_topk_with_score_function_bwd - not implemented in reference backend")

    def fused_score_for_moe_aux_loss_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_score_for_moe_aux_loss_fwd - not implemented in reference backend")

    def fused_score_for_moe_aux_loss_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_score_for_moe_aux_loss_bwd - not implemented in reference backend")

    def fused_moe_aux_loss_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_moe_aux_loss_fwd - not implemented in reference backend")

    def fused_moe_aux_loss_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_moe_aux_loss_bwd - not implemented in reference backend")

    def dropout_fwd(self, input: torch.Tensor, dropout_probability: float, out: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return dropout_fwd_torch(input, dropout_probability, out)

    def dropout_bwd(self, grad_output: torch.Tensor, mask: torch.Tensor, dropout_probability: float, grad_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        return dropout_bwd_torch(grad_output, mask, dropout_probability, grad_input)

    def fp8_transpose(self, input: torch.Tensor, dtype: Any, *, out: torch.Tensor) -> None:
        raise NotImplementedError("fp8_transpose - not implemented in reference backend")

    def swap_first_dims(self, tensor: torch.Tensor, *, out: torch.Tensor) -> None:
        raise NotImplementedError("swap_first_dims - not implemented in reference backend")

    def compute_amax(self, input: torch.Tensor, amax: torch.Tensor) -> None:
        raise NotImplementedError("compute_amax - not implemented in reference backend")

    def fused_amax_and_scale_update_after_reduction(self, *args, **kwargs) -> None:
        raise NotImplementedError("fused_amax_and_scale_update_after_reduction - not implemented in reference backend")

    def fp8_block_scaling_compute_partial_amax(self, *args, **kwargs) -> None:
        raise NotImplementedError("fp8_block_scaling_compute_partial_amax - not implemented in reference backend")

    def fp8_block_scaling_partial_cast(self, *args, **kwargs) -> None:
        raise NotImplementedError("fp8_block_scaling_partial_cast - not implemented in reference backend")

    def fused_multi_row_padding(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_multi_row_padding - not implemented in reference backend")

    def fused_multi_row_unpadding(self, *args, **kwargs) -> Any:
        raise NotImplementedError("fused_multi_row_unpadding - not implemented in reference backend")

    def get_cublasLt_version(self) -> int:
        return 0

    def get_cudnn_version(self) -> int:
        return 0

    def get_num_cublas_streams(self) -> int:
        return 0

    def thd_read_half_tensor(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_read_half_tensor - not implemented in reference backend")

    def thd_second_half_lse_correction(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_second_half_lse_correction - not implemented in reference backend")

    def thd_read_second_half_lse(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_read_second_half_lse - not implemented in reference backend")

    def thd_out_correction(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_out_correction - not implemented in reference backend")

    def thd_grad_correction(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_grad_correction - not implemented in reference backend")

    def thd_get_partitioned_indices(self, *args, **kwargs) -> Any:
        raise NotImplementedError("thd_get_partitioned_indices - not implemented in reference backend")

    def init_nvshmem_backend(self, *args, **kwargs) -> None:
        raise NotImplementedError("init_nvshmem_backend - not implemented in reference backend")

    def create_nvshmem_tensor(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("create_nvshmem_tensor - not implemented in reference backend")

    def nvshmem_send_on_current_stream(self, *args, **kwargs) -> None:
        raise NotImplementedError("nvshmem_send_on_current_stream - not implemented in reference backend")

    def nvshmem_wait_on_current_stream(self, *args, **kwargs) -> None:
        raise NotImplementedError("nvshmem_wait_on_current_stream - not implemented in reference backend")

    def nvshmem_finalize(self) -> None:
        raise NotImplementedError("nvshmem_finalize - not implemented in reference backend")

    def multi_tensor_scale(self, chunk_size: int, noop_flag: torch.Tensor, tensor_lists: List[List[torch.Tensor]], scale: float) -> None:
        return multi_tensor_scale_torch(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_l2norm(self, chunk_size: int, noop_flag: torch.Tensor, tensor_lists: List[List[torch.Tensor]], per_tensor: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        return multi_tensor_l2norm_torch(chunk_size, noop_flag, tensor_lists, per_tensor)

    def multi_tensor_unscale_l2norm(self, chunk_size: int, noop_flag: torch.Tensor, tensor_lists: List[List[torch.Tensor]], scale: torch.Tensor, per_tensor: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute L2 norm after unscaling.

        Note: scale parameter is actually inv_scale (1/loss_scale).
        Unscaling means multiplying by inv_scale (= dividing by loss_scale).
        """
        if noop_flag.item() != 0:
            if per_tensor:
                return [torch.tensor(0.0, device=t.device) for t in tensor_lists[0]]
            else:
                return torch.tensor(0.0, device=tensor_lists[0][0].device)

        # Multiply by inv_scale (scale parameter is actually inverse scale)
        unscaled_tensors = []
        for tensor in tensor_lists[0]:
            unscaled_tensors.append(tensor * scale.item())

        return multi_tensor_l2norm_torch(chunk_size, noop_flag, [unscaled_tensors], per_tensor)

    def multi_tensor_adam(self, *args, **kwargs):
        if not args and not kwargs:
            return multi_tensor_adam_torch
        return multi_tensor_adam_torch(*args, **kwargs)

    def multi_tensor_adam_param_remainder(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_param_remainder - not implemented in reference backend")

    def multi_tensor_adam_fp8(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_fp8 - not implemented in reference backend")

    def multi_tensor_adam_capturable(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_capturable - not implemented in reference backend")

    def multi_tensor_adam_capturable_master(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_adam_capturable_master - not implemented in reference backend")

    def multi_tensor_sgd(self, *args, **kwargs) -> None:
        return multi_tensor_sgd_torch(*args, **kwargs)

    def multi_tensor_compute_scale_and_scale_inv(self, *args, **kwargs) -> None:
        raise NotImplementedError("multi_tensor_compute_scale_and_scale_inv - not implemented in reference backend")

    def bulk_overlap_ag_with_external_gemm(self, *args, **kwargs) -> Any:
        raise NotImplementedError("bulk_overlap_ag_with_external_gemm - not implemented in reference backend")

    def create_fp8_tensor_meta(self) -> FP8TensorMeta:
        return FP8TensorMeta()

    def create_comm_overlap_helper(self, *args, **kwargs) -> Any:
        raise NotImplementedError("create_comm_overlap_helper - not implemented in reference backend")

    def create_comm_overlap(self, *args, **kwargs) -> Any:
        raise NotImplementedError("create_comm_overlap - not implemented in reference backend")

    def create_comm_overlap_p2p(self, *args, **kwargs) -> Any:
        raise NotImplementedError("create_comm_overlap_p2p - not implemented in reference backend")
