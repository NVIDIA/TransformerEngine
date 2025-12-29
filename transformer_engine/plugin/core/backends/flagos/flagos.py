# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
from typing import Any, List, Optional, Tuple, Union

import torch

from ...ops import TEFLBackendBase, FP8TensorMeta, NVTE_Fused_Attn_Backend

from .impl import (
    rmsnorm_fwd_fl, rmsnorm_bwd_fl,
    multi_tensor_scale_fl, multi_tensor_adam_fl,
    multi_tensor_l2_norm_fl,
    generic_gemm_fl
)

def _check_flagos_available() -> bool:
    return True


class FlagOSBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return _check_flagos_available()

    def is_available(self) -> bool:
        return _check_flagos_available()

    def get_flash_attention_class(self):
        from .attention.dot_product_attention.backends import FlashAttentionFL
        return FlashAttentionFL

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
        return generic_gemm_fl(
            A, transA, B, transB, D, quantizer, output_dtype,
            bias, bias_type, gelu, gelu_in, grad,
            workspace, workspace_size, accumulate, use_split_accumulator,
            comm_overlap=comm_overlap, comm_type=comm_type,
            extra_output=extra_output, bulk_overlap=bulk_overlap,
            alpha=alpha, beta=beta
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
        return rmsnorm_fwd_fl(
            input=input, weight=weight, eps=eps, ln_out=ln_out,
            quantizer=quantizer, odtype=otype,
            sm_margin=sm_margin, zero_centered_gamma=zero_centered_gamma,
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
        return rmsnorm_bwd_fl(
            dy=dy, x=x, rsigma=rsigma, gamma=gamma,
            sm_margin=sm_margin, zero_centered_gamma=zero_centered_gamma, eps=eps,
        )

    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ) -> None:
        return multi_tensor_scale_fl(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        result, _ = multi_tensor_l2_norm_fl(chunk_size, noop_flag, tensor_lists, per_tensor)
        return result

    def multi_tensor_adam(
        self,
        chunk_size: int = None,
        noop_flag: torch.Tensor = None,
        tensor_lists: List[List[torch.Tensor]] = None,
        lr: float = None,
        beta1: float = None,
        beta2: float = None,
        eps: float = None,
        step: int = None,
        mode: int = None,
        bias_correction: int = None,
        weight_decay: float = None,
    ):
        if chunk_size is None:
            return multi_tensor_adam_fl
        return multi_tensor_adam_fl(
            chunk_size=chunk_size, noop_flag=noop_flag, tensor_lists=tensor_lists,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            step=step, mode=mode, bias_correction=bias_correction, weight_decay=weight_decay,
        )

    def get_cublasLt_version(self) -> int:
        return 110000

    def get_cudnn_version(self) -> int:
        return 90000

    def get_num_cublas_streams(self) -> int:
        return 0

    def get_fused_attn_backend(self, *args, **kwargs) -> int:
        return NVTE_Fused_Attn_Backend.NVTE_No_Backend

    def create_fp8_tensor_meta(self) -> FP8TensorMeta:
        return FP8TensorMeta()

