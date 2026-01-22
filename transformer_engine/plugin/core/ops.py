# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
from enum import IntEnum
from contextlib import nullcontext
import torch

from .logger_manager import get_logger
logger = get_logger()

class DType(IntEnum):
    kByte = 0
    kInt16 = 1
    kInt32 = 2
    kInt64 = 3
    kFloat32 = 4
    kFloat16 = 5
    kBFloat16 = 6
    kFloat8E4M3 = 7
    kFloat8E5M2 = 8
    kFloat8E8M0 = 9
    kFloat4E2M1 = 10
    kNumTypes = 11

class Float8BlockScaleTensorFormat(IntEnum):
    GEMM_READY = 0
    COMPACT = 1

class NVTE_Activation_Type(IntEnum):
    GELU = 0
    GEGLU = 1
    SILU = 2
    SWIGLU = 3
    RELU = 4
    REGLU = 5
    QGELU = 6
    QGEGLU = 7
    SRELU = 8
    SREGLU = 9
    CLAMPED_SWIGLU = 10

class NVTE_Softmax_Type(IntEnum):
    NVTE_VANILLA_SOFTMAX = 0
    NVTE_OFF_BY_ONE_SOFTMAX = 1
    NVTE_LEARNABLE_SOFTMAX = 2

class CommGemmOverlapRole(IntEnum):
    INPUT = 0
    OUTPUT = 1

class FP8FwdTensors(IntEnum):
    GEMM1_INPUT = 0
    GEMM1_WEIGHT = 1
    GEMM1_OUTPUT = 2
    GEMM2_INPUT = 3
    GEMM2_WEIGHT = 4
    GEMM2_OUTPUT = 5
    GEMM3_INPUT = 6
    GEMM3_WEIGHT = 7
    GEMM3_OUTPUT = 8

class FP8BwdTensors(IntEnum):
    GRAD_OUTPUT1 = 0
    GRAD_INPUT1 = 1
    GRAD_OUTPUT2 = 2
    GRAD_INPUT2 = 3
    GRAD_OUTPUT3 = 4
    GRAD_INPUT3 = 5

class NVTE_Bias_Type(IntEnum):
    NVTE_NO_BIAS = 0
    NVTE_PRE_SCALE_BIAS = 1
    NVTE_POST_SCALE_BIAS = 2
    NVTE_ALIBI = 3

class NVTE_Mask_Type(IntEnum):
    NVTE_NO_MASK = 0
    NVTE_PADDING_MASK = 1
    NVTE_CAUSAL_MASK = 2
    NVTE_PADDING_CAUSAL_MASK = 3
    NVTE_CAUSAL_BOTTOM_RIGHT_MASK = 4
    NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK = 5

class NVTE_Fused_Attn_Backend(IntEnum):
    NVTE_No_Backend = -1
    NVTE_F16_max512_seqlen = 0
    NVTE_F16_arbitrary_seqlen = 1
    NVTE_FP8 = 2

class NVTE_QKV_Format(IntEnum):
    NVTE_SBHD = 0
    NVTE_BSHD = 1
    NVTE_THD = 2
    NVTE_BSHD_2SBHD = 3
    NVTE_SBHD_2BSHD = 4
    NVTE_THD_2BSHD = 5
    NVTE_THD_2SBHD = 6

class NVTE_QKV_Layout(IntEnum):
    NVTE_SB3HD = 0
    NVTE_SBH3D = 1
    NVTE_SBHD_SB2HD = 2
    NVTE_SBHD_SBH2D = 3
    NVTE_SBHD_SBHD_SBHD = 4
    NVTE_BS3HD = 5
    NVTE_BSH3D = 6
    NVTE_BSHD_BS2HD = 7
    NVTE_BSHD_BSH2D = 8
    NVTE_BSHD_BSHD_BSHD = 9
    NVTE_T3HD = 10
    NVTE_TH3D = 11
    NVTE_THD_T2HD = 12
    NVTE_THD_TH2D = 13
    NVTE_THD_THD_THD = 14
    NVTE_SBHD_BSHD_BSHD = 15
    NVTE_BSHD_SBHD_SBHD = 16
    NVTE_THD_BSHD_BSHD = 17
    NVTE_THD_SBHD_SBHD = 18
    NVTE_Paged_KV_BSHD_BSHD_BSHD = 19
    NVTE_Paged_KV_BSHD_SBHD_SBHD = 20
    NVTE_Paged_KV_SBHD_BSHD_BSHD = 21
    NVTE_Paged_KV_SBHD_SBHD_SBHD = 22
    NVTE_Paged_KV_THD_BSHD_BSHD = 23
    NVTE_Paged_KV_THD_SBHD_SBHD = 24

class CommOverlapType(IntEnum):
    RS = 0
    AG = 1

class CommOverlapAlgo(IntEnum):
    BULK_OVERLAP_AG = 0
    BULK_OVERLAP_RS = 1
    SPLIT_PIPELINED_AG_P2P = 2
    SPLIT_PIPELINED_RS = 3
    SPLIT_PIPELINED_RS_P2P = 4
    ATOMIC_GEMM_RS = 5
    ATOMIC_GEMM_AG_P2P = 6
    ATOMIC_GEMM_RS_P2P = 7
    EXTERNAL_BULK_OVERLAP_AG = 8

class FP8TensorMeta:
    def __init__(self):
        self.scale: Optional[torch.Tensor] = None
        self.scale_inv: Optional[torch.Tensor] = None
        self.amax_history: Optional[torch.Tensor] = None

class CommGemmOverlapAlgoConfig:
    def __init__(self, *args, **kwargs):
        pass

class FusedAdamCUDAKernel:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FusedAdamCUDAKernel requires CUDA extensions. "
            "Not supported in FL mode."
        )

class FusedSGDCUDAKernel:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FusedSGDCUDAKernel requires CUDA extensions. "
            "Not supported in FL mode."
        )

class CommOverlapHelper:
    def __init__(self, world_group=None, intra_node_group=None):
        self.world_group = world_group
        self.intra_node_group = intra_node_group

class CommOverlap:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CommOverlap should be created via backend.create_comm_overlap(). "
            "Direct instantiation is not supported in FL mode."
        )

class CommOverlapP2P:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CommOverlapP2P should be created via backend.create_comm_overlap_p2p(). "
            "Direct instantiation is not supported in FL mode."
        )

class TEFLBackendBase(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    def get_flash_attention_class(self) -> Type["FlashAttentionBase"]:
        raise NotImplementedError

    def get_attention_backend(self, attention_params=None):
        raise NotImplementedError

    def quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        output: Optional[torch.Tensor] = None,
        noop: Optional[torch.Tensor] = None,
    ) -> Any:
        raise NotImplementedError

    def dequantize(
        self,
        input: torch.Tensor,
        otype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError

    def bgrad_quantize(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

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
        raise NotImplementedError

    def te_general_grouped_gemm(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def gelu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def geglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def qgelu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def qgeglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def relu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def reglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def srelu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def sreglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def silu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def swiglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def clamped_swiglu(
        self,
        input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        raise NotImplementedError

    def dgelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def dgeglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def dqgelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def dqgeglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def drelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def dreglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def dsrelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def dsreglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def dsilu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Any:
        raise NotImplementedError

    def clamped_dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        raise NotImplementedError

    def dbias_dgelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

    def dbias_dsilu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

    def dbias_drelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

    def dbias_dqgelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

    def dbias_dsrelu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def rmsnorm_bwd_add(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def multi_tensor_quantize(
        self,
        tensor_list: List[torch.Tensor],
        quantizer_list: List[Any],
    ) -> List[Any]:
        raise NotImplementedError

    def split_quantize(
        self,
        tensor: torch.Tensor,
        split_sections: List[int],
        quantizer_list: List[Any],
    ) -> List[Any]:
        raise NotImplementedError

    def moe_permute_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def moe_permute_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def moe_unpermute_fwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def moe_unpermute_bwd(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def scaled_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def scaled_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def scaled_masked_softmax_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def scaled_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def scaled_upper_triang_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def scaled_upper_triang_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def scaled_aligned_causal_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def scaled_aligned_causal_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_fused_attn_backend(
        self,
        *args,
        **kwargs,
    ) -> int:
        raise NotImplementedError

    def fused_attn_fwd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def fused_attn_bwd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def fa_prepare_fwd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def fa_prepare_bwd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def copy_to_kv_cache(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def convert_thd_to_bshd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def convert_bshd_to_thd(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def fused_rope_forward(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def fused_rope_backward(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def fused_qkv_rope_forward(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def fused_qkv_rope_backward(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def fused_topk_with_score_function_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        num_groups: int,
        group_topk: int,
        scaling_factor: float,
        score_function: Any,
        expert_bias: Optional[torch.Tensor],
    ) -> Any:
        raise NotImplementedError

    def fused_topk_with_score_function_bwd(
        self,
        num_tokens: int,
        num_experts: int,
        routing_map: torch.Tensor,
        intermediate_output: torch.Tensor,
        grad_probs: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        scaling_factor: float,
        score_function: Any,
    ) -> Any:
        raise NotImplementedError

    def fused_score_for_moe_aux_loss_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        score_function: Any,
    ) -> Any:
        raise NotImplementedError

    def fused_score_for_moe_aux_loss_bwd(
        self,
        num_tokens: int,
        num_experts: int,
        intermediate_output: torch.Tensor,
        grad_scores: torch.Tensor,
        topk: int,
        score_function: Any,
    ) -> Any:
        raise NotImplementedError

    def fused_moe_aux_loss_fwd(
        self,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_num_tokens: int,
        num_experts: int,
        num_rows: int,
        num_cols: int,
        topk: int,
        coeff: float,
    ) -> Any:
        raise NotImplementedError

    def fused_moe_aux_loss_bwd(
        self,
        Const_buf: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_rows: int,
        num_cols: int,
        grad_aux_loss: torch.Tensor,
    ) -> Any:
        raise NotImplementedError

    def dropout_fwd(
        self,
        input: torch.Tensor,
        dropout_probability: float,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def dropout_bwd(
        self,
        grad_output: torch.Tensor,
        mask: torch.Tensor,
        dropout_probability: float,
        grad_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def fp8_transpose(
        self,
        input: torch.Tensor,
        dtype: Any,
        *,
        out: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def swap_first_dims(
        self,
        tensor: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def compute_amax(
        self,
        input: torch.Tensor,
        amax: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def fused_amax_and_scale_update_after_reduction(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def fp8_block_scaling_compute_partial_amax(
        self,
        tensor: torch.Tensor,
        amax: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
    ) -> None:
        raise NotImplementedError

    def fp8_block_scaling_partial_cast(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        scale: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
        out_dtype: Any,
    ) -> None:
        raise NotImplementedError

    def fused_multi_row_padding(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def fused_multi_row_unpadding(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def get_cublasLt_version(self) -> int:
        raise NotImplementedError

    def get_cudnn_version(self) -> int:
        raise NotImplementedError

    def get_num_cublas_streams(self) -> int:
        raise NotImplementedError

    def thd_read_half_tensor(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def thd_second_half_lse_correction(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def thd_read_second_half_lse(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def thd_out_correction(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def thd_grad_correction(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def thd_get_partitioned_indices(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError

    def init_nvshmem_backend(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def create_nvshmem_tensor(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def nvshmem_send_on_current_stream(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def nvshmem_wait_on_current_stream(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def nvshmem_finalize(self) -> None:
        raise NotImplementedError

    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ) -> None:
        raise NotImplementedError

    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError

    def multi_tensor_unscale_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: torch.Tensor,
        per_tensor: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError

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
        raise NotImplementedError

    def multi_tensor_adam_param_remainder(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def multi_tensor_adam_fp8(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def multi_tensor_adam_capturable(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def multi_tensor_adam_capturable_master(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def multi_tensor_sgd(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def multi_tensor_compute_scale_and_scale_inv(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def bulk_overlap_ag_with_external_gemm(
        self,
        allgather_communicator: Any,
        send_stream: Any,
        recv_stream: Any,
    ) -> Any:
        raise NotImplementedError

    def create_fp8_tensor_meta(self) -> FP8TensorMeta:
        raise NotImplementedError

    def create_comm_overlap_helper(
        self,
        world_group: Optional[Any] = None,
        intra_node_group: Optional[Any] = None,
    ) -> Any:
        raise NotImplementedError

    def create_comm_overlap(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        helper: Any,
        tp_size: int,
        num_splits: int = 3,
        num_max_streams: int = 3,
        comm_cga_size: int = 2,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 16,
        set_sm_margin: bool = True,
        atomic_gemm: bool = False,
        rs_overlap_first_gemm: bool = False,
    ) -> Any:
        raise NotImplementedError

    def create_comm_overlap_p2p(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        helper: Any,
        tp_size: int,
        comm_type: Any,
        num_max_streams: int = 3,
        comm_cga_size: int = 1,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 1,
        set_sm_margin: bool = False,
        atomic_gemm: bool = False,
        use_ce: bool = True,
        aggregate: bool = False,
    ) -> Any:
        raise NotImplementedError

class FlashAttentionBase(torch.nn.Module, ABC):

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = None,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.attention_dropout_ctx = attention_dropout_ctx or nullcontext
        self.attention_type = attention_type
        self.layer_number = 1 if layer_number is None else layer_number
        self.deterministic = deterministic

        # For fallback support
        self._manager = None
        self._init_params = None

    @abstractmethod
    def _forward_impl(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cp_group: Optional[Any] = None,
        cp_global_ranks: Optional[List[int]] = None,
        cp_stream: Optional[torch.cuda.Stream] = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers: Optional[Any] = None,
        inference_params: Optional[Any] = None,
        flash_attention_backend: Optional[Any] = None,
        fp8_output: bool = False,
    ) -> torch.Tensor:
        """
        Actual forward implementation - subclasses must implement this.

        This method contains the backend-specific logic for flash attention.
        """
        raise NotImplementedError("Subclasses must implement _forward_impl()")

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cp_group: Optional[Any] = None,
        cp_global_ranks: Optional[List[int]] = None,
        cp_stream: Optional[torch.cuda.Stream] = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers: Optional[Any] = None,
        inference_params: Optional[Any] = None,
        flash_attention_backend: Optional[Any] = None,
        fp8_output: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with automatic fallback support and caching.
        Delegates to OpManager.call_with_custom_impl for unified dispatch.
        """
        if self._manager is None:
            return self._forward_impl(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask,
                qkv_layout=qkv_layout,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                attn_mask_type=attn_mask_type,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                cp_group=cp_group,
                cp_global_ranks=cp_global_ranks,
                cp_stream=cp_stream,
                cp_comm_type=cp_comm_type,
                fp8=fp8,
                fp8_meta=fp8_meta,
                quantizers=quantizers,
                inference_params=inference_params,
                flash_attention_backend=flash_attention_backend,
                fp8_output=fp8_output,
            )

        def call_impl_fn(impl_class):
            if impl_class == self.__class__:
                return self._forward_impl(
                    query_layer=query_layer,
                    key_layer=key_layer,
                    value_layer=value_layer,
                    attention_mask=attention_mask,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    attn_mask_type=attn_mask_type,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    cp_group=cp_group,
                    cp_global_ranks=cp_global_ranks,
                    cp_stream=cp_stream,
                    cp_comm_type=cp_comm_type,
                    fp8=fp8,
                    fp8_meta=fp8_meta,
                    quantizers=quantizers,
                    inference_params=inference_params,
                    flash_attention_backend=flash_attention_backend,
                    fp8_output=fp8_output,
                )
            else:
                fallback_instance = impl_class(**self._init_params)
                fallback_instance._manager = self._manager
                fallback_instance._init_params = self._init_params
                return fallback_instance._forward_impl(
                    query_layer=query_layer,
                    key_layer=key_layer,
                    value_layer=value_layer,
                    attention_mask=attention_mask,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    attn_mask_type=attn_mask_type,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    cp_group=cp_group,
                    cp_global_ranks=cp_global_ranks,
                    cp_stream=cp_stream,
                    cp_comm_type=cp_comm_type,
                    fp8=fp8,
                    fp8_meta=fp8_meta,
                    quantizers=quantizers,
                    inference_params=inference_params,
                    flash_attention_backend=flash_attention_backend,
                    fp8_output=fp8_output,
                )

        return self._manager.call_with_custom_impl(
            op_name="get_flash_attention_class",
            current_impl_class=self.__class__,
            call_impl_fn=call_impl_fn,
        )

    @property
    def backend_name(self) -> str:
        return self.__class__.__name__


class TEFLModule:
    def __init__(self, manager=None):
        """
        Initialize TEFLModule.

        Args:
            manager: OpManager instance for operator dispatch.
                       If None, will use the global default OpManager.
        """
        # Import here to avoid circular dependency
        from .manager import get_default_manager
        self._manager = manager if manager is not None else get_default_manager()

        self.DType = DType
        self.Float8BlockScaleTensorFormat = Float8BlockScaleTensorFormat
        self.FP8FwdTensors = FP8FwdTensors
        self.FP8BwdTensors = FP8BwdTensors
        self.FP8TensorMeta = FP8TensorMeta
        self.NVTE_Activation_Type = NVTE_Activation_Type
        self.NVTE_Bias_Type = NVTE_Bias_Type
        self.NVTE_Mask_Type = NVTE_Mask_Type
        self.NVTE_Softmax_Type = NVTE_Softmax_Type
        self.NVTE_Fused_Attn_Backend = NVTE_Fused_Attn_Backend
        self.NVTE_QKV_Format = NVTE_QKV_Format
        self.NVTE_QKV_Layout = NVTE_QKV_Layout
        self.CommOverlapType = CommOverlapType
        self.CommOverlapAlgo = CommOverlapAlgo
        self.CommGemmOverlapRole = CommGemmOverlapRole

        self.CommOverlapHelper = CommOverlapHelper
        self.CommOverlap = CommOverlap
        self.CommOverlapP2P = CommOverlapP2P
        self.CommGemmOverlapAlgoConfig = CommGemmOverlapAlgoConfig

        self.FusedAdamCUDAKernel = FusedAdamCUDAKernel
        self.FusedSGDCUDAKernel = FusedSGDCUDAKernel

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically resolve operators through OpManager.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Verify the operator exists before returning the bound call method
        try:
            self._manager.ensure_initialized()
            available_ops = self._manager.registry.list_operators()
            if name not in available_ops:
                raise AttributeError(
                    f"Operator '{name}' not found. "
                    f"Available operators: {available_ops}"
                )
        except RuntimeError as e:
            # Re-raise as AttributeError for better error messages
            raise AttributeError(
                f"Error accessing operator '{name}': {e}"
            ) from e

        # Return a bound call method for this operator
        import functools
        return functools.partial(self._manager.call, name)

    def __dir__(self):
        module_attrs = [
            'DType', 'Float8BlockScaleTensorFormat', 'FP8FwdTensors', 'FP8BwdTensors',
            'FP8TensorMeta', 'NVTE_Activation_Type', 'NVTE_Bias_Type', 'NVTE_Mask_Type',
            'NVTE_Softmax_Type', 'NVTE_Fused_Attn_Backend', 'NVTE_QKV_Format', 'NVTE_QKV_Layout',
            'CommOverlapType', 'CommOverlapAlgo', 'CommGemmOverlapRole',
            'CommOverlapHelper', 'CommOverlap', 'CommOverlapP2P', 'CommGemmOverlapAlgoConfig',
            'FusedAdamCUDAKernel', 'FusedSGDCUDAKernel'
        ]

        # Add operator names from OpManager's registry
        op_attrs = self._manager.registry.list_operators()

        return list(set(module_attrs + op_attrs))

    def __getitem__(self, key: str):
        return self.__getattr__(key)

    @property
    def __all__(self):
        return self.__dir__()

    def flash_attention(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = None,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> "FlashAttentionBase":
        """
        Get FlashAttention implementation through OpManager.
        """
        # Get the flash attention class getter through OpManager.call
        # This provides the same fallback support and logging as other operators
        flash_attn_class = self._manager.call("get_flash_attention_class")

        # Prepare initialization parameters
        init_params = {
            'softmax_scale': softmax_scale,
            'attention_dropout': attention_dropout,
            'attention_dropout_ctx': attention_dropout_ctx,
            'attention_type': attention_type,
            'layer_number': layer_number,
            'deterministic': deterministic,
        }

        # Instantiate the FlashAttention
        instance = flash_attn_class(**init_params)

        # Set manager and init_params for fallback support
        instance._manager = self._manager
        instance._init_params = init_params

        return instance

    def __repr__(self) -> str:
        op_count = len(self._manager.registry.list_operators())
        return f"TEFLModule(operators={op_count}, manager={self._manager.__class__.__name__})"

# Global singleton instance
_global_tefl_module: Optional[TEFLModule] = None
_tefl_module_lock = None

def get_tefl_module() -> TEFLModule:
    """
    Get or create the global TEFLModule instance.

    This function returns a singleton TEFLModule that uses the default OpManager.
    The instance is created lazily on first access.

    Returns:
        The global TEFLModule instance

    Example:
        >>> import core as te_fl
        >>> # Or explicitly:
        >>> from core.base import get_tefl_module
        >>> te_fl = get_tefl_module()
        >>> result = te_fl.rmsnorm_fwd(input, weight, eps=1e-5)
    """
    global _global_tefl_module, _tefl_module_lock

    if _global_tefl_module is None:
        # Import here to avoid issues at module load time
        import threading

        if _tefl_module_lock is None:
            _tefl_module_lock = threading.RLock()

        with _tefl_module_lock:
            if _global_tefl_module is None:
                _global_tefl_module = TEFLModule()

    return _global_tefl_module

def reset_tefl_module() -> None:
    """
    Reset the global TEFLModule instance.

    This is primarily useful for testing. After calling this function,
    the next call to get_tefl_module() will create a fresh instance.

    Warning:
        This function is not thread-safe and should only be used in
        single-threaded test environments.
    """
    global _global_tefl_module, _tefl_module_lock

    if _tefl_module_lock is None:
        import threading
        _tefl_module_lock = threading.RLock()

    with _tefl_module_lock:
        _global_tefl_module = None

# Backward compatibility functions
def get_registry():
    """
    Get the global OpRegistry instance (via OpManager).

    DEPRECATED: Use get_default_manager().registry instead.

    This function is kept for backward compatibility with code that
    expects the old API.

    Returns:
        The OpRegistry instance from the default OpManager

    Example:
        >>> from core.base import get_registry
        >>> registry = get_registry()
        >>> ops = registry.list_operators()
    """
    from .manager import get_default_manager
    return get_default_manager().registry

def get_manager():
    """
    Get the global OpManager instance.

    This is the recommended way to access the OpManager.

    Returns:
        The default OpManager instance

    Example:
        >>> from core.base import get_manager
        >>> manager = get_manager()
        >>> impl_fn = manager.resolve("rmsnorm_fwd")
    """
    from .manager import get_default_manager
    return get_default_manager()

def reset_registry() -> None:
    """
    Reset the global OpManager and OpRegistry.

    DEPRECATED: Use reset_default_manager() instead.

    This function is kept for backward compatibility.
    """
    from .manager import reset_default_manager
    reset_default_manager()
    # Also reset the TEFLModule singleton since it depends on OpManager
    reset_tefl_module()
