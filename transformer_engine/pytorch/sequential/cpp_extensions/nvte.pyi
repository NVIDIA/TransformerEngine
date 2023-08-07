import torch
from enum import Enum

class QKV_Layout(Enum):
    NVTE_NOT_INTERLEAVED = 0
    NVTE_QKV_INTERLEAVED = 1
    NVTE_KV_INTERLEAVED = 2

class Bias_Type(Enum):
    NVTE_NO_BIAS = 0
    NVTE_PRE_SCALE_BIAS = 1
    NVTE_POST_SCALE_BIAS = 2

class Mask_Type(Enum):
    NVTE_NO_MASK = 0
    NVTE_PADDING_MASK = 1
    NVTE_CAUSAL_MASK = 2

class Fused_Attn_Backend(Enum):
    NVTE_No_Backend = -1
    NVTE_F16_max512_seqlen = 0
    NVTE_F16_arbitrary_seqlen = 1
    NVTE_FP8 = 2

class DType(Enum):
    kNVTEByte = 0
    kNVTEInt32 = 1
    kNVTEInt64 = 2
    kNVTEFloat32 = 3
    kNVTEFloat16 = 4
    kNVTEBFloat16 = 5
    kNVTEFloat8E4M3 = 6
    kNVTEFloat8E5M2 = 7

class Tensor:
    def __init__(self, dtype: DType, data: torch.Tensor, amax: torch.Tensor, scale: torch.Tensor, scale_inv: torch.Tensor) -> None: ...

def gelu(input: Tensor, output: Tensor) -> None: ...
def dgelu(grad: Tensor, input: Tensor, output: Tensor) -> None: ...
def geglu(input: Tensor, output: Tensor) -> None: ...
def dgeglu(grad: Tensor, input: Tensor, output: Tensor) -> None: ...
def relu(input: Tensor, output: Tensor) -> None: ...
def drelu(grad: Tensor, input: Tensor, output: Tensor) -> None: ...
def swiglu(input: Tensor, output: Tensor) -> None: ...
def dswiglu(grad: Tensor, input: Tensor, output: Tensor) -> None: ...
def reglu(input: Tensor, output: Tensor) -> None: ...
def dreglu(grad: Tensor, input: Tensor, output: Tensor) -> None: ...
def fp8_quantize(input: Tensor, output: Tensor) -> None: ...
def fp8_dequantize(input: Tensor, output: Tensor) -> None: ...
def get_fused_attn_backend(q_dtype: DType, kv_dtype: DType, qkv_layout: QKV_Layout, bias_type: Bias_Type, attn_mask_type: Mask_Type, dropout: float, max_seqlen_q: int, max_seqlen_kv: int, head_dim: int) -> Fused_Attn_Backend: ...
def fused_attn_fwd_qkvpacked(QKV: Tensor, Bias: Tensor, S: Tensor, O: Tensor, Aux_CTX_Tensors: list[Tensor], cu_seqlens: Tensor, rng_state: Tensor, max_seqlen: int, is_training: bool, attn_scale: float, dropout: float, qkv_layout: QKV_Layout, bias_type: Bias_Type, attn_mask_type: Mask_Type, workspace: Tensor) -> None: ...
def fused_attn_bwd_qkvpacked(QKV: Tensor, O: Tensor, dO: Tensor, S: Tensor, dP: Tensor, Aux_CTX_Tensors: list[Tensor], dQKV: Tensor, dBias: Tensor, cu_seqlens: Tensor, max_seqlen: int, attn_scale: float, dropout: float, qkv_layout: QKV_Layout, bias_type: Bias_Type, attn_mask_type: Mask_Type, workspace: Tensor) -> None: ...
def fused_attn_fwd_kvpacked(Q: Tensor, KV: Tensor, Bias: Tensor, S: Tensor, O: Tensor, Aux_CTX_Tensors: list[Tensor], cu_seqlens_q: Tensor, cu_seqlens_kv: Tensor, rng_state: Tensor, max_seqlen_q: int, max_seqlen_kv: int, is_training: bool, attn_scale: float, dropout: float, qkv_layout: QKV_Layout, bias_type: Bias_Type, attn_mask_type: Mask_Type, workspace: Tensor) -> None: ...
def fused_attn_bwd_kvpacked(Q: Tensor, KV: Tensor, O: Tensor, dO: Tensor, S: Tensor, dP: Tensor, Aux_CTX_Tensors: list[Tensor], dQ: Tensor, dKV: Tensor, dBias: Tensor, cu_seqlens_q: Tensor, cu_seqlens_kv: Tensor, max_seqlen_q: int, max_seqlen_kv: int, attn_scale: float, dropout: float, qkv_layout: QKV_Layout, bias_type: Bias_Type, attn_mask_type: Mask_Type, workspace: Tensor) -> None: ...
def cublas_gemm(A: Tensor, B: Tensor, D: Tensor, bias: Tensor, pre_gelu_out: Tensor, transa: bool, transb: bool, grad: bool, workspace: Tensor, accumulate: bool, use_split_accumulator: bool, math_sm_count: int) -> None: ...
def layernorm_fwd(x: Tensor, gamma: Tensor, beta: Tensor, epsilon: float, z: Tensor, mu: Tensor, rsigma: Tensor, multiprocessorCount: int, workspace: Tensor, barrier: Tensor) -> None: ...
def layernorm1p_fwd(x: Tensor, gamma: Tensor, beta: Tensor, epsilon: float, z: Tensor, mu: Tensor, rsigma: Tensor, multiprocessorCount: int, workspace: Tensor, barrier: Tensor) -> None: ...
def layernorm_bwd(dz: Tensor, x: Tensor, mu: Tensor, rsigma: Tensor, gamma: Tensor, dx: Tensor, dgamma: Tensor, dbeta: Tensor, dgamma_part: Tensor, dbeta_part: Tensor, multiprocessorCount: int, workspace: Tensor, barrier: Tensor) -> None: ...
def layernorm1p_bwd(dz: Tensor, x: Tensor, mu: Tensor, rsigma: Tensor, gamma: Tensor, dx: Tensor, dgamma: Tensor, dbeta: Tensor, dgamma_part: Tensor, dbeta_part: Tensor, multiprocessorCount: int, workspace: Tensor, barrier: Tensor) -> None: ...
def rmsnorm_fwd(x: Tensor, gamma: Tensor, epsilon: float, z: Tensor, rsigma: Tensor, multiprocessorCount: int, workspace: Tensor, barrier: Tensor) -> None: ...
def rmsnorm_bwd(dz: Tensor, x: Tensor, rsigma: Tensor, gamma: Tensor, dx: Tensor, dgamma: Tensor, dgamma_part: Tensor, multiprocessorCount: int, workspace: Tensor, barrier: Tensor) -> None: ...
def scaled_softmax_forward(input: Tensor, softmax_results: Tensor, scale_factor: float) -> None: ...
def scaled_softmax_backward(incoming_grads: Tensor, softmax_results: Tensor, output_grads: Tensor, scale_factor: float) -> None: ...
def scaled_masked_softmax_forward(input: Tensor, mask: Tensor, softmax_results: Tensor, scale_factor: float) -> None: ...
def scaled_masked_softmax_backward(incoming_grads: Tensor, softmax_results: Tensor, output_grads: Tensor, scale_factor: float) -> None: ...
def scaled_upper_triang_masked_softmax_forward(input: Tensor, softmax_results: Tensor, scale_factor: float) -> None: ...
def scaled_upper_triang_masked_softmax_backward(incoming_grads: Tensor, softmax_results: Tensor, output_grads: Tensor, scale_factor: float) -> None: ...
def cast_transpose(input: Tensor, cast_output: Tensor, transposed_output: Tensor) -> None: ...
def transpose(input: Tensor, transposed_output: Tensor) -> None: ...
def cast_transpose_dbias(input: Tensor, cast_output: Tensor, transposed_output: Tensor, dbias: Tensor, workspace: Tensor) -> None: ...
def fp8_transpose_dbias(input: Tensor, transposed_output: Tensor, dbias: Tensor, workspace: Tensor) -> None: ...
def cast_transpose_dbias_dgelu(input: Tensor, gelu_input: Tensor, cast_output: Tensor, transposed_output: Tensor, dbias: 2, workspace: Tensor) -> None: ...
def dgeglu_cast_transpose(input: Tensor, geglu_input: Tensor, cast_output: Tensor, transposed_output: Tensor) -> None: ...