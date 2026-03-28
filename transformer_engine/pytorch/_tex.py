# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Routing module for transformer_engine_torch.

ALL symbols come from the stable ABI module. No pybind11 .so dependency.
"""

# Everything from the stable module
from transformer_engine.pytorch._stable_torch_module import *  # noqa: F401,F403

# DType-taking function wrappers (convert enums to int for stable ops)
import transformer_engine.pytorch._stable_torch_module as _sm

def fp8_transpose(input, otype, *, out=None):
    return _sm._ops.fp8_transpose(input, int(otype), out)

def fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset, block_len, out_dtype):
    _sm._ops.fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset, block_len, int(out_dtype))

def moe_permute_fwd(input, dtype, indices, num_out_tokens, workspace, max_expanded_token_num):
    return _sm._ops.moe_permute_fwd(input, int(dtype), indices, workspace, num_out_tokens, max_expanded_token_num)

def moe_permute_bwd(input, dtype, row_id_map, prob, num_tokens, topK):
    return _sm.moe_permute_bwd(input, int(dtype), row_id_map, prob, num_tokens, topK)

def moe_unpermute_fwd(input, dtype, row_id_map, prob, num_tokens, topK):
    return _sm._ops.moe_unpermute_fwd(input, int(dtype), row_id_map, prob, num_tokens, topK)

def moe_unpermute_bwd(input_bwd, input_fwd, dtype, row_id_map, prob):
    return _sm._ops.moe_unpermute_bwd(input_bwd, input_fwd, int(dtype), row_id_map, prob)

def multi_tensor_adam_fp8(chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
                          epsilon, step, mode, bias_correction, weight_decay, fp8_dtype):
    _sm.multi_tensor_adam_fp8(chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
                              epsilon, step, mode, bias_correction, weight_decay, int(fp8_dtype))
