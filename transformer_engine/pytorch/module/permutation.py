# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import os
import torch
import warnings

# TODO by Jiang Shao, add parameter `out` which can be optionally given to be used as output buffers.

################################################################################################
##
## PermuteMoE topK
##
################################################################################################

class PermuteMoE_topK(torch.autograd.Function):

  workspace_fw=None
  dtype=None
  max_expanded_token_num=0

  @staticmethod
  def forward(ctx, 
              input_act: torch.Tensor,
              indices: torch.Tensor,
              num_out_tokens: int,
              max_token_num: int):
    # Empty input check
    if not input_act.numel():
      return input_act, None

    # Device check
    if input_act.is_cpu:
      raise RuntimeError("[Error] The input `input_act` of permute_topK op is on the device: CPU!")
    if indices.is_cpu:
      warnings.warn("[Warning] The input `indices` of permute_topK op is on the device: CPU!")
      expert_for_rows = expert_for_rows.cuda()

    # Shape check
    if input_act.size(0) != indices.size(0):
      raise RuntimeError(f"[Error] permute_topK op input `indices` shape mismatch! "
                         f"Expect {input_act.size(0)}, but got {indices.size(0)}.")

    # Data type check
    if indices.dtype != torch.int32:
      warnings.warn(f"[Warning] The data type of the input `indices` of permute_topK op is {indices.dtype}! "
            "The recommended type is torch.int32.")
      indices = indices.to(torch.int32)

    # Contiguous check
    if not input_act.is_contiguous():
      warnings.warn("[Warning] The input `input_act` of permute_topK op is discontiguous!")
      input_act = input_act.contiguous()
    if not indices.is_contiguous():
      warnings.warn("[Warning] The input `indices` of permute_topK op is discontiguous!")
      indices = indices.contiguous()

    num_topK = indices.size(1)

    input_max_expanded_token_num = max(max_token_num, input_act.size(0)) * num_topK
    if PermuteMoE_topK.max_expanded_token_num < input_max_expanded_token_num:
      PermuteMoE_topK.max_expanded_token_num = input_max_expanded_token_num
      PermuteMoE_topK.workspace_fw = []

    if PermuteMoE_topK.dtype != input_act.dtype:
      PermuteMoE_topK.dtype = input_act.dtype
      PermuteMoE_topK.workspace_fw = []

    permuted_act, row_id_map, PermuteMoE_topK.workspace_fw = torch.ops.moe_unit_ops.moe_permute_topK_op(
      input_act,
      indices,
      num_out_tokens,
      PermuteMoE_topK.workspace_fw,
      PermuteMoE_topK.max_expanded_token_num)

    ctx.row_id_map = row_id_map
    ctx.num_tokens = indices.size(0)
    ctx.num_topK = indices.size(1)
    return permuted_act, row_id_map


  @staticmethod
  def backward(ctx, permuted_act_grad, _):
    # Empty input check
    if not permuted_act_grad.numel():
      return permuted_act_grad, None, None, None
    
    if not permuted_act_grad.is_contiguous():
      permuted_act_grad = permuted_act_grad.contiguous()

    row_id_map = ctx.row_id_map
    num_tokens = ctx.num_tokens
    num_topK = ctx.num_topK

    unpermuted_act_grad = torch.ops.moe_unit_ops.moe_recover_topK_op(
      permuted_act_grad,
      row_id_map,
      None,
      num_tokens,
      num_topK)
    return unpermuted_act_grad, None, None, None

################################################################################################
##
## UnpermuteMoE topK
##
################################################################################################

class UnpermuteMoE_topK(torch.autograd.Function):

  @staticmethod
  def forward(ctx,
              input_act: torch.Tensor,
              row_id_map: torch.Tensor,
              probs: torch.Tensor):
    # Empty input check
    if not input_act.numel():
      ctx.probs = probs
      return input_act

    # None probs check
    if probs is not None:
      if probs.is_cpu:
        warnings.warn("[Warning] The input `probs` of unpermute_topK op is on the device: CPU!")
        probs = probs.cuda()
      if probs.dtype != torch.float32:
        warnings.warn(f"[Warning] The data type of the input `probs` of unpermute_topK op is {probs.dtype}! "
              "The recommended type is torch.float32.")
        probs = probs.to(torch.float32)
      if not probs.is_contiguous():
        warnings.warn("[Warning] The input `probs` of unpermute_topK op is discontiguous!")
        probs = probs.contiguous()

    # Device check
    if input_act.is_cpu:
      raise RuntimeError("[Error] The input `input_act` of unpermute_topK op is on the device: CPU!")
    if row_id_map.is_cpu:
      warnings.warn("[Warning] The input `row_id_map` of unpermute_topK op is on the device: CPU!")
      row_id_map = row_id_map.cuda()

    # Data type check
    if row_id_map.dtype != torch.int32:
      warnings.warn(f"[Warning] The data type of the input `row_id_map` of unpermute_topK op is {row_id_map.dtype}! "
            "The recommended type is torch.int32.")
      row_id_map = row_id_map.to(torch.int32)

    # Contiguous check
    if not input_act.is_contiguous():
      warnings.warn("[Warning] The input `input_act` of unpermute_topK op is discontiguous!")
      input_act = input_act.contiguous()
    if not row_id_map.is_contiguous():
      warnings.warn("[Warning] The input `row_id_map` of unpermute_topK op is discontiguous!")
      row_id_map = row_id_map.contiguous()

    num_topK = probs.size(1) if probs is not None else 1
    num_tokens = probs.size(0) if probs is not None else row_id_map.size(0)

    unpermuted_output = torch.ops.moe_unit_ops.moe_recover_topK_op(
      input_act,
      row_id_map,
      probs,
      num_tokens,
      num_topK)

    ctx.save_for_backward(input_act, row_id_map, probs)
    return unpermuted_output

  @staticmethod
  def backward(ctx, unpermuted_act_grad):
    # Empty input check
    if not unpermuted_act_grad.numel():
      return unpermuted_act_grad, None, ctx.probs

    if not unpermuted_act_grad.is_contiguous():
      unpermuted_act_grad = unpermuted_act_grad.contiguous()

    input_act, row_id_map, probs = ctx.saved_tensors

    act_grad = None
    if ctx.needs_input_grad[0]:
      act_grad, prob_grad = torch.ops.moe_unit_ops.moe_recover_topK_bwd_op(
        unpermuted_act_grad,
        input_act,
        row_id_map,
        probs)
    
    if not ctx.needs_input_grad[2]:
      prob_grad = None
    return act_grad, None, prob_grad

################################################################################################
##
## Ops Wrapper
##
################################################################################################

def permute(input_act, indices, num_out_tokens=-1, max_token_num=-1):
  return PermuteMoE_topK.apply(input_act, indices, num_out_tokens, max_token_num)

def unpermute(input_act, row_id_map, probs):
  return UnpermuteMoE_topK.apply(input_act, row_id_map, probs)
