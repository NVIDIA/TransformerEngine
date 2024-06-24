# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import os
import torch
import warnings

import transformer_engine_torch as tex


__all__ = [
  'Permute',
  'Unpermute',
]


class _Permute(torch.autograd.Function):

  workspace=None
  dtype=None
  max_expanded_token_num=0

  @staticmethod
  def forward(ctx, 
              inp: torch.Tensor,
              indices: torch.Tensor,
              num_out_tokens: int,
              max_token_num: int):
    # Empty input check
    if not inp.numel():
      return inp, None

    # Device check
    if inp.is_cpu:
      raise RuntimeError("[Error] The input `inp` of permute_topK op is on the device: CPU!")
    if indices.is_cpu:
      warnings.warn("[Warning] The input `indices` of permute_topK op is on the device: CPU!")
      expert_for_rows = expert_for_rows.cuda()

    # Shape check
    if inp.size(0) != indices.size(0):
      raise RuntimeError(f"[Error] permute_topK op input `indices` shape mismatch! "
                         f"Expect {inp.size(0)}, but got {indices.size(0)}.")

    # Data type check
    if indices.dtype != torch.int32:
      warnings.warn(f"[Warning] The data type of the input `indices` of permute_topK op is {indices.dtype}! "
            "The recommended type is torch.int32.")
      indices = indices.to(torch.int32)

    # Contiguous check
    if not inp.is_contiguous():
      warnings.warn("[Warning] The input `inp` of permute_topK op is discontiguous!")
      inp = inp.contiguous()
    if not indices.is_contiguous():
      warnings.warn("[Warning] The input `indices` of permute_topK op is discontiguous!")
      indices = indices.contiguous()

    num_topK = indices.size(1)

    input_max_expanded_token_num = max(max_token_num, inp.size(0)) * num_topK
    if _Permute.max_expanded_token_num < input_max_expanded_token_num:
      _Permute.max_expanded_token_num = input_max_expanded_token_num
      _Permute.workspace = []

    if _Permute.dtype != inp.dtype:
      _Permute.dtype = inp.dtype
      _Permute.workspace = []

    permuted_act, row_id_map, _Permute.workspace = tex.moe_permute(
      inp,
      indices,
      num_out_tokens,
      _Permute.workspace,
      _Permute.max_expanded_token_num)

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

    unpermuted_act_grad = tex.moe_unpermute_fwd(
      permuted_act_grad,
      row_id_map,
      torch.empty(0),
      num_tokens,
      num_topK)
    return unpermuted_act_grad, None, None, None


class _Unpermute(torch.autograd.Function):

  @staticmethod
  def forward(ctx,
              inp: torch.Tensor,
              row_id_map: torch.Tensor,
              probs: torch.Tensor):
    # Empty input check
    if not inp.numel():
      ctx.probs = probs
      return inp

    # None probs check
    if probs.numel():
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
    if inp.is_cpu:
      raise RuntimeError("[Error] The input `inp` of unpermute_topK op is on the device: CPU!")
    if row_id_map.is_cpu:
      warnings.warn("[Warning] The input `row_id_map` of unpermute_topK op is on the device: CPU!")
      row_id_map = row_id_map.cuda()

    # Data type check
    if row_id_map.dtype != torch.int32:
      warnings.warn(f"[Warning] The data type of the input `row_id_map` of unpermute_topK op is {row_id_map.dtype}! "
            "The recommended type is torch.int32.")
      row_id_map = row_id_map.to(torch.int32)

    # Contiguous check
    if not inp.is_contiguous():
      warnings.warn("[Warning] The input `inp` of unpermute_topK op is discontiguous!")
      inp = inp.contiguous()
    if not row_id_map.is_contiguous():
      warnings.warn("[Warning] The input `row_id_map` of unpermute_topK op is discontiguous!")
      row_id_map = row_id_map.contiguous()

    num_topK = probs.size(1) if probs.numel() else 1
    num_tokens = probs.size(0) if probs.numel() else row_id_map.size(0)

    unpermuted_output = tex.moe_unpermute_fwd(
      inp,
      row_id_map,
      probs,
      num_tokens,
      num_topK)

    ctx.save_for_backward(inp, row_id_map, probs)
    return unpermuted_output

  @staticmethod
  def backward(ctx, unpermuted_act_grad):
    # Empty input check
    if not unpermuted_act_grad.numel():
      return unpermuted_act_grad, None, ctx.probs

    if not unpermuted_act_grad.is_contiguous():
      unpermuted_act_grad = unpermuted_act_grad.contiguous()

    inp, row_id_map, probs = ctx.saved_tensors

    act_grad = None
    if ctx.needs_input_grad[0]:
      act_grad, prob_grad = tex.moe_unpermute_bwd(
        unpermuted_act_grad,
        inp,
        row_id_map,
        probs)
    
    if not ctx.needs_input_grad[2]:
      prob_grad = None
    return act_grad, None, prob_grad


def Permute(inp, indices, num_out_tokens=-1, max_token_num=-1):
  return _Permute.apply(inp, indices, num_out_tokens, max_token_num)

def Unpermute(inp, row_id_map, probs):
  return _Unpermute.apply(inp, row_id_map, probs)
