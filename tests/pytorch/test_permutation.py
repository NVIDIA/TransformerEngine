# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import random

import torch
import pytest
from typing import Dict, List

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    moe_permute as te_permute,
    moe_permute_with_probs as te_permute_with_probs,
    moe_unpermute as te_unpermute,
    moe_sort_chunks_by_index as te_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs as te_sort_chunks_by_index_with_probs,
)
from transformer_engine.pytorch import (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    Float8BlockQuantizer,
    MXFP8Quantizer,
)
import transformer_engine_torch as tex
import copy

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def pytorch_permute_index_map(tokens, indices, num_out_tokens: int = None):
    """
    Permute the tokens based on the indices. Token with the same index will be grouped together.
    The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately.

    Args:
        tokens: torch.Tensor
            The input token tensor.
        indices: torch.Tensor
            The token to expert indices tensor, should have a shape of [num_tokens] or [num_tokens, topk].
        num_out_tokens: int, optional
            The effective output token count, when enabling the capacity factor, should equal the number of tokens not dropped.
            By default, set to None, meaning no tokens are dropped.

    Returns:
        torch.Tensor:
            The permuted tensor.
        torch.Tensor:
            The sorted_indices corresponding permuted tensor.
    """
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    num_out_tokens = num_out_tokens if num_out_tokens is not None else flatten_indices.size(0)

    permuted_tokens = tokens.index_select(0, sorted_indices[:num_out_tokens] // topk)
    return permuted_tokens, sorted_indices


def pytorch_unpermute_index_map(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
):
    """
    Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their
    corresponding probabilities.

    Args:
        permuted_tokens: torch.Tensor
            The tensor of permuted tokens to be unpermuted.
        sorted_indices: torch.Tensor
            The tensor of sorted indices used to unpermute the tokens.
        probs: torch.Tensor, optional
            The tensor of probabilities corresponding to the permuted tokens. If provided, the unpermuted tokens will
            be merged with their respective probabilities.

    Returns:
        torch.Tensor:
            The unpermuted tokens, optionally merged with probabilities.
    """

    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = sorted_indices.size(0)
        topk = 1
    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )

    unpermuted_tokens.index_copy_(0, sorted_indices[: permuted_tokens.size(0)], permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)
    return unpermuted_tokens


def pytorch_permute_mask_map(tokens, routing_map):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
    """
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[1]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool().T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def pytorch_unpermute_mask_map(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    _, hidden = restore_shape

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(
        restore_shape, device=permuted_tokens.device, dtype=permuted_tokens.dtype
    )
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens


def pytorch_sort_chunks_by_index(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
):
    """
    Split and sort the input tensor based on the split_sizes and sorted indices.
    return a tuple of (output, row_id_map). row_id_map is only used when fused=True.
    """
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
    return output


def dtype_tols(te_dtype: tex.DType) -> Dict[str, float]:
    """Estimated tolerances for a datatype

    Based on tolerances for torch.testing.assert_close.

    """
    if te_dtype == tex.DType.kFloat32:
        return dict(rtol=1.0e-6, atol=1.0e-6)
    if te_dtype == tex.DType.kFloat16:
        return dict(rtol=3.0e-3, atol=1.0e-5)
    if te_dtype == tex.DType.kBFloat16:
        return dict(rtol=2.0e-2, atol=1.0e-5)
    if te_dtype == tex.DType.kFloat8E5M2 or te_dtype == tex.DType.kFloat8E4M3:
        return dict(rtol=2.0e-1, atol=1.0e-1)
    raise ValueError(f"Unsuppored dtype ({te_dtype})")


def backward_wrapper(
    act, backward_input, forward_input=[], retain_graph=True, accumulate_grad=False
):
    # Set forward_input.grad to None to avoid grad accumulation.
    if accumulate_grad == False:
        for i in forward_input:
            i.grad = None
    return act.backward(backward_input, retain_graph=retain_graph)


def _test_permutation_index_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    with_probs,
    BENCHMARK=False,
):
    if not with_probs and topK > 1:
        pytest.skip("Only permutations with topK=1 and without probabilities are supported.")

    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    print(
        "index map:"
        f" token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} {te_dtype}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == tex.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == tex.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == tex.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    pytorch_permute_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_permute_bwd_input = torch.rand((num_out_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_unpermute_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    pytorch_permute_fwd_input.requires_grad_(True)

    if num_tokens > 0:
        indices = torch.stack([torch.randperm(num_expert)[:topK] for _ in range(num_tokens)])
    else:
        indices = torch.empty((num_tokens, topK))
    indices = indices.to(torch.int32).cuda()

    probs = None
    if with_probs:
        probs = torch.rand(num_tokens, topK).cuda()
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = probs / row_sums
        probs.requires_grad_(True)

    ###################################################################################################################################
    #
    # PyTorch Permutation
    #
    ###################################################################################################################################
    pytorch_permute_output, sorted_indices = pytorch_permute_index_map(
        pytorch_permute_fwd_input, indices, num_out_tokens
    )
    pytorch_permute_output.backward(pytorch_permute_bwd_input, retain_graph=True)

    pytorch_unpermute_fwd_input = pytorch_permute_output.detach()
    pytorch_unpermute_fwd_input.requires_grad_(True)

    pytorch_unpermute_output = pytorch_unpermute_index_map(
        pytorch_unpermute_fwd_input, sorted_indices, probs=probs
    )
    pytorch_unpermute_output.backward(pytorch_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # TE Permutation
    #
    ###################################################################################################################################
    te_permute_fwd_input = pytorch_permute_fwd_input.detach()
    te_permute_fwd_input.requires_grad_(True)
    te_permute_bwd_input = pytorch_permute_bwd_input.detach()

    te_permute_output, row_id_map = te_permute(
        te_permute_fwd_input, indices, num_out_tokens, map_type="index"
    )
    te_permute_output.backward(te_permute_bwd_input, retain_graph=True)

    te_probs = None
    if with_probs:
        te_probs = probs.detach()
        te_probs.requires_grad_(True)
    te_unpermute_fwd_input = te_permute_output.detach()
    te_unpermute_fwd_input.requires_grad_(True)
    te_unpermute_bwd_input = pytorch_unpermute_bwd_input.detach()

    te_unpermute_output = te_unpermute(
        te_unpermute_fwd_input, row_id_map, te_probs, map_type="index"
    )
    te_unpermute_output.backward(te_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Results Check
    #
    ###################################################################################################################################
    tols = dtype_tols(te_dtype)

    te_permute_output_ = te_permute_output.float()
    te_permute_fwd_input_grad = te_permute_fwd_input.grad.float()
    te_unpermute_output_ = te_unpermute_output.float()
    te_unpermute_fwd_input_grad = te_unpermute_fwd_input.grad.float()

    if not BENCHMARK:
        torch.testing.assert_close(
            pytorch_permute_output.float(),
            te_permute_output_,
            msg=f"Mismatch in te_permute fwd",
        )
        torch.testing.assert_close(
            pytorch_permute_fwd_input.grad.float(),
            te_permute_fwd_input_grad,
            msg=f"Mismatch in te_permute bwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_unpermute_output.float(),
            te_unpermute_output_,
            msg=f"Mismatch in te_unpermute fwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_unpermute_fwd_input.grad.float(),
            te_unpermute_fwd_input_grad,
            msg=f"Mismatch in te_unpermute bwd",
            **tols,
        )
        if with_probs:
            torch.testing.assert_close(
                probs.grad.float(),
                te_probs.grad.float(),
                msg=f"Mismatch in te_unpermute bwd",
                **tols,
            )

    if not pytorch_permute_fwd_input.numel():
        print("Empty pytorch_permute_fwd_input activation test passed.")
        return

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:
        t1 = perf_test_cuda_kernel(
            lambda: pytorch_permute_index_map(pytorch_permute_fwd_input, indices, num_out_tokens)
        )
        t2 = perf_test_cuda_kernel(
            lambda: te_permute(te_permute_fwd_input, indices, num_out_tokens, map_type="index")
        )
        print(f"permute\t\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                pytorch_permute_output,
                pytorch_permute_bwd_input,
                forward_input=[pytorch_permute_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_permute_output,
                te_permute_bwd_input,
                forward_input=[te_permute_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"permute\t\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: pytorch_unpermute_index_map(
                pytorch_unpermute_fwd_input, sorted_indices, probs=probs
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: te_unpermute(te_unpermute_fwd_input, row_id_map, te_probs, map_type="index")
        )
        print(f"unpermute\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                pytorch_unpermute_output,
                pytorch_unpermute_bwd_input,
                forward_input=(
                    [pytorch_unpermute_fwd_input, probs]
                    if with_probs
                    else [pytorch_unpermute_fwd_input]
                ),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_unpermute_output,
                te_unpermute_bwd_input,
                forward_input=(
                    [te_unpermute_fwd_input, te_probs] if with_probs else [te_unpermute_fwd_input]
                ),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"unpermute\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")


def _test_permutation_mask_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    with_probs,
    BENCHMARK=False,
):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    print(
        "mask map:"
        f" token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} {te_dtype}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == tex.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == tex.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == tex.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    pytorch_permute_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_permute_bwd_input = torch.rand((num_out_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_unpermute_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    pytorch_permute_fwd_input.requires_grad_(True)

    restore_shape = pytorch_permute_fwd_input.shape

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    probs = None
    if with_probs:
        probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = probs / row_sums
        probs = probs.to(dtype)
        probs.requires_grad_(True)

    ###################################################################################################################################
    #
    # PyTorch Permutation
    #
    ###################################################################################################################################
    pytorch_permute_output, sorted_indices = pytorch_permute_mask_map(
        pytorch_permute_fwd_input, routing_map
    )
    pytorch_permute_output.backward(pytorch_permute_bwd_input, retain_graph=True)

    pytorch_unpermute_fwd_input = pytorch_permute_output.detach()
    pytorch_unpermute_fwd_input.requires_grad_(True)

    pytorch_unpermute_output = pytorch_unpermute_mask_map(
        pytorch_unpermute_fwd_input, sorted_indices, restore_shape, probs, routing_map
    )
    pytorch_unpermute_output.backward(pytorch_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # TE Permutation
    #
    ###################################################################################################################################
    te_permute_fwd_input = pytorch_permute_fwd_input.detach()
    te_permute_fwd_input.requires_grad_(True)
    te_permute_bwd_input = pytorch_permute_bwd_input.detach()

    te_permute_output, row_id_map = te_permute(
        te_permute_fwd_input, routing_map, num_out_tokens=num_out_tokens, map_type="mask"
    )
    te_permute_output.backward(te_permute_bwd_input, retain_graph=True)

    te_probs = None
    if with_probs:
        te_probs = probs.detach()
        te_probs.requires_grad_(True)
    te_unpermute_fwd_input = te_permute_output.detach()
    te_unpermute_fwd_input.requires_grad_(True)
    te_unpermute_bwd_input = pytorch_unpermute_bwd_input.detach()

    te_unpermute_output = te_unpermute(
        te_unpermute_fwd_input, row_id_map, te_probs, restore_shape, map_type="mask"
    )
    te_unpermute_output.backward(te_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Results Check
    #
    ###################################################################################################################################
    tols = dtype_tols(te_dtype)

    te_permute_output_ = te_permute_output.float()
    te_permute_fwd_input_grad = te_permute_fwd_input.grad.float()
    te_unpermute_output_ = te_unpermute_output.float()
    te_unpermute_fwd_input_grad = te_unpermute_fwd_input.grad.float()

    if not BENCHMARK:
        torch.testing.assert_close(
            pytorch_permute_output.float(),
            te_permute_output_,
            msg=f"Mismatch in te_permute fwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_permute_fwd_input.grad.float(),
            te_permute_fwd_input_grad,
            msg=f"Mismatch in te_permute bwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_unpermute_output.float(),
            te_unpermute_output_,
            msg=f"Mismatch in te_unpermute fwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_unpermute_fwd_input.grad.float(),
            te_unpermute_fwd_input_grad,
            msg=f"Mismatch in te_unpermute bwd",
            **tols,
        )
        if with_probs:
            torch.testing.assert_close(
                probs.grad.float(),
                te_probs.grad.float(),
                msg=f"Mismatch in te_unpermute bwd",
                **tols,
            )

    if not pytorch_permute_fwd_input.numel():
        print("Empty pytorch_permute_fwd_input activation test passed.")
        return

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:
        t1 = perf_test_cuda_kernel(
            lambda: pytorch_permute_mask_map(pytorch_permute_fwd_input, routing_map)
        )
        t2 = perf_test_cuda_kernel(
            lambda: te_permute(
                te_permute_fwd_input, routing_map, num_out_tokens=num_out_tokens, map_type="mask"
            )
        )
        print(f"permute\t\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                pytorch_permute_output,
                pytorch_permute_bwd_input,
                forward_input=[pytorch_permute_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_permute_output,
                te_permute_bwd_input,
                forward_input=[te_permute_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"permute\t\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: pytorch_unpermute_mask_map(
                pytorch_unpermute_fwd_input, sorted_indices, restore_shape, probs, routing_map
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: te_unpermute(
                te_unpermute_fwd_input, row_id_map, te_probs, restore_shape, map_type="mask"
            )
        )
        print(f"unpermute\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                pytorch_unpermute_output,
                pytorch_unpermute_bwd_input,
                forward_input=(
                    [pytorch_unpermute_fwd_input, probs]
                    if with_probs
                    else [pytorch_unpermute_fwd_input]
                ),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_unpermute_output,
                te_unpermute_bwd_input,
                forward_input=(
                    [te_unpermute_fwd_input, te_probs] if with_probs else [te_unpermute_fwd_input]
                ),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"unpermute\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")


def _test_permutation_mask_map_fp8(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    recipe,
):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    if recipe.delayed():
        quantizer = Float8Quantizer(
            scale=torch.full([1], 1.0).cuda().squeeze(),
            amax=torch.full([1], 1.0).cuda(),
            fp8_dtype=te_dtype,
        )
    elif recipe.float8_current_scaling():
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=te_dtype,
            device=torch.device("cuda"),
            columnwise=False,
        )
    elif recipe.float8_block_scaling():
        quantizer = Float8BlockQuantizer(
            fp8_dtype=te_dtype,
            rowwise=True,
            columnwise=False,
            amax_epsilon=0.0,
            force_pow_2_scales=True,  # Fp8 sub-channel a2a requires e8 scales
            block_scaling_dim=1,  # 1x128 scaling
        )
    elif recipe.mxfp8():
        quantizer = MXFP8Quantizer(
            fp8_dtype=te_dtype,
            rowwise=True,
            columnwise=False,
        )
    else:
        raise ValueError("Unsupported FP8 recipe")

    permute_fwd_input = torch.rand(
        size=(num_tokens, hidden_size), dtype=torch.float32, device="cuda"
    )
    # Make an empty fp8 tensor
    permute_fwd_input_fp8 = quantizer.make_empty(
        permute_fwd_input.shape,
        dtype=permute_fwd_input.dtype,
        device=permute_fwd_input.device,
    )
    # quantize the tensor
    quantizer.update_quantized(permute_fwd_input, permute_fwd_input_fp8)
    if recipe.float8_block_scaling():
        pytorch_permute_fwd_input = copy.deepcopy(permute_fwd_input_fp8._rowwise_data)
        pytorch_permute_fwd_scale_input = copy.deepcopy(
            permute_fwd_input_fp8._rowwise_scale_inv.T.contiguous()
        )
    elif recipe.mxfp8():
        pytorch_permute_fwd_input = copy.deepcopy(permute_fwd_input_fp8._rowwise_data)
        pytorch_permute_fwd_scale_input = copy.deepcopy(
            permute_fwd_input_fp8._rowwise_scale_inv.contiguous()
        )
    else:
        pytorch_permute_fwd_input = copy.deepcopy(permute_fwd_input_fp8._data)
        pytorch_permute_fwd_scale_input = None

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    # PyTorch Permutaion
    pytorch_permute_output, _ = pytorch_permute_mask_map(pytorch_permute_fwd_input, routing_map)
    if pytorch_permute_fwd_scale_input is not None:
        pytorch_permute_scale_output, _ = pytorch_permute_mask_map(
            pytorch_permute_fwd_scale_input, routing_map
        )

    # TE Permutation
    permute_output, _ = te_permute(
        permute_fwd_input_fp8, routing_map, num_out_tokens=num_out_tokens, map_type="mask"
    )
    if recipe.float8_block_scaling():
        te_permute_output = permute_output._rowwise_data
        te_permute_scale_output = permute_output._rowwise_scale_inv.T.contiguous()
    elif recipe.mxfp8():
        te_permute_output = permute_output._rowwise_data
        te_permute_scale_output = permute_output._rowwise_scale_inv.contiguous()
    else:
        te_permute_output = permute_output._data
        te_permute_scale_output = None

    # check the permute output
    torch.testing.assert_close(
        pytorch_permute_output,
        te_permute_output,
        atol=0,
        rtol=0,
    )
    if recipe.float8_block_scaling() or recipe.mxfp8():
        torch.testing.assert_close(
            pytorch_permute_scale_output,
            te_permute_scale_output,
            atol=0,
            rtol=0,
        )


def _test_moe_chunk_sort(
    te_dtype,
    num_tokens,
    num_expert,
    tp_size,
    hidden_size,
    BENCHMARK=False,
):
    print(
        "chunk permute:"
        f" token:{num_tokens} hidden_size:{hidden_size} num_expert:{num_expert} tp_size:{tp_size} {te_dtype}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == tex.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == tex.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == tex.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    pytorch_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    pytorch_fwd_input.requires_grad_(True)

    _split_sizes = [0] * (num_expert * tp_size)
    for _ in range(num_tokens):
        idx = random.randint(0, num_expert * tp_size - 1)
        _split_sizes[idx] += 1
    split_sizes = torch.tensor(_split_sizes, dtype=torch.int32).ravel()
    split_sizes_cuda = split_sizes.to(device="cuda")

    _sorted_idxs = torch.arange(num_expert * tp_size, dtype=torch.int32)
    sorted_idxs = _sorted_idxs.reshape(tp_size, num_expert).T.ravel()
    sorted_idxs_cuda = sorted_idxs.to(device="cuda")

    ###################################################################################################################################
    #
    # PyTorch Permutation
    #
    ###################################################################################################################################
    pytorch_output = pytorch_sort_chunks_by_index(pytorch_fwd_input, split_sizes, sorted_idxs)
    pytorch_output.backward(pytorch_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # TE Permutation
    #
    ###################################################################################################################################
    te_fwd_input = pytorch_fwd_input.detach()
    te_fwd_input.requires_grad_(True)
    te_bwd_input = pytorch_bwd_input.detach()

    te_output = te_sort_chunks_by_index(te_fwd_input, split_sizes_cuda, sorted_idxs_cuda)
    te_output.backward(te_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Results Check
    #
    ###################################################################################################################################
    tols = dtype_tols(te_dtype)

    te_output_ = te_output.float()
    te_fwd_input_grad = te_fwd_input.grad.float()

    if not BENCHMARK:
        torch.testing.assert_close(
            pytorch_output.float(),
            te_output_,
            msg=f"Mismatch in te_permute fwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_fwd_input.grad.float(),
            te_fwd_input_grad,
            msg=f"Mismatch in te_permute bwd",
            **tols,
        )

    if not pytorch_fwd_input.numel():
        print("Empty pytorch_fwd_input activation test passed.")
        return

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:
        t1 = perf_test_cuda_kernel(
            lambda: pytorch_sort_chunks_by_index(pytorch_fwd_input, split_sizes, sorted_idxs)
        )
        t2 = perf_test_cuda_kernel(
            lambda: te_sort_chunks_by_index(te_fwd_input, split_sizes_cuda, sorted_idxs_cuda)
        )
        print(f"chunk sort\t\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                pytorch_output,
                pytorch_bwd_input,
                forward_input=[pytorch_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_output,
                te_bwd_input,
                forward_input=[te_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"chunk sort\t\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")


def _test_permutation_mask_map_alongside_probs(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    tp_size,
    BENCHMARK=False,
):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    print(
        "mask map alongside probs:"
        f" token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} {te_dtype}"
    )

    # Convert TE dtypes to PyTorch dtypes
    if te_dtype == tex.DType.kFloat32:
        dtype = torch.float32
    elif te_dtype == tex.DType.kFloat16:
        dtype = torch.float16
    elif te_dtype == tex.DType.kBFloat16:
        dtype = torch.bfloat16
    else:
        pytest.skip("Invalid dtype.")

    pytorch_permute_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_unpermute_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    pytorch_permute_fwd_input.requires_grad_(True)

    restore_shape = pytorch_permute_fwd_input.shape

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = probs / row_sums
    probs = probs.to(dtype)
    probs.requires_grad_(True)

    split_sizes = [0] * (num_expert * tp_size)
    for i in range(num_out_tokens):
        idx = random.randint(0, num_expert * tp_size - 1)
        split_sizes[idx] += 1
    split_sizes = torch.tensor(split_sizes, dtype=torch.int32)
    split_sizes_cuda = split_sizes.to(device="cuda")

    _sorted_idxs = torch.arange(num_expert * tp_size, dtype=torch.int32)
    sorted_idxs = _sorted_idxs.reshape(tp_size, num_expert).T.ravel()
    sorted_idxs_cuda = sorted_idxs.to(device="cuda")

    split_sizes_2 = [split_sizes[i] for i in sorted_idxs.tolist()]
    split_sizes_2 = torch.tensor(split_sizes_2, dtype=torch.int32)
    split_sizes_2_cuda = split_sizes_2.to(device="cuda")

    sorted_idxs_2 = [0] * (num_expert * tp_size)
    for i in range(num_expert * tp_size):
        sorted_idxs_2[sorted_idxs[i]] = i
    sorted_idxs_2 = torch.tensor(sorted_idxs_2, dtype=torch.int32)
    sorted_idxs_2_cuda = sorted_idxs_2.to(device="cuda")

    ###################################################################################################################################
    #
    # PyTorch Permutation
    #
    ###################################################################################################################################
    pytorch_permute_output, sorted_indices = pytorch_permute_mask_map(
        pytorch_permute_fwd_input, routing_map
    )

    pytorch_permute_output = pytorch_sort_chunks_by_index(
        pytorch_permute_output, split_sizes, sorted_idxs
    )

    pytorch_permute_output = pytorch_sort_chunks_by_index(
        pytorch_permute_output, split_sizes_2, sorted_idxs_2
    )

    pytorch_unpermute_output = pytorch_unpermute_mask_map(
        pytorch_permute_output, sorted_indices, restore_shape, probs, routing_map
    )
    pytorch_unpermute_output.backward(pytorch_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # TE Permutation
    #
    ###################################################################################################################################
    te_permute_fwd_input = pytorch_permute_fwd_input.detach()
    te_permute_fwd_input.requires_grad_(True)

    te_unpermute_bwd_input = pytorch_unpermute_bwd_input.detach()
    te_probs = probs.detach()
    te_probs.requires_grad_(True)

    te_permute_output, te_permuted_probs, row_id_map = te_permute_with_probs(
        te_permute_fwd_input,
        te_probs,
        routing_map,
        num_out_tokens=num_out_tokens,
    )

    te_permute_output, te_permuted_probs = te_sort_chunks_by_index_with_probs(
        te_permute_output, te_permuted_probs, split_sizes_cuda, sorted_idxs_cuda
    )

    te_permute_output_dtype = te_permute_output.dtype
    te_permute_output = te_permute_output * te_permuted_probs.unsqueeze(-1)
    te_permute_output = te_permute_output.to(dtype=te_permute_output_dtype)

    te_permute_output = te_sort_chunks_by_index(
        te_permute_output, split_sizes_2_cuda, sorted_idxs_2_cuda
    )

    te_unpermute_output = te_unpermute(
        te_permute_output,
        row_id_map,
        restore_shape=restore_shape,
        map_type="mask",
    )
    te_unpermute_output.backward(te_unpermute_bwd_input, retain_graph=True)

    ###############################################################################################

    tols = dtype_tols(te_dtype)

    te_permute_fwd_input_grad = te_permute_fwd_input.grad.float()
    te_unpermute_output_ = te_unpermute_output.float()

    if not BENCHMARK:
        torch.testing.assert_close(
            pytorch_unpermute_output.float(),
            te_unpermute_output_,
            msg=f"Mismatch in fused_unpermute fwd",
            **tols,
        )
        torch.testing.assert_close(
            pytorch_permute_fwd_input.grad.float(),
            te_permute_fwd_input_grad,
            msg=f"Mismatch in fused_permute bwd",
            **tols,
        )
        torch.testing.assert_close(
            probs.grad.float(), te_probs.grad.float(), msg=f"Mismatch in prob grad", **tols
        )

    if BENCHMARK:
        t1 = perf_test_cuda_kernel(
            lambda: te_permute_with_probs(
                te_permute_fwd_input, te_probs, routing_map, num_out_tokens=num_out_tokens
            )
        )
        print(f"permute\t\tfwd: TE: {t1:.3f} ms")

        te_permute_output, te_permuted_probs, row_id_map = te_permute_with_probs(
            te_permute_fwd_input,
            te_probs,
            routing_map,
            num_out_tokens=num_out_tokens,
        )
        te_permute_bwd_input = torch.rand((num_out_tokens, hidden_size), dtype=dtype).cuda()
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_permute_output,
                te_permute_bwd_input,
                forward_input=[te_permute_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"permute\t\tbwd: TE: {t2:.3f} ms")

        chunk_sort_fwd_input = te_permute_output.detach()
        chunk_sort_fwd_input.requires_grad_(True)
        chunk_sort_fwd_probs = te_permuted_probs.detach()
        chunk_sort_fwd_probs.requires_grad_(True)
        t1 = perf_test_cuda_kernel(
            lambda: te_sort_chunks_by_index_with_probs(
                chunk_sort_fwd_input, chunk_sort_fwd_probs, split_sizes_cuda, sorted_idxs_cuda
            )
        )
        print(f"chunk sort\t\tfwd: TE: {t1:.3f} ms")

        chunk_sort_output, _ = te_sort_chunks_by_index_with_probs(
            chunk_sort_fwd_input, chunk_sort_fwd_probs, split_sizes_cuda, sorted_idxs_cuda
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                chunk_sort_output,
                te_permute_bwd_input,
                forward_input=[chunk_sort_fwd_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"chunk sort\t\tbwd: TE: {t2:.3f} ms")


def perf_test_cuda_kernel(cuda_kernel_fn):
    if torch.cuda.is_available():
        # create CUDA event
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # warmup
        for _ in range(50):
            cuda_kernel_fn()

        start_event.record()
        for _ in range(100):
            cuda_kernel_fn()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        return elapsed_time_ms / 100
    else:
        pytest.skip("CUDA is not available.")


# TE tensor dtypes
_te_dtypes: List[tex.DType] = [tex.DType.kFloat32, tex.DType.kFloat16]
if te.is_bf16_available():
    _te_dtypes.append(tex.DType.kBFloat16)


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("topK", [1, 2, 5])
@pytest.mark.parametrize("num_out_tokens", [None, 2039])
def test_permutation_index_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
):
    with_probs = True
    BENCHMARK = False

    _test_permutation_index_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("topK", [1, 2, 5])
@pytest.mark.parametrize("num_out_tokens", [None, 2039])
def test_permutation_mask_map(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
):
    with_probs = True
    BENCHMARK = False

    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
def test_permutation_mask_map_empty_input(te_dtype):
    with_probs = True
    BENCHMARK = False

    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=0,
        num_expert=8,
        hidden_size=4096,
        topK=2,
        num_out_tokens=0,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("topK", [1, 2, 5])
@pytest.mark.parametrize("num_out_tokens", [None, 2039])
@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_permutation_mask_map_alongside_probs(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    tp_size,
):
    _test_permutation_mask_map_alongside_probs(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        tp_size=tp_size,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
def test_permutation_mask_map_alongside_probs_empty_input(te_dtype):
    _test_permutation_mask_map_alongside_probs(
        te_dtype=te_dtype,
        num_tokens=0,
        num_expert=8,
        hidden_size=4096,
        topK=2,
        num_out_tokens=0,
        tp_size=2,
    )


# Only run FP8 tests on H100.
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
fp8_recipes = [
    recipe.MXFP8BlockScaling(),
    recipe.DelayedScaling(),
    recipe.Float8CurrentScaling(),
    recipe.Float8BlockScaling(),
]


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("te_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize("num_tokens", [2048])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("topK", [1, 2, 5])
@pytest.mark.parametrize("num_out_tokens", [None, 2039])
@pytest.mark.parametrize("recipe", fp8_recipes)
def test_permutation_mask_map_fp8(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    recipe,
):
    if recipe.mxfp8() and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    if recipe.float8_block_scaling() and not fp8_block_scaling_available:
        pytest.skip(reason_for_no_fp8_block_scaling)

    _test_permutation_mask_map_fp8(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        recipe=recipe,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
def test_permutation_index_map_topk1_no_probs(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
):
    topK = 1
    num_out_tokens = None
    with_probs = False
    BENCHMARK = False

    _test_permutation_index_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
def test_permutation_mask_map_topk1_no_probs(
    te_dtype,
    num_tokens,
    num_expert,
    hidden_size,
):
    topK = 1
    num_out_tokens = None
    with_probs = False
    BENCHMARK = False

    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=BENCHMARK,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize("hidden_size", [4096])
def test_chunk_permutation(
    te_dtype,
    num_tokens,
    num_expert,
    tp_size,
    hidden_size,
):
    BENCHMARK = False

    _test_moe_chunk_sort(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        tp_size=tp_size,
        hidden_size=hidden_size,
        BENCHMARK=BENCHMARK,
    )


@pytest.mark.parametrize("te_dtype", _te_dtypes)
def test_chunk_permutation_empty_input(te_dtype):
    BENCHMARK = False

    _test_moe_chunk_sort(
        te_dtype=te_dtype,
        num_tokens=0,
        num_expert=8,
        tp_size=2,
        hidden_size=4096,
        BENCHMARK=BENCHMARK,
    )


def test_permutation_single_case():
    print("GPU:", torch.cuda.get_device_name(0))

    # te_dtype = tex.DType.kFloat32
    # te_dtype = tex.DType.kFloat16
    te_dtype = tex.DType.kBFloat16

    num_tokens = 12
    num_expert = 4
    hidden_size = 16
    topK = 2
    num_out_tokens = num_tokens * topK - 1
    with_probs = True
    Benchmark = True

    _test_permutation_index_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=Benchmark,
    )

    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
        BENCHMARK=Benchmark,
    )

    _test_moe_chunk_sort(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        tp_size=4,
        hidden_size=hidden_size,
        BENCHMARK=Benchmark,
    )

    _test_permutation_mask_map_alongside_probs(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        tp_size=4,
    )


def benchmark_single_case(
    te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, ep_size, tp_size
):
    torch.cuda.nvtx.range_push(
        f"{num_tokens}-{num_expert}-{hidden_size}-{topK}-{ep_size}-{tp_size}"
    )

    torch.cuda.nvtx.range_push("permutation_index_map_with_probs")
    _test_permutation_index_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=True,
        BENCHMARK=True,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("permutation_mask_map_with_probs")
    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=True,
        BENCHMARK=True,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("permutation_mask_map_without_probs")
    _test_permutation_mask_map(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=False,
        BENCHMARK=True,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("permutation_mask_map_alongside_probs")
    _test_permutation_mask_map_alongside_probs(
        te_dtype=te_dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        tp_size=tp_size,
        BENCHMARK=True,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()


def benchmark_multiple_cases():
    print("GPU:", torch.cuda.get_device_name(0))

    # te_dtype = tex.DType.kFloat32
    # te_dtype = tex.DType.kFloat16
    te_dtype = tex.DType.kBFloat16

    ep_size = 64
    tp_size = 2
    num_tokens = 4096
    num_expert = 256
    hidden_size = 7168
    topK = 8
    num_out_tokens = num_tokens * topK
    benchmark_single_case(
        te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, ep_size, tp_size
    )

    ep_size = 8
    tp_size = 1
    num_tokens = 8192 * 2
    num_expert = 128
    hidden_size = 4096
    topK = 6
    num_out_tokens = num_tokens * topK
    benchmark_single_case(
        te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, ep_size, tp_size
    )

    ep_size = 64
    tp_size = 2
    num_tokens = 16384
    num_expert = 4
    hidden_size = 7168
    topK = 1
    num_out_tokens = num_tokens * topK
    benchmark_single_case(
        te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, ep_size, tp_size
    )


if __name__ == "__main__":
    benchmark_multiple_cases()
