# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import pytest
from typing import Dict

from transformer_engine.pytorch import Permute as te_permute, Unpermute as te_unpermute
from transformer_engine.pytorch.utils import is_bf16_compatible
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

# Only run FP8 tests on H100.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def pytorch_permute(tokens, indices, num_out_tokens: int = None):
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


def pytorch_unpermute(
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


def dtype_tols(dtype: torch.dtype) -> Dict[str, float]:
    """Estimated tolerances for a datatype

    Based on tolerances for torch.testing.assert_close.

    """
    if dtype == torch.float32:
        return dict(rtol=1.0e-6, atol=1.0e-6)
    if dtype == torch.float16:
        return dict(rtol=3.0e-3, atol=1.0e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=2.0e-2, atol=1.0e-5)
    if dtype == torch.float8_e5m2 or dtype == torch.float8_e4m3fn:
        return dict(rtol=2.0e-1, atol=1.0e-1)
    raise ValueError(f"Unsuppored dtype ({dtype})")


param_dtypes = [torch.float32, torch.float16]
if is_bf16_compatible():
    param_dtypes.append(torch.bfloat16)
if fp8_available:
    param_dtypes.extend([torch.float8_e5m2, torch.float8_e4m3fn])


@pytest.mark.parametrize("dtype", param_dtypes)
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [8, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("topK", [1, 2, 5])
@pytest.mark.parametrize("num_out_tokens", [None, 4050])
@pytest.mark.parametrize("with_probs", [True, False])
def test_permutation(
    dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    with_probs,
    BENCHMARK=False,
):
    if not with_probs and topK > 1:
        print("Only permutations with topK=1 and without probabilities are supported.")
        return

    if topK > num_expert:
        print("topK should be smaller than the number of experts.")
        return

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    print(f"token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} {dtype}")

    permute_input = torch.rand((num_tokens, hidden_size), dtype=torch.float32).cuda().to(dtype)
    if fp8_available and dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        permute_input = permute_input.half()
    permute_input.requires_grad_(True)

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
    # PyTorch
    #
    ###################################################################################################################################
    permute_output, sorted_indices = pytorch_permute(permute_input, indices, num_out_tokens)
    permute_bwd_input = torch.rand_like(permute_output)
    permute_output.backward(permute_bwd_input, retain_graph=True)

    unpermute_input = permute_output.detach()
    unpermute_input.requires_grad_(True)

    unpermute_output = pytorch_unpermute(unpermute_input, sorted_indices, probs=probs)
    unpermute_bwd_input = torch.rand_like(unpermute_output)
    unpermute_output.backward(unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # TE
    #
    ###################################################################################################################################
    te_permute_input = permute_input.detach().to(dtype)
    te_permute_bwd_input = permute_bwd_input.detach().to(dtype)
    te_unpermute_bwd_input = unpermute_bwd_input.detach().to(dtype)
    te_permute_input.requires_grad_(True)

    te_permute_output, row_id_map = te_permute(te_permute_input, indices, num_out_tokens)
    te_permute_output.backward(te_permute_bwd_input, retain_graph=True)

    te_probs = None
    if with_probs:
        te_probs = probs.detach()
        te_probs.requires_grad_(True)
    te_unpermute_input = te_permute_output.detach()
    te_unpermute_input.requires_grad_(True)

    te_unpermute_output = te_unpermute(te_unpermute_input, row_id_map, te_probs)
    te_unpermute_output.backward(te_unpermute_bwd_input, retain_graph=True)

    tols = dtype_tols(dtype)

    torch.testing.assert_close(
        permute_output.float(), te_permute_output.float(), msg=f"Mismatch in te_permute fwd"
    )
    torch.testing.assert_close(
        permute_input.grad.float(),
        te_permute_input.grad.float(),
        msg=f"Mismatch in te_permute bwd",
        **tols,
    )
    torch.testing.assert_close(
        unpermute_output.float(),
        te_unpermute_output.float(),
        msg=f"Mismatch in te_unpermute fwd",
        **tols,
    )
    torch.testing.assert_close(
        unpermute_input.grad.float(),
        te_unpermute_input.grad.float(),
        msg=f"Mismatch in te_unpermute bwd",
        **tols,
    )
    if with_probs:
        torch.testing.assert_close(
            probs.grad.float(), te_probs.grad.float(), msg=f"Mismatch in te_unpermute bwd", **tols
        )

    if not permute_input.numel():
        print("Empty permute_input activation test passed.")
        return

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    def backward_wrapper(
        act, backward_input, forward_input=[], retain_graph=True, accumulate_grad=False
    ):
        # Set forward_input.grad to None to avoid grad accumulation.
        if accumulate_grad == False:
            for i in forward_input:
                i.grad = None
        return act.backward(backward_input, retain_graph=retain_graph)

    if BENCHMARK:
        t1 = perf_test_cuda_kernel(lambda: pytorch_permute(permute_input, indices, num_out_tokens))
        t2 = perf_test_cuda_kernel(lambda: te_permute(te_permute_input, indices, num_out_tokens))
        print(f"permute\t\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                permute_output,
                permute_bwd_input,
                forward_input=[permute_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_permute_output,
                te_permute_bwd_input,
                forward_input=[te_permute_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"permute\t\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: pytorch_unpermute(unpermute_input, sorted_indices, probs=probs)
        )
        t2 = perf_test_cuda_kernel(lambda: te_unpermute(te_unpermute_input, row_id_map, te_probs))
        print(f"unpermute\tfwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")

        t1 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                unpermute_output,
                unpermute_bwd_input,
                forward_input=[unpermute_input, probs] if with_probs else [unpermute_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        t2 = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                te_unpermute_output,
                te_unpermute_bwd_input,
                forward_input=(
                    [te_unpermute_input, te_probs] if with_probs else [te_unpermute_input]
                ),
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"unpermute\tbwd: pytorch: {t1:.3f} ms,  TE: {t2:.3f} ms")


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
        print("CUDA is not available.")
