# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import triton
import torch.cuda.nvtx as nvtx

from transformer_engine.pytorch import Permute as permute_topK, Unpermute as unpermute_topK


def permute(tokens, indices, num_out_tokens: int = 0):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.
       The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens, topk].
        topk (int, optional): The topk value. Defaults to 1.
        num_out_tokens (int, optional): The effective token count, when enabling the capacity factor, should equal the number of tokens not dropped. By default, set to None, meaning no tokens are dropped.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """

    topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    if num_out_tokens > 0:
        sorted_indices = sorted_indices[:num_out_tokens]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = torch.empty(0),
):
    """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
        sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
        probs (torch.Tensor, optional): The tensor of probabilities corresponding to the permuted tokens. If provided, the unpermuted tokens will be merged with their respective probabilities.

    Returns:
        torch.Tensor: The unpermuted tokens, optionally merged with probabilities.
    """
    num_unpermuted_tokens = probs.numel()
    topk = probs.size(1)

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))

    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


def permute_topK_test(
    dtype,
    num_token,
    num_expert,
    hidden_size,
    num_topK,
    num_out_tokens=None,
    PRINT=False,
    BENCHMARK=False,
):

    if num_out_tokens == None:
        num_out_tokens = num_token * num_topK

    print(
        f"{dtype} token:{num_token} hidden_size:{hidden_size} expert:{num_expert} topK:{num_topK}"
    )

    is_fp8 = dtype in [torch.float8_e5m2, torch.float8_e4m3fn]

    permute_input = torch.rand((num_token, hidden_size), dtype=torch.float32).cuda()
    # for i in range(num_token):
    #   for j in range(hidden_size):
    #     permute_input[i][j] = i * 100 + j
    permute_input = permute_input.to(dtype)
    if is_fp8:
        permute_input = permute_input.half()

    permute_input.requires_grad_(True)

    if num_token > 0:
        indices = torch.stack([torch.randperm(num_expert)[:num_topK] for _ in range(num_token)])
    else:
        indices = torch.empty((num_token, num_topK))
    indices = indices.to(torch.int32).cuda()

    # probs = torch.tensor([[0.1, 0.9],
    #                       [0.2, 0.8],
    #                       [0.3, 0.7]])
    # 0.5
    # probs = torch.ones_like(indices) / 2
    # rand
    probs = torch.rand(num_token, num_topK).cuda()
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = probs / row_sums
    probs.requires_grad_(True)

    if PRINT:
        print(permute_input)
        print(indices)
        print(probs)

    ###################################################################################################################################
    #
    # PyTorch
    #
    ###################################################################################################################################
    nvtx.range_push("PyTorch permute forward")
    permute_output, sorted_indices = permute(permute_input, indices, num_out_tokens)
    nvtx.range_pop()

    permute_bwd_input = torch.rand_like(permute_output)
    # for i in range(num_token * num_topK):
    #   for j in range(hidden_size):
    #     permute_bwd_input[i][j] = i * 100 + j

    nvtx.range_push("PyTorch permute backward")
    permute_output.backward(permute_bwd_input, retain_graph=True)
    nvtx.range_pop()

    unpermute_input = permute_output.detach()
    unpermute_input.requires_grad_(True)

    unpermute_output = unpermute(unpermute_input, sorted_indices, probs=probs)

    if PRINT:
        print("--------------unpermute fwd permute_input--------------")
        print(unpermute_input)
        print("--------------unpermute fwd output--------------")
        print(unpermute_output)

    unpermute_bwd_input = torch.rand_like(unpermute_output)
    # for i in range(num_token):
    #   for j in range(hidden_size):
    #     unpermute_bwd_input[i][j] = i * 2000 + j * 20

    if PRINT:
        print("--------------unpermute bwd permute_input--------------")
        print(unpermute_bwd_input)

    unpermute_output.backward(unpermute_bwd_input, retain_graph=True)
    if PRINT:
        print("--------------unpermute bwd output act grad--------------")
        print(permute_output.grad)
        print("--------------unpermute bwd output probs grad--------------")
        print(probs.grad)

    ###################################################################################################################################
    #
    # Mine
    #
    ###################################################################################################################################
    new_permute_input = permute_input.detach().to(dtype)
    new_permute_bwd_input = permute_bwd_input.detach().to(dtype)
    new_unpermute_bwd_input = unpermute_bwd_input.detach().to(dtype)
    new_permute_input.requires_grad_(True)

    new_permute_output, row_id_map = permute_topK(new_permute_input, indices, num_out_tokens)

    assert torch.allclose(permute_output.float(), new_permute_output.float())

    if PRINT:
        print("--------------row_id_map--------------")
        print(row_id_map)
        print("--------------new_permute_input--------------")
        print(new_permute_input)
        print("--------------new_permute_output--------------")
        print(new_permute_output)

    new_permute_output.backward(new_permute_bwd_input, retain_graph=True)

    if torch.allclose(permute_input.grad.float(), new_permute_input.grad.float()) == False:
        original_inputs = new_permute_input.grad.float().cpu().numpy().flatten()
        original_output = permute_input.grad.float().cpu().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"permute_topK bwd max error (mine vs pytorch): \t\t\t{max_abs_error:.3e} ({dtype})")

        if PRINT:
            print(permute_input.grad)
            print(new_permute_input.grad)

    new_probs = probs.detach()
    new_probs.requires_grad_(True)
    if num_topK == 1:
        new_probs = torch.empty(0)
    new_unpermute_input = new_permute_output.detach()
    new_unpermute_input.requires_grad_(True)

    new_unpermute_output = unpermute_topK(new_unpermute_input, row_id_map, new_probs)

    if torch.allclose(unpermute_output.float(), new_unpermute_output.float()) == False:
        original_inputs = unpermute_output.float().cpu().detach().numpy().flatten()
        original_output = new_unpermute_output.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"unpermute_topK fwd max error (mine vs pytorch): \t\t{max_abs_error:.3e} ({dtype})")

        if PRINT:
            print(unpermute_output)
            print(new_unpermute_output)

    new_unpermute_output.backward(new_unpermute_bwd_input, retain_graph=True)

    if torch.allclose(unpermute_input.grad.float(), new_unpermute_input.grad.float()) == False:
        original_inputs = unpermute_input.grad.float().cpu().detach().numpy().flatten()
        original_output = new_unpermute_input.grad.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(
            "unpermute_topK bwd act_grad max error (mine vs pytorch):"
            f" \t{max_abs_error:.3e} ({dtype})"
        )
        if PRINT:
            print(new_unpermute_input.grad)
            print(unpermute_input.grad)

    if num_topK > 1 and torch.allclose(new_probs.grad, probs.grad) == False:
        original_inputs = new_probs.grad.float().cpu().detach().numpy().flatten()
        original_output = probs.grad.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(
            "unpermute_topK bwd prob_grad max error (mine vs pytorch):"
            f" \t{max_abs_error:.3e} ({dtype})"
        )
        if PRINT:
            print(new_probs.grad)
            print(probs.grad)

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
        print(f"----permute topK----")
        t = perf_test_cuda_kernel(lambda: permute(permute_input, indices, num_out_tokens))
        print(f"pytorch fwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(lambda: permute_topK(new_permute_input, indices, num_out_tokens))
        print(f"new     fwd: {t:.3f} ms")

        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                permute_output,
                permute_bwd_input,
                forward_input=[permute_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"pytorch bwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                new_permute_output,
                new_permute_bwd_input,
                forward_input=[new_permute_input],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"new     bwd: {t:.3f} ms")

        print(f"----unpermute topK----")
        t = perf_test_cuda_kernel(lambda: unpermute(unpermute_input, sorted_indices, probs=probs))
        print(f"pytorch fwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(
            lambda: unpermute_topK(new_unpermute_input, row_id_map, new_probs)
        )
        print(f"new     fwd: {t:.3f} ms")

        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                unpermute_output,
                unpermute_bwd_input,
                forward_input=[unpermute_input, probs],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"pytorch bwd: {t:.3f} ms")
        t = perf_test_cuda_kernel(
            lambda: backward_wrapper(
                new_unpermute_output,
                new_unpermute_bwd_input,
                forward_input=[new_unpermute_input, new_probs],
                retain_graph=True,
                accumulate_grad=False,
            )
        )
        print(f"new     bwd: {t:.3f} ms")


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
        # print(f"Elapsed Time: {elapsed_time_ms / 100} ms")
        return elapsed_time_ms / 100
    else:
        print("CUDA is not available.")


def test_permute_topK():

    torch.manual_seed(1)

    num_token = 4096 * 2
    num_expert = 8
    hidden_size = 4096
    num_topK = 1

    num_out_tokens = num_token * num_topK - 20
    # num_out_tokens = 0

    Benchmark = False
    print("GPU:", torch.cuda.get_device_name(0))

    dtype = torch.float32
    permute_topK_test(
        dtype, num_token, num_expert, hidden_size, num_topK, num_out_tokens, False, Benchmark
    )
    dtype = torch.float16
    permute_topK_test(
        dtype, num_token, num_expert, hidden_size, num_topK, num_out_tokens, False, Benchmark
    )
    dtype = torch.bfloat16
    permute_topK_test(
        dtype, num_token, num_expert, hidden_size, num_topK, num_out_tokens, False, Benchmark
    )
    dtype = torch.float8_e5m2
    permute_topK_test(
        dtype, num_token, num_expert, hidden_size, num_topK, num_out_tokens, False, Benchmark
    )
    dtype = torch.float8_e4m3fn
    permute_topK_test(
        dtype, num_token, num_expert, hidden_size, num_topK, num_out_tokens, False, Benchmark
    )
    dtype = torch.bfloat16
    permute_topK_test(dtype, num_token, 4, hidden_size, 1, None, False, Benchmark)
    permute_topK_test(dtype, num_token, 5, hidden_size, 2, None, False, Benchmark)
    permute_topK_test(dtype, num_token, 6, hidden_size, 3, None, False, Benchmark)
    permute_topK_test(dtype, num_token, 7, hidden_size, 4, None, False, Benchmark)
    permute_topK_test(dtype, num_token, 8, hidden_size, 5, None, False, Benchmark)
    num_token = 0
    permute_topK_test(dtype, num_token, 8, hidden_size, 5, None, False, Benchmark)


if __name__ == "__main__":
    test_permute_topK()
