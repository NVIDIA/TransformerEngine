# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import torch
import pytest
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
import triton
import triton.language as tl


def test_cross_entropy_fwd_sum_exp_torch(vocab_parallel_logits, max_logit):

    vocab_parallel_logits = vocab_parallel_logits - max_logit.unsqueeze(dim=-1)
    exp_logits = torch.exp(vocab_parallel_logits)
    ret = torch.sum(exp_logits, dim=-1)
    return ret


def test_cross_entropy_fwd_mean_log_torch(vocab_parallel_logits, max_logit, sum_exp_logits):

    vocab_parallel_logits = vocab_parallel_logits - max_logit.unsqueeze(dim=-1)
    exp_logits = torch.exp(vocab_parallel_logits)
    exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
    log_probs = torch.log(exp_logits)
    mean_log_probs = log_probs.mean(dim=-1)
    return mean_log_probs



@triton.jit
def cross_entropy_bwd_kernel(grad_input_ptr,
                             grad_output_ptr,
                             input_ptr,
                             target_mask_ptr,
                             masked_target_1d_ptr,
                             logits_max_ptr,
                             sum_exp_logits_ptr,
                             n_cols,
                             label_smoothing,
                             vocab_size,
                             BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    grad_input_ptr += row_idx * n_cols
    input_ptr += row_idx * n_cols

    grad_output = tl.load(grad_output_ptr + row_idx)
    target_mask = tl.load(target_mask_ptr + row_idx)
    masked_target_1d = tl.load(masked_target_1d_ptr + row_idx)
    logits_max = tl.load(logits_max_ptr + row_idx)
    sum_exp_logits = tl.load(sum_exp_logits_ptr + row_idx)

    for i in range((n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        input_ptrs = input_ptr + col_offsets
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0)
        row = row.to(tl.float32)
        row = row - logits_max
        row = tl.exp(row)
        sum_exp_row = tl.full((BLOCK_SIZE,), sum_exp_logits, tl.float32)
        row = tl.math.div_rn(row, sum_exp_row)

        softmax_update = 1.0 - target_mask.to(tl.float32)
        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            softmax_update *= (1.0 - smoothing)
        row = tl.where(col_offsets == masked_target_1d, row - softmax_update, row)

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            average_grad = 1.0 / vocab_size
            row -= smoothing * average_grad

        row = row * grad_output
        row = row.to(tl.bfloat16)
        grad_input_ptrs = grad_input_ptr + col_offsets
        tl.store(grad_input_ptrs, row, mask=col_offsets < n_cols)


def test_cross_entropy_bwd_triton(grad_output,
                      inputs,
                      target_mask,
                      masked_target_1d,
                      logits_max,
                      sum_exp_logits,
                      label_smoothing,
                      vocab_size):
    grad_input = torch.empty_like(inputs)

    n_cols = inputs.size(-1)
    inputs = inputs.view(-1, n_cols)
    n_rows = inputs.size(0)

    grad_output = grad_output.view(n_rows)
    grad_output = grad_output.contiguous()

    target_mask = target_mask.view(n_rows)

    masked_target_1d = masked_target_1d.view(n_rows)

    logits_max = logits_max.view(n_rows)

    sum_exp_logits = sum_exp_logits.view(n_rows)

    BLOCK_SIZE = 8*1024
    num_warps = 16

    grad_input_ = grad_input.view(n_rows, n_cols)

    cross_entropy_bwd_kernel[(n_rows, )](
        grad_input_,
        grad_output,
        inputs,
        target_mask,
        masked_target_1d,
        logits_max,
        sum_exp_logits,
        n_cols,
        label_smoothing,
        vocab_size,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_input


@pytest.mark.parametrize("s_size", [3, 128, 1024])
@pytest.mark.parametrize("b_size", [1, 32])
@pytest.mark.parametrize("v_size", [256, 1024, 256000])
def test_check_cross_entropy_fwd_sum_exp_cuda(s_size, b_size, v_size):
    # cuda kernel logic
    s, b, v = s_size, b_size, v_size
    vocab_parallel_logits = torch.randn(s, b, v).to(torch.bfloat16).cuda()  # bf16
    vocab_parallel_logits.to(torch.bfloat16)

    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    logits_max = logits_max.to(torch.float32)
    n_dim = vocab_parallel_logits.size(-1)

    sum_exp_logits = tex.cross_entropy_fwd_sum_exp_cuda(vocab_parallel_logits, logits_max)
   

    sum_exp_logits_torch = test_cross_entropy_fwd_sum_exp_torch(vocab_parallel_logits, logits_max)
    assert torch.allclose(sum_exp_logits, sum_exp_logits_torch)



@pytest.mark.parametrize("s_size", [3, 128])
@pytest.mark.parametrize("b_size", [1, 32])
@pytest.mark.parametrize("v_size", [256, 1024, 256000])
def test_check_cross_entropy_fwd_mean_log_cuda(s_size, b_size, v_size):
    # cuda kernel logic
    s, b, v = s_size, b_size, v_size
    vocab_parallel_logits = torch.randn(s, b, v).to(torch.bfloat16).cuda().uniform_(-0.1, 0.1)  # bf16
    vocab_parallel_logits.to(torch.bfloat16)
    vocab_parallel_logits.fill_(0.55)

    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    logits_max = logits_max.to(torch.float32)
    logits_max.fill_(0.73).to(torch.float32)

    sum_exp_logits = torch.empty_like(logits_max)
    sum_exp_logits.fill_(3).to(torch.float32)

    n_dim = vocab_parallel_logits.size(-1)

    mean_log_probs = tex.cross_entropy_fwd_mean_log_cuda(
        vocab_parallel_logits, logits_max, sum_exp_logits
    )
   
    mean_log_probs_torch = test_cross_entropy_fwd_mean_log_torch(
        vocab_parallel_logits, logits_max, sum_exp_logits
    )

    assert (torch.allclose(mean_log_probs, mean_log_probs_torch))

@pytest.mark.parametrize("s_size", [3, 128])
@pytest.mark.parametrize("b_size", [1, 32])
@pytest.mark.parametrize("v_size", [256, 1024, 256000])
def test_check_cross_entropy_bwd_cuda(s_size, b_size, v_size):
    # cuda kernel logic
    s, b, v = s_size, b_size, v_size
    input_ptr = torch.randn(s, b, v).to(torch.bfloat16).cuda()  # bf16
    input_ptr.to(torch.bfloat16)

    logits_max = torch.max(input_ptr, dim=-1)[0]
    logits_max = logits_max.to(torch.float32)
    logits_max.fill_(0.7).to(torch.float32)

    sum_exp_logits = torch.empty_like(logits_max)
    sum_exp_logits.fill_(3).to(torch.float32)

    label_smoothing = 0.12
    vocab_size = 666
    grad_output_ptr = torch.empty_like(logits_max)
    grad_output_ptr.fill_(0.88).to(torch.float32)

    target_mask_ptr = torch.empty_like(logits_max, dtype=torch.bool)
    target_mask_ptr.fill_(0).to(torch.bool)

    masked_target_1d_ptr = torch.empty_like(logits_max, dtype=torch.int64)
    masked_target_1d_ptr.fill_(1).to(torch.int64)
    masked_target_1d_ptr = masked_target_1d_ptr.view(-1)

    n_dim = input_ptr.size(-1)

    grad_input_ptr_cuda = tex.cross_entropy_bwd_cuda(
        grad_output_ptr,
        input_ptr,
        target_mask_ptr,
        masked_target_1d_ptr,
        logits_max,
        sum_exp_logits,
        label_smoothing,
        vocab_size,
    )

    grad_input_ptr_triton = test_cross_entropy_bwd_triton(grad_output_ptr, input_ptr, target_mask_ptr, masked_target_1d_ptr, logits_max, sum_exp_logits, label_smoothing, vocab_size)

    assert torch.allclose(grad_input_ptr_cuda, grad_input_ptr_triton)


if __name__ == "__main__":
    # test_check_cross_entropy_fwd_sum_exp_cuda(3, 1, 256)
    # test_check_cross_entropy_fwd_mean_log_cuda()
    # test_check_cross_entropy_bwd_cuda()
    print("test")