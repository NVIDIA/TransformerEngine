import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex


def test_cross_entropy_fwd_sum_exp_torch(vocab_parallel_logits, max_logit):

    # step2: substraction max_logit
    vocab_parallel_logits = vocab_parallel_logits - max_logit.unsqueeze(dim=-1)
    # step 3: exp
    exp_logits = torch.exp(vocab_parallel_logits)
    # step 4: sum
    ret = torch.sum(exp_logits, dim=-1)
    return ret


def check_cross_entropy_fwd_sum_exp_cuda():
    # cuda kernel logic
    s, b, v = 3, 1, 1024
    vocab_parallel_logits = torch.randn(s, b, v).to(torch.bfloat16).cuda()  # bf16
    arr = [0.4, 0.5, 0.6]
    for i in range(3):
        vocab_parallel_logits[i] = arr[i]
    # vocab_parallel_logits.fill_(0.55)
    vocab_parallel_logits.to(torch.bfloat16)

    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    logits_max = logits_max.to(torch.float32)
    logits_max.fill_(0.45).to(torch.float32)
    n_dim = vocab_parallel_logits.size(-1)

    sum_exp_logits = tex.cross_entropy_fwd_sum_exp_cuda(vocab_parallel_logits, logits_max)
    print(sum_exp_logits.shape)

    print()

    sum_exp_logits_torch = test_cross_entropy_fwd_sum_exp_torch(vocab_parallel_logits, logits_max)
    print(sum_exp_logits_torch.shape)
    # print(torch.allclose(sum_exp_logits, sum_exp_logits_torch))


def test_cross_entropy_fwd_mean_log_torch(vocab_parallel_logits, max_logit, sum_exp_logits):

    vocab_parallel_logits = vocab_parallel_logits - max_logit.unsqueeze(dim=-1)
    exp_logits = torch.exp(vocab_parallel_logits)
    exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

    log_probs = torch.log(exp_logits)
    mean_log_probs = log_probs.mean(dim=-1)

    return mean_log_probs


def check_cross_entropy_fwd_mean_log_cuda():
    # cuda kernel logic
    s, b, v = 1024, 4, 256000
    vocab_parallel_logits = torch.randn(s, b, v).to(torch.bfloat16).cuda()  # bf16
    # arr = [0.023, 0.643, 0.195]
    # for i in range(3):
    #     vocab_parallel_logits[i] = arr[i]
    vocab_parallel_logits.fill_(0.55)
    vocab_parallel_logits.to(torch.bfloat16)

    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    logits_max = logits_max.to(torch.float32)
    logits_max.fill_(0.73).to(torch.float32)

    sum_exp_logits = torch.empty_like(logits_max)
    sum_exp_logits.fill_(3).to(torch.float32)

    n_dim = vocab_parallel_logits.size(-1)

    mean_log_probs = tex.cross_entropy_fwd_mean_log_cuda(
        vocab_parallel_logits, logits_max, sum_exp_logits
    )
    print(mean_log_probs)

    print("-------------------------")

    mean_log_probs_torch = test_cross_entropy_fwd_mean_log_torch(
        vocab_parallel_logits, logits_max, sum_exp_logits
    )
    print(mean_log_probs_torch)
    # print(torch.allclose(mean_log_probs, mean_log_probs_torch))


def check_cross_entropy_bwd_cuda():
    # cuda kernel logic
    s, b, v = 3, 1, 1025
    input_ptr = torch.randn(s, b, v).to(torch.bfloat16).cuda()  # bf16
    arr = [0.090, 0.777, 0.595]
    for i in range(3):
        input_ptr[i] = arr[i]

    # input_ptr.fill_(0.55)
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

    grad_input_ptr = tex.cross_entropy_bwd_cuda(
        grad_output_ptr,
        input_ptr,
        target_mask_ptr,
        masked_target_1d_ptr,
        logits_max,
        sum_exp_logits,
        label_smoothing,
        vocab_size,
    )
    print(grad_input_ptr)


if __name__ == "__main__":
    # check_cross_entropy_fwd_sum_exp_cuda()
    # check_cross_entropy_fwd_mean_log_cuda()
    check_cross_entropy_bwd_cuda()
