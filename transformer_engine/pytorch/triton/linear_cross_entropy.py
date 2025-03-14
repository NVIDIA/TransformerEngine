
"""
Implementations of the linear cross entropy kernel.
"""

import typing
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from transformer_engine.pytorch.triton.linear_cross_entropy_with_token_entropy import (
    EntropyReductionEnum,
    get_entropy_reduction_enum)

_dedicated_stream, _dedicated_events = None, None

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
                            num_stages=3, num_warps=4)],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_kernel_general_mainloop(num_tokens,
                                              hidden_size,
                                              vocab_size,
                                              vocab_per_split,
                                              rank,
                                              hidden_ptr, stride_hidden_m, stride_hidden_k,
                                              weight_ptr, stride_weight_k, stride_weight_n,
                                              labels_ptr, stride_labels,
                                              max_ptr, stride_max_m, stride_max_n,
                                              accu_ptr, stride_accu_m, stride_accu_n,
                                              global_logprobs_ptr, stride_global_logprobs,
                                              global_logprobs_scalar_ptr,
                                              # Meta-parameters
                                              BLOCK_SIZE_M: tl.constexpr,
                                              BLOCK_SIZE_N: tl.constexpr,
                                              BLOCK_SIZE_K: tl.constexpr):
    """
    forward mainloop
    """
    pid = tl.program_id(axis=0)
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    if pid_m == 0 and pid_n == 0:
        tl.store(global_logprobs_scalar_ptr, 0.0)

    # coordinates
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    hidden_ptrs = hidden_ptr + (offs_am[:,None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

    # load labels for this block
    labels = tl.load(labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens)

    # traverse over N dimension
    _max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _logprobs = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for n in range(0, num_pid_n):
        offs_bn = pid_n * vocab_per_split + n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        weight_ptrs = weight_ptr + (offs_k[:, None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)

        # iterate over K dimension
        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            _hidden = tl.load(hidden_ptrs,
                              mask=(offs_k[None,:] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:,None] < num_tokens),
                              other=0.0)
            _weight = tl.load(weight_ptrs,
                              mask=(offs_k[:,None] < hidden_size - k * BLOCK_SIZE_K)
                                    & (offs_bn[None,:] < (min((pid_n + 1) * vocab_per_split, vocab_size))),
                              other=0.0)
            logits = tl.dot(_hidden, _weight, logits)

            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        hidden_ptrs -= hidden_size * stride_hidden_k

        # update global maximum
        _max_old = _max
        m_pid_n = tl.max(logits, axis=1)
        _max = tl.maximum(_max_old, m_pid_n)

        exp_logits = tl.exp(logits - _max[:,None])
        coeff = tl.exp(_max_old - _max)
        _accu = coeff * _accu + tl.sum(exp_logits, axis=1)

        label_mask = (offs_bn + rank * vocab_size)[None,:] == labels[:,None]
        _logprobs += tl.sum(logits * label_mask, axis=1)

    # store maximum
    maximum_ptrs = max_ptr + pid_n * stride_max_n + offs_am * stride_max_m
    tl.store(maximum_ptrs, _max, mask=(offs_am < num_tokens) & (pid_n < num_splits))
    accu_ptrs = accu_ptr + pid_n * stride_accu_n + offs_am * stride_accu_m
    tl.store(accu_ptrs, _accu, mask=(offs_am < num_tokens) & (pid_n < num_splits))

    # store logprobs
    vocab_left_idx = pid_n * vocab_per_split + rank * vocab_size
    vocab_right_idx = min((pid_n + 1) * vocab_per_split, vocab_size) + rank * vocab_size
    mask = (labels >= vocab_left_idx) & (labels < vocab_right_idx)
    mask &= (offs_am < num_tokens)
    global_logprobs_ptrs = global_logprobs_ptr + offs_am * stride_global_logprobs
    tl.store(global_logprobs_ptrs, _logprobs, mask=mask)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})],
    key=["num_tokens", "num_splits"]
)
@triton.jit
def efficient_entropy_triton_kernel_epilogue(num_tokens, num_splits,
                                             max_ptr, stride_max_m, stride_max_n,
                                             accu_ptr, stride_accu_m, stride_accu_n,
                                             global_max_ptr, stride_global_max,
                                             global_accu_ptr, stride_global_accu,
                                             global_logprobs_ptr, stride_global_logprobs,
                                             global_logprobs_scalar_ptr,
                                             reduction: int,
                                             BLOCK_SIZE_M: tl.constexpr,
                                             BLOCK_SIZE_N: tl.constexpr):
    """
    forward epilogue
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        _max = tl.load(max_ptr + offs_m[:,None] * stride_max_m + offs_n[None,:] * stride_max_n,
                       mask=(offs_m[:,None] < num_tokens) & (offs_n[None,:] < num_splits),
                       other=0.0)
        _accu = tl.load(accu_ptr + offs_m[:,None] * stride_accu_m + offs_n[None,:] * stride_accu_n,
                       mask=(offs_m[:,None] < num_tokens) & (offs_n[None,:] < num_splits),
                       other=0.0)

        # local reduction
        _max_old = global_max
        _local_max, _local_indices = tl.max(_max, axis=1, return_indices=True)
        global_max = tl.maximum(global_max, _local_max)

        _scale = tl.exp(_max - global_max[:,None])
        _coeff = tl.exp(_max_old - global_max)
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)

    # store maximum
    tl.store(global_max_ptr + offs_m * stride_global_max,
             global_max, mask=offs_m < num_tokens)

    # store accumulate
    tl.store(global_accu_ptr + offs_m * stride_global_accu,
             global_accu, mask=offs_m < num_tokens)

    # update logprobs
    global_logprobs_ptrs = global_logprobs_ptr + offs_m * stride_global_logprobs
    global_logprobs = tl.load(global_logprobs_ptrs, mask=offs_m < num_tokens)
    global_logprobs = global_max + tl.log(global_accu) - global_logprobs
    if reduction == 0: # no-reduction
        tl.store(global_logprobs_ptrs, global_logprobs, mask=offs_m < num_tokens)
    elif reduction == 1: # sum
        global_logprobs_scalar = tl.sum(global_logprobs, axis=0)
        tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)
    elif reduction == 2: # mean
        global_logprobs_scalar = tl.sum(global_logprobs, axis=0) / num_tokens.to(tl.float32)
        tl.atomic_add(global_logprobs_scalar_ptr, global_logprobs_scalar)

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})],
    key=["num_tokens", "num_splits"]
)
@triton.jit
def efficient_entropy_triton_kernel_epilogue_tp(num_tokens, num_splits,
                                                reduced_max_ptr, stride_reduced_max_m, stride_reduced_max_n,
                                                original_max_ptr, stride_original_max_m, stride_original_max_n,
                                                accu_ptr, stride_accu_m, stride_accu_n,
                                                global_max_ptr, stride_global_max,
                                                global_accu_ptr, stride_global_accu,
                                                BLOCK_SIZE_M: tl.constexpr,
                                                BLOCK_SIZE_N: tl.constexpr):
    """
    tp forward epilogue
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        _reduced_max = tl.load(reduced_max_ptr + offs_m[:,None] * stride_reduced_max_m + offs_n[None,:] * stride_reduced_max_n,
                               mask=(offs_m[:,None] < num_tokens) & (offs_n[None,:] < num_splits),
                               other=0.0)
        _original_max = tl.load(original_max_ptr + offs_m[:,None] * stride_original_max_m + offs_n[None,:] * stride_original_max_n,
                                mask=(offs_m[:,None] < num_tokens) & (offs_n[None,:] < num_splits),
                                other=0.0)
        _accu = tl.load(accu_ptr + offs_m[:,None] * stride_accu_m + offs_n[None,:] * stride_accu_n,
                       mask=(offs_m[:,None] < num_tokens) & (offs_n[None,:] < num_splits),
                       other=0.0)

        # local reduce-max
        _max_old = global_max
        _local_max = tl.max(_reduced_max, axis=1)
        global_max = tl.maximum(global_max, _local_max)

        # update accumulate
        _coeff = tl.exp(_max_old - global_max)
        _scale = tl.exp(_original_max - global_max[:,None])
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)

    # store
    tl.store(global_max_ptr + offs_m * stride_global_max,
             global_max, mask=offs_m < num_tokens)
    tl.store(global_accu_ptr + offs_m * stride_global_accu,
             global_accu, mask=offs_m < num_tokens)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 16})],
    key=["num_tokens"]
)
@triton.jit
def efficient_entropy_triton_epilogue_tp_logprobs(num_tokens,
                                                  logprobs_ptr, stride_logprobs,
                                                  maximum_ptr, stride_maximum,
                                                  accumulate_ptr, stride_accumulate,
                                                  logprobs_scalar_ptr,
                                                  reduction: int,
                                                  BLOCK_SIZE_M: tl.constexpr):
    """
    tp update logprobs
    """
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    logprobs = tl.load(logprobs_ptr + offs_m * stride_logprobs, mask=offs_m < num_tokens)
    maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens)
    accumulate = tl.load(accumulate_ptr + offs_m * stride_accumulate, mask=offs_m < num_tokens)

    logprobs = maximum + tl.log(accumulate) - logprobs

    if reduction == 0: # none
        tl.store(logprobs_ptr + offs_m * stride_logprobs, logprobs, mask=offs_m < num_tokens)
    elif reduction == 1: # sum
        logprobs_scalar = tl.sum(logprobs, axis=0)
        tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)
    elif reduction == 2: # mean
        logprobs_scalar = tl.sum(logprobs, axis=0) / num_tokens.to(tl.float32)
        tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)


def efficient_entropy_forward(hidden: torch.Tensor,
                              weight: torch.Tensor,
                              labels: torch.Tensor,
                              reduction: typing.Optional[int] = 2,
                              dist_process_group: typing.Optional[dist.ProcessGroup] = None
                              ) -> typing.List[torch.Tensor]:
    """
    forward host function
    """
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[0]

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    _world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    if not hasattr(efficient_entropy_forward, "_initialized"):
        global _dedicated_stream, _dedicated_events
        _dedicated_stream = torch.cuda.Stream(hidden.device)
        _dedicated_events = [torch.cuda.Event() for _ in range(2)]
        efficient_entropy_forward._initialized = True

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    hidden_size, vocab_size = weight.shape
    assert hidden_size % 128 == 0
    assert vocab_size % 128 == 0

    REDUCTION = get_entropy_reduction_enum(reduction)

    if REDUCTION == EntropyReductionEnum._None:
        if dist_process_group is None:
            logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
        else:
            logprobs = torch.zeros((num_tokens,), device=hidden.device, dtype=torch.float32)
    elif REDUCTION in (EntropyReductionEnum._Sum, EntropyReductionEnum._Mean):
        logprobs = torch.empty((), device=hidden.device, dtype=torch.float32)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    # buffers need for backward
    maximum = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    accumulate = torch.empty_like(maximum, dtype=torch.float32)
    assert maximum.is_contiguous() and accumulate.is_contiguous()

    # intermediate buffers
    vocab_per_split = 1024
    assert vocab_per_split % 256 == 0
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    _max = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _accu = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)

    if REDUCTION == EntropyReductionEnum._None:
        _logprobs = logprobs
    else:
        if dist_process_group is None:
            _logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
        else:
            _logprobs = torch.zeros((num_tokens,), device=hidden.device, dtype=torch.float32)

    assert _max.is_contiguous() and _accu.is_contiguous()
    assert _max.is_cuda and _accu.is_cuda
    assert _logprobs.is_contiguous() and _logprobs.is_cuda

    def mainloop_grid(meta):
        return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]) * num_splits,)
    efficient_entropy_kernel_general_mainloop[mainloop_grid](
        num_tokens, hidden_size, vocab_size, vocab_per_split,
        _rank,
        hidden, hidden.stride(0), hidden.stride(1),
        weight, weight.stride(0), weight.stride(1),
        labels, labels.stride(0),
        _max, _max.stride(0), _max.stride(1),
        _accu, _accu.stride(0), _accu.stride(1),
        _logprobs, _logprobs.stride(0),
        logprobs
    )

    def epilogue_grid(meta):
        return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]),)
    if dist_process_group is None:
        # single GPU case
        efficient_entropy_triton_kernel_epilogue[epilogue_grid](
            num_tokens, num_splits,
            _max, _max.stride(0), _max.stride(1),
            _accu, _accu.stride(0), _accu.stride(1),
            maximum, maximum.stride(0),
            accumulate, accumulate.stride(0),
            _logprobs, _logprobs.stride(0),
            logprobs,
            REDUCTION,
        )
    else: # tensor-parallel case
        # launch it to another stream, instead of using current stream
        torch.cuda.current_stream().record_event(_dedicated_events[0])
        with torch.cuda.stream(_dedicated_stream):
            _dedicated_stream.wait_event(_dedicated_events[0])
            dist.all_reduce(_logprobs,
                            op=dist.ReduceOp.SUM,
                            group=dist_process_group)
            _dedicated_stream.record_event(_dedicated_events[1])
        # communicate maximums
        # maybe _max is too large to be communicated, and cause longer communication time
        _max_backup = _max.detach()
        dist.all_reduce(_max, op=dist.ReduceOp.MAX, group=dist_process_group)
        efficient_entropy_triton_kernel_epilogue_tp[epilogue_grid](
            num_tokens, num_splits,
            _max, _max.stride(0), _max.stride(1),
            _max_backup, _max_backup.stride(0), _max_backup.stride(1),
            _accu, _accu.stride(0), _accu.stride(1),
            maximum, maximum.stride(0),
            accumulate, accumulate.stride(0),
        )
        # reduce accumulate
        dist.all_reduce(accumulate, op=dist.ReduceOp.SUM, group=dist_process_group)
        # update logprobs
        torch.cuda.current_stream().wait_event(_dedicated_events[1])
        efficient_entropy_triton_epilogue_tp_logprobs[epilogue_grid](
            num_tokens,
            _logprobs, _logprobs.stride(0),
            maximum, maximum.stride(0),
            accumulate, accumulate.stride(0),
            logprobs,
            REDUCTION,
        )
    return logprobs, maximum, accumulate

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16},
                            num_stages=3, num_warps=4)],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_d_logits(num_tokens,
                                        hidden_size,
                                        vocab_size,
                                        rank,
                                        hidden_ptr, stride_hidden_m, stride_hidden_k,
                                        weight_ptr, stride_weight_k, stride_weight_n,
                                        labels_ptr, stride_labels,
                                        maximum_ptr, stride_maximum,
                                        accu_ptr, stride_accu,
                                        d_logprobs_ptr, stride_d_logprobs,
                                        reduction: int,
                                        d_logits_ptr, stride_d_logits_m, stride_d_logits_n,
                                        BLOCK_SIZE_M: tl.constexpr,
                                        BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_K: tl.constexpr,
                                        GROUP_SIZE_M: tl.constexpr):
    """
    backward d_logits
    """
    # block swizzling
    # pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    # pid_m = pid % num_pid_m
    # pid_n = pid // num_pid_m

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_size, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    maximum = tl.load(maximum_ptr + offs_am * stride_maximum,
                      mask=offs_am < num_tokens)
    accu = tl.load(accu_ptr + offs_am * stride_accu,
                   mask=offs_am < num_tokens, other=1e-6) # epsilon avoid division by zero
    accu_rcp = tl.fdiv(1.0, accu)
    if reduction == 0: # none
        d_logprobs = tl.load(d_logprobs_ptr + offs_am * stride_d_logprobs,
                             mask=offs_am < num_tokens)
    elif reduction == 1: # sum
        d_logprobs = tl.broadcast_to(tl.load(d_logprobs_ptr), (BLOCK_SIZE_M,))
    else: # mean
        d_logprobs = tl.broadcast_to(tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32)), (BLOCK_SIZE_M,))

    d_acc_exp_logits = d_logprobs * accu_rcp
    # d_max = accu * d_acc_exp_logits - d_logprobs

    labels = tl.load(labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0)
    hidden_ptrs = hidden_ptr + (offs_am[:,None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
    weight_ptrs = weight_ptr + (offs_k[:,None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)

    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        _hidden = tl.load(hidden_ptrs,
                          mask=(offs_k[None,:] < hidden_size - k * BLOCK_SIZE_K)
                            & (offs_am[:,None] < num_tokens),
                          other=0.0)
        _weight = tl.load(weight_ptrs,
                          mask=(offs_k[:,None] < hidden_size - k * BLOCK_SIZE_K)
                            & (offs_bn[None,:] < vocab_size),
                          other=0.0)
        logits = tl.dot(_hidden, _weight, logits)

        hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
        weight_ptrs += BLOCK_SIZE_K * stride_weight_k

    exp_logits = tl.exp(logits - maximum[:,None])

    mask = (offs_bn + rank * vocab_size)[None,:] == labels[:,None]

    d_logits = exp_logits * d_acc_exp_logits[:,None]
    d_logits += tl.where(mask, -d_logprobs[:,None], 0.0)

    # store
    tl.store(d_logits_ptr + offs_am[:,None] * stride_d_logits_m + offs_bn[None,:] * stride_d_logits_n,
             d_logits.to(hidden_ptr.dtype.element_ty),
             mask=(offs_am[:,None] < num_tokens) & (offs_bn[None,:] < vocab_size))


def efficient_entropy_backward(dlogprobs: torch.Tensor,
                               hidden: torch.Tensor,
                               weight: torch.Tensor,
                               labels: torch.Tensor,
                               maximum: torch.Tensor,
                               accu: torch.Tensor,
                               reduction: typing.Optional[int] = 2,
                               dist_process_group: typing.Optional[dist.ProcessGroup] = None
) -> typing.List[torch.Tensor]:
    """
    backward host function
    """
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[0]

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    _world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    hidden_size, vocab_size = weight.shape
    assert hidden_size % 128 == 0
    assert vocab_size % 128 == 0

    REDUCTION = get_entropy_reduction_enum(reduction)
    if REDUCTION == EntropyReductionEnum._None:
        assert dlogprobs.shape == (num_tokens,)
    else:
        assert dlogprobs.dim() == 0

    assert dlogprobs.is_contiguous() and dlogprobs.is_cuda
    assert dlogprobs.device == hidden.device

    assert maximum.is_contiguous() and accu.is_contiguous()
    assert maximum.device == hidden.device and accu.device == hidden.device
    assert maximum.shape == labels.shape == accu.shape
    assert maximum.is_cuda and accu.is_cuda

    # respect's input's datatype
    d_hidden = torch.empty_like(hidden, dtype=hidden.dtype, device=hidden.device)
    d_weight = torch.empty_like(weight, dtype=hidden.dtype, device=hidden.device)

    vocab_per_split = 1024
    assert vocab_per_split % 128 == 0

    _d_logits = torch.empty((num_tokens, vocab_size), device=hidden.device, dtype=hidden.dtype)
    assert _d_logits.is_contiguous() and _d_logits.is_cuda

    def d_logits_grid(meta):
        return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"])
                * triton.cdiv(vocab_size, meta["BLOCK_SIZE_N"]),)
    efficient_entropy_backward_d_logits[d_logits_grid](
        num_tokens, hidden_size, vocab_size,
        _rank,
        hidden, hidden.stride(0), hidden.stride(1),
        weight, weight.stride(0), weight.stride(1),
        labels, labels.stride(0),
        maximum, maximum.stride(0),
        accu, accu.stride(0),
        dlogprobs, dlogprobs.stride(0) if REDUCTION == EntropyReductionEnum._None else 0,
        REDUCTION,
        _d_logits, _d_logits.stride(0), _d_logits.stride(1)
    )

    torch.matmul(_d_logits, weight.T, out=d_hidden)
    torch.matmul(hidden.T, _d_logits, out=d_weight)

    return d_hidden, d_weight
