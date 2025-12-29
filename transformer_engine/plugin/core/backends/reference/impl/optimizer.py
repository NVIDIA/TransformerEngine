# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Union
import torch

__all__ = [
    "multi_tensor_scale_torch",
    "multi_tensor_l2norm_torch",
    "multi_tensor_adam_torch",
    "multi_tensor_sgd_torch",
    "multi_tensor_compute_scale_and_scale_inv_torch",
]


def multi_tensor_scale_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    scale: float,
) -> None:
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 2:
        raise ValueError("tensor_lists should contain [input_tensors, output_tensors]")

    input_tensors, output_tensors = tensor_lists

    if len(output_tensors) != len(input_tensors):
        raise ValueError("Output and input tensor lists must have the same length")

    for in_tensor, out_tensor in zip(input_tensors, output_tensors):
        out_tensor.copy_(in_tensor * scale)


def multi_tensor_l2norm_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    per_tensor: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if noop_flag.item() != 0:
        if per_tensor:
            return [torch.tensor(0.0, device=t.device) for t in tensor_lists[0]]
        else:
            return torch.tensor(0.0, device=tensor_lists[0][0].device)

    tensors = tensor_lists[0]

    if per_tensor:
        norms = []
        for tensor in tensors:
            norm = torch.norm(tensor.float(), p=2)
            norms.append(norm)
        return norms
    else:
        total_norm_sq = torch.tensor(0.0, device=tensors[0].device)
        for tensor in tensors:
            total_norm_sq += torch.sum(tensor.float() ** 2)
        return torch.sqrt(total_norm_sq)


def multi_tensor_adam_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
) -> None:
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 4:
        raise ValueError("tensor_lists should contain [grads, params, exp_avgs, exp_avg_sqs]")

    grads, params, exp_avgs, exp_avg_sqs = tensor_lists

    if not (len(params) == len(grads) == len(exp_avgs) == len(exp_avg_sqs)):
        raise ValueError("All tensor lists must have the same length")

    if bias_correction:
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
    else:
        bias_correction1 = 1.0
        bias_correction2 = 1.0

    for grad, param, exp_avg, exp_avg_sq in zip(grads, params, exp_avgs, exp_avg_sqs):
        if grad is None:
            continue

        if mode == 1 and weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        corrected_exp_avg = exp_avg / bias_correction1
        corrected_exp_avg_sq = exp_avg_sq / bias_correction2

        denom = corrected_exp_avg_sq.sqrt().add_(eps)
        param.addcdiv_(corrected_exp_avg, denom, value=-lr)


def multi_tensor_sgd_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
) -> None:
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 3:
        raise ValueError("tensor_lists should contain [params, grads, momentum_buffers]")

    params, grads, momentum_buffers = tensor_lists

    if not (len(params) == len(grads) == len(momentum_buffers)):
        raise ValueError("All tensor lists must have the same length")

    for param, grad, buf in zip(params, grads, momentum_buffers):
        if grad is None:
            continue

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            if buf is None or buf.numel() == 0:
                buf = grad.clone().detach()
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        param.add_(grad, alpha=-lr)


def multi_tensor_compute_scale_and_scale_inv_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    max_fp8: float,
    force_pow_2_scales: bool = False,
    amax_epsilon: float = 0.0,
) -> None:
    """
    Compute scale and scale_inv from amax values for FP8 quantization.

    Args:
        chunk_size: Chunk size (unused in PyTorch implementation)
        noop_flag: If non-zero, skip computation
        tensor_lists: [amaxes, scales, scale_invs]
        max_fp8: Maximum representable value in FP8 format (e.g., 448.0 for E4M3)
        force_pow_2_scales: If True, force scales to be powers of 2
        amax_epsilon: Small epsilon to add to amax to avoid division by zero
    """
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 3:
        raise ValueError("tensor_lists should contain [amaxes, scales, scale_invs]")

    amaxes, scales, scale_invs = tensor_lists

    if not (len(amaxes) == len(scales) == len(scale_invs)):
        raise ValueError("All tensor lists must have the same length")

    for amax, scale, scale_inv in zip(amaxes, scales, scale_invs):
        # Add epsilon to avoid division by zero
        amax_val = amax + amax_epsilon

        # Compute scale: max_fp8 / amax
        # Clamp amax to avoid very small values
        amax_val = torch.clamp(amax_val, min=1e-12)
        computed_scale = max_fp8 / amax_val

        if force_pow_2_scales:
            # Round scale to nearest power of 2
            log2_scale = torch.log2(computed_scale)
            log2_scale = torch.round(log2_scale)
            computed_scale = torch.pow(2.0, log2_scale)

        # Update scale and scale_inv
        scale.copy_(computed_scale)
        scale_inv.copy_(1.0 / computed_scale)
