# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Union
import torch

__all__ = [
    "multi_tensor_scale_torch",
    "multi_tensor_l2norm_torch",
    "multi_tensor_adam_torch",
    "multi_tensor_adam_param_remainder_torch",
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


def multi_tensor_adam_param_remainder_torch(
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
    """
    Adam optimizer with parameter remainders for BF16 precision.

    This variant stores BF16 parameters + int16 remainders to reconstruct FP32 master weights.
    Used when you have BF16 params and need FP32 master params without storing full FP32 copies.

    Args:
        chunk_size: Chunk size for processing (unused in PyTorch implementation)
        noop_flag: If non-zero, skip computation
        tensor_lists: [grads, params (bf16), exp_avgs (fp32), exp_avg_sqs (fp32), param_remainders (int16)]
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Epsilon for numerical stability
        step: Current optimization step
        mode: 0 = L2 regularization, 1 = AdamW (decoupled weight decay)
        bias_correction: Whether to apply bias correction (1 = yes, 0 = no)
        weight_decay: Weight decay coefficient
    """
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 5:
        raise ValueError(
            "tensor_lists should contain [grads, params, exp_avgs, exp_avg_sqs, param_remainders]"
        )

    grads, params, exp_avgs, exp_avg_sqs, param_remainders = tensor_lists

    if not (len(params) == len(grads) == len(exp_avgs) == len(exp_avg_sqs) == len(param_remainders)):
        raise ValueError("All tensor lists must have the same length")

    if bias_correction:
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
    else:
        bias_correction1 = 1.0
        bias_correction2 = 1.0

    for grad, param, exp_avg, exp_avg_sq, param_remainder in zip(
        grads, params, exp_avgs, exp_avg_sqs, param_remainders
    ):
        if grad is None:
            continue

        # Reconstruct FP32 master weight from BF16 param + int16 remainder
        # The CUDA implementation uses bit manipulation to combine them
        # In PyTorch, we approximate this by:
        # 1. Convert param (bf16) to fp32 - this gives us the high-precision bits
        # 2. Add the remainder scaled appropriately
        param_fp32 = param.float()

        # The remainder represents the lower 16 bits lost in BF16 conversion
        # We need to scale it back to the proper magnitude
        # BF16 has 16 bits total (1 sign, 8 exponent, 7 mantissa)
        # The remainder compensates for the lost precision
        param_master = param_fp32 + param_remainder.float() * (2.0 ** -16)

        # Standard Adam update on FP32 master weight
        if mode == 0:  # L2 regularization
            grad_with_decay = grad.float() + weight_decay * param_master
        else:  # mode == 1, AdamW
            grad_with_decay = grad.float()

        # Update moments
        exp_avg.mul_(beta1).add_(grad_with_decay, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_with_decay, grad_with_decay, value=1 - beta2)

        # Apply bias correction
        corrected_exp_avg = exp_avg / bias_correction1
        corrected_exp_avg_sq = exp_avg_sq / bias_correction2

        # Compute update
        denom = corrected_exp_avg_sq.sqrt().add_(eps)
        update = corrected_exp_avg / denom

        if mode == 1:  # AdamW: apply weight decay directly
            update = update + weight_decay * param_master

        # Update master weight
        param_master.add_(update, alpha=-lr)

        # Split back into BF16 param + int16 remainder
        # Convert to BF16 (this is the rounded version)
        param_bf16 = param_master.to(dtype=param.dtype)

        # Compute remainder: difference between FP32 master and BF16 representation
        # Scale and quantize to int16 range
        remainder_fp32 = (param_master - param_bf16.float()) * (2.0 ** 16)
        remainder_int16 = remainder_fp32.round().clamp(-32768, 32767).to(dtype=torch.int16)

        # Write back
        param.copy_(param_bf16)
        param_remainder.copy_(remainder_int16)


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
