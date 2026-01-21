# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional, List
import torch
import flag_gems


def multi_tensor_adam_fl(
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
    inv_scale: Optional[float] = 1.0,
    out_dtype: Optional[torch.dtype] = None,
) -> None:

    num_lists = len(tensor_lists)
    assert num_lists in [4, 5], f"Expected 4 or 5 tensor lists, got {num_lists}"

    num_tensors = len(tensor_lists[0])
    assert num_tensors > 0, "No tensors provided"

    for i, lst in enumerate(tensor_lists):
        assert len(lst) == num_tensors, f"List {i} has {len(lst)} tensors, expected {num_tensors}"

    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if bias_correction == 1:
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

    is_adamw = (mode == 1)

    for i in range(num_tensors):
        g = tensor_lists[0][i]
        p = tensor_lists[1][i]
        m = tensor_lists[2][i]
        v = tensor_lists[3][i]
        p_master = tensor_lists[4][i] if num_lists == 5 else None

        if not g.is_contiguous():
            g = g.contiguous()

        if inv_scale is not None and inv_scale != 1.0:
            g = flag_gems.mul(g, inv_scale)

        m = flag_gems.add_(flag_gems.mul_(m, beta1), g, alpha=1-beta1)
        v = flag_gems.add_(flag_gems.mul_(v, beta2), flag_gems.mul_(flag_gems.mul_(g, g), 1 - beta2))

        m_corr = m.clone()
        v_corr = v.clone()
        if bias_correction == 1:
            m_corr = flag_gems.true_divide(m_corr, bias_correction1)
            v_corr = flag_gems.true_divide(v_corr, bias_correction2)

        update = flag_gems.true_divide(m_corr, flag_gems.add(flag_gems.sqrt(v_corr), eps))

        if is_adamw:
            p = flag_gems.mul_(p, 1 - lr * weight_decay)
        else:
            update = flag_gems.add_(update, p, alpha=weight_decay)

        p = flag_gems.add_(p, update, alpha=-lr)

        if p_master is not None:
            flag_gems.copy_(p_master, p)
            out_dtype = p_master.dtype if out_dtype is None else out_dtype
            p.data = p.data.to(out_dtype)


def multi_tensor_adam_param_remainder_fl(
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
    inv_scale: Optional[float] = 1.0,
) -> None:
    """
    Adam optimizer with parameter remainders for BF16 precision (FlagOS implementation).

    This variant stores BF16 parameters + int16 remainders to reconstruct FP32 master weights.
    Used when you have BF16 params and need FP32 master params without storing full FP32 copies.

    Args:
        chunk_size: Chunk size for processing (unused in this implementation)
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
        inv_scale: Inverse gradient scale for mixed precision training
    """
    if noop_flag.item() != 0:
        return

    num_lists = len(tensor_lists)
    assert num_lists == 5, f"Expected 5 tensor lists, got {num_lists}"

    num_tensors = len(tensor_lists[0])
    assert num_tensors > 0, "No tensors provided"

    for i, lst in enumerate(tensor_lists):
        assert len(lst) == num_tensors, f"List {i} has {len(lst)} tensors, expected {num_tensors}"

    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if bias_correction == 1:
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

    is_adamw = (mode == 1)

    for i in range(num_tensors):
        g = tensor_lists[0][i]
        p = tensor_lists[1][i]  # BF16 parameter
        m = tensor_lists[2][i]  # FP32 first moment
        v = tensor_lists[3][i]  # FP32 second moment
        p_remainder = tensor_lists[4][i]  # int16 remainder

        if not g.is_contiguous():
            g = g.contiguous()

        # Apply gradient unscaling if needed
        if inv_scale is not None and inv_scale != 1.0:
            g = flag_gems.mul(g, inv_scale)

        # Reconstruct FP32 master weight from BF16 param + int16 remainder
        # The remainder represents the lower 16 bits lost in BF16 conversion
        param_fp32 = p.float()
        param_master = flag_gems.add(param_fp32, flag_gems.mul(p_remainder.float(), 2.0 ** -16))

        # Compute gradient with weight decay (if L2 mode)
        grad_with_decay = g.float()
        if not is_adamw:  # L2 regularization mode
            grad_with_decay = flag_gems.add(grad_with_decay, flag_gems.mul(param_master, weight_decay))

        # Update moments
        m = flag_gems.add_(flag_gems.mul_(m, beta1), grad_with_decay, alpha=1 - beta1)
        v = flag_gems.add_(flag_gems.mul_(v, beta2), flag_gems.mul_(flag_gems.mul_(grad_with_decay, grad_with_decay), 1 - beta2))

        # Apply bias correction
        m_corr = m.clone()
        v_corr = v.clone()
        if bias_correction == 1:
            m_corr = flag_gems.true_divide(m_corr, bias_correction1)
            v_corr = flag_gems.true_divide(v_corr, bias_correction2)

        # Compute update
        update = flag_gems.true_divide(m_corr, flag_gems.add(flag_gems.sqrt(v_corr), eps))

        # Apply weight decay (if AdamW mode)
        if is_adamw:
            param_master = flag_gems.mul_(param_master, 1 - lr * weight_decay)

        # Update master weight
        param_master = flag_gems.add_(param_master, update, alpha=-lr)

        # Split back into BF16 param + int16 remainder
        # Convert to BF16 (this is the rounded version)
        param_bf16 = param_master.to(dtype=p.dtype)

        # Compute remainder: difference between FP32 master and BF16 representation
        # Scale and quantize to int16 range
        remainder_fp32 = flag_gems.mul(flag_gems.sub(param_master, param_bf16.float()), 2.0 ** 16)
        remainder_int16 = flag_gems.clamp(torch.round(remainder_fp32), -32768, 32767).to(dtype=torch.int16)

        # Write back
        flag_gems.copy_(p, param_bf16)
        flag_gems.copy_(p_remainder, remainder_int16)
