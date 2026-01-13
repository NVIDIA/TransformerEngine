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
