from __future__ import annotations
import torch
from torch import nn
import transformer_engine.pytorch.sequential as seq
import transformer_engine.pytorch as te

BATCH_SIZE = 512
IN_FEATURES = 768
OUT_FEATURES = 4 * IN_FEATURES


def cpy(dst: torch.Tensor, src: torch.Tensor):
    dst.data = torch.as_tensor(src.data.clone().detach(), dtype=dst.dtype).detach()


def max_abs_diff(ref: torch.Tensor, cand: torch.Tensor):
    # ab = abs(cand-ref).max().item()
    # rl = abs((cand-ref)/ref).max().item()
    # s=""
    # if ab < 0.001:
    #     s += f"a:\033[32m{ab:18.5f}\033[0m,"
    # elif ab< 0.1:
    #     s += f"a:\033[33m{ab:18.5f}\033[0m,"
    # else:
    #     s += f"a:\033[31m{ab:18.5f}\033[0m,"

    # if rl < 0.001:
    #     s += f"r:\033[32m{rl:18.5f}\033[0m"
    # elif rl< 0.1:
    #     s += f"r:\033[33m{rl:18.5f}\033[0m"
    # else:
    #     s += f"r:\033[31m{rl:18.5f}\033[0m"
    # return s

    try:
        torch.testing.assert_close(cand, ref, atol=1e-5, rtol=1e-3)
        ok = True
    except AssertionError as e:
        ok = False
        print(str(e))

    if ok:
        return "\033[32mOK\033[0m"
    else:
        return "\033[31mWA\033[0m"


def test(
    enable_first_linear: bool,
    use_te_linear: bool,
    use_te_act: bool,
    use_relu: bool,
    use_gelu: bool,
    div_std: bool,
    enable_second_linear: bool,
    lin1_w: torch.Tensor,
    lin1_b: torch.Tensor,
    lin2_w: torch.Tensor,
    lin2_b: torch.Tensor,
    inp: torch.Tensor,
):
    if enable_first_linear:
        if use_te_linear:
            lin1 = te.Linear(IN_FEATURES, OUT_FEATURES)
            cpy(lin1.weight, lin1_w)
            cpy(lin1.bias, lin1_b)
        else:
            lin1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
            cpy(lin1.weight, lin1_w)
            cpy(lin1.bias, lin1_b)
    else:
        lin1 = lambda x: x

    if enable_second_linear:
        if enable_first_linear:
            if use_te_linear:
                lin2 = te.Linear(OUT_FEATURES, IN_FEATURES)
                cpy(lin2.weight, lin2_w)
                cpy(lin2.bias, lin2_b)
            else:
                lin2 = nn.Linear(IN_FEATURES, OUT_FEATURES)
                cpy(lin2.weight, lin2_w)
                cpy(lin2.bias, lin2_b)
        else:
            if use_te_linear:
                lin2 = te.Linear(IN_FEATURES, OUT_FEATURES)
                cpy(lin2.weight, lin1_w)
                cpy(lin2.bias, lin1_b)
            else:
                lin2 = nn.Linear(IN_FEATURES, OUT_FEATURES)
                cpy(lin2.weight, lin1_w)
                cpy(lin2.bias, lin1_b)
    else:
        lin2 = lambda x: x

    if use_relu:
        if use_te_act:
            relu = seq.ReLU()
        else:
            relu = nn.ReLU()
    else:
        relu = lambda x: x

    if use_gelu:
        if use_te_act:
            gelu = seq.GELU()
        else:
            gelu = nn.GELU(approximate="tanh")
    else:
        gelu = lambda x: x

    x = inp.detach().clone().requires_grad_()
    x1 = x / x.std() if div_std else x
    x2 = lin1(x1)
    x3 = relu(x2)
    x4 = gelu(x3)
    x5 = lin2(x4)
    x5.sum().backward()
    assert x.grad is not None
    return x.grad


results = {}

for _ in range(50):
    lin1 = nn.Linear(IN_FEATURES, OUT_FEATURES, device="cuda")
    lin2 = nn.Linear(OUT_FEATURES, IN_FEATURES, device="cuda")
    x = torch.rand(BATCH_SIZE, IN_FEATURES, device="cuda") * 2.0 - 1.0

    for i in range(128):
        (
            enable_first_linear,
            use_te_linear,
            use_te_act,
            use_relu,
            use_gelu,
            div_std,
            enable_second_linear,
        ) = (bool(i & (1 << j)) for j in range(7))

        if use_relu and use_gelu:
            continue
        ref_use_te_linear = False
        ref_use_te_act = False
        if ref_use_te_linear == use_te_linear and ref_use_te_act == use_te_act:
            continue
        if (
            not enable_first_linear
            and not enable_second_linear
            and not use_relu
            and not use_gelu
        ):
            continue
        if (
            not use_relu
            and not use_gelu
            and (use_te_act or ref_use_te_linear == use_te_linear)
        ):
            continue
        if (
            not enable_first_linear
            and not enable_second_linear
            and (use_te_linear or ref_use_te_act == use_te_act)
        ):
            continue
        if (
            not enable_first_linear
            and not use_relu
            and not use_gelu
            and enable_second_linear
        ):
            continue

        ref = test(
            enable_first_linear,
            ref_use_te_linear,
            ref_use_te_act,
            use_relu,
            use_gelu,
            div_std,
            enable_second_linear,
            lin1.weight,
            lin1.bias,
            lin2.weight,
            lin2.bias,
            x,
        )
        cand = test(
            enable_first_linear,
            use_te_linear,
            use_te_act,
            use_relu,
            use_gelu,
            div_std,
            enable_second_linear,
            lin1.weight,
            lin1.bias,
            lin2.weight,
            lin2.bias,
            x,
        )
        if i not in results:
            results[i] = [max_abs_diff(ref, cand)]
        else:
            results[i].append(max_abs_diff(ref, cand))

    del lin1, lin2, x

for i, res in results.items():
    (
        enable_first_linear,
        use_te_linear,
        use_te_act,
        use_relu,
        use_gelu,
        div_std,
        enable_second_linear,
    ) = (bool(i & (1 << j)) for j in range(7))

    s = ""
    if div_std:
        s += "RMSNorm, "
    if enable_first_linear:
        if use_te_linear:
            s += "te.Linear, "
        else:
            s += "nn.Linear, "
    if use_relu:
        if use_te_act:
            s += "seq.ReLU, "
        else:
            s += "nn.ReLU, "
    if use_gelu:
        if use_te_act:
            s += "seq.GELU, "
        else:
            s += "nn.GELU, "
    if enable_second_linear:
        if use_te_linear:
            s += "te.Linear, "
        else:
            s += "nn.Linear, "
    s = s[:-2] + ": "
    s = s.rjust(45)

    print(s, end="")
    for r in res:
        print(f"{r}, ", end="")
    print()
