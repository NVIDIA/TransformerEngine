import torch
import transformer_engine.pytorch.sequential as seq
from torch import nn
import transformer_engine.pytorch as te
from math import sqrt

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor):
        x_norm = x.norm(2, dim=-1, keepdim=True)
        rms_x = x_norm / sqrt(self.hidden_dim)
        y = x / (rms_x + self.eps)
        return y * self.weight


torch.set_default_device("cuda")

SEQ_LEN = 128
HIDDEN_DIM = 768


def max_abs_diff(a: torch.Tensor, b: torch.Tensor):
    v = (a - b).abs().max().item()
    if v >= 0.001:
        return f"\033[31m{v}\033[0m"
    else:
        return f"\033[32m{v}\033[0m"


def cpy(dst: torch.Tensor, src: torch.Tensor):
    dst.data = torch.as_tensor(src.data.clone().detach(), dtype=dst.dtype).detach()


def cmp_modules(te: nn.Module, seq: nn.Module, pt: nn.Module):
    x_te = x_src.detach().clone().requires_grad_()
    x_seq = x_src.detach().clone().requires_grad_()
    x_pt = x_src.detach().clone().requires_grad_()

    y_te = te(x_te)
    y_seq = seq(x_seq)
    y_pt = pt(x_pt)

    y_te.sum().backward()
    y_seq.sum().backward()
    y_pt.sum().backward()

    print(f"mad(dx_te, dx_seq): {max_abs_diff(x_te.grad, x_seq.grad):12.10f}")
    print(f"mad(dx_te,  dx_pt): {max_abs_diff(x_te.grad, x_pt.grad):12.10f}")
    print(f"mad(dx_seq, dx_pt): {max_abs_diff(x_seq.grad,x_pt.grad):12.10f}")

    print(f"mad( y_te,  y_seq): {max_abs_diff(y_te, y_seq):12.10f}")
    print(f"mad( y_te,   y_pt): {max_abs_diff(y_te, y_pt):12.10f}")
    print(f"mad( y_seq,  y_pt): {max_abs_diff(y_seq,y_pt):12.10f}")


def cmp_layernorm_mlp(norm: str, act: str):
    m_seq = seq.Sequential(
        seq.LayerNorm(HIDDEN_DIM) if norm == "LayerNorm" else seq.RMSNorm(HIDDEN_DIM),
        seq.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM),
        seq.GELU() if act == "gelu" else seq.ReLU(),
        seq.Linear(3 * HIDDEN_DIM, HIDDEN_DIM),
    )
    m_te = te.LayerNormMLP(
        HIDDEN_DIM, 3 * HIDDEN_DIM, activation=act, normalization=norm
    )
    m_pt = nn.Sequential(
        nn.LayerNorm(HIDDEN_DIM) if norm == "LayerNorm" else RMSNorm(HIDDEN_DIM),
        nn.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM),
        nn.GELU() if act == "gelu" else nn.ReLU(),
        nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM),
    )

    cpy(m_te.layer_norm_weight, m_seq._modules["0"].weight)
    if norm == "LayerNorm":
        cpy(m_te.layer_norm_bias, m_seq._modules["0"].bias)
    cpy(m_te.fc1_weight, m_seq._modules["1"].weight)
    cpy(m_te.fc1_bias, m_seq._modules["1"].bias)
    cpy(m_te.fc2_weight, m_seq._modules["3"].weight)
    cpy(m_te.fc2_bias, m_seq._modules["3"].bias)

    cpy(m_pt[0].weight, m_seq._modules["0"].weight)
    if norm == "LayerNorm":
        cpy(m_pt[0].bias, m_seq._modules["0"].bias)
    cpy(m_pt[1].weight, m_seq._modules["1"].weight)
    cpy(m_pt[1].bias, m_seq._modules["1"].bias)
    cpy(m_pt[3].weight, m_seq._modules["3"].weight)
    cpy(m_pt[3].bias, m_seq._modules["3"].bias)

    cmp_modules(m_te, m_seq, m_pt)


def cmp_layernorm():
    m_seq = seq.LayerNorm(HIDDEN_DIM)
    m_te = te.LayerNorm(HIDDEN_DIM)
    m_pt = nn.LayerNorm(HIDDEN_DIM)

    cpy(m_te.weight, m_seq.weight)
    cpy(m_te.bias, m_seq.bias)
    cpy(m_pt.weight, m_seq.weight)
    cpy(m_pt.bias, m_seq.bias)

    cmp_modules(m_te, m_seq, m_pt)


def cmp_linear():
    m_seq = seq.Linear(HIDDEN_DIM, HIDDEN_DIM)
    m_te = te.Linear(HIDDEN_DIM, HIDDEN_DIM)
    m_pt = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    cpy(m_te.weight, m_seq.weight)
    cpy(m_te.bias, m_seq.bias)
    cpy(m_pt.weight, m_seq.weight)
    cpy(m_pt.bias, m_seq.bias)

    cmp_modules(m_te, m_seq, m_pt)


def cmp_linear_no_bias():
    m_seq = seq.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
    m_te = te.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
    m_pt = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

    cpy(m_te.weight, m_seq.weight)
    cpy(m_pt.weight, m_seq.weight)

    cmp_modules(m_te, m_seq, m_pt)


print("\n ----- FP32 INPUT & WEIGHTS ------")
x_src = torch.rand(SEQ_LEN, HIDDEN_DIM, device="cuda")

print("\n### Comparing LayerNormMPL (gelu) ###")
cmp_layernorm_mlp("LayerNorm", "gelu")

print("\n### Comparing LayerNormMPL (relu) ###")
cmp_layernorm_mlp("LayerNorm", "relu")

print("\n### Comparing RMSNormMPL (gelu) ###")
cmp_layernorm_mlp("RMSNorm", "gelu")

print("\n### Comparing RMSNormMPL (relu) ###")
cmp_layernorm_mlp("RMSNorm", "relu")

print("\n### Comparing LayerNorm ###")
cmp_layernorm()

print("\n### Comparing Linear ###")
cmp_linear()

print("\n### Comparing Linear (no bias) ###")
cmp_linear_no_bias()
