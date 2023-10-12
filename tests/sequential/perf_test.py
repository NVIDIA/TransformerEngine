import torch
import transformer_engine.pytorch.sequential as seq
from torch import nn
import transformer_engine.pytorch as te
from math import sqrt

SEQ_LEN = 4096
HIDDEN_DIM = 1024

seq.Sequential(
    seq.RMSNorm(HIDDEN_DIM),
)


vasavani_dec = te.Sequential(
    te.Residual(
        te.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM),
        te.DotProductAttention(24),
        te.Linear(HIDDEN_DIM, HIDDEN_DIM),
        te.LayerNorm(HIDDEN_DIM),
    ),
    te.Residual(
        te.Linear(HIDDEN_DIM, 4 * HIDDEN_DIM),
        te.ReLU(),
        te.Linear(4 * HIDDEN_DIM, HIDDEN_DIM),
        te.LayerNorm(HIDDEN_DIM),
    ),
)

gpt = te.Sequential(
    te.Residual(
        te.LayerNorm(HIDDEN_DIM),
        te.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM),
        te.DotProductAttention(24),
        te.Linear(HIDDEN_DIM, HIDDEN_DIM),
        te.Dropout(0.1),
    ),
    te.Residual(
        te.LayerNorm(HIDDEN_DIM),
        te.Linear(HIDDEN_DIM, 4 * HIDDEN_DIM),
        te.GELU(),
        te.Linear(4 * HIDDEN_DIM, HIDDEN_DIM),
        te.Dropout(0.1),
    ),
)

llama = te.Sequential(
    te.Residual(
        te.RMSNorm(HIDDEN_DIM),
        te.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM),
        te.DotProductAttention(24),
        te.Linear(HIDDEN_DIM, HIDDEN_DIM),
        te.Dropout(0.1),
    ),
    te.Residual(
        te.RMSNorm(HIDDEN_DIM),
        te.Linear(HIDDEN_DIM, 4 * HIDDEN_DIM),
        te.SwiGLU(),
        te.Linear(4 * HIDDEN_DIM, HIDDEN_DIM),
        te.Dropout(0.1),
    ),
)
