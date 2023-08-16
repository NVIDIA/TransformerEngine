import torch
import transformer_engine.pytorch.sequential as seq

SEQ_LEN = 128
HIDDEN_DIM = 768
FFN_DIM = 4 * HIDDEN_DIM

seq.Sequential(
    seq.Residual(
        seq.RMSNorm(HIDDEN_DIM),
        seq.Linear(HIDDEN_DIM, 3 * HIDDEN_DIM),
        seq.DotProductAttention(),
        seq.Linear(3 * HIDDEN_DIM, HIDDEN_DIM),
    ),
    seq.Residual(
        seq.RMSNorm(HIDDEN_DIM),
        seq.Linear(HIDDEN_DIM, FFN_DIM),
        seq.GELU(),
        seq.Linear(FFN_DIM, HIDDEN_DIM),
    ),
)
