import torch
import transformer_engine.pytorch.sequential as seq

N = 2048
HIDDEN_DIM = 1024
x = torch.rand(N, HIDDEN_DIM, device="cuda", requires_grad=True)

m = seq.Sequential(
    seq.RMSNorm(HIDDEN_DIM),
    seq.Linear(HIDDEN_DIM, 4 * HIDDEN_DIM),
    seq.SwiGLU(),
    seq.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
)

with seq.environment(seq.nvte.DType.Float8E4M3):
    y = m(x)
    y.sum().backward()
    print(x.grad)

with seq.environment(seq.nvte.DType.BFloat16):
    y = m(x)
    y.sum().backward()
    print(x.grad)

with seq.environment(seq.nvte.DType.Float32):
    y = m(x)
    y.sum().backward()
    print(x.grad)
