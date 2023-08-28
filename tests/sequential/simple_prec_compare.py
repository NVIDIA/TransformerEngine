import torch
from torch import nn
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
torch.set_printoptions(precision=4, sci_mode=False)


torch.compile(m.precompiled_for(x), fullgraph=True)(x)

with seq.Recipe(lowp=seq.nvte.DType.Float8E4M3):
    opt: nn.Module = torch.compile(m.precompiled_for(x), fullgraph=True, dynamic=True)
    for _ in range(100):
        y: torch.Tensor = opt(x)
        y.sum().backward()
        print(x.grad)
        x.grad = None

with seq.Recipe(lowp=seq.nvte.DType.BFloat16):
    y = m(x)
    y.sum().backward()
    print(x.grad)
    x.grad = None

with seq.Recipe(lowp=seq.nvte.DType.Float32):
    y = m(x)
    y.sum().backward()
    print(x.grad)
    x.grad = None
