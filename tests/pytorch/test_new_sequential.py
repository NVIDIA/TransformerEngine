import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.fp8 import fp8_autocast
from transformer_engine.pytorch.sequential_new import Sequential, Residual

TOKENS = 512
HIDDEN = 768

model = Sequential(
    Residual(
        te.LayerNorm(HIDDEN),
        te.Linear(HIDDEN, 3 * HIDDEN),
    )
)

rec = recipe.DelayedScaling()

tensor = torch.full((TOKENS, HIDDEN), 10.0, device="cuda", requires_grad=True)
with fp8_autocast(True):
    out = model(tensor)
print(out)
out.sum().backward()
print(tensor.grad)
