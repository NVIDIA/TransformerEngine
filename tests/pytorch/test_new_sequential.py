import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.fp8 import fp8_autocast
from transformer_engine.pytorch.sequential_new import Sequential, Residual

model = Sequential(
    Residual(
        te.LayerNorm(10),
        te.Linear(10, 10),
    )
)

rec = recipe.DelayedScaling()

tensor = torch.randn(10, 10)
with fp8_autocast():
    out = model(tensor)
print(out)
