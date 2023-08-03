from transformer_engine.new.pytorch import (
    Linear,
    LayerNorm,
    Residual,
    Sequential,
    GELU,
)

HIDDEN = 768
FNN = 3072
transformer = 6 * Sequential(
    Residual(
        LayerNorm(HIDDEN), Linear(HIDDEN, 3 * HIDDEN), DPA(), Linear(3 * HIDDEN, HIDDEN)
    ),
    Residual(LayerNorm(HIDDEN), Linear(HIDDEN, FNN), GELU(), Linear(FNN, HIDDEN)),
)
