from re import A
from torch import nn
import torch.cuda
import transformer_engine.pytorch as te

from transformer_engine.pytorch.sequential.simple_compute_pipeline import (
    ComputePipeline,
)


def test(modules: tuple[nn.Module, ...], targets: tuple[nn.Module, ...]):
    pipeline = ComputePipeline(*modules)
    assert all(
        isinstance(pipeline.module[i], type(targets[i]))
        for i in range(max(len(pipeline.module), len(targets)))
    )
    print(f"Passed: {modules} -> {targets}")


torch.set_default_device("cuda")

# Unrecognized modules stay the same
test(
    (nn.AlphaDropout(),),
    (nn.AlphaDropout(),),
)

# Linear promotion
test(
    (nn.Linear(1, 1),),
    (te.Linear(1, 1),),
)
test(
    (nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1)),
    (te.Linear(1, 1), te.Linear(1, 1), te.Linear(1, 1)),
)

# LayerNorm promotion
test(
    (nn.LayerNorm(1),),
    (te.LayerNorm(1),),
)
test(
    (
        nn.LayerNorm(1),
        nn.LayerNorm(1),
        nn.LayerNorm(1),
    ),
    (te.LayerNorm(1), te.LayerNorm(1), te.LayerNorm(1)),
)

# LayerNormLinear merge
test(
    (
        nn.Linear(1, 1),
        nn.LayerNorm(1),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
    ),
    (te.Linear(1, 1), te.LayerNormLinear(1, 1), te.Linear(1, 1)),
)

# LayerNormMLP merge
test(
    (
        nn.Linear(1, 1),
        nn.LayerNorm(1),
        nn.Linear(1, 1),
        nn.GELU(),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
    ),
    (te.Linear(1, 1), te.LayerNormMLP(1, 1), te.Linear(1, 1)),
)
test(
    (
        nn.Linear(1, 1),
        te.LayerNormLinear(1, 1),
        nn.GELU(),
        nn.Linear(1, 1),
        nn.Linear(1, 1),
    ),
    (te.Linear(1, 1), te.LayerNormMLP(1, 1), te.Linear(1, 1)),
)
