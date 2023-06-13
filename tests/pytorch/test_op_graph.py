import transformer_engine.pytorch as te
from transformer_engine.pytorch.sequential import ComputePipeline

seq = te.Sequential(te.LayerNormLinear(1, 1))
c = ComputePipeline(*seq._modules.values())
