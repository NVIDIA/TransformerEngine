import sys
import torch

from transformer_engine.pytorch import Linear, LayerNormLinear, LayerNormMLP, TransformerLayer
from transformer_engine.pytorch import get_cpu_offload_context

# Create a context and the sync function
# Number of layers to offload is 1 which is max_layers - 1
# Offloading both activations and weights
context, sync_function = get_cpu_offload_context(True,1,True,True)

#Create a simple TF layer
layer1 = TransformerLayer(768, 768, 12)
layer2 = TransformerLayer(768, 768, 12)

inp_activation = torch.rand(2048, 2, 768, device="cuda")

#Using a context
with context:
    layer1_activation = layer1(inp_activation)

#Syncrhonize offload/reload and compute
layer1_activation = sync_function(layer1_activation)

with context:
    out_activation = layer2(layer1_activation)

out_activation = sync_function(out_activation)

#Do BackProp
out_activation.sum().backward()
