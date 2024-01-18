# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

from transformer_engine.pytorch import Linear
from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context

# Create a context and the sync function
context, sync_function = get_cpu_offload_context(True,1,True,True)

#Create a simple Linear layer
layer = Linear(3,3)

inp_activation = torch.rand(3,3,device="cuda")

#Using a context
with context:
    out_activation = layer(inp_activation)

#Syncrhonize offload/reload and compute
out_activation = sync_function(out_activation)

#Do BackProp
out_activation.sum().backward()
