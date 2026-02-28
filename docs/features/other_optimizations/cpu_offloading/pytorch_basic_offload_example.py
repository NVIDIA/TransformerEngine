# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_BASIC_EXAMPLE
import torch
from transformer_engine.pytorch import get_cpu_offload_context

# Setup
num_layers = 12
offloaded_layers = 3
layers = [torch.nn.Linear(1024, 1024).cuda() for _ in range(num_layers)]
x = torch.randn(16, 1024, 1024, device="cuda")

# Get offloading context and sync function
cpu_offload_context, sync_function = get_cpu_offload_context(
    enabled=True,
    model_layers=num_layers,
    num_layers=offloaded_layers,
)

# Forward pass
for i in range(num_layers):
    # Context manager captures tensors saved for backward.
    # These tensors will be offloaded to CPU asynchronously.
    with cpu_offload_context:
        x = layers[i](x)

    # sync_function must be called after each layer's forward pass.
    # This cannot be done inside the context manager because
    # it needs the output tensor after the layer has finished.
    x = sync_function(x)

loss = x.sum()
loss.backward()
# END_BASIC_EXAMPLE
