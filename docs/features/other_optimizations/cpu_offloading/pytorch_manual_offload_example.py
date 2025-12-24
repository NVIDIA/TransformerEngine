# START_MANUAL_EXAMPLE
import torch
from transformer_engine.pytorch import get_cpu_offload_context

# Setup
num_layers = 12
layers = [torch.nn.Linear(1024, 1024).cuda() for _ in range(num_layers)]
x = torch.randn(16, 1024, 1024, device="cuda")

offload_stream = torch.cuda.Stream()
cpu_offload_context, sync_function, manual_controller = get_cpu_offload_context(
    enabled=True,
    model_layers=num_layers,
    manual_synchronization=True,
    offload_stream=offload_stream,
)

# Forward pass - manually trigger offload after each layer
for i in range(num_layers):
    with cpu_offload_context:
        x = layers[i](x)
    x = sync_function(x)
    manual_controller.start_offload_layer(i)

# Wait for offloads, then release GPU memory
offload_stream.synchronize()
for i in range(num_layers):
    manual_controller.release_activation_forward_gpu_memory(i)

# Start reloading before backward
for i in range(num_layers - 1, -1, -1):
    manual_controller.start_reload_layer(i)

# Backward pass
loss = x.sum()
loss.backward()
# END_MANUAL_EXAMPLE
