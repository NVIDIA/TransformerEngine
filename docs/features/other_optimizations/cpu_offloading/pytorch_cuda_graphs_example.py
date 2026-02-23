# START_CUDA_GRAPHS_EXAMPLE
import torch
from transformer_engine.pytorch import get_cpu_offload_context, make_graphed_callables

# Setup
num_layers = 12
offloaded_layers = 3
layers = [torch.nn.Linear(1024, 1024).cuda() for _ in range(num_layers)]

# Enable offloading with retained buffers for CUDA graphs
cpu_offload_context, sync_function = get_cpu_offload_context(
    enabled=True,
    model_layers=num_layers,
    num_layers=offloaded_layers,
    retain_pinned_cpu_buffers=True,
)


# Wrap layers in a module that uses offloading
class OffloadedModel(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            with cpu_offload_context:
                x = layer(x)
            x = sync_function(x)
        return x


model = OffloadedModel(layers)
sample_input = (torch.randn(16, 1024, 1024, device="cuda"),)

# Create graphed callable (warmup is handled internally)
graphed_model = make_graphed_callables(model, sample_input)

# Use the graphed model
x = torch.randn(16, 1024, 1024, device="cuda")
out = graphed_model(x)
out.sum().backward()
# END_CUDA_GRAPHS_EXAMPLE
