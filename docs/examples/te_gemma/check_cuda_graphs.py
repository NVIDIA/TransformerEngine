import torch
from transformer_engine.pytorch import Linear, LayerNorm


# 1. Define model with static buffers
class TE_Model(torch.nn.Module):
    def __init__(self, max_seq_len=4096):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.ln = LayerNorm(1024)
        self.attn_proj = Linear(1024, 1024)

        # Pre-allocate static buffers
        self.register_buffer("kv_cache", torch.zeros(max_seq_len, 1024, device="cuda"))
        self.register_buffer(
            "attn_mask", torch.tril(torch.ones(max_seq_len, max_seq_len, device="cuda"))
        )

    def forward(self, hidden_states, seq_start: int):
        # Dynamic slicing of static buffers
        seq_len = hidden_states.size(1)
        current_mask = self.attn_mask[seq_start : seq_start + seq_len, :seq_len]

        x = self.ln(hidden_states)
        x = self.attn_proj(x)
        # Update KV cache (in-place)
        self.kv_cache[seq_start : seq_start + seq_len].copy_(x)
        return x


# 2. Create graphable callables
model = TE_Model().cuda()
static_input = torch.randn(8, 256, 1024, device="cuda")  # (batch, seq, hidden)
seq_start = torch.tensor(0, device="cuda")

# Wrap with CUDA Graphs
graph_model = torch.cuda.make_graphed_callables(
    [model],  # Module list
    sample_args=[(static_input, seq_start)],  # Must match actual input structure
    # memory_pool=torch.cuda.graphs.graph_pool_handle(),
    allow_unused_input=False,
)


# 3. Warmup and execution
def run_inference(x, seq_start):
    # Inputs must match sample_args' device/type/shape
    x = x.to("cuda", non_blocking=True).requires_grad_(False)
    seq_start = seq_start.to("cuda", non_blocking=True)

    with torch.cuda.amp.autocast():
        return graph_model(x, seq_start)


# Warm-up (essential for TE's kernel auto-tuner)
for _ in range(3):
    _ = run_inference(static_input, seq_start)
torch.cuda.synchronize()


# 4. Usage with dynamic sequence lengths
def process_batch(inputs, start_pos):
    # inputs: (batch, seq) on CPU
    inputs_gpu = inputs.to("cuda", non_blocking=True)

    # Output shares memory with pre-allocated buffers
    return run_inference(inputs_gpu, start_pos)
