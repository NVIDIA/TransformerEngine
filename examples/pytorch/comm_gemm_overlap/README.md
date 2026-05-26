# Overlapping Communication with GEMM in TransformerEngine Modules

## Requirements

- Tensor-parallel GPUs must be on a single node, and connected over NVLink/NVSwitch.
- `CUDA_DEVICE_MAX_CONNECTIONS=1` must be enabled in the environment.
- For best performance, point-to-point communication via _CUDA Multicast_ needs CUDA Toolkit 12.0+
  and CUDA driver 535+ on devices with compute capability 9.0 or newer.
- Devices older than compute capability 9.0 require `UB_SKIPMC=1` in the environment in order to fall
  back on a less performant implementation based on CUDA Inter-Process Communication (IPC) handles.

## Enabling overlap in your own module

The example follows the same setup sequence that user code should use:

1. Set `CUDA_DEVICE_MAX_CONNECTIONS=1` before creating the layer.
2. Initialize `torch.distributed` and create the tensor-parallel process group.
3. Call `te.module.base.initialize_ub(...)` with the local activation shape and tensor-parallel
   size before constructing TE layers with userbuffer overlap enabled.
4. Pass the tensor-parallel group, tensor-parallel size, and overlap flags to the TE layer.
5. Call `te.module.base.destroy_ub()` before shutting down the process group.

Minimal setup sketch:

```python
import os
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

dist.init_process_group(backend="nccl")
tp_group = dist.group.WORLD
tp_size = dist.get_world_size(tp_group)

hidden_size = num_heads * head_dim
batched_size = seq_length * batch_size

te.module.base.initialize_ub(
    [batched_size, hidden_size],
    tp_size,
    dtype=torch.bfloat16,
    bootstrap_backend="nccl",
)

layer = te.TransformerLayer(
    hidden_size,
    4 * hidden_size,
    num_heads,
    tp_group=tp_group,
    tp_size=tp_size,
    sequence_parallel=True,
    fuse_qkv_params=True,
    ub_tp_comm_overlap=True,
    ub_overlap_ag=True,
    ub_overlap_rs=True,
    ub_bulk_wgrad=True,
    ub_bulk_dgrad=True,
    seq_length=seq_length,
)

# ... run forward/backward/optimizer steps ...

te.module.base.destroy_ub()
```

`ub_tp_comm_overlap` is the top-level gate on `TransformerLayer`: when it is `False`, the
layer disables the individual userbuffer overlap paths even if the per-path flags are `True`.
For lower-level layers such as `Linear`, `LayerNormLinear`, `LayerNormMLP`, or
`MultiheadAttention`, enable the relevant per-path flags directly (for example
`ub_overlap_ag`, `ub_overlap_rs`, `ub_bulk_wgrad`, and `ub_bulk_dgrad`) and set the `ub_name`
where the layer requires one.

When replacing modules in a Hugging Face model, run the userbuffer initialization once before
constructing the replacement TE modules. The replacement modules need the same tensor-parallel
group, tensor-parallel size, sequence-parallel setting, and overlap flags shown above; the
activation shape passed to `initialize_ub` should match the sequence length, micro-batch size,
and hidden size used by the replaced blocks.

## Examples

### Single node, tensor-parallel LayerNormMLP:

Forward and backward passes with layer weights distributed over all GPUs in a single node.

```bash
$ torchrun --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) te_layer_with_overlap.py

# Sample output on 8x H100s:
#   [rank0:node0] |-- Created tensor-parallel group: [0, 1, 2, 3, 4, 5, 6, 7]
#   !!! [UB] Create UbufP2PCommOverlap Communicator
#   UB_TIMEOUT is set to 110 sec, 217800000000 cycles, freq: 1980000khz
#   MC initialized successfully, window size = 549755813888
#   !!! [UBP2P] Register UBuf 1
#   !!! [UBP2P] Register UBuf 2
#   !!! [UBP2P] Register UBuf 3
#   !!! [UBP2P] Register UBuf 4
#   !!! [UB] Register UBuf 5
#   !!! [UBP2P] Register UBuf 6
#   !!! [UB] Register UBuf 7
#   !!! [UB] Register UBuf 8
#   !!! [UBP2P] Register UBuf 9
#   !!! [UB] Register UBuf 10
#   [rank0:node0] Iter 1
#   [rank0:node0] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
#   [rank0:node0] Iter 2
#   [rank0:node0] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
#   [rank0:node0] Iter 3
#   [rank0:node0] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
#   [rank0:node0] Iter 4
#   [rank0:node0] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
#   [rank0:node0] Iter 5
#   [rank0:node0] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
```
### Single node, mixed data- and tensor-parallel LayerNormMLP:

Uses `torch.nn.parallel.DistributedDataParallel` for replicating the model across 2 tensor-parallel
groups in a single node.

```bash
$ torchrun --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) te_layer_with_overlap.py --num-replicas 2

# Sample output on 8x H100s:
#   [rank0:node0] |-- Created tensor-parallel group: [0, 1, 2, 3]
#   [rank4:node1] |-- Created tensor-parallel group: [4, 5, 6, 7]
#   [rank0:node0] |-- Created data-parallel group: [0, 4]
#   [rank3:node1] |-- Created data-parallel group: [3, 7]
#   [rank1:node1] |-- Created data-parallel group: [1, 5]
#   [rank2:node0] |-- Created data-parallel group: [2, 6]
#   !!! [UB] Create UbufP2PCommOverlap Communicator
#   UB_TIMEOUT is set to 110 sec, 217800000000 cycles, freq: 1980000khz
#   MC initialized successfully, window size = 549755813888
#   !!! [UBP2P] Register UBuf 1
#   !!! [UBP2P] Register UBuf 2
#   !!! [UBP2P] Register UBuf 3
#   !!! [UBP2P] Register UBuf 4
#   !!! [UB] Register UBuf 5
#   !!! [UBP2P] Register UBuf 6
#   !!! [UB] Register UBuf 7
#   !!! [UB] Register UBuf 8
#   !!! [UBP2P] Register UBuf 9
#   !!! [UB] Register UBuf 10
#   [rank4:node1] Iter 1
#   [rank0:node0] Iter 1
#   [rank0:node0] |-- Generate random input batch
#   [rank4:node1] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank4:node1] |-- Forward pass
#   [rank4:node1] |-- Compute loss
#   [rank0:node0] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank4:node1] |-- Backward pass
#   [rank4:node1] |-- Optimizer step
#   [rank0:node0] |-- Optimizer step
#   [rank4:node1] Iter 2
#   [rank0:node0] Iter 2
#   [rank0:node0] |-- Generate random input batch
#   [rank4:node1] |-- Generate random input batch
#   [rank4:node1] |-- Forward pass
#   [rank0:node0] |-- Forward pass
#   [rank4:node1] |-- Compute loss
#   [rank0:node0] |-- Compute loss
#   [rank4:node1] |-- Backward pass
#   [rank0:node0] |-- Backward pass
#   [rank4:node1] |-- Optimizer step
#   [rank0:node0] |-- Optimizer step
#   [rank4:node1] Iter 3
#   [rank0:node0] Iter 3
#   [rank0:node0] |-- Generate random input batch
#   [rank4:node1] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank4:node1] |-- Forward pass
#   [rank4:node1] |-- Compute loss
#   [rank0:node0] |-- Compute loss
#   [rank4:node1] |-- Backward pass
#   [rank0:node0] |-- Backward pass
#   [rank0:node0] |-- Optimizer step
#   [rank4:node1] |-- Optimizer step
#   [rank0:node0] Iter 4
#   [rank4:node1] Iter 4
#   [rank0:node0] |-- Generate random input batch
#   [rank4:node1] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank4:node1] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank4:node1] |-- Compute loss
#   [rank4:node1] |-- Backward pass
#   [rank0:node0] |-- Backward pass
#   [rank4:node1] |-- Optimizer step
#   [rank0:node0] |-- Optimizer step
#   [rank4:node1] Iter 5
#   [rank0:node0] Iter 5
#   [rank0:node0] |-- Generate random input batch
#   [rank4:node1] |-- Generate random input batch
#   [rank0:node0] |-- Forward pass
#   [rank4:node1] |-- Forward pass
#   [rank0:node0] |-- Compute loss
#   [rank4:node1] |-- Compute loss
#   [rank0:node0] |-- Backward pass
#   [rank4:node1] |-- Backward pass
#   [rank4:node1] |-- Optimizer step
#   [rank0:node0] |-- Optimizer step
```

**NOTE:** To run with Fp8 compute on supporting hardware, add the `--fp8` flag to the commands
shown above.
