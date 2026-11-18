# Overlapping Communication with GEMM in TransformerEngine Modules

## Requirements

- Tensor-parallel GPUs must be on a single node, and connected over NVLink/NVSwitch.
- `CUDA_DEVICE_MAX_CONNECTIONS=1` must be enabled in the environment.
- For best performance, point-to-point communication via _CUDA Multicast_ needs CUDA Toolkit 12.0+
  and CUDA driver 535+ on devices with compute capability 9.0 or newer.
- Devices older than compute capability 9.0 require `UB_SKIPMC=1` in the environment in order to fall
  back on a less performant implementation based on CUDA Inter-Process Communication (IPC) handles.

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

## Enabling overlap in your own model

The overlap is set up in two steps: a communicator is created once per process, and each TE layer
is told which of its communications may use it. Both are needed, and both fail loudly rather than
falling back: a layer with the overlap flags on and no communicator raises
`RuntimeError: UB manager is not initialized.` on its first forward, and a communicator created with
a quantization mode the layer does not ask for raises `KeyError: UB for <gemm> with use_fp8=... is
not registered.` (`transformer_engine/pytorch/module/base.py:636-646`).

### 1. Create the Userbuffers communicator

Call `initialize_ub` once per process, **before the first forward pass**, then `destroy_ub` when
training ends. Layers may be constructed at any point before that first forward — the communicator is
looked up in forward, not in the constructor.

```python
import torch
import transformer_engine.pytorch as te

# Tensor-parallel ranks must be contiguous in the world group: initialize_ub splits the world
# into consecutive chunks of tp_size and does not take a process group argument.
shape = [batch_size * seq_length, hidden_size]   # global input shape of a te.TransformerLayer
te.initialize_ub(
    shape,
    tp_size,
    quantization_modes=[te.UserBufferQuantizationMode.FP8 if fp8 else te.UserBufferQuantizationMode.NONE],
    dtype=torch.bfloat16,
    bootstrap_backend="nccl",
)

model = ...                    # build or load the model here

...

te.destroy_ub()                # when training is finished
```

* `shape` is `(sequence_length * batch_size, hidden_size)` with the batch and sequence dimensions
  collapsed together — the same shape a `te.TransformerLayer` receives in its forward pass. The
  buffer is **not resized later and not partially used**: the copy into it checks
  `ubuf.numel() == source.numel()` and fails with
  `Tried to copy an invalid tensor into a Userbuffers buffer` for a smaller batch as well as a
  larger one (`transformer_engine/common/comm_gemm_overlap/comm_gemm_overlap.cpp:904-908`). A step
  with a different `batch_size * sequence_length` therefore needs its own `initialize_ub` call (after
  `destroy_ub`), or pad to a fixed shape.
* `quantization_modes` decides what the buffer holds: `NONE` for BF16/FP16, `FP8` when the layer GEMMs
  run in FP8. The layer looks its buffer up by `(gemm name, quantization mode)`, so a wrong mode here
  surfaces as the `KeyError` quoted above instead of silently running without overlap.
* `initialize_ub` takes no process group. It creates its own world group with the bootstrap backend,
  reads the world size, and assigns rank `r` to domain `r // tp_size`, i.e. TP ranks must be
  **consecutive in the world group** (`base.py:362-376`). If your launch puts a data-parallel rank
  between two tensor-parallel ranks, the communicator will span the wrong ranks.
* `bootstrap_backend` accepts `nccl`, `gloo` and `mpi`. It is **not** `nccl` by default: when omitted,
  the code prefers MPI, then Gloo, then NCCL, depending on what PyTorch was built with
  (`base.py:344-352`).
* Set `CUDA_DEVICE_MAX_CONNECTIONS=1` before launching, and `UB_SKIPMC=1` on devices older than
  compute capability 9.0 — this is what the example script does at import time based on
  `te.cpp_extensions.device_supports_multicast()`.

### 2. Choose which communications overlap

`te.TransformerLayer` is the only module that takes a master switch, `ub_tp_comm_overlap`
(default `False`). The per-communication flags below default to `True` there, and each is
**ANDed** with the master switch (`transformer_engine/pytorch/transformer.py:373-377`), so the master
alone turns on the defaults and an individual flag can only take effect while the master is on.

The flag names describe which GEMM runs *while* the communication happens, so they read backwards
until you see the pairing — `ub_bulk_wgrad`, for example, does not communicate weight gradients.
Each flag is also gated by the layer's parallel mode, so a flag you set can be silently inert:

| argument | default in `TransformerLayer` | communication it overlaps, with what, and when it is active |
|---|---|---|
| `ub_overlap_ag` | `True` | all-gather of `grad_output` with the dgrad GEMM in backward (`linear.py:1193-1196`); in the MLP the all-gather uses `fc2_dgrad`. Row-parallel layers with `sequence_parallel` (`linear.py:2107-2109`) |
| `ub_overlap_rs` | `True` | row-parallel output reduce-scatter in forward; row-parallel + `sequence_parallel` (`linear.py:2104-2106`) |
| `ub_overlap_rs_dgrad` | `False` | reduce-scatter of the input gradient with the dgrad GEMM (`linear.py:1197-1200`); **column**-parallel + `sequence_parallel` (`linear.py:2085-2087`) |
| `ub_bulk_dgrad` | `True` | all-gather of the saved input (`inputmat`, `ln_out` in the MLP) with the **dgrad** GEMM (`linear.py:1202-1206`, `layernorm_mlp.py:1766-1769`); column-parallel + `sequence_parallel`, and dropped when `ub_overlap_rs_dgrad` is set (`linear.py:2089-2096`, `layernorm_mlp.py:2377-2378`) |
| `ub_bulk_wgrad` | `True` | reduce-scatter of the **dgrad** with the **wgrad** GEMM (`linear.py:1207-1210`, `layernorm_mlp.py:1770-1773`); same gating as `ub_bulk_dgrad` |

`ub_overlap_ag` and `ub_overlap_rs_dgrad` are not alternatives — they are active in different parallel
modes (row versus column) and in `LayerNormMLP` they use different buffers (`fc2_dgrad` for the
all-gather, `fc1_dgrad` for the reduce-scatter), so both can stay enabled. The only exclusion is
between `ub_overlap_rs_dgrad` and the two bulk flags, which the constructor turns off.

```python
layer = te.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads,
                            tp_group=tp_group, tp_size=tp_size,
                            ub_tp_comm_overlap=True)
```

`te.Linear`, `te.LayerNormLinear` and `te.LayerNormMLP` do **not** have `ub_tp_comm_overlap`; they
take the five flags directly, with `False` defaults, so each has to be switched on individually
(`transformer_engine/pytorch/module/layernorm_mlp.py:2310-2316`):

```python
layer = te.LayerNormMLP(hidden_size, ffn_hidden_size, tp_size=tp_size, tp_group=tp_group,
                        ub_overlap_ag=True, ub_overlap_rs=True,
                        ub_bulk_dgrad=True, ub_bulk_wgrad=True)
```

Because the flags are ANDed with layer-level preconditions (`parallel_mode`, `sequence_parallel`,
weight-requires-grad), a flag that is set on a layer that does not meet them is silently inert — the
`self.ub_*` attributes the constructor computes are the ones worth checking in a debugger
(`layernorm_mlp.py:2377-2390`).

`initialize_ub` also accepts `ub_cfgs` to pick the algorithm per GEMM (`ring_exchange`, `pipeline`,
`bulk`, `external`) — see the `ub_cfgs` documentation for the per-GEMM keys.

### 3. Hugging Face and other model-replacement workflows

There is no Hugging Face-specific example for the overlap; the integration is the same two steps
around whatever replacement you already use to put TE layers into the model:

1. Give the replacement layers `tp_group` and `tp_size` — a layer without a tensor-parallel group
   has nothing to overlap — and switch on the overlap flags (master switch for `TransformerLayer`,
   individual flags for the others).
2. Make sure `initialize_ub` has run in each process before the first forward, with a `shape` that
   matches that step's `batch_size * sequence_length` exactly. Construction order does not matter;
   the forward is what looks the communicator up.

Three things to check first when the overlap appears to do nothing or crashes on the first step: a
`shape` that does not equal the step's `batch_size * sequence_length` (the copy check above), a
`quantization_modes` entry that does not match the precision the layer GEMMs run in, and TP ranks
that are not consecutive in the world group.
