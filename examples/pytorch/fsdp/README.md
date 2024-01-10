# Basic Example for Using PyTorch Fully Sharded Data Parallel mode with Transformer Engine

```bash
# FSDP without deferred initialization:
#     Duplicate modules initialized on each device. Load on device memory reduced only after
#     torch.distributed.fsdp.FullyShardedDataParallel mode shards model parameters.
$ torchrun --standalone --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) fsdp.py
# Sample output on 8xL40S:
#    [GPU-0] WORLD_SIZE = 8
#    [GPU-0] Pre-FSDP memory use = 8590.196736MiB
#    ...
#    [GPU-0] Post-FSDP memory use = 1073.774592MiB
#    ...

# FSDP with deferred initialization:
#    Modules initialized with empty paramaters via `device='meta'` option. Zero load on device
#    memory until torch.distributed.fsdp.FullyShardedDataParallel mode triggers a reset on
#    on already sharded model parameters.
$ torchrun --standalone --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) fsdp.py --defer-init
# Sample output on 8xL40S:
#    [GPU-0] WORLD_SIZE = 8
#    [GPU-0] Pre-FSDP memory use = 0.0MiB
#    ...
#    [GPU-0] Post-FSDP memory use = 1073.774592MiB
#    ...
```

**NOTE:** This example has `fp8_autocast()` enabled by default. To run on GPUs without Fp8 support
(e.g.: A100), add the `--no-fp8` option to the commands shown above.
