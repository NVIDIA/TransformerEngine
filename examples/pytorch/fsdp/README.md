# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Basic Example for Using PyTorch Fully Sharded Data Parallel mode with Transformer Engine

```bash
# FSDP without deferred initialization:
#     Duplicate modules initialized on each device. Load on device memory reduced only after
#     torch.distributed.fsdp.FullyShardedDataParallel mode shards model parameters.
$ torchrun --standalone --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) fsdp.py
# Sample output on 8xL40S:
#    [GPU-0] WORLD_SIZE = 8
#    [GPU-0] TransformerEngine Model:
#    TransformerLayer(
#    (self_attention): MultiheadAttention(
#        (layernorm_qkv): LayerNormLinear()
#        (core_attention): DotProductAttention(
#        (flash_attention): FlashAttention()
#        (fused_attention): FusedAttention()
#        (unfused_attention): UnfusedDotProductAttention(
#            (scale_mask_softmax): FusedScaleMaskSoftmax()
#            (attention_dropout): Dropout(p=0.1, inplace=False)
#        )
#        )
#        (proj): Linear()
#    )
#    (layernorm_mlp): LayerNormMLP()
#    )
#    [GPU-0] Pre-FSDP memory use = 83.935232MiB
#    [GPU-0] Post-FSDP memory use = 10.491904MiB
#    [GPU-0] Iter. 1
#    [GPU-0] Iter. 2
#    [GPU-0] Iter. 3
#    [GPU-0] Training Time: 6.647654296875s
#    [GPU-0] Avg. Iter. Time: 2.2158847656250003s
#    [GPU-0] Peak memory use = 3000MiB

# FSDP with deferred initialization:
#    Modules initialized with empty parameters via `device='meta'` option. Zero load on device
#    memory until torch.distributed.fsdp.FullyShardedDataParallel mode triggers a reset on
#    on already sharded model parameters.
$ torchrun --standalone --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) fsdp.py --defer-init
# Sample output on 8xL40S:
#    [GPU-0] WORLD_SIZE = 8
#    ...
#    [GPU-0] Pre-FSDP memory use = 0.0MiB
#    [GPU-0] Post-FSDP memory use = 10.491904MiB
#    ...
```

**NOTE:** This example has `fp8_autocast()` enabled by default. To run on GPUs without Fp8 support
(e.g.: A100), add the `--no-fp8` option to the commands shown above.
