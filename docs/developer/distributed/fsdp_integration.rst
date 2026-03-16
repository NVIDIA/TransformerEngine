..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _fsdp-integration:

FSDP Integration
================

Transformer Engine integrates with PyTorch's Fully Sharded Data Parallelism (FSDP) to
enable combined data and tensor parallelism with FP8 quantization.

Challenges
----------

FSDP shards model parameters across data-parallel ranks and gathers them on-demand
during forward/backward. This creates challenges for FP8:

1. **FP8 weight storage**: Weights may be stored in FP8 for memory savings, but FSDP's
   gather/scatter operates on raw tensors — it doesn't know about FP8 scales.
2. **Amax synchronization**: Delayed scaling requires amax values to be consistent across
   FSDP ranks (all ranks must agree on the scale).
3. **Quantized gradients**: FSDP's gradient reduction must handle FP8 gradients correctly.

Solution: FP8-Aware Hooks
--------------------------

TE registers FSDP hooks that handle FP8 tensor conversions:

**Pre-gather hook**: Before FSDP gathers a parameter shard, convert FP8 storage to a
format that can be gathered (e.g., pack data + scale into a single buffer).

**Post-gather hook**: After FSDP gathers the full parameter, reconstruct the FP8 tensor
with proper scales.

**Pre-scatter hook**: Before FSDP scatters gradients, ensure FP8 gradient data and scales
are properly packed.

Usage
-----

.. code-block:: python

   import torch.distributed.fsdp as fsdp
   import transformer_engine.pytorch as te

   model = te.TransformerLayer(...)

   # Wrap with FSDP — TE hooks are registered automatically
   model = fsdp.FullyShardedDataParallel(
       model,
       auto_wrap_policy=...,
   )

   # FP8 training works normally
   with te.fp8_autocast(enabled=True):
       output = model(input)

FP8 Weight Storage with FSDP
-----------------------------

When FP8 weights are enabled, the parameter is stored in FP8 format even when sharded
by FSDP. The lifecycle:

1. **Initialization**: Weight is created in high precision, then optionally cast to FP8.
2. **FSDP shard**: The FP8 data (and associated scale) is sharded across ranks.
3. **FSDP gather**: Full FP8 weight is gathered; TE reconstructs the quantized tensor.
4. **Forward/backward**: GEMM operates on the gathered FP8 weight.
5. **FSDP scatter**: Updated FP8 weight shard is stored locally.

Limitations
-----------

- FSDP2 (``torch.distributed._composable.fsdp``) has better support than FSDP1.
- FP8 weight caching interacts with FSDP gather/scatter — weights may be re-quantized
  on each gather.
- Mixed FSDP + TP configurations require careful process group setup.

See Also
--------

- :doc:`tensor_parallel` — TP can be combined with FSDP
- :doc:`/developer/quantization/class_hierarchy` — FP8 tensor types that FSDP must handle
