..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _fsdp-integration:

FSDP Integration
================

Transformer Engine integrates with PyTorch's Fully Sharded Data Parallelism (FSDP) to
enable combined data and tensor parallelism with FP8 quantization. TE supports both
**FSDP1** (the wrapper-based ``FullyShardedDataParallel``) and **FSDP2** (the composable
``fully_shard`` API), using different mechanisms for each.

Challenges
----------

FSDP shards model parameters across data-parallel ranks and gathers them on-demand
during forward/backward. This creates two challenges for quantized training:

1. **Quantized weight all-gather**: FSDP gathers weight shards into full parameters
   before each forward/backward pass. When weights are stored in FP8, FSDP must gather
   the quantized data *and* its associated scales, then reconstruct a valid quantized
   tensor on the other side.

2. **Activation sharding**: Activations saved for backward consume memory proportional to
   model size. Sharding them across FSDP ranks reduces per-rank memory, but requires
   handling quantized tensor types during scatter/gather.

These two challenges are handled by separate mechanisms, described below.

FSDP2: Quantized Weight All-Gather
------------------------------------

**FSDP2** (``torch.distributed._composable.fsdp.fully_shard``) is the preferred approach.
It uses PyTorch's **tensor subclass protocol** — FSDP2 automatically discovers and calls
``fsdp_pre_all_gather()`` and ``fsdp_post_all_gather()`` methods on tensor subclasses
during weight all-gather. No explicit setup or registration is needed.

**Location**: Methods on ``QuantizedTensor`` subclasses in ``transformer_engine/pytorch/tensor/``.

``fsdp_pre_all_gather(self, mesh, orig_size, contiguous_orig_stride, module, mp_policy)``
   Called **before** FSDP all-gathers a sharded weight parameter. Extracts the raw
   quantized data (and optionally scale_inv tensors) into plain ``torch.Tensor`` objects
   that FSDP can all-gather normally. Returns a tuple of tensors to gather and a metadata
   tuple (dtype, usage flags, scales) that will be passed to ``fsdp_post_all_gather``.

   The method determines which layouts to include based on training state: during the
   forward pass it sends rowwise data; during backward it sends columnwise data (or both,
   depending on the ``reshard_after_forward`` policy).

``fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None)``
   Called **after** FSDP has gathered the full tensors. Reconstructs the ``QuantizedTensor``
   subclass from the gathered data and the metadata. On the first call, creates a new
   tensor; on subsequent calls, reuses the ``out`` tensor (passed by FSDP2) and updates
   it in place.

**Supported tensor types**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tensor Class
     - Notes
   * - ``Float8Tensor``
     - Gathers uint8 data + scale_inv. Handles amax reduction for current scaling.
   * - ``MXFP8Tensor``
     - Gathers data + scale_inv with padding/unpadding for block-aligned scales.

.. note::

   FSDP2 also requires that certain ``torch.ops.aten`` operations preserve the tensor
   subclass (e.g., ``slice``, ``view``, ``copy_``). ``Float8Tensor`` defines
   ``_ops_to_preserve_subclass_in_fsdp2`` to list these operations.

**Usage**:

.. code-block:: python

   from torch.distributed._composable.fsdp import fully_shard
   import transformer_engine.pytorch as te

   model = te.TransformerLayer(...)

   # FSDP2 automatically handles quantized weight all-gather
   fully_shard(model)

   with te.fp8_autocast(enabled=True):
       output = model(input)  # Weights gathered/scattered automatically

FSDP1: Activation Scatter/Gather
-----------------------------------

**FSDP1** uses a different mechanism for a different purpose: sharding *activations* (not
weights) saved for backward. This is implemented via direct function calls within the
``_Linear`` autograd function.

**Location**: ``transformer_engine/pytorch/distributed.py``

``_fsdp_scatter_tensors(fsdp_group, *tensors)``
   Called in ``_Linear.forward()`` after the forward GEMM. Shards saved activations
   (quantized input and optionally quantized weight) across FSDP ranks using
   ``split_tensor_into_1d_equal_chunks()``. This reduces per-rank activation memory
   proportionally to the FSDP world size. Returns the original shapes for reconstruction.

``_fsdp_gather_tensors(fsdp_group, shapes, *tensors)``
   Called in ``_Linear.backward()`` before the dgrad/wgrad GEMMs. Reconstructs full
   activations from shards using ``gather_split_1d_tensor()`` and reshapes to original
   dimensions.

``prepare_te_modules_for_fsdp(fsdp_root)``
   Must be called **after** wrapping with FSDP1. Injects the FSDP process group reference
   into each TE module so that ``_fsdp_scatter_tensors`` / ``_fsdp_gather_tensors`` know
   which group to communicate with.

TE also provides a convenience wrapper:

.. code-block:: python

   # TE's FSDP1 wrapper calls prepare_te_modules_for_fsdp automatically
   from transformer_engine.pytorch.distributed import FullyShardedDataParallel

   model = FullyShardedDataParallel(te_model, ...)

**Usage (explicit setup)**:

.. code-block:: python

   from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
   from transformer_engine.pytorch.distributed import prepare_te_modules_for_fsdp

   model = FSDP(te_model, ...)
   prepare_te_modules_for_fsdp(model)

   with te.fp8_autocast(enabled=True):
       output = model(input)

Summary: Which Mechanism Does What
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - FSDP2 (composable)
     - FSDP1 (wrapper)
   * - What it handles
     - Quantized **weight** all-gather
     - **Activation** scatter/gather
   * - Mechanism
     - ``fsdp_pre_all_gather`` / ``fsdp_post_all_gather`` on tensor subclasses
     - ``_fsdp_scatter_tensors`` / ``_fsdp_gather_tensors`` called from ``_Linear``
   * - Setup required
     - None (automatic discovery)
     - ``prepare_te_modules_for_fsdp()``
   * - Called by
     - PyTorch's FSDP2 internals
     - ``_Linear.forward()`` / ``_Linear.backward()``

.. note::

   The activation scatter/gather from FSDP1 (``_fsdp_scatter_tensors`` /
   ``_fsdp_gather_tensors``) is also used in FSDP2 setups — it is called from
   ``_Linear`` regardless of which FSDP version manages the weights. The two mechanisms
   are complementary, not mutually exclusive.

Limitations
-----------

- FSDP sharding is not valid for models initialized with ``primary_weights_in_fp8=True``.
- FP8 weight caching interacts with FSDP gather/scatter — weights may be re-quantized
  on each gather.
- Mixed FSDP + TP configurations require careful process group setup.
- ``fsdp_pre_all_gather`` / ``fsdp_post_all_gather`` are currently implemented only for
  ``Float8Tensor`` and ``MXFP8Tensor``, not for ``Float8BlockwiseQTensor`` or
  ``NVFP4Tensor``.

See Also
--------

- :doc:`tensor_parallel` — TP can be combined with FSDP
- :doc:`/developer/quantization/class_hierarchy` — FP8 tensor types that FSDP must handle
