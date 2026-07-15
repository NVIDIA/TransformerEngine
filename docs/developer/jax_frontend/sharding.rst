..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Sharding
========

The JAX frontend uses XLA's SPMD partitioner for distributed training. Transformer
Engine provides utilities to configure sharding for TE modules.

MeshResource
------------

**Location**: ``transformer_engine/jax/sharding.py``

``MeshResource`` maps logical parallelism dimensions to physical mesh axes:

.. code-block:: python

   from transformer_engine.jax.sharding import MeshResource

   mesh_resource = MeshResource(
       dp_resource="data",      # Data parallel axis name
       tp_resource="model",     # Tensor parallel axis name
       tpsp_resource="model",   # Tensor + sequence parallel axis name (optional)
       fsdp_resource="fsdp",    # FSDP axis name (optional)
       pp_resource="pipe",      # Pipeline parallel axis name (optional)
       cp_resource="context",   # Context parallel axis name (optional)
   )

This is passed to TE modules, which use it to determine how to partition weights and
activations. The ``global_shard_guard(resource)`` context manager sets the active
``MeshResource`` globally.

Logical Axis Annotations
-------------------------

TE modules annotate their tensors with logical axes (e.g., "hidden", "heads",
"sequence"). These are mapped to physical mesh axes via ``MeshResource``:

.. code-block:: text

   Logical axis "hidden"   → physical axis "model" (TP)
   Logical axis "batch"    → physical axis "data" (DP)
   Logical axis "sequence" → may be split across CP or SP

The mapping is defined by TE's sharding rules registered with each XLA primitive
(see :doc:`xla_ffi_primitives`).

Weight Sharding
---------------

For tensor parallelism, weights are sharded along the appropriate dimension:

**Column-parallel** (QKV projection, MLP FC1):

.. code-block:: python

   # Weight shape: [hidden, heads * head_dim]
   # Sharded: [hidden, heads * head_dim / tp]
   # PartitionSpec: P(None, "tp")

**Row-parallel** (output projection, MLP FC2):

.. code-block:: python

   # Weight shape: [heads * head_dim, hidden]
   # Sharded: [heads * head_dim / tp, hidden]
   # PartitionSpec: P("tp", None)

XLA automatically inserts all-gather and reduce-scatter operations based on these
partition specs.

Activation Sharding
-------------------

Activations are annotated to minimize communication:

- **Before column-parallel**: Full tensor, replicated across TP.
- **After column-parallel**: Sharded along hidden dim.
- **After row-parallel**: Full tensor (XLA inserts all-reduce).

With sequence parallelism, activations outside the parallel region are sharded along the
sequence dimension.

Custom Sharding Rules
---------------------

Each TE XLA primitive registers sharding rules via ``infer_sharding_from_operands()``
and ``partition()`` methods on the ``BasePrimitive`` class. These tell XLA:

1. Given input sharding, what is the output sharding?
2. How to partition the operation across devices.

See Also
--------

- :doc:`/developer/distributed/jax_distributed` — Overview of JAX distributed training
- :doc:`xla_ffi_primitives` — Where sharding rules are registered
