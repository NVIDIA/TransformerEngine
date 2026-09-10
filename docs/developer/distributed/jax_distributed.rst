..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX Distributed
===============

JAX uses a fundamentally different approach to distributed training than PyTorch.
Instead of explicit collective calls, JAX relies on XLA's SPMD (Single Program Multiple
Data) partitioner to automatically insert communication based on tensor sharding
annotations.

Sharding Model
--------------

In JAX + Transformer Engine:

- A ``Mesh`` defines the physical device topology (e.g., 2×4 grid of GPUs).
- ``PartitionSpec`` annotations on tensors specify how each dimension maps to mesh axes.
- XLA's compiler automatically inserts all-gather, reduce-scatter, etc.

.. code-block:: python

   from jax.sharding import Mesh, PartitionSpec as P

   # 2D mesh: 2 data-parallel × 4 tensor-parallel
   mesh = Mesh(devices, axis_names=("dp", "tp"))

   # Weight sharded along TP axis
   weight_spec = P(None, "tp")  # [hidden, hidden/tp]

MeshResource
------------

**Location**: ``transformer_engine/jax/sharding.py``

TE's ``MeshResource`` wraps JAX's mesh configuration for use with TE modules:

.. code-block:: python

   from transformer_engine.jax.sharding import MeshResource

   mesh_resource = MeshResource(
       dp_resource="dp",
       tp_resource="tp",
       pp_resource="pp",  # optional pipeline parallel
   )

This tells TE modules which mesh axes correspond to data, tensor, and pipeline
parallelism.

Comparison with PyTorch TP
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Aspect
     - PyTorch
     - JAX
   * - Communication
     - Explicit (``allreduce()``, etc.)
     - Implicit (XLA compiler inserts)
   * - Weight sharding
     - Manual (``parallel_mode``)
     - Via ``PartitionSpec``
   * - Process groups
     - ``torch.distributed`` groups
     - ``jax.sharding.Mesh`` axes
   * - FSDP
     - ``torch.distributed.fsdp``
     - XLA SPMD (automatic)
   * - Configuration
     - Module constructor params
     - ``MeshResource`` + partition specs

TP in JAX TE Modules
---------------------

JAX TE Flax modules accept sharding annotations:

.. code-block:: python

   from transformer_engine.jax.flax import TransformerLayer

   layer = TransformerLayer(
       hidden_size=4096,
       mlp_hidden_size=16384,
       num_attention_heads=32,
       mesh_resource=mesh_resource,
   )

The module uses the mesh resource to:

1. Shard weight parameters according to column/row parallel patterns.
2. Annotate intermediate tensors for XLA to optimize communication.

See Also
--------

- :doc:`tensor_parallel` — PyTorch TP concepts (same mathematical operations)
- :doc:`/developer/jax_frontend/sharding` — Detailed JAX sharding utilities
