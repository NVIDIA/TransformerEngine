..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Expert parallelism
===================================

.. note::

    NCCL-based expert parallelism requires Hopper (SM90) or later and NCCL 2.30.4
    or newer. It is compiled in by default when Transformer Engine is built for
    these architectures; set ``NVTE_WITH_NCCL_EP=0`` at build time to disable it.

The grouped GEMM keeps all experts on a single device. When the experts no longer
fit there - or to add another dimension of parallelism - they are sharded across
devices, a scheme called expert parallelism (EP). Each device then owns only a
slice of the experts, so a token routed to a non-local expert has to travel to
the device that owns it.

That data movement is two all-to-all collectives wrapped around the local expert
computation: a **dispatch** all-to-all sends each token to the rank that owns its
expert, the local grouped GEMM runs, and a **combine** all-to-all returns the
results to the source rank. It is the distributed counterpart of the
:doc:`token dispatch and token combine kernels <../routing_kernels/routing_kernels>`.

.. raw:: html
   :file: img/moe_expert_parallel.svg

*Figure 1. With experts sharded across ranks, a dispatch all-to-all routes each
token to the rank owning its expert and a combine all-to-all returns the
outputs to the source rank.*

Transformer Engine implements dispatch and combine directly on NCCL, using
NCCL symmetric-memory windows for zero-copy transfers. The backend is a common
C API (``nvte_ep_dispatch`` / ``nvte_ep_combine`` and their backward passes,
declared in ``transformer_engine/common/include/transformer_engine/ep.h``) that
both frameworks build on:

* **PyTorch.** ``transformer_engine.pytorch.ep`` exposes the primitives with
  autograd support. ``ep_bootstrap`` initializes EP once per process on an
  existing process group, an ``EpBuffer`` holds the per-call state, and
  ``ep_dispatch`` / ``ep_combine`` perform the two all-to-alls. The routing
  itself comes from the :doc:`router <../router/router>`; the local experts run
  on the receive buffer between the two calls.
* **JAX.** ``transformer_engine.jax.moe.moe`` runs the entire layer - router,
  dispatch, grouped expert GEMMs, and combine - as a single differentiable call.
  ``ep_axis`` names the mesh axis the experts are sharded over, and the dispatch
  and combine steps become all-to-all collectives over that axis. The underlying
  primitives are also available separately in ``transformer_engine.jax.ep``.
  This API is currently experimental.

.. tabs::

   .. tab:: PyTorch

      .. raw:: html

         <div class="code-block-header">
            Requires SM90 (Hopper) or later
         </div>

      .. literalinclude:: moe_expert_parallel_pytorch.py
         :language: python
         :start-after: # START_MOE_EXPERT_PARALLEL_PYTORCH
         :end-before: # END_MOE_EXPERT_PARALLEL_PYTORCH

   .. tab:: JAX

      .. raw:: html

         <div class="code-block-header">
            Requires SM90 (Hopper) or later
         </div>

      .. literalinclude:: moe_expert_parallel_jax.py
         :language: python
         :start-after: # START_MOE_EXPERT_PARALLEL_JAX
         :end-before: # END_MOE_EXPERT_PARALLEL_JAX

Sizing the receive buffer
-------------------------

Each rank receives a data-dependent number of tokens per step. Passing
``recv_capacity_per_rank`` fixes the size of the receive buffer up front, so the
step needs no device-to-host synchronization and can be captured in a CUDA graph;
the dropless worst case is ``ep_size * max_tokens_per_rank * top_k``. Omitting it
selects eager mode, which sizes the buffer from the actual receive count each
step at the cost of a host sync.

In PyTorch, ``ep_dispatch`` can quantize the tokens on the fly when the
``EpBuffer`` is created with an MXFP8 ``dispatch_fwd_quant_recipe``, so the
all-to-all moves the low-precision payload and the local grouped GEMM consumes
it directly. Complete runnable examples live in ``examples/pytorch/ep/`` and
``examples/jax/ep/`` in the repository.
