..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _moe-putting-it-together:

Building an MoE layer
===================================

The building blocks assemble into the four-stage MoE layer from the
:doc:`introduction <../introduction/introduction>`: route, dispatch, run the
experts, and combine. The example below wires them together for top-k routing.

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: moe_layer_pytorch.py
         :language: python
         :start-after: # START_MOE_LAYER_PYTORCH
         :end-before: # END_MOE_LAYER_PYTORCH

   .. tab:: JAX

      .. literalinclude:: moe_layer_jax.py
         :language: python
         :start-after: # START_MOE_LAYER_JAX
         :end-before: # END_MOE_LAYER_JAX

This uses dropless routing (``num_out_tokens = num_tokens * top_k``), so the
dispatch buffer is sized statically rather than from a device-to-host sync. The
expert step is the grouped MLP from :doc:`Grouped GEMM
<../grouped_gemm/grouped_gemm>`; a full expert MLP stacks two grouped GEMMs
around an activation. Every stage is differentiable, so the assembled layer
trains end to end.

When the experts are sharded across devices, the dispatch and combine steps
become collectives; see :doc:`Expert parallelism
<../expert_parallelism/expert_parallelism>`.
