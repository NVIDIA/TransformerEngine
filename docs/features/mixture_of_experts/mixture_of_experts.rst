..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _moe-overview:

Mixture of Experts
===================================

.. note::

    The MoE building blocks are designed to work with Transformer Engine's
    low-precision recipes. This support is still being extended, so not every
    block works with every recipe yet.

Introduction
------------

A Mixture of Experts (MoE) layer replaces a dense feed-forward network with a set
of expert networks and a router that sends each token to one or more experts.
A token passes through the layer in the following stages:

#. The **router** scores the experts for each token and selects the top-k of
   them.
#. **Token dispatch** gathers the tokens into expert-contiguous order.
#. The **grouped MLP** (the experts) runs a single batched computation over all
   expert blocks.
#. **Token combine** scatters the expert outputs back into the original token
   order, merging the contributions when a token was sent to more than one
   expert.

With expert parallelism the experts are sharded across ranks, and an
**all-to-all dispatch** and **all-to-all combine** take the place of token
dispatch and token combine: the dispatch takes the router output directly and
delivers each rank's tokens already grouped by local expert, and the combine
returns the outputs to the source rank in the original token order.

.. raw:: html
   :file: img/moe_layer_ep.svg

*Figure 1. The stages of an MoE layer on a single device and with expert
parallelism.*

Transformer Engine provides a building block for each stage. They are exposed as
standalone functions, so they can be assembled into a complete MoE layer or
dropped into an existing implementation one piece at a time:

* :ref:`Router <moe-router>`: fused score function and top-k selection, and a
  fused :ref:`load-balancing loss <moe-load-balancing>`.
* :ref:`Token permutation <moe-token-permutation>`: token dispatch and combine
  kernels that move tokens between their original order and the
  expert-contiguous layout.
* :ref:`Grouped GEMM <moe-grouped-gemm>`: the expert linear layers as one call
  over expert-contiguous blocks; the :ref:`grouped MLP <moe-grouped-mlp>` fuses
  the whole expert MLP into one kernel.
* :ref:`Expert parallelism <moe-expert-parallelism>`: all-to-all dispatch and
  combine for experts sharded across devices.

The :ref:`example at the end <moe-putting-it-together>` wires the blocks into a
complete MoE layer.

.. _moe-router:

Router
------

The router decides which experts each token is sent to. It applies a score
function to the gating logits, selects the top-k experts per token, and produces
the two tensors that drive the rest of the layer:

* ``routing_map`` - a ``[num_tokens, num_experts]`` mask marking the selected
  experts. Token dispatch uses it to lay the tokens out by expert.
* ``probs`` - the routing weight of each selected expert. Token combine uses
  these as merging weights when a token was routed to more than one expert.

``fused_topk_with_score_function`` runs the score function and the top-k
selection in a single differentiable kernel. All internal math runs in FP32,
regardless of the logits dtype.

.. raw:: html
   :file: img/moe_router.svg

*Figure 2. The router scores the experts for each token and keeps the top-k.
The selected entries populate* ``routing_map`` *(a 0/1 mask) and* ``probs`` *(the
routing weights); all other entries are zero.*

Options:

* **Score function:** softmax or sigmoid. With softmax, ``use_pre_softmax``
  selects whether the softmax is applied before or after the top-k.
* **Grouped routing:** the experts are split into ``num_groups`` equal groups.
  Each group is scored by the sum of its best expert scores, the top
  ``group_topk`` groups are kept, and the top-k experts are chosen only from
  those groups.
* **Expert bias:** ``expert_bias`` is added to the scores before the top-k
  selection (see :ref:`Load balancing <moe-load-balancing>`).
* **Scaling:** ``scaling_factor`` rescales the returned probabilities.

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: router_pytorch.py
         :language: python
         :start-after: # START_ROUTER_PYTORCH
         :end-before: # END_ROUTER_PYTORCH

   .. tab:: JAX

      .. literalinclude:: router_jax.py
         :language: python
         :start-after: # START_ROUTER_JAX
         :end-before: # END_ROUTER_JAX

.. _moe-load-balancing:

Load balancing
~~~~~~~~~~~~~~

``fused_moe_aux_loss`` computes the auxiliary load-balancing loss that penalizes
uneven token counts across experts. It takes the per-expert token counts and the
*dense* routing scores (one value per expert, not only the selected top-k), so
the loss has a gradient with respect to every expert's logit. The dense scores
are returned by the router functions shown below; add the scaled loss to the
training loss.

``expert_bias`` balances the load without an extra loss term:

* with the sigmoid score function it is added to the scores only for the top-k
  selection, so it changes which experts are picked but not the returned
  routing weights;
* update it between steps: lower it for overloaded experts, raise it for
  under-used ones.

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: router_pytorch.py
         :language: python
         :start-after: # START_ROUTER_AUX_PYTORCH
         :end-before: # END_ROUTER_AUX_PYTORCH

   .. tab:: JAX

      .. literalinclude:: router_jax.py
         :language: python
         :start-after: # START_ROUTER_AUX_JAX
         :end-before: # END_ROUTER_AUX_JAX

.. _moe-token-permutation:

Token permutation
-----------------

Token dispatch moves the tokens into the expert-contiguous layout expected by
the grouped GEMM, and token combine moves the expert outputs back.

* All of these kernels are differentiable.
* The snippets below use the mask-map routing variant. Other variants (for
  example index-map routing) follow the same pattern, see the :doc:`PyTorch API
  reference </api/pytorch>` and :doc:`JAX API reference </api/jax>`.

Token dispatch
~~~~~~~~~~~~~~

Token dispatch takes the token tensor and a routing map describing each
token's destination experts, and returns a permuted token buffer in which all
rows assigned to the same expert are stored contiguously. This is the layout the
grouped GEMM consumes, together with the per-expert token counts.

.. raw:: html
   :file: img/moe_permute.svg

*Figure 3. Token dispatch consumes the input token tensor together with the
routing map and produces an expert-contiguous token tensor; rows assigned to the
same expert are stored back-to-back.*

A typical call looks like:

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: moe_permute_pytorch.py
         :language: python
         :start-after: # START_MOE_PERMUTE_PYTORCH
         :end-before: # END_MOE_PERMUTE_PYTORCH

   .. tab:: JAX

      .. literalinclude:: moe_permute_jax.py
         :language: python
         :start-after: # START_MOE_PERMUTE_JAX
         :end-before: # END_MOE_PERMUTE_JAX

The call returns the permuted token buffer of shape
``[num_out_tokens, hidden_size]`` together with a ``row_id_map`` that token
combine uses to restore the original token order after the experts have run.

Token combine
~~~~~~~~~~~~~

Token combine is the inverse operation: it takes the expert-contiguous output
of the grouped GEMM and the ``row_id_map`` returned by token dispatch, and
returns a tensor of shape ``[num_tokens, hidden_size]`` with the rows written
back into the original token order.

For top-k routing pass the routing weights as ``merging_probs``; the kernel
then computes the weighted sum of the per-expert contributions in the same
fused pass. Without it the contributions are summed unweighted; for top-1
routing it is not needed.

.. raw:: html
   :file: img/moe_unpermute.svg

*Figure 4. Token combine reads the expert-contiguous output tensor and the*
``row_id_map``\ *, and writes each row back to its original token slot. With*
``merging_probs``\ *, contributions from multiple experts to the same token are
combined in the same fused kernel.*

A typical call looks like:

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: moe_unpermute_pytorch.py
         :language: python
         :start-after: # START_MOE_UNPERMUTE_PYTORCH
         :end-before: # END_MOE_UNPERMUTE_PYTORCH

   .. tab:: JAX

      .. literalinclude:: moe_unpermute_jax.py
         :language: python
         :start-after: # START_MOE_UNPERMUTE_JAX
         :end-before: # END_MOE_UNPERMUTE_JAX

Token probabilities
~~~~~~~~~~~~~~~~~~~

The routing weights can be applied in two equivalent places:

* **At combine (output side).** Pass them to token combine as ``merging_probs``,
  as in the examples above.
* **At dispatch (input side).** Pass them to token dispatch as ``probs``. They
  are permuted alongside the tokens into the expert-contiguous layout, so each
  expert's input can be scaled before the grouped GEMM.

Padding and alignment
~~~~~~~~~~~~~~~~~~~~~

Grouped GEMM backends require or prefer each expert's token block to start at
an aligned offset (for example, a multiple of 128 rows). Token dispatch can pad
each block up to a multiple of ``align_size`` in the same kernel.

.. raw:: html
   :file: img/moe_padding.svg

*Figure 5. Each expert's block is rounded up to a multiple of* ``align_size``\ *.
The per-expert padding offsets are returned so that token combine can drop the
padding again.*

The padded dispatch returns the padded token buffer, the aligned per-expert
token counts to pass to the grouped GEMM, and the per-expert ``pad_offsets``
that token combine needs to remove the padding.

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: moe_permute_pad_pytorch.py
         :language: python
         :start-after: # START_MOE_PERMUTE_PAD_PYTORCH
         :end-before: # END_MOE_PERMUTE_PAD_PYTORCH

   .. tab:: JAX

      .. literalinclude:: moe_permute_pad_jax.py
         :language: python
         :start-after: # START_MOE_PERMUTE_PAD_JAX
         :end-before: # END_MOE_PERMUTE_PAD_JAX

Reordering expert chunks
~~~~~~~~~~~~~~~~~~~~~~~~

``sort_chunks_by_index`` reorders whole blocks of rows:

* the input ``[num_tokens, hidden_size]`` is split along the first dimension
  into chunks of the given ``split_sizes``;
* the chunks are concatenated again in the order given by ``sorted_indices``:
  output chunk ``i`` is input chunk ``sorted_indices[i]``, rows inside a chunk
  keep their order;
* the operation is differentiable, and a ``_with_probs`` variant moves a
  per-row probability tensor along with the rows.

The typical use is expert parallelism over a generic all-to-all, where the
received buffer is ordered by source rank and then by expert, while the grouped
GEMM needs all rows of one expert together. With two source ranks and two local
experts:

* received chunks: ``(rank 0, E4)``, ``(rank 0, E5)``, ``(rank 1, E4)``,
  ``(rank 1, E5)``;
* ``sorted_indices = [0, 2, 1, 3]`` regroups them into ``E4, E4, E5, E5`` for
  the grouped GEMM;
* after the experts have run, the inverse permutation restores the rank-major
  order for the combine all-to-all.

.. _moe-grouped-gemm:

Grouped GEMM
------------

The grouped GEMM applies the per-expert linear layers in one call, replacing a
loop of one ``Linear`` call per expert and producing the same outputs.

Let ``G`` be the number of experts. For expert ``i``, ``X_i`` is the routed
token block, ``W_i`` is the expert weight, and ``b_i`` is the optional bias:

.. math::

   Y_i = X_i W_i^T + b_i,\quad i = 0, \ldots, G - 1

The full layer output is the concatenation of all expert outputs:

.. math::

   Y = \mathrm{concat}(Y_0, Y_1, \ldots, Y_{G-1})

The number of token rows belonging to each expert is passed as a per-expert
token-count argument.

.. raw:: html
   :file: img/grouped_linear.svg

*Figure 6. Both paths produce the same outputs from the same inputs. The
baseline launches one* ``Linear`` *per expert; the grouped GEMM replaces the
loop with one call.*

The grouped GEMM works with the :doc:`low-precision training recipes
</features/low_precision_training/index>` available to ``Linear``: the inputs
are quantized per expert and the expert GEMMs run in the recipe's precision.

There are two execution paths:

* **Per-expert GEMMs.** The per-expert token counts are read on the host, the
  input is split and quantized per expert, and one cuBLAS GEMM per expert is
  launched on a pool of CUDA streams (on Hopper, ``NVTE_USE_CUTLASS_GROUPED_GEMM=1``
  switches BF16/FP16 to a CUTLASS grouped GEMM kernel). This path supports all
  recipes, but reading the token counts is a device-to-host synchronization, so
  it cannot be captured in a CUDA graph.
* **Single grouped GEMM.** The token counts stay on the device and all experts
  run as one cuBLASLt grouped GEMM (cuBLAS 13.3 or newer), with the
  quantization fused across experts. There is no host synchronization, so the
  step is CUDA-graph capturable. Supported for BF16/FP16, and for MXFP8 and NVFP4
  on Blackwell; FP8 current scaling and FP8 block scaling on Hopper need cuBLAS
  13.5 / 13.6. FP8 delayed scaling and custom recipes are not supported on this
  path. The snippets show how it is selected.

The snippets assume the tokens have already been permuted into
expert-contiguous order.

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: grouped_linear_pytorch.py
         :language: python
         :start-after: # START_GROUPED_LINEAR_PYTORCH
         :end-before: # END_GROUPED_LINEAR_PYTORCH

   .. tab:: JAX

      .. literalinclude:: grouped_linear_jax.py
         :language: python
         :start-after: # START_GROUPED_LINEAR_JAX
         :end-before: # END_GROUPED_LINEAR_JAX

.. _moe-grouped-mlp:

Grouped MLP
-----------

An expert MLP is two grouped GEMMs with an activation between them. On
Blackwell (SM100) GPUs the whole expert MLP can run as a single CuTe DSL kernel:
the intermediate activation stays on chip and its quantization is folded into
the GEMMs.

.. raw:: html
   :file: img/moe_grouped_mlp.svg

*Figure 7. The operation fuser replaces the two grouped GEMMs and the activation
between them with a single fused grouped-MLP kernel.*

The fusion is applied by the :doc:`operation fuser </examples/op_fuser/op_fuser>`:
a grouped linear, a scaled GLU (or SReLU) activation and another grouped linear
in sequence are replaced with one fused grouped-MLP operation.

.. tabs::

   .. tab:: PyTorch

      .. raw:: html

         <div class="code-block-header">
            Requires SM100 (Blackwell) or later
         </div>

      .. literalinclude:: grouped_mlp_pytorch.py
         :language: python
         :start-after: # START_GROUPED_MLP_PYTORCH
         :end-before: # END_GROUPED_MLP_PYTORCH

The fusion is enabled with ``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`` and requires
Blackwell and a block-scaled recipe (MXFP8 or NVFP4). When the configuration is
not supported, the three operations run separately with identical results.

.. _moe-expert-parallelism:

Expert parallelism
------------------

.. note::

    NCCL-based expert parallelism requires Hopper (SM90) or later and NCCL 2.30.4
    or newer. It is compiled in by default when Transformer Engine is built for
    these architectures; set ``NVTE_WITH_NCCL_EP=0`` at build time to disable it.

With expert parallelism (EP) the experts are sharded across ranks: every rank
keeps its own shard of the tokens and holds only a slice of the experts.

.. raw:: html
   :file: img/moe_expert_placement.svg

*Figure 8. Expert placement: each rank holds its token shard and a subset of the
experts. Tokens t1, t3 and t5 are routed to experts on the other rank.*

A token routed to an expert on another rank has to travel there and back. Two
all-to-all collectives wrap the local expert computation:

* **Dispatch** sends each token to the rank that owns its expert. It takes the
  router output (expert indices and weights) directly and delivers a receive
  buffer grouped by local expert.
* The local grouped GEMM runs on the receive buffer.
* **Combine** returns the results to the source rank and writes them back in the
  original token order.

No separate token dispatch or token combine is needed. Shared experts, which
every token passes through, are not part of the dispatch: they run as a regular
dense MLP on the local tokens on every rank.

.. raw:: html
   :file: img/moe_expert_parallel.svg

*Figure 9. Dispatch routes each token to the rank owning its expert, the local
experts run on the receive buffer, and combine returns the outputs to the source
rank.*

Transformer Engine provides optimized implementations of both operations,
including their backward passes, so the layer does not have to assemble them
from generic collectives and permutation kernels. They are built on the NCCL EP
library (``libnccl_ep``, loaded at runtime) and are differentiable.

* **Communication.** The NCCL EP kernels move each token straight to the slot
  of its expert on the owning rank, so the routing and the communication happen
  in one step.
* **Zero-copy mode.** Optionally, the token and receive buffers are allocated as
  NCCL symmetric memory (``symm_mem_alloc``): the same buffer is registered on
  every rank as a window, so the kernels write directly into the peer's buffer
  instead of staging the payload in internal NCCL buffers. Without it the
  library copies through its own staging buffers.

* **Receive buffer size.** ``recv_capacity_per_rank`` is the maximum number of
  tokens (rows of ``hidden_size``) a rank receives per step. Every rank sends at
  most ``max_tokens_per_rank`` tokens to ``top_k`` experts each, and in the
  worst case all of them are routed to experts on the same rank, so
  ``ep_size * max_tokens_per_rank * top_k`` never drops a token; a smaller
  capacity saves memory but can overflow when the routing is skewed (see
  ``drop_on_overflow``). With a fixed capacity the step is allocation-free and
  CUDA-graph capturable. Without it the buffer is sized from the actual receive
  count each step, at the cost of a host synchronization.
* **Quantized dispatch.** Dispatch can quantize the tokens before the
  all-to-all, so the communication moves the low-precision payload and the local
  grouped GEMM consumes it directly. MXFP8 is supported today; support for
  further recipes is in progress.
* **Examples.** Complete runnable examples:
  `examples/pytorch/ep <https://github.com/NVIDIA/TransformerEngine/tree/main/examples/pytorch/ep>`_
  and `examples/jax/ep <https://github.com/NVIDIA/TransformerEngine/tree/main/examples/jax/ep>`_.

.. tabs::

   .. tab:: PyTorch

      ``transformer_engine.pytorch.ep`` exposes the primitives with autograd
      support:

      * ``ep_bootstrap(ep_group, ...)`` initializes EP once per process on an
        existing process group and fixes the group-wide sizes (number of experts,
        maximum tokens per rank, hidden size, top-k, receive capacity).
      * ``EpBuffer`` holds the routing state of one dispatch/combine pair: where
        each token was sent, how many tokens each local expert received, and the
        metadata the combine and both backward passes need to undo the dispatch.
        Dispatch writes this state and combine and backward read it, so a buffer
        must not be reused until the backward of that call has run. Use one
        buffer per MoE layer, and one per microbatch when several microbatches
        are in flight (pipeline parallelism). ``dispatch_fwd_quant_recipe``
        selects the quantization applied by dispatch; with
        ``MXFP8BlockScaling()`` the receive buffer is returned as an MXFP8
        grouped tensor that the grouped GEMM consumes directly (see
        `tests/pytorch/distributed/run_ep.py <https://github.com/NVIDIA/TransformerEngine/blob/main/tests/pytorch/distributed/run_ep.py>`_).
      * ``ep_dispatch(buffer, tokens, topk_idx, topk_weights)`` returns the
        receive buffer with one fixed slot range per local expert, the routing
        weights of the received tokens, and the number of valid tokens per local
        expert.
      * ``ep_combine(buffer, expert_out)`` returns the summed expert outputs in
        the original token order. The routing weights are applied by the caller
        before the combine.

      Data flow between the calls:

      * ``EpBuffer`` itself allocates only the routing state (a small
        ``handle_mem`` byte buffer and the per-expert token counts).
      * ``ep_dispatch`` allocates the receive buffer
        ``[recv_capacity_per_rank, hidden_size]`` and the received weights on
        every call, or writes into caller-owned buffers passed as
        ``recv_tokens`` / ``recv_topk_weights`` (needed for CUDA graphs and
        zero-copy). The tokens land directly in their expert's slot range.
      * The local experts read the receive buffer as their input and produce a
        new ``expert_out`` tensor of the same shape; padded slots must be zero.
      * ``ep_combine`` reads ``expert_out`` in place and writes the result into a
        newly allocated ``[num_tokens, hidden_size]`` tensor. In zero-copy mode
        ``expert_out`` is transferred straight from that tensor when it is
        symmetric-memory backed; otherwise it goes through the library's
        staging buffers.

      .. raw:: html

         <div class="code-block-header">
            Requires SM90 (Hopper) or later
         </div>

      .. literalinclude:: moe_expert_parallel_pytorch.py
         :language: python
         :start-after: # START_MOE_EXPERT_PARALLEL_PYTORCH
         :end-before: # END_MOE_EXPERT_PARALLEL_PYTORCH

   .. tab:: JAX

      JAX offers two levels of API, both experimental:

      * ``transformer_engine.jax.moe.moe`` runs the whole MoE block (router,
        dispatch, expert MLPs, combine) as a single differentiable call. It is
        executed inside a ``Mesh``; ``ep_axis`` names the mesh axis the experts
        are sharded over and the dispatch and combine become all-to-all
        collectives over that axis. It also returns the load-balancing loss when
        ``aux_loss_coeff`` is non-zero.
      * ``transformer_engine.jax.ep`` exposes the primitives separately. Unlike
        the PyTorch ``EpBuffer``, the routing state is not kept in an object:
        dispatch returns it as arrays and the caller passes them on to combine.

        * ``ep_bootstrap(world_size, rank, num_experts, max_tokens_per_rank,
          recv_capacity_per_rank, hidden_dim, ...)`` initializes the EP group
          once per process. It runs inside the active ``Mesh`` and reads the EP
          axis (and the data-parallel axes) from ``MeshResource``; one process
          per device is required.
        * ``EpLayerConfig(top_k, ...)`` is a small per-layer configuration that
          every per-step call takes as its first argument.
        * ``ep_dispatch(cfg, topk_idx, tokens, topk_weights,
          recv_capacity_per_rank)`` scatters the tokens to the expert ranks and
          returns ``(recv_tokens, recv_topk_weights, handle_mem, token_counts,
          total_recv_tokens)``: the receive buffer grouped by local expert, the
          weights of the received tokens, the routing handle and per-expert
          token counts needed by combine, and the pre-drop receive total that can
          be used to detect overflow.
        * ``ep_combine(cfg, handle_mem, token_counts, expert_out,
          num_local_tokens)`` sums the expert outputs back on the source ranks in
          the original token order. It is unweighted: multiply ``expert_out`` by
          ``recv_topk_weights`` (and zero the padded slots) before calling it.
          ``num_local_tokens`` must be static because it fixes the output shape.

      .. raw:: html

         <div class="code-block-header">
            Requires SM90 (Hopper) or later
         </div>

      .. literalinclude:: moe_expert_parallel_jax.py
         :language: python
         :start-after: # START_MOE_EXPERT_PARALLEL_JAX
         :end-before: # END_MOE_EXPERT_PARALLEL_JAX

.. _moe-putting-it-together:

Example: putting it all together
--------------------------------

The example below wires the blocks together for top-k routing on a single
device: route, dispatch, run the experts, combine.

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

The example uses dropless routing (``num_out_tokens = num_tokens * top_k``), so
the dispatch buffer is sized statically rather than from a device-to-host sync.
Every stage is differentiable, so the assembled layer trains end to end.

With experts sharded across devices there are two options. With a generic
all-to-all, the routing kernels do the reordering on both sides of the
communication: tokens are sorted by destination rank before the all-to-all and
regrouped by local expert after it (see :ref:`Reordering expert chunks
<moe-token-permutation>`). With the NCCL-based :ref:`expert parallelism
<moe-expert-parallelism>` primitives the permutation is folded into the
communication: the dispatch delivers an expert-contiguous receive buffer and the
combine writes the results straight back into the original token order, so no
separate token dispatch or combine is needed.
