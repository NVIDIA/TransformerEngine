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
#. With expert parallelism - experts sharded across devices - an
   **all-to-all dispatch** sends each token to the rank that owns its expert.
#. The **grouped MLP** (the experts) runs a single batched computation over all
   expert blocks local to the device.
#. With expert parallelism, an **all-to-all combine** returns the expert outputs
   to the rank the token came from.
#. **Token combine** scatters the expert outputs back into the original token
   order, merging the contributions when a token was sent to more than one
   expert.

.. raw:: html
   :file: img/moe_layer_ep.svg

*Figure 1. The stages of an MoE layer with expert parallelism. The router produces
the* ``routing_map`` *consumed by token dispatch and the* ``probs`` *used as merging
weights in token combine; the all-to-all dispatch and combine are only present
when the experts are sharded across ranks.*

Transformer Engine provides a building block for each stage. They are exposed as
standalone functions, so they can be assembled into a complete MoE layer or
dropped into an existing implementation one piece at a time:

* :ref:`Routing kernels <moe-routing-kernels>`: a fused router (score function
  and top-k selection), a fused :ref:`load-balancing loss <moe-load-balancing>`,
  and token dispatch and combine kernels that move tokens between their original
  order and the expert-contiguous layout.
* :ref:`Grouped GEMM <moe-grouped-gemm>`: the expert linear layers as one call
  over expert-contiguous blocks; the :ref:`grouped MLP <moe-grouped-mlp>` fuses
  the whole expert MLP into one kernel.
* :ref:`Expert parallelism <moe-expert-parallelism>`: all-to-all dispatch and
  combine for experts sharded across devices.

The :ref:`example at the end <moe-putting-it-together>` wires the blocks into a
complete MoE layer.

.. _moe-routing-kernels:

Routing kernels
---------------

The router produces a routing map. Token dispatch moves the tokens into the
expert-contiguous layout expected by the grouped GEMM, and token combine moves
the expert outputs back. All of these kernels are differentiable. The snippets
below use the mask-map routing variant; other variants (for example index-map
routing) follow the same pattern, see the :doc:`PyTorch API reference
</api/pytorch>` and :doc:`JAX API reference </api/jax>`.

.. _moe-router:

Router
~~~~~~

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

``expert_bias`` balances the load without an extra loss term. With the sigmoid
score function it is added to the scores only for the top-k selection, so it
changes which experts are picked but not the returned routing weights. Update
it between steps: lower it for overloaded experts and raise it for under-used
ones.

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

The sort-chunks-by-index kernels permute contiguous chunks of a token tensor
according to a list of chunk sizes and a permutation of chunk indices, for
example to regroup tokens by destination rank before an all-to-all and to
restore the original grouping afterwards. A ``_with_probs`` variant reorders an
accompanying probability tensor in the same call. See the API reference for the
signatures.

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

* **Backends:** the grouped GEMM backend is selected based on datatype and GPU
  architecture, for example cuBLAS GEMMs on multiple CUDA streams or a single
  grouped GEMM kernel.
* **Low-precision recipes:** the grouped GEMM works with the
  :doc:`low-precision training recipes </features/low_precision_training/index>`
  available to ``Linear``. The supported set depends on the GPU architecture and
  cuBLAS version; see the API reference for the current constraints.
* **Fused quantization:** in low-precision paths the scale computation, casting
  and cast/transpose steps are fused across experts.
* **Fused expert MLP:** the two expert GEMMs and the activation between them can
  be fused into one operation, see :ref:`Grouped MLP <moe-grouped-mlp>`.

.. _moe-grouped-mlp:

Grouped MLP
-----------

An expert MLP is two grouped GEMMs with an activation between them. On
Blackwell (SM100) GPUs the whole expert MLP can run as a single CuTe DSL kernel:
the intermediate activation stays on chip and its quantization is folded into
the GEMMs.

.. raw:: html
   :file: img/moe_grouped_mlp.svg

*Figure 7. The operation fuser replaces the first grouped GEMM, the activation,
and the second grouped GEMM with a single fused grouped-MLP kernel that keeps
the intermediate on chip.*

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

The fused path is taken when all of the following hold; otherwise the three
operations run separately with identical results:

* **Architecture:** Blackwell (SM100) with cuDNN frontend 1.23 or newer.
* **Recipe:** a block-scaled low-precision recipe - MXFP8, or NVFP4 with the
  randomized Hadamard transform enabled.
* **Opt-in:** the environment variable ``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1``.
* **Activation:** a scaled ``SwiGLU`` / ``GeGLU`` (gated) or ``SReLU`` (unary),
  with feature dimensions aligned to 64 and the token count to 128.

.. _moe-expert-parallelism:

Expert parallelism
------------------

.. note::

    NCCL-based expert parallelism requires Hopper (SM90) or later and NCCL 2.30.4
    or newer. It is compiled in by default when Transformer Engine is built for
    these architectures; set ``NVTE_WITH_NCCL_EP=0`` at build time to disable it.

With expert parallelism (EP) the experts are sharded across devices, and each
device owns a slice of them. Two all-to-all collectives wrap the local expert
computation: a **dispatch** all-to-all sends each token to the rank that owns its
expert, the local grouped GEMM runs, and a **combine** all-to-all returns the
results to the source rank.

.. raw:: html
   :file: img/moe_expert_parallel.svg

*Figure 8. With experts sharded across ranks, a dispatch all-to-all routes each
token to the rank owning its expert and a combine all-to-all returns the
outputs to the source rank.*

Dispatch and combine are implemented directly on NCCL, using symmetric-memory
windows for zero-copy transfers. Both are differentiable. The common C API
(``nvte_ep_dispatch`` / ``nvte_ep_combine`` and their backward passes, declared
in ``transformer_engine/common/include/transformer_engine/ep.h``) is exposed in
both frameworks; the snippets show how the dispatch, the local experts and the
combine are wired together.

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
~~~~~~~~~~~~~~~~~~~~~~~~~

Each rank receives a data-dependent number of tokens per step. Passing a fixed
receive capacity (``recv_capacity_per_rank``) sizes the receive buffer up front,
so the step needs no device-to-host synchronization and can be captured in a
CUDA graph; the dropless worst case is ``ep_size * max_tokens_per_rank * top_k``.
Without it the buffer is sized from the actual receive count each step, at the
cost of a host sync.

Dispatch can quantize the tokens to MXFP8 before the all-to-all
(``dispatch_fwd_quant_recipe``), so the communication moves the low-precision
payload and the local grouped GEMM consumes it directly. Complete runnable
examples live in ``examples/pytorch/ep/`` and ``examples/jax/ep/`` in the
repository.

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
<moe-routing-kernels>`). With the NCCL-based :ref:`expert parallelism
<moe-expert-parallelism>` primitives the permutation is folded into the
communication: the dispatch delivers an expert-contiguous receive buffer and the
combine writes the results straight back into the original token order, so no
separate token dispatch or combine is needed.
