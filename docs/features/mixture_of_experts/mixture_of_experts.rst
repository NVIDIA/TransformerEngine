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

Mixture of Experts (MoE) layers replace a dense feed-forward network with a set
of expert networks and a router that sends each token to one or more experts.
This keeps the activated parameter count per token small while allowing the
model to scale to many more total parameters.

A token passes through an MoE layer in four stages:

#. The **router** scores the experts for each token and selects the top-k of
   them.
#. **Token dispatch** gathers the tokens into expert-contiguous order.
#. The **grouped MLP** (the experts) runs a single batched computation over all
   expert blocks.
#. **Token combine** scatters the expert outputs back into the original token
   order, merging the contributions when a token was sent to more than one
   expert.

When the experts do not fit on one device, they are sharded across devices
(**expert parallelism**). The layer then gains two all-to-all collectives: a
dispatch that sends each token to the rank owning its expert, and a combine that
returns the results to the source rank. The experts themselves stay local.

.. raw:: html
   :file: img/moe_layer_ep.svg

*Figure 1. The stages of an MoE layer with expert parallelism. The router produces
the* ``routing_map`` *consumed by token dispatch and the* ``probs`` *used as merging
weights in token combine; the all-to-all dispatch and combine are only present
when the experts are sharded across ranks.*

Transformer Engine provides an optimized building block for each stage. They are
exposed as standalone functions, so they can be assembled into a complete MoE
layer or dropped into an existing implementation one piece at a time:

* :ref:`Routing kernels <moe-routing-kernels>`: the router fuses the score
  function with the top-k selection, and token dispatch and combine move tokens
  between their original order and the expert-contiguous layout with optimized
  kernels instead of Python-level gather / sort / concatenate chains.
* :ref:`Grouped GEMM <moe-grouped-gemm>` primitives execute the expert linear
  layers efficiently once the tokens are laid out in expert-contiguous blocks,
  and the :ref:`grouped MLP <moe-grouped-mlp>` fuses the whole expert MLP into
  one kernel.
* :ref:`Expert parallelism <moe-expert-parallelism>` shards the experts across
  devices with all-to-all dispatch and combine.

The :ref:`example at the end <moe-putting-it-together>` wires the blocks into a
complete MoE layer.

.. _moe-routing-kernels:

Routing kernels
---------------

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

Transformer Engine fuses the score function and the top-k selection into a single
differentiable kernel, exposed as ``fused_topk_with_score_function`` in both
``transformer_engine.pytorch.router`` and ``transformer_engine.jax.router``. All
internal math runs in FP32 for numerical stability, regardless of the logits
dtype.

.. raw:: html
   :file: img/moe_router.svg

*Figure 2. The router scores the experts for each token and keeps the top-k.
The selected entries populate* ``routing_map`` *(a 0/1 mask) and* ``probs`` *(the
routing weights); all other entries are zero.*

The kernel covers the score functions and selection variants used by common MoE
architectures:

* **Score function:** ``"softmax"`` or ``"sigmoid"`` (the PyTorch API also offers
  ``"sqrtsoftplus"``). With softmax, ``use_pre_softmax`` selects whether the
  softmax is applied before or after the top-k.
* **Grouped (device-limited) routing:** ``num_groups`` and ``group_topk`` restrict
  selection to a subset of expert groups, as in DeepSeek-style routing.
* **Expert bias:** with the sigmoid score function, ``expert_bias`` shifts the
  selection without changing the returned weights - the bias-adjustment scheme
  used for auxiliary-loss-free load balancing.
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

Load balancing
~~~~~~~~~~~~~~

Left unconstrained, a router tends to collapse onto a handful of experts. The
usual remedy is an auxiliary load-balancing loss that rewards spreading tokens
evenly across experts. Transformer Engine computes it with ``fused_moe_aux_loss``
from the per-expert token counts and the *dense* routing scores - one value per
expert rather than only the selected top-k - so the loss has a gradient with
respect to every expert's logit. Those dense scores come from
``fused_compute_score_for_moe_aux_loss`` in PyTorch, or from
``fused_topk_with_score_function(..., compute_aux_scores=True)`` in JAX.

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

Once the router has produced a routing map, the tokens must be moved into the
expert-contiguous layout expected by the grouped GEMM (``GroupedLinear`` in
PyTorch, ``grouped_dense`` in JAX) and, afterwards, moved back. Transformer
Engine provides differentiable kernels for both directions. This section focuses on
the two core operations - token dispatch and token combine - because they
illustrate the layout transformation used by the other variants.

The snippets below show one concrete instance of this pattern: the mask-map
routing path, exposed in PyTorch as ``transformer_engine.pytorch.moe_permute``
and ``transformer_engine.pytorch.moe_unpermute``, and in JAX as
``transformer_engine.jax.permutation.token_dispatch`` and
``transformer_engine.jax.permutation.token_combine``. Other routing variants
(for example, index-map routing in PyTorch via ``map_type="index"``) are
available in both frameworks and follow the same pattern; see the
:doc:`PyTorch API reference </api/pytorch>` and
:doc:`JAX API reference </api/jax>` for the complete list and signatures.
The mask-map APIs have different framework-specific wrappers, but lower to the
same shared Triton permutation kernels, and both pairs are differentiable so they
can be used directly inside training graphs.

Token dispatch
~~~~~~~~~~~~~~

Token dispatch is the canonical routing operation: given the original token
tensor and a routing map describing each token's destination expert, it returns
a permuted token buffer in which all rows assigned to the same expert are
stored contiguously. In PyTorch this operation is exposed as ``moe_permute``;
in JAX it is exposed as ``token_dispatch``. This is exactly the layout that
the grouped linear layer consumes via its per-expert token-count argument
(``m_splits`` in PyTorch ``GroupedLinear``, ``group_sizes`` in JAX
``grouped_dense``), so token dispatch followed by the grouped GEMM forms a
typical MoE forward block.

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

Both variants return the permuted token buffer of shape
``[num_out_tokens, hidden_size]`` together with a ``row_id_map`` that
carries enough information for token combine to restore the original token
order once the expert computation is done. Token dispatch and token combine
are typically used as a matched pair around the grouped GEMM call.

Token combine
~~~~~~~~~~~~~

Token combine is the inverse routing operation: it takes the expert-contiguous
output produced by the grouped GEMM (or any per-expert computation) and the
``row_id_map`` returned by token dispatch, and returns a single tensor of
shape ``[num_tokens, hidden_size]`` with the rows written back into the
original token order. In PyTorch this operation is exposed as
``moe_unpermute``; in JAX it is exposed as ``token_combine``.

For top-1 routing each token has exactly one expert contribution, so
``merging_probs`` is omitted. For top-k routing pass the per-token expert
weights as ``merging_probs`` and the kernel computes a weighted sum of the
per-expert contributions in the same fused pass; without it the per-expert
contributions are summed unweighted. In PyTorch, also pass
``restore_shape=(num_tokens, hidden_size)`` whenever the permuted buffer has more
rows than the original tokens (top-k routing); JAX infers the original token
count from the ``row_id_map``.

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

In top-k routing each token contributes to several experts, and those
contributions are recombined using the routing weights. There are two equivalent
places to apply the weights:

* **At combine (output side).** Pass the routing weights to token combine as
  ``merging_probs``; it forms the weighted sum of the per-expert contributions in
  the same fused pass. This is the path used in the examples above.
* **At dispatch (input side).** Scale each expert's input by its routing weight
  before the grouped GEMM. ``moe_permute_with_probs`` (PyTorch) and the ``probs``
  argument of ``token_dispatch`` (JAX) permute a probability tensor alongside the
  tokens, so the weights arrive already aligned with the expert-contiguous
  layout.

Padding and alignment
~~~~~~~~~~~~~~~~~~~~~

Grouped GEMM backends are most efficient when each expert's token block starts at
an aligned offset (for example, a multiple of 128 rows). Because the number of
tokens routed to an expert is data dependent, the blocks are generally ragged.
Transformer Engine can pad each block up to a multiple of ``align_size`` as part
of the dispatch kernel, avoiding a separate padding pass.

.. raw:: html
   :file: img/moe_padding.svg

*Figure 5. Each expert's block is rounded up to a multiple of* ``align_size``\ *.
The per-expert padding offsets are returned so that token combine can drop the
padding again.*

In PyTorch this is ``moe_permute_and_pad_with_probs``; in JAX it is the
``align_size`` argument of ``token_dispatch``. Both return the padded token
buffer, the aligned per-expert token counts (used as ``m_splits`` /
``group_sizes`` for the grouped GEMM), and the per-expert ``pad_offsets`` that
token combine needs in order to remove the padding.

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

When experts are sharded across devices, the per-expert token blocks often have
to be reordered - for example, to regroup tokens by destination rank before an
all-to-all, or to restore the original grouping afterwards.
``moe_sort_chunks_by_index`` (PyTorch) and ``sort_chunks_by_index`` (JAX) permute
contiguous chunks of a token tensor according to a list of chunk sizes and a
permutation of chunk indices, without falling back to Python-level slicing and
concatenation. ``moe_sort_chunks_by_index_with_probs`` reorders an accompanying
probability tensor in the same call.

.. _moe-grouped-gemm:

Grouped GEMM
------------

The straightforward way to apply per-expert linear layers is to loop over the
experts and call a separate ``Linear`` for each one. This is correct, but it
is not the most efficient way to execute many expert GEMMs.

Transformer Engine provides a grouped GEMM primitive
(``GroupedLinear`` in PyTorch and ``grouped_dense`` in JAX) - an optimized
replacement that produces the same outputs as the loop while using
implementations that are better suited for MoE workloads.

Let ``G`` be the number of experts. For expert ``i``, ``X_i`` is the routed
token block, ``W_i`` is the expert weight, and ``b_i`` is the optional bias:

.. math::

   Y_i = X_i W_i^T + b_i,\quad i = 0, \ldots, G - 1

The full layer output is the concatenation of all expert outputs:

.. math::

   Y = \mathrm{concat}(Y_0, Y_1, \ldots, Y_{G-1})

The grouped GEMM is told how many token rows belong to each expert via a
per-expert token-count argument: ``m_splits`` in PyTorch ``GroupedLinear`` and
``group_sizes`` in JAX ``grouped_dense``.

.. raw:: html
   :file: img/grouped_linear.svg

*Figure 6. Both paths produce the same outputs from the same inputs. The
baseline launches one* ``Linear`` *per expert, while the grouped GEMM
(*\ ``GroupedLinear`` *in PyTorch,* ``grouped_dense`` *in JAX) is an optimized
grouped implementation that replaces the loop.*

The following snippets show how to replace the loop with the grouped GEMM.
They assume the tokens have already been permuted into expert-contiguous order.

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

The grouped GEMM uses implementations tuned for grouped expert execution:

* **Optimized backends:** Transformer Engine selects from several grouped GEMM
  backends depending on the framework, datatype, and GPU architecture. This
  can be, for example, cuBLAS GEMMs launched on multiple CUDA streams or a
  single grouped GEMM kernel, among other backend-specific implementations.
* **Recipe compatibility:** the grouped GEMM is integrated with
  Transformer Engine's :doc:`low-precision training stack
  </features/low_precision_training/index>`, so the recipes available to
  regular ``Linear`` layers can also be used for MoE experts. The exact set
  depends on the execution path, GPU architecture, and cuBLAS version; see the
  ``GroupedLinear`` / ``grouped_dense`` API reference for the current
  constraints.
* **Fused quantization:** Low-precision grouped GEMM paths can fuse
  quantization-related work such as scale computation, casting, and
  cast/transpose steps across experts instead of repeating the same work in a
  Python loop.
* **Fused expert MLP:** Through the :doc:`operation-based API
  </examples/op_fuser/op_fuser>`, the two expert GEMMs and the activation
  between them can be fused into a single grouped operation on recent
  architectures; see :ref:`Grouped MLP <moe-grouped-mlp>`.

The PyTorch ``GroupedLinear`` module also supports the features expected of a
Transformer Engine linear layer - tensor and sequence parallelism, gradient
accumulation fusion, and FP8 weight caching - so it can serve as a drop-in expert
layer. See the :doc:`PyTorch API reference </api/pytorch>` for the full
signature.

.. _moe-grouped-mlp:

Grouped MLP
-----------

An expert MLP is two grouped GEMMs with an activation between them: the first
projects into the (gated) feed-forward dimension, the activation is applied, and
the second projects back. Running these as separate kernels writes the large
intermediate activation out to HBM and reads it back for the second GEMM, and
re-quantizes it in a separate pass.

On Blackwell (SM100) GPUs, Transformer Engine can fuse the whole expert MLP -
both grouped GEMMs and the activation - into a single CuTe DSL kernel. The
intermediate stays on chip and the cross-expert quantization is folded into the
GEMMs, removing the HBM round-trip and the extra kernel launches.

.. raw:: html
   :file: img/moe_grouped_mlp.svg

*Figure 7. The operation fuser replaces the first grouped GEMM, the activation,
and the second grouped GEMM with a single fused grouped-MLP kernel that keeps
the intermediate on chip.*

The fusion is exposed through the operation-based API and applied automatically
by the :doc:`operation fuser </examples/op_fuser/op_fuser>`: when it sees a
grouped linear, a scaled GLU (or SReLU) activation, and another grouped linear in
sequence, it replaces them with one fused grouped-MLP operation. No change to the
forward code is needed to opt in.

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

The fused path is taken when all of the following hold; otherwise the three ops
run separately and produce identical results:

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

The grouped GEMM keeps all experts on a single device. When the experts no longer
fit there - or to add another dimension of parallelism - they are sharded across
devices, a scheme called expert parallelism (EP). Each device then owns only a
slice of the experts, so a token routed to a non-local expert has to travel to
the device that owns it.

That data movement is two all-to-all collectives wrapped around the local expert
computation: a **dispatch** all-to-all sends each token to the rank that owns its
expert, the local grouped GEMM runs, and a **combine** all-to-all returns the
results to the source rank. It is the distributed counterpart of the
:ref:`token dispatch and token combine kernels <moe-routing-kernels>`.

.. raw:: html
   :file: img/moe_expert_parallel.svg

*Figure 8. With experts sharded across ranks, a dispatch all-to-all routes each
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
  itself comes from the :ref:`router <moe-router>`; the local experts run
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
~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. _moe-putting-it-together:

Example: putting it all together
--------------------------------

The building blocks assemble into the four stages from the introduction: route,
dispatch, run the experts, and combine. The example below wires them together
for top-k routing on a single device.

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
expert step is built from the :ref:`grouped GEMM <moe-grouped-gemm>`; a full
expert MLP stacks two grouped GEMMs around an activation. Every stage is differentiable, so the assembled layer
trains end to end.

When the experts are sharded across devices, the same layer gains the two
all-to-all collectives from Figure 1 around the local experts: token dispatch
groups the tokens by destination rank, the all-to-all dispatch moves them to the
ranks owning their experts, the local grouped MLP runs, and the all-to-all
combine returns the outputs before token combine restores the original order and
applies the routing weights.

The routing kernels and expert parallelism complement each other. With a generic
all-to-all, the routing kernels do the reordering on both sides of the
communication: tokens are sorted by destination rank before the all-to-all and
regrouped by local expert after it (see :ref:`Reordering expert chunks
<moe-routing-kernels>`). With the NCCL-based :ref:`expert parallelism
<moe-expert-parallelism>` primitives, this permutation is folded into the
communication itself: the dispatch delivers an expert-contiguous receive buffer
and the combine writes the results straight back into the original token order,
so no separate permute or unpermute is needed on the local side.
