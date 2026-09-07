..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Router
===================================

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

*Figure 1. The router scores the experts for each token and keeps the top-k.
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
--------------

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
