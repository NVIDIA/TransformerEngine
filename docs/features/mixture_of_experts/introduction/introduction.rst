..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _moe-overview:

Introduction
===================================

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

.. raw:: html
   :file: img/moe_layer.svg

*Figure 1. The four stages of an MoE layer. The router produces the*
``routing_map`` *consumed by token dispatch and the* ``probs`` *used as merging
weights in token combine.*

Transformer Engine provides an optimized building block for each stage. They are
exposed as standalone functions, so they can be assembled into a complete MoE
layer or dropped into an existing implementation one piece at a time:

* The :doc:`router <../router/router>` fuses the score function with the top-k
  selection, and provides a fused load-balancing loss.
* :doc:`Token dispatch and combine <../routing_kernels/routing_kernels>` move
  tokens between their original order and the expert-contiguous layout using
  optimized kernels instead of Python-level gather / sort / concatenate chains.
* :doc:`Grouped GEMM <../grouped_gemm/grouped_gemm>` primitives execute the
  expert linear layers efficiently once the tokens are laid out in
  expert-contiguous blocks.

:doc:`Building an MoE layer <../moe_layer/moe_layer>` wires the four stages
together into a complete layer, and :doc:`Expert parallelism
<../expert_parallelism/expert_parallelism>` covers sharding the experts across
devices.
