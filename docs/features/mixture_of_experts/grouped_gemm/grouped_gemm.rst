..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Grouped GEMM
===================================

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

*Figure 1. Both paths produce the same outputs from the same inputs. The
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
  </features/low_precision_training/index>`, so the same recipes available to
  regular ``Linear`` layers - FP8 (delayed, current, and blockwise scaling),
  MXFP8, and NVFP4 - can be used for MoE experts.
* **Fused quantization:** Low-precision grouped GEMM paths can fuse
  quantization-related work such as scale computation, casting, and
  cast/transpose steps across experts instead of repeating the same work in a
  Python loop.
* **Fused expert MLP:** Through the :doc:`operation-based API
  </examples/op_fuser/op_fuser>`, the two expert GEMMs and the activation
  between them can be fused into a single grouped operation on recent
  architectures; see :ref:`moe-fused-grouped-mlp` below.

The PyTorch ``GroupedLinear`` module also supports the features expected of a
Transformer Engine linear layer - tensor and sequence parallelism, gradient
accumulation fusion, and FP8 weight caching - so it can serve as a drop-in expert
layer. See the :doc:`PyTorch API reference </api/pytorch>` for the full
signature.

.. _moe-fused-grouped-mlp:

Fused grouped MLP
-----------------

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

*Figure 2. The operation fuser replaces the first grouped GEMM, the activation,
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
