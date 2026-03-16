..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _ops-framework:

Op Fusion Framework
===================

Transformer Engine includes an operation fusion framework (``transformer_engine/pytorch/ops/``)
that enables composing and fusing operations for better performance.

Overview
--------

.. figure:: ./img/ops_fusion.svg
   :align: center
   :width: 70%

   Before and after op fusion: separate ops become a single fused op.

..
   Diagram description for ``ops_fusion.svg``:
   Left side "Before Fusion":
     Three sequential boxes: "LayerNorm" → "Quantize" → "Linear"
     Each box has a separate kernel launch arrow.
   Right side "After Fusion":
     Single box: "LayerNormLinear (fused)"
     Single kernel launch arrow.
   Arrow between left and right labeled "Op Fuser".

The framework defines three levels of abstraction:

- **FusibleOperation** (``ops/op.py``): Abstract base for any operation that can
  participate in fusion.
- **BasicOperation** (``ops/op.py``): Single operations (GEMM, bias, normalization, etc.)
  that serve as building blocks. Holds parameters and quantization state.
- **FusedOperation** (``ops/op.py``): Compound operations composed of multiple basic ops.
  Delegates parameter management to its constituent basic ops.
- **OperationFuser** (``ops/fuser.py``): Manages the fusion pipeline, using a custom
  autograd function (``_OperationFuserAutogradFunction``) to coordinate forward/backward.
- **Sequential** (``ops/sequential.py``): A container that groups consecutive
  ``FusibleOperation`` instances for automatic fusion.

Basic Ops
---------

Basic ops extend ``BasicOperation`` and implement ``op_forward()`` / ``op_backward()``:

.. code-block:: text

   ops/basic/
   ├── linear.py          # BasicLinear — matrix multiplication
   ├── bias.py            # Bias — bias addition
   ├── normalization.py   # LayerNorm / RMSNorm
   ├── activation.py      # GELU, GEGLU, SwiGLU, GLU
   ├── quantization.py    # Quantize — cast to/from FP8
   ├── all_gather.py      # AllGather (distributed)
   ├── all_reduce.py      # AllReduce (distributed)
   ├── reduce_scatter.py  # ReduceScatter (distributed)
   ├── dropout.py         # Dropout
   ├── reshape.py         # Reshape
   └── identity.py        # Identity

Fused Ops
---------

Fused ops extend ``FusedOperation`` and compose basic ops into optimized sequences.
The most important fused op is ``Linear`` (``ops/linear.py``), which composes
``BasicLinear`` + ``Bias`` + optional ``AllReduce``/``ReduceScatter`` for tensor
parallelism:

.. code-block:: python

   # ops/linear.py — row-parallel example
   Linear = FusedOperation([BasicLinear, Bias, ReduceScatter])

Op Fuser
--------

The ``OperationFuser`` (``ops/fuser.py``) manages a pipeline of fusible operations. It
wraps them in ``_OperationFuserAutogradFunction``, which:

- **Forward**: Applies ops sequentially with potential fusion, passing quantizers between
  ops.
- **Backward**: Reverses op order, applies fused/unfused backward, using per-op saved
  contexts.

``Sequential`` (``ops/sequential.py``) is the user-facing container. It groups
consecutive ``FusibleOperation`` instances via ``_make_module_groups()`` and applies
the fuser automatically.

Relationship to Modules
-----------------------

The ops framework is an alternative to the monolithic module approach
(``LayerNormLinear``, ``LayerNormMLP``). Modules like ``te.Linear`` (in ``module/``) use
the ``_Linear`` autograd function directly, while the ops framework allows more flexible
composition and automatic fusion discovery.

Both approaches ultimately call the same C++ extensions and CUDA kernels.
