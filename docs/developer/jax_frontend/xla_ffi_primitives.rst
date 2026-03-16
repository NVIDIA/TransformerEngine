..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

XLA FFI Primitives
==================

The JAX frontend communicates with the C++ core through XLA's Foreign Function Interface
(FFI). Each TE operation is registered as a custom call that XLA can invoke during
compiled execution.

BasePrimitive
-------------

**Location**: ``transformer_engine/jax/cpp_extensions/base.py``

``BasePrimitive`` is the base class for all JAX TE primitives. It provides the machinery
to:

1. **Register** a custom call with XLA (both forward and backward).
2. **Define abstract evaluation** (shape/dtype inference without running the kernel).
3. **Define the custom VJP** (JAX's equivalent of PyTorch's autograd backward).
4. **Handle sharding** (propagate partition specs through the operation).

.. code-block:: python

   class BasePrimitive(metaclass=ABCMeta):

       @classmethod
       def enabled(cls) -> bool: ...  # Enable/disable via NVTE_JAX_CUSTOM_CALLS env var

       @staticmethod
       @abstractmethod
       def abstract(*args, **kwargs): ...     # Shape/dtype inference

       @staticmethod
       @abstractmethod
       def lowering(ctx, *args, **kwargs): ... # MLIR lowering to XLA custom call

       @staticmethod
       @abstractmethod
       def impl(*args, **kwargs): ...          # Eager execution implementation

       @staticmethod
       @abstractmethod
       def batcher(*args): ...                 # vmap batch rules

       @staticmethod
       @abstractmethod
       def partition(*args): ...               # Shardy multi-device partitioning

       @staticmethod
       @abstractmethod
       def shardy_sharding_rule(*args): ...    # Shardy sharding spec string

Registration Flow
-----------------

The ``register_primitive(cls)`` function registers each primitive in two forms:

- **Inner primitive**: Single-device execution, no sharding awareness.
- **Outer primitive**: Multi-device aware, uses Shardy custom partitioning.

When the JAX TE module is imported, each primitive:

1. Creates a JAX ``core.Primitive`` object (both inner and outer).
2. Registers ``abstract`` as the abstract evaluation rule.
3. Registers ``lowering`` as the MLIR lowering rule (maps to XLA custom call via FFI).
4. Registers ``batcher`` for ``jax.vmap`` support.
5. Registers ``partition`` and ``shardy_sharding_rule`` for XLA SPMD.
6. Backward is registered via ``jax.custom_vjp`` on the outer primitive.

Primitives can be selectively enabled/disabled via the ``NVTE_JAX_CUSTOM_CALLS``
environment variable.

Custom Call Execution
---------------------

When XLA encounters a TE custom call during execution:

.. code-block:: text

   JAX trace → MLIR lowering → XLA custom call → C++ handler → CUDA kernel

The C++ handler:

1. Receives raw GPU buffer pointers from XLA.
2. Wraps them in ``NVTETensor`` handles.
3. Calls the same C API functions as the PyTorch frontend.
4. Returns results via pre-allocated output buffers.

Comparison with PyTorch Bridge
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Aspect
     - PyTorch (pybind11)
     - JAX (XLA FFI)
   * - Binding mechanism
     - Python C extension
     - XLA custom call
   * - When called
     - Eagerly per operation
     - During XLA compilation/execution
   * - Autodiff
     - ``torch.autograd.Function``
     - ``jax.custom_vjp``
   * - Shape inference
     - Implicit (PyTorch is eager)
     - Explicit ``abstract()`` method
   * - Parallelism
     - Manual collectives
     - XLA SPMD via sharding rules

Key Primitives
--------------

Major primitives registered in ``transformer_engine/jax/cpp_extensions/``, organized by
functional area:

**GEMM** (``gemm.py``):

- ``te_gemm_v2_ffi`` — Matrix multiplication with FP8
- ``te_grouped_gemm_ffi`` — Grouped GEMM for MoE
- ``te_grouped_gemm_d2h_group_sizes_ffi`` — Device-to-host group size transfer

**Quantization** (``quantization.py``):

- ``te_dbias_quantize_ffi`` — Quantize with optional bias gradient
- ``te_grouped_quantize_ffi`` — Grouped quantization

Exported as: ``quantize``, ``quantize_dbias``, ``grouped_quantize``, ``grouped_dbias``.

**Activation** (``activation.py``):

- ``te_act_lu_ffi`` — Forward activation (GeLU, SiLU, etc.)
- ``te_dact_dbias_quantize_ffi`` — Fused backward activation + bias gradient + quantize

Exported as: ``act_lu``, ``dact_lu``, ``quantize_dact_dbias``.

**Normalization** (``normalization.py``):

- ``te_norm_forward_ffi`` — LayerNorm / RMSNorm forward
- ``te_norm_backward_ffi`` — LayerNorm / RMSNorm backward

**Attention** (``attention.py``):

- ``te_fused_attn_forward_ffi`` — Fused attention forward
- ``te_fused_attn_backward_ffi`` — Fused attention backward

**Softmax** (``softmax.py``):

- ``te_scaled_softmax_forward_ffi`` / ``te_scaled_softmax_backward_ffi``
- ``te_scaled_masked_softmax_forward_ffi`` / ``te_scaled_masked_softmax_backward_ffi``
- ``te_scaled_upper_triang_masked_softmax_forward_ffi`` / ``te_scaled_upper_triang_masked_softmax_backward_ffi``

**Router / MoE** (``router.py``):

- ``te_fused_topk_with_score_function_forward_ffi`` / ``te_fused_topk_with_score_function_backward_ffi``
- ``te_fused_moe_aux_loss_forward_ffi`` / ``te_fused_moe_aux_loss_backward_ffi``

**Amax** (``amax.py``):

- ``te_rht_amax_ffi`` — Randomized Hadamard transform + amax calculation

See Also
--------

- :doc:`/developer/pytorch_frontend/cpp_extensions` — PyTorch equivalent
- :doc:`/developer/cpp_core/kernel_areas` — The C++ kernels being called
