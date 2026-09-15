VMM output localization prototype
=================================

This prototype evaluates a focused MLPerf DSV3 pattern:

``ordinary activation -> MXFP8 quantization -> small GEMM``

It does not localize LayerNorm. Profiles do not show a reliable direct
LayerNorm-to-quantization producer relationship, especially in backward where
gradient accumulation or other elementwise work can occur between the two.

Workflow
--------

The activation remains an ordinary PyTorch allocation. The quantized MXFP8
data buffers use one contiguous virtual address range whose first and second
row slabs are backed by physical memory in locality domains 0 and 1.

Two green-context streams quantize the slabs concurrently:

1. The parent stream records a fork event after the input is ready.
2. Each green stream waits for the fork event.
3. Each stream reads its ordinary-memory input slab and writes its
   locality-backed output slab.
4. The parent stream waits for both completion events.
5. An unchanged full-chip GEMM consumes the single MXFP8 tensor directly.

There is no explicit copy and no split GEMM. Input reads are not localized, so
this intentionally measures the benefit available from localized output writes
and two-domain execution when the producer cannot supply localized memory.

Example
-------

.. code-block:: python

   import torch
   import transformer_engine.pytorch as te
   from transformer_engine.pytorch.cpp_extensions import general_gemm

   x = torch.randn((4096, 32768), device="cuda", dtype=torch.bfloat16)
   quantizer = te.MXFP8Quantizer(
       fp8_dtype=te.DType.kFloat8E4M3,
       rowwise=True,
       columnwise=True,
   )
   quantizer.optimize_for_gemm = True

   workspace = te.localize_mxfp8_output_vmm(x, quantizer)
   workspace.quantize()

   # workspace.output remains one GEMM-ready MXFP8Tensor.
   general_gemm(
       quantized_weight,
       workspace.output,
       out_dtype=torch.bfloat16,
       out=output,
   )

CUDA Graph capture
------------------

Capture quantization and GEMM together. Graph replay removes Python and TE
dispatch gaps between the two green launches and keeps the join-to-GEMM
dependency on the GPU.

Focused validation
------------------

.. code-block:: bash

   pytest -q tests/pytorch/mxfp8/test_mxfp8_localization.py \
     -k bidirectional_swizzled_vmm

   RUN_BENCHMARK_TESTS=1 \
   MXFP8_LOCALIZATION_USE_CUDA_GRAPH=1 \
   MXFP8_LOCALIZATION_GEMM_N=256 \
   pytest -q -s tests/pytorch/mxfp8/test_mxfp8_localization.py \
     -k bidirectional_swizzled_vmm_performance

Prototype limitations
---------------------

* Exactly two locality domains and equal, aligned row slabs.
* Bidirectional MXFP8 with fused GEMM scale swizzling only.
* Ordinary input allocation; only MXFP8 data outputs are VMM-localized.
* Scale buffers remain ordinary allocations.
* Explicit workspace lifetime management is required.
