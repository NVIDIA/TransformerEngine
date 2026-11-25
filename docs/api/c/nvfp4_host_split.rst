..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

NVFP4 Host-Split API Migration
=============================

Host-split APIs consume a host array of expert lengths. Their new names make
that ownership explicit:

* ``nvte_group_amax_host_split``
* ``nvte_group_hadamard_transform_amax_host_split``
* ``nvte_group_hadamard_transform_cast_fusion_host_split``

The previous names remain deprecated forwarding symbols. Renaming does not
change amax semantics or alignment. Reading dynamic expert lengths back from
the device is not CUDA-graph safe; fixed host splits do not require that copy.

The fused host-split RHT quantizer requires each expert's row count and the
shared hidden dimension to be divisible by 128. Empty experts are allowed.
The PyTorch split-quantize dispatcher checks this before selecting the fusion.
For example, splits ``[64, 64]`` cannot use it even though the total is 128;
they use per-expert quantization with the same RHT recipe instead. Scale layout
conversion follows the selected implementation.

``nvte_group_nvfp4_quantize_with_amax`` and
``nvte_group_hadamard_transform_cast_fusion_columnwise`` are deprecated.
PyTorch no longer uses these single-direction grouped kernels. Their common
implementations remain available for compatibility, including their existing
alignment and supplied-amax contracts. A 2D quantizer is not a replacement for
the former's 1D recipe.

``nvte_group_nvfp4_compute_amax`` remains a deprecated host-split alias.
``nvte_group_amax_graph_safe`` is also deprecated as a standalone public entry;
its internal implementations remain available. Ordinary GroupedTensor
quantization uses ``nvte_group_quantize`` for supported recipes. The existing
GroupedTensor RHT APIs are unchanged.
