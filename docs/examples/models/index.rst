..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Model layers
============

``transformer_engine.pytorch.models`` contains complete transformer layers of
specific model families, composed from Transformer Engine modules and
operations. They are the quickest way to run a model's layer on the TE
kernels (fused attention, grouped GEMMs, fused activations, NCCL expert
parallelism, low-precision recipes) without assembling it by hand, and they
serve as reference compositions for frameworks that build their own.

Each page starts with the Transformer Engine features the layer is built
from, then shows how to use the layer and where its time goes.

.. toctree::
   :maxdepth: 1

   deepseek_v3
