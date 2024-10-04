..
    Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Frequently Asked Questions (FAQ)
================================

FP8 checkpoint compatibility 
----------------------------

Transformer Engine starts to support FP8 attention in 1.6. When checkpointing, it stores the FP8 metadata, including the scaling factors and amax history, under a `._extra_state` key. As our FP8 attention support expands from one backend to multiple backends, the location of the `._extra_state` key has also shifted. We take the `MultiheadAttention` module as an example and show the checkpoint structure in different Transformer Engine versions.

.. list-table::
   :widths: 15 25 50
   :header-rows: 1

   * - Version
     - FP8 metadata
     - Checkpoint compatibility (checkpoint version: loading behavior)
   * - <= 1.5
     - None
     -
       - <= 1.5: no FP8 metadata loaded (as expected)
       - > 1.5: "unexpected key" error
   * - 1.6, 1.7
     - `core_attention.fused_attention._extra_state`
     -
       - <= 1.5: initialize FP8 metadata to default, i.e. 1s for scaling factors and 0s for amaxes
       - 1.6, 1.7: load FP8 metadata from checkpoint
       - >= 1.8: "unexpected key" error
   * - >=1.8, <= 1.11
     - `core_attention._extra_state`
     -
       - <= 1.5: initialize FP8 metadata to default, i.e. 1s for scaling factors and 0s for amaxes
       - 1.6, 1.7: this checkpoint save/load version pair relies on users to map the 1.6/1.7 key to the 1.8-1.11 key; otherwise, initialize FP8 metadata to default, i.e. 1s for scaling factors and 0s for amaxes. Mapping in this example can be done by:
         .. code-block:: python

             >>> state_dict["core_attention._extra_state"] = \
                     state_dict["core_attention.fused_attention._extra_state"]
             >>> del state_dict["core_attention.fused_attention._extra_state"]
       - >= 1.8: load FP8 metadata from checkpoint
   * - >=1.12
     - `core_attention._extra_state`
     -
       - <= 1.5: initialize FP8 metadata to default, i.e. 1s for scaling factors and 0s for amaxes
       - >= 1.6: load FP8 metadata from checkpoint 
