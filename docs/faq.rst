..
    Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Frequently Asked Questions (FAQ)
================================

FP8 checkpoint compatibility
----------------------------

Transformer Engine starts to support FP8 attention in 1.6. It stores the FP8 metadata, i.e. scaling factors and amax histories, under a `._extra_state` key in the checkpoint. As the FP8 attention support expands from one backend to multiple backends, the location of the `._extra_state` key has also shifted.

Here, we take the `MultiheadAttention` module as an example. Its FP8 attention metadata in Transformer Engine 1.11 is stored as `core_attention._extra_state` as shown below.

.. code-block:: python

    >>> from transformer_engine.pytorch import MultiheadAttention, fp8_model_init
    >>> with fp8_model_init(enabled=True):
    ...     mha = MultiheadAttention(
    ...         hidden_size=1024,
    ...         num_attention_heads=16,
    ...         bias=True,
    ...         params_dtype=torch.bfloat16,
    ...         input_layernorm=False,
    ...         fuse_qkv_params=True,
    ...         attention_type="self",
    ...         qkv_weight_interleaved=True,
    ...     ).to(dtype=torch.bfloat16, device="cuda")
    ...
    >>> state_dict = mha.state_dict()
    >>> print(state_dict.keys())
    odict_keys(['qkv.weight', 'qkv.bias', 'qkv._extra_state', 'core_attention._extra_state', 'proj.weight', 'proj.bias', 'proj._extra_state'])

Here is a full list of the checkpoint save/load behaviors from all Transformer Engine versions.

.. list-table::

   * - **Version: <= 1.5**

         - Saves no FP8 metadata since FP8 attention is not supported
         - Loading behavior for checkpoints created by the following versions:

             :<= 1.5:    Loads no FP8 metadata
             :>  1.5:    Error: unexpected key
   * - **Version: 1.6, 1.7**

         - Saves FP8 metadata to `core_attention.fused_attention._extra_state`
         - Loading behavior for checkpoints created by the following versions:

             :<= 1.5:    Initializes FP8 metadata to the default, i.e. 1s for scaling factors, and 0s for amaxes
             :1.6, 1.7:  Loads FP8 metadata from checkpoint
             :>= 1.8:    Error: unexpected key
   * - **Version: >=1.8, <= 1.11**

         - Saves FP8 metadata to `core_attention._extra_state`
         - Loading behavior for checkpoints created by the following versions:

             :<= 1.5:    Initializes FP8 metadata to the default, i.e. 1s for scaling factors, and 0s for amaxes
             :1.6, 1.7:  This save/load combination relies on users to map the 1.6/1.7 key to the 1.8-1.11 key. Otherwise, it initializes FP8 metadata to the default, i.e. 1s for scaling factors, and 0s for amaxes. The mapping can be done, in this `MultiheadAttention` example, by

              .. code-block:: python

                  >>> state_dict["core_attention._extra_state"] = \
                          state_dict["core_attention.fused_attention._extra_state"]
                  >>> del state_dict["core_attention.fused_attention._extra_state"]

             :>= 1.8:    Loads FP8 metadata from checkpoint
   * - **Version: >=1.12**

         - Saves FP8 metadata to `core_attention._extra_state`
         - Loading behavior for checkpoints created by the following versions:

             :<= 1.5:    Initializes FP8 metadata to the default, i.e. 1s for scaling factors, and 0s for amaxes
             :>= 1.6:    Loads FP8 metadata from checkpoint
