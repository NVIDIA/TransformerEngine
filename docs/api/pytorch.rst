..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

PyTorch
=======

.. autoapiclass:: transformer_engine.pytorch.Linear(in_features, out_features, bias=True, **kwargs)
  :members: forward, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.GroupedLinear(in_features, out_features, bias=True, **kwargs)
  :members: forward, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.LayerNorm(hidden_size, eps=1e-5, **kwargs)

.. autoapiclass:: transformer_engine.pytorch.RMSNorm(hidden_size, eps=1e-5, **kwargs)

.. autoapiclass:: transformer_engine.pytorch.LayerNormLinear(in_features, out_features, eps=1e-5, bias=True, **kwargs)
  :members: forward, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.LayerNormMLP(hidden_size, ffn_hidden_size, eps=1e-5, bias=True, **kwargs)
  :members: forward, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.DotProductAttention(num_attention_heads, kv_channels, **kwargs)
  :members: forward, set_context_parallel_group

.. autoapiclass:: transformer_engine.pytorch.MultiheadAttention(hidden_size, num_attention_heads, **kwargs)
  :members: forward, set_context_parallel_group, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads, **kwargs)
  :members: forward, set_context_parallel_group, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.dot_product_attention.inference.InferenceParams(max_batch_size, max_sequence_length)
  :members: reset, allocate_memory, pre_step, get_seqlens_pre_step, convert_paged_to_nonpaged, step

.. autoapiclass:: transformer_engine.pytorch.CudaRNGStatesTracker()
  :members: reset, get_states, set_states, add, fork


.. autoapifunction:: transformer_engine.pytorch.autocast

.. autoapifunction:: transformer_engine.pytorch.quantized_model_init

.. autoapifunction:: transformer_engine.pytorch.checkpoint


.. autoapifunction:: transformer_engine.pytorch.make_graphed_callables

.. autoapifunction:: transformer_engine.pytorch.get_cpu_offload_context

.. autoapifunction:: transformer_engine.pytorch.parallel_cross_entropy

Recipe availability
-------------------

.. autoapifunction:: transformer_engine.pytorch.is_fp8_available

.. autoapifunction:: transformer_engine.pytorch.is_mxfp8_available

.. autoapifunction:: transformer_engine.pytorch.is_fp8_block_scaling_available

.. autoapifunction:: transformer_engine.pytorch.is_nvfp4_available

.. autoapifunction:: transformer_engine.pytorch.is_bf16_available

.. autoapifunction:: transformer_engine.pytorch.get_cudnn_version

.. autoapifunction:: transformer_engine.pytorch.get_device_compute_capability

.. autoapifunction:: transformer_engine.pytorch.get_default_recipe

Mixture of Experts (MoE) functions
----------------------------------

.. autoapifunction:: transformer_engine.pytorch.moe_permute

.. autoapifunction:: transformer_engine.pytorch.moe_permute_with_probs

.. autoapifunction:: transformer_engine.pytorch.moe_unpermute

.. autoapifunction:: transformer_engine.pytorch.moe_sort_chunks_by_index

.. autoapifunction:: transformer_engine.pytorch.moe_sort_chunks_by_index_with_probs


Communication-computation overlap
---------------------------------

.. autoapifunction:: transformer_engine.pytorch.initialize_ub

.. autoapifunction:: transformer_engine.pytorch.destroy_ub

.. autoapiclass:: transformer_engine.pytorch.UserBufferQuantizationMode
  :members: FP8, NONE


Quantized tensors
-----------------

.. autoapiclass:: transformer_engine.pytorch.QuantizedTensorStorage
   :members: update_usage, prepare_for_saving, restore_from_saved

.. autoapiclass:: transformer_engine.pytorch.QuantizedTensor(shape, dtype, *, requires_grad=False, device=None)
   :members: dequantize, quantize_

.. autoapiclass:: transformer_engine.pytorch.Float8TensorStorage(data, fp8_scale_inv, fp8_dtype, data_transpose=None, quantizer=None)

.. autoapiclass:: transformer_engine.pytorch.MXFP8TensorStorage(rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv, fp8_dtype, quantizer)

.. autoapiclass:: transformer_engine.pytorch.Float8BlockwiseQTensorStorage(rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv, fp8_dtype, quantizer, is_2D_scaled, data_format)

.. autoapiclass:: transformer_engine.pytorch.NVFP4TensorStorage(rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv, amax_rowwise, amax_columnwise, fp4_dtype, quantizer)

.. autoapiclass:: transformer_engine.pytorch.Float8Tensor(shape, dtype, data, fp8_scale_inv, fp8_dtype, requires_grad=False, data_transpose=None, quantizer=None)

.. autoapiclass:: transformer_engine.pytorch.MXFP8Tensor(rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv, fp8_dtype, quantizer)

.. autoapiclass:: transformer_engine.pytorch.Float8BlockwiseQTensor(rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv, fp8_dtype, quantizer, is_2D_scaled, data_format)

.. autoapiclass:: transformer_engine.pytorch.NVFP4Tensor(rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv, amax_rowwise, amax_columnwise, fp4_dtype, quantizer)

Quantizers
----------

.. autoapiclass:: transformer_engine.pytorch.Quantizer(rowwise, columnwise)
   :members: update_quantized, quantize

.. autoapiclass:: transformer_engine.pytorch.Float8Quantizer(scale, amax, fp8_dtype, *, rowwise=True, columnwise=True)

.. autoapiclass:: transformer_engine.pytorch.Float8CurrentScalingQuantizer(fp8_dtype, device, *, rowwise=True, columnwise=True, **kwargs)

.. autoapiclass:: transformer_engine.pytorch.MXFP8Quantizer(fp8_dtype, *, rowwise=True, columnwise=True)

.. autoapiclass:: transformer_engine.pytorch.Float8BlockQuantizer(fp8_dtype, *, rowwise, columnwise, **kwargs)

.. autoapiclass:: transformer_engine.pytorch.NVFP4Quantizer(fp4_dtype, *, rowwise=True, columnwise=True, **kwargs)

Tensor saving and restoring functions
-------------------------------------

.. autoapifunction:: transformer_engine.pytorch.prepare_for_saving

.. autoapifunction:: transformer_engine.pytorch.restore_from_saved

Deprecated functions
--------------------

.. autoapifunction:: transformer_engine.pytorch.fp8_autocast

.. autoapifunction:: transformer_engine.pytorch.fp8_model_init
