/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "jax/csrc/extensions.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace jax {

NVTE_Fused_Attn_Backend GetFusedAttnBackend(DType q_dtype, DType kv_dtype,
                                            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                            NVTE_Mask_Type mask_type, float dropout_probability,
                                            size_t q_attn_heads, size_t kv_attn_heads,
                                            size_t q_max_seqlen, size_t kv_max_seqlen,
                                            size_t head_dim) {
  auto backend = nvte_get_fused_attn_backend(
      static_cast<NVTEDType>(q_dtype), static_cast<NVTEDType>(kv_dtype), qkv_layout, bias_type,
      mask_type, dropout_probability, q_attn_heads, kv_attn_heads, q_max_seqlen, kv_max_seqlen,
      head_dim);
  return backend;
}

/*
    NOTE: PrepareFusedAttnForwardAuxTensors unifies the auxiliary tensor pack logic from the fused
    attention forward kernels in:
        - common/fused_attn/fused_attn_f16_max512_seqlen.cu lines 594-634 and 773-812
        - common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu lines 1270-1281 and 1348-1359
*/
void PrepareFusedAttnForwardAuxTensors(NVTETensorPack *tensor_pack,
                                       const CustomCallFusedAttnDescriptor *desc,
                                       NVTE_Bias_Type bias_type, NVTE_Fused_Attn_Backend backend,
                                       void *softmax_buf, void *rng_state_buf = nullptr,
                                       void *bias_buf = nullptr) {
  auto input_batch = desc->input_batch;
  auto bias_batch = desc->bias_batch;
  auto attn_heads = desc->attn_heads;
  auto bias_heads = desc->bias_heads;
  auto q_max_seqlen = desc->q_max_seqlen;
  auto kv_max_seqlen = desc->kv_max_seqlen;

  // all backends need softmax but expect different shapes/dtypes
  // start with the max512 sequence length softmax shape/dtype and correct later
  tensor_pack->size = 1;
  Tensor *softmax_aux = reinterpret_cast<Tensor *>(tensor_pack->tensors[0]);
  softmax_aux->data.dptr = softmax_buf;
  softmax_aux->data.shape =
      std::vector<size_t>{input_batch, attn_heads, q_max_seqlen, kv_max_seqlen};
  softmax_aux->data.dtype = desc->dtype;

  // arbitrary sequence length backend needs the RNG state and a different shape/dtype softmax
  if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    tensor_pack->size = 2;
    Tensor *rng_state_aux = reinterpret_cast<Tensor *>(tensor_pack->tensors[1]);
    rng_state_aux->data.dptr = rng_state_buf;
    rng_state_aux->data.shape = std::vector<size_t>{2};
    rng_state_aux->data.dtype = DType::kInt64;
    // correct softmax shape/dtype
    softmax_aux->data.shape.at(3) = 1;  // {B,H,Qs,Ks} -> {B,H,Qs,1}
    softmax_aux->data.dtype = DType::kFloat32;

    // include bias if enabled
    if (bias_type != NVTE_Bias_Type::NVTE_NO_BIAS && bias_type != NVTE_Bias_Type::NVTE_ALIBI) {
      tensor_pack->size = 3;
      Tensor *bias_aux = reinterpret_cast<Tensor *>(tensor_pack->tensors[2]);
      bias_aux->data.dptr = bias_buf;
      bias_aux->data.shape =
          std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};
      bias_aux->data.dtype = desc->dtype;
    }
  }
}

/*
    NOTE: Backward fused attention kernels accept auxiliary tensors as explicit function arguments
    instead of an NVTETensorPack and nvte_fused_attn_bwd() API does all the logic for pulling the
    necessary tensors out of the tensor pack for the active kernel. That means we can just dump
    everything we got into the tensor pack and not worry about its sizing for the backward pass.

    TODO(Alp): Refactor the nvte_fused_attn_fwd() to work like nvte_fused_attn_bwd()?
*/
void PrepareFusedAttnBackwardAuxTensors(NVTETensorPack *tensor_pack,
                                        const CustomCallFusedAttnDescriptor *desc,
                                        NVTE_Fused_Attn_Backend backend, void *softmax_buf,
                                        void *rng_state_buf, void *bias_buf) {
  // Backward calls put everything into the tensor pack for every backend
  // so we set dummy bias_type and backend choices here to follow the correct code path
  auto dummy_bias_type = NVTE_Bias_Type::NVTE_POST_SCALE_BIAS;
  auto dummy_backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
  PrepareFusedAttnForwardAuxTensors(tensor_pack, desc, dummy_bias_type, dummy_backend, softmax_buf,
                                    rng_state_buf, bias_buf);

  // correct softmax shape for max512 sequence length kernel
  if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
    Tensor *softmax_aux = reinterpret_cast<Tensor *>(tensor_pack->tensors[0]);
    softmax_aux->data.shape.at(3) = desc->kv_max_seqlen;  // {B,H,Qs,1} -> {B,H,Qs,Ks}
    softmax_aux->data.dtype = desc->dtype;
  }
}

pybind11::tuple GetFusedAttnForwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training) {
  // For qkv_packed
  auto qkv_shape = std::vector<size_t>{input_batch * q_max_seqlen, 3, attn_heads, head_dim};
  auto qkv_tensor = TensorWrapper(nullptr, qkv_shape, dtype);

  // For kv_packed
  auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
  auto q_tensor = TensorWrapper(nullptr, q_shape, dtype);
  auto kv_shape = std::vector<size_t>{input_batch * kv_max_seqlen, 2, num_gqa_groups, head_dim};
  auto kv_tensor = TensorWrapper(nullptr, kv_shape, dtype);

  // For separate q, k, v
  auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
  auto k_tensor = TensorWrapper(nullptr, k_shape, dtype);
  auto v_shape = k_shape;
  auto v_tensor = TensorWrapper(nullptr, v_shape, dtype);

  auto bias_shape = std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};
  auto bias_tensor = TensorWrapper(nullptr, bias_shape, dtype);

  // F16 doesn't use this tensor
  auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);
  auto o_tensor = TensorWrapper(nullptr, q_shape, dtype);

  auto q_cu_seqlens_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);
  auto kv_cu_seqlens_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);

  auto dummy_rng_state_tensor = TensorWrapper(nullptr, std::vector<size_t>{2}, DType::kInt64);

  NVTETensorPack aux_output_tensors;
  nvte_tensor_pack_create(&aux_output_tensors);

  auto dummy_ragged_offset_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);
  TensorWrapper query_workspace_tensor;
  if (qkv_layout == NVTE_QKV_Layout::NVTE_BS3HD) {
    assert(q_max_seqlen == kv_max_seqlen);
    nvte_fused_attn_fwd_qkvpacked(qkv_tensor.data(), bias_tensor.data(), s_tensor.data(),
                                  o_tensor.data(), &aux_output_tensors, q_cu_seqlens_tensor.data(),
                                  dummy_ragged_offset_tensor.data(), dummy_rng_state_tensor.data(),
                                  q_max_seqlen, is_training, scaling_factor, dropout_probability,
                                  qkv_layout, bias_type, mask_type, query_workspace_tensor.data(),
                                  nullptr);
  } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD) {
    nvte_fused_attn_fwd_kvpacked(
        q_tensor.data(), kv_tensor.data(), bias_tensor.data(), s_tensor.data(), o_tensor.data(),
        &aux_output_tensors, q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
        dummy_ragged_offset_tensor.data(), dummy_ragged_offset_tensor.data(),
        dummy_rng_state_tensor.data(), q_max_seqlen, kv_max_seqlen, is_training, scaling_factor,
        dropout_probability, qkv_layout, bias_type, mask_type, query_workspace_tensor.data(),
        nullptr);
  } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
    nvte_fused_attn_fwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), bias_tensor.data(),
                        s_tensor.data(), o_tensor.data(), &aux_output_tensors,
                        q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
                        dummy_ragged_offset_tensor.data(), dummy_ragged_offset_tensor.data(),
                        dummy_rng_state_tensor.data(), q_max_seqlen, kv_max_seqlen, is_training,
                        scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
                        query_workspace_tensor.data(), nullptr);
  } else {
    NVTE_ERROR("Unsupported QKVLayout.");
  }

  auto workspace_shape = MakeShapeVector(query_workspace_tensor.shape());
  return pybind11::make_tuple(workspace_shape, query_workspace_tensor.dtype());
}

pybind11::tuple GetFusedAttnBackwardWorkspaceSizes(
    size_t batch_size, size_t q_max_seqlen, size_t kv_max_seqlen, size_t attn_heads,
    size_t num_gqa_groups, size_t head_dim, float scaling_factor, float dropout_probability,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype,
    bool is_training) {
  auto output_shape = std::vector<size_t>{batch_size * q_max_seqlen, attn_heads, head_dim};
  auto output_tensor = TensorWrapper(nullptr, output_shape, dtype);
  auto doutput_tensor = TensorWrapper(nullptr, output_shape, dtype);

  auto bias_shape = std::vector<size_t>{1, attn_heads, q_max_seqlen, kv_max_seqlen};
  auto dbias_tensor = TensorWrapper(nullptr, bias_shape, dtype);

  // F16 doesn't use s_tensor
  auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);

  auto q_cu_seqlens_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{batch_size + 1}, DType::kInt32);
  auto kv_cu_seqlens_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{batch_size + 1}, DType::kInt32);

  NVTETensorPack aux_input_tensors;
  nvte_tensor_pack_create(&aux_input_tensors);

  TensorWrapper query_workspace_tensor;

  auto dummy_ragged_offset_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{batch_size + 1}, DType::kInt32);
  if (qkv_layout == NVTE_QKV_Layout::NVTE_BS3HD) {
    assert(q_max_seqlen == kv_max_seqlen);
    auto qkv_shape = std::vector<size_t>{batch_size * q_max_seqlen, 3, attn_heads, head_dim};
    auto qkv_tensor = TensorWrapper(nullptr, qkv_shape, dtype);
    auto dqkv_tensor = TensorWrapper(nullptr, qkv_shape, dtype);
    nvte_fused_attn_bwd_qkvpacked(qkv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
                                  s_tensor.data(),  // not used for F16
                                  s_tensor.data(),  // not used for F16
                                  &aux_input_tensors, dqkv_tensor.data(), dbias_tensor.data(),
                                  q_cu_seqlens_tensor.data(), dummy_ragged_offset_tensor.data(),
                                  q_max_seqlen, scaling_factor, dropout_probability, qkv_layout,
                                  bias_type, mask_type, query_workspace_tensor.data(), nullptr);
  } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD) {
    auto q_shape = std::vector<size_t>{batch_size * q_max_seqlen, attn_heads, head_dim};
    auto q_tensor = TensorWrapper(nullptr, q_shape, dtype);
    auto dq_tensor = TensorWrapper(nullptr, q_shape, dtype);
    auto kv_shape = std::vector<size_t>{batch_size * kv_max_seqlen, 2, num_gqa_groups, head_dim};
    auto kv_tensor = TensorWrapper(nullptr, kv_shape, dtype);
    auto dkv_tensor = TensorWrapper(nullptr, kv_shape, dtype);
    nvte_fused_attn_bwd_kvpacked(
        q_tensor.data(), kv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
        s_tensor.data(),  // not used for F16
        s_tensor.data(),  // not used for F16
        &aux_input_tensors, dq_tensor.data(), dkv_tensor.data(), dbias_tensor.data(),
        q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(), dummy_ragged_offset_tensor.data(),
        dummy_ragged_offset_tensor.data(), q_max_seqlen, kv_max_seqlen, scaling_factor,
        dropout_probability, qkv_layout, bias_type, mask_type, query_workspace_tensor.data(),
        nullptr);
  } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
    auto q_shape = std::vector<size_t>{batch_size * q_max_seqlen, attn_heads, head_dim};
    auto q_tensor = TensorWrapper(nullptr, q_shape, dtype);
    auto dq_tensor = TensorWrapper(nullptr, q_shape, dtype);
    auto k_shape = std::vector<size_t>{batch_size * kv_max_seqlen, num_gqa_groups, head_dim};
    auto k_tensor = TensorWrapper(nullptr, k_shape, dtype);
    auto dk_tensor = TensorWrapper(nullptr, k_shape, dtype);
    auto v_shape = k_shape;
    auto v_tensor = TensorWrapper(nullptr, v_shape, dtype);
    auto dv_tensor = TensorWrapper(nullptr, v_shape, dtype);
    nvte_fused_attn_bwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), output_tensor.data(),
                        doutput_tensor.data(),
                        s_tensor.data(),  // not used for F16
                        s_tensor.data(),  // not used for F16
                        &aux_input_tensors, dq_tensor.data(), dk_tensor.data(), dv_tensor.data(),
                        dbias_tensor.data(), q_cu_seqlens_tensor.data(),
                        kv_cu_seqlens_tensor.data(), dummy_ragged_offset_tensor.data(),
                        dummy_ragged_offset_tensor.data(), q_max_seqlen, kv_max_seqlen,
                        scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
                        query_workspace_tensor.data(), nullptr);
  } else {
    NVTE_ERROR("Unsupported QKVLayout.");
  }

  auto workspace_shape = MakeShapeVector(query_workspace_tensor.shape());
  return pybind11::make_tuple(workspace_shape, query_workspace_tensor.dtype());
}

void FusedAttnForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  const CustomCallFusedAttnDescriptor &descriptor =
      *UnpackOpaque<CustomCallFusedAttnDescriptor>(opaque, opaque_len);

  /* Input buffers from XLA */
  /* Buffers[0-2] are q, k, v, which are parsed later for different qkv_layout */
  void *bias = buffers[3];
  void *q_cu_seqlens = buffers[4];
  void *kv_cu_seqlens = buffers[5];
  void *seed = buffers[6];

  /* Output buffer from XLA */
  void *output = buffers[7];
  void *softmax_aux = buffers[8];
  void *rng_state = buffers[9];
  void *workspace = buffers[10];

  /* Descriptor */
  auto input_batch = descriptor.input_batch;
  auto bias_batch = descriptor.bias_batch;
  auto q_max_seqlen = descriptor.q_max_seqlen;
  auto kv_max_seqlen = descriptor.kv_max_seqlen;
  auto attn_heads = descriptor.attn_heads;
  auto num_gqa_groups = descriptor.num_gqa_groups;
  auto bias_heads = descriptor.bias_heads;
  auto head_dim = descriptor.head_dim;
  auto scaling_factor = descriptor.scaling_factor;
  auto dropout_probability = descriptor.dropout_probability;
  auto bias_type = descriptor.bias_type;
  auto mask_type = descriptor.mask_type;
  auto qkv_layout = descriptor.qkv_layout;
  auto dtype = descriptor.dtype;

  /* Input tensors */
  auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
  auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
  auto v_shape = k_shape;
  auto bias_shape = std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};
  auto bias_tensor = TensorWrapper(bias, bias_shape, dtype);

  /* Output tensors */
  auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);  // not used in F16
  auto o_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
  auto o_tensor = TensorWrapper(output, o_shape, dtype);
  auto q_cu_seqlens_tensor =
      TensorWrapper(q_cu_seqlens, std::vector<size_t>{input_batch + 1}, DType::kInt32);
  auto kv_cu_seqlens_tensor =
      TensorWrapper(kv_cu_seqlens, std::vector<size_t>{input_batch + 1}, DType::kInt32);

  /* Prepare RNG state */
  auto rng_state_tensor = TensorWrapper(rng_state, std::vector<size_t>{2}, DType::kInt64);
  auto backend =
      nvte_get_fused_attn_backend(static_cast<NVTEDType>(dtype), static_cast<NVTEDType>(dtype),
                                  qkv_layout, bias_type, mask_type, dropout_probability, attn_heads,
                                  num_gqa_groups, q_max_seqlen, kv_max_seqlen, head_dim);
  PopulateRngStateAsync(rng_state, seed, q_max_seqlen, kv_max_seqlen, backend, stream);

  /* Auxiliary tensors (to be propagated to the backward pass later) */
  NVTETensorPack aux_output_tensors;
  nvte_tensor_pack_create(&aux_output_tensors);
  PrepareFusedAttnForwardAuxTensors(&aux_output_tensors, &descriptor, bias_type, backend,
                                    softmax_aux);

  /* cuDNN workspace */
  auto workspace_tensor = TensorWrapper(workspace, std::vector<size_t>{descriptor.wkspace_size},
                                        descriptor.wkspace_dtype);

  auto dummy_ragged_offset_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);
  /* Call the underly NVTE API */
  if (qkv_layout == NVTE_QKV_Layout::NVTE_BS3HD) {
    auto qkv = buffers[0];
    auto qkv_shape = std::vector<size_t>{input_batch * q_max_seqlen, 3, attn_heads, head_dim};
    auto qkv_tensor = TensorWrapper(qkv, qkv_shape, dtype);
    nvte_fused_attn_fwd_qkvpacked(
        qkv_tensor.data(), bias_tensor.data(), s_tensor.data(), o_tensor.data(),
        &aux_output_tensors, q_cu_seqlens_tensor.data(), dummy_ragged_offset_tensor.data(),
        rng_state_tensor.data(), q_max_seqlen, descriptor.is_training, descriptor.scaling_factor,
        dropout_probability, qkv_layout, bias_type, mask_type, workspace_tensor.data(), stream);
  } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD) {
    auto q = buffers[0];
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto q_tensor = TensorWrapper(q, q_shape, dtype);
    auto kv = buffers[1];
    auto kv_shape = std::vector<size_t>{input_batch * kv_max_seqlen, 2, num_gqa_groups, head_dim};
    auto kv_tensor = TensorWrapper(kv, kv_shape, dtype);
    nvte_fused_attn_fwd_kvpacked(
        q_tensor.data(), kv_tensor.data(), bias_tensor.data(), s_tensor.data(), o_tensor.data(),
        &aux_output_tensors, q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
        dummy_ragged_offset_tensor.data(), dummy_ragged_offset_tensor.data(),
        rng_state_tensor.data(), q_max_seqlen, kv_max_seqlen, descriptor.is_training,
        scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
        workspace_tensor.data(), stream);
  } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
    auto q = buffers[0];
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto q_tensor = TensorWrapper(q, q_shape, dtype);
    auto k = buffers[1];
    auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
    auto k_tensor = TensorWrapper(k, k_shape, dtype);
    auto v = buffers[2];
    auto v_shape = k_shape;
    auto v_tensor = TensorWrapper(v, v_shape, dtype);
    nvte_fused_attn_fwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), bias_tensor.data(),
                        s_tensor.data(), o_tensor.data(), &aux_output_tensors,
                        q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
                        dummy_ragged_offset_tensor.data(), dummy_ragged_offset_tensor.data(),
                        rng_state_tensor.data(), q_max_seqlen, kv_max_seqlen,
                        descriptor.is_training, scaling_factor, dropout_probability, qkv_layout,
                        bias_type, mask_type, workspace_tensor.data(), stream);
  } else {
    NVTE_ERROR("Unsupported qkv_layout.");
  }

  nvte_tensor_pack_destroy(&aux_output_tensors);
}

pybind11::tuple GetFusedAttnBackwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training) {
  auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
  auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
  auto v_shape = k_shape;
  auto output_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
  auto bias_shape = std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};

  auto q_tensor = TensorWrapper(nullptr, q_shape, dtype);
  auto k_tensor = TensorWrapper(nullptr, k_shape, dtype);
  auto v_tensor = TensorWrapper(nullptr, v_shape, dtype);
  auto doutput_tensor = TensorWrapper(nullptr, output_shape, dtype);
  auto output_tensor = TensorWrapper(nullptr, output_shape, dtype);
  // F16 doesn't use this tensor
  auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);

  auto dq_tensor = TensorWrapper(nullptr, q_shape, dtype);
  auto dk_tensor = TensorWrapper(nullptr, k_shape, dtype);
  auto dv_tensor = TensorWrapper(nullptr, v_shape, dtype);
  auto dbias_tensor = TensorWrapper(nullptr, bias_shape, dtype);

  auto q_cu_seqlens_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);
  auto kv_cu_seqlens_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);

  NVTETensorPack aux_input_tensors;
  nvte_tensor_pack_create(&aux_input_tensors);

  TensorWrapper query_workspace_tensor;
  auto dummy_ragged_offset_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);
  nvte_fused_attn_bwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), output_tensor.data(),
                      doutput_tensor.data(),
                      s_tensor.data(),  // not used for F16
                      s_tensor.data(),  // not used for F16
                      &aux_input_tensors, dq_tensor.data(), dk_tensor.data(), dv_tensor.data(),
                      dbias_tensor.data(), q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
                      dummy_ragged_offset_tensor.data(), dummy_ragged_offset_tensor.data(),
                      q_max_seqlen, kv_max_seqlen, scaling_factor, dropout_probability, qkv_layout,
                      bias_type, mask_type, query_workspace_tensor.data(), nullptr);

  auto work_shape = MakeShapeVector(query_workspace_tensor.shape());
  return pybind11::make_tuple(work_shape, query_workspace_tensor.dtype());
}

void FusedAttnBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  const CustomCallFusedAttnDescriptor &descriptor =
      *UnpackOpaque<CustomCallFusedAttnDescriptor>(opaque, opaque_len);

  /* Input buffers from XLA */
  /* Buffers[0-2] are q, k, v, which are parsed later for different qkv_layout */
  void *bias = buffers[3];
  void *softmax_aux = buffers[4];
  void *rng_state = buffers[5];
  void *output = buffers[6];
  void *doutput = buffers[7];
  void *q_cu_seqlens = buffers[8];
  void *kv_cu_seqlens = buffers[9];

  /* Output buffer from XLA */
  /* Buffers[10-12] are dq, dk, dv, which are parsed later for different qkv_layout */
  void *dbias = buffers[13];
  void *workspace = buffers[14];

  /* Descriptor */
  auto input_batch = descriptor.input_batch;
  auto bias_batch = descriptor.bias_batch;
  auto q_max_seqlen = descriptor.q_max_seqlen;
  auto kv_max_seqlen = descriptor.kv_max_seqlen;
  auto attn_heads = descriptor.attn_heads;
  auto num_gqa_groups = descriptor.num_gqa_groups;
  auto bias_heads = descriptor.bias_heads;
  auto head_dim = descriptor.head_dim;
  auto scaling_factor = descriptor.scaling_factor;
  auto dropout_probability = descriptor.dropout_probability;
  auto bias_type = descriptor.bias_type;
  auto mask_type = descriptor.mask_type;
  auto qkv_layout = descriptor.qkv_layout;
  auto dtype = descriptor.dtype;

  /* Input tensors */
  auto output_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
  auto bias_shape = std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};
  auto output_tensor = TensorWrapper(output, output_shape, dtype);
  auto doutput_tensor = TensorWrapper(doutput, output_shape, dtype);

  /* Output tensors */
  auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);  // not used in F16
  auto dbias_tensor = TensorWrapper(dbias, bias_shape, dtype);
  auto q_cu_seqlens_tensor =
      TensorWrapper(q_cu_seqlens, std::vector<size_t>{input_batch + 1}, DType::kInt32);
  auto kv_cu_seqlens_tensor =
      TensorWrapper(kv_cu_seqlens, std::vector<size_t>{input_batch + 1}, DType::kInt32);

  /* Auxiliary tensors (propagated from the forward pass) */
  NVTETensorPack aux_input_tensors;
  nvte_tensor_pack_create(&aux_input_tensors);
  auto backend =
      nvte_get_fused_attn_backend(static_cast<NVTEDType>(dtype), static_cast<NVTEDType>(dtype),
                                  qkv_layout, bias_type, mask_type, dropout_probability, attn_heads,
                                  num_gqa_groups, q_max_seqlen, kv_max_seqlen, head_dim);
  PrepareFusedAttnBackwardAuxTensors(&aux_input_tensors, &descriptor, backend, softmax_aux,
                                     rng_state, bias);

  /* cuDNN workspace */
  auto wkspace_size = std::vector<size_t>{descriptor.wkspace_size};
  auto wkspace_dtype = descriptor.wkspace_dtype;
  auto workspace_tensor = TensorWrapper(workspace, wkspace_size, wkspace_dtype);

  auto dummy_ragged_offset_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);
  /* Call the underly NVTE API */
  if (qkv_layout == NVTE_QKV_Layout::NVTE_BS3HD) {
    auto qkv = buffers[0];
    auto qkv_shape = std::vector<size_t>{input_batch * q_max_seqlen, 3, attn_heads, head_dim};
    auto qkv_tensor = TensorWrapper(qkv, qkv_shape, dtype);
    auto dqkv = buffers[10];
    auto dqkv_tensor = TensorWrapper(dqkv, qkv_shape, dtype);
    nvte_fused_attn_bwd_qkvpacked(qkv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
                                  s_tensor.data(),  // not used for F16
                                  s_tensor.data(),  // not used for F16
                                  &aux_input_tensors, dqkv_tensor.data(), dbias_tensor.data(),
                                  q_cu_seqlens_tensor.data(), dummy_ragged_offset_tensor.data(),
                                  q_max_seqlen, scaling_factor, dropout_probability, qkv_layout,
                                  bias_type, mask_type, workspace_tensor.data(), stream);
  } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD) {
    auto q = buffers[0];
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto q_tensor = TensorWrapper(q, q_shape, dtype);
    auto kv = buffers[1];
    auto kv_shape = std::vector<size_t>{input_batch * kv_max_seqlen, 2, num_gqa_groups, head_dim};
    auto kv_tensor = TensorWrapper(kv, kv_shape, dtype);
    auto dq = buffers[10];
    auto dq_tensor = TensorWrapper(dq, q_shape, dtype);
    auto dkv = buffers[11];
    auto dkv_tensor = TensorWrapper(dkv, kv_shape, dtype);
    nvte_fused_attn_bwd_kvpacked(
        q_tensor.data(), kv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
        s_tensor.data(),  // not used for F16
        s_tensor.data(),  // not used for F16
        &aux_input_tensors, dq_tensor.data(), dkv_tensor.data(), dbias_tensor.data(),
        q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(), dummy_ragged_offset_tensor.data(),
        dummy_ragged_offset_tensor.data(), q_max_seqlen, kv_max_seqlen, scaling_factor,
        dropout_probability, qkv_layout, bias_type, mask_type, workspace_tensor.data(), stream);
  } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
    auto q = buffers[0];
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto q_tensor = TensorWrapper(q, q_shape, dtype);
    auto k = buffers[1];
    auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
    auto k_tensor = TensorWrapper(k, k_shape, dtype);
    auto v = buffers[2];
    auto v_shape = k_shape;
    auto v_tensor = TensorWrapper(v, v_shape, dtype);
    auto dq = buffers[10];
    auto dq_tensor = TensorWrapper(dq, q_shape, dtype);
    auto dk = buffers[11];
    auto dk_tensor = TensorWrapper(dk, k_shape, dtype);
    auto dv = buffers[12];
    auto dv_tensor = TensorWrapper(dv, v_shape, dtype);
    nvte_fused_attn_bwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), output_tensor.data(),
                        doutput_tensor.data(),
                        s_tensor.data(),  // not used for F16
                        s_tensor.data(),  // not used for F16
                        &aux_input_tensors, dq_tensor.data(), dk_tensor.data(), dv_tensor.data(),
                        dbias_tensor.data(), q_cu_seqlens_tensor.data(),
                        kv_cu_seqlens_tensor.data(), dummy_ragged_offset_tensor.data(),
                        dummy_ragged_offset_tensor.data(), q_max_seqlen, kv_max_seqlen,
                        scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
                        workspace_tensor.data(), stream);
  } else {
    NVTE_ERROR("Unsupported qkv_layout.");
  }

  nvte_tensor_pack_destroy(&aux_input_tensors);
}

}  // namespace jax
}  // namespace transformer_engine
