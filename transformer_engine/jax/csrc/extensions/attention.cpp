/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace jax {

NVTE_Fused_Attn_Backend GetFusedAttnBackend(DType q_dtype, DType kv_dtype,
                                            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                            NVTE_Mask_Type mask_type, float dropout_probability,
                                            size_t q_attn_heads, size_t kv_attn_heads,
                                            size_t q_max_seqlen, size_t kv_max_seqlen,
                                            size_t head_dim, int64_t window_size_left,
                                            int64_t window_size_right) {
  auto backend = nvte_get_fused_attn_backend(
      static_cast<NVTEDType>(q_dtype), static_cast<NVTEDType>(kv_dtype), qkv_layout, bias_type,
      mask_type, dropout_probability, q_attn_heads, kv_attn_heads, q_max_seqlen, kv_max_seqlen,
      head_dim, head_dim, window_size_left, window_size_right);
  return backend;
}

/*
    NOTE: PrepareFusedAttnForwardAuxTensors unifies the auxiliary tensor pack logic from the fused
    attention forward kernels in:
        - common/fused_attn/fused_attn_f16_max512_seqlen.cu lines 594-634 and 773-812
        - common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu lines 1270-1281 and 1348-1359
*/
void PrepareFusedAttnForwardAuxTensors(NVTETensorPack *tensor_pack, const size_t input_batch,
                                       const size_t bias_batch, const size_t attn_heads,
                                       const size_t bias_heads, const size_t q_max_seqlen,
                                       const size_t kv_max_seqlen, DType dtype,
                                       NVTE_Bias_Type bias_type, NVTE_Fused_Attn_Backend backend,
                                       void *softmax_buf, void *rng_state_buf = nullptr,
                                       void *bias_buf = nullptr) {
  // all backends need softmax but expect different shapes/dtypes
  // start with the max512 sequence length softmax shape/dtype and correct later
  tensor_pack->size = 1;
  Tensor *softmax_aux = reinterpret_cast<Tensor *>(tensor_pack->tensors[0]);
  softmax_aux->data.dptr = softmax_buf;
  softmax_aux->data.shape =
      std::vector<size_t>{input_batch, attn_heads, q_max_seqlen, kv_max_seqlen};
  softmax_aux->data.dtype = dtype;

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
      bias_aux->data.dtype = dtype;
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
void PrepareFusedAttnBackwardAuxTensors(NVTETensorPack *tensor_pack, const size_t input_batch,
                                        const size_t bias_batch, const size_t attn_heads,
                                        const size_t bias_heads, const size_t q_max_seqlen,
                                        const size_t kv_max_seqlen, DType dtype,
                                        NVTE_Fused_Attn_Backend backend, void *softmax_buf,
                                        void *rng_state_buf, void *bias_buf) {
  // Backward calls put everything into the tensor pack for every backend
  // so we set dummy bias_type and backend choices here to follow the correct code path
  auto dummy_bias_type = NVTE_Bias_Type::NVTE_POST_SCALE_BIAS;
  auto dummy_backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
  PrepareFusedAttnForwardAuxTensors(tensor_pack, input_batch, bias_batch, attn_heads, bias_heads,
                                    q_max_seqlen, kv_max_seqlen, dtype, dummy_bias_type,
                                    dummy_backend, softmax_buf, rng_state_buf, bias_buf);

  // correct softmax shape for max512 sequence length kernel
  if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
    Tensor *softmax_aux = reinterpret_cast<Tensor *>(tensor_pack->tensors[0]);
    softmax_aux->data.shape.at(3) = kv_max_seqlen;  // {B,H,Qs,1} -> {B,H,Qs,Ks}
    softmax_aux->data.dtype = dtype;
  }
}

pybind11::tuple GetFusedAttnForwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training,
    size_t max_segments_per_seq, int64_t window_size_left, int64_t window_size_right) {
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

  auto dummy_rng_state_tensor = TensorWrapper(nullptr, std::vector<size_t>{2}, DType::kInt64);

  NVTETensorPack aux_output_tensors;
  nvte_tensor_pack_create(&aux_output_tensors);

  TensorWrapper query_workspace_tensor;
  auto layout_group = nvte_get_qkv_layout_group(qkv_layout);
  auto is_ragged = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD;
  // It is a WAR to pre-create all possible cuDNN graph at the JIT compile time
  size_t max_num_segments = is_ragged ? input_batch * max_segments_per_seq : input_batch;
  size_t min_num_segments = input_batch;
  auto cudnn_runtime_version = cudnnGetVersion();
  if (is_ragged && cudnn_runtime_version >= 90300) {
    // For cuDNN < 9.3.0, it requires to run all possible seqlens to address act_seqlen = 0
    min_num_segments = input_batch * max_segments_per_seq;
  }
  for (auto num_segments = min_num_segments; num_segments <= max_num_segments; ++num_segments) {
    // the last one is the largest which will be the returned workspace size
    auto q_cu_seqlens_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{num_segments + 1}, DType::kInt32);
    auto kv_cu_seqlens_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{num_segments + 1}, DType::kInt32);
    auto ragged_offset_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{num_segments + 1}, DType::kInt32);
    if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
      NVTE_CHECK(q_max_seqlen == kv_max_seqlen, "q_max_seqlen must equal to kv_max_seqlen");
      nvte_fused_attn_fwd_qkvpacked(
          qkv_tensor.data(), bias_tensor.data(), s_tensor.data(), o_tensor.data(),
          &aux_output_tensors, q_cu_seqlens_tensor.data(), ragged_offset_tensor.data(),
          dummy_rng_state_tensor.data(), q_max_seqlen, is_training, scaling_factor,
          dropout_probability, qkv_layout, bias_type, mask_type, window_size_left,
          window_size_right, query_workspace_tensor.data(), nullptr);
    } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
      nvte_fused_attn_fwd_kvpacked(
          q_tensor.data(), kv_tensor.data(), bias_tensor.data(), s_tensor.data(), o_tensor.data(),
          &aux_output_tensors, q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
          ragged_offset_tensor.data(), ragged_offset_tensor.data(), dummy_rng_state_tensor.data(),
          q_max_seqlen, kv_max_seqlen, is_training, scaling_factor, dropout_probability, qkv_layout,
          bias_type, mask_type, window_size_left, window_size_right, query_workspace_tensor.data(),
          nullptr);
    } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_HD_HD) {
      nvte_fused_attn_fwd(
          q_tensor.data(), k_tensor.data(), v_tensor.data(), bias_tensor.data(), s_tensor.data(),
          o_tensor.data(), &aux_output_tensors, q_cu_seqlens_tensor.data(),
          kv_cu_seqlens_tensor.data(), ragged_offset_tensor.data(), ragged_offset_tensor.data(),
          dummy_rng_state_tensor.data(), q_max_seqlen, kv_max_seqlen, is_training, scaling_factor,
          dropout_probability, qkv_layout, bias_type, mask_type, window_size_left,
          window_size_right, query_workspace_tensor.data(), nullptr);
    } else {
      NVTE_ERROR("Unsupported QKVLayout.");
    }
  }

  auto workspace_shape = MakeShapeVector(query_workspace_tensor.shape());
  return pybind11::make_tuple(workspace_shape, query_workspace_tensor.dtype());
}

#define FUSED_ATTN_IMPL_COMMON_BLOCK                                                            \
  auto is_ragged = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD;                \
  auto bias_shape = std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};   \
  size_t num_segments = input_batch;                                                            \
  if (is_ragged) {                                                                              \
    auto cudnn_runtime_version = cudnnGetVersion();                                             \
    if (cudnn_runtime_version >= 90300) {                                                       \
      num_segments = input_batch * max_segments_per_seq;                                        \
    } else {                                                                                    \
      size_t runtime_num_segments_q =                                                           \
          GetRuntimeNumSegments(q_cu_seqlens, workspace, input_batch * q_max_seqlen, stream);   \
      size_t runtime_num_segments_kv =                                                          \
          GetRuntimeNumSegments(kv_cu_seqlens, workspace, input_batch * kv_max_seqlen, stream); \
      NVTE_CHECK(runtime_num_segments_q == runtime_num_segments_kv);                            \
      NVTE_CHECK(runtime_num_segments_q <= input_batch * max_segments_per_seq);                 \
      num_segments = runtime_num_segments_q;                                                    \
    }                                                                                           \
  }                                                                                             \
  std::vector<size_t> seq_shape{num_segments + 1};                                              \
  auto q_cu_seqlens_tensor = TensorWrapper(q_cu_seqlens, seq_shape, DType::kInt32);             \
  auto kv_cu_seqlens_tensor = TensorWrapper(kv_cu_seqlens, seq_shape, DType::kInt32);           \
  auto q_seq_offsets_tensor = TensorWrapper(q_seq_offsets, seq_shape, DType::kInt32);           \
  auto k_seq_offsets_tensor = TensorWrapper(k_seq_offsets, seq_shape, DType::kInt32);           \
  auto workspace_tensor =                                                                       \
      TensorWrapper(workspace, std::vector<size_t>{wkspace_size}, wkspace_dtype);               \
  auto layout_group = nvte_get_qkv_layout_group(qkv_layout);

static void FusedAttnForwardImpl(
    cudaStream_t stream, void *q, void *k, void *v, void *bias, void *seed, void *q_cu_seqlens,
    void *kv_cu_seqlens, void *q_seq_offsets, void *k_seq_offsets, void *output, void *softmax_aux,
    void *rng_state, void *workspace, size_t input_batch, size_t bias_batch, size_t q_max_seqlen,
    size_t kv_max_seqlen, size_t attn_heads, size_t num_gqa_groups, size_t bias_heads,
    size_t head_dim, size_t max_segments_per_seq, size_t wkspace_size, float scaling_factor,
    float dropout_probability, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    NVTE_QKV_Layout qkv_layout, DType dtype, DType wkspace_dtype, bool is_training,
    bool deterministic, int64_t window_size_left, int64_t window_size_right) {
  FUSED_ATTN_IMPL_COMMON_BLOCK;

  /* Input tensors */
  auto bias_tensor = TensorWrapper(bias, bias_shape, dtype);

  if (is_ragged) {
    auto output_size = input_batch * q_max_seqlen * attn_heads * head_dim;
    cudaMemsetAsync(output, 0, output_size * typeToSize(dtype), stream);
  }

  /* Output tensors */
  auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);  // not used in F16
  auto o_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
  auto o_tensor = TensorWrapper(output, o_shape, dtype);

  /* Prepare RNG state */
  auto rng_state_tensor = TensorWrapper(rng_state, std::vector<size_t>{2}, DType::kInt64);
  auto backend = nvte_get_fused_attn_backend(
      static_cast<NVTEDType>(dtype), static_cast<NVTEDType>(dtype), qkv_layout, bias_type,
      mask_type, dropout_probability, attn_heads, num_gqa_groups, q_max_seqlen, kv_max_seqlen,
      head_dim, head_dim, window_size_left, window_size_right);
  PopulateRngStateAsync(rng_state, seed, q_max_seqlen, kv_max_seqlen, backend, stream);

  /* Auxiliary tensors (to be propagated to the backward pass later) */
  NVTETensorPack aux_output_tensors;
  nvte_tensor_pack_create(&aux_output_tensors);
  PrepareFusedAttnForwardAuxTensors(&aux_output_tensors, input_batch, bias_batch, attn_heads,
                                    bias_heads, q_max_seqlen, kv_max_seqlen, dtype, bias_type,
                                    backend, softmax_aux);

  /* Call the underlying NVTE API */
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    auto qkv_shape = std::vector<size_t>{input_batch * q_max_seqlen, 3, attn_heads, head_dim};
    auto qkv_tensor = TensorWrapper(q, qkv_shape, dtype);
    nvte_fused_attn_fwd_qkvpacked(qkv_tensor.data(), bias_tensor.data(), s_tensor.data(),
                                  o_tensor.data(), &aux_output_tensors, q_cu_seqlens_tensor.data(),
                                  q_seq_offsets_tensor.data(), rng_state_tensor.data(),
                                  q_max_seqlen, is_training, scaling_factor, dropout_probability,
                                  qkv_layout, bias_type, mask_type, window_size_left,
                                  window_size_right, workspace_tensor.data(), stream);
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto kv_shape = std::vector<size_t>{input_batch * kv_max_seqlen, 2, num_gqa_groups, head_dim};
    auto q_tensor = TensorWrapper(q, q_shape, dtype);
    auto kv_tensor = TensorWrapper(k, kv_shape, dtype);
    nvte_fused_attn_fwd_kvpacked(
        q_tensor.data(), kv_tensor.data(), bias_tensor.data(), s_tensor.data(), o_tensor.data(),
        &aux_output_tensors, q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
        q_seq_offsets_tensor.data(), k_seq_offsets_tensor.data(), rng_state_tensor.data(),
        q_max_seqlen, kv_max_seqlen, is_training, scaling_factor, dropout_probability, qkv_layout,
        bias_type, mask_type, window_size_left, window_size_right, workspace_tensor.data(), stream);
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_HD_HD) {
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
    auto v_shape = k_shape;
    auto q_tensor = TensorWrapper(q, q_shape, dtype);
    auto k_tensor = TensorWrapper(k, k_shape, dtype);
    auto v_tensor = TensorWrapper(v, v_shape, dtype);
    nvte_fused_attn_fwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), bias_tensor.data(),
                        s_tensor.data(), o_tensor.data(), &aux_output_tensors,
                        q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
                        q_seq_offsets_tensor.data(), k_seq_offsets_tensor.data(),
                        rng_state_tensor.data(), q_max_seqlen, kv_max_seqlen, is_training,
                        scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
                        window_size_left, window_size_right, workspace_tensor.data(), stream);
  } else {
    NVTE_ERROR("Unsupported qkv_layout.");
  }

  nvte_tensor_pack_destroy(&aux_output_tensors);
}

void FusedAttnForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  const CustomCallFusedAttnDescriptor &descriptor =
      *UnpackOpaque<CustomCallFusedAttnDescriptor>(opaque, opaque_len);
  auto is_ragged = nvte_get_qkv_format(descriptor.qkv_layout) == NVTE_QKV_Format::NVTE_THD;

  /* Input buffers from XLA */
  void *q = buffers[0];
  void *k = buffers[1];
  void *v = buffers[2];
  void *bias = buffers[3];
  void *seed = buffers[4];
  void *q_cu_seqlens = buffers[5];
  void *kv_cu_seqlens = buffers[6];
  void *q_seq_offsets = is_ragged ? buffers[7] : nullptr;
  void *k_seq_offsets = is_ragged ? buffers[8] : nullptr;

  /* Output buffer from XLA */
  void *output = buffers[9];
  void *softmax_aux = buffers[10];
  void *rng_state = buffers[11];
  void *workspace = buffers[12];

  FusedAttnForwardImpl(
      stream, q, k, v, bias, seed, q_cu_seqlens, kv_cu_seqlens, q_seq_offsets, k_seq_offsets,
      output, softmax_aux, rng_state, workspace, descriptor.input_batch, descriptor.bias_batch,
      descriptor.q_max_seqlen, descriptor.kv_max_seqlen, descriptor.attn_heads,
      descriptor.num_gqa_groups, descriptor.bias_heads, descriptor.head_dim,
      descriptor.max_segments_per_seq, descriptor.wkspace_size, descriptor.scaling_factor,
      descriptor.dropout_probability, descriptor.bias_type, descriptor.mask_type,
      descriptor.qkv_layout, descriptor.dtype, descriptor.wkspace_dtype, descriptor.is_training,
      descriptor.deterministic, descriptor.window_size_left, descriptor.window_size_right);
}

#define FUSED_ATTN_FFI_GET_ATTRS                                                        \
  size_t input_batch = get_attr_value<int64_t>(attrs, "input_batch");                   \
  size_t bias_batch = get_attr_value<int64_t>(attrs, "bias_batch");                     \
  size_t q_max_seqlen = get_attr_value<int64_t>(attrs, "q_max_seqlen");                 \
  size_t kv_max_seqlen = get_attr_value<int64_t>(attrs, "kv_max_seqlen");               \
  size_t attn_heads = get_attr_value<int64_t>(attrs, "attn_heads");                     \
  size_t num_gqa_groups = get_attr_value<int64_t>(attrs, "num_gqa_groups");             \
  size_t bias_heads = get_attr_value<int64_t>(attrs, "bias_heads");                     \
  size_t head_dim = get_attr_value<int64_t>(attrs, "head_dim");                         \
  size_t max_segments_per_seq = get_attr_value<int64_t>(attrs, "max_segments_per_seq"); \
  auto window_size_left = get_attr_value<int64_t>(attrs, "window_size_left");           \
  auto window_size_right = get_attr_value<int64_t>(attrs, "window_size_right");         \
  float scaling_factor = get_attr_value<double>(attrs, "scaling_factor");               \
  float dropout_probability = get_attr_value<double>(attrs, "dropout_probability");     \
  NVTE_Bias_Type bias_type =                                                            \
      static_cast<NVTE_Bias_Type>(get_attr_value<int64_t>(attrs, "bias_type"));         \
  NVTE_Mask_Type mask_type =                                                            \
      static_cast<NVTE_Mask_Type>(get_attr_value<int64_t>(attrs, "mask_type"));         \
  NVTE_QKV_Layout qkv_layout =                                                          \
      static_cast<NVTE_QKV_Layout>(get_attr_value<int64_t>(attrs, "qkv_layout"));       \
  bool is_training = get_attr_value<bool>(attrs, "is_training");                        \
  bool deterministic = get_attr_value<bool>(attrs, "deterministic");                    \
  auto is_ragged = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD;        \
  size_t wkspace_size = product(workspace_buf->dimensions());                           \
  DType dtype = convert_ffi_datatype_to_te_dtype(q_buf.element_type());                 \
  DType wkspace_dtype = convert_ffi_datatype_to_te_dtype(workspace_buf->element_type());

Error_Type FusedAttnForwardFFI(cudaStream_t stream, Buffer_Type q_buf, Buffer_Type k_buf,
                               Buffer_Type v_buf, Buffer_Type bias_buf, Buffer_Type seed_buf,
                               Buffer_Type q_cu_seqlens_buf, Buffer_Type kv_cu_seqlens_buf,
                               Buffer_Type q_seq_offsets_buf, Buffer_Type k_seq_offsets_buf,
                               Variadic_Buffer_Type _unused_args, Result_Type output_buf,
                               Result_Type softmax_aux_buf, Result_Type rng_state_buf,
                               Result_Type workspace_buf, Dictionary attrs) {
  FUSED_ATTN_FFI_GET_ATTRS;

  FusedAttnForwardImpl(
      stream, q_buf.untyped_data(), k_buf.untyped_data(), v_buf.untyped_data(),
      bias_buf.untyped_data(), seed_buf.untyped_data(), q_cu_seqlens_buf.untyped_data(),
      kv_cu_seqlens_buf.untyped_data(), is_ragged ? q_seq_offsets_buf.untyped_data() : nullptr,
      is_ragged ? k_seq_offsets_buf.untyped_data() : nullptr, output_buf->untyped_data(),
      softmax_aux_buf->untyped_data(), rng_state_buf->untyped_data(), workspace_buf->untyped_data(),
      input_batch, bias_batch, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups, bias_heads,
      head_dim, max_segments_per_seq, wkspace_size, scaling_factor, dropout_probability, bias_type,
      mask_type, qkv_layout, dtype, wkspace_dtype, is_training, deterministic, window_size_left,
      window_size_right);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedAttnForwardHandler, FusedAttnForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // q
                                  .Arg<Buffer_Type>()      // k
                                  .Arg<Buffer_Type>()      // v
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // seed_buf
                                  .Arg<Buffer_Type>()      // q_cu_seqlens
                                  .Arg<Buffer_Type>()      // kv_cu_seqlens
                                  .Arg<Buffer_Type>()      // q_seq_offsets
                                  .Arg<Buffer_Type>()      // k_seq_offsets
                                  .RemainingArgs()         // _cp_aux_args unused
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // softmax_aux
                                  .Ret<Buffer_Type>()      // rng_state
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attrs(),
                              FFI_CudaGraph_Traits);

pybind11::tuple GetFusedAttnBackwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training,
    bool deterministic, size_t max_segments_per_seq, int64_t window_size_left,
    int64_t window_size_right) {
  // For qkv_packed
  auto qkv_shape = std::vector<size_t>{input_batch * q_max_seqlen, 3, attn_heads, head_dim};
  auto qkv_tensor = TensorWrapper(nullptr, qkv_shape, dtype);
  auto dqkv_tensor = TensorWrapper(nullptr, qkv_shape, dtype);

  // For kv_packed
  auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
  auto q_tensor = TensorWrapper(nullptr, q_shape, dtype);
  auto dq_tensor = TensorWrapper(nullptr, q_shape, dtype);
  auto kv_shape = std::vector<size_t>{input_batch * kv_max_seqlen, 2, num_gqa_groups, head_dim};
  auto kv_tensor = TensorWrapper(nullptr, kv_shape, dtype);
  auto dkv_tensor = TensorWrapper(nullptr, kv_shape, dtype);

  // For separate q, k, v
  auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
  auto k_tensor = TensorWrapper(nullptr, k_shape, dtype);
  auto dk_tensor = TensorWrapper(nullptr, k_shape, dtype);
  auto v_shape = k_shape;
  auto v_tensor = TensorWrapper(nullptr, v_shape, dtype);
  auto dv_tensor = TensorWrapper(nullptr, v_shape, dtype);

  auto output_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
  auto doutput_tensor = TensorWrapper(nullptr, output_shape, dtype);
  auto output_tensor = TensorWrapper(nullptr, output_shape, dtype);

  // F16 doesn't use this tensor
  auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);

  auto bias_shape = std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};
  auto dbias_tensor = TensorWrapper(nullptr, bias_shape, dtype);

  NVTETensorPack aux_input_tensors;
  nvte_tensor_pack_create(&aux_input_tensors);

  TensorWrapper query_workspace_tensor;

  auto layout_group = nvte_get_qkv_layout_group(qkv_layout);
  auto is_ragged = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD;
  // It is a WAR to pre-create all possible cuDNN graph at the JIT compile time
  size_t max_num_segments = is_ragged ? input_batch * max_segments_per_seq : input_batch;
  size_t min_num_segments = input_batch;
  auto cudnn_runtime_version = cudnnGetVersion();
  if (is_ragged && cudnn_runtime_version >= 90300) {
    // For cuDNN < 9.3.0, it requires to run all possible seqlens to address act_seqlen = 0
    min_num_segments = input_batch * max_segments_per_seq;
  }
  for (auto num_segments = min_num_segments; num_segments <= max_num_segments; ++num_segments) {
    // the last one is the largest which will be the returned workspace size
    auto q_cu_seqlens_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{num_segments + 1}, DType::kInt32);
    auto kv_cu_seqlens_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{num_segments + 1}, DType::kInt32);
    auto dummy_ragged_offset_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{num_segments + 1}, DType::kInt32);
    if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
      nvte_fused_attn_bwd_qkvpacked(qkv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
                                    s_tensor.data(),  // not used for F16
                                    s_tensor.data(),  // not used for F16
                                    &aux_input_tensors, dqkv_tensor.data(), dbias_tensor.data(),
                                    q_cu_seqlens_tensor.data(), dummy_ragged_offset_tensor.data(),
                                    q_max_seqlen, scaling_factor, dropout_probability, qkv_layout,
                                    bias_type, mask_type, window_size_left, window_size_right,
                                    deterministic, query_workspace_tensor.data(), nullptr);
    } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
      nvte_fused_attn_bwd_kvpacked(
          q_tensor.data(), kv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
          s_tensor.data(),  // not used for F16
          s_tensor.data(),  // not used for F16
          &aux_input_tensors, dq_tensor.data(), dkv_tensor.data(), dbias_tensor.data(),
          q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
          dummy_ragged_offset_tensor.data(), dummy_ragged_offset_tensor.data(), q_max_seqlen,
          kv_max_seqlen, scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
          window_size_left, window_size_right, deterministic, query_workspace_tensor.data(),
          nullptr);
    } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_HD_HD) {
      nvte_fused_attn_bwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), output_tensor.data(),
                          doutput_tensor.data(),
                          s_tensor.data(),  // not used for F16
                          s_tensor.data(),  // not used for F16
                          &aux_input_tensors, dq_tensor.data(), dk_tensor.data(), dv_tensor.data(),
                          dbias_tensor.data(), q_cu_seqlens_tensor.data(),
                          kv_cu_seqlens_tensor.data(), dummy_ragged_offset_tensor.data(),
                          dummy_ragged_offset_tensor.data(), q_max_seqlen, kv_max_seqlen,
                          scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
                          window_size_left, window_size_right, deterministic,
                          query_workspace_tensor.data(), nullptr);
    } else {
      NVTE_ERROR("Unsupported qkv_layout.");
    }
  }

  auto work_shape = MakeShapeVector(query_workspace_tensor.shape());
  return pybind11::make_tuple(work_shape, query_workspace_tensor.dtype());
}

static void FusedAttnBackwardImpl(
    cudaStream_t stream, void *q, void *k, void *v, void *bias, void *softmax_aux, void *rng_state,
    void *output, void *doutput, void *q_cu_seqlens, void *kv_cu_seqlens, void *q_seq_offsets,
    void *k_seq_offsets, void *dq, void *dk, void *dv, void *dbias, void *workspace,
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    size_t max_segments_per_seq, size_t wkspace_size, float scaling_factor,
    float dropout_probability, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    NVTE_QKV_Layout qkv_layout, DType dtype, DType wkspace_dtype, bool is_training,
    bool deterministic, int64_t window_size_left, int64_t window_size_right) {
  FUSED_ATTN_IMPL_COMMON_BLOCK;

  /* Input tensors */
  auto output_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
  auto output_tensor = TensorWrapper(output, output_shape, dtype);
  auto doutput_tensor = TensorWrapper(doutput, output_shape, dtype);

  /* Output tensors */
  auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);  // not used in F16
  auto dbias_tensor = TensorWrapper(dbias, bias_shape, dtype);

  /* Auxiliary tensors (propagated from the forward pass) */
  NVTETensorPack aux_input_tensors;
  nvte_tensor_pack_create(&aux_input_tensors);
  auto backend = nvte_get_fused_attn_backend(
      static_cast<NVTEDType>(dtype), static_cast<NVTEDType>(dtype), qkv_layout, bias_type,
      mask_type, dropout_probability, attn_heads, num_gqa_groups, q_max_seqlen, kv_max_seqlen,
      head_dim, head_dim, window_size_left, window_size_right);
  PrepareFusedAttnBackwardAuxTensors(&aux_input_tensors, input_batch, bias_batch, attn_heads,
                                     bias_heads, q_max_seqlen, kv_max_seqlen, dtype, backend,
                                     softmax_aux, rng_state, bias);

  /* Call the underly NVTE API */
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    auto qkv_shape = std::vector<size_t>{input_batch * q_max_seqlen, 3, attn_heads, head_dim};
    auto qkv_tensor = TensorWrapper(q, qkv_shape, dtype);
    auto dqkv_tensor = TensorWrapper(dq, qkv_shape, dtype);
    if (is_ragged) {
      cudaMemsetAsync(dq, 0, transformer_engine::product(qkv_shape) * typeToSize(dtype), stream);
    }
    nvte_fused_attn_bwd_qkvpacked(qkv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
                                  s_tensor.data(),  // not used for F16
                                  s_tensor.data(),  // not used for F16
                                  &aux_input_tensors, dqkv_tensor.data(), dbias_tensor.data(),
                                  q_cu_seqlens_tensor.data(), q_seq_offsets_tensor.data(),
                                  q_max_seqlen, scaling_factor, dropout_probability, qkv_layout,
                                  bias_type, mask_type, window_size_left, window_size_right,
                                  deterministic, workspace_tensor.data(), stream);
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto kv_shape = std::vector<size_t>{input_batch * kv_max_seqlen, 2, num_gqa_groups, head_dim};
    auto q_tensor = TensorWrapper(q, q_shape, dtype);
    auto kv_tensor = TensorWrapper(k, kv_shape, dtype);
    auto dq_tensor = TensorWrapper(dq, q_shape, dtype);
    auto dkv_tensor = TensorWrapper(dk, kv_shape, dtype);
    if (is_ragged) {
      cudaMemsetAsync(dq, 0, transformer_engine::product(q_shape) * typeToSize(dtype), stream);
      cudaMemsetAsync(dk, 0, transformer_engine::product(kv_shape) * typeToSize(dtype), stream);
    }
    nvte_fused_attn_bwd_kvpacked(
        q_tensor.data(), kv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
        s_tensor.data(),  // not used for F16
        s_tensor.data(),  // not used for F16
        &aux_input_tensors, dq_tensor.data(), dkv_tensor.data(), dbias_tensor.data(),
        q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(), q_seq_offsets_tensor.data(),
        k_seq_offsets_tensor.data(), q_max_seqlen, kv_max_seqlen, scaling_factor,
        dropout_probability, qkv_layout, bias_type, mask_type, window_size_left, window_size_right,
        deterministic, workspace_tensor.data(), stream);
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_HD_HD) {
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
    auto v_shape = k_shape;
    auto q_tensor = TensorWrapper(q, q_shape, dtype);
    auto k_tensor = TensorWrapper(k, k_shape, dtype);
    auto v_tensor = TensorWrapper(v, v_shape, dtype);
    auto dq_tensor = TensorWrapper(dq, q_shape, dtype);
    auto dk_tensor = TensorWrapper(dk, k_shape, dtype);
    auto dv_tensor = TensorWrapper(dv, v_shape, dtype);
    if (is_ragged) {
      cudaMemsetAsync(dq, 0, transformer_engine::product(q_shape) * typeToSize(dtype), stream);
      cudaMemsetAsync(dk, 0, transformer_engine::product(k_shape) * typeToSize(dtype), stream);
      cudaMemsetAsync(dv, 0, transformer_engine::product(v_shape) * typeToSize(dtype), stream);
    }
    nvte_fused_attn_bwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), output_tensor.data(),
                        doutput_tensor.data(),
                        s_tensor.data(),  // not used for F16
                        s_tensor.data(),  // not used for F16
                        &aux_input_tensors, dq_tensor.data(), dk_tensor.data(), dv_tensor.data(),
                        dbias_tensor.data(), q_cu_seqlens_tensor.data(),
                        kv_cu_seqlens_tensor.data(), q_seq_offsets_tensor.data(),
                        k_seq_offsets_tensor.data(), q_max_seqlen, kv_max_seqlen, scaling_factor,
                        dropout_probability, qkv_layout, bias_type, mask_type, window_size_left,
                        window_size_right, deterministic, workspace_tensor.data(), stream);
  } else {
    NVTE_ERROR("Unsupported qkv_layout.");
  }

  nvte_tensor_pack_destroy(&aux_input_tensors);
}

void FusedAttnBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  const CustomCallFusedAttnDescriptor &descriptor =
      *UnpackOpaque<CustomCallFusedAttnDescriptor>(opaque, opaque_len);

  auto qkv_layout = descriptor.qkv_layout;
  auto is_ragged = nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD;

  /* Input buffers from XLA */
  void *q = buffers[0];
  void *k = buffers[1];
  void *v = buffers[2];
  void *bias = buffers[3];
  void *softmax_aux = buffers[4];
  void *rng_state = buffers[5];
  void *output = buffers[6];
  void *doutput = buffers[7];
  void *q_cu_seqlens = buffers[8];
  void *kv_cu_seqlens = buffers[9];
  void *q_seq_offsets = is_ragged ? buffers[10] : nullptr;
  void *k_seq_offsets = is_ragged ? buffers[11] : nullptr;

  /* Output buffer from XLA */
  void *dq = buffers[12];
  void *dk = buffers[13];
  void *dv = buffers[14];
  void *dbias = buffers[15];
  void *workspace = buffers[16];

  FusedAttnBackwardImpl(
      stream, q, k, v, bias, softmax_aux, rng_state, output, doutput, q_cu_seqlens, kv_cu_seqlens,
      q_seq_offsets, k_seq_offsets, dq, dk, dv, dbias, workspace, descriptor.input_batch,
      descriptor.bias_batch, descriptor.q_max_seqlen, descriptor.kv_max_seqlen,
      descriptor.attn_heads, descriptor.num_gqa_groups, descriptor.bias_heads, descriptor.head_dim,
      descriptor.max_segments_per_seq, descriptor.wkspace_size, descriptor.scaling_factor,
      descriptor.dropout_probability, descriptor.bias_type, descriptor.mask_type,
      descriptor.qkv_layout, descriptor.dtype, descriptor.wkspace_dtype, descriptor.is_training,
      descriptor.deterministic, descriptor.window_size_left, descriptor.window_size_right);
}

Error_Type FusedAttnBackwardFFI(cudaStream_t stream, Buffer_Type q_buf, Buffer_Type k_buf,
                                Buffer_Type v_buf, Buffer_Type bias_buf,
                                Buffer_Type softmax_aux_buf, Buffer_Type rng_state_buf,
                                Buffer_Type output_buf, Buffer_Type doutput_buf,
                                Buffer_Type q_cu_seqlens_buf, Buffer_Type kv_cu_seqlens_buf,
                                Buffer_Type q_seq_offsets_buf, Buffer_Type k_seq_offsets_buf,
                                Variadic_Buffer_Type _unused_args, Result_Type dq_buf,
                                Result_Type dk_buf, Result_Type dv_buf, Result_Type dbias_buf,
                                Result_Type workspace_buf, Dictionary attrs) {
  FUSED_ATTN_FFI_GET_ATTRS;

  FusedAttnBackwardImpl(
      stream, q_buf.untyped_data(), k_buf.untyped_data(), v_buf.untyped_data(),
      bias_buf.untyped_data(), softmax_aux_buf.untyped_data(), rng_state_buf.untyped_data(),
      output_buf.untyped_data(), doutput_buf.untyped_data(), q_cu_seqlens_buf.untyped_data(),
      kv_cu_seqlens_buf.untyped_data(), is_ragged ? q_seq_offsets_buf.untyped_data() : nullptr,
      is_ragged ? k_seq_offsets_buf.untyped_data() : nullptr, dq_buf->untyped_data(),
      dk_buf->untyped_data(), dv_buf->untyped_data(), dbias_buf->untyped_data(),
      workspace_buf->untyped_data(), input_batch, bias_batch, q_max_seqlen, kv_max_seqlen,
      attn_heads, num_gqa_groups, bias_heads, head_dim, max_segments_per_seq, wkspace_size,
      scaling_factor, dropout_probability, bias_type, mask_type, qkv_layout, dtype, wkspace_dtype,
      is_training, deterministic, window_size_left, window_size_right);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedAttnBackwardHandler, FusedAttnBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // q
                                  .Arg<Buffer_Type>()      // k
                                  .Arg<Buffer_Type>()      // v
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // softmax_aux
                                  .Arg<Buffer_Type>()      // rng_state
                                  .Arg<Buffer_Type>()      // output
                                  .Arg<Buffer_Type>()      // doutput
                                  .Arg<Buffer_Type>()      // q_cu_seqlens
                                  .Arg<Buffer_Type>()      // kv_cu_seqlens
                                  .Arg<Buffer_Type>()      // q_seq_offsets
                                  .Arg<Buffer_Type>()      // k_seq_offsets
                                  .RemainingArgs()         // _cp_aux_args unused
                                  .Ret<Buffer_Type>()      // dq
                                  .Ret<Buffer_Type>()      // dk
                                  .Ret<Buffer_Type>()      // dv
                                  .Ret<Buffer_Type>()      // dbias
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attrs(),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
