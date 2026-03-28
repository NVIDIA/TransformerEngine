/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../stable_common.h"

#include <transformer_engine/fused_attn.h>

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

// ============================================================================
// Flash Attention prepare helpers
// ============================================================================

Tensor fa_prepare_fwd(Tensor qkvi) {
  STD_TORCH_CHECK(qkvi.dim() == 4, "Expected 4-dim tensor.");
  auto dtype = qkvi.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16,
                  "Expected fp16 or bf16 input.");

  auto s = qkvi.size(0), b = qkvi.size(1), n = qkvi.size(2), h3 = qkvi.size(3);
  auto h = h3 / 3;
  auto qkv = allocateStableTensor({3, b, s, n, h}, dtype, qkvi.get_device_index());

  auto te_qkvi = makeTransformerEngineTensor(qkvi);
  auto te_qkv = makeTransformerEngineTensor(qkv);
  nvte_prepare_flash_attn_fwd(te_qkvi.data(), te_qkv.data(),
                               getCurrentCUDAStreamRaw(qkvi.get_device_index()));
  return qkv;
}

Tensor fa_prepare_bwd(Tensor q, Tensor k, Tensor v) {
  STD_TORCH_CHECK(q.dim() == 4, "Expected 4-dim tensor.");
  auto dtype = q.scalar_type();
  auto b = q.size(1), s = q.size(0), n = q.size(2), h = q.size(3);
  auto qkv = allocateStableTensor({b, s, n, 3 * h}, dtype, q.get_device_index());

  auto te_q = makeTransformerEngineTensor(q);
  auto te_k = makeTransformerEngineTensor(k);
  auto te_v = makeTransformerEngineTensor(v);
  auto te_qkv = makeTransformerEngineTensor(qkv);
  nvte_prepare_flash_attn_bwd(te_q.data(), te_k.data(), te_v.data(), te_qkv.data(),
                               getCurrentCUDAStreamRaw(q.get_device_index()));
  return qkv;
}

// ============================================================================
// THD format helpers for Context Parallel
// ============================================================================

Tensor thd_read_half_tensor(Tensor tensor, Tensor cu_seqlens, int64_t half_idx) {
  int seq_dim = tensor.dim() == 3 ? 0 : 1;
  auto sizes = tensor.sizes();
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    shape.push_back(i == seq_dim ? sizes[i] / 2 : sizes[i]);
  }
  auto half = allocateStableTensor(shape, tensor.scalar_type(),
                                   tensor.get_device_index());

  auto te_tensor = makeTransformerEngineTensor(tensor);
  auto te_cu = makeTransformerEngineTensor(cu_seqlens);
  auto te_half = makeTransformerEngineTensor(half);
  nvte_cp_thd_read_half_tensor(te_tensor.data(), te_cu.data(), te_half.data(),
                                static_cast<int>(half_idx),
                                getCurrentCUDAStreamRaw(tensor.get_device_index()));
  return half;
}

void thd_second_half_lse_correction(Tensor lse, Tensor lse_per_step,
                                     Tensor cu_seqlens, bool lse_packed) {
  auto te_lse = makeTransformerEngineTensor(lse);
  auto te_lse_ps = makeTransformerEngineTensor(lse_per_step);
  auto te_cu = makeTransformerEngineTensor(cu_seqlens);
  nvte_cp_thd_second_half_lse_correction(te_lse.data(), te_lse_ps.data(),
                                          te_cu.data(), lse_packed,
                                          getCurrentCUDAStreamRaw(lse.get_device_index()));
}

Tensor thd_read_second_half_lse(Tensor lse, Tensor cu_seqlens,
                                 bool lse_packed, int64_t second_half_lse_seqlen) {
  std::vector<int64_t> shape;
  if (lse_packed) {
    shape = {lse.size(0), second_half_lse_seqlen};
  } else {
    shape = {lse.size(0), lse.size(1), second_half_lse_seqlen};
  }
  auto half_lse = allocateStableTensorZeros(shape, ScalarType::Float,
                                            lse.get_device_index());

  auto te_lse = makeTransformerEngineTensor(lse);
  auto te_cu = makeTransformerEngineTensor(cu_seqlens);
  auto te_half = makeTransformerEngineTensor(half_lse);
  nvte_cp_thd_read_second_half_lse(te_lse.data(), te_cu.data(), te_half.data(),
                                    lse_packed, static_cast<int>(second_half_lse_seqlen),
                                    getCurrentCUDAStreamRaw(lse.get_device_index()));
  return half_lse;
}

void thd_out_correction(Tensor out, Tensor out_per_step, Tensor lse,
                         Tensor lse_per_step, Tensor cu_seqlens,
                         bool only_second_half, bool lse_packed) {
  auto te_out = makeTransformerEngineTensor(out);
  auto te_ops = makeTransformerEngineTensor(out_per_step);
  auto te_lse = makeTransformerEngineTensor(lse);
  auto te_lps = makeTransformerEngineTensor(lse_per_step);
  auto te_cu = makeTransformerEngineTensor(cu_seqlens);
  nvte_cp_thd_out_correction(te_out.data(), te_ops.data(), te_lse.data(),
                              te_lps.data(), te_cu.data(), only_second_half,
                              lse_packed,
                              getCurrentCUDAStreamRaw(out.get_device_index()));
}

void thd_grad_correction(Tensor grad, Tensor grad_per_step,
                          Tensor cu_seqlens, std::string first_half,
                          std::string second_half) {
  auto te_grad = makeTransformerEngineTensor(grad);
  auto te_gps = makeTransformerEngineTensor(grad_per_step);
  auto te_cu = makeTransformerEngineTensor(cu_seqlens);
  nvte_cp_thd_grad_correction(te_grad.data(), te_gps.data(), te_cu.data(),
                               first_half.data(), second_half.data(),
                               getCurrentCUDAStreamRaw(grad.get_device_index()));
}

Tensor thd_get_partitioned_indices(Tensor cu_seqlens, int64_t total_tokens,
                                    int64_t world_size, int64_t rank) {
  auto output = allocateStableTensor(
      {total_tokens / world_size}, ScalarType::Int, cu_seqlens.get_device_index());
  auto te_cu = makeTransformerEngineTensor(cu_seqlens);
  auto te_out = makeTransformerEngineTensor(output);
  nvte_cp_thd_get_partitioned_indices(
      te_cu.data(), te_out.data(), static_cast<int>(total_tokens),
      static_cast<int>(world_size), static_cast<int>(rank),
      getCurrentCUDAStreamRaw(cu_seqlens.get_device_index()));
  return output;
}

// ============================================================================
// Format conversions
// ============================================================================

Tensor convert_thd_to_bshd(Tensor tensor, Tensor cu_seqlens, int64_t b,
                            int64_t max_seq_len) {
  auto h = tensor.size(1), d = tensor.size(2);
  auto new_tensor = allocateStableTensorZeros(
      {b, max_seq_len, h, d}, tensor.scalar_type(), tensor.get_device_index());

  auto te_t = makeTransformerEngineTensor(tensor);
  auto te_cu = makeTransformerEngineTensor(cu_seqlens);
  auto te_new = makeTransformerEngineTensor(new_tensor);
  nvte_convert_thd_to_bshd(te_t.data(), te_cu.data(), te_new.data(),
                            static_cast<int>(b), static_cast<int>(max_seq_len),
                            getCurrentCUDAStreamRaw(tensor.get_device_index()));
  return new_tensor;
}

Tensor convert_bshd_to_thd(Tensor tensor, Tensor cu_seqlens, int64_t t) {
  auto h = tensor.size(2), d = tensor.size(3);
  auto new_tensor = allocateStableTensorZeros(
      {t, h, d}, tensor.scalar_type(), tensor.get_device_index());

  auto te_t = makeTransformerEngineTensor(tensor);
  auto te_cu = makeTransformerEngineTensor(cu_seqlens);
  auto te_new = makeTransformerEngineTensor(new_tensor);
  nvte_convert_bshd_to_thd(te_t.data(), te_cu.data(), te_new.data(),
                            static_cast<int>(t),
                            getCurrentCUDAStreamRaw(tensor.get_device_index()));
  return new_tensor;
}

// ============================================================================
// KV Cache
// ============================================================================

void copy_to_kv_cache(Tensor new_k, Tensor new_v, Tensor k_cache,
                       Tensor v_cache, Tensor page_table, Tensor cu_new_lens,
                       Tensor cu_cached_lens, int64_t qkv_format, int64_t b,
                       int64_t max_ctx_len, int64_t max_seq_len,
                       int64_t max_pages_per_seq, bool is_non_paged) {
  auto te_nk = makeTransformerEngineTensor(new_k);
  auto te_nv = makeTransformerEngineTensor(new_v);
  auto te_kc = makeTransformerEngineTensor(k_cache);
  auto te_vc = makeTransformerEngineTensor(v_cache);
  auto te_pt = makeTransformerEngineTensor(page_table);
  auto te_cnl = makeTransformerEngineTensor(cu_new_lens);
  auto te_ccl = makeTransformerEngineTensor(cu_cached_lens);

  nvte_copy_to_kv_cache(
      te_nk.data(), te_nv.data(), te_kc.data(), te_vc.data(), te_pt.data(),
      te_cnl.data(), te_ccl.data(),
      static_cast<NVTE_QKV_Format>(qkv_format),
      static_cast<int>(b), static_cast<int>(max_ctx_len),
      static_cast<int>(max_seq_len), static_cast<int>(max_pages_per_seq),
      is_non_paged,
      getCurrentCUDAStreamRaw(new_k.get_device_index()));
}

// ============================================================================
// Fused Attention Forward — noalloc variant
//
// Aux tensors are flattened to individual Tensor? args (max 10 per
// NVTETensorPack::MAX_SIZE). The Python shim packs/unpacks them.
// ============================================================================

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor,
           Tensor, Tensor, Tensor, Tensor, Tensor, int64_t>
fused_attn_fwd_noalloc(
    int64_t max_seqlen_q, int64_t max_seqlen_kv, bool is_training,
    double attn_scale, double p_dropout, bool set_zero,
    int64_t qkv_layout, int64_t bias_type, int64_t attn_mask_type,
    int64_t softmax_type, std::vector<int64_t> window_size,
    bool bottom_right_diagonal,
    Tensor cu_seqlens_q, Tensor cu_seqlens_kv,
    // Q/K/V with optional quantization
    Tensor Q_data, int64_t Q_dtype,
    std::optional<Tensor> Q_scale_inv, int64_t Q_scaling_mode,
    Tensor K_data, int64_t K_dtype,
    std::optional<Tensor> K_scale_inv, int64_t K_scaling_mode,
    Tensor V_data, int64_t V_dtype,
    std::optional<Tensor> V_scale_inv, int64_t V_scaling_mode,
    // S (softmax) — usually empty placeholder
    Tensor S_data, int64_t S_dtype,
    std::optional<Tensor> S_amax, std::optional<Tensor> S_scale,
    std::optional<Tensor> S_scale_inv, int64_t S_scaling_mode,
    // O (output) — pre-allocated
    Tensor O_data, int64_t O_dtype,
    std::optional<Tensor> O_amax, std::optional<Tensor> O_scale,
    std::optional<Tensor> O_scale_inv, int64_t O_scaling_mode,
    // Optional tensors
    std::optional<Tensor> cu_seqlens_q_padded,
    std::optional<Tensor> cu_seqlens_kv_padded,
    std::optional<Tensor> page_table_k, std::optional<Tensor> page_table_v,
    std::optional<Tensor> Bias, std::optional<Tensor> SoftmaxOffset,
    // RNG state [seed, offset] as int64 tensor
    Tensor rng_state,
    bool return_max_logit, bool cuda_graph) {

  auto nvte_layout = static_cast<NVTE_QKV_Layout>(qkv_layout);
  auto nvte_bias = static_cast<NVTE_Bias_Type>(bias_type);
  auto nvte_mask = static_cast<NVTE_Mask_Type>(attn_mask_type);
  auto nvte_softmax = static_cast<NVTE_Softmax_Type>(softmax_type);

  auto Q_shape = getStableTensorShape(Q_data);
  auto K_shape = getStableTensorShape(K_data);
  auto V_shape = getStableTensorShape(V_data);

  // Build Q/K/V TensorWrappers
  auto te_Q = makeQuantizedTensorWrapper(
      Q_data, static_cast<DType>(Q_dtype), Q_shape,
      std::nullopt, std::nullopt, Q_scale_inv,
      static_cast<NVTEScalingMode>(Q_scaling_mode));
  auto te_K = makeQuantizedTensorWrapper(
      K_data, static_cast<DType>(K_dtype), K_shape,
      std::nullopt, std::nullopt, K_scale_inv,
      static_cast<NVTEScalingMode>(K_scaling_mode));
  auto te_V = makeQuantizedTensorWrapper(
      V_data, static_cast<DType>(V_dtype), V_shape,
      std::nullopt, std::nullopt, V_scale_inv,
      static_cast<NVTEScalingMode>(V_scaling_mode));

  // Build O TensorWrapper (output shape = Q shape with V's last dim)
  auto O_shape = getStableTensorShape(O_data);
  auto te_O = makeQuantizedTensorWrapper(
      O_data, static_cast<DType>(O_dtype), O_shape,
      O_amax, O_scale, O_scale_inv,
      static_cast<NVTEScalingMode>(O_scaling_mode));

  // Build S TensorWrapper (placeholder — NVTE determines actual shape)
  auto S_shape_vec = getStableTensorShape(S_data);
  auto te_S = makeQuantizedTensorWrapper(
      S_data, static_cast<DType>(S_dtype), S_shape_vec,
      S_amax, S_scale, S_scale_inv,
      static_cast<NVTEScalingMode>(S_scaling_mode));

  // Zero-fill O if needed for THD format
  auto qkv_type = static_cast<DType>(Q_dtype);
  auto device_idx = Q_data.get_device_index();
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  if (set_zero && nvte_get_qkv_format(nvte_layout) == NVTE_QKV_Format::NVTE_THD) {
    te_O.zero_(stream);
  }

  // Optional tensors
  TensorWrapper te_Bias, te_SoftmaxOffset;
  TensorWrapper te_cu_q, te_cu_kv, te_cu_q_pad, te_cu_kv_pad;
  TensorWrapper te_pt_k, te_pt_v;

  te_cu_q = makeTransformerEngineTensor(cu_seqlens_q);
  te_cu_kv = makeTransformerEngineTensor(cu_seqlens_kv);

  if (Bias.has_value()) {
    te_Bias = makeTransformerEngineTensor(Bias.value());
  }
  if (SoftmaxOffset.has_value()) {
    te_SoftmaxOffset = makeTransformerEngineTensor(SoftmaxOffset.value());
  }
  if (cu_seqlens_q_padded.has_value()) {
    te_cu_q_pad = makeTransformerEngineTensor(cu_seqlens_q_padded.value());
  }
  if (cu_seqlens_kv_padded.has_value()) {
    te_cu_kv_pad = makeTransformerEngineTensor(cu_seqlens_kv_padded.value());
  }
  if (page_table_k.has_value()) {
    te_pt_k = makeTransformerEngineTensor(page_table_k.value());
  }
  if (page_table_v.has_value()) {
    te_pt_v = makeTransformerEngineTensor(page_table_v.value());
  }

  auto te_rng = makeTransformerEngineTensor(rng_state);

  // Aux tensor pack
  NVTETensorPack nvte_aux;
  nvte_tensor_pack_create(&nvte_aux);

  // Workspace
  TensorWrapper workspace;

  // Phase 1: shape query
  nvte_fused_attn_fwd(
      te_Q.data(), te_K.data(), te_V.data(), te_Bias.data(),
      te_SoftmaxOffset.data(), te_S.data(), te_O.data(), &nvte_aux,
      te_cu_q.data(), te_cu_kv.data(), te_cu_q_pad.data(), te_cu_kv_pad.data(),
      te_pt_k.data(), te_pt_v.data(), te_rng.data(),
      static_cast<size_t>(max_seqlen_q), static_cast<size_t>(max_seqlen_kv),
      is_training, return_max_logit, cuda_graph,
      static_cast<float>(attn_scale), static_cast<float>(p_dropout),
      nvte_layout, nvte_bias, nvte_mask, nvte_softmax,
      window_size[0], window_size[1], bottom_right_diagonal,
      workspace.data(), stream);

  // Allocate workspace — declare ws_data OUTSIDE the if-block so the Tensor stays alive
  // through Phase 2 execution. If ws_data were declared inside the if-block, it would be
  // destroyed before Phase 2, and subsequent aux-tensor allocations could reuse the same
  // memory, causing Phase 2 to corrupt the aux tensors (seen as err 700 on 3+ layers).
  auto ws_shape = workspace.shape();
  Tensor ws_data;
  if (ws_shape.ndim > 0 && workspace.numel() > 0) {
    ws_data = allocateStableTensor(
        std::vector<int64_t>(ws_shape.data, ws_shape.data + ws_shape.ndim),
        workspace.dtype(), device_idx);
    workspace = makeTransformerEngineTensor(
        ws_data.data_ptr(),
        std::vector<size_t>(ws_shape.data, ws_shape.data + ws_shape.ndim),
        workspace.dtype());
  }

  // Allocate aux tensors and populate the pack.
  // IMPORTANT: the rng_state slot (shape=[2], dtype=int64) must use the caller-supplied
  // rng_state tensor directly — NOT a new allocation. cuDNN (line 1182 in
  // fused_attn_f16_arbitrary_seqlen.cu) overwrites the aux slot's dptr to point to the
  // input rng_state buffer during Phase 2, so whatever the aux pack slot points to
  // before Phase 2 must be the same tensor that will be returned to Python.
  // Pybind11 does this at extensions/attention.cpp:280: set_tensor_param(i++, rng_state).
  std::vector<Tensor> aux_tensors;
  for (size_t i = 0; i < nvte_aux.size; ++i) {
    auto aux_shape = nvte_tensor_shape(nvte_aux.tensors[i]);
    auto aux_dtype = static_cast<DType>(nvte_tensor_type(nvte_aux.tensors[i]));
    std::vector<int64_t> shape_vec;
    for (size_t d = 0; d < aux_shape.ndim; ++d) {
      shape_vec.push_back(static_cast<int64_t>(aux_shape.data[d]));
    }

    // Detect the rng_state slot: shape [2], dtype int64 (kInt64=3).
    // Use the caller-provided rng_state tensor directly for this slot.
    Tensor aux_tensor;
    bool is_rng_slot = (aux_dtype == DType::kInt64 &&
                        aux_shape.ndim == 1 && aux_shape.data[0] == 2);
    if (is_rng_slot) {
      aux_tensor = rng_state;
    } else {
      aux_tensor = allocateStableTensor(shape_vec, aux_dtype, device_idx);
    }
    aux_tensors.push_back(aux_tensor);

    NVTEBasicTensor temp = {
        aux_tensor.data_ptr(),
        nvte_tensor_type(nvte_aux.tensors[i]),
        aux_shape};
    nvte_set_tensor_param(&nvte_aux.tensors[i], kNVTERowwiseData, &temp);
  }

  // Phase 2: execute
  nvte_fused_attn_fwd(
      te_Q.data(), te_K.data(), te_V.data(), te_Bias.data(),
      te_SoftmaxOffset.data(), te_S.data(), te_O.data(), &nvte_aux,
      te_cu_q.data(), te_cu_kv.data(), te_cu_q_pad.data(), te_cu_kv_pad.data(),
      te_pt_k.data(), te_pt_v.data(), te_rng.data(),
      static_cast<size_t>(max_seqlen_q), static_cast<size_t>(max_seqlen_kv),
      is_training, return_max_logit, cuda_graph,
      static_cast<float>(attn_scale), static_cast<float>(p_dropout),
      nvte_layout, nvte_bias, nvte_mask, nvte_softmax,
      window_size[0], window_size[1], bottom_right_diagonal,
      workspace.data(), stream);

  int64_t num_aux = static_cast<int64_t>(aux_tensors.size());
  nvte_tensor_pack_destroy(&nvte_aux);

  // Pad to 10 slots
  while (aux_tensors.size() < 10) {
    aux_tensors.push_back(Tensor());  // empty/undefined tensor
  }
  return std::make_tuple(
      aux_tensors[0], aux_tensors[1], aux_tensors[2], aux_tensors[3],
      aux_tensors[4], aux_tensors[5], aux_tensors[6], aux_tensors[7],
      aux_tensors[8], aux_tensors[9], num_aux);
}

// ============================================================================
// Fused Attention Backward — noalloc variant
//
// dQ/dK/dV: pre-allocated by Python based on layout group.
// Aux_CTX_Tensors: the aux tensors from forward pass.
// ============================================================================

std::tuple<Tensor, Tensor> fused_attn_bwd_noalloc(
    int64_t max_seqlen_q, int64_t max_seqlen_kv,
    double attn_scale, double p_dropout, bool set_zero,
    int64_t qkv_layout, int64_t bias_type, int64_t attn_mask_type,
    int64_t softmax_type, std::vector<int64_t> window_size,
    bool bottom_right_diagonal, bool deterministic,
    Tensor cu_seqlens_q, Tensor cu_seqlens_kv,
    // Q/K/V/O/dO with optional quantization
    Tensor Q_data, int64_t Q_dtype,
    std::optional<Tensor> Q_scale_inv, int64_t Q_scaling_mode,
    Tensor K_data, int64_t K_dtype,
    std::optional<Tensor> K_scale_inv, int64_t K_scaling_mode,
    Tensor V_data, int64_t V_dtype,
    std::optional<Tensor> V_scale_inv, int64_t V_scaling_mode,
    Tensor O_data, int64_t O_dtype,
    std::optional<Tensor> O_scale_inv, int64_t O_scaling_mode,
    Tensor dO_data, int64_t dO_dtype,
    std::optional<Tensor> dO_scale_inv, int64_t dO_scaling_mode,
    // S and dP (softmax tensors)
    Tensor S_data, int64_t S_dtype,
    std::optional<Tensor> S_scale_inv, int64_t S_scaling_mode,
    Tensor dP_data, int64_t dP_dtype,
    std::optional<Tensor> dP_scale_inv, int64_t dP_scaling_mode,
    // dQ/dK/dV pre-allocated by Python
    Tensor dQ_data, int64_t dQ_dtype,
    std::optional<Tensor> dQ_amax, std::optional<Tensor> dQ_scale,
    std::optional<Tensor> dQ_scale_inv, int64_t dQ_scaling_mode,
    Tensor dK_data, int64_t dK_dtype,
    std::optional<Tensor> dK_amax, std::optional<Tensor> dK_scale,
    std::optional<Tensor> dK_scale_inv, int64_t dK_scaling_mode,
    Tensor dV_data, int64_t dV_dtype,
    std::optional<Tensor> dV_amax, std::optional<Tensor> dV_scale,
    std::optional<Tensor> dV_scale_inv, int64_t dV_scaling_mode,
    // dBias/dSoftmaxOffset pre-allocated
    std::optional<Tensor> dBias, std::optional<Tensor> dSoftmaxOffset,
    // Aux context tensors from forward (flattened, max 10)
    int64_t num_aux_tensors,
    std::optional<Tensor> aux0, std::optional<Tensor> aux1,
    std::optional<Tensor> aux2, std::optional<Tensor> aux3,
    std::optional<Tensor> aux4, std::optional<Tensor> aux5,
    std::optional<Tensor> aux6, std::optional<Tensor> aux7,
    std::optional<Tensor> aux8, std::optional<Tensor> aux9,
    std::optional<Tensor> cu_seqlens_q_padded,
    std::optional<Tensor> cu_seqlens_kv_padded,
    bool cuda_graph) {

  auto nvte_layout = static_cast<NVTE_QKV_Layout>(qkv_layout);
  auto nvte_bias = static_cast<NVTE_Bias_Type>(bias_type);
  auto nvte_mask = static_cast<NVTE_Mask_Type>(attn_mask_type);
  auto nvte_softmax = static_cast<NVTE_Softmax_Type>(softmax_type);

  auto device_idx = Q_data.get_device_index();
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  // Build TensorWrappers
  auto te_Q = makeQuantizedTensorWrapper(Q_data, static_cast<DType>(Q_dtype),
      getStableTensorShape(Q_data), std::nullopt, std::nullopt, Q_scale_inv,
      static_cast<NVTEScalingMode>(Q_scaling_mode));
  auto te_K = makeQuantizedTensorWrapper(K_data, static_cast<DType>(K_dtype),
      getStableTensorShape(K_data), std::nullopt, std::nullopt, K_scale_inv,
      static_cast<NVTEScalingMode>(K_scaling_mode));
  auto te_V = makeQuantizedTensorWrapper(V_data, static_cast<DType>(V_dtype),
      getStableTensorShape(V_data), std::nullopt, std::nullopt, V_scale_inv,
      static_cast<NVTEScalingMode>(V_scaling_mode));
  auto te_O = makeQuantizedTensorWrapper(O_data, static_cast<DType>(O_dtype),
      getStableTensorShape(O_data), std::nullopt, std::nullopt, O_scale_inv,
      static_cast<NVTEScalingMode>(O_scaling_mode));
  auto te_dO = makeQuantizedTensorWrapper(dO_data, static_cast<DType>(dO_dtype),
      getStableTensorShape(dO_data), std::nullopt, std::nullopt, dO_scale_inv,
      static_cast<NVTEScalingMode>(dO_scaling_mode));

  auto te_S = makeQuantizedTensorWrapper(S_data, static_cast<DType>(S_dtype),
      getStableTensorShape(S_data), std::nullopt, std::nullopt, S_scale_inv,
      static_cast<NVTEScalingMode>(S_scaling_mode));
  auto te_dP = makeQuantizedTensorWrapper(dP_data, static_cast<DType>(dP_dtype),
      getStableTensorShape(dP_data), std::nullopt, std::nullopt, dP_scale_inv,
      static_cast<NVTEScalingMode>(dP_scaling_mode));

  auto te_dQ = makeQuantizedTensorWrapper(dQ_data, static_cast<DType>(dQ_dtype),
      getStableTensorShape(dQ_data), dQ_amax, dQ_scale, dQ_scale_inv,
      static_cast<NVTEScalingMode>(dQ_scaling_mode));
  auto te_dK = makeQuantizedTensorWrapper(dK_data, static_cast<DType>(dK_dtype),
      getStableTensorShape(dK_data), dK_amax, dK_scale, dK_scale_inv,
      static_cast<NVTEScalingMode>(dK_scaling_mode));
  auto te_dV = makeQuantizedTensorWrapper(dV_data, static_cast<DType>(dV_dtype),
      getStableTensorShape(dV_data), dV_amax, dV_scale, dV_scale_inv,
      static_cast<NVTEScalingMode>(dV_scaling_mode));

  TensorWrapper te_dBias, te_dSoftmaxOffset;
  if (dBias.has_value()) te_dBias = makeTransformerEngineTensor(dBias.value());
  if (dSoftmaxOffset.has_value()) te_dSoftmaxOffset = makeTransformerEngineTensor(dSoftmaxOffset.value());

  auto te_cu_q = makeTransformerEngineTensor(cu_seqlens_q);
  auto te_cu_kv = makeTransformerEngineTensor(cu_seqlens_kv);
  TensorWrapper te_cu_q_pad, te_cu_kv_pad;
  if (cu_seqlens_q_padded.has_value()) te_cu_q_pad = makeTransformerEngineTensor(cu_seqlens_q_padded.value());
  if (cu_seqlens_kv_padded.has_value()) te_cu_kv_pad = makeTransformerEngineTensor(cu_seqlens_kv_padded.value());

  // Build aux tensor pack from flattened forward context tensors
  std::optional<Tensor> aux_slots[] = {
      aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8, aux9};
  NVTETensorPack nvte_aux;
  nvte_tensor_pack_create(&nvte_aux);
  nvte_aux.size = static_cast<size_t>(num_aux_tensors);
  for (size_t i = 0; i < nvte_aux.size; ++i) {
    NVTE_CHECK(aux_slots[i].has_value(), "aux tensor ", i, " is None but num_aux_tensors=", num_aux_tensors);
    auto &t = aux_slots[i].value();
    auto shape_vec = getStableTensorShape(t);
    auto dtype = GetTransformerEngineDType(t.scalar_type());
    NVTEBasicTensor temp = {
        t.data_ptr(),
        static_cast<NVTEDType>(dtype),
        nvte_make_shape(shape_vec.data(), shape_vec.size())};
    nvte_set_tensor_param(&nvte_aux.tensors[i], kNVTERowwiseData, &temp);
  }

  // Zero-fill dQ/dK/dV for THD format if needed
  if (set_zero && nvte_get_qkv_format(nvte_layout) == NVTE_QKV_Format::NVTE_THD) {
    te_dQ.zero_(stream);
    te_dK.zero_(stream);
    te_dV.zero_(stream);
  }
  if (dBias.has_value() && nvte_get_qkv_format(nvte_layout) == NVTE_QKV_Format::NVTE_THD) {
    torch::stable::zero_(dBias.value());
  }

  TensorWrapper workspace;

  // Phase 1: shape query
  nvte_fused_attn_bwd(
      te_Q.data(), te_K.data(), te_V.data(), te_O.data(), te_dO.data(),
      te_S.data(), te_dP.data(), &nvte_aux,
      te_dQ.data(), te_dK.data(), te_dV.data(), te_dBias.data(),
      te_dSoftmaxOffset.data(),
      te_cu_q.data(), te_cu_kv.data(), te_cu_q_pad.data(), te_cu_kv_pad.data(),
      static_cast<size_t>(max_seqlen_q), static_cast<size_t>(max_seqlen_kv),
      static_cast<float>(attn_scale), static_cast<float>(p_dropout),
      nvte_layout, nvte_bias, nvte_mask, nvte_softmax,
      window_size[0], window_size[1], bottom_right_diagonal,
      deterministic, cuda_graph, workspace.data(), stream);

  // Allocate workspace — declare ws_data OUTSIDE the if-block so it stays alive
  // through Phase 2 (same issue as fwd: ws_data inside if-block would be freed
  // before Phase 2, and subsequent allocations could reuse the workspace memory).
  auto ws_shape = workspace.shape();
  Tensor ws_data;
  if (ws_shape.ndim > 0 && workspace.numel() > 0) {
    ws_data = allocateStableTensor(
        std::vector<int64_t>(ws_shape.data, ws_shape.data + ws_shape.ndim),
        workspace.dtype(), device_idx);
    workspace = makeTransformerEngineTensor(
        ws_data.data_ptr(),
        std::vector<size_t>(ws_shape.data, ws_shape.data + ws_shape.ndim),
        workspace.dtype());
  }

  // Phase 2: execute
  nvte_fused_attn_bwd(
      te_Q.data(), te_K.data(), te_V.data(), te_O.data(), te_dO.data(),
      te_S.data(), te_dP.data(), &nvte_aux,
      te_dQ.data(), te_dK.data(), te_dV.data(), te_dBias.data(),
      te_dSoftmaxOffset.data(),
      te_cu_q.data(), te_cu_kv.data(), te_cu_q_pad.data(), te_cu_kv_pad.data(),
      static_cast<size_t>(max_seqlen_q), static_cast<size_t>(max_seqlen_kv),
      static_cast<float>(attn_scale), static_cast<float>(p_dropout),
      nvte_layout, nvte_bias, nvte_mask, nvte_softmax,
      window_size[0], window_size[1], bottom_right_diagonal,
      deterministic, cuda_graph, workspace.data(), stream);

  nvte_tensor_pack_destroy(&nvte_aux);

  // Return dBias and dSoftmaxOffset (dQ/dK/dV are written in-place)
  Tensor ret_dBias = dBias.has_value() ? dBias.value() : Tensor();
  Tensor ret_dSO = dSoftmaxOffset.has_value() ? dSoftmaxOffset.value() : Tensor();
  return std::make_tuple(ret_dBias, ret_dSO);
}

// ============================================================================
// Fused Attention Backward — packed variant (57 args, under 64-arg limit)
//
// dtype_info is a 1-D int64 CPU tensor with 20 values:
//   [Q_dtype, Q_sm, K_dtype, K_sm, V_dtype, V_sm, O_dtype, O_sm,
//    dO_dtype, dO_sm, S_dtype, S_sm, dP_dtype, dP_sm,
//    dQ_dtype, dQ_sm, dK_dtype, dK_sm, dV_dtype, dV_sm]
// dQ/dK/dV/dBias/dSoftmaxOffset are pre-allocated by the Python caller.
// ============================================================================
std::tuple<Tensor, Tensor> fused_attn_bwd_packed(
    // Config (13)
    int64_t max_seqlen_q, int64_t max_seqlen_kv,
    double attn_scale, double p_dropout, bool set_zero,
    int64_t qkv_layout, int64_t bias_type, int64_t attn_mask_type,
    int64_t softmax_type, std::vector<int64_t> window_size,
    bool bottom_right_diagonal, bool deterministic, bool cuda_graph,
    // Sequence lengths (4)
    Tensor cu_seqlens_q, Tensor cu_seqlens_kv,
    std::optional<Tensor> cu_seqlens_q_padded,
    std::optional<Tensor> cu_seqlens_kv_padded,
    // Input tensors: data + scale_inv (10)
    Tensor Q_data, std::optional<Tensor> Q_scale_inv,
    Tensor K_data, std::optional<Tensor> K_scale_inv,
    Tensor V_data, std::optional<Tensor> V_scale_inv,
    Tensor O_data, std::optional<Tensor> O_scale_inv,
    Tensor dO_data, std::optional<Tensor> dO_scale_inv,
    // Softmax buffers (4)
    Tensor S_data, std::optional<Tensor> S_scale_inv,
    Tensor dP_data, std::optional<Tensor> dP_scale_inv,
    // Output grad tensors (12)
    Tensor dQ_data, std::optional<Tensor> dQ_amax,
    std::optional<Tensor> dQ_scale, std::optional<Tensor> dQ_scale_inv,
    Tensor dK_data, std::optional<Tensor> dK_amax,
    std::optional<Tensor> dK_scale, std::optional<Tensor> dK_scale_inv,
    Tensor dV_data, std::optional<Tensor> dV_amax,
    std::optional<Tensor> dV_scale, std::optional<Tensor> dV_scale_inv,
    // Optional bias outputs (2)
    std::optional<Tensor> dBias, std::optional<Tensor> dSoftmaxOffset,
    // Packed dtype info (1): [Q_dtype, Q_sm, K_dtype, K_sm, V_dtype, V_sm,
    //   O_dtype, O_sm, dO_dtype, dO_sm, S_dtype, S_sm, dP_dtype, dP_sm,
    //   dQ_dtype, dQ_sm, dK_dtype, dK_sm, dV_dtype, dV_sm]
    Tensor dtype_info,
    // Aux context from forward (11)
    int64_t num_aux_tensors,
    std::optional<Tensor> aux0, std::optional<Tensor> aux1,
    std::optional<Tensor> aux2, std::optional<Tensor> aux3,
    std::optional<Tensor> aux4, std::optional<Tensor> aux5,
    std::optional<Tensor> aux6, std::optional<Tensor> aux7,
    std::optional<Tensor> aux8, std::optional<Tensor> aux9) {

  // Unpack dtype info from CPU int64 tensor (passed from Python as CPU tensor)
  const auto* dt_ptr = static_cast<const int64_t*>(dtype_info.data_ptr());
  int64_t Q_dtype = dt_ptr[0],  Q_sm = dt_ptr[1];
  int64_t K_dtype = dt_ptr[2],  K_sm = dt_ptr[3];
  int64_t V_dtype = dt_ptr[4],  V_sm = dt_ptr[5];
  int64_t O_dtype = dt_ptr[6],  O_sm = dt_ptr[7];
  int64_t dO_dtype = dt_ptr[8], dO_sm = dt_ptr[9];
  int64_t S_dtype = dt_ptr[10], S_sm = dt_ptr[11];
  int64_t dP_dtype = dt_ptr[12], dP_sm = dt_ptr[13];
  int64_t dQ_dtype = dt_ptr[14], dQ_sm = dt_ptr[15];
  int64_t dK_dtype = dt_ptr[16], dK_sm = dt_ptr[17];
  int64_t dV_dtype = dt_ptr[18], dV_sm = dt_ptr[19];

  return fused_attn_bwd_noalloc(
      max_seqlen_q, max_seqlen_kv, attn_scale, p_dropout, set_zero,
      qkv_layout, bias_type, attn_mask_type, softmax_type, window_size,
      bottom_right_diagonal, deterministic,
      cu_seqlens_q, cu_seqlens_kv,
      Q_data, Q_dtype, Q_scale_inv, Q_sm,
      K_data, K_dtype, K_scale_inv, K_sm,
      V_data, V_dtype, V_scale_inv, V_sm,
      O_data, O_dtype, O_scale_inv, O_sm,
      dO_data, dO_dtype, dO_scale_inv, dO_sm,
      S_data, S_dtype, S_scale_inv, S_sm,
      dP_data, dP_dtype, dP_scale_inv, dP_sm,
      dQ_data, dQ_dtype, dQ_amax, dQ_scale, dQ_scale_inv, dQ_sm,
      dK_data, dK_dtype, dK_amax, dK_scale, dK_scale_inv, dK_sm,
      dV_data, dV_dtype, dV_amax, dV_scale, dV_scale_inv, dV_sm,
      dBias, dSoftmaxOffset,
      num_aux_tensors,
      aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8, aux9,
      cu_seqlens_q_padded, cu_seqlens_kv_padded,
      cuda_graph);
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  // Fused attention forward/backward (noalloc, flattened aux tensors)
  m.def("fused_attn_fwd_noalloc(int max_seqlen_q, int max_seqlen_kv, bool is_training, float attn_scale, float p_dropout, bool set_zero, int qkv_layout, int bias_type, int attn_mask_type, int softmax_type, int[] window_size, bool bottom_right_diagonal, Tensor cu_seqlens_q, Tensor cu_seqlens_kv, Tensor Q_data, int Q_dtype, Tensor? Q_scale_inv, int Q_scaling_mode, Tensor K_data, int K_dtype, Tensor? K_scale_inv, int K_scaling_mode, Tensor V_data, int V_dtype, Tensor? V_scale_inv, int V_scaling_mode, Tensor S_data, int S_dtype, Tensor? S_amax, Tensor? S_scale, Tensor? S_scale_inv, int S_scaling_mode, Tensor O_data, int O_dtype, Tensor? O_amax, Tensor? O_scale, Tensor? O_scale_inv, int O_scaling_mode, Tensor? cu_seqlens_q_padded, Tensor? cu_seqlens_kv_padded, Tensor? page_table_k, Tensor? page_table_v, Tensor? Bias, Tensor? SoftmaxOffset, Tensor rng_state, bool return_max_logit, bool cuda_graph) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int)");
  m.def("fused_attn_bwd_packed("
        "int max_seqlen_q, int max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero, "
        "int qkv_layout, int bias_type, int attn_mask_type, int softmax_type, int[] window_size, "
        "bool bottom_right_diagonal, bool deterministic, bool cuda_graph, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_kv, Tensor? cu_seqlens_q_padded, Tensor? cu_seqlens_kv_padded, "
        "Tensor Q_data, Tensor? Q_scale_inv, Tensor K_data, Tensor? K_scale_inv, "
        "Tensor V_data, Tensor? V_scale_inv, Tensor O_data, Tensor? O_scale_inv, "
        "Tensor dO_data, Tensor? dO_scale_inv, "
        "Tensor S_data, Tensor? S_scale_inv, Tensor dP_data, Tensor? dP_scale_inv, "
        "Tensor dQ_data, Tensor? dQ_amax, Tensor? dQ_scale, Tensor? dQ_scale_inv, "
        "Tensor dK_data, Tensor? dK_amax, Tensor? dK_scale, Tensor? dK_scale_inv, "
        "Tensor dV_data, Tensor? dV_amax, Tensor? dV_scale, Tensor? dV_scale_inv, "
        "Tensor? dBias, Tensor? dSoftmaxOffset, Tensor dtype_info, "
        "int num_aux_tensors, "
        "Tensor? aux0, Tensor? aux1, Tensor? aux2, Tensor? aux3, Tensor? aux4, "
        "Tensor? aux5, Tensor? aux6, Tensor? aux7, Tensor? aux8, Tensor? aux9"
        ") -> (Tensor, Tensor)");
  // fused_attn_bwd_noalloc has 77 args which exceeds the 64-arg PyTorch
  // dispatcher limit. Use fused_attn_bwd_packed instead.
  // Helpers
  m.def("fa_prepare_fwd(Tensor qkvi) -> Tensor");
  m.def("fa_prepare_bwd(Tensor q, Tensor k, Tensor v) -> Tensor");
  m.def("thd_read_half_tensor(Tensor tensor, Tensor cu_seqlens, int half_idx) -> Tensor");
  m.def("thd_second_half_lse_correction(Tensor lse, Tensor lse_per_step, Tensor cu_seqlens, bool lse_packed) -> ()");
  m.def("thd_read_second_half_lse(Tensor lse, Tensor cu_seqlens, bool lse_packed, int second_half_lse_seqlen) -> Tensor");
  m.def("thd_out_correction(Tensor out, Tensor out_per_step, Tensor lse, Tensor lse_per_step, Tensor cu_seqlens, bool only_second_half, bool lse_packed) -> ()");
  m.def("thd_grad_correction(Tensor grad, Tensor grad_per_step, Tensor cu_seqlens, str first_half, str second_half) -> ()");
  m.def("thd_get_partitioned_indices(Tensor cu_seqlens, int total_tokens, int world_size, int rank) -> Tensor");
  m.def("convert_thd_to_bshd(Tensor tensor, Tensor cu_seqlens, int b, int max_seq_len) -> Tensor");
  m.def("convert_bshd_to_thd(Tensor tensor, Tensor cu_seqlens, int t) -> Tensor");
  m.def("copy_to_kv_cache(Tensor new_k, Tensor new_v, Tensor k_cache, Tensor v_cache, Tensor page_table, Tensor cu_new_lens, Tensor cu_cached_lens, int qkv_format, int b, int max_ctx_len, int max_seq_len, int max_pages_per_seq, bool is_non_paged) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("fused_attn_fwd_noalloc", TORCH_BOX(fused_attn_fwd_noalloc));
  m.impl("fused_attn_bwd_packed", TORCH_BOX(fused_attn_bwd_packed));
  // fused_attn_bwd_noalloc not registered (77 args > 64 limit); use fused_attn_bwd_packed
  m.impl("fa_prepare_fwd", TORCH_BOX(fa_prepare_fwd));
  m.impl("fa_prepare_bwd", TORCH_BOX(fa_prepare_bwd));
  m.impl("thd_read_half_tensor", TORCH_BOX(thd_read_half_tensor));
  m.impl("thd_second_half_lse_correction", TORCH_BOX(thd_second_half_lse_correction));
  m.impl("thd_read_second_half_lse", TORCH_BOX(thd_read_second_half_lse));
  m.impl("thd_out_correction", TORCH_BOX(thd_out_correction));
  m.impl("thd_grad_correction", TORCH_BOX(thd_grad_correction));
  m.impl("thd_get_partitioned_indices", TORCH_BOX(thd_get_partitioned_indices));
  m.impl("convert_thd_to_bshd", TORCH_BOX(convert_thd_to_bshd));
  m.impl("convert_bshd_to_thd", TORCH_BOX(convert_bshd_to_thd));
  m.impl("copy_to_kv_cache", TORCH_BOX(copy_to_kv_cache));
}
