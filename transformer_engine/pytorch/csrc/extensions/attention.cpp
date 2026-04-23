/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common.h"
#include "pybind.h"

namespace {

constexpr int block_size = 512;

// fast zero-fills of tensors
void mha_fill(const transformer_engine::TensorWrapper &self, const at::Tensor &start_index) {
  std::vector<size_t> shape = transformer_engine::pytorch::convertShape(self.shape());

  auto max_tokens = shape[0];
  auto fcd_size = 1;
  for (size_t i = 1; i <= shape.size(); i++) {
    fcd_size *= shape[i];
  }

  NVTE_CHECK(fcd_size % block_size == 0, "input size not aligned to block size");

  size_t element_size_bits = transformer_engine::pytorch::typeToNumBits(self.dtype());
  int32_t start_row = start_index.data_ptr<int32_t>()[0];
  void *base_ptr = static_cast<char *>(self.get_rowwise_data().data_ptr) +
                   static_cast<size_t>(start_row) * fcd_size * element_size_bits / 8;
  size_t num_rows_to_zero = max_tokens - start_row;
  size_t total_bytes = num_rows_to_zero * fcd_size * element_size_bits / 8;

  NVTE_SCOPED_GIL_RELEASE(
      { nvte_memset(base_ptr, 0, total_bytes, at::cuda::getCurrentCUDAStream()); });
}

}  // namespace

namespace transformer_engine::pytorch {

// get the fused attention backend
NVTE_Fused_Attn_Backend get_fused_attn_backend(
    bool is_training, const DType q_dtype, const DType kv_dtype, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
    float p_dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left,
    int64_t window_size_right, bool return_max_logit, bool cuda_graph, bool deterministic) {
  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      is_training, static_cast<NVTEDType>(q_dtype), static_cast<NVTEDType>(kv_dtype), qkv_layout,
      bias_type, attn_mask_type, softmax_type, p_dropout, num_attn_heads, num_gqa_groups,
      max_seqlen_q, max_seqlen_kv, head_dim_qk, head_dim_v, window_size_left, window_size_right,
      return_max_logit, cuda_graph, deterministic);
  return fused_attention_backend;
}

// helper function for S and dP quantizers
std::tuple<TensorWrapper, py::object, std::optional<at::Tensor>> quantizer_helper(
    py::handle quantizer, const std::vector<size_t> &shape, DType dtype, bool create_hp_tensor,
    std::optional<at::Tensor> data) {
  std::unique_ptr<Quantizer> T_quantizer = convert_quantizer(quantizer);
  TensorWrapper te_T;
  py::object py_T;
  std::optional<at::Tensor> amax_buf;
  if (quantizer.is_none()) {
    // high precision
    auto *none_quantizer = dynamic_cast<NoneQuantizer *>(T_quantizer.get());
    if (data.has_value()) {
      std::tie(te_T, py_T) = none_quantizer->create_tensor(shape, dtype, data.value());
    } else {
      std::tie(te_T, py_T) = none_quantizer->create_tensor(shape, dtype);
    }
  } else if (detail::IsFloat8Quantizers(quantizer.ptr())) {
    // delayed scaling; this helps initialize scale_inv
    auto *T_quantizer_fp8 = dynamic_cast<Float8Quantizer *>(T_quantizer.get());
    std::tie(te_T, py_T) =
        T_quantizer_fp8->create_tensor(shape, dtype, data, std::nullopt, std::nullopt);
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    // current scaling
    auto *T_quantizer_fp8 = dynamic_cast<Float8CurrentScalingQuantizer *>(T_quantizer.get());
    if (create_hp_tensor) {
      if (data.has_value()) {
        std::tie(te_T, py_T, amax_buf) =
            T_quantizer_fp8->create_unquantized_tensor_with_amax(shape, dtype, data.value());
      } else {
        std::tie(te_T, py_T, amax_buf) =
            T_quantizer_fp8->create_unquantized_tensor_with_amax(shape, dtype);
      }
    } else {
      std::tie(te_T, py_T) = T_quantizer_fp8->create_tensor(shape, dtype);
      NVTE_CHECK(
          !data.has_value(),
          "Float8CurrentScalingQuantizer::create_tensor() does not take data tensor as input!");
    }
  } else if (detail::IsMXFP8Quantizers(quantizer.ptr())) {
    // MXFP8
    if (create_hp_tensor) {
      if (data.has_value()) {
        std::tie(te_T, py_T) = NoneQuantizer(py::none()).create_tensor(shape, dtype, data.value());
      } else {
        std::tie(te_T, py_T) = NoneQuantizer(py::none()).create_tensor(shape, dtype);
      }
    } else {
      auto *T_quantizer_fp8 = dynamic_cast<MXFP8Quantizer *>(T_quantizer.get());
      std::tie(te_T, py_T) = T_quantizer_fp8->create_tensor(shape, dtype);
      NVTE_CHECK(!data.has_value(),
                 "MXFP8Quantizer::create_tensor() does not take data tensor as input!");
    }
  }
  return {std::move(te_T), std::move(py_T), std::move(amax_buf)};
}

// fused attention FWD with separate Q, K and V tensors
std::vector<py::object> fused_attn_fwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training, float attn_scale, float p_dropout,
    bool set_zero, NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
    NVTE_QKV_Format qkv_scale_inv_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    NVTE_Softmax_Type softmax_type, const std::vector<int64_t> window_size,
    bool bottom_right_diagonal, const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv,
    const py::handle Q, const py::handle K, const py::handle V, const at::ScalarType fake_dtype,
    const std::optional<at::Tensor> cu_seqlens_q_padded,
    const std::optional<at::Tensor> cu_seqlens_kv_padded,
    const std::optional<at::Tensor> page_table_k, const std::optional<at::Tensor> page_table_v,
    py::handle s_quantizer, py::handle o_quantizer, const std::optional<at::Tensor> Bias,
    const std::optional<at::Tensor> SoftmaxOffset, const std::optional<at::Generator> rng_gen,
    size_t rng_elts_per_thread, bool return_max_logit, bool cuda_graph) {
  // Ensure that cuDNN handle is created on the correct device,
  // overriding torch.cuda.set_device calls from user side.
  // Assumes all tensors passed are on the same device.
  at::cuda::CUDAGuard device_guard(cu_seqlens_q.device());

  auto none = py::none();

  // create QKV tensor wrappers
  TensorWrapper te_Q, te_K, te_V;
  te_Q = makeTransformerEngineTensor(Q, none);
  te_K = makeTransformerEngineTensor(K, none);
  te_V = makeTransformerEngineTensor(V, none);
  const DType qkv_type = te_Q.dtype();

  // create S tensor
  auto [te_S, py_S, _] = quantizer_helper(s_quantizer, {0}, DType::kFloat32, false, std::nullopt);

  // create O tensor
  std::unique_ptr<Quantizer> O_quantizer = convert_quantizer(o_quantizer);
  std::vector<size_t> q_shape = convertShape(te_Q.shape());
  std::vector<size_t> v_shape = convertShape(te_V.shape());
  auto o_shape_tmp = std::vector<size_t>{q_shape.begin(), q_shape.end()};
  o_shape_tmp[o_shape_tmp.size() - 1] = v_shape[v_shape.size() - 1];
  auto o_shape = std::vector<size_t>{o_shape_tmp.begin(), o_shape_tmp.end()};
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  AttentionShape o_parsed(q_format, o_shape_tmp.data());
  size_t h = o_parsed.h(), d = o_parsed.d();
  o_parsed.to_format(o_format, o_shape.data());
  const DType fake_dtype_te = GetTransformerEngineDType(fake_dtype);
  auto [te_O, py_O, o_amax_buf] =
      quantizer_helper(o_quantizer, o_shape, fake_dtype_te, true, std::nullopt);

  // construct NVTE tensors
  TensorWrapper te_Bias;
  TensorWrapper te_cu_seqlens_q, te_cu_seqlens_kv;
  TensorWrapper te_cu_seqlens_q_padded, te_cu_seqlens_kv_padded;
  TensorWrapper te_page_table_k, te_page_table_v;
  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    // FP8
    if (set_zero && (o_format == NVTE_QKV_Format::NVTE_THD)) {
      if ((h * d) % block_size == 0) {
        mha_fill(te_O, cu_seqlens_q.index({torch::indexing::Slice(-1, torch::indexing::None)}));
      } else {
        te_O.zero_(at::cuda::getCurrentCUDAStream());
      }
    }
  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    if (o_format == NVTE_QKV_Format::NVTE_THD) {
      te_O.zero_(at::cuda::getCurrentCUDAStream());
    }
  } else {
    NVTE_ERROR("Fused attention only supports FP8 and BF16/FP16 data types. \n");
  }
  if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) && (Bias.has_value())) {
    auto bias_sizes = Bias.value().sizes().vec();
    std::vector<size_t> bias_shape{bias_sizes.begin(), bias_sizes.end()};
    te_Bias = makeTransformerEngineTensor(Bias.value().data_ptr(), bias_shape, DType::kFloat32);
  }
  auto cu_seqlens_q_sizes = cu_seqlens_q.sizes().vec();
  std::vector<size_t> cu_seqlens_q_shape{cu_seqlens_q_sizes.begin(), cu_seqlens_q_sizes.end()};
  auto cu_seqlens_kv_sizes = cu_seqlens_kv.sizes().vec();
  std::vector<size_t> cu_seqlens_kv_shape{cu_seqlens_kv_sizes.begin(), cu_seqlens_kv_sizes.end()};
  te_cu_seqlens_q =
      makeTransformerEngineTensor(cu_seqlens_q.data_ptr(), cu_seqlens_q_shape, DType::kInt32);
  te_cu_seqlens_kv =
      makeTransformerEngineTensor(cu_seqlens_kv.data_ptr(), cu_seqlens_kv_shape, DType::kInt32);

  if ((cu_seqlens_q_padded.has_value()) && (cu_seqlens_kv_padded.has_value())) {
    auto cu_seqlens_q_padded_sizes = cu_seqlens_q_padded.value().sizes().vec();
    std::vector<size_t> cu_seqlens_q_padded_shape{cu_seqlens_q_padded_sizes.begin(),
                                                  cu_seqlens_q_padded_sizes.end()};
    auto cu_seqlens_kv_padded_sizes = cu_seqlens_kv_padded.value().sizes().vec();
    std::vector<size_t> cu_seqlens_kv_padded_shape{cu_seqlens_kv_padded_sizes.begin(),
                                                   cu_seqlens_kv_padded_sizes.end()};
    te_cu_seqlens_q_padded = makeTransformerEngineTensor(cu_seqlens_q_padded.value().data_ptr(),
                                                         cu_seqlens_q_padded_shape, DType::kInt32);
    te_cu_seqlens_kv_padded = makeTransformerEngineTensor(
        cu_seqlens_kv_padded.value().data_ptr(), cu_seqlens_kv_padded_shape, DType::kInt32);
  }

  if ((page_table_k.has_value()) && (page_table_v.has_value())) {
    auto page_table_k_sizes = page_table_k.value().sizes().vec();
    std::vector<size_t> page_table_k_shape{page_table_k_sizes.begin(), page_table_k_sizes.end()};
    auto page_table_v_sizes = page_table_v.value().sizes().vec();
    std::vector<size_t> page_table_v_shape{page_table_v_sizes.begin(), page_table_v_sizes.end()};
    te_page_table_k =
        makeTransformerEngineTensor(page_table_k.value().data_ptr(), page_table_k_shape,
                                    DType::kInt32, nullptr, nullptr, nullptr);
    te_page_table_v =
        makeTransformerEngineTensor(page_table_v.value().data_ptr(), page_table_v_shape,
                                    DType::kInt32, nullptr, nullptr, nullptr);
  }

  // softmax offset
  TensorWrapper te_SoftmaxOffset;
  if ((softmax_type != NVTE_VANILLA_SOFTMAX) && (SoftmaxOffset.has_value())) {
    auto SoftmaxOffset_sizes = SoftmaxOffset.value().sizes().vec();
    std::vector<size_t> SoftmaxOffset_shape{SoftmaxOffset_sizes.begin(), SoftmaxOffset_sizes.end()};
    te_SoftmaxOffset =
        makeTransformerEngineTensor(SoftmaxOffset.value().data_ptr(), SoftmaxOffset_shape,
                                    DType::kFloat32, nullptr, nullptr, nullptr);
  }

  // extract rng seed and offset
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      rng_gen, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox_args = init_philox_state(gen, rng_elts_per_thread);
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options);
  philox_unpack(philox_args, static_cast<int64_t *>(rng_state.data_ptr()));
  auto te_rng_state = makeTransformerEngineTensor(rng_state);

  // create auxiliary output tensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_attn_fwd(
        te_Q.data(), te_K.data(), te_V.data(), te_Bias.data(), te_SoftmaxOffset.data(), te_S.data(),
        te_O.data(), &nvte_aux_tensor_pack, te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
        te_cu_seqlens_q_padded.data(), te_cu_seqlens_kv_padded.data(), te_page_table_k.data(),
        te_page_table_v.data(), te_rng_state.data(), max_seqlen_q, max_seqlen_kv, is_training,
        return_max_logit, cuda_graph, attn_scale, p_dropout, qkv_layout, o_format,
        qkv_scale_inv_format, bias_type, attn_mask_type, softmax_type, window_size[0],
        window_size[1], bottom_right_diagonal, workspace.data(), at::cuda::getCurrentCUDAStream());
  });

  // allocate memory for workspace and auxiliary output tensors
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // output_tensors = [O, nvte_aux_tensor_pack.tensors]
  std::vector<py::object> output_tensors;
  output_tensors.push_back(py_O);
  auto set_tensor_param = [&](size_t i, const at::Tensor &output_tensor) {
    output_tensors.push_back(py::cast(output_tensor));
    NVTEBasicTensor temp_data = {output_tensor.data_ptr(),
                                 nvte_tensor_type(nvte_aux_tensor_pack.tensors[i]),
                                 nvte_tensor_shape(nvte_aux_tensor_pack.tensors[i])};
    nvte_set_tensor_param(&nvte_aux_tensor_pack.tensors[i], kNVTERowwiseData, &temp_data);
  };
  // allocate memory for nvte_aux_tensor_pack.tensors
  // f16_max512   : S [b, h, sq, skv]
  // f16_arbitrary:
  // return_max_logit=false: S [b, h, sq, 1], rng_state [2], (optional) Bias [1, h, sq, skv], (optional) SoftmaxOffset [1, h, 1, 1]
  // return_max_logit=true: S [b, h, sq, 1], Max [b, h, sq, 1], rng_state [2], (optional) Bias [1, h, sq, skv], (optional) SoftmaxOffset [1, h, 1, 1]
  // fp8          : M [b, h, sq, 1], optional ZInv [b, h, sq, 1] (T3HD path), rng_state [2]
  size_t i = 0;
  at::Tensor output_tensor;
  // intermediate softmax tensor, S or M (for fp8)
  output_tensor =
      allocateSpace(nvte_shape_to_vector(nvte_tensor_shape(nvte_aux_tensor_pack.tensors[i])),
                    static_cast<DType>(nvte_tensor_type(nvte_aux_tensor_pack.tensors[i])), false);
  set_tensor_param(i++, output_tensor);
  // fp8 T3HD has an additional softmax stats tensor, ZInv; return_max_logit=true has an additional Max tensor
  if (((qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) &&
       qkv_layout == NVTE_QKV_Layout::NVTE_T3HD) ||
      return_max_logit) {
    output_tensor =
        allocateSpace(nvte_shape_to_vector(nvte_tensor_shape(nvte_aux_tensor_pack.tensors[i])),
                      static_cast<DType>(nvte_tensor_type(nvte_aux_tensor_pack.tensors[i])), false);
    set_tensor_param(i++, output_tensor);
  }
  // rng_state
  if (i < nvte_aux_tensor_pack.size) {
    set_tensor_param(i++, rng_state);
  }
  // bias (optional)
  if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) && (Bias.has_value())) {
    set_tensor_param(i++, Bias.value());
  }
  // softmax_offset (optional)
  if ((softmax_type != NVTE_VANILLA_SOFTMAX) && (SoftmaxOffset.has_value())) {
    set_tensor_param(i++, SoftmaxOffset.value());
  }

  // execute the kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_attn_fwd(
        te_Q.data(), te_K.data(), te_V.data(), te_Bias.data(), te_SoftmaxOffset.data(), te_S.data(),
        te_O.data(), &nvte_aux_tensor_pack, te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
        te_cu_seqlens_q_padded.data(), te_cu_seqlens_kv_padded.data(), te_page_table_k.data(),
        te_page_table_v.data(), te_rng_state.data(), max_seqlen_q, max_seqlen_kv, is_training,
        return_max_logit, cuda_graph, attn_scale, p_dropout, qkv_layout, o_format,
        qkv_scale_inv_format, bias_type, attn_mask_type, softmax_type, window_size[0],
        window_size[1], bottom_right_diagonal, workspace.data(), at::cuda::getCurrentCUDAStream());
  });

  // destroy tensor wrappers, but not allocated memory
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  // if training, [O, softmax-related tensors, rng_state]; if inference, [O]
  return output_tensors;
}

// fused attention BWD with separate Q, K and V
std::vector<py::object> fused_attn_bwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format, NVTE_QKV_Format do_format,
    NVTE_QKV_Layout dqkv_layout, NVTE_QKV_Format qkv_scale_inv_format,
    NVTE_QKV_Format do_scale_inv_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    NVTE_Softmax_Type softmax_type, const std::vector<int64_t> window_size,
    bool bottom_right_diagonal, bool deterministic, const at::Tensor cu_seqlens_q,
    const at::Tensor cu_seqlens_kv, const py::handle Q, const py::handle K, const py::handle V,
    const py::handle O, const py::handle dO, const at::ScalarType fake_dtype,
    const std::vector<at::Tensor> Aux_CTX_Tensors,
    const std::optional<at::Tensor> cu_seqlens_q_padded,
    const std::optional<at::Tensor> cu_seqlens_kv_padded, py::handle s_quantizer,
    py::handle dp_quantizer, py::handle dqkv_quantizer, bool cuda_graph) {
  auto none = py::none();

  // create QKV, O, dO tensor wrappers
  TensorWrapper te_Q, te_K, te_V, te_O, te_dO;
  te_Q = makeTransformerEngineTensor(Q, none);
  te_K = makeTransformerEngineTensor(K, none);
  te_V = makeTransformerEngineTensor(V, none);
  te_O = makeTransformerEngineTensor(O, none);
  te_dO = makeTransformerEngineTensor(dO, none);

  // create S and dP tensors
  auto [te_S, py_S, _s] = quantizer_helper(s_quantizer, {0}, DType::kFloat32, false, std::nullopt);
  auto [te_dP, py_dP, _dp] =
      quantizer_helper(dp_quantizer, {0}, DType::kFloat32, false, std::nullopt);

  // create dQ, dK, dV tensors
  TensorWrapper te_dQ, te_dK, te_dV;
  py::object py_dQ, py_dK, py_dV;
  std::optional<at::Tensor> dq_amax_buf, dk_amax_buf, dv_amax_buf;
  std::unique_ptr<Quantizer> dQKV_quantizer = convert_quantizer(dqkv_quantizer);
  std::vector<size_t> q_shape = convertShape(te_Q.shape());
  std::vector<size_t> k_shape = convertShape(te_K.shape());
  std::vector<size_t> v_shape = convertShape(te_V.shape());
  const DType dqkv_fake_dtype = GetTransformerEngineDType(fake_dtype);
  size_t ndim_q = q_shape.size();
  size_t ndim_kv = k_shape.size();
  std::vector<size_t> dQ_shape(ndim_q), dK_shape(ndim_kv), dV_shape(ndim_kv);
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  NVTE_QKV_Format dq_format = nvte_get_q_format(dqkv_layout);
  NVTE_QKV_Format dkv_format = nvte_get_kv_format(dqkv_layout);
  AttentionShape q_parsed(q_format, q_shape.data());
  size_t h_q = q_parsed.h(), d_qk = q_parsed.d();
  q_parsed.to_format(dq_format, dQ_shape.data());
  AttentionShape k_parsed(kv_format, k_shape.data());
  size_t h_kv = k_parsed.h();
  k_parsed.to_format(dkv_format, dK_shape.data());
  AttentionShape v_parsed(kv_format, v_shape.data());
  size_t d_v = v_parsed.d();
  v_parsed.to_format(dkv_format, dV_shape.data());
  at::Tensor dQ, dK, dV, dQKV, dKV;
  // FP16/BF16: dqkv_fake_dtype = kFloat16/kBFloat16, dQ/dK/dV.dtype = torch.float16/torch.bfloat16
  // FP8DS: dqkv_fake_dtype = kFloat16/kBFloat16, dQ/dK/dV.dtype = torch.uint8
  // FP8CS/MXFP8: dqkv_fake_dtype = kFloat16/kBFloat16, dQ/dK/dV.dtype = torch.float16/torch.bfloat16
  auto options = torch::TensorOptions().dtype(fake_dtype).device(torch::kCUDA);
  if (detail::IsFloat8Quantizers(dqkv_quantizer.ptr())) {
    options = options.dtype(torch::kUInt8);
  }

  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(dqkv_layout);
  std::vector<int64_t> tmp_shape;
  switch (layout_group) {
    case NVTE_QKV_Layout_Group::NVTE_3HD:
      tmp_shape = std::vector<int64_t>{dQ_shape.begin(), dQ_shape.end()};
      tmp_shape.insert(tmp_shape.begin() + tmp_shape.size() - 2, int64_t(3));
      dQKV = torch::empty(c10::IntArrayRef(tmp_shape), options);
      dQ = dQKV.index({"...", torch::indexing::Slice(0, 1, 1),
                       torch::indexing::Slice(0, torch::indexing::None, 1),
                       torch::indexing::Slice(0, torch::indexing::None, 1)})
               .squeeze(tmp_shape.size() - 3);
      dK = dQKV.index({"...", torch::indexing::Slice(1, 2, 1),
                       torch::indexing::Slice(0, torch::indexing::None, 1),
                       torch::indexing::Slice(0, torch::indexing::None, 1)})
               .squeeze(tmp_shape.size() - 3);
      dV = dQKV.index({"...", torch::indexing::Slice(2, torch::indexing::None, 1),
                       torch::indexing::Slice(0, torch::indexing::None, 1),
                       torch::indexing::Slice(0, torch::indexing::None, 1)})
               .squeeze(tmp_shape.size() - 3);
      break;
    case NVTE_QKV_Layout_Group::NVTE_H3D:
      tmp_shape = std::vector<int64_t>{dQ_shape.begin(), dQ_shape.end()};
      tmp_shape.insert(tmp_shape.begin() + tmp_shape.size() - 1, int64_t(3));
      dQKV = torch::empty(c10::IntArrayRef(tmp_shape), options);
      dQ = dQKV.index({"...", torch::indexing::Slice(0, 1, 1),
                       torch::indexing::Slice(0, torch::indexing::None, 1)})
               .squeeze(tmp_shape.size() - 2);
      dK = dQKV.index({"...", torch::indexing::Slice(1, 2, 1),
                       torch::indexing::Slice(0, torch::indexing::None, 1)})
               .squeeze(tmp_shape.size() - 2);
      dV = dQKV.index({"...", torch::indexing::Slice(2, torch::indexing::None, 1),
                       torch::indexing::Slice(0, torch::indexing::None, 1)})
               .squeeze(tmp_shape.size() - 2);
      break;
    case NVTE_QKV_Layout_Group::NVTE_HD_2HD:
      tmp_shape = std::vector<int64_t>(dQ_shape.begin(), dQ_shape.end());
      dQ = torch::empty(tmp_shape, options);
      tmp_shape = std::vector<int64_t>{dK_shape.begin(), dK_shape.end()};
      tmp_shape.insert(tmp_shape.begin() + tmp_shape.size() - 2, int64_t(2));
      dKV = torch::empty(c10::IntArrayRef(tmp_shape), options);
      dK = dKV.index({"...", torch::indexing::Slice(0, 1, 1),
                      torch::indexing::Slice(0, torch::indexing::None, 1),
                      torch::indexing::Slice(0, torch::indexing::None, 1)})
               .squeeze(tmp_shape.size() - 3);
      dV = dKV.index({"...", torch::indexing::Slice(1, torch::indexing::None, 1),
                      torch::indexing::Slice(0, torch::indexing::None, 1),
                      torch::indexing::Slice(0, torch::indexing::None, 1)})
               .squeeze(tmp_shape.size() - 3);
      break;
    case NVTE_QKV_Layout_Group::NVTE_HD_H2D:
      tmp_shape = std::vector<int64_t>(dQ_shape.begin(), dQ_shape.end());
      dQ = torch::empty(tmp_shape, options);
      tmp_shape = std::vector<int64_t>{dK_shape.begin(), dK_shape.end()};
      tmp_shape.insert(tmp_shape.begin() + tmp_shape.size() - 1, int64_t(2));
      dKV = torch::empty(c10::IntArrayRef(tmp_shape), options);
      dK = dKV.index({"...", torch::indexing::Slice(0, 1, 1),
                      torch::indexing::Slice(0, torch::indexing::None, 1)})
               .squeeze(tmp_shape.size() - 2);
      dV = dKV.index({"...", torch::indexing::Slice(1, torch::indexing::None, 1),
                      torch::indexing::Slice(0, torch::indexing::None, 1)})
               .squeeze(tmp_shape.size() - 2);
      break;
    case NVTE_QKV_Layout_Group::NVTE_HD_HD_HD:
    case NVTE_QKV_Layout_Group::NVTE_SD_SD_SD:
      tmp_shape = std::vector<int64_t>(dQ_shape.begin(), dQ_shape.end());
      dQ = torch::empty(tmp_shape, options);
      tmp_shape = std::vector<int64_t>(dK_shape.begin(), dK_shape.end());
      dK = torch::empty(tmp_shape, options);
      tmp_shape = std::vector<int64_t>(dV_shape.begin(), dV_shape.end());
      dV = torch::empty(tmp_shape, options);
      break;
    default:
      NVTE_ERROR("QKV layout not supported!");
  }

  std::tie(te_dQ, py_dQ, dq_amax_buf) =
      quantizer_helper(dqkv_quantizer, dQ_shape, dqkv_fake_dtype, true, dQ);
  std::tie(te_dK, py_dK, dk_amax_buf) =
      quantizer_helper(dqkv_quantizer, dK_shape, dqkv_fake_dtype, true, dK);
  std::tie(te_dV, py_dV, dv_amax_buf) =
      quantizer_helper(dqkv_quantizer, dV_shape, dqkv_fake_dtype, true, dV);

  // construct NVTE tensors
  if (detail::IsFloat8Quantizers(dqkv_quantizer.ptr())) {
    // FP8
    if (set_zero) {
      if (dq_format == NVTE_QKV_Format::NVTE_THD) {
        if (((h_q * d_qk) % block_size == 0) && dQ.is_contiguous()) {
          mha_fill(te_dQ, cu_seqlens_q.index({torch::indexing::Slice(-1, torch::indexing::None)}));
        } else {
          dQ.fill_(0);
        }
      }
      if (dkv_format == NVTE_QKV_Format::NVTE_THD) {
        if (((h_kv * d_qk) % block_size == 0) && ((h_kv * d_v) % block_size == 0) &&
            dK.is_contiguous() && dV.is_contiguous()) {
          mha_fill(te_dK, cu_seqlens_kv.index({torch::indexing::Slice(-1, torch::indexing::None)}));
          mha_fill(te_dV, cu_seqlens_kv.index({torch::indexing::Slice(-1, torch::indexing::None)}));
        } else {
          dK.fill_(0);
          dV.fill_(0);
        }
      }
    }
  } else if (dqkv_quantizer.is_none() ||
             detail::IsFloat8CurrentScalingQuantizers(dqkv_quantizer.ptr()) ||
             detail::IsMXFP8Quantizers(dqkv_quantizer.ptr())) {
    if (dq_format == NVTE_QKV_Format::NVTE_THD) {
      dQ.fill_(0);
    }
    if (dkv_format == NVTE_QKV_Format::NVTE_THD) {
      dK.fill_(0);
      dV.fill_(0);
    }
  } else {
    NVTE_ERROR("Fused attention only supports FP8 and BF16/FP16 data types. \n");
  }

  // create cu_seqlens tensorwrappers
  auto cu_seqlens_q_sizes = cu_seqlens_q.sizes().vec();
  std::vector<size_t> cu_seqlens_q_shape{cu_seqlens_q_sizes.begin(), cu_seqlens_q_sizes.end()};
  auto cu_seqlens_kv_sizes = cu_seqlens_kv.sizes().vec();
  std::vector<size_t> cu_seqlens_kv_shape{cu_seqlens_kv_sizes.begin(), cu_seqlens_kv_sizes.end()};
  TensorWrapper te_cu_seqlens_q, te_cu_seqlens_kv;
  te_cu_seqlens_q = makeTransformerEngineTensor(cu_seqlens_q.data_ptr(), cu_seqlens_q_shape,
                                                DType::kInt32, nullptr, nullptr, nullptr);
  te_cu_seqlens_kv = makeTransformerEngineTensor(cu_seqlens_kv.data_ptr(), cu_seqlens_kv_shape,
                                                 DType::kInt32, nullptr, nullptr, nullptr);

  TensorWrapper te_cu_seqlens_q_padded, te_cu_seqlens_kv_padded;
  if ((cu_seqlens_q_padded.has_value()) && (cu_seqlens_kv_padded.has_value())) {
    auto cu_seqlens_q_padded_sizes = cu_seqlens_q_padded.value().sizes().vec();
    std::vector<size_t> cu_seqlens_q_padded_shape{cu_seqlens_q_padded_sizes.begin(),
                                                  cu_seqlens_q_padded_sizes.end()};
    auto cu_seqlens_kv_padded_sizes = cu_seqlens_kv_padded.value().sizes().vec();
    std::vector<size_t> cu_seqlens_kv_padded_shape{cu_seqlens_kv_padded_sizes.begin(),
                                                   cu_seqlens_kv_padded_sizes.end()};
    te_cu_seqlens_q_padded = makeTransformerEngineTensor(cu_seqlens_q_padded.value().data_ptr(),
                                                         cu_seqlens_q_padded_shape, DType::kInt32);
    te_cu_seqlens_kv_padded = makeTransformerEngineTensor(
        cu_seqlens_kv_padded.value().data_ptr(), cu_seqlens_kv_padded_shape, DType::kInt32);
  }

  // convert auxiliary tensors from forward to NVTETensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);
  nvte_aux_tensor_pack.size = Aux_CTX_Tensors.size();
  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
    const std::vector<int64_t> &signed_shape = Aux_CTX_Tensors[i].sizes().vec();
    const std::vector<size_t> tmp(signed_shape.begin(), signed_shape.end());

    NVTEBasicTensor temp_data = {
        Aux_CTX_Tensors[i].data_ptr(),
        static_cast<NVTEDType>(GetTransformerEngineDType(Aux_CTX_Tensors[i].scalar_type())),
        nvte_make_shape(tmp.data(), tmp.size())};
    nvte_set_tensor_param(&nvte_aux_tensor_pack.tensors[i], kNVTERowwiseData, &temp_data);
  }

  // create dBias the same shape as Bias
  at::Tensor dBias;
  TensorWrapper te_dBias;
  if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
    if (nvte_aux_tensor_pack.size >= 2) {
      std::vector<int64_t> bias_shape(Aux_CTX_Tensors[nvte_aux_tensor_pack.size - 1].sizes().vec());
      dBias = torch::empty(bias_shape, options);
      te_dBias = makeTransformerEngineTensor(dBias);
    } else {
      dBias = torch::empty({1, static_cast<int64_t>(h_q), static_cast<int64_t>(max_seqlen_q),
                            static_cast<int64_t>(max_seqlen_kv)},
                           options);
      te_dBias = makeTransformerEngineTensor(dBias);
    }
    if (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD) {
      dBias.fill_(0);
    }
  }

  // create dSoftmaxOffset in the same shape as SoftmaxOffset
  at::Tensor dSoftmaxOffset;
  TensorWrapper te_dSoftmaxOffset;
  if (softmax_type != NVTE_VANILLA_SOFTMAX) {
    options = torch::TensorOptions().dtype(at::kFloat).device(torch::kCUDA);
    dSoftmaxOffset = torch::empty({1, static_cast<int64_t>(h_q), 1, 1}, options);
    te_dSoftmaxOffset = makeTransformerEngineTensor(dSoftmaxOffset);
  }

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_attn_bwd(
        te_Q.data(), te_K.data(), te_V.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
        &nvte_aux_tensor_pack, te_dQ.data(), te_dK.data(), te_dV.data(), te_dBias.data(),
        te_dSoftmaxOffset.data(), te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
        te_cu_seqlens_q_padded.data(), te_cu_seqlens_kv_padded.data(), max_seqlen_q, max_seqlen_kv,
        attn_scale, p_dropout, qkv_layout, o_format, do_format, dqkv_layout, qkv_scale_inv_format,
        do_scale_inv_format, bias_type, attn_mask_type, softmax_type, window_size[0],
        window_size[1], bottom_right_diagonal, deterministic, cuda_graph, workspace.data(),
        at::cuda::getCurrentCUDAStream());
  });

  // allocate memory for workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // execute kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_attn_bwd(
        te_Q.data(), te_K.data(), te_V.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
        &nvte_aux_tensor_pack, te_dQ.data(), te_dK.data(), te_dV.data(), te_dBias.data(),
        te_dSoftmaxOffset.data(), te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
        te_cu_seqlens_q_padded.data(), te_cu_seqlens_kv_padded.data(), max_seqlen_q, max_seqlen_kv,
        attn_scale, p_dropout, qkv_layout, o_format, do_format, dqkv_layout, qkv_scale_inv_format,
        do_scale_inv_format, bias_type, attn_mask_type, softmax_type, window_size[0],
        window_size[1], bottom_right_diagonal, deterministic, cuda_graph, workspace.data(),
        at::cuda::getCurrentCUDAStream());
  });

  // destroy tensor wrappers
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  return {py_dQ, py_dK, py_dV, py::cast(dBias), py::cast(dSoftmaxOffset)};
}

at::Tensor fa_prepare_fwd(at::Tensor qkvi) {
  NVTE_CHECK(qkvi.dim() == 4, "Expected 4-dim tensor.");
  NVTE_CHECK(qkvi.scalar_type() == at::ScalarType::Half ||
             qkvi.scalar_type() == at::ScalarType::BFloat16);
  NVTE_CHECK(qkvi.stride(3) == 1, "Wrong stride.");
  NVTE_CHECK(qkvi.stride(2) == 3 * qkvi.size(3), "Wrong stride.");
  NVTE_CHECK(qkvi.stride(1) == 3 * qkvi.size(3) * qkvi.size(2), "Wrong stride.");
  NVTE_CHECK(qkvi.stride(0) == 3 * qkvi.size(3) * qkvi.size(2) * qkvi.size(1), "Wrong stride.");

  // [s, b, n, h * 3] -> [3, b, s, n, h]
  std::vector<int64_t> shape = {3, qkvi.size(1), qkvi.size(0), qkvi.size(2), qkvi.size(3)};
  at::Tensor qkv = at::empty(shape, at::CUDA(qkvi.scalar_type()));

  auto te_qkvi = makeTransformerEngineTensor(qkvi);
  auto te_qkv = makeTransformerEngineTensor(qkv);

  nvte_prepare_flash_attn_fwd(te_qkvi.data(), te_qkv.data(), at::cuda::getCurrentCUDAStream());

  return qkv;
}

at::Tensor fa_prepare_bwd(at::Tensor q, at::Tensor k, at::Tensor v) {
  NVTE_CHECK(q.is_contiguous());
  NVTE_CHECK(k.is_contiguous());
  NVTE_CHECK(v.is_contiguous());
  NVTE_CHECK(q.dim() == 4, "Expected 4-dim tensor.");
  NVTE_CHECK(k.dim() == 4, "Expected 4-dim tensor.");
  NVTE_CHECK(v.dim() == 4, "Expected 4-dim tensor.");
  NVTE_CHECK(q.scalar_type() == at::ScalarType::Half ||
             q.scalar_type() == at::ScalarType::BFloat16);
  NVTE_CHECK(k.scalar_type() == q.scalar_type());
  NVTE_CHECK(v.scalar_type() == q.scalar_type());

  // 3 x [s, b, n, h] -> [b, s, n, 3 * h]
  std::vector<int64_t> shape = {q.size(1), q.size(0), q.size(2), 3 * q.size(3)};
  at::Tensor qkv = at::empty(shape, at::CUDA(q.scalar_type()));

  auto te_q = makeTransformerEngineTensor(q);
  auto te_k = makeTransformerEngineTensor(k);
  auto te_v = makeTransformerEngineTensor(v);
  auto te_qkv = makeTransformerEngineTensor(qkv);

  nvte_prepare_flash_attn_bwd(te_q.data(), te_k.data(), te_v.data(), te_qkv.data(),
                              at::cuda::getCurrentCUDAStream());

  return qkv;
}

std::vector<std::optional<at::Tensor>> multi_tensor_transpose_to_bhsd(
    std::vector<std::optional<at::Tensor>> inputs, const std::string &original_format,
    std::vector<std::optional<at::Tensor>> outputs) {
  NVTE_CHECK(original_format == "sbhd" || original_format == "bshd",
             "multi_tensor_transpose_to_bhsd: only BSHD/SBHD -> BHSD is currently supported. "
             "Got original_format=\"",
             original_format, "\".");
  const auto original_format_enum = (original_format == "sbhd") ? NVTE_SBHD : NVTE_BSHD;

  if (inputs.empty()) return {};

  const bool has_outputs = !outputs.empty();
  if (has_outputs) {
    NVTE_CHECK(outputs.size() == inputs.size(), "multi_tensor_transpose_to_bhsd: outputs.size() (",
               outputs.size(), ") != inputs.size() (", inputs.size(), ").");
  }

  std::vector<transformer_engine::TensorWrapper> te_ins, te_outs;
  std::vector<std::optional<at::Tensor>> result(inputs.size(), std::nullopt);

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].has_value()) continue;

    auto &input = inputs[i].value();
    NVTE_CHECK(input.is_cuda() && input.dim() == 4, "multi_tensor_transpose_to_bhsd: input ", i,
               " must be a 4D CUDA tensor.");
    input = input.contiguous();
    NVTE_CHECK(input.scalar_type() == at::ScalarType::Half ||
                   input.scalar_type() == at::ScalarType::BFloat16 ||
                   input.scalar_type() == at::ScalarType::Byte,
               "multi_tensor_transpose_to_bhsd: unsupported dtype at index ", i, ".");

    at::Tensor output;
    if (has_outputs && outputs[i].has_value()) {
      output = outputs[i].value();
    } else {
      int64_t B, S, H, D;
      if (original_format_enum == NVTE_SBHD) {
        S = input.size(0);
        B = input.size(1);
        H = input.size(2);
        D = input.size(3);
      } else {
        B = input.size(0);
        S = input.size(1);
        H = input.size(2);
        D = input.size(3);
      }
      output = at::empty({B, H, S, D}, input.options());
    }

    te_ins.push_back(makeTransformerEngineTensor(input));
    te_outs.push_back(makeTransformerEngineTensor(output));
    result[i] = output;
  }

  if (!te_ins.empty()) {
    std::vector<NVTETensor> nvte_ins(te_ins.size()), nvte_outs(te_outs.size());
    for (size_t j = 0; j < te_ins.size(); ++j) {
      nvte_ins[j] = te_ins[j].data();
      nvte_outs[j] = te_outs[j].data();
    }
    nvte_multi_tensor_transpose_to_bhsd(nvte_ins.data(), nvte_outs.data(), te_ins.size(),
                                        original_format_enum, at::cuda::getCurrentCUDAStream());
  }

  return result;
}

std::vector<at::Tensor> multi_tensor_pad_last_dim(std::vector<at::Tensor> inputs,
                                                  int64_t alignment) {
  const auto align = static_cast<size_t>(alignment);
  NVTE_CHECK(align > 0, "multi_tensor_pad_last_dim: alignment must be > 0.");
  NVTE_CHECK(!inputs.empty(), "multi_tensor_pad_last_dim: inputs must not be empty.");

  auto stream = at::cuda::getCurrentCUDAStream();
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());

  std::vector<size_t> kernel_indices;

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &input = inputs[i];

    NVTE_CHECK(input.dim() == 2, "multi_tensor_pad_last_dim: expected 2D input at index ", i,
               ", got ", input.dim(), "D.");
    NVTE_CHECK(input.is_cuda(), "multi_tensor_pad_last_dim: input must be a CUDA tensor at index ",
               i, ".");
    input = input.contiguous();

    const int64_t rows = input.size(0);
    const int64_t in_cols = input.size(1);
    const int64_t padded_cols =
        static_cast<int64_t>(DIVUP_TO_MULTIPLE(static_cast<size_t>(in_cols), align));

    if (in_cols == padded_cols) {
      outputs.push_back(input);
      continue;
    }

    at::Tensor output = at::empty({rows, padded_cols}, input.options());
    outputs.push_back(output);
    kernel_indices.push_back(outputs.size() - 1);
  }

  if (kernel_indices.empty()) return outputs;

  std::vector<transformer_engine::TensorWrapper> te_in_wrappers, te_out_wrappers;
  te_in_wrappers.reserve(kernel_indices.size());
  te_out_wrappers.reserve(kernel_indices.size());

  for (size_t idx : kernel_indices) {
    te_in_wrappers.push_back(makeTransformerEngineTensor(inputs[idx]));
    te_out_wrappers.push_back(makeTransformerEngineTensor(outputs[idx]));
  }

  std::vector<NVTETensor> nvte_inputs(te_in_wrappers.size());
  std::vector<NVTETensor> nvte_outputs(te_out_wrappers.size());
  for (size_t i = 0; i < te_in_wrappers.size(); ++i) {
    nvte_inputs[i] = te_in_wrappers[i].data();
    nvte_outputs[i] = te_out_wrappers[i].data();
  }

  nvte_multi_tensor_pad_last_dim(nvte_inputs.data(), nvte_outputs.data(), te_in_wrappers.size(),
                                 stream);

  return outputs;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Read the half of a THD tensor
 **************************************************************************************************/

at::Tensor thd_read_half_tensor(const at::Tensor &tensor, const at::Tensor &cu_seqlens,
                                int half_idx) {
  NVTE_CHECK(tensor.dim() == 3 || tensor.dim() == 4);
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);
  NVTE_CHECK(cu_seqlens.size(0) >= 2);

  // Shapes of q and dq are [t, h, d], so the dimension of "t" is 0
  // Shapes of kv and dkv are [2, t, h, d], so the dimension of "t" is 1
  int seq_dim = tensor.dim() == 3 ? 0 : 1;

  int num_heads = tensor.size(seq_dim + 1);
  int dim_per_head = tensor.size(seq_dim + 2);
  int hidden_size_in_bytes = num_heads * dim_per_head * c10::elementSize(tensor.scalar_type());

  // For 128-bits load/store
  NVTE_CHECK(hidden_size_in_bytes % 16 == 0);

  // Generate output
  std::vector<int64_t> shape(tensor.dim());
  for (size_t i = 0; i < shape.size(); i++) {
    shape[i] = tensor.size(i);
  }
  shape[seq_dim] /= 2;
  at::Tensor half = at::empty(shape, at::CUDA(tensor.scalar_type()));

  auto te_tensor = makeTransformerEngineTensor(tensor);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_half = makeTransformerEngineTensor(half);

  nvte_cp_thd_read_half_tensor(te_tensor.data(), te_cu_seqlens.data(), te_half.data(), half_idx,
                               at::cuda::getCurrentCUDAStream());

  return half;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: softmax_lse related operations
 **************************************************************************************************/

void thd_second_half_lse_correction(at::Tensor lse, const at::Tensor &lse_per_step,
                                    const at::Tensor &cu_seqlens, bool lse_packed) {
  NVTE_CHECK(lse.scalar_type() == at::ScalarType::Float);
  NVTE_CHECK(lse_per_step.scalar_type() == at::ScalarType::Float);
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  int batch, num_heads, lse_seqlen, second_half_lse_seqlen;

  if (lse_packed) {
    NVTE_CHECK(lse.dim() == 2);
    NVTE_CHECK(lse_per_step.dim() == 2);

    batch = cu_seqlens.size(0) - 1;
    num_heads = lse.size(0);
    lse_seqlen = lse.size(1);
    second_half_lse_seqlen = lse_per_step.size(1);

    NVTE_CHECK(lse_per_step.size(0) == num_heads);
    NVTE_CHECK(second_half_lse_seqlen >= lse_seqlen / 2);
  } else {
    NVTE_CHECK(lse.dim() == 3);
    NVTE_CHECK(lse_per_step.dim() == 3);

    batch = lse.size(0);
    num_heads = lse.size(1);
    lse_seqlen = lse.size(2);
    second_half_lse_seqlen = lse_per_step.size(2);

    NVTE_CHECK(lse_per_step.size(0) == batch);
    NVTE_CHECK(lse_per_step.size(1) == num_heads);
    NVTE_CHECK(second_half_lse_seqlen == lse_seqlen / 2);
    NVTE_CHECK(cu_seqlens.size(0) == batch + 1);
  }

  auto te_lse = makeTransformerEngineTensor(lse);
  auto te_lse_per_step = makeTransformerEngineTensor(lse_per_step);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);

  nvte_cp_thd_second_half_lse_correction(te_lse.data(), te_lse_per_step.data(),
                                         te_cu_seqlens.data(), lse_packed,
                                         at::cuda::getCurrentCUDAStream());
}

at::Tensor thd_read_second_half_lse(const at::Tensor &lse, const at::Tensor &cu_seqlens,
                                    bool lse_packed, int second_half_lse_seqlen) {
  NVTE_CHECK(lse.scalar_type() == at::ScalarType::Float);
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  int batch, num_heads, lse_seqlen;
  std::vector<int64_t> shape;

  if (lse_packed) {
    NVTE_CHECK(lse.dim() == 2);

    batch = cu_seqlens.size(0) - 1;
    num_heads = lse.size(0);
    lse_seqlen = lse.size(1);

    NVTE_CHECK(second_half_lse_seqlen >= lse_seqlen / 2);

    shape = {num_heads, second_half_lse_seqlen};
  } else {
    NVTE_CHECK(lse.dim() == 3);

    batch = lse.size(0);
    num_heads = lse.size(1);
    lse_seqlen = lse.size(2);

    NVTE_CHECK(cu_seqlens.size(0) == batch + 1);
    NVTE_CHECK(second_half_lse_seqlen == lse_seqlen / 2);

    shape = {batch, num_heads, second_half_lse_seqlen};
  }

  at::Tensor half_lse = at::zeros(shape, at::CUDA(lse.scalar_type()));

  auto te_lse = makeTransformerEngineTensor(lse);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_half_lse = makeTransformerEngineTensor(half_lse);

  nvte_cp_thd_read_second_half_lse(te_lse.data(), te_cu_seqlens.data(), te_half_lse.data(),
                                   lse_packed, second_half_lse_seqlen,
                                   at::cuda::getCurrentCUDAStream());

  return half_lse;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Out correction in forward
 **************************************************************************************************/

void thd_out_correction(at::Tensor out, const at::Tensor &out_per_step, const at::Tensor &lse,
                        const at::Tensor &lse_per_step, const at::Tensor &cu_seqlens,
                        bool only_second_half, bool lse_packed) {
  auto te_out = makeTransformerEngineTensor(out);
  auto te_out_per_step = makeTransformerEngineTensor(out_per_step);
  auto te_lse = makeTransformerEngineTensor(lse);
  auto te_lse_per_step = makeTransformerEngineTensor(lse_per_step);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  nvte_cp_thd_out_correction(te_out.data(), te_out_per_step.data(), te_lse.data(),
                             te_lse_per_step.data(), te_cu_seqlens.data(), only_second_half,
                             lse_packed, at::cuda::getCurrentCUDAStream());
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Gradients correction in backward
 **************************************************************************************************/

void thd_grad_correction(at::Tensor grad, const at::Tensor &grad_per_step,
                         const at::Tensor &cu_seqlens, const std::string &first_half,
                         const std::string &second_half) {
  auto te_grad = makeTransformerEngineTensor(grad);
  auto te_grad_per_step = makeTransformerEngineTensor(grad_per_step);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  nvte_cp_thd_grad_correction(te_grad.data(), te_grad_per_step.data(), te_cu_seqlens.data(),
                              first_half.data(), second_half.data(),
                              at::cuda::getCurrentCUDAStream());
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Generate partitioned indices for input tokens
 **************************************************************************************************/

at::Tensor thd_get_partitioned_indices(const at::Tensor &cu_seqlens, int total_tokens,
                                       int world_size, int rank) {
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);
  NVTE_CHECK(cu_seqlens.size(0) >= 2);
  NVTE_CHECK(rank >= 0 && rank < world_size);
  NVTE_CHECK(world_size > 0);
  NVTE_CHECK(total_tokens > 0 && total_tokens % (world_size * 2) == 0);

  std::vector<int64_t> shape = {total_tokens / world_size};
  at::Tensor output = at::empty(shape, at::CUDA(at::ScalarType::Int));

  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_output = makeTransformerEngineTensor(output);

  nvte_cp_thd_get_partitioned_indices(te_cu_seqlens.data(), te_output.data(), total_tokens,
                                      world_size, rank, at::cuda::getCurrentCUDAStream());

  return output;
}

/***************************************************************************************************
 * KV Cache: Convert a tensor from qkv_format = thd to qkv_format = bshd
 **************************************************************************************************/

at::Tensor convert_thd_to_bshd(at::Tensor tensor, at::Tensor cu_seqlens, int b, int max_seq_len) {
  int h = tensor.size(1);
  int d = tensor.size(2);
  std::vector<int64_t> shape = {b, max_seq_len, h, d};
  at::Tensor new_tensor = at::zeros(shape, at::CUDA(tensor.scalar_type()));

  auto te_tensor = makeTransformerEngineTensor(tensor);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_new_tensor = makeTransformerEngineTensor(new_tensor);

  nvte_convert_thd_to_bshd(te_tensor.data(), te_cu_seqlens.data(), te_new_tensor.data(), b,
                           max_seq_len, at::cuda::getCurrentCUDAStream());

  return new_tensor;
}

/***************************************************************************************************
 * KV Cache: Convert a tensor from qkv_format = bshd to qkv_format = thd
 **************************************************************************************************/

at::Tensor convert_bshd_to_thd(at::Tensor tensor, at::Tensor cu_seqlens, int t) {
  int h = tensor.size(2);
  int d = tensor.size(3);
  std::vector<int64_t> shape = {t, h, d};
  at::Tensor new_tensor = at::zeros(shape, at::CUDA(tensor.scalar_type()));

  auto te_tensor = makeTransformerEngineTensor(tensor);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_new_tensor = makeTransformerEngineTensor(new_tensor);

  nvte_convert_bshd_to_thd(te_tensor.data(), te_cu_seqlens.data(), te_new_tensor.data(), t,
                           at::cuda::getCurrentCUDAStream());

  return new_tensor;
}

void copy_to_kv_cache(at::Tensor new_k, at::Tensor new_v, at::Tensor k_cache, at::Tensor v_cache,
                      at::Tensor page_table, at::Tensor cu_new_lens, at::Tensor cu_cached_lens,
                      NVTE_QKV_Format qkv_format, int b, int max_ctx_len, int max_seq_len,
                      int max_pages_per_seq, bool is_non_paged) {
  NVTE_CHECK(k_cache.scalar_type() == v_cache.scalar_type() &&
                 new_k.scalar_type() == new_v.scalar_type() &&
                 new_k.scalar_type() == k_cache.scalar_type(),
             "new_k, new_v, k_cache and v_cache must be of the same data type.");
  NVTE_CHECK(qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD ||
                 qkv_format == NVTE_QKV_Format::NVTE_THD,
             "qkv_format must be {BSHD, SBHD, THD}.");

  auto te_new_k = makeTransformerEngineTensor(new_k);
  auto te_new_v = makeTransformerEngineTensor(new_v);
  auto te_k_cache = makeTransformerEngineTensor(k_cache);
  auto te_v_cache = makeTransformerEngineTensor(v_cache);
  auto te_page_table = makeTransformerEngineTensor(page_table);
  auto te_cu_new_lens = makeTransformerEngineTensor(cu_new_lens);
  auto te_cu_cached_lens = makeTransformerEngineTensor(cu_cached_lens);

  nvte_copy_to_kv_cache(te_new_k.data(), te_new_v.data(), te_k_cache.data(), te_v_cache.data(),
                        te_page_table.data(), te_cu_new_lens.data(), te_cu_cached_lens.data(),
                        qkv_format, b, max_ctx_len, max_seq_len, max_pages_per_seq, is_non_paged,
                        at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
