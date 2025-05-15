/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "extensions.h"
#include "pybind.h"

namespace {

constexpr int block_size = 512;

// fast zero-fills of tensors
void mha_fill(const transformer_engine::TensorWrapper &self, const at::Tensor &start_index) {
  std::vector<size_t> shape = transformer_engine::pytorch::convertShape(self.shape());

  auto max_tokens = shape[0];
  auto fcd_size = 1;
  for (int i = 1; i <= shape.size(); i++) {
    fcd_size *= shape[i];
  }

  NVTE_CHECK(fcd_size % block_size == 0, "input size not aligned to block size");

  size_t element_size = transformer_engine::pytorch::typeToSize(self.dtype());
  int32_t start_row = start_index.data_ptr<int32_t>()[0];
  void *base_ptr = static_cast<char *>(self.get_rowwise_data().data_ptr) +
                   static_cast<size_t>(start_row) * fcd_size * element_size;
  size_t num_rows_to_zero = max_tokens - start_row;
  size_t total_bytes = num_rows_to_zero * fcd_size * element_size;

  NVTE_SCOPED_GIL_RELEASE(
      { nvte_memset(base_ptr, 0, total_bytes, at::cuda::getCurrentCUDAStream()); });
}

void unpack(at::PhiloxCudaState arg, int64_t *rng_state_ptr) {
  NVTE_SCOPED_GIL_RELEASE({
    nvte_extract_seed_and_offset(rng_state_ptr, arg.captured_, arg.seed_.ptr, arg.seed_.val,
                                 arg.offset_.ptr, arg.offset_.val, arg.offset_intragraph_,
                                 at::cuda::getCurrentCUDAStream());
  });
}

// extract PhiloxCudaState from CUDA random number generator
at::PhiloxCudaState init_philox_state(at::CUDAGeneratorImpl *gen, size_t elts_per_thread) {
  at::PhiloxCudaState philox_args;
  std::lock_guard<std::mutex> lock(gen->mutex_);
  philox_args = gen->philox_cuda_state(elts_per_thread);
  return philox_args;
}

}  // namespace

namespace transformer_engine::pytorch {

// get the fused attention backend
NVTE_Fused_Attn_Backend get_fused_attn_backend(
    const DType q_dtype, const DType kv_dtype, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, float p_dropout, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v,
    int64_t window_size_left, int64_t window_size_right) {
  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      static_cast<NVTEDType>(q_dtype), static_cast<NVTEDType>(kv_dtype), qkv_layout, bias_type,
      attn_mask_type, p_dropout, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv,
      head_dim_qk, head_dim_v, window_size_left, window_size_right);
  return fused_attention_backend;
}

// fused attention FWD with separate Q, K and V tensors
std::vector<py::object> fused_attn_fwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training, float attn_scale, float p_dropout,
    bool set_zero, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, const std::vector<int64_t> window_size,
    const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv, const py::handle Q,
    const py::handle K, const py::handle V, const at::ScalarType fake_dtype,
    const std::optional<at::Tensor> cu_seqlens_q_padded,
    const std::optional<at::Tensor> cu_seqlens_kv_padded,
    const std::optional<at::Tensor> page_table_k, const std::optional<at::Tensor> page_table_v,
    py::handle s_quantizer, py::handle o_quantizer, const std::optional<at::Tensor> Bias,
    const std::optional<at::Generator> rng_gen, size_t rng_elts_per_thread) {
  TensorWrapper te_Q, te_K, te_V, te_O, te_S;

  auto none = py::none();
  std::unique_ptr<Quantizer> S_quantizer = convert_quantizer(s_quantizer);
  std::unique_ptr<Quantizer> O_quantizer = convert_quantizer(o_quantizer);

  te_Q = makeTransformerEngineTensor(Q, none);
  te_K = makeTransformerEngineTensor(K, none);
  te_V = makeTransformerEngineTensor(V, none);

  // If qkv has FP8 dtype, fake_dtype_te is equal to the fake dtype of q, k, v - needed since torch do not have fp8 types.
  const DType qkv_type = te_Q.dtype();
  const DType fake_dtype_te = GetTransformerEngineDType(fake_dtype);

  std::vector<size_t> q_shape = convertShape(te_Q.shape());
  std::vector<size_t> k_shape = convertShape(te_K.shape());
  std::vector<size_t> v_shape = convertShape(te_V.shape());
  auto options = torch::TensorOptions().dtype(GetATenDType(qkv_type)).device(torch::kCUDA);
  // create output tensor O

  auto o_shape = std::vector<size_t>{q_shape.begin(), q_shape.end()};
  o_shape[o_shape.size() - 1] = v_shape[v_shape.size() - 1];
  py::object o_python, s_python;
  std::tie(te_O, o_python) = O_quantizer->create_tensor(o_shape, fake_dtype_te);
  std::tie(te_S, s_python) = S_quantizer->create_tensor({0}, DType::kFloat32);
  auto o_shape_int64 = std::vector<int64_t>{o_shape.begin(), o_shape.end()};

  // construct NVTE tensors
  TensorWrapper te_Bias;
  TensorWrapper te_cu_seqlens_q, te_cu_seqlens_kv;
  TensorWrapper te_cu_seqlens_q_padded, te_cu_seqlens_kv_padded;
  TensorWrapper te_page_table_k, te_page_table_v;
  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    // FP8
    auto h = q_shape[q_shape.size() - 2];
    auto d = q_shape[q_shape.size() - 1];
    if (set_zero && ((h * d) % block_size == 0) &&
        (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD)) {
      mha_fill(te_O, cu_seqlens_q.index({torch::indexing::Slice(-1, torch::indexing::None)}));
    } else {
      te_O.zero_(at::cuda::getCurrentCUDAStream());
    }
  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    if (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD) {
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

  // extract rng seed and offset
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      rng_gen, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox_args = init_philox_state(gen, rng_elts_per_thread);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  unpack(philox_args, static_cast<int64_t *>(rng_state.data_ptr()));
  auto te_rng_state = makeTransformerEngineTensor(rng_state);

  // create auxiliary output tensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_attn_fwd(
        te_Q.data(), te_K.data(), te_V.data(), te_Bias.data(), te_S.data(), te_O.data(),
        &nvte_aux_tensor_pack, te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
        te_cu_seqlens_q_padded.data(), te_cu_seqlens_kv_padded.data(), te_page_table_k.data(),
        te_page_table_v.data(), te_rng_state.data(), max_seqlen_q, max_seqlen_kv, is_training,
        attn_scale, p_dropout, qkv_layout, bias_type, attn_mask_type, window_size[0],
        window_size[1], workspace.data(), at::cuda::getCurrentCUDAStream());
  });

  // allocate memory for workspace and auxiliary output tensors
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // output_tensors = [O, nvte_aux_tensor_pack.tensors]
  std::vector<py::object> output_tensors;
  output_tensors.push_back(o_python);
  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
    // allocate memory for nvte_aux_tensor_pack.tensors
    at::Tensor output_tensor;
    if (nvte_aux_tensor_pack.size >= 2) {
      if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) && (Bias.has_value())) {
        if (i < nvte_aux_tensor_pack.size - 2) {
          NVTEShape temp_shape = nvte_tensor_shape(nvte_aux_tensor_pack.tensors[i]);
          output_tensor = allocateSpace(
              nvte_shape_to_vector(temp_shape),
              static_cast<DType>(nvte_tensor_type(nvte_aux_tensor_pack.tensors[i])), false);
        } else if (i == nvte_aux_tensor_pack.size - 2) {
          output_tensor = rng_state;
        } else if (i == nvte_aux_tensor_pack.size - 1) {
          output_tensor = Bias.value();
        }
      } else {
        NVTEShape temp_shape = nvte_tensor_shape(nvte_aux_tensor_pack.tensors[i]);
        output_tensor =
            (i < nvte_aux_tensor_pack.size - 1)
                ? allocateSpace(
                      nvte_shape_to_vector(temp_shape),
                      static_cast<DType>(nvte_tensor_type(nvte_aux_tensor_pack.tensors[i])), false)
                : rng_state;
      }
    } else {
      NVTEShape temp_shape = nvte_tensor_shape(nvte_aux_tensor_pack.tensors[i]);
      output_tensor = allocateSpace(
          nvte_shape_to_vector(temp_shape),
          static_cast<DType>(nvte_tensor_type(nvte_aux_tensor_pack.tensors[i])), false);
    }
    output_tensors.push_back(py::cast(output_tensor));
    NVTEBasicTensor temp_data = {output_tensor.data_ptr(),
                                 nvte_tensor_type(nvte_aux_tensor_pack.tensors[i]),
                                 nvte_tensor_shape(nvte_aux_tensor_pack.tensors[i])};
    nvte_set_tensor_param(&nvte_aux_tensor_pack.tensors[i], kNVTERowwiseData, &temp_data);
  }

  // execute the kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_attn_fwd(
        te_Q.data(), te_K.data(), te_V.data(), te_Bias.data(), te_S.data(), te_O.data(),
        &nvte_aux_tensor_pack, te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
        te_cu_seqlens_q_padded.data(), te_cu_seqlens_kv_padded.data(), te_page_table_k.data(),
        te_page_table_v.data(), te_rng_state.data(), max_seqlen_q, max_seqlen_kv, is_training,
        attn_scale, p_dropout, qkv_layout, bias_type, attn_mask_type, window_size[0],
        window_size[1], workspace.data(), at::cuda::getCurrentCUDAStream());
  });

  // destroy tensor wrappers, but not allocated memory
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  // if training, [O, softmax-related tensors, rng_state]; if inference, [O]
  return output_tensors;
}

// fused attention BWD with separate Q, K and V
std::vector<py::object> fused_attn_bwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const std::vector<int64_t> window_size, bool deterministic, const at::Tensor cu_seqlens_q,
    const at::Tensor cu_seqlens_kv, const py::handle Q, const py::handle K, const py::handle V,
    const py::handle O, const py::handle dO, const at::ScalarType fake_dtype, const DType dqkv_type,
    const std::vector<at::Tensor> Aux_CTX_Tensors,
    const std::optional<at::Tensor> cu_seqlens_q_padded,
    const std::optional<at::Tensor> cu_seqlens_kv_padded, py::handle s_quantizer,
    py::handle dp_quantizer, py::handle dqkv_quantizer) {
  auto none = py::none();
  TensorWrapper te_Q, te_K, te_V, te_O, te_dO, te_S, te_dP, te_dQ, te_dK, te_dV;
  te_Q = makeTransformerEngineTensor(Q, none);
  te_K = makeTransformerEngineTensor(K, none);
  te_V = makeTransformerEngineTensor(V, none);
  te_O = makeTransformerEngineTensor(O, none);
  te_dO = makeTransformerEngineTensor(dO, none);
  // qkv type from the te_Q
  std::unique_ptr<Quantizer> dQKV_quantizer = convert_quantizer(dqkv_quantizer);
  const DType qkv_type = te_Q.dtype();
  const DType fake_dtype_te = GetTransformerEngineDType(fake_dtype);

  py::object s_python, dp_python;
  std::unique_ptr<Quantizer> S_quantizer = convert_quantizer(s_quantizer);
  std::unique_ptr<Quantizer> dP_quantizer = convert_quantizer(dp_quantizer);
  std::tie(te_S, s_python) = S_quantizer->create_tensor({0}, DType::kFloat32);
  std::tie(te_dP, dp_python) = dP_quantizer->create_tensor({0}, DType::kFloat32);

  std::vector<size_t> q_shape = convertShape(te_Q.shape());
  std::vector<size_t> k_shape = convertShape(te_K.shape());
  std::vector<size_t> v_shape = convertShape(te_V.shape());
  auto h_q = q_shape[q_shape.size() - 2];
  auto h_kv = k_shape[k_shape.size() - 2];
  auto d_qk = q_shape[q_shape.size() - 1];
  auto d_v = v_shape[v_shape.size() - 1];
  auto options = torch::TensorOptions().dtype(GetATenDType(dqkv_type)).device(torch::kCUDA);
  std::vector<size_t> o_shape{q_shape.begin(), q_shape.end()};
  o_shape[o_shape.size() - 1] = d_v;

  at::Tensor dQ, dK, dV, dQKV, dKV;
  py::object py_dQ, py_dK, py_dV;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  std::vector<int64_t> tmp_shape;

  switch (layout_group) {
    case NVTE_QKV_Layout_Group::NVTE_3HD:
      tmp_shape = std::vector<int64_t>{q_shape.begin(), q_shape.end()};
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
      tmp_shape = std::vector<int64_t>{q_shape.begin(), q_shape.end()};
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
      tmp_shape = std::vector<int64_t>(q_shape.begin(), q_shape.end());
      dQ = torch::empty(tmp_shape, options);
      tmp_shape = std::vector<int64_t>{k_shape.begin(), k_shape.end()};
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
      tmp_shape = std::vector<int64_t>(q_shape.begin(), q_shape.end());
      dQ = torch::empty(tmp_shape, options);
      tmp_shape = std::vector<int64_t>{k_shape.begin(), k_shape.end()};
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
      tmp_shape = std::vector<int64_t>(q_shape.begin(), q_shape.end());
      dQ = torch::empty(tmp_shape, options);
      tmp_shape = std::vector<int64_t>(k_shape.begin(), k_shape.end());
      dK = torch::empty(tmp_shape, options);
      tmp_shape = std::vector<int64_t>(v_shape.begin(), v_shape.end());
      dV = torch::empty(tmp_shape, options);
      break;
    default:
      NVTE_ERROR("QKV layout not supported!");
  }
  std::tie(te_dQ, py_dQ) = dQKV_quantizer->create_tensor(q_shape, fake_dtype_te, dQ);
  std::tie(te_dK, py_dK) = dQKV_quantizer->create_tensor(k_shape, fake_dtype_te, dK);
  std::tie(te_dV, py_dV) = dQKV_quantizer->create_tensor(v_shape, fake_dtype_te, dV);

  // construct NVTE tensors
  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    // FP8
    if (set_zero && ((h_q * d_qk) % block_size == 0) && ((h_kv * d_qk) % block_size == 0) &&
        dQ.is_contiguous() && dK.is_contiguous() && dV.is_contiguous() &&
        (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD)) {
      mha_fill(te_dQ, cu_seqlens_q.index({torch::indexing::Slice(-1, torch::indexing::None)}));
      mha_fill(te_dK, cu_seqlens_kv.index({torch::indexing::Slice(-1, torch::indexing::None)}));
      mha_fill(te_dV, cu_seqlens_kv.index({torch::indexing::Slice(-1, torch::indexing::None)}));
    } else {
      dQ.fill_(0);
      dK.fill_(0);
      dV.fill_(0);
    }

  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    if (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD) {
      dQ.fill_(0);
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

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_attn_bwd(
        te_Q.data(), te_K.data(), te_V.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
        &nvte_aux_tensor_pack, te_dQ.data(), te_dK.data(), te_dV.data(), te_dBias.data(),
        te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(), te_cu_seqlens_q_padded.data(),
        te_cu_seqlens_kv_padded.data(), max_seqlen_q, max_seqlen_kv, attn_scale, p_dropout,
        qkv_layout, bias_type, attn_mask_type, window_size[0], window_size[1], deterministic,
        workspace.data(), at::cuda::getCurrentCUDAStream());
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
        te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(), te_cu_seqlens_q_padded.data(),
        te_cu_seqlens_kv_padded.data(), max_seqlen_q, max_seqlen_kv, attn_scale, p_dropout,
        qkv_layout, bias_type, attn_mask_type, window_size[0], window_size[1], deterministic,
        workspace.data(), at::cuda::getCurrentCUDAStream());
  });

  // destroy tensor wrappers
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  return {py_dQ, py_dK, py_dV, py::cast(dBias)};
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

  int batch = cu_seqlens.size(0) - 1;
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

  int batch = cu_seqlens.size(0) - 1;

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
  int max_seq_len = tensor.size(1);
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
