/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

constexpr int block_size = 512;
constexpr int ctas_per_sm = 4;

// get the fused attention backend
NVTE_Fused_Attn_Backend get_fused_attn_backend(const transformer_engine::DType q_dtype,
                                               const transformer_engine::DType kv_dtype,
                                               NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                               NVTE_Mask_Type attn_mask_type, float p_dropout,
                                               size_t num_attn_heads, size_t num_gqa_groups,
                                               size_t max_seqlen_q, size_t max_seqlen_kv,
                                               size_t head_dim) {
  NVTE_Fused_Attn_Backend fused_attention_backend =
      nvte_get_fused_attn_backend(static_cast<NVTEDType>(q_dtype), static_cast<NVTEDType>(kv_dtype),
                                  qkv_layout, bias_type, attn_mask_type, p_dropout, num_attn_heads,
                                  num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim);
  return fused_attention_backend;
}

// fast zero-fills of tensors
template <typename scalar_t>
__global__ void __launch_bounds__(block_size)
    mha_fill_kernel(scalar_t *out_tensor, const int32_t *const start_row, const size_t num_rows) {
  size_t row_stride = gridDim.y * blockDim.x;
  size_t row_index = blockIdx.x + static_cast<size_t>(start_row[0]);
  size_t col_index = blockIdx.y * blockDim.x + threadIdx.x;
  while (row_index < num_rows) {
    out_tensor[row_index * row_stride + col_index] = 0;
    row_index += gridDim.x;
  }
}

// fast zero-fills of tensors
void mha_fill(const at::Tensor &self, const at::Tensor &start_index) {
  auto max_tokens = self.size(0);
  auto self_2d = self.view({max_tokens, -1});
  auto fcd_size = self_2d.size(1);
  TORCH_CHECK(self.is_contiguous(), "input not contiguous");
  TORCH_CHECK(fcd_size % block_size == 0, "input size not aligned to block size");
  const int num_mp = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  uint64_t num_blk_y = (uint64_t)(fcd_size / block_size);
  uint64_t num_blk_x = (uint64_t)((num_mp * ctas_per_sm + num_blk_y - 1) / num_blk_y);
  dim3 dim_grid(num_blk_x, num_blk_y);
  dim3 dim_block(block_size);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, self_2d.scalar_type(), "mha_fill", [&]() {
        mha_fill_kernel<<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            self_2d.data_ptr<scalar_t>(), static_cast<int32_t *>(start_index.data_ptr()),
            max_tokens);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

// extract seed and offset from PhiloxCudaState
__global__ void unpack(at::PhiloxCudaState arg, int64_t *rng_state_ptr) {
  if (arg.captured_) {
    rng_state_ptr[0] = static_cast<int64_t>(*arg.seed_.ptr);
    rng_state_ptr[1] =
        static_cast<int64_t>(*(arg.offset_.ptr) + static_cast<int64_t>(arg.offset_intragraph_));
  } else {
    rng_state_ptr[0] = static_cast<int64_t>(arg.seed_.val);
    rng_state_ptr[1] = static_cast<int64_t>(arg.offset_.val);
  }
}

// extract PhiloxCudaState from CUDA random number generator
at::PhiloxCudaState init_philox_state(at::CUDAGeneratorImpl *gen, size_t elts_per_thread) {
  at::PhiloxCudaState philox_args;
  std::lock_guard<std::mutex> lock(gen->mutex_);
  philox_args = gen->philox_cuda_state(elts_per_thread);
  return philox_args;
}

// fused attention FWD with packed QKV
std::vector<at::Tensor> fused_attn_fwd_qkvpacked(
    size_t max_seqlen, bool is_training, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const at::Tensor cu_seqlens, const at::Tensor QKV, const transformer_engine::DType qkv_type,
    const c10::optional<at::Tensor> cu_seqlens_padded, const c10::optional<at::Tensor> descale_QKV,
    const c10::optional<at::Tensor> descale_S, const c10::optional<at::Tensor> scale_S,
    const c10::optional<at::Tensor> scale_O, c10::optional<at::Tensor> amax_S,
    c10::optional<at::Tensor> amax_O, const c10::optional<at::Tensor> Bias,
    const c10::optional<at::Generator> rng_gen, size_t rng_elts_per_thread) {
  using namespace transformer_engine;

  auto qkv_sizes = QKV.sizes().vec();
  std::vector<size_t> qkv_shape{qkv_sizes.begin(), qkv_sizes.end()};
  std::vector<size_t> q_shape;
  for (auto i : qkv_shape) {
    if (i != 3) {
      q_shape.push_back(i);
    }
  }
  std::vector<int64_t> o_shape{q_shape.begin(), q_shape.end()};

  // create output tensor O
  auto options = torch::TensorOptions().dtype(GetATenDType(qkv_type)).device(torch::kCUDA);
  auto O = torch::empty(o_shape, options);

  // construct NVTE tensors
  TensorWrapper te_QKV, te_S, te_O, te_Bias, te_cu_seqlens, te_cu_seqlens_padded;
  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    // FP8
    auto h = q_shape[q_shape.size() - 2];
    auto d = q_shape[q_shape.size() - 1];
    if (set_zero && ((h * d) % block_size == 0) &&
        (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD)) {
      mha_fill(O, cu_seqlens.index({torch::indexing::Slice(-1, torch::indexing::None)}));
    } else {
      O.fill_(0);
    }
    if ((!descale_QKV.has_value()) || (!descale_S.has_value()) || (!scale_S.has_value()) ||
        (!scale_O.has_value()) || (!amax_S.has_value()) || (!amax_O.has_value())) {
      std::string err_tensors = "descale_QKV, descale_S, scale_S, scale_O, amax_S and amax_O ";
      NVTE_ERROR(err_tensors + std::string("are required for FP8 operation. \n"));
    }
    te_QKV = makeTransformerEngineTensor(QKV.data_ptr(), qkv_shape, qkv_type, nullptr, nullptr,
                                         descale_QKV.value().data_ptr());
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, amax_S.value().data_ptr(),
                                       scale_S.value().data_ptr(), descale_S.value().data_ptr());
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, amax_O.value().data_ptr(),
                                       scale_O.value().data_ptr(), nullptr);
  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    // BF16 or FP16
    te_QKV =
        makeTransformerEngineTensor(QKV.data_ptr(), qkv_shape, qkv_type, nullptr, nullptr, nullptr);
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr, nullptr, nullptr);
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, nullptr, nullptr, nullptr);
  } else {
    NVTE_ERROR("Fused attention only supports FP8 and BF16/FP16 data types. \n");
  }
  if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) && (Bias.has_value())) {
    auto bias_sizes = Bias.value().sizes().vec();
    std::vector<size_t> bias_shape{bias_sizes.begin(), bias_sizes.end()};
    te_Bias = makeTransformerEngineTensor(Bias.value().data_ptr(), bias_shape, DType::kFloat32,
                                          nullptr, nullptr, nullptr);
  }
  auto cu_seqlens_sizes = cu_seqlens.sizes().vec();
  std::vector<size_t> cu_seqlens_shape{cu_seqlens_sizes.begin(), cu_seqlens_sizes.end()};
  te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens.data_ptr(), cu_seqlens_shape,
                                              DType::kInt32, nullptr, nullptr, nullptr);

  if (cu_seqlens_padded.has_value()) {
    auto cu_seqlens_padded_sizes = cu_seqlens_padded.value().sizes().vec();
    std::vector<size_t> cu_seqlens_padded_shape{cu_seqlens_padded_sizes.begin(),
                                                cu_seqlens_padded_sizes.end()};
    te_cu_seqlens_padded =
        makeTransformerEngineTensor(cu_seqlens_padded.value().data_ptr(), cu_seqlens_padded_shape,
                                    DType::kInt32, nullptr, nullptr, nullptr);
  }

  // extract random number generator seed and offset
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      rng_gen, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox_args = init_philox_state(gen, rng_elts_per_thread);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  unpack<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      philox_args, static_cast<int64_t *>(rng_state.data_ptr()));
  auto te_rng_state = makeTransformerEngineTensor(rng_state);

  // create auxiliary output tensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_fwd_qkvpacked(te_QKV.data(), te_Bias.data(), te_S.data(), te_O.data(),
                                &nvte_aux_tensor_pack, te_cu_seqlens.data(),
                                te_cu_seqlens_padded.data(), te_rng_state.data(), max_seqlen,
                                is_training, attn_scale, p_dropout, qkv_layout, bias_type,
                                attn_mask_type, workspace.data(), at::cuda::getCurrentCUDAStream());

  // allocate memory for workspace and auxiliary output tensors
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // output_tensors = [O, nvte_aux_tensor_pack.tensors]
  std::vector<at::Tensor> output_tensors;
  output_tensors.push_back(O);
  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
    auto tensor = reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[i]);
    // allocate memory for nvte_aux_tensor_pack.tensors
    at::Tensor output_tensor;
    if (nvte_aux_tensor_pack.size >= 2) {
      if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) && (Bias.has_value())) {
        if (i < nvte_aux_tensor_pack.size - 2) {
          output_tensor = allocateSpace(tensor->data.shape, tensor->data.dtype, false);
        } else if (i == nvte_aux_tensor_pack.size - 2) {
          output_tensor = rng_state;
        } else if (i == nvte_aux_tensor_pack.size - 1) {
          output_tensor = Bias.value();
        }
      } else {
        output_tensor = (i < nvte_aux_tensor_pack.size - 1)
                            ? allocateSpace(tensor->data.shape, tensor->data.dtype, false)
                            : rng_state;
      }
    } else {
      output_tensor = allocateSpace(tensor->data.shape, tensor->data.dtype, false);
    }
    output_tensors.push_back(output_tensor);
    tensor->data.dptr = output_tensor.data_ptr();
  }

  // execute the kernel
  nvte_fused_attn_fwd_qkvpacked(te_QKV.data(), te_Bias.data(), te_S.data(), te_O.data(),
                                &nvte_aux_tensor_pack, te_cu_seqlens.data(),
                                te_cu_seqlens_padded.data(), te_rng_state.data(), max_seqlen,
                                is_training, attn_scale, p_dropout, qkv_layout, bias_type,
                                attn_mask_type, workspace.data(), at::cuda::getCurrentCUDAStream());

  // destroy tensor wrappers, but not allocated memory
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  // if training, [O, softmax-related tensors, rng_state]; if inference, [O]
  return output_tensors;
}

// fused attention BWD with packed QKV
std::vector<at::Tensor> fused_attn_bwd_qkvpacked(
    size_t max_seqlen, float attn_scale, float p_dropout, bool set_zero, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, const at::Tensor cu_seqlens,
    const at::Tensor QKV, const at::Tensor O, const at::Tensor dO,
    const transformer_engine::DType qkv_type, const transformer_engine::DType dqkv_type,
    const std::vector<at::Tensor> Aux_CTX_Tensors,
    const c10::optional<at::Tensor> cu_seqlens_padded, const c10::optional<at::Tensor> descale_QKV,
    const c10::optional<at::Tensor> descale_S, const c10::optional<at::Tensor> descale_O,
    const c10::optional<at::Tensor> descale_dO, const c10::optional<at::Tensor> descale_dP,
    const c10::optional<at::Tensor> scale_S, const c10::optional<at::Tensor> scale_dP,
    const c10::optional<at::Tensor> scale_dQKV, c10::optional<at::Tensor> amax_dP,
    c10::optional<at::Tensor> amax_dQKV) {
  using namespace transformer_engine;

  auto qkv_sizes = QKV.sizes().vec();
  std::vector<size_t> qkv_shape{qkv_sizes.begin(), qkv_sizes.end()};
  std::vector<size_t> q_shape;
  for (auto i : qkv_shape) {
    if (i != 3) {
      q_shape.push_back(i);
    }
  }
  auto h = q_shape[q_shape.size() - 2];

  // create output tensor dQKV
  auto options = torch::TensorOptions().dtype(GetATenDType(dqkv_type)).device(torch::kCUDA);
  at::Tensor dQKV = torch::empty_like(QKV, options);

  // construct NVTE tensors
  TensorWrapper te_QKV, te_O, te_dO, te_S, te_dP, te_dQKV;
  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    // FP8
    auto d = q_shape[q_shape.size() - 1];
    if (set_zero && ((h * d) % block_size == 0) &&
        (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD)) {
      mha_fill(dQKV, cu_seqlens.index({torch::indexing::Slice(-1, torch::indexing::None)}));
    } else {
      dQKV.fill_(0);
    }
    if ((!descale_QKV.has_value()) || (!descale_S.has_value()) || (!descale_O.has_value()) ||
        (!descale_dO.has_value()) || (!descale_dP.has_value()) || (!scale_S.has_value()) ||
        (!scale_dP.has_value()) || (!scale_dQKV.has_value()) || (!amax_dP.has_value()) ||
        (!amax_dQKV.has_value())) {
      std::string err_tensors = "descale_QKV, descale_S, descale_O, descale_dO, descale_dP, ";
      err_tensors = err_tensors + std::string("scale_S, scale_dP, scale_dQKV, ");
      err_tensors = err_tensors + std::string("amax_dP and amax_dQKV ");
      NVTE_ERROR(err_tensors + std::string("are required for FP8 operation. \n"));
    }
    te_QKV = makeTransformerEngineTensor(QKV.data_ptr(), qkv_shape, qkv_type, nullptr, nullptr,
                                         descale_QKV.value().data_ptr());
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, nullptr, nullptr,
                                       descale_O.value().data_ptr());
    te_dO = makeTransformerEngineTensor(dO.data_ptr(), q_shape, dqkv_type, nullptr, nullptr,
                                        descale_dO.value().data_ptr());
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr,
                                       scale_S.value().data_ptr(), descale_S.value().data_ptr());
    te_dP = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, amax_dP.value().data_ptr(),
                                        scale_dP.value().data_ptr(), descale_dP.value().data_ptr());
    te_dQKV = makeTransformerEngineTensor(dQKV.data_ptr(), qkv_shape, dqkv_type,
                                          amax_dQKV.value().data_ptr(),
                                          scale_dQKV.value().data_ptr(), nullptr);
  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    // BF16 or FP16
    te_QKV =
        makeTransformerEngineTensor(QKV.data_ptr(), qkv_shape, qkv_type, nullptr, nullptr, nullptr);
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, nullptr, nullptr, nullptr);
    te_dO =
        makeTransformerEngineTensor(dO.data_ptr(), q_shape, dqkv_type, nullptr, nullptr, nullptr);
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr, nullptr, nullptr);
    te_dP = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr, nullptr, nullptr);
    te_dQKV = makeTransformerEngineTensor(dQKV.data_ptr(), qkv_shape, dqkv_type, nullptr, nullptr,
                                          nullptr);
  } else {
    NVTE_ERROR("Fused attention only supports FP8 and BF16/FP16 data types. \n");
  }

  // convert auxiliary tensors from forward into NVTETensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);
  nvte_aux_tensor_pack.size = Aux_CTX_Tensors.size();
  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
    auto tensor = reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[i]);
    tensor->data.dptr = Aux_CTX_Tensors[i].data_ptr();
    std::vector<int64_t> tmp(Aux_CTX_Tensors[i].sizes().vec());
    tensor->data.shape = std::vector<size_t>(tmp.begin(), tmp.end());
    tensor->data.dtype = GetTransformerEngineDType(Aux_CTX_Tensors[i].scalar_type());
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
      dBias = torch::empty({1, static_cast<int64_t>(h), static_cast<int64_t>(max_seqlen),
                            static_cast<int64_t>(max_seqlen)},
                           options);
      te_dBias = makeTransformerEngineTensor(dBias);
    }
  }

  // create cu_seqlens tensorwrappers
  auto cu_seqlens_sizes = cu_seqlens.sizes().vec();
  std::vector<size_t> cu_seqlens_shape{cu_seqlens_sizes.begin(), cu_seqlens_sizes.end()};
  TensorWrapper te_cu_seqlens = makeTransformerEngineTensor(
      cu_seqlens.data_ptr(), cu_seqlens_shape, DType::kInt32, nullptr, nullptr, nullptr);

  TensorWrapper te_cu_seqlens_padded;
  if (cu_seqlens_padded.has_value()) {
    auto cu_seqlens_padded_sizes = cu_seqlens_padded.value().sizes().vec();
    std::vector<size_t> cu_seqlens_padded_shape{cu_seqlens_padded_sizes.begin(),
                                                cu_seqlens_padded_sizes.end()};
    te_cu_seqlens_padded =
        makeTransformerEngineTensor(cu_seqlens_padded.value().data_ptr(), cu_seqlens_padded_shape,
                                    DType::kInt32, nullptr, nullptr, nullptr);
  }

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_bwd_qkvpacked(te_QKV.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
                                &nvte_aux_tensor_pack, te_dQKV.data(), te_dBias.data(),
                                te_cu_seqlens.data(), te_cu_seqlens_padded.data(), max_seqlen,
                                attn_scale, p_dropout, qkv_layout, bias_type, attn_mask_type,
                                workspace.data(), at::cuda::getCurrentCUDAStream());

  // allocate memory for workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // execute kernel
  nvte_fused_attn_bwd_qkvpacked(te_QKV.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
                                &nvte_aux_tensor_pack, te_dQKV.data(), te_dBias.data(),
                                te_cu_seqlens.data(), te_cu_seqlens_padded.data(), max_seqlen,
                                attn_scale, p_dropout, qkv_layout, bias_type, attn_mask_type,
                                workspace.data(), at::cuda::getCurrentCUDAStream());

  // destroy tensor wrappers
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  return {dQKV, dBias};
}

// fused attention FWD with packed KV
std::vector<at::Tensor> fused_attn_fwd_kvpacked(
    size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training, float attn_scale, float p_dropout,
    bool set_zero, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv,
    const at::Tensor Q, const at::Tensor KV, const transformer_engine::DType qkv_type,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded,
    const c10::optional<at::Tensor> descale_QKV, const c10::optional<at::Tensor> descale_S,
    const c10::optional<at::Tensor> scale_S, const c10::optional<at::Tensor> scale_O,
    c10::optional<at::Tensor> amax_S, c10::optional<at::Tensor> amax_O,
    const c10::optional<at::Tensor> Bias, const c10::optional<at::Generator> rng_gen,
    size_t rng_elts_per_thread) {
  using namespace transformer_engine;

  auto q_sizes = Q.sizes().vec();
  std::vector<size_t> q_shape{q_sizes.begin(), q_sizes.end()};
  auto kv_sizes = KV.sizes().vec();
  std::vector<size_t> kv_shape{kv_sizes.begin(), kv_sizes.end()};
  std::vector<int64_t> o_shape{q_shape.begin(), q_shape.end()};

  // create output tensor O
  auto options = torch::TensorOptions().dtype(GetATenDType(qkv_type)).device(torch::kCUDA);
  auto O = torch::empty(o_shape, options);

  // construct NVTE tensors
  TensorWrapper te_Q, te_KV, te_S, te_O, te_Bias, te_cu_seqlens_q, te_cu_seqlens_kv;
  TensorWrapper te_cu_seqlens_q_padded, te_cu_seqlens_kv_padded;
  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    // FP8
    auto h = q_shape[q_shape.size() - 2];
    auto d = q_shape[q_shape.size() - 1];
    if (set_zero && ((h * d) % block_size == 0) &&
        (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD)) {
      mha_fill(O, cu_seqlens_q.index({torch::indexing::Slice(-1, torch::indexing::None)}));
    } else {
      O.fill_(0);
    }
    if ((!descale_QKV.has_value()) || (!descale_S.has_value()) || (!scale_S.has_value()) ||
        (!scale_O.has_value()) || (!amax_S.has_value()) || (!amax_O.has_value())) {
      std::string err_tensors = "descale_QKV, descale_S, scale_S, scale_O, amax_S and amax_O ";
      NVTE_ERROR(err_tensors + std::string("are required for FP8 operation. \n"));
    }
    te_Q = makeTransformerEngineTensor(Q.data_ptr(), q_shape, qkv_type, nullptr, nullptr,
                                       descale_QKV.value().data_ptr());
    te_KV = makeTransformerEngineTensor(KV.data_ptr(), kv_shape, qkv_type, nullptr, nullptr,
                                        descale_QKV.value().data_ptr());
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, amax_S.value().data_ptr(),
                                       scale_S.value().data_ptr(), descale_S.value().data_ptr());
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, amax_O.value().data_ptr(),
                                       scale_O.value().data_ptr(), nullptr);
  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    // BF16 or FP16
    te_Q = makeTransformerEngineTensor(Q.data_ptr(), q_shape, qkv_type, nullptr, nullptr, nullptr);
    te_KV =
        makeTransformerEngineTensor(KV.data_ptr(), kv_shape, qkv_type, nullptr, nullptr, nullptr);
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr, nullptr, nullptr);
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, nullptr, nullptr, nullptr);
  } else {
    NVTE_ERROR("Fused attention only supports FP8 and BF16/FP16 data types. \n");
  }
  if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) && (Bias.has_value())) {
    auto bias_sizes = Bias.value().sizes().vec();
    std::vector<size_t> bias_shape{bias_sizes.begin(), bias_sizes.end()};
    te_Bias = makeTransformerEngineTensor(Bias.value().data_ptr(), bias_shape, DType::kFloat32,
                                          nullptr, nullptr, nullptr);
  }
  auto cu_seqlens_q_sizes = cu_seqlens_q.sizes().vec();
  std::vector<size_t> cu_seqlens_q_shape{cu_seqlens_q_sizes.begin(), cu_seqlens_q_sizes.end()};
  auto cu_seqlens_kv_sizes = cu_seqlens_kv.sizes().vec();
  std::vector<size_t> cu_seqlens_kv_shape{cu_seqlens_kv_sizes.begin(), cu_seqlens_kv_sizes.end()};
  te_cu_seqlens_q = makeTransformerEngineTensor(cu_seqlens_q.data_ptr(), cu_seqlens_q_shape,
                                                DType::kInt32, nullptr, nullptr, nullptr);
  te_cu_seqlens_kv = makeTransformerEngineTensor(cu_seqlens_kv.data_ptr(), cu_seqlens_kv_shape,
                                                 DType::kInt32, nullptr, nullptr, nullptr);

  if ((cu_seqlens_q_padded.has_value()) && (cu_seqlens_kv_padded.has_value())) {
    auto cu_seqlens_q_padded_sizes = cu_seqlens_q_padded.value().sizes().vec();
    std::vector<size_t> cu_seqlens_q_padded_shape{cu_seqlens_q_padded_sizes.begin(),
                                                  cu_seqlens_q_padded_sizes.end()};
    auto cu_seqlens_kv_padded_sizes = cu_seqlens_kv_padded.value().sizes().vec();
    std::vector<size_t> cu_seqlens_kv_padded_shape{cu_seqlens_kv_padded_sizes.begin(),
                                                   cu_seqlens_kv_padded_sizes.end()};
    te_cu_seqlens_q_padded = makeTransformerEngineTensor(cu_seqlens_q_padded.value().data_ptr(),
                                                         cu_seqlens_q_padded_shape, DType::kInt32,
                                                         nullptr, nullptr, nullptr);
    te_cu_seqlens_kv_padded = makeTransformerEngineTensor(cu_seqlens_kv_padded.value().data_ptr(),
                                                          cu_seqlens_kv_padded_shape, DType::kInt32,
                                                          nullptr, nullptr, nullptr);
  }

  // extract rng seed and offset
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      rng_gen, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox_args = init_philox_state(gen, rng_elts_per_thread);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  unpack<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      philox_args, static_cast<int64_t *>(rng_state.data_ptr()));
  auto te_rng_state = makeTransformerEngineTensor(rng_state);

  // create auxiliary output tensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_fwd_kvpacked(
      te_Q.data(), te_KV.data(), te_Bias.data(), te_S.data(), te_O.data(), &nvte_aux_tensor_pack,
      te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(), te_cu_seqlens_q_padded.data(),
      te_cu_seqlens_kv_padded.data(), te_rng_state.data(), max_seqlen_q, max_seqlen_kv, is_training,
      attn_scale, p_dropout, qkv_layout, bias_type, attn_mask_type, workspace.data(),
      at::cuda::getCurrentCUDAStream());

  // allocate memory for workspace and auxiliary output tensors
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // output_tensors = [O, nvte_aux_tensor_pack.tensors]
  std::vector<at::Tensor> output_tensors;
  output_tensors.push_back(O);
  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
    auto tensor = reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[i]);
    // allocate memory for nvte_aux_tensor_pack.tensors
    at::Tensor output_tensor;
    if (nvte_aux_tensor_pack.size >= 2) {
      if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) && (Bias.has_value())) {
        if (i < nvte_aux_tensor_pack.size - 2) {
          output_tensor = allocateSpace(tensor->data.shape, tensor->data.dtype, false);
        } else if (i == nvte_aux_tensor_pack.size - 2) {
          output_tensor = rng_state;
        } else if (i == nvte_aux_tensor_pack.size - 1) {
          output_tensor = Bias.value();
        }
      } else {
        output_tensor = (i < nvte_aux_tensor_pack.size - 1)
                            ? allocateSpace(tensor->data.shape, tensor->data.dtype, false)
                            : rng_state;
      }
    } else {
      output_tensor = allocateSpace(tensor->data.shape, tensor->data.dtype, false);
    }
    output_tensors.push_back(output_tensor);
    tensor->data.dptr = output_tensor.data_ptr();
  }

  // execute the kernel
  nvte_fused_attn_fwd_kvpacked(
      te_Q.data(), te_KV.data(), te_Bias.data(), te_S.data(), te_O.data(), &nvte_aux_tensor_pack,
      te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(), te_cu_seqlens_q_padded.data(),
      te_cu_seqlens_kv_padded.data(), te_rng_state.data(), max_seqlen_q, max_seqlen_kv, is_training,
      attn_scale, p_dropout, qkv_layout, bias_type, attn_mask_type, workspace.data(),
      at::cuda::getCurrentCUDAStream());

  // destroy tensor wrappers, but not allocated memory
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  // if training, [O, softmax-related tensors, rng_state]; if inference, [O]
  return output_tensors;
}

// fused attention BWD with packed KV
std::vector<at::Tensor> fused_attn_bwd_kvpacked(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv, const at::Tensor Q,
    const at::Tensor KV, const at::Tensor O, const at::Tensor dO,
    const transformer_engine::DType qkv_type, const transformer_engine::DType dqkv_type,
    const std::vector<at::Tensor> Aux_CTX_Tensors,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded,
    const c10::optional<at::Tensor> descale_QKV, const c10::optional<at::Tensor> descale_S,
    const c10::optional<at::Tensor> descale_O, const c10::optional<at::Tensor> descale_dO,
    const c10::optional<at::Tensor> descale_dP, const c10::optional<at::Tensor> scale_S,
    const c10::optional<at::Tensor> scale_dP, const c10::optional<at::Tensor> scale_dQKV,
    c10::optional<at::Tensor> amax_dP, c10::optional<at::Tensor> amax_dQKV) {
  using namespace transformer_engine;

  auto q_sizes = Q.sizes().vec();
  std::vector<size_t> q_shape{q_sizes.begin(), q_sizes.end()};
  auto kv_sizes = KV.sizes().vec();
  std::vector<size_t> kv_shape{kv_sizes.begin(), kv_sizes.end()};
  std::vector<size_t> k_shape;
  for (auto i : kv_shape) {
    if (i != 2) {
      k_shape.push_back(i);
    }
  }
  auto h_q = q_shape[q_shape.size() - 2];
  auto h_kv = k_shape[k_shape.size() - 2];
  auto d = q_shape[q_shape.size() - 1];

  // create output tensors dQ and dKV
  auto options = torch::TensorOptions().dtype(GetATenDType(dqkv_type)).device(torch::kCUDA);
  at::Tensor dQ = torch::empty_like(Q, options);
  at::Tensor dKV = torch::empty_like(KV, options);

  // construct NVTE tensors
  TensorWrapper te_Q, te_KV, te_O, te_dO, te_S, te_dP, te_dQ, te_dKV;
  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    // FP8
    if (set_zero && ((h_q * d) % block_size == 0) && ((h_kv * d) % block_size == 0) &&
        (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD)) {
      mha_fill(dQ, cu_seqlens_q.index({torch::indexing::Slice(-1, torch::indexing::None)}));
      mha_fill(dKV, cu_seqlens_kv.index({torch::indexing::Slice(-1, torch::indexing::None)}));
    } else {
      dQ.fill_(0);
      dKV.fill_(0);
    }
    if ((!descale_QKV.has_value()) || (!descale_S.has_value()) || (!descale_O.has_value()) ||
        (!descale_dO.has_value()) || (!descale_dP.has_value()) || (!scale_S.has_value()) ||
        (!scale_dP.has_value()) || (!scale_dQKV.has_value()) || (!amax_dP.has_value()) ||
        (!amax_dQKV.has_value())) {
      std::string err_tensors = "descale_QKV, descale_S, descale_O, descale_dO, descale_dP, ";
      err_tensors = err_tensors + std::string("scale_S, scale_dP, scale_dQKV, ");
      err_tensors = err_tensors + std::string("amax_dP and amax_dQKV ");
      NVTE_ERROR(err_tensors + std::string("are required for FP8 operation. \n"));
    }
    te_Q = makeTransformerEngineTensor(Q.data_ptr(), q_shape, qkv_type, nullptr, nullptr,
                                       descale_QKV.value().data_ptr());
    te_KV = makeTransformerEngineTensor(KV.data_ptr(), kv_shape, qkv_type, nullptr, nullptr,
                                        descale_QKV.value().data_ptr());
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, nullptr, nullptr,
                                       descale_O.value().data_ptr());
    te_dO = makeTransformerEngineTensor(dO.data_ptr(), q_shape, dqkv_type, nullptr, nullptr,
                                        descale_dO.value().data_ptr());
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr,
                                       scale_S.value().data_ptr(), descale_S.value().data_ptr());
    te_dP = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, amax_dP.value().data_ptr(),
                                        scale_dP.value().data_ptr(), descale_dP.value().data_ptr());
    te_dQ =
        makeTransformerEngineTensor(dQ.data_ptr(), q_shape, dqkv_type, amax_dQKV.value().data_ptr(),
                                    scale_dQKV.value().data_ptr(), nullptr);
    te_dKV = makeTransformerEngineTensor(dKV.data_ptr(), kv_shape, dqkv_type,
                                         amax_dQKV.value().data_ptr(),
                                         scale_dQKV.value().data_ptr(), nullptr);
  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    // BF16 or FP16
    te_Q = makeTransformerEngineTensor(Q.data_ptr(), q_shape, qkv_type, nullptr, nullptr, nullptr);
    te_KV =
        makeTransformerEngineTensor(KV.data_ptr(), kv_shape, qkv_type, nullptr, nullptr, nullptr);
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, nullptr, nullptr, nullptr);
    te_dO =
        makeTransformerEngineTensor(dO.data_ptr(), q_shape, dqkv_type, nullptr, nullptr, nullptr);
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr, nullptr, nullptr);
    te_dP = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr, nullptr, nullptr);
    te_dQ =
        makeTransformerEngineTensor(dQ.data_ptr(), q_shape, dqkv_type, nullptr, nullptr, nullptr);
    te_dKV =
        makeTransformerEngineTensor(dKV.data_ptr(), kv_shape, dqkv_type, nullptr, nullptr, nullptr);
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
                                                         cu_seqlens_q_padded_shape, DType::kInt32,
                                                         nullptr, nullptr, nullptr);
    te_cu_seqlens_kv_padded = makeTransformerEngineTensor(cu_seqlens_kv_padded.value().data_ptr(),
                                                          cu_seqlens_kv_padded_shape, DType::kInt32,
                                                          nullptr, nullptr, nullptr);
  }

  // convert auxiliary tensors from forward to NVTETensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);
  nvte_aux_tensor_pack.size = Aux_CTX_Tensors.size();
  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
    auto tensor = reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[i]);
    tensor->data.dptr = Aux_CTX_Tensors[i].data_ptr();
    std::vector<int64_t> tmp(Aux_CTX_Tensors[i].sizes().vec());
    tensor->data.shape = std::vector<size_t>(tmp.begin(), tmp.end());
    tensor->data.dtype = GetTransformerEngineDType(Aux_CTX_Tensors[i].scalar_type());
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
  }

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_bwd_kvpacked(
      te_Q.data(), te_KV.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
      &nvte_aux_tensor_pack, te_dQ.data(), te_dKV.data(), te_dBias.data(), te_cu_seqlens_q.data(),
      te_cu_seqlens_kv.data(), te_cu_seqlens_q_padded.data(), te_cu_seqlens_kv_padded.data(),
      max_seqlen_q, max_seqlen_kv, attn_scale, p_dropout, qkv_layout, bias_type, attn_mask_type,
      workspace.data(), at::cuda::getCurrentCUDAStream());

  // allocate memory for workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // execute kernel
  nvte_fused_attn_bwd_kvpacked(
      te_Q.data(), te_KV.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
      &nvte_aux_tensor_pack, te_dQ.data(), te_dKV.data(), te_dBias.data(), te_cu_seqlens_q.data(),
      te_cu_seqlens_kv.data(), te_cu_seqlens_q_padded.data(), te_cu_seqlens_kv_padded.data(),
      max_seqlen_q, max_seqlen_kv, attn_scale, p_dropout, qkv_layout, bias_type, attn_mask_type,
      workspace.data(), at::cuda::getCurrentCUDAStream());

  // destroy tensor wrappers
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  return {dQ, dKV, dBias};
}

// fused attention FWD with separate Q, K and V tensors
std::vector<at::Tensor> fused_attn_fwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training, float attn_scale, float p_dropout,
    bool set_zero, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv,
    const at::Tensor Q, const at::Tensor K, const at::Tensor V,
    const transformer_engine::DType qkv_type, const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded,
    const c10::optional<at::Tensor> descale_QKV, const c10::optional<at::Tensor> descale_S,
    const c10::optional<at::Tensor> scale_S, const c10::optional<at::Tensor> scale_O,
    c10::optional<at::Tensor> amax_S, c10::optional<at::Tensor> amax_O,
    const c10::optional<at::Tensor> Bias, const c10::optional<at::Generator> rng_gen,
    size_t rng_elts_per_thread) {
  using namespace transformer_engine;

  auto q_sizes = Q.sizes().vec();
  std::vector<size_t> q_shape{q_sizes.begin(), q_sizes.end()};
  auto k_sizes = K.sizes().vec();
  std::vector<size_t> k_shape{k_sizes.begin(), k_sizes.end()};
  auto v_sizes = V.sizes().vec();
  std::vector<size_t> v_shape{v_sizes.begin(), v_sizes.end()};

  // create output tensor O
  auto O = torch::empty_like(Q);

  // construct NVTE tensors
  TensorWrapper te_Q, te_K, te_V, te_S, te_O, te_Bias;
  TensorWrapper te_cu_seqlens_q, te_cu_seqlens_kv;
  TensorWrapper te_cu_seqlens_q_padded, te_cu_seqlens_kv_padded;
  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    // FP8
    auto h = q_shape[q_shape.size() - 2];
    auto d = q_shape[q_shape.size() - 1];
    if (set_zero && ((h * d) % block_size == 0) &&
        (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD)) {
      mha_fill(O, cu_seqlens_q.index({torch::indexing::Slice(-1, torch::indexing::None)}));
    } else {
      O.fill_(0);
    }
    if ((!descale_QKV.has_value()) || (!descale_S.has_value()) || (!scale_S.has_value()) ||
        (!scale_O.has_value()) || (!amax_S.has_value()) || (!amax_O.has_value())) {
      std::string err_tensors = "descale_QKV, descale_S, scale_S, scale_O, amax_S and amax_O ";
      NVTE_ERROR(err_tensors + std::string("are required for FP8 operation. \n"));
    }
    te_Q = makeTransformerEngineTensor(Q.data_ptr(), q_shape, qkv_type, nullptr, nullptr,
                                       descale_QKV.value().data_ptr());
    te_K = makeTransformerEngineTensor(K.data_ptr(), k_shape, qkv_type, nullptr, nullptr,
                                       descale_QKV.value().data_ptr());
    te_V = makeTransformerEngineTensor(V.data_ptr(), v_shape, qkv_type, nullptr, nullptr,
                                       descale_QKV.value().data_ptr());
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, amax_S.value().data_ptr(),
                                       scale_S.value().data_ptr(), descale_S.value().data_ptr());
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, amax_O.value().data_ptr(),
                                       scale_O.value().data_ptr(), nullptr);
  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    // BF16 or FP16
    te_Q = makeTransformerEngineTensor(Q.data_ptr(), q_shape, qkv_type, nullptr, nullptr, nullptr);
    te_K = makeTransformerEngineTensor(K.data_ptr(), k_shape, qkv_type, nullptr, nullptr, nullptr);
    te_V = makeTransformerEngineTensor(V.data_ptr(), v_shape, qkv_type, nullptr, nullptr, nullptr);
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr, nullptr, nullptr);
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, nullptr, nullptr, nullptr);
  } else {
    NVTE_ERROR("Fused attention only supports FP8 and BF16/FP16 data types. \n");
  }
  if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) && (Bias.has_value())) {
    auto bias_sizes = Bias.value().sizes().vec();
    std::vector<size_t> bias_shape{bias_sizes.begin(), bias_sizes.end()};
    te_Bias = makeTransformerEngineTensor(Bias.value().data_ptr(), bias_shape, DType::kFloat32,
                                          nullptr, nullptr, nullptr);
  }
  auto cu_seqlens_q_sizes = cu_seqlens_q.sizes().vec();
  std::vector<size_t> cu_seqlens_q_shape{cu_seqlens_q_sizes.begin(), cu_seqlens_q_sizes.end()};
  auto cu_seqlens_kv_sizes = cu_seqlens_kv.sizes().vec();
  std::vector<size_t> cu_seqlens_kv_shape{cu_seqlens_kv_sizes.begin(), cu_seqlens_kv_sizes.end()};
  te_cu_seqlens_q = makeTransformerEngineTensor(cu_seqlens_q.data_ptr(), cu_seqlens_q_shape,
                                                DType::kInt32, nullptr, nullptr, nullptr);
  te_cu_seqlens_kv = makeTransformerEngineTensor(cu_seqlens_kv.data_ptr(), cu_seqlens_kv_shape,
                                                 DType::kInt32, nullptr, nullptr, nullptr);

  if ((cu_seqlens_q_padded.has_value()) && (cu_seqlens_kv_padded.has_value())) {
    auto cu_seqlens_q_padded_sizes = cu_seqlens_q_padded.value().sizes().vec();
    std::vector<size_t> cu_seqlens_q_padded_shape{cu_seqlens_q_padded_sizes.begin(),
                                                  cu_seqlens_q_padded_sizes.end()};
    auto cu_seqlens_kv_padded_sizes = cu_seqlens_kv_padded.value().sizes().vec();
    std::vector<size_t> cu_seqlens_kv_padded_shape{cu_seqlens_kv_padded_sizes.begin(),
                                                   cu_seqlens_kv_padded_sizes.end()};
    te_cu_seqlens_q_padded = makeTransformerEngineTensor(cu_seqlens_q_padded.value().data_ptr(),
                                                         cu_seqlens_q_padded_shape, DType::kInt32,
                                                         nullptr, nullptr, nullptr);
    te_cu_seqlens_kv_padded = makeTransformerEngineTensor(cu_seqlens_kv_padded.value().data_ptr(),
                                                          cu_seqlens_kv_padded_shape, DType::kInt32,
                                                          nullptr, nullptr, nullptr);
  }

  // extract rng seed and offset
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      rng_gen, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox_args = init_philox_state(gen, rng_elts_per_thread);
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options);
  unpack<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      philox_args, static_cast<int64_t *>(rng_state.data_ptr()));
  auto te_rng_state = makeTransformerEngineTensor(rng_state);

  // create auxiliary output tensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_fwd(te_Q.data(), te_K.data(), te_V.data(), te_Bias.data(), te_S.data(),
                      te_O.data(), &nvte_aux_tensor_pack, te_cu_seqlens_q.data(),
                      te_cu_seqlens_kv.data(), te_cu_seqlens_q_padded.data(),
                      te_cu_seqlens_kv_padded.data(), te_rng_state.data(), max_seqlen_q,
                      max_seqlen_kv, is_training, attn_scale, p_dropout, qkv_layout, bias_type,
                      attn_mask_type, workspace.data(), at::cuda::getCurrentCUDAStream());

  // allocate memory for workspace and auxiliary output tensors
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // output_tensors = [O, nvte_aux_tensor_pack.tensors]
  std::vector<at::Tensor> output_tensors;
  output_tensors.push_back(O);
  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
    auto tensor = reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[i]);
    // allocate memory for nvte_aux_tensor_pack.tensors
    at::Tensor output_tensor;
    if (nvte_aux_tensor_pack.size >= 2) {
      if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) && (Bias.has_value())) {
        if (i < nvte_aux_tensor_pack.size - 2) {
          output_tensor = allocateSpace(tensor->data.shape, tensor->data.dtype, false);
        } else if (i == nvte_aux_tensor_pack.size - 2) {
          output_tensor = rng_state;
        } else if (i == nvte_aux_tensor_pack.size - 1) {
          output_tensor = Bias.value();
        }
      } else {
        output_tensor = (i < nvte_aux_tensor_pack.size - 1)
                            ? allocateSpace(tensor->data.shape, tensor->data.dtype, false)
                            : rng_state;
      }
    } else {
      output_tensor = allocateSpace(tensor->data.shape, tensor->data.dtype, false);
    }
    output_tensors.push_back(output_tensor);
    tensor->data.dptr = output_tensor.data_ptr();
  }

  // execute the kernel
  nvte_fused_attn_fwd(te_Q.data(), te_K.data(), te_V.data(), te_Bias.data(), te_S.data(),
                      te_O.data(), &nvte_aux_tensor_pack, te_cu_seqlens_q.data(),
                      te_cu_seqlens_kv.data(), te_cu_seqlens_q_padded.data(),
                      te_cu_seqlens_kv_padded.data(), te_rng_state.data(), max_seqlen_q,
                      max_seqlen_kv, is_training, attn_scale, p_dropout, qkv_layout, bias_type,
                      attn_mask_type, workspace.data(), at::cuda::getCurrentCUDAStream());

  // destroy tensor wrappers, but not allocated memory
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  // if training, [O, softmax-related tensors, rng_state]; if inference, [O]
  return output_tensors;
}

// fused attention BWD with separate Q, K and V
std::vector<at::Tensor> fused_attn_bwd(
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float p_dropout, bool set_zero,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    const at::Tensor cu_seqlens_q, const at::Tensor cu_seqlens_kv, const at::Tensor Q,
    const at::Tensor K, const at::Tensor V, const at::Tensor O, const at::Tensor dO,
    const transformer_engine::DType qkv_type, const transformer_engine::DType dqkv_type,
    const std::vector<at::Tensor> Aux_CTX_Tensors,
    const c10::optional<at::Tensor> cu_seqlens_q_padded,
    const c10::optional<at::Tensor> cu_seqlens_kv_padded,
    const c10::optional<at::Tensor> descale_QKV, const c10::optional<at::Tensor> descale_S,
    const c10::optional<at::Tensor> descale_O, const c10::optional<at::Tensor> descale_dO,
    const c10::optional<at::Tensor> descale_dP, const c10::optional<at::Tensor> scale_S,
    const c10::optional<at::Tensor> scale_dP, const c10::optional<at::Tensor> scale_dQKV,
    c10::optional<at::Tensor> amax_dP, c10::optional<at::Tensor> amax_dQKV) {
  using namespace transformer_engine;

  auto q_sizes = Q.sizes().vec();
  std::vector<size_t> q_shape{q_sizes.begin(), q_sizes.end()};
  auto k_sizes = K.sizes().vec();
  std::vector<size_t> k_shape{k_sizes.begin(), k_sizes.end()};
  auto v_sizes = V.sizes().vec();
  std::vector<size_t> v_shape{v_sizes.begin(), v_sizes.end()};
  auto h_q = q_shape[q_shape.size() - 2];
  auto h_kv = k_shape[k_shape.size() - 2];
  auto d = q_shape[q_shape.size() - 1];
  auto options = torch::TensorOptions().dtype(GetATenDType(dqkv_type)).device(torch::kCUDA);

  at::Tensor dQ;
  at::Tensor dK;
  at::Tensor dV;
  at::Tensor dQKV, dKV;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  std::vector<int64_t> tmp_shape;
  switch (layout_group) {
    case NVTE_QKV_Layout_Group::NVTE_3HD:
      tmp_shape = std::vector<int64_t>{q_sizes.begin(), q_sizes.end()};
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
      tmp_shape = std::vector<int64_t>{q_sizes.begin(), q_sizes.end()};
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
      dQ = torch::empty_like(Q, options);
      tmp_shape = std::vector<int64_t>{k_sizes.begin(), k_sizes.end()};
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
      dQ = torch::empty_like(Q, options);
      tmp_shape = std::vector<int64_t>{k_sizes.begin(), k_sizes.end()};
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
      dQ = torch::empty_like(Q, options);
      dK = torch::empty_like(K, options);
      dV = torch::empty_like(V, options);
      break;
    default:
      NVTE_ERROR("QKV layout not supported!");
  }

  // construct NVTE tensors
  TensorWrapper te_Q, te_K, te_V, te_O, te_dO, te_S, te_dP, te_dQ, te_dK, te_dV;
  if (qkv_type == DType::kFloat8E4M3 || qkv_type == DType::kFloat8E5M2) {
    // FP8
    if (set_zero && ((h_q * d) % block_size == 0) && ((h_kv * d) % block_size == 0) &&
        dQ.is_contiguous() && dK.is_contiguous() && dV.is_contiguous() &&
        (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD)) {
      mha_fill(dQ, cu_seqlens_q.index({torch::indexing::Slice(-1, torch::indexing::None)}));
      mha_fill(dK, cu_seqlens_kv.index({torch::indexing::Slice(-1, torch::indexing::None)}));
      mha_fill(dV, cu_seqlens_kv.index({torch::indexing::Slice(-1, torch::indexing::None)}));
    } else {
      dQ.fill_(0);
      dK.fill_(0);
      dV.fill_(0);
    }
    if ((!descale_QKV.has_value()) || (!descale_S.has_value()) || (!descale_O.has_value()) ||
        (!descale_dO.has_value()) || (!descale_dP.has_value()) || (!scale_S.has_value()) ||
        (!scale_dP.has_value()) || (!scale_dQKV.has_value()) || (!amax_dP.has_value()) ||
        (!amax_dQKV.has_value())) {
      std::string err_tensors = "descale_QKV, descale_S, descale_O, descale_dO, descale_dP, ";
      err_tensors = err_tensors + std::string("scale_S, scale_dP, scale_dQKV, ");
      err_tensors = err_tensors + std::string("amax_dP and amax_dQKV ");
      NVTE_ERROR(err_tensors + std::string("are required for FP8 operation. \n"));
    }
    te_Q = makeTransformerEngineTensor(Q.data_ptr(), q_shape, qkv_type, nullptr, nullptr,
                                       descale_QKV.value().data_ptr());
    te_K = makeTransformerEngineTensor(K.data_ptr(), k_shape, qkv_type, nullptr, nullptr,
                                       descale_QKV.value().data_ptr());
    te_V = makeTransformerEngineTensor(V.data_ptr(), v_shape, qkv_type, nullptr, nullptr,
                                       descale_QKV.value().data_ptr());
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, nullptr, nullptr,
                                       descale_O.value().data_ptr());
    te_dO = makeTransformerEngineTensor(dO.data_ptr(), q_shape, dqkv_type, nullptr, nullptr,
                                        descale_dO.value().data_ptr());
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr,
                                       scale_S.value().data_ptr(), descale_S.value().data_ptr());
    te_dP = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, amax_dP.value().data_ptr(),
                                        scale_dP.value().data_ptr(), descale_dP.value().data_ptr());
    te_dQ =
        makeTransformerEngineTensor(dQ.data_ptr(), q_shape, dqkv_type, amax_dQKV.value().data_ptr(),
                                    scale_dQKV.value().data_ptr(), nullptr);
    te_dK =
        makeTransformerEngineTensor(dK.data_ptr(), k_shape, dqkv_type, amax_dQKV.value().data_ptr(),
                                    scale_dQKV.value().data_ptr(), nullptr);
    te_dV =
        makeTransformerEngineTensor(dV.data_ptr(), v_shape, dqkv_type, amax_dQKV.value().data_ptr(),
                                    scale_dQKV.value().data_ptr(), nullptr);
  } else if (qkv_type == DType::kBFloat16 || qkv_type == DType::kFloat16) {
    // BF16 or FP16
    te_Q = makeTransformerEngineTensor(Q.data_ptr(), q_shape, qkv_type, nullptr, nullptr, nullptr);
    te_K = makeTransformerEngineTensor(K.data_ptr(), k_shape, qkv_type, nullptr, nullptr, nullptr);
    te_V = makeTransformerEngineTensor(V.data_ptr(), v_shape, qkv_type, nullptr, nullptr, nullptr);
    te_O = makeTransformerEngineTensor(O.data_ptr(), q_shape, qkv_type, nullptr, nullptr, nullptr);
    te_dO =
        makeTransformerEngineTensor(dO.data_ptr(), q_shape, dqkv_type, nullptr, nullptr, nullptr);
    te_S = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr, nullptr, nullptr);
    te_dP = makeTransformerEngineTensor(nullptr, {0}, DType::kFloat32, nullptr, nullptr, nullptr);
    te_dQ =
        makeTransformerEngineTensor(dQ.data_ptr(), q_shape, dqkv_type, nullptr, nullptr, nullptr);
    te_dK =
        makeTransformerEngineTensor(dK.data_ptr(), k_shape, dqkv_type, nullptr, nullptr, nullptr);
    te_dV =
        makeTransformerEngineTensor(dV.data_ptr(), v_shape, dqkv_type, nullptr, nullptr, nullptr);
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
                                                         cu_seqlens_q_padded_shape, DType::kInt32,
                                                         nullptr, nullptr, nullptr);
    te_cu_seqlens_kv_padded = makeTransformerEngineTensor(cu_seqlens_kv_padded.value().data_ptr(),
                                                          cu_seqlens_kv_padded_shape, DType::kInt32,
                                                          nullptr, nullptr, nullptr);
  }

  // convert auxiliary tensors from forward to NVTETensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);
  nvte_aux_tensor_pack.size = Aux_CTX_Tensors.size();
  for (size_t i = 0; i < nvte_aux_tensor_pack.size; ++i) {
    auto tensor = reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[i]);
    tensor->data.dptr = Aux_CTX_Tensors[i].data_ptr();
    std::vector<int64_t> tmp(Aux_CTX_Tensors[i].sizes().vec());
    tensor->data.shape = std::vector<size_t>(tmp.begin(), tmp.end());
    tensor->data.dtype = GetTransformerEngineDType(Aux_CTX_Tensors[i].scalar_type());
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
  }

  // create workspace
  TensorWrapper workspace;

  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_bwd(te_Q.data(), te_K.data(), te_V.data(), te_O.data(), te_dO.data(), te_S.data(),
                      te_dP.data(), &nvte_aux_tensor_pack, te_dQ.data(), te_dK.data(), te_dV.data(),
                      te_dBias.data(), te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
                      te_cu_seqlens_q_padded.data(), te_cu_seqlens_kv_padded.data(), max_seqlen_q,
                      max_seqlen_kv, attn_scale, p_dropout, qkv_layout, bias_type, attn_mask_type,
                      workspace.data(), at::cuda::getCurrentCUDAStream());

  // allocate memory for workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // execute kernel
  nvte_fused_attn_bwd(te_Q.data(), te_K.data(), te_V.data(), te_O.data(), te_dO.data(), te_S.data(),
                      te_dP.data(), &nvte_aux_tensor_pack, te_dQ.data(), te_dK.data(), te_dV.data(),
                      te_dBias.data(), te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
                      te_cu_seqlens_q_padded.data(), te_cu_seqlens_kv_padded.data(), max_seqlen_q,
                      max_seqlen_kv, attn_scale, p_dropout, qkv_layout, bias_type, attn_mask_type,
                      workspace.data(), at::cuda::getCurrentCUDAStream());

  // destroy tensor wrappers
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);

  return {dQ, dK, dV, dBias};
}

namespace flash_attention {

constexpr int warp_size = 32;
constexpr int type_size = 2;  // FP16 or BF16
constexpr int nvec = sizeof(uint64_t) / type_size;
constexpr int load_size = warp_size * nvec;
constexpr int block_size = 512;

template <typename T>
__launch_bounds__(block_size) __global__
    void prepare_kernel_fwd(const T *qkvi, T *qkv, const size_t B, const size_t S, const size_t Z,
                            const size_t W) {
  const int warpid = (blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
  const int id_in_warp = threadIdx.x % warp_size;
  const size_t offset_input = blockIdx.y * W + warpid * 3 * W * Z + id_in_warp * nvec;
  const T *my_input = qkvi + offset_input;

  const size_t s = warpid / B;
  if (s >= S) return;

  const size_t b = warpid % B;

  const size_t offset_output = blockIdx.y * B * S * Z * W + (s + b * S) * W * Z + id_in_warp * nvec;

  T *my_output = qkv + offset_output;

  for (int i = 0; i < Z; ++i) {
    uint64_t *out = reinterpret_cast<uint64_t *>(my_output + i * load_size);
    *out = *reinterpret_cast<const uint64_t *>(my_input + i * load_size * 3);
  }
}

template <typename T>
__launch_bounds__(block_size) __global__
    void prepare_kernel_bwd(const T *q, const T *k, const T *v, T *qkv, const size_t B,
                            const size_t S, const size_t Z, const size_t W) {
  const T *input = blockIdx.y == 0 ? q : (blockIdx.y == 1 ? k : v);

  const int warpid = (blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
  const int id_in_warp = threadIdx.x % warp_size;
  const size_t offset_input = warpid * W * Z + id_in_warp * nvec;
  const T *my_input = input + offset_input;

  const size_t b = warpid / S;
  if (b >= B) return;

  const size_t s = warpid % S;

  const size_t offset_output = (b + s * B) * 3 * W * Z + id_in_warp * nvec + blockIdx.y * W;

  T *my_output = qkv + offset_output;

  for (int i = 0; i < Z; ++i) {
    uint64_t *out = reinterpret_cast<uint64_t *>(my_output + i * load_size * 3);
    *out = *reinterpret_cast<const uint64_t *>(my_input + i * load_size);
  }
}

}  // namespace flash_attention

at::Tensor fa_prepare_fwd(at::Tensor qkvi) {
  NVTE_CHECK(qkvi.dim() == 4, "Expected 4-dim tensor.");
  NVTE_CHECK(qkvi.scalar_type() == at::ScalarType::Half ||
             qkvi.scalar_type() == at::ScalarType::BFloat16);
  NVTE_CHECK(qkvi.size(3) % flash_attention::load_size == 0);
  NVTE_CHECK(qkvi.size(3) == flash_attention::load_size);
  NVTE_CHECK(qkvi.stride(3) == 1, "Wrong stride.");
  NVTE_CHECK(qkvi.stride(2) == 3 * qkvi.size(3), "Wrong stride.");
  NVTE_CHECK(qkvi.stride(1) == 3 * qkvi.size(3) * qkvi.size(2), "Wrong stride.");
  NVTE_CHECK(qkvi.stride(0) == 3 * qkvi.size(3) * qkvi.size(2) * qkvi.size(1), "Wrong stride.");

  // [s, b, n, h * 3] -> [3, b, s, n, h]
  std::vector<int64_t> shape = {3, qkvi.size(1), qkvi.size(0), qkvi.size(2), qkvi.size(3)};
  at::Tensor qkv = at::empty(shape, at::CUDA(qkvi.scalar_type()));

  size_t warps = qkvi.size(0) * qkvi.size(1);
  size_t warps_per_block = flash_attention::block_size / flash_attention::warp_size;
  size_t blocks = (warps + warps_per_block - 1) / warps_per_block;
  dim3 grid(blocks, 3);
  int threads = flash_attention::block_size;
  if (qkvi.scalar_type() == at::ScalarType::Half) {
    using dtype = at::Half;
    flash_attention::prepare_kernel_fwd<dtype>
        <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            qkvi.data_ptr<dtype>(), qkv.data_ptr<dtype>(), shape[1], shape[2], shape[3], shape[4]);
  } else {
    using dtype = at::BFloat16;
    flash_attention::prepare_kernel_fwd<dtype>
        <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            qkvi.data_ptr<dtype>(), qkv.data_ptr<dtype>(), shape[1], shape[2], shape[3], shape[4]);
  }

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
  NVTE_CHECK(q.size(3) % flash_attention::load_size == 0);
  NVTE_CHECK(q.size(3) == flash_attention::load_size);
  NVTE_CHECK(k.size(3) % flash_attention::load_size == 0);
  NVTE_CHECK(k.size(3) == flash_attention::load_size);
  NVTE_CHECK(v.size(3) % flash_attention::load_size == 0);
  NVTE_CHECK(v.size(3) == flash_attention::load_size);

  // 3 x [s, b, n, h] -> [b, s, n, 3 * h]

  std::vector<int64_t> shape = {q.size(1), q.size(0), q.size(2), 3 * q.size(3)};
  at::Tensor qkv = at::empty(shape, at::CUDA(q.scalar_type()));

  size_t warps = q.size(0) * q.size(1);
  size_t warps_per_block = flash_attention::block_size / flash_attention::warp_size;
  size_t blocks = (warps + warps_per_block - 1) / warps_per_block;
  dim3 grid(blocks, 3);
  int threads = flash_attention::block_size;
  if (q.scalar_type() == at::ScalarType::Half) {
    using dtype = at::Half;
    flash_attention::prepare_kernel_bwd<dtype>
        <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<dtype>(), k.data_ptr<dtype>(), v.data_ptr<dtype>(), qkv.data_ptr<dtype>(),
            q.size(0), q.size(1), q.size(2), q.size(3));
  } else {
    using dtype = at::BFloat16;
    flash_attention::prepare_kernel_bwd<dtype>
        <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<dtype>(), k.data_ptr<dtype>(), v.data_ptr<dtype>(), qkv.data_ptr<dtype>(),
            q.size(0), q.size(1), q.size(2), q.size(3));
  }

  return qkv;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Binary search
 **************************************************************************************************/

__forceinline__ __device__ int binary_search(int target, int *array, int len) {
  int left = 1, right = len - 1;
  while (left < right) {
    int mid = (left + right) / 2;
    if (array[mid] <= target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Read the half of a THD tensor
 **************************************************************************************************/

__global__ void thd_read_half_tensor_kernel(void *half, void *tensor, int *cu_seqlens, int batch,
                                            int hidden_size_in_bytes, int half_idx,
                                            int dim_size_of_token) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    cu_seqlens_s[i] = cu_seqlens[i] / 2;
  }
  __syncthreads();

  int warpid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int laneid = threadIdx.x % 32;
  int num_warps = (blockDim.x * gridDim.x) / 32;
  int num_total_tokens = cu_seqlens_s[batch];
  int num_float4s_per_token = hidden_size_in_bytes / sizeof(float4);

  size_t offset = static_cast<size_t>(dim_size_of_token) * hidden_size_in_bytes;
  half = reinterpret_cast<void *>(reinterpret_cast<char *>(half) + offset / 2 * blockIdx.y);
  tensor = reinterpret_cast<void *>(reinterpret_cast<char *>(tensor) + offset * blockIdx.y);

  for (int token_id = warpid; token_id < num_total_tokens; token_id += num_warps) {
    int seqid = binary_search(token_id, cu_seqlens_s, batch + 1);

    size_t offset_in_bytes = static_cast<size_t>(token_id) * hidden_size_in_bytes;
    float4 *cur_half_token =
        reinterpret_cast<float4 *>(reinterpret_cast<char *>(half) + offset_in_bytes);

    offset_in_bytes =
        (static_cast<size_t>(token_id) + cu_seqlens_s[seqid + half_idx]) * hidden_size_in_bytes;
    float4 *cur_token =
        reinterpret_cast<float4 *>(reinterpret_cast<char *>(tensor) + offset_in_bytes);

    for (int idx = laneid; idx < num_float4s_per_token; idx += 32) {
      cur_half_token[idx] = cur_token[idx];
    }
  }
}

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

  // Launch Kernel
  constexpr unsigned int block = 256;
  unsigned int grid_x = (tensor.size(seq_dim) / 2 * 32 + block - 1) / block;
  unsigned int grid_y = 1;
  for (int i = 0; i < seq_dim; i++) {
    grid_y *= tensor.size(i);
  }
  dim3 grid = {grid_x, grid_y};
  thd_read_half_tensor_kernel<<<grid, block, sizeof(int) * (batch + 1),
                                at::cuda::getCurrentCUDAStream()>>>(
      half.data_ptr(), tensor.data_ptr(), cu_seqlens.data_ptr<int>(), batch, hidden_size_in_bytes,
      half_idx, tensor.size(seq_dim));

  return half;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: softmax_lse related operations
 **************************************************************************************************/

template <typename lse_dtype, typename Functor>
__global__ void thd_lse_kernel(lse_dtype *lse, float *half_lse, int *cu_seqlens, int batch,
                               int num_heads, int max_seqlen) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    cu_seqlens_s[i] = cu_seqlens[i] / 2;
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  int num_total_tokens = cu_seqlens_s[batch];

  for (int token_id = tid; token_id < num_total_tokens; token_id += num_threads) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);
    for (int head_id = blockIdx.y; head_id < num_heads; head_id += gridDim.y) {
      size_t row = static_cast<size_t>(seq_id) * num_heads + head_id;
      int col = token_id - cu_seqlens_s[seq_id];
      int seq_len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];

      size_t idx = row * max_seqlen + col + seq_len;
      size_t half_idx = row * max_seqlen / 2 + col;

      Functor::run(lse, half_lse, idx, half_idx);
    }
  }
}

struct LseCorrectionFunctor {
  __forceinline__ __device__ static void run(double *lse, float *half_lse, size_t idx,
                                             size_t half_idx) {
    double val = lse[idx];
    float val_per_step = half_lse[half_idx];
    double max_scale = max(val, val_per_step);
    double min_scale = min(val, val_per_step);
    lse[idx] = max_scale + log(1.0 + exp(min_scale - max_scale));
  }
};

void thd_second_half_lse_correction(at::Tensor lse, const at::Tensor &lse_per_step,
                                    const at::Tensor &cu_seqlens, int total_tokens) {
  NVTE_CHECK(lse.scalar_type() == at::ScalarType::Double);
  NVTE_CHECK(lse_per_step.scalar_type() == at::ScalarType::Float);
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);

  NVTE_CHECK(lse.dim() == 3);
  NVTE_CHECK(lse_per_step.dim() == 3);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  int batch = lse.size(0);
  int num_heads = lse.size(1);
  int max_seqlen = lse.size(2);

  NVTE_CHECK(lse_per_step.size(0) == batch);
  NVTE_CHECK(lse_per_step.size(1) == num_heads);
  NVTE_CHECK(lse_per_step.size(2) == max_seqlen / 2);
  NVTE_CHECK(cu_seqlens.size(0) == batch + 1);

  constexpr unsigned int block = 256;
  unsigned int grid_x = (total_tokens / 2 + block - 1) / block;
  unsigned int grid_y = num_heads;
  dim3 grid = {grid_x, grid_y};
  thd_lse_kernel<double, LseCorrectionFunctor>
      <<<grid, block, sizeof(int) * (batch + 1), at::cuda::getCurrentCUDAStream()>>>(
          lse.data_ptr<double>(), lse_per_step.data_ptr<float>(), cu_seqlens.data_ptr<int>(), batch,
          num_heads, max_seqlen);
}

struct ReadLseFunctor {
  __forceinline__ __device__ static void run(float *lse, float *half_lse, size_t idx,
                                             size_t half_idx) {
    half_lse[half_idx] = lse[idx];
  }
};

at::Tensor thd_read_second_half_lse(const at::Tensor &lse, const at::Tensor &cu_seqlens,
                                    int total_tokens) {
  NVTE_CHECK(lse.scalar_type() == at::ScalarType::Float);
  NVTE_CHECK(lse.dim() == 3);
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  int batch = lse.size(0);
  int num_heads = lse.size(1);
  int max_seqlen = lse.size(2);

  NVTE_CHECK(cu_seqlens.size(0) == batch + 1);

  std::vector<int64_t> shape = {batch, num_heads, max_seqlen / 2};
  at::Tensor half_lse = at::zeros(shape, at::CUDA(lse.scalar_type()));

  constexpr unsigned int block = 256;
  unsigned int grid_x = (total_tokens / 2 + block - 1) / block;
  unsigned int grid_y = num_heads;
  dim3 grid = {grid_x, grid_y};
  thd_lse_kernel<float, ReadLseFunctor>
      <<<grid, block, sizeof(int) * (batch + 1), at::cuda::getCurrentCUDAStream()>>>(
          lse.data_ptr<float>(), half_lse.data_ptr<float>(), cu_seqlens.data_ptr<int>(), batch,
          num_heads, max_seqlen);

  return half_lse;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Out correction in forward
 **************************************************************************************************/

template <typename dtype, int only_second_half, int tile_size>
__global__ void thd_out_correction_kernel(dtype *out, dtype *out_per_step, float *lse,
                                          float *lse_per_step, int *cu_seqlens, int batch,
                                          int num_heads, int dim_per_head, int max_seqlen) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    cu_seqlens_s[i] = cu_seqlens[i] / (only_second_half + 1);
  }
  __syncthreads();

  int tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / tile_size;
  int lane_id = threadIdx.x % tile_size;
  int num_tiles = (blockDim.x * gridDim.x) / tile_size;
  int num_total_tokens = cu_seqlens_s[batch];
  int num_loops_per_head = dim_per_head * sizeof(dtype) / sizeof(float4);

  for (int token_id = tile_id; token_id < num_total_tokens; token_id += num_tiles) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);
    for (int head_id = blockIdx.y; head_id < num_heads; head_id += gridDim.y) {
      size_t idx, idx_per_step;

      size_t row = static_cast<size_t>(seq_id) * num_heads + head_id;
      int col = token_id - cu_seqlens_s[seq_id];
      int seq_len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];
      idx = row * max_seqlen + col + seq_len * only_second_half;
      idx_per_step = row * max_seqlen / (only_second_half + 1) + col;
      float lse_corrected_exp = exp(lse_per_step[idx_per_step] - lse[idx]);

      idx = token_id + cu_seqlens_s[seq_id + 1] * only_second_half;
      idx = (idx * num_heads + head_id) * dim_per_head;
      idx_per_step = (static_cast<size_t>(token_id) * num_heads + head_id) * dim_per_head;
      dtype *cur_out = out + idx;
      dtype *cur_out_per_step = out_per_step + idx_per_step;

      for (int j = lane_id; j < num_loops_per_head; j += tile_size) {
        float4 data_per_step = reinterpret_cast<float4 *>(cur_out_per_step)[j];
        float4 data = reinterpret_cast<float4 *>(cur_out)[j];
        dtype *p_per_step = reinterpret_cast<dtype *>(&data_per_step);
        dtype *p = reinterpret_cast<dtype *>(&data);
        for (int k = 0; k < sizeof(float4) / sizeof(dtype); k++) {
          p[k] += p_per_step[k] * lse_corrected_exp;
        }
        reinterpret_cast<float4 *>(cur_out)[j] = data;
      }
    }
  }
}

template <typename dtype, int only_second_half>
static void thd_out_correction_helper(at::Tensor out, const at::Tensor &out_per_step,
                                      const at::Tensor &lse, const at::Tensor &lse_per_step,
                                      const at::Tensor &cu_seqlens) {
  NVTE_CHECK(out.scalar_type() == out_per_step.scalar_type());
  NVTE_CHECK(lse.scalar_type() == at::ScalarType::Float);
  NVTE_CHECK(lse_per_step.scalar_type() == at::ScalarType::Float);
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);

  int total_tokens = out.size(0);
  int num_heads = out.size(1);
  int dim_per_head = out.size(2);
  int batch = lse.size(0);
  int max_seqlen = lse.size(2);

  NVTE_CHECK(out_per_step.size(0) == total_tokens / (only_second_half + 1));
  NVTE_CHECK(out_per_step.size(1) == num_heads);
  NVTE_CHECK(out_per_step.size(2) == dim_per_head);
  NVTE_CHECK(lse.size(1) == num_heads);
  NVTE_CHECK(lse_per_step.size(0) == batch);
  NVTE_CHECK(lse_per_step.size(1) == num_heads);
  NVTE_CHECK(lse_per_step.size(2) == max_seqlen / (only_second_half + 1));
  NVTE_CHECK(cu_seqlens.size(0) == batch + 1);

  constexpr int tile = 16;
  constexpr int block = 512;
  unsigned int grid_x =
      (static_cast<size_t>(total_tokens) / (only_second_half + 1) * tile + block - 1) / block;
  dim3 grid = {grid_x, (unsigned int)num_heads};

  thd_out_correction_kernel<dtype, only_second_half, tile>
      <<<grid, block, sizeof(int) * (batch + 1), at::cuda::getCurrentCUDAStream()>>>(
          out.data_ptr<dtype>(), out_per_step.data_ptr<dtype>(), lse.data_ptr<float>(),
          lse_per_step.data_ptr<float>(), cu_seqlens.data_ptr<int>(), batch, num_heads,
          dim_per_head, max_seqlen);
}

void thd_out_correction(at::Tensor out, const at::Tensor &out_per_step, const at::Tensor &lse,
                        const at::Tensor &lse_per_step, const at::Tensor &cu_seqlens,
                        bool only_second_half) {
  if (only_second_half) {
    if (out.scalar_type() == at::ScalarType::Half) {
      using dtype = at::Half;
      thd_out_correction_helper<dtype, 1>(out, out_per_step, lse, lse_per_step, cu_seqlens);
    } else if (out.scalar_type() == at::ScalarType::BFloat16) {
      using dtype = at::BFloat16;
      thd_out_correction_helper<dtype, 1>(out, out_per_step, lse, lse_per_step, cu_seqlens);
    } else if (out.scalar_type() == at::ScalarType::Float) {
      using dtype = float;
      thd_out_correction_helper<dtype, 1>(out, out_per_step, lse, lse_per_step, cu_seqlens);
    } else {
      NVTE_ERROR("Unsupported dtype of out\n");
    }
  } else {
    if (out.scalar_type() == at::ScalarType::Half) {
      using dtype = at::Half;
      thd_out_correction_helper<dtype, 0>(out, out_per_step, lse, lse_per_step, cu_seqlens);
    } else if (out.scalar_type() == at::ScalarType::BFloat16) {
      using dtype = at::BFloat16;
      thd_out_correction_helper<dtype, 0>(out, out_per_step, lse, lse_per_step, cu_seqlens);
    } else if (out.scalar_type() == at::ScalarType::Float) {
      using dtype = float;
      thd_out_correction_helper<dtype, 0>(out, out_per_step, lse, lse_per_step, cu_seqlens);
    } else {
      NVTE_ERROR("Unsupported dtype of out\n");
    }
  }
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Gradients correction in backward
 **************************************************************************************************/

template <typename dtype, typename Functor_0, typename Functor_1, int functor_idx, int group_size>
__global__ void thd_grad_correction_kernel(dtype *grad, dtype *grad_per_step, int *cu_seqlens,
                                           int batch, int hidden_size, int dim_size_of_token) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    if constexpr (functor_idx < 2) {
      cu_seqlens_s[i] = cu_seqlens[i] / 2;
    } else {
      cu_seqlens_s[i] = cu_seqlens[i];
    }
  }
  __syncthreads();

  int group_id = (blockIdx.x * blockDim.x + threadIdx.x) / group_size;
  int lane_id = threadIdx.x % group_size;
  int num_groups = (blockDim.x * gridDim.x) / group_size;
  int num_total_tokens = cu_seqlens_s[batch];
  int num_inner_loops = hidden_size * sizeof(dtype) / sizeof(float4);

  size_t offset = static_cast<size_t>(dim_size_of_token) * hidden_size;
  if constexpr (functor_idx < 2) {
    grad_per_step = grad_per_step + offset / 2 * blockIdx.y;
  } else {
    grad_per_step = grad_per_step + offset * blockIdx.y;
  }
  grad = grad + offset * blockIdx.y;

  for (int token_id = group_id; token_id < num_total_tokens; token_id += num_groups) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);

    int token_offset;
    bool is_first_half;
    if constexpr (functor_idx < 2) {
      token_offset = cu_seqlens_s[seq_id + functor_idx];
      is_first_half = (functor_idx == 0);
    } else {
      token_offset = 0;
      int len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];
      is_first_half = (token_id - cu_seqlens_s[seq_id]) < (len / 2);
    }

    dtype *token = &grad[(token_id + token_offset) * static_cast<size_t>(hidden_size)];
    dtype *token_per_step = &grad_per_step[token_id * static_cast<size_t>(hidden_size)];
    for (int idx = lane_id; idx < num_inner_loops; idx += group_size) {
      if (is_first_half) {
        Functor_0::run(token, token_per_step, idx);
      } else {
        Functor_1::run(token, token_per_step, idx);
      }
    }
  }
}

struct EmptyFunctor {
  __forceinline__ __device__ static void run(void *token, void *token_per_step, int idx) {}
};

struct CopyFunctor {
  __forceinline__ __device__ static void run(void *token, void *token_per_step, int idx) {
    reinterpret_cast<float4 *>(token)[idx] = reinterpret_cast<float4 *>(token_per_step)[idx];
  }
};

template <typename dtype>
struct AddFunctor {
  __forceinline__ __device__ static void run(dtype *token, dtype *token_per_step, int idx) {
    float4 d_ = reinterpret_cast<float4 *>(token)[idx];
    dtype *p_ = reinterpret_cast<dtype *>(&d_);

    float4 d = reinterpret_cast<float4 *>(token_per_step)[idx];
    dtype *p = reinterpret_cast<dtype *>(&d);

#pragma unroll
    for (int i = 0; i < sizeof(float4) / sizeof(dtype); i++) {
      p_[i] += p[i];
    }

    reinterpret_cast<float4 *>(token)[idx] = d_;
  }
};

template <typename dtype, typename Functor_0, typename Functor_1, int functor_idx>
static void thd_grad_correction_helper(at::Tensor grad, const at::Tensor &grad_per_step,
                                       const at::Tensor &cu_seqlens) {
  NVTE_CHECK(grad.dim() == 3 || grad.dim() == 4);
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  // Shape of dq is [t, h, d], so the dimension of "t" is 0
  // Shape of dkv is [2, t, h, d], so the dimension of "t" is 1
  int seq_dim = grad.dim() == 3 ? 0 : 1;

  int total_tokens = grad.size(seq_dim);
  int num_heads = grad.size(seq_dim + 1);
  int dim_per_head = grad.size(seq_dim + 2);
  int batch = cu_seqlens.size(0) - 1;

  if constexpr (functor_idx < 2) {
    NVTE_CHECK(grad_per_step.size(seq_dim) == total_tokens / 2);
  } else {
    NVTE_CHECK(grad_per_step.size(seq_dim) == total_tokens);
  }
  NVTE_CHECK(grad_per_step.size(seq_dim + 1) == num_heads);
  NVTE_CHECK(grad_per_step.size(seq_dim + 2) == dim_per_head);

  size_t hidden_size = num_heads * dim_per_head;
  NVTE_CHECK((hidden_size * c10::elementSize(grad.scalar_type())) % 16 == 0);

  constexpr unsigned int block = 256;
  unsigned int grid_x;
  if constexpr (functor_idx < 2) {
    grid_x = (total_tokens / 2 * 32 + block - 1) / block;
  } else {
    grid_x = (total_tokens * 32 + block - 1) / block;
  }
  unsigned int grid_y = 1;
  for (int i = 0; i < seq_dim; i++) {
    grid_y *= grad.size(i);
  }
  dim3 grid = {grid_x, grid_y};

  thd_grad_correction_kernel<dtype, Functor_0, Functor_1, functor_idx, 32>
      <<<grid, block, sizeof(int) * (batch + 1), at::cuda::getCurrentCUDAStream()>>>(
          grad.data_ptr<dtype>(), grad_per_step.data_ptr<dtype>(), cu_seqlens.data_ptr<int>(),
          batch, hidden_size, total_tokens);
}

template <typename dtype>
static void thd_grad_dispatcher(at::Tensor grad, const at::Tensor &grad_per_step,
                                const at::Tensor &cu_seqlens, const std::string &first_half,
                                const std::string &second_half) {
  if (first_half == "add" && second_half == "none") {
    thd_grad_correction_helper<dtype, AddFunctor<dtype>, EmptyFunctor, 0>(grad, grad_per_step,
                                                                          cu_seqlens);
  } else if (first_half == "copy" && second_half == "none") {
    thd_grad_correction_helper<dtype, CopyFunctor, EmptyFunctor, 0>(grad, grad_per_step,
                                                                    cu_seqlens);
  } else if (first_half == "none" && second_half == "add") {
    thd_grad_correction_helper<dtype, EmptyFunctor, AddFunctor<dtype>, 1>(grad, grad_per_step,
                                                                          cu_seqlens);
  } else if (first_half == "none" && second_half == "copy") {
    thd_grad_correction_helper<dtype, EmptyFunctor, CopyFunctor, 1>(grad, grad_per_step,
                                                                    cu_seqlens);
  } else if (first_half == "add" && second_half == "copy") {
    thd_grad_correction_helper<dtype, AddFunctor<dtype>, CopyFunctor, 2>(grad, grad_per_step,
                                                                         cu_seqlens);
  } else if (first_half == "copy" && second_half == "add") {
    thd_grad_correction_helper<dtype, CopyFunctor, AddFunctor<dtype>, 2>(grad, grad_per_step,
                                                                         cu_seqlens);
  } else {
    NVTE_ERROR("Unsupported Functor of first half and second_half\n");
  }
}

void thd_grad_correction(at::Tensor grad, const at::Tensor &grad_per_step,
                         const at::Tensor &cu_seqlens, const std::string &first_half,
                         const std::string &second_half) {
  if (grad.scalar_type() == at::ScalarType::Half) {
    thd_grad_dispatcher<at::Half>(grad, grad_per_step, cu_seqlens, first_half, second_half);
  } else if (grad.scalar_type() == at::ScalarType::BFloat16) {
    thd_grad_dispatcher<at::BFloat16>(grad, grad_per_step, cu_seqlens, first_half, second_half);
  } else if (grad.scalar_type() == at::ScalarType::Float) {
    thd_grad_dispatcher<float>(grad, grad_per_step, cu_seqlens, first_half, second_half);
  } else {
    NVTE_ERROR("Unsupported dtype of grad\n");
  }
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Generate partitioned indices for input tokens
 **************************************************************************************************/

__global__ void thd_partition_indices_kernel(int *output, int *cu_seqlens, int batch,
                                             int total_tokens, int world_size, int rank) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    int seqlen = cu_seqlens[i];
    // Currently we assume that each sequence length is divisible by (world_size*2) since we have
    // to distribute each sequence evenly to different GPUs.
    assert(seqlen % (world_size * 2) == 0);
    cu_seqlens_s[i] = seqlen / world_size;
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (int token_id = tid; token_id < total_tokens / world_size; token_id += num_threads) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);
    int seq_len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];
    int index = token_id - cu_seqlens_s[seq_id];
    int offset = index < seq_len / 2 ? rank : (world_size - 1) * 2 - rank;
    index += cu_seqlens_s[seq_id] * world_size + seq_len / 2 * offset;
    output[token_id] = index;
  }
}

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

  constexpr unsigned int block = 256;
  unsigned int grid = (output.size(0) + block - 1) / block;
  thd_partition_indices_kernel<<<grid, block, sizeof(int) * (batch + 1),
                                 at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<int>(), cu_seqlens.data_ptr<int>(), batch, total_tokens, world_size, rank);

  return output;
}
