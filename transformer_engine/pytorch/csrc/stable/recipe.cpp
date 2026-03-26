/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../stable_common.h"

#include <transformer_engine/recipe.h>
#include <transformer_engine/fused_attn.h>
#include <transformer_engine/multi_stream.h>

#include <cublasLt.h>
#include <cudnn.h>

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

int64_t get_cublasLt_version() {
  return static_cast<int64_t>(cublasLtGetVersion());
}

int64_t get_cudnn_version() {
  return static_cast<int64_t>(cudnnGetVersion());
}

void compute_amax(Tensor input, Tensor amax) {
  auto input_ = torch::stable::contiguous(input);
  auto input_cu = makeTransformerEngineTensor(input_);
  auto shape = getStableTensorShape(input_);

  // Build output TensorWrapper with amax pointer
  TensorWrapper fake_output(NVTE_DELAYED_TENSOR_SCALING);
  fake_output.set_rowwise_data(nullptr, DType::kFloat32,
                               std::vector<size_t>(shape.begin(), shape.end()));
  fake_output.set_amax(amax.data_ptr(), DType::kFloat32, std::vector<size_t>{1});

  nvte_compute_amax(input_cu.data(), fake_output.data(),
                    getCurrentCUDAStreamRaw(input_.get_device_index()));
}

// fused_amax_and_scale_update_after_reduction uses the pointer-pack pattern:
// Python passes flat int64 tensors with data_ptr() values for amax_histories and scales.
// fused_amax_and_scale_update: use pointer+ndim+shape encoding
// shapes tensor: [num_tensors * 3] — (ndim, dim0, dim1) per tensor. dim1=0 for 1D.
void fused_amax_and_scale_update(
    Tensor amax_reduction_buffer,
    Tensor amax_history_ptrs,   // int64 [num_tensors] — data_ptr() per history
    Tensor amax_history_shapes, // int64 [num_tensors * 3] — (ndim, dim0, dim1) per history
    Tensor scale_ptrs,          // int64 [num_tensors] — data_ptr() per scale
    Tensor scale_shapes,        // int64 [num_tensors * 3] — (ndim, dim0, dim1) per scale
    int64_t num_tensors,
    std::string amax_compute_algo, int64_t fp8_dtype, double margin) {
  auto buf_cu = makeTransformerEngineTensor(amax_reduction_buffer);

  const int64_t* ah_ptrs = static_cast<const int64_t*>(amax_history_ptrs.data_ptr());
  const int64_t* ah_shapes = static_cast<const int64_t*>(amax_history_shapes.data_ptr());
  const int64_t* sc_ptrs = static_cast<const int64_t*>(scale_ptrs.data_ptr());
  const int64_t* sc_shapes = static_cast<const int64_t*>(scale_shapes.data_ptr());

  std::vector<NVTETensor> te_amax_histories;
  std::vector<NVTETensor> te_scales;
  te_amax_histories.reserve(num_tensors);
  te_scales.reserve(num_tensors);

  for (int64_t i = 0; i < num_tensors; i++) {
    te_amax_histories.push_back(nvte_create_tensor(NVTE_DELAYED_TENSOR_SCALING));
    size_t ah_ndim = static_cast<size_t>(ah_shapes[i*3]);
    size_t ah_dims[] = {static_cast<size_t>(ah_shapes[i*3+1]),
                        static_cast<size_t>(ah_shapes[i*3+2])};
    NVTEShape amax_shape = nvte_make_shape(ah_dims, ah_ndim);
    NVTEBasicTensor amax_data = {reinterpret_cast<void*>(ah_ptrs[i]),
                                  static_cast<NVTEDType>(DType::kFloat32), amax_shape};
    nvte_set_tensor_param(&te_amax_histories.back(), kNVTERowwiseData, &amax_data);

    te_scales.push_back(nvte_create_tensor(NVTE_DELAYED_TENSOR_SCALING));
    size_t sc_ndim = static_cast<size_t>(sc_shapes[i*3]);
    size_t sc_dims[] = {static_cast<size_t>(sc_shapes[i*3+1]),
                        static_cast<size_t>(sc_shapes[i*3+2])};
    NVTEShape scale_shape = nvte_make_shape(sc_dims, sc_ndim);
    NVTEBasicTensor scale_data = {reinterpret_cast<void*>(sc_ptrs[i]),
                                   static_cast<NVTEDType>(DType::kFloat32), scale_shape};
    nvte_set_tensor_param(&te_scales.back(), kNVTERowwiseData, &scale_data);
  }

  nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
      buf_cu.data(), te_amax_histories, te_scales,
      amax_compute_algo.c_str(), static_cast<NVTEDType>(fp8_dtype),
      static_cast<float>(margin),
      getCurrentCUDAStreamRaw(amax_reduction_buffer.get_device_index()));

  for (auto& t : te_amax_histories) nvte_destroy_tensor(t);
  for (auto& t : te_scales) nvte_destroy_tensor(t);
}

int64_t get_fused_attn_backend(
    bool is_training, int64_t q_dtype, int64_t kv_dtype,
    int64_t qkv_layout, int64_t bias_type, int64_t attn_mask_type,
    int64_t softmax_type, double p_dropout,
    int64_t num_attn_heads, int64_t num_gqa_groups,
    int64_t max_seqlen_q, int64_t max_seqlen_kv,
    int64_t head_dim_qk, int64_t head_dim_v,
    int64_t window_size_left, int64_t window_size_right,
    bool return_max_logit, bool cuda_graph, bool deterministic) {
  return static_cast<int64_t>(nvte_get_fused_attn_backend(
      is_training,
      static_cast<NVTEDType>(q_dtype), static_cast<NVTEDType>(kv_dtype),
      static_cast<NVTE_QKV_Layout>(qkv_layout),
      static_cast<NVTE_Bias_Type>(bias_type),
      static_cast<NVTE_Mask_Type>(attn_mask_type),
      static_cast<NVTE_Softmax_Type>(softmax_type),
      static_cast<float>(p_dropout),
      static_cast<size_t>(num_attn_heads), static_cast<size_t>(num_gqa_groups),
      static_cast<size_t>(max_seqlen_q), static_cast<size_t>(max_seqlen_kv),
      static_cast<size_t>(head_dim_qk), static_cast<size_t>(head_dim_v),
      window_size_left, window_size_right,
      return_max_logit, cuda_graph, deterministic));
}

int64_t get_num_cublas_streams() {
  return static_cast<int64_t>(nvte_get_num_compute_streams());
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def("get_cublasLt_version() -> int");
  m.def("get_cudnn_version() -> int");
  m.def("compute_amax(Tensor input, Tensor amax) -> ()");
  m.def("fused_amax_and_scale_update(Tensor amax_reduction_buffer, Tensor amax_history_ptrs, Tensor amax_history_shapes, Tensor scale_ptrs, Tensor scale_shapes, int num_tensors, str amax_compute_algo, int fp8_dtype, float margin) -> ()");
  // shapes format: [num_tensors * 3] — (ndim, dim0, dim1) per tensor
  m.def("get_num_cublas_streams() -> int");
  m.def("get_fused_attn_backend(bool is_training, int q_dtype, int kv_dtype, int qkv_layout, int bias_type, int attn_mask_type, int softmax_type, float p_dropout, int num_attn_heads, int num_gqa_groups, int max_seqlen_q, int max_seqlen_kv, int head_dim_qk, int head_dim_v, int window_size_left, int window_size_right, bool return_max_logit, bool cuda_graph, bool deterministic) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  // Version queries registered under CUDA since they call CUDA library functions
  m.impl("get_cublasLt_version", TORCH_BOX(get_cublasLt_version));
  m.impl("get_cudnn_version", TORCH_BOX(get_cudnn_version));
  m.impl("compute_amax", TORCH_BOX(compute_amax));
  m.impl("fused_amax_and_scale_update", TORCH_BOX(fused_amax_and_scale_update));
  m.impl("get_num_cublas_streams", TORCH_BOX(get_num_cublas_streams));
  m.impl("get_fused_attn_backend", TORCH_BOX(get_fused_attn_backend));
}
