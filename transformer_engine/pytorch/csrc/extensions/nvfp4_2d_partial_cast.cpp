/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

namespace transformer_engine::pytorch {

void nvfp4_2d_compute_partial_amax(const at::Tensor &tensor, at::Tensor amax, size_t h, size_t w,
                                   size_t start_offset, size_t block_len) {
  TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");
  TORCH_CHECK(amax.dim() == 2, "amax must be a 2D tensor");
  TORCH_CHECK(amax.scalar_type() == at::ScalarType::Float, "amax must be a float tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float ||
                  tensor.scalar_type() == at::ScalarType::BFloat16,
              "tensor must be a float or bfloat16 tensor");

  const TensorWrapper tensor_cu = makeTransformerEngineTensor(tensor.contiguous());
  TensorWrapper amax_cu = makeTransformerEngineTensor(amax);

  nvte_nvfp4_2d_compute_partial_amax(
      tensor_cu.data(), amax_cu.data(), h, w, amax.stride(0), amax.stride(1), start_offset,
      block_len, at::cuda::getCurrentCUDAStream());
}

void nvfp4_2d_partial_cast(const at::Tensor &inp, py::handle out, const at::Tensor &scale,
                           const at::Tensor &global_scale, size_t h, size_t w, size_t start_offset,
                           size_t block_len) {
  TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");
  TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "scale must be a float tensor");
  TORCH_CHECK(global_scale.numel() == 1, "global_scale must be a scalar tensor");
  TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
              "global_scale must be a float tensor");
  TORCH_CHECK(inp.scalar_type() == at::ScalarType::Float ||
                  inp.scalar_type() == at::ScalarType::BFloat16,
              "input must be a float or bfloat16 tensor");

  const TensorWrapper inp_cu = makeTransformerEngineTensor(inp.contiguous());
  const TensorWrapper out_cu = makeTransformerEngineTensor(out, py::none());
  const TensorWrapper scale_cu = makeTransformerEngineTensor(scale);
  const TensorWrapper global_scale_cu = makeTransformerEngineTensor(global_scale);

  nvte_nvfp4_2d_partial_cast(inp_cu.data(), out_cu.data(), scale_cu.data(),
                             global_scale_cu.data(), h, w, scale.stride(0), scale.stride(1),
                             start_offset, block_len,
                             at::cuda::getCurrentCUDAStream());
}

void nvfp4_multi_tensor_compute_partial_amax(
    std::vector<at::Tensor> master_weight_list,
    std::vector<at::Tensor> partial_amax_list,
    std::vector<at::Tensor> global_amax_list,
    std::vector<int64_t> h_list,
    std::vector<int64_t> w_list,
    std::vector<int64_t> start_offset_list,
    int64_t block_len) {
  
  TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");
  
  const size_t num_tensors = master_weight_list.size();
  TORCH_CHECK(partial_amax_list.size() == num_tensors, "partial_amax_list size mismatch");
  TORCH_CHECK(global_amax_list.size() == num_tensors, "global_amax_list size mismatch");
  TORCH_CHECK(h_list.size() == num_tensors, "h_list size mismatch");
  TORCH_CHECK(w_list.size() == num_tensors, "w_list size mismatch");
  TORCH_CHECK(start_offset_list.size() == num_tensors, "start_offset_list size mismatch");

  if (num_tensors == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  for (size_t i = 0; i < num_tensors; ++i) {
    const auto& master_weight = master_weight_list[i];
    auto& partial_amax = partial_amax_list[i];
    auto& global_amax = global_amax_list[i];
    const size_t h = static_cast<size_t>(h_list[i]);
    const size_t w = static_cast<size_t>(w_list[i]);
    const size_t start_offset = static_cast<size_t>(start_offset_list[i]);

    TORCH_CHECK(partial_amax.dim() == 2, "partial_amax must be a 2D tensor");
    TORCH_CHECK(partial_amax.scalar_type() == at::ScalarType::Float,
                "partial_amax must be a float tensor");
    TORCH_CHECK(master_weight.scalar_type() == at::ScalarType::Float ||
                    master_weight.scalar_type() == at::ScalarType::BFloat16,
                "master_weight must be a float or bfloat16 tensor");
    TORCH_CHECK(global_amax.scalar_type() == at::ScalarType::Float,
                "global_amax must be a float tensor");
    TORCH_CHECK(global_amax.numel() == 1, "global_amax must have exactly one element");

    // Compute partial amax (per-block amax)
    const TensorWrapper tensor_cu = makeTransformerEngineTensor(master_weight.contiguous());
    TensorWrapper amax_cu = makeTransformerEngineTensor(partial_amax);

    nvte_nvfp4_2d_compute_partial_amax(
        tensor_cu.data(), amax_cu.data(), h, w,
        partial_amax.stride(0), partial_amax.stride(1),
        start_offset, static_cast<size_t>(block_len), stream);

    // Compute global amax
    auto* global_amax_ptr = global_amax.data_ptr<float>();
    TensorWrapper fake_te_output(
        /*dptr=*/nullptr, tensor_cu.shape(),
        DType::kFloat32,
        global_amax_ptr);

    nvte_compute_amax(tensor_cu.data(), fake_te_output.data(), stream);
  }
}

}  // namespace transformer_engine::pytorch


