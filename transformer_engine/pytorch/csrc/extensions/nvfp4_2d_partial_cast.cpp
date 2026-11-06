/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

namespace transformer_engine::pytorch {

void nvfp4_2d_compute_partial_amax(const at::Tensor& tensor, at::Tensor amax, size_t h, size_t w,
                                   size_t start_offset, size_t block_len) {
  TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");
  TORCH_CHECK(amax.dim() == 2, "amax must be a 2D tensor");
  TORCH_CHECK(amax.scalar_type() == at::ScalarType::Float, "amax must be a float tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float ||
                  tensor.scalar_type() == at::ScalarType::BFloat16,
              "tensor must be a float or bfloat16 tensor");

  const TensorWrapper tensor_cu = makeTransformerEngineTensor(tensor.contiguous());
  TensorWrapper amax_cu = makeTransformerEngineTensor(amax);

  nvte_nvfp4_2d_compute_partial_amax(tensor_cu.data(), amax_cu.data(), h, w, amax.stride(0),
                                     amax.stride(1), start_offset, block_len,
                                     at::cuda::getCurrentCUDAStream());
}

void nvfp4_2d_partial_cast(const at::Tensor& inp, py::handle out, const at::Tensor& scale,
                           const at::Tensor& global_scale, size_t h, size_t w, size_t start_offset,
                           size_t block_len, const DType scale_dtype) {
  TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");
  TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Byte,
              "scale must contain encoded NVFP4 inverse scales in uint8 storage");
  TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
  TORCH_CHECK(global_scale.numel() == 1, "global_scale must be a scalar tensor");
  TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
              "global_scale must be a float tensor");
  TORCH_CHECK(
      inp.scalar_type() == at::ScalarType::Float || inp.scalar_type() == at::ScalarType::BFloat16,
      "input must be a float or bfloat16 tensor");

  const auto contiguous_inp = inp.contiguous();
  const TensorWrapper inp_cu = makeTransformerEngineTensor(contiguous_inp);
  const TensorWrapper output_storage = makeTransformerEngineTensor(out, py::none());
  TensorWrapper out_cu(NVTE_NVFP4_1D_SCALING);
  out_cu
      .set_rowwise_data(output_storage.dptr(), DType::kFloat4E2M1,
                        std::vector<size_t>{static_cast<size_t>(inp.numel())})
      .set_rowwise_scale_inv(scale.data_ptr(), scale_dtype, getTensorShape(scale))
      .set_scale(global_scale.data_ptr(), DType::kFloat32, std::vector<size_t>{1});

  nvte_partial_cast(inp_cu.data(), out_cu.data(), h, w, start_offset,
                    at::cuda::getCurrentCUDAStream());
}

void nvfp4_multi_tensor_2d_partial_cast(std::vector<at::Tensor> inp_list,
                                        std::vector<at::Tensor> out_list,
                                        std::vector<at::Tensor> scale_list,
                                        std::vector<at::Tensor> global_scale_list,
                                        std::vector<int64_t> h_list, std::vector<int64_t> w_list,
                                        std::vector<int64_t> start_offset_list, int64_t block_len,
                                        const DType scale_dtype) {
  TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");

  const size_t num_tensors = inp_list.size();
  TORCH_CHECK(out_list.size() == num_tensors, "out_list size mismatch");
  TORCH_CHECK(scale_list.size() == num_tensors, "scale_list size mismatch");
  TORCH_CHECK(global_scale_list.size() == num_tensors, "global_scale_list size mismatch");
  TORCH_CHECK(h_list.size() == num_tensors, "h_list size mismatch");
  TORCH_CHECK(w_list.size() == num_tensors, "w_list size mismatch");
  TORCH_CHECK(start_offset_list.size() == num_tensors, "start_offset_list size mismatch");

  if (num_tensors == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  for (size_t i = 0; i < num_tensors; ++i) {
    const auto& inp = inp_list[i];
    const auto& out = out_list[i];
    const auto& scale = scale_list[i];
    const auto& global_scale = global_scale_list[i];
    const size_t h = static_cast<size_t>(h_list[i]);
    const size_t w = static_cast<size_t>(w_list[i]);
    const size_t start_offset = static_cast<size_t>(start_offset_list[i]);

    TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::Byte,
                "scale must contain encoded NVFP4 inverse scales in uint8 storage");
    TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
    TORCH_CHECK(global_scale.numel() == 1, "global_scale must be a scalar tensor");
    TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
                "global_scale must be a float tensor");
    TORCH_CHECK(
        inp.scalar_type() == at::ScalarType::Float || inp.scalar_type() == at::ScalarType::BFloat16,
        "input must be a float or bfloat16 tensor");

    TORCH_CHECK(out.scalar_type() == at::ScalarType::Byte && out.is_contiguous(),
                "output must be contiguous uint8 storage");
    TORCH_CHECK(out.numel() >= (inp.numel() + 1) / 2, "output storage is too small");
    const auto contiguous_inp = inp.contiguous();
    const TensorWrapper inp_cu = makeTransformerEngineTensor(contiguous_inp);
    TensorWrapper out_cu(NVTE_NVFP4_1D_SCALING);
    out_cu
        .set_rowwise_data(out.data_ptr(), DType::kFloat4E2M1,
                          std::vector<size_t>{static_cast<size_t>(inp.numel())})
        .set_rowwise_scale_inv(scale.data_ptr(), scale_dtype, getTensorShape(scale))
        .set_scale(global_scale.data_ptr(), DType::kFloat32, std::vector<size_t>{1});

    nvte_partial_cast(inp_cu.data(), out_cu.data(), h, w, start_offset, stream);
  }
}

void nvfp4_multi_tensor_compute_partial_amax(
    std::vector<at::Tensor> master_weight_list, std::vector<at::Tensor> partial_amax_list,
    std::vector<at::Tensor> global_amax_list, std::vector<int64_t> h_list,
    std::vector<int64_t> w_list, std::vector<int64_t> start_offset_list, int64_t block_len) {
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

    nvte_nvfp4_2d_compute_partial_amax(tensor_cu.data(), amax_cu.data(), h, w,
                                       partial_amax.stride(0), partial_amax.stride(1), start_offset,
                                       static_cast<size_t>(block_len), stream);

    // Compute global amax
    auto* global_amax_ptr = global_amax.data_ptr<float>();
    TensorWrapper fake_te_output(
        /*dptr=*/nullptr, tensor_cu.shape(), DType::kFloat32, global_amax_ptr);

    nvte_compute_amax(tensor_cu.data(), fake_te_output.data(), stream);
  }
}

}  // namespace transformer_engine::pytorch
