/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

namespace transformer_engine::pytorch {

void fp8_block_scaling_compute_partial_amax(const at::Tensor &tensor, at::Tensor amax, size_t h,
                                            size_t w, size_t start_offset, size_t block_len) {
  TORCH_CHECK(block_len == 128, "Currently only block_len = 128 is supported");
  TORCH_CHECK(amax.dim() == 2, "amax must be a 2D tensor");
  TORCH_CHECK(amax.scalar_type() == at::ScalarType::Float, "amax must be a float tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float ||
                  tensor.scalar_type() == at::ScalarType::BFloat16,
              "tensor must be a float or bfloat16 tensor");

  const TensorWrapper tensor_cu = makeTransformerEngineTensor(tensor);
  TensorWrapper amax_cu = makeTransformerEngineTensor(amax);

  nvte_fp8_block_scaling_compute_partial_amax(tensor_cu.data(), amax_cu.data(), h, w,
                                              amax.stride(0), amax.stride(1), start_offset,
                                              block_len, at::cuda::getCurrentCUDAStream());
}

void fp8_block_scaling_partial_cast(const at::Tensor &inp, at::Tensor out, const at::Tensor &scale,
                                    size_t h, size_t w, size_t start_offset, size_t block_len,
                                    const transformer_engine::DType out_dtype) {
  TORCH_CHECK(block_len == 128, "Currently only block_len = 128 is supported");
  TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "scale must be a float tensor");
  TORCH_CHECK(
      inp.scalar_type() == at::ScalarType::Float || inp.scalar_type() == at::ScalarType::BFloat16,
      "input must be a float or bfloat16 tensor");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Byte, "output must be a uint8 tensor");
  TORCH_CHECK(out_dtype == transformer_engine::DType::kFloat8E4M3 ||
                  out_dtype == transformer_engine::DType::kFloat8E5M2,
              "out_dtype must be kFloat8E4M3 or kFloat8E5M2");

  const TensorWrapper inp_cu = makeTransformerEngineTensor(inp);
  const auto contiguous_scale = scale.contiguous();
  TensorWrapper out_cu(NVTE_BLOCK_SCALING_2D);
  out_cu
      .set_rowwise_data(out.data_ptr(), out_dtype,
                        std::vector<size_t>{static_cast<size_t>(out.numel())})
      .set_scale(contiguous_scale.data_ptr(), DType::kFloat32,
                 std::vector<size_t>{static_cast<size_t>(scale.size(0)),
                                     static_cast<size_t>(scale.size(1))});

  nvte_partial_cast(inp_cu.data(), out_cu.data(), h, w, start_offset,
                    at::cuda::getCurrentCUDAStream());
}

void mxfp8_scaling_compute_partial_amax(const at::Tensor &input, at::Tensor amax_rowwise,
                                        at::Tensor amax_colwise, int rows, int cols,
                                        size_t start_offset) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(amax_rowwise.is_contiguous(), "amax_rowwise must be contiguous");
  TORCH_CHECK(amax_colwise.is_contiguous(), "amax_colwise must be contiguous");

  const TensorWrapper input_cu = makeTransformerEngineTensor(input);
  TensorWrapper amax_rowwise_cu = makeTransformerEngineTensor(amax_rowwise);
  // Empty PyTorch views may have a backing pointer. Normalize omission at this boundary.
  TensorWrapper amax_colwise_cu =
      amax_colwise.numel() == 0 ? TensorWrapper() : makeTransformerEngineTensor(amax_colwise);

  nvte_mxfp8_scaling_compute_partial_amax(input_cu.data(), amax_rowwise_cu.data(),
                                          amax_colwise_cu.data(), rows, cols, start_offset,
                                          at::cuda::getCurrentCUDAStream());
}

void mxfp8_scaling_partial_cast(const at::Tensor &input, at::Tensor output_rowwise,
                                at::Tensor output_colwise, const at::Tensor &scale_inv_rowwise,
                                const at::Tensor &scale_inv_colwise, int rows, int cols,
                                size_t start_offset) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output_rowwise.is_contiguous(), "output_rowwise must be contiguous");
  TORCH_CHECK(scale_inv_rowwise.is_contiguous(), "scale_inv_rowwise must be contiguous");
  TORCH_CHECK(output_rowwise.scalar_type() == at::ScalarType::Byte ||
                  output_rowwise.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "rowwise output must be uint8 or float8_e4m3fn");
  TORCH_CHECK(scale_inv_rowwise.scalar_type() == at::ScalarType::Byte,
              "MXFP8 rowwise inverse scales must use uint8 storage");

  const TensorWrapper input_cu = makeTransformerEngineTensor(input);
  TensorWrapper output_cu(NVTE_MXFP8_1D_SCALING);
  output_cu
      .set_rowwise_data(output_rowwise.data_ptr(), DType::kFloat8E4M3,
                        std::vector<size_t>{static_cast<size_t>(output_rowwise.numel())})
      .set_rowwise_scale_inv(scale_inv_rowwise.data_ptr(), DType::kFloat8E8M0,
                             getTensorShape(scale_inv_rowwise));
  // Normalize empty output views, which may still have a backing data pointer,
  // to an absent columnwise data field. Auxiliary scales do not select usage.
  if (output_colwise.numel() != 0) {
    TORCH_CHECK(scale_inv_colwise.numel() != 0, "Columnwise output requires columnwise scales");
    TORCH_CHECK(output_colwise.is_contiguous(), "output_colwise must be contiguous");
    TORCH_CHECK(scale_inv_colwise.is_contiguous(), "scale_inv_colwise must be contiguous");
    TORCH_CHECK(output_colwise.scalar_type() == at::ScalarType::Byte ||
                    output_colwise.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "columnwise output must be uint8 or float8_e4m3fn");
    TORCH_CHECK(scale_inv_colwise.scalar_type() == at::ScalarType::Byte,
                "MXFP8 columnwise inverse scales must use uint8 storage");
    output_cu
        .set_columnwise_data(output_colwise.data_ptr(), DType::kFloat8E4M3,
                             std::vector<size_t>{static_cast<size_t>(output_colwise.numel())})
        .set_columnwise_scale_inv(scale_inv_colwise.data_ptr(), DType::kFloat8E8M0,
                                  getTensorShape(scale_inv_colwise));
  }

  nvte_partial_cast(input_cu.data(), output_cu.data(), rows, cols, start_offset,
                    at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
