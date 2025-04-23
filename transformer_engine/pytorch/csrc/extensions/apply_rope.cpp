/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

at::Tensor fused_rope_forward(const at::Tensor &input, const at::Tensor &freqs,
                              const std::optional<at::Tensor> start_positions,
                              const NVTE_QKV_Format qkv_format, const bool interleaved,
                              const std::optional<at::Tensor> cu_seqlens, const int cp_size,
                              const int cp_rank) {
  using namespace transformer_engine::pytorch;

  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(freqs.scalar_type() == at::ScalarType::Float,
              "Dtype of the freqs tensor must be float");

  // output
  auto act_options = at::TensorOptions().dtype(input.scalar_type()).device(input.device());
  auto output = at::empty(input.sizes(), act_options);

  auto input_cu = makeTransformerEngineTensor(input);
  auto freqs_cu = makeTransformerEngineTensor(freqs);
  auto output_cu = makeTransformerEngineTensor(output);

  auto start_positions_cu = transformer_engine::TensorWrapper();  // empty cu_seqlens tensor
  if (start_positions) {
    start_positions_cu = makeTransformerEngineTensor(start_positions.value());
  }

  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
    TORCH_CHECK(cu_seqlens.has_value(), "expected cu_seqlens tensor");
    TORCH_CHECK(cu_seqlens.value().dim() == 1, "expected 1D tensor");
    TORCH_CHECK(input.size(2) >= freqs.size(3),
                "expected the last dim of the input tensor equals or is "
                "greater than the freqs tensor");

    // input sizes: (t, h, d)
    // t: cumulative sum of sequence lengths
    // h: head num
    // d: dim of each head
    // const int t = input.size(0);
    const int h = input.size(1);
    const int d = input.size(2);
    // input strides
    const int stride_t = input.stride(0);
    const int stride_h = input.stride(1);
    const int stride_d = input.stride(2);
    // batch size
    const int b = cu_seqlens.value().size(0) - 1;
    // freqs' shape is (max_s, 1, 1, d2)
    const int max_s = freqs.size(0);
    const int d2 = freqs.size(3);

    auto cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());

    nvte_fused_rope_forward(input_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                            start_positions_cu.data(), output_cu.data(), qkv_format, interleaved,
                            cp_size, cp_rank, max_s, b, h, d, d2, stride_t, /*stride_b=*/0,
                            stride_h, stride_d, at::cuda::getCurrentCUDAStream());

    return output;
  }

  TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  // input sizes: (s, b, h, d) or (b, s, h, d)
  // s: sequence length
  // b: batch size
  // h: head num
  // d: dim of each head
  const int s = qkv_format == NVTE_QKV_Format::NVTE_SBHD ? input.size(0) : input.size(1);
  const int b = qkv_format == NVTE_QKV_Format::NVTE_SBHD ? input.size(1) : input.size(0);
  const int h = input.size(2);
  const int d = input.size(3);
  // input strides
  const int stride_s = qkv_format == NVTE_QKV_Format::NVTE_SBHD ? input.stride(0) : input.stride(1);
  const int stride_b = qkv_format == NVTE_QKV_Format::NVTE_SBHD ? input.stride(1) : input.stride(0);
  const int stride_h = input.stride(2);
  const int stride_d = input.stride(3);
  // freqs' shape is always (s, 1, 1, d2), so the strides are same under
  // different memory formats
  const int d2 = freqs.size(3);

  TORCH_CHECK(s * cp_size <= freqs.size(0),
              "expected freqs tensor has a longer sequence length than input");
  TORCH_CHECK(d >= d2,
              "expected the last dim of the input tensor equals or is "
              "greater than the freqs tensor");

  auto cu_seqlens_cu = transformer_engine::TensorWrapper();  // empty cu_seqlens tensor
  nvte_fused_rope_forward(input_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                          start_positions_cu.data(), output_cu.data(), qkv_format, interleaved,
                          cp_size, cp_rank, s, b, h, d, d2, stride_s, stride_b, stride_h, stride_d,
                          at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor fused_rope_backward(const at::Tensor &output_grads, const at::Tensor &freqs,
                               const NVTE_QKV_Format qkv_format, const bool interleaved,
                               const std::optional<at::Tensor> cu_seqlens, const int cp_size,
                               const int cp_rank) {
  using namespace transformer_engine::pytorch;
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(freqs.scalar_type() == at::ScalarType::Float,
              "Dtype of the freqs tensor must be float");

  auto act_options =
      at::TensorOptions().dtype(output_grads.scalar_type()).device(output_grads.device());
  auto input_grads = at::empty(output_grads.sizes(), act_options);

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto freqs_cu = makeTransformerEngineTensor(freqs);
  auto input_grads_cu = makeTransformerEngineTensor(input_grads);

  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
    TORCH_CHECK(cu_seqlens.has_value(), "expected cu_seqlens tensor");
    TORCH_CHECK(cu_seqlens.value().dim() == 1, "expected 1D tensor");
    TORCH_CHECK(output_grads.size(2) >= freqs.size(3),
                "expected the last dim of the output_grads tensor equals or is "
                "greater than the freqs tensor");

    // output_grads sizes: (t, h, d)
    // t: cumulative sum of sequence lengths
    // h: head num
    // d: dim of each head
    // const int t = output_grads.size(0);
    const int h = output_grads.size(1);
    const int d = output_grads.size(2);
    // output_grads strides
    const int stride_t = output_grads.stride(0);
    const int stride_h = output_grads.stride(1);
    const int stride_d = output_grads.stride(2);
    // batch size
    const int b = cu_seqlens.value().size(0) - 1;
    // freqs' shape is (max_s, 1, 1, d2)
    const int max_s = freqs.size(0);
    const int d2 = freqs.size(3);

    auto cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());

    nvte_fused_rope_backward(output_grads_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                             input_grads_cu.data(), qkv_format, interleaved, cp_size, cp_rank,
                             max_s, b, h, d, d2, stride_t,
                             /*stride_b=*/0, stride_h, stride_d, at::cuda::getCurrentCUDAStream());

    return input_grads;
  }

  TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  // output_grads sizes: (s, b, h, d)
  // s: sequence length
  // b: batch size
  // h: head num
  // d: dim of each head
  const int s =
      qkv_format == NVTE_QKV_Format::NVTE_SBHD ? output_grads.size(0) : output_grads.size(1);
  const int b =
      qkv_format == NVTE_QKV_Format::NVTE_SBHD ? output_grads.size(1) : output_grads.size(0);
  const int h = output_grads.size(2);
  const int d = output_grads.size(3);
  // output_grads strides
  const int stride_s =
      qkv_format == NVTE_QKV_Format::NVTE_SBHD ? output_grads.stride(0) : output_grads.stride(1);
  const int stride_b =
      qkv_format == NVTE_QKV_Format::NVTE_SBHD ? output_grads.stride(1) : output_grads.stride(0);
  const int stride_h = output_grads.stride(2);
  const int stride_d = output_grads.stride(3);
  // freqs' shape is always (s, 1, 1, d2), so the strides are same under
  // different memory formats
  const int d2 = freqs.size(3);

  TORCH_CHECK(s * cp_size <= freqs.size(0),
              "expected freqs tensor has a longer sequence length than output_grads");
  TORCH_CHECK(d >= d2,
              "expected the last dim of the output_grads tensor equals or is "
              "greater than the freqs tensor");

  auto cu_seqlens_cu = transformer_engine::TensorWrapper();  // empty cu_seqlens tensor
  nvte_fused_rope_backward(output_grads_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                           input_grads_cu.data(), qkv_format, interleaved, cp_size, cp_rank, s, b,
                           h, d, d2, stride_s, stride_b, stride_h, stride_d,
                           at::cuda::getCurrentCUDAStream());

  return input_grads;
}
