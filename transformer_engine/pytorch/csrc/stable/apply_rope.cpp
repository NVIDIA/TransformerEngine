/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/fused_rope.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

Tensor fused_rope_forward(Tensor input, Tensor freqs, std::optional<Tensor> start_positions,
                          int64_t qkv_format, bool interleaved, std::optional<Tensor> cu_seqlens,
                          int64_t cp_size, int64_t cp_rank) {
  auto nvte_qkv_format = static_cast<NVTE_QKV_Format>(qkv_format);

  STD_TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
                  "expected the second and third dims of the freqs tensor equal 1");
  STD_TORCH_CHECK(freqs.scalar_type() == ScalarType::Float,
                  "Dtype of the freqs tensor must be float");

  // Allocate contiguous output (must NOT use empty_like which preserves
  // non-contiguous strides from transposed inputs)
  auto sizes = input.sizes();
  std::vector<int64_t> shape_vec(sizes.begin(), sizes.end());
  auto output = allocateStableTensor(shape_vec, input.scalar_type(), input.get_device_index());

  auto input_cu = makeTransformerEngineTensor(input);
  auto freqs_cu = makeTransformerEngineTensor(freqs);
  auto output_cu = makeTransformerEngineTensor(output);

  auto start_positions_cu = TensorWrapper();
  if (start_positions.has_value()) {
    start_positions_cu = makeTransformerEngineTensor(start_positions.value());
    STD_TORCH_CHECK(start_positions_cu.ndim() == 1, "expected 1D tensor");
  }

  auto stream = getCurrentCUDAStreamRaw(input.get_device_index());

  if (nvte_qkv_format == NVTE_QKV_Format::NVTE_THD) {
    STD_TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
    STD_TORCH_CHECK(cu_seqlens.has_value(), "expected cu_seqlens tensor");
    STD_TORCH_CHECK(cu_seqlens.value().dim() == 1, "expected 1D tensor");

    const int h = static_cast<int>(input.size(1));
    const int d = static_cast<int>(input.size(2));
    const int stride_t = static_cast<int>(input.stride(0));
    const int stride_h = static_cast<int>(input.stride(1));
    const int stride_d = static_cast<int>(input.stride(2));
    const int b = static_cast<int>(cu_seqlens.value().size(0) - 1);
    const int max_s = static_cast<int>(freqs.size(0));
    const int d2 = static_cast<int>(freqs.size(3));

    auto cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());

    nvte_fused_rope_forward(input_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                            start_positions_cu.data(), output_cu.data(), nvte_qkv_format,
                            interleaved, static_cast<int>(cp_size), static_cast<int>(cp_rank),
                            max_s, b, h, d, d2, stride_t, 0, stride_h, stride_d, stream);

    return output;
  }

  STD_TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  const bool is_sbhd = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD;
  const int s = static_cast<int>(is_sbhd ? input.size(0) : input.size(1));
  const int b = static_cast<int>(is_sbhd ? input.size(1) : input.size(0));
  const int h = static_cast<int>(input.size(2));
  const int d = static_cast<int>(input.size(3));
  const int stride_s = static_cast<int>(is_sbhd ? input.stride(0) : input.stride(1));
  const int stride_b = static_cast<int>(is_sbhd ? input.stride(1) : input.stride(0));
  const int stride_h = static_cast<int>(input.stride(2));
  const int stride_d = static_cast<int>(input.stride(3));
  const int d2 = static_cast<int>(freqs.size(3));

  auto cu_seqlens_cu = TensorWrapper();
  nvte_fused_rope_forward(input_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                          start_positions_cu.data(), output_cu.data(), nvte_qkv_format, interleaved,
                          static_cast<int>(cp_size), static_cast<int>(cp_rank), s, b, h, d, d2,
                          stride_s, stride_b, stride_h, stride_d, stream);

  return output;
}

Tensor fused_rope_backward(Tensor output_grads, Tensor freqs, std::optional<Tensor> start_positions,
                           int64_t qkv_format, bool interleaved, std::optional<Tensor> cu_seqlens,
                           int64_t cp_size, int64_t cp_rank) {
  auto nvte_qkv_format = static_cast<NVTE_QKV_Format>(qkv_format);

  STD_TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK(freqs.scalar_type() == ScalarType::Float,
                  "Dtype of the freqs tensor must be float");

  auto og_sizes = output_grads.sizes();
  std::vector<int64_t> og_shape(og_sizes.begin(), og_sizes.end());
  auto input_grads =
      allocateStableTensor(og_shape, output_grads.scalar_type(), output_grads.get_device_index());

  auto output_grads_cu = makeTransformerEngineTensor(output_grads);
  auto freqs_cu = makeTransformerEngineTensor(freqs);
  auto input_grads_cu = makeTransformerEngineTensor(input_grads);

  auto start_positions_cu = TensorWrapper();
  if (start_positions.has_value()) {
    start_positions_cu = makeTransformerEngineTensor(start_positions.value());
  }

  auto stream = getCurrentCUDAStreamRaw(output_grads.get_device_index());

  if (nvte_qkv_format == NVTE_QKV_Format::NVTE_THD) {
    STD_TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
    STD_TORCH_CHECK(cu_seqlens.has_value(), "expected cu_seqlens tensor");

    const int h = static_cast<int>(output_grads.size(1));
    const int d = static_cast<int>(output_grads.size(2));
    const int stride_t = static_cast<int>(output_grads.stride(0));
    const int stride_h = static_cast<int>(output_grads.stride(1));
    const int stride_d = static_cast<int>(output_grads.stride(2));
    const int b = static_cast<int>(cu_seqlens.value().size(0) - 1);
    const int max_s = static_cast<int>(freqs.size(0));
    const int d2 = static_cast<int>(freqs.size(3));

    auto cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());

    nvte_fused_rope_backward(output_grads_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                             start_positions_cu.data(), input_grads_cu.data(), nvte_qkv_format,
                             interleaved, static_cast<int>(cp_size), static_cast<int>(cp_rank),
                             max_s, b, h, d, d2, stride_t, 0, stride_h, stride_d, stream);

    return input_grads;
  }

  STD_TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  const bool is_sbhd = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD;
  const int s = static_cast<int>(is_sbhd ? output_grads.size(0) : output_grads.size(1));
  const int b = static_cast<int>(is_sbhd ? output_grads.size(1) : output_grads.size(0));
  const int h = static_cast<int>(output_grads.size(2));
  const int d = static_cast<int>(output_grads.size(3));
  const int stride_s = static_cast<int>(is_sbhd ? output_grads.stride(0) : output_grads.stride(1));
  const int stride_b = static_cast<int>(is_sbhd ? output_grads.stride(1) : output_grads.stride(0));
  const int stride_h = static_cast<int>(output_grads.stride(2));
  const int stride_d = static_cast<int>(output_grads.stride(3));
  const int d2 = static_cast<int>(freqs.size(3));

  auto cu_seqlens_cu = TensorWrapper();
  nvte_fused_rope_backward(output_grads_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                           start_positions_cu.data(), input_grads_cu.data(), nvte_qkv_format,
                           interleaved, static_cast<int>(cp_size), static_cast<int>(cp_rank), s, b,
                           h, d, d2, stride_s, stride_b, stride_h, stride_d, stream);

  return input_grads;
}

std::tuple<Tensor, Tensor, Tensor> fused_qkv_rope_forward(Tensor qkv_input, Tensor q_freqs,
                                                          Tensor k_freqs,
                                                          std::optional<Tensor> start_positions,
                                                          std::vector<int64_t> qkv_split_arg_list,
                                                          int64_t qkv_format, bool interleaved,
                                                          int64_t cp_size, int64_t cp_rank) {
  auto nvte_qkv_format = static_cast<NVTE_QKV_Format>(qkv_format);

  STD_TORCH_CHECK(q_freqs.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK(k_freqs.dim() == 4, "expected 4D tensor");
  STD_TORCH_CHECK(qkv_input.dim() == 4, "expected 4D input tensor");
  STD_TORCH_CHECK(qkv_input.is_contiguous(), "input tensor must be contiguous");

  auto sizes = qkv_input.sizes();
  auto dtype = qkv_input.scalar_type();
  auto device_idx = qkv_input.get_device_index();

  // q_out shape
  std::vector<int64_t> q_out_size = {sizes[0], sizes[1],
                                     sizes[2] * qkv_split_arg_list[0] / qkv_split_arg_list[1],
                                     qkv_split_arg_list[1]};
  auto q_out = allocateStableTensor(q_out_size, dtype, device_idx);

  std::vector<int64_t> k_out_size = {sizes[0], sizes[1], sizes[2], qkv_split_arg_list[1]};
  auto k_out = allocateStableTensor(k_out_size, dtype, device_idx);

  std::vector<int64_t> v_out_size = {sizes[0], sizes[1], sizes[2], qkv_split_arg_list[2]};
  auto v_out = allocateStableTensor(v_out_size, dtype, device_idx);

  auto qkv_cu = makeTransformerEngineTensor(qkv_input);
  auto q_freqs_cu = makeTransformerEngineTensor(q_freqs);
  auto k_freqs_cu = makeTransformerEngineTensor(k_freqs);
  auto q_out_cu = makeTransformerEngineTensor(q_out);
  auto k_out_cu = makeTransformerEngineTensor(k_out);
  auto v_out_cu = makeTransformerEngineTensor(v_out);

  auto start_positions_cu = TensorWrapper();
  if (start_positions.has_value()) {
    start_positions_cu = makeTransformerEngineTensor(start_positions.value());
  }

  const bool is_sbhd = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD;
  const int s = static_cast<int>(is_sbhd ? qkv_input.size(0) : qkv_input.size(1));
  const int b = static_cast<int>(is_sbhd ? qkv_input.size(1) : qkv_input.size(0));
  const int h = static_cast<int>(qkv_input.size(2));
  const int d = static_cast<int>(qkv_split_arg_list[2]);
  const int d2 = static_cast<int>(q_freqs.size(3));

  nvte_fused_qkv_rope_forward(
      qkv_cu.data(), q_freqs_cu.data(), k_freqs_cu.data(), start_positions_cu.data(),
      q_out_cu.data(), k_out_cu.data(), v_out_cu.data(), nvte_qkv_format, interleaved,
      static_cast<int>(cp_size), static_cast<int>(cp_rank), s, b, h, d, d2,
      static_cast<int>(qkv_split_arg_list[0]), static_cast<int>(qkv_split_arg_list[1]),
      static_cast<int>(qkv_split_arg_list[2]), getCurrentCUDAStreamRaw(device_idx));

  return std::make_tuple(q_out, k_out, v_out);
}

Tensor fused_qkv_rope_backward(Tensor q_grad_out, Tensor k_grad_out, Tensor v_grad_out,
                               Tensor q_freqs, Tensor k_freqs,
                               std::vector<int64_t> qkv_split_arg_list, int64_t qkv_format,
                               bool interleaved, int64_t cp_size, int64_t cp_rank) {
  auto nvte_qkv_format = static_cast<NVTE_QKV_Format>(qkv_format);
  auto dtype = q_grad_out.scalar_type();
  auto device_idx = q_grad_out.get_device_index();

  auto total_hd =
      (q_grad_out.size(2) + k_grad_out.size(2) + v_grad_out.size(2)) * q_grad_out.size(3);
  auto total_d = qkv_split_arg_list[0] + qkv_split_arg_list[1] + qkv_split_arg_list[2];
  std::vector<int64_t> qkv_grad_size = {q_grad_out.size(0), q_grad_out.size(1), total_hd / total_d,
                                        total_d};
  auto qkv_grad_input = allocateStableTensor(qkv_grad_size, dtype, device_idx);

  const bool is_sbhd = nvte_qkv_format == NVTE_QKV_Format::NVTE_SBHD;
  const int s = static_cast<int>(is_sbhd ? q_grad_out.size(0) : q_grad_out.size(1));
  const int b = static_cast<int>(is_sbhd ? q_grad_out.size(1) : q_grad_out.size(0));
  const int h = static_cast<int>(qkv_grad_size[2]);
  const int d = static_cast<int>(qkv_split_arg_list[2]);
  const int d2 = static_cast<int>(q_freqs.size(3));

  auto q_grad_out_cu = makeTransformerEngineTensor(q_grad_out);
  auto k_grad_out_cu = makeTransformerEngineTensor(k_grad_out);
  auto v_grad_out_cu = makeTransformerEngineTensor(v_grad_out);
  auto q_freqs_cu = makeTransformerEngineTensor(q_freqs);
  auto k_freqs_cu = makeTransformerEngineTensor(k_freqs);
  auto qkv_grad_cu = makeTransformerEngineTensor(qkv_grad_input);

  nvte_fused_qkv_rope_backward(
      q_grad_out_cu.data(), k_grad_out_cu.data(), v_grad_out_cu.data(), q_freqs_cu.data(),
      k_freqs_cu.data(), qkv_grad_cu.data(), nvte_qkv_format, interleaved,
      static_cast<int>(cp_size), static_cast<int>(cp_rank), s, b, h, d, d2,
      static_cast<int>(qkv_split_arg_list[0]), static_cast<int>(qkv_split_arg_list[1]),
      static_cast<int>(qkv_split_arg_list[2]), getCurrentCUDAStreamRaw(device_idx));

  return qkv_grad_input;
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  m.impl("fused_rope_forward", TORCH_BOX(fused_rope_forward));
  m.impl("fused_rope_backward", TORCH_BOX(fused_rope_backward));
  m.impl("fused_qkv_rope_forward", TORCH_BOX(fused_qkv_rope_forward));
  m.impl("fused_qkv_rope_backward", TORCH_BOX(fused_qkv_rope_backward));
}

}  // namespace transformer_engine::pytorch::stable
