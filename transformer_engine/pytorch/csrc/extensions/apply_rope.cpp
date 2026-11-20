/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

at::Tensor fused_rope_forward(const at::Tensor &input, const at::Tensor &freqs,
                              const std::optional<at::Tensor> start_positions,
                              const NVTE_QKV_Format qkv_format, const bool interleaved,
                              const std::optional<at::Tensor> cu_seqlens, const int cp_size,
                              const int cp_rank) {
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

  auto start_positions_cu = TensorWrapper();  // empty start_positions tensor
  if (start_positions) {
    start_positions_cu = makeTransformerEngineTensor(start_positions.value());
    TORCH_CHECK(start_positions_cu.ndim() == 1, "expected 1D tensor");
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

  auto cu_seqlens_cu = TensorWrapper();  // empty cu_seqlens tensor
  nvte_fused_rope_forward(input_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                          start_positions_cu.data(), output_cu.data(), qkv_format, interleaved,
                          cp_size, cp_rank, s, b, h, d, d2, stride_s, stride_b, stride_h, stride_d,
                          at::cuda::getCurrentCUDAStream());

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_qkv_rope_forward(
    const at::Tensor &qkv_input, const at::Tensor &q_freqs, const at::Tensor &k_freqs,
    const std::optional<at::Tensor> start_positions, const std::vector<int> &qkv_split_arg_list,
    const NVTE_QKV_Format qkv_format, const bool interleaved, const int cp_size,
    const int cp_rank) {
  TORCH_CHECK(q_freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(q_freqs.size(1) == 1 && q_freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(q_freqs.scalar_type() == at::ScalarType::Float,
              "Dtype of the freqs tensor must be float");
  TORCH_CHECK(k_freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(k_freqs.size(1) == 1 && k_freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(k_freqs.scalar_type() == at::ScalarType::Float,
              "Dtype of the freqs tensor must be float");
  // output
  auto act_options = at::TensorOptions().dtype(qkv_input.scalar_type()).device(qkv_input.device());
  auto q_out_size = qkv_input.sizes().vec();
  q_out_size[2] = q_out_size[2] * qkv_split_arg_list[0] / qkv_split_arg_list[1];
  q_out_size[3] = qkv_split_arg_list[1];
  auto q_out = at::empty(q_out_size, act_options);
  auto k_out_size = qkv_input.sizes().vec();
  k_out_size[3] = qkv_split_arg_list[1];
  auto k_out = at::empty(k_out_size, act_options);
  auto v_out_size = qkv_input.sizes().vec();
  v_out_size[3] = qkv_split_arg_list[2];
  auto v_out = at::empty(v_out_size, act_options);

  auto qkv_cu = makeTransformerEngineTensor(qkv_input);
  auto q_freqs_cu = makeTransformerEngineTensor(q_freqs);
  auto k_freqs_cu = makeTransformerEngineTensor(k_freqs);
  auto q_out_cu = makeTransformerEngineTensor(q_out);
  auto k_out_cu = makeTransformerEngineTensor(k_out);
  auto v_out_cu = makeTransformerEngineTensor(v_out);

  auto start_positions_cu = TensorWrapper();  // empty cu_seqlens tensor
  if (start_positions) {
    start_positions_cu = makeTransformerEngineTensor(start_positions.value());
  }

  TORCH_CHECK(qkv_input.dim() == 4, "expected 4D input tensor");
  TORCH_CHECK(qkv_input.is_contiguous(), "input tensor must be contiguous");

  const bool is_sbhd = qkv_format == NVTE_QKV_Format::NVTE_SBHD;
  const int s = is_sbhd ? qkv_input.size(0) : qkv_input.size(1);
  const int b = is_sbhd ? qkv_input.size(1) : qkv_input.size(0);
  const int h = qkv_input.size(2);
  const int d = qkv_split_arg_list[2];
  const int d2 = q_freqs.size(3);

  nvte_fused_qkv_rope_forward(qkv_cu.data(), q_freqs_cu.data(), k_freqs_cu.data(),
                              start_positions_cu.data(), q_out_cu.data(), k_out_cu.data(),
                              v_out_cu.data(), qkv_format, interleaved, cp_size, cp_rank, s, b, h,
                              d, d2, qkv_split_arg_list[0], qkv_split_arg_list[1],
                              qkv_split_arg_list[2], at::cuda::getCurrentCUDAStream());

  return std::make_tuple(q_out, k_out, v_out);
}

at::Tensor fused_rope_backward(const at::Tensor &output_grads, const at::Tensor &freqs,
                               const std::optional<at::Tensor> start_positions,
                               const NVTE_QKV_Format qkv_format, const bool interleaved,
                               const std::optional<at::Tensor> cu_seqlens, const int cp_size,
                               const int cp_rank) {
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

  auto start_positions_cu = TensorWrapper();  // empty start_positions tensor
  if (start_positions) {
    start_positions_cu = makeTransformerEngineTensor(start_positions.value());
    TORCH_CHECK(start_positions_cu.ndim() == 1, "expected 1D tensor");
  }

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
                             start_positions_cu.data(), input_grads_cu.data(), qkv_format,
                             interleaved, cp_size, cp_rank, max_s, b, h, d, d2, stride_t,
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

  auto cu_seqlens_cu = TensorWrapper();  // empty cu_seqlens tensor
  nvte_fused_rope_backward(output_grads_cu.data(), cu_seqlens_cu.data(), freqs_cu.data(),
                           start_positions_cu.data(), input_grads_cu.data(), qkv_format,
                           interleaved, cp_size, cp_rank, s, b, h, d, d2, stride_s, stride_b,
                           stride_h, stride_d, at::cuda::getCurrentCUDAStream());

  return input_grads;
}

at::Tensor fused_qkv_rope_backward(const at::Tensor &q_grad_out, const at::Tensor &k_grad_out,
                                   const at::Tensor &v_grad_out, const at::Tensor &q_freqs,
                                   const at::Tensor &k_freqs,
                                   const std::vector<int> &qkv_split_arg_list,
                                   const NVTE_QKV_Format qkv_format, const bool interleaved,
                                   const int cp_size, const int cp_rank) {
  auto act_options =
      at::TensorOptions().dtype(q_grad_out.scalar_type()).device(q_grad_out.device());
  auto qkv_grad_size = q_grad_out.sizes().vec();
  auto total_hd =
      (q_grad_out.size(2) + k_grad_out.size(2) + v_grad_out.size(2)) * q_grad_out.size(3);
  auto total_d = qkv_split_arg_list[0] + qkv_split_arg_list[1] + qkv_split_arg_list[2];
  qkv_grad_size[2] = total_hd / total_d;
  qkv_grad_size[3] = total_d;
  auto qkv_grad_input = at::empty(qkv_grad_size, act_options);
  const bool is_sbhd = qkv_format == NVTE_QKV_Format::NVTE_SBHD;
  const int s = is_sbhd ? q_grad_out.size(0) : q_grad_out.size(1);
  const int b = is_sbhd ? q_grad_out.size(1) : q_grad_out.size(0);
  const int h = qkv_grad_input.size(2);
  const int d = qkv_split_arg_list[2];
  const int d2 = q_freqs.size(3);

  auto q_grad_out_cu = makeTransformerEngineTensor(q_grad_out);
  auto k_grad_out_cu = makeTransformerEngineTensor(k_grad_out);
  auto v_grad_out_cu = makeTransformerEngineTensor(v_grad_out);
  auto q_freqs_cu = makeTransformerEngineTensor(q_freqs);
  auto k_freqs_cu = makeTransformerEngineTensor(k_freqs);
  auto qkv_grad_cu = makeTransformerEngineTensor(qkv_grad_input);

  nvte_fused_qkv_rope_backward(q_grad_out_cu.data(), k_grad_out_cu.data(), v_grad_out_cu.data(),
                               q_freqs_cu.data(), k_freqs_cu.data(), qkv_grad_cu.data(), qkv_format,
                               interleaved, cp_size, cp_rank, s, b, h, d, d2, qkv_split_arg_list[0],
                               qkv_split_arg_list[1], qkv_split_arg_list[2],
                               at::cuda::getCurrentCUDAStream());

  return qkv_grad_input;
}

std::tuple<py::object, py::object> fused_mla_kv_rope_mxfp8(
    const at::Tensor &kv, const at::Tensor &k_rope, const at::Tensor &cos, const at::Tensor &sin,
    const int64_t v_head_dim, py::handle key_quantizer, py::handle value_quantizer) {
  TORCH_CHECK(kv.is_cuda(), "expected kv on a CUDA device");
  const auto device = kv.device();
  TORCH_CHECK(k_rope.device() == device && cos.device() == device && sin.device() == device,
              "expected kv, k_rope, cos and sin on the same device");
  at::cuda::CUDAGuard device_guard(device);
  TORCH_CHECK(kv.dim() == 4, "expected kv of shape [s, b, h, k_dim + v_dim], got ", kv.sizes());
  const int64_t s = kv.size(0);
  const int64_t b = kv.size(1);
  const int64_t h = kv.size(2);
  const int64_t emb_dim = k_rope.size(-1);
  TORCH_CHECK(k_rope.dim() >= 3 && k_rope.size(0) == s && k_rope.size(1) == b &&
                  k_rope.numel() == s * b * emb_dim,
              "expected k_rope of shape [s, b, 1, emb_dim] or [s, b, emb_dim], got ",
              k_rope.sizes());
  TORCH_CHECK(v_head_dim > 0 && v_head_dim < kv.size(3), "invalid v_head_dim ", v_head_dim,
              " for kv of shape ", kv.sizes());
  const int64_t key_dim = kv.size(3) - v_head_dim + emb_dim;

  // k_rope is usually a slice of a wider projection output. Its rows are read in place
  // when they are a fixed, 16-byte aligned stride apart.
  at::Tensor k_rope_rows = k_rope;
  int64_t k_rope_stride = b > 1 ? k_rope_rows.stride(1) : k_rope_rows.stride(0);
  if (k_rope_rows.stride(-1) != 1 ||
      (b > 1 && k_rope_rows.stride(0) != b * k_rope_rows.stride(1)) || k_rope_stride < emb_dim ||
      k_rope_stride % 8 != 0 || reinterpret_cast<uintptr_t>(k_rope_rows.data_ptr()) % 16 != 0) {
    k_rope_rows = k_rope.contiguous();
    k_rope_stride = emb_dim;
  }
  const auto kv_c = kv.contiguous();
  const auto cos_c = cos.contiguous();
  const auto sin_c = sin.contiguous();
  auto kv_cu = makeTransformerEngineTensor(kv_c);
  auto k_rope_cu = makeTransformerEngineTensor(
      k_rope_rows.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(s * b), static_cast<size_t>(emb_dim)},
      GetTransformerEngineDType(k_rope_rows.scalar_type()));
  auto cos_cu = makeTransformerEngineTensor(cos_c);
  auto sin_cu = makeTransformerEngineTensor(sin_c);

  auto key_quantizer_cpp = convert_quantizer(key_quantizer);
  auto value_quantizer_cpp = convert_quantizer(value_quantizer);
  const auto *key_mxfp8 = dynamic_cast<MXFP8Quantizer *>(key_quantizer_cpp.get());
  const auto *value_mxfp8 = dynamic_cast<MXFP8Quantizer *>(value_quantizer_cpp.get());
  TORCH_CHECK(key_mxfp8 != nullptr && value_mxfp8 != nullptr,
              "fused_mla_kv_rope_mxfp8 requires MXFP8 quantizers");
  TORCH_CHECK(!key_mxfp8->with_2d_quantization && !value_mxfp8->with_2d_quantization,
              "fused_mla_kv_rope_mxfp8 does not support 2D MXFP8 quantization");
  const auto fake_dtype = GetTransformerEngineDType(kv.scalar_type());
  auto [key_cu, key_py] = key_quantizer_cpp->create_tensor(
      {static_cast<size_t>(s), static_cast<size_t>(b * h * key_dim)}, fake_dtype, device);
  auto [value_cu, value_py] = value_quantizer_cpp->create_tensor(
      {static_cast<size_t>(s), static_cast<size_t>(b * h * v_head_dim)}, fake_dtype, device);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_mla_kv_rope_mxfp8(kv_cu.data(), k_rope_cu.data(), static_cast<int>(k_rope_stride),
                                 cos_cu.data(), sin_cu.data(), key_cu.data(), value_cu.data(),
                                 at::cuda::getCurrentCUDAStream());
  });
  return {key_py, value_py};
}

namespace {

// Buffers of an MXFP8 tensor: row-wise data, row-wise scale_inv, column-wise data, column-wise
// scale_inv (uint8).
std::vector<at::Tensor> empty_mxfp8_buffers(const std::vector<int64_t> &data_shape,
                                            const std::vector<int64_t> &rowwise_scale_shape,
                                            const std::vector<int64_t> &columnwise_scale_shape,
                                            const at::Device &device) {
  const auto opts = at::TensorOptions().dtype(at::kByte).device(device);
  return {at::empty(data_shape, opts), at::empty(rowwise_scale_shape, opts),
          at::empty(data_shape, opts), at::empty(columnwise_scale_shape, opts)};
}

// TE view of MXFP8 buffers (empty_mxfp8_buffers) with GEMM-swizzled scaling factors.
TensorWrapper make_swizzled_mxfp8_tensor(std::vector<at::Tensor> &buffers) {
  auto shape_of = [](const at::Tensor &t) {
    std::vector<size_t> shape;
    for (const auto d : t.sizes()) shape.push_back(static_cast<size_t>(d));
    return shape;
  };
  TensorWrapper out(NVTE_MXFP8_1D_SCALING);
  out.set_rowwise_data(buffers[0].data_ptr(), DType::kFloat8E4M3, shape_of(buffers[0]));
  out.set_rowwise_scale_inv(buffers[1].data_ptr(), DType::kFloat8E8M0, shape_of(buffers[1]));
  out.set_columnwise_data(buffers[2].data_ptr(), DType::kFloat8E4M3, shape_of(buffers[2]));
  out.set_columnwise_scale_inv(buffers[3].data_ptr(), DType::kFloat8E8M0, shape_of(buffers[3]));
  out.set_with_gemm_swizzled_scales(true);
  return out;
}

// MXFP8 buffers of an attention input [s, b, h, d] with BHSD-permuted, GEMM-swizzled scaling
// factors (as mxfp8_transpose_swizzle leaves them).
std::vector<at::Tensor> empty_mxfp8_attn_buffers(const int64_t s, const int64_t b, const int64_t h,
                                                 const int64_t d, const at::Device &device) {
  const int64_t rows = b * h * s;
  return empty_mxfp8_buffers({s, b, h, d}, {rows, (d / 32 + 3) / 4 * 4},
                             {rows / 32, (d + 127) / 128 * 128}, device);
}

void check_mla_rope_inputs(const at::Tensor &x, const char *name,
                           const std::vector<at::Tensor> &others) {
  TORCH_CHECK(x.is_cuda(), "expected ", name, " on a CUDA device");
  for (const auto &t : others) {
    TORCH_CHECK(t.device() == x.device(), "expected all inputs on the device of ", name);
  }
  TORCH_CHECK(x.dim() == 4, "expected ", name, " of shape [s, b, h, d], got ", x.sizes());
}

}  // namespace

std::vector<at::Tensor> fused_mla_q_rope_mxfp8_attn(const at::Tensor &q, const at::Tensor &cos,
                                                    const at::Tensor &sin) {
  check_mla_rope_inputs(q, "q", {cos, sin});
  at::cuda::CUDAGuard device_guard(q.device());
  const auto q_c = q.contiguous();
  const auto cos_c = cos.contiguous();
  const auto sin_c = sin.contiguous();
  auto out = empty_mxfp8_attn_buffers(q.size(0), q.size(1), q.size(2), q.size(3), q.device());
  auto q_cu = makeTransformerEngineTensor(q_c);
  auto cos_cu = makeTransformerEngineTensor(cos_c);
  auto sin_cu = makeTransformerEngineTensor(sin_c);
  auto out_cu = make_swizzled_mxfp8_tensor(out);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_mla_q_rope_mxfp8_attn(q_cu.data(), cos_cu.data(), sin_cu.data(), out_cu.data(),
                                     at::cuda::getCurrentCUDAStream());
  });
  return out;
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> fused_mla_kv_rope_mxfp8_attn(
    const at::Tensor &kv, const at::Tensor &k_rope, const at::Tensor &cos, const at::Tensor &sin,
    const int64_t v_head_dim) {
  check_mla_rope_inputs(kv, "kv", {k_rope, cos, sin});
  at::cuda::CUDAGuard device_guard(kv.device());
  const int64_t s = kv.size(0), b = kv.size(1), h = kv.size(2);
  const int64_t emb_dim = k_rope.size(-1);
  TORCH_CHECK(k_rope.dim() >= 3 && k_rope.size(0) == s && k_rope.size(1) == b &&
                  k_rope.numel() == s * b * emb_dim,
              "expected k_rope of shape [s, b, 1, emb_dim] or [s, b, emb_dim], got ",
              k_rope.sizes());
  TORCH_CHECK(v_head_dim > 0 && v_head_dim < kv.size(3), "invalid v_head_dim ", v_head_dim,
              " for kv of shape ", kv.sizes());
  const int64_t key_dim = kv.size(3) - v_head_dim + emb_dim;
  // k_rope rows are read in place when they are a fixed, 16-byte aligned stride apart.
  at::Tensor k_rope_rows = k_rope;
  int64_t k_rope_stride = b > 1 ? k_rope_rows.stride(1) : k_rope_rows.stride(0);
  if (k_rope_rows.stride(-1) != 1 ||
      (b > 1 && k_rope_rows.stride(0) != b * k_rope_rows.stride(1)) || k_rope_stride < emb_dim ||
      k_rope_stride % 8 != 0 || reinterpret_cast<uintptr_t>(k_rope_rows.data_ptr()) % 16 != 0) {
    k_rope_rows = k_rope.contiguous();
    k_rope_stride = emb_dim;
  }
  const auto kv_c = kv.contiguous();
  const auto cos_c = cos.contiguous();
  const auto sin_c = sin.contiguous();
  auto key = empty_mxfp8_attn_buffers(s, b, h, key_dim, kv.device());
  auto value = empty_mxfp8_attn_buffers(s, b, h, v_head_dim, kv.device());
  auto kv_cu = makeTransformerEngineTensor(kv_c);
  auto k_rope_cu = makeTransformerEngineTensor(
      k_rope_rows.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(s * b), static_cast<size_t>(emb_dim)},
      GetTransformerEngineDType(k_rope_rows.scalar_type()));
  auto cos_cu = makeTransformerEngineTensor(cos_c);
  auto sin_cu = makeTransformerEngineTensor(sin_c);
  auto key_cu = make_swizzled_mxfp8_tensor(key);
  auto value_cu = make_swizzled_mxfp8_tensor(value);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_mla_kv_rope_mxfp8_attn(
        kv_cu.data(), k_rope_cu.data(), static_cast<int>(k_rope_stride), cos_cu.data(),
        sin_cu.data(), key_cu.data(), value_cu.data(), at::cuda::getCurrentCUDAStream());
  });
  return {key, value};
}

at::Tensor fused_mla_q_rope_backward_(at::Tensor dq, const at::Tensor &cos, const at::Tensor &sin) {
  check_mla_rope_inputs(dq, "dq", {cos, sin});
  TORCH_CHECK(dq.is_contiguous(), "expected a contiguous dq");
  at::cuda::CUDAGuard device_guard(dq.device());
  const auto cos_c = cos.contiguous();
  const auto sin_c = sin.contiguous();
  auto dq_cu = makeTransformerEngineTensor(dq);
  auto cos_cu = makeTransformerEngineTensor(cos_c);
  auto sin_cu = makeTransformerEngineTensor(sin_c);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_mla_q_rope_backward(dq_cu.data(), cos_cu.data(), sin_cu.data(),
                                   at::cuda::getCurrentCUDAStream());
  });
  return dq;
}

std::vector<at::Tensor> fused_mla_q_rope_backward_mxfp8(const at::Tensor &dq, const at::Tensor &cos,
                                                        const at::Tensor &sin) {
  check_mla_rope_inputs(dq, "dq", {cos, sin});
  at::cuda::CUDAGuard device_guard(dq.device());
  const auto dq_c = dq.contiguous();
  const auto cos_c = cos.contiguous();
  const auto sin_c = sin.contiguous();
  const int64_t tokens = dq.size(0) * dq.size(1), columns = dq.size(2) * dq.size(3);
  auto out = empty_mxfp8_buffers({tokens, columns}, {tokens, columns / 32}, {tokens / 32, columns},
                                 dq.device());
  auto dq_cu = makeTransformerEngineTensor(dq_c);
  auto cos_cu = makeTransformerEngineTensor(cos_c);
  auto sin_cu = makeTransformerEngineTensor(sin_c);
  auto out_cu = make_swizzled_mxfp8_tensor(out);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_mla_q_rope_backward_mxfp8(dq_cu.data(), cos_cu.data(), sin_cu.data(), out_cu.data(),
                                         at::cuda::getCurrentCUDAStream());
  });
  return out;
}

std::vector<at::Tensor> fused_mla_kv_split_backward_mxfp8(const at::Tensor &dk,
                                                          const at::Tensor &dv,
                                                          const int64_t k_dim) {
  check_mla_rope_inputs(dk, "dk", {dv});
  TORCH_CHECK(dv.dim() == 4, "expected dv of shape [s, b, h, d], got ", dv.sizes());
  at::cuda::CUDAGuard device_guard(dk.device());
  const auto dk_c = dk.contiguous();
  const auto dv_c = dv.contiguous();
  const int64_t tokens = dk.size(0) * dk.size(1);
  TORCH_CHECK(k_dim > 0 && k_dim < dk.size(3), "invalid k_dim ", k_dim, " for dk of shape ",
              dk.sizes());
  const int64_t columns = dk.size(2) * (k_dim + dv.size(3));
  auto out = empty_mxfp8_buffers({tokens, columns}, {tokens, columns / 32}, {tokens / 32, columns},
                                 dk.device());
  auto dk_cu = makeTransformerEngineTensor(dk_c);
  auto dv_cu = makeTransformerEngineTensor(dv_c);
  auto out_cu = make_swizzled_mxfp8_tensor(out);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_fused_mla_kv_split_backward_mxfp8(dk_cu.data(), dv_cu.data(), out_cu.data(),
                                           at::cuda::getCurrentCUDAStream());
  });
  return out;
}

}  // namespace transformer_engine::pytorch
