/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common.h"

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

at::Tensor mla_rope_q_forward(const at::Tensor &q, const at::Tensor &cos, const at::Tensor &sin,
                               const int qk_head_dim,
                               const std::optional<at::Tensor> cu_seqlens,
                               const int cp_size, const int cp_rank,
                               const NVTE_QKV_Format qkv_format) {
  TORCH_CHECK(cos.dim() == 4 && cos.size(1) == 1 && cos.size(2) == 1,
              "cos must be [max_s, 1, 1, emb_dim]");
  TORCH_CHECK(cos.scalar_type() == at::ScalarType::Float, "cos must be float32");
  TORCH_CHECK(sin.scalar_type() == at::ScalarType::Float, "sin must be float32");
  const int emb_dim = cos.size(3);
  auto act_options = at::TensorOptions().dtype(q.scalar_type()).device(q.device());
  auto output = at::empty(q.sizes(), act_options);
  auto q_cu = makeTransformerEngineTensor(q);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  auto output_cu = makeTransformerEngineTensor(output);
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    TORCH_CHECK(q.dim() == 3, "expected 3D tensor for THD");
    TORCH_CHECK(cu_seqlens.has_value(), "cu_seqlens required for THD");
    const int h = q.size(1);
    const int max_s = cos.size(0);
    const int b = cu_seqlens.value().size(0) - 1;
    auto cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());
    nvte_mla_rope_q_forward(q_cu.data(), cu_seqlens_cu.data(), cos_cu.data(), sin_cu.data(),
                            output_cu.data(), qkv_format, cp_size, cp_rank,
                            max_s, b, h, qk_head_dim, emb_dim,
                            q.stride(0), 0, q.stride(1),
                            at::cuda::getCurrentCUDAStream());
  } else {
    TORCH_CHECK(q.dim() == 4, "expected 4D tensor");
    const bool is_sbhd = qkv_format == NVTE_QKV_Format::NVTE_SBHD;
    const int s = is_sbhd ? q.size(0) : q.size(1);
    const int b = is_sbhd ? q.size(1) : q.size(0);
    const int h = q.size(2);
    auto cu_seqlens_cu = TensorWrapper();
    nvte_mla_rope_q_forward(q_cu.data(), cu_seqlens_cu.data(), cos_cu.data(), sin_cu.data(),
                            output_cu.data(), qkv_format, cp_size, cp_rank,
                            s, b, h, qk_head_dim, emb_dim,
                            is_sbhd ? q.stride(0) : q.stride(1),
                            is_sbhd ? q.stride(1) : q.stride(0),
                            q.stride(2), at::cuda::getCurrentCUDAStream());
  }
  return output;
}

at::Tensor mla_rope_q_backward(const at::Tensor &grad, const at::Tensor &cos, const at::Tensor &sin,
                                const int qk_head_dim,
                                const std::optional<at::Tensor> cu_seqlens,
                                const int cp_size, const int cp_rank,
                                const NVTE_QKV_Format qkv_format) {
  const int emb_dim = cos.size(3);
  auto act_options = at::TensorOptions().dtype(grad.scalar_type()).device(grad.device());
  auto dst = at::empty(grad.sizes(), act_options);
  auto grad_cu = makeTransformerEngineTensor(grad);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  auto dst_cu = makeTransformerEngineTensor(dst);
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    TORCH_CHECK(cu_seqlens.has_value(), "cu_seqlens required for THD");
    const int h = grad.size(1);
    const int max_s = cos.size(0);
    const int b = cu_seqlens.value().size(0) - 1;
    auto cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());
    nvte_mla_rope_q_backward(grad_cu.data(), cu_seqlens_cu.data(), cos_cu.data(), sin_cu.data(),
                             dst_cu.data(), qkv_format, cp_size, cp_rank,
                             max_s, b, h, qk_head_dim, emb_dim,
                             grad.stride(0), 0, grad.stride(1),
                             at::cuda::getCurrentCUDAStream());
  } else {
    const bool is_sbhd = qkv_format == NVTE_QKV_Format::NVTE_SBHD;
    const int s = is_sbhd ? grad.size(0) : grad.size(1);
    const int b = is_sbhd ? grad.size(1) : grad.size(0);
    const int h = grad.size(2);
    auto cu_seqlens_cu = TensorWrapper();
    nvte_mla_rope_q_backward(grad_cu.data(), cu_seqlens_cu.data(), cos_cu.data(), sin_cu.data(),
                             dst_cu.data(), qkv_format, cp_size, cp_rank,
                             s, b, h, qk_head_dim, emb_dim,
                             is_sbhd ? grad.stride(0) : grad.stride(1),
                             is_sbhd ? grad.stride(1) : grad.stride(0),
                             grad.stride(2), at::cuda::getCurrentCUDAStream());
  }
  return dst;
}

std::tuple<at::Tensor, at::Tensor> mla_rope_kv_forward(
    const at::Tensor &kv, const at::Tensor &k_pos_emb,
    const at::Tensor &cos, const at::Tensor &sin,
    const int k_dim, const int v_dim,
    const std::optional<at::Tensor> cu_seqlens,
    const int cp_size, const int cp_rank,
    const NVTE_QKV_Format qkv_format) {
  TORCH_CHECK(cos.scalar_type() == at::ScalarType::Float, "cos must be float32");
  TORCH_CHECK(sin.scalar_type() == at::ScalarType::Float, "sin must be float32");
  const int emb_dim = cos.size(3);
  auto act_options = at::TensorOptions().dtype(kv.scalar_type()).device(kv.device());
  auto kv_cu = makeTransformerEngineTensor(kv);
  auto emb_cu = makeTransformerEngineTensor(k_pos_emb);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    TORCH_CHECK(kv.dim() == 3, "expected 3D tensor for THD");
    TORCH_CHECK(cu_seqlens.has_value(), "cu_seqlens required for THD");
    const int h = kv.size(1);
    const int max_s = cos.size(0);
    const int b = cu_seqlens.value().size(0) - 1;
    auto key_out = at::empty({kv.size(0), h, k_dim + emb_dim}, act_options);
    auto val_out = at::empty({kv.size(0), h, v_dim}, act_options);
    auto key_cu = makeTransformerEngineTensor(key_out);
    auto val_cu = makeTransformerEngineTensor(val_out);
    auto cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());
    // k_pos_emb: [total_t, 1, emb_dim], stride_t = emb_dim, stride_b = 0
    nvte_mla_rope_kv_forward(kv_cu.data(), emb_cu.data(), cu_seqlens_cu.data(),
                             cos_cu.data(), sin_cu.data(), key_cu.data(), val_cu.data(),
                             qkv_format, cp_size, cp_rank, max_s, b, h, k_dim, v_dim, emb_dim,
                             kv.stride(0), 0, kv.stride(1),
                             k_pos_emb.stride(0), 0,
                             at::cuda::getCurrentCUDAStream());
    return std::make_tuple(key_out, val_out);
  }
  TORCH_CHECK(kv.dim() == 4, "expected 4D tensor");
  const bool is_sbhd = qkv_format == NVTE_QKV_Format::NVTE_SBHD;
  const int s = is_sbhd ? kv.size(0) : kv.size(1);
  const int b = is_sbhd ? kv.size(1) : kv.size(0);
  const int h = kv.size(2);
  auto key_sizes = kv.sizes().vec();
  key_sizes[3] = k_dim + emb_dim;
  auto val_sizes = kv.sizes().vec();
  val_sizes[3] = v_dim;
  auto key_out = at::empty(key_sizes, act_options);
  auto val_out = at::empty(val_sizes, act_options);
  auto key_cu = makeTransformerEngineTensor(key_out);
  auto val_cu = makeTransformerEngineTensor(val_out);
  auto cu_seqlens_cu = TensorWrapper();
  // k_pos_emb: [s, b, 1, emb_dim] (SBHD) or [b, s, 1, emb_dim] (BSHD)
  nvte_mla_rope_kv_forward(kv_cu.data(), emb_cu.data(), cu_seqlens_cu.data(),
                           cos_cu.data(), sin_cu.data(), key_cu.data(), val_cu.data(),
                           qkv_format, cp_size, cp_rank, s, b, h, k_dim, v_dim, emb_dim,
                           is_sbhd ? kv.stride(0) : kv.stride(1),
                           is_sbhd ? kv.stride(1) : kv.stride(0),
                           kv.stride(2),
                           is_sbhd ? k_pos_emb.stride(0) : k_pos_emb.stride(1),
                           is_sbhd ? k_pos_emb.stride(1) : k_pos_emb.stride(0),
                           at::cuda::getCurrentCUDAStream());
  return std::make_tuple(key_out, val_out);
}

std::tuple<at::Tensor, at::Tensor> mla_rope_kv_backward(
    const at::Tensor &dk, const at::Tensor &dv,
    const at::Tensor &cos, const at::Tensor &sin,
    const int k_dim, const int v_dim,
    const std::optional<at::Tensor> cu_seqlens,
    const int cp_size, const int cp_rank,
    const NVTE_QKV_Format qkv_format) {
  const int emb_dim = cos.size(3);
  auto act_options = at::TensorOptions().dtype(dk.scalar_type()).device(dk.device());
  auto dk_cu = makeTransformerEngineTensor(dk);
  auto dv_cu = makeTransformerEngineTensor(dv);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    TORCH_CHECK(cu_seqlens.has_value(), "cu_seqlens required for THD");
    const int h = dk.size(1);
    const int total_t = dk.size(0);
    const int max_s = cos.size(0);
    const int b = cu_seqlens.value().size(0) - 1;
    auto d_kv = at::empty({total_t, h, k_dim + v_dim}, act_options);
    auto d_emb = at::empty({total_t, 1, emb_dim}, act_options);
    auto d_kv_cu = makeTransformerEngineTensor(d_kv);
    auto d_emb_cu = makeTransformerEngineTensor(d_emb);
    auto cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());
    nvte_mla_rope_kv_backward(dk_cu.data(), dv_cu.data(), cu_seqlens_cu.data(),
                              cos_cu.data(), sin_cu.data(), d_kv_cu.data(), d_emb_cu.data(),
                              qkv_format, cp_size, cp_rank, max_s, b, h, k_dim, v_dim, emb_dim,
                              dk.stride(0), 0, dk.stride(1),
                              dv.stride(0), 0, dv.stride(1),
                              d_emb.stride(0), 0,
                              at::cuda::getCurrentCUDAStream());
    return std::make_tuple(d_kv, d_emb);
  }
  TORCH_CHECK(dk.dim() == 4, "expected 4D tensor");
  const bool is_sbhd = qkv_format == NVTE_QKV_Format::NVTE_SBHD;
  const int s = is_sbhd ? dk.size(0) : dk.size(1);
  const int b = is_sbhd ? dk.size(1) : dk.size(0);
  const int h = dk.size(2);
  auto d_kv_sizes = dk.sizes().vec();
  d_kv_sizes[3] = k_dim + v_dim;
  auto d_emb_sizes = dk.sizes().vec();
  d_emb_sizes[2] = 1;
  d_emb_sizes[3] = emb_dim;
  auto d_kv = at::empty(d_kv_sizes, act_options);
  auto d_emb = at::empty(d_emb_sizes, act_options);
  auto d_kv_cu = makeTransformerEngineTensor(d_kv);
  auto d_emb_cu = makeTransformerEngineTensor(d_emb);
  auto cu_seqlens_cu = TensorWrapper();
  nvte_mla_rope_kv_backward(dk_cu.data(), dv_cu.data(), cu_seqlens_cu.data(),
                            cos_cu.data(), sin_cu.data(), d_kv_cu.data(), d_emb_cu.data(),
                            qkv_format, cp_size, cp_rank, s, b, h, k_dim, v_dim, emb_dim,
                            is_sbhd ? dk.stride(0) : dk.stride(1),
                            is_sbhd ? dk.stride(1) : dk.stride(0),
                            dk.stride(2),
                            is_sbhd ? dv.stride(0) : dv.stride(1),
                            is_sbhd ? dv.stride(1) : dv.stride(0),
                            dv.stride(2),
                            is_sbhd ? d_emb.stride(0) : d_emb.stride(1),
                            is_sbhd ? d_emb.stride(1) : d_emb.stride(0),
                            at::cuda::getCurrentCUDAStream());
  return std::make_tuple(d_kv, d_emb);
}

}  // namespace transformer_engine::pytorch
