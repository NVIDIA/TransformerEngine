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

at::Tensor fused_mla_rope_q_forward(const at::Tensor &q_input, const at::Tensor &cos,
                                    const at::Tensor &sin,
                                    const std::optional<at::Tensor> cu_seqlens,
                                    const int qk_head_dim, const int emb_dim, const int cp_size,
                                    const int cp_rank) {
  TORCH_CHECK(cos.scalar_type() == at::ScalarType::Float, "cos must be float32");
  TORCH_CHECK(sin.scalar_type() == at::ScalarType::Float, "sin must be float32");
  TORCH_CHECK(cos.is_contiguous(), "cos must be contiguous");
  TORCH_CHECK(sin.is_contiguous(), "sin must be contiguous");

  int max_seqlen = 0, batch_size = 0, nheads = 0, headdim = 0, total_seqlen = 0, s = 0, b = 0;
  at::Tensor q_flat;
  if (cu_seqlens.has_value()) {
    TORCH_CHECK(q_input.dim() == 3, "expected 3D tensor for THD format");
    total_seqlen = q_input.size(0);
    nheads = q_input.size(1);
    headdim = q_input.size(2);
    b = cu_seqlens.value().size(0) - 1;
    s = 0;
    q_flat = q_input.contiguous();
  } else {
    TORCH_CHECK(q_input.dim() == 4, "expected 4D tensor for SBHD format");
    max_seqlen = q_input.size(0);
    batch_size = q_input.size(1);
    nheads = q_input.size(2);
    headdim = q_input.size(3);
    q_flat = q_input.contiguous().view({max_seqlen * batch_size, nheads, headdim});
    total_seqlen = q_flat.size(0);
    s = max_seqlen;
    b = batch_size;
  }
  TORCH_CHECK(headdim == qk_head_dim + emb_dim, "headdim must equal qk_head_dim + emb_dim");

  auto q_out = at::empty_like(q_flat);
  auto q_in_cu = makeTransformerEngineTensor(q_flat);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  auto q_out_cu = makeTransformerEngineTensor(q_out);
  auto cu_seqlens_cu = TensorWrapper();
  if (cu_seqlens.has_value()) {
    cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());
  }

  nvte_fused_mla_rope_q_forward(q_in_cu.data(), cos_cu.data(), sin_cu.data(), q_out_cu.data(),
                                cu_seqlens_cu.data(), qk_head_dim, emb_dim, nheads, headdim,
                                total_seqlen, s, b, cp_size, cp_rank,
                                at::cuda::getCurrentCUDAStream());

  if (!cu_seqlens.has_value()) {
    q_out = q_out.view({max_seqlen, batch_size, nheads, headdim});
  }
  return q_out;
}

at::Tensor fused_mla_rope_q_backward(const at::Tensor &grad_output, const at::Tensor &cos,
                                     const at::Tensor &sin,
                                     const std::optional<at::Tensor> cu_seqlens,
                                     const int qk_head_dim, const int emb_dim, const int cp_size,
                                     const int cp_rank) {
  int max_seqlen = 0, batch_size = 0, nheads = 0, headdim = 0, total_seqlen = 0, s = 0, b = 0;
  at::Tensor grad_flat;
  if (cu_seqlens.has_value()) {
    total_seqlen = grad_output.size(0);
    nheads = grad_output.size(1);
    headdim = grad_output.size(2);
    b = cu_seqlens.value().size(0) - 1;
    s = 0;
    grad_flat = grad_output.contiguous();
  } else {
    max_seqlen = grad_output.size(0);
    batch_size = grad_output.size(1);
    nheads = grad_output.size(2);
    headdim = grad_output.size(3);
    grad_flat = grad_output.contiguous().view({max_seqlen * batch_size, nheads, headdim});
    total_seqlen = grad_flat.size(0);
    s = max_seqlen;
    b = batch_size;
  }

  auto grad_in = at::empty_like(grad_flat);
  auto grad_out_cu = makeTransformerEngineTensor(grad_flat);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  auto grad_in_cu = makeTransformerEngineTensor(grad_in);
  auto cu_seqlens_cu = TensorWrapper();
  if (cu_seqlens.has_value()) {
    cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());
  }

  nvte_fused_mla_rope_q_backward(grad_out_cu.data(), cos_cu.data(), sin_cu.data(),
                                 grad_in_cu.data(), cu_seqlens_cu.data(), qk_head_dim, emb_dim,
                                 nheads, headdim, total_seqlen, s, b, cp_size, cp_rank,
                                 at::cuda::getCurrentCUDAStream());

  if (!cu_seqlens.has_value()) {
    grad_in = grad_in.view({max_seqlen, batch_size, nheads, headdim});
  }
  return grad_in;
}

std::tuple<at::Tensor, at::Tensor> fused_mla_rope_kv_forward(
    const at::Tensor &kv_input, const at::Tensor &k_pos_emb, const at::Tensor &cos,
    const at::Tensor &sin, const std::optional<at::Tensor> cu_seqlens, const int emb_dim,
    const int k_dim, const int v_dim, const int cp_size, const int cp_rank) {
  TORCH_CHECK(cos.scalar_type() == at::ScalarType::Float, "cos must be float32");
  TORCH_CHECK(sin.scalar_type() == at::ScalarType::Float, "sin must be float32");
  TORCH_CHECK(cos.is_contiguous(), "cos must be contiguous");
  TORCH_CHECK(sin.is_contiguous(), "sin must be contiguous");
  TORCH_CHECK(kv_input.size(-1) == k_dim + v_dim, "last dim of kv must be k_dim + v_dim");

  int max_seqlen = 0, batch_size = 0, nheads = 0, total_seqlen = 0, s = 0, b_val = 0;
  at::Tensor kv_flat, emb_flat;
  if (cu_seqlens.has_value()) {
    TORCH_CHECK(kv_input.dim() == 3, "expected 3D tensor for THD format");
    total_seqlen = kv_input.size(0);
    nheads = kv_input.size(1);
    b_val = cu_seqlens.value().size(0) - 1;
    s = 0;
    kv_flat = kv_input.contiguous();
    emb_flat = k_pos_emb.contiguous().view({total_seqlen, emb_dim});
  } else {
    TORCH_CHECK(kv_input.dim() == 4, "expected 4D tensor for SBHD format");
    max_seqlen = kv_input.size(0);
    batch_size = kv_input.size(1);
    nheads = kv_input.size(2);
    kv_flat = kv_input.contiguous().view({max_seqlen * batch_size, nheads, k_dim + v_dim});
    emb_flat = k_pos_emb.contiguous().view({max_seqlen * batch_size, emb_dim});
    total_seqlen = kv_flat.size(0);
    s = max_seqlen;
    b_val = batch_size;
  }

  auto opts = at::TensorOptions().dtype(kv_input.scalar_type()).device(kv_input.device());
  auto o_key = at::empty({total_seqlen, nheads, k_dim + emb_dim}, opts);
  auto o_value = at::empty({total_seqlen, nheads, v_dim}, opts);

  auto kv_cu = makeTransformerEngineTensor(kv_flat);
  auto emb_cu = makeTransformerEngineTensor(emb_flat);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  auto okey_cu = makeTransformerEngineTensor(o_key);
  auto oval_cu = makeTransformerEngineTensor(o_value);
  auto cu_seqlens_cu = TensorWrapper();
  if (cu_seqlens.has_value()) {
    cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());
  }

  nvte_fused_mla_rope_kv_forward(kv_cu.data(), emb_cu.data(), cos_cu.data(), sin_cu.data(),
                                 okey_cu.data(), oval_cu.data(), cu_seqlens_cu.data(), emb_dim,
                                 k_dim, v_dim, nheads, total_seqlen, s, b_val, cp_size, cp_rank,
                                 at::cuda::getCurrentCUDAStream());

  if (!cu_seqlens.has_value()) {
    o_key = o_key.view({max_seqlen, batch_size, nheads, k_dim + emb_dim});
    o_value = o_value.view({max_seqlen, batch_size, nheads, v_dim});
  }
  return std::make_tuple(o_key, o_value);
}

std::tuple<at::Tensor, at::Tensor> fused_mla_rope_kv_backward(
    const at::Tensor &dk, const at::Tensor &dv, const at::Tensor &cos, const at::Tensor &sin,
    const std::optional<at::Tensor> cu_seqlens, const int emb_dim, const int k_dim,
    const int v_dim, const int cp_size, const int cp_rank) {
  int max_seqlen = 0, batch_size = 0, nheads = 0, total_seqlen = 0, s = 0, b_val = 0;
  at::Tensor dk_flat, dv_flat;
  if (cu_seqlens.has_value()) {
    total_seqlen = dk.size(0);
    nheads = dk.size(1);
    b_val = cu_seqlens.value().size(0) - 1;
    s = 0;
    dk_flat = dk.contiguous();
    dv_flat = dv.contiguous();
  } else {
    max_seqlen = dk.size(0);
    batch_size = dk.size(1);
    nheads = dk.size(2);
    dk_flat = dk.contiguous().view({max_seqlen * batch_size, nheads, k_dim + emb_dim});
    dv_flat = dv.contiguous().view({max_seqlen * batch_size, nheads, v_dim});
    total_seqlen = dk_flat.size(0);
    s = max_seqlen;
    b_val = batch_size;
  }

  auto opts = at::TensorOptions().dtype(dk.scalar_type()).device(dk.device());
  auto d_kv = at::empty({total_seqlen, nheads, k_dim + v_dim}, opts);
  auto d_emb = at::empty({total_seqlen, emb_dim}, opts);

  auto dk_cu = makeTransformerEngineTensor(dk_flat);
  auto dv_cu = makeTransformerEngineTensor(dv_flat);
  auto cos_cu = makeTransformerEngineTensor(cos);
  auto sin_cu = makeTransformerEngineTensor(sin);
  auto dkv_cu = makeTransformerEngineTensor(d_kv);
  auto demb_cu = makeTransformerEngineTensor(d_emb);
  auto cu_seqlens_cu = TensorWrapper();
  if (cu_seqlens.has_value()) {
    cu_seqlens_cu = makeTransformerEngineTensor(cu_seqlens.value());
  }

  nvte_fused_mla_rope_kv_backward(dk_cu.data(), dv_cu.data(), cos_cu.data(), sin_cu.data(),
                                  dkv_cu.data(), demb_cu.data(), cu_seqlens_cu.data(), emb_dim,
                                  k_dim, v_dim, nheads, total_seqlen, s, b_val, cp_size, cp_rank,
                                  at::cuda::getCurrentCUDAStream());

  if (!cu_seqlens.has_value()) {
    d_kv = d_kv.view({max_seqlen, batch_size, nheads, k_dim + v_dim});
    d_emb = d_emb.view({max_seqlen, batch_size, 1, emb_dim});
  } else {
    d_emb = d_emb.view({total_seqlen, 1, emb_dim});
  }
  return std::make_tuple(d_kv, d_emb);
}

}  // namespace transformer_engine::pytorch
