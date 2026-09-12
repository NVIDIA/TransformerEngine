/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

at::Tensor fa_prepare_fwd(at::Tensor qkvi) {
  NVTE_CHECK(qkvi.dim() == 4, "Expected 4-dim tensor.");
  NVTE_CHECK(qkvi.scalar_type() == at::ScalarType::Half ||
             qkvi.scalar_type() == at::ScalarType::BFloat16);
  NVTE_CHECK(qkvi.stride(3) == 1, "Wrong stride.");
  NVTE_CHECK(qkvi.stride(2) == 3 * qkvi.size(3), "Wrong stride.");
  NVTE_CHECK(qkvi.stride(1) == 3 * qkvi.size(3) * qkvi.size(2), "Wrong stride.");
  NVTE_CHECK(qkvi.stride(0) == 3 * qkvi.size(3) * qkvi.size(2) * qkvi.size(1), "Wrong stride.");

  // [s, b, n, h * 3] -> [3, b, s, n, h]
  std::vector<int64_t> shape = {3, qkvi.size(1), qkvi.size(0), qkvi.size(2), qkvi.size(3)};
  at::Tensor qkv = at::empty(shape, at::CUDA(qkvi.scalar_type()));

  auto te_qkvi = makeTransformerEngineTensor(qkvi);
  auto te_qkv = makeTransformerEngineTensor(qkv);

  nvte_prepare_flash_attn_fwd(te_qkvi.data(), te_qkv.data(), at::cuda::getCurrentCUDAStream());

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

  // 3 x [s, b, n, h] -> [b, s, n, 3 * h]
  std::vector<int64_t> shape = {q.size(1), q.size(0), q.size(2), 3 * q.size(3)};
  at::Tensor qkv = at::empty(shape, at::CUDA(q.scalar_type()));

  auto te_q = makeTransformerEngineTensor(q);
  auto te_k = makeTransformerEngineTensor(k);
  auto te_v = makeTransformerEngineTensor(v);
  auto te_qkv = makeTransformerEngineTensor(qkv);

  nvte_prepare_flash_attn_bwd(te_q.data(), te_k.data(), te_v.data(), te_qkv.data(),
                              at::cuda::getCurrentCUDAStream());

  return qkv;
}

std::vector<std::optional<at::Tensor>> multi_tensor_transpose_to_bhsd(
    std::vector<std::optional<at::Tensor>> inputs, const std::string &original_format,
    std::vector<std::optional<at::Tensor>> outputs) {
  NVTE_CHECK(original_format == "sbhd" || original_format == "bshd",
             "multi_tensor_transpose_to_bhsd: only BSHD/SBHD -> BHSD is currently supported. "
             "Got original_format=\"",
             original_format, "\".");
  const auto original_format_enum = (original_format == "sbhd") ? NVTE_SBHD : NVTE_BSHD;

  if (inputs.empty()) return {};

  const bool has_outputs = !outputs.empty();
  if (has_outputs) {
    NVTE_CHECK(outputs.size() == inputs.size(), "multi_tensor_transpose_to_bhsd: outputs.size() (",
               outputs.size(), ") != inputs.size() (", inputs.size(), ").");
  }

  std::vector<transformer_engine::TensorWrapper> te_ins, te_outs;
  std::vector<std::optional<at::Tensor>> result(inputs.size(), std::nullopt);

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].has_value()) continue;

    auto &input = inputs[i].value();
    NVTE_CHECK(input.is_cuda() && input.dim() == 4, "multi_tensor_transpose_to_bhsd: input ", i,
               " must be a 4D CUDA tensor.");
    input = input.contiguous();
    NVTE_CHECK(input.scalar_type() == at::ScalarType::Half ||
                   input.scalar_type() == at::ScalarType::BFloat16 ||
                   input.scalar_type() == at::ScalarType::Byte,
               "multi_tensor_transpose_to_bhsd: unsupported dtype at index ", i, ".");

    at::Tensor output;
    if (has_outputs && outputs[i].has_value()) {
      output = outputs[i].value();
    } else {
      int64_t B, S, H, D;
      if (original_format_enum == NVTE_SBHD) {
        S = input.size(0);
        B = input.size(1);
        H = input.size(2);
        D = input.size(3);
      } else {
        B = input.size(0);
        S = input.size(1);
        H = input.size(2);
        D = input.size(3);
      }
      output = at::empty({B, H, S, D}, input.options());
    }

    te_ins.push_back(makeTransformerEngineTensor(input));
    te_outs.push_back(makeTransformerEngineTensor(output));
    result[i] = output;
  }

  if (!te_ins.empty()) {
    std::vector<NVTETensor> nvte_ins(te_ins.size()), nvte_outs(te_outs.size());
    for (size_t j = 0; j < te_ins.size(); ++j) {
      nvte_ins[j] = te_ins[j].data();
      nvte_outs[j] = te_outs[j].data();
    }
    nvte_multi_tensor_transpose_to_bhsd(nvte_ins.data(), nvte_outs.data(), te_ins.size(),
                                        original_format_enum, at::cuda::getCurrentCUDAStream());
  }

  return result;
}

std::vector<at::Tensor> multi_tensor_pad_last_dim(std::vector<at::Tensor> inputs,
                                                  int64_t alignment) {
  const auto align = static_cast<size_t>(alignment);
  NVTE_CHECK(align > 0, "multi_tensor_pad_last_dim: alignment must be > 0.");
  NVTE_CHECK(!inputs.empty(), "multi_tensor_pad_last_dim: inputs must not be empty.");

  auto stream = at::cuda::getCurrentCUDAStream();
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());

  std::vector<size_t> kernel_indices;

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &input = inputs[i];

    NVTE_CHECK(input.dim() == 2, "multi_tensor_pad_last_dim: expected 2D input at index ", i,
               ", got ", input.dim(), "D.");
    NVTE_CHECK(input.is_cuda(), "multi_tensor_pad_last_dim: input must be a CUDA tensor at index ",
               i, ".");
    input = input.contiguous();

    const int64_t rows = input.size(0);
    const int64_t in_cols = input.size(1);
    const int64_t padded_cols =
        static_cast<int64_t>(DIVUP_TO_MULTIPLE(static_cast<size_t>(in_cols), align));

    if (in_cols == padded_cols) {
      outputs.push_back(input);
      continue;
    }

    at::Tensor output = at::empty({rows, padded_cols}, input.options());
    outputs.push_back(output);
    kernel_indices.push_back(outputs.size() - 1);
  }

  if (kernel_indices.empty()) return outputs;

  std::vector<transformer_engine::TensorWrapper> te_in_wrappers, te_out_wrappers;
  te_in_wrappers.reserve(kernel_indices.size());
  te_out_wrappers.reserve(kernel_indices.size());

  for (size_t idx : kernel_indices) {
    te_in_wrappers.push_back(makeTransformerEngineTensor(inputs[idx]));
    te_out_wrappers.push_back(makeTransformerEngineTensor(outputs[idx]));
  }

  std::vector<NVTETensor> nvte_inputs(te_in_wrappers.size());
  std::vector<NVTETensor> nvte_outputs(te_out_wrappers.size());
  for (size_t i = 0; i < te_in_wrappers.size(); ++i) {
    nvte_inputs[i] = te_in_wrappers[i].data();
    nvte_outputs[i] = te_out_wrappers[i].data();
  }

  nvte_multi_tensor_pad_last_dim(nvte_inputs.data(), nvte_outputs.data(), te_in_wrappers.size(),
                                 stream);

  return outputs;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Read the half of a THD tensor
 **************************************************************************************************/

at::Tensor thd_read_half_tensor(const at::Tensor &tensor, const at::Tensor &cu_seqlens,
                                int half_idx) {
  NVTE_CHECK(tensor.dim() == 3 || tensor.dim() == 4);
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);
  NVTE_CHECK(cu_seqlens.size(0) >= 2);

  // Shapes of q and dq are [t, h, d], so the dimension of "t" is 0
  // Shapes of kv and dkv are [2, t, h, d], so the dimension of "t" is 1
  int seq_dim = tensor.dim() == 3 ? 0 : 1;

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

  auto te_tensor = makeTransformerEngineTensor(tensor);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_half = makeTransformerEngineTensor(half);

  nvte_cp_thd_read_half_tensor(te_tensor.data(), te_cu_seqlens.data(), te_half.data(), half_idx,
                               at::cuda::getCurrentCUDAStream());

  return half;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: softmax_lse related operations
 **************************************************************************************************/

void thd_second_half_lse_correction(at::Tensor lse, const at::Tensor &lse_per_step,
                                    const at::Tensor &cu_seqlens, bool lse_packed) {
  NVTE_CHECK(lse.scalar_type() == at::ScalarType::Float);
  NVTE_CHECK(lse_per_step.scalar_type() == at::ScalarType::Float);
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  int batch, num_heads, lse_seqlen, second_half_lse_seqlen;

  if (lse_packed) {
    NVTE_CHECK(lse.dim() == 2);
    NVTE_CHECK(lse_per_step.dim() == 2);

    batch = cu_seqlens.size(0) - 1;
    num_heads = lse.size(0);
    lse_seqlen = lse.size(1);
    second_half_lse_seqlen = lse_per_step.size(1);

    NVTE_CHECK(lse_per_step.size(0) == num_heads);
    NVTE_CHECK(second_half_lse_seqlen >= lse_seqlen / 2);
  } else {
    NVTE_CHECK(lse.dim() == 3);
    NVTE_CHECK(lse_per_step.dim() == 3);

    batch = lse.size(0);
    num_heads = lse.size(1);
    lse_seqlen = lse.size(2);
    second_half_lse_seqlen = lse_per_step.size(2);

    NVTE_CHECK(lse_per_step.size(0) == batch);
    NVTE_CHECK(lse_per_step.size(1) == num_heads);
    NVTE_CHECK(second_half_lse_seqlen == lse_seqlen / 2);
    NVTE_CHECK(cu_seqlens.size(0) == batch + 1);
  }

  auto te_lse = makeTransformerEngineTensor(lse);
  auto te_lse_per_step = makeTransformerEngineTensor(lse_per_step);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);

  nvte_cp_thd_second_half_lse_correction(te_lse.data(), te_lse_per_step.data(),
                                         te_cu_seqlens.data(), lse_packed,
                                         at::cuda::getCurrentCUDAStream());
}

at::Tensor thd_read_second_half_lse(const at::Tensor &lse, const at::Tensor &cu_seqlens,
                                    bool lse_packed, int second_half_lse_seqlen) {
  NVTE_CHECK(lse.scalar_type() == at::ScalarType::Float);
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  int batch, num_heads, lse_seqlen;
  std::vector<int64_t> shape;

  if (lse_packed) {
    NVTE_CHECK(lse.dim() == 2);

    batch = cu_seqlens.size(0) - 1;
    num_heads = lse.size(0);
    lse_seqlen = lse.size(1);

    NVTE_CHECK(second_half_lse_seqlen >= lse_seqlen / 2);

    shape = {num_heads, second_half_lse_seqlen};
  } else {
    NVTE_CHECK(lse.dim() == 3);

    batch = lse.size(0);
    num_heads = lse.size(1);
    lse_seqlen = lse.size(2);

    NVTE_CHECK(cu_seqlens.size(0) == batch + 1);
    NVTE_CHECK(second_half_lse_seqlen == lse_seqlen / 2);

    shape = {batch, num_heads, second_half_lse_seqlen};
  }

  at::Tensor half_lse = at::zeros(shape, at::CUDA(lse.scalar_type()));

  auto te_lse = makeTransformerEngineTensor(lse);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_half_lse = makeTransformerEngineTensor(half_lse);

  nvte_cp_thd_read_second_half_lse(te_lse.data(), te_cu_seqlens.data(), te_half_lse.data(),
                                   lse_packed, second_half_lse_seqlen,
                                   at::cuda::getCurrentCUDAStream());

  return half_lse;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Out correction in forward
 **************************************************************************************************/

void thd_out_correction(at::Tensor out, const at::Tensor &out_per_step, const at::Tensor &lse,
                        const at::Tensor &lse_per_step, const at::Tensor &cu_seqlens,
                        bool only_second_half, bool lse_packed) {
  auto te_out = makeTransformerEngineTensor(out);
  auto te_out_per_step = makeTransformerEngineTensor(out_per_step);
  auto te_lse = makeTransformerEngineTensor(lse);
  auto te_lse_per_step = makeTransformerEngineTensor(lse_per_step);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  nvte_cp_thd_out_correction(te_out.data(), te_out_per_step.data(), te_lse.data(),
                             te_lse_per_step.data(), te_cu_seqlens.data(), only_second_half,
                             lse_packed, at::cuda::getCurrentCUDAStream());
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Gradients correction in backward
 **************************************************************************************************/

void thd_grad_correction(at::Tensor grad, const at::Tensor &grad_per_step,
                         const at::Tensor &cu_seqlens, const std::string &first_half,
                         const std::string &second_half) {
  auto te_grad = makeTransformerEngineTensor(grad);
  auto te_grad_per_step = makeTransformerEngineTensor(grad_per_step);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  nvte_cp_thd_grad_correction(te_grad.data(), te_grad_per_step.data(), te_cu_seqlens.data(),
                              first_half.data(), second_half.data(),
                              at::cuda::getCurrentCUDAStream());
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Generate partitioned indices for input tokens
 **************************************************************************************************/

at::Tensor thd_get_partitioned_indices(const at::Tensor &cu_seqlens, int total_tokens,
                                       int world_size, int rank) {
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);
  NVTE_CHECK(cu_seqlens.size(0) >= 2);
  NVTE_CHECK(rank >= 0 && rank < world_size);
  NVTE_CHECK(world_size > 0);
  NVTE_CHECK(total_tokens > 0 && total_tokens % (world_size * 2) == 0);

  std::vector<int64_t> shape = {total_tokens / world_size};
  at::Tensor output = at::empty(shape, at::CUDA(at::ScalarType::Int));

  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_output = makeTransformerEngineTensor(output);

  nvte_cp_thd_get_partitioned_indices(te_cu_seqlens.data(), te_output.data(), total_tokens,
                                      world_size, rank, at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor thd_reorder_between_sequence_and_cp_rank_order(const at::Tensor &inp,
                                                          const at::Tensor &cu_seqlens, int cp_size,
                                                          bool cp_rank_to_sequence_order,
                                                          int total_tokens) {
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1);
  NVTE_CHECK(cu_seqlens.size(0) >= 2);
  NVTE_CHECK(cp_size > 0);
  NVTE_CHECK(total_tokens > 0 && total_tokens % (cp_size * 2) == 0);
  NVTE_CHECK(inp.dim() >= 1 && inp.size(0) == total_tokens);

  auto inp_c = inp.contiguous();
  at::Tensor out = at::empty_like(inp_c);

  auto te_inp = makeTransformerEngineTensor(inp_c);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_out = makeTransformerEngineTensor(out);

  if (cp_rank_to_sequence_order) {
    nvte_thd_cp_rank_order_to_sequence_order(te_inp.data(), te_cu_seqlens.data(), te_out.data(),
                                             cp_size, total_tokens,
                                             at::cuda::getCurrentCUDAStream());
  } else {
    nvte_thd_sequence_order_to_cp_rank_order(te_inp.data(), te_cu_seqlens.data(), te_out.data(),
                                             cp_size, total_tokens,
                                             at::cuda::getCurrentCUDAStream());
  }

  return out;
}

at::Tensor thd_sequence_order_to_cp_rank_order(const at::Tensor &inp, const at::Tensor &cu_seqlens,
                                               int cp_size, int total_tokens) {
  return thd_reorder_between_sequence_and_cp_rank_order(inp, cu_seqlens, cp_size, false,
                                                        total_tokens);
}

at::Tensor thd_cp_rank_order_to_sequence_order(const at::Tensor &inp, const at::Tensor &cu_seqlens,
                                               int cp_size, int total_tokens) {
  return thd_reorder_between_sequence_and_cp_rank_order(inp, cu_seqlens, cp_size, true,
                                                        total_tokens);
}

void thd_copy_valid_tokens_from_per_split_to_rank_local(at::Tensor out, const at::Tensor &inp,
                                                        const at::Tensor &cu_seqlens_padded,
                                                        const at::Tensor &cu_seqlens) {
  NVTE_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens_padded.scalar_type() == at::ScalarType::Int);
  NVTE_CHECK(cu_seqlens.dim() == 1 && cu_seqlens_padded.dim() == 1);
  NVTE_CHECK(cu_seqlens.size(0) >= 2);
  NVTE_CHECK(cu_seqlens_padded.size(0) == cu_seqlens.size(0));
  NVTE_CHECK(inp.dim() >= 1);
  NVTE_CHECK(out.sizes() == inp.sizes() && out.scalar_type() == inp.scalar_type());
  NVTE_CHECK(out.is_contiguous(),
             "thd_copy_valid_tokens_from_per_split_to_rank_local output must be contiguous.");

  auto inp_c = inp.contiguous();
  auto cu_seqlens_padded_c = cu_seqlens_padded.contiguous();
  auto cu_seqlens_c = cu_seqlens.contiguous();
  int total_tokens = inp_c.size(0);
  auto te_inp = makeTransformerEngineTensor(inp_c);
  auto te_cu_seqlens_padded = makeTransformerEngineTensor(cu_seqlens_padded_c);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens_c);
  auto te_out = makeTransformerEngineTensor(out);

  nvte_thd_copy_valid_tokens_from_per_split_to_rank_local(
      te_inp.data(), te_cu_seqlens_padded.data(), te_cu_seqlens.data(), te_out.data(), total_tokens,
      at::cuda::getCurrentCUDAStream());
}

/***************************************************************************************************
 * KV Cache: Convert a tensor from qkv_format = thd to qkv_format = bshd
 **************************************************************************************************/

at::Tensor convert_thd_to_bshd(at::Tensor tensor, at::Tensor cu_seqlens, int b, int max_seq_len) {
  int h = tensor.size(1);
  int d = tensor.size(2);
  std::vector<int64_t> shape = {b, max_seq_len, h, d};
  at::Tensor new_tensor = at::zeros(shape, at::CUDA(tensor.scalar_type()));

  auto te_tensor = makeTransformerEngineTensor(tensor);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_new_tensor = makeTransformerEngineTensor(new_tensor);

  nvte_convert_thd_to_bshd(te_tensor.data(), te_cu_seqlens.data(), te_new_tensor.data(), b,
                           max_seq_len, at::cuda::getCurrentCUDAStream());

  return new_tensor;
}

/***************************************************************************************************
 * KV Cache: Convert a tensor from qkv_format = bshd to qkv_format = thd
 **************************************************************************************************/

at::Tensor convert_bshd_to_thd(at::Tensor tensor, at::Tensor cu_seqlens, int t) {
  int h = tensor.size(2);
  int d = tensor.size(3);
  std::vector<int64_t> shape = {t, h, d};
  at::Tensor new_tensor = at::zeros(shape, at::CUDA(tensor.scalar_type()));

  auto te_tensor = makeTransformerEngineTensor(tensor);
  auto te_cu_seqlens = makeTransformerEngineTensor(cu_seqlens);
  auto te_new_tensor = makeTransformerEngineTensor(new_tensor);

  nvte_convert_bshd_to_thd(te_tensor.data(), te_cu_seqlens.data(), te_new_tensor.data(), t,
                           at::cuda::getCurrentCUDAStream());

  return new_tensor;
}

void copy_to_kv_cache(at::Tensor new_k, at::Tensor new_v, at::Tensor k_cache, at::Tensor v_cache,
                      at::Tensor page_table, at::Tensor cu_new_lens, at::Tensor cu_cached_lens,
                      NVTE_QKV_Format qkv_format, int b, int max_ctx_len, int max_seq_len,
                      int max_pages_per_seq, bool is_non_paged) {
  NVTE_CHECK(k_cache.scalar_type() == v_cache.scalar_type() &&
                 new_k.scalar_type() == new_v.scalar_type() &&
                 new_k.scalar_type() == k_cache.scalar_type(),
             "new_k, new_v, k_cache and v_cache must be of the same data type.");
  NVTE_CHECK(qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD ||
                 qkv_format == NVTE_QKV_Format::NVTE_THD,
             "qkv_format must be {BSHD, SBHD, THD}.");

  auto te_new_k = makeTransformerEngineTensor(new_k);
  auto te_new_v = makeTransformerEngineTensor(new_v);
  auto te_k_cache = makeTransformerEngineTensor(k_cache);
  auto te_v_cache = makeTransformerEngineTensor(v_cache);
  auto te_page_table = makeTransformerEngineTensor(page_table);
  auto te_cu_new_lens = makeTransformerEngineTensor(cu_new_lens);
  auto te_cu_cached_lens = makeTransformerEngineTensor(cu_cached_lens);

  nvte_copy_to_kv_cache(te_new_k.data(), te_new_v.data(), te_k_cache.data(), te_v_cache.data(),
                        te_page_table.data(), te_cu_new_lens.data(), te_cu_cached_lens.data(),
                        qkv_format, b, max_ctx_len, max_seq_len, max_pages_per_seq, is_non_paged,
                        at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
