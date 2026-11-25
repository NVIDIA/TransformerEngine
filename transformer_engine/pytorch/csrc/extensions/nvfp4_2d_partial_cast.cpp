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
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "scale must be a float tensor");
  TORCH_CHECK(global_scale.numel() == 1, "global_scale must be a scalar tensor");
  TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
              "global_scale must be a float tensor");
  TORCH_CHECK(
      inp.scalar_type() == at::ScalarType::Float || inp.scalar_type() == at::ScalarType::BFloat16,
      "input must be a float or bfloat16 tensor");

  const TensorWrapper inp_cu = makeTransformerEngineTensor(inp.contiguous());
  const TensorWrapper out_cu = makeTransformerEngineTensor(out, py::none());
  const TensorWrapper scale_cu = makeTransformerEngineTensor(scale);
  const TensorWrapper global_scale_cu = makeTransformerEngineTensor(global_scale);

  nvte_nvfp4_2d_partial_cast_v2(inp_cu.data(), out_cu.data(), scale_cu.data(),
                                global_scale_cu.data(), h, w, scale.stride(0), scale.stride(1),
                                start_offset, block_len, static_cast<NVTEDType>(scale_dtype),
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
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "scale must be a float tensor");
    TORCH_CHECK(global_scale.numel() == 1, "global_scale must be a scalar tensor");
    TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
                "global_scale must be a float tensor");
    TORCH_CHECK(
        inp.scalar_type() == at::ScalarType::Float || inp.scalar_type() == at::ScalarType::BFloat16,
        "input must be a float or bfloat16 tensor");

    const TensorWrapper inp_cu = makeTransformerEngineTensor(inp.contiguous());
    const TensorWrapper out_cu = makeTransformerEngineTensor(out);
    const TensorWrapper scale_cu = makeTransformerEngineTensor(scale);
    const TensorWrapper global_scale_cu = makeTransformerEngineTensor(global_scale);

    nvte_nvfp4_2d_partial_cast_v2(inp_cu.data(), out_cu.data(), scale_cu.data(),
                                  global_scale_cu.data(), h, w, scale.stride(0), scale.stride(1),
                                  start_offset, static_cast<size_t>(block_len),
                                  static_cast<NVTEDType>(scale_dtype), stream);
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

namespace {

// TE tensors for a list of contiguous torch tensors; `wrappers` keeps them alive.
std::vector<NVTETensor> make_te_tensors(const std::vector<at::Tensor>& tensors,
                                        std::vector<TensorWrapper>* wrappers, const char* name) {
  std::vector<NVTETensor> out;
  out.reserve(tensors.size());
  for (const auto& t : tensors) {
    TORCH_CHECK(t.is_contiguous(), name, " tensors must be contiguous");
    wrappers->emplace_back(makeTransformerEngineTensor(t));
    out.push_back(wrappers->back().data());
  }
  return out;
}

std::vector<size_t> to_size_vector(const std::vector<int64_t>& v, size_t n, const char* name) {
  TORCH_CHECK(v.size() == n, name, " size mismatch");
  std::vector<size_t> out(n);
  for (size_t i = 0; i < n; ++i) {
    TORCH_CHECK(v[i] >= 0, name, " must not be negative");
    out[i] = static_cast<size_t>(v[i]);
  }
  return out;
}

}  // namespace

void nvfp4_2d_multi_tensor_partial_amax(const std::vector<at::Tensor>& inp_list,
                                        const std::vector<at::Tensor>& amax_list,
                                        const std::vector<int64_t>& h_list,
                                        const std::vector<int64_t>& w_list,
                                        const std::vector<int64_t>& start_offset_list,
                                        int64_t block_len, const DType cast_dtype) {
  const size_t n = inp_list.size();
  TORCH_CHECK(amax_list.size() == n, "amax_list size mismatch");
  if (n == 0) return;
  const auto h = to_size_vector(h_list, n, "h_list");
  const auto w = to_size_vector(w_list, n, "w_list");
  const auto start = to_size_vector(start_offset_list, n, "start_offset_list");
  std::vector<TensorWrapper> wrappers;
  wrappers.reserve(2 * n);
  auto inp = make_te_tensors(inp_list, &wrappers, "inp_list");
  auto amax = make_te_tensors(amax_list, &wrappers, "amax_list");
  nvte_nvfp4_2d_multi_tensor_partial_amax(
      inp.data(), amax.data(), h.data(), w.data(), start.data(), n, static_cast<size_t>(block_len),
      static_cast<NVTEDType>(cast_dtype), at::cuda::getCurrentCUDAStream());
}

void nvfp4_2d_multi_tensor_global_amax(const std::vector<at::Tensor>& block_amax_list,
                                       const std::vector<at::Tensor>& global_amax_list) {
  const size_t n = block_amax_list.size();
  TORCH_CHECK(global_amax_list.size() == n, "global_amax_list size mismatch");
  if (n == 0) return;
  std::vector<TensorWrapper> wrappers;
  wrappers.reserve(2 * n);
  auto block_amax = make_te_tensors(block_amax_list, &wrappers, "block_amax_list");
  auto global_amax = make_te_tensors(global_amax_list, &wrappers, "global_amax_list");
  nvte_nvfp4_2d_multi_tensor_global_amax(block_amax.data(), global_amax.data(), n,
                                         at::cuda::getCurrentCUDAStream());
}

void nvfp4_2d_multi_tensor_fused_scale(const std::vector<at::Tensor>& block_amax_list,
                                       const std::vector<at::Tensor>& global_amax_list,
                                       const std::vector<at::Tensor>& per_block_scale_list,
                                       const std::vector<at::Tensor>& target_scale_list,
                                       const std::vector<at::Tensor>& target_amax_list,
                                       const std::vector<int64_t>& tile_rows_list,
                                       const std::vector<int64_t>& tile_cols_list,
                                       const std::vector<int64_t>& rows_padded_list,
                                       int64_t block_len, const DType scale_dtype) {
  const size_t n = block_amax_list.size();
  TORCH_CHECK(global_amax_list.size() == n, "global_amax_list size mismatch");
  TORCH_CHECK(per_block_scale_list.size() == n, "per_block_scale_list size mismatch");
  TORCH_CHECK(target_scale_list.size() == n, "target_scale_list size mismatch");
  TORCH_CHECK(target_amax_list.size() == n, "target_amax_list size mismatch");
  if (n == 0) return;
  const auto tile_rows = to_size_vector(tile_rows_list, n, "tile_rows_list");
  const auto tile_cols = to_size_vector(tile_cols_list, n, "tile_cols_list");
  const auto rows_padded = to_size_vector(rows_padded_list, n, "rows_padded_list");
  for (const auto& t : target_scale_list) {
    TORCH_CHECK(t.scalar_type() == at::kByte, "Target scale must be uint8 scale storage.");
  }
  std::vector<TensorWrapper> wrappers;
  wrappers.reserve(5 * n);
  auto block_amax = make_te_tensors(block_amax_list, &wrappers, "block_amax_list");
  auto global_amax = make_te_tensors(global_amax_list, &wrappers, "global_amax_list");
  auto per_block_scale = make_te_tensors(per_block_scale_list, &wrappers, "per_block_scale_list");
  auto target_scale = make_te_tensors(target_scale_list, &wrappers, "target_scale_list");
  auto target_amax = make_te_tensors(target_amax_list, &wrappers, "target_amax_list");
  nvte_nvfp4_2d_multi_tensor_fused_scale(
      block_amax.data(), global_amax.data(), per_block_scale.data(), target_scale.data(),
      target_amax.data(), tile_rows.data(), tile_cols.data(), rows_padded.data(), n,
      static_cast<size_t>(block_len), static_cast<NVTEDType>(scale_dtype),
      at::cuda::getCurrentCUDAStream());
}

void nvfp4_2d_multi_tensor_partial_cast(
    const std::vector<at::Tensor>& inp_list, const std::vector<at::Tensor>& out_list,
    const std::vector<at::Tensor>& scale_list, const std::vector<at::Tensor>& global_scale_list,
    const std::vector<int64_t>& h_list, const std::vector<int64_t>& w_list,
    const std::vector<int64_t>& start_offset_list, int64_t block_len, const DType cast_dtype,
    const DType scale_dtype) {
  const size_t n = inp_list.size();
  TORCH_CHECK(out_list.size() == n, "out_list size mismatch");
  TORCH_CHECK(scale_list.size() == n, "scale_list size mismatch");
  TORCH_CHECK(global_scale_list.size() == n, "global_scale_list size mismatch");
  if (n == 0) return;
  const auto h = to_size_vector(h_list, n, "h_list");
  const auto w = to_size_vector(w_list, n, "w_list");
  const auto start = to_size_vector(start_offset_list, n, "start_offset_list");
  std::vector<TensorWrapper> wrappers;
  wrappers.reserve(4 * n);
  auto inp = make_te_tensors(inp_list, &wrappers, "inp_list");
  auto out = make_te_tensors(out_list, &wrappers, "out_list");
  auto scale = make_te_tensors(scale_list, &wrappers, "scale_list");
  auto global_scale = make_te_tensors(global_scale_list, &wrappers, "global_scale_list");
  nvte_nvfp4_2d_multi_tensor_partial_cast(
      inp.data(), out.data(), scale.data(), global_scale.data(), h.data(), w.data(), start.data(),
      n, static_cast<size_t>(block_len), static_cast<NVTEDType>(cast_dtype),
      static_cast<NVTEDType>(scale_dtype), at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
