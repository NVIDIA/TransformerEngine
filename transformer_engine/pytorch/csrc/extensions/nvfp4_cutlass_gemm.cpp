/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/nvfp4_cutlass_gemm.h>
#include <transformer_engine/swizzle.h>

#include "../extensions.h"

namespace transformer_engine::pytorch {

void nvfp4_cutlass_gemm(const at::Tensor &a_data, const at::Tensor &b_data, const at::Tensor &a_sf,
                        const at::Tensor &b_sf, at::Tensor d, int64_t m, int64_t n, int64_t k,
                        double alpha, double beta, bool a_sf_swizzled, bool b_sf_swizzled) {
  TORCH_CHECK(
      a_data.is_cuda() && b_data.is_cuda() && a_sf.is_cuda() && b_sf.is_cuda() && d.is_cuda(),
      "All tensors must be CUDA tensors");
  TORCH_CHECK(a_data.is_contiguous() && b_data.is_contiguous() && a_sf.is_contiguous() &&
                  b_sf.is_contiguous() && d.is_contiguous(),
              "All tensors must be contiguous");

  // FP4 packed 2/byte and FP8-e4m3 SFs are both stored as uint8 (TE quantizer
  // wire type). Accumulator is fp32 in TMEM; only the final epilogue cast is bf16.
  TORCH_CHECK(a_data.scalar_type() == at::ScalarType::Byte, "a_data must be uint8 (FP4 packed)");
  TORCH_CHECK(b_data.scalar_type() == at::ScalarType::Byte, "b_data must be uint8 (FP4 packed)");
  TORCH_CHECK(a_sf.scalar_type() == at::ScalarType::Byte, "a_sf must be uint8 (FP8 e4m3)");
  TORCH_CHECK(b_sf.scalar_type() == at::ScalarType::Byte, "b_sf must be uint8 (FP8 e4m3)");
  TORCH_CHECK(d.scalar_type() == at::ScalarType::BFloat16, "d must be bfloat16");

  TORCH_CHECK(a_data.dim() == 2, "a_data must be 2D, got rank=", a_data.dim());
  TORCH_CHECK(b_data.dim() == 2, "b_data must be 2D, got rank=", b_data.dim());
  TORCH_CHECK(d.dim() == 2, "d must be 2D, got rank=", d.dim());

  // Storage shapes must match caller-declared (M, N, K).
  TORCH_CHECK(a_data.size(0) == m && a_data.size(1) * 2 == k,
              "a_data storage shape mismatch: expected (M=", m, ", K/2=", k / 2, "), got (",
              a_data.size(0), ", ", a_data.size(1), ")");
  TORCH_CHECK(b_data.size(0) == n && b_data.size(1) * 2 == k,
              "b_data storage shape mismatch: expected (N=", n, ", K/2=", k / 2, "), got (",
              b_data.size(0), ", ", b_data.size(1), ")");
  TORCH_CHECK(d.size(0) == m && d.size(1) == n, "d shape mismatch: expected (M=", m, ", N=", n,
              "), got (", d.size(0), ", ", d.size(1), ")");

  // CUTLASS NVFP4 mainloop wants SF in SM100 Sm1xxBlkScaledConfig layout;
  // swizzle internally so the caller can pass linear (M, K/16) too.
  const auto stream = at::cuda::getCurrentCUDAStream();

  const std::vector<size_t> a_data_shape = {static_cast<size_t>(m), static_cast<size_t>(k)};
  const std::vector<size_t> b_data_shape = {static_cast<size_t>(n), static_cast<size_t>(k)};
  const std::vector<size_t> a_sf_shape = {static_cast<size_t>(m), static_cast<size_t>(k / 16)};
  const std::vector<size_t> b_sf_shape = {static_cast<size_t>(n), static_cast<size_t>(k / 16)};

  TORCH_CHECK(a_sf.numel() == static_cast<int64_t>(m * k / 16),
              "a_sf size mismatch: expected M*K/16=", m * k / 16, ", got ", a_sf.numel());
  TORCH_CHECK(b_sf.numel() == static_cast<int64_t>(n * k / 16),
              "b_sf size mismatch: expected N*K/16=", n * k / 16, ", got ", b_sf.numel());

  // a_sf_swizzled / b_sf_swizzled = true skip the per-operand swizzle and
  // consume the caller's buffer directly (bench-only fast-path for --gemm-only).
  auto byte_opts = a_sf.options().dtype(at::kByte);
  at::Tensor a_sf_swz_buf;
  at::Tensor b_sf_swz_buf;
  void *a_sf_swz_ptr = nullptr;
  void *b_sf_swz_ptr = nullptr;

  if (a_sf_swizzled) {
    a_sf_swz_ptr = a_sf.data_ptr();
  } else {
    a_sf_swz_buf = at::empty({a_sf.numel()}, byte_opts);
    a_sf_swz_ptr = a_sf_swz_buf.data_ptr();

    TensorWrapper in_nvte(NVTE_NVFP4_1D_SCALING);
    in_nvte.set_rowwise_data(a_data.data_ptr(), DType::kFloat4E2M1, a_data_shape);
    in_nvte.set_rowwise_scale_inv(a_sf.data_ptr(), DType::kFloat8E4M3, a_sf_shape);

    TensorWrapper out_nvte(NVTE_NVFP4_1D_SCALING);
    out_nvte.set_rowwise_data(a_data.data_ptr(), DType::kFloat4E2M1, a_data_shape);
    out_nvte.set_rowwise_scale_inv(a_sf_swz_ptr, DType::kFloat8E4M3, a_sf_shape);
    out_nvte.set_with_gemm_swizzled_scales(true);

    nvte_swizzle_scaling_factors(in_nvte.data(), out_nvte.data(), stream);
  }
  if (b_sf_swizzled) {
    b_sf_swz_ptr = b_sf.data_ptr();
  } else {
    b_sf_swz_buf = at::empty({b_sf.numel()}, byte_opts);
    b_sf_swz_ptr = b_sf_swz_buf.data_ptr();

    TensorWrapper in_nvte(NVTE_NVFP4_1D_SCALING);
    in_nvte.set_rowwise_data(b_data.data_ptr(), DType::kFloat4E2M1, b_data_shape);
    in_nvte.set_rowwise_scale_inv(b_sf.data_ptr(), DType::kFloat8E4M3, b_sf_shape);

    TensorWrapper out_nvte(NVTE_NVFP4_1D_SCALING);
    out_nvte.set_rowwise_data(b_data.data_ptr(), DType::kFloat4E2M1, b_data_shape);
    out_nvte.set_rowwise_scale_inv(b_sf_swz_ptr, DType::kFloat8E4M3, b_sf_shape);
    out_nvte.set_with_gemm_swizzled_scales(true);

    nvte_swizzle_scaling_factors(in_nvte.data(), out_nvte.data(), stream);
  }

  // Logical FP4/FP8 dtypes for the C API; pointers reference uint8 storage.
  TensorWrapper a_te =
      makeTransformerEngineTensor(a_data.data_ptr(), a_data_shape, DType::kFloat4E2M1);
  TensorWrapper b_te =
      makeTransformerEngineTensor(b_data.data_ptr(), b_data_shape, DType::kFloat4E2M1);
  TensorWrapper a_sf_te = makeTransformerEngineTensor(
      a_sf_swz_ptr, std::vector<size_t>{static_cast<size_t>(a_sf.numel())}, DType::kFloat8E4M3);
  TensorWrapper b_sf_te = makeTransformerEngineTensor(
      b_sf_swz_ptr, std::vector<size_t>{static_cast<size_t>(b_sf.numel())}, DType::kFloat8E4M3);
  TensorWrapper d_te = makeTransformerEngineTensor(
      d.data_ptr(), std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)},
      DType::kBFloat16);

  nvte_nvfp4_cutlass_gemm(a_te.data(), b_te.data(), a_sf_te.data(), b_sf_te.data(), d_te.data(),
                          static_cast<float>(alpha), static_cast<float>(beta), stream);
}

// D[i,j] = bf16(alpha_a[i] * alpha_b[j] * (A @ B^T)[i,j]) -- per-row*per-col
// fold REPLACES the trailing nvfp4_per_token_post_scale kernel. Same SF-swizzle
// contract as nvfp4_cutlass_gemm above.
void nvfp4_cutlass_per_token_gemm(const at::Tensor &a_data, const at::Tensor &b_data,
                                  const at::Tensor &a_sf, const at::Tensor &b_sf,
                                  const at::Tensor &alpha_a, const at::Tensor &alpha_b,
                                  at::Tensor d, int64_t m, int64_t n, int64_t k, bool a_sf_swizzled,
                                  bool b_sf_swizzled) {
  TORCH_CHECK(a_data.is_cuda() && b_data.is_cuda() && a_sf.is_cuda() && b_sf.is_cuda() &&
                  alpha_a.is_cuda() && alpha_b.is_cuda() && d.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(a_data.is_contiguous() && b_data.is_contiguous() && a_sf.is_contiguous() &&
                  b_sf.is_contiguous() && alpha_a.is_contiguous() && alpha_b.is_contiguous() &&
                  d.is_contiguous(),
              "All tensors must be contiguous");

  TORCH_CHECK(a_data.scalar_type() == at::ScalarType::Byte, "a_data must be uint8 (FP4 packed)");
  TORCH_CHECK(b_data.scalar_type() == at::ScalarType::Byte, "b_data must be uint8 (FP4 packed)");
  TORCH_CHECK(a_sf.scalar_type() == at::ScalarType::Byte, "a_sf must be uint8 (FP8 e4m3)");
  TORCH_CHECK(b_sf.scalar_type() == at::ScalarType::Byte, "b_sf must be uint8 (FP8 e4m3)");
  TORCH_CHECK(d.scalar_type() == at::ScalarType::BFloat16, "d must be bfloat16");
  TORCH_CHECK(alpha_a.scalar_type() == at::ScalarType::Float, "alpha_a must be float32");
  TORCH_CHECK(alpha_b.scalar_type() == at::ScalarType::Float, "alpha_b must be float32");

  TORCH_CHECK(a_data.dim() == 2 && b_data.dim() == 2 && d.dim() == 2,
              "a_data / b_data / d must all be 2D");

  TORCH_CHECK(a_data.size(0) == m && a_data.size(1) * 2 == k,
              "a_data storage shape mismatch: expected (M=", m, ", K/2=", k / 2, "), got (",
              a_data.size(0), ", ", a_data.size(1), ")");
  TORCH_CHECK(b_data.size(0) == n && b_data.size(1) * 2 == k,
              "b_data storage shape mismatch: expected (N=", n, ", K/2=", k / 2, "), got (",
              b_data.size(0), ", ", b_data.size(1), ")");
  TORCH_CHECK(d.size(0) == m && d.size(1) == n, "d shape mismatch: expected (M=", m, ", N=", n,
              "), got (", d.size(0), ", ", d.size(1), ")");
  TORCH_CHECK(alpha_a.numel() == m, "alpha_a must have M=", m, " elements, got ", alpha_a.numel());
  TORCH_CHECK(alpha_b.numel() == n, "alpha_b must have N=", n, " elements, got ", alpha_b.numel());

  const auto stream = at::cuda::getCurrentCUDAStream();

  const std::vector<size_t> a_data_shape = {static_cast<size_t>(m), static_cast<size_t>(k)};
  const std::vector<size_t> b_data_shape = {static_cast<size_t>(n), static_cast<size_t>(k)};
  const std::vector<size_t> a_sf_shape = {static_cast<size_t>(m), static_cast<size_t>(k / 16)};
  const std::vector<size_t> b_sf_shape = {static_cast<size_t>(n), static_cast<size_t>(k / 16)};

  TORCH_CHECK(a_sf.numel() == static_cast<int64_t>(m * k / 16),
              "a_sf size mismatch: expected M*K/16=", m * k / 16, ", got ", a_sf.numel());
  TORCH_CHECK(b_sf.numel() == static_cast<int64_t>(n * k / 16),
              "b_sf size mismatch: expected N*K/16=", n * k / 16, ", got ", b_sf.numel());

  // SF swizzle (shared logic with the scalar-alpha entry point above).
  auto byte_opts = a_sf.options().dtype(at::kByte);
  at::Tensor a_sf_swz_buf;
  at::Tensor b_sf_swz_buf;
  void *a_sf_swz_ptr = nullptr;
  void *b_sf_swz_ptr = nullptr;

  if (a_sf_swizzled) {
    a_sf_swz_ptr = a_sf.data_ptr();
  } else {
    a_sf_swz_buf = at::empty({a_sf.numel()}, byte_opts);
    a_sf_swz_ptr = a_sf_swz_buf.data_ptr();
    TensorWrapper in_nvte(NVTE_NVFP4_1D_SCALING);
    in_nvte.set_rowwise_data(a_data.data_ptr(), DType::kFloat4E2M1, a_data_shape);
    in_nvte.set_rowwise_scale_inv(a_sf.data_ptr(), DType::kFloat8E4M3, a_sf_shape);
    TensorWrapper out_nvte(NVTE_NVFP4_1D_SCALING);
    out_nvte.set_rowwise_data(a_data.data_ptr(), DType::kFloat4E2M1, a_data_shape);
    out_nvte.set_rowwise_scale_inv(a_sf_swz_ptr, DType::kFloat8E4M3, a_sf_shape);
    out_nvte.set_with_gemm_swizzled_scales(true);
    nvte_swizzle_scaling_factors(in_nvte.data(), out_nvte.data(), stream);
  }
  if (b_sf_swizzled) {
    b_sf_swz_ptr = b_sf.data_ptr();
  } else {
    b_sf_swz_buf = at::empty({b_sf.numel()}, byte_opts);
    b_sf_swz_ptr = b_sf_swz_buf.data_ptr();
    TensorWrapper in_nvte(NVTE_NVFP4_1D_SCALING);
    in_nvte.set_rowwise_data(b_data.data_ptr(), DType::kFloat4E2M1, b_data_shape);
    in_nvte.set_rowwise_scale_inv(b_sf.data_ptr(), DType::kFloat8E4M3, b_sf_shape);
    TensorWrapper out_nvte(NVTE_NVFP4_1D_SCALING);
    out_nvte.set_rowwise_data(b_data.data_ptr(), DType::kFloat4E2M1, b_data_shape);
    out_nvte.set_rowwise_scale_inv(b_sf_swz_ptr, DType::kFloat8E4M3, b_sf_shape);
    out_nvte.set_with_gemm_swizzled_scales(true);
    nvte_swizzle_scaling_factors(in_nvte.data(), out_nvte.data(), stream);
  }

  TensorWrapper a_te =
      makeTransformerEngineTensor(a_data.data_ptr(), a_data_shape, DType::kFloat4E2M1);
  TensorWrapper b_te =
      makeTransformerEngineTensor(b_data.data_ptr(), b_data_shape, DType::kFloat4E2M1);
  TensorWrapper a_sf_te = makeTransformerEngineTensor(
      a_sf_swz_ptr, std::vector<size_t>{static_cast<size_t>(a_sf.numel())}, DType::kFloat8E4M3);
  TensorWrapper b_sf_te = makeTransformerEngineTensor(
      b_sf_swz_ptr, std::vector<size_t>{static_cast<size_t>(b_sf.numel())}, DType::kFloat8E4M3);
  TensorWrapper aa_te = makeTransformerEngineTensor(
      alpha_a.data_ptr(), std::vector<size_t>{static_cast<size_t>(m)}, DType::kFloat32);
  TensorWrapper ab_te = makeTransformerEngineTensor(
      alpha_b.data_ptr(), std::vector<size_t>{static_cast<size_t>(n)}, DType::kFloat32);
  TensorWrapper d_te = makeTransformerEngineTensor(
      d.data_ptr(), std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)},
      DType::kBFloat16);

  nvte_nvfp4_cutlass_per_token_gemm(a_te.data(), b_te.data(), a_sf_te.data(), b_sf_te.data(),
                                    aa_te.data(), ab_te.data(), d_te.data(), stream);
}

}  // namespace transformer_engine::pytorch
