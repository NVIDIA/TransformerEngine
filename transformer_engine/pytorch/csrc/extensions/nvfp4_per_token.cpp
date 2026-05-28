/*************************************************************************
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/nvfp4_per_token.h>
#include <transformer_engine/swizzle.h>
#include <transformer_engine/transformer_engine.h>

#include <array>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "../extensions.h"

namespace transformer_engine::pytorch {

// NVFP4 per-token cast bindings. Shared TensorWrapper assembler dispatches
// composite (K1+K2), K1-only and K2-only via `mode`. bf16-only, M/K % 128 == 0.
// SFs emit in compact (non-swizzled) layout; swizzle for cuBLAS LT lives elsewhere.
namespace {

// Validates the input and assembles ``out_te`` for all 3 modes; caller
// dispatches to the right C-API entry on the caller's stream.
void assemble_per_token_tensors(const at::Tensor& input, at::Tensor q_row, at::Tensor s_dec_row,
                                at::Tensor row_amax, at::Tensor q_col, at::Tensor s_dec_col,
                                at::Tensor col_amax, bool rowwise, bool columnwise, int mode,
                                TensorWrapper& in_te, TensorWrapper& out_te) {
  TORCH_CHECK(rowwise || columnwise, "At least one of rowwise/columnwise must be True.");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16,
              "Per-token cast is bf16-only. Got dtype ", input.scalar_type());
  const int64_t M = input.size(0);
  const int64_t K = input.size(1);
  TORCH_CHECK(M % 128 == 0, "Per-token cast requires M % 128 == 0; got M=", M);
  TORCH_CHECK(K % 128 == 0, "Per-token cast requires K % 128 == 0; got K=", K);

  const std::vector<size_t> in_shape = {static_cast<size_t>(M), static_cast<size_t>(K)};
  in_te = makeTransformerEngineTensor(input.data_ptr(), in_shape, DType::kBFloat16);

  // K1 (mode==1) populates ONLY amax slots; K2 / composite (mode==0/2)
  // populate the FP4 + e4m3 SF slots too. The amax slots are also wired
  // for K2 because the kernel READS them.
  const bool needs_fp4_outputs = (mode == 0) || (mode == 2);

  if (rowwise) {
    TORCH_CHECK(row_amax.is_cuda() && row_amax.is_contiguous(),
                "row_amax must be a contiguous CUDA tensor");
    TORCH_CHECK(row_amax.scalar_type() == at::ScalarType::Float, "row_amax must be float32");
    TORCH_CHECK(row_amax.numel() == M, "row_amax numel mismatch: expected M=", M, ", got ",
                row_amax.numel());
    out_te.set_amax(row_amax.data_ptr(), DType::kFloat32,
                    std::vector<size_t>{static_cast<size_t>(M)});

    if (needs_fp4_outputs) {
      TORCH_CHECK(q_row.is_cuda() && q_row.is_contiguous(),
                  "q_row must be a contiguous CUDA tensor");
      TORCH_CHECK(s_dec_row.is_cuda() && s_dec_row.is_contiguous(),
                  "s_dec_row must be a contiguous CUDA tensor");
      TORCH_CHECK(q_row.scalar_type() == at::ScalarType::Byte, "q_row must be uint8 (FP4 packed)");
      TORCH_CHECK(s_dec_row.scalar_type() == at::ScalarType::Byte,
                  "s_dec_row must be uint8 (FP8 e4m3 raw bytes)");
      TORCH_CHECK(q_row.numel() == M * K / 2, "q_row numel mismatch: expected M*K/2=", M * K / 2,
                  ", got ", q_row.numel());
      TORCH_CHECK(s_dec_row.numel() == M * K / 16,
                  "s_dec_row numel mismatch: expected M*K/16=", M * K / 16, ", got ",
                  s_dec_row.numel());
      out_te.set_rowwise_data(q_row.data_ptr(), DType::kFloat4E2M1, in_shape);
      out_te.set_rowwise_scale_inv(
          s_dec_row.data_ptr(), DType::kFloat8E4M3,
          std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K / 16)});
    }
  }
  if (columnwise) {
    TORCH_CHECK(col_amax.is_cuda() && col_amax.is_contiguous(),
                "col_amax must be a contiguous CUDA tensor");
    TORCH_CHECK(col_amax.scalar_type() == at::ScalarType::Float, "col_amax must be float32");
    TORCH_CHECK(col_amax.numel() == K, "col_amax numel mismatch: expected K=", K, ", got ",
                col_amax.numel());
    out_te.set_columnwise_amax(col_amax.data_ptr(), DType::kFloat32,
                               std::vector<size_t>{static_cast<size_t>(K)});

    if (needs_fp4_outputs) {
      TORCH_CHECK(q_col.is_cuda() && q_col.is_contiguous(),
                  "q_col must be a contiguous CUDA tensor");
      TORCH_CHECK(s_dec_col.is_cuda() && s_dec_col.is_contiguous(),
                  "s_dec_col must be a contiguous CUDA tensor");
      TORCH_CHECK(q_col.scalar_type() == at::ScalarType::Byte, "q_col must be uint8 (FP4 packed)");
      TORCH_CHECK(s_dec_col.scalar_type() == at::ScalarType::Byte,
                  "s_dec_col must be uint8 (FP8 e4m3 raw bytes)");
      TORCH_CHECK(q_col.numel() == K * M / 2, "q_col numel mismatch: expected K*M/2=", K * M / 2,
                  ", got ", q_col.numel());
      TORCH_CHECK(s_dec_col.numel() == K * M / 16,
                  "s_dec_col numel mismatch: expected K*M/16=", K * M / 16, ", got ",
                  s_dec_col.numel());
      out_te.set_columnwise_data(
          q_col.data_ptr(), DType::kFloat4E2M1,
          std::vector<size_t>{static_cast<size_t>(K), static_cast<size_t>(M)});
      out_te.set_columnwise_scale_inv(
          s_dec_col.data_ptr(), DType::kFloat8E4M3,
          std::vector<size_t>{static_cast<size_t>(K), static_cast<size_t>(M / 16)});
    }
  }
}

}  // namespace

// Production composite (K1 + K2 back-to-back). with_rht=true enables the
// 16-pt col-wise RHT in BOTH K1 and K2 so outer + inner SFs stay consistent.
void nvfp4_per_token_quantize(const at::Tensor& input, at::Tensor q_row, at::Tensor s_dec_row,
                              at::Tensor row_amax, at::Tensor q_col, at::Tensor s_dec_col,
                              at::Tensor col_amax, bool rowwise, bool columnwise, bool with_rht,
                              int64_t random_sign_mask_t) {
  TensorWrapper in_te;
  TensorWrapper out_te(NVTE_NVFP4_1D_SCALING);
  assemble_per_token_tensors(input, q_row, s_dec_row, row_amax, q_col, s_dec_col, col_amax, rowwise,
                             columnwise, /*mode=*/0, in_te, out_te);
  const auto stream = at::cuda::getCurrentCUDAStream();
  nvte_nvfp4_per_token_quantize(in_te.data(), nullptr, out_te.data(), with_rht ? 1 : 0,
                                static_cast<int>(random_sign_mask_t & 0xFFFF), stream);
}

// K1-only (diagnostic / bench): populates only amax buffers. with_rht=true
// applies the 16-pt col-wise RHT before amax (rowwise unaffected);
// random_sign_mask_t low 16 bits = sign-flip pattern.
void nvfp4_per_token_amax(const at::Tensor& input, at::Tensor row_amax, at::Tensor col_amax,
                          bool rowwise, bool columnwise, bool with_rht,
                          int64_t random_sign_mask_t) {
  at::Tensor empty_u8;  // not consumed by K1
  TensorWrapper in_te;
  TensorWrapper out_te(NVTE_NVFP4_1D_SCALING);
  assemble_per_token_tensors(input, empty_u8, empty_u8, row_amax, empty_u8, empty_u8, col_amax,
                             rowwise, columnwise, /*mode=*/1, in_te, out_te);
  const auto stream = at::cuda::getCurrentCUDAStream();
  // C-API matches prod's `int` convention; only low 16 bits are consumed.
  nvte_nvfp4_per_token_amax(in_te.data(), nullptr, out_te.data(), with_rht ? 1 : 0,
                            static_cast<int>(random_sign_mask_t & 0xFFFF), stream);
}

// K2-only (diagnostic / bench): reads pre-filled amax buffers, emits FP4 + SFs.
// with_rht=true requires col_amax to have been produced by an earlier K1
// amax call with the SAME mask, else inner SFs are miscalibrated.
void nvfp4_per_token_encode(const at::Tensor& input, at::Tensor q_row, at::Tensor s_dec_row,
                            at::Tensor row_amax, at::Tensor q_col, at::Tensor s_dec_col,
                            at::Tensor col_amax, bool rowwise, bool columnwise, bool with_rht,
                            int64_t random_sign_mask_t) {
  TensorWrapper in_te;
  TensorWrapper out_te(NVTE_NVFP4_1D_SCALING);
  assemble_per_token_tensors(input, q_row, s_dec_row, row_amax, q_col, s_dec_col, col_amax, rowwise,
                             columnwise, /*mode=*/2, in_te, out_te);
  const auto stream = at::cuda::getCurrentCUDAStream();
  nvte_nvfp4_per_token_encode(in_te.data(), nullptr, out_te.data(), with_rht ? 1 : 0,
                              static_cast<int>(random_sign_mask_t & 0xFFFF), stream);
}

// Apply per-token post-scale to a GEMM output (see nvfp4_per_token.h for math).
void nvfp4_per_token_post_scale(at::Tensor d, const at::Tensor& row_amax_a,
                                const at::Tensor& row_amax_b) {
  TORCH_CHECK(d.is_cuda() && d.is_contiguous(), "d must be a contiguous CUDA tensor");
  TORCH_CHECK(row_amax_a.is_cuda() && row_amax_a.is_contiguous(),
              "row_amax_a must be a contiguous CUDA tensor");
  TORCH_CHECK(row_amax_b.is_cuda() && row_amax_b.is_contiguous(),
              "row_amax_b must be a contiguous CUDA tensor");
  TORCH_CHECK(d.dim() == 2, "d must be 2D");
  TORCH_CHECK(d.scalar_type() == at::ScalarType::BFloat16, "d must be bf16");
  TORCH_CHECK(row_amax_a.scalar_type() == at::ScalarType::Float, "row_amax_a must be fp32");
  TORCH_CHECK(row_amax_b.scalar_type() == at::ScalarType::Float, "row_amax_b must be fp32");

  const int64_t M = d.size(0);
  const int64_t N = d.size(1);
  TORCH_CHECK(row_amax_a.numel() == M, "row_amax_a numel mismatch: expected M=", M, ", got ",
              row_amax_a.numel());
  TORCH_CHECK(row_amax_b.numel() == N, "row_amax_b numel mismatch: expected N=", N, ", got ",
              row_amax_b.numel());

  const auto stream = at::cuda::getCurrentCUDAStream();

  TensorWrapper d_te = makeTransformerEngineTensor(
      d.data_ptr(), std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)},
      DType::kBFloat16);
  TensorWrapper ra_te = makeTransformerEngineTensor(
      row_amax_a.data_ptr(), std::vector<size_t>{static_cast<size_t>(M)}, DType::kFloat32);
  TensorWrapper rb_te = makeTransformerEngineTensor(
      row_amax_b.data_ptr(), std::vector<size_t>{static_cast<size_t>(N)}, DType::kFloat32);

  nvte_nvfp4_per_token_post_scale(d_te.data(), ra_te.data(), rb_te.data(), stream);
}

// End-to-end NVFP4 per-token GEMM: swizzle compact SFs -> cuBLAS LT NVFP4
// GEMM (operand amax pinned to 1.0 to cancel the 2688^2 inner-SF factor) ->
// per-row post-scale. beta must be 0.0. Math in nvfp4_per_token.h.
void nvfp4_per_token_gemm(const at::Tensor& a_data, const at::Tensor& b_data,
                          const at::Tensor& a_sf, const at::Tensor& b_sf,
                          const at::Tensor& a_row_amax, const at::Tensor& b_row_amax, at::Tensor d,
                          const at::Tensor& workspace, int64_t m, int64_t n, int64_t k,
                          double alpha, double beta) {
  TORCH_CHECK(a_data.is_cuda() && b_data.is_cuda() && a_sf.is_cuda() && b_sf.is_cuda() &&
                  a_row_amax.is_cuda() && b_row_amax.is_cuda() && d.is_cuda() &&
                  workspace.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(a_data.is_contiguous() && b_data.is_contiguous() && a_sf.is_contiguous() &&
                  b_sf.is_contiguous() && a_row_amax.is_contiguous() &&
                  b_row_amax.is_contiguous() && d.is_contiguous() && workspace.is_contiguous(),
              "All tensors must be contiguous");

  TORCH_CHECK(a_data.scalar_type() == at::ScalarType::Byte, "a_data must be uint8 (FP4 packed)");
  TORCH_CHECK(b_data.scalar_type() == at::ScalarType::Byte, "b_data must be uint8 (FP4 packed)");
  TORCH_CHECK(a_sf.scalar_type() == at::ScalarType::Byte, "a_sf must be uint8 (FP8 e4m3)");
  TORCH_CHECK(b_sf.scalar_type() == at::ScalarType::Byte, "b_sf must be uint8 (FP8 e4m3)");
  TORCH_CHECK(a_row_amax.scalar_type() == at::ScalarType::Float, "a_row_amax must be float32");
  TORCH_CHECK(b_row_amax.scalar_type() == at::ScalarType::Float, "b_row_amax must be float32");
  TORCH_CHECK(d.scalar_type() == at::ScalarType::BFloat16, "d must be bfloat16");
  TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Byte, "workspace must be uint8");

  TORCH_CHECK(a_data.dim() == 2 && b_data.dim() == 2 && d.dim() == 2, "a_data/b_data/d must be 2D");
  TORCH_CHECK(a_data.size(0) == m && a_data.size(1) * 2 == k,
              "a_data shape mismatch: expected (M=", m, ", K/2=", k / 2, "), got (", a_data.size(0),
              ", ", a_data.size(1), ")");
  TORCH_CHECK(b_data.size(0) == n && b_data.size(1) * 2 == k,
              "b_data shape mismatch: expected (N=", n, ", K/2=", k / 2, "), got (", b_data.size(0),
              ", ", b_data.size(1), ")");
  TORCH_CHECK(d.size(0) == m && d.size(1) == n, "d shape mismatch: expected (M=", m, ", N=", n,
              "), got (", d.size(0), ", ", d.size(1), ")");

  TORCH_CHECK(k % 16 == 0, "k must be a multiple of 16 (NVFP4 inner SFVecSize)");
  TORCH_CHECK(a_sf.numel() == static_cast<int64_t>(m * k / 16),
              "a_sf numel mismatch: expected M*K/16=", m * k / 16, ", got ", a_sf.numel());
  TORCH_CHECK(b_sf.numel() == static_cast<int64_t>(n * k / 16),
              "b_sf numel mismatch: expected N*K/16=", n * k / 16, ", got ", b_sf.numel());
  TORCH_CHECK(a_row_amax.numel() == m, "a_row_amax numel mismatch: expected M=", m, ", got ",
              a_row_amax.numel());
  TORCH_CHECK(b_row_amax.numel() == n, "b_row_amax numel mismatch: expected N=", n, ", got ",
              b_row_amax.numel());

  TORCH_CHECK(static_cast<float>(beta) == 0.0f,
              "nvfp4_per_token_gemm: beta != 0 not yet supported. Got beta=", beta);

  const auto stream = at::cuda::getCurrentCUDAStream();

  const std::vector<size_t> a_data_shape = {static_cast<size_t>(m), static_cast<size_t>(k)};
  const std::vector<size_t> b_data_shape = {static_cast<size_t>(n), static_cast<size_t>(k)};
  const std::vector<size_t> a_sf_shape = {static_cast<size_t>(m), static_cast<size_t>(k / 16)};
  const std::vector<size_t> b_sf_shape = {static_cast<size_t>(n), static_cast<size_t>(k / 16)};

  // Swizzled SF buffers (cuBLAS LT requires swizzled layout).
  auto byte_opts = a_sf.options().dtype(at::kByte);
  at::Tensor a_sf_swizzled = at::empty({a_sf.numel()}, byte_opts);
  at::Tensor b_sf_swizzled = at::empty({b_sf.numel()}, byte_opts);

  {
    TensorWrapper in_nvte(NVTE_NVFP4_1D_SCALING);
    in_nvte.set_rowwise_data(a_data.data_ptr(), DType::kFloat4E2M1, a_data_shape);
    in_nvte.set_rowwise_scale_inv(a_sf.data_ptr(), DType::kFloat8E4M3, a_sf_shape);

    TensorWrapper out_nvte(NVTE_NVFP4_1D_SCALING);
    out_nvte.set_rowwise_data(a_data.data_ptr(), DType::kFloat4E2M1, a_data_shape);
    out_nvte.set_rowwise_scale_inv(a_sf_swizzled.data_ptr(), DType::kFloat8E4M3, a_sf_shape);
    out_nvte.set_with_gemm_swizzled_scales(true);

    nvte_swizzle_scaling_factors(in_nvte.data(), out_nvte.data(), stream);
  }
  {
    TensorWrapper in_nvte(NVTE_NVFP4_1D_SCALING);
    in_nvte.set_rowwise_data(b_data.data_ptr(), DType::kFloat4E2M1, b_data_shape);
    in_nvte.set_rowwise_scale_inv(b_sf.data_ptr(), DType::kFloat8E4M3, b_sf_shape);

    TensorWrapper out_nvte(NVTE_NVFP4_1D_SCALING);
    out_nvte.set_rowwise_data(b_data.data_ptr(), DType::kFloat4E2M1, b_data_shape);
    out_nvte.set_rowwise_scale_inv(b_sf_swizzled.data_ptr(), DType::kFloat8E4M3, b_sf_shape);
    out_nvte.set_with_gemm_swizzled_scales(true);

    nvte_swizzle_scaling_factors(in_nvte.data(), out_nvte.data(), stream);
  }

  // Pin operand amaxes to 1.0 so cuBLAS-internal alpha cancels the 2688^2
  // inner-SF factor. Cache one fp32 "1.0" tensor per device to avoid the
  // ~30-50us per-call cost of at::ones({1}) at small shapes.
  static std::array<at::Tensor, 16> s_amax_one_cache;
  static std::array<std::once_flag, 16> s_amax_one_init;
  const int dev_idx = a_data.device().index();
  TORCH_CHECK(dev_idx >= 0 && dev_idx < static_cast<int>(s_amax_one_cache.size()),
              "nvfp4_per_token_gemm: unexpected device index ", dev_idx);
  std::call_once(s_amax_one_init[dev_idx], [&]() {
    auto fp32_opts = a_data.options().dtype(at::kFloat);
    s_amax_one_cache[dev_idx] = at::ones({1}, fp32_opts);
  });
  at::Tensor& amax_one = s_amax_one_cache[dev_idx];

  // Assemble A's NVTE tensor: NVFP4_1D_SCALING + swizzled SF + amax=1.0.
  TensorWrapper a_te(NVTE_NVFP4_1D_SCALING);
  a_te.set_rowwise_data(a_data.data_ptr(), DType::kFloat4E2M1, a_data_shape);
  a_te.set_rowwise_scale_inv(a_sf_swizzled.data_ptr(), DType::kFloat8E4M3, a_sf_shape);
  a_te.set_amax(amax_one.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  a_te.set_with_gemm_swizzled_scales(true);

  TensorWrapper b_te(NVTE_NVFP4_1D_SCALING);
  b_te.set_rowwise_data(b_data.data_ptr(), DType::kFloat4E2M1, b_data_shape);
  b_te.set_rowwise_scale_inv(b_sf_swizzled.data_ptr(), DType::kFloat8E4M3, b_sf_shape);
  b_te.set_amax(amax_one.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  b_te.set_with_gemm_swizzled_scales(true);

  TensorWrapper d_te = makeTransformerEngineTensor(
      d.data_ptr(), std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)},
      DType::kBFloat16);

  TensorWrapper workspace_te = makeTransformerEngineTensor(
      workspace.data_ptr(), std::vector<size_t>{static_cast<size_t>(workspace.numel())},
      DType::kByte);

  // Operands SWAPPED so cuBLAS column-major D = op(B) @ op(A) matches the
  // row-major (M, N) PyTorch expects. transa=T forced (NVFP4 is TN-only).
  // C and D alias (no separate accumulator).
  const float alpha_f = static_cast<float>(alpha);
  const float beta_f = static_cast<float>(beta);
  nvte_cublas_gemm_v2(/*transa=*/1, /*transb=*/0, &alpha_f,
                      b_te.data(),  // cuBLAS-A := caller's B (N, K)
                      a_te.data(),  // cuBLAS-B := caller's A (M, K)
                      &beta_f, d_te.data(), d_te.data(), workspace_te.data(),
                      /*config=*/nullptr, stream);

  // Per-row * per-col post-scale to recover C_true from D_cublas.
  TensorWrapper ra_te = makeTransformerEngineTensor(
      a_row_amax.data_ptr(), std::vector<size_t>{static_cast<size_t>(m)}, DType::kFloat32);
  TensorWrapper rb_te = makeTransformerEngineTensor(
      b_row_amax.data_ptr(), std::vector<size_t>{static_cast<size_t>(n)}, DType::kFloat32);

  nvte_nvfp4_per_token_post_scale(d_te.data(), ra_te.data(), rb_te.data(), stream);
}

// Per-tensor twin of nvfp4_per_token_gemm: scalar amax goes through cuBLAS's
// own amax slot (no post-scale). Bench-only apples-to-apples baseline.
void nvfp4_per_tensor_gemm(const at::Tensor& a_data, const at::Tensor& b_data,
                           const at::Tensor& a_sf, const at::Tensor& b_sf, const at::Tensor& a_amax,
                           const at::Tensor& b_amax, at::Tensor d, const at::Tensor& workspace,
                           int64_t m, int64_t n, int64_t k, double alpha, double beta) {
  TORCH_CHECK(a_data.is_cuda() && b_data.is_cuda() && a_sf.is_cuda() && b_sf.is_cuda() &&
                  a_amax.is_cuda() && b_amax.is_cuda() && d.is_cuda() && workspace.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(a_data.is_contiguous() && b_data.is_contiguous() && a_sf.is_contiguous() &&
                  b_sf.is_contiguous() && a_amax.is_contiguous() && b_amax.is_contiguous() &&
                  d.is_contiguous() && workspace.is_contiguous(),
              "All tensors must be contiguous");
  TORCH_CHECK(a_data.scalar_type() == at::ScalarType::Byte, "a_data must be uint8 (FP4 packed)");
  TORCH_CHECK(b_data.scalar_type() == at::ScalarType::Byte, "b_data must be uint8 (FP4 packed)");
  TORCH_CHECK(a_sf.scalar_type() == at::ScalarType::Byte, "a_sf must be uint8 (FP8 e4m3)");
  TORCH_CHECK(b_sf.scalar_type() == at::ScalarType::Byte, "b_sf must be uint8 (FP8 e4m3)");
  TORCH_CHECK(a_amax.scalar_type() == at::ScalarType::Float, "a_amax must be float32");
  TORCH_CHECK(b_amax.scalar_type() == at::ScalarType::Float, "b_amax must be float32");
  TORCH_CHECK(d.scalar_type() == at::ScalarType::BFloat16, "d must be bfloat16");
  TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Byte, "workspace must be uint8");

  TORCH_CHECK(a_data.dim() == 2 && b_data.dim() == 2 && d.dim() == 2, "a_data/b_data/d must be 2D");
  TORCH_CHECK(a_data.size(0) == m && a_data.size(1) * 2 == k,
              "a_data shape mismatch: expected (M=", m, ", K/2=", k / 2, "), got (", a_data.size(0),
              ", ", a_data.size(1), ")");
  TORCH_CHECK(b_data.size(0) == n && b_data.size(1) * 2 == k,
              "b_data shape mismatch: expected (N=", n, ", K/2=", k / 2, "), got (", b_data.size(0),
              ", ", b_data.size(1), ")");
  TORCH_CHECK(d.size(0) == m && d.size(1) == n, "d shape mismatch: expected (M=", m, ", N=", n,
              "), got (", d.size(0), ", ", d.size(1), ")");

  TORCH_CHECK(k % 16 == 0, "k must be a multiple of 16 (NVFP4 inner SFVecSize)");
  TORCH_CHECK(a_sf.numel() == static_cast<int64_t>(m * k / 16),
              "a_sf numel mismatch: expected M*K/16=", m * k / 16, ", got ", a_sf.numel());
  TORCH_CHECK(b_sf.numel() == static_cast<int64_t>(n * k / 16),
              "b_sf numel mismatch: expected N*K/16=", n * k / 16, ", got ", b_sf.numel());
  TORCH_CHECK(a_amax.numel() == 1, "a_amax must be a scalar (numel=1), got ", a_amax.numel());
  TORCH_CHECK(b_amax.numel() == 1, "b_amax must be a scalar (numel=1), got ", b_amax.numel());

  TORCH_CHECK(static_cast<float>(beta) == 0.0f,
              "nvfp4_per_tensor_gemm: beta != 0 not yet supported. Got beta=", beta);

  const auto stream = at::cuda::getCurrentCUDAStream();

  const std::vector<size_t> a_data_shape = {static_cast<size_t>(m), static_cast<size_t>(k)};
  const std::vector<size_t> b_data_shape = {static_cast<size_t>(n), static_cast<size_t>(k)};
  const std::vector<size_t> a_sf_shape = {static_cast<size_t>(m), static_cast<size_t>(k / 16)};
  const std::vector<size_t> b_sf_shape = {static_cast<size_t>(n), static_cast<size_t>(k / 16)};

  auto byte_opts = a_sf.options().dtype(at::kByte);
  at::Tensor a_sf_swizzled = at::empty({a_sf.numel()}, byte_opts);
  at::Tensor b_sf_swizzled = at::empty({b_sf.numel()}, byte_opts);

  {
    TensorWrapper in_nvte(NVTE_NVFP4_1D_SCALING);
    in_nvte.set_rowwise_data(a_data.data_ptr(), DType::kFloat4E2M1, a_data_shape);
    in_nvte.set_rowwise_scale_inv(a_sf.data_ptr(), DType::kFloat8E4M3, a_sf_shape);

    TensorWrapper out_nvte(NVTE_NVFP4_1D_SCALING);
    out_nvte.set_rowwise_data(a_data.data_ptr(), DType::kFloat4E2M1, a_data_shape);
    out_nvte.set_rowwise_scale_inv(a_sf_swizzled.data_ptr(), DType::kFloat8E4M3, a_sf_shape);
    out_nvte.set_with_gemm_swizzled_scales(true);

    nvte_swizzle_scaling_factors(in_nvte.data(), out_nvte.data(), stream);
  }
  {
    TensorWrapper in_nvte(NVTE_NVFP4_1D_SCALING);
    in_nvte.set_rowwise_data(b_data.data_ptr(), DType::kFloat4E2M1, b_data_shape);
    in_nvte.set_rowwise_scale_inv(b_sf.data_ptr(), DType::kFloat8E4M3, b_sf_shape);

    TensorWrapper out_nvte(NVTE_NVFP4_1D_SCALING);
    out_nvte.set_rowwise_data(b_data.data_ptr(), DType::kFloat4E2M1, b_data_shape);
    out_nvte.set_rowwise_scale_inv(b_sf_swizzled.data_ptr(), DType::kFloat8E4M3, b_sf_shape);
    out_nvte.set_with_gemm_swizzled_scales(true);

    nvte_swizzle_scaling_factors(in_nvte.data(), out_nvte.data(), stream);
  }

  // Per-tensor amaxes go in the amax slot; cuBLAS LT folds them into alpha.
  TensorWrapper a_te(NVTE_NVFP4_1D_SCALING);
  a_te.set_rowwise_data(a_data.data_ptr(), DType::kFloat4E2M1, a_data_shape);
  a_te.set_rowwise_scale_inv(a_sf_swizzled.data_ptr(), DType::kFloat8E4M3, a_sf_shape);
  a_te.set_amax(a_amax.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  a_te.set_with_gemm_swizzled_scales(true);

  TensorWrapper b_te(NVTE_NVFP4_1D_SCALING);
  b_te.set_rowwise_data(b_data.data_ptr(), DType::kFloat4E2M1, b_data_shape);
  b_te.set_rowwise_scale_inv(b_sf_swizzled.data_ptr(), DType::kFloat8E4M3, b_sf_shape);
  b_te.set_amax(b_amax.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  b_te.set_with_gemm_swizzled_scales(true);

  TensorWrapper d_te = makeTransformerEngineTensor(
      d.data_ptr(), std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)},
      DType::kBFloat16);

  TensorWrapper workspace_te = makeTransformerEngineTensor(
      workspace.data_ptr(), std::vector<size_t>{static_cast<size_t>(workspace.numel())},
      DType::kByte);

  // Operand swap: see nvfp4_per_token_gemm.
  const float alpha_f = static_cast<float>(alpha);
  const float beta_f = static_cast<float>(beta);
  nvte_cublas_gemm_v2(/*transa=*/1, /*transb=*/0, &alpha_f,
                      b_te.data(),  // cuBLAS-A := caller's B (N, K)
                      a_te.data(),  // cuBLAS-B := caller's A (M, K)
                      &beta_f, d_te.data(), d_te.data(), workspace_te.data(),
                      /*config=*/nullptr, stream);
  // No post-scale: per-tensor amaxes already folded into cuBLAS-internal alpha.
}

// Grouped (multi-tensor) per-token quantize. Each direction takes 3 lists
// of per-split tensors; ``split_sections[i] = M_i`` (% 128, sum = sum_M).
// Disabled direction's lists are ignored.
namespace {

void build_per_token_output_wrapper(TensorWrapper& out_te, int64_t M_i, int64_t K, bool rowwise,
                                    bool columnwise, const at::Tensor& q_row,
                                    const at::Tensor& s_dec_row, const at::Tensor& row_amax,
                                    const at::Tensor& q_col, const at::Tensor& s_dec_col,
                                    const at::Tensor& col_amax) {
  if (rowwise) {
    TORCH_CHECK(q_row.is_cuda() && q_row.is_contiguous(), "q_row must be a contiguous CUDA tensor");
    TORCH_CHECK(s_dec_row.is_cuda() && s_dec_row.is_contiguous(),
                "s_dec_row must be a contiguous CUDA tensor");
    TORCH_CHECK(row_amax.is_cuda() && row_amax.is_contiguous(),
                "row_amax must be a contiguous CUDA tensor");
    TORCH_CHECK(q_row.scalar_type() == at::ScalarType::Byte, "q_row must be uint8");
    TORCH_CHECK(s_dec_row.scalar_type() == at::ScalarType::Byte, "s_dec_row must be uint8");
    TORCH_CHECK(row_amax.scalar_type() == at::ScalarType::Float, "row_amax must be fp32");
    TORCH_CHECK(q_row.numel() == M_i * K / 2, "q_row numel mismatch for split: expected ",
                M_i * K / 2, ", got ", q_row.numel());
    TORCH_CHECK(s_dec_row.numel() == M_i * K / 16, "s_dec_row numel mismatch for split");
    TORCH_CHECK(row_amax.numel() == M_i, "row_amax numel mismatch for split");
    out_te.set_rowwise_data(q_row.data_ptr(), DType::kFloat4E2M1,
                            std::vector<size_t>{static_cast<size_t>(M_i), static_cast<size_t>(K)});
    out_te.set_rowwise_scale_inv(
        s_dec_row.data_ptr(), DType::kFloat8E4M3,
        std::vector<size_t>{static_cast<size_t>(M_i), static_cast<size_t>(K / 16)});
    out_te.set_amax(row_amax.data_ptr(), DType::kFloat32,
                    std::vector<size_t>{static_cast<size_t>(M_i)});
  }
  if (columnwise) {
    TORCH_CHECK(q_col.is_cuda() && q_col.is_contiguous(), "q_col must be a contiguous CUDA tensor");
    TORCH_CHECK(s_dec_col.is_cuda() && s_dec_col.is_contiguous(),
                "s_dec_col must be a contiguous CUDA tensor");
    TORCH_CHECK(col_amax.is_cuda() && col_amax.is_contiguous(),
                "col_amax must be a contiguous CUDA tensor");
    TORCH_CHECK(q_col.scalar_type() == at::ScalarType::Byte, "q_col must be uint8");
    TORCH_CHECK(s_dec_col.scalar_type() == at::ScalarType::Byte, "s_dec_col must be uint8");
    TORCH_CHECK(col_amax.scalar_type() == at::ScalarType::Float, "col_amax must be fp32");
    TORCH_CHECK(q_col.numel() == K * M_i / 2, "q_col numel mismatch for split");
    TORCH_CHECK(s_dec_col.numel() == K * M_i / 16, "s_dec_col numel mismatch for split");
    TORCH_CHECK(col_amax.numel() == K, "col_amax numel mismatch for split");
    out_te.set_columnwise_data(
        q_col.data_ptr(), DType::kFloat4E2M1,
        std::vector<size_t>{static_cast<size_t>(K), static_cast<size_t>(M_i)});
    out_te.set_columnwise_scale_inv(
        s_dec_col.data_ptr(), DType::kFloat8E4M3,
        std::vector<size_t>{static_cast<size_t>(K), static_cast<size_t>(M_i / 16)});
    out_te.set_columnwise_amax(col_amax.data_ptr(), DType::kFloat32,
                               std::vector<size_t>{static_cast<size_t>(K)});
  }
}

DType resolve_input_dtype(const at::Tensor& input) {
  if (input.scalar_type() == at::ScalarType::BFloat16) return DType::kBFloat16;
  if (input.scalar_type() == at::ScalarType::Float) return DType::kFloat32;
  if (input.scalar_type() == at::ScalarType::Half) return DType::kFloat16;
  TORCH_CHECK(false, "input dtype must be bf16/fp16/fp32, got ", input.scalar_type());
  return DType::kBFloat16;  // unreachable
}

}  // namespace

void nvfp4_per_token_group_quantize(
    const at::Tensor& input, const std::vector<int64_t>& split_sections,
    std::vector<at::Tensor> q_row_list, std::vector<at::Tensor> s_dec_row_list,
    std::vector<at::Tensor> row_amax_list, std::vector<at::Tensor> q_col_list,
    std::vector<at::Tensor> s_dec_col_list, std::vector<at::Tensor> col_amax_list, bool rowwise,
    bool columnwise, bool with_rht, int64_t random_sign_mask_t) {
  TORCH_CHECK(rowwise || columnwise, "At least one of rowwise/columnwise must be True.");
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be a contiguous CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  const int64_t sum_M = input.size(0);
  const int64_t K = input.size(1);
  const size_t num_tensors = split_sections.size();
  TORCH_CHECK(num_tensors > 0, "split_sections must not be empty");

  // Sum + 64-multiple constraint.
  int64_t acc = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    TORCH_CHECK(split_sections[i] >= 0, "split_sections[", i, "] must be non-negative");
    TORCH_CHECK(split_sections[i] % 64 == 0, "split_sections[", i, "] = ", split_sections[i],
                " must be a multiple of 64");
    acc += split_sections[i];
  }
  TORCH_CHECK(acc == sum_M, "sum(split_sections) = ", acc, " must equal input.size(0) = ", sum_M);

  if (rowwise) {
    TORCH_CHECK(q_row_list.size() == num_tensors, "q_row_list size mismatch");
    TORCH_CHECK(s_dec_row_list.size() == num_tensors, "s_dec_row_list size mismatch");
    TORCH_CHECK(row_amax_list.size() == num_tensors, "row_amax_list size mismatch");
  }
  if (columnwise) {
    TORCH_CHECK(q_col_list.size() == num_tensors, "q_col_list size mismatch");
    TORCH_CHECK(s_dec_col_list.size() == num_tensors, "s_dec_col_list size mismatch");
    TORCH_CHECK(col_amax_list.size() == num_tensors, "col_amax_list size mismatch");
  }

  const DType in_dtype = resolve_input_dtype(input);
  const auto stream = at::cuda::getCurrentCUDAStream();

  TensorWrapper in_te = makeTransformerEngineTensor(
      input.data_ptr(), std::vector<size_t>{static_cast<size_t>(sum_M), static_cast<size_t>(K)},
      in_dtype);

  // One TensorWrapper per split; raw NVTETensor handles go into `handles`.
  std::vector<TensorWrapper> wrappers;
  wrappers.reserve(num_tensors);
  std::vector<NVTETensor> handles;
  handles.reserve(num_tensors);
  std::vector<size_t> split_sections_sz(num_tensors);

  at::Tensor empty_dummy;  // for slots we don't populate
  for (size_t i = 0; i < num_tensors; ++i) {
    const int64_t M_i = split_sections[i];
    split_sections_sz[i] = static_cast<size_t>(M_i);
    wrappers.emplace_back(NVTE_NVFP4_1D_SCALING);
    if (M_i == 0) {
      handles.push_back(wrappers.back().data());
      continue;  // empty split is allowed (skipped inside the kernel)
    }
    build_per_token_output_wrapper(
        wrappers.back(), M_i, K, rowwise, columnwise, rowwise ? q_row_list[i] : empty_dummy,
        rowwise ? s_dec_row_list[i] : empty_dummy, rowwise ? row_amax_list[i] : empty_dummy,
        columnwise ? q_col_list[i] : empty_dummy, columnwise ? s_dec_col_list[i] : empty_dummy,
        columnwise ? col_amax_list[i] : empty_dummy);
    handles.push_back(wrappers.back().data());
  }

  nvte_group_nvfp4_per_token_quantize(in_te.data(), handles.data(), split_sections_sz.data(),
                                      num_tensors, rowwise, columnwise, static_cast<int>(with_rht),
                                      static_cast<int>(random_sign_mask_t), stream);
}

// Amax-only grouped variant (K1 only); for allReduce-before-cast flows.
void nvfp4_per_token_group_amax(const at::Tensor& input, const std::vector<int64_t>& split_sections,
                                std::vector<at::Tensor> row_amax_list,
                                std::vector<at::Tensor> col_amax_list, bool rowwise,
                                bool columnwise, bool with_rht, int64_t random_sign_mask_t) {
  TORCH_CHECK(rowwise || columnwise, "At least one of rowwise/columnwise must be True.");
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be a contiguous CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  const int64_t sum_M = input.size(0);
  const int64_t K = input.size(1);
  const size_t num_tensors = split_sections.size();
  TORCH_CHECK(num_tensors > 0, "split_sections must not be empty");
  int64_t acc = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    TORCH_CHECK(split_sections[i] % 64 == 0, "split_sections[", i, "] must be a multiple of 64");
    acc += split_sections[i];
  }
  TORCH_CHECK(acc == sum_M, "sum(split_sections) must equal input.size(0)");
  if (rowwise) TORCH_CHECK(row_amax_list.size() == num_tensors, "row_amax_list size mismatch");
  if (columnwise) TORCH_CHECK(col_amax_list.size() == num_tensors, "col_amax_list size mismatch");

  const DType in_dtype = resolve_input_dtype(input);
  const auto stream = at::cuda::getCurrentCUDAStream();

  TensorWrapper in_te = makeTransformerEngineTensor(
      input.data_ptr(), std::vector<size_t>{static_cast<size_t>(sum_M), static_cast<size_t>(K)},
      in_dtype);

  std::vector<TensorWrapper> wrappers;
  wrappers.reserve(num_tensors);
  std::vector<NVTETensor> handles;
  handles.reserve(num_tensors);
  std::vector<size_t> split_sections_sz(num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    const int64_t M_i = split_sections[i];
    split_sections_sz[i] = static_cast<size_t>(M_i);
    wrappers.emplace_back(NVTE_NVFP4_1D_SCALING);
    if (M_i == 0) {
      handles.push_back(wrappers.back().data());
      continue;
    }
    if (rowwise) {
      const at::Tensor& ra = row_amax_list[i];
      TORCH_CHECK(ra.is_cuda() && ra.scalar_type() == at::ScalarType::Float, "bad row_amax");
      TORCH_CHECK(ra.numel() == M_i, "row_amax numel mismatch");
      wrappers.back().set_amax(ra.data_ptr(), DType::kFloat32,
                               std::vector<size_t>{static_cast<size_t>(M_i)});
    }
    if (columnwise) {
      const at::Tensor& ca = col_amax_list[i];
      TORCH_CHECK(ca.is_cuda() && ca.scalar_type() == at::ScalarType::Float, "bad col_amax");
      TORCH_CHECK(ca.numel() == K, "col_amax numel mismatch");
      wrappers.back().set_columnwise_amax(ca.data_ptr(), DType::kFloat32,
                                          std::vector<size_t>{static_cast<size_t>(K)});
    }
    handles.push_back(wrappers.back().data());
  }

  nvte_group_nvfp4_per_token_amax(in_te.data(), handles.data(), split_sections_sz.data(),
                                  num_tensors, rowwise, columnwise, static_cast<int>(with_rht),
                                  static_cast<int>(random_sign_mask_t), stream);
}

// BULK grouped per-token quantize: alloc + view + dispatch in ONE C++ call.
// Returns 6 per-split tensor lists (s_dec_* pre-cast to Float8_e4m3fn).
// Byte-equal to the prior Python wrap (saves ~70-90us at N=8).
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>,
           std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>>
nvfp4_per_token_group_quantize_bulk(const at::Tensor& input,
                                    const std::vector<int64_t>& split_sections, bool rowwise,
                                    bool columnwise, bool with_rht, int64_t random_sign_mask_t) {
  // Validation mirrors _validate_per_token_group_input in Python.
  TORCH_CHECK(rowwise || columnwise, "At least one of rowwise/columnwise must be True.");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "x_concat must be contiguous (row-major)");
  TORCH_CHECK(input.dim() == 2, "nvfp4_per_token_group_quantize expects a 2D input, got ",
              input.dim(), "D");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16,
              "Per-token grouped kernel is bf16-only; got dtype ", input.scalar_type());

  const int64_t sum_M = input.size(0);
  const int64_t K = input.size(1);
  constexpr int64_t kPerTokenTile = 128;
  constexpr int64_t kBlockK = 16;

  TORCH_CHECK(K % kPerTokenTile == 0, "Per-token grouped kernel requires K % ", kPerTokenTile,
              " == 0; got K=", K);

  const size_t num_tensors = split_sections.size();
  TORCH_CHECK(num_tensors > 0, "split_sections must not be empty");
  TORCH_CHECK(num_tensors <= 64, "num_tensors must be <= 64 (kernel arg-struct cap); got ",
              num_tensors);

  int64_t acc = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    const int64_t M_i = split_sections[i];
    TORCH_CHECK(M_i > 0, "split_sections[", i, "] must be > 0, got ", M_i);
    TORCH_CHECK(M_i % kPerTokenTile == 0, "split_sections[", i, "] = ", M_i,
                " must be a multiple of ", kPerTokenTile);
    acc += M_i;
  }
  TORCH_CHECK(acc == sum_M, "sum(split_sections) = ", acc, " must equal input.size(0) = ", sum_M);

  // Bulk allocation: one at::empty per output type, covers all splits.
  auto opts_u8 = input.options().dtype(at::kByte);
  auto opts_f32 = input.options().dtype(at::kFloat);

  at::Tensor q_row_bulk, s_dec_row_bulk, row_amax_bulk;
  at::Tensor q_col_bulk, s_dec_col_bulk, col_amax_bulk;

  if (rowwise) {
    q_row_bulk = at::empty({sum_M, K / 2}, opts_u8);
    s_dec_row_bulk = at::empty({sum_M, K / kBlockK}, opts_u8);
    row_amax_bulk = at::empty({sum_M}, opts_f32);
  }
  if (columnwise) {
    q_col_bulk = at::empty({K * sum_M / 2}, opts_u8);
    s_dec_col_bulk = at::empty({K * sum_M / kBlockK}, opts_u8);
    col_amax_bulk = at::empty({static_cast<int64_t>(num_tensors), K}, opts_f32);
  }

  // Per-split views built in C++; s_dec_* kept in both uint8 (for binding)
  // and fp8_e4m3fn (returned to Python directly).
  std::vector<at::Tensor> q_row_list, s_dec_row_u8_list, row_amax_list;
  std::vector<at::Tensor> q_col_list, s_dec_col_u8_list, col_amax_list;
  std::vector<at::Tensor> s_dec_row_fp8_list, s_dec_col_fp8_list;
  if (rowwise) {
    q_row_list.reserve(num_tensors);
    s_dec_row_u8_list.reserve(num_tensors);
    row_amax_list.reserve(num_tensors);
    s_dec_row_fp8_list.reserve(num_tensors);
  }
  if (columnwise) {
    q_col_list.reserve(num_tensors);
    s_dec_col_u8_list.reserve(num_tensors);
    col_amax_list.reserve(num_tensors);
    s_dec_col_fp8_list.reserve(num_tensors);
  }

  int64_t m_off = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    const int64_t M_i = split_sections[i];
    if (rowwise) {
      q_row_list.emplace_back(q_row_bulk.narrow(0, m_off, M_i));
      s_dec_row_u8_list.emplace_back(s_dec_row_bulk.narrow(0, m_off, M_i));
      row_amax_list.emplace_back(row_amax_bulk.narrow(0, m_off, M_i));
      s_dec_row_fp8_list.emplace_back(s_dec_row_u8_list.back().view(at::kFloat8_e4m3fn));
    }
    if (columnwise) {
      auto q_col_flat = q_col_bulk.narrow(0, K * m_off / 2, K * M_i / 2);
      q_col_list.emplace_back(q_col_flat.view({K, M_i / 2}));
      auto s_dec_col_flat = s_dec_col_bulk.narrow(0, K * m_off / kBlockK, K * M_i / kBlockK);
      s_dec_col_u8_list.emplace_back(s_dec_col_flat.view({K, M_i / kBlockK}));
      col_amax_list.emplace_back(col_amax_bulk.select(0, static_cast<int64_t>(i)));
      s_dec_col_fp8_list.emplace_back(s_dec_col_u8_list.back().view(at::kFloat8_e4m3fn));
    }
    m_off += M_i;
  }

  // Dispatch K1+K2 grouped kernel via the same C-API the thin entry uses.
  const auto stream = at::cuda::getCurrentCUDAStream();
  TensorWrapper in_te = makeTransformerEngineTensor(
      input.data_ptr(), std::vector<size_t>{static_cast<size_t>(sum_M), static_cast<size_t>(K)},
      DType::kBFloat16);

  std::vector<TensorWrapper> wrappers;
  wrappers.reserve(num_tensors);
  std::vector<NVTETensor> handles;
  handles.reserve(num_tensors);
  std::vector<size_t> split_sections_sz(num_tensors);

  at::Tensor empty_dummy;
  for (size_t i = 0; i < num_tensors; ++i) {
    const int64_t M_i = split_sections[i];
    split_sections_sz[i] = static_cast<size_t>(M_i);
    wrappers.emplace_back(NVTE_NVFP4_1D_SCALING);
    build_per_token_output_wrapper(
        wrappers.back(), M_i, K, rowwise, columnwise, rowwise ? q_row_list[i] : empty_dummy,
        rowwise ? s_dec_row_u8_list[i] : empty_dummy, rowwise ? row_amax_list[i] : empty_dummy,
        columnwise ? q_col_list[i] : empty_dummy, columnwise ? s_dec_col_u8_list[i] : empty_dummy,
        columnwise ? col_amax_list[i] : empty_dummy);
    handles.push_back(wrappers.back().data());
  }

  nvte_group_nvfp4_per_token_quantize(in_te.data(), handles.data(), split_sections_sz.data(),
                                      num_tensors, rowwise, columnwise, static_cast<int>(with_rht),
                                      static_cast<int>(random_sign_mask_t), stream);

  return std::make_tuple(std::move(q_row_list), std::move(s_dec_row_fp8_list),
                         std::move(row_amax_list), std::move(q_col_list),
                         std::move(s_dec_col_fp8_list), std::move(col_amax_list));
}

}  // namespace transformer_engine::pytorch
