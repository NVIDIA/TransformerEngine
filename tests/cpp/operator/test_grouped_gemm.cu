/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <transformer_engine/cast.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/recipe.h>
#include <transformer_engine/swizzle.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/transpose.h>

#include "../test_common.h"
#include "util/cuda_runtime.h"

using namespace transformer_engine;
using namespace test;

namespace {

// std::nullopt means BF16 (no scaling); any other value is the scaling mode for FP8/NVFP4.
using InputRecipe = std::optional<NVTEScalingMode>;

inline const char* recipe_name(const InputRecipe& r) {
  if (!r.has_value()) return "BF16";
  switch (*r) {
    case NVTE_DELAYED_TENSOR_SCALING: return "FP8Current";
    case NVTE_MXFP8_1D_SCALING:       return "MXFP8";
    case NVTE_NVFP4_1D_SCALING:       return "NVFP4";
    case NVTE_BLOCK_SCALING_1D:       return "FP8BlockScaling";
    default:                          return "Unknown";
  }
}

enum class ShapeCase {
  kAllSame,
  kSameFirst,
  kSameLast,
  kAllDifferent,
  // Uniform shapes with dims NOT multiples of 128 — exercises scale_inv padding offsets.
  kUnalignedAllSame,
};

Tensor make_fp8_operand(const std::string& name, const std::vector<size_t>& shape) {
  Tensor input_fp32(name + "_fp32", shape, DType::kFloat32);

  const size_t numel = shape[0] * shape[1];
  std::vector<float> data(numel);
  std::mt19937 gen(std::hash<std::string>{}(name));
  // Random mean and stddev -> different amax per tensor -> different scales
  std::uniform_real_distribution<float> param_dis(0.1f, 10.0f);
  float mean = param_dis(gen);
  float stddev = param_dis(gen);
  std::normal_distribution<float> dis(mean, stddev);
  for (size_t i = 0; i < numel; ++i) {
    data[i] = dis(gen);
  }
  NVTE_CHECK_CUDA(cudaMemcpy(input_fp32.rowwise_dptr(), data.data(),
                             numel * sizeof(float), cudaMemcpyHostToDevice));

  Tensor fp8(name, shape, TypeInfo<fp8e4m3>::dtype, true, true, NVTE_DELAYED_TENSOR_SCALING);

  nvte_compute_amax(input_fp32.data(), fp8.data(), 0);
  QuantizationConfigWrapper config;
  nvte_compute_scale_from_amax(fp8.data(), config, 0);
  nvte_quantize(input_fp32.data(), fp8.data(), 0);
  return fp8;
}

Tensor make_bf16_operand(const std::string& name, const std::vector<size_t>& shape) {
  Tensor t(name, shape, DType::kBFloat16);
  const size_t numel = shape[0] * shape[1];
  std::vector<__nv_bfloat16> ones(numel, __float2bfloat16(1.0f));
  NVTE_CHECK_CUDA(cudaMemcpy(t.rowwise_dptr(), ones.data(),
                             numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  return t;
}

// Creates an MXFP8 operand with the correct data layout for GEMM.
// MXFP8 GEMM requirements (scales are along K dimension):
//   A transposed     -> needs rowwise data/scales
//   A non-transposed -> needs columnwise data/scales
//   B transposed     -> needs columnwise data/scales
//   B non-transposed -> needs rowwise data/scales
Tensor make_mxfp8_operand(const std::string& name, const std::vector<size_t>& shape,
                          bool is_A, bool transposed) {
  // Determine which data layout we need
  bool use_rowwise, use_colwise;
  if (is_A) {
    // A: transposed -> rowwise, non-transposed -> columnwise
    use_rowwise = transposed;
    use_colwise = !transposed;
  } else {
    // B: transposed -> columnwise, non-transposed -> rowwise (opposite of A!)
    use_rowwise = !transposed;
    use_colwise = transposed;
  }

  // Create BF16 input with random data
  Tensor input_bf16(name + "_bf16", shape, DType::kBFloat16);
  fillUniform(&input_bf16);

  // Create MXFP8 tensor with only the required data layout
  Tensor mxfp8(name, shape, TypeInfo<fp8e4m3>::dtype, use_rowwise, use_colwise,
               NVTE_MXFP8_1D_SCALING);

  // Quantize BF16 -> MXFP8
  nvte_quantize(input_bf16.data(), mxfp8.data(), 0);

  // Create output tensor for swizzled scales (same data shape, same layout)
  Tensor mxfp8_swizzled(name + "_swizzled", shape, TypeInfo<fp8e4m3>::dtype,
                        use_rowwise, use_colwise, NVTE_MXFP8_1D_SCALING);
  mxfp8_swizzled.set_with_gemm_swizzled_scales(true);  // Must be set BEFORE swizzle call

  // Copy quantized data from mxfp8 to mxfp8_swizzled
  if (use_rowwise) {
    size_t data_bytes = test::bytes(mxfp8.rowwise_shape(), mxfp8.dtype());
    NVTE_CHECK_CUDA(cudaMemcpy(mxfp8_swizzled.rowwise_dptr(), mxfp8.rowwise_dptr(),
                               data_bytes, cudaMemcpyDeviceToDevice));
  }
  if (use_colwise) {
    size_t data_bytes = test::bytes(mxfp8.columnwise_shape(), mxfp8.dtype());
    NVTE_CHECK_CUDA(cudaMemcpy(mxfp8_swizzled.columnwise_dptr(), mxfp8.columnwise_dptr(),
                               data_bytes, cudaMemcpyDeviceToDevice));
  }

  // Swizzle scales for GEMM
  nvte_swizzle_scaling_factors(mxfp8.data(), mxfp8_swizzled.data(), 0);

  // Sync to ensure operations are complete
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  return mxfp8_swizzled;
}

// Helper: quantize an already-filled BF16 tensor to NVFP4 rowwise-only, swizzle scales,
// return swizzled tensor. Caller owns input_bf16. When nvfp4_2d is true, amax is computed
// per 16x16 block (the scale layout fed to cuBLAS — VEC16 UE4M3 — is unchanged; values
// just repeat across the 16 rows of each block).
Tensor make_nvfp4_rowwise(const std::string& name, Tensor& input_bf16,
                          const std::vector<size_t>& shape, bool nvfp4_2d) {
  Tensor nvfp4(name, shape, DType::kFloat4E2M1, /*rowwise=*/true, /*columnwise=*/false,
               NVTE_NVFP4_1D_SCALING);

  QuantizationConfigWrapper quant_config;
  quant_config.set_nvfp4_2d_quantization(nvfp4_2d);
  nvte_quantize_v2(input_bf16.data(), nvfp4.data(), quant_config, 0);

  Tensor nvfp4_sw(name + "_sw", shape, DType::kFloat4E2M1,
                  /*rowwise=*/true, /*columnwise=*/false, NVTE_NVFP4_1D_SCALING);
  nvfp4_sw.set_with_gemm_swizzled_scales(true);
  size_t data_bytes = test::bytes(nvfp4.rowwise_shape(), nvfp4.dtype());
  NVTE_CHECK_CUDA(cudaMemcpy(nvfp4_sw.rowwise_dptr(), nvfp4.rowwise_dptr(),
                             data_bytes, cudaMemcpyDeviceToDevice));
  nvte_swizzle_scaling_factors(nvfp4.data(), nvfp4_sw.data(), 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  return nvfp4_sw;
}

// Creates an NVFP4 operand with both rowwise and columnwise data, swizzled scales.
// NVFP4 "columnwise" data is the transposed tensor quantized rowwise; rowwise and
// columnwise share the SAME source BF16 input (transposed via nvte_transpose for the
// columnwise path) so the two layouts represent the same logical tensor.
// We can't fuse this into one quantize+swizzle pass:
//   - nvte_quantize_v2 supports filling both rowwise+columnwise in one call, but the
//     optimized NVFP4 kernel hard-fails on with_gemm_swizzled_scales=true and the
//     fallback hard-codes swizzled_scale=false (cast/dispatch/quantize.cuh).
//   - nvte_swizzle_scaling_factors hard-fails if a tensor has both rowwise and
//     columnwise scale_inv (swizzle/swizzle.cu).
Tensor make_nvfp4_operand(const std::string& name, const std::vector<size_t>& shape,
                          bool is_A, bool transposed, bool nvfp4_2d) {
  (void)is_A;
  (void)transposed;

  // Single BF16 input shared by both directions; GPU-side transpose for columnwise path.
  Tensor input_bf16(name + "_bf16", shape, DType::kBFloat16);
  fillUniform(&input_bf16);

  std::vector<size_t> t_shape = {shape[1], shape[0]};
  Tensor input_bf16_t(name + "_bf16_t", t_shape, DType::kBFloat16);
  nvte_transpose(input_bf16.data(), input_bf16_t.data(), 0);

  // 1. Rowwise: quantize + swizzle from input
  Tensor rowwise = make_nvfp4_rowwise(name + "_row", input_bf16, shape, nvfp4_2d);

  // 2. Columnwise: quantize + swizzle from transposed input, as rowwise of transposed shape
  Tensor colwise = make_nvfp4_rowwise(name + "_col", input_bf16_t, t_shape, nvfp4_2d);

  // 3. Assemble: both-layout tensor with rowwise from (1) and columnwise from (2)
  Tensor result(name, shape, DType::kFloat4E2M1, /*rowwise=*/true, /*columnwise=*/true,
                NVTE_NVFP4_1D_SCALING);
  result.set_with_gemm_swizzled_scales(true);

  // Copy rowwise data + scale from rowwise tensor
  {
    size_t data_bytes = test::bytes(rowwise.rowwise_shape(), rowwise.dtype());
    NVTE_CHECK_CUDA(cudaMemcpy(result.rowwise_dptr(), rowwise.rowwise_dptr(),
                               data_bytes, cudaMemcpyDeviceToDevice));
    size_t scale_bytes = test::bytes(rowwise.rowwise_scale_inv_shape(), DType::kFloat8E4M3);
    NVTE_CHECK_CUDA(cudaMemcpy(
        nvte_get_tensor_param(result.data(), kNVTERowwiseScaleInv).data_ptr,
        nvte_get_tensor_param(rowwise.data(), kNVTERowwiseScaleInv).data_ptr,
        scale_bytes, cudaMemcpyDeviceToDevice));
  }

  // Copy colwise data + scale from transposed-rowwise tensor
  // The rowwise data of transposed shape IS the columnwise data of original shape
  {
    size_t data_bytes = test::bytes(colwise.rowwise_shape(), colwise.dtype());
    NVTE_CHECK_CUDA(cudaMemcpy(result.columnwise_dptr(), colwise.rowwise_dptr(),
                               data_bytes, cudaMemcpyDeviceToDevice));
    size_t scale_bytes = test::bytes(colwise.rowwise_scale_inv_shape(), DType::kFloat8E4M3);
    NVTE_CHECK_CUDA(cudaMemcpy(
        nvte_get_tensor_param(result.data(), kNVTEColumnwiseScaleInv).data_ptr,
        nvte_get_tensor_param(colwise.data(), kNVTERowwiseScaleInv).data_ptr,
        scale_bytes, cudaMemcpyDeviceToDevice));
  }

  // Copy amax values (not pointers) so each Tensor stays sole owner of its amax buffer.
  {
    NVTEBasicTensor src = nvte_get_tensor_param(rowwise.data(), kNVTEAmax);
    NVTEBasicTensor dst = nvte_get_tensor_param(result.data(), kNVTEAmax);
    NVTE_CHECK_CUDA(cudaMemcpy(dst.data_ptr, src.data_ptr, sizeof(float),
                               cudaMemcpyDeviceToDevice));
  }
  {
    NVTEBasicTensor src = nvte_get_tensor_param(colwise.data(), kNVTEAmax);
    NVTEBasicTensor dst = nvte_get_tensor_param(result.data(), kNVTEColumnwiseAmax);
    NVTE_CHECK_CUDA(cudaMemcpy(dst.data_ptr, src.data_ptr, sizeof(float),
                               cudaMemcpyDeviceToDevice));
  }

  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  return result;
}

// Creates an FP8 block-scaling operand.
// FP8 block scaling on Hopper requires TN layout:
//   A transposed     -> needs rowwise data
//   A non-transposed -> needs columnwise data (will be flipped to T internally)
//   B transposed     -> needs columnwise data (will be flipped to N internally)
//   B non-transposed -> needs rowwise data
Tensor make_fp8_block_scaling_operand(const std::string& name, const std::vector<size_t>& shape,
                                      bool is_A, bool transposed) {
  // Determine which data layout we need (TN-only on Hopper)
  bool use_rowwise, use_colwise;
  if (is_A) {
    use_rowwise = transposed;
    use_colwise = !transposed;
  } else {
    use_rowwise = !transposed;
    use_colwise = transposed;
  }

  Tensor input_bf16(name + "_bf16", shape, DType::kBFloat16);
  fillUniform(&input_bf16);

  // Create FP8 block scaling tensor (1D scaling)
  Tensor fp8_bs(name, shape, TypeInfo<fp8e4m3>::dtype, use_rowwise, use_colwise,
                NVTE_BLOCK_SCALING_1D);

  // Quantize BF16 -> FP8 block scaling
  QuantizationConfigWrapper quant_config;
  quant_config.set_force_pow_2_scales(true);
  nvte_quantize_v2(input_bf16.data(), fp8_bs.data(), quant_config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  return fp8_bs;
}

struct TestParams {
  InputRecipe recipe;  // std::nullopt = BF16, otherwise the scaling mode.
  bool transa;
  bool transb;
  ShapeCase shape_case;
  bool use_null_c = false;  // When true, pass nullptr for C (valid when beta=0)
  bool nvfp4_2d = false;    // NVFP4-only: use 2D (16x16) amax instead of 1D (1x16).
  DType output_dtype = DType::kBFloat16;  // Implementation also accepts FP16 / FP32.
};

// Returns a vector of (M, N, K) tuples for each GEMM in the group.
// M - number of rows in output D
// N - number of columns in output D
// K - reduction dimension shared between A and B
std::vector<std::tuple<size_t, size_t, size_t>> make_shapes(ShapeCase scase) {
  switch (scase) {
    case ShapeCase::kAllSame:
      return {{128, 256, 384}, {128, 256, 384}, {128, 256, 384}};
    case ShapeCase::kSameFirst:
      // Same M (first dim), varying N and K
      return {{128, 256, 384}, {128, 384, 512}, {128, 512, 640}};
    case ShapeCase::kSameLast:
      // Same N (last dim), varying M and K
      return {{128, 256, 384}, {256, 256, 512}, {384, 256, 640}};
    case ShapeCase::kAllDifferent:
      return {{128, 256, 384}, {256, 384, 512}, {384, 512, 640}};
    case ShapeCase::kUnalignedAllSame:
    default:
      // (M, N, K) all multiples of 32 — satisfies both MXFP8 block (32) and NVFP4
      // quantize TMA 16B alignment (32 NVFP4 elements). None are multiples of 128,
      // so each expert's scale_inv is padded (exposes per-expert offset arithmetic bugs).
      return {{160, 288, 416}, {160, 288, 416}, {160, 288, 416}};
  }
}

// Runtime-only gating: the test calls TE's nvte_grouped_gemm* APIs, not cuBLAS directly,
// so compile-time cuBLAS version is irrelevant here. Use the runtime cuBLAS version
// (and device compute capability) instead.
constexpr size_t kCublasGroupedGemmVersion = 130300;        // Blackwell-only grouped GEMM
constexpr size_t kCublasGroupedGemmHopperVersion = 130400;  // adds Hopper support

inline std::string grouped_gemm_skip_reason(const InputRecipe& recipe) {
  const size_t cublas_ver = transformer_engine::cuda::cublas_version();
  if (cublas_ver < kCublasGroupedGemmVersion) {
    return "Grouped GEMM requires cuBLAS 13.3+, but run-time cuBLAS version is " +
           std::to_string(cublas_ver) + ".";
  }
  const int32_t cc = getDeviceComputeCapability();
  const std::string cc_suffix =
      "but device compute capability is " + std::to_string(cc) + ".";
  if (cc < hopperComputeCapability) {
    return "Grouped GEMM requires Hopper (SM90) or newer, " + cc_suffix;
  }
  if (cc < blackwellComputeCapability && cublas_ver < kCublasGroupedGemmHopperVersion) {
    return "Grouped GEMM on Hopper (SM90) requires cuBLAS 13.4+, but run-time cuBLAS "
           "version is " + std::to_string(cublas_ver) + ".";
  }
  if (recipe.has_value()) {
    const bool is_blackwell_plus = cc >= blackwellComputeCapability;
    if (!is_blackwell_plus && *recipe != NVTE_BLOCK_SCALING_1D) {
      return std::string(recipe_name(recipe)) +
             " grouped GEMM requires Blackwell (SM100) or newer, " + cc_suffix;
    }
    if (is_blackwell_plus && *recipe == NVTE_BLOCK_SCALING_1D) {
      return "FP8 block scaling grouped GEMM is only supported on Hopper (SM90), " + cc_suffix;
    }
  }
  return "";
}

// Reference setup shared by the three run_* variants: builds A/B/D tensors per recipe,
// runs nvte_multi_tensor_gemm to fill D_multi with reference results, and keeps the
// workspaces alive (returned in the struct so callers don't have to track them).
// Output dtype comes from TestParams::output_dtype (BF16 / FP16 / FP32).
struct GroupedGemmRefSetup {
  std::vector<std::tuple<size_t, size_t, size_t>> shapes;
  size_t num_gemms = 0;
  std::vector<Tensor> A_tensors;
  std::vector<Tensor> B_tensors;
  std::vector<Tensor> D_multi;
  std::vector<Tensor> workspaces;
  bool use_split_accum = false;
};

inline GroupedGemmRefSetup make_grouped_gemm_ref(const TestParams& params) {
  GroupedGemmRefSetup s;
  s.shapes = make_shapes(params.shape_case);
  s.num_gemms = s.shapes.size();
  s.A_tensors.reserve(s.num_gemms);
  s.B_tensors.reserve(s.num_gemms);
  s.D_multi.reserve(s.num_gemms);

  for (size_t i = 0; i < s.num_gemms; ++i) {
    const auto [M, N, K] = s.shapes[i];
    const std::vector<size_t> a_shape =
        params.transa ? std::vector<size_t>{N, K} : std::vector<size_t>{K, N};
    const std::vector<size_t> b_shape =
        params.transb ? std::vector<size_t>{K, M} : std::vector<size_t>{M, K};
    if (!params.recipe.has_value()) {
      s.A_tensors.emplace_back(make_bf16_operand("A" + std::to_string(i), a_shape));
      s.B_tensors.emplace_back(make_bf16_operand("B" + std::to_string(i), b_shape));
    } else {
      switch (*params.recipe) {
        case NVTE_DELAYED_TENSOR_SCALING:
          s.A_tensors.emplace_back(make_fp8_operand("A" + std::to_string(i), a_shape));
          s.B_tensors.emplace_back(make_fp8_operand("B" + std::to_string(i), b_shape));
          break;
        case NVTE_MXFP8_1D_SCALING:
          s.A_tensors.emplace_back(make_mxfp8_operand("A" + std::to_string(i), a_shape,
                                                      /*is_A=*/true, params.transa));
          s.B_tensors.emplace_back(make_mxfp8_operand("B" + std::to_string(i), b_shape,
                                                      /*is_A=*/false, params.transb));
          break;
        case NVTE_NVFP4_1D_SCALING:
          s.A_tensors.emplace_back(make_nvfp4_operand("A" + std::to_string(i), a_shape,
                                                      /*is_A=*/true, params.transa,
                                                      params.nvfp4_2d));
          s.B_tensors.emplace_back(make_nvfp4_operand("B" + std::to_string(i), b_shape,
                                                      /*is_A=*/false, params.transb,
                                                      params.nvfp4_2d));
          break;
        case NVTE_BLOCK_SCALING_1D:
          s.A_tensors.emplace_back(make_fp8_block_scaling_operand("A" + std::to_string(i),
                                                                  a_shape, /*is_A=*/true,
                                                                  params.transa));
          s.B_tensors.emplace_back(make_fp8_block_scaling_operand("B" + std::to_string(i),
                                                                  b_shape, /*is_A=*/false,
                                                                  params.transb));
          break;
        default:
          NVTE_ERROR("Unsupported scaling mode in grouped GEMM test: " +
                     to_string(*params.recipe));
      }
    }
    s.D_multi.emplace_back(Tensor("D_multi" + std::to_string(i),
                                  std::vector<size_t>{M, N}, params.output_dtype));
  }

  // FP8 block scaling requires split accumulator (no fast accumulation).
  s.use_split_accum = (params.recipe.has_value() && *params.recipe == NVTE_BLOCK_SCALING_1D);

  std::vector<NVTETensor> A_ptrs(s.num_gemms), B_ptrs(s.num_gemms), D_ptrs(s.num_gemms);
  std::vector<NVTETensor> workspace_ptrs(s.num_gemms, nullptr);
  std::vector<NVTETensor> bias_ptrs(s.num_gemms, nullptr), gelu_ptrs(s.num_gemms, nullptr);
  constexpr size_t cublas_ws_bytes = 32ull * 1024 * 1024;
  s.workspaces.reserve(s.num_gemms);
  for (size_t i = 0; i < s.num_gemms; ++i) {
    A_ptrs[i] = s.A_tensors[i].data();
    B_ptrs[i] = s.B_tensors[i].data();
    D_ptrs[i] = s.D_multi[i].data();
    s.workspaces.emplace_back(Tensor("workspace" + std::to_string(i),
                                     std::vector<size_t>{cublas_ws_bytes}, DType::kByte));
    workspace_ptrs[i] = s.workspaces.back().data();
  }
  nvte_multi_tensor_gemm(A_ptrs.data(), B_ptrs.data(), D_ptrs.data(), bias_ptrs.data(),
                         gelu_ptrs.data(), static_cast<int>(s.num_gemms),
                         params.transa, params.transb, false, workspace_ptrs.data(),
                         false, s.use_split_accum, 0, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  return s;
}

// Allocate and initialize alpha/beta tensors for grouped GEMM.
// Hopper requires a single shared scalar; Blackwell+ uses per-matrix scalars.
struct AlphaBetaTensors {
  Tensor alpha;
  Tensor beta;
};

inline AlphaBetaTensors make_alpha_beta(size_t num_gemms) {
  const int32_t cc = getDeviceComputeCapability();
  const size_t n = cc < blackwellComputeCapability ? 1 : num_gemms;
  AlphaBetaTensors ab{Tensor("alpha", std::vector<size_t>{n}, DType::kFloat32),
                      Tensor("beta", std::vector<size_t>{n}, DType::kFloat32)};
  std::vector<float> a(n, 1.f);
  std::vector<float> b(n, 0.f);
  NVTE_CHECK_CUDA(cudaMemcpy(ab.alpha.rowwise_dptr(), a.data(), n * sizeof(float),
                             cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(ab.beta.rowwise_dptr(), b.data(), n * sizeof(float),
                             cudaMemcpyHostToDevice));
  return ab;
}

// Compare each tensor inside a grouped D buffer (with per-tensor offsets) against the
// reference D_multi[i] tensors.
inline void compare_grouped_d_to_multi(
    const GroupedBuffers& grouped_D,
    const std::vector<std::tuple<size_t, size_t, size_t>>& shapes,
    std::vector<Tensor>& D_multi, const char* tag) {
  for (size_t i = 0; i < shapes.size(); ++i) {
    Tensor grouped_split("grouped_D" + std::to_string(i),
                         std::vector<size_t>{static_cast<size_t>(std::get<0>(shapes[i])),
                                             static_cast<size_t>(std::get<1>(shapes[i]))},
                         D_multi[i].dtype());
    const size_t offset_bytes =
        static_cast<size_t>(grouped_D.offsets_host[i]) * grouped_D.elem_size;
    NVTE_CHECK_CUDA(cudaMemcpy(grouped_split.rowwise_dptr(),
                               static_cast<char*>(grouped_D.get_data()) + offset_bytes,
                               grouped_D.tensor_bytes[i], cudaMemcpyDeviceToDevice));
    grouped_split.to_cpu();
    D_multi[i].to_cpu();
    auto [atol, rtol] = getTolerances(D_multi[i].dtype());
    switch (D_multi[i].dtype()) {
      case DType::kBFloat16:
        compareResults(tag, grouped_split, D_multi[i].rowwise_cpu_dptr<bf16>(), true, atol, rtol);
        break;
      case DType::kFloat16:
        compareResults(tag, grouped_split, D_multi[i].rowwise_cpu_dptr<fp16>(), true, atol, rtol);
        break;
      case DType::kFloat32:
        compareResults(tag, grouped_split, D_multi[i].rowwise_cpu_dptr<float>(), true, atol, rtol);
        break;
      default:
        NVTE_ERROR("Unsupported D dtype in test: " + to_string(D_multi[i].dtype()));
    }
  }
}

void run_grouped_gemm_case(const TestParams& params) {
  if (auto reason = grouped_gemm_skip_reason(params.recipe); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
  auto ref = make_grouped_gemm_ref(params);
  const auto& shapes = ref.shapes;
  const size_t num_gemms = ref.num_gemms;

  std::vector<Tensor*> A_views, B_views;
  A_views.reserve(num_gemms);
  B_views.reserve(num_gemms);
  for (size_t i = 0; i < num_gemms; ++i) {
    A_views.push_back(&ref.A_tensors[i]);
    B_views.push_back(&ref.B_tensors[i]);
  }

  GroupedBuffers grouped_A = build_grouped_tensor(A_views, ref.A_tensors[0].scaling_mode());
  GroupedBuffers grouped_B = build_grouped_tensor(B_views, ref.B_tensors[0].scaling_mode());

  std::vector<Tensor> C_tensors;
  std::vector<Tensor> D_group_tensors;
  C_tensors.reserve(num_gemms);
  D_group_tensors.reserve(num_gemms);
  for (size_t i = 0; i < num_gemms; ++i) {
    const auto [M, N, K] = shapes[i];
    (void)K;
    if (!params.use_null_c) {
      C_tensors.emplace_back(Tensor("C" + std::to_string(i),
                                    std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)},
                                    params.output_dtype));
    }
    D_group_tensors.emplace_back(Tensor("D_group" + std::to_string(i),
                                        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)},
                                        params.output_dtype));
    NVTE_CHECK_CUDA(cudaMemset(D_group_tensors.back().rowwise_dptr(), 0, bytes(D_group_tensors.back().rowwise_shape(), D_group_tensors.back().dtype())));
  }

  std::vector<Tensor*> C_views, D_views;
  for (size_t i = 0; i < num_gemms; ++i) {
    if (!params.use_null_c) {
      C_views.push_back(&C_tensors[i]);
    }
    D_views.push_back(&D_group_tensors[i]);
  }

  std::optional<GroupedBuffers> grouped_C;
  if (!params.use_null_c) {
    grouped_C = build_grouped_tensor(C_views, NVTE_DELAYED_TENSOR_SCALING);
  }
  GroupedBuffers grouped_D = build_grouped_tensor(D_views, NVTE_DELAYED_TENSOR_SCALING);

  AlphaBetaTensors ab = make_alpha_beta(num_gemms);

  constexpr size_t cublas_ws_bytes = 32ull * 1024 * 1024;
  const size_t setup_ws_bytes = nvte_get_grouped_gemm_setup_workspace_size(num_gemms);
  Tensor setup_ws("setup_ws", std::vector<size_t>{setup_ws_bytes}, DType::kByte);
  Tensor cublas_ws("cublas_ws", std::vector<size_t>{cublas_ws_bytes}, DType::kByte);

  GroupedMatmulConfigWrapper grouped_config;
  if (ref.use_split_accum) {
    grouped_config.set_use_split_accumulator(true);
  }

  nvte_grouped_gemm(grouped_A.get_handle(), params.transa, grouped_B.get_handle(), params.transb,
                    params.use_null_c ? nullptr : grouped_C->get_handle(), grouped_D.get_handle(),
                    ab.alpha.data(), ab.beta.data(), setup_ws.data(), cublas_ws.data(),
                    grouped_config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  compare_grouped_d_to_multi(grouped_D, shapes, ref.D_multi, "grouped_vs_multi");
}

void run_grouped_gemm_discrete_out_case(const TestParams& params) {
  if (auto reason = grouped_gemm_skip_reason(params.recipe); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
  auto ref = make_grouped_gemm_ref(params);
  const auto& shapes = ref.shapes;
  const size_t num_gemms = ref.num_gemms;

  std::vector<Tensor*> A_views, B_views;
  A_views.reserve(num_gemms);
  B_views.reserve(num_gemms);
  for (size_t i = 0; i < num_gemms; ++i) {
    A_views.push_back(&ref.A_tensors[i]);
    B_views.push_back(&ref.B_tensors[i]);
  }

  GroupedBuffers grouped_A = build_grouped_tensor(A_views, ref.A_tensors[0].scaling_mode());
  GroupedBuffers grouped_B = build_grouped_tensor(B_views, ref.B_tensors[0].scaling_mode());

  std::vector<Tensor> C_tensors;
  std::vector<Tensor> D_list_tensors;
  C_tensors.reserve(num_gemms);
  D_list_tensors.reserve(num_gemms);
  for (size_t i = 0; i < num_gemms; ++i) {
    const auto [M, N, K] = shapes[i];
    (void)K;
    if (!params.use_null_c) {
      C_tensors.emplace_back(
          Tensor("C" + std::to_string(i), std::vector<size_t>{M, N}, params.output_dtype));
    }
    D_list_tensors.emplace_back(
        Tensor("D_list" + std::to_string(i), std::vector<size_t>{M, N}, params.output_dtype));
    NVTE_CHECK_CUDA(cudaMemset(D_list_tensors.back().rowwise_dptr(), 0,
                               bytes(D_list_tensors.back().rowwise_shape(),
                                     D_list_tensors.back().dtype())));
  }

  std::vector<NVTETensor> C_list_ptrs;
  std::vector<NVTETensor> D_list_ptrs;
  if (!params.use_null_c) {
    C_list_ptrs.reserve(num_gemms);
  }
  D_list_ptrs.reserve(num_gemms);
  for (size_t i = 0; i < num_gemms; ++i) {
    if (!params.use_null_c) {
      C_list_ptrs.push_back(C_tensors[i].data());
    }
    D_list_ptrs.push_back(D_list_tensors[i].data());
  }

  AlphaBetaTensors ab = make_alpha_beta(num_gemms);

  constexpr size_t cublas_ws_bytes = 32ull * 1024 * 1024;
  const size_t setup_ws_bytes = nvte_get_grouped_gemm_setup_workspace_size(num_gemms);
  Tensor setup_ws("setup_ws", std::vector<size_t>{setup_ws_bytes}, DType::kByte);
  Tensor cublas_ws("cublas_ws", std::vector<size_t>{cublas_ws_bytes}, DType::kByte);

  GroupedMatmulConfigWrapper grouped_config;
  if (ref.use_split_accum) {
    grouped_config.set_use_split_accumulator(true);
  }

  nvte_grouped_gemm_with_discrete_out(
      grouped_A.get_handle(), params.transa, grouped_B.get_handle(), params.transb,
      params.use_null_c ? nullptr : C_list_ptrs.data(), params.use_null_c ? 0 : num_gemms,
      D_list_ptrs.data(), num_gemms, ab.alpha.data(), ab.beta.data(), setup_ws.data(),
      cublas_ws.data(), grouped_config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  for (size_t i = 0; i < num_gemms; ++i) {
    D_list_tensors[i].to_cpu();
    ref.D_multi[i].to_cpu();
    auto [atol, rtol] = getTolerances(ref.D_multi[i].dtype());
    switch (ref.D_multi[i].dtype()) {
      case DType::kBFloat16:
        compareResults("grouped_list_vs_multi", D_list_tensors[i],
                       ref.D_multi[i].rowwise_cpu_dptr<bf16>(), true, atol, rtol);
        break;
      case DType::kFloat16:
        compareResults("grouped_list_vs_multi", D_list_tensors[i],
                       ref.D_multi[i].rowwise_cpu_dptr<fp16>(), true, atol, rtol);
        break;
      case DType::kFloat32:
        compareResults("grouped_list_vs_multi", D_list_tensors[i],
                       ref.D_multi[i].rowwise_cpu_dptr<float>(), true, atol, rtol);
        break;
      default:
        NVTE_ERROR("Unsupported D dtype in test: " + to_string(ref.D_multi[i].dtype()));
    }
  }
}

void run_grouped_gemm_discrete_in_case(const TestParams& params) {
  if (auto reason = grouped_gemm_skip_reason(params.recipe); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
  auto ref = make_grouped_gemm_ref(params);
  const auto& shapes = ref.shapes;
  const size_t num_gemms = ref.num_gemms;

  std::vector<Tensor*> B_views;
  B_views.reserve(num_gemms);
  for (size_t i = 0; i < num_gemms; ++i) B_views.push_back(&ref.B_tensors[i]);

  GroupedBuffers grouped_B = build_grouped_tensor(B_views, ref.B_tensors[0].scaling_mode());

  std::vector<Tensor> C_tensors, D_group_tensors;
  C_tensors.reserve(num_gemms);
  D_group_tensors.reserve(num_gemms);
  for (size_t i = 0; i < num_gemms; ++i) {
    const auto [M, N, K] = shapes[i];
    (void)K;
    if (!params.use_null_c) {
      C_tensors.emplace_back(Tensor("C" + std::to_string(i), std::vector<size_t>{M, N},
                                    params.output_dtype));
    }
    D_group_tensors.emplace_back(Tensor("D_group" + std::to_string(i),
                                        std::vector<size_t>{M, N}, params.output_dtype));
    NVTE_CHECK_CUDA(cudaMemset(D_group_tensors.back().rowwise_dptr(), 0,
                               bytes(D_group_tensors.back().rowwise_shape(),
                                     D_group_tensors.back().dtype())));
  }

  std::vector<Tensor*> C_views, D_views;
  for (size_t i = 0; i < num_gemms; ++i) {
    if (!params.use_null_c) C_views.push_back(&C_tensors[i]);
    D_views.push_back(&D_group_tensors[i]);
  }

  std::optional<GroupedBuffers> grouped_C;
  if (!params.use_null_c) {
    grouped_C = build_grouped_tensor(C_views, NVTE_DELAYED_TENSOR_SCALING);
  }
  GroupedBuffers grouped_D = build_grouped_tensor(D_views, NVTE_DELAYED_TENSOR_SCALING);

  AlphaBetaTensors ab = make_alpha_beta(num_gemms);

  constexpr size_t cublas_ws_bytes = 32ull * 1024 * 1024;
  const size_t setup_ws_bytes = nvte_get_grouped_gemm_setup_workspace_size(num_gemms);
  Tensor setup_ws("setup_ws", std::vector<size_t>{setup_ws_bytes}, DType::kByte);
  Tensor cublas_ws("cublas_ws", std::vector<size_t>{cublas_ws_bytes}, DType::kByte);

  std::vector<NVTETensor> A_list_ptrs;
  A_list_ptrs.reserve(num_gemms);
  for (size_t i = 0; i < num_gemms; ++i) A_list_ptrs.push_back(ref.A_tensors[i].data());

  GroupedMatmulConfigWrapper grouped_config;
  if (ref.use_split_accum) {
    grouped_config.set_use_split_accumulator(true);
  }

  nvte_grouped_gemm_with_discrete_inputA(
      A_list_ptrs.data(), num_gemms, params.transa, grouped_B.get_handle(), params.transb,
      params.use_null_c ? nullptr : grouped_C->get_handle(), grouped_D.get_handle(),
      ab.alpha.data(), ab.beta.data(), setup_ws.data(), cublas_ws.data(), grouped_config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  compare_grouped_d_to_multi(grouped_D, shapes, ref.D_multi, "grouped_discrete_in_vs_multi");
}

class GroupedGemmTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(GroupedGemmTest, CompareWithMultiTensorGemm) {
  run_grouped_gemm_case(GetParam());
}

TEST_P(GroupedGemmTest, CompareWithMultiTensorGemmDiscreteOut) {
  run_grouped_gemm_discrete_out_case(GetParam());
}

TEST_P(GroupedGemmTest, CompareWithMultiTensorGemmDiscreteIn) {
  run_grouped_gemm_discrete_in_case(GetParam());
}

std::string MakeGroupedGemmTestName(const testing::TestParamInfo<GroupedGemmTest::ParamType>& info) {
  constexpr const char* kShapeNames[] = {"AllSame", "SameM", "SameN", "AllDiff",
                                         "UnalignedAllSame"};
  const std::string layout = std::string("ta") + (info.param.transa ? "T" : "N") +
                             "tb" + (info.param.transb ? "T" : "N");
  const std::string null_c = info.param.use_null_c ? "_NullC" : "";
  const std::string nvfp4_2d = info.param.nvfp4_2d ? "_2D" : "";
  std::string out_suffix;
  switch (info.param.output_dtype) {
    case DType::kBFloat16: break;  // default, no suffix
    case DType::kFloat16:  out_suffix = "_outFP16"; break;
    case DType::kFloat32:  out_suffix = "_outFP32"; break;
    default:               out_suffix = "_outUnknown"; break;
  }
  return std::string(recipe_name(info.param.recipe)) + nvfp4_2d + "_" +
         kShapeNames[static_cast<int>(info.param.shape_case)] + "_" + layout + null_c + out_suffix;
}

// TestParams: {recipe, transa, transb, shape_case, use_null_c}
// recipe == std::nullopt means BF16 (no scaling), otherwise the FP8/NVFP4 scaling mode.
const std::vector<TestParams> kTestParams = {
    // FP8 tests (each tensor has random mean/stddev -> different scales)
    {NVTE_DELAYED_TENSOR_SCALING, true, false, ShapeCase::kAllDifferent, false},
    {NVTE_DELAYED_TENSOR_SCALING, false, true, ShapeCase::kAllDifferent, false},
    {NVTE_DELAYED_TENSOR_SCALING, false, false, ShapeCase::kAllSame, false},
    // BF16 tests
    {std::nullopt, true, false, ShapeCase::kSameFirst, false},
    {std::nullopt, false, true, ShapeCase::kSameLast, false},
    {std::nullopt, false, false, ShapeCase::kAllSame, false},
    {std::nullopt, true, true, ShapeCase::kAllDifferent, false},
    // Test NULL C (valid when beta=0)
    {std::nullopt, false, false, ShapeCase::kAllSame, true},
    // MXFP8 tests
    {NVTE_MXFP8_1D_SCALING, true, false, ShapeCase::kAllSame, false},
    {NVTE_MXFP8_1D_SCALING, true, false, ShapeCase::kAllDifferent, false},
    {NVTE_MXFP8_1D_SCALING, false, true, ShapeCase::kAllSame, false},
    {NVTE_MXFP8_1D_SCALING, false, true, ShapeCase::kAllDifferent, false},
    {NVTE_MXFP8_1D_SCALING, false, false, ShapeCase::kAllSame, false},
    {NVTE_MXFP8_1D_SCALING, false, false, ShapeCase::kAllDifferent, false},
    {NVTE_MXFP8_1D_SCALING, false, false, ShapeCase::kSameFirst, false},
    // MXFP8 with NULL C
    {NVTE_MXFP8_1D_SCALING, true, false, ShapeCase::kAllSame, true},
    // NVFP4 tests (all transpose combinations - GEMM internally forces TN)
    {NVTE_NVFP4_1D_SCALING, true, false, ShapeCase::kAllSame, false},
    {NVTE_NVFP4_1D_SCALING, true, false, ShapeCase::kAllDifferent, false},
    {NVTE_NVFP4_1D_SCALING, true, false, ShapeCase::kSameFirst, false},
    {NVTE_NVFP4_1D_SCALING, true, false, ShapeCase::kSameLast, false},
    {NVTE_NVFP4_1D_SCALING, false, true, ShapeCase::kAllSame, false},
    {NVTE_NVFP4_1D_SCALING, false, true, ShapeCase::kAllDifferent, false},
    {NVTE_NVFP4_1D_SCALING, false, false, ShapeCase::kAllSame, false},
    {NVTE_NVFP4_1D_SCALING, false, false, ShapeCase::kAllDifferent, false},
    // NVFP4 with NULL C
    {NVTE_NVFP4_1D_SCALING, true, false, ShapeCase::kAllSame, true},
    // NVFP4 with 2D (16x16) quantization — scales fed to cuBLAS keep the VEC16 layout,
    // so this verifies that 2D-quantized inputs also produce the correct GEMM result.
    {NVTE_NVFP4_1D_SCALING, true, false, ShapeCase::kAllSame, false, /*nvfp4_2d=*/true},
    {NVTE_NVFP4_1D_SCALING, false, true, ShapeCase::kAllDifferent, false, /*nvfp4_2d=*/true},
    {NVTE_NVFP4_1D_SCALING, false, false, ShapeCase::kAllSame, false, /*nvfp4_2d=*/true},
    // Non-default output dtypes — implementation accepts BF16/FP16/FP32, verify the
    // less-common ones on a couple of recipes (BF16 covered everywhere else).
    {std::nullopt,                false, false, ShapeCase::kAllSame, false,
     /*nvfp4_2d=*/false, /*output_dtype=*/DType::kFloat16},
    {std::nullopt,                false, false, ShapeCase::kAllSame, false,
     /*nvfp4_2d=*/false, /*output_dtype=*/DType::kFloat32},
    {NVTE_DELAYED_TENSOR_SCALING, true,  false, ShapeCase::kAllSame, false,
     /*nvfp4_2d=*/false, /*output_dtype=*/DType::kFloat16},
    {NVTE_NVFP4_1D_SCALING,       true,  false, ShapeCase::kAllSame, false,
     /*nvfp4_2d=*/false, /*output_dtype=*/DType::kFloat16},
    // FP8 Block Scaling tests (TN layout on Hopper, block size 128)
    {NVTE_BLOCK_SCALING_1D, true, false, ShapeCase::kAllSame, false},
    {NVTE_BLOCK_SCALING_1D, true, false, ShapeCase::kAllDifferent, false},
    {NVTE_BLOCK_SCALING_1D, true, false, ShapeCase::kSameFirst, false},
    {NVTE_BLOCK_SCALING_1D, true, false, ShapeCase::kSameLast, false},
    {NVTE_BLOCK_SCALING_1D, false, true, ShapeCase::kAllSame, false},
    {NVTE_BLOCK_SCALING_1D, false, false, ShapeCase::kAllSame, false},
    // FP8 Block Scaling with NULL C
    {NVTE_BLOCK_SCALING_1D, true, false, ShapeCase::kAllSame, true},
    // Unaligned-dim tests: dims are multiples of 32 / 16 (per-recipe block size) but NOT
    // multiples of 128 — exposes scale_inv padding bugs in per-expert offset arithmetic.
    // MXFP8 covered by upstream PR #2954, the rest by the analogous fix.
    {NVTE_MXFP8_1D_SCALING, true, false, ShapeCase::kUnalignedAllSame, false},
    {NVTE_MXFP8_1D_SCALING, false, true, ShapeCase::kUnalignedAllSame, false},
    {NVTE_MXFP8_1D_SCALING, false, false, ShapeCase::kUnalignedAllSame, false},
    {NVTE_NVFP4_1D_SCALING, true, false, ShapeCase::kUnalignedAllSame, false},
    {NVTE_NVFP4_1D_SCALING, false, true, ShapeCase::kUnalignedAllSame, false},
    {NVTE_NVFP4_1D_SCALING, false, false, ShapeCase::kUnalignedAllSame, false},
    {NVTE_BLOCK_SCALING_1D, true, false, ShapeCase::kUnalignedAllSame, false},
    {NVTE_BLOCK_SCALING_1D, false, true, ShapeCase::kUnalignedAllSame, false},
    {NVTE_BLOCK_SCALING_1D, false, false, ShapeCase::kUnalignedAllSame, false},
};

INSTANTIATE_TEST_SUITE_P(OperatorTest,
                         GroupedGemmTest,
                         ::testing::ValuesIn(kTestParams),
                         MakeGroupedGemmTestName);

}  // namespace
