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

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum class InputCase {
  kFP8Current,
  kBF16,
  kMXFP8,
  kNVFP4,
  kFP8BlockScaling,
};

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

// Helper: quantize BF16 tensor to NVFP4 rowwise-only, swizzle scales, return swizzled tensor.
Tensor make_nvfp4_rowwise(const std::string& name, const std::vector<size_t>& shape) {
  Tensor input_bf16(name + "_bf16", shape, DType::kBFloat16);
  fillUniform(&input_bf16);

  Tensor nvfp4(name, shape, DType::kFloat4E2M1, /*rowwise=*/true, /*columnwise=*/false,
               NVTE_NVFP4_1D_SCALING);

  QuantizationConfigWrapper quant_config;
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
// NVFP4 "columnwise" data is the transposed tensor quantized rowwise.
// We quantize rowwise directly, and for columnwise we quantize the transposed input rowwise.
Tensor make_nvfp4_operand(const std::string& name, const std::vector<size_t>& shape,
                          bool is_A, bool transposed) {
  (void)is_A;
  (void)transposed;

  // 1. Rowwise: quantize + swizzle directly
  Tensor rowwise = make_nvfp4_rowwise(name + "_row", shape);

  // 2. Columnwise: transpose input, quantize + swizzle as rowwise of transposed shape
  std::vector<size_t> t_shape = {shape[1], shape[0]};
  Tensor colwise = make_nvfp4_rowwise(name + "_col", t_shape);

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
  InputCase input_case;
  bool transa;
  bool transb;
  ShapeCase shape_case;
  bool use_null_c = false;  // When true, pass nullptr for C (valid when beta=0)
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
      // (M, N, K) all multiples of 32 (MXFP8 block) and 16 (NVFP4 block), but NONE
      // are multiples of 128 — so each expert's scale_inv is padded.
      return {{160, 288, 416}, {160, 288, 416}, {160, 288, 416}};
  }
}

// Compile-time version macro for Hopper grouped GEMM support (mirrors cublaslt_grouped_gemm.cu)
#define CUBLAS_GROUPED_GEMM_HOPPER_VERSION 130400

inline std::string grouped_gemm_skip_reason(InputCase input_case) {
#if CUBLAS_VERSION < 130300
  return "Grouped GEMM requires cuBLAS 13.3+, but compile-time cuBLAS version is " +
         std::to_string(CUBLAS_VERSION) + ".";
#else
  const int32_t cc = getDeviceComputeCapability();
  const std::string cc_suffix =
      "but device compute capability is " + std::to_string(cc) + ".";
#if CUBLAS_VERSION >= CUBLAS_GROUPED_GEMM_HOPPER_VERSION
  if (cc < hopperComputeCapability) {
    return "Grouped GEMM requires Hopper (SM90) or newer with cuBLAS 13.4+, " + cc_suffix;
  }
  if (cc < blackwellComputeCapability && input_case == InputCase::kFP8Current) {
    return "FP8 tensor scaling grouped GEMM requires Blackwell (SM100) or newer, " + cc_suffix;
  }
  if (cc < blackwellComputeCapability && input_case == InputCase::kMXFP8) {
    return "MXFP8 grouped GEMM requires Blackwell (SM100) or newer, " + cc_suffix;
  }
  if (cc < blackwellComputeCapability && input_case == InputCase::kNVFP4) {
    return "NVFP4 grouped GEMM requires Blackwell (SM100) or newer, " + cc_suffix;
  }
  if (cc >= blackwellComputeCapability && input_case == InputCase::kFP8BlockScaling) {
    return "FP8 block scaling grouped GEMM is only supported on Hopper (SM90), " + cc_suffix;
  }
#else
  if (cc < blackwellComputeCapability) {
    return "Grouped GEMM requires Blackwell (SM100) or newer.";
  }
  if (input_case == InputCase::kFP8BlockScaling) {
    return "FP8 block scaling grouped GEMM is only supported on Hopper (SM90), " + cc_suffix;
  }
#endif
  return "";
#endif
}

// Reference setup shared by the three run_* variants: builds A/B/D tensors per recipe,
// runs nvte_multi_tensor_gemm to fill D_multi with reference results, and keeps the
// workspaces alive (returned in the struct so callers don't have to track them).
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
    switch (params.input_case) {
      case InputCase::kFP8Current:
        s.A_tensors.emplace_back(make_fp8_operand("A" + std::to_string(i), a_shape));
        s.B_tensors.emplace_back(make_fp8_operand("B" + std::to_string(i), b_shape));
        break;
      case InputCase::kBF16:
        s.A_tensors.emplace_back(make_bf16_operand("A" + std::to_string(i), a_shape));
        s.B_tensors.emplace_back(make_bf16_operand("B" + std::to_string(i), b_shape));
        break;
      case InputCase::kMXFP8:
        s.A_tensors.emplace_back(make_mxfp8_operand("A" + std::to_string(i), a_shape,
                                                    /*is_A=*/true, params.transa));
        s.B_tensors.emplace_back(make_mxfp8_operand("B" + std::to_string(i), b_shape,
                                                    /*is_A=*/false, params.transb));
        break;
      case InputCase::kNVFP4:
        s.A_tensors.emplace_back(make_nvfp4_operand("A" + std::to_string(i), a_shape,
                                                    /*is_A=*/true, params.transa));
        s.B_tensors.emplace_back(make_nvfp4_operand("B" + std::to_string(i), b_shape,
                                                    /*is_A=*/false, params.transb));
        break;
      case InputCase::kFP8BlockScaling:
        s.A_tensors.emplace_back(make_fp8_block_scaling_operand("A" + std::to_string(i),
                                                                a_shape, /*is_A=*/true,
                                                                params.transa));
        s.B_tensors.emplace_back(make_fp8_block_scaling_operand("B" + std::to_string(i),
                                                                b_shape, /*is_A=*/false,
                                                                params.transb));
        break;
    }
    s.D_multi.emplace_back(Tensor("D_multi" + std::to_string(i),
                                  std::vector<size_t>{M, N}, DType::kBFloat16));
  }

  // FP8 block scaling requires split accumulator (no fast accumulation).
  s.use_split_accum = (params.input_case == InputCase::kFP8BlockScaling);

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
    compareResults(tag, grouped_split, D_multi[i].rowwise_cpu_dptr<bf16>(), true, atol, rtol);
  }
}

void run_grouped_gemm_case(const TestParams& params) {
  if (auto reason = grouped_gemm_skip_reason(params.input_case); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
#if CUBLAS_VERSION >= 130300
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
                                    DType::kBFloat16));
    }
    D_group_tensors.emplace_back(Tensor("D_group" + std::to_string(i),
                                        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)},
                                        DType::kBFloat16));
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
#endif  // CUBLAS_VERSION >= 130300
}

void run_grouped_gemm_discrete_out_case(const TestParams& params) {
  if (auto reason = grouped_gemm_skip_reason(params.input_case); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
#if CUBLAS_VERSION >= 130300
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
          Tensor("C" + std::to_string(i), std::vector<size_t>{M, N}, DType::kBFloat16));
    }
    D_list_tensors.emplace_back(
        Tensor("D_list" + std::to_string(i), std::vector<size_t>{M, N}, DType::kBFloat16));
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
    compareResults("grouped_list_vs_multi", D_list_tensors[i],
                   ref.D_multi[i].rowwise_cpu_dptr<bf16>(), true, atol, rtol);
  }
#endif  // CUBLAS_VERSION >= 130300
}

void run_grouped_gemm_discrete_in_case(const TestParams& params) {
  if (auto reason = grouped_gemm_skip_reason(params.input_case); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
#if CUBLAS_VERSION >= 130300
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
                                    DType::kBFloat16));
    }
    D_group_tensors.emplace_back(Tensor("D_group" + std::to_string(i),
                                        std::vector<size_t>{M, N}, DType::kBFloat16));
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
#endif  // CUBLAS_VERSION >= 130300
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
  constexpr const char* kInputNames[] = {"FP8Current", "BF16", "MXFP8", "NVFP4", "FP8BlockScaling"};
  constexpr const char* kShapeNames[] = {"AllSame", "SameM", "SameN", "AllDiff",
                                         "UnalignedAllSame"};
  const std::string layout = std::string("ta") + (info.param.transa ? "T" : "N") +
                             "tb" + (info.param.transb ? "T" : "N");
  const std::string null_c = info.param.use_null_c ? "_NullC" : "";
  return std::string(kInputNames[static_cast<int>(info.param.input_case)]) + "_" +
         kShapeNames[static_cast<int>(info.param.shape_case)] + "_" + layout + null_c;
}

// TestParams: {input_case, transa, transb, shape_case, use_null_c}
const std::vector<TestParams> kTestParams = {
    // FP8 tests (each tensor has random mean/stddev -> different scales)
    {InputCase::kFP8Current, true, false, ShapeCase::kAllDifferent, false},
    {InputCase::kFP8Current, false, true, ShapeCase::kAllDifferent, false},
    {InputCase::kFP8Current, false, false, ShapeCase::kAllSame, false},
    // BF16 tests
    {InputCase::kBF16, true, false, ShapeCase::kSameFirst, false},
    {InputCase::kBF16, false, true, ShapeCase::kSameLast, false},
    {InputCase::kBF16, false, false, ShapeCase::kAllSame, false},
    {InputCase::kBF16, true, true, ShapeCase::kAllDifferent, false},
    // Test NULL C (valid when beta=0)
    {InputCase::kBF16, false, false, ShapeCase::kAllSame, true},
    // MXFP8 tests
    {InputCase::kMXFP8, true, false, ShapeCase::kAllSame, false},
    {InputCase::kMXFP8, true, false, ShapeCase::kAllDifferent, false},
    {InputCase::kMXFP8, false, true, ShapeCase::kAllSame, false},
    {InputCase::kMXFP8, false, true, ShapeCase::kAllDifferent, false},
    {InputCase::kMXFP8, false, false, ShapeCase::kAllSame, false},
    {InputCase::kMXFP8, false, false, ShapeCase::kAllDifferent, false},
    {InputCase::kMXFP8, false, false, ShapeCase::kSameFirst, false},
    // MXFP8 with NULL C
    {InputCase::kMXFP8, true, false, ShapeCase::kAllSame, true},
    // NVFP4 tests (all transpose combinations - GEMM internally forces TN)
    {InputCase::kNVFP4, true, false, ShapeCase::kAllSame, false},
    {InputCase::kNVFP4, true, false, ShapeCase::kAllDifferent, false},
    {InputCase::kNVFP4, true, false, ShapeCase::kSameFirst, false},
    {InputCase::kNVFP4, true, false, ShapeCase::kSameLast, false},
    {InputCase::kNVFP4, false, true, ShapeCase::kAllSame, false},
    {InputCase::kNVFP4, false, true, ShapeCase::kAllDifferent, false},
    {InputCase::kNVFP4, false, false, ShapeCase::kAllSame, false},
    {InputCase::kNVFP4, false, false, ShapeCase::kAllDifferent, false},
    // NVFP4 with NULL C
    {InputCase::kNVFP4, true, false, ShapeCase::kAllSame, true},
    // FP8 Block Scaling tests (TN layout on Hopper, block size 128)
    {InputCase::kFP8BlockScaling, true, false, ShapeCase::kAllSame, false},
    {InputCase::kFP8BlockScaling, true, false, ShapeCase::kAllDifferent, false},
    {InputCase::kFP8BlockScaling, true, false, ShapeCase::kSameFirst, false},
    {InputCase::kFP8BlockScaling, true, false, ShapeCase::kSameLast, false},
    {InputCase::kFP8BlockScaling, false, true, ShapeCase::kAllSame, false},
    {InputCase::kFP8BlockScaling, false, false, ShapeCase::kAllSame, false},
    // FP8 Block Scaling with NULL C
    {InputCase::kFP8BlockScaling, true, false, ShapeCase::kAllSame, true},
    // Unaligned-dim tests: dims are multiples of 32 / 16 (per-recipe block size) but NOT
    // multiples of 128 — exposes scale_inv padding bugs in per-expert offset arithmetic.
    // MXFP8 covered by upstream PR #2954, the rest by the analogous fix.
    {InputCase::kMXFP8, true, false, ShapeCase::kUnalignedAllSame, false},
    {InputCase::kMXFP8, false, true, ShapeCase::kUnalignedAllSame, false},
    {InputCase::kMXFP8, false, false, ShapeCase::kUnalignedAllSame, false},
    {InputCase::kNVFP4, true, false, ShapeCase::kUnalignedAllSame, false},
    {InputCase::kNVFP4, false, true, ShapeCase::kUnalignedAllSame, false},
    {InputCase::kNVFP4, false, false, ShapeCase::kUnalignedAllSame, false},
    {InputCase::kFP8BlockScaling, true, false, ShapeCase::kUnalignedAllSame, false},
    {InputCase::kFP8BlockScaling, false, true, ShapeCase::kUnalignedAllSame, false},
    {InputCase::kFP8BlockScaling, false, false, ShapeCase::kUnalignedAllSame, false},
};

INSTANTIATE_TEST_SUITE_P(OperatorTest,
                         GroupedGemmTest,
                         ::testing::ValuesIn(kTestParams),
                         MakeGroupedGemmTestName);

}  // namespace
