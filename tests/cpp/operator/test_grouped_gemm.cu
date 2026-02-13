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
#include <tuple>
#include <vector>

#include <transformer_engine/cast.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/recipe.h>
#include <transformer_engine/transformer_engine.h>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum class InputCase {
  kFP8Current,
  kBF16,
};

enum class ShapeCase {
  kAllSame,
  kSameFirst,
  kSameLast,
  kAllDifferent,
};

size_t grouped_setup_workspace_size(const size_t num_tensors) {
  const size_t ptr_bytes = num_tensors * sizeof(void*);
  const size_t int_bytes = num_tensors * sizeof(int);
  // Layout: 6 pointer arrays (A, B, C, D, alpha, beta) + 6 int arrays (a_rows, a_cols, b_rows, b_cols, d_rows, d_cols)
  size_t size = 6 * ptr_bytes + 6 * int_bytes;
  const size_t alignment = 256;
  size = ((size + alignment - 1) / alignment) * alignment;
  return size;
}

Tensor make_fp8_operand(const std::string& name, const std::vector<size_t>& shape) {
  Tensor input_fp32(name + "_fp32", shape, DType::kFloat32);
  fillUniform(&input_fp32);

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
    default:
      return {{128, 256, 384}, {256, 384, 512}, {384, 512, 640}};
  }
}

void run_grouped_gemm_case(const TestParams& params) {
#if CUBLAS_VERSION < 130200
  GTEST_SKIP() << "Grouped GEMM requires cuBLAS 13.2+, but compile-time cuBLAS version is "
               << CUBLAS_VERSION << ".";
#else
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP() << "Grouped GEMM requires Blackwell (SM100) or newer.";
  }

  const std::vector<std::tuple<size_t, size_t, size_t>> shapes = make_shapes(params.shape_case);

  const size_t num_gemms = shapes.size();
  std::vector<Tensor> A_tensors;
  std::vector<Tensor> B_tensors;
  std::vector<Tensor> D_multi;

  A_tensors.reserve(num_gemms);
  B_tensors.reserve(num_gemms);
  D_multi.reserve(num_gemms);

  for (size_t i = 0; i < num_gemms; ++i) {
    const auto [M, N, K] = shapes[i];

    const std::vector<size_t> a_shape = params.transa ? std::vector<size_t>{N, K}
                                                      : std::vector<size_t>{K, N};
    const std::vector<size_t> b_shape = params.transb ? std::vector<size_t>{K, M}
                                                      : std::vector<size_t>{M, K};
    switch (params.input_case) {
      case InputCase::kFP8Current: {
        A_tensors.emplace_back(make_fp8_operand("A" + std::to_string(i), a_shape));
        B_tensors.emplace_back(make_fp8_operand("B" + std::to_string(i), b_shape));
        break;
      }
      case InputCase::kBF16: {
        A_tensors.emplace_back(make_bf16_operand("A" + std::to_string(i), a_shape));
        B_tensors.emplace_back(make_bf16_operand("B" + std::to_string(i), b_shape));
        break;
      }
    }
    D_multi.emplace_back(Tensor("D_multi" + std::to_string(i),
                                std::vector<size_t>{M, N},
                                DType::kBFloat16));
  }

  std::vector<NVTETensor> A_ptrs(num_gemms);
  std::vector<NVTETensor> B_ptrs(num_gemms);
  std::vector<NVTETensor> D_ptrs(num_gemms);
  std::vector<Tensor> workspaces(num_gemms);
  std::vector<NVTETensor> workspace_ptrs(num_gemms, nullptr);
  std::vector<Tensor*> A_views;
  std::vector<Tensor*> B_views;
  A_views.reserve(num_gemms);
  B_views.reserve(num_gemms);

  // Empty bias/gelu arrays for nvte_multi_tensor_gemm (no epilogues)
  std::vector<NVTETensor> bias_ptrs(num_gemms, nullptr);
  std::vector<NVTETensor> gelu_ptrs(num_gemms, nullptr);

  const size_t cublas_ws_bytes = 32ull * 1024 * 1024;

  for (size_t i = 0; i < num_gemms; ++i) {
    A_ptrs[i] = A_tensors[i].data();
    B_ptrs[i] = B_tensors[i].data();
    D_ptrs[i] = D_multi[i].data();
    workspaces[i] = Tensor("workspace" + std::to_string(i), std::vector<size_t>{cublas_ws_bytes}, DType::kByte);
    workspace_ptrs[i] = workspaces[i].data();
    A_views.push_back(&A_tensors[i]);
    B_views.push_back(&B_tensors[i]);
  }

  nvte_multi_tensor_gemm(A_ptrs.data(),
                         B_ptrs.data(),
                         D_ptrs.data(),
                         bias_ptrs.data(),
                         gelu_ptrs.data(),
                         static_cast<int>(num_gemms),
                         params.transa,
                         params.transb,
                         false,  // grad
                         workspace_ptrs.data(),
                         false,  // accumulate
                         false,  // use_split_accumulator
                         0,      // sm_count
                         0);

  GroupedBuffers grouped_A = build_grouped_tensor(A_views, A_tensors[0].scaling_mode());
  GroupedBuffers grouped_B = build_grouped_tensor(B_views, B_tensors[0].scaling_mode());

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

  // Per-matrix alpha/beta (all 1.0 and 0.0 respectively)
  Tensor alpha_tensor("alpha", std::vector<size_t>{num_gemms}, DType::kFloat32);
  Tensor beta_tensor("beta", std::vector<size_t>{num_gemms}, DType::kFloat32);
  std::vector<float> alpha_vals(num_gemms, 1.f);
  std::vector<float> beta_vals(num_gemms, 0.f);
  NVTE_CHECK_CUDA(cudaMemcpy(alpha_tensor.rowwise_dptr(), alpha_vals.data(),
                             num_gemms * sizeof(float), cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(beta_tensor.rowwise_dptr(), beta_vals.data(),
                             num_gemms * sizeof(float), cudaMemcpyHostToDevice));

  const size_t setup_ws_bytes = grouped_setup_workspace_size(num_gemms);
  Tensor setup_ws("setup_ws", std::vector<size_t>{setup_ws_bytes}, DType::kByte);
  Tensor cublas_ws("cublas_ws", std::vector<size_t>{cublas_ws_bytes}, DType::kByte);

  nvte_grouped_gemm(grouped_A.get_handle(),
                    params.transa,
                    grouped_B.get_handle(),
                    params.transb,
                    params.use_null_c ? nullptr : grouped_C->get_handle(),
                    grouped_D.get_handle(),
                    alpha_tensor.data(),
                    beta_tensor.data(),
                    setup_ws.data(),
                    cublas_ws.data(),
                    nullptr,  // config (use defaults)
                    0);

  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  // Compare results
  for (size_t i = 0; i < num_gemms; ++i) {
    Tensor grouped_split("grouped_D" + std::to_string(i),
                         std::vector<size_t>{static_cast<size_t>(std::get<0>(shapes[i])),
                                             static_cast<size_t>(std::get<1>(shapes[i]))},
                         D_multi[i].dtype());
    const size_t offset_bytes = static_cast<size_t>(grouped_D.offsets_host[i]) * grouped_D.elem_size;
    NVTE_CHECK_CUDA(cudaMemcpy(grouped_split.rowwise_dptr(),
                               static_cast<char*>(grouped_D.get_data()) + offset_bytes,
                               grouped_D.tensor_bytes[i],
                               cudaMemcpyDeviceToDevice));
    grouped_split.to_cpu();
    D_multi[i].to_cpu();
    auto [atol, rtol] = getTolerances(D_multi[i].dtype());
    compareResults("grouped_vs_multi",
                   grouped_split,
                   D_multi[i].rowwise_cpu_dptr<bf16>(),
                   true,
                   atol,
                   rtol);
  }
#endif  // CUBLAS_VERSION >= 130200
}

class GroupedGemmTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(GroupedGemmTest, CompareWithMultiTensorGemm) {
  run_grouped_gemm_case(GetParam());
}

std::string MakeGroupedGemmTestName(const testing::TestParamInfo<GroupedGemmTest::ParamType>& info) {
  constexpr const char* kInputNames[] = {"FP8Current", "BF16"};
  constexpr const char* kShapeNames[] = {"AllSame", "SameM", "SameN", "AllDiff"};
  const std::string layout = std::string("ta") + (info.param.transa ? "T" : "N") +
                             "tb" + (info.param.transb ? "T" : "N");
  const std::string null_c = info.param.use_null_c ? "_NullC" : "";
  return std::string(kInputNames[static_cast<int>(info.param.input_case)]) + "_" +
         kShapeNames[static_cast<int>(info.param.shape_case)] + "_" + layout + null_c;
}

// TestParams: {input_case, transa, transb, shape_case, use_null_c}
const std::vector<TestParams> kTestParams = {
    // Basic tests
    {InputCase::kFP8Current, false, true, ShapeCase::kAllDifferent, false},
    {InputCase::kFP8Current, false, false, ShapeCase::kAllSame, false},
    {InputCase::kBF16, true, false, ShapeCase::kSameFirst, false},
    {InputCase::kBF16, false, true, ShapeCase::kSameLast, false},
    {InputCase::kBF16, false, false, ShapeCase::kAllSame, false},
    {InputCase::kBF16, true, true, ShapeCase::kAllDifferent, false},
    // Test NULL C (valid when beta=0)
    {InputCase::kBF16, false, false, ShapeCase::kAllSame, true},
};

INSTANTIATE_TEST_SUITE_P(OperatorTest,
                         GroupedGemmTest,
                         ::testing::ValuesIn(kTestParams),
                         MakeGroupedGemmTestName);

}  // namespace
