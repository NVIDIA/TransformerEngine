/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/workspace.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include "../test_common.h"
#include "util/workspace.h"

using namespace transformer_engine;

namespace {

// Cross two alignment boundaries so persistent-region checks also cover an unaligned tail.
constexpr size_t kPersistentTableBytes = 2 * kWorkspaceAlignment + 1;
// A recognizable nonzero sentinel detects writes into the shared immutable region.
constexpr unsigned char kPersistentTableFillByte = 0x5a;

template <typename IT, typename OT, typename CT>
void compute_ref_cast_dbias(const IT *input_h,
                            const CT scale,
                            OT *output_c_h,
                            CT *amax_h,
                            IT *dbias_h,
                            const size_t N,
                            const size_t H) {
  CT amax  = 0.;

  std::vector<CT> acc_dbias(H, 0.);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = static_cast<CT>(input_h[i * H + j]);

      // update amax
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;

      output_c_h[i * H + j] = static_cast<OT>(scale * elt);

      // dbias
      acc_dbias[j] += elt;
    }
  }

  *amax_h = amax;

  for (size_t i = 0; i < H; i++) {
    dbias_h[i] = static_cast<IT>(acc_dbias[i]);
  }
}

template <typename IType, typename OType>
void performTest(const std::vector<size_t>& shape, bool use_workspace_descriptor = false) {
  using namespace test;
  using CType = fp32;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  const size_t N = first_dimension(shape);
  const size_t H = last_dimension(shape);

  Tensor input("input", shape, itype);

  Tensor output_c("output_c", shape, otype);
  // dbias has the same data type with "output grad"
  Tensor dbias("dbias", std::vector<size_t>{H}, itype);

  fillUniform(&input);
  setRandomScale(&output_c);
  const float ref_scale = isFp8Type(otype) ? output_c.scale() : 1.0f;

  std::unique_ptr<OType[]> ref_output_c = std::make_unique<OType[]>(N*H);
  std::unique_ptr<IType[]> ref_output_dbias = std::make_unique<IType[]>(H);

  CType ref_amax;
  compute_ref_cast_dbias(input.rowwise_cpu_dptr<IType>(),
                         ref_scale,
                         ref_output_c.get(),
                         &ref_amax,
                         ref_output_dbias.get(),
                         N, H);

  Tensor workspace;
  TensorWrapper descriptor;
  NVTEWorkspaceRegion persistent_region = 0;

  if (use_workspace_descriptor) {
    WorkspacePlanWrapper plan(descriptor.data());
    persistent_region =
        plan.require("test.activation", kPersistentTableBytes, kNVTEWorkspacePersistent);
    nvte_workspace_commit_plan(descriptor.data(), plan.data());
    nvte_quantize_dbias(input.data(), output_c.data(), dbias.data(), descriptor.data(), 0);
    const auto scratch = nvte_workspace_find_region(descriptor.data(), "te.dbias", nullptr, 0);
    ASSERT_NE(scratch, 0);
    EXPECT_NE(scratch, persistent_region);
    workspace = Tensor("workspace", descriptor.shape(), descriptor.dtype());
    descriptor.set_rowwise_data(workspace.rowwise_dptr(), descriptor.dtype(), descriptor.shape());
    ASSERT_EQ(cudaMemset(nvte_workspace_get_region_data(descriptor.data(), persistent_region),
                         kPersistentTableFillByte, kPersistentTableBytes),
              cudaSuccess);
    nvte_workspace_mark_region_initialized(descriptor.data(), persistent_region, nullptr);
    nvte_quantize_dbias(input.data(), output_c.data(), dbias.data(), descriptor.data(), 0);
  } else {

    nvte_quantize_dbias(input.data(),
                        output_c.data(),
                        dbias.data(),
                        workspace.data(),
                        0);

    workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

    nvte_quantize_dbias(input.data(),
                        output_c.data(),
                        dbias.data(),
                        workspace.data(),
                        0);

  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (persistent_region != 0) {
    std::vector<unsigned char> table(kPersistentTableBytes);
    ASSERT_EQ(cudaMemcpy(table.data(),
                         nvte_workspace_get_region_data(descriptor.data(), persistent_region),
                         table.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(std::all_of(table.begin(), table.end(),
                            [](unsigned char x) { return x == kPersistentTableFillByte; }));
    EXPECT_TRUE(nvte_workspace_region_initialized(descriptor.data(), persistent_region));
  }

  if (isFp8Type(otype)) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output_c.amax(), ref_amax, atol_amax, rtol_amax);
    float ref_scale_inv = 1.f / ref_scale;
    compareResults("scale_inv", output_c.rowwise_scale_inv(), ref_scale_inv, atol_amax, rtol_amax);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_c", output_c, ref_output_c.get(), true, atol, rtol);

  auto [atol_dbias, rtol_dbias] = getTolerances(itype);
  rtol_dbias *= 4;
  compareResults("output_dbias", dbias, ref_output_dbias.get(), true, atol_dbias, rtol_dbias);
}

std::vector<std::vector<size_t>> test_cases = {
  {128, 128},
  {256, 256},
  {768, 1024},
  {256, 65536},
  {2048, 12288},
  {65536, 128},
  {65536, 160},
  {16384, 1616},
  {1, 128},
  {1, 1296},
  {1, 16},
  {5, 160},
  {5, 4, 3, 160},
  {217, 256},
};

}  // namespace;


class CastDBiasTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                      transformer_engine::DType,
                                                                      std::vector<size_t>>> {};

TEST(OperatorTest_CastDBias, WorkspaceDescriptorPreservesPersistentRegion) {
    using namespace test;
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }
    performTest<bf16, fp8e4m3>({128, 128}, /*use_workspace_descriptor=*/true);
}

TEST_P(CastDBiasTestSuite, TestCastDBias) {
    using namespace transformer_engine;
    using namespace test;
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<InputType, OutputType>(size);
      );
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    CastDBiasTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::ValuesIn(test_cases)),
    [](const testing::TestParamInfo<CastDBiasTestSuite::ParamType>& info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
      test::typeName(std::get<1>(info.param));
      const auto& shape = std::get<2>(info.param);
      for ( const auto& s: shape) {
        name += "X" + std::to_string(s);
      }
      return name;
    });
