/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

// (per-expert row counts, shared column count); all values are multiples of 128.
struct GroupedAmaxConfig {
  std::vector<size_t> Ms;
  size_t K;
};

std::vector<std::unique_ptr<Tensor>> make_row_scaled_outputs(const std::string& tag,
                                                             const std::vector<size_t>& Ms,
                                                             size_t K) {
  std::vector<std::unique_ptr<Tensor>> outs;
  for (size_t i = 0; i < Ms.size(); ++i) {
    const std::vector<size_t> shape = {Ms[i], K};
    auto t = std::make_unique<Tensor>(tag + std::to_string(i), shape, DType::kFloat4E2M1,
                                      /*rowwise=*/true, /*columnwise=*/true, NVTE_NVFP4_1D_SCALING);
    t->set_row_scaled_nvfp4(true);
    outs.push_back(std::move(t));
  }
  return outs;
}

// Grouped fused amax over a packed (sum_M, K) input must equal the per-row /
// per-col amax of the same BF16 bytes and be deterministic across launches.
void performGroupedFusedAmaxTest(const GroupedAmaxConfig& cfg) {
  const std::vector<size_t>& Ms = cfg.Ms;
  const size_t K = cfg.K;
  const size_t num_tensors = Ms.size();
  size_t sum_M = 0;
  for (size_t m : Ms) sum_M += m;

  // Per-expert BF16 inputs; packed into one buffer with identical bytes so the
  // grouped kernel and the per-expert oracle see the same values.
  std::vector<std::unique_ptr<Tensor>> ins;
  for (size_t i = 0; i < num_tensors; ++i) {
    const std::vector<size_t> shape = {Ms[i], K};
    ins.push_back(std::make_unique<Tensor>("in_" + std::to_string(i), shape, DType::kBFloat16));
    fillCase<fp32>(ins[i].get(), InputsFillCase::uniform);
    ins[i]->to_cpu();
  }

  const std::vector<size_t> packed_shape = {sum_M, K};
  Tensor packed("packed_input", packed_shape, DType::kBFloat16);
  bf16* pdst = packed.rowwise_cpu_dptr<bf16>();
  size_t row_off = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    std::copy_n(ins[i]->rowwise_cpu_dptr<bf16>(), Ms[i] * K, pdst + row_off * K);
    row_off += Ms[i];
  }
  packed.from_cpu();

  const std::vector<size_t> splits(Ms);

  auto outs = make_row_scaled_outputs("gout_", Ms, K);
  std::vector<NVTETensor> out_handles;
  for (auto& t : outs) out_handles.push_back(t->data());
  nvte_group_nvfp4_compute_amax(packed.data(), out_handles.data(), splits.data(), num_tensors, 0);

  auto outs2 = make_row_scaled_outputs("gout2_", Ms, K);
  std::vector<NVTETensor> out_handles2;
  for (auto& t : outs2) out_handles2.push_back(t->data());
  nvte_group_nvfp4_compute_amax(packed.data(), out_handles2.data(), splits.data(), num_tensors, 0);

  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

  for (size_t i = 0; i < num_tensors; ++i) {
    const bf16* src = ins[i]->rowwise_cpu_dptr<bf16>();
    std::vector<float> row_ref(Ms[i], 0.0f);
    std::vector<float> col_ref(K, 0.0f);
    for (size_t r = 0; r < Ms[i]; ++r) {
      for (size_t c = 0; c < K; ++c) {
        const float v = fabsf(static_cast<float>(src[r * K + c]));
        row_ref[r] = fmaxf(row_ref[r], v);
        col_ref[c] = fmaxf(col_ref[c], v);
      }
    }

    outs[i]->to_cpu();
    outs2[i]->to_cpu();

    const float* row_fused = outs[i]->cpu_rowwise_amax_ptr<float>();
    const float* row_fused2 = outs2[i]->cpu_rowwise_amax_ptr<float>();
    for (size_t r = 0; r < Ms[i]; ++r) {
      ASSERT_EQ(row_fused[r], row_ref[r]) << "rowwise amax mismatch: expert " << i << " row " << r;
      ASSERT_EQ(row_fused[r], row_fused2[r])
          << "rowwise amax nondeterministic: expert " << i << " row " << r;
    }

    const float* col_fused = outs[i]->cpu_columnwise_amax_ptr<float>();
    const float* col_fused2 = outs2[i]->cpu_columnwise_amax_ptr<float>();
    for (size_t c = 0; c < K; ++c) {
      ASSERT_EQ(col_fused[c], col_ref[c]) << "columnwise amax mismatch: expert " << i << " col " << c;
      ASSERT_EQ(col_fused[c], col_fused2[c])
          << "columnwise amax nondeterministic: expert " << i << " col " << c;
    }
  }
}

}  // namespace

class NVFP4GroupedRowScaledAmaxTestSuite : public ::testing::TestWithParam<GroupedAmaxConfig> {};

TEST_P(NVFP4GroupedRowScaledAmaxTestSuite, MatchesPerExpertOracle) {
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }
  performGroupedFusedAmaxTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(NVFP4GroupedRowScaledAmax, NVFP4GroupedRowScaledAmaxTestSuite,
                         ::testing::Values(GroupedAmaxConfig{{128}, 128},
                                           GroupedAmaxConfig{{128, 128}, 128},
                                           GroupedAmaxConfig{{128, 256, 128}, 256},
                                           GroupedAmaxConfig{{256, 128}, 512},
                                           GroupedAmaxConfig{{384, 128, 256}, 128},
                                           GroupedAmaxConfig{{512}, 1024}));

// Experts opt into the columnwise direction independently: the kernel must skip
// experts whose columnwise buffer is null (including the first one) instead of
// dereferencing it, while still computing the direction for those that request it.
TEST(NVFP4GroupedRowScaledAmaxTestSuite, HeterogeneousColumnwise) {
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }

  const std::vector<size_t> Ms = {128, 128, 128};
  const size_t K = 256;
  const size_t num_tensors = Ms.size();
  const bool want_col[3] = {false, true, false};

  size_t sum_M = 0;
  for (size_t m : Ms) sum_M += m;

  std::vector<std::unique_ptr<Tensor>> ins;
  for (size_t i = 0; i < num_tensors; ++i) {
    ins.push_back(std::make_unique<Tensor>("hin_" + std::to_string(i),
                                           std::vector<size_t>{Ms[i], K}, DType::kBFloat16));
    fillCase<fp32>(ins[i].get(), InputsFillCase::uniform);
    ins[i]->to_cpu();
  }

  Tensor packed("hpacked", std::vector<size_t>{sum_M, K}, DType::kBFloat16);
  bf16* pdst = packed.rowwise_cpu_dptr<bf16>();
  size_t row_off = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    std::copy_n(ins[i]->rowwise_cpu_dptr<bf16>(), Ms[i] * K, pdst + row_off * K);
    row_off += Ms[i];
  }
  packed.from_cpu();

  std::vector<std::unique_ptr<Tensor>> outs;
  for (size_t i = 0; i < num_tensors; ++i) {
    auto t = std::make_unique<Tensor>("hout_" + std::to_string(i),
                                      std::vector<size_t>{Ms[i], K}, DType::kFloat4E2M1,
                                      /*rowwise=*/true, /*columnwise=*/want_col[i],
                                      NVTE_NVFP4_1D_SCALING);
    t->set_row_scaled_nvfp4(true);
    outs.push_back(std::move(t));
  }
  std::vector<NVTETensor> handles;
  for (auto& t : outs) handles.push_back(t->data());
  const std::vector<size_t> splits(Ms);
  nvte_group_nvfp4_compute_amax(packed.data(), handles.data(), splits.data(), num_tensors, 0);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

  for (size_t i = 0; i < num_tensors; ++i) {
    const bf16* src = ins[i]->rowwise_cpu_dptr<bf16>();
    std::vector<float> row_ref(Ms[i], 0.0f);
    std::vector<float> col_ref(K, 0.0f);
    for (size_t r = 0; r < Ms[i]; ++r) {
      for (size_t c = 0; c < K; ++c) {
        const float v = fabsf(static_cast<float>(src[r * K + c]));
        row_ref[r] = fmaxf(row_ref[r], v);
        col_ref[c] = fmaxf(col_ref[c], v);
      }
    }

    outs[i]->to_cpu();
    const float* row = outs[i]->cpu_rowwise_amax_ptr<float>();
    for (size_t r = 0; r < Ms[i]; ++r) {
      ASSERT_EQ(row[r], row_ref[r]) << "rowwise amax mismatch: expert " << i << " row " << r;
    }
    if (want_col[i]) {
      const float* col = outs[i]->cpu_columnwise_amax_ptr<float>();
      for (size_t c = 0; c < K; ++c) {
        ASSERT_EQ(col[c], col_ref[c]) << "columnwise amax mismatch: expert " << i << " col " << c;
      }
    }
  }
}

namespace {

double median_ms(const std::function<void()>& fn, int iters) {
  for (int i = 0; i < 10; ++i) fn();
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  std::vector<float> ts;
  ts.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    cudaEventRecord(start);
    fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ts.push_back(ms);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

void runLoopVsGroupedBench(size_t num_tensors, size_t M, size_t K, int iters) {
  const std::vector<size_t> Ms(num_tensors, M);
  size_t sum_M = M * num_tensors;

  Tensor packed("packed", std::vector<size_t>{sum_M, K}, DType::kBFloat16);
  fillCase<fp32>(&packed, InputsFillCase::uniform);

  std::vector<std::unique_ptr<Tensor>> ins;
  for (size_t i = 0; i < num_tensors; ++i) {
    ins.push_back(std::make_unique<Tensor>("in_" + std::to_string(i),
                                           std::vector<size_t>{M, K}, DType::kBFloat16));
    fillCase<fp32>(ins[i].get(), InputsFillCase::uniform);
  }

  auto outs_g = make_row_scaled_outputs("bg_", Ms, K);
  std::vector<NVTETensor> h_g;
  for (auto& t : outs_g) h_g.push_back(t->data());
  std::vector<size_t> splits(Ms);

  auto outs_l = make_row_scaled_outputs("bl_", Ms, K);

  double grouped = median_ms(
      [&]() {
        nvte_group_nvfp4_compute_amax(packed.data(), h_g.data(), splits.data(), num_tensors, 0);
      },
      iters);

  double loop = median_ms(
      [&]() {
        for (size_t i = 0; i < num_tensors; ++i) {
          NVTETensor one = outs_l[i]->data();
          size_t s1 = M;
          nvte_group_nvfp4_compute_amax(ins[i]->data(), &one, &s1, 1, 0);
        }
      },
      iters);

  std::printf("%4zu experts  M=%5zu  K=%6zu   loop=%8.4f ms  grouped=%8.4f ms  speedup=%.2fx\n",
              num_tensors, M, K, loop, grouped, loop / grouped);
}

}  // namespace

TEST(NVFP4GroupedRowScaledAmaxBench, DISABLED_LoopVsGrouped) {
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }
  const int iters = 100;
  for (size_t M : {size_t(128), size_t(512)}) {
    for (size_t n : {size_t(2), size_t(4), size_t(8), size_t(16), size_t(32), size_t(64)}) {
      runLoopVsGroupedBench(n, M, 4096, iters);
    }
    std::printf("\n");
  }
}
