/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

// All row counts and K must be multiples of 128.
struct GroupedCastConfig {
  std::vector<size_t> Ms;
  size_t K;
  bool transpose;
};

std::vector<std::unique_ptr<Tensor>> make_outputs(const std::string& tag,
                                                  const std::vector<size_t>& Ms, size_t K,
                                                  bool transpose) {
  std::vector<std::unique_ptr<Tensor>> outs;
  for (size_t i = 0; i < Ms.size(); ++i) {
    const std::vector<size_t> shape = {Ms[i], K};
    auto t = std::make_unique<Tensor>(tag + std::to_string(i), shape, DType::kFloat4E2M1,
                                      /*rowwise=*/true, /*columnwise=*/transpose,
                                      NVTE_NVFP4_1D_SCALING);
    t->set_row_scaled_nvfp4(true);
    outs.push_back(std::move(t));
  }
  return outs;
}

std::vector<std::unique_ptr<Tensor>> make_random_inputs(const std::vector<size_t>& Ms, size_t K) {
  std::vector<std::unique_ptr<Tensor>> ins;
  for (size_t i = 0; i < Ms.size(); ++i) {
    ins.push_back(
        std::make_unique<Tensor>("in_" + std::to_string(i), std::vector<size_t>{Ms[i], K},
                                 DType::kBFloat16));
    fillCase<fp32>(ins[i].get(), InputsFillCase::uniform);
    ins[i]->to_cpu();
  }
  return ins;
}

std::vector<bf16> pack_inputs(const std::vector<std::unique_ptr<Tensor>>& ins,
                              const std::vector<size_t>& Ms, size_t K) {
  size_t sum_M = 0;
  for (size_t m : Ms) sum_M += m;
  std::vector<bf16> packed(sum_M * K);
  size_t row_off = 0;
  for (size_t i = 0; i < ins.size(); ++i) {
    std::copy_n(ins[i]->rowwise_cpu_dptr<bf16>(), Ms[i] * K, packed.data() + row_off * K);
    row_off += Ms[i];
  }
  return packed;
}

struct GroupedReference {
  size_t sum_M = 0;
  size_t K = 0;
  size_t row_scale_cols = 0;  // rowwise E4M3 scale stride (shared across experts)
  std::vector<uint8_t> row_data;
  std::vector<uint8_t> row_scale;
  std::vector<float> row_amax;
  std::vector<uint8_t> col_data;
  std::vector<uint8_t> col_scale;
  std::vector<float> col_amax;
  bool transpose = false;
};

GroupedReference build_reference(const std::vector<std::unique_ptr<Tensor>>& ins,
                                 const std::vector<size_t>& Ms, size_t K, bool transpose) {
  const size_t num_tensors = Ms.size();
  size_t sum_M = 0;
  for (size_t m : Ms) sum_M += m;

  auto ref = make_outputs("ref_", Ms, K, transpose);
  QuantizationConfigWrapper quant_config;
  quant_config.set_stochastic_rounding(false);
  for (size_t i = 0; i < num_tensors; ++i) {
    nvte_quantize_v2(ins[i]->data(), ref[i]->data(), quant_config, 0);
  }
  cudaDeviceSynchronize();

  GroupedReference r;
  r.sum_M = sum_M;
  r.K = K;
  r.transpose = transpose;
  r.row_scale_cols = K / 16;  // contiguous (unpadded) rowwise E4M3 scale stride

  r.row_data.resize(sum_M * K / 2);
  r.row_scale.resize(sum_M * r.row_scale_cols);
  r.row_amax.resize(sum_M);
  if (transpose) {
    r.col_data.resize((K / 2) * sum_M);
    r.col_scale.resize((K / 16) * sum_M);
    r.col_amax.resize(num_tensors * K);
  }

  auto gather_scale = [](const fp8e4m3* src, fp8e4m3* dst, size_t rows, size_t cols,
                         size_t src_stride) {
    for (size_t row = 0; row < rows; ++row) {
      std::copy_n(src + row * src_stride, cols, dst + row * cols);
    }
  };

  size_t row_off = 0;
  size_t col_data_off = 0;
  size_t col_scale_off = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    ref[i]->to_cpu();
    const size_t M = Ms[i];
    std::copy_n(reinterpret_cast<const uint8_t*>(ref[i]->rowwise_cpu_dptr<fp4e2m1>()), M * K / 2,
                r.row_data.data() + row_off * (K / 2));
    const std::array<size_t, 4> sd = get_scale_tensor_dims(M, K, 1, 16);
    gather_scale(ref[i]->rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                 reinterpret_cast<fp8e4m3*>(r.row_scale.data()) + row_off * r.row_scale_cols, sd[0],
                 sd[1], sd[3]);
    std::copy_n(ref[i]->cpu_rowwise_amax_ptr<float>(), M, r.row_amax.data() + row_off);
    if (transpose) {
      std::copy_n(reinterpret_cast<const uint8_t*>(ref[i]->columnwise_cpu_dptr<fp4e2m1>()),
                  K * M / 2, r.col_data.data() + col_data_off);
      const std::array<size_t, 4> sd_t = get_scale_tensor_dims(K, M, 1, 16);
      gather_scale(ref[i]->columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                   reinterpret_cast<fp8e4m3*>(r.col_scale.data()) + col_scale_off, sd_t[0], sd_t[1],
                   sd_t[3]);
      std::copy_n(ref[i]->cpu_columnwise_amax_ptr<float>(), K, r.col_amax.data() + i * K);
      col_data_off += K * M / 2;
      col_scale_off += K * (M / 16);
    }
    row_off += M;
  }
  return r;
}

struct GroupedBuffersNVFP4 {
  size_t capacity_rows = 0;
  size_t K = 0;
  size_t num_tensors = 0;
  size_t row_scale_cols = 0;
  bool transpose = false;

  CudaPtr<bf16> in_data;
  CudaPtr<uint8_t> out_data;
  CudaPtr<uint8_t> out_scale;
  CudaPtr<float> out_amax;
  CudaPtr<uint8_t> out_col_data;
  CudaPtr<uint8_t> out_col_scale;
  CudaPtr<float> out_col_amax;
  CudaPtr<int64_t> offsets;
  CudaPtr<int64_t> first_dims;

  std::unique_ptr<GroupedTensorWrapper> input;
  std::unique_ptr<GroupedTensorWrapper> output;

  void allocate(size_t capacity_rows_, size_t K_, size_t num_tensors_, size_t row_scale_cols_,
                bool transpose_) {
    capacity_rows = capacity_rows_;
    K = K_;
    num_tensors = num_tensors_;
    row_scale_cols = row_scale_cols_;
    transpose = transpose_;

    in_data = cuda_alloc<bf16>(capacity_rows * K * sizeof(bf16));
    out_data = cuda_alloc<uint8_t>(capacity_rows * (K / 2));
    out_scale = cuda_alloc<uint8_t>(capacity_rows * row_scale_cols);
    out_amax = cuda_alloc<float>(capacity_rows * sizeof(float));
    offsets = cuda_alloc<int64_t>((num_tensors + 1) * sizeof(int64_t));
    first_dims = cuda_alloc<int64_t>(num_tensors * sizeof(int64_t));
    if (transpose) {
      out_col_data = cuda_alloc<uint8_t>((K / 2) * capacity_rows);
      out_col_scale = cuda_alloc<uint8_t>((K / 16) * capacity_rows);
      out_col_amax = cuda_alloc<float>(num_tensors * K * sizeof(float));
    }

    input = std::make_unique<GroupedTensorWrapper>(num_tensors,
                                                   std::vector<size_t>{capacity_rows, K});
    output = std::make_unique<GroupedTensorWrapper>(
        num_tensors, std::vector<size_t>{capacity_rows, K}, NVTE_NVFP4_1D_SCALING);

    input->set_rowwise_data(in_data.get(), DType::kBFloat16,
                            std::vector<size_t>{capacity_rows, K});
    input->set_tensor_offsets(offsets.get(), DType::kInt64,
                              std::vector<size_t>{num_tensors + 1});
    input->set_first_dims(first_dims.get(), DType::kInt64, std::vector<size_t>{num_tensors});

    output->set_row_scaled_nvfp4(true);
    output->set_rowwise_data(out_data.get(), DType::kFloat4E2M1,
                             std::vector<size_t>{capacity_rows, K});
    output->set_rowwise_scale_inv(out_scale.get(), DType::kFloat8E4M3,
                                  std::vector<size_t>{capacity_rows, row_scale_cols});
    output->set_amax(out_amax.get(), DType::kFloat32, std::vector<size_t>{capacity_rows});
    output->set_tensor_offsets(offsets.get(), DType::kInt64,
                               std::vector<size_t>{num_tensors + 1});
    output->set_first_dims(first_dims.get(), DType::kInt64, std::vector<size_t>{num_tensors});
    if (transpose) {
      output->set_columnwise_data(out_col_data.get(), DType::kFloat4E2M1,
                                  std::vector<size_t>{K, capacity_rows});
      output->set_columnwise_scale_inv(out_col_scale.get(), DType::kFloat8E4M3,
                                       std::vector<size_t>{K, capacity_rows / 16});
      output->set_columnwise_amax(out_col_amax.get(), DType::kFloat32,
                                  std::vector<size_t>{num_tensors * K});
    }
  }

  void set_routing(const std::vector<size_t>& Ms, const GroupedReference& ref,
                   const std::vector<bf16>& packed_in) {
    std::vector<int64_t> off(num_tensors + 1, 0);
    std::vector<int64_t> fd(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
      fd[i] = static_cast<int64_t>(Ms[i]);
      off[i + 1] = off[i] + static_cast<int64_t>(Ms[i] * K);
    }
    NVTE_CHECK_CUDA(cudaMemcpy(offsets.get(), off.data(), (num_tensors + 1) * sizeof(int64_t),
                               cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemcpy(first_dims.get(), fd.data(), num_tensors * sizeof(int64_t),
                               cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemcpy(in_data.get(), packed_in.data(), ref.sum_M * K * sizeof(bf16),
                               cudaMemcpyHostToDevice));
    std::vector<float> poison_row(capacity_rows, -1.0f);
    NVTE_CHECK_CUDA(cudaMemcpy(out_amax.get(), poison_row.data(), capacity_rows * sizeof(float),
                               cudaMemcpyHostToDevice));
    if (transpose) {
      std::vector<float> poison_col(num_tensors * K, -1.0f);
      NVTE_CHECK_CUDA(cudaMemcpy(out_col_amax.get(), poison_col.data(),
                                 num_tensors * K * sizeof(float), cudaMemcpyHostToDevice));
    }
  }

  void quantize(cudaStream_t stream) {
    QuantizationConfigWrapper quant_config;
    nvte_group_quantize(input->data(), output->data(), quant_config, stream);
  }
};

void expect_matches_reference(const GroupedBuffersNVFP4& b, const GroupedReference& ref) {
  const size_t sum_M = ref.sum_M;
  const size_t K = ref.K;

  std::vector<uint8_t> h_data(sum_M * K / 2);
  std::vector<uint8_t> h_scale(sum_M * ref.row_scale_cols);
  std::vector<float> h_amax(sum_M);
  NVTE_CHECK_CUDA(cudaMemcpy(h_data.data(), b.out_data.get(), h_data.size(), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(
      cudaMemcpy(h_scale.data(), b.out_scale.get(), h_scale.size(), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(
      cudaMemcpy(h_amax.data(), b.out_amax.get(), sum_M * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < sum_M; ++i)
    ASSERT_EQ(h_amax[i], ref.row_amax[i]) << "rowwise amax mismatch at row " << i;
  size_t data_mismatch = 0;
  for (size_t i = 0; i < h_data.size(); ++i)
    if (h_data[i] != ref.row_data[i]) ++data_mismatch;
  ASSERT_EQ(data_mismatch, 0u) << "rowwise FP4 data mismatch (" << data_mismatch << "/"
                               << h_data.size() << ")";
  size_t scale_mismatch = 0;
  for (size_t i = 0; i < h_scale.size(); ++i)
    if (h_scale[i] != ref.row_scale[i]) ++scale_mismatch;
  ASSERT_EQ(scale_mismatch, 0u) << "rowwise scale mismatch (" << scale_mismatch << "/"
                                << h_scale.size() << ")";

  if (!ref.transpose) return;

  std::vector<float> h_col_amax(b.num_tensors * K);
  std::vector<uint8_t> h_col_data((K / 2) * sum_M);
  std::vector<uint8_t> h_col_scale((K / 16) * sum_M);
  NVTE_CHECK_CUDA(cudaMemcpy(h_col_amax.data(), b.out_col_amax.get(),
                             h_col_amax.size() * sizeof(float), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(h_col_data.data(), b.out_col_data.get(), h_col_data.size(),
                             cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(h_col_scale.data(), b.out_col_scale.get(), h_col_scale.size(),
                             cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < h_col_amax.size(); ++i)
    ASSERT_EQ(h_col_amax[i], ref.col_amax[i]) << "columnwise amax mismatch at index " << i;
  size_t col_data_mismatch = 0;
  for (size_t i = 0; i < h_col_data.size(); ++i)
    if (h_col_data[i] != ref.col_data[i]) ++col_data_mismatch;
  ASSERT_EQ(col_data_mismatch, 0u) << "columnwise FP4 data mismatch (" << col_data_mismatch << "/"
                                   << h_col_data.size() << ")";
  size_t col_scale_mismatch = 0;
  for (size_t i = 0; i < h_col_scale.size(); ++i)
    if (h_col_scale[i] != ref.col_scale[i]) ++col_scale_mismatch;
  ASSERT_EQ(col_scale_mismatch, 0u) << "columnwise scale mismatch (" << col_scale_mismatch << "/"
                                    << h_col_scale.size() << ")";
}

void performGroupedRowScaledQuantizeTest(const GroupedCastConfig& cfg) {
  const std::vector<size_t>& Ms = cfg.Ms;
  const size_t K = cfg.K;
  const bool transpose = cfg.transpose;
  size_t sum_M = 0;
  for (size_t m : Ms) sum_M += m;

  auto ins = make_random_inputs(Ms, K);
  const GroupedReference ref = build_reference(ins, Ms, K, transpose);
  const std::vector<bf16> packed_in = pack_inputs(ins, Ms, K);

  GroupedBuffersNVFP4 b;
  b.allocate(sum_M, K, Ms.size(), ref.row_scale_cols, transpose);
  b.set_routing(Ms, ref, packed_in);

  b.quantize(0);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

  expect_matches_reference(b, ref);
}

void performGroupedRowScaledQuantizeReplayTest(size_t K, bool transpose) {
  const std::vector<size_t> Ms_a = {256, 128, 128};
  const std::vector<size_t> Ms_b = {128, 256, 128};
  const size_t num_tensors = Ms_a.size();
  const size_t capacity_rows = 768;  // > sum_M (512): exercises the capacity noop

  auto ins_a = make_random_inputs(Ms_a, K);
  auto ins_b = make_random_inputs(Ms_b, K);
  const GroupedReference ref_a = build_reference(ins_a, Ms_a, K, transpose);
  const GroupedReference ref_b = build_reference(ins_b, Ms_b, K, transpose);
  const std::vector<bf16> packed_a = pack_inputs(ins_a, Ms_a, K);
  const std::vector<bf16> packed_b = pack_inputs(ins_b, Ms_b, K);

  GroupedBuffersNVFP4 b;
  b.allocate(capacity_rows, K, num_tensors, ref_a.row_scale_cols, transpose);
  b.set_routing(Ms_a, ref_a, packed_a);

  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

  // Warm up outside capture so one-time setup (e.g. cudaFuncSetAttribute) does
  // not land in the captured graph.
  b.quantize(stream);
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  b.set_routing(Ms_a, ref_a, packed_a);

  cudaGraph_t graph;
  cudaGraphExec_t exec;
  NVTE_CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
  b.quantize(stream);
  NVTE_CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
  NVTE_CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  NVTE_CHECK_CUDA(cudaGraphLaunch(exec, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
  expect_matches_reference(b, ref_a);

  b.set_routing(Ms_b, ref_b, packed_b);
  NVTE_CHECK_CUDA(cudaGraphLaunch(exec, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());
  expect_matches_reference(b, ref_b);

  NVTE_CHECK_CUDA(cudaGraphExecDestroy(exec));
  NVTE_CHECK_CUDA(cudaGraphDestroy(graph));
  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
}

}  // namespace

class NVFP4GroupedRowScaledQuantizeTestSuite
    : public ::testing::TestWithParam<GroupedCastConfig> {};

TEST_P(NVFP4GroupedRowScaledQuantizeTestSuite, MatchesPerExpertQuantize) {
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }
  performGroupedRowScaledQuantizeTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    NVFP4GroupedRowScaledQuantizeRowwise, NVFP4GroupedRowScaledQuantizeTestSuite,
    ::testing::Values(GroupedCastConfig{{128}, 128, false},
                      GroupedCastConfig{{128, 128}, 128, false},
                      GroupedCastConfig{{128, 256, 128}, 256, false},
                      GroupedCastConfig{{256, 128}, 512, false},
                      GroupedCastConfig{{384, 128, 256}, 128, false},
                      GroupedCastConfig{{512}, 1024, false}));

INSTANTIATE_TEST_SUITE_P(
    NVFP4GroupedRowScaledQuantizeTranspose, NVFP4GroupedRowScaledQuantizeTestSuite,
    ::testing::Values(GroupedCastConfig{{128}, 128, true},
                      GroupedCastConfig{{128, 128}, 128, true},
                      GroupedCastConfig{{128, 256, 128}, 256, true},
                      GroupedCastConfig{{256, 128}, 512, true},
                      GroupedCastConfig{{384, 128, 256}, 128, true},
                      GroupedCastConfig{{512}, 1024, true}));

class NVFP4GroupedRowScaledQuantizeReplaySuite
    : public ::testing::TestWithParam<GroupedCastConfig> {};

TEST_P(NVFP4GroupedRowScaledQuantizeReplaySuite, ReplayWithChangedRouting) {
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }
  performGroupedRowScaledQuantizeReplayTest(GetParam().K, GetParam().transpose);
}

INSTANTIATE_TEST_SUITE_P(NVFP4GroupedRowScaledQuantizeReplay,
                         NVFP4GroupedRowScaledQuantizeReplaySuite,
                         ::testing::Values(GroupedCastConfig{{}, 256, false},
                                           GroupedCastConfig{{}, 256, true}));
