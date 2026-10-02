/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/swizzle.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../test_common.h"

using namespace transformer_engine;

namespace {

constexpr size_t kBlock = 128;

size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }
size_t round_up(size_t a, size_t b) { return ceil_div(a, b) * b; }

struct DeviceBuffer {
  void* ptr = nullptr;
  explicit DeviceBuffer(size_t bytes) {
    if (bytes > 0) cudaMalloc(&ptr, bytes);
  }
  ~DeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Same mapping as compute_ref_swizzle<128, 4, true> in test_swizzle.cu: rearranges a compact
// row-major [M, K] scale matrix (M % 128 == 0, K % 4 == 0) into the GEMM-swizzled layout.
void ref_swizzle_rowwise(const uint8_t* in, uint8_t* out, size_t M, size_t K) {
  constexpr size_t kTileM = 128, kTileK = 4, kNewTileM = kTileM / 4, kNewTileK = kTileK * 4;
  for (size_t m = 0; m < M; ++m) {
    for (size_t k = 0; k < K; ++k) {
      const size_t m_in_tile = m % kTileM;
      const size_t row = m_in_tile % kNewTileM;
      const size_t col = (m_in_tile / kNewTileM) * kTileK + k % kTileK;
      const size_t tile_base = (m / kTileM) * kTileM * K + (k / kTileK) * kTileM * kTileK;
      out[tile_base + row * kNewTileK + col] = in[m * K + k];
    }
  }
}

uint8_t e8m0_of(float scale_inv) {
  uint32_t bits;
  std::memcpy(&bits, &scale_inv, sizeof(bits));
  return static_cast<uint8_t>((bits >> 23) & 0xFF);
}

// Compact FP8 block-scaling scale size (floats) of one tensor with rowwise data [rows, cols].
// Matches padded_block_{1d,2d}_scale_inv_floats(.., rowwise=true) in cublaslt_grouped_gemm.cu.
size_t input_scale_floats(bool is_2d, size_t rows, size_t cols) {
  return is_2d ? ceil_div(rows, kBlock) * round_up(ceil_div(cols, kBlock), 4)
               : ceil_div(cols, kBlock) * round_up(rows, 4);
}

// Swizzled MXFP8 scale size (bytes). Matches padded_mxfp8_scale_inv_bytes(.., rowwise=true).
size_t output_scale_bytes(size_t rows, size_t cols) {
  return round_up(rows, kBlock) * ceil_div(cols, kBlock) * 4;
}

std::vector<float> random_pow2_scales(size_t count, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> exp_dist(-20, 20);
  std::vector<float> scales(count);
  for (auto& s : scales) s = std::ldexp(1.0f, exp_dist(gen));
  return scales;
}

// Reference conversion for one tensor with rowwise data [rows, cols]; `in` is its compact FP32
// scale block. Each FP32 scale's exponent byte becomes the E8M0 scale of every 1x32 block it
// covers. 1D scales are stored transposed ([ceil(cols/128), roundup(rows, 4)]); the kernel
// zero-fills rows past that padding. 2D scales cover whole 128x128 tiles.
std::vector<uint8_t> ref_convert(bool is_2d, size_t rows, size_t cols, const float* in) {
  const size_t padded_rows = round_up(rows, kBlock);
  const size_t scale_cols = ceil_div(cols, kBlock) * 4;
  std::vector<uint8_t> compact(padded_rows * scale_cols, 0);
  for (size_t r = 0; r < padded_rows; ++r) {
    for (size_t c = 0; c < scale_cols; ++c) {
      const size_t col_block = c / 4;
      if (is_2d) {
        const size_t in_cols = round_up(ceil_div(cols, kBlock), 4);
        compact[r * scale_cols + c] = e8m0_of(in[(r / kBlock) * in_cols + col_block]);
      } else if (r < round_up(rows, 4)) {
        compact[r * scale_cols + c] = e8m0_of(in[col_block * round_up(rows, 4) + r]);
      }
    }
  }
  std::vector<uint8_t> swizzled(padded_rows * scale_cols);
  ref_swizzle_rowwise(compact.data(), swizzled.data(), padded_rows, scale_cols);
  return swizzled;
}

// ---------------------------------------------------------------------------------------------
// Non-grouped API (refactored onto the shared tile helpers).

struct SingleCase {
  bool is_2d;
  size_t rows;
  size_t cols;
};

class SwizzleBlockScalingToMXFP8Test : public ::testing::TestWithParam<SingleCase> {};

TEST_P(SwizzleBlockScalingToMXFP8Test, MatchesReference) {
  const SingleCase c = GetParam();
  const size_t in_rows = c.is_2d ? ceil_div(c.rows, kBlock) : ceil_div(c.cols, kBlock);
  const size_t in_cols = c.is_2d ? round_up(ceil_div(c.cols, kBlock), 4) : round_up(c.rows, 4);
  const auto scales_h =
      random_pow2_scales(in_rows * in_cols, static_cast<uint32_t>(c.rows * 131 + c.cols));
  const size_t out_rows = round_up(c.rows, kBlock);
  const size_t out_cols = ceil_div(c.cols, kBlock) * 4;

  DeviceBuffer data(c.rows * c.cols), in_scales(scales_h.size() * sizeof(float)),
      out_scales(out_rows * out_cols);
  cudaMemcpy(in_scales.ptr, scales_h.data(), scales_h.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(out_scales.ptr, 0xAB, out_rows * out_cols);

  TensorWrapper input(c.is_2d ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D);
  input.set_rowwise_data(data.ptr, DType::kFloat8E4M3, std::vector<size_t>{c.rows, c.cols});
  input.set_rowwise_scale_inv(in_scales.ptr, DType::kFloat32,
                              std::vector<size_t>{in_rows, in_cols});
  TensorWrapper output(NVTE_MXFP8_1D_SCALING);
  output.set_rowwise_data(data.ptr, DType::kFloat8E4M3, std::vector<size_t>{c.rows, c.cols});
  output.set_rowwise_scale_inv(out_scales.ptr, DType::kFloat8E8M0,
                               std::vector<size_t>{out_rows, out_cols});
  output.set_with_gemm_swizzled_scales(true);

  nvte_swizzle_block_scaling_to_mxfp8_scaling_factors(input.data(), output.data(), 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> out_h(out_rows * out_cols);
  cudaMemcpy(out_h.data(), out_scales.ptr, out_h.size(), cudaMemcpyDeviceToHost);
  EXPECT_EQ(out_h, ref_convert(c.is_2d, c.rows, c.cols, scales_h.data()));
}

std::vector<SingleCase> single_cases() {
  std::vector<SingleCase> cases;
  for (bool is_2d : {false, true}) {
    for (auto [rows, cols] : std::vector<std::pair<size_t, size_t>>{
             {128, 128}, {256, 384}, {384, 2880}, {160, 512}, {144, 528}, {2880, 256}}) {
      cases.push_back({is_2d, rows, cols});
    }
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(SwizzleBlockScaling, SwizzleBlockScalingToMXFP8Test,
                         ::testing::ValuesIn(single_cases()),
                         [](const ::testing::TestParamInfo<SingleCase>& info) {
                           const auto& c = info.param;
                           return std::string(c.is_2d ? "2D" : "1D") + "_" +
                                  std::to_string(c.rows) + "x" + std::to_string(c.cols);
                         });

// ---------------------------------------------------------------------------------------------
// Grouped API.

enum class Rep { kUniform, kVaryingFirst, kVaryingLast };

struct GroupedCase {
  std::string name;
  bool is_2d;
  Rep rep;
  std::vector<size_t> varying;  // per-tensor varying dim (kVaryingFirst / kVaryingLast)
  size_t common;                // shared dim; for kUniform, the last dim
  size_t uniform_first = 0;     // kUniform only
  size_t num_uniform = 0;       // kUniform only
  size_t capacity_extra = 0;    // logical varying total = sum(varying) + capacity_extra
  std::vector<size_t> replay;   // non-empty: capture with `varying`, replay with `replay`
};

std::vector<std::pair<size_t, size_t>> tensor_dims(const GroupedCase& c,
                                                   const std::vector<size_t>& varying) {
  std::vector<std::pair<size_t, size_t>> dims;
  if (c.rep == Rep::kUniform) {
    for (size_t i = 0; i < c.num_uniform; ++i) dims.push_back({c.uniform_first, c.common});
  } else if (c.rep == Rep::kVaryingFirst) {
    for (size_t v : varying) dims.push_back({v, c.common});
  } else {
    for (size_t v : varying) dims.push_back({c.common, v});
  }
  return dims;
}

// Must match grouped_mxfp8_scale_bytes_upper_bound in swizzle_block_scaling.cu and the
// allocation in convert_grouped_block_scaling_to_mxfp8_tensor (PyTorch binding).
size_t upper_bound_bytes(Rep rep, size_t n, size_t F, size_t L) {
  if (F == 0 || L == 0) return 0;
  if (rep == Rep::kUniform) return n * round_up(F / n, kBlock) * ceil_div(L, kBlock) * 4;
  if (rep == Rep::kVaryingFirst) return (F + 127 * n) * ceil_div(L, kBlock) * 4;
  return round_up(F, kBlock) * 4 * (ceil_div(L, kBlock) + n);
}

class GroupedSwizzleBlockScalingToMXFP8Test : public ::testing::TestWithParam<GroupedCase> {};

TEST_P(GroupedSwizzleBlockScalingToMXFP8Test, MatchesPerTensorReference) {
  const GroupedCase c = GetParam();
  const auto capture_dims = tensor_dims(c, c.varying);
  const auto final_dims = c.replay.empty() ? capture_dims : tensor_dims(c, c.replay);
  const size_t n = capture_dims.size();

  size_t varying_total = 0;
  for (size_t v : c.varying) varying_total += v;
  size_t F = 0, L = 0;
  if (c.rep == Rep::kUniform) {
    F = n * c.uniform_first;
    L = c.common;
  } else if (c.rep == Rep::kVaryingFirst) {
    F = varying_total + c.capacity_extra;
    L = c.common;
  } else {
    F = c.common;
    L = varying_total + c.capacity_extra;
  }
  const size_t bound = upper_bound_bytes(c.rep, n, F, L);

  // Input scales: size for the larger of the capture and replay layouts.
  auto total_input_floats = [&](const std::vector<std::pair<size_t, size_t>>& dims) {
    size_t total = 0;
    for (auto [f, l] : dims) total += input_scale_floats(c.is_2d, f, l);
    return total;
  };
  const size_t in_floats =
      std::max(total_input_floats(capture_dims), total_input_floats(final_dims));
  const auto scales_h =
      random_pow2_scales(std::max<size_t>(in_floats, 1), static_cast<uint32_t>(0x5eed + n));

  DeviceBuffer data(std::max<size_t>(F * L, 1)), in_scales(scales_h.size() * sizeof(float)),
      out_scales(std::max<size_t>(bound, 1)), dims_d(n * sizeof(int64_t));
  cudaMemcpy(in_scales.ptr, scales_h.data(), scales_h.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(out_scales.ptr, 0xAB, std::max<size_t>(bound, 1));
  auto upload_dims = [&](const std::vector<size_t>& v) {
    std::vector<int64_t> v64(v.begin(), v.end());
    cudaMemcpy(dims_d.ptr, v64.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
  };
  if (c.rep != Rep::kUniform) upload_dims(c.varying);

  const std::vector<size_t> logical{F, L};
  const NVTEScalingMode mode = c.is_2d ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D;
  GroupedTensorWrapper input(n, logical, mode);
  GroupedTensorWrapper output(n, logical, NVTE_MXFP8_1D_SCALING);
  for (GroupedTensorWrapper* t : {&input, &output}) {
    t->set_rowwise_data(data.ptr, DType::kFloat8E4M3, std::vector<size_t>{F * L});
    if (c.rep == Rep::kVaryingFirst) {
      t->set_first_dims(dims_d.ptr, DType::kInt64, std::vector<size_t>{n});
    } else if (c.rep == Rep::kVaryingLast) {
      t->set_last_dims(dims_d.ptr, DType::kInt64, std::vector<size_t>{n});
    }
  }
  input.set_rowwise_scale_inv(in_scales.ptr, DType::kFloat32, std::vector<size_t>{in_floats});
  output.set_rowwise_scale_inv(out_scales.ptr, DType::kFloat8E8M0, std::vector<size_t>{bound});
  output.set_with_gemm_swizzled_scales(true);

  if (c.replay.empty()) {
    nvte_swizzle_grouped_block_scaling_to_mxfp8_scaling_factors(input.data(), output.data(), 0);
  } else {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
    nvte_swizzle_grouped_block_scaling_to_mxfp8_scaling_factors(input.data(), output.data(),
                                                                stream);
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, 0), cudaSuccess);
    upload_dims(c.replay);  // change the device-side dims after capture
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Expected: per-tensor references at cumulative offsets; untouched sentinel everywhere else
  // (catches writes past the last real tensor, e.g. into graph-capacity slack).
  std::vector<uint8_t> expected(bound, 0xAB);
  size_t in_offset = 0, out_offset = 0;
  for (auto [f, l] : final_dims) {
    const auto ref = ref_convert(c.is_2d, f, l, scales_h.data() + in_offset);
    ASSERT_LE(out_offset + ref.size(), bound);
    std::copy(ref.begin(), ref.end(), expected.begin() + out_offset);
    in_offset += input_scale_floats(c.is_2d, f, l);
    out_offset += output_scale_bytes(f, l);
  }
  std::vector<uint8_t> out_h(bound);
  if (bound > 0) cudaMemcpy(out_h.data(), out_scales.ptr, bound, cudaMemcpyDeviceToHost);
  EXPECT_EQ(out_h, expected);
}

std::vector<GroupedCase> grouped_cases() {
  std::vector<GroupedCase> cases;
  for (bool is_2d : {false, true}) {
    const std::string d = is_2d ? "2D_" : "1D_";
    cases.push_back({d + "uniform_1x128x128", is_2d, Rep::kUniform, {}, 128, 128, 1});
    cases.push_back({d + "uniform_3x256x384", is_2d, Rep::kUniform, {}, 384, 256, 3});
    cases.push_back({d + "uniform_2x2944x2880", is_2d, Rep::kUniform, {}, 2880, 2944, 2});
    cases.push_back({d + "vfirst_basic", is_2d, Rep::kVaryingFirst, {128, 256, 384}, 384});
    cases.push_back(
        {d + "vfirst_zero_mid_k144", is_2d, Rep::kVaryingFirst, {256, 0, 128, 512}, 144});
    cases.push_back(
        {d + "vfirst_64_experts", is_2d, Rep::kVaryingFirst, std::vector<size_t>(64, 128), 256});
    cases.push_back(
        {d + "vfirst_zero_runs", is_2d, Rep::kVaryingFirst, {0, 0, 128, 0, 256, 0, 0}, 256});
    // More than 48 KiB of per-tensor tables, so the launch opts into extra shared memory.
    cases.push_back({d + "vfirst_2048_experts", is_2d, Rep::kVaryingFirst,
                     std::vector<size_t>(2048, 128), 128});
    GroupedCase capacity{d + "vfirst_capacity", is_2d, Rep::kVaryingFirst, {128, 256}, 512};
    capacity.capacity_extra = 256;
    cases.push_back(capacity);
    cases.push_back({d + "vlast_basic", is_2d, Rep::kVaryingLast, {128, 256, 384}, 384});
    cases.push_back({d + "vlast_k2880_zero_mid", is_2d, Rep::kVaryingLast, {256, 0, 128}, 2880});
    cases.push_back({d + "vlast_k144", is_2d, Rep::kVaryingLast, {128, 128}, 144});
    GroupedCase replay_first{
        d + "graph_replay_vfirst", is_2d, Rep::kVaryingFirst, {128, 256, 128, 0}, 256};
    replay_first.replay = {256, 0, 128, 128};
    cases.push_back(replay_first);
    GroupedCase replay_last{
        d + "graph_replay_vlast", is_2d, Rep::kVaryingLast, {128, 256, 128, 0}, 384};
    replay_last.replay = {0, 128, 256, 128};
    cases.push_back(replay_last);
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(SwizzleBlockScaling, GroupedSwizzleBlockScalingToMXFP8Test,
                         ::testing::ValuesIn(grouped_cases()),
                         [](const ::testing::TestParamInfo<GroupedCase>& info) {
                           return info.param.name;
                         });

}  // namespace
