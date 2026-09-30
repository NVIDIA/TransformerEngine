/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum ShapeRepresentation { SAME_BOTH_DIMS = 0, VARYING_FIRST_DIM = 1 };

enum AmaxMode { SHARED_AMAX = 0, PER_TENSOR_AMAX = 1 };

constexpr size_t kBlockSize = 16;

/**
 * Compare grouped NVFP4 dequantize output against single-tensor nvte_dequantize
 * called in a loop for each tensor. Results must be bitwise identical.
 *
 * All tensors share the last dimension and every first dimension is a multiple
 * of 128, so the padded per-tensor scale layout is one dense [rows, K / 16] array.
 */
template <typename OutputType>
void performTest(const ShapeRepresentation shape_rep, const size_t num_tensors,
                 const std::vector<size_t> &logical_shape_vec,
                 const std::vector<size_t> &first_dims_h, const std::vector<size_t> &offsets_h,
                 const AmaxMode amax_mode) {
  const DType otype = TypeInfo<OutputType>::dtype;

  const size_t rows = logical_shape_vec[0];
  const size_t cols = logical_shape_vec[1];
  const size_t elts_num = rows * cols;
  const size_t data_bytes = elts_num / 2;
  const size_t scale_cols = cols / kBlockSize;
  const size_t total_scales = rows * scale_cols;
  const size_t amax_num = (amax_mode == PER_TENSOR_AMAX) ? num_tensors : 1;

  // Generate random FP4 data, E4M3 scales and amax values
  std::vector<uint8_t> in_data_h(data_bytes);
  std::vector<uint8_t> in_scales_h(total_scales);
  std::vector<float> amax_h(amax_num);

  static std::mt19937 gen(42);
  std::uniform_int_distribution<int> byte_dis(0, 255);
  // Positive, finite E4M3 encodings (0x7F is NaN)
  std::uniform_int_distribution<int> scale_dis(0, 0x7E);
  std::uniform_real_distribution<float> amax_dis(0.5f, 1000.0f);

  for (size_t i = 0; i < data_bytes; ++i) {
    in_data_h[i] = static_cast<uint8_t>(byte_dis(gen));
  }
  for (size_t i = 0; i < total_scales; ++i) {
    in_scales_h[i] = static_cast<uint8_t>(scale_dis(gen));
  }
  for (size_t i = 0; i < amax_num; ++i) {
    amax_h[i] = amax_dis(gen);
  }

  // Allocate device memory
  const size_t out_data_size = elts_num * sizeof(OutputType);

  uint8_t *in_data_d;
  OutputType *out_grouped_d;
  uint8_t *in_scales_d;
  float *amax_d;
  int64_t *first_dims_d;
  int64_t *offsets_d;

  cudaMalloc((void **)&in_data_d, data_bytes);
  cudaMalloc((void **)&out_grouped_d, out_data_size);
  cudaMalloc((void **)&in_scales_d, total_scales);
  cudaMalloc((void **)&amax_d, amax_num * sizeof(float));
  cudaMalloc((void **)&first_dims_d, num_tensors * sizeof(int64_t));
  cudaMalloc((void **)&offsets_d, (num_tensors + 1) * sizeof(int64_t));

  std::vector<int64_t> first_dims_i64(first_dims_h.begin(), first_dims_h.end());
  std::vector<int64_t> offsets_i64(offsets_h.begin(), offsets_h.end());

  cudaMemcpy(in_data_d, in_data_h.data(), data_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(in_scales_d, in_scales_h.data(), total_scales, cudaMemcpyHostToDevice);
  cudaMemcpy(amax_d, amax_h.data(), amax_num * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(first_dims_d, first_dims_i64.data(), num_tensors * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(offsets_d, offsets_i64.data(), (num_tensors + 1) * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemset(out_grouped_d, 0, out_data_size);

  // Set up grouped input and output tensors. Data tensors are 1D and sized in
  // elements, so the FP4 buffer holds two elements per byte.
  const std::vector<size_t> data_shape = {elts_num};
  const std::vector<size_t> scales_shape = {total_scales};
  const std::vector<size_t> amax_shape = {amax_num};
  const std::vector<size_t> first_dims_shape = {num_tensors};
  const std::vector<size_t> offsets_shape = {num_tensors + 1};

  GroupedTensorWrapper in_group(num_tensors, logical_shape_vec, NVTE_NVFP4_1D_SCALING);
  in_group.set_rowwise_data(in_data_d, DType::kFloat4E2M1, data_shape);
  in_group.set_rowwise_scale_inv(in_scales_d, DType::kFloat8E4M3, scales_shape);
  in_group.set_amax(amax_d, DType::kFloat32, amax_shape);

  GroupedTensorWrapper out_group(num_tensors, logical_shape_vec);
  out_group.set_rowwise_data(out_grouped_d, otype, data_shape);

  if (shape_rep == VARYING_FIRST_DIM) {
    in_group.set_first_dims(first_dims_d, DType::kInt64, first_dims_shape);
    in_group.set_tensor_offsets(offsets_d, DType::kInt64, offsets_shape);
    out_group.set_first_dims(first_dims_d, DType::kInt64, first_dims_shape);
    out_group.set_tensor_offsets(offsets_d, DType::kInt64, offsets_shape);
  }

  // Run grouped dequantize
  nvte_group_dequantize(in_group.data(), out_group.data(), 0);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  std::vector<OutputType> out_grouped_h(elts_num);
  cudaMemcpy(out_grouped_h.data(), out_grouped_d, out_data_size, cudaMemcpyDeviceToHost);

  // Reference: single-tensor nvte_dequantize for each tensor
  std::vector<OutputType> out_ref_h(elts_num);

  for (size_t t = 0; t < num_tensors; ++t) {
    const size_t M = first_dims_h[t];
    const size_t K = cols;
    const size_t data_offset = offsets_h[t];
    const size_t row_offset = data_offset / K;

    const size_t single_data_bytes = M * K / 2;
    const size_t single_out_size = M * K * sizeof(OutputType);
    const size_t single_scales_size = M * scale_cols;

    uint8_t *single_in_d;
    OutputType *single_out_d;
    uint8_t *single_scales_d;
    float *single_amax_d;

    cudaMalloc((void **)&single_in_d, single_data_bytes);
    cudaMalloc((void **)&single_out_d, single_out_size);
    cudaMalloc((void **)&single_scales_d, single_scales_size);
    cudaMalloc((void **)&single_amax_d, sizeof(float));

    const float tensor_amax = amax_h[(amax_mode == PER_TENSOR_AMAX) ? t : 0];
    cudaMemcpy(single_in_d, in_data_h.data() + data_offset / 2, single_data_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(single_scales_d, in_scales_h.data() + row_offset * scale_cols, single_scales_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(single_amax_d, &tensor_amax, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(single_out_d, 0, single_out_size);

    const std::vector<size_t> single_shape = {M, K};
    const std::vector<size_t> scale_shape_vec = {M, scale_cols};
    const std::vector<size_t> single_amax_shape = {1};

    TensorWrapper input_w(NVTE_NVFP4_1D_SCALING);
    input_w.set_rowwise_data(single_in_d, DType::kFloat4E2M1, single_shape);
    input_w.set_rowwise_scale_inv(single_scales_d, DType::kFloat8E4M3, scale_shape_vec);
    input_w.set_amax(single_amax_d, DType::kFloat32, single_amax_shape);

    TensorWrapper output_w;
    output_w.set_rowwise_data(single_out_d, otype, single_shape);

    nvte_dequantize(input_w.data(), output_w.data(), 0);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Single-tensor dequantize failed for tensor " << t << ": "
                                << cudaGetErrorString(err);

    cudaMemcpy(out_ref_h.data() + data_offset, single_out_d, single_out_size,
               cudaMemcpyDeviceToHost);

    cudaFree(single_in_d);
    cudaFree(single_out_d);
    cudaFree(single_scales_d);
    cudaFree(single_amax_d);
  }

  // Bitwise comparison
  for (size_t t = 0; t < num_tensors; ++t) {
    const size_t data_offset = offsets_h[t];
    const size_t tensor_elts = first_dims_h[t] * cols;

    int result = memcmp(out_grouped_h.data() + data_offset, out_ref_h.data() + data_offset,
                        tensor_elts * sizeof(OutputType));
    if (result != 0) {
      for (size_t i = 0; i < tensor_elts; ++i) {
        if (memcmp(&out_grouped_h[data_offset + i], &out_ref_h[data_offset + i],
                   sizeof(OutputType)) != 0) {
          GTEST_FAIL() << "Bitwise mismatch at tensor " << t << " element " << i
                       << " (global offset " << (data_offset + i) << "): grouped="
                       << static_cast<float>(out_grouped_h[data_offset + i])
                       << " vs reference=" << static_cast<float>(out_ref_h[data_offset + i]);
        }
      }
    }
  }

  cudaFree(in_data_d);
  cudaFree(out_grouped_d);
  cudaFree(in_scales_d);
  cudaFree(amax_d);
  cudaFree(first_dims_d);
  cudaFree(offsets_d);
}

// {shape_representation, num_tensors, [logical_shape_M, logical_shape_K], [M_i]}
std::vector<std::vector<size_t>> input_configs = {
    {SAME_BOTH_DIMS, 1, 128, 128},
    {SAME_BOTH_DIMS, 2, 256, 256},
    {SAME_BOTH_DIMS, 4, 512, 2048},
    {VARYING_FIRST_DIM, 2, 384, 128, 128, 256},
    {VARYING_FIRST_DIM, 3, 896, 512, 256, 128, 512},
    {VARYING_FIRST_DIM, 5, 4096, 512, 128, 256, 384, 1024, 2304},
};

std::vector<AmaxMode> amax_modes = {
    AmaxMode::SHARED_AMAX,
    AmaxMode::PER_TENSOR_AMAX,
};

}  // namespace

class GroupedDequantizeNVFP4TestSuite
    : public ::testing::TestWithParam<std::tuple<AmaxMode,
                                                 std::vector<size_t>,       // Config
                                                 transformer_engine::DType  // OutputType
                                                 >> {};

TEST_P(GroupedDequantizeNVFP4TestSuite, TestGroupedDequantizeNVFP4) {
  // Skip tests for pre-Blackwell architectures
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }

  using namespace transformer_engine;
  using namespace test;

  const AmaxMode amax_mode = std::get<0>(GetParam());
  const std::vector<size_t> config = std::get<1>(GetParam());
  const DType output_type = std::get<2>(GetParam());

  const ShapeRepresentation shape_rep = static_cast<ShapeRepresentation>(config[0]);
  const size_t num_tensors = config[1];
  const std::vector<size_t> logical_shape = {config[2], config[3]};

  // A shared amax with one tensor is the same case as a per-tensor amax
  if (amax_mode == PER_TENSOR_AMAX && num_tensors == 1) {
    GTEST_SKIP();
  }

  std::vector<size_t> first_dims(num_tensors);
  std::vector<size_t> offsets(num_tensors + 1, 0);
  for (size_t t = 0; t < num_tensors; ++t) {
    first_dims[t] =
        (shape_rep == SAME_BOTH_DIMS) ? logical_shape[0] / num_tensors : config[t + 4];
    offsets[t + 1] = offsets[t] + first_dims[t] * logical_shape[1];
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
      output_type, OutputType,
      performTest<OutputType>(shape_rep, num_tensors, logical_shape, first_dims, offsets,
                              amax_mode););
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, GroupedDequantizeNVFP4TestSuite,
    ::testing::Combine(::testing::ValuesIn(amax_modes), ::testing::ValuesIn(input_configs),
                       ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16)),
    [](const testing::TestParamInfo<GroupedDequantizeNVFP4TestSuite::ParamType> &info) {
      std::string name;
      switch (std::get<0>(info.param)) {
        case AmaxMode::SHARED_AMAX:
          name += "SHARED_AMAX_";
          break;
        case AmaxMode::PER_TENSOR_AMAX:
          name += "PER_TENSOR_AMAX_";
          break;
      }

      const std::vector<size_t> input = std::get<1>(info.param);
      switch (static_cast<ShapeRepresentation>(input[0])) {
        case ShapeRepresentation::SAME_BOTH_DIMS:
          name += "SAME_BOTH_DIMS";
          break;
        case ShapeRepresentation::VARYING_FIRST_DIM:
          name += "VARYING_FIRST_DIM";
          break;
      }

      name += "_N_" + std::to_string(input[1]);
      name += "_SHAPE_" + std::to_string(input[2]) + "X" + std::to_string(input[3]);
      name += "_" + test::typeName(std::get<2>(info.param));
      return name;
    });
