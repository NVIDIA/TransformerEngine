/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum ShapeRepresentation {
  SAME_BOTH_DIMS = 0,
  VARYING_FIRST_DIM = 1,
  VARYING_LAST_DIM = 2,
  VARYING_BOTH_DIMS = 3
};

enum ScalingDirection { ROWWISE = 0, COLWISE = 1 };

/**
 * Compare grouped dequantize output against single-tensor nvte_dequantize
 * called in a loop for each tensor. Results must be bitwise identical.
 */
template <typename InputType, typename OutputType>
void performTest(const ShapeRepresentation shape_rep, const size_t num_tensors,
                 const std::vector<size_t> &logical_shape_vec,
                 const std::vector<size_t> &first_dims_h, const std::vector<size_t> &last_dims_h,
                 const std::vector<size_t> &offsets_h, const bool rowwise) {
  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  const size_t rows = logical_shape_vec[0];
  const size_t cols = logical_shape_vec[1];

  // Compute total elements and per-tensor scale sizes
  size_t elts_num = 0;
  size_t total_scales = 0;

  std::vector<size_t> per_tensor_scales_first_dim(num_tensors);
  std::vector<size_t> per_tensor_scales_last_dim(num_tensors);
  std::vector<size_t> per_tensor_scales_offset(num_tensors + 1, 0);

  for (size_t t = 0; t < num_tensors; ++t) {
    const size_t M = first_dims_h[t];
    const size_t K = last_dims_h[t];
    elts_num += M * K;

    size_t unpadded_scales_Y, unpadded_scales_X;
    if (rowwise) {
      unpadded_scales_Y = M;
      unpadded_scales_X = divide_round_up(K, 32);
      per_tensor_scales_first_dim[t] =
          round_up_to_nearest_multiple(unpadded_scales_Y, scale_tensor_alignment_Y_rowwise);
      per_tensor_scales_last_dim[t] =
          round_up_to_nearest_multiple(unpadded_scales_X, scale_tensor_alignment_X_rowwise);
    } else {
      unpadded_scales_Y = divide_round_up(M, 32);
      unpadded_scales_X = K;
      per_tensor_scales_first_dim[t] =
          round_up_to_nearest_multiple(unpadded_scales_Y, scale_tensor_alignment_Y_colwise);
      per_tensor_scales_last_dim[t] =
          round_up_to_nearest_multiple(unpadded_scales_X, scale_tensor_alignment_X_colwise);
    }

    const size_t tensor_scales = per_tensor_scales_first_dim[t] * per_tensor_scales_last_dim[t];
    total_scales += tensor_scales;
    per_tensor_scales_offset[t + 1] = total_scales;
  }

  // Allocate host data
  std::vector<InputType> in_data_h(elts_num);
  std::vector<fp8e8m0> in_scales_h(total_scales);

  // Generate random FP8 data and scales
  static std::mt19937 gen(42);
  const double minAbs = Numeric_Traits<InputType>::minNorm;
  const double maxAbs = Numeric_Traits<InputType>::maxNorm;
  std::uniform_real_distribution<> dis(minAbs, maxAbs);
  std::uniform_real_distribution<> dis_sign(-1.0, 1.0);
  std::uniform_int_distribution<int> int_dis(0, 255);

  for (size_t i = 0; i < elts_num; ++i) {
    const bool is_negative = (dis_sign(gen) < 0.0);
    double val = dis(gen);
    if (is_negative) val = -val;
    in_data_h[i] = static_cast<InputType>(val);
  }
  for (size_t i = 0; i < total_scales; ++i) {
    in_scales_h[i] = int_dis(gen);
  }

  // Allocate device memory
  const size_t in_data_size = elts_num * sizeof(InputType);
  const size_t out_data_size = elts_num * sizeof(OutputType);
  const size_t scales_size = total_scales * sizeof(fp8e8m0);
  const size_t first_dims_size = num_tensors * sizeof(size_t);
  const size_t last_dims_size = num_tensors * sizeof(size_t);
  const size_t offsets_size = (num_tensors + 1) * sizeof(size_t);

  InputType *in_data_d;
  OutputType *out_grouped_d;
  fp8e8m0 *in_scales_d;
  size_t *first_dims_d;
  size_t *last_dims_d;
  size_t *offsets_d;

  cudaMalloc((void **)&in_data_d, in_data_size);
  cudaMalloc((void **)&out_grouped_d, out_data_size);
  cudaMalloc((void **)&in_scales_d, scales_size);
  cudaMalloc((void **)&first_dims_d, first_dims_size);
  cudaMalloc((void **)&last_dims_d, last_dims_size);
  cudaMalloc((void **)&offsets_d, offsets_size);

  cudaMemcpy(in_data_d, in_data_h.data(), in_data_size, cudaMemcpyHostToDevice);
  cudaMemcpy(in_scales_d, in_scales_h.data(), scales_size, cudaMemcpyHostToDevice);
  cudaMemcpy(first_dims_d, first_dims_h.data(), first_dims_size, cudaMemcpyHostToDevice);
  cudaMemcpy(last_dims_d, last_dims_h.data(), last_dims_size, cudaMemcpyHostToDevice);
  cudaMemcpy(offsets_d, offsets_h.data(), offsets_size, cudaMemcpyHostToDevice);

  // Set up grouped input tensor
  NVTEShape logical_shape = nvte_make_shape(logical_shape_vec.data(), logical_shape_vec.size());

  NVTEShape first_dims_shape;
  NVTEShape last_dims_shape;
  NVTEShape offsets_shape;
  first_dims_shape.ndim = 1;
  last_dims_shape.ndim = 1;
  offsets_shape.ndim = 1;
  first_dims_shape.data[0] = num_tensors;
  last_dims_shape.data[0] = num_tensors;
  offsets_shape.data[0] = num_tensors + 1;

  // Data tensors must be 1D (flattened)
  std::vector<size_t> data_1d_shape = {elts_num};
  NVTEShape data_shape = nvte_make_shape(data_1d_shape.data(), data_1d_shape.size());

  std::vector<size_t> scales_1d_shape = {total_scales};
  NVTEShape scales_shape = nvte_make_shape(scales_1d_shape.data(), scales_1d_shape.size());

  NVTEGroupedTensor in_group_tensor =
      nvte_create_grouped_tensor(NVTE_MXFP8_1D_SCALING, num_tensors, logical_shape);

  // Set input data (rowwise or columnwise) - data shape must be 1D
  NVTEBasicTensor in_data_tensor = {in_data_d, static_cast<NVTEDType>(itype), data_shape};
  if (rowwise) {
    nvte_set_grouped_tensor_param(in_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedRowwiseData, &in_data_tensor,
                                  sizeof(in_data_tensor));
  } else {
    nvte_set_grouped_tensor_param(in_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedColumnwiseData,
                                  &in_data_tensor, sizeof(in_data_tensor));
  }

  // Set scales
  NVTEBasicTensor in_scales_tensor = {in_scales_d, NVTEDType::kNVTEFloat8E8M0, scales_shape};
  if (rowwise) {
    nvte_set_grouped_tensor_param(in_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedRowwiseScaleInv,
                                  &in_scales_tensor, sizeof(in_scales_tensor));
  } else {
    nvte_set_grouped_tensor_param(in_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedColumnwiseScaleInv,
                                  &in_scales_tensor, sizeof(in_scales_tensor));
  }

  // Set shape arrays
  if ((shape_rep == VARYING_FIRST_DIM) || (shape_rep == VARYING_BOTH_DIMS)) {
    NVTEBasicTensor first_dims_tensor = {first_dims_d, kNVTEInt64, first_dims_shape};
    nvte_set_grouped_tensor_param(in_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedFirstDims,
                                  &first_dims_tensor, sizeof(first_dims_tensor));
  }
  if ((shape_rep == VARYING_LAST_DIM) || (shape_rep == VARYING_BOTH_DIMS)) {
    NVTEBasicTensor last_dims_tensor = {last_dims_d, kNVTEInt64, last_dims_shape};
    nvte_set_grouped_tensor_param(in_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedLastDims, &last_dims_tensor,
                                  sizeof(last_dims_tensor));
  }
  if (shape_rep != SAME_BOTH_DIMS) {
    NVTEBasicTensor offsets_tensor = {offsets_d, kNVTEInt64, offsets_shape};
    nvte_set_grouped_tensor_param(in_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets,
                                  &offsets_tensor, sizeof(offsets_tensor));
  }

  // Set up grouped output tensor
  NVTEGroupedTensor out_group_tensor =
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape);

  NVTEBasicTensor out_data_tensor = {out_grouped_d, static_cast<NVTEDType>(otype), data_shape};
  nvte_set_grouped_tensor_param(out_group_tensor,
                                NVTEGroupedTensorParam::kNVTEGroupedRowwiseData, &out_data_tensor,
                                sizeof(out_data_tensor));

  // Set shape arrays on output too
  if ((shape_rep == VARYING_FIRST_DIM) || (shape_rep == VARYING_BOTH_DIMS)) {
    NVTEBasicTensor first_dims_tensor = {first_dims_d, kNVTEInt64, first_dims_shape};
    nvte_set_grouped_tensor_param(out_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedFirstDims,
                                  &first_dims_tensor, sizeof(first_dims_tensor));
  }
  if ((shape_rep == VARYING_LAST_DIM) || (shape_rep == VARYING_BOTH_DIMS)) {
    NVTEBasicTensor last_dims_tensor = {last_dims_d, kNVTEInt64, last_dims_shape};
    nvte_set_grouped_tensor_param(out_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedLastDims, &last_dims_tensor,
                                  sizeof(last_dims_tensor));
  }
  if (shape_rep != SAME_BOTH_DIMS) {
    NVTEBasicTensor offsets_tensor = {offsets_d, kNVTEInt64, offsets_shape};
    nvte_set_grouped_tensor_param(out_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets,
                                  &offsets_tensor, sizeof(offsets_tensor));
  }

  // Run grouped dequantize
  nvte_group_dequantize(in_group_tensor, out_group_tensor, 0);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // Copy grouped output to host
  std::vector<OutputType> out_grouped_h(elts_num);
  cudaMemcpy(out_grouped_h.data(), out_grouped_d, out_data_size, cudaMemcpyDeviceToHost);

  // Now compute reference: run single-tensor nvte_dequantize for each tensor
  std::vector<OutputType> out_ref_h(elts_num);

  for (size_t t = 0; t < num_tensors; ++t) {
    const size_t M = first_dims_h[t];
    const size_t K = last_dims_h[t];
    const size_t data_offset = offsets_h[t];
    const size_t scales_offset = per_tensor_scales_offset[t];
    const size_t tensor_scales_count =
        per_tensor_scales_first_dim[t] * per_tensor_scales_last_dim[t];

    const size_t single_data_size = M * K * sizeof(InputType);
    const size_t single_out_size = M * K * sizeof(OutputType);
    const size_t single_scales_size = tensor_scales_count * sizeof(fp8e8m0);

    // Allocate per-tensor device memory
    InputType *single_in_d;
    OutputType *single_out_d;
    fp8e8m0 *single_scales_d;

    cudaMalloc((void **)&single_in_d, single_data_size);
    cudaMalloc((void **)&single_out_d, single_out_size);
    cudaMalloc((void **)&single_scales_d, single_scales_size);

    cudaMemcpy(single_in_d, in_data_h.data() + data_offset, single_data_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(single_scales_d, in_scales_h.data() + scales_offset, single_scales_size,
               cudaMemcpyHostToDevice);
    cudaMemset(single_out_d, 0, single_out_size);

    // Build single-tensor NVTETensor using TensorWrapper directly
    std::vector<size_t> single_shape = {M, K};
    std::vector<size_t> scale_shape_vec = {per_tensor_scales_first_dim[t],
                                           per_tensor_scales_last_dim[t]};

    TensorWrapper input_w(NVTE_MXFP8_1D_SCALING);
    if (rowwise) {
      input_w.set_rowwise_data(single_in_d, itype, single_shape);
      input_w.set_rowwise_scale_inv(single_scales_d, DType::kFloat8E8M0, scale_shape_vec);
    } else {
      input_w.set_columnwise_data(single_in_d, itype, single_shape);
      input_w.set_columnwise_scale_inv(single_scales_d, DType::kFloat8E8M0, scale_shape_vec);
    }

    TensorWrapper output_w;
    output_w.set_rowwise_data(single_out_d, otype, single_shape);

    nvte_dequantize(input_w.data(), output_w.data(), 0);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Single-tensor dequantize failed for tensor " << t << ": "
                                << cudaGetErrorString(err);

    // Copy reference output to host
    cudaMemcpy(out_ref_h.data() + data_offset, single_out_d, single_out_size,
               cudaMemcpyDeviceToHost);

    cudaFree(single_in_d);
    cudaFree(single_out_d);
    cudaFree(single_scales_d);
  }

  // Bitwise comparison
  for (size_t t = 0; t < num_tensors; ++t) {
    const size_t M = first_dims_h[t];
    const size_t K = last_dims_h[t];
    const size_t data_offset = offsets_h[t];
    const size_t tensor_elts = M * K;

    int result = memcmp(out_grouped_h.data() + data_offset, out_ref_h.data() + data_offset,
                        tensor_elts * sizeof(OutputType));
    if (result != 0) {
      // Find first mismatch for error reporting
      for (size_t i = 0; i < tensor_elts; ++i) {
        if (out_grouped_h[data_offset + i] != out_ref_h[data_offset + i]) {
          GTEST_FAIL() << "Bitwise mismatch at tensor " << t << " element " << i
                       << " (global offset " << (data_offset + i) << "): grouped="
                       << static_cast<float>(out_grouped_h[data_offset + i])
                       << " vs reference=" << static_cast<float>(out_ref_h[data_offset + i]);
        }
      }
    }
  }

  // Cleanup
  cudaFree(in_data_d);
  cudaFree(out_grouped_d);
  cudaFree(in_scales_d);
  cudaFree(first_dims_d);
  cudaFree(last_dims_d);
  cudaFree(offsets_d);
}

// {shape_representation, num_tensors, [logical_shape_M, logical_shape_K], [M_i], [K_i]}
std::vector<std::vector<size_t>> input_configs = {
    {SAME_BOTH_DIMS, 1, 128, 128},
    {SAME_BOTH_DIMS, 2, 256, 128},
    {VARYING_FIRST_DIM, 2, 512, 128, 128, 384},
    {VARYING_FIRST_DIM, 2, 384, 128, 128, 256},
    {VARYING_FIRST_DIM, 5, 4096, 512, 128, 256, 384, 1024, 2304},
    {VARYING_LAST_DIM, 3, 256, 896, 128, 256, 512},
    {VARYING_BOTH_DIMS, 2, 1, (128 * 128) + (256 * 256), 128, 256, 128, 256},
    {VARYING_BOTH_DIMS, 2, 1, (256 * 128) + (512 * 640), 256, 512, 128, 640},
    // Non-128-aligned constant dimensions
    {SAME_BOTH_DIMS, 1, 160, 192},
    {SAME_BOTH_DIMS, 2, 256, 96},
    {VARYING_FIRST_DIM, 2, 384, 160, 128, 256},
    {VARYING_FIRST_DIM, 3, 768, 96, 256, 256, 256},
    {VARYING_LAST_DIM, 2, 160, 384, 128, 256},
    {VARYING_LAST_DIM, 3, 96, 512, 128, 128, 256},
};

std::vector<ScalingDirection> scaling_directions = {
    ScalingDirection::ROWWISE,
    ScalingDirection::COLWISE,
};

}  // namespace

class GroupedDequantizeMXFP8TestSuite
    : public ::testing::TestWithParam<std::tuple<ScalingDirection,
                                                 std::vector<size_t>,        // Config
                                                 transformer_engine::DType,  // InputType
                                                 transformer_engine::DType   // OutputType
                                                 >> {};

TEST_P(GroupedDequantizeMXFP8TestSuite, TestGroupedDequantizeMXFP8) {
  // Skip tests for pre-Blackwell architectures
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }

  using namespace transformer_engine;
  using namespace test;

  const ScalingDirection scaling_direction = std::get<0>(GetParam());
  const std::vector<size_t> config = std::get<1>(GetParam());
  const DType input_type = std::get<2>(GetParam());
  const DType output_type = std::get<3>(GetParam());

  const ShapeRepresentation shape_rep = static_cast<ShapeRepresentation>(config[0]);
  const size_t num_tensors = config[1];
  const std::vector<size_t> logical_shape = {config[2], config[3]};

  const bool rowwise = (scaling_direction == ScalingDirection::ROWWISE);

  std::vector<size_t> first_dims(num_tensors);
  std::vector<size_t> last_dims(num_tensors);
  std::vector<size_t> offsets(num_tensors + 1, 0);

  for (size_t t = 0; t < num_tensors; ++t) {
    switch (shape_rep) {
      case SAME_BOTH_DIMS: {
        first_dims[t] = logical_shape[0] / num_tensors;
        last_dims[t] = logical_shape[1];
        break;
      }
      case VARYING_FIRST_DIM: {
        first_dims[t] = config[t + 4];
        last_dims[t] = logical_shape[1];
        break;
      }
      case VARYING_LAST_DIM: {
        first_dims[t] = logical_shape[0];
        last_dims[t] = config[t + 4];
        break;
      }
      case VARYING_BOTH_DIMS: {
        first_dims[t] = config[t + 4];
        last_dims[t] = config[t + (4 + num_tensors)];
        break;
      }
    }
    offsets[t + 1] = offsets[t] + first_dims[t] * last_dims[t];

    // Skip tests if varying dimensions are not 128-aligned
    const bool first_dim_varies =
        (shape_rep == VARYING_FIRST_DIM || shape_rep == VARYING_BOTH_DIMS);
    const bool last_dim_varies =
        (shape_rep == VARYING_LAST_DIM || shape_rep == VARYING_BOTH_DIMS);
    if (first_dim_varies && (first_dims[t] % 128 != 0)) {
      GTEST_SKIP();
    }
    if (last_dim_varies && (last_dims[t] % 128 != 0)) {
      GTEST_SKIP();
    }
    // TMA requires last_dim * sizeof(FP8) to be 16-byte aligned
    if (last_dims[t] % 16 != 0) {
      GTEST_SKIP();
    }
    // For colwise: first dim must be divisible by 32
    if (!rowwise && (first_dims[t] % 32 != 0)) {
      GTEST_SKIP();
    }
    // For rowwise: last dim must be divisible by 32
    if (rowwise && (last_dims[t] % 32 != 0)) {
      GTEST_SKIP();
    }
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
          output_type, OutputType,
          performTest<InputType, OutputType>(shape_rep, num_tensors, logical_shape, first_dims,
                                            last_dims, offsets, rowwise);););
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, GroupedDequantizeMXFP8TestSuite,
    ::testing::Combine(::testing::ValuesIn(scaling_directions), ::testing::ValuesIn(input_configs),
                       ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
                       ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16)),
    [](const testing::TestParamInfo<GroupedDequantizeMXFP8TestSuite::ParamType> &info) {
      std::string name;
      switch (std::get<0>(info.param)) {
        case ScalingDirection::ROWWISE:
          name += "ROWWISE_";
          break;
        case ScalingDirection::COLWISE:
          name += "COLWISE_";
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
        case ShapeRepresentation::VARYING_LAST_DIM:
          name += "VARYING_LAST_DIM";
          break;
        case ShapeRepresentation::VARYING_BOTH_DIMS:
          name += "VARYING_BOTH_DIMS";
          break;
      }

      name += "_N_" + std::to_string(input[1]);
      name += "_SHAPE_" + std::to_string(input[2]) + "X" + std::to_string(input[3]);
      name += "_" + test::typeName(std::get<2>(info.param));
      name += "_" + test::typeName(std::get<3>(info.param));
      return name;
    });
