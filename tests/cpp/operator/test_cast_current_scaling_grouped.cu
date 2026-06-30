/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/recipe.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum ShapeRepresentation {
  SAME_BOTH_DIMS = 0,
  VARYING_FIRST_DIM = 1,
  VARYING_LAST_DIM = 2,
};

enum ScalingDirection { ROWWISE = 0, COLWISE = 1, BOTH = 2 };

// Reference per-group amax / scale / scale_inv for FP8 current (tensor) scaling.
// Mirrors compute_scale_from_amax on device (epsilon floor, __fdiv_rn, FLT_MAX
// for non-representable scales) with force_pow_2_scales disabled.
template <typename InputType>
void compute_amax_scale_ref(const InputType *data, const size_t size, float *amax_ptr,
                            float *scale_ptr, float *scale_inv_ptr, const float max_fp8,
                            const float epsilon) {
  float current_max = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    const float current = static_cast<float>(data[i]);
    current_max = fmaxf(current_max, fabsf(current));
  }
  *amax_ptr = current_max;

  float clamp_amax = current_max;
  if (clamp_amax < epsilon) {
    clamp_amax = epsilon;
  }

  float scale = 1.0f;
  float scale_inv = 1.0f;
  if (std::isinf(clamp_amax) || clamp_amax == 0.0f || std::isnan(clamp_amax)) {
    *scale_ptr = scale;
    *scale_inv_ptr = scale_inv;
    return;
  }

  scale = max_fp8 / clamp_amax;
  if (std::isinf(scale)) {
    scale = std::numeric_limits<float>::max();
  }
  if (std::isnan(scale)) {
    scale = 1.0f;
  }
  scale_inv = 1.0f / scale;

  *scale_ptr = scale;
  *scale_inv_ptr = scale_inv;
}

// Element-wise comparison that tolerates round-to-nearest ambiguity (a quantized
// value that lands exactly between two representable FP8 values may round either
// way depending on tiny scale differences between host and device).
template <typename T>
void compare_scaled_elts(const std::string &name, const T *ref_data, const T *test_data,
                         const size_t numel, const size_t tolerable_mismatches_limit = 0) {
  size_t mismatches_num = 0;
  int64_t first_mismatch_idx = -1;
  for (size_t i = 0; i < numel; ++i) {
    const double t = static_cast<double>(test_data[i]);
    const double r = static_cast<double>(ref_data[i]);
    if (t == r) {
      continue;
    }
    // Allow a single-ULP round-to-nearest disagreement.
    const double mean = (t + r) / 2;
    const double mean_p = mean >= 0 ? mean * (1 + 1e-6) : mean * (1 - 1e-6);
    const double mean_m = mean >= 0 ? mean * (1 - 1e-6) : mean * (1 + 1e-6);
    const double cast_mean_p = static_cast<double>(static_cast<T>(mean_p));
    const double cast_mean_m = static_cast<double>(static_cast<T>(mean_m));
    const bool round_ambiguity =
        (cast_mean_m == std::min(t, r) && cast_mean_p == std::max(t, r));
    if (round_ambiguity) {
      continue;
    }
    mismatches_num++;
    if (first_mismatch_idx == -1) {
      first_mismatch_idx = static_cast<int64_t>(i);
    }
    if (mismatches_num > tolerable_mismatches_limit) {
      GTEST_FAIL() << mismatches_num << " mismatch(es) in tensor " << name
                   << " (limit " << tolerable_mismatches_limit << "). First at "
                   << first_mismatch_idx << ": "
                   << static_cast<double>(test_data[first_mismatch_idx]) << " vs "
                   << static_cast<double>(ref_data[first_mismatch_idx]);
    }
  }
}

template <typename InputType, typename OutputType>
void performTest(const ShapeRepresentation shape_rep, const size_t num_tensors,
                 const std::vector<size_t> &logical_shape_vec,
                 const std::vector<size_t> &first_dims_h, const std::vector<size_t> &last_dims_h,
                 const std::vector<size_t> &offsets_h, const bool rowwise, const bool colwise) {
  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  const size_t cols = logical_shape_vec[1];

  float max_fp8 = Quantized_Limits<OutputType>::max();

  size_t elts_num = 0;
  for (size_t t = 0; t < num_tensors; ++t) {
    elts_num += first_dims_h[t] * last_dims_h[t];
  }

  // Host inputs
  std::vector<InputType> in_data(elts_num);
  std::mt19937 gen(12345);
  std::uniform_real_distribution<> dis(-2.0, 1.0);
  for (size_t i = 0; i < elts_num; ++i) {
    in_data[i] = static_cast<InputType>(dis(gen));
  }

  // Reference amax / scale / scale_inv (one per group) and output data.
  std::vector<float> ref_amax(num_tensors, 0.0f);
  std::vector<float> ref_scale(num_tensors, 1.0f);
  std::vector<float> ref_scale_inv(num_tensors, 1.0f);
  std::vector<OutputType> ref_out_rowwise(rowwise ? elts_num : 0, static_cast<OutputType>(0.0f));
  std::vector<OutputType> ref_out_colwise(colwise ? elts_num : 0, static_cast<OutputType>(0.0f));

  for (size_t t = 0; t < num_tensors; ++t) {
    const size_t M = first_dims_h[t];
    const size_t K = last_dims_h[t];
    const size_t base = offsets_h[t];
    const size_t group_elts = M * K;

    compute_amax_scale_ref<InputType>(in_data.data() + base, group_elts, &ref_amax[t],
                                      &ref_scale[t], &ref_scale_inv[t], max_fp8, /*epsilon=*/0.0f);

    const float scale = ref_scale[t];
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < K; ++j) {
        const float val = static_cast<float>(in_data[base + i * K + j]) * scale;
        if (rowwise) {
          ref_out_rowwise[base + i * K + j] = static_cast<OutputType>(val);
        }
        if (colwise) {
          // Columnwise output of one group is stored transposed: [col * M + row].
          ref_out_colwise[base + j * M + i] = static_cast<OutputType>(val);
        }
      }
    }
  }

  // Device allocations
  const size_t in_data_size = elts_num * sizeof(InputType);
  const size_t out_data_size = elts_num * sizeof(OutputType);
  const size_t per_group_f32_size = num_tensors * sizeof(float);
  const size_t int64_arr_size = num_tensors * sizeof(int64_t);
  const size_t offsets_size = (num_tensors + 1) * sizeof(int64_t);

  // first_dims/last_dims/offsets must be int64 on device.
  std::vector<int64_t> first_dims_i64(first_dims_h.begin(), first_dims_h.end());
  std::vector<int64_t> last_dims_i64(last_dims_h.begin(), last_dims_h.end());
  std::vector<int64_t> offsets_i64(offsets_h.begin(), offsets_h.end());

  InputType *in_data_d = nullptr;
  float *amax_d = nullptr;
  float *scale_d = nullptr;
  float *scale_inv_d = nullptr;
  OutputType *out_rowwise_d = nullptr;
  OutputType *out_colwise_d = nullptr;
  int64_t *first_dims_d = nullptr;
  int64_t *last_dims_d = nullptr;
  int64_t *offsets_d = nullptr;

  NVTE_CHECK_CUDA(cudaMalloc((void **)&in_data_d, in_data_size));
  NVTE_CHECK_CUDA(cudaMalloc((void **)&amax_d, per_group_f32_size));
  NVTE_CHECK_CUDA(cudaMalloc((void **)&scale_d, per_group_f32_size));
  NVTE_CHECK_CUDA(cudaMalloc((void **)&scale_inv_d, per_group_f32_size));
  NVTE_CHECK_CUDA(cudaMalloc((void **)&first_dims_d, int64_arr_size));
  NVTE_CHECK_CUDA(cudaMalloc((void **)&last_dims_d, int64_arr_size));
  NVTE_CHECK_CUDA(cudaMalloc((void **)&offsets_d, offsets_size));

  NVTE_CHECK_CUDA(cudaMemcpy(in_data_d, in_data.data(), in_data_size, cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemset(amax_d, 0, per_group_f32_size));
  NVTE_CHECK_CUDA(cudaMemset(scale_d, 0, per_group_f32_size));
  NVTE_CHECK_CUDA(cudaMemset(scale_inv_d, 0, per_group_f32_size));
  NVTE_CHECK_CUDA(
      cudaMemcpy(first_dims_d, first_dims_i64.data(), int64_arr_size, cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(
      cudaMemcpy(last_dims_d, last_dims_i64.data(), int64_arr_size, cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(offsets_d, offsets_i64.data(), offsets_size, cudaMemcpyHostToDevice));

  if (rowwise) {
    NVTE_CHECK_CUDA(cudaMalloc((void **)&out_rowwise_d, out_data_size));
    NVTE_CHECK_CUDA(cudaMemset(out_rowwise_d, 0, out_data_size));
  }
  if (colwise) {
    NVTE_CHECK_CUDA(cudaMalloc((void **)&out_colwise_d, out_data_size));
    NVTE_CHECK_CUDA(cudaMemset(out_colwise_d, 0, out_data_size));
  }

  // Shapes
  NVTEShape logical_shape = nvte_make_shape(logical_shape_vec.data(), logical_shape_vec.size());
  std::vector<size_t> data_1d_shape = {elts_num};
  NVTEShape data_shape = nvte_make_shape(data_1d_shape.data(), data_1d_shape.size());
  std::vector<size_t> per_group_shape_vec = {num_tensors};
  NVTEShape per_group_shape = nvte_make_shape(per_group_shape_vec.data(), per_group_shape_vec.size());

  NVTEShape first_dims_shape;
  NVTEShape last_dims_shape;
  NVTEShape offsets_shape;
  first_dims_shape.ndim = 1;
  last_dims_shape.ndim = 1;
  offsets_shape.ndim = 1;
  first_dims_shape.data[0] = num_tensors;
  last_dims_shape.data[0] = num_tensors;
  offsets_shape.data[0] = num_tensors + 1;

  // Input grouped tensor (high precision)
  NVTEGroupedTensor in_group_tensor =
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape);
  NVTEBasicTensor in_data_tensor = {in_data_d, static_cast<NVTEDType>(itype), data_shape};
  nvte_set_grouped_tensor_param(in_group_tensor, kNVTEGroupedRowwiseData, &in_data_tensor,
                                sizeof(in_data_tensor));

  // Output grouped tensor (FP8 tensor scaling)
  NVTEGroupedTensor out_group_tensor =
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape);

  NVTEBasicTensor amax_tensor = {amax_d, kNVTEFloat32, per_group_shape};
  nvte_set_grouped_tensor_param(out_group_tensor, kNVTEGroupedAmax, &amax_tensor,
                                sizeof(amax_tensor));
  NVTEBasicTensor scale_tensor = {scale_d, kNVTEFloat32, per_group_shape};
  nvte_set_grouped_tensor_param(out_group_tensor, kNVTEGroupedScale, &scale_tensor,
                                sizeof(scale_tensor));

  if (rowwise) {
    NVTEBasicTensor out_data_tensor = {out_rowwise_d, static_cast<NVTEDType>(otype), data_shape};
    nvte_set_grouped_tensor_param(out_group_tensor, kNVTEGroupedRowwiseData, &out_data_tensor,
                                  sizeof(out_data_tensor));
    NVTEBasicTensor scale_inv_tensor = {scale_inv_d, kNVTEFloat32, per_group_shape};
    nvte_set_grouped_tensor_param(out_group_tensor, kNVTEGroupedRowwiseScaleInv, &scale_inv_tensor,
                                  sizeof(scale_inv_tensor));
  }
  if (colwise) {
    NVTEBasicTensor out_data_tensor = {out_colwise_d, static_cast<NVTEDType>(otype), data_shape};
    nvte_set_grouped_tensor_param(out_group_tensor, kNVTEGroupedColumnwiseData, &out_data_tensor,
                                  sizeof(out_data_tensor));
    // Current scaling reuses a single scale_inv per group across directions; alias
    // the same buffer (this matches the production quantizer layout).
    NVTEBasicTensor scale_inv_tensor = {scale_inv_d, kNVTEFloat32, per_group_shape};
    nvte_set_grouped_tensor_param(out_group_tensor, kNVTEGroupedColumnwiseScaleInv,
                                  &scale_inv_tensor, sizeof(scale_inv_tensor));
  }

  // Shape metadata on both grouped tensors.
  auto set_shape_metadata = [&](NVTEGroupedTensor gt) {
    if (shape_rep == VARYING_FIRST_DIM) {
      NVTEBasicTensor t = {first_dims_d, kNVTEInt64, first_dims_shape};
      nvte_set_grouped_tensor_param(gt, kNVTEGroupedFirstDims, &t, sizeof(t));
    }
    if (shape_rep == VARYING_LAST_DIM) {
      NVTEBasicTensor t = {last_dims_d, kNVTEInt64, last_dims_shape};
      nvte_set_grouped_tensor_param(gt, kNVTEGroupedLastDims, &t, sizeof(t));
    }
    if (shape_rep != SAME_BOTH_DIMS) {
      NVTEBasicTensor t = {offsets_d, kNVTEInt64, offsets_shape};
      nvte_set_grouped_tensor_param(gt, kNVTEGroupedTensorOffsets, &t, sizeof(t));
    }
  };
  set_shape_metadata(in_group_tensor);
  set_shape_metadata(out_group_tensor);

  // FP8 current-scaling grouped quantize:
  //   1) per-group amax, 2) per-group scale/scale_inv, 3) cast/transpose.
  QuantizationConfigWrapper quant_config;
  nvte_group_compute_amax_with_config(in_group_tensor, out_group_tensor, quant_config, 0);
  nvte_group_compute_scale_from_amax(out_group_tensor, quant_config, 0);
  nvte_group_quantize(in_group_tensor, out_group_tensor, quant_config, 0);

  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // Copy results back.
  std::vector<float> amax_h(num_tensors);
  std::vector<float> scale_h(num_tensors);
  std::vector<float> scale_inv_h(num_tensors);
  NVTE_CHECK_CUDA(cudaMemcpy(amax_h.data(), amax_d, per_group_f32_size, cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(scale_h.data(), scale_d, per_group_f32_size, cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(
      cudaMemcpy(scale_inv_h.data(), scale_inv_d, per_group_f32_size, cudaMemcpyDeviceToHost));

  auto [atol_f32, rtol_f32] = getTolerances(DType::kFloat32);
  for (size_t t = 0; t < num_tensors; ++t) {
    EXPECT_NEAR(amax_h[t], ref_amax[t], atol_f32 + rtol_f32 * std::fabs(ref_amax[t]))
        << "amax mismatch at group " << t;
    EXPECT_NEAR(scale_h[t], ref_scale[t], atol_f32 + rtol_f32 * std::fabs(ref_scale[t]))
        << "scale mismatch at group " << t;
    EXPECT_NEAR(scale_inv_h[t], ref_scale_inv[t], atol_f32 + rtol_f32 * std::fabs(ref_scale_inv[t]))
        << "scale_inv mismatch at group " << t;
  }

  if (rowwise) {
    std::vector<OutputType> out_h(elts_num);
    NVTE_CHECK_CUDA(
        cudaMemcpy(out_h.data(), out_rowwise_d, out_data_size, cudaMemcpyDeviceToHost));
    compare_scaled_elts<OutputType>("rowwise_output", ref_out_rowwise.data(), out_h.data(),
                                    elts_num);
  }
  if (colwise) {
    std::vector<OutputType> out_h(elts_num);
    NVTE_CHECK_CUDA(
        cudaMemcpy(out_h.data(), out_colwise_d, out_data_size, cudaMemcpyDeviceToHost));
    compare_scaled_elts<OutputType>("colwise_output", ref_out_colwise.data(), out_h.data(),
                                    elts_num);
  }

  nvte_destroy_grouped_tensor(in_group_tensor);
  nvte_destroy_grouped_tensor(out_group_tensor);
  cudaFree(in_data_d);
  cudaFree(amax_d);
  cudaFree(scale_d);
  cudaFree(scale_inv_d);
  cudaFree(first_dims_d);
  cudaFree(last_dims_d);
  cudaFree(offsets_d);
  if (rowwise) cudaFree(out_rowwise_d);
  if (colwise) cudaFree(out_colwise_d);
}

// {shape_representation, num_tensors, [logical_shape_M, logical_shape_K], [M_i] or [K_i]}.
// Constant dimensions must be a small-vector multiple of
// (16 for bf16/fp16, 8 for fp32), so the configs below intentionally use sizes
// that are multiples of 16.
std::vector<std::vector<size_t>> input_configs = {
    {SAME_BOTH_DIMS, 1, 128, 128},
    {SAME_BOTH_DIMS, 2, 256, 144},
    {SAME_BOTH_DIMS, 3, 384, 80},
    {VARYING_FIRST_DIM, 2, 512, 128, 128, 384},
    {VARYING_FIRST_DIM, 3, 1024, 144, 128, 384, 512},
    {VARYING_FIRST_DIM, 4, 1536, 160, 128, 384, 512, 512},
    {VARYING_FIRST_DIM, 3, 800, 96, 100, 300, 400},
    // Empty tensor in the middle must not break the per-group loop.
    {VARYING_FIRST_DIM, 4, 512, 160, 128, 0, 128, 256},
    {VARYING_LAST_DIM, 3, 256, 896, 128, 256, 512},
    {VARYING_LAST_DIM, 2, 160, 384, 128, 256},
    {VARYING_LAST_DIM, 3, 80, 512, 128, 128, 256},
};

std::vector<ScalingDirection> scaling_directions = {
    ScalingDirection::ROWWISE,
    ScalingDirection::COLWISE,
    ScalingDirection::BOTH,
};

}  // namespace

class GroupedCastCurrentScalingTestSuite
    : public ::testing::TestWithParam<std::tuple<ScalingDirection,
                                                 std::vector<size_t>,        // Config
                                                 transformer_engine::DType,  // InputType
                                                 transformer_engine::DType   // OutputType
                                                 >> {};

TEST_P(GroupedCastCurrentScalingTestSuite, Test) {
  if (getDeviceComputeCapability() < hopperComputeCapability) {
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

  std::vector<size_t> first_dims(num_tensors);
  std::vector<size_t> last_dims(num_tensors);
  std::vector<size_t> offsets(num_tensors + 1, 0);
  for (size_t t = 0; t < num_tensors; ++t) {
    switch (shape_rep) {
      case SAME_BOTH_DIMS:
        first_dims[t] = logical_shape[0] / num_tensors;
        last_dims[t] = logical_shape[1];
        break;
      case VARYING_FIRST_DIM:
        first_dims[t] = config[t + 4];
        last_dims[t] = logical_shape[1];
        break;
      case VARYING_LAST_DIM:
        first_dims[t] = logical_shape[0];
        last_dims[t] = config[t + 4];
        break;
    }
    offsets[t + 1] = offsets[t] + first_dims[t] * last_dims[t];
  }

  bool rowwise = false;
  bool colwise = false;
  switch (scaling_direction) {
    case ScalingDirection::ROWWISE: rowwise = true; break;
    case ScalingDirection::COLWISE: colwise = true; break;
    case ScalingDirection::BOTH: rowwise = true; colwise = true; break;
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(
          output_type, OutputType,
          performTest<InputType, OutputType>(shape_rep, num_tensors, logical_shape, first_dims,
                                             last_dims, offsets, rowwise, colwise);););
}

std::string MakeGroupedCastCurrentScalingTestName(
    const testing::TestParamInfo<GroupedCastCurrentScalingTestSuite::ParamType> &info) {
  std::string name;
  switch (std::get<0>(info.param)) {
    case ScalingDirection::ROWWISE: name += "ROWWISE_"; break;
    case ScalingDirection::COLWISE: name += "COLWISE_"; break;
    case ScalingDirection::BOTH: name += "BIDIMENSIONAL_"; break;
  }

  const std::vector<size_t> input = std::get<1>(info.param);
  switch (static_cast<ShapeRepresentation>(input[0])) {
    case ShapeRepresentation::SAME_BOTH_DIMS: name += "SAME_BOTH_DIMS"; break;
    case ShapeRepresentation::VARYING_FIRST_DIM: name += "VARYING_FIRST_DIM"; break;
    case ShapeRepresentation::VARYING_LAST_DIM: name += "VARYING_LAST_DIM"; break;
  }
  name += "_N_" + std::to_string(input[1]);
  name += "_SHAPE_" + std::to_string(input[2]) + "X" + std::to_string(input[3]);
  name += "_" + test::typeName(std::get<2>(info.param));
  name += "_" + test::typeName(std::get<3>(info.param));
  return name;
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, GroupedCastCurrentScalingTestSuite,
    ::testing::Combine(::testing::ValuesIn(scaling_directions), ::testing::ValuesIn(input_configs),
                       ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
                       ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2)),
    MakeGroupedCastCurrentScalingTestName);
