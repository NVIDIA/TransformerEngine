/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/swizzle.h>
#include <transformer_engine/transformer_engine.h>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

constexpr size_t kMXFP8ScaleDim = 32;

struct GroupShapeCase {
  std::string name;
  std::vector<std::pair<size_t, size_t>> shapes;
};

struct GroupShapeInfo {
  size_t num_tensors = 0;
  size_t total_elements = 0;
  size_t rowwise_scale_elements = 0;
  size_t columnwise_scale_elements = 0;
  bool same_first_dim = false;
  bool same_last_dim = false;
  std::vector<size_t> logical_shape;
  std::vector<int64_t> first_dims;
  std::vector<int64_t> last_dims;
  std::vector<int64_t> offsets;
};

size_t round_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

GroupShapeInfo make_shape_info(const GroupShapeCase &test_case) {
  GroupShapeInfo info;
  info.num_tensors = test_case.shapes.size();
  info.first_dims.reserve(info.num_tensors);
  info.last_dims.reserve(info.num_tensors);
  info.offsets.reserve(info.num_tensors + 1);
  info.offsets.push_back(0);

  for (const auto &[rows, cols] : test_case.shapes) {
    info.first_dims.push_back(static_cast<int64_t>(rows));
    info.last_dims.push_back(static_cast<int64_t>(cols));
    info.total_elements += rows * cols;
    info.offsets.push_back(static_cast<int64_t>(info.total_elements));
  }

  info.same_first_dim = true;
  info.same_last_dim = true;
  for (size_t i = 1; i < info.num_tensors; ++i) {
    info.same_first_dim &= info.first_dims[i] == info.first_dims[0];
    info.same_last_dim &= info.last_dims[i] == info.last_dims[0];
  }

  if (info.same_first_dim && info.same_last_dim) {
    info.logical_shape = {
        static_cast<size_t>(info.first_dims[0]) * info.num_tensors,
        static_cast<size_t>(info.last_dims[0]),
    };
  } else if (!info.same_first_dim && info.same_last_dim) {
    size_t total_rows = 0;
    for (const int64_t rows : info.first_dims) {
      total_rows += static_cast<size_t>(rows);
    }
    info.logical_shape = {total_rows, static_cast<size_t>(info.last_dims[0])};
  } else if (info.same_first_dim && !info.same_last_dim) {
    size_t total_cols = 0;
    for (const int64_t cols : info.last_dims) {
      total_cols += static_cast<size_t>(cols);
    }
    info.logical_shape = {static_cast<size_t>(info.first_dims[0]), total_cols};
  } else {
    info.logical_shape = {1, info.total_elements};
  }

  // SAME_BOTH_DIMS and VARYING_FIRST_DIM are represented as one tall tensor by
  // the grouped MXFP8 kernels. Their compact scales therefore have padding only
  // at the end of the whole group, not between member tensors.
  if (info.same_last_dim) {
    const size_t total_rows = info.logical_shape[0];
    const size_t cols = info.logical_shape[1];
    const size_t rowwise_stride = round_up(
        (cols + kMXFP8ScaleDim - 1) / kMXFP8ScaleDim,
        scale_tensor_alignment_X_rowwise);
    const size_t columnwise_stride = round_up(cols, scale_tensor_alignment_X_colwise);
    info.rowwise_scale_elements =
        round_up(total_rows, scale_tensor_alignment_Y_rowwise) * rowwise_stride;
    info.columnwise_scale_elements =
        round_up((total_rows + kMXFP8ScaleDim - 1) / kMXFP8ScaleDim,
                 scale_tensor_alignment_Y_colwise) *
        columnwise_stride;
  } else {
    // For varying last dimensions, each member has its own descriptor and its
    // own compact scale range. The test shapes obey the grouped-kernel contract
    // (both member dimensions are multiples of 128), so compact and padded
    // capacities are identical.
    for (const auto &[rows, cols] : test_case.shapes) {
      const size_t rowwise_rows = round_up(rows, scale_tensor_alignment_Y_rowwise);
      const size_t rowwise_cols = round_up(
          (cols + kMXFP8ScaleDim - 1) / kMXFP8ScaleDim,
          scale_tensor_alignment_X_rowwise);
      const size_t columnwise_rows = round_up(
          (rows + kMXFP8ScaleDim - 1) / kMXFP8ScaleDim,
          scale_tensor_alignment_Y_colwise);
      const size_t columnwise_cols = round_up(cols, scale_tensor_alignment_X_colwise);
      info.rowwise_scale_elements += rowwise_rows * rowwise_cols;
      info.columnwise_scale_elements += columnwise_rows * columnwise_cols;
    }
  }

  return info;
}

size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return sizeof(float);
    case DType::kBFloat16:
      return sizeof(bf16);
    case DType::kFloat8E4M3:
      return sizeof(fp8e4m3);
    case DType::kFloat8E5M2:
      return sizeof(fp8e5m2);
    default:
      NVTE_ERROR("Unsupported dtype in grouped MXFP8 requantize test: ",
                 static_cast<int>(dtype));
  }
  return 0;
}

struct OwnedGroupedTensor {
  std::unique_ptr<GroupedTensorWrapper> tensor;
  test::CudaPtr<> rowwise_data;
  test::CudaPtr<> columnwise_data;
  test::CudaPtr<> rowwise_scale_inv;
  test::CudaPtr<> columnwise_scale_inv;
  test::CudaPtr<int64_t> first_dims;
  test::CudaPtr<int64_t> last_dims;
  test::CudaPtr<int64_t> offsets;
  size_t data_bytes = 0;
  size_t rowwise_scale_bytes = 0;
  size_t columnwise_scale_bytes = 0;

  NVTEGroupedTensor data() const { return tensor->data(); }
};

OwnedGroupedTensor make_grouped_tensor(const GroupShapeInfo &shape_info, DType dtype,
                                       NVTEScalingMode scaling_mode, bool rowwise,
                                       bool columnwise, bool swizzled = false) {
  OwnedGroupedTensor result;
  result.tensor = std::make_unique<GroupedTensorWrapper>(
      shape_info.num_tensors, shape_info.logical_shape, scaling_mode);

  const std::vector<size_t> flat_data_shape = {shape_info.total_elements};
  result.data_bytes = shape_info.total_elements * dtype_size(dtype);
  if (rowwise) {
    result.rowwise_data = test::cuda_alloc(result.data_bytes);
    result.tensor->set_rowwise_data(result.rowwise_data.get(), dtype, flat_data_shape);
  }
  if (columnwise) {
    result.columnwise_data = test::cuda_alloc(result.data_bytes);
    result.tensor->set_columnwise_data(result.columnwise_data.get(), dtype, flat_data_shape);
  }

  if (scaling_mode == NVTE_MXFP8_1D_SCALING) {
    if (rowwise) {
      result.rowwise_scale_bytes = shape_info.rowwise_scale_elements;
      result.rowwise_scale_inv = test::cuda_alloc(result.rowwise_scale_bytes);
      result.tensor->set_rowwise_scale_inv(
          result.rowwise_scale_inv.get(), DType::kFloat8E8M0,
          std::vector<size_t>{shape_info.rowwise_scale_elements});
    }
    if (columnwise) {
      result.columnwise_scale_bytes = shape_info.columnwise_scale_elements;
      result.columnwise_scale_inv = test::cuda_alloc(result.columnwise_scale_bytes);
      result.tensor->set_columnwise_scale_inv(
          result.columnwise_scale_inv.get(), DType::kFloat8E8M0,
          std::vector<size_t>{shape_info.columnwise_scale_elements});
    }
    result.tensor->set_with_gemm_swizzled_scales(swizzled);
  }

  const std::vector<size_t> dims_shape = {shape_info.num_tensors};
  if (!shape_info.same_first_dim) {
    const size_t bytes = shape_info.num_tensors * sizeof(int64_t);
    result.first_dims = test::cuda_alloc<int64_t>(bytes);
    NVTE_CHECK_CUDA(cudaMemcpy(result.first_dims.get(), shape_info.first_dims.data(), bytes,
                               cudaMemcpyHostToDevice));
    result.tensor->set_first_dims(result.first_dims.get(), DType::kInt64, dims_shape);
  }
  if (!shape_info.same_last_dim) {
    const size_t bytes = shape_info.num_tensors * sizeof(int64_t);
    result.last_dims = test::cuda_alloc<int64_t>(bytes);
    NVTE_CHECK_CUDA(cudaMemcpy(result.last_dims.get(), shape_info.last_dims.data(), bytes,
                               cudaMemcpyHostToDevice));
    result.tensor->set_last_dims(result.last_dims.get(), DType::kInt64, dims_shape);
  }
  if (!shape_info.same_first_dim || !shape_info.same_last_dim) {
    const size_t bytes = (shape_info.num_tensors + 1) * sizeof(int64_t);
    result.offsets = test::cuda_alloc<int64_t>(bytes);
    NVTE_CHECK_CUDA(cudaMemcpy(result.offsets.get(), shape_info.offsets.data(), bytes,
                               cudaMemcpyHostToDevice));
    result.tensor->set_tensor_offsets(
        result.offsets.get(), DType::kInt64,
        std::vector<size_t>{shape_info.num_tensors + 1});
  }

  return result;
}

void fill_source(OwnedGroupedTensor &source, const GroupShapeInfo &shape_info) {
  std::vector<bf16> host_data(shape_info.total_elements);
  for (size_t i = 0; i < host_data.size(); ++i) {
    const int value = static_cast<int>((i * 17 + i / 13) % 251) - 125;
    host_data[i] = static_cast<bf16>(static_cast<float>(value) / 16.0f);
  }
  NVTE_CHECK_CUDA(cudaMemcpy(source.rowwise_data.get(), host_data.data(), source.data_bytes,
                             cudaMemcpyHostToDevice));
}

void fill_output_sentinel(OwnedGroupedTensor &tensor) {
  if (tensor.rowwise_data != nullptr) {
    NVTE_CHECK_CUDA(cudaMemset(tensor.rowwise_data.get(), 0xA5, tensor.data_bytes));
  }
  if (tensor.columnwise_data != nullptr) {
    NVTE_CHECK_CUDA(cudaMemset(tensor.columnwise_data.get(), 0xA5, tensor.data_bytes));
  }
  if (tensor.rowwise_scale_inv != nullptr) {
    NVTE_CHECK_CUDA(
        cudaMemset(tensor.rowwise_scale_inv.get(), 0xA5, tensor.rowwise_scale_bytes));
  }
  if (tensor.columnwise_scale_inv != nullptr) {
    NVTE_CHECK_CUDA(cudaMemset(tensor.columnwise_scale_inv.get(), 0xA5,
                               tensor.columnwise_scale_bytes));
  }
}

void expect_bytes_equal(const std::string &name, const void *actual_dptr,
                        const void *expected_dptr, size_t bytes) {
  std::vector<uint8_t> actual(bytes);
  std::vector<uint8_t> expected(bytes);
  NVTE_CHECK_CUDA(
      cudaMemcpy(actual.data(), actual_dptr, bytes, cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(
      cudaMemcpy(expected.data(), expected_dptr, bytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < bytes; ++i) {
    ASSERT_EQ(actual[i], expected[i])
        << name << " mismatch at byte " << i << ": actual="
        << static_cast<int>(actual[i]) << ", expected=" << static_cast<int>(expected[i]);
  }
}

using RequantizeParam = std::tuple<GroupShapeCase, DType, bool, bool>;

class GroupedRequantizeMXFP8TestSuite : public ::testing::TestWithParam<RequantizeParam> {};

TEST_P(GroupedRequantizeMXFP8TestSuite, MatchesDequantizeThenQuantize) {
  if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) {
    GTEST_SKIP();
  }

  const auto &test_case = std::get<0>(GetParam());
  const DType input_dtype = std::get<1>(GetParam());
  const bool use_fast_math = std::get<2>(GetParam());
  const bool output_swizzled = std::get<3>(GetParam());
  const DType intermediate_dtype =
      use_fast_math ? DType::kBFloat16 : DType::kFloat32;
  const GroupShapeInfo shape_info = make_shape_info(test_case);

  // Build a production-like wire tensor: high precision -> grouped rowwise
  // MXFP8 with compact scales. This is the only input representation accepted
  // by nvte_group_requantize_mxfp8.
  auto source = make_grouped_tensor(shape_info, DType::kBFloat16,
                                    NVTE_DELAYED_TENSOR_SCALING,
                                    /*rowwise=*/true, /*columnwise=*/false);
  auto input = make_grouped_tensor(shape_info, input_dtype, NVTE_MXFP8_1D_SCALING,
                                   /*rowwise=*/true, /*columnwise=*/false,
                                   /*swizzled=*/false);
  fill_source(source, shape_info);
  fill_output_sentinel(input);
  nvte_group_quantize(source.data(), input.data(), nullptr, 0);

  // Code under test.
  auto actual = make_grouped_tensor(shape_info, DType::kFloat8E4M3,
                                    NVTE_MXFP8_1D_SCALING,
                                    /*rowwise=*/false, /*columnwise=*/true,
                                    output_swizzled);
  fill_output_sentinel(actual);
  QuantizationConfigWrapper quant_config;
  quant_config.set_use_fast_math(use_fast_math);
  nvte_group_requantize_mxfp8(input.data(), actual.data(), quant_config, 0);

  // Exact reference. The intermediate precision is part of the requantize
  // contract, so fast_math selects BF16 here and the default path selects FP32.
  auto dequantized = make_grouped_tensor(shape_info, intermediate_dtype,
                                         NVTE_DELAYED_TENSOR_SCALING,
                                         /*rowwise=*/true, /*columnwise=*/false);
  fill_output_sentinel(dequantized);
  nvte_group_dequantize(input.data(), dequantized.data(), 0);

  auto reference_compact = make_grouped_tensor(
      shape_info, DType::kFloat8E4M3, NVTE_MXFP8_1D_SCALING,
      /*rowwise=*/false, /*columnwise=*/true, /*swizzled=*/false);
  fill_output_sentinel(reference_compact);
  nvte_group_quantize(dequantized.data(), reference_compact.data(), nullptr, 0);

  std::unique_ptr<OwnedGroupedTensor> reference_swizzled;
  if (output_swizzled) {
    reference_swizzled = std::make_unique<OwnedGroupedTensor>(make_grouped_tensor(
        shape_info, DType::kFloat8E4M3, NVTE_MXFP8_1D_SCALING,
        /*rowwise=*/false, /*columnwise=*/true, /*swizzled=*/true));
    fill_output_sentinel(*reference_swizzled);
    nvte_swizzle_grouped_scaling_factors(reference_compact.data(),
                                          reference_swizzled->data(), 0);
  }

  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Scale layout does not affect FP8 data. Only the scale reference changes
  // when the target requests GEMM-swizzled scales.
  expect_bytes_equal("columnwise data", actual.columnwise_data.get(),
                     reference_compact.columnwise_data.get(), actual.data_bytes);
  const void *expected_scales = output_swizzled
                                    ? reference_swizzled->columnwise_scale_inv.get()
                                    : reference_compact.columnwise_scale_inv.get();
  expect_bytes_equal("columnwise scales", actual.columnwise_scale_inv.get(),
                     expected_scales, actual.columnwise_scale_bytes);
  EXPECT_EQ(actual.tensor->get_with_gemm_swizzled_scales(), output_swizzled);
}

const std::vector<GroupShapeCase> kGroupShapeCases = {
    {"SameBothDims_1024x4096",
     {{1024, 4096}, {1024, 4096}, {1024, 4096}, {1024, 4096}}},
    {"SameBothDims_2048x8192",
     {{2048, 8192}, {2048, 8192}, {2048, 8192}, {2048, 8192}}},
    {"SameBothDims_4096x16384",
     {{4096, 16384}, {4096, 16384}, {4096, 16384}, {4096, 16384}}},
    {"VaryingFirstDim_512to2048x4096",
     {{512, 4096}, {512, 4096}, {1024, 4096}, {2048, 4096}}},
    {"VaryingFirstDim_1024to4096x8192",
     {{1024, 8192}, {1024, 8192}, {2048, 8192}, {4096, 8192}}},
    {"VaryingFirstDim_2048to8192x16384",
     {{2048, 16384}, {2048, 16384}, {4096, 16384}, {8192, 16384}}},
    // // An empty member in the middle must not terminate the persistent work loop.
    // {"VaryingFirstDimWithEmpty",
    //  {{128, 256}, {0, 256}, {384, 256}, {512, 256}}},
    // {"VaryingLastDim",
    //  {{256, 128}, {256, 384}, {256, 640}}},
    // {"VaryingBothDims",
    //  {{128, 128}, {256, 384}, {512, 640}}},
};

std::string make_test_name(
    const testing::TestParamInfo<GroupedRequantizeMXFP8TestSuite::ParamType> &info) {
  const auto &test_case = std::get<0>(info.param);
  const DType input_dtype = std::get<1>(info.param);
  const bool use_fast_math = std::get<2>(info.param);
  const bool output_swizzled = std::get<3>(info.param);

  std::string name = test_case.name;
  name += input_dtype == DType::kFloat8E4M3 ? "_E4M3" : "_E5M2";
  name += use_fast_math ? "_FastMath" : "_FP32Math";
  name += output_swizzled ? "_Swizzled" : "_Compact";
  return name;
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, GroupedRequantizeMXFP8TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(kGroupShapeCases),
        ::testing::Values(DType::kFloat8E4M3),
        ::testing::Values(true),
        ::testing::Values(true)
      ),
    make_test_name);

}  // namespace
