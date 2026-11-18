/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/cast.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace {

using transformer_engine::DType;
using transformer_engine::TensorWrapper;

struct Block128Contract {
  transformer_engine::GroupedTensorWrapper input;
  transformer_engine::GroupedTensorWrapper output;
  TensorWrapper dequantized;

  explicit Block128Contract(size_t rows = 0)
      : input(0, std::vector<size_t>{rows, 128}, NVTE_BLOCK_SCALING_1D),
        output(0, std::vector<size_t>{rows, 128}, NVTE_BLOCK_SCALING_1D) {
    input.set_rowwise_data(nullptr, DType::kFloat8E4M3, std::vector<size_t>{rows * 128});
    input.set_rowwise_scale_inv(nullptr, DType::kFloat32, std::vector<size_t>{rows});
    input.set_with_gemm_swizzled_scales(true);
    input.set_tensor_offsets(nullptr, DType::kInt64, std::vector<size_t>{1});
    output.set_rowwise_scale_inv(nullptr, DType::kFloat32, std::vector<size_t>{rows});
    dequantized.set_rowwise_data(nullptr, DType::kBFloat16, std::vector<size_t>{rows, 128});
  }

  void run() {
    nvte_group_requantize(input.data(), output.data(), dequantized.data(), nullptr, nullptr);
  }
};

void expect_error(Block128Contract& tensors, const char* message) {
  try {
    tensors.run();
    FAIL() << "Expected common API rejection containing: " << message;
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
  }
}

TEST(Block128GroupRequantizeContract, EmptyWorkDoesNotRequireStorage) {
  for (size_t capacity : {0, 128}) {
    Block128Contract tensors(capacity);
    EXPECT_NO_THROW(tensors.run());
  }
}

TEST(Block128GroupRequantizeContract, ValidatesScaleCapacityBeforeEmptyReturn) {
  Block128Contract tensors(128);
  tensors.input.set_rowwise_scale_inv(nullptr, DType::kFloat32, std::vector<size_t>{64});
  expect_error(tensors, "scale capacity");
}

TEST(Block128GroupRequantizeContract, ValidatesOffsetsMetadataBeforeEmptyReturn) {
  Block128Contract tensors;
  tensors.input.set_tensor_offsets(nullptr, DType::kInt32, std::vector<size_t>{1});
  expect_error(tensors, "int64 tensor offsets");
  tensors.input.set_tensor_offsets(nullptr, DType::kInt64, std::vector<size_t>{0});
  expect_error(tensors, "num_tensors + 1 offsets");
}

TEST(Block128GroupRequantizeContract, ValidatesBF16ReturnMetadata) {
  Block128Contract tensors;
  tensors.dequantized.set_rowwise_data(nullptr, DType::kFloat32, std::vector<size_t>{0, 128});
  expect_error(tensors, "BF16 with the input shape");
  tensors.dequantized.set_rowwise_data(nullptr, DType::kBFloat16, std::vector<size_t>{1, 128});
  expect_error(tensors, "BF16 with the input shape");
}

TEST(Block128GroupRequantizeContract, RejectsVaryingLastDimension) {
  // Host metadata pointers are sufficient: unsupported shapes must fail before launch.
  int64_t dim = 128;
  for (bool varying_first : {false, true}) {
    Block128Contract tensors;
    tensors.input.set_last_dims(&dim, DType::kInt64, std::vector<size_t>{1});
    tensors.output.set_last_dims(&dim, DType::kInt64, std::vector<size_t>{1});
    if (varying_first) {
      tensors.input.set_first_dims(&dim, DType::kInt64, std::vector<size_t>{1});
      tensors.output.set_first_dims(&dim, DType::kInt64, std::vector<size_t>{1});
    }
    expect_error(tensors, "supports only SAME_BOTH_DIMS and VARYING_FIRST_DIM");
  }
}

}  // namespace
