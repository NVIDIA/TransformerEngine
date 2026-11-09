/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/recipe.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "util/logging.h"

using namespace transformer_engine;

namespace {

// Keep these tests independent of recipe-specific allocation in the shared test tensor:
// the API under test gets all quantization metadata through TensorWrapper.
class Buffer {
 public:
  Buffer(const char *, const std::vector<size_t> &shape, DType dtype) {
    size_t numel = 1;
    for (auto dim : shape) numel *= dim;
    size_t element_size = 1;
    if (dtype == DType::kFloat32) element_size = 4;
    if (dtype == DType::kBFloat16 || dtype == DType::kFloat16) element_size = 2;
    size_ = numel * element_size;
    host_.resize((size_ + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    NVTE_CHECK_CUDA(cudaMalloc(&device_, size_));
    tensor_.set_rowwise_data(device_, dtype, shape);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ~Buffer() { cudaFree(device_); }
  template <typename T>
  T *rowwise_cpu_dptr() {
    return reinterpret_cast<T *>(host_.data());
  }
  void *rowwise_dptr() { return device_; }
  NVTETensor data() { return tensor_.data(); }
  void from_cpu() {
    NVTE_CHECK_CUDA(cudaMemcpy(device_, host_.data(), size_, cudaMemcpyHostToDevice));
  }
  void to_cpu() {
    NVTE_CHECK_CUDA(cudaMemcpy(host_.data(), device_, size_, cudaMemcpyDeviceToHost));
  }

 private:
  std::vector<uint64_t> host_;
  void *device_ = nullptr;
  size_t size_ = 0;
  TensorWrapper tensor_;
};

uint8_t fp4_reference(float value) {
  const float levels[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  int best = 0;
  for (int i = 1; i < 8; ++i) {
    const float distance = std::abs(std::abs(value) - levels[i]);
    const float previous = std::abs(std::abs(value) - levels[best]);
    if (distance < previous || (distance == previous && i % 2 == 0)) best = i;
  }
  return static_cast<uint8_t>(best | (std::signbit(value) ? 8 : 0));
}

// Use full backing allocations for block FP8, whose in-place all-gather
// contract permits writes to other elements in an overlapping quantization tile.
void check_partial_cast(NVTEScalingMode mode, DType dtype, size_t h, size_t w, size_t offset,
                        size_t length, DType input_dtype,
                        DType inverse_scale_dtype = DType::kFloat8E4M3, int legacy_api = 0) {
  cudaDeviceProp prop;
  NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  if (mode == NVTE_NVFP4_1D_SCALING && prop.major < 10) {
    GTEST_SKIP() << "NVFP4 conversion requires compute capability 10.0 or newer";
  }
  Buffer input("input", std::vector<size_t>{length}, input_dtype);
  std::vector<float> values(length);
  for (size_t i = 0; i < length; ++i) {
    values[i] = static_cast<float>(static_cast<int>(i % 47) - 23) / 8.f;
    if (input_dtype == DType::kBFloat16) {
      input.rowwise_cpu_dptr<__nv_bfloat16>()[i] = __nv_bfloat16(values[i]);
    } else if (input_dtype == DType::kFloat16) {
      input.rowwise_cpu_dptr<__half>()[i] = __half(values[i]);
    } else {
      input.rowwise_cpu_dptr<float>()[i] = values[i];
    }
  }
  input.from_cpu();

  const bool fp4 = mode == NVTE_NVFP4_1D_SCALING;
  const size_t block = fp4 ? 16 : 128;
  const size_t scale_rows = mode == NVTE_MXFP8_1D_SCALING ? ((h + 127) / 128) * 128 : h;
  const size_t scale_cols =
      mode == NVTE_MXFP8_1D_SCALING ? ((w / 32 + 3) / 4) * 4 : (w + block - 1) / block;
  const size_t encoded_rows = mode == NVTE_BLOCK_SCALING_2D ? (h + 127) / 128 : scale_rows;
  const size_t col_rows = ((h / 32 + 3) / 4) * 4;
  const size_t col_cols = ((w + 127) / 128) * 128;
  const DType scale_dtype = mode == NVTE_BLOCK_SCALING_2D ? DType::kFloat32 : DType::kByte;
  Buffer scales("scales", std::vector<size_t>{encoded_rows, scale_cols}, scale_dtype);
  Buffer col_scales("col_scales", std::vector<size_t>{col_rows, col_cols}, DType::kByte);
  Buffer global("global", std::vector<size_t>{1}, DType::kFloat32);
  global.rowwise_cpu_dptr<float>()[0] = 2.f;
  global.from_cpu();
  for (size_t r = 0; r < encoded_rows; ++r) {
    for (size_t c = 0; c < scale_cols; ++c) {
      const size_t i = r * scale_cols + c;
      if (mode == NVTE_BLOCK_SCALING_2D) {
        const size_t scale_index = legacy_api ? c * encoded_rows + r : i;
        scales.rowwise_cpu_dptr<float>()[scale_index] =
            static_cast<float>(1u << ((r + 2 * c) % 3));
      } else if (fp4) {
#if CUDA_VERSION >= 13040
        if (inverse_scale_dtype == DType::kFloat8UE5M3) {
          const __nv_fp8_ue5m3 scale(static_cast<float>(1u << ((r / 16 + c) % 3)));
          scales.rowwise_cpu_dptr<uint8_t>()[i] = *reinterpret_cast<const uint8_t *>(&scale);
          continue;
        }
#endif
        const __nv_fp8_e4m3 scale(static_cast<float>(1u << ((r / 16 + c) % 3)));
        scales.rowwise_cpu_dptr<uint8_t>()[i] = *reinterpret_cast<const uint8_t *>(&scale);
      } else {
        scales.rowwise_cpu_dptr<uint8_t>()[i] = 127 + (r + c) % 3;
      }
    }
  }
  for (size_t i = 0; i < col_rows * col_cols; ++i) {
    col_scales.rowwise_cpu_dptr<uint8_t>()[i] = 127 + (i / col_cols + i % col_cols + 1) % 3;
  }
  scales.from_cpu();
  col_scales.from_cpu();

  Buffer row_storage("row_output", std::vector<size_t>{fp4 ? (h * w + 1) / 2 : h * w},
                     DType::kByte);
  Buffer col_storage("col_output", std::vector<size_t>{h * w}, DType::kByte);
  std::fill_n(row_storage.rowwise_cpu_dptr<uint8_t>(), fp4 ? (h * w + 1) / 2 : h * w, 0xa5);
  std::fill_n(col_storage.rowwise_cpu_dptr<uint8_t>(), h * w, 0xa5);
  row_storage.from_cpu();
  col_storage.from_cpu();
  TensorWrapper output(mode);
  output.set_rowwise_data(
      static_cast<uint8_t *>(row_storage.rowwise_dptr()) + (fp4 ? offset / 2 : offset), dtype,
      std::vector<size_t>{length});
  if (mode == NVTE_BLOCK_SCALING_2D) {
    output.set_scale(scales.rowwise_dptr(), DType::kFloat32,
                     std::vector<size_t>{encoded_rows, scale_cols});
  } else if (fp4) {
    output.set_scale(global.rowwise_dptr(), DType::kFloat32, std::vector<size_t>{1})
        .set_rowwise_scale_inv(scales.rowwise_dptr(), inverse_scale_dtype,
                               std::vector<size_t>{encoded_rows, scale_cols});
  } else {
    output
        .set_columnwise_data(static_cast<uint8_t *>(col_storage.rowwise_dptr()) + offset, dtype,
                             std::vector<size_t>{length})
        .set_rowwise_scale_inv(scales.rowwise_dptr(), DType::kFloat8E8M0,
                               std::vector<size_t>{encoded_rows, scale_cols})
        .set_columnwise_scale_inv(col_scales.rowwise_dptr(), DType::kFloat8E8M0,
                                  std::vector<size_t>{col_rows, col_cols});
  }
  if (legacy_api == 0) {
    nvte_partial_cast(input.data(), output.data(), h, w, offset, nullptr);
  } else {
    TensorWrapper row;
    row.set_rowwise_data(output.dptr(), DType::kByte,
                         std::vector<size_t>{fp4 ? (length + 1) / 2 : length});
    if (mode == NVTE_BLOCK_SCALING_2D) {
      nvte_fp8_block_scaling_partial_cast(input.data(), row.data(), scales.data(), h, w,
                                          1, encoded_rows, offset, 128,
                                          static_cast<NVTEDType>(dtype), nullptr);
    } else if (fp4) {
      const size_t tile_rows = (h + 15) / 16;
      Buffer tiles("legacy_tile_scales", std::vector<size_t>{scale_cols, tile_rows},
                    DType::kFloat32);
      for (size_t r = 0; r < tile_rows; ++r) {
        for (size_t c = 0; c < scale_cols; ++c) {
          // The adapter must round float32 tile scales to their encoded type.
          tiles.rowwise_cpu_dptr<float>()[c * tile_rows + r] =
              static_cast<float>(1u << ((r + c) % 3)) + .01f;
        }
      }
      tiles.from_cpu();
      if (legacy_api == 1) {
        nvte_nvfp4_2d_partial_cast(input.data(), row.data(), tiles.data(), global.data(), h, w,
                                   1, tile_rows, offset, 16, nullptr);
      } else {
        nvte_nvfp4_2d_partial_cast_v2(input.data(), row.data(), tiles.data(), global.data(), h, w,
                                      1, tile_rows, offset, 16,
                                      static_cast<NVTEDType>(inverse_scale_dtype), nullptr);
      }
    } else {
      TensorWrapper col;
      col.set_rowwise_data(output.get_columnwise_data().data_ptr, DType::kByte,
                           std::vector<size_t>{length});
      nvte_mxfp8_scaling_partial_cast(input.data(), row.data(), col.data(), scales.data(),
                                       col_scales.data(), h, w, offset, nullptr);
    }
  }
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  row_storage.to_cpu();
  col_storage.to_cpu();
  for (size_t i = 0; i < length; ++i) {
    const size_t index = offset + i;
    const size_t r = index / w;
    const size_t c = index % w;
    float scaled = values[i];
    if (mode == NVTE_BLOCK_SCALING_2D) {
      scaled *= static_cast<float>(1u << ((r / 128 + 2 * (c / 128)) % 3));
    } else if (fp4) {
      scaled *= 2.f / static_cast<float>(1u << ((r / 16 + c / 16) % 3));
    } else {
      scaled /= static_cast<float>(1u << ((r + c / 32) % 3));
    }
    if (fp4) {
      const uint8_t actual =
          (row_storage.rowwise_cpu_dptr<uint8_t>()[index / 2] >> (4 * (index % 2))) & 0xf;
      ASSERT_EQ(actual, fp4_reference(scaled)) << "index=" << index;
    } else if (dtype == DType::kFloat8E5M2) {
      const __nv_fp8_e5m2 expected(scaled);
      ASSERT_EQ(row_storage.rowwise_cpu_dptr<uint8_t>()[index],
                *reinterpret_cast<const uint8_t *>(&expected));
    } else {
      const __nv_fp8_e4m3 expected(scaled);
      ASSERT_EQ(row_storage.rowwise_cpu_dptr<uint8_t>()[index],
                *reinterpret_cast<const uint8_t *>(&expected));
    }
    if (mode == NVTE_MXFP8_1D_SCALING) {
      const __nv_fp8_e4m3 expected(values[i] / static_cast<float>(1u << ((r / 32 + c + 1) % 3)));
      ASSERT_EQ(col_storage.rowwise_cpu_dptr<uint8_t>()[index],
                *reinterpret_cast<const uint8_t *>(&expected));
    }
  }
}

TEST(PartialCast, BlockFp8) {
  for (auto input_dtype : {DType::kFloat32, DType::kBFloat16, DType::kFloat16}) {
    for (auto dtype : {DType::kFloat8E4M3, DType::kFloat8E5M2}) {
      check_partial_cast(NVTE_BLOCK_SCALING_2D, dtype, 132, 130, 126, 395, input_dtype);
      check_partial_cast(NVTE_BLOCK_SCALING_2D, dtype, 128, 128, 0, 128 * 128, input_dtype);
    }
  }
}

TEST(PartialCast, Mxfp8) {
  for (auto input_dtype : {DType::kFloat32, DType::kBFloat16, DType::kFloat16}) {
    check_partial_cast(NVTE_MXFP8_1D_SCALING, DType::kFloat8E4M3, 64, 256, 31, 8193, input_dtype);
  }
}

TEST(PartialCast, Nvfp4) {
#if CUDA_VERSION >= 12080
  for (auto input_dtype : {DType::kFloat32, DType::kBFloat16, DType::kFloat16}) {
    check_partial_cast(NVTE_NVFP4_1D_SCALING, DType::kFloat4E2M1, 32, 64, 30, 1043, input_dtype);
#if CUDA_VERSION >= 13040
    check_partial_cast(NVTE_NVFP4_1D_SCALING, DType::kFloat4E2M1, 32, 64, 30, 1043, input_dtype,
                       DType::kFloat8UE5M3);
#endif
  }
#else
  GTEST_SKIP() << "NVFP4 conversion requires CUDA 12.8 or newer";
#endif
}

TEST(PartialCast, LegacyEntryPoints) {
  for (auto input_dtype : {DType::kFloat32, DType::kBFloat16, DType::kFloat16}) {
    for (auto dtype : {DType::kFloat8E4M3, DType::kFloat8E5M2}) {
      check_partial_cast(NVTE_BLOCK_SCALING_2D, dtype, 132, 130, 126, 395, input_dtype,
                         DType::kFloat8E4M3, 1);
    }
    check_partial_cast(NVTE_MXFP8_1D_SCALING, DType::kFloat8E4M3, 64, 256, 31, 8193,
                       input_dtype, DType::kFloat8E4M3, 1);
  }
}

TEST(PartialCast, LegacyNvfp4EntryPoints) {
#if CUDA_VERSION >= 12080
  for (int api : {1, 2}) {
    check_partial_cast(NVTE_NVFP4_1D_SCALING, DType::kFloat4E2M1, 32, 64, 30, 1043,
                       DType::kFloat32, DType::kFloat8E4M3, api);
  }
#if CUDA_VERSION >= 13040
  check_partial_cast(NVTE_NVFP4_1D_SCALING, DType::kFloat4E2M1, 32, 64, 30, 1043,
                     DType::kFloat32, DType::kFloat8UE5M3, 2);
#endif
#else
  GTEST_SKIP() << "NVFP4 conversion requires CUDA 12.8 or newer";
#endif
}

TEST(PartialCast, RejectInvalidMetadata) {
  float input_data = 1.f;
  uint8_t output_data = 0;
  TensorWrapper input(&input_data, std::vector<size_t>{1}, DType::kFloat32);
  TensorWrapper output;
  output.set_rowwise_data(&output_data, DType::kFloat8E4M3, std::vector<size_t>{1});
  EXPECT_THROW(nvte_partial_cast(input.data(), output.data(), 128, 128, 0, nullptr),
               std::runtime_error);
  TensorWrapper block_output(NVTE_BLOCK_SCALING_2D);
  block_output.set_rowwise_data(&output_data, DType::kFloat8E4M3, std::vector<size_t>{1});
  EXPECT_THROW(nvte_partial_cast(input.data(), block_output.data(), 128, 128, 16384, nullptr),
               std::runtime_error);
  EXPECT_THROW(nvte_partial_cast(input.data(), block_output.data(), 128, 128, 0, nullptr),
               std::runtime_error);
  EXPECT_THROW(nvte_partial_cast(input.data(), block_output.data(),
                                 std::numeric_limits<size_t>::max(), 2, 0, nullptr),
               std::runtime_error);
}

TEST(PartialCast, Mxfp8UsageFollowsOutputData) {
  Buffer input("row_only_input", std::vector<size_t>{3}, DType::kFloat32);
  Buffer row("row_only_output", std::vector<size_t>{3}, DType::kByte);
  Buffer col("columnwise_output", std::vector<size_t>{3}, DType::kByte);
  Buffer scales("row_only_scales", std::vector<size_t>{128, 4}, DType::kByte);
  const float values[] = {1.f, 2.f, -3.f};
  std::copy_n(values, 3, input.rowwise_cpu_dptr<float>());
  std::fill_n(scales.rowwise_cpu_dptr<uint8_t>(), 128 * 4, 127);
  input.from_cpu();
  scales.from_cpu();

  TensorWrapper output(NVTE_MXFP8_1D_SCALING);
  output.set_rowwise_data(row.rowwise_dptr(), DType::kFloat8E4M3, std::vector<size_t>{3})
      .set_rowwise_scale_inv(scales.rowwise_dptr(), DType::kFloat8E8M0,
                             std::vector<size_t>{128, 4})
      // Invalid auxiliary metadata must be ignored without columnwise data.
      .set_columnwise_scale_inv(input.rowwise_dptr(), DType::kFloat32, std::vector<size_t>{3});
  nvte_partial_cast(input.data(), output.data(), 64, 64, 31, nullptr);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  row.to_cpu();
  for (size_t i = 0; i < 3; ++i) {
    const __nv_fp8_e4m3 expected(values[i]);
    EXPECT_EQ(row.rowwise_cpu_dptr<uint8_t>()[i], *reinterpret_cast<const uint8_t *>(&expected));
  }

  // With output data present, invalid columnwise scales must be rejected.
  output.set_columnwise_data(col.rowwise_dptr(), DType::kFloat8E4M3, std::vector<size_t>{3});
  EXPECT_THROW(nvte_partial_cast(input.data(), output.data(), 64, 64, 31, nullptr),
               std::runtime_error);

  // The legacy wrapper normalizes an empty output view with a backing pointer,
  // and also ignores the auxiliary scale tensor when output is absent.
  TensorWrapper omitted;
  omitted.set_rowwise_data(col.rowwise_dptr(), DType::kByte, std::vector<size_t>{0});
  nvte_mxfp8_scaling_partial_cast(input.data(), row.data(), omitted.data(), scales.data(),
                                   input.data(), 64, 64, 31, nullptr);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  row.to_cpu();
  for (size_t i = 0; i < 3; ++i) {
    const __nv_fp8_e4m3 expected(values[i]);
    EXPECT_EQ(row.rowwise_cpu_dptr<uint8_t>()[i], *reinterpret_cast<const uint8_t *>(&expected));
  }
}

TEST(PartialCast, EmptyShard) {
  TensorWrapper input;
  TensorWrapper output(NVTE_BLOCK_SCALING_2D);
  EXPECT_NO_THROW(nvte_partial_cast(input.data(), output.data(), 128, 128, 128 * 128, nullptr));
}

TEST(PartialCast, RejectUnsupportedLayouts) {
  float input_data = 1.f;
  uint8_t output_data = 0;
  float global_scale = 1.f;
  TensorWrapper input(&input_data, std::vector<size_t>{1}, DType::kFloat32);
  TensorWrapper output(NVTE_NVFP4_1D_SCALING);
  output.set_rowwise_data(&output_data, DType::kFloat4E2M1, std::vector<size_t>{1})
      .set_rowwise_scale_inv(&output_data, DType::kFloat8E4M3, std::vector<size_t>{32, 4})
      .set_scale(&global_scale, DType::kFloat32, std::vector<size_t>{1});
  output.set_row_scaled_nvfp4(true);
  EXPECT_THROW(nvte_partial_cast(input.data(), output.data(), 32, 64, 0, nullptr),
               std::runtime_error);
  output.set_row_scaled_nvfp4(false);
  output.set_with_gemm_swizzled_scales(true);
  EXPECT_THROW(nvte_partial_cast(input.data(), output.data(), 32, 64, 0, nullptr),
               std::runtime_error);
}

}  // namespace
