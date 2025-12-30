/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include <transformer_engine/cast.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/recipe.h>
#include <transformer_engine/transformer_engine.h>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum class InputCase {
  kFP8Current,
  kBF16,
};

enum class ShapeCase {
  kAllSame,
  kSameFirst,
  kSameLast,
  kAllDifferent,
};

// Custom deleters for RAII
struct CudaDeleter {
  void operator()(void* p) const { if (p) cudaFree(p); }
};
struct GroupedTensorDeleter {
  void operator()(NVTEGroupedTensor h) const { if (h) nvte_destroy_grouped_tensor(h); }
};

template <typename T = void>
using CudaPtr = std::unique_ptr<T, CudaDeleter>;
using GroupedTensorHandle = std::unique_ptr<std::remove_pointer_t<NVTEGroupedTensor>, GroupedTensorDeleter>;

// Helper to allocate CUDA memory into a CudaPtr
template <typename T = void>
CudaPtr<T> cuda_alloc(size_t bytes) {
  void* ptr = nullptr;
  NVTE_CHECK_CUDA(cudaMalloc(&ptr, bytes));
  return CudaPtr<T>(static_cast<T*>(ptr));
}

// Helper owning GPU buffers that back NVTEGroupedTensor.
// NVTEGroupedTensor does not own memory; data/offsets/scales
// must be allocated and freed by the test.
struct GroupedBuffers {
  GroupedTensorHandle handle;
  CudaPtr<> data;
  CudaPtr<> scale_inv;
  CudaPtr<int64_t> first_dims_dev;
  CudaPtr<int64_t> last_dims_dev;
  CudaPtr<int64_t> offsets_dev;
  CudaPtr<> columnwise_data;
  NVTEShape logical_shape{};
  std::vector<int64_t> offsets_host;
  std::vector<size_t> tensor_bytes;
  size_t num_tensors{0};
  size_t elem_size{0};
  DType dtype{DType::kFloat32};
  NVTEScalingMode scaling_mode{NVTE_DELAYED_TENSOR_SCALING};

  GroupedBuffers() = default;
  GroupedBuffers(const GroupedBuffers&) = delete;
  GroupedBuffers& operator=(const GroupedBuffers&) = delete;
  GroupedBuffers(GroupedBuffers&&) = default;
  GroupedBuffers& operator=(GroupedBuffers&&) = default;
  ~GroupedBuffers() = default;

  // Convenience accessors for raw pointers
  NVTEGroupedTensor get_handle() const { return handle.get(); }
  void* get_data() const { return data.get(); }
};

size_t grouped_setup_workspace_size(const size_t num_tensors) {
  const size_t ptr_bytes = num_tensors * sizeof(void*);
  const size_t int_bytes = num_tensors * sizeof(int);
  // Layout: 6 pointer arrays (A, B, C, D, alpha, beta) + 3 int arrays (M, N, K)
  size_t size = 6 * ptr_bytes + 3 * int_bytes;
  const size_t alignment = 256;
  size = ((size + alignment - 1) / alignment) * alignment;
  return size;
}

GroupedBuffers build_grouped_tensor(const std::vector<Tensor*>& tensors,
                                    const NVTEScalingMode scaling_mode) {
  NVTE_CHECK(!tensors.empty(), "No tensors provided for grouped tensor build.");
  const NVTEShape shape = tensors[0]->rowwise_shape();
  const DType dtype = tensors[0]->dtype();
  const size_t num_tensors = tensors.size();
  const size_t elem_size = typeToNumBits(dtype) / 8;
  GroupedBuffers grouped;
  grouped.elem_size = elem_size;
  grouped.num_tensors = num_tensors;
  grouped.dtype = dtype;
  grouped.scaling_mode = scaling_mode;
  grouped.tensor_bytes.resize(num_tensors);
  grouped.offsets_host.resize(num_tensors, 0);

  std::vector<int64_t> first_dims(num_tensors);
  std::vector<int64_t> last_dims(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    const auto s = tensors[i]->rowwise_shape();
    NVTE_CHECK(s.ndim == 2, "Grouped GEMM test expects 2D tensors.");
    first_dims[i] = static_cast<int64_t>(s.data[0]);
    last_dims[i] = static_cast<int64_t>(s.data[1]);
    grouped.tensor_bytes[i] = bytes(s, dtype);
  }

  const bool same_first = std::all_of(first_dims.begin(), first_dims.end(),
                                      [&](int64_t v) { return v == first_dims[0]; });
  const bool same_last = std::all_of(last_dims.begin(), last_dims.end(),
                                     [&](int64_t v) { return v == last_dims[0]; });

  std::vector<int64_t> offsets(num_tensors, 0);
  auto random_padding = [&]() -> int64_t {
    // Random padding ensuring 16-byte alignment regardless of element size
    // cuBLAS requires aligned pointers for vectorized loads
    static std::mt19937 gen(12345);
    std::uniform_int_distribution<int64_t> dist(0, 3);
    // Calculate elements needed for 16-byte alignment in bytes, rounded up
    const size_t align_elements =
        std::max<size_t>(1, (16 + elem_size - 1) / elem_size);  // 16 bytes / element_size
    return dist(gen) * static_cast<int64_t>(align_elements);
  };

  auto numel = [&](size_t idx) -> int64_t {
    return first_dims[idx] * last_dims[idx];
  };

  const bool need_offsets = !same_first || !same_last;
  if (need_offsets) {
    offsets[0] = 0;
    for (size_t i = 1; i < num_tensors; ++i) {
      offsets[i] = offsets[i - 1] + numel(i - 1) + random_padding();
    }
  } else {
    for (size_t i = 0; i < num_tensors; ++i) {
      offsets[i] = static_cast<int64_t>(i) * numel(0);
    }
  }
  grouped.offsets_host = offsets;

  int64_t logical_first = 0;
  int64_t logical_last = 0;
  if (same_first && same_last) {
    logical_first = first_dims[0] * static_cast<int64_t>(num_tensors);
    logical_last = last_dims[0];
  } else if (same_first && !same_last) {
    logical_first = first_dims[0];
    logical_last = std::accumulate(last_dims.begin(), last_dims.end(), int64_t{0});
  } else if (!same_first && same_last) {
    logical_first = std::accumulate(first_dims.begin(), first_dims.end(), int64_t{0});
    logical_last = last_dims[0];
  } else {
    logical_first = 1;
    logical_last = 0;
    for (size_t i = 0; i < num_tensors; ++i) {
      logical_last += first_dims[i] * last_dims[i];
    }
  }
  size_t logical_data[2] = {static_cast<size_t>(logical_first),
                            static_cast<size_t>(logical_last)};
  grouped.logical_shape = nvte_make_shape(logical_data, 2);
  grouped.handle.reset(nvte_create_grouped_tensor(scaling_mode, num_tensors, grouped.logical_shape));

  const int64_t last_idx = static_cast<int64_t>(num_tensors - 1);
  const int64_t total_elems = need_offsets
                                  ? (offsets[last_idx] + numel(last_idx))
                                  : (logical_first * logical_last);
  const size_t total_bytes = static_cast<size_t>(total_elems) * elem_size;

  grouped.data = cuda_alloc(total_bytes);
  for (size_t i = 0; i < num_tensors; ++i) {
    const size_t offset_bytes = static_cast<size_t>(offsets[i]) * elem_size;
    NVTE_CHECK_CUDA(cudaMemcpy(static_cast<char*>(grouped.data.get()) + offset_bytes,
                               tensors[i]->rowwise_dptr(),
                               grouped.tensor_bytes[i],
                               cudaMemcpyDeviceToDevice));
  }

  NVTEBasicTensor data_tensor{grouped.data.get(), static_cast<NVTEDType>(dtype), grouped.logical_shape};
  NVTEGroupedTensor h = grouped.handle.get();
  nvte_set_grouped_tensor_param(&h, kNVTEGroupedRowwiseData, &data_tensor);

  const bool include_columnwise = isFp8Type(dtype) || isFp4Type(dtype);
  if (include_columnwise) {
    grouped.columnwise_data = cuda_alloc(total_bytes);
    for (size_t i = 0; i < num_tensors; ++i) {
      const size_t offset_bytes = static_cast<size_t>(offsets[i]) * elem_size;
      NVTE_CHECK_CUDA(cudaMemcpy(static_cast<char*>(grouped.columnwise_data.get()) + offset_bytes,
                                 tensors[i]->columnwise_dptr(),
                                 grouped.tensor_bytes[i],
                                 cudaMemcpyDeviceToDevice));
    }
    NVTEBasicTensor col_tensor{grouped.columnwise_data.get(),
                               static_cast<NVTEDType>(dtype),
                               grouped.logical_shape};
    nvte_set_grouped_tensor_param(&h, kNVTEGroupedColumnwiseData, &col_tensor);
  }

  if (!same_first) {
    grouped.first_dims_dev = cuda_alloc<int64_t>(num_tensors * sizeof(int64_t));
    NVTE_CHECK_CUDA(cudaMemcpy(grouped.first_dims_dev.get(), first_dims.data(),
                               num_tensors * sizeof(int64_t), cudaMemcpyHostToDevice));
    NVTEShape fd_shape = nvte_make_shape(&num_tensors, 1);
    NVTEBasicTensor fd_tensor{grouped.first_dims_dev.get(), kNVTEInt64, fd_shape};
    nvte_set_grouped_tensor_param(&h, kNVTEGroupedFirstDims, &fd_tensor);
  }

  if (!same_last) {
    grouped.last_dims_dev = cuda_alloc<int64_t>(num_tensors * sizeof(int64_t));
    NVTE_CHECK_CUDA(cudaMemcpy(grouped.last_dims_dev.get(), last_dims.data(),
                               num_tensors * sizeof(int64_t), cudaMemcpyHostToDevice));
    NVTEShape ld_shape = nvte_make_shape(&num_tensors, 1);
    NVTEBasicTensor ld_tensor{grouped.last_dims_dev.get(), kNVTEInt64, ld_shape};
    nvte_set_grouped_tensor_param(&h, kNVTEGroupedLastDims, &ld_tensor);
  }

  if (!same_first || !same_last) {
    grouped.offsets_dev = cuda_alloc<int64_t>(num_tensors * sizeof(int64_t));
    NVTE_CHECK_CUDA(cudaMemcpy(grouped.offsets_dev.get(), offsets.data(),
                               num_tensors * sizeof(int64_t), cudaMemcpyHostToDevice));
    NVTEShape off_shape = nvte_make_shape(&num_tensors, 1);
    NVTEBasicTensor off_tensor{grouped.offsets_dev.get(), kNVTEInt64, off_shape};
    nvte_set_grouped_tensor_param(&h, kNVTEGroupedTensorOffsets, &off_tensor);
  }

  if (isFp8Type(dtype)) {
    std::vector<float> scale_inv_cpu(num_tensors, 1.f);
    for (size_t i = 0; i < num_tensors; ++i) {
      tensors[i]->to_cpu();
      scale_inv_cpu[i] = tensors[i]->rowwise_cpu_scale_inv_ptr<float>()[0];
    }
    grouped.scale_inv = cuda_alloc(sizeof(float) * num_tensors);
    NVTE_CHECK_CUDA(cudaMemcpy(grouped.scale_inv.get(), scale_inv_cpu.data(),
                               sizeof(float) * num_tensors, cudaMemcpyHostToDevice));
    NVTEShape scale_shape = nvte_make_shape(&num_tensors, 1);
    NVTEBasicTensor scale_tensor{grouped.scale_inv.get(), kNVTEFloat32, scale_shape};
    nvte_set_grouped_tensor_param(&h, kNVTEGroupedRowwiseScaleInv, &scale_tensor);
    nvte_set_grouped_tensor_param(&h, kNVTEGroupedColumnwiseScaleInv, &scale_tensor);
  }

  return grouped;
}

Tensor make_fp8_operand(const std::string& name, const std::vector<size_t>& shape) {
  Tensor input_fp32(name + "_fp32", shape, DType::kFloat32);
  fillUniform(&input_fp32);

  Tensor fp8(name, shape, TypeInfo<fp8e4m3>::dtype, true, true, NVTE_DELAYED_TENSOR_SCALING);

  nvte_compute_amax(input_fp32.data(), fp8.data(), 0);
  QuantizationConfigWrapper config;
  nvte_compute_scale_from_amax(fp8.data(), config, 0);
  nvte_quantize(input_fp32.data(), fp8.data(), 0);
  return fp8;
}

Tensor make_bf16_operand(const std::string& name, const std::vector<size_t>& shape) {
  Tensor t(name, shape, DType::kBFloat16);
  // Fill with ones for easier debugging
  //fillUniform(&t);
  const size_t numel = shape[0] * shape[1];
  std::vector<__nv_bfloat16> ones(numel, __float2bfloat16(1.0f));
  NVTE_CHECK_CUDA(cudaMemcpy(t.rowwise_dptr(), ones.data(),
                             numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  return t;
}

struct TestParams {
  InputCase input_case;
  bool transa;
  bool transb;
  ShapeCase shape_case;
  bool use_null_c = false;           // When true, pass nullptr for C (valid when beta=0)
  bool use_split_accumulator = false; // Whether to use split accumulator for FP8 GEMM
};

// Returns a vector of (M, N, K) tuples for each GEMM in the group.
// M - number of rows in output D
// N - number of columns in output D
// K - reduction dimension shared between A and B
std::vector<std::tuple<size_t, size_t, size_t>> make_shapes(ShapeCase scase) {
  switch (scase) {
    case ShapeCase::kAllSame:
      return {{64, 64, 32}, {64, 64, 32}, {64, 64, 32}};
    case ShapeCase::kSameFirst:
      // Same M (first dim), varying N and K
      return {{64, 80, 32}, {64, 96, 48}, {64, 112, 64}};
    case ShapeCase::kSameLast:
      // Same N (last dim), varying M and K
      return {{64, 80, 32}, {80, 80, 48}, {96, 80, 64}};
    case ShapeCase::kAllDifferent:
    default:
      return {{64, 96, 32}, {80, 112, 48}, {96, 128, 64}};
  }
}

void run_grouped_gemm_case(const TestParams& params) {
#if CUBLAS_VERSION < 130100
  GTEST_SKIP() << "Grouped GEMM requires cuBLAS 13.1+, but compile-time cuBLAS version is "
               << CUBLAS_VERSION << ".";
#else
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP() << "Grouped GEMM requires Blackwell (SM100) or newer.";
  }

  const std::vector<std::tuple<size_t, size_t, size_t>> shapes = make_shapes(params.shape_case);

  const size_t num_gemms = shapes.size();
  std::vector<Tensor> A_tensors;
  std::vector<Tensor> B_tensors;
  std::vector<Tensor> D_multi;

  A_tensors.reserve(num_gemms);
  B_tensors.reserve(num_gemms);
  D_multi.reserve(num_gemms);

  for (size_t i = 0; i < num_gemms; ++i) {
    const auto [M, N, K] = shapes[i];
    const std::vector<size_t> a_shape = params.transa ? std::vector<size_t>{M, K}
                                                      : std::vector<size_t>{K, M};
    const std::vector<size_t> b_shape = params.transb ? std::vector<size_t>{K, N}
                                                      : std::vector<size_t>{N, K};
    switch (params.input_case) {
      case InputCase::kFP8Current: {
        A_tensors.emplace_back(make_fp8_operand("A" + std::to_string(i), a_shape));
        B_tensors.emplace_back(make_fp8_operand("B" + std::to_string(i), b_shape));
        break;
      }
      case InputCase::kBF16: {
        A_tensors.emplace_back(make_bf16_operand("A" + std::to_string(i), a_shape));
        B_tensors.emplace_back(make_bf16_operand("B" + std::to_string(i), b_shape));
        break;
      }
    }
    D_multi.emplace_back(Tensor("D_multi" + std::to_string(i),
                                std::vector<size_t>{M, N},
                                DType::kBFloat16));
  }

  std::vector<NVTETensor> A_ptrs(num_gemms);
  std::vector<NVTETensor> B_ptrs(num_gemms);
  std::vector<NVTETensor> D_ptrs(num_gemms);
  std::vector<Tensor> workspaces(num_gemms);
  std::vector<NVTETensor> workspace_ptrs(num_gemms, nullptr);
  std::vector<Tensor*> A_views;
  std::vector<Tensor*> B_views;
  A_views.reserve(num_gemms);
  B_views.reserve(num_gemms);

  // Empty bias/gelu arrays for nvte_multi_tensor_gemm (no epilogues)
  std::vector<NVTETensor> bias_ptrs(num_gemms, nullptr);
  std::vector<NVTETensor> gelu_ptrs(num_gemms, nullptr);

  const size_t cublas_ws_bytes = 32ull * 1024 * 1024;

  for (size_t i = 0; i < num_gemms; ++i) {
    A_ptrs[i] = A_tensors[i].data();
    B_ptrs[i] = B_tensors[i].data();
    D_ptrs[i] = D_multi[i].data();
    workspaces[i] = Tensor("workspace" + std::to_string(i), std::vector<size_t>{cublas_ws_bytes}, DType::kByte);
    workspace_ptrs[i] = workspaces[i].data();
    A_views.push_back(&A_tensors[i]);
    B_views.push_back(&B_tensors[i]);
  }

  nvte_multi_tensor_gemm(A_ptrs.data(),
                         B_ptrs.data(),
                         D_ptrs.data(),
                         bias_ptrs.data(),
                         gelu_ptrs.data(),
                         static_cast<int>(num_gemms),
                         params.transa,
                         params.transb,
                         false,  // grad
                         workspace_ptrs.data(),
                         false,  // accumulate
                         params.use_split_accumulator,
                         0,      // sm_count
                         0);

  GroupedBuffers grouped_A = build_grouped_tensor(A_views, A_tensors[0].scaling_mode());
  GroupedBuffers grouped_B = build_grouped_tensor(B_views, B_tensors[0].scaling_mode());

  std::vector<Tensor> C_tensors;
  std::vector<Tensor> D_group_tensors;
  C_tensors.reserve(num_gemms);
  D_group_tensors.reserve(num_gemms);
  for (size_t i = 0; i < num_gemms; ++i) {
    const auto [M, N, K] = shapes[i];
    (void)K;
    if (!params.use_null_c) {
      C_tensors.emplace_back(Tensor("C" + std::to_string(i),
                                    std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)},
                                    DType::kBFloat16));
    }
    D_group_tensors.emplace_back(Tensor("D_group" + std::to_string(i),
                                        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)},
                                        DType::kBFloat16));
    NVTE_CHECK_CUDA(cudaMemset(D_group_tensors.back().rowwise_dptr(), 0, bytes(D_group_tensors.back().rowwise_shape(), D_group_tensors.back().dtype())));
  }

  std::vector<Tensor*> C_views, D_views;
  for (size_t i = 0; i < num_gemms; ++i) {
    if (!params.use_null_c) {
      C_views.push_back(&C_tensors[i]);
    }
    D_views.push_back(&D_group_tensors[i]);
  }

  std::optional<GroupedBuffers> grouped_C;
  if (!params.use_null_c) {
    grouped_C = build_grouped_tensor(C_views, NVTE_DELAYED_TENSOR_SCALING);
  }
  GroupedBuffers grouped_D = build_grouped_tensor(D_views, NVTE_DELAYED_TENSOR_SCALING);

  // Per-matrix alpha/beta (all 1.0 and 0.0 respectively)
  Tensor alpha_tensor("alpha", std::vector<size_t>{num_gemms}, DType::kFloat32);
  Tensor beta_tensor("beta", std::vector<size_t>{num_gemms}, DType::kFloat32);
  std::vector<float> alpha_vals(num_gemms, 1.f);
  std::vector<float> beta_vals(num_gemms, 0.f);
  NVTE_CHECK_CUDA(cudaMemcpy(alpha_tensor.rowwise_dptr(), alpha_vals.data(),
                             num_gemms * sizeof(float), cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(beta_tensor.rowwise_dptr(), beta_vals.data(),
                             num_gemms * sizeof(float), cudaMemcpyHostToDevice));

  const size_t setup_ws_bytes = grouped_setup_workspace_size(num_gemms);
  Tensor setup_ws("setup_ws", std::vector<size_t>{setup_ws_bytes}, DType::kByte);
  Tensor cublas_ws("cublas_ws", std::vector<size_t>{cublas_ws_bytes}, DType::kByte);

  // Create config with use_split_accumulator setting
  transformer_engine::GroupedMatmulConfigWrapper config;
  config.set_use_split_accumulator(params.use_split_accumulator);

  nvte_grouped_gemm(params.transa,
                    params.transb,
                    alpha_tensor.data(),
                    grouped_A.get_handle(),
                    grouped_B.get_handle(),
                    beta_tensor.data(),
                    params.use_null_c ? nullptr : grouped_C->get_handle(),
                    grouped_D.get_handle(),
                    setup_ws.data(),
                    cublas_ws.data(),
                    config,
                    0);

  for (size_t i = 0; i < num_gemms; ++i) {
    Tensor grouped_split("grouped_D" + std::to_string(i),
                         std::vector<size_t>{static_cast<size_t>(std::get<0>(shapes[i])),
                                             static_cast<size_t>(std::get<1>(shapes[i]))},
                         D_multi[i].dtype());
    const size_t offset_bytes = static_cast<size_t>(grouped_D.offsets_host[i]) * grouped_D.elem_size;
    NVTE_CHECK_CUDA(cudaMemcpy(grouped_split.rowwise_dptr(),
                               static_cast<char*>(grouped_D.get_data()) + offset_bytes,
                               grouped_D.tensor_bytes[i],
                               cudaMemcpyDeviceToDevice));
    grouped_split.to_cpu();
    D_multi[i].to_cpu();
    auto [atol, rtol] = getTolerances(D_multi[i].dtype());
    compareResults("grouped_vs_multi",
                   grouped_split,
                   D_multi[i].rowwise_cpu_dptr<bf16>(),
                   true,
                   atol,
                   rtol);
  }
#endif  // CUBLAS_VERSION >= 130100
}

class GroupedGemmTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(GroupedGemmTest, CompareWithMultiTensorGemm) {
  run_grouped_gemm_case(GetParam());
}

std::string MakeGroupedGemmTestName(const testing::TestParamInfo<GroupedGemmTest::ParamType>& info) {
  constexpr const char* kInputNames[] = {"FP8Current", "BF16"};
  constexpr const char* kShapeNames[] = {"AllSame", "SameM", "SameN", "AllDiff"};
  const std::string layout = std::string("ta") + (info.param.transa ? "T" : "N") +
                             "tb" + (info.param.transb ? "T" : "N");
  const std::string null_c = info.param.use_null_c ? "_NullC" : "";
  const std::string split_acc = info.param.use_split_accumulator ? "_SplitAcc" : "";
  return std::string(kInputNames[static_cast<int>(info.param.input_case)]) + "_" +
         kShapeNames[static_cast<int>(info.param.shape_case)] + "_" + layout + null_c + split_acc;
}

// TestParams: {input_case, transa, transb, shape_case, use_null_c, use_split_accumulator}
const std::vector<TestParams> kTestParams = {
    // Basic tests (no split accumulator)
    {InputCase::kFP8Current, true, false, ShapeCase::kAllDifferent, false, false},
    {InputCase::kFP8Current, false, true, ShapeCase::kAllDifferent, false, false},
    {InputCase::kFP8Current, false, false, ShapeCase::kAllSame, false, false},
    {InputCase::kBF16, true, false, ShapeCase::kSameFirst, false, false},
    {InputCase::kBF16, false, true, ShapeCase::kSameLast, false, false},
    {InputCase::kBF16, false, false, ShapeCase::kAllSame, false, false},
    {InputCase::kBF16, true, true, ShapeCase::kAllDifferent, false, false},
    // Test NULL C (valid when beta=0)
    {InputCase::kBF16, false, false, ShapeCase::kAllSame, true, false},

    // Split accumulator tests
    {InputCase::kFP8Current, true, false, ShapeCase::kAllDifferent, false, true},
    {InputCase::kFP8Current, false, true, ShapeCase::kAllDifferent, false, true},
    {InputCase::kFP8Current, false, false, ShapeCase::kAllSame, false, true},
    {InputCase::kFP8Current, true, false, ShapeCase::kSameFirst, false, true},
};

INSTANTIATE_TEST_SUITE_P(OperatorTest,
                         GroupedGemmTest,
                         ::testing::ValuesIn(kTestParams),
                         MakeGroupedGemmTestName);

}  // namespace
