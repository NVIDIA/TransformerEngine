/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/recipe.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "../util/logging.h"
#include "../util/vectorized_pointwise.h"
#include "common.h"
#include "pybind.h"
#include "recipe_common.cuh"
#include "torch/torch.h"

namespace transformer_engine {
namespace {

constexpr int amax_kernel_threads = 512;

__launch_bounds__(1) __global__ void zero_amax_kernel(float *amax_ptr, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }
  *amax_ptr = 0;
}

// --- 核心 Kernel 修改 ---

template <int nvec, bool aligned, typename InputType>
__launch_bounds__(amax_kernel_threads) __global__
    void amax_kernel(const InputType *input_a, const InputType *input_b, const InputType *input_c,
                     InputType *output,  // [新增] 输出结果指针
                     float *amax, const size_t N, const size_t num_aligned_elements,
                     const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  // 定义 Shared Memory
  // 大小计算: 512线程 * nvec元素/线程 * 3个数组
  // 注意：对于 heavy usage，这可能会占用较多 SMEM，需确保 Occupancy
  __shared__ InputType s_a[amax_kernel_threads * nvec];
  __shared__ InputType s_b[amax_kernel_threads * nvec];
  __shared__ InputType s_c[amax_kernel_threads * nvec];

  InputType max = 0.f;
  const int warp_id = threadIdx.x / 32;

  // 加载器：这里我们仍然用 loader 来处理全局内存的对齐读取，
  // 但我们不再读到私有寄存器处理，而是先写入 Shared Memory。
  VectorizedLoader<InputType, nvec, aligned> loader_a(input_a, N);
  VectorizedLoader<InputType, nvec, aligned> loader_b(input_b, N);
  VectorizedLoader<InputType, nvec, aligned> loader_c(input_c, N);

  // 输出写回也需要向量化
  // 简单起见，我们假设 output 和 input 对齐情况一致，重用 loader 逻辑进行偏移计算
  // 实际写回时我们会手动计算指针

  const size_t M = num_aligned_elements;

  // Grid-Stride Loop
  for (size_t tid_base = blockIdx.x * blockDim.x; tid_base < M;
       tid_base += gridDim.x * blockDim.x) {
    const size_t tid = tid_base + threadIdx.x;

    // --- Phase 1: Global -> Shared (协作加载) ---
    // 即使 tid >= N (越界)，loader 内部通常有处理，或者我们需要在此处保护
    if (tid < N / nvec) {  // 粗略的边界检查，假设 N 是 nvec 的倍数或 loader 处理了边界
      loader_a.load(tid, N);
      loader_b.load(tid, N);
      loader_c.load(tid, N);

// 将寄存器中的数据转存到 Shared Memory
// 这实现了 "Buffered" 读取
#pragma unroll
      for (int i = 0; i < nvec; ++i) {
        int smem_idx = threadIdx.x * nvec + i;
        s_a[smem_idx] = loader_a.separate()[i];
        s_b[smem_idx] = loader_b.separate()[i];
        s_c[smem_idx] = loader_c.separate()[i];
      }
    }

    // 必须同步，确保整个 Tile 加载完成
    __syncthreads();

    // --- Phase 2: Compute from Shared -> Write Global & Update Max ---

    // 我们需要再次向量化写回 Output，为了性能，我们构建一个临时数组存结果
    InputType local_results[nvec];

    if (tid < N / nvec) {  // 同样的边界保护
#pragma unroll
      for (int i = 0; i < nvec; ++i) {
        int smem_idx = threadIdx.x * nvec + i;

        // 从 Shared Memory 读取
        InputType val_a = s_a[smem_idx];
        InputType val_b = s_b[smem_idx];
        InputType val_c = s_c[smem_idx];

        // 计算 A + B - C
        // 转为 float 计算以防溢出
        float res_f =
            static_cast<float>(val_a) + static_cast<float>(val_b) - static_cast<float>(val_c);
        InputType res = static_cast<InputType>(res_f);

        // 存入本地寄存器数组，准备向量化写回
        local_results[i] = res;

        // 计算 Max (Amax逻辑)
        if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
          max = __hmax(__habs(res), max);
#else
          max = static_cast<__nv_bfloat16>(
              fmaxf(fabsf(static_cast<float>(res)), static_cast<float>(max)));
#endif
        } else if constexpr (std::is_same_v<InputType, __half>) {
          max = __hmax(__habs(res), max);
        } else {
          max = fmaxf(fabsf(res), max);
        }
      }

      // --- Phase 3: Write Back to Global Output ---
      // 这里手动实现向量化写回，假设 output 指针支持向量化访问
      // 使用 reinterpret_cast 将 InputType* 强转为对应的向量类型 (如 float4, int4 等)
      // 注意：真正的生产代码需要更严谨的 Vector Store 封装
      // using VecType = typename VectorizedLoader<InputType, nvec, aligned>::VecType;
      // 假设 VectorizedLoader 内部定义了 VecType，或者我们可以根据 nvec 推导

      // 简化的写回逻辑 (模拟 Vectorized Store):
      InputType *output_vec_ptr = reinterpret_cast<InputType *>(output);
      InputType vec_data;
      // 将 local_results 数据塞入 vec_data (这里依赖内存布局，通常 memcpy 或 union)
      memcpy(&vec_data, local_results, sizeof(InputType));

      output_vec_ptr[tid] = vec_data;  // tid 实际上是向量索引
    }

    // 同步，防止下一轮循环覆盖当前未被读取的 Shared Memory
    __syncthreads();
  }

  // Reduce amax over block
  max = reduce_max<amax_kernel_threads / 32>(max, warp_id);
  if (threadIdx.x == 0) {
    atomicMaxFloat(amax, max);
  }
}

template <int nvec, bool aligned, typename InputType>
__launch_bounds__(amax_kernel_threads) __global__
    void amax_kernel(const InputType *input, float *amax, const size_t N,
                     const size_t num_aligned_elements, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  VectorizedLoader<InputType, nvec, aligned> loader(input, N);
  InputType max = 0.f;
  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const size_t M = num_aligned_elements;

  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < M; tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const InputType val = static_cast<InputType>(loader.separate()[i]);
      __builtin_assume(max >= InputType{0.f});
      if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
        max = __hmax(__habs(val), max);
#else  // Turing
        max = static_cast<__nv_bfloat16>(
            fmaxf(fabsf(static_cast<float>(val)), static_cast<float>(max)));
#endif
      } else if constexpr (std::is_same_v<InputType, __half>) {
        max = __hmax(__habs(val), max);
      } else {
        max = fmaxf(fabsf(val), max);
      }
    }
  }

  // Reduce amax over block
  max = reduce_max<amax_kernel_threads / THREADS_PER_WARP>(max, warp_id);
  if (threadIdx.x == 0) {
    atomicMaxFloat(amax, max);
  }
}

template <int nvec, typename InputType>
void launch_amax_kernel(const InputType *input, float *amax, const size_t N, const float *noop_ptr,
                        cudaStream_t stream) {
  // Zero out amax so we can update with atomic max
  zero_amax_kernel<<<1, 1, 0, stream>>>(amax, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

  // Return immediately if tensor is empty
  if (N == 0) {
    return;
  }

  // Figure out alignment
  auto align = CheckAlignment(N, nvec, input);
  size_t num_aligned_elements = get_num_aligned_elements(input, N, nvec, sizeof(InputType));

  // Figure out CUDA blocks
  constexpr size_t threads = amax_kernel_threads;
  size_t num_blocks = DIVUP(num_aligned_elements, threads);
  constexpr size_t max_blocks = 65535;
  num_blocks = std::min(num_blocks, max_blocks);

  // Launch kernel
  switch (align) {
    case Alignment::SAME_ALIGNED:
      amax_kernel<nvec, true, InputType>
          <<<num_blocks, threads, 0, stream>>>(input, amax, N, num_aligned_elements, noop_ptr);
      break;
    case Alignment::SAME_UNALIGNED:
      amax_kernel<nvec, false, InputType>
          <<<num_blocks, threads, 0, stream>>>(input, amax, N, num_aligned_elements, noop_ptr);
      break;
    case Alignment::DIFFERENT: {
      // This case is a logic error, since there is only one pointer (input)
      // in the alignment check. Still safe to process without vectorization.
      amax_kernel<1, true, InputType>
          <<<num_blocks, threads, 0, stream>>>(input, amax, N, N, noop_ptr);
      break;
    }
  }

  // Check results
  NVTE_CHECK_CUDA(cudaGetLastError());
}
template <int nvec, typename InputType>
void launch_amax_kernel(const InputType *input0, const InputType *input1, const InputType *input2,
                        InputType *output, float *amax, const size_t N, const float *noop_ptr,
                        cudaStream_t stream) {
  // Zero out amax so we can update with atomic max
  zero_amax_kernel<<<1, 1, 0, stream>>>(amax, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

  // Return immediately if tensor is empty
  if (N == 0) {
    return;
  }

  // Figure out alignment
  auto align = CheckAlignment(N, nvec, input0);
  size_t num_aligned_elements = get_num_aligned_elements(input0, N, nvec, sizeof(InputType));

  // Figure out CUDA blocks
  constexpr size_t threads = amax_kernel_threads;
  size_t num_blocks = DIVUP(num_aligned_elements, threads);
  constexpr size_t max_blocks = 65535;
  num_blocks = std::min(num_blocks, max_blocks);

  // Launch kernel
  switch (align) {
    case Alignment::SAME_ALIGNED:
      amax_kernel<nvec, true, InputType><<<num_blocks, threads, 0, stream>>>(
          input0, input1, input2, output, amax, N, num_aligned_elements, noop_ptr);
      break;
    case Alignment::SAME_UNALIGNED:
      amax_kernel<nvec, false, InputType><<<num_blocks, threads, 0, stream>>>(
          input0, input1, input2, output, amax, N, num_aligned_elements, noop_ptr);
      break;
    case Alignment::DIFFERENT: {
      // This case is a logic error, since there is only one pointer (input)
      // in the alignment check. Still safe to process without vectorization.
      amax_kernel<1, true, InputType><<<num_blocks, threads, 0, stream>>>(
          input0, input1, input2, output, amax, N, N, noop_ptr);
      break;
    }
  }

  // Check results
  NVTE_CHECK_CUDA(cudaGetLastError());
}
}  // namespace
}  // namespace transformer_engine

namespace {

void compute_amax_impl(const NVTETensor input_, const NVTETensor output_, cudaStream_t stream,
                       const NVTEQuantizationConfig config_) {
  using namespace transformer_engine;

  // Check input tensor
  NVTE_CHECK(input_ != nullptr, "Invalid input tensor (got NULL)");
  const auto &input = *convertNVTETensorCheck(input_);
  NVTE_CHECK(input.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor for amax computation must unquantized, "
             "but got scaling_mode=",
             to_string(input.scaling_mode));
  NVTE_CHECK(!is_fp8_dtype(input.data.dtype),
             "Input tensor for amax computation must be unquantized, but got dtype=",
             to_string(input.data.dtype));
  NVTE_CHECK(input.data.dptr != nullptr, "Input tensor for amax computation has no data");
  CheckInputTensor(input, "input_compute_amax");

  // Check output tensor
  NVTE_CHECK(output_ != nullptr, "Invalid output tensor (got NULL)");
  auto &output = *convertNVTETensorCheck(output_);
  NVTE_CHECK(output.scaling_mode == NVTE_DELAYED_TENSOR_SCALING ||
                 output.scaling_mode == NVTE_NVFP4_1D_SCALING,
             "Output tensor for amax computation must be FP8 tensor with per-tensor scaling or "
             "NVFP4 1D scaling, "
             "but got scaling_mode=",
             to_string(output.scaling_mode));
  NVTE_CHECK(output.amax.numel() == 1,
             "Output tensor for amax computation has invalid amax tensor "
             "(expected 1 entry, got shape=",
             output.amax.shape, ")");
  NVTE_CHECK(output.amax.dptr != nullptr || output.columnwise_amax.dptr != nullptr,
             "Output tensor for amax computation has amax tensor without data");
  NVTE_CHECK(output.amax.dtype == DType::kFloat32,
             "Output tensor for amax computation has invalid amax tensor  "
             "(expected FP32, got dtype=",
             to_string(output.amax.dtype), ")");
  CheckOutputTensor(output, "output_compute_amax", true);

  float *noop_ptr = nullptr;
  if (config_ != nullptr) {
    const QuantizationConfig *config_cpp = reinterpret_cast<const QuantizationConfig *>(config_);

    // extract noop tensor from quant_config_cpp if it's not null
    const NVTETensor noop = config_cpp ? config_cpp->noop_tensor : nullptr;
    noop_ptr = reinterpret_cast<float *>(
        (noop != nullptr ? convertNVTETensorCheck(noop)->data.dptr : nullptr));
  }

  // Compute amax
  float *amax_ptr = reinterpret_cast<float *>(
      (output.amax.dptr != nullptr) ? output.amax.dptr : output.columnwise_amax.dptr);
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType, constexpr int nvec = 32 / sizeof(IType); launch_amax_kernel<nvec>(
          reinterpret_cast<const IType *>(input.data.dptr), amax_ptr, input.data.numel(), noop_ptr,
          stream););  // NOLINT(*)
}

}  // anonymous namespace

void nvte_compute_amax(const NVTETensor input_, const NVTETensor output_, cudaStream_t stream) {
  NVTE_API_CALL(nvte_compute_amax);
  compute_amax_impl(input_, output_, stream, nullptr);
}

void nvte_compute_amax_with_config(const NVTETensor input_, const NVTETensor output_,
                                   const NVTEQuantizationConfig config_, cudaStream_t stream) {
  NVTE_API_CALL(nvte_compute_amax_with_config);
  compute_amax_impl(input_, output_, stream, config_);
}

namespace transformer_engine {
namespace {

__global__ void compute_scale_from_amax_kernel(const float *amax_ptr, float *scale_ptr,
                                               const float max_fp8, const bool force_pow_2_scales,
                                               const float epsilon, const float *noop_ptr) {
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  *scale_ptr = compute_scale_from_amax(*amax_ptr, max_fp8, force_pow_2_scales, epsilon,
                                       std::numeric_limits<float>::max());
}

}  // namespace
}  // namespace transformer_engine

void nvte_compute_scale_from_amax(NVTETensor output_, const NVTEQuantizationConfig config_,
                                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_compute_scale_from_amax);
  using namespace transformer_engine;

  // Check output tensor
  NVTE_CHECK(output_ != nullptr, "Invalid output tensor (got NULL)");
  auto &output = *convertNVTETensorCheck(output_);
  NVTE_CHECK(output.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Tensor must be FP8 tensor with per-tensor scaling, "
             "but got scaling_mode=",
             to_string(output.scaling_mode));
  NVTE_CHECK(is_fp8_dtype(output.data.dtype),
             "Tensor must be FP8, but got dtype=", to_string(output.data.dtype));
  NVTE_CHECK(output.amax.numel() == 1,
             "Tensor has invalid amax tensor (expected 1 entry, got shape=", output.amax.shape,
             ")");
  NVTE_CHECK(output.amax.dptr != nullptr, "Tensor has amax tensor without data");
  NVTE_CHECK(output.amax.dtype == DType::kFloat32,
             "Tensor has invalid amax tensor (expected FP32, got dtype=",
             to_string(output.amax.dtype), ")");
  NVTE_CHECK(output.scale.numel() == 1,
             "Tensor has invalid scale tensor (expected 1 entry, got shape=", output.scale.shape,
             ")");
  NVTE_CHECK(output.scale.dptr != nullptr, "Tensor has scale tensor without data");
  NVTE_CHECK(output.scale.dtype == DType::kFloat32,
             "Tensor has invalid scale tensor (expected FP32, got dtype=",
             to_string(output.scale.dtype), ")");

  // Check config
  NVTE_CHECK(config_ != nullptr, "Invalid config (got NULL)");
  const auto &config = *reinterpret_cast<const QuantizationConfig *>(config_);

  // Maximum FP8 value
  float max_fp8 = 0.f;
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output.data.dtype, DType,
                                         max_fp8 = Quantized_Limits<DType>::max_norm;);

  // noop tensor for cuda graph
  float *noop_ptr = nullptr;
  if (config_ != nullptr) {
    const QuantizationConfig *config_cpp = reinterpret_cast<const QuantizationConfig *>(config_);

    // extract noop tensor from quant_config_cpp if it's not null
    const NVTETensor noop = config_cpp ? config_cpp->noop_tensor : nullptr;
    noop_ptr = reinterpret_cast<float *>(
        (noop != nullptr ? convertNVTETensorCheck(noop)->data.dptr : nullptr));
  }

  // Update scale
  compute_scale_from_amax_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<const float *>(output.amax.dptr),
      reinterpret_cast<float *>(output.scale.dptr), max_fp8, config.force_pow_2_scales,
      config.amax_epsilon, noop_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void verify_results(float *h_a, float *h_b, float *h_c, float *h_out, float gpu_amax, int N) {
  float cpu_max = 0.0f;
  float max_error = 0.0f;
  const float epsilon = 1e-5f;

  for (int i = 0; i < N; ++i) {
    float expected = h_a[i] + h_b[i] - h_c[i];
    float diff = std::abs(expected - h_out[i]);
    max_error = std::max(max_error, diff);

    cpu_max = std::max(cpu_max, std::abs(expected));

    if (diff > epsilon && i < 5) {  // 仅打印前5个错误
      std::cout << "Mismatch at index " << i << ": "
                << "Expected " << expected << ", Got " << h_out[i] << "\n";
    }
  }

  std::cout << "--- Verification Results ---" << std::endl;
  std::cout << "Max Output Error: " << max_error << (max_error < epsilon ? " [PASS]" : " [FAIL]")
            << std::endl;
  std::cout << "CPU Amax: " << cpu_max << std::endl;
  std::cout << "GPU Amax: " << gpu_amax << std::endl;

  if (std::abs(cpu_max - gpu_amax) < epsilon) {
    std::cout << "Amax Check: [PASS]" << std::endl;
  } else {
    std::cout << "Amax Check: [FAIL]" << std::endl;
  }
}

using bf16 = nv_bfloat16;
int main() {
  // 参数设置
  const int N = 1024 * 1024;  // 1M 元素
  // const int nvec = 32 / sizeof(float);        // bfloat16
  const size_t bytes = N * sizeof(float);

  // 1. Host 内存分配与初始化
  std::vector<float> h_a(N), h_b(N), h_c(N), h_out(N);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  for (int i = 0; i < N; ++i) {
    h_a[i] = dist(gen);
    h_b[i] = dist(gen);
    h_c[i] = dist(gen);
  }

  // 2. Device 内存分配
  float *d_a, *d_b, *d_c, *d_out, *d_amax, *d_noop;
  NVTE_CHECK_CUDA(cudaMalloc(&d_a, bytes));
  NVTE_CHECK_CUDA(cudaMalloc(&d_b, bytes));
  NVTE_CHECK_CUDA(cudaMalloc(&d_c, bytes));
  NVTE_CHECK_CUDA(cudaMalloc(&d_out, bytes));
  NVTE_CHECK_CUDA(cudaMalloc(&d_amax, sizeof(float)));
  NVTE_CHECK_CUDA(cudaMalloc(&d_noop, sizeof(float)));  // 可选

  // 3. 数据拷贝 H2D
  NVTE_CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice));
  auto stream = at::cuda::getCurrentCUDAStream();
  // 初始化 noop 和 amax
  float zero = 0.0f;
  auto t = transformer_engine::DType::kFloat32;  // float32
  NVTE_CHECK_CUDA(cudaMemcpy(d_noop, &zero, sizeof(float), cudaMemcpyHostToDevice));
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      t, IType, constexpr int nvec = 32 / sizeof(IType); launch_amax_kernel<nvec>(
          reinterpret_cast<const IType *>(d_a), reinterpret_cast<const IType *>(d_b),
          reinterpret_cast<const IType *>(d_c), reinterpret_cast<IType *>(d_out), d_amax, N,
          nullptr,
          stream););  // NOLINT(*)

  // 4. 执行 Kernel
  // Block Size = 512
  // 每个线程处理 nvec (4) 个元素
  // 总共有 N / 4 个向量任务
  // size_t num_aligned_elements = N / nvec;
  // int grid_size = 256; // 设定 Grid 大小，依靠 Grid-Stride Loop 处理全部数据

  // std::cout << "Launching kernel with Grid=" << grid_size
  //           << ", Block=" << amax_kernel_threads
  //           << ", N=" << N << std::endl;

  // // amax_kernel<nvec, true, float><<<grid_size, amax_kernel_threads>>>(
  // //     d_a, d_b, d_c, d_out, d_amax,
  // //     N, num_aligned_elements, d_noop
  // // );
  NVTE_CHECK_CUDA(cudaGetLastError());
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  // 5. 结果拷贝 D2H
  float gpu_amax;
  NVTE_CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(&gpu_amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));

  // 6. 验证
  verify_results(h_a.data(), h_b.data(), h_c.data(), h_out.data(), gpu_amax, N);

  // 7. 清理
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_out);
  cudaFree(d_amax);
  cudaFree(d_noop);

  return 0;
}
