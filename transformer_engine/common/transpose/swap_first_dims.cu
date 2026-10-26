/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/transpose.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/logging.h"
#include "../util/rtc.h"
#include "../util/string.h"

namespace transformer_engine {

namespace {

// Bounded CP4 copy. Constant 32-bit indexing and a capped grid avoid the
// per-element 64-bit divisions and oversized grid of the generic transpose.
constexpr uint32_t kTokens = 131072;
constexpr uint32_t kCp = 4;

template <uint32_t VectorsPerHeadGroup, bool Unpack>
__global__ void cp_a2a_pack16_kernel(const uint4 *__restrict__ src, uint4 *__restrict__ dst) {
  constexpr uint32_t kRankVectors = kTokens * VectorsPerHeadGroup;
  constexpr uint32_t kTotalVectors = kCp * kRankVectors;
  for (uint32_t out = blockIdx.x * blockDim.x + threadIdx.x; out < kTotalVectors;
       out += gridDim.x * blockDim.x) {
    uint32_t in;
    if constexpr (Unpack) {
      const uint32_t token = out / (kCp * VectorsPerHeadGroup);
      const uint32_t rank = (out / VectorsPerHeadGroup) % kCp;
      in = rank * kRankVectors + token * VectorsPerHeadGroup + out % VectorsPerHeadGroup;
    } else {
      const uint32_t rank = out / kRankVectors;
      const uint32_t token = (out % kRankVectors) / VectorsPerHeadGroup;
      in = token * kCp * VectorsPerHeadGroup + rank * VectorsPerHeadGroup +
           out % VectorsPerHeadGroup;
    }
    dst[out] = src[in];
  }
}

template <uint32_t VectorsPerHeadGroup>
void launch_cp_a2a_pack16(const void *src, void *dst, int unpack, uint32_t blocks,
                          cudaStream_t stream) {
  if (unpack) {
    cp_a2a_pack16_kernel<VectorsPerHeadGroup, true>
        <<<blocks, 256, 0, stream>>>(static_cast<const uint4 *>(src), static_cast<uint4 *>(dst));
  } else {
    cp_a2a_pack16_kernel<VectorsPerHeadGroup, false>
        <<<blocks, 256, 0, stream>>>(static_cast<const uint4 *>(src), static_cast<uint4 *>(dst));
  }
}

// String with RTC kernel implementation
#include "string_code_transpose_rtc_swap_first_dims_cu.h"

// Hard-coded kernel parameters
constexpr size_t block_size = 128;

/* Performance heuristics for optimized kernel parameters */
struct KernelConfig {
  /* Vector load/store size */
  size_t vector_size;

  /* Whether config is valid */
  bool valid = false;
  /* Number of CUDA blocks */
  size_t num_blocks = 0;
  /* Whether each thread needs to make exactly one load/store */
  bool single_load_store = true;

  /* Number of active SMs */
  size_t active_sm_count = 0;
  /* Used bytes per L1 cache load */
  size_t bytes_per_load = 0;
  /* Used bytes per L1 cache store */
  size_t bytes_per_store = 0;

  KernelConfig(size_t dim0, size_t dim1, size_t dim2, size_t sm_count, size_t vector_size_)
      : vector_size{vector_size_} {
    // Check that tiles are correctly aligned
    if (dim2 % vector_size_ != 0) {
      return;
    }
    valid = true;

    // Number of CUDA blocks
    num_blocks = DIVUP(dim0 * dim1 * dim2 / vector_size, block_size);
    if (num_blocks > 2147483647ull) {
      // Maximum number of CUDA blocks
      single_load_store = false;
      num_blocks = 2147483647ull;
    } else if (num_blocks * block_size != dim0 * dim1 * dim2 / vector_size) {
      single_load_store = false;
    }

    // SM occupancy
    constexpr size_t warp_size = 32;
    constexpr size_t warps_per_sm = 16;  // Rough estimate for saturated SMs
    active_sm_count = std::min(DIVUP(num_blocks * block_size / warp_size, warps_per_sm), sm_count);

    // L1 cache efficiency
    constexpr size_t cache_line_size = 128;
    bytes_per_store = std::min(cache_line_size, warp_size * vector_size);  // Contiguous writes
    bytes_per_load = bytes_per_store;
    if (dim2 % (vector_size * warp_size) != 0) {
      // Some warps are reading from two non-contiguous regions
      bytes_per_load /= 2;
    }
  }

  /* Compare by estimated cost */
  bool operator<(const KernelConfig &other) const {
    if (this->valid && other.valid) {
      // cost ~ (1/bytes_per_load + 1/bytes_per_store) / active_sms
      // Note: Integer arithmetic ensures stable ordering
      const auto &l1 = this->bytes_per_load;
      const auto &s1 = this->bytes_per_store;
      const auto &p1 = this->active_sm_count;
      const auto &l2 = other.bytes_per_load;
      const auto &s2 = other.bytes_per_store;
      const auto &p2 = other.active_sm_count;
      const auto scale = l1 * s1 * p1 * l2 * s2 * p2;
      const auto cost1 = (scale / l1 + scale / s1) / p1;
      const auto cost2 = (scale / l2 + scale / s2) / p2;
      return cost1 < cost2;
    } else {
      return this->valid && !other.valid;
    }
  }
};

template <typename Type>
__global__ void __launch_bounds__(block_size)
    swap_first_dims_untuned_kernel(const Type *__restrict__ input, Type *__restrict__ output,
                                   const size_t dim0, const size_t dim1, const size_t dim2) {
  const size_t gid = threadIdx.x + blockIdx.x * block_size;
  const size_t nthreads = gridDim.x * block_size;
  for (size_t idx = gid; idx < dim0 * dim1 * dim2; idx += nthreads) {
    const auto idx2 = idx % dim2;
    const auto idx1 = (idx / dim2) % dim1;
    const auto idx0 = (idx / dim2) / dim1;
    const auto in_offset = idx1 * dim0 * dim2 + idx0 * dim2 + idx2;
    output[idx] = input[in_offset];
  }
}

}  // namespace

void swap_first_dims(const Tensor &input, Tensor &output, cudaStream_t stream) {
  // Check tensors
  NVTE_CHECK(input.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be simple tensor, but scaling mode is ",
             to_string(input.scaling_mode), ".");
  NVTE_CHECK(output.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Output tensor must be simple tensor, but scaling mode is ",
             to_string(output.scaling_mode), ".");
  NVTE_CHECK(input.dtype() == output.dtype(), "Input tensor (dtype=", to_string(input.dtype()),
             ") and output tensor (dtype=", to_string(output.dtype()), ") do not match.");
  NVTE_CHECK(input.data.dptr != nullptr, "Input is not allocated.");
  NVTE_CHECK(output.data.dptr != nullptr, "Output is not allocated.");

  // Check tensor dimensions
  const auto input_shape = input.shape();
  const auto output_shape = output.shape();
  NVTE_CHECK(input_shape.size() >= 2, "Invalid input tensor dimensions (shape=", input_shape, ").");
  NVTE_CHECK(output_shape.size() == input_shape.size(), "Input tensor (shape=", input_shape,
             ") and output tensor (shape=", output_shape, ") do not match.");
  NVTE_CHECK(input_shape[0] == output_shape[1], "Input tensor (shape=", input_shape,
             ") and output tensor (shape=", output_shape, ") do not match.");
  NVTE_CHECK(input_shape[1] == output_shape[0], "Input tensor (shape=", input_shape,
             ") and output tensor (shape=", output_shape, ") do not match.");
  for (size_t i = 2; i < input_shape.size(); ++i) {
    NVTE_CHECK(input_shape[i] == output_shape[i], "Input tensor (shape=", input_shape,
               ") and output tensor (shape=", output_shape, ") do not match.");
  }

  // Reinterpret tensors as 3D tensors of bytes
  const size_t dim0 = output_shape[0];
  const size_t dim1 = output_shape[1];
  size_t dim2 = 1;
  for (size_t i = 2; i < output_shape.size(); ++i) {
    dim2 *= output_shape[i];
  }
  dim2 = get_buffer_size_bytes(dim2, output.dtype());

  // The CP opt-in applies to both attention backends through this existing API.
  // Keep every other shape, dtype, architecture and pointer layout on the generic path.
  const char *cp_pack16 = std::getenv("NVTE_CP_EXPERIMENTAL_A2A_PACK16");
  const uintptr_t src = reinterpret_cast<uintptr_t>(input.data.dptr);
  const uintptr_t dst = reinterpret_cast<uintptr_t>(output.data.dptr);
  const bool cp_shape = input_shape.size() == 4 &&
                        ((dim0 == kCp && dim1 == kTokens) || (dim0 == kTokens && dim1 == kCp)) &&
                        (input_shape[2] == 2 || input_shape[2] == 8) &&
                        (input_shape[3] == 128 || input_shape[3] == 256);
  if (cp_pack16 != nullptr && std::string(cp_pack16) == "1" && cp_shape &&
      input.dtype() == DType::kBFloat16 && (src | dst) % 16 == 0 && cuda::sm_arch() == 100) {
    const size_t bytes = dim0 * dim1 * dim2;
    if (src <= UINTPTR_MAX - bytes && dst <= UINTPTR_MAX - bytes &&
        (src + bytes <= dst || dst + bytes <= src)) {
      constexpr uint32_t blocks = 16384;
      const bool unpack = dim0 == kTokens;
      switch (dim2) {
        case 512:
          launch_cp_a2a_pack16<32>(input.data.dptr, output.data.dptr, unpack, blocks, stream);
          break;
        case 1024:
          launch_cp_a2a_pack16<64>(input.data.dptr, output.data.dptr, unpack, blocks, stream);
          break;
        case 2048:
          launch_cp_a2a_pack16<128>(input.data.dptr, output.data.dptr, unpack, blocks, stream);
          break;
        case 4096:
          launch_cp_a2a_pack16<256>(input.data.dptr, output.data.dptr, unpack, blocks, stream);
          break;
        default:
          NVTE_ERROR("Unexpected bounded CP copy width: ", dim2);
      }
      NVTE_CHECK_CUDA(cudaGetLastError());
      return;
    }
  }

  // Choose kernel config with performance heuristics
  const size_t sm_count = static_cast<size_t>(cuda::sm_count());
  KernelConfig config(dim0, dim1, dim2, sm_count, 1);
  if (rtc::is_enabled()) {
    auto try_config = [&](size_t vector_size) {
      KernelConfig new_config(dim0, dim1, dim2, sm_count, vector_size);
      if (new_config < config) {
        config = new_config;
      }
    };
    try_config(16);
    try_config(8);
    try_config(4);
    try_config(2);
  }
  const size_t vector_size = config.vector_size;

  // Launch kernel
  if (vector_size == 1) {
    // General kernel
    swap_first_dims_untuned_kernel<<<config.num_blocks, block_size, 0, stream>>>(
        static_cast<const uint8_t *>(input.data.dptr), static_cast<uint8_t *>(output.data.dptr),
        dim0, dim1, dim2);
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else {
    // Compile NVRTC kernel if needed
    auto &rtc_manager = rtc::KernelManager::instance();
    const std::string kernel_label =
        concat_strings("swap_first_dims,vector_size=", vector_size,
                       ",single_load_store=", config.single_load_store);
    if (!rtc_manager.is_compiled(kernel_label)) {
      std::string code = string_code_transpose_rtc_swap_first_dims_cu;
      code = regex_replace(code, "__VECTOR_SIZE__", vector_size);
      code = regex_replace(code, "__BLOCK_SIZE__", block_size);
      code =
          regex_replace(code, "__SINGLE_LOAD_STORE__", static_cast<int>(config.single_load_store));
      rtc_manager.compile(kernel_label, "swap_first_dims_kernel", code,
                          "transformer_engine/common/transpose/rtc/swap_first_dims.cu");
    }

    // Launch NVRTC kernel
    rtc_manager.launch(kernel_label, config.num_blocks, block_size, 0, stream, input.data.dptr,
                       output.data.dptr, dim0, dim1, dim2 / vector_size);
  }
}

}  // namespace transformer_engine

void nvte_swap_first_dims(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_swap_first_dims);
  using namespace transformer_engine;
  swap_first_dims(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output), stream);
}
