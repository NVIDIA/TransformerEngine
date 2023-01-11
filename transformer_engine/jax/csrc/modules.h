/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "jax/csrc/utils.h"
#include "transformer_engine/logging.h"

namespace transformer_engine {
namespace jax {

struct TEMatDescriptor {
  std::uint64_t m;
  std::uint64_t n;
  std::uint64_t in_ctype;
  std::uint64_t out_ctype;
};

pybind11::bytes PackTEMatDescriptor(std::uint64_t m, std::uint64_t n,
                                    std::uint64_t fwd_dtype,
                                    std::uint64_t bwd_dtype);

struct TEGemmDescriptor {
  std::uint64_t m;
  std::uint64_t n;
  std::uint64_t k;
  std::uint64_t A_ctype;
  std::uint64_t B_ctype;
  std::uint64_t D_ctype;
  bool transa;
  bool transb;
  bool use_split_accumulator;
};

pybind11::bytes PackTEGemmDescriptor(std::uint64_t m, std::uint64_t n,
                                     std::uint64_t k, std::uint64_t A_ctype,
                                     std::uint64_t B_ctype,
                                     std::uint64_t D_ctype, bool transa,
                                     bool transb, bool use_split_accumulator);

struct LayerNormDescriptor {
  std::uint64_t n;
  std::uint64_t hidden;
  std::uint64_t x_dtype;
  std::uint64_t w_dtype;
  float eps;
};

pybind11::bytes PackLayerNormDescriptor(std::uint64_t n, std::uint64_t hidden,
                                        std::uint64_t x_dtype,
                                        std::uint64_t w_dtype, float eps);

void TETranspose(cudaStream_t stream, void **buffers, const char *opaque,
                 std::size_t opaque_len);

void TECastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                     std::size_t opaque_len);

void TEGatedGelu(cudaStream_t stream, void **buffers, const char *opaque,
                 std::size_t opaque_len);

void TEGatedGeluFP8(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len);

void TEDGatedGelu(cudaStream_t stream, void **buffers, const char *opaque,
                  std::size_t opaque_len);

void TEDGatedGeluCastTranspose(cudaStream_t stream, void **buffers,
                               const char *opaque, std::size_t opaque_len);

void TEGemm(cudaStream_t stream, void **buffers, const char *opaque,
            std::size_t opaque_len);

void TELayerNormForward(cudaStream_t stream, void **buffers, const char *opaque,
                        std::size_t opaque_len);

void TELayerNormForwardFP8(cudaStream_t stream, void **buffers,
                           const char *opaque, std::size_t opaque_len);

void TELayerNormBackward(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len);

void TERMSNormForward(cudaStream_t stream, void **buffers, const char *opaque,
                      std::size_t opaque_len);

void TERMSNormForwardFP8(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len);

void TERMSNormBackward(cudaStream_t stream, void **buffers, const char *opaque,
                       std::size_t opaque_len);

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_
