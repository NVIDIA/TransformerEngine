/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_NORM_KERNEL_PARAMS_H_
#define TRANSFORMER_ENGINE_COMMON_NORM_KERNEL_PARAMS_H_

// POD kernel-parameter structs shared between the host norm dispatchers and
// the NVRTC kernel sources. This header is intentionally free of host-only
// includes (no <transformer_engine/...>, no cudnn) so it can be compiled by
// NVRTC as well.

namespace transformer_engine {
namespace normalization {

struct KernelParamsBase {
  // For Multi-CTA, number of different CTA groups. Otherwise same as gridDim.x.
  int ctas_per_col = 0;
  // Size of CTA group.
  int ctas_per_row = 0;

  // Input is interpreted as matrix. We normalize across columns.
  int rows = 0;
  int cols = 0;

  // Common data pointers.
  void* x = nullptr;
  void* mu = nullptr;
  void* rs = nullptr;
  void* gamma = nullptr;

  // Multi-CTA workspace in gmem.
  void* workspace = nullptr;

  // Multi-CTA sync barriers in gmem.
  int* barrier = nullptr;

  // Whether gamma is centered around 0
  bool zero_centered_gamma = false;
};

struct ForwardKernelParams : public KernelParamsBase {
  // Output of LN FWD.
  void* z = nullptr;
  void* beta = nullptr;
  float epsilon = 0.f;

  // Scaling factor
  void* scale = nullptr;
  int scale_byte_size = 0;

  // Inverse of scaling factor
  void* scale_inv = nullptr;

  // AMax output
  void* amax = nullptr;
  int amax_byte_size = 0;

  // Whether to compute scale and amax
  bool fp8_out = false;
};

struct BackwardKernelParams : public KernelParamsBase {
  // Input: gradient wrt. LN FWD output.
  void* dz = nullptr;

  // Input: extra tensor to add for fused backward+add
  void* add = nullptr;

  // Workspace for Wgrad pre-reduction.
  void* dbeta_part = nullptr;
  void* dgamma_part = nullptr;

  // Output: Dgrad.
  void* dx = nullptr;
  // Output: Wgrad.
  void* dbeta = nullptr;
  void* dgamma = nullptr;
};

using BackwardAddKernelParams = BackwardKernelParams;

}  // namespace normalization
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_NORM_KERNEL_PARAMS_H_
