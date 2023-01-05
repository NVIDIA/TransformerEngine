/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/extension.h>

#include "ATen/cuda/CUDAContext.h"
#include "softmax.h"

/*

Supported Type combinations:

input    compute   output
=============================
fp32     fp32      fp32
fp16     fp32      fp16
bf16     fp32      bf16
fp32     fp32      fp16
fp32     fp32      bf16

Remarks:
Compute always in FP32

*/

namespace softmax {

// Create registries and provide runtime versions of config hash functions.

FwdRegistry FWD_FUNCS;
BwdRegistry BWD_FUNCS;

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t get_type_id(torch::Dtype dtype) {
  if (dtype == torch::kFloat16) {
    return TypeId<fp16>::Value;
  } else if (dtype == torch::kBFloat16) {
    return TypeId<bf16>::Value;
  } else if (dtype == torch::kFloat32) {
    return TypeId<fp32>::Value;
  } else {
    TORCH_CHECK(false, "Type not supported: ", dtype);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t get_key(torch::Dtype itype, torch::Dtype otype, torch::Dtype ctype,
                 uint64_t hidden_size, MaskMode mask_mode) {
  using namespace softmax;
  uint64_t type_key = (get_type_id(itype) << 0) | (get_type_id(otype) << 2) |
                      (get_type_id(ctype) << 4);
  uint64_t mask_mode_(mask_mode);
  uint64_t launcher_key = (((type_key << 32) | hidden_size) << 2) | mask_mode_;
  return launcher_key;
}

}  // namespace softmax

////////////////////////////////////////////////////////////////////////////////////////////////////

softmax::FwdFunction &get_fwd_launcher(torch::Dtype itype, torch::Dtype otype,
                                       torch::Dtype ctype, uint32_t hidden_size,
                                       softmax::MaskMode mask_mode) {
  auto iter = softmax::FWD_FUNCS.find(
      softmax::get_key(itype, otype, ctype, hidden_size, mask_mode));
  if (iter != softmax::FWD_FUNCS.end()) {
    return iter->second;
  } else {
    TORCH_CHECK(false, "FWD: Unsupported hidden_size or types: ", hidden_size,
                itype, otype, ctype, mask_mode);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

softmax::BwdFunction &get_bwd_launcher(torch::Dtype itype, torch::Dtype otype,
                                       torch::Dtype ctype,
                                       uint32_t hidden_size) {
  auto iter = softmax::BWD_FUNCS.find(
      softmax::get_key(itype, otype, ctype, hidden_size, softmax::SELF));
  if (iter != softmax::BWD_FUNCS.end()) {
    return iter->second;
  } else {
    TORCH_CHECK(false, "BWD: Unsupported hidden_size or types: ", hidden_size,
                itype, otype, ctype);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

at::Tensor softmax_fwd(const at::Tensor &x,           // BxHxSqxSk
                       const at::Tensor &cu_seqlens,  // B+1
                       const float scale_pre_softmax, const float p_dropout,
                       const softmax::MaskMode mask_mode,
                       c10::optional<at::Generator> gen_) {
  auto itype = x.scalar_type();
  auto otype = itype;
  auto ctype = torch::kFloat32;

  TORCH_CHECK(x.is_cuda())

  TORCH_CHECK(x.is_contiguous());
  auto sizes = x.sizes();
  TORCH_CHECK(sizes.size() == 4);
  const int b = sizes[0];
  const int h = sizes[1];
  const int sq = sizes[2];
  const int sk = sizes[3];

  TORCH_CHECK(p_dropout >= 0.f);

  auto opts = x.options();

  auto z = torch::empty(sizes, opts.dtype(otype));

  softmax::LaunchParams<softmax::FwdParams> launch_params;

  launch_params.props = at::cuda::getCurrentDeviceProperties();
  launch_params.stream = at::cuda::getCurrentCUDAStream().stream();
  launch_params.mask_mode = mask_mode;

  // Request the kernel launcher.
  auto launcher = get_fwd_launcher(itype, otype, ctype, sk, mask_mode);

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  // Set the kernel runtime parameters.
  softmax::FwdParams &params = launch_params.params;
  params.b = b;
  params.h = h;
  params.sq = sq;
  params.sk = sk;
  params.x = x.data_ptr();
  params.z = z.data_ptr();
  params.p_keep = 1.f - p_dropout;
  params.p_keep_inv = 1.f / (1.f - p_dropout);
  params.scale_pre_softmax = scale_pre_softmax;

  if (mask_mode == softmax::SELF) {
    TORCH_CHECK(sq == sk);
    TORCH_CHECK(b + 1 == cu_seqlens.numel());
    params.cu_seqlens = cu_seqlens.data_ptr<int>();
  }

  {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    // number of times random will be generated per thread, to offset philox
    // counter in thc random state
    const int64_t counter_offset = launch_params.elts_per_thread;
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
  }

  // Launch the kernel.
  launcher(launch_params, false);

  return z;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

at::Tensor softmax_bwd(const at::Tensor &dz,          // BxHxSqxSk
                       const at::Tensor &smat_dmask,  // BxHxSqxSk
                       const float scale_pre_softmax, float p_dropout) {
  auto itype = dz.scalar_type();
  auto otype = itype;
  auto ctype = torch::kFloat32;

  TORCH_CHECK(dz.is_cuda())
  TORCH_CHECK(smat_dmask.is_cuda())

  TORCH_CHECK(dz.is_contiguous());
  auto sizes = dz.sizes();
  TORCH_CHECK(sizes.size() == 4);
  const int b = sizes[0];
  const int h = sizes[1];
  const int sq = sizes[2];
  const int sk = sizes[3];

  TORCH_CHECK(p_dropout >= 0.f);

  auto opts = dz.options();
  auto dx = torch::empty(dz.sizes(), opts);
  // auto dx = torch::empty_like(dz);

  softmax::LaunchParams<softmax::BwdParams> launch_params;

  launch_params.props = at::cuda::getCurrentDeviceProperties();
  launch_params.stream = at::cuda::getCurrentCUDAStream().stream();

  // Request the kernel launcher.
  auto launcher = get_bwd_launcher(itype, otype, ctype, sk);

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  // Set the kernel runtime parameters.
  softmax::BwdParams &params = launch_params.params;
  params.b = b;
  params.h = h;
  params.sq = sq;
  params.sk = sk;
  params.dx = dx.data_ptr();
  params.dz = dz.data_ptr();
  params.smat_dmask = smat_dmask.data_ptr();
  params.p_keep = (1.f - p_dropout);
  params.p_keep_inv = 1.f /(1.f - p_dropout);
  params.scale_pre_softmax = scale_pre_softmax;

  // Launch the kernel.
  launcher(launch_params, false);

  return dx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
