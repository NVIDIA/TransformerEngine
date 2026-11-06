/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NVFP4 token rows for the EP dispatch and combine backward. Independent of NCCL EP.

#include <ATen/cuda/CUDAContext.h>

#include <optional>
#include <vector>

#include "../common.h"
#include "../extensions.h"
#include "../pybind.h"
#include "common/util/system.h"
#include "transformer_engine/ep.h"

namespace transformer_engine::pytorch {

namespace {

int64_t nvfp4_row_bytes(int64_t hidden) {
  return static_cast<int64_t>(roundup(static_cast<size_t>(hidden / 2 + hidden / 16 + 4), 16));
}

int use_fast_math() { return transformer_engine::getenv<bool>("NVTE_USE_FAST_MATH") ? 1 : 0; }

// (seed, offset) from torch's CUDA generator; under graph capture the offset is read on device,
// so every replay draws fresh bits.
at::Tensor make_rng_state(size_t elts_per_thread) {
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox_args = init_philox_state(gen, elts_per_thread);
  auto rng_state = at::empty({2}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  philox_unpack(philox_args, static_cast<int64_t *>(rng_state.data_ptr()));
  return rng_state;
}

}  // namespace

at::Tensor ep_nvfp4_pack(const at::Tensor &tokens, bool stochastic) {
  NVTE_CHECK(tokens.is_cuda() && tokens.dim() == 2, "tokens must be a 2D CUDA tensor.");
  const auto x = tokens.contiguous();
  auto packed = at::empty({x.size(0), nvfp4_row_bytes(x.size(1))}, x.options().dtype(at::kByte));
  std::optional<at::Tensor> rng_state;
  TensorWrapper rng_state_nvte;
  if (stochastic) {
    rng_state = make_rng_state(4);
    rng_state_nvte = makeTransformerEngineTensor(*rng_state);
  }
  auto x_nvte = makeTransformerEngineTensor(x);
  auto packed_nvte = makeTransformerEngineTensor(packed);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_ep_nvfp4_pack(x_nvte.data(), packed_nvte.data(), rng_state_nvte.data(), use_fast_math(),
                       at::cuda::getCurrentCUDAStream());
  });
  return packed;
}

at::Tensor ep_nvfp4_unpack(const at::Tensor &packed, int64_t hidden,
                           std::optional<at::Tensor> counts, std::optional<at::Tensor> out) {
  NVTE_CHECK(packed.is_cuda() && packed.dim() == 2 && packed.scalar_type() == at::kByte,
             "packed must be a 2D uint8 CUDA tensor.");
  const auto p = packed.contiguous();
  if (!out) {
    out = at::empty({p.size(0), hidden}, p.options().dtype(at::kBFloat16));
  }
  NVTE_CHECK(out->is_contiguous() && out->numel() == p.size(0) * hidden,
             "out must be a contiguous [", p.size(0), ", ", hidden, "] tensor.");
  auto out_2d = out->view({p.size(0), hidden});
  TensorWrapper counts_nvte;
  if (counts) {
    counts = counts->contiguous();
    counts_nvte = makeTransformerEngineTensor(*counts);
  }
  auto p_nvte = makeTransformerEngineTensor(p);
  auto out_nvte = makeTransformerEngineTensor(out_2d);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_ep_nvfp4_unpack(p_nvte.data(), counts_nvte.data(), out_nvte.data(),
                         at::cuda::getCurrentCUDAStream());
  });
  return *out;
}

std::vector<at::Tensor> ep_nvfp4_direct_receive(const at::Tensor &packed, int64_t hidden,
                                                const at::Tensor &counts, int64_t sign_mask,
                                                bool stochastic) {
  NVTE_CHECK(packed.is_cuda() && packed.dim() == 2 && packed.scalar_type() == at::kByte,
             "packed must be a 2D uint8 CUDA tensor.");
  NVTE_CHECK(counts.scalar_type() == at::kLong, "counts must be int64.");
  const auto p = packed.contiguous();
  const auto c = counts.contiguous();
  const int64_t capacity = p.size(0);
  const int64_t num_groups = c.numel();
  const auto u8 = p.options().dtype(at::kByte);
  const auto f32 = p.options().dtype(at::kFloat);
  auto row_data = at::empty({capacity * hidden / 2}, u8);
  auto row_scale = at::empty({capacity * hidden / 16}, u8);
  auto col_data = at::empty({capacity * hidden / 2}, u8);
  auto col_scale = at::empty({capacity * hidden / 16}, u8);
  auto row_amax = at::empty({num_groups}, f32);
  auto col_amax = at::empty({num_groups}, f32);
  auto workspace = at::empty({num_groups}, p.options().dtype(at::kInt));
  std::optional<at::Tensor> rng_state;
  TensorWrapper rng_state_nvte;
  if (stochastic) {
    rng_state = make_rng_state(32);
    rng_state_nvte = makeTransformerEngineTensor(*rng_state);
  }
  auto p_nvte = makeTransformerEngineTensor(p);
  auto c_nvte = makeTransformerEngineTensor(c);
  auto workspace_nvte = makeTransformerEngineTensor(workspace);
  auto row_data_nvte = makeTransformerEngineTensor(row_data);
  auto row_scale_nvte = makeTransformerEngineTensor(row_scale);
  auto col_data_nvte = makeTransformerEngineTensor(col_data);
  auto col_scale_nvte = makeTransformerEngineTensor(col_scale);
  auto row_amax_nvte = makeTransformerEngineTensor(row_amax);
  auto col_amax_nvte = makeTransformerEngineTensor(col_amax);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_ep_nvfp4_direct_receive(
        p_nvte.data(), c_nvte.data(), workspace_nvte.data(), static_cast<uint32_t>(sign_mask),
        row_data_nvte.data(), row_scale_nvte.data(), col_data_nvte.data(), col_scale_nvte.data(),
        row_amax_nvte.data(), col_amax_nvte.data(), rng_state_nvte.data(),
        static_cast<size_t>(hidden), use_fast_math(), at::cuda::getCurrentCUDAStream());
  });
  return {row_data, row_scale, col_data, col_scale, row_amax, col_amax};
}

}  // namespace transformer_engine::pytorch
