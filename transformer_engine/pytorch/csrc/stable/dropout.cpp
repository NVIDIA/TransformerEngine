/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/dropout.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

// ============================================================================
// Dropout forward — RNG state extracted in Python, passed as tensor
//
// Python shim does:
//   gen = torch.cuda.default_generators[device]
//   philox_state = gen.get_state()  # or philox_cuda_state for graph capture
//   seed, offset = extract_seed_offset(philox_state)
//   rng_state = torch.tensor([seed, offset], dtype=torch.int64, device='cuda')
// ============================================================================

std::tuple<Tensor, Tensor> dropout_fwd(Tensor input, Tensor rng_state, double dropout_probability) {
  auto input_cu = makeTransformerEngineTensor(input);

  auto device_idx = input.get_device_index();
  auto shape = getStableTensorShape(input);
  size_t total = 1;
  for (auto s : shape) total *= s;

  // Mask: 1 bit per element, packed into uint8
  auto mask =
      allocateStableTensor({static_cast<int64_t>((total + 7) / 8)}, ScalarType::Byte, device_idx);

  auto output = torch::stable::empty_like(input);

  auto output_cu = makeTransformerEngineTensor(output);
  auto mask_cu = makeTransformerEngineTensor(mask);
  auto rng_state_cu = makeTransformerEngineTensor(rng_state);

  nvte_dropout_fwd(input_cu.data(), output_cu.data(), mask_cu.data(), rng_state_cu.data(),
                   static_cast<float>(dropout_probability), getCurrentCUDAStreamRaw(device_idx));

  return std::make_tuple(output, mask);
}

// ============================================================================
// Dropout backward
// ============================================================================

Tensor dropout_bwd(Tensor grad_output, Tensor mask, double dropout_probability,
                   std::optional<Tensor> grad_input) {
  auto grad_output_ = torch::stable::contiguous(grad_output);

  Tensor grad_in;
  if (grad_input.has_value()) {
    grad_in = grad_input.value();
  } else {
    grad_in = torch::stable::empty_like(grad_output_);
  }

  auto grad_output_cu = makeTransformerEngineTensor(grad_output_);
  auto mask_cu = makeTransformerEngineTensor(mask);
  auto grad_input_cu = makeTransformerEngineTensor(grad_in);

  nvte_dropout_bwd(grad_output_cu.data(), mask_cu.data(), grad_input_cu.data(),
                   static_cast<float>(dropout_probability),
                   getCurrentCUDAStreamRaw(grad_output_.get_device_index()));

  return grad_in;
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def(
      "dropout_fwd(Tensor input, Tensor rng_state, float dropout_probability) -> (Tensor, Tensor)");
  m.def(
      "dropout_bwd(Tensor grad_output, Tensor mask, float dropout_probability, Tensor? grad_input) "
      "-> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("dropout_fwd", TORCH_BOX(dropout_fwd));
  m.impl("dropout_bwd", TORCH_BOX(dropout_bwd));
}
