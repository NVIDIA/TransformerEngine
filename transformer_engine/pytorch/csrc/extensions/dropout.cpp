/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/dropout.h"

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <pybind.h>

#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include "../common.h"
#include "../extensions.h"
#include "../pybind.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace pytorch {

std::vector<py::object> dropout_fwd(const py::handle &input, float dropout_probability,
                                    std::optional<at::Tensor> out) {
  using namespace transformer_engine::pytorch::detail;

  // Input tensor
  const TensorWrapper input_nvte = makeTransformerEngineTensor(input, py::none());

  // Allocate output tensor if needed
  if (!out) {
    at::ScalarType dtype = GetATenDType(input_nvte.dtype());
    if (dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e5m2) {
      dtype = input.attr("dtype").cast<at::ScalarType>();
    }
    const auto shape_uint64 = convertShape(input_nvte.shape());
    const std::vector<int64_t> shape_int64(shape_uint64.begin(), shape_uint64.end());
    const auto opts = at::TensorOptions().dtype(dtype).device(torch::kCUDA);
    out = at::empty(shape_int64, opts);
  }
  TensorWrapper out_nvte = makeTransformerEngineTensor(*out);

  // Mask tensor
  auto mask_pyt = allocateTorchTensor(input_nvte.numel() / 8, DType::kByte);
  auto mask_nvte = makeTransformerEngineTensor(mask_pyt);

  // RNG state tensor
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox_args;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    constexpr int64_t rng_elts_per_thread = 4;
    philox_args = gen->philox_cuda_state(rng_elts_per_thread);
  }
  auto rng_state_pyt = allocateTorchTensor(2, DType::kInt64);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_extract_seed_and_offset(
        reinterpret_cast<int64_t *>(rng_state_pyt.data_ptr()), philox_args.captured_,
        philox_args.seed_.ptr, philox_args.seed_.val, philox_args.offset_.ptr,
        philox_args.offset_.val, philox_args.offset_intragraph_, at::cuda::getCurrentCUDAStream());
  });
  auto rng_state_nvte = makeTransformerEngineTensor(rng_state_pyt);

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_dropout_fwd(input_nvte.data(), out_nvte.data(), mask_nvte.data(), rng_state_nvte.data(),
                     dropout_probability, at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(std::move(*out)), py::cast(mask_pyt)};
}

py::object dropout_bwd(const at::Tensor &grad_output, const at::Tensor &mask,
                       const float dropout_probability, std::optional<at::Tensor> grad_input) {
  const auto grad_output_nvte = makeTransformerEngineTensor(grad_output);
  const auto mask_nvte = makeTransformerEngineTensor(mask);
  if (!grad_input) {
    grad_input = at::empty_like(grad_output);
  }
  auto grad_input_nvte = makeTransformerEngineTensor(*grad_input);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_dropout_bwd(grad_output_nvte.data(), mask_nvte.data(), grad_input_nvte.data(),
                     dropout_probability, at::cuda::getCurrentCUDAStream());
  });
  return py::cast(std::move(*grad_input));
}

}  // namespace pytorch
}  // namespace transformer_engine
