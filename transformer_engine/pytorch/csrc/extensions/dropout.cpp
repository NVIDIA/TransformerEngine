/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "pybind.h"
#include "transformer_engine/dropout.h"
#include "transformer_engine/transformer_engine.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

namespace transformer_engine::pytorch {

void unpack(at::PhiloxCudaState arg, int64_t *rng_state_ptr) {
  NVTE_SCOPED_GIL_RELEASE({
    nvte_extract_seed_and_offset(rng_state_ptr, arg.captured_, arg.seed_.ptr, arg.seed_.val,
                                  arg.offset_.ptr, arg.offset_.val, arg.offset_intragraph_,
                                  at::cuda::getCurrentCUDAStream());
  });
}

std::vector<py::object> dropout_fwd(const at::Tensor& input, float dropout_probability, bool is_training) {
  using namespace transformer_engine::pytorch::detail;

  auto input_tensor = makeTransformerEngineTensor(input);
  const int rng_block_size=16;
  int numel = input.numel();

  // Allocate output tensor
  std::vector<int> output_shape;
  for (auto s : input.sizes()) {
    output_shape.emplace_back(static_cast<int>(s));
  }
  auto output = allocateTorchTensor(output_shape[0], output_shape[1], input_tensor.dtype());
  //auto output = at::empty(output_shape, input_tensor.dtype());
  auto output_tensor = makeTransformerEngineTensor(output);

  // Allocate mask tensor
  std::vector<size_t> mask_shape{input.numel() / 16};
  auto mask = allocateTorchTensor(mask_shape[0], DType::kInt16);
  //auto mask = at::empty(mask_shape, at::CUDA(at::kUInt16));
  auto mask_tensor = makeTransformerEngineTensor(mask);

  if (!is_training) {
    return {py::cast(input), py::cast(mask)};
  }

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  // Offset can be 1 but setting it to 16 to be safe
  int64_t rng_elts_per_thread = rng_block_size;
  at::PhiloxCudaState philox_args;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    philox_args = gen->philox_cuda_state(rng_elts_per_thread);
  }
  auto options = torch::TensorOptions().device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  unpack(philox_args, static_cast<int64_t *>(rng_state.data_ptr()));
  auto te_rng_state = makeTransformerEngineTensor(rng_state);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_dropout_fwd(input_tensor.data(), output_tensor.data(), mask_tensor.data(),
                te_rng_state.data(), dropout_probability, at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(output), py::cast(mask)};
}

std::vector<py::object> dropout_fwd_fp8(const py::handle &input, at::Tensor &output, const float dropout_probability) {
  const auto none = py::none();
  auto input_tensor = makeTransformerEngineTensor(input, none);
  const int rng_block_size=16;
  auto input_shape = input_tensor.shape();
  int numel = input_shape.data[0] * input_shape.data[1];

  auto output_tensor = makeTransformerEngineTensor(output);
  // Allocate mask tensor
  std::vector<size_t> mask_shape{numel / 16};
  auto mask = allocateTorchTensor(mask_shape[0], DType::kInt16);
  //auto mask = at::empty(mask_shape, at::CUDA(at::kUInt16));
  auto mask_tensor = makeTransformerEngineTensor(mask);

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  // Offset can be 1 but setting it to 16 to be safe
  int64_t rng_elts_per_thread = rng_block_size;
  at::PhiloxCudaState philox_args;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    philox_args = gen->philox_cuda_state(rng_elts_per_thread);
  }
  auto options = torch::TensorOptions().device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  unpack(philox_args, static_cast<int64_t *>(rng_state.data_ptr()));
  auto te_rng_state = makeTransformerEngineTensor(rng_state);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_dropout_fwd_fp8(input_tensor.data(), output_tensor.data(), mask_tensor.data(),
                te_rng_state.data(), dropout_probability, at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(output), py::cast(mask)};
}

py::object dropout_bwd(const at::Tensor &grad_output, const at::Tensor &mask, const float dropout_probability) {
  auto grad_output_tensor = makeTransformerEngineTensor(grad_output);
  auto mask_tensor = makeTransformerEngineTensor(mask);
  // Allocate output tensor
  std::vector<int> grad_input_shape;
  for (auto s : grad_output.sizes()) {
    grad_input_shape.emplace_back(static_cast<int>(s));
  }
  auto grad_input = allocateTorchTensor(grad_input_shape[0], grad_input_shape[1], grad_output_tensor.dtype());
  auto grad_input_tensor = makeTransformerEngineTensor(grad_input);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_dropout_bwd(grad_output_tensor.data(), mask_tensor.data(), grad_input_tensor.data(),
                dropout_probability, at::cuda::getCurrentCUDAStream());
  });

  return py::cast(grad_input);
}

}  // namespace transformer_engine::pytorch
