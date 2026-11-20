/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

namespace transformer_engine::pytorch {

py::object cross_entropy_backward_mxfp8(const at::Tensor &logits, const at::Tensor &target,
                                        const at::Tensor &stats,
                                        const std::optional<at::Tensor> &n_non_ignore,
                                        const at::Tensor &grad_output, const int64_t ignore_idx,
                                        const bool reduce_loss, py::handle grad_quantizer) {
  TORCH_CHECK(logits.is_cuda(), "expected logits on a CUDA device");
  const auto device = logits.device();
  at::cuda::CUDAGuard device_guard(device);
  TORCH_CHECK(logits.scalar_type() == at::ScalarType::BFloat16, "expected BF16 logits");
  TORCH_CHECK(logits.is_contiguous() && logits.dim() >= 2, "expected contiguous logits");
  TORCH_CHECK(
      target.device() == device && stats.device() == device && grad_output.device() == device,
      "expected logits, target, stats and grad_output on the same device");
  TORCH_CHECK(!reduce_loss || n_non_ignore.has_value(), "reduce_loss needs n_non_ignore");

  auto quantizer_cpp = convert_quantizer(grad_quantizer);
  const auto *mxfp8 = dynamic_cast<MXFP8Quantizer *>(quantizer_cpp.get());
  TORCH_CHECK(mxfp8 != nullptr, "cross_entropy_backward_mxfp8 requires an MXFP8 quantizer");
  TORCH_CHECK(!mxfp8->with_2d_quantization,
              "cross_entropy_backward_mxfp8 does not support 2D MXFP8 quantization");

  const auto target_c = target.contiguous();
  const auto stats_c = stats.contiguous();
  const auto grad_output_c = grad_output.contiguous();
  auto logits_cu = makeTransformerEngineTensor(logits);
  auto target_cu = makeTransformerEngineTensor(target_c);
  auto stats_cu = makeTransformerEngineTensor(stats_c);
  auto grad_output_cu = makeTransformerEngineTensor(grad_output_c);
  TensorWrapper n_non_ignore_cu;
  if (n_non_ignore.has_value()) {
    n_non_ignore_cu = makeTransformerEngineTensor(*n_non_ignore);
  }

  std::vector<size_t> shape(logits.sizes().begin(), logits.sizes().end());
  auto [grad_cu, grad_py] =
      quantizer_cpp->create_tensor(shape, GetTransformerEngineDType(logits.scalar_type()), device);

  NVTETensor n_non_ignore_nvte = n_non_ignore.has_value() ? n_non_ignore_cu.data() : nullptr;
  NVTETensor grad_nvte = grad_cu.data();
  auto stream = at::cuda::getCurrentCUDAStream();
  {
    // Called from Python with the GIL held; the kernel launch does not need it.
    pybind11::gil_scoped_release release;
    nvte_cross_entropy_backward_mxfp8(logits_cu.data(), target_cu.data(), stats_cu.data(),
                                      n_non_ignore_nvte, grad_output_cu.data(), grad_nvte,
                                      ignore_idx, reduce_loss, stream);
  }
  return grad_py;
}

}  // namespace transformer_engine::pytorch
