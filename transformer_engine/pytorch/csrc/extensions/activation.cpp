/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

at::Tensor gelu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype, amax.data_ptr(),
                                               scale.data_ptr(), scale_inv.data_ptr());

  nvte_gelu(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor dgelu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto gtype = GetTransformerEngineDType(grad.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto grad_cu = makeTransformerEngineTensor(grad.data_ptr(), {M, N}, gtype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype);

  nvte_dgelu(grad_cu.data(), input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor relu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = static_cast<size_t>(input.numel()) / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype, amax.data_ptr(),
                                               scale.data_ptr(), scale_inv.data_ptr());

  nvte_relu(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor drelu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto gtype = GetTransformerEngineDType(grad.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto grad_cu = makeTransformerEngineTensor(grad.data_ptr(), {M, N}, gtype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype);

  nvte_drelu(grad_cu.data(), input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor geglu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                 transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N / 2, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto output_cu =
      makeTransformerEngineTensor(output.data_ptr(), {M, N / 2}, otype, amax.data_ptr(),
                                  scale.data_ptr(), scale_inv.data_ptr());

  nvte_geglu(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor dgeglu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto gtype = GetTransformerEngineDType(grad.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto grad_cu = makeTransformerEngineTensor(grad.data_ptr(), {M, N / 2}, gtype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype);

  nvte_dgeglu(grad_cu.data(), input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor reglu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                 transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N / 2, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto output_cu =
      makeTransformerEngineTensor(output.data_ptr(), {M, N / 2}, otype, amax.data_ptr(),
                                  scale.data_ptr(), scale_inv.data_ptr());

  nvte_reglu(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor dreglu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto gtype = GetTransformerEngineDType(grad.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto grad_cu = makeTransformerEngineTensor(grad.data_ptr(), {M, N / 2}, gtype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype);

  nvte_dreglu(grad_cu.data(), input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor swiglu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                  transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N / 2, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto output_cu =
      makeTransformerEngineTensor(output.data_ptr(), {M, N / 2}, otype, amax.data_ptr(),
                                  scale.data_ptr(), scale_inv.data_ptr());

  nvte_swiglu(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor dswiglu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto gtype = GetTransformerEngineDType(grad.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto grad_cu = makeTransformerEngineTensor(grad.data_ptr(), {M, N / 2}, gtype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype);

  nvte_dswiglu(grad_cu.data(), input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor qgelu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                 transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype, amax.data_ptr(),
                                               scale.data_ptr(), scale_inv.data_ptr());

  nvte_qgelu(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor dqgelu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto gtype = GetTransformerEngineDType(grad.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto grad_cu = makeTransformerEngineTensor(grad.data_ptr(), {M, N}, gtype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype);

  nvte_dqgelu(grad_cu.data(), input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor srelu(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                 transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = static_cast<size_t>(input.numel()) / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype, amax.data_ptr(),
                                               scale.data_ptr(), scale_inv.data_ptr());

  nvte_srelu(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor dsrelu(at::Tensor grad, at::Tensor input, transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t N = static_cast<size_t>(input.size(-1));
  size_t M = input.numel() / N;

  auto output = allocateTorchTensor(M, N, otype);

  auto itype = GetTransformerEngineDType(input.scalar_type());
  auto gtype = GetTransformerEngineDType(grad.scalar_type());
  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, itype);
  auto grad_cu = makeTransformerEngineTensor(grad.data_ptr(), {M, N}, gtype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype);

  nvte_dsrelu(grad_cu.data(), input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}
