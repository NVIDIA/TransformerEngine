/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"


void te_gemm(at::Tensor A,
             at::Tensor A_scale_inverse,
             transformer_engine::DType A_type,
             bool transa,
             at::Tensor B,
             at::Tensor B_scale_inverse,
             transformer_engine::DType B_type,
             bool transb,
             at::Tensor D,
             transformer_engine::DType D_type,
             at::Tensor bias,
             at::Tensor pre_gelu_out,
             bool grad,
             at::Tensor workspace,
             size_t workspaceSize,
             bool accumulate,
             bool use_split_accumulator
) {
  using namespace transformer_engine;
  auto te_A = makeTransformerEngineTensor(A.data_ptr(),
                                          {static_cast<size_t>(A.size(0)),
                                           static_cast<size_t>(A.size(1))},
                                          A_type);
  auto te_A_scale_inverse = makeTransformerEngineTensor(A_scale_inverse.data_ptr(), {1},
                                                        GetTransformerEngineDType(
                                                            A_scale_inverse.scalar_type()));
  auto te_B = makeTransformerEngineTensor(B.data_ptr(),
                                          {static_cast<size_t>(B.size(0)),
                                           static_cast<size_t>(B.size(1))},
                                          B_type);
  auto te_B_scale_inverse = makeTransformerEngineTensor(B_scale_inverse.data_ptr(), {1},
                                                        GetTransformerEngineDType(
                                                            B_scale_inverse.scalar_type()));
  auto te_D = makeTransformerEngineTensor(D.data_ptr(),
                                          {static_cast<size_t>(D.size(0)),
                                           static_cast<size_t>(D.size(1))},
                                          D_type);
  auto te_bias = makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))},
                                             GetTransformerEngineDType(bias.scalar_type()));

  const auto gelu_shape = pre_gelu_out.data_ptr() == nullptr
                          ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                          : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                static_cast<size_t>(pre_gelu_out.size(1))};
  auto te_pre_gelu_out = makeTransformerEngineTensor(pre_gelu_out.data_ptr(),
                                                     gelu_shape,
                                                     GetTransformerEngineDType(
                                                         pre_gelu_out.scalar_type()));
  auto te_workspace = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  {workspaceSize},
                                                  DType::kByte);

  nvte_cublas_gemm(te_A.data(),
                   te_A_scale_inverse.data(),
                   te_B.data(),
                   te_B_scale_inverse.data(),
                   te_D.data(),
                   te_bias.data(),
                   te_pre_gelu_out.data(),
                   transa,
                   transb,
                   grad,
                   te_workspace.data(),
                   accumulate,
                   use_split_accumulator,
                   at::cuda::getCurrentCUDAStream());
}


void fused_cast_transpose(at::Tensor input,
                          at::Tensor scale,
                          at::Tensor amax,
                          at::Tensor scale_inv,
                          at::Tensor input_cast,
                          at::Tensor input_transpose,
                          transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  DType inp_type = GetTransformerEngineDType(input.scalar_type());

  dispatch_cast_transpose_fusion(
          input.data_ptr(), {M, N}, inp_type,
          scale.data_ptr(), {1}, DType::kFloat32,
          input_cast.data_ptr(), {M, N}, otype,
          input_transpose.data_ptr(), {N, M}, otype,
          amax.data_ptr(), {1}, DType::kFloat32,
          scale_inv.data_ptr(), {1}, DType::kFloat32);
}


std::vector<at::Tensor> fused_cast_transpose_bgrad(at::Tensor grad_output,
                                                   at::Tensor scale,
                                                   at::Tensor amax,
                                                   at::Tensor scale_inv,
                                                   transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto grad_output_cast =
            allocateTorchTensor(grad_output.size(0),
                                grad_output.size(1),
                                DType::kByte);
  auto grad_output_transpose =
            allocateTorchTensor(grad_output.size(1),
                                grad_output.size(0),
                                DType::kByte);

  dispatch_bgrad_cast_transpose_fusion(
          grad_output.data_ptr(), {M, N}, grad_output_type,
          scale.data_ptr(), {1}, DType::kFloat32,
          grad_output_cast.data_ptr(), {M, N}, otype,
          grad_output_transpose.data_ptr(), {N, M}, otype,
          amax.data_ptr(), {1}, DType::kFloat32,
          grad_bias.data_ptr(), {N}, grad_output_type,
          scale_inv.data_ptr(), {1}, DType::kFloat32);

  return {grad_bias, grad_output_cast, grad_output_transpose};
}


std::vector<at::Tensor> fused_cast_transpose_bgrad_dgelu(at::Tensor grad_output,
                                                         at::Tensor gelu_input,
                                                         at::Tensor scale,
                                                         at::Tensor amax,
                                                         at::Tensor scale_inv,
                                                         transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dgelu =
            allocateTorchTensor(grad_output.size(0),
                                grad_output.size(1),
                                DType::kByte);
  auto dgelu_transpose =
            allocateTorchTensor(grad_output.size(1),
                                grad_output.size(0),
                                DType::kByte);

  dispatch_bgrad_dgelu_cast_transpose_fusion(
          grad_output.data_ptr(), {M, N}, grad_output_type,
          gelu_input.data_ptr(), {M, N}, grad_output_type,
          scale.data_ptr(), {1}, DType::kFloat32,
          dgelu.data_ptr(), {M, N}, otype,
          dgelu_transpose.data_ptr(), {N, M}, otype,
          amax.data_ptr(), {1}, DType::kFloat32,
          grad_bias.data_ptr(), {N}, grad_output_type,
          scale_inv.data_ptr(), {1}, DType::kFloat32);

  return {grad_bias, dgelu, dgelu_transpose};
}


at::Tensor fp8_transpose(at::Tensor input,
                         transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto input_transpose =
            allocateTorchTensor(input.size(1),
                                input.size(0),
                                DType::kByte);
  dispatch_transpose(input.data_ptr(), {M, N}, otype,
                     input_transpose.data_ptr(), {N, M}, otype);

  return input_transpose;
}


at::Tensor fp8_gelu(at::Tensor input,
                    at::Tensor scale,
                    at::Tensor amax,
                    at::Tensor scale_inv,
                    transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  DType input_type = GetTransformerEngineDType(input.scalar_type());

  auto output =
            allocateTorchTensor(input.size(0),
                                input.size(1),
                                DType::kByte);

  dispatch_gelu(input.data_ptr(), {M, N}, input_type,
                scale.data_ptr(), {1}, DType::kFloat32,
                output.data_ptr(), {M, N}, otype,
                amax.data_ptr(), {1}, DType::kFloat32,
                scale_inv.data_ptr(), {1}, DType::kFloat32);

  return output;
}


std::vector<at::Tensor> layernorm_bwd(const at::Tensor &dz,
                                      const at::Tensor &x,
                                      const at::Tensor &mu,
                                      const at::Tensor &rsigma,
                                      const at::Tensor &gamma
) {
    auto dx = at::empty_like(x);
    auto dgamma = at::empty_like(gamma);
    auto dbeta = at::empty_like(gamma);
    transformer_engine::TensorWrapper workspace, barrier, dgamma_part, dbeta_part;

    auto dz_cu      = makeTransformerEngineTensor(dz);
    auto x_cu       = makeTransformerEngineTensor(x);
    auto mu_cu      = makeTransformerEngineTensor(mu);
    auto rsigma_cu  = makeTransformerEngineTensor(rsigma);
    auto gamma_cu   = makeTransformerEngineTensor(gamma);
    auto dx_cu      = makeTransformerEngineTensor(dx);
    auto dgamma_cu  = makeTransformerEngineTensor(dgamma);
    auto dbeta_cu   = makeTransformerEngineTensor(dbeta);

    // This call populates tensors with the required config.
    nvte_layernorm_bwd(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                       dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(),
                       dbeta_part.data(), at::cuda::getCurrentCUDAStream(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
                       workspace.data(), barrier.data());

    // Alloc space for Tensors.
    auto workspace_data     = allocateSpace(workspace.shape(), workspace.dtype());
    auto barrier_data       = allocateSpace(barrier.shape(), barrier.dtype(), true);
    auto dgamma_part_data   = allocateSpace(dgamma_part.shape(), dgamma_part.dtype());
    auto dbeta_part_data    = allocateSpace(dbeta_part.shape(), dbeta_part.dtype());
    workspace   = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                              workspace.shape(),
                                              workspace.dtype());
    barrier     = makeTransformerEngineTensor(barrier_data.data_ptr(),
                                              barrier.shape(),
                                              barrier.dtype());
    dgamma_part = makeTransformerEngineTensor(dgamma_part_data.data_ptr(),
                                              dgamma_part.shape(),
                                              dgamma_part.dtype());
    dbeta_part  = makeTransformerEngineTensor(dbeta_part_data.data_ptr(),
                                              dbeta_part.shape(),
                                              dbeta_part.dtype());

    // Actual call to bwd kernel.
    nvte_layernorm_bwd(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                       dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(),
                       dbeta_part.data(), at::cuda::getCurrentCUDAStream(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
                       workspace.data(), barrier.data());

    return { dx, dgamma, dbeta };
}


std::vector<at::Tensor> layernorm_fwd_fp8(const at::Tensor &input,
                                          const at::Tensor &weight,
                                          const at::Tensor &bias,
                                          float eps,
                                          at::Tensor scale,
                                          at::Tensor amax,
                                          at::Tensor scale_inv,
                                          transformer_engine::DType otype
) {
    using namespace transformer_engine;

    size_t N = static_cast<size_t>(input.size(0));
    size_t H = static_cast<size_t>(input.size(1));

    DType itype = GetTransformerEngineDType(input.scalar_type());

    auto ln_out = at::empty_like(input, at::CUDA(GetATenDType(otype)));
    auto mu = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
    auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));

    dispatch_layernorm(
            input.data_ptr(), {N, H}, itype,
            weight.data_ptr(), {H}, itype,
            bias.data_ptr(), {H}, itype,
            scale.data_ptr(), {1}, DType::kFloat32,
            eps,
            ln_out.data_ptr(), {N, H}, otype,
            mu.data_ptr(), {N}, DType::kFloat32,
            rsigma.data_ptr(), {N}, DType::kFloat32,
            amax.data_ptr(), {1}, DType::kFloat32,
            scale_inv.data_ptr(), {1}, DType::kFloat32,
            at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
            true);

    return {ln_out, mu, rsigma};
}


std::vector<at::Tensor> layernorm_fwd(const at::Tensor &input,
                                      const at::Tensor &weight,
                                      const at::Tensor &bias,
                                      float eps
) {
    using namespace transformer_engine;

    size_t N = static_cast<size_t>(input.size(0));
    size_t H = static_cast<size_t>(input.size(1));

    DType itype = GetTransformerEngineDType(input.scalar_type());

    auto ln_out = at::empty_like(input, at::CUDA(GetATenDType(itype)));
    auto mu = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
    auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));

    dispatch_layernorm(input.data_ptr(), {N, H}, itype,
                       weight.data_ptr(), {H}, itype,
                       bias.data_ptr(), {H}, itype,
                       nullptr, {1}, DType::kFloat32,
                       eps,
                       ln_out.data_ptr(), {N, H}, itype,
                       mu.data_ptr(), {N}, DType::kFloat32,
                       rsigma.data_ptr(), {N}, DType::kFloat32,
                       nullptr, {1}, DType::kFloat32,
                       nullptr, {1}, DType::kFloat32,
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
                       false);

    return {ln_out, mu, rsigma};
}


at::Tensor cast_to_fp8(const at::Tensor &input,
                       const at::Tensor &scale,
                       at::Tensor amax,
                       at::Tensor scale_inv,
                       transformer_engine::DType otype
) {
    using namespace transformer_engine;
    size_t N = static_cast<size_t>(input.size(0));
    size_t H = static_cast<size_t>(input.size(1));

    auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

    auto input_cu     = makeTransformerEngineTensor(input);
    auto output_cu    = makeTransformerEngineTensor(output.data_ptr(), {N, H}, otype);
    auto scale_cu     = makeTransformerEngineTensor(scale.data_ptr(), {1}, DType::kFloat32);
    auto amax_cu      = makeTransformerEngineTensor(amax.data_ptr(), {1}, DType::kFloat32);
    auto scale_inv_cu = makeTransformerEngineTensor(scale_inv.data_ptr(), {1}, DType::kFloat32);

    nvte_fp8_quantize(input_cu.data(), scale_cu.data(), output_cu.data(),
                      amax_cu.data(), scale_inv_cu.data(),
                      at::cuda::getCurrentCUDAStream());

    return output;
}


at::Tensor cast_from_fp8(const at::Tensor &input,
                         const at::Tensor &scale_inv,
                         transformer_engine::DType itype,
                         transformer_engine::DType otype
) {
    using namespace transformer_engine;
    size_t N = static_cast<size_t>(input.size(0));
    size_t H = static_cast<size_t>(input.size(1));

    auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

    auto input_cu     = makeTransformerEngineTensor(input.data_ptr(), {N, H}, itype);
    auto output_cu    = makeTransformerEngineTensor(output);
    auto scale_inv_cu = makeTransformerEngineTensor(scale_inv.data_ptr(), {1}, DType::kFloat32);

    nvte_fp8_dequantize(input_cu.data(), scale_inv_cu.data(), output_cu.data(),
                        at::cuda::getCurrentCUDAStream());

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Granular functions
  m.def("layernorm_fwd_fp8", &layernorm_fwd_fp8, "LN FWD FP8");
  m.def("layernorm_bwd", &layernorm_bwd, "LN BWD");
  m.def("layernorm_fwd", &layernorm_fwd, "LN FWD");
  m.def("fused_cast_transpose", &fused_cast_transpose, "Fused Cast + Transpose");
  m.def("fused_cast_transpose_bgrad", &fused_cast_transpose_bgrad,
                                              "Fused Cast + Transpose + BGRAD");
  m.def("fused_cast_transpose_bgrad_dgelu", &fused_cast_transpose_bgrad_dgelu,
                                              "Fused Cast + Transpose + BGRAD + DGELU");
  m.def("cast_to_fp8", &cast_to_fp8, "Cast to FP8");
  m.def("cast_from_fp8", &cast_from_fp8, "Cast from FP8");
  m.def("te_gemm", &te_gemm, "CublasLt GEMM");
  m.def("fp8_transpose", &fp8_transpose, "Transpose with FP8 I/O");
  m.def("fp8_gelu", &fp8_gelu, "GeLU with FP8 output");

  // Data structures
  py::class_<transformer_engine::FP8TensorMeta>(m, "FP8TensorMeta")
    .def(py::init<>())
    .def_readwrite("scale", &transformer_engine::FP8TensorMeta::scale)
    .def_readwrite("scale_inv", &transformer_engine::FP8TensorMeta::scale_inv)
    .def_readwrite("amax_history", &transformer_engine::FP8TensorMeta::amax_history);

  py::enum_<transformer_engine::DType>(m, "DType")
    .value("kByte", transformer_engine::DType::kByte)
    .value("kInt32", transformer_engine::DType::kInt32)
    .value("kFloat32", transformer_engine::DType::kFloat32)
    .value("kFloat16", transformer_engine::DType::kFloat16)
    .value("kBFloat16", transformer_engine::DType::kBFloat16)
    .value("kFloat8E4M3", transformer_engine::DType::kFloat8E4M3)
    .value("kFloat8E5M2", transformer_engine::DType::kFloat8E5M2);

  py::enum_<transformer_engine::FP8FwdTensors>(m, "FP8FwdTensors")
    .value("GEMM1_INPUT", transformer_engine::FP8FwdTensors::GEMM1_INPUT)
    .value("GEMM1_WEIGHT", transformer_engine::FP8FwdTensors::GEMM1_WEIGHT)
    .value("GEMM2_INPUT", transformer_engine::FP8FwdTensors::GEMM2_INPUT)
    .value("GEMM2_WEIGHT", transformer_engine::FP8FwdTensors::GEMM2_WEIGHT);

  py::enum_<transformer_engine::FP8BwdTensors>(m, "FP8BwdTensors")
    .value("GRAD_OUTPUT1", transformer_engine::FP8BwdTensors::GRAD_OUTPUT1)
    .value("GRAD_OUTPUT2", transformer_engine::FP8BwdTensors::GRAD_OUTPUT2);
}
