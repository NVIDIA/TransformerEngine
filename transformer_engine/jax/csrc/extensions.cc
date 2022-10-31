/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/include/transformer_engine/transformer_engine.h"
#include "jax/csrc/modules.h"
#include "jax/csrc/utils.h"

namespace te_jax = ::transformer_engine::jax;

namespace {

pybind11::dict TERegistrations() {
  pybind11::dict dict;
  dict["te_transpose"] = te_jax::EncapsulateFunction(te_jax::TETranspose);
  dict["te_cast_transpose"] =
      te_jax::EncapsulateFunction(te_jax::TECastTranspose);
  dict["te_gated_gelu"] = te_jax::EncapsulateFunction(te_jax::TEGatedGelu);
  dict["te_cast_transpose_dgated_gelu"] =
      te_jax::EncapsulateFunction(te_jax::TECastTransposeDGatedGelu);
  dict["te_gemm"] = te_jax::EncapsulateFunction(te_jax::TEGemm);
  dict["te_rmsnorm_forward"] =
      te_jax::EncapsulateFunction(te_jax::TERMSNormForward);
  dict["te_rmsnorm_forward_fp8"] =
      te_jax::EncapsulateFunction(te_jax::TERMSNormForwardFP8);
  dict["te_rmsnorm_backward"] =
      te_jax::EncapsulateFunction(te_jax::TERMSNormBackward);
  return dict;
}

PYBIND11_MODULE(transformer_engine_jax, m) {
  m.def("te_registrations", &TERegistrations);
  m.def("build_te_mat_descriptor", &te_jax::PackTEMatDescriptor);
  m.def("build_te_gemm_descriptor", &te_jax::PackTEGemmDescriptor);
  m.def("build_te_rmsnorm_descriptor", &te_jax::PackRMSNormDescriptor);

  pybind11::enum_<transformer_engine::DType>(m, "DType")
      .value("kByte", transformer_engine::DType::kByte)
      .value("kInt32", transformer_engine::DType::kInt32)
      .value("kFloat32", transformer_engine::DType::kFloat32)
      .value("kFloat16", transformer_engine::DType::kFloat16)
      .value("kBFloat16", transformer_engine::DType::kBFloat16)
      .value("kFloat8E4M3", transformer_engine::DType::kFloat8E4M3)
      .value("kFloat8E5M2", transformer_engine::DType::kFloat8E5M2);
}
}  // namespace
