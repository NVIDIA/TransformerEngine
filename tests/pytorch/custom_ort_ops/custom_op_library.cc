/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <exception>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include <cuda_fp8.h>

namespace {

template <typename IType, typename OType, typename CType>
void Quantize(OrtKernelContext* context,
              const Ort::Custom::Tensor<IType>& input,
              const Ort::Custom::Tensor<CType>& scale_inv,
              Ort::Custom::Tensor<unsigned char>& output) {
  auto raw_input = input.Data();
  auto raw_scale_inv = scale_inv.Data();
  auto raw_output = reinterpret_cast<OType*>(output.Allocate(input.Shape()));
  const auto rs = static_cast<CType>(raw_scale_inv[0]);
  const size_t N = input.NumberOfElement();
  for (size_t i = 0; i < N; ++i) {
    const auto x = static_cast<CType>(raw_input[i]);
    raw_output[i] = static_cast<OType>(x / rs);
  }
}

template <typename IType, typename OType, typename CType>
void Dequantize(OrtKernelContext* context,
                const Ort::Custom::Tensor<unsigned char>& input,
                const Ort::Custom::Tensor<CType>& scale_inv,
                Ort::Custom::Tensor<OType>& output) {
  auto raw_input = reinterpret_cast<const IType*>(input.Data());
  auto raw_scale_inv = scale_inv.Data();
  auto raw_output = output.Allocate(input.Shape());
  const auto rs = static_cast<CType>(raw_scale_inv[0]);
  const size_t N = input.NumberOfElement();
  for (size_t i = 0; i < N; ++i) {
    const auto x = rs * static_cast<CType>(raw_input[i]);
    raw_output[i] = static_cast<OType>(x);
  }
}

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

}  // namespace

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  // Namespace for custom ops
  static const char* c_OpDomain = "trt";

  // Construct custom ops
  static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> c_Quantize{
    Ort::Custom::CreateLiteCustomOp("TRT_FP8QuantizeLinear",
                                    "CPUExecutionProvider",
                                    Quantize<float, __nv_fp8_e4m3, float>)
  };
  static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> c_Dequantize{
    Ort::Custom::CreateLiteCustomOp("TRT_FP8DequantizeLinear",
                                    "CPUExecutionProvider",
                                    Dequantize<__nv_fp8_e4m3, float, float>)
  };

  // Register custom ops
  OrtStatus* result = nullptr;
  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(c_Quantize.get());
    domain.Add(c_Dequantize.get());
    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}
