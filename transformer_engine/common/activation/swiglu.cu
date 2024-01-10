/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/activation.h>
#include <cuda_runtime.h>
#include "../util/vectorized_pointwise.h"
#include "../util/math.h"
#include "../common.h"


namespace transformer_engine {

void swiglu(const Tensor &input,
            Tensor *output,
            cudaStream_t stream) {
  CheckInputTensor(input, "geglu_input");
  CheckOutputTensor(*output, "geglu_output");
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->data.shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(input.data.shape[0] == output->data.shape[0],
             "Input shape[0] must be equal to output shape[0].");
  NVTE_CHECK(input.data.shape[1] == output->data.shape[1] * 2,
             "Input shape[1] must be 2x larger than output shape[1].");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output->data.dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      GatedActivationKernelLauncher<nvec, fp32, Empty, swish<fp32, fp32>>(
        reinterpret_cast<const IType*>(input.data.dptr),
        reinterpret_cast<OType*>(output->data.dptr),
        reinterpret_cast<const fp32*>(output->scale.dptr),
        reinterpret_cast<fp32*>(output->amax.dptr),
        output->data.shape[0],
        output->data.shape[1],
        {},
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

void dswiglu(const Tensor &grad,
             const Tensor &input,
             Tensor *output,
             cudaStream_t stream) {
  CheckInputTensor(grad, "dswiglu_grad");
  CheckInputTensor(input, "dswiglu_input");
  CheckOutputTensor(*output, "dswiglu_output");
  NVTE_CHECK(grad.data.shape.size() == 2, "Grad must have 2 dimensions.");
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->data.shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(output->data.shape[0] == grad.data.shape[0],
             "Output shape[0] must be equal to grad shape[0].");
  NVTE_CHECK(output->data.shape[1] == grad.data.shape[1] * 2,
             "Output shape[1] must be 2x larger than grad shape[1].");
  NVTE_CHECK(input.data.shape == output->data.shape,
             "Input and output shapes must match.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output->data.dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      DGatedActivationKernelLauncher<nvec, fp32, Empty, swish<fp32, fp32>, dswish<fp32, fp32>>(
        reinterpret_cast<const IType*>(grad.data.dptr),
        reinterpret_cast<const IType*>(input.data.dptr),
        reinterpret_cast<OType*>(output->data.dptr),
        grad.data.shape[0],
        grad.data.shape[1],
        {},
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_swiglu(const NVTETensor input,
                 NVTETensor output,
                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_swiglu);
  using namespace transformer_engine;
  swiglu(*reinterpret_cast<const Tensor*>(input),
         reinterpret_cast<Tensor*>(output),
         stream);
}

void nvte_dswiglu(const NVTETensor grad,
                  const NVTETensor input,
                  NVTETensor output,
                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_dswiglu);
  using namespace transformer_engine;
  dswiglu(*reinterpret_cast<const Tensor*>(grad),
          *reinterpret_cast<const Tensor*>(input),
          reinterpret_cast<Tensor*>(output),
          stream);
}
