/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/activation.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>
#include "../utils.cuh"
#include "../common.h"
#include <cstdlib>
#include <../util/vectorized_pointwise.h>
#include "../util/math.h"

namespace transformer_engine {

void gelu(const Tensor &input,
          Tensor *output,
          cudaStream_t stream) {
  CheckInputTensor(input, "gelu_input");
  CheckOutputTensor(*output, "gelu_output");
  NVTE_CHECK(input.data.shape == output->data.shape, "Input and output shapes must match.");
  const size_t tot_elts = product(input.data.shape);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output->data.dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      VectorizedUnaryKernelLauncher<nvec, Empty, gelu<fp32, fp32> >(
        reinterpret_cast<const IType*>(input.data.dptr),
        reinterpret_cast<OType*>(output->data.dptr),
        reinterpret_cast<const fp32*>(output->scale.dptr),
        reinterpret_cast<fp32*>(output->amax.dptr),
        tot_elts,
        Empty(),
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

void dgelu(const Tensor &grad,
           const Tensor &input,
           Tensor *output,
           cudaStream_t stream) {
  CheckInputTensor(input, "dgelu_input");
  CheckInputTensor(grad, "dgelu_input_grad");
  CheckOutputTensor(*output, "dgelu_output");
  NVTE_CHECK(input.data.shape == output->data.shape, "Input and output shapes must match.");
  NVTE_CHECK(input.data.dtype == grad.data.dtype,
             "Input and incoming gradient types must match.");
  const size_t tot_elts = product(input.data.shape);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output->data.dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      VectorizedUnaryGradKernelLauncher<nvec, Empty, dgelu<fp32, fp32>>(
        reinterpret_cast<const IType*>(grad.data.dptr),
        reinterpret_cast<const IType*>(input.data.dptr),
        reinterpret_cast<OType*>(output->data.dptr),
        reinterpret_cast<const fp32*>(output->scale.dptr),
        reinterpret_cast<fp32*>(output->amax.dptr),
        tot_elts,
        {},
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

void geglu(const Tensor &input,
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
      GatedActivationKernelLauncher<nvec, fp32, Empty, gelu<fp32, fp32>>(
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

void dgeglu(const Tensor &grad,
            const Tensor &input,
            Tensor *output,
            cudaStream_t stream) {
  CheckInputTensor(grad, "dgeglu_grad");
  CheckInputTensor(input, "dgeglu_input");
  CheckOutputTensor(*output, "dgeglu_output");
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
      DGatedActivationKernelLauncher<nvec, fp32, Empty, gelu<fp32, fp32>, dgelu<fp32, fp32>>(
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

void qgelu(const Tensor &input,
  Tensor *output,
  cudaStream_t stream) {
  CheckInputTensor(input, "qgelu_input");
  CheckOutputTensor(*output, "qgelu_output");
  NVTE_CHECK(input.data.shape == output->data.shape, "Input and output shapes must match.");
  const size_t tot_elts = product(input.data.shape);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output->data.dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      VectorizedUnaryKernelLauncher<nvec, Empty, qgelu<fp32, fp32> >(
        reinterpret_cast<const IType*>(input.data.dptr),
        reinterpret_cast<OType*>(output->data.dptr),
        reinterpret_cast<const fp32*>(output->scale.dptr),
        reinterpret_cast<fp32*>(output->amax.dptr),
        tot_elts,
        Empty(),
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

void dqgelu(const Tensor &grad,
   const Tensor &input,
   Tensor *output,
   cudaStream_t stream) {
  CheckInputTensor(input, "dqgelu_input");
  CheckInputTensor(grad, "dqgelu_input_grad");
  CheckOutputTensor(*output, "dqgelu_output");
  NVTE_CHECK(input.data.shape == output->data.shape, "Input and output shapes must match.");
  NVTE_CHECK(input.data.dtype == grad.data.dtype,
      "Input and incoming gradient types must match.");
  const size_t tot_elts = product(input.data.shape);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(output->data.dtype, OType,
      constexpr int nvec = 32 / sizeof(IType);
      VectorizedUnaryGradKernelLauncher<nvec, Empty, dqgelu<fp32, fp32>>(
        reinterpret_cast<const IType*>(grad.data.dptr),
        reinterpret_cast<const IType*>(input.data.dptr),
        reinterpret_cast<OType*>(output->data.dptr),
        reinterpret_cast<const fp32*>(output->scale.dptr),
        reinterpret_cast<fp32*>(output->amax.dptr),
        tot_elts,
        {},
        stream);
    );  // NOLINT(*)
  );  // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_gelu(const NVTETensor input,
               NVTETensor output,
               cudaStream_t stream) {
  NVTE_API_CALL(nvte_gelu);
  using namespace transformer_engine;
  gelu(*reinterpret_cast<const Tensor*>(input),
       reinterpret_cast<Tensor*>(output),
       stream);
}

void nvte_dgelu(const NVTETensor grad,
                const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_dgelu);
  using namespace transformer_engine;
  dgelu(*reinterpret_cast<const Tensor*>(grad),
        *reinterpret_cast<const Tensor*>(input),
        reinterpret_cast<Tensor*>(output),
        stream);
}

void nvte_geglu(const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_geglu);
  using namespace transformer_engine;
  geglu(*reinterpret_cast<const Tensor*>(input),
        reinterpret_cast<Tensor*>(output),
        stream);
}

void nvte_dgeglu(const NVTETensor grad,
                 const NVTETensor input,
                 NVTETensor output,
                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_dgeglu);
  using namespace transformer_engine;
  dgeglu(*reinterpret_cast<const Tensor*>(grad),
         *reinterpret_cast<const Tensor*>(input),
         reinterpret_cast<Tensor*>(output),
         stream);
}

void nvte_qgelu(const NVTETensor input,
  NVTETensor output,
  cudaStream_t stream) {
  NVTE_API_CALL(nvte_qgelu);
  using namespace transformer_engine;
  qgelu(*reinterpret_cast<const Tensor*>(input),
        reinterpret_cast<Tensor*>(output),
        stream);
}

void nvte_dqgelu(const NVTETensor grad,
   const NVTETensor input,
   NVTETensor output,
   cudaStream_t stream) {
  NVTE_API_CALL(nvte_dqgelu);
  using namespace transformer_engine;
  dqgelu(*reinterpret_cast<const Tensor*>(grad),
        *reinterpret_cast<const Tensor*>(input),
        reinterpret_cast<Tensor*>(output),
        stream);
}
