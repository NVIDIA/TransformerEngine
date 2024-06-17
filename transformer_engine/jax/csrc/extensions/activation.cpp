/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/activation.h"

#include "jax/csrc/extensions.h"
#include "transformer_engine/transpose.h"

namespace transformer_engine {
namespace jax {

size_t get_activation_len(NVTE_Activation_Type activation_enum) {
  switch (activation_enum) {
    case NVTE_Activation_Type::GELU:
      return 1;
    case NVTE_Activation_Type::GEGLU:
      return 2;
    case NVTE_Activation_Type::SILU:
      return 1;
    case NVTE_Activation_Type::SWIGLU:
      return 2;
    case NVTE_Activation_Type::RELU:
      return 1;
    case NVTE_Activation_Type::REGLU:
      return 2;
    case NVTE_Activation_Type::QGELU:
      return 1;
    case NVTE_Activation_Type::QGEGLU:
      return 2;
    case NVTE_Activation_Type::SRELU:
      return 1;
    case NVTE_Activation_Type::SREGLU:
      return 2;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
      return -1;
  }
}

void ActLuImpl(void *input, size_t m, size_t n, DType in_dtype, DType out_dtype, float *scale,
               cudaStream_t stream, float *scale_inverse, float *amax, void *output,
               NVTE_Activation_Type act_enum) {
  auto act_len = get_activation_len(act_enum);
  auto input_shape = std::vector<size_t>{m, n * act_len};
  auto output_shape = std::vector<size_t>{m, n};
  auto input_tensor = TensorWrapper(input, input_shape, static_cast<DType>(in_dtype));
  auto output_tensor = TensorWrapper(output, output_shape, static_cast<DType>(out_dtype), amax,
                                     scale, scale_inverse);
  switch (act_enum) {
    case NVTE_Activation_Type::GELU:
      nvte_gelu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::GEGLU:
      nvte_geglu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SILU:
      nvte_silu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SWIGLU:
      nvte_swiglu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::RELU:
      nvte_relu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::REGLU:
      nvte_reglu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGELU:
      nvte_qgelu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGEGLU:
      nvte_qgeglu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SRELU:
      nvte_srelu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SREGLU:
      nvte_sreglu(input_tensor.data(), output_tensor.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }
}

void ActLu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  auto *output = buffers[1];

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto act_enum = static_cast<NVTE_Activation_Type>(desc.act_enum);
  ;

  ActLuImpl(input, m, n, desc.in_dtype, desc.out_dtype, nullptr, stream, nullptr, nullptr, output,
            act_enum);
}

void ActLuFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  float *amax = reinterpret_cast<float *>(buffers[1]);
  float *scale = reinterpret_cast<float *>(buffers[2]);
  float *scale_inv = reinterpret_cast<float *>(buffers[3]);
  auto *output = buffers[4];
  float *amax_out = reinterpret_cast<float *>(buffers[5]);
  NVTE_CHECK(amax == amax_out, "amax not bound to amax_out in TE/JAX ActLuFP8 primitive.");

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  if (!use_fp8(desc.out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto act_enum = static_cast<NVTE_Activation_Type>(desc.act_enum);
  ;

  ActLuImpl(input, m, n, desc.in_dtype, desc.out_dtype, scale, stream, scale_inv, amax_out, output,
            act_enum);
}

void DActLu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  auto *act_input = buffers[1];
  auto *output = buffers[2];

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto act_enum = static_cast<NVTE_Activation_Type>(desc.act_enum);
  ;

  auto act_len = get_activation_len(act_enum);
  auto input_shape = std::vector<size_t>{m, n};
  auto act_input_shape = std::vector<size_t>{m, n * act_len};
  auto output_shape = std::vector<size_t>{m, n * act_len};

  auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
  auto act_input_tensor = TensorWrapper(act_input, act_input_shape, desc.in_dtype);
  auto output_tensor = TensorWrapper(output, output_shape, desc.out_dtype);

  switch (act_enum) {
    case NVTE_Activation_Type::GELU:
      nvte_dgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::GEGLU:
      nvte_dgeglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SILU:
      nvte_dsilu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SWIGLU:
      nvte_dswiglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::RELU:
      nvte_drelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::REGLU:
      nvte_dreglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGELU:
      nvte_dqgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGEGLU:
      nvte_dqgeglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SRELU:
      nvte_dsrelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SREGLU:
      nvte_dsreglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }
}

pybind11::tuple GetDActDBiasCastTransposeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                        DType in_dtype, DType out_dtype) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto dact_input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_trans_shape = std::vector<size_t>{hidden_size, batch_size};
  auto dbias_shape = std::vector<size_t>{hidden_size};

  auto input_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
  auto dact_input_tensor = TensorWrapper(nullptr, dact_input_shape, in_dtype);
  auto output_tensor = TensorWrapper(nullptr, output_shape, out_dtype);
  auto output_trans_tensor = TensorWrapper(nullptr, output_trans_shape, out_dtype);
  auto dbias_tensor = TensorWrapper(nullptr, dbias_shape, in_dtype);

  TensorWrapper dummy_workspace;

  // For now, all dbias_dact(-s) have the same workspace size
  nvte_cast_transpose_dbias_dgelu(input_tensor.data(), dact_input_tensor.data(),
                                  output_tensor.data(), output_trans_tensor.data(),
                                  dbias_tensor.data(), dummy_workspace.data(), nullptr);

  auto work_shape = MakeShapeVector(dummy_workspace.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, dummy_workspace.dtype()));
}

void DActLuDBiasCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                              size_t opaque_len) {
  auto *input = buffers[0];
  auto *act_input = buffers[1];
  float *amax = reinterpret_cast<float *>(buffers[2]);
  float *scale = reinterpret_cast<float *>(buffers[3]);
  float *scale_inv = reinterpret_cast<float *>(buffers[4]);
  auto *output = buffers[5];
  auto *output_trans = buffers[6];
  auto *dbias = buffers[7];
  float *amax_out = reinterpret_cast<float *>(buffers[8]);
  void *workspace_ptr = buffers[9];

  const auto &desc = *UnpackOpaque<CustomCallCommonWkDescriptor>(opaque, opaque_len);
  NVTE_CHECK(amax == amax_out,
             "amax not bound to amax_out in TE/JAX DActLuDBiasCastTranspose primitive.");
  if (!use_fp8(desc.out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto act_enum = static_cast<NVTE_Activation_Type>(desc.act_enum);
  ;
  auto input_shape = std::vector<size_t>{m, n};
  auto act_input_shape = std::vector<size_t>{m, n};
  auto output_shape = std::vector<size_t>{m, n};
  auto output_trans_shape = std::vector<size_t>{n, m};
  auto dbias_shape = std::vector<size_t>{n};

  auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
  auto act_input_tensor = TensorWrapper(act_input, act_input_shape, desc.in_dtype);
  auto output_tensor =
      TensorWrapper(output, output_shape, desc.out_dtype, amax_out, scale, scale_inv);
  auto output_trans_tensor =
      TensorWrapper(output_trans, output_trans_shape, desc.out_dtype, amax_out, scale, scale_inv);
  auto dbias_tensor = TensorWrapper(dbias, dbias_shape, desc.in_dtype);

  auto workspace = TensorWrapper(workspace_ptr, desc.wkshape.to_vector(), desc.wk_dtype);

  switch (act_enum) {
    case NVTE_Activation_Type::GELU:
      nvte_cast_transpose_dbias_dgelu(input_tensor.data(), act_input_tensor.data(),
                                      output_tensor.data(), output_trans_tensor.data(),
                                      dbias_tensor.data(), workspace.data(), stream);
      break;
    case NVTE_Activation_Type::SILU:
      nvte_cast_transpose_dbias_dsilu(input_tensor.data(), act_input_tensor.data(),
                                      output_tensor.data(), output_trans_tensor.data(),
                                      dbias_tensor.data(), workspace.data(), stream);
      break;
    case NVTE_Activation_Type::RELU:
      nvte_cast_transpose_dbias_drelu(input_tensor.data(), act_input_tensor.data(),
                                      output_tensor.data(), output_trans_tensor.data(),
                                      dbias_tensor.data(), workspace.data(), stream);
      break;
    case NVTE_Activation_Type::QGELU:
      nvte_cast_transpose_dbias_dqgelu(input_tensor.data(), act_input_tensor.data(),
                                       output_tensor.data(), output_trans_tensor.data(),
                                       dbias_tensor.data(), workspace.data(), stream);
      break;
    case NVTE_Activation_Type::SRELU:
      nvte_cast_transpose_dbias_dsrelu(input_tensor.data(), act_input_tensor.data(),
                                       output_tensor.data(), output_trans_tensor.data(),
                                       dbias_tensor.data(), workspace.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }
}

void DGatedActLuCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                              size_t opaque_len) {
  auto *input = buffers[0];
  auto *act_input = buffers[1];
  float *amax = reinterpret_cast<float *>(buffers[2]);
  float *scale = reinterpret_cast<float *>(buffers[3]);
  float *scale_inv = reinterpret_cast<float *>(buffers[4]);
  auto *output = buffers[5];
  auto *output_trans = buffers[6];
  float *amax_out = reinterpret_cast<float *>(buffers[7]);

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  NVTE_CHECK(amax == amax_out,
             "amax not bound to amax_out in TE/JAX DGatedActLuCastTranspose primitive.");
  if (!use_fp8(desc.out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto act_enum = static_cast<NVTE_Activation_Type>(desc.act_enum);
  ;
  auto input_shape = desc.shape.to_vector();
  auto act_input_shape = std::vector<size_t>{m, n * 2};
  auto output_shape = std::vector<size_t>{m, n * 2};
  auto output_trans_shape = std::vector<size_t>{n * 2, m};

  auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
  auto act_input_tensor = TensorWrapper(act_input, act_input_shape, desc.in_dtype);
  auto output_tensor =
      TensorWrapper(output, output_shape, desc.out_dtype, amax_out, scale, scale_inv);
  auto output_trans_tensor =
      TensorWrapper(output_trans, output_trans_shape, desc.out_dtype, amax_out, scale, scale_inv);

  switch (act_enum) {
    case NVTE_Activation_Type::GEGLU:
      nvte_dgeglu_cast_transpose(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                 output_trans_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SWIGLU:
      nvte_dswiglu_cast_transpose(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), output_trans_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::REGLU:
      nvte_dreglu_cast_transpose(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                 output_trans_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGEGLU:
      nvte_dqgeglu_cast_transpose(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), output_trans_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SREGLU:
      nvte_dsreglu_cast_transpose(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), output_trans_tensor.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }
}

}  // namespace jax
}  // namespace transformer_engine
