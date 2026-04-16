/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/air_topk.h"

#include "../extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

// ---------------------------------------------------------------------------
// JAX FFI handler
// ---------------------------------------------------------------------------

Error_Type AirTopkFFI(cudaStream_t stream, Buffer_Type keys_in_buf, Buffer_Type lengths_buf,
                      Result_Type keys_out_buf, Result_Type indices_out_buf,
                      Result_Type workspace_buf, int64_t k_value, int64_t workbuf_bytes) {
  auto keys_in_dtype = convert_ffi_datatype_to_te_dtype(keys_in_buf.element_type());
  auto keys_out_dtype = convert_ffi_datatype_to_te_dtype(keys_out_buf->element_type());
  auto idx_out_dtype = convert_ffi_datatype_to_te_dtype(indices_out_buf->element_type());
  NVTE_CHECK(keys_in_dtype == keys_out_dtype, "AirTopkFFI: input and output key dtypes must match");
  NVTE_CHECK(idx_out_dtype == DType::kInt32, "AirTopkFFI: index output must be int32");

  auto keys_in_shape = keys_in_buf.dimensions();
  NVTE_CHECK(keys_in_shape.size() == 2, "AirTopkFFI: keys input must be 2D (batch_size, seq_len)");

  int batch_size = static_cast<int>(keys_in_shape[0]);
  int seq_len = static_cast<int>(keys_in_shape[1]);
  int k = static_cast<int>(k_value);

  // Element byte widths for computing flat buffer sizes.
  size_t keys_element_bytes;
  switch (keys_in_dtype) {
    case DType::kFloat32:
      keys_element_bytes = 4;
      break;
    case DType::kBFloat16:
      keys_element_bytes = 2;
      break;
    default:
      NVTE_ERROR("AirTopkFFI: unsupported key dtype (float32 and bfloat16 only)");
  }

  // Build flat TensorWrappers over the full (batch_size * seq_len) / (batch_size * k) buffers.
  auto flat_in_shape =
      std::vector<size_t>{static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len)};
  auto flat_out_shape =
      std::vector<size_t>{static_cast<size_t>(batch_size) * static_cast<size_t>(k)};
  auto len_shape = std::vector<size_t>{static_cast<size_t>(batch_size)};
  auto ws_shape = std::vector<size_t>{static_cast<size_t>(workbuf_bytes)};

  auto keys_in_tensor = TensorWrapper(keys_in_buf.untyped_data(), flat_in_shape, keys_in_dtype);
  auto lengths_tensor = TensorWrapper(lengths_buf.untyped_data(), len_shape, DType::kInt32);
  auto keys_out_tensor =
      TensorWrapper(keys_out_buf->untyped_data(), flat_out_shape, keys_out_dtype);
  auto idx_out_tensor =
      TensorWrapper(indices_out_buf->untyped_data(), flat_out_shape, DType::kInt32);
  auto workspace_tensor = TensorWrapper(workspace_buf->untyped_data(), ws_shape, DType::kByte);

  nvte_air_topk(stream, keys_in_tensor.data(), lengths_tensor.data(), keys_out_tensor.data(),
                idx_out_tensor.data(), workspace_tensor.data(), batch_size, seq_len, k,
                static_cast<size_t>(workbuf_bytes));

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(AirTopkHandler, AirTopkFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // keys_in
                                  .Arg<Buffer_Type>()      // lengths
                                  .Ret<Buffer_Type>()      // keys_out
                                  .Ret<Buffer_Type>()      // indices_out
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<int64_t>("k_value")
                                  .Attr<int64_t>("workbuf_bytes"),
                              FFI_CudaGraph_Traits);

// ---------------------------------------------------------------------------
// Workspace-size query exposed to Python
// ---------------------------------------------------------------------------

int64_t GetAirTopkWorkspaceBytes(int batch_size, int seq_len, int k) {
  return static_cast<int64_t>(nvte_get_air_topk_workspace_bytes(batch_size, seq_len, k));
}

}  // namespace jax
}  // namespace transformer_engine
