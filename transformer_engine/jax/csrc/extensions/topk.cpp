/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/topk.h"

#include "../extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

// ---------------------------------------------------------------------------
// JAX FFI handler
// ---------------------------------------------------------------------------

Error_Type TopkFFI(cudaStream_t stream, Buffer_Type keys_in_buf, Buffer_Type lengths_buf,
                   Result_Type keys_out_buf, Result_Type indices_out_buf, Result_Type workspace_buf,
                   int64_t k_value) {
  auto keys_in_dtype = convert_ffi_datatype_to_te_dtype(keys_in_buf.element_type());
  auto keys_out_dtype = convert_ffi_datatype_to_te_dtype(keys_out_buf->element_type());
  auto idx_out_dtype = convert_ffi_datatype_to_te_dtype(indices_out_buf->element_type());
  NVTE_CHECK(keys_in_dtype == keys_out_dtype, "TopkFFI: input and output key dtypes must match");
  NVTE_CHECK(idx_out_dtype == DType::kInt32, "TopkFFI: index output must be int32");

  auto keys_in_shape = keys_in_buf.dimensions();
  NVTE_CHECK(keys_in_shape.size() == 2, "TopkFFI: keys input must be 2D (batch_size, seq_len)");

  int batch_size = static_cast<int>(keys_in_shape[0]);
  int seq_len = static_cast<int>(keys_in_shape[1]);
  int k = static_cast<int>(k_value);

  // Validate key dtype (float32 and bfloat16 only).
  switch (keys_in_dtype) {
    case DType::kFloat32:
    case DType::kBFloat16:
      break;
    default:
      NVTE_ERROR("TopkFFI: unsupported key dtype (float32 and bfloat16 only)");
  }

  auto workbuf_bytes = product(workspace_buf->dimensions());

  // Build flat TensorWrappers over the full (batch_size * seq_len) / (batch_size * k) buffers.
  auto flat_in_shape =
      std::vector<size_t>{static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len)};
  auto flat_out_shape =
      std::vector<size_t>{static_cast<size_t>(batch_size) * static_cast<size_t>(k)};
  auto len_shape = std::vector<size_t>{static_cast<size_t>(batch_size)};
  auto ws_shape = std::vector<size_t>{workbuf_bytes};

  auto keys_in_tensor = TensorWrapper(keys_in_buf.untyped_data(), flat_in_shape, keys_in_dtype);
  auto lengths_tensor = TensorWrapper(lengths_buf.untyped_data(), len_shape, DType::kInt32);
  auto keys_out_tensor =
      TensorWrapper(keys_out_buf->untyped_data(), flat_out_shape, keys_out_dtype);
  auto idx_out_tensor =
      TensorWrapper(indices_out_buf->untyped_data(), flat_out_shape, DType::kInt32);
  auto workspace_tensor = TensorWrapper(workspace_buf->untyped_data(), ws_shape, DType::kByte);

  nvte_topk(stream, keys_in_tensor.data(), lengths_tensor.data(), keys_out_tensor.data(),
            idx_out_tensor.data(), workspace_tensor.data(), batch_size, seq_len, k);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TopkHandler, TopkFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // keys_in
                                  .Arg<Buffer_Type>()      // lengths
                                  .Ret<Buffer_Type>()      // keys_out
                                  .Ret<Buffer_Type>()      // indices_out
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<int64_t>("k_value"),
                              FFI_CudaGraph_Traits);

// ---------------------------------------------------------------------------
// Workspace-size query exposed to Python
// ---------------------------------------------------------------------------

pybind11::tuple GetTopkWorkspaceSizes(int batch_size, int seq_len, int k) {
  auto flat_in_shape =
      std::vector<size_t>{static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len)};
  auto flat_out_shape =
      std::vector<size_t>{static_cast<size_t>(batch_size) * static_cast<size_t>(k)};
  auto len_shape = std::vector<size_t>{static_cast<size_t>(batch_size)};

  auto keys_in_tensor = TensorWrapper(nullptr, flat_in_shape, DType::kFloat32);
  auto lengths_tensor = TensorWrapper(nullptr, len_shape, DType::kInt32);
  auto keys_out_tensor = TensorWrapper(nullptr, flat_out_shape, DType::kFloat32);
  auto idx_out_tensor = TensorWrapper(nullptr, flat_out_shape, DType::kInt32);
  TensorWrapper workspace_tensor;

  nvte_topk(nullptr, keys_in_tensor.data(), lengths_tensor.data(), keys_out_tensor.data(),
            idx_out_tensor.data(), workspace_tensor.data(), batch_size, seq_len, k);

  auto work_shape = MakeShapeVector(workspace_tensor.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, workspace_tensor.dtype()));
}

}  // namespace jax
}  // namespace transformer_engine
