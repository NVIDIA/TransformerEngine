/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/cudnn.h"

#include "extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

Error_Type CudnnHandleInitFFI(Variadic_Buffer_Type args, Variadic_Result_Type rets,
                              Dictionary attrs) {
  nvte_cudnn_handle_init();
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CudnnHandleInitHandler, CudnnHandleInitFFI,
                              FFI::Bind<FFI_Prepare>().RemainingArgs().RemainingRets().Attrs());
}  // namespace jax
}  // namespace transformer_engine
