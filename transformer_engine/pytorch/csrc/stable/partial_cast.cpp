/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/cast.h>
#include <transformer_engine/cast_transpose_noop.h>
#include <transformer_engine/recipe.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

// FP8 block scaling
void fp8_block_scaling_compute_partial_amax(Tensor tensor, Tensor amax, int64_t h, int64_t w,
                                            int64_t start_offset, int64_t block_len) {
  auto t_cu = makeTransformerEngineTensor(tensor);
  auto a_cu = makeTransformerEngineTensor(amax);
  nvte_fp8_block_scaling_compute_partial_amax(t_cu.data(), a_cu.data(), h, w, amax.stride(0),
                                              amax.stride(1), start_offset, block_len,
                                              getCurrentCUDAStreamRaw(tensor.get_device_index()));
}

void fp8_block_scaling_partial_cast(Tensor inp, Tensor out, Tensor scale, int64_t h, int64_t w,
                                    int64_t start_offset, int64_t block_len, int64_t out_dtype) {
  auto i_cu = makeTransformerEngineTensor(inp);
  auto o_cu = makeTransformerEngineTensor(out);
  auto s_cu = makeTransformerEngineTensor(scale);
  nvte_fp8_block_scaling_partial_cast(i_cu.data(), o_cu.data(), s_cu.data(), h, w, scale.stride(0),
                                      scale.stride(1), start_offset, block_len,
                                      static_cast<NVTEDType>(out_dtype),
                                      getCurrentCUDAStreamRaw(inp.get_device_index()));
}

// MXFP8 scaling
void mxfp8_scaling_compute_partial_amax(Tensor input, Tensor amax_rowwise, Tensor amax_colwise,
                                        int64_t rows, int64_t cols, int64_t start_offset) {
  auto i_cu = makeTransformerEngineTensor(input);
  auto ar_cu = makeTransformerEngineTensor(amax_rowwise);
  auto ac_cu = makeTransformerEngineTensor(amax_colwise);
  nvte_mxfp8_scaling_compute_partial_amax(i_cu.data(), ar_cu.data(), ac_cu.data(), rows, cols,
                                          start_offset,
                                          getCurrentCUDAStreamRaw(input.get_device_index()));
}

void mxfp8_scaling_partial_cast(Tensor input, Tensor output_rowwise, Tensor output_colwise,
                                Tensor scale_inv_rowwise, Tensor scale_inv_colwise, int64_t rows,
                                int64_t cols, int64_t start_offset) {
  auto i_cu = makeTransformerEngineTensor(input);
  auto or_cu = makeTransformerEngineTensor(output_rowwise);
  auto oc_cu = makeTransformerEngineTensor(output_colwise);
  auto sr_cu = makeTransformerEngineTensor(scale_inv_rowwise);
  auto sc_cu = makeTransformerEngineTensor(scale_inv_colwise);
  nvte_mxfp8_scaling_partial_cast(i_cu.data(), or_cu.data(), oc_cu.data(), sr_cu.data(),
                                  sc_cu.data(), rows, cols, start_offset,
                                  getCurrentCUDAStreamRaw(input.get_device_index()));
}

// NVFP4 2D
void nvfp4_2d_compute_partial_amax(Tensor tensor, Tensor amax, int64_t h, int64_t w,
                                   int64_t start_offset, int64_t block_len) {
  auto t_cu = makeTransformerEngineTensor(tensor);
  auto a_cu = makeTransformerEngineTensor(amax);
  nvte_nvfp4_2d_compute_partial_amax(t_cu.data(), a_cu.data(), h, w, amax.stride(0), amax.stride(1),
                                     start_offset, block_len,
                                     getCurrentCUDAStreamRaw(tensor.get_device_index()));
}

void nvfp4_2d_partial_cast_noalloc(Tensor inp, Tensor out_data, int64_t out_dtype,
                                   std::optional<Tensor> out_scale_inv, int64_t out_scaling_mode,
                                   Tensor scale, Tensor global_scale, int64_t h, int64_t w,
                                   int64_t start_offset, int64_t block_len) {
  auto i_cu = makeTransformerEngineTensor(inp);
  auto out_shape = getStableTensorShape(out_data);
  auto o_cu = makeQuantizedTensorWrapper(out_data, static_cast<DType>(out_dtype), out_shape,
                                         std::nullopt, std::nullopt, out_scale_inv,
                                         static_cast<NVTEScalingMode>(out_scaling_mode));
  auto s_cu = makeTransformerEngineTensor(scale);
  auto gs_cu = makeTransformerEngineTensor(global_scale);
  nvte_nvfp4_2d_partial_cast(i_cu.data(), o_cu.data(), s_cu.data(), gs_cu.data(), h, w,
                             scale.stride(0), scale.stride(1), start_offset, block_len,
                             getCurrentCUDAStreamRaw(inp.get_device_index()));
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def(
      "fp8_block_scaling_compute_partial_amax(Tensor tensor, Tensor amax, int h, int w, int "
      "start_offset, int block_len) -> ()");
  m.def(
      "fp8_block_scaling_partial_cast(Tensor inp, Tensor out, Tensor scale, int h, int w, int "
      "start_offset, int block_len, int out_dtype) -> ()");
  m.def(
      "mxfp8_scaling_compute_partial_amax(Tensor input, Tensor amax_rowwise, Tensor amax_colwise, "
      "int rows, int cols, int start_offset) -> ()");
  m.def(
      "mxfp8_scaling_partial_cast(Tensor input, Tensor output_rowwise, Tensor output_colwise, "
      "Tensor scale_inv_rowwise, Tensor scale_inv_colwise, int rows, int cols, int start_offset) "
      "-> ()");
  m.def(
      "nvfp4_2d_compute_partial_amax(Tensor tensor, Tensor amax, int h, int w, int start_offset, "
      "int block_len) -> ()");
  m.def(
      "nvfp4_2d_partial_cast_noalloc(Tensor inp, Tensor out_data, int out_dtype, Tensor? "
      "out_scale_inv, int out_scaling_mode, Tensor scale, Tensor global_scale, int h, int w, int "
      "start_offset, int block_len) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("fp8_block_scaling_compute_partial_amax",
         TORCH_BOX(fp8_block_scaling_compute_partial_amax));
  m.impl("fp8_block_scaling_partial_cast", TORCH_BOX(fp8_block_scaling_partial_cast));
  m.impl("mxfp8_scaling_compute_partial_amax", TORCH_BOX(mxfp8_scaling_compute_partial_amax));
  m.impl("mxfp8_scaling_partial_cast", TORCH_BOX(mxfp8_scaling_partial_cast));
  m.impl("nvfp4_2d_compute_partial_amax", TORCH_BOX(nvfp4_2d_compute_partial_amax));
  m.impl("nvfp4_2d_partial_cast_noalloc", TORCH_BOX(nvfp4_2d_partial_cast_noalloc));
}
