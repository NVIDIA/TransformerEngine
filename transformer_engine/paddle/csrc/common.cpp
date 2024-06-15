/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"

namespace transformer_engine {
namespace paddle_ext {

TensorWrapper MakeNvteTensor(const void *data_ptr, const std::vector<size_t> &shape,
                             const DType type) {
  return TensorWrapper(const_cast<void *>(data_ptr), shape, type);
}

TensorWrapper MakeNvteTensor(void *data_ptr, const NVTEShape &shape, const DType type) {
  return TensorWrapper(data_ptr, shape, type);
}

TensorWrapper MakeNvteTensor(void *data_ptr, const std::vector<size_t> &shape, const DType type,
                             void *amax_ptr, void *scale_ptr, void *scale_inv_ptr) {
  return TensorWrapper(data_ptr, shape, type, reinterpret_cast<float *>(amax_ptr),
                       reinterpret_cast<float *>(scale_ptr),
                       reinterpret_cast<float *>(scale_inv_ptr));
}

TensorWrapper MakeNvteTensor(paddle::Tensor &tensor) {  // NOLINT
  return MakeNvteTensor(tensor.data(), GetShapeArray(tensor), Paddle2NvteDType(tensor.dtype()));
}

TensorWrapper MakeNvteTensor(const paddle::Tensor &tensor) {
  return MakeNvteTensor(const_cast<void *>(tensor.data()), GetShapeArray(tensor),
                        Paddle2NvteDType(tensor.dtype()));
}

paddle::Tensor AllocateSpace(const NVTEShape &shape, const DType type, const paddle::Place &place,
                             bool init_to_zeros) {
  auto size = shape.ndim;
  if (size == 2 && init_to_zeros) {
    return paddle::zeros({static_cast<int64_t>(shape.data[0]), static_cast<int64_t>(shape.data[1])},
                         Nvte2PaddleDType(type), place);
  } else if (size == 2) {
    return paddle::empty({static_cast<int64_t>(shape.data[0]), static_cast<int64_t>(shape.data[1])},
                         Nvte2PaddleDType(type), place);
  } else if (size == 1 && init_to_zeros) {
    return paddle::zeros({static_cast<int64_t>(shape.data[0])}, Nvte2PaddleDType(type), place);
  } else if (size == 1) {
    return paddle::empty({static_cast<int64_t>(shape.data[0])}, Nvte2PaddleDType(type), place);
  }
  NVTE_CHECK(false, "Should never reach here! func: AllocateSpace");
}

// MHA utils
// convert QKV layout to enum
NVTE_QKV_Layout get_nvte_qkv_layout(const std::string &qkv_layout) {
  static const std::unordered_map<std::string, NVTE_QKV_Layout> layout_map = {
      {"sb3hd", NVTE_QKV_Layout::NVTE_SB3HD},
      {"sbh3d", NVTE_QKV_Layout::NVTE_SBH3D},
      {"sbhd_sb2hd", NVTE_QKV_Layout::NVTE_SBHD_SB2HD},
      {"sbhd_sbh2d", NVTE_QKV_Layout::NVTE_SBHD_SBH2D},
      {"sbhd_sbhd_sbhd", NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD},
      {"bs3hd", NVTE_QKV_Layout::NVTE_BS3HD},
      {"bsh3d", NVTE_QKV_Layout::NVTE_BSH3D},
      {"bshd_bs2hd", NVTE_QKV_Layout::NVTE_BSHD_BS2HD},
      {"bshd_bsh2d", NVTE_QKV_Layout::NVTE_BSHD_BSH2D},
      {"bshd_bshd_bshd", NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD},
      {"t3hd", NVTE_QKV_Layout::NVTE_T3HD},
      {"th3d", NVTE_QKV_Layout::NVTE_TH3D},
      {"thd_t2hd", NVTE_QKV_Layout::NVTE_THD_T2HD},
      {"thd_th2d", NVTE_QKV_Layout::NVTE_THD_TH2D},
      {"thd_thd_thd", NVTE_QKV_Layout::NVTE_THD_THD_THD},
  };

  auto it = layout_map.find(qkv_layout);
  if (it != layout_map.end()) {
    return it->second;
  } else {
    NVTE_ERROR("Invalid QKV layout string: " + qkv_layout);
  }
}

}  // namespace paddle_ext
}  // namespace transformer_engine
