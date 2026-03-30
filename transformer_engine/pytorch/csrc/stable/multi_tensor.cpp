/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/multi_tensor.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

// ============================================================================
// Multi-tensor helper: reconstruct NVTETensor** from flat pointer/shape tensors
//
// Python packs tensor_lists as:
//   ptrs:   int64 tensor [num_lists * num_tensors] — data_ptr() for each tensor
//   shapes: int64 tensor [num_lists * num_tensors * 2] — (numel, element_size)
//   dtypes: int64 tensor [num_lists * num_tensors] — TE DType values
//
// C++ reconstructs the 2D NVTETensor** structure.
// ============================================================================

namespace {

struct MultiTensorPack {
  std::vector<TensorWrapper> wrappers;
  std::vector<std::vector<NVTETensor>> lists;
  std::vector<NVTETensor*> list_ptrs;

  void build(const Tensor& ptrs, const Tensor& shapes, const Tensor& dtypes, int64_t num_lists,
             int64_t num_tensors) {
    auto ptrs_cpu = ptrs;  // already on CPU or we read via data_ptr
    auto shapes_cpu = shapes;
    auto dtypes_cpu = dtypes;

    const int64_t* p = static_cast<const int64_t*>(ptrs_cpu.data_ptr());
    const int64_t* s = static_cast<const int64_t*>(shapes_cpu.data_ptr());
    const int64_t* d = static_cast<const int64_t*>(dtypes_cpu.data_ptr());

    wrappers.reserve(num_lists * num_tensors);
    lists.resize(num_lists);

    for (int64_t li = 0; li < num_lists; ++li) {
      lists[li].reserve(num_tensors);
      for (int64_t ti = 0; ti < num_tensors; ++ti) {
        int64_t idx = li * num_tensors + ti;
        void* data = reinterpret_cast<void*>(p[idx]);
        size_t numel = static_cast<size_t>(s[idx * 2]);
        auto dtype = static_cast<DType>(d[idx]);
        wrappers.emplace_back(makeTransformerEngineTensor(data, std::vector<size_t>{numel}, dtype));
        lists[li].push_back(wrappers.back().data());
      }
    }

    list_ptrs.reserve(num_lists);
    for (auto& l : lists) {
      list_ptrs.push_back(l.data());
    }
  }
};

}  // namespace

// ============================================================================
// Multi-tensor scale
// ============================================================================

void multi_tensor_scale(int64_t chunk_size, Tensor is_infinite, Tensor ptrs, Tensor shapes,
                        Tensor dtypes, int64_t num_lists, int64_t num_tensors, double scale) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);

  auto is_inf_cu = makeTransformerEngineTensor(is_infinite);
  nvte_multi_tensor_scale_cuda(static_cast<int>(chunk_size), is_inf_cu.data(),
                               pack.list_ptrs.data(), static_cast<size_t>(num_lists),
                               static_cast<size_t>(num_tensors), static_cast<float>(scale),
                               getCurrentCUDAStreamRaw(is_infinite.get_device_index()));
}

void multi_tensor_scale_tensor(int64_t chunk_size, Tensor is_infinite, Tensor ptrs, Tensor shapes,
                               Tensor dtypes, int64_t num_lists, int64_t num_tensors,
                               Tensor scale) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);

  auto is_inf_cu = makeTransformerEngineTensor(is_infinite);
  auto scale_cu = makeTransformerEngineTensor(scale);
  nvte_multi_tensor_scale_tensor_cuda(static_cast<int>(chunk_size), is_inf_cu.data(),
                                      pack.list_ptrs.data(), static_cast<size_t>(num_lists),
                                      static_cast<size_t>(num_tensors), scale_cu.data(),
                                      getCurrentCUDAStreamRaw(is_infinite.get_device_index()));
}

// ============================================================================
// Multi-tensor L2 norm
// ============================================================================

std::tuple<Tensor, Tensor> multi_tensor_l2norm(int64_t chunk_size, Tensor noop_flag, Tensor ptrs,
                                               Tensor shapes, Tensor dtypes, int64_t num_lists,
                                               int64_t num_tensors, bool per_tensor) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);

  auto device_idx = noop_flag.get_device_index();
  auto noop_cu = makeTransformerEngineTensor(noop_flag);

  // Max chunks per tensor
  int max_chunks_per_tensor = -1;
  const int64_t* s = static_cast<const int64_t*>(shapes.data_ptr());
  for (int64_t ti = 0; ti < num_tensors; ++ti) {
    int chunks = (static_cast<int>(s[ti * 2]) + chunk_size - 1) / static_cast<int>(chunk_size);
    if (chunks > max_chunks_per_tensor) max_chunks_per_tensor = chunks;
  }

  auto output = allocateStableTensorZeros({320}, ScalarType::Float, device_idx);
  auto ret = allocateStableTensor({1}, ScalarType::Float, device_idx);
  auto output_per_tensor =
      per_tensor
          ? allocateStableTensorZeros({static_cast<int64_t>(num_tensors) * max_chunks_per_tensor},
                                      ScalarType::Float, device_idx)
          : allocateStableTensor({1}, ScalarType::Float, device_idx);
  auto ret_per_tensor = per_tensor ? allocateStableTensor({static_cast<int64_t>(num_tensors)},
                                                          ScalarType::Float, device_idx)
                                   : allocateStableTensor({1}, ScalarType::Float, device_idx);

  auto output_cu = makeTransformerEngineTensor(output);
  auto ret_cu = makeTransformerEngineTensor(ret);
  auto opt_cu = makeTransformerEngineTensor(output_per_tensor);
  auto rpt_cu = makeTransformerEngineTensor(ret_per_tensor);

  nvte_multi_tensor_l2norm_cuda(static_cast<int>(chunk_size), noop_cu.data(), pack.list_ptrs.data(),
                                static_cast<size_t>(num_lists), static_cast<size_t>(num_tensors),
                                output_cu.data(), opt_cu.data(), ret_cu.data(), rpt_cu.data(),
                                per_tensor, max_chunks_per_tensor,
                                getCurrentCUDAStreamRaw(device_idx));

  return std::make_tuple(ret, ret_per_tensor);
}

// ============================================================================
// Multi-tensor Adam
// ============================================================================

void multi_tensor_adam(int64_t chunk_size, Tensor noop_flag, Tensor ptrs, Tensor shapes,
                       Tensor dtypes, int64_t num_lists, int64_t num_tensors, double lr,
                       double beta1, double beta2, double epsilon, int64_t step, int64_t mode,
                       int64_t bias_correction, double weight_decay) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);
  auto noop_cu = makeTransformerEngineTensor(noop_flag);

  nvte_multi_tensor_adam_cuda(
      static_cast<int>(chunk_size), noop_cu.data(), pack.list_ptrs.data(),
      static_cast<size_t>(num_lists), static_cast<size_t>(num_tensors), static_cast<float>(lr),
      static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon),
      static_cast<int>(step), static_cast<int>(mode), static_cast<int>(bias_correction),
      static_cast<float>(weight_decay), getCurrentCUDAStreamRaw(noop_flag.get_device_index()));
}

void multi_tensor_adam_capturable(int64_t chunk_size, Tensor noop_flag, Tensor ptrs, Tensor shapes,
                                  Tensor dtypes, int64_t num_lists, int64_t num_tensors, Tensor lr,
                                  double beta1, double beta2, double epsilon, Tensor step,
                                  int64_t mode, int64_t bias_correction, double weight_decay,
                                  Tensor inv_scale) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);
  auto noop_cu = makeTransformerEngineTensor(noop_flag);
  auto lr_cu = makeTransformerEngineTensor(lr);
  auto step_cu = makeTransformerEngineTensor(step);
  auto inv_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_adam_capturable_cuda(
      static_cast<int>(chunk_size), noop_cu.data(), pack.list_ptrs.data(),
      static_cast<size_t>(num_lists), static_cast<size_t>(num_tensors), lr_cu.data(),
      static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon),
      step_cu.data(), static_cast<int>(mode), static_cast<int>(bias_correction),
      static_cast<float>(weight_decay), inv_cu.data(),
      getCurrentCUDAStreamRaw(noop_flag.get_device_index()));
}

// ============================================================================
// Multi-tensor SGD
// ============================================================================

void multi_tensor_sgd(int64_t chunk_size, Tensor noop_flag, Tensor ptrs, Tensor shapes,
                      Tensor dtypes, int64_t num_lists, int64_t num_tensors, double wd,
                      double momentum, double dampening, double lr, bool nesterov, bool first_run,
                      bool wd_after_momentum, double scale) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);
  auto noop_cu = makeTransformerEngineTensor(noop_flag);

  nvte_multi_tensor_sgd_cuda(static_cast<int>(chunk_size), noop_cu.data(), pack.list_ptrs.data(),
                             static_cast<size_t>(num_lists), static_cast<size_t>(num_tensors),
                             static_cast<float>(wd), static_cast<float>(momentum),
                             static_cast<float>(dampening), static_cast<float>(lr), nesterov,
                             first_run, wd_after_momentum, static_cast<float>(scale),
                             getCurrentCUDAStreamRaw(noop_flag.get_device_index()));
}

// ============================================================================
// Remaining Adam variants
// ============================================================================

void multi_tensor_adam_param_remainder(int64_t chunk_size, Tensor noop_flag, Tensor ptrs,
                                       Tensor shapes, Tensor dtypes, int64_t num_lists,
                                       int64_t num_tensors, double lr, double beta1, double beta2,
                                       double epsilon, int64_t step, int64_t mode,
                                       int64_t bias_correction, double weight_decay) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);
  auto noop_cu = makeTransformerEngineTensor(noop_flag);
  nvte_multi_tensor_adam_param_remainder_cuda(
      static_cast<int>(chunk_size), noop_cu.data(), pack.list_ptrs.data(),
      static_cast<size_t>(num_lists), static_cast<size_t>(num_tensors), static_cast<float>(lr),
      static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon),
      static_cast<int>(step), static_cast<int>(mode), static_cast<int>(bias_correction),
      static_cast<float>(weight_decay), getCurrentCUDAStreamRaw(noop_flag.get_device_index()));
}

void multi_tensor_adam_fp8(int64_t chunk_size, Tensor noop_flag, Tensor ptrs, Tensor shapes,
                           Tensor dtypes, int64_t num_lists, int64_t num_tensors, double lr,
                           double beta1, double beta2, double epsilon, int64_t step, int64_t mode,
                           int64_t bias_correction, double weight_decay, int64_t fp8_dtype) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);
  auto noop_cu = makeTransformerEngineTensor(noop_flag);
  nvte_multi_tensor_adam_fp8_cuda(
      static_cast<int>(chunk_size), noop_cu.data(), pack.list_ptrs.data(),
      static_cast<size_t>(num_lists), static_cast<size_t>(num_tensors), static_cast<float>(lr),
      static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon),
      static_cast<int>(step), static_cast<int>(mode), static_cast<int>(bias_correction),
      static_cast<float>(weight_decay), static_cast<NVTEDType>(fp8_dtype),
      getCurrentCUDAStreamRaw(noop_flag.get_device_index()));
}

void multi_tensor_adam_capturable_master(int64_t chunk_size, Tensor noop_flag, Tensor ptrs,
                                         Tensor shapes, Tensor dtypes, int64_t num_lists,
                                         int64_t num_tensors, Tensor lr, double beta1, double beta2,
                                         double epsilon, Tensor step, int64_t mode,
                                         int64_t bias_correction, double weight_decay,
                                         Tensor inv_scale) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);
  auto noop_cu = makeTransformerEngineTensor(noop_flag);
  auto lr_cu = makeTransformerEngineTensor(lr);
  auto step_cu = makeTransformerEngineTensor(step);
  auto inv_cu = makeTransformerEngineTensor(inv_scale);
  nvte_multi_tensor_adam_capturable_master_cuda(
      static_cast<int>(chunk_size), noop_cu.data(), pack.list_ptrs.data(),
      static_cast<size_t>(num_lists), static_cast<size_t>(num_tensors), lr_cu.data(),
      static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon),
      step_cu.data(), static_cast<int>(mode), static_cast<int>(bias_correction),
      static_cast<float>(weight_decay), inv_cu.data(),
      getCurrentCUDAStreamRaw(noop_flag.get_device_index()));
}

// ============================================================================
// Multi-tensor scale computation
// ============================================================================

void multi_tensor_compute_scale_and_scale_inv(int64_t chunk_size, Tensor noop_flag, Tensor ptrs,
                                              Tensor shapes, Tensor dtypes, int64_t num_lists,
                                              int64_t num_tensors, double max_fp8,
                                              bool force_pow_2_scales, double epsilon) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);
  auto noop_cu = makeTransformerEngineTensor(noop_flag);
  nvte_multi_tensor_compute_scale_and_scale_inv_cuda(
      static_cast<int>(chunk_size), noop_cu.data(), pack.list_ptrs.data(),
      static_cast<size_t>(num_lists), static_cast<size_t>(num_tensors), static_cast<float>(max_fp8),
      force_pow_2_scales, static_cast<float>(epsilon),
      getCurrentCUDAStreamRaw(noop_flag.get_device_index()));
}

void multi_tensor_compute_scale_inv_e8m0(int64_t chunk_size,
                                         Tensor dummy_cuda,  // dummy CUDA tensor for dispatch
                                         Tensor ptrs, Tensor shapes, Tensor dtypes,
                                         int64_t num_lists, int64_t num_tensors) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);
  nvte_multi_tensor_compute_scale_inv_e8m0_cuda(
      static_cast<int>(chunk_size), pack.list_ptrs.data(), static_cast<size_t>(num_lists),
      static_cast<size_t>(num_tensors), getCurrentCUDAStreamRaw());
}

std::tuple<Tensor, Tensor> multi_tensor_unscale_l2norm(int64_t chunk_size, Tensor noop_flag,
                                                       Tensor ptrs, Tensor shapes, Tensor dtypes,
                                                       int64_t num_lists, int64_t num_tensors,
                                                       Tensor inv_scale, bool per_tensor) {
  MultiTensorPack pack;
  pack.build(ptrs, shapes, dtypes, num_lists, num_tensors);

  auto device_idx = noop_flag.get_device_index();
  auto noop_cu = makeTransformerEngineTensor(noop_flag);
  auto inv_cu = makeTransformerEngineTensor(inv_scale);

  int max_chunks_per_tensor = -1;
  const int64_t* s = static_cast<const int64_t*>(shapes.data_ptr());
  for (int64_t ti = 0; ti < num_tensors; ++ti) {
    int chunks = (static_cast<int>(s[ti * 2]) + static_cast<int>(chunk_size) - 1) /
                 static_cast<int>(chunk_size);
    if (chunks > max_chunks_per_tensor) max_chunks_per_tensor = chunks;
  }

  auto output = allocateStableTensorZeros({320}, ScalarType::Float, device_idx);
  auto ret = allocateStableTensor({1}, ScalarType::Float, device_idx);
  auto opt = per_tensor ? allocateStableTensorZeros({num_tensors * max_chunks_per_tensor},
                                                    ScalarType::Float, device_idx)
                        : allocateStableTensor({1}, ScalarType::Float, device_idx);
  auto rpt = per_tensor ? allocateStableTensor({num_tensors}, ScalarType::Float, device_idx)
                        : allocateStableTensor({1}, ScalarType::Float, device_idx);

  auto output_cu = makeTransformerEngineTensor(output);
  auto ret_cu = makeTransformerEngineTensor(ret);
  auto opt_cu = makeTransformerEngineTensor(opt);
  auto rpt_cu = makeTransformerEngineTensor(rpt);

  nvte_multi_tensor_unscale_l2norm_cuda(
      static_cast<int>(chunk_size), noop_cu.data(), pack.list_ptrs.data(),
      static_cast<size_t>(num_lists), static_cast<size_t>(num_tensors), output_cu.data(),
      opt_cu.data(), ret_cu.data(), rpt_cu.data(), inv_cu.data(), per_tensor, max_chunks_per_tensor,
      getCurrentCUDAStreamRaw(device_idx));

  return std::make_tuple(ret, rpt);
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def(
      "multi_tensor_scale(int chunk_size, Tensor is_infinite, Tensor ptrs, Tensor shapes, Tensor "
      "dtypes, int num_lists, int num_tensors, float scale) -> ()");
  m.def(
      "multi_tensor_scale_tensor(int chunk_size, Tensor is_infinite, Tensor ptrs, Tensor shapes, "
      "Tensor dtypes, int num_lists, int num_tensors, Tensor scale) -> ()");
  m.def(
      "multi_tensor_l2norm(int chunk_size, Tensor noop_flag, Tensor ptrs, Tensor shapes, Tensor "
      "dtypes, int num_lists, int num_tensors, bool per_tensor) -> (Tensor, Tensor)");
  m.def(
      "multi_tensor_adam(int chunk_size, Tensor noop_flag, Tensor ptrs, Tensor shapes, Tensor "
      "dtypes, int num_lists, int num_tensors, float lr, float beta1, float beta2, float epsilon, "
      "int step, int mode, int bias_correction, float weight_decay) -> ()");
  m.def(
      "multi_tensor_adam_capturable(int chunk_size, Tensor noop_flag, Tensor ptrs, Tensor shapes, "
      "Tensor dtypes, int num_lists, int num_tensors, Tensor lr, float beta1, float beta2, float "
      "epsilon, Tensor step, int mode, int bias_correction, float weight_decay, Tensor inv_scale) "
      "-> ()");
  m.def(
      "multi_tensor_sgd(int chunk_size, Tensor noop_flag, Tensor ptrs, Tensor shapes, Tensor "
      "dtypes, int num_lists, int num_tensors, float wd, float momentum, float dampening, float "
      "lr, bool nesterov, bool first_run, bool wd_after_momentum, float scale) -> ()");
  m.def(
      "multi_tensor_adam_param_remainder(int chunk_size, Tensor noop_flag, Tensor ptrs, Tensor "
      "shapes, Tensor dtypes, int num_lists, int num_tensors, float lr, float beta1, float beta2, "
      "float epsilon, int step, int mode, int bias_correction, float weight_decay) -> ()");
  m.def(
      "multi_tensor_adam_fp8(int chunk_size, Tensor noop_flag, Tensor ptrs, Tensor shapes, Tensor "
      "dtypes, int num_lists, int num_tensors, float lr, float beta1, float beta2, float epsilon, "
      "int step, int mode, int bias_correction, float weight_decay, int fp8_dtype) -> ()");
  m.def(
      "multi_tensor_adam_capturable_master(int chunk_size, Tensor noop_flag, Tensor ptrs, Tensor "
      "shapes, Tensor dtypes, int num_lists, int num_tensors, Tensor lr, float beta1, float beta2, "
      "float epsilon, Tensor step, int mode, int bias_correction, float weight_decay, Tensor "
      "inv_scale) -> ()");
  m.def(
      "multi_tensor_compute_scale_and_scale_inv(int chunk_size, Tensor noop_flag, Tensor ptrs, "
      "Tensor shapes, Tensor dtypes, int num_lists, int num_tensors, float max_fp8, bool "
      "force_pow_2_scales, float epsilon) -> ()");
  m.def(
      "multi_tensor_compute_scale_inv_e8m0(int chunk_size, Tensor dummy_cuda, Tensor ptrs, Tensor "
      "shapes, Tensor dtypes, int num_lists, int num_tensors) -> ()");
  m.def(
      "multi_tensor_unscale_l2norm(int chunk_size, Tensor noop_flag, Tensor ptrs, Tensor shapes, "
      "Tensor dtypes, int num_lists, int num_tensors, Tensor inv_scale, bool per_tensor) -> "
      "(Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("multi_tensor_scale", TORCH_BOX(multi_tensor_scale));
  m.impl("multi_tensor_scale_tensor", TORCH_BOX(multi_tensor_scale_tensor));
  m.impl("multi_tensor_l2norm", TORCH_BOX(multi_tensor_l2norm));
  m.impl("multi_tensor_adam", TORCH_BOX(multi_tensor_adam));
  m.impl("multi_tensor_adam_capturable", TORCH_BOX(multi_tensor_adam_capturable));
  m.impl("multi_tensor_sgd", TORCH_BOX(multi_tensor_sgd));
  m.impl("multi_tensor_adam_param_remainder", TORCH_BOX(multi_tensor_adam_param_remainder));
  m.impl("multi_tensor_adam_fp8", TORCH_BOX(multi_tensor_adam_fp8));
  m.impl("multi_tensor_adam_capturable_master", TORCH_BOX(multi_tensor_adam_capturable_master));
  m.impl("multi_tensor_compute_scale_and_scale_inv",
         TORCH_BOX(multi_tensor_compute_scale_and_scale_inv));
  m.impl("multi_tensor_compute_scale_inv_e8m0", TORCH_BOX(multi_tensor_compute_scale_inv_e8m0));
  m.impl("multi_tensor_unscale_l2norm", TORCH_BOX(multi_tensor_unscale_l2norm));
}
