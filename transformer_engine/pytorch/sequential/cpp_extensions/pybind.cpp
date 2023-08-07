/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/DispatchStub.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <exception>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/fused_attn.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/layer_norm.h>
#include <transformer_engine/rmsnorm.h>
#include <transformer_engine/softmax.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/transpose.h>
#include <type_traits>

#include "type_list.h"

namespace py = pybind11;

struct Tensor {
  NVTETensor impl;

  static float *getDataPtr(at::Tensor t) {
    if (t.numel() > 0) {
      return reinterpret_cast<float *>(t.data_ptr());
    } else {
      return nullptr;
    }
  }

  Tensor(NVTEDType dtype, at::Tensor data, at::Tensor amax, at::Tensor scale,
         at::Tensor scale_inv) {
    NVTEShape shape{(size_t *)(data.sizes().data()), data.sizes().size()};
    impl = nvte_create_tensor(getDataPtr(data), shape, dtype, getDataPtr(amax),
                              getDataPtr(scale), getDataPtr(scale_inv));
  }
  ~Tensor() { nvte_destroy_tensor(impl); }
};

struct TensorPack : NVTETensorPack {
  TensorPack(const std::vector<Tensor> &tensors_) : NVTETensorPack{} {
    size = tensors_.size();
    if (size > MAX_SIZE) {
      throw std::runtime_error("TensorPack size exceeds MAX_SIZE");
    }
    for (size_t i = 0; i < size; ++i) {
      tensors[i] = tensors_[i].impl;
    }
    nvte_tensor_pack_create(this);
  }
  operator NVTETensorPack *() { return this; }
  ~TensorPack() { nvte_tensor_pack_destroy(this); }
};

template <typename T> struct trait {
  using type = T;
};

template <typename T> struct wrapped_arg : trait<T> {};
template <> struct wrapped_arg<NVTETensor> : trait<Tensor> {};
template <> struct wrapped_arg<NVTETensorPack> : trait<std::vector<Tensor>> {};

template <typename T> using wrapped_arg_t = typename wrapped_arg<T>::type;

template <typename T> decltype(auto) unwrap_arg(T &&arg) {
  if constexpr (std::is_same_v<std::decay_t<T>, wrapped_arg_t<NVTETensor>>) {
    return arg.impl;
  } else if constexpr (std::is_same_v<std::decay_t<T>,
                                      wrapped_arg_t<NVTETensorPack>>) {
    return TensorPack(arg);
  } else {
    { return arg; }
  }
}

template <typename Ret, typename... PrefixArgs, typename... SuffixArgs,
          typename... Args>
constexpr auto
remove_cuda_stream_arg_helper(Ret(func)(Args...), type_list<PrefixArgs...>,
                              type_list<SuffixArgs...>) noexcept {
  return [func](wrapped_arg_t<PrefixArgs>... prefixArgs,
                wrapped_arg_t<SuffixArgs>... suffixArgs) -> Ret {
    return func(unwrap_arg(prefixArgs)..., at::cuda::getCurrentCUDAStream(),
                unwrap_arg(suffixArgs)...);
  };
}

template <typename Ret, typename... Args>
constexpr auto wrap(Ret(func)(Args...)) noexcept {
  using tl = type_list<Args...>;
  if constexpr (tl::template contains<cudaStream_t>) {
    constexpr size_t stream_arg_idx = tl::template find<cudaStream_t>;
    using prefix = typename tl::template pop_back<tl::size - stream_arg_idx>;
    using suffix = typename tl::template pop_front<stream_arg_idx + 1>;
    return remove_cuda_stream_arg_helper(func, prefix(), suffix());
  } else {
    return [func](wrapped_arg_t<Args>... args) -> Ret {
      return func(unwrap_arg(args)...);
    };
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gelu", wrap(nvte_gelu));
  m.def("dgelu", wrap(nvte_dgelu));
  m.def("geglu", wrap(nvte_geglu));
  m.def("dgeglu", wrap(nvte_dgeglu));
  m.def("relu", wrap(nvte_relu));
  m.def("drelu", wrap(nvte_drelu));
  m.def("swiglu", wrap(nvte_swiglu));
  m.def("dswiglu", wrap(nvte_dswiglu));
  m.def("reglu", wrap(nvte_reglu));
  m.def("dreglu", wrap(nvte_dreglu));
  m.def("fp8_quantize", wrap(nvte_fp8_quantize));
  m.def("fp8_dequantize", wrap(nvte_fp8_dequantize));
  m.def("get_fused_attn_backend", wrap(nvte_get_fused_attn_backend));
  m.def("fused_attn_fwd_qkvpacked", wrap(nvte_fused_attn_fwd_qkvpacked));
  m.def("fused_attn_bwd_qkvpacked", wrap(nvte_fused_attn_bwd_qkvpacked));
  m.def("fused_attn_fwd_kvpacked", wrap(nvte_fused_attn_fwd_kvpacked));
  m.def("fused_attn_bwd_kvpacked", wrap(nvte_fused_attn_bwd_kvpacked));
  m.def("cublas_gemm", wrap(nvte_cublas_gemm));
  m.def("layernorm_fwd", wrap(nvte_layernorm_fwd));
  m.def("layernorm1p_fwd", wrap(nvte_layernorm1p_fwd));
  m.def("layernorm_bwd", wrap(nvte_layernorm_bwd));
  m.def("layernorm1p_bwd", wrap(nvte_layernorm1p_bwd));
  m.def("rmsnorm_fwd", wrap(nvte_rmsnorm_fwd));
  m.def("rmsnorm_bwd", wrap(nvte_rmsnorm_bwd));
  m.def("scaled_softmax_forward", wrap(nvte_scaled_softmax_forward));
  m.def("scaled_softmax_backward", wrap(nvte_scaled_softmax_backward));
  m.def("scaled_masked_softmax_forward",
        wrap(nvte_scaled_masked_softmax_forward));
  m.def("scaled_masked_softmax_backward",
        wrap(nvte_scaled_masked_softmax_backward));
  m.def("scaled_upper_triang_masked_softmax_forward",
        wrap(nvte_scaled_upper_triang_masked_softmax_forward));
  m.def("scaled_upper_triang_masked_softmax_backward",
        wrap(nvte_scaled_upper_triang_masked_softmax_backward));
  m.def("create_tensor", wrap(nvte_create_tensor));
  m.def("destroy_tensor", wrap(nvte_destroy_tensor));
  m.def("tensor_type", wrap(nvte_tensor_type));
  m.def("tensor_shape", wrap(nvte_tensor_shape));
  m.def("tensor_data", wrap(nvte_tensor_data));
  m.def("tensor_amax", wrap(nvte_tensor_amax));
  m.def("tensor_scale", wrap(nvte_tensor_scale));
  m.def("tensor_scale_inv", wrap(nvte_tensor_scale_inv));
  m.def("tensor_pack_create", wrap(nvte_tensor_pack_create));
  m.def("tensor_pack_destroy", wrap(nvte_tensor_pack_destroy));
  m.def("cast_transpose", wrap(nvte_cast_transpose));
  m.def("transpose", wrap(nvte_transpose));
  m.def("cast_transpose_dbias", wrap(nvte_cast_transpose_dbias));
  m.def("fp8_transpose_dbias", wrap(nvte_fp8_transpose_dbias));
  m.def("cast_transpose_dbias_dgelu", wrap(nvte_cast_transpose_dbias_dgelu));
  m.def("dgeglu_cast_transpose", wrap(nvte_dgeglu_cast_transpose));

  py::class_<NVTEShape>(m, "Shape")
      .def(py::init<>())
      .def_readwrite("data", &NVTEShape::data)
      .def_readwrite("ndim", &NVTEShape::ndim);

  py::class_<Tensor>(m, "Tensor")
      .def(py::init<NVTEDType, at::Tensor, at::Tensor, at::Tensor,
                    at::Tensor>());
}
