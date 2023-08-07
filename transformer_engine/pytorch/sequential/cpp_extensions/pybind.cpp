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
    using prefix = tl::template pop_back<tl::size - stream_arg_idx>;
    using suffix = tl::template pop_front<stream_arg_idx + 1>;
    return remove_cuda_stream_arg_helper(func, prefix(), suffix());
  } else {
    return [func](wrapped_arg_t<Args>... args) -> Ret {
      return func(unwrap_arg(args)...);
    };
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvte_gelu", wrap(nvte_gelu));
  m.def("nvte_dgelu", wrap(nvte_dgelu));
  m.def("nvte_geglu", wrap(nvte_geglu));
  m.def("nvte_dgeglu", wrap(nvte_dgeglu));
  m.def("nvte_relu", wrap(nvte_relu));
  m.def("nvte_drelu", wrap(nvte_drelu));
  m.def("nvte_swiglu", wrap(nvte_swiglu));
  m.def("nvte_dswiglu", wrap(nvte_dswiglu));
  m.def("nvte_reglu", wrap(nvte_reglu));
  m.def("nvte_dreglu", wrap(nvte_dreglu));
  m.def("nvte_fp8_quantize", wrap(nvte_fp8_quantize));
  m.def("nvte_fp8_dequantize", wrap(nvte_fp8_dequantize));
  m.def("nvte_get_fused_attn_backend", wrap(nvte_get_fused_attn_backend));
  m.def("nvte_fused_attn_fwd_qkvpacked", wrap(nvte_fused_attn_fwd_qkvpacked));
  m.def("nvte_fused_attn_bwd_qkvpacked", wrap(nvte_fused_attn_bwd_qkvpacked));
  m.def("nvte_fused_attn_fwd_kvpacked", wrap(nvte_fused_attn_fwd_kvpacked));
  m.def("nvte_fused_attn_bwd_kvpacked", wrap(nvte_fused_attn_bwd_kvpacked));
  m.def("nvte_cublas_gemm", wrap(nvte_cublas_gemm));
  m.def("nvte_layernorm_fwd", wrap(nvte_layernorm_fwd));
  m.def("nvte_layernorm1p_fwd", wrap());
  m.def("nvte_layernorm_bwd", wrap(nvte_layernorm_bwd));
  m.def("nvte_layernorm1p_bwd", wrap(nvte_layernorm1p_bwd));
  m.def("nvte_rmsnorm_fwd", wrap(nvte_rmsnorm_fwd));
  m.def("nvte_rmsnorm_bwd", wrap(nvte_rmsnorm_bwd));
  m.def("nvte_scaled_softmax_forward", wrap(nvte_scaled_softmax_forward));
  m.def("nvte_scaled_softmax_backward", wrap(nvte_scaled_softmax_backward));
  m.def("nvte_scaled_masked_softmax_forward",
        wrap(nvte_scaled_masked_softmax_forward));
  m.def("nvte_scaled_masked_softmax_backward",
        wrap(nvte_scaled_masked_softmax_backward));
  m.def("nvte_scaled_upper_triang_masked_softmax_forward",
        wrap(nvte_scaled_upper_triang_masked_softmax_forward));
  m.def("nvte_scaled_upper_triang_masked_softmax_backward",
        wrap(nvte_scaled_upper_triang_masked_softmax_backward));
  m.def("nvte_create_tensor", wrap(nvte_create_tensor));
  m.def("nvte_destroy_tensor", wrap(nvte_destroy_tensor));
  m.def("nvte_tensor_type", wrap(nvte_tensor_type));
  m.def("nvte_tensor_shape", wrap(nvte_tensor_shape));
  m.def("nvte_tensor_data", wrap(nvte_tensor_data));
  m.def("nvte_tensor_amax", wrap(nvte_tensor_amax));
  m.def("nvte_tensor_scale", wrap(nvte_tensor_scale));
  m.def("nvte_tensor_scale_inv", wrap(nvte_tensor_scale_inv));
  m.def("nvte_tensor_pack_create", wrap(nvte_tensor_pack_create));
  m.def("nvte_tensor_pack_destroy", wrap(nvte_tensor_pack_destroy));
  m.def("nvte_cast_transpose", wrap(nvte_cast_transpose));
  m.def("nvte_transpose", wrap(nvte_transpose));
  m.def("nvte_cast_transpose_dbias", wrap(nvte_cast_transpose_dbias));
  m.def("nvte_fp8_transpose_dbias", wrap(nvte_fp8_transpose_dbias));
  m.def("nvte_cast_transpose_dbias_dgelu",
        wrap(nvte_cast_transpose_dbias_dgelu));
  m.def("nvte_multi_cast_transpose", wrap(nvte_multi_cast_transpose));
  m.def("nvte_dgeglu_cast_transpose", wrap(nvte_dgeglu_cast_transpose));

  py::enum_<NVTEDType>(m, "NVTEDType")
      .value("kNVTEByte", kNVTEByte)
      .value("kNVTEInt32", kNVTEInt32)
      .value("kNVTEInt64", kNVTEInt64)
      .value("kNVTEFloat32", kNVTEFloat32)
      .value("kNVTEFloat16", kNVTEFloat16)
      .value("kNVTEBFloat16", kNVTEBFloat16)
      .value("kNVTEFloat8E4M3", kNVTEFloat8E4M3)
      .value("kNVTEFloat8E5M2", kNVTEFloat8E5M2);

  py::enum_<NVTE_Fused_Attn_Backend>(m, "NVTE_Fused_Attn_Backend")
      .value("NVTE_No_Backend", NVTE_No_Backend)
      .value("NVTE_F16_max512_seqlen", NVTE_F16_max512_seqlen)
      .value("NVTE_F16_arbitrary_seqlen", NVTE_F16_arbitrary_seqlen)
      .value("NVTE_FP8", NVTE_FP8);

  py::enum_<NVTE_QKV_Layout>(m, "NVTE_QKV_Layout")
      .value("NVTE_NOT_INTERLEAVED", NVTE_NOT_INTERLEAVED)
      .value("NVTE_QKV_INTERLEAVED", NVTE_QKV_INTERLEAVED)
      .value("NVTE_KV_INTERLEAVED", NVTE_KV_INTERLEAVED);

  py::enum_<NVTE_Bias_Type>(m, "NVTE_Bias_Type")
      .value("NVTE_NO_BIAS", NVTE_NO_BIAS)
      .value("NVTE_PRE_SCALE_BIAS", NVTE_PRE_SCALE_BIAS)
      .value("NVTE_POST_SCALE_BIAS", NVTE_POST_SCALE_BIAS);

  py::enum_<NVTE_Mask_Type>(m, "NVTE_Mask_Type")
      .value("NVTE_NO_MASK", NVTE_NO_MASK)
      .value("NVTE_PADDING_MASK", NVTE_PADDING_MASK)
      .value("NVTE_CAUSAL_MASK", NVTE_CAUSAL_MASK);

  py::class_<NVTEShape>(m, "NVTEShape")
      .def(py::init<>())
      .def_readwrite("data", &NVTEShape::data)
      .def_readwrite("ndim", &NVTEShape::ndim);

  py::class_<Tensor>(m, "NVTETensor")
      .def(py::init<NVTEDType, at::Tensor, at::Tensor, at::Tensor,
                    at::Tensor>());
}
