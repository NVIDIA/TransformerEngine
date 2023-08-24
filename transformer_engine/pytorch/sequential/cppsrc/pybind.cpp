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
#include <cstdlib>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <exception>
#include <memory>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/script.h>
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

// ----------- Wrapper for NVTETensor -----------
void cuda_check() {
  static const bool perform_check = []() {
    const char *var = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (var && var[0] == '1') {
      return true;
    }
    return false;
  }();

  if (perform_check) {
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "TE kernel error: " + std::string(cudaGetErrorName(err)) + ": " +
          cudaGetErrorString(err));
    }
  }
}
struct Tensor : torch::CustomClassHolder {
  static_assert(std::is_same_v<NVTETensor, void *>);

  std::shared_ptr<void> pimpl;
  at::Tensor data;
  at::Tensor amax;
  at::Tensor scale;
  at::Tensor scale_inv;

  static float *getDataPtr(at::Tensor t) {
    if (t.numel() > 0) {
      if (!t.is_cuda()) {
        throw std::runtime_error(
            "Cannot create NVTE Tensor: !tensor.is_cuda()");
      }
      if (!t.is_contiguous()) {
        throw std::runtime_error(
            "Cannot create NVTE Tensor: !tensor.is_contiguous()");
      }
      return reinterpret_cast<float *>(t.data_ptr());
    } else {
      return nullptr;
    }
  }

  Tensor(int64_t dtype, at::Tensor data, at::Tensor amax, at::Tensor scale,
         at::Tensor scale_inv)
      : pimpl{nvte_create_tensor(getDataPtr(data),
                                 NVTEShape{(size_t *)(data.sizes().data()),
                                           data.sizes().size()},
                                 NVTEDType(dtype), getDataPtr(amax),
                                 getDataPtr(scale), getDataPtr(scale_inv)),
              [](NVTETensor impl) { nvte_destroy_tensor(impl); }},
        data{data}, amax{amax}, scale{scale}, scale_inv{scale_inv} {}
};

// ----------- Wrapper for NVTETensorPack -----------
struct TensorPack : NVTETensorPack {
  TensorPack(const std::vector<c10::intrusive_ptr<Tensor>> &tensors_)
      : NVTETensorPack{} {
    size = tensors_.size();
    if (size > MAX_SIZE) {
      throw std::runtime_error("TensorPack size exceeds MAX_SIZE");
    }
    for (size_t i = 0; i < size; ++i) {
      tensors[i] = (NVTETensor)(tensors_[i]->pimpl.get());
    }
    nvte_tensor_pack_create(this);
  }
  operator NVTETensorPack *() { return this; }
  operator const NVTETensorPack *() const { return this; }
  ~TensorPack() { nvte_tensor_pack_destroy(this); }
};

// ----------- Function substitution template machinery -----------
template <typename T> struct trait {
  using type = T;
};

template <typename T> struct wrapped_arg;

#define TO_INT64_T(...)                                                        \
  template <> struct wrapped_arg<__VA_ARGS__> : trait<int64_t> {               \
    static int64_t unwrap(__VA_ARGS__ arg) { return (int64_t)arg; }            \
  }

TO_INT64_T(char);
TO_INT64_T(unsigned char);
TO_INT64_T(signed char);
TO_INT64_T(unsigned short);
TO_INT64_T(signed short);
TO_INT64_T(unsigned int);
TO_INT64_T(signed int);
TO_INT64_T(unsigned long);
TO_INT64_T(signed long);
TO_INT64_T(unsigned long long);

template <typename T> struct wrapped_arg : trait<T> {
  static T unwrap(T arg) { return arg; }
};
template <> struct wrapped_arg<float> : trait<double> {
  static double unwrap(float arg) { return arg; }
};
template <>
struct wrapped_arg<NVTETensor> : trait<const c10::intrusive_ptr<Tensor> &> {
  static NVTETensor unwrap(const c10::intrusive_ptr<Tensor> &arg) {
    return (NVTETensor)(arg->pimpl.get());
  }
};
template <>
struct wrapped_arg<NVTETensorPack *>
    : trait<std::vector<c10::intrusive_ptr<Tensor>>> {
  static TensorPack unwrap(const std::vector<c10::intrusive_ptr<Tensor>> &arg) {
    return TensorPack(arg);
  }
};
template <>
struct wrapped_arg<const NVTETensorPack *>
    : trait<std::vector<c10::intrusive_ptr<Tensor>>> {
  static TensorPack unwrap(const std::vector<c10::intrusive_ptr<Tensor>> &arg) {
    return TensorPack(arg);
  }
};
template <> struct wrapped_arg<NVTEDType> : trait<int64_t> {
  static NVTEDType unwrap(int64_t arg) { return NVTEDType(arg); }
};
template <> struct wrapped_arg<NVTE_Fused_Attn_Backend> : trait<int64_t> {
  static NVTE_Fused_Attn_Backend unwrap(int64_t arg) {
    return NVTE_Fused_Attn_Backend(arg);
  }
};
template <> struct wrapped_arg<NVTE_QKV_Layout> : trait<int64_t> {
  static NVTE_QKV_Layout unwrap(int64_t arg) { return NVTE_QKV_Layout(arg); }
};
template <> struct wrapped_arg<NVTE_Bias_Type> : trait<int64_t> {
  static NVTE_Bias_Type unwrap(int64_t arg) { return NVTE_Bias_Type(arg); }
};
template <> struct wrapped_arg<NVTE_Mask_Type> : trait<int64_t> {
  static NVTE_Mask_Type unwrap(int64_t arg) { return NVTE_Mask_Type(arg); }
};
template <typename T> using wrapped_arg_t = typename wrapped_arg<T>::type;
struct at_scope_exit {
  void (*ptr)();
  ~at_scope_exit() { ptr(); }
};

template <typename Ret, typename... PrefixArgs, typename... SuffixArgs,
          typename... Args>
constexpr auto
remove_cuda_stream_arg_helper(Ret(func)(Args...), type_list<PrefixArgs...>,
                              type_list<SuffixArgs...>) noexcept {
  return [func](wrapped_arg_t<PrefixArgs>... prefixArgs,
                wrapped_arg_t<SuffixArgs>... suffixArgs) -> Ret {
    at_scope_exit _{cuda_check};
    return func(wrapped_arg<PrefixArgs>::unwrap(prefixArgs)...,
                at::cuda::getCurrentCUDAStream(),
                wrapped_arg<SuffixArgs>::unwrap(suffixArgs)...);
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
      at_scope_exit _{cuda_check};
      return func(wrapped_arg<Args>::unwrap(args)...);
    };
  }
}

// Manual wrapper around nvte_multi_cast_transpose
void multi_cast_transpose(
    const std::vector<c10::intrusive_ptr<Tensor>> &inputs,
    const std::vector<c10::intrusive_ptr<Tensor>> &cast_outs,
    const std::vector<c10::intrusive_ptr<Tensor>> &transposed_outs) {
  auto count = inputs.size();
  std::vector<NVTETensor> inputs_(count);
  std::vector<NVTETensor> cast_outs_(count);
  std::vector<NVTETensor> transposed_outs_(count);

  for (int i = 0; i < inputs.size(); ++i) {
    inputs_[i] = (NVTETensor)(inputs[i]->pimpl.get());
    cast_outs_[i] = (NVTETensor)(cast_outs[i]->pimpl.get());
    transposed_outs_[i] = (NVTETensor)(transposed_outs[i]->pimpl.get());
  }

  nvte_multi_cast_transpose(count, inputs_.data(), cast_outs_.data(),
                            transposed_outs_.data(),
                            at::cuda::getCurrentCUDAStream());

  cuda_check();
}

// ----------- Registration of torch.ops -----------
TORCH_LIBRARY(transformer_engine_cuda, m) {
  m.class_<Tensor>("Tensor")
      .def(torch::init<int64_t, at::Tensor, at::Tensor, at::Tensor,
                       at::Tensor>())
      .def_property("dtype",
                    [](const c10::intrusive_ptr<Tensor> &self) {
                      return (int64_t)nvte_tensor_type(
                          (NVTETensor)(self->pimpl.get()));
                    })
      .def_property("shape",
                    [](const c10::intrusive_ptr<Tensor> &self) {
                      NVTEShape s =
                          nvte_tensor_shape((NVTETensor)(self->pimpl.get()));
                      return std::vector<int64_t>(s.data, s.data + s.ndim);
                    })
      .def_readonly("data", &Tensor::data)
      .def_readonly("amax", &Tensor::amax)
      .def_readonly("scale", &Tensor::scale)
      .def_readonly("scale_inv", &Tensor::scale_inv);

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
  m.def("cast_transpose", wrap(nvte_cast_transpose));
  m.def("transpose", wrap(nvte_transpose));
  m.def("cast_transpose_dbias", wrap(nvte_cast_transpose_dbias));
  m.def("fp8_transpose_dbias", wrap(nvte_fp8_transpose_dbias));
  m.def("cast_transpose_dbias_dgelu", wrap(nvte_cast_transpose_dbias_dgelu));
  m.def("dgeglu_cast_transpose", wrap(nvte_dgeglu_cast_transpose));
  m.def("multi_cast_transpose", &multi_cast_transpose);
}
