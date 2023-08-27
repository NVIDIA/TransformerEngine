/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdlib>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <exception>
#include <memory>
#include <stdexcept>
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
#include <utility>

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

class Tensor {
  NVTETensor tensor;

public:
  Tensor(void *data, const NVTEShape &shape, NVTEDType dtype, float *amax,
         float *scale, float *scale_inv)
      : tensor{nvte_tensor_create(data, shape, dtype, amax, scale, scale_inv)} {
  }
  Tensor(NVTETensor &&tensor_) : tensor{std::exchange{tensor_, nullptr}} {}
  Tensor(Tensor &&other) noexcept
      : tensor{std::exchange(other.tensor, nullptr)} {}
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor &operator=(Tensor &&) = delete;
  ~Tensor() {
    if (tensor)
      nvte_tensor_destroy(tensor);
  }
  operator NVTETensor() const { return tensor; }
  NVTEDType dtype() const { return nvte_tensor_type(tensor); }
  NVTEShape shape() const { return nvte_tensor_shape(tensor); }
  void *data() const { return nvte_tensor_data(tensor); }
  float *amax() const { return nvte_tensor_amax(tensor); }
  float *scale() const { return nvte_tensor_scale(tensor); }
  float *scale_inv() const { return nvte_tensor_scale_inv(tensor); }
};

// ----------- Wrapper for NVTETensorPack -----------
struct TensorPack : NVTETensorPack {
  TensorPack(const std::vector<Tensor> &tensors_) : NVTETensorPack{} {
    size = tensors_.size();
    if (size > MAX_SIZE) {
      throw std::runtime_error("TensorPack size exceeds MAX_SIZE");
    }
    for (size_t i = 0; i < size; ++i) {
      tensors[i] = static_cast<NVTETensor>(tensors_[i]);
    }
    nvte_tensor_pack_create(this);
  }
  operator std::vector<Tensor>() const {
    std::vector<Tensor> tensors_(size);
    for (size_t i = 0; i < size; ++i) {
      tensors_[i] = static_cast<Tensor>(tensors[i]);
    }
    return tensors_;
  }
  operator NVTETensorPack *() { return this; }
  operator const NVTETensorPack *() const { return this; }
  ~TensorPack() { nvte_tensor_pack_destroy(this); }
};

// ----------- Function substitution template machinery -----------
template <typename T> struct exposed_type {
  using type = T;
};

template <typename T> struct wrapped;
template <typename T> struct wrapped : exposed_type<T> {
  static T wrap(T arg) { return arg; }
  static T unwrap(T arg) { return arg; }
};
template <> struct wrapped<void> : exposed_type<void> {
  // Intentionally left blank
  // ie. this should never be used
  // because an argument cannot have
  // void type, while conversion
  // should be skipped for void return type.
};
template <> struct wrapped<NVTETensor> : exposed_type<Tensor> {
  static Tensor wrap(NVTETensor arg) { return static_cast<Tensor>(arg); }
  static NVTETensor unwrap(Tensor arg) { return static_cast<NVTETensor>(arg); }
};
template <>
struct wrapped<NVTETensorPack *> : exposed_type<std::vector<Tensor>> {
  static TensorPack unwrap(const std::vector<Tensor> &arg) {
    return TensorPack(arg);
  }
};
template <>
struct wrapped<const NVTETensorPack *> : exposed_type<std::vector<Tensor>> {
  static TensorPack unwrap(const std::vector<Tensor> &arg) {
    return TensorPack(arg);
  }
};
template <> struct wrapped<NVTEShape> : exposed_type<std::vector<size_t>> {
  static std::vector<size_t> wrap(NVTEShape arg) {
    return std::vector<size_t>(arg.data, arg.data + arg.ndim);
  }
  static NVTEShape unwrap(const std::vector<size_t> &arg) {
    NVTEShape shape{};
    shape.ndim = arg.size();
    shape.data = arg.data();
    return shape;
  }
};

template <typename T> using wrapped_t = typename wrapped<T>::type;
struct at_scope_exit {
  void (*ptr)();
  ~at_scope_exit() { ptr(); }
};

// Makes the cuda stream argument always be the last argument
template <typename Ret, typename... PrefixArgs, typename... SuffixArgs,
          typename... Args>
constexpr auto cuda_stream_arg_helper(Ret(func)(Args...),
                                      type_list<PrefixArgs...>,
                                      type_list<SuffixArgs...>) noexcept {
  return [func](wrapped_t<PrefixArgs>... prefixArgs,
                wrapped_t<SuffixArgs>... suffixArgs,
                cudaStream_t stream) -> wrapped_t<Ret> {
    at_scope_exit _{cuda_check};
    if constexpr (!std::is_same_v<Ret, void>) {
      return wrapped<Ret>::wrap(
          func(wrapped<PrefixArgs>::unwrap(prefixArgs)..., stream,
               wrapped<SuffixArgs>::unwrap(suffixArgs)...));
    } else {
      return func(wrapped<PrefixArgs>::unwrap(prefixArgs)..., stream,
                  wrapped<SuffixArgs>::unwrap(suffixArgs)...);
    }
  };
}

template <typename Ret, typename... Args>
constexpr auto wrap(Ret(func)(Args...)) noexcept {
  using tl = type_list<Args...>;
  if constexpr (tl::template contains<cudaStream_t>) {
    constexpr size_t stream_arg_idx = tl::template find<cudaStream_t>;
    using prefix = typename tl::template pop_back<tl::size - stream_arg_idx>;
    using suffix = typename tl::template pop_front<stream_arg_idx + 1>;
    return cuda_stream_arg_helper(func, prefix(), suffix());
  } else {
    return [func](wrapped_t<Args>... args) -> wrapped_t<Ret> {
      at_scope_exit _{cuda_check};
      if constexpr (!std::is_same_v<Ret, void>) {
        return wrapped<Ret>::wrap(func(wrapped<Args>::unwrap(args)...));
      } else {
        return func(wrapped<Args>::unwrap(args)...);
      }
    };
  }
}

// Manual wrapper around nvte_multi_cast_transpose
void multi_cast_transpose(const std::vector<Tensor> &inputs,
                          const std::vector<Tensor> &cast_outs,
                          const std::vector<Tensor> &transposed_outs,
                          cudaStream_t stream) {
  auto inputs_ = *reinterpret_cast<const std::vector<NVTETensor> *>(&inputs);
  auto cast_outs_ =
      *reinterpret_cast<const std::vector<NVTETensor> *>(&cast_outs);
  auto transposed_outs_ =
      *reinterpret_cast<const std::vector<NVTETensor> *>(&transposed_outs);
  nvte_multi_cast_transpose(inputs_.size(), inputs_.data(), cast_outs_.data(),
                            transposed_outs_.data(), stream);

  cuda_check();
}

// ----------- Registration of torch.ops -----------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::enum_<NVTEDType>(m, "DType", py::module_local())
      .value("Byte", kNVTEByte)
      .value("Int32", kNVTEInt32)
      .value("Int64", kNVTEInt64)
      .value("Float32", kNVTEFloat32)
      .value("Float16", kNVTEFloat16)
      .value("BFloat16", kNVTEBFloat16)
      .value("Float8E4M3", kNVTEFloat8E4M3)
      .value("Float8E5M2", kNVTEFloat8E5M2);

  py::enum_<NVTE_Fused_Attn_Backend>(m, "FusedAttnBackend", py::module_local())
      .value("No_Backend", NVTE_No_Backend)
      .value("F16_max512_seqlen", NVTE_F16_max512_seqlen)
      .value("F16_arbitrary_seqlen", NVTE_F16_arbitrary_seqlen)
      .value("FP8", NVTE_FP8);

  py::enum_<NVTE_QKV_Layout>(m, "QKVLayout", py::module_local())
      .value("NOT_INTERLEAVED", NVTE_NOT_INTERLEAVED)
      .value("QKV_INTERLEAVED", NVTE_QKV_INTERLEAVED)
      .value("KV_INTERLEAVED", NVTE_KV_INTERLEAVED);

  py::enum_<NVTE_Bias_Type>(m, "BiasType", py::module_local())
      .value("NO_BIAS", NVTE_NO_BIAS)
      .value("PRE_SCALE_BIAS", NVTE_PRE_SCALE_BIAS)
      .value("POST_SCALE_BIAS", NVTE_POST_SCALE_BIAS);

  py::enum_<NVTE_Mask_Type>(m, "MaskType", py::module_local())
      .value("NO_MASK", NVTE_NO_MASK)
      .value("PADDING_MASK", NVTE_PADDING_MASK)
      .value("CAUSAL_MASK", NVTE_CAUSAL_MASK);

  py::class_<Tensor>(m, "Tensor", py::module_local())
      .def(py::init<void *, const NVTEShape &, NVTEDType, float *, float *,
                    float *>())
      .def_property_readonly("dtype", &Tensor::dtype)
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("data", &Tensor::data)
      .def_property_readonly("amax", &Tensor::amax)
      .def_property_readonly("scale", &Tensor::scale)
      .def_property_readonly("scale_inv", &Tensor::scale_inv);

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
