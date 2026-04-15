#ifndef TRANSFORMER_ENGINE_MUSA_COMMON_UTIL_MUDNN_H_
#define TRANSFORMER_ENGINE_MUSA_COMMON_UTIL_MUDNN_H_

#include <vector>
#include <utility>

#include <c10/util/strides.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch_musa/csrc/aten/utils/Context.h>
#include <torch_musa/csrc/aten/utils/Utils.h>

namespace transformer_engine::musa {

using ScalarType = typename at::ScalarType;
using DimVector = typename c10::DimVector;

using MUTensor = typename at::musa::muTensor;

inline std::vector<size_t> Flat2DimShape(const Tensor* t) {
  return {t->flat_first_dim(), t->flat_last_dim()};
}

inline std::pair<DimVector, DimVector>
make_mudnn_sizes_strides(const std::vector<size_t>& shape) {
  auto mudnn_sizes = DimVector(shape.cbegin(), shape.cend());
  auto mudnn_strides = c10::contiguous_strides(mudnn_sizes);
  return std::make_pair(std::move(mudnn_sizes), std::move(mudnn_strides));
}

inline ScalarType ToTorchDtype(DType te_dtype) {
  auto th_dtype = ScalarType::Undefined;
  switch (te_dtype) {
    case DType::kFloat16:
      th_dtype = ScalarType::Half;
      break;
    case DType::kBFloat16:
      th_dtype = ScalarType::BFloat16;
      break;
    case DType::kFloat32:
      th_dtype = ScalarType::Float;
      break;
    case DType::kFloat8E4M3:
      th_dtype = ScalarType::Float8_e4m3fn;
      break;
    case DType::kFloat8E5M2:
      th_dtype = ScalarType::Float8_e5m2;
      break;
    default:
      break;
  }
  return th_dtype;
}

inline void SetMUTensorDType(DType te_dtype, MUTensor& m_t) {
  at::musa::SetMUTensorDType(ToTorchDtype(te_dtype), m_t);
}

inline void SetMUTensorFormat(
    const std::vector<size_t>& shape,
    MUTensor& m_t) {
  const int ndim = shape.size();
  const auto mudnn_format = (ndim == 5) ? MUTensor::Format::NCDHW
                                        : MUTensor::Format::NCHW;
  m_t.SetFormat(mudnn_format);

  const auto ss = make_mudnn_sizes_strides(shape);
  m_t.SetNdInfo(ndim, ss.first.data(), ss.second.data());
}

inline MUTensor CreateMUTensor(const SimpleTensor& st) {
  MUTensor m_t;
  SetMUTensorDType(st.dtype, m_t);
  at::musa::SetMUTensorAddr(st.dptr, m_t);
  SetMUTensorFormat(st.shape, m_t);
  return m_t;
}

inline MUTensor CreateMUTensor(
    const SimpleTensor& st,
    const std::vector<size_t>& shape) {
  MUTensor m_t;
  SetMUTensorDType(st.dtype, m_t);
  at::musa::SetMUTensorAddr(st.dptr, m_t);
  SetMUTensorFormat(shape, m_t);
  return m_t;
}

} // namespace transformer_engine::musa

#endif  // TRANSFORMER_ENGINE_MUSA_COMMON_UTIL_MUDNN_H_
