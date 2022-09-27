/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file transformer_engine.h
 *  \brief Base classes and functions of Transformer Engine API.
 */

#ifndef TRANSFORMER_ENGINE_TRANSFORMER_ENGINE_H_
#define TRANSFORMER_ENGINE_TRANSFORMER_ENGINE_H_

#include <stddef.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \enum NVTEDType
 *  \brief TE datatype.
 */
enum NVTEDType {
    kNVTEByte       = 0,  /*!< Byte */
    kNVTEInt32      = 1,  /*!< 32-bit integer */
    kNVTEFloat32    = 2,  /*!< 32-bit float */
    kNVTEFloat16    = 3,  /*!< 16-bit float (E5M10) */
    kNVTEBFloat16   = 4,  /*!< 16-bit bfloat (E8M7) */
    kNVTEFloat8E4M3 = 5,  /*!< 8-bit float (E4M3) */
    kNVTEFloat8E5M2 = 6,  /*!< 8-bit float (E5M2) */
    kNVTENumTypes         /*!< Number of supported types */
};

/*! \struct NVTEShape
 *  \brief Shape of the tensor.
 */
struct NVTEShape {
  /*! \brief Shape data, of size ndim. */
  const size_t *data;
  /*! \brief Number of dimensions. */
  size_t ndim;
};

/*! \brief TE Tensor type
 *
 * NVTETensor is a contiguous tensor type storing a pointer
 * to data of a given shape and type. It does not own the
 * memory it points to.
 */
typedef void* NVTETensor;

/*! \brief Create a new TE tensor.
 *
 * Create a new TE tensor with a given shape, datatype and data.
 * TE tensors are just wrappers on top of raw data and do not
 * own memory.
 *
 *  \param[in] dptr  Pointer to the tensor data.
 *  \param[in] shape Shape of the tensor.
 *  \param[in] dtype Data type of the tensor.
 *
 *  \return A new TE tensor.
 */
NVTETensor nvte_create_tensor(void *dptr,
                              const NVTEShape shape,
                              const NVTEDType dtype);

/*! \brief Destroy a TE tensor.
 *
 * Since the TE tensor does not own memory, the underlying
 * data is not freed during this operation.
 *
 *  \param[in] tensor Tensor to be destroyed.
 */
void nvte_destroy_tensor(NVTETensor tensor);

/*! \brief Get a tensor's data type.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A data type of the input tensor.
 */
NVTEDType nvte_tensor_type(const NVTETensor tensor);

/*! \brief Get a tensor's data shape.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A shape of the input tensor.
 */
NVTEShape nvte_tensor_shape(const NVTETensor tensor);

/*! \brief Get a raw pointer to the tensor's data.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A raw pointer to tensor's data.
 */
void *nvte_tensor_data(const NVTETensor tensor);


#ifdef __cplusplus
}  // extern "C"

#include <vector>

/*! \namespace transformer_engine
 *  \brief Namespace containing C++ API of Transformer Engine.
 */
namespace transformer_engine {

/*! \enum DType
 *  \brief TE datatype.
 */
enum class DType {
  kByte       = 0,
  kInt32      = 1,
  kFloat32    = 2,
  kFloat16    = 3,
  kBFloat16   = 4,
  kFloat8E4M3 = 5,
  kFloat8E5M2 = 6,
  kNumTypes
};

/*! \struct TensorWrapper
 *  \brief C++ wrapper for the NVTETensor class.
 */
class TensorWrapper {
 public:
  /*! \brief Constructs new TensorWrapper.
   *
   * Create a new TE tensor with a given shape, datatype and data.
   * TE tensors are just wrappers on top of raw data and do not
   * own memory.
   *
   *  \param[in] dptr  Pointer to the tensor data.
   *  \param[in] shape Shape of the tensor.
   *  \param[in] dtype Data type of the tensor.
   */
  TensorWrapper(void *dptr, const NVTEShape &shape, const DType dtype) :
    tensor_(nvte_create_tensor(dptr, shape, static_cast<NVTEDType>(dtype))) {}

  /*! \brief Constructs new TensorWrapper.
   *
   * Create a new TE tensor with a given shape, datatype and data.
   * TE tensors are just wrappers on top of raw data and do not
   * own memory.
   *
   *  \param[in] dptr  Pointer to the tensor data.
   *  \param[in] shape Shape of the tensor.
   *  \param[in] dtype Data type of the tensor.
   */
  TensorWrapper(void *dptr, const std::vector<size_t> &shape, const DType dtype) :
    TensorWrapper(dptr, NVTEShape{shape.data(), shape.size()}, dtype) {}

  /*! \brief Constructs new empty TensorWrapper.
   *
   * Create a new empty TE tensor which holds nothing.
   */
  TensorWrapper() : TensorWrapper(nullptr, std::vector<size_t>(), DType::kFloat32) {}

  /*! \brief TensorWrapper destructor. */
  ~TensorWrapper() {
    nvte_destroy_tensor(tensor_);
  }

  TensorWrapper& operator=(const TensorWrapper &other) = delete;
  TensorWrapper(const TensorWrapper &other) = delete;

  /*! \brief Constructs new TensorWrapper from existing TensorWrapper.
   *
   * Pass an existing TE tensor to a new TensorWrapper.
   *
   *  \param[in,out] other The source of the data.
   */
  TensorWrapper(TensorWrapper &&other) {
    tensor_ = other.tensor_;
    other.tensor_ = nullptr;
  }

  /*! \brief Assign the data from existing TensorWrapper.
   *
   * Change ownership of an existing TE tensor.
   *
   *  \param[in,out] other The source of the data.
   */
  TensorWrapper& operator=(TensorWrapper &&other) {
    if (this == &other) return *this;
    nvte_destroy_tensor(tensor_);
    tensor_ = other.tensor_;
    other.tensor_ = nullptr;
    return *this;
  }

  /*! \brief Get an underlying NVTETensor.
   *
   *  \return NVTETensor held by this TensorWrapper.
   */
  NVTETensor data() const noexcept {
    return tensor_;
  }

  /*! \brief Get the shape of this TensorWrapper.
   *
   *  \return Shape of this TensorWrapper.
   */
  const NVTEShape shape() const noexcept {
    if (tensor_ == nullptr) return NVTEShape{nullptr, 0};
    return nvte_tensor_shape(tensor_);
  }

  /*! \brief Get the data type of this TensorWrapper.
   *
   *  \return Data type of this TensorWrapper.
   */
  DType dtype() const noexcept {
    if (tensor_ == nullptr) return DType::kNumTypes;
    return static_cast<DType>(nvte_tensor_type(tensor_));
  }

  /*! \brief Get a raw pointer to the tensor's data.
   *
   *  \return A raw pointer to tensor's data.
   */
  void *dptr() const noexcept {
    if (tensor_ == nullptr) return nullptr;
    return nvte_tensor_data(tensor_);
  }

 private:
  /*! \brief Wrapped NVTETensor. */
  NVTETensor tensor_ = nullptr;
};

}  // namespace transformer_engine

#endif

#endif  // TRANSFORMER_ENGINE_TRANSFORMER_ENGINE_H_
