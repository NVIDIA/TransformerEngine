/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file transformer_engine.h
 *  \brief Base classes and functions of Transformer Engine API.
 */

#ifndef TRANSFORMER_ENGINE_TRANSFORMER_ENGINE_H_
#define TRANSFORMER_ENGINE_TRANSFORMER_ENGINE_H_

#include <cuda_runtime_api.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \enum NVTEDType
 *  \brief TE datatype.
 */
enum NVTEDType {
  kNVTEByte = 0,       /*!< Byte */
  kNVTEInt32 = 1,      /*!< 32-bit integer */
  kNVTEInt64 = 2,      /*!< 64-bit integer */
  kNVTEFloat32 = 3,    /*!< 32-bit float */
  kNVTEFloat16 = 4,    /*!< 16-bit float (E5M10) */
  kNVTEBFloat16 = 5,   /*!< 16-bit bfloat (E8M7) */
  kNVTEFloat8E4M3 = 6, /*!< 8-bit float (E4M3) */
  kNVTEFloat8E5M2 = 7, /*!< 8-bit float (E5M2) */
  kNVTEFloat8E8M0 = 8, /*!< 8-bit float (E8M0) */
  kNVTENumTypes        /*!< Number of supported types */
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

/*! \struct NVTEBasicTensor
 *  \brief A basic tensor type used to populate parameters of NVTETensor.
 *  It does not own the memory it points to.
 */
struct NVTEBasicTensor {
  void *data_ptr;
  NVTEDType dtype;
  NVTEShape shape;
};

/*! \enum NVTETensorParam
 *  \brief Indicates the kind of the tensor parameter to set/get.
 */
enum NVTETensorParam {
  kNVTERowwiseData = 0,        /*!< Data usable in rowwise manner */
  kNVTEColumnwiseData = 1,     /*!< Data usable in columnwise manner */
  kNVTEScale = 2,              /*!< Scale tensor */
  kNVTEAmax = 3,               /*!< Amax tensor */
  kNVTERowwiseScaleInv = 4,    /*!< Scale inverse tensor for decoding Rowwise Data */
  kNVTEColumnwiseScaleInv = 5, /*!< Scale inverse tensor for decoding Columnwise Data */
  kNVTENumTensorParams
};

/*! \enum NVTEScalingMode
 * \brief Tensor data format.
 */
enum NVTEScalingMode {
  /*! Either an unquantized tensor or an FP8 tensor with per-tensor scaling
   *
   * Not necessary used for delayed tensor scaling. The unintuitive
   * name reflects legacy usage.
   */
  NVTE_DELAYED_TENSOR_SCALING = 0,
  /*! Single scale per block of 32 elements consecutive in either
      rowwise or columnwise direction */
  NVTE_MXFP8_1D_SCALING = 1,
  /*! Tensor is split into NxN quantization tiles or 1xN quantization tiles,
    which each yield a scale. The block_scaling_dim property of the quantizer
    selects the granularity.
   */
  NVTE_BLOCK_SCALING_1D = 2,
  NVTE_BLOCK_SCALING_2D = 3,
  NVTE_INVALID_SCALING
};

/*! \brief TE Tensor type
 *
 * NVTETensor is a contiguous tensor type storing a pointer
 * to data of a given shape and type. It does not own the
 * memory it points to.
 */
typedef void *NVTETensor;

/*! \brief Create a new TE tensor.
 *
 * Create a new TE tensor. Before use its parameters need to be set.
 * TE tensors are just wrappers on top of raw data and do not
 * own memory.
 *
 *  \param[in] scaling_mode    Scaling mode of the tensor.
 *
 *  \return A new TE tensor.
 */
NVTETensor nvte_create_tensor(NVTEScalingMode scaling_mode);

/*! \brief Destroy a TE tensor.
 *
 * Since the TE tensor does not own memory, the underlying
 * data is not freed during this operation.
 *
 *  \param[in] tensor Tensor to be destroyed.
 */
void nvte_destroy_tensor(NVTETensor tensor);

/*! \brief Get a raw pointer to the tensor's rowwise data.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A raw pointer to tensor's rowwise data.
 */
void *nvte_tensor_data(const NVTETensor tensor);

/*! \brief Get a raw pointer to the tensor's columnwise data.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A raw pointer to tensor's columnwise data.
 */
void *nvte_tensor_columnwise_data(const NVTETensor tensor);

/*! \brief Get a tensor's data shape.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A shape of the input tensor.
 */
NVTEShape nvte_tensor_shape(const NVTETensor tensor);

/*! \brief Get a tensor's data shape.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A shape of the input tensor.
 */
NVTEShape nvte_tensor_columnwise_shape(const NVTETensor tensor);

/*! \brief Get a tensor's number of dimensions.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return Number of tensor dimensions.
 */
size_t nvte_tensor_ndims(const NVTETensor tensor);

/*! \brief Get the size of a specific tensor dimension.
 *
 *  \param[in] tensor Tensor.
 *  \param[in] size_t Dimension index.
 *
 *  \return Size of the tensor at the specified dimension.
 */
size_t nvte_tensor_size(const NVTETensor tensor, const size_t dim);

/*! \brief Get a tensor's total number of elements.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return Number of elements in the tensor.
 */
size_t nvte_tensor_numel(const NVTETensor tensor);

/*! \brief Get the byte size for the tensor's data type.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return Byte size of the tensor's data type.
 */
size_t nvte_tensor_element_size(const NVTETensor tensor);

/*! \brief Get a tensor's data type.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A data type of the input tensor.
 */
NVTEDType nvte_tensor_type(const NVTETensor tensor);

/*! \brief Get a pointer to the tensor's amax data.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A pointer to tensor's amax data.
 */
float *nvte_tensor_amax(const NVTETensor tensor);

/*! \brief Get a pointer to the tensor's scale data.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A pointer to tensor's scale data.
 */
float *nvte_tensor_scale(const NVTETensor tensor);

/*! \brief Get a pointer to the tensor's inverse of scale data.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A pointer to tensor's inverse of scale data.
 */
float *nvte_tensor_scale_inv(const NVTETensor tensor);

/*! \brief Get a tensor's scale_inv shape.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A scale_inv shape of the input tensor.
 */
NVTEShape nvte_tensor_scale_inv_shape(const NVTETensor tensor);

/*! \brief Reset tensor value to zero.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A scale_inv shape of the input tensor.
 */
void nvte_zero_tensor(const NVTETensor tensor, cudaStream_t stream);

/*! \brief Set a parameter of the tensor.
 *
 *  \param[in/out] tensor Tensor.
 *  \param[in] param_name The parameter to be set.
 *  \param[in] param The value to be set.
 */
void nvte_set_tensor_param(NVTETensor *tensor, NVTETensorParam param_name,
                           const NVTEBasicTensor *param);

/*! \brief Set a quantization option for whether to force power of 2 scales.
 *
 *  \param[in/out] tensor Tensor.
 *  \param[in] zero_if_false Whether to force power of 2 scales.
 *
 *  \return zero if the tensor supports this option and it was set. non-zero if
 *   call had no effect.
 */
int nvte_set_qopt_force_pow_2_scales(NVTETensor tensor, int zero_if_false);

/*! \brief Set a quantization option for epsilon to set floor of amax.
 *
 *  \param[in/out] tensor Tensor.
 *  \param[in] amax_epsilon Epsilon to use for amax calculation.
 *
 *  \return zero if the tensor supports this option and it was set. non-zero if
 *   call had no effect.
 */
int nvte_set_qopt_amax_epsilon(NVTETensor tensor, float amax_epsilon);

/*! \brief Get a quantization option for whether to force power of 2 scales.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return zero if the tensor will not force power of 2 scales or if the
 *   setting is irrelevant. non-zero if the flag is configured.
 */
int nvte_get_qopt_force_pow_2_scales(NVTETensor tensor);

/*! \brief Get a quantization option for amax epsilon.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return amax_epsilon value or zero if not applicable.
 */
float nvte_get_qopt_amax_epsilon(const NVTETensor tensor);

/*! \brief Get the number of dimensions in the quantization blocks.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return zero if the quantization does not support the block_scaling_dim
 *   option or the block_scaling_dim configured.
 */
int nvte_get_qopt_block_scaling_dim(const NVTETensor tensor);

/*! \brief Get a value of the parameter of the tensor.
 *
 *  \param[in] tensor Tensor.
 *  \param[in] param_name The parameter to be set.
 */
NVTEBasicTensor nvte_get_tensor_param(const NVTETensor tensor, NVTETensorParam param_name);

/*! \brief Get the granularity of scaling of this tensor.
 *
 *  \param[in] tensor Tensor.
 *
 *  \return A struct containing the granularity of tensor's scaling.
 */
NVTEScalingMode nvte_tensor_scaling_mode(const NVTETensor tensor);

/*! \struct NVTETensorPack
    \brief Pack of tensors, generally used for auxiliary outputs.
 */
struct NVTETensorPack {
  /*! Max number of tensors in the pack. Assumed <= 10. */
  static const int MAX_SIZE = 10;
  /*! Wrappers of tensors. They do not hold the associated memory. */
  NVTETensor tensors[MAX_SIZE];
  /*! Actual number of tensors in the pack, 0 <= size <= MAX_SIZE. */
  size_t size = 0;
};

/*! \brief Create `tensors` in NVTETensorPack.
 */
void nvte_tensor_pack_create(NVTETensorPack *pack);

/*! \brief Destroy `tensors` in NVTETensorPack.
 */
void nvte_tensor_pack_destroy(NVTETensorPack *pack);

/*! \brief Configuration for tensor quantization. */
typedef void *NVTEQuantizationConfig;

/*! \enum NVTEQuantizationConfigAttribute
 * \brief Type of option for tensor quantization.
 */
enum NVTEQuantizationConfigAttribute {
  /*! Whether to force power of 2 scales */
  kNVTEQuantizationConfigForcePow2Scales = 0,
  /*! Small value to add to amax for numerical stability */
  kNVTEQuantizationConfigAmaxEpsilon = 1,
  kNVTEQuantizationConfigNumAttributes
};

/*! \brief Create a new quantization config.
 *  \return A new quantization config.
 */
NVTEQuantizationConfig nvte_create_quantization_config();

/*! \brief Query an option in quantization config.
 *
 *  \param[in] config Quantization config.
 *  \param[in] attr Option type.
 *  \param[out] buf Memory address to write option value. Ignored if
 *                  NULL.
 *  \param[in] size_in_bytes Size of buf.
 *  \param[out] size_written Number of bytes that have been written to
 *                           buf. If buf is NULL, then the number of
 *                           bytes that would have been written.
 */
void nvte_get_quantization_config_attribute(NVTEQuantizationConfig config,
                                            NVTEQuantizationConfigAttribute attr, void *buf,
                                            size_t size_in_bytes, size_t *size_written);

/*! \brief Set an option in quantization config.
 *
 *  \param[in] config Quantization config.
 *  \param[in] attr Option type.
 *  \param[out] buf Memory address to read option value.
 *  \param[in] size_in_bytes Size of buf.
 */
void nvte_set_quantization_config_attribute(NVTEQuantizationConfig config,
                                            NVTEQuantizationConfigAttribute attr, const void *buf,
                                            size_t size_in_bytes);

/*! \brief Destroy a quantization config.
 *
 *  \param[in] config Config to be destroyed.
 */
void nvte_destroy_quantization_config(NVTEQuantizationConfig config);

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
  kByte = 0,
  kInt32 = 1,
  kInt64 = 2,
  kFloat32 = 3,
  kFloat16 = 4,
  kBFloat16 = 5,
  kFloat8E4M3 = 6,
  kFloat8E5M2 = 7,
  kFloat8E8M0 = 8,
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
   *  \param[in] amax_dptr       Pointer to the AMAX value.
   *  \param[in] scale_dptr      Pointer to the scale value.
   *  \param[in] scale_inv_shape Shape of scale_inv
   *  \param[in] scale_inv_dptr  Pointer to the inverse of scale value.
   */
  TensorWrapper(void *dptr, const NVTEShape &shape, const DType dtype, float *amax_dptr = nullptr,
                float *scale_dptr = nullptr, float *scale_inv_dptr = nullptr,
                const NVTEShape scale_inv_shape = defaultShape,
                const NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING) {
    tensor_ = nvte_create_tensor(scaling_mode);
    NVTEBasicTensor data = {dptr, static_cast<NVTEDType>(dtype), shape};
    nvte_set_tensor_param(&tensor_, kNVTERowwiseData, &data);
    NVTEBasicTensor amax = {amax_dptr, kNVTEFloat32, defaultShape};
    nvte_set_tensor_param(&tensor_, kNVTEAmax, &amax);
    NVTEBasicTensor scale = {scale_dptr, kNVTEFloat32, defaultShape};
    nvte_set_tensor_param(&tensor_, kNVTEScale, &scale);
    NVTEBasicTensor scale_inv = {scale_inv_dptr, kNVTEFloat32, scale_inv_shape};
    nvte_set_tensor_param(&tensor_, kNVTERowwiseScaleInv, &scale_inv);
  }

  /*! \brief Constructs new TensorWrapper.
   *
   * Create a new TE tensor with a given shape, datatype and data.
   * TE tensors are just wrappers on top of raw data and do not
   * own memory.
   *
   *  \param[in] dptr  Pointer to the tensor data.
   *  \param[in] shape Shape of the tensor.
   *  \param[in] dtype Data type of the tensor.
   *  \param[in] amax_dptr       Pointer to the AMAX value.
   *  \param[in] scale_dptr      Pointer to the scale value.
   *  \param[in] scale_inv_shape Shape of scale_inv
   *  \param[in] scale_inv_dptr  Pointer to the inverse of scale value.
   */
  TensorWrapper(void *dptr, const std::vector<size_t> &shape, const DType dtype,
                float *amax_dptr = nullptr, float *scale_dptr = nullptr,
                float *scale_inv_dptr = nullptr, const std::vector<size_t> &scale_inv_shape = {1},
                const NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING)
      : TensorWrapper(dptr, NVTEShape{shape.data(), shape.size()}, dtype, amax_dptr, scale_dptr,
                      scale_inv_dptr, NVTEShape{scale_inv_shape.data(), scale_inv_shape.size()},
                      scaling_mode) {}

  /*! \brief Constructs new empty TensorWrapper.
   *
   * Create a new empty TE tensor which holds nothing.
   */
  explicit TensorWrapper(const NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING)
      : tensor_(nvte_create_tensor(scaling_mode)) {}

  /*! \brief TensorWrapper destructor. */
  ~TensorWrapper() { nvte_destroy_tensor(tensor_); }

  TensorWrapper &operator=(const TensorWrapper &other) = delete;
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
  TensorWrapper &operator=(TensorWrapper &&other) {
    if (this == &other) return *this;
    nvte_destroy_tensor(tensor_);
    tensor_ = other.tensor_;
    other.tensor_ = nullptr;
    return *this;
  }

  // Parameter setters
  template <typename ShapeType>
  TensorWrapper &set_parameter(const NVTETensorParam param, void *dptr, DType type,
                               const ShapeType &shape) noexcept {
    NVTEShape nvte_shape = this->convertShape(shape);
    NVTEBasicTensor data = {dptr, static_cast<NVTEDType>(type), nvte_shape};
    nvte_set_tensor_param(&tensor_, param, &data);
    return *this;
  }

  template <typename ShapeType>
  TensorWrapper &set_rowwise_data(void *dptr, DType type, const ShapeType &shape) noexcept {
    return set_parameter(kNVTERowwiseData, dptr, type, shape);
  }

  template <typename ShapeType>
  TensorWrapper &set_columnwise_data(void *dptr, DType type, const ShapeType &shape) noexcept {
    return set_parameter(kNVTEColumnwiseData, dptr, type, shape);
  }

  template <typename ShapeType>
  TensorWrapper &set_scale(void *dptr, DType type, const ShapeType &shape) noexcept {
    return set_parameter(kNVTEScale, dptr, type, shape);
  }

  template <typename ShapeType>
  TensorWrapper &set_amax(void *dptr, DType type, const ShapeType &shape) noexcept {
    return set_parameter(kNVTEAmax, dptr, type, shape);
  }

  template <typename ShapeType>
  TensorWrapper &set_rowwise_scale_inv(void *dptr, DType type, const ShapeType &shape) noexcept {
    return set_parameter(kNVTERowwiseScaleInv, dptr, type, shape);
  }

  template <typename ShapeType>
  TensorWrapper &set_columnwise_scale_inv(void *dptr, DType type, const ShapeType &shape) noexcept {
    return set_parameter(kNVTEColumnwiseScaleInv, dptr, type, shape);
  }

  // Parameter getters

  NVTEBasicTensor get_parameter(const NVTETensorParam param) const noexcept {
    return nvte_get_tensor_param(tensor_, param);
  }

  NVTEBasicTensor get_rowwise_data() const noexcept { return get_parameter(kNVTERowwiseData); }

  NVTEBasicTensor get_columnwise_data() const noexcept {
    return get_parameter(kNVTEColumnwiseData);
  }

  NVTEBasicTensor get_scale() const noexcept { return get_parameter(kNVTEScale); }

  NVTEBasicTensor get_amax() const noexcept { return get_parameter(kNVTEAmax); }

  NVTEBasicTensor get_rowwise_scale_inv() const noexcept {
    return get_parameter(kNVTERowwiseScaleInv);
  }

  NVTEBasicTensor get_columnwise_scale_inv() const noexcept {
    return get_parameter(kNVTEColumnwiseScaleInv);
  }

  /*! \brief Get an underlying NVTETensor.
   *
   *  \return NVTETensor held by this TensorWrapper.
   */
  NVTETensor data() const noexcept { return tensor_; }

  /*! \brief Get the shape of this TensorWrapper.
   *
   *  \return Shape of this TensorWrapper.
   */
  const NVTEShape shape() const noexcept {
    if (tensor_ == nullptr) return NVTEShape{nullptr, 0};
    return nvte_tensor_shape(tensor_);
  }

  /*! \brief Get the shape of this TensorWrapper.
   *
   *  \return Shape of this TensorWrapper.
   */
  const NVTEShape columnwise_shape() const noexcept {
    if (tensor_ == nullptr) return NVTEShape{nullptr, 0};
    return nvte_tensor_columnwise_shape(tensor_);
  }

  /*! \brief Get the size of this TensorWrapper in the given dimension.
   *
   *  \param[in] size_t Dimension index.
   *
   *  \return Size of this TensorWrapper in given dimension.
   */
  size_t size(const size_t dim) const {
    if (tensor_ == nullptr) return 0;
    return nvte_tensor_size(tensor_, dim);
  }

  /*! \brief Get the number of dimensions for this TensorWrapper.
   *
   *  \return Number of dimensions for this TensorWrapper.
   */
  size_t ndim() const noexcept {
    if (tensor_ == nullptr) return 0;
    return nvte_tensor_ndims(tensor_);
  }

  /*! \brief Get the number of allocated elements in the tensor. This will return 0 for tensors
   *         with nullptr data even if the TensorWrapper has a non-zero shape.
   *
   *
   *  \return Number of elements in the tensor.
   */
  size_t numel() const noexcept {
    if (tensor_ == nullptr) return 0;
    return nvte_tensor_numel(tensor_);
  }

  /*! \brief Get the tensor's element size in bytes.
   *
   *  \return Element size in bytes.
   */
  size_t element_size() const noexcept {
    if (tensor_ == nullptr) return 0;
    return nvte_tensor_element_size(tensor_);
  }

  /*! \brief Get the tensor's allocated size in bytes. This will return 0 for tensors with nullptr
   *         data even if the TensorWrapper has a non-zero shape and valid dtype.
   *
   *  \return Total tensor size in bytes.
   */
  size_t bytes() const noexcept {
    if (tensor_ == nullptr || this->dptr() == nullptr) return 0;
    return nvte_tensor_numel(tensor_) * nvte_tensor_element_size(tensor_);
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

  /*! \brief Get a raw pointer to the tensor's data.
   *
   *  \return A raw pointer to tensor's data.
   */
  void *columnwise_dptr() const noexcept {
    if (tensor_ == nullptr) return nullptr;
    return nvte_tensor_columnwise_data(tensor_);
  }

  /*! \brief Get a pointer to the tensor's amax data.
   *
   *  \return A pointer to tensor's amax data.
   */
  float *amax() const noexcept {
    if (tensor_ == nullptr) return nullptr;
    return nvte_tensor_amax(tensor_);
  }

  /*! \brief Get a pointer to the tensor's scale data.
   *
   *  \return A pointer to tensor's scale data.
   */
  float *scale() const noexcept {
    if (tensor_ == nullptr) return nullptr;
    return nvte_tensor_scale(tensor_);
  }

  /*! \brief Get a pointer to the tensor's inverse of scale data.
   *
   *  \return A pointer to tensor's inverse of scale data.
   */
  float *scale_inv() const noexcept {
    if (tensor_ == nullptr) return nullptr;
    return nvte_tensor_scale_inv(tensor_);
  }

  /*! \brief Get the scale_inv_shape of this TensorWrapper.
   *
   *  \return scale_inv_shape of this TensorWrapper.
   */
  const NVTEShape scale_inv_shape() const noexcept {
    if (tensor_ == nullptr) return NVTEShape{nullptr, 0};
    return nvte_tensor_scale_inv_shape(tensor_);
  }

  /*! \brief Get a scaling mode of the tensor.
   *
   *  \return Scaling mode of the tensor.
   */
  NVTEScalingMode scaling_mode() const noexcept {
    if (tensor_ == nullptr) return NVTE_DELAYED_TENSOR_SCALING;
    return nvte_tensor_scaling_mode(tensor_);
  }

  void zero_(cudaStream_t stream) { nvte_zero_tensor(tensor_, stream); }

  int set_qopt_force_pow_2_scales(bool flag) {
    return nvte_set_qopt_force_pow_2_scales(tensor_, flag ? 1 : 0);
  }

  int set_qopt_amax_epsilon(float eps) { return nvte_set_qopt_amax_epsilon(tensor_, eps); }

  bool get_qopt_force_pow_2_scales() const {
    return nvte_get_qopt_force_pow_2_scales(tensor_) != 0;
  }

  float get_qopt_amax_epsilon() const { return nvte_get_qopt_amax_epsilon(tensor_); }

  static constexpr size_t defaultData = 1;
  static constexpr NVTEShape defaultShape = {&defaultData, 1};

 private:
  NVTEShape convertShape(const NVTEShape &s) { return s; }

  NVTEShape convertShape(const std::vector<size_t> &s) { return {s.data(), s.size()}; }

  /*! \brief Wrapped NVTETensor. */
  NVTETensor tensor_ = nullptr;
};

/*! \struct QuantizationConfigWrapper
 *  \brief C++ wrapper for NVTEQuantizationConfigWrapper.
 */
class QuantizationConfigWrapper {
 public:
  QuantizationConfigWrapper() : config_{nvte_create_quantization_config()} {}

  QuantizationConfigWrapper(const QuantizationConfigWrapper &) = delete;
  QuantizationConfigWrapper &operator=(const QuantizationConfigWrapper &) = delete;

  QuantizationConfigWrapper(QuantizationConfigWrapper &&other) : config_{other.config_} {
    other.config_ = nullptr;
  }
  QuantizationConfigWrapper &operator=(QuantizationConfigWrapper &&other) {
    if (config_ != nullptr) {
      nvte_destroy_quantization_config(config_);
    }
    config_ = other.config_;
    other.config_ = nullptr;
    return *this;
  }

  ~QuantizationConfigWrapper() {
    if (config_ != nullptr) {
      nvte_destroy_quantization_config(config_);
      config_ = nullptr;
    }
  }

  /*! \brief Get the underlying NVTEQuantizationConfig.
   *
   *  \return NVTEQuantizationConfig held by this QuantizationConfigWrapper.
   */
  operator NVTEQuantizationConfig() const noexcept { return config_; }

  /*! \brief Set whether to force power of 2 scales */
  void set_force_pow_2_scales(bool force_pow_2_scales) {
    nvte_set_quantization_config_attribute(config_, kNVTEQuantizationConfigForcePow2Scales,
                                           &force_pow_2_scales, sizeof(bool));
  }

  /*! \brief Set small value to add to amax */
  void set_amax_epsilon(float amax_epsilon) {
    nvte_set_quantization_config_attribute(config_, kNVTEQuantizationConfigAmaxEpsilon,
                                           &amax_epsilon, sizeof(float));
  }

 private:
  /*! \brief Wrapped NVTEQuantizationConfig. */
  NVTEQuantizationConfig config_ = nullptr;
};

}  // namespace transformer_engine

#endif  // __cplusplus

#endif  // TRANSFORMER_ENGINE_TRANSFORMER_ENGINE_H_
