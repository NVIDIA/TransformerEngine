/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <pybind11/pybind11.h>

#include <string>

#include "common/include/transformer_engine/activation.h"
#include "common/include/transformer_engine/cast.h"
#include "common/include/transformer_engine/gemm.h"
#include "common/include/transformer_engine/layer_norm.h"
#include "common/include/transformer_engine/softmax.h"
#include "common/include/transformer_engine/transformer_engine.h"
#include "common/include/transformer_engine/transpose.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace transformer_engine {

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8FwdTensors {
  GEMM1_INPUT = 0,
  GEMM1_WEIGHT = 1,
  GEMM2_INPUT = 2,
  GEMM2_WEIGHT = 3
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8BwdTensors { GRAD_OUTPUT1 = 0, GRAD_OUTPUT2 = 1 };

}  // namespace transformer_engine

namespace {

void CheckTensorIsOnGPU(TFE_TensorHandle* tensor, TF_Status* status) {
  const char* device_type = TFE_TensorHandleDeviceType(tensor, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  CHECK_EQ(std::string(device_type), std::string("GPU"))
      << "Tensor must be on the GPU, but got device_type=" << device_type;
}

std::vector<size_t> TensorShapeAsVector(TFE_TensorHandle* tensor,
                                        TF_Status* status) {
  std::vector<size_t> shape(TFE_TensorHandleNumDims(tensor, status));
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = TFE_TensorHandleDim(tensor, i, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }
  return shape;
}

transformer_engine::DType GetNVTEDataType(TF_DataType t) {
  switch (t) {
    case TF_HALF:
      return transformer_engine::DType::kFloat16;
    case TF_FLOAT:
      return transformer_engine::DType::kFloat32;
    case TF_BFLOAT16:
      return transformer_engine::DType::kBFloat16;
    case TF_BOOL:
    case TF_INT8:
      return transformer_engine::DType::kByte;
    case TF_INT32:
      return transformer_engine::DType::kInt32;
    default:
      CHECK(false) << "TF dtype is not supported: " << t;
  }
}

TF_DataType GetTFDataType(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kFloat16:
      return TF_HALF;
    case transformer_engine::DType::kFloat32:
      return TF_FLOAT;
    case transformer_engine::DType::kBFloat16:
      return TF_BFLOAT16;
    case transformer_engine::DType::kByte:
    case transformer_engine::DType::kFloat8E4M3:
    case transformer_engine::DType::kFloat8E5M2:
      return TF_INT8;
    case transformer_engine::DType::kInt32:
      return TF_INT32;
    default:
      CHECK(false) << "NVTE dtype is not supported: " << static_cast<int>(t);
  }
}

void* TFE_TensorHandleDevicePointerNoSync(TFE_TensorHandle* h,
                                          TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  tensorflow::ImmediateExecutionTensorHandle* unwrapped_handle =
      tensorflow::unwrap(h);
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (tensorflow::CustomDeviceTensorHandle::classof(unwrapped_handle)) {
    return tensorflow::down_cast<tensorflow::CustomDeviceTensorHandle*>(
               unwrapped_handle)
        ->DevicePointer();
  }
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (!tensorflow::TensorHandle::classof(unwrapped_handle)) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  tensorflow::TensorHandle* handle =
      tensorflow::TensorHandleFromInterface(unwrapped_handle);

  if (handle->Type() != tensorflow::TensorHandle::LOCAL) {
    status->status = tensorflow::errors::InvalidArgument(
        "TFE_TensorHandleDevicePointer may not be called on a ",
        handle->TypeString(), " tensor handle.");
    return nullptr;
  }

  const tensorflow::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return nullptr;
  }
  return const_cast<void*>(
      static_cast<const void*>(tensor->tensor_data().data()));
}

// We assume the dptr is float when applying the offset. The offset is only
// meaningful for the amax/scale/scale_inv tensors.
void* GetDevicePtr(const pybind11::handle& handle, int offset = 0) {
  if (offset == -1) return nullptr;

  CHECK(EagerTensor_CheckExact(handle.ptr())) << "EagerTensor required!";
  auto in_eager = EagerTensor_Handle(handle.ptr());

  auto status = TF_NewStatus();
  CheckTensorIsOnGPU(in_eager, status);

  void* in_dptr = nullptr;
  if (in_eager) {
    in_dptr = TFE_TensorHandleDevicePointerNoSync(in_eager, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }

  TF_DeleteStatus(status);

  return reinterpret_cast<float*>(in_dptr) + offset;
}

std::vector<size_t> GetShape(const pybind11::handle& handle) {
  TFE_TensorHandle* in_eager = EagerTensor_Handle(handle.ptr());
  TF_Status* status = TF_NewStatus();
  std::vector<size_t> shape = TensorShapeAsVector(in_eager, status);
  TF_DeleteStatus(status);
  return shape;
}

transformer_engine::DType GetDataType(const pybind11::handle& handle) {
  TFE_TensorHandle* in_eager = EagerTensor_Handle(handle.ptr());
  auto tf_itype = TFE_TensorHandleDataType(in_eager);
  return GetNVTEDataType(tf_itype);
}

transformer_engine::TensorWrapper MakeNVTETensor(
    void* data_ptr, const std::vector<size_t>& shape,
    const transformer_engine::DType type, void* amax_ptr = nullptr,
    void* scale_ptr = nullptr, void* scale_inv_ptr = nullptr) {
  return transformer_engine::TensorWrapper(
      data_ptr, shape, type, reinterpret_cast<float*>(amax_ptr),
      reinterpret_cast<float*>(scale_ptr),
      reinterpret_cast<float*>(scale_inv_ptr));
}

tensorflow::Allocator* GetAllocator() {
  static tensorflow::Allocator* allocator = nullptr;
  if (allocator == nullptr) {
    tensorflow::GPUOptions gpu_options;
    tsl::TfDeviceId device_id(0);
    allocator = tensorflow::GPUProcessState::singleton()->GetGPUAllocator(
        gpu_options, device_id, /*total_bytes=*/1, /*peer_gpu_ids=*/{});
  }
  return allocator;
}

TFE_Context* GetContext(TF_Status* status) {
  // Cache TF context.
  static TFE_Context* context = nullptr;
  if (context == nullptr) {
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    context = TFE_NewContext(opts, status);
  }
  return context;
}

void Deallocator(void* data, size_t unused, void* tensor_handle) {
  GetAllocator()->DeallocateRaw(data);
}

void* AllocateSpace(const std::vector<size_t>& shape,
                    transformer_engine::DType te_dtype, cudaStream_t stream = 0,
                    bool init_to_zeros = false) {
  auto dtype = GetTFDataType(te_dtype);

  // Allocate GPU memory.
  size_t num_bytes = TF_DataTypeSize(dtype);
  for (int i = 0; i < shape.size(); ++i) num_bytes *= shape[i];
  void* data = GetAllocator()->AllocateRaw(
      tensorflow::Allocator::kAllocatorAlignment, num_bytes);
  if (init_to_zeros) {
    CHECK_EQ(cudaMemsetAsync(data, 0, num_bytes, stream), cudaSuccess);
  }
  return data;
}

TFE_TensorHandle* CreateTensor(void* data, const std::vector<size_t>& shape,
                               transformer_engine::DType te_dtype) {
  auto dtype = GetTFDataType(te_dtype);

  size_t num_bytes = TF_DataTypeSize(dtype);
  for (int i = 0; i < shape.size(); ++i) num_bytes *= shape[i];

  TF_Status* status = TF_NewStatus();
  TFE_Context* ctx = GetContext(status);

  // Get first GPU device name.
  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  int num_devices = TF_DeviceListCount(devices);
  const char* device_name = nullptr;
  for (int i = 0; i < num_devices; ++i) {
    const char* name = TF_DeviceListName(devices, i, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    if (std::string(name).find("GPU") != std::string::npos) {
      device_name = name;
      break;
    }
  }
  CHECK_NE(device_name, nullptr) << "No GPU device found.";

  std::vector<int64_t> shape64(shape.size());
  std::transform(shape.cbegin(), shape.cend(), shape64.begin(),
                 [](const size_t& v) { return static_cast<int64_t>(v); });
  TFE_TensorHandle* tensor = TFE_NewTensorHandleFromDeviceMemory(
      ctx, device_name, dtype, shape64.data(), shape64.size(), data, num_bytes,
      &Deallocator, nullptr, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  return tensor;
}

void dispatch_cast_transpose_fusion(
    void* input,  // i
    const std::vector<size_t>& input_shape,
    const transformer_engine::DType input_type,
    void* scale,  // i
    const std::vector<size_t>& scale_shape,
    const transformer_engine::DType scale_type,
    void* output_cast,  // o
    const std::vector<size_t>& output_cast_shape,
    const transformer_engine::DType output_cast_type,
    void* output_transpose,  // o
    const std::vector<size_t>& output_transpose_shape,
    const transformer_engine::DType output_transpose_type,
    void* amax,  // o
    const std::vector<size_t>& amax_shape,
    const transformer_engine::DType amax_type,
    void* scale_inv,  // o
    const std::vector<size_t>& scale_inv_shape,
    const transformer_engine::DType scale_inv_type, cudaStream_t stream) {
  auto input_cu = MakeNVTETensor(input, input_shape, input_type);
  auto output_cast_cu = MakeNVTETensor(
      output_cast, output_cast_shape, output_cast_type, amax, scale, scale_inv);
  auto output_transpose_cu =
      MakeNVTETensor(output_transpose, output_transpose_shape,
                     output_transpose_type, amax, scale, scale_inv);

  nvte_cast_transpose(input_cu.data(), output_cast_cu.data(),
                      output_transpose_cu.data(), stream);
}

void dispatch_transpose(void* input,  // i
                        const std::vector<size_t>& input_shape,
                        const transformer_engine::DType input_type,
                        void* output,  // o
                        const std::vector<size_t>& output_shape,
                        const transformer_engine::DType output_type,
                        cudaStream_t stream) {
  auto input_cu = MakeNVTETensor(input, input_shape, input_type);
  auto output_cu = MakeNVTETensor(output, output_shape, output_type);

  nvte_transpose(input_cu.data(), output_cu.data(), stream);
}

void dispatch_bgrad_cast_transpose_fusion(
    void* input,  // i
    const std::vector<size_t>& input_shape,
    const transformer_engine::DType input_type,
    void* scale,  // i
    const std::vector<size_t>& scale_shape,
    const transformer_engine::DType scale_type,
    void* cast_output,  // o
    const std::vector<size_t>& cast_output_shape,
    const transformer_engine::DType cast_output_type,
    void* transposed_output,  // o
    const std::vector<size_t>& transposed_output_shape,
    const transformer_engine::DType transposed_output_type,
    void* amax,  // o
    const std::vector<size_t>& amax_shape,
    const transformer_engine::DType amax_type,
    void* dbias,  // o
    const std::vector<size_t>& dbias_shape,
    const transformer_engine::DType dbias_type,
    void* scale_inv,  // o
    const std::vector<size_t>& scale_inv_shape,
    const transformer_engine::DType scale_inv_type, cudaStream_t stream) {
  auto input_cu = MakeNVTETensor(input, input_shape, input_type);
  auto cast_output_cu = MakeNVTETensor(
      cast_output, cast_output_shape, cast_output_type, amax, scale, scale_inv);
  auto transposed_output_cu =
      MakeNVTETensor(transposed_output, transposed_output_shape,
                     transposed_output_type, amax, scale, scale_inv);
  auto dbias_cu = MakeNVTETensor(dbias, dbias_shape, dbias_type);
  transformer_engine::TensorWrapper workspace;

  nvte_cast_transpose_dbias(input_cu.data(), cast_output_cu.data(),
                            transposed_output_cu.data(), dbias_cu.data(),
                            workspace.data(), stream);

  // Fill workspace
  auto w_s = workspace.shape();
  std::vector<size_t> w_shape_vec{w_s.data, w_s.data + w_s.ndim};

  void* workspace_ptr = AllocateSpace(w_shape_vec, workspace.dtype());

  workspace = MakeNVTETensor(workspace_ptr, w_shape_vec, workspace.dtype());

  nvte_cast_transpose_dbias(input_cu.data(), cast_output_cu.data(),
                            transposed_output_cu.data(), dbias_cu.data(),
                            workspace.data(), stream);
}

void dispatch_layernorm(void* input,  // i
                        const std::vector<size_t>& input_shape,
                        const transformer_engine::DType input_type,
                        void* gamma,  // i
                        const std::vector<size_t>& gamma_shape,
                        const transformer_engine::DType gamma_type,
                        void* beta,  // i
                        const std::vector<size_t>& beta_shape,
                        const transformer_engine::DType beta_type,
                        void* scale,  // i
                        const std::vector<size_t>& scale_shape,
                        const transformer_engine::DType scale_type,
                        const float epsilon,  // i
                        void* z,              // o
                        const std::vector<size_t>& z_shape,
                        const transformer_engine::DType z_type,
                        void* mu,  // o
                        const std::vector<size_t>& mu_shape,
                        const transformer_engine::DType mu_type,
                        void* rsigma,  // o
                        const std::vector<size_t>& rsigma_shape,
                        const transformer_engine::DType rsigma_type,
                        void* amax,  // o
                        const std::vector<size_t>& amax_shape,
                        const transformer_engine::DType amax_type,
                        void* scale_inv,  // o
                        const std::vector<size_t>& scale_inv_shape,
                        const transformer_engine::DType scale_inv_type,
                        const int multiProcessorCount, cudaStream_t stream) {
  auto input_cu = MakeNVTETensor(input, input_shape, input_type);
  auto gamma_cu = MakeNVTETensor(gamma, gamma_shape, gamma_type);
  auto beta_cu = MakeNVTETensor(beta, beta_shape, beta_type);
  auto z_cu = MakeNVTETensor(z, z_shape, z_type, amax, scale, scale_inv);
  auto mu_cu = MakeNVTETensor(mu, mu_shape, mu_type);
  auto rsigma_cu = MakeNVTETensor(rsigma, rsigma_shape, rsigma_type);

  transformer_engine::TensorWrapper workspace, barrier;

  // This call populates workspace and barrier tensors with the required config
  nvte_layernorm_fwd(input_cu.data(), gamma_cu.data(), beta_cu.data(), epsilon,
                     z_cu.data(), mu_cu.data(), rsigma_cu.data(), stream,
                     multiProcessorCount, workspace.data(), barrier.data());

  // Fill workspace and barrier
  auto w_s = workspace.shape();
  auto b_s = barrier.shape();
  std::vector<size_t> w_shape_vec{w_s.data, w_s.data + w_s.ndim};
  std::vector<size_t> b_shape_vec{b_s.data, b_s.data + b_s.ndim};

  void* workspace_ptr = AllocateSpace(w_shape_vec, workspace.dtype());
  void* barrier_ptr = AllocateSpace(b_shape_vec, barrier.dtype(), stream, true);

  workspace = MakeNVTETensor(workspace_ptr, w_shape_vec, workspace.dtype());
  barrier = MakeNVTETensor(barrier_ptr, b_shape_vec, barrier.dtype());

  // Actual call to fwd kernel
  nvte_layernorm_fwd(input_cu.data(), gamma_cu.data(), beta_cu.data(), epsilon,
                     z_cu.data(), mu_cu.data(), rsigma_cu.data(), stream,
                     multiProcessorCount, workspace.data(), barrier.data());
}

void dispatch_gelu(void* input,  // i
                   const std::vector<size_t>& input_shape,
                   const transformer_engine::DType input_type,
                   void* scale,  // i
                   const std::vector<size_t>& scale_shape,
                   const transformer_engine::DType scale_type,
                   void* output,  // o
                   const std::vector<size_t>& output_shape,
                   const transformer_engine::DType output_type,
                   void* amax,  // o
                   const std::vector<size_t>& amax_shape,
                   const transformer_engine::DType amax_type,
                   void* scale_inv,  // o
                   const std::vector<size_t>& scale_inv_shape,
                   const transformer_engine::DType scale_inv_type,
                   cudaStream_t stream) {
  auto input_cu = MakeNVTETensor(input, input_shape, input_type);
  auto output_cu =
      MakeNVTETensor(output, output_shape, output_type, amax, scale, scale_inv);

  nvte_gelu(input_cu.data(), output_cu.data(), stream);
}

void dispatch_bgrad_dgelu_cast_transpose_fusion(
    void* input,  // i
    const std::vector<size_t>& input_shape,
    const transformer_engine::DType input_type,
    void* gelu_input,  // i
    const std::vector<size_t>& gelu_input_shape,
    const transformer_engine::DType gelu_input_type,
    void* scale,  // i
    const std::vector<size_t>& scale_shape,
    const transformer_engine::DType scale_type,
    void* cast_output,  // o
    const std::vector<size_t>& cast_output_shape,
    const transformer_engine::DType cast_output_type,
    void* transposed_output,  // o
    const std::vector<size_t>& transposed_output_shape,
    const transformer_engine::DType transposed_output_type,
    void* amax,  // o
    const std::vector<size_t>& amax_shape,
    const transformer_engine::DType amax_type,
    void* dbias,  // o
    const std::vector<size_t>& dbias_shape,
    const transformer_engine::DType dbias_type,
    void* scale_inv,  // o
    const std::vector<size_t>& scale_inv_shape,
    const transformer_engine::DType scale_inv_type, cudaStream_t stream) {
  auto gelu_input_cu =
      MakeNVTETensor(gelu_input, gelu_input_shape, gelu_input_type);
  auto input_cu = MakeNVTETensor(input, input_shape, input_type);
  auto cast_output_cu = MakeNVTETensor(
      cast_output, cast_output_shape, cast_output_type, amax, scale, scale_inv);
  auto transposed_output_cu =
      MakeNVTETensor(transposed_output, transposed_output_shape,
                     transposed_output_type, amax, scale, scale_inv);
  auto dbias_cu = MakeNVTETensor(dbias, dbias_shape, dbias_type);

  transformer_engine::TensorWrapper workspace;

  nvte_cast_transpose_dbias_dgelu(
      input_cu.data(), gelu_input_cu.data(), cast_output_cu.data(),
      transposed_output_cu.data(), dbias_cu.data(), workspace.data(), stream);

  // Fill workspace
  auto w_s = workspace.shape();
  std::vector<size_t> w_shape_vec{w_s.data, w_s.data + w_s.ndim};

  void* workspace_ptr = AllocateSpace(w_shape_vec, workspace.dtype());

  workspace = MakeNVTETensor(workspace_ptr, w_shape_vec, workspace.dtype());

  nvte_cast_transpose_dbias_dgelu(
      input_cu.data(), gelu_input_cu.data(), cast_output_cu.data(),
      transposed_output_cu.data(), dbias_cu.data(), workspace.data(), stream);
}

TFE_TensorHandle* GetTFETensorHandle(const pybind11::handle tensor) {
  CHECK(EagerTensor_CheckExact(tensor.ptr()))
      << "All inputs must be EagerTensors.";
  return EagerTensor_Handle(tensor.ptr());
}

int GetDeviceMultiProcessorCount() {
  static int count = [] {
    cudaDeviceProp properties;
    // Get current device
    int device = -1;
    CHECK_EQ(cudaGetDevice(&device), cudaSuccess)
        << "Got invalid GPU" << device;
    CHECK_EQ(cudaGetDeviceProperties(&properties, device), cudaSuccess);
    return properties.multiProcessorCount;
  }();

  return count;
}

py::object TFE_Py_TeGemm_wrapper(
    const pybind11::handle& a_mat, const pybind11::handle& a_scale_inv,
    const transformer_engine::DType atype, const int a_offset,
    const pybind11::handle& b_mat, const pybind11::handle& b_scale_inv,
    const transformer_engine::DType btype, const int b_offset,
    const pybind11::handle& workspace, const bool use_bias,
    const pybind11::handle& bias, const bool use_gelu,
    const pybind11::handle& gelu_input, const bool transa,
    const bool transb, const bool grad, const bool accumulate,
    const bool use_split_accumulate, const transformer_engine::DType otype,
    const int64_t stream_id) {
  using namespace transformer_engine;

  std::vector<size_t> a_shape = GetShape(a_mat);
  std::vector<size_t> b_shape = GetShape(b_mat);
  CHECK_EQ(a_shape.size(), 2);
  CHECK_EQ(b_shape.size(), 2);

  std::vector<size_t> d_shape{transb ? b_shape[1] : b_shape[0],
                              transa ? a_shape[0] : a_shape[1]};

  auto a_tensor =
      MakeNVTETensor(GetDevicePtr(a_mat), a_shape, atype, nullptr,
                     nullptr, GetDevicePtr(a_scale_inv, a_offset));

  auto b_tensor =
      MakeNVTETensor(GetDevicePtr(b_mat), b_shape, btype, nullptr,
                     nullptr, GetDevicePtr(b_scale_inv, b_offset));

  NVTEShape empty_shape;
  TensorWrapper bias_tensor(nullptr, empty_shape, DType::kBFloat16);
  if (use_bias) {
    bias_tensor = MakeNVTETensor(GetDevicePtr(bias), GetShape(bias),
                                 GetDataType(bias));
  }

  TensorWrapper gelu_input_tensor(nullptr, empty_shape, DType::kBFloat16);
  void* gelu_input_ptr = nullptr;
  if (use_gelu && !grad) {
    gelu_input_ptr = AllocateSpace(d_shape, otype);
    gelu_input_tensor = MakeNVTETensor(gelu_input_ptr, d_shape, otype);
  } else if (use_gelu) {
    gelu_input_tensor =
        MakeNVTETensor(GetDevicePtr(gelu_input), GetShape(gelu_input),
                       GetDataType(gelu_input));
  }

  auto workspace_tensor =
      MakeNVTETensor(GetDevicePtr(workspace), GetShape(workspace),
                     GetDataType(workspace));

  void* d_ptr = AllocateSpace(d_shape, otype);
  auto d_tensor = MakeNVTETensor(d_ptr, d_shape, otype);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
  nvte_cublas_gemm(a_tensor.data(), b_tensor.data(), d_tensor.data(),
                   bias_tensor.data(), gelu_input_tensor.data(), transa,
                   transb, grad, workspace_tensor.data(), accumulate,
                   use_split_accumulate, stream);

  auto d_eager = CreateTensor(d_ptr, d_shape, otype);
  if (use_gelu && !grad) {
    auto gelu_input_eager = CreateTensor(gelu_input_ptr, d_shape, otype);
    PyObject* result(PyList_New(2));
    PyList_SET_ITEM(result, 0, EagerTensorFromHandle(d_eager));
    PyList_SET_ITEM(result, 1, EagerTensorFromHandle(gelu_input_eager));
    return tensorflow::PyoOrThrow(result);
  }
  return tensorflow::PyoOrThrow(EagerTensorFromHandle(d_eager));
}
      
}  // end namespace

PYBIND11_MODULE(transformer_engine_tensorflow, m) {
  py::enum_<transformer_engine::DType>(m, "DType")
      .value("kByte", transformer_engine::DType::kByte)
      .value("kInt32", transformer_engine::DType::kInt32)
      .value("kFloat32", transformer_engine::DType::kFloat32)
      .value("kFloat16", transformer_engine::DType::kFloat16)
      .value("kBFloat16", transformer_engine::DType::kBFloat16)
      .value("kFloat8E4M3", transformer_engine::DType::kFloat8E4M3)
      .value("kFloat8E5M2", transformer_engine::DType::kFloat8E5M2);

  py::enum_<transformer_engine::FP8FwdTensors>(m, "FP8FwdTensors",
                                               py::arithmetic())
      .value("GEMM1_INPUT", transformer_engine::FP8FwdTensors::GEMM1_INPUT)
      .value("GEMM1_WEIGHT", transformer_engine::FP8FwdTensors::GEMM1_WEIGHT)
      .value("GEMM2_INPUT", transformer_engine::FP8FwdTensors::GEMM2_INPUT)
      .value("GEMM2_WEIGHT", transformer_engine::FP8FwdTensors::GEMM2_WEIGHT);

  py::enum_<transformer_engine::FP8BwdTensors>(m, "FP8BwdTensors",
                                               py::arithmetic())
      .value("GRAD_OUTPUT1", transformer_engine::FP8BwdTensors::GRAD_OUTPUT1)
      .value("GRAD_OUTPUT2", transformer_engine::FP8BwdTensors::GRAD_OUTPUT2);

  m.def("cast_to_fp8",
        [](const pybind11::handle& input, const pybind11::handle& scale,
           const transformer_engine::DType otype, const pybind11::handle& amax,
           const pybind11::handle& scale_inv, const int offset,
           const int64_t stream_id) {
          std::vector<size_t> shape_c = GetShape(input);
          CHECK_EQ(shape_c.size(), 2);

          auto input_tensor =
              MakeNVTETensor(GetDevicePtr(input), shape_c, GetDataType(input));

          void* out_c_ptr = AllocateSpace(shape_c, otype);

          auto output_tensor = MakeNVTETensor(
              out_c_ptr, shape_c, otype, GetDevicePtr(amax, offset),
              GetDevicePtr(scale, offset), GetDevicePtr(scale_inv, offset));

          cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
          nvte_fp8_quantize(input_tensor.data(), output_tensor.data(), stream);

          auto out_c_eager = CreateTensor(out_c_ptr, shape_c, otype);
          return tensorflow::PyoOrThrow(EagerTensorFromHandle(out_c_eager));
        });
  m.def("cast_from_fp8", [](const pybind11::handle& input,
                            const pybind11::handle& scale_inv,
                            const transformer_engine::DType itype,
                            const transformer_engine::DType otype,
                            const int offset, const int64_t stream_id) {
    std::vector<size_t> shape_c = GetShape(input);
    CHECK_EQ(shape_c.size(), 2);

    auto input_tensor =
        MakeNVTETensor(GetDevicePtr(input), shape_c, itype, nullptr, nullptr,
                       GetDevicePtr(scale_inv, offset));

    void* out_ptr = AllocateSpace(shape_c, otype);

    auto output_tensor = MakeNVTETensor(out_ptr, shape_c, otype);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    nvte_fp8_dequantize(input_tensor.data(), output_tensor.data(), stream);

    auto out_eager = CreateTensor(out_ptr, shape_c, otype);
    return tensorflow::PyoOrThrow(EagerTensorFromHandle(out_eager));
  });
  m.def("fp8_cast_transpose_fused",
        [](const pybind11::handle& input, const pybind11::handle& scale,
           const transformer_engine::DType otype, const pybind11::handle& amax,
           const pybind11::handle& scale_inv, const int offset,
           const int64_t stream_id) {
          using namespace transformer_engine;

          std::vector<size_t> shape_c = GetShape(input);
          CHECK_EQ(shape_c.size(), 2);
          std::vector<size_t> shape_t{shape_c[1], shape_c[0]};

          void* out_c_ptr = AllocateSpace(shape_c, otype);
          void* out_t_ptr = AllocateSpace(shape_t, otype);

          cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
          dispatch_cast_transpose_fusion(
              GetDevicePtr(input), shape_c, GetDataType(input),
              GetDevicePtr(scale, offset), {1}, DType::kFloat32, out_c_ptr,
              shape_c, otype, out_t_ptr, shape_t, otype,
              GetDevicePtr(amax, offset), {1}, DType::kFloat32,
              GetDevicePtr(scale_inv, offset), {1}, DType::kFloat32, stream);

          auto out_c_eager = CreateTensor(out_c_ptr, shape_c, otype);
          auto out_t_eager = CreateTensor(out_t_ptr, shape_t, otype);
          PyObject* result(PyList_New(2));
          PyList_SET_ITEM(result, 0, EagerTensorFromHandle(out_c_eager));
          PyList_SET_ITEM(result, 1, EagerTensorFromHandle(out_t_eager));
          return tensorflow::PyoOrThrow(result);
        });
  m.def("fp8_transpose", [](const pybind11::handle& input,
                            transformer_engine::DType otype,
                            const int64_t stream_id) {
    std::vector<size_t> shape_c = GetShape(input);
    CHECK_EQ(shape_c.size(), 2);
    std::vector<size_t> shape_t{shape_c[1], shape_c[0]};

    void* out_t_ptr = AllocateSpace(shape_t, otype);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    dispatch_transpose(GetDevicePtr(input), shape_c, otype, out_t_ptr, shape_t,
                       otype, stream);

    TFE_TensorHandle* out_t_eager = CreateTensor(out_t_ptr, shape_t, otype);
    return tensorflow::PyoOrThrow(EagerTensorFromHandle(out_t_eager));
  });
  m.def("fp8_cast_transpose_bgrad_fused",
        [](const pybind11::handle& grad_out, const pybind11::handle& scale,
           const transformer_engine::DType otype, const pybind11::handle& amax,
           const pybind11::handle& scale_inv, const int offset,
           const int64_t stream_id) {
          using namespace transformer_engine;

          std::vector<size_t> shape_c = GetShape(grad_out);
          CHECK_EQ(shape_c.size(), 2);
          std::vector<size_t> shape_t{shape_c[1], shape_c[0]};
          std::vector<size_t> shape_b{shape_c[1]};

          auto itype = GetDataType(grad_out);
          void* grad_bias_ptr = AllocateSpace(shape_b, itype);
          void* grad_out_c_ptr = AllocateSpace(shape_c, otype);
          void* grad_out_t_ptr = AllocateSpace(shape_t, otype);

          cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
          dispatch_bgrad_cast_transpose_fusion(
              GetDevicePtr(grad_out), shape_c, itype,
              GetDevicePtr(scale, offset), {1}, DType::kFloat32, grad_out_c_ptr,
              shape_c, otype, grad_out_t_ptr, shape_t, otype,
              GetDevicePtr(amax, offset), {1}, DType::kFloat32, grad_bias_ptr,
              shape_b, itype, GetDevicePtr(scale_inv, offset), {1},
              DType::kFloat32, stream);

          auto grad_bias_eager = CreateTensor(grad_bias_ptr, shape_b, itype);
          auto grad_out_c_eager = CreateTensor(grad_out_c_ptr, shape_c, otype);
          auto grad_out_t_eager = CreateTensor(grad_out_t_ptr, shape_t, otype);
          PyObject* result(PyList_New(3));
          PyList_SET_ITEM(result, 0, EagerTensorFromHandle(grad_bias_eager));
          PyList_SET_ITEM(result, 1, EagerTensorFromHandle(grad_out_c_eager));
          PyList_SET_ITEM(result, 2, EagerTensorFromHandle(grad_out_t_eager));
          return tensorflow::PyoOrThrow(result);
        });
  m.def(
      "te_gemm",
      [](const pybind11::handle& a_mat, const pybind11::handle& a_scale_inv,
         const transformer_engine::DType atype, const int a_offset,
         const pybind11::handle& b_mat, const pybind11::handle& b_scale_inv,
         const transformer_engine::DType btype, const int b_offset,
         const pybind11::handle& workspace, const bool use_bias,
         const pybind11::handle& bias, const bool use_gelu,
         const pybind11::handle& gelu_input, const bool transa,
         const bool transb, const bool grad, const bool accumulate,
         const bool use_split_accumulate, const transformer_engine::DType otype,
         const int64_t stream_id) {
      return TFE_Py_TeGemm_wrapper(a_mat, a_scale_inv, atype, a_offset, b_mat,
                                   b_scale_inv, btype, b_offset, workspace,
                                   use_bias, bias, use_gelu, gelu_input, transa,
                                   transb, grad, accumulate,
                                   use_split_accumulate, otype, stream_id);
      });
  m.def("layernorm_fwd",
        [](const pybind11::handle& input, const pybind11::handle& gamma,
           const pybind11::handle& beta, float eps, const int64_t stream_id) {
          using namespace transformer_engine;

          std::vector<size_t> shape_c = GetShape(input);
          CHECK_EQ(shape_c.size(), 2);
          std::vector<size_t> shape_g{shape_c[1]};
          std::vector<size_t> shape_m{shape_c[0]};

          auto itype = GetDataType(input);
          auto mtype = DType::kFloat32;
          void* ln_out_ptr = AllocateSpace(shape_c, itype);
          void* mu_ptr = AllocateSpace(shape_m, mtype);
          void* rsigma_ptr = AllocateSpace(shape_m, mtype);

          cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
          dispatch_layernorm(
              GetDevicePtr(input), shape_c, itype, GetDevicePtr(gamma), shape_g,
              itype, GetDevicePtr(beta), shape_g, itype, nullptr, {1}, mtype,
              eps, ln_out_ptr, shape_c, itype, mu_ptr, shape_m, mtype,
              rsigma_ptr, shape_m, mtype, nullptr, {1}, mtype, nullptr, {1},
              mtype, GetDeviceMultiProcessorCount(), stream);

          auto ln_out_eager = CreateTensor(ln_out_ptr, shape_c, itype);
          auto mu_eager = CreateTensor(mu_ptr, shape_m, mtype);
          auto rsigma_eager = CreateTensor(rsigma_ptr, shape_m, mtype);
          PyObject* result(PyList_New(3));
          PyList_SET_ITEM(result, 0, EagerTensorFromHandle(ln_out_eager));
          PyList_SET_ITEM(result, 1, EagerTensorFromHandle(mu_eager));
          PyList_SET_ITEM(result, 2, EagerTensorFromHandle(rsigma_eager));
          return tensorflow::PyoOrThrow(result);
        });
  m.def("layernorm_fwd_fp8",
        [](const pybind11::handle& input, const pybind11::handle& gamma,
           const pybind11::handle& beta, float eps,
           const pybind11::handle& scale, const transformer_engine::DType otype,
           const pybind11::handle& amax, const pybind11::handle& scale_inv,
           const int offset, const int64_t stream_id) {
          using namespace transformer_engine;

          std::vector<size_t> shape_c = GetShape(input);
          CHECK_EQ(shape_c.size(), 2);
          std::vector<size_t> shape_g{shape_c[1]};
          std::vector<size_t> shape_m{shape_c[0]};

          auto itype = GetDataType(input);
          auto mtype = DType::kFloat32;
          void* ln_out_ptr = AllocateSpace(shape_c, otype);
          void* mu_ptr = AllocateSpace(shape_m, mtype);
          void* rsigma_ptr = AllocateSpace(shape_m, mtype);

          cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
          dispatch_layernorm(
              GetDevicePtr(input), shape_c, itype, GetDevicePtr(gamma), shape_g,
              itype, GetDevicePtr(beta), shape_g, itype,
              GetDevicePtr(scale, offset), {1}, DType::kFloat32, eps,
              ln_out_ptr, shape_c, otype, mu_ptr, shape_m, mtype, rsigma_ptr,
              shape_m, mtype, GetDevicePtr(amax, offset), {1}, DType::kFloat32,
              GetDevicePtr(scale_inv, offset), {1}, DType::kFloat32,
              GetDeviceMultiProcessorCount(), stream);

          auto ln_out_eager = CreateTensor(ln_out_ptr, shape_c, otype);
          auto mu_eager = CreateTensor(mu_ptr, shape_m, mtype);
          auto rsigma_eager = CreateTensor(rsigma_ptr, shape_m, mtype);
          PyObject* result(PyList_New(3));
          PyList_SET_ITEM(result, 0, EagerTensorFromHandle(ln_out_eager));
          PyList_SET_ITEM(result, 1, EagerTensorFromHandle(mu_eager));
          PyList_SET_ITEM(result, 2, EagerTensorFromHandle(rsigma_eager));
          return tensorflow::PyoOrThrow(result);
        });
  m.def("layernorm_bwd", [](const pybind11::handle& dz,
                            const pybind11::handle& x,
                            const pybind11::handle& mu,
                            const pybind11::handle& rsigma,
                            const pybind11::handle& gamma,
                            const int64_t stream_id) {
    using namespace transformer_engine;

    std::vector<size_t> shape_x = GetShape(x);
    CHECK_EQ(shape_x.size(), 2);
    std::vector<size_t> shape_g{shape_x[1]};
    std::vector<size_t> shape_m{shape_x[0]};

    auto xtype = GetDataType(x);
    auto gtype = GetDataType(gamma);
    auto mtype = GetDataType(mu);
    void* dx_ptr = AllocateSpace(shape_x, xtype);
    void* dgamma_ptr = AllocateSpace(shape_g, gtype);
    void* dbeta_ptr = AllocateSpace(shape_g, gtype);

    auto x_tensor = MakeNVTETensor(GetDevicePtr(x), shape_x, xtype);
    auto gamma_tensor = MakeNVTETensor(GetDevicePtr(gamma), shape_g, gtype);
    auto dz_tensor = MakeNVTETensor(GetDevicePtr(dz), shape_x, xtype);
    auto mu_tensor = MakeNVTETensor(GetDevicePtr(mu), shape_m, mtype);
    auto rsigma_tensor = MakeNVTETensor(GetDevicePtr(rsigma), shape_m, mtype);
    auto dx_tensor = MakeNVTETensor(dx_ptr, shape_x, xtype);
    auto dgamma_tensor = MakeNVTETensor(dgamma_ptr, shape_g, gtype);
    auto dbeta_tensor = MakeNVTETensor(dbeta_ptr, shape_g, gtype);

    TensorWrapper workspace, barrier, dgamma_part, dbeta_part;

    // This call populates tensors with the required config.
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    nvte_layernorm_bwd(
        dz_tensor.data(), x_tensor.data(), mu_tensor.data(),
        rsigma_tensor.data(), gamma_tensor.data(), dx_tensor.data(),
        dgamma_tensor.data(), dbeta_tensor.data(), dgamma_part.data(),
        dbeta_part.data(), stream, GetDeviceMultiProcessorCount(),
        workspace.data(), barrier.data());

    // Alloc space for Tensors.
    auto w_s = workspace.shape();
    auto b_s = barrier.shape();
    auto dg_s = dgamma_part.shape();
    auto db_s = dbeta_part.shape();
    std::vector<size_t> w_shape_vec{w_s.data, w_s.data + w_s.ndim};
    std::vector<size_t> b_shape_vec{b_s.data, b_s.data + b_s.ndim};
    std::vector<size_t> dg_shape_vec{dg_s.data, dg_s.data + dg_s.ndim};
    std::vector<size_t> db_shape_vec{db_s.data, db_s.data + db_s.ndim};

    void* workspace_ptr = AllocateSpace(w_shape_vec, workspace.dtype());
    void* barrier_ptr =
        AllocateSpace(b_shape_vec, barrier.dtype(), stream, true);
    void* dgamma_part_ptr = AllocateSpace(dg_shape_vec, dgamma_part.dtype());
    void* dbeta_part_ptr = AllocateSpace(db_shape_vec, dbeta_part.dtype());

    workspace = MakeNVTETensor(workspace_ptr, w_shape_vec, workspace.dtype());
    barrier = MakeNVTETensor(barrier_ptr, b_shape_vec, barrier.dtype());
    dgamma_part =
        MakeNVTETensor(dgamma_part_ptr, dg_shape_vec, dgamma_part.dtype());
    dbeta_part =
        MakeNVTETensor(dbeta_part_ptr, db_shape_vec, dbeta_part.dtype());

    // Actual call to bwd kernel.
    nvte_layernorm_bwd(
        dz_tensor.data(), x_tensor.data(), mu_tensor.data(),
        rsigma_tensor.data(), gamma_tensor.data(), dx_tensor.data(),
        dgamma_tensor.data(), dbeta_tensor.data(), dgamma_part.data(),
        dbeta_part.data(), stream, GetDeviceMultiProcessorCount(),
        workspace.data(), barrier.data());

    auto dx_eager = CreateTensor(dx_ptr, shape_x, xtype);
    auto dgamma_eager = CreateTensor(dgamma_ptr, shape_g, gtype);
    auto dbeta_eager = CreateTensor(dbeta_ptr, shape_g, gtype);
    PyObject* result(PyList_New(3));
    PyList_SET_ITEM(result, 0, EagerTensorFromHandle(dx_eager));
    PyList_SET_ITEM(result, 1, EagerTensorFromHandle(dgamma_eager));
    PyList_SET_ITEM(result, 2, EagerTensorFromHandle(dbeta_eager));
    return tensorflow::PyoOrThrow(result);
  });
  m.def("te_gelu",
        [](const pybind11::handle& input, const pybind11::handle& scale,
           const transformer_engine::DType otype, const pybind11::handle& amax,
           const pybind11::handle& scale_inv, const int offset,
           const int64_t stream_id) {
          using namespace transformer_engine;

          std::vector<size_t> shape_c = GetShape(input);
          CHECK_EQ(shape_c.size(), 2);

          void* out_ptr = AllocateSpace(shape_c, otype);

          auto itype = GetDataType(input);

          void* scale_ptr = nullptr;
          void* amax_ptr = nullptr;
          void* scale_inv_ptr = nullptr;
          if (otype == DType::kFloat8E4M3 || otype == DType::kFloat8E5M2) {
            scale_ptr = GetDevicePtr(scale, offset);
            amax_ptr = GetDevicePtr(amax, offset);
            scale_inv_ptr = GetDevicePtr(scale_inv, offset);
          }

          cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
          dispatch_gelu(GetDevicePtr(input), shape_c, itype, scale_ptr, {1},
                        DType::kFloat32, out_ptr, shape_c, otype, amax_ptr, {1},
                        DType::kFloat32, scale_inv_ptr, {1}, DType::kFloat32,
                        stream);

          auto out_eager = CreateTensor(out_ptr, shape_c, otype);
          return tensorflow::PyoOrThrow(EagerTensorFromHandle(out_eager));
        });
  m.def("fp8_fused_cast_transpose_bgrad_dgelu",
        [](const pybind11::handle& grad_output,
           const pybind11::handle& gelu_input, const pybind11::handle& scale,
           const transformer_engine::DType otype, const pybind11::handle& amax,
           const pybind11::handle& scale_inv, const int offset,
           const int64_t stream_id) {
          using namespace transformer_engine;

          std::vector<size_t> shape_c = GetShape(grad_output);
          CHECK_EQ(shape_c.size(), 2);
          std::vector<size_t> shape_t{shape_c[1], shape_c[0]};
          std::vector<size_t> shape_b{shape_c[1]};

          auto itype = GetDataType(grad_output);
          void* grad_bias_ptr = AllocateSpace(shape_b, itype);
          void* dgelu_c_ptr = AllocateSpace(shape_c, otype);
          void* dgelu_t_ptr = AllocateSpace(shape_t, otype);

          cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
          dispatch_bgrad_dgelu_cast_transpose_fusion(
              GetDevicePtr(grad_output), shape_c, itype,
              GetDevicePtr(gelu_input), shape_c, itype,
              GetDevicePtr(scale, offset), {1}, DType::kFloat32, dgelu_c_ptr,
              shape_c, otype, dgelu_t_ptr, shape_t, otype,
              GetDevicePtr(amax, offset), {1}, DType::kFloat32, grad_bias_ptr,
              shape_b, itype, GetDevicePtr(scale_inv, offset), {1},
              DType::kFloat32, stream);

          auto grad_bias_eager = CreateTensor(grad_bias_ptr, shape_b, itype);
          auto dgelu_c_eager = CreateTensor(dgelu_c_ptr, shape_c, otype);
          auto dgelu_t_eager = CreateTensor(dgelu_t_ptr, shape_t, otype);
          PyObject* result(PyList_New(3));
          PyList_SET_ITEM(result, 0, EagerTensorFromHandle(grad_bias_eager));
          PyList_SET_ITEM(result, 1, EagerTensorFromHandle(dgelu_c_eager));
          PyList_SET_ITEM(result, 2, EagerTensorFromHandle(dgelu_t_eager));
          return tensorflow::PyoOrThrow(result);
        });
  m.def(
      "scaled_upper_triang_masked_softmax_forward",
      [](const pybind11::handle& input, const float scale_factor,
         const int64_t stream_id) {
        using namespace transformer_engine;

        std::vector<size_t> shape_in = GetShape(input);
        CHECK_EQ(shape_in.size(), 3);
        auto itype = GetDataType(input);
        CHECK(itype == DType::kFloat16 || itype == DType::kBFloat16);

        const size_t attn_batches = shape_in[0];
        const size_t seq_len = shape_in[1];
        CHECK_LE(seq_len, 2048);

        auto input_cu = MakeNVTETensor(GetDevicePtr(input), shape_in, itype);
        void* softmax_ptr = AllocateSpace(shape_in, itype);
        auto softmax_results_cu = MakeNVTETensor(softmax_ptr, shape_in, itype);

        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
        nvte_scaled_upper_triang_masked_softmax_forward(
            input_cu.data(), softmax_results_cu.data(), scale_factor, stream);

        auto softmax_results_eager = CreateTensor(softmax_ptr, shape_in, itype);
        return tensorflow::PyoOrThrow(
            EagerTensorFromHandle(softmax_results_eager));
      });
  m.def("scaled_upper_triang_masked_softmax_backward",
        [](const pybind11::handle& dy, const pybind11::handle& y,
           const float scale_factor, const int64_t stream_id) {
          using namespace transformer_engine;

          std::vector<size_t> shape_dy = GetShape(dy);
          std::vector<size_t> shape_y = GetShape(y);
          CHECK_EQ(shape_dy.size(), 3);
          CHECK_EQ(shape_y.size(), 3);
          auto dytype = GetDataType(dy);
          auto ytype = GetDataType(y);
          CHECK(dytype == DType::kFloat16 || dytype == DType::kBFloat16);
          CHECK(ytype == DType::kFloat16 || ytype == DType::kBFloat16);

          CHECK_EQ(shape_dy[1], shape_dy[2]);

          auto dy_cu = MakeNVTETensor(GetDevicePtr(dy), shape_dy, dytype);
          auto y_cu = MakeNVTETensor(GetDevicePtr(y), shape_y, ytype);
          void* dx_ptr = AllocateSpace(shape_dy, dytype);
          auto dx_cu = MakeNVTETensor(dx_ptr, shape_dy, dytype);

          cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
          nvte_scaled_upper_triang_masked_softmax_backward(
              dy_cu.data(), y_cu.data(), dx_cu.data(), scale_factor, stream);

          auto dx_eager = CreateTensor(dx_ptr, shape_dy, dytype);
          return tensorflow::PyoOrThrow(EagerTensorFromHandle(dx_eager));
        });
  m.def("scaled_masked_softmax_forward", [](const pybind11::handle& x,
                                            const pybind11::handle& mask,
                                            const float scale_factor,
                                            const int64_t stream_id) {
    using namespace transformer_engine;

    std::vector<size_t> shape_x = GetShape(x);
    std::vector<size_t> shape_m = GetShape(mask);
    CHECK_EQ(shape_x.size(), 4) << "expected 4D tensor";
    CHECK_EQ(shape_m.size(), 4) << "expected 4D tensor";
    auto xtype = GetDataType(x);
    auto mtype = GetDataType(mask);
    CHECK(xtype == DType::kFloat16 || xtype == DType::kBFloat16)
        << "Only fp16 and bf16 are supported";
    CHECK(mtype == DType::kByte) << "Only bool are supported for mask";

    const size_t batches = shape_x[0];
    const size_t pad_batches = shape_m[0];
    const size_t attn_heads = shape_x[1];
    const size_t query_seq_len = shape_x[2];
    const size_t key_seq_len = shape_x[3];
    CHECK_LE(key_seq_len, 4096);
    CHECK_GT(query_seq_len, 1);
    CHECK(pad_batches == 1 || pad_batches == batches);
    CHECK_EQ(shape_m[1], 1);
    CHECK(shape_m[2] == query_seq_len);
    CHECK(shape_m[3] == key_seq_len);

    void* softmax_ptr = AllocateSpace(shape_x, xtype);
    auto softmax_results_cu = MakeNVTETensor(softmax_ptr, shape_x, xtype);

    auto input_cu = MakeNVTETensor(GetDevicePtr(x), shape_x, xtype);
    auto mask_cu = MakeNVTETensor(GetDevicePtr(mask), shape_m, mtype);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    nvte_scaled_masked_softmax_forward(input_cu.data(), mask_cu.data(),
                                       softmax_results_cu.data(), scale_factor,
                                       stream);

    auto softmax_results_eager = CreateTensor(softmax_ptr, shape_x, xtype);
    return tensorflow::PyoOrThrow(EagerTensorFromHandle(softmax_results_eager));
  });
  m.def("scaled_masked_softmax_backward",
        [](const pybind11::handle& dy, const pybind11::handle& y,
           const float scale_factor, const int64_t stream_id) {
          using namespace transformer_engine;

          std::vector<size_t> shape_dy = GetShape(dy);
          std::vector<size_t> shape_y = GetShape(y);
          CHECK_EQ(shape_dy.size(), 4) << "expected 4D tensor";
          CHECK_EQ(shape_y.size(), 4) << "expected 4D tensor";
          auto dytype = GetDataType(dy);
          auto ytype = GetDataType(y);
          CHECK(dytype == DType::kFloat16 || dytype == DType::kBFloat16)
              << "Only fp16 and bf16 are supported";
          CHECK(ytype == DType::kFloat16 || ytype == DType::kBFloat16)
              << "Only fp16 and bf16 are supported";

          auto dy_cu = MakeNVTETensor(GetDevicePtr(dy), shape_dy, dytype);
          auto y_cu = MakeNVTETensor(GetDevicePtr(y), shape_y, ytype);
          void* dx_ptr = AllocateSpace(shape_dy, dytype);
          auto dx_cu = MakeNVTETensor(dx_ptr, shape_dy, dytype);

          cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
          nvte_scaled_masked_softmax_backward(
              dy_cu.data(), y_cu.data(), dx_cu.data(), scale_factor, stream);

          auto dx_eager = CreateTensor(dx_ptr, shape_dy, dytype);
          return tensorflow::PyoOrThrow(EagerTensorFromHandle(dx_eager));
        });
  m.def("scaled_softmax_forward", [](const pybind11::handle& x,
                                     const float scale_factor,
                                     const int64_t stream_id) {
    using namespace transformer_engine;

    std::vector<size_t> shape_x = GetShape(x);
    CHECK_EQ(shape_x.size(), 4) << "expected 4D tensor";
    auto xtype = GetDataType(x);
    CHECK(xtype == DType::kFloat16 || xtype == DType::kBFloat16)
        << "Only fp16 and bf16 are supported";

    const size_t batches = shape_x[0];
    const size_t attn_heads = shape_x[1];
    const size_t query_seq_len = shape_x[2];
    const size_t key_seq_len = shape_x[3];

    CHECK_LE(key_seq_len, 4096);
    CHECK_GT(query_seq_len, 1);

    void* softmax_ptr = AllocateSpace(shape_x, xtype);
    auto softmax_results_cu = MakeNVTETensor(softmax_ptr, shape_x, xtype);

    auto input_cu = MakeNVTETensor(GetDevicePtr(x), shape_x, xtype);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    nvte_scaled_softmax_forward(input_cu.data(), softmax_results_cu.data(),
                                scale_factor, stream);

    auto softmax_results_eager = CreateTensor(softmax_ptr, shape_x, xtype);
    return tensorflow::PyoOrThrow(EagerTensorFromHandle(softmax_results_eager));
  });
  m.def("scaled_softmax_backward",
        [](const pybind11::handle& dy, const pybind11::handle& y,
           const float scale_factor, const int64_t stream_id) {
          using namespace transformer_engine;

          std::vector<size_t> shape_dy = GetShape(dy);
          std::vector<size_t> shape_y = GetShape(y);

          CHECK_EQ(shape_dy.size(), 4) << "expected 4D tensor";
          CHECK_EQ(shape_y.size(), 4) << "expected 4D tensor";

          auto dytype = GetDataType(dy);
          auto ytype = GetDataType(y);
          CHECK(dytype == DType::kFloat16 || dytype == DType::kBFloat16)
              << "Only fp16 and bf16 are supported";
          CHECK(ytype == DType::kFloat16 || ytype == DType::kBFloat16)
              << "Only fp16 and bf16 are supported";

          auto dy_cu = MakeNVTETensor(GetDevicePtr(dy), shape_dy, dytype);
          auto y_cu = MakeNVTETensor(GetDevicePtr(y), shape_y, ytype);
          void* dx_ptr = AllocateSpace(shape_dy, dytype);
          auto dx_cu = MakeNVTETensor(dx_ptr, shape_dy, dytype);

          cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
          nvte_scaled_softmax_backward(dy_cu.data(), y_cu.data(), dx_cu.data(),
                                       scale_factor, stream);

          auto dx_eager = CreateTensor(dx_ptr, shape_dy, dytype);
          return tensorflow::PyoOrThrow(EagerTensorFromHandle(dx_eager));
        });
}
