#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/musa/MUSACachingAllocator.h>
#include <mudnncxx/mudnn.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "../common.h"
#include "../util/logging.h"
#include "../util/mtfp8_utils.muh"
#include "../util/mudnn.h"

namespace transformer_engine {

namespace {

using mtfp8::next_power_of_2;
using transformer_engine::musa::CreateMUTensor;
using transformer_engine::musa::Flat2DimShape;
using transformer_engine::musa::MUTensor;
using transformer_engine::musa::SetMUTensorDType;
using transformer_engine::musa::ToTorchDtype;

#define CHECK_MUDNN_STATUS_CPP(result, message)                       \
  TORCH_CHECK((result) == ::musa::dnn::Status::SUCCESS, __FUNCTION__, \
              " muDNN failed in: ", (message))

inline ::musa::dnn::MemoryHandler InternalMemAlloc(size_t size) {
  musaStreamCaptureStatus status = musaStreamCaptureStatusNone;
  musaStream_t stream = at::musa::getCurrentMUSAStream().stream();
  if (musaStreamIsCapturing(stream, &status) == musaSuccess &&
      status == musaStreamCaptureStatusActive) {
    at::Tensor workspace = at::empty({static_cast<int64_t>(size)},
                                     at::TensorOptions().dtype(at::kByte).device(at::kMUSA));
    workspace.zero_();
    static std::vector<at::Tensor> captured_workspaces;
    captured_workspaces.push_back(workspace);
    return ::musa::dnn::MemoryHandler(workspace.data_ptr(), [](void* pointer) { (void)pointer; });
  }

  void* pointer = c10::musa::MUSACachingAllocator::raw_alloc(size);
  return ::musa::dnn::MemoryHandler(
      pointer, [](void* value) { c10::musa::MUSACachingAllocator::raw_delete(value); });
}

inline ::musa::dnn::Handle& GetMudnnHandle() {
  int device = 0;
  NVTE_CHECK_CUDA(musaGetDevice(&device));
  static std::unordered_map<int, std::unique_ptr<::musa::dnn::Handle>> handles;
  auto& handle = handles[device];
  if (!handle) handle = std::make_unique<::musa::dnn::Handle>(device);
  return *handle;
}

using MudnnComputeMode = ::musa::dnn::MatMul::ComputeMode;

inline MudnnComputeMode GetComputeModeFromCtx(at::ScalarType dtype) {
  return static_cast<MudnnComputeMode>(at::musa::GetComputeModeFromCtx(dtype));
}

inline MudnnComputeMode toMudnnComputeMode(at::musa::ComputeMode mode) {
  return static_cast<MudnnComputeMode>(mode);
}

const auto empty_te_tensor = Tensor();
inline MUTensor CreateEmptyMUTensor() {
  MUTensor tensor;
  tensor.SetType(::musa::dnn::Tensor::Type::FLOAT);
  return tensor;
}
const auto empty_mu_tensor = CreateEmptyMUTensor();

constexpr int num_streams = 4;

std::once_flag init_flag;
musaStream_t compute_streams[num_streams];
musaEvent_t cublas_event[num_streams];
bool multistream_to_use;

void init_streams_and_events() {
  for (int i = 0; i < num_streams; i++) {
    NVTE_CHECK_CUDA(musaStreamCreateWithPriority(&compute_streams[i], musaStreamNonBlocking, -1));
    NVTE_CHECK_CUDA(musaEventCreate(&cublas_event[i]));
  }

  multistream_to_use = false;
  const char* multi_stream_env = std::getenv("TE_MULTI_STREAM_GROUPGEMM");
  if (multi_stream_env == nullptr) {
    multi_stream_env = std::getenv("MULTI_STREAM_GROUPGEMM");
  }
  if (multi_stream_env != nullptr && std::string(multi_stream_env) == "1") {
    multistream_to_use = true;
  }
}

const SimpleTensor* get_data(const Tensor* te_tensor, bool trans) {
  if (trans && te_tensor->has_columnwise_data()) {
    return &(te_tensor->columnwise_data);
  }
  return &(te_tensor->data);
}

struct GEMM_INFO {
  const SimpleTensor* data_a = nullptr;
  const SimpleTensor* sinv_a = nullptr;
  const SimpleTensor* data_b = nullptr;
  const SimpleTensor* sinv_b = nullptr;
  bool is_per_tensor = true;
};

GEMM_INFO get_gemm_info(const Tensor* a, bool trans_a, const Tensor* b, bool trans_b) {
  NVTE_CHECK(
      a->scaling_mode == b->scaling_mode ||
          (a->scaling_mode == NVTE_BLOCK_SCALING_1D && b->scaling_mode == NVTE_BLOCK_SCALING_2D) ||
          (a->scaling_mode == NVTE_BLOCK_SCALING_2D && b->scaling_mode == NVTE_BLOCK_SCALING_1D),
      "Inputs A and B to GEMM need to have compatible scaling modes! ",
      "A scaling mode: ", to_string(a->scaling_mode),
      ", B scaling mode: ", to_string(b->scaling_mode));
  NVTE_CHECK(a->has_data() || a->has_columnwise_data(), "Input A does not hold any data!");
  NVTE_CHECK(b->has_data() || b->has_columnwise_data(), "Input B does not hold any data!");

  GEMM_INFO info;
  info.is_per_tensor = is_tensor_scaling(a->scaling_mode);
  if (info.is_per_tensor) {
    info.data_a = &(a->data);
    info.sinv_a = &(a->scale_inv);
    info.data_b = &(b->data);
    info.sinv_b = &(b->scale_inv);
    return info;
  }

  const bool weight_is_nn_block = (not(a->data).shape.empty()) &&
                                  (product(a->data.shape, 0, a->data.shape.size() - 1) !=
                                   product(a->scale_inv.shape, 0, a->scale_inv.shape.size() - 1));

  if (weight_is_nn_block || trans_a) {
    info.data_a = &(a->data);
    info.sinv_a = &(a->scale_inv);
  } else {
    info.data_a = &(a->columnwise_data);
    info.sinv_a = &(a->columnwise_scale_inv);
  }

  if (trans_b) {
    info.data_b = &(b->columnwise_data);
    info.sinv_b = &(b->columnwise_scale_inv);
  } else {
    info.data_b = &(b->data);
    info.sinv_b = &(b->scale_inv);
  }

  return info;
}

}  // anonymous namespace

void non_fp8_gemm(const Tensor* inputA, bool transa, const Tensor* inputB, bool transb,
                  Tensor* outputD, const Tensor* biasTensor, bool accumulate, int math_sm_count,
                  musaStream_t stream) {
  musaStreamCaptureStatus capture_status = musaStreamCaptureStatusNone;
  musaStreamIsCapturing(stream, &capture_status);
  if (capture_status == musaStreamCaptureStatusActive && !accumulate) {
    auto a_shape = Flat2DimShape(inputA);
    auto b_shape = Flat2DimShape(inputB);
    auto d_shape = Flat2DimShape(outputD);
    auto weight =
        at::from_blob(const_cast<void*>(get_data(inputA, transa)->dptr),
                      {static_cast<int64_t>(a_shape[0]), static_cast<int64_t>(a_shape[1])},
                      at::TensorOptions().dtype(ToTorchDtype(inputA->dtype())).device(at::kMUSA));
    auto activation =
        at::from_blob(const_cast<void*>(get_data(inputB, transb)->dptr),
                      {static_cast<int64_t>(b_shape[0]), static_cast<int64_t>(b_shape[1])},
                      at::TensorOptions().dtype(ToTorchDtype(inputB->dtype())).device(at::kMUSA));
    auto output = at::from_blob(
        outputD->data.dptr, {static_cast<int64_t>(d_shape[0]), static_cast<int64_t>(d_shape[1])},
        at::TensorOptions().dtype(ToTorchDtype(outputD->dtype())).device(at::kMUSA));
    auto lhs = transb ? activation.t() : activation;
    auto rhs = transa ? weight.t() : weight;
    if (biasTensor->has_data()) {
      auto bias = at::from_blob(
          biasTensor->data.dptr, {static_cast<int64_t>(biasTensor->data.shape[0])},
          at::TensorOptions().dtype(ToTorchDtype(biasTensor->dtype())).device(at::kMUSA));
      at::addmm_out(output, bias, lhs, rhs);
    } else {
      at::mm_out(output, lhs, rhs);
    }
    return;
  }

  auto& h = GetMudnnHandle();
  h.SetStream(stream);

  const bool has_bias = biasTensor->has_data();
  auto mu_l = CreateMUTensor(*get_data(inputB, transb), Flat2DimShape(inputB));
  auto mu_r = CreateMUTensor(*get_data(inputA, transa), Flat2DimShape(inputA));
  auto mu_b = has_bias ? CreateMUTensor(biasTensor->data) : empty_mu_tensor;
  auto mu_o = CreateMUTensor(outputD->data, Flat2DimShape(outputD));

  ::musa::dnn::MatMul op;
  CHECK_MUDNN_STATUS_CPP(op.SetTranspose(transb, transa), "SetTranspose");
  CHECK_MUDNN_STATUS_CPP(op.SetComputeMode(GetComputeModeFromCtx(ToTorchDtype(inputB->dtype()))),
                         "SetComputeMode");
  CHECK_MUDNN_STATUS_CPP(op.SetAlpha(1.0), "SetAlpha");
  CHECK_MUDNN_STATUS_CPP(op.SetBeta(accumulate ? 1.0 : 0.0), "SetBeta");
  CHECK_MUDNN_STATUS_CPP(op.SetGamma(has_bias ? 1.0 : 0.0), "SetGamma");

  CHECK_MUDNN_STATUS_CPP(op.RunWithBiasAdd(h, mu_o, mu_l, mu_r, mu_o, mu_b, InternalMemAlloc),
                         "RunWithBiasAdd");
}

void fp8_gemm(const Tensor* inputA, bool transa, const Tensor* inputB, bool transb, Tensor* outputD,
              const Tensor* biasTensor, bool accumulate, int math_sm_count, musaStream_t stream) {
  auto& h = GetMudnnHandle();
  h.SetStream(stream);

  const bool has_bias = biasTensor->has_data();
  const bool has_bias_scale = (biasTensor->scale_inv.dptr != nullptr);

  const bool has_output_scale = (outputD->scale.dptr != nullptr);
  const bool has_output_amax = (outputD->amax.dptr != nullptr);

  const auto info = get_gemm_info(inputA, transa, inputB, transb);
  const auto& data_b = *(info.data_b);
  const auto& sinv_b = *(info.sinv_b);
  const auto& data_a = *(info.data_a);
  const auto& sinv_a = *(info.sinv_a);

  auto mu_l = CreateMUTensor(data_b, Flat2DimShape(inputB));
  auto mu_r = CreateMUTensor(data_a, Flat2DimShape(inputA));
  auto mu_b = has_bias ? CreateMUTensor(biasTensor->data) : empty_mu_tensor;
  auto mu_o = CreateMUTensor(outputD->data, Flat2DimShape(outputD));
  if (!has_bias) {
    SetMUTensorDType(outputD->dtype(), mu_b);
  }

  auto mu_scale_l = CreateMUTensor(sinv_b);
  auto mu_scale_r = CreateMUTensor(sinv_a);
  auto mu_scale_b = has_bias_scale ? CreateMUTensor(biasTensor->scale_inv) : empty_mu_tensor;
  auto mu_scale_o = has_output_scale ? CreateMUTensor(outputD->scale) : empty_mu_tensor;
  auto mu_amax_o = has_output_amax ? CreateMUTensor(outputD->amax) : empty_mu_tensor;

  ::musa::dnn::BatchMatMul op;
  CHECK_MUDNN_STATUS_CPP(op.SetTranspose(transb, transa), "SetTranspose");
  CHECK_MUDNN_STATUS_CPP(op.SetComputeMode(GetComputeModeFromCtx(ToTorchDtype(inputB->dtype()))),
                         "SetComputeMode");
  CHECK_MUDNN_STATUS_CPP(op.SetAlpha(1.0), "SetAlpha");
  CHECK_MUDNN_STATUS_CPP(op.SetBeta(accumulate ? 1.0 : 0.0), "SetBeta");
  CHECK_MUDNN_STATUS_CPP(op.SetGamma(has_bias ? 1.0 : 0.0), "SetGamma");
  if (math_sm_count != 0) {
    CHECK_MUDNN_STATUS_CPP(op.SetMpCountTarget(math_sm_count), "SetMpCountTarget");
  }

  ::musa::dnn::MatMulLtParam param;
  if (info.is_per_tensor) {
    CHECK_MUDNN_STATUS_CPP(param.SetScale(mu_scale_l, mu_scale_r, mu_scale_b, mu_scale_o),
                           "SetScale");
  } else {
    NVTE_CHECK(inputB->scale_inv.shape.size() == 2);
    const auto tile_size =
        static_cast<int>(next_power_of_2(inputB->flat_last_dim() / inputB->scale_inv.shape[1]));
    CHECK_MUDNN_STATUS_CPP(
        param.SetScale(mu_scale_l, mu_scale_r, mu_scale_b, mu_scale_o, tile_size), "SetScale");
  }
  CHECK_MUDNN_STATUS_CPP(param.SetAmaxD(mu_amax_o), "SetAmax");

  CHECK_MUDNN_STATUS_CPP(op.RunLt(h, mu_o, mu_l, mu_r, mu_o, mu_b, param, InternalMemAlloc),
                         "RunLt");
}

void no_fp8_grad_bias(const Tensor* gradO, bool trans, const Tensor* gradB, musaStream_t stream) {
  using REDUCE_MODE = ::musa::dnn::Reduce::Mode;
  const int reduce_dim = trans ? 0 : 1;

  auto& h = GetMudnnHandle();
  h.SetStream(stream);

  auto mu_i = CreateMUTensor(gradO->data, Flat2DimShape(gradO));
  auto mu_o = CreateMUTensor(gradB->data, Flat2DimShape(gradB));

  ::musa::dnn::Reduce rdc;
  CHECK_MUDNN_STATUS_CPP(rdc.SetMode(REDUCE_MODE::ADD), "SetMode");
  CHECK_MUDNN_STATUS_CPP(rdc.SetDim({reduce_dim}), "SetDim");
  CHECK_MUDNN_STATUS_CPP(rdc.Run(h, mu_o, mu_i, InternalMemAlloc), "Run");
}

}  // namespace transformer_engine

// D = B @ A.T
void mudnn_gemm(const NVTETensor A, const NVTETensor B, NVTETensor D, const NVTETensor bias,
                NVTETensor pre_gelu_out, bool transa, bool transb, bool grad, NVTETensor workspace,
                bool accumulate, bool use_split_accumulator, int math_sm_count,
                musaStream_t stream) {
  using namespace transformer_engine;

  const Tensor* inputA = convertNVTETensorCheck(A);
  const Tensor* inputB = convertNVTETensorCheck(B);
  Tensor* outputD = convertNVTETensorCheck(D);
  const Tensor* biasTensor = convertNVTETensor(bias);
  Tensor* geluOut = convertNVTETensor(pre_gelu_out);
  Tensor* wspace = convertNVTETensor(workspace);

  NVTE_CHECK(outputD->has_data());
  NVTE_CHECK(!geluOut->has_data(), "Gelu epilogue is not supported!");

  const auto A_type = inputA->dtype();
  const auto is_fp8_A = is_fp8_dtype(A_type);

  const auto B_type = inputB->dtype();
  const auto is_fp8_B = is_fp8_dtype(B_type);

  NVTE_CHECK(is_fp8_A == is_fp8_B, "Inputs to muDNN GEMM must all be non-fp8 or fp8 dtypes!");
  if (!is_fp8_A) {
    NVTE_CHECK(A_type == B_type, "Both inputs to muDNN non-FP8 GEMM must have the same dtype!");
  }
  if (biasTensor->has_data() && !grad) {
    NVTE_CHECK(
        biasTensor->data.shape.size() == 1 && biasTensor->data.shape[0] == outputD->flat_last_dim(),
        "Mismatch bias shape, expect ", outputD->flat_last_dim(), ", but got ",
        biasTensor->data.shape[0]);
  }

  const auto* fwd_bias = grad ? &transformer_engine::empty_te_tensor : biasTensor;
  if (is_fp8_A) {
    fp8_gemm(inputA, transa, inputB, transb, outputD, fwd_bias, accumulate, math_sm_count, stream);
  } else {
    non_fp8_gemm(inputA, transa, inputB, transb, outputD, fwd_bias, accumulate, math_sm_count,
                 stream);
  }

  if (!grad || !(biasTensor->has_data())) {
    return;
  }

  if (!is_fp8_A) {
    no_fp8_grad_bias(inputB, transb, biasTensor, stream);
  }
}

void nvte_cublas_gemm(const NVTETensor A, const NVTETensor B, NVTETensor D, const NVTETensor bias,
                      NVTETensor pre_gelu_out, bool transa, bool transb, bool grad,
                      NVTETensor workspace, bool accumulate, bool use_split_accumulator,
                      int math_sm_count, musaStream_t stream) {
  NVTE_API_CALL(nvte_cublas_gemm);
  mudnn_gemm(A, B, D, bias, pre_gelu_out, transa, transb, grad, workspace, accumulate,
             use_split_accumulator, math_sm_count, stream);
}

void nvte_cublas_atomic_gemm(const NVTETensor A, const NVTETensor B, NVTETensor D,
                             const NVTETensor bias, NVTETensor pre_gelu_out, bool transa,
                             bool transb, bool grad, NVTETensor workspace, bool accumulate,
                             bool use_split_accumulator, int math_sm_count, int m_split,
                             int n_split, bool gemm_producer, const NVTETensor counter,
                             musaStream_t stream) {
  NVTE_API_CALL(nvte_cublas_atomic_gemm);
  NVTE_CHECK(false, "atomic_gemm is not supported.");
}

void nvte_multi_stream_cublas_gemm(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                                   const NVTETensor* bias, NVTETensor* pre_gelu_out,
                                   const int num_gemms, bool transa, bool transb, bool grad,
                                   NVTETensor* workspace, bool accumulate,
                                   bool use_split_accumulator, int math_sm_count,
                                   musaStream_t stream) {
  NVTE_API_CALL(nvte_multi_stream_cublas_gemm);
  using namespace transformer_engine;

  std::call_once(init_flag, init_streams_and_events);

  if (!multistream_to_use) {
    // Keep graph capture and eager execution on the caller's current stream.
    // Using the legacy stream here lets grouped GEMMs escape a non-default
    // capture stream and allows their inputs to be reused before completion.
    for (int i = 0; i < num_gemms; i++) {
      mudnn_gemm(A[i], B[i], D[i], bias[i], pre_gelu_out[i], transa, transb, grad,
                 workspace[i % num_streams], accumulate, use_split_accumulator, math_sm_count,
                 stream);
    }
    return;
  }

  int num_stream_used = std::min(num_streams, num_gemms);
  // wait for current stream to finish
  NVTE_CHECK_CUDA(musaEventRecord(cublas_event[0], stream));
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(musaStreamWaitEvent(compute_streams[s], cublas_event[0]));
  }

  for (int i = 0; i < num_gemms; i++) {
    mudnn_gemm(A[i], B[i], D[i], bias[i], pre_gelu_out[i], transa, transb, grad,
               workspace[i % num_streams], accumulate, use_split_accumulator, math_sm_count,
               compute_streams[i % num_streams]);
  }

  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(musaEventRecord(cublas_event[s], compute_streams[s]));
  }
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(musaStreamWaitEvent(stream, cublas_event[s]));
  }
}

void nvte_grouped_mudnn_gemm(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                             const NVTETensor* bias, NVTETensor* pre_gelu_out, const int num_gemms,
                             bool transa, bool transb, bool grad, NVTETensor* workspace,
                             bool accumulate, bool use_split_accumulator, int math_sm_count,
                             musaStream_t stream) {
  NVTE_API_CALL(nvte_grouped_mudnn_gemm);
  using namespace transformer_engine;

  NVTE_CHECK(num_gemms >= 0, "The number of grouped GEMMs must be non-negative.");
  if (num_gemms == 0) {
    return;
  }
  NVTE_CHECK(A != nullptr, "Grouped GEMM A tensor array is null.");
  NVTE_CHECK(B != nullptr, "Grouped GEMM B tensor array is null.");
  NVTE_CHECK(D != nullptr, "Grouped GEMM output tensor array is null.");
  NVTE_CHECK(bias != nullptr, "Grouped GEMM bias tensor array is null.");
  NVTE_CHECK(pre_gelu_out != nullptr, "Grouped GEMM pre-GELU tensor array is null.");
  NVTE_CHECK(B[0] != nullptr, "Grouped GEMM received an invalid first B tensor handle.");

  std::vector<MUTensor> inputL(num_gemms);
  std::vector<MUTensor> inputR(num_gemms);
  std::vector<MUTensor> inputBias(num_gemms);
  std::vector<MUTensor> inputOut(num_gemms);
  std::vector<::musa::dnn::MatMulLtParam> lt_parap_vec(num_gemms);
  const auto B_type = convertNVTETensorCheck(B[0])->dtype();
  bool with_bias = false;
  for (int i = 0; i < num_gemms; i++) {
    // trans NVTETensor to Tensor
    const auto* inputA = convertNVTETensorCheck(A[i]);
    const auto* inputB = convertNVTETensorCheck(B[i]);
    auto* outputD = convertNVTETensorCheck(D[i]);
    const auto* biasTensor = convertNVTETensor(bias[i]);
    auto* geluOut = convertNVTETensor(pre_gelu_out[i]);

    NVTE_CHECK(outputD->has_data());
    NVTE_CHECK(!geluOut->has_data(), "Gelu epilogue is not supported!");

    const auto A_type = inputA->dtype();
    const auto is_fp8_A = is_fp8_dtype(A_type);

    const auto B_type = inputB->dtype();
    const auto is_fp8_B = is_fp8_dtype(B_type);

    NVTE_CHECK(is_fp8_A == is_fp8_B, "Inputs to muDNN GEMM must all be non-fp8 or fp8 dtypes!");

    if (biasTensor->has_data() && !grad) {
      NVTE_CHECK(biasTensor->data.shape.size() == 1 &&
                     biasTensor->data.shape[0] == outputD->flat_last_dim(),
                 "Mismatch bias shape, expect ", outputD->flat_last_dim(), ", but got ",
                 biasTensor->data.shape[0]);
    }
    if (is_fp8_A) {
      const bool has_bias_scale = (biasTensor->scale_inv.dptr != nullptr);

      const bool has_output_scale = (outputD->scale.dptr != nullptr);
      const bool has_output_amax = (outputD->amax.dptr != nullptr);

      const auto info = get_gemm_info(inputA, transa, inputB, transb);
      const auto& data_b = *(info.data_b);
      const auto& sinv_b = *(info.sinv_b);
      const auto& data_a = *(info.data_a);
      const auto& sinv_a = *(info.sinv_a);
      //set scales which will be used in mudnn kernel.
      auto mu_scale_l = CreateMUTensor(sinv_b);
      auto mu_scale_r = CreateMUTensor(sinv_a);
      auto mu_scale_b = has_bias_scale ? CreateMUTensor(biasTensor->scale_inv) : empty_mu_tensor;
      auto mu_scale_o = has_output_scale ? CreateMUTensor(outputD->scale) : empty_mu_tensor;
      auto mu_amax_o = has_output_amax ? CreateMUTensor(outputD->amax) : empty_mu_tensor;

      if (info.is_per_tensor) {
        CHECK_MUDNN_STATUS_CPP(
            lt_parap_vec[i].SetScale(mu_scale_l, mu_scale_r, mu_scale_b, mu_scale_o), "SetScale");
      } else {
        NVTE_CHECK(inputB->scale_inv.shape.size() == 2);
        const auto tile_size =
            static_cast<int>(next_power_of_2(inputB->flat_last_dim() / inputB->scale_inv.shape[1]));
        CHECK_MUDNN_STATUS_CPP(
            lt_parap_vec[i].SetScale(mu_scale_l, mu_scale_r, mu_scale_b, mu_scale_o, tile_size),
            "SetScale");
      }
      CHECK_MUDNN_STATUS_CPP(lt_parap_vec[i].SetAmaxD(mu_amax_o), "SetAmax");
      inputR[i] = CreateMUTensor(data_a, Flat2DimShape(inputA));
      inputL[i] = CreateMUTensor(data_b, Flat2DimShape(inputB));
    } else {
      NVTE_CHECK(A_type == B_type, "Both inputs to muDNN non-FP8 GEMM must have the same dtype!");
      inputR[i] = CreateMUTensor(*get_data(inputA, transa), Flat2DimShape(inputA));
      inputL[i] = CreateMUTensor(*get_data(inputB, transb), Flat2DimShape(inputB));
    }

    // trans NVTETenso to MUTensor
    const bool has_bias = biasTensor->has_data();
    auto mu_b = has_bias ? CreateMUTensor(biasTensor->data) : empty_mu_tensor;
    auto mu_o = CreateMUTensor(outputD->data, Flat2DimShape(outputD));
    if (!has_bias) {
      SetMUTensorDType(outputD->dtype(), mu_b);
    }
    with_bias = with_bias || has_bias;
    inputBias[i] = mu_b;
    inputOut[i] = mu_o;
  }
  const auto& bias_ptr = with_bias ? inputBias.data() : nullptr;
  bool split_k = true;
  int current_device = 0;
  musaGetDevice(&current_device);

  // GRAPH-SAFE FALLBACK: mudnn's GroupedMatMul kernels (both the split-k
  // persistent variant and the deterministic variant) are not MUSA-graph
  // replayable -- the persistent kernel deadlocks replaying against stale
  // workspace flag state, and the deterministic variant computes wrong
  // results on replay (verified by miniexp/graph_gemm_hang_probe.py).
  // muBLAS via at::mm_out IS verified graph-replay-safe for all four layout
  // combinations (miniexp/mm_layout_safety.py). So when the stream is
  // capturing a graph, unroll the group into per-expert muBLAS GEMMs recorded
  // as ordinary aten kernels. The eager path keeps the fused grouped kernel.
  //
  // Semantics identical to non_fp8_gemm's fallback: D = (transb ? B^T : B) @
  // (transa ? A^T : A), applied per expert.
  musaStreamCaptureStatus cap_status = musaStreamCaptureStatusNone;
  musaStreamIsCapturing(stream, &cap_status);
  if (cap_status == musaStreamCaptureStatusActive && !with_bias && !accumulate) {
    for (int i = 0; i < num_gemms; ++i) {
      const auto* tA = convertNVTETensorCheck(A[i]);
      const auto* tB = convertNVTETensorCheck(B[i]);
      auto* dst = convertNVTETensorCheck(D[i]);
      auto sA = Flat2DimShape(tA);
      auto sB = Flat2DimShape(tB);
      auto sD = Flat2DimShape(dst);
      auto a_t =
          at::from_blob(const_cast<void*>(reinterpret_cast<const void*>(tA->data.dptr)),
                        {static_cast<int64_t>(sA[0]), static_cast<int64_t>(sA[1])},
                        at::TensorOptions().dtype(ToTorchDtype(tA->dtype())).device(at::kMUSA));
      auto b_t =
          at::from_blob(const_cast<void*>(reinterpret_cast<const void*>(tB->data.dptr)),
                        {static_cast<int64_t>(sB[0]), static_cast<int64_t>(sB[1])},
                        at::TensorOptions().dtype(ToTorchDtype(tB->dtype())).device(at::kMUSA));
      auto d_t =
          at::from_blob(dst->data.dptr, {static_cast<int64_t>(sD[0]), static_cast<int64_t>(sD[1])},
                        at::TensorOptions().dtype(ToTorchDtype(dst->dtype())).device(at::kMUSA));
      auto lhs = transb ? b_t.t() : b_t;  // (transb ? B^T : B)
      auto rhs = transa ? a_t.t() : a_t;  // (transa ? A^T : A)
      at::mm_out(d_t, lhs, rhs);
    }
    return;
  }

  static std::unordered_map<int, std::unique_ptr<::musa::dnn::Handle>> handle_pool;
  if (handle_pool.find(current_device) == handle_pool.end()) {
    handle_pool[current_device] = std::make_unique<::musa::dnn::Handle>(current_device);
  }
  auto& h = *handle_pool[current_device];
  h.SetStream(stream);
  ::musa::dnn::GroupedMatMul op;
  CHECK_MUDNN_STATUS_CPP(op.SetTranspose(transb, transa), "SetTranspose");
  // GRAPH-SAFE ALGORITHM: the non-deterministic path picks a split-k
  // PERSISTENT kernel (musa_asm_..._persis_stage2) whose cross-block
  // reduction relies on workspace-resident counter/barrier state that is set
  // up at launch time by the host. Under MUSA graph capture (record-only),
  // that per-launch setup is not re-done at replay and the kernel deadlocks
  // spinning on its flags ("WAIT FC ZERO" warp status in device dumps).
  // Force the deterministic (non-split-k) algorithm whenever the stream is
  // capturing so the captured kernel is self-contained and replayable. The
  // eager path keeps split_k.
  {
    musaStreamCaptureStatus cap_status = musaStreamCaptureStatusNone;
    musaStreamIsCapturing(stream, &cap_status);
    if (cap_status == musaStreamCaptureStatusActive) {
      split_k = false;
    }
  }
  CHECK_MUDNN_STATUS_CPP(op.SetDeterministic(!split_k), "SetDeterministic");
  CHECK_MUDNN_STATUS_CPP(op.SetComputeMode(toMudnnComputeMode(at::musa::GetComputeModeFromCtx(
                             transformer_engine::musa::ToTorchDtype(B_type)))),
                         "SetComputeMode");
  CHECK_MUDNN_STATUS_CPP(op.SetBeta(accumulate ? 1.0 : 0.0), "SetBeta");

  CHECK_MUDNN_STATUS_CPP(op.RunLt(h, inputOut.data(), inputL.data(), inputR.data(), inputOut.data(),
                                  bias_ptr, lt_parap_vec.data(), num_gemms, InternalMemAlloc),
                         "RunLt");
}
