#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include "../common.h"
#include "../util/logging.h"
#include "../util/mudnn.h"
#include "../util/mtfp8_utils.muh"

namespace transformer_engine {

namespace {

using at::musa::InternalMemAlloc;
using at::musa::GetComputeModeFromCtx;
using transformer_engine::musa::Flat2DimShape;
using transformer_engine::musa::CreateMUTensor;
using transformer_engine::musa::ToTorchDtype;
using transformer_engine::musa::SetMUTensorDType;
using mtfp8::next_power_of_2;

const auto empty_te_tensor = Tensor();
const auto empty_mu_tensor = at::musa::CreateEmptyMUTensor();

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
  if (std::getenv("MULTI_STREAM_GROUPGEMM") != nullptr
      && std::string(std::getenv("MULTI_STREAM_GROUPGEMM")) == "1") {
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

GEMM_INFO get_gemm_info(
    const Tensor* a,
    bool trans_a,
    const Tensor* b,
    bool trans_b) {
  NVTE_CHECK(a->scaling_mode == b->scaling_mode ||
                 (a->scaling_mode == NVTE_BLOCK_SCALING_1D &&
                  b->scaling_mode == NVTE_BLOCK_SCALING_2D) ||
                 (a->scaling_mode == NVTE_BLOCK_SCALING_2D &&
                  b->scaling_mode == NVTE_BLOCK_SCALING_1D),
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

  const bool weight_is_nn_block = (not (a->data).shape.empty()) &&
                                  (product(a->data.shape, 0, a->data.shape.size() - 1) != product(a->scale_inv.shape, 0, a->scale_inv.shape.size() - 1));

  if (weight_is_nn_block || trans_a) {
    info.data_a = &(a->data);
    info.sinv_a = &(a->scale_inv);
  } else  {
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

} // anonymous namespace

void non_fp8_gemm(
    const Tensor* inputA,
    bool transa,
    const Tensor* inputB,
    bool transb,
    Tensor* outputD,
    const Tensor* biasTensor,
    bool accumulate,
    int math_sm_count,
    musaStream_t stream) {
  auto& h = at::GetMudnnHandle();
  h.SetStream(stream);

  const bool has_bias = biasTensor->has_data();
  auto mu_l = CreateMUTensor(*get_data(inputB, transb), Flat2DimShape(inputB));
  auto mu_r = CreateMUTensor(*get_data(inputA, transa), Flat2DimShape(inputA));
  auto mu_b = has_bias ? CreateMUTensor(biasTensor->data) : empty_mu_tensor;
  auto mu_o = CreateMUTensor(outputD->data, Flat2DimShape(outputD));

  ::musa::dnn::MatMul op;
  CHECK_MUDNN_STATUS(op.SetTranspose(transb, transa), "SetTranspose");
  CHECK_MUDNN_STATUS(
      op.SetComputeMode(GetComputeModeFromCtx(ToTorchDtype(inputB->dtype()))),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(op.SetAlpha(1.0), "SetAlpha");
  CHECK_MUDNN_STATUS(op.SetBeta(accumulate ? 1.0 : 0.0), "SetBeta");
  CHECK_MUDNN_STATUS(op.SetGamma(has_bias ? 1.0 : 0.0), "SetGamma");

  CHECK_MUDNN_STATUS(
      op.RunWithBiasAdd(
          h, mu_o, mu_l, mu_r, mu_o, mu_b, InternalMemAlloc),
      "RunWithBiasAdd");
}

void fp8_gemm(
    const Tensor* inputA,
    bool transa,
    const Tensor* inputB,
    bool transb,
    Tensor* outputD,
    const Tensor* biasTensor,
    bool accumulate,
    int math_sm_count,
    musaStream_t stream) {
  auto& h = at::GetMudnnHandle();
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
  auto mu_scale_b = has_bias_scale
      ? CreateMUTensor(biasTensor->scale_inv) : empty_mu_tensor;
  auto mu_scale_o = has_output_scale
      ? CreateMUTensor(outputD->scale): empty_mu_tensor;
  auto mu_amax_o = has_output_amax
      ? CreateMUTensor(outputD->amax): empty_mu_tensor;

  ::musa::dnn::BatchMatMul op;
  CHECK_MUDNN_STATUS(op.SetTranspose(transb, transa), "SetTranspose");
  CHECK_MUDNN_STATUS(
      op.SetComputeMode(GetComputeModeFromCtx(ToTorchDtype(inputB->dtype()))),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(op.SetAlpha(1.0), "SetAlpha");
  CHECK_MUDNN_STATUS(op.SetBeta(accumulate ? 1.0 : 0.0), "SetBeta");
  CHECK_MUDNN_STATUS(op.SetGamma(has_bias ? 1.0 : 0.0), "SetGamma");
  if (math_sm_count != 0) {
    CHECK_MUDNN_STATUS(op.SetMpCountTarget(math_sm_count), "SetMpCountTarget");
  }

  ::musa::dnn::MatMulLtParam param;
  if (info.is_per_tensor) {
    CHECK_MUDNN_STATUS(param.SetScale(mu_scale_l, mu_scale_r, mu_scale_b, mu_scale_o), "SetScale");
  } else {
    NVTE_CHECK(inputB->scale_inv.shape.size() == 2);
    const auto tile_size = static_cast<int>(next_power_of_2(inputB->flat_last_dim() / inputB->scale_inv.shape[1]));
    CHECK_MUDNN_STATUS(param.SetScale(mu_scale_l, mu_scale_r, mu_scale_b, mu_scale_o, tile_size), "SetScale");
  }
  CHECK_MUDNN_STATUS(param.SetAmaxD(mu_amax_o), "SetAmax");

  CHECK_MUDNN_STATUS(op.RunLt(h, mu_o, mu_l, mu_r, mu_o, mu_b, param, InternalMemAlloc), "RunLt");
}

void no_fp8_grad_bias(
    const Tensor* gradO,
    bool trans,
    const Tensor* gradB,
    musaStream_t stream) {
  using REDUCE_MODE = ::musa::dnn::Reduce::Mode;
  const int reduce_dim = trans ? 0 : 1;

  auto& h = at::GetMudnnHandle();
  h.SetStream(stream);

  auto mu_i = CreateMUTensor(gradO->data, Flat2DimShape(gradO));
  auto mu_o = CreateMUTensor(gradB->data, Flat2DimShape(gradB));

  ::musa::dnn::Reduce rdc;
  CHECK_MUDNN_STATUS(rdc.SetMode(REDUCE_MODE::ADD), "SetMode");
  CHECK_MUDNN_STATUS(rdc.SetDim({reduce_dim}), "SetDim");
  CHECK_MUDNN_STATUS(rdc.Run(h, mu_o, mu_i, InternalMemAlloc), "Run");
}

} // namespace transformer_engine

// D = B @ A.T
void mudnn_gemm(
    const NVTETensor A,
    const NVTETensor B,
    NVTETensor D,
    const NVTETensor bias,
    NVTETensor pre_gelu_out,
    bool transa,
    bool transb,
    bool grad,
    NVTETensor workspace,
    bool accumulate,
    bool use_split_accumulator,
    int math_sm_count,
    musaStream_t stream) {
  using namespace transformer_engine;

  const Tensor *inputA = convertNVTETensorCheck(A);
  const Tensor *inputB = convertNVTETensorCheck(B);
  Tensor *outputD = convertNVTETensorCheck(D);
  const Tensor *biasTensor = convertNVTETensor(bias);
  Tensor *geluOut = convertNVTETensor(pre_gelu_out);
  Tensor *wspace = convertNVTETensor(workspace);

  NVTE_CHECK(outputD->has_data());
  NVTE_CHECK(!geluOut->has_data(), "Gelu epilogue is not supported!");

  const auto A_type = inputA->dtype();
  const auto is_fp8_A = is_fp8_dtype(A_type);

  const auto B_type = inputB->dtype();
  const auto is_fp8_B = is_fp8_dtype(B_type);

  NVTE_CHECK(
      is_fp8_A == is_fp8_B,
      "Inputs to muDNN GEMM must all be non-fp8 or fp8 dtypes!");
  if (!is_fp8_A) {
    NVTE_CHECK(
        A_type == B_type,
        "Both inputs to muDNN non-FP8 GEMM must have the same dtype!");
  }
  if (biasTensor->has_data() && !grad) {
    NVTE_CHECK(
        biasTensor->data.shape.size() == 1 &&
            biasTensor->data.shape[0] == outputD->flat_last_dim(),
        "Mismatch bias shape, expect ",
        outputD->flat_last_dim(),
        ", but got ",
        biasTensor->data.shape[0]);
  }

  const auto* fwd_bias = grad ? &transformer_engine::empty_te_tensor : biasTensor;
  if (is_fp8_A) {
    fp8_gemm(inputA, transa, inputB, transb, outputD, fwd_bias, accumulate, math_sm_count, stream);
  } else {
    non_fp8_gemm(inputA, transa, inputB, transb, outputD, fwd_bias, accumulate, math_sm_count, stream);
  }

  if (!grad || !(biasTensor->has_data())) {
    return;
  }

  if (!is_fp8_A) {
    no_fp8_grad_bias(inputB, transb, biasTensor, stream);
  }
}

void nvte_cublas_gemm(
    const NVTETensor A,
    const NVTETensor B,
    NVTETensor D,
    const NVTETensor bias,
    NVTETensor pre_gelu_out,
    bool transa,
    bool transb,
    bool grad,
    NVTETensor workspace,
    bool accumulate,
    bool use_split_accumulator,
    int math_sm_count,
    musaStream_t stream) {
  NVTE_API_CALL(nvte_cublas_gemm);
  mudnn_gemm(
      A, B, D, bias, pre_gelu_out, transa, transb, grad, workspace,
      accumulate, use_split_accumulator, math_sm_count, stream);
}

void nvte_cublas_atomic_gemm(
    const NVTETensor A,
    const NVTETensor B,
    NVTETensor D,
    const NVTETensor bias,
    NVTETensor pre_gelu_out,
    bool transa,
    bool transb,
    bool grad,
    NVTETensor workspace,
    bool accumulate,
    bool use_split_accumulator,
    int math_sm_count,
    int m_split,
    int n_split,
    bool gemm_producer,
    const NVTETensor counter,
    musaStream_t stream) {
  NVTE_API_CALL(nvte_cublas_atomic_gemm);
  NVTE_CHECK(false, "atomic_gemm is not supported.");
}

void nvte_multi_stream_cublas_gemm(
    const NVTETensor* A,
    const NVTETensor* B,
    NVTETensor* D,
    const NVTETensor* bias,
    NVTETensor* pre_gelu_out,
    const int num_gemms,
    bool transa,
    bool transb,
    bool grad,
    NVTETensor* workspace,
    bool accumulate,
    bool use_split_accumulator,
    int math_sm_count,
    musaStream_t stream) {
  NVTE_API_CALL(nvte_multi_stream_cublas_gemm);
  using namespace transformer_engine;

  std::call_once(init_flag, init_streams_and_events);

  int num_stream_used = std::min(num_streams, num_gemms);
  // wait for current stream to finish
  NVTE_CHECK_CUDA(musaEventRecord(cublas_event[0], stream));
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(musaStreamWaitEvent(compute_streams[s], cublas_event[0]));
  }

  for (int i = 0; i < num_gemms; i++) {
    musaStream_t stream_to_use;
    if (multistream_to_use) {
      stream_to_use = compute_streams[i % num_streams];
    } else {
      stream_to_use = musaStreamDefault;
    }

    mudnn_gemm(
        A[i], B[i], D[i], bias[i], pre_gelu_out[i], transa, transb, grad,
        workspace[i % num_streams], accumulate, use_split_accumulator, math_sm_count,
        stream_to_use); // compute_streams[i % num_streams]
  }
  if (multistream_to_use) {
    // record events on compute streams
    for (int s = 0; s < num_stream_used; s++) {
      NVTE_CHECK_CUDA(musaEventRecord(cublas_event[s], compute_streams[s]));
    }
    // wait for all compute streams to finish
    for (int s = 0; s < num_stream_used; s++) {
      NVTE_CHECK_CUDA(musaStreamWaitEvent(stream, cublas_event[s]));
    }
  }
}
