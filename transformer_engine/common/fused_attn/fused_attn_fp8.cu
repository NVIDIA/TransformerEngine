/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn.h"

#include "../common.h"
#include "utils.h"
#include "fused_attn_fp8.h"

namespace transformer_engine {
namespace fused_attn {

using namespace transformer_engine;

#if (CUDNN_VERSION >= 8900)
std::unordered_map<std::string, int> tensor_name_to_uid = {
  {"Q",                            1},
  {"K",                            2},
  {"V",                            3},
  {"O",                            4},
  {"S",                            5},
  {"B",                            6},
  {"DROPOUT_SCALE",                7},
  {"S_CONST",                      8},
  {"MNK_OVERRIDE",                 9},
  {"dQ",                          11},
  {"dK",                          12},
  {"dV",                          13},
  {"dO",                          14},
  {"MASK_VAL",                    15},
  {"dS",                          16},
  {"O_SEQLEN",                    17},
  {"M",                           18},
  {"Z",                           19},
  {"descaleQ",                    20},
  {"descaleK",                    21},
  {"descaleV",                    22},
  {"descaleS",                    23},
  {"scaleS",                      24},
  {"amaxS",                       25},
  {"amaxO",                       26},
  {"QKV_RAGGED",                  27},
  {"O_RAGGED",                    28},
  {"K_TRANSPOSE",                 29},
  {"AttnScale",                   30},
  {"scaleO",                      31},
  {"Z_INV",                       32},
  {"descaleO",                    33},
  {"descaledO",                   34},
  {"descaledS",                   35},
  {"descaledQ",                   36},
  {"descaledK",                   37},
  {"descaledV",                   38},
  {"scaledS",                     39},
  {"scaledQ",                     40},
  {"scaledK",                     41},
  {"scaledV",                     42},
  {"amaxdS",                      43},
  {"amaxdQ",                      44},
  {"amaxdK",                      45},
  {"amaxdV",                      46},
  {"V_TRANSPOSE",                 47},
  {"AttnScale_dS_K",              48},
  {"AttnScale_dSTranspose_Q",     49},
  {"DROPOUT_SCALE_dOVt_OdO",      50},
  {"DROPOUT_OFFSET",              51},
  {"DROPOUT_SEED",                52},
  {"VIRTUAL",                     80}
};

static cudnn_frontend::Tensor createAmax(
            const std::string& amax_tensor_name,
            const cudnn_frontend::Tensor& prevBlockOutputTensor,
            std::vector<cudnn_frontend::Operation>* ops) {
  int64_t amax_dim[4] = {1, 1, 1, 1};
  int64_t amax_stride[4] = {1, 1, 1, 1};
  auto amaxTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid[amax_tensor_name],
                  amax_dim, amax_stride, false, false);

  // Define the amax descriptor
  auto reductionDesc = cudnn_frontend::ReductionDescBuilder()
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .setReductionOp(CUDNN_REDUCE_TENSOR_AMAX)
                            .build();

  // Create a reduction amax Node
  auto reduction_op = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                          .setxDesc(prevBlockOutputTensor)
                          .setyDesc(amaxTensor)
                          .setreductionDesc(reductionDesc)
                          .build();
  ops->push_back(std::move(reduction_op));
  return amaxTensor;
}

static cudnn_frontend::Tensor createScale(
                const cudnn_frontend::Tensor& prevBlockOutputTensor,
                const std::string& scale_tensor_name,
                cudnnDataType_t tensorType,
                bool isOutputVirtual, bool isScaleByValue,
                std::vector<cudnn_frontend::Operation>* ops,
                const std::string& output_tensor_name ="") {
  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  int64_t output_dim[4];
  int64_t output_stride[4];

  for (int i = 0; i < 4; i++) {
      output_dim[i] = prevBlockOutputTensor.getDim()[i];
      output_stride[i] = prevBlockOutputTensor.getStride()[i];
  }

  auto scaleTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid[scale_tensor_name],
                  scale_dim, scale_stride, false, isScaleByValue);  // is by value

  int64_t outputUID = isOutputVirtual ? tensor_name_to_uid["VIRTUAL"]
          + tensor_name_to_uid[scale_tensor_name] + 5000 :
          tensor_name_to_uid[output_tensor_name];
  auto afterScaleKTensor = tensor_create(
                  tensorType, outputUID, output_dim,
                  output_stride, isOutputVirtual, false);  // is virtual

  // Define the scale descriptor
  auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a Scale Node
  auto scale_op = binary_pw_op_create(
                  prevBlockOutputTensor, scaleTensor, afterScaleKTensor, scaleDesc);

  ops->push_back(std::move(scale_op));
  return afterScaleKTensor;
}

static cudnn_frontend::Tensor createScale(
                const cudnn_frontend::Tensor& prevBlockOutputTensor,
                const cudnn_frontend::Tensor& scaleTensor,
                cudnnDataType_t tensorType,
                bool isOutputVirtual, bool isScaleByValue,
                std::vector<cudnn_frontend::Operation>* ops,
                int UID_offset, const std::string& output_tensor_name ="") {
  int64_t output_dim[4];
  int64_t output_stride[4];
  for (int i = 0; i < 4; i++) {
      output_dim[i] = prevBlockOutputTensor.getDim()[i];
      output_stride[i] = prevBlockOutputTensor.getStride()[i];
  }

  int64_t outputUID = isOutputVirtual ?
          tensor_name_to_uid["VIRTUAL"] + UID_offset :
          tensor_name_to_uid[output_tensor_name];
  auto afterScaleTensor = tensor_create(
                  tensorType, outputUID, output_dim,
                  output_stride, isOutputVirtual, false);  // is virtual

  // Define the scale descriptor
  auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a Scale Node
  auto scale_op = binary_pw_op_create(
                  prevBlockOutputTensor, scaleTensor, afterScaleTensor, scaleDesc);

  ops->push_back(std::move(scale_op));
  return afterScaleTensor;
}

static cudnn_frontend::Tensor createScaleWithOffset(
            const cudnn_frontend::Tensor& prevBlockOutputTensor,
            const std::string& scale_tensor_name,
            NVTE_QKV_Layout layout,
            cudnnDataType_t tensorType,
            bool isOutputVirtual,
            bool isScaleByValue,
            std::vector<cudnn_frontend::Operation>* ops,
            std::shared_ptr<cudnn_frontend::Tensor> offsetTensor,
            const std::string& output_tensor_name ="") {
  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  int64_t output_dim[4];
  int64_t output_stride[4];
  // If output tensor is dQ, dK, or dV, we need to generate QKV interleaved strides
  if (output_tensor_name == "dQ" || output_tensor_name == "dK" || output_tensor_name == "dV") {
      for (int i = 0; i < 4; i++) {
          output_dim[i] = prevBlockOutputTensor.getDim()[i];
      }
      generateMatrixStrides(output_dim[0], output_dim[1], output_dim[2],
                      0  /*s_kv = 0 for placeholder*/,
                      output_dim[3], output_stride,
                      layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
  } else {
      // Otherwise output dim and stride should be the same as prev block dim and stride
      for (int i = 0; i < 4; i++) {
          output_dim[i] = prevBlockOutputTensor.getDim()[i];
          output_stride[i] = prevBlockOutputTensor.getStride()[i];
      }
  }

  auto scaleTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid[scale_tensor_name],
                  scale_dim, scale_stride, false, isScaleByValue);  // is by value

  cudnnDataType_t outputDataType = isOutputVirtual ? CUDNN_DATA_FLOAT : tensorType;
  int64_t outputUID = isOutputVirtual ?
          tensor_name_to_uid["VIRTUAL"] + tensor_name_to_uid[scale_tensor_name] + 7000 :
          tensor_name_to_uid[output_tensor_name];
  auto afterScaleTensor = tensor_create_with_offset(
                  outputDataType, outputUID, output_dim,
                  output_stride, isOutputVirtual, false, offsetTensor);  // is virtual

  // Define the scale descriptor
  auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a Scale Node
  auto scale_op = binary_pw_op_create(
                  prevBlockOutputTensor, scaleTensor, afterScaleTensor, scaleDesc);

  ops->push_back(std::move(scale_op));
  return afterScaleTensor;
}

static cudnn_frontend::Tensor createSoftmaxForward(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor& prevBlockOutputTensor,
            bool isTraining) {
  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t afterReduction_dim[4] = {b, h, s_q, 1};
  int64_t afterReduction_stride[4] = {h * s_q, s_q, 1, 1};

  // max (x) (M tensor)
  auto afterMaxReductionTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["M"],
                  afterReduction_dim, afterReduction_stride,
                  !isTraining, false);  // not virtual if training is true,
                                        // virtual if training is false
  // x - max(x)
  auto afterSubtractionTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 151,
                  afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual
  // e^(x - max(x))
  auto afterExponentTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 152,
                  afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual;
  // sum (e^(x - max(x))) (Z tensor)
  auto zTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["Z"],
                  afterReduction_dim, afterReduction_stride, true, false);  // is virtual
  // 1 / sum (e^(x - max(x))) (Z_INV tensor)
  auto zInvTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["Z_INV"],
                  afterReduction_dim, afterReduction_stride,
                  !isTraining, false);  // not virtual if training is true,
                                        // virtual if training is false
  // Final softmax output (After exponent * Z_INV)
  auto beforeDropoutTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 153,
                  afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual

  // Define the reduction descriptor
  auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                              .build();

  // Create a reduction max Node
  auto reductionMax_op = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                  .setxDesc(prevBlockOutputTensor)
                  .setyDesc(afterMaxReductionTensor)
                  .setreductionDesc(reductionMaxDesc)
                  .build();

  // Define the subtract descriptor
  auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

  // Create a subtract Node
  auto subtract_op = binary_pw_op_create(
                  prevBlockOutputTensor, afterMaxReductionTensor,
                  afterSubtractionTensor, subtractDesc);

  // Define the exponent descriptor
  auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

  // Create a exponent Node
  auto exponent_op = unary_pw_op_create(
                  afterSubtractionTensor, afterExponentTensor, exponentDesc);

  // Define the reduction descriptor
  auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                              .build();

  // Create a reduction add Node
  auto reductionAdd_op = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                  .setxDesc(afterExponentTensor)
                  .setyDesc(zTensor)
                  .setreductionDesc(reductionAddDesc)
                  .build();

  // Define the reciprocal descriptor
  auto reciprocalDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_RECIPROCAL);

  // Create a reciprocal Node
  auto reciprocal_op = unary_pw_op_create(zTensor, zInvTensor, reciprocalDesc);

  // Define the pw multiply descriptor
  auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply Node
  auto mutliply_op = binary_pw_op_create(
                  afterExponentTensor, zInvTensor, beforeDropoutTensor, multiplyDesc);

  ops->push_back(std::move(reductionMax_op));
  ops->push_back(std::move(subtract_op));
  ops->push_back(std::move(exponent_op));
  ops->push_back(std::move(reductionAdd_op));
  ops->push_back(std::move(reciprocal_op));
  ops->push_back(std::move(mutliply_op));

  return beforeDropoutTensor;
}

static cudnn_frontend::Tensor createDropoutForward(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
            double probability,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor& beforeDropoutTensor) {
  NVTE_CHECK(ops->size() > 0,
             "Dropout DAG constructed incorrectly as the first one");

  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  // Mask for the dropout
  auto dropoutMaskTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 250,
                  afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual
  auto dropoutSeedTensor = tensor_create(
                  CUDNN_DATA_INT64, tensor_name_to_uid["DROPOUT_SEED"],
                  scale_dim, scale_stride, false, false);  // is by value
  auto dropoutOffsetTensor = tensor_create(
                  CUDNN_DATA_INT64, tensor_name_to_uid["DROPOUT_OFFSET"],
                  scale_dim, scale_stride, false, false);  // is by value

  // After dropout tensor befor scale
  auto beforeDropoutScaleTensor = cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setId(tensor_name_to_uid["VIRTUAL"] + 201)
          .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(CUDNN_DATA_FLOAT)
          .setVirtual(true)
          .setByValue(false)
          .setReorderType(cudnn_frontend::TensorReordering_t::F16x16)
          .build();
  // Scale after dropout
  auto scaleDropoutTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["DROPOUT_SCALE"],
                  scale_dim, scale_stride, false, true);  // is by value
  // After Scale
  auto afterDropout_before_quan_S = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 202,
                  afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual

  // Define the reduction descriptor
  auto rngDesc = cudnn_frontend::RngDescBuilder()
                              .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                              .setBernoulliDistProbability(1.0 - probability)
                              .build();

  // Create a rng Node
  auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                              .setyDesc(dropoutMaskTensor)
                              .setSeedDesc(dropoutSeedTensor)
                              .setOffsetDesc(dropoutOffsetTensor)
                              .setRngDesc(rngDesc)
                              .build();


  // Define the multiply mask descriptor
  auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply mask Node
  auto maskMul_op = binary_pw_op_create(
                  beforeDropoutTensor, dropoutMaskTensor,
                  beforeDropoutScaleTensor, maskMulDesc);

  // Define the multiply scale descriptor
  auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply mask Node
  auto scaleMul_op = binary_pw_op_create(
                  beforeDropoutScaleTensor, scaleDropoutTensor,
                  afterDropout_before_quan_S, scaleMulDesc);

  ops->push_back(std::move(rng_op));
  ops->push_back(std::move(maskMul_op));
  ops->push_back(std::move(scaleMul_op));

  return afterDropout_before_quan_S;
}

static cudnn_frontend::Tensor createDropoutBackward(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
            double probability,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor& beforeDropoutTensor,
            const cudnn_frontend::Tensor& dropoutMaskTensor) {
  NVTE_CHECK(ops->size() > 0,
             "Dropout DAG constructed incorrectly as the first one");

  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  auto dropoutSeedTensor = tensor_create(
                  CUDNN_DATA_INT64, tensor_name_to_uid["DROPOUT_SEED"],
                  scale_dim, scale_stride, false, false);  // is by value
  auto dropoutOffsetTensor = tensor_create(
                  CUDNN_DATA_INT64, tensor_name_to_uid["DROPOUT_OFFSET"],
                  scale_dim, scale_stride, false, false);  // is by value

  // After dropout tensor befor scale
  auto beforeDropoutScaleTensor = cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setId(tensor_name_to_uid["VIRTUAL"] + 201)
          .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(CUDNN_DATA_FLOAT)
          .setVirtual(true)
          .setByValue(false)
          .setReorderType(cudnn_frontend::TensorReordering_t::F16x16)
          .build();
  // Scale after dropout (1 / (1 - p))
  auto scaleDropoutTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["DROPOUT_SCALE"],
                  scale_dim, scale_stride, false, true);  // is by value
  // After Scale
  auto afterDropout_before_quan_S = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 202,
                  afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual

  // Define the reduction descriptor
  auto rngDesc = cudnn_frontend::RngDescBuilder()
                              .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                              .setBernoulliDistProbability(1.0 - probability)
                              .build();

  // Create a rng Node
  auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                              .setyDesc(dropoutMaskTensor)
                              .setSeedDesc(dropoutSeedTensor)
                              .setOffsetDesc(dropoutOffsetTensor)
                              .setRngDesc(rngDesc)
                              .build();

  // Define the multiply mask descriptor
  auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply mask Node
  auto maskMul_op = binary_pw_op_create(
                  beforeDropoutTensor, dropoutMaskTensor,
                  beforeDropoutScaleTensor, maskMulDesc);

  // Define the multiply scale descriptor
  auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply mask Node
  auto scaleMul_op = binary_pw_op_create(
                  beforeDropoutScaleTensor, scaleDropoutTensor,
                  afterDropout_before_quan_S, scaleMulDesc);

  ops->push_back(std::move(rng_op));
  ops->push_back(std::move(maskMul_op));
  ops->push_back(std::move(scaleMul_op));

  return afterDropout_before_quan_S;
}

static cudnn_frontend::Tensor createSoftmaxBackward(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor& dyTensor) {
  NVTE_CHECK(ops->size() > 0,
             "Softmax backward constructed incorrectly as the first one");

  int64_t dx_dim[4] = {b, h, s_q, s_kv};
  int64_t dx_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t M_Z_dim[4] = {b, h, s_q, 1};
  int64_t M_Z_stride[4] = {h * s_q, s_q, 1, 1};

  // Creating all tensors
  auto MTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["M"],
                  M_Z_dim, M_Z_stride, false, false);  // not virtual
  auto ZInvTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["Z_INV"],
                  M_Z_dim, M_Z_stride, false, false);  // not virtual
  auto dxAfterSubtractionTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 252,
                  dx_dim, dx_stride, true, false);  // is virtual
  auto dxAfterExponentiation = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 253,
                  dx_dim, dx_stride, true, false);  // is virtual
  auto dxBeforeDropout_QKt_Tensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 254,
                  dx_dim, dx_stride, true, false);  // is virtual

  // Creating all ops
  // sub (dy - M)
  auto subtractionDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);
  auto subtractionOp = binary_pw_op_create(
                  dyTensor, MTensor, dxAfterSubtractionTensor, subtractionDesc);

  // Define the exponent descriptor
  auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

  // Create a exponent Node. (exp(dy - M))
  auto exponentOp = unary_pw_op_create(
                  dxAfterSubtractionTensor, dxAfterExponentiation, exponentDesc);

  // Define the pw multiply descriptor
  auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply Node
  auto mutliplyOp = binary_pw_op_create(
                  dxAfterExponentiation, ZInvTensor, dxBeforeDropout_QKt_Tensor, multiplyDesc);

  ops->push_back(std::move(subtractionOp));
  ops->push_back(std::move(exponentOp));
  ops->push_back(std::move(mutliplyOp));

  return dxBeforeDropout_QKt_Tensor;
}

static cudnn_frontend::Tensor createQKBMM(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            NVTE_QKV_Layout layout,
            cudnnDataType_t tensorType,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor &qTensor,
            const cudnn_frontend::Tensor &kTensor,
            const cudnn_frontend::Tensor &mnkOverride,
            std::shared_ptr<cudnn_frontend::Tensor> QKVRaggedOffsetTensor) {
  // Creates the necessary tensor descriptors
  int64_t k_transpose_dim[4] = {b, h, d, s_kv};
  int64_t k_transpose_stride[4];
  generateMatrixStrides(
                  b, h, s_q, s_kv, d,
                  k_transpose_stride, layout, NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose);

  int64_t s_dim[4] = {b, h, s_q, s_kv};
  int64_t s_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, s_stride, layout, NVTE_QKV_Matrix::NVTE_S_Matrix);

  auto kTransposeTensor = tensor_create_with_offset(
                  tensorType, tensor_name_to_uid["K_TRANSPOSE"],
                  k_transpose_dim, k_transpose_stride,
                  false, false, QKVRaggedOffsetTensor);  // is virtual

  // First GEMM output
  auto afterQKTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 1,
                  s_dim, s_stride, true, false);  // is virtual

  // Define the matmul desc
  auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                                  .setComputeType(CUDNN_DATA_FLOAT)
                                  .setPaddingValue(-2000000)
                                  .build();

  // Create reshape node for K -> K.T
  auto reshape_op = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                          .setxDesc(kTensor)
                          .setyDesc(kTransposeTensor)
                          .build();

  // Create a matmul Node
  auto matmulOp = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                          .setaMatDesc(qTensor)
                          .setbMatDesc(kTransposeTensor)
                          .setcMatDesc(afterQKTensor)
                          .setmOverrideDesc(mnkOverride)
                          .setnOverrideDesc(mnkOverride)
                          .setmatmulDesc(matmulDesc)
                          .build();

  ops->push_back(std::move(reshape_op));
  ops->push_back(std::move(matmulOp));

  return afterQKTensor;
}

static cudnn_frontend::Tensor createSVBMM(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            NVTE_QKV_Layout layout,
            cudnnDataType_t tensorType,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor &softmaxTensor,
            const cudnn_frontend::Tensor &mnkOverride,
            std::shared_ptr<cudnn_frontend::Tensor> QKVRaggedOffsetTensor) {
  NVTE_CHECK(ops->size() > 0,
             "BMM2 op constructed incorrectly as the first one");

  int64_t v_dim[4] =  {b, h, s_kv, d};
  int64_t v_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, v_stride, layout, NVTE_QKV_Matrix::NVTE_V_Matrix);

  int64_t o_dim[4] =  {b, h, s_q, d};
  int64_t o_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, o_stride, layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

  auto vTensor = tensor_create_with_offset(
                  tensorType, tensor_name_to_uid["V"],
                  v_dim, v_stride, false, false, QKVRaggedOffsetTensor);
  // Second fprop GEMM output
  auto oTensor = tensor_create(
                  tensorType, tensor_name_to_uid["VIRTUAL"] + 300,
                  o_dim, o_stride, true, false);  // is virtual

  // Define the matmul desc
  auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();

  // Create a matmul Node
  auto matmulOp = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                          .setaMatDesc(softmaxTensor)
                          .setbMatDesc(vTensor)
                          .setcMatDesc(oTensor)
                          .setmOverrideDesc(mnkOverride)
                          .setkOverrideDesc(mnkOverride)
                          .setmatmulDesc(matmulDesc)
                          .build();

  ops->push_back(std::move(matmulOp));

  return oTensor;
}

static cudnn_frontend::Tensor createSdOBMM(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            cudnnDataType_t tensorType,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor &softmaxTensor,
            const cudnn_frontend::Tensor &dOTensor,
            const cudnn_frontend::Tensor &mnkOverride) {
  NVTE_CHECK(ops->size() > 0,
             "BMM2 op constructed incorrectly as the first one");

  int64_t s_dim_transpose[4] =  {b, h, s_kv, s_q};
  int64_t s_stride_transpose[4] = {h * s_kv * s_q, s_kv * s_q, 1, s_kv};

  int64_t v_dim[4] =  {b, h, s_kv, d};
  int64_t v_stride[4] = {h * s_kv * d, d, h * d, 1};

  auto sTransposeTensor = tensor_create(
                  tensorType, tensor_name_to_uid["VIRTUAL"] + 499,
                  s_dim_transpose, s_stride_transpose,
                  true, false);  // is virtual
  // S.T * dO
  auto dVTensor_before_dequan_S = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 500,
                  v_dim, v_stride,
                  true, false);  // is virtual

  // Create reshape node for softmax -> softmax.T
  auto reshape_op = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                          .setxDesc(softmaxTensor)
                          .setyDesc(sTransposeTensor)
                          .build();

  // Define the matmul desc
  auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                                  .setComputeType(CUDNN_DATA_FLOAT)
                                  .setPaddingValue(0)
                                  .build();

  // Create a matmul Node
  auto matmulOp = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                          .setaMatDesc(sTransposeTensor)
                          .setbMatDesc(dOTensor)
                          .setcMatDesc(dVTensor_before_dequan_S)
                          .setmOverrideDesc(mnkOverride)
                          .setkOverrideDesc(mnkOverride)
                          .setmatmulDesc(matmulDesc)
                          .build();

  ops->push_back(std::move(reshape_op));
  ops->push_back(std::move(matmulOp));

  return dVTensor_before_dequan_S;
}

static cudnn_frontend::Tensor createdOVBMM(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            NVTE_QKV_Layout layout,
            cudnnDataType_t tensorType,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor &dOTensor,
            const cudnn_frontend::Tensor &mnkOverride,
            std::shared_ptr<cudnn_frontend::Tensor> QKVRaggedOffsetTensor) {
  // Creates the necessary tensor descriptors
  int64_t v_dim[4] =  {b, h, s_kv, d};
  int64_t v_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, v_stride, layout, NVTE_QKV_Matrix::NVTE_V_Matrix);

  int64_t v_transpose_dim[4] = {b, h, d, s_kv};
  int64_t v_transpose_stride[4];
  v_transpose_stride[0] = v_stride[0];
  v_transpose_stride[1] = v_stride[1];
  v_transpose_stride[2] = v_stride[3];
  v_transpose_stride[3] = v_stride[2];

  int64_t s_dim[4] = {b, h, s_q, s_kv};
  int64_t s_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, s_stride, layout, NVTE_QKV_Matrix::NVTE_S_Matrix);

  auto vTensor = tensor_create_with_offset(
                  tensorType, tensor_name_to_uid["V"],
                  v_dim, v_stride,
                  false, false, QKVRaggedOffsetTensor);
  auto vTransposeTensor = tensor_create_with_offset(
                  tensorType, tensor_name_to_uid["V_TRANSPOSE"],
                  v_transpose_dim, v_transpose_stride,
                  false, false, QKVRaggedOffsetTensor);  // is virtual

  // dO * V.T
  auto afterdOVTensor = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 600,
                  s_dim, s_stride, true, false);  // is virtual

  // Define the matmul desc
  auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                                          .setComputeType(CUDNN_DATA_FLOAT)
                                          .setPaddingValue(0)
                                          .build();

  // Create reshape node for V -> V.T
  auto reshape_op = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                          .setxDesc(vTensor)
                          .setyDesc(vTransposeTensor)
                          .build();

  // Create a matmul Node
  auto matmulOp = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                          .setaMatDesc(dOTensor)
                          .setbMatDesc(vTransposeTensor)
                          .setcMatDesc(afterdOVTensor)
                          .setmOverrideDesc(mnkOverride)
                          .setnOverrideDesc(mnkOverride)
                          .setmatmulDesc(matmulDesc)
                          .build();

  ops->push_back(std::move(reshape_op));
  ops->push_back(std::move(matmulOp));

  return afterdOVTensor;
}

static cudnn_frontend::Tensor createdOAndORowReductionChain(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            NVTE_QKV_Layout layout,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor &O_after_dequan,
            const cudnn_frontend::Tensor &dO_after_dequan,
            const cudnn_frontend::Tensor &dropoutScale_dOVt_OdO_Tensor) {
  int64_t o_dim[4] = {b, h, s_q, d};
  int64_t o_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, o_stride, layout, NVTE_QKV_Matrix::NVTE_O_Matrix);
  int64_t o_dim_row_sum[4] = {b, h, s_q, 1};
  int64_t o_dim_row_sum_stride[4] = {s_q * h, s_q, 1, 1};

  auto O_dO_after_pointwise_multiply = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 700,
                  o_dim, o_stride, true, false);  // is virtual
  auto O_dO_after_dropout_scale = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 701,
                  o_dim, o_stride, true, false);  // is virtual
  auto O_dO_after_rowsum = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 702,
                  o_dim_row_sum, o_dim_row_sum_stride, true, false);  // is virtual

  // Define the pw multiply descriptor
  auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply Node
  auto mutliply_op = binary_pw_op_create(
                  O_after_dequan, dO_after_dequan,
                  O_dO_after_pointwise_multiply, multiplyDesc);

  // Create multiply node with dropout scale
  auto dropout_scale_multiply_op = binary_pw_op_create(
                  O_dO_after_pointwise_multiply, dropoutScale_dOVt_OdO_Tensor,
                  O_dO_after_dropout_scale, multiplyDesc);

  // Define the reduction descriptor
  auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                              .build();

  // Create a reduction add Node
  auto reductionAdd_op = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                              .setxDesc(O_dO_after_dropout_scale)
                              .setyDesc(O_dO_after_rowsum)
                              .setreductionDesc(reductionAddDesc)
                              .build();

  ops->push_back(std::move(mutliply_op));
  ops->push_back(std::move(dropout_scale_multiply_op));
  ops->push_back(std::move(reductionAdd_op));

  return O_dO_after_rowsum;
}

static cudnn_frontend::Tensor createBiasSubtractionSoftmaxMulChain(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            NVTE_QKV_Layout layout,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor &dS_after_dropout,
            const cudnn_frontend::Tensor &AfterDropout_before_quan_S,
            const cudnn_frontend::Tensor &O_dO_after_rowsum,
            const cudnn_frontend::Tensor &attnScale) {
  int64_t o_dim[4] = {b, h, s_q, s_kv};
  int64_t o_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, o_stride, layout, NVTE_QKV_Matrix::NVTE_S_Matrix);
  auto dS_minus_O_dO = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 800,
                  o_dim, o_stride, true, false);  // is virtual
  auto AfterAttnScale_before_dS = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 801,
                  o_dim, o_stride, true, false);  // is virtual
  auto S_mul_dS_minus_O_dO = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 802,
                  o_dim, o_stride, true, false);  // is virtual

  // Define the pw subtraction descriptor
  auto subDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

  // Create a subtraction Node
  auto sub_op = binary_pw_op_create(
                  dS_after_dropout, O_dO_after_rowsum, dS_minus_O_dO, subDesc);

  // Define the pw multiplication descriptor
  auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // dS_minus_O_dO * attnScale
  auto mutliply_attn_scale_op = binary_pw_op_create(
          dS_minus_O_dO, attnScale,
          AfterAttnScale_before_dS, multiplyDesc);

  // AfterDropout_before_quan_S * AfterAttnScale_before_dS
  auto mutliply_op = binary_pw_op_create(
          AfterDropout_before_quan_S, AfterAttnScale_before_dS,
          S_mul_dS_minus_O_dO, multiplyDesc);

  ops->push_back(std::move(sub_op));
  ops->push_back(std::move(mutliply_attn_scale_op));
  ops->push_back(std::move(mutliply_op));

  return S_mul_dS_minus_O_dO;
}

static cudnn_frontend::Tensor createdSKBMM(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor &dSTensor,
            const cudnn_frontend::Tensor &kTensor,
            const cudnn_frontend::Tensor &mnkOverride) {
  // Creates the necessary tensor descriptors
  int64_t after_dSK_dim[4] = {b, h, s_kv, d};
  int64_t after_dSK_stride[4] = {h * s_kv * d, d, h * d, 1};
  // dS * K
  auto After_dS_K = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 875,
                  after_dSK_dim, after_dSK_stride, true, false);  // is virtual

  // Define the matmul desc
  auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                                  .setComputeType(CUDNN_DATA_FLOAT)
                                  .setPaddingValue(0)
                                  .build();

  // Create a matmul Node
  auto matmulOp = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                          .setaMatDesc(dSTensor)
                          .setbMatDesc(kTensor)
                          .setcMatDesc(After_dS_K)
                          .setmOverrideDesc(mnkOverride)
                          .setkOverrideDesc(mnkOverride)
                          .setmatmulDesc(matmulDesc)
                          .build();

  ops->push_back(std::move(matmulOp));

  return After_dS_K;
}

static cudnn_frontend::Tensor createdSQBMM(
            int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            NVTE_QKV_Layout layout,
            std::vector<cudnn_frontend::Operation>* ops,
            const cudnn_frontend::Tensor &dSTensor,
            const cudnn_frontend::Tensor &qTensor,
            const cudnn_frontend::Tensor &mnkOverride) {
  // Creates the necessary tensor descriptors
  int64_t dS_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, dS_stride, layout, NVTE_QKV_Matrix::NVTE_S_Matrix);

  int64_t dS_transpose_dim[4] = {b, h, s_kv, s_q};
  int64_t dS_transpose_stride[4];
  dS_transpose_stride[0] = dS_stride[0];
  dS_transpose_stride[1] = dS_stride[1];
  dS_transpose_stride[2] = dS_stride[3];
  dS_transpose_stride[3] = dS_stride[2];

  int64_t after_dSTranspose_Q_dim[4] = {b, h, s_kv, d};
  int64_t after_dSTranspose_Q_stride[4] = {h * s_kv * d, d, h * d, 1};

  auto dSTransposeTensor = tensor_create(
                  CUDNN_DATA_FP8_E5M2, tensor_name_to_uid["VIRTUAL"] + 650,
                  dS_transpose_dim, dS_transpose_stride, true, false);  // is virtual

  // dS.T * Q
  auto After_dSTranspose_Q = tensor_create(
                  CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 651,
                  after_dSTranspose_Q_dim, after_dSTranspose_Q_stride,
                  true, false);  // is virtual

  // Create reshape node for V -> V.T
  auto reshape_op = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                          .setxDesc(dSTensor)
                          .setyDesc(dSTransposeTensor)
                          .build();

  // Define the matmul desc
  auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .setPaddingValue(0)
                            .build();

  // Create a matmul Node
  auto matmulOp = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                          .setaMatDesc(dSTransposeTensor)
                          .setbMatDesc(qTensor)
                          .setcMatDesc(After_dSTranspose_Q)
                          .setmOverrideDesc(mnkOverride)
                          .setkOverrideDesc(mnkOverride)
                          .setmatmulDesc(matmulDesc)
                          .build();

  ops->push_back(std::move(reshape_op));
  ops->push_back(std::move(matmulOp));

  return After_dSTranspose_Q;
}

// fused attention FWD FP8
void fused_attn_fp8_fwd_impl(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            bool isTraining, float attnScale,
            float dropoutProbability, NVTE_QKV_Layout layout,
            void* devPtrQ, void* devPtrK, void* devPtrV,
            void* devPtrM, void* devPtrZInv,
            void* devPtrO,
            void* devPtrDescaleQ, void* devPtrDescaleK, void* devPtrDescaleV,
            void* devPtrDescaleS, void* devPtrScaleS, void* devPtrScaleO,
            void* devPtrAmaxO, void* devPtrAmaxS,
            void* devPtrcuSeqlensQ, void* devPtrcuSeqlensKV,
            void* devPtrDropoutSeed, void* devPtrDropoutOffset,
            cudnnDataType_t tensorType,
            void* workspace_ptr,
            size_t* workspace_size,
            cudaStream_t stream,
            cudnnHandle_t handle_) {
  try {
      FADescriptor descriptor{
              b, h, s_q, s_kv, d,
              attnScale, isTraining, dropoutProbability, layout,
              NVTE_Bias_Type::NVTE_NO_BIAS, NVTE_Mask_Type::NVTE_PADDING_MASK, tensorType, false};

      using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
      static thread_local CacheType fa_fprop_cache;

      // Get plan from cache if cache is available, otherwise create one
      auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {
          // If hit, return
          auto it = cache.find(descriptor);
          if (it != cache.end()) {
            auto plan = it->second;
            return plan;
          }

          // Otherwise, build the op_graph and the plan. Then update cache
          std::vector<cudnn_frontend::Operation const*> all_ops;
          std::vector<cudnn_frontend::Operation> ops;

          NVTE_CHECK(dropoutProbability == 0.0f || isTraining,
                     "Dropout probability should be 0.0f for inference mode");
          NVTE_CHECK(dropoutProbability != 1.0f,
                     "Dropout probability cannot be 1.0");

          int64_t raggedDim[4] =  {b + 1, 1, 1, 1};
          int64_t raggedStride[4] = {1, 1, 1, 1};
          // Create offset tensors
          auto QKVOffsetTensor = tensor_create(
                          CUDNN_DATA_INT32, tensor_name_to_uid["QKV_RAGGED"],
                          raggedDim, raggedStride, false, false);
          auto ORaggedOffsetTensor = tensor_create(
                          CUDNN_DATA_INT32, tensor_name_to_uid["O_RAGGED"],
                          raggedDim, raggedStride, false, false);

          int64_t seqlen_dim[4] =  {b, 1, 1, 1};
          int64_t seqlen_stride[4] = {1, 1, 1, 1};
          // Create override tensors
          auto seqlenMNKTensor = tensor_create(
                          CUDNN_DATA_INT32, tensor_name_to_uid["MNK_OVERRIDE"],
                          seqlen_dim, seqlen_stride, false, false);

          // Create shared ptrs to ragged offset tensors
          // for multiple tensors to use ragged offset
          std::shared_ptr<cudnn_frontend::Tensor> QKVRaggedOffsetTensorPtr =
                  std::make_shared<cudnn_frontend::Tensor>(std::move(QKVOffsetTensor));
          std::shared_ptr<cudnn_frontend::Tensor> ORaggedOffsetTensorPtr =
                  std::make_shared<cudnn_frontend::Tensor>(std::move(ORaggedOffsetTensor));

          // Create Q and K tensors that are used in different places
          int64_t q_dim[4] = {b, h, s_q, d};
          int64_t q_stride[4];
          generateMatrixStrides(b, h, s_q, s_kv, d, q_stride, layout,
                          NVTE_QKV_Matrix::NVTE_Q_Matrix);

          int64_t k_dim[4] =  {b, h, s_kv, d};
          int64_t k_stride[4];
          generateMatrixStrides(b, h, s_q, s_kv, d, k_stride, layout,
                          NVTE_QKV_Matrix::NVTE_K_Matrix);

          auto qTensor = tensor_create_with_offset(
                          tensorType, tensor_name_to_uid["Q"],
                          q_dim, q_stride, false, false,
                          QKVRaggedOffsetTensorPtr);
          auto kTensor = tensor_create_with_offset(
                          tensorType, tensor_name_to_uid["K"],
                          k_dim, k_stride, false, false,
                          QKVRaggedOffsetTensorPtr);

          // Q * K.T
          auto afterQKTensor = createQKBMM(
                          b, h, s_q, s_kv, d, layout, tensorType,
                          &ops, qTensor, kTensor,
                          seqlenMNKTensor, QKVRaggedOffsetTensorPtr);

          // QK.T * attn scale
          auto AfterAttnScale_before_dequan_Q_tensor = createScale(
                          afterQKTensor,  // input tensor
                          "AttnScale",  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          true,  // scale is by value
                          &ops);

          // QK.T * attn scale * dequant_Q
          auto AfterAttnScale_before_dequan_K_tensor = createScale(
                          AfterAttnScale_before_dequan_Q_tensor,  // input tensor
                          "descaleQ",  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops);

          // QK.T * attn scale * dequant_Q * dequant_K
          auto AfterAttnScale_tensor = createScale(
                          AfterAttnScale_before_dequan_K_tensor,  // input tensor
                          "descaleK",  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops);

          auto BeforeDropoutTensor = createSoftmaxForward(
                          b, h, s_q, s_kv, &ops,
                          AfterAttnScale_tensor, isTraining);

          auto AfterDropout_before_quan_S = createDropoutForward(
                          b, h, s_q, s_kv, dropoutProbability,
                          &ops, BeforeDropoutTensor);

          // Amax for S
          createAmax("amaxS", BeforeDropoutTensor, &ops);

          // After softmax * dropout * scale S -> fp8 input to next bmm with V
          auto AfterMultiplyDropout = createScale(
                          AfterDropout_before_quan_S,  // input tensor
                          "scaleS",  // scale tensor
                          tensorType,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops);

          // After softmax * Dropout * V
          auto OTensor_before_dequan_S_tensor = createSVBMM(
                          b, h, s_q, s_kv, d, layout, tensorType,
                          &ops, AfterMultiplyDropout,
                          seqlenMNKTensor, QKVRaggedOffsetTensorPtr);

          // O * dequant_S
          auto OTensor_before_dequan_V_tensor = createScale(
                          OTensor_before_dequan_S_tensor,  // input tensor
                          "descaleS",  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops);

          // O * dequant_S * dequant_V
          auto OTensor_before_quan_O_tensor = createScale(
                          OTensor_before_dequan_V_tensor,  // input tensor
                          "descaleV",  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops);

          // O * dequant_S * dequant_V * scale O
          auto OTensor = createScaleWithOffset(
                          OTensor_before_quan_O_tensor,  // input tensor
                          "scaleO",  // scale tensor
                          layout,  // qkv layout
                          tensorType,  // output tensor type
                          false,  // output not virtual
                          false,  // scale is by value
                          &ops,
                          ORaggedOffsetTensorPtr,  // ragged offset
                          "O");

          // Amax for O
          createAmax("amaxO", OTensor_before_quan_O_tensor, &ops);

          for (unsigned int i = 0; i < ops.size(); i++) {
              all_ops.push_back(&ops[i]);
          }

          // Create an Operation Graph
          auto opGraph = cudnn_frontend::OperationGraphBuilder()
                             .setHandle(handle_)
                             .setOperationGraph(all_ops.size(), all_ops.data())
                             .build();

          cudnn_frontend::EngineConfigList filtered_configs;
          auto statuses = cudnn_frontend::get_heuristics_list<1>(
                          {"heuristics_instant"}, opGraph,
                          allowAllConfig, filtered_configs, true);

          if (filtered_configs.size() == 0) {
              cudnn_frontend::set_error_and_throw_exception(
                      nullptr,
                      CUDNN_STATUS_NOT_SUPPORTED,
                      "run_mha_fprop: No config returned by the heuristics");
          }

          auto plan = cudnn_frontend::ExecutionPlanBuilder()
                  .setHandle(handle_)
                  .setEngineConfig(filtered_configs[0], opGraph.getTag())
                  .build();
          cache.insert({descriptor, plan});
          return plan;
      };  // end of get_plan

      auto plan = get_plan(fa_fprop_cache, descriptor);
      size_t wkspace_size = static_cast<size_t>(plan.getWorkspaceSize());

      // Exit to request upper level API to allocate memory if needed
      if (workspace_ptr == nullptr) {
          *workspace_size = wkspace_size + ((b + 1) * 2 + b) * sizeof(int32_t);
          return;
      }

      // cuDNN stream check needs to be moved here to support dummy kernel calls with
      // null streams for sizing the cuDNN workspace.
      NVTE_CHECK_CUDNN(cudnnSetStream(handle_, stream));

      int32_t* qkv_ragged_offset = reinterpret_cast<int32_t*>(
                  reinterpret_cast<int8_t*>(workspace_ptr) + wkspace_size);
      int32_t* o_ragged_offset = reinterpret_cast<int32_t*>(
                  reinterpret_cast<int8_t*>(workspace_ptr)
                  + wkspace_size + (b + 1) * sizeof(int32_t));
      int32_t* actual_seqlens_q = reinterpret_cast<int32_t*>(
                  reinterpret_cast<int8_t*>(workspace_ptr)
                  + wkspace_size + (b + 1) * 2 * sizeof(int32_t));
      // FP8 currently only supports self-attention, so doesn't use devPtrcuSeqlensKV
      dim3 blockDims(128);
      dim3 gridDims((b + blockDims.x)/blockDims.x);
      cu_seqlens_to_offsets<<<gridDims, blockDims, 0, stream>>>(
                      b, h, d, reinterpret_cast<int32_t*>(devPtrcuSeqlensQ),
                      actual_seqlens_q, qkv_ragged_offset, o_ragged_offset);
      void* devPtrQKVRaggedOffset = reinterpret_cast<void *>(qkv_ragged_offset);
      void* devPtrORaggedOffset = reinterpret_cast<void *>(o_ragged_offset);
      void* devPtrMNKOverride = reinterpret_cast<void *>(actual_seqlens_q);

      float dropoutScale = 1.0f/(1.0f - dropoutProbability);

      std::set<std::pair<uint64_t, void*>> data_ptrs;
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Q"], devPtrQ));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K"], devPtrK));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K_TRANSPOSE"], devPtrK));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["V"], devPtrV));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["AttnScale"], &attnScale));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["DROPOUT_SCALE"], &dropoutScale));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["DROPOUT_SEED"], devPtrDropoutSeed));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["DROPOUT_OFFSET"], devPtrDropoutOffset));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["O"], devPtrO));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaleQ"], devPtrDescaleQ));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaleK"], devPtrDescaleK));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaleV"], devPtrDescaleV));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaleS"], devPtrDescaleS));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["scaleS"], devPtrScaleS));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["scaleO"], devPtrScaleO));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["amaxO"], devPtrAmaxO));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["amaxS"], devPtrAmaxS));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["QKV_RAGGED"], devPtrQKVRaggedOffset));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["O_RAGGED"], devPtrORaggedOffset));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["MNK_OVERRIDE"], devPtrMNKOverride));

      // If training, then we need to write out M and Z_INV
      if (isTraining) {
          data_ptrs.emplace(std::pair<uint64_t, void*>(
                                  tensor_name_to_uid["M"], devPtrM));
          data_ptrs.emplace(std::pair<uint64_t, void*>(
                                  tensor_name_to_uid["Z_INV"], devPtrZInv));
      }

      auto variantPack  = cudnn_frontend::VariantPackBuilder()
                             .setWorkspacePointer(workspace_ptr)
                             .setDataPointers(data_ptrs)
                             .build();

      NVTE_CHECK_CUDNN(cudnnBackendExecute(handle_,
                                           plan.get_raw_desc(),
                                           variantPack.get_raw_desc()));
  } catch (cudnn_frontend::cudnnException& e) {
      struct cudaDeviceProp prop;
      NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

      // This example is only for GH100 cards (cudnn Version >= 8900)
      if (!((prop.major == 9 && prop.minor == 0 && CUDNN_VERSION >= 8900))
                      && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH
                              || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
          std::cout << "Example is only supported for GH100 (cuDNN >= 8900) GPUs" << std::endl;
      }  else {
          std::cout << "[ERROR] Exception " << e.what() << std::endl;
      }
  }
}

// fused attention BWD FP8
void fused_attn_fp8_bwd_impl(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            float attnScale, float dropoutProbability, NVTE_QKV_Layout layout,
            void* devPtrQ, void* devPtrK, void* devPtrV,
            void* devPtrM, void* devPtrZInv,
            void* devPtrO, void* devPtrdO,
            void* devPtrdQ, void* devPtrdK, void* devPtrdV,
            void* devPtrDescaleQ, void* devPtrDescaleK, void* devPtrDescaleV,
            void* devPtrDescaleO, void* devPtrDescaledO,
            void* devPtrDescaleS, void* devPtrDescaledS,
            void* devPtrScaleS, void* devPtrScaledS,
            void* devPtrScaledQ, void* devPtrScaledK, void* devPtrScaledV,
            void* devPtrAmaxdS,
            void* devPtrAmaxdQ, void* devPtrAmaxdK, void* devPtrAmaxdV,
            void* devPtrcuSeqlensQ, void* devPtrcuSeqlensKV,
            void* devPtrDropoutSeed, void* devPtrDropoutOffset,
            cudnnDataType_t tensorType,
            void* workspace_ptr,
            size_t* workspace_size,
            cudaStream_t stream,
            cudnnHandle_t handle_) {
  try {
      FADescriptor descriptor{
              b, h, s_q, s_kv, d,
              attnScale, false, dropoutProbability, layout,
              NVTE_Bias_Type::NVTE_NO_BIAS, NVTE_Mask_Type::NVTE_PADDING_MASK, tensorType, false};

      using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
      static thread_local CacheType fa_bprop_cache;

      // Get plan from cache if cache is available, otherwise create one
      auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {
          // If hit, return
          auto it = cache.find(descriptor);
          if (it != cache.end()) {
            auto plan = it->second;
            return plan;
          }

          // Otherwise, build the op_graph and the plan. Then update cache
          std::vector<cudnn_frontend::Operation const*> all_ops;
          std::vector<cudnn_frontend::Operation> ops;

          NVTE_CHECK(dropoutProbability != 1.0f,
                     "Dropout probability cannot be 1.0");

          int64_t raggedDim[4] =  {b + 1, 1, 1, 1};
          int64_t raggedStride[4] = {1, 1, 1, 1};
          // Create offset tensors
          auto QKVOffsetTensor = tensor_create(
                          CUDNN_DATA_INT32, tensor_name_to_uid["QKV_RAGGED"],
                          raggedDim, raggedStride, false, false);
          auto ORaggedOffsetTensor = tensor_create(
                          CUDNN_DATA_INT32, tensor_name_to_uid["O_RAGGED"],
                          raggedDim, raggedStride, false, false);

          // Create shared ptrs to ragged offset tensors for multiple tensors
          std::shared_ptr<cudnn_frontend::Tensor> QKVRaggedOffsetTensorPtr =
                  std::make_shared<cudnn_frontend::Tensor>(std::move(QKVOffsetTensor));
          std::shared_ptr<cudnn_frontend::Tensor> ORaggedOffsetTensorPtr =
                  std::make_shared<cudnn_frontend::Tensor>(std::move(ORaggedOffsetTensor));

          // Create Q and K tensors that are used in different places
          int64_t q_dim[4] = {b, h, s_q, d};
          int64_t q_stride[4];
          generateMatrixStrides(b, h, s_q, s_kv, d, q_stride, layout,
                          NVTE_QKV_Matrix::NVTE_Q_Matrix);

          int64_t k_dim[4] =  {b, h, s_kv, d};
          int64_t k_stride[4];
          generateMatrixStrides(b, h, s_q, s_kv, d, k_stride, layout,
                          NVTE_QKV_Matrix::NVTE_K_Matrix);

          auto qTensor = tensor_create_with_offset(
                          tensorType, tensor_name_to_uid["Q"],
                          q_dim, q_stride, false, false, QKVRaggedOffsetTensorPtr);
          auto kTensor = tensor_create_with_offset(
                          tensorType, tensor_name_to_uid["K"],
                          k_dim, k_stride, false, false, QKVRaggedOffsetTensorPtr);

          int64_t scale_dim[4] = {1, 1, 1, 1};
          int64_t scale_stride[4] = {1, 1, 1, 1};

          // Create attnScale tensor for multiple ops to use
          auto attnScaleTensor = tensor_create(
                          CUDNN_DATA_FLOAT, tensor_name_to_uid["AttnScale"],
                          scale_dim, scale_stride, false, true);  // is by value

          // Create descale Q K dO dS global tensors since they are used in multiple places
          auto descaleQTensor = tensor_create(
                          CUDNN_DATA_FLOAT, tensor_name_to_uid["descaleQ"],
                          scale_dim, scale_stride, false, false);
          auto descaleKTensor = tensor_create(
                          CUDNN_DATA_FLOAT, tensor_name_to_uid["descaleK"],
                          scale_dim, scale_stride, false, false);
          auto descaledOTensor = tensor_create(
                          CUDNN_DATA_FLOAT, tensor_name_to_uid["descaledO"],
                          scale_dim, scale_stride, false, false);
          auto descaledSTensor = tensor_create(
                          CUDNN_DATA_FLOAT, tensor_name_to_uid["descaledS"],
                          scale_dim, scale_stride, false, false);

          int64_t seqlen_dim[4] =  {b, 1, 1, 1};
          int64_t seqlen_stride[4] = {1, 1, 1, 1};
          // Create MNK override tensor
          auto seqlenMNKTensor = tensor_create(
                          CUDNN_DATA_INT32, tensor_name_to_uid["MNK_OVERRIDE"],
                          seqlen_dim, seqlen_stride, false, false);

          int64_t O_dim[4] =  {b, h, s_q, d};
          int64_t O_stride[4];
          generateMatrixStrides(b, h, s_q, s_kv, d, O_stride, layout,
                          NVTE_QKV_Matrix::NVTE_O_Matrix);
          // Create O and loss tensor
          auto OTensor = tensor_create_with_offset(
                          tensorType, tensor_name_to_uid["O"],
                          O_dim, O_stride, false, false, ORaggedOffsetTensorPtr);
          // dO is used in multiple places and E5M2
          auto dOTensor = tensor_create_with_offset(
                          CUDNN_DATA_FP8_E5M2, tensor_name_to_uid["dO"],
                          O_dim, O_stride, false, false, ORaggedOffsetTensorPtr);

          // Q * K.T
          auto afterQKTensor = createQKBMM(
                          b, h, s_q, s_kv, d, layout, tensorType,
                          &ops, qTensor, kTensor,
                          seqlenMNKTensor, QKVRaggedOffsetTensorPtr);

          // QK.T * attn scale
          auto AfterAttnScale_before_dequan_Q_tensor = createScale(
                          afterQKTensor,  // input tensor
                          attnScaleTensor,  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          true,  // scale is by value
                          &ops,
                          1999  /*UID offset*/);

          // QK.T * attn scale * dequant_Q
          auto AfterAttnScale_before_dequan_K_tensor = createScale(
                          AfterAttnScale_before_dequan_Q_tensor,  // input tensor
                          descaleQTensor,  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops,
                          2000  /*UID offset*/);

          // QK.T * attn scale * dequant_Q * dequant_K
          auto AfterAttnScale_tensor = createScale(
                          AfterAttnScale_before_dequan_K_tensor,  // input tensor
                          descaleKTensor,  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops,
                          2001  /*UID offset*/);

          auto beforeDropout_QKt_Tensor = createSoftmaxBackward(
                          b, h, s_q, s_kv, &ops, AfterAttnScale_tensor);

          int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
          int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

          // mask for the dropout. Used in different places
          auto dropoutMaskTensor = tensor_create(
                          CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 200,
                          afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual

          auto AfterDropout_before_quan_S = createDropoutBackward(
                          b, h, s_q, s_kv, dropoutProbability,
                          &ops, beforeDropout_QKt_Tensor, dropoutMaskTensor);

          // After softmax * scale S -> fp8 input to next bmm with V
          auto AfterMultiply = createScale(
                          AfterDropout_before_quan_S,  // input tensor
                          "scaleS",  // scale tensor
                          tensorType,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops);

          // After softmax * dO
          auto dVTensor_before_dequan_S = createSdOBMM(
                          b, h, s_q, s_kv, d, tensorType,
                          &ops, AfterMultiply, dOTensor, seqlenMNKTensor);

          // O * dequant_S
          auto dVTensor_before_dequan_dO = createScale(
                          dVTensor_before_dequan_S,  // input tensor
                          "descaleS",  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops);

          // O * dequant_S * dequant_dO
          auto dVTensor_before_quan_dV = createScale(
                          dVTensor_before_dequan_dO,  // input tensor
                          descaledOTensor,  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops,
                          2002  /*UID offset*/);

          // O * dequant_S * dequant_dO * scale dV
          auto dVTensor = createScaleWithOffset(
                          dVTensor_before_quan_dV,  // input tensor
                          "scaledV",  // scale tensor
                          layout,  // qkv layout
                          CUDNN_DATA_FP8_E5M2,  // output tensor type
                          false,  // output not virtual
                          false,  // scale is by value
                          &ops,
                          QKVRaggedOffsetTensorPtr,  // ragged offset
                          "dV"  /*Output tensor name*/);

          // Amax for dV
          createAmax("amaxdV", dVTensor_before_quan_dV, &ops);

          auto dS_before_dequan_dO_Tensor = createdOVBMM(
                          b, h, s_q, s_kv, d, layout, tensorType,
                          &ops, dOTensor, seqlenMNKTensor, QKVRaggedOffsetTensorPtr);

          // dS * dequant_dO
          auto dS_before_dequan_V = createScale(
                          dS_before_dequan_dO_Tensor,  // input tensor
                          descaledOTensor,  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops,
                          2003  /*UID offset*/);

          // O * dequant_S * dequant_dV
          auto dS_after_dequan = createScale(
                          dS_before_dequan_V,  // input tensor
                          "descaleV",  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops);

          // RNG Multiply
          auto beforeDropoutScale_dOVt_Tensor = tensor_create(
                          CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 350,
                          afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual
          // After dropout mask and scale
          auto dS_after_dropout = tensor_create(
                          CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 351,
                          afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual

          // Define the multiply mask descriptor
          auto mulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

          // Create a multiply mask Node
          auto maskMul_op = binary_pw_op_create(
                          dS_after_dequan, dropoutMaskTensor,
                          beforeDropoutScale_dOVt_Tensor, mulDesc);

          ops.push_back(std::move(maskMul_op));

          // scale after dropout for dO and O chain
          auto dropoutScale_dOVt_OdO_Tensor = tensor_create(
                          tensorType, tensor_name_to_uid["DROPOUT_SCALE_dOVt_OdO"],
                          scale_dim, scale_stride, false, true);  // is by value

          // Create a multiply dropout scale Node
          auto mul_dropout_scale_op = binary_pw_op_create(
                          beforeDropoutScale_dOVt_Tensor,
                          dropoutScale_dOVt_OdO_Tensor,
                          dS_after_dropout, mulDesc);

          ops.push_back(std::move(mul_dropout_scale_op));

          // O * dequant_O
          auto O_after_dequan_Tensor = createScale(OTensor,  // input tensor
                                          "descaleO",  // scale tensor
                                          CUDNN_DATA_FLOAT,  // output tensor type
                                          true,  // output is virtual
                                          false,  // scale is by value
                                          &ops);

          // dO * dequant_dO
          auto dO_after_dequan_Tensor = createScale(dOTensor,  // input tensor
                                          descaledOTensor,  // scale tensor
                                          CUDNN_DATA_FLOAT,  // output tensor type
                                          true,  // output is virtual
                                          false,  // scale is by value
                                          &ops,
                                          2004  /*UID offset*/);

          // row reduction sum[(dO * dequant_dO) * (O * dequant_O) * (1 - p)]
          auto O_dO_after_rowsum = createdOAndORowReductionChain(
                          b, h, s_q, s_kv, d, layout,
                          &ops, O_after_dequan_Tensor,
                          dO_after_dequan_Tensor, dropoutScale_dOVt_OdO_Tensor);

          // (dS_after_dropout - O_dO_after_rowsum) * AfterDropout_before_quan_S * attnScale
          auto S_mul_dS_minus_O_dO = createBiasSubtractionSoftmaxMulChain(
              b, h, s_q, s_kv, d, layout,
              &ops, dS_after_dropout,
              AfterDropout_before_quan_S, O_dO_after_rowsum,
              attnScaleTensor);


          // S_mul_dS_minus_O_dO * scaledS
          auto S_mul_dS_minus_O_dO_after_quan_dS = createScale(
                          S_mul_dS_minus_O_dO,  // input tensor
                          "scaledS",  // scale tensor
                          CUDNN_DATA_FP8_E5M2,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops);

          // Amax for dS
          createAmax("amaxdS", S_mul_dS_minus_O_dO, &ops);

          // dS @ K
          auto After_dS_K = createdSKBMM(
                          b, h, s_q, s_kv, d, &ops,
                          S_mul_dS_minus_O_dO_after_quan_dS,
                          kTensor, seqlenMNKTensor);

          // (dS * K) * descale dS
          auto After_dS_K_before_dequan_K = createScale(
                          After_dS_K,  // input tensor
                          descaledSTensor,  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops,
                          2006  /*UID offset*/);

          // (dS * K) * descale dS * descale K
          auto After_dS_K_before_quan_dQ = createScale(
                          After_dS_K_before_dequan_K,  // input tensor
                          descaleKTensor,  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops,
                          2007  /*UID offset*/);

          // (dS * K) * descale dS * descale K * scale dQ
          auto dQ = createScaleWithOffset(
                          After_dS_K_before_quan_dQ,  // input tensor
                          "scaledQ",  // scale tensor
                          layout,  // qkv layout
                          CUDNN_DATA_FP8_E5M2,  // output tensor type
                          false,  // output not virtual
                          false,  // scale is by value
                          &ops,
                          QKVRaggedOffsetTensorPtr,  // ragged offset
                          "dQ");

          // Amax for dQ
          createAmax("amaxdQ", After_dS_K_before_quan_dQ, &ops);

          // dS.T @ Q
          auto After_dSTranspose_Q = createdSQBMM(
                          b, h, s_q, s_kv, d, layout, &ops,
                          S_mul_dS_minus_O_dO_after_quan_dS,
                          qTensor, seqlenMNKTensor);

          // (dS.T * Q) * descale dS
          auto After_dSTranspose_Q_before_dequan_Q = createScale(
                          After_dSTranspose_Q,  // input tensor
                          descaledSTensor,  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops,
                          2009  /*UID offset*/);

          // (dS.T * Q) * descale dS * descale Q
          auto After_dSTranspose_Q_before_quan_dK = createScale(
                          After_dSTranspose_Q_before_dequan_Q,  // input tensor
                          descaleQTensor,  // scale tensor
                          CUDNN_DATA_FLOAT,  // output tensor type
                          true,  // output is virtual
                          false,  // scale is by value
                          &ops,
                          2010  /*UID offset*/);

          // (dS.T * Q) * descale dS * descale Q * scale dK
          auto dK = createScaleWithOffset(
                          After_dSTranspose_Q_before_quan_dK,  // input tensor
                          "scaledK",  // scale tensor
                          layout,  // qkv layout
                          CUDNN_DATA_FP8_E5M2,  // output tensor type
                          false,  // output not virtual
                          false,  // scale is by value
                          &ops,
                          QKVRaggedOffsetTensorPtr,  // ragged offset
                          "dK");

          // Amax for dK
          createAmax("amaxdK", After_dSTranspose_Q_before_quan_dK, &ops);

          for (unsigned int i = 0; i < ops.size(); i++) {
              all_ops.push_back(&ops[i]);
          }

          // Create an Operation Graph
          auto opGraph = cudnn_frontend::OperationGraphBuilder()
                             .setHandle(handle_)
                             .setOperationGraph(all_ops.size(), all_ops.data())
                             .build();

          cudnn_frontend::EngineConfigList filtered_configs;
          auto statuses = cudnn_frontend::get_heuristics_list<1>(
                          {"heuristics_instant"}, opGraph,
                          allowAllConfig, filtered_configs, true);

          if (filtered_configs.size() == 0) {
              cudnn_frontend::set_error_and_throw_exception(
                      nullptr,
                      CUDNN_STATUS_NOT_SUPPORTED,
                      "run_mha_bprop: No config returned by the heuristics");
          }

          auto plan = cudnn_frontend::ExecutionPlanBuilder()
                  .setHandle(handle_)
                  .setEngineConfig(filtered_configs[0], opGraph.getTag())
                  .build();
          cache.insert({descriptor, plan});
          return plan;
      };

      auto plan = get_plan(fa_bprop_cache, descriptor);
      size_t wkspace_size = static_cast<size_t>(plan.getWorkspaceSize());

      // Exit to request upper level API to allocate memory if needed
      if (workspace_ptr == nullptr) {
          *workspace_size = wkspace_size + ((b + 1) * 2 + b) * sizeof(int32_t);
          return;
      }

      // cuDNN stream check needs to be moved here to support dummy kernel calls with
      // null streams for sizing the cuDNN workspace.
      NVTE_CHECK_CUDNN(cudnnSetStream(handle_, stream));

      int32_t* qkv_ragged_offset = reinterpret_cast<int32_t*>(
                  reinterpret_cast<int8_t*>(workspace_ptr) + wkspace_size);
      int32_t* o_ragged_offset = reinterpret_cast<int32_t*>(
                  reinterpret_cast<int8_t*>(workspace_ptr)
                  + wkspace_size + (b + 1) * sizeof(int32_t));
      int32_t* actual_seqlens_q = reinterpret_cast<int32_t*>(
                  reinterpret_cast<int8_t*>(workspace_ptr)
                  + wkspace_size + (b + 1) * 2 * sizeof(int32_t));
      // FP8 currently only supports self-attention, so doesn't use devPtrcuSeqlensKV
      dim3 blockDims(128);
      dim3 gridDims((b + blockDims.x)/blockDims.x);
      cu_seqlens_to_offsets<<<gridDims, blockDims, 0, stream>>>(
                      b, h, d, reinterpret_cast<int32_t*>(devPtrcuSeqlensQ),
                      actual_seqlens_q, qkv_ragged_offset, o_ragged_offset);
      void* devPtrQKVRaggedOffset = reinterpret_cast<void *>(qkv_ragged_offset);
      void* devPtrORaggedOffset = reinterpret_cast<void *>(o_ragged_offset);
      void* devPtrMNKOverride = reinterpret_cast<void *>(actual_seqlens_q);

      std::set<std::pair<uint64_t, void*>> data_ptrs;
      float dropoutScale = 1.0f/(1.0f - dropoutProbability);
      float dropoutScale_dOVt_OdO = 1.0f - dropoutProbability;
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Q"], devPtrQ));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K"], devPtrK));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["K_TRANSPOSE"], devPtrK));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["V"], devPtrV));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["V_TRANSPOSE"], devPtrV));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dQ"], devPtrdQ));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dK"], devPtrdK));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dV"], devPtrdV));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dO"], devPtrdO));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["AttnScale"], &attnScale));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["DROPOUT_SCALE"], &dropoutScale));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["DROPOUT_SCALE_dOVt_OdO"],
                              &dropoutScale_dOVt_OdO));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["DROPOUT_SEED"], devPtrDropoutSeed));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["DROPOUT_OFFSET"], devPtrDropoutOffset));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["M"], devPtrM));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Z_INV"], devPtrZInv));
      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["O"], devPtrO));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaleQ"], devPtrDescaleQ));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaleK"], devPtrDescaleK));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaleV"], devPtrDescaleV));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaleS"], devPtrDescaleS));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaledS"], devPtrDescaledS));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaleO"], devPtrDescaleO));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["descaledO"], devPtrDescaledO));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["scaleS"], devPtrScaleS));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["scaledS"], devPtrScaledS));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["scaledQ"], devPtrScaledQ));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["scaledK"], devPtrScaledK));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["scaledV"], devPtrScaledV));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["amaxdS"], devPtrAmaxdS));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["amaxdQ"], devPtrAmaxdQ));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["amaxdK"], devPtrAmaxdK));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["amaxdV"], devPtrAmaxdV));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["QKV_RAGGED"], devPtrQKVRaggedOffset));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["O_RAGGED"], devPtrORaggedOffset));
      data_ptrs.emplace(std::pair<uint64_t, void*>(
                              tensor_name_to_uid["MNK_OVERRIDE"], devPtrMNKOverride));

      auto variantPack  = cudnn_frontend::VariantPackBuilder()
                             .setWorkspacePointer(workspace_ptr)
                             .setDataPointers(data_ptrs)
                             .build();
      NVTE_CHECK_CUDNN(cudnnBackendExecute(handle_,
                                           plan.get_raw_desc(),
                                           variantPack.get_raw_desc()));
  } catch (cudnn_frontend::cudnnException& e) {
      struct cudaDeviceProp prop;
      NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

      // This example is only for GH100 cards (cudnn Version >= 8900)
      if (!((prop.major == 9 && prop.minor == 0 && CUDNN_VERSION >= 8900))
                      && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH
                              || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
          std::cout << "Example is only supported for GH100 (cuDNN >= 8900) GPUs" << std::endl;
      }  else {
          std::cout << "[ERROR] Exception " << e.what() << std::endl;
      }
  }
}

#endif

}  // namespace fused_attn

#if (CUDNN_VERSION >= 8900)
// fused attention FWD FP8 with packed QKV
void fused_attn_fp8_fwd_qkvpacked(
            size_t b, size_t h, size_t max_seqlen, size_t d,
            bool is_training, float attn_scale,
            float p_dropout, NVTE_QKV_Layout qkv_layout,
            const Tensor *input_QKV,
            Tensor *input_output_S,
            Tensor *output_O,
            NVTETensorPack* Aux_CTX_Tensors,
            const Tensor *cu_seqlens,
            const Tensor *rng_state,
            Tensor *workspace,
            cudaStream_t stream,
            cudnnHandle_t handle) {
  using namespace transformer_engine;
  // QKV shape is [total_seqs, 3, h, d]
  void* devPtrQKV = input_QKV->data.dptr;
  void* devPtrQ = reinterpret_cast<void *>(devPtrQKV);
  void* devPtrK = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrQKV) + h * d);
  void* devPtrV = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrQKV) + 2 * h * d);
  void* devPtrDescaleQ = input_QKV->scale_inv.dptr;
  void* devPtrDescaleK = input_QKV->scale_inv.dptr;
  void* devPtrDescaleV = input_QKV->scale_inv.dptr;

  void* devPtrO = output_O->data.dptr;
  void* devPtrAmaxO = output_O->amax.dptr;
  void* devPtrScaleO = output_O->scale.dptr;

  void* devPtrM = nullptr;
  void* devPtrZInv = nullptr;
  if (Aux_CTX_Tensors->size == 0) {
    if (is_training) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_M = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
      Tensor *output_ZInv = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
      Tensor *output_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
      output_M->data.dptr = nullptr;
      output_M->data.shape = {b, h, max_seqlen, 1};
      output_M->data.dtype = DType::kFloat32;
      output_ZInv->data.dptr = nullptr;
      output_ZInv->data.shape = {b, h, max_seqlen, 1};
      output_ZInv->data.dtype = DType::kFloat32;
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_M = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
    Tensor *output_ZInv = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
    Tensor *output_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
    devPtrM = output_M->data.dptr;
    devPtrZInv = output_ZInv->data.dptr;
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void* devPtrAmaxS = input_output_S->amax.dptr;
  void* devPtrScaleS = input_output_S->scale.dptr;
  void* devPtrDescaleS = input_output_S->scale_inv.dptr;

  void* devPtrcuSeqlens = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens->data.dptr));
  void* devPtrDropoutSeed = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_QKV->data.dtype;
  size_t workspace_size = 0;

  fused_attn::fused_attn_fp8_fwd_impl(
                  b, h, max_seqlen, max_seqlen, d,
                  is_training, attn_scale, p_dropout, qkv_layout,
                  devPtrQ, devPtrK, devPtrV,
                  devPtrM, devPtrZInv,
                  devPtrO,
                  devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
                  devPtrDescaleS, devPtrScaleS, devPtrScaleO,
                  devPtrAmaxO, devPtrAmaxS,
                  devPtrcuSeqlens, devPtrcuSeqlens,
                  devPtrDropoutSeed, devPtrDropoutOffset,
                  get_cudnn_dtype(QKV_type),
                  workspace->data.dptr, &workspace_size, stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = { workspace_size };
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = { 1 };
    workspace->data.dtype = DType::kByte;
    return;
  }
}
// fused attention BWD FP8 with packed QKV
void fused_attn_fp8_bwd_qkvpacked(
            size_t b, size_t h, size_t max_seqlen, size_t d,
            float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
            const Tensor *input_QKV,
            const Tensor *input_O,
            const Tensor *input_dO,
            const Tensor *input_M,
            const Tensor *input_ZInv,
            const Tensor *input_S,
            Tensor *input_output_dP,
            const Tensor *output_dQKV,
            const Tensor *cu_seqlens,
            const Tensor *rng_state,
            Tensor *workspace,
            cudaStream_t stream,
            cudnnHandle_t handle) {
  using namespace transformer_engine;
  // QKV shape is [total_seqs, 3, h, d]
  void* devPtrQKV = input_QKV->data.dptr;
  void* devPtrQ = reinterpret_cast<void *>(devPtrQKV);
  void* devPtrK = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrQKV) + h * d);
  void* devPtrV = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrQKV) + 2 * h * d);
  void* devPtrDescaleQ = input_QKV->scale_inv.dptr;
  void* devPtrDescaleK = input_QKV->scale_inv.dptr;
  void* devPtrDescaleV = input_QKV->scale_inv.dptr;

  void* devPtrO = input_O->data.dptr;
  void* devPtrDescaleO = input_O->scale_inv.dptr;
  void* devPtrdO = input_dO->data.dptr;
  void* devPtrDescaledO = input_dO->scale_inv.dptr;

  void* devPtrM = input_M->data.dptr;
  void* devPtrZInv = input_ZInv->data.dptr;

  void* devPtrScaleS = input_S->scale.dptr;
  void* devPtrDescaleS = input_S->scale_inv.dptr;
  void* devPtrAmaxdS = input_output_dP->amax.dptr;
  void* devPtrScaledS = input_output_dP->scale.dptr;
  void* devPtrDescaledS = input_output_dP->scale_inv.dptr;

  // dQKV shape is [total_seqs, 3, h, d]
  void* devPtrdQKV = output_dQKV->data.dptr;
  void* devPtrdQ = reinterpret_cast<void *>(devPtrdQKV);
  void* devPtrdK = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrdQKV) + h * d);
  void* devPtrdV = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrdQKV) + 2 * h * d);
  void* devPtrAmaxdQ = output_dQKV->amax.dptr;
  void* devPtrAmaxdK = output_dQKV->amax.dptr;
  void* devPtrAmaxdV = output_dQKV->amax.dptr;
  void* devPtrScaledQ = output_dQKV->scale.dptr;
  void* devPtrScaledK = output_dQKV->scale.dptr;
  void* devPtrScaledV = output_dQKV->scale.dptr;

  void* devPtrcuSeqlens = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens->data.dptr));
  void* devPtrDropoutSeed = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_QKV->data.dtype;
  size_t workspace_size = 0;

  fused_attn::fused_attn_fp8_bwd_impl(
                  b, h, max_seqlen, max_seqlen, d,
                  attn_scale, p_dropout, qkv_layout,
                  devPtrQ, devPtrK, devPtrV,
                  devPtrM, devPtrZInv,
                  devPtrO, devPtrdO,
                  devPtrdQ, devPtrdK, devPtrdV,
                  devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
                  devPtrDescaleO, devPtrDescaledO,
                  devPtrDescaleS, devPtrDescaledS,
                  devPtrScaleS, devPtrScaledS,
                  devPtrScaledQ, devPtrScaledK, devPtrScaledV,
                  devPtrAmaxdS,
                  devPtrAmaxdQ, devPtrAmaxdK, devPtrAmaxdV,
                  devPtrcuSeqlens, devPtrcuSeqlens,
                  devPtrDropoutSeed, devPtrDropoutOffset,
                  get_cudnn_dtype(QKV_type),
                  workspace->data.dptr, &workspace_size, stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = { workspace_size };
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = { 1 };
    workspace->data.dtype = DType::kByte;
    return;
  }
}
// fused attention FWD FP8 with separate Q, K, V
void fused_attn_fp8_fwd(
            size_t b, size_t h, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
            bool is_training, float attn_scale,
            float p_dropout, NVTE_QKV_Layout qkv_layout,
            const Tensor *input_Q,
            const Tensor *input_K,
            const Tensor *input_V,
            Tensor *input_output_S,
            Tensor *output_O,
            NVTETensorPack* Aux_CTX_Tensors,
            const Tensor *cu_seqlens_q,
            const Tensor *cu_seqlens_kv,
            const Tensor *rng_state,
            Tensor *workspace,
            cudaStream_t stream,
            cudnnHandle_t handle) {
  using namespace transformer_engine;
  void* devPtrQ = input_Q->data.dptr;
  void* devPtrK = input_K->data.dptr;
  void* devPtrV = input_V->data.dptr;
  void* devPtrDescaleQ = input_Q->scale_inv.dptr;
  void* devPtrDescaleK = input_Q->scale_inv.dptr;
  void* devPtrDescaleV = input_Q->scale_inv.dptr;

  void* devPtrO = output_O->data.dptr;
  void* devPtrAmaxO = output_O->amax.dptr;
  void* devPtrScaleO = output_O->scale.dptr;

  void* devPtrM = nullptr;
  void* devPtrZInv = nullptr;
  if (Aux_CTX_Tensors->size == 0) {
    if (is_training) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_M = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
      Tensor *output_ZInv = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
      Tensor *output_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
      output_M->data.dptr = nullptr;
      output_M->data.shape = {b, h, max_seqlen_q, 1};
      output_M->data.dtype = DType::kFloat32;
      output_ZInv->data.dptr = nullptr;
      output_ZInv->data.shape = {b, h, max_seqlen_q, 1};
      output_ZInv->data.dtype = DType::kFloat32;
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_M = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
    Tensor *output_ZInv = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
    Tensor *output_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
    devPtrM = output_M->data.dptr;
    devPtrZInv = output_ZInv->data.dptr;
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void* devPtrAmaxS = input_output_S->amax.dptr;
  void* devPtrScaleS = input_output_S->scale.dptr;
  void* devPtrDescaleS = input_output_S->scale_inv.dptr;

  void* devPtrcuSeqlensQ = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens_q->data.dptr));
  void* devPtrcuSeqlensKV = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens_kv->data.dptr));
  void* devPtrDropoutSeed = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_Q->data.dtype;
  size_t workspace_size = 0;

  fused_attn::fused_attn_fp8_fwd_impl(
                  b, h, max_seqlen_q, max_seqlen_kv, d,
                  is_training, attn_scale, p_dropout, qkv_layout,
                  devPtrQ, devPtrK, devPtrV,
                  devPtrM, devPtrZInv,
                  devPtrO,
                  devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
                  devPtrDescaleS, devPtrScaleS, devPtrScaleO,
                  devPtrAmaxO, devPtrAmaxS,
                  devPtrcuSeqlensQ, devPtrcuSeqlensKV,
                  devPtrDropoutSeed, devPtrDropoutOffset,
                  get_cudnn_dtype(QKV_type),
                  workspace->data.dptr, &workspace_size, stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = { workspace_size };
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = { 1 };
    workspace->data.dtype = DType::kByte;
    return;
  }
}
// fused attention BWD FP8 with separate Q, K, V
void fused_attn_fp8_bwd(
            size_t b, size_t h, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
            float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
            const Tensor *input_Q,
            const Tensor *input_K,
            const Tensor *input_V,
            const Tensor *input_O,
            const Tensor *input_dO,
            const Tensor *input_M,
            const Tensor *input_ZInv,
            const Tensor *input_S,
            Tensor *input_output_dP,
            const Tensor *output_dQ,
            const Tensor *output_dK,
            const Tensor *output_dV,
            const Tensor *cu_seqlens_q,
            const Tensor *cu_seqlens_kv,
            const Tensor *rng_state,
            Tensor *workspace,
            cudaStream_t stream,
            cudnnHandle_t handle) {
  using namespace transformer_engine;
  void* devPtrQ = input_Q->data.dptr;
  void* devPtrK = input_K->data.dptr;
  void* devPtrV = input_V->data.dptr;
  void* devPtrDescaleQ = input_Q->scale_inv.dptr;
  void* devPtrDescaleK = input_Q->scale_inv.dptr;
  void* devPtrDescaleV = input_Q->scale_inv.dptr;

  void* devPtrO = input_O->data.dptr;
  void* devPtrDescaleO = input_O->scale_inv.dptr;
  void* devPtrdO = input_dO->data.dptr;
  void* devPtrDescaledO = input_dO->scale_inv.dptr;

  void* devPtrM = input_M->data.dptr;
  void* devPtrZInv = input_ZInv->data.dptr;

  void* devPtrScaleS = input_S->scale.dptr;
  void* devPtrDescaleS = input_S->scale_inv.dptr;
  void* devPtrAmaxdS = input_output_dP->amax.dptr;
  void* devPtrScaledS = input_output_dP->scale.dptr;
  void* devPtrDescaledS = input_output_dP->scale_inv.dptr;

  void* devPtrdQ = output_dQ->data.dptr;
  void* devPtrdK = output_dK->data.dptr;
  void* devPtrdV = output_dV->data.dptr;
  void* devPtrAmaxdQ = output_dQ->amax.dptr;
  void* devPtrAmaxdK = output_dQ->amax.dptr;
  void* devPtrAmaxdV = output_dQ->amax.dptr;
  void* devPtrScaledQ = output_dQ->scale.dptr;
  void* devPtrScaledK = output_dQ->scale.dptr;
  void* devPtrScaledV = output_dQ->scale.dptr;

  void* devPtrcuSeqlensQ = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens_q->data.dptr));
  void* devPtrcuSeqlensKV = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens_kv->data.dptr));
  void* devPtrDropoutSeed = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_Q->data.dtype;
  size_t workspace_size = 0;

  fused_attn::fused_attn_fp8_bwd_impl(
                  b, h, max_seqlen_q, max_seqlen_kv, d,
                  attn_scale, p_dropout, qkv_layout,
                  devPtrQ, devPtrK, devPtrV,
                  devPtrM, devPtrZInv,
                  devPtrO, devPtrdO,
                  devPtrdQ, devPtrdK, devPtrdV,
                  devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
                  devPtrDescaleO, devPtrDescaledO,
                  devPtrDescaleS, devPtrDescaledS,
                  devPtrScaleS, devPtrScaledS,
                  devPtrScaledQ, devPtrScaledK, devPtrScaledV,
                  devPtrAmaxdS,
                  devPtrAmaxdQ, devPtrAmaxdK, devPtrAmaxdV,
                  devPtrcuSeqlensQ, devPtrcuSeqlensKV,
                  devPtrDropoutSeed, devPtrDropoutOffset,
                  get_cudnn_dtype(QKV_type),
                  workspace->data.dptr, &workspace_size, stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = { workspace_size };
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = { 1 };
    workspace->data.dtype = DType::kByte;
    return;
  }
}
#endif  // end of CUDNN>=8900
}  // namespace transformer_engine
