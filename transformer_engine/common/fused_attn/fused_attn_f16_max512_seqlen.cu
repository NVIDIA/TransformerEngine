/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cudnn_frontend.h>

#include <map>
#include <vector>

#include "../common.h"
#include "fused_attn_f16_max512_seqlen.h"
#include "utils.h"

#if (CUDNN_VERSION >= 8901)
#define Q_ID 1
#define K_ID 2
#define V_ID 3
#define O_ID 4
#define S_ID 5
#define B_ID 6
#define DROPOUT_CONST_ID 7
#define S_CONST_ID 8
#define Q_SEQLEN_ID 9
#define K_SEQLEN_ID 10
#define dQ_ID 11
#define dK_ID 12
#define dV_ID 13
#define dO_ID 14
#define MASK_VAL_ID 15
#define dS_ID 16
#define dBias_ID 17
#define DROPOUT_SEED_ID 18
#define DROPOUT_OFFSET_ID 19

#define VIRTUAL_ID 20

namespace transformer_engine {
namespace fused_attn {

static void createScale(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                        NVTE_QKV_Layout layout, cudnnDataType_t tensorType,
                        std::vector<cudnn_frontend::Operation> &ops) {
  // scale
  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  int64_t k_dim[4] = {b, h, d, s_kv};
  int64_t k_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, k_stride, layout,
                        NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose);

  auto scaleTensor =
      tensor_create(tensorType, S_CONST_ID, scale_dim, scale_stride, false, true);  // is by value
  auto kTensor = tensor_create(tensorType, K_ID, k_dim, k_stride, false, false);
  auto afterScaleKTensor =
      tensor_create(tensorType, VIRTUAL_ID, k_dim, k_stride, true, false);  // is virtual

  // Define the scale descriptor
  auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a Scale Node.
  auto scale_op = binary_pw_op_create(kTensor, scaleTensor, afterScaleKTensor, scaleDesc);

  ops.push_back(std::move(scale_op));
}

static cudnn_frontend::Tensor createBMM1(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                                         NVTE_QKV_Layout layout, cudnnDataType_t tensorType,
                                         bool zero_s, std::vector<cudnn_frontend::Operation> &ops) {
  // Creates the necessary tensor descriptors
  int64_t q_dim[4] = {b, h, s_q, d};
  int64_t q_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, q_stride, layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);

  int64_t k_dim[4] = {b, h, d, s_kv};
  int64_t k_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, k_stride, layout,
                        NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose);

  int64_t p_dim[4] = {b, h, s_q, s_kv};
  int64_t p_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, p_stride, layout, NVTE_QKV_Matrix::NVTE_S_Matrix);

  int64_t seqlen_dim[4] = {b, 1, 1, 1};
  int64_t seqlen_stride[4] = {1, 1, 1, 1};

  auto qTensor = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
  auto afterScaleKTensor =
      tensor_create(tensorType, VIRTUAL_ID, k_dim, k_stride, true, false);  // is virtual
  // first GEMM output
  auto pTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 1, p_dim, p_stride, true,
                               false);  // is virtual

  auto seqlenQTensor =
      tensor_create(CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
  auto seqlenKTensor =
      tensor_create(CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);

  // Define the matmul 1 desc
  // set padding value optionally to 0 for writing zeros to S tensor (if not set, old behaviour)
  auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();

  if (zero_s) {
    matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                        .setComputeType(CUDNN_DATA_FLOAT)
                        .setPaddingValue(0.0f)
                        .build();
  }

  // Create a matmul 1 Node
  auto matmul_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                        .setaMatDesc(qTensor)
                        .setbMatDesc(afterScaleKTensor)
                        .setcMatDesc(pTensor)
                        .setmOverrideDesc(seqlenQTensor)
                        .setnOverrideDesc(seqlenKTensor)
                        .setmatmulDesc(matmul_1_Desc)
                        .build();

  ops.push_back(std::move(matmul_op1));

  return pTensor;
}

static cudnn_frontend::Tensor createBias(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                                         NVTE_QKV_Layout layout, cudnnDataType_t tensorType,
                                         std::vector<cudnn_frontend::Operation> &ops,
                                         cudnn_frontend::Tensor const &prevBlockOutputTensor) {
  NVTE_CHECK(ops.size() != 0, "Bias op constructed incorrectly as the first one.");

  int64_t b_dim[4] = {1, h, s_q, s_kv};
  int64_t b_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t afterBias_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBias_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, afterBias_stride, layout,
                        NVTE_QKV_Matrix::NVTE_S_Matrix);

  // bias
  auto bTensor = tensor_create(tensorType, B_ID, b_dim, b_stride, false, false);
  // output
  auto afterBiasTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 50, afterBias_dim,
                                       afterBias_stride, true, false);  // is virtual

  // Define the bias descriptor
  auto biasDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_ADD);

  // Create a Bias Node.
  auto bias_op = binary_pw_op_create(prevBlockOutputTensor, bTensor, afterBiasTensor, biasDesc);

  ops.push_back(std::move(bias_op));

  return afterBiasTensor;
}

static cudnn_frontend::Tensor createMask(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                                         NVTE_QKV_Layout layout, NVTE_Mask_Type mask_type,
                                         cudnnDataType_t tensorType,
                                         std::vector<cudnn_frontend::Operation> &ops,
                                         cudnn_frontend::Tensor const &prevBlockOutputTensor,
                                         bool is_bprop) {
  NVTE_CHECK(ops.size() != 0, "Padding mask constructed incorrectly as the first one.");

  // subtraction output
  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t seqlen_dim[4] = {b, 1, 1, 1};
  int64_t seqlen_stride[4] = {1, 1, 1, 1};

  int64_t maskVal_dim[4] = {1, 1, 1, 1};
  int64_t maskVal_stride[4] = {1, 1, 1, 1};

  // mask value to put in the masked pixels
  auto maskValTensor = tensor_create(CUDNN_DATA_FLOAT, MASK_VAL_ID, maskVal_dim, maskVal_stride,
                                     false, true);  // is by value

  auto seqlenQTensor =
      tensor_create(CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
  auto seqlenKTensor =
      tensor_create(CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
  // gen index row output
  auto rowIndexTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 100, afterBMM1_dim,
                                      afterBMM1_stride, true, false);  // is virtual
  // gen index column output
  auto columnIndexTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 101, afterBMM1_dim,
                                         afterBMM1_stride, true, false);  // is virtual
  // less than row output
  auto lessThanRowTensor =
      tensor_create(CUDNN_DATA_BOOLEAN, VIRTUAL_ID + 102, afterBMM1_dim, afterBMM1_stride, true,
                    false);  // is virtual
                             // less than column output
  auto lessThanColTensor = tensor_create(CUDNN_DATA_BOOLEAN, VIRTUAL_ID + 103, afterBMM1_dim,
                                         afterBMM1_stride, true, false);  // is virtual
  // padding mask (lessthanRow && lessthanCol)
  auto paddingMaskTensor = tensor_create(CUDNN_DATA_BOOLEAN, VIRTUAL_ID + 104, afterBMM1_dim,
                                         afterBMM1_stride, true, false);  // is virtual
  // row >= col check for causal mask
  auto rowGreaterColTensor = tensor_create(CUDNN_DATA_BOOLEAN, VIRTUAL_ID + 105, afterBMM1_dim,
                                           afterBMM1_stride, true, false);  // is virtual
  // create causal mask (padding && row >= col)
  auto causalMaskTensor = tensor_create(CUDNN_DATA_BOOLEAN, VIRTUAL_ID + 106, afterBMM1_dim,
                                        afterBMM1_stride, true, false);  // is virtual

  // output after masking
  int64_t maskOutputTensor_id = VIRTUAL_ID + 107;
  int64_t maskOutputTensor_virtual = true;
  cudnnDataType_t maskOutputTensor_dataType = CUDNN_DATA_FLOAT;
  auto maskOutputTensor_reorderType = cudnn_frontend::TensorReordering_t::NONE;

  if (is_bprop) {
    maskOutputTensor_id = dS_ID;
    maskOutputTensor_virtual = false;
    maskOutputTensor_dataType = tensorType;
    maskOutputTensor_reorderType = cudnn_frontend::TensorReordering_t::F16x16;
  }

  auto maskOutputTensor =
      cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
          .setByValue(false)
          .setDataType(maskOutputTensor_dataType)
          .setVirtual(maskOutputTensor_virtual)
          .setId(maskOutputTensor_id)
          .setReorderType(maskOutputTensor_reorderType)
          .build();

  // Define the gen index for row descriptor
  auto genIndexRowDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_GEN_INDEX)
                             .setAxis(2)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();

  // Create a gen index Node.
  auto genIndexRow_op = unary_pw_op_create(prevBlockOutputTensor, rowIndexTensor, genIndexRowDesc);

  // Define the gen index for row descriptor
  auto genIndexColumnDesc = cudnn_frontend::PointWiseDescBuilder()
                                .setMode(CUDNN_POINTWISE_GEN_INDEX)
                                .setAxis(3)
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .build();

  // Create a gen index Node.
  auto genIndexColumn_op =
      unary_pw_op_create(prevBlockOutputTensor, columnIndexTensor, genIndexColumnDesc);

  // Define the less than comparison for row descriptor
  auto lessThanRowDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_CMP_LT);

  // Create a less than comparison for row Node.
  auto lessThanRow_op =
      binary_pw_op_create(rowIndexTensor, seqlenQTensor, lessThanRowTensor, lessThanRowDesc);

  // Define the less than comparison for column descriptor
  auto lessThanColDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_CMP_LT);

  // Create a less than comparison for col Node.
  auto lessThanCol_op =
      binary_pw_op_create(columnIndexTensor, seqlenKTensor, lessThanColTensor, lessThanColDesc);

  // Define the less than comparison for column descriptor
  auto paddingMaskAndDesc = pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_LOGICAL_AND);

  // Create a and node for combining lessThanRow and lessThanCol
  auto paddingMaskAnd_op = binary_pw_op_create(lessThanRowTensor, lessThanColTensor,
                                               paddingMaskTensor, paddingMaskAndDesc);

  // Define the greater than equal to comparison descriptor
  auto rowGreaterColDesc = pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_CMP_GE);

  // Create a greater than equal to Node.
  auto rowGreaterCol_op = binary_pw_op_create(rowIndexTensor, columnIndexTensor,
                                              rowGreaterColTensor, rowGreaterColDesc);

  // Define the and to create causal mask descriptor
  auto causalMaskAndDesc = pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_LOGICAL_AND);

  // Create a causal Mask Node.
  auto causalMaskAnd_op = binary_pw_op_create(paddingMaskTensor, rowGreaterColTensor,
                                              causalMaskTensor, causalMaskAndDesc);

  /////////////////// Apply the mask //////////////////////////

  auto maskTensor = (mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
                     mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)
                        ? std::move(causalMaskTensor)
                        : std::move(paddingMaskTensor);

  // Define the binary select to perform masking descriptor
  auto maskDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_BINARY_SELECT);

  // Create a binary select Node.
  auto mask_op = ternary_pw_op_create(prevBlockOutputTensor, maskValTensor, maskTensor,
                                      maskOutputTensor, maskDesc);

  ops.push_back(std::move(genIndexRow_op));
  ops.push_back(std::move(genIndexColumn_op));
  ops.push_back(std::move(lessThanRow_op));
  ops.push_back(std::move(lessThanCol_op));
  ops.push_back(std::move(paddingMaskAnd_op));
  if (mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
      mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) {
    ops.push_back(std::move(rowGreaterCol_op));
    ops.push_back(std::move(causalMaskAnd_op));
  }
  ops.push_back(std::move(mask_op));

  return maskOutputTensor;
}

static cudnn_frontend::Tensor createSoftmaxForward(
    int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, NVTE_QKV_Layout layout,
    bool enable_dropout, bool softmax_output_virtual, cudnnDataType_t tensorType,
    std::vector<cudnn_frontend::Operation> &ops,
    cudnn_frontend::Tensor const &prevBlockOutputTensor) {
  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t afterReduction_dim[4] = {b, h, s_q, 1};
  int64_t afterReduction_stride[4] = {h * s_q, s_q, 1, 1};

  cudnnDataType_t softmaxOutputType = enable_dropout ? CUDNN_DATA_FLOAT : tensorType;
  uint64_t softmaxOutputName = softmax_output_virtual ? VIRTUAL_ID + 154 : S_ID;

  // max (x)
  auto afterMaxReductionTensor =
      tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 150, afterReduction_dim, afterReduction_stride,
                    true, false);  // is virtual
  // x - max(x)
  auto afterSubtractionTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 151, afterBMM1_dim,
                                              afterBMM1_stride, true, false);  // is virtual
  // e^(x - max(x))
  auto afterExponentTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 152, afterBMM1_dim,
                                           afterBMM1_stride, true, false);  // is virtual;
  // sum (e^(x - max(x)))
  auto afterAddReductionTensor =
      tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 153, afterReduction_dim, afterReduction_stride,
                    true, false);  // is virtual
  // divide (e/ sum(e))

  auto reorder_type = cudnn_frontend::TensorReordering_t::F16x16;

  auto afterDivisionTensor =
      cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setId(softmaxOutputName)
          .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(softmaxOutputType)
          .setVirtual(softmax_output_virtual)
          .setByValue(false)
          .setReorderType(reorder_type)
          .build();

  // Define the reduction descriptor
  auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                              .build();

  // Create a reduction max Node.
  auto reductionMax_op =
      cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
          .setxDesc(prevBlockOutputTensor)
          .setyDesc(afterMaxReductionTensor)
          .setreductionDesc(reductionMaxDesc)
          .build();

  // Define the subtract descriptor
  auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

  // Create a subtract Node.
  auto subtract_op = binary_pw_op_create(prevBlockOutputTensor, afterMaxReductionTensor,
                                         afterSubtractionTensor, subtractDesc);

  // Define the exponent descriptor
  auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

  // Create a exponent Node.
  auto exponent_op = unary_pw_op_create(afterSubtractionTensor, afterExponentTensor, exponentDesc);

  // Define the reduction descriptor
  auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                              .build();

  // Create a reduction add Node.
  auto reductionAdd_op =
      cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
          .setxDesc(afterExponentTensor)
          .setyDesc(afterAddReductionTensor)
          .setreductionDesc(reductionAddDesc)
          .build();

  // Define the division descriptor
  auto divisionDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_DIV);

  // Create a subtract Node.
  auto division_op = binary_pw_op_create(afterExponentTensor, afterAddReductionTensor,
                                         afterDivisionTensor, divisionDesc);

  ops.push_back(std::move(reductionMax_op));
  ops.push_back(std::move(subtract_op));
  ops.push_back(std::move(exponent_op));
  ops.push_back(std::move(reductionAdd_op));
  ops.push_back(std::move(division_op));

  return afterDivisionTensor;
}

static cudnn_frontend::Tensor createDropout(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
                                            int64_t d, double probability,
                                            cudnnDataType_t tensorType,
                                            std::vector<cudnn_frontend::Operation> &ops,
                                            cudnn_frontend::Tensor const &prevBlockOutputTensor) {
  NVTE_CHECK(ops.size() != 0, "Dropout DAG constructed incorrectly as the first one");

  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  // mask for the dropout
  auto dropoutMaskTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 200, afterBMM1_dim,
                                         afterBMM1_stride, true, false);  // is virtual

  auto reorder_type = cudnn_frontend::TensorReordering_t::F16x16;

  // after dropout tensor
  auto afterDropoutTensor =
      cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setId(S_ID)
          .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(tensorType)
          .setVirtual(false)
          .setByValue(false)
          .setReorderType(reorder_type)
          .build();
  // scale after dropout
  auto scaleDropoutTensor =
      tensor_create(tensorType, DROPOUT_CONST_ID, scale_dim, scale_stride, false,
                    true);  // is by value
  // after Scale
  auto afterScaleTensor = tensor_create(tensorType, VIRTUAL_ID + 201, afterBMM1_dim,
                                        afterBMM1_stride, true, false);  // is virtual

  // Define the reduction descriptor
  auto rngDesc = cudnn_frontend::RngDescBuilder()
                     .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                     .setBernoulliDistProbability(1.0 - probability)
                     .build();

  auto dropoutSeed =
      tensor_create(CUDNN_DATA_INT64, DROPOUT_SEED_ID, scale_dim, scale_stride, false, false);
  auto dropoutOffset =
      tensor_create(CUDNN_DATA_INT64, DROPOUT_OFFSET_ID, scale_dim, scale_stride, false, false);

  // Create a rng Node.
  auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                    .setyDesc(dropoutMaskTensor)
                    .setSeedDesc(dropoutSeed)
                    .setOffsetDesc(dropoutOffset)
                    .setRngDesc(rngDesc)
                    .build();

  // Define the multiply mask descriptor
  auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply mask Node.
  auto maskMul_op = binary_pw_op_create(prevBlockOutputTensor, dropoutMaskTensor,
                                        afterDropoutTensor, maskMulDesc);

  // Define the multiply scale descriptor
  auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply mask Node.
  auto scaleMul_op =
      binary_pw_op_create(afterDropoutTensor, scaleDropoutTensor, afterScaleTensor, scaleMulDesc);

  ops.push_back(std::move(rng_op));
  ops.push_back(std::move(maskMul_op));
  ops.push_back(std::move(scaleMul_op));

  return afterScaleTensor;
}

static void createBMM2(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                       NVTE_QKV_Layout layout, cudnnDataType_t tensorType,
                       std::vector<cudnn_frontend::Operation> &ops,
                       cudnn_frontend::Tensor const &prevBlockOutputTensor) {
  NVTE_CHECK(ops.size() != 0, "BMM2 op constructed incorrectly as the first one");

  int64_t seqlen_dim[4] = {b, 1, 1, 1};
  int64_t seqlen_stride[4] = {1, 1, 1, 1};

  int64_t v_dim[4] = {b, h, s_kv, d};
  int64_t v_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, v_stride, layout, NVTE_QKV_Matrix::NVTE_V_Matrix);

  int64_t o_dim[4] = {b, h, s_q, d};
  int64_t o_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, o_stride, layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

  auto seqlenQTensor =
      tensor_create(CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
  auto seqlenKTensor =
      tensor_create(CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
  auto vTensor = tensor_create(tensorType, V_ID, v_dim, v_stride, false, false);
  // second GEMM output
  auto oTensor = tensor_create(tensorType, O_ID, o_dim, o_stride, false, false);

  // Define the matmul 2 desc
  // set padding value optionally to 0 for writing zeros to O tensor (if not set, old behaviour)
  auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder()
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .setPaddingValue(0.0f)
                           .build();

  // Create a matmul 2 Node
  auto matmul_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                        .setaMatDesc(prevBlockOutputTensor)
                        .setbMatDesc(vTensor)
                        .setcMatDesc(oTensor)
                        .setmOverrideDesc(seqlenQTensor)
                        .setkOverrideDesc(seqlenKTensor)
                        .setmatmulDesc(matmul_2_Desc)
                        .build();

  ops.push_back(std::move(matmul_op2));
}

static cudnn_frontend::Tensor createSoftmaxBackward(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
                                                    int64_t d, NVTE_QKV_Layout layout,
                                                    cudnnDataType_t tensorType,
                                                    std::vector<cudnn_frontend::Operation> &ops,
                                                    cudnn_frontend::Tensor const &yTensor,
                                                    cudnn_frontend::Tensor const &dyTensor) {
  NVTE_CHECK(ops.size() != 0, "Softmax backward constructed incorrectly as the first one");

  int64_t p_dim[4] = {b, h, s_q, s_kv};
  int64_t p_stride[4];
  generateMatrixStrides(b, h, s_q, s_kv, d, p_stride, layout, NVTE_QKV_Matrix::NVTE_S_Matrix);

  int64_t p_reduction_dim[4] = {b, h, s_q, 1};
  int64_t p_reduction_stride[4];

  p_reduction_stride[3] = 1;
  p_reduction_stride[2] = 1;
  p_reduction_stride[1] = s_q;
  p_reduction_stride[0] = s_q * h;

  int64_t const_dim[4] = {1, 1, 1, 1};
  int64_t const_stride[4] = {1, 1, 1, 1};

  // creating all tensors
  auto softmaxScaleTensor =
      tensor_create(CUDNN_DATA_FLOAT, S_CONST_ID, const_dim, const_stride, false, true);
  auto dyMulYTensor =
      tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 250, p_dim, p_stride, true, false);
  auto dxAfterReductionTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 251, p_reduction_dim,
                                              p_reduction_stride, true, false);
  auto dxAfterSubtractionTensor =
      tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 252, p_dim, p_stride, true, false);
  auto dxUnscaleTensor =
      tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 253, p_dim, p_stride, true, false);
  auto dxTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 254, p_dim, p_stride, true, false);

  // creating all ops
  // mul (y * dy)
  auto mul_1_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
  auto mul_1_op = binary_pw_op_create(yTensor, dyTensor, dyMulYTensor, mul_1_desc);

  // reduction add sum (y * dy)
  auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                              .build();

  auto reductionAdd_op =
      cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
          .setxDesc(dyMulYTensor)
          .setyDesc(dxAfterReductionTensor)
          .setreductionDesc(reductionAddDesc)
          .build();

  // subtraction (dy - sum(y * dy))
  auto sub_0_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);
  auto sub_0_op =
      binary_pw_op_create(dyTensor, dxAfterReductionTensor, dxAfterSubtractionTensor, sub_0_desc);

  // mul (y * (dy - sum(y * dy)))
  auto mul_2_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
  auto mul_2_op =
      binary_pw_op_create(yTensor, dxAfterSubtractionTensor, dxUnscaleTensor, mul_2_desc);

  // mul (scale * dx)
  auto mul_3_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
  auto mul_3_op = binary_pw_op_create(dxUnscaleTensor, softmaxScaleTensor, dxTensor, mul_3_desc);

  ops.push_back(std::move(mul_1_op));
  ops.push_back(std::move(reductionAdd_op));
  ops.push_back(std::move(sub_0_op));
  ops.push_back(std::move(mul_2_op));
  ops.push_back(std::move(mul_3_op));

  return dxTensor;
}

void fused_attn_max_512_fwd_impl(
    int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, bool is_training,
    float scaling_factor, float dropout_probability, NVTE_QKV_Layout layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, void *devPtrQ, void *devPtrK, void *devPtrV,
    void *devPtrS, void *devPtrO, void *devPtrBias, void *devPtrCuSeqlenQ, void *devPtrCuSeqlenKV,
    void *devPtrDropoutSeed, void *devPtrDropoutOffset, void *workspace, size_t *workspace_size,
    cudnnDataType_t tensorType, cudaStream_t stream, cudnnHandle_t handle) {
  try {
    FADescriptor descriptor{b,           h,
                            s_q,         s_kv,
                            d,           scaling_factor,
                            is_training, dropout_probability,
                            layout,      bias_type,
                            mask_type,   tensorType,
                            false};

    using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
    static thread_local CacheType fmha_fprop_cache;

    // softmax auxiliary is only used in the training mode
    bool enable_dropout = is_training && (dropout_probability != 0.0f);

    // two conditions that make softmax auxiliary in virtual
    // 1. inference mode (not is_training)
    // 2. dropout enabled: the auxiliary becomes the dropout output
    bool softmax_output_virtual = !is_training || enable_dropout;

    // Get plan from cache if cache is available, otherwise create one
    auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {
      // if hit, return
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        auto plan = it->second;
        return plan;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      std::vector<cudnn_frontend::Operation const *> all_ops;
      std::vector<cudnn_frontend::Operation> ops;

      createScale(b, h, s_q, s_kv, d, layout, tensorType, ops);

      // if bias, we need to memset the S buffer to correctly computate dbias
      // WAR: causal_mask without bias needs memset the S buffer
      // inference mode doesn't need the S auxiliary
      auto zero_s = (bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) ||
                    (mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)) &&
                        is_training;
      std::shared_ptr<cudnn_frontend::Tensor> maskInput;
      auto bmm1_output = createBMM1(b, h, s_q, s_kv, d, layout, tensorType, zero_s, ops);

      NVTE_CHECK(bias_type != NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS,
                 "NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS has not been implemented.");

      if (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) {
        auto bias_output = createBias(b, h, s_q, s_kv, d, layout, tensorType, ops, bmm1_output);
        maskInput = std::make_shared<cudnn_frontend::Tensor>(std::move(bias_output));
      }
      if (bias_type == NVTE_Bias_Type::NVTE_NO_BIAS) {
        maskInput = std::make_shared<cudnn_frontend::Tensor>(std::move(bmm1_output));
      }

      auto mask_output = createMask(b, h, s_q, s_kv, d, layout, mask_type, tensorType, ops,
                                    *maskInput.get(), false);

      NVTE_CHECK(dropout_probability != 1.0f, "Dropout probability cannot be 1.0.");

      auto softmax_output =
          createSoftmaxForward(b, h, s_q, s_kv, d, layout, enable_dropout, softmax_output_virtual,
                               tensorType, ops, mask_output);

      if (enable_dropout) {
        auto dropout_output =
            createDropout(b, h, s_q, s_kv, d, dropout_probability, tensorType, ops, softmax_output);
        createBMM2(b, h, s_q, s_kv, d, layout, tensorType, ops, dropout_output);
      } else {
        createBMM2(b, h, s_q, s_kv, d, layout, tensorType, ops, softmax_output);
      }

      for (unsigned int i = 0; i < ops.size(); i++) {
        all_ops.push_back(&ops[i]);
      }

      // Create an Operation Graph
      auto opGraph = cudnn_frontend::OperationGraphBuilder()
                         .setHandle(handle)
                         .setOperationGraph(all_ops.size(), all_ops.data())
                         .build();

      cudnn_frontend::EngineConfigList filtered_configs;
      auto statuses = cudnn_frontend::get_heuristics_list<1>(
          {"heuristics_instant"}, opGraph, allowAllConfig, filtered_configs, true);

      if (filtered_configs.size() == 0) {
        cudnn_frontend::set_error_and_throw_exception(
            nullptr, CUDNN_STATUS_NOT_SUPPORTED,
            "run_mha_fprop: No config returned by the heuristics");
      }
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                      .setHandle(handle)
                      .setEngineConfig(filtered_configs[0], opGraph.getTag())
                      .build();
      cache.insert({descriptor, plan});
      return plan;
    };

    auto plan = get_plan(fmha_fprop_cache, descriptor);

    auto plan_workspace_size = plan.getWorkspaceSize();

    // Exit to request upper level API to allocate memory if needed
    if (workspace == nullptr) {
      size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
      *workspace_size = plan_workspace_size + actual_seqlen_workspace_size;
      return;
    }

    // cuDNN stream check needs to be moved here to support dummy kernel calls with
    // null streams for sizing the cuDNN workspace.
    NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

    // Prepare actual seqlen
    constexpr size_t nthreads_per_block = 128;
    const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
    void *devActualSeqlenQ = static_cast<int8_t *>(workspace) + plan_workspace_size;
    void *devActualSeqlenK = static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
    cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
        b, static_cast<const int32_t *>(devPtrCuSeqlenQ),
        static_cast<const int32_t *>(devPtrCuSeqlenKV), static_cast<int32_t *>(devActualSeqlenQ),
        static_cast<int32_t *>(devActualSeqlenK));
    NVTE_CHECK_CUDA(cudaGetLastError());

    // change this if you have access to float_min
    float negInfinity = -1.0E+10;
    float scale_dropout = 1 / (1 - dropout_probability);

    std::set<std::pair<uint64_t, void *>> data_ptrs;
    // add all the data pointers to be used in the variant pack
    data_ptrs.insert(std::pair<uint64_t, void *>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_ID, devPtrK));
    data_ptrs.insert(std::pair<uint64_t, void *>(V_ID, devPtrV));
    data_ptrs.insert(std::pair<uint64_t, void *>(Q_SEQLEN_ID, devActualSeqlenQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_SEQLEN_ID, devActualSeqlenK));
    data_ptrs.insert(std::pair<uint64_t, void *>(MASK_VAL_ID, &negInfinity));

    __half half_cast_scaling_factor{scaling_factor};
    __nv_bfloat16 bfloat_cast_scaling_factor{scaling_factor};

    if (tensorType == CUDNN_DATA_FLOAT) {
      data_ptrs.insert(std::pair<uint64_t, void *>(S_CONST_ID, &scaling_factor));
    } else if (tensorType == CUDNN_DATA_HALF) {
      data_ptrs.insert(std::pair<uint64_t, void *>(S_CONST_ID, &half_cast_scaling_factor));
    } else if (tensorType == CUDNN_DATA_BFLOAT16) {
      data_ptrs.insert(std::pair<uint64_t, void *>(S_CONST_ID, &bfloat_cast_scaling_factor));
    } else {
      NVTE_ERROR("Unsupported tensor type.");
    }

    data_ptrs.insert(std::pair<uint64_t, void *>(O_ID, devPtrO));

    if (bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) {
      data_ptrs.insert(std::pair<uint64_t, void *>(B_ID, devPtrBias));
    }

    // if enable_dropout, S is the result after dropout
    // if not enable dropout, S is the result after softmax
    if (enable_dropout || !softmax_output_virtual) {
      data_ptrs.insert(std::pair<uint64_t, void *>(S_ID, devPtrS));
    }

    __half half_cast_scale_dropout{scale_dropout};
    __nv_bfloat16 bfloat16_cast_scale_dropout{scale_dropout};

    if (enable_dropout) {
      // TODO(rewang): make a util func
      if (tensorType == CUDNN_DATA_FLOAT) {
        data_ptrs.insert(std::pair<uint64_t, void *>(DROPOUT_CONST_ID, &scale_dropout));
      } else if (tensorType == CUDNN_DATA_HALF) {
        data_ptrs.insert(std::pair<uint64_t, void *>(DROPOUT_CONST_ID, &half_cast_scale_dropout));
      } else if (tensorType == CUDNN_DATA_BFLOAT16) {
        data_ptrs.insert(
            std::pair<uint64_t, void *>(DROPOUT_CONST_ID, &bfloat16_cast_scale_dropout));
      } else {
        NVTE_ERROR("Unsupported tensor type.");
      }
      data_ptrs.insert(std::pair<uint64_t, void *>(DROPOUT_SEED_ID, devPtrDropoutSeed));
      data_ptrs.insert(std::pair<uint64_t, void *>(DROPOUT_OFFSET_ID, devPtrDropoutOffset));
    }

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace)
                           .setDataPointers(data_ptrs)
                           .build();

    NVTE_CHECK_CUDNN(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
  } catch (cudnn_frontend::cudnnException &e) {
    NVTE_ERROR(e.what());
  }
}

void fused_attn_max_512_bwd_impl(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                                 float scaling_factor, float dropout_probability,
                                 NVTE_QKV_Layout layout, NVTE_Mask_Type mask_type,
                                 NVTE_Bias_Type bias_type, void *devPtrQ, void *devPtrK,
                                 void *devPtrV, void *devPtrS, void *devPtrdQ, void *devPtrdK,
                                 void *devPtrdV, void *devPtrdO, void *devPtrdS, void *devPtrdBias,
                                 void *devPtrCuSeqlenQ, void *devPtrCuSeqlenKV, void *workspace,
                                 size_t *workspace_size, cudnnDataType_t tensorType,
                                 cudaStream_t stream, cudnnHandle_t handle) {
  try {
    FADescriptor descriptor{
        b,      h,         s_q,       s_kv,       d,    scaling_factor, true, dropout_probability,
        layout, bias_type, mask_type, tensorType, false};

    using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
    static thread_local CacheType fmha_bprop_cache;

    auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        return it->second;
      }

      std::vector<cudnn_frontend::Operation const *> all_ops;
      std::vector<cudnn_frontend::Operation> ops;

      // Creates the necessary tensor descriptors
      int64_t q_dim[4] = {b, h, s_q, d};
      int64_t q_stride[4];
      generateMatrixStrides(b, h, s_q, s_kv, d, q_stride, layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);

      int64_t k_dim[4] = {b, h, s_kv, d};
      int64_t k_stride[4];
      generateMatrixStrides(
          b, h, s_q, s_kv, d, k_stride, layout,
          NVTE_QKV_Matrix::NVTE_K_Matrix);  // type is correct as K is not transposed

      int64_t v_dim[4] = {b, h, d, s_kv};
      int64_t v_stride[4];
      generateMatrixStrides(
          b, h, s_q, s_kv, d, v_stride, layout,
          NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose);  // type is correct as V is transposed

      int64_t p_dim[4] = {b, h, s_q, s_kv};
      int64_t p_stride[4];
      generateMatrixStrides(b, h, s_q, s_kv, d, p_stride, layout, NVTE_QKV_Matrix::NVTE_S_Matrix);

      int64_t p_transpose_dim[4] = {b, h, s_kv, s_q};
      int64_t p_transpose_stride[4];
      p_transpose_stride[0] = p_stride[0];
      p_transpose_stride[1] = p_stride[1];
      p_transpose_stride[2] = p_stride[3];
      p_transpose_stride[3] = p_stride[2];

      int64_t o_dim[4] = {b, h, s_q, d};
      int64_t o_stride[4];
      generateMatrixStrides(b, h, s_q, s_kv, d, o_stride, layout, NVTE_QKV_Matrix::NVTE_O_Matrix);

      int64_t seqlen_dim[4] = {b, 1, 1, 1};
      int64_t seqlen_stride[4] = {1, 1, 1, 1};

      int64_t scale_dim[4] = {1, 1, 1, 1};
      int64_t scale_stride[4] = {1, 1, 1, 1};

      // inputs to fprop
      auto qTensor = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
      auto kTensor = tensor_create(tensorType, K_ID, k_dim, k_stride, false, false);
      auto vTensor = tensor_create(tensorType, V_ID, v_dim, v_stride, false, false);
      auto seqlenQTensor =
          tensor_create(CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
      auto seqlenKTensor =
          tensor_create(CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);

      // gradient of the output
      auto doTensor = tensor_create(tensorType, dO_ID, o_dim, o_stride, false, false);

      auto reorder_type = cudnn_frontend::TensorReordering_t::F16x16;

      // activation from fprop
      auto pTensor = cudnn_frontend::TensorBuilder()
                         .setDim(4, p_dim)
                         .setStride(4, p_stride)
                         .setId(S_ID)
                         .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                         .setDataType(tensorType)
                         .setVirtual(false)
                         .setByValue(false)
                         .setReorderType(reorder_type)
                         .build();

      // outputs from bprop
      auto dqTensor = tensor_create(tensorType, dQ_ID, q_dim, q_stride, false, false);
      auto dkTensor = tensor_create(tensorType, dK_ID, k_dim, k_stride, false, false);
      auto dvTensor = tensor_create(tensorType, dV_ID, k_dim, k_stride, false,
                                    false);  // not transposed therefore k_dim and k_stride

      ////////////////////////////////////////////////////////
      // start creating the ops and the intermediate tensors
      auto pReshapeTensor = tensor_create(tensorType, VIRTUAL_ID + 300, p_transpose_dim,
                                          p_transpose_stride, true, false);

      // reshape to perform transpose and make pReshape
      auto reshape_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                            .setxDesc(pTensor)
                            .setyDesc(pReshapeTensor)
                            .build();

      ops.push_back(std::move(reshape_op));

      // scale dropout
      auto dropoutScaleTensor = tensor_create(CUDNN_DATA_FLOAT, DROPOUT_CONST_ID, scale_dim,
                                              scale_stride, false, true);  // is by value
      auto pAfterScaleTensor = tensor_create(tensorType, VIRTUAL_ID + 301, p_transpose_dim,
                                             p_transpose_stride, true, false);

      auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
      auto scaleMul_op =
          binary_pw_op_create(pReshapeTensor, dropoutScaleTensor, pAfterScaleTensor, scaleMulDesc);
      ops.push_back(std::move(scaleMul_op));

      // perform absolute operation to remove the mask bit
      auto pTransposeAfterAbsTensor = tensor_create(tensorType, VIRTUAL_ID + 302, p_transpose_dim,
                                                    p_transpose_stride, true, false);

      auto absDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_ABS);
      auto abs_op = unary_pw_op_create(pAfterScaleTensor, pTransposeAfterAbsTensor, absDesc);
      ops.push_back(std::move(abs_op));

      // matmul to calculate dvTensor
      // set padding value optionally to 0 for writing zeros to dV tensor (if not set, old
      // behaviour)
      auto matmul_0_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();

      auto matmul_op0 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(pTransposeAfterAbsTensor)
                            .setbMatDesc(doTensor)
                            .setcMatDesc(dvTensor)
                            .setmOverrideDesc(seqlenKTensor)
                            .setkOverrideDesc(seqlenQTensor)
                            .setmatmulDesc(matmul_0_Desc)
                            .build();

      ops.push_back(std::move(matmul_op0));

      // matmul to calculate dpTensor
      auto dpTensor =
          tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 303, p_dim, p_stride, true, false);

      auto matmul_1_Desc =
          cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();

      auto matmul_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(doTensor)
                            .setbMatDesc(vTensor)
                            .setcMatDesc(dpTensor)
                            .setmOverrideDesc(seqlenQTensor)
                            .setnOverrideDesc(seqlenKTensor)
                            .setmatmulDesc(matmul_1_Desc)
                            .build();

      ops.push_back(std::move(matmul_op1));

      // mask the values which were dropped in dropout
      auto pAbsTensor = tensor_create(tensorType, VIRTUAL_ID + 304, p_dim, p_stride, true, false);

      auto p_absDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_ABS);
      auto p_abs_op = unary_pw_op_create(pTensor, pAbsTensor, p_absDesc);
      ops.push_back(std::move(p_abs_op));

      // create the dropout mask
      auto zeroTensor = tensor_create(CUDNN_DATA_FLOAT, MASK_VAL_ID, scale_dim, scale_stride, false,
                                      true);  // is by value
      auto dropoutMaskTensor =
          tensor_create(CUDNN_DATA_BOOLEAN, VIRTUAL_ID + 305, p_dim, p_stride, true, false);

      auto greater_than_0_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_CMP_GT);
      auto greater_than_0_op =
          binary_pw_op_create(pTensor, zeroTensor, dropoutMaskTensor, greater_than_0_desc);
      ops.push_back(std::move(greater_than_0_op));

      // scale for the dropout
      auto dpAfterScaleTensor =
          tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 306, p_dim, p_stride, true, false);

      auto mul_0_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
      auto mul_0_op =
          binary_pw_op_create(dpTensor, dropoutScaleTensor, dpAfterScaleTensor, mul_0_desc);
      ops.push_back(std::move(mul_0_op));

      // drop the values based on the dropout mask
      auto dpAfterDropoutTensor =
          tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 307, p_dim, p_stride, true, false);

      auto selection_0_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_BINARY_SELECT);
      auto selection_0_op = ternary_pw_op_create(dpAfterScaleTensor, zeroTensor, dropoutMaskTensor,
                                                 dpAfterDropoutTensor, selection_0_desc);
      ops.push_back(std::move(selection_0_op));

      // softmax backward
      auto dsTensor = createSoftmaxBackward(b, h, s_q, s_kv, d, layout, tensorType, ops, pAbsTensor,
                                            dpAfterDropoutTensor);

      // mask
      auto dsAfterMaskTensor =
          createMask(b, h, s_q, s_kv, d, layout, mask_type, tensorType, ops, dsTensor, true);

      // dbias tensor
      int64_t dbias_dim[4] = {1, h, s_q, s_kv};
      int64_t dbias_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};
      auto dBiasTensor = tensor_create(tensorType, dBias_ID, dbias_dim, dbias_stride, false, false);

      if (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) {
        auto softmaxScaleTensor =
            tensor_create(CUDNN_DATA_FLOAT, S_CONST_ID, scale_dim, scale_stride, false, true);
        auto softmaxScaleReciprocalTensor =
            tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 401, scale_dim, scale_stride, true, false);
        auto dbiasBeforeScaleTensor =
            tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 402, dbias_dim, dbias_stride, true, false);

        // Define the reduction descriptor
        auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                    .setComputeType(CUDNN_DATA_FLOAT)
                                    .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                    .build();

        // Create a reduction add node to compute the dbias
        auto reductionAdd_op =
            cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                .setxDesc(dsAfterMaskTensor)
                .setyDesc(dbiasBeforeScaleTensor)
                .setreductionDesc(reductionAddDesc)
                .build();
        ops.push_back(std::move(reductionAdd_op));

        // take the reciprocal of the scale
        auto reciprocal_scale_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_RECIPROCAL);
        auto reciprocal_scale_op = unary_pw_op_create(
            softmaxScaleTensor, softmaxScaleReciprocalTensor, reciprocal_scale_desc);
        ops.push_back(std::move(reciprocal_scale_op));

        // apply the scale
        auto dBias_scale_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
        auto dBias_scale_op = binary_pw_op_create(
            dbiasBeforeScaleTensor, softmaxScaleReciprocalTensor, dBiasTensor, dBias_scale_desc);
        ops.push_back(std::move(dBias_scale_op));
      }

      // matmul to calculate dqTensor
      // set padding value optionally to 0 for writing zeros to dqTensor (if not set, old
      // behaviour)
      auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();

      auto matmul_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(dsAfterMaskTensor)
                            .setbMatDesc(kTensor)
                            .setcMatDesc(dqTensor)
                            .setmOverrideDesc(seqlenQTensor)
                            .setkOverrideDesc(seqlenKTensor)
                            .setmatmulDesc(matmul_2_Desc)
                            .build();

      ops.push_back(std::move(matmul_op2));

      // reshape for transpose of ds
      auto dsAfterMaskReshapeTensor = tensor_create(tensorType, VIRTUAL_ID + 308, p_transpose_dim,
                                                    p_transpose_stride, true, false);

      auto reshape_2_op =
          cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
              .setxDesc(dsAfterMaskTensor)
              .setyDesc(dsAfterMaskReshapeTensor)
              .build();

      ops.push_back(std::move(reshape_2_op));

      // matmul to calculate dkTensor
      // set padding value optionally to 0 for writing zeros to dktensor (if not set, old
      // behaviour)
      auto matmul_3_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();

      auto matmul_op3 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(dsAfterMaskReshapeTensor)
                            .setbMatDesc(qTensor)
                            .setcMatDesc(dkTensor)
                            .setmOverrideDesc(seqlenKTensor)
                            .setkOverrideDesc(seqlenQTensor)
                            .setmatmulDesc(matmul_3_Desc)
                            .build();

      ops.push_back(std::move(matmul_op3));

      /////////////////////////////////////////////////////////////////

      for (unsigned int i = 0; i < ops.size(); i++) {
        all_ops.push_back(&ops[i]);
      }

      // Create an Operation Graph
      auto opGraph = cudnn_frontend::OperationGraphBuilder()
                         .setHandle(handle)
                         .setOperationGraph(all_ops.size(), all_ops.data())
                         .build();

      cudnn_frontend::EngineConfigList filtered_configs;
      auto statuses = cudnn_frontend::get_heuristics_list<1>(
          {"heuristics_instant"}, opGraph, allowAllConfig, filtered_configs, true);

      if (filtered_configs.size() == 0) {
        cudnn_frontend::set_error_and_throw_exception(
            nullptr, CUDNN_STATUS_NOT_SUPPORTED,
            "run_mha_bprop: No config returned by the heuristics");
      }

      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                      .setHandle(handle)
                      .setEngineConfig(filtered_configs[0], opGraph.getTag())
                      .build();
      cache.insert({descriptor, plan});
      return plan;
    };

    auto plan = get_plan(fmha_bprop_cache, descriptor);

    auto plan_workspace_size = plan.getWorkspaceSize();

    // Exit to request upper level API to allocate memory if needed
    if (workspace == nullptr) {
      size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
      *workspace_size = plan_workspace_size + actual_seqlen_workspace_size;
      return;
    }

    // cuDNN stream check needs to be moved here to support dummy kernel calls with
    // null streams for sizing the cuDNN workspace.
    NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

    constexpr size_t nthreads_per_block = 128;
    const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
    void *devActualSeqlenQ = static_cast<int8_t *>(workspace) + plan_workspace_size;
    void *devActualSeqlenK = static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
    cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
        b, static_cast<const int32_t *>(devPtrCuSeqlenQ),
        static_cast<const int32_t *>(devPtrCuSeqlenKV), static_cast<int32_t *>(devActualSeqlenQ),
        static_cast<int32_t *>(devActualSeqlenK));
    NVTE_CHECK_CUDA(cudaGetLastError());

    std::set<std::pair<uint64_t, void *>> data_ptrs;
    // add all the data pointers to be used in the variant pack
    data_ptrs.insert(std::pair<uint64_t, void *>(dQ_ID, devPtrdQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(dK_ID, devPtrdK));
    data_ptrs.insert(std::pair<uint64_t, void *>(dV_ID, devPtrdV));

    data_ptrs.insert(std::pair<uint64_t, void *>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_ID, devPtrK));
    data_ptrs.insert(std::pair<uint64_t, void *>(V_ID, devPtrV));
    data_ptrs.insert(std::pair<uint64_t, void *>(S_ID, devPtrS));
    data_ptrs.insert(std::pair<uint64_t, void *>(dO_ID, devPtrdO));
    data_ptrs.insert(std::pair<uint64_t, void *>(dS_ID, devPtrdS));
    data_ptrs.insert(std::pair<uint64_t, void *>(Q_SEQLEN_ID, devActualSeqlenQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_SEQLEN_ID, devActualSeqlenK));

    if (bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) {
      data_ptrs.insert(std::pair<uint64_t, void *>(dBias_ID, devPtrdBias));
    }

    float zeroVal = 0.0f;
    float dropoutScale = 1.0f / (1.0f - dropout_probability);

    data_ptrs.insert(std::pair<uint64_t, void *>(DROPOUT_CONST_ID, &dropoutScale));
    data_ptrs.insert(std::pair<uint64_t, void *>(S_CONST_ID, &scaling_factor));
    data_ptrs.insert(std::pair<uint64_t, void *>(MASK_VAL_ID, &zeroVal));

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace)
                           .setDataPointers(data_ptrs)
                           .build();

    NVTE_CHECK_CUDNN(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
  } catch (cudnn_frontend::cudnnException &e) {
    NVTE_ERROR(e.what());
  }
}

}  // namespace fused_attn

using namespace transformer_engine::fused_attn;
void fused_attn_max_512_fwd_qkvpacked(
    size_t batch, size_t num_head, size_t max_seqlen, size_t head_dim, bool is_training,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, const Tensor *input_QKV, const Tensor *input_Bias, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens, const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  // QKV shape is [b, s, 3, h, d]
  void *devPtrQKV = input_QKV->data.dptr;
  const auto stride = 2 * num_head * head_dim;

  void *devPtrQ = static_cast<void *>(devPtrQKV);
  void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);

  void *devPtrBias = static_cast<void *>(input_Bias->data.dptr);

  void *devPtrO = output_O->data.dptr;

  void *devPtrS = nullptr;

  if (Aux_CTX_Tensors->size == 0) {
    Aux_CTX_Tensors->size = 1;
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    output_S->data.dptr = nullptr;
    output_S->data.shape = {batch, num_head, max_seqlen, max_seqlen};
    output_S->data.dtype = input_QKV->data.dtype;
  } else if (Aux_CTX_Tensors->size == 1) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void *devPtrCuSeqlen = cu_seqlens->data.dptr;

  const DType rng_state_type = rng_state->data.dtype;
  NVTE_CHECK(rng_state_type == DType::kInt64);
  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      static_cast<void *>(static_cast<uint64_t *>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_QKV->data.dtype;
  size_t workspace_size = 0;

  fused_attn_max_512_fwd_impl(
      batch, num_head, max_seqlen, max_seqlen, head_dim, is_training, attn_scale, p_dropout,
      qkv_layout, bias_type, mask_type, devPtrQ, devPtrK, devPtrV, devPtrS, devPtrO, devPtrBias,
      devPtrCuSeqlen, devPtrCuSeqlen, devPtrDropoutSeed, devPtrDropoutOffset, workspace->data.dptr,
      &workspace_size, get_cudnn_dtype(QKV_type), stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_ERROR("Unexpected workspace_size.");
  }
}

void fused_attn_max_512_fwd_kvpacked(size_t batch, size_t num_head, size_t q_max_seqlen,
                                     size_t kv_max_seqlen, size_t head_dim, bool is_training,
                                     float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                                     NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                     const Tensor *input_Q, const Tensor *input_KV,
                                     const Tensor *input_Bias, Tensor *output_O,
                                     NVTETensorPack *Aux_CTX_Tensors, const Tensor *q_cu_seqlens,
                                     const Tensor *kv_cu_seqlens, const Tensor *rng_state,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  NVTE_CHECK(bias_type == NVTE_Bias_Type::NVTE_NO_BIAS ||
                 bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS,
             "NVTE_PRE_SCALE_BIAS is not implemented in fused_attn_max_512.");

  // Q shape is [b, s, h, d]
  void *devPtrQ = input_Q->data.dptr;

  // KV shape is [b, s, 2, h, d]
  const auto stride = 2 * num_head * head_dim;
  void *devPtrK = input_KV->data.dptr;
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrK) + stride);

  void *devPtrBias = input_Bias->data.dptr;

  void *devPtrO = output_O->data.dptr;

  void *devPtrS = nullptr;

  const DType q_type = input_Q->data.dtype;
  const DType kv_type = input_KV->data.dtype;
  NVTE_CHECK(q_type == kv_type, "data type of Q must be equal to data type of KV.");

  if (Aux_CTX_Tensors->size == 0) {
    Aux_CTX_Tensors->size = 1;
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    output_S->data.dptr = nullptr;
    output_S->data.shape = {batch, num_head, q_max_seqlen, kv_max_seqlen};
    output_S->data.dtype = q_type;
  } else if (Aux_CTX_Tensors->size == 1) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void *devQCuSeqlen = q_cu_seqlens->data.dptr;
  void *devKVCuSeqlen = kv_cu_seqlens->data.dptr;

  const DType rng_state_type = rng_state->data.dtype;
  NVTE_CHECK(rng_state_type == DType::kInt64);
  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      static_cast<void *>(static_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_max_512_fwd_impl(
      batch, num_head, q_max_seqlen, kv_max_seqlen, head_dim, is_training, attn_scale, p_dropout,
      qkv_layout, bias_type, mask_type, devPtrQ, devPtrK, devPtrV, devPtrS, devPtrO, devPtrBias,
      devQCuSeqlen, devKVCuSeqlen, devPtrDropoutSeed, devPtrDropoutOffset, workspace->data.dptr,
      &workspace_size, get_cudnn_dtype(q_type), stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_ERROR("Unexpected workspace_size.");
  }
}
void fused_attn_max_512_fwd(size_t batch, size_t num_head, size_t q_max_seqlen,
                            size_t kv_max_seqlen, size_t head_dim, bool is_training,
                            float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                            NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                            const Tensor *input_Q, const Tensor *input_K, const Tensor *input_V,
                            const Tensor *input_Bias, Tensor *output_O,
                            NVTETensorPack *Aux_CTX_Tensors, const Tensor *q_cu_seqlens,
                            const Tensor *kv_cu_seqlens, const Tensor *rng_state, Tensor *workspace,
                            cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  void *devPtrQ = input_Q->data.dptr;
  void *devPtrK = input_K->data.dptr;
  void *devPtrV = input_V->data.dptr;

  void *devPtrBias = input_Bias->data.dptr;

  void *devPtrO = output_O->data.dptr;

  void *devPtrS = nullptr;

  const DType q_type = input_Q->data.dtype;
  const DType kv_type = input_K->data.dtype;
  NVTE_CHECK(q_type == kv_type, "data type of Q must be equal to data type of KV.");

  if (Aux_CTX_Tensors->size == 0) {
    Aux_CTX_Tensors->size = 1;
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    output_S->data.dptr = nullptr;
    output_S->data.shape = {batch, num_head, q_max_seqlen, kv_max_seqlen};
    output_S->data.dtype = q_type;
  } else if (Aux_CTX_Tensors->size == 1) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void *devQCuSeqlen = q_cu_seqlens->data.dptr;
  void *devKVCuSeqlen = kv_cu_seqlens->data.dptr;

  const DType rng_state_type = rng_state->data.dtype;
  NVTE_CHECK(rng_state_type == DType::kInt64);
  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      static_cast<void *>(static_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_max_512_fwd_impl(
      batch, num_head, q_max_seqlen, kv_max_seqlen, head_dim, is_training, attn_scale, p_dropout,
      qkv_layout, bias_type, mask_type, devPtrQ, devPtrK, devPtrV, devPtrS, devPtrO, devPtrBias,
      devQCuSeqlen, devKVCuSeqlen, devPtrDropoutSeed, devPtrDropoutOffset, workspace->data.dptr,
      &workspace_size, get_cudnn_dtype(q_type), stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_ERROR("Unexpected workspace_size.");
  }
}

void fused_attn_max_512_bwd_qkvpacked(size_t batch, size_t num_head, size_t max_seqlen,
                                      size_t head_dim, float attn_scale, float p_dropout,
                                      NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                      NVTE_Mask_Type mask_type, const Tensor *input_QKV,
                                      const Tensor *input_dO, Tensor *output_S, Tensor *output_dQKV,
                                      Tensor *output_dBias, const Tensor *cu_seqlens,
                                      Tensor *workspace, cudaStream_t stream,
                                      cudnnHandle_t handle) {
  using namespace transformer_engine;

  // QKV shape is [b, s, 3, h, d]
  void *devPtrQKV = input_QKV->data.dptr;

  auto stride = 2 * num_head * head_dim;
  void *devPtrQ = devPtrQKV;
  void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);

  void *devPtrdO = input_dO->data.dptr;

  // dQKV shape is [b, s, 3, h, d]
  void *devPtrdQKV = output_dQKV->data.dptr;
  void *devPtrdQ = devPtrdQKV;
  void *devPtrdK = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + stride);
  void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + 2 * stride);

  void *devPtrdBias = output_dBias->data.dptr;

  void *devPtrS = output_S->data.dptr;

  // devPtrdS reuses the memory of devPtrS
  void *devPtrdS = devPtrS;

  void *devPtrCuSeqlens = cu_seqlens->data.dptr;

  const auto qkv_type = input_QKV->data.dtype;
  size_t workspace_size = 0;

  fused_attn_max_512_bwd_impl(batch, num_head, max_seqlen, max_seqlen, head_dim, attn_scale,
                              p_dropout, qkv_layout, mask_type, bias_type, devPtrQ, devPtrK,
                              devPtrV, devPtrS, devPtrdQ, devPtrdK, devPtrdV, devPtrdO, devPtrdS,
                              devPtrdBias, devPtrCuSeqlens, devPtrCuSeqlens, workspace->data.dptr,
                              &workspace_size, get_cudnn_dtype(qkv_type), stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_ERROR("Unexpected workspace_size.");
  }
}

void fused_attn_max_512_bwd_kvpacked(size_t batch, size_t num_head, size_t q_max_seqlen,
                                     size_t kv_max_seqlen, size_t head_dim, float attn_scale,
                                     float p_dropout, NVTE_QKV_Layout qkv_layout,
                                     NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                     const Tensor *input_Q, const Tensor *input_KV,
                                     const Tensor *input_dO, Tensor *output_S, Tensor *output_dQ,
                                     Tensor *output_dKV, Tensor *output_dBias,
                                     const Tensor *q_cu_seqlens, const Tensor *kv_cu_seqlens,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  // Q shape is [b, s, h, d]
  // KV shape is [b, s, 2, h, d]
  auto stride = 2 * num_head * head_dim;
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrK = input_KV->data.dptr;
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrK) + stride);

  void *devPtrdO = input_dO->data.dptr;

  // dQ shape is [b, s, h, d]
  // dKV shape is [b, s, 2, h, d]
  void *devPtrdQ = output_dQ->data.dptr;
  void *devPtrdK = output_dKV->data.dptr;
  void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdK) + stride);

  void *devPtrdBias = output_dBias->data.dptr;

  void *devPtrS = output_S->data.dptr;

  // devPtrdS reuses the memory of devPtrS
  void *devPtrdS = devPtrS;

  void *devPtrQCuSeqlens = q_cu_seqlens->data.dptr;
  void *devPtrKVCuSeqlens = kv_cu_seqlens->data.dptr;

  const auto q_type = input_Q->data.dtype;
  const auto kv_type = input_KV->data.dtype;
  NVTE_CHECK(q_type == kv_type, "data type of Q must be equal to data type of KV.");
  size_t workspace_size = 0;

  fused_attn_max_512_bwd_impl(
      batch, num_head, q_max_seqlen, kv_max_seqlen, head_dim, attn_scale, p_dropout, qkv_layout,
      mask_type, bias_type, devPtrQ, devPtrK, devPtrV, devPtrS, devPtrdQ, devPtrdK, devPtrdV,
      devPtrdO, devPtrdS, devPtrdBias, devPtrQCuSeqlens, devPtrKVCuSeqlens, workspace->data.dptr,
      &workspace_size, get_cudnn_dtype(q_type), stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_ERROR("Unexpected workspace_size.");
  }
}
void fused_attn_max_512_bwd(size_t batch, size_t num_head, size_t q_max_seqlen,
                            size_t kv_max_seqlen, size_t head_dim, float attn_scale,
                            float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                            NVTE_Mask_Type mask_type, const Tensor *input_Q, const Tensor *input_K,
                            const Tensor *input_V, const Tensor *input_dO, Tensor *output_S,
                            Tensor *output_dQ, Tensor *output_dK, Tensor *output_dV,
                            Tensor *output_dBias, const Tensor *q_cu_seqlens,
                            const Tensor *kv_cu_seqlens, Tensor *workspace, cudaStream_t stream,
                            cudnnHandle_t handle) {
  using namespace transformer_engine;

  void *devPtrQ = input_Q->data.dptr;
  void *devPtrK = input_K->data.dptr;
  void *devPtrV = input_V->data.dptr;

  void *devPtrdO = input_dO->data.dptr;

  void *devPtrdQ = output_dQ->data.dptr;
  void *devPtrdK = output_dK->data.dptr;
  void *devPtrdV = output_dV->data.dptr;

  void *devPtrdBias = output_dBias->data.dptr;

  void *devPtrS = output_S->data.dptr;

  // devPtrdS reuses the memory of devPtrS
  void *devPtrdS = devPtrS;

  void *devPtrQCuSeqlens = q_cu_seqlens->data.dptr;
  void *devPtrKVCuSeqlens = kv_cu_seqlens->data.dptr;

  const auto q_type = input_Q->data.dtype;
  const auto kv_type = input_K->data.dtype;
  NVTE_CHECK(q_type == kv_type, "data type of Q must be equal to data type of KV.");
  size_t workspace_size = 0;

  fused_attn_max_512_bwd_impl(
      batch, num_head, q_max_seqlen, kv_max_seqlen, head_dim, attn_scale, p_dropout, qkv_layout,
      mask_type, bias_type, devPtrQ, devPtrK, devPtrV, devPtrS, devPtrdQ, devPtrdK, devPtrdV,
      devPtrdO, devPtrdS, devPtrdBias, devPtrQCuSeqlens, devPtrKVCuSeqlens, workspace->data.dptr,
      &workspace_size, get_cudnn_dtype(q_type), stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_ERROR("Unexpected workspace_size.");
  }
}
}  // namespace transformer_engine
#endif  // CUDNN_VERSION >= 8901
