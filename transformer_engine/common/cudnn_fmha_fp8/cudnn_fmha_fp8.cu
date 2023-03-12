/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//#include <transformer_engine/transformer_engine.h>
//#include "transformer_engine/helpers.h"
//#include "transformer_engine/error_util.h"


#include "transformer_engine/cudnn_fmha_fp8.h"
#include <cudnn_frontend.h>
#include "../common.h"

namespace transformer_engine {
namespace cudnn_fmha {

using namespace transformer_engine;

void generateMHAStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, int64_t* strideA, MHA_Layout layout, MHA_Matrix matrix) {
    CUDNN_FRONTEND_UNUSED(b);
    constexpr int batch_dim_idx   = 0;
    constexpr int head_dim_idx    = 1;
    constexpr int seqlen_dim_idx  = 2;
    constexpr int hidden_dim_idx  = 3;

    constexpr int seqlen_transpose_dim_idx = 3;
    constexpr int hidden_transpose_dim_idx = 2;

    constexpr int seqlen_q_dim_idx = 2;
    constexpr int seqlen_kv_dim_idx = 3;

    switch (matrix)
    {
        case MHA_Matrix::Q_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_q * 3 * h * d;
            } else {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_q * h * d;
            }
            break;
        case MHA_Matrix::K_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else {
                strideA[seqlen_transpose_dim_idx] = h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::V_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 2* h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::S_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = s_kv;
            strideA[head_dim_idx] = s_q * s_kv;
            strideA[batch_dim_idx] = h * s_q * s_kv;
            break;
        case MHA_Matrix::O_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = h * d;
            strideA[head_dim_idx] = d;
            strideA[batch_dim_idx] = s_q * h * d;
            break;
    }
}

//#if (CUDNN_VERSION >= 8900)
std::unordered_map<std::string, int> tensor_name_to_uid = {
    {"Q",                 1},
    {"K",                 2},
    {"V",                 3},
    {"O",                 4},
    {"S",                 5},
    {"B",                 6},
    {"D_CONST",           7},
    {"S_CONST",           8},
    {"Q_SEQLEN",          9},
    {"K_SEQLEN",         10},
    {"dQ",               11},
    {"dK",               12},
    {"dV",               13},
    {"dO",               14},
    {"MASK_VAL",         15},
    {"dS",               16},
    {"O_SEQLEN",         17},
    {"M",                18},
    {"Z",                19},
    {"descaleQ",         20},
    {"descaleK",         21}, 
    {"descaleV",         22},
    {"descaleS",         23},
    {"scaleS",           24},
    {"amaxS",            25},
    {"amaxO",            26},
    {"QKV_RAGGED",       27},
    {"O_RAGGED",         28},
    {"K_TRANSPOSE",      29},
    {"AttnScale",        30},
    {"scaleO",           31},
    {"ZInv",             32},
    {"VIRTUAL",          40}
};

bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

static cudnn_frontend::Tensor tensor_create(cudnnDataType_t type, int64_t id, int64_t const * dim, 
                                int64_t const * stride, bool is_virtual, bool is_value) {
    int nbDims = 4;
    auto tensor_created = cudnn_frontend::TensorBuilder()
            .setDim(nbDims, dim)
            .setStride(nbDims, stride)
            .setId(id) 
            .setAlignment(16) // 16B alignment is needed to run a tensor core engine
            .setDataType(type)
            .setVirtual(is_virtual)
            .setByValue(is_value)
            .build();
    std::cout << tensor_created.describe() << std::endl;
    return tensor_created;
};

static cudnn_frontend::Tensor tensor_create_with_offset(cudnnDataType_t type, int64_t id, int64_t const * dim, 
                                int64_t const * stride, bool is_virtual, bool is_value, std::shared_ptr<cudnn_frontend::Tensor>& raggedOffset) {
    int nbDims = 4;
    auto tensor_created = cudnn_frontend::TensorBuilder()
            .setDim(nbDims, dim)
            .setStride(nbDims, stride)
            .setId(id) 
            .setAlignment(16) // 16B alignment is needed to run a tensor core engine
            .setDataType(type)
            .setVirtual(is_virtual)
            .setByValue(is_value)
            .setRaggedOffset(raggedOffset)
            .build();
    std::cout << tensor_created.describe() << std::endl;
    return tensor_created;
};

static cudnn_frontend::PointWiseDesc pw_desc_create(cudnnDataType_t type, cudnnPointwiseMode_t mode) {
    auto pw_desc_created = cudnn_frontend::PointWiseDescBuilder()
            .setMode(mode)
            .setComputeType(type)
            .build();
    
    std::cout << pw_desc_created.describe() << std::endl;
    return pw_desc_created;
} 

static cudnn_frontend::Operation unary_pw_op_create(cudnn_frontend::Tensor const &xDesc, cudnn_frontend::Tensor const &yDesc, 
                                                    cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(xDesc)
                        .setyDesc(yDesc)
                        .setpwDesc(pwDesc)
                        .build();
    std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Operation binary_pw_op_create(cudnn_frontend::Tensor const &xDesc, cudnn_frontend::Tensor const &bDesc, 
                                                    cudnn_frontend::Tensor const &yDesc, cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(xDesc)
                        .setbDesc(bDesc)
                        .setyDesc(yDesc)
                        .setpwDesc(pwDesc)
                        .build();
    std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Operation ternary_pw_op_create(cudnn_frontend::Tensor const &xDesc, cudnn_frontend::Tensor const &bDesc, cudnn_frontend::Tensor const &tDesc, 
                                            cudnn_frontend::Tensor const &yDesc, cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(xDesc)
                        .setbDesc(bDesc)
                        .settDesc(tDesc)
                        .setyDesc(yDesc)
                        .setpwDesc(pwDesc)
                        .build();
    std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Tensor
createAmax(const std::string& amax_tensor_name,
            cudnn_frontend::Tensor& prevBlockOutputTensor,
            std::vector<cudnn_frontend::Operation>& ops) {

        // Amax is just a scalar
        int64_t amax_dim [4] = {1, 1, 1, 1};
        int64_t amax_stride [4] = {1, 1, 1, 1};

        auto amaxTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid[amax_tensor_name], amax_dim, amax_stride, false, false);

        // Define the amax descriptor
        auto redunctionDesc = cudnn_frontend::ReductionDescBuilder()
                                  .setMathPrecision(CUDNN_DATA_FLOAT)
                                  .setReductionOp(CUDNN_REDUCE_TENSOR_AMAX)
                                  .build();
        std::cout << redunctionDesc.describe() << std::endl;

        // Create a reduction amax Node.
        auto reduction_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(prevBlockOutputTensor)
                                .setyDesc(amaxTensor)
                                .setreductionDesc(redunctionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;
        ops.push_back(std::move(reduction_op));
        return amaxTensor;
}

static cudnn_frontend::Tensor 
createScale(cudnn_frontend::Tensor& prevBlockOutputTensor,
            const std::string& scale_tensor_name,
            cudnnDataType_t tensorType,
            bool isOutputVirtual,
            bool isScaleByValue,
            std::vector<cudnn_frontend::Operation>& ops,
            const std::string& output_tensor_name ="") {

    // scale
    int64_t scale_dim [4] = {1, 1, 1, 1};
    int64_t scale_stride [4] = {1, 1, 1, 1};

    int64_t k_dim [4];
    int64_t k_stride [4];
    // K dim and stride should be the same as prev block dim and stride
    k_dim[0] = prevBlockOutputTensor.getDim()[0];
    k_dim[1] = prevBlockOutputTensor.getDim()[1];
    k_dim[2] = prevBlockOutputTensor.getDim()[2];
    k_dim[3] = prevBlockOutputTensor.getDim()[3];
    k_stride[0] = prevBlockOutputTensor.getStride()[0];
    k_stride[1] = prevBlockOutputTensor.getStride()[1];
    k_stride[2] = prevBlockOutputTensor.getStride()[2];
    k_stride[3] = prevBlockOutputTensor.getStride()[3];

    auto scaleTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid[scale_tensor_name], scale_dim, scale_stride, false, isScaleByValue); // is by value

    cudnnDataType_t outputDataType = isOutputVirtual ? CUDNN_DATA_FLOAT : tensorType;
    // Hack to get the virtual id to not be same for all the virtual tensors
    int64_t outputUID = isOutputVirtual ? tensor_name_to_uid["VIRTUAL"] + tensor_name_to_uid[scale_tensor_name] + 200 : tensor_name_to_uid[output_tensor_name];
    auto afterScaleKTensor = tensor_create(outputDataType, outputUID, k_dim, k_stride, true, false); // is virtual

    // Define the scale descriptor
    auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a Scale Node.
    auto scale_op = binary_pw_op_create(prevBlockOutputTensor, scaleTensor, afterScaleKTensor, scaleDesc);

    ops.push_back(std::move(scale_op));
    return afterScaleKTensor;
}

static cudnn_frontend::Tensor 
createScaleWithOffset(cudnn_frontend::Tensor& prevBlockOutputTensor,
            const std::string& scale_tensor_name,
            cudnnDataType_t tensorType,
            bool isOutputVirtual,
            bool isScaleByValue,
            std::vector<cudnn_frontend::Operation>& ops,
            std::shared_ptr<cudnn_frontend::Tensor>& offsetTensor,
            const std::string& output_tensor_name ="") {

    // scale
    int64_t scale_dim [4] = {1, 1, 1, 1};
    int64_t scale_stride [4] = {1, 1, 1, 1};

    int64_t k_dim [4];
    int64_t k_stride [4];
    // K dim and stride should be the same as prev block dim and stride
    k_dim[0] = prevBlockOutputTensor.getDim()[0];
    k_dim[1] = prevBlockOutputTensor.getDim()[1];
    k_dim[2] = prevBlockOutputTensor.getDim()[2];
    k_dim[3] = prevBlockOutputTensor.getDim()[3];
    k_stride[0] = prevBlockOutputTensor.getStride()[0];
    k_stride[1] = prevBlockOutputTensor.getStride()[1];
    k_stride[2] = prevBlockOutputTensor.getStride()[2];
    k_stride[3] = prevBlockOutputTensor.getStride()[3];


    auto scaleTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid[scale_tensor_name], scale_dim, scale_stride, false, isScaleByValue); // is by value

    cudnnDataType_t outputDataType = isOutputVirtual ? CUDNN_DATA_FLOAT : tensorType;
    // Hack to get the virtual id to not be same for all the virtual tensors
    int64_t outputUID = isOutputVirtual ? tensor_name_to_uid["VIRTUAL"] + tensor_name_to_uid[scale_tensor_name] + 200 : tensor_name_to_uid[output_tensor_name];
    auto afterScaleKTensor = tensor_create_with_offset(outputDataType, outputUID, k_dim, k_stride, isOutputVirtual, false, offsetTensor); // is virtual

    // Define the scale descriptor
    auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a Scale Node.
    auto scale_op = binary_pw_op_create(prevBlockOutputTensor, scaleTensor, afterScaleKTensor, scaleDesc);

    ops.push_back(std::move(scale_op));
    return afterScaleKTensor;
}

static cudnn_frontend::Tensor
createBMM1(int64_t b, 
           int64_t h, 
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops,
           std::shared_ptr<cudnn_frontend::Tensor>& QKVRaggedOffsetTensor) {
    // Creates the necessary tensor descriptors
    int64_t q_dim [4] = {b, h, s_q, d};
    int64_t q_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

    int64_t k_dim [4] =  {b, h, s_kv, d};
    int64_t k_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, k_stride, layout, MHA_Matrix::K_Matrix);

    int64_t k_transpose_dim [4] = {b, h, d, s_kv};
    int64_t k_transpose_stride [4];
    k_transpose_stride[0] = k_stride[0];
    k_transpose_stride[1] = k_stride[1];
    k_transpose_stride[2] = k_stride[3];
    k_transpose_stride[3] = k_stride[2];

    int64_t s_dim [4] = {b, h, s_q, s_kv};
    int64_t s_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

    int64_t seqlen_dim [4] =  {b, 1, 1, 1};
    int64_t seqlen_stride [4] = {1, 1, 1, 1};


    auto qTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["Q"], q_dim, q_stride, false, false, QKVRaggedOffsetTensor);
    auto kTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["K"], k_dim, k_stride, false, false, QKVRaggedOffsetTensor);
    auto kTransposeTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["K_TRANSPOSE"], k_transpose_dim, k_transpose_stride, false, false, QKVRaggedOffsetTensor); // is virtual


    // first GEMM output
    auto afterQKTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 1, s_dim, s_stride, true, false); // is virtual
    
    auto seqlenQTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["Q_SEQLEN"], seqlen_dim, seqlen_stride, false, false);
    auto seqlenKTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["K_SEQLEN"], seqlen_dim, seqlen_stride, false, false);

    // Define the matmul 1 desc
    auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                                        .setComputeType(CUDNN_DATA_FLOAT)
                                        .setPaddingValue(-2000000) // really large negative number for dead pixels
                                        .build();
    std::cout << matmul_1_Desc.describe() << std::endl;

    // Create reshape node for Q -> Q.T
    auto reshape_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                            .setxDesc(kTensor)
                            .setyDesc(kTransposeTensor)
                            .build();

    std::cout << reshape_op.describe() << std::endl;
    ops.push_back(std::move(reshape_op));

    // Create a matmul 1 Node
    auto matmul_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(qTensor)
                            .setbMatDesc(kTransposeTensor)
                            .setcMatDesc(afterQKTensor)
                            .setmOverrideDesc(seqlenQTensor)
                            .setnOverrideDesc(seqlenKTensor)
                            .setmatmulDesc(matmul_1_Desc)
                            .build();

    std::cout << matmul_op1.describe() << std::endl;

    ops.push_back(std::move(matmul_op1));

    return afterQKTensor;
}

static cudnn_frontend::Tensor
createSoftmaxForward(int64_t b, 
                     int64_t h, 
                     int64_t s_q,
                     int64_t s_kv,
                     int64_t d,
                     MHA_Layout layout,
                     bool softmax_output_virtual,
                     cudnnDataType_t tensorType,
                     std::vector<cudnn_frontend::Operation>& ops,
                     cudnn_frontend::Tensor& prevBlockOutputTensor) {
    CUDNN_FRONTEND_UNUSED(d);
    CUDNN_FRONTEND_UNUSED(layout);

    int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t afterReduction_dim [4] = {b, h, s_q, 1};
    int64_t afterReduction_stride [4] = {h * s_q, s_q, 1, 1};

    cudnnDataType_t softmaxOutputType = (softmax_output_virtual) ? CUDNN_DATA_FLOAT : tensorType;
    uint64_t softmaxOutputUID = softmax_output_virtual ? tensor_name_to_uid["VIRTUAL"] + 154 : tensor_name_to_uid["S"];

    // max (x) (M tensor)
    auto afterMaxReductionTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["M"], afterReduction_dim, afterReduction_stride, false, false); // not virtual
    // x - max(x)
    auto afterSubtractionTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 151, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
    // e^(x - max(x))
    auto afterExponentTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 152, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual;
    // sum (e^(x - max(x))) (Z tensor)
    auto ZTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["Z"], afterReduction_dim, afterReduction_stride, true, false); // is virtual
    // 1 / sum (e^(x - max(x))) (Z_INV tensor)
    auto ZInvTensor = tensor_create(CUDNN_DATA_FLOAT, tensor_name_to_uid["Z_INV"], afterReduction_dim, afterReduction_stride, false, false); // not virtual
    // Final softmax output (After exponent * Z_INV)
    auto AfterDivisionBeforeQuanSTensor = tensor_create(softmaxOutputType, softmaxOutputUID, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // Define the reduction descriptor
    auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                                .build();
    std::cout << reductionMaxDesc.describe() << std::endl;

    // Create a reduction max Node.
    auto reductionMax_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(prevBlockOutputTensor)
                                .setyDesc(afterMaxReductionTensor)
                                .setreductionDesc(reductionMaxDesc)
                                .build();
    std::cout << reductionMax_op.describe() << std::endl;

    // Define the subtract descriptor
    auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

    // Create a subtract Node.
    auto subtract_op = binary_pw_op_create(prevBlockOutputTensor, afterMaxReductionTensor, afterSubtractionTensor, subtractDesc);

    // Define the exponent descriptor
    auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

    // Create a exponent Node.
    auto exponent_op = unary_pw_op_create(afterSubtractionTensor, afterExponentTensor, exponentDesc);

    // Define the reduction descriptor
    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                .build();
    std::cout << reductionAddDesc.describe() << std::endl;

    // Create a reduction add Node.
    auto reductionAdd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(afterExponentTensor)
                                .setyDesc(ZTensor)
                                .setreductionDesc(reductionAddDesc)
                                .build();

    std::cout << reductionAdd_op.describe() << std::endl;

    // Define the reciprocal descriptor
    auto reciprocalDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_RCP);

    // Create a reciprocal Node.
    auto reciprocal_op = unary_pw_op_create(ZTensor, ZInvTensor, reciprocalDesc);

    // Define the pw multiply descriptor
    auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply Node.
    auto mutliply_op = binary_pw_op_create(afterExponentTensor, ZInvTensor, AfterDivisionBeforeQuanSTensor, multiplyDesc);

    ops.push_back(std::move(reductionMax_op));
    ops.push_back(std::move(subtract_op));
    ops.push_back(std::move(exponent_op));
    ops.push_back(std::move(reductionAdd_op));
    ops.push_back(std::move(reciprocal_op));
    ops.push_back(std::move(mutliply_op));

    return AfterDivisionBeforeQuanSTensor;
}

static cudnn_frontend::Tensor
createBMM2(int64_t b, 
           int64_t h, 
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops,
           const cudnn_frontend::Tensor &prevBlockOutputTensor,
           std::shared_ptr<cudnn_frontend::Tensor>& QKVRaggedOffsetTensor) {

    cudnn_frontend::throw_if(ops.size() == 0, "BMM2 op constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    int64_t seqlen_dim [4] =  {b, 1, 1, 1};
    int64_t seqlen_stride [4] = {1, 1, 1, 1};

    int64_t v_dim [4] =  {b, h, s_kv, d};
    int64_t v_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, v_stride, layout, MHA_Matrix::V_Matrix);

    int64_t o_dim [4] =  {b, h, s_q, d};
    int64_t o_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);
    
    auto seqlenQTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["Q_SEQLEN"], seqlen_dim, seqlen_stride, false, false);
    auto seqlenKTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["O_SEQLEN"], seqlen_dim, seqlen_stride, false, false);
    auto vTensor = tensor_create_with_offset(tensorType, tensor_name_to_uid["V"], v_dim, v_stride, false, false, QKVRaggedOffsetTensor);
    // second GEMM output
    auto oTensor = tensor_create(tensorType, tensor_name_to_uid["VIRTUAL"] + 300, o_dim, o_stride, true, false); // is virtual

    // Define the matmul 2 desc
    auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder()
                                        .setComputeType(CUDNN_DATA_FLOAT)
                                        .setPaddingValue(0) // Padding to 0 for offset override dead pixels
                                        .build();
    std::cout << matmul_2_Desc.describe() << std::endl;

    // Create a matmul 2 Node
    auto matmul_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(prevBlockOutputTensor)
                            .setbMatDesc(vTensor)
                            .setcMatDesc(oTensor)
                            .setmOverrideDesc(seqlenQTensor)
                            .setkOverrideDesc(seqlenKTensor)
                            .setmatmulDesc(matmul_2_Desc)
                            .build();

    std::cout << matmul_op2.describe() << std::endl;

    ops.push_back(std::move(matmul_op2));

    return oTensor;
}

void 
run_fp8_flash_mha_fprop(int64_t b, 
                int64_t h, 
                int64_t s_q,
                int64_t s_kv,
                int64_t d,
                float attnScale,
                MHA_Layout layout,
                void* devPtrQKV, 
                void* devPtrM,
                void* devPtrZInv,  
                void* devPtrO,
                void* devPtrDescaleQ,
                void* devPtrDescaleK,
                void* devPtrDescaleV,
                void* devPtrDescaleS,
                void* devPtrScaleS,
                void* devPtrScaleO,
                void* devPtrAmaxO,
                void* devPtrAmaxS,
                void* devPtrQKVRaggedOffset,
                void* devPtrORaggeDOffset,
                void* devPtrMOverride,
                void* devPtrNOverride,
                void* devPtrKOverride,
                cudnnDataType_t tensorType) {
                
    using namespace cudnn_fmha;
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        //checkCudnnErr(cudnnCreate(&handle_));
        cudnnCreate(&handle_);

        std::vector<cudnn_frontend::Operation const*> all_ops;
        std::vector<cudnn_frontend::Operation> ops;
        std::set<std::pair<uint64_t, void*>> data_ptrs;


        // Ragged tensors have b + 1 elements
        int64_t raggedDim [4] =  {b + 1, 1, 1, 1};
        int64_t raggedStride [4] = {1, 1, 1, 1};
        // Create offset tensors
	printf(">>> QKV_RAGGED \n");
        auto QKVOffsetTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["QKV_RAGGED"], raggedDim, raggedStride, false, false);
	printf(">>> O_RAGGED \n");
        auto ORaggedOffsetTensor = tensor_create(CUDNN_DATA_INT32, tensor_name_to_uid["O_RAGGED"], raggedDim, raggedStride, false, false);

        // Create shared ptrs to ragged offset tensors for multiple tensors to use ragged offset
	printf(">>> make shared x 2 \n");
        std::shared_ptr<cudnn_frontend::Tensor> QKVRaggedOffsetTensorPtr = std::make_shared<cudnn_frontend::Tensor>(std::move(QKVOffsetTensor));
        std::shared_ptr<cudnn_frontend::Tensor> ORaggedOffsetTensorPtr = std::make_shared<cudnn_frontend::Tensor>(std::move(ORaggedOffsetTensor));

	printf(">>> createBMM1 \n");
        auto afterQKTensor = createBMM1(b, h, s_q, s_kv, d, layout, tensorType, ops, QKVRaggedOffsetTensorPtr);

        // QK.T * attn scale
	printf(">>> scale \n");
        auto AfterAttnScale_before_dequan_Q_tensor = createScale(afterQKTensor, // input tensor
                                                                "AttnScale", // scale tensor
                                                                tensorType,  // output tensor type if output is not virtual
                                                                true, // output is virtual
                                                                true, // scale is by value
                                                                ops);

        // QK.T * attn scale * dequant_Q
	printf(">>> scale \n");
        auto AfterAttnScale_before_dequan_K_tensor = createScale(AfterAttnScale_before_dequan_Q_tensor, // input tensor
                                                                "descaleQ", // scale tensor
                                                                tensorType, // output tensor type if output is not virtual
                                                                true, // output is virtual
                                                                false, // scale is by value
                                                                ops);

        // QK.T * attn scale * dequant_Q * dequant_K
	printf(">>> scale \n");
        auto AfterAttnScale_tensor = createScale(AfterAttnScale_before_dequan_K_tensor, // input tensor
                                                "descaleK", // scale tensor
                                                tensorType, // output tensor type if output is not virtual
                                                true, // output is virtual
                                                false, // scale is by value
                                                ops);


        bool softmax_output_virtual = true;
	printf(">>> softmax \n");
        auto AfterDivision_before_quan_S_tensor = createSoftmaxForward(b, h, s_q, s_kv, d, layout, softmax_output_virtual, tensorType, ops, AfterAttnScale_tensor);

        // Amax for S
	printf(">>> amax \n");
        createAmax("amaxS", AfterDivision_before_quan_S_tensor, ops);

        // After softmax * scale S -> fp8 input to next bmm with V
	printf(">>> scale \n");
        auto AfterDivision_tensor = createScale(AfterDivision_before_quan_S_tensor, // input tensor
                                                "scaleS", // scale tensor
                                                tensorType, // output tensor type if output is not virtual
                                                true, // output is virtual
                                                false, // scale is by value
                                                ops);

	printf(">>> BMM2 \n");
        auto OTensor_before_dequan_S_tensor = createBMM2(b, h, s_q, s_kv, d, layout, tensorType, ops, AfterDivision_tensor, QKVRaggedOffsetTensorPtr);

        // O * dequant_S
	printf(">>> scale \n");
        auto OTensor_before_dequan_V_tensor = createScale(OTensor_before_dequan_S_tensor, // input tensor
                                                        "descaleS", // scale tensor
                                                        tensorType, // output tensor type if output is not virtual
                                                        true, // output is virtual
                                                        false, // scale is by value
                                                        ops);

        // O * dequant_S * dequant_V
	printf(">>> scale \n");
        auto OTensor_before_quan_O_tensor = createScale(OTensor_before_dequan_V_tensor, // input tensor
                                                        "descaleV", // scale tensor
                                                        tensorType, // output tensor type if output is not virtual
                                                        true, // output is virtual
                                                        false, // scale is by value
                                                        ops);

        // O * dequant_S * dequant_V * scale O
	printf(">>> scale with offset \n");
        auto OTensor = createScaleWithOffset(OTensor_before_quan_O_tensor, // input tensor
                            "scaleO", // scale tensor
                            tensorType, // output tensor type if output is not virtual
                            false, // output not virtual
                            false, // scale is by value
                            ops,
                            ORaggedOffsetTensorPtr, // ragged offset
                            "O");

        // Amax for O
	printf(">>> amax \n");
        createAmax("amaxO", OTensor_before_quan_O_tensor, ops);

        std::cout << "Total ops created: " << ops.size() << std::endl;

        for (unsigned int i = 0; i < ops.size(); i++) {
            all_ops.push_back(&ops[i]);
        }

        // Create an Operation Graph
        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(all_ops.size(), all_ops.data())
                           .build();


        // Selecting engine 0 for the Flash FMHA kernel
        auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
	printf(">>> engine \n");
        std::cout << engine.describe() << std::endl;
        auto& knobs = engine.getSupportedKnobs();
	printf(">>> knobs \n");
        for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
        }

        if (knobs.begin() != knobs.end()) {
	    printf(">>> knob choice \n");
            std::cout << "Updated knob choice" << std::endl;
            knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
            std::cout << knobs.begin()->describe() << std::endl;
        }
        auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
	printf(">>> engine config \n");
        std::cout << engine_config.describe() << std::endl;
        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
	printf(">>> workspace \n");
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            //checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
            cudaMalloc(&workspace_ptr, workspace_size);
        }

        void* devPtrQ = (void *) devPtrQKV; // q points to the top of qkv
        void* devPtrK = (void *)(static_cast<int8_t*>(devPtrQKV) + h * d); // k is at an offset of h * d
        void* devPtrV = (void *)(static_cast<int8_t*>(devPtrQKV) + 2 * h * d); // v is at an offset of 2 * h * d
        
        // add all the data pointers to be used in the variant pack
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Q"], devPtrQ));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K"], devPtrK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K_TRANSPOSE"], devPtrK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["V"], devPtrV));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["AttnScale"], &attnScale));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["M"], devPtrM));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Z_INV"], devPtrZInv));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["O"], devPtrO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleQ"], devPtrDescaleQ));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleK"], devPtrDescaleK));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleV"], devPtrDescaleV));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["descaleS"], devPtrDescaleS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["scaleS"], devPtrScaleS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["scaleO"], devPtrScaleO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["amaxO"], devPtrAmaxO));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["amaxS"], devPtrAmaxS));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["QKV_RAGGED"], devPtrQKVRaggedOffset));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["O_RAGGED"], devPtrORaggeDOffset));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Q_SEQLEN"], devPtrMOverride));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K_SEQLEN"], devPtrNOverride));
        data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["O_SEQLEN"], devPtrKOverride));

        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(data_ptrs)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
	printf(">>> execution \n");
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            //checkCudaErr(cudaFree(workspace_ptr));
            cudaFree(workspace_ptr);
        }

        //checkCudnnErr(cudnnDestroy(handle_));
        cudnnDestroy(handle_);

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        //checkCudaErrors(cudaGetDeviceProperties( &prop, 0 ));
        cudaGetDeviceProperties( &prop, 0 );
        
        // this example is only for GA100 cards (cudnn Version >= 8700) and GH100 cards (cudnn Version >= 8800)
        if (!((prop.major == 8 && prop.minor == 0) || (prop.major == 9 && prop.minor == 0 && CUDNN_VERSION >= 8800)) && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for GA100 (cuDNN >= 8700) and GH100 (cuDNN >= 8800) GPUs" << std::endl; 
        }  else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            //CHECK(false);
        }
    }
}

} // namespace cudnn_fmha

void fp8_flash_mha_fprop(int64_t b, 
                int64_t h, 
                int64_t s_q,
                int64_t s_kv,
                int64_t d,
                float attnScale,
                MHA_Layout layout,
		const Tensor *inputQKV,
		const Tensor *inputM,
                const Tensor *inputZInv,
                const Tensor *inputO,
                const Tensor *inputDescaleQ,
                const Tensor *inputDescaleK,
                const Tensor *inputDescaleV,
                const Tensor *inputDescaleS,
                const Tensor *inputScaleS,
                const Tensor *inputScaleO,
                const Tensor *inputAmaxS,
                const Tensor *inputAmaxO,
                const Tensor *inputQKVRaggedOffset,
                const Tensor *inputORaggedOffset,
		const Tensor *inputActualSeqlenQ,
		const Tensor *inputActualSeqlenK,
		const Tensor *inputActualSeqlenO,
                void* workspace,
                size_t workspaceSize,
                cudaStream_t stream
) {
  void* devPtrQKV = inputQKV->data.dptr;
  void* devPtrM = inputM->data.dptr;
  void* devPtrZInv = inputZInv->data.dptr;
  void* devPtrO = inputO->data.dptr;
  void* devPtrDescaleQ = inputDescaleQ->data.dptr;
  void* devPtrDescaleK = inputDescaleK->data.dptr;
  void* devPtrDescaleV = inputDescaleV->data.dptr;
  void* devPtrDescaleS = inputDescaleS->data.dptr;
  void* devPtrScaleS = inputScaleS->data.dptr;
  void* devPtrScaleO = inputScaleO->data.dptr;
  void* devPtrAmaxS = inputAmaxS->data.dptr;
  void* devPtrAmaxO = inputAmaxO->data.dptr;
  void* devPtrQKVRaggedOffset = inputQKVRaggedOffset->data.dptr;
  void* devPtrORaggedOffset = inputORaggedOffset->data.dptr;
  void* devPtrActualSeqlenQ = inputActualSeqlenQ->data.dptr;
  void* devPtrActualSeqlenK = inputActualSeqlenK->data.dptr;
  void* devPtrActualSeqlenO = inputActualSeqlenO->data.dptr;

  cudnn_fmha::run_fp8_flash_mha_fprop(b, h, s_q, s_kv, d, attnScale, layout,
                devPtrQKV, 
                devPtrM,
                devPtrZInv,  
                devPtrO,
                devPtrDescaleQ,
                devPtrDescaleK,
                devPtrDescaleV,
                devPtrDescaleS,
                devPtrScaleS,
                devPtrScaleO,
                devPtrAmaxO,
                devPtrAmaxS,
                devPtrQKVRaggedOffset,
                devPtrORaggedOffset,
                devPtrActualSeqlenQ,
                devPtrActualSeqlenK,
                devPtrActualSeqlenO,
                CUDNN_DATA_FP8_E4M3);

}

} // namespace transformer_engine

void nvte_cudnn_fmha_fp8_fwd(const NVTETensor QKV,
		const NVTETensor M,
                const NVTETensor ZInv,
                const NVTETensor O,
                const NVTETensor DescaleQ,
                const NVTETensor DescaleK,
                const NVTETensor DescaleV,
                const NVTETensor DescaleS,
                const NVTETensor ScaleS,
                const NVTETensor ScaleO,
                const NVTETensor AmaxS,
                const NVTETensor AmaxO,
                const NVTETensor QKVRaggedOffset,
                const NVTETensor ORaggedOffset,
		const NVTETensor ActualSeqlenQ,
		const NVTETensor ActualSeqlenK,
		const NVTETensor ActualSeqlenO,
		NVTETensor workspace,
		cudaStream_t stream
) {
  NVTE_API_CALL(nvte_cudnn_fmha_fp8_fwd);
  using namespace transformer_engine;
  const Tensor *inputQKV = reinterpret_cast<const Tensor*>(QKV);
  const Tensor *inputM = reinterpret_cast<const Tensor*>(M);
  const Tensor *inputZInv = reinterpret_cast<const Tensor*>(ZInv);
  const Tensor *inputO = reinterpret_cast<const Tensor*>(O);
  const Tensor *inputDescaleQ = reinterpret_cast<const Tensor*>(DescaleQ);
  const Tensor *inputDescaleK = reinterpret_cast<const Tensor*>(DescaleK);
  const Tensor *inputDescaleV = reinterpret_cast<const Tensor*>(DescaleV);
  const Tensor *inputDescaleS = reinterpret_cast<const Tensor*>(DescaleS);
  const Tensor *inputScaleS = reinterpret_cast<const Tensor*>(ScaleS);
  const Tensor *inputScaleO = reinterpret_cast<const Tensor*>(ScaleO);
  const Tensor *inputAmaxS = reinterpret_cast<const Tensor*>(AmaxS);
  const Tensor *inputAmaxO = reinterpret_cast<const Tensor*>(AmaxO);
  const Tensor *inputQKVRaggedOffset = reinterpret_cast<const Tensor*>(QKVRaggedOffset);
  const Tensor *inputORaggedOffset = reinterpret_cast<const Tensor*>(ORaggedOffset);
  const Tensor *inputActualSeqlenQ = reinterpret_cast<const Tensor*>(ActualSeqlenQ);
  const Tensor *inputActualSeqlenK = reinterpret_cast<const Tensor*>(ActualSeqlenK);
  const Tensor *inputActualSeqlenO = reinterpret_cast<const Tensor*>(ActualSeqlenO);
  const int64_t b = inputQKV->data.shape[0];
  const int64_t s_q = inputQKV->data.shape[1];
  const int64_t s_kv = s_q;
  const int64_t h = inputQKV->data.shape[3];
  const int64_t d = inputQKV->data.shape[4];
  const float attnScale = static_cast<float>(1.0 / sqrt(static_cast<double>(d)));
  Tensor *wspace = reinterpret_cast<Tensor*>(workspace);
  MHA_Layout layout = MHA_Layout::QKV_INTERLEAVED;

  fp8_flash_mha_fprop(b, h, s_q, s_kv, d, attnScale, layout,
		  inputQKV, inputM, inputZInv, inputO,
		  inputDescaleQ, inputDescaleK, inputDescaleV, inputDescaleS,
		  inputScaleS, inputScaleO,
		  inputAmaxS, inputAmaxO,
		  inputQKVRaggedOffset, inputORaggedOffset,
		  inputActualSeqlenQ, inputActualSeqlenK, inputActualSeqlenO,
		  wspace->data.dptr,
		  wspace->data.shape[0],
		  stream);

}

//#endif
