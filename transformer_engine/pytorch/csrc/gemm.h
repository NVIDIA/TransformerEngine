#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>


enum GEMM_INPUT_LAYOUT{
  TN = 0,
  NN = 1,
  NT = 2,
};

std::tuple<bool, bool> get_gemm_input_layout(const GEMM_INPUT_LAYOUT input_layout)
{
    switch (input_layout) {
        case GEMM_INPUT_LAYOUT::TN:
            return std::make_tuple(true, false);
        case GEMM_INPUT_LAYOUT::NN:
            return std::make_tuple(false, false);
        case GEMM_INPUT_LAYOUT::NT:
            return std::make_tuple(false, true);
        default:
            NVTE_ERROR("Invalid layout");
    }
}

cudaDataType_t get_cuda_dtype(torch::Tensor &t, bool fp8)
{
    if (fp8) {
        return CUDA_R_8F_E4M3;
    } else {
        if (t.dtype() == torch::kHalf)
            return CUDA_R_16F;
        else if  (t.dtype() == torch::kBFloat16)
            return CUDA_R_16BF;
        else if  (t.dtype() == torch::kFloat32)
            return CUDA_R_32F;
        else
            NVTE_ERROR("Invalid data type");
    }

}

bool is_fp8(torch::Tensor &input_a, torch::Tensor &input_b)
{
    if (input_a.dtype() == at::kByte && input_b.dtype() == at::kByte)
        return true;
    return false;
}


void matmul_cuda(torch::Tensor &input_a,
                 torch::Tensor &input_b,
                 torch::Tensor &output,
                 torch::Tensor &psum,
                 int m,
                 int n,
                 int k,
                 bool transa,
                 bool transb,
                 cudaStream_t stream,
                 void *lt_workspace,
                 int math_sms,
                 bool fast_accum,
                 bool fp8,
                 bool grad_accum)
{
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasLtHandle_t ltHandle = (cublasLtHandle_t) handle;

    int lda, ldb, ldd;
    if (transa && !transb) {  // TN
        lda = k;
        ldb = k;
        ldd = m;
    } else if (!transa && !transb) {  // NN
        lda = m;
        ldb = k;
        ldd = m;
    } else if (!transa && transb) {  // NT
        lda = m;
        ldb = n;
        ldd = m;
    } else {  // TT
        NVTE_ERROR("TT layout not allowed.");
    }

    const float alpha = 1.0;
    const float beta = (grad_accum) ? 1.0 : 0.0;
    size_t workspaceSize = 1 << 25;
    cublasLtMatmulDescOpaque_t operationDesc = {};
    cublasLtMatrixLayoutOpaque_t Adesc = {}, Bdesc = {}, Cdesc = {};
    cublasLtMatmulPreferenceOpaque_t preference = {};
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    
    // input layout
    cublasOperation_t transa_op = (transa) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb_op = (transb) ? CUBLAS_OP_T : CUBLAS_OP_N;
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescInit(
        &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        &operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa_op, sizeof(transa_op)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        &operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb_op, sizeof(transb_op)));
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        &operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    
    // Input/output description
    cudaDataType_t cuda_dtype_a, cuda_dtype_b, cuda_dtype_c;
    cuda_dtype_a = get_cuda_dtype(input_a, fp8);
    cuda_dtype_b = get_cuda_dtype(input_b, fp8);
    cuda_dtype_c = get_cuda_dtype(output, false);
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutInit(
        &Adesc, cuda_dtype_a, transa_op == CUBLAS_OP_N ? m : k, transa_op == CUBLAS_OP_N ? k : m, lda));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutInit(
        &Bdesc, cuda_dtype_b, transb_op == CUBLAS_OP_N ? k : n, transb_op == CUBLAS_OP_N ? n : k, ldb));
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutInit(&Cdesc, cuda_dtype_c, m, n, ldd));

    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceInit(&preference));
    NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        &preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // MatMul heuristic
    NVTE_CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, &operationDesc, &Adesc, &Bdesc, &Cdesc, &Cdesc, &preference, 1, &heuristicResult, &returnedResults));
    
    // Set math SM count
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        &operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sms, sizeof(math_sms)));

    // FP8 specifics
//    if (fp8) {
//        const int8_t fastAccuMode = (fast_accum) ? 1 : 0;
//        NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(&operationDesc,
//                                                         CUBLASLT_MATMUL_DESC_FAST_ACCUM,
//                                                         &fastAccuMode,
//                                                         sizeof(fastAccuMode)));
//    }

    if (returnedResults == 0) NVTE_CHECK_CUBLAS(CUBLAS_STATUS_NOT_SUPPORTED);

    void *A = input_a.data_ptr();
    void *B = input_b.data_ptr();
    void *C = (grad_accum) ? psum.data_ptr() : output.data_ptr();
    void *D = output.data_ptr();

    NVTE_CHECK_CUBLAS(cublasLtMatmul(
        ltHandle,
        &operationDesc,
        &alpha,
        A,
        &Adesc,
        B,
        &Bdesc,
        &beta,
        C,
        &Cdesc,
        D,
        &Cdesc,
        &heuristicResult.algo,
        lt_workspace,
        workspaceSize,
        stream));
}

