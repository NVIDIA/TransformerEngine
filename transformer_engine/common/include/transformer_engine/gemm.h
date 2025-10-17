/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file gemm.h
 *  \brief Functions for matrix multiplication.
 */

#ifndef TRANSFORMER_ENGINE_GEMM_H_
#define TRANSFORMER_ENGINE_GEMM_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/*! \brief Configuration for matrix multiplication. */
typedef void *NVTEMatmulConfig;

/*! \enum NVTEMatmulConfigAttribute
 * \brief Type of option for matrix multiplication.
 */
enum NVTEMatmulConfigAttribute {
  /*! Bias tensor
   *
   * If provided, the bias tensor is applied in the GEMM epilogue.
   */
  kNVTEMatmulConfigBiasTensor = 0,
  /*! Bias gradient tensor
   *
   * If provided, the bias gradient tensor will be filled in the GEMM epilogue.
   */
  kNVTEMatmulConfigDBiasTensor = 1,
  /*! Whether to compute GELU in GEMM epilogue. */
  kNVTEMatmulConfigWithGELUEpilogue = 2,
  /*! Whether to compute GELU backward in GEMM epilogue. */
  kNVTEMatmulConfigWithDGELUEpilogue = 3,
  /*! Auxilliary tensor for GEMM epilogue.
   *
   * For GELU, this will be filled with the GELU input. For GELU
   * backward, this is expected to already be filled with the GELU
   * input.
   */
  kNVTEMatmulConfigEpilogueAuxTensor = 4,
  /*! Whether to use split accumulator for FP8 GEMM. */
  kNVTEMatmulConfigUseSplitAccumulator = 5,
  /*! Number of streaming multiprocessors to use in GEMM kernel. */
  kNVTEMatmulConfigSMCount = 6,
  kNVTEMatmulConfigNumAttributes
};

/*! \brief Create a matrix multiplication configuration. */
NVTEMatmulConfig nvte_create_matmul_config();

/*! \brief Query an option in matrix multiplication configuration.
 *
 *  \param[in] config Matrix multiplication configuration.
 *  \param[in] attr Option type.
 *  \param[out] buf Memory address to write option value. Ignored if
 *                  NULL.
 *  \param[in] size_in_bytes Size of buf.
 *  \param[out] size_written Number of bytes that have been written to
 *                           buf. If buf is NULL, then the number of
 *                           bytes that would have been written.
 */
void nvte_get_matmul_config_attribute(NVTEMatmulConfig config, NVTEMatmulConfigAttribute attr,
                                      void *buf, size_t size_in_bytes, size_t *size_written);

/*! \brief Set an option in matrix multiplication configuration.
 *
 *  \param[in] config Matrix multiplication configuration.
 *  \param[in] attr Option type.
 *  \param[out] buf Memory address to read option value.
 *  \param[in] size_in_bytes Size of buf.
 */
void nvte_set_matmul_config_attribute(NVTEMatmulConfig config, NVTEMatmulConfigAttribute attr,
                                      const void *buf, size_t size_in_bytes);

/*! \brief Destroy a matrix multiplication configuration. */
void nvte_destroy_matmul_config(NVTEMatmulConfig config);

/*! \brief Compute matrix multiplication of 2 matrices, potentially fused with other operations (deprecated).
 *
 * This has been deprecated in favor of nvte_cublas_gemm_v2.
 *
 * Computes:
 *  - `D = AB` if both `bias` and `pre_gelu_out` are empty tensors
 *  - `D = AB + bias` if `pre_gelu_out` is empty and `bias` is not empty
 *  - `D = GELU(AB + bias)` if both `bias` and `pre_gelu_out` are not empty tensors
 *
 *  \param[in]     A                     The A matrix.
 *  \param[in]     B                     The B matrix.
 *  \param[in,out] D                     Output matrix.
 *  \param[in]     bias                  Bias tensor.
 *  \param[in,out] pre_gelu_out          Output matrix before GELU activation.
 *  \param[in]     transa                Whether A matrix is transposed.
 *  \param[in]     transb                Whether B matrix is transposed.
 *  \param[in]     grad                  Whether this operation is part of the
 *                                       gradient computation.
 *  \param[out]    workspace             Workspace tensor.
 *  \param[in]     accumulate            Whether to accumulate the result into the D matrix.
 *  \param[in]     use_split_accumulator Whether to use split accumulator in the FP8 GEMM.
 *  \param[in]     math_sm_count         Number of GPU SMs to use (default=0: use cuBLAS heuristics)
 *  \param[in]     stream                CUDA stream used for the operation.
 */
void nvte_cublas_gemm(const NVTETensor A, const NVTETensor B, NVTETensor D, const NVTETensor bias,
                      NVTETensor pre_gelu_out, bool transa, bool transb, bool grad,
                      NVTETensor workspace, bool accumulate, bool use_split_accumulator,
                      int math_sm_count, cudaStream_t stream);

/*! \brief Compute matrix multiplication of 2 matrices, potentially fused with other operations.
 *
 * Computes:
 *  - `D = alpha * op(A) * op(B) + beta * C`
 *
 *  \param[in]  transa    Whether to transpose A matrix.
 *  \param[in]  transb    Whether to transpose B matrix.
 *  \param[in]  alpha     Scaling factor applied to matmul output.
 *  \param[in]  A         A matrix.
 *  \param[in]  B         B matrix.
 *  \param[in]  beta      Scaling factor applied to C matrix.
 *  \param[in]  C         C matrix.
 *  \param[out] D         Output matrix.
 *  \param[in]  workspace Workspace tensor.
 *  \param[in]  config    Additional configuration.
 *  \param[in]  stream    CUDA stream used for the operation.
 */
void nvte_cublas_gemm_v2(int transa, int transb, const float *alpha, const NVTETensor A,
                         const NVTETensor B, const float *beta, const NVTETensor C, NVTETensor D,
                         NVTETensor workspace, NVTEMatmulConfig config, cudaStream_t stream);

/*! \brief Compute matrix multiplication of 2 matrices, potentially fused with other operations,
 * allowing for using a scaling factor for the GEMM result and the accumulation input (deprecated)
 *
 * This has been deprecated in favor of nvte_cublas_gemm_v2.
 *
 * Computes:
 *  - `D = alpha*AB` if both `bias` and `pre_gelu_out` are empty tensors
 *  - `D = alpha*AB + bias` if `pre_gelu_out` is empty and `bias` is not empty
 *  - `D = GELU(alpha*AB + bias)` if both `bias` and `pre_gelu_out` are not empty tensors
 *
 *  \param[in]     A                     The A matrix.
 *  \param[in]     B                     The B matrix.
 *  \param[in,out] D                     Output matrix.
 *  \param[in]     bias                  Bias tensor.
 *  \param[in,out] pre_gelu_out          Output matrix before GELU activation.
 *  \param[in]     transa                Whether A matrix is transposed.
 *  \param[in]     transb                Whether B matrix is transposed.
 *  \param[in]     grad                  Whether this operation is part of the
 *                                       gradient computation.
 *  \param[out]    workspace             Workspace tensor.
 *  \param[in]     alpha                 Scaling factor applied to the result of the GEMM
 *  \param[in]     beta                  Scaling factor applied to original value of D when
 *                                       accumulating into it. beta=0 means no accumulation.
 *  \param[in]     use_split_accumulator Whether to use split accumulator in the FP8 GEMM.
 *  \param[in]     math_sm_count         Number of GPU SMs to use (default=0: use cuBLAS heuristics)
 *  \param[in]     stream                CUDA stream used for the operation.
 */
void nvte_cublas_gemm_scaled(const NVTETensor A, const NVTETensor B, NVTETensor D,
                             const NVTETensor bias, NVTETensor pre_gelu_out, bool transa,
                             bool transb, bool grad, NVTETensor workspace, float alpha, float beta,
                             bool use_split_accumulator, int math_sm_count, cudaStream_t stream);

/*! \brief Compute matrix multiplication of 2 matrices with chunking and atomic counters.
 *
 * \warning   Cublas atomic gemm uses a beta API and is not tested for all use cases.
 *
 * Computes:
 *  - `D = AB` if both `bias` and `pre_gelu_out` are empty tensors
 *  - `D = AB + bias` if `pre_gelu_out` is empty and `bias` is not empty
 *  - `D = GELU(AB + bias)` if both `bias` and `pre_gelu_out` are not empty tensors
 *
 *  \param[in]     A                     The A matrix.
 *  \param[in]     B                     The B matrix.
 *  \param[in,out] D                     Output matrix.
 *  \param[in]     bias                  Bias tensor.
 *  \param[in,out] pre_gelu_out          Output matrix before GELU activation.
 *  \param[in]     transa                Whether A matrix is transposed.
 *  \param[in]     transb                Whether B matrix is transposed.
 *  \param[in]     grad                  Whether this operation is part of the
 *                                       gradient computation.
 *  \param[out]    workspace             Workspace tensor.
 *  \param[in]     accumulate            Whether to accumulate the result into the D matrix.
 *  \param[in]     use_split_accumulator Whether to use split accumulator in the FP8 GEMM.
 *  \param[in]     math_sm_count         Number of GPU SMs to use (default=0: use cuBLAS heuristics)
 *  \param[in]     m_split               Number of chunks/splits along m-dimension for Atomic GEMM.
 *  \param[in]     n_split               Number of chunks/splits along n-dimension for Atomic GEMM.
 *  \param[in]     gemm_producer         Whether Atomic GEMM is the producer or consumer.
 *  \param[in,out] counter               counter[chunk_i]=0 indicates chunk_i has been produced.
 *  \param[in]     stream                CUDA stream used for the operation.
 */
void nvte_cublas_atomic_gemm(const NVTETensor A, const NVTETensor B, NVTETensor D,
                             const NVTETensor bias, NVTETensor pre_gelu_out, bool transa,
                             bool transb, bool grad, NVTETensor workspace, bool accumulate,
                             bool use_split_accumulator, int math_sm_count, int m_split,
                             int n_split, bool gemm_producer, const NVTETensor counter,
                             cudaStream_t stream);

/*! \brief Compute multiple pairs of matrix multiplication, potentially fused with other operations,
 * on multiple streams.
 *
 * Computes:
 *  - `D = AB` if both `bias` and `pre_gelu_out` are empty tensors
 *  - `D = AB + bias` if `pre_gelu_out` is empty and `bias` is not empty
 *  - `D = GELU(AB + bias)` if both `bias` and `pre_gelu_out` are not empty tensors
 *
 *  \param[in]     A                     The list of A matrices.
 *  \param[in]     B                     The list of B matrices.
 *  \param[in,out] D                     List of output matrices.
 *  \param[in]     bias                  List of bias tensors.
 *  \param[in,out] pre_gelu_out          List of output matrix before GELU activation.
 *  \param[in]     num_gemms             Number of GEMMs to compute.
 *  \param[in]     transa                Whether A matrix is transposed.
 *  \param[in]     transb                Whether B matrix is transposed.
 *  \param[in]     grad                  Whether this operation is part of the
 *                                       gradient computation.
 *  \param[out]    workspace             List of workspace tensors.
 *  \param[in]     accumulate            Whether to accumulate the result into the D matrix.
 *  \param[in]     use_split_accumulator Whether to use split accumulator in the FP8 GEMM.
 *  \param[in]     math_sm_count         Number of GPU SMs to use (default=0: use cuBLAS heuristics)
 *  \param[in]     stream                CUDA stream to wait on.
 */
void nvte_multi_tensor_gemm(const NVTETensor *A, const NVTETensor *B, NVTETensor *D,
                            const NVTETensor *bias, NVTETensor *pre_gelu_out, const int num_gemms,
                            bool transa, bool transb, bool grad, NVTETensor *workspace,
                            bool accumulate, bool use_split_accumulator, int math_sm_count,
                            cudaStream_t stream);
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#ifdef __cplusplus

/*! \namespace transformer_engine
 */
namespace transformer_engine {

/*! \brief TE/JAX cudaGraph requires the cuBLAS initialization to happen outside of the capturing
 * region. This function is a helper to call cublasCreate() which allocate memory for the handle.
 * The function will be called in the initialize phase of the related XLA custom calls.
 */

void nvte_cublas_handle_init();

/*! \struct MatmulConfigWrapper
 *  \brief C++ wrapper for NVTEMatmulConfig.
 */
class MatmulConfigWrapper {
 public:
  MatmulConfigWrapper() : config_{nvte_create_matmul_config()} {}

  MatmulConfigWrapper(const MatmulConfigWrapper &) = delete;
  MatmulConfigWrapper &operator=(const MatmulConfigWrapper &) = delete;

  MatmulConfigWrapper(MatmulConfigWrapper &&other) : config_{other.config_} {
    other.config_ = nullptr;
  }
  MatmulConfigWrapper &operator=(MatmulConfigWrapper &&other) {
    if (config_ != nullptr) {
      nvte_destroy_matmul_config(config_);
    }
    config_ = other.config_;
    other.config_ = nullptr;
    return *this;
  }

  ~MatmulConfigWrapper() {
    if (config_ != nullptr) {
      nvte_destroy_matmul_config(config_);
      config_ = nullptr;
    }
  }

  /*! \brief Get the underlying NVTEMatmulConfig.
   *
   *  \return NVTEMatmulConfig held by this MatmulConfigWrapper.
   */
  operator NVTEMatmulConfig() const noexcept { return config_; }

  /*! \brief Set bias tensor. */
  void set_bias_tensor(NVTETensor bias_tensor) {
    nvte_set_matmul_config_attribute(config_, kNVTEMatmulConfigBiasTensor, &bias_tensor,
                                     sizeof(NVTETensor));
  }

  /*! \brief Set bias gradient tensor. */
  void set_dbias_tensor(NVTETensor dbias_tensor) {
    nvte_set_matmul_config_attribute(config_, kNVTEMatmulConfigDBiasTensor, &dbias_tensor,
                                     sizeof(NVTETensor));
  }

  /*! \brief Set whether to compute GELU in GEMM epilogue. */
  void set_with_gelu_epilogue(bool with_gelu_epilogue) {
    nvte_set_matmul_config_attribute(config_, kNVTEMatmulConfigWithGELUEpilogue,
                                     &with_gelu_epilogue, sizeof(bool));
  }

  /*! \brief Set whether to compute GELU backward in GEMM epilogue. */
  void set_with_dgelu_epilogue(bool with_dgelu_epilogue) {
    nvte_set_matmul_config_attribute(config_, kNVTEMatmulConfigWithDGELUEpilogue,
                                     &with_dgelu_epilogue, sizeof(bool));
  }

  /*! \brief Set auxilliary tensor for GEMM epilogue. */
  void set_epilogue_aux_tensor(NVTETensor epilogue_aux_tensor) {
    nvte_set_matmul_config_attribute(config_, kNVTEMatmulConfigEpilogueAuxTensor,
                                     &epilogue_aux_tensor, sizeof(NVTETensor));
  }

  /*! \brief Set whether to use split accumulator for FP8 GEMM. */
  void set_use_split_accumulator(bool use_split_accumulator) {
    nvte_set_matmul_config_attribute(config_, kNVTEMatmulConfigUseSplitAccumulator,
                                     &use_split_accumulator, sizeof(bool));
  }

  /*! \brief Set number of streaming multiprocessors to use in GEMM kernel. */
  void set_sm_count(int sm_count) {
    nvte_set_matmul_config_attribute(config_, kNVTEMatmulConfigSMCount, &sm_count, sizeof(int));
  }

 private:
  /*! \brief Wrapped NVTEMatmulConfig. */
  NVTEMatmulConfig config_ = nullptr;
};

}  // namespace transformer_engine

#endif  // __cplusplus

#endif  // TRANSFORMER_ENGINE_GEMM_H_
