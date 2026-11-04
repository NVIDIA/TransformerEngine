/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/cuda_runtime.h"
#include "../util/logging.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_mxfp4_mxfp8_gemm.h"

namespace transformer_engine {
namespace {

using namespace cute;

// CUTLASS A is TE's B (MXFP8) and CUTLASS B is TE's A (MXFP4).
template <typename LayoutBTag, typename ElementOut>
struct Mxfp4Mxfp8GemmConfig {
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 16;
  using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  static constexpr int AlignmentB = 128;
  using ElementC = ElementOut;
  using ElementD = ElementOut;
  using LayoutCTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOut>::value;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using TileShape = cute::Shape<_128, _128, _128>;
  using ClusterShape = cute::Shape<_1, _1, _1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC, ElementD, LayoutCTag, AlignmentC,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Config>
void run_gemm(const void *a_fp8, const void *a_scale, const void *b_fp4, const void *b_scale,
              void *d, int m, int n, int k, float alpha, float beta, void *workspace,
              size_t workspace_size, cudaStream_t stream) {
  using Gemm = typename Config::Gemm;
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementD = typename Config::ElementD;
  using BlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {m, n, 1});
  auto layout_SFA = BlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  auto layout_SFB = BlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto *d_ptr = reinterpret_cast<ElementD *>(d);
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<const typename ElementA::DataType *>(a_fp8), stride_A,
       reinterpret_cast<const typename ElementB::DataType *>(b_fp4), stride_B,
       reinterpret_cast<const typename ElementA::ScaleFactorType *>(a_scale), layout_SFA,
       reinterpret_cast<const typename ElementB::ScaleFactorType *>(b_scale), layout_SFB},
      {{alpha, beta}, d_ptr, stride_D, d_ptr, stride_D}};

  Gemm gemm;
  NVTE_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
             "MXFP4 x MXFP8 GEMM does not support this problem (M=", m, ", N=", n, ", K=", k, ").");
  const size_t required_workspace = Gemm::get_workspace_size(arguments);
  NVTE_CHECK(required_workspace <= workspace_size, "MXFP4 x MXFP8 GEMM requires ",
             required_workspace, " bytes of workspace, but only ", workspace_size,
             " are available.");
  NVTE_CHECK(gemm.initialize(arguments, workspace, stream) == cutlass::Status::kSuccess,
             "Failed to initialize MXFP4 x MXFP8 GEMM.");
  NVTE_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "Failed to run MXFP4 x MXFP8 GEMM.");
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename OType, typename BiasType>
__global__ void broadcast_bias_kernel(OType *out, const BiasType *bias, size_t rows, size_t cols) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx < rows * cols) {
    out[idx] = static_cast<OType>(static_cast<float>(bias[idx % cols]));
  }
}

}  // namespace

bool is_mxfp4_mxfp8_gemm(const Tensor &A, const Tensor &B) {
  if (!is_mxfp_scaling(A.scaling_mode) || !is_mxfp_scaling(B.scaling_mode)) {
    return false;
  }
  return is_fp4_dtype(A.dtype()) != is_fp4_dtype(B.dtype());
}

void cutlass_mxfp4_mxfp8_gemm(const Tensor *inputA, const Tensor *inputB, Tensor *outputD,
                              const Tensor *inputBias, bool transa, bool transb, float alpha,
                              float beta, void *workspace, size_t workspace_size,
                              cudaStream_t stream) {
  NVTE_CHECK(cuda::sm_arch() >= 100, "MXFP4 x MXFP8 GEMM requires Blackwell (SM 10.0+).");
  NVTE_CHECK(is_fp4_dtype(inputA->dtype()) && is_fp8_dtype(inputB->dtype()),
             "MXFP4 x MXFP8 GEMM requires A to be MXFP4 and B to be MXFP8.");
  NVTE_CHECK(!transb, "MXFP4 x MXFP8 GEMM requires a non-transposed B operand.");

  const auto [A0, A1] = inputA->flat_2d_dims();
  const auto [B0, B1] = inputB->flat_2d_dims();
  const size_t m = transa ? A0 : A1;
  const size_t k = transa ? A1 : A0;
  const size_t n = B0;
  NVTE_CHECK(B1 == k, "GEMM inputs have incompatible dimensions (A is ", A0, "x", A1, ", B is ", B0,
             "x", B1, ").");
  const auto [D0, D1] = outputD->flat_2d_dims();
  NVTE_CHECK(D0 == n && D1 == m, "GEMM output has invalid shape (expected ", n, "x", m, ", got ",
             D0, "x", D1, ").");
  if (m == 0 || n == 0) {
    return;
  }
  NVTE_CHECK(k % 128 == 0, "MXFP4 x MXFP8 GEMM requires K (", k, ") to be a multiple of 128.");
  NVTE_CHECK(transa || m % 128 == 0, "MXFP4 x MXFP8 GEMM with a column-wise MXFP4 operand ",
             "requires M (", m, ") to be a multiple of 128.");
  NVTE_CHECK(inputA->with_gemm_swizzled_scales && inputB->with_gemm_swizzled_scales,
             "MXFP4 x MXFP8 GEMM requires GEMM-swizzled scales.");

  const SimpleTensor &a_data = transa ? inputA->data : inputA->columnwise_data;
  const SimpleTensor &a_scale = transa ? inputA->scale_inv : inputA->columnwise_scale_inv;
  NVTE_CHECK(a_data.has_data(), "MXFP4 operand is missing ", transa ? "row-wise" : "column-wise",
             " data.");
  NVTE_CHECK(is_fp4_dtype(a_data.dtype), "MXFP4 operand data must be FP4.");
  NVTE_CHECK(inputB->data.has_data(), "MXFP8 operand is missing row-wise data.");
  NVTE_CHECK(inputB->data.dtype == DType::kFloat8E4M3, "MXFP8 operand must be E4M3.");

  // Row-major CUTLASS problem: D[n, m] = B[n, k] * A[m, k]^T.
  const int cm = static_cast<int>(n);
  const int cn = static_cast<int>(m);
  const int ck = static_cast<int>(k);
  const DType out_dtype = outputD->data.dtype;
  NVTE_CHECK(m % 8 == 0, "MXFP4 x MXFP8 GEMM requires M (", m, ") to be a multiple of 8.");

  // D = alpha * acc + bias, with bias as the epilogue source operand.
  if (inputBias != nullptr && inputBias->data.has_data()) {
    NVTE_CHECK(beta == 0.0f, "MXFP4 x MXFP8 GEMM does not support bias with accumulation.");
    NVTE_CHECK(inputBias->data.numel() == m, "Bias has invalid size (expected ", m, ", got ",
               inputBias->data.numel(), ").");
    constexpr size_t threads = 256;
    const size_t numel = m * n;
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        out_dtype, OType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
            inputBias->data.dtype, BiasType,
            broadcast_bias_kernel<<<DIVUP(numel, threads), threads, 0, stream>>>(
                reinterpret_cast<OType *>(outputD->data.dptr),
                reinterpret_cast<const BiasType *>(inputBias->data.dptr), n, m);););
    NVTE_CHECK_CUDA(cudaGetLastError());
    beta = 1.0f;
  }

#define NVTE_MXFP4_MXFP8_GEMM_LAUNCH(LAYOUT_B, OTYPE)                                           \
  run_gemm<Mxfp4Mxfp8GemmConfig<LAYOUT_B, OTYPE>>(                                              \
      inputB->data.dptr, inputB->scale_inv.dptr, a_data.dptr, a_scale.dptr, outputD->data.dptr, \
      cm, cn, ck, alpha, beta, workspace, workspace_size, stream)

  if (out_dtype == DType::kBFloat16) {
    if (transa) {
      NVTE_MXFP4_MXFP8_GEMM_LAUNCH(cutlass::layout::ColumnMajor, cutlass::bfloat16_t);
    } else {
      NVTE_MXFP4_MXFP8_GEMM_LAUNCH(cutlass::layout::RowMajor, cutlass::bfloat16_t);
    }
  } else if (out_dtype == DType::kFloat32) {
    if (transa) {
      NVTE_MXFP4_MXFP8_GEMM_LAUNCH(cutlass::layout::ColumnMajor, float);
    } else {
      NVTE_MXFP4_MXFP8_GEMM_LAUNCH(cutlass::layout::RowMajor, float);
    }
  } else {
    NVTE_ERROR("MXFP4 x MXFP8 GEMM supports BF16 and FP32 outputs, got ", to_string(out_dtype),
               ".");
  }
#undef NVTE_MXFP4_MXFP8_GEMM_LAUNCH
}

}  // namespace transformer_engine
