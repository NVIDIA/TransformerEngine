/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// CUTLASS NVFP4xNVFP4 -> BF16 GEMM kernels (modeled on CUTLASS example 72a).
// Two C-API entry points: scalar (alpha, beta), and per-row*per-col fused EVT.

#include <transformer_engine/nvfp4_cutlass_gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>

#include "../common.h"
#include "../util/logging.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/functional.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace transformer_engine {
namespace nvfp4_cutlass {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace cute_ = cute;

// CUTLASS GEMM type config (mirrors 72a). BF16 output matches the production
// TE NVFP4 GEMM (cublasLt path), making this a drop-in at the GEMM boundary.

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
// CUTLASS epilogue uses 128-bit vector loads/stores; bf16 packs 8 elts/128b.
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using MmaTileShape = cute_::Shape<cute_::_128, cute_::_128, cute_::_256>;
using ClusterShape = cute_::Shape<cute_::_1, cute_::_1, cute_::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<cute_::Shape<int, int, int, int>, CollectiveMainloop,
                                         CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

using ElementADataPtr = typename ElementA::DataType const*;
using ElementBDataPtr = typename ElementB::DataType const*;
using ElementASfPtr = typename ElementA::ScaleFactorType const*;
using ElementBSfPtr = typename ElementB::ScaleFactorType const*;

// Core launcher (scalar alpha/beta).

static void run_cutlass_gemm(void const* a_data_ptr, void const* b_data_ptr, void const* a_sf_ptr,
                             void const* b_sf_ptr, void* d_ptr, int M, int N, int K, float alpha,
                             float beta, cudaStream_t stream) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute_::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute_::make_shape(M, N, K, 1));

  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                {M, N, K, 1},
                                {reinterpret_cast<ElementADataPtr>(a_data_ptr), stride_A,
                                 reinterpret_cast<ElementBDataPtr>(b_data_ptr), stride_B,
                                 reinterpret_cast<ElementASfPtr>(a_sf_ptr), layout_SFA,
                                 reinterpret_cast<ElementBSfPtr>(b_sf_ptr), layout_SFB},
                                {{alpha, beta},
                                 reinterpret_cast<ElementC const*>(d_ptr),
                                 stride_C,
                                 reinterpret_cast<ElementD*>(d_ptr),
                                 stride_D}};

  Gemm gemm;

  // Stream-ordered workspace alloc; tight perf loops should pre-allocate.
  size_t workspace_size = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    NVTE_CHECK_CUDA(cudaMallocAsync(&workspace, workspace_size, stream));
  }

  cutlass::Status status = gemm.can_implement(args);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 GEMM cannot implement: ", cutlassGetStatusString(status), " (M=", M,
             " N=", N, " K=", K, ")");

  status = gemm.initialize(args, workspace, stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 GEMM initialize failed: ", cutlassGetStatusString(status));

  status = gemm.run(stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 GEMM run failed: ", cutlassGetStatusString(status));

  if (workspace != nullptr) {
    NVTE_CHECK_CUDA(cudaFreeAsync(workspace, stream));
  }
}

// Per-token fused variant: same NVFP4 mainloop, custom EVT folds the
// cuBLAS-LT-equivalent D[i,j] = bf16(NVFP4_DEQUANT_K * alpha_a[i] *
// alpha_b[j] * acc) in one launch; no separate post-scale, no HBM round-trip.

// NVFP4 has TWO-LEVEL dequant: per-block SF (mainloop) + outer 1/2688^2.
// cuBLAS-LT auto-folds the outer via amax slot (see nvte_nvfp4_compute_per_tensor_scale);
// CUTLASS NVFP4 is "raw", the EVT must apply 1/2688^2 explicitly.

// EVT (all fp32 until final cast):
//   L1 tmp1 = alpha_a[i] * acc;   L2 tmp2 = alpha_b[j] * tmp1;
//   L3 out  = bf16(NVFP4_DEQUANT_K * tmp2).

// CUTLASS naming note: Sm90Row/ColBroadcast = "load a row/col vector AND
// broadcast across the orthogonal dim". Sm90ColBroadcast (Stride<_1,_0,_0>)
// indexes M -> per-row; Sm90RowBroadcast (Stride<_0,_1,_0>) indexes N -> per-col.

namespace fusion = cutlass::epilogue::fusion;
constexpr cutlass::FloatRoundStyle kRoundStyleFused = cutlass::FloatRoundStyle::round_to_nearest;

using ElementScale = float;

// NVFP4 spec constant: 1 / (fp4_max^2 * fp8_max^2) = 1/(6^2 * 448^2) = 1/7,225,344.
constexpr float kNvfp4DequantFactor = 1.0f / (6.0f * 6.0f * 448.0f * 448.0f);

using AccFetchNode = fusion::Sm90AccFetch;

using RowScaleNode = fusion::Sm90ColBroadcast<
    /*Stages=*/0,
    /*CtaTileShapeMNK=*/MmaTileShape,
    /*ElementInput=*/ElementScale,
    /*ElementCompute=*/ElementAccumulator>;

using ColScaleNode = fusion::Sm90RowBroadcast<
    /*Stages=*/0,
    /*CtaTileShapeMNK=*/MmaTileShape,
    /*ElementInput=*/ElementScale,
    /*ElementCompute=*/ElementAccumulator>;

// Tile-wide constant (the NVFP4 spec factor 1/2688^2); same pattern as
// Sm90LinCombPerRowBias scalar alpha/beta in sm90_callbacks_tma_warpspecialized.
using ConstScaleNode = fusion::Sm90ScalarBroadcast<ElementScale>;

// L1: tmp1 = alpha_a[i] * acc.
using MulAccByRowEVT = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementAccumulator,
                                                           ElementAccumulator, kRoundStyleFused>,
                                       RowScaleNode, AccFetchNode>;

// L2: tmp2 = alpha_b[j] * tmp1 (still fp32; bf16 cast deferred to L3).
using MulByColEVT = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementAccumulator,
                                                        ElementAccumulator, kRoundStyleFused>,
                                    ColScaleNode, MulAccByRowEVT>;

// L3: D = bf16(NVFP4_DEQUANT_K * tmp2). ElementD=bf16 forces round-to-nearest.
using FusedEVT = fusion::Sm90EVT<
    fusion::Sm90Compute<cutlass::multiplies, ElementD, ElementAccumulator, kRoundStyleFused>,
    ConstScaleNode, MulByColEVT>;

using CollectiveEpilogueFused = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusedEVT>::CollectiveOp;

using CollectiveMainloopFused = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogueFused::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernelFused =
    cutlass::gemm::kernel::GemmUniversal<cute_::Shape<int, int, int, int>, CollectiveMainloopFused,
                                         CollectiveEpilogueFused, void>;

using GemmFused = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelFused>;

static void run_cutlass_per_token_gemm(void const* a_data_ptr, void const* b_data_ptr,
                                       void const* a_sf_ptr, void const* b_sf_ptr,
                                       float const* alpha_a_ptr, float const* alpha_b_ptr,
                                       void* d_ptr, int M, int N, int K, cudaStream_t stream) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute_::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute_::make_shape(M, N, K, 1));

  // EVT args order = children first, then this node's args (empty for
  // Sm90Compute<multiplies>). Sm90ScalarBroadcast<float> has 3 ARRAY fields
  // (BroadcastCount=1) -> constant takes {{value}, {nullptr}, {Stride{}}}.
  typename FusedEVT::Arguments fusion_args{
      // L3 child[0]: ConstScaleNode args -- NVFP4 spec factor 1/2688^2.
      {/*scalars=*/{kNvfp4DequantFactor},
       /*scalar_ptrs=*/{nullptr},
       /*dScalar=*/{}},
      // L3 child[1]: L2 (MulByColEVT) args.
      {
          // L2 child[0]: alpha_b per-col broadcast (Sm90RowBroadcast Arguments).
          {alpha_b_ptr, /*null_default=*/ElementScale{0}, /*dRow=*/{}},
          // L2 child[1]: L1 (MulAccByRowEVT) args.
          {
              // L1 child[0]: alpha_a per-row broadcast (Sm90ColBroadcast Arguments).
              {alpha_a_ptr, /*null_default=*/ElementScale{0}, /*dCol=*/{}},
              // L1 child[1]: AccFetch (empty Arguments).
              {},
              // L1 node: Sm90Compute<multiplies> (empty Arguments).
              {},
          },
          // L2 node: Sm90Compute<multiplies> (empty Arguments).
          {},
      },
      // L3 node: Sm90Compute<multiplies, ElementD=bf16> (empty Arguments).
      {},
  };

  typename GemmFused::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementADataPtr>(a_data_ptr), stride_A,
       reinterpret_cast<ElementBDataPtr>(b_data_ptr), stride_B,
       reinterpret_cast<ElementASfPtr>(a_sf_ptr), layout_SFA,
       reinterpret_cast<ElementBSfPtr>(b_sf_ptr), layout_SFB},
      {fusion_args,
       /*ptr_C=*/nullptr, stride_C,  // EVT has no SrcFetch; C unused.
       reinterpret_cast<ElementD*>(d_ptr), stride_D}};

  GemmFused gemm;

  size_t workspace_size = GemmFused::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    NVTE_CHECK_CUDA(cudaMallocAsync(&workspace, workspace_size, stream));
  }

  cutlass::Status status = gemm.can_implement(args);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 per-token fused GEMM cannot implement: ",
             cutlassGetStatusString(status), " (M=", M, " N=", N, " K=", K, ")");

  status = gemm.initialize(args, workspace, stream);
  NVTE_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS NVFP4 per-token fused GEMM initialize failed: ", cutlassGetStatusString(status));

  status = gemm.run(stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 per-token fused GEMM run failed: ", cutlassGetStatusString(status));

  if (workspace != nullptr) {
    NVTE_CHECK_CUDA(cudaFreeAsync(workspace, stream));
  }
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

}  // namespace nvfp4_cutlass
}  // namespace transformer_engine

// C API.

void nvte_nvfp4_cutlass_gemm(const NVTETensor a_data, const NVTETensor b_data,
                             const NVTETensor a_sf, const NVTETensor b_sf, NVTETensor d,
                             float alpha, float beta, cudaStream_t stream) {
  using namespace transformer_engine;

  auto* a_t = convertNVTETensorCheck(a_data);
  auto* b_t = convertNVTETensorCheck(b_data);
  auto* sa_t = convertNVTETensorCheck(a_sf);
  auto* sb_t = convertNVTETensorCheck(b_sf);
  auto* d_t = convertNVTETensorCheck(d);

  // Logical shapes are interpreted in elements (FP4 storage is packed 2/byte).
  const auto a_shape = a_t->data.shape;
  const auto b_shape = b_t->data.shape;
  const auto d_shape = d_t->data.shape;

  NVTE_CHECK(a_shape.size() == 2, "A must be 2D (M, K), got rank=", a_shape.size());
  NVTE_CHECK(b_shape.size() == 2, "B must be 2D (N, K), got rank=", b_shape.size());
  NVTE_CHECK(d_shape.size() == 2, "D must be 2D (M, N), got rank=", d_shape.size());

  const int M = static_cast<int>(a_shape[0]);
  const int K = static_cast<int>(a_shape[1]);
  const int N = static_cast<int>(b_shape[0]);

  NVTE_CHECK(static_cast<int>(b_shape[1]) == K, "A.K (", K, ") and B.K (", b_shape[1],
             ") must match");
  NVTE_CHECK(static_cast<int>(d_shape[0]) == M, "D.M (", d_shape[0], ") must match A.M (", M, ")");
  NVTE_CHECK(static_cast<int>(d_shape[1]) == N, "D.N (", d_shape[1], ") must match B.N (", N, ")");

  NVTE_CHECK(a_t->data.dtype == DType::kFloat4E2M1, "A data must be FP4 e2m1");
  NVTE_CHECK(b_t->data.dtype == DType::kFloat4E2M1, "B data must be FP4 e2m1");
  NVTE_CHECK(d_t->data.dtype == DType::kBFloat16, "D must be BF16");

  // CUTLASS mainloop expects e4m3 SF; accept raw uint8 (PyTorch's wire type).
  NVTE_CHECK(sa_t->data.dtype == DType::kFloat8E4M3 || sa_t->data.dtype == DType::kByte,
             "A scale must be FP8 e4m3 (or raw uint8 byte)");
  NVTE_CHECK(sb_t->data.dtype == DType::kFloat8E4M3 || sb_t->data.dtype == DType::kByte,
             "B scale must be FP8 e4m3 (or raw uint8 byte)");

  NVTE_CHECK(M > 0 && N > 0 && K > 0, "M, N, K must be positive");
  NVTE_CHECK(M % 256 == 0 && N % 256 == 0 && K % 256 == 0,
             "CUTLASS NVFP4 GEMM (Stage 1) requires M, N, K to be multiples of 256, got M=", M,
             " N=", N, " K=", K, ". Use a TileShape-aware variant for smaller K.");

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  nvfp4_cutlass::run_cutlass_gemm(a_t->data.dptr, b_t->data.dptr, sa_t->data.dptr, sb_t->data.dptr,
                                  d_t->data.dptr, M, N, K, alpha, beta, stream);
#else
  NVTE_ERROR(
      "CUTLASS NVFP4 GEMM requires SM100 (Blackwell). Build with the sm_100a/sm_100f arch flag.");
#endif
}

void nvte_nvfp4_cutlass_per_token_gemm(const NVTETensor a_data, const NVTETensor b_data,
                                       const NVTETensor a_sf, const NVTETensor b_sf,
                                       const NVTETensor alpha_a, const NVTETensor alpha_b,
                                       NVTETensor d, cudaStream_t stream) {
  using namespace transformer_engine;

  auto* a_t = convertNVTETensorCheck(a_data);
  auto* b_t = convertNVTETensorCheck(b_data);
  auto* sa_t = convertNVTETensorCheck(a_sf);
  auto* sb_t = convertNVTETensorCheck(b_sf);
  auto* aa_t = convertNVTETensorCheck(alpha_a);
  auto* ab_t = convertNVTETensorCheck(alpha_b);
  auto* d_t = convertNVTETensorCheck(d);

  const auto a_shape = a_t->data.shape;
  const auto b_shape = b_t->data.shape;
  const auto d_shape = d_t->data.shape;

  NVTE_CHECK(a_shape.size() == 2, "A must be 2D (M, K), got rank=", a_shape.size());
  NVTE_CHECK(b_shape.size() == 2, "B must be 2D (N, K), got rank=", b_shape.size());
  NVTE_CHECK(d_shape.size() == 2, "D must be 2D (M, N), got rank=", d_shape.size());

  const int M = static_cast<int>(a_shape[0]);
  const int K = static_cast<int>(a_shape[1]);
  const int N = static_cast<int>(b_shape[0]);

  NVTE_CHECK(static_cast<int>(b_shape[1]) == K, "A.K (", K, ") and B.K (", b_shape[1],
             ") must match");
  NVTE_CHECK(static_cast<int>(d_shape[0]) == M, "D.M (", d_shape[0], ") must match A.M (", M, ")");
  NVTE_CHECK(static_cast<int>(d_shape[1]) == N, "D.N (", d_shape[1], ") must match B.N (", N, ")");

  NVTE_CHECK(a_t->data.dtype == DType::kFloat4E2M1, "A data must be FP4 e2m1");
  NVTE_CHECK(b_t->data.dtype == DType::kFloat4E2M1, "B data must be FP4 e2m1");
  NVTE_CHECK(d_t->data.dtype == DType::kBFloat16, "D must be BF16");
  NVTE_CHECK(sa_t->data.dtype == DType::kFloat8E4M3 || sa_t->data.dtype == DType::kByte,
             "A scale must be FP8 e4m3 (or raw uint8 byte)");
  NVTE_CHECK(sb_t->data.dtype == DType::kFloat8E4M3 || sb_t->data.dtype == DType::kByte,
             "B scale must be FP8 e4m3 (or raw uint8 byte)");
  NVTE_CHECK(aa_t->data.dtype == DType::kFloat32, "alpha_a must be FP32");
  NVTE_CHECK(ab_t->data.dtype == DType::kFloat32, "alpha_b must be FP32");

  // alpha_a/b accepted as 1D or (M,1)/(N,1); only element count is validated.
  const size_t aa_numel = aa_t->data.numel();
  const size_t ab_numel = ab_t->data.numel();
  NVTE_CHECK(aa_numel == static_cast<size_t>(M), "alpha_a must have M=", M, " elements, got ",
             aa_numel);
  NVTE_CHECK(ab_numel == static_cast<size_t>(N), "alpha_b must have N=", N, " elements, got ",
             ab_numel);

  NVTE_CHECK(M > 0 && N > 0 && K > 0, "M, N, K must be positive");
  NVTE_CHECK(M % 256 == 0 && N % 256 == 0 && K % 256 == 0,
             "CUTLASS NVFP4 per-token fused GEMM requires M, N, K to be multiples of 256, got M=",
             M, " N=", N, " K=", K, ".");

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  nvfp4_cutlass::run_cutlass_per_token_gemm(
      a_t->data.dptr, b_t->data.dptr, sa_t->data.dptr, sb_t->data.dptr,
      reinterpret_cast<float const*>(aa_t->data.dptr),
      reinterpret_cast<float const*>(ab_t->data.dptr), d_t->data.dptr, M, N, K, stream);
#else
  NVTE_ERROR(
      "CUTLASS NVFP4 per-token fused GEMM requires SM100 (Blackwell). Build with sm_100a/sm_100f.");
#endif
}
