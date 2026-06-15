/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Grouped (MoE) NVFP4xNVFP4 -> BF16 GEMM with the per-token (per-row * per-col)
// fused EVT epilogue. Single CUTLASS ptr-array grouped launch replaces the
// per-expert Python loop in general_grouped_gemm.
//
// Design mirrors the dense per-token kernel in nvfp4_cutlass_gemm.cu:
//   * one physical layout (A row-major, B col-major, D = A @ B^T row-major);
//     TE's TN/NN/NT directions are realized by the caller choosing rowwise vs
//     columnwise operands, exactly like the dense per-token dispatcher.
//   * the same 3-level Sm90 EVT: D = bf16(1/2688^2 * alpha_a[i] * alpha_b[j] * acc).
// The only structural change vs. dense is ptr-array grouping:
//   * GroupProblemShape<Shape<int,int,int>> + KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
//   * the EVT row/col broadcast leaves take ElementInput_ = float* so CUTLASS
//     switches them to per-group array-of-pointers mode (ptr_col[l] / ptr_row[l]).

#include <transformer_engine/nvfp4_cutlass_gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "../util/logging.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/functional.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace transformer_engine {
namespace nvfp4_cutlass {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace cute_ = cute;
namespace fusion = cutlass::epilogue::fusion;

// ---- Type config (mirrors the dense per-token kernel) ---------------------

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
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ElementScale = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using MmaTileShape = cute_::Shape<cute_::_128, cute_::_128, cute_::_256>;
using ClusterShape = cute_::Shape<cute_::_1, cute_::_1, cute_::_1>;

// Ptr-array (grouped) schedules. NVFP4 = e2m1 data + ue4m3 SF, 1x16 vec.
using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

// Per-group problem shape <M, N, K>.
using ProblemShape = cutlass::gemm::GroupProblemShape<cute_::Shape<int, int, int>>;

constexpr cutlass::FloatRoundStyle kRoundStyleFused = cutlass::FloatRoundStyle::round_to_nearest;
// NVFP4 spec constant: 1 / (fp4_max^2 * fp8_max^2) = 1/(6^2 * 448^2).
constexpr float kNvfp4DequantFactor = 1.0f / (6.0f * 6.0f * 448.0f * 448.0f);

// ---- Per-token fused EVT, lifted to grouped (array-of-pointers) ------------
// Sm90Col/RowBroadcast instantiate IsArrayOfPointers=true when ElementInput_
// is a pointer type (float*); then ptr_col/ptr_row become float const* const*
// and are indexed per group l. Everything else matches the dense EVT.

using AccFetchNode = fusion::Sm90AccFetch;

using RowScaleNode = fusion::Sm90ColBroadcast<
    /*Stages=*/0,
    /*CtaTileShapeMNK=*/MmaTileShape,
    /*ElementInput_=*/ElementScale*,  // pointer type -> per-group ptr array
    /*ElementCompute=*/ElementAccumulator>;

using ColScaleNode = fusion::Sm90RowBroadcast<
    /*Stages=*/0,
    /*CtaTileShapeMNK=*/MmaTileShape,
    /*ElementInput_=*/ElementScale*,  // pointer type -> per-group ptr array
    /*ElementCompute=*/ElementAccumulator>;

// Uniform NVFP4 dequant constant (same for all groups).
using ConstScaleNode = fusion::Sm90ScalarBroadcast<ElementScale>;

// L1: tmp1 = alpha_a[i] * acc.
using MulAccByRowEVT = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementAccumulator,
                                                           ElementAccumulator, kRoundStyleFused>,
                                       RowScaleNode, AccFetchNode>;
// L2: tmp2 = alpha_b[j] * tmp1.
using MulByColEVT = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementAccumulator,
                                                        ElementAccumulator, kRoundStyleFused>,
                                    ColScaleNode, MulAccByRowEVT>;
// L3: D = bf16(NVFP4_DEQUANT_K * tmp2).
using FusedEVT = fusion::Sm90EVT<
    fusion::Sm90Compute<cutlass::multiplies, ElementD, ElementAccumulator, kRoundStyleFused>,
    ConstScaleNode, MulByColEVT>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag*, AlignmentC, ElementD, LayoutDTag*, AlignmentD, EpilogueSchedule,
    FusedEVT>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA, ElementB, LayoutBTag*, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopSchedule>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

using ElementADataT = typename ElementA::DataType;
using ElementBDataT = typename ElementB::DataType;
using ElementSFT = typename ElementA::ScaleFactorType;

// ---- fp32-output accumulate variant (grouped wgrad into fp32 main_grad) -----
// D_g = float(beta * C_g + NVFP4_DEQUANT_K * alpha_a_g[i] * alpha_b_g[j] * acc).
// Reuses the per-group ptr-array scale subtree; only the output element type and
// the beta*C add differ from the overwrite EVT. beta == 0 skips the C load.
using ElementCAcc = float;
using ElementDAcc = float;
constexpr int AlignmentCAcc = 128 / cutlass::sizeof_bits<ElementCAcc>::value;
constexpr int AlignmentDAcc = 128 / cutlass::sizeof_bits<ElementDAcc>::value;

// Z = NVFP4_DEQUANT_K * alpha_b[j] * (alpha_a[i] * acc), in fp32.
using ScaledAccEVT = fusion::Sm90EVT<
    fusion::Sm90Compute<cutlass::multiplies, ElementAccumulator, ElementAccumulator,
                        kRoundStyleFused>,
    ConstScaleNode, MulByColEVT>;
using BetaNode = fusion::Sm90ScalarBroadcast<ElementScale>;
using AccumEVT = fusion::Sm90EVT<
    fusion::Sm90Compute<cutlass::homogeneous_multiply_add, ElementDAcc, ElementAccumulator,
                        kRoundStyleFused>,
    BetaNode, fusion::Sm90SrcFetch<ElementCAcc>, ScaledAccEVT>;

using CollectiveEpilogueAcc = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementCAcc, LayoutCTag*, AlignmentCAcc, ElementDAcc, LayoutDTag*, AlignmentDAcc,
    EpilogueSchedule, AccumEVT>::CollectiveOp;

using CollectiveMainloopAcc = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA, ElementB, LayoutBTag*, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogueAcc::SharedStorage))>,
    MainloopSchedule>::CollectiveOp;

using GemmKernelAcc =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopAcc, CollectiveEpilogueAcc>;
using GemmAcc = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelAcc>;

// Query the SM count exactly once (cudaGetDeviceProperties is very slow and was
// adding a ~ms fixed cost to every grouped launch).
static int cached_sm_count() {
  static int sm = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  return sm;
}

static inline size_t align256(size_t b) { return (b + 255) / 256 * 256; }

// Process-persistent device buffers reused across launches, to avoid the
// per-call cudaMalloc/cudaFree churn that dominated runtime for small grouped
// GEMMs. which=0 -> metadata scratch, which=1 -> CUTLASS workspace.
// Assumes grouped GEMMs are issued serially on one stream (the TE norm); the
// stream-ordered free on regrow keeps it safe under that assumption.
static void* persistent_buffer(size_t bytes, cudaStream_t stream, int which) {
  static void* bufs[2] = {nullptr, nullptr};
  static size_t caps[2] = {0, 0};
  if (bytes > caps[which]) {
    if (bufs[which] != nullptr) {
      NVTE_CHECK_CUDA(cudaFreeAsync(bufs[which], stream));
    }
    const size_t newcap = bytes + bytes / 2;  // slack to avoid frequent regrows
    NVTE_CHECK_CUDA(cudaMallocAsync(&bufs[which], newcap, stream));
    caps[which] = newcap;
  }
  return bufs[which];
}

// Build the shared Z-subtree EVT arguments:
//   Z = NVFP4_DEQUANT_K * alpha_b[j] * (alpha_a[i] * acc).
// ArgsT is FusedEVT::Arguments (overwrite path) or ScaledAccEVT::Arguments
// (accumulate path). The two are structurally identical aggregates but distinct
// C++ types (the enclosing Sm90EVT differs only in its top-level output element),
// so the target type must be named explicitly per call site. aa_d/ab_d are the
// per-group device pointer arrays consumed by the array-of-pointers broadcasts.
template <class ArgsT, class P>
static ArgsT make_z_args(P aa_d, P ab_d) {
  // clang-format off
  return ArgsT{
      {/*scalars=*/{kNvfp4DequantFactor}, /*scalar_ptrs=*/{nullptr}, /*dScalar=*/{}},
      {
          {ab_d, /*null_default=*/ElementScale{0}, /*dRow=*/{}},
          {
              {aa_d, /*null_default=*/ElementScale{0}, /*dCol=*/{}},
              {},  // AccFetch
              {},  // multiplies
          },
          {},  // multiplies
      },
      {},  // multiplies
  };
  // clang-format on
}

// Core launcher. All *_ptrs are host vectors of device pointers (length G);
// Ms/Ns/Ks are host per-group extents. SFs must already be swizzled.
// Shared workspace-alloc + launch tail for both output paths.
template <class GemmT>
static void run_grouped_gemm(GemmT& gemm, typename GemmT::Arguments& args, int G,
                             cudaStream_t stream) {
  size_t workspace_size = GemmT::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    workspace = persistent_buffer(workspace_size, stream, /*which=*/1);
  }

  cutlass::Status status = gemm.can_implement(args);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 grouped per-token GEMM cannot implement: ",
             cutlassGetStatusString(status), " (num_groups=", G, ")");

  status = gemm.initialize(args, workspace, stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 grouped per-token GEMM initialize failed: ",
             cutlassGetStatusString(status));

  status = gemm.run(stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 grouped per-token GEMM run failed: ", cutlassGetStatusString(status));

  // No per-call frees: scratch + workspace live in persistent_buffer and are
  // reused across launches (and grow on demand).
}

// Accumulate=false -> overwrite, ElementD=bf16. Accumulate=true -> fp32 output
// with D += beta * C (beta=1 accumulates into main_grad, beta=0 overwrites).
template <bool Accumulate>
static void run_cutlass_grouped_per_token_gemm_impl(
    const std::vector<const void*>& a_data_ptrs, const std::vector<const void*>& b_data_ptrs,
    const std::vector<const void*>& a_sf_ptrs, const std::vector<const void*>& b_sf_ptrs,
    const std::vector<const float*>& alpha_a_ptrs, const std::vector<const float*>& alpha_b_ptrs,
    const std::vector<void*>& d_ptrs, const std::vector<int>& Ms, const std::vector<int>& Ns,
    const std::vector<int>& Ks, float beta, cudaStream_t stream) {
  using GemmT = std::conditional_t<Accumulate, GemmAcc, Gemm>;
  using StrideAT = typename GemmT::GemmKernel::InternalStrideA;
  using StrideBT = typename GemmT::GemmKernel::InternalStrideB;
  using StrideCT = typename GemmT::GemmKernel::InternalStrideC;
  using StrideDT = typename GemmT::GemmKernel::InternalStrideD;
  using LayoutSFAT = typename GemmT::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFBT = typename GemmT::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using BlkCfgT = typename GemmT::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementDT = std::conditional_t<Accumulate, ElementDAcc, ElementD>;

  const int G = static_cast<int>(Ms.size());

  // Host-side per-group metadata.
  std::vector<typename ProblemShape::UnderlyingProblemShape> problems(G);
  std::vector<StrideAT> stride_A_h(G);
  std::vector<StrideBT> stride_B_h(G);
  std::vector<StrideCT> stride_C_h(G);
  std::vector<StrideDT> stride_D_h(G);
  std::vector<LayoutSFAT> layout_SFA_h(G);
  std::vector<LayoutSFBT> layout_SFB_h(G);

  std::vector<const ElementADataT*> a_ptr_h(G);
  std::vector<const ElementBDataT*> b_ptr_h(G);
  std::vector<const ElementSFT*> sfa_ptr_h(G);
  std::vector<const ElementSFT*> sfb_ptr_h(G);
  std::vector<ElementDT*> d_ptr_h(G);

  for (int g = 0; g < G; ++g) {
    const int M = Ms[g], N = Ns[g], K = Ks[g];
    problems[g] = {M, N, K};
    stride_A_h[g] = cutlass::make_cute_packed_stride(StrideAT{}, {M, K, 1});
    stride_B_h[g] = cutlass::make_cute_packed_stride(StrideBT{}, {N, K, 1});
    stride_C_h[g] = cutlass::make_cute_packed_stride(StrideCT{}, {M, N, 1});
    stride_D_h[g] = cutlass::make_cute_packed_stride(StrideDT{}, {M, N, 1});
    layout_SFA_h[g] = BlkCfgT::tile_atom_to_shape_SFA(cute_::make_shape(M, N, K, 1));
    layout_SFB_h[g] = BlkCfgT::tile_atom_to_shape_SFB(cute_::make_shape(M, N, K, 1));

    a_ptr_h[g] = reinterpret_cast<const ElementADataT*>(a_data_ptrs[g]);
    b_ptr_h[g] = reinterpret_cast<const ElementBDataT*>(b_data_ptrs[g]);
    sfa_ptr_h[g] = reinterpret_cast<const ElementSFT*>(a_sf_ptrs[g]);
    sfb_ptr_h[g] = reinterpret_cast<const ElementSFT*>(b_sf_ptrs[g]);
    d_ptr_h[g] = reinterpret_cast<ElementDT*>(d_ptrs[g]);
  }

  // Mirror all per-group metadata to device through ONE persistent scratch
  // buffer (one H2D copy per array, zero per-call cudaMalloc/Free). All arrays
  // are O(G) and tiny; 256B sub-alignment is safe for every cute POD type here.
  const size_t need =
      align256(problems.size() * sizeof(problems[0])) +
      align256(stride_A_h.size() * sizeof(StrideAT)) +
      align256(stride_B_h.size() * sizeof(StrideBT)) +
      align256(stride_C_h.size() * sizeof(StrideCT)) +
      align256(stride_D_h.size() * sizeof(StrideDT)) +
      align256(layout_SFA_h.size() * sizeof(LayoutSFAT)) +
      align256(layout_SFB_h.size() * sizeof(LayoutSFBT)) +
      align256(a_ptr_h.size() * sizeof(a_ptr_h[0])) +
      align256(b_ptr_h.size() * sizeof(b_ptr_h[0])) +
      align256(sfa_ptr_h.size() * sizeof(sfa_ptr_h[0])) +
      align256(sfb_ptr_h.size() * sizeof(sfb_ptr_h[0])) +
      align256(d_ptr_h.size() * sizeof(d_ptr_h[0])) +
      align256(alpha_a_ptrs.size() * sizeof(alpha_a_ptrs[0])) +
      align256(alpha_b_ptrs.size() * sizeof(alpha_b_ptrs[0]));
  uint8_t* scr = static_cast<uint8_t*>(persistent_buffer(need, stream, /*which=*/0));
  size_t off = 0;
  auto put = [&](const auto& vec) {
    using T = typename std::decay_t<decltype(vec)>::value_type;
    T* p = reinterpret_cast<T*>(scr + off);
    const size_t bytes = vec.size() * sizeof(T);
    NVTE_CHECK_CUDA(
        cudaMemcpyAsync(p, vec.data(), bytes, cudaMemcpyHostToDevice, stream));
    off += align256(bytes);
    return p;
  };
  auto* problems_d = put(problems);
  auto* stride_A_d = put(stride_A_h);
  auto* stride_B_d = put(stride_B_h);
  auto* stride_C_d = put(stride_C_h);
  auto* stride_D_d = put(stride_D_h);
  auto* layout_SFA_d = put(layout_SFA_h);
  auto* layout_SFB_d = put(layout_SFB_h);
  auto* a_ptr_d = put(a_ptr_h);
  auto* b_ptr_d = put(b_ptr_h);
  auto* sfa_ptr_d = put(sfa_ptr_h);
  auto* sfb_ptr_d = put(sfb_ptr_h);
  auto* d_ptr_d = put(d_ptr_h);
  // Per-token outer-scale ptr arrays (consumed by the array-of-pointers EVT).
  auto* alpha_a_d = put(alpha_a_ptrs);
  auto* alpha_b_d = put(alpha_b_ptrs);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cached_sm_count();

  GemmT gemm;
  if constexpr (Accumulate) {
    // D = float(beta * C + Z). beta == 0 skips the C load (uninitialized D safe);
    // beta == 1 accumulates in place (ptr_C aliases ptr_D == main_grad).
    typename AccumEVT::Arguments fusion_args{
        {/*scalars=*/{beta}, /*scalar_ptrs=*/{nullptr}, /*dScalar=*/{}},      // beta
        {},                                                                  // C source fetch
        make_z_args<typename ScaledAccEVT::Arguments>(alpha_a_d, alpha_b_d),  // Z subtree
        {},                                                                  // multiply_add
    };
    // ptr_C aliases ptr_D (== main_grad). The epilogue wants ElementC const**;
    // d_ptr_d is ElementCAcc** (non-const), so round-trip through void* to add
    // the inner const (a direct reinterpret_cast would reject the qualifier change).
    auto* c_ptr_d = reinterpret_cast<const ElementCAcc**>(reinterpret_cast<void*>(d_ptr_d));
    typename GemmT::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {G, problems_d, /*host_problem_shapes=*/nullptr},
        {a_ptr_d, stride_A_d, b_ptr_d, stride_B_d, sfa_ptr_d, layout_SFA_d, sfb_ptr_d, layout_SFB_d},
        {fusion_args, /*ptr_C=*/c_ptr_d, stride_C_d, d_ptr_d, stride_D_d},
        hw_info};
    run_grouped_gemm(gemm, args, G, stream);
  } else {
    // Overwrite path: D = bf16(Z). GemmT == Gemm here.
    typename FusedEVT::Arguments fusion_args =
        make_z_args<typename FusedEVT::Arguments>(alpha_a_d, alpha_b_d);
    typename GemmT::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {G, problems_d, /*host_problem_shapes=*/nullptr},
        {a_ptr_d, stride_A_d, b_ptr_d, stride_B_d, sfa_ptr_d, layout_SFA_d, sfb_ptr_d, layout_SFB_d},
        {fusion_args, /*ptr_C=*/nullptr, stride_C_d, d_ptr_d, stride_D_d},
        hw_info};
    run_grouped_gemm(gemm, args, G, stream);
  }
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

}  // namespace nvfp4_cutlass
}  // namespace transformer_engine

// ---- C API ----------------------------------------------------------------

void nvte_nvfp4_cutlass_grouped_per_token_gemm(
    int num_groups, const NVTETensor* a_data, const NVTETensor* b_data, const NVTETensor* a_sf,
    const NVTETensor* b_sf, const NVTETensor* alpha_a, const NVTETensor* alpha_b, NVTETensor* d,
    bool accumulate, cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(num_groups > 0, "num_groups must be positive, got ", num_groups);

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::vector<const void*> a_data_ptrs(num_groups), b_data_ptrs(num_groups), a_sf_ptrs(num_groups),
      b_sf_ptrs(num_groups);
  std::vector<const float*> alpha_a_ptrs(num_groups), alpha_b_ptrs(num_groups);
  std::vector<void*> d_ptrs(num_groups);
  std::vector<int> Ms(num_groups), Ns(num_groups), Ks(num_groups);

  // Output dtype must be uniform across groups (one kernel instance per launch).
  // BF16 -> overwrite. FP32 -> accumulate-capable (wgrad into fp32 main_grad).
  const bool d_is_fp32 = convertNVTETensorCheck(d[0])->data.dtype == DType::kFloat32;
  NVTE_CHECK(!accumulate || d_is_fp32,
             "NVFP4 grouped per-token GEMM accumulate=true requires FP32 outputs (main_grad)");

  for (int g = 0; g < num_groups; ++g) {
    auto* a_t = convertNVTETensorCheck(a_data[g]);
    auto* b_t = convertNVTETensorCheck(b_data[g]);
    auto* sa_t = convertNVTETensorCheck(a_sf[g]);
    auto* sb_t = convertNVTETensorCheck(b_sf[g]);
    auto* aa_t = convertNVTETensorCheck(alpha_a[g]);
    auto* ab_t = convertNVTETensorCheck(alpha_b[g]);
    auto* d_t = convertNVTETensorCheck(d[g]);

    const auto a_shape = a_t->data.shape;
    const auto b_shape = b_t->data.shape;
    const auto d_shape = d_t->data.shape;
    NVTE_CHECK(a_shape.size() == 2, "A[", g, "] must be 2D (M, K)");
    NVTE_CHECK(b_shape.size() == 2, "B[", g, "] must be 2D (N, K)");
    NVTE_CHECK(d_shape.size() == 2, "D[", g, "] must be 2D (M, N)");

    const int M = static_cast<int>(a_shape[0]);
    const int K = static_cast<int>(a_shape[1]);
    const int N = static_cast<int>(b_shape[0]);

    NVTE_CHECK(static_cast<int>(b_shape[1]) == K, "group ", g, ": A.K/B.K mismatch");
    NVTE_CHECK(static_cast<int>(d_shape[0]) == M && static_cast<int>(d_shape[1]) == N,
               "group ", g, ": D shape mismatch");
    NVTE_CHECK(a_t->data.dtype == DType::kFloat4E2M1 && b_t->data.dtype == DType::kFloat4E2M1,
               "group ", g, ": A/B must be FP4 e2m1");
    NVTE_CHECK((d_t->data.dtype == DType::kFloat32) == d_is_fp32,
               "group ", g, ": D dtype must be uniform across groups");
    NVTE_CHECK(d_t->data.dtype == DType::kBFloat16 || d_t->data.dtype == DType::kFloat32,
               "group ", g, ": D must be BF16 or FP32");
    NVTE_CHECK(aa_t->data.dtype == DType::kFloat32 && ab_t->data.dtype == DType::kFloat32,
               "group ", g, ": alpha_a/alpha_b must be FP32");
    NVTE_CHECK(aa_t->data.numel() == static_cast<size_t>(M), "group ", g, ": alpha_a must be (M,)");
    NVTE_CHECK(ab_t->data.numel() == static_cast<size_t>(N), "group ", g, ": alpha_b must be (N,)");
    NVTE_CHECK(M > 0 && N > 0 && K > 0, "group ", g, ": M, N, K must be positive (filter empties)");
    NVTE_CHECK(M % 128 == 0 && N % 128 == 0 && K % 128 == 0, "group ", g,
               ": M, N, K must be multiples of 128 (1-CTA MmaTile = (128,128,256)), got M=", M,
               " N=", N, " K=", K);

    a_data_ptrs[g] = a_t->data.dptr;
    b_data_ptrs[g] = b_t->data.dptr;
    a_sf_ptrs[g] = sa_t->data.dptr;
    b_sf_ptrs[g] = sb_t->data.dptr;
    alpha_a_ptrs[g] = reinterpret_cast<const float*>(aa_t->data.dptr);
    alpha_b_ptrs[g] = reinterpret_cast<const float*>(ab_t->data.dptr);
    d_ptrs[g] = d_t->data.dptr;
    Ms[g] = M;
    Ns[g] = N;
    Ks[g] = K;
  }

  if (d_is_fp32) {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_impl</*Accumulate=*/true>(
        a_data_ptrs, b_data_ptrs, a_sf_ptrs, b_sf_ptrs, alpha_a_ptrs, alpha_b_ptrs, d_ptrs, Ms, Ns,
        Ks, /*beta=*/accumulate ? 1.0f : 0.0f, stream);
  } else {
    nvfp4_cutlass::run_cutlass_grouped_per_token_gemm_impl</*Accumulate=*/false>(
        a_data_ptrs, b_data_ptrs, a_sf_ptrs, b_sf_ptrs, alpha_a_ptrs, alpha_b_ptrs, d_ptrs, Ms, Ns,
        Ks, /*beta=*/0.0f, stream);
  }
#else
  NVTE_ERROR(
      "CUTLASS NVFP4 grouped per-token GEMM requires SM100 (Blackwell). Build with "
      "sm_100a/sm_100f.");
#endif
}
