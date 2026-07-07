/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Grouped (MoE) per-tensor NVFP4xNVFP4 -> BF16 GEMM. A single CUTLASS ptr-array
// grouped launch replaces the per-expert multi-stream cuBLASLt loop used by the
// production NVFP4 grouped path (multi_stream_cublas_gemm).
//
// Design notes:
//   * Mainloop / block-scaled / ptr-array config is identical to the per-token
//     grouped kernel (nvfp4_cutlass_grouped_gemm on the nvFP4 per-token recipe):
//     A row-major, B col-major, D = A @ B^T row-major; NVFP4 = e2m1 data +
//     ue4m3 1x16 block scale-factors. The caller (dispatcher) realizes TE's
//     TN direction and the cuBLAS->CUTLASS A/B swap before calling in.
//   * The main structural difference vs. per-token is the epilogue: per-tensor
//     scaling collapses the two per-row/col vector broadcasts into one fp32
//     scalar per group, so the no-bias case uses the default LinearCombination
//     fusion with the per-group alpha_ptr_array (D = alpha[g] * acc).
//   * Optional fused per-group bias (fprop) reuses the per-token grouped
//     kernel's array-of-pointers EVT pattern: a hand-built Sm90 EVT computing
//     D = ElementOut(alpha[g]*acc + bias[g]) with a ptr-array Sm90RowBroadcast
//     bias leaf (ElementBias* -> per-group ptr_row[g]). bias[g] is a length-N
//     vector broadcast along M (== cuBLAS per-row bias of length m after the
//     A/B swap). The EVT is passed straight to the CollectiveBuilder, so no
//     custom FusionCallbacks specialization is required.

#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/logging.h"
#include "common/util/system.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "nvfp4_cutlass_grouped_gemm.cuh"

namespace transformer_engine {
namespace nvfp4_cutlass {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace cute_ = cute;
namespace fusion = cutlass::epilogue::fusion;

// ---- Type config (mirrors the per-token grouped kernel) -------------------
//
// Templated on the output element so we can instantiate a BF16-output kernel
// (fprop / dgrad, overwrite) and an FP32-output kernel (wgrad, optionally
// accumulating into main_grad). A second flag selects the epilogue fusion:
// stock LinearCombination (no bias) or a hand-built per-group bias EVT (fprop
// only, see below). Everything else is identical across instantiations.

template <typename ElementOutT, bool kHasBias_ = false>
struct PerTensorCfg {
  static constexpr bool kHasBias = kHasBias_;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  // C (accumulate source) and D (output) share the element type. For accumulate
  // (wgrad), C == D == the fp32 main_grad buffer; for overwrite, C is unused.
  using ElementC = ElementOutT;
  using ElementD = ElementOutT;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using ElementCompute = float;
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

  static constexpr cutlass::FloatRoundStyle kRoundStyle =
      cutlass::FloatRoundStyle::round_to_nearest;

  // Bias element / alignment (per-col bias == one vector of length N per
  // group). Only used when kHasBias; matches the output element type.
  using ElementBias = ElementOutT;
  static constexpr int AlignmentBias = 128 / cutlass::sizeof_bits<ElementBias>::value;

  // Per-tensor epilogue:
  //   * no bias -> D = alpha[g] * acc + beta * C   (stock LinearCombination,
  //                exposes per-group alpha_ptr_array + scalar beta).
  //   * bias    -> D = ElementOut(alpha[g] * acc + bias[g])   (fprop, overwrite).
  // The bias variant is a hand-built Sm90 EVT, mirroring the per-token grouped
  // kernel's array-of-pointers broadcast pattern: alpha[g] is a per-group
  // scalar (Sm90ScalarBroadcastPtrArray, indexed scalar_ptr_array[g]) and
  // bias[g] is a per-group length-N vector (Sm90RowBroadcast with ElementBias*
  // -> IsArrayOfPointers, indexed ptr_row[g]; broadcast along M / indexed by N
  // == cuBLAS per-row bias of length m after the A/B swap). The whole EVT is
  // passed straight to the CollectiveBuilder, so no custom FusionCallbacks
  // specialization is needed.
  using BiasAlphaNode =
      fusion::Sm90ScalarBroadcastPtrArray<ElementScale,
                                          cute_::Stride<cute_::_0, cute_::_0, int64_t>>;
  using BiasNode =
      fusion::Sm90RowBroadcast<0, MmaTileShape, ElementBias *, ElementCompute,
                               cute_::Stride<cute_::_0, cute_::_1, int64_t>, AlignmentBias>;
  using BiasEVT = fusion::Sm90EVT<
      fusion::Sm90Compute<cutlass::homogeneous_multiply_add, ElementD, ElementCompute, kRoundStyle>,
      BiasAlphaNode, fusion::Sm90AccFetch, BiasNode>;  // alpha[g] * acc + bias[g]

  using FusionOp =
      std::conditional_t<kHasBias, BiasEVT,
                         cutlass::epilogue::fusion::LinearCombination<
                             ElementD, ElementCompute, ElementC, ElementScale, kRoundStyle>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutCTag *, AlignmentC, ElementD, LayoutDTag *, AlignmentD, EpilogueSchedule,
      FusionOp>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutATag *, AlignmentA, ElementB, LayoutBTag *,
      AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
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
};

static inline size_t align256(size_t b) { return (b + 255) / 256 * 256; }

// ---- Graph-safe device-array path (grouped-tensor / cublasLt grouped API) ----
//
// Builds all per-group CUTLASS metadata on device from the dim arrays produced
// by the shared GroupedTensor setup kernel, so there is no host<->device sync
// and the launch is CUDA-graph capturable. Only the problem shape and the
// block-scale layouts depend on the (dynamic) token dim M; the strides depend
// only on the static N/K but are built here too to keep everything on device.
template <typename Cfg>
__global__ void build_grouped_metadata_kernel(
    int G, const int *M_arr, const int *N_arr, const int *a_rows, const int *a_cols, bool a_trans,
    typename Cfg::ProblemShape::UnderlyingProblemShape *problems, typename Cfg::StrideA *stride_A,
    typename Cfg::StrideB *stride_B, typename Cfg::StrideC *stride_C,
    typename Cfg::StrideD *stride_D, typename Cfg::LayoutSFA *layout_SFA,
    typename Cfg::LayoutSFB *layout_SFB) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= G) return;
  const int M = M_arr[g];                         // cuBLAS d_cols == CUTLASS M (tokens)
  const int N = N_arr[g];                         // cuBLAS d_rows == CUTLASS N (out_features)
  const int K = a_trans ? a_rows[g] : a_cols[g];  // contraction (hidden)
  problems[g] = cute_::make_shape(M, N, K);
  stride_A[g] =
      cutlass::make_cute_packed_stride(typename Cfg::StrideA{}, cute_::make_shape(M, K, 1));
  stride_B[g] =
      cutlass::make_cute_packed_stride(typename Cfg::StrideB{}, cute_::make_shape(N, K, 1));
  stride_C[g] =
      cutlass::make_cute_packed_stride(typename Cfg::StrideC{}, cute_::make_shape(M, N, 1));
  stride_D[g] =
      cutlass::make_cute_packed_stride(typename Cfg::StrideD{}, cute_::make_shape(M, N, 1));
  layout_SFA[g] = Cfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute_::make_shape(M, N, K, 1));
  layout_SFB[g] = Cfg::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute_::make_shape(M, N, K, 1));
}

template <typename ElementOutT>
static void run_impl_device(void **A_ptrs, void **B_ptrs, void **a_scale_inv_ptrs,
                            void **b_scale_inv_ptrs, float **alpha_ptrs, void **C_ptrs,
                            void **D_ptrs, float **beta_ptrs, const int *a_rows, const int *a_cols,
                            const int *d_rows, const int *d_cols, bool a_trans, int G,
                            void *ext_workspace, size_t ext_workspace_bytes, cudaStream_t stream) {
  using Cfg = PerTensorCfg<ElementOutT, /*kHasBias=*/false>;
  using Gemm = typename Cfg::Gemm;
  using StrideA = typename Cfg::StrideA;
  using StrideB = typename Cfg::StrideB;
  using StrideC = typename Cfg::StrideC;
  using StrideD = typename Cfg::StrideD;
  using LayoutSFA = typename Cfg::LayoutSFA;
  using LayoutSFB = typename Cfg::LayoutSFB;
  using ElementADataT = typename Cfg::ElementADataT;
  using ElementBDataT = typename Cfg::ElementBDataT;
  using ElementSFT = typename Cfg::ElementSFT;
  using ElementC = typename Cfg::ElementC;
  using ElementD = typename Cfg::ElementD;
  using ProblemShape = typename Cfg::ProblemShape;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

  // Resolve the active device once to tell the CUTLASS scheduler which device it
  // is running on. Do not assume device 0 -- in multi-GPU-per-process setups
  // (e.g. pipeline parallelism) the current device may be non-zero.
  const int device = transformer_engine::cuda::current_device();

  // Device scratch for the per-group metadata (built by the kernel below), carved
  // from the front of the caller-provided workspace. Sized only by G, which is
  // static for a given grouped GEMM, so no allocation happens here -> nothing to
  // capture during CUDA-graph replay.
  const size_t need = align256(G * sizeof(UnderlyingProblemShape)) + align256(G * sizeof(StrideA)) +
                      align256(G * sizeof(StrideB)) + align256(G * sizeof(StrideC)) +
                      align256(G * sizeof(StrideD)) + align256(G * sizeof(LayoutSFA)) +
                      align256(G * sizeof(LayoutSFB));
  NVTE_CHECK(need <= ext_workspace_bytes,
             "CUTLASS NVFP4 grouped per-tensor GEMM: provided workspace too small for metadata "
             "scratch (need ",
             need, " bytes, have ", ext_workspace_bytes, ").");
  uint8_t *scr = static_cast<uint8_t *>(ext_workspace);
  size_t off = 0;
  auto carve = [&](size_t bytes) {
    uint8_t *p = scr + off;
    off += align256(bytes);
    return p;
  };
  auto *problems_d =
      reinterpret_cast<UnderlyingProblemShape *>(carve(G * sizeof(UnderlyingProblemShape)));
  auto *stride_A_d = reinterpret_cast<StrideA *>(carve(G * sizeof(StrideA)));
  auto *stride_B_d = reinterpret_cast<StrideB *>(carve(G * sizeof(StrideB)));
  auto *stride_C_d = reinterpret_cast<StrideC *>(carve(G * sizeof(StrideC)));
  auto *stride_D_d = reinterpret_cast<StrideD *>(carve(G * sizeof(StrideD)));
  auto *layout_SFA_d = reinterpret_cast<LayoutSFA *>(carve(G * sizeof(LayoutSFA)));
  auto *layout_SFB_d = reinterpret_cast<LayoutSFB *>(carve(G * sizeof(LayoutSFB)));

  {
    constexpr int kThreads = 128;
    const int blocks = (G + kThreads - 1) / kThreads;
    build_grouped_metadata_kernel<Cfg><<<blocks, kThreads, 0, stream>>>(
        G, d_cols, d_rows, a_rows, a_cols, a_trans, problems_d, stride_A_d, stride_B_d, stride_C_d,
        stride_D_d, layout_SFA_d, layout_SFB_d);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }

  // cuBLAS -> CUTLASS A/B swap: CUTLASS A := B operand, CUTLASS B := A operand.
  // The pointer arrays are reinterpreted (void* and the typed element pointers
  // are bit-identical); the underlying addresses are unchanged. C-style casts are
  // used for the const-qualified targets because reinterpret_cast cannot add
  // const across the extra pointer level (void** -> const T**).
  auto *a_ptr_d = (const ElementADataT **)B_ptrs;           // NOLINT
  auto *b_ptr_d = (const ElementBDataT **)A_ptrs;           // NOLINT
  auto *sfa_ptr_d = (const ElementSFT **)b_scale_inv_ptrs;  // NOLINT
  auto *sfb_ptr_d = (const ElementSFT **)a_scale_inv_ptrs;  // NOLINT
  auto *d_ptr_d = reinterpret_cast<ElementD **>(D_ptrs);
  auto *c_ptr_d = (const ElementC **)C_ptrs;  // NOLINT
  const float *const *alpha_ptr_array_d = alpha_ptrs;
  const float *const *beta_ptr_array_d = beta_ptrs;  // nullptr => overwrite
  const bool per_group_beta = (beta_ptrs != nullptr);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count = transformer_engine::cuda::sm_count(device);

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;  // overridden per-group by alpha_ptr_array
  // Scalar beta is 0 for both modes; when per_group_beta the real beta[g] comes
  // from beta_ptr_array (D = alpha[g]*acc + beta[g]*C). Overwrite keeps ptr_C
  // null so C is never loaded regardless of beta.
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = alpha_ptr_array_d;
  fusion_args.beta_ptr_array = per_group_beta ? beta_ptr_array_d : nullptr;
  fusion_args.dAlpha = {cute_::_0{}, cute_::_0{}, 0};  // one scalar per group
  fusion_args.dBeta = {cute_::_0{}, cute_::_0{}, 0};

  // ptr_C is read only in the per-group-beta (accumulate) mode; pass the C
  // buffers (== D for in-place wgrad). Overwrite passes null so uninitialized D
  // is never loaded.
  const ElementC **ptr_C = per_group_beta ? c_ptr_d : nullptr;

  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {G, problems_d, /*host_problem_shapes=*/nullptr},
      {a_ptr_d, stride_A_d, b_ptr_d, stride_B_d, sfa_ptr_d, layout_SFA_d, sfb_ptr_d, layout_SFB_d},
      {fusion_args, ptr_C, stride_C_d, d_ptr_d, stride_D_d},
      hw_info};

  // CUTLASS GEMM workspace is carved from the tail of the same caller-provided
  // buffer (after the metadata scratch above). No allocation happens in TE
  // common -- upstream (PyTorch) owns the buffer -- so this stays graph-safe.
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  off = align256(off);
  NVTE_CHECK(off + workspace_size <= ext_workspace_bytes,
             "CUTLASS NVFP4 grouped per-tensor GEMM: provided workspace too small (need ",
             off + workspace_size, " bytes, have ", ext_workspace_bytes, ").");
  void *cutlass_workspace = workspace_size > 0 ? static_cast<void *>(scr + off) : nullptr;

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 grouped per-tensor GEMM (grouped-tensor) cannot implement: ",
             cutlassGetStatusString(status), " (num_groups=", G, ")");

  status = gemm.initialize(arguments, cutlass_workspace, stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 grouped per-tensor GEMM (grouped-tensor) initialize failed: ",
             cutlassGetStatusString(status));

  status = gemm.run(stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 grouped per-tensor GEMM (grouped-tensor) run failed: ",
             cutlassGetStatusString(status));
}

void run_nvfp4_graph_safe_grouped_gemm(void **A_ptrs, void **B_ptrs, void **a_scale_inv_ptrs,
                                       void **b_scale_inv_ptrs, float **alpha_ptrs, void **C_ptrs,
                                       void **D_ptrs, float **beta_ptrs, const int *a_rows,
                                       const int *a_cols, const int *d_rows, const int *d_cols,
                                       bool a_trans, int num_groups, bool fp32_output,
                                       void *workspace, size_t workspace_bytes,
                                       cudaStream_t stream) {
  if (fp32_output) {
    run_impl_device<float>(A_ptrs, B_ptrs, a_scale_inv_ptrs, b_scale_inv_ptrs, alpha_ptrs, C_ptrs,
                           D_ptrs, beta_ptrs, a_rows, a_cols, d_rows, d_cols, a_trans, num_groups,
                           workspace, workspace_bytes, stream);
  } else {
    NVTE_CHECK(beta_ptrs == nullptr,
               "CUTLASS NVFP4 grouped per-tensor GEMM: per-group beta (accumulate) requires FP32 "
               "output.");
    run_impl_device<cutlass::bfloat16_t>(
        A_ptrs, B_ptrs, a_scale_inv_ptrs, b_scale_inv_ptrs, alpha_ptrs, C_ptrs, D_ptrs, beta_ptrs,
        a_rows, a_cols, d_rows, d_cols, a_trans, num_groups, workspace, workspace_bytes, stream);
  }
}

#else   // !CUTLASS_ARCH_MMA_SM100_SUPPORTED

void run_nvfp4_graph_safe_grouped_gemm(void **, void **, void **, void **, float **, void **,
                                       void **, float **, const int *, const int *, const int *,
                                       const int *, bool, int, bool, void *, size_t, cudaStream_t) {
  NVTE_ERROR(
      "CUTLASS NVFP4 grouped per-tensor GEMM requires SM100 (Blackwell). Build with "
      "sm_100a/sm_100f.");
}
#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

}  // namespace nvfp4_cutlass
}  // namespace transformer_engine
