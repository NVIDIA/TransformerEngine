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

// Query the SM count exactly once (cudaGetDeviceProperties is very slow).
static int cached_sm_count() {
  static int sm = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  return sm;
}

static inline size_t align256(size_t b) { return (b + 255) / 256 * 256; }

// Process-persistent device buffers reused across launches, to avoid per-call
// cudaMalloc/cudaFree churn. which=0 -> metadata scratch, which=1 -> CUTLASS
// workspace. Assumes grouped GEMMs are issued serially on one stream (the TE
// norm); the stream-ordered free on regrow keeps it safe under that assumption.
static void *persistent_buffer(size_t bytes, cudaStream_t stream, int which) {
  static void *bufs[2] = {nullptr, nullptr};
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

// Reusable pageable host staging buffer for the single batched H2D copy of all
// per-group metadata. Pageable (not pinned) is intentional: cudaMemcpyAsync
// stages pageable source into a driver buffer before returning, so the host
// buffer can be safely overwritten by the next launch without extra sync.
static void *persistent_host_buffer(size_t bytes) {
  static std::vector<uint8_t> buf;
  if (buf.size() < bytes) {
    buf.resize(bytes + bytes / 2);
  }
  return buf.data();
}

template <typename ElementOutT, bool kHasBias>
static void run_impl(const std::vector<const void *> &a_data,
                     const std::vector<const void *> &b_data, const std::vector<const void *> &a_sf,
                     const std::vector<const void *> &b_sf,
                     const std::vector<const float *> &alpha_ptrs,
                     const std::vector<void *> &d_ptrs, const std::vector<const void *> &bias_ptrs,
                     const std::vector<int> &Ms, const std::vector<int> &Ns,
                     const std::vector<int> &Ks, bool accumulate, cudaStream_t stream) {
  using Cfg = PerTensorCfg<ElementOutT, kHasBias>;
  using Gemm = typename Cfg::Gemm;
  using StrideA = typename Cfg::StrideA;
  using StrideB = typename Cfg::StrideB;
  using StrideC = typename Cfg::StrideC;
  using StrideD = typename Cfg::StrideD;
  using LayoutSFA = typename Cfg::LayoutSFA;
  using LayoutSFB = typename Cfg::LayoutSFB;
  using Sm1xxBlkScaledConfig = typename Cfg::Sm1xxBlkScaledConfig;
  using ElementADataT = typename Cfg::ElementADataT;
  using ElementBDataT = typename Cfg::ElementBDataT;
  using ElementSFT = typename Cfg::ElementSFT;
  using ElementC = typename Cfg::ElementC;
  using ElementD = typename Cfg::ElementD;
  using ElementBias = typename Cfg::ElementBias;
  using ProblemShape = typename Cfg::ProblemShape;

  const int G = static_cast<int>(Ms.size());

  // Host-side per-group metadata.
  std::vector<typename ProblemShape::UnderlyingProblemShape> problems(G);
  std::vector<StrideA> stride_A_h(G);
  std::vector<StrideB> stride_B_h(G);
  std::vector<StrideC> stride_C_h(G);
  std::vector<StrideD> stride_D_h(G);
  std::vector<LayoutSFA> layout_SFA_h(G);
  std::vector<LayoutSFB> layout_SFB_h(G);

  std::vector<const ElementADataT *> a_ptr_h(G);
  std::vector<const ElementBDataT *> b_ptr_h(G);
  std::vector<const ElementSFT *> sfa_ptr_h(G);
  std::vector<const ElementSFT *> sfb_ptr_h(G);
  std::vector<ElementD *> d_ptr_h(G);
  // C source pointers. For accumulate, C == D (read-modify-write main_grad).
  std::vector<const ElementC *> c_ptr_h(G);
  // Per-group bias pointers (length N each). Only populated when kHasBias.
  std::vector<const ElementBias *> bias_ptr_h(kHasBias ? G : 0);

  for (int g = 0; g < G; ++g) {
    const int M = Ms[g], N = Ns[g], K = Ks[g];
    problems[g] = {M, N, K};
    stride_A_h[g] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    stride_B_h[g] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    stride_C_h[g] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    stride_D_h[g] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    layout_SFA_h[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute_::make_shape(M, N, K, 1));
    layout_SFB_h[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute_::make_shape(M, N, K, 1));

    a_ptr_h[g] = reinterpret_cast<const ElementADataT *>(a_data[g]);
    b_ptr_h[g] = reinterpret_cast<const ElementBDataT *>(b_data[g]);
    sfa_ptr_h[g] = reinterpret_cast<const ElementSFT *>(a_sf[g]);
    sfb_ptr_h[g] = reinterpret_cast<const ElementSFT *>(b_sf[g]);
    d_ptr_h[g] = reinterpret_cast<ElementD *>(d_ptrs[g]);
    c_ptr_h[g] = reinterpret_cast<const ElementC *>(d_ptrs[g]);
    if constexpr (kHasBias) {
      bias_ptr_h[g] = reinterpret_cast<const ElementBias *>(bias_ptrs[g]);
    }
  }

  // Mirror all per-group metadata to device through ONE persistent scratch
  // buffer with a single batched H2D copy.
  const size_t need = align256(problems.size() * sizeof(problems[0])) +
                      align256(stride_A_h.size() * sizeof(StrideA)) +
                      align256(stride_B_h.size() * sizeof(StrideB)) +
                      align256(stride_C_h.size() * sizeof(StrideC)) +
                      align256(stride_D_h.size() * sizeof(StrideD)) +
                      align256(layout_SFA_h.size() * sizeof(LayoutSFA)) +
                      align256(layout_SFB_h.size() * sizeof(LayoutSFB)) +
                      align256(a_ptr_h.size() * sizeof(a_ptr_h[0])) +
                      align256(b_ptr_h.size() * sizeof(b_ptr_h[0])) +
                      align256(sfa_ptr_h.size() * sizeof(sfa_ptr_h[0])) +
                      align256(sfb_ptr_h.size() * sizeof(sfb_ptr_h[0])) +
                      align256(d_ptr_h.size() * sizeof(d_ptr_h[0])) +
                      align256(c_ptr_h.size() * sizeof(c_ptr_h[0])) +
                      align256(alpha_ptrs.size() * sizeof(alpha_ptrs[0])) +
                      align256(bias_ptr_h.size() * sizeof(const ElementBias *));

  uint8_t *scr = static_cast<uint8_t *>(persistent_buffer(need, stream, /*which=*/0));
  uint8_t *hscr = static_cast<uint8_t *>(persistent_host_buffer(need));
  size_t off = 0;
  auto put = [&](const auto &vec) {
    using T = typename std::decay_t<decltype(vec)>::value_type;
    const size_t bytes = vec.size() * sizeof(T);
    T *p = reinterpret_cast<T *>(scr + off);
    std::memcpy(hscr + off, vec.data(), bytes);
    off += align256(bytes);
    return p;
  };
  auto *problems_d = put(problems);
  auto *stride_A_d = put(stride_A_h);
  auto *stride_B_d = put(stride_B_h);
  auto *stride_C_d = put(stride_C_h);
  auto *stride_D_d = put(stride_D_h);
  auto *layout_SFA_d = put(layout_SFA_h);
  auto *layout_SFB_d = put(layout_SFB_h);
  auto *a_ptr_d = put(a_ptr_h);
  auto *b_ptr_d = put(b_ptr_h);
  auto *sfa_ptr_d = put(sfa_ptr_h);
  auto *sfb_ptr_d = put(sfb_ptr_h);
  auto *d_ptr_d = put(d_ptr_h);
  auto *c_ptr_d = put(c_ptr_h);
  // Per-group second-level scale pointer array (consumed by alpha_ptr_array).
  auto *alpha_ptr_array_d = put(alpha_ptrs);
  // Per-group bias pointer array (only staged when kHasBias).
  const ElementBias **bias_ptr_array_d = nullptr;
  if constexpr (kHasBias) {
    bias_ptr_array_d = put(bias_ptr_h);
  }
  NVTE_CHECK_CUDA(cudaMemcpyAsync(scr, hscr, off, cudaMemcpyHostToDevice, stream));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cached_sm_count();

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  if constexpr (kHasBias) {
    // Hand-built EVT args (nested), matching BiasEVT's child order:
    //   D = ElementOut( homogeneous_multiply_add(alpha[g], acc, bias[g]) ).
    // alpha[g] via scalar_ptr_array; bias[g] via the ptr-array RowBroadcast
    // (ptr_row, null_default, dRow). dRow == {} -> L-stride 0 since each
    // bias_ptr_array[g] already points at group g's length-N bias base.
    fusion_args = {
        {/*scalars=*/{}, /*scalar_ptrs=*/{}, /*scalar_ptr_arrays=*/{alpha_ptr_array_d},
         /*dScalar=*/{}},                        // alpha[g]
        {},                                      // acc
        {bias_ptr_array_d, ElementBias(0), {}},  // bias[g]
        {}                                       // homogeneous_multiply_add
    };
  } else {
    fusion_args.alpha = 1.0f;  // overridden per-group by alpha_ptr_array
    // beta == 1 -> D = alpha[g]*acc + C (accumulate into main_grad); 0 -> overwrite.
    fusion_args.beta = accumulate ? 1.0f : 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = alpha_ptr_array_d;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute_::_0{}, cute_::_0{}, 0};  // one scalar per group
    fusion_args.dBeta = {cute_::_0{}, cute_::_0{}, 0};
  }

  // ptr_C is only read when beta != 0 (no-bias accumulate); pass D's buffers so
  // accumulate is in-place. The bias path is overwrite-only (no C source).
  const ElementC **ptr_C = accumulate ? c_ptr_d : nullptr;

  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {G, problems_d, /*host_problem_shapes=*/nullptr},
      {a_ptr_d, stride_A_d, b_ptr_d, stride_B_d, sfa_ptr_d, layout_SFA_d, sfb_ptr_d, layout_SFB_d},
      {fusion_args, ptr_C, stride_C_d, d_ptr_d, stride_D_d},
      hw_info};

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    workspace = persistent_buffer(workspace_size, stream, /*which=*/1);
  }

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 grouped per-tensor GEMM cannot implement: ",
             cutlassGetStatusString(status), " (num_groups=", G, ")");

  status = gemm.initialize(arguments, workspace, stream);
  NVTE_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS NVFP4 grouped per-tensor GEMM initialize failed: ", cutlassGetStatusString(status));

  status = gemm.run(stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 grouped per-tensor GEMM run failed: ", cutlassGetStatusString(status));
}

void run_grouped_per_tensor_gemm(
    const std::vector<const void *> &a_data, const std::vector<const void *> &b_data,
    const std::vector<const void *> &a_sf, const std::vector<const void *> &b_sf,
    const std::vector<const float *> &alpha_ptrs, const std::vector<void *> &d_ptrs,
    const std::vector<const void *> &bias_ptrs, const std::vector<int> &Ms,
    const std::vector<int> &Ns, const std::vector<int> &Ks, bool fp32_output, bool accumulate,
    cudaStream_t stream) {
  static const std::vector<const void *> kNoBias;
  const bool has_bias = !bias_ptrs.empty();
  if (has_bias) {
    // Fused per-group bias is fprop-only: BF16 output, overwrite (no accumulate).
    NVTE_CHECK(!fp32_output && !accumulate,
               "CUTLASS NVFP4 grouped per-tensor GEMM: fused bias requires BF16 output and "
               "overwrite (no accumulate).");
    run_impl<cutlass::bfloat16_t, /*kHasBias=*/true>(a_data, b_data, a_sf, b_sf, alpha_ptrs, d_ptrs,
                                                     bias_ptrs, Ms, Ns, Ks, accumulate, stream);
  } else if (fp32_output) {
    run_impl<float, /*kHasBias=*/false>(a_data, b_data, a_sf, b_sf, alpha_ptrs, d_ptrs, kNoBias, Ms,
                                        Ns, Ks, accumulate, stream);
  } else {
    NVTE_CHECK(!accumulate,
               "CUTLASS NVFP4 grouped per-tensor GEMM: accumulate requires FP32 output.");
    run_impl<cutlass::bfloat16_t, /*kHasBias=*/false>(
        a_data, b_data, a_sf, b_sf, alpha_ptrs, d_ptrs, kNoBias, Ms, Ns, Ks, accumulate, stream);
  }
}

#else   // !CUTLASS_ARCH_MMA_SM100_SUPPORTED

void run_grouped_per_tensor_gemm(const std::vector<const void *> &,
                                 const std::vector<const void *> &,
                                 const std::vector<const void *> &,
                                 const std::vector<const void *> &,
                                 const std::vector<const float *> &, const std::vector<void *> &,
                                 const std::vector<const void *> &, const std::vector<int> &,
                                 const std::vector<int> &, const std::vector<int> &, bool, bool,
                                 cudaStream_t) {
  NVTE_ERROR(
      "CUTLASS NVFP4 grouped per-tensor GEMM requires SM100 (Blackwell). Build with "
      "sm_100a/sm_100f.");
}
#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

}  // namespace nvfp4_cutlass
}  // namespace transformer_engine
