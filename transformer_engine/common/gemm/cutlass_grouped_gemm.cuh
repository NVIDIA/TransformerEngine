/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

//
// Copyright (c) 2025 Shopee Inc. All Rights Reserved.
//

/**
 * @file: cutlass_grouped_gemm.cuh
 * @author: min.yang@shopee.com, yangfan.bai@shopee.com, finch.li@shopee.com
 * @date: 2025-08-08 16:20:00
 * @brief: cutlass group gemm kernel.
 **/

#pragma once

#include <transformer_engine/transformer_engine.h>

#include <atomic>
#include <cub/cub.cuh>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "../util/logging.h"
#include "common/util/system.h"
#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

namespace transformer_engine {
namespace grouped_gemm {

template <bool trans_a>
using GroupedGemmInputALayout =
    std::conditional_t<trans_a, ::cutlass::layout::ColumnMajor, ::cutlass::layout::RowMajor>;

template <bool trans_b>
using GroupedGemmInputBLayout =
    std::conditional_t<trans_b, ::cutlass::layout::ColumnMajor, ::cutlass::layout::RowMajor>;

using ProblemShapeType = cute::Shape<int, int, int>;
using ProblemShape = cutlass::gemm::GroupProblemShape<ProblemShapeType>;  // <M,N,K> per group
template <typename ScheduleConfig>
struct GemmGivenSchedule {
  using ElementA = typename ScheduleConfig::DataType;  // Element type for A matrix operand
  using ElementB = typename ScheduleConfig::DataType;  // Element type for B matrix operand
  using ElementC = typename ScheduleConfig::DataType;  // Element type for C and D matrix operands

  // A matrix configuration
  using LayoutA = typename ScheduleConfig::LayoutA;  // Layout type for A matrix operand
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<
                ElementA>::value;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using LayoutB = typename ScheduleConfig::LayoutB;  // Layout type for B matrix operand
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<
                ElementB>::value;  // Alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using LayoutC = typename ScheduleConfig::LayoutC;  // Layout type for C and D matrix operands
  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<
                ElementC>::value;  // Alignment of C matrix in units of elements (up to 16 bytes)

  // Core kernel configurations
  using ElementAccumulator = float;  // Element type for internal accumulation
  using ArchTag =
      typename ScheduleConfig::ArchTag;  // SM90 (Hopper) or SM100 (Blackwell), from ScheduleConfig
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based on the tile size

  using TileShape = typename ScheduleConfig::TileShape;  // Threadblock-level tile size
  using ClusterShape =
      typename ScheduleConfig::ClusterShape;  // Shape of the threadblocks in a cluster
  using KernelSchedule = typename ScheduleConfig::KernelSchedule;      // Kernel to launch
  using EpilogueSchedule = typename ScheduleConfig::EpilogueSchedule;  // Epilogue to launch

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC, EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// kSm100=false -> Hopper (SM90) Ptr-Array TMA warp-specialized Pingpong (original path).
// kSm100=true  -> Blackwell (SM100) Ptr-Array TMA warp-specialized tcgen05 UMMA. Two schedule
//                 families are wired through Sm100ScheduleSelector:
//   - 2SM (KernelPtrArrayTmaWarpSpecialized2SmSm100 + PtrArrayTmaWarpSpecialized2Sm, Cluster<2,1,1>)
//     used for tile_id=0/1/2 with TileM=256. Optimal at M-per-expert >= 256 (B300/B200 4K-MoE).
//   - 1SM (KernelPtrArrayTmaWarpSpecialized1SmSm100 + PtrArrayTmaWarpSpecialized1Sm, Cluster<1,1,1>)
//     used for tile_id=3 with TileM=128. At per-expert M=96 (Case 7: hidden=2048, ffn=512, EP=8,
//     MBS=4, GBS=8192, topk=12, 256 experts -> M=96), the 2SM cluster's effective M-tile = 512
//     wastes 81% of M; the 1SM TileM=128 cuts the waste to 25%. Empirically the right pick when
//     M_per_expert < 256.
//
// kTileId variants (cluster + tile bundled via Sm100ScheduleSelector):
//   0 = 2SM 256x256x64 cluster<2,1,1>  (default; best at M>=256, large-N)
//   1 = 2SM 256x128x64 cluster<2,1,1>  (less N-tail waste at small N)
//   2 = 2SM 256x192x64 cluster<2,1,1>  (quack-style finer N-tile; defined only)
//   3 = 1SM 128x256x64 cluster<1,1,1>  (B200 small-M: per-expert M < 256)
// Non-SM100 ignores kTileId (Hopper 128x128x128).
template <bool kSm100, int kTileId>
struct Sm100ScheduleSelector {
  using TileShape = cute::Shape<cute::_256, cute::_256, cute::_64>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};
template <int kTileId>
struct Sm100ScheduleSelector<false, kTileId> {
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
};
template <>
struct Sm100ScheduleSelector<true, 1> {
  using TileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};
template <>
struct Sm100ScheduleSelector<true, 2> {
  using TileShape = cute::Shape<cute::_256, cute::_192, cute::_64>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};
// tile_id=3: B200 small-M (per-expert M < 256). 1-SM schedule, TileM=128, cluster<1,1,1>.
template <>
struct Sm100ScheduleSelector<true, 3> {
  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

template <typename DataType_, bool trans_a, bool trans_b, bool kSm100 = false, int kTileId = 0>
struct ScheduleConfig {
  using ArchTag = std::conditional_t<kSm100, cutlass::arch::Sm100, cutlass::arch::Sm90>;
  using Sel = Sm100ScheduleSelector<kSm100, kTileId>;
  using KernelSchedule = typename Sel::KernelSchedule;
  using EpilogueSchedule = typename Sel::EpilogueSchedule;
  using TileShape = typename Sel::TileShape;
  using ClusterShape = typename Sel::ClusterShape;

  using LayoutA = GroupedGemmInputALayout<trans_a>;
  using LayoutB = GroupedGemmInputBLayout<trans_b>;
  using LayoutC = cutlass::layout::RowMajor;
  using DataType = DataType_;
};

template <typename DataType_, bool trans_a, bool trans_b, bool kSm100 = false, int kTileId = 0>
using GemmGrouped =
    typename GemmGivenSchedule<ScheduleConfig<DataType_, trans_a, trans_b, kSm100, kTileId>>::Gemm;

template <typename GemmT, typename ElementA, typename ElementB, typename ElementC, typename StrideA,
          typename StrideB, typename StrideC>
typename GemmT::Arguments MakeArguments(int num_experts, void* problem_sizes_host,
                                        void* problem_sizes, const ElementA** ptr_A,
                                        StrideA* stride_A, const ElementB** ptr_B,
                                        StrideB* stride_B, ElementC** ptr_C, StrideC* stride_C,
                                        float alpha, float beta, int device, int math_sm_count) {
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.

  cutlass::KernelHardwareInfo kernel_hw_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename GemmT::GemmKernel>(
          device, math_sm_count);

  typename GemmT::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;

  fusion_args.alpha = alpha;
  fusion_args.beta = beta;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  // Single alpha and beta for all groups
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  arguments =
      typename GemmT::Arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                {num_experts, reinterpret_cast<ProblemShapeType*>(problem_sizes),
                                 reinterpret_cast<ProblemShapeType const*>(problem_sizes_host)},
                                {ptr_A, stride_A, ptr_B, stride_B},
                                {
                                    fusion_args,
                                    (beta > 0.0) ? (const ElementC**)ptr_C : nullptr,  // NOLINT(*)
                                    stride_C,
                                    ptr_C,
                                    stride_C,
                                },
                                kernel_hw_info};

  return arguments;
}

template <typename T>
inline __device__ __host__ T ROUND_UP(T m, T n) {
  return (m + n - 1) / n * n;
}

template <typename T>
void debug_type() {
  std::cout << typeid(T).name() << std::endl;
}

int64_t inline getGemmCoordSize(int64_t num_gemms) {
  return (int64_t)(ROUND_UP(num_gemms * sizeof(ProblemShapeType), 128UL));
}

int64_t inline getPtrSize(int64_t num_gemms) {
  return (int64_t)(ROUND_UP(num_gemms * sizeof(half*), 128UL));
}

int64_t inline getLddSize(int64_t num_gemms) {
  return (int64_t)(ROUND_UP(num_gemms * sizeof(int64_t), 128UL));
}

// Per grouped-GEMM host staging slot. Holds problem_sizes + per-expert ptr/stride arrays for the
// host-loop launchers (~num_gemms * 60 B); 64 KB fits ~1000 experts (Case 7=32, Case 10=256) with
// ample headroom. Slots are intentionally SMALL so the ring can be DEEP (depth, not slot size, is
// what must cover the launch-ahead -- see kHostRingSlots).
static constexpr size_t kHostSlotSize = 64 * 1024;
// RING DEPTH (double-buffering). The host staging buffer's ONLY consumer is the async H2D copy
// (cudaMemcpyAsync from pinned host -> device); the device buffer that the kernel reads during run
// is protected by stream ordering (the next call's copy is enqueued after this call's kernel). The
// HOST fill, however, is out-of-band (plain CPU writes, not stream-ordered), so with a SINGLE buffer
// the next call's fill overwrites a slot whose copy is still pending -> corrupt sizes/ptrs to device
// -> CUTLASS illegal/misaligned memory or NaN grads (Case 7/10; hidden under CUDA_LAUNCH_BLOCKING).
// Giving each call its OWN rotating slot lets the prior copy drain slot K while the next fills slot
// K+1 -> no race, NO cudaStreamSynchronize. The DEPTH must exceed the CPU's launch-ahead: the eager
// backward enqueues ~6 grouped GEMMs/MoE-layer * ~42 layers ~= 250+/iter before the GPU drains them
// (depth 64 was too shallow -> wrapped mid-iter -> NaN at iter 4). 1024 slots cover a full iter's
// launch-ahead with large margin (also >= the CUDA kernel launch-queue bound on in-flight ops).
static constexpr int kHostRingSlots = 1024;
static constexpr size_t kCPUWorkSpaceSize =
    kHostSlotSize * kHostRingSlots;  // 64 MB pinned (one-time)

static char* getHostWorkspace() {
  static std::once_flag flag;
  static std::shared_ptr<char> workspace;
  static std::atomic<uint64_t> ring_idx{0};

  std::call_once(flag, [&]() {
    // PINNED (page-locked) host memory. The per-expert pointer/problem-size arrays staged here are
    // copied to device via cudaMemcpyAsync; from PAGEABLE memory that copy implicitly SYNCHRONIZES
    // (it stages through an internal pinned bounce buffer), blocking the host ~29us/call. Pinning
    // makes the H2D copy truly asynchronous, removing that per-call CPU stall.
    char* raw = nullptr;
    if (cudaMallocHost(reinterpret_cast<void**>(&raw), kCPUWorkSpaceSize) != cudaSuccess || !raw) {
      throw std::bad_alloc();
    }
    workspace = std::shared_ptr<char>(raw, [](char* p) {
      if (p) cudaFreeHost(p);
    });
  });

  // Hand out the next slot round-robin (ring buffer). See kHostRingSlots above for why this removes
  // the host-buffer reuse race without a per-call stream sync.
  const uint64_t slot = ring_idx.fetch_add(1, std::memory_order_relaxed) % kHostRingSlots;
  return workspace.get() + slot * kHostSlotSize;
}

template <bool trans_a, bool trans_b, typename Element, bool kSm100 = false>
void CutlassGroupedGemm(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                        NVTETensor* workspace, float alpha, float beta, int num_gemms,
                        cudaStream_t stream, int device, int math_sm_count) {
  using Gemm = GemmGrouped<Element, trans_a, trans_b, kSm100>;
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  typename Gemm::Arguments arguments;
  size_t kernel_workspace_size = Gemm::get_workspace_size(arguments);
  auto gemm_coord_size = getGemmCoordSize(num_gemms);
  auto ptr_size = getPtrSize(num_gemms);
  auto ldd_size = getLddSize(num_gemms);
  auto param_workspace_size = 3 * ptr_size + 3 * ldd_size + gemm_coord_size;

  NVTE_CHECK(
      param_workspace_size < kCPUWorkSpaceSize,
      "Insufficient kCPUWorkSpaceSize size: required=", static_cast<int64_t>(param_workspace_size),
      ", available=", static_cast<int64_t>(kCPUWorkSpaceSize), " for CUTLASS grouped GEMM.");

  auto total_workspace_size = param_workspace_size + kernel_workspace_size;
  transformer_engine::Tensor* wspace = transformer_engine::convertNVTETensor(workspace[0]);

  NVTE_CHECK(total_workspace_size < wspace->numel(), "Insufficient workspace[0] size: required=",
             static_cast<int64_t>(total_workspace_size),
             ", available=", static_cast<int64_t>(wspace->numel()), " for CUTLASS grouped GEMM.");

  char* workspace_ptr = reinterpret_cast<char*>(wspace->data.dptr);

  char* kernel_workspace_ptr = nullptr;

  char* host_workspace = getHostWorkspace();

  ProblemShapeType* problem_sizes_host = reinterpret_cast<ProblemShapeType*>(host_workspace);

  ElementA** ptr_A_host = reinterpret_cast<ElementA**>(host_workspace + gemm_coord_size);
  ElementB** ptr_B_host = reinterpret_cast<ElementB**>(host_workspace + gemm_coord_size + ptr_size);
  ElementC** ptr_C_host =
      reinterpret_cast<ElementC**>(host_workspace + gemm_coord_size + 2 * ptr_size);
  int64_t* lda_host =
      reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 3 * ptr_size + 0 * ldd_size);
  int64_t* ldb_host =
      reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 3 * ptr_size + 1 * ldd_size);
  int64_t* ldc_host =
      reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 3 * ptr_size + 2 * ldd_size);

  for (size_t i = 0; i < num_gemms; i++) {
    const transformer_engine::Tensor* inputA = transformer_engine::convertNVTETensorCheck(A[i]);
    const transformer_engine::Tensor* inputB = transformer_engine::convertNVTETensorCheck(B[i]);
    transformer_engine::Tensor* outputD = transformer_engine::convertNVTETensor(D[i]);

    const int m = trans_a ? inputA->data.shape[1] : inputA->data.shape[0];
    const int k = trans_a ? inputA->data.shape[0] : inputA->data.shape[1];
    const int n = trans_b ? inputB->data.shape[0] : inputB->data.shape[1];

    auto problem = ProblemShapeType(m, n, k);
    problem_sizes_host[i] = problem;

    ptr_A_host[i] = reinterpret_cast<ElementA*>(inputA->data.dptr);
    ptr_B_host[i] = reinterpret_cast<ElementB*>(inputB->data.dptr);
    ptr_C_host[i] = reinterpret_cast<ElementC*>(outputD->data.dptr);

    lda_host[i] = LayoutA::packed({m, k}).stride(0);
    ldb_host[i] = LayoutB::packed({k, n}).stride(0);
    ldc_host[i] = LayoutC::packed({m, n}).stride(0);
  }

  cudaMemcpyAsync(workspace_ptr, host_workspace, param_workspace_size, cudaMemcpyHostToDevice,
                  stream);

  char* param_workspace_ptr = workspace_ptr;
  ProblemShapeType* problem_sizes_device = reinterpret_cast<ProblemShapeType*>(param_workspace_ptr);
  const ElementA** ptr_A = reinterpret_cast<const ElementA**>(
      reinterpret_cast<char*>(param_workspace_ptr) + gemm_coord_size);
  const ElementB** ptr_B = reinterpret_cast<const ElementB**>(
      reinterpret_cast<char*>(param_workspace_ptr) + gemm_coord_size + 1 * ptr_size);
  ElementC** ptr_C = reinterpret_cast<ElementC**>(reinterpret_cast<char*>(param_workspace_ptr) +
                                                  gemm_coord_size + 2 * ptr_size);

  StrideA* lda = reinterpret_cast<StrideA*>(reinterpret_cast<char*>(param_workspace_ptr) +
                                            gemm_coord_size + 3 * ptr_size + 0 * ldd_size);
  StrideB* ldb = reinterpret_cast<StrideB*>(reinterpret_cast<char*>(param_workspace_ptr) +
                                            gemm_coord_size + 3 * ptr_size + 1 * ldd_size);
  StrideC* ldc = reinterpret_cast<StrideC*>(reinterpret_cast<char*>(param_workspace_ptr) +
                                            gemm_coord_size + 3 * ptr_size + 2 * ldd_size);

  kernel_workspace_ptr = workspace_ptr + param_workspace_size;

  arguments = MakeArguments<Gemm, ElementA, ElementB, ElementC, StrideA, StrideB, StrideC>(
      num_gemms, problem_sizes_host, problem_sizes_device, ptr_A, lda, ptr_B, ldb, ptr_C, ldc,
      alpha, beta, device, math_sm_count);

  Gemm gemm;

  // Check can implement the kernel.
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    NVTE_ERROR("Failed to implement CUTLASS Grouped GEMM with ", num_gemms, " GEMMs");
  }

  // Initialize the kernel.
  if (gemm.initialize(arguments, kernel_workspace_ptr) != cutlass::Status::kSuccess) {
    NVTE_ERROR("Failed to initialize CUTLASS Grouped GEMM with ", num_gemms, " GEMMs");
  }

  // Execute the kernel in the current stream.
  if (gemm.run(stream) != cutlass::Status::kSuccess) {
    NVTE_ERROR("Failed to run CUTLASS Grouped GEMM with ", num_gemms, " GEMMs");
  }
}

// ---------------------------------------------------------------------------------------------------
// SonicMoE: CUTLASS grouped GEMM driven by the GROUPED-TENSOR path's ON-DEVICE per-expert arrays
// (A_ptrs/B_ptrs/D_ptrs + d_rows/d_cols from setup_grouped_gemm_kernel). Unlike CutlassGroupedGemm,
// this builds NO host-side pointer/problem arrays and issues NO cudaMemcpyAsync of them -- that host
// loop + pageable H2D copy is exactly the ~115us/call CPU stall on the discrete path. Here the per-
// expert pointers are already on device, and the per-expert problem sizes + strides are packed on
// device by the small cutlass_pack_device_args kernel. problem_sizes_host is filled with the AVERAGE
// (avg_m, avg_n, K): the GroupProblemShape *host* pointer only sizes the launch/scheduler estimate
// (whose TOTAL is correct = num*avg), while the per-tile work reads the exact *device* problem sizes
// -> no D2H sync. M = d_rows (output rows), N = d_cols (output cols), K = uniform contraction.
template <typename ProblemShapeT, typename LayoutA, typename LayoutB, typename LayoutC>
__global__ void cutlass_pack_device_args(int num, const int* m_arr, const int* n_arr,
                                         const int* k_arr, ProblemShapeT* problems, int64_t* lda,
                                         int64_t* ldb, int64_t* ldc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num) return;
  const int m = m_arr[i];
  const int n = n_arr[i];
  const int k =
      k_arr[i];  // exact per-expert contraction (NOT config.avg_k, which is the cuBLAS hint).
  problems[i] = ProblemShapeT(m, n, k);
  // Mirror CutlassGroupedGemm's host stride computation (int64 leading dim, reinterpreted as Stride*).
  lda[i] = LayoutA::packed({m, k}).stride(0);
  ldb[i] = LayoutB::packed({k, n}).stride(0);
  ldc[i] = LayoutC::packed({m, n}).stride(0);
}

template <bool trans_a, bool trans_b, typename Element, bool kSm100 = false, int kTileId = 0>
void CutlassGroupedGemmDevice(void** A_ptrs, void** B_ptrs, void** D_ptrs, const int* m_arr,
                              const int* n_arr, const int* k_arr, int avg_k, int num_gemms,
                              void* workspace_ptr_raw, size_t workspace_bytes, float alpha,
                              float beta, int avg_m, int avg_n, cudaStream_t stream, int device,
                              int math_sm_count) {
  using Gemm = GemmGrouped<Element, trans_a, trans_b, kSm100, kTileId>;
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  typename Gemm::Arguments arguments;
  size_t kernel_workspace_size = Gemm::get_workspace_size(arguments);
  auto gemm_coord_size = getGemmCoordSize(num_gemms);
  auto ldd_size = getLddSize(num_gemms);
  // Device param workspace: problem_sizes + 3 stride arrays. NO pointer arrays (the on-device A/B/D
  // pointer arrays from the grouped setup are passed straight through to CUTLASS).
  auto param_workspace_size = gemm_coord_size + 3 * ldd_size;
  auto total_workspace_size = param_workspace_size + kernel_workspace_size;

  NVTE_CHECK(total_workspace_size < workspace_bytes,
             "Insufficient workspace for CUTLASS device grouped GEMM: required=",
             static_cast<int64_t>(total_workspace_size),
             ", available=", static_cast<int64_t>(workspace_bytes));
  char* workspace_ptr = reinterpret_cast<char*>(workspace_ptr_raw);

  (void)avg_m;
  (void)avg_n;
  (void)avg_k;
  ProblemShapeType* problem_sizes_host = nullptr;

  // Device param arrays (packed on device below).
  ProblemShapeType* problem_sizes_device = reinterpret_cast<ProblemShapeType*>(workspace_ptr);
  int64_t* lda64 = reinterpret_cast<int64_t*>(workspace_ptr + gemm_coord_size + 0 * ldd_size);
  int64_t* ldb64 = reinterpret_cast<int64_t*>(workspace_ptr + gemm_coord_size + 1 * ldd_size);
  int64_t* ldc64 = reinterpret_cast<int64_t*>(workspace_ptr + gemm_coord_size + 2 * ldd_size);

  constexpr int kBlock = 128;
  int grid = (num_gemms + kBlock - 1) / kBlock;
  cutlass_pack_device_args<ProblemShapeType, LayoutA, LayoutB, LayoutC>
      <<<grid, kBlock, 0, stream>>>(num_gemms, m_arr, n_arr, k_arr, problem_sizes_device, lda64,
                                    ldb64, ldc64);

  StrideA* lda = reinterpret_cast<StrideA*>(lda64);
  StrideB* ldb = reinterpret_cast<StrideB*>(ldb64);
  StrideC* ldc = reinterpret_cast<StrideC*>(ldc64);
  const ElementA** ptr_A = const_cast<const ElementA**>(reinterpret_cast<ElementA**>(A_ptrs));
  const ElementB** ptr_B = const_cast<const ElementB**>(reinterpret_cast<ElementB**>(B_ptrs));
  ElementC** ptr_C = reinterpret_cast<ElementC**>(D_ptrs);

  char* kernel_workspace_ptr = workspace_ptr + param_workspace_size;

  arguments = MakeArguments<Gemm, ElementA, ElementB, ElementC, StrideA, StrideB, StrideC>(
      num_gemms, problem_sizes_host, problem_sizes_device, ptr_A, lda, ptr_B, ldb, ptr_C, ldc,
      alpha, beta, device, math_sm_count);

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    NVTE_ERROR("CUTLASS device grouped GEMM: can_implement failed (", num_gemms, " groups)");
  }
  if (gemm.initialize(arguments, kernel_workspace_ptr) != cutlass::Status::kSuccess) {
    NVTE_ERROR("CUTLASS device grouped GEMM: initialize failed (", num_gemms, " groups)");
  }
  if (gemm.run(stream) != cutlass::Status::kSuccess) {
    NVTE_ERROR("CUTLASS device grouped GEMM: run failed (", num_gemms, " groups)");
  }
}

// kBigN selects the SM100 wgrad N-tile: false=256x128x64 (best for small-K, latency-bound),
// true=256x256x64 (best for large-K). Chosen at runtime by average K (see cutlass_grouped_gemm.cu).
template <bool trans_a, bool trans_b, typename ElementD = float, bool kSm100 = false,
          bool kBigN = false>
struct GemmGivenScheduleWgrad;

// Base config shared by both FP32 and BF16 output specialisations.
// Subclasses override TileShape / ClusterShape / KernelSchedule / EpilogueSchedule.
template <bool trans_a, bool trans_b, typename ElementD>
struct GemmGivenScheduleWgradBase {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementC = ElementD;
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using LayoutA = GroupedGemmInputALayout<trans_a>;
  using LayoutB = GroupedGemmInputBLayout<trans_b>;
  using LayoutC = cutlass::layout::RowMajor;
  // TMA minimum 16 B: 8×BF16 or 4×FP32.
  static constexpr int AlignmentA = 8;
  static constexpr int AlignmentB = 8;
  static constexpr int AlignmentC = static_cast<int>(16 / sizeof(ElementD));
};

// FP32 output: Cooperative 128×128×64, ClusterShape 1×1×1.
// Two warpgroups keep both the MMA pipeline and the FP32 epilogue busy.
template <bool trans_a, bool trans_b, bool kSm100, bool kBigN>
struct GemmGivenScheduleWgrad<trans_a, trans_b, float, kSm100, kBigN>
    : GemmGivenScheduleWgradBase<trans_a, trans_b, float> {
  using Base = GemmGivenScheduleWgradBase<trans_a, trans_b, float>;
  using ElementD = float;
  using ElementC = float;
  using ElementAccumulator = float;
  using ArchTag = std::conditional_t<kSm100, cutlass::arch::Sm100, cutlass::arch::Sm90>;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using LayoutA = typename Base::LayoutA;
  using LayoutB = typename Base::LayoutB;
  using LayoutC = typename Base::LayoutC;
  static constexpr int AlignmentA = Base::AlignmentA;
  static constexpr int AlignmentB = Base::AlignmentB;
  static constexpr int AlignmentC = Base::AlignmentC;

  // SM90: Cooperative 128x128x64. SM100: 256x128 (small-K) or 256x256 (large-K), by kBigN.
  using TileShape =
      std::conditional_t<kSm100,
                         std::conditional_t<kBigN, cute::Shape<cute::_256, cute::_256, cute::_64>,
                                            cute::Shape<cute::_256, cute::_128, cute::_64>>,
                         cute::Shape<cute::_128, cute::_128, cute::_64>>;
  using ClusterShape = std::conditional_t<kSm100, cute::Shape<cute::_2, cute::_1, cute::_1>,
                                          cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using KernelSchedule =
      std::conditional_t<kSm100, cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100,
                         cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
  using EpilogueSchedule =
      std::conditional_t<kSm100, cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm,
                         cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC, ElementD, LayoutC*, AlignmentC, EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator>>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, typename Base::ElementA, LayoutA*, AlignmentA,
      typename Base::ElementB, LayoutB*, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// BF16-output specialization: TileShape 128x128x128, ClusterShape 1x2x1, Ptr-Array TMA
// warp-specialized Pingpong schedule (SM90). The 8-element (kWgradMinAlign) alignment on the
// expert/hidden dims is validated before launch; any remaining tile/shape constraints are
// enforced by the kernel's can_implement check inside CutlassGroupedGemmWgrad.
template <bool trans_a, bool trans_b, bool kSm100, bool kBigN>
struct GemmGivenScheduleWgrad<trans_a, trans_b, cutlass::bfloat16_t, kSm100, kBigN>
    : GemmGivenScheduleWgradBase<trans_a, trans_b, cutlass::bfloat16_t> {
  using Base = GemmGivenScheduleWgradBase<trans_a, trans_b, cutlass::bfloat16_t>;
  using ElementD = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ArchTag = std::conditional_t<kSm100, cutlass::arch::Sm100, cutlass::arch::Sm90>;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using LayoutA = typename Base::LayoutA;
  using LayoutB = typename Base::LayoutB;
  using LayoutC = typename Base::LayoutC;
  static constexpr int AlignmentA = Base::AlignmentA;
  static constexpr int AlignmentB = Base::AlignmentB;
  static constexpr int AlignmentC = Base::AlignmentC;

  // SM90: Pingpong 128x128x128. SM100: 256x128 (small-K) or 256x256 (large-K), by kBigN.
  using TileShape =
      std::conditional_t<kSm100,
                         std::conditional_t<kBigN, cute::Shape<cute::_256, cute::_256, cute::_64>,
                                            cute::Shape<cute::_256, cute::_128, cute::_64>>,
                         cute::Shape<cute::_128, cute::_128, cute::_128>>;
  using ClusterShape = std::conditional_t<kSm100, cute::Shape<cute::_2, cute::_1, cute::_1>,
                                          cute::Shape<cute::_1, cute::_2, cute::_1>>;
  using KernelSchedule =
      std::conditional_t<kSm100, cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100,
                         cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>;
  using EpilogueSchedule =
      std::conditional_t<kSm100, cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm,
                         cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC, ElementD, LayoutC*, AlignmentC, EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator>>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, typename Base::ElementA, LayoutA*, AlignmentA,
      typename Base::ElementB, LayoutB*, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <bool trans_a, bool trans_b, typename ElementD = float, bool kSm100 = false,
          bool kBigN = false>
using GemmGroupedWgrad =
    typename GemmGivenScheduleWgrad<trans_a, trans_b, ElementD, kSm100, kBigN>::Gemm;

template <bool trans_a, bool trans_b, typename ElementD = float, bool kSm100 = false,
          bool kBigN = false>
void CutlassGroupedGemmWgrad(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                             NVTETensor* workspace, float alpha, float beta, int num_gemms,
                             cudaStream_t stream, int device, int math_sm_count) {
  using Config = GemmGivenScheduleWgrad<trans_a, trans_b, ElementD, kSm100, kBigN>;
  using Gemm = GemmGroupedWgrad<trans_a, trans_b, ElementD, kSm100, kBigN>;
  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutC = typename Config::LayoutC;
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementC = typename Config::ElementC;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  typename Gemm::Arguments arguments;
  const size_t kernel_workspace_size = Gemm::get_workspace_size(arguments);
  const auto gemm_coord_size = getGemmCoordSize(num_gemms);
  const auto ptr_size = getPtrSize(num_gemms);
  const auto ldd_size = getLddSize(num_gemms);
  const auto param_workspace_size = 3 * ptr_size + 3 * ldd_size + gemm_coord_size;

  NVTE_CHECK(param_workspace_size < kCPUWorkSpaceSize,
             "Insufficient kCPUWorkSpaceSize for wgrad grouped GEMM: required=",
             static_cast<int64_t>(param_workspace_size));

  const auto total_workspace_size = param_workspace_size + kernel_workspace_size;
  transformer_engine::Tensor* wspace = transformer_engine::convertNVTETensor(workspace[0]);

  NVTE_CHECK(total_workspace_size < wspace->numel(),
             "Insufficient workspace[0] for wgrad grouped GEMM: required=",
             static_cast<int64_t>(total_workspace_size),
             ", available=", static_cast<int64_t>(wspace->numel()));

  char* workspace_ptr = reinterpret_cast<char*>(wspace->data.dptr);
  char* host_workspace = getHostWorkspace();

  auto* problem_sizes_host = reinterpret_cast<ProblemShapeType*>(host_workspace);
  auto* ptr_A_host = reinterpret_cast<ElementA**>(host_workspace + gemm_coord_size);
  auto* ptr_B_host = reinterpret_cast<ElementB**>(host_workspace + gemm_coord_size + ptr_size);
  auto* ptr_C_host = reinterpret_cast<ElementC**>(host_workspace + gemm_coord_size + 2 * ptr_size);
  auto* lda_host = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 3 * ptr_size);
  auto* ldb_host =
      reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 3 * ptr_size + ldd_size);
  auto* ldc_host =
      reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 3 * ptr_size + 2 * ldd_size);

  for (int i = 0; i < num_gemms; i++) {
    const auto* inputA = transformer_engine::convertNVTETensorCheck(A[i]);
    const auto* inputB = transformer_engine::convertNVTETensorCheck(B[i]);
    auto* outputD = transformer_engine::convertNVTETensor(D[i]);

    const int m =
        trans_a ? static_cast<int>(inputA->data.shape[1]) : static_cast<int>(inputA->data.shape[0]);
    const int k =
        trans_a ? static_cast<int>(inputA->data.shape[0]) : static_cast<int>(inputA->data.shape[1]);
    const int n =
        trans_b ? static_cast<int>(inputB->data.shape[0]) : static_cast<int>(inputB->data.shape[1]);

    problem_sizes_host[i] = ProblemShapeType(m, n, k);
    ptr_A_host[i] = reinterpret_cast<ElementA*>(inputA->data.dptr);
    ptr_B_host[i] = reinterpret_cast<ElementB*>(inputB->data.dptr);
    ptr_C_host[i] = reinterpret_cast<ElementC*>(outputD->data.dptr);
    lda_host[i] = LayoutA::packed({m, k}).stride(0);
    ldb_host[i] = LayoutB::packed({k, n}).stride(0);
    ldc_host[i] = LayoutC::packed({m, n}).stride(0);
  }

  cudaMemcpyAsync(workspace_ptr, host_workspace, param_workspace_size, cudaMemcpyHostToDevice,
                  stream);

  auto* problem_sizes_device = reinterpret_cast<ProblemShapeType*>(workspace_ptr);
  const ElementA** ptr_A = reinterpret_cast<const ElementA**>(workspace_ptr + gemm_coord_size);
  const ElementB** ptr_B =
      reinterpret_cast<const ElementB**>(workspace_ptr + gemm_coord_size + ptr_size);
  ElementC** ptr_C = reinterpret_cast<ElementC**>(workspace_ptr + gemm_coord_size + 2 * ptr_size);
  auto* lda = reinterpret_cast<StrideA*>(workspace_ptr + gemm_coord_size + 3 * ptr_size);
  auto* ldb = reinterpret_cast<StrideB*>(workspace_ptr + gemm_coord_size + 3 * ptr_size + ldd_size);
  auto* ldc =
      reinterpret_cast<StrideC*>(workspace_ptr + gemm_coord_size + 3 * ptr_size + 2 * ldd_size);

  char* kernel_workspace_ptr = workspace_ptr + param_workspace_size;

  arguments = MakeArguments<Gemm, ElementA, ElementB, ElementC, StrideA, StrideB, StrideC>(
      num_gemms, problem_sizes_host, problem_sizes_device, ptr_A, lda, ptr_B, ldb, ptr_C, ldc,
      alpha, beta, device, math_sm_count);

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    NVTE_ERROR("Wgrad grouped GEMM: can_implement check failed (", num_gemms, " groups)");
  }
  if (gemm.initialize(arguments, kernel_workspace_ptr) != cutlass::Status::kSuccess) {
    NVTE_ERROR("Wgrad grouped GEMM: initialize failed (", num_gemms, " groups)");
  }
  if (gemm.run(stream) != cutlass::Status::kSuccess) {
    NVTE_ERROR("Wgrad grouped GEMM: run failed (", num_gemms, " groups)");
  }
}

// On-device variant of CutlassGroupedGemmWgrad: dispatches the DEDICATED wgrad kernel (GemmGroupedWgrad --
// FP32-capable epilogue, 256x128 wgrad tile, varlen-K) straight from the grouped setup's on-device
// pointer/dim arrays (no host pointer loop, no cudaMemcpyAsync of pointers). Body mirrors
// CutlassGroupedGemmDevice but over the wgrad Config. avg_m/avg_n MUST be the UNIFORM weight output dims
// (NOT the token avg, which would under-size the grid); k_arr is the per-expert RAGGED token contraction.
template <bool trans_a, bool trans_b, typename ElementD, bool kSm100 = false, bool kBigN = false>
void CutlassGroupedGemmWgradDevice(void** A_ptrs, void** B_ptrs, void** D_ptrs, const int* m_arr,
                                   const int* n_arr, const int* k_arr, int avg_k, int num_gemms,
                                   void* workspace_ptr_raw, size_t workspace_bytes, float alpha,
                                   float beta, int avg_m, int avg_n, cudaStream_t stream,
                                   int device, int math_sm_count) {
  using Config = GemmGivenScheduleWgrad<trans_a, trans_b, ElementD, kSm100, kBigN>;
  using Gemm = GemmGroupedWgrad<trans_a, trans_b, ElementD, kSm100, kBigN>;
  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutC = typename Config::LayoutC;
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementC = typename Config::ElementC;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  typename Gemm::Arguments arguments;
  size_t kernel_workspace_size = Gemm::get_workspace_size(arguments);
  auto gemm_coord_size = getGemmCoordSize(num_gemms);
  auto ldd_size = getLddSize(num_gemms);
  auto param_workspace_size = gemm_coord_size + 3 * ldd_size;
  auto total_workspace_size = param_workspace_size + kernel_workspace_size;

  NVTE_CHECK(total_workspace_size < workspace_bytes,
             "Insufficient workspace for CUTLASS device wgrad grouped GEMM: required=",
             static_cast<int64_t>(total_workspace_size),
             ", available=", static_cast<int64_t>(workspace_bytes));
  char* workspace_ptr = reinterpret_cast<char*>(workspace_ptr_raw);

  char* host_workspace = getHostWorkspace();
  ProblemShapeType* problem_sizes_host = reinterpret_cast<ProblemShapeType*>(host_workspace);
  for (int i = 0; i < num_gemms; i++) {
    problem_sizes_host[i] = ProblemShapeType(avg_m, avg_n, avg_k);
  }

  ProblemShapeType* problem_sizes_device = reinterpret_cast<ProblemShapeType*>(workspace_ptr);
  int64_t* lda64 = reinterpret_cast<int64_t*>(workspace_ptr + gemm_coord_size + 0 * ldd_size);
  int64_t* ldb64 = reinterpret_cast<int64_t*>(workspace_ptr + gemm_coord_size + 1 * ldd_size);
  int64_t* ldc64 = reinterpret_cast<int64_t*>(workspace_ptr + gemm_coord_size + 2 * ldd_size);

  constexpr int kBlock = 128;
  int grid = (num_gemms + kBlock - 1) / kBlock;
  cutlass_pack_device_args<ProblemShapeType, LayoutA, LayoutB, LayoutC>
      <<<grid, kBlock, 0, stream>>>(num_gemms, m_arr, n_arr, k_arr, problem_sizes_device, lda64,
                                    ldb64, ldc64);

  StrideA* lda = reinterpret_cast<StrideA*>(lda64);
  StrideB* ldb = reinterpret_cast<StrideB*>(ldb64);
  StrideC* ldc = reinterpret_cast<StrideC*>(ldc64);
  const ElementA** ptr_A = const_cast<const ElementA**>(reinterpret_cast<ElementA**>(A_ptrs));
  const ElementB** ptr_B = const_cast<const ElementB**>(reinterpret_cast<ElementB**>(B_ptrs));
  ElementC** ptr_C = reinterpret_cast<ElementC**>(D_ptrs);

  char* kernel_workspace_ptr = workspace_ptr + param_workspace_size;

  arguments = MakeArguments<Gemm, ElementA, ElementB, ElementC, StrideA, StrideB, StrideC>(
      num_gemms, problem_sizes_host, problem_sizes_device, ptr_A, lda, ptr_B, ldb, ptr_C, ldc,
      alpha, beta, device, math_sm_count);

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    NVTE_ERROR("CUTLASS device wgrad grouped GEMM: can_implement failed (", num_gemms, " groups)");
  }
  if (gemm.initialize(arguments, kernel_workspace_ptr) != cutlass::Status::kSuccess) {
    NVTE_ERROR("CUTLASS device wgrad grouped GEMM: initialize failed (", num_gemms, " groups)");
  }
  if (gemm.run(stream) != cutlass::Status::kSuccess) {
    NVTE_ERROR("CUTLASS device wgrad grouped GEMM: run failed (", num_gemms, " groups)");
  }
}

}  // namespace grouped_gemm
}  // namespace transformer_engine

void cutlass_grouped_gemm(const NVTETensor* A, const NVTETensor* B, NVTETensor* D, int num_gemms,
                          bool transa, bool transb, bool grad, NVTETensor* workspace,
                          bool accumulate, int device, int math_sm_count, cudaStream_t stream);

void cutlass_grouped_gemm_varlen_k(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                                   int num_gemms, bool transa, bool transb, bool grad,
                                   NVTETensor* workspace, bool accumulate, int device,
                                   int math_sm_count, cudaStream_t stream);
