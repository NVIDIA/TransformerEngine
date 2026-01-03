/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <transformer_engine/hadamard_transform.h>

#include <cuda/barrier>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>

#include "common/common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/curanddx.hpp"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"
#include "customized_pipeline.cuh"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/platform/platform.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/print_error.hpp"

// clang-format off

namespace transformer_engine {
namespace detail {
namespace {

using namespace cute;

// Ensure Tensor refers to cute::Tensor, not transformer_engine::Tensor
using cute::Tensor;

struct CLCResponse { uint32_t data[4] = {0}; };


CUTLASS_DEVICE
cutlass::Array<cutlass::float_e2m1_t, 8> StochasticNumericConverterBase(
    cutlass::Array<float, 8> const &input, cutlass::Array<uint32_t, 2> const &rbits) {
  using result_type = cutlass::Array<cutlass::float_e2m1_t, 8>;
  result_type output;
  auto output_ptr = reinterpret_cast<uint16_t *>(&output);
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    asm volatile(
        "{\n"
        "cvt.rs.satfinite.e2m1x4.f32   %0, {%5, %4, %3, %2}, %10;\n"
        "cvt.rs.satfinite.e2m1x4.f32   %1, {%9, %8, %7, %6}, %11;\n"
        "}"
        : "=h"(output_ptr[0]), "=h"(output_ptr[1])
        : "f"(input[0]), "f"(input[1]), "f"(input[2]), "f"(input[3]), "f"(input[4]), "f"(input[5]),
          "f"(input[6]), "f"(input[7]), "r"(rbits[0]), "r"(rbits[1]));
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return output;
}

CUTLASS_DEVICE
cutlass::Array<cutlass::float_e2m1_t, 16>
StochasticNumericConverter(cutlass::Array<float, 16> const &input, cutlass::Array<uint32_t, 4> const &rbits) {
  using result_type = cutlass::Array<cutlass::float_e2m1_t, 16>;
  result_type output;
  cutlass::Array<cutlass::float_e2m1_t, 8> *result_ptr = reinterpret_cast<cutlass::Array<cutlass::float_e2m1_t, 8> *>(&output);
  cutlass::Array<float, 8> const *source_ptr = reinterpret_cast<cutlass::Array<float, 8> const *>(&input);
  cutlass::Array<uint32_t, 2> const *rbits_ptr = reinterpret_cast<cutlass::Array<uint32_t, 2> const *>(&rbits);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; i++) {
    result_ptr[i] = StochasticNumericConverterBase(source_ptr[i], rbits_ptr[i]);
  }
  return output;
}

template <
  class ElementA,
  class ElementB,
  class ASmemLayout,
  class BSmemLayout,
  class ClusterShape,
  int AccumulatorPipelineStageCount_,
  int EpilogueUnrollFactor_,
  int SchedulerPipelineStageCount_>
struct SharedStorage {
  static int constexpr AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
  static int constexpr EpilogueUnrollFactor = EpilogueUnrollFactor_;
  using AtomThrShapeMNK = cute::Shape<_1, _1, _1>;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / EpilogueUnrollFactor, AtomThrShapeMNK>;
  using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;

  static int constexpr MainloopPipelineStageCount = size<3>(ASmemLayout{});
  using MainloopPipeline = cutlass::detail::CustomizedPipelineTmaUmmaAsync<
    MainloopPipelineStageCount,
    Shape<_1,_1,_1>,
    AtomThrShapeMNK>;
  using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;
  using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount_, ClusterShape>;
  using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
  using CLCThrottlePipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount_>;
  using CLCThrottlePipelineStorage = typename CLCThrottlePipeline::SharedStorage;

  struct TensorStorage : cute::aligned_struct<128, _1> {
    // cute::array_aligned<ElementA, cute::cosize_v<ASmemLayout>> smem_A;
    cute::array_aligned<ElementA, cute::cosize_v<ASmemLayout>> smem_A;
    cute::array_aligned<ElementB, cute::cosize_v<BSmemLayout>> smem_B;
  } tensors;

  alignas(16) AccumulatorPipelineStorage accumulator;
  alignas(16) MainloopPipelineStorage mainloop;
  alignas(16) cute::uint64_t tma_barrier[1];
  alignas(16) CLCPipelineStorage clc;
  alignas(16) CLCThrottlePipelineStorage clc_throttle;
  alignas(16) CLCResponse clc_response[SchedulerPipelineStageCount_];
  uint32_t tmem_base_ptr;
};

template <class MShape, class NShape, class KShape, class ClusterShape, class ClusterTileShape,
          class TA, class AStride, class ASmemLayout, class TmaLoadA,
          class TB, class BStride, class BSmemLayout, class TmaLoadB,
          class TD, class DStride, class DSmemLayout,
          class TSFD, class TSFDLayout,
          class TQA, class QAStride,
          class TSFA, class TSFALayout,
          class TiledMMA,
          int AccumulatorPipelineStageCount_,
          int SchedulerPipelineStageCount_,
          bool kEnableStochasticRounding_ = false,
          bool kEnableRHTColQuant_ = true,
          bool kEnableRowQuant_ = true,
          bool kUseFastMath_ = true>
__launch_bounds__(512, 1)
__global__ static void row_col_rht_gemm_device(
    MShape M,
    NShape N,
    KShape K,
    ClusterShape cluster_shape,
    ClusterTileShape cluster_tile,
    TA const* A,
    AStride dA,
    ASmemLayout sAlayout,
    CUTE_GRID_CONSTANT TmaLoadA const tma_load_a,
    TB const* B,
    BStride dB,
    BSmemLayout sBlayout,
    CUTE_GRID_CONSTANT TmaLoadB const tma_load_b,
    TD* D,
    DStride dD,
    DSmemLayout,
    TSFD* SFD,
    TSFDLayout sfd_layout,
    TQA* QA,
    QAStride dQA,
    TSFA* SFA,
    TSFALayout sfa_layout,
    TiledMMA mma,
    float const* a_global_amax,
    float const* c_global_amax,
    const size_t* rng_state) {
  using namespace cute;
  using X = Underscore;
  // static constexpr bool kApplyStochasticRounding = true;
  using ElementAccumulator = float;
  static int constexpr K_PIPE_MAX = size<3>(ASmemLayout{});
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMMA::ThrLayoutVMNK{})), _1, _1>;
  static uint32_t constexpr kTmaTransactionBytes = cutlass::bits_to_bytes(
      size(AtomThrShapeMNK{}) * cosize(take<0,3>(ASmemLayout{})) * cute::sizeof_bits_v<TA>);
  static constexpr bool kEnableStochasticRounding = kEnableStochasticRounding_;
  static constexpr bool kEnableRHTColQuant = kEnableRHTColQuant_;
  static constexpr bool kEnableRowQuant = kEnableRowQuant_;
  static constexpr bool kUseFastMath = kUseFastMath_;
  static int constexpr RhtTensorSize = 16;
  static int constexpr kTmaRhtTensorTransactionBytes = cutlass::bits_to_bytes(
      RhtTensorSize * RhtTensorSize * cute::sizeof_bits_v<TB>);
  static int constexpr AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
  static int constexpr SchedulerPipelineStageCount = SchedulerPipelineStageCount_;

  static int constexpr MainloopPipelineStageCount = size<3>(ASmemLayout{});
  using MainloopPipeline = cutlass::detail::CustomizedPipelineTmaUmmaAsync<
    MainloopPipelineStageCount,
    ClusterShape,
    AtomThrShapeMNK>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;
  using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;
  using CLCThrottlePipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount>;
  using CLCThrottlePipelineState = typename CLCThrottlePipeline::PipelineState;

  static_assert(ClusterShape{} == Shape<_1,_1,_1>{}, "ClusterShape must be Shape<_1,_1,_1>");

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  static int constexpr VectorSize = RhtTensorSize;
  // Preconditions
  CUTE_STATIC_ASSERT(is_static<ASmemLayout>::value);
  CUTE_STATIC_ASSERT(is_static<BSmemLayout>::value);
  CUTE_STATIC_ASSERT(is_static<DSmemLayout>::value);
  auto cluster_size = size<0>(cluster_shape);
  auto mainloop_tiler = Shape<_128,_16,_128>{};
  auto epilogue_tiler = Shape<_128,_128,_128>{};

  static int constexpr EpilogueUnrollFactor = size<2>(epilogue_tiler) / size<2>(cluster_tile);

  // Get the appropriate blocks for this Cluster
  dim3 cluster_coord_in_grid = cluster_id_in_grid();

  // Total number of k-tiles
  int const K_TILE_MAX = ceil_div(min(N, K), size<2>(epilogue_tiler));

  struct TileScheduler {
    struct WorkTileInfo {
      uint32_t m_idx = 0;
      uint32_t n_idx = 0;
      uint32_t l_idx = 0;
      bool is_valid_tile = false;
    };
    uint32_t tiles_in_m = 0;
    uint32_t tiles_in_n = 0;

    int k_tile_max = 0;

    int wave_cnt = 0;
    WorkTileInfo work_tile_info;
    WorkTileInfo next_work_tile_info;
    CLCResponse* clc_response_ptr_;
    CUTLASS_DEVICE TileScheduler(uint32_t tiles_m, uint32_t tiles_n, int kmax, CLCResponse* clc_response_ptr)
      : tiles_in_m(tiles_m),
        tiles_in_n(tiles_n),

        k_tile_max(kmax),
        work_tile_info({blockIdx.x, blockIdx.y, blockIdx.z, blockIdx.x<tiles_m && blockIdx.y<tiles_n}),
        next_work_tile_info({blockIdx.x, blockIdx.y, blockIdx.z, blockIdx.x<tiles_m && blockIdx.y<tiles_n}),
        clc_response_ptr_(clc_response_ptr) {}

    CUTLASS_DEVICE uint32_t tile_m() const {
      return work_tile_info.m_idx;
    }
    CUTLASS_DEVICE uint32_t tile_n_base() const {
      return work_tile_info.n_idx * uint32_t(k_tile_max);
    }

    CUTLASS_DEVICE uint32_t tiles_m() const { return tiles_in_m; }
    CUTLASS_DEVICE uint32_t tiles_n() const { return tiles_in_n; }
    CUTLASS_DEVICE bool is_valid() const {
      return cute::elem_less(cute::make_coord(work_tile_info.m_idx, work_tile_info.n_idx), cute::make_coord(tiles_in_m, tiles_in_n)) && work_tile_info.is_valid_tile;
    }
    CUTLASS_DEVICE bool is_first_wave() const { return wave_cnt == 0; }
    CUTLASS_DEVICE auto advance_to_next_work(CLCPipeline& clc_pipeline, CLCPipelineState clc_pipe_producer_state) {
      uint32_t mbarrier_addr = clc_pipeline.producer_get_barrier(clc_pipe_producer_state);
      // Wait for clcID buffer to become empty with a flipped phase
      clc_pipeline.producer_acquire(clc_pipe_producer_state);

      if (cute::elect_one_sync()) {
        issue_clc_query(clc_pipe_producer_state, mbarrier_addr, clc_response_ptr_);
      }

      ++clc_pipe_producer_state;
      return clc_pipe_producer_state;
    }

    CUTLASS_DEVICE auto fetch_next_work(CLCPipeline& clc_pipeline, CLCPipelineState clc_pipe_producer_state) {
      clc_pipeline.consumer_wait(clc_pipe_producer_state);
      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&clc_response_ptr_[clc_pipe_producer_state.index()]);
      next_work_tile_info = work_tile_info_from_clc_response(smem_addr);
      clc_pipeline.consumer_release(clc_pipe_producer_state);
      wave_cnt++;
      return;
    }

    CUTLASS_DEVICE auto update_work_tile_info() {
      work_tile_info = next_work_tile_info;
      return;
    }

    CUTLASS_DEVICE uint32_t get_linear_tile_idx() const {
      return work_tile_info.m_idx + work_tile_info.n_idx * tiles_in_m;
    }

    CUTLASS_HOST_DEVICE
    static void
    issue_clc_query(CLCPipelineState state, uint32_t mbarrier_addr, CLCResponse* clc_response_ptr) {
    #if defined(CUTLASS_ARCH_CLC_ENABLED)
        uint32_t result_addr = cute::cast_smem_ptr_to_uint(reinterpret_cast<const void*>(
              &clc_response_ptr[state.index()]));
        asm volatile(
          "{\n\t"
          "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n\t"
          "}\n"
          :
          : "r"(result_addr), "r"(mbarrier_addr));
    #else
        CUTLASS_NOT_IMPLEMENTED();
    #endif
    }
  CUTLASS_DEVICE
  static WorkTileInfo
  work_tile_info_from_clc_response(uint32_t result_addr) {
    WorkTileInfo work_tile_info;
    uint32_t valid = 0;
    #if defined(CUTLASS_ARCH_CLC_ENABLED)
      asm volatile(
        "{\n"
        ".reg .pred p1;\n\t"
        ".reg .b128 clc_result;\n\t"
        "ld.shared.b128 clc_result, [%4];\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"
        "selp.u32 %3, 1, 0, p1;\n\t"
        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_result;\n\t"
        "}\n"
        : "=r"(work_tile_info.m_idx), "=r"(work_tile_info.n_idx), "=r"(work_tile_info.l_idx), "=r"(valid)
        : "r"(result_addr)
        : "memory"
      );

      cutlass::arch::fence_view_async_shared();
    #else
      CUTLASS_NOT_IMPLEMENTED();
    #endif
    work_tile_info.is_valid_tile = (valid == 1);
    return work_tile_info;
  }
  };



  // Allocate SMEMork
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout, ClusterShape, AccumulatorPipelineStageCount, EpilogueUnrollFactor, SchedulerPipelineStageCount>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);
  uint32_t tiles_in_m = uint32_t(size(ceil_div(M, size<0>(cluster_tile))));
  uint32_t tiles_in_n = uint32_t(size(ceil_div(N, size<2>(epilogue_tiler))));
  TileScheduler scheduler(tiles_in_m, tiles_in_n, K_TILE_MAX, shared_storage.clc_response);

  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto acc_shape_mma = make_shape(take<0,2>(mainloop_tiler), _1{}, _1{});
  auto acc_shape_epilogue = make_shape(take<0,2>(epilogue_tiler), _1{}, _1{});

  auto acc_mainloop_pipelined_shape = append(acc_shape_mma, Int<AccumulatorPipelineStageCount>{});
  auto bulk_tmem_mma = TiledMMA::make_fragment_C(acc_mainloop_pipelined_shape);

  static int constexpr NumEpilogueColQuantThreadCount = kEnableRHTColQuant ? 128 : 0;
  static int constexpr NumEpilogueRowQuantThreadCount = kEnableRowQuant ? 256 : 0;
  static int constexpr NumMmaThreadCount = kEnableRHTColQuant? 32: 0;
  static int constexpr NumMmaIssueThreadCount = kEnableRHTColQuant? 1: 0;
  static int constexpr NumSchedThreads = 32;
  static int constexpr NumMainloopLoadThreads = 32;
  static int constexpr NumEpilogueThreads = NumEpilogueColQuantThreadCount + NumEpilogueRowQuantThreadCount;

  TmemAllocator tmem_allocator{};
  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
    NumMmaThreadCount + NumEpilogueColQuantThreadCount,
    cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

  int warp_idx = cutlass::canonical_warp_idx_sync();

  // warp assignment
  bool is_mma_warp = (warp_idx == 0);
  bool is_dma_warp = (warp_idx == 1);
  bool is_sched_warp = (warp_idx == 2);
  bool is_epilogue_col_quant_warp = (warp_idx >= 4 && warp_idx <= 7);
  bool is_epilogue_row_quant_warp = (warp_idx >= 8 && warp_idx <= 15);

  if (is_epilogue_col_quant_warp && elect_one_sync()) {
    cute::prefetch(raw_pointer_cast(c_global_amax));
  }
  if (is_epilogue_row_quant_warp && elect_one_sync()) {
    cute::prefetch(raw_pointer_cast(a_global_amax));
  }

  typename MainloopPipeline::Params mainloop_pipeline_params;
  if (is_dma_warp) {
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (is_mma_warp) {
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  mainloop_pipeline_params.is_leader = cute::elect_one_sync() && is_dma_warp;
  mainloop_pipeline_params.transaction_bytes = kTmaTransactionBytes;
  mainloop_pipeline_params.initializing_warp = 0;
  mainloop_pipeline_params.num_consumers = NumEpilogueRowQuantThreadCount + NumMmaIssueThreadCount;
  MainloopPipeline mainloop_pipeline(
    shared_storage.mainloop,
    mainloop_pipeline_params,
    cluster_shape,
    cute::true_type{},  // Perform barrier init
    cute::true_type{}); // Delay mask calculation

  MainloopPipelineState mainloop_pipe_consumer_state;
  MainloopPipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / EpilogueUnrollFactor, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  AccumulatorPipelineState accumulator_pipe_consumer_state;
  AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

  typename AccumulatorPipeline::Params accumulator_pipeline_params;
  if (is_mma_warp) {
    accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
  }
  if (is_epilogue_col_quant_warp) {
    accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
  }
  // Only one producer thread arrives on this barrier.
  accumulator_pipeline_params.producer_arv_count = 1;
  accumulator_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpilogueColQuantThreadCount;
  accumulator_pipeline_params.initializing_warp = 1;
  using IsInitAccumulatorPipeline = cute::conditional_t<kEnableRHTColQuant, cute::true_type, cute::false_type>;
  AccumulatorPipeline accumulator_pipeline(
    shared_storage.accumulator,
    accumulator_pipeline_params,
    cluster_shape,
    IsInitAccumulatorPipeline{},  // Perform barrier init
    cute::true_type{}); // Delay mask calculation
  // CLC pipeline
  typename CLCPipeline::Params clc_pipeline_params;
  if (is_sched_warp) {
    clc_pipeline_params.role = CLCPipeline::ThreadCategory::ProducerConsumer;
  } else {
    clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
  }
  clc_pipeline_params.producer_blockid = 0;
  clc_pipeline_params.producer_arv_count = 1;
  clc_pipeline_params.consumer_arv_count = NumSchedThreads + cluster_size *
                                                (NumMainloopLoadThreads + NumEpilogueThreads + NumMmaThreadCount);
  clc_pipeline_params.transaction_bytes = sizeof(CLCResponse);
  clc_pipeline_params.initializing_warp = 3;
  CLCPipeline clc_pipeline(shared_storage.clc, clc_pipeline_params, cluster_shape);
  CLCPipelineState clc_pipeline_consumer_state;
  CLCPipelineState clc_pipeline_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

  // CLC throttle pipeline
  typename CLCThrottlePipeline::Params clc_throttle_pipeline_params;
  if (is_dma_warp) {
    clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Producer;
  }
  if (is_sched_warp) {
    clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Consumer;
  }
  clc_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
  clc_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
  clc_throttle_pipeline_params.dst_blockid = 0;
  clc_throttle_pipeline_params.initializing_warp = 4;

  CLCThrottlePipeline clc_throttle_pipeline(shared_storage.clc_throttle, clc_throttle_pipeline_params);
  CLCThrottlePipelineState clc_pipe_throttle_consumer_state;
  CLCThrottlePipelineState clc_pipe_throttle_producer_state = cutlass::make_producer_start_state<CLCThrottlePipeline>();

  if (warp_idx == 2 && elect_one_sync()) {
    cute::initialize_barrier(shared_storage.tma_barrier[0], /* num_threads */ 1);
  }
  __syncthreads();

  if (is_dma_warp) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    Tensor mA = tma_load_a.get_tma_tensor(make_shape(M,N));
    Tensor mB = tma_load_b.get_tma_tensor(make_shape(RhtTensorSize, RhtTensorSize));

    Tensor gA_mk = local_tile(mA, mainloop_tiler, make_coord(_,_, _), Step<_1, X,_1>{});
    Tensor gB_nk = local_tile(mB, cluster_tile, make_coord(_,_, _), Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)

    Tensor tCsA = make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()), sAlayout); // (MMA,MMA_M,MMA_N,PIPE)
    Tensor tCsB = make_tensor(make_smem_ptr(shared_storage.tensors.smem_B.data()), sBlayout); // (MMA,MMA_N,MMA_K,PIPE)

    int block_rank_in_cluster = cute::block_rank_in_cluster();
    ThrMMA thr_mma = mma.get_slice(block_rank_in_cluster);               // blk idx
    Tensor tCgA = thr_mma.partition_A(gA_mk);                                                    // (MMA,MMA_M,MMA_K,k)
    Tensor tCgB = thr_mma.partition_B(gB_nk);                                                    // (MMA,MMA_N,MMA_K,k)

    Layout cta_layout_mnk  = make_layout(cluster_shape);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMMA::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

    auto [tAgA, tAsA] = tma_partition(
      tma_load_a,
      get<2>(cta_coord_vmnk),
      make_layout(size<2>(cta_layout_vmnk)),
      group_modes<0,3>(tCsA),
      group_modes<0,3>(tCgA));

    auto [tBgB, tBsB] = tma_partition(
      tma_load_b,
      get<1>(cta_coord_vmnk),
      make_layout(size<1>(cta_layout_vmnk)),
      group_modes<0,3>(tCsB),
      group_modes<0,3>(tCgB));

    uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);
    if constexpr (kEnableRHTColQuant) {
      if (elect_one_sync()) {
        cute::set_barrier_transaction_bytes(shared_storage.tma_barrier[0], kTmaRhtTensorTransactionBytes);
        copy(tma_load_b.with(shared_storage.tma_barrier[0], tma_mcast_mask_b), tBgB(_,0,0), tBsB(_,0));
      }
    }

    do {
      bool is_first_wave = scheduler.is_first_wave();
      uint32_t skip_wait = is_first_wave;
      auto tAgA_mk = tAgA(_,scheduler.tile_m(),_);
      int k_tile = 0;
      // Throttle CLC producer
      clc_throttle_pipeline.producer_acquire(clc_pipe_throttle_producer_state);
      clc_throttle_pipeline.producer_commit(clc_pipe_throttle_producer_state);
      ++clc_pipe_throttle_producer_state;

      CUTLASS_PRAGMA_NO_UNROLL
      while (k_tile < K_TILE_MAX && k_tile + scheduler.tile_n_base() < scheduler.tiles_n()) {

        int k_tile_idx_n = scheduler.tile_n_base() + k_tile;
        ++k_tile;
        skip_wait = (is_first_wave && k_tile < MainloopPipelineStageCount);
        mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(
          mainloop_pipe_producer_state);
        int write_stage = mainloop_pipe_producer_state.index();
        ++mainloop_pipe_producer_state;
        if (cute::elect_one_sync()) {
          copy(
            tma_load_a.with(*tma_barrier, tma_mcast_mask_a),
            tAgA_mk(_,k_tile_idx_n),
            tAsA(_,write_stage));
        }
      }
      scheduler.fetch_next_work(clc_pipeline, clc_pipeline_consumer_state);
      ++clc_pipeline_consumer_state;
      scheduler.update_work_tile_info();
    } while (scheduler.is_valid());
    mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
  } else if (is_mma_warp) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if constexpr (kEnableRHTColQuant) {
      Tensor tCsA = make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()), sAlayout); // (MMA,MMA_M,MMA_N,PIPE)
      Tensor tCsB = make_tensor(make_smem_ptr(shared_storage.tensors.smem_B.data()), sBlayout); // (MMA,MMA_N,MMA_K,PIPE)

      int block_rank_in_cluster = cute::block_rank_in_cluster();
      ThrMMA thr_mma = mma.get_slice(block_rank_in_cluster);               // blk idx
      // Allocate "fragments" -- these are actually umma smem descriptors
      Tensor tCrA = thr_mma.make_fragment_A(tCsA);                                              // (MMA,MMA_M,MMA_K,PIPE)
      Tensor tCrB = thr_mma.make_fragment_B(tCsB);                                              // (MMA,MMA_M,MMA_K,PIPE)

      mma.accumulate_ = UMMA::ScaleOut::Zero;

      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      bulk_tmem_mma.data() = tmem_base_ptr;
      cute::wait_barrier(shared_storage.tma_barrier[0], 0 /*tma_phase_bit*/);
      do {
        uint32_t skip_wait = K_TILE_MAX <= 0;

        auto barrier_token = mainloop_pipeline.consumer_try_wait(
          mainloop_pipe_consumer_state,
          skip_wait);
        scheduler.fetch_next_work(clc_pipeline, clc_pipeline_consumer_state);
        ++clc_pipeline_consumer_state;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile = 0; k_tile < K_TILE_MAX && k_tile + scheduler.tile_n_base() < scheduler.tiles_n(); ) {
          mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);
          int read_stage = mainloop_pipe_consumer_state.index();
          auto tCrA_mk = tCrA(_,_,_,read_stage);
          auto tCrB_nk = tCrB(_,_,0,0);
          CUTLASS_PRAGMA_UNROLL
          for (int k_block = 0; k_block < size<2>(tCrA) / EpilogueUnrollFactor; ++k_block)
          {
            int accumulator_k_block = accumulator_pipe_producer_state.index() * EpilogueUnrollFactor;
            int tCrA_k_block = k_block * EpilogueUnrollFactor;
            accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < EpilogueUnrollFactor; i++) {
              auto accumulators = bulk_tmem_mma(_,_,_,accumulator_k_block + i);
              gemm(mma, tCrA_mk(_,_,tCrA_k_block + i), tCrB_nk, accumulators);
            }

            accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
            ++accumulator_pipe_producer_state;
          }
          auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;
          ++mainloop_pipe_consumer_state;
          ++k_tile;
          skip_wait = k_tile >= K_TILE_MAX;
          mainloop_pipeline.umma_consumer_release(curr_mainloop_pipe_consumer_state);
          barrier_token = mainloop_pipeline.consumer_try_wait(
            mainloop_pipe_consumer_state,
            skip_wait);
        }
        scheduler.update_work_tile_info();
      } while (scheduler.is_valid());
      tmem_allocator.release_allocation_lock();
      accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
      tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
  } else if(is_sched_warp) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    do {
      clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
      clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);
      ++clc_pipe_throttle_consumer_state;
      clc_pipeline_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipeline_producer_state);
      scheduler.fetch_next_work(clc_pipeline, clc_pipeline_consumer_state);
      ++clc_pipeline_consumer_state;
      scheduler.update_work_tile_info();
    } while (scheduler.is_valid());
  } else if (is_epilogue_col_quant_warp) {
    cutlass::arch::warpgroup_reg_alloc<192>();
    if constexpr (kEnableRHTColQuant) {
      using TMEM_LOAD_NEW = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b64x;

      float const c_global_amax_val = *c_global_amax;
      auto acc_epilogue_pipelined_shape = append(acc_shape_epilogue, Int<AccumulatorPipelineStageCount / EpilogueUnrollFactor>{});
      auto bulk_tmem_epilogue_layout = make_layout(
        acc_epilogue_pipelined_shape,
        make_stride(
          stride<0>(bulk_tmem_mma),
          Int<0>{},
          Int<0>{},
          size<1>(epilogue_tiler)));
      auto bulk_tmem_epilogue = make_tensor(make_tmem_ptr<uint32_t>(), bulk_tmem_epilogue_layout);

      // leveraging 256-bit writes to global memory
      static int constexpr FragmentSize = 256 / sizeof_bits_v<TD>;

      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      bulk_tmem_epilogue.data() = tmem_base_ptr;
      int global_thread_idx = threadIdx.x;
      int local_thread_idx = global_thread_idx % cutlass::NumThreadsPerWarpGroup;

      size_t rng_seed = 0;
      size_t rng_offset = 0;
      if constexpr (kEnableStochasticRounding) {
        rng_seed = rng_state != nullptr ? __ldg(rng_state) : 0;
        rng_offset = rng_state != nullptr ? __ldg(rng_state + 1) : 0;
      }

      Tensor mD = make_tensor(
        cute::subbyte_iterator<TD>(D),
        make_shape(M,N),
        dD); // (M,N)
      Tensor gD_mn = local_tile(
        mD,
        epilogue_tiler,
        make_coord(_,_, _),
        Step<_1,_1, X>{}); // (BLK_M,BLK_N)
      Tensor pD = make_identity_tensor(mD.shape());
      Tensor pD_mn = local_tile(
        pD,
        epilogue_tiler,
        make_coord(_,_, _),
        Step<_1,_1, X>{}); // (BLK_M,BLK_N)
      Tensor mSFD = make_tensor(make_gmem_ptr(SFD), sfd_layout);
      Tensor gSFD_mn = local_tile(mSFD, epilogue_tiler, make_coord(_,_, _), Step<_1,_1, X>{});           // (BLK_M,BLK_N)
      Tensor pSFD = make_identity_tensor(mSFD.shape());
      Tensor pSFD_mn = local_tile(pSFD, epilogue_tiler, make_coord(_,_, _), Step<_1,_1, X>{});           // (BLK_M,BLK_N)

      Tensor gD_mn_view = tiled_divide(gD_mn, take<0,2>(epilogue_tiler));
      Tensor pD_mn_view = tiled_divide(pD_mn, take<0,2>(epilogue_tiler));
      auto tiled_t2r = make_tmem_copy(TMEM_LOAD_NEW{}, bulk_tmem_epilogue(_,_,_,_0{}));
      auto tiled_r2g = make_tiled_copy_D(
        Copy_Atom<SM100_STORE_256bit_CACHE_NOALLOCATION, TD>{},
        tiled_t2r);
      auto thr_t2r   = tiled_t2r.get_slice(local_thread_idx);
      auto thr_r2g = tiled_r2g.get_slice(local_thread_idx);

      // Aligning with TensorEngine's recipe to generate scale factors // {$nv-internal-release}
      static constexpr float fp4_max = 6.0f;
      static constexpr float fp8_max = 448.0f;
      float const fp4_max_inv = 1.0f / fp4_max;
      float const global_encode_scale = c_global_amax_val > 0.0f
        ? cutlass::minimum_with_nan_propagation<float>{}(
          (fp8_max * fp4_max) / c_global_amax_val,
          cutlass::platform::numeric_limits<float>::max())
        : 1.0f;

      float const global_decode_scale = 1.0f / global_encode_scale;
      // Scaling factor for fast math path
      float global_encode_scale_multiplier = 1.0f;
      if constexpr (kUseFastMath) {
        global_encode_scale_multiplier = global_encode_scale * fp4_max_inv;
      }
      auto sfc_converter = cutlass::NumericConverter<TSFD, float>{};

      do {
        scheduler.fetch_next_work(clc_pipeline, clc_pipeline_consumer_state);
        ++clc_pipeline_consumer_state;
        for (int k_tile = 0; k_tile < K_TILE_MAX && k_tile + scheduler.tile_n_base() < scheduler.tiles_n(); ++k_tile) {
          Tensor tDgD_mn = gD_mn_view(_,_,_,scheduler.tile_m(), scheduler.tile_n_base() + k_tile);
          Tensor tDgSFD_mn = gSFD_mn(_,_,scheduler.tile_m(), scheduler.tile_n_base() + k_tile);
          Tensor tDpD_mn = pD_mn_view(_,_,_,scheduler.tile_m(), scheduler.tile_n_base() + k_tile);
          Tensor tDpSFD_mn = pSFD_mn(_,_,scheduler.tile_m(), scheduler.tile_n_base() + k_tile);

          accumulator_pipeline.consumer_wait(accumulator_pipe_consumer_state);

          auto Acc = bulk_tmem_epilogue(_,_,_,accumulator_pipe_consumer_state.index());
          Tensor tDtAcc = thr_t2r.partition_S(Acc); // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
          Tensor tDgD = thr_t2r.partition_D(tDgD_mn); // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
          Tensor tDpD = thr_t2r.partition_D(tDpD_mn); // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
          Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tDgD)); // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
          Tensor tDrD = make_tensor<TD>(shape(tDgD));
          Tensor tTR_rAcc_frag = recast<cutlass::Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc));
          Tensor tDrD_frag = recast<cutlass::Array<TD, FragmentSize>>(coalesce(tDrD));

          Tensor src = thr_r2g.retile_S(tDrD);
          Tensor dst = thr_r2g.retile_D(tDgD);
          Tensor pSrc = thr_r2g.retile_D(tDpD);

          Tensor tDgSFD_view = make_tensor(
            tDgSFD_mn.data(),
              make_layout(
                make_shape(shape(tDgSFD_mn), Int<1>{}, Int<1>{}),
                make_stride(stride(tDgSFD_mn), Int<0>{}, Int<0>{})));
          Tensor tDpSFD_view = make_tensor(
            tDpSFD_mn.data(),
              make_layout(
                make_shape(shape(tDpSFD_mn), Int<1>{}, Int<1>{}),
                make_stride(stride(tDpSFD_mn), Int<0>{}, Int<0>{})));
          Tensor tDgSFD = filter(thr_t2r.partition_D(tDgSFD_view));
          Tensor tDrSFD = make_tensor<TSFD>(shape(tDgSFD));
          Tensor tDpSFD = filter(thr_t2r.partition_D(tDpSFD_view));
          static int constexpr NumVecs = size(tDgD) / VectorSize;
          Tensor tD_rRowSFD_frg = recast<cutlass::Array<TSFD, NumVecs>>(tDrSFD);

          cutlass::maximum_absolute_value_reduction<cutlass::Array<ElementAccumulator, VectorSize>, true> amax_reduction;
          cutlass::Array<ElementAccumulator, NumVecs> vec_maxs;
          cutlass::Array<ElementAccumulator, NumVecs> pvscales;
          // TMEM_LOAD
          copy(tiled_t2r, tDtAcc, tTR_rAcc);
          cutlass::arch::fence_view_async_tmem_load();
          accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);
          ++accumulator_pipe_consumer_state;

          if constexpr (!kUseFastMath) {
            // Downcast to BF16 for bit-wise compatibility with
            // unfused kernels
            auto convert_accum_to_bf16 =
                cutlass::NumericArrayConverter<cutlass::bfloat16_t, ElementAccumulator,
                                               FragmentSize>{};
            auto convert_bf16_to_accum =
                cutlass::NumericArrayConverter<ElementAccumulator, cutlass::bfloat16_t,
                                               FragmentSize>{};
            tTR_rAcc_frag(_0{}) = convert_bf16_to_accum(convert_accum_to_bf16(tTR_rAcc_frag(_0{})));
            tTR_rAcc_frag(_1{}) = convert_bf16_to_accum(convert_accum_to_bf16(tTR_rAcc_frag(_1{})));
          }

          auto compute_frgs = reinterpret_cast<cutlass::Array< ElementAccumulator, VectorSize> *>(tTR_rAcc_frag.data());
          auto output_frgs = reinterpret_cast<cutlass::Array< TD, VectorSize> *>(tDrD_frag.data());
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < NumVecs; v++) {
            vec_maxs[v] = amax_reduction(ElementAccumulator(0), compute_frgs[v]);
          }

          if constexpr (kUseFastMath) {
            // Fast math: multiply with precomputed reciprocal
            pvscales = cutlass::multiplies<cutlass::Array<ElementAccumulator, NumVecs>>{}(
                vec_maxs, global_encode_scale_multiplier);
          } else {
            // Accurate math: perform division
            pvscales =
                cutlass::divides<cutlass::Array<ElementAccumulator, NumVecs>>{}(vec_maxs, fp4_max);
            pvscales = cutlass::multiplies<cutlass::Array<ElementAccumulator, NumVecs>>{}(
                pvscales, global_encode_scale);
          }
          auto pvscales_cvted = cutlass::NumericArrayConverter<TSFD, ElementAccumulator, NumVecs>{}(pvscales);

          tD_rRowSFD_frg(_0{}) = pvscales_cvted;
          auto qpvscale_ups = cutlass::NumericArrayConverter<ElementAccumulator, TSFD, NumVecs>{}(tD_rRowSFD_frg(_0{}));
          auto qpvscale_scaled = cutlass::multiplies<cutlass::Array<ElementAccumulator, NumVecs>>{}(
            qpvscale_ups,
            global_decode_scale);

          cutlass::Array<ElementAccumulator, NumVecs> acc_scales;
          if constexpr (kUseFastMath) {
            // fast math: use reciprocal approximate to replace div
            acc_scales = cutlass::reciprocal_approximate_ftz<decltype(qpvscale_scaled)>{}(qpvscale_scaled);
          } else {
            // regular path for slower math, use divide to replace div
            acc_scales = cutlass::divides<cutlass::Array<ElementAccumulator, NumVecs>>{}(1.0, qpvscale_scaled);
          }

          uint4 random_uint4 = uint4{0, 0, 0, 0};
          transformer_engine::curanddx::detail::philox4x32_native_state<10> rng;
          // "Prefetch" a stochastic rounding state for the first tile
          if constexpr (kEnableStochasticRounding) {
            const size_t rng_sequence = global_thread_idx + k_tile * 512 + scheduler.get_linear_tile_idx() * K_TILE_MAX * 512;
            rng.init(rng_seed, rng_sequence, rng_offset);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < NumVecs; v++) {
            auto acc_scale = cutlass::minimum_with_nan_propagation<ElementAccumulator>{}(
              acc_scales[v],
              cutlass::platform::numeric_limits<ElementAccumulator>::max());
            if constexpr (kEnableStochasticRounding) {
              random_uint4 = rng.generate4();
              output_frgs[v] = StochasticNumericConverter(cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(compute_frgs[v], acc_scale), *reinterpret_cast<cutlass::Array<uint32_t, 4>*>(&random_uint4));
            } else {
              output_frgs[v] = cutlass::NumericArrayConverter<TD, ElementAccumulator, VectorSize>{}(
              cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(
                compute_frgs[v],
                acc_scale));
            }

          }

          copy_if(tiled_r2g,
                [&](auto coord){
                  Tensor pSrc_view = group_modes<1,rank(pSrc)>(pSrc);
                  return elem_less(pSrc_view(_0{},coord), shape(mD));
                },
                src, dst);
          // 32bit vectorization copy 4 e4m3 SFD for per 64/(16,4):(0, 1)  element
          constexpr int vec_len = 32 / sizeof_bits_v<TSFD>;
          Tensor  tDrSFD_v = recast<uint_bit_t<32>>(tDrSFD);
          Tensor  tDgSFD_v = recast<uint_bit_t<32>>(tDgSFD);
          copy_if(
                  [&](auto coord){
                    Tensor tDpSFD_view = group_modes<1,rank(tDpSFD)>(tDpSFD);
                    return elem_less(tDpSFD_view(_0{}, coord * vec_len), shape(mSFD));
                  },
                  tDrSFD_v, tDgSFD_v);
        }
        scheduler.update_work_tile_info();
      } while (scheduler.is_valid());
    }
  } else if (is_epilogue_row_quant_warp) {
    cutlass::arch::warpgroup_reg_alloc<136>();
    if constexpr (kEnableRowQuant) {
      using S2RVectorType = uint128_t;
      float const a_global_amax_val = *a_global_amax;
      int global_thread_idx = threadIdx.x;
      int local_thread_idx = global_thread_idx % 256;
      size_t rng_seed = 0;
      size_t rng_offset = 0;
      if constexpr (kEnableStochasticRounding) {
        rng_seed = rng_state != nullptr ? __ldg(rng_state) : 0;
        rng_offset = rng_state != nullptr ? __ldg(rng_state + 1) : 0;
      }
      Tensor mQA = make_tensor(cute::subbyte_iterator<TQA>(QA), make_layout(make_shape(M, N), dQA));
      Tensor gQA_mn = local_tile(mQA, epilogue_tiler, make_coord(_,_, _), Step<_1,X,_1>{});
      Tensor pQA = make_identity_tensor(mQA.shape());
      Tensor pQA_mn = local_tile(pQA, epilogue_tiler, make_coord(_,_, _), Step<_1,X,_1>{});

      Tensor mSFA = make_tensor(make_gmem_ptr(SFA), sfa_layout);
      Tensor gSFA_mn = local_tile(mSFA, epilogue_tiler, make_coord(_,_, _), Step<_1,X,_1>{});           // (BLK_M,BLK_N)
      Tensor pSFA = make_identity_tensor(mSFA.shape());
      Tensor pSFA_mn = local_tile(pSFA, epilogue_tiler, make_coord(_,_, _), Step<_1,X,_1>{});
      Tensor sA = as_position_independent_swizzle_tensor(
                    group_modes<0,2>(coalesce(make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()), sAlayout)))); // (BLOCK_M, BLOCK_M,PIPE)
      using S2RWarpLayout = Layout<Shape<_2,_16>>;
      using WarpGroupLayout = Layout<Shape<_1,_8>>;
      using S2RThreadLayout = decltype(blocked_product(S2RWarpLayout{}, WarpGroupLayout{}));
      using S2RValLayout = Layout<Shape<Int<64>, _1>>;
      using S2RAtomA = Copy_Atom<AutoVectorizingCopy, TA>;
      using R2GAtomQA = Copy_Atom<SM100_STORE_256bit_CACHE_NOALLOCATION, TQA>;

      auto tiled_s2r = make_tiled_copy(S2RAtomA{}, S2RThreadLayout{}, S2RValLayout{});
      auto tiled_r2g_QA = make_tiled_copy_D(R2GAtomQA{}, tiled_s2r);

      auto thr_s2r = tiled_s2r.get_slice(local_thread_idx);
      auto thr_r2g_QA = tiled_r2g_QA.get_slice(local_thread_idx);

      Tensor tQAsA = thr_s2r.partition_S(sA); // (Copy, Copy_M, Copy_N, PIPE)

      Tensor tQArA = make_tensor_like<TA>(make_layout(tQAsA(_, _, _, _0{}).shape())); // (Copy, Copy_M, Copy_N)
      // Tensor tQArA_PI = thr_s2r.partition_S(sA_PI);
      Tensor tQAgQA = thr_r2g_QA.partition_D(gQA_mn);
      Tensor tQArQA = make_tensor_like(tQAgQA(_, _, _, _0{}, _0{}));
      Tensor tQApQA = thr_r2g_QA.partition_D(pQA_mn);

      Tensor tQAgSFA = thr_s2r.partition_D(gSFA_mn);
      Tensor tQArSFA = make_tensor_like(tQAgSFA(_, _, _, _0{}, _0{}));
      Tensor tQApSFA = thr_s2r.partition_D(pSFA_mn);

      // Aligning with TensorEngine's recipe to generate scale factors // {$nv-internal-release}
      static constexpr float fp4_max = 6.0f;
      static constexpr float fp8_max = 448.0f;
      float const fp4_max_inv = 1.0f / fp4_max;
      float const global_encode_scale = a_global_amax_val > 0.0f
        ? cutlass::minimum_with_nan_propagation<float>{}(
          (fp8_max * fp4_max) / a_global_amax_val,
          cutlass::platform::numeric_limits<float>::max())
        : 1.0f;

      float const global_decode_scale = 1.0f / global_encode_scale;
      // Scaling factor for fast math path
      float global_encode_scale_multiplier = 1.0f;
      if constexpr (kUseFastMath) {
        global_encode_scale_multiplier = global_encode_scale * fp4_max_inv;
      }
      auto sfa_converter = cutlass::NumericConverter<TSFA, ElementAccumulator>{};
      do {
        uint32_t skip_wait = K_TILE_MAX <= 0;

        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile = 0; k_tile < K_TILE_MAX && k_tile + scheduler.tile_n_base() < scheduler.tiles_n(); ) {
          auto tQAgSFA_mn = tQAgSFA(_, _, _, scheduler.tile_m(), scheduler.tile_n_base() + k_tile);
          auto tQAgQA_mn = tQAgQA(_, _, _, scheduler.tile_m(), scheduler.tile_n_base() + k_tile);
          auto tQApSFA_mn = tQApSFA(_, _, _, scheduler.tile_m(), scheduler.tile_n_base() + k_tile);
          auto tQApQA_mn = tQApQA(_, _, _, scheduler.tile_m(), scheduler.tile_n_base() + k_tile);
          auto barrier_token = mainloop_pipeline.consumer_try_wait(
            mainloop_pipe_consumer_state);
          mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);
          copy(tiled_s2r, tQAsA(_, _, _, mainloop_pipe_consumer_state.index()), tQArA);
          cutlass::arch::fence_view_async_shared();
          auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;
          ++mainloop_pipe_consumer_state;
          ++k_tile;
          skip_wait = k_tile >= K_TILE_MAX;
          mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
          // static int constexpr NumVecs = size(tQArA) / VectorSize;
          cutlass::maximum_absolute_value_reduction<cutlass::Array<ElementAccumulator, VectorSize>, true> amax_reduction;
          auto compute_frgs = reinterpret_cast<cutlass::Array<TA, VectorSize> *>(tQArA.data());
          auto output_frgs = reinterpret_cast<cutlass::Array<TQA, VectorSize> *>(raw_pointer_cast(tQArQA.data()));
          transformer_engine::curanddx::detail::philox4x32_native_state<10> rng;
          if constexpr (kEnableStochasticRounding) {
            const size_t rng_sequence = global_thread_idx + k_tile * 512 + scheduler.get_linear_tile_idx() * K_TILE_MAX * 512;
            rng.init(rng_seed, rng_sequence, rng_offset);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < size(tQArA)/VectorSize; v++) {
            auto compute_frgs_up = cutlass::NumericArrayConverter<ElementAccumulator, TA, VectorSize>{}(compute_frgs[v]);
            auto amax = amax_reduction(ElementAccumulator(0), compute_frgs_up);
            // declare pvscales
            ElementAccumulator pvscales;
            if constexpr (kUseFastMath) {
              // Fast math: multiply with precomputed reciprocal
              pvscales = cutlass::multiplies<ElementAccumulator>{}(amax, global_encode_scale_multiplier);
            } else {
              // Accurate math: perform division
              pvscales = cutlass::divides<ElementAccumulator>{}(amax, fp4_max);
              pvscales = cutlass::multiplies<ElementAccumulator>{}(pvscales, global_encode_scale);
            }
            filter(tQArSFA)(v) = sfa_converter(pvscales);
            auto qpvscale_ups = cutlass::NumericConverter<ElementAccumulator, TSFA>{}(filter(tQArSFA)(v));
            auto qpvscale_scaled = cutlass::multiplies<ElementAccumulator>{}(qpvscale_ups, global_decode_scale);
            ElementAccumulator acc_scales;
            if constexpr (kUseFastMath) {
              // fast math: use reciprocal approximate to replace div
              acc_scales = cutlass::reciprocal_approximate_ftz<decltype(qpvscale_scaled)>{}(qpvscale_scaled);
            } else {
              // regular path for slower math, use divide to replace div
              acc_scales = cutlass::divides<ElementAccumulator>{}(1.0, qpvscale_scaled);
            }
            auto acc_scale = cutlass::minimum_with_nan_propagation<ElementAccumulator>{}(
              acc_scales,
              cutlass::platform::numeric_limits<ElementAccumulator>::max());
            uint4 random_uint4 = uint4{0, 0, 0, 0};
            if constexpr (kEnableStochasticRounding) {
              random_uint4 = rng.generate4();
              output_frgs[v] = StochasticNumericConverter(cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(compute_frgs_up, acc_scale), *reinterpret_cast<cutlass::Array<uint32_t, 4>*>(&random_uint4));
            } else {
              output_frgs[v] = cutlass::NumericArrayConverter<TQA, ElementAccumulator, VectorSize>{}(
                cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(
                  compute_frgs_up,
                  acc_scale));
            }
          }

          copy_if(tiled_r2g_QA,
                  [&](auto coord){
                    Tensor tQApQA_view = group_modes<1,rank(tQApQA_mn)>(tQApQA_mn);
                    return elem_less(tQApQA_view(_0{}, coord), shape(mQA));
                  },
                  tQArQA, tQAgQA_mn);
          // 32bit vectorization copy 4 e4m3 SFA for per 64/(16,4):(0, 1)  element
          constexpr int vec_len = 32 / sizeof_bits_v<TSFD>;
          Tensor  tQArSFA_v = recast<uint_bit_t<32>>(filter(tQArSFA));
          Tensor  tQAgSFA_v = recast<uint_bit_t<32>>(filter(tQAgSFA_mn));
          copy_if(
                  [&](auto coord){
                    Tensor tQApSFA_view = filter(tQApSFA_mn);
                    return elem_less(tQApSFA_view(_0{}, coord * vec_len), shape(mSFA));
                  },
                  tQArSFA_v, tQAgSFA_v);
        }
        scheduler.fetch_next_work(clc_pipeline, clc_pipeline_consumer_state);
        ++clc_pipeline_consumer_state;
        scheduler.update_work_tile_info();
      }while (scheduler.is_valid());
    }
  } else {
      cutlass::arch::warpgroup_reg_dealloc<32>();
  }
} // NOLINT(readability/fn_size)


// this function computes RHT-GEMM for
// m = hidden_size, n = sequence_length
// A: m x n: col-major
// B: 16 x 16: row-major
// D: m x n: row-major
// SFD: m x (n/16): row-major
// QA: m x n: col-major
// SFA: m/16 x n: col-major
template <bool kEnableStochasticRounding, bool kEnableRHTColQuant, bool kEnableRowQuant, bool kEnableSwizzleSFOutput,
class TA, class TB, class TD, class TSFD, class TQA, class TSFA, bool kUseFastMath=true>
void row_col_rht_gemm_ntt_w_sfc(
    int sequence_length,
    int hidden_size,
    TA const* A,
    TB const* B,
    TD* D,
    TSFD* SFD,
    TQA* QA,
    TSFA* SFA,
    float const* a_global_amax,
    float const* d_global_amax,
    const size_t* rng_state,
    uint32_t sm_count,
    cudaStream_t stream,
    int k_tile_size = 1024) {
  using namespace cute;
  static int constexpr SFVecSize = 16;
  static int constexpr RhtTensorSize = 16;

  static_assert(RhtTensorSize == 16, "RhtTensorSize must be 16");
  using LinearSFALayout = decltype(make_layout(make_shape(make_shape(Int<SFVecSize>{}, 0), 0), make_stride(make_stride(_0{}, _1{}), 0)));
  using LinearSFCLayout = decltype(make_layout(make_shape(0, make_shape(Int<SFVecSize>{}, 0)), make_stride(0, make_stride(_0{}, _1{}))));

  using SwizzledSFALayoutAtom = cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVecSize, UMMA::Major::MN>::SfAtom;
  using SwizzledSFDLayoutAtom = cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVecSize, UMMA::Major::K>::SfAtom;
  using SwizzledSFALayout = decltype(tile_to_shape(SwizzledSFALayoutAtom{}, make_shape(hidden_size,sequence_length), Step<_1,_2>{}));
  using SwizzledSFDLayout = decltype(tile_to_shape(SwizzledSFDLayoutAtom{}, make_shape(hidden_size,sequence_length), Step<_2,_1>{}));

  using SFALayout = cute::conditional_t<kEnableSwizzleSFOutput, SwizzledSFALayout, LinearSFALayout>;
  using SFCLayout = cute::conditional_t<kEnableSwizzleSFOutput, SwizzledSFDLayout, LinearSFCLayout>;
  SFALayout sfa_layout;
  SFCLayout sfd_layout;

  if constexpr (kEnableSwizzleSFOutput) {
    sfa_layout = tile_to_shape(SwizzledSFALayoutAtom{}, make_shape(hidden_size, sequence_length), Step<_1,_2>{});
    sfd_layout = tile_to_shape(SwizzledSFDLayoutAtom{}, make_shape(hidden_size, sequence_length), Step<_2,_1>{});
  } else {
    sfa_layout = make_layout(make_shape(make_shape(Int<SFVecSize>{}, hidden_size/SFVecSize), sequence_length), make_stride(make_stride(_0{}, _1{}), hidden_size/SFVecSize));
    sfd_layout = make_layout(make_shape(hidden_size, make_shape(Int<SFVecSize>{}, sequence_length/SFVecSize)), make_stride(sequence_length/SFVecSize, make_stride(_0{}, _1{})));
  }
  // Define shapes (dynamic)
  auto M = hidden_size;
  auto N = sequence_length;
  Tensor tensorA = make_tensor(A, make_shape(hidden_size, sequence_length), LayoutLeft{});
  Tensor tensorB = make_tensor(B, make_shape(RhtTensorSize, RhtTensorSize), LayoutLeft{});
  Tensor tensorD = make_tensor(D, make_shape(hidden_size, sequence_length), LayoutRight{});
  Tensor tensorQA = make_tensor(QA, make_shape(hidden_size, sequence_length), LayoutLeft{});
  Tensor tensorSFD = make_tensor(SFD, sfd_layout);
  Tensor tensorSFA = make_tensor(SFA, sfa_layout);
  // Define strides (from tensors)
  auto dA = stride(tensorA);   // (dM,dK)
  auto dB = stride(tensorB);   // (dN,dK)
  auto dD = stride(tensorD);   // (dM,dN)
  auto dQA = stride(tensorQA); // (dM,dK)
  using ClusterShape = Shape<  _1,  _1, _1>;
  auto cluster_shape = ClusterShape{};
  auto cluster_tile_shape = Shape<_128,Int<RhtTensorSize>,Int<RhtTensorSize>>{};
  auto cluster_tile_mainloop = Shape<_128,Int<RhtTensorSize>,_128>{};

  // Each mainloop / epilogue loads 128 x 64 tiles while each MMA proceeds with 128 x 16 tiles
  static int constexpr EpilogueUnrollFactor =
    size<2>(cluster_tile_mainloop) / size<2>(cluster_tile_shape);
  // Construct the MMA
  auto mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TA, TB, float,
                                               size<0>(cluster_tile_shape), size<1>(cluster_tile_shape),
                                               UMMA::Major::MN, UMMA::Major::MN>{},
                            Layout<Shape<_1,_1>>{});

  // Assert that the TiledMMA uses all CTAs in the CGA.
  CUTE_STATIC_ASSERT_V(size(cluster_shape) == size(mma));
  CUTE_STATIC_ASSERT_V(evenly_divides(cluster_tile_shape, tile_shape(mma)));

  // Determine the A and B shapes
  auto mma_shape_B = partition_shape_B(mma, make_shape(size<1>(cluster_tile_shape), size<2>(cluster_tile_shape)));

  using TiledMma = decltype(mma);
  using AtomThrID = typename TiledMma::AtomThrID;

  using SmemShape_M = decltype(shape_div(shape<0>(cluster_tile_shape), shape_div(shape<0>(cluster_tile_shape), size<0>(cluster_tile_shape) / size(AtomThrID{}))));
  using SmemShape_N = decltype(shape_div(shape<1>(cluster_tile_shape), shape_div(shape<1>(cluster_tile_shape), size<1>(cluster_tile_shape) / size(AtomThrID{}))));
  using SmemShape_K = decltype(cute::get<2>(cluster_tile_shape));

  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      cute::UMMA::Major::MN, TB, SmemShape_N, SmemShape_K>());

  auto mma_shape_A = partition_shape_A(mma, make_shape(size<0>(cluster_tile_mainloop), size<2>(cluster_tile_mainloop)));
  using SmemShape_M_A = decltype(shape_div(shape<0>(cluster_tile_mainloop), shape_div(shape<0>(cluster_tile_mainloop), size<0>(cluster_tile_mainloop) / size(AtomThrID{}))));
  using SmemShape_K_A = decltype(cute::get<2>(cluster_tile_mainloop));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      cute::UMMA::Major::MN, TA, SmemShape_M_A, SmemShape_K_A>());

  static uint32_t constexpr TotalTmemRows = 128;
  static uint32_t constexpr Sm100TmemCapacityColumns = 512;
  static uint32_t constexpr TotalTmem = TotalTmemRows * Sm100TmemCapacityColumns;
  static uint32_t constexpr AccumulatorPipelineStageCount =
      TotalTmem /
      (cute::size<0>(cluster_tile_shape) * cute::size<1>(cluster_tile_shape));

  // Define the smem layouts (static)
  // Calculate max pipeline stages based on Blackwell SM100's 232KB shared memory
  constexpr int SchedulerPipelineStageCount = 6;
  static int constexpr MainloopPipelineBytes = sizeof(typename cutlass::detail::CustomizedPipelineTmaUmmaAsync<
                                                1,
                                                Shape<_1,_1,_1>,
                                                Shape<_1, _1, _1>>::SharedStorage);

  static int constexpr ClcResponseBytes = sizeof(CLCResponse) * SchedulerPipelineStageCount;
  static int constexpr CLCThrottlePipelineBytes = sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
  static int constexpr CLCPipelineBytes = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>::SharedStorage);
  static int constexpr TmemDeallocBytes = sizeof(cutlass::arch::ClusterBarrier);
  static int constexpr BTensorBytes = cute::size(mma_shape_B) * sizeof(TB);
  static int constexpr AccPipelineBytes = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / EpilogueUnrollFactor, Shape<_1, _1, _1>>::SharedStorage);
  static int constexpr TmemBasePtrsBytes = sizeof(uint32_t);
  static int constexpr kBlackwellSmemSize = 232448; // 232KB in bytes
  static int constexpr kBytesPerStage =
    cute::size(mma_shape_A) * sizeof(TA) + MainloopPipelineBytes;
  static int constexpr kReservedBytes = ClcResponseBytes + CLCThrottlePipelineBytes + TmemBasePtrsBytes +
                                    CLCPipelineBytes + TmemDeallocBytes+BTensorBytes + AccPipelineBytes; // Reserve for barriers and other uses
  static int constexpr kMaxStages = (kBlackwellSmemSize - kReservedBytes) / kBytesPerStage;
  auto sP = Int<kMaxStages>{};      // SMEM pipelines
  auto sA = UMMA::tile_to_mma_shape(
    SmemLayoutAtomA{},
    append(mma_shape_A, sP), Step<_2,_1,_3>{}); // (MMA,MMA_M,MMA_K,PIPE)
  auto sB = UMMA::tile_to_mma_shape(
    SmemLayoutAtomB{},
    append(mma_shape_B, _1{})); // (MMA,MMA_N,MMA_K, _1)
  auto sD = Layout<_1>{};  // XXX Dummy

  auto tma_load_a = make_tma_copy_A_sm100(
    SM90_TMA_LOAD{},
    tensorA,
    sA(_,_,_,0),
    cluster_tile_mainloop,
    mma);
  auto tma_load_b = make_tma_copy_B_sm100(
    SM90_TMA_LOAD{},
    tensorB,
    sB(_,_,_,0),
    cluster_tile_shape,
    mma);

  // Assert checks problem size should be multiple of 64
  assert(M % 64 == 0);
  assert(N % 64 == 0);

  uint32_t tiles_in_m = uint32_t(size(ceil_div(M, size<0>(cluster_tile_shape))));
  uint32_t tiles_in_n = uint32_t(size(ceil_div(N, k_tile_size)));
  uint32_t tiles = tiles_in_m * tiles_in_n;

  dim3 dimBlock(512);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
  dim3 dimGrid(tiles_in_m, tiles_in_n, 1);

  int smem_size = sizeof(
    SharedStorage<
    TA,
    TB,
    decltype(sA),
    decltype(sB),
    ClusterShape,
    AccumulatorPipelineStageCount,
    EpilogueUnrollFactor,
    SchedulerPipelineStageCount>);

  auto* kernel_ptr = &row_col_rht_gemm_device<
    decltype(M), decltype(N), decltype(k_tile_size),
    decltype(cluster_shape), decltype(cluster_tile_shape),
    TA, decltype(dA), decltype(sA), decltype(tma_load_a),
    TB, decltype(dB), decltype(sB), decltype(tma_load_b),
    TD, decltype(dD), decltype(sD),
    TSFD, decltype(sfd_layout),
    TQA, decltype(dQA),
    TSFA, decltype(sfa_layout),
    decltype(mma),
    AccumulatorPipelineStageCount,
    SchedulerPipelineStageCount,
    kEnableStochasticRounding,
    kEnableRHTColQuant,
    kEnableRowQuant,
    kUseFastMath>;

  NVTE_CHECK_CUDA(cudaFuncSetAttribute(*kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
     params, (void const *)kernel_ptr, M, N, k_tile_size, cluster_shape, cluster_tile_shape,
     tensorA.data(), dA, sA, tma_load_a,
     tensorB.data(), dB, sB, tma_load_b,
     tensorD.data(), dD, sD,
     tensorSFD.data(), sfd_layout,
     tensorQA.data(), dQA,
     tensorSFA.data(), sfa_layout,
     mma, a_global_amax, d_global_amax, rng_state);

  NVTE_CHECK_CUDA(cudaGetLastError());
  NVTE_CHECK(status == cutlass::Status::kSuccess, "Kernel launch failed.");

}

}  // namespace
}  // namespace detail

// clang-format on

void hadamard_transform_cast_fusion(const Tensor &input_, Tensor &output_,
                                    const Tensor &hadamard_matrix_, QuantizationConfig quant_config,
                                    cudaStream_t stream) {
  NVTE_API_CALL(hadamard_transform_cast_fusion);

  // Check input and output tensors
  NVTE_CHECK(input_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be BF16 tensor, but scaling mode is ",
             to_string(input_.scaling_mode), ".");
  NVTE_CHECK(input_.dtype() == transformer_engine::DType::kBFloat16,
             "Input tensor must be BF16 tensor, but dtype is ", to_string(input_.dtype()), ".");
  NVTE_CHECK(input_.dim() >= 2, "Input must be a 2D tensor.");
  const SimpleTensor &input = input_.data;

  // rowwise cast and columnwise cast has different output data pointers
  bool has_rowwise_quant = false;
  bool has_columnwise_quant = false;
  void *rowwise_data_ptr = nullptr;
  void *rowwise_scale_inv_ptr = nullptr;
  void *rowwise_amax_ptr = nullptr;
  void *columnwise_data_ptr = nullptr;
  void *columnwise_scale_inv_ptr = nullptr;
  void *columnwise_amax_ptr = nullptr;

  // examine the output tensor (single tensor for dense)
  if (output_.data.dptr != nullptr) {
    has_rowwise_quant = true;
    rowwise_data_ptr = output_.data.dptr;
    rowwise_scale_inv_ptr = output_.scale_inv.dptr;
    rowwise_amax_ptr = output_.amax.dptr;
  }

  if (output_.columnwise_data.dptr != nullptr) {
    has_columnwise_quant = true;
    columnwise_data_ptr = output_.columnwise_data.dptr;
    columnwise_scale_inv_ptr = output_.columnwise_scale_inv.dptr;
    columnwise_amax_ptr = output_.columnwise_amax.dptr;
  }

  NVTE_CHECK(has_rowwise_quant || has_columnwise_quant,
             "Output tensor must have rowwise or columnwise quant.");

  // Stochastic rounding config
  const bool use_stochastic_rounding = quant_config.stochastic_rounding;
  const size_t *rng_state = nullptr;
  if (quant_config.rng_state != nullptr) {
    Tensor &rng_state_tensor = *convertNVTETensor(quant_config.rng_state);
    NVTE_CHECK(rng_state_tensor.dtype() == DType::kInt64,
               "RNG state should contain 2 64-bit values.");
    NVTE_CHECK(rng_state_tensor.data.shape == std::vector<size_t>{2},
               "Shape of the RNG state should be [2], but got ", rng_state_tensor.data.shape);
    rng_state = reinterpret_cast<const size_t *>(rng_state_tensor.data.dptr);
  }

  // Template arguments
  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  using TD = cutlass::float_e2m1_t;
  using TSFD = cutlass::float_ue4m3_t;
  using TQA = TD;
  using TSFA = TSFD;

  checkCuDriverContext(stream);

  // Check Hadamard matrix
  constexpr int kHadamardDimension = 16;
  NVTE_CHECK(hadamard_matrix_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Hadamard matrix must be BF16 tensor, but scaling mode is ",
             to_string(hadamard_matrix_.scaling_mode), ".");
  NVTE_CHECK(hadamard_matrix_.dtype() == transformer_engine::DType::kBFloat16,
             "Hadamard matrix must be BF16 tensor, but dtype is ",
             to_string(hadamard_matrix_.dtype()), ".");
  const SimpleTensor &hadamard_matrix = hadamard_matrix_.data;
  NVTE_CHECK(
      (hadamard_matrix_.shape() == std::vector<size_t>{kHadamardDimension, kHadamardDimension}),
      "Hadamard matrix must have shape=",
      std::vector<size_t>{kHadamardDimension, kHadamardDimension},
      ", but got shape=", hadamard_matrix_.shape(), ".");
  const size_t hadamard_dimension = hadamard_matrix.shape[0];

  const size_t ndim = input.shape.size();
  const size_t n = input.shape[ndim - 1];
  size_t m = 1;
  for (size_t i = 0; i < ndim - 1; ++i) {
    m *= input.shape[i];
  }

  auto sm_count = transformer_engine::cuda::sm_count();

  NVTE_CHECK(n % hadamard_dimension == 0, "row_length must be divisible by hadamard_dimension.");

  NVTE_CHECK(m % hadamard_dimension == 0, "num_rows must be divisible by hadamard_dimension");

  int k_tile_size = 1024;

  // TODO: add support for swizzle sf output
  const bool use_swizzle_sf_output = false;

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_stochastic_rounding, kEnableStochasticRounding,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          has_columnwise_quant, kEnableRhtColQuant,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              has_rowwise_quant, kEnableRowQuant,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  use_swizzle_sf_output, kEnableSwizzleSFOutput,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      quant_config.use_fast_math, kUseFastMath,

                      if constexpr (kEnableRhtColQuant || kEnableRowQuant) {
                        detail::row_col_rht_gemm_ntt_w_sfc<
                            kEnableStochasticRounding, kEnableRhtColQuant, kEnableRowQuant,
                            kEnableSwizzleSFOutput, TA, TB, TD, TSFD, TQA, TSFA, kUseFastMath>(
                            /*sequence_length=*/m, /*hidden_size=*/n,
                            /*A=*/reinterpret_cast<TA const *>(input.dptr),
                            /*B=*/reinterpret_cast<TB const *>(hadamard_matrix.dptr),
                            /*D=*/reinterpret_cast<TD *>(columnwise_data_ptr),
                            /*SFD=*/reinterpret_cast<TSFD *>(columnwise_scale_inv_ptr),
                            /*QA=*/reinterpret_cast<TQA *>(rowwise_data_ptr),
                            /*SFA=*/reinterpret_cast<TSFA *>(rowwise_scale_inv_ptr),
                            /*a_global_amax=*/reinterpret_cast<float const *>(rowwise_amax_ptr),
                            /*d_global_amax=*/reinterpret_cast<float const *>(columnwise_amax_ptr),
                            /*rng_state=*/rng_state, /*sm_count=*/sm_count,
                            /*stream=*/stream, /*k_tile_size=*/k_tile_size);
                      } else {
                        NVTE_ERROR("Invalid kernel configuration (kEnableRHTColQuant=",
                                   kEnableRhtColQuant, ", kEnableRowQuant=", kEnableRowQuant, ").");
                      }

                  );););););
}

}  // namespace transformer_engine

void nvte_hadamard_transform_cast_fusion(const NVTETensor input, NVTETensor output,
                                         const NVTETensor hadamard_matrix,
                                         const NVTEQuantizationConfig quant_config,
                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_hadamard_transform_cast_fusion);
  using namespace transformer_engine;
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }
  hadamard_transform_cast_fusion(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
                                 *convertNVTETensorCheck(hadamard_matrix), quant_config_cpp,
                                 stream);
}
