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

namespace transformer_engine {
namespace detail {
namespace {

using namespace cute;

// Ensure Tensor refers to cute::Tensor, not transformer_engine::Tensor
using cute::Tensor;

constexpr int kMaxTensorsPerKernel = 64;

struct MultiAmaxHadamardCastFusionArgs {
  // (output) Amax buffer for input A amax buffer
  void *global_a_amax_list[kMaxTensorsPerKernel];
  // (output) Amax buffer for pre-RHT amax buffer
  void *global_d_amax_list[kMaxTensorsPerKernel];
  // output D pointers for each tensor
  void *output_colwise_list[kMaxTensorsPerKernel];
  // output SFD inverse pointers for each tensor
  void *output_colwise_scale_inv_list[kMaxTensorsPerKernel];
  // split sections of each tensor of input
  int split_sections[kMaxTensorsPerKernel];
  // Prefix sum (with leading zero) of split_sections of each tensor of input
  int split_sections_range[kMaxTensorsPerKernel + 1];

  // Number of tensors (splits) being processed by kernel
  int num_tensors;
};

__device__ __forceinline__ int GetGroupIdx(MultiAmaxHadamardCastFusionArgs *kernel_args_ptr,
                                           int offset) {
  // Check the kernel args and get the corresponding id
  const int num_tensors = kernel_args_ptr->num_tensors;
  if (offset >= kernel_args_ptr->split_sections_range[num_tensors]) {
    return num_tensors - 1;
  }
  int group_idx = 0;
  while (kernel_args_ptr->split_sections_range[group_idx + 1] <= offset) {
    ++group_idx;
  }
  return group_idx;
}

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
cutlass::Array<cutlass::float_e2m1_t, 16> StochasticNumericConverter(
    cutlass::Array<float, 16> const &input, cutlass::Array<uint32_t, 4> const &rbits) {
  using result_type = cutlass::Array<cutlass::float_e2m1_t, 16>;
  result_type output;
  cutlass::Array<cutlass::float_e2m1_t, 8> *result_ptr =
      reinterpret_cast<cutlass::Array<cutlass::float_e2m1_t, 8> *>(&output);
  cutlass::Array<float, 8> const *source_ptr =
      reinterpret_cast<cutlass::Array<float, 8> const *>(&input);
  cutlass::Array<uint32_t, 2> const *rbits_ptr =
      reinterpret_cast<cutlass::Array<uint32_t, 2> const *>(&rbits);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; i++) {
    result_ptr[i] = StochasticNumericConverterBase(source_ptr[i], rbits_ptr[i]);
  }
  return output;
}

template <class ElementA, class ElementB, class ASmemLayout, class BSmemLayout, class ClusterShape,
          int AccumulatorPipelineStageCount_, int EpilogueUnrollFactor_,
          int SchedulerPipelineStageCount_>
struct SharedStorage {
  static int constexpr AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
  static int constexpr EpilogueUnrollFactor = EpilogueUnrollFactor_;
  using AtomThrShapeMNK = cute::Shape<_1, _1, _1>;

  using AccumulatorPipeline =
      cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / EpilogueUnrollFactor,
                                 AtomThrShapeMNK>;
  using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;

  static int constexpr MainloopPipelineStageCount = size<3>(ASmemLayout{});
  using MainloopPipeline =
      cutlass::detail::CustomizedPipelineTmaUmmaAsync<MainloopPipelineStageCount, Shape<_1, _1, _1>,
                                                      AtomThrShapeMNK>;
  using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;
  using SchedPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount_, ClusterShape>;
  using SchedPipelineStorage = typename SchedPipeline::SharedStorage;
  using SchedThrottlePipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount_>;
  using SchedThrottlePipelineStorage = typename SchedThrottlePipeline::SharedStorage;

  struct TensorStorage : cute::aligned_struct<128, _1> {
    cute::array_aligned<ElementA, cute::cosize_v<ASmemLayout>> smem_A;
    cute::array_aligned<ElementB, cute::cosize_v<BSmemLayout>> smem_B;
  } tensors;

  alignas(16) AccumulatorPipelineStorage accumulator;
  alignas(16) MainloopPipelineStorage mainloop;
  alignas(16) cute::uint64_t tma_barrier[1];
  alignas(16) SchedPipelineStorage sched;
  alignas(16) SchedThrottlePipelineStorage sched_throttle;
  alignas(16) int32_t atomic_tile_id[SchedulerPipelineStageCount_];
  alignas(16) float global_a_amax[kMaxTensorsPerKernel];
  alignas(16) float global_d_amax[kMaxTensorsPerKernel];
  uint32_t atomic_tile_counter[SchedulerPipelineStageCount_];
  uint32_t tmem_base_ptr;
};

// Main RHT GEMM kernel entry -- highly templated for flexible architecture/config support
template <class MShape, class NShape, class KShape, class ClusterShape, class ClusterTileShape,
          class TA, class AStride, class ASmemLayout, class TmaLoadA, class TB, class BStride,
          class BSmemLayout, class TmaLoadB, class TD, class DStride, class DSmemLayout, class TSFD,
          class TSFDLayout, class TQA, class QAStride, class TSFA, class TSFALayout, class TiledMMA,
          int AccumulatorPipelineStageCount_, int SchedulerPipelineStageCount_,
          bool kEnableStochasticRounding_ = false, bool kEnableRHTColQuant_ = true,
          bool kEnableRowQuant_ = true, bool kEnableSwizzleSFOutput_ = false,
          bool kUseFastMath_ = false>
__launch_bounds__(512, 1) __global__ static void group_row_col_rht_gemm_device(
    MShape M, NShape packed_N, KShape K, ClusterShape cluster_shape, ClusterTileShape cluster_tile,
    TA const *A, AStride dA, ASmemLayout sAlayout, CUTE_GRID_CONSTANT TmaLoadA const tma_load_a,
    TB const *B, BStride dB, BSmemLayout sBlayout, CUTE_GRID_CONSTANT TmaLoadB const tma_load_b,
    TQA *QA, QAStride dQA, TSFA *SFA, TSFALayout sfa_layout, MultiAmaxHadamardCastFusionArgs args,
    uint32_t *tile_scheduler_workspace, TiledMMA mma, const size_t *rng_state) {
  using namespace cute;

  // Abort immediately if compilation is not supported
  constexpr bool is_blackwell_arch = ARCH_BLACKWELL_FAMILY;
  if constexpr (!is_blackwell_arch) {
    NVTE_DEVICE_ERROR(
        "group_row_col_rht_gemm_device is only supported on Blackwell "
        "with architecture-specific compilation. "
        "Try recompiling with sm_100a or similar.");
    return;
  }
  static_assert(kEnableRHTColQuant_ || kEnableRowQuant_,
                "group_row_col_rht_gemm_device must generate row-wise "
                "and/or column-wise output.");
#if !defined(CUTLASS_ARCH_CLC_ENABLED)
  CUTLASS_NOT_IMPLEMENTED();
  return;
#endif

  using X = Underscore;
  // Accumulator data type for main computation
  using ElementAccumulator = float;
  static int constexpr K_PIPE_MAX = size<3>(ASmemLayout{});
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMMA::ThrLayoutVMNK{})), _1, _1>;
  static uint32_t constexpr kTmaTransactionBytes = cutlass::bits_to_bytes(
      size(AtomThrShapeMNK{}) * cosize(take<0, 3>(ASmemLayout{})) * cute::sizeof_bits_v<TA>);
  static constexpr bool kEnableStochasticRounding = kEnableStochasticRounding_;
  static constexpr bool kEnableRHTColQuant = kEnableRHTColQuant_;
  static constexpr bool kEnableRowQuant = kEnableRowQuant_;
  static constexpr bool kEnableSwizzleSFOutput = kEnableSwizzleSFOutput_;
  static constexpr bool kUseFastMath = kUseFastMath_;

  // Constant for RHT tensor processing (tile size etc)
  static int constexpr RhtTensorSize = 16;

  // Transaction bytes for TMA transfer on RHT tensor blocks
  static int constexpr kTmaRhtTensorTransactionBytes =
      cutlass::bits_to_bytes(RhtTensorSize * RhtTensorSize * cute::sizeof_bits_v<TB>);
  static int constexpr AccumulatorPipelineStageCount = AccumulatorPipelineStageCount_;
  static int constexpr SchedulerPipelineStageCount = SchedulerPipelineStageCount_;

  // Mainloop pipeline stage calculation, vectorization parameters for scaling factors
  static int constexpr MainloopPipelineStageCount = size<3>(ASmemLayout{});
  static int constexpr SFVecSize = 16;
  // Swizzle output layout for scaling factor arrays
  using SwizzledSFALayoutAtom =
      cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVecSize, UMMA::Major::MN>::SfAtom;
  using SwizzledSFDLayoutAtom =
      cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVecSize, UMMA::Major::K>::SfAtom;

  // Mainloop pipeline types for TMA async execution and epilogue cluster scheduling
  using MainloopPipeline =
      cutlass::detail::CustomizedPipelineTmaUmmaAsync<MainloopPipelineStageCount, ClusterShape,
                                                      AtomThrShapeMNK>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;
  using SchedPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using SchedPipelineState = typename SchedPipeline::PipelineState;
  using SchedThrottlePipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount>;
  using SchedThrottlePipelineState = typename SchedThrottlePipeline::PipelineState;

  static_assert(ClusterShape{} == Shape<_1, _1, _1>{}, "ClusterShape must be Shape<_1,_1,_1>");

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  static int constexpr VectorSize = RhtTensorSize;

  // Compile-time safety: static shapes required for shared memory layouts
  CUTE_STATIC_ASSERT(is_static<ASmemLayout>::value);
  CUTE_STATIC_ASSERT(is_static<BSmemLayout>::value);
  //   CUTE_STATIC_ASSERT(is_static<DSmemLayout>::value);

  auto cluster_size = size<0>(cluster_shape);
  auto mainloop_tiler = Shape<_128, _16, _128>{};
  auto epilogue_tiler = Shape<_128, _128, _128>{};

  static int constexpr EpilogueUnrollFactor = size<2>(epilogue_tiler) / size<2>(cluster_tile);

  // Get the appropriate blocks for this Cluster
  dim3 cluster_coord_in_grid = cluster_id_in_grid();

  // Total number of k-tiles
  int const K_TILE_MAX = min(packed_N, K) / size<2>(epilogue_tiler);

  struct TileScheduler {
    uint32_t tiles_in_m = 0;
    uint32_t tiles_in_n = 0;
    uint32_t linear_idx = 0;
    uint32_t next_linear_idx = 0;
    uint32_t start_idx = 0;
    uint32_t tile_m_idx = 0;
    uint32_t tile_n_idx = 0;
    int k_tile_max = 0;
    uint32_t *atomic_tile_index_;
    uint32_t *smem_tile_counter;
    uint32_t atomic_offset;
    cutlass::FastDivmodU64 divmod_tiles_in_m;

    CUTLASS_DEVICE TileScheduler(uint32_t tiles_m, uint32_t tiles_n, int kmax,
                                 uint32_t *atomic_tile_index, uint32_t *smem_tile_counter)
        : tiles_in_m(tiles_m),
          tiles_in_n(tiles_n),
          linear_idx(blockIdx.x),
          next_linear_idx(blockIdx.x),
          start_idx(blockIdx.x),
          k_tile_max(kmax),
          atomic_tile_index_(atomic_tile_index),
          smem_tile_counter(smem_tile_counter),
          atomic_offset(gridDim.x),
          divmod_tiles_in_m(uint64_t(tiles_m)) {
      update_tile_idx();
    }
    CUTLASS_DEVICE void update_tile_idx() {
      uint64_t q, r;
      divmod_tiles_in_m(q, r, uint64_t(linear_idx));
      tile_m_idx = static_cast<uint32_t>(r);
      tile_n_idx = static_cast<uint32_t>(q) * uint32_t(k_tile_max);
    }
    CUTLASS_DEVICE uint32_t tile_m() const { return tile_m_idx; }
    CUTLASS_DEVICE uint32_t tile_n_base() const { return tile_n_idx; }
    CUTLASS_DEVICE uint32_t tiles_m() const { return tiles_in_m; }

    CUTLASS_DEVICE uint32_t tiles_n() const { return tiles_in_n; }

    CUTLASS_DEVICE bool is_valid() const {
      return cute::elem_less(cute::make_coord(tile_m(), tile_n_base()),
                             cute::make_coord(tiles_in_m, tiles_in_n));
    }

    CUTLASS_DEVICE bool is_first_wave() const { return linear_idx == start_idx; }

    CUTLASS_DEVICE uint32_t get_linear_tile_idx() const { return linear_idx; }

    // Fetch a new tile_id using atomics.
    CUTLASS_DEVICE uint32_t fetch_tile_id_counter(int pred) {
      uint32_t tile_id_counter = 0;
      asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.eq.u32 p, %2, 1;\n\t"
          "@p atom.global.add.u32 %0, [%1], 1; \n\t"
          "}"
          : "=r"(tile_id_counter)
          : "l"(atomic_tile_index_), "r"(pred));

      return tile_id_counter;
    }

    CUTLASS_DEVICE auto fetch_next_work(SchedPipeline &sched_pipeline,
                                        SchedPipelineState sched_pipeline_consumer_state) {
      sched_pipeline.consumer_wait(sched_pipeline_consumer_state);
      next_linear_idx = smem_tile_counter[sched_pipeline_consumer_state.index()];
      cutlass::arch::fence_view_async_shared();
      sched_pipeline.consumer_release(sched_pipeline_consumer_state);
      return;
    }

    CUTLASS_DEVICE auto advance_to_next_work(SchedPipeline &sched_pipeline,
                                             SchedPipelineState sched_pipeline_producer_state) {
      uint32_t mbarrier_addr = sched_pipeline.producer_get_barrier(sched_pipeline_producer_state);
      // Wait for clcID buffer to become empty with a flipped phase
      sched_pipeline.producer_acquire(sched_pipeline_producer_state);
      auto is_leading_thread = cute::elect_one_sync();
      uint32_t tile_id_counter = fetch_tile_id_counter(is_leading_thread) + atomic_offset;
      uint32_t smem_addr =
          cute::cast_smem_ptr_to_uint(&smem_tile_counter[sched_pipeline_producer_state.index()]);
      if (is_leading_thread) {
        cute::store_shared_remote(tile_id_counter, smem_addr, mbarrier_addr, 0);
      }

      ++sched_pipeline_producer_state;
      return sched_pipeline_producer_state;
    }

    CUTLASS_DEVICE auto update_work_tile_info() {
      linear_idx = next_linear_idx;
      update_tile_idx();
      return;
    }
  };

  // Allocate and alias shared memory to the kernel's shared storage type
  extern __shared__ char shared_memory[];
  using SharedStorage =
      SharedStorage<TA, TB, ASmemLayout, BSmemLayout, ClusterShape, AccumulatorPipelineStageCount,
                    EpilogueUnrollFactor, SchedulerPipelineStageCount>;
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

  // Compute the number of tiles in M and N after tiling and assign scheduler
  uint32_t tiles_in_m = uint32_t(size(ceil_div(M, size<0>(cluster_tile))));
  uint32_t tiles_in_n = uint32_t(
      size(ceil_div(args.split_sections_range[args.num_tensors], size<2>(epilogue_tiler))));

  TileScheduler scheduler(tiles_in_m, tiles_in_n, K_TILE_MAX, tile_scheduler_workspace,
                          shared_storage.atomic_tile_counter);

  int block_rank_in_cluster = cute::block_rank_in_cluster();

  // Shapes for accumulated tiles in mainloop and epilogue
  auto acc_shape_mma = make_shape(take<0, 2>(mainloop_tiler), _1{}, _1{});
  auto acc_shape_epilogue = make_shape(take<0, 2>(epilogue_tiler), _1{}, _1{});

  // Shape of the accumulator fragment for the main loop pipeline, with pipeline stages appended
  auto acc_mainloop_pipelined_shape = append(acc_shape_mma, Int<AccumulatorPipelineStageCount>{});
  auto bulk_tmem_mma = TiledMMA::make_fragment_C(acc_mainloop_pipelined_shape);

  // Number of threads assigned for various epilogue roles depending on quantization settings
  static int constexpr NumEpilogueColQuantThreadCount = kEnableRHTColQuant ? 128 : 0;
  static int constexpr NumEpilogueRowQuantThreadCount = kEnableRowQuant ? 256 : 0;
  static int constexpr NumMmaThreadCount = kEnableRHTColQuant ? 32 : 0;
  static int constexpr NumMmaIssueThreadCount = kEnableRHTColQuant ? 1 : 0;
  static int constexpr NumSchedThreads = 32;
  static int constexpr NumMainloopLoadThreads = 32;
  static int constexpr NumEpilogueThreads =
      NumEpilogueColQuantThreadCount + NumEpilogueRowQuantThreadCount;

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

  MainloopPipeline mainloop_pipeline(shared_storage.mainloop, mainloop_pipeline_params,
                                     cluster_shape, cute::true_type{},  // Perform barrier init
                                     cute::true_type{});                // Delay mask calculation

  MainloopPipelineState mainloop_pipe_consumer_state;
  MainloopPipelineState mainloop_pipe_producer_state =
      cutlass::make_producer_start_state<MainloopPipeline>();

  using AccumulatorPipeline =
      cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / EpilogueUnrollFactor,
                                 AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;
  using AccumulatorPipelineInitBarriers = cute::bool_constant<kEnableRHTColQuant>;

  AccumulatorPipelineState accumulator_pipe_consumer_state;
  AccumulatorPipelineState accumulator_pipe_producer_state =
      cutlass::make_producer_start_state<AccumulatorPipeline>();

  typename AccumulatorPipeline::Params accumulator_pipeline_params;
  if (is_mma_warp) {
    accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
  }
  if (is_epilogue_col_quant_warp) {
    accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
  }
  // Only one producer thread arrives on this barrier.
  accumulator_pipeline_params.producer_arv_count = 1;
  accumulator_pipeline_params.consumer_arv_count =
      size(AtomThrShapeMNK{}) * NumEpilogueColQuantThreadCount;
  accumulator_pipeline_params.initializing_warp = 1;
  AccumulatorPipeline accumulator_pipeline(shared_storage.accumulator, accumulator_pipeline_params,
                                           cluster_shape, AccumulatorPipelineInitBarriers{},
                                           cute::true_type{});  // Delay mask calculation
  typename SchedPipeline::Params sched_pipeline_params;
  if (is_sched_warp) {
    sched_pipeline_params.role = SchedPipeline::ThreadCategory::ProducerConsumer;
  } else {
    sched_pipeline_params.role = SchedPipeline::ThreadCategory::Consumer;
  }
  sched_pipeline_params.producer_blockid = 0;
  sched_pipeline_params.producer_arv_count = 1;
  sched_pipeline_params.consumer_arv_count =
      NumSchedThreads +
      cluster_size * (NumMainloopLoadThreads + NumEpilogueThreads + NumMmaThreadCount);
  sched_pipeline_params.transaction_bytes = sizeof(uint32_t);
  sched_pipeline_params.initializing_warp = 3;
  SchedPipeline sched_pipeline(shared_storage.sched, sched_pipeline_params, cluster_shape);
  SchedPipelineState sched_pipeline_consumer_state;
  SchedPipelineState sched_pipeline_producer_state =
      cutlass::make_producer_start_state<SchedPipeline>();

  typename SchedThrottlePipeline::Params sched_throttle_pipeline_params;
  if (is_dma_warp) {
    sched_throttle_pipeline_params.role = SchedThrottlePipeline::ThreadCategory::Producer;
  }
  if (is_sched_warp) {
    sched_throttle_pipeline_params.role = SchedThrottlePipeline::ThreadCategory::Consumer;
  }
  sched_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
  sched_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
  sched_throttle_pipeline_params.dst_blockid = 0;
  sched_throttle_pipeline_params.initializing_warp = 4;

  SchedThrottlePipeline sched_throttle_pipeline(shared_storage.sched_throttle,
                                                sched_throttle_pipeline_params);
  SchedThrottlePipelineState sched_pipeline_throttle_consumer_state;
  SchedThrottlePipelineState sched_pipeline_throttle_producer_state =
      cutlass::make_producer_start_state<SchedThrottlePipeline>();

  if (warp_idx == 2 && elect_one_sync()) {
    cute::initialize_barrier(shared_storage.tma_barrier[0], /* num_threads */ 1);
  }
  __syncthreads();

  // Warp group roles: DMA (global->shared copy), MMA (tensor core gemm), scheduler, column quantizer, row quantizer
  if (is_dma_warp) {
    // Warp responsible for loading input from global to shared memory using TMA (Tensor Memory Access).
    cutlass::arch::warpgroup_reg_dealloc<32>();
    // Get TMA tensors for input matrix A and B (Hadamard/transform matrix) from global memory.
    Tensor mA = tma_load_a.get_tma_tensor(make_shape(M, packed_N));
    Tensor mB = tma_load_b.get_tma_tensor(make_shape(RhtTensorSize, RhtTensorSize));

    // Partition tensors for tiling according to the mainloop and cluster tilers.
    Tensor gA_mk = local_tile(mA, mainloop_tiler, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor gB_nk =
        local_tile(mB, cluster_tile, make_coord(_, _, _), Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)

    // Shared memory tensors for pipeline
    Tensor tCsA = make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()),
                              sAlayout);  // (MMA,MMA_M,MMA_N,PIPE)
    Tensor tCsB = make_tensor(make_smem_ptr(shared_storage.tensors.smem_B.data()),
                              sBlayout);  // (MMA,MMA_N,MMA_K,PIPE)

    // Determine warp/tile positioning
    int block_rank_in_cluster = cute::block_rank_in_cluster();
    ThrMMA thr_mma = mma.get_slice(block_rank_in_cluster);  // blk idx
    // Partition global to local fragments for A and B
    Tensor tCgA = thr_mma.partition_A(gA_mk);  // (MMA,MMA_M,MMA_K,k)
    Tensor tCgB = thr_mma.partition_B(gB_nk);  // (MMA,MMA_N,MMA_K,k)

    Layout cta_layout_mnk = make_layout(cluster_shape);
    Layout cta_layout_vmnk =
        tiled_divide(cta_layout_mnk, make_tile(typename TiledMMA::AtomThrID{}));
    auto cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

    auto [tAgA, tAsA] =
        tma_partition(tma_load_a, get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                      group_modes<0, 3>(tCsA), group_modes<0, 3>(tCgA));

    auto [tBgB, tBsB] =
        tma_partition(tma_load_b, get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                      group_modes<0, 3>(tCsB), group_modes<0, 3>(tCgB));

    uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);
    if constexpr (kEnableRHTColQuant) {
      if (elect_one_sync()) {
        cute::set_barrier_transaction_bytes(shared_storage.tma_barrier[0],
                                            kTmaRhtTensorTransactionBytes);
        copy(tma_load_b.with(shared_storage.tma_barrier[0], tma_mcast_mask_b), tBgB(_, 0, 0),
             tBsB(_, 0));
      }
    }

    do {
      // is_first_wave indicates whether this scheduler wave is the first among a group.
      bool is_first_wave = scheduler.is_first_wave();
      uint32_t skip_wait = is_first_wave;
      auto tAgA_mk = tAgA(_, scheduler.tile_m(), _);
      int k_tile = 0;

      sched_throttle_pipeline.producer_acquire(sched_pipeline_throttle_producer_state);
      sched_throttle_pipeline.producer_commit(sched_pipeline_throttle_producer_state);
      ++sched_pipeline_throttle_producer_state;
      CUTLASS_PRAGMA_NO_UNROLL
      while (k_tile < K_TILE_MAX && k_tile + scheduler.tile_n_base() < scheduler.tiles_n()) {
        int k_tile_idx_n = scheduler.tile_n_base() + k_tile;
        ++k_tile;
        skip_wait = (is_first_wave && k_tile < MainloopPipelineStageCount);
        mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType *tma_barrier =
            mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);
        int write_stage = mainloop_pipe_producer_state.index();
        ++mainloop_pipe_producer_state;
        if (cute::elect_one_sync()) {
          copy(tma_load_a.with(*tma_barrier, tma_mcast_mask_a), tAgA_mk(_, k_tile_idx_n),
               tAsA(_, write_stage));
        }
      }
      scheduler.fetch_next_work(sched_pipeline, sched_pipeline_consumer_state);
      ++sched_pipeline_consumer_state;
      scheduler.update_work_tile_info();
      // scheduler.advance();
    } while (scheduler.is_valid());
    mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
  } else if (is_mma_warp) {
    // This warp executes the main tensor core matrix-multiply-accumulate for the Hadamard transform.
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if constexpr (kEnableRHTColQuant) {
      // Setup shared memory fragments for A and B tiles.
      Tensor tCsA = make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()),
                                sAlayout);  // (MMA,MMA_M,MMA_N,PIPE)
      Tensor tCsB = make_tensor(make_smem_ptr(shared_storage.tensors.smem_B.data()),
                                sBlayout);  // (MMA,MMA_N,MMA_K,PIPE)

      int block_rank_in_cluster = cute::block_rank_in_cluster();
      ThrMMA thr_mma = mma.get_slice(block_rank_in_cluster);  // blk idx
      // Allocate "fragments" -- these are actually umma smem descriptors
      Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
      Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_M,MMA_K,PIPE)

      mma.accumulate_ = UMMA::ScaleOut::Zero;

      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                              &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      bulk_tmem_mma.data() = tmem_base_ptr;
      // Wait until the B (Hadamard) tensor copy is complete
      cute::wait_barrier(shared_storage.tma_barrier[0], 0 /*tma_phase_bit*/);
      do {
        uint32_t skip_wait = K_TILE_MAX <= 0;

        auto barrier_token =
            mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);
        scheduler.fetch_next_work(sched_pipeline, sched_pipeline_consumer_state);
        ++sched_pipeline_consumer_state;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile = 0;
             k_tile < K_TILE_MAX && k_tile + scheduler.tile_n_base() < scheduler.tiles_n();) {
          mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);
          int read_stage = mainloop_pipe_consumer_state.index();
          auto tCrA_mk = tCrA(_, _, _, read_stage);
          auto tCrB_nk = tCrB(_, _, 0, 0);
          CUTLASS_PRAGMA_UNROLL
          for (int k_block = 0; k_block < size<2>(tCrA) / EpilogueUnrollFactor; ++k_block) {
            int accumulator_k_block =
                accumulator_pipe_producer_state.index() * EpilogueUnrollFactor;
            int tCrA_k_block = k_block * EpilogueUnrollFactor;
            accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < EpilogueUnrollFactor; i++) {
              auto accumulators = bulk_tmem_mma(_, _, _, accumulator_k_block + i);
              gemm(mma, tCrA_mk(_, _, tCrA_k_block + i), tCrB_nk, accumulators);
            }

            accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
            ++accumulator_pipe_producer_state;
          }
          auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;
          ++mainloop_pipe_consumer_state;
          ++k_tile;
          skip_wait = k_tile >= K_TILE_MAX;
          mainloop_pipeline.umma_consumer_release(curr_mainloop_pipe_consumer_state);
          barrier_token =
              mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);
        }
        scheduler.update_work_tile_info();
      } while (scheduler.is_valid());
      tmem_allocator.release_allocation_lock();
      accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
      tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
  } else if (is_sched_warp) {
    // Scheduler warp manages tile assignment and pipeline progress for warps
    cutlass::arch::warpgroup_reg_dealloc<32>();
    do {
      sched_throttle_pipeline.consumer_wait(sched_pipeline_throttle_consumer_state);
      sched_throttle_pipeline.consumer_release(sched_pipeline_throttle_consumer_state);
      ++sched_pipeline_throttle_consumer_state;
      sched_pipeline_producer_state =
          scheduler.advance_to_next_work(sched_pipeline, sched_pipeline_producer_state);
      scheduler.fetch_next_work(sched_pipeline, sched_pipeline_consumer_state);
      ++sched_pipeline_consumer_state;
      scheduler.update_work_tile_info();
    } while (scheduler.is_valid());
  } else if (is_epilogue_col_quant_warp) {
    // Warp responsible for quantizing output of Hadamard transform to FP4 for columnwise usage,
    // and writing result tensors/scales to global memory.
    cutlass::arch::warpgroup_reg_alloc<192>();
    if constexpr (kEnableRHTColQuant) {
      using TMEM_LOAD_NEW = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b64x;

      auto acc_epilogue_pipelined_shape =
          append(acc_shape_epilogue, Int<AccumulatorPipelineStageCount / EpilogueUnrollFactor>{});
      auto bulk_tmem_epilogue_layout = make_layout(
          acc_epilogue_pipelined_shape,
          make_stride(stride<0>(bulk_tmem_mma), Int<0>{}, Int<0>{}, size<1>(epilogue_tiler)));
      auto bulk_tmem_epilogue = make_tensor(make_tmem_ptr<uint32_t>(), bulk_tmem_epilogue_layout);

      // Use 256-bit fragments for aligned bulk stores
      static int constexpr FragmentSize = 256 / sizeof_bits_v<TD>;

      // Wait for TMEM allocation for this pipeline to finish
      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      bulk_tmem_epilogue.data() = tmem_base_ptr;
      int global_thread_idx = threadIdx.x;
      int local_thread_idx = global_thread_idx % cutlass::NumThreadsPerWarpGroup;
      // g2s load all global_d_amax
      CUTLASS_PRAGMA_NO_UNROLL
      for (int g = local_thread_idx; g < args.num_tensors; g += NumEpilogueColQuantThreadCount) {
        shared_storage.global_d_amax[g] =
            __ldg(reinterpret_cast<float *>(args.global_d_amax_list[g]));
      }

      size_t rng_seed = 0;
      size_t rng_offset = 0;
      // Setup RNG for stochastic rounding
      if constexpr (kEnableStochasticRounding) {
        rng_seed = rng_state != nullptr ? __ldg(rng_state) : 0;
        rng_offset = rng_state != nullptr ? __ldg(rng_state + 1) : 0;
      }
      int group_idx = GetGroupIdx(&args, scheduler.tile_n_base() * size<1>(epilogue_tiler));

      // Determine quantization scale factor layouts/output splits for this group
      TSFDLayout sfd_layout;
      int cur_N = args.split_sections[group_idx];
      if constexpr (kEnableSwizzleSFOutput) {
        sfd_layout = tile_to_shape(SwizzledSFDLayoutAtom{}, make_shape(M, cur_N), Step<_2, _1>{});
      } else {
        sfd_layout = make_layout(make_shape(M, make_shape(Int<SFVecSize>{}, cur_N / SFVecSize)),
                                 make_stride(cur_N / SFVecSize, make_stride(_0{}, _1{})));
      }
      // Build output tensors for columns and their quant scales
      Tensor mD = make_tensor(
          cute::subbyte_iterator<TD>(reinterpret_cast<TD *>(args.output_colwise_list[group_idx])),
          make_shape(M, cur_N), DStride{});  // (M,packed_N)
      Tensor gD_mn =
          local_tile(mD, epilogue_tiler, make_coord(_, _, _), Step<_1, _1, X>{});  // (BLK_M,BLK_N)

      Tensor mSFD = make_tensor(make_gmem_ptr<TSFD>(reinterpret_cast<TSFD *>(
                                    args.output_colwise_scale_inv_list[group_idx])),
                                sfd_layout);
      Tensor gSFD_mn = local_tile(mSFD, epilogue_tiler, make_coord(_, _, _),
                                  Step<_1, _1, X>{});  // (BLK_M,BLK_N)

      Tensor gD_mn_view = tiled_divide(gD_mn, take<0, 2>(epilogue_tiler));

      // Setup tile-level TMEM (t2r) and global memory (r2g) copy descriptors
      auto tiled_t2r = make_tmem_copy(TMEM_LOAD_NEW{}, bulk_tmem_epilogue(_, _, _, _0{}));
      auto tiled_r2g =
          make_tiled_copy_D(Copy_Atom<SM100_STORE_256bit_CACHE_NOALLOCATION, TD>{}, tiled_t2r);
      auto thr_t2r = tiled_t2r.get_slice(local_thread_idx);
      auto thr_r2g = tiled_r2g.get_slice(local_thread_idx);

      cutlass::arch::NamedBarrier::sync(NumEpilogueColQuantThreadCount,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      // Aligning with TensorEngine's recipe to generate scale factors // {$nv-internal-release}
      static constexpr float fp4_max = 6.0f;
      static constexpr float fp8_max = 448.0f;
      static constexpr float fp4_max_inv = 1.0f / fp4_max;
      float c_global_amax_val = shared_storage.global_d_amax[group_idx];
      float global_encode_scale = c_global_amax_val > 0.0f
                                      ? cutlass::minimum_with_nan_propagation<float>{}(
                                            (fp8_max * fp4_max) / c_global_amax_val,
                                            cutlass::platform::numeric_limits<float>::max())
                                      : 1.0f;
      float global_decode_scale = 1.0f / global_encode_scale;

      // Scaling factor for fast math path
      float global_encode_scale_multiplier = 1.0f;
      if constexpr (kUseFastMath) {
        global_encode_scale_multiplier = global_encode_scale * fp4_max_inv;
      }

      do {
        scheduler.fetch_next_work(sched_pipeline, sched_pipeline_consumer_state);
        ++sched_pipeline_consumer_state;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile = 0;
             k_tile < K_TILE_MAX && k_tile + scheduler.tile_n_base() < scheduler.tiles_n();
             ++k_tile) {
          int global_tile_n_offset = (scheduler.tile_n_base() + k_tile) * size<1>(epilogue_tiler);

          int cur_group_idx = GetGroupIdx(&args, global_tile_n_offset);

          if (cur_group_idx != group_idx) {
            group_idx = cur_group_idx;
            c_global_amax_val = shared_storage.global_d_amax[group_idx];
            // update amax
            global_encode_scale = c_global_amax_val > 0.0f
                                      ? cutlass::minimum_with_nan_propagation<float>{}(
                                            (fp8_max * fp4_max) / c_global_amax_val,
                                            cutlass::platform::numeric_limits<float>::max())
                                      : 1.0f;
            global_decode_scale = 1.0f / global_encode_scale;
            if constexpr (kUseFastMath) {
              global_encode_scale_multiplier = global_encode_scale * fp4_max_inv;
            }
            cur_N = args.split_sections[group_idx];
            if constexpr (kEnableSwizzleSFOutput) {
              sfd_layout =
                  tile_to_shape(SwizzledSFDLayoutAtom{}, make_shape(M, cur_N), Step<_2, _1>{});
            } else {
              sfd_layout =
                  make_layout(make_shape(M, make_shape(Int<SFVecSize>{}, cur_N / SFVecSize)),
                              make_stride(cur_N / SFVecSize, make_stride(_0{}, _1{})));
            }
            // update tensor
            mD = make_tensor(cute::subbyte_iterator<TD>(
                                 reinterpret_cast<TD *>(args.output_colwise_list[group_idx])),
                             make_shape(M, cur_N), DStride{});
            gD_mn = local_tile(mD, epilogue_tiler, make_coord(_, _, _),
                               Step<_1, _1, X>{});  // (BLK_M,BLK_N)
            mSFD = make_tensor(make_gmem_ptr<TSFD>(reinterpret_cast<TSFD *>(
                                   args.output_colwise_scale_inv_list[group_idx])),
                               sfd_layout);
            gSFD_mn = local_tile(mSFD, epilogue_tiler, make_coord(_, _, _),
                                 Step<_1, _1, X>{});  // (BLK_M,BLK_N)

            gD_mn_view = tiled_divide(gD_mn, take<0, 2>(epilogue_tiler));
          }
          int group_start_offset = args.split_sections_range[group_idx];
          int local_tile_n_idx =
              (global_tile_n_offset - group_start_offset) / size<1>(epilogue_tiler);
          Tensor tDgD_mn = gD_mn_view(_, _, _, scheduler.tile_m(), local_tile_n_idx);

          Tensor tDgSFD_mn = gSFD_mn(_, _, scheduler.tile_m(), local_tile_n_idx);
          accumulator_pipeline.consumer_wait(accumulator_pipe_consumer_state);

          auto Acc = bulk_tmem_epilogue(_, _, _, accumulator_pipe_consumer_state.index());
          Tensor tDtAcc = thr_t2r.partition_S(Acc);    // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
          Tensor tDgD = thr_t2r.partition_D(tDgD_mn);  // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)

          Tensor tTR_rAcc =
              make_tensor<ElementAccumulator>(shape(tDgD));  // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
          Tensor tDrD = make_tensor<TD>(shape(tDgD));
          Tensor tTR_rAcc_frag =
              recast<cutlass::Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc));
          Tensor tDrD_frag = recast<cutlass::Array<TD, FragmentSize>>(coalesce(tDrD));

          Tensor src = thr_r2g.retile_S(tDrD);
          Tensor dst = thr_r2g.retile_D(tDgD);

          Tensor tDgSFD_view = make_tensor(
              tDgSFD_mn.data(), make_layout(make_shape(shape(tDgSFD_mn), Int<1>{}, Int<1>{}),
                                            make_stride(stride(tDgSFD_mn), Int<0>{}, Int<0>{})));
          Tensor tDgSFD = filter(thr_t2r.partition_D(tDgSFD_view));
          Tensor tDrSFD = make_tensor<TSFD>(shape(tDgSFD));

          static int constexpr NumVecs = size(tDgD) / VectorSize;
          Tensor tD_rRowSFD_frg = recast<cutlass::Array<TSFD, NumVecs>>(tDrSFD);

          // Compute amax and quantization scales for this tile
          cutlass::maximum_absolute_value_reduction<cutlass::Array<ElementAccumulator, VectorSize>,
                                                    true>
              amax_reduction;
          cutlass::Array<ElementAccumulator, NumVecs> vec_maxs;
          cutlass::Array<ElementAccumulator, NumVecs> pvscales;
          // Copy from TMEM to registers
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

          auto compute_frgs = reinterpret_cast<cutlass::Array<ElementAccumulator, VectorSize> *>(
              tTR_rAcc_frag.data());
          auto output_frgs = reinterpret_cast<cutlass::Array<TD, VectorSize> *>(tDrD_frag.data());
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
          auto pvscales_cvted =
              cutlass::NumericArrayConverter<TSFD, ElementAccumulator, NumVecs>{}(pvscales);

          tD_rRowSFD_frg(_0{}) = pvscales_cvted;
          auto qpvscale_ups = cutlass::NumericArrayConverter<ElementAccumulator, TSFD, NumVecs>{}(
              tD_rRowSFD_frg(_0{}));
          auto qpvscale_scaled = cutlass::multiplies<cutlass::Array<ElementAccumulator, NumVecs>>{}(
              qpvscale_ups, global_decode_scale);
          cutlass::Array<ElementAccumulator, NumVecs> acc_scales;
          if constexpr (kUseFastMath) {
            // Fast math: compute approximate reciprocal
            acc_scales =
                cutlass::reciprocal_approximate_ftz<decltype(qpvscale_scaled)>{}(qpvscale_scaled);
          } else {
            // Accurate math: compute reciprocal with division
            acc_scales = cutlass::divides<cutlass::Array<ElementAccumulator, NumVecs>>{}(
                1.0, qpvscale_scaled);
          }

          // Prepare stochastic rounding random state if enabled
          uint4 random_uint4 = uint4{0, 0, 0, 0};
          transformer_engine::curanddx::detail::philox4x32_native_state<10> rng;
          // "Prefetch" a stochastic rounding state for the first tile
          if constexpr (kEnableStochasticRounding) {
            const size_t rng_sequence = global_thread_idx + k_tile * 512 +
                                        scheduler.get_linear_tile_idx() * K_TILE_MAX * 512;
            rng.init(rng_seed, rng_sequence, rng_offset);
          }
          CUTLASS_PRAGMA_UNROLL
          // Apply round/quantize to each fragment, with or without stochastic rounding
          for (int v = 0; v < NumVecs; v++) {
            auto acc_scale = cutlass::minimum_with_nan_propagation<ElementAccumulator>{}(
                acc_scales[v], cutlass::platform::numeric_limits<ElementAccumulator>::max());
            if constexpr (kEnableStochasticRounding) {
              random_uint4 = rng.generate4();
              output_frgs[v] = StochasticNumericConverter(
                  cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(
                      compute_frgs[v], acc_scale),
                  *reinterpret_cast<cutlass::Array<uint32_t, 4> *>(&random_uint4));
            } else {
              output_frgs[v] = cutlass::NumericArrayConverter<TD, ElementAccumulator, VectorSize>{}(
                  cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(
                      compute_frgs[v], acc_scale));
            }
          }

          // Write quantized FP4 tile and dequant scale to gmem
          copy(tiled_r2g, src, dst);
          copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tDrSFD, tDgSFD);
        }
        scheduler.update_work_tile_info();
      } while (scheduler.is_valid());
    }
  } else if (is_epilogue_row_quant_warp) {
    // Warp responsible for quantizing the input (before Hadamard transform) to FP4 for row-wise usage.
    cutlass::arch::warpgroup_reg_alloc<136>();
    if constexpr (kEnableRowQuant) {
      using S2RVectorType = uint128_t;

      int global_thread_idx = threadIdx.x;
      int local_thread_idx = global_thread_idx % 256;
      size_t rng_seed = 0;
      size_t rng_offset = 0;
      // g2s load all global_a_amax for all groups/tensors
      CUTLASS_PRAGMA_NO_UNROLL
      for (int g = local_thread_idx; g < args.num_tensors; g += NumEpilogueRowQuantThreadCount) {
        shared_storage.global_a_amax[g] =
            __ldg(reinterpret_cast<float *>(args.global_a_amax_list[g]));
      }
      // RNG for stochastic rounding
      if constexpr (kEnableStochasticRounding) {
        rng_seed = rng_state != nullptr ? __ldg(rng_state) : 0;
        rng_offset = rng_state != nullptr ? __ldg(rng_state + 1) : 0;
      }
      // Input/output tensors/partitions for row quant warp
      Tensor mQA =
          make_tensor(cute::subbyte_iterator<TQA>(QA), make_layout(make_shape(M, packed_N), dQA));
      Tensor gQA_mn = local_tile(mQA, epilogue_tiler, make_coord(_, _, _), Step<_1, X, _1>{});
      Tensor mSFA = make_tensor(make_gmem_ptr(SFA), sfa_layout);

      Tensor gSFA_mn = local_tile(mSFA, epilogue_tiler, make_coord(_, _, _),
                                  Step<_1, X, _1>{});  // (BLK_M,BLK_N)
      // Swizzled shared memory A tile, with layout
      Tensor sA = as_position_independent_swizzle_tensor(group_modes<0, 2>(
          coalesce(make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()),
                               sAlayout))));  // (BLOCK_M, BLOCK_M,PIPE)

      // Set up layouts for partitioning – tile-by-warp, with vector granularity
      using S2RWarpLayout = Layout<Shape<_8, _4>>;
      using WarpGroupLayout = Layout<Shape<_1, _8>>;
      using S2RThreadLayout = decltype(blocked_product(S2RWarpLayout{}, WarpGroupLayout{}));
      using S2RValLayout = Layout<Shape<Int<VectorSize>, _1>>;
      using S2RAtomA = Copy_Atom<AutoVectorizingCopy, TA>;
      using R2GAtomQA = Copy_Atom<AutoVectorizingCopy, TQA>;
      using R2GAtomSFA = Copy_Atom<AutoVectorizingCopy, TSFA>;
      auto tiled_s2r = make_tiled_copy(S2RAtomA{}, S2RThreadLayout{}, S2RValLayout{});
      auto tiled_r2g_QA = make_tiled_copy(R2GAtomQA{}, S2RThreadLayout{}, S2RValLayout{});
      auto tiled_r2g_SFA = make_tiled_copy(R2GAtomSFA{}, S2RThreadLayout{}, S2RValLayout{});

      auto thr_s2r = tiled_s2r.get_slice(local_thread_idx);
      auto thr_r2g_QA = tiled_r2g_QA.get_slice(local_thread_idx);
      auto thr_r2g_SFA = tiled_r2g_SFA.get_slice(local_thread_idx);
      Tensor tQAsA = thr_s2r.partition_S(sA);  // (Copy, Copy_M, Copy_N, PIPE)

      // Allocate temporary register tensors for copying quantization => output
      Tensor tQArA = make_tensor_like<TA>(
          make_layout(tQAsA(_, _, _, _0{}).shape()));  // (Copy, Copy_M, Copy_N)
      Tensor tQAgQA = thr_r2g_QA.partition_S(gQA_mn);
      Tensor tQArQA = make_tensor_like(tQAgQA(_, _, _, _0{}, _0{}));

      Tensor tQAgSFA = thr_r2g_SFA.partition_S(gSFA_mn);
      Tensor tQArSFA = make_tensor_like(tQAgSFA(_, _, _, _0{}, _0{}));

      // Will result in barrier_id=10 passed to bar.sync instr as cutlass adds 8
      // in order to go over the reserved named barrier count.
      constexpr int row_quant_barrier_id = 2;
      cutlass::arch::NamedBarrier::sync(NumEpilogueRowQuantThreadCount, row_quant_barrier_id);

      int group_idx = GetGroupIdx(&args, scheduler.tile_n_base() * size<1>(epilogue_tiler));
      float a_global_amax_val = shared_storage.global_a_amax[group_idx];
      // Aligning with TensorEngine's recipe to generate scale factors // {$nv-internal-release}
      static constexpr float fp4_max = 6.0f;
      static constexpr float fp8_max = 448.0f;
      static constexpr float fp4_max_inv = 1.0f / fp4_max;
      float global_encode_scale = a_global_amax_val > 0.0f
                                      ? cutlass::minimum_with_nan_propagation<float>{}(
                                            (fp8_max * fp4_max) / a_global_amax_val,
                                            cutlass::platform::numeric_limits<float>::max())
                                      : 1.0f;

      float global_decode_scale = 1.0f / global_encode_scale;
      float global_encode_scale_multiplier = 1.0f;
      if constexpr (kUseFastMath) {
        global_encode_scale_multiplier = global_encode_scale * fp4_max_inv;
      }
      auto sfa_converter = cutlass::NumericConverter<TSFA, ElementAccumulator>{};
      do {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile = 0;
             k_tile < K_TILE_MAX && k_tile + scheduler.tile_n_base() < scheduler.tiles_n();) {
          int global_tile_n_offset = (scheduler.tile_n_base() + k_tile) * size<1>(epilogue_tiler);

          int cur_group_idx = GetGroupIdx(&args, global_tile_n_offset);
          if (cur_group_idx != group_idx) {
            group_idx = cur_group_idx;
            a_global_amax_val = shared_storage.global_a_amax[group_idx];
            // Update group quantization parameters/scaling
            global_encode_scale = a_global_amax_val > 0.0f
                                      ? cutlass::minimum_with_nan_propagation<float>{}(
                                            (fp8_max * fp4_max) / a_global_amax_val,
                                            cutlass::platform::numeric_limits<float>::max())
                                      : 1.0f;
            global_decode_scale = 1.0f / global_encode_scale;
            if constexpr (kUseFastMath) {
              global_encode_scale_multiplier = global_encode_scale * fp4_max_inv;
            }
          }

          auto tQAgSFA_mn = tQAgSFA(_, _, _, scheduler.tile_m(), scheduler.tile_n_base() + k_tile);
          auto tQAgQA_mn = tQAgQA(_, _, _, scheduler.tile_m(), scheduler.tile_n_base() + k_tile);
          auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state);
          mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);
          copy(tiled_s2r, tQAsA(_, _, _, mainloop_pipe_consumer_state.index()), tQArA);
          cutlass::arch::fence_view_async_shared();
          mainloop_pipeline.consumer_release(mainloop_pipe_consumer_state);
          ++mainloop_pipe_consumer_state;
          ++k_tile;

          // static int constexpr NumVecs = size(tQArA) / VectorSize;
          cutlass::maximum_absolute_value_reduction<cutlass::Array<ElementAccumulator, VectorSize>,
                                                    true>
              amax_reduction;
          auto compute_frgs = reinterpret_cast<cutlass::Array<TA, VectorSize> *>(tQArA.data());
          auto output_frgs =
              reinterpret_cast<cutlass::Array<TQA, VectorSize> *>(raw_pointer_cast(tQArQA.data()));
          Tensor amax =
              make_tensor<ElementAccumulator>(prepend(take<1, rank(tQArA)>(tQArA.shape()), _1{}));
          Tensor pvscales = make_tensor_like<ElementAccumulator>(amax);
          transformer_engine::curanddx::detail::philox4x32_native_state<10> rng;
          if constexpr (kEnableStochasticRounding) {
            const size_t rng_sequence = global_thread_idx + k_tile * 512 +
                                        scheduler.get_linear_tile_idx() * K_TILE_MAX * 512 +
                                        tiles_in_m * tiles_in_n * K_TILE_MAX * 512;
            rng.init(rng_seed, rng_sequence, rng_offset);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < size<1>(group_modes<1, rank(tQArA)>(tQArA)); v++) {
            auto amax_view = group_modes<1, rank(amax)>(amax);
            auto pvscales_view = group_modes<1, rank(pvscales)>(pvscales);
            auto compute_frgs_up =
                cutlass::NumericArrayConverter<ElementAccumulator, TA, VectorSize>{}(
                    compute_frgs[v]);
            amax_view(_0{}, v) = amax_reduction(ElementAccumulator(0), compute_frgs_up);
            if constexpr (kUseFastMath) {
              // Fast math: multiply with precomputed reciprocal
              pvscales_view(_0{}, v) = cutlass::multiplies<ElementAccumulator>{}(
                  amax_view(_0{}, v), global_encode_scale_multiplier);
            } else {
              // Accurate math: perform division
              pvscales_view(_0{}, v) =
                  cutlass::divides<ElementAccumulator>{}(amax_view(_0{}, v), fp4_max);
              pvscales_view(_0{}, v) = cutlass::multiplies<ElementAccumulator>{}(
                  pvscales_view(_0{}, v), global_encode_scale);
            }
            filter(tQArSFA)(v) = sfa_converter(pvscales_view(_0{}, v));
            auto qpvscale_ups =
                cutlass::NumericConverter<ElementAccumulator, TSFA>{}(filter(tQArSFA)(v));
            auto qpvscale_scaled =
                cutlass::multiplies<ElementAccumulator>{}(qpvscale_ups, global_decode_scale);
            ElementAccumulator acc_scales;
            if constexpr (kUseFastMath) {
              // Fast math: compute approximate reciprocal
              acc_scales =
                  cutlass::reciprocal_approximate_ftz<decltype(qpvscale_scaled)>{}(qpvscale_scaled);
            } else {
              // Accurate math: compute reciprocal with division
              acc_scales = cutlass::divides<ElementAccumulator>{}(1.0, qpvscale_scaled);
            }
            auto acc_scale = cutlass::minimum_with_nan_propagation<ElementAccumulator>{}(
                acc_scales, cutlass::platform::numeric_limits<ElementAccumulator>::max());
            uint4 random_uint4 = uint4{0, 0, 0, 0};
            if constexpr (kEnableStochasticRounding) {
              random_uint4 = rng.generate4();
              output_frgs[v] = StochasticNumericConverter(
                  cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(
                      compute_frgs_up, acc_scale),
                  *reinterpret_cast<cutlass::Array<uint32_t, 4> *>(&random_uint4));
            } else {
              output_frgs[v] =
                  cutlass::NumericArrayConverter<TQA, ElementAccumulator, VectorSize>{}(
                      cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(
                          compute_frgs_up, acc_scale));
            }
          }
          copy(tiled_r2g_QA, tQArQA, tQAgQA_mn);
          copy(tiled_r2g_SFA, filter(tQArSFA), filter(tQAgSFA_mn));
        }
        // scheduler.advance();
        scheduler.fetch_next_work(sched_pipeline, sched_pipeline_consumer_state);
        ++sched_pipeline_consumer_state;
        scheduler.update_work_tile_info();
      } while (scheduler.is_valid());
    }

  } else {
    cutlass::arch::warpgroup_reg_dealloc<32>();
  }
}  // NOLINT(readability/fn_size)

template <bool kEnableStochasticRounding, bool kEnableRHTColQuant, bool kEnableRowQuant,
          bool kEnableSwizzleSFOutput, class TA, class TB, class TQA, class TSFA, class TD = TQA,
          class TSFD = TSFA, bool kUseFastMath = false>
void group_row_col_rht_gemm_ntt_w_sfc(int packed_sequence_length, int hidden_size, TA const *A,
                                      TB const *B, TQA *QA, TSFA *SFA,
                                      MultiAmaxHadamardCastFusionArgs &args,
                                      const size_t *rng_state, uint32_t *tile_scheduler_workspace,
                                      uint32_t sm_count, cudaStream_t stream,
                                      int k_tile_size = 1024) {
  using namespace cute;
  static int constexpr SFVecSize = 16;
  static int constexpr RhtTensorSize = 16;

  static_assert(RhtTensorSize == 16, "RhtTensorSize must be 16");
  using LinearSFALayout = decltype(make_layout(make_shape(make_shape(Int<SFVecSize>{}, 0), 0),
                                               make_stride(make_stride(_0{}, _1{}), 0)));
  using LinearSFDLayout = decltype(make_layout(make_shape(0, make_shape(Int<SFVecSize>{}, 0)),
                                               make_stride(0, make_stride(_0{}, _1{}))));

  using SwizzledSFALayoutAtom =
      cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVecSize, UMMA::Major::MN>::SfAtom;
  using SwizzledSFDLayoutAtom =
      cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVecSize, UMMA::Major::K>::SfAtom;
  using SwizzledSFALayout = decltype(tile_to_shape(
      SwizzledSFALayoutAtom{}, make_shape(hidden_size, packed_sequence_length), Step<_1, _2>{}));
  using SwizzledSFDLayout = decltype(tile_to_shape(
      SwizzledSFDLayoutAtom{}, make_shape(hidden_size, packed_sequence_length), Step<_2, _1>{}));

  using SFALayout = cute::conditional_t<kEnableSwizzleSFOutput, SwizzledSFALayout, LinearSFALayout>;
  using SFDLayout = cute::conditional_t<kEnableSwizzleSFOutput, SwizzledSFDLayout, LinearSFDLayout>;
  SFALayout sfa_layout;
  SFDLayout sfd_layout;

  if constexpr (kEnableSwizzleSFOutput) {
    sfa_layout = tile_to_shape(SwizzledSFALayoutAtom{},
                               make_shape(hidden_size, packed_sequence_length), Step<_1, _2>{});
    sfd_layout = tile_to_shape(SwizzledSFDLayoutAtom{},
                               make_shape(hidden_size, packed_sequence_length), Step<_2, _1>{});
  } else {
    sfa_layout = make_layout(
        make_shape(make_shape(Int<SFVecSize>{}, hidden_size / SFVecSize), packed_sequence_length),
        make_stride(make_stride(_0{}, _1{}), hidden_size / SFVecSize));
    sfd_layout = make_layout(
        make_shape(hidden_size, make_shape(Int<SFVecSize>{}, packed_sequence_length / SFVecSize)),
        make_stride(packed_sequence_length / SFVecSize, make_stride(_0{}, _1{})));
  }

  // Define shapes (dynamic)
  auto M = hidden_size;
  auto N = packed_sequence_length;
  Tensor tensorA = make_tensor(A, make_shape(hidden_size, packed_sequence_length), LayoutLeft{});
  Tensor tensorB = make_tensor(B, make_shape(RhtTensorSize, RhtTensorSize), LayoutLeft{});
  Tensor tensorQA = make_tensor(QA, make_shape(hidden_size, packed_sequence_length), LayoutLeft{});
  Tensor tensorSFA = make_tensor(SFA, sfa_layout);

  // Define strides (from tensors)
  auto dA = stride(tensorA);    // (dM,dK)
  auto dB = stride(tensorB);    // (dN,dK)
  auto dD = LayoutRight{};      // (dM,dN)
  auto dQA = stride(tensorQA);  // (dM,dK)
  using ClusterShape = Shape<_1, _1, _1>;
  auto cluster_shape = ClusterShape{};
  auto cluster_tile_shape = Shape<_128, Int<RhtTensorSize>, Int<RhtTensorSize>>{};
  auto cluster_tile_mainloop = Shape<_128, Int<RhtTensorSize>, _128>{};

  // Each mainloop / epilogue loads 128 x 64 tiles while each MMA proceeds with 128 x 16 tiles
  static int constexpr EpilogueUnrollFactor =
      size<2>(cluster_tile_mainloop) / size<2>(cluster_tile_shape);
  // Construct the MMA
  auto mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TA, TB, float, size<0>(cluster_tile_shape), size<1>(cluster_tile_shape),
                           UMMA::Major::MN, UMMA::Major::MN>{},
      Layout<Shape<_1, _1>>{});

  // Assert that the TiledMMA uses all CTAs in the CGA.
  CUTE_STATIC_ASSERT_V(size(cluster_shape) == size(mma));
  CUTE_STATIC_ASSERT_V(evenly_divides(cluster_tile_shape, tile_shape(mma)));

  // Determine the A and B shapes
  auto mma_shape_B =
      partition_shape_B(mma, make_shape(size<1>(cluster_tile_shape), size<2>(cluster_tile_shape)));

  using TiledMma = decltype(mma);
  using AtomThrID = typename TiledMma::AtomThrID;

  using SmemShape_M = decltype(shape_div(
      shape<0>(cluster_tile_shape),
      shape_div(shape<0>(cluster_tile_shape), size<0>(cluster_tile_shape) / size(AtomThrID{}))));
  using SmemShape_N = decltype(shape_div(
      shape<1>(cluster_tile_shape),
      shape_div(shape<1>(cluster_tile_shape), size<1>(cluster_tile_shape) / size(AtomThrID{}))));
  using SmemShape_K = decltype(cute::get<2>(cluster_tile_shape));

  using SmemLayoutAtomB =
      decltype(cutlass::gemm::collective::detail::sm100_smem_selector<cute::UMMA::Major::MN, TB,
                                                                      SmemShape_N, SmemShape_K>());

  auto mma_shape_A = partition_shape_A(
      mma, make_shape(size<0>(cluster_tile_mainloop), size<2>(cluster_tile_mainloop)));
  using SmemShape_M_A =
      decltype(shape_div(shape<0>(cluster_tile_mainloop),
                         shape_div(shape<0>(cluster_tile_mainloop),
                                   size<0>(cluster_tile_mainloop) / size(AtomThrID{}))));
  using SmemShape_K_A = decltype(cute::get<2>(cluster_tile_mainloop));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                                   cute::UMMA::Major::MN, TA, SmemShape_M_A, SmemShape_K_A>());

  static uint32_t constexpr TotalTmemRows = 128;
  static uint32_t constexpr Sm100TmemCapacityColumns = 512;
  static uint32_t constexpr TotalTmem = TotalTmemRows * Sm100TmemCapacityColumns;
  static uint32_t constexpr AccumulatorPipelineStageCount =
      TotalTmem / (cute::size<0>(cluster_tile_shape) * cute::size<1>(cluster_tile_shape));

  // Define the smem layouts (static)
  // Calculate max pipeline stages based on Blackwell SM100's 232KB shared memory
  constexpr int SchedulerPipelineStageCount = 4;
  static int constexpr MainloopPipelineBytes = sizeof(
      typename cutlass::detail::CustomizedPipelineTmaUmmaAsync<1, Shape<_1, _1, _1>,
                                                               Shape<_1, _1, _1>>::SharedStorage);

  static int constexpr SchedulerWorkspaceBytes = sizeof(int) * SchedulerPipelineStageCount;
  static int constexpr SchedulerThrottlePipelineBytes =
      sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
  static int constexpr SchedulerPipelineBytes =
      sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount,
                                                     ClusterShape>::SharedStorage);

  static int constexpr TmemDeallocBytes = sizeof(cutlass::arch::ClusterBarrier);
  static int constexpr BTensorBytes = cute::size(mma_shape_B) * sizeof(TB);
  static int constexpr AccPipelineBytes = sizeof(
      typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / EpilogueUnrollFactor,
                                          Shape<_1, _1, _1>>::SharedStorage);
  static int constexpr TmemBasePtrsBytes = sizeof(uint32_t);
  static int constexpr kBlackwellSmemSize = 232448;  // 232KB in bytes
  static int constexpr kBytesPerStage =
      cute::size(mma_shape_A) * sizeof(TA) + MainloopPipelineBytes;
  static int constexpr kReservedBytes = SchedulerWorkspaceBytes + SchedulerThrottlePipelineBytes +
                                        SchedulerPipelineBytes + TmemBasePtrsBytes +
                                        TmemDeallocBytes + BTensorBytes +
                                        AccPipelineBytes;  // Reserve for barriers and other uses
  static int constexpr kMaxStages = (kBlackwellSmemSize - kReservedBytes) / kBytesPerStage;
  auto sP = Int<kMaxStages>{};  // SMEM pipelines

  auto sA = UMMA::tile_to_mma_shape(SmemLayoutAtomA{}, append(mma_shape_A, sP),
                                    Step<_2, _1, _3>{});  // (MMA,MMA_M,MMA_K,PIPE)
  auto sB = UMMA::tile_to_mma_shape(SmemLayoutAtomB{},
                                    append(mma_shape_B, _1{}));  // (MMA,MMA_N,MMA_K, _1)
  auto sD = Layout<_1>{};                                        // XXX Dummy

  auto tma_load_a =
      make_tma_copy_A_sm100(SM90_TMA_LOAD{}, tensorA, sA(_, _, _, 0), cluster_tile_mainloop, mma);
  auto tma_load_b =
      make_tma_copy_B_sm100(SM90_TMA_LOAD{}, tensorB, sB(_, _, _, 0), cluster_tile_shape, mma);

  // Assert checks on tile sizes -- no predication
  assert(M % size<0>(cluster_tile_shape) == 0);
  assert(N % size<1>(cluster_tile_shape) == 0);

  dim3 dimBlock(512);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
  dim3 dimGrid(sm_count, 1, 1);

  int smem_size = sizeof(
      SharedStorage<TA, TB, decltype(sA), decltype(sB), ClusterShape, AccumulatorPipelineStageCount,
                    EpilogueUnrollFactor, SchedulerPipelineStageCount>);

  auto *kernel_ptr = &group_row_col_rht_gemm_device<
      decltype(M), decltype(N), decltype(k_tile_size), decltype(cluster_shape),
      decltype(cluster_tile_shape), TA, decltype(dA), decltype(sA), decltype(tma_load_a), TB,
      decltype(dB), decltype(sB), decltype(tma_load_b), TD, decltype(dD), decltype(sD), TSFD,
      decltype(sfd_layout), TQA, decltype(dQA), TSFA, decltype(sfa_layout), decltype(mma),
      AccumulatorPipelineStageCount, SchedulerPipelineStageCount, kEnableStochasticRounding,
      kEnableRHTColQuant, kEnableRowQuant, kEnableSwizzleSFOutput, kUseFastMath>;

  NVTE_CHECK_CUDA(
      cudaFuncSetAttribute(*kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  // Set workspace and set to zero
  NVTE_CHECK_CUDA(cudaMemsetAsync(reinterpret_cast<void *>(tile_scheduler_workspace), 0,
                                  sizeof(uint32_t), stream));

  // Launch kernel
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const *)kernel_ptr, M, N, k_tile_size, cluster_shape, cluster_tile_shape, A, dA,
      sA, tma_load_a, B, dB, sB, tma_load_b, QA, dQA, SFA, sfa_layout, args,
      tile_scheduler_workspace, mma, rng_state);
  NVTE_CHECK_CUDA(cudaGetLastError());
  NVTE_CHECK(status == cutlass::Status::kSuccess, "Kernel launch failed.");
}

}  // namespace
}  // namespace detail

void group_hadamard_transform_cast_fusion(const Tensor &input_, std::vector<Tensor *> &output_list,
                                          const size_t *split_sections, size_t num_tensors,
                                          const Tensor &hadamard_matrix_,
                                          QuantizationConfig &quant_config, Tensor &quant_workspace,
                                          cudaStream_t stream) {
  NVTE_API_CALL(group_hadamard_transform_cast_fusion);

  using transformer_engine::detail::kMaxTensorsPerKernel;
  using transformer_engine::detail::MultiAmaxHadamardCastFusionArgs;

  NVTE_CHECK(input_.dtype() == transformer_engine::DType::kBFloat16,
             "Input tensor must be BF16 tensor, but dtype is ", to_string(input_.dtype()), ".");
  NVTE_CHECK(input_.dim() >= 2, "Input must be a 2D tensor.");
  const SimpleTensor &input = input_.data;

  NVTE_CHECK(output_list.size() == num_tensors,
             "Number of output tensors should match number of tensors.");

  NVTE_CHECK(num_tensors <= kMaxTensorsPerKernel,
             "Number of tensors should be less than or equal to ", kMaxTensorsPerKernel);

  // construct the multi-tensor args
  MultiAmaxHadamardCastFusionArgs kernel_args;
  kernel_args.num_tensors = 0;
  kernel_args.split_sections_range[0] = 0;
  bool all_has_row_quant = true;
  bool all_has_col_quant = true;
  void *rowwise_data_base_ptr = nullptr;
  void *rowwise_scale_inv_base_ptr = nullptr;
  for (size_t i = 0; i < num_tensors; ++i) {
    NVTE_CHECK(split_sections[i] % 128 == 0, "component ", i,
               " of split_sections should be 128 multiple");
    if (split_sections[i] == 0) {
      continue;
    }
    bool has_row_quant = output_list[i]->data.dptr != nullptr;
    bool has_col_quant = output_list[i]->columnwise_data.dptr != nullptr;
    all_has_row_quant = all_has_row_quant && has_row_quant;
    all_has_col_quant = all_has_col_quant && has_col_quant;
    // sanity check, the two bool flags cannot be both false
    NVTE_CHECK(has_row_quant || has_col_quant,
               "At least one of the output tensors must have row or column quant.");
    void *amax_rowwise_ptr =
        has_row_quant ? reinterpret_cast<void *>(output_list[i]->amax.dptr) : nullptr;
    void *amax_colwise_ptr =
        has_col_quant ? reinterpret_cast<void *>(output_list[i]->columnwise_amax.dptr) : nullptr;
    void *rowwise_data_ptr =
        has_row_quant ? reinterpret_cast<void *>(output_list[i]->data.dptr) : nullptr;
    void *rowwise_scale_inv_ptr =
        has_row_quant ? reinterpret_cast<void *>(output_list[i]->scale_inv.dptr) : nullptr;
    if (all_has_row_quant &&
        (rowwise_data_base_ptr == nullptr || rowwise_scale_inv_base_ptr == nullptr)) {
      rowwise_data_base_ptr = rowwise_data_ptr;
      rowwise_scale_inv_base_ptr = rowwise_scale_inv_ptr;
    }
    void *output_colwise_ptr =
        has_col_quant ? reinterpret_cast<void *>(output_list[i]->columnwise_data.dptr) : nullptr;
    void *output_colwise_scale_inv_ptr =
        has_col_quant ? reinterpret_cast<void *>(output_list[i]->columnwise_scale_inv.dptr)
                      : nullptr;
    kernel_args.global_a_amax_list[kernel_args.num_tensors] = amax_rowwise_ptr;
    kernel_args.global_d_amax_list[kernel_args.num_tensors] = amax_colwise_ptr;
    kernel_args.output_colwise_list[kernel_args.num_tensors] = output_colwise_ptr;
    kernel_args.output_colwise_scale_inv_list[kernel_args.num_tensors] =
        output_colwise_scale_inv_ptr;
    kernel_args.split_sections[kernel_args.num_tensors] = split_sections[i];
    kernel_args.split_sections_range[kernel_args.num_tensors + 1] =
        kernel_args.split_sections_range[kernel_args.num_tensors] + split_sections[i];
    kernel_args.num_tensors++;
  }

  // Stochastic rounding config
  const bool use_stochastic_rounding = quant_config.stochastic_rounding;
  const size_t *rng_state = nullptr;
  if (use_stochastic_rounding) {
    NVTE_CHECK(quant_config.rng_state != nullptr,
               "Enabled stochastic rounding without providing RNG state");
    const Tensor &rng_state_tensor = *convertNVTETensorCheck(quant_config.rng_state);
    NVTE_CHECK(rng_state_tensor.dtype() == DType::kInt64,
               "RNG state should contain 2 64-bit values.");
    NVTE_CHECK(rng_state_tensor.data.shape == std::vector<size_t>{2},
               "Shape of the RNG state should be [2], but got ", rng_state_tensor.data.shape);
    rng_state = reinterpret_cast<const size_t *>(rng_state_tensor.data.dptr);
  }

  uint32_t *tile_scheduler_workspace = nullptr;
  NVTE_CHECK(quant_workspace.data.dptr != nullptr, "Quantization workspace must be provided.");
  NVTE_CHECK(quant_workspace.data.buffer_size_bytes() >= sizeof(uint32_t),
             "Quantization workspace must be at least 4 bytes.");
  tile_scheduler_workspace = reinterpret_cast<uint32_t *>(quant_workspace.data.dptr);

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

  const bool use_swizzle_sf_output = false;

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_stochastic_rounding, kEnableStochasticRounding,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          all_has_col_quant, kEnableRhtColQuant,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              all_has_row_quant, kEnableRowQuant,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  use_swizzle_sf_output, kEnableSwizzleSFOutput,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      quant_config.use_fast_math, kUseFastMath,

                      if constexpr (kEnableRhtColQuant || kEnableRowQuant) {
                        detail::group_row_col_rht_gemm_ntt_w_sfc<
                            kEnableStochasticRounding, kEnableRhtColQuant, kEnableRowQuant,
                            kEnableSwizzleSFOutput, TA, TB, TQA, TSFA, TD, TSFD, kUseFastMath>(
                            /*packed_sequence_length=*/m, /*hidden_size=*/n,
                            /*A=*/reinterpret_cast<TA const *>(input.dptr),
                            /*B=*/reinterpret_cast<TB const *>(hadamard_matrix.dptr),
                            /*QA=*/reinterpret_cast<TQA *>(rowwise_data_base_ptr),
                            /*SFA=*/reinterpret_cast<TSFA *>(rowwise_scale_inv_base_ptr),
                            /*args=*/kernel_args,
                            /*rng_state=*/rng_state,
                            /*tile_scheduler_workspace=*/tile_scheduler_workspace,
                            /*sm_count=*/sm_count,
                            /*stream=*/stream, /*k_tile_size=*/k_tile_size);
                      } else {
                        NVTE_ERROR("Invalid kernel configuration (kEnableRHTColQuant=",
                                   kEnableRhtColQuant, ", kEnableRowQuant=", kEnableRowQuant, ").");
                      }

                  );););););
}

}  // namespace transformer_engine

void nvte_group_hadamard_transform_cast_fusion(const NVTETensor input, NVTETensor *outputs,
                                               const NVTETensor hadamard_matrix,
                                               const size_t *split_sections,
                                               const size_t num_tensors,
                                               const NVTEQuantizationConfig quant_config,
                                               NVTETensor quant_workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_hadamard_transform_cast_fusion);
  using namespace transformer_engine;
  NVTE_CHECK(num_tensors > 0, "Number of tensors should be greater than 0.");

  Tensor *input_tensor = convertNVTETensorCheck(input);
  std::vector<Tensor *> output_list(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    output_list[i] = convertNVTETensorCheck(outputs[i]);
  }

  Tensor *quant_workspace_tensor = convertNVTETensorCheck(quant_workspace);

  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }

  // Call the multi-tensor Hadamard transform amax implementation.
  group_hadamard_transform_cast_fusion(*input_tensor, output_list, split_sections, num_tensors,
                                       *convertNVTETensorCheck(hadamard_matrix), quant_config_cpp,
                                       *quant_workspace_tensor, stream);
}
