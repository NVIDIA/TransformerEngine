/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "common/util/ptx.cuh"
#include "common/utils.cuh"
#include "curanddx.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/numeric_conversion.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

// clang-format off

namespace transformer_engine {
namespace detail {
namespace {

// Define a cuRANDDx descriptor
// Note curanddx::PhiloxRounds<4> means 4 rounds of philox4_32. If the operator is not specified, it will be default to 10.
// curanddx::SM<800>() does NOT mean the code can only run on SM 800. The operator is used for do some internal checks, e.g.,
// if shared memory, if needed, is enough for the described problem, usually not applicable.

// curanddx doc: https://docs.nvidia.com/cuda/curanddx/index.html
using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() + curanddx::PhiloxRounds<10>() + curanddx::SM<800>() + curanddx::Thread());


using namespace cute;
using cute::Tensor;  // Ensure unqualified Tensor refers to cute::Tensor, not transformer_engine::Tensor

// calculate the global encode scale factor for a given global amax.
__device__ __forceinline__ float ComputeGlobalEncodeScaleFP4(const float global_amax) {
  constexpr float kFP8E4M3Max = 448.0f;
  constexpr float kFP4E2M1Max = 6.0f;
  // If scale is infinity, return max value of float32
  float global_encode_scale = cutlass::minimum_with_nan_propagation<float>{}(
    kFP8E4M3Max * kFP4E2M1Max / global_amax, cutlass::platform::numeric_limits<float>::max());
  // If global amax is 0 or infinity, return 1
  return (global_amax == 0.f || global_encode_scale == 0.f) ? 1.f : global_encode_scale;
}

template <class ElementA,
          class ElementB,
          class ASmemLayout,
          class BSmemLayout>
struct SharedStorage {
  static constexpr int AccumulatorPipelineStageCount = 16;
  using AtomThrShapeMNK = cute::Shape<_1, _1, _1>;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / 4, AtomThrShapeMNK>;
  using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;

  static constexpr int MainloopPipelineStageCount = size<3>(ASmemLayout{});
  using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
                             MainloopPipelineStageCount,
                             Shape<_1,_1,_1>,
                             AtomThrShapeMNK>;
  using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;

  alignas(16) AccumulatorPipelineStorage accumulator;
  alignas(16) MainloopPipelineStorage mainloop;
  alignas(16) cute::uint64_t tma_barrier[1];
  uint32_t tmem_base_ptr;

  struct TensorStorage : cute::aligned_struct<128, _1> {
    // cute::array_aligned<ElementA, cute::cosize_v<ASmemLayout>> smem_A;
    cute::array_aligned<ElementA, cute::cosize_v<ASmemLayout>> smem_A;
    cute::array_aligned<ElementB, cute::cosize_v<BSmemLayout>> smem_B;
  } tensors;

};

CUTLASS_DEVICE
cutlass::Array<cutlass::float_e2m1_t, 8>
StochasticNumericConverterBase(cutlass::Array<float, 8> const &input, cutlass::Array<uint32_t, 2> const &rbits) {
  using result_type = cutlass::Array<cutlass::float_e2m1_t, 8>;
  result_type output;
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  auto output_ptr = reinterpret_cast<uint16_t *>(&output);
  asm volatile( \
      "{\n" \
      "cvt.rs.satfinite.e2m1x4.f32   %0, {%5, %4, %3, %2}, %10;\n" \
      "cvt.rs.satfinite.e2m1x4.f32   %1, {%9, %8, %7, %6}, %11;\n" \
      "}" \
      : "=h"(output_ptr[0]),
        "=h"(output_ptr[1])
      : "f"(input[0]), "f"(input[1]), "f"(input[2]), "f"(input[3]),
        "f"(input[4]), "f"(input[5]), "f"(input[6]), "f"(input[7]),
        "r"(rbits[0]), "r"(rbits[1]));
#else
  NVTE_DEVICE_ERROR("FP4 cvt PTX instructions are architecture-specific. "
                    "Try recompiling with sm_XXXa instead of sm_XXX.");
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  return output;
}

CUTLASS_DEVICE
cutlass::Array<cutlass::float_e2m1_t, 16>
StochasticNumericConverter(cutlass::Array<float, 16> const &input, cutlass::Array<uint32_t, 4> const *rbits) {
  using result_type = cutlass::Array<cutlass::float_e2m1_t, 16>;
  result_type output;
  cutlass::Array<cutlass::float_e2m1_t, 8> *result_ptr = reinterpret_cast<cutlass::Array<cutlass::float_e2m1_t, 8> *>(&output);
  cutlass::Array<float, 8> const *source_ptr = reinterpret_cast<cutlass::Array<float, 8> const *>(&input);
  cutlass::Array<uint32_t, 2> const *rbits_ptr = reinterpret_cast<cutlass::Array<uint32_t, 2> const *>(rbits);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; i++) {
    result_ptr[i] = StochasticNumericConverterBase(source_ptr[i], rbits_ptr[i]);
  }
  return output;
}

template <class MShape, class NShape, class KShape, class ClusterTileShape,
          class TA, class AStride, class ASmemLayout, class TmaLoadA,
          class TB, class BStride, class BSmemLayout, class TmaLoadB,
          class TC, class CStride, class CSmemLayout,
          class TSFC,
          class TiledMMA,
          bool kEnableStochasticRounding = false>
__global__ static
void
rht_gemm_device(MShape M, NShape N, KShape K, ClusterTileShape cluster_tile,
            TA const* A, AStride dA, ASmemLayout sAlayout, CUTE_GRID_CONSTANT TmaLoadA const tma_load_a,
            TB const* B, BStride dB, BSmemLayout sBlayout, CUTE_GRID_CONSTANT TmaLoadB const tma_load_b,
            TC      * C, CStride dC, CSmemLayout         ,
            TSFC    * SFC,
            TiledMMA mma,
            float const* global_amax,
            const size_t* rng_state)
{
  using namespace cute;
  using X = Underscore;
  // static constexpr bool kApplyStochasticRounding = true;
  using ElementAccumulator = float;
  static constexpr int K_PIPE_MAX = size<3>(ASmemLayout{});
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMMA::ThrLayoutVMNK{})), _1, _1>;
  static constexpr uint32_t kTmaTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(ASmemLayout{})) * cute::sizeof_bits_v<TA>);

  static constexpr int kTmaRhtTensorTransactionBytes =
    cutlass::bits_to_bytes(16 * 16 * cute::sizeof_bits_v<TB>);
  static constexpr int AccumulatorPipelineStageCount = 16;

  static constexpr int MainloopPipelineStageCount = size<3>(ASmemLayout{});
  using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
                             MainloopPipelineStageCount,
                             Shape<_1,_1,_1>,
                             AtomThrShapeMNK>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  static constexpr int VectorSize = 16;
  const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  // Preconditions
  CUTE_STATIC_ASSERT(is_static<ASmemLayout>::value);
  CUTE_STATIC_ASSERT(is_static<BSmemLayout>::value);
  CUTE_STATIC_ASSERT(is_static<CSmemLayout>::value);

  // Represent the full tensors
  Tensor mA = tma_load_a.get_tma_tensor(make_shape(M,N));
  Tensor mB = tma_load_b.get_tma_tensor(make_shape(16,16));
  Tensor mC = make_tensor(cute::subbyte_iterator<TC>(C), make_shape(M,N), dC);      // (M,N)

  auto sfc_shape  = make_shape(
    M,
    make_shape( make_shape(Int<16>{}, _4{}), N / 64 )
  );

  auto sfc_stride = make_stride(
    N / 16,
    make_stride( make_stride(_0{}, _1{}), _4{} )
  );

  auto sfc_layout = make_layout(sfc_shape, sfc_stride);
  Tensor mSFC = make_tensor(make_gmem_ptr(SFC), sfc_layout);

  auto cluster_shape = Shape<  _1,  _1, _1>{};

  // Get the appropriate blocks for this Cluster
  dim3 cluster_coord_in_grid = cluster_id_in_grid();

  // Total number of k-tiles
  const int K_TILE_MAX  = min(N, K) / 64;
  uint32_t tiles_in_m = (M + size<0>(cluster_tile) - 1) / size<0>(cluster_tile);
  uint32_t tiles_in_n = (N + 64 - 1) / 64;
  uint32_t linear_tile_idx = blockIdx.x;
  uint32_t tile_idx_m = linear_tile_idx % tiles_in_m;
  uint32_t tile_idx_n = (linear_tile_idx / tiles_in_m) * K_TILE_MAX;


  auto mainloop_tiler = Shape<_128,_16,_64>{};
  auto epilogue_tiler = Shape<_128,_64,_64>{};
  Tensor gA_mk = local_tile(mA, mainloop_tiler, make_coord(_,_, _), Step<_1, X,_1>{});
  Tensor gB_nk = local_tile(mB, cluster_tile, make_coord(_,_, _), Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC_mn = local_tile(mC, epilogue_tiler, make_coord(_,_, _), Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  Tensor gSFC_mn = local_tile(mSFC, epilogue_tiler, make_coord(_,_, _), Step<_1,_1, X>{});  // (BLK_M,BLK_N)
  // Allocate SMEM
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor tCsA = make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()), sAlayout);  // (MMA,MMA_M,MMA_N,PIPE)
  Tensor tCsB = make_tensor(make_smem_ptr(shared_storage.tensors.smem_B.data()), sBlayout);  // (MMA,MMA_N,MMA_K,PIPE)


  //
  // MMA: Define C accumulators and A/B partitioning
  //

  int block_rank_in_cluster = cute::block_rank_in_cluster();
  ThrMMA thr_mma = mma.get_slice(block_rank_in_cluster);               // blk idx
  Tensor tCgB = thr_mma.partition_B(gB_nk);                               // (MMA,MMA_N,MMA_K,k)

  auto mma_epilogue = make_tiled_mma(SM100_MMA_F16BF16_SS<TA, TB, ElementAccumulator,
                                               128, 64,
                                               UMMA::Major::MN, UMMA::Major::MN>{},
                            Layout<Shape<_1,_1>>{});
  ThrMMA thr_mma_epilogue = mma_epilogue.get_slice(block_rank_in_cluster);


  using TiledMmaEpilogue = decltype(mma_epilogue);
  Tensor tCgA = thr_mma.partition_A(gA_mk);
  // Allocate "fragments" -- these are actually umma smem descriptors
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_M,MMA_K,PIPE)

  auto acc_shape_mma = partition_shape_C(TiledMMA{}, take<0,2>(ClusterTileShape{}));
  auto acc_shape_epilogue = partition_shape_C(TiledMmaEpilogue{}, take<0,2>(epilogue_tiler));

  auto bulk_tmem_mma = TiledMMA::make_fragment_C(append(acc_shape_mma,
                                                      Int<AccumulatorPipelineStageCount>{}));

  auto bulk_tmem_epilogue = TiledMmaEpilogue::make_fragment_C(append(acc_shape_epilogue,
                                                      Int<AccumulatorPipelineStageCount / 4>{}));

  TmemAllocator tmem_allocator{};
  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

  Layout cta_layout_mnk  = make_layout(cluster_shape);
  Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMMA::AtomThrID{}));
  auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

  auto [tAgA, tAsA] = tma_partition(tma_load_a,
    get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
    group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));

  auto [tBgB, tBsB] = tma_partition(tma_load_b,
    get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
    group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));

  uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
  uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

  int warp_idx = cutlass::canonical_warp_idx_sync();

  bool is_mma_warp = (warp_idx == 0);
  bool is_dma_warp = (warp_idx == 1);
  bool is_epilogue_warp = (warp_idx >= 4 && warp_idx <= 7);

  if (is_epilogue_warp && elect_one_sync()) {
    cute::prefetch(raw_pointer_cast(global_amax));
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
  MainloopPipeline mainloop_pipeline(shared_storage.mainloop,
                                       mainloop_pipeline_params,
                                       cluster_shape,
                                       cute::true_type{},   // Perform barrier init
                                       cute::true_type{}); // Delay mask calculation

  MainloopPipelineState mainloop_pipe_consumer_state;
  MainloopPipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();



  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / 4, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  AccumulatorPipelineState accumulator_pipe_consumer_state;
  AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

  typename AccumulatorPipeline::Params accumulator_pipeline_params;
  if (is_mma_warp) {
    accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
  }
  if (is_epilogue_warp) {
    accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
  }
  // Only one producer thread arrives on this barrier.
  accumulator_pipeline_params.producer_arv_count = 1;
  accumulator_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * 128;
  accumulator_pipeline_params.initializing_warp = 1;
  AccumulatorPipeline accumulator_pipeline(shared_storage.accumulator,
                                           accumulator_pipeline_params,
                                           cluster_shape,
                                           cute::true_type{},   // Perform barrier init
                                           cute::true_type{}); // Delay mask calculation

  if (warp_idx == 2 && elect_one_sync()) {
    cute::initialize_barrier(shared_storage.tma_barrier[0], /* num_threads */ 1);
  }
  __syncthreads();
  using TMEM_LOAD_NEW = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b64x;

  if (is_dma_warp) {
    if (elect_one_sync()) {
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier[0], kTmaRhtTensorTransactionBytes);
      copy(tma_load_b.with(shared_storage.tma_barrier[0], tma_mcast_mask_b), tBgB(_,0,0), tBsB(_,0));
    }
    cute::wait_barrier(shared_storage.tma_barrier[0], 0 /*tma_phase_bit*/);
    do {
      bool is_first_wave = linear_tile_idx == blockIdx.x;
      uint32_t skip_wait = is_first_wave;
      auto tAgA_mk = tAgA(_,tile_idx_m,_);
      int k_tile = 0;
      auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state, skip_wait);


      CUTE_NO_UNROLL
      while (k_tile < K_TILE_MAX && k_tile + tile_idx_n < tiles_in_n) {
        int k_tile_idx_n = tile_idx_n + k_tile;
        ++k_tile;
        skip_wait = (is_first_wave && k_tile < MainloopPipelineStageCount);
        mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);
        int write_stage = mainloop_pipe_producer_state.index();
        ++mainloop_pipe_producer_state;
        barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state, skip_wait);
        if (cute::elect_one_sync()) {
          copy(tma_load_a.with(*tma_barrier, tma_mcast_mask_a), tAgA_mk(_,k_tile_idx_n), tAsA(_,write_stage));
        }
      }
      linear_tile_idx += gridDim.x;
      tile_idx_m = linear_tile_idx % tiles_in_m;
      tile_idx_n = (linear_tile_idx / tiles_in_m) * K_TILE_MAX;
    } while (tile_idx_m < tiles_in_m && tile_idx_n < tiles_in_n);
    mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
  } else if (is_mma_warp) {
    mma.accumulate_ = UMMA::ScaleOut::Zero;

    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
    __syncwarp();
    tmem_allocation_result_barrier.arrive();
    uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
    bulk_tmem_mma.data() = tmem_base_ptr;

    do {
      uint32_t skip_wait = K_TILE_MAX <= 0;
      auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);
      CUTE_NO_UNROLL
      for (int k_tile = 0; k_tile < K_TILE_MAX && k_tile + tile_idx_n < tiles_in_n; )
      {
        mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);
        int read_stage = mainloop_pipe_consumer_state.index();
        auto tCrA_mk = tCrA(_,_,_,read_stage);
        auto tCrB_nk = tCrB(_,_,0,0);
        CUTE_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA) / 4; ++k_block)
        {
          accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
          CUTE_UNROLL
          for (int i = 0; i < 4; i++) {
            auto accumulators = bulk_tmem_mma(_,_,_,accumulator_pipe_producer_state.index() * 4 + i);
            gemm(mma, tCrA_mk(_,_,k_block * 4 + i), tCrB_nk, accumulators);
          }

          accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
          ++accumulator_pipe_producer_state;
        }
        auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;
        ++mainloop_pipe_consumer_state;
        ++k_tile;
        skip_wait = k_tile >= K_TILE_MAX;
        barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);
        mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
      }

      linear_tile_idx += gridDim.x;
      tile_idx_m = linear_tile_idx % tiles_in_m;
      tile_idx_n = (linear_tile_idx / tiles_in_m) * K_TILE_MAX;
    } while (tile_idx_m < tiles_in_m && tile_idx_n < tiles_in_n);
    tmem_allocator.release_allocation_lock();
    accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  } else if (is_epilogue_warp) {
    const float global_amax_val = *global_amax;
    static constexpr int FragmentSize = 256 / sizeof_bits_v<TC>;

    tmem_allocation_result_barrier.arrive_and_wait();
    uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
    bulk_tmem_epilogue.data() = tmem_base_ptr;
    int thread_idx = threadIdx.x % 128;

    Tensor tCgC = thr_mma_epilogue.partition_C(gC_mn);                             // (MMA,MMA_M,MMA_N)                             // (MMA,MMA_M,MMA_N)
    auto tiled_t2r = make_tmem_copy(TMEM_LOAD_NEW{}, bulk_tmem_epilogue(_,_,_,_0{}));
    auto tiled_r2g = make_tiled_copy_D(Copy_Atom<SM100_STORE_256bit_CACHE_NOALLOCATION, TC>{}, tiled_t2r);
    auto thr_t2r   = tiled_t2r.get_slice(thread_idx);
    auto thr_r2g = tiled_r2g.get_slice(thread_idx);

    // NVFP4 non-E8 recipe constants and global scales
    static constexpr float fp4_max = 6.0f;

    const float global_encode_scale = ComputeGlobalEncodeScaleFP4(global_amax_val);
    const float global_decode_scale = 1.0f / global_encode_scale;
    auto sfd_converter = cutlass::NumericConverter<TSFC, float>{};

    do {
      for (int k_tile = 0; k_tile < K_TILE_MAX && k_tile + tile_idx_n < tiles_in_n; ++k_tile) {
        Tensor tCgC_mn = tCgC(_,_,_,tile_idx_m,tile_idx_n+k_tile);

        Tensor tCgSFC_mn = gSFC_mn(_,_,tile_idx_m,tile_idx_n+k_tile);
        accumulator_pipeline.consumer_wait(accumulator_pipe_consumer_state);

        auto tCtC = bulk_tmem_epilogue(_,_,_,accumulator_pipe_consumer_state.index());
        Tensor tDtC = thr_t2r.partition_S(tCtC);                   // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
        Tensor tDgC = thr_t2r.partition_D(tCgC_mn);                   // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)

        Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tDgC));                 // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
        Tensor tDrC = make_tensor<TC>(shape(tDgC));
        Tensor tTR_rAcc_frag = recast<cutlass::Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc));
        Tensor tDrC_frag = recast<cutlass::Array<TC, FragmentSize>>(coalesce(tDrC));

        Tensor src = thr_r2g.retile_S(tDrC);
        Tensor dst = thr_r2g.retile_D(tDgC);

        Tensor tCgSFC = make_tensor(tCgSFC_mn.data(), make_layout(
                                    make_shape(shape(tCgSFC_mn), Int<1>{}, Int<1>{}),
                                    make_stride(stride(tCgSFC_mn), Int<0>{}, Int<0>{})
                                   ));

        Tensor tDgSFC = filter(thr_t2r.partition_D(tCgSFC));
        Tensor tDrSFC = make_tensor<TSFC>(shape(tDgSFC));

        static constexpr int NumVecs = size(tDgC) / VectorSize;
        Tensor tC_rRowSFD_frg = recast<cutlass::Array<TSFC, NumVecs>>(tDrSFC);

        cutlass::maximum_absolute_value_reduction<cutlass::Array<ElementAccumulator, VectorSize>, true> amax_reduction;
        cutlass::Array<ElementAccumulator, NumVecs> vec_maxs;
        cutlass::Array<ElementAccumulator, NumVecs> pvscales;
        // TMEM_LOAD
        copy(tiled_t2r, tDtC, tTR_rAcc);
        cutlass::arch::fence_view_async_tmem_load();

        accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);

        ++accumulator_pipe_consumer_state;

        // Cast data from FP32 to BF16 to FP32.
        auto convert_accum_to_bf16 = cutlass::NumericArrayConverter<cutlass::bfloat16_t, ElementAccumulator, FragmentSize>{};
        auto convert_bf16_to_accum = cutlass::NumericArrayConverter<ElementAccumulator, cutlass::bfloat16_t, FragmentSize>{};
        tTR_rAcc_frag(_0{}) = convert_bf16_to_accum(convert_accum_to_bf16(tTR_rAcc_frag(_0{})));

        auto compute_frgs = reinterpret_cast<cutlass::Array< ElementAccumulator, VectorSize> *>(tTR_rAcc_frag.data());
        auto output_frgs = reinterpret_cast<cutlass::Array< TC, VectorSize> *>(tDrC_frag.data());
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < NumVecs; v++) {
          vec_maxs[v] = amax_reduction(ElementAccumulator(0), compute_frgs[v]);
        }

        pvscales = cutlass::divides<cutlass::Array<ElementAccumulator, NumVecs>>{}(vec_maxs, fp4_max);
        pvscales = cutlass::multiplies<cutlass::Array<ElementAccumulator, NumVecs>>{}(pvscales, global_encode_scale);
        auto pvscales_cvted = cutlass::NumericArrayConverter<TSFC, ElementAccumulator, NumVecs>{}(pvscales);

        tC_rRowSFD_frg(_0{}) = pvscales_cvted;
        auto qpvscale_ups = cutlass::NumericArrayConverter<ElementAccumulator, TSFC, NumVecs>{}(tC_rRowSFD_frg(_0{}));
        auto qpvscale_scaled = cutlass::multiplies<cutlass::Array<ElementAccumulator, NumVecs>>{}(qpvscale_ups, global_decode_scale);
        auto acc_scales = cutlass::divides<cutlass::Array<ElementAccumulator, NumVecs>>{}(1.0, qpvscale_scaled);

        // Initialize RNG for tile
        const size_t rng_sequence
          = thread_idx + k_tile * 256 + linear_tile_idx * K_TILE_MAX * 256;
        RNG rng(rng_seed, rng_sequence, rng_offset);
        curanddx::uniform_bits dist;
        uint4 random_uint4 = uint4{0, 0, 0, 0};

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < NumVecs; v++) {
          auto acc_scale = cutlass::minimum_with_nan_propagation<ElementAccumulator>{}(acc_scales[v], cutlass::platform::numeric_limits<ElementAccumulator>::max());
          // auto acc_scale = acc_scales[v];
          if constexpr (kEnableStochasticRounding) {
            random_uint4 = dist.generate4(rng);
            output_frgs[v] = StochasticNumericConverter(
              cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(
                compute_frgs[v],
                acc_scale
              ),
              reinterpret_cast<cutlass::Array<uint32_t, 4>*>(&random_uint4));
          } else {
            output_frgs[v] = cutlass::NumericArrayConverter<TC, ElementAccumulator, VectorSize>{}(cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(compute_frgs[v], acc_scale));
          }
        }

        copy(tiled_r2g, src, dst);

        copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tDrSFC, tDgSFC);

      }
      linear_tile_idx += gridDim.x;
      tile_idx_m = linear_tile_idx % tiles_in_m;
      tile_idx_n = (linear_tile_idx / tiles_in_m) * K_TILE_MAX;
    } while (tile_idx_m < tiles_in_m && tile_idx_n < tiles_in_n);
  }
}

// this function computes RHT-GEMM for
// A: m x n: col-major
// B: 16 x 16: row-major
// C: m x n: row-major
// SFC: m x (n/16): row-major
template <typename TA, typename TB, typename TC, typename TSFC, bool kEnableStochasticRounding = false>
void
rht_gemm_ntt_w_sfc(int m, int n,
        TA const* A,
        TB const* B,
        TC      * C,
        TSFC    * SFC,
        float const* global_amax,
        const size_t* rng_state,
        uint32_t sm_count,
        cudaStream_t stream,
        int k_tile_size = 2048)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = static_cast<int>(m);
  auto N = static_cast<int>(n);

  // Define strides (mixed)
  auto dA = make_stride(Int<1>{}, m);  // (dM,dK)
  auto dB = make_stride(Int<1>{}, 16);  // (dN,dK)
  auto dC = make_stride(n, Int<1>{});  // (dM,dN)

  auto cga_shape      = Shape<  _1,  _1, _1>{};
  auto cga_tile_shape = Shape<_128,_16,_16>{};
  auto cluster_tile_mainloop = Shape<_128,_16,_64>{};

  // Construct the MMA
  auto mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TA, TB, float,
                                               128, 16,
                                               UMMA::Major::MN, UMMA::Major::MN>{},
                            Layout<Shape<_1,_1>>{});

  // MMA in CGA Layout XXX: Need to generalize synchro? {$nv-release-never}

  // Assert that the TiledMMA uses all CTAs in the CGA.
  CUTE_STATIC_ASSERT_V(size(cga_shape) == size(mma));
  CUTE_STATIC_ASSERT_V(evenly_divides(cga_tile_shape, tile_shape(mma)));

  // Determine the A and B shapes
  auto mma_shape_B = partition_shape_B(mma, make_shape(size<1>(cga_tile_shape), size<2>(cga_tile_shape)));

  using TiledMma = decltype(mma);
  using AtomThrID = typename TiledMma::AtomThrID;

  using SmemShape_M = decltype(shape_div(shape<0>(cga_tile_shape), shape_div(shape<0>(cga_tile_shape), size<0>(cga_tile_shape) / size(AtomThrID{}))));
  using SmemShape_N = decltype(shape_div(shape<1>(cga_tile_shape), shape_div(shape<1>(cga_tile_shape), size<1>(cga_tile_shape) / size(AtomThrID{}))));
  using SmemShape_K = decltype(cute::get<2>(cga_tile_shape));

  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      cute::UMMA::Major::MN, TB, SmemShape_N, SmemShape_K>());

  auto mma_shape_A = partition_shape_A(mma, make_shape(size<0>(cluster_tile_mainloop), size<2>(cluster_tile_mainloop)));
  using SmemShape_M_A = decltype(shape_div(shape<0>(cluster_tile_mainloop), shape_div(shape<0>(cluster_tile_mainloop), size<0>(cluster_tile_mainloop) / size(AtomThrID{}))));
  using SmemShape_K_A = decltype(cute::get<2>(cluster_tile_mainloop));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      cute::UMMA::Major::MN, TA, SmemShape_M_A, SmemShape_K_A>());

  // Define the smem layouts (static)
  // Calculate max pipeline stages based on Blackwell SM100's 232KB shared memory
  constexpr int kBlackwellSmemSize = 232448; // 232KB in bytes
  constexpr int kBytesPerStage = cute::size(mma_shape_A) * sizeof(TA) + cute::size(mma_shape_B) * sizeof(TB);
  constexpr int kReservedBytes = 256; // Reserve for barriers and other uses
  constexpr int kMaxStages = (kBlackwellSmemSize - kReservedBytes) / kBytesPerStage;
  auto sP = Int<kMaxStages>{};      // SMEM pipelines
  auto sA = UMMA::tile_to_mma_shape(SmemLayoutAtomA{}, append(mma_shape_A, sP)); // (MMA,MMA_M,MMA_K,PIPE)
  auto sB = UMMA::tile_to_mma_shape(SmemLayoutAtomB{}, append(mma_shape_B, sP)); // (MMA,MMA_N,MMA_K,PIPE)
  auto sC = Layout<_1>{};  // XXX Dummy

  // Create GMEM tensors
  Tensor tensorA = make_tensor(A, make_layout(make_shape(M,N), dA));      // (M,N)
  Tensor tensorB = make_tensor(B, make_layout(make_shape(16,16), dB));      // (16,16)

  // Create the TiledCopy

  auto tma_load_a = make_tma_copy_A_sm100(
        SM90_TMA_LOAD{},
        tensorA,
        sA(_,_,_,0),
        cluster_tile_mainloop,
        mma);
  auto tma_load_b =  make_tma_copy_B_sm100(
        SM90_TMA_LOAD{},
        tensorB,
        sB(_,_,_,0),
        cga_tile_shape,
        mma);

  // Assert checks on tile sizes -- no predication
  NVTE_CHECK(M % size<0>(cga_tile_shape) == 0,
             "Inner dimension must be divisible by ", static_cast<size_t>(size<0>(cga_tile_shape)), " but got ", M, ".");
  NVTE_CHECK(N % (4 * size<1>(cga_tile_shape)) == 0,
             "Outer dimension must be divisible by ", 4 * static_cast<size_t>(size<1>(cga_tile_shape)),
             " but got ", N, ".");

  uint32_t tiles = size(ceil_div(M, get<0>(cga_tile_shape))) * size(ceil_div(N, k_tile_size));

  tiles = (tiles < sm_count) ? tiles : sm_count;

  dim3 dimBlock(256);
  dim3 dimCluster(size<0>(cga_shape), size<1>(cga_shape), size<2>(cga_shape));
  dim3 dimGrid(tiles, 1, 1);

  int smem_size = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);
  auto* kernel_ptr = &rht_gemm_device<
                                  decltype(M), decltype(N), decltype(k_tile_size), decltype(cga_tile_shape),
                                  TA, decltype(dA), decltype(sA), decltype(tma_load_a),
                                  TB, decltype(dB), decltype(sB), decltype(tma_load_b),
                                  TC, decltype(dC), decltype(sC),
                                  TSFC,
                                  decltype(mma),
                                  kEnableStochasticRounding>;

  bool status = cudaFuncSetAttribute(*kernel_ptr,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_size);

  if (status != cudaSuccess) {
    std::cerr << "Error: Failed to set Shared Memory size." << std::endl;
    return;
  }
  (*kernel_ptr)
      <<< dimGrid, dimBlock, smem_size, stream >>>
      (M,  N,  k_tile_size, cga_tile_shape,
       A, dA, sA, tma_load_a,
       B, dB, sB, tma_load_b,
       C, dC, sC,
       SFC,
       mma, global_amax,
       rng_state);
}

// this function is used to wrap the rht_gemm_ntt_w_sfc function
//to transpose the input tensor A
template <typename TA, typename TB, typename TC, typename TSFC, bool kEnableStochasticRounding = false>
void
rht_gemm_ttt_wrapper(int m, int n,
        TA const* A,
        TB const* B,
        TC      * C,
        TSFC    * SFC,
        float const* global_amax,
        const size_t* rng_state,
        uint32_t sm_count,
        cudaStream_t stream,
        int k_tile_size = 1024)
{
  // in addition to transpose the input tensor A
  // we also need to reshape m, n to at best
  // ultilize as many SMs as possible while keeping
  // a relatively large contiguous dimension.
  // for example, after swapping m, n for transpose purposes,
  // the input / output tensor shapes for RHT-GEMM are:
  // A: n x m: col-major
  // B: 16 x 16: row-major
  // C: n x m: row-major
  // SFC: n x (m/16): row-major
  rht_gemm_ntt_w_sfc<TA, TB, TC, TSFC, kEnableStochasticRounding>(
    n, m,
    A, B, C,
    SFC, global_amax,
    rng_state,
    sm_count, stream,
    k_tile_size);
}

}  // namespace
}  // namespace detail

// clang-format on

void hadamard_transform_cast_fusion_columnwise(const Tensor &input_, Tensor &output_,
                                               const Tensor &hadamard_matrix_,
                                               QuantizationConfig quant_config,
                                               cudaStream_t stream) {
  NVTE_API_CALL(hadamard_transform_cast_fusion_columnwise);

  // Check input and output tensors
  NVTE_CHECK(input_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be BF16 tensor, but scaling mode is ",
             to_string(input_.scaling_mode), ".");
  NVTE_CHECK(input_.dtype() == transformer_engine::DType::kBFloat16,
             "Input tensor must be BF16 tensor, but dtype is ", to_string(input_.dtype()), ".");
  NVTE_CHECK(input_.dim() >= 2, "Input must be a 2D tensor.");
  const SimpleTensor &input = input_.data;
  SimpleTensor &global_amax = output_.amax;
  SimpleTensor &output_t = output_.data;
  SimpleTensor &scale_inv_t = output_.scale_inv;

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
  using TC = cutlass::float_e2m1_t;
  using TSFC = cutlass::float_ue4m3_t;

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

  if (m == 8192 && n == 5120) {
    k_tile_size = 512;
  } else if (m == 8192 && n == 10240) {
    k_tile_size = 1024;
  } else if (m == 8192 && n == 2560) {
    k_tile_size = 1280;
  } else if (m == 8192 && n == 11328) {
    k_tile_size = 1024;
  } else if (m == 8192 && n == 512) {
    k_tile_size = 256;
  } else if (m == 8192 && n == 3584) {
    k_tile_size = 512;
  } else if (m == 11328 && n == 8192) {
    k_tile_size = 1024;
  } else if (m == 5120 && n == 8192) {
    k_tile_size = 512;
  } else if (m == 10240 && n == 8192) {
    k_tile_size = 1024;
  } else if (m == 2560 && n == 8192) {
    k_tile_size = 1280;
  } else if (m == 512 && n == 8192) {
    k_tile_size = 256;
  } else if (m == 3584 && n == 8192) {
    k_tile_size = 512;
  } else if (m < 1024 || n < 1024) {
    k_tile_size = 512;
  }
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_stochastic_rounding, kUseStochasticRounding,
      detail::rht_gemm_ttt_wrapper<TA, TB, TC, TSFC, kUseStochasticRounding>(
          /*m=*/m,
          /*n=*/n,
          /*A=*/reinterpret_cast<TA const *>(input.dptr),
          /*B=*/reinterpret_cast<TB const *>(hadamard_matrix.dptr),
          /*C=*/reinterpret_cast<TC *>(output_t.dptr),
          /*SFC=*/reinterpret_cast<TSFC *>(scale_inv_t.dptr),
          /*global_amax=*/reinterpret_cast<float const *>(global_amax.dptr),
          /*rng_state=*/rng_state,
          /*sm_count=*/sm_count,
          /*stream=*/stream,
          /*k_tile_size=*/k_tile_size););
}

}  // namespace transformer_engine

void nvte_hadamard_transform_cast_fusion_columnwise(const NVTETensor input, NVTETensor output,
                                                    const NVTETensor hadamard_matrix,
                                                    const NVTEQuantizationConfig quant_config,
                                                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_hadamard_transform_cast_fusion_columnwise);
  using namespace transformer_engine;
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }
  hadamard_transform_cast_fusion_columnwise(
      *convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
      *convertNVTETensorCheck(hadamard_matrix), quant_config_cpp, stream);
}
