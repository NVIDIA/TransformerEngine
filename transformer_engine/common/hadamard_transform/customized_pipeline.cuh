/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_HADAMARD_TRANSFORM_CUSTOMIZED_PIPELINE_CUH_
#define TRANSFORMER_ENGINE_COMMON_HADAMARD_TRANSFORM_CUSTOMIZED_PIPELINE_CUH_

#include "cutlass/pipeline/sm100_pipeline.hpp"

namespace cutlass {

using namespace cute;
namespace detail {
// Producer-consumer pipeline implementation
// for UMMA producer. In this case, UMMA barrier arrives are used
// by producer_commit. Use case, accumulator generation as
// the result of MMA instructions.
template <int Stages_, class ClusterShape = Shape<int, int, _1>,
          class AtomThrShape_MNK_ = Shape<_1, _1, _1> >
class CustomizedPipelineTmaUmmaAsync {
 public:
  static constexpr uint32_t Stages = Stages_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;

 private:
  using Impl = PipelineTmaAsync<Stages>;

 public:
  using FullBarrier = typename Impl::FullBarrier;
  using EmptyBarrier = typename Impl::EmptyBarrier;
  using ProducerBarrierType = typename Impl::ProducerBarrierType;
  using ConsumerBarrierType = typename Impl::ConsumerBarrierType;
  using PipelineState = typename Impl::PipelineState;
  using SharedStorage = typename Impl::SharedStorage;
  using ThreadCategory = typename Impl::ThreadCategory;
  using Params = typename Impl::Params;

  using McastDirection = McastDirection;

  // Helper function to initialize barriers
  static CUTLASS_DEVICE void init_barriers(SharedStorage& storage, Params params,
                                           ClusterShape cluster_shape) {
    int warp_idx = canonical_warp_idx_sync();
    if (warp_idx == params.initializing_warp) {
      // Barrier FULL and EMPTY init
      constexpr int producer_arv_cnt = 1;
      auto atom_thr_shape = AtomThrShape_MNK{};

      uint32_t multicast_consumer_arrival_count = params.num_consumers;  // If cluster_size is 1
      if (cute::size(cluster_shape) > 1) {
        multicast_consumer_arrival_count =
            ((cute::size<0>(cluster_shape) / cute::size<0>(atom_thr_shape)) +
             (cute::size<1>(cluster_shape) / cute::size<1>(atom_thr_shape)) - 1) *
            params.num_consumers;
      }
      CUTLASS_ASSERT(multicast_consumer_arrival_count > 0 &&
                     "Multicast consumer arrival count must be non-zero");
      CUTLASS_ASSERT(producer_arv_cnt > 0 && "Producer arrival count must be non-zero");
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<
          decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt,
          multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  CUTLASS_DEVICE
  void init_masks(ClusterShape cluster_shape,
                  dim3 block_id_in_cluster = cute::block_id_in_cluster()) {
    // Calculate consumer mask
    if (params_.role == ThreadCategory::Consumer) {
      block_id_mask_ = detail::calculate_multicast_mask<McastDirection::kRowCol>(
          cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    }
  }

  CUTLASS_DEVICE
  void init_masks(ClusterShape cluster_shape, McastDirection mcast_direction) {
    // Calculate consumer mask
    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    if (mcast_direction == McastDirection::kRow) {
      block_id_mask_ = detail::calculate_multicast_mask<McastDirection::kRow>(
          cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    } else {
      block_id_mask_ = detail::calculate_multicast_mask<McastDirection::kCol>(
          cluster_shape, AtomThrShape_MNK{}, block_id_in_cluster);
    }
  }

  // Constructor by default initializes barriers and calculates masks.
  // These operations can be explicity deferred by specifying InitBarriers and InitMasks.
  // If deferred, user code needs to guarantee init_masks and/or init_barriers is/are called.
  template <typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  CUTLASS_DEVICE CustomizedPipelineTmaUmmaAsync(SharedStorage& storage, Params params,
                                                ClusterShape cluster_shape, InitBarriers = {},
                                                InitMasks = {})
      : impl_(storage, params, cluster_shape, cute::false_type{}, InitMasks{}),
        params_(params),
        empty_barrier_ptr_(&storage.empty_barrier_[0]),
        full_barrier_ptr_(&storage.full_barrier_[0]) {
    static_assert(cute::is_same_v<InitBarriers, cute::true_type> ||
                  cute::is_same_v<InitBarriers, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params_, cluster_shape);
    }

    static_assert(cute::is_same_v<InitMasks, cute::true_type> ||
                  cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      init_masks(cluster_shape);
    }
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.
  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return impl_.producer_try_acquire(state, skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state,
                        ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.producer_acquire(state, barrier_token);
  }

  CUTLASS_DEVICE
  void producer_expect_transaction(PipelineState state, uint32_t transaction_bytes) {
    impl_.producer_expect_transaction(state, transaction_bytes);
  }

  // NOP for TMA based mainloop
  CUTLASS_DEVICE
  void producer_commit(PipelineState state, uint32_t bytes) { impl_.producer_commit(state, bytes); }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) { impl_.producer_tail(state); }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return impl_.producer_get_barrier(state);
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return impl_.consumer_try_wait(state, skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state,
                     ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    impl_.consumer_wait(state, barrier_token);
  }

  CUTLASS_DEVICE
  void umma_consumer_release(PipelineState state) { umma_consumer_release(state.index(), false); }
  CUTLASS_DEVICE
  void consumer_release(PipelineState state) { impl_.consumer_release(state); }

 private:
  Impl impl_;
  Params params_;
  EmptyBarrier* empty_barrier_ptr_;
  FullBarrier* full_barrier_ptr_;
  uint16_t block_id_mask_ = 0;
  static constexpr bool is_2sm_mma = size(AtomThrShape_MNK{}) > 1;

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notified.
  CUTLASS_DEVICE
  void umma_consumer_release(uint32_t stage, uint32_t skip) {
    detail::pipeline_check_is_consumer(params_.role);
    uint64_t* smem_ptr = reinterpret_cast<uint64_t*>(&empty_barrier_ptr_[stage]);
    // {$nv-release-never begin}
    // TODO: Needs to be updated once Blackwell specialized pipeline is implemented.
    // XMMA style bar_peek will be tested. We will need to revisit skip interface and
    // what skip means when we have bar_peek functionality.
    // A separate MR will implement MMA_2x1SM specialized pipeline.
    // {$nv-release-never end}
    if constexpr (is_2sm_mma) {  // Mma cluster shape is 2x1
      if (!skip) {
        cutlass::arch::umma_arrive_multicast_2x1SM(smem_ptr, block_id_mask_);
      }
    } else {
      if (!skip) {
        if constexpr (cute::is_static_v<ClusterShape> && size(ClusterShape{}) == 1) {
          cutlass::arch::umma_arrive(smem_ptr);
        } else {
          cutlass::arch::umma_arrive_multicast(smem_ptr, block_id_mask_);
        }
      }
    }
  }
};
}  // namespace detail
}  // namespace cutlass

#endif  // TRANSFORMER_ENGINE_COMMON_HADAMARD_TRANSFORM_CUSTOMIZED_PIPELINE_CUH_
