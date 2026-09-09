/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

#ifdef NVTE_WITH_NCCL_DEVICE_CP

#include <nccl.h>

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 7)
#error "Native CP transport requires NCCL 2.29.7 or newer"
#endif

#include <nccl_device.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace transformer_engine::pytorch {
namespace {

#define NVTE_CP_NCCL_CHECK(call)                                                          \
  do {                                                                                    \
    const ncclResult_t result_ = (call);                                                  \
    TORCH_CHECK(result_ == ncclSuccess, #call, " failed: ", ncclGetErrorString(result_)); \
  } while (0)

#define NVTE_CP_CUDA_CHECK(call)                                                          \
  do {                                                                                    \
    const cudaError_t result_ = (call);                                                   \
    TORCH_CHECK(result_ == cudaSuccess, #call, " failed: ", cudaGetErrorString(result_)); \
  } while (0)

constexpr size_t kArenaAlignment = 256;
constexpr int kThreads = 256;
constexpr int kMaxCopyBlocks = 16;
constexpr int kNumChannels = 3;  // forward, backward, aux-loss
constexpr int kGinContexts = 1;
constexpr int kGinQueueDepth = 8;
using Counter = unsigned long long;  // NOLINT(runtime/int)

size_t align_up(size_t value) {
  return (value + kArenaAlignment - 1) / kArenaAlignment * kArenaAlignment;
}

struct NativeCPTransport {
  ncclComm_t comm = nullptr;
  ncclDevComm dev_comm{};
  ncclWindow_t window = nullptr;
  void *allocation = nullptr;
  void *payload = nullptr;
  size_t payload_bytes = 0;
  size_t payload_offset = 0;
  size_t signal_offset = 0;
  size_t signal_shadow_offset = 0;
  size_t ready_offset = 0;
  int device = -1;
  int rank = -1;
  int nranks = 0;
  bool dev_comm_created = false;
  bool window_registered = false;
  std::array<cudaStream_t, kNumChannels> streams{};
  std::array<cudaEvent_t, kNumChannels> ready_events{};
  std::array<cudaEvent_t, kNumChannels> done_events{};
  std::array<bool, kNumChannels> outstanding{};
  std::vector<Counter> ready_expected;

  ~NativeCPTransport() noexcept {
    int previous_device = -1;
    bool restore_device = false;
    if (device >= 0) {
      if (cudaGetDevice(&previous_device) == cudaSuccess) {
        restore_device = previous_device != device;
      }
      if (previous_device != device) cudaSetDevice(device);
    }
    for (cudaStream_t stream : streams) {
      if (stream != nullptr) cudaStreamSynchronize(stream);
    }
    for (cudaEvent_t event : ready_events) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    for (cudaEvent_t event : done_events) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    for (cudaStream_t stream : streams) {
      if (stream != nullptr) cudaStreamDestroy(stream);
    }
    if (dev_comm_created) ncclDevCommDestroy(comm, &dev_comm);
    if (window_registered) ncclCommWindowDeregister(comm, window);
    if (allocation != nullptr) ncclMemFree(allocation);
    if (restore_device) cudaSetDevice(previous_device);
  }
};

NativeCPTransport *unwrap(int64_t handle) {
  TORCH_CHECK(handle != 0, "Native CP transport handle is null");
  return reinterpret_cast<NativeCPTransport *>(handle);
}

__device__ __forceinline__ Counter system_load(Counter *ptr) { return atomicAdd_system(ptr, 0ULL); }

__device__ __forceinline__ void copy_to_lsa_peer(void *dst_void, const void *src_void,
                                                 size_t bytes) {
  auto *dst = static_cast<uint8_t *>(dst_void);
  const auto *src = static_cast<const uint8_t *>(src_void);
  const uintptr_t packed =
      reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src) | bytes;
  if ((packed & (alignof(uint4) - 1)) == 0) {
    auto *dst4 = reinterpret_cast<uint4 *>(dst);
    const auto *src4 = reinterpret_cast<const uint4 *>(src);
    const size_t count4 = bytes / sizeof(uint4);
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count4;
         index += blockDim.x * gridDim.x) {
      dst4[index] = src4[index];
    }
    return;
  }
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < bytes;
       index += blockDim.x * gridDim.x) {
    dst[index] = src[index];
  }
}

__global__ void cp_native_prepare_kernel(ncclDevComm dev_comm, ncclWindow_t window, int rank,
                                         int recv_peer, unsigned int channel, size_t ready_offset) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  const ncclTeam world = ncclTeamWorld(dev_comm);
  const ncclTeam lsa = ncclTeamLsa(dev_comm);
  const bool recv_is_lsa = ncclTeamRankIsMember(lsa, world, recv_peer);
  const size_t ready_index = (static_cast<size_t>(rank) * kNumChannels + channel) * sizeof(Counter);

  // A direct store has no implicit ncclRecv rendezvous. Advertise that this
  // rank has finished with its receive buffer before the peer may overwrite it.
  if (recv_is_lsa) {
    const int recv_lsa_rank = ncclTeamRankToTeam(lsa, world, recv_peer);
    auto *remote_ready = static_cast<Counter *>(
        ncclGetLsaPointer(window, ready_offset + ready_index, recv_lsa_rank));
    __threadfence_system();
    atomicAdd_system(remote_ready, 1ULL);
  } else {
    ncclGin gin{dev_comm, 0};
    gin.signal(world, recv_peer, ncclGin_VASignalInc{window, ready_offset + ready_index},
               ncclCoopThread{});
    gin.flush(ncclCoopThread{});
  }
}

__global__ void cp_native_send_recv_kernel(ncclDevComm dev_comm, ncclWindow_t window,
                                           size_t send_offset, size_t recv_offset, size_t bytes,
                                           int send_peer, int recv_peer, int rank,
                                           unsigned int channel, Counter ready_expected,
                                           size_t signal_offset, size_t signal_shadow_offset,
                                           size_t ready_offset) {
  const ncclTeam world = ncclTeamWorld(dev_comm);
  const ncclTeam lsa = ncclTeamLsa(dev_comm);
  const bool send_is_lsa = ncclTeamRankIsMember(lsa, world, send_peer);
  const bool recv_is_lsa = ncclTeamRankIsMember(lsa, world, recv_peer);

  if (send_is_lsa) {
    const size_t ready_index =
        (static_cast<size_t>(send_peer) * kNumChannels + channel) * sizeof(Counter);
    if (threadIdx.x == 0) {
      auto *local_ready =
          static_cast<Counter *>(ncclGetLocalPointer(window, ready_offset + ready_index));
      while (system_load(local_ready) < ready_expected) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
      }
    }
    __syncthreads();
    const int send_lsa_rank = ncclTeamRankToTeam(lsa, world, send_peer);
    void *send_local = ncclGetLocalPointer(window, send_offset);
    void *recv_remote = ncclGetLsaPointer(window, recv_offset, send_lsa_rank);
    copy_to_lsa_peer(recv_remote, send_local, bytes);
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence_system();
      auto *remote_signal = static_cast<Counter *>(ncclGetLsaPointer(
          window,
          signal_offset + (static_cast<size_t>(rank) * kNumChannels + channel) * sizeof(Counter),
          send_lsa_rank));
      atomicAdd_system(remote_signal, 1ULL);
    }
    __syncthreads();
  } else if (blockIdx.x == 0) {
    ncclGin gin{dev_comm, 0};
    const size_t ready_index =
        (static_cast<size_t>(send_peer) * kNumChannels + channel) * sizeof(Counter);
    gin.waitSignal(ncclCoopCta{}, window, ready_offset + ready_index, ready_expected);

    const size_t completion_index =
        (static_cast<size_t>(rank) * kNumChannels + channel) * sizeof(Counter);
    gin.put(world, send_peer, window, recv_offset, window, send_offset, bytes,
            ncclGin_VASignalInc{window, signal_offset + completion_index}, ncclGin_None{},
            ncclCoopCta{});
    gin.flush(ncclCoopCta{});
  }

  if (recv_is_lsa) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      const size_t completion_index =
          (static_cast<size_t>(recv_peer) * kNumChannels + channel) * sizeof(Counter);
      auto *local_signal =
          static_cast<Counter *>(ncclGetLocalPointer(window, signal_offset + completion_index));
      auto *local_shadow = static_cast<Counter *>(
          ncclGetLocalPointer(window, signal_shadow_offset + completion_index));
      const Counter expected = *local_shadow + gridDim.x;
      while (system_load(local_signal) < expected) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
      }
      *local_shadow = expected;
      __threadfence_system();
    }
    __syncthreads();
  } else if (blockIdx.x == 0) {
    ncclGin gin{dev_comm, 0};
    const size_t completion_index =
        (static_cast<size_t>(recv_peer) * kNumChannels + channel) * sizeof(Counter);
    auto *local_shadow = static_cast<Counter *>(
        ncclGetLocalPointer(window, signal_shadow_offset + completion_index));
    const Counter expected = *local_shadow + 1ULL;
    gin.waitSignal(ncclCoopCta{}, window, signal_offset + completion_index, expected);
    if (threadIdx.x == 0) {
      *local_shadow = expected;
      __threadfence_system();
    }
    __syncthreads();
  }
}

void validate_tensor(const NativeCPTransport &transport, const at::Tensor &tensor,
                     const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.get_device() == transport.device, name, " is on CUDA device ",
              tensor.get_device(), " but transport uses ", transport.device);
  const auto begin = reinterpret_cast<uintptr_t>(transport.payload);
  const auto end = begin + transport.payload_bytes;
  const auto tensor_begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const auto tensor_end = tensor_begin + tensor.nbytes();
  TORCH_CHECK(tensor_begin >= begin && tensor_end <= end, name,
              " must be a view of the native CP transport arena");
}

}  // namespace

std::tuple<int64_t, at::Tensor> cp_native_transport_create(int64_t nccl_comm_ptr,
                                                           int64_t payload_bytes) {
  TORCH_CHECK(nccl_comm_ptr != 0, "nccl_comm_ptr must not be null");
  TORCH_CHECK(payload_bytes > 0, "payload_bytes must be positive");
  TORCH_CHECK(payload_bytes <= std::numeric_limits<int64_t>::max() - 2 * kArenaAlignment,
              "payload_bytes is too large");

  std::unique_ptr<NativeCPTransport> transport(new NativeCPTransport());
  transport->comm = reinterpret_cast<ncclComm_t>(nccl_comm_ptr);
  transport->device = c10::cuda::current_device();
  transport->payload_bytes = static_cast<size_t>(payload_bytes);

  int runtime_version = 0;
  NVTE_CP_NCCL_CHECK(ncclGetVersion(&runtime_version));
  TORCH_CHECK(runtime_version == NCCL_VERSION_CODE,
              "NCCL Device API GIN requires matching compile/runtime versions; ", "compiled with ",
              NCCL_VERSION_CODE, ", loaded ", runtime_version);

  ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  NVTE_CP_NCCL_CHECK(ncclCommQueryProperties(transport->comm, &properties));
  TORCH_CHECK(properties.deviceApiSupport,
              "The parent NCCL communicator does not support NCCL Device API");
  transport->rank = properties.rank;
  transport->nranks = properties.nRanks;
  const int lsa_size = ncclTeamLsa(transport->comm).nRanks;
  const bool needs_gin = transport->nranks != lsa_size;
  TORCH_CHECK(transport->nranks == lsa_size || properties.ginType != NCCL_GIN_TYPE_NONE,
              "The parent communicator spans multiple LSA domains but GIN is unavailable");

  const size_t peer_channel_bytes =
      static_cast<size_t>(transport->nranks) * kNumChannels * sizeof(Counter);
  transport->signal_offset = 0;
  transport->signal_shadow_offset = align_up(transport->signal_offset + peer_channel_bytes);
  transport->ready_offset = align_up(transport->signal_shadow_offset + peer_channel_bytes);
  transport->payload_offset = align_up(transport->ready_offset + peer_channel_bytes);
  TORCH_CHECK(
      transport->payload_bytes <= std::numeric_limits<size_t>::max() - transport->payload_offset,
      "Native CP transport allocation size overflow");
  const size_t allocation_bytes = transport->payload_offset + transport->payload_bytes;

  NVTE_CP_NCCL_CHECK(ncclMemAlloc(&transport->allocation, allocation_bytes));
  NVTE_CP_CUDA_CHECK(cudaMemset(transport->allocation, 0, transport->payload_offset));
  NVTE_CP_NCCL_CHECK(ncclCommWindowRegister(transport->comm, transport->allocation,
                                            allocation_bytes, &transport->window,
                                            NCCL_WIN_DEFAULT));
  transport->window_registered = true;

  ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  if (needs_gin) {
    requirements.ginContextCount = kGinContexts;
    requirements.ginQueueDepth = kGinQueueDepth;
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  }
  NVTE_CP_NCCL_CHECK(ncclDevCommCreate(transport->comm, &requirements, &transport->dev_comm));
  transport->dev_comm_created = true;

  int least_priority = 0;
  int greatest_priority = 0;
  NVTE_CP_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  transport->ready_expected.resize(transport->nranks * kNumChannels, 0);
  for (int channel = 0; channel < kNumChannels; ++channel) {
    NVTE_CP_CUDA_CHECK(cudaStreamCreateWithPriority(&transport->streams[channel],
                                                    cudaStreamNonBlocking, greatest_priority));
    NVTE_CP_CUDA_CHECK(
        cudaEventCreateWithFlags(&transport->ready_events[channel], cudaEventDisableTiming));
    NVTE_CP_CUDA_CHECK(
        cudaEventCreateWithFlags(&transport->done_events[channel], cudaEventDisableTiming));
  }

  transport->payload = static_cast<uint8_t *>(transport->allocation) + transport->payload_offset;
  auto options =
      at::TensorOptions().device(at::Device(at::kCUDA, transport->device)).dtype(at::kByte);
  at::Tensor arena = at::from_blob(transport->payload, {payload_bytes}, [](void *) {}, options);
  const auto handle = reinterpret_cast<int64_t>(transport.release());
  return {handle, arena};
}

void cp_native_transport_destroy(int64_t handle) { delete unwrap(handle); }

int64_t cp_native_transport_send_recv(int64_t handle, at::Tensor send_tensor,
                                      at::Tensor recv_tensor, int64_t send_peer, int64_t recv_peer,
                                      int64_t channel) {
  NativeCPTransport *transport = unwrap(handle);
  at::cuda::CUDAGuard device_guard(at::Device(at::kCUDA, transport->device));
  TORCH_CHECK(channel >= 0 && channel < kNumChannels,
              "channel is outside the transport channel range");
  TORCH_CHECK(send_peer >= 0 && send_peer < transport->nranks,
              "send_peer is outside the parent communicator");
  TORCH_CHECK(recv_peer >= 0 && recv_peer < transport->nranks,
              "recv_peer is outside the parent communicator");
  TORCH_CHECK(send_peer != transport->rank || recv_peer != transport->rank,
              "Native CP send/recv is unnecessary when both peers are self");
  validate_tensor(*transport, send_tensor, "send_tensor");
  validate_tensor(*transport, recv_tensor, "recv_tensor");
  TORCH_CHECK(send_tensor.nbytes() == recv_tensor.nbytes(),
              "send_tensor and recv_tensor must have the same byte size");
  TORCH_CHECK(!transport->outstanding[channel], "Native CP transport channel ", channel,
              " is reused before its previous work is waited");

  const auto base = reinterpret_cast<uintptr_t>(transport->allocation);
  const size_t send_offset = reinterpret_cast<uintptr_t>(send_tensor.data_ptr()) - base;
  const size_t recv_offset = reinterpret_cast<uintptr_t>(recv_tensor.data_ptr()) - base;
  const size_t bytes = send_tensor.nbytes();
  TORCH_CHECK(bytes > 0, "Native CP transport does not accept an empty payload");
  const size_t bytes_per_copy_block = kThreads * sizeof(uint4);
  const int copy_blocks = static_cast<int>(
      std::min<size_t>(kMaxCopyBlocks, (bytes + bytes_per_copy_block - 1) / bytes_per_copy_block));

  const cudaStream_t caller_stream = at::cuda::getCurrentCUDAStream().stream();
  const int channel_index = static_cast<int>(channel);
  Counter &ready_expected =
      transport->ready_expected[static_cast<size_t>(send_peer) * kNumChannels + channel_index];
  ++ready_expected;
  NVTE_CP_CUDA_CHECK(cudaEventRecord(transport->ready_events[channel_index], caller_stream));
  NVTE_CP_CUDA_CHECK(cudaStreamWaitEvent(transport->streams[channel_index],
                                         transport->ready_events[channel_index], 0));
  cp_native_prepare_kernel<<<1, 1, 0, transport->streams[channel_index]>>>(
      transport->dev_comm, transport->window, transport->rank, static_cast<int>(recv_peer),
      static_cast<unsigned int>(channel), transport->ready_offset);
  NVTE_CP_CUDA_CHECK(cudaGetLastError());
  cp_native_send_recv_kernel<<<copy_blocks, kThreads, 0, transport->streams[channel_index]>>>(
      transport->dev_comm, transport->window, send_offset, recv_offset, bytes,
      static_cast<int>(send_peer), static_cast<int>(recv_peer), transport->rank,
      static_cast<unsigned int>(channel), ready_expected, transport->signal_offset,
      transport->signal_shadow_offset, transport->ready_offset);
  NVTE_CP_CUDA_CHECK(cudaGetLastError());
  NVTE_CP_CUDA_CHECK(
      cudaEventRecord(transport->done_events[channel_index], transport->streams[channel_index]));
  transport->outstanding[channel_index] = true;
  return channel;
}

void cp_native_transport_wait(int64_t handle, int64_t channel) {
  NativeCPTransport *transport = unwrap(handle);
  at::cuda::CUDAGuard device_guard(at::Device(at::kCUDA, transport->device));
  TORCH_CHECK(channel >= 0 && channel < kNumChannels,
              "channel is outside the transport channel range");
  const int channel_index = static_cast<int>(channel);
  TORCH_CHECK(transport->outstanding[channel_index], "Native CP transport channel ", channel,
              " has no outstanding work");
  const cudaStream_t caller_stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CP_CUDA_CHECK(cudaStreamWaitEvent(caller_stream, transport->done_events[channel_index], 0));
  transport->outstanding[channel_index] = false;
}

}  // namespace transformer_engine::pytorch

#endif  // NVTE_WITH_NCCL_DEVICE_CP
