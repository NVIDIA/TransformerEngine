/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../stable_common.h"

#include <transformer_engine/comm_gemm_overlap.h>

#include <mutex>
#include <unordered_map>

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;
namespace te = transformer_engine;

// ============================================================================
// CommOverlap object registry
//
// CommOverlap objects are created by the Python shim and stored here.
// The stable ABI passes opaque int64_t handles (pointers cast to int).
// ============================================================================

static std::mutex g_comm_overlap_mutex;
static std::unordered_map<int64_t, std::unique_ptr<te::CommOverlapBase>> g_comm_overlaps;
static std::unordered_map<int64_t, std::unique_ptr<te::CommOverlapP2PBase>> g_comm_overlaps_p2p;
static int64_t g_next_handle = 1;

// ============================================================================
// Allgather/barrier callback registration
//
// Python registers a callback pair that implements allgather/barrier using
// torch.distributed. The callbacks are stored here and passed to
// CommOverlapCore during construction.
// ============================================================================

using AllgatherCallback = void (*)(void* global, size_t global_bytes,
                                   void* local, size_t local_bytes,
                                   const char* group);
using BarrierCallback = void (*)(const char* group);

static AllgatherCallback g_allgather_cb = nullptr;
static BarrierCallback g_barrier_cb = nullptr;

void register_comm_callbacks(int64_t allgather_fn_ptr, int64_t barrier_fn_ptr) {
  g_allgather_cb = reinterpret_cast<AllgatherCallback>(allgather_fn_ptr);
  g_barrier_cb = reinterpret_cast<BarrierCallback>(barrier_fn_ptr);
}

// ============================================================================
// CommOverlapBase construction/destruction
// ============================================================================

int64_t create_comm_overlap(
    std::vector<int64_t> buffer_shape, int64_t buffer_dtype,
    int64_t myrank, int64_t numranks, int64_t mylocal, int64_t numlocal,
    int64_t mynode, int64_t numnodes, int64_t tp_size,
    int64_t num_splits, int64_t num_max_streams, int64_t comm_cga_size,
    int64_t gemm_priority, int64_t comm_priority, int64_t num_comm_sm,
    bool set_sm_margin, bool atomic_gemm, bool rs_overlap_first_gemm) {
  std::vector<size_t> shape(buffer_shape.begin(), buffer_shape.end());
  auto dtype = static_cast<DType>(buffer_dtype);

  ExtAllgatherOp allgather_op;
  ExtBarrierOp barrier_op;

  if (g_allgather_cb && g_barrier_cb) {
    allgather_op = [](void* g, size_t gb, void* l, size_t lb, ExtComm comm) {
      g_allgather_cb(g, gb, l, lb, comm);
    };
    barrier_op = [](ExtComm comm) {
      g_barrier_cb(comm);
    };
  }

  auto co = std::make_unique<te::CommOverlapBase>(
      shape, dtype,
      static_cast<int>(myrank), static_cast<int>(numranks),
      static_cast<int>(mylocal), static_cast<int>(numlocal),
      static_cast<int>(mynode), static_cast<int>(numnodes),
      static_cast<int>(tp_size),
      allgather_op, barrier_op,
      static_cast<int>(num_splits), static_cast<int>(num_max_streams),
      static_cast<int>(comm_cga_size), static_cast<int>(gemm_priority),
      static_cast<int>(comm_priority), static_cast<int>(num_comm_sm),
      set_sm_margin, atomic_gemm, rs_overlap_first_gemm);

  std::lock_guard<std::mutex> lock(g_comm_overlap_mutex);
  int64_t handle = g_next_handle++;
  g_comm_overlaps[handle] = std::move(co);
  return handle;
}

void destroy_comm_overlap(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_comm_overlap_mutex);
  g_comm_overlaps.erase(handle);
}

// ============================================================================
// CommOverlapP2PBase construction/destruction
// ============================================================================

int64_t create_comm_overlap_p2p(
    std::vector<int64_t> buffer_shape, int64_t buffer_dtype,
    int64_t myrank, int64_t numranks, int64_t mylocal, int64_t numlocal,
    int64_t mynode, int64_t numnodes, int64_t tp_size,
    int64_t comm_type,
    int64_t num_max_streams, int64_t comm_cga_size,
    int64_t gemm_priority, int64_t comm_priority, int64_t num_comm_sm,
    bool set_sm_margin, bool use_ce, bool atomic_gemm, bool aggregate) {
  std::vector<size_t> shape(buffer_shape.begin(), buffer_shape.end());
  auto dtype = static_cast<DType>(buffer_dtype);

  ExtAllgatherOp allgather_op;
  ExtBarrierOp barrier_op;

  if (g_allgather_cb && g_barrier_cb) {
    allgather_op = [](void* g, size_t gb, void* l, size_t lb, ExtComm comm) {
      g_allgather_cb(g, gb, l, lb, comm);
    };
    barrier_op = [](ExtComm comm) {
      g_barrier_cb(comm);
    };
  }

  auto co = std::make_unique<te::CommOverlapP2PBase>(
      shape, dtype,
      static_cast<int>(myrank), static_cast<int>(numranks),
      static_cast<int>(mylocal), static_cast<int>(numlocal),
      static_cast<int>(mynode), static_cast<int>(numnodes),
      static_cast<int>(tp_size),
      allgather_op, barrier_op,
      static_cast<te::CommOverlapType>(comm_type),
      static_cast<int>(num_max_streams), static_cast<int>(comm_cga_size),
      static_cast<int>(gemm_priority), static_cast<int>(comm_priority),
      static_cast<int>(num_comm_sm),
      set_sm_margin, use_ce, atomic_gemm, aggregate);

  std::lock_guard<std::mutex> lock(g_comm_overlap_mutex);
  int64_t handle = g_next_handle++;
  g_comm_overlaps_p2p[handle] = std::move(co);
  return handle;
}

void destroy_comm_overlap_p2p(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_comm_overlap_mutex);
  g_comm_overlaps_p2p.erase(handle);
}

// ============================================================================
// Buffer operations (hot path wrappers — just pointer extraction)
// ============================================================================

static te::CommOverlapCore* get_core(int64_t handle) {
  {
    auto it = g_comm_overlaps.find(handle);
    if (it != g_comm_overlaps.end()) return it->second.get();
  }
  {
    auto it = g_comm_overlaps_p2p.find(handle);
    if (it != g_comm_overlaps_p2p.end()) return it->second.get();
  }
  NVTE_ERROR("Invalid CommOverlap handle: ", handle);
}

void comm_overlap_copy_into_buffer(Tensor input, int64_t handle,
                                   bool local_chunk) {
  auto input_ = torch::stable::contiguous(input);
  auto *co = get_core(handle);

  const size_t elem_size = input_.element_size();
  const size_t input_numel = static_cast<size_t>(input_.numel());
  const void *src = input_.data_ptr();

  const size_t ubuf_numel = co->get_ubuf().numel();
  void *dst = co->get_ubuf().dptr();
  int tp_size = co->get_tp_size();
  int tp_id = co->get_tp_id();

  if (local_chunk) {
    NVTE_CHECK(input_numel * tp_size == ubuf_numel,
               "Invalid tensor for local chunk copy");
    dst = reinterpret_cast<char*>(dst) +
          (ubuf_numel / tp_size) * tp_id * elem_size;
  } else {
    NVTE_CHECK(input_numel == ubuf_numel,
               "Invalid tensor for buffer copy");
  }

  auto stream = getCurrentCUDAStreamRaw(input_.get_device_index());
  cudaMemcpyAsync(dst, src, input_numel * elem_size,
                  cudaMemcpyDeviceToDevice, stream);
}

Tensor comm_overlap_get_buffer(int64_t handle, bool local_chunk,
                               int64_t dim0, int64_t dim1) {
  auto *co = get_core(handle);
  int tp_size = co->get_tp_size();
  int tp_id = co->get_tp_id();
  const auto &ubuf = co->get_ubuf();
  const size_t ubuf_numel = ubuf.numel();

  if (dim0 <= 0 || dim1 <= 0) {
    dim0 = static_cast<int64_t>(ubuf.size(0));
    dim1 = static_cast<int64_t>(ubuf.size(1));
    if (local_chunk) dim0 /= tp_size;
  }

  void *ptr = ubuf.dptr();
  if (local_chunk) {
    ptr = reinterpret_cast<char*>(ptr) +
          (ubuf_numel / tp_size) * tp_id * ubuf.element_size();
  }

  auto dtype = GetStableScalarType(ubuf.dtype());
  auto device_idx = torch::stable::accelerator::getCurrentDeviceIndex();
  std::vector<int64_t> shape = {dim0, dim1};
  std::vector<int64_t> strides = {dim1, 1};
  torch::headeronly::IntHeaderOnlyArrayRef size_ref(shape.data(), shape.size());
  torch::headeronly::IntHeaderOnlyArrayRef stride_ref(strides.data(), strides.size());
  torch::stable::Device device(torch::headeronly::DeviceType::CUDA, device_idx);

  return torch::stable::from_blob(ptr, size_ref, stride_ref, device, dtype);
}

// Return communication stream as raw cudaStream_t (cast to int64)
// Python wraps with torch.cuda.ExternalStream
int64_t comm_overlap_get_stream(int64_t handle) {
  auto it = g_comm_overlaps.find(handle);
  if (it != g_comm_overlaps.end()) {
    return reinterpret_cast<int64_t>(it->second->get_comm_stream());
  }
  NVTE_ERROR("Invalid CommOverlapBase handle: ", handle);
}

std::tuple<int64_t, int64_t> comm_overlap_p2p_get_streams(int64_t handle) {
  auto it = g_comm_overlaps_p2p.find(handle);
  if (it != g_comm_overlaps_p2p.end()) {
    auto &streams = it->second->get_send_streams();
    auto recv = it->second->get_recv_stream();
    return std::make_tuple(
        reinterpret_cast<int64_t>(streams.empty() ? nullptr : streams[0]),
        reinterpret_cast<int64_t>(recv));
  }
  NVTE_ERROR("Invalid CommOverlapP2PBase handle: ", handle);
}

// Bulk overlap AG with external GEMM
void bulk_overlap_ag_with_external_gemm(
    int64_t handle, int64_t send_stream_ptr, int64_t recv_stream_ptr) {
  auto it = g_comm_overlaps.find(handle);
  NVTE_CHECK(it != g_comm_overlaps.end(), "Invalid CommOverlapBase handle");
  auto main_stream = getCurrentCUDAStreamRaw();
  it->second->bulk_overlap_external_ag(
      reinterpret_cast<cudaStream_t>(send_stream_ptr),
      reinterpret_cast<cudaStream_t>(recv_stream_ptr),
      main_stream);
}

// Query helpers
int64_t comm_overlap_get_tp_size(int64_t handle) {
  return get_core(handle)->get_tp_size();
}

bool comm_overlap_is_atomic_gemm(int64_t handle) {
  return get_core(handle)->is_atomic_gemm();
}

bool comm_overlap_is_p2p(int64_t handle) {
  return get_core(handle)->is_p2p_overlap();
}

bool comm_overlap_is_fp8_ubuf(int64_t handle) {
  return get_core(handle)->is_fp8_ubuf();
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  // Callback registration
  m.def("register_comm_callbacks(int allgather_fn_ptr, int barrier_fn_ptr) -> ()");
  // CommOverlapBase lifecycle
  m.def("create_comm_overlap(int[] buffer_shape, int buffer_dtype, int myrank, int numranks, int mylocal, int numlocal, int mynode, int numnodes, int tp_size, int num_splits, int num_max_streams, int comm_cga_size, int gemm_priority, int comm_priority, int num_comm_sm, bool set_sm_margin, bool atomic_gemm, bool rs_overlap_first_gemm) -> int");
  m.def("destroy_comm_overlap(int handle) -> ()");
  // CommOverlapP2PBase lifecycle
  m.def("create_comm_overlap_p2p(int[] buffer_shape, int buffer_dtype, int myrank, int numranks, int mylocal, int numlocal, int mynode, int numnodes, int tp_size, int comm_type, int num_max_streams, int comm_cga_size, int gemm_priority, int comm_priority, int num_comm_sm, bool set_sm_margin, bool use_ce, bool atomic_gemm, bool aggregate) -> int");
  m.def("destroy_comm_overlap_p2p(int handle) -> ()");
  // Buffer operations
  m.def("comm_overlap_copy_into_buffer(Tensor input, int handle, bool local_chunk) -> ()");
  m.def("comm_overlap_get_buffer(int handle, bool local_chunk, int dim0, int dim1) -> Tensor");
  // Stream access
  m.def("comm_overlap_get_stream(int handle) -> int");
  m.def("comm_overlap_p2p_get_streams(int handle) -> (int, int)");
  // Queries
  m.def("bulk_overlap_ag_with_external_gemm(int handle, int send_stream_ptr, int recv_stream_ptr) -> ()");
  m.def("comm_overlap_get_tp_size(int handle) -> int");
  m.def("comm_overlap_is_atomic_gemm(int handle) -> bool");
  m.def("comm_overlap_is_p2p(int handle) -> bool");
  m.def("comm_overlap_is_fp8_ubuf(int handle) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("register_comm_callbacks", TORCH_BOX(register_comm_callbacks));
  m.impl("create_comm_overlap", TORCH_BOX(create_comm_overlap));
  m.impl("destroy_comm_overlap", TORCH_BOX(destroy_comm_overlap));
  m.impl("create_comm_overlap_p2p", TORCH_BOX(create_comm_overlap_p2p));
  m.impl("destroy_comm_overlap_p2p", TORCH_BOX(destroy_comm_overlap_p2p));
  m.impl("comm_overlap_copy_into_buffer", TORCH_BOX(comm_overlap_copy_into_buffer));
  m.impl("comm_overlap_get_buffer", TORCH_BOX(comm_overlap_get_buffer));
  m.impl("comm_overlap_get_stream", TORCH_BOX(comm_overlap_get_stream));
  m.impl("comm_overlap_p2p_get_streams", TORCH_BOX(comm_overlap_p2p_get_streams));
  m.impl("bulk_overlap_ag_with_external_gemm", TORCH_BOX(bulk_overlap_ag_with_external_gemm));
  m.impl("comm_overlap_get_tp_size", TORCH_BOX(comm_overlap_get_tp_size));
  m.impl("comm_overlap_is_atomic_gemm", TORCH_BOX(comm_overlap_is_atomic_gemm));
  m.impl("comm_overlap_is_p2p", TORCH_BOX(comm_overlap_is_p2p));
  m.impl("comm_overlap_is_fp8_ubuf", TORCH_BOX(comm_overlap_is_fp8_ubuf));
}
