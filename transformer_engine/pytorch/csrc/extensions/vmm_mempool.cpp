/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#include <pybind11/pybind11.h>

#ifdef NVTE_WITH_VMM_MEMPOOL
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/driver_api.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

#include <atomic>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>

#include "vmm_mempool_core.h"

namespace transformer_engine::pytorch {
namespace {

struct CUDAOps {
  int device;
  using CapturePin = std::shared_ptr<std::atomic<bool>>;

  // CUDA owns a reference across capture, graph cloning, instantiation and
  // in-flight replay. Its destructor callback only updates CPU state.
  CapturePin pin_capture(uintptr_t stream) {
    cudaStreamCaptureStatus status;
    unsigned long long capture_id = 0;
    cudaGraph_t graph = nullptr;
    // CUDA 13 folds the versioned entry points into the unversioned API and
    // adds the edge-data output parameter. Select by the runtime header version.
#if CUDART_VERSION >= 13000
    C10_CUDA_CHECK(cudaStreamGetCaptureInfo(reinterpret_cast<cudaStream_t>(stream), &status,
                                          &capture_id, &graph, nullptr, nullptr, nullptr));
#else
    C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(reinterpret_cast<cudaStream_t>(stream), &status,
                                             &capture_id, &graph, nullptr, nullptr));
#endif
    TORCH_CHECK(status == cudaStreamCaptureStatusActive && graph,
                "VMM allocation requires a valid active capture graph");
    auto pin = std::make_shared<std::atomic<bool>>(false);
    auto payload = std::make_unique<CapturePin>(pin);
    cudaUserObject_t object;
    C10_CUDA_CHECK(cudaUserObjectCreate(
        &object, payload.get(),
        [](void *value) {
          auto *owner = static_cast<CapturePin *>(value);
          (*owner)->store(true, std::memory_order_release);
          delete owner;
        },
        1, cudaUserObjectNoDestructorSync));
    payload.release();  // The user object owns the callback payload now.
    const auto result = cudaGraphRetainUserObject(graph, object, 1, cudaGraphUserObjectMove);
    if (result != cudaSuccess) {
      cudaUserObjectRelease(object);
      C10_CUDA_CHECK(result);
    }
    return pin;
  }
  bool capture_done(const CapturePin &pin) { return pin->load(std::memory_order_acquire); }

  struct OutOfMemory : std::runtime_error {
    OutOfMemory() : std::runtime_error("cuMemCreate exhausted VMM physical memory") {}
  };

  CUmemAllocationProp properties(int domain) const {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // CUDA 13.4 device-memory-node ABI, also used by the original vmm.py.
    prop.location.type = static_cast<CUmemLocationType>(6);
    prop.location.id = device | (domain << 8);
    return prop;
  }

  static auto api() { return c10::cuda::DriverAPI::get(); }

  uint64_t reserve(size_t bytes, size_t alignment) {
    CUdeviceptr base;
    C10_CUDA_DRIVER_CHECK(api()->cuMemAddressReserve_(&base, bytes, alignment, 0, 0));
    return base;
  }
  void free_address(uint64_t base, size_t bytes) {
    C10_CUDA_DRIVER_CHECK(api()->cuMemAddressFree_(base, bytes));
  }
  uint64_t create(size_t bytes, int domain) {
    CUmemGenericAllocationHandle handle;
    auto prop = properties(domain);
    const auto result = api()->cuMemCreate_(&handle, bytes, &prop, 0);
    if (result == CUDA_ERROR_OUT_OF_MEMORY) throw OutOfMemory();
    C10_CUDA_DRIVER_CHECK(result);
    return handle;
  }
  void release(uint64_t handle) { C10_CUDA_DRIVER_CHECK(api()->cuMemRelease_(handle)); }
  void map(uint64_t base, size_t bytes, uint64_t handle) {
    C10_CUDA_DRIVER_CHECK(api()->cuMemMap_(base, bytes, 0, handle, 0));
  }
  void unmap(uint64_t base, size_t bytes) {
    C10_CUDA_DRIVER_CHECK(api()->cuMemUnmap_(base, bytes));
  }
  void access(uint64_t base, size_t bytes) {
    CUmemAccessDesc desc{};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    C10_CUDA_DRIVER_CHECK(api()->cuMemSetAccess_(base, bytes, &desc, 1));
  }
  bool capturing(uintptr_t stream) {
    cudaStreamCaptureStatus status;
    C10_CUDA_CHECK(cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(stream), &status));
    return status != cudaStreamCaptureStatusNone;
  }
  uintptr_t event_create() {
    cudaEvent_t event;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return reinterpret_cast<uintptr_t>(event);
  }
  void event_record(uintptr_t event, uintptr_t stream) {
    C10_CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                                   reinterpret_cast<cudaStream_t>(stream)));
  }
  bool event_ready(uintptr_t event) {
    auto result = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
    if (result == cudaErrorNotReady) return false;
    C10_CUDA_CHECK(result);
    return true;
  }
  void event_destroy(uintptr_t event) {
    C10_CUDA_CHECK(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)));
  }
  void synchronize() { C10_CUDA_CHECK(cudaDeviceSynchronize()); }
};

using Engine = VMMPhysicalBlockPool<CUDAOps>;
std::mutex registry_mutex;
// Process-lifetime cache ownership also covers a Python controller disappearing
// before its tensors/graphs. Explicit cache drains return unused physical RAM.
std::vector<std::pair<int, std::shared_ptr<Engine>>> engines;

bool contains_pointer(int device, uint64_t pointer) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  for (const auto &[owner_device, engine] : engines) {
    if (owner_device == device && engine->contains(pointer)) return true;
  }
  return false;
}

void empty_all_caches() {
  std::vector<std::pair<int, std::shared_ptr<Engine>>> owners;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    owners = engines;
  }
  for (const auto &[device, engine] : owners) {
    c10::cuda::CUDAGuard guard(device);
    engine->empty_cache();
  }
}

struct AllocationPlan {
  std::shared_ptr<Engine> engine;
  int device;
  size_t logical_bytes;
  size_t mapped_bytes;
  uint64_t base = 0;
  std::atomic<bool> returned{false};
  std::mutex mutex;
  std::optional<c10::weak_intrusive_ptr<c10::StorageImpl>> storage;
};

// One ticket, one ordinary MemPool, one backing tensor. Its otherwise-unused
// native segment tail is never handed to another tensor.
class VMMAllocationTicket {
 public:
  VMMAllocationTicket(std::shared_ptr<Engine> engine, int device, size_t logical_bytes,
                      size_t mapped_bytes) {
    plan_ = std::make_shared<AllocationPlan>();
    plan_->engine = std::move(engine);
    plan_->device = device;
    plan_->logical_bytes = logical_bytes;
    plan_->mapped_bytes = mapped_bytes;
    allocator_ = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
        [plan = plan_](size_t segment_bytes, int device, cudaStream_t stream) -> void * {
          TORCH_CHECK(device == plan->device, "VMM allocation used on the wrong device");
          std::lock_guard<std::mutex> lock(plan->mutex);
          TORCH_CHECK(plan->base == 0,
                      "A VMM allocation ticket may back only one tensor/segment");
          TORCH_CHECK(segment_bytes >= plan->mapped_bytes,
                      "Native MemPool segment is too small for the localized tensor");
          c10::cuda::CUDAGuard guard(device);
          plan->base = plan->engine->allocate(plan->logical_bytes,
                                              reinterpret_cast<uintptr_t>(stream), segment_bytes);
          return reinterpret_cast<void *>(plan->base);
        },
        [plan = plan_](void *ptr, size_t, int device, cudaStream_t stream) {
          TORCH_CHECK(device == plan->device, "VMM allocation freed on the wrong device");
          c10::cuda::CUDAGuard guard(device);
          plan->engine->free(reinterpret_cast<uintptr_t>(ptr), reinterpret_cast<uintptr_t>(stream));
          plan->returned.store(true, std::memory_order_release);
        });
  }

  auto allocator() { return allocator_; }
  size_t mapped_bytes() { return plan_->mapped_bytes; }
  void watch(const at::Tensor &backing) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(backing.data_ptr()) == plan_->base,
                "Dedicated MemPool must return the segment base without an offset");
    TORCH_CHECK(backing.storage().nbytes() == plan_->mapped_bytes,
                "Unexpected localized backing Storage size");
    plan_->storage.emplace(backing.storage().getWeakStorageImpl());
  }
  bool storage_alive() { return plan_->storage && !plan_->storage->expired(); }
  bool trim(c10::cuda::MempoolId_t id) {
    // The Python owner drops this dedicated pool before calling trim. Its
    // standard native cache can now return the segment once Storage/streams/
    // native graph references permit it. Never flush the default pool.
    c10::cuda::CUDAGuard guard(plan_->device);
    if (plan_->base && !plan_->returned.load(std::memory_order_acquire)) {
      c10::cuda::CUDACachingAllocator::emptyCache(id);
    }
    return !plan_->base || plan_->returned.load(std::memory_order_acquire);
  }

 private:
  std::shared_ptr<AllocationPlan> plan_;
  std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator_;
};

class VMMMemPoolBackend {
 public:
  VMMMemPoolBackend(int device, int domain, size_t block_bytes)
      : device_(device), domain_(domain) {
    c10::cuda::CUDAGuard guard(device_);
    int version = 0, count = 0;
    C10_CUDA_CHECK(cudaDriverGetVersion(&version));
    TORCH_CHECK(version >= 13040, "VMM MemPool requires a CUDA 13.4 or newer driver");
    C10_CUDA_DRIVER_CHECK(CUDAOps::api()->cuDeviceGetAttribute_(
        &count, static_cast<CUdevice_attribute>(149), device_));
    TORCH_CHECK(count >= 2, "VMM MemPool requires at least two locality domains");
    CUDAOps ops{device_};
    size_t granularity = 1;
    for (int ld : {0, 1}) {
      size_t value = 0;
      auto prop = ops.properties(ld);
      C10_CUDA_DRIVER_CHECK(CUDAOps::api()->cuMemGetAllocationGranularity_(
          &value, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
      granularity = std::lcm(granularity, value);
    }
    if (!block_bytes) block_bytes = granularity;
    TORCH_CHECK(block_bytes % granularity == 0 && (block_bytes & (block_bytes - 1)) == 0,
                "VMM block_bytes must be a power of two and a multiple of both "
                "locality-domain granularities");
    engine_ = std::make_shared<Engine>(ops, block_bytes, domain);
    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      engines.emplace_back(device_, engine_);
    }
    block_bytes_ = block_bytes;
    // This adapter uses ordinary pluggable-allocator C++ APIs. Managed routing
    // itself is a runtime feature of the patched PyTorch MemPool, not a TE build
    // dependency on its optional managed-allocator factory/Python helper.
    allocator_ = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
        [engine = engine_, device](size_t bytes, int owner, cudaStream_t stream) -> void * {
          TORCH_CHECK(owner == device, "Managed VMM pool used on the wrong device");
          c10::cuda::CUDAGuard guard(device);
          return reinterpret_cast<void *>(
              engine->allocate(bytes, reinterpret_cast<uintptr_t>(stream)));
        },
        [engine = engine_, device](void *ptr, size_t, int owner, cudaStream_t stream) {
          TORCH_CHECK(owner == device, "Managed VMM pool freed on the wrong device");
          c10::cuda::CUDAGuard guard(device);
          engine->free(reinterpret_cast<uintptr_t>(ptr), reinterpret_cast<uintptr_t>(stream));
        });
    using Pluggable = torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator;
    std::static_pointer_cast<Pluggable>(allocator_)->set_reset_fn([engine = engine_, device] {
      c10::cuda::CUDAGuard guard(device);
      engine->empty_cache();
    });
  }

  auto allocator() { return allocator_; }
  std::shared_ptr<VMMAllocationTicket> ticket(size_t logical_bytes) {
    const size_t parts = domain_ == -1 ? 2 : 1;
    TORCH_CHECK(logical_bytes > 0 &&
                logical_bytes <= std::numeric_limits<size_t>::max() - (parts - 1),
                "Invalid localized allocation size");
    size_t half = (logical_bytes + parts - 1) / parts;
    TORCH_CHECK(half <= std::numeric_limits<size_t>::max() - block_bytes_ + 1,
                "Localized allocation size overflow");
    half = (half + block_bytes_ - 1) / block_bytes_ * block_bytes_;
    TORCH_CHECK(half <= std::numeric_limits<size_t>::max() / parts,
                "Localized allocation size overflow");
    return std::make_shared<VMMAllocationTicket>(engine_, device_, logical_bytes, half * parts);
  }
  auto stats() { return engine_->stats(); }
  void collect_completed() {
    c10::cuda::CUDAGuard guard(device_);
    CUDAOps ops{device_};
    const auto stream = c10::cuda::getCurrentCUDAStream(device_);
    TORCH_CHECK(!ops.capturing(reinterpret_cast<uintptr_t>(stream.stream())),
                "Cannot collect VMM mappings during capture");
    engine_->collect_completed();
  }
  bool contains(uint64_t ptr) { return engine_->contains(ptr); }
  void empty_cache() {
    c10::cuda::CUDAGuard guard(device_);
    engine_->empty_cache();
  }

 private:
  int device_;
  int domain_;
  size_t block_bytes_;
  std::shared_ptr<Engine> engine_;
  std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator_;
};

}  // namespace

void register_vmm_mempool(pybind11::module_ &m) {
  namespace py = pybind11;
  m.def("vmm_mempool_contains", &contains_pointer);
  m.def("vmm_mempool_empty_all_caches", &empty_all_caches);
  py::class_<VMMAllocationTicket, std::shared_ptr<VMMAllocationTicket>>(m, "VMMAllocationTicket")
      .def_property_readonly("allocator", &VMMAllocationTicket::allocator)
      .def_property_readonly("mapped_bytes", &VMMAllocationTicket::mapped_bytes)
      .def("watch", &VMMAllocationTicket::watch)
      .def("storage_alive", &VMMAllocationTicket::storage_alive)
      .def("trim", &VMMAllocationTicket::trim);
  py::class_<VMMMemPoolBackend>(m, "VMMMemPoolBackend")
      .def(py::init<int, int, size_t>(), py::arg("device"), py::arg("domain") = -1,
           py::arg("block_bytes") = 0)
      .def_property_readonly("allocator", &VMMMemPoolBackend::allocator)
      .def("ticket", &VMMMemPoolBackend::ticket)
      .def("empty_cache", &VMMMemPoolBackend::empty_cache)
      .def("collect_completed", &VMMMemPoolBackend::collect_completed)
      .def("stats", &VMMMemPoolBackend::stats)
      .def("contains", &VMMMemPoolBackend::contains);
}
}  // namespace transformer_engine::pytorch
#else
namespace transformer_engine::pytorch {
void register_vmm_mempool(pybind11::module_ &) {}
}  // namespace transformer_engine::pytorch
#endif
