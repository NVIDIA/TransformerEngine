/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#include <pybind11/pybind11.h>

#ifdef NVTE_WITH_VMM_MEMPOOL
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/driver_api.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <numeric>
#include <string>

#include "vmm_mempool_core.h"

namespace transformer_engine::pytorch {
namespace {

struct CUDAOps {
  int device;
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
std::vector<std::pair<int, std::weak_ptr<Engine>>> engines;

bool contains_pointer(int device, uint64_t pointer) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  bool found = false;
  for (auto it = engines.begin(); it != engines.end();) {
    if (auto engine = it->second.lock()) {
      if (it->first == device && engine->contains(pointer)) found = true;
      ++it;
    } else {
      it = engines.erase(it);
    }
  }
  return found;
}

class VMMMemPoolBackend {
 public:
  VMMMemPoolBackend(int device, int domain, size_t block_bytes) : device_(device) {
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
    using namespace torch::cuda::CUDAPluggableAllocator;
    // No callback enters Python or reenters PyTorch's caching allocator.
    allocator_ = createCustomManagedPoolAllocator(
        [engine = engine_, device](size_t size, int requested_device,
                                   cudaStream_t stream) -> void * {
          TORCH_CHECK(requested_device == device, "VMM MemPool used on the wrong device");
          c10::cuda::CUDAGuard guard(device);
          return reinterpret_cast<void *>(
              engine->allocate(size, reinterpret_cast<uintptr_t>(stream)));
        },
        [engine = engine_, device](void *ptr, size_t, int requested_device, cudaStream_t stream) {
          TORCH_CHECK(requested_device == device, "VMM MemPool freed on the wrong device");
          c10::cuda::CUDAGuard guard(device);
          engine->free(reinterpret_cast<uintptr_t>(ptr), reinterpret_cast<uintptr_t>(stream));
        },
        [engine = engine_, device] {
          c10::cuda::CUDAGuard guard(device);
          engine->empty_cache();
        });
  }

  auto allocator() { return allocator_; }
  auto stats() { return engine_->stats(); }
  bool contains(uint64_t ptr) { return engine_->contains(ptr); }

 private:
  int device_;
  std::shared_ptr<Engine> engine_;
  std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator_;
};

}  // namespace

void register_vmm_mempool(pybind11::module_ &m) {
  namespace py = pybind11;
  m.def("vmm_mempool_contains", &contains_pointer);
  py::class_<VMMMemPoolBackend>(m, "VMMMemPoolBackend")
      .def(py::init<int, int, size_t>(), py::arg("device"), py::arg("domain") = -1,
           py::arg("block_bytes") = 0)
      .def_property_readonly("allocator", &VMMMemPoolBackend::allocator)
      .def("stats", &VMMMemPoolBackend::stats)
      .def("contains", &VMMMemPoolBackend::contains);
}
}  // namespace transformer_engine::pytorch
#else
namespace transformer_engine::pytorch {
void register_vmm_mempool(pybind11::module_ &) {}
}  // namespace transformer_engine::pytorch
#endif
