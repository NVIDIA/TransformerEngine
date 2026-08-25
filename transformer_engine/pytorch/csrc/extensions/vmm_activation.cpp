/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../common.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace transformer_engine::pytorch {
namespace {

std::string musa_error_text(MUresult result) {
  const char *name = nullptr;
  const char *message = nullptr;
  muGetErrorName(result, &name);
  muGetErrorString(result, &message);
  std::ostringstream out;
  out << (name ? name : "MUSA_ERROR_UNKNOWN") << " (" << static_cast<int>(result) << ")";
  if (message != nullptr) out << ": " << message;
  return out.str();
}

void check_musa(MUresult result, const char *operation) {
  if (result != MUSA_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed: " + musa_error_text(result));
  }
}

MUmemAllocationProp allocation_properties(int device) {
  MUmemAllocationProp properties{};
  properties.type = MU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = device;
  properties.requestedHandleTypes = MU_MEM_HANDLE_TYPE_NONE;
  return properties;
}

size_t round_up(size_t value, size_t alignment) {
  if (value == 0 || alignment == 0 ||
      value > std::numeric_limits<size_t>::max() - alignment + 1) {
    throw std::runtime_error("invalid VMM allocation size or alignment");
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

class VMMActivationSlot {
 public:
  VMMActivationSlot(size_t requested_bytes, int device)
      : device_(device), requested_bytes_(requested_bytes) {
    check_musa(muInit(0), "muInit");
    check_musa(muDeviceGet(&musa_device_, device_), "muDeviceGet");
    int supported = 0;
    check_musa(muDeviceGetAttribute(&supported,
                                    MU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                    musa_device_),
               "muDeviceGetAttribute(VMM_SUPPORTED)");
    if (!supported) throw std::runtime_error("MUSA device does not support VMM");
    const auto properties = allocation_properties(device_);
    check_musa(muMemGetAllocationGranularity(&granularity_, &properties,
                                             MU_MEM_ALLOC_GRANULARITY_MINIMUM),
               "muMemGetAllocationGranularity");
    bytes_ = round_up(requested_bytes_, granularity_);
    check_musa(muMemAddressReserve(&address_, bytes_, 0, 0, 0), "muMemAddressReserve");
    reserved_ = true;
    try {
      map_new_physical_allocation();
    } catch (...) {
      muMemAddressFree(address_, bytes_);
      reserved_ = false;
      throw;
    }
  }

  ~VMMActivationSlot() { close_noexcept(); }
  VMMActivationSlot(const VMMActivationSlot &) = delete;
  VMMActivationSlot &operator=(const VMMActivationSlot &) = delete;

  at::Tensor tensor(const std::vector<int64_t> &sizes, const std::vector<int64_t> &strides,
                    at::ScalarType dtype) {
    ensure_mapped();
    if (sizes.empty() || sizes.size() != strides.size()) {
      throw std::runtime_error("VMM tensor sizes and strides must have equal nonzero rank");
    }
    size_t maximum_element_offset = 0;
    for (size_t index = 0; index < sizes.size(); ++index) {
      if (sizes[index] <= 0 || strides[index] < 0) {
        throw std::runtime_error("VMM tensor dimensions must be positive and strides non-negative");
      }
      maximum_element_offset += static_cast<size_t>(sizes[index] - 1) *
                                static_cast<size_t>(strides[index]);
    }
    const size_t required = (maximum_element_offset + 1) * c10::elementSize(dtype);
    if (required > requested_bytes_) {
      throw std::runtime_error("requested tensor view exceeds VMM activation slot");
    }
    auto options = at::TensorOptions().dtype(dtype).device(
        c10::Device(c10::DeviceType::PrivateUse1, device_));
    return at::from_blob(reinterpret_cast<void *>(static_cast<uintptr_t>(address_)), sizes,
                         strides, [](void *) {}, options);
  }

  void unmap_and_release() {
    ensure_reserved();
    if (!mapped_) throw std::runtime_error("VMM activation slot is already unmapped");
    check_musa(muMemUnmap(address_, bytes_), "muMemUnmap");
    mapped_ = false;
    check_musa(muMemRelease(handle_), "muMemRelease");
    handle_ = 0;
  }

  void create_and_remap() {
    ensure_reserved();
    if (mapped_) throw std::runtime_error("VMM activation slot is already mapped");
    map_new_physical_allocation();
  }

  py::dict info() const {
    py::dict result;
    result["device"] = device_;
    result["requested_bytes"] = requested_bytes_;
    result["aligned_bytes"] = bytes_;
    result["granularity"] = granularity_;
    result["address"] = static_cast<uint64_t>(address_);
    result["reserved"] = reserved_;
    result["mapped"] = mapped_;
    result["handle"] = static_cast<uint64_t>(handle_);
    return result;
  }

  void close() {
    if (!reserved_) return;
    if (mapped_) {
      check_musa(muMemUnmap(address_, bytes_), "muMemUnmap(close)");
      mapped_ = false;
      check_musa(muMemRelease(handle_), "muMemRelease(close)");
      handle_ = 0;
    }
    check_musa(muMemAddressFree(address_, bytes_), "muMemAddressFree");
    reserved_ = false;
    address_ = 0;
  }

 private:
  void ensure_reserved() const {
    if (!reserved_) throw std::runtime_error("VMM activation slot reservation is closed");
  }
  void ensure_mapped() const {
    ensure_reserved();
    if (!mapped_) throw std::runtime_error("VMM activation slot has no physical mapping");
  }
  void map_new_physical_allocation() {
    const auto properties = allocation_properties(device_);
    check_musa(muMemCreate(&handle_, bytes_, &properties, 0), "muMemCreate");
    bool mapping_created = false;
    try {
      check_musa(muMemMap(address_, bytes_, 0, handle_, 0), "muMemMap");
      mapping_created = true;
      MUmemAccessDesc access{};
      access.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = device_;
      access.flags = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      check_musa(muMemSetAccess(address_, bytes_, &access, 1), "muMemSetAccess");
      mapped_ = true;
    } catch (...) {
      if (mapping_created) muMemUnmap(address_, bytes_);
      muMemRelease(handle_);
      handle_ = 0;
      throw;
    }
  }
  void close_noexcept() noexcept {
    if (!reserved_) return;
    if (mapped_) {
      muMemUnmap(address_, bytes_);
      mapped_ = false;
      muMemRelease(handle_);
      handle_ = 0;
    }
    muMemAddressFree(address_, bytes_);
    reserved_ = false;
    address_ = 0;
  }
  int device_ = 0;
  MUdevice musa_device_ = 0;
  size_t requested_bytes_ = 0;
  size_t bytes_ = 0;
  size_t granularity_ = 0;
  MUdeviceptr address_ = 0;
  MUmemGenericAllocationHandle handle_ = 0;
  bool reserved_ = false;
  bool mapped_ = false;
};

py::dict vmm_driver_memory_info() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check_musa(muMemGetInfo(&free_bytes, &total_bytes), "muMemGetInfo");
  py::dict result;
  result["free_bytes"] = free_bytes;
  result["total_bytes"] = total_bytes;
  return result;
}

}  // namespace

void init_vmm_activation_extension(py::module_ &module) {
  module.def("vmm_driver_memory_info", &vmm_driver_memory_info);
  py::class_<VMMActivationSlot>(module, "VMMActivationSlot")
      .def(py::init<size_t, int>(), py::arg("requested_bytes"), py::arg("device") = 0)
      .def("tensor", &VMMActivationSlot::tensor)
      .def("unmap_and_release", &VMMActivationSlot::unmap_and_release)
      .def("create_and_remap", &VMMActivationSlot::create_and_remap)
      .def("info", &VMMActivationSlot::info)
      .def("close", &VMMActivationSlot::close);
}

}  // namespace transformer_engine::pytorch
