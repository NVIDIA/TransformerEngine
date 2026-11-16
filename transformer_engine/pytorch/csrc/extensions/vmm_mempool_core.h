/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace transformer_engine::pytorch {

// Ops owns the device API; this policy is also tested with a simulated driver.
// Every physical handle is one page. Virtual mappings are NEVER cached for reuse.
// Captured mappings are quarantined until the owning MemPool permits emptyCache.
template <class Ops>
class VMMPhysicalBlockPool {
 public:
  using Address = uint64_t;
  using Handle = uint64_t;
  using Stream = uintptr_t;
  using Event = uintptr_t;

  struct Mapping {
    size_t requested = 0;
    size_t half = 0;
    Stream stream = 0;
    Event event = 0;
    bool captured = false;
    std::vector<std::pair<int, Handle>> pages;
  };

  VMMPhysicalBlockPool(Ops ops, size_t page_bytes, int domain)
      : ops_(std::move(ops)), page_(page_bytes), domain_(domain) {
    if (!page_ || domain < -1 || domain > 1) throw std::invalid_argument("Invalid VMM pool layout");
  }

  Address allocate(size_t size, Stream stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool capturing = ops_.capturing(stream);
    if (!capturing) collect();
    const size_t partitions = domain_ == -1 ? 2 : 1;
    if (size > std::numeric_limits<size_t>::max() - (partitions - 1)) {
      throw std::overflow_error("VMM allocation size overflow");
    }
    const size_t half = round_up((size + partitions - 1) / partitions);
    if (half > std::numeric_limits<size_t>::max() / partitions) {
      throw std::overflow_error("VMM mapping size overflow");
    }
    Mapping mapping;
    mapping.requested = size;
    mapping.half = half;
    mapping.stream = stream;
    mapping.captured = capturing;
    const size_t count = half * partitions / page_;
    mapping.pages.reserve(count);
    // Reserve cache capacity before acquiring any resources (including rollback).
    for (auto &cache : cache_) cache.reserve(cache.size() + count);
    const auto base = ops_.reserve(half * partitions, page_);
    size_t mapped = 0;
    try {
      if (!capturing) mapping.event = ops_.event_create();
      for (size_t i = 0; i < count; ++i) {
        const int domain = domain_ == -1 ? static_cast<int>(i * page_ / half) : domain_;
        auto &cache = cache_[domain];
        Handle handle;
        if (cache.empty()) {
          try {
            handle = ops_.create(page_, domain);
          } catch (const typename Ops::OutOfMemory &) {
            // Release only unmapped cached backing. Captured and live mappings
            // are never evicted, and callbacks never call torch.empty_cache().
            release_cached_handles();
            handle = ops_.create(page_, domain);
          }
          ++created_;
        } else {
          handle = cache.back();
          cache.pop_back();
          ++reused_;
        }
        mapping.pages.emplace_back(domain, handle);
        ops_.map(base + i * page_, page_, handle);
        ++mapped;
        ++maps_;
      }
      ops_.access(base, half * partitions);
      active_.emplace(base, std::move(mapping));
    } catch (...) {
      // Do not return a handle to the cache until every alias has been unmapped.
      for (size_t i = 0; i < mapped; ++i) ops_.unmap(base + i * page_, page_);
      for (const auto &[domain, handle] : mapping.pages) cache_[domain].push_back(handle);
      if (mapping.event) ops_.event_destroy(mapping.event);
      ops_.free_address(base, half * partitions);
      throw;
    }
    ++allocations_;
    return base;
  }

  void free(Address base, Stream stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_.find(base);
    if (it == active_.end() || it->second.stream != stream) {
      throw std::runtime_error("Unknown VMM allocation or allocation stream mismatch");
    }
    auto &mapping = it->second;
    mapping.captured = mapping.captured || ops_.capturing(stream);
    // PyTorch waits for recorded streams. Its FREE_COMPLETED does not imply
    // completion on the allocation stream: remapping needs this additional event.
    if (!mapping.captured) ops_.event_record(mapping.event, stream);
    retired_.insert(active_.extract(it));
    ++frees_;
  }

  // ONLY call through PyTorch's managed-pool emptyCache contract: no capturing
  // graph may retain this pool. Live Tensor/Storage allocations remain mapped.
  void empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    ops_.synchronize();
    for (auto it = retired_.begin(); it != retired_.end();) reclaim(it++);
    release_cached_handles();
  }

  bool contains(Address pointer) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_.upper_bound(pointer);
    if (it == active_.begin()) return false;
    --it;
    return pointer - it->first < it->second.requested;
  }

  std::map<std::string, uint64_t> stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t active = 0, pending = 0, captured = 0;
    for (const auto &[base, mapping] : active_) active += mapping.pages.size() * page_;
    for (const auto &[base, mapping] : retired_) {
      (mapping.captured ? captured : pending) += mapping.pages.size() * page_;
    }
    const uint64_t cached = (cache_[0].size() + cache_[1].size()) * page_;
    return {{"block_bytes", page_}, {"active_mapped_bytes", active},
            {"pending_mapped_bytes", pending}, {"captured_retired_bytes", captured},
            {"cached_physical_bytes", cached},
            {"physical_bytes", active + pending + captured + cached},
            {"allocation_requests", allocations_}, {"free_requests", frees_},
            {"physical_blocks_created", created_}, {"physical_blocks_reused", reused_},
            {"physical_blocks_released", released_}, {"map_calls", maps_},
            {"unmap_calls", unmaps_}};
  }

 private:
  using Mappings = std::map<Address, Mapping>;

  size_t round_up(size_t size) const {
    if (!size || size > std::numeric_limits<size_t>::max() - page_ + 1) {
      throw std::overflow_error("Invalid VMM allocation size");
    }
    return (size + page_ - 1) / page_ * page_;
  }

  void release_cached_handles() {
    for (auto &cache : cache_) {
      while (!cache.empty()) {
        ops_.release(cache.back());
        cache.pop_back();
        ++released_;
      }
    }
  }

  void collect() {
    for (auto it = retired_.begin(); it != retired_.end();) {
      auto current = it++;
      if (!current->second.captured && ops_.event_ready(current->second.event)) reclaim(current);
    }
  }

  void reclaim(typename Mappings::iterator it) {
    auto &mapping = it->second;
    // Allocate bookkeeping before changing any CUDA resources.
    for (auto &cache : cache_) cache.reserve(cache.size() + mapping.pages.size());
    for (size_t i = 0; i < mapping.pages.size(); ++i) {
      ops_.unmap(it->first + i * page_, page_);
      ++unmaps_;
    }
    for (const auto &[domain, handle] : mapping.pages) cache_[domain].push_back(handle);
    if (mapping.event) ops_.event_destroy(mapping.event);
    ops_.free_address(it->first, mapping.pages.size() * page_);
    retired_.erase(it);
  }

  Ops ops_;
  size_t page_;
  int domain_;
  std::mutex mutex_;
  Mappings active_, retired_;
  std::array<std::vector<Handle>, 2> cache_;
  uint64_t allocations_ = 0, frees_ = 0, created_ = 0, reused_ = 0, released_ = 0;
  uint64_t maps_ = 0, unmaps_ = 0;
};

}  // namespace transformer_engine::pytorch
