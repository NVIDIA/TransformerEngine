/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/pytorch/csrc/extensions/vmm_mempool_core.h"

#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <set>

namespace {
struct State {
  uint64_t next_address = 4096, next_handle = 1, next_event = 1;
  bool capturing = false, ready = true;
  int graph = 0;
  std::vector<std::pair<int, std::shared_ptr<std::atomic<bool>>>> pins;
  int fail_map = -1, fail_create = -1;
  bool fail_access = false, fail_event = false;
  std::map<uint64_t, size_t> reservations;
  std::map<uint64_t, int> handles;
  std::map<uint64_t, uint64_t> mappings;
  std::set<uintptr_t> events;
};
struct FakeOps {
  struct OutOfMemory : std::runtime_error {
    OutOfMemory() : std::runtime_error("out of memory") {}
  };
  std::shared_ptr<State> s;
  using CapturePin = std::shared_ptr<std::atomic<bool>>;
  CapturePin pin_capture(uintptr_t) {
    assert(s->capturing);
    auto pin = std::make_shared<std::atomic<bool>>(false);
    s->pins.emplace_back(s->graph, pin);
    return pin;
  }
  bool capture_done(const CapturePin &pin) { return pin->load(); }
  uint64_t reserve(size_t bytes, size_t) {
    auto base = s->next_address;
    s->next_address += bytes + 4096;
    s->reservations.emplace(base, bytes);
    return base;
  }
  void free_address(uint64_t base, size_t bytes) {
    assert(s->reservations.at(base) == bytes);
    assert(s->mappings.lower_bound(base) == s->mappings.lower_bound(base + bytes));
    s->reservations.erase(base);
  }
  uint64_t create(size_t, int domain) {
    if (s->fail_create == 0) { s->fail_create = -1; throw OutOfMemory(); }
    if (s->fail_create > 0) --s->fail_create;
    auto h = s->next_handle++;
    s->handles.emplace(h, domain);
    return h;
  }
  void release(uint64_t handle) {
    for (auto [base, mapped] : s->mappings) assert(mapped != handle);
    assert(s->handles.erase(handle) == 1);
  }
  void map(uint64_t base, size_t, uint64_t handle) {
    if (s->fail_map == 0) { s->fail_map = -1; throw std::runtime_error("map failure"); }
    if (s->fail_map > 0) --s->fail_map;
    assert(s->handles.count(handle));
    for (auto [address, mapped] : s->mappings) assert(mapped != handle);
    assert(s->mappings.emplace(base, handle).second);
  }
  void unmap(uint64_t base, size_t) { assert(s->mappings.erase(base) == 1); }
  void access(uint64_t, size_t) {
    if (s->fail_access) throw std::runtime_error("access failure");
  }
  bool capturing(uintptr_t) { return s->capturing; }
  uintptr_t event_create() {
    if (s->fail_event) throw std::runtime_error("event failure");
    auto e = s->next_event++;
    s->events.insert(e);
    return e;
  }
  void event_record(uintptr_t event, uintptr_t) { assert(s->events.count(event)); }
  bool event_ready(uintptr_t event) {
    assert(!s->capturing);  // Event queries must never run during capture.
    assert(s->events.count(event));
    return s->ready;
  }
  void event_destroy(uintptr_t event) { assert(s->events.erase(event) == 1); }
  void synchronize() { assert(!s->capturing); s->ready = true; }
};
using Pool = transformer_engine::pytorch::VMMPhysicalBlockPool<FakeOps>;

void reuse_and_stream_ordering() {
  auto s = std::make_shared<State>();
  Pool p(FakeOps{s}, 64, -1);
  auto a = p.allocate(256, 1);
  assert(s->handles.at(s->mappings.at(a)) == 0);
  assert(s->handles.at(s->mappings.at(a + 128)) == 1);
  assert(p.contains(a + 255) && !p.contains(a + 256));
  p.free(a, 1);
  s->ready = false;
  auto b = p.allocate(128, 2);
  assert(p.stats().at("physical_blocks_reused") == 0);
  assert(s->mappings.count(a));  // No premature unmap, even on another stream.
  s->ready = true;
  p.collect_completed();
  assert(!s->mappings.count(a));
  assert(p.stats().at("cached_physical_bytes") == 256);
  assert(p.stats().at("physical_blocks_released") == 0);
  auto c = p.allocate(128, 2);
  assert(c != a && p.stats().at("physical_blocks_reused") == 2);
  assert(!s->mappings.count(a));
  p.empty_cache();
  assert(p.contains(b) && p.contains(c));  // Draining preserves live mappings.
  p.free(b, 2);
  p.free(c, 2);
  p.empty_cache();
  assert(s->handles.empty() && s->mappings.empty() && s->reservations.empty());
  assert(s->events.empty());
}

void graph_quarantine() {
  auto s = std::make_shared<State>();
  Pool p(FakeOps{s}, 64, -1);
  auto eager = p.allocate(128, 7);
  s->capturing = true;
  auto a = p.allocate(256, 7);
  p.free(eager, 7);  // Eager allocation freed in capture must also stay mapped.
  p.free(a, 7);
  auto b = p.allocate(128, 7);
  assert(p.stats().at("physical_blocks_reused") == 0);
  s->capturing = false;
  p.free(b, 7);  // Capture allocation freed outside capture still stays mapped.
  auto c = p.allocate(128, 7);
  assert(p.stats().at("captured_retired_bytes") == 512);
  assert(s->mappings.count(a) && s->mappings.count(b));
  // Even an early drain cannot unmap memory held by a CUDA graph user object.
  p.empty_cache();
  assert(s->mappings.count(a) && s->mappings.count(b));
  for (const auto &[graph, pin] : s->pins) pin->store(true);
  p.empty_cache();
  assert(!s->mappings.count(a) && !s->mappings.count(b));
  p.free(c, 7);
  p.empty_cache();
  assert(s->handles.empty() && s->events.empty());
}

void independent_graph_lifetimes() {
  auto s = std::make_shared<State>();
  Pool p(FakeOps{s}, 64, -1);
  s->capturing = true;
  auto a = p.allocate(128, 1);
  p.free(a, 1);
  s->graph = 1;
  auto b = p.allocate(128, 2);
  p.free(b, 2);
  s->capturing = false;
  for (const auto &[graph, pin] : s->pins) if (graph == 0) pin->store(true);
  p.empty_cache();
  assert(!s->mappings.count(a) && s->mappings.count(b));
  for (const auto &[graph, pin] : s->pins) pin->store(true);
  p.empty_cache();
  assert(s->handles.empty() && s->reservations.empty());
}

void sparse_native_segment() {
  auto s = std::make_shared<State>();
  Pool p(FakeOps{s}, 64, -1);
  auto a = p.allocate(130, 1, 1280);
  assert(s->reservations.at(a) == 1280);
  assert(s->mappings.size() == 4);  // Four pages, not twenty pages of physical RAM.
  assert(p.stats().at("physical_bytes") == 256);
  assert(p.stats().at("virtual_reserved_bytes") == 1280);
  p.free(a, 1);
  auto b = p.allocate(128, 2, 1024);
  assert(!s->reservations.count(a));
  assert(p.stats().at("physical_blocks_reused") == 2);
  p.free(b, 2);
  p.empty_cache();
  assert(s->handles.empty() && s->reservations.empty());
  bool threw = false;
  try { p.allocate(256, 1, 128); }
  catch (const std::invalid_argument &) { threw = true; }
  assert(threw && s->reservations.empty());
  s->fail_map = 2;
  threw = false;
  try { p.allocate(256, 1, 1280); }
  catch (const std::runtime_error &) { threw = true; }
  assert(threw && s->reservations.empty() && s->mappings.empty());
  p.empty_cache();
  assert(s->handles.empty());
}

void rollback() {
  for (int fail = 0; fail < 6; ++fail) {
    auto s = std::make_shared<State>();
    Pool p(FakeOps{s}, 64, -1);
    if (fail < 4) s->fail_map = fail;
    if (fail == 4) s->fail_access = true;
    if (fail == 5) s->fail_event = true;
    bool threw = false;
    try { p.allocate(256, 1); } catch (const std::runtime_error &) { threw = true; }
    assert(threw && s->mappings.empty() && s->reservations.empty() && s->events.empty());
    s->fail_access = s->fail_event = false;
    auto a = p.allocate(256, 1);
    p.free(a, 1);
    p.empty_cache();
    assert(s->handles.empty());
  }
}

void rounding_and_single_domain() {
  for (int domain : {-1, 0, 1}) {
    auto s = std::make_shared<State>();
    Pool p(FakeOps{s}, 64, domain);
    auto a = p.allocate(130, 1);
    assert(s->reservations.at(a) == (domain == -1 ? 256 : 192));
    if (domain != -1) {
      for (auto [h, ld] : s->handles) assert(ld == domain);
    }
    p.free(a, 1);
    p.empty_cache();
    bool threw = false;
    try { p.allocate(std::numeric_limits<size_t>::max(), 1); }
    catch (const std::overflow_error &) { threw = true; }
    assert(threw && s->reservations.empty());
  }
}

void oom_reclaims_unmapped_cache() {
  auto s = std::make_shared<State>();
  Pool p(FakeOps{s}, 64, -1);
  auto a = p.allocate(256, 1);
  p.free(a, 1);
  // Four pages are reclaimed; the next request needs two additional pages.
  s->fail_create = 0;
  auto b = p.allocate(384, 1);
  assert(p.stats().at("physical_blocks_released") > 0);
  p.free(b, 1);
  p.empty_cache();
  assert(s->handles.empty());
}
}  // namespace

int main() {
  reuse_and_stream_ordering();
  graph_quarantine();
  independent_graph_lifetimes();
  sparse_native_segment();
  rollback();
  rounding_and_single_domain();
  oom_reclaims_unmapped_cache();
  std::cout << "VMM physical-block pool: all policy tests passed\n";
}
