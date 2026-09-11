/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_EXTENSIONS_ATTENTION_CACHE_DEBUG_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_EXTENSIONS_ATTENTION_CACHE_DEBUG_H_

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace transformer_engine {
namespace jax {
namespace attention_cache_debug {
namespace detail {

constexpr size_t kSiteCount = 4;
constexpr size_t kStageCount = 5;
inline constexpr std::array<const char *, kStageCount> kStageNames = {
    "validate", "build_operation_graph", "create_execution_plans", "check_support", "build_plans"};

inline int DebugLevel() {
  static const int level = [] {
    const char *value = std::getenv("NVTE_FUSED_ATTN_CACHE_DEBUG");
    if (value == nullptr || value[0] == '\0' || value[0] == '0') return 0;
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : 1;
  }();
  return level;
}

inline int LauncherRank() {
  static const int rank = [] {
    for (const char *name : {"RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"}) {
      const char *value = std::getenv(name);
      if (value != nullptr && value[0] != '\0') return std::atoi(value);
    }
    return -1;
  }();
  return rank;
}

inline bool Enabled() {
  static const bool enabled = [] {
    if (DebugLevel() < 1) return false;
    const int rank = LauncherRank();
    if (rank < 0) return true;
    const char *value = std::getenv("NVTE_FUSED_ATTN_CACHE_DEBUG");
    const char *separator = value == nullptr ? nullptr : std::strchr(value, ':');
    if (separator == nullptr) return rank == 0;
    const std::string ranks(separator + 1);
    if (ranks == "all") return true;
    for (size_t start = 0; start <= ranks.size();) {
      const size_t end = ranks.find(',', start);
      const std::string token = ranks.substr(start, end - start);
      if (!token.empty() && std::atoi(token.c_str()) == rank) return true;
      if (end == std::string::npos) break;
      start = end + 1;
    }
    return false;
  }();
  return enabled;
}

inline bool TraceEnabled() { return Enabled() && DebugLevel() >= 2; }

inline size_t SiteIndex(std::string_view backend, std::string_view direction) {
  const size_t backend_index = backend == "f16" ? 0 : backend == "fp8" ? 1 : kSiteCount;
  const size_t direction_index = direction == "fwd" ? 0 : direction == "bwd" ? 1 : 2;
  if (backend_index > 1 || direction_index > 1) {
    throw std::invalid_argument("Invalid fused-attention cache diagnostic site: " +
                                std::string(backend) + " " + std::string(direction));
  }
  return backend_index * 2 + direction_index;
}

struct Counters {
  std::atomic<uint64_t> hit{0};
  std::atomic<uint64_t> miss{0};
  std::atomic<uint64_t> create_graph{0};
  std::atomic<uint64_t> cache_graph{0};
  std::atomic<uint64_t> build_plans{0};
  std::atomic<uint64_t> execute{0};
};

struct Timing {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> elapsed_ns{0};
};

inline std::array<Counters, kSiteCount> &AllCounters() {
  static auto *counters = new std::array<Counters, kSiteCount>();
  return *counters;
}

inline std::array<Timing, kSiteCount * kStageCount> &AllTimings() {
  static auto *timings = new std::array<Timing, kSiteCount * kStageCount>();
  return *timings;
}

inline std::mutex &OutputMutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

inline const char *Backend(size_t site) { return site < 2 ? "f16" : "fp8"; }
inline const char *Direction(size_t site) { return site % 2 == 0 ? "fwd" : "bwd"; }

inline std::string RankTag() {
  const int rank = LauncherRank();
  return rank < 0 ? "" : "rank=" + std::to_string(rank) + " | ";
}

inline void Write(const std::string &message) {
  std::lock_guard<std::mutex> lock(OutputMutex());
  std::fwrite(message.data(), 1, message.size(), stderr);
  std::fflush(stderr);
}

inline uint64_t Load(const std::atomic<uint64_t> &value) {
  return value.load(std::memory_order_relaxed);
}

inline std::string CounterLine(size_t site, const char *event = nullptr, int device = -1,
                               bool all_devices = false) {
  const Counters &counters = AllCounters()[site];
  const std::string device_name = all_devices ? "all" : std::to_string(device);
  char line[640];
  std::snprintf(line, sizeof(line),
                "[FUSED-ATTN-CACHE] %sdev=%-3s | %s %s %-12s | hit=%4" PRIu64 ", miss=%4" PRIu64
                ", create_graph=%4" PRIu64 ", cache_graph=%4" PRIu64 ", build_plans=%4" PRIu64
                ", execute=%4" PRIu64 "\n",
                RankTag().c_str(), device_name.c_str(), Backend(site), Direction(site),
                event == nullptr ? "" : event, Load(counters.hit), Load(counters.miss),
                Load(counters.create_graph), Load(counters.cache_graph), Load(counters.build_plans),
                Load(counters.execute));
  return line;
}

inline void PrintSummary() {
  if (!Enabled()) return;
  const std::string marker = "[FUSED-ATTN-CACHE] " + RankTag() + "===== summary ";
  std::string output = marker + "begin =====\n";
  for (size_t site = 0; site < kSiteCount; ++site) {
    const Counters &counters = AllCounters()[site];
    if ((Load(counters.hit) | Load(counters.miss) | Load(counters.create_graph) |
         Load(counters.cache_graph) | Load(counters.build_plans) | Load(counters.execute)) != 0) {
      output += CounterLine(site, nullptr, -1, true);
    }
  }
  for (size_t site = 0; site < kSiteCount; ++site) {
    for (size_t stage = 0; stage < kStageCount; ++stage) {
      const Timing &timing = AllTimings()[site * kStageCount + stage];
      const uint64_t calls = Load(timing.calls);
      if (calls == 0) continue;
      const double milliseconds = static_cast<double>(Load(timing.elapsed_ns)) / calls / 1e6;
      char line[320];
      std::snprintf(line, sizeof(line),
                    "[FUSED-ATTN-CACHE] %s%s %-3s %-22s | calls=%" PRIu64 " | time=%9.3f ms/call\n",
                    RankTag().c_str(), Backend(site), Direction(site), kStageNames[stage], calls,
                    milliseconds);
      output += line;
    }
  }
  output += marker + "end =====\n";
  Write(output);
}

inline void RegisterSummary() {
  static const bool registered = [] {
    std::atexit(PrintSummary);
    return true;
  }();
  (void)registered;
}

inline std::atomic<uint64_t> *EventCounter(Counters &counters, std::string_view event) {
  if (event == "hit") return &counters.hit;
  if (event == "miss") return &counters.miss;
  if (event == "create_graph") return &counters.create_graph;
  if (event == "cache_graph") return &counters.cache_graph;
  if (event == "plans_built") return &counters.build_plans;
  if (event == "execute") return &counters.execute;
  return nullptr;
}

inline size_t StageIndex(std::string_view event) {
  for (size_t index = 0; index < kStageCount; ++index) {
    if (event == kStageNames[index]) return index;
  }
  return kStageCount;
}

}  // namespace detail

inline void Record(std::string_view backend, std::string_view direction, std::string_view event,
                   int device = -1, std::string_view key = {}, uint64_t elapsed_ns = 0) {
  if (!detail::Enabled()) return;
  detail::RegisterSummary();
  const size_t site = detail::SiteIndex(backend, direction);
  const size_t stage = detail::StageIndex(event);
  if (stage < detail::kStageCount) {
    detail::Timing &timing = detail::AllTimings()[site * detail::kStageCount + stage];
    timing.calls.fetch_add(1, std::memory_order_relaxed);
    timing.elapsed_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    return;
  }

  detail::Counters &counters = detail::AllCounters()[site];
  std::atomic<uint64_t> *counter = detail::EventCounter(counters, event);
  if (counter == nullptr) {
    throw std::invalid_argument("Invalid fused-attention cache diagnostic event: " +
                                std::string(event));
  }
  counter->fetch_add(1, std::memory_order_relaxed);
  if (!detail::TraceEnabled()) return;
  if ((event == "hit" || event == "miss") && !key.empty()) {
    detail::Write("[FUSED-ATTN-CACHE] " + detail::RankTag() + "dev=" + std::to_string(device) +
                  " | " + std::string(backend) + " " + std::string(direction) + " " +
                  std::string(event) + " | " + std::string(key) + "\n");
  } else {
    std::string uppercase(event == "plans_built" ? "build_plans" : event);
    for (char &character : uppercase) {
      if (character >= 'a' && character <= 'z') character -= 'a' - 'A';
    }
    detail::Write(detail::CounterLine(site, uppercase.c_str(), device));
  }
}

}  // namespace attention_cache_debug
}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_EXTENSIONS_ATTENTION_CACHE_DEBUG_H_
