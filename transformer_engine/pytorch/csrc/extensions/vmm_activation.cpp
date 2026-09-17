/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../common.h"

#include <ATen/record_function.h>
#include <ATen/ThreadLocalState.h>

#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <vector>

#include <musa.h>
#include <musa_runtime.h>

namespace py = pybind11;

namespace transformer_engine::pytorch {
namespace {

bool vmm_remap_debug_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("MEGATRON_VMM_REMAP_DEBUG");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

using VmmClock = std::chrono::steady_clock;

std::mutex vmm_trace_mutex;
std::vector<std::string> vmm_trace_records;
std::atomic<bool> vmm_trace_enabled{false};
std::string vmm_trace_path;

void vmm_enable_trace(bool enabled, const std::string &path) {
  std::lock_guard<std::mutex> lock(vmm_trace_mutex);
  vmm_trace_enabled.store(enabled, std::memory_order_release);
  vmm_trace_path = path;
  if (!enabled) vmm_trace_records.clear();
}

void vmm_reset_trace() {
  std::lock_guard<std::mutex> lock(vmm_trace_mutex);
  vmm_trace_records.clear();
}

void vmm_dump_trace() {
  std::lock_guard<std::mutex> lock(vmm_trace_mutex);
  if (vmm_trace_path.empty()) throw std::runtime_error("VMM trace path is empty");
  std::ofstream out(vmm_trace_path, std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open VMM trace path: " + vmm_trace_path);
  for (const auto &record : vmm_trace_records) out << record << '\\n';
  out.flush();
}

void vmm_remap_trace(const char *message, double elapsed_ms = -1.0) {
  const bool trace_enabled = vmm_trace_enabled.load(std::memory_order_acquire);
  if (!trace_enabled && !vmm_remap_debug_enabled()) return;
  const auto thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
  if (elapsed_ms >= 0.0) {
    std::ostringstream record;
    record << "{\"ts_ns\":"
           << std::chrono::duration_cast<std::chrono::nanoseconds>(VmmClock::now().time_since_epoch()).count()
           << ",\"thread\":" << thread_id << ",\"message\":\"" << message
           << "\",\"elapsed_ms\":" << elapsed_ms << "}";
    if (vmm_trace_enabled.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(vmm_trace_mutex);
      vmm_trace_records.push_back(record.str());
    }
    std::fprintf(stdout, "[vmm-remap tid=%zu] %s elapsed_ms=%.3f\n", thread_id, message,
                 elapsed_ms);
  } else {
    std::fprintf(stdout, "[vmm-remap tid=%zu] %s\n", thread_id, message);
  }
  std::fflush(stdout);
}

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

// Runtime-API counterpart: musaLaunchHostFunc returns musaError_t, not MUresult.
std::string musa_runtime_error_text(musaError_t error) {
  const char *name = musaGetErrorName(error);
  const char *message = musaGetErrorString(error);
  std::ostringstream out;
  out << (name ? name : "musaErrorUnknown") << " (" << static_cast<int>(error) << ")";
  if (message != nullptr) out << ": " << message;
  return out.str();
}

void check_musa_runtime(musaError_t error, const char *operation) {
  if (error != MUSA_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             musa_runtime_error_text(error));
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
    throw std::runtime_error("invalid value/alignment for round-up");
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

// ---------------------------------------------------------------------------
// Asynchronous release infrastructure
//
// MUSA host functions (musaLaunchHostFunc) may NOT call driver APIs from the
// callback thread: muMemUnmap/muMemRelease deadlocked there in our probe
// (/tmp/test_vmm_hostfn2.c, 2026-08-28). The supported shape is therefore:
//
//   D2H burst completes
//     -> host func fires: move the shared context into a queue, notify CV
//        (fast, no MUSA calls, callback returns immediately so the stream
//        unblocks)
//     -> resident release worker wakes, performs muMemUnmap + muMemRelease,
//        publishes error status and the completion flag
//   backward side reads the completion flag (no event synchronize at all)
//
// Ownership: the queue, the host-func user data and each slot's `pending_`
// all hold shared_ptr copies of the context, so the context outlives every
// participant regardless of Python GC or slot destruction timing.
// ---------------------------------------------------------------------------

struct ReleaseRequest {
  std::string slot_id;
  MUdeviceptr address;
  size_t bytes;
  MUmemGenericAllocationHandle handle;
};

struct ReleaseHookContext {
  std::vector<ReleaseRequest> requests;
  std::shared_ptr<at::ThreadLocalState> tls_state;
  std::atomic<int> worker_has_callbacks{0};
  std::atomic<int> done{0};          // 1 once unmap+release finished (or failed)
  std::atomic<int> error{0};         // 1 if any muMemUnmap/muMemRelease failed
  std::atomic<int> error_code{0};    // first failing MUresult
  std::atomic<int> callback_fired{0};
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
};

struct RemapReleaseNotifier {
  std::mutex mutex;
  std::condition_variable cv;
  uint64_t generation{0};
};

RemapReleaseNotifier &remap_release_notifier() {
  static RemapReleaseNotifier notifier;
  return notifier;
}

void notify_remap_worker_release_ready();

// Serial-driver mode (VMM_SERIAL_DRIVER_WORKERS=1): unmap/release and
// create/map/setAccess run on one resident thread instead of two concurrent
// ones.  Two driver workers contending on the same page-table lock produced
// ~40ms muMemSetAccess tails when a remap landed while a release batch was
// mid-unmap; serializing the driver calls removes that interleaving.
// The Python setter (vmm_set_serial_driver_workers) overrides the env var;
// -1 = unset.  The mode must be fixed before the first remap enqueue.
std::atomic<int> vmm_serial_driver_override{-1};
bool vmm_serial_driver_workers() {
  const int override_value = vmm_serial_driver_override.load(std::memory_order_acquire);
  if (override_value >= 0) return override_value == 1;
  static const bool enabled = [] {
    const char *env = std::getenv("VMM_SERIAL_DRIVER_WORKERS");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}
struct RemapRequest {
  std::string slot_id;
  MUdeviceptr address;
  size_t bytes;
  size_t copy_bytes{0};
  const void *host_address{nullptr};
  int device;
  std::shared_ptr<ReleaseHookContext> release_dependency;
  MUmemGenericAllocationHandle handle{0};
  bool mapped{false};
  bool copy_submitted{false};
  size_t slot_index{0};
  musaEvent_t done_event{nullptr};
  std::shared_ptr<std::atomic<int>> event_recorded;
  // Set with acquire/release semantics when ownership of the mapping moves to
  // the consumer slot. Cleanup must never unmap or release an adopted mapping.
  std::shared_ptr<std::atomic<int>> adopted;
  // Explicit graph-consumption gate.  A request must be submitted by the
  // consumer before the worker may remap or enqueue its H2D.
  std::shared_ptr<std::atomic<int>> submitted;
  // Deferred-H2D mode: set once the worker finished the VMM transition.  The
  // H2D memcpy is then launched by remap_slot_launch_h2d at replay time.
  std::shared_ptr<std::atomic<int>> remapped{std::make_shared<std::atomic<int>>(0)};
  // Per-slot completion synchronization; batch context lock is not used for waits.
  std::shared_ptr<std::mutex> state_mutex{std::make_shared<std::mutex>()};
  std::shared_ptr<std::condition_variable> state_cv{
      std::make_shared<std::condition_variable>()};
};

struct RemapHookContext {
  std::vector<RemapRequest> requests;
  std::shared_ptr<at::ThreadLocalState> tls_state;
  std::atomic<int> worker_has_callbacks{0};
  // Keep pinned CPU storage alive until every H2D copy submitted by the worker
  // has completed. ReloadCleanupWorker retains this context through that point.
  std::vector<at::Tensor> host_tensors;
  musaStream_t copy_stream{nullptr};
  musaEvent_t copy_done_event{nullptr};
  std::vector<musaEvent_t> slot_done_events;
  bool copy_enabled{false};
  // Deferred-H2D mode: the worker only performs the VMM transition; H2D is
  // launched at replay time via remap_slot_launch_h2d.
  bool defer_copy{false};
  std::atomic<size_t> h2d_launched{0};
  std::atomic<int> done{0};  // remap (and optional H2D/event submission) finished
  std::atomic<int> error{0};
  std::atomic<int> error_code{0};
  std::atomic<int> callback_fired{0};
  std::atomic<size_t> releases_ready{0};
  std::atomic<size_t> remaps_done{0};
  std::atomic<size_t> copies_submitted{0};
  std::mutex completion_mutex;
  std::condition_variable completion_cv;

  ~RemapHookContext() {
    for (auto event : slot_done_events) {
      if (event != nullptr) musaEventDestroy(event);
    }
    if (copy_done_event != nullptr) musaEventDestroy(copy_done_event);
  }
};

// Retain each reload context until its H2D event completes on an ordinary
// resident thread. In particular, do not release the final shared_ptr from a
// musaLaunchHostFunc callback: RemapHookContext destruction calls
// musaEventDestroy, and MUSA runtime/driver APIs can deadlock on that callback
// thread.
class ReloadCleanupWorker {
 public:
  static ReloadCleanupWorker &instance() {
    static ReloadCleanupWorker worker;
    return worker;
  }

  // Called only after the copy stream reaches the lifetime host callback. Move
  // the reference into this queue without invoking any MUSA API or destructing
  // the event-owning context on the callback thread.
  void enqueue_from_callback(std::shared_ptr<RemapHookContext> context) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        // Process teardown is already in progress. Intentionally retain the
        // context rather than destroying its event on the callback thread.
        new std::shared_ptr<RemapHookContext>(std::move(context));
        return;
      }
      queue_.push_back(std::move(context));
    }
    cv_.notify_one();
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }

 private:
  ReloadCleanupWorker() { thread_ = std::thread(&ReloadCleanupWorker::run, this); }
  ~ReloadCleanupWorker() { shutdown(); }
  ReloadCleanupWorker(const ReloadCleanupWorker &) = delete;
  ReloadCleanupWorker &operator=(const ReloadCleanupWorker &) = delete;

  void run() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      while (queue_.empty() && !shutdown_) cv_.wait(lock);
      if (queue_.empty() && shutdown_) return;
      auto context = std::move(queue_.front());
      queue_.pop_front();
      lock.unlock();
      // The callback reached this point only after H2D completion. Destruction
      // and musaEventDestroy now run on this ordinary worker thread.
      context.reset();
      lock.lock();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<RemapHookContext>> queue_;
  bool shutdown_ = false;
  std::thread thread_;
};

void reload_lifetime_host_func(void *user_data) {
  auto *boxed = static_cast<std::shared_ptr<RemapHookContext> *>(user_data);
  std::shared_ptr<RemapHookContext> context(std::move(*boxed));
  delete boxed;
  ReloadCleanupWorker::instance().enqueue_from_callback(std::move(context));
  // No MUSA API and no RemapHookContext destruction on this callback thread.
}

template <typename Context>
void complete_context(Context &context) {
  {
    std::lock_guard<std::mutex> lock(context.completion_mutex);
    context.done.store(1, std::memory_order_release);
  }
  context.completion_cv.notify_all();
  notify_remap_worker_release_ready();
}

template <typename Context>
void wait_for_context(const std::shared_ptr<Context> &context) {
  if (context->done.load(std::memory_order_acquire) == 1) return;
  std::unique_lock<std::mutex> lock(context->completion_mutex);
  context->completion_cv.wait(lock, [&] {
    return context->done.load(std::memory_order_acquire) == 1;
  });
}

template <typename Context>
bool wait_for_context(const std::shared_ptr<Context> &context,
                      std::chrono::milliseconds timeout) {
  if (context->done.load(std::memory_order_acquire) == 1) return true;
  std::unique_lock<std::mutex> lock(context->completion_mutex);
  return context->completion_cv.wait_for(lock, timeout, [&] {
    return context->done.load(std::memory_order_acquire) == 1;
  });
}

// Sentinel error_code used when the worker thread is already gone (process
// teardown); waiters wake with an error and fall back to synchronous unmap.
constexpr int kWorkerUnavailable = 9999;

class ReleaseWorker {
 public:
  static ReleaseWorker &instance() {
    // Function-local static: thread-safe init; the constructor starts the
    // resident thread, so the first call to instance() guarantees a running
    // worker.
    static ReleaseWorker worker;
    return worker;
  }

  // Serial-driver mode entry point: a remap batch joins the same FIFO as
  // release batches so one thread performs every driver call.  The batch is
  // requeued (never blocked on) when its release dependency is still pending.
  void enqueue_remap(std::shared_ptr<RemapHookContext> context);

  // Called from the host-func callback thread. Must not block beyond the
  // queue push and must not call any MUSA API.
  void enqueue_from_callback(std::shared_ptr<ReleaseHookContext> context) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        // Worker is gone; wake waiters with an error instead of hanging. The
        // driver calls must not run on this callback thread.
        context->error_code.store(kWorkerUnavailable, std::memory_order_relaxed);
        context->error.store(1, std::memory_order_relaxed);
        complete_context(*context);
        return;
      }
      queue_.push_back(WorkItem{std::move(context)});
    }
    cv_.notify_one();
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }

  // Wake a serial worker that requeued a remap batch waiting on a release
  // dependency; that dependency may have completed on this very thread.
  void poke() { cv_.notify_all(); }

 private:
  ReleaseWorker() { thread_ = std::thread(&ReleaseWorker::run, this); }
  ~ReleaseWorker() { shutdown(); }
  ReleaseWorker(const ReleaseWorker &) = delete;
  ReleaseWorker &operator=(const ReleaseWorker &) = delete;

  // FIFO entry: a release batch, a remap batch, or a requeued remap batch.
  struct WorkItem {
    std::shared_ptr<ReleaseHookContext> release;
    std::shared_ptr<RemapHookContext> remap;
    bool requeued{false};
  };

  // Returns true when the batch finished (all slots terminal or failed);
  // false means a release dependency is still pending and the serial worker
  // must requeue it instead of blocking the only driver thread.
  bool process_remap_batch(const std::shared_ptr<RemapHookContext> &context, bool requeued);

  // Driver calls run only on the dedicated worker thread, never on the
  // host-func callback thread.
  static void process_release_batch(const std::shared_ptr<ReleaseHookContext> &context) {
    RECORD_USER_SCOPE("vmm::ReleaseWorker");
    std::unique_ptr<at::ThreadLocalStateGuard> tls_guard;
    if (context->tls_state) tls_guard = std::make_unique<at::ThreadLocalStateGuard>(*context->tls_state);
    context->worker_has_callbacks.store(at::hasCallbacks(), std::memory_order_release);
    auto &ctx = *context;
    auto trace_stage = [](const char *stage, const auto &stage_started) {
      const double elapsed = std::chrono::duration<double, std::milli>(
          VmmClock::now() - stage_started).count();
      vmm_remap_trace(stage, elapsed);
    };
    vmm_remap_trace("release worker start (batch)");
    for (size_t index = 0; index < ctx.requests.size(); ++index) {
      const auto &request = ctx.requests[index];
      char label[64];
      const auto request_started = VmmClock::now();
      std::snprintf(label, sizeof(label), "release[%zu] id=%s begin address=0x%llx bytes=%zu handle=0x%llx", index,
                    request.slot_id.c_str(),
                    static_cast<unsigned long long>(request.address), request.bytes,
                    static_cast<unsigned long long>(request.handle));
      vmm_remap_trace(label);
      std::snprintf(label, sizeof(label), "release[%zu] muMemUnmap begin", index);
      const auto unmap_started = VmmClock::now();
      MUresult unmap_rc;
      {
        RECORD_USER_SCOPE("vmm::ReleaseWorker.muMemUnmap");
        unmap_rc = muMemUnmap(request.address, request.bytes);
      }
      trace_stage("release muMemUnmap", unmap_started);
      if (unmap_rc != MUSA_SUCCESS && !ctx.error.load(std::memory_order_relaxed)) {
        ctx.error_code.store(static_cast<int>(unmap_rc), std::memory_order_relaxed);
        ctx.error.store(1, std::memory_order_relaxed);
      }
      const auto release_started = VmmClock::now();
      std::snprintf(label, sizeof(label), "release[%zu] muMemRelease begin", index);
      vmm_remap_trace(label);
      MUresult release_rc;
      {
        RECORD_USER_SCOPE("vmm::ReleaseWorker.muMemRelease");
        release_rc = muMemRelease(request.handle);
      }
      std::snprintf(label, sizeof(label), "release[%zu] muMemRelease", index);
      trace_stage(label, release_started);
      if (release_rc != MUSA_SUCCESS && !ctx.error.load(std::memory_order_relaxed)) {
        ctx.error_code.store(static_cast<int>(release_rc), std::memory_order_relaxed);
        ctx.error.store(1, std::memory_order_relaxed);
      }
      std::snprintf(label, sizeof(label), "release[%zu] request total", index);
      trace_stage(label, request_started);
    }
    vmm_remap_trace("release worker batch done");
    complete_context(ctx);
  }

  void run() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      while (queue_.empty() && !shutdown_) cv_.wait(lock);
      // Drain everything already queued even when shutting down.
      if (queue_.empty() && shutdown_) return;
      auto item = std::move(queue_.front());
      queue_.pop_front();
      lock.unlock();
      if (item.release) {
        process_release_batch(item.release);
        // A release just completed: a requeued serial remap batch waiting on
        // it can proceed now, and this notify is what wakes the worker loop.
        poke();
        lock.lock();
      } else {
        const bool terminal = process_remap_batch(item.remap, item.requeued);
        lock.lock();
        if (!terminal) {
          // Release dependency still pending: the dependency's release batch
          // is either behind this item in the FIFO or arrives via the host
          // func. Requeue at the tail — never block the only driver thread.
          item.requeued = true;
          queue_.push_back(std::move(item));
          if (queue_.size() == 1 && !shutdown_) {
            // Nothing else to make progress; a short timed wait avoids a hot
            // spin. enqueue_from_callback()'s notify wakes us immediately
            // when the release batch lands.
            cv_.wait_for(lock, std::chrono::microseconds(200),
                         [&] { return queue_.size() > 1 || shutdown_; });
          }
        }
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<WorkItem> queue_;
  bool shutdown_ = false;
  std::thread thread_;  // declared last; started in the constructor body
};

void ReleaseWorker::enqueue_remap(std::shared_ptr<RemapHookContext> context) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_) {
      context->error_code.store(kWorkerUnavailable, std::memory_order_relaxed);
      context->error.store(1, std::memory_order_relaxed);
      complete_context(*context);
      return;
    }
    WorkItem item{};
    item.remap = std::move(context);
    queue_.push_back(std::move(item));
  }
  cv_.notify_one();
}

static bool request_ready(const RemapRequest &request) {
  return !request.release_dependency ||
      request.release_dependency->done.load(std::memory_order_acquire) != 0;
}

static size_t find_ready_request(const RemapHookContext &context,
                                 const std::vector<bool> &processed) {
  // Reload in reverse submission order.  If the newest request is still
  // waiting for release, fall back to any older request that is ready rather
  // than reintroducing head-of-line blocking.
  for (size_t index = context.requests.size(); index-- > 0;) {
    if (!processed[index] && request_ready(context.requests[index])) return index;
  }
  return context.requests.size();
}

class RemapWorker {
 public:
  static RemapWorker &instance() {
    static RemapWorker worker;
    return worker;
  }

  void enqueue(std::shared_ptr<RemapHookContext> context) {
    if (vmm_serial_driver_workers()) {
      // Serial-driver mode: join the release worker's FIFO so every driver
      // call (unmap/release AND create/map/setAccess) runs on one thread.
      ReleaseWorker::instance().enqueue_remap(std::move(context));
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) {
        context->error_code.store(kWorkerUnavailable, std::memory_order_relaxed);
        context->error.store(1, std::memory_order_relaxed);
        complete_context(*context);
        return;
      }
      queue_.push_back(std::move(context));
    }
    cv_.notify_one();
  }

  void notify_release_ready() { cv_.notify_all(); }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }

 private:
  RemapWorker() { thread_ = std::thread(&RemapWorker::run, this); }
  ~RemapWorker() { shutdown(); }
  RemapWorker(const RemapWorker &) = delete;
  RemapWorker &operator=(const RemapWorker &) = delete;

 public:
  // Also invoked by ReleaseWorker in serial-driver mode; serial_requeue=true
  // makes an unsatisfied release dependency return instead of blocking.

  static void set_error(RemapHookContext &context, int error_code) {
    if (!context.error.exchange(1, std::memory_order_relaxed)) {
      context.error_code.store(error_code, std::memory_order_relaxed);
    }
  }

  static void cleanup_requests(RemapHookContext &context) {
    for (auto &request : context.requests) {
      // An adopted mapping belongs to the consumer until its completion path
      // explicitly releases it; rolling back here would fault graph replay.
      if (request.adopted && request.adopted->load(std::memory_order_acquire) != 0) {
        continue;
      }
      if (request.mapped) {
        muMemUnmap(request.address, request.bytes);
        request.mapped = false;
      }
      if (request.handle != 0) {
        muMemRelease(request.handle);
        request.handle = 0;
      }
    }
  }

  static void process(const std::shared_ptr<RemapHookContext> &context, bool serial_requeue) {
    std::unique_ptr<at::ThreadLocalStateGuard> tls_guard;
    if (context->tls_state) tls_guard = std::make_unique<at::ThreadLocalStateGuard>(*context->tls_state);
    RECORD_USER_SCOPE("vmm::RemapWorker.process");
    auto &ctx = *context;
    auto trace_stage = [&](const char *stage, const auto &stage_started) {
      vmm_remap_trace(stage, std::chrono::duration<double, std::milli>(
          VmmClock::now() - stage_started).count());
    };
    vmm_remap_trace(serial_requeue ? "worker start (serial, requeued)" : "worker start");
    bool any_copy_submitted = false;
    auto fail = [&](int error_code) {
      set_error(ctx, error_code);
      // A previously submitted copy may still access its new mapping. Finish
      // those copies before rolling the batch back on a later-slot failure.
      if (any_copy_submitted) musaStreamSynchronize(ctx.copy_stream);
      cleanup_requests(ctx);
      complete_context(ctx);
    };

    // A requeued serial batch may have processed some slots on an earlier
    // pass; rebuild completion from request terminal states instead of a
    // fresh vector, which would redo finished slots.
    std::vector<bool> processed(ctx.requests.size(), false);
    size_t completed_requests = 0;
    for (size_t index = 0; index < ctx.requests.size(); ++index) {
      const auto &request = ctx.requests[index];
      const bool terminal = request.remapped->load(std::memory_order_acquire) != 0 ||
          (request.done_event != nullptr && request.event_recorded->load(std::memory_order_acquire) != 0);
      if (terminal) {
        processed[index] = true;
        ++completed_requests;
      }
    }
    while (completed_requests < ctx.requests.size()) {
      size_t slot_index = find_ready_request(ctx, processed);
      if (slot_index == ctx.requests.size()) {
        // All remaining releases are pending.
        if (serial_requeue) {
          // Serial-driver mode: the pending release batch is queued behind
          // this one on the same worker; blocking here would deadlock the
          // only driver thread. Requeue and let the notifier wake the loop.
          vmm_remap_trace("serial requeue: release dependency pending");
          return;
        }
        // Wait for a completion notification, then rescan so ready fallback
        // remains effective.
        auto &notifier = remap_release_notifier();
        std::unique_lock<std::mutex> notifier_lock(notifier.mutex);
        const uint64_t generation = notifier.generation;
        notifier.cv.wait(notifier_lock, [&] {
          return notifier.generation != generation;
        });
        continue;
      }
      processed[slot_index] = true;
      ++completed_requests;
      auto &request = ctx.requests[slot_index];
      char label[256];
      std::snprintf(label, sizeof(label), "remap request id=%s begin address=0x%llx",
                    request.slot_id.c_str(),
                    static_cast<unsigned long long>(request.address));
      vmm_remap_trace(label);
      const auto request_started = VmmClock::now();
      auto trace_request_stage = [&](const char *stage, const auto &stage_started) {
        char stage_label[320];
        std::snprintf(stage_label, sizeof(stage_label), "request id=%s slot=%zu stage=%s",
                      request.slot_id.c_str(), request.slot_index, stage);
        vmm_remap_trace(stage_label, std::chrono::duration<double, std::milli>(
            VmmClock::now() - stage_started).count());
      };
      if (request.release_dependency) {
        trace_request_stage("release_dependency_ready", request_started);
        if (request.release_dependency->error.load(std::memory_order_acquire)) {
          fail(request.release_dependency->error_code.load(std::memory_order_relaxed));
          return;
        }
      }
      ctx.releases_ready.fetch_add(1, std::memory_order_release);
      const auto properties = allocation_properties(request.device);
      MUresult rc;
      const auto create_started = VmmClock::now();
      { RECORD_USER_SCOPE("vmm::remap.muMemCreate");
        rc = muMemCreate(&request.handle, request.bytes, &properties, 0); }
      trace_request_stage("muMemCreate", create_started);
      if (rc != MUSA_SUCCESS) { fail(static_cast<int>(rc)); return; }
      const auto map_started = VmmClock::now();
      { RECORD_USER_SCOPE("vmm::remap.muMemMap");
        rc = muMemMap(request.address, request.bytes, 0, request.handle, 0); }
      trace_request_stage("muMemMap", map_started);
      if (rc != MUSA_SUCCESS) { fail(static_cast<int>(rc)); return; }
      request.mapped = true;
      MUmemAccessDesc access{};
      access.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = request.device;
      access.flags = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      const auto access_started = VmmClock::now();
      { RECORD_USER_SCOPE("vmm::remap.muMemSetAccess");
        rc = muMemSetAccess(request.address, request.bytes, &access, 1); }
      trace_request_stage("muMemSetAccess", access_started);
      if (rc != MUSA_SUCCESS) { fail(static_cast<int>(rc)); return; }
      ctx.remaps_done.fetch_add(1, std::memory_order_release);
      if (ctx.defer_copy) {
        // Deferred-H2D mode: the mapping transition is complete, but the copy
        // is launched later at graph-replay time by remap_slot_launch_h2d.
        {
          std::lock_guard<std::mutex> lock(ctx.completion_mutex);
          request.remapped->store(1, std::memory_order_release);
        }
        request.state_cv->notify_all();
        ctx.completion_cv.notify_all();
        notify_remap_worker_release_ready();
      } else if (ctx.copy_enabled) {
        musaError_t runtime_rc = musaSetDevice(request.device);
        if (runtime_rc != MUSA_SUCCESS) { fail(static_cast<int>(runtime_rc)); return; }
        { RECORD_USER_SCOPE("vmm::remap.slot_h2d_submit");
          runtime_rc = musaMemcpyAsync(reinterpret_cast<void *>(static_cast<uintptr_t>(request.address)),
              request.host_address, request.copy_bytes, musaMemcpyHostToDevice, ctx.copy_stream); }
        if (runtime_rc != MUSA_SUCCESS) { fail(static_cast<int>(runtime_rc)); return; }
        request.copy_submitted = true;
        any_copy_submitted = true;
        ctx.copies_submitted.fetch_add(1, std::memory_order_release);
        { RECORD_USER_SCOPE("vmm::remap.slot_event_record");
          runtime_rc = musaEventRecord(ctx.slot_done_events[slot_index], ctx.copy_stream); }
        if (runtime_rc != MUSA_SUCCESS) { fail(static_cast<int>(runtime_rc)); return; }
        {
          std::lock_guard<std::mutex> lock(ctx.completion_mutex);
          request.event_recorded->store(1, std::memory_order_release);
        }
        request.state_cv->notify_all();
        ctx.completion_cv.notify_all();
      }
      trace_request_stage("vmm_create_to_map", create_started);
      trace_stage("request total", request_started);
    }

    if (ctx.defer_copy) {
      // Remap-only batch: there is no copy stream to gate a host func on. The
      // final reference transfer happens in remap_slot_launch_h2d once the
      // last deferred H2D has been submitted; nothing to do here.
      complete_context(ctx);
      return;
    }
    if (ctx.copy_enabled) {
      musaError_t rc;
      {
        RECORD_USER_SCOPE("vmm::remap.copy_done_event_record");
        rc = musaEventRecord(ctx.copy_done_event, ctx.copy_stream);
      }
      if (rc != MUSA_SUCCESS) {
        fail(static_cast<int>(rc));
        return;
      }
      // The callback runs after H2D and transfers its stream-owned reference to
      // ReloadCleanupWorker. It neither calls a MUSA API nor destroys the context.
      auto *lifetime = new std::shared_ptr<RemapHookContext>(context);
      rc = musaLaunchHostFunc(ctx.copy_stream, reload_lifetime_host_func, lifetime);
      if (rc != MUSA_SUCCESS) {
        delete lifetime;
        fail(static_cast<int>(rc));
        return;
      }
    }
    // This publishes API submission, not H2D GPU completion. Consumers install
    // a stream wait on copy_done_event instead of synchronizing the host.
    complete_context(ctx);
  }

  void run() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      while (queue_.empty() && !shutdown_) cv_.wait(lock);
      if (queue_.empty() && shutdown_) return;
      auto context = std::move(queue_.front());
      queue_.pop_front();
      lock.unlock();
      process(context, /*serial_requeue=*/false);
      lock.lock();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<RemapHookContext>> queue_;
  bool shutdown_ = false;
  std::thread thread_;
};

// Returns true when the batch finished (all slots terminal or failed); false
// means a release dependency is still pending and the serial worker must
// requeue it instead of blocking the only driver thread.  Defined after
// RemapWorker, whose process() it delegates to.
bool ReleaseWorker::process_remap_batch(const std::shared_ptr<RemapHookContext> &context,
                                        bool requeued) {
  RemapWorker::process(context, /*serial_requeue=*/requeued);
  return context->done.load(std::memory_order_acquire) != 0;
}

void notify_remap_worker_release_ready() {
  auto &notifier = remap_release_notifier();
  {
    std::lock_guard<std::mutex> lock(notifier.mutex);
    ++notifier.generation;
  }
  notifier.cv.notify_all();
}

void release_host_func(void *user_data) {
  // The box keeps the context alive until the stream actually reaches this
  // host func, even if every Python/slot reference is gone before that.
  auto *boxed = static_cast<std::shared_ptr<ReleaseHookContext> *>(user_data);
  std::shared_ptr<ReleaseHookContext> context(std::move(*boxed));
  delete boxed;
  context->callback_fired.store(1, std::memory_order_release);
  vmm_remap_trace("release host func fired (D2H burst complete, batch queued)");
  ReleaseWorker::instance().enqueue_from_callback(std::move(context));
  // NOTE: no MUSA/driver API calls here (they deadlock on the callback thread).
}

void launch_release_hook(const std::shared_ptr<ReleaseHookContext> &context,
                         musaStream_t stream) {
  auto *boxed = new std::shared_ptr<ReleaseHookContext>(context);
  musaError_t rc = musaLaunchHostFunc(stream, release_host_func, boxed);
  if (rc != MUSA_SUCCESS) {
    delete boxed;
    throw std::runtime_error(std::string("musaLaunchHostFunc failed: ") +
                             musa_runtime_error_text(rc));
  }
}

void remap_host_func(void *user_data) {
  auto *boxed = static_cast<std::shared_ptr<RemapHookContext> *>(user_data);
  std::shared_ptr<RemapHookContext> context(std::move(*boxed));
  delete boxed;
  context->callback_fired.store(1, std::memory_order_release);
  RemapWorker::instance().enqueue(std::move(context));
}

void launch_remap_hook(const std::shared_ptr<RemapHookContext> &context,
                       musaStream_t stream) {
  auto *boxed = new std::shared_ptr<RemapHookContext>(context);
  musaError_t rc = musaLaunchHostFunc(stream, remap_host_func, boxed);
  if (rc != MUSA_SUCCESS) {
    delete boxed;
    throw std::runtime_error(std::string("musaLaunchHostFunc(remap) failed: ") +
                             musa_runtime_error_text(rc));
  }
}

class VMMActivationSlot;
std::shared_ptr<ReleaseHookContext> release_hooks_after(
    std::vector<std::shared_ptr<VMMActivationSlot>> slots, uintptr_t raw_stream);
std::shared_ptr<RemapHookContext> remap_hooks_after(
    std::vector<std::shared_ptr<VMMActivationSlot>> slots, uintptr_t raw_stream);
std::shared_ptr<RemapHookContext> remap_and_copy_after(
    std::vector<std::shared_ptr<VMMActivationSlot>> slots,
    std::vector<at::Tensor> host_tensors, uintptr_t raw_stream);
std::shared_ptr<RemapHookContext> remap_and_copy_slot_after(
    std::shared_ptr<VMMActivationSlot> slot, at::Tensor host_tensor, uintptr_t raw_stream);

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
    // NOTE: the returned tensor is a non-owning view; it must never be used
    // while the slot is unmapped (hard page fault on dereference).
  }

  // -------------------------------------------------------------------------
  // DEPRECATED: synchronous release. Prefer `release_hook_after` + the
  // completion flag, which moves unmap/release off the critical path. This
  // method remains for backward compatibility and testing.
  // -------------------------------------------------------------------------
  void unmap_and_release() {
    RECORD_USER_SCOPE("vmm::unmap_and_release");
    drain_remap();
    drain_pending();
    ensure_reserved();
    if (!mapped_) throw std::runtime_error("VMM activation slot is already unmapped");
    check_musa(muMemUnmap(address_, bytes_), "muMemUnmap");
    mapped_ = false;
    check_musa(muMemRelease(handle_), "muMemRelease");
    handle_ = 0;
  }

  void create_and_remap() {
    RECORD_USER_SCOPE("vmm::create_and_remap");
    drain_remap();
    drain_pending();
    ensure_reserved();
    if (mapped_) throw std::runtime_error("VMM activation slot is already mapped");
    map_new_physical_allocation();
  }

  // -------------------------------------------------------------------------
  // Async release API
  // -------------------------------------------------------------------------

  // Snapshot the raw (address, bytes, handle) needed for release. The slot
  // keeps its reservation and object identity; `mapped_` is *not* cleared
  // here because the physical release has not happened yet.
  ReleaseRequest make_release_request() const {
    ensure_reserved();
    if (!mapped_) throw std::runtime_error("VMM activation slot is already unmapped");
    return ReleaseRequest{slot_id_, address_, bytes_, handle_};
  }

  // Called once the release worker reports completion: clears the mapped
  // state so the slot can be remapped later.
  void on_async_release_completed() {
    ensure_reserved();
    mapped_ = false;
    handle_ = 0;
  }

  bool async_release_done() const {
    auto context = pending_;
    if (!context) return true;
    return context->done.load(std::memory_order_acquire) == 1;
  }

  // Enqueue a host function on `stream` that fires after all previously
  // enqueued work (the D2H burst) completes and hands the raw release work
  // to the resident worker. Returns the context whose `done` flag flips to 1
  // once unmap+release have actually finished on the worker thread.
  std::shared_ptr<ReleaseHookContext> release_hook_after(uintptr_t raw_stream) {
    RECORD_USER_SCOPE("vmm::release_hook_after");
    drain_remap();
    ensure_reserved();
    if (pending_) {
      throw std::runtime_error("VMM activation slot already has an in-flight release");
    }
    if (!mapped_) throw std::runtime_error("VMM activation slot is already unmapped");
    auto context = std::make_shared<ReleaseHookContext>();
    context->tls_state = std::make_shared<at::ThreadLocalState>();
    context->requests.push_back(make_release_request());
    pending_ = context;
    launch_release_hook(context, reinterpret_cast<musaStream_t>(raw_stream));
    return context;
  }

  // Block until the in-flight async release finished and apply its state.
  // This compatibility API uses a condition variable and never polls or
  // synchronizes a MUSA stream/event.
  void wait_for_async_release() {
    RECORD_USER_SCOPE("vmm::wait_for_async_release");
    drain_pending();
  }

  bool async_remap_done() const {
    auto context = pending_remap_;
    if (!context) return true;
    return context->done.load(std::memory_order_acquire) == 1;
  }

  void wait_for_async_remap() {
    RECORD_USER_SCOPE("vmm::wait_for_async_remap");
    drain_remap();
  }

  void adopt_async_remap() {
    RECORD_USER_SCOPE("vmm::adopt_async_remap");
    auto context = pending_remap_;
    if (!context) return;
    auto &request = context->requests.at(pending_remap_index_);
    // A slot is independently adoptable as soon as its own mapping, H2D, and
    // completion event have been submitted.  Do not wait for later requests
    // in the shared worker context (select/epoll-style readiness).
    if (context->copy_enabled &&
        request.event_recorded->load(std::memory_order_acquire) == 0) {
      throw std::runtime_error("VMM async remap slot submission is not complete");
    }
    if (!request.mapped || request.handle == 0) {
      if (context->error.load(std::memory_order_acquire)) {
        throw std::runtime_error("VMM async remap failed on worker: MUResult " +
                                 std::to_string(context->error_code.load()));
      }
      throw std::runtime_error("VMM async remap slot has no valid mapping");
    }
    // The worker waits for this slot's release dependency before publishing
    // request.mapped.  Release contexts are per slot in the training path.
    drain_pending();
    request.adopted->store(1, std::memory_order_release);
    handle_ = request.handle;
    mapped_ = true;
    pending_remap_.reset();
  }

  void set_slot_id(const std::string &slot_id) { slot_id_ = slot_id; }
  const std::string &slot_id() const { return slot_id_; }

  py::dict info() const {
    py::dict result;
    result["slot_id"] = slot_id_;
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
    drain_remap();
    drain_pending();
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
  friend std::shared_ptr<ReleaseHookContext> release_hooks_after(
      std::vector<std::shared_ptr<VMMActivationSlot>> slots, uintptr_t raw_stream);
  friend std::shared_ptr<RemapHookContext> remap_hooks_after(
      std::vector<std::shared_ptr<VMMActivationSlot>> slots, uintptr_t raw_stream);
  friend std::shared_ptr<RemapHookContext> remap_and_copy_after(
      std::vector<std::shared_ptr<VMMActivationSlot>> slots,
      std::vector<at::Tensor> host_tensors, uintptr_t raw_stream);
  friend std::shared_ptr<RemapHookContext> remap_only_slot_after(
      std::shared_ptr<VMMActivationSlot> slot, at::Tensor host_tensor,
      uintptr_t raw_stream);

  void drain_remap() {
    auto context = pending_remap_;
    if (!context) return;
    wait_for_context(context);
    if (context->error.load(std::memory_order_acquire)) {
      pending_remap_.reset();
      throw std::runtime_error("VMM async remap failed on worker: MUResult " +
                               std::to_string(context->error_code.load()));
    }
    // The remap worker waits for release completion before creating the new
    // mapping. Apply the release-side bookkeeping before adopting its handle.
    drain_pending();
    auto &request = context->requests.at(pending_remap_index_);
    if (!request.mapped || request.handle == 0) {
      pending_remap_.reset();
      throw std::runtime_error("VMM async remap completed without a valid mapping");
    }
    request.adopted->store(1, std::memory_order_release);
    handle_ = request.handle;
    mapped_ = true;
    pending_remap_.reset();
  }

  // Wait out the pending context, surface worker errors, apply the state
  // transition. A context is guaranteed to complete: the worker is started
  // before any hook can be enqueued and drains its queue on shutdown.
  void drain_pending() {
    auto context = pending_;
    if (!context) return;
    const auto wait_started = VmmClock::now();
    wait_for_context(context);
    vmm_remap_trace("wait_for_async_release host wait",
                    std::chrono::duration<double, std::milli>(
                        VmmClock::now() - wait_started).count());
    pending_.reset();
    if (context->error.load(std::memory_order_acquire)) {
      throw std::runtime_error("VMM async release failed on worker: MUResult " +
                               std::to_string(context->error_code.load()));
    }
    on_async_release_completed();
  }

  // Bounded variant used on teardown paths where an unrun host func must not
  // wedge the process. Returns true when `done` flipped within the timeout.
  template <typename Context>
  static bool wait_until_done(const std::shared_ptr<Context> &context,
                              std::chrono::milliseconds timeout) {
    return wait_for_context(context, timeout);
  }

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
    if (pending_remap_) {
      // Never free a VA range while a callback/worker may still map into it.
      // On teardown timeout, intentionally leak the reservation rather than
      // allow a late remap to corrupt a subsequently reused address range.
      if (!wait_until_done(pending_remap_, std::chrono::seconds(10))) return;
      if (!pending_remap_->error.load(std::memory_order_acquire)) {
        const auto &request = pending_remap_->requests[pending_remap_index_];
        if (request.mapped && request.handle != 0) {
          mapped_ = true;
          handle_ = request.handle;
        }
      }
      pending_remap_.reset();
      // Successful remap implies its release dependencies completed. On a
      // remap error, handle the release context independently below.
      if (mapped_) pending_.reset();
    }
    if (pending_) {
      if (!wait_until_done(pending_, std::chrono::seconds(10))) return;
      if (!pending_->error.load(std::memory_order_acquire)) {
        mapped_ = false;
        handle_ = 0;
      }
      pending_.reset();
    }
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
  std::shared_ptr<ReleaseHookContext> pending_;  // in-flight async release
  std::shared_ptr<RemapHookContext> pending_remap_;
  size_t pending_remap_index_ = 0;
  std::string slot_id_ = "unassigned";
};

// ---------------------------------------------------------------------------
// Python-facing batch helper: enqueue ONE host func after a whole batch of
// slots' D2H copies so the DMA burst stays contiguous; the callback hands all
// raw release work to the worker in one go.
// ---------------------------------------------------------------------------

std::shared_ptr<ReleaseHookContext> release_hooks_after(
    std::vector<std::shared_ptr<VMMActivationSlot>> slots, uintptr_t raw_stream) {
  RECORD_USER_SCOPE("vmm::release_hooks_after");
  // Ensure the worker thread exists before any host func can fire.
  ReleaseWorker::instance();
  auto context = std::make_shared<ReleaseHookContext>();
  context->tls_state = std::make_shared<at::ThreadLocalState>();
  for (auto &slot : slots) {
    if (slot->pending_remap_) {
      throw std::runtime_error("VMM batch release hook got a slot with an in-flight remap");
    }
    if (slot->pending_) {
      throw std::runtime_error("VMM batch release hook got a slot with an in-flight release");
    }
    context->requests.push_back(slot->make_release_request());
    if (context->requests.back().handle == 0) {
      throw std::runtime_error("VMM batch release hook got an unmapped slot");
    }
  }
  for (auto &slot : slots) slot->pending_ = context;
  try {
    launch_release_hook(context, reinterpret_cast<musaStream_t>(raw_stream));
  } catch (...) {
    for (auto &slot : slots) {
      if (slot->pending_ == context) slot->pending_.reset();
    }
    throw;
  }
  return context;
}

std::shared_ptr<RemapHookContext> remap_hooks_after(
    std::vector<std::shared_ptr<VMMActivationSlot>> slots, uintptr_t raw_stream) {
  // This worker must be distinct from ReleaseWorker: it may wait for a release
  // whose callback has not reached the release queue yet.
  RemapWorker::instance();
  auto context = std::make_shared<RemapHookContext>();
  context->tls_state = std::make_shared<at::ThreadLocalState>();
  std::unordered_set<VMMActivationSlot *> seen_slots;
  context->requests.reserve(slots.size());
  for (size_t index = 0; index < slots.size(); ++index) {
    auto &slot = slots[index];
    if (!slot) throw std::runtime_error("VMM batch remap hook got a null slot");
    if (!seen_slots.insert(slot.get()).second) {
      throw std::runtime_error("VMM batch remap hook got a duplicate slot");
    }
    slot->ensure_reserved();
    if (slot->pending_remap_) {
      throw std::runtime_error("VMM batch remap hook got a slot with an in-flight remap");
    }
    if (slot->mapped_ && !slot->pending_) {
      throw std::runtime_error("VMM batch remap hook got a mapped slot without a pending release");
    }
    RemapRequest request{slot->slot_id(), slot->address_, slot->bytes_, 0, nullptr,
                         slot->device_, slot->pending_, 0, false, false,
                         index, nullptr, std::make_shared<std::atomic<int>>(0),
                         std::make_shared<std::atomic<int>>(0),
                         std::make_shared<std::atomic<int>>(1)};
    context->requests.push_back(std::move(request));
  }
  for (size_t index = 0; index < slots.size(); ++index) {
    slots[index]->pending_remap_ = context;
    slots[index]->pending_remap_index_ = index;
  }
  try {
    launch_remap_hook(context, reinterpret_cast<musaStream_t>(raw_stream));
  } catch (...) {
    for (auto &slot : slots) {
      if (slot->pending_remap_ == context) slot->pending_remap_.reset();
    }
    throw;
  }
  return context;
}

std::shared_ptr<RemapHookContext> remap_and_copy_after(
    std::vector<std::shared_ptr<VMMActivationSlot>> slots,
    std::vector<at::Tensor> host_tensors, uintptr_t raw_stream) {
  RECORD_USER_SCOPE("vmm::remap_and_copy_after");
  if (slots.size() != host_tensors.size()) {
    throw std::runtime_error("VMM remap-and-copy requires one host tensor per slot");
  }
  ReloadCleanupWorker::instance();
  RemapWorker::instance();
  auto context = std::make_shared<RemapHookContext>();
  context->tls_state = std::make_shared<at::ThreadLocalState>();
  context->copy_stream = reinterpret_cast<musaStream_t>(raw_stream);
  context->copy_enabled = true;
  context->host_tensors = std::move(host_tensors);
  check_musa_runtime(musaEventCreateWithFlags(&context->copy_done_event, musaEventDisableTiming),
                     "musaEventCreateWithFlags(remap-and-copy)");
  context->slot_done_events.resize(slots.size(), nullptr);
  for (auto &event : context->slot_done_events) {
    check_musa_runtime(musaEventCreateWithFlags(&event, musaEventDisableTiming),
                       "musaEventCreateWithFlags(remap-slot)");
  }

  std::unordered_set<VMMActivationSlot *> seen_slots;
  context->requests.reserve(slots.size());
  for (size_t index = 0; index < slots.size(); ++index) {
    auto &slot = slots[index];
    auto &host_tensor = context->host_tensors[index];
    if (!slot) throw std::runtime_error("VMM remap-and-copy got a null slot");
    if (!seen_slots.insert(slot.get()).second) {
      throw std::runtime_error("VMM remap-and-copy got a duplicate slot");
    }
    slot->ensure_reserved();
    if (slot->pending_remap_) {
      throw std::runtime_error("VMM remap-and-copy got a slot with an in-flight remap");
    }
    if (slot->mapped_ && !slot->pending_) {
      throw std::runtime_error("VMM remap-and-copy got a mapped slot without a pending release");
    }
    if (!host_tensor.device().is_cpu() || !host_tensor.is_pinned()) {
      throw std::runtime_error("VMM remap-and-copy source must be a pinned CPU tensor");
    }
    if (host_tensor.storage_offset() != 0 ||
        host_tensor.storage().nbytes() < slot->requested_bytes_) {
      throw std::runtime_error("VMM remap-and-copy source storage is too small or offset");
    }
    RemapRequest request{slot->slot_id(), slot->address_, slot->bytes_, slot->requested_bytes_,
                         host_tensor.data_ptr(), slot->device_, slot->pending_,
                         0, false, false, index, context->slot_done_events[index],
                         std::make_shared<std::atomic<int>>(0),
                         std::make_shared<std::atomic<int>>(0),
                         std::make_shared<std::atomic<int>>(0)};
    context->requests.push_back(std::move(request));
  }

  for (size_t index = 0; index < slots.size(); ++index) {
    slots[index]->pending_remap_ = context;
    slots[index]->pending_remap_index_ = index;
  }
  // Reload correctness is carried by each slot's release context, not by work
  // already queued on the H2D stream. Dispatch immediately so remap can overlap
  // the predecessor graph instead of waiting for that stream to reach a callback.
  context->callback_fired.store(1, std::memory_order_release);
  RemapWorker::instance().enqueue(context);
  return context;
}

std::shared_ptr<RemapHookContext> remap_and_copy_slot_after(
    std::shared_ptr<VMMActivationSlot> slot, at::Tensor host_tensor, uintptr_t raw_stream) {
  if (!slot) throw std::runtime_error("VMM remap slot request got a null slot");
  // Keep this API intentionally one-slot: one call is one server request, one
  // mapping transition, one H2D submission, and one completion event.
  vmm_remap_trace((std::string("enqueue remap request id=") + slot->slot_id()).c_str());
  std::vector<std::shared_ptr<VMMActivationSlot>> slots;
  slots.push_back(std::move(slot));
  std::vector<at::Tensor> host_tensors;
  host_tensors.push_back(std::move(host_tensor));
  return remap_and_copy_after(std::move(slots), std::move(host_tensors), raw_stream);
}

// Deferred-H2D variant: the worker performs only the VMM transition now; the
// H2D memcpy is submitted later by remap_slot_launch_h2d at replay time.
std::shared_ptr<RemapHookContext> remap_only_slot_after(
    std::shared_ptr<VMMActivationSlot> slot, at::Tensor host_tensor, uintptr_t raw_stream) {
  if (!slot) throw std::runtime_error("VMM remap-only slot request got a null slot");
  vmm_remap_trace((std::string("enqueue remap-only request id=") + slot->slot_id()).c_str());
  ReloadCleanupWorker::instance();
  RemapWorker::instance();
  auto context = std::make_shared<RemapHookContext>();
  context->tls_state = std::make_shared<at::ThreadLocalState>();
  context->defer_copy = true;
  context->copy_stream = reinterpret_cast<musaStream_t>(raw_stream);
  context->host_tensors.push_back(std::move(host_tensor));
  context->slot_done_events.resize(1, nullptr);
  check_musa_runtime(musaEventCreateWithFlags(&context->slot_done_events[0],
                                              musaEventDisableTiming),
                     "musaEventCreateWithFlags(remap-only-slot)");
  auto &host_tensor_ref = context->host_tensors[0];
  auto &checked_slot = slot;
  if (!checked_slot) throw std::runtime_error("VMM remap-only got a null slot");
  checked_slot->ensure_reserved();
  if (checked_slot->pending_remap_) {
    throw std::runtime_error("VMM remap-only got a slot with an in-flight remap");
  }
  if (checked_slot->mapped_ && !checked_slot->pending_) {
    throw std::runtime_error("VMM remap-only got a mapped slot without a pending release");
  }
  if (!host_tensor_ref.device().is_cpu() || !host_tensor_ref.is_pinned()) {
    throw std::runtime_error("VMM remap-only source must be a pinned CPU tensor");
  }
  if (host_tensor_ref.storage_offset() != 0 ||
      host_tensor_ref.storage().nbytes() < checked_slot->requested_bytes_) {
    throw std::runtime_error("VMM remap-only source storage is too small or offset");
  }
  RemapRequest request{checked_slot->slot_id(), checked_slot->address_, checked_slot->bytes_,
                       checked_slot->requested_bytes_, host_tensor_ref.data_ptr(),
                       checked_slot->device_, checked_slot->pending_,
                       0, false, false, 0, context->slot_done_events[0],
                       std::make_shared<std::atomic<int>>(0),
                       std::make_shared<std::atomic<int>>(0),
                       std::make_shared<std::atomic<int>>(1),
                       std::make_shared<std::atomic<int>>(0)};
  context->requests.push_back(std::move(request));
  checked_slot->pending_remap_ = context;
  checked_slot->pending_remap_index_ = 0;
  context->callback_fired.store(1, std::memory_order_release);
  RemapWorker::instance().enqueue(context);
  return context;
}


void remap_wait_on_stream(const std::shared_ptr<RemapHookContext> &context,
                          uintptr_t raw_stream) {
  RECORD_USER_SCOPE("vmm::wait_remap_copy_on_stream");
  const auto wait_started = VmmClock::now();
  vmm_remap_trace("wait_on_stream begin");
  {
    RECORD_USER_SCOPE("vmm::wait_remap_copy_on_stream.wait_context");
    wait_for_context(context);
  }
  vmm_remap_trace(
      "wait_on_stream context ready",
      std::chrono::duration<double, std::milli>(VmmClock::now() - wait_started).count());
  if (context->error.load(std::memory_order_acquire)) {
    throw std::runtime_error("VMM asynchronous reload submission failed: error " +
                             std::to_string(context->error_code.load()));
  }
  if (!context->copy_enabled || context->copy_done_event == nullptr) {
    throw std::runtime_error("VMM remap context has no asynchronous H2D completion event");
  }
  {
    RECORD_USER_SCOPE("vmm::wait_remap_copy_on_stream.stream_wait_event");
    check_musa_runtime(
        musaStreamWaitEvent(reinterpret_cast<musaStream_t>(raw_stream),
                            context->copy_done_event, 0),
        "musaStreamWaitEvent(remap-and-copy)");
  }
}

// Enqueue only the device-side dependency. The RemapWorker remains responsible
// for VMM remap and H2D submission; the caller does not wait for its context.
// The consumer stream waits on copy_done_event, so a following graph cannot
// touch the reloaded VA before the worker's H2D has completed.
void remap_enqueue_wait_on_stream(const std::shared_ptr<RemapHookContext> &context,
                                  uintptr_t raw_stream) {
  RECORD_USER_SCOPE("vmm::enqueue_remap_copy_wait");
  if (!context->copy_enabled || context->copy_done_event == nullptr) {
    throw std::runtime_error("VMM remap context has no asynchronous H2D completion event");
  }
  {
    RECORD_USER_SCOPE("vmm::enqueue_remap_copy_wait.stream_wait_event");
    check_musa_runtime(
        musaStreamWaitEvent(reinterpret_cast<musaStream_t>(raw_stream),
                            context->copy_done_event, 0),
        "musaStreamWaitEvent(remap-and-copy)");
  }
}

void wait_until_slot_submitted(const std::shared_ptr<RemapHookContext> &context,
                               size_t slot_index) {
  if (slot_index >= context->requests.size()) {
    throw std::out_of_range("VMM remap slot index out of range");
  }
  auto &request = context->requests[slot_index];
  std::unique_lock<std::mutex> lock(*request.state_mutex);
  request.state_cv->wait_for(lock, std::chrono::milliseconds(1), [&] {
    return request.event_recorded->load(std::memory_order_acquire) != 0 ||
           context->done.load(std::memory_order_acquire) != 0;
  });
  while (request.event_recorded->load(std::memory_order_acquire) == 0 &&
         context->done.load(std::memory_order_acquire) == 0) {
    request.state_cv->wait_for(lock, std::chrono::milliseconds(1));
  }
  if (context->error.load(std::memory_order_acquire)) {
    throw std::runtime_error("VMM asynchronous reload submission failed: error " +
                             std::to_string(context->error_code.load()));
  }
  if (request.event_recorded->load(std::memory_order_acquire) == 0) {
    throw std::runtime_error("VMM remap slot completed without an H2D event");
  }
}

void remap_enqueue_slot_wait(const std::shared_ptr<RemapHookContext> &context,
                             size_t slot_index, uintptr_t raw_stream) {
  RECORD_USER_SCOPE("vmm::enqueue_remap_slot_wait");
  wait_until_slot_submitted(context, slot_index);
  auto event = context->requests[slot_index].done_event;
  if (event == nullptr) {
    throw std::runtime_error("VMM remap slot has no completion event");
  }
  check_musa_runtime(
      musaStreamWaitEvent(reinterpret_cast<musaStream_t>(raw_stream), event, 0),
      "musaStreamWaitEvent(remap-slot)");
}

// Deferred-H2D launch: the worker finished the VMM transition for this slot
// earlier; submit the H2D memcpy now (graph-replay time) on the caller's
// stream.  If the remap is not finished yet, block until it is, so the graph
// and its H2D are issued at the same host moment.
void remap_slot_launch_h2d(const std::shared_ptr<RemapHookContext> &context,
                           size_t slot_index, uintptr_t raw_stream) {
  RECORD_USER_SCOPE("vmm::remap_slot_launch_h2d");
  if (!context->defer_copy) {
    throw std::runtime_error("VMM slot H2D launch requires a deferred-copy context");
  }
  if (slot_index >= context->requests.size()) {
    throw std::out_of_range("VMM remap slot index out of range");
  }
  auto &ctx = *context;
  auto &request = ctx.requests[slot_index];
  auto stream = reinterpret_cast<musaStream_t>(raw_stream);
  const auto wait_started = VmmClock::now();
  {
    std::unique_lock<std::mutex> lock(*request.state_mutex);
    request.state_cv->wait(lock, [&] {
      return request.remapped->load(std::memory_order_acquire) != 0 ||
             ctx.done.load(std::memory_order_acquire) != 0;
    });
  }
  const double wait_ms = std::chrono::duration<double, std::milli>(
      VmmClock::now() - wait_started).count();
  if (wait_ms > 0.1) {
    char label[256];
    std::snprintf(label, sizeof(label),
                  "h2d_launch id=%s remap_wait_ms=%.3f", request.slot_id.c_str(), wait_ms);
    vmm_remap_trace(label);
  }
  if (ctx.error.load(std::memory_order_acquire)) {
    throw std::runtime_error("VMM asynchronous reload submission failed: error " +
                             std::to_string(ctx.error_code.load()));
  }
  if (request.remapped->load(std::memory_order_acquire) == 0) {
    throw std::runtime_error("VMM remap slot completed without a mapping");
  }
  musaError_t rc = musaSetDevice(request.device);
  if (rc != MUSA_SUCCESS) {
    throw std::runtime_error(std::string("musaSetDevice failed: ") + musaGetErrorString(rc));
  }
  {
    RECORD_USER_SCOPE("vmm::remap_slot_launch_h2d.memcpy_async");
    rc = musaMemcpyAsync(reinterpret_cast<void *>(static_cast<uintptr_t>(request.address)),
                         request.host_address, request.copy_bytes,
                         musaMemcpyHostToDevice, stream);
  }
  if (rc != MUSA_SUCCESS) {
    throw std::runtime_error(std::string("musaMemcpyAsync(H2D launch) failed: ") +
                             musaGetErrorString(rc));
  }
  request.copy_submitted = true;
  ctx.copies_submitted.fetch_add(1, std::memory_order_release);
  {
    RECORD_USER_SCOPE("vmm::remap_slot_launch_h2d.event_record");
    rc = musaEventRecord(request.done_event, stream);
  }
  if (rc != MUSA_SUCCESS) {
    throw std::runtime_error(std::string("musaEventRecord(H2D launch) failed: ") +
                             musaGetErrorString(rc));
  }
  {
    std::lock_guard<std::mutex> lock(ctx.completion_mutex);
    request.event_recorded->store(1, std::memory_order_release);
  }
  request.state_cv->notify_all();
  ctx.completion_cv.notify_all();
  const size_t launched = ctx.h2d_launched.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (launched == ctx.requests.size()) {
    // Last deferred H2D submitted: hand the stream-owned reference to the
    // cleanup worker so the context outlives the copies without a host-side
    // block here.
    auto *lifetime = new std::shared_ptr<RemapHookContext>(context);
    rc = musaLaunchHostFunc(stream, reload_lifetime_host_func, lifetime);
    if (rc != MUSA_SUCCESS) {
      delete lifetime;
      throw std::runtime_error(std::string("musaLaunchHostFunc(H2D launch) failed: ") +
                               musaGetErrorString(rc));
    }
  }
}



void remap_enqueue_slot_waits(const std::shared_ptr<RemapHookContext> &context,
                              uintptr_t raw_stream) {
  RECORD_USER_SCOPE("vmm::enqueue_remap_slot_waits");
  wait_for_context(context);
  if (context->error.load(std::memory_order_acquire)) {
    throw std::runtime_error("VMM asynchronous reload submission failed: error " +
                             std::to_string(context->error_code.load()));
  }
  auto stream = reinterpret_cast<musaStream_t>(raw_stream);
  for (auto event : context->slot_done_events) {
    if (event == nullptr) throw std::runtime_error("VMM slot has no completion event");
    check_musa_runtime(musaStreamWaitEvent(stream, event, 0),
                       "musaStreamWaitEvent(remap-slot)");
  }
}
py::dict release_hook_status(const std::shared_ptr<ReleaseHookContext> &context) {
  py::dict result;
  result["callback_fired"] = context->callback_fired.load(std::memory_order_acquire);
  result["done"] = context->done.load(std::memory_order_acquire);
  result["error"] = context->error.load(std::memory_order_acquire);
  result["error_code"] = context->error_code.load(std::memory_order_acquire);
  return result;
}

py::dict remap_hook_status(const std::shared_ptr<RemapHookContext> &context) {
  py::dict result;
  result["callback_fired"] = context->callback_fired.load(std::memory_order_acquire);
  result["done"] = context->done.load(std::memory_order_acquire);
  result["error"] = context->error.load(std::memory_order_acquire);
  result["error_code"] = context->error_code.load(std::memory_order_acquire);
  result["request_count"] = context->requests.size();
  result["releases_ready"] = context->releases_ready.load(std::memory_order_acquire);
  result["remaps_done"] = context->remaps_done.load(std::memory_order_acquire);
  result["copies_submitted"] = context->copies_submitted.load(std::memory_order_acquire);
  return result;
}

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

py::dict vmm_profiler_status() {
  py::dict result;
  result["record_function_callbacks"] = at::hasCallbacks();
  result["native_trace_enabled"] = vmm_trace_enabled.load(std::memory_order_acquire);
  {
    std::lock_guard<std::mutex> lock(vmm_trace_mutex);
    result["native_trace_records"] = vmm_trace_records.size();
    result["native_trace_path"] = vmm_trace_path;
  }
  return result;
}

void vmm_initialize_workers() {
  ReleaseWorker::instance();
  RemapWorker::instance();
  ReloadCleanupWorker::instance();
}

void vmm_set_serial_driver_workers(bool enabled) {
  vmm_serial_driver_override.store(enabled ? 1 : 0, std::memory_order_release);
}

bool vmm_get_serial_driver_workers() {
  return vmm_serial_driver_workers();
}

void init_vmm_activation_extension(py::module_ &module) {
  module.def("vmm_enable_trace", &vmm_enable_trace, py::arg("enabled"), py::arg("path"));
  module.def("vmm_dump_trace", &vmm_dump_trace);
  module.def("vmm_reset_trace", &vmm_reset_trace);
  module.def("vmm_driver_memory_info", &vmm_driver_memory_info);
  module.def("vmm_set_serial_driver_workers", &vmm_set_serial_driver_workers,
             py::arg("enabled"));
  module.def("vmm_get_serial_driver_workers", &vmm_get_serial_driver_workers);
  py::class_<VMMActivationSlot, std::shared_ptr<VMMActivationSlot>>(module, "VMMActivationSlot")
      .def(py::init<size_t, int>(), py::arg("requested_bytes"), py::arg("device") = 0)
      .def("tensor", &VMMActivationSlot::tensor)
      .def("set_slot_id", &VMMActivationSlot::set_slot_id, py::arg("slot_id"))
      // Deprecated synchronous interfaces (kept for compatibility)
      .def("unmap_and_release", &VMMActivationSlot::unmap_and_release)
      .def("create_and_remap", &VMMActivationSlot::create_and_remap)
      // New async-release interfaces
      .def("release_hook_after", &VMMActivationSlot::release_hook_after, py::arg("stream"))
      .def("wait_for_async_release", &VMMActivationSlot::wait_for_async_release)
      .def("async_release_done", &VMMActivationSlot::async_release_done)
      .def("wait_for_async_remap", &VMMActivationSlot::wait_for_async_remap)
      .def("adopt_async_remap", &VMMActivationSlot::adopt_async_remap)
      .def("async_remap_done", &VMMActivationSlot::async_remap_done)
      .def("info", &VMMActivationSlot::info)
      .def("close", &VMMActivationSlot::close);
  py::class_<ReleaseHookContext, std::shared_ptr<ReleaseHookContext>>(module,
                                                                      "VMMReleaseHookContext")
      .def(py::init<>())
      .def("status", &release_hook_status);
  py::class_<RemapHookContext, std::shared_ptr<RemapHookContext>>(module,
                                                                  "VMMRemapHookContext")
      .def(py::init<>())
      .def("status", &remap_hook_status)
      .def("wait_on_stream", &remap_wait_on_stream, py::arg("stream"))
      .def("enqueue_wait_on_stream", &remap_enqueue_wait_on_stream, py::arg("stream"))
      .def("enqueue_slot_waits", &remap_enqueue_slot_waits, py::arg("stream"))
      .def("wait_until_slot_submitted", &wait_until_slot_submitted,
           py::arg("slot_index"))
      .def("enqueue_slot_wait", &remap_enqueue_slot_wait,
           py::arg("slot_index"), py::arg("stream"))
      .def("launch_slot_h2d", &remap_slot_launch_h2d,
           py::arg("slot_index"), py::arg("stream"));
  module.def("release_hooks_after", &release_hooks_after,
             py::arg("slots"), py::arg("stream"));
  module.def("remap_hooks_after", &remap_hooks_after,
             py::arg("slots"), py::arg("stream"));
  module.def("remap_and_copy_after", &remap_and_copy_after,
             py::arg("slots"), py::arg("host_tensors"), py::arg("stream"));
  module.def("remap_and_copy_slot_after", &remap_and_copy_slot_after,
             py::arg("slot"), py::arg("host_tensor"), py::arg("stream"));
  module.def("remap_only_slot_after", &remap_only_slot_after,
             py::arg("slot"), py::arg("host_tensor"), py::arg("stream"));
}

}  // namespace transformer_engine::pytorch
