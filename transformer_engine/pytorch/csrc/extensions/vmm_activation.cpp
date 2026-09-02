/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../common.h"

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

void vmm_remap_trace(const char *message, double elapsed_ms = -1.0) {
  if (!vmm_remap_debug_enabled()) return;
  if (elapsed_ms >= 0.0) {
    std::fprintf(stderr, "[vmm-remap] %s elapsed_ms=%.3f\\n", message, elapsed_ms);
  } else {
    std::fprintf(stderr, "[vmm-remap] %s\\n", message);
  }
  std::fflush(stderr);
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
  MUdeviceptr address;
  size_t bytes;
  MUmemGenericAllocationHandle handle;
};

struct ReleaseHookContext {
  std::vector<ReleaseRequest> requests;
  std::atomic<int> done{0};          // 1 once unmap+release finished (or failed)
  std::atomic<int> error{0};         // 1 if any muMemUnmap/muMemRelease failed
  std::atomic<int> error_code{0};    // first failing MUresult
  std::atomic<int> callback_fired{0};
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
};

struct RemapRequest {
  MUdeviceptr address;
  size_t bytes;
  size_t copy_bytes{0};
  const void *host_address{nullptr};
  int device;
  std::shared_ptr<ReleaseHookContext> release_dependency;
  MUmemGenericAllocationHandle handle{0};
  bool mapped{false};
  bool copy_submitted{false};
};

struct RemapHookContext {
  std::vector<RemapRequest> requests;
  // Keep pinned CPU storage alive until every H2D copy submitted by the worker
  // has completed. ReloadCleanupWorker retains this context through that point.
  std::vector<at::Tensor> host_tensors;
  musaStream_t copy_stream{nullptr};
  musaEvent_t copy_done_event{nullptr};
  bool copy_enabled{false};
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
  ReleaseWorker() { thread_ = std::thread(&ReleaseWorker::run, this); }
  ~ReleaseWorker() { shutdown(); }
  ReleaseWorker(const ReleaseWorker &) = delete;
  ReleaseWorker &operator=(const ReleaseWorker &) = delete;

  // Driver calls run only on the dedicated worker thread, never on the
  // host-func callback thread.
  static void process(const std::shared_ptr<ReleaseHookContext> &context) {
    auto &ctx = *context;
    for (const auto &request : ctx.requests) {
      MUresult unmap_rc = muMemUnmap(request.address, request.bytes);
      if (unmap_rc != MUSA_SUCCESS && !ctx.error.load(std::memory_order_relaxed)) {
        ctx.error_code.store(static_cast<int>(unmap_rc), std::memory_order_relaxed);
        ctx.error.store(1, std::memory_order_relaxed);
      }
      MUresult release_rc = muMemRelease(request.handle);
      if (release_rc != MUSA_SUCCESS && !ctx.error.load(std::memory_order_relaxed)) {
        ctx.error_code.store(static_cast<int>(release_rc), std::memory_order_relaxed);
        ctx.error.store(1, std::memory_order_relaxed);
      }
    }
    complete_context(ctx);
  }

  void run() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      while (queue_.empty() && !shutdown_) cv_.wait(lock);
      // Drain everything already queued even when shutting down.
      if (queue_.empty() && shutdown_) return;
      auto context = std::move(queue_.front());
      queue_.pop_front();
      lock.unlock();
      process(context);
      lock.lock();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<ReleaseHookContext>> queue_;
  bool shutdown_ = false;
  std::thread thread_;  // declared last; started in the constructor body
};

class RemapWorker {
 public:
  static RemapWorker &instance() {
    static RemapWorker worker;
    return worker;
  }

  void enqueue(std::shared_ptr<RemapHookContext> context) {
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

  static void set_error(RemapHookContext &context, int error_code) {
    if (!context.error.exchange(1, std::memory_order_relaxed)) {
      context.error_code.store(error_code, std::memory_order_relaxed);
    }
  }

  static void cleanup_requests(RemapHookContext &context) {
    for (auto &request : context.requests) {
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

  static void process(const std::shared_ptr<RemapHookContext> &context) {
    auto &ctx = *context;
    auto trace_stage = [&](const char *stage, const auto &stage_started) {
      vmm_remap_trace(stage, std::chrono::duration<double, std::milli>(
          VmmClock::now() - stage_started).count());
    };
    vmm_remap_trace("worker start");
    bool any_copy_submitted = false;
    auto fail = [&](int error_code) {
      set_error(ctx, error_code);
      // A previously submitted copy may still access its new mapping. Finish
      // those copies before rolling the batch back on a later-slot failure.
      if (any_copy_submitted) musaStreamSynchronize(ctx.copy_stream);
      cleanup_requests(ctx);
      complete_context(ctx);
    };

    // Remap the full runner batch before submitting any H2D. The requests are
    // ordered by backward consumption, with the latest D2H/release dependency
    // first. Once that dependency completes, earlier groups on the shared D2H
    // stream are also releasable. Separating remap from copy keeps the H2D
    // submissions contiguous instead of inserting each following slot's VMM
    // driver latency between copies.
    for (auto &request : ctx.requests) {
      const auto request_started = VmmClock::now();
      if (request.release_dependency) {
        vmm_remap_trace("waiting for release dependency");
        wait_for_context(request.release_dependency);
        trace_stage("release dependency ready", request_started);
        if (request.release_dependency->error.load(std::memory_order_acquire)) {
          fail(request.release_dependency->error_code.load(std::memory_order_relaxed));
          return;
        }
      }
      ctx.releases_ready.fetch_add(1, std::memory_order_release);

      const auto properties = allocation_properties(request.device);
      const auto api_started = VmmClock::now();
      MUresult rc = muMemCreate(&request.handle, request.bytes, &properties, 0);
      trace_stage("muMemCreate", api_started);
      if (rc != MUSA_SUCCESS) {
        fail(static_cast<int>(rc));
        return;
      }
      const auto map_started = VmmClock::now();
      rc = muMemMap(request.address, request.bytes, 0, request.handle, 0);
      trace_stage("muMemMap", map_started);
      if (rc != MUSA_SUCCESS) {
        fail(static_cast<int>(rc));
        return;
      }
      request.mapped = true;
      MUmemAccessDesc access{};
      access.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = request.device;
      access.flags = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      const auto access_started = VmmClock::now();
      rc = muMemSetAccess(request.address, request.bytes, &access, 1);
      trace_stage("muMemSetAccess", access_started);
      if (rc != MUSA_SUCCESS) {
        fail(static_cast<int>(rc));
        return;
      }
      ctx.remaps_done.fetch_add(1, std::memory_order_release);
      trace_stage("request total", request_started);
    }

    if (ctx.copy_enabled) {
      for (auto &request : ctx.requests) {
        musaError_t runtime_rc = musaSetDevice(request.device);
        if (runtime_rc != MUSA_SUCCESS) {
          fail(static_cast<int>(runtime_rc));
          return;
        }
        runtime_rc = musaMemcpyAsync(
            reinterpret_cast<void *>(static_cast<uintptr_t>(request.address)),
            request.host_address, request.copy_bytes, musaMemcpyHostToDevice,
            ctx.copy_stream);
        if (runtime_rc != MUSA_SUCCESS) {
          fail(static_cast<int>(runtime_rc));
          return;
        }
        request.copy_submitted = true;
        any_copy_submitted = true;
        ctx.copies_submitted.fetch_add(1, std::memory_order_release);
      }
    }

    if (ctx.copy_enabled) {
      musaError_t rc = musaEventRecord(ctx.copy_done_event, ctx.copy_stream);
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
      process(context);
      lock.lock();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<RemapHookContext>> queue_;
  bool shutdown_ = false;
  std::thread thread_;
};

void release_host_func(void *user_data) {
  // The box keeps the context alive until the stream actually reaches this
  // host func, even if every Python/slot reference is gone before that.
  auto *boxed = static_cast<std::shared_ptr<ReleaseHookContext> *>(user_data);
  std::shared_ptr<ReleaseHookContext> context(std::move(*boxed));
  delete boxed;
  context->callback_fired.store(1, std::memory_order_release);
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
    return ReleaseRequest{address_, bytes_, handle_};
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
    drain_remap();
    drain_pending();
    ensure_reserved();
    if (!mapped_) throw std::runtime_error("VMM activation slot is already unmapped");
    auto context = std::make_shared<ReleaseHookContext>();
    context->requests.push_back(make_release_request());
    pending_ = context;
    launch_release_hook(context, reinterpret_cast<musaStream_t>(raw_stream));
    return context;
  }

  // Block until the in-flight async release finished and apply its state.
  // This compatibility API uses a condition variable and never polls or
  // synchronizes a MUSA stream/event.
  void wait_for_async_release() { drain_pending(); }

  bool async_remap_done() const {
    auto context = pending_remap_;
    if (!context) return true;
    return context->done.load(std::memory_order_acquire) == 1;
  }

  void wait_for_async_remap() { drain_remap(); }

  void adopt_async_remap() {
    auto context = pending_remap_;
    if (!context) return;
    if (context->done.load(std::memory_order_acquire) != 1) {
      throw std::runtime_error("VMM async remap submission is not complete");
    }
    drain_remap();
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
    wait_for_context(context);
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
};

// ---------------------------------------------------------------------------
// Python-facing batch helper: enqueue ONE host func after a whole batch of
// slots' D2H copies so the DMA burst stays contiguous; the callback hands all
// raw release work to the worker in one go.
// ---------------------------------------------------------------------------

std::shared_ptr<ReleaseHookContext> release_hooks_after(
    std::vector<std::shared_ptr<VMMActivationSlot>> slots, uintptr_t raw_stream) {
  // Ensure the worker thread exists before any host func can fire.
  ReleaseWorker::instance();
  auto context = std::make_shared<ReleaseHookContext>();
  for (auto &slot : slots) {
    if (!slot) throw std::runtime_error("VMM batch release hook got a null slot");
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
  std::unordered_set<VMMActivationSlot *> seen_slots;
  context->requests.reserve(slots.size());
  for (auto &slot : slots) {
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
    context->requests.push_back(RemapRequest{slot->address_, slot->bytes_, 0, nullptr,
                                             slot->device_, slot->pending_, 0, false,
                                             false});
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
  if (slots.size() != host_tensors.size()) {
    throw std::runtime_error("VMM remap-and-copy requires one host tensor per slot");
  }
  ReloadCleanupWorker::instance();
  RemapWorker::instance();
  auto context = std::make_shared<RemapHookContext>();
  context->copy_stream = reinterpret_cast<musaStream_t>(raw_stream);
  context->copy_enabled = true;
  context->host_tensors = std::move(host_tensors);
  check_musa_runtime(musaEventCreateWithFlags(&context->copy_done_event, musaEventDisableTiming),
                     "musaEventCreateWithFlags(remap-and-copy)");

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
    context->requests.push_back(RemapRequest{
        slot->address_, slot->bytes_, slot->requested_bytes_, host_tensor.data_ptr(),
        slot->device_, slot->pending_, 0, false, false});
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

void remap_wait_on_stream(const std::shared_ptr<RemapHookContext> &context,
                          uintptr_t raw_stream) {
  const auto wait_started = VmmClock::now();
  vmm_remap_trace("wait_on_stream begin");
  wait_for_context(context);
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
  check_musa_runtime(
      musaStreamWaitEvent(reinterpret_cast<musaStream_t>(raw_stream),
                          context->copy_done_event, 0),
      "musaStreamWaitEvent(remap-and-copy)");
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

void init_vmm_activation_extension(py::module_ &module) {
  module.def("vmm_driver_memory_info", &vmm_driver_memory_info);
  py::class_<VMMActivationSlot, std::shared_ptr<VMMActivationSlot>>(module, "VMMActivationSlot")
      .def(py::init<size_t, int>(), py::arg("requested_bytes"), py::arg("device") = 0)
      .def("tensor", &VMMActivationSlot::tensor)
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
      .def("wait_on_stream", &remap_wait_on_stream, py::arg("stream"));
  module.def("release_hooks_after", &release_hooks_after,
             py::arg("slots"), py::arg("stream"));
  module.def("remap_hooks_after", &remap_hooks_after,
             py::arg("slots"), py::arg("stream"));
  module.def("remap_and_copy_after", &remap_and_copy_after,
             py::arg("slots"), py::arg("host_tensors"), py::arg("stream"));
}

}  // namespace transformer_engine::pytorch
