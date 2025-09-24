/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CGEMM_HELPER_H_
#define TRANSFORMER_ENGINE_JAX_CGEMM_HELPER_H_

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>

#include "../extensions.h"
#include "common/comm_gemm_overlap/userbuffers/userbuffers.h"
#include "common/util/cuda_runtime.h"
#include "common/util/logging.h"
#include "transformer_engine/comm_gemm_overlap.h"

namespace transformer_engine {
namespace jax {

#ifndef MAX_DEVICES
#define MAX_DEVICES 8
#endif

// Configuration singleton for CGEMM parameters
class CgemmConfig {
 public:
  int num_max_streams;
  int gemm_priority;
  int comm_priority;
  int num_comm_sm;
  bool use_ce;
  bool aggregate_ag;

  static void init(int _num_max_streams, int _gemm_priority, int _comm_priority, int _num_comm_sm,
                   bool _use_ce, bool _aggregate_ag) {
    auto &config = get(false);
    config._initialized = true;
    config.num_max_streams = _num_max_streams;
    config.gemm_priority = _gemm_priority;
    config.comm_priority = _comm_priority;
    config.num_comm_sm = _num_comm_sm;
    config.use_ce = _use_ce;
    config.aggregate_ag = _aggregate_ag;
  }

  static CgemmConfig &get(bool is_initialized = true) {
    static thread_local CgemmConfig instance;
    NVTE_CHECK(
        instance._initialized == is_initialized,
        "CgemmConfig must be initialized before using it, got is_initialized=", is_initialized);
    return instance;
  }

  CgemmConfig(const CgemmConfig &) = delete;
  CgemmConfig &operator=(const CgemmConfig &) = delete;

 private:
  CgemmConfig() = default;
  ~CgemmConfig() = default;
  bool _initialized = false;
};

// Forward declaration
class CollectiveGemmPlanRegistry;

// NCCL communicator handler for collective GEMM operations
// Support both single process single device AND single process multi device
// Two scenarios:
// 1. Single process multiple devices: TP domain = process (num_devices_per_process == tp_size)
// 2. Single process single device: TP domain spans processes (num_devices_per_process == 1)
class CommunicatorHandler {
 public:
  int num_total_devices = -1;
  int num_devices_per_process = -1;
  int process_id = -1;
  int num_processes = -1;

  int tp_size = -1;
  int tp_num_domains = -1;
  int local_device_ids_within_tp_domain[MAX_DEVICES] = {-1};
  int tp_domain_ids[MAX_DEVICES] = {-1};
  ncclComm_t tp_comms[MAX_DEVICES];

  int local_device_ids_within_process[MAX_DEVICES];
  int global_device_ids[MAX_DEVICES];

  int get_global_rank() const {
    int device_idx = get_local_device_idx_for_current_device();
    return global_device_ids[device_idx];
  }

  void nccl_device_barrier_impl(ExtComm);
  void nccl_allgather_impl(void *output_buf, size_t output_bytes, void *input_buf,
                           size_t input_bytes, ExtComm);

  ncclComm_t get_comm_for_current_device() const {
    int device_idx = get_local_device_idx_for_current_device();
    return tp_comms[device_idx];
  }

  int get_local_device_idx_for_current_device() const {
    int current_device;
    NVTE_CHECK_CUDA(cudaGetDevice(&current_device));
    for (int i = 0; i < num_devices_per_process; i++) {
      if (local_device_ids_within_process[i] == current_device) {
        return i;
      }
    }
    NVTE_ERROR("Current CUDA device ", current_device,
               " not found in local_device_ids_within_process");
  }

  int get_local_device_id_within_tp_domain() const {
    int device_idx = get_local_device_idx_for_current_device();
    return local_device_ids_within_tp_domain[device_idx];
  }

  int get_tp_domain_id() const {
    int device_idx = get_local_device_idx_for_current_device();
    return tp_domain_ids[device_idx];
  }

  int get_tp_num_domains() const { return tp_num_domains; }

  static void init(int num_total_devices, int num_devices_per_process, int process_id, int tp_size);

 private:
  ncclUniqueId coordinate_nccl_unique_id(const std::string &id_type);

 public:
  static CommunicatorHandler &get(bool is_initialized = true) {
    static CommunicatorHandler instance;
    NVTE_CHECK(instance._initialize == is_initialized,
               "CommunicatorHandler._initialize=", instance._initialize,
               ", is_initialized=", is_initialized);
    return instance;
  }

  ExtAllgatherOp allgather_func;
  ExtBarrierOp barrier_func;

  CommunicatorHandler(const CommunicatorHandler &) = delete;
  CommunicatorHandler &operator=(const CommunicatorHandler &) = delete;

 private:
  CommunicatorHandler();
  ~CommunicatorHandler();

  bool _initialize = false;
  int *_device_barrier = nullptr;
  std::vector<std::string> _nccl_id_file_name;
};

// Plan registry for caching collective GEMM executors
class CollectiveGemmPlanRegistry {
 public:
  static CollectiveGemmPlanRegistry &getInstance() {
    static thread_local CollectiveGemmPlanRegistry instance;
    return instance;
  }

  CommOverlapCore *get_executor(std::vector<size_t> buffer_shape, DType dtype,
                                JAXX_Collective_Op collective_op);

 private:
  CollectiveGemmPlanRegistry() {}
  CollectiveGemmPlanRegistry(const CollectiveGemmPlanRegistry &) = delete;
  CollectiveGemmPlanRegistry &operator=(const CollectiveGemmPlanRegistry &) = delete;

  std::unordered_map<int64_t, std::unique_ptr<CommOverlapCore>> plan_map;
};

// Function declarations
void InitializeCgemmCommunicator(int num_total_devices, int num_devices_per_process, int process_id,
                                 int tp_size, int num_max_streams, int gemm_priority,
                                 int comm_priority, int num_comm_sm, bool use_ce,
                                 bool aggregate_ag);

int GetCgemmNumMaxStreams();

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CGEMM_HELPER_H_
