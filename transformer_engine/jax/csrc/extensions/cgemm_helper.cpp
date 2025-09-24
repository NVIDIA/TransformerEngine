/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cgemm_helper.h"

#include "common/util/system.h"
#include "nccl.h"

namespace transformer_engine {
namespace jax {

ncclUniqueId CommunicatorHandler::coordinate_nccl_unique_id(const std::string &id_type) {
  ncclUniqueId unique_id;

  int tp_domain_id = get_tp_domain_id();
  bool is_tp_leader = (get_local_device_id_within_tp_domain() == 0);

  pid_t pgid = getpgid(0);

  std::string base_path = getenv<std::string>("NVTE_JAX_NCCL_FILE_PATH", "/tmp");
  std::string id_file = base_path + "/nccl_" + id_type + "_unique_id_pgid_" + std::to_string(pgid) +
                        "_" + std::to_string(num_total_devices) + "_" + std::to_string(tp_size) +
                        "_domain_" + std::to_string(tp_domain_id) + ".bin";

  if (is_tp_leader) {
    NVTE_CHECK_NCCL(ncclGetUniqueId(&unique_id));

    // Write the ID to a temporary file
    std::ofstream file(id_file, std::ios::binary);
    NVTE_CHECK(file.is_open(), "Failed to create NCCL unique ID file: ", id_file);
    file.write(reinterpret_cast<const char *>(&unique_id), sizeof(ncclUniqueId));
    file.close();
  } else {
    // Wait for the ID file to be created and read it
    int attempts = 0;
    const int max_attempts = 100;
    while (attempts < max_attempts) {
      std::ifstream file(id_file, std::ios::binary);
      if (file.is_open()) {
        file.read(reinterpret_cast<char *>(&unique_id), sizeof(ncclUniqueId));
        if (file.gcount() == sizeof(ncclUniqueId)) {
          file.close();
          break;
        }
        file.close();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      attempts++;
    }
    NVTE_CHECK(attempts < max_attempts,
               "Timeout waiting for " + id_type + " NCCL unique ID file from leader: ", id_file);
  }

  if (is_tp_leader) {
    _nccl_id_file_name.push_back(id_file);
  }

  return unique_id;
}

void CommunicatorHandler::init(int num_total_devices, int num_devices_per_process, int process_id,
                               int tp_size) {
  // Validate inputs
  NVTE_CHECK(num_devices_per_process <= MAX_DEVICES,
             "num_devices_per_process exceeds MAX_DEVICES=", MAX_DEVICES,
             ", got num_devices_per_process=", num_devices_per_process);
  NVTE_CHECK(num_devices_per_process == 1,
             "num_devices_per_process must be == 1, got num_devices_per_process=",
             num_devices_per_process);
  NVTE_CHECK(num_total_devices >= 1,
             "num_total_devices must be >= 1, got num_total_devices=", num_total_devices);
  NVTE_CHECK(
      num_total_devices % num_devices_per_process == 0,
      "num_total_devices must be divisible by num_devices_per_process, got num_total_devices=",
      num_total_devices, ", num_devices_per_process=", num_devices_per_process);

  // Validate TP size
  NVTE_CHECK(tp_size > 0, "tp_size must be > 0, got tp_size=", tp_size);
  NVTE_CHECK(num_total_devices % tp_size == 0,
             "num_total_devices must be divisible by tp_size, got num_total_devices=",
             num_total_devices, ", tp_size=", tp_size);

  auto &handler = get(false);
  handler.num_total_devices = num_total_devices;
  handler.num_devices_per_process = num_devices_per_process;
  handler.process_id = process_id;
  handler.num_processes = num_total_devices / num_devices_per_process;
  handler.tp_size = tp_size;
  handler.tp_num_domains = num_total_devices / tp_size;

  NVTE_CHECK(0 <= process_id && process_id < handler.num_processes,
             "Invalid process_id=", process_id, ", which is out of range [0, ",
             handler.num_processes, ")");

  // Initialize local devices and calculate their global device IDs and TP topology
  for (int local_idx = 0; local_idx < num_devices_per_process; local_idx++) {
    // Use the device that JAX has already assigned to this process
    int current_device;
    NVTE_CHECK_CUDA(cudaGetDevice(&current_device));
    handler.local_device_ids_within_process[local_idx] = current_device;
    handler.global_device_ids[local_idx] = process_id * num_devices_per_process + local_idx;

    // Calculate TP-related values for this device
    int global_device_id = handler.global_device_ids[local_idx];
    if (num_devices_per_process == tp_size) {
      // Scenario 1: Multi-device per process - TP domain = single process
      handler.local_device_ids_within_tp_domain[local_idx] = local_idx;
      handler.tp_domain_ids[local_idx] = process_id;
    } else {
      // Scenario 2: Single device per process - TP domain spans multiple processes
      handler.local_device_ids_within_tp_domain[local_idx] = global_device_id % tp_size;
      handler.tp_domain_ids[local_idx] = global_device_id / tp_size;
    }
  }

  ncclUniqueId tp_id = handler.coordinate_nccl_unique_id("tp");

  NVTE_CHECK_NCCL(ncclGroupStart());
  for (int local_idx = 0; local_idx < num_devices_per_process; local_idx++) {
    NVTE_CHECK_CUDA(cudaSetDevice(handler.local_device_ids_within_process[local_idx]));
    int tp_local_rank = handler.local_device_ids_within_tp_domain[local_idx];
    NVTE_CHECK_NCCL(
        ncclCommInitRank(&handler.tp_comms[local_idx], handler.tp_size, tp_id, tp_local_rank));
  }
  NVTE_CHECK_NCCL(ncclGroupEnd());

  // Allocate device memory for barrier operations
  NVTE_CHECK_CUDA(cudaMalloc(&handler._device_barrier, sizeof(int)));

  handler._initialize = true;

  // Bootstrap UB via creating a dummy CommOverlapP2PBase object
  std::vector<size_t> buffer_shape{1, 1};
  auto _ = CollectiveGemmPlanRegistry::getInstance().get_executor(buffer_shape, DType::kFloat32,
                                                                  JAXX_Collective_Op::ALL_GATHER);
}

void InitializeCgemmCommunicator(int num_total_devices, int num_devices_per_process, int process_id,
                                 int tp_size, int num_max_streams, int gemm_priority,
                                 int comm_priority, int num_comm_sm, bool use_ce,
                                 bool aggregate_ag) {
  auto &config = CgemmConfig::get(false);
  config.init(num_max_streams, gemm_priority, comm_priority, num_comm_sm, use_ce, aggregate_ag);
  auto &handler = CommunicatorHandler::get(false);
  handler.init(num_total_devices, num_devices_per_process, process_id, tp_size);
}

int GetCgemmNumMaxStreams() {
  auto &config = CgemmConfig::get();
  return config.num_max_streams;
}

CommOverlapCore *CollectiveGemmPlanRegistry::get_executor(std::vector<size_t> buffer_shape,
                                                          DType dtype,
                                                          JAXX_Collective_Op collective_op) {
  auto &comm_handler = CommunicatorHandler::get();
  auto &cgemm_config = CgemmConfig::get();

  int device_idx = comm_handler.get_local_device_idx_for_current_device();
  int64_t plan_id = 0;
  hash_combine(plan_id, buffer_shape[0], buffer_shape[1], static_cast<size_t>(dtype),
               static_cast<int>(collective_op), comm_handler.tp_size, cgemm_config.num_max_streams,
               cgemm_config.gemm_priority, cgemm_config.comm_priority, cgemm_config.num_comm_sm,
               cgemm_config.use_ce, cgemm_config.aggregate_ag, device_idx);

  auto it = plan_map.find(plan_id);
  if (it != plan_map.end()) {
    return it->second.get();
  }

  if (comm_handler.num_devices_per_process == comm_handler.tp_size) {
    // Multi-device per process
  } else if (comm_handler.num_devices_per_process == 1) {
    // Single device per process
    NVTE_CHECK(comm_handler.num_total_devices % comm_handler.tp_size == 0,
               "For single device per process, num_total_devices must be divisible by tp_size, "
               "got num_total_devices=",
               comm_handler.num_total_devices, ", tp_size=", comm_handler.tp_size);
  } else {
    NVTE_ERROR("Unsupported TP configuration: num_devices_per_process=",
               comm_handler.num_devices_per_process, ", tp_size=", comm_handler.tp_size,
               ". Supported scenarios: "
               "(1) num_devices_per_process == tp_size (multi-device per process), "
               "(2) num_devices_per_process == 1 (single device per process)");
  }

  std::unique_ptr<CommOverlapCore> executor;
  executor = std::make_unique<CommOverlapP2PBase>(
      buffer_shape, dtype, comm_handler.get_global_rank(), comm_handler.num_total_devices,
      comm_handler.get_local_device_id_within_tp_domain(), comm_handler.tp_size,
      comm_handler.get_tp_domain_id(), comm_handler.get_tp_num_domains(), comm_handler.tp_size,
      comm_handler.allgather_func, comm_handler.barrier_func, get_nvte_collective_op(collective_op),
      cgemm_config.num_max_streams, 1 /*comm_cga_size*/, cgemm_config.gemm_priority,
      cgemm_config.comm_priority, cgemm_config.num_comm_sm, true /*set_sm_margin*/,
      cgemm_config.use_ce, false /*atomic_gemm*/, cgemm_config.aggregate_ag);

  CommOverlapCore *executor_ptr = executor.get();
  plan_map[plan_id] = std::move(executor);
  return executor_ptr;
}

void CommunicatorHandler::nccl_device_barrier_impl(ExtComm) {
  NVTE_CHECK(_initialize, "CommunicatorHandler must be initialized before using barrier");

  int device_idx = get_local_device_idx_for_current_device();
  ncclComm_t tp_comm = tp_comms[device_idx];

  NVTE_CHECK_NCCL(
      ncclAllReduce(_device_barrier, _device_barrier, 1, ncclInt, ncclSum, tp_comm, nullptr));
  cudaDeviceSynchronize();
}

void CommunicatorHandler::nccl_allgather_impl(void *output_buf, size_t output_bytes,
                                              void *input_buf, size_t input_bytes, ExtComm) {
  NVTE_CHECK(_initialize, "CommunicatorHandler must be initialized before using allgather");

  int device_idx = get_local_device_idx_for_current_device();
  ncclComm_t tp_comm = tp_comms[device_idx];

  size_t expected_output_bytes = input_bytes * tp_size;
  NVTE_CHECK(output_bytes == expected_output_bytes, "TP allgather buffer size mismatch: expected ",
             expected_output_bytes, ", got ", output_bytes);

  NVTE_CHECK_NCCL(ncclAllGather(input_buf, output_buf, input_bytes, ncclChar, tp_comm, nullptr));
  cudaDeviceSynchronize();
}

CommunicatorHandler::CommunicatorHandler() : _device_barrier(nullptr) {
  for (int i = 0; i < MAX_DEVICES; i++) {
    local_device_ids_within_process[i] = -1;
    local_device_ids_within_tp_domain[i] = -1;
    tp_domain_ids[i] = -1;
    global_device_ids[i] = -1;
    tp_comms[i] = nullptr;
  }

  allgather_func = [this](void *output_buf, size_t output_bytes, void *input_buf,
                          size_t input_bytes, ExtComm comm) {
    this->nccl_allgather_impl(output_buf, output_bytes, input_buf, input_bytes, comm);
  };
  barrier_func = [this](ExtComm comm) { this->nccl_device_barrier_impl(comm); };
}

CommunicatorHandler::~CommunicatorHandler() {
  if (_initialize) {
    for (int i = 0; i < num_devices_per_process; i++) {
      if (tp_comms[i] != nullptr) {
        ncclCommDestroy(tp_comms[i]);
      }
    }
  }
  if (_device_barrier) cudaFree(_device_barrier);

  for (const auto &file_path : _nccl_id_file_name) {
    std::remove(file_path.c_str());
  }
}

}  // namespace jax
}  // namespace transformer_engine
