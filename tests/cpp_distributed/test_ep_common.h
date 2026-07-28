/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Shared TE EP test infrastructure. Include once per TU; ep_bootstrap() in
 * each test binary's main() populates process-level globals.
 * Defaults: 4 experts/rank, hidden_dim=256, max_tokens_per_rank=64.
 */
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <transformer_engine/comm_window.h>
#include <transformer_engine/ep.h>
#include <transformer_engine/transformer_engine.h>
#include "../cpp/test_common.h"
#include "util/logging.h"

using transformer_engine::DType;
using transformer_engine::TensorWrapper;

#define CHECK_MPI(expr)                                            \
  do {                                                             \
    int _err_mpi = (expr);                                         \
    NVTE_CHECK(_err_mpi == MPI_SUCCESS, "MPI error: ", _err_mpi);  \
  } while (false)

// -- Process-level state -------------------------------------------------------

static int         g_process_id          = -1;
static int         g_num_processes       = -1;

static int         g_sm_major            = -1;   // set by ep_bootstrap; -1 until then
static int         g_ep_size             = -1;
static int         g_num_experts         = -1;
static int         g_hidden_dim          = 256;
static int         g_max_tokens_per_rank = 64;
static NVTEDType   g_max_token_dtype     = kNVTEFloat32;  // staging-buffer sizing
static bool        g_ep_initialized      = false;
static ncclComm_t  g_ep_comm             = nullptr;  // owned by harness, destroyed in ep_teardown

// RAII owner for a cudaMalloc'd device buffer; element-count API on top of
// test::CudaPtr.
template <typename T>
struct DevBuf {
  test::CudaPtr<T> ptr;
  size_t count = 0;

  DevBuf() = default;
  explicit DevBuf(size_t n) { alloc(n); }

  void alloc(size_t n) {
    count = n;
    ptr = (n > 0) ? test::cuda_alloc<T>(n * sizeof(T)) : test::CudaPtr<T>{};
  }
  void reset() {
    ptr.reset();
    count = 0;
  }

  T* get() const { return ptr.get(); }
  size_t bytes() const { return count * sizeof(T); }
};

// -- Shared routing helper -----------------------------------------------------

// Balanced round-robin routing: token t on rank r maps top_k experts to
//   (r * num_tokens * top_k + t * top_k + k) % num_experts
// i.e. a single global counter over all (rank, t, k) triples mod num_experts.
static inline std::vector<int64_t> routing_balanced(
    int rank, int num_tokens, int top_k, int num_experts, int /*num_local_experts*/) {
  std::vector<int64_t> idx(num_tokens * top_k);
  for (int t = 0; t < num_tokens; ++t)
    for (int k = 0; k < top_k; ++k)
      idx[t * top_k + k] = (rank * num_tokens * top_k + t * top_k + k) % num_experts;
  return idx;
}

// -- ncclUniqueId exchange via MPI ---------------------------------------------

static void exchange_unique_id(ncclUniqueId* uid) {
  if (g_process_id == 0) NVTE_CHECK_NCCL(ncclGetUniqueId(uid));
  CHECK_MPI(MPI_Bcast(uid, sizeof(*uid), MPI_BYTE, 0, MPI_COMM_WORLD));
}

// -- CLI parsing ---------------------------------------------------------------

static void ep_parse_args(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a.rfind("--max-token-dtype=", 0) == 0)
      g_max_token_dtype = static_cast<NVTEDType>(std::stoi(a.substr(18)));
  }
}

// -- Bootstrap / teardown ------------------------------------------------------

// Returns false if the binary should exit without running tests (wrong SM, etc.).
static bool ep_bootstrap(int argc, char* argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) CHECK_MPI(MPI_Init(&argc, &argv));
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &g_process_id));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &g_num_processes));

  ep_parse_args(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  int device_count;
  cudaGetDeviceCount(&device_count);
  cudaSetDevice(g_process_id % device_count);

  int device, major;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  g_sm_major = major;
  if (major < 9) {
    if (g_process_id == 0)
      printf("SKIP: EP requires SM_90+ (device is SM_%d0)\n", major);
    return false;
  }
  if (g_num_processes < 2) {
    if (g_process_id == 0)
      printf("SKIP: at least 2 processes required\n");
    return false;
  }

  g_ep_size    = g_num_processes;
  g_num_experts = g_ep_size * 4;  // 4 experts per rank

  ncclUniqueId uid{};
  exchange_unique_id(&uid);

  NVTEEpGroupConfig group_config = NVTE_EP_GROUP_CONFIG_INIT;
  group_config.ep_size                  = g_ep_size;
  group_config.num_experts              = g_num_experts;
  group_config.max_tokens_per_rank      = g_max_tokens_per_rank;
  // Worst-case for top_k fan-out: ep_size * max_tokens_per_rank * 2.
  group_config.max_recv_tokens_per_rank = g_ep_size * g_max_tokens_per_rank * 2;
  group_config.hidden_dim               = g_hidden_dim;
  group_config.max_token_dtype          = g_max_token_dtype;

  NVTE_CHECK_NCCL(ncclCommInitRank(&g_ep_comm, g_num_processes, uid, g_process_id));
  nvte_ep_initialize(static_cast<void*>(g_ep_comm), &group_config);

  if (g_process_id == 0) {
    printf("EP initialized: ep_size=%d num_experts=%d "
           "hidden_dim=%d max_tokens_per_rank=%d\n",
           g_ep_size, g_num_experts, g_hidden_dim, g_max_tokens_per_rank);
  }

  g_ep_initialized = true;
  return true;
}

// Re-bootstrap the EP backend on the existing g_ep_comm with a new zero_copy
// setting.
static void ep_reinitialize(int zero_copy) {
  if (!g_ep_initialized) return;
  nvte_ep_shutdown();
  NVTEEpGroupConfig group_config = NVTE_EP_GROUP_CONFIG_INIT;
  group_config.ep_size                  = g_ep_size;
  group_config.num_experts              = g_num_experts;
  group_config.max_tokens_per_rank      = g_max_tokens_per_rank;
  group_config.max_recv_tokens_per_rank = g_ep_size * g_max_tokens_per_rank * 2;
  group_config.hidden_dim               = g_hidden_dim;
  group_config.max_token_dtype          = g_max_token_dtype;
  group_config.zero_copy                = zero_copy;
  nvte_ep_initialize(static_cast<void*>(g_ep_comm), &group_config);
}

// Tear down in dependency order: backend's ep_group reads from ep_comm,
// so destroy the group first, then the comm.
static void ep_teardown() {
  if (g_ep_initialized) {
    nvte_ep_shutdown();
    if (g_ep_comm != nullptr) {
      ncclCommDestroy(g_ep_comm);
      g_ep_comm = nullptr;
    }
    g_ep_initialized = false;
  }
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) MPI_Finalize();
}
