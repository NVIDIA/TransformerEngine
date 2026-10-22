/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>
#include <transformer_engine/comm_gemm.h>

#include <cstdlib>
#include <string>

#define CHECK_MPI(expr)                                              \
  do {                                                               \
    int err = (expr);                                                \
    if (err != MPI_SUCCESS) {                                        \
      char err_str[MPI_MAX_ERROR_STRING + 1]{};                      \
      int len{};                                                     \
      MPI_Error_string(err, err_str, &len);                          \
      EXPECT_TRUE(false) << "MPI error: " << err << ": " << err_str; \
    }                                                                \
  } while (false)

#define CHECK_NCCL(expr)                                                             \
  do {                                                                               \
    ncclResult_t err = (expr);                                                       \
    if (err != ncclSuccess) {                                                        \
      EXPECT_TRUE(false) << "NCCL error: " << err << ": " << ncclGetErrorString(err); \
    }                                                                                \
  } while (false)

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  CHECK_MPI(MPI_Init(&argc, &argv));
  auto ret = RUN_ALL_TESTS();
  CHECK_MPI(MPI_Finalize());
  return ret;
}

TEST(CommGemmStreamLogging, LogsStreamIdToStdout) {
  int nranks{};
  int rank{};
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  ASSERT_EQ(cudaSetDevice(rank), cudaSuccess);

  ncclUniqueId id{};
  if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm{};
  CHECK_NCCL(ncclCommInitRank(&comm, nranks, id, rank));
  ASSERT_EQ(setenv("NVTE_LOG_COLLECTIVE_GEMM_STREAM_IDS", "1", 1), 0);

  testing::internal::CaptureStdout();
  NVTECommGemmCtx* ctx = nvte_comm_gemm_ctx_create(comm, nranks, rank);
  const std::string output = testing::internal::GetCapturedStdout();

  const std::string prefix =
      "NVTE collective GEMM stream ID (rank " + std::to_string(rank) + "): ";
  EXPECT_NE(output.find(prefix), std::string::npos) << "Captured stdout: " << output;

  nvte_comm_gemm_ctx_destroy(ctx);
  CHECK_NCCL(ncclCommDestroy(comm));
}
