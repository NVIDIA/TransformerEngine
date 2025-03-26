#include <gtest/gtest.h>
#include <mpi.h>
#include <transformer_engine/comm_gemm.h>

#define CHECK_MPI(expr)                                              \
  do {                                                               \
    int err = (expr);                                                \
    if (err != MPI_SUCCESS) {                                        \
      char err_str[MPI_MAX_ERROR_STRING + 1]{};                      \
      int _len{};                                                    \
      MPI_Error_string(err, err_str, &_len);                         \
      EXPECT_TRUE(false) << "MPI error: " << err << ": " << err_str; \
    }                                                                \
  } while (false)

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  CHECK_MPI(MPI_Init(&argc, &argv));
  auto ret = RUN_ALL_TESTS();
  CHECK_MPI(MPI_Finalize());
  return ret;
}

class CommGemmTest : public ::testing::Test {
 protected:
  CommGemmTest() {
    int nranks{};
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    int rank{};
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int local_device = rank;
    ctx_ = nvte_comm_gemm_ctx_create(nranks, rank, local_device);
  }
  ~CommGemmTest() { nvte_comm_gemm_ctx_destroy(ctx_); }

  CommGemmCtx* ctx_;
};

TEST_F(CommGemmTest, Rank) {
  NVTETensor a{};
  NVTETensor b{};
  NVTETensor d{};
  NVTETensor bias{};
  NVTETensor pre_gelu_out{};
  bool transa = false;
  bool transb = false;
  bool grad = false;
  bool accumulate = false;
  int comm_sm_count = 0;
  nvte_comm_gemm(ctx_, a, b, d, bias, pre_gelu_out, transa, transb, grad, accumulate,
                 comm_sm_count);
}
