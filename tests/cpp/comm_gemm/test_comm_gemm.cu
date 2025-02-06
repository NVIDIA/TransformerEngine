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

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  CHECK_MPI(MPI_Init(&argc, &argv));
  auto ret = RUN_ALL_TESTS();
  CHECK_MPI(MPI_Finalize());
  return ret;
}

TEST(TestCommGemm, Rank) {
  int rank = -1;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  EXPECT_EQ(rank, 0);
  nvte_comm_gemm();
}
