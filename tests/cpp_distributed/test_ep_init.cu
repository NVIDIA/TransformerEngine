/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Unit tests for EP initialization paths.
 *
 * Tests:
 *   EPInitTest/InitPath           — backend is live after init, handle_mem_size > 0
 *   EPInitTest/NumLocalExperts    — handle_mem_size is consistent across num_local_experts values
 *
 * Run via run_test_ep.sh (both uid and comm init paths are tested by the script).
 */

#include "test_ep_common.h"

// ── Fixture ───────────────────────────────────────────────────────────────────

class EPInitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (g_sm_major < 9)
      GTEST_SKIP() << "EP requires SM_90+ (device is SM_" << g_sm_major << "0)";
    ASSERT_GE(g_num_processes, 2) << "EP tests require at least 2 processes";
    ASSERT_TRUE(g_ep_initialized) << "EP not initialized";
  }
};

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_F(EPInitTest, InitPath) {
  int nle = g_num_experts / g_ep_size;
  NVTEEpLayerConfig cfg{nle, /*top_k=*/2};
  size_t sz = 0;
  (void)nvte_ep_register_layer(cfg, &sz);
  ASSERT_GT(sz, 0u) << "handle_mem_size must be > 0 after init";

  if (g_process_id == 0) {
    printf("  handle_mem   : %zu bytes\n", sz);
  }
}

TEST_F(EPInitTest, NumLocalExperts) {
  // handle_mem_size should be > 0 for any valid num_local_experts value.
  for (int nle : {1, g_num_experts / g_ep_size}) {
    NVTEEpLayerConfig cfg{nle, /*top_k=*/2};
    size_t sz = 0;
    (void)nvte_ep_register_layer(cfg, &sz);
    ASSERT_GT(sz, 0u) << "num_local_experts=" << nle;
    if (g_process_id == 0)
      printf("  nle=%-3d  handle_mem_size=%zu bytes\n", nle, sz);
  }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (!ep_bootstrap(argc, argv)) return 0;
  int ret = RUN_ALL_TESTS();
  ep_teardown();
  return ret;
}
