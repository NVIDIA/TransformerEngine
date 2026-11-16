# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

# Also configure directly to test the policy on a host without CUDA or PyTorch.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(te_vmm_policy_tests LANGUAGES CXX)
    enable_testing()
endif()

find_package(Threads REQUIRED)
add_executable(test_vmm_mempool_core test_vmm_mempool_core.cpp)
target_compile_features(test_vmm_mempool_core PRIVATE cxx_std_17)
target_include_directories(test_vmm_mempool_core PRIVATE ../../..)
target_link_libraries(test_vmm_mempool_core PRIVATE Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_vmm_mempool_core PRIVATE -UNDEBUG -Wall -Wextra -Werror)
endif()
add_test(NAME vmm_mempool_core COMMAND test_vmm_mempool_core)
