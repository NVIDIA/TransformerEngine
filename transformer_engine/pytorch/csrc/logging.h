/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_LOGGING_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_LOGGING_H_

#include <stdexcept>
#include <string>

#ifndef NVTE_ERROR
#define NVTE_ERROR(message)                                             \
  do {                                                                  \
    throw std::runtime_error(std::string(__FILE__ ":")                  \
                             + std::to_string(__LINE__)                 \
                             + " in function " + __func__ + ": "        \
                             + message);                                \
  } while (false)
#endif  // NVTE_ERROR

#ifndef NVTE_CHECK
#define NVTE_CHECK(expr, ...)                                   \
  do {                                                          \
    if (!(expr)) {                                              \
      NVTE_ERROR(std::string("Assertion failed: " #expr ". ")   \
                 + std::string(__VA_ARGS__));                   \
    }                                                           \
  } while (false)
#endif  // NVTE_CHECK

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_LOGGING_H_
