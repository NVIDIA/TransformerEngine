/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_SYSTEM_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_SYSTEM_H_

#include <string>

#include "../common.h"

namespace transformer_engine {

/*! \brief Get environment variable and convert to type
 *
 * If the environment variable is unset or empty, a falsy value is
 * returned.
 */
template <typename T = std::string>
T getenv(const char *variable);

/*! \brief Get environment variable and convert to type */
template <typename T = std::string>
T getenv(const char *variable, const T &default_value);

/*! \brief Check if a file exists and can be read */
bool file_exists(const std::string &path);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_SYSTEM_H_
