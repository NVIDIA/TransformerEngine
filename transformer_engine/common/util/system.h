/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_SYSTEM_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_SYSTEM_H_

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "logging.h"

namespace transformer_engine {

namespace detail {

/*! \brief Template specialization to get the env var for numeric data types */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type getenv_helper(
    const char *variable, const T &default_value) {
  // Implementation for numeric types
  const char *env = std::getenv(variable);
  if (env == nullptr || env[0] == '\0') {
    return default_value;
  }
  T value;
  std::istringstream iss(env);
  iss >> value;
  NVTE_CHECK(iss, "Invalid environment variable value");
  return value;
}

/*! \brief Template specialization to get the env var for string-like data types */
template <typename T>
inline typename std::enable_if<!std::is_arithmetic<T>::value, T>::type getenv_helper(
    const char *variable, const T &default_value) {
  // Implementation for string-like types
  const char *env = std::getenv(variable);
  if (env == nullptr || env[0] == '\0') {
    return default_value;
  } else {
    return env;
  }
}

/*! \brief Template specialization to get the default values for different
* numeric data types
*/
template <typename T>
inline T getenv_default_value() {
  return 0;
}

/*! \brief Template specialization to get the default values for bool */
template <>
inline bool getenv_default_value<bool>() {
  return false;
}

/*! \brief Template specialization to get the default values for string */
template <>
inline std::string getenv_default_value<std::string>() {
  return std::string();
}

/*! \brief Template specialization to get the default values for filesystem
* path data type */
template <>
inline std::filesystem::path getenv_default_value<std::filesystem::path>() {
  return std::filesystem::path();
}

}  // namespace detail

/*! \brief Get environment variable and convert to type
 *
 * If the environment variable is unset or empty, a falsy value is
 * returned.
*/
template <typename T = std::string>
inline T getenv(const char *variable) {
  return detail::getenv_helper<T>(variable, detail::getenv_default_value<T>());
}

/*! \brief Get environment variable and convert to type */
template <typename T = std::string>
inline T getenv(const char *variable, const T &default_value) {
  return detail::getenv_helper<T>(variable, default_value);
}

inline bool file_exists(const std::string &path) {
  return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

}  // namespace transformer_engine
#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_SYSTEM_H_
