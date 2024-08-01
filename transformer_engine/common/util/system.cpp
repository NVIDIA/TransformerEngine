/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/system.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "../common.h"

namespace transformer_engine {

namespace {

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

}  // namespace

#define NVTE_INSTANTIATE_GETENV(T, default_value)              \
  template <>                                                  \
  T getenv<T>(const char *variable, const T &default_value_) { \
    return getenv_helper<T>(variable, default_value_);         \
  }                                                            \
  template <>                                                  \
  T getenv<T>(const char *variable) {                          \
    return getenv_helper<T>(variable, default_value);          \
  }
NVTE_INSTANTIATE_GETENV(bool, false);
NVTE_INSTANTIATE_GETENV(float, 0.f);
NVTE_INSTANTIATE_GETENV(double, 0.);
NVTE_INSTANTIATE_GETENV(int8_t, 0);
NVTE_INSTANTIATE_GETENV(int16_t, 0);
NVTE_INSTANTIATE_GETENV(int32_t, 0);
NVTE_INSTANTIATE_GETENV(int64_t, 0);
NVTE_INSTANTIATE_GETENV(uint8_t, 0);
NVTE_INSTANTIATE_GETENV(uint16_t, 0);
NVTE_INSTANTIATE_GETENV(uint32_t, 0);
NVTE_INSTANTIATE_GETENV(uint64_t, 0);
NVTE_INSTANTIATE_GETENV(std::string, std::string());
NVTE_INSTANTIATE_GETENV(std::filesystem::path, std::filesystem::path());

bool file_exists(const std::string &path) { return static_cast<bool>(std::ifstream(path.c_str())); }

}  // namespace transformer_engine
