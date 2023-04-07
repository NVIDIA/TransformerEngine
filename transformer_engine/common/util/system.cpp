/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "../common.h"
#include "../util/system.h"

namespace transformer_engine {

template <>
std::string getenv<std::string>(const std::string &variable,
                                const std::string &default_value) {
  const char *env = std::getenv(variable.c_str());
  if (env == nullptr || env[0] == '\0') {
    return default_value;
  } else {
    return env;
  }
}

template <>
std::string getenv<std::string>(const std::string &variable) {
  return getenv<std::string>(variable, "");
}

template <typename T>
T getenv(const std::string &variable, const T &default_value) {
  const char *env = std::getenv(variable.c_str());
  if (env == nullptr || env[0] == '\0') {
    return default_value;
  }
  T value;
  std::istringstream iss(env);
  iss >> value;
  NVTE_CHECK(iss, "Invalid environment variable value");
  return value;
}

#define NVTE_INSTANTIATE_GETENV(T, default_value)               \
  template T getenv<T>(const std::string &, const T &);         \
  template <> T getenv<T>(const std::string &variable) {        \
    return getenv<T>(variable, default_value);                  \
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

bool file_exists(const std::string &path) {
  return static_cast<bool>(std::ifstream(path.c_str()));
}

}  // namespace transformer_engine
