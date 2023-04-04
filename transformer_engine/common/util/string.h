/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_STRING_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_STRING_H_

#include <string>
#include <type_traits>

namespace transformer_engine {

namespace detail {

// Helper function that converts to a type compatible with
// std::string::operator+=

template <typename T,
          typename = typename std::enable_if<!std::is_arithmetic<T>::value>::type>
inline const T& to_string_like(const T& val) {
  return val;
}

template <typename T,
          typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline std::string to_string_like(const T &val) {
  return std::to_string(val);
}

}  // namespace detail

/*! \brief Convert arguments to string and concatenate */
template <typename... Ts>
inline std::string concat_strings(const Ts &... args) {
  std::string str;
  str.reserve(1024);
  (..., (str += detail::to_string_like(args)));
  return str;
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_STRING_H_
