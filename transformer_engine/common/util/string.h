/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_STRING_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_STRING_H_

#include <regex>  // NOLINT(*)
#include <string>
#include <type_traits>

namespace transformer_engine {

/*! \brief Convert to C-style or C++-style string */
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline std::string to_string_like(const T &val) {
  return std::to_string(val);
}

inline const std::string &to_string_like(const std::string &val) noexcept { return val; }

constexpr const char *to_string_like(const char *val) noexcept { return val; }

/*! \brief Convert arguments to strings and concatenate */
template <typename... Ts>
inline std::string concat_strings(const Ts &...args) {
  std::string str;
  str.reserve(1024);  // Assume strings are <1 KB
  (..., (str += to_string_like(args)));
  return str;
}

/*! \brief Substitute regex occurances in string
 *
 * This is a convenience wrapper around std::regex_replace.
 */
template <typename T>
inline std::string regex_replace(const std::string &str, const std::string &pattern,
                                 const T &replacement) {
  return std::regex_replace(str, std::regex(pattern), to_string_like(replacement));
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_STRING_H_
