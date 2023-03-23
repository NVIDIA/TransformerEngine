/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_STRING_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_STRING_H_

#include <iostream>
#include <sstream>

namespace transformer_engine {

namespace detail {

inline void concat_strings_helper(std::ostringstream& oss) {}

template <typename HeadT, typename... TailTs>
inline void concat_strings_helper(std::ostringstream& oss,
                                  const HeadT& head,
                                  const TailTs&... tail) {
  oss << head;
  concat_strings_helper(oss, tail...);
}

}  // namespace detail

/// TODO Consider fast impl with reserved buffer size
template <typename... Ts>
inline std::string concat_strings(const Ts&... args) {
  std::ostringstream oss;
  detail::concat_strings_helper(oss, args...);
  return oss.str();
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_STRING_H_
