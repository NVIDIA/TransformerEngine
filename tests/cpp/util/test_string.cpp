/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <string>

#include <gtest/gtest.h>

#include "util/string.h"

using namespace transformer_engine;

TEST(UtilTest, ToStringLike) {  // to_string_like
  // Strings
  using namespace std::string_literals;
  EXPECT_EQ(to_string_like(std::string("")), "");
  EXPECT_EQ(to_string_like(""), "");
  EXPECT_EQ(to_string_like(std::string("Hello")), "Hello");
  EXPECT_EQ(to_string_like("world!"), "world!");
  EXPECT_EQ(to_string_like(" \0\n\\\t\"\' This is a weird C++ string"s),
            " \0\n\\\t\"\' This is a weird C++ string"s);
  EXPECT_EQ(to_string_like("  Here is a bizarre C string \n\\\t\"\'"),
            "  Here is a bizarre C string \n\\\t\"\'");

  // Zero integer types
  EXPECT_EQ(to_string_like(19), "19");
  EXPECT_EQ(to_string_like(static_cast<char>(0)), "0");
  EXPECT_EQ(to_string_like(static_cast<unsigned char>(0)), "0");
  EXPECT_EQ(to_string_like(static_cast<short int>(0)), "0");
  EXPECT_EQ(to_string_like(static_cast<unsigned short int>(0)), "0");
  EXPECT_EQ(to_string_like(static_cast<int>(0)), "0");
  EXPECT_EQ(to_string_like(static_cast<unsigned int>(0)), "0");
  EXPECT_EQ(to_string_like(static_cast<long long int>(0)), "0");
  EXPECT_EQ(to_string_like(static_cast<unsigned long long int>(0)), "0");

  // Non-zero integer types
  EXPECT_EQ(to_string_like(static_cast<char>(1)), "1");
  EXPECT_EQ(to_string_like(static_cast<char>(-1)), "-1");
  EXPECT_EQ(to_string_like(static_cast<unsigned char>(2)), "2");
  EXPECT_EQ(to_string_like(static_cast<short>(3)), "3");
  EXPECT_EQ(to_string_like(static_cast<short>(-5)), "-5");
  EXPECT_EQ(to_string_like(static_cast<unsigned short>(8)), "8");
  EXPECT_EQ(to_string_like(static_cast<int>(13)), "13");
  EXPECT_EQ(to_string_like(static_cast<int>(-21)), "-21");
  EXPECT_EQ(to_string_like(static_cast<unsigned int>(34)), "34");
  EXPECT_EQ(to_string_like(static_cast<long long>(55)), "55");
  EXPECT_EQ(to_string_like(static_cast<long long>(-89)), "-89");
  EXPECT_EQ(to_string_like(static_cast<unsigned long long>(144)), "144");
  EXPECT_EQ(to_string_like(static_cast<size_t>(233)), "233");

  // Floating-point types
  EXPECT_EQ(std::stof(to_string_like(0.f)), 0.f);
  EXPECT_EQ(std::stod(to_string_like(0.)), 0.);
  EXPECT_EQ(std::stof(to_string_like(1.25f)), 1.25f);
  EXPECT_EQ(std::stof(to_string_like(-2.5f)), -2.5f);
  EXPECT_EQ(std::stod(to_string_like(2.25)), 2.25);
  EXPECT_EQ(std::stod(to_string_like(-4.5)), -4.5);
}

TEST(UtilTest, ConcatStringsTest) {  // concat_strings
  // Strings
  EXPECT_EQ(concat_strings(), "");
  EXPECT_EQ(concat_strings(std::string("")), "");
  EXPECT_EQ(concat_strings(""), "");
  EXPECT_EQ(concat_strings(std::string(""), "", std::string(""), ""), "");
  EXPECT_EQ(concat_strings("C string"), "C string");
  EXPECT_EQ(concat_strings(std::string("C++ string")), "C++ string");
  EXPECT_EQ(concat_strings("C string ", std::string("and"),
                           " ", std::string("C++ string")),
            "C string and C++ string");

  // Numeric types
  EXPECT_EQ(concat_strings("int ", static_cast<int>(-123),
                           ", uint ", static_cast<unsigned int>(456)),
            "int -123, uint 456");
  EXPECT_EQ(concat_strings("char ", static_cast<char>(13),
                           ", uchar ", static_cast<unsigned char>(24)),
            "char 13, uchar 24");
  EXPECT_EQ(concat_strings("int16 ", static_cast<short>(-35),
                           ", uint16 ", static_cast<unsigned short>(46)),
            "int16 -35, uint16 46");
  EXPECT_EQ(concat_strings("int64 ", static_cast<long long>(57),
                           ", uint64 ", static_cast<unsigned long long>(68)),
            "int64 57, uint64 68");
  EXPECT_EQ(std::stof(concat_strings("-", 3.25f)), -3.25f);
  EXPECT_EQ(std::stof(concat_strings(6.5f)), 6.5f);
  EXPECT_EQ(std::stod(concat_strings("-", 4.25)), -4.25);
  EXPECT_EQ(std::stod(concat_strings(8.5)), 8.5);
}

TEST(UtilTest, RegexReplaceTest) {  // regex_replace
  EXPECT_EQ(regex_replace("this test FAILS", "FAILS", "PASSES"),
            "this test PASSES");
  EXPECT_EQ(regex_replace("status = 0000", "0", 1), "status = 1111");
  EXPECT_EQ(regex_replace("I um sound um \t  very umconfident", R"(um\s*)", ""),
            "I sound very confident");
}
