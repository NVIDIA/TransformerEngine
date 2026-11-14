/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Host IEEE-754 model of compute_scale_from_amax without fast-math.

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

float bits_to_f32(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t f32_to_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool is_subnormal(float value) {
  const uint32_t bits = f32_to_bits(value);
  return (bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u;
}

bool is_finite_normal(float value) {
  return std::isfinite(value) && !is_subnormal(value) && value != 0.f;
}

struct Scales {
  float scale;
  float scale_inv;
};

Scales compute_scale(float amax, float max_fp8, bool force_pow_2, bool floor_tiny,
                     float epsilon) {
  if (amax < epsilon) {
    amax = epsilon;
  }
  float scale = 1.f;
  const bool early = std::isinf(amax) || std::isnan(amax) || amax == 0.f;
  if (!early) {
    if (floor_tiny) {
      const float amax_floor = max_fp8 * FLT_MIN;
      if (amax < amax_floor) {
        amax = amax_floor;
      }
    }
    scale = max_fp8 / amax;
    if (std::isinf(scale)) {
      scale = std::numeric_limits<float>::max();
    }
    if (force_pow_2) {
      uint32_t scale_bits = f32_to_bits(scale);
      scale_bits &= 0xFF800000u;
      scale = bits_to_f32(scale_bits);
    }
  }
  Scales out;
  out.scale = scale;
  out.scale_inv = 1.0f / scale;
  return out;
}

void require(bool condition, const std::string &message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

void emit(const char *fmt, int pow2, const char *label, float amax, const Scales &fixed,
          const Scales &legacy) {
  std::printf("%s %d %s %08x %08x %08x %08x %08x\n", fmt, pow2, label, f32_to_bits(amax),
              f32_to_bits(fixed.scale), f32_to_bits(fixed.scale_inv), f32_to_bits(legacy.scale),
              f32_to_bits(legacy.scale_inv));
}

void check_format(const char *name, float max_fp8, uint32_t expected_threshold_bits) {
  const float threshold = max_fp8 * FLT_MIN;
  require(f32_to_bits(threshold) == expected_threshold_bits, std::string(name) + " threshold bits");
  require(static_cast<double>(threshold) / static_cast<double>(max_fp8) ==
              static_cast<double>(FLT_MIN),
          std::string(name) + " exact quotient");

  const float below = bits_to_f32(expected_threshold_bits - 1u);
  const float above = bits_to_f32(expected_threshold_bits + 1u);
  require(static_cast<double>(below) / static_cast<double>(max_fp8) < static_cast<double>(FLT_MIN),
          std::string(name) + " -1 ULP is mathematically below FLT_MIN");
  require(static_cast<double>(above) / static_cast<double>(max_fp8) > static_cast<double>(FLT_MIN),
          std::string(name) + " +1 ULP is mathematically above FLT_MIN");

  const float tiny = 7.7e-37f;
  require(tiny > 0.f && tiny < threshold, std::string(name) + " 7.7e-37 is below the guard");

  const float samples[] = {0.f, tiny, below, threshold, above};
  const char *labels[] = {"zero", "tiny_7_7e_37", "minus_1ulp", "exact", "plus_1ulp"};

  for (int pow2 = 0; pow2 <= 1; ++pow2) {
    for (int i = 0; i < 5; ++i) {
      const Scales fixed = compute_scale(samples[i], max_fp8, pow2, true, 0.f);
      const Scales legacy = compute_scale(samples[i], max_fp8, pow2, false, 0.f);
      emit(name, pow2, labels[i], samples[i], fixed, legacy);

      if (i == 0) {
        require(fixed.scale == 1.f && fixed.scale_inv == 1.f, std::string(name) + " zero scale");
        require(fixed.scale == legacy.scale && fixed.scale_inv == legacy.scale_inv,
                std::string(name) + " zero matches legacy");
      } else if (i < 3) {
        // Below the floor, scale is 2^126 and scale_inv is FLT_MIN.
        require(f32_to_bits(fixed.scale) == 0x7e800000u, std::string(name) + " floored scale 2^126");
        require(f32_to_bits(fixed.scale_inv) == 0x00800000u, std::string(name) + " floored inv FLT_MIN");
        require(samples[i] * fixed.scale > 1.f, std::string(name) + " tiny value stays above 1");
        // Power-of-two rounding already collapses the -1 ULP sample onto 2^126.
        if (pow2 && i == 2) {
          require(fixed.scale == legacy.scale && fixed.scale_inv == legacy.scale_inv,
                  std::string(name) + " pow2 -1 ULP matches legacy");
        } else {
          require(is_subnormal(legacy.scale_inv), std::string(name) + " legacy inv subnormal");
          require(f32_to_bits(fixed.scale_inv) != f32_to_bits(legacy.scale_inv),
                  std::string(name) + " floor changes the inverse");
        }
      } else if (i == 3) {
        require(f32_to_bits(fixed.scale) == 0x7e800000u, std::string(name) + " exact scale 2^126");
        require(f32_to_bits(fixed.scale_inv) == 0x00800000u, std::string(name) + " exact inv FLT_MIN");
        require(fixed.scale == legacy.scale && fixed.scale_inv == legacy.scale_inv,
                std::string(name) + " exact matches legacy");
      } else {
        require(is_finite_normal(fixed.scale) && is_finite_normal(fixed.scale_inv),
                std::string(name) + " +1 ULP normal");
        require(fixed.scale_inv >= FLT_MIN, std::string(name) + " +1 ULP inv >= FLT_MIN");
        require(fixed.scale == legacy.scale && fixed.scale_inv == legacy.scale_inv,
                std::string(name) + " +1 ULP matches legacy");
      }
    }
  }

  // Epsilon above the guard keeps the historical floor behavior.
  const float epsilon = 1.0e-6f;
  const Scales lifted = compute_scale(tiny, max_fp8, false, true, epsilon);
  require(lifted.scale == max_fp8 / epsilon, "epsilon floor still applies before the guard");

  // Normal magnitudes, zeros, inf, and NaN stay bitwise-identical.
  const float normals[] = {1.0e-20f, 1.0e-5f, 0.125f, 1.f, 448.f, 1.0e10f,
                           std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN(), -0.f, 0.f};
  for (float amax : normals) {
    for (int pow2 = 0; pow2 <= 1; ++pow2) {
      const Scales fixed = compute_scale(amax, max_fp8, pow2, true, 0.f);
      const Scales legacy = compute_scale(amax, max_fp8, pow2, false, 0.f);
      require(f32_to_bits(fixed.scale) == f32_to_bits(legacy.scale), "normal scale bits");
      require(f32_to_bits(fixed.scale_inv) == f32_to_bits(legacy.scale_inv), "normal inv bits");
    }
  }
}

}  // namespace

int main() {
  std::printf("THRESHOLD e4m3 %08x\n", f32_to_bits(448.0f * FLT_MIN));
  std::printf("THRESHOLD e5m2 %08x\n", f32_to_bits(57344.0f * FLT_MIN));
  std::printf("FLT_MIN %08x\n", f32_to_bits(FLT_MIN));
  std::printf("TINY %08x\n", f32_to_bits(7.7e-37f));
  check_format("e4m3", 448.0f, 0x04e00000u);
  check_format("e5m2", 57344.0f, 0x08600000u);
  std::printf("HOST_OK\n");
  return 0;
}
