// Fast float approximations for the busy loops (Jyrki 2026-11-13: losing a few float bits for speed is fine).
// Branch-free and call-free, so the loops around them vectorize.
#ifndef PUHUJA_FASTMATH_H_
#define PUHUJA_FASTMATH_H_

#include <cmath>
#include <cstdint>
#include <cstring>

namespace puhuja {

// exp(x) for the log areas (|x| < ~10): 2^n x a degree-6 polynomial on [-ln2/2, ln2/2]; relative error ~2e-7.
// Branch-free and call-free, so the sections loop vectorizes (AVX-512: 16 points per instruction).
inline float fastExp(float x) {
  const float n = std::nearbyint(x * 1.44269504f);
  const float r = x - n * 0.693145752f - n * 1.42860677e-6f;
  float p = 1.0f / 720.0f;
  p = p * r + 1.0f / 120.0f;
  p = p * r + 1.0f / 24.0f;
  p = p * r + 1.0f / 6.0f;
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  p = p * r + 1.0f;
  const int32_t e = static_cast<int32_t>(n) + 127;
  float scale;
  const int32_t bits = e << 23;
  std::memcpy(&scale, &bits, 4);
  return p * scale;
}
// log(1 + e) for e in [0, 1]: 2 atanh(t) with t = e / (2 + e) in [0, 1/3], the odd series to t^13 (relative
// error < 1e-7).
inline float fastLog1pUnit(float e) {
  const float t = e / (2.0f + e);
  const float t2 = t * t;
  float p = 1.0f / 13.0f;
  p = p * t2 + 1.0f / 11.0f;
  p = p * t2 + 1.0f / 9.0f;
  p = p * t2 + 1.0f / 7.0f;
  p = p * t2 + 1.0f / 5.0f;
  p = p * t2 + 1.0f / 3.0f;
  p = p * t2 + 1.0f;
  return 2.0f * t * p;
}

}  // namespace puhuja

#endif  // PUHUJA_FASTMATH_H_
