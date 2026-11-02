#include "tract.h"

#include <algorithm>
#include <cmath>

namespace puhuja {

namespace {
constexpr float kPi = 3.14159265f;
// A smooth step from 0 to 1 over [x0, x1].
float ramp(float x, float x0, float x1) {
  const float t = std::clamp((x - x0) / std::max(x1 - x0, 1e-3f), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}
}  // namespace

float TractShape::area(float x) const {
  // The base: larynx tube, then pharynx, then mouth, in log area.
  const float lnLar = std::log(std::max(aLarynx, 0.05f));
  const float lnPh = std::log(std::max(aPharynx, 0.05f));
  const float lnOr = std::log(std::max(aOral, 0.05f));
  float lnA = lnLar + (lnPh - lnLar) * ramp(x, lLarynx - 0.5f, lLarynx + 0.5f);
  const float xv = xVelum * length;
  lnA += (lnOr - lnPh) * ramp(x, xv - 1.0f, xv + 1.0f);
  // The tongue constriction: a cosine^2 hump in log area toward ln(ac).
  const float d = (x - xc * length) / std::max(wc, 0.2f);
  if (std::fabs(d) < 1.0f) {
    const float g = std::cos(0.5f * kPi * d);
    lnA += (std::log(std::max(ac, 0.05f)) - lnA) * g * g;
  }
  // The lips: the last lLip cm at aLip, with a half-cm blend.
  const float lnLip = std::log(std::max(aLip, 0.05f));
  lnA += (lnLip - lnA) * ramp(x, length - lLip - 0.5f, length - lLip + 0.5f);
  return std::exp(lnA);
}

std::vector<float> TractShape::sections(int n) const {
  std::vector<float> a(n);
  const float dx = length / n;
  for (int k = 0; k < n; ++k) {
    float sum = 0;
    for (int s = 0; s < 8; ++s) sum += area((k + (s + 0.5f) / 8.0f) * dx);
    a[k] = sum / 8.0f;
  }
  return a;
}

}  // namespace puhuja
