#include "tract.h"

#include <algorithm>
#include <cmath>

namespace puhuja {

namespace {
constexpr float kPi = 3.14159265f;
// A smooth step from 0 to 1 over [x0, x1]; its width is computed once per shape.
struct Ramp {
  float x0 = 0, w = 1;
  Ramp() = default;
  Ramp(float a, float b) : x0(a), w(std::max(b - a, 1e-3f)) {}
  float operator()(float x) const {
    const float t = std::clamp((x - x0) / w, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
  }
};

// Everything of TractShape::area that does not depend on x: built once per shape (sections() calls area 8 n
// times), the same expressions in the same order, so bit-identical to computing them at every point. The 7 logs and
// the ramp edges per point were ~45 % of a render's CPU (gprof, 2026-11-13); this cut a render's time by ~38 %.
struct AreaPre {
  float lnLar, lnPh, lnOr, lnHy = 0, lnFront = 0, lnAc, lnLip, xcL, wcM, scM;
  bool hypo, front, hump;
  Ramp lar, hyIn, hyOut, velum, frIn, frOut, lip;
  explicit AreaPre(const TractShape& s) {
    lnLar = std::log(std::max(s.aLarynx, 0.05f));
    lnPh = std::log(std::max(s.aPharynx, 0.05f));
    lnOr = std::log(std::max(s.aOral, 0.05f));
    lar = Ramp(s.lLarynx - 0.5f, s.lLarynx + 0.5f);
    hypo = s.lHypo > 0 && s.aHypo > 0;
    if (hypo) {
      hyIn = Ramp(s.lLarynx - 0.25f, s.lLarynx + 0.25f);
      hyOut = Ramp(s.lLarynx + s.lHypo - 0.5f, s.lLarynx + s.lHypo + 0.5f);
      lnHy = std::log(s.aHypo);
    }
    const float xv = s.xVelum * s.length;
    velum = Ramp(xv - 1.0f, xv + 1.0f);
    front = s.lFront > 0 && s.mFront > 1.0f;
    if (front) {
      const float xe = s.length - s.lLip;
      const float hb = std::min(0.5f, 0.5f * s.lFront);
      lnFront = std::log(s.mFront);
      frIn = Ramp(xe - s.lFront - hb, xe - s.lFront + hb);
      frOut = Ramp(xe - 0.25f, xe + 0.25f);
    }
    xcL = s.xc * s.length;
    wcM = std::max(s.wc, 0.2f);
    hump = s.sc > 0;
    scM = std::min(s.sc, 1.0f);
    lnAc = std::log(std::max(s.ac, 0.05f));
    lnLip = std::log(std::max(s.aLip, 0.05f));
    lip = Ramp(s.length - s.lLip - 0.5f, s.length - s.lLip + 0.5f);
  }
  float area(float x) const {
    // The base: larynx tube, then pharynx, then mouth, in log area.
    float lnA = lnLar + (lnPh - lnLar) * lar(x);
    // The hypopharynx: past the larynx tube the airway opens to aHypo over lHypo before the tongue root's pharynx
    // (Jyrki 2026-11-11: "right after the constriction the wind pipe is about 3-4x larger in surface area than at the
    // constriction"). It only widens: a pharynx already wider stays.
    if (hypo) {
      const float w = hyIn(x) * (1.0f - hyOut(x));
      if (lnHy > lnA) lnA += (lnHy - lnA) * w;
    }
    lnA += (lnOr - lnPh) * velum(x);
    // The space right behind the teeth (Jyrki 2026-11-11: "perhaps adding some space in the wind pipe right before the
    // teeth would also give a more natural color to the sound"): over the lFront cm before the lip tube the area grows
    // x mFront, with blends of up to half a cm. Before the tongue's hump, so a constriction there (t, s, n) still closes.
    if (front) lnA += lnFront * frIn(x) * (1.0f - frOut(x));
    // The tongue constriction: a cosine^2 hump in log area toward ln(ac). Its strength sc scales the hump, so a
    // constriction grows and fades in place instead of travelling in from wherever the last one was.
    const float d = (x - xcL) / wcM;
    if (std::fabs(d) < 1.0f && hump) {
      const float g = std::cos(0.5f * kPi * d);
      lnA += scM * (lnAc - lnA) * g * g;
    }
    // The lips: the last lLip cm at aLip, with a half-cm blend.
    lnA += (lnLip - lnA) * lip(x);
    return std::exp(lnA);
  }
};
}  // namespace

float TractShape::area(float x) const { return AreaPre(*this).area(x); }

std::vector<float> TractShape::sections(int n) const {
  const AreaPre pre(*this);
  std::vector<float> a(n);
  const float dx = length / n;
  for (int k = 0; k < n; ++k) {
    float sum = 0;
    for (int s = 0; s < 8; ++s) sum += pre.area((k + (s + 0.5f) / 8.0f) * dx);
    a[k] = sum / 8.0f;
  }
  return a;
}

}  // namespace puhuja
