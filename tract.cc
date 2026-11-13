#include "tract.h"
#include "fastmath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

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
    front = s.lFront > 0 && s.mFront != 1.0f;  // (< 1: the tongue fills it, LINE_CAV_*)
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

// cos^2(pi/2 d) for |d| <= 1, as a polynomial in u = d^2 (the Taylor series of (1 + cos(pi d)) / 2 to u^7; error
// < 3e-6 at |d| = 1).
inline float cos2Half(float d) {
  const float u = d * d;
  const float c[] = {1.0f, -2.4674011f, 2.0293560f, -0.66763150f, 0.11766490f, -0.012903506f, 0.00096478745f, -5.2321e-5f};
  float p = c[7];
  for (int i = 6; i >= 0; --i) p = p * u + c[i];
  return p;
}

std::vector<float> TractShape::sections(int n) const {
  const AreaPre p(*this);
  // 8 points per section (the average smooths the hump's and the ramps' edges), all at once: the loop has no
  // calls and no branches on x, so it runs 16 points per AVX-512 instruction.
  const int m = 8 * n;
  const float dx = length / n;
  std::vector<float> v(m);
  const float iLar = 1.0f / p.lar.w, iHyIn = 1.0f / p.hyIn.w, iHyOut = 1.0f / p.hyOut.w, iVel = 1.0f / p.velum.w,
              iFrIn = 1.0f / p.frIn.w, iFrOut = 1.0f / p.frOut.w, iLip = 1.0f / p.lip.w, iWc = 1.0f / p.wcM;
  auto rmp = [](float x, float x0, float inv) {
    const float t = std::min(std::max((x - x0) * inv, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
  };
  const bool hypo = p.hypo, front = p.front, hump = p.hump;
  for (int i = 0; i < m; ++i) {
    const float x = (static_cast<float>(i >> 3) + (static_cast<float>(i & 7) + 0.5f) * 0.125f) * dx;
    float lnA = p.lnLar + (p.lnPh - p.lnLar) * rmp(x, p.lar.x0, iLar);
    if (hypo) {
      const float w = rmp(x, p.hyIn.x0, iHyIn) * (1.0f - rmp(x, p.hyOut.x0, iHyOut));
      lnA += std::max(p.lnHy - lnA, 0.0f) * w;
    }
    lnA += (p.lnOr - p.lnPh) * rmp(x, p.velum.x0, iVel);
    if (front) lnA += p.lnFront * rmp(x, p.frIn.x0, iFrIn) * (1.0f - rmp(x, p.frOut.x0, iFrOut));
    if (hump) {
      const float d = (x - p.xcL) * iWc;
      const float in = std::fabs(d) < 1.0f ? 1.0f : 0.0f;
      lnA += in * p.scM * (p.lnAc - lnA) * cos2Half(std::min(std::fabs(d), 1.0f));
    }
    lnA += (p.lnLip - lnA) * rmp(x, p.lip.x0, iLip);
    v[i] = fastExp(lnA);
  }
  std::vector<float> a(n);
  for (int k = 0; k < n; ++k) {
    float sum = 0;
    for (int s = 0; s < 8; ++s) sum += v[8 * k + s];
    a[k] = sum * 0.125f;
  }
  return a;
}

}  // namespace puhuja
