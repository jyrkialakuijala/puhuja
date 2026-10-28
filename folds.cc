#include "folds.h"

#include <algorithm>
#include <cmath>

namespace puhuja {
namespace {

// Ishizaka & Flanagan (1972) male values.
constexpr float kM1 = 0.125f, kM2 = 0.025f;        // g
constexpr float kK1 = 80000, kK2 = 8000;           // dyn/cm
constexpr float kKc = 25000;                       // coupling spring
constexpr float kD1 = 0.25f, kD2 = 0.05f;          // mass thickness (cm)
constexpr float kLength = 1.4f;                    // fold length (cm)
constexpr float kZeta1 = 0.1f, kZeta2 = 0.6f;      // damping ratios, open
constexpr float kZetaClosed = 1.0f;                // added while touching
constexpr float kCollision = 3.0f;                 // collision stiffness / k
constexpr float kRho = 0.00113f;                   // air density (g/cm^3)
// Inertance of the air in the glottis and the lower vocal tract (g/cm^4).
constexpr float kInertance = 0.005f;  // physically ~0.01; more makes the highs buzz
constexpr float kRound = 0.002f;  // rounding of the heard area (cm^2)
const float kSqrtM1K1 = std::sqrt(kM1 * kK1), kSqrtM2K2 = std::sqrt(kM2 * kK2);
constexpr int kSubsteps = 1;  // per call; the synth calls it 4x per sample

}  // namespace

void VocalFolds::reset() {
  x1_ = v1_ = x2_ = v2_ = u_ = uOut_ = 0;
  wasOpen_ = closedNow_ = false;
}

float VocalFolds::area() const {
  const float a1 = rest_ + 2 * kLength * x1_, a2 = rest_ + 2 * kLength * x2_;
  return std::max(std::min(a1, a2), 0.0f);
}

float VocalFolds::step(float dt, const Input& in) {
  rest_ = in.rest;
  const float q = std::max(in.q, 0.2f);
  // Tension: masses lighter and springs stiffer by q (frequencies x q).
  const float m1 = kM1 / q, m2 = kM2 / q;
  const float k1 = kK1 * q, k2 = kK2 * q, kc = kKc * q;
  const float h = dt / kSubsteps;
  closedNow_ = false;
  for (int s = 0; s < kSubsteps; ++s) {
    const float a1 = rest_ + 2 * kLength * x1_;
    const float a2 = rest_ + 2 * kLength * x2_;
    const float amin = std::min(a1, a2);
    const float drive = in.ps - in.p1;
    // Flow: implicit step of L dU/dt = drive - rho/2 U^2 / a^2 (U >= 0).
    if (amin <= 0) {
      u_ = 0;
    } else {
      const float A = 0.5f * kRho / (amin * amin);
      const float B = kInertance / h;
      const float C = B * u_ + drive;
      // Root of A u^2 + B u - C = 0 in the cancellation-free form.
      u_ = C <= 0 ? 0.0f : 2 * C / (B + std::sqrt(B * B + 4 * A * C));
    }
    // The flow that is heard goes through a rounded area: min(a1, a2) and
    // the cut at 0 put corners into each pulse, which comb-filter the
    // source (a ripple every 3 harmonics that sweeps through the formants on
    // pitch glides, like a phaser). The mechanics keep the exact area.
    {
      const float sm = 0.5f * (a1 + a2) - std::sqrt(0.25f * (a1 - a2) * (a1 - a2) + kRound * kRound);
      // softplus(sm) ~ sm for a wide opening: past 30 kRound it is taken
      // as it is (clamping the exponent there capped the heard area at
      // 0.06 cm^2: a flat-topped pulse with no 2nd harmonic, a hollow voice).
      const float af = sm > 30.0f * kRound ? sm : kRound * std::log1p(std::exp(sm / kRound));
      const float A = 0.5f * kRho / (af * af + 1e-12f);
      const float B = kInertance / h;
      const float C = B * uOut_ + drive;
      uOut_ = C <= 0 ? 0.0f : 2 * C / (B + std::sqrt(B * B + 4 * A * C));
    }
    // Pressures on the masses: Bernoulli inside the glottis.
    float pm1, pm2;
    if (amin <= 0) {
      pm1 = in.ps;                        // closed: full pressure from below
      pm2 = a1 <= 0 ? in.p1 : in.ps;      // upper edge sees ps if only it is shut
    } else {
      const float vel = u_ / a1;
      pm1 = in.ps - 0.5f * kRho * vel * vel;
      pm2 = in.p1;
    }
    const float f1 = kLength * kD1 * pm1;
    const float f2 = kLength * kD2 * pm2;
    // Collision: when a slit closes, the tissue is compressed.
    float c1 = 0, c2 = 0, z1 = kZeta1, z2 = kZeta2;
    if (a1 < 0) {
      c1 = -kCollision * k1 * a1 / (2 * kLength);
      z1 += kZetaClosed;
    }
    if (a2 < 0) {
      c2 = -kCollision * k2 * a2 / (2 * kLength);
      z2 += kZetaClosed;
    }
    // sqrt(m k) does not depend on the tension q (m / q times k q).
    const float r1 = 2 * z1 * kSqrtM1K1, r2 = 2 * z2 * kSqrtM2K2;
    // Semi-implicit Euler: velocities first, then positions.
    v1_ += h * (f1 - r1 * v1_ - k1 * x1_ - kc * (x1_ - x2_) + c1) / m1;
    v2_ += h * (f2 - r2 * v2_ - k2 * x2_ - kc * (x2_ - x1_) + c2) / m2;
    x1_ += h * v1_;
    x2_ += h * v2_;
    const float aNew = std::min(rest_ + 2 * kLength * x1_, rest_ + 2 * kLength * x2_);
    const bool open = aNew > 0;
    if (wasOpen_ && !open) {
      closedNow_ = true;
      // Within a substep each slit's area moves linearly (x += h v), so
      // the closure is exactly where the first slit reaches zero.
      const float b1 = rest_ + 2 * kLength * x1_, b2 = rest_ + 2 * kLength * x2_;
      float f = 1.0f;
      if (a1 > 0 && b1 <= 0) f = std::min(f, a1 / (a1 - b1));
      if (a2 > 0 && b2 <= 0) f = std::min(f, a2 / (a2 - b2));
      closeFrac_ = (s + std::clamp(f, 0.0f, 1.0f)) / kSubsteps;
    }
    wasOpen_ = open;
  }
  return uOut_;
}

}  // namespace puhuja
