// The lane loops below want the full vectorizer and no fused multiply-adds, so
// that the AVX-512, AVX2 and baseline clones of step() compute the same bits
// (see block.cc).
#pragma GCC optimize("O3", "fp-contract=off")

#include "string.h"
#include "params.h"

#include "filters.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace puhuja {
namespace {

constexpr int L = StringFolds::kLayers, N = StringFolds::kLanes;
constexpr int kFirst = StringFolds::kFirst, kLast = StringFolds::kLast, kMoving = kLast - kFirst + 1;
// Research knobs: STRING_<NAME>=value in the environment or in the parameter
// file (PUHUJA_PARAMS) overrides a default while the model is fitted;
// tools/apply_params.py writes a search's result back here.
constexpr float kLength = 1.4f;     // fold length (cm), the moving part
constexpr float kThickness = 0.3f;  // the medial surface's vertical extent (cm)
constexpr float kSeg = kLength / kMoving, kLayer = kThickness / L;
// Mass of the fold (g): Ishizaka-Flanagan's 0.15 g, split over the two
// layers with the superior edge lighter (their 0.125 / 0.025 is too extreme
// for a string: the light layer would race ahead), tapered along the cord.
constexpr float kMass = 0.15f;
const float kTopMass = param("STRING_TOPMASS", 0.5481f) /* cost 0.0328298499863392 n13-stringline-w30-a */;  // the superior layer's share
const float kTaper = param("STRING_TAPER", 0.4419f) /* cost 0.0328298499863392 n13-stringline-w30-a */;        // half-sine taper, 1 - kTaper at the anchors
// Natural frequency (Hz) of the fold's lowest mode at tension 1: the wall
// spring and the string's first mode together. The driver's pitch controller
// takes up the rest, as for the other fold models.
const float kNaturalF0 = param("STRING_F0", 192.2f) /* cost 0.0328298499863392 n13-stringline-w30-a */;
// The wall spring's share of each lane's restoring stiffness at the lowest
// mode; the string tension carries the rest (1 - share), uniform along the
// cord as a string's tension is. (First sweep, 2026-10-31: at 0.3 with shear
// 0.3 and kappa 1 the fold sat open and drifted at 5 Hz; 0.7 with shear 1
// and kappa 2 phonates at 20 % closure, jitter 1.2 %.)
const float kWallShare = param("STRING_WALLSHARE", 0.2466f) /* cost 0.0328298499863392 n13-stringline-w30-a */;
// Stiffness of the superior layer relative to the inferior (softer: the
// layers' phase lag, which feeds the oscillation from the flow).
const float kTopStiff = param("STRING_TOPSTIFF", 0.5377f) /* cost 0.0328298499863392 n13-stringline-w30-a */;
// Shear coupling between the two layers relative to the wall spring
// ("connected mildly"; below ~1 the layers do not drive each other enough
// to oscillate against the static pressure).
const float kShearRel = param("STRING_SHEAR", 1.207f) /* cost 0.0328298499863392 n13-stringline-w30-a */;
// Damping ratios while open per layer; extra while touching the wall.
constexpr float kZeta[L] = {0.1f, 0.4f};
const float kZetaClosed = param("STRING_ZCLOSED", 1.099f) /* cost 0.0328298499863392 n13-stringline-w30-a */;
constexpr float kCollision = 3.0f;  // collision stiffness / the lane's total stiffness
// The share of the air pressure on the surface that drives it.
const float kKappa = param("STRING_KAPPA", 2.148f) /* cost 0.0328298499863392 n13-stringline-w30-a */;
// The cords' own sound: surface area times lateral velocity, both folds.
const float kSurface = param("STRING_SURFACE", 0.02016f) /* cost 0.0328298499863392 n13-stringline-w30-a */;
// The abduction rest area scaled for this model's own amplitude.
const float kRestScale = param("STRING_REST", 0.06898f) /* cost 0.0328298499863392 n13-stringline-w30-a */;
// Rest width along the cord relative to the mean: front (anterior) against back.
const float kRestFront = param("STRING_RESTFRONT", 0.03916f) /* cost 0.0328298499863392 n13-stringline-w30-a */, kRestBack = 2.0f - kRestFront;
constexpr float kRho = 0.00113f;      // air density (g/cm^3)
constexpr float kInertance = 0.005f;  // as folds.cc
// Discharge coefficient of the glottal orifice (vena contracta): the flow
// through a slit of area a is that of an ideal orifice of kDischarge x a.
// Without it the folds passed ~2x a real glottis's flow and the pressure
// above them swung by a third of the lung pressure (review 2026-11-01).
const float kDischarge = param("STRING_CD", 0.9307f) /* cost 0.0328298499863392 n13-stringline-w30-a */;  // (0.65-0.85 with the folds as tuned: the closure is lost, 31 % -> 20 % -> 1 %, 2026-11-01)
// Rounding of the heard area (cm^2): the flow fades over this much area as a
// lane closes instead of cutting off (Jyrki 2026-11-01: "closing should be a
// slow process ... or closing happens over a longer distance, i.e. the
// frequencies are smeared"; with the line tract the step-like cut-off left
// soft phrase ends as pure tones).
const float kRound = param("STRING_ROUND", 0.002f);
// ... and in time: a lane's heard channel may close no faster than this time
// constant (ms; 0: at once). It opens at once.
const float kCloseMs = param("STRING_CLOSEMS", 0.0f);
// The membranous part whose closure counts as the glottal closure: the moving
// lanes but the last, which carries the posterior chink.
constexpr int kMembranousEnd = kLast;
// Continuity regularization as in block.cc: positions and velocities relax
// toward the mean of the two neighbours along the cord with these time
// constants (ms; 0 = off), the forces mix (1 - s) of their own and s of the
// neighbours' mean; a touching lane's velocity relaxes toward its layer's
// mean velocity with kClosedSmoothMs.
const float kSmoothMs = param("STRING_SMOOTHMS", 30.13f) /* cost 0.0328298499863392 n13-stringline-w30-a */, kVSmoothMs = param("STRING_VSMOOTHMS", 54.22f) /* cost 0.0328298499863392 n13-stringline-w30-a */;
const float kForceSmooth = param("STRING_FSMOOTH", 0.001675f) /* cost 0.0328298499863392 n13-stringline-w30-a */, kClosedSmoothMs = param("STRING_CSMOOTHMS", 2.404f) /* cost 0.0328298499863392 n13-stringline-w30-a */;
// The other layer's weight in the neighbour mean (the two along the cord weigh 1 each).
const float kMixLayer = param("STRING_MIXLAYER", 0.007426f) /* cost 0.0328298499863392 n13-stringline-w30-a */;

// The share of the way a value moves toward its target in one step of h
// seconds with a time constant in milliseconds (the 0.001 is ms -> s).
inline float relaxShare(float h, float tauMs) {
  constexpr float kMsToS = 0.001f;
  return std::min(h / (tauMs * kMsToS), 1.0f);
}
template <class A>
void zeroAnchors(A& x) {
  for (int j = 0; j < L; ++j) x[j][0] = x[j][N - 1] = 0;
}
// The weighted mean over the neighbours of each lane; zero at the anchors.
template <class A>
void neighbourMean(const A& x, float (&mean)[L][N]) {
  const float norm = 1.0f / (2.0f + kMixLayer);
  for (int j = 0; j < L; ++j) {
    mean[j][0] = mean[j][N - 1] = 0;
    for (int l = 1; l + 1 < N; ++l) mean[j][l] = norm * (x[j][l - 1] + x[j][l + 1] + kMixLayer * x[1 - j][l]);
  }
}

__attribute__((noinline)) void traceLine(unsigned mask, float flow, float area) {
  static FILE* trace = [] {
    const char* f = std::getenv("STRING_TRACE");
    return f ? std::fopen(f, "w") : nullptr;
  }();
  static int traceCount = 0;
  if (trace && (++traceCount & 3) == 0) std::fprintf(trace, "%u\t%.1f\t%.5f\n", mask, flow, area);
}

}  // namespace

void StringFolds::reset() {
  for (int j = 0; j < L; ++j)
    for (int l = 0; l < N; ++l) x_[j][l] = v_[j][l] = 0;
  for (int l = 0; l < N; ++l) u_[l] = uOut_[l] = 0;
  uSum_ = uOutSum_ = area_ = 0;
  for (int l = 0; l < kLanes; ++l) aHeard_[l] = 0;
  openMask_ = 0;
  wasOpen_ = closedNow_ = false;
}

__attribute__((target_clones("avx512f", "avx2", "default")))
float StringFolds::step(float h, const VocalFolds::Input& in) {
  const float q = std::max(in.q, 0.2f);
  // Rest slit width per lane (both folds together), plus the posterior chink
  // on the last moving lane.
  const float restMean = kRestScale * in.rest / kLength;  // area -> width
  for (int l = 0; l < N; ++l)
    restW_[l] = restMean * (kRestFront + (kRestBack - kRestFront) * (l - kFirst + 0.5f) / kMoving);
  restW_[kLast] += in.gap / kSeg;
  // Masses and stiffnesses. Stiffness ~ tension^2 (frequencies ~ tension).
  // Per layer: the lowest mode at w0 = 2 pi f0 q from the wall spring (share
  // kWallShare of m w0^2 per lane) and the string tension (the rest: a string
  // of N - 1 intervals with fixed ends has its first mode at
  // w^2 = 4 kS / m sin^2(pi / (2 (N - 1))), on the mean mass).
  const float w0 = 2 * kPi * kNaturalF0 * q;
  const float s1 = std::sin(kPi / (2.0f * (N - 1)));
  alignas(64) float mass[L][N], invMass[L][N], kWall[L][N], damp[L][N];
  float kString[L], kShear[L];
  for (int j = 0; j < L; ++j) {
    const float layerMass = kMass * (j == 0 ? 1.0f - kTopMass : kTopMass);
    const float stiff = j == 0 ? 1.0f : kTopStiff;
    const float meanMass = layerMass / kMoving;
    kString[j] = (1.0f - kWallShare) * meanMass * w0 * w0 * stiff / (4.0f * s1 * s1);
    kShear[j] = kShearRel * kWallShare * meanMass * w0 * w0 * stiff;
    for (int l = 0; l < N; ++l) {
      const float taper = 1.0f - kTaper + kTaper * std::sin(kPi * (l - kFirst + 0.5f) / kMoving);
      mass[j][l] = meanMass * taper;
      invMass[j][l] = 1.0f / mass[j][l];
      kWall[j][l] = kWallShare * mass[j][l] * w0 * w0 * stiff;
      damp[j][l] = 2 * std::sqrt(mass[j][l] * mass[j][l] * w0 * w0 * stiff);  // x zeta: critical for the lane's total stiffness
    }
  }
  closedNow_ = false;
  // The slit at each layer and lane: the rest width plus this fold's
  // displacement and its mirror image's.
  alignas(64) float g[L][N];
  for (int j = 0; j < L; ++j)
    for (int l = 0; l < N; ++l) g[j][l] = restW_[l] + 2.0f * x_[j][l];
  // Areas before the step: each lane channel's narrowest layer.
  alignas(64) float aMin[N];
  int jMin[N];
  float aOld = 0;
  for (int l = 0; l < N; ++l) {
    aMin[l] = l >= kFirst && l <= kLast ? kSeg * std::min(g[0][l], g[1][l]) : 0.0f;
    jMin[l] = g[1][l] < g[0][l] ? 1 : 0;
    if (l >= kFirst && l < kMembranousEnd) aOld += std::max(aMin[l], 0.0f);
  }
  area_ = aOld;
  const float drive = in.ps - in.p1;
  // Flow per lane channel: implicit step of Lg du/dt = drive - rho/2 u^2 / a^2
  // (u >= 0), each channel with kMoving times the glottal inertance (in
  // parallel they add up to it). The heard flow goes through the rounded area.
  const float B = kInertance * kMoving / h;
  openMask_ = 0;
  uSum_ = uOutSum_ = 0;
  for (int l = kFirst; l <= kLast; ++l) {
    const float a = aMin[l];
    if (a <= 0) {
      u_[l] = 0;
    } else {
      openMask_ |= 1u << l;
      const float A = 0.5f * kRho / (kDischarge * kDischarge * a * a);
      const float C = B * u_[l] + drive;
      u_[l] = C <= 0 ? 0.0f : 2 * C / (B + std::sqrt(B * B + 4 * A * C));
    }
    float af = a > 30.0f * kRound ? a : kRound * std::log1p(std::exp(a / kRound));
    // The heard channel closes no faster than kCloseMs (it opens at once):
    // the area lags, so the flow still answers the pressure instantly (a lag
    // on the flow itself ran away against the sealed tract of a b, 2026-11-01).
    if (kCloseMs > 0) {
      if (af < aHeard_[l]) aHeard_[l] += (af - aHeard_[l]) * relaxShare(h, kCloseMs);
      else aHeard_[l] = af;
      af = aHeard_[l];
    }
    const float A = 0.5f * kRho / (kDischarge * kDischarge * af * af + 1e-12f);
    const float C = B * uOut_[l] + drive;
    uOut_[l] = C <= 0 ? 0.0f : 2 * C / (B + std::sqrt(B * B + 4 * A * C));
    uSum_ += u_[l];
    uOutSum_ += uOut_[l];
  }
  // Air pressure on each layer's surface: in an open lane the Bernoulli
  // pressure at its own width down to the narrowest layer, the tract pressure
  // above it (separated flow); in a shut lane the full subglottal pressure
  // below the first closed layer, the tract pressure from there up.
  alignas(64) float fAir[L][N];
  for (int l = 0; l < N; ++l) {
    for (int j = 0; j < L; ++j) {
      float p;
      if (l < kFirst || l > kLast) {
        p = 0;
      } else if (aMin[l] <= 0) {
        const int jClosed = g[0][l] <= 0 ? 0 : 1;
        p = j < jClosed ? in.ps : in.p1;
      } else if (j <= jMin[l]) {
        const float vel = u_[l] / std::max(kSeg * g[j][l], 1e-6f);
        p = in.ps - 0.5f * kRho * vel * vel;
      } else {
        p = in.p1;
      }
      fAir[j][l] = p * kSeg * kLayer * kKappa;
    }
  }
  // Forces, lane by lane: the wall spring, the string tension to the two
  // neighbours (the anchors are zero lanes, so the row shifted by one lane is
  // always right), the shear to the other layer, the collision spring with
  // the wall and its extra damping while touching.
  alignas(64) float force[L][N];
  for (int j = 0; j < L; ++j) {
    alignas(64) float xl[N], xr[N];
    xl[0] = xr[N - 1] = 0;
    for (int l = 1; l < N; ++l) xl[l] = x_[j][l - 1];
    for (int l = 0; l + 1 < N; ++l) xr[l] = x_[j][l + 1];
    const float kS = kString[j], kSh = kShear[j];
    for (int l = 0; l < N; ++l) {
      const float x = x_[j][l], gap = g[j][l];
      const float touching = gap < 0 ? 1.0f : 0.0f;
      const float kTotal = kWall[j][l] + 4.0f * kS * s1 * s1;  // the lane's share of the lowest mode's stiffness
      const float coll = -kCollision * kTotal * std::min(gap, 0.0f) * 0.5f;
      const float zeta = kZeta[j] + kZetaClosed * touching;
      force[j][l] = fAir[j][l] + coll - kWall[j][l] * x - zeta * damp[j][l] * v_[j][l] +
                    kS * (xl[l] - 2 * x + xr[l]) + kSh * (x_[1 - j][l] - x);
    }
  }
  zeroAnchors(force);
  // Continuity of the forces.
  if (kForceSmooth > 0) {
    alignas(64) float mean[L][N];
    neighbourMean(force, mean);
    const float keep = 1.0f - kForceSmooth;
    for (int j = 0; j < L; ++j)
      for (int l = 0; l < N; ++l) force[j][l] = keep * force[j][l] + kForceSmooth * mean[j][l];
  }
  for (int j = 0; j < L; ++j)
    for (int l = 0; l < N; ++l) v_[j][l] += h * force[j][l] * invMass[j][l];
  // Continuity of the velocities.
  if (kVSmoothMs > 0) {
    const float mix = relaxShare(h, kVSmoothMs);
    alignas(64) float mean[L][N];
    neighbourMean(v_, mean);
    for (int j = 0; j < L; ++j)
      for (int l = 0; l < N; ++l) v_[j][l] += mix * (mean[j][l] - v_[j][l]);
  }
  // Closed lanes share their layer's velocity.
  if (kClosedSmoothMs > 0) {
    const float mix = relaxShare(h, kClosedSmoothMs);
    for (int j = 0; j < L; ++j) {
      float sum = 0;
      for (int l = kFirst; l <= kLast; ++l) sum += v_[j][l];
      const float m = sum / kMoving;
      for (int l = 0; l < N; ++l) {
        const float touching = g[j][l] < 0 ? 1.0f : 0.0f;
        v_[j][l] += touching * mix * (m - v_[j][l]);
      }
    }
  }
  surface_ = 0;
  for (int j = 0; j < L; ++j)
    for (int l = 0; l < N; ++l) {
      x_[j][l] += h * v_[j][l];
      surface_ += 2.0f * kSeg * kLayer * v_[j][l];  // both folds
    }
  // Continuity of the positions.
  if (kSmoothMs > 0) {
    const float mix = relaxShare(h, kSmoothMs);
    alignas(64) float mean[L][N];
    neighbourMean(x_, mean);
    for (int j = 0; j < L; ++j)
      for (int l = 0; l < N; ++l) x_[j][l] += mix * (mean[j][l] - x_[j][l]);
  }
  zeroAnchors(v_);
  zeroAnchors(x_);
  // The membranous area after the step.
  float aNew = 0;
  for (int l = kFirst; l < kMembranousEnd; ++l)
    aNew += std::max(kSeg * std::min(restW_[l] + 2.0f * x_[0][l], restW_[l] + 2.0f * x_[1][l]), 0.0f);
  // Debug: STRING_TRACE=file writes, once per output sample (every 4th step),
  // the open-lane mask, the flow and the membranous area.
  traceLine(openMask_, uSum_, aNew);
  const bool open = aNew > 0;
  if (wasOpen_ && !open) {
    closedNow_ = true;
    closeFrac_ = aOld > 0 ? std::clamp(aOld / (aOld - aNew + 1e-12f), 0.0f, 1.0f) : 1.0f;
  }
  wasOpen_ = open;
  return uOutSum_ + kSurface * surface_;
}

}  // namespace puhuja
