// The lane loops below want the full vectorizer (the default -O2 uses the
// very cheap cost model and leaves the force loop scalar). No fused
// multiply-adds, so that the AVX-512, AVX2 and baseline clones of step()
// compute the same bits (the folds are a chaotic oscillator: a rounding
// difference grows into a different render).
#pragma GCC optimize("O3", "fp-contract=off")

#include "block.h"
#include "params.h"

#include "filters.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace puhuja {
namespace {

constexpr int D = BlockFolds::kDepth, L = BlockFolds::kLayers, S = BlockFolds::kSegments, N = BlockFolds::kLanes;
constexpr int kFirst = BlockFolds::kFirst, kLast = BlockFolds::kLast, kMoving = kLast - kFirst + 1;
// Research knobs: BLOCK_<NAME>=value in the environment or in the parameter
// file (PUHUJA_PARAMS / GRID_PARAMS) overrides a default while the model is
// fitted; tools/apply_params.py writes a search's result back here.
constexpr float kLength = 1.4f;     // fold length (cm), the moving part
constexpr float kThickness = 0.3f;  // the medial surface's vertical extent (cm)
constexpr float kSeg = kLength / kMoving, kLayer = kThickness / L;
// Mass (g, whole fold) per depth and layer: the cover (depth 0) is the thin
// lamina propria and epithelium over the ligament, thinnest at the superior
// edge; the body (depth 1) is the thyroarytenoid muscle behind it, backed
// below by the conus elasticus. grid.cc's four layers, folded in depth; the
// whole 0.15 g is Ishizaka-Flanagan's.
constexpr float kMass[D][L] = {{0.033f, 0.022f}, {0.050f, 0.045f}};
// Multipliers on the mass, the stiffness and the damping ratio per depth
// (BLOCK_DEP), per layer (BLOCK_ROW) and per segment (BLOCK_COL1..6; the
// anchors 0 and 7 have none), the same on both folds, for the simplex search
// (as grid.cc's GRID_ROW/COL); 1 = the structured default.
const float kDepMass[D] = {param("BLOCK_DEP0", 1.0f), param("BLOCK_DEP1", 1.0f)};
const float kRowMass[L] = {param("BLOCK_ROW0", 1.0f), param("BLOCK_ROW1", 1.0f)};
const float kColMass[S] = {1.0f, param("BLOCK_COL1", 1.0f), param("BLOCK_COL2", 1.0f), param("BLOCK_COL3", 1.0f),
                           param("BLOCK_COL4", 1.0f), param("BLOCK_COL5", 1.0f), param("BLOCK_COL6", 1.0f), 1.0f};
const float kDepStiff[D] = {param("BLOCK_KDEP0", 1.0f), param("BLOCK_KDEP1", 1.0f)};
const float kRowStiff[L] = {param("BLOCK_KROW0", 1.0f), param("BLOCK_KROW1", 1.0f)};
const float kColStiff[S] = {1.0f, param("BLOCK_KCOL1", 1.0f), param("BLOCK_KCOL2", 1.0f), param("BLOCK_KCOL3", 1.0f),
                            param("BLOCK_KCOL4", 1.0f), param("BLOCK_KCOL5", 1.0f), param("BLOCK_KCOL6", 1.0f), 1.0f};
const float kDepDamp[D] = {param("BLOCK_DDEP0", 1.0f), param("BLOCK_DDEP1", 1.0f)};
const float kRowDamp[L] = {param("BLOCK_DROW0", 1.0f), param("BLOCK_DROW1", 1.0f)};
const float kColDamp[S] = {1.0f, param("BLOCK_DCOL1", 1.0f), param("BLOCK_DCOL2", 1.0f), param("BLOCK_DCOL3", 1.0f),
                           param("BLOCK_DCOL4", 1.0f), param("BLOCK_DCOL5", 1.0f), param("BLOCK_DCOL6", 1.0f), 1.0f};
// Along the cord the cross-section tapers toward the ends (half-sine, 1 -
// kTaper at the anchors). The two folds are identical: no hashed mass
// pattern and no heavier right fold (Jyrki, 2026-10-31: "remove all the
// randomness in the initialization of the block", "we can ask for symmetry
// of the vocal coords", against the wobble; the grid keeps both).
const float kTaper = param("BLOCK_TAPER", 0.4132f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
// The body springs' natural frequency at tension 1 (the couplings and the
// cover springs raise the fold's lowest mode; the driver's pitch controller
// takes up the rest, as for the other fold models).
// (2026-10-31: at the grid's 147.5 the block vibrated 20 % under the target
// with the driver's correction pinned at its 1.4 limit, at 250 30 % over it;
// the heavy closure damping lengthens the cycle.)
const float kNaturalF0 = param("BLOCK_F0", 161.4f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
// Stiffness per layer relative to the mass's share: the superior edge softer
// than the inferior (Ishizaka-Flanagan's 90 against 127 Hz), which gives the
// layers their phase lag and the fold its energy transfer from the flow.
const float kTopStiff = param("BLOCK_TOPSTIFF", 0.4335f) /* cost 0.0342462691864234 block8-w15-us5-mf */, kBottomStiff = param("BLOCK_BOTSTIFF", 0.7833f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
// The cover spring (cover to body) and the body spring (body to wall),
// each relative to its mass's share of the fold's natural frequency: Story &
// Titze's body spring is ten times the cover's (100 against 10 kdyn/cm on 5
// times the mass), so the body hardly yields to the static pressure and the
// cover vibrates on it. With both near 1 the cover and the body in series
// were so compliant that the folds blew open to 0.7 cm^2 and the flow (3 x
// the grid's) drained the driver's lung every phrase: the voice died for a
// second or two until the next breath (Jyrki, 2026-10-31: "the sound fully
// disappears and then comes back repeatedly").
const float kCoverStiff = param("BLOCK_COVERSTIFF", 0.9583f) /* cost 0.0342462691864234 block8-w15-us5-mf */, kBodyStiff = param("BLOCK_BODYSTIFF", 3.509f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
// Couplings relative to the spring of the same mass: along the cord (string
// tension; the anchors hold the ends) and between the two layers (shear).
const float kLongRel = param("BLOCK_LONG", 0.5867f) /* cost 0.0342462691864234 block8-w15-us5-mf */, kShearRel = param("BLOCK_SHEAR", 0.7424f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
// Damping ratios while open: the cover's superior edge damps most
// (Ishizaka-Flanagan's upper mass), the body little; extra while touching.
constexpr float kZetaCover[L] = {0.1f, 0.6f}, kZetaBody = 0.1f;
// (3.0 and kKappa 0.5 together, 2026-10-31: closed 0.32-0.40 of the cycle, open area
// to 0.09 cm^2, mean flow 60-90 cm^3/s on a sentence, as the grid; at the grid's
// 0.43 and kappa 1 the closure was 0.15-0.25, the area 0.3-0.7, the flow 3 x.)
const float kZetaClosed = param("BLOCK_ZCLOSED", 1.061f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
constexpr float kCollision = 3.0f;  // collision stiffness / cover stiffness
// The share of the air pressure on a cover's surface that drives it (1: the
// surface faces the midline squarely; see kZetaClosed).
const float kKappa = param("BLOCK_KAPPA", 1.192f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
// The cords' own sound (Jyrki): the moving medial surfaces push air, a volume
// velocity of the surface area times the lateral velocity, summed over the
// covers, heard with the flow (also while the glottis is shut).
const float kSurface = param("BLOCK_SURFACE", 0.8316f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
// The abduction rest area scaled for this model's own amplitude (1: the
// two-mass model's rest area).
const float kRestScale = param("BLOCK_REST", 0.9911f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
// Rest width along the cord relative to the mean: wider toward the arytenoid.
const float kRestFront = param("BLOCK_RESTFRONT", 1.818f) /* cost 0.0342462691864234 block8-w15-us5-mf */, kRestBack = 2.0f - kRestFront;
constexpr float kRho = 0.00113f;      // air density (g/cm^3)
constexpr float kInertance = 0.005f;  // as folds.cc
constexpr float kRound = 0.002f;      // rounding of the heard area (cm^2)
// The membranous part whose closure counts as the glottal closure: the
// moving segments but the last, which carries the posterior chink.
constexpr int kMembranousEnd = kLast;
// Continuity regularization (Jyrki, 2026-10-31: "mix a bit of 6-connected
// average into the position value"): each position relaxes toward the mean
// of its lattice neighbours (the two along the cord, the anchors among them,
// the other layer, the other depth) with this time constant (ms; 0 = off),
// so the fold stays a continuous surface whatever the forces do to one mass.
const float kSmoothMs = param("BLOCK_SMOOTHMS", 2.58f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
// ... the same for the velocities (Jyrki: "try to regularize the speed and
// forces, too"), and for the forces a convex mix at every step:
// force = (1 - s) * force + s * neighbours' mean force (0 = off).
const float kVSmoothMs = param("BLOCK_VSMOOTHMS", 5.448f) /* cost 0.0342462691864234 block8-w15-us5-mf */, kForceSmooth = param("BLOCK_FSMOOTH", 0.08024f) /* cost 0.0342462691864234 block8-w15-us5-mf */;
// Each of the three is applied this many rounds per step (Jyrki, 2026-10-31:
// "run two rounds of regularization instead of one").
const int kSmoothRounds = static_cast<int>(param("BLOCK_SMOOTHROUNDS", 1.0f));
// While a point of the cord is closed (its cover touches the other fold's),
// its velocity relaxes toward the mean velocity of its whole fold block
// (Jyrki, 2026-10-31: "if the coords are closed, then velocity is smoothed
// across the block"); time constant in ms, 0 = off.
const float kClosedSmoothMs = param("BLOCK_CSMOOTHMS", 2.009f) /* cost 0.0342462691864234 block8-w15-us5-mf */;

// The neighbourhood's weights: the two along the cord weigh 1 each (the
// anchors among them), the other layer kMixLayer and the other depth
// kMixDepth. (Mixing the full 6-connected mean into the velocities and the
// forces, 2026-10-31, stopped the vibration: the cover averaged with the
// body, which hardly moves, and the lower layer with the upper lost the
// phase lag between them that feeds the oscillation; closure 0.22 -> 0.01.)
const float kMixLayer = param("BLOCK_MIXLAYER", 0.04781f) /* cost 0.0342462691864234 block8-w15-us5-mf */, kMixDepth = param("BLOCK_MIXDEPTH", 0.1494f) /* cost 0.0342462691864234 block8-w15-us5-mf */;

// The anchors (lanes 0 and 7 of each fold) are not masked (Jyrki): every lane
// loop runs over all 16, and the anchors' force, velocity and position are
// overwritten with zero after (positions are displacements from the rest
// position, so zero is the anchor's place).
// The share of the way a value moves toward its target in one step of h
// seconds when it relaxes with a time constant given in milliseconds: the
// step length over the time constant, at most all the way. (The 0.001 is the
// ms -> s conversion of the knob, not a mixing weight.)
inline float relaxShare(float h, float tauMs) {
  constexpr float kMsToS = 0.001f;
  return std::min(h / (tauMs * kMsToS), 1.0f);
}
template <class A>
void zeroAnchors(A& x) {
  for (int d = 0; d < D; ++d)
    for (int j = 0; j < L; ++j) x[d][j][0] = x[d][j][S - 1] = x[d][j][S] = x[d][j][N - 1] = 0;
}
// The weighted mean over the lattice neighbours of each lane; zero at the
// anchors, so that a value relaxing toward it stays zero there.
template <class A>
void neighbourMean(const A& x, float (&mean)[D][L][N]) {
  const float norm = 1.0f / (2.0f + kMixLayer + kMixDepth);
  for (int d = 0; d < D; ++d)
    for (int j = 0; j < L; ++j) {
      mean[d][j][0] = mean[d][j][N - 1] = 0;
      for (int l = 1; l + 1 < N; ++l)
        mean[d][j][l] = norm * (x[d][j][l - 1] + x[d][j][l + 1] + kMixLayer * x[d][1 - j][l] + kMixDepth * x[1 - d][j][l]);
    }
  zeroAnchors(mean);
}

__attribute__((noinline)) void traceLine(unsigned mask, float flow, float area) {
  static FILE* trace = [] {
    const char* f = std::getenv("BLOCK_TRACE");
    return f ? std::fopen(f, "w") : nullptr;
  }();
  static int traceCount = 0;
  if (trace && (++traceCount & 3) == 0) std::fprintf(trace, "%u\t%.1f\t%.5f\n", mask, flow, area);
}

}  // namespace

void BlockFolds::reset() {
  for (int d = 0; d < D; ++d)
    for (int j = 0; j < L; ++j)
      for (int l = 0; l < N; ++l) x_[d][j][l] = v_[d][j][l] = 0;
  for (int i = 0; i < S; ++i) u_[i] = uOut_[i] = 0;
  uSum_ = uOutSum_ = area_ = 0;
  openMask_ = 0;
  wasOpen_ = closedNow_ = false;
}

// Compiled for AVX-512, AVX2 and the baseline, chosen at run time by the CPU
// (the lane loops are one 512-bit vector each on AVX-512; Jyrki, 2026-10-31:
// "it can be placed in avx512 registers in a way that it forms a 2x2x16
// dynamic in float 32s").
__attribute__((target_clones("avx512f", "avx2", "default")))
float BlockFolds::step(float h, const VocalFolds::Input& in) {
  const float q = std::max(in.q, 0.2f);
  // Rest width along the moving part (the abduction, scaled by kRestScale),
  // plus the posterior chink on the last moving segment (the cartilaginous
  // glottis at the vocal processes).
  const float restMean = kRestScale * in.rest / kLength;  // area -> width
  for (int i = 0; i < S; ++i)
    restW_[i] = restMean * (kRestFront + (kRestBack - kRestFront) * (i - kFirst + 0.5f) / kMoving);
  restW_[kLast] += in.gap / kSeg;
  // Stiffness ~ tension^2 (frequencies ~ tension), masses fixed, damping
  // ratios constant. Per mass: the depth's and layer's share, the taper, the
  // search's multipliers. The spring keeps its mass's natural frequency
  // (k ~ m).
  const float q2 = q * q, w0 = 2 * kPi * kNaturalF0;
  alignas(64) float spring[D][L][N], mass[D][L][N], invMass[D][L][N], damp[D][L][N];
  for (int d = 0; d < D; ++d)
    for (int j = 0; j < L; ++j) {
      const float layerStiff = kBottomStiff + (kTopStiff - kBottomStiff) * j / (L - 1.0f);
      const float stiff = layerStiff * (d == 0 ? kCoverStiff : kBodyStiff) * kDepStiff[d] * kRowStiff[j];
      for (int l = 0; l < N; ++l) {
        const int i = l & (S - 1);
        const float taper = 1.0f - kTaper + kTaper * std::sin(kPi * (i - kFirst + 0.5f) / kMoving);
        const float share = kMass[d][j] / kMoving * taper;
        mass[d][j][l] = share * kDepMass[d] * kRowMass[j] * kColMass[i];
        spring[d][j][l] = share * w0 * w0 * q2 * stiff * kColStiff[i];
        damp[d][j][l] = 2 * std::sqrt(mass[d][j][l] * spring[d][j][l]) * kDepDamp[d] * kRowDamp[j] * kColDamp[i];  // x zeta
        invMass[d][j][l] = 1.0f / mass[d][j][l];
      }
    }
  closedNow_ = false;
  // The slit at each layer and segment from the two covers (lane i and lane
  // i + 8): the width the air and the contact see. (The neighbour assist of
  // the grid, an open neighbour helping a point open, was removed here on
  // 2026-10-31 at Jyrki's request.) The anchors have no channel.
  const auto gaps = [this](float (&ge)[L][S]) {
    for (int j = 0; j < L; ++j)
      for (int i = 0; i < S; ++i) ge[j][i] = restW_[i] + x_[0][j][i] + x_[0][j][i + S];
  };
  float ge[L][S];
  gaps(ge);
  // Areas before the step: each segment's narrowest layer.
  float aMin[S], aOld = 0;
  int jMin[S];
  for (int i = 0; i < S; ++i) {
    aMin[i] = i >= kFirst && i <= kLast ? kSeg * std::min(ge[0][i], ge[1][i]) : 0.0f;
    jMin[i] = ge[1][i] < ge[0][i] ? 1 : 0;
    if (i >= kFirst && i < kMembranousEnd) aOld += std::max(aMin[i], 0.0f);
  }
  area_ = aOld;
  const float drive = in.ps - in.p1;
  // Flow per segment channel: implicit step of L du/dt = drive - rho/2 u^2 / a^2
  // (u >= 0), each channel with kMoving times the glottal inertance (in
  // parallel they add up to it). The heard flow goes through the rounded area.
  const float B = kInertance * kMoving / h;
  openMask_ = 0;
  uSum_ = uOutSum_ = 0;
  for (int i = kFirst; i <= kLast; ++i) {
    const float a = aMin[i];
    if (a <= 0) {
      u_[i] = 0;
    } else {
      openMask_ |= 1u << i;
      const float A = 0.5f * kRho / (a * a);
      const float C = B * u_[i] + drive;
      u_[i] = C <= 0 ? 0.0f : 2 * C / (B + std::sqrt(B * B + 4 * A * C));
    }
    const float af = a > 30.0f * kRound ? a : kRound * std::log1p(std::exp(a / kRound));
    const float A = 0.5f * kRho / (af * af + 1e-12f);
    const float C = B * uOut_[i] + drive;
    uOut_[i] = C <= 0 ? 0.0f : 2 * C / (B + std::sqrt(B * B + 4 * A * C));
    uSum_ += u_[i];
    uOutSum_ += uOut_[i];
  }
  // Air pressure on each cover surface, the same on both folds: in an open
  // segment the Bernoulli pressure at its own width down to the narrowest
  // layer (Ishizaka-Flanagan's lower mass), the tract pressure above it
  // (separated flow); in a shut segment the full subglottal pressure below
  // the first closed layer, the tract pressure from there up.
  alignas(64) float fAir[L][N];
  for (int i = 0; i < S; ++i) {
    for (int j = 0; j < L; ++j) {
      float p;
      if (i < kFirst || i > kLast) {
        p = 0;
      } else if (aMin[i] <= 0) {
        const int jClosed = ge[0][i] <= 0 ? 0 : 1;
        p = j < jClosed ? in.ps : in.p1;
      } else if (j <= jMin[i]) {
        const float vel = u_[i] / std::max(kSeg * ge[j][i], 1e-6f);
        p = in.ps - 0.5f * kRho * vel * vel;
      } else {
        p = in.p1;
      }
      fAir[j][i] = fAir[j][i + S] = p * kSeg * kLayer * kKappa;
    }
  }
  // Forces and the semi-implicit Euler step, lane by lane: the cover hangs
  // on the body by the cover spring, the body on the wall by the body spring;
  // string coupling to the neighbours along the cord (the anchors are the
  // zero lanes at a fold's ends, so the row shifted by one lane either way is
  // always right), shear coupling to the other layer, the collision spring
  // and its damping when the opposing covers overlap. The anchors' forces are
  // zeroed afterwards. The rows are copied to local arrays and the
  // neighbours to shifted copies so that the lane loop has plain indexing
  // and no aliasing: the compiler vectorizes it (16 lanes = one AVX-512
  // register of floats with -march=native).
  alignas(64) float gl[L][N];
  for (int j = 0; j < L; ++j)
    for (int l = 0; l < N; ++l) gl[j][l] = ge[j][l & (S - 1)];
  alignas(64) float xc[L][N], xb[L][N], force[D][L][N];
  for (int j = 0; j < L; ++j)
    for (int l = 0; l < N; ++l) {
      xc[j][l] = x_[0][j][l];
      xb[j][l] = x_[1][j][l];
    }
  for (int j = 0; j < L; ++j) {
    alignas(64) float xcl[N], xcr[N], xbl[N], xbr[N];
    xcl[0] = xbl[0] = xcr[N - 1] = xbr[N - 1] = 0;  // (the ends are anchors: any value)
    for (int l = 1; l < N; ++l) {
      xcl[l] = xc[j][l - 1];
      xbl[l] = xb[j][l - 1];
    }
    for (int l = 0; l + 1 < N; ++l) {
      xcr[l] = xc[j][l + 1];
      xbr[l] = xb[j][l + 1];
    }
    for (int l = 0; l < N; ++l) {
      const float gap = gl[j][l];
      const float touching = gap < 0 ? 1.0f : 0.0f;
      const float kc = spring[0][j][l], kb = spring[1][j][l];
      const float coll = -kCollision * kc * std::min(gap, 0.0f) * 0.5f;
      const float zetaC = kZetaCover[j] + kZetaClosed * touching;
      const float c = xc[j][l], b = xb[j][l];
      const float fc = fAir[j][l] + coll - kc * (c - b) - zetaC * damp[0][j][l] * v_[0][j][l] +
                       kLongRel * kc * (xcl[l] - 2 * c + xcr[l]) + kShearRel * kc * (xc[1 - j][l] - c);
      const float fb = kc * (c - b) - kb * b - kZetaBody * damp[1][j][l] * v_[1][j][l] +
                       kLongRel * kb * (xbl[l] - 2 * b + xbr[l]) + kShearRel * kb * (xb[1 - j][l] - b);
      force[0][j][l] = fc;
      force[1][j][l] = fb;
    }
  }
  zeroAnchors(force);
  // Continuity of the forces: (1 - s) of each force plus s of the
  // neighbours' mean force.
  for (int r = 0; r < kSmoothRounds && kForceSmooth > 0; ++r) {
    alignas(64) float mean[D][L][N];
    neighbourMean(force, mean);
    const float keep = 1.0f - kForceSmooth;
    for (int d = 0; d < D; ++d)
      for (int j = 0; j < L; ++j)
        for (int l = 0; l < N; ++l) force[d][j][l] = keep * force[d][j][l] + kForceSmooth * mean[d][j][l];
  }
  for (int d = 0; d < D; ++d)
    for (int j = 0; j < L; ++j)
      for (int l = 0; l < N; ++l) v_[d][j][l] += h * force[d][j][l] * invMass[d][j][l];
  // Continuity of the velocities: they relax toward the neighbours' mean.
  for (int r = 0; r < kSmoothRounds && kVSmoothMs > 0; ++r) {
    const float mix = relaxShare(h, kVSmoothMs);
    alignas(64) float mean[D][L][N];
    neighbourMean(v_, mean);
    for (int d = 0; d < D; ++d)
      for (int j = 0; j < L; ++j)
        for (int l = 0; l < N; ++l) v_[d][j][l] += mix * (mean[d][j][l] - v_[d][j][l]);
  }
  // Closed points share their fold's velocity: each touching lane relaxes
  // toward the mean velocity of its fold's block (both depths, both layers,
  // the moving lanes).
  if (kClosedSmoothMs > 0) {
    const float mix = relaxShare(h, kClosedSmoothMs);
    float blockMean[N];
    for (int f = 0; f < 2; ++f) {
      float sum = 0;
      for (int d = 0; d < D; ++d)
        for (int j = 0; j < L; ++j)
          for (int i = kFirst; i <= kLast; ++i) sum += v_[d][j][f * S + i];
      const float m = sum / (D * L * kMoving);
      for (int i = 0; i < S; ++i) blockMean[f * S + i] = m;
    }
    for (int d = 0; d < D; ++d)
      for (int j = 0; j < L; ++j)
        for (int l = 0; l < N; ++l) {
          const float touching = gl[j][l] < 0 ? 1.0f : 0.0f;
          v_[d][j][l] += touching * mix * (blockMean[l] - v_[d][j][l]);
        }
  }
  surface_ = 0;
  for (int d = 0; d < D; ++d)
    for (int j = 0; j < L; ++j)
      for (int l = 0; l < N; ++l) {
        x_[d][j][l] += h * v_[d][j][l];
        if (d == 0) surface_ += kSeg * kLayer * v_[d][j][l];
      }
  // Continuity: mix the neighbours' mean into each moving position.
  for (int r = 0; r < kSmoothRounds && kSmoothMs > 0; ++r) {
    const float mix = relaxShare(h, kSmoothMs);
    alignas(64) float mean[D][L][N];
    neighbourMean(x_, mean);
    for (int d = 0; d < D; ++d)
      for (int j = 0; j < L; ++j)
        for (int l = 0; l < N; ++l) x_[d][j][l] += mix * (mean[d][j][l] - x_[d][j][l]);
  }
  zeroAnchors(v_);
  zeroAnchors(x_);
  // The membranous area after the step.
  float aNew = 0;
  gaps(ge);
  for (int i = kFirst; i < kMembranousEnd; ++i) aNew += std::max(kSeg * std::min(ge[0][i], ge[1][i]), 0.0f);
  // Debug: BLOCK_TRACE=file writes, once per output sample (every 4th step),
  // the open-segment mask, the flow and the membranous area.
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
