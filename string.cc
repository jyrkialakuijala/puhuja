// The lane loops below want the full vectorizer and no fused multiply-adds, so
// that the AVX-512, AVX2 and baseline clones of step() compute the same bits
// (see block.cc).
#pragma GCC optimize("O3", "fp-contract=off")

#include "string.h"
#include "params.h"

#include "filters.h"
#include "fastmath.h"

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
const float kTopMass = param("STRING_TOPMASS", 0.464394f) /* cost 1814 whisper-run1-phon_err */ /* 2026-11-04 LF fit (tools/lf_fit.py, ref/lf/fit2) 0.358205, then 15 % back toward the earlier 0.7614 (Jyrki: the more trained voice) */;  // the superior layer's share
const float kTaper = param("STRING_TAPER", 0.5019889f) /* cost 1814 whisper-run1-phon_err */ /* 2026-11-04 LF fit (tools/lf_fit.py, ref/lf/fit2) 0.185574, then 15 % back toward the earlier 0.1888 (Jyrki: the more trained voice) */;        // half-sine taper, 1 - kTaper at the anchors
// Natural frequency (Hz) of the fold's lowest mode at tension 1: the wall
// spring and the string's first mode together. The driver's pitch controller
// takes up the rest, as for the other fold models.
const float kNaturalF0 = param("STRING_F0", 105.7f) /* cost 1058 whisper-sens1-combo-phase1-err */;
// The wall spring's share of each lane's restoring stiffness at the lowest
// mode; the string tension carries the rest (1 - share), uniform along the
// cord as a string's tension is. (First sweep, 2026-10-31: at 0.3 with shear
// 0.3 and kappa 1 the fold sat open and drifted at 5 Hz; 0.7 with shear 1
// and kappa 2 phonates at 20 % closure, jitter 1.2 %.)
const float kWallShare = param("STRING_WALLSHARE", 0.1595f) /* cost Whisper 852 composite 23.90/26.04 92b-cheap9 */ /* 2026-11-04 LF fit (tools/lf_fit.py, ref/lf/fit2) 0.163924, then 15 % back toward the earlier 0.1804 (Jyrki: the more trained voice) */;
// Stiffness of the superior layer relative to the inferior (softer: the
// layers' phase lag, which feeds the oscillation from the flow).
const float kTopStiff = param("STRING_TOPSTIFF", 1.123f) /* cost Whisper 852 composite 23.90/26.04 92b-cheap9 */ /* 2026-11-04 LF fit (tools/lf_fit.py, ref/lf/fit2) 1.091, then 15 % back toward the earlier 0.6737 (Jyrki: the more trained voice) */;
// Shear coupling between the two layers relative to the wall spring
// ("connected mildly"; below ~1 the layers do not drive each other enough
// to oscillate against the static pressure).
const float kShearRel = param("STRING_SHEAR", 6.899f) /* cost Whisper 852 composite 23.90/26.04 92b-cheap9 */ /* 2026-11-04 LF fit (tools/lf_fit.py, ref/lf/fit2) 3.4412, then 15 % back toward the earlier 2.275 (Jyrki: the more trained voice) */;
// Damping ratios while open per layer; extra while touching the wall.
constexpr float kZeta[L] = {0.1f, 0.4f};
const float kZetaClosed = param("STRING_ZCLOSED", 0.08202616f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */ /* 2026-11-04 LF fit (tools/lf_fit.py, ref/lf/fit2) 0.207612, then 15 % back toward the earlier 0.1868 (Jyrki: the more trained voice) */;
const float kCollision = param("STRING_COLLISION", 2.79108f);  // collision stiffness / the lane's total stiffness (a knob since 2026-11-07: the closing phase)
// The share of the air pressure on the surface that drives it.
// The oral pressure's slow part on the folds' superior surface (2026-11-11,
// review voice.md B1): the layers took the absolute tract pressure P1 as an
// opening push and nothing pressed on the upper surface, so a pressurised
// mouth blew the folds open: in the harness voicing stopped at P1 ~ 0.4 Ps
// (held open at 0.043 cm^2) while the expansion sink holds a b at 0.6 Ps, so
// b/d/g closures and z/v went silent ~20 ms in. STRING_P1SUP is the share of
// P1, low-passed at STRING_P1SUP_HZ, taken off every layer's pressure (1: a
// fold immersed in the mouth's pressure feels only the drop across it; 0: as
// before). The corner stays low (<= ~15 Hz): at 50 Hz it cancels part of the
// f0 coupling and the vowels rose 1 dB. Tried (2026-11-11, notes in
// ref/opt/lab/BUG-VOICING): a fast fall (200 Hz) tracks each cycle's P1 minimum
// and the male b/d/g bars died again (-25..-31 dB); taking off only the part
// above 0.1 Ps loses the sink's suction (a negative P1 that helps the folds
// close) and the bars died too. With 1, the voiceless fricatives' hiss is
// ~0.8 dB weaker (their P1 no longer helps hold the glottis open) and the
// vowels after t s ~0.5 dB less aspirated.
const float kP1Sup = param("STRING_P1SUP", 0.0f), kP1SupHz = param("STRING_P1SUP_HZ", 10.0f);
const float kKappa = param("STRING_KAPPA", 0.4327f) /* cost Whisper 852 composite 23.90/26.04 92b-cheap9 */ /* 2026-11-04 LF fit (tools/lf_fit.py, ref/lf/fit2) 2.24987, then 15 % back toward the earlier 3 (Jyrki: the more trained voice) */;
// The cords' own sound: surface area times lateral velocity, both folds.
const float kSurface = param("STRING_SURFACE", 0.086025414f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */;
// The abduction rest area scaled for this model's own amplitude.
const float kRestScale = param("STRING_REST", 0.0212255146f) /* cost Whisper 852 composite 23.90/26.04 92b-cheap9 */ /* 2026-11-04 LF fit (tools/lf_fit.py, ref/lf/fit2) 0.0539805, then 15 % back toward the earlier 0.04671 (Jyrki: the more trained voice) */;
// Rest width along the cord relative to the mean: front (anterior) against back.
const float kRestFront = param("STRING_RESTFRONT", 1.5785679f) /* cost 1814 whisper-run1-phon_err */ /* 2026-11-04 LF fit (tools/lf_fit.py, ref/lf/fit2) 1.58689, then 15 % back toward the earlier 1.51 (Jyrki: the more trained voice) */, kRestBack = 2.0f - kRestFront;
constexpr float kRho = 0.00113f;      // air density (g/cm^3)
// 2026-11-07: 0.005 -> 0.035 (x7, Jyrki's number) with kRound 0.0005 -> 0.000125: the flow lags the area, the pulse
// skews and its cut-off sharpens; the voiced spectrum's 9 dB deficit against Kokoro at 1-3 kHz closes to 0 and the
// brightness is harmonic (HNR up in every band), where the fold-knob fits had made noise (ref/opt/zopt3/
// ZIMTOHRLI_PROGRESS.md sections 11-13). Jyrki: "p12-i7r.25.wav was the best of these". 4-5 dB louder.
const float kInertance = param("STRING_INERTANCE", 0.008914332f) /* cost Whisper 874 composite 24.40/27.30 94e-inert040 */;  // as folds.cc (a knob since 2026-11-06: it smooths the flow's cut-off, the return phase)
// Discharge coefficient of the glottal orifice (vena contracta): the flow
// through a slit of area a is that of an ideal orifice of kDischarge x a.
// Without it the folds passed ~2x a real glottis's flow and the pressure
// above them swung by a third of the lung pressure (review 2026-11-01).
const float kDischarge = param("STRING_CD", 0.92f);  // (Jyrki 2026-11-04: 0.92, a smooth converging glottis; 1.0 from the search)  // (0.65-0.85 with the folds as tuned: the closure is lost, 31 % -> 20 % -> 1 %, 2026-11-01)
const float kRhoOver2Cd2 = 0.5f * kRho / (kDischarge * kDischarge);  // rho / (2 Cd^2): the Bernoulli head per (u / a)^2
// Rounding of the heard area (cm^2): the flow fades over this much area as a
// lane closes instead of cutting off (Jyrki 2026-11-01: "closing should be a
// slow process ... or closing happens over a longer distance, i.e. the
// frequencies are smeared"; with the line tract the step-like cut-off left
// soft phrase ends as pure tones).
const float kRound = param("STRING_ROUND", 0.0001282f) /* cost Whisper 852 composite 23.90/26.04 92b-cheap9 */;  // (0.0005 until 2026-11-07, see kInertance)
const float kRoundInv = 1.0f / kRound;
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
const float kSmoothMs = param("STRING_SMOOTHMS", 43.212888f) /* cost 1814 whisper-run1-phon_err */, kVSmoothMs = param("STRING_VSMOOTHMS", 43.972203f) /* cost 1814 whisper-run1-phon_err */;
const float kForceSmooth = param("STRING_FSMOOTH", 0.00094663794f) /* cost 1814 whisper-run1-phon_err */, kClosedSmoothMs = param("STRING_CSMOOTHMS", 1.5224563f) /* cost 1814 whisper-run1-phon_err */;
// The other layer's weight in the neighbour mean (the two along the cord weigh 1 each).
const float kMixLayer = param("STRING_MIXLAYER", 0.007317233f) /* cost 1814 whisper-run1-phon_err */;

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
// A row of the 16 lanes as one vector (GCC vector extensions: one AVX-512 register, two AVX2 ones). The lane
// shifts are done in registers: a scalar loop reading x[l - 1] and x[l + 1] from a row just stored as a vector waited
// for the store (store forwarding), and its 14 lanes kept it scalar (2026-11-25: 8.6 % of a render).
typedef float V16 __attribute__((vector_size(64)));
typedef int I16 __attribute__((vector_size(64)));
static_assert(N == 16, "the lane vectors hold 16 lanes");
inline V16 loadRow(const float* r) { V16 v; __builtin_memcpy(&v, r, sizeof v); return v; }
inline void storeRow(float* r, V16 v) { __builtin_memcpy(r, &v, sizeof v); }
// Lane l gets lane l - 1 (lane 0 gets 0), lane l + 1 (lane 15 gets 0).
inline V16 fromLeft(V16 x) { return __builtin_shuffle(x, V16{}, I16{16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}); }
inline V16 fromRight(V16 x) { return __builtin_shuffle(x, V16{}, I16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}); }
// 1 in the lanes 1..14, 0 at the two anchors.
inline V16 innerLanes() { return V16{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}; }
template <class A>
void neighbourMean(const A& x, float (&mean)[L][N]) {
  const float norm = 1.0f / (2.0f + kMixLayer);
  const V16 x0 = loadRow(x[0]), x1 = loadRow(x[1]);
  const I16 inner = innerLanes() > 0;
  for (int j = 0; j < L; ++j) {
    const V16 xj = j == 0 ? x0 : x1, xo = j == 0 ? x1 : x0;
    const V16 m = norm * (fromLeft(xj) + fromRight(xj) + kMixLayer * xo);
    storeRow(mean[j], inner ? m : V16{});
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
  p1Lp_ = 0;
  openMask_ = 0;
  wasOpen_ = closedNow_ = false;
}

__attribute__((target_clones("avx512f", "avx2", "default")))
float StringFolds::step(float h, const VocalFolds::Input& in) {
  constexpr float kMovingInv = 1.0 / kMoving;
  const float q = std::max(in.q, 0.2f);
  // Rest slit width per lane (both folds together), plus the posterior chink
  // on the last moving lane.
  const float restMean = kRestScale * in.rest / kLength;  // area -> width
  for (int l = 0; l < N; ++l)
    restW_[l] = restMean * (kRestFront + (kRestBack - kRestFront) * (l - kFirst + 0.5f) * kMovingInv);
  restW_[kLast] += in.gap / kSeg;
  // Masses and stiffnesses. Stiffness ~ tension^2 (frequencies ~ tension).
  // Per layer: the lowest mode at w0 = 2 pi f0 q from the wall spring (share
  // kWallShare of m w0^2 per lane) and the string tension (the rest: a string
  // of N - 1 intervals with fixed ends has its first mode at
  // w^2 = 4 kS / m sin^2(pi / (2 (N - 1))), on the mean mass).
  const float w0 = 2 * kPi * kNaturalF0 * q;
  const float s1 = std::sin(kPi / (2.0f * (N - 1)));
  // The masses depend only on the layer and the lane (the taper): computed
  // once. (Rebuilt every substep they cost 32 sinf calls per substep, 8 % of
  // a render, 2026-11-05.) Only the stiffnesses and the damping follow q.
  struct Masses { alignas(64) float mass[L][N]; alignas(64) float invMass[L][N]; };
  static const Masses masses = [] {
    Masses t;
    for (int j = 0; j < L; ++j) {
      const float layerMass = kMass * (j == 0 ? 1.0f - kTopMass : kTopMass);
      const float meanMass = layerMass * kMovingInv;
      for (int l = 0; l < N; ++l) {
        const float taper = 1.0f - kTaper + kTaper * std::sin(kPi * (l - kFirst + 0.5f) * kMovingInv);
        t.mass[j][l] = meanMass * taper;
        t.invMass[j][l] = 1.0f / t.mass[j][l];
      }
    }
    return t;
  }();
  // (pajupilli, singing: CHANGES.md 2026-11-26) lighter folds for a higher voice: every mass x in.mass (the
  // stiffnesses below follow, so the pitch stays the tension's to set). 1: the table itself, bit for bit.
  const float massX = in.mass > 0 ? in.mass : 1.0f;
  Masses scaled;
  if (massX != 1.0f)
    for (int j = 0; j < L; ++j)
      for (int l = 0; l < N; ++l) {
        scaled.mass[j][l] = masses.mass[j][l] * massX;
        scaled.invMass[j][l] = masses.invMass[j][l] / massX;
      }
  const auto& mass = massX != 1.0f ? scaled.mass : masses.mass;
  const auto& invMass = massX != 1.0f ? scaled.invMass : masses.invMass;
  alignas(64) float kWall[L][N], damp[L][N];
  float kString[L], kShear[L];
  for (int j = 0; j < L; ++j) {
    const float layerMass = massX * kMass * (j == 0 ? 1.0f - kTopMass : kTopMass);
    const float stiff = j == 0 ? 1.0f : kTopStiff;
    const float sqrt_stiff = j == 0 ? 1.0f : std::sqrt(kTopStiff);
    const float meanMass = layerMass * kMovingInv;
    kString[j] = (1.0f - kWallShare) * meanMass * w0 * w0 * stiff / (4.0f * s1 * s1);
    kShear[j] = kShearRel * kWallShare * meanMass * w0 * w0 * stiff;
    for (int l = 0; l < N; ++l) {
      kWall[j][l] = kWallShare * mass[j][l] * w0 * w0 * stiff;
      damp[j][l] = 2 * mass[j][l] * w0 * sqrt_stiff;  // x zeta: critical for the lane's total stiffness
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
  float aOld = 0;
  for (int l = 0; l < N; ++l) {
    aMin[l] = l >= kFirst && l <= kLast ? kSeg * std::min(g[0][l], g[1][l]) : 0.0f;
    if (l >= kFirst && l < kMembranousEnd) aOld += std::max(aMin[l], 0.0f);
  }
  area_ = aOld;
  const float drive = in.ps - in.p1;
  const float supHz = in.shield >= 0 ? in.shieldHz : kP1SupHz;  // (Input::shield: a singer's own share and corner)
  if (h != p1LpH_ || supHz != p1LpHz_) p1LpH_ = h, p1LpHz_ = supHz, p1LpA_ = 1.0f - std::exp(-2.0f * kPi * supHz * h);
  p1Lp_ += p1LpA_ * (in.p1 - p1Lp_);
  const float pRef = (in.shield >= 0 ? in.shield : kP1Sup) * p1Lp_;  // (0 by default: the layers' pressures as before, bit for bit)
  // Flow per lane channel: implicit step of Lg du/dt = drive - rho/2 u^2 / a^2
  // (u >= 0), each channel with kMoving times the glottal inertance (in
  // parallel they add up to it). The heard flow goes through the rounded area.
  const float B = kInertance * kMoving / h;
  // All lanes at once, branch-free (2026-11-13: the lane-by-lane loop with its sqrt, divisions and branches was
  // ~10 % of a render): no libm calls, the sums in a second loop (no -ffast-math, so a sum in the loop would keep it
  // scalar). The two directions of the implicit step share one form: for C >= 0, 2 C / (B + sqrt(B^2 + 4 A C));
  // for C < 0, -2 C / (B + sqrt(B^2 - 4 A C)); both are 2 |C| / (B + sqrt(B^2 + 4 A |C|)).
  // (log1p(e^z) = z + log1p(e^-z): for z <= 0 the shifted softplus is under zero and clamps to 0, so it is only
  // used for 0 < z < 30, with fastmath.h's call-free exp and log1p.)
  constexpr float kLn2 = 0.69314718f;
  const float B2 = B * B, aRound = 30.0f * kRound;
  const float closeMix = kCloseMs > 0 ? relaxShare(h, kCloseMs) : 0.0f;
  for (int l = 0; l < N; ++l) {  // all 16 lanes (whole vectors); the two anchors keep their zeros
    const bool moving = l >= kFirst && l <= kLast;
    const float a = aMin[l];
    const bool open = a > 0;
    const float aa = open ? a : 1.0f;
    const float A = kRhoOver2Cd2 / (aa * aa);
    // (either direction: with the flow clamped at zero, air pushed into a
    // sealed mouth by the closure's own transient could never return, the
    // cavity sat at 1.3x the lung pressure and the folds were held 0.5 cm^2
    // open through every b, d, g, 2026-11-03)
    const float C = std::fabs(B * u_[l] + drive);
    const float uNew = 2 * C / (B + std::sqrt(B2 + 4 * A * C));
    u_[l] = open ? uNew : 0.0f;
    // (the plain softplus left kRound ln 2 = 0.0014 cm^2 per shut lane, 0.02
    // over the cord, half the open peak: a steady leak that kept the flow at
    // 50 % of its mean through the closed phase, a dark breathy source,
    // 2026-11-02; shifted by its value at zero a shut lane passes nothing)
    const float z = std::min(std::max(a, 0.0f), aRound) * kRoundInv;
    const float sp = std::max(kRound * (z + fastLog1pUnit(fastExp(-z)) - kLn2), 0.0f);
    float af = a > aRound ? a : open ? sp : 0.0f;
    // The heard channel closes no faster than kCloseMs (it opens at once):
    // the area lags, so the flow still answers the pressure instantly (a lag
    // on the flow itself ran away against the sealed tract of a b, 2026-11-01).
    if (kCloseMs > 0) {
      const float h0 = aHeard_[l];
      af = af < h0 ? h0 + (af - h0) * closeMix : af;
      aHeard_[l] = moving ? af : h0;
    }
    const float A2 = kRhoOver2Cd2 / (af * af + 1e-12f);
    const float C2 = std::fabs(B * uOut_[l] + drive);
    const float uo = 2 * C2 / (B + std::sqrt(B2 + 4 * A2 * C2));
    uOut_[l] = moving ? uo : uOut_[l];
  }
  openMask_ = 0;
  uSum_ = uOutSum_ = 0;
  for (int l = kFirst; l <= kLast; ++l) {
    openMask_ |= (aMin[l] > 0 ? 1u : 0u) << l;
    uSum_ += u_[l];
    uOutSum_ += uOut_[l];
  }
  {
    // Debug: PUHUJA_STRING_TRACE=file writes the glottal area (cm^2, the lanes' minima summed) every 4th substep.
    static FILE* strace = std::getenv("PUHUJA_STRING_TRACE") ? std::fopen(std::getenv("PUHUJA_STRING_TRACE"), "w") : nullptr;
    static long scount = 0;
    if (strace && (scount++ & 3) == 0) {
      float aTot = 0;
      for (int l = kFirst; l <= kLast; ++l) aTot += std::max(aMin[l], 0.0f);
      std::fprintf(strace, "%.5f\t%d\t%.5f\n", aTot, __builtin_popcount(openMask_), uOutSum_);  // (heard flow since 2026-11-07)
    }
  }
  // Air pressure on each layer's surface: in an open lane the Bernoulli
  // pressure at its own width down to the narrowest layer, the tract pressure
  // above it (separated flow); in a shut lane the full subglottal pressure
  // below the first closed layer, the tract pressure from there up.
  // (Whole rows, branch-free: the lane-by-lane branches were 7 % of a render, 2026-11-25.)
  alignas(64) float fAir[L][N];
  {
    const V16 g0 = loadRow(g[0]), g1 = loadRow(g[1]), a = loadRow(aMin), u = loadRow(u_);
    const V16 ps = V16{} + in.ps, p1 = V16{} + in.p1;
    const I16 moving = innerLanes() > 0, shut = a <= 0, oneNarrower = g1 < g0;
    for (int j = 0; j < L; ++j) {
      const V16 gj = j == 0 ? g0 : g1;
      // A shut lane: the subglottal pressure below the first closed layer, the tract's from there up.
      const V16 pShut = j == 0 ? (g0 <= 0 ? p1 : ps) : p1;
      const V16 w = kSeg * gj;
      const V16 vel = u / (w < 1e-6f ? V16{} + 1e-6f : w);
      const V16 pBelow = ps - 0.5f * kRho * vel * vel;
      const V16 pOpen = j == 0 ? pBelow : (oneNarrower ? pBelow : p1);  // up to the narrowest layer, then the tract's
      const V16 p = shut ? pShut : pOpen;
      storeRow(fAir[j], (moving ? p - pRef : V16{}) * kSeg * kLayer * kKappa);
    }
  }
  // Forces, lane by lane: the wall spring, the string tension to the two
  // neighbours (the anchors are zero lanes, so the row shifted by one lane is
  // always right), the shear to the other layer, the collision spring with
  // the wall and its extra damping while touching.
  alignas(64) float force[L][N];
  {
    const V16 x0 = loadRow(x_[0]), x1 = loadRow(x_[1]);
    for (int j = 0; j < L; ++j) {
      const V16 x = j == 0 ? x0 : x1, xo = j == 0 ? x1 : x0, gap = loadRow(g[j]), kW = loadRow(kWall[j]);
      const float kS = kString[j], kSh = kShear[j];
      const I16 touching = gap < 0;
      const V16 kTotal = kW + 4.0f * kS * s1 * s1;  // the lane's share of the lowest mode's stiffness
      const V16 coll = -kCollision * kTotal * (0.0f < gap ? V16{} : gap) * 0.5f;  // (std::min(gap, 0.0f))
      const V16 zeta = touching ? V16{} + (kZeta[j] + kZetaClosed * 1.0f) : V16{} + (kZeta[j] + kZetaClosed * 0.0f);
      const V16 f = loadRow(fAir[j]) + coll - kW * x - zeta * loadRow(damp[j]) * loadRow(v_[j]) +
                    kS * (fromLeft(x) - 2 * x + fromRight(x)) + kSh * (xo - x);
      storeRow(force[j], f);
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
      const float m = sum * kMovingInv;
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
