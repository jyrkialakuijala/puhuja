#include "line.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#include "filters.h"
#include "params.h"
#include "tract.h"

namespace puhuja {

namespace {
constexpr float kPiL = 3.14159265f;
constexpr float kM2PiL = -2.0f * kPiL;
// Wall per unit area (Ishizaka, Flanagan & Matsudaira 1975): mass (g/cm^2),
// resistance (dyn s/cm^3), stiffness (dyn/cm^3).
const float kLungR = param("LINE_LUNG_R", 7.0089434f) /* cost Whisper 852 composite 23.90/26.04 92b-cheap9 */, kLungRInv = 1.0f / kLungR;  // the respiratory system's DC resistance (dyn s/cm^5)
const float kWallMass = param("LINE_WALL_M", 1.0450999f) /* cost Whisper 852 composite 23.90/26.04 92b-cheap9 */, kWallMassInv = 1.0f / kWallMass;
const float kNoiseSt = param("LINE_NOISE_ST", 0.0707841477f);         // Strouhal number of the noise corners (0: the fixed LINE_NOISE_HP; 0.1 on 2026-11-04: the s spectrum near Kokoro's, Whisper neutral)
const float kNoiseLpRatio = param("LINE_NOISE_LPR", 10.3959664f);  // (2.5 -> 10 on 2026-11-04: in a 1.3-octave band the noise's peak hopped 0.30 octave per 10 ms frame, the readers' 0.06-0.08: the "zipping, zonking, pinging" Jyrki heard; at 10: 0.11; Whisper neutral)
const float kNoiseStSmoothGMs = param("LINE_NOISE_ST_SMOOTH_G_MS", 15.001933f);  // 2026-11-05: the glottal jet's corner smoothed over this alone (ms; < 0: LINE_NOISE_ST_SMOOTH_MS). Its corner zips 200 <-> 12000 Hz once per pitch period in every voiced sound.
const float kNoiseStSmoothFMs = param("LINE_NOISE_ST_SMOOTH_F_MS", -1.0f);  // ... the constriction's jet's
const float kNoiseFcMax = param("LINE_NOISE_FC_MAX", 8000.0f);  // the corners' ceiling (Hz; a closing constriction sat at the clamp: 8-12 ms pings at 12 kHz in d, t, m)
const float kNoiseUMax = param("LINE_NOISE_U_MAX", 1.01498f);  // > 0: the jet speed behind the corners and the constriction's Reynolds number capped at this x the Bernoulli speed sqrt(2 Ps / rho) (the line's flow lags a closing area: 1.3-1.4 x the limit into a p)
const float kNoiseAreaExp = param("LINE_NOISE_AREA_EXP", 1.5155573f);  // 2026-11-05: the constriction noise drive x (LINE_NOISE_AREA_REF / area)^this. The Re^2 law (U^2 d^2 ~ u^2 / a) grows as a groove opens on the glottis-limited flow (u ~ a): the s of "save" rose 12 dB over its last 13 ms as its groove opened and its corner fell 8.8 -> 2.1 kHz, then the fade cut it (Kokoro: a 20 dB fade over the last 30 ms). Stevens: the source goes as U^3 / A^2.5. 0: as before.
const float kNoiseAreaRef = param("LINE_NOISE_AREA_REF", 0.05f);  // cm^2 (an s groove): no change there
const float kNoiseAreaTarget = param("LINE_NOISE_AREA_TARGET", 0.0009813799f);  // > 0: the reference is instead the constriction's own area (the smaller end of its glide, at least this many cm^2), and the factor never exceeds 1: every steady fricative keeps its level, only the glides out of and into a groove or a closure are damped (the absolute law cut z v ð by 15 dB and lifted f ʃ by 8: Whisper -45)
const float kNoiseStSnap = param("LINE_NOISE_ST_SNAP", 0.0f);  // 2026-11-05: > 0: a smoothed corner follows its target at once while its jet is silent, so a hiss starts at its own corner and the smoothing shapes only the sweeps inside it (smoothed from the vowel floor, an s began 20 ms of dull f: Whisper -70 in round 6; the t release into ə swept 11 -> 4 kHz over 30 ms in the female voice)
const float kExpandOnMs = param("LINE_EXPAND_ON_MS", 3.0015464f);  // the voiced closure sink's share eases in over this (ms; 0: a step)
const float kNoiseStSmoothMs = param("LINE_NOISE_ST_SMOOTH_MS", 0.0f);  // ... the corners smoothed over this long (ms; 0: none)   // ... the low-pass corner at this times the high-pass one
const float kGlottisLength = param("LINE_GLOTTIS_L", 1.284385f);  // the glottis's length (cm): its width = area / this
// The glottal jet's own low-pass ratio and floor (2026-11-04): at a p puff's
// 0.6 cm^2 glottis fc ~0.6 kHz put the low-pass at 1.6 kHz, so the aspiration
// had no 2-5 kHz hiss (its 1.5-3 kHz band 12-14 dB under 0.5-1.5 kHz; the
// readers' is level or brighter). <= 0: LINE_NOISE_LPR; the floor in Hz.
const float kGlottalLpRatio = param("LINE_GLOTTAL_LPR", -1.0f);
const float kGlottalLpMin = param("LINE_GLOTTAL_LPMIN", 0.0f);
const float kSlitWidth = param("LINE_SLIT_W", 0.9962687f);         // a fricative slit's width (cm): its height = area / this
const float kWallRadiate = param("LINE_WALL_RADIATE", 10.050009f);  // the walls' volume velocity radiated with the lips' (0: not; 2026-11-04: 10 with LINE_VOICED_SOFT 2, d words right 24 -> 28 of 108, the vowels 2-4 dB less bright above 1 kHz)
const float kWallDamp = param("LINE_WALL_B", 1.019e+04f) /* cost 0.0308115287161875 wobble */;
const float kWallStiff = param("LINE_WALL_K", 3.0e5f);
const float kClosureMin = param("LINE_CLOSURE_MIN", 0.0020072206f);  // the smallest constriction area the hump reaches (cm^2)
// Viscous loss along the wall: R = S/A^2 sqrt(omega rho mu / 2) per cm,
// taken at this frequency (mu = 1.86e-4 poise).
const float kViscousHz = param("LINE_VISC_HZ", 1000.0f);
constexpr float kMu = 1.86e-4f;
}  // namespace

// The losses that grow with frequency: the viscous loss above its reference
// frequency, the heat conduction into the walls, the soft and uneven walls.
// Without them the line's upper resonances rang with 12-25 Hz bandwidths
// (u: F3-F5 at 12-16 Hz, i: F2 at 23 Hz; a vocal tract has 60-200 Hz there):
// tones that ring ~20 ms after every pulse and glide with the shape, the
// whistling between phonemes (2026-11-03). A resistance on the high-passed
// flow, du/dt = -2 pi B u_hp: B Hz of bandwidth well above the corner, half of
// it at the corner, little at F1. Passive (Re{s/(s+wc)} >= 0), so stable.
const float kHfBandwidth = param("LINE_HF_BW", 101.453446f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */;  // Hz added well above the corner
const float kHfCorner = param("LINE_HF_FC", 3000.0f) /* cost 1814 whisper-run1-phon_err */;    // Hz

void TransmissionLine::hfLoss(int nb) {
  // (setHfF2: a loss whose bandwidth grows as f^2, the flow diffusing along the line: u_k += D (u_{k-1} - 2 u_k + u_{k+1});
  // a mode of wavenumber 2 pi f / c decays at D (2 pi f dx / c)^2 per substep; no effect on a steady flow)
  if (hfF2Bw_ > 0) {
    const float w = 2.0f * kPiL * 8000.0f * dx_ / kC, d = std::min(0.4f, hfF2Bw_ * kPiL * dt_ / (w * w));
    float prev = u_[0];
    for (int k = 1; k < nb; ++k) {
      const float cur = u_[k];
      u_[k] = cur + d * (prev - 2.0f * cur + u_[k + 1]);
      prev = cur;
    }
  }
  if (hfK_ <= 0) return;
  for (int k = 1; k <= nb; ++k) {
    const float hp = u_[k] - uLp_[k];
    u_[k] -= hfK_ * hp;
    uLp_[k] += hfA_ * (u_[k] - uLp_[k]);
  }
}

void TransmissionLine::setHfLoss(float bandwidthHz, float cornerHz) {
  hfK_ = 2.0f * kPiL * bandwidthHz * dt_;
  hfA_ = 1.0f - std::exp(kM2PiL * cornerHz * dt_);
}

void TransmissionLine::init(float length, float fs, int sections) {
  fs_ = fs;
  dt_ = 1.0f / (kSubsteps * fs);
  n_ = std::min(kMaxSections, std::max(2, static_cast<int>(std::floor(length * kCourant / (kC * dt_)))));
  // A fixed count (Jyrki: 64, four AVX-512 blocks) as long as it keeps the
  // Courant number under 0.98: at 48 kHz x 4 and 17.5 cm, 64 sections of
  // 2.7 mm give 0.67; the formants moved under 1 % against 86 sections.
  if (sections > 0) n_ = std::min({sections, kMaxSections, static_cast<int>(length * 0.98f / (kC * dt_))});
  dx_ = length / n_;
  invDx_ = 1.0f / dx_;
  blocks_ = (n_ + kBlock - 1) / kBlock;
  for (int k = 0; k < kMaxSections; ++k) {
    a_[k] = aTarget_[k] = k < n_ ? 3.0f : 0.0f;
    aStep_[k] = 1;
    wallV_[k] = wallX_[k] = 0;
  }
  for (int k = 0; k < kMaxSections + kBlock; ++k) u_[k] = p_[k] = uLp_[k] = 0;
  hfK_ = 2.0f * kPiL * kHfBandwidth * dt_;
  hfA_ = 1.0f - std::exp(kM2PiL * kHfCorner * dt_);
  uL_ = uN_ = prevUN_ = radiated_ = 0;
  glideLeft_ = 0;
  coefficients();
}

void TransmissionLine::setLength(float length) {
  // Never below what keeps c dt / dx under 0.98 for this many sections.
  dx_ = std::max(length / n_, kC * dt_ / 0.98f);
  invDx_ = 1.0f / dx_;
}

void TransmissionLine::setAreas(const std::vector<float>& a, bool glide) {
  for (int k = 0; k < n_; ++k) {
    aTarget_[k] = std::min(std::max(k < static_cast<int>(a.size()) ? a[k] : a.back(), kClosureMin), 1e3f);
    if (glide) {
      aStep_[k] = std::pow(aTarget_[k] / a_[k], 1.0f / kGlide);
    } else {
      a_[k] = aTarget_[k];
      aStep_[k] = 1;
    }
  }
  glideLeft_ = glide ? kGlide : 0;
  if (!glide) coefficients();
}

void TransmissionLine::coefficients() {
  const float viscK = std::sqrt(kPiL * kViscousHz * kRho * kMu);  // sqrt(omega rho mu / 2)
  const float comp = kRho * kC * kC;
  const float wDen = 1.0f / (1.0f + dt_ * kWallDamp * wallLoss_ * kWallMassInv);
  const float dtOverRhoDx = dt_ * invDx_ * (1.0f / kRho);
  const float halfDtOverDx = 0.5f * dt_ * invDx_;
  const float dtComp = dt_ * comp * invDx_;
  // (This runs every substep while the areas glide, and the oral line always glides: a new shape every 4th sample.
  // So only the n sections are computed, in loops without branches or carried values that vectorize; the unused
  // interfaces and sections up to the last whole block keep the neutral values. 2026-11-25: 7 % of a render.)
  const int n = n_, nb = blocks_ * kBlock;
  cU_[0] = 0; gU_[0] = 1; oriInv_[0] = 0;
  for (int k = n; k <= nb; ++k) { cU_[k] = 0; gU_[k] = 1; oriInv_[k] = 0; }
  // Interfaces 1..n-1 (0 is the glottis, n the lips: set each substep).
  for (int k = 1; k < n; ++k) {
    const float ah = std::min(a_[k], a_[k - 1]);
    const float aw = std::max(a_[k], a_[k - 1]);
    const float s = 2.0f * std::sqrt(kPiL * ah);
    const float lm = lossMul_[k] > 0 ? lossMul_[k] : 1.0f;
    const float invAh = 1.0f / ah;
    const float r = s * invAh * invAh * viscK * viscLoss_ * lm;  // dyn s / cm^6 per cm
    cU_[k] = ah * dtOverRhoDx;
    gU_[k] = 1.0f / (1.0f + dt_ * ah * (1.0f / kRho) * r);  // the loss, implicit
    // The Borda-Carnot loss of an area change: rho u|u| / (2 A^2) times
    // (1 - A_small / A_large)^2 at this interface, applied implicitly in
    // substep as 1 / (1 + oriInv |u|). Zero in a uniform tube; over a
    // contraction it sums to one dynamic head. (Applied as a full head at
    // every interface it added ~30 Hz to every formant's bandwidth, growing
    // with the section count, review 2026-11-01.)
    const float contraction = 1.0f - ah / aw;
    oriInv_[k] = halfDtOverDx * invAh * contraction * contraction;
  }
  // LINE_JET_SEP (0: off, 1: full; 2026-11-23): the jet out of a narrowing stays separated while the tract widens, so
  // an expansion loses (1/Amin - 1/Aend)^2 of rho u^2 / 2 whatever the number of sections it is drawn over: at each
  // widening interface the difference of (1/Amin - 1/a)^2 across it (it telescopes; the first one after the minimum is
  // the term above). Step by step the sh's exit (0.07 -> 1.7 cm^2 over ten sections of ratio ~0.75) lost a small part
  // of one head where the s's (two sections) lost all of it: too much flow through the sh, and so little differential
  // resistance that the back cavity rang through it (the hiss a comb, notches every 1.05 kHz from 4.8 to 11.4 kHz).
  static const float kJetSep = param("LINE_JET_SEP", 0.0f);
  if (kJetSep > 0) {
    float jetMin = a_[0];
    for (int k = 1; k < n; ++k) {
      if (a_[k] < a_[k - 1]) jetMin = a_[k];
      else if (a_[k] > a_[k - 1] && jetMin > 0) {
        const float ah = std::min(a_[k], a_[k - 1]);
        const float i0 = 1.0f / jetMin, d1 = i0 - 1.0f / a_[k], d0 = std::max(0.0f, i0 - 1.0f / a_[k - 1]);
        const float sep = halfDtOverDx * ah * (d1 * d1 - d0 * d0);
        oriInv_[k] += kJetSep * (sep - oriInv_[k]);
      }
    }
  }
  for (int k = 0; k < n; ++k) {
    cP_[k] = dtComp / a_[k];
    sWall_[k] = 2.0f * std::sqrt(kPiL * a_[k]) * dx_;
    wA_[k] = wDen;
    wB_[k] = dt_ * kWallMassInv * wDen;
  }
  for (int k = n; k < nb; ++k) cP_[k] = sWall_[k] = wA_[k] = wB_[k] = 0;
  // Lips: Flanagan's piston in a baffle, R_r = 128 rho c / (9 pi^2 A) and
  // L_r = 8 rho / (3 pi^2 a) in parallel.
  const float aLip = a_[n_ - 1];
  const float radius = std::sqrt(aLip / kPiL);
  const float rr = 128.0f / (9.0f * kPiL * kPiL) * kRho * kC / aLip;
  const float lr = 8.0f / (3.0f * kPiL) * kRho / (kPiL * radius);
  rrInv_ = 1.0f / rr;
  lrInvDt_ = dt_ / lr;
  const float t = std::clamp((aLip - 0.005f) / 0.025f, 0.0f, 1.0f);
  lipOpen_ = t * t * (3.0f - 2.0f * t);
}

float TransmissionLine::substepSubglottal(float lungPressure, float uGlottis) {
  const int n = n_, nb = blocks_ * kBlock;
  // The lungs: a matched source. The flow into section 0 is what a wave of
  // the pressure difference carries: (ps - p_0) A / (rho c), so a wave
  // arriving from the glottis is absorbed and the DC pressure is held.
  // The matched source alone would drop the DC pressure by rho c u / A (4800
  // dyn/cm^2 at 300 cm^3/s): the respiratory system holds the subglottal
  // pressure, so the lungs supply the slowly tracked mean flow on top.
  if (glideLeft_ > 0) { --glideLeft_; for (int k = 0; k < n; ++k) a_[k] *= aStep_[k]; coefficients(); }
  // The DC is supplied as a pressure source through the respiratory
  // resistance (kLungR, ~3 dyn s/cm^5: bronchi and lungs), the AC sees the
  // matched impedance. (Tracking the mean flow instead kept pumping air into
  // a sealed mouth: behind a b the cavity reached 1.3x the lung pressure,
  // the folds were blown 0.5 cm^2 open and fell silent, 2026-11-03.)
  const float dp = lungPressure - p_[0];
  lungDc_ += (dp - lungDc_) * (dt_ / 0.003f);  // the slow (DC) part of the pressure difference
  u_[0] = lungDc_ * kLungRInv + (dp - lungDc_) * a_[0] * (1.0f / (kRho * kC));
  for (int k = 1; k <= nb; ++k) u_[k] = gU_[k] / (1.0f + oriInv_[k] * std::fabs(u_[k])) * (u_[k] - cU_[k] * (p_[k] - p_[k - 1]));
  hfLoss(nb);
  u_[n] = uGlottis;  // the glottal end: the folds' flow leaves here
  for (int k = 0; k < nb; ++k) {
    const float v = wA_[k] * wallV_[k] + wB_[k] * (p_[k] - kWallStiff * wallX_[k]);
    wallX_[k] += dt_ * v;
    wallV_[k] = v;
  }
  for (int k = 0; k < nb; ++k) p_[k] -= cP_[k] * (u_[k + 1] - u_[k] + sWall_[k] * wallV_[k]);
  return p_[n - 1];
}

float TransmissionLine::substep(float u0) {
  const int n = n_, nb = blocks_ * kBlock;
  // Debug: PUHUJA_LINE_TRACE=file writes u0, p0, lip flow per substep (the oral line only: n > 50).
  static FILE* trace = [] { const char* f = std::getenv("PUHUJA_LINE_TRACE"); return f ? std::fopen(f, "w") : nullptr; }();
  if (trace && n > 50) std::fprintf(trace, "%g\t%g\t%g\t%g\t%g\n", u0, p_[0], uN_, traceExtra_, traceExtra2_);
  if (glideLeft_ > 0) {
    --glideLeft_;
    for (int k = 0; k < n; ++k) a_[k] *= aStep_[k];
    coefficients();
  }
  // Momentum over all interfaces 1..nb (cU is 0 past n-1, so those stay 0);
  // u_[0] and u_[n] are set outside the loop.
  u_[0] = u0;
  for (int k = 1; k <= nb; ++k) u_[k] = gU_[k] / (1.0f + oriInv_[k] * std::fabs(u_[k])) * (u_[k] - cU_[k] * (p_[k] - p_[k - 1]));
  hfLoss(nb);
  // Lips: Flanagan's piston, R and L in parallel, driven by p_{n-1}.
  // Closed lips are a rigid end: the load's flow fades with the opening and
  // the inertance's stored flow drains.
  uL_ = lipOpen_ * (uL_ + lrInvDt_ * p_[n - 1]);
  uN_ = lipOpen_ * (p_[n - 1] * rrInv_ + uL_);
  radiated_ = (uN_ - prevUN_) * (1.0f / dt_);
  prevUN_ = uN_;
  u_[n] = uN_;
  // Walls, then continuity, over whole blocks (zero coefficients past n).
  for (int k = 0; k < nb; ++k) {
    const float v = wA_[k] * wallV_[k] + wB_[k] * (p_[k] - kWallStiff * wallX_[k]);
    wallX_[k] += dt_ * v;
    wallV_[k] = v;
  }
  for (int k = 0; k < nb; ++k) p_[k] -= cP_[k] * (u_[k + 1] - u_[k] + sWall_[k] * wallV_[k]);
  return p_[0];
}


// --- The whole airway ---------------------------------------------------

namespace {
const float kVelumScale = param("LINE_VELUM", 6.0575128f) /* cost 1814 whisper-run1-phon_err */;  // the port area per unit of the velum opening
// The nasal line, velum to nostrils (cm). With the velum at 55 % of a 17.5 cm
// tract (9.6 cm from the glottis) the nasal path is 20.6 cm, 3.1 cm longer
// than the oral one (Jyrki 2026-11-02: "the nasal tract may be longer,
// perhaps 3 cm, than the mouth tract"); anatomical values 12-12.5 cm would
// make it 4-4.6 cm longer.
const float kNasalLength = param("LINE_NASAL_CM", 12.392948f);
const float kReynoldsCrit = param("LINE_RE_CRIT", 1400.0f);  // 1800 until 2026-11-06: the voiced z's jet (Re 800-1600) never hissed, "is" heard as "in"; 1400 with LINE_AG_Z 0.16, LINE_Z_AC 1 (zA): Whisper 801 -> 853, Jyrki "sounded good"
const float kFricA0 = param("LINE_FRIC_A0", 0.11673874f);  // the frication source at full strength below this constriction area (cm^2; the widest slit, ð, is 0.084) ...
const float kFricA1 = param("LINE_FRIC_A1", 0.2156f) /* cost 1814 whisper-run1-phon_err */;  // ... and silent above this (the approximants: 0.25-0.35; at 0.3 the r after a vowel, 0.25 cm^2, still hissed +17 dB)
constexpr float kMuAir = 1.86e-4f;
// Re = 2 rho |u| / (mu L) (slit) or 2 rho |u| / (mu sqrt(pi A)) (round duct): the constant factors once.
const float kReSlitFactor = 2.0f * TransmissionLine::kRho / (kMuAir * kGlottisLength);
constexpr float kReRoundFactor = 2.0f * TransmissionLine::kRho / kMuAir;

// The generator's first form, on int64_t: its += overflowed and rot() shifted negative values left, both undefined
// behaviour (UBSan; review cpp.md B4), and its >> is arithmetic, so rot() was no rotation. Here the same sequence in
// unsigned arithmetic (wrapping, as the compiler did it; bit-identical output): >> stays arithmetic (sar).
uint64_t sar(uint64_t a, int shift) { return static_cast<uint64_t>(static_cast<int64_t>(a) >> shift); }
uint64_t rotOld(uint64_t a, int shift) { return sar(a, shift) | (a << (64 - shift)); }

// LINE_NOISE_RNG 1: filters.h's nextNoise, uniform in [-1, 1). The first form's output lies in [-2, 0) (mean -1.12,
// 62 % below -1): harmless while every consumer is high-passed, but a thump on each fricative on a path that is not
// (review dsp.md B4). A different sequence, so a different draw of every hiss: not yet heard.
const bool kNoiseRngFixed = param("LINE_NOISE_RNG", 0.0f) > 0;
}  // namespace

float VocalTractLine::noise() {
  if (kNoiseRngFixed) return nextNoise(rng_);
  rng_ += rotOld(rng_, 15);
  rng_ ^= sar(rng_, 17);
  rng_ += rotOld(rng_, 31);
  rng_ ^= rotOld(rng_, 23);
  rng_ += rotOld(rng_, 11);
  return static_cast<float>(static_cast<int64_t>(rng_)) * (1.0f / 9223372036854775808.0f) - 1.0f;
}

void TransmissionLine::addSinus(int k, float volume, float neckArea, float neckLength, float q) {
  if (nSinus_ >= 4) return;
  Sinus& s = sinus_[nSinus_++];
  s.k = k;
  s.m = kRho * neckLength / neckArea;             // inertance of the neck
  s.cInv = kRho * kC * kC / volume;               // 1 / compliance of the volume
  const float w0 = std::sqrt(s.cInv / s.m);       // resonance (rad/s)
  s.r = s.m * w0 / q;                              // resistance for the quality factor q
  s.dtOverM = dt_ / s.m;                           // the per-substep factors (dt_ is set: init first)
  s.invDen = 1.0f / (1.0f + dt_ * s.r / s.m);
  s.u = s.p = 0;
}

void TransmissionLine::retuneSinus(int i, float volume, float neckArea, float neckLength, float q) {  // (NASAL2 agent33)
  if (i < 0 || i >= nSinus_) return;
  Sinus& s = sinus_[i];
  s.m = kRho * neckLength / neckArea;
  s.cInv = kRho * kC * kC / volume;
  const float w0 = std::sqrt(s.cInv / s.m);
  s.r = s.m * w0 / q;
  s.dtOverM = dt_ / s.m;
  s.invDen = 1.0f / (1.0f + dt_ * s.r / s.m);
}

// The sinuses in two halves around the line's own substep, so that the neck
// flows, like the line's interface flows, come from the pressures before the
// substep, and every pressure (the section's, the volume's) moves with the new
// flows: the staggered leapfrog, which conserves energy. In one step after the
// line, the neck read pressures the line had already advanced, and the
// maxillary branch (464 Hz) grew without any source whenever the velum was
// open behind a shut mouth: a hum up to 18 dB over a final n once the breath
// had stopped, before the next phrase (2026-11-03).
void TransmissionLine::sinusFlows() {
  for (int i = 0; i < nSinus_; ++i) {
    Sinus& s = sinus_[i];
    s.u = (s.u + s.dtOverM * (p_[s.k] - s.p)) * s.invDen;  // the loss implicit
  }
}

void TransmissionLine::sinusPressures() {
  for (int i = 0; i < nSinus_; ++i) {
    Sinus& s = sinus_[i];
    s.p += dt_ * s.cInv * s.u;
    if (!(std::fabs(s.p) < 1e7f && std::fabs(s.u) < 1e6f)) s.u = s.p = 0;  // never let a branch blow up into the output
    p_[s.k] -= cP_[s.k] * s.u;  // the volume drawn from the section
  }
}

const float kNoiseAsPressure = param("LINE_NOISE_P", 2.989489f);  // as a flow drive the hiss grew only 0.45 dB per dB of gain (s stuck 18 dB under the vowels); as a section pressure x2 it sits at -9, Kokoro -5..-7 (2026-11-02)  // > 0: inject the turbulence as a section pressure (scaled by this) instead of an interface flow
const float kExpandHold = param("LINE_EXPAND_HOLD", 0.6003093f);  // a voiced closure's cavity pressure is held near this share of the lungs'
const float kExpandLp = param("LINE_EXPAND_LP", 52.262566f);  // > 0: the voiced closure's expansion follows the cavity pressure low-passed at this (Hz), not as an acoustic shunt; 0: the instantaneous pressure (b's voice bar silent)
const float kHumpReleaseMs = param("LINE_HUMP_RELEASE_MS", 40.0f);  // the constriction's area releases with this time constant beyond kHumpFastTo
// The hump's sampled minimum held at the commanded area (0: off, the cos^2 sampled at the section centres as before; 1: the
// hump's profile divided by its largest sampled value, so the section nearest the centre always reaches acEff; between, a
// partial power). A narrow hump (hw 0.5 cm, 0.28 cm sections) gliding along the tube reached its area only when its
// centre sat on a section: in between, 0.028 -> 0.050 cm^2 at the f -> ɹ of "different" (place 0.95 -> 0.68 while the
// f's 0.03 cm^2 was still shut), the flow under 9 kPa beating 81 -> 124 cm^3/s once per section crossed (~1 kHz,
// falling): a 300-1500 Hz pulse +27 dB in 1 ms, the "ping-knock at the ff" (Jyrki 2026-11-11, PUHUJA_ENERGY_TRACE).
const float kHumpNorm = param("LINE_HUMP_NORM", 1.0076912f);
const float kHumpFastTo = param("LINE_HUMP_FAST_TO", 2.0f);  // (2.0: off; 0.4 held F2 low through the vowel and affricated the t puff)        // ... after opening fast (6 ms) to this area (cm^2): the acoustic release
const float kNoiseDx = param("LINE_NOISE_DX", 0.19959644f);  // the turbulence source this far (cm) past the hump's downstream edge
const float kNasalCavityArea = param("LINE_NASAL_CAVITY_A", 3.70378914f);  // the main nasal cavity, both sides (cm^2)
const float kNasalPortArea = param("LINE_NASAL_PORT_A", 1.50579897f);      // the nasopharynx behind it (cm^2)
const float kNostrilArea = param("LINE_NOSTRIL_A", 1.53975054f) /* cost 1814 whisper-run1-phon_err */;  // both nostrils together (cm^2; 1.2 put the n level with the vowel, the readers 9-14 dB under)

void VocalTractLine::setFemale(bool female) {
  // LINE_HF_BW_F: the female voice's oral HF loss (< 0: LINE_HF_BW; round Z VQ agent28)
  static const float kHfBwF = param("LINE_HF_BW_F", 197.193336f);
  if (kHfBwF >= 0) oral_.setHfLoss(female ? kHfBwF : kHfBandwidth, kHfCorner);
  if (kHfBwF >= 0) sub_.setHfLoss(female ? kHfBwF : kHfBandwidth, kHfCorner);  // (the trachea's from the same knob)
  oralHfBw_ = kHfBwF >= 0 && female ? kHfBwF : kHfBandwidth;
  // LINE_HF_F2_BW (Hz at 8 kHz, 0: off; _F the female's, < 0: the male's): the oral line's loss ~ f^2 (radiation, the
  // boundary layers, and above 4-5 kHz the cross modes that scatter what a 1-D line keeps as sharp whole-tract modes).
  // 2026-11-23: the sh's hiss showed those modes as a comb, teeth every c / 2L = 1.05 kHz from 6 to 10 kHz, 4-11 dB.
  static const float kF2 = param("LINE_HF_F2_BW", 0.0f), kF2F = param("LINE_HF_F2_BW_F", -1.0f);
  oral_.setHfF2(female && kF2F >= 0 ? kF2F : kF2);
  // Round AB (VQ3 agent30): the piriform fossae as a Helmholtz side branch on the oral line LINE_PIRIFORM_AT cm above
  // the glottis, tuned to LINE_PIRIFORM_HZ (0: none; _F: the female's, < 0: the male's): its zero makes the trough that
  // real voices have between F3 / F4 and the 5 kHz region (Dang & Honda 1997). deployed156's male LTAS sat 6-9 dB over
  // 11labs and Kokoro onyx at 3-4 kHz. LINE_PIRIFORM_V: the volume (cm^3, both sides), _A the neck area, _Q the loss.
  static const float pfHz = param("LINE_PIRIFORM_HZ", 0.0f), pfHzF = param("LINE_PIRIFORM_HZ_F", 0.0f);
  static const float pfV = param("LINE_PIRIFORM_V", 2.0f), pfA = param("LINE_PIRIFORM_A", 1.0f);
  static const float pfQ = param("LINE_PIRIFORM_Q", 4.0f), pfAt = param("LINE_PIRIFORM_AT", 1.5f);
  // ... and the oral walls' damping per voice (LINE_WALL_B x LINE_ORAL_WALL_X; _F the female's, < 0: the male's):
  // round AA's LINE_WALL_B x1.3 brought both voices' timbre closer, but for every line and both voices at once.
  static const float wallX = param("LINE_ORAL_WALL_X", 1.0f), wallXF = param("LINE_ORAL_WALL_X_F", -1.0f);
  oral_.setWallLoss(female && wallXF >= 0 ? wallXF : wallX);
  const float hz = female && pfHzF >= 0 ? pfHzF : pfHz;
  oral_.clearSinuses();
  oralSinus_ = hz > 0;
  if (oralSinus_) {
    const float w0 = 2.0f * 3.14159265f * hz;
    const float len = TransmissionLine::kC * TransmissionLine::kC * pfA / (pfV * w0 * w0);  // w0^2 = c^2 A / (V L)
    oral_.addSinus(std::max(1, std::min(oral_.sections() - 2, static_cast<int>(pfAt / oral_.dx()))), pfV, pfA, len, pfQ);
  }
}

void VocalTractLine::setOralHfMix(float consBw, float openness) {  // (round AB VQ3: LINE_HF_BW_CONS)
  oral_.setHfLoss(consBw + (oralHfBw_ - consBw) * openness, kHfCorner);
}

void VocalTractLine::init(float fs, float nasalScale) {
  fs_ = fs;
  dt_ = 1.0f / (TransmissionLine::kSubsteps * fs);
  oral_.init(17.5f, fs, kOralSections);
  nasal_.init(nasalScale == 1.0f ? kNasalLength : nasalScale * kNasalLength, fs, kNasalSections);
  static const float kNasalLoss = param("LINE_NASAL_LOSS", 0.98207026f);
  nasal_.setWallLoss(kNasalLoss);  // the nasal walls (mucosa, the sinuses) lose more than the mouth
  // The turbinates split the nose into narrow slits: everything above the
  // murmur's low resonance is damped. With the mouth's losses the n kept its
  // overtones (1-5 kHz only 10-20 dB under the 250 Hz peak; the readers'
  // 25-40), heard as l ("Never." -> "later", "Now." -> "hello", 2026-11-04).
  // Damped like the readers' (1000 Hz above 600) the murmur fell 6 dB and
  // Whisper heard b d instead (10 -> 1 of 52 initial n m words); with wider
  // nostrils and port to win the level back, 8 of 52. Lightly (300 Hz above
  // 1000), with the slower nasal release, 40 of 70 m words were heard
  // starting with m (24 without it; Whisper, 2026-11-04).
  static const float kNasalHfBw = param("LINE_NASAL_HF_BW", 100.0f), kNasalHfFc = param("LINE_NASAL_HF_FC", 1000.0f);
  nasal_.setHfLoss(kNasalHfBw, kNasalHfFc);
  nasHfFc_ = kNasalHfFc;
  nasalScale_ = nasalScale;
  static const float kNasalVisc = param("LINE_NASAL_VISC", 4.2286265f);
  nasal_.setViscLoss(kNasalVisc);  // (30: the m of "his mortal", starting from the voiceless z, fell to -26 dB)  // the turbinates: far more surface per volume than a tube
  // The nasal passage: the port and nasopharynx, the cavity, the nostrils.
  std::vector<float> na(nasal_.sections());
  for (int k = 0; k < nasal_.sections(); ++k) {
    const float x = (k + 0.5f) * nasal_.dx();
    na[k] = x < 2.0f * nasalScale ? kNasalPortArea : x < 9.0f * nasalScale ? kNasalCavityArea : kNostrilArea;
  }
  nasal_.setAreas(na, false);
  // The paranasal sinuses as Helmholtz side branches (Dang & Honda 1994: the
  // maxillary pair gives the murmur its zero near 500 Hz, the frontal and
  // sphenoid ones near 1-1.5 kHz; without them the line's n carried 1-2 kHz
  // only 17 dB under its low band, Kokoro 27-45, and "known" was "whole").
  {
    static const float v1 = param("LINE_SINUS1_V", 22.097188f), a1 = param("LINE_SINUS1_A", 0.10478747f), l1 = param("LINE_SINUS1_L", 0.60069922f);
    static const float v2 = param("LINE_SINUS2_V", 0.0f), a2 = param("LINE_SINUS2_A", 0.09755848f), l2 = param("LINE_SINUS2_L", 0.48846771f);
    static const float q = param("LINE_SINUS_Q", 5.0f);
    // The maxillary pair as two cavities, one LINE_SINUS1_SPLIT times the other's volume (0: one cavity for both).
    // Jyrki 2026-11-10: "make one sinus 12 % larger in volume than the other". Each half gets half the neck, so the
    // pair's mean resonance stays near the single cavity's and the two zeros part by sqrt(split).
    static const float split = param("LINE_SINUS1_SPLIT", 1.12f);
    nasal_.clearSinuses();
    if (v1 > 0 && split > 0) {
      nasal_.addSinus(static_cast<int>(5.0f * nasalScale / nasal_.dx()), v1 * split / (1.0f + split), 0.5f * a1, l1, q);
      nasal_.addSinus(static_cast<int>(5.0f * nasalScale / nasal_.dx()), v1 / (1.0f + split), 0.5f * a1, l1, q);
    } else if (v1 > 0) {
      nasal_.addSinus(static_cast<int>(5.0f * nasalScale / nasal_.dx()), v1, a1, l1, q);
    }
    if (v2 > 0) nasal_.addSinus(static_cast<int>(7.0f * nasalScale / nasal_.dx()), v2, a2, l2, q);
  }
  velumK_ = static_cast<int>(0.55f * oral_.sections());
#if defined(__SSE__)
  _mm_setcsr(_mm_getcsr() | 0x8040);  // flush denormals to zero: silence decays into them and the loops crawl
#endif
  // The trachea: 12 cm of 2.5 cm^2 from the lungs to the glottis.
  sub_.init(12.0f, fs, 32);
  sub_.setSubglottal(true);
  sub_.setAreas(std::vector<float>(sub_.sections(), 2.5f), false);
  pSub_ = 0;
  // 4th-order Butterworth at 18 kHz at the substep rate before the 4:1
  // decimation (one pole at 16 kHz let 24-96 kHz fold back nearly flat).
  aa1_.set(18000.0f, TransmissionLine::kSubsteps * fs, 0.5412f);
  aa2_.set(18000.0f, TransmissionLine::kSubsteps * fs, 1.3066f);
  // The walls' sound passes through the cheeks and the neck, which pass only
  // the low frequencies. Unfiltered, b's voice bar under a strong expansion
  // had its 2-8 kHz only 6 dB under the whole band (Kokoro's: 35-45 dB under)
  // (2026-11-04; 0: off).
  static const float kWallRadLp = param("LINE_WALL_RAD_LP", 0.0f);
  wallLpHz_ = kWallRadLp;
  if (wallLpHz_ > 0) wallLp_.set(wallLpHz_, TransmissionLine::kSubsteps * fs, 0.7071f);
  uAcc_ = prevMean_ = 0;
  subCount_ = 0;
  {
    static const float hp = param("LINE_ASP_HP", 0.0f);
    aspHpA_ = aspHpBaseA_ = hp > 0 ? 1.0f - std::exp(kM2PiL * hp * dt_) : 0.0f;
    static const float nhp = param("LINE_NOISE_HP", 1500.0f);  // the turbulence noise's high-pass corner (Hz; 0: none; 1500 on 2026-11-04, Jyrki: the s's 0.5-2 kHz bands 4-8 dB toward Kokoro's)
    noiseHpA_ = nhp > 0 ? 1.0f - std::exp(kM2PiL * nhp * dt_) : 0.0f;
  }
  static const float kNoiseLp = param("LINE_NOISE_LP", 5000.0f);
  aaNoise_ = 1.0f - std::exp(kM2PiL * kNoiseLp /* 2026-11-04: the search had 2265 (s heard as f); 6000: initial s 3 -> 8 of 10 (Whisper), 4000 matched Kokoro s level; Jyrki: 5000 */ * dt_);  // the readers' s peaks at 8-12 kHz (goog "house"); at 5.5 kHz the line's s had its energy at 4-6 kHz
  aaNoiseDef_ = aaNoise_;
  out_ = uNasal_ = an_ = 0;
}

void VocalTractLine::setNoiseLp(float hz, bool asDefault) {
  if (hz <= 0) { aaNoise_ = aaNoiseDef_; return; }
  aaNoise_ = 1.0f - std::exp(kM2PiL * hz * dt_);
  if (asDefault) aaNoiseDef_ = aaNoise_;
}

void VocalTractLine::setShape(const TractShape& shape, float x, float ac, float an) {
  setShapeAreas(shape.sections(oral_.sections()), shape.length, x, ac, an);
}

void VocalTractLine::setShapeAreas(std::vector<float> a, float length, float x, float ac, float an) {
  // The length is the speaker's, 15.5-19.5 cm: a vowel's shape may not stretch
  // the tract (the ɪ search took it to 23 cm to lower every formant,
  // 2026-11-02). (Re-initialising the line here wiped its state every
  // update, 2026-11-01.)
  const float shapeLength = std::clamp(length + closureExtra_, 15.5f, 19.5f);  // (the hump's place x is a share of the tube that exists: with an unclamped 32 cm every constriction fell past the lips, 2026-11-02)
  oral_.setLength(shapeLength);
  noiseK_ = -1;
  // The constriction: a cosine^2 hump in log area down to ac, 1 cm wide each
  // way, at x. Its depth fades in as ac falls from 1.0 to 0.7 cm^2 (switched
  // on at 1.0 it stepped the area 3:1 at every consonant onset, review
  // 2026-11-01); the noise goes in just downstream of its centre.
  // The tongue (or lips) leave a constriction slowly: the hump's depth
  // follows the opening at once when closing, but releases with a time
  // constant of kHumpReleaseMs, so the formants glide from the stop's locus
  // into the vowel over tens of milliseconds. Tied to the opening alone the
  // hump vanished within the release and b, d, g, p all started the vowel
  // at its own F2: no place cue, "abolish" heard as "adolish" (2026-11-03).
  // (the depth alone did nothing: the hump's area target is the opening
  // itself, so once the opening passed 1 cm^2 the hump was gone whatever its
  // depth. The constriction's own area, humpAc_, closes at once, opens fast
  // to kHumpFastTo, the acoustic release, and from there slowly.)
  const float step = 4.0f / (fs_ * 0.001f);  // this is called every 4th sample: ms per call
  if (ac <= humpAc_) humpAc_ = ac;
  else if (humpAc_ < kHumpFastTo) humpAc_ += (std::min(ac, kHumpFastTo) - humpAc_) * std::min(1.0f, step / 6.0f);
  else humpAc_ += (ac - humpAc_) * std::min(1.0f, step / kHumpReleaseMs);
  const float acH = humpAc_;
  const float depth = 1.0f - std::clamp((acH - 0.7f) / 0.3f, 0.0f, 1.0f);
  if (x >= 0 && depth > 0) {
    // (floor 0.01 cm^2 let 35 cm^3/s leak through every stop closure, Re 2400:
    // a faint hiss in each closure; 0.002 leaks 7 cm^3/s, below the gate)
    const float acEff = std::max(acH, kClosureMin);
    const float xc = x * shapeLength;
    // LINE_POST_CH_HW (cm, 0: off; 2026-11-23): the post-alveolar groove (the hump of LINE_POST_HW, 4 cm each way) as a
    // channel instead: this half-width, its middle share LINE_POST_CH_FLAT at the constriction's area, cos^2 ends. The
    // 8 cm log-area hump is an exponential horn (flare 1.45 / cm, cutoff m c / 4 pi = 4 kHz): above 4-5 kHz nothing
    // reflects at it, the sh's hiss took the whole tract's modes (a comb every c / 2L = 1.05 kHz up to 12 kHz) and its
    // source, at the hump's edge, sat at the lips.
    static const float kPostChHw = param("LINE_POST_CH_HW", 0.0f), kPostChFlat = param("LINE_POST_CH_FLAT", 0.55f);
    // (by how far the half-width, which glides, is on its way from 1 cm to the groove's: no step where a sh meets an n)
    static const float kPostHwRef = param("LINE_POST_HW", 3.9062579f);
    const float postW = kPostChHw > 0 && kPostHwRef > 1.0f ? std::clamp((humpHw_ - 1.0f) / (kPostHwRef - 1.0f), 0.0f, 1.0f) : 0.0f;
    const float hwEff = postW > 0 ? 1.0f + postW * (kPostChHw - 1.0f) : humpHw_, flatEff = postW * kPostChFlat;
    float gNorm = 1.0f;  // LINE_HUMP_NORM: 1 / (the largest sampled cos)^k
    const float humpNorm = humpNorm_ >= 0 ? humpNorm_ : kHumpNorm;  // (setHumpNorm: a p's release into l ɹ, LINE_P_LIQ_NORM)
    if (humpNorm > 0) {
      const float dmin = std::fabs((std::floor(xc / oral_.dx()) + 0.5f) * oral_.dx() - xc) / humpHw_;  // the nearest centre
      const float gmax = std::cos(0.5f * kPiL * std::min(dmin, 1.0f));
      if (gmax > 1e-3f) gNorm = std::pow(gmax, -humpNorm);
    }
    const float lnC = std::log(acEff);  // (loop-invariant)
    for (int k = 0; k < oral_.sections(); ++k) {
      float d = ((k + 0.5f) * oral_.dx() - xc) / hwEff;
      if (std::fabs(d) >= 1.0f) continue;
      if (flatEff > 0) d = std::max(0.0f, std::fabs(d) - flatEff) / (1.0f - flatEff);
      const float g = humpNorm > 0 ? std::min(1.0f, std::cos(0.5f * kPiL * d) * gNorm) : std::cos(0.5f * kPiL * d);
      const float lnA = std::log(a[k]);
      if (lnC < lnA) a[k] = std::exp(lnA + (lnC - lnA) * g * g * depth);
    }
    // The turbulence source sits where the jet leaves the constriction and
    // meets the wide section again (~1.2 cm past the centre): a dipole at a
    // tiny-area interface moves almost no flow (cU ~ A), downstream it does.
    // The Reynolds number is the constriction's own (flow through its centre).
    // The lips rounded over the last 1.5 cm (ʃ: its front cavity needs the
    // rounding to sit near 2.5-3.5 kHz; open-lipped the hiss peaked at 10 kHz).
    // LINE_LIP_CAP_GLIDE_MS (0: at once; 2026-11-23; 10 since 2026-11-25): the rounding comes and goes over this long (in log area). At once,
    // the lips of a sh went 1.5 -> 5.7 cm^2 in one step where it met an n, under the hiss: a 4 ms full-band click.
    static const float kLipGlideMs = param("LINE_LIP_CAP_GLIDE_MS", 10.0f);
    float lipCap = lipCap_;
    if (kLipGlideMs > 0) {
      const float target = std::log(lipCap_ > 0 ? lipCap_ : 8.0f);
      lipCapLn_ += (target - lipCapLn_) * std::min(1.0f, step / kLipGlideMs);
      lipCap = lipCapLn_ < 2.07f ? std::exp(lipCapLn_) : -1.0f;  // (ln 8 = 2.079: open)
    }
    if (lipCap > 0) {
      const float round = depth;
      const float lnC = std::log(lipCap);  // (loop-invariant)
      for (int k = 0; k < oral_.sections(); ++k) {
        if ((k + 0.5f) * oral_.dx() < shapeLength - 1.5f) continue;
        const float lnA = std::log(a[k]);
        if (lnC < lnA) a[k] = std::exp(lnA + (lnC - lnA) * round);
      }
    }
    // A soft closure: the hump's sections lose closureSoft_ times more.
    {
      const int k0 = std::max(1, static_cast<int>((xc - humpHw_) / oral_.dx()));
      const int k1 = std::min(oral_.sections(), static_cast<int>((xc + humpHw_) / oral_.dx()) + 1);
      oral_.setSectionLoss(k0, k1, depth > 0.5f ? closureSoft_ : 1.0f);
    }
    noiseK_ = std::min(oral_.sections() - 2, static_cast<int>((xc + kNoiseDx + hwEff) / oral_.dx()));  // (-1: at the last interface, the lip load took the drive: "votes" lost its s)
    noiseKc_ = std::min(oral_.sections() - 1, static_cast<int>(xc / oral_.dx()));
  }
  if (!(x >= 0 && depth > 0)) oral_.setSectionLoss(0, 0, 1.0f);
  oral_.setAreas(a);
  velumK_ = std::min(oral_.sections() - 2, static_cast<int>(0.55f * oral_.sections()));
  an_ = std::max(an * kVelumScale, 1e-3f);  // never shut by a switch: the orifice loss closes it
  velarG_ = dt_ * an_ * (1.0f / TransmissionLine::kRho) * nasal_.invDx();
  velarInv2A2_ = 1.0f / (2.0f * an_ * an_);
  static int calls = 0;
  static const bool kShapeTrace = std::getenv("PUHUJA_LINE_TRACE") != nullptr;  // (read once: this runs every 4th sample)
  if (kShapeTrace && (calls++ % 50) == 0) {
    float mn = 1e9f; int at = 0;
    for (int k = 0; k < oral_.sections(); ++k) if (a[k] < mn) { mn = a[k]; at = k; }
    std::fprintf(stderr, "shape: x %.3f ac %.3f an %.3f  L %.1f  min area %.3f at %d\n", x, ac, an, shapeLength, mn, at);
  }
}

void VocalTractLine::setNoise(float fricGain, float aspGain, float glottalArea) {
  fricGain_ = fricGain;
  aspGain_ = aspGain;
  glotArea_ = std::max(glottalArea, 0.005f);
}

float VocalTractLine::substep(float u0) {
  // The velum port: a short orifice (a slug of air of the port's area and one
  // section's length) with the Bernoulli loss rho u^2 / (2 an^2), implicit.
  // Without the loss the pressurised oral cavity rang through the closing
  // port at 329 Hz (the cowbell on every nasal before a voiceless sound,
  // Jyrki 2026-11-01). The loss grows as 1/an^2, so a closing port shuts
  // itself; no switch.
  {
    const float rho = TransmissionLine::kRho;
    float dp;
    if (nasKf_ >= 0) {  // (NASAL2 agent33: setNasalCoupling)
      const float tgt = nasKfT_ >= 0 ? nasKfT_ * oral_.sections() : static_cast<float>(velumK_);
      nasKf_ = nasKf_ < tgt ? std::min(tgt, nasKf_ + nasKStep_) : std::max(tgt, nasKf_ - nasKStep_);
      nasKf_ = std::clamp(nasKf_, 1.0f, static_cast<float>(oral_.sections() - 2));
      nasK0_ = static_cast<int>(nasKf_);
      nasKw_ = nasKf_ - nasK0_;
      dp = (1.0f - nasKw_) * oral_.pressureAt(nasK0_) + nasKw_ * oral_.pressureAt(nasK0_ + 1) - nasal_.pressureAt(0);
    } else {
      dp = oral_.pressureAt(velumK_) - nasal_.pressureAt(0);
    }
    const float g = velarG_;  // dt an / (rho dx), set with an_
    const float r = rho * std::fabs(uNasal_) * velarInv2A2_;
    uNasal_ = (uNasal_ + g * dp) / (1.0f + g * r);
  }
  // Turbulence: a dipole source (a pressure difference across an interface)
  // whose strength follows the dynamic head, (Re^2 - Re_c^2) / Re_c^2, with
  // the noise low-passed at ~5.5 kHz; at the glottis with the folds' own area,
  // at the constriction with the section's. The gains are in dyn/cm^2 of
  // pressure difference per unit of that ratio (independent of the substep).
  const float white = noise();
  lpNoise_ += aaNoise_ * (white - lpNoise_);
  lpNoise2_ += aaNoise_ * (lpNoise_ - lpNoise2_);
  // The noise is made locally by the jet and has no reservoir for slow
  // pressure changes: those are carried by the bulk flow, so the source falls
  // off below the jet's Strouhal frequency (~0.2 U / d: 2-3 kHz for an s or an
  // aspiration). Two one-pole high-passes (Jyrki 2026-11-04).
  float noise = lpNoise2_;
  if (noiseHpA_ > 0) {
    noiseHp1_ += noiseHpA_ * (noise - noiseHp1_);
    noise -= noiseHp1_;
    noiseHp2_ += noiseHpA_ * (noise - noiseHp2_);
    noise -= noiseHp2_;
  }
  const float reC2 = kReynoldsCrit * kReynoldsCrit, invReC2 = 1.0f / reC2;
  // With LINE_NOISE_ST > 0 each jet has its own corner, St U / d, from its
  // speed and its width (the glottis: area / its length; the slit: area / its
  // width), 12 dB/octave below it (the k^4 continuity rise of 3-D turbulence,
  // tools/noise_hp_fit.py), updated every 16 substeps (2026-11-04).
  const bool strouhal = kNoiseSt > 0;
  if (strouhal && (stCount_++ & 15) == 0) {
    // The band of each jet: high-pass at fc, low-pass at kNoiseLpRatio x fc
    // (the 3-D turbulence fit: 2.5-5.6). With only the high-pass under the
    // fixed 5 kHz low-pass, the s's fc ~12 kHz left it no hiss (2026-11-04).
    // The corner smoothed in log frequency over LINE_NOISE_ST_SMOOTH_MS: the
    // glottal jet's speed and width swing with every voicing cycle, and the
    // narrow band swept 200 -> 12000 Hz at the pitch rate, a zip in the voice
    // and a ping or a zonk where a slit opened (Jyrki 2026-11-04: "ringing,
    // zipping, zonking, pinging").
    const float uMax = kNoiseUMax > 0 ? kNoiseUMax * std::sqrt(2.0f * std::max(lungPressure_, 1.0f) / TransmissionLine::kRho) : 0.0f;
    auto alphaOf = [&](float flow, float area, float width, float* lp, float jetD, float jetMix, float slitScale, float lpr, float lpMin, float* logFc, float smoothMs, bool active, float fcMin, float fcMax) {
      float U = std::fabs(flow) / std::max(area, 1e-4f);
      if (uMax > 0) U = std::min(U, uMax);
      // The jet: a fixed thickness (jetMix 1), the slit's own height x slitScale (jetMix 0; the corner then
      // sweeps with the closing gesture, Whisper +8..13 of 188 ʃʒʧʤ clips on 2026-11-08), or a geometric blend.
      const float s = std::max(area / width, 1e-3f) * slitScale;
      const float d = jetD > 0 && jetMix > 0 ? (jetMix >= 1 ? jetD : std::exp(jetMix * std::log(jetD) + (1.0f - jetMix) * std::log(s))) : s;
      float fc = std::clamp(kNoiseSt * U / d, fcMin > 0 ? fcMin : 200.0f, fcMax > 0 ? fcMax : kNoiseFcMax);  // (a jet's own range: ʃ ʒ ʧ ʤ, LINE_POST_FC_MIN/MAX)
      if (smoothMs > 0) {
        const float smoothK = 1.0f - std::exp(-16.0f * dt_ / (0.001f * smoothMs));
        const float target = std::log(fc);
        if (*logFc <= 0 || (kNoiseStSnap > 0 && !active)) *logFc = target;
        *logFc += smoothK * (target - *logFc);
        fc = std::exp(*logFc);
      }
      *lp = 1.0f - std::exp(kM2PiL * std::min(std::max(lpr * fc, lpMin), 40000.0f) * dt_);
      return 1.0f - std::exp(kM2PiL * fc * dt_);
    };
    stAlphaG_ = alphaOf(u0, glotArea_, kGlottisLength, &stLpG_, 0, 1, glotSlitScale_, puffLpr_ > 0 ? puffLpr_ : kGlottalLpRatio > 0 ? kGlottalLpRatio : kNoiseLpRatio, kGlottalLpMin, &stLogFcG_,
                        kNoiseStSmoothGMs >= 0 ? kNoiseStSmoothGMs : kNoiseStSmoothMs, aspActive_, 0, 0);
    if (noiseK_ >= 0) stAlphaF_ = alphaOf(oral_.flowAt(noiseKc_), oral_.areaAt(noiseKc_), kSlitWidth, &stLpF_, jetD_, jetMix_, slitScale_, fricLpr_ > 0 ? fricLpr_ : kNoiseLpRatio, 0, &stLogFcF_,
                                         jetSmoothMs_ >= 0 ? jetSmoothMs_ : kNoiseStSmoothFMs >= 0 ? kNoiseStSmoothFMs : kNoiseStSmoothMs, fricActive_, jetFcMin_, jetFcMax_);
  }
  // Each jet's band-limited noise from the shared low-passed white noise.
  // LINE_NOISE_ST_PRELP (2026-11-05): the jets' bands start from the noise
  // under LINE_NOISE_LP (5 kHz, Jyrki's choice of 2026-11-04) instead of the
  // white noise; with the Strouhal bands that low-pass was bypassed and every
  // hiss reached 12+ kHz (Jyrki 2026-11-05: "s, sh, t, z, f, -tion ...
  // excessively bright"). x LINE_NOISE_ST_PRELP_GAIN.
  static const bool stPreLp = param("LINE_NOISE_ST_PRELP", 1.0f) > 0;
  static const float stPreGain = param("LINE_NOISE_ST_PRELP_GAIN", 1.4116297f);
  const float stSrc = stPreLp ? stPreGain * lpNoise2_ : white;
  // ... for the glottal jet too (LINE_NOISE_ST_PRELP_G 1); 0: the glottis keeps
  // the white source (x LINE_NOISE_ST_PRELP_GAIN_G): a t's aspiration is
  // mostly 4-12 kHz (goog "to": 15-20 dB over 0.5-3 kHz); under 5 kHz the
  // tract-filtered low part dominated, a "gr-hu" (Jyrki 2026-11-05).
  static const bool stPreLpG = param("LINE_NOISE_ST_PRELP_G", 1.0f) > 0;
  static const float stGainG = param("LINE_NOISE_ST_PRELP_GAIN_G", 1.0f);
  const float stSrcG = stPreLp && !stPreLpG ? stGainG * white : stSrc;
  auto band = [&](float* s, float a, float lp, float src) {  // s: 2 low-pass then 2 high-pass states
    s[0] += lp * (src - s[0]);
    s[1] += lp * (s[0] - s[1]);
    float x = s[1];
    s[2] += a * (x - s[2]); x -= s[2];
    s[3] += a * (x - s[3]); x -= s[3];
    return x;
  };
  const float noiseG = strouhal ? band(stG_, stAlphaG_, stLpG_, stSrcG) : noise;
  const float noiseF = strouhal ? band(stF_, stAlphaF_, stLpF_, stSrc) : noise;
  if (aspGain_ > 0) {
    // LINE_GLOT_RE_SLIT (2026-11-05): the glottis is a slit, Re = 2 rho |u| /
    // (mu L) with L its length, independent of the gap; the round-duct form
    // (gap -> 0: Re -> infinity) made the aspiration grow as the folds closed
    // after a p t k (a crescendo into the vowel; goog's decays). x
    // LINE_GLOT_RE_SLIT as a scale (the threshold is then LINE_RE_CRIT / it).
    static const float reSlit = param("LINE_GLOT_RE_SLIT", 0.0f);
    const float re = reSlit > 0 ? reSlit * kReSlitFactor * std::fabs(u0)
                                : kReRoundFactor * std::fabs(u0) / std::sqrt(kPiL * glotArea_);
    // High-passed: the white source drove the open tract's low resonances,
    // so a t's aspiration peaked under 200 Hz (the readers' at 3-5 kHz, 20 dB
    // less below 350 Hz; 2026-11-04).
    // LINE_ASP_HP_POLES (2026-11-05): more one-pole stages. With one (6
    // dB/octave) the tract's F1 still lifted the aspiration's low part to the
    // level of its 2.5-12 kHz hiss after a t ("power to": 0 dB; goog and
    // Kokoro: the hiss 15-22 dB over 0.3-2.5 kHz).
    static const int aspHpPolesBase = static_cast<int>(param("LINE_ASP_HP_POLES", 1.0f));
    const int aspHpPoles = aspHpPolesNow_ > 0 ? aspHpPolesNow_ : aspHpPolesBase;
    float n = noiseG;
    if (aspHpA_ > 0) {
      aspHpState_ += aspHpA_ * (n - aspHpState_);
      n -= aspHpState_;
      for (int k = 1; k < aspHpPoles && k < 4; ++k) {
        aspHpMore_[k] += aspHpA_ * (n - aspHpMore_[k]);
        n -= aspHpMore_[k];
      }
    }
    if (csAspMix_ > 0.0f) {  // (VQCS agent33)
      csAspLp_[0] += csAspA_ * (n - csAspLp_[0]);
      csAspLp_[1] += csAspA_ * (csAspLp_[0] - csAspLp_[1]);
      n += csAspMix_ * (csAspLp_[1] - n);
    }
    aspActive_ = re > kReynoldsCrit;
    if (aspActive_) oral_.addFlowDrive(1, aspGain_ * (re * re - reC2) * invReC2 * n);
    if (breathGain_ > 0) {  // (LINE_BREATH: no threshold; LINE_BREATH_MS smooths its envelope: Jyrki heard the instantaneous Re^2 drive as erratic)
      static const float breathMs = param("LINE_BREATH_MS", 0.0f);
      static const float breathA = breathMs > 0 ? 1.0f - std::exp(-1000.0f * dt_ / breathMs) : 1.0f;
      breathEnv_ += breathA * (re * re * invReC2 - breathEnv_);
      // LINE_BREATH_FC (Hz; 0: the aspiration's Strouhal band, whose corner moves with the flow): a fixed band, white
      // noise through two one-pole high-passes at this corner, so the breath's colour does not change with every pulse.
      static const float breathFc = param("LINE_BREATH_FC", 0.0f);
      static const float breathHpA = breathFc > 0 ? 1.0f - std::exp(kM2PiL * breathFc * dt_) : 0.0f;
      float bn = n;
      if (breathFc > 0) {
        bn = lpNoise2_;  // (the LINE_NOISE_LP low-passed source: white was +10 dB over Kokoro at 8-10 kHz)
        breathHp_[0] += breathHpA * (bn - breathHp_[0]); bn -= breathHp_[0];
        breathHp_[1] += breathHpA * (bn - breathHp_[1]); bn -= breathHp_[1];
      }
      oral_.addFlowDrive(1, breathGain_ * breathEnv_ * bn);
    }
  } else aspActive_ = false;
  fricActive_ = false;
  if (noiseK_ >= 0 && fricGain_ > 0) {
    const float u = oral_.flowAt(noiseKc_);
    const float a = oral_.areaAt(noiseKc_);
    // (x fricReMul_, LINE_Z_RE_SCALE: a z's jet as if this much faster or wider, so a slit wide enough to keep the folds
    // going still hisses; 1: off, a multiplication by exactly 1)
    float re = fricReMul_ * kReRoundFactor * std::fabs(u) / std::sqrt(kPiL * a);
    // (= rho U d / mu with d = 2 sqrt(a / pi); LINE_NOISE_U_MAX caps U at the Bernoulli speed)
    if (kNoiseUMax > 0) re = std::min(re, kReRoundFactor * kNoiseUMax * std::sqrt(2.0f * std::max(lungPressure_, 1.0f) / TransmissionLine::kRho) * std::sqrt(a / kPiL));
    const float reRaw = re;  // (LINE_NOISE_SOFT)
    // Only a fricative-sized slit makes a jet that hisses: the source fades out
    // as the constriction widens past kFricA0..kFricA1 (log scale). The Re^2
    // law alone barely depends on the area at a given flow (Stevens' measured
    // source goes as U^3 / A^2.5), and with the voiced flow through them the
    // approximants' 0.25-0.35 cm^2 constrictions hissed: r +33 dB, w +16,
    // l +13, n +7 dB of 3-8 kHz noise, the "smoosh" (2026-11-03).
    // log(a / (A0 f)) / log(A1 / A0) as (log a - log(A0 f)) x a constant: one log per substep.
    static const float invLogRatio = 1.0f / std::log(kFricA1 / kFricA0);
    if (fricLogA0_ == 0) fricLogA0_ = std::log(kFricA0 * fricFade_);
    const float wide = std::clamp((std::log(a) - fricLogA0_ - slitLogW_) * (slitSpan_ != 1.0f ? invLogRatio * slitSpan_ : invLogRatio), 0.0f, 1.0f);  // (slitLogW_: GAP agent29; slitSpan_: SH agent30)
    const float slit = 1.0f - wide * wide * (3.0f - 2.0f * wide);
    if (kNoiseAreaExp != 0 && (areaCount_++ & 15) == 0) {
      if (kNoiseAreaTarget > 0 && noiseAreaRef_ >= 0) {
        const float ref = std::max(noiseAreaRef_, kNoiseAreaTarget);
        areaGain_ = std::min(1.0f, std::pow(ref / std::max(a, 0.005f), kNoiseAreaExp));
      } else {
        areaGain_ = std::pow(kNoiseAreaRef / std::max(a, 0.005f), kNoiseAreaExp);
      }
    }
    // The jet follows the flow with a lag (LINE_NOISE_RE_MS, 0: none): its Reynolds number low-passed, so a flow surge
    // (the narrowing forming faster than the air column can follow, PUHUJA_ENERGY_TRACE 2026-11-11) does not switch the
    // hiss (~ Re^2) on 30 dB within a few ms.
    static const float kNoiseReMs = param("LINE_NOISE_RE_MS", 10.0f);
    if (kNoiseReMs > 0) {
      // ... except while the constriction opens (a release: p t k bursts must start at once; LINE_NOISE_RE_REL 1,
      // 0: the lag everywhere, which cost ~100 words1000 at 10 ms)
      static const float kReK0 = 1.0f - std::exp(-dt_ * 1000.0f / kNoiseReMs);
      const float kReK = reKNow_ > 0 ? reKNow_ : kReK0;  // (GAP agent29: setReMs)
      static const bool kReRel = param("LINE_NOISE_RE_REL", 1.0f) > 0;
      const bool opening = kReRel && noiseKc_ == reK_ && a > reA_ * 1.0005f;
      // The lag on Re itself (LINE_NOISE_RE_ENV_MS 0) follows its running minimum: down at once, up over 10 ms, and
      // in a voiced fricative the glottal pulses (4-10 ms) modulate the flow, so it settles on each cycle's trough and
      // the hiss (~ Re^2) is driven by that trough (aero review B1: the ð of "those" lagged Re 444 vs raw 1196 +- 731;
      // 3-10 kHz vs no lag: ʒ -19..-25 dB, ð -5..-20, z -2..-5). LINE_NOISE_RE_ENV_MS > 0: the lag acts on a mean of
      // Re over about a pitch period, and the instantaneous Re is scaled by lagged mean / mean (<= 1): the onset is still
      // smoothed, the voicing's modulation of the noise (real in voiced fricatives) kept. At 4 ms: ʒ +5..+19 dB, ZH lab
      // errors 78 -> 58, V 151 -> 134, Z 62 -> 59, DH 57 -> 58; pclick on the snap list unchanged (2026-11-11).
      static const float kReEnvMs = param("LINE_NOISE_RE_ENV_MS", 3.998969f);
      // (LINE_V_RE_ENV_MS, V agent round F: the envelope through a v alone while LINE_NOISE_RE_ENV_MS is 0; with the
      // global one the female Kennedy passage gained a 17-19 dB click at a z, a t and an s)
      static const float kVReEnvMs = param("LINE_V_RE_ENV_MS", 4.0f);
      if (kReEnvMs > 0 || (fricReSmooth_ > 0 && kVReEnvMs > 0)) {
        static const float kEnvK = 1.0f - std::exp(-dt_ * 1000.0f / (kReEnvMs > 0 ? kReEnvMs : kVReEnvMs));
        reMean_ += (re - reMean_) * kEnvK;
        reLp_ = opening || (reMean_ < reLp_ && !fricReBoth_) ? reMean_ : reLp_ + (reMean_ - reLp_) * kReK;
        if (reMean_ > 1.0f) re *= std::min(1.0f, reLp_ / reMean_);
        else re = reLp_;  // (no flow to speak of: as without the envelope)
        if (fricReSmooth_ > 0) re += fricReSmooth_ * (std::min(reMean_, reLp_) - re);  // (a v: toward the lagged mean, LINE_V_RE_SMOOTH)
      } else {
        reLp_ = opening || (re < reLp_ && !fricReBoth_) ? re : reLp_ + (re - reLp_) * kReK;  // (and down at once: only the rise lags; both ways for a z, LINE_Z_RE_BOTH)
        re = reLp_;
        reMean_ = re;  // (the v's envelope, LINE_V_RE_ENV_MS, starts from here)
      }
      reA_ = a;
      reK_ = noiseKc_;
    }
    const float rcF = critMulF_ != 1.0f ? kReynoldsCrit * critMulF_ : kReynoldsCrit;  // (GAP agent29: setFricCritMul)
    const float rcF2 = critMulF_ != 1.0f ? rcF * rcF : reC2;
    fricActive_ = re > rcF && slit > 0;
    // LINE_NOISE_SOFT (0: off; 2026-11-22). The hiss as a hard gate was the root of the "tsirps" and "klumbgns" (Jyrki):
    // (Re^2 - Rc^2) on the instantaneous flow is zero under the threshold and steep just over it, so a groove forming
    // switched the hiss on by 30-40 dB within 10 ms (11labs / Google: 5-12 dB), off by 38 dB in 5 ms, and in a voiced
    // fricative every glottal pulse crossed the threshold on its own: full-band ticks 45 dB deep (the ð of "which this").
    // With it: the jet's Re is averaged over about a pitch period (two poles of LINE_NOISE_SOFT_RE_MS), the law has a
    // knee instead of a threshold, r^4 / (1 + r^2) with r = Re / Rc (-> r^2 - 1 well over it, a U^4 tail under it), the
    // drive's envelope has two poles of LINE_NOISE_SOFT_ATT_MS up (LINE_NOISE_SOFT_REL_ATT_MS while the constriction
    // opens: a release) and LINE_NOISE_SOFT_DEC_MS down, and the pulses modulate it by (Re / mean)^LINE_NOISE_SOFT_MOD
    // within 1/3..3 only.
    static const float kSoft = param("LINE_NOISE_SOFT", 1.0f);
    float softDrive = 0.0f;
    if (kSoft > 0) {
      static const float kSoftReMs = param("LINE_NOISE_SOFT_RE_MS", 2.0f), kSoftAttMs = param("LINE_NOISE_SOFT_ATT_MS", 1.0f),
                         kSoftRelAttMs = param("LINE_NOISE_SOFT_REL_ATT_MS", 3.0f), kSoftDecMs = param("LINE_NOISE_SOFT_DEC_MS", 10.0f),
                         kSoftMod = param("LINE_NOISE_SOFT_MOD", 0.5f);
      static const float kRe = 1.0f - std::exp(-dt_ * 1000.0f / kSoftReMs), kAtt = 1.0f - std::exp(-dt_ * 1000.0f / kSoftAttMs),
                         kRelAtt = 1.0f - std::exp(-dt_ * 1000.0f / kSoftRelAttMs), kDec = 1.0f - std::exp(-dt_ * 1000.0f / kSoftDecMs);
      softRe_[0] += kRe * (reRaw - softRe_[0]);
      softRe_[1] += kRe * (softRe_[0] - softRe_[1]);
      // LINE_NOISE_SOFT_TROUGH (0..1): the share of the law's Re taken from the pulses' troughs (a running minimum: down at
      // once, up over 15 ms) rather than their mean. A voiceless jet is steady (trough = mean, nothing changes); in a
      // voiced fricative only the steady part of the jet keeps the turbulence up: on the mean alone the ð of "which this"
      // hissed as loud as the ʧ before it.
      static const float kSoftTrough = param("LINE_NOISE_SOFT_TROUGH", 0.7f);
      static const float kTroughUp = 1.0f - std::exp(-dt_ * 1000.0f / param("LINE_NOISE_SOFT_TROUGH_MS", 8.0f));
      softMin_ = reRaw < softMin_ ? reRaw : softMin_ + kTroughUp * (reRaw - softMin_);
      float reEff = softRe_[1] + kSoftTrough * (std::min(softMin_, softRe_[1]) - softRe_[1]);
      // LINE_NOISE_SOFT_WIN_MS (> 0): the mean and the trough over a true window of this length (1 ms blocks) instead of
      // the poles: a steady jet counts in full once it has lasted the window, where the poles took 30-40 ms and left a
      // silent gap between a vowel and its sh (11labs: the hiss is at its level while the voice still fades).
      static const float kSoftWinMs = param("LINE_NOISE_SOFT_WIN_MS", 0.0f);
      if (kSoftWinMs > 0) {
        static const int kBlk = std::max(1, (int)std::lround(0.001f / dt_)), kNb = std::clamp((int)std::lround(kSoftWinMs), 1, 31);
        if (winN_ == 0) { winMin_[winI_] = reRaw; winSum_[winI_] = 0; }
        winMin_[winI_] = std::min(winMin_[winI_], reRaw); winSum_[winI_] += reRaw;
        float mn = winMin_[winI_], sum = winSum_[winI_];
        if (++winN_ >= kBlk) {
          winN_ = 0; winI_ = (winI_ + 1) & 31;
          float m2 = 1e30f, s2 = 0;
          for (int i = 1; i < kNb; ++i) { m2 = std::min(m2, winMin_[(winI_ - i) & 31]); s2 += winSum_[(winI_ - i) & 31]; }
          winMinOld_ = m2; winSumOld_ = s2;
        }
        mn = std::min(mn, winMinOld_); 
        const float mean = (sum + winSumOld_) / (float)((kNb - 1) * kBlk + std::max(winN_, 1));
        reEff = mean + kSoftTrough * (std::min(mn, mean) - mean);
      }
      const bool opens = noiseKc_ == softK_ && a > softA_ * 1.0005f;
      softA_ = a; softK_ = noiseKc_;
      // LINE_NOISE_SOFT_REL_RAW (0: off): while the constriction opens (a release) the law takes the jet's Re as it is,
      // not its mean: a p t k burst starts with its flow (the full-dictionary gate lost ~840 words to the soft law with the
      // mean: initial t 4.6 %, "Cap" -> tap, "Tucker" -> pucker, "Kate" -> hate).
      static const bool kSoftRelRaw = param("LINE_NOISE_SOFT_REL_RAW", 1.0f) > 0;
      if (kSoftRelRaw && opens) reEff = std::max(reEff, reRaw);
      const float r = reEff / rcF, r2 = r * r;
      static const float kSoftGain = param("LINE_NOISE_SOFT_GAIN", 1.0f);  // (the hiss's level under the new law, x)
      // LINE_NOISE_SOFT_KNEE (1: r^4 / (1 + r^2); n: the same in r^n, a sharper knee: less hiss under the threshold, where
      // an initial r or l got a noise onset: "Rad" -> grad, "Rap" -> crap, "Reads" -> breeds)
      static const float kSoftKnee = param("LINE_NOISE_SOFT_KNEE", 2.0f);
      float target = kSoftGain * slit * fricGain_ * areaGain_ * r2 * r2 / (1.0f + r2);
      if (kSoftKnee != 1.0f) { const float q = std::pow(r2, kSoftKnee); target = kSoftGain * slit * fricGain_ * areaGain_ * r2 * q / (1.0f + q); }
      const float k = target > softEnv_[1] ? (opens ? kRelAtt : kAtt) : kDec;
      softEnv_[0] += k * (target - softEnv_[0]);
      softEnv_[1] += k * (softEnv_[0] - softEnv_[1]);
      float mod = 1.0f;
      if (kSoftMod > 0 && softRe_[1] > 1.0f) mod = std::pow(std::clamp(reRaw / softRe_[1], 0.3333f, 3.0f), kSoftMod);
      softDrive = softEnv_[1] * mod;
      // LINE_NOISE_SOFT_HOLD (1; 0: off): the envelope's tail is only as strong as the jet that is left. While the envelope
      // is fed the constriction's area and the jet's mean Re are noted; while it decays the drive is scaled by
      // (area / noted) (a pressure source put in as a flow: ~ the jet's area; without it the tail went into a tract sealing
      // for the n of "nation" at the full flow drive, +12 dB over the sh itself, 2.7 kHz, 20 ms: Jyrki, "tion ... very
      // strange") and by (Re / noted)^2 (no flow, no turbulence), each at most 1.
      static const bool kSoftHold = param("LINE_NOISE_SOFT_HOLD", 1.0f) > 0;
      static const bool kSoftHoldRaw = param("LINE_NOISE_SOFT_HOLD_RAW", 1.0f) > 0;
      if (kSoftHold) {
        if (target >= 0.7f * softEnv_[1]) { softHoldA_ = a; softHoldRe_ = softRe_[0]; }
        else if (softHoldA_ > 0) {
          const float fa = std::min(1.0f, a / softHoldA_), fr = softHoldRe_ > 1.0f ? std::min(1.0f, (kSoftHoldRaw ? reRaw : softRe_[0]) / softHoldRe_) : 1.0f;
          softDrive *= fa * fr * fr;
        }
      }
      fricActive_ = softDrive > 1e-5f;
    }
    if (riseMul_ > 0 && !fricActive_) riseEnv_ = 0.0f;
    // (SH agent30: setFallLimit) the drive's level tracked; while the limit is on it falls at most fallMul_ per substep
    bool fallOnly = false;
    float fallScale = 1.0f;
    if (fallTrack_) {
      const float target = fricActive_ ? slit * fricGain_ * areaGain_ * (re * re - rcF2) * invReC2 : 0.0f;
      if (fallMul_ > 0) {
        const float lim = fallEnv_ * fallMul_;
        fallEnv_ = target < lim && lim > fallStop_ ? lim : target;
        if (!fricActive_ && fallEnv_ > 0) fricActive_ = fallOnly = true;
        else if (target > 0) fallScale = fallEnv_ / target;
      } else fallEnv_ = target;
    }
    if (fricActive_) {
      float drive = kSoft > 0 ? softDrive * noiseF : fallOnly ? fallEnv_ * noiseF : slit * fricGain_ * areaGain_ * (re * re - rcF2) * invReC2 * noiseF;
      if (fallScale != 1.0f && !(kSoft > 0)) drive *= fallScale;
      if (riseMul_ > 0 && !(kSoft > 0)) {  // (GAP agent29: setRiseLimit)
        const float target = slit * fricGain_ * areaGain_ * (re * re - rcF2) * invReC2;
        riseEnv_ = riseEnv_ < 0 || target <= riseEnv_ ? (riseEnv_ < 0 ? target * riseFloor_ : target) : std::min(target, std::max(riseEnv_ * riseMul_, target * riseFloor_));
        if (target > 0) drive *= riseEnv_ / target;
      }
      // When the source moves (a new constriction takes over, "forebears fought"), the old place keeps a share that
      // fades over LINE_NOISE_MOVE_MS (0: the source jumps), so the hiss's colour does not switch in one substep.
      static const float kNoiseMoveMs = param("LINE_NOISE_MOVE_MS", 0.0f);
      if (kNoiseMoveMs > 0) {
        if (lastNoiseK_ >= 0 && noiseK_ != lastNoiseK_) {
          moveFromK_ = lastNoiseK_;
          moveW_ = 1.0f;
        }
        lastNoiseK_ = noiseK_;
        moveW_ *= std::exp(-dt_ * 1000.0f / kNoiseMoveMs);
      }
      const float wOld = kNoiseMoveMs > 0 && moveFromK_ >= 0 ? moveW_ : 0.0f;
      if (kNoiseAsPressure > 0) {
        oral_.addPressureDrive(noiseK_, drive * kNoiseAsPressure * (1.0f - wOld));
        if (wOld > 0.001f) oral_.addPressureDrive(moveFromK_, drive * kNoiseAsPressure * wOld);
      } else {
        oral_.addFlowDrive(noiseK_, drive * (1.0f - wOld));
        if (wOld > 0.001f) oral_.addFlowDrive(moveFromK_, drive * wOld);
      }
    }
    static FILE* ntrace = std::getenv("PUHUJA_NOISE_TRACE") ? std::fopen(std::getenv("PUHUJA_NOISE_TRACE"), "w") : nullptr;
    static long ncount = 0;
    if (ntrace && (ncount++ % 192) == 0) {
      std::fprintf(ntrace, "%ld\t%d\t%d\t%.4f\t%.4f\t%.1f\t%.0f\t%.3f\t%.1f", ncount / 192, noiseKc_, noiseK_, a, oral_.areaAt(noiseK_), u, re,
                   re > kReynoldsCrit ? (re * re - reC2) / reC2 : 0.0f, oral_.areaAt(oral_.sections() - 1));
      for (int k = oral_.sections() - 12; k < oral_.sections(); ++k) std::fprintf(ntrace, "\t%.3f", oral_.areaAt(k));
      const float fcF = stAlphaF_ > 0 && stAlphaF_ < 1 ? -std::log(1.0f - stAlphaF_) / (2.0f * kPiL * dt_) : 0.0f;
      const float fcG = stAlphaG_ > 0 && stAlphaG_ < 1 ? -std::log(1.0f - stAlphaG_) / (2.0f * kPiL * dt_) : 0.0f;
      std::fprintf(ntrace, "\t%.1f\t%.1f\t%.1f\t%.1f\t%.3f\t%.2f\t%.2f\t%.0f\t%.0f\t%.4f\t%.1f\n", oral_.lipFlow(), lungPressure_, expandFlow_,
                   oral_.pressureAt(std::max(velumK_ - 4, 1)), slit, fricGain_, aspGain_, fcF, fcG, glotArea_, u0);
    }
  }
  oral_.traceExtra_ = nasal_.lipFlow();  // the nostril flow, for the trace
  const float expandLp = expandLpHz_ >= 0 ? expandLpHz_ : kExpandLp;  // (setExpandLp: a closure's own, BDG-Z agent20)
  if (expandLp > 0 && !(expandFlow_ > 0 && expandSlow_)) expandPc_ = oral_.pressureAt(std::max(velumK_ - 4, 1));  // (tracks while idle)
  if (oralSinus_) oral_.sinusFlows();  // (round AB VQ3: the piriform fossae, LINE_PIRIFORM_HZ)
  const float p1 = oral_.substep(u0);
  if (oralSinus_) oral_.sinusPressures();
  if (nasKf_ >= 0) {
    oral_.sink(nasK0_, (1.0f - nasKw_) * uNasal_);
    oral_.sink(nasK0_ + 1, nasKw_ * uNasal_);
    if (nasKfT_ < 0 && nasKf_ == static_cast<float>(velumK_)) nasKf_ = -1.0f;  // back home: the plain path again
  } else {
    oral_.sink(velumK_, uNasal_);
  }
  // LINE_EXPAND_PHYS (2026-11-23; on since the owner's "you can turn it on by default, start retune"; 0: the servo below): the enlargement behind a voiced closure as a yielding
  // element at the sink's section, a compliance C behind a resistance R and an inertance M:
  //   M dU/dt = p - R U - V / C,  dV/dt = U,  V >= 0,
  // driven by the section's own pressure. Root cause of the "klumbgn" (Jyrki: "as if the airflow was fully constricted
  // for some time and then released"): the servo is a pump (LINE_JH_EXPAND_FLOW 770 cm^3/s, b d g 200-560) steered by
  // the cavity pressure through a 52 Hz low-pass and a 3 ms ease, so it kept drawing 300-480 cm^3/s for 5 ms after the
  // sealed cavity was empty: "Judge." fell from +8400 to -21000 dyn/cm^2 (twice the lung pressure, negative), the folds
  // were sucked shut (ag 0.002) and the mouth stayed at -2600 for 40 ms until the release filled it: the thump. 25 of
  // 25 words with b d g ʤ went below -1000 dyn/cm^2 for 7-60 ms (Globe -8500, Web -9900, Ago -12800). The element
  // takes air only while the pressure exceeds what it already holds, so it cannot make a vacuum; M keeps it out of
  // the voice's AC (the servo's low-pass did that), the pressure rises as it fills and the voice bar decays with it.
  // Outside a closure it only gives its volume back, through R x LINE_EXPAND_RET.
  if (squeeze_ > 0 || squeezeU_ > 0) {
    static const float kSqCc = param("LINE_SQUEEZE_CC", 1.0f), kSqMs = param("LINE_SQUEEZE_ON_MS", 10.0f), kSqAmin = param("LINE_EXPAND_AMIN", 0.5f);
    const float tgt = squeezeV_ < kSqCc ? squeeze_ : 0.0f;
    squeezeU_ += (tgt - squeezeU_) * std::min(1.0f, dt_ / (0.001f * std::max(kSqMs, 0.01f)));
    if (tgt <= 0 && squeezeU_ < 1e-3f) squeezeU_ = 0.0f;
    squeezeV_ += dt_ * squeezeU_;
    int k = std::max(velumK_ - 4, 1);
    while (k > 1 && oral_.areaAt(k) < kSqAmin) --k;
    oral_.sink(k, -squeezeU_);
  }
  static const float kExpPhys = param("LINE_EXPAND_PHYS", 1.0f);
  bool physOn = false;
  if (kExpPhys > 0) {
    static const float kExpC = param("LINE_EXPAND_C", 3.0e-3f), kExpR = param("LINE_EXPAND_R", 15.0f), kExpM = param("LINE_EXPAND_M", 0.002f),
                       kExpRet = param("LINE_EXPAND_RET", 5.0f), kExpAmin = param("LINE_EXPAND_AMIN", 0.5f), kExpFRef = param("LINE_EXPAND_FREF", 0.0f);
    // (a closure only: through a z or ð the sink stays the servo's relief valve above LINE_EXPAND_HOLD of the lungs;
    // the element there drew 3-5 cm^3 out of the fricative, "Fees." "These." "Cruise.": the dictionary's final z)
    const bool on = expandFlow_ > 0 && expandClosure_;
    physOn = on;
    if (on) expandShare_ = 0.0f;
    if (on || expV_ > 0) {
      // at the old sink section, or behind the closure (a g closes at the servo's own section, where the area is ~0)
      int k = std::max(velumK_ - 4, 1);
      while (k > 1 && oral_.areaAt(k) < kExpAmin) --k;  // (to the nearest open section behind it)
      const float c = kExpC * (kExpFRef > 0 && on ? expandFlow_ / kExpFRef : 1.0f);
      // (LINE_SQUEEZE_TENSE) the wall tensed while the cavity is squeezed: its compliance x this
      static const float kSqTense = param("LINE_SQUEEZE_TENSE", 1.0f);
      if (on) expC_ = squeeze_ > 0 ? c * kSqTense : c;
      // the element comes in over LINE_EXPAND_PHYS_ON_MS (its conductance eased from zero): switched on at once into a
      // cavity already at 5000 dyn/cm^2 it took 360 cm^3/s in a step, a click at the seal ("Judge.": 0.25 re the vowel's 0.15)
      static const float kExpOnMs = param("LINE_EXPAND_PHYS_ON_MS", 8.0f);
      if (on) expOn_ += (1.0f - expOn_) * std::min(1.0f, dt_ / (0.001f * std::max(kExpOnMs, 0.01f))); else expOn_ = 0.0f;
      const float r = on ? kExpR / std::max(expOn_, 1e-3f) : kExpR * kExpRet;
      // implicit in U, in the volume and in the section's own pressure drop (cP U)
      // The return follows the section's slow pressure only (LINE_EXPAND_RET_LP_MS): on the instantaneous one the
      // emptying element was a one-way leak of 1 / (R RET) at mid-tract for ~200 ms after every b d g, and the vowels
      // after it lost 2.3-2.7 dB ("Bad. Hat.": both vowels; "Hat." alone was bit-identical). Dictionary gate with
      // that leak: 47362 vs 48285, a quarter of the loss in words without b d g j.
      static const float kExpRetLpMs = param("LINE_EXPAND_RET_LP_MS", 20.0f);
      const float pk = oral_.pressureAt(k);
      expPs_ += (pk - expPs_) * std::min(1.0f, dt_ / (0.001f * std::max(kExpRetLpMs, 0.01f)));
      const float pDrive = on || kExpRetLpMs <= 0 ? pk : expPs_;
      float u = (expU_ + dt_ / kExpM * (pDrive - expV_ / expC_)) / (1.0f + dt_ / kExpM * (r + (on ? oral_.cPAt(k) : 0.0f) + dt_ / expC_));
      // LINE_EXPAND_PHYS_OFF_MS (0: off, the uptake stops at once; 15 since 2026-11-25): at the release the element was still taking the
      // glottal flow (ʤ: ~200 cm^3/s) and its resistance went x LINE_EXPAND_RET in one substep; the cavity pressure
      // went 4200 -> 8400 dyn/cm^2 within a millisecond behind the just opening slit: a click of 0.76 ("huge") ..
      // 1.3 ("an orange") x the vowel's peak. Its conductance now fades over this while it still takes air (it gives
      // air back as before). LINE_EXPAND_PHYS_OFF_FRIC 1: only a closure released into a fricative.
      static const float kExpOffMs = param("LINE_EXPAND_PHYS_OFF_MS", 15.0f);
      static const bool kExpOffFricOnly = param("LINE_EXPAND_PHYS_OFF_FRIC", 1.0f) > 0;
      if (on) expOff_ = 1.0f;
      else if (kExpOffMs > 0 && expOff_ > 0) {
        expOff_ = !kExpOffFricOnly || expandOffAllowed_ || expandOffHandover_ ? expOff_ - dt_ / (0.001f * kExpOffMs) : 0.0f;
        if (expOff_ > 0) {
          const float uf = (expU_ + dt_ / kExpM * (pk - expV_ / expC_)) / (1.0f + dt_ / kExpM * (kExpR / expOff_ + oral_.cPAt(k) + dt_ / expC_));
          if (uf > 0) u = uf; else expOff_ = 0.0f;
        }
      }
      if (!on && u > 0 && !(kExpOffMs > 0 && expOff_ > 0)) u = 0;
      expV_ += dt_ * u;
      if (expV_ <= 0) { expV_ = 0; if (u < 0) u = 0; }
      expU_ = u;
      oral_.sink(k, u);
    }
  }
  if (!physOn && expandFlow_ > 0) {
    // The expanding cavity behind a voiced closure (cheeks, larynx lowering):
    // a leak that holds the cavity near kExpandHold of the lung pressure, so
    // less pressure gathers for b than for p (Jyrki 2026-11-03).
    float pc = oral_.pressureAt(std::max(velumK_ - 4, 1));
    if (expandLp > 0 && expandSlow_) {
      // LINE_EXPAND_LP (Hz): the share follows the cavity's slow pressure
      // only. Driven by the instantaneous pressure the sink was a resistive
      // shunt of ~0.1 cm^5/(dyn s) at mid-tract (in its linear range, b's
      // closure at share ~0.9): the folds lost their load, p0's AC was 55
      // dyn/cm^2 (d's 510), voicing died within 5 ms and the closure was
      // silent; d's bar kept only the 1.2 kHz closed-tube mode (2026-11-05).
      expandPc_ += (1.0f - std::exp(kM2PiL * expandLp * dt_)) * (pc - expandPc_);
      pc = expandPc_;
    }
    const float hold = kExpandHold * lungPressure_;
    float share = std::clamp((pc - hold) / std::max(0.2f * lungPressure_, 1.0f), 0.0f, 1.0f);
    // LINE_EXPAND_OPEN (cm^3/s, 2026-11-11, review voice.md B2; 0: off): a planned closure's enlargement as an
    // open-loop volume velocity (Westbury 1983: 10-20 cm^3 over a b closure, a slow muscular gesture), not a servo on
    // the cavity pressure: the servo against the sealed tract made the "voice bars" a 45-65 Hz loop (P1 swinging
    // -3000..+9900 dyn/cm^2), not voicing at f0. Only a near-empty cavity (under LINE_EXPAND_OPEN_FLOOR of the lungs,
    // slow pressure) eases it off, so it never sucks the mouth below zero.
    static const float kExpandOpen = param("LINE_EXPAND_OPEN", 0.0f), kExpandOpenFloor = param("LINE_EXPAND_OPEN_FLOOR", 0.1f);
    if (kExpandOpen > 0 && expandSlow_) share = std::clamp(pc / std::max(kExpandOpenFloor * lungPressure_, 1.0f), 0.0f, 1.0f) * (kExpandOpen / std::max(expandFlow_, 1.0f));
    // LINE_EXPAND_ON_MS (2026-11-05): the sink's share eases in and out over
    // this. As a voiced closure sealed, the cavity pressure reached the hold
    // within a few ms and the 200 cm^3/s sink switched on as a step at
    // mid-tract: a 2-8 kHz click 15 dB over the vowel's, the "blib" at the end
    // of "of" in "hand of God" (Jyrki). 0: as before.
    if (kExpandOnMs > 0) {
      expandShare_ += (1.0f - std::exp(-dt_ / (0.001f * kExpandOnMs))) * (share - expandShare_);
      share = expandShare_;
    }
    oral_.sink(std::max(velumK_ - 4, 1), expandFlow_ * share);
    expandLast_ = expandFlow_;
  }
  // LINE_EXPAND_OFF_MS (0: off): when the closure ends, the sink fades over this instead of stopping in one substep.
  // Jyrki 2026-11-11: "\"hardship\" has a snap between d and s": a 0.2 click at 705 ms, 16 ms into the ʃ, that stays with
  // every noise source off and moves away with LINE_EXPAND_FLOW 0.
  static const float kExpandOffMs = param("LINE_EXPAND_OFF_MS", 5.007732f);
  // LINE_EXPAND_OFF_FRIC 1: only for a closure released into a fricative ("hardship"; expandOffAllowed_, set by the
  // synth per closure); the fade on every voiced stop cost g 161 -> 211 errors of 604 in the lab (2026-11-11).
  static const bool kExpandOffFric = param("LINE_EXPAND_OFF_FRIC", 0.0f) > 0;
  if (!(expandFlow_ > 0) && kExpandOffMs > 0 && expandShare_ > 1e-4f && (!kExpandOffFric || expandOffAllowed_)) {
    expandShare_ *= std::exp(-dt_ / (0.001f * kExpandOffMs));
    oral_.sink(std::max(velumK_ - 4, 1), expandLast_ * expandShare_);
  } else if (!(expandFlow_ > 0)) {
    expandShare_ = 0.0f;  // (the eased share starts from zero at the next closure)
  }
  oral_.traceExtra2_ = lungPressure_;
  nasal_.sinusFlows();
  nasal_.substep(uNasal_);
  nasal_.sinusPressures();
  pSub_ = sub_.substepSubglottal(lungPressure_, u0);
  // The microphone: the radiated pressure is the derivative of the total
  // flow out of the mouth and the nostrils, low-passed at the substep rate
  // and averaged over the sample before the derivative is taken at 48 kHz.
  // ... plus the walls' motion: with the lips and the velum shut (a voiced
  // closure) nothing else radiates, and the voice bar of b d g was silent
  // (-50 dB re the word while the folds swung; the readers' -18..-24 dB,
  // 2026-11-04).
  float wall = kWallRadiate > 0 ? kWallRadiate * oral_.wallFlow() : 0.0f;
  // The yielding wall behind a voiced closure (LINE_EXPAND_PHYS) is the cheeks and the neck: what flows into it moves
  // them, and that is the voice bar's sound. LINE_EXPAND_RADIATE: its flow radiated like the walls' (0: silent).
  static const float kExpRadiate = param("LINE_EXPAND_RADIATE", 0.0f);
  if (kExpRadiate != 0) wall += kExpRadiate * expU_;
  // ... and the tissue around the other airways (Jyrki 2026-11-05: "some
  // sound leaks through facial tissue directly, through throat, through
  // cheeks, through chest, forehead"): the trachea's walls through the neck
  // and chest (LINE_TRACHEA_RADIATE, x its wall flow; their motion carries
  // the subglottal resonances) and the nasal walls through the cheeks and
  // nose (LINE_NASAL_WALL_RADIATE). 0: off. All of it through the tissue's
  // low-pass, LINE_WALL_RAD_LP (Hz; 0: none).
  {
    // The nasal walls at 40 (2026-11-06, "in his mortal hands": the murmur sat 4-6 dB under the next vowel, Kokoro's m n
    // at the vowel's level; the nasal line's loss, viscosity, port, cavity and nostril knobs moved it < 1 dB). Only
    // while the velum is open: the port never shuts fully (an_ >= 1e-3), and ungated the leak put a -22 dB copy of
    // every vowel through the nasal walls. Full from LINE_NASAL_RAD_AN of opening (the line's an_, cm^2).
    static const float tracheaRad = param("LINE_TRACHEA_RADIATE", 0.0f), nasalRad = param("LINE_NASAL_WALL_RADIATE", 0.0f);
    static const float nasalRadAn = param("LINE_NASAL_RAD_AN", 0.10156428f), invNasalRadAn = 1.0f / std::max(nasalRadAn, 1e-4f);
    if (tracheaRad != 0) wall += tracheaRad * sub_.wallFlow();
    if (nasalRad != 0) wall += nasalRad * nasRadScale_ * std::min(1.0f, an_ * invNasalRadAn) * nasal_.wallFlow();
  }
  if (wallLpHz_ > 0) wall = wallLp_.process(wall);
  uAcc_ += aa2_.process(aa1_.process(oral_.lipFlow() + nasal_.lipFlow() + wall));
  if (++subCount_ == TransmissionLine::kSubsteps) {
    const float mean = uAcc_ / TransmissionLine::kSubsteps;
    out_ = (mean - prevMean_) * fs_;
    prevMean_ = mean;
    uAcc_ = 0;
    subCount_ = 0;
  }
  return p1;
}

void VocalTractLine::setNasalGeom(float portA, float cavA, float nostA, float endX, float endA) {  // (NASAL agent32)
  if (portA < 0) portA = kNasalPortArea;
  if (cavA < 0) cavA = kNasalCavityArea;
  if (nostA < 0) nostA = kNostrilArea;
  std::vector<float> na(nasal_.sections());
  for (int k = 0; k < nasal_.sections(); ++k) {
    const float x = (k + 0.5f) * nasal_.dx();
    na[k] = x < 2.0f * nasalScale_ ? portA : x < 9.0f * nasalScale_ ? cavA : nostA;
    if (endX >= 0 && x >= endX * nasalScale_) na[k] = endA;
  }
  nasal_.setAreas(na, true);
}

void VocalTractLine::setSinusScale(float vx1, float vx2, float qx) {  // (NASAL2 agent33; the init's sinus layout)
  const float v1 = param("LINE_SINUS1_V", 22.097188f), a1 = param("LINE_SINUS1_A", 0.10478747f), l1 = param("LINE_SINUS1_L", 0.60069922f);
  const float v2 = param("LINE_SINUS2_V", 0.0f), a2 = param("LINE_SINUS2_A", 0.09755848f), l2 = param("LINE_SINUS2_L", 0.48846771f);
  const float q = param("LINE_SINUS_Q", 5.0f) * qx;
  const float split = param("LINE_SINUS1_SPLIT", 1.12f);
  int i = 0;
  if (v1 > 0 && split > 0) {
    nasal_.retuneSinus(i++, vx1 * v1 * split / (1.0f + split), 0.5f * a1, l1, q);
    nasal_.retuneSinus(i++, vx1 * v1 / (1.0f + split), 0.5f * a1, l1, q);
  } else if (v1 > 0) {
    nasal_.retuneSinus(i++, vx1 * v1, a1, l1, q);
  }
  if (v2 > 0) nasal_.retuneSinus(i, vx2 * v2, a2, l2, q);
}

void VocalTractLine::setNasalCoupling(float frac, float ms) {  // (NASAL2 agent33)
  if (frac >= 0 && nasKf_ < 0) nasKf_ = static_cast<float>(velumK_);
  if (nasKf_ < 0) return;
  nasKfT_ = frac;
  const float tgt = frac >= 0 ? frac * oral_.sections() : static_cast<float>(velumK_);
  const float n = std::max(1.0f, 0.001f * ms / dt_);
  nasKStep_ = std::max(std::fabs(tgt - nasKf_) / n, 1e-4f);
}

void VocalTractLine::setAspHp(float hz, int poles) {
  aspHpA_ = hz > 0 ? 1.0f - std::exp(kM2PiL * hz * dt_) : aspHpBaseA_;
  aspHpPolesNow_ = hz > 0 ? poles : 0;
}

}  // namespace puhuja
