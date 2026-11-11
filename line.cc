#include "line.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#include "params.h"
#include "tract.h"

namespace puhuja {

namespace {
constexpr float kPiL = 3.14159265f;
constexpr float kM2PiL = -2.0f * kPiL;
// Wall per unit area (Ishizaka, Flanagan & Matsudaira 1975): mass (g/cm^2),
// resistance (dyn s/cm^3), stiffness (dyn/cm^3).
const float kLungR = param("LINE_LUNG_R", 6.973f) /* cost Whisper 852 composite 23.90/26.04 92b-cheap9 */, kLungRInv = 1.0f / kLungR;  // the respiratory system's DC resistance (dyn s/cm^5)
const float kWallMass = param("LINE_WALL_M", 1.083f) /* cost Whisper 852 composite 23.90/26.04 92b-cheap9 */, kWallMassInv = 1.0f / kWallMass;
const float kNoiseSt = param("LINE_NOISE_ST", 0.07f);         // Strouhal number of the noise corners (0: the fixed LINE_NOISE_HP; 0.1 on 2026-11-04: the s spectrum near Kokoro's, Whisper neutral)
const float kNoiseLpRatio = param("LINE_NOISE_LPR", 10.0f);  // (2.5 -> 10 on 2026-11-04: in a 1.3-octave band the noise's peak hopped 0.30 octave per 10 ms frame, the readers' 0.06-0.08: the "zipping, zonking, pinging" Jyrki heard; at 10: 0.11; Whisper neutral)
const float kNoiseStSmoothGMs = param("LINE_NOISE_ST_SMOOTH_G_MS", 15.0f);  // 2026-11-05: the glottal jet's corner smoothed over this alone (ms; < 0: LINE_NOISE_ST_SMOOTH_MS). Its corner zips 200 <-> 12000 Hz once per pitch period in every voiced sound.
const float kNoiseStSmoothFMs = param("LINE_NOISE_ST_SMOOTH_F_MS", -1.0f);  // ... the constriction's jet's
const float kNoiseFcMax = param("LINE_NOISE_FC_MAX", 8000.0f);  // the corners' ceiling (Hz; a closing constriction sat at the clamp: 8-12 ms pings at 12 kHz in d, t, m)
const float kNoiseUMax = param("LINE_NOISE_U_MAX", 1.0f);  // > 0: the jet speed behind the corners and the constriction's Reynolds number capped at this x the Bernoulli speed sqrt(2 Ps / rho) (the line's flow lags a closing area: 1.3-1.4 x the limit into a p)
const float kNoiseAreaExp = param("LINE_NOISE_AREA_EXP", 1.5f);  // 2026-11-05: the constriction noise drive x (LINE_NOISE_AREA_REF / area)^this. The Re^2 law (U^2 d^2 ~ u^2 / a) grows as a groove opens on the glottis-limited flow (u ~ a): the s of "save" rose 12 dB over its last 13 ms as its groove opened and its corner fell 8.8 -> 2.1 kHz, then the fade cut it (Kokoro: a 20 dB fade over the last 30 ms). Stevens: the source goes as U^3 / A^2.5. 0: as before.
const float kNoiseAreaRef = param("LINE_NOISE_AREA_REF", 0.05f);  // cm^2 (an s groove): no change there
const float kNoiseAreaTarget = param("LINE_NOISE_AREA_TARGET", 0.001f);  // > 0: the reference is instead the constriction's own area (the smaller end of its glide, at least this many cm^2), and the factor never exceeds 1: every steady fricative keeps its level, only the glides out of and into a groove or a closure are damped (the absolute law cut z v ð by 15 dB and lifted f ʃ by 8: Whisper -45)
const float kNoiseStSnap = param("LINE_NOISE_ST_SNAP", 0.0f);  // 2026-11-05: > 0: a smoothed corner follows its target at once while its jet is silent, so a hiss starts at its own corner and the smoothing shapes only the sweeps inside it (smoothed from the vowel floor, an s began 20 ms of dull f: Whisper -70 in round 6; the t release into ə swept 11 -> 4 kHz over 30 ms in the female voice)
const float kExpandOnMs = param("LINE_EXPAND_ON_MS", 3.0f);  // the voiced closure sink's share eases in over this (ms; 0: a step)
const float kNoiseStSmoothMs = param("LINE_NOISE_ST_SMOOTH_MS", 0.0f);  // ... the corners smoothed over this long (ms; 0: none)   // ... the low-pass corner at this times the high-pass one
const float kGlottisLength = param("LINE_GLOTTIS_L", 1.3f);  // the glottis's length (cm): its width = area / this
// The glottal jet's own low-pass ratio and floor (2026-11-04): at a p puff's
// 0.6 cm^2 glottis fc ~0.6 kHz put the low-pass at 1.6 kHz, so the aspiration
// had no 2-5 kHz hiss (its 1.5-3 kHz band 12-14 dB under 0.5-1.5 kHz; the
// readers' is level or brighter). <= 0: LINE_NOISE_LPR; the floor in Hz.
const float kGlottalLpRatio = param("LINE_GLOTTAL_LPR", -1.0f);
const float kGlottalLpMin = param("LINE_GLOTTAL_LPMIN", 0.0f);
const float kSlitWidth = param("LINE_SLIT_W", 1.0f);         // a fricative slit's width (cm): its height = area / this
const float kWallRadiate = param("LINE_WALL_RADIATE", 10.0f);  // the walls' volume velocity radiated with the lips' (0: not; 2026-11-04: 10 with LINE_VOICED_SOFT 2, d words right 24 -> 28 of 108, the vowels 2-4 dB less bright above 1 kHz)
const float kWallDamp = param("LINE_WALL_B", 1.019e+04f) /* cost 0.0308115287161875 wobble */;
const float kWallStiff = param("LINE_WALL_K", 3.0e5f);
const float kClosureMin = param("LINE_CLOSURE_MIN", 0.002f);  // the smallest constriction area the hump reaches (cm^2)
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
const float kHfBandwidth = param("LINE_HF_BW", 198.6f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */;  // Hz added well above the corner
const float kHfCorner = param("LINE_HF_FC", 3000.0f) /* cost 1814 whisper-run1-phon_err */;    // Hz

void TransmissionLine::hfLoss(int nb) {
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
  for (int k = 0; k < kMaxSections + kBlock; ++k) { cU_[k] = 0; gU_[k] = 1; oriInv_[k] = 0; }
  // Interfaces 1..n-1 (0 is the glottis, n the lips: set each substep).
  for (int k = 1; k < n_; ++k) {
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
  for (int k = 0; k < kMaxSections; ++k) {
    if (k < n_) {
      cP_[k] = dtComp / a_[k];
      sWall_[k] = 2.0f * std::sqrt(kPiL * a_[k]) * dx_;
      wA_[k] = wDen;
      wB_[k] = dt_ * kWallMassInv * wDen;
    } else {
      cP_[k] = sWall_[k] = wA_[k] = wB_[k] = 0;
    }
  }
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
const float kVelumScale = param("LINE_VELUM", 6.01f) /* cost 1814 whisper-run1-phon_err */;  // the port area per unit of the velum opening
// The nasal line, velum to nostrils (cm). With the velum at 55 % of a 17.5 cm
// tract (9.6 cm from the glottis) the nasal path is 20.6 cm, 3.1 cm longer
// than the oral one (Jyrki 2026-11-02: "the nasal tract may be longer,
// perhaps 3 cm, than the mouth tract"); anatomical values 12-12.5 cm would
// make it 4-4.6 cm longer.
const float kNasalLength = param("LINE_NASAL_CM", 11.0f);
const float kReynoldsCrit = param("LINE_RE_CRIT", 1400.0f);  // 1800 until 2026-11-06: the voiced z's jet (Re 800-1600) never hissed, "is" heard as "in"; 1400 with LINE_AG_Z 0.16, LINE_Z_AC 1 (zA): Whisper 801 -> 853, Jyrki "sounded good"
const float kFricA0 = param("LINE_FRIC_A0", 0.12f);  // the frication source at full strength below this constriction area (cm^2; the widest slit, ð, is 0.084) ...
const float kFricA1 = param("LINE_FRIC_A1", 0.2156f) /* cost 1814 whisper-run1-phon_err */;  // ... and silent above this (the approximants: 0.25-0.35; at 0.3 the r after a vowel, 0.25 cm^2, still hissed +17 dB)
constexpr float kMuAir = 1.86e-4f;
// Re = 2 rho |u| / (mu L) (slit) or 2 rho |u| / (mu sqrt(pi A)) (round duct): the constant factors once.
const float kReSlitFactor = 2.0f * TransmissionLine::kRho / (kMuAir * kGlottisLength);
constexpr float kReRoundFactor = 2.0f * TransmissionLine::kRho / kMuAir;
}  // namespace

int64_t rot(int64_t a, int shift) {
  return (a >> shift) | (a << (64 - shift));
}

float VocalTractLine::noise() {
  rng_ += rot(rng_, 15);
  rng_ ^= rng_ >> 17;
  rng_ += rot(rng_, 31);
  rng_ ^= rot(rng_, 23);
  rng_ += rot(rng_, 11);
  return rng_ * (1.0f / 9223372036854775808.0f) - 1.0f;
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

const float kNoiseAsPressure = param("LINE_NOISE_P", 3.0f);  // as a flow drive the hiss grew only 0.45 dB per dB of gain (s stuck 18 dB under the vowels); as a section pressure x2 it sits at -9, Kokoro -5..-7 (2026-11-02)  // > 0: inject the turbulence as a section pressure (scaled by this) instead of an interface flow
const float kExpandHold = param("LINE_EXPAND_HOLD", 0.6f);  // a voiced closure's cavity pressure is held near this share of the lungs'
const float kExpandLp = param("LINE_EXPAND_LP", 50.0f);  // > 0: the voiced closure's expansion follows the cavity pressure low-passed at this (Hz), not as an acoustic shunt; 0: the instantaneous pressure (b's voice bar silent)
const float kHumpReleaseMs = param("LINE_HUMP_RELEASE_MS", 40.0f);  // the constriction's area releases with this time constant beyond kHumpFastTo
const float kHumpFastTo = param("LINE_HUMP_FAST_TO", 2.0f);  // (2.0: off; 0.4 held F2 low through the vowel and affricated the t puff)        // ... after opening fast (6 ms) to this area (cm^2): the acoustic release
const float kNoiseDx = param("LINE_NOISE_DX", 0.2f);  // the turbulence source this far (cm) past the hump's downstream edge
const float kNasalCavityArea = param("LINE_NASAL_CAVITY_A", 3.5f);  // the main nasal cavity, both sides (cm^2)
const float kNasalPortArea = param("LINE_NASAL_PORT_A", 1.5f);      // the nasopharynx behind it (cm^2)
const float kNostrilArea = param("LINE_NOSTRIL_A", 1.594f) /* cost 1814 whisper-run1-phon_err */;  // both nostrils together (cm^2; 1.2 put the n level with the vowel, the readers 9-14 dB under)

void VocalTractLine::init(float fs) {
  fs_ = fs;
  dt_ = 1.0f / (TransmissionLine::kSubsteps * fs);
  oral_.init(17.5f, fs, kOralSections);
  nasal_.init(kNasalLength, fs, kNasalSections);
  static const float kNasalLoss = param("LINE_NASAL_LOSS", 1.0f);
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
  static const float kNasalHfBw = param("LINE_NASAL_HF_BW", 300.0f), kNasalHfFc = param("LINE_NASAL_HF_FC", 1000.0f);
  nasal_.setHfLoss(kNasalHfBw, kNasalHfFc);
  static const float kNasalVisc = param("LINE_NASAL_VISC", 4.221f);
  nasal_.setViscLoss(kNasalVisc);  // (30: the m of "his mortal", starting from the voiceless z, fell to -26 dB)  // the turbinates: far more surface per volume than a tube
  // The nasal passage: the port and nasopharynx, the cavity, the nostrils.
  std::vector<float> na(nasal_.sections());
  for (int k = 0; k < nasal_.sections(); ++k) {
    const float x = (k + 0.5f) * nasal_.dx();
    na[k] = x < 2.0f ? kNasalPortArea : x < 9.0f ? kNasalCavityArea : kNostrilArea;
  }
  nasal_.setAreas(na, false);
  // The paranasal sinuses as Helmholtz side branches (Dang & Honda 1994: the
  // maxillary pair gives the murmur its zero near 500 Hz, the frontal and
  // sphenoid ones near 1-1.5 kHz; without them the line's n carried 1-2 kHz
  // only 17 dB under its low band, Kokoro 27-45, and "known" was "whole").
  {
    static const float v1 = param("LINE_SINUS1_V", 24.0f), a1 = param("LINE_SINUS1_A", 0.1f), l1 = param("LINE_SINUS1_L", 0.6f);
    static const float v2 = param("LINE_SINUS2_V", 4.0f), a2 = param("LINE_SINUS2_A", 0.1f), l2 = param("LINE_SINUS2_L", 0.5f);
    static const float q = param("LINE_SINUS_Q", 5.0f);
    // The maxillary pair as two cavities, one LINE_SINUS1_SPLIT times the other's volume (0: one cavity for both).
    // Jyrki 2026-11-10: "make one sinus 12 % larger in volume than the other". Each half gets half the neck, so the
    // pair's mean resonance stays near the single cavity's and the two zeros part by sqrt(split).
    static const float split = param("LINE_SINUS1_SPLIT", 1.12f);
    nasal_.clearSinuses();
    if (v1 > 0 && split > 0) {
      nasal_.addSinus(static_cast<int>(5.0f / nasal_.dx()), v1 * split / (1.0f + split), 0.5f * a1, l1, q);
      nasal_.addSinus(static_cast<int>(5.0f / nasal_.dx()), v1 / (1.0f + split), 0.5f * a1, l1, q);
    } else if (v1 > 0) {
      nasal_.addSinus(static_cast<int>(5.0f / nasal_.dx()), v1, a1, l1, q);
    }
    if (v2 > 0) nasal_.addSinus(static_cast<int>(7.0f / nasal_.dx()), v2, a2, l2, q);
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
  out_ = uNasal_ = an_ = 0;
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
    for (int k = 0; k < oral_.sections(); ++k) {
      const float d = ((k + 0.5f) * oral_.dx() - xc) / humpHw_;
      if (std::fabs(d) >= 1.0f) continue;
      const float g = std::cos(0.5f * kPiL * d);
      const float lnA = std::log(a[k]);
      const float lnC = std::log(acEff);
      if (lnC < lnA) a[k] = std::exp(lnA + (lnC - lnA) * g * g * depth);
    }
    // The turbulence source sits where the jet leaves the constriction and
    // meets the wide section again (~1.2 cm past the centre): a dipole at a
    // tiny-area interface moves almost no flow (cU ~ A), downstream it does.
    // The Reynolds number is the constriction's own (flow through its centre).
    // The lips rounded over the last 1.5 cm (ʃ: its front cavity needs the
    // rounding to sit near 2.5-3.5 kHz; open-lipped the hiss peaked at 10 kHz).
    if (lipCap_ > 0) {
      const float round = depth;
      for (int k = 0; k < oral_.sections(); ++k) {
        if ((k + 0.5f) * oral_.dx() < shapeLength - 1.5f) continue;
        const float lnA = std::log(a[k]), lnC = std::log(lipCap_);
        if (lnC < lnA) a[k] = std::exp(lnA + (lnC - lnA) * round);
      }
    }
    // A soft closure: the hump's sections lose closureSoft_ times more.
    {
      const int k0 = std::max(1, static_cast<int>((xc - humpHw_) / oral_.dx()));
      const int k1 = std::min(oral_.sections(), static_cast<int>((xc + humpHw_) / oral_.dx()) + 1);
      oral_.setSectionLoss(k0, k1, depth > 0.5f ? closureSoft_ : 1.0f);
    }
    noiseK_ = std::min(oral_.sections() - 2, static_cast<int>((xc + kNoiseDx + humpHw_) / oral_.dx()));  // (-1: at the last interface, the lip load took the drive: "votes" lost its s)
    noiseKc_ = std::min(oral_.sections() - 1, static_cast<int>(xc / oral_.dx()));
  }
  if (!(x >= 0 && depth > 0)) oral_.setSectionLoss(0, 0, 1.0f);
  oral_.setAreas(a);
  velumK_ = std::min(oral_.sections() - 2, static_cast<int>(0.55f * oral_.sections()));
  an_ = std::max(an * kVelumScale, 1e-3f);  // never shut by a switch: the orifice loss closes it
  velarG_ = dt_ * an_ * (1.0f / TransmissionLine::kRho) * nasal_.invDx();
  velarInv2A2_ = 1.0f / (2.0f * an_ * an_);
  static int calls = 0;
  if (std::getenv("PUHUJA_LINE_TRACE") && (calls++ % 50) == 0) {
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
    const float dp = oral_.pressureAt(velumK_) - nasal_.pressureAt(0);
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
  static const float stPreGain = param("LINE_NOISE_ST_PRELP_GAIN", 1.4f);
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
    // Only a fricative-sized slit makes a jet that hisses: the source fades out
    // as the constriction widens past kFricA0..kFricA1 (log scale). The Re^2
    // law alone barely depends on the area at a given flow (Stevens' measured
    // source goes as U^3 / A^2.5), and with the voiced flow through them the
    // approximants' 0.25-0.35 cm^2 constrictions hissed: r +33 dB, w +16,
    // l +13, n +7 dB of 3-8 kHz noise, the "smoosh" (2026-11-03).
    // log(a / (A0 f)) / log(A1 / A0) as (log a - log(A0 f)) x a constant: one log per substep.
    static const float invLogRatio = 1.0f / std::log(kFricA1 / kFricA0);
    if (fricLogA0_ == 0) fricLogA0_ = std::log(kFricA0 * fricFade_);
    const float wide = std::clamp((std::log(a) - fricLogA0_) * invLogRatio, 0.0f, 1.0f);
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
      static const float kReK = 1.0f - std::exp(-dt_ * 1000.0f / kNoiseReMs);
      static const bool kReRel = param("LINE_NOISE_RE_REL", 1.0f) > 0;
      const bool opening = kReRel && noiseKc_ == reK_ && a > reA_ * 1.0005f;
      reLp_ = opening || (re < reLp_ && !fricReBoth_) ? re : reLp_ + (re - reLp_) * kReK;  // (and down at once: only the rise lags; both ways for a z, LINE_Z_RE_BOTH)
      reA_ = a;
      reK_ = noiseKc_;
      re = reLp_;
    }
    fricActive_ = re > kReynoldsCrit && slit > 0;
    if (fricActive_) {
      const float drive = slit * fricGain_ * areaGain_ * (re * re - reC2) * invReC2 * noiseF;
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
  if (kExpandLp > 0 && !(expandFlow_ > 0 && expandSlow_)) expandPc_ = oral_.pressureAt(std::max(velumK_ - 4, 1));  // (tracks while idle)
  const float p1 = oral_.substep(u0);
  oral_.sink(velumK_, uNasal_);
  if (expandFlow_ > 0) {
    // The expanding cavity behind a voiced closure (cheeks, larynx lowering):
    // a leak that holds the cavity near kExpandHold of the lung pressure, so
    // less pressure gathers for b than for p (Jyrki 2026-11-03).
    float pc = oral_.pressureAt(std::max(velumK_ - 4, 1));
    if (kExpandLp > 0 && expandSlow_) {
      // LINE_EXPAND_LP (Hz): the share follows the cavity's slow pressure
      // only. Driven by the instantaneous pressure the sink was a resistive
      // shunt of ~0.1 cm^5/(dyn s) at mid-tract (in its linear range, b's
      // closure at share ~0.9): the folds lost their load, p0's AC was 55
      // dyn/cm^2 (d's 510), voicing died within 5 ms and the closure was
      // silent; d's bar kept only the 1.2 kHz closed-tube mode (2026-11-05).
      expandPc_ += (1.0f - std::exp(kM2PiL * kExpandLp * dt_)) * (pc - expandPc_);
      pc = expandPc_;
    }
    const float hold = kExpandHold * lungPressure_;
    float share = std::clamp((pc - hold) / std::max(0.2f * lungPressure_, 1.0f), 0.0f, 1.0f);
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
  static const float kExpandOffMs = param("LINE_EXPAND_OFF_MS", 5.0f);
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
    static const float tracheaRad = param("LINE_TRACHEA_RADIATE", 0.0f), nasalRad = param("LINE_NASAL_WALL_RADIATE", 20.0f);
    static const float nasalRadAn = param("LINE_NASAL_RAD_AN", 0.1f), invNasalRadAn = 1.0f / std::max(nasalRadAn, 1e-4f);
    if (tracheaRad != 0) wall += tracheaRad * sub_.wallFlow();
    if (nasalRad != 0) wall += nasalRad * std::min(1.0f, an_ * invNasalRadAn) * nasal_.wallFlow();
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

void VocalTractLine::setAspHp(float hz, int poles) {
  aspHpA_ = hz > 0 ? 1.0f - std::exp(kM2PiL * hz * dt_) : aspHpBaseA_;
  aspHpPolesNow_ = hz > 0 ? poles : 0;
}

}  // namespace puhuja
