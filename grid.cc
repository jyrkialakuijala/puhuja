#include "grid.h"

#include "filters.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace puhuja {
namespace {

constexpr int L = GridFolds::kLayers, S = GridFolds::kSegments;
// Research knobs: GRID_<NAME>=value in the environment overrides a default
// while the model is being fitted (read once). GRID_PARAMS=file reads a
// parameter file of "name value" lines first (the optimizer's vector): the
// knob names, and per-mass multipliers M_f_j_i and K_f_j_i (fold, layer,
// segment) on the mass and the body stiffness, 1 = the structured default.
const std::map<std::string, float>& paramFile() {
  static const std::map<std::string, float> m = [] {
    std::map<std::string, float> out;
    if (const char* f = std::getenv("GRID_PARAMS")) {
      if (FILE* fp = std::fopen(f, "r")) {
        char name[64];
        double v;
        while (std::fscanf(fp, "%63s %lf", name, &v) == 2) out[name] = static_cast<float>(v);
        std::fclose(fp);
      }
    }
    return out;
  }();
  return m;
}
float knob(const char* name, float def) {
  const auto& pf = paramFile();
  const auto it = pf.find(name);
  if (it != pf.end()) return it->second;
  const char* v = std::getenv(name);
  return v ? std::strtof(v, nullptr) : def;
}
// The per-mass multipliers from the parameter file (1 when absent).
float massMul(char kind, int f, int j, int i) {
  static float table[2][2][L][S];
  static bool loaded = false;
  if (!loaded) {
    for (int k = 0; k < 2; ++k)
      for (int ff = 0; ff < 2; ++ff)
        for (int jj = 0; jj < L; ++jj)
          for (int ii = 0; ii < S; ++ii) {
            char name[32];
            std::snprintf(name, sizeof name, "%c_%d_%d_%d", k == 0 ? 'M' : 'K', ff, jj, ii);
            const auto& pf = paramFile();
            const auto it = pf.find(name);
            table[k][ff][jj][ii] = it != pf.end() ? it->second : 1.0f;
          }
    loaded = true;
  }
  return table[kind == 'M' ? 0 : 1][f][j][i];
}
constexpr float kLength = 1.4f;     // fold length (cm)
constexpr float kThickness = 0.3f;  // the medial surface's vertical extent (cm)
constexpr float kSeg = kLength / S, kLayer = kThickness / L;
// Mass per layer (g, whole fold; Ishizaka-Flanagan's 0.15 g; tissue density
// ~1.04 g/cm^3 over the vibrating 1.4 x 0.3 x ~0.3 cm gives ~0.13 g). The
// division follows the anatomy (DESIGN_larynx.md section 5): the inferior
// layers are backed by the conus elasticus and the bulk of the thyroarytenoid
// muscle (~6 mm deep), the superior edge is the thin cover (lamina propria
// 1-2 mm, epithelium 0.1-0.2 mm) over the ligament with little body behind.
constexpr float kLayerMass[L] = {0.050f, 0.045f, 0.033f, 0.022f};
// Along the cord the cross-section tapers toward the anterior commissure
// (quadratically; the depth at the anterior end ~0.77 of the middle) and the
// cover is thickest mid-membranous: mass per segment 1 - kTaper at the ends,
// 1 in the middle (half-sine). The posterior end sits on the vocal process
// of the arytenoid (cartilage) and the ligament is thickest there: stiffer
// by kPosteriorStiff at the last segment.
const float kTaper = knob("GRID_TAPER", 0.25f), kPosteriorStiff = knob("GRID_POSTSTIFF", 0.0f);  // (0.5: the fold's mode rose ~15 %, the pitch controller fell short (92.7 Hz for 82.8) and the voicing weakened; the posterior segments hardly vibrate anyway, their rest gap is widest)
// Tissue is not uniform: a fixed +-kInhom pattern on the masses (a hash of
// the position, the same every run) and the right fold kAsym heavier than
// the left. Jyrki (2026-10-30): "less microfluctuations, perhaps some mass
// inhomogeneity": the irregularity of the voice comes from the structure
// instead of from random modulation.
const float kInhom = knob("GRID_INHOM", 0.08f), kAsym = knob("GRID_ASYM", 0.03f);
// The coordinate is the vertical displacement (Jyrki, 2026-10-30: "up and
// down is the right one to be modeled, and when sufficiently high up there
// is an opening between the two folds"). The medial surface's wedge slope
// turns it into the slit width (see gap()); the air pressure on the sloped
// inferior surface pushes the tissue up (its normal has an upward component,
// and its lateral component does work through the slope), with kKappa the
// effective share of the pressure force that drives z.
const float kSlope = knob("GRID_SLOPE", 1.0f), kKappa = knob("GRID_KAPPA", 1.0f);  // (slope 0.5 and 0.3: weak, intermittent voicing; 1.0 phonates in both voices, 2026-10-30)
// The cords' own sound (Jyrki): the moving surface pushes air, a volume
// velocity of the surface area times the vertical velocity, summed over the
// masses, heard with the flow (also while the glottis is shut); 1 is the
// geometric value for the medial surface's footprint.
const float kSurface = knob("GRID_SURFACE", 1.0f);
// The abduction rest area scaled for the grid (its own amplitude differs
// from the two-mass model's; the glottis was open 95 % of the cycle at 1).
const float kRestScale = knob("GRID_REST", 1.0f);
const float kAssist = knob("GRID_ASSIST", 0.6f);  // see gapEff (0: level -39/-42 dB m/f, 0.3: -36/-40, 1.0: -32/-36 but the male pitch 89 for 83 and fewer voiced frames; 2026-10-30)
// The body springs' natural frequency at tension 1 (the couplings raise the
// fold's lowest mode above it; the driver's pitch controller takes up the
// rest, as it does for the two-mass folds).
const float kNaturalF0 = knob("GRID_F0", 140);
// Body stiffness per layer relative to the mass's share: the upper edge is
// softer than the body-like lower layers (Ishizaka-Flanagan: the upper mass's
// own frequency 90 Hz against the lower's 127), which gives the layers their
// phase lag and the fold its energy transfer from the flow. Linear from
// kBottomStiff at the inferior layer to kTopStiff at the superior.
const float kTopStiff = knob("GRID_TOPSTIFF", 0.7f), kBottomStiff = knob("GRID_BOTSTIFF", 1.2f);  // (sweep 2026-10-30: 0.7 and shear 1.5 gave the right pitch in both voices, 62 % voiced frames like the two-mass, level -2.7 dB male, -8 dB female)
// Couplings relative to the body spring of the same mass: along the cord
// (string tension; ends anchored) and between layers (shear).
const float kLongRel = knob("GRID_LONG", 1.0f), kShearRel = knob("GRID_SHEAR", 1.5f);
// Damping ratios while open, per layer (the upper edge damps most, as
// Ishizaka-Flanagan's upper mass), and the extra while touching.
constexpr float kZeta[L] = {0.1f, 0.1f, 0.2f, 0.6f};
const float kZetaClosed = knob("GRID_ZCLOSED", 1.0f);
constexpr float kCollision = 3.0f;  // collision stiffness / body stiffness
// Friction between the cords in a pressed contact, as the grid sees it: the
// layers of a contact slide against the other cord with the mucosal wave;
// a Coulomb-viscous force mu F_n tanh(dv / v0) between adjacent layers that
// are both in contact resists their relative lateral motion there.
const float kFrictionMu = knob("GRID_MU", 0.2f), kFrictionV0 = knob("GRID_V0", 5.0f);  // (cm/s)
constexpr float kRho = 0.00113f;    // air density (g/cm^3)
constexpr float kInertance = 0.005f;  // as folds.cc
constexpr float kRound = 0.002f;    // rounding of the heard area (cm^2)
// Rest width along the cord relative to the mean: wider toward the
// arytenoid (posterior), where the folds part first.
const float kRestFront = knob("GRID_RESTFRONT", 0.6f), kRestBack = 2.0f - kRestFront;
// The membranous part whose closure counts as the glottal closure (a
// posterior chink may stay open).
constexpr int kMembranous = 6;

}  // namespace

void GridFolds::reset() {
  for (int f = 0; f < 2; ++f)
    for (int j = 0; j < L; ++j)
      for (int i = 0; i < S; ++i) z_[f][j][i] = v_[f][j][i] = 0;
  for (int i = 0; i < S; ++i) u_[i] = uOut_[i] = 0;
  uSum_ = uOutSum_ = area_ = 0;
  openMask_ = 0;
  wasOpen_ = closedNow_ = false;
}

// The slit width at a layer from the two folds' vertical displacements: the
// medial surface is a wedge sloping inward as it rises (slope kSlope,
// lateral per vertical), so tissue lifted by z at a height leaves the surface
// there kSlope * z further from the midline. Lifted enough, the folds part.
float GridFolds::gap(int j, int i) const { return restW_[i] + kSlope * (z_[0][j][i] + z_[1][j][i]); }

// The slit width that counts for the air and the contact: the tissue is
// continuous, so a point next to an open neighbour parts more easily
// (Jyrki: "the opening threshold should be slightly reduced if a neighboring
// pixel is opened"): the raw width plus kAssist times the mean opening of
// its neighbours along the cord and in the layers above and below.
float GridFolds::gapEff(int j, int i) const {
  float g = gap(j, i);
  if (kAssist <= 0) return g;
  float open = 0;
  int n = 0;
  if (i > 0) { open += std::max(gap(j, i - 1), 0.0f); ++n; }
  if (i + 1 < S) { open += std::max(gap(j, i + 1), 0.0f); ++n; }
  if (j > 0) { open += std::max(gap(j - 1, i), 0.0f); ++n; }
  if (j + 1 < L) { open += std::max(gap(j + 1, i), 0.0f); ++n; }
  return g + kAssist * open / n;
}

float GridFolds::step(float h, const VocalFolds::Input& in) {
  const float q = std::max(in.q, 0.2f);
  // Rest width along the cord (the abduction, scaled by kRestScale for this
  // model's own amplitude), plus the posterior chink on the last two
  // segments (the cartilaginous glottis between the vocal processes).
  const float restMean = kRestScale * in.rest / kLength;  // area -> width
  for (int i = 0; i < S; ++i)
    restW_[i] = restMean * (kRestFront + (kRestBack - kRestFront) * (i + 0.5f) / S);
  for (int i = S - 2; i < S; ++i) restW_[i] += in.gap / (2 * kSeg);
  // Stiffness ~ tension^2 (frequencies ~ tension), masses fixed, damping
  // ratios constant.
  const float q2 = q * q;
  // Per mass: the layer's share, the taper along the cord, the fixed
  // inhomogeneity and the left-right asymmetry. The body spring keeps the
  // layer's natural frequency (k ~ m), so the inhomogeneity detunes the
  // couplings and the collisions, not the tuning of each mass.
  float kBody[2][L][S], mass[2][L][S], damp[2][L][S];
  for (int f = 0; f < 2; ++f)
    for (int j = 0; j < L; ++j)
      for (int i = 0; i < S; ++i) {
        const float taper = 1.0f - kTaper + kTaper * std::sin(kPi * (i + 0.5f) / S);
        unsigned hsh = 2654435761u * (1u + f * 1000u + j * 100u + i);
        hsh ^= hsh >> 13;
        hsh *= 0x5bd1e995u;
        const float pattern = static_cast<float>((hsh >> 8) & 0xffff) / 32767.5f - 1.0f;  // -1..1
        mass[f][j][i] = kLayerMass[j] / S * taper * (1.0f + kInhom * pattern) * (f == 1 ? 1.0f + kAsym : 1.0f) * massMul('M', f, j, i);
        const float w0 = 2 * kPi * kNaturalF0;
        const float layerStiff = kBottomStiff + (kTopStiff - kBottomStiff) * j / (L - 1.0f);
        const float post = (i + 0.5f) / S;
        kBody[f][j][i] = kLayerMass[j] / S * taper * w0 * w0 * q2 * layerStiff * (1.0f + kPosteriorStiff * post * post) * massMul('K', f, j, i);
        damp[f][j][i] = 2 * std::sqrt(mass[f][j][i] * kBody[f][j][i]);  // x zeta
      }
  closedNow_ = false;
  // Areas before the step.
  float aMin[S], aOld = 0;
  int jMin[S];
  float total = 0;
  for (int i = 0; i < S; ++i) {
    aMin[i] = 1e9f;
    jMin[i] = 0;
    for (int j = 0; j < L; ++j) {
      const float a = kSeg * gapEff(j, i);
      if (a < aMin[i]) {
        aMin[i] = a;
        jMin[i] = j;
      }
    }
    total += std::max(aMin[i], 0.0f);
    if (i < kMembranous) aOld += std::max(aMin[i], 0.0f);
  }
  area_ = aOld;
  const float drive = in.ps - in.p1;
  // Flow per segment channel: implicit step of L du/dt = drive - rho/2 u^2 / a^2
  // (u >= 0), each channel with S times the glottal inertance (in parallel
  // they add up to it). The heard flow goes through the rounded area.
  const float B = kInertance * S / h;
  openMask_ = 0;
  uSum_ = uOutSum_ = 0;
  for (int i = 0; i < S; ++i) {
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
  (void)total;
  // Forces and the semi-implicit Euler step.
  for (int i = 0; i < S; ++i) {
    const float ui = u_[i];
    // The first closed layer from below, if the segment is shut.
    int jClosed = L;
    if (aMin[i] <= 0)
      for (int j = 0; j < L; ++j)
        if (gapEff(j, i) <= 0) {
          jClosed = j;
          break;
        }
    for (int j = 0; j < L; ++j) {
      const float g = gapEff(j, i);
      // Air pressure on this layer's surface.
      float p;
      if (aMin[i] <= 0) {
        p = j < jClosed ? in.ps : in.p1;
      } else if (j < jMin[i]) {
        const float a = kSeg * g;
        const float vel = ui / std::max(a, 1e-6f);
        p = in.ps - 0.5f * kRho * vel * vel;
      } else {
        p = in.p1;  // at and above the narrowest layer: separated flow
      }
      const float fAir = p * kSeg * kLayer * kKappa;
      for (int f = 0; f < 2; ++f) {
        // Collision when the opposing surfaces overlap: the contact pushes
        // the tissue up (which, through the slope, opens the slit).
        float coll = 0, zeta = kZeta[j];
        if (g < 0) {
          coll = -kCollision * kBody[f][j][i] * g * 0.5f;
          zeta += kZetaClosed;
        }
        const float x = z_[f][j][i], v = v_[f][j][i];
        // String coupling along the cord; the ends are anchored.
        const float xl = i > 0 ? z_[f][j][i - 1] : 0.0f;
        const float xr = i + 1 < S ? z_[f][j][i + 1] : 0.0f;
        const float kb = kBody[f][j][i];
        float force = fAir + coll - kb * x - zeta * damp[f][j][i] * v +
                      kLongRel * kb * (xl - 2 * x + xr);
        // Shear coupling to the layers below and above (free ends).
        if (j > 0) force += kShearRel * kb * (z_[f][j - 1][i] - x);
        if (j + 1 < L) force += kShearRel * kb * (z_[f][j + 1][i] - x);
        // Friction between the cords: in contact, the two surfaces slide
        // vertically against each other; a Coulomb-viscous force mu F_n
        // tanh(dv / v0) resists their relative vertical velocity (Jyrki's
        // idea of the cords' friction with each other).
        if (kFrictionMu > 0 && g < 0) {
          const float dv = v - v_[1 - f][j][i];
          force -= kFrictionMu * coll * std::tanh(dv / kFrictionV0);
        }
        v_[f][j][i] += h * force / mass[f][j][i];
      }
    }
  }
  float aNew = 0;
  surface_ = 0;
  for (int f = 0; f < 2; ++f)
    for (int j = 0; j < L; ++j)
      for (int i = 0; i < S; ++i) {
        z_[f][j][i] += h * v_[f][j][i];
        surface_ += kSeg * kLayer * kSlope * v_[f][j][i];
      }
  for (int i = 0; i < kMembranous; ++i) {
    float a = 1e9f;
    for (int j = 0; j < L; ++j) a = std::min(a, kSeg * gapEff(j, i));
    aNew += std::max(a, 0.0f);
  }
  // Debug: GRID_TRACE=file writes, once per output sample (every 4th step),
  // the open-segment mask, the flow and the membranous area, for the zipper
  // pattern of opening and closing.
  static FILE* trace = [] {
    const char* f = std::getenv("GRID_TRACE");
    return f ? std::fopen(f, "w") : nullptr;
  }();
  static int traceCount = 0;
  if (trace && (++traceCount & 3) == 0) std::fprintf(trace, "%u\t%.1f\t%.5f\n", openMask_, uSum_, aNew);
  const bool open = aNew > 0;
  if (wasOpen_ && !open) {
    closedNow_ = true;
    closeFrac_ = aOld > 0 ? std::clamp(aOld / (aOld - aNew + 1e-12f), 0.0f, 1.0f) : 1.0f;
  }
  wasOpen_ = open;
  return uOutSum_ + kSurface * surface_;
}

}  // namespace puhuja
