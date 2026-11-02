// A vocal tract as a transmission line (Jyrki, 2026-11-01: "a fresh start
// based on the string model of the larynx and model the rest in a more
// disciplined way, too, where air pressure moves back and forth in a
// transfer line, of about 80 points"; 48 kHz output, the line steps 4x;
// "write the transmission line with alignas and float32 ... to make it
// compatible with avx512").
//
// The tract from the glottis to the lips is N short sections of the local
// area A_k (cm^2) and length dx = L / N, chosen so that c dt / dx = kCourant
// (< 1: the leapfrog below stays inside its stability limit at area steps).
// State: the pressure p_k at the middle of each section and the volume
// velocity u_k through each interface (u_0 at the glottis is the folds'
// flow, u_N at the lips the radiated flow). Per substep, CGS units:
//   rho/A'_k  du_k/dt = -(p_k - p_{k-1}) / dx - R_k u_k        (momentum; A'
//                        the smaller of the two neighbouring areas, which keeps
//                        the scheme's energy bounded; R the viscous wall loss)
//   A_k dx / (rho c^2)  dp_k/dt = -(u_{k+1} - u_k) - u_wall,k   (continuity;
//                        the yielding wall takes volume too)
//   m_w dv_k/dt = p_k - b_w v_k - k_w xi_k,  dxi_k/dt = v_k,  u_wall = S_k dx v_k
//                        (the wall per unit area: Ishizaka, Flanagan &
//                        Matsudaira 1975; it puts the floor under F1 near
//                        180-200 Hz and widens the low formants)
//   lips: u_N = p_{N-1} / R_r + u_L,  L_r du_L/dt = p_{N-1}  (Flanagan's piston
//                        radiation load, a resistance in parallel with an
//                        inertance), the sound heard is d u_N / dt.
// The folds see p_0 as the pressure just above them (two-way coupling) and
// give back u_0.
//
// Layout for AVX-512: every per-section array is 64-byte aligned float32
// of kMaxSections (a multiple of 16) entries; the loops run over whole
// blocks of 16 with no data-dependent branches, and sections past N carry
// zero coefficients, so they stay at zero. The geometry (areas, wall
// circumference, viscous loss, the radiation load) is turned into per-section
// coefficients once per output sample; a substep is multiply-adds only.
// float32 is kept over float16/int16: the static lung pressure (~8000
// dyn/cm^2) and the acoustic ripple that matters (~1) are 13 bits apart.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace puhuja {

// A second-order low-pass (RBJ), self-contained so the line links alone.
struct LineLowPass {
  float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, z1 = 0, z2 = 0;
  void set(float freq, float fs, float q) {
    const float w0 = 2.0f * 3.14159265f * freq / fs, cw = std::cos(w0), sw = std::sin(w0);
    const float alpha = sw / (2.0f * q), a0 = 1 + alpha;
    b0 = b2 = (1 - cw) / 2 / a0; b1 = (1 - cw) / a0; a1 = -2 * cw / a0; a2 = (1 - alpha) / a0;
  }
  float process(float x) {  // transposed direct form II
    const float y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }
};

class TransmissionLine {
 public:
  static constexpr float kRho = 0.00113f;  // air density (g/cm^3)
  static constexpr float kC = 35000.0f;    // speed of sound (cm/s), body temperature
  static constexpr int kSubsteps = 4;      // line steps per output sample
  static constexpr float kCourant = 0.9f;  // c dt / dx
  static constexpr int kBlock = 16;        // floats per AVX-512 register
  static constexpr int kMaxSections = 128; // 8 blocks: tracts up to ~23 cm at 48 kHz
  // length: glottis to lips (cm). fs: the output rate. sections: 0 picks the
  // most that keep c dt / dx <= kCourant; a fixed count (64: four AVX-512
  // blocks) lowers the Courant number instead, which is stable too.
  void init(float length, float fs, int sections = 0);
  int sections() const { return n_; }
  float dx() const { return dx_; }
  // The tract's length from this output sample on (the section count stays;
  // the section length follows, within the Courant limit).
  void setLength(float length);
  // The area function (cm^2 per section): glided to over kGlide substeps, or
  // set at once (glide = false, for a static start).
  void setAreas(const std::vector<float>& a, bool glide = true);
  // One substep: the glottal flow u0 (cm^3/s) enters; returns the pressure
  // just above the glottis (dyn/cm^2) for the folds.
  float substep(float u0);
  // As a subglottal line: section 0 is the lungs, a matched source at the
  // lung pressure (waves reaching it are absorbed), and the far end is the
  // glottis, whose flow uGlottis leaves the line; no lips. Returns the
  // pressure at the glottal end.
  void setSubglottal(bool on) { subglottal_ = on; }
  float substepSubglottal(float lungPressure, float uGlottis);
  // The sound: the radiated flow's rate of change (cm^3/s^2).
  float radiated() const { return radiated_; }
  float lipFlow() const { return uN_; }
  float pressureAt(int k) const { return p_[k]; }
  float areaAt(int k) const { return a_[k]; }
  float flowAt(int k) const { return u_[k]; }  // interface k
  // Side effects between substeps: a flow leaving section k (into a side
  // branch) and a pressure added at section k (a noise source).
  void sink(int k, float u) { p_[k] -= cP_[k] * u; }
  void addPressure(int k, float dp) { p_[k] += dp; }
  // A pressure difference dp (dyn/cm^2) driving interface k this substep (a
  // dipole source, as turbulence downstream of a constriction is).
  void addFlowDrive(int k, float dp) { u_[k] += cU_[k] * dp; }

 private:
  void coefficients();
  int n_ = 0, blocks_ = 0;
  float fs_ = 48000, dt_ = 0, dx_ = 0;
  alignas(64) float a_[kMaxSections] = {};      // areas (cm^2)
  alignas(64) float p_[kMaxSections + kBlock] = {};  // pressure per section (padded: the momentum loop reads p_[nb])
  alignas(64) float u_[kMaxSections + kBlock] = {};  // flow per interface (u_[0] glottis, u_[n] lips)
  alignas(64) float wallV_[kMaxSections] = {};  // wall velocity and displacement
  alignas(64) float wallX_[kMaxSections] = {};
  // Per-sample coefficients.
  alignas(64) float cU_[kMaxSections + kBlock] = {};  // du = -cU (p_k - p_{k-1}) ...
  alignas(64) float gU_[kMaxSections + kBlock] = {};  // ... and u *= gU (viscous loss)
  alignas(64) float oriInv_[kMaxSections + kBlock] = {};  // dt / (2 A_k dx) per interface: the orifice (Bernoulli) loss, u |u| / A^2
  alignas(64) float cP_[kMaxSections] = {};     // dp = -cP (u_{k+1} - u_k + uWall)
  alignas(64) float sWall_[kMaxSections] = {};  // wall surface per section (cm^2): uWall = sWall v
  alignas(64) float wA_[kMaxSections] = {};     // wall: v = wA v + wB (p - k x), x += dt v
  alignas(64) float wB_[kMaxSections] = {};
  // The areas glide to their targets over kGlide substeps after setAreas (an
  // area that jumps between updates clicks: the leapfrog's state is not
  // rescaled with it) and the coefficients are recomputed from the gliding
  // areas every substep, so the Courant condition holds throughout (gliding
  // cU and cP separately could breach it mid-glide, review 2026-11-01).
  static constexpr int kGlide = 16;
  int glideLeft_ = 0;
  alignas(64) float aTarget_[kMaxSections] = {};
  alignas(64) float aStep_[kMaxSections] = {};  // per-substep area ratio
  float rrInv_ = 0, lrInvDt_ = 0;  // the lips' load
  // How open the lips are for radiation (0: a rigid end). Only a true closure
  // (under ~0.03 cm^2) is sealed; a labial fricative's 0.1 cm^2 radiates.
  float lipOpen_ = 1;
  bool subglottal_ = false;
  float lungDc_ = 0;  // the mean flow the lungs supply (the matched source absorbs waves, not the DC)
  float uL_ = 0, uN_ = 0, prevUN_ = 0, radiated_ = 0;
};

}  // namespace puhuja

namespace puhuja {

struct TractShape;

// The whole airway as lines: the oral line from the glottis to the lips, a
// nasal line from the velum to the nostrils, the consonant's constriction
// laid over the vowel shape, turbulence noise where the flow is fast (at
// the constriction and at the glottis), and the sound radiated from both
// openings. Driven one substep at a time by the folds' flow; gives back
// the pressure above the folds.
class VocalTractLine {
 public:
  static constexpr int kOralSections = 64, kNasalSections = 40;
  void init(float fs);
  // The vowel shape, with a constriction of area ac (cm^2; >= 1: none) at
  // x (share of the length; < 0: none) and the velum open by an (cm^2).
  void setShape(const TractShape& shape, float x, float ac, float an);
  // Turbulence: gains for the constriction and the glottis, and the glottal
  // area the flow passes (cm^2).
  void setNoise(float fricGain, float aspGain, float glottalArea);
  void setGlottalArea(float a) { glotArea_ = std::max(a, 0.002f); }  // the folds' instantaneous area, per substep
  float substep(float u0);  // -> the pressure above the glottis
  // The lung pressure for the trachea (dyn/cm^2, per output sample), and the
  // pressure just below the folds after the last substep (the trachea's
  // acoustics on top of the lung pressure).
  void setLungPressure(float ps) { lungPressure_ = ps; }
  float subglottalPressure() const { return pSub_; }
  float output() const { return out_; }  // the radiated sound, anti-aliased
  const TransmissionLine& oral() const { return oral_; }

 private:
  float noise();
  TransmissionLine oral_, nasal_, sub_;  // sub_: the trachea, lungs to glottis
  float fs_ = 48000, dt_ = 0;
  float lungPressure_ = 0, pSub_ = 0;   // the lung pressure set per sample; the pressure under the folds
  int velumK_ = 0, noiseK_ = -1, noiseKc_ = 0;  // noiseK_: where the turbulence drives; noiseKc_: the constriction's centre
  float an_ = 0, uNasal_ = 0;
  float fricGain_ = 0, aspGain_ = 0, glotArea_ = 0.05f;
  float out_ = 0, aaNoise_ = 0, lpNoise_ = 0, lpNoise2_ = 0;
  // Decimation: the lip + nostril flow through a 4th-order low-pass at 18 kHz
  // at the substep rate, averaged over the kSubsteps of a sample, then
  // differentiated at the output rate (the radiated pressure).
  LineLowPass aa1_, aa2_;
  float uAcc_ = 0, prevMean_ = 0;
  int subCount_ = 0;
  unsigned rng_ = 0x9e3779b9u;
};

}  // namespace puhuja
