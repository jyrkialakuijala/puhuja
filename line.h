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
#include <cstdint>
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
  float invDx() const { return invDx_; }
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
  void setWallLoss(float mul) { wallLoss_ = mul; }
  // Extra loss over sections [k0, k1): a soft closure (the lips' or tongue's
  // tissue damping the tube where it is shut). Takes effect at the next setAreas.
  void setSectionLoss(int k0, int k1, float mul) { for (int k = 0; k < kMaxSections; ++k) lossMul_[k] = (k >= k0 && k < k1) ? mul : 1.0f; }
  void setViscLoss(float mul) { viscLoss_ = mul; }
  void setHfLoss(float bandwidthHz, float cornerHz);  // the losses that grow with frequency (after init)
  void setHfF2(float bwHzAt8k) { hfF2Bw_ = bwHzAt8k; }  // a loss ~ f^2: this many Hz of bandwidth at 8 kHz (0: none)
  void setSubglottal(bool on) { subglottal_ = on; }
  float substepSubglottal(float lungPressure, float uGlottis);
  // The sound: the radiated flow's rate of change (cm^3/s^2).
  float radiated() const { return radiated_; }
  float lipFlow() const { return uN_; }
  // The walls' total volume velocity (cm^3/s): the yielding tissue moves with
  // the pressure, and radiates (the throat and cheeks carry a voice bar).
  float wallFlow() const {
    float s = 0;
    for (int k = 0; k < n_; ++k) s += sWall_[k] * wallV_[k];
    return s;
  }
  float pressureAt(int k) const { return p_[k]; }
  float cPAt(int k) const { return cP_[k]; }  // dp per unit of net outflow in a substep (LINE_EXPAND_PHYS)
  float areaAt(int k) const { return a_[k]; }
  float flowAt(int k) const { return u_[k]; }  // interface k
  // Side effects between substeps: a flow leaving section k (into a side
  // branch) and a pressure added at section k (a noise source).
  void sink(int k, float u) { p_[k] -= cP_[k] * u; }
  void addPressure(int k, float dp) { p_[k] += dp; }
  // A pressure difference dp (dyn/cm^2) driving interface k this substep (a
  // dipole source, as turbulence downstream of a constriction is).
  void addFlowDrive(int k, float dp) { u_[k] += cU_[k] * dp; }
  void addPressureDrive(int k, float dp) { p_[k] += dp; }  // a pressure source in section k (not throttled by the interfaces' orifice loss)

 private:
  void coefficients();
  void hfLoss(int nb);  // the losses that grow with frequency (line.cc)
  alignas(64) float uLp_[kMaxSections + kBlock] = {};  // the flow low-passed at LINE_HF_FC, per interface
  float hfK_ = 0, hfA_ = 0;
  float hfF2Bw_ = 0;  // (setHfF2)
  int n_ = 0, blocks_ = 0;
  float fs_ = 48000, dt_ = 0, dx_ = 0, invDx_ = 0;  // invDx_ = 1 / dx_ (divisions in the per-substep loops are multiplications)
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
  float wallLoss_ = 1.0f;  // multiplies the wall damping (set before the areas)
  alignas(64) float lossMul_[kMaxSections] = {};  // per-section multiplier of the viscous and wall losses (1 = none; 0 is treated as 1)
  float viscLoss_ = 1.0f;  // multiplies the viscous series loss

 public:
  float traceExtra_ = 0;  // a value the owner wants in the trace (the nostril flow)
  float traceExtra2_ = 0;  // ... and another (the lung pressure)
  // A Helmholtz side branch (a paranasal sinus): a neck of area aN and length
  // lN into a volume v, opening at section k. Its flow is a sink there.
  struct Sinus { int k = -1; float m = 0, cInv = 0, r = 0, u = 0, p = 0, dtOverM = 0, invDen = 0; };  // dtOverM, invDen: per-substep factors, from m and r
  void addSinus(int k, float volume, float neckArea, float neckLength, float q);
  void retuneSinus(int i, float volume, float neckArea, float neckLength, float q);  // (NASAL2 agent33) keeps its state
  void clearSinuses() { nSinus_ = 0; }  // (init runs more than once: without this the slots filled up and later sets were dropped)
  void sinusFlows();      // before substep: the neck flows from the old pressures
  void sinusPressures();  // after substep: the volumes and the sections move with them
  Sinus sinus_[4]; int nSinus_ = 0;  // (4, not 3: the members after keep their 64-byte phase, the sound stays bit-exact)

 private:
  float lungDc_ = 0;  // the slow part of the lung-to-trachea pressure difference (driven through kLungR)
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
  void setFemale(bool female);  // per-voice oral losses (LINE_HF_BW_F)
  void setOralHfMix(float consBw, float openness);  // the oral HF loss between consBw (closed) and the voice's (open)
  float oralHfBw_ = 100.0f;  // the voice's oral HF bandwidth (setFemale)
  bool oralSinus_ = false;  // a piriform side branch on the oral line (LINE_PIRIFORM_HZ)
  static constexpr int kOralSections = 64, kNasalSections = 40;
  void init(float fs, float nasalScale = 1.0f);  // nasalScale: the nasal line's length x (the female voice, LINE_NASAL_SCALE_F)
  // The vowel shape, with a constriction of area ac (cm^2; >= 1: none) at
  // x (share of the length; < 0: none) and the velum open by an (cm^2).
  void setShape(const TractShape& shape, float x, float ac, float an);
  // The same from an area function already sampled (cm^2 per oral section)
  // and a length: for shapes interpolated geometrically, section by section.
  void setShapeAreas(std::vector<float> areas, float length, float x, float ac, float an);
  // The constriction's half-width (cm; 1.0 for most consonants, ~1.8 for the
  // long groove of ʃ ʒ ʧ ʤ) and a lip opening cap (cm^2; <= 0: the vowel's):
  // the rounding of ʃ. Set before setShape.
  void setConstriction(float halfWidth, float lipCap) { humpHw_ = halfWidth; lipCap_ = lipCap; }
  void setHumpNorm(float k) { humpNorm_ = k; }
  void setExpandLp(float hz) { expandLpHz_ = hz; }  // LINE_EXPAND_LP for now (< 0: LINE_EXPAND_LP; BDG-Z agent20, LINE_BZ_EXPAND_LP)  // LINE_HUMP_NORM's power for now (< 0: LINE_HUMP_NORM; P agent 10, LINE_P_LIQ_NORM)
  // The frication jet's thickness for its Strouhal noise corner (cm; 0: the
  // slit's height, area / LINE_SLIT_W). f's air leaks through a few wider
  // gaps between lip and teeth, not one 0.3 mm sheet (2026-11-04).
  void setJetThickness(float d, float mix = 1, float slitScale = 1, float smoothMs = -1, float fcMin = 0, float fcMax = 0) {
    jetD_ = d; jetMix_ = mix; slitScale_ = slitScale; jetSmoothMs_ = smoothMs; jetFcMin_ = fcMin; jetFcMax_ = fcMax;
  }  // mix: 1 fixed d, 0 the slit's height x slitScale, between: geometric blend
  // GAP agent29: the frication jet's Re rise lag for now (ms; <= 0: LINE_NOISE_RE_MS)
  void setReMs(float ms) {
    if (ms == reMsNow_) return;
    reMsNow_ = ms;
    reKNow_ = ms > 0 ? 1.0f - std::exp(-dt_ * 1000.0f / ms) : -1.0f;
  }
  // SH agent30: the frication drive's fall limited to this many dB per ms (<= 0: off), down to floorDb under the level
  // it had when the limit was switched on; below that it follows its target (an offset's hiss fades instead of stopping)
  void setFallTrack() { fallTrack_ = true; }
  void setFallLimit(float dbPerMs, float floorDb) {
    if (dbPerMs == fallDbMs_ && floorDb == fallFloorDb_) return;
    if (dbPerMs > 0 && fallDbMs_ <= 0) fallStop_ = fallEnv_ * std::pow(10.0f, -floorDb / 20.0f);
    fallDbMs_ = dbPerMs;
    fallFloorDb_ = floorDb;
    fallMul_ = dbPerMs > 0 ? std::pow(10.0f, -dbPerMs * dt_ * 1000.0f / 20.0f) : 0.0f;
  }
  // GAP agent29: the frication drive's rise limited to this many dB per ms (<= 0: off), from floorDb under its target
  void setRiseLimit(float dbPerMs, float floorDb) {
    if (dbPerMs == riseDbMs_ && floorDb == riseFloorDb_) return;
    riseDbMs_ = dbPerMs;
    riseFloorDb_ = floorDb;
    riseMul_ = dbPerMs > 0 ? std::pow(10.0f, dbPerMs * dt_ * 1000.0f / 20.0f) : 0.0f;
    riseFloor_ = std::pow(10.0f, -floorDb / 20.0f);
    if (dbPerMs <= 0) riseEnv_ = -1;
  }
  // GAP agent29: the frication source's fade-out areas (LINE_FRIC_A0 / A1) x this for now (1: as set)
  void setSlitWiden(float w) { slitLogW_ = w > 0 && w != 1.0f ? std::log(w) : 0.0f; }
  // SH agent30: the frication source's fade-out window (LINE_FRIC_A0 .. A1 in log area) this many times as wide (1: as set)
  void setSlitSpan(float s) { slitSpan_ = s > 0 ? 1.0f / s : 1.0f; }
  // GAP agent29: the frication jet's critical Reynolds number x this for now (1: LINE_RE_CRIT)
  void setFricCritMul(float m) { critMulF_ = m > 0 ? m : 1.0f; }
  void setNoiseLp(float hz, bool asDefault = false);  // the turbulence noise's two-pole low-pass corner (Hz; <= 0: back to the default; BLEND agent28)
  void setFricLpRatio(float r) { fricLpr_ = r; }  // the frication band's low-pass / its corner (0: LINE_NOISE_LPR)
  void setFricReMul(float m) { fricReMul_ = m; }
  void setFricReSmooth(float s) { fricReSmooth_ = s; }  // (LINE_V_RE_SMOOTH)
  void setFricReBoth(bool b) { fricReBoth_ = b; }  // the jet's Reynolds number low-passed both ways (LINE_Z_RE_BOTH; false: only its rise)  // the frication jet's Reynolds number x this (LINE_Z_RE_SCALE; 1: as computed)
  void setFricFade(float s) { fricFade_ = s; fricLogA0_ = 0; }  // the frication source's fade-out areas x this (voiced fricatives)
  // The glottal noise's low-pass ratio during a voiceless stop's puff (<= 0:
  // LINE_GLOTTAL_LPR / LINE_NOISE_LPR): the puff's hiss reaches 2-5 kHz while
  // an h keeps its darker band (2026-11-04).
  void setPuffLpRatio(float r) { puffLpr_ = r; }
  void setGlotFcScale(float s) { glotSlitScale_ = s > 0 ? 1.0f / s : 1.0f; }  // the glottal noise band's corner x s (LINE_H_GFC)
  // The aspiration's high-pass corner from now on (Hz; 0: LINE_ASP_HP's), with
  // `poles` one-pole stages (0: LINE_ASP_HP_POLES): a t's own breath.
  void setAspHp(float hz, int poles);
  // (VQCS agent33) the glottal noise low-passed (two poles at hz) by this share (0: off): connected-speech breath without 8-10 kHz hiss
  void setCsAspLp(float hz, float mix) { csAspA_ = 1.0f - std::exp(-6.2831853f * hz * dt_); csAspMix_ = mix; }
  // NASAL agent32 (round AD): the nasal walls' radiation x s (1: as before); the nasal line's HF loss bandwidth (Hz)
  void setNasalRad(float s) { nasRadScale_ = s; }
  void setNasalHfBw(float bw) { nasal_.setHfLoss(bw, nasHfFc_); }
  // NASAL agent32: the nasal passage's port / cavity / nostril areas (cm^2; < 0: LINE_NASAL_PORT_A / _CAVITY_A / LINE_NOSTRIL_A)
  void setNasalGeom(float portA, float cavA, float nostA, float endX = -1.0f, float endA = 8.0f);  // endX >= 0: from endX cm on, endA (an open end there)
  // NASAL2 agent33 (round AE): the velum port's coupling point on the oral line moves to `frac` of the oral tract (from
  // the glottis) over `ms` (linear; a fractional point between two sections); frac < 0: back to velumK_ (0.55), then off.
  // Only the port's pressure tap and sink move (the stops' cavity taps stay at velumK_ - 4).
  void setNasalCoupling(float frac, float ms);
  // NASAL2 agent33: the paranasal sinuses' volumes x v1 (the maxillary pair) / v2 (the frontal), their Q x q (1: init's)
  void setSinusScale(float v1, float v2, float q);
  // A voiced closure: softer (its sections lose soft times more) and longer by
  // extraLength cm (b against p: the lips compress and protrude; Jyrki
  // 2026-11-03: "b needs to be a softer, more damped closure ... two
  // millimeters longer", its resonances ~1 % under p's).
  void setClosureSoftness(float soft, float extraLength) { closureSoft_ = soft; closureExtra_ = extraLength; }
  // A volume sink in the mouth (cm^3/s): the cavity expanding behind a voiced
  // closure (cheeks, larynx lowering), which keeps the pressure down so the
  // folds go on vibrating (the voice bar of b d g).
  // The sealed cavity squeezed (tongue root, larynx): an inflow behind the closure, cm^3/s (0: none).
  void setSqueeze(float flow) { if (flow > 0 && squeeze_ <= 0) squeezeV_ = 0; squeeze_ = flow; }
  void setExpansion(float flow, bool slow = false, bool closure = true) { expandFlow_ = flow; expandSlow_ = slow; expandClosure_ = closure; }
  void setExpandOffAllowed(bool on) { expandOffAllowed_ = on; }  // LINE_EXPAND_OFF_FRIC: this closure ends into a fricative  // slow: a planned (expanding) voiced closure, LINE_EXPAND_LP applies
  // ... and when the voiced closure hands over to the next stop's closure (d -> t: no release; LINE_EXPAND_OFF_STOP)
  void setExpandOffHandover(bool on) { expandOffHandover_ = on; }
  // Turbulence: gains for the constriction and the glottis, and the glottal
  // area the flow passes (cm^2).
  void setNoise(float fricGain, float aspGain, float glottalArea);
  // A continuous breath during voicing (LINE_BREATH): aspiration noise following the glottal flow's Re^2 without the
  // critical-Reynolds threshold, so it never switches off between phones (Jyrki 2026-11-06: the thresholded one "goes on
  // and off as an extra breathiness").
  void setBreath(float g) { breathGain_ = g; }
  void setNoiseAreaRef(float a) { noiseAreaRef_ = a; }
  void setGlottalArea(float a) { glotArea_ = std::max(a, 0.002f); }  // the folds' instantaneous area, per substep
  float substep(float u0);  // -> the pressure above the glottis
  // The lung pressure for the trachea (dyn/cm^2, per output sample), and the
  // pressure just below the folds after the last substep (the trachea's
  // acoustics on top of the lung pressure).
  void setLungPressure(float ps) { lungPressure_ = ps; }
  float subglottalPressure() const { return pSub_; }
  float output() const { return out_; }  // the radiated sound, anti-aliased
  const TransmissionLine& oral() const { return oral_; }
  float velumArea() const { return an_; }  // the velum's opening as the nasal branch sees it (cm^2)
  void reseed(uint64_t s) { rng_ = s | 1; }  // PUHUJA_SEED: the frication noise starts over from s (never 0)

 private:
  float noise();
  TransmissionLine oral_, nasal_, sub_;  // sub_: the trachea, lungs to glottis
  float fs_ = 48000, dt_ = 0;
  float lungPressure_ = 0, pSub_ = 0;   // the lung pressure set per sample; the pressure under the folds
  float humpHw_ = 1.0f, lipCap_ = -1.0f;
  float lipCapLn_ = 2.0794415f;  // (LINE_LIP_CAP_GLIDE_MS: the rounding now, ln cm^2; ln 8: none)
  float humpNorm_ = -1.0f;  // (setHumpNorm)
  float expandLpHz_ = -1.0f;  // (setExpandLp)
  float jetD_ = 0, jetMix_ = 1, slitScale_ = 1, jetSmoothMs_ = -1, jetFcMin_ = 0, jetFcMax_ = 0;
  float fricLpr_ = 0;
  float fricReMul_ = 1.0f;
  float fricReSmooth_ = 0.0f;
  bool fricReBoth_ = false;
  float puffLpr_ = 0;
  float glotSlitScale_ = 1;  // (setGlotFcScale)
  float expandFlow_ = 0, expandLast_ = 0;
  float expV_ = 0, expU_ = 0, expC_ = 1.0e-3f, expOn_ = 0, expPs_ = 0;  // LINE_EXPAND_PHYS: the yielding element's volume (cm^3), flow (cm^3/s) and compliance
  float expOff_ = 0;  // the yielding element's conductance share after its closure (LINE_EXPAND_PHYS_OFF_MS)
  bool expandOffAllowed_ = true;  // expandLast_: the flow the sink had (LINE_EXPAND_OFF_MS)
  bool expandOffHandover_ = false;
  float expandPc_ = 0;  // the cavity pressure the expansion sees, low-passed (LINE_EXPAND_LP)
  bool expandSlow_ = false;
  bool expandClosure_ = true;
  float squeeze_ = 0.0f, squeezeU_ = 0.0f, squeezeV_ = 0.0f;  // the flow is a closure's (LINE_EXPAND_PHYS), not a fricative's sink
  float humpAc_ = 2.0f;  // the constriction's own area: closes at once, releases slowly
  float closureSoft_ = 1.0f, closureExtra_ = 0.0f;
  int velumK_ = 0, noiseK_ = -1, noiseKc_ = 0;  // noiseK_: where the turbulence drives; noiseKc_: the constriction's centre
  float an_ = 0, uNasal_ = 0;
  float nasKf_ = -1.0f, nasKfT_ = -1.0f, nasKStep_ = 0.0f, nasKw_ = 0.0f;  // NASAL2 agent33: the coupling point (sections; < 0: off)
  int nasK0_ = 0;
  float velarG_ = 0, velarInv2A2_ = 0;  // dt an / (rho dx_nasal) and 1 / (2 an^2), set with an_
  float fricGain_ = 0, aspGain_ = 0, glotArea_ = 0.05f;
  int lastNoiseK_ = -1, moveFromK_ = -1; float moveW_ = 0;
  float reLp_ = 0;  // LINE_NOISE_RE_MS: the jet's Reynolds number, low-passed
  float reMsNow_ = 0, reKNow_ = -1;  // GAP agent29 (setReMs)
  float slitLogW_ = 0;  // GAP agent29 (setSlitWiden)
  float slitSpan_ = 1;  // SH agent30 (setSlitSpan; stored inverted)
  float critMulF_ = 1.0f;  // GAP agent29 (setFricCritMul)
  float riseDbMs_ = 0, riseFloorDb_ = 0, riseMul_ = 0, riseFloor_ = 0, riseEnv_ = -1;  // GAP agent29 (setRiseLimit)
  float fallDbMs_ = 0, fallFloorDb_ = 0, fallMul_ = 0, fallStop_ = 0, fallEnv_ = 0;  // SH agent30 (setFallLimit)
  bool fallTrack_ = false;
  float reMean_ = 0;  // LINE_NOISE_RE_ENV_MS: ... its mean over about a pitch period, which the lag acts on
  float reA_ = 0; int reK_ = -1;  // ... the constriction's area and place a substep ago (an opening one does not lag)  // LINE_NOISE_MOVE_MS: the source fading out of its old place
  float breathGain_ = 0;  // (setBreath, LINE_BREATH)
  float breathEnv_ = 0;   // the breath drive's envelope, Re^2 low-passed over LINE_BREATH_MS (0: instantaneous)
  float breathHp_[2] = {0, 0};  // the breath's own fixed band (LINE_BREATH_FC): two one-pole high-passes on the white source
  float aaNoiseDef_ = 0;
  float out_ = 0, aaNoise_ = 0, lpNoise_ = 0, lpNoise2_ = 0;
  float aspHpState_ = 0, aspHpA_ = 0, aspHpBaseA_ = 0;
  int aspHpPolesNow_ = 0;  // 0: LINE_ASP_HP_POLES
  float aspHpMore_[4] = {0, 0, 0, 0};  // LINE_ASP_HP_POLES' extra stages
  float csAspA_ = 0, csAspMix_ = 0, csAspLp_[2] = {0, 0};  // (VQCS agent33) setCsAspLp
  float noiseHp1_ = 0, noiseHp2_ = 0, noiseHpA_ = 0;
  float stG_[4] = {}, stF_[4] = {}, stAlphaG_ = 0.1f, stAlphaF_ = 0.1f, stLpG_ = 0.5f, stLpF_ = 0.5f;  // the Strouhal-banded noises (glottis, slit)
  unsigned stCount_ = 0;
  float noiseAreaRef_ = -1.0f;  // the constriction area the glide comes from or goes to, whichever is smaller (LINE_NOISE_AREA_TARGET)
  float areaGain_ = 1.0f;  // (LINE_NOISE_AREA_EXP) the constriction noise area factor, updated every 16 substeps
  unsigned areaCount_ = 0;
  float expandShare_ = 0.0f;  // the expansion sink's eased share (LINE_EXPAND_ON_MS)
  bool fricActive_ = false, aspActive_ = false;  // the jets drove noise in the last substep (LINE_NOISE_ST_SNAP)
  float stLogFcG_ = 0, stLogFcF_ = 0;
  float fricFade_ = 1.0f;  // x LINE_FRIC_A0 / A1 for the current sound
  float fricLogA0_ = 0;  // log(kFricA0 * fricFade_), cached (0: recompute)  // the corners' smoothed log frequencies (0: not yet set)  // all the turbulence noise: a 2-pole high-pass (LINE_NOISE_HP)  // the glottal noise's high-pass (turbulence has little below ~500 Hz)
  // Decimation: the lip + nostril flow through a 4th-order low-pass at 18 kHz
  // at the substep rate, averaged over the kSubsteps of a sample, then
  // differentiated at the output rate (the radiated pressure).
  LineLowPass aa1_, aa2_;
  LineLowPass wallLp_;  // the walls' radiation through the cheeks and the neck (LINE_WALL_RAD_LP)
  float wallLpHz_ = 0;
  float nasRadScale_ = 1.0f, nasHfFc_ = 1000.0f, nasalScale_ = 1.0f;  // NASAL agent32: the nasal walls' radiation x this (setNasalRad); the nasal HF loss corner
  float uAcc_ = 0, prevMean_ = 0;
  int subCount_ = 0;
  // The frication noise's state. Unsigned: the generator's signed form overflowed (UB; review cpp.md B4, dsp.md B4).
  uint64_t rng_ = 1611686018427387904ull;
  // LINE_NOISE_SOFT (2026-11-22, the tsirp / klumbgn root cause): the slit jet's Reynolds number averaged over about a
  // pitch period (two poles) and the drive's envelope (two poles, attack / release). Last, so the members before keep their place.
  float softRe_[2] = {}, softEnv_[2] = {};
  float softMin_ = 0;  // the jet's Re, its running minimum (LINE_NOISE_SOFT_TROUGH)
  float softHoldA_ = -1.0f, softHoldRe_ = 0;  // the area and the mean Re while the envelope was last fed (LINE_NOISE_SOFT_HOLD)
  float winMin_[32] = {}, winSum_[32] = {}, winMinOld_ = 0, winSumOld_ = 0; int winI_ = 0, winN_ = 0;  // (LINE_NOISE_SOFT_WIN_MS)
  float softA_ = -1.0f; int softK_ = -1;  // the area and place a substep ago (is the constriction opening?)
};

}  // namespace puhuja
