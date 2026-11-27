#include "larynx.h"
#include "params.h"
#include "line.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>

namespace puhuja {

namespace {

// --- two-mass folds and their driving -----------------------------------
// Fold rest position from the glottal opening: above the voicing position
// the folds move apart; below it they press together, firmly enough at
// ag = 0 (a pause, a glottal stop) to stop the vibration.
constexpr float kFoldRestPerAg = 0.8f;    // rest slit area (cm^2) per unit ag ...
constexpr float kFoldRestOffset = -0.02f; // ... plus this: ag 0.05 -> 0.02 cm^2
constexpr float kFoldRestPressed = -0.15f; // ag 0: pressed shut (vibration stops below ~-0.12)
// Abducted for a voiceless sound (ag = kAgOpen) the folds are this far
// apart. On the linear mapping (0.22 cm^2) all three fold models kept
// vibrating at full abduction, grid and string louder than in the vowel:
// every h was a voiced vowel with noise on top, +4 dB over its vowel in
// "hands" where readers are 3-13 dB under or voiceless (INPUT.md
// 2026-10-31). Real folds stop phonating well before 0.3 cm^2. From
// kAgBreathy (a voiced h) the rest grows smoothly to this.
const float kFoldRestOpen = param("kFoldRestOpen", 5.0327052f);  // (1.5 with STRING_REST 0.047 left the folds vibrating in f, a pitch swoop 126 -> 82 Hz through the f of "different", Jyrki: "a swirp", 2026-11-02)
// f0 of the fold model at tension 1: the tension mapping q = target / this.
// 155 is the two-mass model's; the string folds (the default source,
// STRING_F0 105.7) sound at about 128 Hz at q = 1 under the tract's load and
// the speech pressure, so with 155 every vowel came out 0.82 of its command
// (male 117.8 -> 97.6 Hz, female 168.6 -> 136.3; review voice.md B4,
// 2026-11-11). The controller below then engaged only in strong vowels
// (physVib > kPitchLearnVib) and pushed those up to 1.24x, so the melody
// was partly the folds' (Kokoro mode: 20 % of cycles corrected, per-vowel
// error sd 0.81 st, realised-vs-commanded slope 0.88). With 126 the vowels
// sit at 1.00 / 0.98 of the command (sd 0.27 st, slope 0.98) and the
// controller stays at 1. To keep today's average pitch with that, lower
// kMaleF0 / kFemaleF0 by 0.83 / 0.81 (ref/opt/lab/BUG-PITCH/keep.params).
const float kFoldNaturalF0 = param("PITCH_NATURAL_F0", 124.28105f);
const float kPitchControl = param("PITCH_CONTROL", 0.30141608f);     // per cycle: share of pitch error corrected
const float kPitchLearnVib = param("kPitchLearnVib", 98.049835f);     // flow fluctuation (cm^3/s) needed to learn (100 before 2026-11-02)
const float kPitchCorrMin = param("kPitchCorrMin", 0.5f), kPitchCorrMax = param("kPitchCorrMax", 2.0f);  // (0.7-1.4 pinned the string folds half an octave low, 2026-11-02)  // the tension correction's range
// Tract load on the folds: each formant mode turns the flow into pressure
// above the glottis (physically the coupling is 30-100x stronger; see
// DESIGN_folds_physics.md).
constexpr float kTractCoupling[3] = {27.0f, 13.5f, 7.2f};
constexpr float kPhysicalGain = 0.5460f;  // flow change per sample to source level
constexpr float kAirPulse = 0.25f;  // share of the high breath pulsed by the flow
// A half-open glottis lets the folds sag in pitch; the speaker's tension
// makes up for it (see TwoMassLarynx::step).
constexpr float kBreathyPitchComp = 0.2f;
constexpr float kBreathyCompMaxAg = 0.18f;
constexpr float kPressedPitchComp = 0.0f;  // tension eased when pressed shut (off: the upturn was the pause's)
// Micro-variation of tension, rest gap and pressure, low-passed to ~kMicroHz
// (real voices: jitter ~1 %, shimmer ~0.6 dB); kMicro2Gain is its overall
// scale (the voice's microGain is relative to 3).
constexpr float kMicro2Gain = 3.0f;
constexpr float kTensionJitter = 0.4f, kRestJitter = 0.12f, kPressureShimmer = 1.6f;
// Per-cycle gaussian deviations (off: they cost HNR). Knobs since round Z (VQ agent28, 2026-11-15): puhuja's
// frame-to-frame f0 jitter 0.19 % (m) / 0.27 % (f) against 11labs 0.38 %, Kokoro 0.30-0.57 %.
const float kCycleJitter = param("LARYNX_CYCLE_JITTER", 0.006f), kCycleShimmer = param("LARYNX_CYCLE_SHIMMER", 0.029724675f);
// The slow micro-variation x this (1: as before).
const float kMicroX = param("LARYNX_MICRO_X", 1.0f);
// Round Z (VQ agent28): a first-order high shelf on the folds' flow as the line tract receives it (dB above
// LARYNX_SRC_SHELF_HZ; 0: none). A sharper effective closure (the mucosal wave's abrupt cut-off): puhuja's voiced
// 4-8 kHz is 12-18 dB under 11labs / Kokoro, tilt -12.3 dB/oct vs -9.4; only the voice, not the jets' noise.
const float kSrcShelfDb = param("LARYNX_SRC_SHELF_DB", 7.9091164f), kSrcShelfHz = param("LARYNX_SRC_SHELF_HZ", 1875.0f);
// ... a second shelf after it (a negative one above the first makes a broad mid lift), and the female voice's own
// values (_F_; < -900 dB or <= 0 Hz: the male's). Cycle jitter for the female voice: LARYNX_CYCLE_JITTER_F (< 0: the male's).
const float kSrcShelf2Db = param("LARYNX_SRC_SHELF2_DB", 0.0f), kSrcShelf2Hz = param("LARYNX_SRC_SHELF2_HZ", 4000.0f);
const float kSrcShelfFDb = param("LARYNX_SRC_SHELF_F_DB", 0.0f), kSrcShelfFHz = param("LARYNX_SRC_SHELF_F_HZ", 0.0f);
const float kSrcShelf2FDb = param("LARYNX_SRC_SHELF2_F_DB", 0.0f), kSrcShelf2FHz = param("LARYNX_SRC_SHELF2_F_HZ", 0.0f);
const float kCycleJitterF = param("LARYNX_CYCLE_JITTER_F", 0.0f), kCycleShimmerF = param("LARYNX_CYCLE_SHIMMER_F", 0.0f);
// Round AA (VQ2 agent29): a peaking filter (RBJ, LARYNX_SRC_DIP_DB at LARYNX_SRC_DIP_HZ, Q LARYNX_SRC_DIP_Q) after the
// shelves, x shelfScale like them: the 1875 Hz shelf lifts 3-4 kHz 6-8 dB over both 11labs and Kokoro onyx (they dip
// there). 0 dB: none. _F_DB: the female's (< -900: the male's).
const float kSrcDipDb = param("LARYNX_SRC_DIP_DB", 0.0f), kSrcDipHz = param("LARYNX_SRC_DIP_HZ", 3600.0f);
const float kSrcDipQ = param("LARYNX_SRC_DIP_Q", 1.0f), kSrcDipFDb = param("LARYNX_SRC_DIP_F_DB", -999.0f);
// Round AE (VQCS agent33, lab; 0: off): connected-speech source quality, x LarynxControl::cs (0 in single words, see
// VQCS_ON in puhuja.cc): extra shelf dB (x shelfScale like the shelves: the vowels) VQCS_SHELF_DB / VQCS_SHELF2_DB, the
// voicing rest gap x (1 + (VQCS_REST_X - 1) cs) (the open quotient). _F: the female's (< -900: the male's); VQCS_SHELF_HZ_F /
// VQCS_SHELF2_HZ_F: the female's corners while her own shelves are off (0: the male's).
const float kVqcsShelfDb = param("VQCS_SHELF_DB", 0.0f), kVqcsShelfFDb0 = param("VQCS_SHELF_F_DB", -999.0f);
const float kVqcsShelf2Db = param("VQCS_SHELF2_DB", 0.0f), kVqcsShelf2FDb0 = param("VQCS_SHELF2_F_DB", -999.0f);
const float kVqcsShelfFDb = kVqcsShelfFDb0 > -900 ? kVqcsShelfFDb0 : kVqcsShelfDb;
const float kVqcsShelf2FDb = kVqcsShelf2FDb0 > -900 ? kVqcsShelf2FDb0 : kVqcsShelf2Db;
const float kVqcsShelfFHz = param("VQCS_SHELF_HZ_F", 0.0f), kVqcsShelf2FHz = param("VQCS_SHELF2_HZ_F", 0.0f);
// VQCS_LOW_DB / _HZ (_F): a low shelf on the source flow in connected speech, + dB below the corner (H1, a rounder pulse).
const float kVqcsLowDb = param("VQCS_LOW_DB", 0.0f), kVqcsLowFDb0 = param("VQCS_LOW_F_DB", -999.0f);
const float kVqcsLowFDb = kVqcsLowFDb0 > -900 ? kVqcsLowFDb0 : kVqcsLowDb;
const float kVqcsLowHz = param("VQCS_LOW_HZ", 200.0f), kVqcsLowFHz = param("VQCS_LOW_F_HZ", 300.0f);
// (FEMV agent34 round AF, lab; 0: off) VQCS_LOW_HP_HZ: the low shelf boosts only the band between this corner and
// VQCS_LOW_(F_)HZ (a band boost around H1): the flow's DC and slow drift are left alone (no thumps at voicing edges).
const float kVqcsLowHpHz = param("VQCS_LOW_HP_HZ", 0.0f);
// (FEMV agent34, lab; 0: off) VQCS_LOW_OPEN 1: the low (band) boost x shelfScale (the open-tract gate of the source
// shelves, LARYNX_SRC_SHELF_OPEN): not in consonants, so the flow transients of releases are not boosted; 2: also x the
// vibration estimate vibDrive (0..1: only while the folds vibrate).
const float kVqcsLowOpen = param("VQCS_LOW_OPEN", 0.0f);
const float kVqcsRestX = param("VQCS_REST_X", 1.0f), kVqcsRestXF0 = param("VQCS_REST_X_F", -999.0f);
const float kVqcsRestXF = kVqcsRestXF0 > -900 ? kVqcsRestXF0 : kVqcsRestX;
constexpr float kMicroHz = 8;
constexpr int kFoldOversample = 4;
constexpr float kFoldAntiAliasHz = 18000;

// The grid's share of the micro-variation (Jyrki: less micro-fluctuation
// there; its irregularity comes from its mass inhomogeneity instead).
constexpr float kGridMicroScale = 0.33f;
// The block folds get none for now (Jyrki, 2026-10-31: "remove all
// micro-variations for now"); their static mass pattern (BLOCK_INHOM, BLOCK_ASYM) stays.
constexpr float kBlockMicroScale = 0.0f;

// --- the drawn pulse ------------------------------------------------------
constexpr float kShapedGain = 10.0f;  // the vowels at the physical source's level (the pulse rides on the folds' envelope)
constexpr float kShapedAcFull = 7.63f;    // the folds' envelope (foldAc) of a full vowel
constexpr float kShapedEnvToPeak = 7.2f;  // envelope to cycle peak
constexpr float kShapedFastFloor = 0.15f;  // the fast envelope under this share of the cycle peak: the folds went quiet
constexpr float kShapedPeriodShare = 0.95f;
constexpr bool kShapedLocked = false;  // restart the pulse at each fold closure (beat-free, but worse in round trips)

// --- the simple waveforms -------------------------------------------------
constexpr float kOpenQuotient = 0.6f;   // glottal pulse: open share of the period
constexpr float kSquareGain = 0.37f;
constexpr float kTriangleGain = 2.48f;
constexpr float kTriangleOpenQuotient = 0.8f;

}  // namespace

std::vector<Larynx::Param> Larynx::controls() const {
  return {{"f0", "Hz", 40, 1000, 100}, {"ps", "dyn/cm^2", 0, 3 * kLungPressure, kLungPressure},
          {"ag", "cm^2", 0, 1, kAgVoice}};
}

// ---------------------------------------------------------------------------
// FoldLarynx

void FoldLarynx::init(float fs, float baseF0, uint64_t* rng) {
  (void)baseF0;
  fs_ = fs;
  rng_ = rng;
  kVibTrack_ = 1.0f - std::exp(-1.0f / msToSamples(10, fs_));
  foldAcHp2_.highPass(250.0f, fs_);
  kPitchRelax_ = 1.0f - std::exp(-1.0f / msToSamples(200, fs_));
  kBreathyCompFollow_ = 1.0f - std::exp(-1.0f / msToSamples(8.0f, fs_));
  kMicroSmooth_ = 1.0f - std::exp(-2.0f * kPi * kMicroHz / fs_);
  // A 6th-order Butterworth (three identical Q 0.707 sections drooped
  // 3.7 dB at 12-16 kHz).
  const float aaQ[3] = {0.5176f, 0.7071f, 1.9319f};
  for (int k = 0; k < 3; ++k) foldAa_[k].lowPass(kFoldAntiAliasHz, kFoldOversample * fs_, aaQ[k]);
  // Let the folds settle, pressed together, before the first sample (from
  // the neutral position they would spring open with a click).
  // Without lung pressure, as at the first sample (silence has ps = 0):
  // settled under kLungPressure, the grid's pressed folds sprang back when
  // the pressure vanished at the first step, a 500 Hz thump before the
  // speech (Jyrki: "a blimb before it starts", 2026-10-30).
  core_->reset();
  for (int i = 0; i < static_cast<int>(0.05f * fs_); ++i)
    core_->step(1.0f / fs_, {0, 0, 0.7f, kFoldRestPressed});
  prevFlow_ = core_->flow();
}

std::vector<Larynx::Param> FoldLarynx::controls() const {
  auto p = Larynx::controls();
  p.push_back({"gap", "cm^2", 0, 0.1f, 0});
  p.push_back({"irregular", "", 1, 3, 1});
  p.push_back({"microGain", "", 0, 10, 3});
  return p;
}

void FoldLarynx::step(const LarynxControl& c, LarynxOutput& out) {
  // Pitch is set by fold tension; a slow controller steers the measured
  // f0 toward the target (as a speaker does), leaving fast effects of
  // pressure and coupling in place.
  const float target = c.f0;
  const float ag = c.ag;
  VocalFolds::Input in;
  in.ps = c.ps;
  in.p1 = p1_;
  in.q = target / kFoldNaturalF0 * pitchCorr_ * c.tensionPrior;
  // (pajupilli, singing) the folds' mass towards what is asked: a critically damped follower, 98 % of a change in
  // 3 ms (w = 5.83 / 3 ms), so the mass never jumps under the tension that was fitted to it.
  if (c.foldMass != foldMass_ || foldMassV_ != 0) {
    const float w = 1944.0f, h = 1.0f / fs_;
    foldMassV_ += h * (w * w * (c.foldMass - foldMass_) - 2 * w * foldMassV_);
    foldMass_ += h * foldMassV_;
    if (std::fabs(c.foldMass - foldMass_) < 1e-5f && std::fabs(foldMassV_) < 1e-3f) foldMass_ = c.foldMass, foldMassV_ = 0;
  }
  in.mass = foldMass_;
  in.shield = c.shield, in.shieldHz = c.shieldHz;
  // A half-open glottis (breathy h between vowels) lets the folds sag in
  // pitch (-2.4 st in "few who"), faster than the pitch control learns:
  // the tension makes up for it, following the opening over ~8 ms (at
  // once it dropped a semitone at the vowel; over 40 ms it lagged).
  // Only in the breathy range (an h between vowels): opening further, for
  // a voiceless sound, the folds stop anyway, and the added tension raised
  // the pitch at the end of a vowel before it ("God." -> t).
  // Pressing together (a pause's closing, creak) raises the folds' pitch:
  // the phrase ended with a small upturn; the tension eases off for it.
  const float pressed = ag < kAgVoice ? -kPressedPitchComp / kBreathyPitchComp * (kAgVoice - ag) / kAgVoice : 0.0f;
  // (Tapered out between 0.15 and 0.21 cm^2: cut off at 0.18 the tension
  // dropped up to 6 % within 8 ms at vowel -> s/f offsets.)
  const float breathy = (std::clamp((ag - kAgVoice) / (kAgOpen - kAgVoice), 0.0f, 1.0f) + pressed) *
                        (1.0f - smoothstep(kBreathyCompMaxAg - 0.03f, kBreathyCompMaxAg + 0.03f, ag));
  breathyComp_ += (breathy - breathyComp_) * kBreathyCompFollow_;
  in.q *= 1.0f + kBreathyPitchComp * breathyComp_;
  // Micro-variation that the pitch controller does not follow (it learns
  // only slowly): tension, rest gap and lung pressure fluctuate a little,
  // low-passed to ~30 Hz. Real voices: jitter ~1 %, shimmer ~0.6 dB. It
  // also keeps the highest harmonics from being perfectly periodic, which
  // buzzes when the highs are brought up.
  // Two one-pole stages (one let the variation's slope through as noise
  // in the highs: -6 dB/oct only); x1.4 keeps its RMS.
  float micro[3];
  for (int k = 0; k < 3; ++k) {
    microLp_[k] += (noise() - microLp_[k]) * kMicroSmooth_;
    microLp2_[k] += (microLp_[k] - microLp2_[k]) * kMicroSmooth_;
    micro[k] = kMicro2Gain * c.microGain * microScale_ / 3.0f * microLp2_[k];
    if (kMicroX != 1.0f) micro[k] *= kMicroX;
  }
  const float irregular = c.irregular;
  in.q *= 1.0f + kTensionJitter * irregular * micro[0];
  in.ps *= 1.0f + kPressureShimmer * micro[2];
  in.q *= 1.0f + (c.female && kCycleJitterF >= 0 ? kCycleJitterF : kCycleJitter) * c.cycleJitter * cycleJ_;
  in.ps *= 1.0f + (c.female && kCycleShimmerF >= 0 ? kCycleShimmerF : kCycleShimmer) * c.cycleShimmer * cycleS_;
  // Round AC (CREAK agent31): period doubling at a creaky phrase end, alternate cycles stronger / weaker and longer /
  // shorter (0: off, as before).
  if (c.creakJ > 0) in.q *= 1.0f + c.creakJ * cycleJ_;
  if (c.creakS > 0) in.ps *= 1.0f + c.creakS * cycleS_;
  if (c.diplo > 0) {
    static const float kDiploQ = param("CREAK_DIPLO_Q", 0.5f);
    in.ps *= 1.0f + c.diplo * diploSign_;
    in.q *= 1.0f + kDiploQ * c.diplo * diploSign_;
  }
  // Fold rest position from the glottal opening (see kFoldRestPerAg).
  const float restVoice = kFoldRestPerAg * kAgVoice + kFoldRestOffset;
  in.rest = ag >= kAgVoice ? kFoldRestPerAg * ag + kFoldRestOffset
                           : kFoldRestPressed + (restVoice - kFoldRestPressed) * ag / kAgVoice;
  if (ag > kAgBreathy) {
    const float restBreathy = kFoldRestPerAg * kAgBreathy + kFoldRestOffset;
    in.rest = restBreathy + (kFoldRestOpen - restBreathy) * smoothstep(kAgBreathy, kAgOpen, ag);
    // Wider than a fricative's: a stop's puff, the tract open and the full
    // lung pressure across the folds, which at kFoldRestOpen kept flapping
    // (a breathy voice under 200 Hz for an aspirated t, 2026-11-04).
    if (ag > kAgOpen) in.rest = kFoldRestOpen * ag / kAgOpen;
  }
  // Its micro-variation: added after the rest position is set (before
  // it, the assignment overwrote it, and the jitter never acted).
  in.rest += kRestJitter * irregular * micro[1];
  // A posterior gap (female voices): softer closure, stronger fundamental.
  // The two-mass folds take it in their rest area; a model with segments
  // along the cord places it at the posterior end itself.
  if (core_->ownChink()) in.gap = c.gap * smoothstep(0.0f, kAgVoice, ag);
  else in.rest += c.gap * smoothstep(0.0f, kAgVoice, ag);
  // Round AA (VQ2 agent29): the voicing rest gap x LARYNX_REST_X (male) / LARYNX_REST_X_F (female; < 0: the male's),
  // fading to x1 as the glottis opens past breathy (a per-voice STRING_REST: the open quotient, H1-H2). 1: as before.
  {
    static const float restX = param("LARYNX_REST_X", 1.015464f), restXF = param("LARYNX_REST_X_F", -1.0f);
    const float x = c.female && restXF >= 0 ? restXF : restX;
    if (x != 1.0f) {
      const float k = 1.0f + (x - 1.0f) * (1.0f - smoothstep(kAgBreathy, kAgOpen, ag));
      in.rest *= k;
      in.gap *= k;
    }
  }
  {  // (VQCS agent33) the connected-speech open quotient
    const float x = c.female ? kVqcsRestXF : kVqcsRestX;
    if (x != 1.0f && c.cs > 0.0f) {
      const float k = 1.0f + (x - 1.0f) * c.cs * (1.0f - smoothstep(kAgBreathy, kAgOpen, ag));
      in.rest *= k;
      in.gap *= k;
    }
  }
  if (c.restX != 1.0f) {  // (pajupilli, singing: Synth::setAdduction) the voicing rest gap and the chink x restX
    const float k = 1.0f + (c.restX - 1.0f) * (1.0f - smoothstep(kAgBreathy, kAgOpen, ag));
    in.rest *= k;
    in.gap *= k;
  }
  // The folds run kFoldOversample times faster than the output, and their
  // flow is low-passed before it is taken once per sample: the flow drops
  // to zero within a sample at each closure, and sampled directly that
  // corner aliases (inharmonic hiss 13-20 dB under the harmonics above
  // 8 kHz).
  float u = 0;
  bool closed = false;
  double closeAt = 0;  // closure time within this sample (0..1)
  // With a line tract the lungs' pressure (with its shimmer) drives the
  // trachea, and the folds see the trachea's pressure below and the line's
  // above, whole: the coupling is the physics (the halved flow and feedback
  // of the first version were patches for an uncalibrated fold flow, review
  // 2026-11-01).
  if (line_) line_->setLungPressure(in.ps);
  for (int k = 0; k < kFoldOversample; ++k) {
    if (line_) {
      in.p1 = p1_;
      in.ps = line_->subglottalPressure();
      line_->setGlottalArea(core_->area());
    }
    const float raw = core_->step(1.0f / (kFoldOversample * fs_), in);
    if (line_) {
      const float db1Full = c.female && kSrcShelfFDb > -900 ? kSrcShelfFDb : kSrcShelfDb;
      const float db2Full = c.female && kSrcShelf2FDb > -900 ? kSrcShelf2FDb : kSrcShelf2Db;
      const float dbDFull = c.female && kSrcDipFDb > -900 ? kSrcDipFDb : kSrcDipDb;
      const float cs1 = c.female ? kVqcsShelfFDb : kVqcsShelfDb, cs2 = c.female ? kVqcsShelf2FDb : kVqcsShelf2Db;
      const float csL = c.female ? kVqcsLowFDb : kVqcsLowDb;
      if (db1Full != 0.0f || db2Full != 0.0f || dbDFull != 0.0f || cs1 != 0.0f || cs2 != 0.0f || csL != 0.0f) {
        // (the low-passes keep running where shelfScale is 0: no stale state when the shelf fades back in)
        float db1 = db1Full * c.shelfScale, db2 = db2Full * c.shelfScale;
        if (cs1 != 0.0f) db1 += cs1 * c.cs * c.shelfScale;  // (VQCS agent33)
        if (cs2 != 0.0f) db2 += cs2 * c.cs * c.shelfScale;
        // y = x + (G - 1) (x - lp(x)): unity below the corner, G above it.
        const float hz1 = c.female && kSrcShelfFHz > 0 ? kSrcShelfFHz : c.female && db1Full == 0.0f && kVqcsShelfFHz > 0 ? kVqcsShelfFHz : kSrcShelfHz;
        const float hz2 = c.female && kSrcShelf2FHz > 0 ? kSrcShelf2FHz : c.female && db2Full == 0.0f && kVqcsShelf2FHz > 0 ? kVqcsShelf2FHz : kSrcShelf2Hz;
        const auto memo = [](Memo& m, float x, auto f) { if (!(x == m.in)) { m.in = x; m.out = f(x); } return m.out; };
        const auto p10 = [](float x) { return std::pow(10.0f, x); };
        const auto ex = [](float x) { return std::exp(x); };
        const float g1 = memo(mG1_, db1 / 20.0f, p10) - 1.0f, g2 = memo(mG2_, db2 / 20.0f, p10) - 1.0f;
        const float a1 = 1.0f - memo(mA1_, -2.0f * 3.14159265f * hz1 / (kFoldOversample * fs_), ex);
        const float a2 = 1.0f - memo(mA2_, -2.0f * 3.14159265f * hz2 / (kFoldOversample * fs_), ex);
        srcLp_ += a1 * (raw - srcLp_);
        const float y1 = raw + g1 * (raw - srcLp_);
        srcLp2_ += a2 * (y1 - srcLp2_);
        float y2 = y1 + g2 * (y1 - srcLp2_);
        if (dbDFull != 0.0f) {
          const float A = memo(mDipA_, dbDFull * c.shelfScale / 40.0f, p10);
          const float w0 = 2.0f * 3.14159265f * kSrcDipHz / (kFoldOversample * fs_);
          if (!(w0 == dipW0In_)) { dipW0In_ = w0; dipCs_ = std::cos(w0); dipSn_ = std::sin(w0); }
          const float cs = dipCs_, al = dipSn_ / (2.0f * kSrcDipQ);
          const float ia0 = 1.0f / (1.0f + al / A);
          const float b0 = (1.0f + al * A) * ia0, b1 = -2.0f * cs * ia0, b2 = (1.0f - al * A) * ia0;
          const float a1d = -2.0f * cs * ia0, a2d = (1.0f - al / A) * ia0;
          const float o = b0 * y2 + dipZ1_;
          dipZ1_ = b1 * y2 - a1d * o + dipZ2_;
          dipZ2_ = b2 * y2 - a2d * o;
          y2 = o;
        }
        if (csL != 0.0f) {  // (VQCS agent33) the low shelf: y + (G - 1) lp(y)
          const float aL = 1.0f - memo(mAL_, -2.0f * 3.14159265f * (c.female ? kVqcsLowFHz : kVqcsLowHz) / (kFoldOversample * fs_), ex);
          srcLpL_ += aL * (y2 - srcLpL_);
          float lowBand = srcLpL_;
          if (kVqcsLowHpHz > 0.0f) {  // (FEMV agent34) minus the DC / drift below VQCS_LOW_HP_HZ
            const float aH = 1.0f - memo(mAH_, -2.0f * 3.14159265f * kVqcsLowHpHz / (kFoldOversample * fs_), ex);
            srcLpLH_ += aH * (y2 - srcLpLH_);
            lowBand -= srcLpLH_;
          }
          if (c.cs > 0.0f) y2 += (memo(mCsL_, csL * c.cs * (kVqcsLowOpen > 1.5f ? std::clamp(c.vibDrive, 0.0f, 1.0f) * c.shelfScale : kVqcsLowOpen > 0.0f ? c.shelfScale : 1.0f) / 20.0f, p10) - 1.0f) * lowBand;
        }
        p1_ = line_->substep(y2);
      } else {
        p1_ = line_->substep(raw);
      }
    }
    u = foldAa_[2].process(foldAa_[1].process(foldAa_[0].process(raw)));
    if (core_->closedNow()) {
      closed = true;
      closeAt = (k + core_->closeFraction()) / kFoldOversample;
    }
  }
  out.flow = u;
  const float pitchLearnVib = c.pitchLearnVib >= 0 ? c.pitchLearnVib : kPitchLearnVib;  // (singing: Synth::setPitchControl)
  const float pitchControl = c.pitchControl >= 0 ? c.pitchControl : kPitchControl;
  if (physVib_ < pitchLearnVib) pitchCorrTarget_ += (1.0f - pitchCorrTarget_) * kPitchRelax_;
  // The cycle for the pitch controller is timed from the flow waveform: its
  // upward crossing of the mean after a dip under half of it, once per
  // period. The fold models' closure events came 2-4 times per cycle with
  // the string folds (the lanes shut at different moments), the controller
  // read the voice as too high and held it half an octave under its
  // command (93 Hz for 126; the whole sentence a drone, 2026-11-02).
  bool cycleMark = false;
  if (u < 0.5f * flowMean_) belowHalf_ = true;
  else if (belowHalf_ && u > flowMean_ && flowMean_ > 5.0f) {
    belowHalf_ = false;
    cycleMark = true;
  }
  if (cycleMark) {
    // Period from closure to closure, timed to a small fraction of a
    // sample (the zero crossing of the opening between the folds'
    // substeps; whole samples were only good to ~3 cents at 92 Hz).
    const double closeTime = c.t + (closed ? closeAt : 0.5);
    const double period = closeTime - lastCloseTime_;
    // The next cycle's deviations (a rough gaussian from two draws).
    cycleJ_ = std::clamp(0.87f * (noise() + noise()), -2.5f, 2.5f);
    cycleS_ = std::clamp(0.87f * (noise() + noise()), -2.5f, 2.5f);
    diploSign_ = -diploSign_;
    lastCloseTime_ = closeTime;
    const float measured = static_cast<float>(fs_ / std::max(period, 1.0));
    {
      static FILE* ptrace = std::getenv("PUHUJA_PITCH_TRACE") ? std::fopen(std::getenv("PUHUJA_PITCH_TRACE"), "w") : nullptr;
      if (ptrace) std::fprintf(ptrace, "%.4f\t%.1f\t%.1f\t%.3f\t%.1f\n", c.t / fs_, target, measured, pitchCorrTarget_, physVib_);
    }
    // Learn only from steady, strong vibration: during onsets and offsets
    // the closures are irregular and would detune the folds.
    // (the correction's range was 0.7-1.4: with the string folds swinging
    // wider the voice sat at 93 Hz against a commanded 126 and the
    // controller could not reach it, the sentence a flat drone, 2026-11-02)
    if (physVib_ > pitchLearnVib && measured > 0.5f * target && measured < 2.0f * target)
      pitchCorrTarget_ = std::clamp(
          pitchCorrTarget_ * static_cast<float>(std::pow(static_cast<double>(target) / measured, pitchControl)),
          kPitchCorrMin, kPitchCorrMax);
  }
  // The correction glides over about a cycle instead of jumping at each
  // closure.
  pitchCorr_ += (pitchCorrTarget_ - pitchCorr_) * std::min(1.0f, target / fs_);
  // Tract feedback: each formant mode turns the flow into pressure.
  // (With a line tract attached, p1_ is the line's own pressure from the
  // last substep and stays: overwritten here, the folds saw a different
  // pressure on the first substep of every sample, a 48 kHz glitch that
  // put 30 dB of hiss on the voice, 2026-11-01.)
  if (!line_) {
    const float nyq = 0.45f * fs_;
    p1_ = 0;
    for (int k = 0; k < 3; ++k) {
      tractMode_[k].set(std::clamp(c.formant[k], 100.0f, nyq), std::max(c.bw[k], 20.0f), fs_);
      p1_ += kTractCoupling[k] * tractMode_[k].process(u);
    }
  }
  // Radiated sound ~ rate of change of the flow (lip radiation).
  out.source = (u - prevFlow_) * kPhysicalGain;
  // The folds' loudness: their radiated sound's envelope (~10 ms). The
  // flow's slow changes as the glottis opens or closes do not count, so
  // under an aspiration or an f it is ~0, unlike physVib_.
  // Measured above 250 Hz: soft phonation (the folds barely touching,
  // as a stop closes or voicing starts) is nearly sinusoidal and has almost
  // nothing there, and the drawn pulse, which keeps its harmonics at any
  // level, must go as quiet as the folds' sound does. The flow's slow
  // wander (the glottis opening, the micro-variation) goes with it.
  const float dflow = u - prevFlow_;
  const float dflowHp = foldAcHp2_.process(dflow);
  foldAc_ += (std::fabs(dflowHp) - foldAc_) * kVibTrack_;
  // A fast envelope too: within a steady cycle it swings between
  // the closure spike and the open phase, so it is not a level, but when
  // it falls under kShapedFastFloor of the last cycle's peak the folds have
  // gone quiet mid-cycle (a stop closing, voicing dying) and the drawn
  // pulse follows within a quarter cycle instead of a whole one.
  // Its time constant is a quarter of the current period (a fixed 2 ms
  // let it sag through a low voice's long open phase, and the floor then
  // modulated the pulse within the cycle: crest 11 dB again).
  foldFast_ += (std::fabs(dflowHp) - foldFast_) * std::min(4.0f * c.phaseStep, 1.0f);
  // Per cycle: the loudest sample of the folds' sound in the current cycle
  // of the drawn pulse; the drawn pulse plays the previous cycle's.
  foldCycleMax_ = std::max(foldCycleMax_, std::fabs(dflowHp));
  if (closed && kShapedLocked) {
    foldCycleAmp_ = foldCycleMax_;
    foldCycleMax_ = 0;
  }
  sinceClose_ = closed ? 0 : sinceClose_ + 1;
  prevFlow_ = u;
  // Vibration strength: the flow's fluctuation around its mean, which
  // gates the breath noise (none while the folds are apart and still).
  flowMean_ += (u - flowMean_) * kVibTrack_;
  physVib_ += (std::fabs(u - flowMean_) - physVib_) * kVibTrack_;
  out.puff = std::min(physVib_ / 150.0f, 1.5f) * std::min(u / 400.0f, 1.5f);
  // High breath: mostly steady while the folds vibrate; only kAirPulse of
  // it follows the flow. Switched fully on and off by each glottal cycle
  // it is heard as a buzzing hiss. No stronger in h than in vowels.
  out.airPuff = std::min(physVib_ / 150.0f, 1.0f) *
                (1.0f - kAirPulse + kAirPulse * std::min(u / 400.0f, 1.0f));
  out.closed = closed;
  out.closeAt = static_cast<float>(closeAt);
}

// ---------------------------------------------------------------------------
// PulseLarynx

void PulseLarynx::init(float fs, float baseF0, uint64_t* rng) {
  fs_ = fs;
  baseF0_ = baseF0;
  folds_.init(fs, baseF0, rng);
  kTriLeak_ = 2.0f * kPi * 5.0f / fs_;  // ~5 Hz: removes drift, keeps the shape
  lipGain_ = fs_ / (2.0f * kPi * kLipRadiationHz);
}

void PulseLarynx::setPulse(const PulseShape& p) {
  pulse_ = &p;
  // On the source only: bursts, frication and aspiration are made in the
  // mouth and must not depend on how the cords work (Jyrki).
  shapedTreble_.highShelf(kTrebleShelfHz, p.trebleDb, fs_);
  pulseAtPitch(baseF0_);
}

std::vector<Larynx::Param> PulseLarynx::controls() const {
  auto p = folds_.controls();
  p.push_back({"f0Wave", "Hz", 40, 1000, 100});
  return p;
}

void PulseLarynx::pulseAtPitch(float f0) {
  const PulseShape& p = *pulse_;
  const float ratio = std::max(f0, 1.0f) / baseF0_;
  float share[PulseShape::kMaxN];
  float open = 0;
  for (int i = 0; i + 1 < p.n; ++i) {
    share[i] = p.share[i] * std::pow(ratio, p.pitchExp[i]);
    open += share[i];
  }
  // The closed phase takes the rest; if the open segments would overflow the
  // period, they shrink together to leave the minimum closed phase.
  const float maxOpen = 1.0f - PulseShape::kMinClosed;
  if (open > maxOpen) {
    for (int i = 0; i + 1 < p.n; ++i) share[i] *= maxOpen / open;
    open = maxOpen;
  }
  share[p.n - 1] = 1.0f - open;
  pulseAt_[0] = 0;
  float mean = 0, prev = 0;
  for (int i = 0; i < p.n; ++i) {
    pulseAt_[i + 1] = pulseAt_[i] + share[i];
    pulseSlope_[i] = (p.height[i] - prev) / std::max(share[i], 1e-4f);
    mean += 0.5f * (prev + p.height[i]) * share[i];
    prev = p.height[i];
  }
  pulseAt_[p.n] = 1.0f;
  pulseMean_ = mean;
}

void PulseLarynx::step(const LarynxControl& c, LarynxOutput& out) {
  // The folds run for everything but the pulse's shape: the airflow, the
  // mouth pressure, the lungs, the onset and offset of the vibration, its
  // weakening under a constriction and the breath noise all come from them,
  // so the stops and fricatives are the same under either source.
  folds_.step(c, out);
  const float dt = c.phaseStep;
  if (out.closed && kShapedLocked) {
    // The drawn pulse is locked to the folds, cycle for cycle: at each
    // closure it restarts, with the period the folds just made, and takes
    // the loudness of the cycle that just ended. If the folds do not close
    // again the pulse finishes its cycle and stays in its closed phase.
    const float period = std::clamp(static_cast<float>(folds_.sinceClose() + 1), fs_ / 500.0f, fs_ / 40.0f);
    sPhase_ = 0;
    // Run a little fast (kShapedPeriodShare of the period just measured):
    // the folds' cycles vary by a few per cent, and a pulse still falling
    // when the next closure restarts it would end in a click.
    sDt_ = 1.0f / (kShapedPeriodShare * period);
    pulseAtPitch(fs_ / period);
  }
  // The shaped pulse: a piecewise-linear flow integrated from its slope,
  // each slope step smoothed with PolyBLEP (band-limited), like the
  // triangle wave. Segment ends, slopes and the mean come from pulseAtPitch.
  const int n = pulse_->n;
  if (!kShapedLocked) {
    // Free-running: the phase wraps at the commanded pitch.
    if (sPhase_ >= 1.0f) {
      sPhase_ -= 1.0f;
      pulseAtPitch(dt * fs_);
    }
    sDt_ = dt;
  }
  const bool running = sPhase_ < 1.0f;
  float slope = 0;
  if (running)
    for (int i = 0; i < n; ++i)
      if (sPhase_ >= pulseAt_[i] && sPhase_ < pulseAt_[i + 1]) slope = pulseSlope_[i];
  // Each slope step is smoothed over the sample before and after it (the
  // one at the restart has no "before": the folds decide when it comes).
  for (int i = 0; i < n; ++i) {
    const float t = sPhase_ - pulseAt_[i];
    if (t >= 0 && t < sDt_)
      slope += 0.5f * (pulseSlope_[i] - pulseSlope_[(i + n - 1) % n]) * polyBlep(t, sDt_);
    else if (t < 0 && t > -sDt_)
      slope += 0.5f * (pulseSlope_[i] - pulseSlope_[(i + n - 1) % n]) * polyBlep(1.0f + t, sDt_);
  }
  tri_ += slope * sDt_;
  tri_ -= tri_ * kTriLeak_;
  sPhase_ = std::min(sPhase_ + sDt_, 1.0f);
  // The folds' loudness, cycle by cycle, with the fast guard (see
  // kShapedFastFloor).
  const float vib = kShapedLocked ? std::min(folds_.cyclePeak(), folds_.fast() / kShapedFastFloor) / kShapedAcFull
                                  : std::min(folds_.loudness() * kShapedEnvToPeak, folds_.fast() / kShapedFastFloor) / kShapedAcFull;
  // The flow plus its rate of change (the lip radiation the other drawn
  // carriers get at the output), applied here at the source so that the
  // output stage is the folds' and the mouth's sounds pass it unchanged.
  out.source = shapedTreble_.process(((tri_ - pulseMean_) + lipGain_ * (tri_ - prevTri_)) * kShapedGain) * vib;
  prevTri_ = tri_;
}

// ---------------------------------------------------------------------------
// WaveLarynx

void WaveLarynx::init(float fs, float baseF0, uint64_t* rng) {
  (void)baseF0;
  (void)rng;
  kTriLeak_ = 2.0f * kPi * 5.0f / fs;  // ~5 Hz: removes drift, keeps the shape
}

const char* WaveLarynx::name() const {
  return kind_ == Carrier::Square ? "square wave" : kind_ == Carrier::Triangle ? "triangle pulse" : "glottal pulse";
}

std::vector<Larynx::Param> WaveLarynx::controls() const {
  return {{"f0Wave", "Hz", 40, 1000, 100}, {"duty", "", 0.1f, 0.9f, 0.5f}, {"vibDrive", "", 0, 1, 1}};
}

void WaveLarynx::step(const LarynxControl& c, LarynxOutput& out) {
  const float phase = c.phase, dt = c.phaseStep, duty = c.duty;
  float g = 0;
  if (kind_ == Carrier::Square) {
    // Band-limited pulse wave: high for the first duty of the period,
    // low for the rest, shifted to zero mean, with PolyBLEP-smoothed edges.
    float fall = phase + 1.0f - duty;
    if (fall >= 1.0f) fall -= 1.0f;
    float sq = (phase < duty ? 1.0f : -1.0f) - (2.0f * duty - 1.0f) +
               polyBlep(phase, dt) - polyBlep(fall, dt);
    g = sq * kSquareGain;
  } else if (kind_ == Carrier::Triangle) {
    // Triangle pulse of glottal flow: rises 0 -> 1 while the folds open
    // (duty of the open phase), falls back to 0 as they close, and stays 0
    // while closed. Built by integrating its slope; each slope step is
    // smoothed with PolyBLEP, which keeps the triangle band-limited.
    const float rise = duty * kTriangleOpenQuotient;
    const float fallLen = kTriangleOpenQuotient - rise;
    float slope = phase < rise ? 1.0f / rise : phase < kTriangleOpenQuotient ? -1.0f / fallLen : 0.0f;
    auto edge = [&](float at, float height) {
      float t = phase - at;
      if (t < 0) t += 1.0f;
      slope += 0.5f * height * polyBlep(t, dt);
    };
    edge(0, 1.0f / rise);
    edge(rise, -1.0f / rise - 1.0f / fallLen);
    edge(kTriangleOpenQuotient, 1.0f / fallLen);
    tri_ += slope * dt;
    tri_ -= tri_ * kTriLeak_;
    g = (tri_ - 0.5f * kTriangleOpenQuotient) * kTriangleGain;  // zero mean
  } else if (phase < kOpenQuotient) {
    // Glottal pulse: derivative of the flow pulse U = x^2 - x^3 during the
    // open phase (x in [0,1)), zero while closed. The sharp drop at closure
    // is the main excitation, as in real voicing.
    float x = phase / kOpenQuotient;
    g = 2.0f * x - 3.0f * x * x;
  }
  out.source = g;
  out.flow = 0;  // the synthesizer estimates the flow from the glottal opening
  // Breath noise swelling and fading smoothly with each glottal cycle.
  out.puff = 0;
  if (phase < kTriangleOpenQuotient) {
    const float w = std::sin(kPi * phase / kTriangleOpenQuotient);
    out.puff = w * w;
  }
  out.airPuff = 0;
  out.closed = false;
}

std::unique_ptr<Larynx> makeLarynx(Carrier c) {
  switch (c) {
    case Carrier::Physical: return std::make_unique<FoldLarynx>(std::make_unique<CoreOf<VocalFolds>>(), "two-mass");
    case Carrier::Grid: return std::make_unique<FoldLarynx>(std::make_unique<CoreOf<GridFolds>>(), "4x8 grid", kGridMicroScale);
    case Carrier::Block: return std::make_unique<FoldLarynx>(std::make_unique<CoreOf<BlockFolds>>(), "2x2x6 block", kBlockMicroScale);
    case Carrier::String: return std::make_unique<FoldLarynx>(std::make_unique<CoreOf<StringFolds>>(), "2x16 string", kBlockMicroScale);
    case Carrier::Shaped: return std::make_unique<PulseLarynx>();
    default: return std::make_unique<WaveLarynx>(c);
  }
}

}  // namespace puhuja
