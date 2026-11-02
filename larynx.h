// The larynx: the voice source as a module, in the spirit of VocalTractLab's
// glottis models. The synthesizer (puhuja.cc) owns the lungs, the mouth
// pressure, the tract and the noise sources; the larynx gets the speaker's
// commands (pitch aimed at, subglottal pressure, abduction, the posterior
// chink, voice-quality settings) and the tract's acoustic pressure above the
// glottis, and gives back the glottal flow, the radiated voice source and
// what the rest of the model needs to know about the folds: how open they
// are, how strongly they vibrate, when they close.
//
// Models:
//   FoldLarynx      a fold model (FoldCore: the Ishizaka-Flanagan two-mass
//                   folds of folds.h, or the 4 x 8 grid of grid.h) with the
//                   speaker's pitch control and the voice's micro-variation
//   PulseLarynx     a drawn (piecewise-linear) flow pulse riding on the
//                   two-mass folds: the folds decide when and how loud, the
//                   drawing decides the spectrum
//   WaveLarynx      the simple waveforms (triangle, square, glottal pulse)
//                   driven by the aerodynamic vibration estimate
// New models (an anatomy-based vocal cord with tens of degrees of freedom,
// contact and friction between the cords) plug in the same way: derive from
// Larynx, fill LarynxOutput from LarynxControl, add a Carrier value and a
// line to makeLarynx.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "filters.h"
#include "block.h"
#include "string.h"
#include "folds.h"
#include "grid.h"

namespace puhuja {

// Glottal openings (cm^2) that the whole model agrees on.
constexpr float kAgVoice = 0.05f;    // glottis adducted, vibrating
constexpr float kAgBreathy = 0.125f;  // half open: breathy voice (voiced h)
constexpr float kAgOpen = 0.3f;      // abducted: voiceless sounds
// Subglottal pressure of normal speech.
constexpr float kLungPressure = 7840.0f;    // dyn/cm^2 (8 cmH2O)
// Shared frequency constants of the source shaping and the lip radiation.
constexpr float kTrebleShelfHz = 5000;  // (2500: the female voice was shrill at 2-4.5 kHz without air above)
constexpr float kLipRadiationHz = 1500;

// Voice source: Physical is the two-mass vocal fold model coupled to the
// tract (folds.h); Shaped a drawn pulse on those folds; the others are fixed
// waveforms shaped by the airflow model's vibration amplitude.
enum class Carrier { Physical, Triangle, Square, Glottal, Shaped, Grid, Block, String };

// A piecewise-linear glottal flow pulse for Carrier::Shaped: n segments, each
// a share of the period at the voice's base pitch, each ending at the given
// flow height (the last at 0, the closed phase). The excitation is the flow's
// slope. With pitch the shares scale as (f0 / base f0)^pitchExp: 0 keeps the
// shape, 1 keeps the segment's length in milliseconds (the fast opening of
// real folds is such a fixed-time event); the last, closed segment takes what
// is left (at least kMinClosed), so the open quotient grows with pitch, as it
// does in voices (H1-H2 rises 6-14 dB per octave of pitch in the references).
struct PulseShape {
  static constexpr int kMaxN = 10;
  static constexpr float kMinClosed = 0.05f;
  int n;
  float share[kMaxN];
  float height[kMaxN];
  float pitchExp[kMaxN];
  float trebleDb;  // a treble shelf on the pulse (it has little above 4 kHz of its own)
};

// The speaker's commands and the surroundings of the larynx, per sample.
struct LarynxControl {
  double t = 0;        // time (samples), for sub-sample closure timing
  float f0 = 100;      // the pitch aimed at (Hz): melody, micro-intonation and the slow
                       // tension that abduction adds; the model's pitch control follows it
  float f0Wave = 100;  // the pitch of a drawn waveform (Hz): f0 with the fast aerodynamic effects
  float phase = 0;     // the commanded pitch phase (0..1) and its step, for the drawn waveforms
  float phaseStep = 0;
  float duty = 0.5f;   // the drawn waveforms' open share of the period
  float ps = 0;        // subglottal pressure minus the mouth pressure (dyn/cm^2), the
                       // lungs' and the phrase's share included; 0 in a pause
  float ag = 0;        // glottal abduction (cm^2): 0 pressed shut, kAgVoice voicing, kAgOpen open
  float irregular = 1;  // 1 + the creak irregularity of the micro-variation
  float gap = 0;       // posterior chink kept open while voicing (cm^2)
  float microGain = 3, cycleJitter = 1, cycleShimmer = 1;  // the voice's variation scales
  float formant[3] = {500, 1500, 2500};  // the tract's F1-F3 (Hz) and bandwidths: its load on the folds
  float bw[3] = {80, 100, 120};
  float vibDrive = 0;  // the aerodynamic vibration estimate (0..1) for the drawn waveforms
};

// What the larynx gives back for this sample.
struct LarynxOutput {
  float flow = 0;     // glottal volume velocity (cm^3/s), what the lungs lose
  float source = 0;   // the radiated voice source, before the voicing level and the tract
  float puff = 0;     // breath-noise modulation by the glottal cycle (0..~1.5)
  float airPuff = 0;  // the same for the high breath noise
  bool closed = false;  // the folds closed within this sample ...
  float closeAt = 0;    // ... at this fraction of it
};

class VocalTractLine;

class Larynx {
 public:
  virtual ~Larynx() = default;
  // Sample rate, the voice's base pitch and the shared noise generator (the
  // models draw their micro-variation from it in a fixed order).
  virtual void init(float fs, float baseF0, uint64_t* rng) = 0;
  virtual void setBaseF0(float f0) { (void)f0; }
  virtual void setPulse(const PulseShape& p) { (void)p; }
  virtual void step(const LarynxControl& c, LarynxOutput& out) = 0;
  // The fold model runs: the aerodynamics, the breath noise and the stops
  // follow the folds instead of the per-phoneme voicing levels.
  virtual bool physical() const { return false; }
  // State the tract and the noise sources read (the last step's).
  virtual float flow() const { return 0; }      // the folds' own flow (cm^3/s)
  virtual float flowMean() const { return 0; }  // ~10 ms mean of the flow heard
  virtual float vib() const { return 0; }       // the flow's fluctuation (cm^3/s)
  virtual float area() const { return 0; }      // glottal area (cm^2), 0 when closed
  virtual float loudness() const { return 0; }  // the folds' radiated sound, ~10 ms envelope
  virtual float cyclePeak() const { return 0; } // its peak in the last cycle
  virtual float pulsePhase() const { return 0; }  // a drawn pulse's phase (trace)
  virtual const char* name() const = 0;
  // A transmission-line tract (line.h): the folds then feed it every substep
  // and take its pressure as the load above them.
  virtual void attachLine(VocalTractLine* line) { (void)line; }
  // The control parameters the model uses, with their ranges (for tools).
  struct Param {
    const char* name;
    const char* unit;
    float min, max, neutral;
  };
  virtual std::vector<Param> controls() const;
};

// A fold model the driver below can run: the two-mass folds (folds.h), the
// 4 x 8 grid (grid.h), the 2 x 2 x 6 block (block.h), or the next one. Each step advances by dt seconds and
// returns the glottal flow heard (cm^3/s).
class FoldCore {
 public:
  virtual ~FoldCore() = default;
  virtual void reset() = 0;
  virtual float step(float dt, const VocalFolds::Input& in) = 0;
  virtual float flow() const = 0;
  virtual float area() const = 0;
  virtual bool closedNow() const = 0;
  virtual float closeFraction() const = 0;
  // The model places the posterior chink itself (gets it in Input::gap
  // instead of added to the rest area).
  virtual bool ownChink() const { return false; }
};

template <class F>
class CoreOf : public FoldCore {
 public:
  void reset() override { f_.reset(); }
  float step(float dt, const VocalFolds::Input& in) override { return f_.step(dt, in); }
  float flow() const override { return f_.flow(); }
  float area() const override { return f_.area(); }
  bool closedNow() const override { return f_.closedNow(); }
  float closeFraction() const override { return f_.closeFraction(); }
  bool ownChink() const override { return F::kOwnChink; }

 private:
  F f_;
};

// A fold model with the speaker's pitch control and the voice's
// micro-variation: the driver of any FoldCore.
class FoldLarynx : public Larynx {
 public:
  // microScale: the voice's micro-variation (tension, rest, pressure) scaled
  // for this core (the grid has structural irregularity of its own).
  FoldLarynx(std::unique_ptr<FoldCore> core, const char* name, float microScale = 1)
      : core_(std::move(core)), name_(name), microScale_(microScale) {}
  void init(float fs, float baseF0, uint64_t* rng) override;
  void step(const LarynxControl& c, LarynxOutput& out) override;
  bool physical() const override { return true; }
  float flow() const override { return core_->flow(); }
  float flowMean() const override { return flowMean_; }
  float vib() const override { return physVib_; }
  float area() const override { return core_->area(); }
  float loudness() const override { return foldAc_; }
  float cyclePeak() const override { return foldCycleAmp_; }
  const char* name() const override { return name_; }
  std::vector<Param> controls() const override;
  // For the drawn pulse riding on these folds.
  void attachLine(VocalTractLine* line) override { line_ = line; }
  float fast() const { return foldFast_; }
  int sinceClose() const { return sinceClose_; }
  float prevFlow() const { return prevFlow_; }

 private:
  float fs_ = 48000;
  uint64_t* rng_ = nullptr;
  float noise() { return nextNoise(*rng_); }
  std::unique_ptr<FoldCore> core_;
  const char* name_;
  float microScale_ = 1;
  Biquad foldAa_[3];  // anti-alias low-pass of the oversampled glottal flow
  float microLp_[3] = {0, 0, 0}, microLp2_[3] = {0, 0, 0}, kMicroSmooth_ = 0.004f;
  float breathyComp_ = 0, kBreathyCompFollow_ = 0;  // fold tension for a held-open glottis
  float cycleJ_ = 0, cycleS_ = 0;  // this cycle's random tension and pressure deviation
  float pitchCorr_ = 1, pitchCorrTarget_ = 1;  // pitch controller: tension correction
  double lastCloseTime_ = 0;  // last fold closure (samples, sub-sample)
  float kPitchRelax_ = 0, kVibTrack_ = 0;
  BandPass tractMode_[3];  // formant modes of the tract input impedance
  float p1_ = 0;           // acoustic pressure above the folds (feedback)
  VocalTractLine* line_ = nullptr;  // the line tract, if attached (p1 comes from it)
  float prevFlow_ = 0;     // glottal flow one sample ago
  float flowMean_ = 0, physVib_ = 0;  // flow mean and fluctuation (~10 ms)
  float foldAc_ = 0;       // the folds' radiated-sound envelope
  Biquad foldAcHp2_;       // the folds' sound above 250 Hz, for its envelope
  float foldFast_ = 0;     // the same, quarter-period envelope
  float foldCycleMax_ = 0, foldCycleAmp_ = 0;  // the folds' peak this cycle / last cycle
  int sinceClose_ = 0;     // samples since the folds last closed
};

// A drawn flow pulse riding on the two-mass folds.
class PulseLarynx : public Larynx {
 public:
  void init(float fs, float baseF0, uint64_t* rng) override;
  void setBaseF0(float f0) override { baseF0_ = f0; }
  void setPulse(const PulseShape& p) override;
  void step(const LarynxControl& c, LarynxOutput& out) override;
  bool physical() const override { return true; }
  float flow() const override { return folds_.flow(); }
  float flowMean() const override { return folds_.flowMean(); }
  float vib() const override { return folds_.vib(); }
  float area() const override { return folds_.area(); }
  float loudness() const override { return folds_.loudness(); }
  float cyclePeak() const override { return folds_.cyclePeak(); }
  float pulsePhase() const override { return sPhase_; }
  const char* name() const override { return "drawn pulse on two-mass folds"; }
  std::vector<Param> controls() const override;

 private:
  void pulseAtPitch(float f0);  // the segment ends and slopes for this cycle
  FoldLarynx folds_{std::make_unique<CoreOf<VocalFolds>>(), "two-mass"};
  float fs_ = 48000, baseF0_ = 100, lipGain_ = 1, kTriLeak_ = 0;
  const PulseShape* pulse_ = nullptr;
  Biquad shapedTreble_;  // the pulse's own treble shelf
  float tri_ = 0, prevTri_ = 0;  // the integrated flow and its previous value
  float sPhase_ = 0, sDt_ = 0;   // the pulse's phase (1 = closed, waiting) and step
  float pulseAt_[PulseShape::kMaxN + 1] = {0}, pulseSlope_[PulseShape::kMaxN] = {0}, pulseMean_ = 0;
};

// The simple waveforms, driven by the aerodynamic vibration estimate.
class WaveLarynx : public Larynx {
 public:
  explicit WaveLarynx(Carrier kind) : kind_(kind) {}
  void init(float fs, float baseF0, uint64_t* rng) override;
  void step(const LarynxControl& c, LarynxOutput& out) override;
  const char* name() const override;
  std::vector<Param> controls() const override;

 private:
  Carrier kind_;
  float tri_ = 0, kTriLeak_ = 0;  // triangle: integrated slope (glottal flow)
};

std::unique_ptr<Larynx> makeLarynx(Carrier c);

}  // namespace puhuja
