#include "larynx.h"

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
constexpr float kFoldNaturalF0 = 155;     // f0 of the fold model at tension 1
constexpr float kPitchControl = 0.3f;     // per cycle: share of pitch error corrected
constexpr float kPitchLearnVib = 100;     // flow fluctuation (cm^3/s) needed to learn
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
constexpr float kCycleJitter = 0.0f, kCycleShimmer = 0.0f;  // per-cycle gaussian deviations (off: they cost HNR)
constexpr float kMicroHz = 8;
constexpr int kFoldOversample = 4;
constexpr float kFoldAntiAliasHz = 18000;

// The grid's share of the micro-variation (Jyrki: less micro-fluctuation
// there; its irregularity comes from its mass inhomogeneity instead).
constexpr float kGridMicroScale = 0.33f;

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
  core_->reset();
  for (int i = 0; i < static_cast<int>(0.05f * fs_); ++i)
    core_->step(1.0f / fs_, {kLungPressure, 0, 0.7f, kFoldRestPressed});
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
  in.q = target / kFoldNaturalF0 * pitchCorr_;
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
  }
  const float irregular = c.irregular;
  in.q *= 1.0f + kTensionJitter * irregular * micro[0];
  in.ps *= 1.0f + kPressureShimmer * micro[2];
  in.q *= 1.0f + kCycleJitter * c.cycleJitter * cycleJ_;
  in.ps *= 1.0f + kCycleShimmer * c.cycleShimmer * cycleS_;
  // Fold rest position from the glottal opening (see kFoldRestPerAg).
  const float restVoice = kFoldRestPerAg * kAgVoice + kFoldRestOffset;
  in.rest = ag >= kAgVoice ? kFoldRestPerAg * ag + kFoldRestOffset
                           : kFoldRestPressed + (restVoice - kFoldRestPressed) * ag / kAgVoice;
  // Its micro-variation: added after the rest position is set (before
  // it, the assignment overwrote it, and the jitter never acted).
  in.rest += kRestJitter * irregular * micro[1];
  // A posterior gap (female voices): softer closure, stronger fundamental.
  // The two-mass folds take it in their rest area; a model with segments
  // along the cord places it at the posterior end itself.
  if (core_->ownChink()) in.gap = c.gap * smoothstep(0.0f, kAgVoice, ag);
  else in.rest += c.gap * smoothstep(0.0f, kAgVoice, ag);
  // The folds run kFoldOversample times faster than the output, and their
  // flow is low-passed before it is taken once per sample: the flow drops
  // to zero within a sample at each closure, and sampled directly that
  // corner aliases (inharmonic hiss 13-20 dB under the harmonics above
  // 8 kHz).
  float u = 0;
  bool closed = false;
  double closeAt = 0;  // closure time within this sample (0..1)
  for (int k = 0; k < kFoldOversample; ++k) {
    const float raw = core_->step(1.0f / (kFoldOversample * fs_), in);
    u = foldAa_[2].process(foldAa_[1].process(foldAa_[0].process(raw)));
    if (core_->closedNow()) {
      closed = true;
      closeAt = (k + core_->closeFraction()) / kFoldOversample;
    }
  }
  out.flow = u;
  if (physVib_ < kPitchLearnVib) pitchCorrTarget_ += (1.0f - pitchCorrTarget_) * kPitchRelax_;
  if (closed) {
    // Period from closure to closure, timed to a small fraction of a
    // sample (the zero crossing of the opening between the folds'
    // substeps; whole samples were only good to ~3 cents at 92 Hz).
    const double closeTime = c.t + closeAt;
    const double period = closeTime - lastCloseTime_;
    // The next cycle's deviations (a rough gaussian from two draws).
    cycleJ_ = std::clamp(0.87f * (noise() + noise()), -2.5f, 2.5f);
    cycleS_ = std::clamp(0.87f * (noise() + noise()), -2.5f, 2.5f);
    lastCloseTime_ = closeTime;
    const float measured = static_cast<float>(fs_ / std::max(period, 1.0));
    // Learn only from steady, strong vibration: during onsets and offsets
    // the closures are irregular and would detune the folds.
    if (physVib_ > kPitchLearnVib && measured > 0.7f * target && measured < 1.4f * target)
      pitchCorrTarget_ = std::clamp(
          pitchCorrTarget_ * static_cast<float>(std::pow(static_cast<double>(target) / measured, kPitchControl)),
          0.7f, 1.4f);
  }
  // The correction glides over about a cycle instead of jumping at each
  // closure.
  pitchCorr_ += (pitchCorrTarget_ - pitchCorr_) * std::min(1.0f, target / fs_);
  // Tract feedback: each formant mode turns the flow into pressure.
  const float nyq = 0.45f * fs_;
  p1_ = 0;
  for (int k = 0; k < 3; ++k) {
    tractMode_[k].set(std::clamp(c.formant[k], 100.0f, nyq), std::max(c.bw[k], 20.0f), fs_);
    p1_ += kTractCoupling[k] * tractMode_[k].process(u);
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
    case Carrier::Shaped: return std::make_unique<PulseLarynx>();
    default: return std::make_unique<WaveLarynx>(c);
  }
}

}  // namespace puhuja
