// puhuja: a small streaming formant speech synthesizer.
//
// Source-filter model with an aerodynamic source:
//
//   lungs --> glottis (area Ag) --> mouth cavity (pressure Po) --> constriction
//                                                   |             (area Ac)
//                                                   +--> nasal port (area An)
//
// Each phoneme sets targets for the three areas, which move smoothly like
// articulators. The mouth pressure follows from the airflow in through the
// glottis and out through the mouth and nose, and drives everything else:
//
//   voicing: triangle wave, its strength set by the pressure drop across a
//            nearly closed glottis; pitch rises with that drop. The
//            vibration builds up over a few cycles whenever it starts.
//   breath noise at the glottis (h, aspiration): from the glottal opening
//   noise at the constriction (fricatives, bursts): from mouth pressure and
//            the constriction opening, shaped per place of articulation
//   release transient: the change in airflow out of the mouth
//
// So at p t k the voice stops because pressure builds behind the closure,
// the burst is the trapped air escaping, and voicing returns once the
// pressure has fallen and the glottis has closed again.
//
//   voicing + breath noise --> nasal pole/zero --> F5..F1 --> radiation --+--> out
//   constriction noise (labials: into the tract) -------------------------+
//
// The mixed output goes through a 250 Hz high-pass (-6 dB/octave).
//
// Slow parameters: each phoneme places keyframes (parameter targets) on a
// timeline. Between keyframes the formant frequencies, bandwidths, nasal
// zero, voicing amplitude, trill depth and pitch follow Catmull-Rom splines,
// passing exactly through each target. Resonator coefficients are recomputed
// every sample.
//
// Input: Finnish spelling or IPA, as UTF-8 (see pushText). The stream holds
// one phoneme back (to merge "kk", "aa", "ng" and to know which vowel a
// plosive releases into), and a spline segment needs one keyframe of
// lookahead. Call finish() at end of input.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

#include "folds.h"

namespace puhuja {

enum class Ph : uint8_t {
  // Vowels: Finnish (and shared), then the English-only ones
  // ɪ ɛ ʊ ɔ ʌ ɑ ɒ ɜ.
  A, E, I, O, U, Y, AE, OE, Schwa,
  IH, EH, UH, AO, AH, AA, OH, ER,
  // Plosives, then the affricates ʧ ʤ (a plosive released into a fricative).
  P, T, K, B, D, G, CH, JH,
  // Nasals.
  M, N, NG,
  // Liquids and glides (RR: English approximant ɹ; R is the trilled r).
  L, R, J, V, RR, W,
  // Fricatives (Z, TH, DH, VV: English z θ ð v; V above is the Finnish
  // approximant ʋ).
  S, SH, ZH, F, Z, TH, DH, VV, H,
  // Silence.
  Stop,
  Count
};

// Second-order resonator (Klatt 1980 form), unity gain at DC.
class Resonator {
 public:
  void set(float freq, float bw, float fs);
  float process(float x) {
    double y = a_ * x + b_ * y1_ + c_ * y2_;
    y2_ = y1_;
    y1_ = y;
    return static_cast<float>(y);
  }

 private:
  friend class AntiResonator;
  // Double: the cascade's resonators have large gains that cancel each
  // other, and float round-off there is heard as hiss.
  double a_ = 1, b_ = 0, c_ = 0, y1_ = 0, y2_ = 0;
};

// Second-order zero (inverse of Resonator), for the nasal anti-formant.
class AntiResonator {
 public:
  void set(float freq, float bw, float fs);
  float process(float x) {
    double y = a_ * x + b_ * x1_ + c_ * x2_;
    x2_ = x1_;
    x1_ = x;
    return static_cast<float>(y);
  }

 private:
  double a_ = 1, b_ = 0, c_ = 0, x1_ = 0, x2_ = 0;
};

// Band-pass biquad with 0 dB peak gain (RBJ cookbook), for noise sources.
class BandPass {
 public:
  void set(float freq, float bw, float fs);
  void reset() { x1_ = x2_ = y1_ = y2_ = 0; }
  float process(float x) {
    double y = b0_ * x + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;
    x2_ = x1_;
    x1_ = x;
    y2_ = y1_;
    y1_ = y;
    return static_cast<float>(y);
  }

 private:
  double b0_ = 0, b2_ = 0, a1_ = 0, a2_ = 0;
  double x1_ = 0, x2_ = 0, y1_ = 0, y2_ = 0;
};

// General biquad (RBJ cookbook) with Butterworth Q.
class Biquad {
 public:
  void lowPass(float freq, float fs, float q = 0.7071f);
  void highPass(float freq, float fs);
  void highPass1(float freq, float fs);  // first order: -6 dB/octave
  void highShelf(float freq, float gainDb, float fs);  // RBJ, slope 1
  void peaking(float freq, float bw, float gainDb, float fs);  // RBJ bell
  void reset() { x1_ = x2_ = y1_ = y2_ = 0; }
  float process(float x) {
    double y = b0_ * x + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;
    x2_ = x1_;
    x1_ = x;
    y2_ = y1_;
    y1_ = y;
    return y;
  }

 private:
  double b0_ = 1, b1_ = 0, b2_ = 0, a1_ = 0, a2_ = 0;
  double x1_ = 0, x2_ = 0, y1_ = 0, y2_ = 0;
};

// Spectral shape of a noise source: a band from lo to hi with steep
// (24 dB/octave) edges, plus a resonance from the cavity in front of the
// constriction (peakGain = how far it rises above the band).
struct FricShape {
  float lo, hi;          // band edges (Hz)
  float peakF, peakBw;   // front-cavity resonance (Hz)
  float peakGain;        // 0 = flat band
};

class FricationFilter {
 public:
  void set(const FricShape& s, float fs);
  // Clears the filter memory. Needed when a filter that has been idle is
  // reconfigured: old state under new coefficients can ring up a large spike.
  void reset() {
    hp1_.reset();
    hp2_.reset();
    lp1_.reset();
    lp2_.reset();
    peak_.reset();
  }
  float process(float x) {
    x = hp1_.process(x);
    x = hp2_.process(x);
    x = lp1_.process(x);
    x = lp2_.process(x);
    return x + peakGain_ * peak_.process(x);
  }

 private:
  Biquad hp1_, hp2_, lp1_, lp2_;
  BandPass peak_;
  float peakGain_ = 0;
};

// An articulator: a value that moves to each new target along an S-curve
// (raised cosine) of a given length, like a tongue or lips moving.
class Gesture {
 public:
  void set(float v) {
    v_ = v;
    left_ = 0;
    vel_ = m0_ = 0;
  }
  void start(float to, float samples) {
    if (left_ >= 1.0f && to == to_) return;  // already moving there
    from_ = v_;
    to_ = to;
    len_ = left_ = samples;
    // A restart in mid-movement keeps the current speed (it decays over the
    // new movement): a jump in speed is a step in the airflow's change,
    // heard as a soft thump through the release transient.
    m0_ = vel_ * samples;
    if (samples < 1) set(to);
  }
  float tick();
  float value() const { return v_; }

 private:
  float v_ = 0, from_ = 0, to_ = 0, len_ = 1, left_ = 0;
  float vel_ = 0, m0_ = 0;  // speed per sample; initial slope of a movement
};

// Indices into the parameter vector that the splines interpolate.
enum Param {
  kF1, kF2, kF3,  // formant frequencies (Hz)
  kB1, kB2, kB3,  // formant bandwidths (Hz)
  kFNZ,           // nasal zero (Hz); equal to the nasal pole when oral
  kAV,            // voicing amplitude
  kTrill,         // depth of the r trill amplitude modulation
  kF0Mul,         // pitch relative to the base pitch
  kNumParams
};

struct Params {
  float v[kNumParams] = {500, 1500, 2500, 80, 100, 150, 270, 1, 0, 1};
  float& operator[](int i) { return v[i]; }
  float operator[](int i) const { return v[i]; }
};

struct Keyframe {
  Params p;
  double t;  // time in samples
  bool vowel = false;  // a vowel's target
  bool edge = false;   // a held target's edge: not a formant point (see splineAt)
  float f0Mod = 1;     // micro-intonation on top of the pitch line
};

// Articulatory event: starts movements of the airway openings at time t,
// and/or changes the constriction noise.
struct Event {
  double t;  // time in samples
  // Area targets (cm^2), each with its movement time (ms). < 0: no change.
  float ac = -1, ag = -1, an = -1;
  float acMs = 0, agMs = 0, anMs = 0;
  // Constriction noise: spectrum, strength, and the share of its energy
  // that goes through the vocal tract (the cavity behind the constriction
  // resonating); the rest is added in parallel (the front cavity).
  bool setNoise = false;
  FricShape shape{};
  float noiseGain = 0;
  float noiseTract = 0;
  float noiseTractGain = 1;  // gain of the tract part
  float noisePost = 1;  // level after the turbulence ceiling (English per-phoneme level)
  // Strength of the release transient (pop) from now on; < 0: no change.
  float pop = -1;
  // Relative lung pressure (phrase endings ease off), reached over pressMs;
  // < 0: no change.
  float press = -1, pressMs = 0;
  float aspGain = -1;  // breath-noise strength from now on (< 0: unchanged)
  bool aspBurst = false;  // an aspirated release: the puff's noise starts
  float creak = -1;       // creaky voice from now on (0..1; < 0: unchanged)
  float noiseFadeMs = -1;  // with setNoise: fade-in from silence (< 0: the default)
  // Cavity compliance multiplier (active expansion during voiced closures),
  // reached over expandMs; < 0: no change.
  float expand = -1, expandMs = 0;
  // Breathe in over this many ms (lungs refill, inhaling noise); 0: no.
  float inhaleMs = 0;
  bool inhaleSnatched = false;  // the quick between-words breath (see breathPoint)
  // Speech level target: 0 at a pause (fades out exponentially), 1 when
  // speaking; < 0: no change.
  float speech = -1;
  float speechAttackMs = -1;  // with speech: its rise time (< 0: the default)
};

// Voice source: Physical is the two-mass vocal fold model coupled to the
// tract (folds.h); the others are fixed waveforms shaped by the airflow
// model's vibration amplitude.
enum class Carrier { Physical, Triangle, Square, Glottal };

// How the formants move between the phonemes' targets: a Catmull-Rom spline
// through the keyframes (each target reached on time), or a critically
// damped follower (a spring and damper chasing the current target, as
// articulators do: short sounds undershoot, nothing overshoots).
enum class FormantMotion { Spline, Damped };

// Speaker presets. Male is calibrated to a Finnish male reader (CSS10);
// female to the female speakers of FLEURS Finnish.
enum class VoicePreset { Male, Female };

struct Voice {
  float f0;              // base pitch (Hz)
  float formant[3];      // scale of F1, F2, F3 (shorter vocal tract: higher)
  float highFormants;    // scale of F4, F5 and the higher tract resonances
  float rate;            // duration multiplier (< 1: faster)
  float doubledVowel;    // length of "aa" relative to "a"
  float breathiness;     // multiplier of the breath noise in voicing
  float gain;            // output level
  float treble;          // output treble shelf above 2.5 kHz (dB)
  float air;             // multiplier of the high breath noise (above ~4 kHz)
  float bw[3];           // scale of the F1-F3 bandwidths
  float noiseScale = 1;  // frequency scale of the fricative noise shapes (shorter front cavity)
  float shGain = 1;      // ʃ ʒ noise multiplier
  float glottalGap = 0;  // posterior glottal gap kept open while voicing (cm^2)
  float f4 = 3500, f5 = 4500;  // F4, F5 (Hz) at a neutral F3 and the base pitch
};

class Synth {
 public:
  explicit Synth(float sampleRate = 48000, float baseF0 = 0);  // 0: the voice preset's pitch

  void setCarrier(Carrier c) { carrier_ = c; }
  void setFormantMotion(FormantMotion m) { motion_ = m; }

  // Select a speaker preset (pitch, formant scaling, tempo, breathiness).
  // Call before pushing text; setBaseF0 afterwards overrides the pitch.
  void setVoice(VoicePreset preset);
  void setBaseF0(float f0) { baseF0_ = f0; }
  float baseF0() const { return baseF0_; }

  // Part of the source that bypasses the formant cascade, relative to the
  // cascade's low-frequency gain of 1. 0 is a pure cascade; 0.03 keeps the
  // highs about 35 dB down instead of vanishing.
  void setLeak(float leak) { leak_ = leak; }

  // Queue one phoneme. wordStart marks the first phoneme of a word
  // (used for the Finnish-like raised first syllable). Two equal phonemes
  // in a row ("kk", "aa") become one long phoneme: a doubled vowel is 1.7x
  // a single one (female voice 1.45x), a doubled consonant 1.4x (plosives:
  // longer closure); ː makes a sound 1.55x.
  void push(Ph ph, bool wordStart = false);

  // Finnish spelling merges in push(): "ng" -> long ŋ, "nk" -> ŋk, "np" ->
  // mp, "tʃ"/"dʒ" -> affricates. On by default; the English reader turns
  // them off since its phonemes are already exact.
  void setSpellingRules(bool on) { spellingRules_ = on; }

  // Intonation: pitch of the next pushed phoneme (if a vowel), in semitones
  // relative to the base pitch. Overrides the built-in stress / first-syllable
  // pitch for that vowel; the English reader sets every vowel this way.
  // The shared table's formant k (0-2) of a phoneme.
  static float tableFormant(Ph ph, int k);
  void setPitch(float semitones) { nextPitch_ = semitones; }
  // A second pitch for the next vowel, reached at 3/4 of it (a fall
  // inside a final accented word).
  void setPitchEnd(float semitones) { nextPitchEnd_ = semitones; }
  static constexpr float kNoPitch = -1000.0f;

  // Adjust the next pushed phoneme only (language-specific variants of the
  // shared table): formant targets (0 = keep), duration and level factors.
  struct Shape {
    float f1 = 0, f2 = 0, f3 = 0;
    float length = 1;  // duration multiplier
    float level = 1;   // voicing (and frication noise) multiplier
    float ms = 0;      // absolute duration (plosives: closure), 0 = table
    float press = 1;   // vowels: glottal opening multiplier (< 1 pressed, > 1 lax)
    bool flap = false;  // a d said as a tap (English t/d between vowels)
  };
  void shapeNext(const Shape& s) { nextShape_ = s; }

  // Level of fricative noise (not plosive bursts) relative to the
  // per-phoneme levels, which are calibrated to Piper's Finnish voice.
  // The English reader uses 4.5 (+13 dB), measured on an English voice-over.
  void setFricationGain(float g) { fricationGain_ = g; }
  // Dark (velarised) l, as in English: the tongue back holds its own
  // position, so l follows the neighbouring vowels much less.
  void setDarkL(bool on) { darkL_ = on; }
  // English: voiceless plosives before a vowel or approximant are
  // aspirated (the glottis closes slowly after the release), except after s.
  void setAspiration(bool on) { aspirate_ = on; }
  // h between voiced sounds: extra breath noise (x the h gain).
  void setVoicedHNoise(float g, float ag = 0.125f) { voicedHNoise_ = g; voicedHAg_ = ag; }
  // h takes the next vowel's formants early (the vowel whispered).
  void setHTakesVowel(bool on) { hTakesVowel_ = on; }
  // English: a consonant's formants are ranges, not targets. Each formant
  // runs on the line from the previous vowel's to the next vowel's value
  // and is only kept inside the consonant's range (see kFormantRanges), so
  // the formants do not swing to a consonant target and back.
  void setFormantRanges(bool on) { formantRanges_ = on; }
  // Strength of the open-glottis breath noise, x default: in aspiration and
  // the other sounds, and in h.
  void setAspirationGain(float g, float h) { aspirationGain_ = g; hAspirationGain_ = h; }

  // Make the most recently pushed phoneme long (IPA "ː").
  void lengthen();

  // Queue UTF-8 text; may be split anywhere, even inside a character.
  //   Finnish letters: a e i o u y ä ö å p t k b d g m n l r s š h j v f
  //                    (also c q w x z), "ng" = long ŋ, "nk" = ŋk.
  //   IPA: æ ø œ ɑ ɛ ɔ ɪ ʊ ə ɐ ɜ ɚ ʌ ɒ ɨ ɯ ŋ ɡ ʃ ʒ ʧ ʤ ʋ ɾ ʀ, ː or ':' for
  //   length. "tʃ" and "dʒ" are merged into the affricates ʧ and ʤ. The
  //   English vowels ɪ ɛ ʊ ɔ ʌ ɑ ɒ ɜ have their own targets.
  //   Stress: ˈ (primary) and ˌ (secondary) raise the pitch of the syllable
  //   that follows and lengthen its vowels a little; ˌ also inserts an
  //   extra 60 ms pause before it.
  //   Once any stress mark has been seen, unmarked syllables are no longer
  //   raised.
  //   ' ' and newline: 15 ms pause. ',' ';': 195 ms (and a breath when the
  //   air runs low). '.' '!' '?': 150 ms and a breath (180-600 ms).
  //   '-': word boundary without a pause.
  void pushText(std::string_view text);

  // A place where the speaker may breathe in: before a sentence always
  // (sentence = true), at a comma only when the lungs are running low. The
  // inhalation (longer the more air was used) is added as a pause with a soft
  // breathing-in noise. The first sentence gets one automatically.
  // snatched: between words, with no good place for it (a quick, sharper breath).
  void breathPoint(bool sentence, bool paragraph = false, bool snatched = false);
  float plannedAir() const { return planVolume_; }  // lung volume planned so far (cm^3)
  // The next vowel starts with a glottal onset (folds pressed, then released).
  void glottalOnset() { glottalOnsetNext_ = true; }
  // The next vowel ends in creaky voice (a low statement ending).
  void creakyEnd() { creakNext_ = true; }

  // End of input: lets the last phonemes be rendered.
  void finish();

  // Render up to n mono float32 samples. Returns how many were written;
  // fewer than n means more input (or finish()) is needed.
  size_t pull(float* out, size_t n);

  float sampleRate() const { return fs_; }

  // Phoneme timing of the rendered audio: start and end in seconds from the
  // first output sample, and the IPA symbol ("_" for a pause, "ː" appended
  // when long). Complete after finish() and pulling all audio.
  struct Label {
    double start, end;
    std::string symbol;
  };
  std::vector<Label> labels() const;

 private:
  struct Token {
    Ph ph;
    bool isLong;
    bool wordStart;
    uint8_t stress;  // stress mark before this phoneme: 0 none, 1 ˌ, 2 ˈ
    bool doubled = false;  // long because the same vowel was written twice
    float pitch = kNoPitch;  // semitones from the base pitch (vowels; set by setPitch)
    Shape shape;             // per-phoneme adjustment (shapeNext)
    float pitchEnd = kNoPitch;  // a vowel's second pitch, reached at 3/4 of it (setPitchEnd)
  };

  void pushCodepoint(uint32_t cp);
  void emit(const Token& tok, const Ph* next);
  void emitPlosive(const Token& tok, const Ph* next);
  void coarticulate(Params& p, Ph ph, const Ph* next) const;
  float nextVowelF(Ph next, int k) const;
  void emitAffricate(const Token& tok, const Ph* next);
  void addKeyframe(const Params& p, double t, bool vowel = false);
  void addHeld(const Params& p, double t0, double dur, double a, double b, bool vowel = false);
  void addEvent(const Event& e);
  void addEventAt(const Event& e);
  bool startSegment();
  float splineAt(int param, double t) const;
  void resolveF1(float target, double tTarget);
  void addLabel(const Token& tok);
  float noise();
  float tick();

  void setFixedResonances();

  float fs_;
  float baseF0_;
  Voice voice_;

  // Keyframes: k_[0] is the one before the current segment, k_[1]..k_[2]
  // span the current segment, k_[3] is the one after.
  std::deque<Keyframe> k_;
  // Pitch line: the last vowel centre's time (samples, -1: none) and pitch.
  double pitchAnchorT_ = -1;
  float pitchAnchor_ = 1;
  std::deque<Event> events_;  // sorted by time
  double timeline_ = 0;       // where the next phoneme starts (samples)

  // Current segment as cubic polynomials in s in [0,1):
  // p(s) = c0 + c1 s + c2 s^2 + c3 s^3, per parameter.
  float c_[4][kNumParams] = {};
  bool inSegment_ = false;
  double segStart_ = 0, segLen_ = 1, t_ = 0;
  bool finished_ = false;
  double outputStart_ = -1;  // timeline position of the first output sample
  std::vector<std::pair<double, std::string>> labelStarts_;  // (samples, symbol)

  // Input state.
  uint32_t utf8Cp_ = 0;
  int utf8Need_ = 0;
  bool atWordStart_ = true;
  bool hasPending_ = false;
  Token pending_{};
  uint8_t stressMark_ = 0;  // stress mark waiting for the next phoneme
  float nextPitch_ = kNoPitch;  // setPitch() value waiting for the next phoneme
  float nextPitchEnd_ = kNoPitch;  // setPitchEnd() value
  Shape nextShape_;   // shapeNext() value waiting for the next phoneme
  Shape nextShape2_;  // the shape of the sound after the one being emitted
  bool nextWordStart_ = false;  // the sound after the one being emitted starts a word

  // Prosody / context state.
  int syllable_ = 0;
  bool prevWasVowel_ = false;
  bool prevWasPlosive_ = false;
  bool prevWasPause_ = false;  // the last emitted phoneme was a pause
  float lastVowelF_[3] = {0, 0, 0};  // previous vowel's own targets (for sharpening)
  // Lungs: air left for speaking (cm^3), as planned while phonemes are
  // queued (for breathing decisions) and as simulated while rendering.
  float planVolume_, lungVolume_;
  bool hasSpoken_ = false;     // the first sentence's breath has been taken
  bool spokeSinceBreath_ = false;
  float nextInhaleMs_ = 0;     // inhalation for the next emitted pause
  bool nextInhaleSnatched_ = false, inhaleSnatched_ = false;
  float inhaleLeft_ = 0, inhaleLen_ = 0, inhaleRate_ = 0;  // samples, cm^3/sample
  FricationFilter inhaleFilter_;  // airy breath-in noise (bypasses the tract)
  FricationFilter inhaleSnatchFilter_;  // the quick breath's brighter noise
  float speechEnv_ = 1, speechTarget_ = 1;  // fade-out at pauses
  float speechAttackNow_ = 0;  // rise rate in use (per sample)
  float aspBurst_ = 1, kAspBurstDecay_ = 1;  // p t k puff: x aspirationGain_, back to 1
  float kSpeechDecay_ = 0, kSpeechAttack_ = 0;
  bool prevVoiced_ = false;
  bool inAffricate_ = false;  // emitting the frication part of ʧ / ʤ
  double phonemeStart_ = 0;   // where the previous phoneme started (samples)
  // Once the input uses stress marks, they decide the raised syllables
  // instead of the Finnish first-syllable rule.
  bool explicitStress_ = false;
  uint8_t curStress_ = 0;  // stress waiting for the next syllable's vowel
  uint8_t sylStress_ = 0;  // stress of the current syllable
  Params last_;  // last keyframe target

  // Aerodynamics.
  Gesture ac_, ag_, an_;  // constriction, glottis, nasal port areas (cm^2)
  Gesture press_;         // relative lung pressure (phrase endings)
  Gesture expand_;        // cavity compliance multiplier (1 = rigid walls)
  float po_ = 0;          // mouth pressure relative to the lungs (0..1)
  float flowGain_;        // pressure change per sample per unit flow
  float prevOralFlow_ = 0;
  Biquad foldAa_[3];  // anti-alias low-pass of the oversampled glottal flow
  float microLp_[3] = {0, 0, 0}, kMicroSmooth_ = 0.004f;  // fold micro-variation
  float microLp2_[3] = {0, 0, 0};                         // ... its second stage
  float popScale_ = 1, popTarget_ = 1;
  double pauseStartT_ = -1;  // time of the last pause not yet followed by a pitched vowel
  Ph lastPh_ = Ph::Stop;  // the previous phoneme (micro-intonation)
  bool glottalOnsetNext_ = false;
  bool creakNext_ = false;
  bool hasPendingRelease_ = false, hasPendingCreak_ = false;
  Event pendingCreak_{0};
  Event pendingRelease_{0};
  float kPopSmooth_ = 0.01f;  // one-pole speed of popScale_ (set from fs: ~2 ms)
  float tension_ = 0;     // slow trace of glottal abduction (raises pitch)
  float kTension_;
  float vibAmp_ = 0;      // current vocal fold vibration amplitude (0..1)
  VocalFolds folds_;      // physical source
  BandPass tractMode_[3];  // formant modes of the tract input impedance
  float p1_ = 0;          // acoustic pressure above the folds (feedback)
  float prevFlow_ = 0;    // glottal flow one sample ago
  Biquad physTilt_;       // treble tilt of the physical source
  Biquad physUpperCut_;   // cut around F3 (physical source)
  Biquad physMidLift_;    // lift between F1 and F2 (physical source)
  float physTiltGain_ = 0;
  float flowMean_ = 0, physVib_ = 0;  // flow mean and fluctuation (~10 ms)
  float kVibTrack_ = 0;
  float kPitchRelax_ = 0;  // pitch correction eases back to 1 when not voicing
  float pitchCorr_ = 1;   // pitch controller: tension correction (smoothed)
  float pitchCorrTarget_ = 1;  // its per-cycle estimate
  double lastCloseTime_ = 0;   // last fold closure (samples, sub-sample)
  float kVibGrowth_, kVibDecay_;

  // Constriction noise: two filter slots, crossfaded when the shape changes.
  FricationFilter nf_;
  // Constriction noise: one filter whose shape and gains glide from the
  // previous sound's to the next one's (see tick).
  struct NoiseState {
    FricShape shape{1000, 8000, 3000, 1000, 0};
    float gain = 0, tract = 0, tractGain = 1, post = 1;
  };
  NoiseState nFrom_, nTo_, nNow_;
  float nGlide_ = 1, nGlideStep_ = 0;  // progress 0..1 and its step per sample
  float nLevel_ = 0, nPeak_ = 0;       // recent turbulence strength and its peak
  float kNoiseLevelFollow_ = 0, kNoisePeakDecay_ = 1;
  float trillPhase_ = 0;
  float trillAmp_ = 0;    // current trill depth (follows the airflow)
  float kTrillDecay_ = 0, kTrillAttack_ = 0;

  // Voicing source state.
  Carrier carrier_ = Carrier::Physical;
  FormantMotion motion_ = FormantMotion::Damped;
  float formPos_[3] = {0, 0, 0}, formVel_[3] = {0, 0, 0};
  bool formInit_ = false;
  float tri_ = 0;      // triangle: integrated slope (glottal flow)
  float kTriLeak_;     // keeps the integrator from drifting
  float leak_ = 0.03f;
  float phase_ = 0;
  bool spellingRules_ = true;
  float fricationGain_ = 1;
  bool darkL_ = false;
  bool aspirate_ = false;
  float voicedHNoise_ = 1, voicedHAg_ = 0.125f;  // h between voiced sounds: noise, glottis
  bool hTakesVowel_ = false;
  bool formantRanges_ = false;
  const Token* rangeTok_ = nullptr;  // the consonant being emitted, if its formants are ranges
  struct RangePending {
    double t;
    Ph ph;
    float centre[3];  // its tuned (coarticulated) F1-F3
    bool f1Line;      // its F1 is drawn by resolveF1 (m n ŋ, an initial ɹ)
  };
  std::vector<RangePending> rangePending_;  // consonant keyframes waiting for the next vowel
  float rangePrevF_[3] = {0, 0, 0};  // previous vowel's formants (as keyframed) ...
  double rangePrevT_ = -1;          // ... and the end of its held part (< 0: none)
  void resolveRanges(const float* nextF, double tNext);
  double hKeyframeT_ = -1;  // time of the h keyframe the next vowel redraws
  float aspirationGain_ = 1, hAspirationGain_ = 1;
  float aspGainTarget_ = 1, aspGainNow_ = 1;  // smoothed toward the target
  float kAspGainFollow_ = 0;
  double kLookAheadSamples_ = 0;  // how far ahead the targets are read
  float breathyComp_ = 0, kBreathyCompFollow_ = 0;  // fold tension for a held-open glottis
  float creakTarget_ = 0, creakNow_ = 0, kCreakFollow_ = 0;  // creaky irregularity
  bool aspirating_ = false;  // the last plosive's release is closing the glottis
  Event pendingNoise_{0};  // a voiced fricative's noise, after the glottis closed
  bool hasPendingNoise_ = false;
  bool prevUnreleased_ = false;
  // F1 through l r d: their keyframes waiting for the next vowel, and the
  // last vowel's F1 and time.
  std::vector<std::pair<double, float>> f1Pending_;  // time, least share of the next vowel's F1
  float f1MinShare_ = 0;
  bool f1Transparent_ = false;
  float f1Prev_ = -1;
  double f1PrevT_ = -1;  // the last plosive ran its closure into this one
  float jitter_ = 1;
  float drift_ = 0;   // slow pitch drift
  float pitch1_ = 0, pitch2_ = 0;  // smoothed pitch multiplier (two stages)
  float kPitchLag_ = 0;
  float duty_ = 0.5f;  // carrier duty for the current cycle
  uint64_t rng_ = 0x123456789abcdef1ull;

  Biquad outHp_;  // output high-pass (kOutputHighPassHz, -6 dB/octave)
  Biquad lowCut_;  // and a 2nd-order cut at kLowCutHz
  float nasLp_ = 0, kNasLp_ = 0;  // the nasal murmur's low cut
  float roomLp1_ = 0, roomLp2_ = 0, kRoomLp_ = 0;  // room-tone floor
  Biquad shelf_;  // treble shelf on the voice
  Biquad trebleShelf_;  // per-voice output treble (Voice::treble)
  Biquad polesHigh_[3];  // higher-pole correction (physical source)
  Biquad headShelf_;    // head diffraction (physical source)
  Biquad chestLp_;      // the voice heard straight from the chest (kChestDirect)
  Biquad midCut_;  // low-mid cut on the voice
  Biquad presence_;  // presence lift on the voice
  Biquad breathLp_;  // keeps breath noise out of the highs
  Biquad hbHp1_, hbHp2_;  // the high breath noise's high-pass
  float aspLp_ = 0, kAspLp_ = 1;  // aspiration source tilt (one pole)
  float aspHpState_ = 0, kAspHp_ = 0;  // and its low cut (one pole)
  Biquad airHp1_, airHp2_, airLp_;  // high breath noise: ~4-8 kHz
  BandPass body_[3];  // body (subglottal) resonances of voiced sound
  float shelfGain_ = 0;
  AntiResonator nz_;
  Resonator np_, r1_, r2_, r3_, r4_, r5_;
  Resonator tube_[17];     // the tract's resonances above F5 (see puhuja.cc)
  Resonator cavPole_[10];  // nose / sinus cavities: resonances...
  AntiResonator cavZero_[10];  // ...and their anti-resonances
  float prevTract_ = 0, lipGain_ = 1;  // lip radiation differentiator
};

}  // namespace puhuja
