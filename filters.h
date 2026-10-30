// Filters, articulator gestures and small helpers shared by the synthesizer
// (puhuja.cc) and the larynx models (larynx.cc).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace puhuja {

constexpr float kPi = 3.14159265358979f;

inline float smoothstep(float a, float b, float x) {
  float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

inline float msToSamples(float ms, float fs) { return ms * 0.001f * fs; }

// PolyBLEP correction: smooths a unit step at phase 0 over one sample on
// each side, which removes most of the aliasing of a naive square wave.
inline float polyBlep(float t, float dt) {
  if (t < dt) {
    t /= dt;
    return t + t - t * t - 1.0f;
  }
  if (t > 1.0f - dt) {
    t = (t - 1.0f) / dt;
    return t * t + t + t + 1.0f;
  }
  return 0.0f;
}

// 64-bit noise: a multiply by a large odd constant (carries the low bits
// up), then xorshifts to the right (carry the high bits back down). Each
// step is a bijection, so the state never reaches 0. The top 24 bits are
// used. Uniform in [-1, 1).
inline float nextNoise(uint64_t& rng) {
  rng *= 0x9e3779b97f4a7c15ull;
  rng ^= rng >> 32;
  rng ^= rng >> 29;
  return static_cast<float>(rng >> 40) * (1.0f / 8388608.0f) - 1.0f;
}

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

}  // namespace puhuja
