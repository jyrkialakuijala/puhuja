#include "puhuja.h"
#include "params.h"

#include <cstdio>
#include <cstdlib>

#if defined(__SSE__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

#include <algorithm>
#include <cmath>

namespace puhuja {

namespace {


// Finnish-like prosody: the first syllable of a word is a bit higher.
constexpr float kFirstSyllableF0 = 1.15f;
// Explicit stress marks: pitch raise and vowel lengthening.
constexpr float kPrimaryStressF0 = 1.15f;
constexpr float kSecondaryStressF0 = 1.07f;
constexpr float kPrimaryStressLength = 1.25f;
constexpr float kSecondaryStressLength = 1.1f;
// Extra pause that ˌ inserts before its syllable, in 15 ms stops.
constexpr int kSecondaryStressPauseStops = 4;

// (The waveforms' own constants: larynx.cc.)
// Duty cycle of the square wave. At 0.5 it has only odd harmonics.
constexpr float kSquareDuty = 0.5f;
// Duty of the triangle: like real vocal folds it rises (opens) slowly and
// falls (closes) quickly, which gives it the even harmonics a symmetric
// triangle lacks while keeping the -12 dB/octave slope. 0.85 matches the
// CSS10 reader's harmonics (H2..H11 within ~5 dB). Every duty weakens some
// harmonic (here the 7th; simple fractions like 2/3 cancel whole series).
// Letting the duty wander from cycle to cycle (jitter > 0) fills the gaps
// between harmonics with noise and sounds raspy: at 0.04 the harmonics stood
// only 13 dB above it below 1 kHz (25 dB without; the CSS10 reader 18 dB).
constexpr float kTriangleDuty = 0.85f;
constexpr float kTriangleDutyJitter = 0.0f;
// Shaped glottal pulse (Jyrki's loop, 2026-10-30): "before the vocal cords
// separate they move a bit, a small slope before they open; then it rises
// fast because the cords opened fast; closes at moderate speed, perhaps a
// short slow tail; then flat". The fast rise sets how steeply the spectrum
// falls (Jyrki): at 0.13 of the period the 1-2 kHz band was 10 dB under
// Kokoro am_onyx's, at 0.015 (0.16 ms at 92 Hz) the bands up to 4 kHz match
// it within 1 dB and the valley contrast is its 28 dB. One shape per voice.
// Jyrki's loop as first set by hand:
//   {5, {0.12, 0.015, 0.25, 0.10, 0.515}, {0.15, 1.0, 0.12, 0, 0}, {0.5, 1.0, 0.5, 0.5, 0}, 6}
// The shapes below come from tools/shape_opt.py (2026-10-30), searched
// separately per voice against a reference reading of the Kennedy speech on
// the voiced frames' 1/3-octave spectrum, valley contrast, crest factor,
// H1-H2 and their change with pitch. Male: Kokoro am_onyx (distance 6.05;
// crest 8.2 vs 6.8, the physical folds 11). Female: ElevenLabs reading 2
// (distance 7.4; crest, contrast and H1-H2 all within 0.1 dB of it).
constexpr PulseShape kMalePulse = {6, {0.1505f, 0.1818f, 0.1569f, 0.1913f, 0.0212f, 0.2983f},
                                   {0.352f, 0.143f, 0.831f, 1.0f, 0.752f, 0.0f},
                                   {0.81f, 0.46f, 1.44f, 1.06f, 0.36f, 0.0f}, 9.5f};
constexpr PulseShape kFemalePulse = {6, {0.1195f, 0.0876f, 0.2684f, 0.1897f, 0.1167f, 0.2180f},
                                     {0.276f, 1.0f, 0.925f, 0.445f, 0.388f, 0.0f},
                                     {1.23f, 0.79f, 1.45f, 0.52f, 1.05f, 0.0f}, 12.3f};
// (The pulse's gains and its coupling to the folds: larynx.cc.)
// Treble shelf on the voice source (not on noise): with the -12 dB/octave
// triangle, the voice otherwise lacks the energy a real voice has above
// ~1 kHz. Set against both Piper and the CSS10 reader's long-term spectrum.
constexpr float kShelfHz = 1000;
constexpr float kShelfDb = 10;
// Gentle cut of the voice source around F1: the closed phase makes the
// 500-1000 Hz band louder than in a real voice.
constexpr float kMidCutHz = 700;
constexpr float kMidCutBw = 600;
constexpr float kMidCutDb = -5;
// Presence lift of the voice source around 4.5 kHz, like the "ring" of a
// trained voice: the highs come from clean harmonics instead of hiss.
constexpr float kPresenceHz = 4500;
constexpr float kPresenceBw = 3000;
constexpr float kPresenceDb = 7;

// Overall output level: speech peaks around 0.13-0.17 at the default pitch,
// leaving headroom for higher pitches with the male voice (about 0.45 at
// f0 = 160-200 Hz, 0.6 at 250 Hz; the female preset turns itself down).
const float kMasterGain = param("kMasterGain", 0.028f);

// Lip radiation (see tick()).
constexpr bool kLipRadiation = true;
// Head diffraction (physical source): high shelf.
constexpr float kHeadShelfHz = 1500, kHeadShelfDb = 5;
// Higher-pole correction: three high shelves (Hz, dB).
constexpr float kPolesHigh[3][2] = {{3500, 10}, {6000, 10}, {10000, 8}};

// Higher vocal tract resonances. A tube has a resonance about every 1 kHz
// all the way up. In the cascade every formant resonator keeps cutting
// 12 dB/octave above its frequency; in a real tract the resonances above
// cancel that, so a cascade that stops at F5 dies away above ~4.5 kHz
// ("higher pole correction", Fant / Klatt). So the series continues to near
// the top of the band. A real tract is not a uniform tube: its higher
// resonances are unevenly spaced and of different widths, and a perfectly
// regular series sounds mechanical, so they are spread irregularly here.
// Broad bandwidths keep the response smooth. Being in the cascade, they
// follow the vowel's colour.
struct TubePole {
  float freq, bw;
};
constexpr TubePole kTubePoles[] = {
    {5350, 900}, {6750, 1550}, {7400, 1400}, {8800, 2200}, {9250, 1650},
    {10650, 2350}, {11450, 2300}, {12700, 3050}, {13200, 2250}, {14600, 3050},
    {15750, 3600}, {16300, 2950}, {17550, 4400}, {18350, 3500}, {19800, 4350},
    {20400, 4100}, {21500, 5150}};
constexpr int kNumTubePoles = sizeof(kTubePoles) / sizeof(kTubePoles[0]);

// Nose and sinus cavities: in every voiced sound a little of it couples into
// the nasal cavity and the sinuses beside it. Each cavity adds a weak
// resonance with an anti-resonance close by (a pole/zero pair), giving the
// small irregular bumps and dips of a living voice. Weak: pole and zero are
// near each other, so each cavity only changes the spectrum by a few dB.
struct CavityPair {
  float pole, zero, bw;
};
// Many spaces of different sizes: the nasal passages and the maxillary,
// frontal, ethmoid and sphenoid sinuses, mostly on both sides.
constexpr CavityPair kCavities[] = {
    {520, 555, 120},   {780, 825, 140},   {1050, 1100, 150}, {1430, 1385, 180},
    {1900, 1955, 200}, {2620, 2550, 220}, {3150, 3080, 250}, {3700, 3800, 300},
    {4300, 4390, 350}, {5100, 4980, 400}};
constexpr int kNumCavities = sizeof(kCavities) / sizeof(kCavities[0]);
// On again since the final EQ (item 19 of ref/kennedy/variants/PLAN.md):
// with the source EQ refitted the voice-over's vowel bands match within
// ~1 dB at 2-4 kHz, and without them the formant structure is flatter
// than in real speech (tools/artifacts.py contrast).
constexpr bool kUseCavities = true;
// Depth of the cavity bumps: each zero sits this share of the way from its
// pole to its full-strength place (1 = full, 0 = no effect).
constexpr float kCavityDepth = 0.5f;
constexpr bool kUseHighPoles = true;

// Breathiness: turbulent air passing the vibrating glottis, in bursts with
// each cycle (louder while the folds are open). It goes through the vocal
// tract like the voicing, so it takes the vowel's colour. Scaled by the
// vibration, so it is absent in p t k and fricatives.
constexpr float kLowCutHz = 90;
constexpr float kRoomTone = 0.038f, kRoomToneHz = 1500;  // room-tone floor: level, darkness  // 2nd-order output cut (male boom)
constexpr float kBreathiness = 0.08f;
// High breath noise in voicing (see the glottal mix): gain, corner, pulsed share.
// (2.0: 4-8 kHz 2 dB above goog.wav and the voice-overs, between-harmonic
// noise 2 dB over goog's.)
constexpr float kHighBreathGain = 1.2f, kHighBreathHz = 5000, kHighBreathPulse = 0.6f;
// Breath noise comes in puffs while the folds are open (none while shut)
// and is kept below kBreathLowPassHz, so that the voice's highs are clean
// harmonics instead of hiss.
constexpr float kBreathLowPassHz = 3000;
// Soft ceiling of constriction noise strength (gain x turbulence).
constexpr float kNoiseCeiling = 4.0f;
// Noise pulsing reaches its full depth at this vibration strength (cm^3/s).
constexpr float kNoisePulseVib = 150;
// Pitch variation: a slow random drift (follows a new random target each
// cycle with weight kPitchDriftFollow) plus a little cycle-to-cycle jitter.
// Per-cycle jitter shifts harmonic k by k times as much, smearing the highs
// into noise; a slow drift keeps every harmonic a clean line.
constexpr float kPitchDrift = 0.012f;  // see kMicro2Gain
constexpr float kPitchDriftFollow = 0.15f;
// The pitch follows its per-syllable targets through a smooth two-stage lag,
// so it glides continuously instead of holding still within each vowel.
constexpr float kPitchLagMs = 25;  // (40: peaks landed 54 ms late, long vowels slid upward; fixes/07)
// The pitch target is read this far ahead (the two lag stages delay the
// melody by ~80 ms).
constexpr float kPitchLookAheadMs = 40;  // (50: peaks 17 ms early)
constexpr float kPitchJitter = 0.0f;
constexpr float kBreathPuffGain = 1.26f;  // same breath power below the low-pass as before

// Body resonance of voiced sound: below the vibrating glottis the windpipe
// and chest resonate too (subglottal resonances, ~600 / 1400 / 2300 Hz for a
// male voice). Only voicing reaches them: the input is the voiced part,
// gated by the fold vibration, so bursts, hiss and breath noise of p t k s h
// do not. It partly follows the colour of the sound: the input blends the
// raw voicing with the vocal tract output (kBodyFollow = share of the
// latter). Broad, linear band-passes; no sharp peaks.
struct BodyResonance {
  float freq, bw, gain;
};
// Gains lowered (fixes/03): they filled the valleys between the formants and
// made an edgy peak near 2.5 kHz (was 0.75, 1.8, 2.2).
constexpr BodyResonance kBodyResonances[] = {{600, 400, 0.1f}, {1400, 600, 0.36f}, {2300, 800, 0.3f}};  // (review3: they filled the valleys, contrast)
constexpr int kNumBodyResonances = sizeof(kBodyResonances) / sizeof(kBodyResonances[0]);
// 0.75 and the lower gains: more of the body path is driven by the voice
// source itself (the subglottal system is excited by the glottal flow, not
// by the mouth). Further, the formants flatten below the range of real
// speech (tools/artifacts.py contrast).
constexpr float kBodyFollow = 0.75f;

// Fixed output gains so vowels peak at roughly +-0.5. No clipping is applied.
constexpr float kOutGain = 0.25f;
const float kAspirationGain = param("kAspirationGain", 0.3659f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // glottal breath noise into the tract
// Aspiration source: a band of one-pole slopes (-6 dB/octave above the tilt
// corner and below the low cut), and the gain that keeps its level around
// 1-2 kHz where the white source had it.
constexpr float kAspirationTiltHz = 1000;
// and below this the turbulence is weak too (it drove F1 too hard: goog.wav's
// h is strongest at its vowel's F2, not F1).
constexpr float kAspirationLowCutHz = 600;
const float kAspirationTiltGain = param("kAspirationTiltGain", 1.919f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;
// The h's own breath-noise gain, times the language's h gain
// (setAspirationGain). It is deliberately not kAspirationGain x
// kAspirationTiltGain: the stops' search (stops-w1k-us5-mf-3) doubled that
// product for the p t k puff and every h came along, +6 dB; "hands" in
// kennedy.txt was then 4 dB above its vowel where readers are 3-13 dB under
// or voiced throughout (INPUT.md 2026-10-31). 0.35 is the product before
// that search.
const float kHGain = param("kHGain", 0.35f);
constexpr float kPopGain = 120.0f;        // release transient into the tract

// --- The line tract (Tract::Line) ------------------------------------------
// The line tract's vowel shapes (tract.h TractShape), one per vowel phoneme
// (Ph index): the tongue constriction's place as a share of the length, its
// area (cm^2) and half-width (cm), the lip area and length, the pharynx, mouth
// and larynx-tube areas, and the tract length (cm); then the formants (Hz) the
// line gave for the shape when it was fitted, which place the shape in
// lineShapeFor's blend. First fitted to the English formant targets on the
// line (make linefit && ./linefit, ref/line/vowel_tract.txt), then each
// searched on the 857-set words that contain the vowel with the smeared
// Zimtohrli (word_opt.py --set vowel<ph>; Jyrki 2026-11-02: "search the
// shapes"). The search writes its result here. y, ø and ɒ have no shape (no
// English targets); diphthongs and consonant loci are blends of these.
static const struct {
  int ph;
  float xc, ac, wc, aLip, aPharynx, aOral, lLip, aLarynx, length;
  float f1, f2, f3;
} kLineVowels[] = {
    {0, param("LINE_V0_XC", 0.5980f), param("LINE_V0_AC", 2.7494f), param("LINE_V0_WC", 5.6828f), param("LINE_V0_ALIP", 3.7767f), param("LINE_V0_APH", 1.6187f), param("LINE_V0_AOR", 5.8295f), param("LINE_V0_LLIP", 0.3597f), param("LINE_V0_ALAR", 0.7200f), param("LINE_V0_LEN", 17.5051f), 710.0f, 1510.0f, 2397.0f},  // a
    {1, param("LINE_V1_XC", 0.9500f), param("LINE_V1_AC", 3.4609f), param("LINE_V1_WC", 4.1552f), param("LINE_V1_ALIP", 4.9275f), param("LINE_V1_APH", 4.6285f), param("LINE_V1_AOR", 1.3659f), param("LINE_V1_LLIP", 2.1214f), param("LINE_V1_ALAR", 0.5601f), param("LINE_V1_LEN", 17.4931f), 449.0f, 1982.0f, 2530.0f},  // e
    {2, param("LINE_V2_XC", 0.7101f), param("LINE_V2_AC", 0.8010f), param("LINE_V2_WC", 5.1921f), param("LINE_V2_ALIP", 4.3338f), param("LINE_V2_APH", 9.9445f), param("LINE_V2_AOR", 2.5965f), param("LINE_V2_LLIP", 1.9178f), param("LINE_V2_ALAR", 0.3000f), param("LINE_V2_LEN", 17.4513f), 290.0f, 2301.0f, 2720.0f},  // i
    {3, param("LINE_V3_XC", 0.7174f), param("LINE_V3_AC", 1.4332f), param("LINE_V3_WC", 3.7126f), param("LINE_V3_ALIP", 2.4300f), param("LINE_V3_APH", 7.5480f), param("LINE_V3_AOR", 2.4277f), param("LINE_V3_LLIP", 1.0280f), param("LINE_V3_ALAR", 1.2129f), param("LINE_V3_LEN", 17.5958f), 355.0f, 1111.0f, 1923.0f},  // o
    {4, param("LINE_V4_XC", 0.5127f), param("LINE_V4_AC", 0.6356f), param("LINE_V4_WC", 3.6064f), param("LINE_V4_ALIP", 2.5240f), param("LINE_V4_APH", 6.8763f), param("LINE_V4_AOR", 3.1570f), param("LINE_V4_LLIP", 1.0322f), param("LINE_V4_ALAR", 0.7670f), param("LINE_V4_LEN", 17.1611f), 391.0f, 1395.0f, 2229.0f},  // u
    {6, param("LINE_V6_XC", 0.6700f), param("LINE_V6_AC", 1.4674f), param("LINE_V6_WC", 4.0161f), param("LINE_V6_ALIP", 5.3100f), param("LINE_V6_APH", 1.7226f), param("LINE_V6_AOR", 5.2040f), param("LINE_V6_LLIP", 1.0084f), param("LINE_V6_ALAR", 0.3261f), param("LINE_V6_LEN", 17.4609f), 675.0f, 1729.0f, 2466.0f},  // æ
    {8, param("LINE_V8_XC", 0.7060f), param("LINE_V8_AC", 2.0636f), param("LINE_V8_WC", 2.9885f), param("LINE_V8_ALIP", 1.9251f), param("LINE_V8_APH", 7.2924f), param("LINE_V8_AOR", 3.4373f), param("LINE_V8_LLIP", 1.0725f), param("LINE_V8_ALAR", 0.8776f), param("LINE_V8_LEN", 17.3168f), 401.0f, 1764.0f, 2440.0f},  // ə
    {9, param("LINE_V9_XC", 0.6787f), param("LINE_V9_AC", 1.2918f), param("LINE_V9_WC", 3.7543f), param("LINE_V9_ALIP", 3.3058f), param("LINE_V9_APH", 5.7075f), param("LINE_V9_AOR", 2.0436f), param("LINE_V9_LLIP", 0.6712f), param("LINE_V9_ALAR", 0.8472f), param("LINE_V9_LEN", 17.4781f), 391.0f, 1936.0f, 2536.0f},  // ɪ
    {10, param("LINE_V10_XC", 0.6836f), param("LINE_V10_AC", 1.6646f), param("LINE_V10_WC", 3.1008f), param("LINE_V10_ALIP", 3.2115f), param("LINE_V10_APH", 3.0208f), param("LINE_V10_AOR", 3.3872f), param("LINE_V10_LLIP", 0.8379f), param("LINE_V10_ALAR", 0.7210f), param("LINE_V10_LEN", 17.4857f), 544.0f, 1713.0f, 2449.0f},  // ɛ
    {11, param("LINE_V11_XC", 0.5106f), param("LINE_V11_AC", 1.5459f), param("LINE_V11_WC", 3.4623f), param("LINE_V11_ALIP", 2.4409f), param("LINE_V11_APH", 3.7919f), param("LINE_V11_AOR", 3.6185f), param("LINE_V11_LLIP", 1.0889f), param("LINE_V11_ALAR", 1.0739f), param("LINE_V11_LEN", 17.5692f), 525.0f, 1393.0f, 2706.0f},  // ʊ
    {12, param("LINE_V12_XC", 0.8847f), param("LINE_V12_AC", 2.3152f), param("LINE_V12_WC", 2.8176f), param("LINE_V12_ALIP", 3.7496f), param("LINE_V12_APH", 4.8983f), param("LINE_V12_AOR", 1.7043f), param("LINE_V12_LLIP", 1.0914f), param("LINE_V12_ALAR", 1.1582f), param("LINE_V12_LEN", 17.1433f), 445.0f, 1123.0f, 1887.0f},  // ɔ
    {13, param("LINE_V13_XC", 0.5048f), param("LINE_V13_AC", 1.3557f), param("LINE_V13_WC", 3.3916f), param("LINE_V13_ALIP", 3.4257f), param("LINE_V13_APH", 5.1333f), param("LINE_V13_AOR", 3.5296f), param("LINE_V13_LLIP", 0.9395f), param("LINE_V13_ALAR", 1.0201f), param("LINE_V13_LEN", 17.3766f), 491.0f, 1451.0f, 2142.0f},  // ʌ
    {14, param("LINE_V14_XC", 0.5206f), param("LINE_V14_AC", 2.1668f), param("LINE_V14_WC", 3.0848f), param("LINE_V14_ALIP", 3.0285f), param("LINE_V14_APH", 2.9957f), param("LINE_V14_AOR", 6.6021f), param("LINE_V14_LLIP", 0.4428f), param("LINE_V14_ALAR", 1.1014f), param("LINE_V14_LEN", 17.6478f), 624.0f, 1305.0f, 2565.0f},  // ɑ
    {16, param("LINE_V16_XC", 0.6191f), param("LINE_V16_AC", 1.1420f), param("LINE_V16_WC", 2.1593f), param("LINE_V16_ALIP", 2.6026f), param("LINE_V16_APH", 6.2070f), param("LINE_V16_AOR", 5.6483f), param("LINE_V16_LLIP", 1.3305f), param("LINE_V16_ALAR", 1.2830f), param("LINE_V16_LEN", 17.6893f), 429.0f, 1378.0f, 1743.0f},  // ɜ
};
// The line's output level, and its turbulence gains (noise pressure per unit
// of Reynolds number past critical, at the constriction and at the glottis).
const float kLineGain = param("LINE_GAIN", 1.438e-05f) /* cost 0.0304788830321763 line3-w15-sm2-s1-mf */;  // vowels ~-20 dB rms, kennedy.txt peaks under 1.0
const float kLineFric = param("LINE_FRIC", 11.18f) /* cost 0.0328298499863392 n13-stringline-w30-a */;  // dyn/cm^2 per unit of (Re^2 - Re_c^2) / Re_c^2 at the jet: s at the cascade's level
const float kLineAsp = param("LINE_ASP", 1.753f) /* cost 0.0328298499863392 n13-stringline-w30-a */;  // the same at the glottis (h between vowels is voiced and needs none)
const float kLineShapeSigma = param("LINE_SIGMA", 0.08f);  // the formant-space width of a vowel shape (log units); 0.35 (a search result) blurred every vowel into one
// The string folds under the line's load pass ~1.7x the air of the planner's
// table (kennedy.txt: 449 cm^3/s of speech against 257 in the cascade,
// 2026-11-01); the breath planner plans with this until the folds' flow is
// calibrated.
const float kLineAirScale = param("LINE_AIR", 1.1f);  // (1.75 before the discharge coefficient)
// The vowel shape for formant values: the fitted shapes weighted by their
// distance in log-formant space (a soft lookup, so the shape moves smoothly
// along the formant path; consonants' loci land between the vowels).
TractShape lineShapeFor(float f1, float f2, float f3) {
  const int n = sizeof(kLineVowels) / sizeof(kLineVowels[0]);
  const float l1 = std::log(std::max(f1, 100.0f)), l2 = std::log(std::max(f2, 300.0f)), l3 = std::log(std::max(f3, 1000.0f));
  float w[32], sum = 0, best = 1e9f;
  for (int i = 0; i < n; ++i) {
    const auto& v = kLineVowels[i];
    const float d1 = l1 - std::log(v.f1), d2 = l2 - std::log(v.f2), d3 = 0.5f * (l3 - std::log(v.f3));
    const float d2s = d1 * d1 + d2 * d2 + d3 * d3;
    w[i] = d2s;
    best = std::min(best, d2s);
  }
  const float inv = 1.0f / (2.0f * kLineShapeSigma * kLineShapeSigma);
  for (int i = 0; i < n; ++i) { w[i] = std::exp(-(w[i] - best) * inv); sum += w[i]; }
  TractShape s;
  float xc = 0, lac = 0, wc = 0, lLip = 0, lPh = 0, lOr = 0, ll = 0, lLar = 0, len = 0;
  for (int i = 0; i < n; ++i) {
    const auto& v = kLineVowels[i];
    const float g = w[i] / sum;
    xc += g * v.xc; lac += g * std::log(v.ac); wc += g * v.wc; lLip += g * std::log(v.aLip);
    lPh += g * std::log(v.aPharynx); lOr += g * std::log(v.aOral); ll += g * v.lLip; lLar += g * std::log(v.aLarynx); len += g * v.length;
  }
  s.xc = xc; s.ac = std::exp(lac); s.wc = wc; s.aLip = std::exp(lLip); s.aPharynx = std::exp(lPh); s.aOral = std::exp(lOr);
  s.lLip = ll; s.aLarynx = std::exp(lLar); s.length = len;
  return s;
}
// Where a consonant constricts the line (share of the tract's length).
float linePlaceOf(Ph ph) {
  switch (ph) {
    case Ph::P: case Ph::B: case Ph::M: case Ph::W: return 0.97f;
    case Ph::F: case Ph::VV: case Ph::V: return 0.95f;
    case Ph::TH: case Ph::DH: return 0.90f;
    case Ph::S: case Ph::Z: return 0.90f;  // the front cavity to the teeth ~1.7 cm: the hiss near 5 kHz
    case Ph::T: case Ph::D: case Ph::N: case Ph::L: case Ph::R: return 0.87f;
    case Ph::SH: case Ph::ZH: case Ph::CH: case Ph::JH: return 0.78f;
    case Ph::RR: return 0.75f;
    case Ph::J: return 0.70f;
    case Ph::K: case Ph::G: case Ph::NG: return 0.62f;
    default: return -1.0f;
  }
}

// --- Aerodynamics ---------------------------------------------------------
// Areas in cm^2. Pressures are relative to the lung pressure (8 cmH2O).
constexpr float kAcOpen = 2.0f;      // open vocal tract (vowels)
constexpr float kAnNasal = 0.25f;    // open velum (nasals)
// Lung pressure and the compliance of the mouth cavity (air plus yielding
// walls) set how fast pressure builds behind a closure: ~30 ms with a
// vibrating glottis, a few ms with an open one.
constexpr float kAirDensity = 0.00114f;     // g/cm^3
constexpr float kCavityCompliance = 7e-4f;  // cm^5/dyn
// The vocal folds vibrate when the pressure drop across them exceeds this
// fraction of the lung pressure (~2 cmH2O).
constexpr float kPhonationThreshold = 0.25f;
// Physical source (Carrier::Physical; the fold driving itself: larynx.cc).
constexpr bool kPhysicalShaping = false;  // the triangle's treble shaping
constexpr float kGlottalB1 = 150;         // F1 widening (Hz) at 0.1 cm^2 glottal opening
constexpr float kGlottalB23[2] = {0.6f, 0.4f};  // F2, F3 widening relative to F1's
constexpr float kPhysTiltHz = 1500;       // gentle treble cut of the physical source
constexpr float kPhysTiltDb = -2;
constexpr bool kPhysMidCut = false;  // the low-mid cut around F1 (kMidCut*); off: it was tuned around the clipped flow (fixes/01)
// Cut of the physical source around F3 (all three references: Piper, the
// CSS10 reader and the voice-over, have 6-10 dB less at 2-3 kHz).
// Source lift between F1 and F2: the vowels and l w r were 3-5 dB weaker
// than goog's at 0.9-1.3 kHz (both voices), a gap under F2 in the sonograms.
constexpr float kPhysMidLiftHz = 1000, kPhysMidLiftBw = 500, kPhysMidLiftDb = 4;
constexpr float kPhysUpperCutHz = 3000, kPhysUpperCutBw = 1500, kPhysUpperCutDb = -1.5;  // (2.5-3.5 kHz was 3-5 dB hot; -7 at 2600 was tuned around the clipped flow)
// High breath (off): an added hiss above ~4 kHz sounded bad either way:
// pulsed fully by the glottal cycles it buzzed, steady it sounded like
// quantization noise (tools/artifacts.py: hissbuzz above / below the range
// of real speech). The notes below are from when it was tried:
// above ~4 kHz a real voice is mostly breath noise, not
// harmonics (the voice-over's vowel middles: nearly flat 4-10 kHz at about
// -24 dB from the 0.5-1 kHz level, harmonics-to-noise ~0 dB there). Puffed
// with the glottal cycle and gated by the vibration like the breath below
// 3 kHz; added after the vocal tract (above F5 it is nearly flat anyway).
constexpr float kAirHz = 3500, kAirTopHz = 8000;
constexpr float kAirGain = 0.0f;
constexpr float kPhysBreath = 1.0f;
constexpr float kNoisePulse = 0.8f;       // share of fricative noise pulsed by the glottal flow       // breath noise level (physical source)
// Mouth pressure (relative to the lungs) needed for audible turbulence.
constexpr float kTurbulenceOnset = 0.06f;
// Vocal fold vibration does not start at full swing: from a small seed it
// grows over a few cycles toward what the airflow allows (about
// kOnsetMs to full), and dies down within kOffsetMs when it is no longer
// sustained.
const float kOnsetMs = param("kOnsetMs", 50.64) /* cost 0.0390638945257385 stops-c32-mf */;
// Sound radiates from the lips and nostrils; with both closed only a little
// gets out through the cheeks and throat (the "voice bar" of b d g).
constexpr float kWallRadiation = 0.4f;  // (0.3: b d g closures 10-15 dB under am_onyx's and ElevenLabs'; +2.5 dB, 2026-10-30)
// Active cavity expansion in voiced closures (compliance multiplier).
constexpr float kVoicedExpansion = 8.0f;
constexpr float kVoicedExpansionReleaseMs = 30;
// Burst and pop of a glottalised plosive (before ð θ), relative to normal.
constexpr float kGlottalisedRelease = 0.15f;
// Share of burst energy through the tract, by place (lips, tongue tip,
// back, postalveolar): the rest comes straight from the front cavity.
constexpr float kBurstTractShare[4] = {0.05f, 0.2f, 0.25f, 0.2f};
// b d g: a little more through the tract (the cavity behind is set ringing
// by the voicing too).
constexpr float kVoicedBurstTractExtra = 0.03f;
// Gain for the tract part of a burst, by place, so that the shares above are
// shares of output energy (measured: through the nearly closed tract the
// same noise comes out 22 dB weaker for p, ~10 dB for t and k).
constexpr float kBurstTractNorm[4] = {12.7f, 3.43f, 3.02f, 3.43f};
// Fricatives: a small share of the constriction noise excites the cavity
// behind it too, so it takes the colour of the current (coarticulated)
// vowel shape. The voice-over's f θ ð follow the neighbouring vowels' F2
// and F3 clearly; through the parallel path alone puhuja's did not at all.
// Share of energy and the place (index into kBurstTractNorm).
struct FricTract {
  float share;
  int place;
};
FricTract fricTract(Ph ph) {
  switch (ph) {
    case Ph::F: return {0.15f, 0};
    case Ph::TH: case Ph::DH: return {0.1f, 1};
    case Ph::S: case Ph::Z: return {0.1f, 1};  // (0.05: the vowels' colour did not reach the hiss)
    case Ph::SH: case Ph::ZH: return {0.1f, 3};
    default: return {0, 1};
  }
}
constexpr float kGlottalisedClosure = 0.7f;           // closure length factor
constexpr float kGlottalisedAg = 0.4f * kAgVoice;     // pressed: creaky voice
// The radiated level drops off as the opening (cm^2) closes below this; the
// nostrils count kNostrilFactor times the velum opening.
constexpr float kRadiationOpenArea = 0.15f;
constexpr float kNostrilFactor = 4.0f;
constexpr float kOffsetMs = 10;
constexpr float kOnsetSeed = 0.03f;
// Pitch: rises with the pressure drop across the glottis (lower in voiced
// constrictions), and a little after voiceless sounds (fold tension).
constexpr float kF0Pressure = 0.15f;  // see kMicro2Gain
constexpr float kF0Tension = 0.05f;
// Fold tension added when the glottis is held open (see the fold input).
constexpr float kTensionMs = 40;
// Articulator movement times (ms).
constexpr float kBoundaryMs = 20;    // between continuant sounds
constexpr float kGlottalMs = 15;     // glottis opening/closing
// h before a vowel: the vowel's formants are reached this far into the h.
// Formant targets are points (one per phoneme, at its centre) on a
// continuous Catmull-Rom path, not held plateaus (see addHeld).
constexpr bool kFormantPointTargets = true;
constexpr float kHVowelAt = 0.3f;
constexpr float kVoicedHVoice = 0.5f;  // voicing amplitude of an h between voiced sounds, on top of its level
// A vowel falling to a second pitch (a statement's last accented vowel):
// its pitch is anchored this far into it and the fall ends there.
constexpr double kFallAnchorAt = 0.25, kFallEndAt = 0.9;
// h before j ("human", "huge"): a palatal fricative [ç], frication at a
// narrow palatal constriction (cm^2) rather than breath at the glottis.
// Its noise peaks around 4 kHz (goog.wav "human": 3-6 kHz within 2 dB of
// each other, 2-3 kHz 5-8 dB under; as glottal breath through the j shape
// it peaked at 2-3 kHz and was 7 dB too loud, -4 dB against the vowel where
// goog's is -11 to -15).
constexpr FricShape kPalatalHShape = {2500, 8000, 4000, 3000, 2.0f};
constexpr float kPalatalHAc = 0.15f;
constexpr float kPalatalHLevel = 0.4f;  // on top of the h's level, after the turbulence ceiling
// English u after j ("human", "few", "use"): fronted, most of the way to the
// j's F2 (goog.wav: F2 ~2.5 kHz through the u of "human", falling only in the
// m; with a back u the bright hj did not belong to the dark "uman").
constexpr float kUAfterJ = 0.8f;
// Closing into a plosive: a fast approach to a nearly closed opening, then
// a slower final seal, as lips and tongue decelerate on contact. The sound
// fades out during the seal.
constexpr float kApproachMs = 12;
constexpr float kSealMs = 18;
constexpr float kSealArea = 0.15f;   // cm^2: where the slow seal begins
constexpr float kStopGlottalMs = 280;  // glottis closing for a pause (under the fade)
constexpr float kPreposeMs = 40;     // after a pause, articulators move this early
// Damped formant motion: time constant per formant, by the articulator that
// mostly moves it. F1 follows the jaw (the largest mass: slowest), F2 the
// tongue body, F3 the tongue tip and the lips (light: fastest).
constexpr float kFormantTauMs[3] = {16, 12, 9};
// With a planned path (FormantMotion::Planned) the follower only smooths the
// plan's grid: a short constant.
constexpr float kPlanTauMs = 3;
// Damped motion: the follower aims at the smooth spline through the
// keyframes, read this many time constants ahead (a critically damped
// follower lags ~2 tau). The target glides every sample; aiming at each
// keyframe's value in turn moved the formants in steps.
constexpr float kFormantLookAhead = 2.0f;
// Read 2 tau ahead, the critically damped follower returns the path plus
// tau^2 times its curvature: every vowel peak came out lowered and flat
// (F1 peak curvature half a real speaker's), and only a held plateau let
// it reach the target. The target is pre-emphasised by the opposite term,
// so the follower reaches the path's peaks (1: exact to second order).
constexpr float kFormantPreemphasis = 1.0f;
constexpr float kVowelSharpen = 0.05f;
// English vowels next to w, dark l, r: share of the consonant's F2 per side.
constexpr float kVowelContextPull = 0.15f;
constexpr float kSchwaContext = 0.4f;  // English ə: share of its neighbours' mean F2
constexpr float kPullF2W = 750, kPullF2DarkL = 1000, kPullF2R = 1300;
// A word-initial r: least share of the next vowel's F1 (the jaw opens for the
// vowel already; "rievolution" had F1 rising through the r), and of its F2
// (copying F2 too made r words harder: round trip r set 15.8% -> 21%).
constexpr float kRInitialF1 = 0.8f, kRInitialF2 = 0.0f;
// ...and after a vowel ("the rights"): share of that vowel's F1 it keeps.
constexpr float kRAfterVowelF1 = 0.9f;
// Nasals on the F1 line between the vowels: dipped by this many octaves at
// their keyframes. Real m n ŋ sit 0.2-0.45 octave under the vowels around
// them (thirteen speakers), but 0.3 cost the male 1.4 points on the 200-
// statement round trip ("won" -> "in", "miles" -> "models"); off for now.
constexpr float kNasalF1DipOctaves = 0.0f;
const float kNasalF1Dip = std::exp2(-kNasalF1DipOctaves);
// r takes this share of the following / preceding vowel's formants.
// k/g burst resonance: share from the preceding vowel (the rest: next).
constexpr float kBurstPrevVowel = 0.25f;  // vowel chains: overshoot away from neighbours
// At a pause speech fades out exponentially (linear in dB): ~3 dB per 10 ms.
constexpr float kSpeechFadeMs = 28;
constexpr float kSpeechAttackMs = 5;
// A voiceless sound starting a phrase (f, s, h) swells in over this. (50:
// the readers' phrase-initial f swells 20 dB over 40-60 ms (ElevenLabs,
// Google); at 59 puhuja's took 100 ms and the f stayed 24 dB under the vowel.
// Not searched on single words: every word is phrase-initial there and the
// Zimtohrli search stretched it to 166 ms, 2026-10-30.)
const float kVoicelessOnsetMs = param("kVoicelessOnsetMs", 50);
// Lungs. Volume is the air available for speaking (cm^3).
// After a full breath. The readers hold a breath group for 3-5 s typically
// and 8-11 s at the 90th percentile (ref/kennedy/*.words.tsv, gaps > 0.3 s);
// puhuja drains ~260 cm^3/s of speech (kennedy.txt trace, 2026-11-01), so
// a breath must carry ~2400 cm^3 above kLungRest for such groups. (2000:
// groups of 3.8 s median, 6.5 s at most, and the voice faded at the end of
// the long ones.)
const float kLungMax = param("kLungMax", 3400.0f);
constexpr float kLungRest = 1000;        // before the first breath
constexpr float kCommaBreathBelow = 900; // breathe at a comma below this (1600: three breaths in a row at the commas of "Let every nation know, whether it wishes us well or ill, that we shall pay any price,", 360-435 ms each where the recordings' commas are 170 ms; 1000: breaths fell at phrase splits)
constexpr float kInhaleRate = 2500;      // cm^3/s while breathing in
constexpr float kInhaleMinMs = 180, kInhaleMaxMs = 600;
constexpr float kCatchInhaleMaxMs = 300;  // within a sentence (at a comma or a seam): a quick catch breath
// The breath's length by its place: a catch breath inside a sentence shorter,
// between sentences longer, after an empty line (a paragraph) longer still.
constexpr float kCatchBreathScale = 0.85f, kSentenceBreathScale = 1.10f, kParagraphBreathScale = 1.30f;
constexpr int kBreathHoldStops = 0;  // silence between the inhale and the first sound
constexpr float kInhaleNoise = 0.035f;  // breathing-in noise (0.09: 23 dB under the vowels, a reader's breath is 35 under; heard as artificial)
// Breathing in: turbulence at the lips, nose and glottis, heard as a broad
// airy hiss (not shaped by the vocal tract, which would make it a nasal hum).
// Darker and flatter than it was ({700, 6500, 2500, 3000}): a reader's
// 400 ms breath has its energy at 1-3 kHz, 3-4 kHz 10 dB under and 4-6 kHz
// 16 under; puhuja's reached 6 kHz within 8 dB, a "niiuf".
constexpr FricShape kInhaleShape = {500, 3000, 1400, 1600, 1.0f};  // ({600, 4500, 1800, 2500}: 4-6 kHz was 11 dB under the peak, the reader's 16)
// Its envelope: a quick rise, a flat run and a gentle fall (a half sine over
// the whole breath swelled up and down; the reader's stays within 5 dB).
constexpr float kInhaleRiseMs = 70, kInhaleFallMs = 120;
// A breath snatched between words (the air ran out with no comma or seam
// near): the old quick breath, brighter and swelling (the user: a "niiuf"
// is right there, the calm "hhhh" at the proper places).
constexpr FricShape kInhaleSnatchShape = {700, 6500, 2500, 3000, 1.0f};
// Lung pressure eases off as the breath is used up (loudness declination)
// and falls steeply near empty.
constexpr float kLungPressureSpent = 0.85f;  // relative pressure at kLungRest and below
constexpr float kLungEmpty = 150;            // below this the pressure collapses
constexpr float kVowelLevelRiseMs = 40;  // a vowel's level after a sonorant consonant
constexpr float kNoiseCrossfadeMs = 4;  // noise fading in from silence
// Noise shape glide between two noisy sounds (s -> ʃ, a burst into s).
constexpr float kNoiseGlideMs = 10;
constexpr float kNoiseRiseGlideMs = 25;  // ... when the noise gets much stronger
constexpr float kSoftNoiseOnsetMs = 12;  // ð θ f v fading in from silence
constexpr float kPlosiveToSoftNoiseMs = 30;  // ... after a plosive's release (t d -> ð)
// The noise counts as silent below this share of its recent peak.
constexpr float kNoiseSilent = 0.01f;  // (0.1: -20 dB, resets while still audible)
constexpr float kAcSoft = 0.001f;
constexpr float kGlottalOnsetMs = 20;
// Creaky statement ending: pressed glottis from this share of the last vowel.
constexpr float kCreakStart = 0.5f, kCreakMs = 60;
constexpr float kCreakAg = 0.8f * kAgVoice;  // (0.5: pressed, squeezed)
// Creak: micro-variation multiplier (+x3) and breath pressure.
constexpr float kCreakJitter = 2.0f, kCreakPress = 0.6f;  // glottal onset: press before, release over
constexpr float kEndPressure = 0.65f;  // breath pressure at a phrase end, relative (0.75)
constexpr float kEndEaseMs = 220;     // eased off over this long before the pause (150: endings faded instead of releasing)
constexpr float kPressResumeMs = 30;
constexpr float kOnsetRiseVoiceless = 1.047f, kOnsetDipVoiced = 0.971f;  // +0.8 / -0.5 st
constexpr float kPhraseOnsetDrop = 0.944f;  // a phrase starts 1 semitone below its first target
// Micro-variation of the folds (relative sizes; the low-passed noise has an
// RMS of ~0.08, so these are ~10x the resulting RMS variation).
// The second stage, its strength. 3.0 (with kPitchDrift 0.012 and kF0Pressure
// 0.15) gave the folds a cycle-to-cycle jitter of 5.1 % against 2.1-2.4 % in
// the recorded voices (Kokoro am_onyx, ElevenLabs, ryan-ai); Jyrki asked for
// a more regular physical voice (2026-10-30), and with these three at 0.5,
// 0.004 and 0.05 the jitter is 2.6 %; he then asked for the micro-variation
// at one third of the old value (1.0; jitter ~3.3 %). (1.4 had kept the RMS; the HNR was
// then 2.7 dB "too clean", but HNR counts a clear voice's transients as
// noise, so it was not a target worth matching.) The round trips slipped
// with each fold change (steps 19-24: 8.1 -> 10.0 % male), and Jyrki chose
// to return to the step 15 values (3.0, 0.012, 0.15; readings
// ref/kennedy/steps/15-onyx-{m,f}.wav).
// Cycle-to-cycle jitter and shimmer: a new random deviation of the fold
// tension and the lung pressure at every closure, held through the next
// cycle (relative, for a unit-variance draw). The slow 8 Hz stage above
// cannot give this: raised, it became a tremor. Male readers: jitter
// 0.9-1.4 %, shimmer 0.44-0.81 dB, HNR 5-9 dB; puhuja had 0.6 %, 0.22 dB,
// 13 dB.
// Below ~10 Hz the variation is heard as a natural, slow unsteadiness; at
// 30 Hz it was roughness (graininess): harmonics above 1.5 kHz broke into
// noise, vowel HNR 1-2 kHz 13 dB (goog.wav 19), shimmer 0.20 dB (0.11),
// and the pitch wobbled 15 cents within a vowel (6). At 8 Hz with gain 3:
// 23 dB, 0.06 dB, 7 cents.

constexpr float kNasalPoleHz = 270;
// Level given back for the nasal zero: (zero / pole)^this (1: the highs
// above the zero lose half their dB, the lows gain as much). With the
// zeros at m 1000, n 2200, ŋ 2500 Hz, 0 fits the voice-over's nasal spectra
// best, but turns m n into a dull low hum that sounds like a buzz
// (tools/artifacts.py: roughness beyond real speech); 0.7 keeps it within.
constexpr float kNasalZeroComp = 1.0f;
// Width of the nasal pole and zero (equal, so they cancel in oral sounds).
// Broad: a narrow zero carves a deep notch into the nasal murmur.
constexpr float kNasalBw = 250;
// Nasal murmur low cut: share of the signal below kNasalLowCutHz removed (0.29: -3 dB).
constexpr float kNasalLowCut = 0.29f, kNasalLowCutHz = 400;
// First-order high-pass on the output: -6 dB/octave below.
constexpr float kOutputHighPassHz = 100;
constexpr float kTrillHz = 26;
// Airflow through the glottis (cm^3/s, physical source) the trill needs:
// none below kTrillFlowMin, full depth above kTrillFlowFull.
constexpr float kTrillFlowMin = 60;
constexpr float kTrillFlowFull = 160;
constexpr float kTrillDecayMs = 40;   // exponential fade when the flow drops
constexpr float kTrillAttackMs = 10;
constexpr float kTrillSlowing = 0.3f; // tap rate drops by this share without flow
constexpr float kLongFactor = 1.66f;             // ː
// Doubled letters, from the Finnish male reader (CSS10): a long vowel is
// ~1.7x a short one, a long consonant ~1.4x (plosives: own closure table).
constexpr float kDoubledConsonantFactor = 1.4f;

// Bandwidth scales from LPC of strongly voiced frames: the male voice-overs
// have B1 77-90, B2 132-210, B3 170-209 Hz; the female one 131 / 223 / 231.
// Bandwidths x0.6/0.7/0.75 (LPC on 30 s of kennedy.txt: B1/B2 146/173 Hz with
// the scale at 1/1.15 against ryan-ai's 79/117 and seven male readers'
// 102-159/116-173; the third-octave peak-valley contrast at 300-3500 Hz was
// 19 dB against ryan's 28 and the readers' 21-24: the "porridge". At 0.6:
// 106/105, 21 dB.)
// Male voice tuned toward Kokoro am_onyx (ref/kennedy/koko-0006-m.wav,
// 2026-10-30): a deep, steady, dark voice. Against it puhuja had 6-12 dB
// too much above 2 kHz (2-3 kHz -11 vs -17 relative to 250-500 Hz, 4-6 kHz
// -16 vs -23), valleys between the formants 8 dB shallower (third-octave
// contrast 20 vs 28 dB) and B1/B2 105/114 vs 98/102. Treble shelf -6 dB,
// air 0.3, bandwidths x0.5/0.55/0.6 and the source tilt -8 dB, 3 kHz cut
// -10 dB, 1 kHz lift +6 dB land on its spectrum within 1-2 dB, contrast 24.
// (Pitch and gain are research parameters: tools/word_opt.py --sweep.)
const Voice kMaleVoice = {param("kMaleF0", 104.8) /* cost 0.0322859615827058 n13-speaker-w30 */, {1.0f, 1.0f, 1.0f}, 1.0f, 1.0f, 1.7f, 1.0f, param("kMaleGain", 4.163f) /* cost 0.0322859615827058 n13-speaker-w30 */, -6, 0.3f,
                          {0.5f, 0.55f, 0.6f}, 1.0f, 1.0f, 0.0f, 3500, 4500, 3.0f, 1.0f, 1.0f, -8, -10, 6};
// Female: F1/F2/F3 from the measured FLEURS female / CSS10 male ratios
// (F1 x1.27, F2 x1.04, F3 x1.18), moderated toward usual values; tempo and
// long-vowel length from FLEURS female durations. Its higher pitch packs
// more energy into the formants, so its output level is turned down.
// Treble and high breath from the female voice-over (demo-female.wav): it is
// brighter than the male one, 6 dB above 2.5 kHz and 9-10 dB above 5 kHz.
// Female voice against goog.wav (tools/harmonicity.py, 2026-10-29): as it
// was (188 Hz, F1-F3 x1.2/1.1/1.1, bandwidths x1.5/1.4/1.3, gap 0.03,
// breathiness 1.5, micro 3) its vowels were 5 dB cleaner than goog's (HNR
// 17.9 vs 12.8), with the formant peaks 4-5 dB weaker against the
// fundamental (H1-A1 +0.9 vs -3.5, H1-A3 19.4 vs 14.1), half the pitch
// movement and F0 195 vs 212: dark and even, "half man, half robot". Now:
// 200 Hz (208 realised), F1-F3 x1.12/1.15/1.06 (LPC of goog's vowels said
// 1.02 for F3, but the ɜ-r words want the r colour higher: 1.02/1.06/1.10
// gave ɜ-r WER 14.1/13.1/12.1 and 60-statement 7.2/6.9/8.3),
// bandwidths x1.15/1.1/1.05, no gap, breathiness 1.1, micro 3: HNR ~18,
// H1-H2 1.7 (goog 2.0), H1-A1 -2.5, H1-A3 15.4, jitter 0.8 % (goog 1.2),
// F0 207; female Whisper WER on 60 statements 10.4 -> 7.6 %.
const Voice kFemaleVoice = {param("kFemaleF0", 165.7), {1.12f, 1.15f, 1.06f}, 1.15f, 0.92f, 1.8f, 1.1f, param("kFemaleGain", 3.704f) /* cost 0.0322859615827058 n13-speaker-w30 */, 7, 1.0f,
                                {1.15f, 1.1f, 1.05f}, 1.12f, 1.41f, 0.0f,
                                // F4, F5: goog and the female voice-over have F4 at ~3.98 kHz (LPC of
                                // the vowel middles). F3 x1.15 (2960 Hz vs 2770-2840) with F4 3850
                                // crowded F3 and F4 into one bump: vowels 5-8 dB hot at 3-4 kHz.
                                4000, 4500, 3.0f};  // (micro 5 matched goog's jitter but as an 8 Hz wobble of +-24 cents: an old woman's tremor, "now" wobbly)
// Bandwidth scale of consonant targets (see setTargets in emit).
constexpr float kConsonantBw = 1.5f;
// Voice heard through the chest wall and neck, without the mouth's
// resonances: driven by the glottal flow itself (the subglottal pressure
// follows the flow, not its change), inverted relative to the mouth's
// sound, and low (kChestHz). Weak: about -20 dB.
constexpr float kChestDirect = 0.005f;
constexpr float kChestHz = 500;

// ʤ's frication is stronger than a plain ʒ's (measured against Piper).
// ... and its heard level, after the ceiling: at 1 the ʧ was 5 dB under the
// vowel before it (1-8 kHz), the readers 0 to +18 (median +4); a final ʧ in
// a recording of "rich." is loud and 230 ms.
constexpr float kChFricationLevel = 3.0f;
const float kChFricationBoost = param("kChFricationBoost", 0.9964f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // ʧ's +5 dB (it was ~11 dB under the voice-over's)
const float kJhFricationBoost = param("kJhFricationBoost", 4.553f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // (1.6, +10 % by ear for "pledge", +10 % for "generosity")
// Its noise band: lower and wider than ʒ's {1800, 7000, 2800, 1200, 3}.
constexpr FricShape kJhNoiseShape = {1000, 7000, 2500, 1500, 3.0f};
// The frication part of an affricate is shorter than a plain fricative.
constexpr float kAffricateFricationLength = 0.55f;
constexpr float kJhFricationLength = 1.1f;
// English ʧ: frication x1.3 (53 ms was heard as a stop, 77 ms as ʃ) and a fast release.
constexpr float kEnglishChFricationLength = 1.6f, kEnglishChReleaseMs = 8;
// English affricates: frication length as a multiple of the reader's closure
// (1.8: a stressed ʧ's 50 ms closure + 90 ms frication, as before).
const float kAffricateFricationPerClosure = param("kAffricateFricationPerClosure", 0.8389f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // (1.9: onset and medial ʤ ʧ 137-182 ms in kennedy.txt against 70-150, median ~115, in six references; "generosity" 163 vs 100-150, "torch" 137 vs 70-80, 2026-10-30)  // (2.2: ʧ ʤ 145-168 ms against 90-120 in am_onyx and ElevenLabs; a quarter of the way back, 2026-10-30)  // (1.8: a final "rich" came back from Whisper as "right"; a recording's final ʧ has 230 ms of hiss)  // (1.3: ʧ 124 ms, the readers 120-190, a final one in a recording 230)  // ... and ʤ's 10 % longer (by ear)

enum class Cls { Vowel, Plosive, Nasal, Liquid, Trill, Glide, Fricative, H, Stop };

struct Info {
  Cls cls;
  float f1, f2, f3;  // formant targets (Hz)
  float b1, b2, b3;  // bandwidths (Hz)
  float fnz;         // nasal zero (Hz), kNasalPoleHz when oral
  float av;          // voicing amplitude
  float ms;          // duration (short)
  float noise;       // fricatives: constriction noise gain (shape: fricShape)
};

// Approximate values for a Finnish male voice. Durations are the medians of a
// Finnish male reader (CSS10, aligned). Vowel formants: e i y ä from that
// reader, a o u ö from Piper's fi_FI-harri voice, then spread apart with
// tools/vowel_dispersion.py (penalty 0.3, jointly with the English vowels):
// the closest Finnish pair rose from 1.0 to 1.9 Bark, the closest English
// pair from 0.3 to 1.3 Bark, with moves of at most 0.5 Bark.
const Info kInfo[] = {
    //  cls          f1    f2    f3   b1   b2   b3   fnz  av    ms  noise
    {Cls::Vowel,       698, 1132, 2211,  80, 100, 150, 270, 1.14f,  90, 0},    // A
    {Cls::Vowel,       458, 1763, 2347,  70, 100, 150, 270, 0.802f,  70, 0},    // E
    {Cls::Vowel,       300, 2059, 2575,  60, 100, 150, 270, 1.56f,  70, 0},    // I
    {Cls::Vowel,       438,  790, 2184,  70, 100, 150, 270, 1.15f, 110, 0},    // O
    {Cls::Vowel,       299,  618, 2347,  60, 100, 150, 270, 1.57f, 100, 0},    // U
    {Cls::Vowel,       283, 1573, 2233,  60, 100, 150, 270, 2.07f,  80, 0},    // Y
    {Cls::Vowel,       707, 1546, 2443,  80, 100, 150, 270, 0.912f,  70, 0},    // AE
    {Cls::Vowel,       508, 1341, 2241,  70, 100, 150, 270, 1.02f,  80, 0},    // OE
    {Cls::Vowel,     546, 1517, 2456,  80, 110, 160, 270, 0.9f,  80, 0},     // Schwa
    // English-only vowels: Hillenbrand et al. (1995) men (British values for
    // ɒ ɜ), spread together with the shared ones (see vowel_dispersion.py).
    {Cls::Vowel,     429, 2093, 2701,  60, 100, 150, 270, 1.82f,  70, 0},     // IH ɪ
    {Cls::Vowel,     600, 1809, 2615,  70, 100, 150, 270, 0.8f,  80, 0},     // EH ɛ
    {Cls::Vowel,     471, 1111, 2435,  60, 100, 150, 270, 1.7f,  70, 0},     // UH ʊ
    {Cls::Vowel,     681, 1005, 2539,  70, 100, 150, 270, 1.11f, 100, 0},     // AO ɔ
    {Cls::Vowel,     619, 1215, 2550,  80, 100, 150, 270, 1.47f,  80, 0},     // AH ʌ
    {Cls::Vowel,     780, 1296, 2525,  80, 100, 150, 270, 1.72f, 100, 0},     // AA ɑ
    {Cls::Vowel,     549,  890, 2507,  70, 100, 150, 270, 1.03f,  85, 0},     // OH ɒ
    {Cls::Vowel,     433, 1346, 2393,  80, 110, 160, 270, 0.9f, 100, 0},     // ER ɜ
    {Cls::Plosive,    0,    0,    0,   0,   0,   0, 270, 1.0f,   0, 0},     // P
    {Cls::Plosive,    0,    0,    0,   0,   0,   0, 270, 1.0f,   0, 0},     // T
    {Cls::Plosive,    0,    0,    0,   0,   0,   0, 270, 1.0f,   0, 0},     // K
    {Cls::Plosive,    0,    0,    0,   0,   0,   0, 270, 1.0f,   0, 0},     // B
    {Cls::Plosive,    0,    0,    0,   0,   0,   0, 270, 1.0f,   0, 0},     // D
    {Cls::Plosive,    0,    0,    0,   0,   0,   0, 270, 1.0f,   0, 0},     // G
    {Cls::Plosive,    0,    0,    0,   0,   0,   0, 270, 1.0f,   0, 0},     // CH
    {Cls::Plosive,    0,    0,    0,   0,   0,   0, 270, 1.0f,   0, 0},     // JH
    {Cls::Nasal,    480, 1150, 2200, 150, 100, 120, 700, 0.428f,  65, 0},    // M
    {Cls::Nasal,    480, 1600, 2600, 150, 100, 120, 1000, 0.328f,  60, 0},    // N
    {Cls::Nasal,    480, 1900, 2400, 150, 100, 120, 1300, 0.276f,  50, 0},    // NG
    {Cls::Liquid,   360, 1300, 2800, 100, 120, 200, 270, 1.08f,  50, 0},    // L
    {Cls::Trill,    450, 1350, 2500, 100, 150, 250, 270, 0.58f,  90, 0},    // R
    {Cls::Glide,    360, 2050, 2750, 120, 90, 120, 270, 0.79f,  90, 0},    // J
    {Cls::Glide,    420, 1200, 2300, 120, 90, 120, 270, 0.355f,  80, 0},    // V
    // English ɹ: low F3 gives the r colour. w: like u with rounded lips.
    // F3 1780 (1650: LPC on a male reader gives ɹ 1830, ɜ 1745; puhuja's 1650
    // and 1550 made "who are rich" one long r, "where").
    {Cls::Glide,    350, 1100, 1650, 120, 100, 120, 270, 1.0f,  70, 0},     // RR ɹ (F3 1780: puhuja's r reached 1730 Hz where the male readings' word-initial r reaches 1450-1600; "r should be pronounced much more", 2026-10-30)
    {Cls::Glide,    320,  680, 2200, 120,  90, 120, 270, 0.9f,  70, 0},     // W
    {Cls::Fricative, 350, 1700, 2700, 200, 200, 300, 270, 1.0f,  90, 0.118f},    // S
    {Cls::Fricative, 300, 1800, 2400, 200, 200, 300, 270, 1.0f, 110, 0.06f},   // SH
    {Cls::Fricative, 300, 1800, 2400, 150, 200, 300, 270, 1.0f, 100, 0.18f},   // ZH
    {Cls::Fricative, 350, 1000, 2300, 200, 200, 300, 270, 1.0f,  65, 0.104f},    // F
    {Cls::Fricative, 350, 1700, 2700, 150, 200, 300, 270, 1.0f,  80, 0.5f},     // Z
    {Cls::Fricative, 350, 1400, 2600, 200, 200, 300, 270, 1.0f,  80, 0.045f},    // TH θ
    // ð: tongue against the teeth: weakened voicing and dental turbulence
    // (flat: nothing in front of the teeth resonates), pulsed by the voice.
    {Cls::Fricative, 380, 1500, 2600, 100, 100, 150, 270, 0.55f,  60, 0.8f},      // DH ð (more voice, less hiss: nearly an approximant)
    // English v: the lower lip against the teeth, a real constriction with
    // frication (the Finnish ʋ above is an open approximant: as that, with
    // its level, "revolutionary" dipped 5 dB where goog.wav dips 18 and the
    // references 3-10; no hiss). Formants and bandwidths as f. With full
    // voicing the constriction alone dips it 7 dB before a consonant and 9
    // before a vowel over kennedy.txt (goog 8; at 0.6 it was 11 and 13).
    {Cls::Fricative, 350, 1000, 2300, 200, 200, 300, 270, 1.0f,  60, 0.6f},      // VV v
    {Cls::H,          0,    0,    0,   0,   0,   0, 270, 1.0f,  80, 0},    // H
    {Cls::Stop,       0,    0,    0,   0,   0,   0, 270, 0.0f,  15, 0},     // Stop
};
static_assert(sizeof(kInfo) / sizeof(kInfo[0]) == static_cast<size_t>(Ph::Count));

const Info& info(Ph ph) { return kInfo[static_cast<int>(ph)]; }
bool isVowel(Ph ph) { return info(ph).cls == Cls::Vowel; }
bool isSonorant(Ph ph) {
  Cls c = info(ph).cls;
  return c == Cls::Vowel || c == Cls::Nasal || c == Cls::Liquid ||
         c == Cls::Trill || c == Cls::Glide;
}

struct PlosiveInfo {
  float closureMs;      // closure, from full closing to the release
  float longClosureMs;  // geminate ("kk") closure
  float releaseMs;      // how long the opening movement takes: a slow
                        // release (velars) gives a longer burst
  float adductMs;       // voiceless: glottis closing again after the release;
                        // voicing restarts on the way (voice onset time)
  float noiseGain;      // strength of the noise at the release
  bool voiced;          // glottis stays vibrating through the closure
  float pop = 1;        // strength of the release transient
  float adductLeadMs = 0;  // voiceless: glottis starts closing this long
                           // before the release (unaspirated: quick voicing)
  float closureAg = -1;    // glottal area during the closure (cm^2); < 0:
                           // vibrating position if voiced, open if voiceless
};

// English aspiration (setAspiration): how long the glottis takes to close
// after the release of p t k: stressed syllables (VOT ~60-75 ms) and
// unstressed ones (~35-45 ms).
// d before a dental fricative: closure share and release strength.
constexpr float kDentalDClosure = 0.5f, kDentalDRelease = 0.25f;
constexpr float kDentalReleaseMs = 30;  // t d into ð θ: the opening
const float kUnreleasedClosure = param("kUnreleasedClosure", 0.8351f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // a plosive before another: closure share
constexpr float kLinkedDRelease = 2.5f;  // word-final d before a vowel-initial word: burst and pop
constexpr float kLinkedDReleaseSlow = 2.0f;  // and its opening this much slower (a longer burst)
const float kVoicedAfterConsonantRelease = param("kVoicedAfterConsonantRelease", 1.15f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // b d g after a consonant: burst and pop
// English b after a pause or consonant: glottal opening during the closure
// (share of voice -> open), burst noise and pop multiplier.
const float kEnglishBOpening = param("kEnglishBOpening", 0.4368f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */, kEnglishBNoise = param("kEnglishBNoise", 0.09007f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */, kEnglishBPop = param("kEnglishBPop", 27.69f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;
const float kDevoiceBeforeFric = param("kDevoiceBeforeFric", 1.007f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // b d g before f θ s ʃ: closure glottis share
const float kFlapRelease = param("kFlapRelease", 2.405f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // a flap's burst and pop
const float kEnglishKBurst = param("kEnglishKBurst", 9.019f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // (11 until 2026-10-30: k bursts 6 dB under the readings')  // English k burst (5: +14 dB, +8 left cow -> power; 11: at 5 its burst peak was 5.5 dB under the recordings' against the vowel; "cannot" as "(k)annot")
const float kEnglishKReleaseMs = param("kEnglishKReleaseMs", 14.47) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // English k release (25)
// (90 and 1.25: word-initial k p had a voice onset time of 124-142 ms against
// 55-90 in Kokoro (am_onyx 66-140, am_adam 34-74) on cup, kate, coat, cope,
// printer, pointed; the aspiration ~4 dB quieter but twice as long; the word
// survey's worst words were these stops. Half-way: VOT ~80 ms, 2026-10-30.)
const float kAspiratedAdductMs = param("kAspiratedAdductMs", 14.79) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */, kAspiratedUnstressedMs = param("kAspiratedUnstressedMs", 23.82) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // (70: VOT still 106-120; the folds take ~40 ms to start after the glottis closes)
const float kStrongAspirationLength = param("kStrongAspirationLength", 0.6184f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */, kStrongAspirationGain = param("kStrongAspirationGain", 2.594f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // word-initial p t k of a primary-stressed syllable
// Finnish plosives are unaspirated: the glottis closes again quickly.
PlosiveInfo plosiveInfo(Ph ph) {
  switch (ph) {
    case Ph::P: return {80, 100, 6, 10, 0.082f, false, 0.105f, 8};  // +10 % (by ear)
    case Ph::T: return {70, 110, 12, 25, 0.122f, false};
    case Ph::K: return {70, 100, 25, 25, 0.0684f, false};
    case Ph::B: return {60, 100, 10, 0, 0.0157f, true};
    case Ph::D: return {70, 110, 10, 0, 0.123f, true, 0.5f};
    case Ph::G:  // 15% of the way toward k: firmer closure, crisper release
      return {80, 120, 21, 0, 0.089f, true, 1.34f, 0,
              kAgVoice + 0.15f * (kAgOpen - kAgVoice)};
    // Affricates: released slowly into the fricative constriction.
    case Ph::CH: return {50, 120, 15, 25, 0.053f, false};
    default:     return {45, 110, 15, 0, 0.085f, true};  // JH
  }
}

// Place of articulation of a plosive: P/B labial, T/D alveolar, K/G velar,
// CH/JH postalveolar (just behind the alveolar ridge, as for ʃ).
int place(Ph ph) {
  switch (ph) {
    case Ph::P: case Ph::B: return 0;
    case Ph::T: case Ph::D: return 1;
    case Ph::CH: case Ph::JH: return 3;
    default: return 2;
  }
}

// Formants at the moment of closure/release, from the place of articulation
// and the neighbouring vowel ("locus equations"): the vowel pulls the
// consonant's formants toward itself, most strongly for velars.
void plosiveLocus(Ph ph, float vowelF2, float* f2, float* f3) {
  switch (place(ph)) {
    case 0:  // lips: everything goes down
      *f2 = 0.5f * vowelF2 + 400;
      *f3 = 2200;
      break;
    case 1:  // tongue tip: F2 toward ~1800 Hz
      *f2 = 0.3f * vowelF2 + 1250;
      *f3 = 2700;
      break;
    case 3:  // postalveolar: F2 high, F3 a bit lowered (lip rounding)
      *f2 = 0.3f * vowelF2 + 1400;
      *f3 = 2400;
      break;
    default:  // back of the tongue: F2 and F3 come together
      *f2 = 0.6f * vowelF2 + 700;
      *f3 = *f2 + 500;
      break;
  }
}

// Noise spectra of the fricatives:
//   s: little below ~4.5 kHz, peak around 6 kHz (matched to Piper; Finnish s
//      is a little lower than English), falling off above ~10 kHz.
//   ʃ ʒ: longer front cavity (lips rounded), so lower: peak ~2.8 kHz.
//   f: weak and flat over a wide band, no front cavity to resonate.
FricShape fricShape(Ph ph) {
  switch (ph) {
    case Ph::SH: case Ph::ZH: return {1800, 7000, 2800, 1200, 3.0f};
    case Ph::F: case Ph::VV: return {1000, 6500, 3500, 5000, 0.0f};  // (to 9-11 kHz it was too bright, "forget")
    case Ph::TH: case Ph::DH: return {1200, 11000, 6000, 6000, 0.0f};
    default:                  return {3000, 10000, 6000, 2500, 2.5f};  // S (low edge 4500: a thin s)
  }
}

// Burst spectra, following the classic descriptions of release bursts
// (Halle, Hughes & Radley 1957; Stevens & Blumstein 1978):
//   labial p b:   diffuse-falling. Nothing in front of the lips resonates, so
//                 the energy is spread out, mostly low, and falls off. Made
//                 by exciting the vocal tract itself (pop + flat noise).
//   alveolar t d: diffuse-rising. A tiny front cavity: little below 2.5 kHz,
//                 energy rising to a broad peak around 4-5 kHz.
//   velar k g:    compact. One strong, narrow mid-frequency peak from the
//                 cavity in front of the tongue body, which follows the next
//                 vowel's F2: ~1.4 kHz before u, ~2.2 kHz before i.
//   postalveolar ʧ ʤ: like ʃ, a peak around 2.8 kHz.
FricShape burstShape(int place, float nextF2) {
  switch (place) {
    case 0: return {150, 6000, 800, 1500, 1.0f};  // lips: diffuse, falling
    case 1: return {2500, 10000, 4500, 3500, 2.0f};
    case 3: return {1800, 8000, 2800, 1500, 2.5f};
    default: {
      float peak = std::clamp(0.6f * nextF2 + 900, 1200.0f, 2800.0f);
      return {0.6f * peak, 4000, peak, 500, 6.0f};
    }
  }
}

// Airway targets of a phoneme: constriction, glottis and nasal port areas.
struct Aero {
  float ac, ag, an;
};

Aero aeroOf(Ph ph) {
  switch (info(ph).cls) {
    case Cls::Vowel: return {kAcOpen, kAgVoice, 0};
    case Cls::Nasal: return {0, kAgVoice, kAnNasal};  // mouth closed, nose open
    case Cls::Liquid: return {0.35f, kAgVoice, 0};
    case Cls::Trill: return {0.3f, kAgVoice, 0};
    case Cls::Glide: return {ph == Ph::J ? 0.3f : 0.25f, kAgVoice, 0};
    case Cls::Fricative:
      switch (ph) {
        case Ph::S: return {0.05f, kAgOpen, 0};
        case Ph::SH: return {0.08f, kAgOpen, 0};
        case Ph::ZH: return {0.08f, kAgVoice, 0};  // voiced: less pressure drop
        case Ph::Z: return {0.05f, kAgVoice, 0};
        case Ph::DH: return {0.14f, kAgVoice, 0};
        case Ph::VV: return {0.08f, kAgVoice, 0};
        default: return {0.1f, kAgOpen, 0};        // F
      }
    case Cls::H: return {1.5f, kAgOpen, 0};
    case Cls::Stop: return {kAcOpen, 0, 0};  // glottis closed: no airflow
    case Cls::Plosive: return {0, kAgOpen, 0};  // the closure
  }
  return {kAcOpen, kAgVoice, 0};
}

}  // namespace

float Gesture::tick() {
  const float prev = v_;
  if (left_ >= 1.0f) {
    left_ -= 1.0f;
    const float t = 1.0f - left_ / len_;
    const float x = 0.5f - 0.5f * std::cos(kPi * t);
    // Raised-cosine path plus the carried-over speed (Hermite h10 term).
    v_ = from_ + (to_ - from_) * x + m0_ * t * (1.0f - t) * (1.0f - t);
  }
  vel_ = v_ - prev;
  return v_;
}

void Resonator::set(float freq, float bw, float fs) {
  float r = std::exp(-kPi * bw / fs);
  c_ = -r * r;
  b_ = 2.0f * r * std::cos(2.0f * kPi * freq / fs);
  a_ = 1.0f - b_ - c_;
}

void AntiResonator::set(float freq, float bw, float fs) {
  Resonator r;
  r.set(freq, bw, fs);
  a_ = 1.0f / r.a_;
  b_ = -r.b_ / r.a_;
  c_ = -r.c_ / r.a_;
}

void Biquad::lowPass(float freq, float fs, float q) {
  float w0 = 2.0f * kPi * std::min(freq, 0.45f * fs) / fs;
  float alpha = std::sin(w0) / (2.0f * q);
  float cw = std::cos(w0), a0 = 1.0f + alpha;
  b0_ = (1.0f - cw) / 2.0f / a0;
  b1_ = (1.0f - cw) / a0;
  b2_ = b0_;
  a1_ = -2.0f * cw / a0;
  a2_ = (1.0f - alpha) / a0;
}

void Biquad::highPass(float freq, float fs) {
  float w0 = 2.0f * kPi * std::min(freq, 0.45f * fs) / fs;
  float alpha = std::sin(w0) / (2.0f * 0.7071f);
  float cw = std::cos(w0), a0 = 1.0f + alpha;
  b0_ = (1.0f + cw) / 2.0f / a0;
  b1_ = -(1.0f + cw) / a0;
  b2_ = b0_;
  a1_ = -2.0f * cw / a0;
  a2_ = (1.0f - alpha) / a0;
}

void Biquad::highPass1(float freq, float fs) {
  // Bilinear transform of s / (s + w0), prewarped.
  float k = std::tan(kPi * std::min(freq, 0.45f * fs) / fs);
  b0_ = 1.0f / (1.0f + k);
  b1_ = -b0_;
  b2_ = 0;
  a1_ = (k - 1.0f) / (k + 1.0f);
  a2_ = 0;
}

void Biquad::highShelf(float freq, float gainDb, float fs) {
  float A = std::pow(10.0f, gainDb / 40.0f);
  float w0 = 2.0f * kPi * std::min(freq, 0.45f * fs) / fs;
  float cw = std::cos(w0), sw = std::sin(w0);
  float alpha = sw / 2.0f * std::sqrt(2.0f);  // shelf slope S = 1
  float sA = 2.0f * std::sqrt(A) * alpha;
  float a0 = (A + 1) - (A - 1) * cw + sA;
  b0_ = A * ((A + 1) + (A - 1) * cw + sA) / a0;
  b1_ = -2 * A * ((A - 1) + (A + 1) * cw) / a0;
  b2_ = A * ((A + 1) + (A - 1) * cw - sA) / a0;
  a1_ = 2 * ((A - 1) - (A + 1) * cw) / a0;
  a2_ = ((A + 1) - (A - 1) * cw - sA) / a0;
}

void Biquad::peaking(float freq, float bw, float gainDb, float fs) {
  float A = std::pow(10.0f, gainDb / 40.0f);
  float w0 = 2.0f * kPi * std::min(freq, 0.45f * fs) / fs;
  float alpha = std::sin(w0) * bw / (2.0f * freq);
  float a0 = 1 + alpha / A;
  b0_ = (1 + alpha * A) / a0;
  b1_ = -2 * std::cos(w0) / a0;
  b2_ = (1 - alpha * A) / a0;
  a1_ = b1_;
  a2_ = (1 - alpha / A) / a0;
}

void FricationFilter::set(const FricShape& s, float fs) {
  hp1_.highPass(s.lo, fs);
  hp2_.highPass(s.lo, fs);
  lp1_.lowPass(s.hi, fs);
  lp2_.lowPass(s.hi, fs);
  peak_.set(s.peakF, s.peakBw, fs);
  peakGain_ = s.peakGain;
}

void BandPass::set(float freq, float bw, float fs) {
  freq = std::min(freq, 0.45f * fs);
  float w0 = 2.0f * kPi * freq / fs;
  float alpha = std::sin(w0) * bw / (2.0f * freq);  // sin(w0) / (2Q)
  float a0 = 1.0f + alpha;
  b0_ = alpha / a0;
  b2_ = -alpha / a0;
  a1_ = -2.0f * std::cos(w0) / a0;
  a2_ = (1.0f - alpha) / a0;
}

Synth::Synth(float sampleRate, float baseF0) : fs_(sampleRate), baseF0_(baseF0 > 0 ? baseF0 : kMaleVoice.f0) {
  // Flow through an opening of area A: U = A * sqrt(2 dP / rho). In the
  // cavity, dPo/dt = (U_in - U_out) / C. With pressures relative to the lung
  // pressure, one sample changes po by flowGain_ * (area * sqrt(drop)).
  flowGain_ = std::sqrt(2.0f * kLungPressure / kAirDensity) /
              (kLungPressure * kCavityCompliance * fs_);
  kTension_ = 1.0f - std::exp(-1.0f / msToSamples(kTensionMs, fs_));
  // Logistic growth from kOnsetSeed to ~97% takes about 7 / rate.
  kVibGrowth_ = 7.0f / msToSamples(kOnsetMs, fs_);
  kVibDecay_ = 1.0f - std::exp(-1.0f / msToSamples(kOffsetMs, fs_));
  kNoiseLevelFollow_ = 1.0f - std::exp(-1.0f / msToSamples(10.0f, fs_));
  kNoisePeakDecay_ = std::exp(-1.0f / msToSamples(1000.0f, fs_));
  ac_.set(kAcOpen);
  expand_.set(1);
  planVolume_ = lungVolume_ = kLungRest;
  inhaleFilter_.set(kInhaleShape, fs_);
  inhaleSnatchFilter_.set(kInhaleSnatchShape, fs_);
  ag_.set(0);
  an_.set(0);
  np_.set(kNasalPoleHz, kNasalBw, fs_);
  kRoomLp_ = 1.0f - std::exp(-2.0f * kPi * kRoomToneHz / fs_);
  kNasLp_ = 1.0f - std::exp(-2.0f * kPi * kNasalLowCutHz / fs_);
  outHp_.highPass1(kOutputHighPassHz, fs_);
  lowCut_.highPass(kLowCutHz, fs_);
  for (int i = 0; i < kNumBodyResonances; ++i)
    body_[i].set(kBodyResonances[i].freq, kBodyResonances[i].bw, fs_);
  shelf_.highPass1(kShelfHz, fs_);
  midCut_.peaking(kMidCutHz, kMidCutBw, kMidCutDb, fs_);
  breathLp_.lowPass(kBreathLowPassHz, fs_);
  hbHp1_.highPass(kHighBreathHz, fs_);
  hbHp2_.highPass(kHighBreathHz, fs_);
  kAspLp_ = 1.0f - std::exp(-2.0f * kPi * kAspirationTiltHz / fs_);
  kAspHp_ = 1.0f - std::exp(-2.0f * kPi * kAspirationLowCutHz / fs_);
  presence_.peaking(kPresenceHz, kPresenceBw, kPresenceDb, fs_);
  kPitchLag_ = 1.0f - std::exp(-1.0f / msToSamples(kPitchLagMs, fs_));
  kAspGainFollow_ = 1.0f - std::exp(-1.0f / msToSamples(3.0f, fs_));
  aspGainTarget_ = aspGainNow_ = kAspirationGain * kAspirationTiltGain;
  kLookAheadSamples_ = std::max(kPitchLookAheadMs, kFormantLookAhead * kFormantTauMs[0]) * 0.001 * fs_;
  kCreakFollow_ = 1.0f - std::exp(-1.0f / msToSamples(20.0f, fs_));
  kAspBurstDecay_ = std::exp(-1.0f / msToSamples(kAspiratedAdductMs / 3.0f, fs_));
  shelfGain_ = std::pow(10.0f, kShelfDb / 20.0f) - 1.0f;
  physTilt_.highPass1(kPhysTiltHz, fs_);
  physTiltGain_ = std::pow(10.0f, voice_.sourceTiltDb / 20.0f) - 1.0f;
  airHp1_.highPass(kAirHz, fs_);
  airHp2_.highPass(kAirHz, fs_);
  airLp_.lowPass(kAirTopHz, fs_);
  kTrillDecay_ = 1.0f - std::exp(-1.0f / msToSamples(kTrillDecayMs, fs_));
  kTrillAttack_ = 1.0f - std::exp(-1.0f / msToSamples(kTrillAttackMs, fs_));
  kSpeechDecay_ = 1.0f - std::exp(-1.0f / msToSamples(kSpeechFadeMs, fs_));
  kPopSmooth_ = 1.0f - std::exp(-1.0f / msToSamples(2.0f, fs_));
  kSpeechAttack_ = 1.0f - std::exp(-1.0f / msToSamples(kSpeechAttackMs, fs_));
  speechAttackNow_ = kSpeechAttack_;
  voice_ = kMaleVoice;
  pulse_ = &kMalePulse;
  voice_.f0 = baseF0_;
  larynx_ = makeLarynx(Carrier::Physical);
  larynx_->init(fs_, voice_.f0, &rng_);
  larynx_->setPulse(*pulse_);
  setTract(tract_);  // the line is the default tract: initialise and attach it
  press_.set(1.0f);
  setFixedResonances();
  chestLp_.lowPass(kChestHz, fs_);
  physUpperCut_.peaking(kPhysUpperCutHz, kPhysUpperCutBw, voice_.sourceUpperCutDb, fs_);
  physMidLift_.peaking(kPhysMidLiftHz, kPhysMidLiftBw, voice_.sourceMidLiftDb, fs_);
  trebleShelf_.highShelf(kTrebleShelfHz, voice_.treble, fs_);  // (only setVoice set it: the male shelf was never on)
  for (int i = 0; i < kNumCavities; ++i) {
    cavPole_[i].set(kCavities[i].pole, kCavities[i].bw, fs_);
    cavZero_[i].set(kCavities[i].pole + kCavityDepth * (kCavities[i].zero - kCavities[i].pole),
                    kCavities[i].bw, fs_);
  }
  lipGain_ = fs_ / (2.0f * kPi * kLipRadiationHz);
  headShelf_.highShelf(kHeadShelfHz, kHeadShelfDb, fs_);
  for (int i = 0; i < 3; ++i) polesHigh_[i].highShelf(kPolesHigh[i][0] * voice_.highFormants, kPolesHigh[i][1], fs_);
}

// F4 and F5 travel a little. They are set mostly by the tract's length and
// the tube just above the folds, not by the tongue: lip rounding lengthens
// the tract and lowers them, F3 drags F4 along (up in i, down in ɹ and
// "world"), and the larynx rises a little with pitch. F3 carries the
// rounding too (u w ʃ ɹ lower it), so they follow (F3 / kF3Neutral)^k and
// (pitch)^kHighFormantPitch: about +-5-10 % over a sentence, as in goog.
constexpr float kF3Neutral = 2500;  // the reading's average voiced F3
constexpr float kF4FollowF3 = 0.3f, kF5FollowF3 = 0.2f;
constexpr float kHighFormantPitch = 0.05f;  // +3.5 % an octave up
// F4, F5 bandwidths (Hz). (330, 450: vowels 2-5 dB over the voice-overs at
// 3-5 kHz.)
constexpr float kF4Bw = 500, kF5Bw = 700;

void Synth::setFixedResonances() {
  const float s = voice_.highFormants;
  // F4/F5 bandwidths from the losses at those frequencies (Fant: ~300 and
  // ~450 Hz); narrower they ring at a constant pitch (the voice-overs' LPC B4
  // is 220-290 Hz, puhuja's was 121-140).
  r4_.set(voice_.f4, kF4Bw * s, fs_);
  r5_.set(voice_.f5, kF5Bw * s, fs_);
  for (int i = 0; i < kNumTubePoles; ++i)
    // Poles that would fall above 0.42 fs are left out (identity), not clamped:
    // clamped, several stacked at the same frequency (female voice: three at
    // 21.6 kHz) and the cascade's internal gain became huge (+147 dB), with
    // float round-off heard as hiss.
    tube_[i] = kTubePoles[i].freq * s < 0.42f * fs_
                   ? (tube_[i].set(kTubePoles[i].freq * s, kTubePoles[i].bw * s, fs_), tube_[i])
                   : Resonator();
}

void Synth::setTract(Tract t) {
  tract_ = t;
  if (t == Tract::Line) {
    line_.init(fs_);
    lineXFollow_ = 1.0f - std::exp(-1.0f / msToSamples(kBoundaryMs, fs_));
    larynx_->attachLine(&line_);
  } else {
    larynx_->attachLine(nullptr);
  }
}

void Synth::setCarrier(Carrier c) {
  carrier_ = c;
  larynx_ = makeLarynx(c);
  larynx_->init(fs_, voice_.f0, &rng_);
  larynx_->setBaseF0(voice_.f0);
  larynx_->setPulse(*pulse_);
  if (tract_ == Tract::Line) larynx_->attachLine(&line_);
}

void Synth::setVoice(VoicePreset preset) {
  voice_ = preset == VoicePreset::Female ? kFemaleVoice : kMaleVoice;
  pulse_ = preset == VoicePreset::Female ? &kFemalePulse : &kMalePulse;
  baseF0_ = voice_.f0;
  larynx_->setBaseF0(voice_.f0);
  larynx_->setPulse(*pulse_);
  trebleShelf_.highShelf(kTrebleShelfHz, voice_.treble, fs_);
  // The F3-region source cut follows the voice's F3 (higher for a woman).
  physUpperCut_.peaking(kPhysUpperCutHz * voice_.formant[2], kPhysUpperCutBw * voice_.formant[2],
                        voice_.sourceUpperCutDb, fs_);
  physMidLift_.peaking(kPhysMidLiftHz * voice_.formant[1], kPhysMidLiftBw * voice_.formant[1], voice_.sourceMidLiftDb, fs_);
  physTiltGain_ = std::pow(10.0f, voice_.sourceTiltDb / 20.0f) - 1.0f;
  setFixedResonances();
}

// ---------------------------------------------------------------------------
// Input: UTF-8 text -> phonemes.

void Synth::pushText(std::string_view text) {
  for (unsigned char c : text) {
    if (utf8Need_ > 0) {
      if ((c & 0xc0) == 0x80) {
        utf8Cp_ = (utf8Cp_ << 6) | (c & 0x3f);
        if (--utf8Need_ == 0) pushCodepoint(utf8Cp_);
        continue;
      }
      utf8Need_ = 0;  // broken sequence: drop it, treat c as a new start
    }
    if (c < 0x80) {
      pushCodepoint(c);
    } else if ((c & 0xe0) == 0xc0) {
      utf8Cp_ = c & 0x1f;
      utf8Need_ = 1;
    } else if ((c & 0xf0) == 0xe0) {
      utf8Cp_ = c & 0x0f;
      utf8Need_ = 2;
    } else if ((c & 0xf8) == 0xf0) {
      utf8Cp_ = c & 0x07;
      utf8Need_ = 3;
    }
  }
}

void Synth::pushCodepoint(uint32_t cp) {
  if (cp >= 'A' && cp <= 'Z') cp += 'a' - 'A';
  auto ph = [this](Ph p) {
    push(p, atWordStart_);
    atWordStart_ = false;
  };
  // Pauses are built from 15 ms stops.
  auto pause = [this](int stops) {
    for (int i = 0; i < stops; ++i) push(Ph::Stop);
    atWordStart_ = true;
  };
  switch (cp) {
    case 'a': ph(Ph::A); break;
    case 'e': ph(Ph::E); break;
    case 'i': ph(Ph::I); break;
    case 'o': case 0xe5: case 0xc5: /* å */ ph(Ph::O); break;
    case 'u': ph(Ph::U); break;
    case 0x26a: /* ɪ */ ph(Ph::IH); break;
    case 0x25b: /* ɛ */ ph(Ph::EH); break;
    case 0x28a: /* ʊ */ ph(Ph::UH); break;
    case 0x254: /* ɔ */ ph(Ph::AO); break;
    case 0x28c: /* ʌ */ ph(Ph::AH); break;
    case 0x251: /* ɑ */ ph(Ph::AA); break;
    case 0x252: /* ɒ */ ph(Ph::OH); break;
    case 0x25c: /* ɜ */ case 0x25a: /* ɚ */ ph(Ph::ER); break;
    case 'y': ph(Ph::Y); break;
    case 0xe4: case 0xc4: /* ä */ case 0xe6: case 0xc6: /* æ */ ph(Ph::AE); break;
    case 0xf6: case 0xd6: /* ö */ case 0xf8: case 0xd8: /* ø */
    case 0x153: /* œ */ ph(Ph::OE); break;
    case 0x259: /* ə */ case 0x250: /* ɐ */ ph(Ph::Schwa); break;
    case 0x268: /* ɨ */ ph(Ph::I); break;
    case 0x26f: /* ɯ */ ph(Ph::U); break;
    case 'p': ph(Ph::P); break;
    case 't': ph(Ph::T); break;
    case 'k': case 'c': case 'q': ph(Ph::K); break;
    case 'b': ph(Ph::B); break;
    case 'd': ph(Ph::D); break;
    case 'g': case 0x261: /* ɡ */ ph(Ph::G); break;
    case 'm': ph(Ph::M); break;
    case 'n': ph(Ph::N); break;
    case 0x14b: case 0x14a: /* ŋ */ ph(Ph::NG); break;
    case 'l': ph(Ph::L); break;
    case 'r': case 0x27e: /* ɾ */ case 0x280: /* ʀ */ ph(Ph::R); break;
    case 's': ph(Ph::S); break;
    case 0x161: case 0x160: /* š */ case 0x283: /* ʃ */ ph(Ph::SH); break;
    case 0x17e: case 0x17d: /* ž */ case 0x292: /* ʒ */ ph(Ph::ZH); break;
    case 0x2a7: /* ʧ */ ph(Ph::CH); break;
    case 0x2a4: /* ʤ */ ph(Ph::JH); break;
    case 'f': ph(Ph::F); break;
    case 'h': ph(Ph::H); break;
    case 'j': ph(Ph::J); break;
    case 'v': case 0x28b: /* ʋ */ ph(Ph::V); break;
    case 'w': ph(Ph::W); break;
    case 0x279: /* ɹ */ ph(Ph::RR); break;
    case 0x3b8: /* θ */ ph(Ph::TH); break;
    case 0xf0: /* ð */ ph(Ph::DH); break;
    case 'x': ph(Ph::K); ph(Ph::S); break;
    case 'z': ph(Ph::T); ph(Ph::S); break;
    case ':': case 0x2d0: /* ː */ lengthen(); break;
    case ' ': case '\t': case '\n': case '\r': pause(1); break;  // 15 ms
    case ',': case ';': pause(13); breathPoint(false); break;     // 195 ms (+ breath)
    case '.': case '!': case '?': pause(10); breathPoint(true); break;  // 150 ms + breath
    case '-': atWordStart_ = true; break;
    case 0x2c8: /* ˈ */ stressMark_ = 2; explicitStress_ = true; break;
    case 0x2cc: /* ˌ */
      // Secondary stress also acts as a boundary: a longer pause before it.
      for (int i = 0; i < kSecondaryStressPauseStops; ++i) push(Ph::Stop);
      stressMark_ = 1;
      explicitStress_ = true;
      break;
    default: break;  // unknown symbols are ignored
  }
}

void Synth::push(Ph ph, bool wordStart) {
  if (!hasSpoken_ && ph != Ph::Stop) {
    // Breathe in before the first sentence; the marks set for this phoneme
    // wait until after it.
    const uint8_t mark = stressMark_;
    const float pitch = nextPitch_, pitchEnd = nextPitchEnd_;
    const Shape shape = nextShape_;
    stressMark_ = 0;
    nextPitch_ = nextPitchEnd_ = kNoPitch;
    nextShape_ = Shape();
    breathPoint(true);
    stressMark_ = mark;
    nextPitch_ = pitch;
    nextPitchEnd_ = pitchEnd;
    nextShape_ = shape;
  }
  if (ph != Ph::Stop) spokeSinceBreath_ = true;
  uint8_t stress = stressMark_;
  stressMark_ = 0;
  const float pitch = nextPitch_, pitchEnd = nextPitchEnd_;
  nextPitch_ = nextPitchEnd_ = kNoPitch;
  const Shape shape = nextShape_;
  nextShape_ = Shape();
  if (hasPending_) {
    Token& p = pending_;
    // English: the same consonant across a word boundary ("this century",
    // "bus stop") is one long consonant too (two s of 116 and 143 ms were
    // heard as "a little funny").
    const bool sameAcross = wordStart && aspirate_ && ph == p.ph && !isVowel(ph) && p.ph != Ph::Stop && !p.isLong;
    if ((!wordStart || sameAcross) && p.ph != Ph::Stop) {
      if (ph == p.ph && !p.isLong) {
        p.isLong = true;  // "kk", "aa": one long phoneme
        p.doubled = true;
        if (sameAcross) p.wordJoin = true;
        p.stress = std::max(p.stress, stress);
        return;
      }
      if (!spellingRules_) {
        nextShape2_ = shape;  // the next sound's own formants (an English vowel)
        nextWordStart_ = wordStart;
        emit(p, &ph);
        pending_ = Token{ph, false, wordStart, stress, false, kNoPitch, Shape()};
  pending_.word = curWord_;
        pending_.word = curWord_;
        pending_.pitch = pitch;
        pending_.pitchEnd = pitchEnd;
        pending_.shape = shape;
        return;
      }
      if (p.ph == Ph::N && ph == Ph::G) {  // "ng" is a long ŋ
        p.ph = Ph::NG;
        p.isLong = true;
        p.stress = std::max(p.stress, stress);
        return;
      }
      if ((p.ph == Ph::T && ph == Ph::SH) || (p.ph == Ph::D && ph == Ph::ZH)) {
        p.ph = p.ph == Ph::T ? Ph::CH : Ph::JH;  // "tʃ" = ʧ, "dʒ" = ʤ
        return;
      }
      if (p.ph == Ph::N && ph == Ph::K) p.ph = Ph::NG;  // "nk" = ŋk
      if (p.ph == Ph::N && (ph == Ph::P || ph == Ph::B)) p.ph = Ph::M;
    }
    nextShape2_ = shape;
    nextWordStart_ = wordStart;
    emit(p, &ph);
  }
  pending_ = Token{ph, false, wordStart, stress, false, kNoPitch, Shape()};
  pending_.word = curWord_;
  pending_.pitch = pitch;
  pending_.pitchEnd = pitchEnd;
  pending_.shape = shape;
  hasPending_ = true;
}

// Air a phoneme class takes (cm^3/s), for breath planning: the simulated
// drain per class over 40 s of kennedy.txt (see emit). The English reader
// plans its breaths over the rest of the sentence with the same table.
float Synth::airRate(Ph ph) {
  const Cls cls = info(ph).cls;
  return cls == Cls::Stop ? 0.0f
         : cls == Cls::H ? 550.0f
         : cls == Cls::Fricative ? (aeroOf(ph).ag > kAgVoice ? 170.0f : 280.0f)
         : cls == Cls::Plosive ? (aeroOf(ph).ag > kAgVoice ? 90.0f : 180.0f)
                               : 310.0f;
}
float Synth::airScale() const { return tract_ == Tract::Line ? kLineAirScale : 1.0f; }
float Synth::lungMax() const { return kLungMax; }
float Synth::lungFloor() const { return kLungRest; }
float Synth::lungEmpty() const { return kLungEmpty; }

void Synth::breathPoint(bool sentence, bool paragraph, bool snatched, bool planned) {
  if (!sentence && !planned && planVolume_ > kCommaBreathBelow) return;
  if (hasSpoken_ && !spokeSinceBreath_) return;  // "...": one breath is enough
  spokeSinceBreath_ = false;
  const float deficit = std::max(kLungMax - planVolume_, 0.0f);
  const float ms = std::clamp(deficit / kInhaleRate * 1000.0f, kInhaleMinMs,
                              sentence ? kInhaleMaxMs : kCatchInhaleMaxMs) *
                   (paragraph ? kParagraphBreathScale : sentence ? kSentenceBreathScale : kCatchBreathScale);
  nextInhaleMs_ = ms;
  nextInhaleSnatched_ = snatched;
  // ... and a short silent hold after it before the first sound (the inhale
  // ran up to it: its noise just before a vowel was heard as ð, "a" -> "the").
  const int stops = static_cast<int>(std::ceil(ms / 15.0f)) + kBreathHoldStops;
  for (int i = 0; i < stops; ++i) push(Ph::Stop);
  atWordStart_ = true;
  planVolume_ = kLungMax;
  hasSpoken_ = true;
}

void Synth::lengthen() {
  if (hasPending_) pending_.isLong = true;
}

void Synth::finish() {
  if (hasPending_) {
    nextShape2_ = Shape();
    nextWordStart_ = false;
    emit(pending_, nullptr);
    hasPending_ = false;
  }
  planSentence();
  finished_ = true;
}

// ---------------------------------------------------------------------------
// Phonemes -> keyframes and events.

// Gives the F1-transparent keyframes since the last vowel (l r d) their F1:
// on the line from the last vowel's F1 to target at tTarget (< 0: hold).
// target < 0: leave them as they are.
void Synth::resolveF1(float target, double tTarget) {
  if (target > 0 && !f1Pending_.empty()) {
    float last = target;
    for (Keyframe& kf : k_) {
      size_t j = 0;
      while (j < f1Pending_.size() && f1Pending_[j].first != kf.t) ++j;
      if (j == f1Pending_.size()) continue;
      // After a vowel: mostly that vowel's F1 (1 - kRAfterVowelF1 of the way
      // to the next); without one: kRInitialF1 of the way from its own.
      const float from = f1Prev_ > 0 ? f1Prev_ : kf.p[kF1];
      const bool line = f1Pending_[j].second < 0;
      float s;
      if (line)  // between the vowels, by time (without a vowel before: the next one's)
        s = f1Prev_ > 0 && tTarget > f1PrevT_
                ? static_cast<float>(std::clamp((kf.t - f1PrevT_) / (tTarget - f1PrevT_), 0.0, 1.0))
                : 1.0f;
      else
        s = f1Prev_ > 0 ? 1.0f - kRAfterVowelF1 : f1Pending_[j].second;
      if (tTarget < 0) s = 0;  // no vowel follows: hold
      float f = from + s * (target - from);
      // The nasal murmur's own low formant: real m n ŋ sit 0.2-0.45 octave
      // under the vowels around them at their centre (thirteen speakers);
      // on the bare line the trough fell at the nasal's end, not its middle.
      if (line) f *= kNasalF1Dip;
      kf.p[kF1] = f;
      last = f;
    }
    last_[kF1] = last;
  }
  f1Pending_.clear();
}

void Synth::addKeyframe(const Params& p, double t, bool vowel) {
  if (f1Transparent_) f1Pending_.push_back({t, f1MinShare_});
  if (rangeTok_) rangePending_.push_back({t, rangeTok_->ph, {p[kF1], p[kF2], p[kF3]}, f1Transparent_});
  // The very first keyframe is doubled so the first segment has a predecessor.
  if (k_.empty()) k_.push_back({p, t, vowel});
  k_.push_back({p, t, vowel});
  k_.back().ph = curPh_;
  if (k_.size() == 2) k_.front().ph = curPh_;
}

// A target held from a to b (shares of the phoneme's duration): the edge
// keyframes hold the voicing level, pitch and the rest, while the formants
// pass through one point at the centre (splineAt skips the edges for them),
// so they never stand still: holding them made each phoneme a plateau and
// the paths a staircase of plateaus and quick moves.
// Each target gets two points, this share of the held span apart: through
// one point the formants turned before reaching the targets (open vowels'
// F1 7-11 % low, i's F2 6 %, consonants likewise; female Whisper WER
// 10 -> 15 %; vowel points only: 12.6 %, both: 9.9 %). Between two equal
// points the spline still bends (it rises past them and turns), no plateau.
constexpr double kVowelPointSpread = 0.7, kConsonantPointSpread = 1.0;
// A consonant's formant targets are held from this share of its duration to
// that: reached as it starts and left inside it, as in the recordings (the
// transition into a nasal happens in the vowel's last 60 ms; with 0.2-0.8 it
// came 60 ms late and the F2 rise out of a consonant ran 2.4x faster than the
// fall into it; 0-0.6: 1.3x, real speakers 0.6-1.3; round trip unchanged).
constexpr double kConsonantHeldA = 0.0, kConsonantHeldB = 0.6;
constexpr double kRHeldA = 0.3;  // English ɹ: its colour needs a moment to settle
// Coda l: F1 and F2 relative to the vowel before it (2^-0.2, 2^-0.35).
constexpr float kCodaLF1 = 0.87f, kCodaLF2 = 0.78f;  // (-0.2 and -0.36 oct; readers' coda l F1 -0.07..-0.26, F2 -0.29..-0.45)
void Synth::addHeld(const Params& p, double t0, double dur, double a, double b, bool vowel) {
  addKeyframe(p, t0 + a * dur, vowel);
  k_.back().edge = kFormantPointTargets;
  const double c = 0.5 * (a + b),
               w = 0.5 * (b - a) * (vowel ? kVowelPointSpread : kConsonantPointSpread);
  if (kFormantPointTargets) {
    addKeyframe(p, t0 + (c - w) * dur, vowel);
    addKeyframe(p, t0 + (c + w) * dur, vowel);
  } else {
    addKeyframe(p, t0 + c * dur, vowel);
  }
  addKeyframe(p, t0 + b * dur, vowel);
  k_.back().edge = kFormantPointTargets;
}

// An event that only changes the breath pressure goes into the queue at its
// own time (it does not conflict with the airway movements, whose order
// addEvent keeps by moving later ones).
void Synth::addEventAt(const Event& e) {
  auto it = events_.end();
  while (it != events_.begin() && (it - 1)->t > e.t) --it;
  events_.insert(it, e);
}

void Synth::addEvent(const Event& e) {
  // Movements must start in order: a later phoneme's movement starting
  // before an earlier one's would be overridden by it.
  Event ev = e;
  if (!events_.empty()) ev.t = std::max(ev.t, events_.back().t);
  events_.push_back(ev);
}


const char* symbolOf(Ph ph) {
  static const char* const kSymbols[] = {
      "a", "e", "i", "o", "u", "y", "æ", "ø", "ə", "ɪ", "ɛ", "ʊ", "ɔ", "ʌ", "ɑ", "ɒ", "ɜ",
      "p", "t", "k", "b", "d", "g",
      "ʧ", "ʤ", "m", "n", "ŋ", "l", "r", "j", "ʋ", "ɹ", "w", "s", "ʃ", "ʒ", "f", "z", "θ", "ð",
      "v", "h", "_"};
  static_assert(sizeof(kSymbols) / sizeof(kSymbols[0]) == static_cast<size_t>(Ph::Count));
  return kSymbols[static_cast<int>(ph)];
}

void Synth::addLabel(const Token& tok) {
  // Debug: PUHUJA_WORDLAB=file writes the start of every word (samples on the
  // timeline; the first line is the first label's time, the labels' zero).
  static FILE* words = [] {
    const char* f = std::getenv("PUHUJA_WORDLAB");
    return f ? std::fopen(f, "w") : nullptr;
  }();
  if (words && labelStarts_.empty()) std::fprintf(words, "first %.0f\n", timeline_);
  if (words && tok.wordStart) std::fprintf(words, "%.0f\n", timeline_);
  std::string s = symbolOf(tok.ph);
  if (tok.isLong && tok.ph != Ph::Stop) s += "ː";
  // Consecutive pauses form one pause.
  if (tok.ph == Ph::Stop && !labelStarts_.empty() && labelStarts_.back().second == "_") return;
  labelStarts_.push_back({timeline_, s});
  labelWordStarts_.push_back(tok.ph == Ph::Stop ? 0 : tok.wordJoin ? 2 : tok.wordStart ? 1 : 0);
  labelWords_.push_back(tok.ph == Ph::Stop ? -1 : tok.word);
}

std::vector<Synth::Label> Synth::labels() const {
  std::vector<Label> out;
  const double start = outputStart_ < 0 ? 0 : outputStart_;
  for (size_t i = 0; i < labelStarts_.size(); ++i) {
    double a = labelStarts_[i].first;
    double b = i + 1 < labelStarts_.size() ? labelStarts_[i + 1].first : timeline_;
    out.push_back({std::max(a - start, 0.0) / fs_, std::max(b - start, 0.0) / fs_,
                   labelStarts_[i].second, labelWordStarts_[i] != 0, labelWordStarts_[i] == 2, labelWords_[i]});
  }
  return out;
}


void Synth::emit(const Token& tok, const Ph* next) {
  Info in = info(tok.ph);
  curPh_ = tok.ph;
  {
    const Cls cl = in.cls;
    rangeTok_ = formantRanges_ && (cl == Cls::Nasal || cl == Cls::Liquid || cl == Cls::Trill ||
                                   cl == Cls::Glide || cl == Cls::Fricative)
                    ? &tok : nullptr;
  }
  if (tok.shape.f1 > 0) in.f1 = tok.shape.f1;
  if (tok.shape.f2 > 0) in.f2 = tok.shape.f2;
  if (tok.shape.f3 > 0) in.f3 = tok.shape.f3;
  in.ms *= tok.shape.length;
  in.av *= tok.shape.level;
  if (!inAffricate_) addLabel(tok);
  if (tok.wordStart) {
    syllable_ = 0;
    prevWasVowel_ = false;
  }
  // A stress mark stands before the syllable's onset consonant; it applies
  // to the next vowel.
  if (tok.stress) curStress_ = tok.stress;
  // F1 of a word-initial r comes from the vowels around it (the jaw is free):
  // its keyframes are drawn when the next vowel comes.
  {
    // Only a word-initial r: l and d keep their own F1 (its dip is a cue;
    // with F1 running through l r d the round trip went 16.5% -> 21.2%,
    // "bed" "bad" "cat" all heard as "that").
    const bool rInitial = (tok.ph == Ph::RR || tok.ph == Ph::R) && tok.wordStart;
    // Nasals keep the surrounding F1 too (the jaw stays with the vowels).
    const bool f1Line = tok.ph == Ph::M || tok.ph == Ph::N || tok.ph == Ph::NG;
    const bool f1Clear = rInitial || f1Line;
    if (in.cls != Cls::Vowel && !f1Clear) {
      resolveF1(f1Prev_, -1);  // no vowel follows: hold the last vowel's F1
      f1Prev_ = -1;
    }
    f1Transparent_ = f1Clear;
    // A word-initial r keeps kRAfterVowelF1 of the previous vowel's F1, or
    // without one takes kRInitialF1 of the next vowel's.
    f1MinShare_ = rInitial ? kRInitialF1 : -1.0f;  // < 0: on the line between the vowels
  }
  if (tok.ph == Ph::CH || tok.ph == Ph::JH) {
    emitAffricate(tok, next);
    return;
  }
  if (in.cls == Cls::Plosive) {
    emitPlosive(tok, next);
    prevWasVowel_ = false;
    prevWasPlosive_ = true;
    prevVoiced_ = false;
    return;
  }

  const bool newSyllable = in.cls == Cls::Vowel && !prevWasVowel_;
  if (newSyllable) {
    sylStress_ = curStress_;
    curStress_ = 0;
  }
  float lengthMul = !tok.isLong ? 1.0f
                    : !tok.doubled ? kLongFactor
                    : isVowel(tok.ph) ? voice_.doubledVowel
                                      : kDoubledConsonantFactor;
  lengthMul *= voice_.rate;
  if (inAffricate_)  // (ʤ's frication part is ʒ: a little longer, "generosity")
    lengthMul *= kAffricateFricationLength * (tok.ph == Ph::ZH ? kJhFricationLength : 1.0f) *
                 (aspirate_ && tok.ph == Ph::SH ? kEnglishChFricationLength : 1.0f);
  if (in.cls == Cls::Vowel) {
    if (sylStress_ == 2) lengthMul *= kPrimaryStressLength;
    if (sylStress_ == 1) lengthMul *= kSecondaryStressLength;
  }
  const double t0 = timeline_;
  // An absolute duration from shapeNext() replaces the table's and all
  // lengthening rules (the English reader computes its own durations).
  // (A consonant doubled across words, "this century": the reader's length
  // is one consonant's; the doubling factor applies on top.)
  const double dur = (tok.shape.ms > 0 ? tok.shape.ms * voice_.rate * (tok.doubled && !isVowel(tok.ph) ? kDoubledConsonantFactor : 1.0f)
                                       : in.ms * lengthMul) * 0.001 * fs_;
  float planAir = 0;
  {
    // Air this sound will use, for breath planning: the simulated drain per
    // class over 40 s of kennedy.txt (cm^3/s). With vowels planned at 210
    // the plan ended a 7-second sentence at 420 cm^3 while the lungs were at
    // 50: no breath was taken at its "and", and its last words faded.
    const float flow = airRate(tok.ph) * airScale();
    planAir = flow * static_cast<float>(dur / fs_);
    planVolume_ -= planAir * drainScale_;  // (the scale learned from the simulated drain)
  }
  Params p = last_;
  auto setTargets = [&] {
    p[kF1] = in.f1;
    p[kF2] = in.f2;
    p[kF3] = in.f3;
    p[kB1] = in.b1;
    p[kB2] = in.b2;
    p[kB3] = in.b3;
    p[kFNZ] = in.fnz;
    p[kAV] = in.av;
    p[kTrill] = in.cls == Cls::Trill ? 1.0f : 0.0f;
    // Consonants: the resonances are not what is being made, so the tract is
    // held with less tension and they are broader (they glide with the
    // formants into and out of the vowels).
    if (in.cls != Cls::Vowel)
      for (int k = 0; k < 3; ++k) p[kB1 + k] *= kConsonantBw;
  };
  // Airway movement into this sound, centred on its start. After a plosive
  // the release already moves into this sound's targets.
  const float ms = 0.001f * fs_;
  Event e{std::max(t0 - 0.5 * kBoundaryMs * ms, phonemeStart_ + 1)};
  e.planAir = planAir;
  Aero aero = aeroOf(tok.ph);
  e.ac = aero.ac;
  e.lineX = linePlaceOf(tok.ph);
  e.acMs = kBoundaryMs;
  e.ag = aero.ag;
  if (in.cls == Cls::Vowel) e.ag *= tok.shape.press;  // voice quality by prosody
  e.agMs = kGlottalMs;
  e.aspGain = in.cls == Cls::H ? hAspirationGain_ * kHGain : kAspirationGain * kAspirationTiltGain;
  e.creak = 0;  // a creak lasts until the next sound
  // After an aspirated release the glottis is already closing slowly into
  // this sound; a quick move here would cut the aspiration short.
  if (aspirating_) {
    e.ag = -1;
    aspirating_ = false;
  }
  e.an = aero.an;
  e.anMs = kBoundaryMs;
  e.speech = in.cls == Cls::Stop ? 0.0f : 1.0f;
  if (prevWasPause_ && in.cls != Cls::Vowel && in.cls != Cls::Stop) {
    // After a pause the tongue and lips take their place before the voice
    // starts (no vowel-like onset before "they"); the glottis opens on time.
    Event pre{t0 - kPreposeMs * ms};
    pre.ac = e.ac;
    pre.acMs = kBoundaryMs;
    pre.an = e.an;
    pre.anMs = kBoundaryMs;
    addEvent(pre);
    e.ac = e.an = -1;
    // An open glottis (h, s, f...) is open already before the sound starts,
    // as after breathing in: opened at the sound's start, the sudden airflow
    // was a low thump that made "He" heard as "They".
    if (e.ag > kAgBreathy) {
      // ...and its noise swells in (goog.wav's initial f: over ~100 ms).
      e.speechAttackMs = kVoicelessOnsetMs;
      Event open{t0 - kPreposeMs * ms};
      open.ag = e.ag;
      open.agMs = kGlottalMs;
      addEvent(open);
      e.ag = -1;
    }
  }

  switch (in.cls) {
    case Cls::Vowel: {
      setTargets();
      // Vowel chains (diphthongs, Finnish "ai", "uo"): a vowel followed by
      // another aims a little past its target, away from the next one
      // (unsharp masking in log frequency), so the smooth transition does
      // not blur them together. The second vowel is left as it is.
      const float own[3] = {p[kF1], p[kF2], p[kF3]};
      for (int k = 0; k < 3; ++k) {
        float f = own[k];
        if (next && isVowel(*next) && *next != tok.ph) {
          const float nf = nextVowelF(*next, k);
          f *= std::clamp(std::pow(own[k] / nf, kVowelSharpen), 0.75f, 1.33f);
        }
        p[kF1 + k] = f;
        lastVowelF_[k] = own[k];
      }
      // English schwa: no target of its own, it takes much of its shape from
      // its neighbours (the tuned ə, F2 ~1800, is a coronal-context value;
      // between ɪ and a dark l, "loyalty", it was an extra up-down step).
      if (darkL_ && tok.ph == Ph::Schwa && !tok.stress) {
        const float before = last_[kF2];
        const float after = next ? (isVowel(*next) ? nextVowelF(*next, 1)
                                    : *next == Ph::L ? kPullF2DarkL : info(*next).f2) : before;
        p[kF2] += kSchwaContext * (0.5f * (before + after) - p[kF2]);
      }
      // English: a vowel next to w, a dark l or r does not reach its full F2
      // ("welcome": ɛ between w and l at F2 ~1480 in goog.wav; at its full
      // target, 1776, it jumped forward between them).
      if (darkL_) {
        auto pull = [&](Ph c) {
          const float f2 = c == Ph::W ? kPullF2W : c == Ph::L ? kPullF2DarkL
                           : c == Ph::RR ? kPullF2R : 0.0f;
          if (f2 > 0) p[kF2] += kVowelContextPull * (f2 - p[kF2]);
        };
        if (!prevWasVowel_) pull(lastPh_);
        if (next) pull(*next);
        if (lastPh_ == Ph::J && (tok.ph == Ph::U || tok.ph == Ph::UH))
          p[kF2] += kUAfterJ * (info(Ph::J).f2 - p[kF2]);
      }
      // The consonants since the last vowel: on the line to this one.
      {
        const float nf[3] = {p[kF1], p[kF2], p[kF3]};
        resolveRanges(nf, t0 + 0.25 * dur);
      }
      // F1 through the l r d since the last vowel (see above).
      resolveF1(p[kF1], t0);
      // After an h that took the vowel's shape: give it this vowel's actual
      // formants (with the shared table's, "man holds" wobbled: F2 down in
      // the h, up at the vowel's start, down again into its glide).
      if (hKeyframeT_ >= 0) {
        for (Keyframe& kf : k_)
          if (kf.t == hKeyframeT_)
            for (int k = 0; k < 3; ++k) kf.p[kF1 + k] = p[kF1 + k];
        for (int k = 0; k < 3; ++k) last_[kF1 + k] = p[kF1 + k];
        hKeyframeT_ = -1;
      }
      // Diphthongs (vowel after vowel) stay in the same syllable.
      if (newSyllable) ++syllable_;
      const bool pitched = tok.pitch != kNoPitch;
      if (pitched) {
        p[kF0Mul] = std::exp2(tok.pitch / 12.0f);
        // First pitched vowel after a pause: the pause and the consonants
        // since take the new phrase's pitch (a little below it), instead of
        // the last phrase's low ending, so the phrase does not start with a
        // swoop up from there.
        if (pauseStartT_ >= 0) {
          const float onset = p[kF0Mul] * kPhraseOnsetDrop;
          for (Keyframe& kf : k_)
            if (kf.t >= pauseStartT_) kf.p[kF0Mul] = onset;
          last_[kF0Mul] = onset;
          pauseStartT_ = -1;
          pitchAnchorT_ = -1;
        }
      } else if (explicitStress_) {
        p[kF0Mul] = sylStress_ == 2   ? kPrimaryStressF0
                    : sylStress_ == 1 ? kSecondaryStressF0
                                      : 1.0f;
      } else {
        p[kF0Mul] = syllable_ <= 1 ? kFirstSyllableF0 : 1.0f;
      }
      // Glottal onset (English vowel-initial accented word at a phrase
      // start): the folds are pressed together just before the vowel and
      // released into it, a soft catch that marks the word's start.
      if (glottalOnsetNext_) {
        glottalOnsetNext_ = false;
        Event press{t0 - kGlottalOnsetMs * 0.001 * fs_};
        press.ag = kGlottalisedAg;
        press.agMs = 5;
        addEvent(press);
        Event release{t0 + kGlottalOnsetMs * 0.001 * fs_};
        release.ag = in.cls == Cls::Vowel ? kAgVoice * tok.shape.press : -1;
        release.agMs = kGlottalOnsetMs;
        pendingRelease_ = release;
        hasPendingRelease_ = true;
        e.ag = -1;  // the release opens the glottis, not this phoneme's event
      }
      // Creaky ending: the folds press together over the second half of the
      // vowel (low, irregular pulses), as real voices often do at a low
      // statement end.
      if (creakNext_) {
        creakNext_ = false;
        Event creak{t0 + kCreakStart * dur};
        creak.ag = kCreakAg;
        creak.agMs = kCreakMs;
        // Slack and irregular, not only pressed: the micro-variation grows
        // and the breath pressure eases (pressing alone squeezed the voice).
        creak.creak = 1;
        creak.press = kCreakPress;
        creak.pressMs = kCreakMs;
        pendingCreak_ = creak;
        hasPendingCreak_ = true;
      }
      // Micro-intonation: after a voiceless consonant the vowel starts a
      // little higher (the folds are still stiff), after a voiced obstruent
      // a little lower; the onset keyframe settles into the vowel's pitch.
      {
        const bool voicelessBefore = lastPh_ == Ph::P || lastPh_ == Ph::T || lastPh_ == Ph::K ||
                                     lastPh_ == Ph::CH || lastPh_ == Ph::S || lastPh_ == Ph::SH ||
                                     lastPh_ == Ph::F || lastPh_ == Ph::TH;  // (not h: often voiced, "who" jumped)
        const bool voicedBefore = lastPh_ == Ph::B || lastPh_ == Ph::D || lastPh_ == Ph::G ||
                                  lastPh_ == Ph::JH || lastPh_ == Ph::Z || lastPh_ == Ph::ZH || lastPh_ == Ph::VV ||
                                  lastPh_ == Ph::DH;
        if (voicelessBefore || voicedBefore) {
          // Only the pitch differs: the rest is where the transition from
          // the consonant would be at that moment.
          Params onset = p;
          for (int i = 0; i < kNumParams; ++i) onset[i] = last_[i] + 0.2f * (p[i] - last_[i]);
          onset[kF0Mul] = p[kF0Mul] * (voicelessBefore ? kOnsetRiseVoiceless : kOnsetDipVoiced);
          addKeyframe(onset, t0 + 0.05 * dur, true);
          k_.back().f0Mod = onset[kF0Mul] / p[kF0Mul];
          k_.back().edge = kFormantPointTargets;  // pitch only: not a formant point
        }
      }
      // After a nasal, l, r or a glide the voicing level is at the vowel's
      // within kVowelLevelRiseMs, not at a quarter of the vowel (in a long
      // vowel, "dare not", the rise took ~100 ms and the vowel began weak).
      // The other parameters stay on their way to the vowel's targets.
      if (!prevWasVowel_ && (info(lastPh_).cls == Cls::Nasal || info(lastPh_).cls == Cls::Liquid ||
                             info(lastPh_).cls == Cls::Glide || info(lastPh_).cls == Cls::Trill)) {
        const double rise = kVowelLevelRiseMs * ms;
        if (rise < 0.25 * dur - 1) {
          const float s = static_cast<float>(rise / (0.25 * dur));
          Params q = p;
          for (int i = 0; i < kNumParams; ++i) q[i] = last_[i] + s * (p[i] - last_[i]);
          q[kAV] = p[kAV];
          addKeyframe(q, t0 + rise, true);
          k_.back().edge = kFormantPointTargets;  // the voicing level only
        }
      }
      // Targets over the middle half (formants: a point at the centre).
      addHeld(p, t0, dur, 0.25, 0.75, true);
      f1Prev_ = p[kF1];
      f1PrevT_ = t0 + 0.75 * dur;
      for (int k = 0; k < 3; ++k) rangePrevF_[k] = p[kF1 + k];
      rangePrevT_ = t0 + 0.75 * dur;
      // Pitch line (English melody): the pitch is anchored at the vowel
      // centres and moves steadily between them, through the consonants.
      // Holding each vowel's pitch over its middle half made a rise over
      // several syllables a staircase. Keyframes after this vowel's centre
      // take its pitch until the next vowel redraws them.
      if (pitched) {
        // A vowel that falls to a second pitch anchors its first early, so
        // the fall spans most of it (see below).
        const double tc = t0 + (tok.pitchEnd != kNoPitch ? kFallAnchorAt : 0.5) * dur;
        const float target = p[kF0Mul];
        for (Keyframe& kf : k_) {
          if (pitchAnchorT_ < 0 || kf.t <= pitchAnchorT_) {
            if (kf.t > tc) kf.p[kF0Mul] = target * kf.f0Mod;  // after the centre
            continue;
          }
          const float f = kf.t >= tc ? 1.0f : static_cast<float>((kf.t - pitchAnchorT_) / (tc - pitchAnchorT_));
          kf.p[kF0Mul] = (pitchAnchor_ + f * (target - pitchAnchor_)) * kf.f0Mod;
        }
        last_[kF0Mul] = target;
        pitchAnchorT_ = tc;
        pitchAnchor_ = target;
        // A second pitch (a final accented word falling): a steady fall from
        // the anchor to kFallEndAt of the vowel, and the next pitch line
        // starts there. (Set only from the vowel's 3/4 on, the whole fall
        // happened between the keyframes at 0.675 and 0.75: a 4 st drop in
        // 13 ms, smoothed by the pitch lag into 80 ms, heard pulse by pulse
        // as steps at a male voice's 70-85 Hz: "now" as "o-o-o-w-w-w-w".)
        if (tok.pitchEnd != kNoPitch) {
          const float end = std::exp2(tok.pitchEnd / 12.0f);
          const double te = t0 + kFallEndAt * dur;
          for (Keyframe& kf : k_) {
            if (kf.t <= tc) continue;
            const float f = kf.t >= te ? 1.0f : static_cast<float>((kf.t - tc) / (te - tc));
            kf.p[kF0Mul] = (target + f * (end - target)) * kf.f0Mod;
          }
          last_[kF0Mul] = end;
          p[kF0Mul] = end;  // what the next sound (a pause) starts from: last_ = p below
          pitchAnchorT_ = te;
          pitchAnchor_ = end;
        }
      } else {
        pitchAnchorT_ = -1;
      }
      break;
    }
    case Cls::Nasal:
    case Cls::Liquid:
    case Cls::Trill:
      setTargets();
      coarticulate(p, tok.ph, next);
      // An l after the r-coloured vowel ("world", "girl", "curl"): the r
      // colour stays into it. goog.wav's "world" holds F3 at ~1400 Hz from
      // the vowel's end to the d; puhuja's l took its own high F3, so F3
      // swept 1600 -> 2800 Hz in 80 ms between the vowel and the l, an edge
      // heard as artificial. Held at the r's F3 through the whole l, the
      // sweep moved to the d (1700 -> 2700 in 60 ms; "too much formant
      // activity around r and l", 2026-10-30); the male readings (Kokoro,
      // ElevenLabs) let F3 rise through the l, 1750 -> 2650 over 70-100 ms.
      // The l now aims half-way (geometric mean), so each sweep is half.
      if (tok.ph == Ph::L && (lastPh_ == Ph::ER || lastPh_ == Ph::RR)) p[kF3] = std::sqrt(std::min(p[kF3], last_[kF3]) * p[kF3]);
      // A coda l (after a vowel, before a consonant or a pause: "world",
      // "all forms") keeps most of the vowel's jaw and darkens: LPC on three
      // readers, the l's middle against its vowel, F1 -0.07..-0.26 oct and
      // F2 -0.29..-0.45; puhuja's own targets gave -0.38 and -0.06 (an onset
      // l, "li la lo", is set by the coming vowel through coarticulate()).
      if (tok.ph == Ph::L && prevWasVowel_ && !(next && isVowel(*next))) {
        p[kF1] = std::max(p[kF1], last_[kF1] * kCodaLF1);
        p[kF2] = last_[kF2] * kCodaLF2;  // relative to the vowel, not the fixed dark-l F2 ("help": 1700 -> 1000 was a leap)
      }
      addHeld(p, t0, dur, kConsonantHeldA, kConsonantHeldB);
      break;
    case Cls::Glide:
      setTargets();
      if (tok.ph == Ph::RR) {  // English ɹ: the r colour is held, not passed through
        coarticulate(p, tok.ph, next);  // F3 (the r colour) stays low
        addHeld(p, t0, dur, kRHeldA, kConsonantHeldB);
      } else {
        // (Holding a post-consonantal glide's targets from its start, as
        // the consonants are, put the j's F2 top at the release like a
        // reader's, but cost 3 WER points on both voices: not done.)
        coarticulate(p, tok.ph, next);
        addKeyframe(p, t0 + 0.5 * dur);
      }
      break;
    case Cls::Fricative:
      // A narrow constriction with an open glottis: the flow through it
      // becomes turbulent noise, and the pressure behind it stops the voice.
      setTargets();
      coarticulate(p, tok.ph, next);
      e.setNoise = true;
      e.shape = fricShape(tok.ph);
      if (tok.ph == Ph::DH || tok.ph == Ph::TH || tok.ph == Ph::F || tok.ph == Ph::VV)
        e.noiseFadeMs = kSoftNoiseOnsetMs;
      e.noiseGain = in.noise * fricationGain_;
      // The per-phoneme level scales the heard noise, after the turbulence
      // ceiling: scaling the gain before it hardly changes a saturated θ.
      e.noisePost = tok.shape.level;
      e.noiseTract = fricTract(tok.ph).share;
      e.noiseTractGain = kBurstTractNorm[fricTract(tok.ph).place];
      // A voiced fricative after a voiceless sound ("belief that"): its much
      // stronger noise gain (the voiced constriction has little pressure
      // behind it) comes in once the glottis has closed. Together with the
      // voiceless sound's full pressure it burst out (x4 in 4 ms, a snap).
      if ((tok.ph == Ph::DH || tok.ph == Ph::Z || tok.ph == Ph::ZH || tok.ph == Ph::VV) &&
          !prevVoiced_ && !prevWasPause_ && !prevWasPlosive_) {
        pendingNoise_ = e;
        pendingNoise_.t = std::max(e.t, t0 + kGlottalMs * ms);
        pendingNoise_.ac = pendingNoise_.ag = pendingNoise_.an = -1;
        pendingNoise_.speech = -1;
        pendingNoise_.press = -1;
        pendingNoise_.aspGain = -1;
        pendingNoise_.expand = -1;
        pendingNoise_.inhaleMs = 0;
        pendingNoise_.pop = -1;
        e.setNoise = false;
        hasPendingNoise_ = true;
      }
      addHeld(p, t0, dur, kConsonantHeldA, kConsonantHeldB);
      break;
    case Cls::H: {
      // Breath noise through the vocal tract. Before a vowel the mouth takes
      // the vowel's shape early in the h, so the breath noise is that vowel
      // whispered and runs into it (goog.wav's "hands": F2 at the vowel's
      // during the h). Otherwise no keyframes of its own: the formants glide
      // straight from the previous to the next sound. Between voiced sounds
      // h is partly voiced.
      if (hTakesVowel_ && next && isVowel(*next)) {
        Params hp = last_;
        hp[kF1] = nextVowelF(*next, 0);
        hp[kF2] = nextVowelF(*next, 1);
        hp[kF3] = nextVowelF(*next, 2);
        hp[kAV] *= tok.shape.level;  // its voicing is breathy and weak (English h level)
        addKeyframe(hp, t0 + kHVowelAt * dur);
        for (int k = 0; k < 3; ++k) p[kF1 + k] = hp[kF1 + k];  // the vowel starts from this shape
        // The vowel's own formants (an English variant, a diphthong's start)
        // are known only when it comes: it redraws this keyframe then.
        hKeyframeT_ = t0 + kHVowelAt * dur;
      }
      // h before j ("human", "huge"): the tongue is already up for the j,
      // so the breath is a palatal hiss [ç] (kPalatalHShape), voiceless
      // even after a voiced sound (goog.wav "of human": ~120 ms of noise,
      // no voice; voiced and shapeless it was heard as "ovuman").
      const bool beforeJ = hTakesVowel_ && next && *next == Ph::J;
      if (beforeJ) {
        Params hp = last_;
        hp[kF1] = nextVowelF(*next, 0);
        hp[kF2] = nextVowelF(*next, 1);
        hp[kF3] = nextVowelF(*next, 2);
        hp[kAV] *= tok.shape.level;
        addKeyframe(hp, t0 + kHVowelAt * dur);
        for (int k = 0; k < 3; ++k) p[kF1 + k] = hp[kF1 + k];
        e.ac = kPalatalHAc;
        e.setNoise = true;
        e.shape = kPalatalHShape;
        e.noiseGain = fricationGain_;
        e.noisePost = tok.shape.level * kPalatalHLevel;
        e.noiseTract = fricTract(Ph::SH).share;
        e.noiseTractGain = kBurstTractNorm[fricTract(Ph::SH).place];
      }
      bool voicedH = prevVoiced_ && next && isSonorant(*next) && !beforeJ;
      if (voicedH) {
        e.ag = voicedHAg_;
        e.aspGain *= voicedHNoise_;
        // Its voicing is weaker still: with the vowel's shape and level it was
        // an approximant ("hand" as "land").
        for (Keyframe& kf : k_)
          if (kf.t >= hKeyframeT_ - 1 && kf.t <= hKeyframeT_ + 1) kf.p[kAV] *= kVoicedHVoice;
        p[kAV] *= kVoicedHVoice;
      }
      break;
    }
    case Cls::Stop:
      // The glottis closes; the mouth stays as it was (a trill keeps rolling
      // until the airflow runs out).
      if (!prevWasPause_) {
        // Before the pause the breath pressure eases off: the phrase ends
        // softer, a little lower and less bright.
        Event ease{t0 - kEndEaseMs * 0.001 * fs_};
        ease.press = kEndPressure;
        ease.pressMs = kEndEaseMs;
        addEventAt(ease);  // in time order: queued after later events, it was moved late
      }
      e.ac = -1;
      e.inhaleMs = nextInhaleMs_;
      e.inhaleSnatched = nextInhaleSnatched_;
      nextInhaleMs_ = 0;
      nextInhaleSnatched_ = false;
      e.agMs = kStopGlottalMs;
      syllable_ = 0;
      resolveRanges(nullptr, -1);  // no vowel follows: they hold the last one's
      rangePrevT_ = -1;
      addKeyframe(p, t0 + 0.25 * dur, true);
      addKeyframe(p, t0 + 0.75 * dur, true);
      break;
    case Cls::Plosive:
      break;
  }
  if (prevWasPause_ && in.cls != Cls::Stop) {  // speaking again: full pressure
    e.press = 1.0f;
    e.pressMs = kPressResumeMs;
  }
  if (!prevWasPlosive_) addEvent(e);
  if (hasPendingNoise_) {
    addEvent(pendingNoise_);
    hasPendingNoise_ = false;
  }
  if (hasPendingCreak_) {
    addEvent(pendingCreak_);
    hasPendingCreak_ = false;
  }
  if (hasPendingRelease_) {  // after this phoneme's own airway event
    addEvent(pendingRelease_);
    hasPendingRelease_ = false;
  }
  last_ = p;
  phonemeStart_ = t0;
  timeline_ += dur;
  prevWasVowel_ = in.cls == Cls::Vowel;
  prevWasPlosive_ = false;
  prevWasPause_ = in.cls == Cls::Stop;
  // The new phrase's pitch takes over from the last 15 ms of the pause (from
  // its first, read ahead, the old phrase's ending turned upward).
  if (in.cls == Cls::Stop) pauseStartT_ = t0 + 0.5 * dur;
  if (in.cls == Cls::Stop) pitchAnchorT_ = -1;  // the pitch line restarts after a pause
  prevVoiced_ = isSonorant(tok.ph);
  lastPh_ = tok.ph;
  rangeTok_ = nullptr;
}

// An affricate is a plosive released straight into a short fricative at the
// same place: ʧ = closure + postalveolar burst + short ʃ, ʤ likewise with ʒ.
void Synth::emitAffricate(const Token& tok, const Ph* next) {
  const Ph fric = tok.ph == Ph::CH ? Ph::SH : Ph::ZH;
  emitPlosive(tok, &fric);  // a fricative next: no aspiration, no voice onset
  prevWasPlosive_ = true;
  inAffricate_ = true;
  // The English reader times the closure (tok.shape.ms); the frication
  // follows it in proportion, so a shortened affricate (an unaccented
  // "which") shortens as a whole. (With the table's length the frication
  // stayed 92 ms whatever the reader asked for.)
  Shape fs;
  if (tok.shape.ms > 0) fs.ms = tok.shape.ms * kAffricateFricationPerClosure;
  emit(Token{fric, false, false, 0, false, kNoPitch, fs}, next);
  inAffricate_ = false;
}

// Coarticulation: the parts of the mouth a consonant does not use take part
// of the neighbouring vowels' shape, so its formants move partway toward
// them. Each formant follows its own articulator, so each has its own share:
// F1 mostly the jaw (openness), F2 the tongue body (front/back), F3 the
// tongue tip and the lip rounding. Shares of the next and the previous vowel
// per formant; the previous vowel's share is smaller (anticipation is
// stronger than carry-over), except where the voice-over recordings show
// otherwise. h has no targets of its own anyway.
namespace {
struct Coart {
  float next[3], prev[3];  // F1 F2 F3
};
// English l (voice-over reference): F2 follows the next vowel a little (l
// before a vowel is clear) and hardly the previous one (after a vowel it is
// dark: the tongue back holds its own position).
// F1: an onset l's jaw is the coming vowel's ("li la lo": LPC on three readers,
// the l's F1 -0.03..+0.24 oct against its vowel; with the share at 0.2
// puhuja's was -0.32; 0.9 put the l within 0.1 oct of the vowel but "human life" lost its l, "ife": 0.6).
constexpr Coart kDarkL = {{0.6f, 0.15f, 0}, {0.15f, 0.1f, 0}};
Coart coartOf(Ph ph) {
  switch (ph) {
    // Lips and teeth only: the whole tongue is free (F2, F3 follow); the
    // jaw stays fairly closed for the lip-teeth contact (F1 less).
    case Ph::V:
    case Ph::VV:
    case Ph::F:  return {{0.2f, 0.7f, 0.5f}, {0.2f, 0.4f, 0.3f}};
    // Tongue tip between the teeth: the body is partly free, the tip (F3)
    // and the jaw (F1) are not.
    case Ph::TH:
    case Ph::DH: return {{0.1f, 0.25f, 0.05f}, {0.05f, 0.35f, 0.25f}};
    // Grooved tongue blade, jaw high: little is free.
    case Ph::S:
    case Ph::Z:  return {{0.15f, 0.2f, 0}, {0.2f, 0.25f, 0}};  // jaw and body follow the vowels ("first")
    // Raised blade and rounded lips: F2 and F3 are the noise's own shape.
    case Ph::SH:
    case Ph::ZH: return {{0.05f, 0.05f, 0}, {0.03f, 0.1f, 0}};
    // Clear l (Finnish): tip on the ridge, the body is free.
    case Ph::L:  return {{0.2f, 0.3f, 0.05f}, {0.1f, 0.2f, 0.03f}};
    // Trill: the tip vibrates, the body and jaw are free.
    case Ph::R:  return {{0.25f, 0.3f, 0.15f}, {0.15f, 0.2f, 0.1f}};
    // ɹ: the bunched tongue is the r; F3 (the r colour) stays low.
    case Ph::RR: return {{0.25f, 0.25f, 0}, {0.25f, 0.25f, 0}};
    // j: the tongue body is the glide (F2 fixed); the jaw is freer.
    case Ph::J:  return {{0.1f, 0.05f, 0.05f}, {0.05f, 0.02f, 0.03f}};
    // w: lips and tongue back are the glide (F2 fixed); jaw and tip freer.
    case Ph::W:  return {{0.15f, 0.05f, 0.1f}, {0.1f, 0.02f, 0.05f}};
    // Nasals: the tongue keeps its place, but the jaw follows the vowels a
    // little (the voice-over's m n: F1 slope 0.1-0.3), and for m the free
    // tongue a little too.
    case Ph::M:  return {{0.15f, 0, 0}, {0.2f, 0.1f, 0}};
    case Ph::N:
    case Ph::NG: return {{0.15f, 0, 0}, {0.2f, 0, 0}};
    default:     return {{0, 0, 0}, {0, 0, 0}};
  }
}
}  // namespace


// English consonants: how far each formant may move from its tuned value
// (the coarticulated target, which carries the place cue: the low F2 of m
// and f, the low F3 of ɹ) toward the line between the vowels around it.
// Inside the band the formant takes the shortest path from vowel to vowel;
// the consonant does not swing to its exact target and back.
constexpr float kRangeTolerance[3] = {0.12f, 0.06f, 0.04f};  // share of the value, F1 F2 F3
constexpr float kRangeToleranceLF1 = 0.8f;  // l: its F1 follows the vowels (readers: an onset l within 0.2 oct of its vowel)

// Redraws the consonant keyframes since the last vowel, now that the next
// vowel (nextF at tNext; nullptr: a pause, none) is known: each formant on
// the line from the previous vowel's end to the next vowel's start (or the
// one vowel there is), kept within the tolerance of its tuned value.
void Synth::resolveRanges(const float* nextF, double tNext) {
  if (rangePending_.empty()) return;
  const bool hasPrev = rangePrevT_ >= 0;
  for (Keyframe& kf : k_)
    for (const RangePending& rp : rangePending_) {
      if (kf.t != rp.t) continue;
      for (int k = rp.f1Line ? 1 : 0; k < 3; ++k) {  // (the F1 of m n ŋ and an initial ɹ: resolveF1)
        float f;
        if (hasPrev && nextF && tNext > rangePrevT_) {
          const float w = static_cast<float>(std::clamp((kf.t - rangePrevT_) / (tNext - rangePrevT_), 0.0, 1.0));
          f = rangePrevF_[k] + w * (nextF[k] - rangePrevF_[k]);
        } else if (nextF) {
          f = nextF[k];
        } else if (hasPrev) {
          f = rangePrevF_[k];
        } else {
          continue;
        }
        const float c = rp.centre[k];
        // l's jaw is free: an onset l takes the coming vowel's F1 (see kDarkL).
        const float tol = rp.ph == Ph::L && k == 0 ? kRangeToleranceLF1 : kRangeTolerance[k];
        kf.p[kF1 + k] = std::clamp(f, c * (1 - tol), c * (1 + tol));
      }
      // ɹ: F2 stays clearly below its low F3.
      if (rp.ph == Ph::RR) kf.p[kF2] = std::min(kf.p[kF2], kf.p[kF3] - 250.0f);
    }
  // The sound after them starts from where they ended.
  if (!k_.empty() && k_.back().t == rangePending_.back().t)
    for (int k = rangePending_.back().f1Line ? 1 : 0; k < 3; ++k) last_[kF1 + k] = k_.back().p[kF1 + k];
  rangePending_.clear();
}

float Synth::tableFormant(Ph ph, int k) {
  const Info& in = info(ph);
  return k == 0 ? in.f1 : k == 1 ? in.f2 : in.f3;
}

// The next vowel's formant k: its own (English) value when the next sound
// carries one, else the shared table's.
float Synth::nextVowelF(Ph next, int k) const {
  const float own = k == 0 ? nextShape2_.f1 : k == 1 ? nextShape2_.f2 : nextShape2_.f3;
  if (own > 0) return own;
  return k == 0 ? info(next).f1 : k == 1 ? info(next).f2 : info(next).f3;
}

void Synth::coarticulate(Params& p, Ph ph, const Ph* next) const {
  const Coart c = ph == Ph::L && darkL_ ? kDarkL : coartOf(ph);
  // English l with no vowel after it ("world", "all", "ill"): dark, the
  // tongue back raised; F2 ~1000 (goog: F2 ~1100 in "world", nothing above
  // 1.4 kHz; at the shared 1300 it had a peak at 1.6 kHz).
  if (ph == Ph::L && darkL_ && !(next && isVowel(*next))) p[kF2] = kPullF2DarkL;
  for (int k = 0; k < 3; ++k) {
    float f = p[kF1 + k];
    if (next && isVowel(*next)) {
      const float nf = nextVowelF(*next, k);
      float share = c.next[k];
      // An r with no vowel before it: its F2 may move toward the vowel
      // (kRInitialF2; its F1 is drawn by resolveF1).
      if ((ph == Ph::RR || ph == Ph::R) && !prevWasVowel_ && k == 1) share = std::max(share, kRInitialF2);
      f += share * (nf - p[kF1 + k]);
    }
    if (prevWasVowel_ && lastVowelF_[k] > 0) f += c.prev[k] * (lastVowelF_[k] - p[kF1 + k]);
    p[kF1 + k] = f;
  }
  // ɹ: its low F3 is the r colour; F2 stays clearly below it.
  if (ph == Ph::RR) p[kF2] = std::min(p[kF2], p[kF3] - 250.0f);
}

void Synth::emitPlosive(const Token& tok, const Ph* next) {
  PlosiveInfo pi = plosiveInfo(tok.ph);
  // English b not between vowels ("I said bat", "big"): a mostly devoiced
  // closure and a clear release. Fully voiced with almost no burst it was
  // heard as m or not at all (language researcher: b words 2/9 -> 7/9).
  const bool englishB = aspirate_ && tok.ph == Ph::B && !prevWasVowel_;
  // English b d g before a voiceless fricative ("said ship"): the closure
  // devoices, so the voicing does not run on into the hiss (heard as "fib").
  const bool devoiceBefore = aspirate_ && pi.voiced && next &&
                             (*next == Ph::F || *next == Ph::TH || *next == Ph::S || *next == Ph::SH);
  if (devoiceBefore) pi.closureAg = kAgVoice + kDevoiceBeforeFric * (kAgOpen - kAgVoice);
  // English k: a stronger burst (the Finnish calibration cut it 13 dB;
  // "carrot" was heard as "parrot", "cow" as "power").
  // English ʧ: the release opens fast, so the hiss starts abruptly (built
  // up over 25 ms it sounded like a fricative: "choose" -> "sure").
  if (aspirate_ && tok.ph == Ph::CH) pi.releaseMs = kEnglishChReleaseMs;
  // English word-final d into a vowel-initial word: released fully (see below).
  const bool linkedD = aspirate_ && tok.ph == Ph::D && nextWordStart_ && next && isVowel(*next);
  if (linkedD) pi.releaseMs *= kLinkedDReleaseSlow;
  if (aspirate_ && tok.ph == Ph::K) {
    pi.noiseGain *= kEnglishKBurst;
    pi.releaseMs = kEnglishKReleaseMs;  // a longer, compact velar burst ("cow", "carrot")
  }
  if (tok.shape.flap) {  // a tap: a quick touch, hardly a release
    pi.noiseGain *= kFlapRelease;
    pi.pop *= kFlapRelease;
  }
  if (englishB) {
    pi.closureAg = kAgVoice + kEnglishBOpening * (kAgOpen - kAgVoice);
    pi.noiseGain = kEnglishBNoise;
    pi.pop *= kEnglishBPop;
  }
  float prevF2 = last_[kF2];
  float nextF2 = next && isVowel(*next) ? nextVowelF(*next, 1) : prevF2;
  double closure = (tok.shape.ms > 0 ? tok.shape.ms
                                      : (tok.isLong ? pi.longClosureMs : pi.closureMs)) *
                   voice_.rate * 0.001 * fs_;
  // A voiceless plosive before ð or θ ("that the", "at this") is glottalised
  // in English: a short closure with creaky (pressed, weak) voicing instead
  // of an open glottis, so no pressure builds behind the tongue and the
  // release into the dental fricative is soft.
  const bool glottalised = !pi.voiced && next && (*next == Ph::DH || *next == Ph::TH);
  if (glottalised) closure *= kGlottalisedClosure;
  // d before ð θ ("around the") is dental: the tongue moves to the teeth and
  // runs into the fricative with a short closure and no real burst. Fully
  // released, it snapped (a 20 dB jump at the ð).
  const bool dentalD = tok.ph == Ph::D && next && (*next == Ph::DH || *next == Ph::TH);
  if (dentalD) closure *= kDentalDClosure;
  // Before another plosive ("cannot do", "that time") the first is not
  // released: its closure runs into the next one's. Released, its burst
  // was a lone click in the silence (+24 dB).
  const bool unreleased = next && info(*next).cls == Cls::Plosive;
  if (unreleased) closure *= kUnreleasedClosure;
  const bool afterUnreleased = prevUnreleased_;
  prevUnreleased_ = unreleased;
  double release = timeline_ + closure;
  const float ms = 0.001f * fs_;
  const bool nextIsPause = !next || *next == Ph::Stop;
  const Aero nextAero = next ? aeroOf(*next) : aeroOf(Ph::Stop);

  // Closure start: formants bend toward the consonant from the sound before.
  Params a = last_;
  a[kF1] = 250;
  a[kFNZ] = kNasalPoleHz;
  a[kTrill] = 0;
  a[kAV] = 1;
  plosiveLocus(tok.ph, prevF2, &a[kF2], &a[kF3]);
  // The tongue tip closes for t d while the body keeps an r colour ("world",
  // "hard"): F3 stays down into the closure (goog.wav: ~1400 Hz up to the
  // d of "world"); rising to the locus it swept 1600 -> 2700 Hz in 40 ms
  // at the l's end.
  if ((tok.ph == Ph::D || tok.ph == Ph::T) && last_[kF3] < a[kF3]) a[kF3] = last_[kF3];
  // ... and after an l the body stays where the l left it: F2 keeps the l's
  // value into the closure ("holds": the alveolar locus pulled F2 from 700
  // to 1500 Hz during the last 40 ms of the l, too much activity).
  if ((tok.ph == Ph::D || tok.ph == Ph::T) && lastPh_ == Ph::L) a[kF2] = last_[kF2];
  addKeyframe(a, timeline_);

  // Release: formants start from the locus toward the following vowel.
  Params b = a;
  plosiveLocus(tok.ph, nextF2, &b[kF2], &b[kF3]);
  addKeyframe(b, release);
  last_ = b;

  // Closing: the lips or tongue approach fast, then seal slowly, reaching
  // full closure as the closure starts. As the opening narrows the sound
  // fades, pressure builds up behind it and the voice dies away.
  const double closeStart =
      std::max(timeline_ - (kApproachMs + kSealMs) * ms, phonemeStart_ + 1);
  const float closeMs = static_cast<float>((timeline_ - closeStart) / ms);
  const float sealMs = std::min(kSealMs, closeMs);
  if (closeMs > sealMs + 1 && !afterUnreleased) {  // (closed already after an unreleased one)
    Event approach{closeStart};
    approach.ac = kSealArea;
    approach.acMs = closeMs - sealMs;
    addEvent(approach);
  }
  Event close{timeline_ - sealMs * ms};
  // Speaking again after a pause: full breath pressure (as emit() does for
  // the other classes; a plosive returns before that, so every phrase that
  // began with b d g p t k after a pause kept the ending's eased pressure,
  // 4-6 dB under: "..., but from the hand of God").
  if (prevWasPause_) {
    close.press = 1.0f;
    close.pressMs = kPressResumeMs;
  }
  close.speech = 1;  // speaking again (after a pause the level had faded)
  close.ac = 0;
  close.acMs = sealMs;
  close.an = 0;
  close.anMs = closeMs;
  close.setNoise = true;
  // The front cavity of k/g also keeps part of the preceding vowel's shape.
  const float burstF2 = prevWasVowel_ ? (1.0f - kBurstPrevVowel) * nextF2 + kBurstPrevVowel * prevF2
                                      : nextF2;
  close.shape = burstShape(place(tok.ph), burstF2);
  close.noiseGain = pi.noiseGain;
  // Most of the burst comes from the cavity in front of the closure; a part
  // excites the cavity behind it too and takes the tract's resonances.
  close.noiseTract = kBurstTractShare[place(tok.ph)] + (pi.voiced ? kVoicedBurstTractExtra : 0.0f);
  close.noiseTractGain = kBurstTractNorm[place(tok.ph)];
  if (tok.ph == Ph::D) {
    // d: a softer, darker release than t, with a lift around 6 kHz for some
    // tongue-tip brightness.
    close.shape = {200, 10000, 6000, 4000, 5.0f};  // (10: a bright click, "do" snapped)
  }
  close.pop = pi.pop;
  if (unreleased) {
    close.pop = 0;
    close.noiseGain = 0;
  }
  // A voiced plosive after an unreleased one ("cannot do"): its release
  // comes out of the shared closure, and a full burst after the silent
  // closure snapped. After a nasal ("undoing", "can do") the release is
  // kept: there it is what makes the d heard.
  // English word-final d released into a vowel-initial word ("world is",
  // "and all"): a full burst, the transient that starts the next word
  // (goog.wav: flat noise ~12 dB below the vowel; here it was ~20 dB down,
  // a voiced tap, and "is" had no onset).
  if (linkedD && !unreleased) {
    close.noiseGain *= kLinkedDRelease;
    close.pop *= kLinkedDRelease;
  }
  if (pi.voiced && afterUnreleased && !dentalD) {
    close.pop *= kVoicedAfterConsonantRelease;
    close.noiseGain *= kVoicedAfterConsonantRelease;
  }
  if (dentalD) {
    close.pop = pi.pop * kDentalDRelease;
    close.noiseGain = pi.noiseGain * kDentalDRelease;
  }
  if (glottalised) {
    close.pop = pi.pop * kGlottalisedRelease;
    close.noiseGain = pi.noiseGain * kGlottalisedRelease;
  }
  if ((pi.voiced && !englishB && !devoiceBefore) || glottalised) {
    // Voiced (or creaky) closure: the speaker expands the mouth cavity
    // (larynx down, cheeks slack), so air keeps flowing and the folds keep
    // vibrating.
    close.expand = kVoicedExpansion;
    close.expandMs = closeMs;
  }
  addEvent(close);
  // Glottis during the closure: opens for voiceless plosives, vibrating
  // position for voiced ones (also needed after a pause, when it is shut).
  Event glottis{timeline_};
  glottis.ag = glottalised ? kGlottalisedAg
               : pi.closureAg >= 0 ? pi.closureAg : pi.voiced ? kAgVoice : kAgOpen;
  glottis.agMs = glottalised ? kGlottalMs
                 : pi.voiced ? kGlottalMs : 0.6f * static_cast<float>(closure / ms);
  addEvent(glottis);

  // Release: the closure opens toward the next sound. The trapped air rushes
  // out as the burst; the glottis closes again and voicing restarts once the
  // pressure has fallen.
  Event rel{release};
  rel.expand = 1;
  rel.expandMs = kVoicedExpansionReleaseMs;
  rel.ac = nextIsPause ? kAcOpen : nextAero.ac;
  rel.acMs = pi.releaseMs;
  // Into a dental fricative ("that the", "around the") the tongue slides
  // from the ridge to the teeth: a slow opening, so the ð noise grows in
  // (opened in 12 ms it rose 45 dB in 8 ms, a snap).
  if (glottalised || dentalD) rel.acMs = std::max(rel.acMs, kDentalReleaseMs);
  if (unreleased) rel.ac = 0;  // stays closed into the next plosive
  rel.an = nextAero.an;
  rel.anMs = kBoundaryMs;
  if (nextIsPause) {
    rel.ag = 0;
    rel.agMs = 40;
  } else if (glottalised) {
    rel.ag = nextAero.ag;  // straight into voicing for ð
    rel.agMs = kGlottalMs;
  } else if (aspirate_ && !pi.voiced && !glottalised && next && lastPh_ != Ph::S &&
             (isVowel(*next) || info(*next).cls == Cls::Liquid || info(*next).cls == Cls::Glide ||
              info(*next).cls == Cls::Trill)) {
    // English aspiration: the glottis stays open past the release and closes
    // slowly; voicing starts some 50-70 ms later, after a breathy puff.
    rel.ag = nextAero.ag;
    rel.agMs = curStress_ ? kAspiratedAdductMs : kAspiratedUnstressedMs;
    rel.aspBurst = true;  // the puff's breath noise (x aspirationGain_), fading
    // A word that carries its stress on the syllable this stop opens
    // ("tiger", "tyranny", "power"): a longer, stronger puff. The user: the
    // more powerful the word, the more it gains from an aspirated p t k.
    if (tok.wordStart && curStress_ == 2) {
      rel.agMs *= kStrongAspirationLength;
      rel.aspBurstGain = kStrongAspirationGain;
    }
    aspirating_ = true;
  } else if (!pi.voiced && pi.adductLeadMs > 0) {
    // The glottis closes during the end of the closure, overlapping the
    // release, so voicing restarts soon after it.
    Event adduct{release - std::min<double>(pi.adductLeadMs * ms, 0.5 * closure)};
    adduct.ag = nextAero.ag;
    adduct.agMs = pi.adductMs;
    addEvent(adduct);
  } else {
    rel.ag = nextAero.ag;
    rel.agMs = pi.voiced ? kGlottalMs : pi.adductMs;
  }
  addEvent(rel);
  if (next && info(*next).cls == Cls::Fricative) {
    // ks, ts, ʧ: the burst turns into the fricative's noise.
    Event fr{release + 0.5 * rel.acMs * ms};
    fr.setNoise = true;
    // Into ð θ f v the noise grows in slowly (from t's or d's closure into
    // "the" it snapped).
    if (*next == Ph::DH || *next == Ph::TH || *next == Ph::F || *next == Ph::VV)
      fr.noiseFadeMs = kPlosiveToSoftNoiseMs;
    fr.shape = fricShape(*next);
    // ʤ: its noise reaches lower than a plain ʒ's (more of the ~1-2 kHz
    // part, "pledge").
    if (tok.ph == Ph::JH) fr.shape = kJhNoiseShape;
    fr.noiseGain = info(*next).noise * fricationGain_ *
                   (tok.ph == Ph::JH ? kJhFricationBoost : tok.ph == Ph::CH ? kChFricationBoost : 1.0f);
    if (tok.ph == Ph::CH) fr.noisePost = kChFricationLevel;  // after the turbulence ceiling
    addEvent(fr);
  }
  phonemeStart_ = timeline_;
  timeline_ = release;
}

// ---------------------------------------------------------------------------
// Rendering.

// Sets up the cubic for the segment k_[1] -> k_[2]. Tangents are the
// Catmull-Rom finite differences, divided by time so that uneven keyframe
// spacing (short vowels next to long stops) does not cause overshoot kinks.
bool Synth::startSegment() {
  // Drop keyframes whose segment is already done.
  while (k_.size() >= 3 && t_ >= k_[2].t) k_.pop_front();
  bool haveNext = k_.size() >= 4;
  if (k_.size() < 3 || (!haveNext && !finished_)) return false;

  const Keyframe& k0 = k_[0];
  const Keyframe& k1 = k_[1];
  const Keyframe& k2 = k_[2];
  const Keyframe& k3 = haveNext ? k_[3] : k_[2];  // clamp at the end

  segStart_ = k1.t;
  if (outputStart_ < 0) outputStart_ = std::max(t_, segStart_);
  segLen_ = std::max(k2.t - k1.t, 1.0);
  if (t_ < segStart_) t_ = segStart_;
  double h = segLen_;
  double d02 = std::max(k2.t - k0.t, 1.0);
  double d13 = std::max(k3.t - k1.t, 1.0);
  for (int i = 0; i < kNumParams; ++i) {
    float p1 = k1.p[i], p2 = k2.p[i];
    // Tangents scaled to the [0,1] parameter of this segment.
    float m1 = static_cast<float>((k2.p[i] - k0.p[i]) / d02 * h);
    float m2 = static_cast<float>((k3.p[i] - k1.p[i]) / d13 * h);
    if (i == kAV) {
      // The voicing level is kept monotone between keyframes (flat at a
      // keyframe that is a local extreme, slopes limited to 3x the secant):
      // a quiet sound after a loud one (n after ɹ, "dare not") dipped below
      // its own level, and the vowel after it rose from there.
      const float d = p2 - p1;
      if ((p1 - k0.p[i]) * d <= 0) m1 = 0;
      if ((k3.p[i] - p2) * d <= 0) m2 = 0;
      m1 = std::clamp(m1, -3 * std::abs(d), 3 * std::abs(d));
      m2 = std::clamp(m2, -3 * std::abs(d), 3 * std::abs(d));
    }
    c_[0][i] = p1;
    c_[1][i] = m1;
    c_[2][i] = -3 * p1 + 3 * p2 - 2 * m1 - m2;
    c_[3][i] = 2 * p1 - 2 * p2 + m1 + m2;
  }
  inSegment_ = true;
  return true;
}

// The keyframe spline (as in startSegment) at any queued time t: used to
// read the formant path ahead of the current sample. Past the last keyframe
// it holds the last value.
float Synth::splineAt(int i, double t) const {
  const size_t n = k_.size();
  if (n == 0) return 0;
  if (i >= kF1 && i <= kF3) {
    // Formants: Catmull-Rom through the keyframes that are not held
    // targets' edges (see addHeld).
    size_t idx[4];
    int m = 0;
    size_t j = n;  // the last point at or before t
    for (size_t q = 0; q < n; ++q) {
      if (k_[q].edge) continue;
      if (k_[q].t <= t) j = q;
      else break;
    }
    auto prevPoint = [&](size_t q) {
      while (q > 0) {
        --q;
        if (!k_[q].edge) return q;
      }
      return n;
    };
    auto nextPoint = [&](size_t q) {
      for (++q; q < n; ++q)
        if (!k_[q].edge) return q;
      return n;
    };
    if (j == n) {  // before the first point
      const size_t f = nextPoint(static_cast<size_t>(-1));
      return f < n ? k_[f].p[i] : k_[0].p[i];
    }
    const size_t j2 = nextPoint(j);
    if (j2 == n) return k_[j].p[i];  // past the last point: hold it
    const size_t j0 = prevPoint(j), j3 = nextPoint(j2);
    idx[0] = j0 < n ? j0 : j;
    idx[1] = j;
    idx[2] = j2;
    idx[3] = j3 < n ? j3 : j2;
    (void)m;
    const Keyframe& k0 = k_[idx[0]];
    const Keyframe& k1 = k_[idx[1]];
    const Keyframe& k2 = k_[idx[2]];
    const Keyframe& k3 = k_[idx[3]];
    const double h = std::max(k2.t - k1.t, 1.0);
    const float s = static_cast<float>((t - k1.t) / h);
    const float p1 = k1.p[i], p2 = k2.p[i];
    const float m1 = static_cast<float>((k2.p[i] - k0.p[i]) / std::max(k2.t - k0.t, 1.0) * h);
    const float m2 = static_cast<float>((k3.p[i] - k1.p[i]) / std::max(k3.t - k1.t, 1.0) * h);
    const float c2 = -3 * p1 + 3 * p2 - 2 * m1 - m2, c3 = 2 * p1 - 2 * p2 + m1 + m2;
    return p1 + s * (m1 + s * (c2 + s * c3));
  }
  if (t <= k_[0].t) return k_[0].p[i];
  size_t j = 0;
  while (j + 1 < n && k_[j + 1].t <= t) ++j;
  if (j + 1 >= n) return k_[n - 1].p[i];
  const Keyframe& k0 = k_[j > 0 ? j - 1 : j];
  const Keyframe& k1 = k_[j];
  const Keyframe& k2 = k_[j + 1];
  const Keyframe& k3 = k_[j + 2 < n ? j + 2 : j + 1];
  const double h = std::max(k2.t - k1.t, 1.0);
  const float s = static_cast<float>((t - k1.t) / h);
  const float p1 = k1.p[i], p2 = k2.p[i];
  const float m1 = static_cast<float>((k2.p[i] - k0.p[i]) / std::max(k2.t - k0.t, 1.0) * h);
  const float m2 = static_cast<float>((k3.p[i] - k1.p[i]) / std::max(k3.t - k1.t, 1.0) * h);
  const float c2 = -3 * p1 + 3 * p2 - 2 * m1 - m2, c3 = 2 * p1 - 2 * p2 + m1 + m2;
  return p1 + s * (m1 + s * (c2 + s * c3));
}

size_t Synth::pull(float* out, size_t n) {
#if defined(__SSE__) || defined(_M_X64)
  // Denormals flushed to zero while rendering (decaying filter states in
  // long pauses), whatever the caller has set; restored on return.
  const unsigned csr = _mm_getcsr();
  _mm_setcsr(csr | 0x8040);
  struct Restore {
    unsigned v;
    ~Restore() { _mm_setcsr(v); }
  } restore{csr};
#endif
  size_t i = 0;
  while (i < n) {
    if (!inSegment_ || t_ >= segStart_ + segLen_) {
      inSegment_ = false;
      if (!startSegment()) break;
      continue;
    }
    // The pitch and formant targets are read ahead: wait until the queued
    // keyframes cover that far (else the output depended on how the text
    // was split into pushes: up to 4 st at phrase starts).
    if (!finished_ && (k_.empty() || k_.back().t < t_ + kLookAheadSamples_ + 1)) break;
    out[i++] = tick();
    t_ += 1;
  }
  return i;
}

float Synth::noise() { return nextNoise(rng_); }

float Synth::tick() {
  // Start the articulator movements and noise changes that are due.
  while (!events_.empty() && events_.front().t <= t_) {
    const Event& e = events_.front();
    if (e.press >= 0) press_.start(e.press, msToSamples(e.pressMs, fs_));
    if (e.aspGain >= 0) aspGainTarget_ = e.aspGain;
    if (e.lineX >= 0) lineXTarget_ = e.lineX;
    planUsed_ += e.planAir;
    if (e.creak >= 0) creakTarget_ = e.creak;
    if (e.aspBurst) aspBurst_ = aspirationGain_ * e.aspBurstGain;
    if (e.ac >= 0) ac_.start(e.ac, msToSamples(e.acMs, fs_));
    if (e.ag >= 0) ag_.start(e.ag, msToSamples(e.agMs, fs_));
    if (e.an >= 0) an_.start(e.an, msToSamples(e.anMs, fs_));
    if (e.expand >= 0) expand_.start(e.expand, msToSamples(e.expandMs, fs_));
    if (e.speech >= 0) {
      speechTarget_ = e.speech;
      speechAttackNow_ = e.speechAttackMs > 0
                             ? 1.0f - std::exp(-1.0f / msToSamples(e.speechAttackMs, fs_))
                             : kSpeechAttack_;
    }
    if (e.inhaleMs > 0) {
      // The planner's table against the simulated drain since the last
      // breath: the next phrases are planned with the larynx's real thirst.
      if (planUsed_ > 150.0f) {
        const float ratio = actualUsed_ / planUsed_;
        drainScale_ = std::clamp(0.5f * (drainScale_ + ratio), 0.5f, 4.0f);
      }
      planUsed_ = actualUsed_ = 0;
      inhaleLen_ = inhaleLeft_ = msToSamples(e.inhaleMs, fs_);
      inhaleSnatched_ = e.inhaleSnatched;
      inhaleRate_ = std::max(kLungMax - lungVolume_, 0.0f) / inhaleLen_;
    }
    if (e.setNoise) {
      // The noise glides from where it is to the new sound's shape and
      // gains. While the noise is (nearly) silent the new shape is taken at
      // once and only the gains glide in.
      nFrom_ = nNow_;
      nTo_ = {e.shape, e.noiseGain, e.noiseTract, e.noiseTractGain, e.noisePost};
      // The voice's noise colours: a female voice's front cavities are shorter
      // (her s ʃ were dull, 8-12 kHz 5-10 dB under) and her ʃ stands out less
      // from her bright vowels.
      nTo_.shape.lo *= voice_.noiseScale;
      nTo_.shape.hi = std::min(nTo_.shape.hi * voice_.noiseScale, 0.45f * fs_);
      nTo_.shape.peakF *= voice_.noiseScale;
      nTo_.shape.peakBw *= voice_.noiseScale;
      if (e.shape.peakF > 2000 && e.shape.peakF < 3200) nTo_.gain *= voice_.shGain;  // ʃ ʒ
      const bool silent = nLevel_ < kNoiseSilent * nPeak_;
      if (silent) {
        nFrom_.shape = nTo_.shape;
        nf_.reset();
        nf_.set(nTo_.shape, fs_);
      }
      // From silence: fast for a burst or sibilant, softer for the weak
      // fricatives (ð θ f v; a 4 ms onset of ð after "that" snapped; real fricative onsets take 20-40 ms). Between
      // noisy sounds a big rise in strength (f -> ð: x18 while the pressure
      // behind the lips is still high) glides slower, as the glottis closes.
      const float fadeMs = silent ? (e.noiseFadeMs > 0 ? e.noiseFadeMs : kNoiseCrossfadeMs)
                           : nTo_.gain > 2.0f * nFrom_.gain ? kNoiseRiseGlideMs : kNoiseGlideMs;
      nGlide_ = 0;
      nGlideStep_ = 1.0f / msToSamples(fadeMs, fs_);
    }
    if (e.pop >= 0) popTarget_ = e.pop;
    events_.pop_front();
  }
  // Soft floor at 0 (C1): a hard max() puts a corner into the airflow.
  const float acRaw = ac_.tick();
  const float ac = acRaw >= kAcSoft ? acRaw
                   : acRaw <= -kAcSoft ? 0.0f
                                       : (acRaw + kAcSoft) * (acRaw + kAcSoft) / (4 * kAcSoft);
  const float ag = std::max(ag_.tick(), 0.0f);
  const float an = std::max(an_.tick(), 0.0f);
  const float expand = std::max(expand_.tick(), 1.0f);

  // Mouth pressure po (relative to the lungs), one implicit Euler step of
  //   dpo = flowGain * (ag * sqrt(1 - po) - (ac + an) * sqrt(po)),
  // solved by safeguarded Newton (in double, from the previous value; a step
  // leaving the bracket falls back to bisection): the step is stiff when the
  // mouth is wide open. f(p) increases monotonically on [0, 1].
  {
    const double out = ac + an, g = flowGain_ / expand, prev = po_;
    double lo = 0, hi = 1, p = std::clamp(prev, 1e-9, 1.0 - 1e-9);
    for (int i = 0; i < 30; ++i) {
      const double sq1 = std::sqrt(1.0 - p), sq0 = std::sqrt(p);
      const double f = p - prev - g * (ag * sq1 - out * sq0);
      if (f > 0) hi = p; else lo = p;
      const double df = 1.0 + g * (0.5 * ag / std::max(sq1, 1e-9) + 0.5 * out / std::max(sq0, 1e-9));
      double next = p - f / df;
      if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
      const bool done = std::fabs(next - p) < 1e-10;
      p = next;
      if (done) break;
    }
    po_ = static_cast<float>(p);
  }
  const float drop = 1.0f - po_;  // pressure drop across the glottis
  // Lung pressure from the air left.
  const float lungFactor =
      (kLungPressureSpent + (1.0f - kLungPressureSpent) *
                                std::clamp((lungVolume_ - kLungRest) / (kLungMax - kLungRest), 0.0f, 1.0f)) *
      smoothstep(0.0f, kLungEmpty, lungVolume_);

  // Phonation: the folds vibrate when nearly closed (but not pressed shut)
  // and the pressure drop across them is above the threshold.
  const float adduction =
      smoothstep(0.0f, 0.025f, ag) * (1.0f - smoothstep(0.07f, 0.2f, ag));
  const float vib =
      adduction * std::clamp((drop - kPhonationThreshold) / (1.0f - kPhonationThreshold),
                             0.0f, 1.0f);
  // The vibration builds up (logistic growth from a small seed) or dies
  // down toward what the airflow allows at this moment.
  if (vib > vibAmp_) {
    vibAmp_ = std::max(vibAmp_, kOnsetSeed * vib);
    vibAmp_ += kVibGrowth_ * vibAmp_ * (1.0f - vibAmp_ / vib);
  } else {
    vibAmp_ += (vib - vibAmp_) * kVibDecay_;
  }
  tension_ += (smoothstep(0.07f, 0.25f, ag) - tension_) * kTension_;
  const float f0Aero = 1.0f + kF0Pressure * (drop - 1.0f) + kF0Tension * tension_;

  // Turbulence noise: at the constriction from the mouth pressure (stronger
  // through a wider opening), at the glottis from the opening beyond the
  // vibrating position.
  // Below a minimum pressure the flow stays smooth: approximants (l r j v)
  // make no hiss.
  float fricAmp = lungFactor * po_ * std::sqrt(po_) * std::sqrt(ac / 0.1f) *
                  smoothstep(kTurbulenceOnset, 2.0f * kTurbulenceOnset, po_);
  // The shaped pulse runs the folds too, for everything but the pulse's
  // shape: the airflow, the mouth pressure, the lungs, the onset and offset
  // of the vibration, its weakening under a constriction and the breath
  // noise all come from the folds, so the stops and fricatives are the same
  // under either source (Jyrki: t k p d g b s z sh are made in the mouth).
  const bool runFolds = larynx_->physical();
  if (runFolds) {
    // The turbulence follows the airflow: with vibrating folds it comes in
    // puffs with each glottal pulse (the buzzy noise of ð z ʒ v), weighted
    // by how strongly the folds vibrate. (A hard gate on the mean flow
    // switched the pulsing on and off within a sample, x2.2 steps.)
    const float pulse = std::clamp(larynx_->flow() / std::max(larynx_->flowMean(), 20.0f), 0.0f, 2.5f);
    const float w = std::min(larynx_->vib() / kNoisePulseVib, 1.0f);
    fricAmp *= 1.0f + kNoisePulse * w * (pulse - 1.0f);
  }
  const float openness = std::sqrt(std::max(ag - kAgVoice, 0.0f) / (kAgOpen - kAgVoice));
  const float aspAmp = drop * std::sqrt(drop) * openness;
  aspGainNow_ += (aspGainTarget_ - aspGainNow_) * kAspGainFollow_;  // ~3 ms
  aspBurst_ = 1.0f + (aspBurst_ - 1.0f) * kAspBurstDecay_;  // the puff fades as the glottis closes
  // Release transient: the change of airflow out of the mouth.
  const float oralFlow = ac * std::sqrt(po_);
  popScale_ += (popTarget_ - popScale_) * kPopSmooth_;  // no steps in the transient's scale
  const float pop = (oralFlow - prevOralFlow_) * kPopGain * popScale_;
  prevOralFlow_ = oralFlow;

  // Evaluate the spline for every parameter at this sample.
  float s = static_cast<float>((t_ - segStart_) / segLen_);
  Params p;
  for (int i = 0; i < kNumParams; ++i)
    p[i] = c_[0][i] + s * (c_[1][i] + s * (c_[2][i] + s * c_[3][i]));
  float planned[4];
  const bool usePlan = motion_ == FormantMotion::Planned && planAt(t_ + kPlanTauMs * 0.001 * fs_, planned);
  if (motion_ != FormantMotion::Spline) {
    // Critically damped: x'' = w^2 (target - x) - 2 w x'. With a plan the
    // target is the planned path (read one time constant ahead) and the
    // follower only smooths the plan's 5 ms grid.
    const float dt = 1.0f / fs_;
    for (int k = 0; k < 3; ++k) {
      const float tauMs = usePlan ? kPlanTauMs : kFormantTauMs[k];
      const float w = 1000.0f / tauMs;
      if (!formInit_) formPos_[k] = p[kF1 + k];
      const double tauS = tauMs * 0.001 * fs_;
      const double ahead = t_ + kFormantLookAhead * tauS;
      float target;
      if (usePlan) {
        target = std::exp2(planned[k]);
      } else {
        target = splineAt(kF1 + k, ahead);
        target -= kFormantPreemphasis * (splineAt(kF1 + k, ahead + tauS) - 2.0f * target + splineAt(kF1 + k, ahead - tauS));
      }
      formVel_[k] += dt * (w * w * (target - formPos_[k]) - 2.0f * w * formVel_[k]);
      formPos_[k] += dt * formVel_[k];
      p[kF1 + k] = formPos_[k];
    }
    formInit_ = true;
  }
  {
    // Debug: PUHUJA_FTRACE=file writes F1-F3 every 5 ms (time as in -l labels).
    static FILE* trace = [] {
      const char* f = std::getenv("PUHUJA_FTRACE");
      return f ? std::fopen(f, "w") : nullptr;
    }();
    // Further columns: pitch (Hz, the command after its lag), the glottal
    // and oral openings (cm^2), the speech envelope, the noise level, B1,
    // the mouth pressure, the pressure drop, the lung volume, the folds'
    // loudness, their last cycle's peak and the drawn pulse's phase.
    if (trace && static_cast<long long>(t_) % static_cast<long long>(fs_ / 200) == 0)
      std::fprintf(trace, "%.4f\t%.1f\t%.1f\t%.1f\t%.3f\t%.2f\t%.4f\t%.4f\t%.3f\t%.4f\t%.1f\t%.3f\t%.3f\t%.0f\t%.2f\t%.2f\t%.2f\n",
                   (t_ - std::max(outputStart_, 0.0)) / fs_, p[kF1], p[kF2], p[kF3], p[kAV],
                   baseF0_ * pitch2_, ag, ac, speechEnv_, nLevel_, p[kB1], po_, drop, lungVolume_, larynx_->loudness(), larynx_->cyclePeak(), larynx_->pulsePhase());
  }

  // Splines may overshoot a little; keep values physically sensible.
  const float nyq = 0.45f * fs_;
  // The voice's formant scaling (vocal tract length) applies to all targets.
  // Glottal damping: while the folds are open, air escapes downward and F1
  // widens (physical source only).
  const float openB1 = runFolds
                           ? kGlottalB1 * std::min(larynx_->area() / 0.1f, 1.5f) : 0.0f;
  r1_.set(std::clamp(p[kF1] * voice_.formant[0], 100.0f, nyq),
          std::max(p[kB1] * voice_.bw[0], 20.0f) + openB1, fs_);
  // The open glottis damps every mode (each has its pressure maximum at the
  // glottis), so F2 and F3 widen too, a little less than F1.
  r2_.set(std::clamp(p[kF2] * voice_.formant[1], 100.0f, nyq),
          std::max(p[kB2] * voice_.bw[1], 20.0f) + kGlottalB23[0] * openB1, fs_);
  r3_.set(std::clamp(p[kF3] * voice_.formant[2], 100.0f, nyq),
          std::max(p[kB3] * voice_.bw[2], 20.0f) + kGlottalB23[1] * openB1, fs_);
  nz_.set(std::clamp(p[kFNZ], 100.0f, nyq), kNasalBw, fs_);
  {
    // F4 and F5 follow F3 and the pitch (see kF4FollowF3), kept above F3.
    const float f3 = std::clamp(p[kF3] / kF3Neutral, 0.6f, 1.4f);
    const float up = std::pow(std::max(pitch2_, 0.25f), kHighFormantPitch);
    const float f3Hz = p[kF3] * voice_.formant[2];
    const float f4 = std::max(voice_.f4 * std::pow(f3, kF4FollowF3) * up, f3Hz + 250.0f);
    const float f5 = std::max(voice_.f5 * std::pow(f3, kF5FollowF3) * up, f4 + 300.0f);
    const float s = voice_.highFormants;
    r4_.set(std::min(f4, nyq), kF4Bw * s, fs_);
    r5_.set(std::min(f5, nyq), kF5Bw * s, fs_);
  }

  // Trill (Finnish r): the airflow past the tongue tip makes it tap ~26 times
  // a second, interrupting the voice. It needs enough flow: as the flow
  // weakens the taps slow down and die away exponentially.
  const float trillDrive =
      runFolds
          ? std::clamp((larynx_->flowMean() - kTrillFlowMin) / (kTrillFlowFull - kTrillFlowMin), 0.0f, 1.0f)
          : vibAmp_;
  const float trillTarget = std::clamp(p[kTrill], 0.0f, 1.0f) * trillDrive;
  trillAmp_ += (trillTarget - trillAmp_) * (trillTarget > trillAmp_ ? kTrillAttack_ : kTrillDecay_);
  trillPhase_ += kTrillHz * (1.0f - kTrillSlowing * (1.0f - trillDrive)) / fs_;
  if (trillPhase_ >= 1.0f) trillPhase_ -= 1.0f;
  const float trill = trillAmp_;
  // At full depth each tap briefly closes the voice off completely.
  float trillMod = 1.0f - trill * 0.5f * (1.0f + std::cos(2.0f * kPi * trillPhase_));
  float av = std::max(p[kAV], 0.0f) * trillMod *
             (runFolds ? 1.0f : vibAmp_);  // folds: they decide

  if (pitch1_ <= 0) pitch1_ = pitch2_ = p[kF0Mul];
  // The pitch command is read a little ahead (like the formants), so the
  // laryngeal lag delays the melody less.
  {
    float pl[4];
    const double tp = t_ + kPitchLookAheadMs * 0.001 * fs_;
    const float cmd = motion_ == FormantMotion::Planned && planAt(tp, pl) ? std::exp2(pl[3]) : splineAt(kF0Mul, tp);
    pitch1_ += kPitchLag_ * (cmd - pitch1_);
  }
  pitch2_ += kPitchLag_ * (pitch1_ - pitch2_);
  float f0 = baseF0_ * pitch2_ * f0Aero * jitter_;
  float dt = f0 / fs_;
  phase_ += dt;
  if (phase_ >= 1.0f) {
    phase_ -= 1.0f;
    drift_ += kPitchDriftFollow * (kPitchDrift * noise() - drift_);
    jitter_ = 1.0f + drift_ + kPitchJitter * noise();
    duty_ = carrier_ == Carrier::Triangle ? kTriangleDuty + kTriangleDutyJitter * noise()
                                          : kSquareDuty;
  }
  // Air leaving the lungs (cm^3/s); the fold models replace this estimate.
  float glottalFlow = 4000.0f * ag * std::sqrt(drop) * lungFactor;
  // The larynx: the speaker's commands in, the voice source out.
  LarynxOutput lo;
  if (tract_ == Tract::Line) {
    // The line's shape: the vowel shape for the running formants, the
    // consonant's constriction (the oral opening ac at its place) and the
    // velum; updated every fourth sample (the articulators are slow).
    lineX_ += (lineXTarget_ - lineX_) * lineXFollow_;
    if ((lineCount_++ & 3) == 0) {
      const TractShape sh = lineShapeFor(p[kF1] * voice_.formant[0], p[kF2] * voice_.formant[1], p[kF3] * voice_.formant[2]);
      line_.setShape(sh, lineX_, ac, an);
    }
    // Glottal turbulence mostly with the glottis apart (h, aspiration, a
    // breathy voice): in modal voicing the Reynolds gate alone put 2-8 kHz
    // hiss on every vowel.
    const float aspGate = 0.15f + 0.85f * smoothstep(kAgVoice, kAgBreathy, ag);
    line_.setNoise(kLineFric, kLineAsp * aspGate, std::max(ag, 0.005f));
  }
  {
    LarynxControl lc;
    lc.t = t_;
    // The folds' pitch target: the melody with the slow tension that
    // abduction adds; the fast aerodynamic effects are the folds' own.
    lc.f0 = baseF0_ * pitch2_ * jitter_ * (1.0f + kF0Tension * tension_);
    lc.f0Wave = f0;
    lc.phase = phase_;
    lc.phaseStep = dt;
    lc.duty = duty_;
    // With the glottis fully closed (a pause) the speaker also stops pushing
    // air from the lungs, which ends any remaining vibration.
    // With the line tract the pressure above the folds is the line's own
    // (p1 from the last substep), so the lung pressure goes in whole; the
    // cascade's mouth-pressure estimate (drop) is for the formant tract.
    const float dropForFolds = tract_ == Tract::Line ? 1.0f : drop;
    lc.ps = runFolds ? kLungPressure * lungFactor * dropForFolds * smoothstep(0.0f, 0.5f * kAgVoice, ag) * press_.tick() : 0.0f;
    lc.ag = ag;
    creakNow_ += (creakTarget_ - creakNow_) * kCreakFollow_;
    lc.irregular = 1.0f + kCreakJitter * creakNow_;
    lc.gap = voice_.glottalGap;
    lc.microGain = voice_.microGain;
    lc.cycleJitter = voice_.cycleJitter;
    lc.cycleShimmer = voice_.cycleShimmer;
    for (int k = 0; k < 3; ++k) lc.formant[k] = p[kF1 + k] * voice_.formant[k];
    lc.bw[0] = p[kB1];
    lc.bw[1] = p[kB2];
    lc.bw[2] = p[kB3];
    lc.vibDrive = vibAmp_;
    larynx_->step(lc, lo);
  }
  if (runFolds) glottalFlow = lo.flow;
  float g = lo.source;
  const float physPuff = lo.puff, airPuff = lo.airPuff;

  // Lungs: speaking uses up air; breathing in refills it, with a soft noise
  // swelling and fading over the inhalation.
  lungVolume_ = std::max(lungVolume_ - glottalFlow / fs_, 0.0f);
  actualUsed_ += glottalFlow / fs_;
  float inhale = 0;
  if (inhaleLeft_ > 0) {
    lungVolume_ = std::min(lungVolume_ + inhaleRate_, kLungMax);
    const float done = inhaleLen_ - inhaleLeft_;
    const float rise = std::min(msToSamples(kInhaleRiseMs, fs_), 0.3f * inhaleLen_);
    const float fall = std::min(msToSamples(kInhaleFallMs, fs_), 0.4f * inhaleLen_);
    if (inhaleSnatched_) {
      const float ws = std::sin(kPi * (1.0f - inhaleLeft_ / inhaleLen_));
      inhale = inhaleSnatchFilter_.process(noise()) * kInhaleNoise * ws * ws;
    } else {
      const float w = std::min(smoothstep(0.0f, rise, done), smoothstep(0.0f, fall, inhaleLeft_));
      inhale = inhaleFilter_.process(noise()) * kInhaleNoise * w;
    }
    inhaleLeft_ -= 1.0f;
  }

  // Constriction noise through its place-dependent shape. When the sound
  // changes, the shape glides (band edges and resonance in log frequency)
  // and the filter is retuned every sample, instead of crossfading between
  // two fixed spectra.
  const float n = noise();
  if (nGlide_ < 1.0f) {
    nGlide_ = std::min(1.0f, nGlide_ + nGlideStep_);
    const float w = nGlide_ * nGlide_ * (3.0f - 2.0f * nGlide_);  // smoothstep
    auto lg = [w](float a, float b) { return a * std::pow(b / a, w); };
    auto ln = [w](float a, float b) { return a + w * (b - a); };
    const FricShape& a = nFrom_.shape;
    const FricShape& b = nTo_.shape;
    nNow_.shape = {lg(a.lo, b.lo), lg(a.hi, b.hi), lg(a.peakF, b.peakF), lg(a.peakBw, b.peakBw),
                   ln(a.peakGain, b.peakGain)};
    nNow_.gain = ln(nFrom_.gain, nTo_.gain);
    nNow_.tract = ln(nFrom_.tract, nTo_.tract);
    nNow_.tractGain = ln(nFrom_.tractGain, nTo_.tractGain);
    nNow_.post = ln(nFrom_.post, nTo_.post);
    nf_.set(nNow_.shape, fs_);
  }
  float cTract = 0, cPar = 0;
  {
    // Turbulence saturates: a soft ceiling on the noise strength keeps a
    // fricative after a pressure build-up (k -> ð) from bursting out.
    const float g = nNow_.gain * fricAmp;
    nLevel_ += (g - nLevel_) * kNoiseLevelFollow_;
    nPeak_ = std::max(nLevel_, nPeak_ * kNoisePeakDecay_);
    const float v = nf_.process(n) * nNow_.post * g / std::sqrt(1.0f + (g / kNoiseCeiling) * (g / kNoiseCeiling));
    cTract += v * std::sqrt(nNow_.tract) * nNow_.tractGain;
    cPar += v * std::sqrt(1.0f - nNow_.tract);
  }

  // Vocal tract: voicing, glottal breath noise, the release transient and
  // labial noise through the nasal pole/zero pair and the formant cascade.
  // The shaped pulse takes the physical source's spectral shaping (the
  // voice's tilt, 3 kHz cut and 1 kHz lift), not the triangle's shelf.
  const bool physLike = runFolds;
  if (physLike) {
    g += physTiltGain_ * physTilt_.process(g);
    if (kPhysMidCut) g = midCut_.process(g);
    g = physUpperCut_.process(g);
    g = physMidLift_.process(g);
  }
  if (!physLike || kPhysicalShaping) {
    g += shelfGain_ * shelf_.process(g);  // source treble shelf: + (gain-1) * high-pass
    g = midCut_.process(g);
    g = presence_.process(g);
  }
  const float voiced = g * av;
  // Breath noise swelling and fading smoothly with each glottal cycle.
  const float puff = physPuff;
  // Physical source: breath follows the folds' own vibration and flow, not
  // the per-phoneme voicing levels (which calibrate the voiced sound).
  const float breathAmp = runFolds ? kPhysBreath * trillMod : av;
  const float breath = breathLp_.process(noise()) * kBreathiness * voice_.breathiness *
                       breathAmp * kBreathPuffGain * puff;
  speechEnv_ += (speechTarget_ - speechEnv_) *
                (speechTarget_ < speechEnv_ ? kSpeechDecay_ : speechAttackNow_);
  // Higher-pole correction (glottal sources only): the cascade's poles stop
  // at ~20 kHz, and the missing ones would each lift the highs a little
  // (together +14 dB at 6 kHz, more above). Bursts and frication through
  // the tract are left out (boosted, their pulsing buzzes).
  float glottal = voiced + breath;
  if (carrier_ == Carrier::Physical)
    for (Biquad& b : polesHigh_) glottal = b.process(glottal);
  // High breath noise (above ~5 kHz, through the formants): the breath above
  // is low-passed at 3 kHz, which left the highs between the harmonics
  // cleaner than any real voice. Partly pulsed with the cycle, partly
  // steady (fully pulsed it buzzed), by how strongly the folds vibrate.
  if (runFolds) {
    const float hb = hbHp2_.process(hbHp1_.process(noise()));
    glottal += hb * kHighBreathGain * voice_.breathiness * std::min(larynx_->vib() / kNoisePulseVib, 1.0f) *
               (1.0f - kHighBreathPulse + kHighBreathPulse * std::min(puff, 2.0f));
  }
  // Aspiration (h, the puff after p t k): turbulence at the glottis, a
  // source falling ~6 dB/octave above kAspirationTiltHz, shaped by the tract
  // like the vowel it belongs to. Kept out of the higher-pole correction:
  // white and boosted there it hissed like s instead of breathing.
  aspLp_ += kAspLp_ * (noise() - aspLp_);
  aspHpState_ += kAspHp_ * (aspLp_ - aspHpState_);
  glottal += (aspLp_ - aspHpState_) * aspAmp * aspGainNow_ * aspBurst_;  // aspGainNow_ carries the gain constants
  float y = (glottal + pop + cTract) * speechEnv_;
  if (runFolds)
    cPar += airLp_.process(airHp2_.process(airHp1_.process(noise()))) * kAirGain * voice_.air * voice_.breathiness *
            kPhysBreath * trillMod * airPuff * std::max(1.0f - openness, 0.0f);
  cPar = cPar * speechEnv_ + inhale;
  //
  // A small part of the source leaks past the whole cascade, so the highs
  // settle at the leak level instead of falling off at -12 dB/octave per
  // resonator. Above F3 the cascade output is phase-inverted, so the leak is
  // subtracted: it then adds up with the cascade where it takes over,
  // instead of cancelling it into a notch.
  const float src = y;
  y = nz_.process(y);
  y = np_.process(y);
  // Both are unity at DC, so with the zero above the pole everything above
  // them ends up (pole / zero)^2 weaker. Half of that is given back, so the
  // nasal murmur keeps more of its upper formants.
  y *= std::pow(std::clamp(p[kFNZ], 100.0f, nyq) / kNasalPoleHz, kNasalZeroComp);
  // The murmur below ~400 Hz a little lower (-3 dB, as far as the sound is
  // nasal): with the compensation above, m n boomed.
  {
    nasLp_ += kNasLp_ * (y - nasLp_);
    const float nasal = std::clamp((p[kFNZ] - kNasalPoleHz) / 500.0f, 0.0f, 1.0f);
    y -= kNasalLowCut * nasal * nasLp_;
  }
  if (kUseCavities)
    for (int i = 0; i < kNumCavities; ++i) y = cavPole_[i].process(cavZero_[i].process(y));
  if (kUseHighPoles)
    for (int i = 0; i < kNumTubePoles; ++i) y = tube_[i].process(y);
  y = r5_.process(y);
  y = r4_.process(y);
  y = r3_.process(y);
  y = r2_.process(y);
  y = r1_.process(y);
  y -= leak_ * src;
  // Chest: part of the voice is heard straight through the chest wall and
  // the neck, without the mouth's resonances (mostly lows).
  if (carrier_ == Carrier::Physical)
    y -= kChestDirect * chestLp_.process((glottalFlow - larynx_->flowMean()) * av * speechEnv_);
  // Body resonance (see kBodyResonances): voiced sound only. The tract output
  // is gated by the vibration strength, which is ~0 during bursts and hiss.
  {
    const float in = (1.0f - kBodyFollow) * voiced * speechEnv_ + kBodyFollow * y * vibAmp_;
    float body = 0;
    for (int i = 0; i < kNumBodyResonances; ++i)
      body += kBodyResonances[i].gain * body_[i].process(in);
    y += body;
  }
  const float radiation =
      kWallRadiation + (1.0f - kWallRadiation) *
                           smoothstep(0.0f, kRadiationOpenArea, ac + kNostrilFactor * an);
  y *= radiation;
  // Lip radiation: the radiated pressure rises 6 dB/octave with frequency.
  // Here flat below kLipRadiationHz and rising above it, so the lows keep
  // their weight; with the triangle's -12 dB/octave this gives a natural
  // voice's gentler tilt in the highs.
  // The physical source's output is already the flow derivative, which is
  // the lip radiation: there only the head's diffraction is added, a
  // bounded step of about +5 dB from ~1.5 kHz (radiating into a half space
  // in front of the head instead of all around).
  if (runFolds) {
    y = headShelf_.process(y);
  } else if (kLipRadiation) {
    const float d = (y - prevTract_) * lipGain_;
    prevTract_ = y;
    y += d;
  }

  // A 2nd-order cut below ~90 Hz on top (the male voice boomed: 63-100 Hz
  // 6-10 dB over the voice-overs, H1-H2 +3.5 against -0.7).
  if (tract_ == Tract::Line) {
    // The line tract made the sound, breath and all: the cascade's y, the
    // parallel noise and the pops are not used.
    // Not gated by speechEnv_ (pauses are silent physically; the gate clipped
    // onsets) and no head shelf (the piston in a baffle is the half-space).
    y = line_.output() * kLineGain;
    cPar = 0;
  }
  float out = lowCut_.process(outHp_.process(y * kOutGain + cPar));
  if (voice_.treble != 0) out = trebleShelf_.process(out);
  // Room tone: a faint, dark noise floor, so pauses never fall to digital
  // silence (which sounded like the recording switching off).
  roomLp1_ += kRoomLp_ * (noise() - roomLp1_);
  roomLp2_ += kRoomLp_ * (roomLp1_ - roomLp2_);
  out += kRoomTone * roomLp2_;
  return kMasterGain * voice_.gain * out;
}


// ---------------------------------------------------------------------------
// Sentence planner (FormantMotion::Planned; DESIGN_formant_planning.md).
//
// The keyframes of a sentence are turned into demands: for each phoneme and
// each formant it defines, a target, a tolerance (a dead zone), a time window
// and a weight. Formants a phoneme does not define are free. Then, per
// channel (F1, F2, F3, pitch), an iteratively reweighted least-squares solve
// over 5 ms frames minimises
//   reach/dwell: w (|x - target| - tol)+^2 inside each window
//   speed:       mu (|dx| - vmax)+^2          (the articulator's speed limit)
//   relaxation:  rel (x - r)^2 on free frames (r: the line between the
//                neighbouring demands, else the spline)
//   smoothness:  eps (second difference)^2    (anti-jaggedness only).
// A vowel's target moves through its window toward its neighbours' targets
// by kPlanCoart (the recordings' formants keep moving inside vowels).
// Everything is in log2 units (a semitone costs the same everywhere).

namespace {

float planKnob(const char* name, float d) {
  const char* v = std::getenv(name);
  return v ? static_cast<float>(std::atof(v)) : d;
}

struct PlanWeights {
  float hopMs = planKnob("PLAN_HOP", 5);
  float w = planKnob("PLAN_W", 1.0f);          // reach/dwell weight of a full demand
  float tol = planKnob("PLAN_TOL", 0.3f);      // tolerance as a share of kRangeTolerance
  float vmax[4] = {planKnob("PLAN_VMAX1", 8), planKnob("PLAN_VMAX2", 5), planKnob("PLAN_VMAX3", 5), planKnob("PLAN_F0VMAX", 3.3f)};  // oct/s
  float mu = planKnob("PLAN_MU", 2.0f);
  float rel = planKnob("PLAN_REL", 0.02f);
  float eps = planKnob("PLAN_EPS", 0.05f);
  // How far a vowel's start and end lean toward the neighbours' targets, per
  // formant. (0.5 on all three: the female round trip 13 %, 6.2 % with 0:
  // the jaw must not lean half-way to a stop's closed F1.)
  float coart[3] = {planKnob("PLAN_COART1", 0.15f), planKnob("PLAN_COART2", 0.3f), planKnob("PLAN_COART3", 0.2f)};
  float f0w = planKnob("PLAN_F0W", 1.0f);
  int rounds = static_cast<int>(planKnob("PLAN_ROUNDS", 10));
};

struct Demand {
  int ch;        // 0-2 formant, 3 pitch
  int i0, i1;    // frame window, inclusive
  float t0, t1;  // target (log2) at i0 and at i1, linear between
  float tol, w;
  bool followSpline = false;  // the target is the keyframe spline itself (the pitch line)
};

// Symmetric pentadiagonal solve (LDL^T): d diagonal, e first off-diagonal
// (i, i+1), f second (i, i+2); rhs in b, solution returned in b.
void solvePenta(std::vector<double>& d, std::vector<double>& e, std::vector<double>& f, std::vector<double>& b) {
  const int n = static_cast<int>(d.size());
  // Factor: L unit lower with bands l1 (i, i-1), l2 (i, i-2); D in d.
  std::vector<double> l1(n, 0), l2(n, 0);
  for (int i = 0; i < n; ++i) {
    if (i >= 2) {
      l2[i] = f[i - 2] / d[i - 2];
    }
    if (i >= 1) {
      double v = e[i - 1];
      if (i >= 2) v -= l2[i] * d[i - 2] * l1[i - 1];
      l1[i] = v / d[i - 1];
    }
    double dd = d[i];
    if (i >= 1) dd -= l1[i] * l1[i] * d[i - 1];
    if (i >= 2) dd -= l2[i] * l2[i] * d[i - 2];
    d[i] = std::max(dd, 1e-12);
  }
  // Forward: L y = b.
  for (int i = 0; i < n; ++i) {
    if (i >= 1) b[i] -= l1[i] * b[i - 1];
    if (i >= 2) b[i] -= l2[i] * b[i - 2];
  }
  // Diagonal, then back: L^T x = y.
  for (int i = 0; i < n; ++i) b[i] /= d[i];
  for (int i = n - 1; i >= 0; --i) {
    if (i + 1 < n) b[i] -= l1[i + 1] * b[i + 1];
    if (i + 2 < n) b[i] -= l2[i + 2] * b[i + 2];
  }
}

}  // namespace

bool Synth::planAt(double t, float out[4]) const {
  const size_t n = plan_[0].size();
  if (n < 2 || planStartT_ < 0) return false;
  const double x = (t - planStartT_) / planHop_;
  if (x < 0 || x >= static_cast<double>(n - 1)) return false;
  const size_t i = static_cast<size_t>(x);
  const float s = static_cast<float>(x - static_cast<double>(i));
  for (int c = 0; c < 4; ++c) {
    const std::vector<float>& v = plan_[c];
    const float p0 = v[i > 0 ? i - 1 : 0], p1 = v[i], p2 = v[i + 1], p3 = v[std::min(i + 2, n - 1)];
    // Catmull-Rom over the grid: C1 across frames.
    const float m1 = 0.5f * (p2 - p0), m2 = 0.5f * (p3 - p1);
    out[c] = p1 + s * (m1 + s * (-3 * p1 + 3 * p2 - 2 * m1 - m2 + s * (2 * p1 - 2 * p2 + m1 + m2)));
  }
  return true;
}

void Synth::planSentence() {
  if (motion_ != FormantMotion::Planned) return;
  static const PlanWeights W;
  planHop_ = msToSamples(W.hopMs, fs_);
  // The keyframes not planned yet, grouped into phoneme spans: a run of
  // keyframes with the same phoneme and the same formant targets (addHeld
  // makes four per target: two edges and two inner points).
  struct Span {
    Ph ph; Cls cls; bool vowel;
    double tA, tB, tI0, tI1;  // extent and inner window (samples)
    float f[3], f0;           // targets (Hz, multiplier)
  };
  std::vector<Span> spans;
  double tEnd = planFromT_;
  for (size_t q = 0; q < k_.size(); ++q) {
    const Keyframe& kf = k_[q];
    if (kf.t < planFromT_ - 0.5) continue;
    tEnd = std::max(tEnd, kf.t);
    // One span per phoneme: its onset keyframe (a coarticulated start) and
    // the held group together; the targets are the last keyframe's.
    const bool same = !spans.empty() && spans.back().ph == kf.ph;
    if (same) {
      Span& sp = spans.back();
      // Inner window: the keyframes that are not edges.
      if (!kf.edge) {
        if (sp.tI0 < 0) sp.tI0 = kf.t;
        sp.tI1 = kf.t;
      }
      sp.tB = kf.t;
      sp.f[0] = kf.p[kF1]; sp.f[1] = kf.p[kF2]; sp.f[2] = kf.p[kF3];
      sp.f0 = kf.p[kF0Mul];
    } else {
      Span sp;
      sp.ph = kf.ph;
      sp.cls = info(kf.ph).cls;
      sp.vowel = kf.vowel;
      sp.tA = sp.tB = kf.t;
      sp.tI0 = sp.tI1 = kf.edge ? -1 : kf.t;
      sp.f[0] = kf.p[kF1]; sp.f[1] = kf.p[kF2]; sp.f[2] = kf.p[kF3];
      sp.f0 = kf.p[kF0Mul];
      spans.push_back(sp);
    }
  }
  if (spans.empty() || tEnd <= planFromT_ + planHop_) {
    planFromT_ = std::max(planFromT_, tEnd);
    return;
  }
  for (Span& sp : spans)
    if (sp.tI0 < 0) { sp.tI0 = sp.tA; sp.tI1 = sp.tB; }

  // The frame grid continues the existing plan.
  if (planStartT_ < 0) planStartT_ = planFromT_;
  const size_t first = plan_[0].size();
  const double gridStart = planStartT_ + static_cast<double>(first) * planHop_;
  const int n = std::max(2, static_cast<int>((tEnd - gridStart) / planHop_) + 1);
  if (n > 200000) return;  // a sentence of over 15 minutes: leave it to the spline
  auto frameOf = [&](double t) { return std::clamp(static_cast<int>(std::lround((t - gridStart) / planHop_)), 0, n - 1); };

  // Demands. Per class: which formants the phoneme defines, with weights
  // (share of W.w) and tolerance factors.
  std::vector<Demand> demands;
  auto defineAt = [&](int ch, int i0, int i1, float wShare, float tolFactor, float tgtA, float tgtB) {
    Demand d;
    d.ch = ch;
    d.i0 = i0;
    d.i1 = std::max(i1, i0);
    d.t0 = tgtA;
    d.t1 = tgtB;
    const float tolShare = ch < 3 ? kRangeTolerance[ch] : 0.0175f;  // pitch: 0.3 st
    d.tol = std::log2(1.0f + tolShare) * (ch < 3 ? W.tol : 1.0f) * tolFactor;
    d.w = (ch < 3 ? W.w : W.f0w) * wShare;
    demands.push_back(d);
  };
  auto define = [&](const Span& sp, int ch, float wShare, float tolFactor, float tgtA, float tgtB) {
    defineAt(ch, frameOf(sp.tI0), frameOf(sp.tI1), wShare, tolFactor, tgtA, tgtB);
  };
  auto shares = [&](const Span& sp, float out[3], float tolF[3]) {
    for (int c = 0; c < 3; ++c) { out[c] = 0; tolF[c] = 1; }
    switch (sp.cls) {
      case Cls::Vowel: out[0] = out[1] = out[2] = 1; break;
      case Cls::Glide: out[0] = out[1] = out[2] = 0.8f; break;
      case Cls::Liquid:
      case Cls::Trill:
        if (sp.ph == Ph::L) { out[1] = 0.6f; out[2] = 0.8f; }
        else { out[1] = 0.3f; out[2] = 1.0f; }  // ɹ, r: the F3 colour
        break;
      // Obstruents and nasals close the jaw: their low F1 is a demand too
      // (the recordings' F1 dips at every consonant; that is where most of
      // their in-vowel F1 motion comes from). Liquids leave F1 to the vowel.
      case Cls::Nasal: out[0] = 0.8f; out[1] = 0.4f; out[2] = 0.2f; tolF[1] = 2; break;
      case Cls::Plosive: out[0] = 0.5f; out[1] = 0.3f; out[2] = 0.2f; tolF[0] = tolF[1] = tolF[2] = 2; break;
      case Cls::Fricative: out[0] = 0.5f; out[1] = 0.3f; out[2] = 0.3f; tolF[0] = tolF[1] = tolF[2] = 2; break;
      case Cls::H:
      case Cls::Stop: break;
    }
  };
  for (size_t s = 0; s < spans.size(); ++s) {
    const Span& sp = spans[s];
    float sh[3], tf[3];
    shares(sp, sh, tf);
    for (int c = 0; c < 3; ++c) {
      if (sh[c] <= 0 || sp.f[c] <= 0) continue;
      const float T = std::log2(sp.f[c]);
      float ta = T, tb = T;
      if (sp.vowel && W.coart[c] > 0) {
        // Toward the neighbours that define this formant.
        for (size_t q = s; q-- > 0;) {
          float sh2[3], tf2[3]; shares(spans[q], sh2, tf2);
          if (spans[q].cls == Cls::Stop || spans[q].cls == Cls::H) break;
          if (sh2[c] > 0 && spans[q].f[c] > 0) { ta = T + W.coart[c] * (std::log2(spans[q].f[c]) - T); break; }
        }
        for (size_t q = s + 1; q < spans.size(); ++q) {
          float sh2[3], tf2[3]; shares(spans[q], sh2, tf2);
          if (spans[q].cls == Cls::Stop || spans[q].cls == Cls::H) break;
          if (sh2[c] > 0 && spans[q].f[c] > 0) { tb = T + W.coart[c] * (std::log2(spans[q].f[c]) - T); break; }
        }
      }
      if (sp.vowel && (ta != T || tb != T)) {
        // A vowel's path: from the neighbour-biased start, through the full
        // target at the centre, to the neighbour-biased end, over the whole
        // vowel (the recordings' formants keep moving inside vowels, and the
        // target is reached, not skirted).
        const int ia = frameOf(sp.tA), ib = frameOf(sp.tB);
        const int ic = frameOf(0.5 * (sp.tI0 + sp.tI1));
        defineAt(c, ia, ic, sh[c], tf[c], ta, T);
        defineAt(c, std::min(ic + 1, ib), ib, sh[c], tf[c], T, tb);
      } else {
        define(sp, c, sh[c], tf[c], ta, tb);
      }
    }
    if (sp.cls != Cls::Stop && sp.cls != Cls::H && sp.f0 > 0) {
      // Pitch: the keyframe pitch line (accents, falls, micro-intonation) is
      // the demand, frame by frame, over the whole phoneme; the planner
      // smooths it and limits its speed.
      const float T = std::log2(sp.f0);
      defineAt(3, frameOf(sp.tA), frameOf(sp.tB), sp.vowel ? 1.0f : 0.3f, 1, T, T);
      demands.back().followSpline = true;
    }
  }

  if (std::getenv("PLAN_DEBUG")) {
    std::fprintf(stderr, "plan: %zu spans, %zu demands, %d frames from %.3f s\n", spans.size(), demands.size(), n, gridStart / fs_);
    for (const Span& sp : spans)
      std::fprintf(stderr, "  span ph%d cls%d v%d  %.3f-%.3f win %.3f-%.3f  F %.0f %.0f %.0f f0 %.3f\n", static_cast<int>(sp.ph), static_cast<int>(sp.cls), sp.vowel,
                   sp.tA / fs_, sp.tB / fs_, sp.tI0 / fs_, sp.tI1 / fs_, sp.f[0], sp.f[1], sp.f[2], sp.f0);
    for (const Demand& d : demands)
      std::fprintf(stderr, "  demand ch%d frames %d-%d target %.0f -> %.0f tol %.3f w %.2f\n", d.ch, d.i0, d.i1, std::exp2(d.t0), std::exp2(d.t1), d.tol, d.w);
  }
  // Solve each channel.
  const float h = W.hopMs * 0.001f;  // seconds per frame
  for (int c = 0; c < 4; ++c) {
    const int param = c < 3 ? kF1 + c : kF0Mul;
    std::vector<double> x(n), spline(n);
    for (int i = 0; i < n; ++i) {
      const float v = splineAt(param, gridStart + static_cast<double>(i) * planHop_);
      spline[i] = std::log2(std::max(v, 1e-3f));
      x[i] = spline[i];
    }
    // Demand coverage and the relaxation line between demands.
    std::vector<char> covered(n, 0);
    std::vector<double> r(spline);
    std::vector<int> lastDemandEnd(n, -1);
    for (const Demand& d : demands) {
      if (d.ch != c) continue;
      for (int i = d.i0; i <= d.i1; ++i) covered[i] = 1;
    }
    {
      // r on free frames: the line from the previous demand's end value to
      // the next demand's start value; before the first / after the last,
      // the spline.
      std::vector<std::pair<int, double>> ends, starts;  // (frame, target)
      for (const Demand& d : demands) {
        if (d.ch != c) continue;
        if (d.followSpline) continue;  // (the spline is the line already)
        starts.push_back({d.i0, d.t0});
        ends.push_back({d.i1, d.t1});
      }
      std::sort(starts.begin(), starts.end());
      std::sort(ends.begin(), ends.end());
      for (int i = 0; i < n; ++i) {
        if (covered[i]) continue;
        // previous end <= i, next start >= i
        auto ne = std::lower_bound(ends.begin(), ends.end(), std::make_pair(i, -1e9));
        auto ns = std::lower_bound(starts.begin(), starts.end(), std::make_pair(i, -1e9));
        const bool havePrev = ne != ends.begin(), haveNext = ns != starts.end();
        if (havePrev && haveNext) {
          const auto& pe = *(ne - 1);
          const auto& nx = *ns;
          const double f = nx.first > pe.first ? static_cast<double>(i - pe.first) / (nx.first - pe.first) : 0.0;
          r[i] = pe.second + f * (nx.second - pe.second);
        } else if (havePrev) {
          r[i] = (ne - 1)->second;
        } else if (haveNext) {
          r[i] = ns->second;
        }
      }
    }
    const double vmaxF = W.vmax[c] * h;  // oct per frame
    for (int round = 0; round < W.rounds; ++round) {
      std::vector<double> d(n, 0), e(n, 0), f(n, 0), b(n, 0);
      // Smoothness, everywhere.
      for (int i = 1; i + 1 < n; ++i) {
        d[i - 1] += W.eps; d[i] += 4 * W.eps; d[i + 1] += W.eps;
        e[i - 1] += -2 * W.eps; e[i] += -2 * W.eps;
        f[i - 1] += W.eps;
      }
      // Relaxation on free frames (and a tiny anchor everywhere, so the
      // system is never singular).
      for (int i = 0; i < n; ++i) {
        const double wr = covered[i] ? 1e-6 : W.rel;
        d[i] += wr; b[i] += wr * r[i];
      }
      // Demands: the dead zone's nearest edge pulls once outside it.
      for (const Demand& dm : demands) {
        if (dm.ch != c) continue;
        for (int i = dm.i0; i <= dm.i1; ++i) {
          const double fr = dm.i1 > dm.i0 ? static_cast<double>(i - dm.i0) / (dm.i1 - dm.i0) : 0.0;
          const double tgt = dm.followSpline ? spline[i] : dm.t0 + fr * (dm.t1 - dm.t0);
          const double dev = x[i] - tgt;
          if (std::fabs(dev) <= dm.tol) continue;
          const double edge = tgt + (dev > 0 ? dm.tol : -dm.tol);
          d[i] += dm.w; b[i] += dm.w * edge;
        }
      }
      // Speed limit: on frames moving faster than vmax, the IRLS weight
      // mu (1 - vmax / |dx|) on dx^2 approximates mu (|dx| - vmax)^2.
      for (int i = 1; i < n; ++i) {
        const double dx = std::fabs(x[i] - x[i - 1]);
        if (dx <= vmaxF) continue;
        const double wm = W.mu * (1.0 - vmaxF / dx);
        d[i] += wm; d[i - 1] += wm; e[i - 1] -= wm;
      }
      solvePenta(d, e, f, b);
      x.swap(b);
    }
    if (std::getenv("PLAN_DEBUG") && c < 3)
      for (const Demand& dm : demands)
        if (dm.ch == c && dm.i1 > dm.i0)
          std::fprintf(stderr, "  ch%d frames %d-%d: target %.0f -> %.0f, spline %.0f -> %.0f, planned %.0f -> %.0f (mid %.0f)\n", c, dm.i0, dm.i1,
                       std::exp2(dm.t0), std::exp2(dm.t1), std::exp2(spline[dm.i0]), std::exp2(spline[dm.i1]), std::exp2(x[dm.i0]), std::exp2(x[dm.i1]), std::exp2(x[(dm.i0 + dm.i1) / 2]));
    // Append (overwriting nothing: the grid continues the previous plan).
    plan_[c].resize(first + static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) plan_[c][first + static_cast<size_t>(i)] = static_cast<float>(x[i]);
  }
  planFromT_ = gridStart + static_cast<double>(n - 1) * planHop_;
}

}  // namespace puhuja
