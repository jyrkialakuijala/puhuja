#include "puhuja.h"
#include "prosody.h"
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
// lineShapeFor's blend (refresh them with ./linefit --measure whenever a
// shape changes: stale labels misplace the diphthong and locus blends,
// Whisper -47 on 2026-11-05). First fitted to the English formant targets on the
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
    {0, param("LINE_V0_XC", 0.5347f) /* cost 0.0360125915678626 n13-vowel0-c */, param("LINE_V0_AC", 1.995f) /* cost 0.0360125915678626 n13-vowel0-c */, param("LINE_V0_WC", 4.296f) /* cost 0.0360125915678626 n13-vowel0-c */, param("LINE_V0_ALIP", 8.94f) /* cost 0.0360125915678626 n13-vowel0-c */, param("LINE_V0_APH", 0.7533f) /* cost 0.0360125915678626 n13-vowel0-c */, param("LINE_V0_AOR", 6.264f) /* cost 0.0360125915678626 n13-vowel0-c */, param("LINE_V0_LLIP", 1.582f) /* cost 0.0360125915678626 n13-vowel0-c */, param("LINE_V0_ALAR", 0.3f) /* cost 0.0360125915678626 n13-vowel0-c */, param("LINE_V0_LEN", 17.38f), 852.0f, 1471.0f, 2447.0f},  // a
    {1, param("LINE_V1_XC", 0.1426f) /* cost 0.0393701754025880 n13-vowel1-c */, param("LINE_V1_AC", 3.41f) /* cost 0.0393701754025880 n13-vowel1-c */, param("LINE_V1_WC", 1.775f) /* cost 0.0393701754025880 n13-vowel1-c */, param("LINE_V1_ALIP", 3.617f) /* cost 0.0393701754025880 n13-vowel1-c */, param("LINE_V1_APH", 4.766f) /* cost 0.0393701754025880 n13-vowel1-c */, param("LINE_V1_AOR", 3.455f) /* cost 0.0393701754025880 n13-vowel1-c */, param("LINE_V1_LLIP", 3.721f) /* cost 0.0393701754025880 n13-vowel1-c */, param("LINE_V1_ALAR", 0.9144f) /* cost 0.0393701754025880 n13-vowel1-c */, param("LINE_V1_LEN", 17.4787f), 471.0f, 1603.0f, 2508.0f},  // e
    {2, param("LINE_V2_XC", 0.5287f), param("LINE_V2_AC", 2.003f), param("LINE_V2_WC", 2.45f), param("LINE_V2_ALIP", 3.575f), param("LINE_V2_APH", 10.86f), param("LINE_V2_AOR", 0.6748f), param("LINE_V2_LLIP", 1.759f), param("LINE_V2_ALAR", 0.6139f), param("LINE_V2_LEN", 17.44f), 209.0f, 2232.0f, 2916.0f},  // i: 2026-11-04 refit (linefit) to 280/2600/3200: the searched shape (XC 0.13, i.e. the hump in the larynx tube, no palatal narrowing) gave male F2 ~1830, an ɪ; now a narrow 0.42 cm^2 palatal channel at 0.67 of the length, rendered male "Need" 361/2286/2824 (Kokoro F2 ~2350). Whisper 30 iː words m+f: iː heard 10 -> 41 of 60, guards ɪ/eɪ/j same
    {3, param("LINE_V3_XC", 0.488f), param("LINE_V3_AC", 2.762f), param("LINE_V3_WC", 3.873f), param("LINE_V3_ALIP", 2.743f), param("LINE_V3_APH", 1.022f), param("LINE_V3_AOR", 7.843f), param("LINE_V3_LLIP", 0.5993f), param("LINE_V3_ALAR", 2.784f), param("LINE_V3_LEN", 17.55f), 599.0f, 1029.0f, 2124.0f},  // o
    // u refit 2026-11-04 (linefit toward 300/820/2250): the old shape (AC 0.92,
    // lips 1.31) was half an o, F1 ~400 rendered, heard as OW; now "Who." 328/846.
    {4, param("LINE_V4_XC", 0.3234f), param("LINE_V4_AC", 1.631f), param("LINE_V4_WC", 1.648f), param("LINE_V4_ALIP", 1.55f), param("LINE_V4_APH", 11.93f), param("LINE_V4_AOR", 2.527f), param("LINE_V4_LLIP", 1.654f), param("LINE_V4_ALAR", 0.7042f), param("LINE_V4_LEN", 17.53f), 303.0f, 1205.0f, 2049.0f},  // u
    {6, param("LINE_V6_XC", 0.5796f), param("LINE_V6_AC", 2.362f), param("LINE_V6_WC", 6.0f), param("LINE_V6_ALIP", 4.874f), param("LINE_V6_APH", 1.33f), param("LINE_V6_AOR", 4.752f), param("LINE_V6_LLIP", 1.357f), param("LINE_V6_ALAR", 0.3f), param("LINE_V6_LEN", 17.48f), 690.0f, 1566.0f, 2474.0f},  // æ: 2026-11-04 refit (linefit) to 730/1880/2500: the searched shape (5 cm lip tube) rendered F2 ~1510, an ɑ-like vowel heard as EH/AH/"ha"; now rendered m ~725/1700, Whisper æ 24 -> 33 of 50
    {8, param("LINE_V8_XC", 0.2969f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */, param("LINE_V8_AC", 2.156f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */, param("LINE_V8_WC", 0.8017f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */, param("LINE_V8_ALIP", 2.0f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */, param("LINE_V8_APH", 9.094f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */, param("LINE_V8_AOR", 2.246f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */, param("LINE_V8_LLIP", 1.298f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */, param("LINE_V8_ALAR", 2.641f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */, param("LINE_V8_LEN", 17.55f), 309.0f, 1491.0f, 2152.0f},  // ə: wider mouth, narrower pharynx (2026-11-04; was ALIP 2.55 APH 5.74 AOR 1.85, F1/F2 ~430/1600, a front vowel heard as ɪ; now ~490/1450, Kokoro ~520/1500; Whisper medial ə 12 -> 18 of 42)
    {9, param("LINE_V9_XC", 0.3742f), param("LINE_V9_AC", 5.945f), param("LINE_V9_WC", 3.347f), param("LINE_V9_ALIP", 8.144f), param("LINE_V9_APH", 3.955f), param("LINE_V9_AOR", 0.7924f), param("LINE_V9_LLIP", 2.756f), param("LINE_V9_ALAR", 0.5738f), param("LINE_V9_LEN", 17.46f), 323.0f, 2197.0f, 2348.0f},  // ɪ
    {10, param("LINE_V10_XC", 0.1734f), param("LINE_V10_AC", 1.402f), param("LINE_V10_WC", 1.959f), param("LINE_V10_ALIP", 10.27f), param("LINE_V10_APH", 3.99f), param("LINE_V10_AOR", 2.227f), param("LINE_V10_LLIP", 2.541f), param("LINE_V10_ALAR", 0.3034f), param("LINE_V10_LEN", 17.39f), 550.0f, 1943.0f, 2497.0f},  // ɛ: 2026-11-04 refit (linefit) to 560/1800/2500: the searched shape (XC 0.83, big mouth) had drifted to F2 1585, below ə, a central ʌ-like vowel heard as ɪ/ə/æ; rendered male F2 1495 -> 1556 (Kokoro 1660-1710). Whisper 30 ɛ words m+f: medial ɛ 18 -> 28 of 60, guards same
    {11, param("LINE_V11_XC", 0.5771f) /* cost 0.0367746817152231 n13-vowel11-c */, param("LINE_V11_AC", 1.482f) /* cost 0.0367746817152231 n13-vowel11-c */, param("LINE_V11_WC", 1.258f) /* cost 0.0367746817152231 n13-vowel11-c */, param("LINE_V11_ALIP", 2.199f) /* cost 0.0367746817152231 n13-vowel11-c */, param("LINE_V11_APH", 3.906f) /* cost 0.0367746817152231 n13-vowel11-c */, param("LINE_V11_AOR", 5.916f) /* cost 0.0367746817152231 n13-vowel11-c */, param("LINE_V11_LLIP", 1.491f) /* cost 0.0367746817152231 n13-vowel11-c */, param("LINE_V11_ALAR", 0.735f) /* cost 0.0367746817152231 n13-vowel11-c */, param("LINE_V11_LEN", 17.5f), 462.0f, 1039.0f, 2300.0f},  // ʊ (2026-11-09: refit toward Kokoro male 462/1039, ref/opt/lab/UH/fit.py; the old 537/1379 render aimed at a wrong 1486)
    {12, param("LINE_V12_XC", 0.3935f) /* cost 0.0365812236825875 n13-vowel12-c */, param("LINE_V12_AC", 1.121f) /* cost 0.0365812236825875 n13-vowel12-c */, param("LINE_V12_WC", 4.349f) /* cost 0.0365812236825875 n13-vowel12-c */, param("LINE_V12_ALIP", 2.616f) /* cost 0.0365812236825875 n13-vowel12-c */, param("LINE_V12_APH", 3.436f) /* cost 0.0365812236825875 n13-vowel12-c */, param("LINE_V12_AOR", 8.741f) /* cost 0.0365812236825875 n13-vowel12-c */, param("LINE_V12_LLIP", 2.157f) /* cost 0.0365812236825875 n13-vowel12-c */, param("LINE_V12_ALAR", 2.412f) /* cost 0.0365812236825875 n13-vowel12-c */, param("LINE_V12_LEN", 17.51f), 488.0f, 903.0f, 2760.0f},  // ɔ
    {13, param("LINE_V13_XC", 0.9141f), param("LINE_V13_AC", 2.893f), param("LINE_V13_WC", 2.8f), param("LINE_V13_ALIP", 2.98f), param("LINE_V13_APH", 2.285f), param("LINE_V13_AOR", 4.138f), param("LINE_V13_LLIP", 0.9681f), param("LINE_V13_ALAR", 1.077f), param("LINE_V13_LEN", 17.5f), 551.0f, 1356.0f, 2569.0f},  // ʌ: 2026-11-04 refit (linefit) to 620/1200/2550: the searched shape (open pharynx APH 4.1, AOR 3.8) gave F1 too low, rendered male ~470/1340, next to today's ə (426/1441), heard as ER/UH/ə; now ~600/1250 (narrow pharynx, open mouth). Whisper 31 ʌ words m+f: ʌ heard 26 -> 39 of 62, words 6 -> 12
    {14, param("LINE_V14_XC", 0.6448f) /* cost 0.0360966861522094 n13-vowel14-c */, param("LINE_V14_AC", 3.265f) /* cost 0.0360966861522094 n13-vowel14-c */, param("LINE_V14_WC", 1.164f) /* cost 0.0360966861522094 n13-vowel14-c */, param("LINE_V14_ALIP", 3.578f), param("LINE_V14_APH", 0.5f), param("LINE_V14_AOR", 8.584f), param("LINE_V14_LLIP", 0.4685f) /* cost 0.0360966861522094 n13-vowel14-c */, param("LINE_V14_ALAR", 0.6215f) /* cost 0.0360966861522094 n13-vowel14-c */, param("LINE_V14_LEN", 17.51f), 722.0f, 1338.0f, 2461.0f},  // ɑ  // 2026-11-04: narrow pharynx, wide mouth, smaller lips (was ALIP 2.6 APH 1.6 AOR 3.6, F2 ~1310: heard as ʌ); Whisper medial ɑ 6 -> 45, words 11 -> 21 of 110
    {15, param("LINE_V15_XC", 0.1969f), param("LINE_V15_AC", 0.3039f), param("LINE_V15_WC", 1.0418f), param("LINE_V15_ALIP", 0.8546f), param("LINE_V15_APH", 13.5401f), param("LINE_V15_AOR", 9.6664f), param("LINE_V15_LLIP", 0.6944f), param("LINE_V15_ALAR", 2.9333f), param("LINE_V15_LEN", 17.6406f), 327.0f, 838.0f, 1515.0f},  // ɒ: the onset r (Kokoro "read", "right": F2 815-930, F3 1440-1530), not a vowel of English here
    {16, param("LINE_V16_XC", 0.7417f), param("LINE_V16_AC", 0.6857f), param("LINE_V16_WC", 2.048f), param("LINE_V16_ALIP", 1.46f), param("LINE_V16_APH", 5.402f), param("LINE_V16_AOR", 6.937f), param("LINE_V16_LLIP", 0.9941f), param("LINE_V16_ALAR", 3.0f), param("LINE_V16_LEN", 17.53f), 318.0f, 1403.0f, 1725.0f},  // ɜ
};
// The line's output level, and its turbulence gains (noise pressure per unit
// of Reynolds number past critical, at the constriction and at the glottis).
const float kLineGain = param("LINE_GAIN", 9.2e-6f) /* 2026-11-05: M90 = 90 % G1 + 10 % old defaults (geometric) */ /* 2026-11-05: x3.4 after the schwa fit to Kokoro (the vowels 10.6 dB weaker; G1, with LINE_FRIC 22 -> 11.5) */ /* 2026-11-04: x2.55 for the LF-fitted folds (7 dB weaker) with the noise at -7 dB: Kennedy rms -23.6 dB as before the fit */;  // vowels ~-20 dB rms, kennedy.txt peaks under 1.0
// Output EQ of the line tract (2026-11-07). Kokoro's voiced spectrum is 9-14
// dB above ours at 1-3 kHz (ref/opt/zopt3/rasp/eqtest.out); the line's losses
// are not the cause (wall damping, HF loss: <= 2 dB), the source is. Jyrki
// heard offline EQs on the paragraph (oracle, oracle/2, shelf 1 kHz +6, peak
// 2 kHz +9): "all ... were good". A high shelf and a bell on the line's
// output, voice and noise alike as in what he heard. Zimtohrli is nearly
// blind to this (level-normalised NSIM): the ear decides it.
// Off by default since the sharper closing phase (string.cc kInertance, the
// same day): Jyrki chose the fold variant without the EQ over the EQ'd default.
const float kLineEqShelfHz = param("LINE_EQ_SHELF_HZ", 1000.0f), kLineEqShelfDb = param("LINE_EQ_SHELF_DB", 0.0f);
const float kLineEqPeakHz = param("LINE_EQ_PEAK_HZ", 2000.0f), kLineEqPeakBw = param("LINE_EQ_PEAK_BW", 2000.0f), kLineEqPeakDb = param("LINE_EQ_PEAK_DB", 0.0f);
const float kLinePostHw = param("LINE_POST_HW", 4.0f) /* 2.5 until 2026-11-06: the ʃ a narrow hiss peaking at 4 kHz, "too scratchy" (Jyrki); 4 flattens it 1.6-6 kHz like Kokoro's, abolish-hw4nolp "is better"; kennedy Whisper 859 */;   // ʃ ʒ ʧ ʤ: the groove's half-width (cm)
// ʃ ʒ ʧ ʤ: the jet's thickness for its noise band (cm; 0: the slit's
// height). The slit's height (0.08 cm^2 over the slit's width, ~0.3 mm)
// put the band's corner at the 12 kHz clamp: the hiss sat at 8-11 kHz,
// 10-15 dB over the 2.5-4 kHz body where goog's ʃ is 10-14 dB under it
// ("revolution", "station" sounded like -sn, Jyrki 2026-11-05).
const float kLinePostJetD = param("LINE_POST_JET_D", 0.1f);
// ... and the soft switch between that fixed thickness (1) and the slit's own height (0), which sweeps the
// noise corner with the gesture. 2026-11-08 (Jyrki: "let's go for best whisper scores"): the slit's height
// alone heard 123-128 instead of 110-120 of the 188 ʃʒʧʤ clips in words1000 over three draws (totals
// 941/892/927 vs 919/893/902); fixed thin jets 0.01-0.03 did not help (the corner pinned at LINE_NOISE_FC_MAX),
// 0.3 hurt. Jyrki: "could there be a new fuzzy soft switch then? looks like this is important to get right".
const float kLinePostJetMix = param("LINE_POST_JET_MIX", 1.0f);
const float kLinePostSlitScale = param("LINE_POST_SLIT_SCALE", 1.0f);  // x the followed slit height for ʃ ʒ ʧ ʤ
// ... and the smoothing of their noise corner alone (ms; < 0: the global one). The followed slit starts the ʃ with a
// 5.6 kHz spike in its first 4 ms ("revolu[ping]ionary", Jyrki 2026-11-09); the global smoothing cost words1000 55-70 words.
const float kLinePostStSmoothMs = param("LINE_POST_ST_SMOOTH_MS", -1.0f);
// ... and the range of their noise corner (Hz; 0: the global 200 / LINE_NOISE_FC_MAX): a floor keeps the followed slit's
// smoothed corner from starting the ʃ as a low thud, a ceiling from spiking (the "t pling" of "revolu[t]ionary").
const float kLinePostFcMin = param("LINE_POST_FC_MIN", 0.0f);
const float kLinePostFcMax = param("LINE_POST_FC_MAX", 0.0f);
// ... and their band's low-pass, x the corner (0: LINE_NOISE_LPR). Kokoro's ʃ
// falls off above ~7 kHz; Jyrki 2026-11-05: "-tion could be slightly darker".
const float kLinePostLpr = param("LINE_POST_LPR", 2.0f);  // ʃ ʒ only (Whisper SH 72 -> 70 of 106, within noise)
const float kLinePostLip = param("LINE_POST_LIP", 1.5f);  // (0.5 -> 1.5 on 2026-11-04: medial ʃ heard 12 -> 20 of 40, Whisper) (0.7 put ʃ at -3..-7 dB under the vowel, Kokoro -5..-23; the ablation charged it 0.0007 on the 857 words)  // ... and their lip rounding (cm^2)
const float kLineRestartMs = param("LINE_RESTART_MS", 0.0f);  // after a voiceless fricative: shut for this long, then voice (0: adduct only)
const float kLineAdductMs = param("LINE_ADDUCT_MS", 30.0f) /* cost 1814 whisper-run1-phon_err */;  // adduction into voice after a voiceless fricative
const float kLineRRound = param("LINE_R_ROUND", 0.588f) /* cost 1814 whisper-run1-phon_err */;  // an onset r: share of u's shape mixed into the r-coloured one
// u after j ("human", "you", "few", "music"): the English u there is fronted.
// The line used the back u's own shape: "human" F2 ~830 Hz through the u,
// the readers 1250-1580 (11labs, goog). Share of i's shape mixed in (0: off;
// 2026-11-04: 0.4 gives F2 ~1740 -> 1200-1400 through the u; Whisper on 18
// ju words m+f: medial u heard 16 -> 24 of 32, words 7 -> 12 of 36).
const float kLineUAfterJ = param("LINE_U_AFTER_J", 0.2f);
const float kLineLevelExp = param("LINE_LEVEL_EXP", 1.0f);  // lung pressure ~ level^this
const float kLineAnticipNasal = param("LINE_ANTICIP_NASAL", 0.4162f) /* cost 1814 whisper-run1-phon_err */;  // the velum's opening during a vowel before a nasal, as a share of the nasal's
const float kLineReleaseKick = param("LINE_RELEASE_KICK", 0.0f);  // (0.2 gave a voiced stop a 23 ms voiceless puff: "abolish" was heard as "atolish"; off since 2026-11-03)  // a voiced stop's release: the glottis parts to this area (cm^2; 0: off) ...
const float kLineReleaseKickMs = param("LINE_RELEASE_KICK_MS", 15.0f);  // ... for this long before voicing
const float kLineFricAc = param("LINE_FRIC_AC", 0.882f) /* cost 1058 whisper-sens1-combo-phase1-err */;  // the fricatives' constriction area on the line, relative to the aero table
// The puff of p t k (2026-11-04, Whisper on 45 t-initial words: t heard at the
// start 9 -> 21 of 90 with these and the noise at 5000 Hz; the simplex had
// shrunk the hold to 7 ms, and at the old 0.3 cm^2 the folds kept flapping).
const float kLineVotHoldUnstressed = param("LINE_VOT_HOLD_UNSTR", 0.3f);  // x LINE_VOT_HOLD_MS for an unstressed syllable's p t k ("to", "poverty")
const float kLineVotHoldMs = param("LINE_VOT_HOLD_MS", 35.0f);     // a voiceless stop's puff: the glottis held open this long after the release (stressed; 60 % unstressed)
const float kLineAspAg = param("LINE_ASP_AG", 0.6f);  // the glottis during a voiceless stop's puff (cm^2)
const float kLineStopAspMul = param("LINE_STOP_ASP", 22.5f);  // ... its glottal noise relative to LINE_ASP
const float kLineTStopAspMul = param("LINE_T_STOP_ASP", 7.41f) /* 2026-11-07: 7.0 -> 7.41 (+0.5 dB, Jyrki: "t like in tiger could be 1 db stronger ... perhaps better with +0.7 db" ... "a bit softer at 0.5 db") */ /* 18 until 2026-11-05: Jyrki chose the softer, shorter puff (pta-R) for "power to abolish" */;  // t's own puff noise (< 0: LINE_STOP_ASP); 2026-11-04 with the release below: initial t 18 -> 27 of 90 (Whisper); 33 -> 18 with LINE_PUFF_LPR 10 (the wide band made the t puff -8 dB re the vowel, Kokoro -10..-15; now -12)
const float kLineTJetD = param("LINE_T_JET_D", 0.0f);  // t's jet thickness at its closure and release (cm; 0: the slit's height), below
const float kLineTLpr = param("LINE_T_LPR", 0.0f);  // ... and its band's low-pass / its corner meanwhile (0: LINE_NOISE_LPR)
// p's (P lab 2026-11-10; defaults: the sound unchanged): its jet thickness at the closure and the release (cm; 0: the
// slit's height) and that band's low-pass, its own puff hold (< 0: LINE_VOT_HOLD_MS), a final p's hiss and its length
// (< 0: LINE_FINAL_ASP / LINE_FINAL_ASP_MS).
const float kLinePJetD = param("LINE_P_JET_D", 0.0f);
const float kLinePLpr = param("LINE_P_LPR", 0.0f);
const float kLinePVotHoldMs = param("LINE_P_VOT_HOLD_MS", -1.0f);
const float kLineFinalPAsp = param("LINE_FINAL_P_ASP", -1.0f);
const float kLineFinalPAspMs = param("LINE_FINAL_P_ASP_MS", -1.0f);
const float kLinePPuffLpr = param("LINE_P_PUFF_LPR", -1.0f);  // p's own puff band low-pass ratio (<= 0: LINE_PUFF_LPR)
const float kLinePAspAg = param("LINE_P_ASP_AG", -1.0f);  // p's own glottis during the puff (cm^2; < 0: LINE_ASP_AG)
// k's own (lab knobs 2026-11-10, K agent 2; < 0 or 1: as before; LINE_K_X_FRONT / _BACK at lineKPlace). The dictionary heard k as p 1240 and as t 796 times
// of 17040; initial k before a front vowel worst (k+iː 83 %, k+eɪ 76 %, k+æ 56 % errors, mostly p). Kokoro's initial k
// (am_adam, af_heart): VOT ~70 ms, puhuja's ~40 (the p's range); burst peak 2.3-2.7 kHz before front vowels, 1.4-1.8
// kHz before back ones, puhuja's 800 Hz (an LF thump) before æ eɪ and back vowels; the female puff 9 dB under Kokoro's.
const float kLineKVotHoldMs = param("LINE_K_VOT_HOLD_MS", -1.0f);   // the k's puff hold (ms; < 0: LINE_VOT_HOLD_MS)
const float kLineKReleaseScale = param("LINE_K_RELEASE_SCALE", -1.0f);  // the k's opening time scale (< 0: LINE_RELEASE_SCALE)
const float kLineKAspAg = param("LINE_K_ASP_AG", -1.0f);            // the k's glottis during the puff (cm^2; < 0: LINE_ASP_AG)
const float kLineKPuffFemale = param("LINE_K_PUFF_FEMALE", -1.0f);  // the female k puff (< 0: LINE_PUFF_FEMALE)
const float kLineKPuffMedial = param("LINE_K_PUFF_MEDIAL", -1.0f);  // a word-medial k's puff (< 0: LINE_PUFF_MEDIAL)
const float kLineKPuffLpr = param("LINE_K_PUFF_LPR", -1.0f);        // the k puff's band (< 0: the glottal default, as LINE_P_PUFF_LPR)
const float kLineFinalKAsp = param("LINE_FINAL_K_ASP", -1.0f);      // a final k's release hiss (< 0: LINE_FINAL_ASP)
const float kLineFinalKAspMs = param("LINE_FINAL_K_ASP_MS", -1.0f); // ... its length (< 0: LINE_FINAL_ASP_MS)
// A k after s ("school", "ask", "discuss": heard as p 180 of 968): English leaves it unaspirated; > 0: a puff
// anyway, its hold x this (0: as before, no puff).
const float kLineSkAsp = param("LINE_SK_ASP", 0.0f);
const float kLineTSoftBurst = param("LINE_T_SOFT_BURST", 0.0f);  // t's burst area into f v θ ð x this (0: none; 1: as p k), below
const float kLineCloseFric = param("LINE_CLOSE_FRIC", 1.0f);  // 2026-11-05: every other non-fricative closure's (p b k g m n ŋ d) slit-noise gain while it closes and opens (x LINE_FRIC; 1: as before). The lips closing on a vowel's 300-440 cm^3/s made a 3000 cm/s jet, Re 7000, a hiss 9 x the f's with its band sweeping 1.2 -> 11 kHz in 35 ms, the m's release the same downward: the "slurps" (Jyrki: "weird slirps ... mouth accidents rather than phonemes"). t: LINE_T_CLOSE_FRIC, dʒ: LINE_JH_CLOSE_FRIC.
const float kLineTCloseFric = param("LINE_T_CLOSE_FRIC", 0.3f);  // t's frication gain while it closes (x LINE_FRIC), below
const float kLineStopFOwn = param("LINE_STOP_F_OWN", 1.0f);  // > 0: a p t k released into f takes f's own jet, band, width and gain (below)
const float kLineTReleaseScale = param("LINE_T_RELEASE_SCALE", 2.0f);  // t's own release time scale (< 0: LINE_RELEASE_SCALE): the tongue tip leaves the ridge slower than the shared 0.58
// t's own (lab knobs 2026-11-11, T agent 3; < 0, 0 or 1: as before). The dictionary (build 414a35d) heard t as p 1134,
// k 863, d 487 times of 23452 (18.8 %, Kokoro 4.8 %); initial t 38 % (p 394), tr 48 %, ts 27 % (k 230), final 24 %.
// Kokoro's initial t (am_adam, af_heart): burst peak 5-6.5 kHz, the puff -9..-14 dB re the vowel, VOT 60-90 ms;
// puhuja's burst peaked at 3.5 kHz (male) / 4.6 kHz (female), a k's band, the puff -15..-28 dB, VOT 44 ms.
// The lab (300 hard t words: initial t, tr, final t after a vowel, ts; 682 targets, base 227 / 225 t errors on two
// draws, heard as p 99): LINE_T_BURST_MS 20 alone 149 (p 33: the release's 0.5-2 kHz "thump" that made it a p is gone,
// the burst 15-30 dB lower there), with LINE_T_BURST_LIQ 1 110; + LINE_T_APPROACH_MS 50 72 / 100; with
// LINE_T_BURST_JET_D 0.2 and LINE_T_APPROACH_A 0.6 (no hybrid veto) 94 / 98, top-300 t words 82 -> 52 / 80 -> 50,
// words1000 1199 -> 1258, 1201 -> 1264.
const float kLineTBurstMs = param("LINE_T_BURST_MS", 20.0f);     // a two-step release into a vowel (as k's LINE_K_BURST_MS; 0: off)
const float kLineTBurstA = param("LINE_T_BURST_A", 0.1f);       // ... the slit's area meanwhile (cm^2)
const float kLineTBurstJetD = param("LINE_T_BURST_JET_D", 0.2f);  // ... its jet thickness (cm; 0: the slit's own)
const float kLineTBurstLiq = param("LINE_T_BURST_LIQ", 1.0f);   // > 0: the two-step release also before l r w j
const float kLineTBurstFric = param("LINE_T_BURST_FRIC", -1.0f);  // ... the slit's frication gain (< 0: the release's, LINE_T_REL_FRIC)
const float kLineTBurstX = param("LINE_T_BURST_X", -1.0f);       // ... the slit's place (< 0: the t's own)
const float kLineTVotHoldMs = param("LINE_T_VOT_HOLD_MS", -1.0f);   // the t's puff hold (ms; < 0: LINE_VOT_HOLD_MS)
const float kLineTAspAg = param("LINE_T_ASP_AG", -1.0f);            // the t's glottis during the puff (cm^2; < 0: LINE_ASP_AG)
const float kLineTPuffLpr = param("LINE_T_PUFF_LPR", -1.0f);        // the t puff's band (<= 0: LINE_PUFF_LPR)
const float kLineTPuffFemale = param("LINE_T_PUFF_FEMALE", -1.0f);  // the female t puff (< 0: LINE_PUFF_FEMALE)
const float kLineTPuffMedial = param("LINE_T_PUFF_MEDIAL", -1.0f);  // a word-medial t's puff (< 0: LINE_PUFF_MEDIAL)
const float kLineTLiqVot = param("LINE_T_LIQ_VOT", 1.0f);           // a t before l r w j: the puff's hold x this
const float kLineTLiqAsp = param("LINE_T_LIQ_ASP", 1.0f);           // ... and its noise x this
const float kLineFinalTAsp = param("LINE_FINAL_T_ASP", -1.0f);      // a final t's release hiss (< 0: LINE_FINAL_ASP)
const float kLineFinalTAspMs = param("LINE_FINAL_T_ASP_MS", -1.0f); // ... its length (< 0: LINE_FINAL_ASP_MS)
const float kLineTRelShape = param("LINE_T_REL_SHAPE", 0.0f);       // an alveolar pinch after the release (the k g one's, at t's place; 0: off)
const float kLineTRelMs = param("LINE_T_REL_MS", 25.0f);            // ... its length (ms)
const float kLineTRelAc = param("LINE_T_REL_AC", 0.6f);             // ... its hump's area (cm^2)
const float kLineTRelX = param("LINE_T_REL_X", -1.0f);              // ... its place (< 0: the t's)
// A vowel before t narrows at the ridge this long before the closure (0: off; at <= ~30 ms it falls inside the closing
// and does nothing). "Get": F2 rose 2010 -> 2100 and F3 to 3320 into the closure (a velar-like convergence; heard k,
// deleted), with 50 ms it falls 1930 -> 1720 toward the alveolar locus as Kokoro's 1640 -> 1570; hard list deletions
// 32 -> 12, errors 227 -> 190 alone.
const float kLineTApproachMs = param("LINE_T_APPROACH_MS", 50.0f);
const float kLineTApproachA = param("LINE_T_APPROACH_A", 0.6f);     // ... to this area (cm^2)
// The glottal noise band's low-pass ratio during a p t k puff only (<= 0: off,
// the h's band; 2026-11-04: at the puff's glottis the band ended at ~1.6 kHz,
// no 2-5 kHz hiss; widening it for all glottal noise made the h an f). 10 on
// 2026-11-04 (with LINE_T_STOP_ASP 18): initial p 13 -> 17 of 18, t 3 -> 6 of 20
// (Whisper), the h unchanged byte for byte.
const float kLinePuffLpr = param("LINE_PUFF_LPR", 10.0f);
const float kLinePStopAspMul = param("LINE_P_STOP_ASP", 3.0f);  // p's own puff noise (< 0: LINE_STOP_ASP); 2026-11-04: with the fast release, initial p heard 25 -> 38 of 84 (Whisper)
// g's glottis in its closure, as a fraction of the way from voicing toward open (0.15: a cascade-era "firmer
// closure"; a more open glottis fills the mouth faster, stops the voice, and after the release breathes an h,
// "Good." -> "huh", 2026-11-04). And g's own release time scale (< 0: the shared one).
const float kLineGClosureAgFrac = param("LINE_G_CLOSURE_AG", 0.15f);
const float kLineGReleaseScale = param("LINE_G_RELEASE_SCALE", -1.0f);
const float kLinePReleaseScale = param("LINE_P_RELEASE_SCALE", 1.5f);  // p's own release time scale (< 0: LINE_RELEASE_SCALE): the lips part fast, a weaker puff
// The voiceless h on the line (2026-11-04): it was mostly the flow step of the
// glottis opening (a thump then a decaying tail), with too little glottal hiss;
// the stop puff's mechanism: its own glottis width and noise gain (reset at the
// next sound) and a slower opening.
const float kLineHAg = param("LINE_H_AG", 0.3f);      // the glottis during a voiceless h (cm^2)
const float kLineHAspMul = param("LINE_H_ASP", 30.0f);  // its glottal noise relative to LINE_ASP (30 on 2026-11-04: the h was ~-48 dB re the vowel, Kokoro -15..-25; now ~-32, dropped h 14 -> 8 of 48)
const float kLineHRiseMs = param("LINE_H_RISE_MS", 15.0f);  // how fast the glottis opens for it (ms)
// The glottal noise of an h between voiced sounds ("man holds"), relative to
// LINE_ASP. At 1 that h was a 45 ms gap: the glottis at 0.25 cm^2 stops the
// folds and LINE_ASP alone gave its 2-8 kHz band ~20 dB under the vowel's
// (2026-11-04). 30: a hiss at -11 dB, Whisper heard s/f (words 15 -> 9 of
// 46); the narrower glottis kEnglishVoicedHAg 0.18 (+ 8) keeps the folds
// going, like 11labs' breathy h, but dropped h 4 -> 8. Kept at 1.
const float kLineVHAspMul = param("LINE_VH_ASP", 3.0f);
// A vowel before a coda r: its share of ɜ and where the r-coloured demand starts
// (share of the vowel); the open vowels ɑ, a (aɪ, aʊ) on their own (< 0: as the others).
const float kLColourMix = param("LINE_LCOLOR_MIX", 0.0f);  // a vowel before an l bends toward the l's shape at its late key (goog, 11labs: F1 drifts down over 100-150 ms into the l)
const float kRColourMix = param("LINE_RCOLOR_MIX", 0.8f);  // (0.3 also liked in "very different", 2026-11-05, but kept for the coda r of "mortal")
const float kRColourAt = param("LINE_RCOLOR_AT", 0.6f);
const float kRColourOpenMix = param("LINE_RCOLOR_OPEN_MIX", 0.4f);  // (2026-11-04: ɑ+r heard with ɜ 48 -> 31 of 60, with AT 0.85)
const float kRColourOpenAt = param("LINE_RCOLOR_OPEN_AT", 0.85f);
// A vowel before a coda r keeps its own bump place and area with at least this weight (SPLAN_PRE_R_VOWEL_W; 0: as any
// vowel, a wide bump leaves them to the r). Its wide bump left the place to the r's bunched one through the whole
// vowel: "dare" ɛ F2 1290 Hz flat ("den" 1780, Kokoro 1650 falling to 1540); Jyrki 2026-11-06: "dare sounds very strange".
const float kPreRVowelW = param("SPLAN_PRE_R_VOWEL_W", 0.0f);
// ... and the front vowels' own (ɪ ɛ e i æ: "dare", "their", "heirs", "here"; the back and open vowels of "for", "more",
// "are" keep the colouring that fits them): SPLAN_PRE_R_FRONT_W, LINE_RCOLOR_FRONT_MIX / _AT (-1: as all vowels).
const float kPreRFrontW = param("SPLAN_PRE_R_FRONT_W", 1.0f);  // Jyrki 2026-11-06: "ok, use your fix" ("dare" heard as "don't": ɛ F2 1290 -> 1546 Hz; kennedy Whisper 853 -> 854)
const float kRColourFrontMix = param("LINE_RCOLOR_FRONT_MIX", 0.8f);
const float kRColourFrontAt = param("LINE_RCOLOR_FRONT_AT", 0.85f);  // the colouring from 85 % of a front vowel (all vowels: 60 %)
// ... and the r-coloured ɜ (ER) itself: its bunched tongue is a wide bump too, so before a labial ("for which", "for
// progress") the lips' demand won and its F3 stayed at 2800 Hz, no r colour ("for" heard as "fur"/"fuh", Jyrki 2026-11-06).
const float kErPlaceW = param("SPLAN_ER_PLACE_W", 1.0f);  // Jyrki 2026-11-06: "deploy the fix as default" ("for which" ɜ F3 2666 -> 1832 Hz; kennedy Whisper 854 -> 856)
bool frontVowel(Ph p) { return p == Ph::E || p == Ph::I || p == Ph::IH || p == Ph::EH || p == Ph::AE || p == Ph::Y || p == Ph::OE; }
// ɜ's own shape demand (2026-11-04): a short unstressed final -er (Water,
// Never, Power, 173 ms) did not reach the r-coloured shape from the vowel or
// t before it: F3 stayed 2200-2800 Hz (the stressed Her/Per reach ~1780,
// Kokoro ~1600-1800). Held from/to these shares of the vowel, with this weight
// (0.3 / 0.7 / 1 before; Whisper final ɜ 8 -> 27 of 34 with these and CHAIN 0).
const float kLineErFrom = param("LINE_ER_FROM", 0.1f);
const float kLineErTo = param("LINE_ER_TO", 1.0f);
const float kLineErW = param("LINE_ER_W", 3.0f);
// 0: a ɜ after a vowel is not a chain's second vowel (reached only from 80 %);
// "Water", "Better" counted as chains across their t (lastPh_ is the vowel).
const float kLineErChain = param("LINE_ER_CHAIN", 0.0f);
// 1: a vowel after a plosive is never a chain's second vowel. lastPh_ skips
// plosives, so "poverty" (ɜ t i) was a ɜ-i glide: the i was wanted only from
// 80 % of it and the r colour ran on through the t into the i ("povertry",
// Jyrki 2026-11-05; F2/F3 ~1200/1740 Hz through the sentence's i, Kokoro
// ~1850/2550 at the release). 0: as before.
const float kLineChainStop = param("LINE_CHAIN_STOP", 1.0f) /* 0 until 2026-11-05: Jyrki, "poverty-C is good" */;
// A vowel after a plosive is wanted from this many ms before its start (in
// the closure: Kokoro's i in "poverty" starts at its F2 at the release) and
// not only from 30 % of it (0: as before).
const float kLineAfterStopLeadMs = param("LINE_AFTER_STOP_LEAD_MS", 30.0f) /* 0 until 2026-11-05 (pov-C) */;
// The oʊ offglide's ʊ (u's shape) is wanted from this share of the ʊ on (<0:
// SPLAN_GLIDE_END, 0.8). From 80 % the planner never got there: "Go", "Know"
// held F2 ~1340 Hz to 80 % of the vowel and ended ~1280 (Kokoro falls to
// ~950 by the middle), so oʊ was heard as u or ʌ (2026-11-04). From 10 %:
// "Go" 517/1251 at 20 %, 434/1037 at 50 %, 386/933 at 80 % (Kokoro
// 528/1340, 451/968, 340-399/850-950); Whisper oʊ 24 -> 32 of 50.
const float kLineOwGlideFrom = param("LINE_OW_GLIDE_FROM", 0.1f);
// The same for eɪ (e + ɪ, the ɪ in i's shape; <0: SPLAN_GLIDE_END). From 80 %
// "Day", "Make", "Name", "Take", "Say" stayed at ~470/1500 Hz through the whole
// ɪ (Kokoro rises to 2100-2300), heard as ɛ or "nerd" (2026-11-04). From 10 %:
// "Day" ends 368/2187, "Make" 363/2286 (Kokoro 361/2272); Whisper eɪ 10 -> 34 of 54.
const float kLineEyGlideFrom = param("LINE_EY_GLIDE_FROM", 0.85f);
// The same for aɪ (a + ɪ) and aʊ (a + ʊ) (<0: SPLAN_GLIDE_END). From 80 %
// "Time", "My", "Now", "How" stayed on the a (~650/1290 Hz) to the very end:
// no glide at all (Kokoro "My" 580/1150 -> 390/2300, "Now" 600/1550 -> 370/870).
// From 30 %: the a is held ~60 %, then "My" ends 428/1612, "Now" 431/1094
// (from 10 %: earlier glide, 410/1640 and 420/1045, Whisper the same within
// noise, aɪ heard more as eɪ/iː); Whisper aɪ 1 -> 11 of 26, aʊ 4 -> 6 of 26 (2026-11-04).
const float kLineAyGlideFrom = param("LINE_AY_GLIDE_FROM", 0.85f);
const float kLineAwGlideFrom = param("LINE_AW_GLIDE_FROM", 0.35f) /* 0.3 until 2026-11-05: Jyrki chose the glide from the start (vdn-U) for "now"; 0 until 2026-11-08: with the ɔ end (LINE_AW_END_AO) the end is reached late, now-ao100g50 */;
// The ʊ end of aʊ takes o's shape (row 3); this mixes u's shape (row 4) into
// it (0 = o, 1 = u). With o, "Now", "How" ended at F2 ~1100-1160 Hz, Kokoro
// "Now" 370/870, "Power" 400/760 (2026-11-04).
const float kLineAwOffU = param("LINE_AW_OFF_U", 0.5f) /* 0 until 2026-11-05: the u offglide, vdn-U; 1 -> 0.5 later that day: "the uu part feels too long", vdn-N1 */;
// The aʊ end mixed back toward the a's shape by this share (Jyrki 2026-11-05: the
// second part of "now" "more of a mixture of a and u"; 0: off).
const float kLineAwEndA = param("LINE_AW_END_A", 0.0f) /* 0 until 2026-11-08: on the ɔ end, F1 stays nearer Kokoro's open end (553 vs 457 Hz at 80 %); Jyrki: a30 and a50 "also ok, choose yourself" */;
// Jyrki 2026-11-08: "now sounds like 'na-o' and not 'nau' with a blend between a and u". Kokoro's
// "now" keeps F1 open (650-680 Hz) to the end while F2 falls 1600 -> 1100; puhuja closed F1 to 420
// (the o/u end) with F2 standing at ~1300. The end mixed toward ɔ's shape (open back) by this share,
// and the a start toward æ (Kokoro starts at F2 ~1600; a's row gives 1350).
const float kLineAwEndAo = param("LINE_AW_END_AO", 1.0f) /* Jyrki 2026-11-08: "now-ao100g50-cut.wav is ok" */;
const float kLineAwStartAe = param("LINE_AW_START_AE", 0.0f);
// ... and toward ɛ's shape (LINE_AW_START_E; 2026-11-09 AW agent: the æ mix lowered F1 760 -> 620 but left F2 at 1390,
// the mix keeps the a's bump place; Kokoro's aʊ starts at ~600/1600, between æ and ɛ).
const float kLineAwStartE = param("LINE_AW_START_E", 0.0f);
// ɔɪ lab knobs (OY agent 2026-11-10; the defaults reproduce the previous sound). Whisper heard the ɔɪ of the dictionary
// words as aɪ / eɪ (a fronted onset: in "voice" the plan glided from the ɔ's 10-30 % straight to the ɪ's 80 %, F2 rising
// from the ɔ's middle) or as ɔ / ɑ ("boys" -> "boss": the ɪ of a phrase-final ɔɪ was demanded only at 80-90 % of it,
// beyond the pause's plan horizon, and never reached). LINE_OY_GLIDE_FROM: the ɪ's demand from this share of it (<0:
// SPLAN_GLIDE_END); LINE_OY_HOLD_TO: the ɔ's demand to this share of it (<0: SPLAN_GLIDE_START); LINE_OY_AO_W /
// LINE_OY_IH_W: the two demands' weights x these.
const float kLineOyGlideFrom = param("LINE_OY_GLIDE_FROM", -1.0f);
const float kLineOyHoldTo = param("LINE_OY_HOLD_TO", 0.8f);
const float kLineOyAoW = param("LINE_OY_AO_W", 1.0f), kLineOyIhW = param("LINE_OY_IH_W", 1.0f);
// The shapes: the ɔ's lip area / lip length / pharynx area x these, mixed toward o (LINE_OY_START_O) or ɑ
// (LINE_OY_START_AA); the ɪ mixed toward i (LINE_OY_END_I) or ɛ (LINE_OY_END_E) (0 = none).
const float kLineOyLipA = param("LINE_OY_LIPA", 1.0f), kLineOyLipL = param("LINE_OY_LIPL", 1.0f), kLineOyAph = param("LINE_OY_APH", 1.0f);
const float kLineOyStartO = param("LINE_OY_START_O", 0.0f), kLineOyStartAa = param("LINE_OY_START_AA", 0.0f);
const float kLineOyEndI = param("LINE_OY_END_I", 0.0f), kLineOyEndE = param("LINE_OY_END_E", 0.0f);
// ... the ɔ's tongue constriction place / area x these, the ɪ's oral cavity area / lip area x these, and the ɔ's demand
// from this share of it (<0: 0.1 as every chain start).
const float kLineOyXc = param("LINE_OY_XC", 1.0f), kLineOyAc = param("LINE_OY_AC", 1.0f);
const float kLineOyIhAor = param("LINE_OY_IH_AOR", 1.0f), kLineOyIhLipA = param("LINE_OY_IH_LIPA", 1.0f);
const float kLineOyAoFrom = param("LINE_OY_AO_FROM", -1.0f);
// The first pause after speech plans this far into itself instead of stopping LINE_PAUSE_PLAN_HOLD_MS before the end of
// the pause token (<0: off). In a pause run that stop fell ~135 ms before the speech's end, so the last sound's late
// demands (a final ɔɪ's ɪ) were planned three frames at a time, too late to be reached: "Boy." held F2 ~890 Hz to the
// last 30 ms of its ɪ.
const float kLinePauseRunPlanMs = param("LINE_PAUSE_RUN_PLAN_MS", 0.0f);
// ... 1: only when an ɔɪ ended inside the part the hold would leave unplanned (0: after every sound).
const float kLinePauseRunOy = param("LINE_PAUSE_RUN_OY", 0.0f);
// aɪ's own shape and timing (AY agent, 2026-11-10; every default reproduces the sound before). Whisper heard aɪ as eɪ
// 677 times of 5292 (dictionary rank 8): against ten Kokoro voices puhuja's aɪ starts front (male F2 1275 Hz at the
// start, 1714 at 40 %; Kokoro 988 and 1219) and glides from the a's 30 % on, a straight ramp where Kokoro holds the
// open back start and rises late (F2 1510 at 60 %, 1813 at 80 %; tools in ref/opt/lab/AY/agent2).
// The a mixed toward the row LINE_AY_START_ROW (a Ph index: 14 ɑ, 12 ɔ, 13 ʌ) by LINE_AY_START_MIX (0: off), and its
// pharynx and lip areas scaled, its tract lengthened (cm):
const float kLineAyStartMix = param("LINE_AY_START_MIX", 0.6f);
const int kLineAyStartRow = static_cast<int>(param("LINE_AY_START_ROW", 14.0f));
const float kLineAyAAph = param("LINE_AY_A_APH", 1.0f), kLineAyALip = param("LINE_AY_A_ALIP", 1.0f), kLineAyALen = param("LINE_AY_A_LEN", 0.0f);
// The ɪ end mixed toward the row LINE_AY_END_ROW (2 i: closer and fronter; 10 ɛ, 1 e: less close) by LINE_AY_END_MIX.
const float kLineAyEndMix = param("LINE_AY_END_MIX", 0.0f);
const int kLineAyEndRow = static_cast<int>(param("LINE_AY_END_ROW", 2.0f));
// The a demanded until this share of it (< 0: SPLAN_GLIDE_START, 0.3), the ɪ until this share (< 0: 0.9), and the
// weights of the two demands (x SPLAN_GLIDE_W).
const float kLineAyHold = param("LINE_AY_HOLD", 0.15f), kLineAyGlideTo = param("LINE_AY_GLIDE_TO", 1.0f);
const float kLineAyAW = param("LINE_AY_A_W", 1.0f), kLineAyIW = param("LINE_AY_I_W", 1.0f);
const float kLineAyPreMs = param("LINE_AY_PRE_MS", 0.0f), kLineAyStopLeadMs = param("LINE_AY_STOP_LEAD_MS", 0.0f);
// The a's tongue place and area count in the plan at least this much (the planner's placeFloor; 0: by the bump's
// strength, which leaves the open a's place free) and the ɪ end's lip area x LINE_AY_END_ALIP (> 1: spread).
const float kLineAyPlaceFloor = param("LINE_AY_PLACE_FLOOR", 0.0f), kLineAyEndALip = param("LINE_AY_END_ALIP", 1.0f);
// A lazy whole-vowel glide (see its use): the number of points (0: off), the progress exponent, the a's share, weight.
const float kLineAyRamp = param("LINE_AY_RAMP", 12.0f), kLineAyRampPow = param("LINE_AY_RAMP_POW", 2.0f);
const float kLineAyRampSplit = param("LINE_AY_RAMP_SPLIT", 0.6f), kLineAyRampW = param("LINE_AY_RAMP_W", 1.0f);
const float kLineVotAdductMs = param("LINE_VOT_ADDUCT_MS", 20.0f);  // ... then adducting into voice over this
const float kLineExpandFlow = param("LINE_EXPAND_FLOW", 200.0f);  // cm^3/s drawn by the expanding cavity at full expansion behind a voiced closure
const float kLineVoicedReleaseScale = param("LINE_VOICED_RELEASE", 2.5f);  // a voiced stop's release time relative to a voiceless one's
// ʤ on the line (2026-11-04): the release ramped through the ʒ slit in ~22 ms with the glottis in
// the voiced-fricative position, so only ~10 ms of weak noise came out and ʤ was heard as d/t.
// LINE_JH_RELEASE: ʤ's own release time scale (< 0: shared x voiced); LINE_JH_AG: the glottal area
// during its frication (< 0: LINE_AG_VFRIC; English ʤ is largely devoiced, a wider glottis hisses).
const float kLineJhReleaseScale = param("LINE_JH_RELEASE", 0.3f);  // 2026-11-07: -1 (the plosive's 49 ms) -> 0.3 (15 ms): the ʒ of ʤ then holds its slit for ~40 ms as Kokoro's does, instead of gliding shut-to-open through it
const float kLineJhAg = param("LINE_JH_AG", 0.4f);  // (0.4 for an hour on 2026-11-07: the devoiced hiss measured like Kokoro's but Jyrki preferred the voiced one, "it could be more dzenerosity": the hiss comes from LINE_JH_REL_FRIC instead)  // 2026-11-07: -1 (voiced, 0.05: the ʒ of ʤ got no flow for its hiss, -22..-35 dB re the s) -> 0.4 (hiss -9..-3 re the s for ~35 ms; Kokoro -5..+5 for 50)
// ʤ between vowels (Jyrki 2026-11-05: the g of "villages" "too separate and buzzy"):
// its closure drew the shared LINE_EXPAND_FLOW (200) and the folds stalled, a
// 50 ms hole at -40 dB where Kokoro keeps a voice bar at -12 dB; the tongue
// closing with the vowel's flow hissed 10 ms at -4 dB (Kokoro: a 40 ms fade,
// no hiss); its ʒ ran 45 ms at -7 dB (Kokoro 25 ms at -12..-18). Its own
// expansion flow (cm^3/s; 0: the shared one), closing hiss x this, ʒ noise x this.
const float kLineJhExpandFlow = param("LINE_JH_EXPAND_FLOW", 800.0f) /* vil-J4, Jyrki 2026-11-05 */;
const float kLineJhCloseFric = param("LINE_JH_CLOSE_FRIC", 0.3f);
const float kLineZhFric = param("LINE_ZH_FRIC", 2.65f);
const float kLineJhFric = param("LINE_JH_FRIC", 4.0f);  // the ʤ's frication gain (< 0: LINE_ZH_FRIC; the ʤ's frication is an emitted ʒ) (2026-11-12: inert for the ʤ: after the closure the ʒ's own event is not queued, the release carries its hiss: LINE_JH_REL_FRIC)
// ʤ lab knobs (JH agent, round H, 2026-11-12; every default reproduces the deployed sound). The dictionary's ʤ errors
// by place: initial 21.5 % (heard t, ʧ, ɹ), medial 11.6 % (t, ʧ), final 15.5 % (ʧ 47 of 64: stressed final 31 %).
// Against Kokoro (am_adam / af_heart, "job", "major", "page"): puhuja's ʤ frication is 3.3-4 kHz (Kokoro 4.5-5.5),
// -4..-9 dB re the word's peak and voiceless (Kokoro -8..-16, partly voiced in the middle of a word, a final ʤ often
// only -25..-45 with its voice bar running on), and an initial one leaves a 15-20 ms voiced gap before the vowel.
// Per place (INIT after a pause, MED, FINAL before a pause; 1 / < 0: off): _CLOSE x the closure's length, _FLEN x the
// frication's, _FRIC x its hiss gain, _AG the glottis at the release (< 0: LINE_JH_AG), _ZAG the glottis of the
// frication's second part (< 0: off; LINE_JH_ZAG_AT).
const float kLineJhPosClose[3] = {param("LINE_JH_INIT_CLOSE", 1.0f), param("LINE_JH_MED_CLOSE", 1.0f), param("LINE_JH_FINAL_CLOSE", 1.0f)};
const float kLineJhPosFlen[3] = {param("LINE_JH_INIT_FLEN", 1.0f), param("LINE_JH_MED_FLEN", 1.0f), param("LINE_JH_FINAL_FLEN", 1.0f)};
const float kLineJhPosFric[3] = {param("LINE_JH_INIT_FRIC", 1.0f), param("LINE_JH_MED_FRIC", 1.0f), param("LINE_JH_FINAL_FRIC", 1.0f)};
const float kLineJhPosAg[3] = {param("LINE_JH_INIT_AG", -1.0f), param("LINE_JH_MED_AG", -1.0f), param("LINE_JH_FINAL_AG", -1.0f)};
const float kLineJhPosZag[3] = {param("LINE_JH_INIT_ZAG", -1.0f), param("LINE_JH_MED_ZAG", -1.0f), param("LINE_JH_FINAL_ZAG", -1.0f)};
// ... and for every ʤ (< 0 / 1: the shared ʃ ʒ ʧ ʤ value): LINE_JH_FRIC_X the place of its frication (LINE_SH_X),
// LINE_JH_SLIT x its slit, LINE_JH_LIP its lip rounding (LINE_POST_LIP), LINE_JH_JET_D its jet (LINE_POST_JET_D),
// LINE_JH_LPR its band low-pass (LINE_POST_LPR), LINE_JH_FC_MIN its noise corner floor (LINE_POST_FC_MIN),
// LINE_JH_HW its constriction width (LINE_POST_HW), LINE_JH_FADE its frication fade (LINE_VFRIC_FADE),
// LINE_JH_OWN_REL_FRIC its release hiss (LINE_JH_REL_FRIC, shared with ʧ), LINE_JH_PRESS the lung pressure share
// through its frication, LINE_JH_FINAL_EXPAND x a final ʤ's closure expansion flow (its voice bar).
const float kLineJhFricX = param("LINE_JH_FRIC_X", -1.0f), kLineJhSlit = param("LINE_JH_SLIT", 1.0f),
            kLineJhLip = param("LINE_JH_LIP", -1.0f), kLineJhJetD = param("LINE_JH_JET_D", -1.0f),
            kLineJhLpr = param("LINE_JH_LPR", -1.0f), kLineJhFcMin = param("LINE_JH_FC_MIN", -1.0f),
            kLineJhHw = param("LINE_JH_HW", -1.0f), kLineJhFade = param("LINE_JH_FADE", -1.0f),
            kLineJhOwnRelFric = param("LINE_JH_OWN_REL_FRIC", -1.0f), kLineJhPress = param("LINE_JH_PRESS", 1.0f),
            kLineJhFinalExpand = param("LINE_JH_FINAL_EXPAND", 1.0f);
// The hiss's course through the frication (1 / 0: off): LINE_JH_RAMP_MS the release hiss rises from LINE_JH_RAMP0 x its
// gain to the full gain over this long (Kokoro's initial ʤ hiss rises 20 dB over 20 ms; puhuja's switched on in 5 ms,
// snap.py 4-27 dB at the release); LINE_JH_DECAY x the gain from LINE_JH_DECAY_AT of the way through the frication
// (Kokoro's final ʤ fades from -14 to -25 dB re the word's peak).
const float kLineJhRampMs = param("LINE_JH_RAMP_MS", 0.0f), kLineJhRamp0 = param("LINE_JH_RAMP0", 0.3f),
            kLineJhDecay = param("LINE_JH_DECAY", 1.0f), kLineJhDecayAt = param("LINE_JH_DECAY_AT", 0.5f);
// ʒ's own jet thickness / band low-pass / frication fade window (< 0: the ʃ ʒ ʧ ʤ LINE_POST_JET_D, LINE_POST_LPR, LINE_VFRIC_FADE; ʒ lab 2026-11-09)
const float kLineZhJetD = param("LINE_ZH_JET_D", 0.055f), kLineZhLpr = param("LINE_ZH_LPR", 4.0f), kLineZhFade = param("LINE_ZH_FADE", -1.0f);
const float kLineVoicedSoft = param("LINE_VOICED_SOFT", 2.0f);     // a voiced stop's closure: its sections lose this many times more (soft tissue; 6 -> 2 on 2026-11-04 with the wall radiation, Whisper on d words)
const float kLineVoicedExtraCm = param("LINE_VOICED_EXTRA_CM", 0.15f);  // b: the lip tube is this much longer (the lips move forward off the teeth; Jyrki: "perhaps 1.5 mm is correct")
const float kLineLabialO = param("LINE_LABIAL_O", 1.0f);  // 0: labial stops keep the vowel's shape; 1: b takes o's; 2: b and p
const float kLineBExtraMs = param("LINE_B_EXTRA_MS", 0.0f);  /* 30 until 2026-11-06: Jyrki chose the 46 ms closure (beliefs-vb0bx0; Kokoro 20 ms); kennedy Whisper 856 -> 812 */  // b holds its closure this much longer than the table says
const float kLineBExpandFlow = param("LINE_B_EXPAND_FLOW", -1.0f);  // b's own LINE_EXPAND_FLOW (the voice bar in its closure); < 0: the shared one (2026-11-04)
const float kLineBVbarFlow = param("LINE_B_VBAR_FLOW", 400.0f);  // b's expansion flow in a voiced (expanding: between vowels) closure only, the English initial b left alone; < 0: off. With LINE_EXPAND_LP 50: 400 (2026-11-05)
const float kLineOnsetMs = param("LINE_ONSET_MS", 80.0f);
const float kLineOnsetShare = param("LINE_ONSET_SHARE", 0.5f);
const float kLineOnsetKickAg = param("LINE_ONSET_KICK_AG", 0.02f);  // the soft onset starts the folds with a brief near-closure (cm^2; 0: none)
const float kLineOnsetKickMs = param("LINE_ONSET_KICK_MS", 3.6f) /* cost Whisper 898 composite 24.95/28.30 95d-all3 */;  // ... at most this share of the sound's own length (2026-11-05: "It cannot." had a silent "It")  // a vowel after a pause: breathy opening adducting into voice over this long
const float kLineReleaseScale = param("LINE_RELEASE_SCALE", 1.3f) /* cost 1058 whisper-sens1-combo-phase1-err */;  // a stop's release time on the line, relative to the cascade's table
const float kLineFinalFricMs = param("LINE_FINAL_FRIC_MS", 40.0f);  // a final voiceless fricative: the pause's glottal closing waits this long
const float kLinePausePressMs = param("LINE_PAUSE_PRESS_MS", 80.0f);
const float kLineFinalNasalOpen = param("LINE_FINAL_NASAL_MS", 100.0f);  // a pause after a nasal closes the velum over this long (0: kBoundaryMs, the old cut-off)
// A phrase-initial nasal (m n ŋ right after a pause, "Name.") starts in two
// steps, as a speaker does: the lips or tongue close the mouth and the velum
// (soft palate) drops first, then the vocal folds come together (adduct)
// and start vibrating. Lead: how early, before the nasal's nominal start,
// the mouth gesture begins; voice lag: how long after that the folds start
// to close; onset: how long they take to reach full voicing.
// 20 -> 40 ms (2026-11-04): after the gentler folds the initial m's murmur was
// ~60 ms (Kokoro ~80); Whisper heard initial m as m 19 -> 25 of 32.
const float kLineNasalLeadMs = param("LINE_NASAL_LEAD_MS", 40.0f);   // a phrase-initial nasal's gesture starts this long before it, in the pause (<= 0: the vowel's soft onset instead)
const float kLineNasalOnsetMs = param("LINE_NASAL_ONSET_MS", 25.0f);  // ... and its voicing adducts over this long
const float kLineNasalRelMs = param("LINE_NASAL_REL_MS", 35.0f);       // a nasal's release into a vowel: the mouth opens over this long
const float kLineNasalVelMs = param("LINE_NASAL_VEL_MS", 30.0f);       // ... the velum closes over this long
const float kLineNasalVelLagMs = param("LINE_NASAL_VEL_LAG_MS", 40.0f); // ... starting this long after the release
// The same for n (and ŋ) alone; < 0: as for m. Whisper on the n-initial
// words (2026-11-04, 40 variants): n wants a longer lead and no velum lag
// (19-20 of 34 heard starting with n, 10 before), m the opposite.
const float kLineNRelMs = param("LINE_N_REL_MS", -1.0f);
const float kLineNCloseMs = param("LINE_N_CLOSE_MS", -1.0f);
const float kLineLCloseMs = param("LINE_L_CLOSE_MS", 40.0f);  // the l's closing ramp from a vowel (-1: kBoundaryMs); "all": the 20 ms close dropped 6 dB and F1 441 -> 356 at once, a snap (Jyrki 2026-11-05)  // the n's closing ramp from a vowel (-1: kBoundaryMs, 20 ms)
// An m after a vowel (2026-11-04): the lips closed with one 20 ms linear ramp
// starting 10 ms before the m, so the vowel kept its formants to its last
// frame and then stepped into the murmur (no falling F2, the lips' place cue;
// Whisper heard "Some" as "son"). Approach: this long before the m the lips
// narrow to LINE_M_APPROACH_A cm^2 at the lips' place, as a b's do; then the
// usual closure seals them (0: off). 50 ms: F2 falls 1300 -> 1000 Hz into
// the m like Kokoro's; Whisper heard final m as m 4 -> 16 of 24, medial 11 -> 14 of 18.
const float kLineMApproachMs = param("LINE_M_APPROACH_MS", 50.0f);
const float kLineMApproachA = param("LINE_M_APPROACH_A", 0.4f);
// The same approach for ŋ after a vowel (2026-11-04): the tongue's back
// narrows at the velum (ŋ's place) before the seal, so the vowel's F2 and F3
// converge into it (the "velar pinch"); without it ŋ was heard as n. 0: off.
// 50 ms: Whisper heard ŋ as ŋ 31 -> 41 of 48 (as n 10 -> 2, as m 1 -> 4).
const float kLineNgApproachMs = param("LINE_NG_APPROACH_MS", 50.0f);
const float kLineNgApproachA = param("LINE_NG_APPROACH_A", 0.4f);
// Stop knobs (2026-11-04), all off by default (today's sound), to be tried one
// at a time; pass R2 b and g (DESIGN_phoneme_process.md) asked for them.
// k and g into a vowel open in two steps: first to LINE_K_BURST_A cm^2 for
// LINE_K_BURST_MS ms (the velar burst hisses through the slit), then to the
// vowel, as a stop before a fricative does (LINE_BURST_AREA). 0: off.
const float kLineKBurstMs = param("LINE_K_BURST_MS", 35.0f);
const float kLineKBurstA = param("LINE_K_BURST_A", 0.2f);
// The k's burst jet (LINE_K_JET_D cm, lab knob 2026-11-10, K agent 2; 0: the slit's own height): a thicker jet lowers
// the slit's noise band toward the velar 1.3-2.2 kHz (Kokoro's initial k: 1.6-2.6 kHz; the 0.12 cm^2 slit alone hissed
// at 3.8-4 kHz). The lab (top-300 k words, two draws; base 138 and 126 k errors of 630): LINE_K_BURST_MS 35,
// LINE_K_BURST_A 0.2, LINE_K_JET_D 0.2 103 and 98 (a velar affrication through the slit: k heard as p 44 -> 28); on
// 300 hard words (initial k + front vowel, final k, kl kr kw) 303 -> 246 of 674.
const float kLineKJetD = param("LINE_K_JET_D", 0.3f);
// k's own puff noise (< 0: LINE_STOP_ASP), like LINE_T_STOP_ASP.
const float kLineKStopAspMul = param("LINE_K_STOP_ASP", 10.0f);
// A vowel before k or g narrows at the velum this long before the closure
// (to LINE_KG_APPROACH_A cm^2), as before ŋ; and before b or p at the lips,
// as before m. 0: off.
const float kLineKgApproachMs = param("LINE_KG_APPROACH_MS", 50.0f);
const float kLineKgApproachA = param("LINE_KG_APPROACH_A", 0.4f);
const float kLineBpApproachMs = param("LINE_BP_APPROACH_MS", 0.0f);
const float kLineBpApproachA = param("LINE_BP_APPROACH_A", 0.4f);
// After a k or g released into a vowel, the vowel's first 25 ms are asked to
// keep a narrow tongue hump at the velar place (the vowel's own shape
// otherwise), so F2 and F3 start close together and part into the vowel (the
// velar pinch). The value is the demand's weight (0: off; ~0..1).
const float kLineKgRelShape = param("LINE_KG_REL_SHAPE", 1.0f);
// A phrase-initial stop's breath starts this long before its release (0: as
// the mouth closes), rising over LINE_PAUSE_STOP_PRESS_RAMP_MS.
// English g d after a pause or a voiceless sound: the closure's glottal
// opening, from voicing (0, off) toward open (1), as kEnglishBOpening for b.
const float kLineEnGdDevoice = param("LINE_EN_GD_DEVOICE", 0.2f);
const float kLinePauseStopPressMs = param("LINE_PAUSE_STOP_PRESS_MS", 0.0f);
const float kLinePauseStopPressRampMs = param("LINE_PAUSE_STOP_PRESS_RAMP_MS", 30.0f);
// A short voice bar in b d g: the closure's expansion draws LINE_VBAR_FLOW
// cm^3/s (< 0: the usual LINE_EXPAND_FLOW / LINE_B_EXPAND_FLOW) only for the
// closure's first LINE_VBAR_MS ms, then stops, so the pressure builds for the
// burst. 0: off (the expansion runs through the whole closure).
const float kLineVbarMs = param("LINE_VBAR_MS", 0.0f);
const float kLineVbarFlow = param("LINE_VBAR_FLOW", 1000.0f);
// Voicing into a voiceless closure (the blend exploration, 2026-11-05): Kokoro's
// p t k closures carry a voice bar at -21..-28 dB re the vowels, nearly all of it
// under 300 Hz, where puhuja's fell to a -44 dB floor within 8 ms: the folds
// stall once the shut mouth's pressure equalises. The cavity expands in every
// stop, so for the closure's first LINE_VSTOP_VBAR_MS the line draws
// LINE_VSTOP_VBAR_FLOW (cm^3/s; < 0: LINE_EXPAND_FLOW) as in a voiced closure
// (0: off), and the glottis's opening at the closure can start
// LINE_VSTOP_AG_LAG_MS later.
const float kLineVstopVbarMs = param("LINE_VSTOP_VBAR_MS", 0.0f);
const float kLineVstopVbarFlow = param("LINE_VSTOP_VBAR_FLOW", -1.0f);
const float kLineVstopAgLagMs = param("LINE_VSTOP_AG_LAG_MS", 0.0f);
const float kLineGdAgLagMs = param("LINE_GD_AG_LAG_MS", 0.0f);  // a devoiced b d g: the glottis opens this long after the closure has sealed (ms; 0: with it)
// ... and the glottis opens into a voiceless fricative over LINE_ABDUCT_MS (the
// generic 15; Kokoro's vowel falls over ~40 ms into an s, puhuja's in 16).
const float kLineAbductMs = param("LINE_ABDUCT_MS", 35.0f);
// The burst noise of b d g at their release, x this (1: today; the line's
// frication gain while the stop opens, the cascade's burst noise gain).
const float kLineVStopBurst = param("LINE_VSTOP_BURST", 0.0f);  // 1.0 until 2026-11-06 (Jyrki: the b of "belief" had "too much aspiration"; the vowel after it carried hiss at -18 dB above 5 kHz, Kokoro -42; chose beliefs-vb0bx0)
const float kLineXSnapArea = param("LINE_X_SNAP_AREA", 1.0f);  // the constriction place jumps to its target while the hump is wider than this (cm^2; 0: glides)
// The lips and the tongue are two articulators: when the place target moves
// from one to the other (x >= 0.94 is the lips: p b m w f v), the place jumps
// at once (0) or over this time constant (ms) instead of the 20 ms glide; < 0:
// glides. The one hump slid from the tongue to the lips while shut: "and me"
// m closed at x 0.80 (n's 0.78, m's 0.97), "for mankind" at 0.85, its first
// 30-50 ms an n's murmur (nasal review 1.2, 2026-11-11). Proposed 8: at 0
// guards.py found 26 clicks / bangs, at 4 "Napkin" n -> p snapped (snap.py
// 0 -> 10.9 / 8); at 8 the snap list is 6.33 dB (base 6.55).
const float kLineXArticMs = param("LINE_X_ARTIC_MS", 8.0f);
// LINE_X_SNAP_AREA only for a closure's place (a stop's, a nasal's; 1; 0: every place). The fricatives
// keep their glide from the vowel: at their own place from the vowel on (the snap), the v of "revolutionary"
// and the ʃ of "-lution" snapped (tools/snap.py, female v 0 -> 10.1, "revolutionary war" male v 0 -> 18.4),
// their onsets were tuned on the glide.
const bool kLineXSnapClosures = param("LINE_X_SNAP_CLOSURES", 1) > 0;
// A stop's approach (the narrowing 2 -> 0.15 cm^2 before its seal) at the
// stop's own place (1; 0: the place set by the closure, after the approach).
// It had none: the narrowing formed at the last consonant's place, "keep" p
// sealed at 0.68-0.78 and slid 2.3 cm to the lips while closed, "Bad" d
// narrowed at the lips (artsynth review B1, 2026-11-11).
const float kLineStopApproachX = param("LINE_STOP_APPROACH_X", 1.0f);
const float kLineKRelFric = param("LINE_K_REL_FRIC", 0.3f);  // k's release hiss (x LINE_FRIC while the closure opens; 1: as before)
const float kLinePRelFric = param("LINE_P_REL_FRIC", 0.3f);  // p's
// t's (2026-11-06, Jyrki: in "torch has" "the t sounds off"): across kennedy puhuja's t bursts peaked -2.7 dB re the
// vowel (p90 +3.4), Kokoro's -17.8 (p90 -9.1); the puffs (2.5 kHz+ over the vowel's first 40 ms) -12 vs -23.
const float kLineTRelFric = param("LINE_T_REL_FRIC", 0.265f);  // 2026-11-07, Jyrki: "t like in tiger could be 1 db stronger, both t and its associated aspiration", then +0.7, then "a bit softer at 0.5 db": 0.25 -> 0.265
// ʧ's frication (x LINE_FRIC through its ʃ phase; 1: as ʃ). Kokoro's ʧ hiss is 7-9 dB louder than puhuja's
// (2.5-5 kHz -10.7 vs -18.3 re the vowel, > 5 kHz -17.4 vs -26.9; "torch has", 2026-11-06).
const float kLineChFric = param("LINE_CH_FRIC", 2.0f);
// h after a voiceless consonant (1) or also after a pause (2) is the breathy voiced h of the voiced context (0: the
// open-glottis hiss). Kokoro's h is breathy voiced in every context (centroid 120-750 Hz, > 5 kHz -25..-47 dB re the
// vowel); puhuja's after ʧ, s or a pause was a hiss with its centroid at 5-8.5 kHz, > 5 kHz -5..-8 dB ("torch has":
// Jyrki 2026-11-06, "both of the h are also not good, those are poorly integrated and pop out").
const int kLineHVoicedCtx = param("LINE_H_VOICED_CTX", 2);
// ... after a pause (LINE_H_VOICED_CTX 2) the folds start from the closed pause with the onset kick (LINE_ONSET_KICK_*)
// and adduct to this glottal area (cm^2; < 0: the voiced h's 0.125, at which they did not start: the h was silent).
const float kLineHPauseAg = param("LINE_H_PAUSE_AG", 0.37f) /* cost words1000 935/934 h-words 33/36 of 94 vs 30/28 pauseag30 */;
// The h's breath on the line (HH agent, 2026-11-10). Against Kokoro (am_adam, af_heart; isolated words, the h's mean
// spectrum re the word peak) puhuja's initial h is 10-25 dB weaker under 2 kHz and its 4.5-7 kHz hiss stands alone
// (female "hope": -19.5 there, every other band -35..-45): Whisper hears f (h before ʌ 86 % errors, 69 of them f;
// ɝ 72 %, eɪ 79 %, oʊ 38 %). Kokoro's h is the vowel's formants on a dark breath (bands 0.3-3 kHz -19..-30).
// LINE_H_GFC: the glottal noise band's corner (Strouhal, St U / d) x this during an h (1: as every aspiration).
const float kLineHGfc = param("LINE_H_GFC", 0.4f);
// LINE_H_LPR: the glottal noise band's low-pass ratio during an h (0: LINE_GLOTTAL_LPR / LINE_NOISE_LPR).
const float kLineHLpr = param("LINE_H_LPR", 0.0f);
// LINE_H_BREATH: an unthresholded breath (the LINE_BREATH mechanism, x LINE_ASP, its band LINE_BREATH_FC) during an
// h, whatever the glottis's width (0: off).
const float kLineHBreath = param("LINE_H_BREATH", 0.0f);
// LINE_H_PRESS: the lung pressure eases to (< 1) or rises to (> 1) this share through an h, back on the next sound (1: off).
const float kLineHPress = param("LINE_H_PRESS", 1.0f);
// LINE_H_PAUSE_KICK: the onset kick of an h after a pause (1; 0: the folds adduct to LINE_H_PAUSE_AG directly). The
// kick is a voiced LF thump at the h's start (male "have": 0-1 kHz -22 dB re the vowel 15 ms before the breath).
const float kLineHPauseKick = param("LINE_H_PAUSE_KICK", 0.0f);
// LINE_H_VOWEL_AG_MS: how fast the folds adduct into the vowel after an h (ms; 0: LINE_ADDUCT_MS, 30). Kokoro's h runs
// into the vowel as a breathy crescendo over ~30 ms; puhuja's vowel switched on in 10-15 ms.
const float kLineHVowelAgMs = param("LINE_H_VOWEL_AG_MS", 0.0f);
// LINE_H_LIP / LINE_H_AC: the h's tract (the next vowel's shape) with its lip opening / tongue constriction area x
// this (1: the vowel's own); LINE_H_OPEN_MIX: the h's shape mixed this far toward the open a (0: off).
const float kLineHLip = param("LINE_H_LIP", 1.0f), kLineHAc = param("LINE_H_AC", 1.0f), kLineHOpenMix = param("LINE_H_OPEN_MIX", 0.4f);
// LINE_H_APH: the h's pharynx area x this (1: the vowel's own); LINE_H_GFC_F: the female voice's LINE_H_GFC x this
// (her h's 4.5-7 kHz band stands 6-8 dB over the male's, -29 vs -37 re the word peak; Kokoro af_heart -36).
const float kLineHAph = param("LINE_H_APH", 1.0f), kLineHGfcF = param("LINE_H_GFC_F", 1.0f);
// In a sentence (Kennedy, koko-0006-m vs puhuja, 16 / 30 h's, the h's mean spectrum re the next vowel's peak) the
// medial h is the other way round: puhuja -18.7 -18.8 -22.8 -29.5 -33.4 -36.0 -42.2 dB (0.1-0.3 ... 4.5-7 kHz),
// Kokoro -15.1 -22.5 -27.1 -25.9 -25.9 -34.2 -40.5: too much voice under 1.2 kHz, 4-7 dB too little F2/F3 breath.
// LINE_H_GFC_MED: LINE_H_GFC for an h not after a pause (< 0: LINE_H_GFC); LINE_H_PAUSE_ASP: the noise gain of an
// h after a pause (< 0: LINE_VH_ASP, as the medial h's).
const float kLineHGfcMed = param("LINE_H_GFC_MED", -1.0f), kLineHPauseAsp = param("LINE_H_PAUSE_ASP", -1.0f);
// A phrase-initial b d g is prevoiced: the closure is formed in the pause and
// the folds start to vibrate this long before the release (as a phrase-initial
// nasal's lead). 0: off.
const float kLineVStopLeadMs = param("LINE_VSTOP_LEAD_MS", 0.0f);
const float kLineNVelMs = param("LINE_N_VEL_MS", 20.0f);
const float kLineNVelLagMs = param("LINE_N_VEL_LAG_MS", 20.0f);
const float kLineNLeadMs = param("LINE_N_LEAD_MS", 40.0f);
const float kLineNVoiceLagMs = param("LINE_N_VOICE_LAG_MS", -1.0f);
// Picks n's knob for n and ŋ when it is set (>= 0), else m's.
inline float nasalKnob(Ph ph, float m, float n) { return ph != Ph::M && n >= 0 ? n : m; }
const float kLineNasalVoiceLagMs = param("LINE_NASAL_VOICE_LAG_MS", 20.0f);  // a phrase-initial nasal's voicing starts this long after its gesture (m; n: LINE_N_VOICE_LAG_MS)
// The same set-up for the other voiced consonants that can open a phrase,
// l, w, j, ɹ (2026-11-04). Before, they took the vowel's soft onset: the
// glottis starting half open (breathy) and closing into voice over
// LINE_ONSET_MS (80 ms), longer than the l w j r itself. Their narrow mouth
// opening holds air back, so little of the lung pressure falls across the
// glottis, too little to set half-open folds vibrating: the consonant was a
// 25-40 ms near-silent hole (-30..-40 dB), and the vowel then started hard:
// heard as b, g, d or nothing ("Life." -> "hi", "Will." -> "bill", "Yes."
// -> "good", "Read." -> "bead"). Lead, voice lag and onset as for the
// nasals; onset < 0: the old soft onset. J and R (ɹ, r): exactly -1 takes
// GLIDE's (w's) value.
const float kLineLLeadMs = param("LINE_L_LEAD_MS", 0.0f);
const float kLineLVoiceLagMs = param("LINE_L_VOICE_LAG_MS", 0.0f);
const float kLineLOnsetMs = param("LINE_L_ONSET_MS", 25.0f);  // (2026-11-04: initial l heard 33 -> 51 of 60 with no lead and no lag; a lead put a ə before it)
const float kLineGlideLeadMs = param("LINE_GLIDE_LEAD_MS", 0.0f);
const float kLineGlideVoiceLagMs = param("LINE_GLIDE_VOICE_LAG_MS", 0.0f);
const float kLineGlideOnsetMs = param("LINE_GLIDE_ONSET_MS", -1.0f);
const float kLineWOnsetMs = param("LINE_W_ONSET_MS", -2.0f);  // (W agent round I, lab; -2: LINE_GLIDE_ONSET_MS) the phrase-initial w's voice onset
const float kLineJLeadMs = param("LINE_J_LEAD_MS", -1.0f);
const float kLineJVoiceLagMs = param("LINE_J_VOICE_LAG_MS", -1.0f);
const float kLineJOnsetMs = param("LINE_J_ONSET_MS", 25.0f);  // (2026-11-04: with the j shaped in the pause, see LINE_J_PRE_SHAPE_MS)
const float kLineRLeadMs = param("LINE_R_LEAD_MS", 0.0f);
const float kLineRVoiceLagMs = param("LINE_R_VOICE_LAG_MS", 0.0f);
const float kLineROnsetMs = param("LINE_R_ONSET_MS", 15.0f);  // (2026-11-04: initial ɹ heard 26 -> 35 of 60, no lead or lag)
// An initial j's tongue shape (i's) was demanded from 30 % of the j, and the
// pause had planned its shapes to its end before that demand existed: F2 rose
// from the pause's ə (1550 Hz) to ~1750 by the j's end (Kokoro 2100-2200 held
// from the start), heard as g, l, d or dropped (2026-11-04). The shape demand
// of an initial j / l now starts this long before it, in the pause (0: at
// 30 %), and a pause before one leaves this much of its end to the next
// phrase's plan. Male j F2 1840-1930 from its start, female 2150-2300; l F2
// ~950 held (Kokoro ~950; was 1200 -> 1100). With LINE_J_ONSET_MS 25, Whisper
// (m+f): initial j 10 -> 23 of 34, words 7 -> 9; l, ɹ, w the same.
const float kLineJPreShapeMs = param("LINE_J_PRE_SHAPE_MS", 40.0f);
const float kLineLPreShapeMs = param("LINE_L_PRE_SHAPE_MS", 40.0f);
const float kLineWPreShapeMs = param("LINE_W_PRE_SHAPE_MS", 60.0f);  // the same for a phrase-initial w (0: at 30 %; 2026-11-09 W agent: the w's F2 sat at the pause's 1100-1170 Hz for its first 15-20 ms)
const float kLineWKeyTo = param("LINE_W_KEY_TO", 1.2f);
const float kLineWOnMs = param("LINE_W_ON_MS", -1.0f);  // (W agent round I, lab; < 0: the 20 ms boundary) the lips close into a w over this
// Lab knobs (R agent 2026-11-10; defaults reproduce the previous sound): a phrase-initial ɹ shaped in the pause this long
// before its voice (0: at 30 %), and the ɹ holds its shape from / to these shares of it (-1: the 0.3 / 0.7 of all sounds).
const float kLineRPreShapeMs = param("LINE_R_PRE_SHAPE_MS", 0.0f);
// A phrase-initial ɜ ("earn", "early", "urban") shaped in the pause this many ms before its start, like j l w ɹ (0: off;
// ER agent 2026-11-11: word-initial ɜ 47 % errors against 12 % overall).
const float kLineErPreMs = param("LINE_ER_PRE_MS", 0.0f);
const float kLineRKeyFrom = param("LINE_R_KEY_FROM", -1.0f);
const float kLineRKeyTo = param("LINE_R_KEY_TO", -1.0f);  // a w holds its shape to this share of it (the vowels 0.7; 2026-11-09 W agent)
const float kLinePausePlanHoldMs = param("LINE_PAUSE_PLAN_HOLD_MS", 150.0f);
struct SonorantOnset { float leadMs, voiceLagMs, onsetMs; };
// Returns onsetMs < 0 for sounds that take the vowel's soft onset.
inline SonorantOnset sonorantOnset(Ph ph) {
  auto pick = [](float own, float glide) { return own != -1.0f ? own : glide; };  // -1: as the glides; another onset < 0: the soft onset
  switch (ph) {
    case Ph::L: return {kLineLLeadMs, kLineLVoiceLagMs, kLineLOnsetMs};
    case Ph::J:
      return {pick(kLineJLeadMs, kLineGlideLeadMs), pick(kLineJVoiceLagMs, kLineGlideVoiceLagMs),
              pick(kLineJOnsetMs, kLineGlideOnsetMs)};
    case Ph::RR: case Ph::R:
      return {pick(kLineRLeadMs, kLineGlideLeadMs), pick(kLineRVoiceLagMs, kLineGlideVoiceLagMs),
              pick(kLineROnsetMs, kLineGlideOnsetMs)};
    case Ph::W: return {kLineGlideLeadMs, kLineGlideVoiceLagMs, kLineWOnsetMs != -2.0f ? kLineWOnsetMs : kLineGlideOnsetMs};
    case Ph::V: return {kLineGlideLeadMs, kLineGlideVoiceLagMs, kLineGlideOnsetMs};
    default: return {0, 0, -1};
  }
}
const float kLinePrePressMs = param("LINE_PRE_PRESS_MS", 80.0f);     // ... and rises again this long before the next phrase's first sound  // the breath pressure falls to zero over this at a pause
const float kLineDentalAc = param("LINE_DENTAL_AC", 0.3121f) /* cost 1814 whisper-run1-phon_err */ /* 2026-11-04 (Jyrki): 0.0324 left θ's slit at 0.0032 cm^2, practically closed (-45 dB, no hiss; Kokoro -24); 0.3: -22 dB, Whisper neutral */;        // θ ð slit area on the line, relative to the aero table (weak fricatives)
const float kLineFAc = param("LINE_F_AC", 0.306f) /* cost 1814 whisper-run1-phon_err */;                   // f likewise: the lower lip on the upper teeth all but seals (Jyrki 2026-11-03: "more or less fully closing the opening, only what comes from the closing being not tight gives air through"); 0.03 cm^2 holds 95 % of the lung pressure behind it, 112 cm^3/s, Kennedy's f -9.8 dB under the vowels (readers -5..-16.5; 0.5: 190 cm^3/s, -7.1)
// f's lip-teeth contact (cm, half of its length along the tract). With 1.0
// the contact at 0.95 reached the lips, and the hiss source (placed just past
// the contact, clamped before the lips) landed inside its flank: in
// "different" the section's area fell 2.9 -> 0.29 cm^2 while the place glided
// in from the d, and the hiss fell 30 dB through the f at a steady 104
// cm^3/s (the readers' f is steady, 2026-11-04). 0.5: the hiss steady (its
// middle third -39 -> -22 dB re the ɪ, 2-8 kHz), but bright: heard as s.
const float kLineFHw = param("LINE_F_HW", 0.5f);
// f's jet thickness for its noise band (cm; 0: the slit's height 0.03 cm^2 /
// LINE_SLIT_W = 0.3 mm, whose Strouhal corner St U / d is ~12 kHz: the hiss
// peaked at 3-5 kHz like an s). 0.25 (the air leaks through a few wider gaps
// between lip and teeth): corner ~1.4 kHz, the hiss strongest at 1.5-5 kHz
// like the readers' f; Whisper F heard 13 -> 29 of 52 (2026-11-04).
const float kLineFJetD = param("LINE_F_JET_D", 0.2f);
// f's noise band low-pass / its corner (0: LINE_NOISE_LPR). The hiss is
// made at the lips with no front cavity, so the radiation's +6 dB/octave
// tilted the flat band: it rose 25 dB from 1 to 7 kHz, an s-like whistle
// (the readers' and Kokoro's f are level from 0.5 to 11 kHz, 2026-11-05).
const float kLineFLpr = param("LINE_F_LPR", 0.0f);
// ... for an f that does not start a stressed syllable (2026-11-06, Jyrki: "we can follow your refined proposal about f"):
// Kokoro's typical f is dark (median centroid 1.3 kHz, the voice bleeding in) and only its stressed f ("forms") is the
// bright 5.3 kHz one, so the thin jet and the gain lift go to stressed onsets; the voice tail (LINE_ABDUCT_MS) is global.
const float kLineFJetDUnstr = param("LINE_F_JET_D_UNSTR", 0.2f);
const float kLineFFricUnstr = param("LINE_F_FRIC_UNSTR", 1.062f) /* cost Whisper 897 composite 24.46/27.74 96b-f-3db */;
// v's jet (2026-11-06, Jyrki: "govern", "the v whisles a bit too much ... v has too much high frequency content I think,
// not just too much power -- I'd like to have a strong v, but just not whistling"): Kokoro's v is a voiced approximant
// whose spectrum falls steadily (2-3 kHz -36..-42 dB re its low peak, 4-5 kHz -46); puhuja's carried a flat noise
// shelf -25..-30 dB from 1.6 to 6.3 kHz and, in "governments", a 3.1 kHz hump at -9 dB. Thickness in cm (0: the slit's
// height) and the band's low-pass ratio (0: LINE_NOISE_LPR), as LINE_F_JET_D / LINE_F_LPR.
const float kLineVJetD = param("LINE_V_JET_D", 0.1f);
const float kLineVLpr = param("LINE_V_LPR", 5.0f);
const float kLineSFric = param("LINE_S_FRIC", 1.0f);  // s frication gain (x LINE_FRIC). 2026-11-05, the s of "save": its steady hiss -16 dB re the vowel, Kokoro -10 (Jyrki: "more open and less z like"); a t release into an s keeps it.
const float kLineFFric = param("LINE_F_FRIC", 1.062f) /* cost Whisper 897 composite 24.46/27.74 96b-f-3db */;  // ... and f's frication gain (x LINE_FRIC): the narrower band is quieter
// ʃ (2026-11-06, Jyrki: "shall" -- "the sh sounds a little off"): puhuja's ʃ was 6 dB under Kokoro's across kennedy
// (-9.2 vs -3.5 dB re the vowel), 4 dB under at 2.5-5 kHz and 13 dB under above 5 kHz; "shall" itself -10 / -14 / -30 dB
// vs Kokoro -3.5 / -5.5 / -13 (centroid 2.8 vs 3.4 kHz). Kokoro's ʃ before a full vowel is bright (centroid 3-3.7 kHz),
// its -tion ʃ dark (0.4-1.3 kHz), which Jyrki wanted kept dark. LINE_SH_FRIC: every ʃ's gain (x LINE_FRIC); before a
// vowel other than ə additionally LINE_SH_VOWEL_FRIC, and the place LINE_SH_VOWEL_X (< 0: LINE_SH_X).
const float kLineShFric = param("LINE_SH_FRIC", 1.0f);
const float kLineShVowelFric = param("LINE_SH_VOWEL_FRIC", 3.0f);
// ... for a ʃ that does not start its word (a word-final ʃ linking into the next word's vowel, "abolish all": Jyrki
// 2026-11-06 "sounds too sharp"): at 6 it ran 7 dB louder than Kokoro's (level +1.4 vs -3.8 re the vowel, > 5 kHz
// -8.6 vs -16) while the word-initial ones ("shall") sat at Kokoro's. < 0: LINE_SH_VOWEL_FRIC.
const float kLineShVowelFricLink = param("LINE_SH_VOWEL_FRIC_LINK", 2.2f);
const float kLineShVowelX = param("LINE_SH_VOWEL_X", 0.78f);
const float kLineShVowelAc = param("LINE_SH_VOWEL_AC", 1.0f);  // ... and its slit area x this (narrower: a faster jet, a higher band; it only lost level)
const float kLineFinalPressMs = param("LINE_FINAL_PRESS_MS", 60.0f);  // (2026-11-04: final k heard 4 -> 8 of 80, Whisper; the "uh" after final p t k gone)
//  // after a final voiceless stop the breath stops over this long from the release (0: at the pause)
const float kLineFinalClose = param("LINE_FINAL_CLOSE", 1.0f);  // ... and the glottis closes after LINE_FINAL_ASP_MS (0: stays open)
const float kLineFinalAg = param("LINE_FINAL_AG", 0.3f);  // ... the glottis during its release (cm^2)
const float kLineFinalAspMul = param("LINE_FINAL_ASP", 1.0f);  // ... its glottal noise relative to LINE_ASP
const float kLineFinalAspMs = param("LINE_FINAL_ASP_MS", 60.0f);  // a final voiceless stop's open-glottis release before the pause closes it
// A small push from the diaphragm before English v f s ʃ, d g and the
// aspirated p t k (2026-11-04, Jyrki): the lung pressure rises briefly by
// LINE_PULSE (0.15 = +15 %), peaks LINE_PULSE_LEAD_MS before the hiss or the
// stop's release, and falls back over LINE_PULSE_FALL_MS: more air for the
// fricative's jet and the burst. 0: no pulse.
const float kLinePulse = param("LINE_PULSE", 0.0f);
const float kLinePulseLeadMs = param("LINE_PULSE_LEAD_MS", 10.0f);  // the peak this long before the sound
const float kStressPulse = param("LINE_STRESS_PULSE", 0.0f);     // a stressed vowel's raise as a pressure pulse, x the raise (0: none)
const float kStressHold = param("LINE_STRESS_HOLD", 1.0f);       // ... and the share of the raise held through the vowel (1: all, today's)
const float kStressPeakMs = param("LINE_STRESS_PEAK_MS", 10.0f); // the pulse's peak after the vowel's start (ms)
const float kStressFallMs = param("LINE_STRESS_FALL_MS", 120.0f);  // ... and its fall back (ms)
const float kLinePulseRiseMs = param("LINE_PULSE_RISE_MS", 40.0f);  // ... reached over this
const float kLinePulseFallMs = param("LINE_PULSE_FALL_MS", 60.0f);  // ... and back to normal over this after the peak
const float kLinePulseFric = param("LINE_PULSE_FRIC", 1.0f);  // the pulse's size before the fricatives, x LINE_PULSE
const float kLinePulseStop = param("LINE_PULSE_STOP", 1.0f);  // ... before the stops' releases
const float kLineBurstArea = param("LINE_BURST_AREA", 0.2f) /* cost 1814 whisper-run1-phon_err */;  // a stop's release opening before a fricative (cm^2)
const float kLineBurstMs = param("LINE_BURST_MS", 20.0f) /* cost 1814 whisper-run1-phon_err */;     // ... for this long
const float kLineFric = param("LINE_FRIC", 7.54f) /* 2026-11-07: x1.6 (4.7182 -> 7.54) against the 4-5 dB louder voice of the sharper closing phase; kennedy round 60: s -4.1 -> -1.5, ʃ -7.4 -> -4.8 dB re the refs, male composite 25.60 -> 24.60, Whisper 773 -> 796; LINE_ASP left (x1.3 / x0.8 within noise) */ /* 2026-11-05: 22 -> 11.5 with LINE_GAIN x3.4 (G1) */ /* 2026-11-04: 29 with the Strouhal bands, then 22 with the 10x wider band (LINE_NOISE_LPR 10): the fricatives' level kept (Same. s -14 dB re the vowel) */;  // (8.375 from the 30-word search was a metric exploit: the s hiss fell to -20..-26 dB under the vowels, the readers have -5..-7; "house" was "hou", 2026-11-02)  // dyn/cm^2 per unit of (Re^2 - Re_c^2) / Re_c^2 at the jet: s at the cascade's level
const float kLineAsp = param("LINE_ASP", 6.7098f) /* cost 5.3123456789012345 schwa-run2-kokoro-dB */ /* 2026-11-04: 30 x0.45 x1.25 with LINE_FRIC */;  // the same at the glottis (h between vowels is voiced and needs none)
// How the line moves between two vowel shapes: 0 interpolates the shape
// parameters (a constriction travels), 1 the area functions themselves,
// section by section in log area (one constriction fades while the next
// grows; Jyrki 2026-11-02: "perhaps the shapes should be interpolated more
// geometrically?").
const float kLineInterp = param("LINE_INTERP", 1.0f);
const float kLineShapeSigma = param("LINE_SIGMA", 0.08f);  // the formant-space width of a vowel shape (log units); 0.35 (a search result) blurred every vowel into one
// The string folds under the line's load pass ~1.7x the air of the planner's
// table (kennedy.txt: 449 cm^3/s of speech against 257 in the cascade,
// 2026-11-01); the breath planner plans with this until the folds' flow is
// calibrated.
const float kLineAirScale = param("LINE_AIR", 1.1f);  // (1.75 before the discharge coefficient)
// The vowel shape for formant values: the fitted shapes weighted by their
// distance in log-formant space (a soft lookup, so the shape moves smoothly
// along the formant path; consonants' loci land between the vowels).
// A fitted vowel's bump strength: 0 when the hump lies wholly past the lips
// (the open vowels' fits park it there: no tongue constriction), else 1.
float lineVowelStrength(float xc, float wc, float length) {
  return xc * length - std::max(wc, 0.2f) >= length ? 0.0f : 1.0f;
}
// The epilarynx: the tube from the vocal folds to where the pharynx widens (Jyrki 2026-11-10: "make the first 3 cm
// after vocal coords to have a small constriction, then an expansion where the tongue starts"). The fitted vowel
// shapes give it ~1.9 cm2 over 2 cm, about no narrowing; MRI shows ~0.3-0.5 cm2 over 2-3 cm. LINE_LARYNX_AREA scales
// its area (1: as fitted), LINE_LARYNX_LEN sets its length (cm). A narrowing at the closed (glottis) end, where every
// formant has its pressure maximum, raises the formants, F1 most (puhuja's F1 is 2-5 st under Kokoro's).
void epilarynx(TractShape& s, bool female) {
  static const float kArea = param("LINE_LARYNX_AREA", 0.82f), kLen = param("LINE_LARYNX_LEN", 2.5f);
  s.aLarynx *= kArea;
  s.lLarynx = kLen;
  // ... and above it the hypopharynx: LINE_HYPO_MUL x the larynx tube's area over LINE_HYPO_LEN cm (0: none). The
  // female voice has its own pair, LINE_HYPO_MUL_F / _LEN_F (< 0: the male value): the length is in cm while her tract
  // is shorter, and the male h4l2 (4x over 2 cm) cost her ~230 dictionary words a draw (2026-11-11).
  static const float kHypoMul = param("LINE_HYPO_MUL", 4.0f), kHypoLen = param("LINE_HYPO_LEN", 2.0f);
  static const float kHypoMulF = param("LINE_HYPO_MUL_F", 3.0f), kHypoLenF = param("LINE_HYPO_LEN_F", 1.0f);
  const float mul = female && kHypoMulF >= 0 ? kHypoMulF : kHypoMul;
  const float len = female && kHypoLenF >= 0 ? kHypoLenF : kHypoLen;
  if (mul > 0) {
    s.aHypo = s.aLarynx * mul;
    s.lHypo = len;
  }
  // ... and at the other end the space behind the teeth: LINE_FRONT_MUL x the area over LINE_FRONT_LEN cm before the
  // lip tube (1: none); the female voice LINE_FRONT_MUL_F / _LEN_F (< 0: the male value).
  static const float kFrontMul = param("LINE_FRONT_MUL", 2.25f), kFrontLen = param("LINE_FRONT_LEN", 0.5f);
  static const float kFrontMulF = param("LINE_FRONT_MUL_F", -1.0f), kFrontLenF = param("LINE_FRONT_LEN_F", -1.0f);
  const float fm = female && kFrontMulF >= 0 ? kFrontMulF : kFrontMul;
  if (fm > 1.0f) {
    s.mFront = fm;
    s.lFront = female && kFrontLenF >= 0 ? kFrontLenF : kFrontLen;
  }
}

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
  float xc = 0, lac = 0, wc = 0, sc = 0, lLip = 0, lPh = 0, lOr = 0, ll = 0, lLar = 0, len = 0;
  for (int i = 0; i < n; ++i) {
    const auto& v = kLineVowels[i];
    const float g = w[i] / sum, gs = g * lineVowelStrength(v.xc, v.wc, v.length);
    // The bump's place, area and width by the weight of the bumps that exist.
    sc += gs; xc += gs * v.xc; lac += gs * std::log(v.ac); wc += gs * v.wc;
    lLip += g * std::log(v.aLip);
    lPh += g * std::log(v.aPharynx); lOr += g * std::log(v.aOral); ll += g * v.lLip; lLar += g * std::log(v.aLarynx); len += g * v.length;
  }
  if (sc > 1e-6f) { xc /= sc; lac /= sc; wc /= sc; } else { xc = 0.5f; lac = std::log(4.0f); wc = 3.0f; }
  s.xc = xc; s.ac = std::exp(lac); s.wc = wc; s.sc = sc; s.aLip = std::exp(lLip); s.aPharynx = std::exp(lPh); s.aOral = std::exp(lOr);
  s.lLip = ll; s.aLarynx = std::exp(lLar); s.length = len;
  return s;
}
// The line's shape for a phoneme: the vowels' own from kLineVowels; w has the
// shape of u under its lip constriction (the raised tongue back that makes a
// w; with the lips alone it was a weak labial), j the shape of i. Others have
// none: the vowels on both sides define the path and the constriction is laid
// over it.
bool lineShapeOf(Ph ph, TractShape* out) {
  // w: u's shape; j: i's; l: the dark English l has the tongue back raised
  // behind its tip contact (F2 ~1000 Hz), so it takes ɔ's shape under the
  // alveolar constriction ("life" was "ife" with the neighbours' shapes,
  // Jyrki 2026-11-02); a pause: the neutral ə (the articulators rest, and
  // the next word must not start from the last one's vowel).
  // ɹ: the r-coloured ɜ's shape (its low F3) under the constriction; without
  // it the final r of "more" was the neighbours' blend and heard as an n.
  // b, p: the tract behind the lips is o's (Jyrki 2026-11-03: "the resonance
  // spaces of the transmission line should be close to what 'o' has, not 'e'
  // -- just the lips are different"): the closure is cut into o's shape and
  // the vowel's formants glide out of o's, F2 from low.
  // d (LINE_D_SHAPE, lab knob 2026-11-10, D agent): 0: no shape of its own (the neighbours' blend), 1: e's shape under
  // the tongue-tip closure (the alveolar F2 locus ~1800 Hz), 2: ɪ's, 3: ə's.
  static const int kDShape = param("LINE_D_SHAPE", 2);
  // p (LINE_P_SHAPE, lab knob 2026-11-10, P agent): 0: LINE_LABIAL_O decides; 1: o's, 2: uː's, 3: ɔ's, 4: ʊ's, 5: ə's.
  static const int kPShape = param("LINE_P_SHAPE", 1);
  const Ph look = ph == Ph::W ? Ph::U : ph == Ph::J ? Ph::I : ph == Ph::L ? Ph::AO : ph == Ph::RR ? Ph::ER
                  : ph == Ph::D && kDShape > 0 ? (kDShape == 1 ? Ph::E : kDShape == 2 ? Ph::IH : kDShape == 4 ? Ph::I : Ph::Schwa)
                  : ph == Ph::P && kPShape > 0 ? (kPShape == 1 ? Ph::O : kPShape == 2 ? Ph::U : kPShape == 3 ? Ph::AO : kPShape == 4 ? Ph::UH : Ph::Schwa)
                  : (ph == Ph::B && kLineLabialO >= 1.0f) || (ph == Ph::P && kLineLabialO >= 2.0f) ? Ph::O : ph == Ph::Stop ? Ph::Schwa : ph;
  // 2026-11-04: l keeps its own copy of the old ɔ row (fitted 464/1005/2436, a
  // close o-like shape) so the ɔ row can open toward Kokoro's ~560/1050/2400
  // without moving l (LINE_L_OWN 0 = l follows the ɔ row again).
  static const float kLOwn = param("LINE_L_OWN", 1.0f);
  if (ph == Ph::L && kLOwn > 0.5f) {
    out->xc = 0.4006f; out->ac = 0.983f; out->wc = 2.372f; out->aLip = 2.144f; out->aPharynx = 4.456f;
    out->aOral = 8.019f; out->lLip = 3.227f; out->aLarynx = 1.925f; out->length = 17.5066f;
    out->sc = lineVowelStrength(out->xc, out->wc, out->length);
    return true;
  }
  for (const auto& v : kLineVowels) {
    if (v.ph != static_cast<int>(look)) continue;
    out->xc = v.xc; out->ac = v.ac; out->wc = v.wc; out->aLip = v.aLip; out->aPharynx = v.aPharynx;
    out->aOral = v.aOral; out->lLip = v.lLip; out->aLarynx = v.aLarynx; out->length = v.length;
    if (ph == Ph::W) {
      // The w took uː's shape whole, and this uː is a fronted American one (F2
      // 1130-1230 Hz), so the w's F2 sat at 1050-1170: an l ("by war" -> "by
      // loor", Jyrki 2026-11-07). Kokoro's w: F2 520-570. Its own rounding and
      // back constriction on top of the row (1 / -1: the row's).
      static const float kWLipA = param("LINE_W_LIPA", 0.7031f), kWXc = param("LINE_W_XC", 0.35f), kWShapeAc = param("LINE_W_SHAPE_AC", 0.05f), kWLipL = param("LINE_W_LIPL", 1.0f);
      out->aLip *= kWLipA;
      out->lLip *= kWLipL;
      out->ac *= kWShapeAc;
      if (kWXc >= 0) out->xc = kWXc;
    }
    out->sc = lineVowelStrength(out->xc, out->wc, out->length);
    return true;
  }
  return false;
}
// Interpolation of two shapes (log areas, linear places and lengths).
TractShape lineShapeMix(const TractShape& a, const TractShape& b, float w) {
  TractShape s;
  auto lin = [&](float x, float y) { return x + (y - x) * w; };
  auto lg = [&](float x, float y) { return std::exp(std::log(std::max(x, 0.01f)) + (std::log(std::max(y, 0.01f)) - std::log(std::max(x, 0.01f))) * w); };
  // The bump: its strength blends; its place, area and width follow the
  // stronger bump (a mix with a bumpless shape keeps the other's place).
  s.sc = lin(a.sc, b.sc);
  const float ga = a.sc * (1 - w), gb = b.sc * w, gs = ga + gb;
  if (gs > 1e-6f) {
    const float u = gb / gs;
    s.xc = a.xc + (b.xc - a.xc) * u; s.wc = a.wc + (b.wc - a.wc) * u;
    s.ac = std::exp(std::log(std::max(a.ac, 0.01f)) + (std::log(std::max(b.ac, 0.01f)) - std::log(std::max(a.ac, 0.01f))) * u);
  } else {
    s.xc = lin(a.xc, b.xc); s.ac = lg(a.ac, b.ac); s.wc = lin(a.wc, b.wc);
  }
  s.aLip = lg(a.aLip, b.aLip);
  s.aPharynx = lg(a.aPharynx, b.aPharynx); s.aOral = lg(a.aOral, b.aOral); s.lLip = lin(a.lLip, b.lLip);
  s.aLarynx = lg(a.aLarynx, b.aLarynx); s.length = lin(a.length, b.length);
  return s;
}
// aɪ's start (see kLineAyStartMix): from a's row toward an open back row, a wider or narrower pharynx and lips, a
// longer tract; and its end (kLineAyEndMix, kLineAyEndALip) from ɪ's row.
void ayStartShape(TractShape* sh) {
  TractShape o;
  if (kLineAyStartMix > 0 && lineShapeOf(static_cast<Ph>(kLineAyStartRow), &o)) *sh = lineShapeMix(*sh, o, kLineAyStartMix);
  sh->aPharynx *= kLineAyAAph;
  sh->aLip *= kLineAyALip;
  sh->length += kLineAyALen;
  if (kLineAyALen != 0.0f) sh->sc = lineVowelStrength(sh->xc, sh->wc, sh->length);
}
void ayEndShape(TractShape* sh) {
  sh->aLip *= kLineAyEndALip;
  TractShape o;
  if (kLineAyEndMix > 0 && lineShapeOf(static_cast<Ph>(kLineAyEndRow), &o)) *sh = lineShapeMix(*sh, o, kLineAyEndMix);
}

// Where a consonant constricts the line (share of the tract's length).
float linePlaceOf(Ph ph) {
  switch (ph) {
    case Ph::P: case Ph::B: case Ph::M: case Ph::W: return 0.97f;
    case Ph::F: case Ph::VV: case Ph::V: return 0.95f;
    case Ph::TH: { static const float x = param("LINE_TH_X", 0.93f); return x; }  // the θ's place (lab 2026-11-09: the same as s's 0.90)
    case Ph::DH: { static const float x = param("LINE_DH_X", 0.90f); return x; }  // (a lab knob 2026-11-09: the ð was heard as v; a v constricts at 0.95)
    case Ph::S: { static const float x = param("LINE_S_X", 0.90f); return x; }
    case Ph::Z: { static const float x = param("LINE_Z_X", 0.90f); return x; }  // the front cavity to the teeth ~1.7 cm: the hiss near 5 kHz (a lab knob 2026-11-11)
    // n further back than t d (0.87): there Whisper heard it as l or bl
    // ("Now." -> "hello", "New." -> "blue"; 0.77-0.80 best, 2026-11-04).
    case Ph::N: { static const float x = param("LINE_N_X", 0.8f); return x; }
    case Ph::T: { static const float x = param("LINE_T_X", 0.87f); return x; }
    case Ph::D: { static const float x = param("LINE_D_X", 0.83f); return x; }  // (a lab knob 2026-11-10, D agent; t: LINE_T_X)
    case Ph::L: case Ph::R: return 0.87f;
    case Ph::SH: case Ph::ZH: { static const float x = param("LINE_SH_X", 0.78f); return x; }
    case Ph::CH: return 0.78f;
    case Ph::JH: { static const float x = param("LINE_JH_X", 0.78f); return x; }  // the ʤ closure's place (JH agent round H)
    case Ph::RR: { static const float x = param("LINE_R_X", 0.68f); return x; }  // (its hump: LINE_R_AC; 0.75 until 2026-11-04: F3 1700-1800 with the wider hump)
    case Ph::J: { static const float x = param("LINE_J_X", 0.70f); return x; }  // (0.66 / 0.74: F2 within 50 Hz, 2026-11-04)
    case Ph::K: { static const float x = param("LINE_K_X", 0.62f); return x; }
    case Ph::G: { static const float x = param("LINE_G_X", 0.72f); return x; }  // (a lab knob 2026-11-09, G agent)
    case Ph::NG: { static const float x = param("LINE_NG_X", 0.62f); return x; }  // (a lab knob 2026-11-11, NG agent 3)
    default: return -1.0f;
  }
}

// k's place by the vowel it releases into (or, final or before a consonant, the vowel before it): LINE_K_X_FRONT before
// i ɪ e ɛ æ y ø j, LINE_K_X_BACK before u ʊ o ɔ ʌ ɑ ɒ w (lab knobs 2026-11-10, K agent 2; < 0: LINE_K_X). English k is
// fronted before front vowels (its burst at the next vowel's F2/F3); one place for all gave a low burst before æ eɪ.
float lineKPlace(Ph ph, Ph vowel) {
  static const float front = param("LINE_K_X_FRONT", 0.68f), back = param("LINE_K_X_BACK", -1.0f);
  static const float tFront = param("LINE_T_X_FRONT", -1.0f), tBack = param("LINE_T_X_BACK", -1.0f);  // (t's by vowel, T agent 3; < 0: LINE_T_X)
  if (ph == Ph::T) {
    switch (vowel) {
      case Ph::I: case Ph::E: case Ph::AE: case Ph::Y: case Ph::OE: case Ph::IH: case Ph::EH: case Ph::J:
        if (tFront >= 0) return tFront;
        break;
      case Ph::O: case Ph::U: case Ph::UH: case Ph::AO: case Ph::AH: case Ph::AA: case Ph::OH: case Ph::W:
        if (tBack >= 0) return tBack;
        break;
      default: break;
    }
    return linePlaceOf(ph);
  }
  if (ph == Ph::G) {
    // g's place by what it releases into (lab knobs 2026-11-11, G agent 5; < 0: LINE_G_X): LINE_G_X_FRONT before
    // i ɪ e ɛ æ y ø j, LINE_G_X_BACK before u ʊ o ɔ ʌ ɑ ɒ w, LINE_G_X_LIQ before l ɹ (or w).
    static const float gFront = param("LINE_G_X_FRONT", -1.0f), gBack = param("LINE_G_X_BACK", -1.0f),
                       gLiq = param("LINE_G_X_LIQ", -1.0f);
    switch (vowel) {
      case Ph::I: case Ph::E: case Ph::AE: case Ph::Y: case Ph::OE: case Ph::IH: case Ph::EH: case Ph::J:
        if (gFront >= 0) return gFront;
        break;
      case Ph::O: case Ph::U: case Ph::UH: case Ph::AO: case Ph::AH: case Ph::AA: case Ph::OH: case Ph::W:
        if (gBack >= 0) return gBack;
        break;
      case Ph::L: case Ph::RR:
        if (gLiq >= 0) return gLiq;
        break;
      default: break;
    }
    return linePlaceOf(ph);
  }
  if (ph != Ph::K) return linePlaceOf(ph);
  // (K agent 9 round J, lab; < 0: LINE_K_X_FRONT) before iː its own place: initial kiː 69 % wrong, heard p ("Key").
  static const float kIy = param("LINE_K_X_IY", -1.0f);
  if (vowel == Ph::I && kIy >= 0) return kIy;
  // (K-INITIAL agent 13 round L, lab; < 0: LINE_K_X) before l ɹ its own place (kl kɹ heard as p: "Cried" -> pride).
  static const float kLiqX = param("LINE_K_X_LIQ", -1.0f);
  if ((vowel == Ph::L || vowel == Ph::RR) && kLiqX >= 0) return kLiqX;
  switch (vowel) {
    case Ph::I: case Ph::E: case Ph::AE: case Ph::Y: case Ph::OE: case Ph::IH: case Ph::EH: case Ph::J:
      if (front >= 0) return front;
      break;
    case Ph::O: case Ph::U: case Ph::UH: case Ph::AO: case Ph::AH: case Ph::AA: case Ph::OH: case Ph::W:
      if (back >= 0) return back;
      break;
    default: break;
  }
  return linePlaceOf(ph);
}

// ŋ's place (lab knobs 2026-11-11, NG agent 3). The dictionary heard ŋ as n 314 of 571 errors: Kokoro's vowel before ŋ
// ends with F2 and F3 converging (the velar pinch, "sing": F2 2320 / F3 2450 at its end), puhuja's kept them 1100 Hz
// apart (2350 / 3450). LINE_NG_X_FRONT: the place after i ɪ e ɛ æ y ø j (< 0: LINE_NG_X); LINE_NG_X_KG: the share of
// the way to a following k's or g's own place (0: none, 1: homorganic; the k after ŋ sits at LINE_K_X_FRONT after a
// front vowel, the g at LINE_G_X, while ŋ was at 0.62).
float lineNgPlace(Ph prev, const Ph* next) {
  static const float front = param("LINE_NG_X_FRONT", -1.0f), kg = param("LINE_NG_X_KG", 0.0f);
  float x = linePlaceOf(Ph::NG);
  const bool frontV = prev == Ph::I || prev == Ph::E || prev == Ph::AE || prev == Ph::Y || prev == Ph::OE ||
                      prev == Ph::IH || prev == Ph::EH || prev == Ph::J;
  if (front >= 0 && frontV) x = front;
  if (kg > 0 && next && (*next == Ph::K || *next == Ph::G))
    x += kg * ((*next == Ph::K ? lineKPlace(Ph::K, prev) : linePlaceOf(Ph::G)) - x);
  return x;
}

// --- Aerodynamics ---------------------------------------------------------
// Areas in cm^2. Pressures are relative to the lung pressure (8 cmH2O).
constexpr float kAcOpen = 2.0f;      // open vocal tract (vowels)
const float kAgVoicedFric = param("LINE_AG_VFRIC", 0.25f);  // glottal area in z ʒ v ð (kAgVoice 0.045 for the vowels)
// z's own (LINE_AG_Z; default = LINE_AG_VFRIC). The z of "is", "was" had no hiss: with 0.08 the folds opened to 0.03 cm^2 at
// most, the jet's Re 800-1600 under the 1800 threshold ("is" heard as "in", Jyrki 2026-11-06); after an h ("his") they
// stayed at 0.06-0.2 and the z hissed. v keeps 0.08: at 0.16 its voicing fell 16 dB.
const float kAgZ = param("LINE_AG_Z", 0.202f);  /* 2026-11-07: 0.16 -> 0.202 with EN_CLUSTER_Z_LEN 1.35 (the z of "holds", Lag70) */  // Jyrki 2026-11-06: "worldis-zA.wav sounded good" (with LINE_Z_AC 1, LINE_RE_CRIT 1400: Whisper 801 -> 853)
const float kAgFricOpen = param("LINE_AG_FRIC", kAgOpen);  // ... and in s ʃ f θ (kAgOpen 0.3; a narrower glottis adds its own tract-coloured hiss under the fricative, Jyrki's blend exploration 2026-11-05)
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
// ... with the line tract there is no output fade, so the glottis closes at
// a speaker's pace: with 280 ms the voice went on for 150 ms of every comma
// pause at -7 dB (Jyrki 2026-11-02: "would also be nice if the sound stopped
// in between of the words").
const float kLineStopGlottalMs = param("LINE_STOP_MS", 50.0f);
constexpr float kPreposeMs = 40;     // after a pause, articulators move this early
// Damped formant motion: time constant per formant, by the articulator that
// mostly moves it. F1 follows the jaw (the largest mass: slowest), F2 the
// tongue body, F3 the tongue tip and the lips (light: fastest).
constexpr float kFormantTauMs[3] = {16, 12, 9};
// With a planned path (FormantMotion::Planned) the follower only smooths the
// plan's grid: a short constant.
constexpr float kPlanTauMs = 3;
// Tract shape planner (-m planned with the line; Synth::planShapes). Per
// phrase and per shape channel, over 5 ms frames, it minimises
//   demands:     w (|x - target| - tol)+^2 in each sound's window, only in
//                the channels that sound needs (a w: the lips, a j: the tongue)
//   effort:      vel (dx)^2 + acc (d2x)^2   (a slow, smooth articulator)
//   speed limit: mu (|dx| - vmax)+^2
// by reweighted least squares (the dead zones and the speed limit are
// re-decided each round). Free channels coast on the neighbours' path.
const float kShapeW = param("SPLAN_W", 1.0f);
const float kShapeVel = param("SPLAN_VEL", 10.98f) /* cost 1814 whisper-run1-phon_err */;  // (0.3, acc 30, vmax 8: transitions 3x slower; Jyrki 2026-11-03 took the 3x faster ones)
const float kShapeAcc = param("SPLAN_ACC", 1.052f) /* cost 1814 whisper-run1-phon_err */;
const float kShapeMu = param("SPLAN_MU", 9.67f) /* cost 1814 whisper-run1-phon_err */;
const float kShapeVmax = param("SPLAN_VMAX", 10.01f) /* cost 1058 whisper-sens1-combo-phase1-err */;  // log units (or channel ranges) per second
const float kShapeTol = param("SPLAN_TOL", 0.0913f) /* cost 1814 whisper-run1-phon_err */;   // dead zone (log units, or shares of the channel range)
const float kShapeRel = param("SPLAN_REL", 0.001f);  // a faint pull to the phrase mean where nothing is demanded
const float kShapePauseW = param("SPLAN_PAUSEW", 0.3558f) /* cost 1814 whisper-run1-phon_err */;      // a pause's neutral shape
const float kShapeBorrowedW = param("SPLAN_BORROWEDW", 1.641f) /* cost 1814 whisper-run1-phon_err */;  // a consonant in a vowel's shape (f, b, p)
const float kShapeOpenA = param("SPLAN_OPEN_A", 2.5f);      // cm^2: a vowel constriction this wide has a free place
const float kShapeNarrowA = param("SPLAN_NARROW_A", 0.5f);  // cm^2: this narrow, the place counts in full
const float kShapeOpenXcW = param("SPLAN_OPEN_XCW", 0.01f);  // the place's least weight in an open vowel
const float kShapeGlideStart = param("SPLAN_GLIDE_START", 0.3f);  // a chain's first vowel is held until this share of it
const float kShapeGlideEnd = param("SPLAN_GLIDE_END", 0.8f);      // ... the second from this share of it
const float kShapeGlideW = param("SPLAN_GLIDE_W", 1.0f);  // ... their weight (3-10 reached the ends better but brought the F2 rise back: the straight shape path is not straight in formants)
const int kShapeRounds = static_cast<int>(param("SPLAN_ROUNDS", 10));
constexpr float kShapeHopMs = 5;
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
const float kLungRangeInv = 1.0f / (kLungMax - kLungRest);
constexpr float kCommaBreathBelow = 900; // breathe at a comma below this (1600: three breaths in a row at the commas of "Let every nation know, whether it wishes us well or ill, that we shall pay any price,", 360-435 ms each where the recordings' commas are 170 ms; 1000: breaths fell at phrase splits)
constexpr float kInhaleRate = 2500;      // cm^3/s while breathing in
const float kInhaleMinMs = param("kInhaleMinMs", 180.0f), kInhaleMaxMs = param("kInhaleMaxMs", 600.0f);
const float kCatchInhaleMaxMs = param("kCatchInhaleMaxMs", 300.0f);  // within a sentence (at a comma or a seam): a quick catch breath
// The breath's length by its place: a catch breath inside a sentence shorter,
// between sentences longer, after an empty line (a paragraph) longer still.
const float kCatchBreathScale = param("kCatchBreathScale", 0.85f), kSentenceBreathScale = param("kSentenceBreathScale", 1.10f), kParagraphBreathScale = param("kParagraphBreathScale", 1.30f);
constexpr int kBreathHoldStops = 0;  // silence between the inhale and the first sound
const float kInhaleNoise = param("kInhaleNoise", 0.035f);  // breathing-in noise (0.09: 23 dB under the vowels, a reader's breath is 35 under; heard as artificial)
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
const float kLungPressureSpent = param("kLungPressureSpent", 0.85f);  // relative pressure at kLungRest and below
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
// (a knob since 2026-11-08: Jyrki, "heritage is missing its end, it sounds like 'heriti'": the final ʤ's
// frication ran inside this easing and came out 11-14 dB under a non-final one at the same slit, glottis
// and gain; the hiss falls faster than the pressure past the Reynolds threshold)
const float kEndPressure = param("LINE_END_PRESS", 0.95f);  // breath pressure at a phrase end, relative (0.75)
const float kEndEaseMs = param("LINE_END_EASE_MS", 220.0f);     // eased off over this long before the pause (150: endings faded instead of releasing)
const float kPressResumeMs = param("LINE_PRESS_RESUME_MS", 30.0f);  // the breath comes back over this long after a pause (a lab knob 2026-11-09: the pause-initial ð's voicing)
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
const Voice kMaleVoice = {param("kMaleF0", 95.5) /* cost 1814 whisper-run1-phon_err */, {param("kMaleFormant", 0.9625), param("kMaleFormant", 0.9625), param("kMaleFormant", 0.9625)}, 1.0f, 1.0f, 1.7f, 1.0f, param("kMaleGain", 4.163f) /* cost 0.0322859615827058 n13-speaker-w30 */, -6, 0.3f,
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
const Voice kFemaleVoice = {param("kFemaleF0", 134) /* cost 0.0298241432933307 n14-stringline-w30-c */, {1.12f, 1.15f, 1.06f}, 1.15f, 0.92f, 1.8f, 1.1f, param("kFemaleGain", 3.704f) /* cost 0.0322859615827058 n13-speaker-w30 */, 7, 1.0f,
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
const float kAffricateFricationPerClosure = param("kAffricateFricationPerClosure", 1.4f);  // (1.4, 2026-11-04: at 0.84 the line's hiss after the closure was ~30 ms, Kokoro's 40-80; Whisper on 24 ʤ ʧ words + 6 guards, m+f: words 7 -> 17, initial ʧ 0 -> 5 of 16, initial ʤ 1 -> 4 of 16, guards unchanged)  // (0.8389: optimizer, stops-w1k-us5-mf-3)  // (1.9: onset and medial ʤ ʧ 137-182 ms in kennedy.txt against 70-150, median ~115, in six references; "generosity" 163 vs 100-150, "torch" 137 vs 70-80, 2026-10-30)  // (2.2: ʧ ʤ 145-168 ms against 90-120 in am_onyx and ElevenLabs; a quarter of the way back, 2026-10-30)  // (1.8: a final "rich" came back from Whisper as "right"; a recording's final ʧ has 230 ms of hiss)  // (1.3: ʧ 124 ms, the readers 120-190, a final one in a recording 230)  // ... and ʤ's 10 % longer (by ear)

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
// 1: a voiced fricative (v ð z ʒ), 2: a voiced plosive or affricate (b d g ʤ), 0: neither.
int voicedObstruent(Ph ph) {
  switch (ph) {
    case Ph::VV: case Ph::DH: case Ph::Z: case Ph::ZH: return 1;
    case Ph::B: case Ph::D: case Ph::G: case Ph::JH: return 2;
    default: return 0;
  }
}

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
const float kAspiratedAdductMs = param("kAspiratedAdductMs", 100.0) /* 14.8 from a stops search gave VOT 7-18 ms: voiceless stops heard voiced (Whisper: time -> die, price -> lie, 2026-11-02); English stressed 50-80 */, kAspiratedUnstressedMs = param("kAspiratedUnstressedMs", 50.0);  // (70: VOT still 106-120; the folds take ~40 ms to start after the glottis closes)
const float kStrongAspirationLength = param("kStrongAspirationLength", 0.6184f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */, kStrongAspirationGain = param("kStrongAspirationGain", 2.594f) /* cost 0.0394612677442557 stops-w1k-us5-mf-3 */;  // word-initial p t k of a primary-stressed syllable
// Finnish plosives are unaspirated: the glottis closes again quickly.
PlosiveInfo plosiveInfo(Ph ph) {
  switch (ph) {
    case Ph::P: return {80, 100, 6, 10, 0.082f, false, 0.105f, 8};  // +10 % (by ear)
    case Ph::T: return {70, 110, 12, 25, 0.122f, false};
    case Ph::K: return {70, 100, 25, 25, 0.0684f, false};
    case Ph::B: return {60, 100, 10, 0, 0.0157f, true};
    case Ph::D: {  // (lab knobs 2026-11-10, D agent: the closure, the opening time and the pop of the d)
      static const float kDCloseMs = param("LINE_D_CLOSURE_MS", 70.0f), kDRelMs = param("LINE_D_REL_MS", 10.0f), kDPop = param("LINE_D_POP", 0.5f);
      return {kDCloseMs, 110, kDRelMs, 0, 0.123f, true, kDPop};
    }
    case Ph::G: {  // 15% of the way toward k: firmer closure, crisper release
      // (lab knobs 2026-11-09, G agent: the closure, the opening time and the pop of the g)
      static const float kGCloseMs = param("LINE_G_CLOSURE_MS", 80.0f), kGRelMs = param("LINE_G_REL_MS", 21.0f),
                         kGPop = param("LINE_G_POP", 1.34f);
      return {kGCloseMs, 120, kGRelMs, 0, 0.089f, true, kGPop, 0,
              kAgVoice + kLineGClosureAgFrac * (kAgOpen - kAgVoice)};
    }
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
    case Cls::Vowel: {
      // (round-H ʊ agent, lab; < 0: as every vowel) the ʊ's own glottis: a breathier lax vowel.
      static const float agUh = param("LINE_AG_UH", -1.0f);
      return {kAcOpen, ph == Ph::UH && agUh >= 0 ? agUh : kAgVoice, 0};
    }
    case Cls::Nasal: return {0, kAgVoice, kAnNasal};  // mouth closed, nose open
    case Cls::Liquid: return {0.35f, kAgVoice, 0};
    case Cls::Trill: return {0.3f, kAgVoice, 0};
    case Cls::Glide: {
      static const float jAc = param("LINE_J_AC", 0.3f);  // j (0.2: 3 dB weaker, F2 +50 Hz, 2026-11-04)
      // w (2026-11-07, Jyrki: "tempered by war" sounds like "by loor"): with
      // 0.25 cm^2 the w's F2 stayed at 1100-1300 Hz after "by", an l; Kokoro's
      // w is at 520-570 Hz.
      static const float wAc = param("LINE_W_AC", 0.25f);
      // (W agent round I, lab; 1: as before) the w's glottal area x this (> 1 breathier, < 1 pressed).
      static const float wAg = param("LINE_W_AG", 1.0f);
      return {ph == Ph::J ? jAc : wAc, ph == Ph::W && wAg != 1.0f ? kAgVoice * wAg : kAgVoice, 0};
    }
    case Cls::Fricative:
      switch (ph) {
        case Ph::S: return {0.05f, kAgFricOpen, 0};
        case Ph::SH: return {0.08f, kAgFricOpen, 0};
        // The voiced fricatives phonate with the glottis a little apart (more
        // flow through the groove: z had no hiss at all on the line, centroid
        // 330 Hz, and Whisper heard "is", "his" as "dead", 2026-11-02).
        case Ph::ZH: { static const float acZh = param("LINE_AC_ZH", 0.08f), agZh = param("LINE_AG_ZH", 0.17f); return {acZh, agZh >= 0 ? agZh : kAgVoicedFric, 0}; }  // ʒ's own glottis (< 0: LINE_AG_VFRIC; lab 2026-11-09)  // voiced: less pressure drop (a knob since 2026-11-07: the ʤ's frication)
        case Ph::Z: return {0.05f, kAgZ, 0};
        case Ph::DH: { static const float agDh = param("LINE_AG_DH", 0.08f); return {0.14f, agDh >= 0 ? agDh : kAgVoicedFric, 0}; }  // ð's own glottis (< 0: LINE_AG_VFRIC; at 0.25 the pause-initial ð lost its voice, 2026-11-09)
        case Ph::VV: { static const float agV = param("LINE_AG_V", 0.17f); return {0.08f, agV >= 0 ? agV : kAgVoicedFric, 0}; }  // v's own glottis (< 0: LINE_AG_VFRIC; lab 2026-11-09)
        case Ph::TH: { static const float agTh = param("LINE_AG_TH", -1.0f); return {0.1f, agTh >= 0 ? agTh : kAgFricOpen, 0}; }  // θ's own glottis (< 0: LINE_AG_FRIC; lab 2026-11-09)
        default: return {0.1f, kAgFricOpen, 0};        // F
      }
    case Cls::H: return {1.5f, kAgOpen, 0};
    case Cls::Stop: return {kAcOpen, 0, 0};  // glottis closed: no airflow
    case Cls::Plosive: return {0, kAgOpen, 0};  // the closure
  }
  return {kAcOpen, kAgVoice, 0};
}

// The line's slit for a fricative: the cascade's table area scaled. The
// slits are wider than the cascade's table: a 0.05 cm^2 s passes at most
// 175 cm^3/s at 7 cmH2O where a spoken s runs 300-500, and its hiss sat 18 dB
// under the vowels (Kokoro 5-7), 2026-11-02; the dentals' 0.1-0.14 cm^2 slits
// passed so much air that θ and ð came out louder than the vowels at the
// s-matching noise level (kLineDentalAc, kLineFAc).
float lineSlit(Ph ph, float ac) {
  if (info(ph).cls == Cls::Fricative) ac *= kLineFricAc;
  if (ph == Ph::TH || ph == Ph::DH) ac *= kLineDentalAc;
  // ... and ð's slit wider than θ's (x this; 1: the same). At 0.044 cm^2 the mouth pressure behind the ð rose so far that
  // the folds did not start for 40 ms after a pause ("The": voiceless for its first 40 ms, heard as v t d; Kokoro's ð is
  // voiced and dark from its first ms, nearly an approximant), 2026-11-09.
  { static const float kLineDhAc = param("LINE_DH_AC", 1.0f); if (ph == Ph::DH) ac *= kLineDhAc; }
  if (ph == Ph::F) ac *= kLineFAc;
  // ʃ ʒ ʧ ʤ: how much the groove opens (x the table's 0.08 cm^2; Jyrki 2026-11-08: "how wide the opening is vs
  // how much the opening is -- if these matter"; its width along the tract is LINE_POST_HW, the jet follows its height)
  if (ph == Ph::SH || ph == Ph::ZH || ph == Ph::CH || ph == Ph::JH) { static const float kLinePostAc = param("LINE_POST_AC", 1.0f); ac *= kLinePostAc; }
  // z: with s's 0.05 cm^2 slit the mouth pressure rose to near the lungs'
  // and the folds stopped: the z was a voiceless hiss (0-400 Hz -50 dB re the
  // word's peak, Kokoro's z keeps a voice bar), heard as s f θ, medially t.
  // A wider slit keeps the voicing (x3.2: -15..-20 dB) at a weaker hiss
  // (2026-11-04).
  if (ph == Ph::Z) { static const float zAc = param("LINE_Z_AC", 1.0f);  /* 3.2 2026-11-05..06 (the "his" choice); back to 1 with the z's own abduction LINE_AG_Z (zA, 2026-11-06) */  /* 1.0 until 2026-11-05: Jyrki wanted "hiz, soft without edges" and chose x3.2 (hih-Z1) */ ac *= zAc; }
  // v: the same lips-on-teeth slit question as z (2026-11-04, LINE_V_AC).
  if (ph == Ph::VV) { static const float vAc = param("LINE_V_AC", 2.0f); ac *= vAc; }
  // ɹ: the oral hump (0.25 cm^2 at 0.75) on top of the onset shape's own
  // pharyngeal narrowing (the ɒ slot, 0.30 cm^2): two narrow constrictions
  // left F2 and F3 damped (LPC bandwidths 250 / ~1000 Hz, Kokoro 134 / 306),
  // 1.3-2 kHz 15 dB and the r 3-4 dB under Kokoro's. 0.6 cm^2 at 0.68
  // (LINE_R_X): F2/F3 820-990/1460-1500 bw 160/205, level -11 -> -8 dB re
  // the vowel (Kokoro -7..-8); Whisper initial ɹ 23 -> 37 of 40, medial
  // 6 -> 11 of 16 (2026-11-04).
  if (ph == Ph::RR) { static const float rAc = param("LINE_R_AC", 0.6f); ac = rAc; }
  if (ph == Ph::L) { static const float lAc = param("LINE_L_AC", 1.7f); ac *= lAc; }  // the lateral opening (x the table's 0.35 cm^2)
  return ac;
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
  lineEqShelf_.highShelf(kLineEqShelfHz, kLineEqShelfDb, fs_);
  lineEqPeak_.peaking(kLineEqPeakHz, kLineEqPeakBw, kLineEqPeakDb, fs_);
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
  female_ = preset == VoicePreset::Female;
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
        prosodyOverride(&p);
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
    prosodyOverride(&p);
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

float Synth::drainScale() const {
  static const float kDrainScale0 = param("SYNTH_DRAIN_SCALE", 0.5f);
  return drainScale_ < 0 ? kDrainScale0 : drainScale_;
}

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
  // Debug: PUHUJA_BREATH_TRACE=file writes every breath taken (timeline s, ms, kind, the planned air before it).
  static FILE* btrace = std::getenv("PUHUJA_BREATH_TRACE") ? std::fopen(std::getenv("PUHUJA_BREATH_TRACE"), "w") : nullptr;
  if (btrace) std::fprintf(btrace, "%.4f\t%.0f\t%d\t%.0f\n", timeline_ / fs_, ms, paragraph ? 2 : sentence ? 1 : 0, planVolume_), std::fflush(btrace);
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
    prosodyOverride(&pending_);
    emit(pending_, nullptr);
    hasPending_ = false;
  }
  planShapes(timeline_);
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

// The fields an event sets (one bit per thing it changes; tick() applies each
// on its own), for addEvent's per-field clamp.
static uint64_t eventFields(const Event& e) {
  uint64_t m = 0;
  int b = 0;
  auto bit = [&](bool on) { m |= static_cast<uint64_t>(on) << b++; };
  bit(e.ac >= 0); bit(e.ag >= 0); bit(e.an >= 0); bit(e.setNoise); bit(e.pop >= 0); bit(e.press >= 0);
  bit(e.aspGain >= 0); bit(e.lineAspMul >= 0); bit(e.lineGlotFc >= 0); bit(e.lineHBreath >= 0);
  bit(e.linePuffLpr >= 0); bit(e.lineFricFade >= 0); bit(e.lineFricReBoth >= 0); bit(e.lineFricReMul >= 0);
  bit(e.lineFricMul >= 0); bit(e.lineLevel > 0); bit(e.lineSoft >= 0); bit(e.lineExtra >= 0);
  bit(e.lineExpand >= 0); bit(e.lineHw >= 0); bit(e.lineLip >= 0); bit(e.lineAspHp >= 0); bit(e.lineFricLpr >= 0);
  bit(e.lineJetD >= 0); bit(e.lineX >= 0); bit(e.aspBurst); bit(e.creak >= 0); bit(e.expand >= 0);
  bit(e.inhaleMs > 0); bit(e.speech >= 0); bit(e.lineFricReSmooth >= 0); bit(e.lineHumpNorm > -1.5f);
  return m;
}

void Synth::addEvent(const Event& e, int srcLine) {
  // Movements must start in order: a later phoneme's movement starting
  // before an earlier one's would be overridden by it.
  // SYNTH_EVENT_CLAMP 0: after the last queued event (as before). 1: only after the last queued event that sets one
  // of the same fields, inserted in time order. The queue also holds events dated later with addEventAt (a stop's
  // post-release back events, the velum, the f ramp, approaches), and the whole-queue clamp dragged unrelated events
  // behind them (review cpp.md B1, Kennedy paragraph: 128 events late; 96 the vowel's level after a stop, median
  // 44 ms, max 150 ms; the 40 ms early pre-positioning after a pause 15-42 ms late, "Judge"'s ʤ 2 ms after its start).
  static const bool kPerField = param("SYNTH_EVENT_CLAMP", 1.0f) > 0;
  static const bool kTrace = std::getenv("PUHUJA_CLAMP_TRACE") != nullptr;  // clamps over 1 ms, with the caller's line
  Event ev = e;
  if (events_.empty()) {
    events_.push_back(ev);
    return;
  }
  if (!kPerField) {
    ev.t = std::max(ev.t, events_.back().t);
  } else {
    const uint64_t m = eventFields(ev);
    for (auto it = events_.rbegin(); it != events_.rend(); ++it)
      if (eventFields(*it) & m) {
        ev.t = std::max(ev.t, it->t);
        break;
      }
  }
  if (kTrace && ev.t - e.t > 0.001 * fs_)
    std::fprintf(stderr, "clamp t %.3f +%.1f ms line %d fields %llx\n", (e.t - (outputStart_ < 0 ? 0 : outputStart_)) / fs_,
                 1000.0 * (ev.t - e.t) / fs_, srcLine, static_cast<unsigned long long>(eventFields(ev)));
  if (kPerField) addEventAt(ev); else events_.push_back(ev);
}

// The diaphragm's push before an obstruent (LINE_PULSE): the lung pressure
// rises by amount to peak at peak (not starting before lowBound) and falls
// back. Only on top of the normal speech pressure: a phrase's first sound
// rides on the rise back to it (the push starts with that rise, when it
// starts later), and nothing else planned near the peak is disturbed (a
// phrase end keeps its own fall).
void Synth::addPressPulse(double peak, double lowBound, float amount, float riseMs, float fallMs) {
  if (amount <= 0) return;
  const double ms = 0.001 * fs_;
  if (riseMs < 0) riseMs = kLinePulseRiseMs;
  if (fallMs < 0) fallMs = kLinePulseFallMs;
  double rise = std::max(peak - riseMs * ms, lowBound);
  const double end = peak + fallMs * ms;
  float base = pressTo_;  // the target in force at the rise
  for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
    if (it->press < 0 || it->t >= end) continue;
    base = it->press;
    float before = pressTo_;  // ... and the one before it
    for (auto jt = it + 1; jt != events_.rend(); ++jt)
      if (jt->press >= 0) {
        before = jt->press;
        break;
      }
    const bool resume = before < base - 1e-3f;  // the rise after a pause
    if (it->t >= peak - 2 * ms && !resume) return;  // another movement in the way
    if (it->t >= rise) {
      // A movement to speech pressure under way: the push starts from it.
      // After a pause it peaks where that rise ends; pushed within 5 ms
      // from the pause's zero, the f of "Food." began with a click as loud
      // as the vowel.
      rise = it->t + 1;
      if (resume) peak = std::max(peak, it->t + it->pressMs * ms);
    }
    break;
  }
  for (const Event& q : events_)  // (the peak may have moved later)
    if (q.press >= 0 && q.t >= rise && q.t < peak + fallMs * ms) return;
  if (std::fabs(base - 1.0f) > 1e-3f || peak - rise < 2 * ms) return;
  for (int k = 0; k < 2; ++k) {
    Event pe{k == 0 ? rise : peak};
    pe.press = k == 0 ? 1.0f + amount : 1.0f;
    pe.pressMs = static_cast<float>(k == 0 ? (peak - rise) / ms : fallMs);
    pe.ac = pe.ag = pe.an = -1;
    pe.speech = -1;
    pe.aspGain = -1;
    pe.expand = -1;
    pe.inhaleMs = 0;
    pe.pop = -1;
    addEventAt(pe);
  }
  pulseEnd_ = peak + fallMs * ms;
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

// Kokoro mode (prosody.h): the guide's labels layer, one call per label the
// token would add (addLabel's rule: pauses merge). The guide answers with the
// planned length and a vowel's two pitches; anything it does not have keeps
// puhuja's own.
void Synth::prosodyOverride(Token* tok) {
  if (!prosodyGuide_) return;
  if (tok->ph == Ph::Stop) {
    if (!prosodyGuide_->hasWords()) return;
    prosodyCloseWord();
    // The rhythm layer's pauses: the run of pause tokens (the reader's and the
    // synth's own inhale) is cut to the reference's gap after the last word
    // (its lead-in before the first, at least 80 ms for the onset): the tokens
    // past it shrink to a millisecond. Puhuja's pause model keeps its events.
    if (prosodyPauseT0_ < 0) prosodyPauseT0_ = timeline_;
    const float gap = prosodyPauseMs(prosodyLastWord_);
    if (gap >= 0) {
      const double limit = std::max(gap, prosodyLastWord_ <= 0 ? 80.0f : 30.0f) * 0.001 * fs_;
      if (timeline_ - prosodyPauseT0_ >= limit) tok->shape.ms = 1.0f / std::max(voice_.rate, 1e-3f);
    }
    return;
  }
  prosodyPauseT0_ = -1;
  std::string s = symbolOf(tok->ph);
  if (tok->isLong && tok->ph != Ph::Stop) s += "ː";
  float ms, pitch, pitchEnd;
  const bool labelled = prosodyGuide_->hasLabels() && prosodyGuide_->nextLabel(s, &ms, &pitch, &pitchEnd);
  if (labelled) {
    if (tok->ph != Ph::Stop && ms > 0) {
      tok->shape.ms = ms / std::max(voice_.rate, 1e-3f);  // (emit multiplies by the rate)
      tok->doubled = false;
    }
    if (isVowel(tok->ph)) {
      if (pitch != ProsodyProfile::kNone) tok->pitch = pitch;
      if (pitchEnd != ProsodyProfile::kNone) tok->pitchEnd = pitchEnd;
    }
  }
  // The words layer: the word's tokens scaled to the reference word's length
  // (the labels layer, where it matched, has set the lengths already), and the
  // word's span registered for the tracks' time warp.
  if (!prosodyGuide_->hasWords()) return;
  // The level layer replaces puhuja's own prosodic level (accent, function
  // word, arc: Shape.prosody), so the reference's word levels are not added on top of it.
  if (prosodyGuide_->has(ProsodyGuide::kLevel) && tok->shape.prosody > 0 && tok->shape.prosody != 1.0f) {
    tok->shape.level /= tok->shape.prosody;
    tok->shape.prosody = 1.0f;
  }
  if (tok->wordStart) {
    prosodyCloseWord();
    prosodyLastWord_ = tok->word;
    // The English reader's planned total is the sum before its later
    // lengthenings (vowels, stressed onsets, the f, the h): the final lengths
    // run kPlannedToActual over it (measured 1.03-1.31 on kennedy's words).
    static const float kPlannedToActual = param("PROSODY_PLANNED_TO_ACTUAL", 1.2f);
    float sc;
    const int iw = tok->word - 1;  // the synth counts words from 1
    prosodyWordScale_ = !labelled && tok->shape.wordMs > 0 && prosodyGuide_->wordScale(iw, tok->shape.wordMs * kPlannedToActual * voice_.rate, &sc) ? sc : 1.0f;
    prosodyWord_ = iw;
    prosodyWordT0_ = timeline_;
    static const bool trace = std::getenv("PUHUJA_PROSODY_TRACE") != nullptr;
    if (trace) std::fprintf(stderr, "prosody word %d: wordMs %.0f scale %.2f at timeline %.3f s\n", iw, tok->shape.wordMs, prosodyWordScale_, timeline_ / fs_);
  }
  if (!labelled && prosodyWordScale_ != 1.0f) {
    if (tok->shape.ms > 0) tok->shape.ms *= prosodyWordScale_; else tok->shape.length *= prosodyWordScale_;
  }
}

float Synth::prosodyPauseMs(int sourceWord) const {
  if (!prosodyGuide_ || !prosodyGuide_->hasWords()) return -1.0f;
  if (sourceWord <= 0) return prosodyGuide_->leadInMs();
  float ms;
  return prosodyGuide_->pauseAfterMs(sourceWord - 1, &ms) ? ms : -1.0f;
}

// The open word's actual span on the timeline, registered for the tracks' warp.
void Synth::prosodyCloseWord() {
  if (prosodyWord_ < 0 || !prosodyGuide_) return;
  prosodyGuide_->wordStart(prosodyWord_, prosodyWordT0_ / fs_, (timeline_ - prosodyWordT0_) / fs_ * 1000.0);
  prosodyWord_ = -1;
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

std::vector<Synth::ArticulationFrame> Synth::articulation() const {
  std::vector<ArticulationFrame> out;
  const double start = outputStart_ < 0 ? 0 : outputStart_;
  for (const auto& f : art_) {
    if (f.t < start) continue;
    out.push_back(f);
    out.back().t = (f.t - start) / fs_;
  }
  return out;
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


namespace {

// oʊ lab knobs (OW agent 2026-11-11; the defaults reproduce the previous sound). Dictionary rank 9: oʊ 1093 errors
// of 5208 (21 %, Kokoro 4.4 %); word-final 54 % wrong, heard as "-al" (Eco -> equal, Hugo -> neutral, Rio -> real).
// LPC on single words: puhuja's oʊ holds F2 1170-1290 (male) / 1400-1500 (female) to its end with F1 falling to
// 320-340 and F3 to 1900-2150; Kokoro's F2 falls steadily 1250 -> 830-900 (female ~1000), F1 500 -> 400-440, F3
// level 2400-2500. The ʊ end takes u's row, a fronted American u (fitted 303/1205/2049).
// The o start: the ə mix (LINE_OW_START_AH, 0.5 as before), then toward ɔ (LINE_OW_START_AO), lip area / length x
// LINE_OW_START_LIP / _LIPL.
const float kLineOwStartAh = param("LINE_OW_START_AH", 0.5f);
void owStartShape(TractShape* sh) {
  static const float kAo = param("LINE_OW_START_AO", 0.0f), kLip = param("LINE_OW_START_LIP", 1.0f),
                     kLipL = param("LINE_OW_START_LIPL", 1.0f);
  TractShape o;
  if (kAo > 0 && lineShapeOf(Ph::AO, &o)) *sh = lineShapeMix(*sh, o, kAo);
  if (kLip != 1.0f) sh->aLip *= kLip;
  if (kLipL != 1.0f) sh->lLip *= kLipL;
}
// The ʊ end (u's row): mixed toward ɔ (LINE_OW_END_AO: 488/903/2760, back with a high F3) and ʊ's own row
// (LINE_OW_END_UH, 462/1039/2300); lip area / length x LINE_OW_END_LIP / _LIPL; the constriction's place set to
// LINE_OW_END_XC (<0: the row's; the w's back tongue is 0.25), its area x LINE_OW_END_AC.
void owEndShape(TractShape* sh) {
  static const float kAo = param("LINE_OW_END_AO", 0.5f), kUh = param("LINE_OW_END_UH", 0.0f),
                     kLip = param("LINE_OW_END_LIP", 0.5f), kLipL = param("LINE_OW_END_LIPL", 1.0f),
                     kXc = param("LINE_OW_END_XC", -1.0f), kAc = param("LINE_OW_END_AC", 1.0f);
  TractShape o;
  if (kUh > 0 && lineShapeOf(Ph::UH, &o)) *sh = lineShapeMix(*sh, o, kUh);
  if (kAo > 0 && lineShapeOf(Ph::AO, &o)) *sh = lineShapeMix(*sh, o, kAo);
  if (kLip != 1.0f) sh->aLip *= kLip;
  if (kLipL != 1.0f) sh->lLip *= kLipL;
  if (kAc != 1.0f) sh->ac *= kAc;
  if (kXc >= 0) { sh->xc = kXc; sh->sc = lineVowelStrength(sh->xc, sh->wc, sh->length); }
}

// eɪ lab knobs (EY agent 2026-11-10; the defaults reproduce the previous sound). Whisper heard eɪ as iː 733 times in
// 6980 (dict rank 8); against ten Kokoro voices the e start sits low in F1 (male 410-470 vs 480-560 Hz, female
// 430-610 vs 500-850) and back in F2.
// The e start: mixed toward ɛ (LINE_EY_START_EH) or æ (LINE_EY_START_AE), its place moved by LINE_EY_START_DX, its
// opening, pharynx and lip area x LINE_EY_START_AC / _APH / _LIP.
void eyStartShape(TractShape* sh, bool female, unsigned ctx) {
  static const float kEyStartEh = param("LINE_EY_START_EH", 0.0f), kEyStartAe = param("LINE_EY_START_AE", 0.0f),
                     kEyStartDx = param("LINE_EY_START_DX", 0.0f), kEyStartAc = param("LINE_EY_START_AC", 1.0f),
                     kEyStartAph = param("LINE_EY_START_APH", 0.5f), kEyStartLip = param("LINE_EY_START_LIP", 1.0f);
  TractShape o;
  if (kEyStartEh > 0 && lineShapeOf(Ph::EH, &o)) *sh = lineShapeMix(*sh, o, kEyStartEh);
  if (kEyStartAe > 0 && lineShapeOf(Ph::AE, &o)) *sh = lineShapeMix(*sh, o, kEyStartAe);
  if (kEyStartAc != 1.0f) sh->ac *= kEyStartAc;
  if (kEyStartAph != 1.0f) sh->aPharynx *= kEyStartAph;
  if (kEyStartLip != 1.0f) sh->aLip *= kEyStartLip;
  if (kEyStartDx != 0) { sh->xc += kEyStartDx; sh->sc = lineVowelStrength(sh->xc, sh->wc, sh->length); }
  // Round K (EY agent10, lab knobs; 0 / 1: as before): eɪ heard as ɪ 280 times in 6980 (combo9), female 14 % vs male
  // 8 %. Against ten Kokoro voices (30 words of words1k) the e's middle sits at male 440/1650 Hz (Kokoro 520/1680) and
  // female 580/1790 (Kokoro 700/2100): F1 2-3 st low in both, the female's F2 too 2-3 st low. The female e mixed
  // toward ɛ (LINE_EY_F_START_EH) or æ (LINE_EY_F_START_AE), its pharynx x LINE_EY_F_START_APH, its place moved by
  // LINE_EY_F_START_DX; in a closed syllable (Shape::eyctx & 1: "make", "late", "came") toward ɛ by LINE_EY_CL_START_EH,
  // before a nasal ("name", "main") by LINE_EY_NAS_START_EH as well.
  static const float kEyFStartEh = param("LINE_EY_F_START_EH", 0.0f), kEyFStartAe = param("LINE_EY_F_START_AE", 0.0f),
                     kEyFStartAph = param("LINE_EY_F_START_APH", 1.0f), kEyFStartDx = param("LINE_EY_F_START_DX", 0.0f),
                     kEyClStartEh = param("LINE_EY_CL_START_EH", 0.4f), kEyNasStartEh = param("LINE_EY_NAS_START_EH", 0.0f);
  const float ehMix = (female ? kEyFStartEh : 0.0f) + ((ctx & 1) ? kEyClStartEh : 0.0f) + ((ctx & 2) ? kEyNasStartEh : 0.0f);
  if (ehMix > 0 && lineShapeOf(Ph::EH, &o)) *sh = lineShapeMix(*sh, o, std::min(ehMix, 1.0f));
  if (female && kEyFStartAe > 0 && lineShapeOf(Ph::AE, &o)) *sh = lineShapeMix(*sh, o, kEyFStartAe);
  if (female && kEyFStartAph != 1.0f) sh->aPharynx *= kEyFStartAph;
  if (female && kEyFStartDx != 0) { sh->xc += kEyFStartDx; sh->sc = lineVowelStrength(sh->xc, sh->wc, sh->length); }
  // The female e mixed toward the glide's end (i's row mixed 0.7 toward ɪ's, as LINE_EY_OFF_IH) by LINE_EY_F_START_END:
  // her F2 stays flat at ~1830 Hz over the first half of the e -> ɪ path and then steps ~300 Hz in 20-30 ms ("make").
  static const float kEyFStartEnd = param("LINE_EY_F_START_END", 0.0f);
  if (female && kEyFStartEnd > 0 && lineShapeOf(Ph::I, &o)) {
    TractShape ih;
    if (lineShapeOf(Ph::IH, &ih)) o = lineShapeMix(o, ih, 0.7f);
    *sh = lineShapeMix(*sh, o, kEyFStartEnd);
  }
}
// The ɪ end (i's shape): mixed back toward ɪ's own row by LINE_EY_OFF_IH (lab 2026-11-10: the i-shaped end is what
// Whisper hears as iː; 0.7 took the lab's eɪ errors 119 -> 39 of 1224 and words1000 1151 -> 1179); its place moved by
// LINE_EY_OFF_DX, its opening, lip area, pharynx and bump half-width x LINE_EY_OFF_AC / _LIP / _APH / _WC.
void eyEndShape(TractShape* sh, bool female, unsigned ctx) {
  static const float kEyOffDx = param("LINE_EY_OFF_DX", 0.0f), kEyOffAc = param("LINE_EY_OFF_AC", 1.0f),
                     kEyOffLip = param("LINE_EY_OFF_LIP", 1.0f), kEyOffAph = param("LINE_EY_OFF_APH", 1.0f),
                     kEyOffWc = param("LINE_EY_OFF_WC", 1.0f), kEyOffIh = param("LINE_EY_OFF_IH", 0.7f);
  // Round K (EY agent10, lab knobs; < 0 / 0: as before): the end's mix toward ɪ's row for the female voice
  // (LINE_EY_F_OFF_IH; her end F2 2230 vs Kokoro 2500) and in a closed syllable (LINE_EY_CL_OFF_IH); the end mixed back
  // toward the e start in a closed syllable (LINE_EY_CL_END_E: a shallower glide, the coda's own transition does the
  // rest).
  static const float kEyFOffIh = param("LINE_EY_F_OFF_IH", -1.0f), kEyClOffIh = param("LINE_EY_CL_OFF_IH", -1.0f),
                     kEyClEndE = param("LINE_EY_CL_END_E", 0.0f), kEyFOffDx = param("LINE_EY_F_OFF_DX", 0.0f);
  const float offIh = (ctx & 1) && kEyClOffIh >= 0 ? kEyClOffIh : female && kEyFOffIh >= 0 ? kEyFOffIh : kEyOffIh;
  TractShape ih;
  if (offIh > 0 && lineShapeOf(Ph::IH, &ih)) *sh = lineShapeMix(*sh, ih, offIh);
  if ((ctx & 1) && kEyClEndE > 0 && lineShapeOf(Ph::E, &ih)) { eyStartShape(&ih, female, ctx); *sh = lineShapeMix(*sh, ih, kEyClEndE); }
  if (female && kEyFOffDx != 0) { sh->xc += kEyFOffDx; sh->sc = lineVowelStrength(sh->xc, sh->wc, sh->length); }
  if (kEyOffWc != 1.0f) sh->wc *= kEyOffWc;
  if (kEyOffAc != 1.0f) sh->ac *= kEyOffAc;
  if (kEyOffLip != 1.0f) sh->aLip *= kEyOffLip;
  if (kEyOffAph != 1.0f) sh->aPharynx *= kEyOffAph;
  if (kEyOffDx != 0 || kEyOffWc != 1.0f) { sh->xc += kEyOffDx; sh->sc = lineVowelStrength(sh->xc, sh->wc, sh->length); }
}

// The standalone ɪ (not the end of eɪ aɪ ɔɪ, which read ɪ's row through lineShapeOf; nor d, which borrows it): mixed
// toward e (LINE_IH_MIX_E), ɛ (LINE_IH_MIX_EH) or ə (LINE_IH_MIX_SCHWA), its place moved by LINE_IH_DX, its opening x
// LINE_IH_AC (IH agent 2026-11-11 lab knobs; 0 / 0 / 0 / 0 / 1: as before). Against five Kokoro voices per sex (the
// one-syllable words of words1k) the ɪ's middle sits at 374/1973 Hz (male; Kokoro 469/1795), female 434/2230
// (547/2106): F2 at puhuja's own iː (1969) and eɪ's end (1923), F1 2-4 st low; Whisper heard ɪ as eɪ 361 and iː 178
// times in one-syllable words (dict rank 9: 45 % wrong).
void ihShape(TractShape* sh) {
  static const float kIhE = param("LINE_IH_MIX_E", 0.3f), kIhEh = param("LINE_IH_MIX_EH", 0.0f),
                     kIhSchwa = param("LINE_IH_MIX_SCHWA", 0.0f), kIhDx = param("LINE_IH_DX", 0.0f),
                     kIhAc = param("LINE_IH_AC", 1.0f);
  TractShape o;
  if (kIhE > 0 && lineShapeOf(Ph::E, &o)) *sh = lineShapeMix(*sh, o, kIhE);
  if (kIhEh > 0 && lineShapeOf(Ph::EH, &o)) *sh = lineShapeMix(*sh, o, kIhEh);
  if (kIhSchwa > 0 && lineShapeOf(Ph::Schwa, &o)) *sh = lineShapeMix(*sh, o, kIhSchwa);
  if (kIhAc != 1.0f) sh->ac *= kIhAc;
  if (kIhDx != 0) { sh->xc += kIhDx; sh->sc = lineVowelStrength(sh->xc, sh->wc, sh->length); }
}


// The stressed ʌ's own shape on top of its row (round-F ʌ agent 2026-11-11, lab knobs; defaults reproduce the row,
// which also makes the word-final ə (LINE_SCHWA_FINAL_MIX) and the oʊ start, so the row itself stays). Dictionary
// rank 11: ʌ 19.9 % wrong (Kokoro 10.1 %), heard as ɛ 147, æ 93, ə 69, ɝ 54; against ten Kokoro voices its F2 is
// +1.2 st (male), F3 -2.0 / -2.1 st, F1 -1.8 st (female). Mixed toward ɑ (LINE_AH_MIX_AA), ɔ (_AO), ə (_SCHWA);
// place moved by LINE_AH_DX; opening / width / pharynx / oral / lips / larynx x _AC _WC _APH _AOR _LIPA _LIPL _ALAR;
// LINE_AH_LEN cm added to the tract; after a coronal or before a nasal a further share of ɑ (_COR_BACK, _NAS_BACK);
// the female voice's opening and pharynx x LINE_AH_F_AC, _F_APH.
bool coronalPh(Ph p) {
  return p == Ph::T || p == Ph::D || p == Ph::S || p == Ph::Z || p == Ph::N || p == Ph::L || p == Ph::TH ||
         p == Ph::DH || p == Ph::SH || p == Ph::ZH || p == Ph::CH || p == Ph::JH || p == Ph::RR;
}
void ahShape(TractShape* sh, Ph prev, const Ph* next, bool female, bool afterPause) {
  static const float kAa = param("LINE_AH_MIX_AA", 0.1f), kAo = param("LINE_AH_MIX_AO", 0.0f),
                     kSw = param("LINE_AH_MIX_SCHWA", 0.0f), kAe = param("LINE_AH_MIX_AE", 0.0f),
                     kDx = param("LINE_AH_DX", 0.0f), kAc = param("LINE_AH_AC", 1.0f), kWc = param("LINE_AH_WC", 1.0f),
                     kAph = param("LINE_AH_APH", 0.7f), kAor = param("LINE_AH_AOR", 1.3f), kLipA = param("LINE_AH_LIPA", 1.0f),
                     kLipL = param("LINE_AH_LIPL", 1.0f), kAlar = param("LINE_AH_ALAR", 1.0f), kLen = param("LINE_AH_LEN", 0.0f),
                     kCorBack = param("LINE_AH_COR_BACK", 0.0f), kNasBack = param("LINE_AH_NAS_BACK", 0.0f),
                     kFAc = param("LINE_AH_F_AC", 1.0f), kFAph = param("LINE_AH_F_APH", 1.0f);
  TractShape o;
  const bool nas = next && (*next == Ph::N || *next == Ph::M || *next == Ph::NG);
  const float aa = kAa + (coronalPh(prev) ? kCorBack : 0.0f) + (nas ? kNasBack : 0.0f);
  if (aa > 0 && lineShapeOf(Ph::AA, &o)) *sh = lineShapeMix(*sh, o, std::min(aa, 1.0f));
  if (kAo > 0 && lineShapeOf(Ph::AO, &o)) *sh = lineShapeMix(*sh, o, kAo);
  if (kSw > 0 && lineShapeOf(Ph::Schwa, &o)) *sh = lineShapeMix(*sh, o, kSw);
  if (kAe > 0 && lineShapeOf(Ph::AE, &o)) *sh = lineShapeMix(*sh, o, kAe);
  sh->ac *= kAc * (female ? kFAc : 1.0f);
  sh->wc *= kWc; sh->aPharynx *= kAph * (female ? kFAph : 1.0f); sh->aOral *= kAor; sh->aLip *= kLipA; sh->lLip *= kLipL; sh->aLarynx *= kAlar;
  sh->length += kLen;
  if (kDx != 0 || kWc != 1.0f || kLen != 0.0f) { sh->xc += kDx; sh->sc = lineVowelStrength(sh->xc, sh->wc, sh->length); }
  // (round-I ʌ agent 2026-11-12, lab knobs; all 0 / 1: as before) by context: the dictionary's ʌ errors sit after a
  // labial (p m: 22-26 %, heard æ ɛ: "punk" -> pang, "money" -> many), before ŋ (24 %: hunks -> hanks), before a coda l
  // (18 %, heard oʊ ɔ ʊ: "gulf" -> golf, "null" -> mol), word-initial (18 %, heard ə ɑ: "other" -> over), and in the
  // female voice (17 vs 12 %; its F1 666 Hz against Kokoro's 760, F3 2610 vs 3263). A further share of ɑ after a
  // labial (LINE_AH_LAB_BACK), after a nasal (_NASPREV_BACK), before ŋ (_NG_BACK); a share of ə before l
  // (_PRE_L_SCHWA); the opening x LINE_AH_PAUSE_AC after a pause; the female voice's ɑ share (_F_MIX_AA), tract
  // length (_F_LEN cm) and lip area (x _F_LIPA).
  static const float kLabBack = param("LINE_AH_LAB_BACK", 0.0f), kNasPrevBack = param("LINE_AH_NASPREV_BACK", 0.0f),
                     kNgBack = param("LINE_AH_NG_BACK", 0.0f), kPreLSw = param("LINE_AH_PRE_L_SCHWA", 0.0f),
                     kPauseAc = param("LINE_AH_PAUSE_AC", 1.0f), kFMixAa = param("LINE_AH_F_MIX_AA", 0.0f),
                     kFLen = param("LINE_AH_F_LEN", 0.0f), kFLipA = param("LINE_AH_F_LIPA", 1.0f);
  const bool lab = prev == Ph::P || prev == Ph::B || prev == Ph::M || prev == Ph::F || prev == Ph::VV;
  const bool nasPrev = prev == Ph::M || prev == Ph::N || prev == Ph::NG;
  const float aa2 = (lab ? kLabBack : 0.0f) + (nasPrev ? kNasPrevBack : 0.0f) + (next && *next == Ph::NG ? kNgBack : 0.0f) +
                    (female ? kFMixAa : 0.0f);
  if (aa2 > 0 && lineShapeOf(Ph::AA, &o)) *sh = lineShapeMix(*sh, o, std::min(aa2, 1.0f));
  if (kPreLSw > 0 && next && *next == Ph::L && lineShapeOf(Ph::Schwa, &o)) *sh = lineShapeMix(*sh, o, kPreLSw);
  if (afterPause) sh->ac *= kPauseAc;
  if (female) sh->aLip *= kFLipA;
  if (female && kFLen != 0.0f) { sh->length += kFLen; sh->sc = lineVowelStrength(sh->xc, sh->wc, sh->length); }
}

// The standalone ʊ's own shape on top of its row (round-H ʊ agent 2026-11-12, lab knobs; defaults reproduce the row, which
// also ends aʊ / oʊ elsewhere, so the row itself stays). Dictionary rank 14: ʊ 31.5 % wrong (Kokoro 13.9 %), heard as ʌ
// (male: "stood" -> stud, "booking" -> bucking) and oʊ (female: "pulls" -> polls, "bull" -> bowl). Mixed toward uː
// (LINE_UH_MIX_U), ə (_SCHWA), ɪ (_IH), ɔ (_AO); place moved by LINE_UH_DX; opening / width / pharynx / oral / lips /
// larynx x _AC _WC _APH _AOR _LIPA _LIPL _ALAR; LINE_UH_LEN cm added to the tract; the female voice's opening, pharynx
// and lips x LINE_UH_F_AC, _F_APH, _F_LIPA, and LINE_UH_F_MIX_U more of uː; after a labial (p b f v m w) a further
// LINE_UH_LAB_MIX_U of uː.
void uhShape(TractShape* sh, Ph prev, const Ph* next, bool female) {
  static const float kU = param("LINE_UH_MIX_U", 0.0f), kSw = param("LINE_UH_MIX_SCHWA", 0.0f), kIh = param("LINE_UH_MIX_IH", 0.0f),
                     kAo = param("LINE_UH_MIX_AO", 0.0f), kDx = param("LINE_UH_DX", 0.0f), kAc = param("LINE_UH_AC", 1.0f),
                     kWc = param("LINE_UH_WC", 1.0f), kAph = param("LINE_UH_APH", 1.0f), kAor = param("LINE_UH_AOR", 1.0f),
                     kLipA = param("LINE_UH_LIPA", 1.0f), kLipL = param("LINE_UH_LIPL", 1.0f), kAlar = param("LINE_UH_ALAR", 1.0f),
                     kLen = param("LINE_UH_LEN", 0.0f), kFAc = param("LINE_UH_F_AC", 1.0f), kFAph = param("LINE_UH_F_APH", 1.0f),
                     kFLipA = param("LINE_UH_F_LIPA", 1.0f), kFU = param("LINE_UH_F_MIX_U", 0.0f), kLabU = param("LINE_UH_LAB_MIX_U", 0.0f);
  const bool lab = prev == Ph::P || prev == Ph::B || prev == Ph::F || prev == Ph::VV || prev == Ph::M || prev == Ph::W;
  const float u = kU + (female ? kFU : 0.0f) + (lab ? kLabU : 0.0f);
  TractShape o;
  if (u > 0 && lineShapeOf(Ph::U, &o)) *sh = lineShapeMix(*sh, o, std::min(u, 1.0f));
  if (kSw > 0 && lineShapeOf(Ph::Schwa, &o)) *sh = lineShapeMix(*sh, o, kSw);
  if (kIh > 0 && lineShapeOf(Ph::IH, &o)) *sh = lineShapeMix(*sh, o, kIh);
  if (kAo > 0 && lineShapeOf(Ph::AO, &o)) *sh = lineShapeMix(*sh, o, kAo);
  // Before an ɹ ("sure", "tour", "insure"): Kokoro male holds F2 ~900 Hz (back, rounded) for ~90 ms before the r-colour,
  // puhuja 1200 with F3 falling from a third in, heard as ɝ ("sir", "answer"): toward o (LINE_UH_R_MIX_O), lips x
  // LINE_UH_R_LIPA.
  static const float kRO = param("LINE_UH_R_MIX_O", 0.0f), kRLipA = param("LINE_UH_R_LIPA", 1.0f);
  const bool preR = next && *next == Ph::RR;
  if (preR && kRO > 0 && lineShapeOf(Ph::O, &o)) *sh = lineShapeMix(*sh, o, kRO);
  if (preR) sh->aLip *= kRLipA;
  sh->ac *= kAc * (female ? kFAc : 1.0f);
  sh->wc *= kWc; sh->aPharynx *= kAph * (female ? kFAph : 1.0f); sh->aOral *= kAor; sh->aLip *= kLipA * (female ? kFLipA : 1.0f);
  sh->lLip *= kLipL; sh->aLarynx *= kAlar;
  sh->length += kLen;
  if (kDx != 0 || kWc != 1.0f || kLen != 0.0f) { sh->xc += kDx; sh->sc = lineVowelStrength(sh->xc, sh->wc, sh->length); }
}


}  // namespace

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
  // On the line the per-phoneme level (the arc, accent, stress) is a lung
  // pressure factor: the output is the physics', so the level had no way in
  // and every syllable came at one pressure (the last syllable of "robot"
  // as loud as the first; Kokoro 5-15 dB down, 2026-11-02).
  // (set on the event, not here: emit() plans a whole phrase before any of it
  // is synthesized, so a member set here is always the phrase's last level)
  // (the prosodic factor only: shape.level also carries the cascade's
  // per-phoneme loudness table, and fed to the pressure it took 10-15 dB
  // off ɪ, f, ɹ, ð and z in "The world is very different now", 14:40)
  const float lineLevelNow = tract_ == Tract::Line && (isVowel(tok.ph) || isSonorant(tok.ph)) ? std::clamp(tok.shape.prosody, 0.3f, 3.0f) : -1.0f;
  if (lineLevelNow > 0 && std::getenv("PUHUJA_LINE_TRACE")) std::fprintf(stderr, "linelevel t %.3f ph %d level %.3f\n", timeline_ / fs_, static_cast<int>(tok.ph), tok.shape.prosody);
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
    prevVoicedObstruent_ = 0;
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
    // SYNTH_DRAIN_SCALE: the start (with SYNTH_DRAIN_LEARN 0, the whole) of the planner's air scale. Learned on
    // kennedy.txt it sits at its 0.5 clamp from the second breath on, both voices (2026-11-11).
    if (drainScale_ < 0) drainScale_ = drainScale();
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
  // The ʤ's frication is an emitted ʒ (emitAffricate): the ʒ's own glottis (LINE_AG_ZH) stays with the standalone ʒ
  // (2026-11-09: the ʒ candidate 0.17 took the ʤ's release hiss away, ʤ errors 172 -> 205 in the lab).
  if (inAffricate_ && tok.ph == Ph::ZH) aero.ag = kAgVoicedFric;
  if (tract_ == Tract::Line && jhPressed_ && e.press < 0) { e.press = 1.0f; e.pressMs = 20; jhPressed_ = false; }  // (after LINE_JH_PRESS)
  // LINE_V_PRESS (lab 2026-11-09): the lung pressure eases to this share through a v (the voice bar and the hiss both
  // softer, an approximant-like v as Kokoro's) and comes back on the next sound; 1: unchanged.
  { static const float kVPress = param("LINE_V_PRESS", 0.85f);
    if (tract_ == Tract::Line && kVPress < 1.0f) {
      if (tok.ph == Ph::VV) { e.press = kVPress; e.pressMs = 20; }
      else if (lastPh_ == Ph::VV && e.press < 0) { e.press = 1.0f; e.pressMs = 20; }
    } }
  // z lab knobs (round D, 2026-11-11; the z is heard as s 1068 of 1892 dictionary errors: its voicing dies 20 ms in
  // while the glottis opens to LINE_AG_Z over LINE_ABDUCT_MS, voice bar -13 -> -35 dB re the word's peak, hiss -12):
  // LINE_Z_PRESS the lung pressure share through a z (1: unchanged); LINE_Z_AG_FINAL / LINE_Z_AG_MED the z's glottis
  // before a pause / before a vowel (< 0: LINE_AG_Z); LINE_Z_ABDUCT_MS its opening time (< 0: LINE_ABDUCT_MS);
  // LINE_Z_EXPAND the cavity walls' expansion sink through a z (as behind a voiced closure; <= 1: off).
  static const float kZPress = param("LINE_Z_PRESS", 1.0f), kZAgFinal = param("LINE_Z_AG_FINAL", -1.0f),
                     kZAgMed = param("LINE_Z_AG_MED", -1.0f), kZAbductMs = param("LINE_Z_ABDUCT_MS", -1.0f),
                     kZExpand = param("LINE_Z_EXPAND", 0.0f);
  if (tract_ == Tract::Line && kZPress < 1.0f) {
    if (tok.ph == Ph::Z) { e.press = kZPress; e.pressMs = 20; }
    else if (lastPh_ == Ph::Z && e.press < 0) { e.press = 1.0f; e.pressMs = 20; }
  }
  // (LINE_DZ_Z_AG / LINE_DZ_Z_EXPAND, D agent9 round J, below at the d's release, where a z after a d gets its glottis
  // and sink: its own event rides on the release) here: the same for a z that emits after a d, and the sink's end.
  static const float kDzZAg = param("LINE_DZ_Z_AG", -1.0f), kDzZExpand = param("LINE_DZ_Z_EXPAND", 0.0f);
  if (tract_ == Tract::Line && tok.ph == Ph::Z && lastPh_ == Ph::D) {
    if (kDzZAg >= 0) aero.ag = kDzZAg;
    if (kDzZExpand > 1.0f) { e.expand = kDzZExpand; e.expandMs = static_cast<float>(dur / ms); }
  } else if (tract_ == Tract::Line && kDzZExpand > 1.0f && lastPh_ == Ph::Z && e.expand < 0) { e.expand = 1.0f; e.expandMs = 5; }
  if (tract_ == Tract::Line && tok.ph == Ph::Z) {
    const bool zFinal = !next || *next == Ph::Stop;
    if (zFinal && kZAgFinal >= 0) aero.ag = kZAgFinal;
    else if (next && isVowel(*next) && kZAgMed >= 0) aero.ag = kZAgMed;
  }
  if (tract_ == Tract::Line && kZExpand > 1.0f) {
    if (tok.ph == Ph::Z) { e.expand = kZExpand; e.expandMs = static_cast<float>(dur / ms); }
    else if (lastPh_ == Ph::Z && e.expand < 0) { e.expand = 1.0f; e.expandMs = 5; }
  }
  const bool shFullVowel = tok.ph == Ph::SH && next && isVowel(*next) && *next != Ph::Schwa;  // (LINE_SH_VOWEL_*)
  // ʃ before a full vowel, its own (SH agent 10, round K; lab knobs, < 0 / 1: as before). Dictionary (combo9): this ʃ
  // is 40 % wrong word-initially (heard s 177, f 105: "Shall" -> sell, "Share" -> fair), the -tion ʃ before ə 3 %.
  // Against Kokoro's initial ʃ (14 words, 4 voices) puhuja's is silent for ~25 ms and then swells over 40-50 ms to
  // -5..-12 dB re the vowel (Kokoro a flat block at -12..-16 from its first 5 ms), peaks at 4.5 kHz (Kokoro male
  // 3.4) and leaves no palatal transition (the 0.7-2.7 kHz centroid at the vowel's start 800-1300 Hz, Kokoro male
  // 1500-1900 falling ~400 Hz in 60 ms). Not the ʧ's frication (inAffricate_), not the -tion ʃ.
  //   LINE_SH_V_AG: its glottis (cm^2; < 0: LINE_AG_FRIC). LINE_SH_V_PRESS: the breath x this through it (back at the
  //   vowel). LINE_SH_V_PAUSE_FRIC: its frication gain x this after a pause (a word-initial ʃ phrase-first).
  const bool shV = shFullVowel && !inAffricate_;
  static const float kShVAg = param("LINE_SH_V_AG", -1.0f), kShVPress = param("LINE_SH_V_PRESS", 1.0f),
                     kShVPressMs = param("LINE_SH_V_PRESS_MS", 20.0f), kShVPauseFric = param("LINE_SH_V_PAUSE_FRIC", 1.0f);
  if (tract_ == Tract::Line && shV && kShVAg >= 0) aero.ag = kShVAg;
  if (tract_ == Tract::Line && kShVPress != 1.0f) {
    if (shV) { e.press = kShVPress; e.pressMs = kShVPressMs; }
    else if (lastPh_ == Ph::SH && e.press < 0) { e.press = 1.0f; e.pressMs = 20; }
  }
  // ð lab knobs (DH agent7, round H; < 0 / 1: as before): after a pause its glottis LINE_DH_PAUSE_AG (cm^2) and its slit
  // x LINE_DH_PAUSE_AC (pause-initial ð 67 % wrong in the dictionary, medial 44 %); the lung pressure share through a ð
  // LINE_DH_PRESS and the cavity walls' expansion sink LINE_DH_EXPAND (as the z's: keep the voice under the slit).
  static const float kDhPauseAg = param("LINE_DH_PAUSE_AG", -1.0f), kDhPauseAc = param("LINE_DH_PAUSE_AC", 1.0f),
                     kDhPress = param("LINE_DH_PRESS", 1.0f), kDhExpand = param("LINE_DH_EXPAND", 0.0f);
  if (tract_ == Tract::Line && tok.ph == Ph::DH && prevWasPause_ && kDhPauseAg >= 0) aero.ag = kDhPauseAg;
  if (tract_ == Tract::Line && kDhPress != 1.0f) {
    if (tok.ph == Ph::DH) { e.press = kDhPress; e.pressMs = 20; }
    else if (lastPh_ == Ph::DH && e.press < 0) { e.press = 1.0f; e.pressMs = 20; }
  }
  // ... and a sink flow through the ð (LINE_DH_SINK_FLOW cm^3/s; 0: off): the pause-initial ð's voice bar sat at -22..-39 dB
  // re the word's peak through "Then." (Kokoro -10..-16 from its first 5 ms), the mouth pressure behind the slit stalling the folds.
  static const float kDhSinkFlow = param("LINE_DH_SINK_FLOW", 0.0f);
  if (tract_ == Tract::Line && kDhSinkFlow > 0) {
    if (tok.ph == Ph::DH) e.lineSink = kDhSinkFlow;
    else if (lastPh_ == Ph::DH) e.lineSink = 0.0f;
  }
  if (tract_ == Tract::Line && kDhExpand > 1.0f) {
    if (tok.ph == Ph::DH) { e.expand = kDhExpand; e.expandMs = static_cast<float>(dur / ms); }
    else if (lastPh_ == Ph::DH && e.expand < 0) { e.expand = 1.0f; e.expandMs = 5; }
  }
  e.ac = aero.ac;
  // The line's fricative slits are wider than the cascade's table: a 0.05 cm^2
  // s passes at most 175 cm^3/s at 7 cmH2O where a spoken s runs 300-500, and
  // its hiss sat 18 dB under the vowels (Kokoro 5-7), 2026-11-02.
  if (tract_ == Tract::Line) e.ac = lineSlit(tok.ph, e.ac);
  // the standalone ʒ's slit x LINE_ZH_AC_MUL (ZH agent, round H; 1: off; LINE_AC_ZH also sets the ʤ's frication)
  { static const float kLineZhAcMul = param("LINE_ZH_AC_MUL", 1.0f);
    if (tract_ == Tract::Line && kLineZhAcMul != 1.0f && tok.ph == Ph::ZH && !inAffricate_ && e.ac > 0) e.ac *= kLineZhAcMul; }
  if (tract_ == Tract::Line && tok.ph == Ph::DH && prevWasPause_ && kDhPauseAc != 1.0f) e.ac *= kDhPauseAc;  // (LINE_DH_PAUSE_AC)
  // l's lateral opening by its place (L agent, round E; 1: LINE_L_AC alone): a coda l (no vowel next) x LINE_L_CODA_AC
  // (a wider opening raises its F1 toward Kokoro's 450), an l after a plosive x LINE_L_STOP_AC.
  if (tract_ == Tract::Line && tok.ph == Ph::L) {
    static const float kLCodaAc = param("LINE_L_CODA_AC", 1.0f), kLStopAc = param("LINE_L_STOP_AC", 1.0f);
    if (!(next && isVowel(*next))) e.ac *= kLCodaAc;
    else if (prevWasPlosive_) e.ac *= kLStopAc;
  }
  // (W agent round G, lab; 1: as before) a w's own area after a plosive x LINE_W_STOP_EAC, after s x LINE_W_S_AC. After a
  // phrase-initial plosive this area still lands: the w's pre-positioning event (prevWasPause_) is clamped to just
  // after the release and overrides the release's area (LINE_W_REL_AC applies to the medial ones).
  if (tract_ == Tract::Line && tok.ph == Ph::W) {
    static const float kWStopEAc = param("LINE_W_STOP_EAC", 1.0f), kWSAc = param("LINE_W_S_AC", 1.0f);
    if (prevWasPlosive_) e.ac *= kWStopEAc;
    else if (lastPh_ == Ph::S) e.ac *= kWSAc;
  }
  // v voiced through a wider slit (LINE_V_AC) has a weak hiss: its own gain
  // (2026-11-04; 1 = today's).
  static const float kLineVFric = param("LINE_V_FRIC", 8.0f);
  static const float kLineZFric = param("LINE_Z_FRIC", 1.0f);
  if (tract_ == Tract::Line) e.lineFricMul = tok.ph == Ph::VV ? kLineVFric : tok.ph == Ph::F ? (curStress_ ? kLineFFric : kLineFFricUnstr) : tok.ph == Ph::Z ? kLineZFric : tok.ph == Ph::ZH ? (inAffricate_ && kLineJhFric >= 0 ? kLineJhFric : kLineZhFric) : tok.ph == Ph::S ? kLineSFric : tok.ph == Ph::SH ? kLineShFric * (shFullVowel ? (tok.wordStart || kLineShVowelFricLink < 0 ? kLineShVowelFric : kLineShVowelFricLink) : 1.0f) : 1.0f;
  if (tract_ == Tract::Line && shV && prevWasPause_ && kShVPauseFric != 1.0f) e.lineFricMul *= kShVPauseFric;  // (LINE_SH_V_PAUSE_FRIC)
  // a word-final ʒ's frication gain x LINE_ZH_FRIC_FINAL (ZH agent, round H; 1: off): Kokoro's final ʒ ("garage", "beige",
  // "rouge") is voiceless (80-500 Hz -37..-45 dB re the word's peak) with a quieter hiss (2-8 kHz -17..-26) than a medial one
  { static const float kLineZhFricFinal = param("LINE_ZH_FRIC_FINAL", 0.37f);
    if (tract_ == Tract::Line && kLineZhFricFinal != 1.0f && tok.ph == Ph::ZH && !inAffricate_ && (!next || *next == Ph::Stop)) e.lineFricMul *= kLineZhFricFinal; }
  // The v by its place in the phrase (V agent, round F; all 1 / < 0: off). The dictionary's v errors by position: initial
  // 43 % (the phrase-initial v silent for its first 90 ms: both bands -40..-50 dB re the word's peak while the breath
  // rises; Kokoro voiced at -13..-18 from its first 10 ms), medial 16 %, final 27 %. LINE_V_FRIC_INIT / _MED / _FINAL
  // x the v's hiss gain, LINE_V_AC_INIT / _MED / _FINAL x its slit, LINE_V_AG_INIT / _FINAL its glottis (< 0: LINE_AG_V).
  vAfterFric_ = false;
  if (tract_ == Tract::Line && tok.ph == Ph::VV) {
    static const float kVFricInit = param("LINE_V_FRIC_INIT", 1.0f), kVFricMed = param("LINE_V_FRIC_MED", 1.0f),
                       kVFricFinal = param("LINE_V_FRIC_FINAL", 0.5f), kVAcInit = param("LINE_V_AC_INIT", 1.0f),
                       kVAcMed = param("LINE_V_AC_MED", 1.0f), kVAcFinal = param("LINE_V_AC_FINAL", 1.0f),
                       kVAgInit = param("LINE_V_AG_INIT", -1.0f), kVAgFinal = param("LINE_V_AG_FINAL", -1.0f);
    const bool vFinal = !next || *next == Ph::Stop;
    const float fm = prevWasPause_ ? kVFricInit : vFinal ? kVFricFinal : kVFricMed;
    const float am = prevWasPause_ ? kVAcInit : vFinal ? kVAcFinal : kVAcMed;
    if (fm != 1.0f && e.lineFricMul > 0) e.lineFricMul *= fm;
    // LINE_V_FRIC_AFTER_FRIC (< 0: off): the v's hiss gain right after another fricative ("is very": its event starts
    // while the z's slit and flow are still there, and the v's louder hiss (LINE_V_FRIC 8, LINE_V_RE_SCALE 2.5) came
    // in as a 17 dB click at the z-v boundary); there the v's own Re scale is off too.
    static const float kVFricAfterFric = param("LINE_V_FRIC_AFTER_FRIC", 1.0f);
    if (kVFricAfterFric >= 0 && (lastPh_ == Ph::S || lastPh_ == Ph::Z || lastPh_ == Ph::SH || lastPh_ == Ph::ZH ||
                                 lastPh_ == Ph::F || lastPh_ == Ph::TH || lastPh_ == Ph::DH)) {
      e.lineFricMul = kVFricAfterFric;
      vAfterFric_ = true;
    }
    if (am != 1.0f && e.ac > 0) e.ac *= am;
    if (prevWasPause_ && kVAgInit >= 0) aero.ag = kVAgInit;
    else if (vFinal && kVAgFinal >= 0) aero.ag = kVAgFinal;
  }
  // The token after a θ keeps x this of the θ's hiss gain (LINE_TH_FRIC) while the slit opens (0: today's: the gain reverted to 1 at the
  // boundary while the slit was still narrow, a +10 dB burst into the vowel and a +10 dB s-like tail into the pause; lab 2026-11-09).
  { static const float kThCarry = param("LINE_TH_FRIC_CARRY", 0.0f);
    if (tract_ == Tract::Line && kThCarry > 0) {
      static const float thFric = param("LINE_TH_FRIC", 0.3f);
      if (lastPh_ == Ph::TH) thCarry_ = in.cls == Cls::Fricative ? -1.0f : thFric * kThCarry;  // ... and through the whole pause after it
      else if (in.cls != Cls::Stop) thCarry_ = -1.0f;
      if (thCarry_ >= 0 && e.lineFricMul == 1.0f) e.lineFricMul = thCarry_;
      if (std::getenv("PUHUJA_LINE_TRACE")) std::fprintf(stderr, "thcarry t %.3f cls %d last %d mul %.3f\n", e.t / fs_, static_cast<int>(in.cls), static_cast<int>(lastPh_), e.lineFricMul);
    } }
  // LINE_TH_PAUSE_CARRY (TH agent7, round H; 0: off): only the pause after a θ keeps x this of its gain (LINE_TH_FRIC), not
  // the vowel after it (the +10 dB end burst into a vowel is a cue Whisper uses: LINE_TH_FRIC_CARRY cost 20 errors), while
  // the pause's +10 dB, 5.5-6 kHz tail after a final θ is an s ("paths" -> "pass", "Seth" -> "sex", "both" -> "bose").
  { static const float kThPauseCarry = param("LINE_TH_PAUSE_CARRY", 0.0f);
    if (tract_ == Tract::Line && kThPauseCarry > 0) {
      static const float thFric = param("LINE_TH_FRIC", 0.3f);
      if (lastPh_ == Ph::TH) thPauseCarry_ = tok.ph == Ph::Stop ? thFric * kThPauseCarry : -1.0f;  // (the pause is a run of tokens)
      else if (tok.ph != Ph::Stop) thPauseCarry_ = -1.0f;
      if (thPauseCarry_ >= 0 && e.lineFricMul == 1.0f) e.lineFricMul = thPauseCarry_;
    } }
  // The token after a ð opens the slit with x this of the frication gain (LINE_DH_REL_FRIC, DH agent7, round H; < 0: off,
  // the token's own). The ð of "Other." had no hiss of its own (LINE_DH_FRIC 0 / 0.5 / 1 rendered alike) and a 4-8 kHz flash
  // as the slit opened into the vowel: -18 dB re the word's peak where the vowels' band sits at -50 (Kokoro: a weak hiss
  // through the ð at the vowels' level, no flash). But Whisper uses the flash as the ð's frication: 0 -> 111 errors of 280
  // (base 88, b 24 d 20), 0.3 -> 96; 0.7 keeps the front key's gain (75 vs 76) with the lab's snap term 4σ under the base's.
  { static const float kDhRelFric = param("LINE_DH_REL_FRIC", -1.0f);
    if (tract_ == Tract::Line && kDhRelFric >= 0 && lastPh_ == Ph::DH && in.cls != Cls::Fricative && e.lineFricMul == 1.0f) e.lineFricMul = kDhRelFric; }
  // The f before another voiceless fricative ("beliefs") x this (LINE_F_CLUSTER_FRIC; Jyrki 2026-11-08: "in 'beliefs',
  // the f and s could have roughly equal strength and duration": the f 6 dB under the s).
  if (tract_ == Tract::Line && tok.ph == Ph::F && next && (*next == Ph::S || *next == Ph::SH || *next == Ph::TH) && e.lineFricMul > 0) {
    static const float kLineFClusterFric = param("LINE_F_CLUSTER_FRIC", 1.41f);
    e.lineFricMul *= kLineFClusterFric;
  }
  // ... and the s after an f/θ/ʃ x this (LINE_CLUSTER_S_FRIC; Jyrki: "better if f and s are both -42 db in beliefs").
  if (tract_ == Tract::Line && tok.ph == Ph::S && (lastPh_ == Ph::F || lastPh_ == Ph::TH || lastPh_ == Ph::SH) && e.lineFricMul > 0) {
    static const float kLineClusterSFric = param("LINE_CLUSTER_S_FRIC", 0.71f);
    e.lineFricMul *= kLineClusterSFric;
    static const float kLineThSFric = param("LINE_TH_S_FRIC", 1.0f);  // ... x this more after a θ ("paths" -> "pass"; TH agent7)
    if (lastPh_ == Ph::TH) e.lineFricMul *= kLineThSFric;
  }
  // ... and a phrase-final f x this (LINE_FINAL_F_FRIC; Jyrki 2026-11-08: "the ending f could be less strong by 3 db
  // in 'life'": that f -36 dB, 10 dB over the one of "belief").
  if (tract_ == Tract::Line && tok.ph == Ph::F && (!next || *next == Ph::Stop) && e.lineFricMul > 0) {
    static const float kLineFinalFFric = param("LINE_FINAL_F_FRIC", 0.71f);
    e.lineFricMul *= kLineFinalFFric;
  }
  // ... and a phrase-final θ x this (LINE_FINAL_TH_FRIC; lab 2026-11-09: "month" -35 dB against Kokoro's -19..-28).
  if (tract_ == Tract::Line && tok.ph == Ph::TH && (!next || *next == Ph::Stop) && e.lineFricMul > 0) {
    static const float kLineFinalThFric = param("LINE_FINAL_TH_FRIC", 1.0f);
    e.lineFricMul *= kLineFinalThFric;
  }
  // ... and a phrase-initial θ x LINE_INIT_TH_FRIC, a θ inside the phrase (not before a pause) x LINE_MED_TH_FRIC (TH agent7,
  // round H; 1: as before).
  if (tract_ == Tract::Line && tok.ph == Ph::TH && e.lineFricMul > 0) {
    static const float kLineInitThFric = param("LINE_INIT_TH_FRIC", 1.0f), kLineMedThFric = param("LINE_MED_TH_FRIC", 1.0f);
    if (prevWasPause_) e.lineFricMul *= kLineInitThFric;
    else if (next && *next != Ph::Stop) e.lineFricMul *= kLineMedThFric;
  }
  // ... and a phrase-final z x this (LINE_FINAL_Z_FRIC, lab 2026-11-11; 1: as before): "his" hissed at -6..-9 dB re the
  // word's peak for 140 ms (its 59 ms and ~80 more into the pause while the breath decays), Kokoro's final z -17..-20.
  if (tract_ == Tract::Line && tok.ph == Ph::Z && (!next || *next == Ph::Stop) && e.lineFricMul > 0) {
    static const float kLineFinalZFric = param("LINE_FINAL_Z_FRIC", 0.7f);
    e.lineFricMul *= kLineFinalZFric;
  }
  // (the pause after a z keeps its gain while the slit is still narrow: reset to 1 there, LINE_FINAL_Z_FRIC 0.35 came
  // back +12 dB at the pause's start)
  {
    static const float kLineFinalZFric = param("LINE_FINAL_Z_FRIC", 0.7f);
    if (tok.ph == Ph::Z) zPauseFric_ = kLineZFric * kLineFinalZFric;
    else if (tok.ph != Ph::Stop) zPauseFric_ = 1.0f;
    if (tract_ == Tract::Line && tok.ph == Ph::Stop && zPauseFric_ != 1.0f) e.lineFricMul = zPauseFric_;
    // Z-FINAL agent13 round L (lab; 1: as before): the hiss held into the pause after a final z x LINE_Z_PAUSE_FRIC.
    // Puhuja's final z hisses on 130-140 ms after its voice has gone (Kokoro 30-105 ms; an s 150-165): the tail is s.
    static const float kZPauseFric = param("LINE_Z_PAUSE_FRIC", 1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::Stop && zPauseFric_ != 1.0f && kZPauseFric != 1.0f) e.lineFricMul *= kZPauseFric;
  }
  // Z-FINAL agent13 round L (lab; 1: off): a final z's lung pressure x LINE_FINAL_Z_PRESS over LINE_FINAL_Z_PRESS_MS
  // (back to 1 after the pause): the breath eases off at the phrase's end, the hiss weakens, the folds keep going longer.
  {
    static const float kFinalZPress = param("LINE_FINAL_Z_PRESS", 1.0f), kFinalZPressMs = param("LINE_FINAL_Z_PRESS_MS", 30.0f);
    if (tract_ == Tract::Line && kFinalZPress != 1.0f && tok.ph == Ph::Z && (!next || *next == Ph::Stop)) {
      e.press = kFinalZPress;
      e.pressMs = kFinalZPressMs;
    }
  }
  if (tract_ == Tract::Line && kLineCloseFric != 1.0f && in.cls == Cls::Nasal) e.lineFricMul = kLineCloseFric;  // (LINE_CLOSE_FRIC: the nasal's oral closure)
  // LINE_T_ASP_HP (2026-11-05): a t's breath alone is high-passed (Hz, with
  // LINE_T_ASP_HP_POLES stages); every other sound puts LINE_ASP_HP back.
  // goog's and Kokoro's t aspiration ("power to") is a 4-12 kHz hiss 15-22 dB
  // over 0.3-2.5 kHz, ours a "hu" murmur as loud as the hiss (heard as
  // "grhu"); for p k h the low, formant-shaped part is what Whisper needs
  // (high-passing every breath: initial p 74 -> 36, h 27 -> 12 of 58).
  static const float kLineTAspHp = param("LINE_T_ASP_HP", 0.0f);
  if (tract_ == Tract::Line && kLineTAspHp > 0) e.lineAspHp = 0.0f;
  // A voiced fricative needs a slit wide enough to keep the folds going
  // (z, v, ð: at s's 0.05 cm^2 the mouth pressure rose to the lungs' and
  // the voice stopped), but the frication source faded above 0.12-0.22 cm^2
  // (set so that r, w, l would not hiss), so voiced it had no hiss: z and v
  // were heard as n, l, d (2026-11-04). Its own, wider fade range.
  static const float kVFricFade = param("LINE_VFRIC_FADE", 1.0f);
  static const float shFade = param("LINE_SH_FADE", 1.0f);  // ʃ: its frication's fade window's areas x this
  static const float kLineZFade = param("LINE_Z_FADE", -1.0f);  // z's own (< 0: LINE_VFRIC_FADE; lab 2026-11-11)
  // LINE_Z_RE_SCALE (lab 2026-11-11; 1: off): the z's jet Reynolds number x this. With a slit wide enough to keep the
  // voice (LINE_Z_AC 3, LINE_AG_Z 0.08: voice bar -2..-14 dB in "his") the jet stayed under LINE_RE_CRIT, no hiss
  // (-35..-48) at any LINE_Z_FRIC; at today's slit the mouth pressure (4400 dyn/cm^2) stops the folds within 10 ms.
  static const float kLineZReScale = param("LINE_Z_RE_SCALE", 1.0f);
  if (tok.ph != Ph::Stop) zBeforePause_ = tok.ph == Ph::Z;
  // LINE_Z_RE_BOTH (lab 2026-11-11; 0: off): through a z the jet's Reynolds number is low-passed (LINE_NOISE_RE_MS) on its
  // falls too: with the voice on, the flow pulses, and the hiss switched off and on with every glottal cycle (5-ms
  // frames -3..-45 dB, snap.py 6.7 -> 9.4 on the z words with LINE_Z_RE_SCALE 2.5).
  static const float kLineZReBoth = param("LINE_Z_RE_BOTH", 0.0f);
  if (tract_ == Tract::Line && kLineZReBoth > 0) e.lineFricReBoth = tok.ph == Ph::Z || (tok.ph == Ph::Stop && zBeforePause_) ? 1.0f : 0.0f;
  if (tract_ == Tract::Line && kLineZReScale != 1.0f) e.lineFricReMul = tok.ph == Ph::Z || (tok.ph == Ph::Stop && zBeforePause_) ? kLineZReScale : 1.0f;  // (kept into the pause while the slit is still narrow)
  // Z-FINAL agent13 round L (lab; 1: off): the same for a phrase-final z alone (and its pause), x LINE_FINAL_Z_RE: the
  // hiss starts earlier under the voice while the folds still run (Kokoro's final z: 20-40 ms of overlap, puhuja 0-10).
  static const float kLineFinalZRe = param("LINE_FINAL_Z_RE", 1.0f);
  if (tract_ == Tract::Line && kLineFinalZRe != 1.0f) {
    const bool fz = tok.ph == Ph::Z && (!next || *next == Ph::Stop);
    e.lineFricReMul = fz || (tok.ph == Ph::Stop && zBeforePause_) ? kLineFinalZRe * kLineZReScale : tok.ph == Ph::Z ? kLineZReScale : 1.0f;
  }
  {
    static const float kBzRe = param("LINE_BZ_RE", 1.0f);
    if (tract_ == Tract::Line && kBzRe != 1.0f && kLineFinalZRe == 1.0f && kLineZReScale == 1.0f && tok.ph != Ph::Z &&
        !(tok.ph == Ph::Stop && zBeforePause_) && info(tok.ph).cls != Cls::Plosive)
      e.lineFricReMul = 1.0f;
  }
  // LINE_V_RE_SCALE (V agent, round F; 1: off): the v's jet Reynolds number x this. Through the v's 0.147 cm^2 slit
  // (LINE_V_AC 2) the voiced flow pulses 0..230 cm^3/s and Re crosses LINE_RE_CRIT 1400 only at each pulse's peak
  // (noise trace of "over": the drive term > 0 a quarter of the cycle): the v's 2-8 kHz band sat at -42..-50 dB re the
  // word's peak in every position (Kokoro adam -14..-30), a hiss-less dip Whisper hears as b (419 of 830 substitutions).
  static const float kLineVReScale = param("LINE_V_RE_SCALE", 2.5f);
  if (tract_ == Tract::Line && kLineVReScale != 1.0f) {
    if (tok.ph == Ph::VV) e.lineFricReMul = vAfterFric_ ? 1.0f : kLineVReScale;
    else if (lastPh_ == Ph::VV && e.lineFricReMul < 0) e.lineFricReMul = 1.0f;
  }
  // LINE_V_RE_SMOOTH (V agent, round F; 0: off; the envelope of LINE_NOISE_RE_ENV_MS, or LINE_V_RE_ENV_MS through the v alone):
  // through a v the jet's Re is pulled this share toward its pitch-period mean, so the louder hiss does not pulse on and
  // off with each glottal cycle (snap.py 8-13 dB 6-10 ms into "give" / "love" at LINE_V_RE_SCALE 2, LINE_V_FRIC 5).
  static const float kLineVReSmooth = param("LINE_V_RE_SMOOTH", 1.0f);
  if (tract_ == Tract::Line && kLineVReSmooth > 0) {
    if (tok.ph == Ph::VV) e.lineFricReSmooth = kLineVReSmooth;
    else if (lastPh_ == Ph::VV) e.lineFricReSmooth = 0.0f;
  }
  // ʒ's jet (ZH agent, round H; LINE_ZH_RE_SCALE 1, LINE_ZH_RE_SMOOTH 0, LINE_ZH_RE_BOTH 0: off). With the folds going
  // through the ʒ the flow pulses, and the Re lag (LINE_NOISE_RE_MS, a running-minimum follower) sits on each cycle's
  // trough: the ʒ of "leisure", "pleasure", "Asia" had no hiss (2-8 kHz -38..-44 dB re the word's peak, Kokoro adam
  // -14, bella -14..-18; centroid 0.3-0.7 kHz vs 3.5-4.4) but a voice bar 6-15 dB over Kokoro's: a voiced stop,
  // heard as T / D ("letter", "platter", "beta"). The v's cure: the jet's Re x LINE_ZH_RE_SCALE and pulled
  // LINE_ZH_RE_SMOOTH toward its pitch-period mean (the envelope of LINE_V_RE_ENV_MS); LINE_ZH_RE_BOTH lags its falls too.
  // ZH lab (102 words x 2 voices, 2 draws): errors 73 -> 42 (RE_SCALE 2 + RE_SMOOTH 1; T 20 -> 5); with LINE_ZH_JET_D
  // 0.05, LINE_ZH_OFF_MS 50, LINE_ZH_FRIC 2.5 and the final ʒ (LINE_ZH_AG_FINAL 0.3, LINE_ZH_FRIC_FINAL 0.3): 32, words
  // right 111 -> 154, words1000 1383 -> 1383; RE_SCALE 2 alone: 35 but the hiss pulses with each cycle (snap +7 sigma).
  {
    static const float kLineZhReScale = param("LINE_ZH_RE_SCALE", 1.9f);
    static const float kLineZhReSmooth = param("LINE_ZH_RE_SMOOTH", 0.9f);
    static const float kLineZhReBoth = param("LINE_ZH_RE_BOTH", 0.0f);
    const bool zh = tok.ph == Ph::ZH && !inAffricate_;
    if (tract_ == Tract::Line && kLineZhReScale != 1.0f) {
      if (zh) e.lineFricReMul = kLineZhReScale;
      else if (lastPh_ == Ph::ZH && e.lineFricReMul < 0) e.lineFricReMul = 1.0f;
    }
    if (tract_ == Tract::Line && kLineZhReSmooth > 0) {
      if (zh) e.lineFricReSmooth = kLineZhReSmooth;
      else if (lastPh_ == Ph::ZH && e.lineFricReSmooth < 0) e.lineFricReSmooth = 0.0f;
    }
    if (tract_ == Tract::Line && kLineZhReBoth > 0) {
      if (zh) e.lineFricReBoth = 1.0f;
      else if (lastPh_ == Ph::ZH && e.lineFricReBoth < 0) e.lineFricReBoth = 0.0f;
    }
  }
  static const float kLineDhFade = param("LINE_DH_FADE", -1.0f);  // ð's frication fade-out areas (< 0: LINE_VFRIC_FADE; DH agent7)
  // ð's jet Reynolds number x LINE_DH_RE_SCALE and pulled LINE_DH_RE_SMOOTH toward its pitch-period mean (DH agent7, round
  // H; 1 / < 0: as before): the v's LINE_V_RE_SCALE route to a hiss that runs with the voice (lab: 2.5 heard as z / s, 94 vs 88 errors).
  static const float kLineDhReScale = param("LINE_DH_RE_SCALE", 1.0f), kLineDhReSmooth = param("LINE_DH_RE_SMOOTH", -1.0f);
  if (tract_ == Tract::Line && kLineDhReScale != 1.0f) {
    if (tok.ph == Ph::DH) e.lineFricReMul = kLineDhReScale;
    else if (lastPh_ == Ph::DH && e.lineFricReMul < 0) e.lineFricReMul = 1.0f;
  }
  if (tract_ == Tract::Line && kLineDhReSmooth >= 0) {
    if (tok.ph == Ph::DH) e.lineFricReSmooth = kLineDhReSmooth;
    else if (lastPh_ == Ph::DH && e.lineFricReSmooth < 0) e.lineFricReSmooth = 0.0f;
  }
  if (tract_ == Tract::Line)
    e.lineFricFade = tok.ph == Ph::ZH && !inAffricate_ && kLineZhFade >= 0 ? kLineZhFade : tok.ph == Ph::Z && kLineZFade >= 0 ? kLineZFade : tok.ph == Ph::DH && kLineDhFade >= 0 ? kLineDhFade : tok.ph == Ph::Z || tok.ph == Ph::ZH || tok.ph == Ph::VV || tok.ph == Ph::DH ? kVFricFade : tok.ph == Ph::SH ? shFade : 1.0f;
  e.lineX = tok.ph == Ph::NG ? lineNgPlace(lastPh_, next) : linePlaceOf(tok.ph);
  { static const float kLineZhX = param("LINE_ZH_X", -1.0f);  // ʒ's own place (ZH agent, round H; < 0: LINE_SH_X)
    if (tok.ph == Ph::ZH && !inAffricate_ && kLineZhX >= 0) e.lineX = kLineZhX; }
  e.lineXClosure = in.cls == Cls::Nasal;
  // A word-final n after an unstressed vowel closes at LINE_FN_X (Shape::nctx 2; N agent13 round L lab knob, 0: LINE_N_X):
  // its murmur's F2 sat at 1500-1700 Hz (male), Kokoro's at 1250-1400 ("Hansen", "Ryan", "Gavin").
  { static const float kFnX = param("LINE_FN_X", 0.0f);
    if (tok.ph == Ph::N && (tok.shape.nctx & 2) && kFnX > 0) e.lineX = kFnX; }
  if (tok.ph == Ph::SH && shFullVowel && kLineShVowelX > 0) e.lineX = kLineShVowelX;  // (LINE_SH_VOWEL_X)
  if (tok.ph == Ph::SH && shFullVowel && kLineShVowelAc != 1.0f && e.ac > 0) e.ac *= kLineShVowelAc;  // (LINE_SH_VOWEL_AC)
  // Stress as a push of the diaphragm (Jyrki 2026-11-04: "a soft pulse in
  // the beginning of the syllable rather than ramping up during the
  // syllable"): of a vowel's raise above the phrase's level 1, only
  // LINE_STRESS_HOLD is held through it; LINE_STRESS_PULSE x the raise comes
  // as a pressure pulse at the syllable's start (below, after the events).
  float stressExtra = 0;
  if (lineLevelNow > 1.0f && isVowel(tok.ph) && kStressPulse > 0) stressExtra = lineLevelNow - 1.0f;
  e.lineLevel = stressExtra > 0 ? 1.0f + kStressHold * stressExtra : lineLevelNow;
  // ŋ's breath x LINE_NG_PRESS over 20 ms, back to 1 at the next sound (lab knob 2026-11-11, NG agent 3; 1: as
  // before): in isolated words puhuja's ŋ murmur was 3 dB under the word's peak, Kokoro's 13 (male) and 9 (female).
  {
    static const float kLineNgPress = param("LINE_NG_PRESS", 1.0f);
    if (tract_ == Tract::Line && kLineNgPress != 1.0f) {
      if (tok.ph == Ph::NG) { e.press = kLineNgPress; e.pressMs = 20; }
      else if (lastPh_ == Ph::NG && e.press < 0) { e.press = 1.0f; e.pressMs = 20; }
    }
  }
  // ʒ's breath x LINE_ZH_PRESS over LINE_ZH_PRESS_MS, back to 1 at the next sound (ZH agent, round H; 1: off): more flow
  // through the slit, a hiss above LINE_RE_CRIT with the folds still going.
  {
    static const float kLineZhPress = param("LINE_ZH_PRESS", 1.0f), kLineZhPressMs = param("LINE_ZH_PRESS_MS", 20.0f);
    if (tract_ == Tract::Line && kLineZhPress != 1.0f) {
      if (tok.ph == Ph::ZH && !inAffricate_) { e.press = kLineZhPress; e.pressMs = kLineZhPressMs; }
      else if (lastPh_ == Ph::ZH && e.press < 0) { e.press = 1.0f; e.pressMs = 20; }
    }
  }
  {
    const bool post = tok.ph == Ph::SH || tok.ph == Ph::ZH || tok.ph == Ph::CH || tok.ph == Ph::JH;
    static const float kLineZHw = param("LINE_Z_HW", 1.0f);  // the z's constriction width along the tract (lab 2026-11-11)
    static const float kLineNgHw = param("LINE_NG_HW", 1.0f);  // ŋ's constriction half-width (lab knob 2026-11-11, NG agent 3; 1: as before)
    // l's constriction half-width along the tract (L agent, round E; 1 as before): the tongue tip's contact is short, and a
    // long 0.6 cm^2 channel low-passes the l (2-5 kHz 15-20 dB under Kokoro's onset l, re the vowel).
    static const float kLineLHw = param("LINE_L_HW", 1.0f);
    // ʒ's own groove half-width, lip rounding, slit-height scale and jet mix (ZH agent, round H; < 0: the ʃ ʒ ʧ ʤ ones)
    static const float kLineZhHw = param("LINE_ZH_HW", -1.0f), kLineZhLip = param("LINE_ZH_LIP", -1.0f);
    static const float kLineZhSlit = param("LINE_ZH_SLIT_SCALE", -1.0f), kLineZhJetMix = param("LINE_ZH_JET_MIX", -1.0f);
    const bool zhOwn = tok.ph == Ph::ZH && !inAffricate_;
    static const float kLineDhHw = param("LINE_DH_HW", 1.0f);  // ð's constriction half-width along the tract (DH agent7; 1: as before)
    e.lineHw = zhOwn && kLineZhHw >= 0 ? kLineZhHw : post ? kLinePostHw : tok.ph == Ph::DH ? kLineDhHw : tok.ph == Ph::F ? kLineFHw : tok.ph == Ph::Z ? kLineZHw : tok.ph == Ph::NG ? kLineNgHw : tok.ph == Ph::L ? kLineLHw : 1.0f;
    e.lineLip = zhOwn && kLineZhLip >= 0 ? kLineZhLip : post ? kLinePostLip : 0.0f;
    static const float kLineZJetD = param("LINE_Z_JET_D", 0.0f), kLineZLpr = param("LINE_Z_LPR", 0.0f);  // z's jet and band (0: the shared ones; lab 2026-11-11)
    e.lineJetD = tok.ph == Ph::Z && kLineZJetD > 0 ? kLineZJetD : tok.ph == Ph::F ? (curStress_ ? kLineFJetD : kLineFJetDUnstr) : tok.ph == Ph::VV ? kLineVJetD : tok.ph == Ph::ZH && !inAffricate_ && kLineZhJetD >= 0 ? kLineZhJetD : post ? kLinePostJetD : 0.0f;
    if (post) e.lineJetMix = kLinePostJetMix, e.lineSlitScale = kLinePostSlitScale, e.lineJetSmoothMs = kLinePostStSmoothMs, e.lineJetFcMin = kLinePostFcMin, e.lineJetFcMax = kLinePostFcMax;
    if (zhOwn && kLineZhSlit >= 0) e.lineSlitScale = kLineZhSlit;
    if (zhOwn && kLineZhJetMix >= 0) e.lineJetMix = kLineZhJetMix;
    e.lineFricLpr = tok.ph == Ph::Z && kLineZLpr > 0 ? kLineZLpr : tok.ph == Ph::ZH && !inAffricate_ && kLineZhLpr >= 0 ? kLineZhLpr : tok.ph == Ph::SH || tok.ph == Ph::ZH ? kLinePostLpr : tok.ph == Ph::F ? kLineFLpr : tok.ph == Ph::VV ? kLineVLpr : 0.0f;  // (not ʧ ʤ: at 3 they lost 39 -> 34, 26 -> 23 of 54, 56)
    // The full-vowel ʃ's own groove (SH agent 10, round K; < 0: the shared ʃ ʒ ʧ ʤ values above): lip rounding
    // LINE_SH_V_LIP (cm^2), groove half-width LINE_SH_V_HW, jet thickness LINE_SH_V_JET_D, noise band LINE_SH_V_LPR,
    // slit-height scale LINE_SH_V_SLIT_SCALE, jet mix LINE_SH_V_JET_MIX; its jet's Reynolds number x LINE_SH_V_RE (1: as
    // before; an earlier, steadier hiss). Kokoro's male ʃ is flat 2.5-5 kHz (puhuja's peaks at 4-5 kHz, 2.5 kHz -7 dB).
    {
      static const float kShVLip = param("LINE_SH_V_LIP", -1.0f), kShVHw = param("LINE_SH_V_HW", -1.0f),
                         kShVJetD = param("LINE_SH_V_JET_D", -1.0f), kShVLpr = param("LINE_SH_V_LPR", -1.0f),
                         kShVSlit = param("LINE_SH_V_SLIT_SCALE", -1.0f), kShVJetMix = param("LINE_SH_V_JET_MIX", -1.0f),
                         kShVRe = param("LINE_SH_V_RE", 1.0f);
      if (shV) {
        if (kShVLip >= 0) e.lineLip = kShVLip;
        if (kShVHw >= 0) e.lineHw = kShVHw;
        if (kShVJetD >= 0) e.lineJetD = kShVJetD;
        if (kShVLpr >= 0) e.lineFricLpr = kShVLpr;
        if (kShVSlit >= 0) e.lineSlitScale = kShVSlit;
        if (kShVJetMix >= 0) e.lineJetMix = kShVJetMix;
      }
      if (tract_ == Tract::Line && kShVRe != 1.0f) {
        if (shV) e.lineFricReMul = kShVRe;
        else if (lastPh_ == Ph::SH && e.lineFricReMul < 0) e.lineFricReMul = 1.0f;
      }
    }
    // θ's constriction length (LINE_TH_HW, x the shared 1; TH agent7 round H: the tongue tip on the teeth is a short contact)
    // and its noise band (LINE_TH_LPR: low-pass at this x the jet's corner; 0: LINE_NOISE_LPR 10).
    { static const float kLineThHw = param("LINE_TH_HW", 1.0f), kLineThLpr = param("LINE_TH_LPR", 0.0f);
      if (tok.ph == Ph::TH && kLineThHw != 1.0f) e.lineHw = kLineThHw;
      if (tok.ph == Ph::TH && kLineThLpr > 0) e.lineFricLpr = kLineThLpr; }
    // A sound without frication leaves the jet as it was (LINE_JET_KEEP): the
    // n after ʃ snapped it back to the slit's height and the full band while
    // the slit was still hissing, a +10 dB spike at the end of every -tion.
    static const bool jetKeep = param("LINE_JET_KEEP", 1.0f) > 0;
    if (jetKeep && in.cls != Cls::Fricative) e.lineJetD = e.lineFricLpr = -1.0f;
    // f's jet around it (LINE_F_EDGE): the sound before an f takes f's jet
    // already and the one after keeps it. The lip closed under the vowel's
    // jet (the slit's height, a 12 kHz corner) and opened under the r's: a
    // 15 kHz click at the f's start and a +15 dB, 8-12 kHz flash at its end,
    // the f's whistle in "different" (2026-11-05).
    static const bool fEdge = param("LINE_F_EDGE", 1.0f) > 0;
    if (fEdge && in.cls != Cls::Fricative) {
      if (next && *next == Ph::F) e.lineJetD = kLineFJetD, e.lineFricLpr = kLineFLpr;
      else if (lastPh_ == Ph::F) e.lineJetD = e.lineFricLpr = -1.0f;
    }
    // θ ð: the jet between the tongue tip and the teeth is a wide flat one,
    // not the slit's height (0.03 cm^2 / 1 cm = 0.3 mm put the noise band's
    // corner near 12 kHz; Kokoro's θ is loudest at 3-5 kHz). Thickness in
    // cm, 0: the slit's height (2026-11-04).
    {
      static const float dentalJetD = param("LINE_DENTAL_JET_D", 0.0f);
      if (tok.ph == Ph::TH || tok.ph == Ph::DH) e.lineJetD = dentalJetD;
      // ð's own jet thickness (cm; < 0: LINE_DENTAL_JET_D) and noise band low-pass (0: the shared one); DH agent7, round H:
      // the ð's hiss centroid sat at 717 Hz (male, kennedy) against Kokoro's 372, its 2-8 kHz 8 dB over Kokoro's.
      static const float kLineDhJetD = param("LINE_DH_JET_D", -1.0f), kLineDhLpr = param("LINE_DH_LPR", 0.0f);
      if (tok.ph == Ph::DH && kLineDhJetD >= 0) e.lineJetD = kLineDhJetD;
      if (tok.ph == Ph::DH && kLineDhLpr > 0) e.lineFricLpr = kLineDhLpr;
      // ... and θ's own hiss gain (x LINE_FRIC; Kokoro's θ is 4-8 dB under its s, puhuja's equal to it; 2026-11-09)
      static const float kLineThFric = param("LINE_TH_FRIC", 0.3f);
      if (tok.ph == Ph::TH && kLineThFric != 1.0f && e.lineFricMul != 0) e.lineFricMul = (e.lineFricMul < 0 ? 1.0f : e.lineFricMul) * kLineThFric;
      // ... and ð's (LINE_DH_FRIC; 0: no hiss, an approximant-like ð. Its hiss reached -19 dB re the word peak at 5 kHz by its
      // end where Kokoro's ð keeps its 2-8 kHz 23-30 dB under; lab 2026-11-09 evening)
      static const float kLineDhFric = param("LINE_DH_FRIC", 1.0f);
      if (tok.ph == Ph::DH && kLineDhFric != 1.0f && e.lineFricMul != 0) e.lineFricMul = (e.lineFricMul < 0 ? 1.0f : e.lineFricMul) * kLineDhFric;
    }
    // A pause leaves the groove as it is: snapping ʃ's 2.5 cm groove to a
    // 1 cm one while the hiss still ran put a +14 dB blip at the end of
    // "abolish" (2026-11-03).
    if (tok.ph == Ph::Stop) e.lineHw = e.lineLip = -1.0f;
  }
  {
    const bool voicedStop = tok.ph == Ph::B || tok.ph == Ph::D || tok.ph == Ph::G;
    e.lineSoft = voicedStop ? kLineVoicedSoft : 1.0f;
    e.lineExtra = tok.ph == Ph::B ? kLineVoicedExtraCm : 0.0f;
    e.lineExpand = 0.0f;
  }
  if (tract_ == Tract::Line) {
    // The ʊ offglide of aʊ ends darker than a ʊ of its own (the references'
    // F2 ~850 Hz against ʊ's 1300): it takes o's shape. After o (oʊ) it keeps
    // ʊ's own, so the glide moves ("slow" was "sluu" with o held, Jyrki
    // 2026-11-02).
    // ... and before an l (holds, old, cold) the oʊ stays at o: the readers'
    // F2 falls into the dark l, never rises ("holds" was "hulds", 2026-11-02).
    const bool glideToL = next && *next == Ph::L;
    // The oʊ offglide takes u's shape: the American oʊ ends close, back and
    // rounded (Kokoro "know", "go": F1/F2 360-470/830-900 at 80 %).
    // The eɪ offglide ends close and front: Kokoro's "day", "make" reach
    // F2 2100-2400 Hz, ɪ's shape stopped at 1870 (2026-11-02).
    const Ph shapePh = tok.ph == Ph::UH && (lastPh_ == Ph::A || (lastPh_ == Ph::O && glideToL)) ? Ph::O
                       : tok.ph == Ph::UH && lastPh_ == Ph::O ? Ph::U
                       : tok.ph == Ph::IH && lastPh_ == Ph::E ? Ph::I : tok.ph;
    TractShape sh;
    bool have = lineShapeOf(shapePh, &sh);
    // f before a vowel: the tract is already in the vowel's shape; only the
    // lower lip makes the f against the upper teeth, and opens into the vowel
    // (Jyrki 2026-11-03: "in 'forget' the f is already in the 'o' space").
    // Without a key of its own the f glided between the vowels around it.
    if (!have && tok.ph == Ph::F && next && isVowel(*next)) have = lineShapeOf(*next, &sh);
    // ð lab knobs (DH agent7, round H; 0: off, as before): the ð gets a key of its own, the next vowel's shape (as the f)
    // with LINE_DH_FRONT of the ɪ's tongue mixed in, its demand x LINE_DH_SHAPE. Without a key the ð glided between its
    // neighbours and Whisper heard a labial (dictionary ð: b 49, d 32, v 20 of 162 errors, "Other" -> over / rubber): F2 fell
    // into the ð (kennedy, 20 ms before it: 1074 Hz, Kokoro 1378). Lab (140 ð words x 2 voices, 2 draws): SHAPE 1 alone 122
    // errors of 280 (base 88), FRONT 0.4 / 0.5 / 0.6 74 / 76 / 76, 1.0 96 (d t), seed 5: FRONT 0.5 78 vs 98; words1000 +6.
    // LINE_DH_FRONT_I 1: the iː's tongue instead of the ɪ's; LINE_DH_SHAPE_ANY 1: a ð with no vowel next ("smooth",
    // "rhythm", "bathe") takes the ə's shape (else none, as before).
    static const float kLineDhShape = param("LINE_DH_SHAPE", 0.0f), kLineDhFront = param("LINE_DH_FRONT", 0.0f),
                       kLineDhFrontI = param("LINE_DH_FRONT_I", 0.0f), kLineDhShapeAny = param("LINE_DH_SHAPE_ANY", 0.0f);
    if (!have && tok.ph == Ph::DH && kLineDhShape > 0 && ((next && isVowel(*next)) || kLineDhShapeAny > 0)) {
      have = lineShapeOf(next && isVowel(*next) ? *next : Ph::Schwa, &sh);
      TractShape ihDh;
      if (have && kLineDhFront > 0 && lineShapeOf(kLineDhFrontI > 0 ? Ph::I : Ph::IH, &ihDh)) sh = lineShapeMix(sh, ihDh, std::min(kLineDhFront, 1.0f));
    }
    // The h takes the next vowel's shape, as the cascade's formants do (hTakesVowel_): with no keys of its own the
    // line glided from the previous sound's last key to the vowel's first in one straight move, ~200 ms for
    // "mortal hands" (the l held to 70 % of its 175 ms, F2 1040 -> 1400 only by 70 ms into the æ); 11labs and
    // Kokoro leave the l 60 ms before the h and breathe the h on the vowel's shape. Jyrki 2026-11-09:
    // "mortallllhaaands", "puhuja makes fast transition between a and n [h and a], while 11labs makes a smooth natural transition".
    {
      static const float kLineHTakesVowel = param("LINE_H_TAKES_VOWEL", 1.0f);
      if (!have && tok.ph == Ph::H && kLineHTakesVowel > 0 && next && isVowel(*next)) have = lineShapeOf(*next, &sh);
      if (have && tok.ph == Ph::H) {  // (LINE_H_LIP, LINE_H_AC, LINE_H_OPEN_MIX)
        TractShape a;
        if (kLineHOpenMix > 0 && lineShapeOf(Ph::A, &a)) sh = lineShapeMix(sh, a, kLineHOpenMix);
        if (kLineHLip != 1.0f) sh.aLip *= kLineHLip;
        if (kLineHAc != 1.0f) sh.ac *= kLineHAc;
        if (kLineHAph != 1.0f) sh.aPharynx *= kLineHAph;
      }
    }
    // The m's own key (round J, M agent; LINE_M_SHAPE 0: none, as before: the m glided between its neighbours' keys).
    // m was heard as n 602 times in the dictionary (word-initial before ɑ 31 %: "Mom" -> nom, "Ma" -> nah; final 18 %).
    // With no key, an initial m sat in the pause's ə (F2 1250 Hz at the release) and the ɑ's F2 fell 1250 -> 900 over
    // 100 ms after it: an alveolar transition. Kokoro's m sits in the next vowel's shape (F2 at the release = the
    // vowel's, then level or rising). LINE_M_SHAPE 1: the next vowel's shape (else the previous vowel's, else ə's), mixed
    // LINE_M_SHAPE_MIX toward the row LINE_M_SHAPE_ROW (0 o, 1 uː, 2 ʊ, 3 ɔ, 4 ə; the tract behind the lips, as b / p
    // take o's); LINE_M_SHAPE_FINAL_MIX (< 0: the same) when no vowel follows. Lab (300 hard m words, 2 voices, two
    // pairs of draws): SHAPE 1, MIX 0.3, FINAL_MIX 0.7: m errors 302 -> 141 and 268 -> 132 (heard as n 196 -> 64),
    // words right +82 / +76 of 600, words1000 1406 (1414); "Kim" F2 1810 -> 946 Hz into the m (Kokoro 1970 -> 1100).
    {
      static const float kMShape = param("LINE_M_SHAPE", 1.0f), kMShapeMix = param("LINE_M_SHAPE_MIX", 0.3f),
                         kMShapeFinalMix = param("LINE_M_SHAPE_FINAL_MIX", 0.7f);
      static const int kMShapeRow = static_cast<int>(param("LINE_M_SHAPE_ROW", 0.0f));
      if (!have && tok.ph == Ph::M && kMShape > 0) {
        const bool vNext = next && isVowel(*next);
        const Ph nb = vNext ? *next : isVowel(lastPh_) ? lastPh_ : Ph::Schwa;
        have = lineShapeOf(nb, &sh);
        static const Ph kRows[5] = {Ph::O, Ph::U, Ph::UH, Ph::AO, Ph::Schwa};
        const float mix = !vNext && kMShapeFinalMix >= 0 ? kMShapeFinalMix : kMShapeMix;
        TractShape r;
        if (have && mix > 0 && lineShapeOf(kRows[std::clamp(kMShapeRow, 0, 4)], &r)) sh = lineShapeMix(sh, r, std::min(mix, 1.0f));
        // ... its lips x LINE_M_LIPL long (protruded: every formant lower in the transitions; 1: the row's)
        static const float kMLipL = param("LINE_M_LIPL", 1.0f);
        if (have && kMLipL != 1.0f) sh.lLip *= kMLipL;
      }
    }
    // (only a word-initial r, after a pause or a vowel: in a cluster, "different",
    // the shape's pharyngeal narrowing glided in from the f as a chirp, F1
    // 1500 -> 470 Hz over 80 ms; Jyrki: "a wild spectral event, a swirp")
    if (have && tok.ph == Ph::RR && next && isVowel(*next) && (lastPh_ == Ph::Stop || isVowel(lastPh_))) {
      // An onset r is rounded and retracted: Kokoro's "read", "right", "run"
      // start at F2 815-930 Hz with F3 1440-1530; the r-coloured vowel's
      // shape alone gave F2 1320-1350 (mixing u in pulled F3 to 1700). Its
      // own shape, fitted to 350/880/1480, sits in the ɒ slot (2026-11-02).
      TractShape r;
      if (lineShapeOf(Ph::OH, &r)) {
        // Between vowels ("very") the readers keep F2 at 1370-1640 through the
        // r (goog, 11labs, Kokoro); the onset shape's 880 made it a w, "vewy"
        // (Jyrki 2026-11-05). LINE_R_MEDIAL: the onset shape's share after a
        // vowel (1 = all of it, 0 = the r-coloured vowel's shape).
        static const float kRMedial = param("LINE_R_MEDIAL", 0.0f);  // Jyrki 2026-11-05: liked the ɜ shape (W1, W2) for "very"
        sh = isVowel(lastPh_) && kRMedial < 1.0f ? lineShapeMix(sh, r, kRMedial) : r;
        // (R agent 2026-11-10, lab) the onset shape after a pause or stop mixed toward the ɜ row by this share (0: none)
        static const float kROnsetEr = param("LINE_R_ONSET_ER", 0.0f);
        if (kROnsetEr > 0 && !isVowel(lastPh_)) { TractShape er; if (lineShapeOf(Ph::ER, &er)) sh = lineShapeMix(sh, er, kROnsetEr); }
      }
    }
    if (have && tok.ph == Ph::RR) {
      // Lab knobs (R agent 2026-11-10; defaults reproduce the previous sound): the ɹ after a plosive takes this share of the
      // onset (ɒ-slot) shape over the ɜ row (-1: as before, by lastPh_); its lip area / length, pharynx x these; hump half-width (-1: the row's).
      static const float kRClusterOh = param("LINE_R_CLUSTER_OH", -1.0f), kRLipA = param("LINE_R_LIPA", 1.0f), kRLipL = param("LINE_R_LIPL", 1.0f),
                         kRAph = param("LINE_R_APH", 0.5f), kRWc = param("LINE_R_WC", -1.0f);
      if (kRClusterOh >= 0 && prevWasPlosive_ && next && isVowel(*next)) {
        TractShape er, oh;
        if (lineShapeOf(Ph::ER, &er) && lineShapeOf(Ph::OH, &oh)) sh = lineShapeMix(er, oh, kRClusterOh);
      }
      if (kRLipA != 1.0f) sh.aLip *= kRLipA;
      if (kRLipL != 1.0f) sh.lLip *= kRLipL;
      if (kRAph != 1.0f) sh.aPharynx *= kRAph;
      if (kRWc > 0) { sh.wc = kRWc; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
      // (R agent round F, lab) the ɹ after a plosive ("bring", "tree": 1644 of the 2665 ɹ errors, F3 1800-2000 where
      // Kokoro's holds 1490-1550): its pharynx, lip area, oral (front, sublingual) area x these; 1: as before.
      static const float kRStopAph = param("LINE_R_STOP_APH", 1.0f), kRStopLipA = param("LINE_R_STOP_LIPA", 1.0f),
                         kRStopAor = param("LINE_R_STOP_AOR", 1.0f);
      if (prevWasPlosive_) {
        if (kRStopAph != 1.0f) sh.aPharynx *= kRStopAph;
        if (kRStopLipA != 1.0f) sh.aLip *= kRStopLipA;
        if (kRStopAor != 1.0f) sh.aOral *= kRStopAor;
      }
      // (R agent 2026-11-10) the female onset ɹ sat at F2 ~1400 (Kokoro's female 900-1050): female-only pharynx / lip area x these,
      // constriction place set to this (-1: the row's); only before a vowel and not after one (onset and cluster ɹ)
      static const float kRFAph = param("LINE_R_F_APH", 1.0f), kRFLipA = param("LINE_R_F_LIPA", 1.0f), kRFX = param("LINE_R_F_X", -1.0f);
      if (female_ && next && isVowel(*next) && !isVowel(lastPh_)) {
        if (kRFAph != 1.0f) sh.aPharynx *= kRFAph;
        if (kRFLipA != 1.0f) sh.aLip *= kRFLipA;
        if (kRFX > 0) { sh.xc = kRFX; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
      }
    }
    // (Y agent round G, lab; 1: as before) the j after a plosive: its pharynx, lip area, oral area and hump x these.
    if (have && tok.ph == Ph::J && prevWasPlosive_) {
      static const float kJStopAph = param("LINE_J_STOP_APH", 1.0f), kJStopLipA = param("LINE_J_STOP_LIPA", 1.0f),
                         kJStopAor = param("LINE_J_STOP_AOR", 1.0f), kJStopAc = param("LINE_J_STOP_AC", 1.0f);
      if (kJStopAph != 1.0f) sh.aPharynx *= kJStopAph;
      if (kJStopLipA != 1.0f) sh.aLip *= kJStopLipA;
      if (kJStopAor != 1.0f) sh.aOral *= kJStopAor;
      if (kJStopAc != 1.0f) sh.ac *= kJStopAc;
    }
    // (W agent round G, lab; 1 / -1: as before) the w after a plosive ("twice", "quick": w after p t k 21 % errors, heard
    // as l ɹ: "twice" -> "place", "quite" -> "pride") or after s (LINE_W_S_*: "swim" -> "slim"): its lip area, lip tube
    // length, velar hump area, hump place (< 0: the row's), pharynx x these.
    if (have && tok.ph == Ph::W && (prevWasPlosive_ || lastPh_ == Ph::S)) {
      const bool afterS = !prevWasPlosive_;
      static const float kWStopLipA = param("LINE_W_STOP_LIPA", 1.0f), kWStopLipL = param("LINE_W_STOP_LIPL", 1.0f),
                         kWStopAc = param("LINE_W_STOP_AC", 0.3f), kWStopXc = param("LINE_W_STOP_XC", -1.0f),
                         kWStopAph = param("LINE_W_STOP_APH", 1.5f), kWSLipA = param("LINE_W_S_LIPA", -1.0f);
      const float lipA = afterS && kWSLipA >= 0 ? kWSLipA : kWStopLipA;
      if (lipA != 1.0f) sh.aLip *= lipA;
      if (kWStopLipL != 1.0f) sh.lLip *= kWStopLipL;
      if (kWStopAc != 1.0f) sh.ac *= kWStopAc;
      if (kWStopAph != 1.0f) sh.aPharynx *= kWStopAph;
      if (kWStopXc >= 0) { sh.xc = kWStopXc; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
    }
    // (W agent round I, lab; 1 / -1 / 0: as before) every other w (after a pause, a vowel, l, n ...: word-initial w 24 % errors,
    // heard as l or ɹ, "wine" -> "line", "went" -> "rent"): its hump area, hump place (< 0: the row's), pharynx, lip area, lip
    // length x these (LINE_W_GEN_INIT 1: only a w after a pause); the female voice's hump area x LINE_W_F_AC, its pharynx x
    // LINE_W_F_APH, its lip length x LINE_W_F_LIPL on top (the female w 24 % vs male 14 %).
    if (have && tok.ph == Ph::W && !prevWasPlosive_ && lastPh_ != Ph::S) {
      static const float kWGenAc = param("LINE_W_GEN_AC", 1.0f), kWGenXc = param("LINE_W_GEN_XC", -1.0f),
                         kWGenAph = param("LINE_W_GEN_APH", 1.0f), kWGenLipA = param("LINE_W_GEN_LIPA", 1.0f),
                         kWGenLipL = param("LINE_W_GEN_LIPL", 1.0f), kWGenInit = param("LINE_W_GEN_INIT", 0.0f),
                         kWFAc = param("LINE_W_F_AC", 1.0f), kWFAph = param("LINE_W_F_APH", 1.0f),
                         kWFLipL = param("LINE_W_F_LIPL", 1.0f);
      if (kWGenInit == 0.0f || prevWasPause_) {
        if (kWGenAc != 1.0f) sh.ac *= kWGenAc;
        if (kWGenAph != 1.0f) sh.aPharynx *= kWGenAph;
        if (kWGenLipA != 1.0f) sh.aLip *= kWGenLipA;
        if (kWGenLipL != 1.0f) sh.lLip *= kWGenLipL;
        if (kWGenXc >= 0) { sh.xc = kWGenXc; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
      }
      if (female_ && kWFAc != 1.0f) sh.ac *= kWFAc;
      if (female_ && kWFAph != 1.0f) sh.aPharynx *= kWFAph;
      if (female_ && kWFLipL != 1.0f) sh.lLip *= kWFLipL;
    }
    // ɜ's own shape (ER agent 2026-11-11, lab knobs; 1 / 0: as before): its pharynx, constriction area, lip area and
    // length x these, mixed toward ə by LINE_ER_SCHWA_MIX; the stressed (long) ɜː's pharynx x LINE_ER_LONG_APH, the
    // female voice's pharynx and lips x LINE_ER_F_APH / LINE_ER_F_LIPA. Only the ɜ token: the r colour of the vowels
    // before a coda r and the ɹ's onset shapes still mix in the row itself. Whisper heard puhuja's ɜ as u ("shirt"
    // -> "fruit", "term" -> "room", "were" -> "wu"; ER -> UW 234 of 1312 errors): its F1 sat 4.5 (male) and 6.0
    // (female) semitones under ten Kokoro voices' on the ɜ frames, 1.5-2 more than the other vowels' offset.
    if (have && tok.ph == Ph::ER) {
      static const float kErAph = param("LINE_ER_APH", 0.6f), kErAc = param("LINE_ER_AC", 1.0f), kErLipA = param("LINE_ER_LIPA", 1.0f),
                         kErLipL = param("LINE_ER_LIPL", 1.0f), kErSchwaMix = param("LINE_ER_SCHWA_MIX", 0.0f),
                         kErLongAph = param("LINE_ER_LONG_APH", 1.0f), kErFAph = param("LINE_ER_F_APH", 1.0f),
                         kErFLipA = param("LINE_ER_F_LIPA", 1.0f), kErXc = param("LINE_ER_XC", 1.0f), kErAor = param("LINE_ER_AOR", 1.0f);
      TractShape s8;
      if (kErSchwaMix > 0 && lineShapeOf(Ph::Schwa, &s8)) sh = lineShapeMix(sh, s8, kErSchwaMix);
      if (kErAph != 1.0f) sh.aPharynx *= kErAph;
      if (tok.isLong && kErLongAph != 1.0f) sh.aPharynx *= kErLongAph;
      if (female_ && kErFAph != 1.0f) sh.aPharynx *= kErFAph;
      if (kErAc != 1.0f) sh.ac *= kErAc;
      if (kErLipA != 1.0f) sh.aLip *= kErLipA;
      if (female_ && kErFLipA != 1.0f) sh.aLip *= kErFLipA;
      if (kErLipL != 1.0f) sh.lLip *= kErLipL;
      if (kErAor != 1.0f) sh.aOral *= kErAor;
      if (kErXc != 1.0f) { sh.xc *= kErXc; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
    }
    // ə's own shape by position (AX agent 2026-11-11, lab knobs; 0 / 1: as before). Whisper heard the word-final ə as
    // u or ɝ ("China", "data": 70 % errors in the final cells against 16 % medial): on ten Kokoro voices' final ə
    // frames F1 sat 8.1 (male, 325 vs 516 Hz) and 12.6 (female, 358 vs 849) semitones under theirs, F3 2 under;
    // the American final ə is an open ʌ / ɐ. Before a pause: mixed toward LINE_SCHWA_FINAL_ROW (0: ʌ, 1: ɑ, 2: a,
    // 3: æ) by LINE_SCHWA_FINAL_MIX, pharynx / lips x _APH / _LIPA; elsewhere (medial, initial) by LINE_SCHWA_MIX
    // toward the same row, and a word-initial ə's pharynx x LINE_SCHWA_INIT_APH.
    if (have && tok.ph == Ph::Schwa) {
      static const float kSchwaFinalMix = param("LINE_SCHWA_FINAL_MIX", 0.75f), kSchwaMix = param("LINE_SCHWA_MIX", 0.0f),
                         kSchwaFinalAph = param("LINE_SCHWA_FINAL_APH", 1.0f), kSchwaFinalLipA = param("LINE_SCHWA_FINAL_LIPA", 1.0f),
                         kSchwaInitAph = param("LINE_SCHWA_INIT_APH", 1.0f), kSchwaLen = param("LINE_SCHWA_FINAL_LEN", 1.0f);
      static const int kSchwaRow = static_cast<int>(param("LINE_SCHWA_FINAL_ROW", 0.0f));
      const bool final = !next || *next == Ph::Stop;
      const float mix = final ? kSchwaFinalMix : kSchwaMix;
      TractShape o;
      const Ph row = kSchwaRow == 1 ? Ph::AA : kSchwaRow == 2 ? Ph::A : kSchwaRow == 3 ? Ph::AE : Ph::AH;
      if (mix > 0 && lineShapeOf(row, &o)) sh = lineShapeMix(sh, o, mix);
      if (final) {
        if (kSchwaFinalAph != 1.0f) sh.aPharynx *= kSchwaFinalAph;
        if (kSchwaFinalLipA != 1.0f) sh.aLip *= kSchwaFinalLipA;
        static const float kSchwaFinalFAph = param("LINE_SCHWA_FINAL_F_APH", 1.0f);  // ... the female voice's pharynx x this
        if (female_ && kSchwaFinalFAph != 1.0f) sh.aPharynx *= kSchwaFinalFAph;
        if (kSchwaLen != 1.0f) { sh.length *= kSchwaLen; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
      }
      if (tok.wordStart && kSchwaInitAph != 1.0f) sh.aPharynx *= kSchwaInitAph;
      // (round-J ə agent 9, lab; 0 / 1: as before) the ə by context. Dictionary (combo7, 28864 ə): word-initial 19 %
      // wrong (heard ɪ 108, deleted 106: "aside" -> decide, "abide" -> divide, "allowed" -> loud), the last syllable
      // of a two-syllable word 11.4 % (ɪ 298: "Nixon" -> mixing, "Gibson" -> gets in), word-final 15 % (ɝ 160),
      // medial in longer words 5 %. Ten Kokoro voices (k1k words, DTW frames, male): initial 349/1432/2225 Hz vs
      // 457/1284/2563 (F1 -3.2, F2 +1.3, F3 -2.4 st: closer and fronter, an ɪ), last-syllable 334/1448/2242 vs
      // 392/1496/2534 (F1 -2.8, F3 -1.2 st). A word-initial ə toward LINE_SCHWA_INIT_ROW (0: ʌ, 1: ɑ, 2: a) by
      // LINE_SCHWA_INIT_MIX, the female voice's x LINE_SCHWA_F_INIT_MIX more; an ə inside a word (not its first or
      // last sound, so not "the" / "a": LINE_SCHWA_MIX made "the generosity"'s ʤ snap) by LINE_SCHWA_IN_MIX, before a
      // nasal by LINE_SCHWA_N_MIX instead (>= 0), after a vowel ("media", "Ian": heard j / uː) by LINE_SCHWA_HIATUS_MIX;
      // a final ə after ɹ by LINE_SCHWA_FINAL_R_MIX more (heard ɝ, 22 %); every ə's lip area / lip length / tract
      // length / constriction area x LINE_SCHWA_LIPA / _LIPL / _SHLEN / _AC (F3 sits 1-2.4 st low everywhere).
      static const float kSwInitMix = param("LINE_SCHWA_INIT_MIX", 0.5f), kSwFInitMix = param("LINE_SCHWA_F_INIT_MIX", 0.0f),
                         kSwInMix = param("LINE_SCHWA_IN_MIX", 0.3f), kSwNMix = param("LINE_SCHWA_N_MIX", -1.0f),
                         kSwHiatusMix = param("LINE_SCHWA_HIATUS_MIX", 0.0f), kSwFinalRMix = param("LINE_SCHWA_FINAL_R_MIX", 0.0f),
                         kSwLipA = param("LINE_SCHWA_LIPA", 1.0f), kSwLipL = param("LINE_SCHWA_LIPL", 1.0f),
                         kSwShLen = param("LINE_SCHWA_SHLEN", 1.0f), kSwAc = param("LINE_SCHWA_AC", 1.0f);
      static const int kSwInitRow = static_cast<int>(param("LINE_SCHWA_INIT_ROW", 0.0f));
      const bool inWord = !tok.wordStart && !final && next && !nextWordStart_ && *next != Ph::JH && *next != Ph::CH;
      const bool nasNext = next && info(*next).cls == Cls::Nasal;
      float cmix = tok.wordStart && !final ? kSwInitMix + (female_ ? kSwFInitMix : 0.0f)
                   : inWord ? (isVowel(lastPh_) && kSwHiatusMix > 0 ? kSwHiatusMix : nasNext && kSwNMix >= 0 ? kSwNMix : kSwInMix)
                   : 0.0f;
      const Ph crow = tok.wordStart ? (kSwInitRow == 1 ? Ph::AA : kSwInitRow == 2 ? Ph::A : Ph::AH) : Ph::AH;
      if (final && lastPh_ == Ph::RR) cmix = kSwFinalRMix;
      if (cmix > 0 && lineShapeOf(crow, &o)) sh = lineShapeMix(sh, o, std::min(cmix, 1.0f));
      if (kSwLipA != 1.0f) sh.aLip *= kSwLipA;
      if (kSwLipL != 1.0f) sh.lLip *= kSwLipL;
      if (kSwAc != 1.0f) sh.ac *= kSwAc;
      if (kSwShLen != 1.0f) { sh.length *= kSwShLen; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
    }
    // ɔ's own shape (AO agent 2026-11-11, lab knobs; 0 / 1: as before): mixed toward ɑ by LINE_AO_AA_MIX, its pharynx,
    // constriction place and area, oral cavity, lip area and length x these. Only the ɔ token (not ɔɪ's start, and before
    // an ɹ only with LINE_AO_R 1): the ɔ row also shapes aʊ's end, ɔɪ's start, the l and the p. Whisper heard the
    // dictionary's ɔ as oʊ (219 of 848 errors: "Paul" -> "pole", "author" -> "over", "sort" -> "sword") and ɑ (116):
    // male 479/920 Hz against Kokoro's 520/1035 (adam, onyx: "law, talk, dog, all"), F1 -1.4 and F2 -2 semitones.
    if (have && tok.ph == Ph::AO && !(next && *next == Ph::IH)) {
      static const float kAoAaMix = param("LINE_AO_AA_MIX", 0.25f), kAoAph = param("LINE_AO_APH", 1.0f),
                         kAoXc = param("LINE_AO_XC", 1.0f), kAoAc = param("LINE_AO_AC", 1.0f), kAoAor = param("LINE_AO_AOR", 1.0f),
                         kAoLipA = param("LINE_AO_LIPA", 1.0f), kAoLipL = param("LINE_AO_LIPL", 1.0f), kAoR = param("LINE_AO_R", 1.0f),
                         kAoFLipA = param("LINE_AO_F_LIPA", 1.0f);
      // (AO agent 8, round I, lab; 15 / 1: as before) LINE_AO_CTX: which ɔ these knobs shape, by what follows in the
      // stream: 1 an l, 2 a voiceless obstruent (p t k ʧ f θ s ʃ), 4 a pause (word-final before silence), 8 anything
      // else (the dictionary: 30 % / 31-67 % / 60 % / 15-41 % wrong). LINE_AO_SHLEN: the tract length x (< 1 raises every
      // formant together: Kokoro's ɔ sits +4 st F1, +2 st F2 above puhuja's).
      static const int kAoCtx = param("LINE_AO_CTX", 15);
      static const float kAoShLen = param("LINE_AO_SHLEN", 1.0f);
      const int aoCtx = next && *next == Ph::L ? 1 :
                        next && (*next == Ph::P || *next == Ph::T || *next == Ph::K || *next == Ph::CH || *next == Ph::F ||
                                 *next == Ph::TH || *next == Ph::S || *next == Ph::SH) ? 2 :
                        !next || *next == Ph::Stop ? 4 : 8;
      if ((kAoR > 0 || !(next && *next == Ph::RR)) && (kAoCtx & aoCtx)) {
        if (kAoShLen != 1.0f) sh.length *= kAoShLen;
        // (AO agent 8; 0 / 1: as before) the female voice's own ɑ mix (her F2 sits -4.2 st under Kokoro's, his -2.2) and the
        // constriction's half-width x LINE_AO_WC.
        static const float kAoFAaMix = param("LINE_AO_F_AA_MIX", 0.0f), kAoWc = param("LINE_AO_WC", 1.0f);
        TractShape fa;
        if (female_ && kAoFAaMix > 0 && lineShapeOf(Ph::AA, &fa)) sh = lineShapeMix(sh, fa, kAoFAaMix);
        if (kAoWc != 1.0f) { sh.wc *= kAoWc; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
        TractShape a;
        if (kAoAaMix > 0 && lineShapeOf(Ph::AA, &a)) sh = lineShapeMix(sh, a, kAoAaMix);
        if (kAoAph != 1.0f) sh.aPharynx *= kAoAph;
        if (kAoAc != 1.0f) sh.ac *= kAoAc;
        if (kAoAor != 1.0f) sh.aOral *= kAoAor;
        if (kAoLipA != 1.0f) sh.aLip *= kAoLipA;
        if (female_ && kAoFLipA != 1.0f) sh.aLip *= kAoFLipA;
        if (kAoLipL != 1.0f) sh.lLip *= kAoLipL;
        if (kAoXc != 1.0f) { sh.xc *= kAoXc; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
      }
    }
    // (AO agent 8, round I, lab; 0: off) an l right after an ɔ mixed toward ə by LINE_L_AFTER_AO: puhuja's coda l after
    // ɔ ran ~320 / 740 Hz, the vowel gliding down into it as an oʊ does ("call" -> "cole", "Paul" -> "pole", "mall" ->
    // "mole": 72 of the 128 ɔ errors before l are oʊ); Kokoro's l there sits at 350-430 / 950-1050.
    if (have && tok.ph == Ph::L && lastPh_ == Ph::AO) {
      static const float kLAfterAo = param("LINE_L_AFTER_AO", 0.0f);
      TractShape c;
      if (kLAfterAo > 0 && lineShapeOf(Ph::Schwa, &c)) sh = lineShapeMix(sh, c, kLAfterAo);
    }
    if (have && tok.ph == Ph::O && next && *next == Ph::UH) {
      // ... and starts open-mid and central (Kokoro 530-600/1060-1290 at 20 %;
      // puhuja's o held 450/1070 and drifted up: "know" was heard as "no-ʉ").
      TractShape mid;
      if (lineShapeOf(Ph::AH, &mid)) sh = lineShapeMix(sh, mid, kLineOwStartAh);
      owStartShape(&sh);
    }
    if (have && tok.ph == Ph::UH && lastPh_ == Ph::O && shapePh == Ph::U) owEndShape(&sh);
    // Before an l the ʊ keeps o's shape (above), which held F1 at ~600 Hz (the ə-mixed start) into the l: "whole" ->
    // "hull", "goal" -> "trawl" (Kokoro's F1 falls 500 -> 360-420, F2 to ~900). LINE_OW_L_U: this share of the ʊ end
    // (u's row through owEndShape) mixed in (0: as before).
    static const float kLineOwLU = param("LINE_OW_L_U", 0.6f);
    if (have && tok.ph == Ph::UH && lastPh_ == Ph::O && shapePh == Ph::O && kLineOwLU > 0) {
      TractShape u;
      if (lineShapeOf(Ph::U, &u)) { owEndShape(&u); sh = lineShapeMix(sh, u, kLineOwLU); }
    }
    if (have && tok.ph == Ph::UH && lastPh_ == Ph::A && shapePh == Ph::O && kLineAwOffU > 0) {
      TractShape u;  // aʊ's end: toward u (see kLineAwOffU)
      if (lineShapeOf(Ph::U, &u)) sh = lineShapeMix(sh, u, kLineAwOffU);
    }
    if (have && tok.ph == Ph::UH && lastPh_ == Ph::A && shapePh == Ph::O && kLineAwEndAo > 0) {
      TractShape ao;  // aʊ's end toward ɔ: open and back (see kLineAwEndAo)
      if (lineShapeOf(Ph::AO, &ao)) sh = lineShapeMix(sh, ao, kLineAwEndAo);
    }
    if (have && tok.ph == Ph::UH && lastPh_ == Ph::A && shapePh == Ph::O && kLineAwEndA > 0) {
      TractShape a;  // ... or stops short of it, part way back to the a (LINE_AW_END_A)
      if (lineShapeOf(Ph::A, &a)) sh = lineShapeMix(sh, a, kLineAwEndA);
    }
    if (have && tok.ph == Ph::A && next && *next == Ph::UH && kLineAwStartAe > 0) {
      TractShape ae;  // aʊ's start toward æ (see kLineAwStartAe)
      if (lineShapeOf(Ph::AE, &ae)) sh = lineShapeMix(sh, ae, kLineAwStartAe);
    }
    if (have && tok.ph == Ph::A && next && *next == Ph::UH && kLineAwStartE > 0) {
      TractShape e;  // aʊ's start toward ɛ (see kLineAwStartE)
      if (lineShapeOf(Ph::E, &e)) sh = lineShapeMix(sh, e, kLineAwStartE);
    }
    if (have && tok.ph == Ph::AO && next && *next == Ph::IH) {  // ɔɪ's start (see kLineOyLipA)
      TractShape o;
      if (kLineOyStartO > 0 && lineShapeOf(Ph::O, &o)) sh = lineShapeMix(sh, o, kLineOyStartO);
      if (kLineOyStartAa > 0 && lineShapeOf(Ph::AA, &o)) sh = lineShapeMix(sh, o, kLineOyStartAa);
      sh.aLip *= kLineOyLipA;
      sh.lLip *= kLineOyLipL;
      sh.aPharynx *= kLineOyAph;
      if (kLineOyXc != 1.0f || kLineOyAc != 1.0f) {
        sh.xc *= kLineOyXc;
        sh.ac *= kLineOyAc;
        sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length);
      }
    }
    if (have && tok.ph == Ph::IH && lastPh_ == Ph::AO) {  // ɔɪ's end
      TractShape o;
      if (kLineOyEndI > 0 && lineShapeOf(Ph::I, &o)) sh = lineShapeMix(sh, o, kLineOyEndI);
      if (kLineOyEndE > 0 && lineShapeOf(Ph::E, &o)) sh = lineShapeMix(sh, o, kLineOyEndE);
      sh.aOral *= kLineOyIhAor;
      sh.aLip *= kLineOyIhLipA;
    }
    if (have && tok.ph == Ph::A && next && *next == Ph::IH) ayStartShape(&sh);
    if (have && tok.ph == Ph::EH) {
      // The ɛ's own shape on its row (round F EH agent 2026-11-11 lab knobs; 0 / 0 / 1 / 0: the row as before): mixed
      // toward e (LINE_EH_MIX_E) or ɪ (LINE_EH_MIX_IH), its opening x LINE_EH_AC, and before a nasal mixed toward e by
      // LINE_EH_NASAL_MIX_E more. Whisper heard ɛ as æ 784 times (352 before a nasal: "end" -> and); ɪ moved 0.3 toward
      // e since (LINE_IH_MIX_E), which left room above ɛ.
      static const float kEhE = param("LINE_EH_MIX_E", 0.0f), kEhIh = param("LINE_EH_MIX_IH", 0.0f),
                         kEhAc = param("LINE_EH_AC", 1.0f), kEhNasE = param("LINE_EH_NASAL_MIX_E", 0.0f);
      const float mixE = kEhE + (next && info(*next).cls == Cls::Nasal ? kEhNasE : 0.0f);
      TractShape o;
      if (mixE > 0 && lineShapeOf(Ph::E, &o)) sh = lineShapeMix(sh, o, mixE);
      if (kEhIh > 0 && lineShapeOf(Ph::IH, &o)) sh = lineShapeMix(sh, o, kEhIh);
      if (kEhAc != 1.0f) sh.ac *= kEhAc;
      // (round I EH agent 2026-11-12 lab knobs; 0: off) mixed toward ʌ (LINE_EH_MIX_AH) or ə (LINE_EH_MIX_AX), before an
      // l toward ʌ by LINE_EH_L_MIX_AH more: against ten Kokoro voices the male ɛ's middle third sits at F2 +200..+250 Hz
      // and F3 -150..-250 Hz (one-syllable words 1885-1906 / 2410 vs 1660-1680 / 2600); before l ("sells" -> sales) +230.
      static const float kEhAh = param("LINE_EH_MIX_AH", 0.0f), kEhAx = param("LINE_EH_MIX_AX", 0.0f),
                         kEhLAh = param("LINE_EH_L_MIX_AH", 0.0f);
      const float mixAh = kEhAh + (next && *next == Ph::L ? kEhLAh : 0.0f);
      if (mixAh > 0 && lineShapeOf(Ph::AH, &o)) sh = lineShapeMix(sh, o, mixAh);
      if (kEhAx > 0 && lineShapeOf(Ph::Schwa, &o)) sh = lineShapeMix(sh, o, kEhAx);
    }
    if (have && next && *next == Ph::NG && (tok.ph == Ph::IH || tok.ph == Ph::EH || tok.ph == Ph::AE)) {
      // A front vowel before ŋ mixed toward i over its whole length (LINE_NG_PRE_I_MIX, lab knob 2026-11-11, NG agent 3;
      // 0: off): English raises it there ("sing" ~ [siŋ]).
      static const float kLineNgPreIMix = param("LINE_NG_PRE_I_MIX", 0.0f);
      TractShape o;
      if (kLineNgPreIMix > 0 && lineShapeOf(Ph::I, &o)) sh = lineShapeMix(sh, o, kLineNgPreIMix);
    }
    if (have && tok.ph == Ph::IH && lastPh_ == Ph::A) ayEndShape(&sh);
    if (have && tok.ph == Ph::AA && next && *next == Ph::RR) {
      // The ɑ before an r mixed toward ɔ over its whole length (LINE_AA_R_AO_MIX; 2026-11-10 AA agent: "part, card, start").
      static const float kLineAaRAoMix = param("LINE_AA_R_AO_MIX", 0.0f);
      TractShape o;
      if (kLineAaRAoMix > 0 && lineShapeOf(Ph::AO, &o)) sh = lineShapeMix(sh, o, kLineAaRAoMix);
    }
    if (have && tok.ph == Ph::AA) {
      // The ɑ mixed toward ɔ (LINE_AA_AO_MIX; 2026-11-10 AA agent: Kokoro's ɑ "lot" and ɔ "law" measure the same, F1/F2
      // 550/1160 (am_adam), 526/1045 (am_onyx); puhuja's ɑ 669/1296 sits on its ʌ label 551/1356).
      static const float kLineAaAoMix = param("LINE_AA_AO_MIX", 0.0f);
      TractShape o;
      if (kLineAaAoMix > 0 && lineShapeOf(Ph::AO, &o)) sh = lineShapeMix(sh, o, kLineAaAoMix);
      // (AA agent 10, round K, lab knobs; 1 / 0 / -1: off) the ɑ token's own shape, leaving row 14 alone (ɔ, ʌ, the ɔɪ
      // start and the schwa variants mix toward the row). Since tonight's ɔ fix (LINE_AO_AA_MIX 0.25) puhuja's ɑ and ɔ
      // differ in F1 only: male middle third 620/930 Hz vs 525/960 ("Lot" vs "Law"); Kokoro's ɑ sits at F2 1000-1200
      // (am_onyx / am_adam) and its F2 rises over the last third in open syllables ("Pa" 1215 -> 1257, "La" 1201 ->
      // 1303). Dictionary: ɑ heard as ɔ 221 of 677 errors. Lips (area x LINE_AA_ALIP, length x LINE_AA_LLIP), mouth
      // cavity x LINE_AA_AOR, pharynx x LINE_AA_APH, constriction area x LINE_AA_AC, place LINE_AA_XC (-1: the row's),
      // tract cm added LINE_AA_LEN_CM; whole mixes toward æ (LINE_AA_MIX_AE), ʌ (LINE_AA_MIX_AH), the open front a
      // (LINE_AA_MIX_A); by context toward row LINE_AA_CTX_ROW: before a pause (LINE_AA_FINAL_MIX), before a nasal
      // (LINE_AA_NAS_MIX), before an l (LINE_AA_L_MIX), after a pause (LINE_AA_INIT_MIX); the female's lips x LINE_AA_F_ALIP.
      static const float kAaALip = param("LINE_AA_ALIP", 1.0f), kAaLLip = param("LINE_AA_LLIP", 1.0f),
                         kAaAor = param("LINE_AA_AOR", 1.0f), kAaAph = param("LINE_AA_APH", 1.0f),
                         kAaAc = param("LINE_AA_AC", 1.0f), kAaXc = param("LINE_AA_XC", -1.0f),
                         kAaLenCm = param("LINE_AA_LEN_CM", 0.0f), kAaFALip = param("LINE_AA_F_ALIP", 1.0f);
      static const float kAaMixAe = param("LINE_AA_MIX_AE", 0.0f), kAaMixAh = param("LINE_AA_MIX_AH", 0.0f),
                         kAaMixA = param("LINE_AA_MIX_A", 0.2f);
      static const float kAaFinalMix = param("LINE_AA_FINAL_MIX", 0.0f), kAaNasMix = param("LINE_AA_NAS_MIX", 0.0f),
                         kAaLMix = param("LINE_AA_L_MIX", 0.0f), kAaInitMix = param("LINE_AA_INIT_MIX", 0.0f);
      static const int kAaCtxRow = param("LINE_AA_CTX_ROW", static_cast<int>(Ph::A));
      if (kAaMixAe > 0 && lineShapeOf(Ph::AE, &o)) sh = lineShapeMix(sh, o, kAaMixAe);
      if (kAaMixAh > 0 && lineShapeOf(Ph::AH, &o)) sh = lineShapeMix(sh, o, kAaMixAh);
      if (kAaMixA > 0 && lineShapeOf(Ph::A, &o)) sh = lineShapeMix(sh, o, kAaMixA);
      const bool aaFinal = !next || *next == Ph::Stop;
      const bool aaNas = next && (*next == Ph::M || *next == Ph::N || *next == Ph::NG);
      const float aaCtx = (aaFinal ? kAaFinalMix : 0.0f) + (aaNas ? kAaNasMix : 0.0f) +
                          (next && *next == Ph::L ? kAaLMix : 0.0f) + (prevWasPause_ ? kAaInitMix : 0.0f);
      if (aaCtx > 0 && lineShapeOf(static_cast<Ph>(kAaCtxRow), &o)) sh = lineShapeMix(sh, o, std::min(aaCtx, 1.0f));
      sh.aLip *= kAaALip * (female_ ? kAaFALip : 1.0f);
      sh.lLip *= kAaLLip;
      sh.aOral *= kAaAor;
      sh.aPharynx *= kAaAph;
      sh.ac *= kAaAc;
      sh.length += kAaLenCm;
      if (kAaXc >= 0) sh.xc = kAaXc;
      if (kAaXc >= 0 || kAaLenCm != 0.0f) sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length);
    }
    if (have && tok.ph == Ph::L) {
      // l's shape by its place (L agent, round E 2026-11-11; all 1 / 0: the one o-like row as before). Kokoro's onset l
      // (play, black, clean) sits at the vowel's level with 2-5 kHz 15-20 dB stronger than puhuja's rounded-lip l; its
      // coda l (feel) has F1 440-460 where puhuja's is 330 (u-like: coda l heard as UW OW ER). Lips (area x, length x),
      // pharynx area x, and a mix toward a vowel row (onset: LINE_L_ONSET_ROW, coda: LINE_L_CODA_ROW, Ph numbers).
      static const float kLOnALip = param("LINE_L_ONSET_ALIP", 1.0f), kLOnLLip = param("LINE_L_ONSET_LLIP", 1.0f);
      static const float kLCoALip = param("LINE_L_CODA_ALIP", 1.0f), kLCoLLip = param("LINE_L_CODA_LLIP", 1.0f);
      static const float kLOnAph = param("LINE_L_ONSET_APH", 1.0f), kLCoAph = param("LINE_L_CODA_APH", 1.0f);
      static const float kLOnMix = param("LINE_L_ONSET_MIX", 0.0f), kLCoMix = param("LINE_L_CODA_MIX", 0.0f);
      static const int kLOnRow = param("LINE_L_ONSET_ROW", static_cast<int>(Ph::E)), kLCoRow = param("LINE_L_CODA_ROW", static_cast<int>(Ph::AA));
      const bool coda = !(next && isVowel(*next));
      const float mix = coda ? kLCoMix : kLOnMix;
      TractShape o;
      if (mix > 0 && lineShapeOf(static_cast<Ph>(coda ? kLCoRow : kLOnRow), &o)) sh = lineShapeMix(sh, o, mix);
      sh.aLip *= coda ? kLCoALip : kLOnALip;
      sh.lLip *= coda ? kLCoLLip : kLOnLLip;
      sh.aPharynx *= coda ? kLCoAph : kLOnAph;
    }
    if (have && tok.ph == Ph::E && next && *next == Ph::IH) eyStartShape(&sh, female_, tok.shape.eyctx);
    if (have && tok.ph == Ph::IH && lastPh_ == Ph::E) eyEndShape(&sh, female_, tok.shape.eyctx);
    if (have && tok.ph == Ph::IH && lastPh_ != Ph::E && lastPh_ != Ph::A && lastPh_ != Ph::AO) ihShape(&sh);
    if (have && tok.ph == Ph::AH) ahShape(&sh, lastPh_, next, female_, prevWasPause_);
    // (round-H ʊ agent, lab) the standalone ʊ (not the end of aʊ / oʊ): its own shape (uhShape); uhRow keeps the row's
    // shape for its late key (LINE_UH_LATE_ROW below).
    const bool uhAlone = have && tok.ph == Ph::UH && lastPh_ != Ph::A && lastPh_ != Ph::O;
    const TractShape uhRow = sh;
    if (uhAlone) uhShape(&sh, lastPh_, next, female_);
    // (Y agent round G, lab) after a j released from a plosive: LINE_U_AFTER_STOP_J (< 0: LINE_U_AFTER_J); an ʊ after
    // any j fronted likewise by LINE_UH_AFTER_J (0: off; "pure", "bureau").
    static const float kUAfterStopJ = param("LINE_U_AFTER_STOP_J", 0.2f), kUhAfterJ = param("LINE_UH_AFTER_J", 0.0f);
    const float uAfterJ = jAfterStop_ && kUAfterStopJ >= 0 ? kUAfterStopJ : kLineUAfterJ;
    if (have && tok.ph == Ph::U && lastPh_ == Ph::J && uAfterJ > 0) {
      TractShape front;
      if (lineShapeOf(Ph::I, &front)) sh = lineShapeMix(sh, front, uAfterJ);
    }
    if (have && tok.ph == Ph::UH && lastPh_ == Ph::J && kUhAfterJ > 0) {
      TractShape front;
      if (lineShapeOf(Ph::I, &front)) sh = lineShapeMix(sh, front, kUhAfterJ);
    }
    // The vowel uː's own shape on top of its row (the row also makes w, the ɹ rounding and the oʊ offglide, so the
    // row stays). Round-E uː agent (2026-11-11): the row narrows mid-pharynx with 1.55 cm^2 lips; a real uː is
    // rounded and protruded (lips 0.1-0.5 cm^2, longer), its constriction at the velum, the larynx lower
    // (ref/opt/review/phonetician.md 6, artsynth.md O2). Defaults reproduce the row.
    if (have && tok.ph == Ph::U && shapePh == Ph::U) {
      static const float kUwLipA = param("LINE_UW_LIPA", 1.0f), kUwLipL = param("LINE_UW_LIPL", 1.0f);
      static const float kUwXc = param("LINE_UW_XC", 0.4f), kUwAc = param("LINE_UW_AC", 0.9f), kUwWc = param("LINE_UW_WC", 0.7f);
      static const float kUwAph = param("LINE_UW_APH", 1.0f), kUwAor = param("LINE_UW_AOR", 1.3f), kUwAlar = param("LINE_UW_ALAR", 1.0f);
      static const float kUwLen = param("LINE_UW_LEN", 0.0f);  // cm added (larynx lowering + lip protrusion)
      // after a coronal (t d s z n l θ ð ʃ ʒ ʧ ʤ ɹ) the American uː is fronted (F2 1100-1500): share of ɪ's row mixed in
      static const float kUwCorFront = param("LINE_UW_COR_FRONT", 0.0f);
      static const float kUwFrontAll = param("LINE_UW_FRONT", 0.0f);  // ... and the same mix after anything (0: off)
      sh.aLip *= kUwLipA; sh.lLip *= kUwLipL; sh.ac *= kUwAc; sh.wc *= kUwWc;
      sh.aPharynx *= kUwAph; sh.aOral *= kUwAor; sh.aLarynx *= kUwAlar; sh.length += kUwLen;
      if (kUwXc >= 0) sh.xc = kUwXc;
      if (kUwXc >= 0 || kUwWc != 1.0f || kUwLen != 0.0f) sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length);
      const bool cor = lastPh_ == Ph::T || lastPh_ == Ph::D || lastPh_ == Ph::S || lastPh_ == Ph::Z || lastPh_ == Ph::N ||
                       lastPh_ == Ph::L || lastPh_ == Ph::TH || lastPh_ == Ph::DH || lastPh_ == Ph::SH || lastPh_ == Ph::ZH ||
                       lastPh_ == Ph::CH || lastPh_ == Ph::JH || lastPh_ == Ph::RR;
      const float fr = kUwFrontAll + (cor ? kUwCorFront : 0.0f);
      TractShape ih;
      if (fr > 0 && lastPh_ != Ph::J && lineShapeOf(Ph::IH, &ih)) sh = lineShapeMix(sh, ih, std::min(fr, 1.0f));
      // (round-I uː agent, lab; -1 / -1 / 1: off) the place by voice and context. Whisper heard uː as ɝ 146 times in
      // the dictionary (pool -> pearl, tune -> turn, booth -> birth): F3 sits 2.5-4 st under Kokoro's in the vowel's
      // middle (male 2092 vs 2426, female 2386 vs 2749; after b p m f v 2015 vs 2557) with F2 central; the place moved
      // forward raises F3. LINE_UW_XC_F: the female's place (-1: LINE_UW_XC); LINE_UW_LAB_XC: the place after a labial
      // (b p m f v w; -1: as elsewhere); LINE_UW_AOR_F: the female's mouth cavity x (on top of LINE_UW_AOR).
      static const float kUwXcF = param("LINE_UW_XC_F", 0.4f), kUwLabXc = param("LINE_UW_LAB_XC", -1.0f),
                         kUwAorF = param("LINE_UW_AOR_F", 1.0f);
      const bool lab = lastPh_ == Ph::B || lastPh_ == Ph::P || lastPh_ == Ph::M || lastPh_ == Ph::F || lastPh_ == Ph::V || lastPh_ == Ph::W;
      const float xcCtx = lab && kUwLabXc >= 0 ? kUwLabXc : female_ && kUwXcF >= 0 ? kUwXcF : -1.0f;
      if (xcCtx >= 0) { sh.xc = xcCtx; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
      if (female_) sh.aOral *= kUwAorF;
    }
    if (have) {
      // A vowel holds its shape over the middle of its sound (keys at 30 and
      // 70 %): with one key at the middle every vowel was a glide and reached
      // its target only for an instant. Consonants and pauses: one key.
      const bool hold = tok.ph != Ph::Stop;  // vowels and the shaped consonants (w j l ɹ) hold; a pause is one key
      if (!hold && haveLastKeyShape_) {
        // A pause: the mouth keeps the last sound's shape until the voice has
        // stopped, and relaxes to neutral only then. Interpolating toward the
        // neutral key through the vowel's tail made every final vowel glide
        // to ə (Whisper: "to" -> "Do it", "you" -> "Yolo it", 2026-11-02).
        lineKeys_.push_back({t0 + 0.15 * dur, lastKeyShape_});
        lineKeys_.push_back({t0 + 0.6 * dur, sh});
        addShapeDemand(t0, t0 + 0.15 * dur, lastKeyShape_, lastPh_, 1.0f);
        addShapeDemand(t0 + 0.6 * dur, t0 + dur, sh, Ph::Stop, kShapePauseW);
      } else {
        // The ɑ's hold keys at these shares instead of 0.3 / 0.7 (LINE_AA_KEY_FROM / _TO, -1 = as every vowel; 2026-11-10 AA agent:
        // does the coda alveolar's coarticulation pull the ɑ's F2 up?).
        static const float kLineAaKeyFrom = param("LINE_AA_KEY_FROM", -1.0f), kLineAaKeyTo = param("LINE_AA_KEY_TO", -1.0f);
        // A vowel after a w j l ɹ reaches its own shape at LINE_GLIDE_VOWEL_KEY of its length instead of 0.3: the glide's
        // rise ("way": F2 843 -> 1431 in 60 ms, Kokoro ~100 ms) is the gap from the glide's last key to this one
        // (Jyrki 2026-11-10: "way" too fast, "still not smooth enough").
        static const float kLineGlideVowelKey = param("LINE_GLIDE_VOWEL_KEY", 0.3f);
        const bool afterGlide = isVowel(tok.ph) && (lastPh_ == Ph::W || lastPh_ == Ph::J || lastPh_ == Ph::L || lastPh_ == Ph::RR);
        if (hold) lineKeys_.push_back({t0 + (tok.ph == Ph::AA && kLineAaKeyFrom >= 0 ? kLineAaKeyFrom : afterGlide ? kLineGlideVowelKey : 0.3) * dur, sh});
        // A vowel before a coda r is r-coloured through its second half (the
        // readers' "mortal": F3 falls from 2400 to 1600-2000 by 60 %; here the
        // ɔ held to a 41 ms ɹ, "maw-tal").
        TractShape late = sh;
        if (isVowel(tok.ph) && next && *next == Ph::RR) {
          TractShape r;
          // (0.5 in log area moved F3 only 2409 -> 2340; 0.8 gives 1937, the readers 1600-2000;
          // for ɑ it made "Part", "Card", "Start" heard as ɜ, 49 of 84, 2026-11-04: knobs)
          const bool open = tok.ph == Ph::AA || tok.ph == Ph::A;
          // (round-H ʊ agent, lab; < 0: as the other back vowels) a standalone ʊ before ɹ: its own r-colour share and key
          // (LINE_UH_R_MIX, LINE_UH_R_AT): "sure", "insure", "tour", "Moore" were heard as ɝ ("answer", "sir", "tur"),
          // 100 of the 280 ʊ word errors in the dictionary are before an ɹ.
          static const float kUhRMix = param("LINE_UH_R_MIX", -1.0f);
          // (round I EH agent, lab; < 0: as the front vowels) the ɛ's own r-colour share: LINE_EH_R_MIX.
          static const float kEhRMix = param("LINE_EH_R_MIX", -1.0f);
          const float mix = tok.ph == Ph::EH && kEhRMix >= 0 ? kEhRMix : uhAlone && kUhRMix >= 0 ? kUhRMix : open && kRColourOpenMix >= 0 ? kRColourOpenMix : frontVowel(tok.ph) && kRColourFrontMix >= 0 ? kRColourFrontMix : kRColourMix;
          if (lineShapeOf(Ph::ER, &r)) late = lineShapeMix(sh, r, mix);
        }
        if (isVowel(tok.ph) && next && *next == Ph::L && kLColourMix > 0) {
          TractShape l;
          if (lineShapeOf(Ph::L, &l)) late = lineShapeMix(late, l, kLColourMix);
        }
        // The ʊ's late key mixed toward ə (LINE_UH_OFF_MIX; 2026-11-09 UH agent: Kokoro's ʊ is not held, its F2 rises
        // 1100 -> 1300-1450 and F1 450 -> 500 over the last third before alveolars and velars; puhuja's is flat).
        static const float kUhOffMix = param("LINE_UH_OFF_MIX", 0.0f);
        if (tok.ph == Ph::UH && kUhOffMix > 0) {
          TractShape s8;
          if (lineShapeOf(Ph::Schwa, &s8)) late = lineShapeMix(late, s8, kUhOffMix);
        }
        // (round-H ʊ agent, lab; 0 / 1: off) the standalone ʊ's late key: back toward the row's own shape by
        // LINE_UH_LATE_ROW (uhShape then shapes the onset and middle only); before a coronal (t d s z n ʃ ...) toward ɪ by
        // LINE_UH_COR_OFF (Kokoro male "Put", "Could", "Hood": F2 1000-1100 -> 1350-1450 over the vowel, puhuja flat
        // 1200); its pharynx x LINE_UH_LATE_APH (Kokoro's F1 rises 440 -> 470-530 into the coda, puhuja's falls).
        static const float kUhLateRow = param("LINE_UH_LATE_ROW", 0.0f), kUhCorOff = param("LINE_UH_COR_OFF", 0.0f),
                           kUhLateAph = param("LINE_UH_LATE_APH", 1.0f);
        if (uhAlone) {
          if (kUhLateRow > 0) late = lineShapeMix(late, uhRow, kUhLateRow);
          TractShape ih;
          if (kUhCorOff > 0 && next && coronalPh(*next) && *next != Ph::L && *next != Ph::RR && lineShapeOf(Ph::IH, &ih))
            late = lineShapeMix(late, ih, kUhCorOff);
          late.aPharynx *= kUhLateAph;
        }
        // The ʌ's late key mixed toward ɑ (LINE_AH_OFF_MIX; round-F ʌ agent 2026-11-11: against the coda's fronting,
        // "sun", "fun" heard as "son"/"fan"/"fence"). 0: off.
        static const float kAhOffMix = param("LINE_AH_OFF_MIX", 0.0f);
        if (tok.ph == Ph::AH && kAhOffMix > 0) {
          TractShape aa;
          if (lineShapeOf(Ph::AA, &aa)) late = lineShapeMix(late, aa, kAhOffMix);
        }
        // (round-I ʌ agent 2026-11-12, lab; 0 / 0 / 1 / 1: off) the ʌ's late key toward ə (LINE_AH_OFF_SCHWA), before a
        // coronal obstruent or n toward ɪ (LINE_AH_COR_OFF), its pharynx and opening x LINE_AH_LATE_APH, _LATE_AC.
        // Kokoro's ʌ is not held: over its last fifth F1 falls and F2 rises (one-syllable words, male 579 -> 514 /
        // 1183 -> 1305 Hz, female 760 -> 593 / 1548 -> 1755; into t s ʧ: F2 to 1500-1700), puhuja's stays at
        // 567 / 1193, 691 / 1491: a held back vowel, heard as ɑ ("nut" -> not, "shut" -> shot, "slut" -> slot).
        static const float kAhOffSw = param("LINE_AH_OFF_SCHWA", 0.25f), kAhCorOff = param("LINE_AH_COR_OFF", 0.0f),
                           kAhLateAph = param("LINE_AH_LATE_APH", 1.0f), kAhLateAc = param("LINE_AH_LATE_AC", 1.0f);
        if (tok.ph == Ph::AH) {
          TractShape o2;
          if (kAhOffSw > 0 && lineShapeOf(Ph::Schwa, &o2)) late = lineShapeMix(late, o2, kAhOffSw);
          if (kAhCorOff > 0 && next && coronalPh(*next) && *next != Ph::L && *next != Ph::RR && lineShapeOf(Ph::IH, &o2))
            late = lineShapeMix(late, o2, kAhCorOff);
          late.aPharynx *= kAhLateAph;
          late.ac *= kAhLateAc;
        }
        // The uː's late key mixed back toward u's own row (LINE_UW_LATE_BACK; round-E uː agent 2026-11-11): Kokoro's ju
        // and coronal uː start fronted and fall (after j F2 1774 -> 1317 -> 1214 Hz over the thirds, male), the
        // fronting mixes (LINE_U_AFTER_J, LINE_UW_COR_FRONT) held puhuja's at 1577 -> 1555. 0: off.
        static const float kUwLateBack = param("LINE_UW_LATE_BACK", 0.0f);
        if (tok.ph == Ph::U && kUwLateBack > 0) {
          TractShape u;
          if (lineShapeOf(Ph::U, &u)) late = lineShapeMix(late, u, kUwLateBack);
        }
        // (round-I uː agent, lab; all 1 / -1: off) the uː's late key backs further: Kokoro's F2 falls through the uː
        // (male 1551 -> 1165 -> 1158 Hz over the thirds; word-final 1535 -> 1140 -> 952, female 2045 -> 1407 -> 1189),
        // puhuja's holds (1380 / 1316 / 1348; final female 1717 / 1590 / 1600) and was heard as ɝ at word end 61 times
        // (hebrew -> paper, unto -> hunter). Lips x LINE_UW_LATE_LIPA, mouth cavity x LINE_UW_LATE_AOR, constriction
        // area x LINE_UW_LATE_AC (all uː); before a pause x LINE_UW_END_LIPA / _AOR / _AC more, and its place there
        // LINE_UW_END_XC (-1: the vowel's). The lips alone (0.6) lowered F2 only 1348 -> 1320 and F1, F3 with it;
        // the cavity and the constriction move F2. Lab (300 words x 2 voices): LATE_AOR 1.4, LATE_AC 0.7 with LATE_AT 0.5
        // and LINE_U_AFTER_J 0.2: 52 uː errors against 94 (base) / 86 +- 8.5 (placebos). lateNoUw keeps these out of the
        // r-colour test below; uwLate adds their demand.
        static const float kUwLateLipA = param("LINE_UW_LATE_LIPA", 1.0f), kUwLateAor = param("LINE_UW_LATE_AOR", 1.4f),
                           kUwEndLipA = param("LINE_UW_END_LIPA", 1.0f), kUwEndAor = param("LINE_UW_END_AOR", 1.0f),
                           kUwLateAc = param("LINE_UW_LATE_AC", 0.7f), kUwEndAc = param("LINE_UW_END_AC", 1.0f),
                           kUwEndXc = param("LINE_UW_END_XC", -1.0f);
        const bool uwEnd = tok.ph == Ph::U && (!next || *next == Ph::Stop);
        TractShape lateNoUw = late;  // (refreshed after the ɛ's late mix below: only the uː's own changes are kept out)
        if (tok.ph == Ph::U) {
          late.aLip *= kUwLateLipA * (uwEnd ? kUwEndLipA : 1.0f);
          late.aOral *= kUwLateAor * (uwEnd ? kUwEndAor : 1.0f);
          late.ac *= kUwLateAc * (uwEnd ? kUwEndAc : 1.0f);
          if (uwEnd && kUwEndXc >= 0) { late.xc = kUwEndXc; late.sc = lineVowelStrength(late.xc, late.wc, late.length); }
        }
        const bool uwLate = late.aLip != lateNoUw.aLip || late.aOral != lateNoUw.aOral || late.ac != lateNoUw.ac || late.xc != lateNoUw.xc;
        // The standalone ɪ's late key mixed toward ə (LINE_IH_OFF_MIX; IH agent 2026-11-11: Kokoro's ɪ centralises over its
        // last third, F2 1795 -> 1663 Hz male, 2106 -> 1931 female; puhuja's rises 1861 -> 1934, toward eɪ's end).
        static const float kIhOffMix = param("LINE_IH_OFF_MIX", 0.0f);
        if (tok.ph == Ph::IH && kIhOffMix > 0 && lastPh_ != Ph::E && lastPh_ != Ph::A && lastPh_ != Ph::AO) {
          TractShape s8;
          if (lineShapeOf(Ph::Schwa, &s8)) late = lineShapeMix(late, s8, kIhOffMix);
        }
        // The ɑ's late key mixed toward ɔ (LINE_AA_OFF_MIX; 2026-11-10 AA agent: a back offglide against the coda's fronting).
        static const float kAaOffMix = param("LINE_AA_OFF_MIX", 0.0f);
        if (tok.ph == Ph::AA && kAaOffMix > 0) {
          TractShape o;
          if (lineShapeOf(Ph::AO, &o)) late = lineShapeMix(late, o, kAaOffMix);
        }
        // (AA agent 10, round K, lab knobs; 0 / 1: off) the ɑ's late key opens or fronts instead: toward row LINE_AA_OFF_ROW
        // by LINE_AA_OFF_FRONT, and by LINE_AA_FINAL_OFF more before a pause; its lips x LINE_AA_LATE_LIPA. Kokoro's ɑ
        // ends with F2 rising 100-250 Hz in "Pa", "La", "Las", "Lot", "Hot"; puhuja's holds ~910 to the end, where an ɔ
        // (rounding, F2 falling) and an ɑ differ. Not before ɹ. Lab (hard300: 180 dictionary words with >= 2 of 4 ɑ errors +
        // 120 common, both voices, two draws): LINE_AA_OFF_FRONT 0.5 223 -> 177 errors, words right 290 -> 326 (common 300:
        // 54 -> 42, words1000 1447 -> 1458); half of it is the late key becoming the ɑ's own demand (LATE_LIPA 1.001: 196).
        static const float kAaOffFront = param("LINE_AA_OFF_FRONT", 0.5f), kAaFinalOff = param("LINE_AA_FINAL_OFF", 0.0f),
                           kAaLateLipA = param("LINE_AA_LATE_LIPA", 1.0f);
        static const int kAaOffRow = param("LINE_AA_OFF_ROW", static_cast<int>(Ph::AH));
        if (tok.ph == Ph::AA && !(next && *next == Ph::RR) && (kAaOffFront > 0 || kAaFinalOff > 0 || kAaLateLipA != 1.0f)) {
          const float off = kAaOffFront + (!next || *next == Ph::Stop ? kAaFinalOff : 0.0f);
          TractShape o;
          if (off > 0 && lineShapeOf(static_cast<Ph>(kAaOffRow), &o)) late = lineShapeMix(late, o, std::min(off, 1.0f));
          late.aLip *= kAaLateLipA;
          lateNoUw = late;  // (an ɑ has no uː changes: its late mix counts as its own end, as the ɔ's)
        }
        // The ɛ's late key (round I EH agent 2026-11-12 lab knobs; 0: off): toward ə (LINE_EH_OFF_MIX; before a nasal
        // LINE_EH_N_OFF_MIX instead when >= 0), toward the l before an l (LINE_EH_L_OFF_MIX), toward ɪ before a nasal
        // (LINE_EH_N_OFF_IH). Kokoro's ɛ before n falls into it (male one-syllable last third F1/F2 453/1507), puhuja's
        // rises (576/2006: the tensed American æ of "bank"; "bent" -> bank, "tend" -> tank, 71 of the 150 ɛn errors).
        // Lab: LINE_EH_N_OFF_MIX 0.5: "ten" end 638/2113 -> 570/1950 Hz, ɛ errors 82 -> 71, words 386 -> 408, no veto;
        // with LINE_EH_W 2: 62 / 412, words1000 1409 -> 1397, VETO fspeed +1.0σ, bright +1.1σ.
        static const float kEhOffMix = param("LINE_EH_OFF_MIX", 0.0f), kEhNOffMix = param("LINE_EH_N_OFF_MIX", -1.0f),
                           kEhLOffMix = param("LINE_EH_L_OFF_MIX", 0.0f), kEhNOffIh = param("LINE_EH_N_OFF_IH", 0.0f);
        if (tok.ph == Ph::EH) {
          const bool nas = next && info(*next).cls == Cls::Nasal;
          const float off = nas && kEhNOffMix >= 0 ? kEhNOffMix : kEhOffMix;
          TractShape o;
          if (off > 0 && lineShapeOf(Ph::Schwa, &o)) late = lineShapeMix(late, o, off);
          if (nas && kEhNOffIh > 0 && lineShapeOf(Ph::IH, &o)) late = lineShapeMix(late, o, kEhNOffIh);
          if (next && *next == Ph::L && kEhLOffMix > 0 && lineShapeOf(Ph::L, &o)) late = lineShapeMix(late, o, kEhLOffMix);
          lateNoUw = late;  // (an ɛ has no uː changes: its late mix counts as its own end, as in the EH agent's lab)
        }
        // The ɔ's late key mixed toward ʌ (LINE_AO_OFF_MIX; AO agent 2026-11-11, 0 = as before): a centring end against
        // the oʊ reading (Whisper: "Paul" -> "pole"; oʊ closes, the American ɔ opens or centres).
        static const float kAoOffMix = param("LINE_AO_OFF_MIX", 0.0f);
        if (tok.ph == Ph::AO && kAoOffMix > 0 && !(next && (*next == Ph::IH || *next == Ph::RR))) {
          TractShape c;
          if (lineShapeOf(Ph::AH, &c)) late = lineShapeMix(late, c, kAoOffMix);
        }
        // (AO agent 8, round I, lab; 0 / 1: off) the ɔ's late key against the oʊ reading: toward ɑ by LINE_AO_OFF_AA (the
        // jaw keeps opening: Kokoro's F1 rises to its peak at 50-70 % of the vowel, puhuja's stays at 400-450), toward ə
        // before a pause only by LINE_AO_FINAL_OFF (Kokoro's "law", "saw" end level or rising in F2; puhuja's fall to
        // ~800 and the dictionary's final ɔ is heard as l: "saw" -> "soul", "utah" -> "little"), its lips x
        // LINE_AO_LATE_LIPA (the oʊ closes its lips at the end; the ɔ does not). Not before ɪ (ɔɪ) or ɹ.
        static const float kAoOffAa = param("LINE_AO_OFF_AA", 0.5f), kAoFinalOff = param("LINE_AO_FINAL_OFF", 0.0f),
                           kAoLateLipA = param("LINE_AO_LATE_LIPA", 1.0f);
        if (tok.ph == Ph::AO && !(next && (*next == Ph::IH || *next == Ph::RR))) {
          TractShape c;
          if (kAoOffAa > 0 && lineShapeOf(Ph::AA, &c)) late = lineShapeMix(late, c, kAoOffAa);
          if (kAoFinalOff > 0 && (!next || *next == Ph::Stop) && lineShapeOf(Ph::Schwa, &c)) late = lineShapeMix(late, c, kAoFinalOff);
          if (kAoLateLipA != 1.0f) late.aLip *= kAoLateLipA;
          lateNoUw = late;  // (an ɔ has no uː changes: its late mix counts as its own end, as in the AO agent's lab)
        }
        // A coda l before a vowel or h across the word boundary ("mortal hands") leaves its shape at this share of
        // its length instead of 0.7 (LINE_CODA_L_KEY; the readers' l moves toward the vowel over its last 60 ms).
        static const float kLineCodaLKey = param("LINE_CODA_L_KEY", 0.7f);
        // The ɔ's late hold key at this share instead of 0.7 (LINE_AO_KEY_TO, -1 = as every vowel; AO agent 2026-11-11: the coda
        // pulled "talk" 574 -> 469 and "cross" F2 down over the last third).
        static const float kLineAoKeyTo = param("LINE_AO_KEY_TO", -1.0f);
        const bool codaLRelease = tok.ph == Ph::L && next && *next == Ph::H;  // (a vowel next: no word-boundary knowledge here, so only before h)
        lineKeys_.push_back({t0 + (codaLRelease ? kLineCodaLKey : tok.ph == Ph::W ? kLineWKeyTo : tok.ph == Ph::AA && kLineAaKeyTo >= 0 ? kLineAaKeyTo : tok.ph == Ph::AO && kLineAoKeyTo >= 0 ? kLineAoKeyTo : hold ? 0.7 : 0.5) * dur, late});
        {
          // The planner's demands: the held middle; an r-coloured end of its own.
          const bool rColoured = lateNoUw.ac != sh.ac || lateNoUw.xc != sh.xc || lateNoUw.aLip != sh.aLip || ((uhAlone || tok.ph == Ph::AH) && late.aPharynx != sh.aPharynx);  // (ʊ: LINE_UH_LATE_APH; uː: LINE_UW_LATE_*; ʌ: LINE_AH_LATE_APH)
          const float preR = isVowel(tok.ph) && next && *next == Ph::RR ? std::max(kPreRVowelW, frontVowel(tok.ph) ? kPreRFrontW : 0.0f) : 0.0f;  // see kPreRVowelW
          TractShape own;
          const bool borrowed = !isVowel(tok.ph) && !lineShapeOf(tok.ph, &own);  // f in the next vowel's shape
          float ws = borrowed ? kShapeBorrowedW : 1.0f;
          // The m's key demanded from LINE_M_PRE_MS before it (round J, M agent; 0: off), weight x LINE_M_PRE_W: the
          // vowel's F2 bends toward the lips over its end (Kokoro "Kim" 1970 -> 1100 over 50-60 ms into the m).
          static const float kMPreMs = param("LINE_M_PRE_MS", 0.0f), kMPreW = param("LINE_M_PRE_W", 1.0f);
          if (tok.ph == Ph::M && kMPreMs > 0 && prevWasVowel_)
            addShapeDemand(std::max(t0 - kMPreMs * ms, phonemeStart_ + 1), t0 + 0.5 * dur, sh, Ph::M, kMPreW * ws);
          // ... and before a consonant the m's shape held to LINE_M_HOLD_TO of its length (round J, M agent; 0: off): the
          // next consonant's key pulled the closed tract toward it ("Claimed": the murmur's F2 rose 1263 -> 1644 Hz over
          // the m's last 30 ms toward the d's ɪ shape; Kokoro's stays ~1100 to the d).
          static const float kMHoldTo = param("LINE_M_HOLD_TO", 0.0f);
          if (tok.ph == Ph::M && kMHoldTo > 0.7f && next && !isVowel(*next) && *next != Ph::Stop)
            lineKeys_.push_back({t0 + std::min(kMHoldTo, 1.0f) * dur, sh});
          if (tok.ph == Ph::DH && kLineDhShape > 0) ws *= kLineDhShape;  // (LINE_DH_SHAPE)
          // A vowel chain (aʊ, aɪ, oʊ, Finnish ai, uo) is one glide: the first
          // vowel is reached early and briefly, the second only near its end.
          // Held over both middles, the whole tract changed in ~100 ms between
          // them ("now": an event where none should be, Jyrki 2026-11-03).
          const bool chainStart = isVowel(tok.ph) && next && isVowel(*next) && *next != tok.ph;
          const bool chainEnd = isVowel(tok.ph) && isVowel(lastPh_) && lastPh_ != tok.ph &&
                                !(tok.ph == Ph::ER && kLineErChain == 0) && !(kLineChainStop > 0 && prevWasPlosive_);
          if (!hold) addShapeDemand(t0 + 0.5 * dur, t0 + 0.5 * dur, sh, tok.ph, ws);
          else if (!rColoured && (chainStart || chainEnd)) {
            const bool owGlide = chainEnd && tok.ph == Ph::UH && lastPh_ == Ph::O && kLineOwGlideFrom >= 0;
            const bool eyGlide = chainEnd && tok.ph == Ph::IH && lastPh_ == Ph::E && kLineEyGlideFrom >= 0;
            const bool ayGlide = chainEnd && tok.ph == Ph::IH && lastPh_ == Ph::A && kLineAyGlideFrom >= 0;
            const bool awGlide = chainEnd && tok.ph == Ph::UH && lastPh_ == Ph::A && kLineAwGlideFrom >= 0;
            const bool oyEnd = chainEnd && tok.ph == Ph::IH && lastPh_ == Ph::AO;
            const bool oyStart = chainStart && tok.ph == Ph::AO && *next == Ph::IH;
            // eɪ's planner demands (EY agent 2026-11-10 lab knobs; -1 / 1 / 0: as before): the e demanded from LINE_EY_E_FROM
            // (else 0.1) to LINE_EY_HOLD_TO of its length (else SPLAN_GLIDE_START), the ɪ demanded to LINE_EY_GLIDE_TO of its
            // length (else 0.9; > 1 runs into the next sound), its weight x LINE_EY_GLIDE_W; the tongue place's least weight in
            // the e and the ɪ demands (LINE_EY_E_PLACE_W, LINE_EY_PLACE_W; the ɪ's place weighs only 0.14, its 2 cm^2 opening
            // counting as half open).
            static const float kEyHoldTo = param("LINE_EY_HOLD_TO", 0.15f), kEyGlideTo = param("LINE_EY_GLIDE_TO", 1.0f),
                               kEyGlideW = param("LINE_EY_GLIDE_W", 1.0f), kEyPlaceW = param("LINE_EY_PLACE_W", 0.0f),
                               kEyEPlaceW = param("LINE_EY_E_PLACE_W", 0.0f), kEyEFrom = param("LINE_EY_E_FROM", 0.0f);
            const bool eyStart = chainStart && tok.ph == Ph::E && *next == Ph::IH;
            const double a0 = owGlide ? kLineOwGlideFrom : eyGlide ? kLineEyGlideFrom :
                             ayGlide ? kLineAyGlideFrom : awGlide ? kLineAwGlideFrom :
                             oyEnd && kLineOyGlideFrom >= 0 ? kLineOyGlideFrom :
                             chainEnd ? kShapeGlideEnd : oyStart && kLineOyAoFrom >= 0 ? kLineOyAoFrom : eyStart && kEyEFrom >= 0 ? kEyEFrom : 0.1;
            const bool ayStart = chainStart && tok.ph == Ph::A && *next == Ph::IH;  // aɪ's a (kLineAyHold, kLineAyAW)
            // A hiatus iː ("media", "create", "area"; IY agent 2026-11-11 lab knobs, -1 / 1: as before): demanded only from
            // 0.1 to SPLAN_GLIDE_START (0.3) of its length like a diphthong's start, its F2 stopped at 1850-1950 Hz after d
            // ("media", "radio", "video") and at 1600 in "create" (Kokoro holds 2250-2350 for ~60 ms); 33 % of these iː
            // were wrong, heard as j, the next ə as u ("area" -> "value"). LINE_IY_CHAIN_FROM / _TO: the iː's demand
            // window (shares of it), LINE_IY_CHAIN_W its weight; LINE_IY_NEXT_FROM / _TO: the following vowel's window
            // (else SPLAN_GLIDE_END / 0.9).
            static const float kIyChainFrom = param("LINE_IY_CHAIN_FROM", -1.0f), kIyChainTo = param("LINE_IY_CHAIN_TO", -1.0f),
                               kIyChainW = param("LINE_IY_CHAIN_W", 1.0f), kIyNextFrom = param("LINE_IY_NEXT_FROM", -1.0f),
                               kIyNextTo = param("LINE_IY_NEXT_TO", -1.0f), kIyChainPlaceW = param("LINE_IY_CHAIN_PLACE_W", 1.0f);
            const bool iyStart = chainStart && tok.ph == Ph::I, iyNext = chainEnd && lastPh_ == Ph::I;
            // (round-I uː agent, lab; -1 / 1: as before) a uː before another vowel (viewer -> year, newer -> gear, sewer
            // -> sir; 26 % wrong, 38 deleted) is demanded only to SPLAN_GLIDE_START (0.3) of its length like a
            // diphthong's start: LINE_UW_CHAIN_TO its end (share), LINE_UW_CHAIN_W its weight.
            static const float kUwChainTo = param("LINE_UW_CHAIN_TO", -1.0f), kUwChainW = param("LINE_UW_CHAIN_W", 1.0f);
            const bool uwStart = chainStart && tok.ph == Ph::U;
            // ... and the vowel after it demanded from LINE_UW_NEXT_FROM of its length (-1: SPLAN_GLIDE_END).
            static const float kUwNextFrom = param("LINE_UW_NEXT_FROM", -1.0f);
            const bool uwNext = chainEnd && lastPh_ == Ph::U;
            const double a = iyStart && kIyChainFrom >= 0 ? kIyChainFrom : iyNext && kIyNextFrom >= 0 ? kIyNextFrom
                             : uwNext && kUwNextFrom >= 0 ? kUwNextFrom : a0;
            const double b = iyStart && kIyChainTo >= 0 ? kIyChainTo : iyNext && kIyNextTo >= 0 ? kIyNextTo
                             : uwStart && kUwChainTo >= 0 ? kUwChainTo
                             : oyStart && kLineOyHoldTo >= 0 ? kLineOyHoldTo : ayStart && kLineAyHold >= 0 ? kLineAyHold
                             : chainStart ? (eyStart && kEyHoldTo >= 0 ? kEyHoldTo : kShapeGlideStart)
                             : ayGlide && kLineAyGlideTo >= 0 ? kLineAyGlideTo : eyGlide && kEyGlideTo >= 0 ? kEyGlideTo : 0.9;
            const float oyW = (oyStart ? kLineOyAoW : oyEnd ? kLineOyIhW : ayStart ? kLineAyAW : ayGlide ? kLineAyIW : 1.0f) *
                              (iyStart ? kIyChainW : 1.0f) * (uwStart ? kUwChainW : 1.0f);
            // aɪ's a wanted before its start after a pause (LINE_AY_PRE_MS: "I", "item" start from the pause's ə) or
            // in a plosive's closure (LINE_AY_STOP_LEAD_MS: "time", "type"); 0: from 10 % of the a, as before.
            const float ayLead = !ayStart ? 0.0f : prevWasPause_ ? kLineAyPreMs : prevWasPlosive_ ? kLineAyStopLeadMs : 0.0f;
            // ... and eɪ's e likewise (LINE_EY_PRE_MS after a pause: "eight", "able"; LINE_EY_STOP_LEAD_MS in a plosive's
            // closure: "take", "came", "paper"; EY agent10 round K lab knobs, 0: off; eɪ after a pause 24 % wrong).
            static const float kEyPreMs = param("LINE_EY_PRE_MS", 0.0f), kEyStopLeadMs = param("LINE_EY_STOP_LEAD_MS", 0.0f);
            const float lead = ayLead > 0 ? ayLead : !eyStart ? 0.0f : prevWasPause_ ? kEyPreMs : prevWasPlosive_ ? kEyStopLeadMs : 0.0f;
            addShapeDemand(lead > 0 ? t0 - lead * ms : t0 + std::min(a, b) * dur, t0 + std::max(a, b) * dur, sh, tok.ph,
                           ws * kShapeGlideW * oyW * (eyGlide ? kEyGlideW : 1.0f),
                           ayStart ? kLineAyPlaceFloor : eyGlide ? kEyPlaceW : eyStart ? kEyEPlaceW : iyStart ? kIyChainPlaceW : 0.0f);
            if ((ayStart || ayGlide) && kLineAyRamp > 1) {
              // A lazy aɪ (LINE_AY_RAMP points over the whole diphthong, 0: off): point demands on the straight path
              // from the a's shape to the ɪ's, progress x^LINE_AY_RAMP_POW of the diphthong's time (the a taken as
              // LINE_AY_RAMP_SPLIT of it). Jyrki 2026-11-10: Kokoro's diphthongs move steadily through the whole
              // vowel; puhuja's "holds, rushes, holds".
              TractShape s0, s1;
              if (lineShapeOf(Ph::A, &s0) && lineShapeOf(Ph::IH, &s1)) {
                ayStartShape(&s0);
                ayEndShape(&s1);
                const int n = static_cast<int>(kLineAyRamp);
                for (int q = 1; q < n; ++q) {
                  const double x = static_cast<double>(q) / n;  // the diphthong's time
                  const double u = ayStart ? x / kLineAyRampSplit : (x - kLineAyRampSplit) / (1.0 - kLineAyRampSplit);  // this part's
                  if (u < 0 || u > 1) continue;
                  const float frac = static_cast<float>(std::pow(x, static_cast<double>(kLineAyRampPow)));
                  const double t = t0 + u * dur;
                  addShapeDemand(t, t, lineShapeMix(s0, s1, frac), tok.ph, ws * kShapeGlideW * kLineAyRampW);
                }
              }
            }
            // A lazy eɪ (LINE_EY_RAMP points over the whole diphthong, 0: off; the aɪ agent's LINE_AY_RAMP): point demands
            // on the straight path from the e's shape to the ɪ's, progress x^LINE_EY_RAMP_POW of the diphthong's time (the
            // e taken as LINE_EY_RAMP_SPLIT of it), weight x LINE_EY_RAMP_W. Jyrki 2026-11-10: "the planning module cannot
            // make a good formant run for a single word sentence like 'Day.' but makes a very rapid formant change";
            // Kokoro's "Day." F2 rises steadily 1590 -> 2370 over ~350 ms, puhuja's dipped, held, rushed ~100 ms, held.
            static const float kEyRamp = param("LINE_EY_RAMP", 8.0f), kEyRampPow = param("LINE_EY_RAMP_POW", 1.5f),
                               kEyRampSplit = param("LINE_EY_RAMP_SPLIT", 0.6f), kEyRampW = param("LINE_EY_RAMP_W", 1.0f);
            // In a closed syllable the ramp's power LINE_EY_CL_RAMP_POW (<= 0: LINE_EY_RAMP_POW; a later glide keeps the
            // e longer where Whisper hears ɪ) and the ramp's weight for the female voice x LINE_EY_F_RAMP_W (round K, 1: as before).
            static const float kEyClRampPow = param("LINE_EY_CL_RAMP_POW", 0.0f), kEyFRampW = param("LINE_EY_F_RAMP_W", 1.0f);
            const float eyRampPow = (tok.shape.eyctx & 1) && kEyClRampPow > 0 ? kEyClRampPow : kEyRampPow;
            const float eyRampW = kEyRampW * (female_ ? kEyFRampW : 1.0f);
            if ((eyStart || eyGlide) && kEyRamp > 1) {
              TractShape s0, s1;
              if (lineShapeOf(Ph::E, &s0) && lineShapeOf(Ph::I, &s1)) {
                eyStartShape(&s0, female_, tok.shape.eyctx);
                eyEndShape(&s1, female_, tok.shape.eyctx);
                const int n = static_cast<int>(kEyRamp);
                for (int q = 1; q < n; ++q) {
                  const double x = static_cast<double>(q) / n;  // the diphthong's time
                  const double u = eyStart ? x / kEyRampSplit : (x - kEyRampSplit) / (1.0 - kEyRampSplit);  // this part's
                  if (u < 0 || u > 1) continue;
                  const float frac = static_cast<float>(std::pow(x, static_cast<double>(eyRampPow)));
                  addShapeDemand(t0 + u * dur, t0 + u * dur, lineShapeMix(s0, s1, frac), tok.ph, ws * kShapeGlideW * eyRampW);
                }
              }
            }
            // A lazy oʊ (LINE_OW_RAMP points over the whole diphthong, 0: off; the eɪ's LINE_EY_RAMP): point demands on
            // the straight path from the o start's shape to the ʊ end's, progress x^LINE_OW_RAMP_POW of the diphthong's
            // time (the o taken as LINE_OW_RAMP_SPLIT of it), weight x LINE_OW_RAMP_W. Kokoro's "go", "know": F2 falls
            // steadily 1250 -> 850 over the vowel; puhuja's held ~1250.
            static const float kOwRamp = param("LINE_OW_RAMP", 0.0f), kOwRampPow = param("LINE_OW_RAMP_POW", 1.0f),
                               kOwRampSplit = param("LINE_OW_RAMP_SPLIT", 0.6f), kOwRampW = param("LINE_OW_RAMP_W", 1.0f);
            const bool owStart = chainStart && tok.ph == Ph::O && *next == Ph::UH;
            if ((owStart || owGlide) && kOwRamp > 1) {
              TractShape s0, s1, mid;
              if (lineShapeOf(Ph::O, &s0) && lineShapeOf(Ph::U, &s1)) {
                if (lineShapeOf(Ph::AH, &mid)) s0 = lineShapeMix(s0, mid, kLineOwStartAh);
                owStartShape(&s0);
                owEndShape(&s1);
                const int n = static_cast<int>(kOwRamp);
                for (int q = 1; q < n; ++q) {
                  const double x = static_cast<double>(q) / n;  // the diphthong's time
                  const double u = owStart ? x / kOwRampSplit : (x - kOwRampSplit) / (1.0 - kOwRampSplit);  // this part's
                  if (u < 0 || u > 1) continue;
                  const float frac = static_cast<float>(std::pow(x, static_cast<double>(kOwRampPow)));
                  addShapeDemand(t0 + u * dur, t0 + u * dur, lineShapeMix(s0, s1, frac), tok.ph, ws * kShapeGlideW * kOwRampW);
                }
              }
            }
          } else if (!rColoured) {
            // A phrase-initial j or l: the tongue in its place before the voice
            // (2026-11-04: the j's F2 rose from the pause's ə, 1550 Hz, and
            // reached only ~1750 by its end; Kokoro holds 2100-2200 from the start).
            float preMs = !prevWasPause_ ? (prevWasPlosive_ && isVowel(tok.ph) ? kLineAfterStopLeadMs : 0.0f) : tok.ph == Ph::J ? kLineJPreShapeMs : tok.ph == Ph::L ? kLineLPreShapeMs : tok.ph == Ph::W ? kLineWPreShapeMs : tok.ph == Ph::RR ? kLineRPreShapeMs : tok.ph == Ph::ER ? kLineErPreMs : 0.0f;
            const bool er = tok.ph == Ph::ER;
            // (round I EH agent, lab; 0: none, as before) a phrase-initial ɛ's shape asked for this long before its
            // voice (LINE_EH_PAUSE_PRE_MS): word-initial ɛ 17.5 % wrong, heard as æ 83, ɪ 59, with a consonant ("Ellis" ->
            // valleys/palace, "Erik" -> direct): its onset glided from the pause's ə.
            static const float kEhPausePreMs = param("LINE_EH_PAUSE_PRE_MS", 0.0f);
            if (prevWasPause_ && tok.ph == Ph::EH && kEhPausePreMs > 0) preMs = kEhPausePreMs;
            // A held iː's demand window and weight (IY agent 2026-11-11 lab knobs; -1 / -1 / 1: 0.3-0.7, as before): in
            // connected speech (kennedy) its F2 sat at 1730 Hz against Kokoro's 2010 (isolated "Dee" 2055 vs ~2300).
            static const float kIyFrom = param("LINE_IY_FROM", -1.0f), kIyTo = param("LINE_IY_TO", -1.0f), kIyW = param("LINE_IY_W", 3.0f),
                               kIyPlaceW = param("LINE_IY_PLACE_W", 1.0f);  // the tongue place's least weight (its 2 cm^2 opening counts as half open)
            const bool iy = tok.ph == Ph::I;
            // (round-H ʊ agent, lab; -1 / -1 / 1 / 0: as before) the same for a standalone ʊ: LINE_UH_FROM, _TO, _W, _PLACE_W.
            static const float kUhFrom = param("LINE_UH_FROM", -1.0f), kUhTo = param("LINE_UH_TO", -1.0f), kUhW = param("LINE_UH_W", 1.0f),
                               kUhPlaceW = param("LINE_UH_PLACE_W", 0.0f);
            // (round-I uː agent, lab; -1 / -1 / 1 / 0 / 0: as before) the same for uː: LINE_UW_FROM, _TO, _W, _PLACE_W, and
            // its lips rounded LINE_UW_LEAD_MS before it after a consonant (not after a pause or a vowel).
            static const float kUwFrom = param("LINE_UW_FROM", -1.0f), kUwTo = param("LINE_UW_TO", -1.0f), kUwW = param("LINE_UW_W", 1.0f),
                               kUwPlaceW = param("LINE_UW_PLACE_W", 0.0f), kUwLeadMs = param("LINE_UW_LEAD_MS", 0.0f);
            // ... and before an l its demand ends at LINE_UW_PRE_L_TO of its length (-1: as LINE_UW_TO): pools -> pearls,
            // module -> natural (before l 29 % wrong; F3 2003 vs Kokoro's 2415 in the middle, male).
            static const float kUwPreLTo = param("LINE_UW_PRE_L_TO", -1.0f);
            const bool uwD = tok.ph == Ph::U;
            if (uwD && kUwLeadMs > 0 && !prevWasPause_ && !isVowel(lastPh_)) preMs = std::max(preMs, kUwLeadMs);
            // (round I EH agent, lab; -1 / -1 / 1 / 0: as before) the same for ɛ: LINE_EH_FROM, _TO, _W, _PLACE_W. Lab
            // (300 one-syllable / stressed two-syllable ɛ words, 2 voices, 2 draws): _W 2: ɛ errors 82 -> 67, words right
            // 386 -> 406, words1000 1409 -> 1402, no hybrid veto, guards clean (base placebos 67 / 86: noise is wide).
            static const float kEhFrom = param("LINE_EH_FROM", -1.0f), kEhTo = param("LINE_EH_TO", -1.0f), kEhW = param("LINE_EH_W", 1.0f),
                               kEhPlaceW = param("LINE_EH_PLACE_W", 0.0f);
            // (round-I ʌ agent, lab; -1 / -1 / 1 / 0: as before) the same for the stressed ʌ: LINE_AH_FROM, _TO, _W, _PLACE_W
            // (a lax vowel: does the planner reach it, or should it leave earlier?).
            static const float kAhFrom = param("LINE_AH_FROM", -1.0f), kAhTo = param("LINE_AH_TO", -1.0f), kAhW = param("LINE_AH_W", 1.0f),
                               kAhPlaceW = param("LINE_AH_PLACE_W", 0.0f);
            const bool ahD = tok.ph == Ph::AH && (kAhFrom >= 0 || kAhTo >= 0 || kAhW != 1.0f || kAhPlaceW != 0.0f);
            const bool uhD = uhAlone || (tok.ph == Ph::EH && (kEhFrom >= 0 || kEhTo >= 0 || kEhW != 1.0f || kEhPlaceW > 0));
            const bool eh = tok.ph == Ph::EH;
            // (AO agent 8, round I, lab; -1 / -1 / 1 / 0: as before) the same for the ɔ: LINE_AO_FROM, _TO, _W, _PLACE_W.
            // puhuja's male ɔ ran 400-450 / 750-900 Hz with F2 falling through the vowel ("call" 972 -> 750, "saw"
            // 911 -> 815), Kokoro's 530-600 / 1000-1100 with F2 level or rising; Whisper heard oʊ in 251 of 760 errors.
            static const float kAoFrom = param("LINE_AO_FROM", -1.0f), kAoTo = param("LINE_AO_TO", -1.0f), kAoW = param("LINE_AO_W", 1.0f),
                               kAoPlaceW = param("LINE_AO_PLACE_W", 0.0f);
            const bool aoD = tok.ph == Ph::AO;
            // ... and before an l its window ends at LINE_AO_L_TO instead (< 0: as above): the l pulled the vowel down
            // from 30 % of it on ("call": F2 972 -> 750 Hz over the vowel).
            static const float kAoLTo = param("LINE_AO_L_TO", -1.0f);
            const float aoTo = aoD && next && *next == Ph::L && kAoLTo >= 0 ? kAoLTo : kAoTo;
            // (round-J ə agent 9, lab; -1 / -1 / 1 / 0 / 0: as before) the same for the ə: LINE_SCHWA_FROM, _TO, _W, _PLACE_W,
            // and a word-initial ə's shape asked for LINE_SCHWA_PAUSE_LEAD_MS before its voice after a pause (it glided out
            // of the pause's own ə shape: Kokoro's "about", "again" start at F1 460-520 Hz, puhuja's at 330-350).
            static const float kSwFrom = param("LINE_SCHWA_FROM", -1.0f), kSwTo = param("LINE_SCHWA_TO", -1.0f), kSwW = param("LINE_SCHWA_W", 1.0f),
                               kSwPlaceW = param("LINE_SCHWA_PLACE_W", 0.0f), kSwPauseLeadMs = param("LINE_SCHWA_PAUSE_LEAD_MS", 60.0f);
            const bool swD = tok.ph == Ph::Schwa && (kSwFrom >= 0 || kSwTo >= 0 || kSwW != 1.0f || kSwPlaceW != 0.0f);
            if (tok.ph == Ph::Schwa && prevWasPause_ && kSwPauseLeadMs > 0) preMs = kSwPauseLeadMs;
            const float vFrom = iy ? kIyFrom : uwD ? kUwFrom : eh ? kEhFrom : ahD ? kAhFrom : aoD ? kAoFrom : swD ? kSwFrom : uhD ? kUhFrom : -1.0f,
                        vTo = iy ? kIyTo : uwD ? (next && *next == Ph::L && kUwPreLTo >= 0 ? kUwPreLTo : kUwTo) : eh ? kEhTo : ahD ? kAhTo : aoD ? aoTo : swD ? kSwTo : uhD ? kUhTo : -1.0f,
                        vW = iy ? kIyW : uwD ? kUwW : eh ? kEhW : ahD ? kAhW : aoD ? kAoW : swD ? kSwW : kUhW, vPlaceW = iy ? kIyPlaceW : uwD ? kUwPlaceW : eh ? kEhPlaceW : ahD ? kAhPlaceW : aoD ? kAoPlaceW : swD ? kSwPlaceW : kUhPlaceW;
            // ... and its lead after a plosive (LINE_UH_STOP_LEAD_MS; < 0: LINE_AFTER_STOP_LEAD_MS).
            static const float kUhStopLeadMs = param("LINE_UH_STOP_LEAD_MS", -1.0f);
            if (uhD && !prevWasPause_ && prevWasPlosive_ && kUhStopLeadMs >= 0) preMs = kUhStopLeadMs;
            // ... and after a fricative, nasal or liquid (LINE_UH_LEAD_MS; 0: none, as before): the lips round in the
            // consonant ("should", "full", "look").
            static const float kUhLeadMs = param("LINE_UH_LEAD_MS", 0.0f);
            if (uhD && !prevWasPause_ && !prevWasPlosive_ && !isVowel(lastPh_) && kUhLeadMs > 0) preMs = kUhLeadMs;
            // (round-I ʌ agent, lab; 0 / 0: none, as before) the stressed ʌ's shape asked for this long before its start
            // after a pause (LINE_AH_PAUSE_LEAD_MS) and after a labial p b m f v (LINE_AH_LAB_LEAD_MS: the tongue is free
            // in the closure). Kokoro's word-initial ʌ ("other", "up", "us", "under") starts at F1 530-640 / F2
            // 1030-1130 Hz (male), puhuja's at 441-468 / 1253-1289, a glide out of the pause's ə (heard ə: 9 of the 19
            // initial errors); after m ("money", "month", "must") Kokoro's starts at F2 990-1130, puhuja's at 1220-1270
            // ("money" -> many, "mother" -> never).
            static const float kAhPauseLeadMs = param("LINE_AH_PAUSE_LEAD_MS", 0.0f), kAhLabLeadMs = param("LINE_AH_LAB_LEAD_MS", 0.0f);
            if (tok.ph == Ph::AH && prevWasPause_ && kAhPauseLeadMs > 0) preMs = kAhPauseLeadMs;
            if (tok.ph == Ph::AH && !prevWasPause_ && kAhLabLeadMs > 0 &&
                (lastPh_ == Ph::P || lastPh_ == Ph::B || lastPh_ == Ph::M || lastPh_ == Ph::F || lastPh_ == Ph::VV)) preMs = kAhLabLeadMs;
            // (R agent round F, lab; 0 / 1: as before) the ɹ after a plosive: its shape asked for from this long before
            // its start (the tongue bunched during the closure) and its demand's weight x LINE_R_STOP_W.
            static const float kRStopLeadMs = param("LINE_R_STOP_LEAD_MS", 0.0f), kRStopW = param("LINE_R_STOP_W", 1.0f);
            const bool rStop = tok.ph == Ph::RR && prevWasPlosive_;
            if (rStop && kRStopLeadMs > 0) preMs = kRStopLeadMs;
            if (rStop && kRStopW != 1.0f) ws *= kRStopW;
            // (Y agent round G, lab; 0 / 1: as before) the same for a j after a plosive ("cute", "beauty").
            static const float kJStopLeadMs = param("LINE_J_STOP_LEAD_MS", 30.0f), kJStopW = param("LINE_J_STOP_W", 4.0f);
            const bool jStop = tok.ph == Ph::J && prevWasPlosive_;
            if (jStop && kJStopLeadMs > 0) preMs = kJStopLeadMs;
            if (jStop && kJStopW != 1.0f) ws *= kJStopW;
            // (W agent round G, lab; 0 / 1 / -1: as before) the same for a w after a plosive (the lips round during the
            // closure: "twice", "quick") and after s (LINE_W_S_LEAD_MS: the s rounded, "swim"); the demand's end x dur.
            static const float kWStopLeadMs = param("LINE_W_STOP_LEAD_MS", 0.0f), kWStopW = param("LINE_W_STOP_W", 1.0f),
                               kWSLeadMs = param("LINE_W_S_LEAD_MS", 0.0f), kWStopKeyTo = param("LINE_W_STOP_KEY_TO", 1.2f);
            const bool wStop = tok.ph == Ph::W && prevWasPlosive_, wS = tok.ph == Ph::W && !prevWasPlosive_ && lastPh_ == Ph::S;
            if (wStop && kWStopLeadMs > 0) preMs = kWStopLeadMs;
            if (wS && kWSLeadMs > 0) preMs = kWSLeadMs;
            if ((wStop || wS) && kWStopW != 1.0f) ws *= kWStopW;
            // (W agent round I, lab; 1 / 1 / 0: as before) every other w: its demand's weight x LINE_W_GEN_W, x LINE_W_OPENV_W
            // more before an open vowel (ɑ aɪ ɔ oʊ: initial w before aɪ 77 %, ɑ 60 % errors, before ɪ iː ɜ 1-7 %), its lead
            // before the w after a vowel / liquid (LINE_W_GEN_LEAD_MS). (Its end: LINE_W_KEY_TO, which only this w uses.)
            static const float kWGenW = param("LINE_W_GEN_W", 1.0f), kWOpenVW = param("LINE_W_OPENV_W", 1.0f),
                               kWGenLeadMs = param("LINE_W_GEN_LEAD_MS", 0.0f);
            const bool wGen = tok.ph == Ph::W && !wStop && !wS;
            const bool openNext = next && (*next == Ph::AA || *next == Ph::A || *next == Ph::AO || *next == Ph::O || *next == Ph::OH);
            if (wGen && kWGenW != 1.0f) ws *= kWGenW;
            if (wGen && openNext && kWOpenVW != 1.0f) ws *= kWOpenVW;
            if (wGen && !prevWasPause_ && kWGenLeadMs > 0) preMs = kWGenLeadMs;
            // (W agent round I, lab; < 0: 0.3, as before) a vowel after a w asks for its own shape from this share of it: the
            // w's rise into the vowel (LINE_GLIDE_VOWEL_KEY moves only the unplanned keys).
            static const float kLineWVowelKey = param("LINE_W_VOWEL_KEY", -1.0f);
            const double vowelFrom = isVowel(tok.ph) && lastPh_ == Ph::W && kLineWVowelKey >= 0 ? kLineWVowelKey : 0.3;
            const double wKeyTo = (wStop || wS) && kWStopKeyTo >= 0 ? kWStopKeyTo : kLineWKeyTo;
            addShapeDemand(preMs > 0 ? t0 - preMs * ms : t0 + (er ? kLineErFrom : tok.ph == Ph::RR && kLineRKeyFrom >= 0 ? kLineRKeyFrom : vFrom >= 0 ? vFrom : vowelFrom) * dur, t0 + (er ? kLineErTo : tok.ph == Ph::W ? wKeyTo : tok.ph == Ph::RR && kLineRKeyTo >= 0 ? kLineRKeyTo : vTo >= 0 ? vTo : 0.7) * dur, sh, tok.ph, er ? ws * kLineErW : iy || uhD || uwD || ahD || aoD || swD ? ws * vW : ws, er ? std::max(preR, kErPlaceW) : iy || uhD || uwD || ahD || aoD || swD ? std::max(preR, vPlaceW) : preR);
            // (round-I uː agent) the rounder late key (LINE_UW_LATE_*, LINE_UW_END_*) asked for over the last share of
            // the uː from LINE_UW_LATE_AT (0.7).
            static const float kUwLateAt = param("LINE_UW_LATE_AT", 0.5f);
            if (uwD && uwLate) addShapeDemand(t0 + kUwLateAt * dur, t0 + dur, late, tok.ph, ws * vW, vPlaceW);
          }
          else {
            const bool open = tok.ph == Ph::AA || tok.ph == Ph::A;
            static const float kUhRAt = param("LINE_UH_R_AT", -1.0f);
            // (round I EH agent; < 0: as below) the ɛ's late key at LINE_EH_R_AT before an ɹ, LINE_EH_LATE_AT otherwise:
            // Kokoro's ɛ before r has its F3 down by the last third (male 2313, female 2187; puhuja 2396 / 2688).
            static const float kEhRAt = param("LINE_EH_R_AT", -1.0f), kEhLateAt = param("LINE_EH_LATE_AT", -1.0f);
            const float ehAt = next && *next == Ph::RR ? kEhRAt : kEhLateAt;
            const double at = tok.ph == Ph::EH && ehAt >= 0 ? ehAt : uhAlone && kUhRAt >= 0 && next && *next == Ph::RR ? kUhRAt : open && kRColourOpenAt >= 0 ? kRColourOpenAt : frontVowel(tok.ph) && kRColourFrontAt >= 0 ? kRColourFrontAt : kRColourAt;
            // (round-I ʌ agent, lab) LINE_AH_W weighs the ʌ's two demands here too (its late key, LINE_AH_OFF_SCHWA,
            // lands in this branch).
            static const float kAhW2 = param("LINE_AH_W", 1.0f);
            const float wv = tok.ph == Ph::AH && kAhW2 != 1.0f ? ws * kAhW2 : ws;
            addShapeDemand(t0 + 0.3 * dur, t0 + std::max(0.3, at - 0.15) * dur, sh, tok.ph, wv, preR);
            addShapeDemand(t0 + at * dur, t0 + std::min(1.0, at + 0.15) * dur, late, tok.ph, wv);
          }
        }
        lastKeyShape_ = late;
        haveLastKeyShape_ = true;
      }
      if (std::getenv("PUHUJA_LINE_TRACE"))
        std::fprintf(stderr, "linekey t %.3f ph %d shape %d xc %.2f ac %.2f aLip %.2f aPh %.2f aOr %.2f\n", (t0 + 0.5 * dur) / fs_, static_cast<int>(tok.ph), static_cast<int>(shapePh), sh.xc, sh.ac, sh.aLip, sh.aPharynx, sh.aOral);
    }
  }
  e.acMs = kBoundaryMs;
  // The tongue tip closes an n faster than the 20 ms boundary: with the
  // slow ramp a vowel-n-vowel n was half open (l-like) for most of its
  // ~45 ms, and Whisper heard "Any" as "elite" (2026-11-04).
  if (tract_ == Tract::Line && tok.ph == Ph::N && kLineNCloseMs >= 0) e.acMs = kLineNCloseMs;
  if (tract_ == Tract::Line && tok.ph == Ph::L && kLineLCloseMs >= 0) e.acMs = kLineLCloseMs;
  // ʃ ʒ: the tongue rises to the palate and leaves it over this long
  // (LINE_SH_ON_MS, LINE_SH_OFF_MS; 0: the 20 ms boundary). In 20 ms the slit
  // passed the frication's 0.22-0.12 cm^2 window in ~1 ms: the hiss switched
  // on and off; Kokoro's -tion ʃ fades out over 55-100 ms (Jyrki 2026-11-05:
  // "good if it ramps up and down as the tongue moves to the palate and back").
  if (tract_ == Tract::Line) {
    static const float shOn = param("LINE_SH_ON_MS", 25.0f), shOff = param("LINE_SH_OFF_MS", 25.0f);
    const float acMsOwn = e.acMs;  // (before the ʃ ʒ ramps: LINE_ZH_OFF_MS replaces LINE_SH_OFF_MS after a ʒ)
    if ((tok.ph == Ph::SH || tok.ph == Ph::ZH) && shOn > 0) e.acMs = shOn;
    if ((lastPh_ == Ph::SH || lastPh_ == Ph::ZH) && shOff > 0) e.acMs = std::max(e.acMs, shOff);
    // ʒ's own (ZH agent, round H; < 0: LINE_SH_ON_MS, LINE_SH_OFF_MS)
    static const float zhOn = param("LINE_ZH_ON_MS", -1.0f), zhOff = param("LINE_ZH_OFF_MS", 47.5f);
    if (tok.ph == Ph::ZH && !inAffricate_ && zhOn >= 0) e.acMs = zhOn;
    if (lastPh_ == Ph::ZH && lastZhOwn_ && zhOff >= 0) e.acMs = std::max(acMsOwn, zhOff);  // (not after a ʤ's ʒ)
    lastZhOwn_ = tok.ph == Ph::ZH && !inAffricate_;
    // The full-vowel ʃ's own (SH agent 10, round K; < 0: as above): not after a pause its slit closes over
    // LINE_SH_V_ON_MS (after a pause: LINE_SH_INIT_ON_MS, below); the vowel after it opens over LINE_SH_V_OFF_MS.
    static const float shVOn = param("LINE_SH_V_ON_MS", -1.0f), shVOff = param("LINE_SH_V_OFF_MS", -1.0f);
    if (shV && !prevWasPause_ && shVOn >= 0) e.acMs = shVOn;
    if (lastPh_ == Ph::SH && lastShV_ && shVOff >= 0) e.acMs = std::max(acMsOwn, shVOff);
    lastShV_ = shV;
    // s f θ into a voiced sound: the groove opens over LINE_FRIC_OFF_MS (0: the
    // 20 ms boundary). Kokoro's s in "save" fades over its last 30 ms before
    // the vowel starts; puhuja's hiss rose 12 dB in its last 10 ms with the
    // voice already on, a z (2026-11-05).
    static const float fricOff = param("LINE_FRIC_OFF_MS", 0.0f);
    if ((lastPh_ == Ph::S || lastPh_ == Ph::F || lastPh_ == Ph::TH) && fricOff > 0 && in.cls != Cls::Fricative && in.cls != Cls::Plosive) e.acMs = std::max(e.acMs, fricOff);
    // ... θ's own (LINE_TH_OFF_MS, TH agent7 round H; 0: as above): the slit opens into the next voiced sound over this.
    static const float thOff = param("LINE_TH_OFF_MS", 0.0f);
    if (lastPh_ == Ph::TH && thOff > 0 && in.cls != Cls::Fricative && in.cls != Cls::Plosive) e.acMs = thOff;
    // ... and its closing (LINE_TH_ON_MS; 0: the 20 ms boundary). With the glottis opened early (LINE_TH_AB_LEAD_MS) the
    // hiss rose 20 dB in 6 ms as the slit closed (snap 2.2 dB mean on ten medial-θ words, female; 35 ms: 1.5, 50 ms: 0.7).
    static const float thOn = param("LINE_TH_ON_MS", 35.0f);
    if (tok.ph == Ph::TH && thOn > 0 && !prevWasPause_) e.acMs = thOn;
    // v into a vowel or sonorant (V agent, round F; 0: the 20 ms boundary): the lip leaves the teeth over
    // LINE_V_OFF_MS. A quick opening from a hiss-less v is a b's release to Whisper (initial v: B 214 of 467 errors).
    static const float vOff = param("LINE_V_OFF_MS", 40.0f);
    if (lastPh_ == Ph::VV && vOff > 0 && in.cls != Cls::Fricative && in.cls != Cls::Plosive) e.acMs = std::max(e.acMs, vOff);
    // ð's slit closes over LINE_DH_ON_MS and opens into the next sound over LINE_DH_OFF_MS (DH agent7, round H; 0: the
    // 20 ms boundary), as the v's LINE_V_OFF_MS: a quick opening is a stop's release to Whisper (ð heard as b d).
    static const float dhOn = param("LINE_DH_ON_MS", 0.0f), dhOff = param("LINE_DH_OFF_MS", 0.0f);
    if (tok.ph == Ph::DH && dhOn > 0) e.acMs = dhOn;
    if (lastPh_ == Ph::DH && dhOff > 0 && in.cls != Cls::Fricative && in.cls != Cls::Plosive) e.acMs = dhOff;
    // (W agent round I, lab; < 0: the 20 ms boundary, as before) the lips open from a w into the next sound over LINE_W_OFF_MS
    // and close into a w over LINE_W_ON_MS: the w's F2 and F3 switched on in one step at the vowel (spectrogram "wine",
    // "went": a dark F1-only murmur, then the vowel's formants at once, an l's release; Kokoro's w rises over ~100 ms).
    static const float wOff = param("LINE_W_OFF_MS", -1.0f);
    if (lastPh_ == Ph::W && wOff >= 0 && in.cls != Cls::Fricative && in.cls != Cls::Plosive) e.acMs = wOff;
    if (tok.ph == Ph::W && kLineWOnMs >= 0) e.acMs = kLineWOnMs;
  }
  e.ag = aero.ag;
  if (in.cls == Cls::Vowel) e.ag *= tok.shape.press;  // voice quality by prosody
  {  // ŋ's glottis (cm^2; LINE_NG_AG, lab knob 2026-11-11, NG agent 3; < 0: the voicing rest)
    static const float kLineNgAg = param("LINE_NG_AG", -1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::NG && kLineNgAg >= 0) e.ag = kLineNgAg;
  }
  {  // a word-final ʒ's glottis (cm^2; ZH agent, round H; < 0: LINE_AG_ZH): English final voiced fricatives devoice
     // ("garage" -> wrong / broad, "beige" -> big: a voiced murmur without hiss before the pause)
    static const float kLineZhAgFinal = param("LINE_ZH_AG_FINAL", 0.287f);
    if (tract_ == Tract::Line && tok.ph == Ph::ZH && !inAffricate_ && kLineZhAgFinal >= 0 && (!next || *next == Ph::Stop)) e.ag = kLineZhAgFinal;
  }
  e.agMs = tract_ == Tract::Line && in.cls == Cls::Fricative && aero.ag > kAgBreathy ? kLineAbductMs : kGlottalMs;
  if (tract_ == Tract::Line && tok.ph == Ph::Z && kZAbductMs >= 0 && aero.ag > kAgBreathy) e.agMs = kZAbductMs;
  // θ's own abduction (LINE_TH_ABDUCT_MS, TH agent7 round H; < 0: LINE_ABDUCT_MS): in "author" the glottis reached the
  // θ's opening 30 ms into its 59 ms and began closing 15 ms before its end, the voice running through half the θ
  // ("author" -> "over": V the commonest medial-θ confusion); Kokoro's θ is voiceless from its first ms.
  { static const float kThAbductMs = param("LINE_TH_ABDUCT_MS", -1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::TH && kThAbductMs >= 0 && aero.ag > kAgBreathy) e.agMs = kThAbductMs; }
  { static const float kDhAbductMs = param("LINE_DH_ABDUCT_MS", -1.0f);  // ð's glottal opening time (< 0: as every fricative; DH agent7)
    if (tract_ == Tract::Line && tok.ph == Ph::DH && kDhAbductMs >= 0) e.agMs = kDhAbductMs; }
  e.aspGain = in.cls == Cls::H ? hAspirationGain_ * kHGain : kAspirationGain * kAspirationTiltGain;
  if (lineHAsp_ && in.cls != Cls::H) {  // after a voiceless h: the noise gain back
    e.lineAspMul = 1.0f;
    lineHAsp_ = false;
  }
  // θ's glottal hiss x LINE_TH_ASP (TH agent7, round H; 1: off): a broadband breath noise under the dental slit's
  // narrow band (puhuja's θ 0.5-2 kHz 12-15 dB under its 2-8 kHz, Kokoro's ~8: flatter). Reset at the next sound.
  { static const float kLineThAsp = param("LINE_TH_ASP", 1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::TH && kLineThAsp != 1.0f) { e.lineAspMul = kLineThAsp; lineHAsp_ = true; } }
  if (lineHSpec_ && in.cls != Cls::H) {  // after an h with its own breath (LINE_H_GFC ...): back to the shared values
    e.lineGlotFc = 1.0f;
    if (kLineHBreath > 0) e.lineHBreath = 0.0f;
    if (kLineHLpr > 0) e.linePuffLpr = 0.0f;
    if (kLineHPress != 1.0f && e.press < 0) { e.press = 1.0f; e.pressMs = 20; }
    lineHSpec_ = false;
  }
  e.creak = 0;  // a creak lasts until the next sound
  // After an aspirated release the glottis is already closing slowly into
  // this sound; a quick move here would cut the aspiration short.
  if (aspirating_) {
    // ... except an h: it is a sound of its own, not the stop's tail. Run on
    // the t's fading high-passed aspiration, the h of "at home" was 30 dB
    // under the vowel in its middle and bright at its end, heard as an f ("at
    // phone"; Jyrki 2026-11-07: "it continues from the t, athhhome"). Kokoro's
    // h there: a steady breath 16 dB under the vowel, centroid 1.1-1.3 kHz.
    static const float kLineHOwnAfterStop = param("LINE_H_OWN_AFTER_STOP", 0.0f);  // (0: Jyrki chose the h of the old path at 0.6 length, "let's make that the default then own60")
    if (!(tract_ == Tract::Line && in.cls == Cls::H && kLineHOwnAfterStop > 0.5f)) e.ag = -1;
    aspirating_ = false;
  }
  e.an = aero.an;
  e.anMs = kBoundaryMs;
  // A phrase-final ŋ's breath eases to LINE_NG_FINAL_PRESS over the ŋ (lab knob 2026-11-11, NG agent 3; 1: as
  // before): Kokoro's final ŋ fades ~10 dB through its murmur, puhuja's held its level to the end.
  {
    static const float kLineNgFinalPress = param("LINE_NG_FINAL_PRESS", 1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::NG && (!next || *next == Ph::Stop) && kLineNgFinalPress != 1.0f && e.press < 0) {
      e.press = kLineNgFinalPress;
      e.pressMs = static_cast<float>(dur / ms);
    }
  }
  // n's breath (round J, N agent 9, lab knobs; 1: as before): eases to LINE_N_PRESS over the n (before a pause
  // LINE_N_FINAL_PRESS, < 0: the same). Kokoro's word-final murmur sits 6-9 dB under the vowel ("Thin", "Don":
  // adam -6..-14, puhuja 0..-4 dB), and a murmur as loud as the vowel with a flat F2 is heard as ŋ (final n 11 %
  // wrong, ŋ 347 of 626).
  {
    static const float kLineNPress = param("LINE_N_PRESS", 1.0f), kLineNFinalPress = param("LINE_N_FINAL_PRESS", -1.0f);
    const bool fin = !next || *next == Ph::Stop;
    // ... a word-final n after an unstressed vowel (Shape::nctx 2; N agent13 round L lab knob, < 0: as above) eases to
    // LINE_FN_PRESS: the -ən / -ɪn murmur is 1.5-3 dB under its vowel in ElevenLabs' Kennedy readings (n -7..-11 dB re
    // the local peak, 33 / 32 / 26 tokens), in puhuja 2.7 dB over it (-3.1 vs -5.8).
    static const float kLineFnPress = param("LINE_FN_PRESS", -1.0f);
    const float p = (tok.shape.nctx & 2) && kLineFnPress >= 0 ? kLineFnPress : fin && kLineNFinalPress >= 0 ? kLineNFinalPress : kLineNPress;
    if (tract_ == Tract::Line && tok.ph == Ph::N && p != 1.0f && e.press < 0) {
      e.press = p;
      e.pressMs = static_cast<float>(dur / ms);
    }
  }
  // ŋ's velum port x LINE_NG_AN (lab knob 2026-11-11, NG agent 3; 1: as m n).
  {
    static const float kLineNgAn = param("LINE_NG_AN", 1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::NG) e.an *= kLineNgAn;
  }
  // An m before a non-labial obstruent (d t k g s z ʃ θ ...) shuts its velum over its last LINE_M_VEL_LEAD_MS (round J,
  // M agent; 0: off, as before: the port closed with the next sound's own event, after the tongue had closed at the
  // alveolar ridge, so the d of "armed" began as an n's murmur: m before d 46 % wrong, "armed" -> darned).
  {
    static const float kMVelLeadMs = param("LINE_M_VEL_LEAD_MS", 0.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::M && kMVelLeadMs > 0 && next) {
      const Cls nc = info(*next).cls;
      const bool labialNext = *next == Ph::P || *next == Ph::B || *next == Ph::F || *next == Ph::VV;
      if ((nc == Cls::Plosive || nc == Cls::Fricative) && !labialNext) {
        const float lead = std::min(kMVelLeadMs, 0.8f * static_cast<float>(dur / ms));
        Event shut{e.t + dur - lead * ms};
        shut.an = 0.0f;
        shut.anMs = lead;
        addEventAt(shut);
      }
    }
  }
  // m's own port x LINE_M_PORT_SCALE (round J, M agent; 1: as before).
  {
    static const float kMPort = param("LINE_M_PORT_SCALE", 1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::M && kMPort != 1.0f) e.an *= kMPort;
  }
  // n's own port (round E, N agent): 1.5 cm^2 against a physiological
  // 0.2-0.5, and the murmur sits 5-9 dB over Kokoro's. LINE_N_PORT_SCALE
  // scales it (1: as before).
  {
    static const float kNPort = param("LINE_N_PORT_SCALE", 1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::N && kNPort != 1.0f) e.an *= kNPort;
    // ... a word-final n after an unstressed vowel x LINE_FN_PORT as well (Shape::nctx 2; N agent13 round L; 1: as before).
    static const float kFnPort = param("LINE_FN_PORT", 0.6f);
    if (tract_ == Tract::Line && tok.ph == Ph::N && (tok.shape.nctx & 2) && kFnPort != 1.0f) e.an *= kFnPort;
  }
  // English nasalizes a vowel before a nasal through most of it: the velum
  // opens part way during the vowel ("man", "and", "on"); on the line the
  // port was shut until the nasal's own event (2026-11-02; set after the aero default, which had overwritten it).
  if (tract_ == Tract::Line && isVowel(tok.ph) && next && info(*next).cls == Cls::Nasal) {
    // (before ŋ: LINE_NG_ANTICIP instead, lab knob 2026-11-11, NG agent 3; < 0: LINE_ANTICIP_NASAL)
    static const float kLineNgAnticip = param("LINE_NG_ANTICIP", -1.0f);
    e.an = kAnNasal * (*next == Ph::NG && kLineNgAnticip >= 0 ? kLineNgAnticip : kLineAnticipNasal);
    e.anMs = static_cast<float>(dur / ms);
    // Before an n (round E, N agent): the port in the vowel is 0.63 cm^2,
    // 3-6x a physiological 0.05-0.2, and it pulls F2 (the n's place cue)
    // under a nasal pole near 800 Hz. LINE_ANTICIP_N_SCALE scales the
    // vowel's port before n; LINE_ANTICIP_N_LATE (share of the vowel)
    // keeps the port shut until that point, then opens it over the rest
    // (Krakow 1993: the port opens late). Defaults 1 / 0: as before.
    static const float kAnN = param("LINE_ANTICIP_N_SCALE", 1.0f), kAnNLate = param("LINE_ANTICIP_N_LATE", 0.0f);
    // The ɛ before any nasal (round F EH agent 2026-11-11 lab knobs; -1 / 0: as above): its port x LINE_EH_ANTICIP_SCALE
    // (< 0: as every vowel), shut until LINE_EH_NASAL_LATE of the vowel and opened over the rest. A ɛ nasalized
    // over its whole 139 ms is the American [ɛə̃n] of "and": ɛ before a nasal was heard as æ 352 times ("send" -> sand).
    static const float kEhAnScale = param("LINE_EH_ANTICIP_SCALE", 1.3f), kEhAnLate = param("LINE_EH_NASAL_LATE", 0.85f);
    const bool ehLate = tok.ph == Ph::EH && (kEhAnScale >= 0 || kEhAnLate > 0);
    // ... and before an m (round J, M agent; 1 / 0: as before): LINE_ANTICIP_M_SCALE, LINE_ANTICIP_M_LATE as the n's.
    static const float kAnM = param("LINE_ANTICIP_M_SCALE", 1.0f), kAnMLate = param("LINE_ANTICIP_M_LATE", 0.0f);
    const bool mAnt = !ehLate && *next == Ph::M && (kAnM != 1.0f || kAnMLate > 0);
    // ... and the unstressed vowel before its word's final n (Shape::nctx 1; N agent13 round L lab knobs, < 0 / 0: as
    // above): its port x LINE_FN_ANTICIP, shut until LINE_FN_ANTICIP_LATE of the vowel. Over the whole vowel the port
    // put "Gavin"'s ɪ at F2 2170 / F3 3500 Hz (male; port 0: 1860 / 2250; before s 1745 / 2310; Kokoro 1500-1750 /
    // 2400-2600), the very ɪ puhuja says before ŋ ("Driving" 2156 / 3140): heard as -ing. Alone (0) it did not move the
    // n lab (56 -> 54 errors); with the -ɪn said -ən (EN_FN_IH_SCHWA 1) 34.
    static const float kFnAn = param("LINE_FN_ANTICIP", 0.0f), kFnAnLate = param("LINE_FN_ANTICIP_LATE", 0.0f);
    const bool fnAnt = !ehLate && (tok.shape.nctx & 1) && *next == Ph::N && (kFnAn >= 0 || kFnAnLate > 0);
    if (fnAnt) {
      e.an = kAnNasal * kLineAnticipNasal * (kFnAn >= 0 ? kFnAn : kAnN);
      const float lateShare = kFnAnLate > 0 ? kFnAnLate : kAnNLate;
      if (lateShare > 0) {
        Event late{e.t + lateShare * dur};
        late.an = e.an;
        late.anMs = static_cast<float>((1.0f - lateShare) * dur / ms);
        addEventAt(late);
        e.an = aero.an;
        e.anMs = kBoundaryMs;
      }
    } else if (mAnt) {
      e.an = kAnNasal * kLineAnticipNasal * kAnM;
      if (kAnMLate > 0) {
        Event late{e.t + kAnMLate * dur};
        late.an = e.an;
        late.anMs = static_cast<float>((1.0f - kAnMLate) * dur / ms);
        addEventAt(late);
        e.an = aero.an;
        e.anMs = kBoundaryMs;
      }
    } else if (ehLate || (*next == Ph::N && (kAnN != 1.0f || kAnNLate > 0))) {
      const float lateShare = ehLate ? kEhAnLate : kAnNLate;
      if (ehLate) {
        if (kEhAnScale >= 0) e.an *= kEhAnScale;
      } else {
        e.an = kAnNasal * kLineAnticipNasal * kAnN;
      }
      if (lateShare > 0) {
        Event late{e.t + lateShare * dur};
        late.an = e.an;
        late.anMs = static_cast<float>((1.0f - lateShare) * dur / ms);
        addEventAt(late);
        e.an = aero.an;
        e.anMs = kBoundaryMs;
      }
    }
  }
  // The ʌ's velum port before a nasal x LINE_AH_ANTICIP (round-F ʌ agent; 1: as before): ʌ before a nasal 23 % wrong,
  // heard as æ 63 / ɛ 62 ("funds" -> fans, "sons" -> sense): the port's pole near 800 Hz pulls F2 / F1.
  {
    static const float kAhAn = param("LINE_AH_ANTICIP", 1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::AH && kAhAn != 1.0f && next && info(*next).cls == Cls::Nasal) e.an *= kAhAn;
  }
  // The release of a nasal into a vowel: how fast the mouth opens and the
  // velum closes, and how long the velum stays open into the vowel.
  Event velLag{0}, closeFricBack{0};
  bool hasVelLag = false, hasCloseFricBack = false;
  if (tract_ == Tract::Line && prevWasNasal_ && isVowel(tok.ph)) {
    const float velMs = nasalKnob(lastPh_, kLineNasalVelMs, kLineNVelMs);
    const float velLagMs = nasalKnob(lastPh_, kLineNasalVelLagMs, kLineNVelLagMs);
    e.acMs = nasalKnob(lastPh_, kLineNasalRelMs, kLineNRelMs);
    e.anMs = velMs;
    if (kLineCloseFric != 1.0f) {  // (LINE_CLOSE_FRIC: the oral closure opening into the vowel keeps the closure's gain, then the vowel's own)
      closeFricBack = Event{e.t + (e.acMs + 10.0f) * ms};
      closeFricBack.lineFricMul = e.lineFricMul < 0 ? 1.0f : e.lineFricMul;
      e.lineFricMul = kLineCloseFric;
      hasCloseFricBack = true;
    }
    if (velLagMs > 0 && e.an >= 0) {
      velLag = Event{e.t + velLagMs * ms};
      velLag.an = e.an;
      velLag.anMs = velMs;
      e.an = -1;
      hasVelLag = true;
    }
  }
  // Z-FINAL agent13 round L (lab; 0: off): a word-final z after a nasal keeps the velum port open (LINE_NZ_VEL_A x the
  // nasals' port) for its first LINE_NZ_VEL_MS, closing over that time: the mouth pressure leaks through the nose, the
  // folds keep going under the rising hiss. Puhuja's "pens" "means": the voice bar -38 dB re the word's peak over the
  // hiss's first 20 ms, overlap 0 ms; Kokoro -18..-20 dB, the voice and the hiss overlap 20-30 ms.
  {
    static const float kNzVelMs = param("LINE_NZ_VEL_MS", 0.0f), kNzVelA = param("LINE_NZ_VEL_A", 0.5f);
    // ... and LINE_VZ_VEL_MS / _A the same after a vowel (a sink that keeps the folds going, not a nasal: the vowel's
    // port is shut, so the leak opens with the z).
    static const float kVzVelMs = param("LINE_VZ_VEL_MS", 0.0f), kVzVelA = param("LINE_VZ_VEL_A", 0.2f);
    const bool fz = tract_ == Tract::Line && tok.ph == Ph::Z && (!next || *next == Ph::Stop);
    const float velMs = fz && prevWasNasal_ ? kNzVelMs : fz && prevWasVowel_ ? kVzVelMs : 0.0f;
    if (velMs > 0) {
      e.an = kAnNasal * (prevWasNasal_ ? kNzVelA : kVzVelA);
      e.anMs = prevWasNasal_ ? kBoundaryMs : 10.0f;
      Event shut{e.t + velMs * ms};
      shut.an = 0.0f;
      shut.anMs = velMs;
      addEventAt(shut);
    }
  }
  e.speech = in.cls == Cls::Stop ? 0.0f : 1.0f;
  if (prevWasPause_ && in.cls != Cls::Vowel && in.cls != Cls::Stop) {
    // After a pause the tongue and lips take their place before the voice
    // starts (no vowel-like onset before "they"); the glottis opens on time.
    Event pre{t0 - kPreposeMs * ms};
    pre.ac = e.ac;
    pre.acMs = tok.ph == Ph::W && tract_ == Tract::Line && kLineWOnMs >= 0 ? kLineWOnMs : kBoundaryMs;  // (LINE_W_ON_MS)
    // (SH agent 10, round K; < 0: as before) a full-vowel ʃ's slit closes over LINE_SH_INIT_ON_MS here
    static const float kShInitOnMs = param("LINE_SH_INIT_ON_MS", -1.0f);
    if (tract_ == Tract::Line && kShInitOnMs >= 0 && tok.ph == Ph::SH && !inAffricate_ && next && isVowel(*next) && *next != Ph::Schwa)
      pre.acMs = kShInitOnMs;
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
          static const float kZOnsetDip = param("LINE_Z_ONSET_DIP", -1.0f);  // after a z (< 0: kOnsetDipVoiced; lab 2026-11-11)
          onset[kF0Mul] = p[kF0Mul] * (voicelessBefore ? kOnsetRiseVoiceless : lastPh_ == Ph::Z && kZOnsetDip > 0 ? kZOnsetDip : kOnsetDipVoiced);
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
      { static const float kThNoiseOn = param("LINE_TH_NOISE_ON_MS", -1.0f); if (tok.ph == Ph::TH && kThNoiseOn >= 0) e.noiseFadeMs = kThNoiseOn; }  // θ's own fade-in (< 0: 12 ms; lab 2026-11-09)
      { static const float kDhNoiseOn = param("LINE_DH_NOISE_ON_MS", -1.0f); if (tok.ph == Ph::DH && kDhNoiseOn >= 0) e.noiseFadeMs = kDhNoiseOn; }  // ð's own (< 0: 12 ms; DH agent7)
      e.noiseGain = in.noise * fricationGain_;
      // The per-phoneme level scales the heard noise, after the turbulence
      // ceiling: scaling the gain before it hardly changes a saturated θ.
      e.noisePost = tok.shape.level;
      e.noiseTract = fricTract(tok.ph).share;
      { static const float kDhTract = param("LINE_DH_TRACT", -1.0f); if (tok.ph == Ph::DH && kDhTract >= 0) e.noiseTract = kDhTract; }  // ð's noise share through the tract (< 0: 0.1; DH agent7)
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
        // LINE_HJ_AC / LINE_HJ_LEVEL (HH agent 2026-11-10: h before j is 51 % wrong in the dictionary, 18 of 37 deleted).
        static const float kLineHjAc = param("LINE_HJ_AC", kPalatalHAc), kLineHjLevel = param("LINE_HJ_LEVEL", kPalatalHLevel);
        e.ac = tract_ == Tract::Line ? kLineHjAc : kPalatalHAc;
        e.setNoise = true;
        e.shape = kPalatalHShape;
        e.noiseGain = fricationGain_;
        e.noisePost = tok.shape.level * (tract_ == Tract::Line ? kLineHjLevel : kPalatalHLevel);
        e.noiseTract = fricTract(Ph::SH).share;
        e.noiseTractGain = kBurstTractNorm[fricTract(Ph::SH).place];
      }
      const bool voicedCtx = prevVoiced_ || (tract_ == Tract::Line && (kLineHVoicedCtx >= 2 || (kLineHVoicedCtx >= 1 && !prevWasPause_)));  // (LINE_H_VOICED_CTX)
      bool voicedH = voicedCtx && next && isSonorant(*next) && !beforeJ;
      // After a voiceless consonant ("torch has", "this house") the h keeps the breath going with the glottis this open
      // (LINE_H_AFTER_VLESS_AG cm^2; < 0: the voiced h as everywhere). Jyrki 2026-11-11: "\"torch has\" has a strange
      // sound between the words, but \"torch\" and \"has\" are separately ok": the ʧ's hiss (peak 0.106) dropped into a
      // breathy-voiced h (glottis 0.30 -> 0.17 cm^2) at 0.006-0.010 while the folds restarted.
      static const float kLineHAfterVlessAg = param("LINE_H_AFTER_VLESS_AG", -1.0f);
      const bool afterVless = tract_ == Tract::Line && kLineHAfterVlessAg >= 0 && !prevVoiced_ && !prevWasPause_;
      if (voicedH && afterVless) {
        e.ag = kLineHAfterVlessAg;
      } else if (voicedH) {
        e.ag = voicedHAg_;
        e.aspGain *= voicedHNoise_;
        if (tract_ == Tract::Line && prevWasPause_) {  // (LINE_H_VOICED_CTX 2: the kick starts the folds, as a phrase-initial vowel's)
          if (kLineHPauseAg >= 0) e.ag = kLineHPauseAg;
          if (kLineOnsetKickAg > 0 && kLineOnsetKickMs > 0 && kLineHPauseKick > 0) {
            Event kick{e.t};
            kick.ag = kLineOnsetKickAg;
            kick.agMs = kLineOnsetKickMs;
            addEventAt(kick);
            Event adduct{e.t + kLineOnsetKickMs * ms};
            adduct.ag = e.ag;
            adduct.agMs = std::max(kLineHRiseMs, 5.0f);
            e.ag = -1;
            addEventAt(adduct);
          }
        }
        // Its voicing is weaker still: with the vowel's shape and level it was
        // an approximant ("hand" as "land").
        for (Keyframe& kf : k_)
          if (kf.t >= hKeyframeT_ - 1 && kf.t <= hKeyframeT_ + 1) kf.p[kAV] *= kVoicedHVoice;
        p[kAV] *= kVoicedHVoice;
        if (tract_ == Tract::Line && kLineVHAspMul != 1.0f) {
          e.lineAspMul = kLineVHAspMul;
          lineHAsp_ = true;
        }
        if (tract_ == Tract::Line && prevWasPause_ && kLineHPauseAsp >= 0) {  // (LINE_H_PAUSE_ASP)
          e.lineAspMul = kLineHPauseAsp;
          lineHAsp_ = true;
        }
      }
      if (tract_ == Tract::Line && !voicedH && !beforeJ) {
        e.ag = kLineHAg;
        e.agMs = kLineHRiseMs;
        e.lineAspMul = kLineHAspMul;
        lineHAsp_ = true;
      }
      const float hGfc = (prevWasPause_ || kLineHGfcMed < 0 ? kLineHGfc : kLineHGfcMed) * (female_ ? kLineHGfcF : 1.0f);
      if (tract_ == Tract::Line && (hGfc != 1.0f || kLineHBreath > 0 || kLineHLpr > 0 || kLineHPress != 1.0f)) {
        if (hGfc != 1.0f) e.lineGlotFc = hGfc;
        if (kLineHBreath > 0) e.lineHBreath = kLineHBreath;
        if (kLineHLpr > 0) e.linePuffLpr = kLineHLpr;
        if (kLineHPress != 1.0f) { e.press = kLineHPress; e.pressMs = 20; }
        lineHSpec_ = true;
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
      if (tract_ == Tract::Line) {
        // The mouth relaxes open in a pause: left at a final s's 0.05 cm^2, the
        // next word's vowel opened from a constriction like a stop's release
        // (Whisper heard "at", "an", "is" as "Dad", "Dave", 2026-11-02). It
        // relaxes only once the breath has stopped: opening the groove with
        // the glottis still apart and the lungs still pushing let out a puff
        // 10 dB over the ʃ at the end of "abolish" (2026-11-03).
        e.ac = -1;
        // Once per pause, at its own time: a pause comes as several tokens
        // (15 ms each at the start of the text), and each queued its relax
        // 80 ms ahead; since events start in order, every later event, the
        // next word's lip closure, glottal opening and breath included, was
        // held back behind it: the initial f of "forget" began 70-95 ms late
        // and sounded for ~28 ms of its 93 (2026-11-03).
        if (!prevWasPause_) relaxAt_ = e.t + kLinePausePressMs * ms;
        // After a final z (lab knobs 2026-11-11): the groove eases open from the pause's start to LINE_Z_FINAL_OPEN_A
        // cm^2 over LINE_Z_FINAL_OPEN_MS (0: off; the slit held 0.044 cm^2 for 70 ms into the pause, a voiceless hiss
        // twice the z's own length), and the glottis goes to LINE_Z_FINAL_AG over 30 ms (< 0: unchanged).
        if (!prevWasPause_ && lastPh_ == Ph::Z) {
          static const float zOpenMs = param("LINE_Z_FINAL_OPEN_MS", 0.0f), zOpenA = param("LINE_Z_FINAL_OPEN_A", 0.3f),
                             zFinalAg = param("LINE_Z_FINAL_AG", -1.0f);
          if (zOpenMs > 0 || zFinalAg >= 0) {
            Event op{e.t};
            if (zOpenMs > 0) { op.ac = zOpenA; op.acMs = zOpenMs; }
            if (zFinalAg >= 0) { op.ag = zFinalAg; op.agMs = 30; }
            addEventAt(op);
          }
        }
        // After a final ə (AX agent 2026-11-11, lab knobs; 0: off): the glottis eases open to LINE_SCHWA_END_AG
        // (share of the way from voicing to kAgOpen) over LINE_SCHWA_END_MS from LINE_SCHWA_END_LEAD_MS before the
        // pause, a breathy fade: Kokoro's final ə falls ~20 dB over 50-60 ms, puhuja's held its level and was cut in
        // ~30 ms.
        if (!prevWasPause_ && lastPh_ == Ph::Schwa) {
          static const float kSchwaEndMs = param("LINE_SCHWA_END_MS", 0.0f), kSchwaEndAg = param("LINE_SCHWA_END_AG", 0.5f),
                             kSchwaEndLead = param("LINE_SCHWA_END_LEAD_MS", 40.0f);
          if (kSchwaEndMs > 0) {
            Event op{e.t - kSchwaEndLead * ms};
            op.ag = kAgVoice + kSchwaEndAg * (kAgOpen - kAgVoice);
            op.agMs = kSchwaEndMs;
            addEventAt(op);
          }
        }
        if (!prevWasPause_) pauseEventT_ = e.t;
        const bool relaxNow = relaxAt_ >= 0 && relaxAt_ < t0 + dur;
        pendingRelax_ = Event{relaxNow ? relaxAt_ : e.t};
        pendingRelax_.ac = kAcOpen;
        pendingRelax_.acMs = 100;
        pendingRelax_.ag = pendingRelax_.an = -1;
        pendingRelax_.speech = -1;
        pendingRelax_.press = -1;
        pendingRelax_.aspGain = -1;
        pendingRelax_.expand = -1;
        pendingRelax_.inhaleMs = 0;
        pendingRelax_.pop = -1;
        hasPendingRelax_ = relaxNow;
        if (relaxNow) relaxAt_ = -1;
      }
      e.inhaleMs = nextInhaleMs_;
      e.inhaleSnatched = nextInhaleSnatched_;
      nextInhaleMs_ = 0;
      nextInhaleSnatched_ = false;
      e.agMs = tract_ == Tract::Line ? kLineStopGlottalMs : kStopGlottalMs;
      // After a final voiceless plosive the glottis stays open through the
      // release and its aspiration before it closes for the pause (Kokoro
      // adam's final k: 60-130 ms of release noise; closing with the release
      // left a 10 ms click, "make", "take" 2026-11-02).
      if (tract_ == Tract::Line && prevWasPlosive_ && !prevVoiced_) e.t += kLineFinalAspMs * ms;
      // ... and after a voiceless fricative the glottis stays open while the
      // groove relaxes: closing it inside the ʃ of "abolish" drove a last
      // puff of flow through the narrowing folds, a sharp s-like tail 10 dB
      // louder than the ʃ itself ("abolisz", 2026-11-03). The recorded ʃ
      // decays smoothly.
      else if (tract_ == Tract::Line && !prevWasPlosive_ && !prevVoiced_ && !prevWasPause_) {
        static const float kLineFinalThMs = param("LINE_FINAL_TH_MS", -1.0f);  // ... the wait after a final θ (< 0: LINE_FINAL_FRIC_MS; lab 2026-11-09)
        e.t += (lastPh_ == Ph::TH && kLineFinalThMs >= 0 ? kLineFinalThMs : kLineFinalFricMs) * ms;
      }
      // At a pause the lungs stop pushing (press -> 0 over kLinePausePressMs)
      // and the glottis is left where it is after a voiceless sound: closing
      // it from the open s/ʃ position passed through the breathy zone, where
      // the string folds restart, and put a voiced grunt (flow 460 cm^3/s) or,
      // closed fast, a click at the end of every final s, ʃ, f, t. With the
      // breath gone the hiss decays as the groove relaxes (2026-11-03).
      if (tract_ == Tract::Line) {
        e.press = 0.0f;
        e.pressMs = kLinePausePressMs;
        if (!prevVoiced_ && !prevWasPause_) e.ag = -1;
        // The pause's later tokens (15 ms each) change nothing: each one
        // re-sent the closed glottis, the zero pressure and the silence, and
        // so undid the next phrase's breath and voicing queued inside the
        // pause (the initial n of "Name." stayed 45 dB down, 2026-11-04).
        if (prevWasPause_) e.ag = e.an = e.press = e.speech = -1;
        // After a final nasal the velum stays open: the mouth is still shut,
        // and closing the port sealed the airway, cutting the n off within
        // 10 ms ("revolution." ended abruptly, Jyrki 2026-11-03; the readers'
        // final n fades over ~60 ms). The breath out through the nose is
        // what the speaker does anyway; the next phrase's sound sets it.
        if (prevWasNasal_ && kLineFinalNasalOpen > 0) e.anMs = kLineFinalNasalOpen;
        // ... and after a final ŋ the tongue leaves the velum at the pause, to LINE_NG_FINAL_REL_A cm^2 over
        // LINE_NG_FINAL_REL_MS (lab knobs 2026-11-11, NG agent 3; 0: shut until the relax, 80 ms after the breath
        // stopped): Kokoro's final ŋ ("sing") ends in a faint release, F2 2300 / F3 3450 25-30 dB under the vowel.
        static const float kLineNgFinalRelMs = param("LINE_NG_FINAL_REL_MS", 0.0f), kLineNgFinalRelA = param("LINE_NG_FINAL_REL_A", 0.5f);
        if (prevWasNasal_ && lastPh_ == Ph::NG && kLineNgFinalRelMs > 0 && !prevWasPause_) {
          e.ac = kLineNgFinalRelA;
          e.acMs = kLineNgFinalRelMs;
        }
        // ... and after a final n the tongue tip leaves the ridge at the pause, to LINE_N_FINAL_REL_A cm^2 over
        // LINE_N_FINAL_REL_MS (N agent13 round L lab knobs; 0: as before): Kokoro's final n ("Linen", "Ryan") ends
        // in an audible release, breath noise 13-23 dB under the vowel, the alveolar cue a ŋ does not have. N lab: 30 ms
        // alone 56 -> 42 n errors (placebo sd 5.3); with EN_FN_IH_SCHWA 1, LINE_FN_ANTICIP 0, LINE_FN_PORT 0.6: 16 (heard
        // ŋ 40.5 -> 1), words1000 1478 -> 1496, the lab's level term +2.6σ (the brighter tail).
        static const float kLineNFinalRelMs = param("LINE_N_FINAL_REL_MS", 30.0f), kLineNFinalRelA = param("LINE_N_FINAL_REL_A", 0.5f);
        if (prevWasNasal_ && lastPh_ == Ph::N && kLineNFinalRelMs > 0 && !prevWasPause_) {
          e.ac = kLineNFinalRelA;
          e.acMs = kLineNFinalRelMs;
        }
      }
      syllable_ = 0;
      resolveRanges(nullptr, -1);  // no vowel follows: they hold the last one's
      rangePrevT_ = -1;
      addKeyframe(p, t0 + 0.25 * dur, true);
      addKeyframe(p, t0 + 0.75 * dur, true);
      break;
    case Cls::Plosive:
      break;
  }
  if (tract_ == Tract::Line && !prevWasPause_ && !prevVoiced_ && !prevWasPlosive_ && e.ag >= 0 &&
      (in.cls == Cls::Vowel || in.cls == Cls::Nasal || in.cls == Cls::Liquid || in.cls == Cls::Glide || in.cls == Cls::Trill)) {
    // After a voiceless fricative the folds adduct slowly, through the
    // breathy zone: snapped from the open s position to the voicing rest in
    // 15 ms the string folds hardly restarted ("see", "say": flow 10-20
    // cm^3/s for the whole vowel, 10 dB under a vowel after a pause), while
    // the h -> vowel path, which ends breathy, voiced at once (2026-11-02).
    e.agMs = kLineAdductMs;
    if (kLineHVowelAgMs > 0 && lastPh_ == Ph::H && isVowel(tok.ph)) e.agMs = kLineHVowelAgMs;  // (LINE_H_VOWEL_AG_MS; the h path lands here, 30 ms)
    // After a θ (TH agent7, round H; < 0: as for the others): the folds adduct over LINE_TH_ADDUCT_MS, starting
    // LINE_TH_ADDUCT_LEAD_MS before the opening (below; LINE_ADDUCT_LEAD_MS 15 put the voice under the θ's last 15 ms).
    static const float kThAdductMs = param("LINE_TH_ADDUCT_MS", -1.0f);
    if (lastPh_ == Ph::TH && kThAdductMs >= 0) e.agMs = kThAdductMs;
    if (kLineRestartMs > 0) {
      // ... or, what the string folds actually need: a moment pressed shut
      // first, as after a pause, from which they start within a cycle (the
      // slow adduction alone left the vowel 6-9 dB weak to its end).
      Event shut{e.t};
      shut.ag = 0;
      shut.agMs = 6;
      addEvent(shut);
      e.t += kLineRestartMs * ms;
      e.agMs = 10;
    }
    // LINE_ADDUCT_LAG_MS (2026-11-05): the folds start adducting this late, so
    // the hiss fades (the groove opening) before the voice starts. In "save"
    // the voice was on 10-15 ms before the s stopped (Kokoro: the hiss gone
    // 10 ms before the vowel), Jyrki: "less z like".
    static const float kLineAdductLagMs = param("LINE_ADDUCT_LAG_MS", 0.0f);
    if (kLineAdductLagMs > 0 && e.ag >= 0) {
      Event adduct{e.t + kLineAdductLagMs * ms};
      adduct.ag = e.ag;
      adduct.agMs = e.agMs;
      e.ag = -1;
      addEventAt(adduct);
    }
    // LINE_ADDUCT_LEAD_MS (2026-11-06, Jyrki: in "abolish abolish all all"
    // "between the 2nd abolish and the 1st all there is a strange sound"):
    // the folds start adducting this early, before the groove opens. With the
    // glottis still at the fricative's 0.3 cm^2 when the ʃ released, the hiss
    // ran on at full drive for 5 ms into the opening with its band corner
    // jumping 1.1 -> 2.6 kHz: a 4-6 ms broadband click 8-20 dB over the ʃ's
    // own 3-8 kHz level (Kokoro's ʃ fades 10 dB over its last 15 ms, the voice
    // already starting). Closing the glottis first, the flow and the Re^2
    // hiss fall before the release, and the voice is there when it opens.
    // 0: off (the adduction starts with the release, as before).
    static const float kLineAdductLeadMs = param("LINE_ADDUCT_LEAD_MS", 15.0f);
    static const float kThAdductLeadMs = param("LINE_TH_ADDUCT_LEAD_MS", -1.0f);
    const float adductLeadMs = lastPh_ == Ph::TH && kThAdductLeadMs >= 0 ? kThAdductLeadMs : kLineAdductLeadMs;
    if (adductLeadMs > 0 && e.ag >= 0) {
      Event adduct{std::max(e.t - adductLeadMs * ms, phonemeStart_ + 1.0)};
      adduct.ag = e.ag;
      adduct.agMs = e.agMs;
      e.ag = -1;
      addEventAt(adduct);
    }
  }
  double inPauseFrom = phonemeStart_ + 1;
  // English v f s ʃ get the diaphragm's push (LINE_PULSE, queued below).
  const bool pulseFric = tract_ == Tract::Line && aspirate_ && kLinePulse * kLinePulseFric > 0 && !prevWasPlosive_ &&
                         !inAffricate_ && (tok.ph == Ph::VV || tok.ph == Ph::F || tok.ph == Ph::S || tok.ph == Ph::SH);
  if (prevWasPause_ && in.cls != Cls::Stop) {  // speaking again: full pressure
    e.press = 1.0f;
    e.pressMs = kPressResumeMs;
    if (tract_ == Tract::Line) {
      // The lungs push before the first sound: with the breath stopped in the
      // pause, a phrase-initial θ or ð had no pressure for its first 30 ms
      // and came out silent ("Three.", "This.", 2026-11-03).
      // Inside the pause, once its breath has stopped and the mouth has
      // relaxed (the phonemeStart_ bound kept the rise inside the first
      // sound: the n of "Name." had no pressure for its first 50 ms).
      inPauseFrom = std::max(pauseEventT_ + kLinePausePressMs * ms, t_ + 1);
      // Only for the voiced sonorants: before a voiceless fricative or h the
      // breath rising in the pause, mouth still open, glottis opening, burst
      // out as a transient 15 dB over the hiss ("She." heard as a stop, ʃ at
      // the start 4 -> 0 of 12; the 15 ms before the sound are enough, 2026-11-04).
      // LINE_V_EARLY_BREATH (lab 2026-11-09): > 0, a phrase-initial v too. Its 55 ms sat inside the 80 ms pressure
      // ramp that starts at the phoneme (voicing -43 dB, hiss -50 re the word's peak: silent, "very" -> "fairy");
      // Kokoro's initial v is voiced from its first 10 ms (-17..-20 dB).
      static const float kVEarlyBreath = param("LINE_V_EARLY_BREATH", 1.0f);
      static const float kZEarlyBreath = param("LINE_Z_EARLY_BREATH", 1.0f);  // (the same for a phrase-initial z: "zero" silent for 20 ms, -43 dB)
      const bool earlyBreath = in.cls == Cls::Nasal || in.cls == Cls::Liquid || in.cls == Cls::Glide || in.cls == Cls::Trill ||
                               (tok.ph == Ph::VV && kVEarlyBreath > 0) || (tok.ph == Ph::Z && kZEarlyBreath > 0);
      Event prePress{std::max(e.t - kLinePrePressMs * ms, earlyBreath ? std::min(inPauseFrom, e.t) : phonemeStart_ + 1)};
      prePress.press = 1.0f;
      prePress.pressMs = kLinePrePressMs;
      // A phrase-initial θ (TH agent7, round H; LINE_TH_PRE_MS 0: off): the breath starts rising this long before it,
      // over LINE_TH_PRE_RAMP_MS (< 0: LINE_PRE_PRESS_MS). The 80 ms rise from the θ's own start left "three"'s 60 ms θ
      // at -52 dB re the word's peak (2-8 kHz) for its first 50 ms, hiss only in its last 15; Kokoro -41 rising to -19.
      static const float kThPreMs = param("LINE_TH_PRE_MS", 0.0f), kThPreRampMs = param("LINE_TH_PRE_RAMP_MS", -1.0f);
      if (tok.ph == Ph::TH && kThPreMs > 0) {
        prePress.t = std::max(e.t - kThPreMs * ms, std::min(inPauseFrom, e.t));
        if (kThPreRampMs >= 0) prePress.pressMs = kThPreRampMs;
      }
      // A phrase-initial ʃ before a full vowel (SH agent 10, round K; LINE_SH_PRE_MS 0: off): the breath starts rising
      // this long before it, over LINE_SH_PRE_RAMP_MS (< 0: LINE_PRE_PRESS_MS), and the tongue is at the slit (its own
      // place) from the same moment over LINE_SH_PRE_AC_MS, so the rise does not puff out of an open mouth. Puhuja's
      // initial ʃ swelled with the 80 ms pressure rise: -36 / -12 / -10 dB re the vowel over its thirds, Kokoro's flat
      // -16 / -13 / -14 (male) from its first 5 ms; heard s / f ("Shall" -> sell, "Share" -> fair).
      static const float kShPreMs = param("LINE_SH_PRE_MS", 0.0f), kShPreRampMs = param("LINE_SH_PRE_RAMP_MS", -1.0f),
                         kShPreAcMs = param("LINE_SH_PRE_AC_MS", 20.0f);
      const bool shPre = kShPreMs > 0 && tok.ph == Ph::SH && !inAffricate_ && next && isVowel(*next) && *next != Ph::Schwa;
      if (shPre) {
        prePress.t = std::max(e.t - kShPreMs * ms, std::min(inPauseFrom, e.t));
        if (kShPreRampMs >= 0) prePress.pressMs = kShPreRampMs;
      }
      prePress.ac = prePress.ag = prePress.an = -1;
      prePress.speech = -1;
      prePress.aspGain = -1;
      prePress.expand = -1;
      prePress.inhaleMs = 0;
      prePress.pop = -1;
      // (not over a stop's push, LINE_PULSE: it starts from the same rise)
      if (prePress.t >= pulseEnd_) addEventAt(prePress);
      if (shPre && prePress.t >= pulseEnd_ && prePress.t < e.t) {
        // (the whole ʃ setting: its gain, groove, lips and band; a slit alone, with the ʃ's gain and groove switched on at
        // the ʃ itself, burst 6-10 dB over the hiss at 1.5-4 kHz there)
        Event sl = e;
        sl.t = std::max(prePress.t - kShPreAcMs * ms, std::min(inPauseFrom, e.t));
        if (sl.ac <= 0) sl.ac = aero.ac;
        sl.acMs = kShPreAcMs;
        sl.planAir = 0; sl.press = -1; sl.speech = -1; sl.aspGain = -1; sl.expand = -1; sl.inhaleMs = 0; sl.pop = -1;
        // ... and the glottis opens to the ʃ's over LINE_SH_PRE_AG_MS from the same moment (0: at the ʃ, as before:
        // with the breath already up, the glottis opening at the ʃ burst out over the hiss)
        static const float kShPreAgMs = param("LINE_SH_PRE_AG_MS", 0.0f);
        sl.ag = -1;
        if (kShPreAgMs > 0) { sl.ag = aero.ag; sl.agMs = kShPreAgMs; }
        addEventAt(sl);
      }
      // (prePress is the rise; e's own, at the sound, would cut the push)
      if (pulseFric) e.press = -1;
    }
    if (tract_ == Tract::Line && !prevWasPlosive_ && (in.cls == Cls::Vowel || in.cls == Cls::Nasal || in.cls == Cls::Liquid ||
                                  in.cls == Cls::Glide || in.cls == Cls::Trill)) {
      // (not after a plosive: prevWasPause_ still held from the pause before
      // "Time", and the breathy pre-opening left the whole vowel at 0.125)
      // The first voiced sound after a pause (a vowel, or a phrase-initial
      // nasal, l, w, j, r). A soft onset: the folds part to a breathy
      // (kAgBreathy, half open) opening first and adduct (come together)
      // into voicing over kLineOnsetMs. From the shut glottis the
      // string folds reached full amplitude within a cycle, a glottal attack
      // Whisper heard as d or b ("at" -> "Dad", "is" -> "Dave"); Kokoro onyx
      // starts 10 dB down and rises over 100-200 ms.
      const bool nasalOnset = in.cls == Cls::Nasal && nasalKnob(tok.ph, kLineNasalLeadMs, kLineNLeadMs) > 0;
      const SonorantOnset son = nasalOnset ? SonorantOnset{nasalKnob(tok.ph, kLineNasalLeadMs, kLineNLeadMs),
                                                           nasalKnob(tok.ph, kLineNasalVoiceLagMs, kLineNVoiceLagMs),
                                                           kLineNasalOnsetMs}
                                           : sonorantOnset(tok.ph);
      if (nasalOnset || son.onsetMs >= 0) {
        // A phrase-initial nasal: closed mouth, open velum and voicing are
        // set up at the end of the pause, so the murmur sounds from its
        // start (the soft vowel onset left "Name.", "Make." starting with a
        // silent n, heard as "hey"; Kokoro's initial n m are 5-10 dB under
        // the vowel). The closed mouth softens the attack. The same for an
        // initial l w j ɹ with their own knobs (LINE_L_*, LINE_GLIDE_*).
        // Three events: the gesture (e, moved son.leadMs earlier, but not
        // before the pause's breath has stopped), a short breathy opening
        // (pre, 10 ms before the voice), and the voice (the folds adducting
        // from breathy to e's voiced opening over son.onsetMs). The
        // constriction and velum areas themselves were already queued by the
        // pre-pose above (kPreposeMs early); e keeps the rest (place, level).
        e.t = std::max(e.t - son.leadMs * ms, std::min(inPauseFrom, e.t));
        // The voice waits for the closed mouth: voiced while the tongue was
        // still closing, the murmur began with a vowel ("Number." -> "an
        // emperor", "Many." -> "I'm on it", 2026-11-04). A lag < 0 would put
        // the voice before pre, which then left the glottis breathy.
        Event voice{e.t + std::max(son.voiceLagMs, 0.0f) * ms};
        voice.ag = e.ag;
        voice.agMs = son.onsetMs;
        e.ag = -1;
        // With no lead and no lag (l, ɹ) pre falls on the voice's own time
        // and the voice, queued after it, overrides it: the folds go straight
        // from the pause's opening to voice over son.onsetMs.
        Event pre{std::max(voice.t - 10 * ms, e.t)};
        pre.ag = kAgBreathy;
        pre.agMs = 10;
        addEventAt(pre);
        addEventAt(voice);
      } else {
        // The soft onset (vowels, w j by default): breathy first, then e
        // adducts into voice over kLineOnsetMs. pre goes through addEvent, so
        // it cannot start before the events already queued (for a consonant,
        // the pre-pose's kPreposeMs before the sound).
        // (AO agent 8, round I, lab; < 0: LINE_ONSET_MS) an ɔ after a pause adducts over this long instead: the
        // dictionary's word-initial ɔ is 27 % wrong (oʊ 71 of 151: "author" -> "over", "all" -> "goal", "ought" -> "dope").
        static const float kAoOnsetMs = param("LINE_AO_ONSET_MS", -1.0f);
        const float onsetMs = tok.ph == Ph::AO && kAoOnsetMs >= 0 ? kAoOnsetMs : kLineOnsetMs;
        Event pre{std::max(e.t - onsetMs * ms, phonemeStart_ + 1)};
        pre.ag = kAgBreathy;
        pre.agMs = 10;
        addEvent(pre);
        // ... but within the vowel: an unaccented "It" (67 ms) before the t's
        // open glottis never reached voicing with the 80 ms adduction, so
        // "It cannot." lost its "It" (Jyrki 2026-11-05). At most
        // LINE_ONSET_SHARE of the sound's length.
        e.agMs = std::min(onsetMs, kLineOnsetShare * static_cast<float>(dur / fs_ * 1000.0));
        // ... and the string folds do not start from an open glottis on their
        // own: at 0.054 cm^2 with full pressure they sat still through the
        // whole "It" (silent, -56 dB), while an accented word's glottal onset
        // (pressed shut, then released) starts them within a cycle. A brief
        // near-closure (LINE_ONSET_KICK_AG for LINE_ONSET_KICK_MS) as the
        // adduction begins; 0: none.
        // (round-J ə agent 9, lab; < 0: LINE_ONSET_KICK_AG) a word-initial ə's own kick: Whisper heard a consonant before
        // 200 of the 351 wrong initial ə ("aside" -> decide, "assured" -> referred, "abide" -> divide: d 71, r 56, t 26).
        static const float kSwKickAg = param("LINE_SCHWA_KICK_AG", -1.0f);
        const float kickAg = tok.ph == Ph::Schwa && kSwKickAg >= 0 ? kSwKickAg : kLineOnsetKickAg;
        if (kickAg > 0 && kLineOnsetKickMs > 0) {
          Event kick{e.t};
          kick.ag = kickAg;
          kick.agMs = kLineOnsetKickMs;
          addEventAt(kick);
          Event adduct{e.t + kLineOnsetKickMs * ms};
          adduct.ag = e.ag;
          adduct.agMs = std::max(e.agMs - kLineOnsetKickMs, 5.0f);
          e.ag = -1;
          addEventAt(adduct);
        }
      }
    }
    // A pause-initial ð (DH agent7, round H; LINE_DH_PAUSE_KICK_MS 0: off): the fricative took none of the sonorants' onset,
    // and from the pause's glottis the folds started only 20-30 ms into the ð ("Then.": 0-500 Hz at -22..-39 dB re the
    // word's peak through its first 30 ms; Kokoro's voice bar -10..-16 from its first 5 ms; LINE_DH_SINK_FLOW, _PAUSE_AG,
    // _ABDUCT_MS moved nothing). The onset kick (a near-closure, LINE_ONSET_KICK_AG) for this long, then the ð's glottis,
    // the whole gesture LINE_DH_PAUSE_LEAD_MS earlier inside the pause.
    static const float kDhPauseKickMs = param("LINE_DH_PAUSE_KICK_MS", 0.0f), kDhPauseLeadMs = param("LINE_DH_PAUSE_LEAD_MS", 0.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::DH && kDhPauseKickMs > 0 && e.ag >= 0) {
      if (kDhPauseLeadMs > 0) e.t = std::max(e.t - kDhPauseLeadMs * ms, std::min(inPauseFrom, e.t));
      Event kick{e.t};
      kick.ag = std::max(kLineOnsetKickAg, 0.005f);
      kick.agMs = kDhPauseKickMs;
      addEventAt(kick);
      Event adduct{e.t + kDhPauseKickMs * ms};
      adduct.ag = e.ag;
      adduct.agMs = 5.0f;
      e.ag = -1;
      addEventAt(adduct);
    }
  }
  const bool isM = tok.ph == Ph::M;
  // ŋ's approach (lab knobs 2026-11-11, NG agent 3): LINE_NG_APPROACH_SHARE > 0 starts it at least this share of
  // the vowel's length before the ŋ (a lazy pinch through the whole vowel); LINE_NG_APPROACH_HW its half-width
  // (< 0: unchanged); LINE_NG_APPROACH_X its place (< 0: the ŋ's own).
  static const float kNgApShare = param("LINE_NG_APPROACH_SHARE", 0.0f), kNgApHw = param("LINE_NG_APPROACH_HW", -1.0f),
                     kNgApX = param("LINE_NG_APPROACH_X", -1.0f);
  // n after a vowel (round E, N agent): n had no approach, so its F2 was flat into the closure (ʌn 1326-1347 Hz; Kokoro
  // "Sun" rises ~220 Hz). LINE_N_APPROACH_MS > 0 narrows the tongue toward LINE_N_APPROACH_X (0: the n's own place) to
  // LINE_N_APPROACH_A cm^2; the closure itself is still at LINE_N_X.
  static const float kLineNApproachMs = param("LINE_N_APPROACH_MS", 0.0f), kLineNApproachA = param("LINE_N_APPROACH_A", 0.4f),
                     kLineNApproachX = param("LINE_N_APPROACH_X", 0.0f);
  const bool isN = tok.ph == Ph::N;
  float approachMs = isM ? kLineMApproachMs : tok.ph == Ph::NG ? kLineNgApproachMs : isN ? kLineNApproachMs : 0.0f;
  if (tok.ph == Ph::NG && kNgApShare > 0) approachMs = std::max(approachMs, kNgApShare * static_cast<float>((e.t - phonemeStart_) / ms));
  // (round I EH agent 2026-11-12 lab knobs; 0: off) an n or m after ɛ: the approach over LINE_EH_N_APPROACH_MS to
  // LINE_EH_N_APPROACH_A cm^2 (an m keeps its own when longer). Kokoro's ɛ closes into the n (male F1 550 -> 450 -> 300,
  // F2 1750 -> 1550 over the last 60 ms); puhuja's kept its formants to the closure and rose (F1 +120, F2 +500 Hz).
  static const float kEhNApMs = param("LINE_EH_N_APPROACH_MS", 0.0f), kEhNApA = param("LINE_EH_N_APPROACH_A", 0.4f);
  const bool ehNAp = (isN || isM) && lastPh_ == Ph::EH && kEhNApMs > 0 && kEhNApMs > approachMs;
  if (ehNAp) approachMs = kEhNApMs;
  if (tract_ == Tract::Line && approachMs > 0 && prevWasVowel_ && !prevWasPlosive_) {
    Event ap{std::max(e.t - approachMs * ms, phonemeStart_ + 1)};
    ap.ac = ehNAp ? kEhNApA : isM ? kLineMApproachA : isN ? kLineNApproachA : kLineNgApproachA;
    ap.acMs = static_cast<float>((e.t - ap.t) / ms);
    ap.lineX = tok.ph == Ph::NG ? (kNgApX >= 0 ? kNgApX : lineNgPlace(lastPh_, next)) : isN && kLineNApproachX > 0 ? kLineNApproachX : linePlaceOf(tok.ph);
    if (tok.ph == Ph::NG && kNgApHw >= 0) ap.lineHw = kNgApHw;
    ap.lineXClosure = true;
    addEventAt(ap);
  }
  // The velar pinch into ŋ as the tongue body's own move (LINE_NG_PRE_SHAPE, lab knob 2026-11-11, NG agent 3; 0: off):
  // the approach above narrows a hump on top of the vowel's shape, and the ɪ's own front constriction kept F3 at
  // 3450-3550 Hz to the closure (approach area 0.1 or place 0.55-0.75: F2 / F3 within 100 Hz). As after k g
  // (LINE_KG_REL_SHAPE), the vowel's last LINE_NG_PRE_MS and the ŋ are asked for the vowel's shape with its tongue
  // hump at the ŋ's place (LINE_NG_PRE_X, < 0: the ŋ's own), LINE_NG_PRE_AC cm^2, half-width 2 cm. Lab (300 hard ŋ
  // words, 2 voices, three independent pairs of draws): weight 1 at 0.70 with LINE_NG_PRE_BACK_W 0: ŋ errors 69 / 72
  // (base 101 / 94), heard as n 40 -> 28, words1000 1205 (1199), snap -1.5 sigma; "Long." F2 830 -> 1160 and F3
  // 2750 -> 2050 over its last 50 ms (Kokoro 1150 -> 1880, F3 2420).
  {
    static const float kNgPreW = param("LINE_NG_PRE_SHAPE", 1.0f), kNgPreMs = param("LINE_NG_PRE_MS", 60.0f),
                       kNgPreAc = param("LINE_NG_PRE_AC", 0.6f), kNgPreX = param("LINE_NG_PRE_X", 0.71f);
    TractShape v;
    if (tract_ == Tract::Line && tok.ph == Ph::NG && kNgPreW > 0 && prevWasVowel_ && lineShapeOf(lastPh_, &v)) {
      v.xc = kNgPreX >= 0 ? kNgPreX : lineNgPlace(lastPh_, next);
      v.ac = kNgPreAc;
      v.wc = 2.0f;
      v.sc = 1.0f;
      // After ɔ ɑ ɒ o u ʊ the weight is LINE_NG_PRE_BACK_W instead (< 0: the same): the pinch's low F3 there was
      // heard as r + m ("Strong" -> "storm", "Wrong" -> "storm", "Hong" -> "harm", "Belong" -> "velorm").
      static const float kNgPreBackW = param("LINE_NG_PRE_BACK_W", 0.0f);
      const bool back = lastPh_ == Ph::AO || lastPh_ == Ph::AA || lastPh_ == Ph::OH || lastPh_ == Ph::O ||
                        lastPh_ == Ph::U || lastPh_ == Ph::UH;
      const float w = back && kNgPreBackW >= 0 ? kNgPreBackW : kNgPreW;
      if (w > 0) addShapeDemand(std::max(e.t - kNgPreMs * ms, phonemeStart_ + 1), e.t + dur, v, Ph::J, w);
    }
    // ... or the vowel's whole shape mixed LINE_NG_PRE_MIX toward the row LINE_NG_PRE_ROW (a Ph index; 16: ɜ, whose
    // low F3 the pinch needs) over the same window, weighted LINE_NG_PRE_MIX_W (0: off).
    static const float kNgPreMix = param("LINE_NG_PRE_MIX", 0.0f), kNgPreMixW = param("LINE_NG_PRE_MIX_W", 1.0f);
    static const int kNgPreRow = static_cast<int>(param("LINE_NG_PRE_ROW", 16.0f));
    TractShape o;
    if (tract_ == Tract::Line && tok.ph == Ph::NG && kNgPreMix > 0 && prevWasVowel_ && lineShapeOf(lastPh_, &v) &&
        lineShapeOf(static_cast<Ph>(kNgPreRow), &o))
      addShapeDemand(std::max(e.t - kNgPreMs * ms, phonemeStart_ + 1), e.t + dur, lineShapeMix(v, o, kNgPreMix), lastPh_, kNgPreMixW);
  }
  // n's tongue body (round J, N agent 9, lab knobs; 0: off). The n's closure is only the tip: the body stayed in the
  // vowel's shape, so F2 ran flat or the wrong way into and out of it. Kokoro's F2 heads for the alveolar locus
  // (male ~1400-1550 Hz): "Don" 1000 -> 1390 over the ɑ's last 70 ms (puhuja flat ~950), "Thin" 1880 -> 1550 (puhuja
  // rises 2000 -> 2290 with F3 jumping: a velar pinch, heard "thing"). Dictionary (combo7): coda n after ɑ 41 %
  // wrong (ŋ), after ɪ 32 % (ŋ), initial n before oʊ uː iː 29-49 % (m).
  // Into the n: the vowel's last LINE_N_PRE_MS and the n are asked for the vowel's shape mixed LINE_N_PRE_MIX (after
  // a back vowel ɑ ɔ ʌ ʊ uː oʊ LINE_N_PRE_BACK_MIX, < 0: the same) toward the row LINE_N_PRE_ROW (a Ph index; 8: ə),
  // weight LINE_N_PRE_W. Lab (round J, 300 hard n words x 2 voices, two draws; base 288 n errors, placebo sd 6.7):
  // LINE_N_PRE_MIX 0.6 alone 176 (words right 142 -> 192, words1000 1409 -> 1431) but bright +5.4σ (the front vowels'
  // end darkens with the lips and length mixed toward ə); with LINE_N_MIX_TONGUE 1 206, bright +0.8σ. With
  // LINE_N_VEL_LAG_MS 20, LINE_N_REL_MIX 0.6 / _ON_MS 50 / _W 2 / _ROW 9 (ɪ) and LINE_N_X 0.80: 158, words 175,
  // words1000 1428, no veto; "Don" F2 970 -> 1400 into the n (Kokoro 1000 -> 1390).
  {
    static const float kNPreMix = param("LINE_N_PRE_MIX", 0.6f), kNPreBackMix = param("LINE_N_PRE_BACK_MIX", -1.0f),
                       kNPreMs = param("LINE_N_PRE_MS", 60.0f), kNPreW = param("LINE_N_PRE_W", 1.0f);
    static const int kNPreRow = static_cast<int>(param("LINE_N_PRE_ROW", 8.0f));
    const bool backPrev = lastPh_ == Ph::AO || lastPh_ == Ph::AA || lastPh_ == Ph::OH || lastPh_ == Ph::O ||
                          lastPh_ == Ph::U || lastPh_ == Ph::UH || lastPh_ == Ph::AH;
    const float mix = backPrev && kNPreBackMix >= 0 ? kNPreBackMix : kNPreMix;
    // LINE_N_MIX_TONGUE 1: both mixes move the tongue only (the place, opening, width of the hump and the mouth and
    // pharynx areas); the lips, larynx and tract length stay the vowel's (0: the whole shape is mixed).
    static const float kNMixTongue = param("LINE_N_MIX_TONGUE", 1.0f);
    auto mixN = [&](const TractShape& a, const TractShape& b, float w) {
      TractShape m = lineShapeMix(a, b, w);
      if (kNMixTongue > 0) {
        m.aLip = a.aLip; m.lLip = a.lLip; m.aLarynx = a.aLarynx; m.length = a.length;
        m.sc = lineVowelStrength(m.xc, m.wc, m.length);
      }
      return m;
    };
    TractShape v, o;
    // A word-final n after an unstressed vowel (Shape::nctx 2; N agent13 round L lab knobs, LINE_FN_PRE_MIX < 0: as
    // above): the mix LINE_FN_PRE_MIX toward the row LINE_FN_PRE_ROW (< 0: LINE_N_PRE_ROW) over the vowel's last
    // LINE_FN_PRE_MS (< 0: LINE_N_PRE_MS; long: the whole vowel), weight LINE_FN_PRE_W (< 0: LINE_N_PRE_W).
    static const float kFnPreMix = param("LINE_FN_PRE_MIX", -1.0f), kFnPreMs = param("LINE_FN_PRE_MS", -1.0f),
                       kFnPreW = param("LINE_FN_PRE_W", -1.0f);
    static const int kFnPreRow = static_cast<int>(param("LINE_FN_PRE_ROW", -1.0f));
    const bool fnPre = isN && (tok.shape.nctx & 2) && kFnPreMix >= 0;
    const float preMix = fnPre ? kFnPreMix : mix;
    const float preMs = fnPre && kFnPreMs >= 0 ? kFnPreMs : kNPreMs, preW = fnPre && kFnPreW >= 0 ? kFnPreW : kNPreW;
    const int preRow = fnPre && kFnPreRow >= 0 ? kFnPreRow : kNPreRow;
    if (tract_ == Tract::Line && isN && preMix > 0 && prevWasVowel_ && lineShapeOf(lastPh_, &v) &&
        lineShapeOf(static_cast<Ph>(preRow), &o))
      addShapeDemand(std::max(e.t - preMs * ms, phonemeStart_ + 1), e.t + dur, mixN(v, o, preMix), lastPh_, preW);
    // Out of an n not after a vowel (word-initial, after a consonant; LINE_N_REL_ALL 1: every n before a vowel): the n
    // and the next vowel's first LINE_N_REL_ON_MS are asked for the vowel's shape mixed LINE_N_REL_MIX (before a back
    // vowel LINE_N_REL_BACK_MIX, < 0: the same) toward the row LINE_N_REL_ROW (< 0: LINE_N_PRE_ROW), the lips at least LINE_N_REL_LIPA cm^2 (0: as
    // mixed), weight LINE_N_REL_W. The initial n borrowed the oʊ's or uː's rounded shape whole: F2 rose from low into
    // the vowel as after an m ("known" -> bone, "node" -> mode, "nude" -> mood).
    static const float kNRelMix = param("LINE_N_REL_MIX", 0.6f), kNRelBackMix = param("LINE_N_REL_BACK_MIX", -1.0f),
                       kNRelOnMs = param("LINE_N_REL_ON_MS", 50.0f), kNRelW = param("LINE_N_REL_W", 2.0f),
                       kNRelLipA = param("LINE_N_REL_LIPA", 0.0f), kNRelAll = param("LINE_N_REL_ALL", 0.0f);
    static const int kNRelRow = static_cast<int>(param("LINE_N_REL_ROW", 9.0f));
    if (tract_ == Tract::Line && isN && next && isVowel(*next) && (kNRelAll > 0 || !prevWasVowel_)) {
      const bool backNext = *next == Ph::AO || *next == Ph::AA || *next == Ph::OH || *next == Ph::O || *next == Ph::U ||
                            *next == Ph::UH || *next == Ph::AH;
      const float rmix = backNext && kNRelBackMix >= 0 ? kNRelBackMix : kNRelMix;
      if ((rmix > 0 || kNRelLipA > 0) && lineShapeOf(*next, &v) && lineShapeOf(static_cast<Ph>(kNRelRow >= 0 ? kNRelRow : kNPreRow), &o)) {
        TractShape s = mixN(v, o, rmix);
        if (kNRelLipA > 0) s.aLip = std::max(s.aLip, kNRelLipA);
        addShapeDemand(e.t, e.t + dur + kNRelOnMs * ms, s, *next, kNRelW);
      }
    }
  }
  // The full-vowel ʃ's tongue body (SH agent 10, round K; lab knobs, LINE_SH_REL_MIX 0: off). The ʃ had no shape of its
  // own: after a pause the line glided from the pause's ə straight into the vowel, which started at its own F2 (the
  // 0.7-2.7 kHz centroid at its start 800-1300 Hz, flat), a labial-like onset Whisper heard as f ("Share" -> fair,
  // "Shown" -> phone). Kokoro's ʃ raises the tongue body toward the palate: the vowel starts at ~1500-1900 Hz (male)
  // and falls ~400 Hz over 60 ms. The ʃ and the vowel's first LINE_SH_REL_ON_MS are asked for the vowel's shape mixed
  // LINE_SH_REL_MIX toward the row LINE_SH_REL_ROW (a Ph index; 2: i, 9: ɪ), weight LINE_SH_REL_W; LINE_SH_REL_TONGUE 1:
  // the tongue only (lips, larynx, length stay the vowel's); the lips at most LINE_SH_REL_LIPA cm^2 (rounded; 0: as mixed);
  // LINE_SH_REL_PRE_MS: the demand starts this long before the ʃ (into the pause before it). Lab (the 220 dictionary
  // words with a ʃ before a stressed vowel + 80 -tion words, x 2 voices, two draws; base 246 ʃ errors, 313.5 words
  // right, words1000 1447): MIX 1, W 4: 105.5 / 386 / 1456; W 2: 86 / 404; ROW 9 (ɪ), MIX 0.8, W 4: 81.5 / 431.5; with
  // the lower slit LINE_SH_VOWEL_X 0.78 and LINE_SH_VOWEL_FRIC 3: MIX 1, W 2 41 / 438 / 1450, ROW 9, W 4 49.5 / 425.5 /
  // 1452 (no hybrid veto). The vowel's centroid at its start 1023 -> 1571 Hz (male), +60 ms 1311 (Kokoro 1727, 1300).
  // Longer (ON_MS 80) the vowel was heard as iː; rounded lips (LIPA 0.5) cost level and brightness.
  {
    static const float kShRelMix = param("LINE_SH_REL_MIX", 1.0f), kShRelOnMs = param("LINE_SH_REL_ON_MS", 50.0f),
                       kShRelW = param("LINE_SH_REL_W", 4.0f), kShRelTongue = param("LINE_SH_REL_TONGUE", 1.0f),
                       kShRelLipA = param("LINE_SH_REL_LIPA", 0.0f), kShRelPreMs = param("LINE_SH_REL_PRE_MS", 0.0f);
    static const int kShRelRow = static_cast<int>(param("LINE_SH_REL_ROW", 9.0f));
    const bool shVHere = tok.ph == Ph::SH && !inAffricate_ && next && isVowel(*next) && *next != Ph::Schwa;
    TractShape v, o;
    if (tract_ == Tract::Line && shVHere && kShRelMix > 0 && lineShapeOf(*next, &v) && lineShapeOf(static_cast<Ph>(kShRelRow), &o)) {
      TractShape m = lineShapeMix(v, o, std::min(kShRelMix, 1.0f));
      if (kShRelTongue > 0) {
        m.aLip = v.aLip; m.lLip = v.lLip; m.aLarynx = v.aLarynx; m.length = v.length;
        m.sc = lineVowelStrength(m.xc, m.wc, m.length);
      }
      if (kShRelLipA > 0) m.aLip = std::min(m.aLip, kShRelLipA);
      addShapeDemand(std::max(e.t - kShRelPreMs * ms, phonemeStart_ + 1), e.t + dur + kShRelOnMs * ms, m, *next, kShRelW);
    }
  }
  // ʃ ʒ after a vowel (LINE_SH_APPROACH_MS, 2026-11-05): the tongue rises
  // toward the palate first (to LINE_SH_APPROACH_A cm^2, still above the
  // frication's window), then narrows to the slit over LINE_SH_ON_MS, so
  // the hiss grows as the slit closes in, as in Kokoro's -tion (rising
  // over 20-25 ms); in one 20 ms move it jumped on within 5 ms.
  {
    static const float shApMs = param("LINE_SH_APPROACH_MS", 0.0f), shApA = param("LINE_SH_APPROACH_A", 0.4f);
    if (tract_ == Tract::Line && shApMs > 0 && (tok.ph == Ph::SH || tok.ph == Ph::ZH) && prevWasVowel_ && !prevWasPlosive_) {
      Event ap{std::max(e.t - shApMs * ms, phonemeStart_ + 1)};
      ap.ac = shApA;
      ap.acMs = static_cast<float>((e.t - ap.t) / ms);
      ap.lineX = linePlaceOf(tok.ph);
      addEventAt(ap);
    }
  }
  // The other fricatives (f v s z θ ð) after a vowel or a sonorant: the same two-stage approach (LINE_FRIC_APPROACH_MS,
  // 0: off; to LINE_FRIC_APPROACH_A cm^2 first). PUHUJA_ENERGY_TRACE, 2026-11-11: the v of "revolutionary" narrowed
  // 0.55 -> 0.14 cm^2 in ~5 ms with 250-350 cm^3/s still flowing; the flow collapsed (reversed) in 1 ms, the pressure
  // behind swung -1500 / +900 dyn/cm^2, and the surge after it switched the hiss on: the snap (13.3, Kokoro 0.8).
  {
    static const float frApMs = param("LINE_FRIC_APPROACH_MS", 25.0f), frApA = param("LINE_FRIC_APPROACH_A", 0.4f);
    const bool fric = tok.ph == Ph::F || tok.ph == Ph::V || tok.ph == Ph::VV || tok.ph == Ph::S || tok.ph == Ph::Z ||
                      tok.ph == Ph::TH || tok.ph == Ph::DH;
    const bool sonBefore = prevWasVowel_ || lastPh_ == Ph::RR || lastPh_ == Ph::R || lastPh_ == Ph::L || lastPh_ == Ph::W ||
                           lastPh_ == Ph::J || lastPh_ == Ph::M || lastPh_ == Ph::N || lastPh_ == Ph::NG;
    static const float zApMs = param("LINE_Z_APPROACH_MS", -1.0f);  // the z's own (< 0: LINE_FRIC_APPROACH_MS)
    // the v's own (V agent, round F; < 0: LINE_FRIC_APPROACH_MS / _A): a b closes the lips in ~10 ms, a v's lip comes
    // up to the teeth more slowly; Whisper hears 419 of the v's 830 dictionary substitutions as b.
    static const float vApMs = param("LINE_V_APPROACH_MS", -1.0f), vApA = param("LINE_V_APPROACH_A", -1.0f);
    const float apMs = tok.ph == Ph::Z && zApMs >= 0 ? zApMs : tok.ph == Ph::VV && vApMs >= 0 ? vApMs : frApMs;
    // ... not after an m with LINE_M_FRIC_AP 0 (round J, M agent; 1: as before): the approach moved the closure from the
    // lips to the z's place over the m's last 25 ms while the port was open, an n's murmur ("seems" -> scenes, "cams"
    // -> cans; m before z 28 % wrong, heard as n 87 of 120). Lab: worse (m errors +30 alone, +43 with LINE_M_SHAPE).
    static const float kMFricAp = param("LINE_M_FRIC_AP", 1.0f);
    if (tract_ == Tract::Line && apMs > 0 && fric && sonBefore && !prevWasPlosive_ && !(lastPh_ == Ph::M && kMFricAp <= 0)) {
      Event ap{std::max(e.t - apMs * ms, phonemeStart_ + 1)};
      ap.ac = tok.ph == Ph::VV && vApA >= 0 ? vApA : frApA;
      ap.acMs = static_cast<float>((e.t - ap.t) / ms);
      ap.lineX = linePlaceOf(tok.ph);
      addEventAt(ap);
    }
  }
  // l after a vowel (LINE_L_APPROACH_MS, 2026-11-05): the tongue tip rises
  // toward the ridge first (to LINE_L_APPROACH_A cm^2, where the hump just
  // begins to count), then closes to the lateral opening over LINE_L_CLOSE_MS.
  // In one 20 ms move the hump only existed for its last few ms: "all" dropped
  // 6 dB and F1 441 -> 356 at once, "atll" (Jyrki); goog and 11labs drift F1
  // down over 100-150 ms into the l at a steady level.
  {
    static const float lApMs = param("LINE_L_APPROACH_MS", 80.0f) /* Jyrki 2026-11-05: B3 for "all", "but unsure -- still not right" */, lApA = param("LINE_L_APPROACH_A", 0.8f);
    if (tract_ == Tract::Line && lApMs > 0 && tok.ph == Ph::L && prevWasVowel_ && !prevWasPlosive_) {
      Event ap{std::max(e.t - lApMs * ms, phonemeStart_ + 1)};
      ap.ac = lApA;
      ap.acMs = static_cast<float>((e.t - ap.t) / ms);
      ap.lineX = linePlaceOf(tok.ph);
      addEventAt(ap);
    }
  }
  // LINE_Z_AG_LAG_MS (lab 2026-11-11; 0: off): the z's glottis opens this long after its gesture starts, at first
  // only to LINE_Z_AG_LAG_AG (< 0: it stays at the vowel's), so that the voice runs on under the rising hiss
  // (Kokoro's final z: the voice bar fades over its first 30-40 ms while the hiss is already at -18 dB).
  {
    static const float zLagMs = param("LINE_Z_AG_LAG_MS", 0.0f), zLagAg = param("LINE_Z_AG_LAG_AG", -1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::Z && zLagMs > 0 && !prevWasPlosive_ && e.ag >= 0) {
      Event late{e.t + zLagMs * ms};
      late.ag = e.ag;
      late.agMs = e.agMs;
      e.ag = zLagAg;
      e.agMs = kGlottalMs;
      addEventAt(late);
    }
  }
  // LINE_TH_AB_LEAD_MS (TH agent7, round H; 0: off): the θ's glottis starts opening this long before its gesture,
  // inside the vowel before it (the same voice-into-θ as LINE_TH_ABDUCT_MS, from the other end). Lab (297 θ words, both
  // voices, two draws): 20 ms: θ errors 160 -> 124, words right 376 -> 403, words1000 1383 -> 1382; with LINE_TH_ABDUCT_MS
  // 15 and LINE_TH_ON_MS 35: 114 / 414 / 1385. "Author": the θ voiced 29 % -> 0 %, its hiss -47 -> -42 dB (Kokoro -26).
  {
    static const float thAbLead = param("LINE_TH_AB_LEAD_MS", 20.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::TH && thAbLead > 0 && !prevWasPlosive_ && !prevWasPause_ && e.ag >= 0) {
      Event early{std::max(e.t - thAbLead * ms, phonemeStart_ + 1.0)};
      early.ag = e.ag;
      early.agMs = e.agMs;
      e.ag = -1;
      addEventAt(early);
    }
  }
  if (!prevWasPlosive_) {
    if (!events_.empty() && e.t < events_.back().t) addEventAt(e); else addEvent(e);
    if (hasVelLag) addEventAt(velLag);
    if (hasCloseFricBack) addEventAt(closeFricBack);
  }
  // f's hiss fades in over LINE_F_RAMP_MS (0: off) in four steps from
  // LINE_F_RAMP0 x its gain: as the lip met the teeth the slit passed the
  // frication's window with the vowel's flow still running, a 3-5 ms burst
  // 12-15 dB over the f's hiss (at any closing time, 20-45 ms); Kokoro's f
  // fades in (2026-11-05).
  {
    static const float fRampMs = param("LINE_F_RAMP_MS", 70.0f), fRamp0 = param("LINE_F_RAMP0", 0.0f);
    if (tract_ == Tract::Line && fRampMs > 0 && tok.ph == Ph::F && !prevWasPlosive_ && e.lineFricMul >= 0) {
      const double step = std::min(static_cast<double>(fRampMs) * ms, 0.8 * dur) / 4;
      for (int k = 0; k <= 4; ++k) {
        Event r{e.t + k * step};
        r.lineFricMul = e.lineFricMul * (fRamp0 + (1.0f - fRamp0) * k / 4.0f);
        addEventAt(r);
      }
    }
    // ... and the v's the same way (V agent, round F; LINE_V_RAMP_MS 0: off, from LINE_V_RAMP0 x its gain): with the v's
    // hiss restored (LINE_V_RE_SCALE 2.5, LINE_V_FRIC 8) it switched on in a few ms after a vowel, snap.py 5-7 dB in
    // "give", "love", "over" (0 at the base, where the v had no hiss).
    static const float vRampMs = param("LINE_V_RAMP_MS", 60.0f), vRamp0 = param("LINE_V_RAMP0", 0.1f);
    if (tract_ == Tract::Line && vRampMs > 0 && tok.ph == Ph::VV && !prevWasPlosive_ && e.lineFricMul >= 0) {
      const double step = std::min(static_cast<double>(vRampMs) * ms, 0.8 * dur) / 4;
      for (int k = 0; k <= 4; ++k) {
        Event r{e.t + k * step};
        r.lineFricMul = e.lineFricMul * (vRamp0 + (1.0f - vRamp0) * k / 4.0f);
        addEventAt(r);
      }
    }
  }
  if (hasPendingRelax_) {
    addEvent(pendingRelax_);
    hasPendingRelax_ = false;
  }
  else if (lineLevelNow > 0) {
    // After a plosive the release carries this sound's targets and e is not
    // queued; the line's level still needs its event ("antibody": the ɑ
    // after b kept the previous level, 2026-11-02).
    Event lv{e.t};
    lv.lineLevel = lineLevelNow;
    lv.ac = lv.ag = lv.an = -1;
    lv.speech = -1;
    lv.press = -1;
    lv.aspGain = -1;
    lv.expand = -1;
    lv.inhaleMs = 0;
    lv.pop = -1;
    addEvent(lv);
  }
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
  if (pulseFric)
    addPressPulse(t0 - std::max(kLinePulseLeadMs, 0.0f) * ms, std::max(phonemeStart_, t_) + 1,
                  kLinePulse * kLinePulseFric);  // the diaphragm's push (LINE_PULSE)
  // θ's own push (LINE_TH_PULSE, TH agent7 round H; 0: off, as LINE_PULSE 0 leaves every fricative): the breath pressure
  // rises by this (relative) to a peak LINE_TH_PULSE_AT_MS into the θ: its hiss 15-25 dB under Kokoro's, where a louder
  // gain alone (LINE_TH_FRIC) brought s confusions; flow, not the noise gain, makes a dental hiss.
  { static const float kThPulse = param("LINE_TH_PULSE", 0.0f), kThPulseAt = param("LINE_TH_PULSE_AT_MS", 20.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::TH && kThPulse > 0)
      addPressPulse(t0 + kThPulseAt * ms, std::max(phonemeStart_, t_) + 1, kThPulse); }
  // The syllable's start: its first onset consonant after a vowel or a pause.
  if (in.cls != Cls::Vowel && in.cls != Cls::Stop && (prevWasVowel_ || prevWasPause_ || syllOnsetT_ < 0)) syllOnsetT_ = t0;
  if (stressExtra > 0) {
    const double onset = syllOnsetT_ >= 0 && syllOnsetT_ < t0 ? syllOnsetT_ : t0;
    const double peak = t0 + kStressPeakMs * ms;
    addPressPulse(peak, std::max({onset, phonemeStart_, t_ + 1.0}), kStressPulse * stressExtra,
                  static_cast<float>(std::max(peak - onset, 20.0 * ms) / ms), kStressFallMs);
  }
  if (in.cls == Cls::Vowel) syllOnsetT_ = -1;
  last_ = p;
  phonemeStart_ = t0;
  timeline_ += dur;
  prevWasVowel_ = in.cls == Cls::Vowel;
  jAfterStop_ = tok.ph == Ph::J && prevWasPlosive_;
  prevWasPlosive_ = false;
  prevWasNasal_ = in.cls == Cls::Nasal;
  const bool pauseRunFirst = in.cls == Cls::Stop && !prevWasPause_ &&
                             (kLinePauseRunOy < 0.5f || oyEndT_ > timeline_ - kLinePausePlanHoldMs * 0.001 * fs_);  // (see kLinePauseRunPlanMs)
  if (tok.ph == Ph::IH && lastPh_ == Ph::AO) oyEndT_ = timeline_;
  prevWasPause_ = in.cls == Cls::Stop;
  if (!prevWasPause_) relaxAt_ = -1;
  // The new phrase's pitch takes over from the last 15 ms of the pause (from
  // its first, read ahead, the old phrase's ending turned upward).
  if (in.cls == Cls::Stop) pauseStartT_ = t0 + 0.5 * dur;
  if (in.cls == Cls::Stop) pitchAnchorT_ = -1;  // the pitch line restarts after a pause
  prevVoiced_ = isSonorant(tok.ph);
  prevVoicedObstruent_ = voicedObstruent(tok.ph);
  lastPh_ = tok.ph;
  rangeTok_ = nullptr;
  // A pause closes a phrase: plan its shapes and pitch (the line renders
  // only what is planned, see pull()).
  if (in.cls == Cls::Stop) {
    // (only within a pause run or before a j or l with a pre-shape: before
    // vowels and nasals the shapes in the pause changed their starts, guards
    // 10 -> 6 words of 24)
    const bool hold = next && (*next == Ph::Stop || (*next == Ph::J && kLineJPreShapeMs > 0) || (*next == Ph::L && kLineLPreShapeMs > 0) || (*next == Ph::W && kLineWPreShapeMs > 0) || (*next == Ph::RR && kLineRPreShapeMs > 0));
    double planEnd = timeline_ - (hold ? kLinePausePlanHoldMs * 0.001 * fs_ : 0.0);
    if (hold && kLinePauseRunPlanMs >= 0 && pauseRunFirst)  // the run's first pause: the speech's own demands (see kLinePauseRunPlanMs)
      planEnd = std::min(timeline_, std::max(planEnd, t0 + kLinePauseRunPlanMs * 0.001 * fs_));
    planShapes(planEnd);
    planSentence();
  }
}

// An affricate is a plosive released straight into a short fricative at the
// same place: ʧ = closure + postalveolar burst + short ʃ, ʤ likewise with ʒ.
void Synth::emitAffricate(const Token& tok, const Ph* next) {
  const Ph fric = tok.ph == Ph::CH ? Ph::SH : Ph::ZH;
  jhPos_ = tok.ph != Ph::JH ? -1 : prevWasPause_ ? 0 : (!next || *next == Ph::Stop) ? 2 : 1;  // (the LINE_JH_INIT_* ... knobs)
  Token closure = tok;
  if (jhPos_ >= 0 && closure.shape.ms > 0) closure.shape.ms *= kLineJhPosClose[jhPos_];
  emitPlosive(closure, &fric);  // a fricative next: no aspiration, no voice onset
  prevWasPlosive_ = true;
  inAffricate_ = true;
  // The English reader times the closure (tok.shape.ms); the frication
  // follows it in proportion, so a shortened affricate (an unaccented
  // "which") shortens as a whole. (With the table's length the frication
  // stayed 92 ms whatever the reader asked for.)
  Shape fs;
  if (tok.shape.ms > 0) fs.ms = tok.shape.ms * kAffricateFricationPerClosure * (jhPos_ >= 0 ? kLineJhPosFlen[jhPos_] : 1.0f);
  const double tf0 = timeline_;
  emit(Token{fric, false, false, 0, false, kNoPitch, fs}, next);
  // LINE_JH_*_ZAG (< 0: off): the glottis moves to this area LINE_JH_ZAG_AT of the way through the frication (the
  // release opened it to LINE_JH_AG for the whole ʒ part): Kokoro's medial and final ʤ voice again before the vowel.
  static const float kLineJhZagAt = param("LINE_JH_ZAG_AT", 0.5f);
  if (tract_ == Tract::Line && jhPos_ >= 0 && kLineJhPosZag[jhPos_] >= 0) {
    Event g{tf0 + kLineJhZagAt * (timeline_ - tf0)};
    g.ac = g.an = -1;
    g.ag = kLineJhPosZag[jhPos_];
    g.agMs = kGlottalMs;
    addEventAt(g);
  }
  if (tract_ == Tract::Line && jhPos_ >= 0 && jhRelFric_ > 0) {
    const float ms = 0.001f * fs_;
    if (kLineJhRampMs > 0)
      for (int k = 1; k <= 4; ++k) {
        Event r{tf0 + std::min(static_cast<double>(kLineJhRampMs) * ms, 0.8 * (timeline_ - tf0)) * k / 4};
        r.ac = r.ag = r.an = -1;
        r.lineFricMul = jhRelFric_ * (kLineJhRamp0 + (1.0f - kLineJhRamp0) * k / 4.0f);
        addEventAt(r);
      }
    if (kLineJhDecay != 1.0f) {
      Event d{tf0 + kLineJhDecayAt * (timeline_ - tf0)};
      d.ac = d.ag = d.an = -1;
      d.lineFricMul = jhRelFric_ * kLineJhDecay;
      addEventAt(d);
    }
  }
  jhRelFric_ = -1;
  inAffricate_ = false;
  jhPos_ = -1;
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
    case Ph::TH: { static const float s = param("LINE_TH_COART", 1.0f); return {{0.1f * s, 0.25f * s, 0.05f * s}, {0.05f * s, 0.35f * s, 0.25f * s}}; }  // x the θ's shares (lab 2026-11-09)
    case Ph::DH: { static const float s = param("LINE_DH_COART", 1.0f); return {{0.1f * s, 0.25f * s, 0.05f * s}, {0.05f * s, 0.35f * s, 0.25f * s}}; }  // x the ð's shares (DH agent7)
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
  // (LINE_K_X_FRONT / _BACK: the vowel that sets the k's place)
  const Ph kVowel = next && isVowel(*next) ? *next : prevWasVowel_ ? lastPh_ : next ? *next : Ph::Stop;
  // (K-INITIAL agent 13 round L, lab; < 0: LINE_K_X_FRONT) the place of a k released into i ɪ e ɛ æ (an onset k only;
  // a final k after a front vowel keeps LINE_K_X_FRONT): Kokoro's and 11labs' k before a front vowel bursts at 2.7-3 kHz,
  // puhuja's at 1.5-1.8 kHz ("Came" -> pain, "Key" -> pea).
  static const float kKXFrontOnset = param("LINE_K_X_FRONT_ONSET", -1.0f);
  const bool kOnsetFront = tok.ph == Ph::K && kKXFrontOnset >= 0 && next &&
                           (*next == Ph::I || *next == Ph::IH || *next == Ph::E || *next == Ph::EH || *next == Ph::AE);
  const float kPlaceTok = kOnsetFront ? kKXFrontOnset : lineKPlace(tok.ph, kVowel);
  PlosiveInfo pi = plosiveInfo(tok.ph);
  // English b not between vowels ("I said bat", "big"): a mostly devoiced
  // closure and a clear release. Fully voiced with almost no burst it was
  // heard as m or not at all (language researcher: b words 2/9 -> 7/9).
  // LINE_EN_B_VOICED_CTX (2026-11-11, review voice.md B3; 0: as before): 1: a word-initial b after a vowel-final word
  // ("the bigger", "a baby", "to bed") counts as between vowels (prevWasVowel_ is reset at every word start, so these
  // were devoiced with an abducted glottis, ag 0.159, while readers and Kokoro voice them); 2: also after any sonorant
  // (as g d with LINE_GD_VOICED_CTX). The utterance-initial and post-obstruent b keep the opening.
  static const int kEnBVoicedCtx = param("LINE_EN_B_VOICED_CTX", 0);
  const bool bVoicedBefore = !prevWasPause_ && (kEnBVoicedCtx >= 2 ? prevVoiced_ : kEnBVoicedCtx >= 1 && tok.wordStart && isVowel(lastPh_));
  const bool englishB = aspirate_ && tok.ph == Ph::B && !prevWasVowel_ && !bVoicedBefore;
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
    static const float flapPop = param("LINE_FLAP_POP", 0.2f);  // ... x this on the line
    if (tract_ == Tract::Line) {
      pi.noiseGain *= flapPop;
      pi.pop *= flapPop;
    }
  }
  // English g d not after a vowel (LINE_EN_GD_DEVOICE, 2026-11-04): like b,
  // a mostly devoiced closure. Voiced through, the closure's 40 ms voice bar
  // was heard as a vowel of its own ("Good." -> "ugh", "Go." -> "boo").
  // LINE_GD_VOICED_CTX (2026-11-06): 1: a voiced fricative before ("of God": v g) is voiced context too, 2: also a
  // voiced plosive. Only sonorants counted, so the g of "hand of God" devoiced after the v: its closure was 50 ms of
  // dead air between the v and the burst (Kokoro: a voice bar 25-30 dB under the vowel, continuous with the v).
  static const int kGdVoicedCtx = param("LINE_GD_VOICED_CTX", 1);
  const bool voicedBefore = (prevVoiced_ || (kGdVoicedCtx > 0 && prevVoicedObstruent_ > 0 && prevVoicedObstruent_ <= kGdVoicedCtx)) && !prevWasPause_;
  const bool englishGD = kLineEnGdDevoice > 0 && aspirate_ && (tok.ph == Ph::G || tok.ph == Ph::D) && !prevWasVowel_ &&
                         !voicedBefore;
  // LINE_EN_G_DEVOICE (lab knob 2026-11-09, G agent; < 0: LINE_EN_GD_DEVOICE): the g's own closure glottis.
  static const float kLineEnGDevoice = param("LINE_EN_G_DEVOICE", -1.0f);
  static const float kLineEnDDevoice = param("LINE_EN_D_DEVOICE", -1.0f);  // (lab knob 2026-11-10, D agent; < 0: LINE_EN_GD_DEVOICE): the d's own closure glottis
  if (englishGD) pi.closureAg = kAgVoice + (tok.ph == Ph::G && kLineEnGDevoice >= 0 ? kLineEnGDevoice : tok.ph == Ph::D && kLineEnDDevoice >= 0 ? kLineEnDDevoice : kLineEnGdDevoice) * (kAgOpen - kAgVoice);
  // ... and the same for an English ʤ not after a voiced sound (LINE_EN_JH_DEVOICE, 0: off): after a pause its closure
  // carried a voice bar at -5..-15 dB re the vowel for 30-40 ms, the cue of a d ("just" -> "dust", 80 % of initial ʤ
  // misheard in the dictionary, 2026-11-09); Kokoro's initial ʤ is 50-65 ms of voiceless hiss.
  {
    static const float kLineEnJhDevoice = param("LINE_EN_JH_DEVOICE", 0.0f);
    if (kLineEnJhDevoice > 0 && aspirate_ && tok.ph == Ph::JH && !prevWasVowel_ && !voicedBefore)
      pi.closureAg = kAgVoice + kLineEnJhDevoice * (kAgOpen - kAgVoice);
  }
  {  // (D agent9 round J lab knob; < 0: off) the closure glottis of a d between a vowel and an unstressed vowel (cm^2)
    static const float kDVvAg = param("LINE_D_VV_AG", -1.0f);
    if (tok.ph == Ph::D && tok.shape.dctx == 1 && kDVvAg >= 0) pi.closureAg = kDVvAg;
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
  // d before s z ("holds", "needs"; LINE_D_SIB_CLOSURE, 2026-11-05): the
  // tongue is already on the ridge for the sibilant; goog's d of "holds" is a
  // 15-20 ms dip, ours a 40 ms silence. x this (1: as before).
  {
    static const float dSib = param("LINE_D_SIB_CLOSURE", 0.5f);
    if (tok.ph == Ph::D && dSib != 1.0f && next && (*next == Ph::S || *next == Ph::Z)) closure *= dSib;
  }
  // d before ð θ ("around the") is dental: the tongue moves to the teeth and
  // runs into the fricative with a short closure and no real burst. Fully
  // released, it snapped (a 20 dB jump at the ð).
  const bool dentalD = tok.ph == Ph::D && next && (*next == Ph::DH || *next == Ph::TH);
  if (dentalD) closure *= kDentalDClosure;
  // Before another plosive ("cannot do", "that time") the first is not
  // released: its closure runs into the next one's. Released, its burst
  // was a lone click in the silence (+24 dB).
  // A voiced stop before a voiceless fricative ("and foo", "and pharaoh") is
  // released into the fricative's constriction, not popped through an open
  // mouth: unreleased here (Jyrki 2026-11-07: "a strange sound between the
  // words"; the pop was 10 dB over the f that followed). LINE_VSTOP_FRIC_UNREL 0: as before.
  // Jyrki 2026-11-07: before the voiced ð of "United there" the unreleased d
  // "is good"; before the voiceless f of "and foo" he preferred the released
  // one, softened ("mix 75 % of unrel0 and 25 % of unrel1"). So: unreleased
  // before a voiced fricative (LINE_VSTOP_VFRIC_UNREL 1), released before a
  // voiceless one (LINE_VSTOP_FRIC_UNREL 0) with its burst at
  // LINE_VSTOP_FRIC_REL_GAIN of the fricative's gain.
  static const float kVstopFricUnrel = param("LINE_VSTOP_FRIC_UNREL", 0.0f), kVstopVfricUnrel = param("LINE_VSTOP_VFRIC_UNREL", 1.0f);
  const bool nextFric = next && info(*next).cls == Cls::Fricative;
  const bool nextVoicedFric = nextFric && (*next == Ph::VV || *next == Ph::Z || *next == Ph::ZH || *next == Ph::DH);
  if (tract_ == Tract::Line && pi.voiced) line_.setExpandOffAllowed(nextFric);  // (LINE_EXPAND_OFF_FRIC)
  // LINE_DZ_REL (D agent9 round J lab knob; 0: as before): 1: a d before z is released into it (the dictionary hears the
  // d + z of "beads", "codes" as t + s: 106 of 692; its 17 ms unreleased closure is a bare gap before the z).
  static const float kDzRel = param("LINE_DZ_REL", 0.0f);
  const bool dzRelD = tok.ph == Ph::D && kDzRel > 0.5f && next && *next == Ph::Z;
  const bool unreleased = next && (info(*next).cls == Cls::Plosive ||
                                   (tract_ == Tract::Line && pi.voiced && nextFric && tok.ph != Ph::JH && !dzRelD && (nextVoicedFric ? kVstopVfricUnrel : kVstopFricUnrel) > 0.5f));  // (not the affricate's own release into its ʒ: inAffricate_ is set after this)
  if (unreleased) closure *= kUnreleasedClosure;
  const bool afterUnreleased = prevUnreleased_;
  prevUnreleased_ = unreleased;
  // A stop after a homorganic nasal (n d, m b, ŋ g and the voiceless ones; 2026-11-06, Jyrki: "hand of God" makes
  // weird blib-blib sounds on both beginning and end of 'of'): the mouth is shut already, yet the approach below
  // opened the n's seal to 0.15 cm^2 for 25 ms and closed it again, and the velum shut with the closure, so the
  // murmur died into 40 ms of near silence before a burst 15 dB louder than Kokoro's. Kokoro keeps a d after a
  // nasal within 3 dB of the nasal (the murmur goes on; voiceless stops drop 5-13 dB).
  // LINE_NASAL_STOP_HOLD 1: no re-opening, the seal is kept. LINE_NASAL_STOP_VEL_MS > 0 (a voiced stop): the velum
  // closes only this long before the release, the murmur running through the closure. LINE_NASAL_STOP_BURST: that
  // stop's slit-noise gain while the closure opens (1: as any; the burst was the voice's flow hissing through the
  // opening slit, -4.5 dB re the vowel peak, Kokoro -20); LINE_NASAL_STOP_REL scales its opening time (1: as any).
  static const int kNasalStopHold = param("LINE_NASAL_STOP_HOLD", 1);
  static const float kNasalStopVelMs = param("LINE_NASAL_STOP_VEL_MS", 20.0f);
  static const float kNasalStopBurst = param("LINE_NASAL_STOP_BURST", 1.0f);
  static const float kNasalStopRel = param("LINE_NASAL_STOP_REL", 2.0f);
  const bool homorganicNasal = prevWasNasal_ && ((lastPh_ == Ph::M && place(tok.ph) == 0) || (lastPh_ == Ph::N && place(tok.ph) == 1) ||
                                                (lastPh_ == Ph::NG && (tok.ph == Ph::K || tok.ph == Ph::G)));
  const bool holdNasal = tract_ == Tract::Line && kNasalStopHold > 0 && homorganicNasal;
  const bool nasalVel = tract_ == Tract::Line && kNasalStopVelMs > 0 && homorganicNasal && pi.voiced && !unreleased;
  // b on the line: the lips hold 30 ms longer, gathering more pressure, and
  // release it softly (Jyrki 2026-11-03: "collect more pressure for b,
  // perhaps 30 ms longer, but release it softly").
  if (tract_ == Tract::Line && tok.ph == Ph::B && !unreleased) closure += kLineBExtraMs * 0.001 * fs_;
  double release = timeline_ + closure;
  const float ms = 0.001f * fs_;
  // The line's shape through the closure: a labial stop is cut into o's
  // shape (the tract behind the lips is o's; only the lips differ), so the
  // vowel's formants glide out of o's, F2 from low. Plosives bypass emit()'s
  // key code, so until now no stop pushed a shape key.
  if (tract_ == Tract::Line) {
    TractShape sh;
    if (lineShapeOf(tok.ph, &sh)) {
      // A b / p before a ʊ in this share of ʊ's own shape instead of o's (LINE_UH_LABIAL_OWN, round-H ʊ agent, lab; 0:
      // off): the o behind the closed lips made the ʊ start as o, F1 falling 620 -> 530 Hz (female "Put", "Book",
      // "Pull"; Kokoro's rises), an oʊ: "pulls" -> polls, "bull" -> bowl, "Brooke" -> broke.
      static const float kUhLabialOwn = param("LINE_UH_LABIAL_OWN", 0.0f);
      if (kUhLabialOwn > 0 && (tok.ph == Ph::P || tok.ph == Ph::B) && next && *next == Ph::UH) {
        TractShape u;
        if (lineShapeOf(Ph::UH, &u)) { uhShape(&u, tok.ph, nullptr, female_); sh = lineShapeMix(sh, u, kUhLabialOwn); }
      }
      // ... and the same before a uː (LINE_UW_LAB_OWN, round-I uː agent, lab; 0: off): after b the uː was wrong 56 % of
      // the time in the dictionary (boot -> book / boat, bool -> bowl, booth -> birth), after p 39 %.
      static const float kUwLabOwn = param("LINE_UW_LAB_OWN", 0.0f);
      if (kUwLabOwn > 0 && (tok.ph == Ph::P || tok.ph == Ph::B) && next && *next == Ph::U) {
        TractShape u;
        if (lineShapeOf(Ph::U, &u)) sh = lineShapeMix(sh, u, std::min(kUwLabOwn, 1.0f));
      }
      lineKeys_.push_back({timeline_ + 0.5 * closure, sh});
      addShapeDemand(timeline_ + 0.3 * closure, timeline_ + 0.7 * closure, sh, tok.ph, kShapeBorrowedW);
      lastKeyShape_ = sh;
      haveLastKeyShape_ = true;
    }
  }
  const bool nextIsPause = !next || *next == Ph::Stop;
  Aero nextAero = next ? aeroOf(*next) : aeroOf(Ph::Stop);
  {  // (D agent9 round J lab knob; < 0: as before) the glottis of a z after a d (cm^2; the z's event rides on the d's
     // release): the z of "beads" ran voiceless (LF -43 dB re the word peak, Kokoro's -15..-20): d + s = "beats".
    static const float kDzZAg2 = param("LINE_DZ_Z_AG", -1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::D && next && *next == Ph::Z && kDzZAg2 >= 0) nextAero.ag = kDzZAg2;
  }

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
  // A prevoiced phrase-initial b d g (LINE_VSTOP_LEAD_MS, 2026-11-04): when
  // the voice is to start before the closure's nominal start, the mouth
  // closes that much earlier, inside the pause (once its breath has stopped,
  // as for a phrase-initial nasal); tClosed is when the closure is complete.
  const bool vstopLead = tract_ == Tract::Line && kLineVStopLeadMs > 0 && pi.voiced && prevWasPause_ && !unreleased;
  double tClosed = timeline_;
  if (vstopLead) {
    const double inPause = std::max(pauseEventT_ + kLinePausePressMs * ms, t_ + 1.0) + (kApproachMs + kSealMs) * ms;
    tClosed = std::max(std::min(release - kLineVStopLeadMs * ms, timeline_), std::min(inPause, timeline_));
  }
  const double closeStart =
      std::max(tClosed - (kApproachMs + kSealMs) * ms, phonemeStart_ + 1);
  const float closeMs = static_cast<float>((tClosed - closeStart) / ms);
  const float sealMs = std::min(kSealMs, closeMs);
  // A vowel before k g (LINE_KG_APPROACH_MS) or b p (LINE_BP_APPROACH_MS)
  // narrows at the stop's place before the closing below, as before ŋ and m
  // (2026-11-04): the tongue back or the lips move early, the vowel's F2 and
  // F3 bend into the stop.
  {
    const bool velar = tok.ph == Ph::K || tok.ph == Ph::G, labial = tok.ph == Ph::B || tok.ph == Ph::P;
    const bool alveolarT = tok.ph == Ph::T;  // (LINE_T_APPROACH_MS)
    // (G agent 7 round H, lab; < 0: the k g ones) the g's own approach: LINE_G_APPROACH_MS, its area LINE_G_APPROACH_A.
    static const float kGApproachMs = param("LINE_G_APPROACH_MS", -1.0f), kGApproachA = param("LINE_G_APPROACH_A", -1.0f);
    const bool gOwnAp = tok.ph == Ph::G && kGApproachMs >= 0;
    const float apMs = gOwnAp ? kGApproachMs : velar ? kLineKgApproachMs : labial ? kLineBpApproachMs : alveolarT ? kLineTApproachMs : 0.0f;
    // (K agent 9 round J, lab; 0: off) the k's velar approach also after ɹ l (LINE_K_APPROACH_LIQ 1): final k after ɹ
    // was 57 % wrong ("Park" -> part, "Mark" -> north, "Stark" -> start).
    static const float kKApproachLiq = param("LINE_K_APPROACH_LIQ", 0.0f);
    const bool kApLiq = tok.ph == Ph::K && kKApproachLiq > 0 && (lastPh_ == Ph::RR || lastPh_ == Ph::L);
    // (T agent 10 round K, lab; < 0 / 0: as before) the t's own approach after a back vowel ɑ ɔ ʌ o u ʊ
    // (LINE_T_APPROACH_MS_BACK, its area LINE_T_APPROACH_A_BACK) and after ɹ l (LINE_T_APPROACH_LIQ 1): the final t
    // after ɑ is 62 % wrong in the dictionary, heard p / k ("Not" -> knob, "Hot" -> hop, "Knots" -> knox), after ɹ l
    // 18 % (d: "Cart" -> card, "Built" -> build).
    static const float kTApMsBack = param("LINE_T_APPROACH_MS_BACK", -1.0f), kTApABack = param("LINE_T_APPROACH_A_BACK", -1.0f),
                       kTApLiq = param("LINE_T_APPROACH_LIQ", 0.0f);
    const bool tBackV = alveolarT && (lastPh_ == Ph::AA || lastPh_ == Ph::AO || lastPh_ == Ph::AH || lastPh_ == Ph::OH ||
                                      lastPh_ == Ph::O || lastPh_ == Ph::U || lastPh_ == Ph::UH);
    const bool tApLiq = alveolarT && kTApLiq > 0 && (lastPh_ == Ph::RR || lastPh_ == Ph::L);
    const float apMsT = tBackV && kTApMsBack >= 0 ? kTApMsBack : apMs;
    if (tract_ == Tract::Line && apMsT > 0 && (prevWasVowel_ || kApLiq || tApLiq) && !prevWasPlosive_ && !afterUnreleased) {
      Event ap{std::max(tClosed - apMsT * ms, phonemeStart_ + 1)};
      if (ap.t < closeStart - ms) {
        ap.ac = tok.ph == Ph::G && kGApproachA >= 0 ? kGApproachA : velar ? kLineKgApproachA : labial ? kLineBpApproachA : tBackV && kTApABack >= 0 ? kTApABack : kLineTApproachA;
        ap.acMs = static_cast<float>((closeStart - ap.t) / ms);
        ap.lineX = kPlaceTok;
        ap.lineXClosure = true;
        addEventAt(ap);
      }
    }
  }
  // t's tongue body (T agent 10 round K, lab knobs; 0: off), as the n's LINE_N_PRE_MIX: the t's closure is only the
  // tip, so after a back vowel the body stayed back and the ridge narrowing alone lowered F2 into the closure, a labial
  // pattern ("Not" F2 ~930 -> 770 over its last 20 ms; Kokoro ~1150 -> 1500-1590: the alveolar locus). The final t
  // after ɑ is 62 % wrong ("Not" -> knob, "Hot" -> hop), ts after ɑ alike ("Knots" -> knox).
  // The vowel's last LINE_T_PRE_MS and the closure are asked for the vowel's shape mixed LINE_T_PRE_MIX (after a front
  // vowel LINE_T_PRE_FRONT_MIX, < 0: the same) toward the row LINE_T_PRE_ROW (a Ph index; 8: ə, 9: ɪ), tongue only
  // (lips, larynx, length stay the vowel's), weight LINE_T_PRE_W.
  {
    static const float kTPreMix = param("LINE_T_PRE_MIX", 0.4f), kTPreFrontMix = param("LINE_T_PRE_FRONT_MIX", -1.0f),
                       kTPreMs = param("LINE_T_PRE_MS", 60.0f), kTPreW = param("LINE_T_PRE_W", 1.0f);
    static const int kTPreRow = static_cast<int>(param("LINE_T_PRE_ROW", 9.0f));
    const bool backPrev = lastPh_ == Ph::AO || lastPh_ == Ph::AA || lastPh_ == Ph::OH || lastPh_ == Ph::O ||
                          lastPh_ == Ph::U || lastPh_ == Ph::UH || lastPh_ == Ph::AH;
    const float mix = !backPrev && kTPreFrontMix >= 0 ? kTPreFrontMix : kTPreMix;
    TractShape v, o;
    if (tract_ == Tract::Line && tok.ph == Ph::T && mix > 0 && prevWasVowel_ && !afterUnreleased && lineShapeOf(lastPh_, &v) &&
        lineShapeOf(static_cast<Ph>(kTPreRow), &o)) {
      TractShape m = lineShapeMix(v, o, mix);
      m.aLip = v.aLip; m.lLip = v.lLip; m.aLarynx = v.aLarynx; m.length = v.length;
      m.sc = lineVowelStrength(m.xc, m.wc, m.length);
      addShapeDemand(std::max(closeStart - kTPreMs * ms, phonemeStart_ + 1.0), release, m, lastPh_, kTPreW);
    }
  }
  // d after l (LINE_LD_SEAL_FADE, 2026-11-05; 0: off): the tongue tip is on
  // the ridge already, the closure only seals the sides, no jet. Narrowing
  // from the l's 0.35 cm^2 through the fricative range with the voice's flow
  // it hissed: "world is", a 10-15 ms, 4-8 kHz puff 16 dB under the vowel at
  // the closing (the readers and Kokoro: nothing). The frication's fade
  // areas x this through the closing; the release restores them.
  static const float kLdSealFade = param("LINE_LD_SEAL_FADE", 0.3f);
  const bool ldSeal = tract_ == Tract::Line && kLdSealFade > 0 && tok.ph == Ph::D && lastPh_ == Ph::L;
  if (closeMs > sealMs + 1 && !afterUnreleased && !holdNasal) {  // (closed already after an unreleased one, or a homorganic nasal)
    Event approach{closeStart};
    if (ldSeal) approach.lineFricFade = kLdSealFade;
    approach.ac = kSealArea;
    approach.acMs = closeMs - sealMs;
    if (tract_ == Tract::Line && tok.ph == Ph::T && kLineTJetD > 0) approach.lineJetD = kLineTJetD, approach.lineFricLpr = kLineTLpr;
    if (tract_ == Tract::Line && tok.ph == Ph::P && kLinePJetD > 0) approach.lineJetD = kLinePJetD, approach.lineFricLpr = kLinePLpr;  // (LINE_P_JET_D)  // (LINE_T_JET_D, below)
    if (tract_ == Tract::Line && tok.ph == Ph::T && kLineTCloseFric != 1.0f) approach.lineFricMul = kLineTCloseFric;  // (LINE_T_CLOSE_FRIC, below)
    if (tract_ == Tract::Line && tok.ph == Ph::JH && kLineJhCloseFric != 1.0f) approach.lineFricMul = kLineJhCloseFric;
    if (tract_ == Tract::Line && kLineCloseFric != 1.0f && approach.lineFricMul < 0 && tok.ph != Ph::T && tok.ph != Ph::JH && tok.ph != Ph::CH) approach.lineFricMul = kLineCloseFric;  // (LINE_CLOSE_FRIC)  // (its ʒ sets its own at the release)
    if (tract_ == Tract::Line && kLineStopApproachX > 0) approach.lineX = kPlaceTok, approach.lineXClosure = true;  // (LINE_STOP_APPROACH_X)
    addEvent(approach);
  }
  Event close{tClosed - sealMs * ms};
  // The line's constriction for this closure: its place, groove width and
  // rounding, and a voiced stop's soft, longer closure. (Plosives bypass the
  // generic event that carries these, so until 2026-11-03 every b, p, g, k
  // closed wherever the last consonant had left the tongue, mostly the
  // alveolar ridge: "abolish" was heard as "atolish".)
  close.lineX = kPlaceTok;
  close.lineXClosure = true;
  {  // (K agent 9 round J, lab; < 0: as above) a k after s at LINE_SK_X (sk heard as sp 81 of 159 errors: "Ski" -> speed,
     // "Skating" -> splitting); a k before w with its lips rounded already (LINE_KW_LIP, 0: none; kw heard tw 46 times).
    static const float kSkX = param("LINE_SK_X", -1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::K && lastPh_ == Ph::S && kSkX >= 0) close.lineX = kSkX;
  }
  {
    const bool post = tok.ph == Ph::CH || tok.ph == Ph::JH;
    close.lineHw = post ? kLinePostHw : 1.0f;
    close.lineLip = post ? kLinePostLip : 0.0f;
    if (tok.ph == Ph::JH && kLineJhLip >= 0) close.lineLip = kLineJhLip;
    static const float kKwLip2 = param("LINE_KW_LIP", 0.0f);  // (LINE_KW_LIP, above)
    if (tract_ == Tract::Line && tok.ph == Ph::K && next && *next == Ph::W && kKwLip2 > 0) close.lineLip = kKwLip2;
    close.lineJetD = post ? kLinePostJetD : 0.0f;
    if (post) close.lineJetMix = kLinePostJetMix, close.lineSlitScale = kLinePostSlitScale, close.lineJetSmoothMs = kLinePostStSmoothMs, close.lineJetFcMin = kLinePostFcMin, close.lineJetFcMax = kLinePostFcMax;
    // t's jet (LINE_T_JET_D, cm; 0: the slit's height): the tongue tip closing
    // on and leaving the ridge made its noise through a slit of a few
    // hundredths of a mm, the band's corner at the 12 kHz clamp: a 12-24 kHz
    // click at the closure and the release, the "sizzle" of "not" (2026-11-05;
    // Kokoro's and the readers' t have nothing above 12 kHz).
    close.lineFricLpr = 0.0f;
    if (tract_ == Tract::Line && tok.ph == Ph::T && kLineTJetD > 0) close.lineJetD = kLineTJetD, close.lineFricLpr = kLineTLpr;
    if (tract_ == Tract::Line && tok.ph == Ph::P && kLinePJetD > 0) close.lineJetD = kLinePJetD, close.lineFricLpr = kLinePLpr;
    // t's closing hiss x LINE_T_CLOSE_FRIC (1: as before; back to 1 at the
    // release): the tongue tip passing the fricative areas onto the ridge with
    // the vowel's flow made a 4-24 kHz click at the vowel's end, -7 dB re the
    // vowel in "at" (2026-11-05; none in Kokoro or the readers).
    if (tract_ == Tract::Line && tok.ph == Ph::T && kLineTCloseFric != 1.0f) close.lineFricMul = kLineTCloseFric;
    close.lineSoft = pi.voiced ? kLineVoicedSoft : 1.0f;
    close.lineExtra = tok.ph == Ph::B ? kLineVoicedExtraCm : 0.0f;  // the lips only: they move forward off the teeth for b
    // LINE_G_EXPAND_FLOW (lab knob 2026-11-11, G agent 5; < 0: the shared LINE_EXPAND_FLOW): the g's own cavity
    // expansion. The velar closure leaves the smallest cavity behind it, so its voicing stops soonest and the pressure
    // builds for the burst; puhuja's g closure held 2500 dyn/cm^2 against the k's 7400 and released no burst.
    static const float kLineGExpandFlow = param("LINE_G_EXPAND_FLOW", -1.0f);
    close.lineExpand = tok.ph == Ph::JH ? kLineJhExpandFlow * (jhPos_ == 2 ? kLineJhFinalExpand : 1.0f) : tok.ph == Ph::G && kLineGExpandFlow >= 0 ? std::max(kLineGExpandFlow, 1e-3f) : 0.0f;
    if (tract_ == Tract::Line && tok.ph == Ph::JH && kLineJhCloseFric != 1.0f) close.lineFricMul = kLineJhCloseFric;
    if (tract_ == Tract::Line && kLineCloseFric != 1.0f && close.lineFricMul < 0 && tok.ph != Ph::T && tok.ph != Ph::JH && tok.ph != Ph::CH) close.lineFricMul = kLineCloseFric;
  }
  // Speaking again after a pause: full breath pressure (as emit() does for
  // the other classes; a plosive returns before that, so every phrase that
  // began with b d g p t k after a pause kept the ending's eased pressure,
  // 4-6 dB under: "..., but from the hand of God").
  // A phrase-initial stop's breath (LINE_PAUSE_STOP_PRESS_MS, 2026-11-04):
  // the lungs push only this long before the release, not as the mouth
  // closes. Pushing at the closure, the glottal flow's step behind the shut
  // mouth radiated through the walls as a 25 ms thump 50 ms before the
  // release, heard as a syllable of its own ("Back." -> "clack", "Gov." ->
  // "the sinner").
  const bool lateBreath = tract_ == Tract::Line && prevWasPause_ && kLinePauseStopPressMs > 0;
  if (prevWasPause_ && !lateBreath) {
    close.press = 1.0f;
    close.pressMs = kPressResumeMs;
  }
  if (lateBreath) {
    Event breath{std::max(release - kLinePauseStopPressMs * ms, t_ + 1.0)};
    breath.press = 1.0f;
    breath.pressMs = kLinePauseStopPressRampMs;
    addEventAt(breath);
  }
  close.speech = 1;  // speaking again (after a pause the level had faded)
  // The diaphragm does not keep pushing against a closed tract (Jyrki 2026-11-10: "probably the diaphragm senses the
  // pressure and doesn't keep building more pressure when a limit is achieved -- this might help with the
  // snappiness"): inside a full closure the lung pressure eases to LINE_CLOSURE_PRESS (1: off) over
  // LINE_CLOSURE_PRESS_MS, and comes back with the release (closurePressEased_).
  static const float kLineClosurePress = param("LINE_CLOSURE_PRESS", 1.0f), kLineClosurePressMs = param("LINE_CLOSURE_PRESS_MS", 30.0f);
  closurePressEased_ = false;
  if (tract_ == Tract::Line && kLineClosurePress < 1.0f && !nextIsPause && close.press < 0) {
    close.press = kLineClosurePress;
    close.pressMs = kLineClosurePressMs;
    closurePressEased_ = true;
  }
  // A voiced stop before a pause ("the hand of God."): the lung pressure eases to this share from the
  // closure on, so its voice bar and release come out softer (Jyrki 2026-11-09, with the final d kept
  // voiced: "d in God can be a bit less loud, perhaps -3 db"; 0.7 is about -3 dB).
  {
    static const float kLineFinalVstopPress = param("LINE_FINAL_VSTOP_PRESS", 0.8f);
    if (tract_ == Tract::Line && nextIsPause && pi.voiced && kLineFinalVstopPress < 1.0f && close.press < 0) {
      close.press = kLineFinalVstopPress;
      close.pressMs = 30;
    }
  }
  // LINE_D_PRESS (lab knob 2026-11-10, D agent): the lung pressure through a d's closure (not before a pause: that is
  // LINE_FINAL_VSTOP_PRESS), back to 1 at the release; the stored pressure makes the pop (1: as before).
  static const float kLineDPress = param("LINE_D_PRESS", 1.0f);
  const bool dPress = tract_ == Tract::Line && tok.ph == Ph::D && !nextIsPause && kLineDPress < 1.0f;
  if (dPress) { close.press = kLineDPress; if (close.pressMs < 0) close.pressMs = 20; }  // (after a pause the resume above set press 1: that ramp goes to this instead)
  close.ac = 0;
  // ʤ: a leak through its "closure" (LINE_JH_CLOSE_AC, cm^2; 0: sealed). Sealed, the folds weaken 15 dB over the 18 ms
  // seal ramp and restart behind it with a lone loud pulse and alternating weak ones ("a strange klik before g", Jyrki
  // 2026-11-06) where Kokoro's ʤ is voiced throughout. Tested 0.02-0.1: the leak weakened the voicing further; 0 kept.
  {
    static const float jhCloseAc = param("LINE_JH_CLOSE_AC", 0.0f);
    if (tok.ph == Ph::JH && jhCloseAc > 0) close.ac = jhCloseAc;
  }
  // A tap on the line (LINE_FLAP_AC, 2026-11-05): the tongue touches without
  // sealing, the voice runs through it (Kokoro's "mortal": a 20 ms dip of
  // 15-28 dB, no burst); a full seal gave a 50 dB gap and two clicks.
  {
    static const float flapAc = param("LINE_FLAP_AC", 0.05f);
    if (tract_ == Tract::Line && tok.shape.flap && flapAc > 0) {
      close.ac = flapAc;
      close.lineFricFade = 0.2f;  // no hiss through the touch (its 0.05 cm^2 is a fricative's slit)
    }
  }
  if (ldSeal) close.lineFricFade = kLdSealFade;  // (LINE_LD_SEAL_FADE, above)
  // D agent9 round J lab knobs (0 / < 0: off), by the d's context (english.cc Shape::dctx): a d between a vowel and an
  // unstressed vowel (1: "reader", "lady") held with a slit of LINE_D_VV_AC cm^2 (the voice runs through: Kokoro's
  // d there dips 13-17 dB with its voice bar at -12..-14 dB re the word peak, puhuja's closure -27..-29 dB, bar -21; the
  // dictionary hears 224 of these as t) and with the cavity's expansion flow LINE_D_VV_EXPAND (cm^3/s); a word-final
  // d (3 after n l, 4 after a vowel) with LINE_D_FINAL_EXPAND.
  {
    static const float kDVvAc = param("LINE_D_VV_AC", 0.0f), kDVvExpand = param("LINE_D_VV_EXPAND", -1.0f),
                       kDFinalExpand = param("LINE_D_FINAL_EXPAND", -1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::D && !tok.shape.flap) {
      if (tok.shape.dctx == 1 && kDVvAc > 0) { close.ac = kDVvAc; close.lineFricFade = 0.2f; }
      static const float kDLvAc = param("LINE_D_LV_AC", 0.0f);  // ... and between l and an unstressed vowel (7: "builder" -> belter)
      if (tok.shape.dctx == 7 && kDLvAc > 0) { close.ac = kDLvAc; close.lineFricFade = 0.2f; }
      if (tok.shape.dctx == 1 && kDVvExpand >= 0) close.lineExpand = std::max(kDVvExpand, 1e-3f);
      if ((tok.shape.dctx == 3 || tok.shape.dctx == 4) && kDFinalExpand >= 0) close.lineExpand = std::max(kDFinalExpand, 1e-3f);
      static const float kDzDExpand = param("LINE_DZ_D_EXPAND", -1.0f);  // ... a d before s z (2): "beads", "hands"
      if (tok.shape.dctx == 2 && kDzDExpand >= 0) close.lineExpand = std::max(kDzDExpand, 1e-3f);
    }
  }
  close.acMs = sealMs;
  close.an = nasalVel ? -1.0f : 0.0f;  // (LINE_NASAL_STOP_VEL_MS: the velum stays open, closed below)
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
  if ((pi.voiced && !englishB && !englishGD && !devoiceBefore) || glottalised) {
    // Voiced (or creaky) closure: the speaker expands the mouth cavity
    // (larynx down, cheeks slack), so air keeps flowing and the folds keep
    // vibrating.
    close.expand = kVoicedExpansion;
    close.expandMs = closeMs;
  }
  // (the cascade's b d g burst, LINE_VSTOP_BURST; the line's is at the release)
  if (tract_ != Tract::Line && (tok.ph == Ph::B || tok.ph == Ph::D || tok.ph == Ph::G)) close.noiseGain *= kLineVStopBurst;
  addEvent(close);
  if (nasalVel) {  // the velum closes LINE_NASAL_STOP_VEL_MS before the release (the pressure for the burst builds only then)
    Event vel{std::max(release - kNasalStopVelMs * ms, tClosed + 1.0)};
    vel.an = 0;
    vel.anMs = std::max(1.0f, static_cast<float>((release - vel.t) / ms));
    addEventAt(vel);
  }
  // Glottis during the closure: opens for voiceless plosives, vibrating
  // position for voiced ones (also needed after a pause, when it is shut).
  Event glottis{timeline_};
  glottis.ag = glottalised ? kGlottalisedAg
               : pi.closureAg >= 0 ? pi.closureAg : pi.voiced ? kAgVoice : kAgOpen;
  glottis.agMs = glottalised ? kGlottalMs
                 : pi.voiced ? kGlottalMs : 0.6f * static_cast<float>(closure / ms);
  if (tract_ == Tract::Line && !glottalised && !pi.voiced && pi.closureAg < 0 && kLineVstopAgLagMs > 0) glottis.t += kLineVstopAgLagMs * ms;
  // A devoiced English b d g (LINE_EN_GD_DEVOICE): the glottis opens only
  // LINE_GD_AG_LAG_MS after the closure has sealed. Opening with the closure
  // still forming, the breath through the half-open glottis and the closing
  // mouth made a 2-8 kHz puff: the "blib" at the end of "of" in "hand of God"
  // (v straight into g, Jyrki 2026-11-05). 0: with the closure, as before.
  if (tract_ == Tract::Line && !glottalised && pi.voiced && pi.closureAg >= 0 && kLineGdAgLagMs > 0) glottis.t += kLineGdAgLagMs * ms;  // (LINE_VSTOP_AG_LAG_MS)
  if (vstopLead) {
    // ... the folds come together into voice once the mouth is closed, and
    // no earlier than LINE_VSTOP_LEAD_MS before the release (b's devoiced
    // closure, LINE_G_CLOSURE_AG: voiced here, it is the voice bar).
    glottis.t = std::max(release - kLineVStopLeadMs * ms, tClosed);
    glottis.ag = kAgVoice;
  }
  addEvent(glottis);

  // Release: the closure opens toward the next sound. The trapped air rushes
  // out as the burst; the glottis closes again and voicing restarts once the
  // pressure has fallen.
  Event rel{release};
  if (dPress) { rel.press = 1.0f; rel.pressMs = 20; }  // (LINE_D_PRESS, above)
  bool dzRel = false;  // (LINE_DZ_REL_FRIC, below)
  float dzBackFric = 1.0f;
  {
    static const float tAspHp = param("LINE_T_ASP_HP", 0.0f);
    if (tract_ == Tract::Line && tAspHp > 0) rel.lineAspHp = tok.ph == Ph::T ? tAspHp : 0.0f;
  }
  if (ldSeal) {  // (LINE_LD_SEAL_FADE: the burst's own hiss, and a next fricative's fade as emit() sets it: "holds")
    static const float vFade = param("LINE_VFRIC_FADE", 1.0f), shFade = param("LINE_SH_FADE", 1.0f);
    const Ph n = next ? *next : Ph::Stop;
    rel.lineFricFade = n == Ph::Z || n == Ph::ZH || n == Ph::VV || n == Ph::DH ? vFade : n == Ph::SH ? shFade : 1.0f;
  }
  bool aspiratedRelease = false;  // (for the diaphragm's push, LINE_PULSE)
  rel.expand = 1;
  rel.expandMs = kVoicedExpansionReleaseMs;
  {  // (D agent9 round J lab knob; <= 1: off) the cavity walls' expansion sink through a z after a d, from the d's release
    static const float kDzZExpand2 = param("LINE_DZ_Z_EXPAND", 0.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::D && next && *next == Ph::Z && kDzZExpand2 > 1.0f) { rel.expand = kDzZExpand2; rel.expandMs = 10; }
  }
  rel.ac = nextIsPause ? kAcOpen : nextAero.ac;
  // On the line the release is also the next sound's own event (emit drops
  // the event of a sound after a plosive): it carries that sound's slit, place
  // and groove. Without them the f of "not forget" was made at the t's
  // alveolar place with the table's 0.1 cm^2: an s-like hiss 12 dB over the
  // vowels, 600 cm^3/s through what should be a near-sealed lip (2026-11-03).
  // A nasal after a stop (LINE_REL_NASAL_X 1; 0: as before) closes at its own place: the release
  // is its event too, and it set no place, so "and me" d -> m, "atmosphere", "acne" made the whole
  // nasal at the stop's place (m at x 0.83 throughout: an n murmur; PUHUJA_X_TRACE, 2026-11-11).
  static const float kLineRelNasalX = param("LINE_REL_NASAL_X", 1.0f);
  if (tract_ == Tract::Line && kLineRelNasalX > 0 && next && !nextIsPause && info(*next).cls == Cls::Nasal)
    rel.lineX = linePlaceOf(*next), rel.lineXClosure = true;
  // The same for an l (LINE_L_REL_OWN, L agent round E; 0: as before): its event dropped, the l after a plosive kept the
  // plosive's place (p b: the lips, k g: the velum at 0.62) with the table's 0.35 cm^2 instead of the l's ridge and
  // LINE_L_AC: "clean" F3 2020 Hz, "glass" 2210 (Kokoro 2330-2440), "black" F2 646 (Kokoro 900-1000): heard as ɹ
  // (pl bl kl gl -> ɹ: 251 of the 385 l errors after a plosive). x LINE_L_STOP_AC.
  static const float kLRelOwn = param("LINE_L_REL_OWN", 1.0f);
  if (tract_ == Tract::Line && kLRelOwn > 0 && next && *next == Ph::L) {
    static const float kLRelStopAc = param("LINE_L_STOP_AC", 1.0f);
    rel.ac = lineSlit(Ph::L, rel.ac) * kLRelStopAc;
    static const float kLRelX = param("LINE_L_REL_X", 0.8f);  // the released l's place (< 0: the l's own; a little back lowers F3)
    rel.lineX = kLRelX >= 0 ? kLRelX : linePlaceOf(Ph::L);
  }
  // The same for an ɹ (LINE_R_REL_OWN, R agent round F; 0: as before): the release kept the plosive's place with the
  // table's 0.25 cm^2 (then LINE_R_AC's 0.6 from the second event, still at the plosive's place): the ɹ of "bring",
  // "price" was a lip constriction at x 0.97 (a w), of "tree", "dry" one at the ridge (0.90), of "great" at the velum
  // (0.62), instead of the bunched hump at LINE_R_X. Its place x LINE_R_REL_X (< 0: the ɹ's own), its hump x LINE_R_REL_AC.
  static const float kRRelOwn = param("LINE_R_REL_OWN", 1.0f);
  if (tract_ == Tract::Line && kRRelOwn > 0 && next && *next == Ph::RR) {
    static const float kRRelAc = param("LINE_R_REL_AC", 1.5f), kRRelX = param("LINE_R_REL_X", -1.0f);
    rel.ac = lineSlit(Ph::RR, rel.ac) * kRRelAc;
    rel.lineX = kRRelX >= 0 ? kRRelX : linePlaceOf(Ph::RR);
  }
  // (K-INITIAL agent 13 round L, lab; < 0: as above) a k's release into l / ɹ at LINE_KL_REL_X / LINE_KR_REL_X. The
  // closure hump slides from the velum (x 0.62) to the l's ridge (0.8) while it opens: kl kɹ bursts are diffuse with a
  // 0.5-1 kHz thump (-3 dB of the burst; Kokoro's compact at 1-2 kHz), heard as p ("Cried" -> pride, "Claimed" -> planned).
  // (The hump area here is overridden by the liquid's second step, LINE_R_REL_AC: an area knob was inert.)
  static const float kKlRelX = param("LINE_KL_REL_X", -1.0f), kKrRelX = param("LINE_KR_REL_X", -1.0f);
  if (tract_ == Tract::Line && tok.ph == Ph::K && next && *next == Ph::L && kLRelOwn > 0 && kKlRelX >= 0) rel.lineX = kKlRelX;
  if (tract_ == Tract::Line && tok.ph == Ph::K && next && *next == Ph::RR && kRRelOwn > 0 && kKrRelX >= 0) rel.lineX = kKrRelX;
  // ... and for a w or j (LINE_GLIDE_REL_OWN, R agent round F, lab; 0: as before): the same dropped event, "twin" made
  // its w at the ridge, "cute" its j at the velum (w after p t k: 21 % errors, j after p b: 23-27 %, 9 % after n).
  static const float kGlideRelOwn = param("LINE_GLIDE_REL_OWN", 1.0f);
  if (tract_ == Tract::Line && kGlideRelOwn > 0 && next && *next == Ph::W) {
    rel.ac = lineSlit(*next, rel.ac);
    rel.lineX = linePlaceOf(*next);
    // (W agent round G, lab; 1 / -1: as before) the w's release area x LINE_W_REL_AC, its place (< 0: the lips).
    static const float kWRelAc = param("LINE_W_REL_AC", 1.0f), kWRelX = param("LINE_W_REL_X", -1.0f);
    if (*next == Ph::W) {
      rel.ac *= kWRelAc;
      if (kWRelX >= 0) rel.lineX = kWRelX;
    }
  }
  // (Y agent round G, lab) the j alone: LINE_J_REL_OWN (< 0: LINE_GLIDE_REL_OWN), its place LINE_J_REL_X (< 0: the j's),
  // its slit x LINE_J_REL_AC, the opening time LINE_J_REL_MS (< 0: the plosive's), the release's frication gain
  // LINE_J_REL_FRIC (< 0: unchanged). The j of "pure", "beauty" was a 0.3 cm^2 constriction at the lips (F2 1130-1480 Hz,
  // Kokoro 2100-2450): "Bureau" -> plural, "Cue" -> pew. With LINE_J_STOP_W 4, _LEAD_MS 30, REL_MS 30 (30 ms: the restored
  // p/b burst no snappier than the base's), LINE_U_AFTER_STOP_J 0.2: dictionary j words 66 -> 48 Y errors (two draws).
  static const float kJRelOwn = param("LINE_J_REL_OWN", 1.0f), kJRelX = param("LINE_J_REL_X", -1.0f),
                     kJRelAc = param("LINE_J_REL_AC", 1.0f), kJRelMs = param("LINE_J_REL_MS", 30.0f),
                     kJRelFric = param("LINE_J_REL_FRIC", -1.0f);
  const bool jRel = tract_ == Tract::Line && next && *next == Ph::J && (kJRelOwn >= 0 ? kJRelOwn : kGlideRelOwn) > 0;
  if (jRel) {
    rel.ac = lineSlit(Ph::J, rel.ac) * kJRelAc;
    rel.lineX = kJRelX >= 0 ? kJRelX : linePlaceOf(Ph::J);
  }
  if (tract_ == Tract::Line && next && !nextIsPause && info(*next).cls == Cls::Fricative) {
    rel.ac = lineSlit(*next, rel.ac);
    rel.lineX = linePlaceOf(*next);
    const bool post = *next == Ph::SH || *next == Ph::ZH;
    rel.lineHw = post ? kLinePostHw : 1.0f;
    rel.lineLip = post ? kLinePostLip : 0.0f;
    if (post) rel.lineJetD = kLinePostJetD, rel.lineJetMix = kLinePostJetMix, rel.lineSlitScale = kLinePostSlitScale, rel.lineJetSmoothMs = kLinePostStSmoothMs, rel.lineJetFcMin = kLinePostFcMin, rel.lineJetFcMax = kLinePostFcMax, rel.lineFricLpr = kLinePostLpr;
    if (tok.ph == Ph::JH) {  // (the ʤ's own frication knobs, LINE_JH_FRIC_X ...: the release carries the ʒ's targets, its own event is not queued)
      if (kLineJhFade >= 0) rel.lineFricFade = kLineJhFade;
      if (kLineJhPress < 1.0f) { rel.press = kLineJhPress; rel.pressMs = 20; jhPressed_ = true; }
      if (kLineJhFricX >= 0) rel.lineX = kLineJhFricX;
      if (kLineJhLip >= 0) rel.lineLip = kLineJhLip;
      if (kLineJhJetD >= 0) rel.lineJetD = kLineJhJetD;
      if (kLineJhLpr >= 0) rel.lineFricLpr = kLineJhLpr;
      if (kLineJhFcMin >= 0) rel.lineJetFcMin = kLineJhFcMin;
      if (kLineJhHw >= 0) rel.lineHw = kLineJhHw;
      if (kLineJhSlit != 1.0f) rel.ac *= kLineJhSlit;
    }
    // ... and f's own jet, band, lip width and gain (LINE_STOP_F_OWN): the f
    // of "not forget", "at first" was made with the slit's jet and the full
    // width, a +10 dB, 4-24 kHz burst where the readers' t is silent and their
    // f 25 dB under the vowel (2026-11-05).
    {
      static const float zFric = param("LINE_Z_FRIC", 1.0f);
      if (*next == Ph::Z && zFric != 1.0f) rel.lineFricMul = zFric;
      // d released into z ("deeds in", "needs it"; LINE_DZ_REL_FRIC, 2026-11-06, Jyrki: "end of deeds is too buzzy"):
      // the d's pressure jets through the forming z groove over the 32 ms release at the z's full gain, a 20 ms hiss
      // at the vowel's level (its 2 ms peak +9 dB re the vowel; Kokoro's z after d -6..-8, and its steady hiss 15
      // dB under the vowel). The slit gain x this while the groove forms, the z's own (LINE_Z_FRIC) once it has
      // (rel.acMs + 10 ms later). < 0: off.
      static const float dzRelFric = param("LINE_DZ_REL_FRIC", 0.3f);
      dzRel = tract_ == Tract::Line && dzRelFric >= 0 && *next == Ph::Z && pi.voiced;
      if (dzRel) rel.lineFricMul = dzRelFric * zFric;
      dzBackFric = zFric;
      // Z-FINAL agent13 round L (lab; 1: as before): the z after b d g ("beds", "dogs": nearly always word-final) x
      // LINE_BZ_FRIC. It rides on the stop's release and never got LINE_FINAL_Z_FRIC: its hiss -15 / -10 dB re the
      // word's peak (m / f), Kokoro's -21..-24.
      static const float kBzFric = param("LINE_BZ_FRIC", 1.0f);
      if (tract_ == Tract::Line && kBzFric != 1.0f && *next == Ph::Z && pi.voiced) {
        if (rel.lineFricMul >= 0) rel.lineFricMul *= kBzFric;
        dzBackFric *= kBzFric;
      }
      // ... and its jet's Reynolds number x LINE_BZ_RE (1: off; back to 1 at the next phrase's first sound): the hiss
      // starts while the folds still run on from the stop's closure.
      static const float kBzRe = param("LINE_BZ_RE", 1.0f);
      if (tract_ == Tract::Line && kBzRe != 1.0f && *next == Ph::Z && pi.voiced) rel.lineFricReMul = kBzRe;
    }
    if (*next == Ph::F && kLineStopFOwn > 0) {
      rel.lineHw = kLineFHw;
      rel.lineJetD = curStress_ ? kLineFJetD : kLineFJetDUnstr;
      rel.lineFricLpr = kLineFLpr;
      rel.lineFricMul = curStress_ ? kLineFFric : kLineFFricUnstr;
    }
    if (tok.ph == Ph::T && kLineTJetD > 0 && rel.lineJetD < 0) rel.lineJetD = rel.lineFricLpr = 0.0f;  // (the s of "its": its own jet)
    if (tract_ == Tract::Line && tok.ph == Ph::P && kLinePJetD > 0 && rel.lineJetD < 0) rel.lineJetD = kLinePJetD, rel.lineFricLpr = kLinePLpr;  // (p bursts through its thick jet)
  }
  if (tract_ == Tract::Line && pi.voiced && nextFric && !nextVoicedFric && tok.ph != Ph::JH && !unreleased) {
    // The voiced stop's burst into a voiceless fricative at a fraction of the
    // fricative's gain (the 75/25 mix above; 1: as before).
    static const float kVstopFricRelGain = param("LINE_VSTOP_FRIC_REL_GAIN", 1.0f);
    if (kVstopFricRelGain != 1.0f) rel.lineFricMul = (rel.lineFricMul < 0 ? 1.0f : rel.lineFricMul) * kVstopFricRelGain;
  }
  // On the line a voiced stop opens more slowly than a voiceless one: the
  // lips part softly, the burst keeps all its frequencies but builds over a
  // longer time (Jyrki 2026-11-03: "open in a way that there are all
  // frequencies but slower than p").
  static const float kLineDReleaseScale = param("LINE_D_RELEASE_SCALE", -1.0f);  // (lab knob 2026-11-10, D agent; < 0: LINE_RELEASE_SCALE x LINE_VOICED_RELEASE): the d's opening time re its releaseMs
  rel.acMs = pi.releaseMs * (tract_ == Tract::Line ? (tok.ph == Ph::D && kLineDReleaseScale >= 0 ? kLineDReleaseScale : (tok.ph == Ph::P && kLinePReleaseScale >= 0 ? kLinePReleaseScale : tok.ph == Ph::K && kLineKReleaseScale >= 0 ? kLineKReleaseScale : tok.ph == Ph::G && kLineGReleaseScale >= 0 ? kLineGReleaseScale : tok.ph == Ph::T && kLineTReleaseScale >= 0 ? kLineTReleaseScale : kLineReleaseScale) * (pi.voiced ? kLineVoicedReleaseScale : 1.0f)) : 1.0f);
  // LINE_FINAL_D_RELEASE_SCALE (lab knob 2026-11-10, D agent; < 0: off): a d before a pause opens over releaseMs x this
  // (Kokoro's final d: a soft -21..-24 dB release where puhuja's pops at -9).
  static const float kLineFinalDReleaseScale = param("LINE_FINAL_D_RELEASE_SCALE", -1.0f);
  if (tract_ == Tract::Line && tok.ph == Ph::D && nextIsPause && kLineFinalDReleaseScale >= 0) rel.acMs = pi.releaseMs * kLineFinalDReleaseScale;
  {  // (D agent9 round J lab knob; 1: as before) a d between a vowel and an unstressed vowel opens x LINE_D_VV_REL faster
    static const float kDVvRel = param("LINE_D_VV_REL", 1.0f);
    if (tract_ == Tract::Line && tok.ph == Ph::D && tok.shape.dctx == 1 && kDVvRel != 1.0f) rel.acMs *= kDVvRel;
  }
  if (jRel && kJRelMs >= 0) rel.acMs = kJRelMs;  // (LINE_J_REL_MS)
  // A voiced stop before a fricative, released (LINE_VSTOP_FRIC_UNREL 0): the
  // release slowed x this, a softer pop (Jyrki 2026-11-07 on "and foo": the
  // released one better than the unreleased, "but mix 75 % of unrel0 and 25 %
  // of unrel1"; 1: as before).
  {
    static const float kVstopFricRelSlow = param("LINE_VSTOP_FRIC_REL_SLOW", 1.0f);
    if (tract_ == Tract::Line && pi.voiced && nextFric && tok.ph != Ph::JH && !unreleased && kVstopFricRelSlow != 1.0f) rel.acMs *= kVstopFricRelSlow;
  }
  // Into a dental fricative ("that the", "around the") the tongue slides
  // from the ridge to the teeth: a slow opening, so the ð noise grows in
  // (opened in 12 ms it rose 45 dB in 8 ms, a snap).
  if (tract_ == Tract::Line && tok.ph == Ph::JH && kLineJhReleaseScale >= 0) rel.acMs = pi.releaseMs * kLineJhReleaseScale;
  if (glottalised || dentalD) rel.acMs = std::max(rel.acMs, kDentalReleaseMs);
  if (nasalVel && kNasalStopRel != 1.0f) rel.acMs *= kNasalStopRel;  // (LINE_NASAL_STOP_REL)
  {  // (R agent round F, lab; < 0: as before) a plosive opens into an ɹ over this many ms
    static const float kRRelMs = param("LINE_R_REL_MS", -1.0f);
    if (tract_ == Tract::Line && kRRelMs >= 0 && next && *next == Ph::RR) rel.acMs = kRRelMs;
  }
  {  // (W agent round G, lab; < 0: as before) a plosive opens into a w over LINE_W_REL_MS ms
    static const float kWRelMs = param("LINE_W_REL_MS", -1.0f);
    if (tract_ == Tract::Line && next && *next == Ph::W) {
      if (kWRelMs >= 0) rel.acMs = kWRelMs;
    }
  }
  // Unreleased into a plosive: stays closed. Unreleased into a fricative: the
  // closure opens straight into the fricative's slit over LINE_VSTOP_FRIC_REL_MS
  // (the pressure vents through the forming slit, no pop; 2026-11-07, "and
  // foe": the pending release's ac 0 had overridden the f's own slit event at
  // the same instant, and the f was silent for its whole length).
  if (unreleased) {
    static const float kVstopFricRelMs = param("LINE_VSTOP_FRIC_REL_MS", 40.0f);
    if (nextFric && tract_ == Tract::Line) rel.acMs = kVstopFricRelMs;
    else rel.ac = 0;
  }
  rel.an = nextAero.an;
  rel.anMs = kBoundaryMs;
  if (nextIsPause) {
    rel.ag = 0;
    rel.agMs = 40;
  } else if (glottalised) {
    rel.ag = nextAero.ag;  // straight into voicing for ð
    rel.agMs = kGlottalMs;
  } else if (aspirate_ && !pi.voiced && !glottalised && next && (lastPh_ != Ph::S || (tok.ph == Ph::K && kLineSkAsp > 0)) &&
             (isVowel(*next) || info(*next).cls == Cls::Liquid || info(*next).cls == Cls::Glide ||
              info(*next).cls == Cls::Trill)) {
    // English aspiration: the glottis stays open past the release and closes
    // slowly; voicing starts some 50-70 ms later, after a breathy puff.
    rel.ag = nextAero.ag;
    rel.agMs = curStress_ ? kAspiratedAdductMs : kAspiratedUnstressedMs;
    rel.aspBurst = true;  // the puff's breath noise (x aspirationGain_), fading
    if (tract_ == Tract::Line) {
      // On the line the puff is the open glottis itself: it stays open for
      // kLineVotHoldMs after the release (a clean puff of air through the
      // opened tongue closure, hiss only), then adducts quickly into voice.
      // The slow glide gave a breathy voiced transition instead (HF share
      // -8..-16 dB in the 40 ms after the t of "poverty"; goog's puff 0 dB),
      // Jyrki 2026-11-02: "how the puff of air meets the tongue".
      // Open wide: at kAgOpen (0.3 cm^2) the string folds kept vibrating,
      // a breathy voice under 200 Hz in place of the hiss (2026-11-04).
      rel.ag = tok.ph == Ph::P && kLinePAspAg >= 0 ? kLinePAspAg : tok.ph == Ph::K && kLineKAspAg >= 0 ? kLineKAspAg : tok.ph == Ph::T && kLineTAspAg >= 0 ? kLineTAspAg : kLineAspAg;
      rel.agMs = 5;
      // ... and the puff hisses: the turbulence of the full lung pressure
      // through the open glottis, far stronger than an h's (LINE_ASP alone
      // left it 30 dB under the readers' 3-5 kHz puff).
      rel.lineAspMul = tok.ph == Ph::P && kLinePStopAspMul >= 0 ? kLinePStopAspMul : tok.ph == Ph::T && kLineTStopAspMul >= 0 ? kLineTStopAspMul
                       : tok.ph == Ph::K && kLineKStopAspMul >= 0 ? kLineKStopAspMul : kLineStopAspMul;
      // The female's puffs ran 3 dB over her vowels where the male's were 5 dB under
      // ("power to abolish", 2026-11-05): her puff noise x LINE_PUFF_FEMALE.
      static const float kLinePuffFemale = param("LINE_PUFF_FEMALE", 0.25f);
      if (female_) rel.lineAspMul *= tok.ph == Ph::K && kLineKPuffFemale >= 0 ? kLineKPuffFemale : tok.ph == Ph::T && kLineTPuffFemale >= 0 ? kLineTPuffFemale : kLinePuffFemale;
      // A word-medial p t k puff x LINE_PUFF_MEDIAL (2026-11-06): Kokoro's medial p ("oppose", "support", -29..-52 dB
      // burst re the vowel) puffs ~10 dB under its word-initial ones ("poverty", "powers", -17..-22), whatever the
      // syllable's stress; LINE_VOT_HOLD_UNSTR shortens an unstressed one only.
      static const float kLinePuffMedial = param("LINE_PUFF_MEDIAL", 0.3f);
      if (!tok.wordStart) rel.lineAspMul *= tok.ph == Ph::K && kLineKPuffMedial >= 0 ? kLineKPuffMedial : tok.ph == Ph::T && kLineTPuffMedial >= 0 ? kLineTPuffMedial : kLinePuffMedial;
      // A p before l ɹ w j (LINE_P_LIQ_VOT x the puff's hold, LINE_P_LIQ_ASP x its noise; lab knobs 2026-11-10, P agent:
      // "price" -> "lice", "provide" -> "revived"; 1: as before).
      static const float kLinePLiqVot = param("LINE_P_LIQ_VOT", 1.0f), kLinePLiqAsp = param("LINE_P_LIQ_ASP", 3.0f);
      const bool pLiq = tok.ph == Ph::P && !isVowel(*next);
      if (pLiq) rel.lineAspMul *= kLinePLiqAsp;
      const bool tLiq = tok.ph == Ph::T && !isVowel(*next);  // (LINE_T_LIQ_VOT / _ASP)
      if (tLiq) rel.lineAspMul *= kLineTLiqAsp;
      // (K agent 9 round J, lab; 1: as before) a k before l ɹ w j: the puff's hold x LINE_K_LIQ_VOT, its noise x
      // LINE_K_LIQ_ASP (the liquid devoiced, as the p's and t's): kɹ was heard as gɹ 21 times, p 58 ("Cross" -> gross).
      // A word-initial unstressed k's hold x LINE_K_VOT_UNSTR_INIT instead of LINE_VOT_HOLD_UNSTR (< 0: as that):
      // "Career", "Correct", "Corrupt", "Karate" -> g (kə- 31 % wrong, g 28 of 52).
      static const float kKLiqVot = param("LINE_K_LIQ_VOT", 1.0f), kKLiqAsp = param("LINE_K_LIQ_ASP", 1.0f),
                         kKVotUnstrInit = param("LINE_K_VOT_UNSTR_INIT", -1.0f);
      const bool kLiq = tok.ph == Ph::K && !isVowel(*next);
      if (kLiq) rel.lineAspMul *= kKLiqAsp;
      // (K-INITIAL agent 13 round L, lab; 1: as before) a word-initial stressed k's puff: its hold x LINE_K_INIT_VOT, its
      // noise x LINE_K_INIT_ASP. puhuja's VOT is 40-47 ms in every context, Kokoro's 55-91 before stressed vowels and l ɹ
      // (a short lag is the cue of p b g; the velar's is the longest): initial k heard as p 159 of 5608, g 78.
      static const float kKInitVot = param("LINE_K_INIT_VOT", 1.0f), kKInitAsp = param("LINE_K_INIT_ASP", 1.0f);
      const bool kInitStr = tok.ph == Ph::K && tok.wordStart && curStress_;
      if (kInitStr) rel.lineAspMul *= kKInitAsp;
      const float unstrHold = tok.ph == Ph::K && tok.wordStart && kKVotUnstrInit >= 0 ? kKVotUnstrInit : kLineVotHoldUnstressed;
      // (Y agent round G, lab; 1: as before) a p t k before j: the puff's noise x LINE_J_ASP, its hold x LINE_J_VOT
      // (an English pj kj devoices the j: [pçʊ]).
      static const float kJAsp = param("LINE_J_ASP", 1.0f), kJVot = param("LINE_J_VOT", 1.0f);
      const bool stopJ = *next == Ph::J;
      if (stopJ && kJAsp != 1.0f) rel.lineAspMul *= kJAsp;
      // (W agent round G, lab; 1 / -1: as before) a p t k before w: the puff's hold x LINE_W_VOT, its noise x LINE_W_ASP,
      // the adduction into voice over LINE_W_ADDUCT_MS (< 0: LINE_VOT_ADDUCT_MS). "twice": the puff ran through the
      // whole 64 ms w, voiced only its last 25 ms (Kokoro's: ~45 ms of voiced w at F2 800 after the puff).
      static const float kWVot = param("LINE_W_VOT", 1.0f), kWAsp = param("LINE_W_ASP", 1.0f), kWAdductMs = param("LINE_W_ADDUCT_MS", -1.0f);
      const bool wNext = *next == Ph::W;
      if (wNext) rel.lineAspMul *= kWAsp;
      // (K-INITIAL agent 13 round L, lab; < 0: as before) a word-initial stressed k's adduction into voice over
      // LINE_K_INIT_ADDUCT_MS: Kokoro's voice comes in over 10-20 ms of breathy aspiration after an initial k ("Cross"),
      // puhuja's at once (+20 dB in 3 ms).
      static const float kKInitAdductMs = param("LINE_K_INIT_ADDUCT_MS", -1.0f);
      const float adductMs = wNext && kWAdductMs >= 0 ? kWAdductMs : kInitStr && kKInitAdductMs >= 0 ? kKInitAdductMs : kLineVotAdductMs;
      Event adductLine{release + (tok.ph == Ph::P && kLinePVotHoldMs >= 0 ? kLinePVotHoldMs : tok.ph == Ph::K && kLineKVotHoldMs >= 0 ? kLineKVotHoldMs : tok.ph == Ph::T && kLineTVotHoldMs >= 0 ? kLineTVotHoldMs : kLineVotHoldMs) * (curStress_ ? 1.0f : unstrHold) * (pLiq ? kLinePLiqVot : 1.0f) * (tLiq ? kLineTLiqVot : 1.0f) * (kLiq ? kKLiqVot : 1.0f) * (kInitStr ? kKInitVot : 1.0f) * (lastPh_ == Ph::S ? kLineSkAsp : 1.0f) * (wNext ? kWVot : 1.0f) * ms};
      if (stopJ && kJVot != 1.0f) adductLine.t = release + (adductLine.t - release) * kJVot;
      adductLine.ag = nextAero.ag;
      // (W agent round G, lab; < 0: as before) ... and adducts only to LINE_W_STOP_AG (cm^2): a breathy w (kw tw devoiced)
      static const float kWStopAg = param("LINE_W_STOP_AG", -1.0f);
      if (wNext && kWStopAg >= 0) adductLine.ag = kWStopAg;
      adductLine.agMs = adductMs;
      pendingKick_ = adductLine;
      // The hiss lasts while the folds close (cut at the adduction's start it
      // left a 10 ms silence before the voice).
      // (queued after the release, with the adduction, below)
      pendingAspEnd_ = Event{adductLine.t + adductMs * ms};
      pendingAspEnd_.lineAspMul = 1.0f;
      if (kLinePuffLpr > 0) {
        rel.linePuffLpr = tok.ph == Ph::P && kLinePPuffLpr > 0 ? kLinePPuffLpr : tok.ph == Ph::K && kLineKPuffLpr > 0 ? kLineKPuffLpr : tok.ph == Ph::T && kLineTPuffLpr > 0 ? kLineTPuffLpr : kLinePuffLpr;
        pendingAspEnd_.linePuffLpr = 0.0f;
      }
      hasPendingAspEnd_ = true;
      hasPendingKick_ = true;
    }
    // A word that carries its stress on the syllable this stop opens
    // ("tiger", "tyranny", "power"): a longer, stronger puff. The user: the
    // more powerful the word, the more it gains from an aspirated p t k.
    if (tok.wordStart && curStress_ == 2) {
      rel.agMs *= kStrongAspirationLength;
      rel.aspBurstGain = kStrongAspirationGain;
    }
    aspirating_ = true;
    aspiratedRelease = true;
  } else if (!pi.voiced && pi.adductLeadMs > 0) {
    // The glottis closes during the end of the closure, overlapping the
    // release, so voicing restarts soon after it.
    Event adduct{release - std::min<double>(pi.adductLeadMs * ms, 0.5 * closure)};
    adduct.ag = nextAero.ag;
    adduct.agMs = pi.adductMs;
    addEvent(adduct);
  } else {
    rel.ag = nextAero.ag;
    // LINE_G_VOT_MS (lab knob 2026-11-09, G agent; < 0: kGlottalMs): how long the devoiced g's glottis takes to close
    // again after the release (Kokoro's initial g: 15-25 ms of noisy VOT, puhuja's voicing was back in 15 ms).
    static const float kLineGVotMs = param("LINE_G_VOT_MS", -1.0f);
    static const float kLineDVotMs = param("LINE_D_VOT_MS", -1.0f);  // (the same for the d; D agent 2026-11-10)
    rel.agMs = pi.voiced ? (tok.ph == Ph::G && kLineGVotMs >= 0 ? kLineGVotMs : tok.ph == Ph::D && kLineDVotMs >= 0 ? kLineDVotMs : kGlottalMs) : pi.adductMs;
    if (tract_ == Tract::Line && tok.ph == Ph::JH && kLineJhAg >= 0) rel.ag = jhPos_ >= 0 && kLineJhPosAg[jhPos_] >= 0 ? kLineJhPosAg[jhPos_] : kLineJhAg;  // (half-devoiced ʤ)
    // A voiced stop releasing into a voiceless fricative ("and foo", "and
    // pharaoh"): the glottis opened in kGlottalMs at the release, so the burst
    // came out through an open glottis 20 dB louder than before a vowel
    // (Jyrki 2026-11-07: "a strange sound between the words"). The abduction
    // takes this long instead (ms; < 0: as before).
    {
      static const float kVstopFricAgMs = param("LINE_VSTOP_FRIC_AGMS", -1.0f);
      if (tract_ == Tract::Line && pi.voiced && kVstopFricAgMs >= 0 && next && info(*next).cls == Cls::Fricative && *next != Ph::VV && *next != Ph::Z && *next != Ph::ZH && *next != Ph::DH) rel.agMs = kVstopFricAgMs;
    }
  }
  if (tract_ == Tract::Line && kLineReleaseKick > 0 && pi.voiced && !nextIsPause && !unreleased && next && isVowel(*next)) {
    // A voiced stop into a vowel: the folds part briefly at the release and
    // swing back into voice. From the weak vibration of the closure the
    // string folds restarted at a fraction of their amplitude ("different"
    // lost its ɪ, "is" vanished, Jyrki 2026-11-02 14:30).
    rel.ag = kLineReleaseKick;
    rel.agMs = 6;
    Event back{release + kLineReleaseKickMs * ms};
    back.ag = nextAero.ag;
    back.agMs = 10;
    pendingKick_ = back;
    hasPendingKick_ = true;
  }
  // b d g released into l ɹ w j (BDG agent 2026-11-11; Kokoro's l ɹ after a b / g carry 2-7 kHz at -23..-34 dB, the
  // release noise running into a partly devoiced liquid; puhuja's at -49):
  // LINE_VSTOP_LIQ_AG (< 0: off): the glottis opens to this (cm^2) at the release and closes into voice after
  // LINE_VSTOP_LIQ_AG_MS, its breath noise x LINE_VSTOP_LIQ_ASP (< 0: unchanged) meanwhile;
  // LINE_VSTOP_LIQ_REL_SCALE: the opening time x this (1: unchanged).
  {
    static const float kVstopLiqAg = param("LINE_VSTOP_LIQ_AG", -1.0f), kVstopLiqAgMs = param("LINE_VSTOP_LIQ_AG_MS", 20.0f),
                       kVstopLiqAsp = param("LINE_VSTOP_LIQ_ASP", -1.0f), kVstopLiqRelScale = param("LINE_VSTOP_LIQ_REL_SCALE", 1.0f);
    const bool liq = tract_ == Tract::Line && pi.voiced && !unreleased && next &&
                     (*next == Ph::L || *next == Ph::RR || *next == Ph::W || *next == Ph::J) && tok.ph != Ph::JH;
    if (liq && kVstopLiqRelScale != 1.0f) rel.acMs *= kVstopLiqRelScale;
    // LINE_G_REL_AG (lab knob 2026-11-11, G agent 5; < 0: off): a g not after a vowel releases with the glottis parted
    // to this (cm^2) for LINE_G_REL_AG_MS, its breath noise x LINE_G_REL_ASP (< 0: unchanged): a short-lag VOT.
    // LINE_G_REL_AG_WHERE: 0 before vowels and l ɹ w j, 1 only before l ɹ w j, 2 only before vowels.
    // (Kokoro's "Glue" "Grow": a burst at -10..-19 dB and 25-35 ms of noise before the voice; puhuja's g: no burst,
    // the voice at once into a dark liquid; "glue" -> blue, "grow" -> bro.)
    static const float kGRelAg = param("LINE_G_REL_AG", -1.0f), kGRelAgMs = param("LINE_G_REL_AG_MS", 20.0f),
                       kGRelAsp = param("LINE_G_REL_ASP", -1.0f);
    static const int kGRelAgWhere = param("LINE_G_REL_AG_WHERE", 0);
    const bool gRelAg = tract_ == Tract::Line && kGRelAg >= 0 && tok.ph == Ph::G && pi.voiced && !unreleased && next &&
                        !prevWasVowel_ && !hasPendingKick_ && !hasPendingAspEnd_ &&
                        ((liq && kGRelAgWhere != 2) || (isVowel(*next) && kGRelAgWhere != 1));
    if (gRelAg) {
      rel.ag = kGRelAg;
      rel.agMs = 5;
      Event back{release + kGRelAgMs * ms};
      back.ag = nextAero.ag;
      back.agMs = 10;
      pendingKick_ = back;
      hasPendingKick_ = true;
      if (kGRelAsp >= 0) {
        rel.lineAspMul = kGRelAsp;
        pendingAspEnd_ = Event{back.t + 10.0 * ms};
        pendingAspEnd_.lineAspMul = 1.0f;
        hasPendingAspEnd_ = true;
      }
    }
    if (liq && kVstopLiqAg >= 0 && !hasPendingKick_ && !hasPendingAspEnd_) {
      rel.ag = kVstopLiqAg;
      rel.agMs = 5;
      Event back{release + kVstopLiqAgMs * ms};
      back.ag = nextAero.ag;
      back.agMs = 10;
      pendingKick_ = back;
      hasPendingKick_ = true;
      if (kVstopLiqAsp >= 0) {
        rel.lineAspMul = kVstopLiqAsp;
        pendingAspEnd_ = Event{back.t + 10.0 * ms};
        pendingAspEnd_.lineAspMul = 1.0f;
        hasPendingAspEnd_ = true;
      }
    }
    // LINE_VSTOP_PRESS (1: off): the breath behind a b d g released into a vowel, liquid or nasal goes to this share
    // over the last LINE_VSTOP_PRESS_LEAD_MS of the closure (> 1: a stronger burst) and back to 1 over 15 ms.
    static const float kVstopPress0 = param("LINE_VSTOP_PRESS", 0.7f), kVstopPressLeadMs = param("LINE_VSTOP_PRESS_LEAD_MS", 30.0f);
    static const float kGPress = param("LINE_G_PRESS", -1.0f);  // (the g's own; < 0: LINE_VSTOP_PRESS; G agent 5)
    const float kVstopPress = tok.ph == Ph::G && kGPress >= 0 ? kGPress : kVstopPress0;
    if (tract_ == Tract::Line && kVstopPress != 1.0f && pi.voiced && tok.ph != Ph::JH && !unreleased && next && !nextIsPause &&
        info(*next).cls != Cls::Fricative && rel.press < 0) {
      Event ease{std::max(release - kVstopPressLeadMs * ms, t_ + 1.0)};
      ease.press = kVstopPress;
      ease.pressMs = kVstopPressLeadMs;
      addEventAt(ease);
      rel.press = 1.0f;
      rel.pressMs = 15;
    }
  }
  // A final voiced stop before a pause: the breath stops this long before the release (ms; negative: after it;
  // <= -500: off), ramping down over
  // LINE_FINAL_VSTOP_CUT_RAMP_MS, so no voicing follows the release (Jyrki 2026-11-09: "God" "sounds a bit like
  // 'God-a', perhaps the pressure could stop 15 ms earlier"); the stored mouth pressure still gives a soft release.
  if (tract_ == Tract::Line && nextIsPause && pi.voiced) {
    static const float kLineFinalVstopCutMs0 = param("LINE_FINAL_VSTOP_CUT_MS", -1000.0f);
    static const float kLineGFinalCutMs = param("LINE_G_FINAL_CUT_MS", -1000.0f);  // (the g's own, G agent 5; <= -500: the shared)
    const float kLineFinalVstopCutMs = tok.ph == Ph::G && kLineGFinalCutMs > -500.0f ? kLineGFinalCutMs : kLineFinalVstopCutMs0;
    static const float kLineFinalVstopCutRampMs = param("LINE_FINAL_VSTOP_CUT_RAMP_MS", 10.0f);
    if (kLineFinalVstopCutMs > -500.0f) {
      Event cut{std::max(release - kLineFinalVstopCutMs * ms, t_ + 1.0)};
      cut.press = 0.0f;
      cut.pressMs = kLineFinalVstopCutRampMs;
      addEventAt(cut);
    }
  }
  // A stop released into a fricative ("hardship", "outside"): the breath eases to LINE_STOP_FRIC_PRESS_FLOOR over the last
  // LINE_STOP_FRIC_PRESS_LEAD_MS of the closure (0: off) and comes back over LINE_STOP_FRIC_PRESS_BACK_MS from the
  // release, so the slit does not open onto the full stored pressure. Jyrki 2026-11-11: "\"hardship\" has a snap
  // between d and s"; PUHUJA_ENERGY_TRACE: the d still nearly shut (0.006 cm^2) 15 ms into the ʃ with 3170 dyn/cm^2
  // behind it, then a 33 dB jump of the 2-8 kHz band in ~1 ms as it opened.
  {
    static const float kStopFricLeadMs = param("LINE_STOP_FRIC_PRESS_LEAD_MS", 25.0f), kStopFricFloor = param("LINE_STOP_FRIC_PRESS_FLOOR", 0.6f),
                       kStopFricBackMs = param("LINE_STOP_FRIC_PRESS_BACK_MS", 25.0f);
    // ... and an affricate's own release into its ʃ / ʒ (LINE_AFFR_PRESS_LEAD_MS, 0: off; the same floor and return).
    // Jyrki 2026-11-11: "\"cultural spiritual adventurous\" all have a snap at t" (their ʧ: 11-18).
    static const float kAffrLeadMs = param("LINE_AFFR_PRESS_LEAD_MS", 0.0f);
    if (tract_ == Tract::Line && kAffrLeadMs > 0 && !nextIsPause && (tok.ph == Ph::CH || tok.ph == Ph::JH)) {
      Event ease{std::max(release - kAffrLeadMs * ms, t_ + 1.0)};
      ease.press = kStopFricFloor;
      ease.pressMs = kAffrLeadMs;
      addEventAt(ease);
      rel.press = 1.0f;
      rel.pressMs = kStopFricBackMs;
    }
    if (tract_ == Tract::Line && kStopFricLeadMs > 0 && !nextIsPause && nextFric && tok.ph != Ph::JH && tok.ph != Ph::CH) {
      Event ease{std::max(release - kStopFricLeadMs * ms, t_ + 1.0)};
      ease.press = kStopFricFloor;
      ease.pressMs = kStopFricLeadMs;
      addEventAt(ease);
      rel.press = 1.0f;
      rel.pressMs = kStopFricBackMs;
    }
  }
  // The same breath ease before a final b d g (LINE_FINAL_VPRESS_LEAD_MS, 0: off; floor LINE_FINAL_VPRESS_FLOOR): the
  // ease above is gated to p t k, so a final d stored the full pressure and popped it. Perception review 2026-11-11:
  // Head, Thread, Red, Made, Said peaked +6..+13 dB (1-8 kHz) over the word's loudest 50 ms, Kokoro -1..-23.
  // 46 ms to 0.5 brings that to -11..+2 dB (pclick 40-49 -> 15-31); the pop is a threshold: 0.3 is no quieter and cost
  // 16 more b d g errors in Whisper (and the level guard), 0.7 pops again. BUG-POP 2026-11-11.
  {
    static const float kLineFinalVPressLeadMs = param("LINE_FINAL_VPRESS_LEAD_MS", 46.0f), kLineFinalVPressFloor = param("LINE_FINAL_VPRESS_FLOOR", 0.5f);
    if (tract_ == Tract::Line && nextIsPause && pi.voiced && !unreleased && kLineFinalVPressLeadMs > 0) {
      Event ease{std::max(release - kLineFinalVPressLeadMs * ms, t_ + 1.0)};
      ease.press = kLineFinalVPressFloor;
      ease.pressMs = kLineFinalVPressLeadMs;
      addEventAt(ease);
    }
  }
  if (tract_ == Tract::Line && nextIsPause && !pi.voiced) {
    // ... and its release opens this many times slower (LINE_FINAL_REL_SCALE; 1: as any other): the final t of "out"
    // burst at the vowel's level, a 4 ms click 20 dB over Kokoro's release (Jyrki 2026-11-10: "too much of a snap").
    static const float kLineFinalRelScale = param("LINE_FINAL_REL_SCALE", 1.0f);
    if (kLineFinalRelScale != 1.0f) rel.acMs *= kLineFinalRelScale;
    // ... and the breath behind the closure eases to LINE_FINAL_PRESS_FLOOR over the last LINE_FINAL_PRESS_LEAD_MS of
    // the closure (0: off), so less pressure is stored for the pop: the click is the pressure release itself, not the
    // glottis or the opening speed (2026-11-10).
    {
      static const float kLineFinalPressLeadMs = param("LINE_FINAL_PRESS_LEAD_MS", 46.0f), kLineFinalPressFloor = param("LINE_FINAL_PRESS_FLOOR", 0.3f);
      // (T agent 10 round K, lab; < 0: LINE_FINAL_PRESS_LEAD_MS) a final t's own lead: an earlier ease stops the voice
      // sooner in its closure (the voice bar of a final t is a d cue: heard as d 238 times word-finally).
      static const float kLineFinalTPressLeadMs = param("LINE_FINAL_T_PRESS_LEAD_MS", -1.0f);
      const float leadMs = tok.ph == Ph::T && kLineFinalTPressLeadMs >= 0 ? kLineFinalTPressLeadMs : kLineFinalPressLeadMs;
      if (leadMs > 0) {
        Event ease{std::max(release - leadMs * ms, t_ + 1.0)};
        // (K agent 9 round J, lab; < 0: LINE_FINAL_PRESS_FLOOR) a final k's own floor: at 0.3 its release came out as a
        // 1-kHz thump with its 1-8 kHz part 28-42 dB under the vowel ("Park", "Like"; Kokoro's final k -19 dB at 2-4 kHz).
        // Lab (300 k-context words, two draws, base 168.5 k errors of 634): 0.45 150, 0.7 140.5; with the final slit
        // (LINE_K_FINAL_SLIT_MS 30, _A 0.1) 0.6 130.5, 0.45 132, words1000 +13 / +15; clip snap of 20 final-k words
        // 8.4 -> 12.3 dB male (Kokoro's 16.8): the release comes back toward Kokoro's, not over it.
        static const float kLineFinalKPressFloor = param("LINE_FINAL_K_PRESS_FLOOR", 0.45f);
        // (T agent 10 round K, lab; < 0: LINE_FINAL_PRESS_FLOOR) a final t's own floor (the k's fix: the t's release
        // peaked at 2.3-3.7 kHz, a k's band, against Kokoro's 5 kHz; heard p / k / d 14 % word-finally).
        static const float kLineFinalTPressFloor = param("LINE_FINAL_T_PRESS_FLOOR", -1.0f);
        ease.press = tok.ph == Ph::K && kLineFinalKPressFloor >= 0 ? kLineFinalKPressFloor
                     : tok.ph == Ph::T && kLineFinalTPressFloor >= 0 ? kLineFinalTPressFloor : kLineFinalPressFloor;
        ease.pressMs = leadMs;
        addEventAt(ease);
      }
    }
    // A final voiceless stop before a pause: the glottis stays open through
    // the release and its aspiration, then closes (the release event closed it
    // with the pause's ag = 0 at once: a 10 ms click for "make", "take").
    rel.ag = kLineFinalAg;
    static const float kLineFinalAgMs = param("LINE_FINAL_AG_MS", 5.0f);  // how fast the glottis opens at a final stop's release (ms)
    rel.agMs = kLineFinalAgMs;
    // ... and hisses like an aspirated release (the readers' final k: 60-130
    // ms of noise), the hiss ending with the breath.
    const float finalAspMul = tok.ph == Ph::P && kLineFinalPAsp >= 0 ? kLineFinalPAsp : tok.ph == Ph::K && kLineFinalKAsp >= 0 ? kLineFinalKAsp : tok.ph == Ph::T && kLineFinalTAsp >= 0 ? kLineFinalTAsp : kLineFinalAspMul;
    const float finalAspMs = tok.ph == Ph::P && kLineFinalPAspMs >= 0 ? kLineFinalPAspMs : tok.ph == Ph::K && kLineFinalKAspMs >= 0 ? kLineFinalKAspMs : tok.ph == Ph::T && kLineFinalTAspMs >= 0 ? kLineFinalTAspMs : kLineFinalAspMs;
    if (finalAspMul != 1.0f) {
      rel.lineAspMul = finalAspMul;
      pendingAspEnd_ = Event{release + (finalAspMs + std::max(kLineFinalPressMs, 0.0f)) * ms};
      pendingAspEnd_.lineAspMul = 1.0f;
      hasPendingAspEnd_ = true;
    }
  }
  // b d g's burst on the line (LINE_VSTOP_BURST, 2026-11-04): the slit's
  // frication gain while the stop opens, back to 1 once it is open (not into
  // a fricative, which keeps its own).
  const bool vstopBurst = tract_ == Tract::Line && kLineVStopBurst != 1.0f && !unreleased &&
                          (tok.ph == Ph::B || tok.ph == Ph::D || tok.ph == Ph::G) &&
                          !(next && info(*next).cls == Cls::Fricative);
  // LINE_G_BURST (lab knob 2026-11-09, G agent; < 0: LINE_VSTOP_BURST): the g's own release frication gain.
  static const float kLineGBurst = param("LINE_G_BURST", 0.3f);
  static const float kLineDBurst = param("LINE_D_BURST", 0.5f);  // (the same for the d; D agent 2026-11-10)
  static const float kLineBBurst = param("LINE_B_BURST", -1.0f);  // (the same for the b; BDG agent 2026-11-11)
  // (G agent 7 round H, lab; 0: off) the g's release jet as if LINE_G_REL_RE x faster (its Reynolds number) for
  // LINE_G_REL_RE_MS: at the release 66 cm^3/s through 0.09 cm^2 gives Re ~2400, barely over critical, and the velar
  // release made no noise (Kokoro's g: a 20-30 ms burst peaking at 1-2 kHz, -10..-25 dB re the vowel). Needs a
  // frication gain (LINE_G_BURST > 0 before vowels, where LINE_VSTOP_BURST is 0). Lab, 200 dictionary g words, two
  // draws: x2 with LINE_G_BURST 0.3, LINE_G_JET_D 0.3 (the velar band), LINE_G_REL_RE_FINAL 1.5: g errors 93 -> 77 and
  // 102 -> 74 (another seed), the B of glue / grow / grand 22 -> 14; without the jet the same burst gained 9 of 28.
  static const float kGRelRe = param("LINE_G_REL_RE", 2.0f), kGRelReMs = param("LINE_G_REL_RE_MS", 20.0f);
  const bool gRelRe = tract_ == Tract::Line && kGRelRe > 0 && tok.ph == Ph::G && !unreleased && !(next && info(*next).cls == Cls::Fricative);
  // ... a final g's (before a pause) x LINE_G_REL_RE_FINAL instead (< 0: LINE_G_REL_RE): its closure stores more
  // pressure (LINE_FINAL_VSTOP_PRESS), and with LINE_G_REL_RE 3 its burst came out at the vowel's level ("Dog.": -2 dB).
  static const float kGRelReFinal = param("LINE_G_REL_RE_FINAL", 1.5f);
  if (gRelRe) rel.lineFricReMul = (!next || nextIsPause) && kGRelReFinal >= 0 ? kGRelReFinal : kGRelRe;  // (back to 1 after addEvent(rel), below)
  // (K agent 9 round J, lab; 0: off) the same for the k: its release jet as if LINE_K_REL_RE x faster for
  // LINE_G_REL_RE_MS (a final k's: LINE_K_REL_RE_FINAL, < 0: LINE_K_REL_RE).
  static const float kKRelRe = param("LINE_K_REL_RE", 0.0f), kKRelReFinal = param("LINE_K_REL_RE_FINAL", -1.0f);
  const float kRelReV = (!next || nextIsPause) && kKRelReFinal >= 0 ? kKRelReFinal : kKRelRe;
  const bool kRelRe = tract_ == Tract::Line && kRelReV > 0 && tok.ph == Ph::K && !unreleased && !(next && info(*next).cls == Cls::Fricative);
  if (kRelRe) rel.lineFricReMul = kRelReV;
  if (vstopBurst) rel.lineFricMul = tok.ph == Ph::G && kLineGBurst >= 0 ? kLineGBurst : tok.ph == Ph::D && kLineDBurst >= 0 ? kLineDBurst : tok.ph == Ph::B && kLineBBurst >= 0 ? kLineBBurst : kLineVStopBurst;
  // ... before l ɹ w j (LINE_VSTOP_BURST_LIQ) or a nasal (LINE_VSTOP_BURST_NAS); < 0: as above (BDG agent 2026-11-11:
  // Kokoro's g / b release into l ɹ lifts 2-7 kHz by 27-56 dB, puhuja's none: "glad", "bring").
  static const float kVstopBurstLiq = param("LINE_VSTOP_BURST_LIQ", 1.0f), kVstopBurstNas = param("LINE_VSTOP_BURST_NAS", -1.0f);
  const bool vNextLiq = next && (*next == Ph::L || *next == Ph::RR || *next == Ph::W || *next == Ph::J);
  const bool vNextNas = next && (*next == Ph::M || *next == Ph::N || *next == Ph::NG);
  if (vstopBurst && vNextLiq && kVstopBurstLiq >= 0) rel.lineFricMul = kVstopBurstLiq;
  if (vstopBurst && vNextNas && kVstopBurstNas >= 0) rel.lineFricMul = kVstopBurstNas;
  if (tract_ == Tract::Line && tok.ph == Ph::T && kLineTCloseFric != 1.0f && rel.lineFricMul < 0) rel.lineFricMul = next && *next == Ph::S ? kLineSFric : 1.0f;
  // The affricate's closure ran its slit noise at LINE_JH_CLOSE_FRIC (0.3) and
  // nothing reset it: the ʒ of ʤ hissed 15 dB under the word's s with 220
  // cm^3/s flowing through it ("the generosity" heard as "denerosity", Jyrki
  // 2026-11-07). The release restores the gain (LINE_JH_REL_FRIC).
  if (tract_ == Tract::Line && (tok.ph == Ph::JH || tok.ph == Ph::CH) && rel.lineFricMul < 0) {
    static const float kLineJhRelFric = param("LINE_JH_REL_FRIC", 5.0f);  // 2026-11-07, Jyrki: "pledge-l20vr5.wav is the best" (voiced ʤ, the hiss x5: -8..-15 dB re the s through the frication)
    rel.lineFricMul = tok.ph == Ph::JH && kLineJhOwnRelFric >= 0 ? kLineJhOwnRelFric : kLineJhRelFric;
    if (tok.ph == Ph::JH && jhPos_ >= 0) rel.lineFricMul *= kLineJhPosFric[jhPos_];  // (LINE_JH_INIT_FRIC ...)
    if (tok.ph == Ph::JH && jhPos_ >= 0) jhRelFric_ = rel.lineFricMul;  // (LINE_JH_RAMP_MS, LINE_JH_DECAY)
    if (tok.ph == Ph::JH && jhPos_ >= 0 && kLineJhRampMs > 0) rel.lineFricMul *= kLineJhRamp0;
  }
  if (tract_ == Tract::Line && kLineCloseFric != 1.0f && rel.lineFricMul < 0 && tok.ph != Ph::T && tok.ph != Ph::JH && tok.ph != Ph::CH) rel.lineFricMul = 1.0f;  // (the release's burst keeps the full gain)
  // k's (and p's) release hiss x LINE_K_REL_FRIC / LINE_P_REL_FRIC while the
  // closure opens, back to 1 once it is open (2026-11-05, "It cannot.": the
  // k release peaked 3 dB over the following vowel across kennedy, Kokoro's
  // 7 dB under; the aspiration alone (LINE_K_STOP_ASP) took only 4 dB off).
  // (K agent 9 round J, lab; < 0: LINE_K_REL_FRIC) a final k's (before a pause) own release hiss.
  static const float kLineKFinalRelFric = param("LINE_K_FINAL_REL_FRIC", -1.0f);
  // (T agent 10 round K, lab; < 0: LINE_T_REL_FRIC) a final t's own release hiss.
  static const float kLineTFinalRelFric = param("LINE_T_FINAL_REL_FRIC", -1.0f);
  // (K-INITIAL agent 13 round L, lab; < 0: LINE_K_REL_FRIC) a word-initial k's own release hiss (K agent 2 round C: 0.6 for
  // every k took 16 of 138 k errors, rejected then for its snaps).
  static const float kKInitRelFric = param("LINE_K_INIT_REL_FRIC", -1.0f);
  const float kRelFricNf = tok.wordStart && kKInitRelFric >= 0 ? kKInitRelFric : kLineKRelFric;
  const float relFric = tok.ph == Ph::K ? ((!next || nextIsPause) && kLineKFinalRelFric >= 0 ? kLineKFinalRelFric : kRelFricNf)
                        : tok.ph == Ph::P ? kLinePRelFric : tok.ph == Ph::T ? ((!next || nextIsPause) && kLineTFinalRelFric >= 0 ? kLineTFinalRelFric : kLineTRelFric) : 1.0f;
  const bool relFricOn = tract_ == Tract::Line && relFric != 1.0f && !unreleased && !(next && info(*next).cls == Cls::Fricative);
  const bool nasalBurst = nasalVel && kNasalStopBurst != 1.0f && !relFricOn && !(next && info(*next).cls == Cls::Fricative);
  if (relFricOn) rel.lineFricMul = relFric;
  if (nasalBurst) rel.lineFricMul = kNasalStopBurst;  // (LINE_NASAL_STOP_BURST, restored below)
  // (D agent9 round J lab knobs; < 0: as above) a word-final d's release frication gain after a nasal or l
  // (LINE_ND_BURST: "band" -> bang, "cleaned" -> clean) or after a vowel (LINE_VD_BURST: "bad" -> bag).
  static const float kNdBurst = param("LINE_ND_BURST", -1.0f), kVdBurst = param("LINE_VD_BURST", -1.0f);
  const bool dCtxBurst = tract_ == Tract::Line && tok.ph == Ph::D && !unreleased && !(next && info(*next).cls == Cls::Fricative) &&
                         ((tok.shape.dctx == 3 && kNdBurst >= 0) || (tok.shape.dctx == 4 && kVdBurst >= 0));
  if (dCtxBurst) rel.lineFricMul = tok.shape.dctx == 3 ? kNdBurst : kVdBurst;
  {  // (W agent round G, lab; < 0: as before) the release's slit noise into a w x LINE_W_BURST (back to 1 with the gain below)
    static const float kWBurst2 = param("LINE_W_BURST", -1.0f);
    if (tract_ == Tract::Line && kWBurst2 >= 0 && next && *next == Ph::W && relFricOn) rel.lineFricMul *= kWBurst2;
  }
  // ʧ's ʃ phase: its emit() adds no airway event after the plosive (the release carries the targets), so the gain
  // rides on the release; the next sound's emit() sets its own (LINE_CH_FRIC).
  if (tract_ == Tract::Line && tok.ph == Ph::CH && kLineChFric != 1.0f) rel.lineFricMul = kLineChFric;
  if (jRel && kJRelFric >= 0) rel.lineFricMul = kJRelFric;  // (LINE_J_REL_FRIC)
  if (closurePressEased_) {  // (LINE_CLOSURE_PRESS: the breath comes back with the release)
    static const float kLineClosurePressBackMs = param("LINE_CLOSURE_PRESS_BACK_MS", 15.0f);
    rel.press = 1.0f;
    rel.pressMs = kLineClosurePressBackMs;
    closurePressEased_ = false;
  }
  // (P agent 10 round K, lab; 0: off) a p before l ɹ opens at the lips first: the release keeps the p's place for
  // LINE_P_LIQ_LIP_MS (its area LINE_P_LIQ_LIP_A, < 0: the liquid's), then the place moves on to the liquid's with its
  // area. It carried the liquid's place (LINE_R_REL_OWN / LINE_L_REL_OWN): the closed hump slid from the lips to x 0.68
  // in ~8 ms (LINE_X_ARTIC_MS) while the area opened, so the release was a tongue-body one, its burst at 0.3-2 kHz
  // 3-8 dB under the word's peak (Kokoro: low and diffuse). With LINE_HUMP_NORM 1 (2026-11-12) the hump reaches its
  // area between sections too: initial pɹ pl heard as k 128 -> 338 of 1680 (combo7 -> combo8: "Press" -> crash,
  // "Plots" -> clocks).
  // The same before j (LINE_P_J_LIP_MS: "Pure", "Popular"? pj 56 % wrong, mostly deleted) and before s z ʃ ʒ
  // (LINE_P_SIB_LIP_MS: the closure slid to the ridge, x 0.90, while it opened into the s: "Reps" -> rats, ps heard t 40).
  static const float kPLiqLipMs = param("LINE_P_LIQ_LIP_MS", 20.0f), kPLiqLipA = param("LINE_P_LIQ_LIP_A", -1.0f),
                     kPJLipMs = param("LINE_P_J_LIP_MS", 0.0f), kPSibLipMs = param("LINE_P_SIB_LIP_MS", 0.0f);
  const bool pSibNext = next && (*next == Ph::S || *next == Ph::Z || *next == Ph::SH || *next == Ph::ZH);
  const float pLipMs = !next ? 0.0f : (*next == Ph::L || *next == Ph::RR) ? kPLiqLipMs : *next == Ph::J ? kPJLipMs : pSibNext ? kPSibLipMs : 0.0f;
  const bool pLiqLip = tract_ == Tract::Line && tok.ph == Ph::P && pLipMs > 0 && !unreleased && rel.lineX >= 0;
  Event pLiqOpen{release + pLipMs * ms};
  if (pLiqLip) {
    pLiqOpen.lineX = rel.lineX, pLiqOpen.lineXClosure = rel.lineXClosure;
    pLiqOpen.ac = rel.ac;
    pLiqOpen.acMs = rel.acMs;
    rel.lineX = -1.0f;  // (the place stays at the lips)
    if (kPLiqLipA >= 0 && !pSibNext) rel.ac = kPLiqLipA;
  }
  // (K-INITIAL agent 13 round L, lab; 0: off) the same for a k before l ɹ: the release keeps the k's place (the velum)
  // for LINE_K_LIQ_VEL_MS (its area LINE_K_LIQ_VEL_A, < 0: the liquid's), then moves on to the liquid's place and area.
  static const float kKLiqVelMs = param("LINE_K_LIQ_VEL_MS", 0.0f), kKLiqVelA = param("LINE_K_LIQ_VEL_A", -1.0f);
  const bool kLiqVel = tract_ == Tract::Line && tok.ph == Ph::K && kKLiqVelMs > 0 && !unreleased && rel.lineX >= 0 && next &&
                       (*next == Ph::L || *next == Ph::RR);
  Event kLiqOpen{release + kKLiqVelMs * ms};
  if (kLiqVel) {
    kLiqOpen.lineX = rel.lineX, kLiqOpen.lineXClosure = rel.lineXClosure;
    kLiqOpen.ac = rel.ac;
    kLiqOpen.acMs = rel.acMs;
    rel.lineX = -1.0f;  // (the place stays at the velum)
    if (kKLiqVelA >= 0) rel.ac = kKLiqVelA;
  }
  // (P agent 10 round K, lab; < 0: off) ... or the hump's sampling as before LINE_HUMP_NORM 1 (LINE_P_LIQ_NORM: its power)
  // from the p's release for LINE_P_LIQ_NORM_MS: the sliding hump leaked between sections and its burst stayed weak
  // (initial pɹ pl heard k 128 of 1680 at norm 0, 338 at 1).
  static const float kPLiqNorm = param("LINE_P_LIQ_NORM", 0.5f), kPLiqNormMs = param("LINE_P_LIQ_NORM_MS", 40.0f);
  const bool pLiqNorm = tract_ == Tract::Line && tok.ph == Ph::P && kPLiqNorm >= 0 && !unreleased && next && (*next == Ph::L || *next == Ph::RR);
  if (pLiqNorm) rel.lineHumpNorm = kPLiqNorm;
  addEvent(rel);
  if (pLiqLip) addEventAt(pLiqOpen);
  if (kLiqVel) addEventAt(kLiqOpen);
  if (pLiqNorm) {
    Event back{release + kPLiqNormMs * ms};
    back.lineHumpNorm = -1.0f;
    addEventAt(back);
  }
  // (P agent 10 round K, lab; 0: off) a p into a vowel in two steps, as the k's slit (LINE_K_BURST_MS): the lips part
  // through a slit of LINE_P_BURST_A cm^2 for LINE_P_BURST_MS first (its jet's hiss the burst), then open with the
  // release's own time. puhuja's initial p burst is an 80-300 Hz thump with 2-8 kHz at -35..-46 dB re the word's peak
  // in its first 12 ms ("Porch", "Pay", "Pit"); Kokoro's (af_heart, af_bella, am_adam) carry 2-8 kHz at -10..-30:
  // a p before ɔ ɝ ɑ is heard as k (Porch -> courts, Purse -> curse, Pearl -> girl), elsewhere b or nothing.
  static const float kPBurstMs = param("LINE_P_BURST_MS", 0.0f), kPBurstA = param("LINE_P_BURST_A", 0.05f);
  if (tract_ == Tract::Line && tok.ph == Ph::P && kPBurstMs > 0 && !unreleased && next && isVowel(*next)) {
    Event slit{release};
    slit.ac = kPBurstA;
    slit.acMs = 3;
    addEvent(slit);
    Event open{release + kPBurstMs * ms};
    open.ac = rel.ac;
    open.acMs = rel.acMs;
    addEventAt(open);
  }
  if (gRelRe || kRelRe) {  // (LINE_G_REL_RE, LINE_K_REL_RE, above)
    Event back{release + kGRelReMs * ms};
    back.lineFricReMul = 1.0f;
    addEventAt(back);
  }
  // k and g into a vowel in two steps (LINE_K_BURST_MS, 2026-11-04): the
  // closure opens first to a slit at the velum, its hiss the burst, and only
  // after LINE_K_BURST_MS on toward the vowel (with rel's own opening time).
  // (LINE_G_BURST_MS / LINE_G_BURST_A, lab knobs 2026-11-09, G agent; < 0: the k's)
  static const float kLineGBurstMs = param("LINE_G_BURST_MS", 10.0f), kLineGBurstA = param("LINE_G_BURST_A", -1.0f);
  // ... and b d in two steps too (LINE_B_BURST_MS / _A, LINE_D_BURST_MS / _A; 0: off), each voiced stop's slit also
  // before l ɹ w j (LINE_VSTOP_SLIT_LIQ 1; 0: before vowels only), and the slit's hiss through a thick jet at the
  // place's band (LINE_G_JET_D, LINE_B_JET_D, LINE_D_JET_D; 0: off) as for k (BDG agent 2026-11-11).
  static const float kLineBBurstMs = param("LINE_B_BURST_MS", 0.0f), kLineBBurstA = param("LINE_B_BURST_A", 0.1f),
                     kLineDBurstMs = param("LINE_D_BURST_MS", 10.0f), kLineDBurstA = param("LINE_D_BURST_A", 0.1f),
                     kVstopSlitLiq = param("LINE_VSTOP_SLIT_LIQ", 1.0f), kLineGJetD = param("LINE_G_JET_D", 0.3f),
                     kLineBJetD = param("LINE_B_JET_D", 0.0f), kLineDJetD = param("LINE_D_JET_D", 0.0f);
  const float kgBurstMs = tok.ph == Ph::G && kLineGBurstMs >= 0 ? kLineGBurstMs : tok.ph == Ph::B ? kLineBBurstMs : tok.ph == Ph::D ? kLineDBurstMs : kLineKBurstMs;
  const bool vSlitLiq = pi.voiced && kVstopSlitLiq > 0 && vNextLiq;
  // (D agent9 round J lab knob; 0: off) a word-final d (after a vowel, a nasal or l) opens in two steps too, through
  // the slit for LINE_D_FINAL_SLIT_MS: without it the final release made no jet whatever its frication gain
  // (LINE_ND_BURST, LINE_VD_BURST did nothing), and the d of "band" is heard as ŋ 68 times, "bad" as g 62.
  static const float kDFinalSlitMs = param("LINE_D_FINAL_SLIT_MS", 0.0f);
  const bool dFinalSlit = tract_ == Tract::Line && kDFinalSlitMs > 0 && tok.ph == Ph::D && !unreleased &&
                          (tok.shape.dctx == 3 || tok.shape.dctx == 4) && !(next && info(*next).cls == Cls::Fricative);
  // (K agent 9 round J, lab; 0: off) the k's velar slit also before l ɹ w j (LINE_K_SLIT_LIQ_MS, at the k's own place
  // as LINE_G_LIQ_SLIT_X: the release carries the liquid's) and into a pause (LINE_K_FINAL_SLIT_MS): the dictionary
  // heard initial kw kɹ kɝ as t / p / g 20-35 % ("Queens" -> twins, "Cross" -> gross), final k 17 % (after ɹ 57 %,
  // "Park" -> part); their releases had no velar 1.3-3 kHz burst (cross: 0.5-1 kHz, -20 dB; Kokoro 1.2 kHz -15).
  static const float kKSlitLiqMs = param("LINE_K_SLIT_LIQ_MS", 0.0f), kKFinalSlitMs = param("LINE_K_FINAL_SLIT_MS", 40.0f),
                     kKFinalSlitA = param("LINE_K_FINAL_SLIT_A", 0.06f);
  const bool kSlitLiq = tract_ == Tract::Line && tok.ph == Ph::K && kKSlitLiqMs > 0 && !unreleased && vNextLiq;
  const bool kSlitFinal = tract_ == Tract::Line && tok.ph == Ph::K && kKFinalSlitMs > 0 && !unreleased && nextIsPause;
  const bool kBurst = (tract_ == Tract::Line && ((kgBurstMs > 0 && (tok.ph == Ph::K || tok.ph == Ph::G || tok.ph == Ph::B || tok.ph == Ph::D) &&
                       !unreleased && next && (isVowel(*next) || vSlitLiq)) || kSlitLiq || kSlitFinal)) || dFinalSlit;
  const float vJetD = tract_ != Tract::Line || unreleased || !next || nextIsPause || info(*next).cls == Cls::Fricative ? 0.0f
                      : tok.ph == Ph::G ? kLineGJetD : tok.ph == Ph::B ? kLineBJetD : tok.ph == Ph::D ? kLineDJetD : 0.0f;
  // t into a vowel (or l r w j: LINE_T_BURST_LIQ) in two steps too (LINE_T_BURST_MS, T agent 3; 0: off): the tongue
  // tip leaves the ridge through a slit first, an alveolar affrication (Kokoro's t burst peaks at 5-6.5 kHz).
  static const float kLineTBurstW = param("LINE_T_BURST_W", 1.0f);  // (W agent round G, lab; 0: no two-step t burst before a w)
  // (T agent 10 round K, lab; 0 / < 0: off / as the vowel's) the same slit step into a pause (LINE_T_FINAL_SLIT_MS, its
  // area LINE_T_FINAL_SLIT_A, jet LINE_T_FINAL_JET_D, place LINE_T_FINAL_SLIT_X) and before an s (LINE_T_SLIT_S_MS):
  // the final t released straight from the closure, its noise peaking at 2.3-3.7 kHz with 4.5-7 kHz 10-31 dB under it
  // (a k's or p's spectrum: "Not" -> knob, "Hot" -> hop, "Knots" -> knox); Kokoro's peaks at 5.0-5.4 kHz.
  static const float kTFinalSlitMs = param("LINE_T_FINAL_SLIT_MS", 45.0f), kTFinalSlitA = param("LINE_T_FINAL_SLIT_A", -1.0f),
                     kTFinalJetD = param("LINE_T_FINAL_JET_D", 0.05f), kTFinalSlitX = param("LINE_T_FINAL_SLIT_X", -1.0f),
                     kTSlitSMs = param("LINE_T_SLIT_S_MS", 0.0f);
  const bool tFinalSlit = tract_ == Tract::Line && kTFinalSlitMs > 0 && tok.ph == Ph::T && !unreleased && nextIsPause;
  const bool tSlitS = tract_ == Tract::Line && kTSlitSMs > 0 && tok.ph == Ph::T && !unreleased && next && *next == Ph::S;
  const bool tBurst = (tract_ == Tract::Line && kLineTBurstMs > 0 && tok.ph == Ph::T && !unreleased && next &&
                       (isVowel(*next) || (kLineTBurstLiq > 0 && (*next == Ph::L || *next == Ph::RR || (*next == Ph::W && kLineTBurstW > 0) || *next == Ph::J)))) ||
                      tFinalSlit || tSlitS;
  const float tSlitMs = tFinalSlit ? kTFinalSlitMs : tSlitS ? kTSlitSMs : kLineTBurstMs;
  const float tSlitJetD = tFinalSlit && kTFinalJetD >= 0 ? kTFinalJetD : kLineTBurstJetD;
  if (tBurst) {
    Event slit{release};
    slit.ac = tFinalSlit && kTFinalSlitA >= 0 ? kTFinalSlitA : kLineTBurstA;
    slit.acMs = 3;
    if (tSlitJetD > 0) slit.lineJetD = tSlitJetD, slit.lineJetMix = 1.0f;
    if (kLineTBurstFric >= 0) slit.lineFricMul = kLineTBurstFric;
    if (kLineTBurstX >= 0) slit.lineX = kLineTBurstX;
    if (tFinalSlit && kTFinalSlitX >= 0) slit.lineX = kTFinalSlitX;
    addEvent(slit);
  }
  // (G agent 7 round H, lab; 0: off) a g released into l ɹ makes its slit at its own place, the velum (LINE_G_LIQ_SLIT_X 1;
  // the release carries the liquid's place, so the slit sat at the ridge for an l, at 0.68 for an ɹ), for
  // LINE_G_LIQ_BURST_MS (< 0: LINE_G_BURST_MS), then on to the liquid's place.
  static const float kGLiqSlitX = param("LINE_G_LIQ_SLIT_X", 0.0f), kGLiqBurstMs = param("LINE_G_LIQ_BURST_MS", -1.0f);
  // (K agent 9 round J, lab; 0: off) a final k's release through LINE_K_FINAL_JET_D's thick jet (no slit step).
  static const float kKFinalJetD = param("LINE_K_FINAL_JET_D", 0.0f);
  const bool kFinalJet = tract_ == Tract::Line && tok.ph == Ph::K && kKFinalJetD > 0 && !unreleased && nextIsPause;
  // (K-INITIAL agent 13 round L, lab; 0: off) a k's release into l ɹ through LINE_K_LIQ_JET_D's thick jet (no slit
  // step), for LINE_K_LIQ_JET_MS (< 0: the opening + 10 ms): the kɹ release hiss peaks at 3-4.5 kHz (-3 dB of the burst,
  // through the thin default jet; LINE_K_REL_FRIC 0 removes it), Kokoro's kɹ kl at 1-2 kHz with 3-4.5 kHz 17-30 dB under.
  static const float kKLiqJetD = param("LINE_K_LIQ_JET_D", 0.0f), kKLiqJetMs = param("LINE_K_LIQ_JET_MS", -1.0f);
  const bool kLiqJet = tract_ == Tract::Line && tok.ph == Ph::K && kKLiqJetD > 0 && !unreleased && next && (*next == Ph::L || *next == Ph::RR);
  const bool gLiqSlit = kBurst && tok.ph == Ph::G && kGLiqSlitX > 0 && next && (*next == Ph::L || *next == Ph::RR);
  // (D agent9 round J lab knobs; < 0: as before) a d before ɹ: its slit held LINE_DR_SLIT_MS with area LINE_DR_SLIT_A
  // (cm^2), frication gain LINE_DR_FRIC and place LINE_DR_X (share of the line: behind the ridge, the [dʒɹ] of
  // English "drain"). Kokoro's dr opens with 30-40 ms of 2-8 kHz affrication at -9..-15 dB re the word peak;
  // puhuja's has none (below -27 dB) and the dictionary hears dr as br 35 times, gr 13 ("drain" -> brain).
  static const float kDrSlitMs = param("LINE_DR_SLIT_MS", -1.0f), kDrSlitA = param("LINE_DR_SLIT_A", -1.0f),
                     kDrFric = param("LINE_DR_FRIC", -1.0f), kDrX = param("LINE_DR_X", -1.0f);
  const bool drSlit = kBurst && tok.ph == Ph::D && next && *next == Ph::RR;
  const float slitMs = gLiqSlit && kGLiqBurstMs >= 0 ? kGLiqBurstMs : kSlitLiq ? kKSlitLiqMs : kSlitFinal ? kKFinalSlitMs : dFinalSlit ? kDFinalSlitMs : drSlit && kDrSlitMs >= 0 ? kDrSlitMs : kgBurstMs;
  // (K agent 9 round J, lab; < 0: LINE_K_BURST_A / LINE_K_JET_D) the k's slit and jet before i ɪ e ɛ æ j (iː 69 %
  // wrong word-initially, heard p: "Key" -> p, "Keep" -> peeps).
  static const float kKBurstAFront = param("LINE_K_BURST_A_FRONT", -1.0f), kKJetDFront = param("LINE_K_JET_D_FRONT", -1.0f);
  const bool kFrontV = tok.ph == Ph::K && next && (*next == Ph::I || *next == Ph::IH || *next == Ph::E || *next == Ph::EH || *next == Ph::AE || *next == Ph::J);
  // (K-INITIAL agent 13 round L, lab; < 0: as above) the k's slit before l ɹ w j (LINE_K_SLIT_LIQ_MS): its area
  // LINE_K_SLIT_LIQ_A, its jet LINE_K_SLIT_LIQ_JET_D; before i ɪ e ɛ æ j its place LINE_K_SLIT_X_FRONT (share of the line,
  // the closure's at LINE_K_X_FRONT) and length LINE_K_BURST_MS_FRONT. Kokoro's k before a front vowel bursts at
  // 2.7-3.1 kHz (2-4.5 kHz bands dominant), puhuja's at 1.5-1.8 kHz (1-2 kHz dominant): "Came", "Cane" -> pain.
  static const float kKSlitLiqA = param("LINE_K_SLIT_LIQ_A", -1.0f), kKSlitLiqJetD = param("LINE_K_SLIT_LIQ_JET_D", -1.0f),
                     kKSlitXFront = param("LINE_K_SLIT_X_FRONT", -1.0f), kKBurstMsFront = param("LINE_K_BURST_MS_FRONT", -1.0f);
  const float slitMsK = kBurst && kFrontV && kKBurstMsFront >= 0 && !kSlitLiq && !kSlitFinal ? kKBurstMsFront : slitMs;
  if (kBurst) {
    Event slit{release};
    if (gLiqSlit) slit.lineX = lineKPlace(Ph::G, *next), slit.lineXClosure = true;
    if (kSlitLiq) slit.lineX = lineKPlace(Ph::K, kVowel), slit.lineXClosure = true;
    slit.ac = tok.ph == Ph::G && kLineGBurstA >= 0 ? kLineGBurstA : tok.ph == Ph::B ? kLineBBurstA : tok.ph == Ph::D ? kLineDBurstA : kLineKBurstA;
    if (kSlitFinal && kKFinalSlitA >= 0) slit.ac = kKFinalSlitA;
    if (kFrontV && kKBurstAFront >= 0) slit.ac = kKBurstAFront;
    slit.acMs = 3;
    if (tok.ph == Ph::K && kLineKJetD > 0) slit.lineJetD = kFrontV && kKJetDFront >= 0 ? kKJetDFront : kLineKJetD, slit.lineJetMix = 1.0f;  // (LINE_K_JET_D)
    if (vJetD > 0) slit.lineJetD = vJetD, slit.lineJetMix = 1.0f;
    if (drSlit && kDrSlitA >= 0) slit.ac = kDrSlitA;
    if (drSlit && kDrFric >= 0) slit.lineFricMul = kDrFric;
    if (drSlit && kDrX >= 0) slit.lineX = kDrX, slit.lineXClosure = true;
    if (kSlitLiq && kKSlitLiqA >= 0) slit.ac = kKSlitLiqA;
    if (kSlitLiq && kKSlitLiqJetD >= 0) slit.lineJetD = kKSlitLiqJetD, slit.lineJetMix = 1.0f;
    if (kFrontV && kKSlitXFront >= 0) slit.lineX = kKSlitXFront, slit.lineXClosure = true;
    addEvent(slit);
  } else if (vJetD > 0 || kFinalJet || kLiqJet) {  // (no slit: the jet rides on the release, back to the slit's own below)
    Event jet{release};
    jet.lineJetD = kFinalJet ? kKFinalJetD : kLiqJet ? kKLiqJetD : vJetD;
    jet.lineJetMix = 1.0f;
    addEvent(jet);
    Event back{release + (kLiqJet && kKLiqJetMs >= 0 ? kKLiqJetMs : rel.acMs + 10.0f) * ms};
    back.lineJetD = 0.0f;
    addEventAt(back);
  }
  // (Y agent round G, lab; 0: off) into a j in two steps: a palatal slit of LINE_J_BURST_A cm^2 at the j's place first
  // (its hiss the [ç] of "cute", "pure"), on to the j's own slit after LINE_J_BURST_MS.
  static const float kJBurstMs = param("LINE_J_BURST_MS", 0.0f), kJBurstA = param("LINE_J_BURST_A", 0.1f);
  const bool jBurst = jRel && kJBurstMs > 0 && !kBurst && !unreleased;
  if (jBurst) {
    Event slit{release};
    slit.ac = kJBurstA;
    slit.acMs = 3;
    addEvent(slit);
  }
  if (hasPendingKick_) {
    addEvent(pendingKick_);
    hasPendingKick_ = false;
  }
  if (hasPendingAspEnd_) {
    // The stop's aspiration-end event (gain back to 1) lands at the same
    // instant as this sound's own airway event and overrides it. For an h that
    // meant the h's breath gain (LINE_H_ASP) never applied after a t: "at
    // home" -> "at phone" (Jyrki 2026-11-07). The h keeps its own; the next
    // sound restores the gain (lineHAsp_).
    static const float kLineHOwnAfterStop2 = param("LINE_H_OWN_AFTER_STOP", 0.0f);
    if (!(tract_ == Tract::Line && next && info(*next).cls == Cls::H && kLineHOwnAfterStop2 > 0.5f)) addEvent(pendingAspEnd_);
    hasPendingAspEnd_ = false;
  }
  if (tBurst) {
    Event open{release + tSlitMs * ms};
    open.ac = rel.ac;
    open.acMs = rel.acMs;
    if (tSlitJetD > 0) open.lineJetD = 0.0f;
    if (kLineTBurstFric >= 0) open.lineFricMul = relFricOn ? relFric : 1.0f;
    addEventAt(open);
  }
  if (kBurst) {
    Event open{release + slitMsK * ms};
    open.ac = rel.ac;
    open.acMs = rel.acMs;
    if (gLiqSlit || kSlitLiq) open.lineX = rel.lineX, open.lineXClosure = rel.lineXClosure;
    if (kFrontV && kKSlitXFront >= 0 && rel.lineX < 0) open.lineX = kPlaceTok, open.lineXClosure = true;  // (LINE_K_SLIT_X_FRONT: back to the k's place)
    if ((tok.ph == Ph::K && kLineKJetD > 0) || vJetD > 0) open.lineJetD = 0.0f;  // (the slit's own jet again)
    if (drSlit && kDrFric >= 0) open.lineFricMul = rel.lineFricMul >= 0 ? rel.lineFricMul : 1.0f;  // (LINE_DR_FRIC)
    if (drSlit && kDrX >= 0) open.lineX = rel.lineX, open.lineXClosure = rel.lineXClosure;
    addEventAt(open);
  }
  if (jBurst) {
    Event open{release + kJBurstMs * ms};
    open.ac = rel.ac;
    open.acMs = rel.acMs;
    addEventAt(open);
  }
  if (dzRel) {  // (LINE_DZ_REL_FRIC: the z's own gain once the groove has formed)
    Event back{release + (rel.acMs + 10.0f) * ms};
    back.lineFricMul = dzBackFric;
    addEventAt(back);
  }
  // LINE_VSTOP_BURST_BACK_MS (< 0: the opening + 10 ms): when a voiced stop's release gain returns to 1.
  static const float kVstopBurstBackMs = param("LINE_VSTOP_BURST_BACK_MS", -1.0f);
  if (vstopBurst || relFricOn || nasalBurst) {
    static const float kGBurstBackMs = param("LINE_G_BURST_BACK_MS", -1.0f);  // (the g's own; < 0: as above; G agent 5)
    const float vBackMs = vstopBurst && tok.ph == Ph::G && kGBurstBackMs >= 0 ? kGBurstBackMs : vstopBurst && kVstopBurstBackMs >= 0 ? kVstopBurstBackMs : -1.0f;
    Event back{release + (vBackMs >= 0 ? vBackMs : rel.acMs + 10.0f) * ms};
    back.lineFricMul = 1.0f;
    // (P agent 10 round K, lab; < 0: 1) a p before l ɹ: the frication gain from here to the puff's end x LINE_P_LIQ_FRIC
    // (the liquid's hump hisses with the puff's flow through it: a tongue-body noise, the k's cue).
    static const float kPLiqFric = param("LINE_P_LIQ_FRIC", -1.0f);
    const bool pLiqFric = tract_ == Tract::Line && kPLiqFric >= 0 && tok.ph == Ph::P && next && (*next == Ph::L || *next == Ph::RR) &&
                          pendingAspEnd_.t > back.t;
    if (pLiqFric) back.lineFricMul = kPLiqFric;
    addEventAt(back);
    if (pLiqFric) {
      Event full{pendingAspEnd_.t};
      full.lineFricMul = 1.0f;
      addEventAt(full);
    }
  }
  if (tract_ == Tract::Line && tok.ph == Ph::P && kLinePJetD > 0 && rel.lineJetD == kLinePJetD) {
    Event back{release + (rel.acMs + 10.0f) * ms};  // (p's jet back to the slit's once the lips have parted)
    back.lineJetD = back.lineFricLpr = 0.0f;
    addEventAt(back);
  }
  if (tract_ == Tract::Line && tok.ph == Ph::T && kLineTJetD > 0 && rel.lineJetD < 0) {
    // t's jet (LINE_T_JET_D) back to the slit's once the tongue has left.
    Event back{release + (rel.acMs + 10.0f) * ms};
    back.lineJetD = back.lineFricLpr = 0.0f;
    addEventAt(back);
  }
  // LINE_KG_REL_SHAPE_L (lab knob 2026-11-09, G agent; 0: off): the pinch also before l r w j ("great" -> "rate",
  // "glass": the g released into the liquid with F2 at 1000 Hz, Kokoro's 2300), LINE_KG_REL_MS / _AC its length and hump.
  static const float kLineKgRelShapeL = param("LINE_KG_REL_SHAPE_L", 1.0f), kLineKgRelMs = param("LINE_KG_REL_MS", 25.0f),
                     kLineKgRelAc = param("LINE_KG_REL_AC", 0.6f);
  const bool kgLiquid = next && (*next == Ph::L || *next == Ph::RR || *next == Ph::W || *next == Ph::J);
  static const float kLineKgRelW = param("LINE_KG_REL_W", 1.0f);  // (W agent round G, lab; 0: no velar pinch before a w, whose F2 is low)
  if (tract_ == Tract::Line && (tok.ph == Ph::K || tok.ph == Ph::G) && !unreleased && next &&
      ((kLineKgRelShape > 0 && isVowel(*next)) || (kLineKgRelShapeL > 0 && kgLiquid && (*next != Ph::W || kLineKgRelW > 0)))) {
    // The velar pinch after k g (LINE_KG_REL_SHAPE, 2026-11-04): the vowel's
    // first 25 ms are asked for its own shape with a narrow tongue hump at
    // the velum (0.6 cm^2, half-width 2 cm, at k's place), weighted by the
    // knob and only in the hump's channels (as for j), so F2 and F3 leave the
    // release together and part into the vowel.
    TractShape v;
    if (lineShapeOf(*next, &v)) {
      // The k's own (LINE_K_REL_MS / _AC / _SHAPE, lab knobs 2026-11-10, K agent 2; < 0: the k g ones): the k's puff
      // lasts ~40 ms, so a 25 ms pinch was over before the voice began, and the voiced F2 started flat at the
      // vowel's (Coat 1320 Hz, Could 1290) where Kokoro's starts at the velar locus and falls (1780 -> 1090, 2240 -> 910).
      static const float kLineKRelMs = param("LINE_K_REL_MS", -1.0f), kLineKRelAc = param("LINE_K_REL_AC", -1.0f),
                         kLineKRelShape = param("LINE_K_REL_SHAPE", -1.0f), kLineKRelX = param("LINE_K_REL_X", -1.0f);
      const bool isK = tok.ph == Ph::K;
      // (G agent 7 round H, lab; < 0: the k g ones) the g's own pinch: LINE_G_PINCH_MS / _AC / _X / _W (before a vowel)
      // and LINE_G_PINCH_LIQ_W (before l ɹ w j).
      static const float kGPinchMs = param("LINE_G_PINCH_MS", -1.0f), kGPinchAc = param("LINE_G_PINCH_AC", -1.0f),
                         kGPinchX = param("LINE_G_PINCH_X", -1.0f), kGPinchW = param("LINE_G_PINCH_W", -1.0f),
                         kGPinchLiqW = param("LINE_G_PINCH_LIQ_W", -1.0f);
      const bool isG = tok.ph == Ph::G;
      v.xc = isK && kLineKRelX >= 0 ? kLineKRelX : isG && kGPinchX >= 0 ? kGPinchX : kPlaceTok;
      v.ac = isK && kLineKRelAc >= 0 ? kLineKRelAc : isG && kGPinchAc >= 0 ? kGPinchAc : kLineKgRelAc;
      v.wc = 2.0f;
      v.sc = 1.0f;
      const float relMs = isK && kLineKRelMs >= 0 ? kLineKRelMs : isG && kGPinchMs >= 0 ? kGPinchMs : kLineKgRelMs;
      float relW = isVowel(*next) ? (isK && kLineKRelShape >= 0 ? kLineKRelShape : isG && kGPinchW >= 0 ? kGPinchW : kLineKgRelShape)
                                  : (isG && kGPinchLiqW >= 0 ? kGPinchLiqW : kLineKgRelShapeL);
      // (K agent 9 round J, lab; < 0: as above) the k's pinch weight before i ɪ e ɛ æ (LINE_K_PINCH_FRONT) and before
      // l ɹ w j (LINE_K_PINCH_LIQ), and its length there (LINE_K_PINCH_FRONT_MS).
      static const float kKPinchFront = param("LINE_K_PINCH_FRONT", -1.0f), kKPinchLiq = param("LINE_K_PINCH_LIQ", -1.0f),
                         kKPinchFrontMs = param("LINE_K_PINCH_FRONT_MS", -1.0f);
      const bool kFront = isK && (*next == Ph::I || *next == Ph::IH || *next == Ph::E || *next == Ph::EH || *next == Ph::AE);
      if (kFront && kKPinchFront >= 0) relW = kKPinchFront;
      if (isK && !isVowel(*next) && kKPinchLiq >= 0) relW = kKPinchLiq;
      const float relMsK = kFront && kKPinchFrontMs >= 0 ? kKPinchFrontMs : relMs;
      // (Y agent round G, lab; < 0: as the liquids') the pinch before a j: its weight, place and hump.
      static const float kKgRelJ = param("LINE_KG_REL_J", -1.0f), kKgRelJX = param("LINE_KG_REL_J_X", -1.0f),
                         kKgRelJAc = param("LINE_KG_REL_J_AC", -1.0f);
      if (*next == Ph::J) {
        if (kKgRelJ >= 0) relW = kKgRelJ;
        if (kKgRelJX >= 0) v.xc = kKgRelJX;
        if (kKgRelJAc >= 0) v.ac = kKgRelJAc;
      }
      addShapeDemand(release, release + relMsK * ms, v, Ph::J, relW);
    }
  }
  if (tract_ == Tract::Line && tok.ph == Ph::T && kLineTRelShape > 0 && !unreleased && next && isVowel(*next)) {
    // An alveolar pinch after t (LINE_T_REL_SHAPE, T agent 3): the vowel's first LINE_T_REL_MS asked for its own shape
    // with a narrow hump at the ridge, as the velar pinch above: F2 starts near the alveolar locus (~1800 Hz).
    TractShape v;
    if (lineShapeOf(*next, &v)) {
      v.xc = kLineTRelX >= 0 ? kLineTRelX : linePlaceOf(Ph::T);
      v.ac = kLineTRelAc;
      v.wc = 2.0f;
      v.sc = 1.0f;
      addShapeDemand(release, release + kLineTRelMs * ms, v, Ph::J, kLineTRelShape);
    }
  }
  if (tract_ == Tract::Line && nextIsPause && !pi.voiced) {
    // (queued after the release: the event queue is in insertion order, and a
    // later-timed event ahead of the release held it back 60 ms)
    // Closing the glottis while the lungs still pushed restarted the folds:
    // a 110 ms "uh" after every final p t k ("Back." -> "back-uh", heard as
    // t + a vowel, 2026-11-04). The breath can stop with the release, and the
    // glottis can stay open into the pause.
    if (kLineFinalPressMs > 0) {
      Event ease{release};
      ease.press = 0.0f;
      ease.pressMs = kLineFinalPressMs;
      addEvent(ease);
    }
    if (kLineFinalClose > 0) {
      Event close{release + (tok.ph == Ph::P && kLineFinalPAspMs >= 0 ? kLineFinalPAspMs : tok.ph == Ph::T && kLineFinalTAspMs >= 0 ? kLineFinalTAspMs : kLineFinalAspMs) * ms};
      close.ag = 0;
      close.agMs = kLineStopGlottalMs;
      addEvent(close);
    }
  }
  // t into f v θ ð (LINE_T_SOFT_BURST < 1): the burst opened at the f's place
  // (the release's), 0.2 cm^2 at the lips, and closed again to the slit: two
  // 4-24 kHz pops 20 ms apart over the vowels' level in "not forget", "at
  // first" (2026-11-05), where Kokoro's and the readers' t are silent
  // (held, glottalised) and their f 25 dB under the vowel. 0: no burst, the
  // closure opens straight into the slit; between: the burst's area x this.
  // ... and a d into s z (LINE_D_SOFT_BURST, 2026-11-05): "holds in" had a
  // dead d closure, two clicks and then a z 10 dB under the readers' (their
  // -lds: a short voiced dip straight into a 50 ms hiss at -11..-15 dB).
  static const float dSoftBurst = param("LINE_D_SOFT_BURST", 0.0f);
  const bool dSoft = tract_ == Tract::Line && tok.ph == Ph::D && dSoftBurst < 1.0f && next && (*next == Ph::S || *next == Ph::Z);
  const bool tSoft = dSoft || (tract_ == Tract::Line && tok.ph == Ph::T && kLineTSoftBurst < 1.0f && next &&
                     (*next == Ph::F || *next == Ph::VV || *next == Ph::TH || *next == Ph::DH));
  const float softBurst = dSoft ? dSoftBurst : kLineTSoftBurst;
  // A voiced stop into a voiceless fricative ("and foe", "absolute"): no burst opening, the closure opens straight
  // into the fricative's slit (LINE_VSTOP_FRIC_BURST 0; 1: the burst as for p t k). Kokoro's "and foe" has no burst at
  // all: -23 dB re the vowel through the d's release into the f; puhuja's 20 ms 0.2 cm^2 opening through the already
  // open glottis was a +4 dB puff and a second pop as the lips closed to the f (Jyrki 2026-11-09: "a strange snip").
  static const float kVstopFricBurst = param("LINE_VSTOP_FRIC_BURST", 0.0f);
  const bool vstopFricNoBurst = tract_ == Tract::Line && pi.voiced && next && info(*next).cls == Cls::Fricative &&
                                *next != Ph::VV && *next != Ph::Z && *next != Ph::ZH && *next != Ph::DH && kVstopFricBurst <= 0;
  if (tract_ == Tract::Line && next && info(*next).cls == Cls::Fricative && !nextIsPause && !unreleased &&
      !(tok.ph == Ph::CH || tok.ph == Ph::JH) && !(tSoft && softBurst <= 0) && !vstopFricNoBurst) {
    // ks, ts, ps on the line: the stop bursts open before the tongue forms
    // the fricative's groove (released straight into 0.05 cm^2 there was no
    // burst at all: "picks", "ask" 2026-11-02). The affricates keep their
    // slow release into the groove.
    Event burst{release};
    // ... into a sibilant its own area (LINE_BURST_AREA_SIB, < 0: LINE_BURST_AREA): a t/k released into s at 0.2 cm^2
    // rang the front cavity as a click (perception review 2026-11-11: Sites, Gifts, Rights, Costs +7..+14 dB over the
    // word's loudest 50 ms in 1-8 kHz, Kokoro -1..-19). 0.06: pclick 45-62 -> 19-33 (Kokoro's maximum 17-49), the
    // t k s errors unchanged in Whisper (0.05: 79 -> 80 of 1228). BUG-POP 2026-11-11.
    static const float kLineBurstAreaSib = param("LINE_BURST_AREA_SIB", 0.06f);
    const bool sib = *next == Ph::S || *next == Ph::Z || *next == Ph::SH || *next == Ph::ZH;
    burst.ac = (sib && kLineBurstAreaSib >= 0 ? kLineBurstAreaSib : kLineBurstArea) * (tSoft ? softBurst : 1.0f);
    burst.acMs = 3;
    addEvent(burst);
    Event narrow{release + kLineBurstMs * ms};
    narrow.ac = tract_ == Tract::Line ? lineSlit(*next, nextAero.ac) : nextAero.ac;  // (the table's 0.1 cm^2 left the f of "not forget" leaking 380 cm^3/s)
    narrow.acMs = 8;
    {  // (LINE_R_REL_OWN: this step also takes the ɹ's hump x LINE_R_REL_AC)
      static const float kRRelOwn2 = param("LINE_R_REL_OWN", 1.0f), kRRelAc2 = param("LINE_R_REL_AC", 1.5f);
      if (tract_ == Tract::Line && kRRelOwn2 > 0 && *next == Ph::RR) narrow.ac *= kRRelAc2;
    }
    addEvent(narrow);
  }
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
  // The diaphragm's push into the release of d g and the aspirated p t k
  // (LINE_PULSE), inside the closure.
  if (tract_ == Tract::Line && aspirate_ && !nextIsPause && !unreleased && !dentalD && !tok.shape.flap &&
      (aspiratedRelease || tok.ph == Ph::D || tok.ph == Ph::G))
    addPressPulse(release - std::max(kLinePulseLeadMs, 0.0f) * ms, std::max(timeline_, t_) + 1,
                  kLinePulse * kLinePulseStop);
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
    // The planned line renders a phrase once it is planned.
    if (!finished_ && motion_ == FormantMotion::Planned && tract_ == Tract::Line && t_ + 1 > shapePlanEnd_) break;
    out[i++] = tick();
    t_ += 1;
  }
  return i;
}

float Synth::noise() { return nextNoise(rng_); }

namespace {
uint64_t splitMix(uint64_t x) {  // a 64-bit mix (splitmix64's finaliser)
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
}  // namespace

void Synth::seedNoise(uint64_t seed) {
  seed_ = seed ? seed : 1;
  rng_ = splitMix(seed_) | 1;
  line_.reseed(splitMix(seed_ ^ 0x6c696e65ull));
}

void Synth::markInputLine(uint64_t key) {
  // At the plan time: everything before it (the last line's pause) is planned, the line's sounds start after it.
  // Kept apart from events_, whose time order the planner reads (addEvent).
  if (seed_) reseeds_.emplace_back(timeline_, key);
}

float Synth::tick() {
  while (!reseeds_.empty() && reseeds_.front().first <= t_) {  // PUHUJA_SEED: an input line begins
    const uint64_t s = splitMix(seed_ ^ splitMix(reseeds_.front().second));
    rng_ = s | 1;
    line_.reseed(splitMix(s ^ 0x6c696e65ull));
    if (std::getenv("PUHUJA_SEED_TRACE")) std::fprintf(stderr, "reseed t %.4f key %016llx\n", t_ / fs_, static_cast<unsigned long long>(reseeds_.front().second));
    reseeds_.pop_front();
  }
  // Start the articulator movements and noise changes that are due.
  while (!events_.empty() && events_.front().t <= t_) {
    const Event& e = events_.front();
    if (e.press >= 0) {
      press_.start(e.press, msToSamples(e.pressMs, fs_));
      pressTo_ = e.press;
    }
    if (e.aspGain >= 0) aspGainTarget_ = e.aspGain;
    if (e.lineX >= 0) {
      if (kLineXArticMs >= 0 && (e.lineX >= 0.94f) != (lineX_ >= 0.94f)) lineXArtic_ = true;  // (LINE_X_ARTIC_MS)
      lineXTarget_ = e.lineX;
      lineXSnapOk_ = !kLineXSnapClosures || e.lineXClosure;
      static const bool kXTrace = std::getenv("PUHUJA_X_TRACE") != nullptr;  // the place events as they start
      if (kXTrace) std::fprintf(stderr, "xev t %.3f x %.3f ac %.3f\n", t_ / fs_, e.lineX, e.ac);
    }
    // The groove width and rounding step at once (the place glides): an
    // affricate's 30-60 ms of hiss never reached a followed target.
    if (e.lineLevel > 0) lineLevelTarget_ = e.lineLevel;
    if (e.lineAspMul >= 0) lineAspMul_ = e.lineAspMul;
    if (e.lineFricMul >= 0) lineFricMul_ = e.lineFricMul;
    if (e.lineFricMul >= 0 && std::getenv("PUHUJA_LINE_TRACE")) std::fprintf(stderr, "fricmul t %.3f mul %.3f\n", t_ / fs_, e.lineFricMul);
    if (e.lineFricFade >= 0) { lineFricFade_ = e.lineFricFade; line_.setFricFade(lineFricFade_); }
    if (e.lineFricReMul >= 0) line_.setFricReMul(e.lineFricReMul);
    if (e.lineFricReSmooth >= 0) line_.setFricReSmooth(e.lineFricReSmooth);
    if (e.lineFricReBoth >= 0) line_.setFricReBoth(e.lineFricReBoth > 0);
    if (e.lineSoft >= 0) lineSoft_ = e.lineSoft;
    if (e.lineExtra >= 0) lineExtra_ = e.lineExtra;
    if (e.lineExpand >= 0) lineExpand_ = e.lineExpand;
    if (e.lineSink >= 0) lineSink_ = e.lineSink;
    // The constriction's width glides to its new value over LINE_HW_GLIDE_MS (0: set at once). Jyrki 2026-11-11: "\"torch
    // has\" has a strange sound between the words": the ʧ's ʃ (a 4-wide slit, LINE_POST_HW) went to width 1 in one step as
    // it opened toward the h; the slit's air mass (1240 erg of the line's kinetic energy) vanished in 1 ms and the
    // pressure behind it jumped 1450 dyn/cm^2.
    static const float kLineHwGlideMs = param("LINE_HW_GLIDE_MS", 10.0f);
    if (e.lineHw >= 0) {
      lineHwTarget_ = e.lineHw;
      if (!(kLineHwGlideMs > 0)) lineHw_ = e.lineHw;
    }
    if (e.lineJetD >= 0) line_.setJetThickness(e.lineJetD, e.lineJetMix, e.lineSlitScale, e.lineJetSmoothMs, e.lineJetFcMin, e.lineJetFcMax);
    if (e.lineFricLpr >= 0) line_.setFricLpRatio(e.lineFricLpr);
    if (e.lineAspHp >= 0) {
      static const int tAspPoles = static_cast<int>(param("LINE_T_ASP_HP_POLES", 3.0f));
      line_.setAspHp(e.lineAspHp, tAspPoles);
    }
    if (e.linePuffLpr >= 0) line_.setPuffLpRatio(e.linePuffLpr);
    if (e.lineHumpNorm > -1.5f) line_.setHumpNorm(e.lineHumpNorm);
    if (e.lineGlotFc >= 0) line_.setGlotFcScale(e.lineGlotFc);
    if (e.lineHBreath >= 0) lineHBreath_ = e.lineHBreath;
    if (e.lineLip >= 0) lineLip_ = lineLipTarget_ = e.lineLip;
    planUsed_ += e.planAir;
    if (e.creak >= 0) creakTarget_ = e.creak;
    if (e.aspBurst) aspBurst_ = aspirationGain_ * e.aspBurstGain;
    if (e.ac >= 0) ac_.start(e.ac, msToSamples(e.acMs, fs_));
    if (e.ag >= 0) ag_.start(e.ag, msToSamples(e.agMs, fs_));
    if (e.ag >= 0 && std::getenv("PUHUJA_LINE_TRACE")) std::fprintf(stderr, "agevent t %.3f ag %.3f over %.0f ms\n", t_ / fs_, e.ag, e.agMs);
    if (e.ac >= 0 && std::getenv("PUHUJA_LINE_TRACE")) std::fprintf(stderr, "acevent t %.3f ac %.3f over %.0f ms x %.3f hw %.2f\n", t_ / fs_, e.ac, e.acMs, e.lineX, e.lineHw);
    if (e.lineAspMul >= 0 && std::getenv("PUHUJA_LINE_TRACE")) std::fprintf(stderr, "aspevent t %.3f aspMul %.2f aspGain %.3f\n", t_ / fs_, e.lineAspMul, e.aspGain);
    if (e.an >= 0) an_.start(e.an, msToSamples(e.anMs, fs_));
    if (e.expand >= 0) expand_.start(e.expand, msToSamples(e.expandMs, fs_));
    if (e.expand >= 0 && std::getenv("PUHUJA_LINE_TRACE")) std::fprintf(stderr, "expandevent t %.3f expand %.1f over %.0f ms\n", t_ / fs_, e.expand, e.expandMs);
    if (e.speech >= 0) {
      speechTarget_ = e.speech;
      speechAttackNow_ = e.speechAttackMs > 0
                             ? 1.0f - std::exp(-1.0f / msToSamples(e.speechAttackMs, fs_))
                             : kSpeechAttack_;
    }
    if (e.inhaleMs > 0) {
      // The planner's table against the simulated drain since the last
      // breath: the next phrases are planned with the larynx's real thirst.
      // SYNTH_DRAIN_LEARN 0: the planner's air scale stays 1. Learned here, at render time, it reached the planner
      // only when rendering ran ahead of planning (input split into chunks), so the same text rendered whole or in
      // chunks breathed differently (review cpp.md 2026-11-11: a pause 45 ms apart, and every word after it).
      // PLAN_INVARIANT 1 turns it off: no render-time state reaches the plan. With SYNTH_DRAIN_SCALE 0.5, the value
      // it learns, kennedy.txt (whole lines) breathes as before after its first 25 s (male: one pause 30 ms shorter
      // at 113 s); in those 25 s, where it was still learning (1 -> 0.64 -> 0.53), pauses move by up to 255 ms.
      static const bool kDrainLearn = param("SYNTH_DRAIN_LEARN", 1.0f) > 0 && !(param("PLAN_INVARIANT", 1.0f) > 0);
      if (drainScale_ < 0) drainScale_ = drainScale();
      if (kDrainLearn && planUsed_ > 150.0f) {
        const float ratio = actualUsed_ / planUsed_;
        drainScale_ = std::clamp(0.5f * (drainScale_ + ratio), 0.5f, 4.0f);
        if (std::getenv("PUHUJA_DRAIN_TRACE")) std::fprintf(stderr, "drain t %.2f ratio %.3f scale %.3f\n", t_ / fs_, ratio, drainScale_);
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
      // On the line the flow itself gates the turbulence, so the level
      // steps at once (a t -> s glide of 25 ms left "votes" with 20 ms of s).
      const float fadeMs = silent || tract_ == Tract::Line ? (e.noiseFadeMs > 0 ? e.noiseFadeMs : kNoiseCrossfadeMs)
                           : nTo_.gain > 2.0f * nFrom_.gain ? kNoiseRiseGlideMs : kNoiseGlideMs;
      nGlide_ = 0;
      nGlideStep_ = 1.0f / msToSamples(fadeMs, fs_);
    }
    if (e.pop >= 0) popTarget_ = e.pop;
    // Debug: PUHUJA_EVENT_TRACE prints every event as it is applied.
    static const bool evTrace = std::getenv("PUHUJA_EVENT_TRACE") != nullptr;
    if (evTrace)
      std::fprintf(stderr, "event t %.3f  ag %.3f/%.0f  ac %.3f/%.0f  an %.2f  press %.2f/%.0f  lineX %.2f\n", (e.t - (outputStart_ < 0 ? 0 : outputStart_)) / fs_,
                   e.ag, e.agMs, e.ac, e.acMs, e.an, e.press, e.pressMs, e.lineX);
    events_.pop_front();
  }
  // Soft floor at 0 (C1): a hard max() puts a corner into the airflow.
  const float acRaw = ac_.tick();
  const float ac = acRaw >= kAcSoft ? acRaw
                   : acRaw <= -kAcSoft ? 0.0f
                                       : (acRaw + kAcSoft) * (acRaw + kAcSoft) / (4 * kAcSoft);
  const float ag = std::max(ag_.tick(), 0.0f);
  {
    static const bool agTrace = std::getenv("PUHUJA_EVENT_TRACE") != nullptr;
    static long agCount = 0;
    if (agTrace && (agCount++ % 48) == 0)
      std::fprintf(stderr, "ag t %.3f  ag %.4f raw %.4f  ac %.3f\n", (t_ - (outputStart_ < 0 ? 0 : outputStart_)) / fs_, ag, ag_.value(), ac);
  }
  const float an = std::max(an_.tick(), 0.0f);
  const float expand = std::max(expand_.tick(), 1.0f);
  if (tract_ == Tract::Line) {
    // (LINE_NOISE_AREA_TARGET) the constriction noise's reference area: the
    // glide's smaller end, except out of a closure, where the release's burst
    // is the opening slit's own noise: the destination (2026-11-05, round 8:
    // damped releases cost t and d, Whisper -44).
    const float ref = ac_.glideFrom() < 0.01f ? ac_.glideTo() : ac_.glideMin();
    line_.setNoiseAreaRef(std::max(ref, 0.0f));
  }

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
                                std::clamp((lungVolume_ - kLungRest) * kLungRangeInv, 0.0f, 1.0f)) *
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
    if (motion_ == FormantMotion::Planned && planAt(t_, pl)) {
      // The planned melody is the pitch itself: no lag stages.
      pitch1_ = pitch2_ = std::exp2(pl[3]);
    } else {
      const double tp = t_ + kPitchLookAheadMs * 0.001 * fs_;
      pitch1_ += kPitchLag_ * (splineAt(kF0Mul, tp) - pitch1_);
      pitch2_ += kPitchLag_ * (pitch1_ - pitch2_);
    }
  }
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
    // The place jumps while the constriction is open (LINE_X_SNAP_AREA, cm^2;
    // 0: always glides). A v straight into a g: the one hump slid from the
    // lips to the velum over 18 ms while closing, a bump racing along the tube,
    // the "blib" at the end of "of" in "hand of God" (Jyrki 2026-11-05). Two
    // articulators would release one place and close the other.
    if (kLineXSnapArea > 0 && ac > kLineXSnapArea && lineXSnapOk_) lineX_ = lineXTarget_, lineXArtic_ = false;
    else if (lineXArtic_) {  // (LINE_X_ARTIC_MS) the other articulator: fast, until it has arrived
      static const float kArticK = kLineXArticMs > 0 ? 1.0f - std::exp(-1000.0f / (kLineXArticMs * fs_)) : 1.0f;
      lineX_ += (lineXTarget_ - lineX_) * kArticK;
      if (std::fabs(lineXTarget_ - lineX_) < 0.002f) lineX_ = lineXTarget_, lineXArtic_ = false;
    } else lineX_ += (lineXTarget_ - lineX_) * lineXFollow_;
    lineLevel_ += (lineLevelTarget_ - lineLevel_) * lineXFollow_;
    {
      static const float kHwGlideMs = param("LINE_HW_GLIDE_MS", 10.0f);
      static const float kHwK = kHwGlideMs > 0 ? 1.0f - std::exp(-1000.0f / (kHwGlideMs * fs_)) : 0.0f;
      lineHw_ += (lineHwTarget_ - lineHw_) * (kHwGlideMs > 0 ? kHwK : lineXFollow_);
    }
    lineLip_ += (lineLipTarget_ - lineLip_) * lineXFollow_;
    line_.setConstriction(lineHw_, lineLip_ > 0.05f ? lineLip_ : -1.0f);
    line_.setClosureSoftness(lineSoft_, 0.0f);  // the extra length is in the shapes' lip tube, not a stretch of the whole tube
    if ((lineCount_++ & 3) == 0) {
      // The shape at this time: between the two vowel shapes around it.
      while (lineKeys_.size() >= 2 && lineKeys_[1].t <= t_) lineKeys_.pop_front();
      TractShape sh;
      if (motion_ == FormantMotion::Planned && shapePlanAt(t_, &sh)) goto lineShapePlanned;
      if (std::getenv("PUHUJA_LINE_TRACE") && (lineCount_ % 400) == 1)
        std::fprintf(stderr, "lineshape t %.3f keys %zu first %.3f %s\n", t_ / fs_, lineKeys_.size(), lineKeys_.empty() ? -1.0 : lineKeys_[0].t / fs_,
                     lineKeys_.empty() ? "fallback(formants)" : (lineKeys_.size() == 1 || t_ <= lineKeys_[0].t) ? "hold" : "mix");
      if (lineKeys_.empty()) {
        sh = lineShapeFor(p[kF1] * voice_.formant[0], p[kF2] * voice_.formant[1], p[kF3] * voice_.formant[2]);
      } else if (lineKeys_.size() == 1 || t_ <= lineKeys_[0].t) {
        sh = lineKeys_[0].s;
      } else {
        const double span = lineKeys_[1].t - lineKeys_[0].t;
        const float w = span > 0 ? static_cast<float>((t_ - lineKeys_[0].t) / span) : 1.0f;
        if (kLineInterp >= 0.5f) {
          // Geometric: the two area functions, section by section in log area.
          const float fem = (voice_.formant[0] + voice_.formant[1] + voice_.formant[2]) / 3.0f;
          TractShape a = lineKeys_[0].s, b = lineKeys_[1].s;
          a.length /= fem; b.length /= fem;
          // b: the lips move a little forward off the teeth (Jyrki 2026-11-03:
          // "the teeth stay in their place and the transmission line only
          // extends its length where the lips are"): the lip tube and the
          // length grow by lineExtra_, everything behind stays put.
          a.lLip += lineExtra_; a.length += lineExtra_;
          b.lLip += lineExtra_; b.length += lineExtra_;
          epilarynx(a, female_);
          epilarynx(b, female_);
          const int n = line_.oral().sections();
          std::vector<float> va = a.sections(n), vb = b.sections(n);
          for (int k = 0; k < n; ++k) va[k] = std::exp(std::log(va[k]) + (std::log(vb[k]) - std::log(va[k])) * w);
          line_.setShapeAreas(std::move(va), a.length + (b.length - a.length) * w, lineX_, ac, an);
          goto lineShapeDone;
        }
        sh = lineShapeMix(lineKeys_[0].s, lineKeys_[1].s, w);
      }
    lineShapePlanned:
      // The female voice: a shorter tract in place of the formant multipliers.
      sh.length /= (voice_.formant[0] + voice_.formant[1] + voice_.formant[2]) / 3.0f;
      sh.lLip += lineExtra_; sh.length += lineExtra_;  // (as above)
      epilarynx(sh, female_);
      line_.setShape(sh, lineX_, ac, an);
    lineShapeDone:;
    }
    // PUHUJA_ENERGY_TRACE=file: per output sample the acoustic energy stored in the oral line (reads only):
    // t, the potential energy sum A dx p^2 / (2 rho c^2) and the kinetic sum rho dx u^2 / (2 A) (erg), the section
    // holding the most energy and its share, the narrowest section, its area and flow, and the pressure behind it
    // (for "where is the energy stored for the clicks and snaps", Jyrki 2026-11-11).
    {
      static FILE* etrace = std::getenv("PUHUJA_ENERGY_TRACE") ? std::fopen(std::getenv("PUHUJA_ENERGY_TRACE"), "w") : nullptr;
      if (etrace && tract_ == Tract::Line) {
        const TransmissionLine& o = line_.oral();
        const int n = o.sections();
        const double dx = o.dx(), rho = TransmissionLine::kRho, c2 = double(TransmissionLine::kC) * TransmissionLine::kC;
        double pot = 0, kin = 0, emax = 0;
        int kmax = 0, kmin = 0;
        for (int k = 0; k < n; ++k) {
          const double a = o.areaAt(k), pk = o.pressureAt(k);
          const double ai = k > 0 ? 0.5 * (a + o.areaAt(k - 1)) : a, uk = o.flowAt(k);
          const double ep = a * dx * pk * pk / (2 * rho * c2), ek = rho * dx * uk * uk / (2 * std::max(ai, 1e-4));
          pot += ep;
          kin += ek;
          if (ep + ek > emax) { emax = ep + ek; kmax = k; }
          if (a < o.areaAt(kmin)) kmin = k;
        }
        std::fprintf(etrace, "%.5f\t%.4g\t%.4g\t%d\t%.3f\t%d\t%.4f\t%.1f\t%.1f\n", t_ / fs_, pot, kin, kmax,
                     emax / std::max(pot + kin, 1e-30), kmin, o.areaAt(kmin), o.flowAt(kmin), o.pressureAt(std::max(kmin - 1, 0)));
      }
    }
    if (artFrameMs_ > 0 && t_ >= artNext_) {  // recordArticulation: a snapshot (reads only)
      artNext_ = t_ + artFrameMs_ * 0.001 * fs_;
      const TransmissionLine& o = line_.oral();
      ArticulationFrame f{t_, f0, ag, line_.velumArea(), ac < 1.0f ? lineX_ : -1.0f, ac, o.dx() * o.sections(), {}};
      f.areas.resize(o.sections());
      for (int k = 0; k < o.sections(); ++k) f.areas[k] = o.areaAt(k);
      art_.push_back(std::move(f));
    }
    // Glottal turbulence mostly with the glottis apart (h, aspiration, a
    // breathy voice): in modal voicing the Reynolds gate alone put 2-8 kHz
    // hiss on every vowel.
    // The aspiration noise's share during modal voicing (LINE_ASP_VOICED, 2026-11-06; was a fixed 0.15). Jyrki: "perhaps the
    // phonems change too abruptly and create a lot of events": the events are the glottal pulses themselves. Within one
    // pitch period puhuja's 4-7 kHz energy swings 27 dB (1-4 kHz 22), Kokoro's 12 (15); Kokoro's high band is filled
    // between the pulses by noise (its 3-6 kHz harmonic-to-noise ratio -3.4 dB against puhuja's -0.3), a lax voice's breath.
    static const float kLineAspVoiced = param("LINE_ASP_VOICED", 0.15f);
    const float aspGate = kLineAspVoiced + (1.0f - kLineAspVoiced) * smoothstep(kAgVoice, kAgBreathy, ag);
    // The per-phoneme frication gain glides over LINE_FRIC_GLIDE_MS instead of stepping at the token (0: steps).
    // Jyrki 2026-11-10: "\"forebears fought\" has a snap in between" (the z's and the f's gains differ).
    static const float kLineFricGlideMs = param("LINE_FRIC_GLIDE_MS", 0.0f);
    static const float kFricGlideK = kLineFricGlideMs > 0 ? 1.0f - std::exp(-1000.0f / (kLineFricGlideMs * fs_)) : 1.0f;
    lineFricMulS_ += (lineFricMul_ - lineFricMulS_) * kFricGlideK;
    line_.setNoise(kLineFric * (kLineFricGlideMs > 0 ? lineFricMulS_ : lineFricMul_), kLineAsp * aspGate * lineAspMul_, std::max(ag, 0.005f));
    // LINE_BREATH (x LINE_ASP): the continuous breath of a lax voice, fading out as the glottis opens past the voicing range
    // (the thresholded aspiration takes over there).
    static const float kLineBreath = param("LINE_BREATH", 0.0f);
    line_.setBreath(lineHBreath_ > 0 ? lineHBreath_ * kLineAsp : kLineBreath > 0 ? kLineBreath * kLineAsp * lineAspMul_ * (1.0f - smoothstep(kAgVoice, kAgBreathy, ag)) : 0.0f);  // (LINE_H_BREATH: an h's own)
    // The voiced closure's expansion (expand_ runs 1 -> kVoicedExpansion over
    // the closure and back after the release): as a sink flow, since the line
    // has no compliance knob. The folds stalled 10 ms into every b, d, g
    // closure (the mouth pressure reached 0.37 Ps with nothing yielding), so
    // the closure was silence and "abolish" lost its b (2026-11-03).
    // b alone (lineExtra_ > 0 marks it: only b lengthens the lips) may draw
    // more: at 200 cm^3/s its closure was silence (-45 dB re the word's peak;
    // Kokoro's b closure is a voice bar at -10..-23 dB), while a stronger
    // expansion on d made Whisper hear g (pass 3). 2026-11-04.
    float expandFlow = lineExtra_ > 0.0f && kLineBExpandFlow >= 0.0f ? kLineBExpandFlow : kLineExpandFlow;
    const bool voicedClosure = ac < 0.05f && lineSoft_ > 1.0f;
    if (kLineVbarMs > 0) {
      // A short voice bar (LINE_VBAR_MS, 2026-11-04): the expansion (at
      // LINE_VBAR_FLOW) only for the closure's first LINE_VBAR_MS, then none,
      // and the pressure builds for the burst.
      if (!voicedClosure) lineVbarFrom_ = -1;
      else if (lineVbarFrom_ < 0) lineVbarFrom_ = t_;
      if (kLineVbarFlow >= 0) expandFlow = kLineVbarFlow;
      if (voicedClosure && t_ - lineVbarFrom_ > kLineVbarMs * 0.001 * fs_) expandFlow = 0.0f;
    }
    // LINE_B_VBAR_FLOW: b between vowels (the planner's expansion running)
    // needs more than d: its closed folds blew open (u0 220 cm^3/s DC, more
    // than the 200 the expansion draws, 2026-11-05).
    if (kLineBVbarFlow >= 0 && lineExtra_ > 0.0f && expand > 1.0f) expandFlow = kLineBVbarFlow;
    if (lineExpand_ > 0.0f) expandFlow = lineExpand_;  // ʤ's own (LINE_JH_EXPAND_FLOW)
    // A voiceless closure's first LINE_VSTOP_VBAR_MS: the expansion keeps the folds going (the voice bar Kokoro's p t k have).
    // (a real closure only: s and f sit at 0.04-0.09 cm^2 and drew the sink too at < 0.05: f +11 dB, s +7)
    if (ac < 0.005f) { if (lineCloseFrom_ < 0) lineCloseFrom_ = t_; } else lineCloseFrom_ = -1;
    // The flow fades out over the second half of that time (switching the sink
    // off in one step made a thump through the walls at -6 dB re the vowels),
    // and it follows the cavity's slow pressure as a voiced closure's does.
    const double vbarLen = kLineVstopVbarMs * 0.001 * fs_;
    const bool vbarClosure = !voicedClosure && ac < 0.005f && kLineVstopVbarMs > 0 && t_ - lineCloseFrom_ < vbarLen;
    float vbarFlow = 0.0f;
    if (vbarClosure) {
      const float frac = static_cast<float>((t_ - lineCloseFrom_) / vbarLen);
      vbarFlow = (kLineVstopVbarFlow >= 0 ? kLineVstopVbarFlow : kLineExpandFlow) * std::min(1.0f, 2.0f * (1.0f - frac));
    }
    // LINE_DH_SINK_FLOW: through a ð the walls yield too (the expansion of a voiced closure only ran below 0.05 cm^2).
    const bool sink = !voicedClosure && !vbarClosure && lineSink_ > 0.0f;
    line_.setExpansion(voicedClosure ? expandFlow : sink ? lineSink_ : vbarFlow, expand > 1.0f || vbarClosure || sink);  // a voiced closure (lineSoft_ marks it); the line holds the cavity at LINE_EXPAND_HOLD of the lung pressure
  }
  {
    LarynxControl lc;
    lc.t = t_;
    // The folds' pitch target: the melody with the slow tension that
    // abduction adds; the fast aerodynamic effects are the folds' own.
    lc.f0 = baseF0_ * pitch2_ * jitter_ * (1.0f + kF0Tension * tension_);
    // Kokoro mode: the reference's f0 track through the time warp replaces the melody (prosody.h).
    // Smoothed over PROSODY_F0_MS and held through the reference's unvoiced frames for PROSODY_F0_HOLD_MS (Kokoro's
    // creaky phrase ends jitter frame to frame; raw, they made the voice irregular).
    if (prosodyGuide_) {
      static const float f0A = 1.0f - std::exp(-1.0f / (param("PROSODY_F0_MS", 25.0f) * 0.001f * fs_));
      static const double holdS = param("PROSODY_F0_HOLD_MS", 120.0f) * 0.001 * fs_;
      float st;
      if (prosodyGuide_->f0At(t_ / fs_, &st)) {
        if (prosodyF0HeldT_ < 0) prosodyF0St_ = st;  // (a fresh start: no glide from the old value)
        prosodyF0St_ += f0A * (st - prosodyF0St_);
        prosodyF0HeldT_ = t_;
      } else if (prosodyF0HeldT_ >= 0 && t_ - prosodyF0HeldT_ > holdS) {
        prosodyF0HeldT_ = -1;
      }
      if (prosodyF0HeldT_ >= 0) lc.f0 = baseF0_ * std::exp2(prosodyF0St_ / 12.0f) * jitter_ * (1.0f + kF0Tension * tension_);
    }
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
    if (tract_ == Tract::Line) lc.ps *= std::pow(lineLevel_, kLineLevelExp);  // voice amplitude ~ Ps^1.5: a level factor L needs Ps x L^(1/1.5)
    // Kokoro mode: the reference's level track (dB re its voiced median) through the warp, as a pressure factor.
    // Per word by default (Jyrki 2026-11-06: the 5 ms track made the level "bounce" where Kokoro is "continuous and
    // nicely leveled"), eased over PROSODY_LEVEL_MS; no reference: back to 0 dB.
    if (prosodyGuide_) {
      static const bool perWord = param("PROSODY_LEVEL_WORD", 1.0f) > 0;
      static const float levelA = 1.0f - std::exp(-1.0f / (param("PROSODY_LEVEL_MS", 60.0f) * 0.001f * fs_));
      float db = 0;
      prosodyGuide_->levelAt(t_ / fs_, &db, perWord);
      prosodyLevelDb_ += levelA * (db - prosodyLevelDb_);
      if (prosodyLevelDb_ != 0) lc.ps *= std::pow(std::pow(10.0f, prosodyLevelDb_ / 20.0f), kLineLevelExp);
    }
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
    if (kLineEqShelfDb != 0) y = lineEqShelf_.process(y);
    if (kLineEqPeakDb != 0) y = lineEqPeak_.process(y);
    // (PROSODY agent11, lab knob; 0: as before) the breathing-in noise x LINE_INHALE_GAIN: the line tract dropped it
    // with the parallel branch, so every breath was silent (the 11labs readers' are ~9 dB over puhuja's room tone).
    static const float kLineInhaleGain = param("LINE_INHALE_GAIN", 0.0f);
    cPar = kLineInhaleGain != 0 ? inhale * kLineInhaleGain : 0.0f;
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

// ---------------------------------------------------------------------------
// Tract shape planner (see kShapeW). The channels, in lineShapeMix's
// conventions: places and lengths linear, areas in log; the range scales
// the tolerance and the speed limit of a linear channel.
namespace {

constexpr int kCh = 10;  // = Synth::kShapeCh (checked in planShapes)
struct ShapeChannel { bool log; float range; };
constexpr ShapeChannel kShapeChannels[kCh] = {
    {false, 0.3f},  // xc: the tongue's place (share of the length)
    {true, 1.0f},   // ac: its opening
    {false, 1.0f},  // wc: its half-width (cm)
    {true, 1.0f},   // aLip
    {true, 1.0f},   // aPharynx
    {true, 1.0f},   // aOral
    {false, 0.5f},  // lLip (cm)
    {true, 1.0f},   // aLarynx
    {false, 1.0f},  // length (cm)
    {false, 0.5f},  // sc: the tongue constriction's strength (0..1)
};

void shapeToChannels(const TractShape& s, float x[kCh]) {
  const float v[kCh] = {s.xc, s.ac, s.wc, s.aLip, s.aPharynx, s.aOral, s.lLip, s.aLarynx, s.length, s.sc};
  for (int c = 0; c < kCh; ++c) x[c] = kShapeChannels[c].log ? std::log(std::max(v[c], 0.01f)) : v[c];
}

TractShape channelsToShape(const float x[kCh]) {
  float v[kCh];
  for (int c = 0; c < kCh; ++c) v[c] = kShapeChannels[c].log ? std::exp(x[c]) : x[c];
  TractShape s;
  s.xc = v[0]; s.ac = v[1]; s.wc = v[2]; s.aLip = v[3]; s.aPharynx = v[4];
  s.aOral = v[5]; s.lLip = v[6]; s.aLarynx = v[7]; s.length = v[8]; s.sc = std::clamp(v[9], 0.0f, 1.0f);
  return s;
}

// Which channels a sound needs (the others are free): weights per channel.
void shapeNeeds(Ph ph, float w[kCh]) {
  enum { XC, AC, WC, LIPA, PHAR, ORAL, LIPL, LAR, LEN, SC };
  auto all = [&](float v) { for (int c = 0; c < kCh; ++c) w[c] = v; };
  if (isVowel(ph) || ph == Ph::Stop) { all(1.0f); return; }
  switch (ph) {
    case Ph::W: {  // the lips round; the tongue back rises a little
      // (SPLAN_W_BACK_W, 2026-11-07: at 0.5 the w's back constriction lost to the
      // neighbours and its F2 stayed at 1000-1300 Hz, an l: "by war" -> "by loor".)
      static const float kWBackW = param("SPLAN_W_BACK_W", 2.5f);
      // (W agent round I, lab; 0.2 / 1: as before) the w's other channels (jaw/oral, pharynx, larynx, length) and its lips:
      // with 0.2 the open vowel after it pulled them (female w F1 422 Hz before ɑ aɪ ɔ oʊ, Kokoro 314: "wine" -> "line").
      static const float kWRestW = param("SPLAN_W_REST_W", 0.2f), kWLipW = param("SPLAN_W_LIP_W", 1.0f);
      all(kWRestW); w[LIPA] = w[LIPL] = kWLipW; w[XC] = w[AC] = w[SC] = kWBackW; return;
    }
    case Ph::J:  // the tongue body at the palate
      all(0.2f); w[XC] = w[AC] = w[WC] = w[SC] = 1.0f; return;
    case Ph::L:  // the tongue's back (its tip is the constriction)
      all(0.2f); w[XC] = w[AC] = w[WC] = w[SC] = 1.0f; w[ORAL] = 0.5f; return;
    case Ph::RR:  // bunched, rounded, the pharynx narrowed
      all(0.3f); w[XC] = w[AC] = w[SC] = w[LIPA] = w[LIPL] = w[PHAR] = 1.0f; return;
    default:
      all(1.0f); return;
  }
}

}  // namespace

void Synth::addShapeDemand(double tA, double tB, const TractShape& s, Ph ph, float wScale, float placeFloor) {
  if (tract_ != Tract::Line || motion_ != FormantMotion::Planned) return;
  ShapeDemand d;
  d.tA = tA;
  d.tB = std::max(tA, tB);
  shapeToChannels(s, d.x);
  shapeNeeds(ph, d.w);
  // The bump's place, area and width count only as much as the bump exists
  // (its strength): a vowel without a tongue constriction (a) leaves them
  // free, and the next sound's bump grows in its own place instead of
  // travelling there across the palate. Before the strength, "now" slid a
  // fairly narrow bump from past the lips to ʊ's place, and F2 rose before
  // it fell ("navu", Jyrki 2026-11-03). A vowel's place also counts only as
  // much as its bump is narrow (in a wide one the tongue can already move).
  {
    const float str = std::clamp(d.x[9], 0.0f, 1.0f);
    d.w[1] *= std::max({kShapeOpenXcW, str, placeFloor});
    d.w[2] *= std::max({kShapeOpenXcW, str, placeFloor});
    float place = str;
    if (isVowel(ph)) {
      const float open = std::log(kShapeOpenA), narrow = std::log(kShapeNarrowA);
      place *= std::clamp((open - d.x[1]) / (open - narrow), 0.0f, 1.0f);
    }
    d.w[0] *= std::max({kShapeOpenXcW, place, placeFloor});
  }
  for (float& w : d.w) w *= wScale * kShapeW;
  shapeDemands_.push_back(d);
  // A phrase without pauses: plan all but its last second.
  if (tA - shapePlanEnd_ > 4.0 * fs_) planShapes(tA - 1.0 * fs_);
}

bool Synth::shapePlanAt(double t, TractShape* s) const {
  const size_t frames = shapePlan_.size() / kShapeCh;
  if (frames < 2 || t < 0 || t > shapePlanEnd_) return false;
  const double hop = kShapeHopMs * 0.001 * fs_;
  const double f = t / hop;
  const size_t i = std::min(static_cast<size_t>(f), frames - 2);
  const float u = static_cast<float>(std::clamp(f - static_cast<double>(i), 0.0, 1.0));
  float x[kShapeCh];
  for (int c = 0; c < kShapeCh; ++c)
    x[c] = shapePlan_[i * kShapeCh + c] + (shapePlan_[(i + 1) * kShapeCh + c] - shapePlan_[i * kShapeCh + c]) * u;
  *s = channelsToShape(x);
  return true;
}

void Synth::planShapes(double tEnd) {
  static_assert(kCh == kShapeCh);
  if (tract_ != Tract::Line || motion_ != FormantMotion::Planned) return;
  const double hop = kShapeHopMs * 0.001 * fs_;
  const size_t first = shapePlan_.size() / kShapeCh;
  const double gridStart = static_cast<double>(first) * hop;
  const int n = static_cast<int>((tEnd - gridStart) / hop) + 1;
  if (n < 1 || n > 200000) return;
  // The last two planned frames are held: the new phrase continues the old
  // one's path and speed.
  const int a = static_cast<int>(std::min<size_t>(first, 2));
  const int N = a + n;
  auto frameOf = [&](double t) { return a + static_cast<int>(std::lround((t - gridStart) / hop)); };
  struct Entry { int i; int d; };  // a demanded frame and its demand
  std::vector<Entry> entries;
  for (int q = 0; q < static_cast<int>(shapeDemands_.size()); ++q) {
    const ShapeDemand& d = shapeDemands_[q];
    const int i0 = std::max(a, frameOf(d.tA)), i1 = std::min(N - 1, frameOf(d.tB));
    for (int i = i0; i <= i1; ++i) entries.push_back({i, q});
  }
  const double vel = kShapeVel, acc = kShapeAcc, mu = kShapeMu;
  std::vector<double> x(N), d(N), e(N), f(N), b(N);
  for (int c = 0; c < kShapeCh; ++c) {
    const ShapeChannel& ch = kShapeChannels[c];
    const double tol = kShapeTol * ch.range, vmax = kShapeVmax * ch.range * kShapeHopMs * 0.001;
    // The faint pull where nothing is demanded: the phrase's mean demand.
    double mean = 0, wsum = 0;
    for (const Entry& en : entries) { mean += shapeDemands_[en.d].w[c] * shapeDemands_[en.d].x[c]; wsum += shapeDemands_[en.d].w[c]; }
    if (wsum > 0) mean /= wsum;
    else if (first > 0) mean = shapePlan_[(first - 1) * kShapeCh + c];
    else { TractShape neutral; lineShapeOf(Ph::Stop, &neutral); float v[kShapeCh]; shapeToChannels(neutral, v); mean = v[c]; }
    std::fill(x.begin(), x.end(), mean);
    for (int round = 0; round <= kShapeRounds; ++round) {
      std::fill(d.begin(), d.end(), kShapeRel);
      std::fill(e.begin(), e.end(), 0.0);
      std::fill(f.begin(), f.end(), 0.0);
      for (int i = 0; i < N; ++i) b[i] = kShapeRel * mean;
      for (int i = 0; i < a; ++i) {  // held frames
        const double v = shapePlan_[(first - a + i) * kShapeCh + c];
        d[i] += 1e6; b[i] += 1e6 * v;
      }
      // Demands, with the dead zone decided from the last round's path (the
      // first round: everything active, aiming at the centre).
      for (const Entry& en : entries) {
        const ShapeDemand& dm = shapeDemands_[en.d];
        const double w = dm.w[c];
        if (w <= 0) continue;
        double target = dm.x[c];
        if (round > 0) {
          const double off = x[en.i] - target;
          if (std::fabs(off) <= tol) continue;
          target += off > 0 ? tol : -tol;
        }
        d[en.i] += w; b[en.i] += w * target;
      }
      // Effort: first and second differences (and the speed limit, from the
      // last round's path, as a first difference aimed at vmax).
      for (int i = 1; i < N; ++i) {
        double wv = vel, tv = 0;
        if (round > 0) {
          const double dx = x[i] - x[i - 1];
          if (std::fabs(dx) > vmax) { tv = mu * (dx > 0 ? vmax : -vmax); wv += mu; }
        }
        d[i - 1] += wv; d[i] += wv; e[i - 1] -= wv;
        b[i] += tv; b[i - 1] -= tv;
      }
      for (int i = 1; i + 1 < N; ++i) {
        d[i - 1] += acc; d[i] += 4 * acc; d[i + 1] += acc;
        e[i - 1] -= 2 * acc; e[i] -= 2 * acc;
        f[i - 1] += acc;
      }
      solvePenta(d, e, f, b);
      x = b;
    }
    // Append the new frames.
    if (c == 0) shapePlan_.resize((first + static_cast<size_t>(n)) * kShapeCh);
    for (int i = a; i < N; ++i) shapePlan_[(first + static_cast<size_t>(i - a)) * kShapeCh + c] = static_cast<float>(x[i]);
  }
  shapePlanEnd_ = gridStart + static_cast<double>(n - 1) * hop;
  if (std::getenv("PUHUJA_SPLAN_TRACE")) {
    std::fprintf(stderr, "splan %.3f..%.3f s: %d frames, %zu demand frames\n", gridStart / fs_, shapePlanEnd_ / fs_, n, entries.size());
    // The planned frames, and the demand in each channel (or nan if free).
    for (int i = a; i < N; ++i) {
      std::fprintf(stderr, "sframe %.3f", (gridStart + (i - a) * hop) / fs_);
      for (int c = 0; c < kShapeCh; ++c) std::fprintf(stderr, " %.4f", shapePlan_[(first + static_cast<size_t>(i - a)) * kShapeCh + c]);
      for (int c = 0; c < kShapeCh; ++c) {
        double tw = 0, ww = 0;
        for (const Entry& en : entries)
          if (en.i == i) { tw += shapeDemands_[en.d].w[c] * shapeDemands_[en.d].x[c]; ww += shapeDemands_[en.d].w[c]; }
        std::fprintf(stderr, " %.4f", ww > 0 ? tw / ww : std::nan(""));
      }
      std::fprintf(stderr, "\n");
    }
  }
  // Demands that end well before the horizon are done.
  const double keep = shapePlanEnd_ - 0.5 * fs_;
  shapeDemands_.erase(std::remove_if(shapeDemands_.begin(), shapeDemands_.end(), [&](const ShapeDemand& dm) { return dm.tB < keep; }),
                      shapeDemands_.end());
}

}  // namespace puhuja
