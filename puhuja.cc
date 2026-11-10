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
const float kLineUAfterJ = param("LINE_U_AFTER_J", 0.4f);
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
const float kLineTSoftBurst = param("LINE_T_SOFT_BURST", 0.0f);  // t's burst area into f v θ ð x this (0: none; 1: as p k), below
const float kLineCloseFric = param("LINE_CLOSE_FRIC", 1.0f);  // 2026-11-05: every other non-fricative closure's (p b k g m n ŋ d) slit-noise gain while it closes and opens (x LINE_FRIC; 1: as before). The lips closing on a vowel's 300-440 cm^3/s made a 3000 cm/s jet, Re 7000, a hiss 9 x the f's with its band sweeping 1.2 -> 11 kHz in 35 ms, the m's release the same downward: the "slurps" (Jyrki: "weird slirps ... mouth accidents rather than phonemes"). t: LINE_T_CLOSE_FRIC, dʒ: LINE_JH_CLOSE_FRIC.
const float kLineTCloseFric = param("LINE_T_CLOSE_FRIC", 0.3f);  // t's frication gain while it closes (x LINE_FRIC), below
const float kLineStopFOwn = param("LINE_STOP_F_OWN", 1.0f);  // > 0: a p t k released into f takes f's own jet, band, width and gain (below)
const float kLineTReleaseScale = param("LINE_T_RELEASE_SCALE", 2.0f);  // t's own release time scale (< 0: LINE_RELEASE_SCALE): the tongue tip leaves the ridge slower than the shared 0.58
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
const float kLineEyGlideFrom = param("LINE_EY_GLIDE_FROM", 0.1f);
// The same for aɪ (a + ɪ) and aʊ (a + ʊ) (<0: SPLAN_GLIDE_END). From 80 %
// "Time", "My", "Now", "How" stayed on the a (~650/1290 Hz) to the very end:
// no glide at all (Kokoro "My" 580/1150 -> 390/2300, "Now" 600/1550 -> 370/870).
// From 30 %: the a is held ~60 %, then "My" ends 428/1612, "Now" 431/1094
// (from 10 %: earlier glide, 410/1640 and 420/1045, Whisper the same within
// noise, aɪ heard more as eɪ/iː); Whisper aɪ 1 -> 11 of 26, aʊ 4 -> 6 of 26 (2026-11-04).
const float kLineAyGlideFrom = param("LINE_AY_GLIDE_FROM", 0.3f);
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
const float kLineZhFric = param("LINE_ZH_FRIC", 4.0f);
const float kLineJhFric = param("LINE_JH_FRIC", -1.0f);  // the ʤ's frication gain (< 0: LINE_ZH_FRIC; the ʤ's frication is an emitted ʒ)
// ʒ's own jet thickness / band low-pass / frication fade window (< 0: the ʃ ʒ ʧ ʤ LINE_POST_JET_D, LINE_POST_LPR, LINE_VFRIC_FADE; ʒ lab 2026-11-09)
const float kLineZhJetD = param("LINE_ZH_JET_D", -1.0f), kLineZhLpr = param("LINE_ZH_LPR", 4.0f), kLineZhFade = param("LINE_ZH_FADE", -1.0f);
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
const float kLineKBurstMs = param("LINE_K_BURST_MS", 0.0f);
const float kLineKBurstA = param("LINE_K_BURST_A", 0.12f);
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
const float kLineEnGdDevoice = param("LINE_EN_GD_DEVOICE", 0.7f);
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
const float kLineXSnapArea = param("LINE_X_SNAP_AREA", 0.0f);  // the constriction place jumps to its target while the hump is wider than this (cm^2; 0: glides)
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
// A phrase-initial b d g is prevoiced: the closure is formed in the pause and
// the folds start to vibrate this long before the release (as a phrase-initial
// nasal's lead). 0: off.
const float kLineVStopLeadMs = param("LINE_VSTOP_LEAD_MS", 0.0f);
const float kLineNVelMs = param("LINE_N_VEL_MS", 20.0f);
const float kLineNVelLagMs = param("LINE_N_VEL_LAG_MS", 0.0f);
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
const float kLineWKeyTo = param("LINE_W_KEY_TO", 0.7f);
// Lab knobs (R agent 2026-11-10; defaults reproduce the previous sound): a phrase-initial ɹ shaped in the pause this long
// before its voice (0: at 30 %), and the ɹ holds its shape from / to these shares of it (-1: the 0.3 / 0.7 of all sounds).
const float kLineRPreShapeMs = param("LINE_R_PRE_SHAPE_MS", 0.0f);
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
    case Ph::W: case Ph::V: return {kLineGlideLeadMs, kLineGlideVoiceLagMs, kLineGlideOnsetMs};
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
const float kLineShVowelFric = param("LINE_SH_VOWEL_FRIC", 6.0f);
// ... for a ʃ that does not start its word (a word-final ʃ linking into the next word's vowel, "abolish all": Jyrki
// 2026-11-06 "sounds too sharp"): at 6 it ran 7 dB louder than Kokoro's (level +1.4 vs -3.8 re the vowel, > 5 kHz
// -8.6 vs -16) while the word-initial ones ("shall") sat at Kokoro's. < 0: LINE_SH_VOWEL_FRIC.
const float kLineShVowelFricLink = param("LINE_SH_VOWEL_FRIC_LINK", 2.2f);
const float kLineShVowelX = param("LINE_SH_VOWEL_X", 0.84f);
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
void epilarynx(TractShape& s) {
  static const float kArea = param("LINE_LARYNX_AREA", 0.82f), kLen = param("LINE_LARYNX_LEN", 2.5f);
  s.aLarynx *= kArea;
  s.lLarynx = kLen;
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
      static const float kWLipA = param("LINE_W_LIPA", 0.7031f), kWXc = param("LINE_W_XC", 0.25f), kWShapeAc = param("LINE_W_SHAPE_AC", 0.05f), kWLipL = param("LINE_W_LIPL", 1.0f);
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

// Where a consonant constricts the line (share of the tract's length).
float linePlaceOf(Ph ph) {
  switch (ph) {
    case Ph::P: case Ph::B: case Ph::M: case Ph::W: return 0.97f;
    case Ph::F: case Ph::VV: case Ph::V: return 0.95f;
    case Ph::TH: { static const float x = param("LINE_TH_X", 0.93f); return x; }  // the θ's place (lab 2026-11-09: the same as s's 0.90)
    case Ph::DH: { static const float x = param("LINE_DH_X", 0.90f); return x; }  // (a lab knob 2026-11-09: the ð was heard as v; a v constricts at 0.95)
    case Ph::S: { static const float x = param("LINE_S_X", 0.90f); return x; }
    case Ph::Z: return 0.90f;  // the front cavity to the teeth ~1.7 cm: the hiss near 5 kHz
    // n further back than t d (0.87): there Whisper heard it as l or bl
    // ("Now." -> "hello", "New." -> "blue"; 0.77-0.80 best, 2026-11-04).
    case Ph::N: { static const float x = param("LINE_N_X", 0.78f); return x; }
    case Ph::T: { static const float x = param("LINE_T_X", 0.87f); return x; }
    case Ph::D: { static const float x = param("LINE_D_X", 0.83f); return x; }  // (a lab knob 2026-11-10, D agent; t: LINE_T_X)
    case Ph::L: case Ph::R: return 0.87f;
    case Ph::SH: case Ph::ZH: { static const float x = param("LINE_SH_X", 0.78f); return x; }
    case Ph::CH: case Ph::JH: return 0.78f;
    case Ph::RR: { static const float x = param("LINE_R_X", 0.68f); return x; }  // (its hump: LINE_R_AC; 0.75 until 2026-11-04: F3 1700-1800 with the wider hump)
    case Ph::J: { static const float x = param("LINE_J_X", 0.70f); return x; }  // (0.66 / 0.74: F2 within 50 Hz, 2026-11-04)
    case Ph::K: { static const float x = param("LINE_K_X", 0.62f); return x; }
    case Ph::G: { static const float x = param("LINE_G_X", 0.72f); return x; }  // (a lab knob 2026-11-09, G agent)
    case Ph::NG: return 0.62f;
    default: return -1.0f;
  }
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
const Voice kMaleVoice = {param("kMaleF0", 115.5) /* cost 1814 whisper-run1-phon_err */, {param("kMaleFormant", 0.9625), param("kMaleFormant", 0.9625), param("kMaleFormant", 0.9625)}, 1.0f, 1.0f, 1.7f, 1.0f, param("kMaleGain", 4.163f) /* cost 0.0322859615827058 n13-speaker-w30 */, -6, 0.3f,
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
const Voice kFemaleVoice = {param("kFemaleF0", 165.7) /* cost 0.0298241432933307 n14-stringline-w30-c */, {1.12f, 1.15f, 1.06f}, 1.15f, 0.92f, 1.8f, 1.1f, param("kFemaleGain", 3.704f) /* cost 0.0322859615827058 n13-speaker-w30 */, 7, 1.0f,
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
    case Cls::Vowel: return {kAcOpen, kAgVoice, 0};
    case Cls::Nasal: return {0, kAgVoice, kAnNasal};  // mouth closed, nose open
    case Cls::Liquid: return {0.35f, kAgVoice, 0};
    case Cls::Trill: return {0.3f, kAgVoice, 0};
    case Cls::Glide: {
      static const float jAc = param("LINE_J_AC", 0.3f);  // j (0.2: 3 dB weaker, F2 +50 Hz, 2026-11-04)
      // w (2026-11-07, Jyrki: "tempered by war" sounds like "by loor"): with
      // 0.25 cm^2 the w's F2 stayed at 1100-1300 Hz after "by", an l; Kokoro's
      // w is at 520-570 Hz.
      static const float wAc = param("LINE_W_AC", 0.25f);
      return {ph == Ph::J ? jAc : wAc, kAgVoice, 0};
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

void Synth::addEvent(const Event& e) {
  // Movements must start in order: a later phoneme's movement starting
  // before an earlier one's would be overridden by it.
  Event ev = e;
  if (!events_.empty()) ev.t = std::max(ev.t, events_.back().t);
  events_.push_back(ev);
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
  // LINE_V_PRESS (lab 2026-11-09): the lung pressure eases to this share through a v (the voice bar and the hiss both
  // softer, an approximant-like v as Kokoro's) and comes back on the next sound; 1: unchanged.
  { static const float kVPress = param("LINE_V_PRESS", 1.0f);
    if (tract_ == Tract::Line && kVPress < 1.0f) {
      if (tok.ph == Ph::VV) { e.press = kVPress; e.pressMs = 20; }
      else if (lastPh_ == Ph::VV && e.press < 0) { e.press = 1.0f; e.pressMs = 20; }
    } }
  const bool shFullVowel = tok.ph == Ph::SH && next && isVowel(*next) && *next != Ph::Schwa;  // (LINE_SH_VOWEL_*)
  e.ac = aero.ac;
  // The line's fricative slits are wider than the cascade's table: a 0.05 cm^2
  // s passes at most 175 cm^3/s at 7 cmH2O where a spoken s runs 300-500, and
  // its hiss sat 18 dB under the vowels (Kokoro 5-7), 2026-11-02.
  if (tract_ == Tract::Line) e.ac = lineSlit(tok.ph, e.ac);
  // v voiced through a wider slit (LINE_V_AC) has a weak hiss: its own gain
  // (2026-11-04; 1 = today's).
  static const float kLineVFric = param("LINE_V_FRIC", 1.0f);
  static const float kLineZFric = param("LINE_Z_FRIC", 1.0f);
  if (tract_ == Tract::Line) e.lineFricMul = tok.ph == Ph::VV ? kLineVFric : tok.ph == Ph::F ? (curStress_ ? kLineFFric : kLineFFricUnstr) : tok.ph == Ph::Z ? kLineZFric : tok.ph == Ph::ZH ? (inAffricate_ && kLineJhFric >= 0 ? kLineJhFric : kLineZhFric) : tok.ph == Ph::S ? kLineSFric : tok.ph == Ph::SH ? kLineShFric * (shFullVowel ? (tok.wordStart || kLineShVowelFricLink < 0 ? kLineShVowelFric : kLineShVowelFricLink) : 1.0f) : 1.0f;
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
  if (tract_ == Tract::Line)
    e.lineFricFade = tok.ph == Ph::ZH && !inAffricate_ && kLineZhFade >= 0 ? kLineZhFade : tok.ph == Ph::Z || tok.ph == Ph::ZH || tok.ph == Ph::VV || tok.ph == Ph::DH ? kVFricFade : tok.ph == Ph::SH ? shFade : 1.0f;
  e.lineX = linePlaceOf(tok.ph);
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
  {
    const bool post = tok.ph == Ph::SH || tok.ph == Ph::ZH || tok.ph == Ph::CH || tok.ph == Ph::JH;
    e.lineHw = post ? kLinePostHw : tok.ph == Ph::F ? kLineFHw : 1.0f;
    e.lineLip = post ? kLinePostLip : 0.0f;
    e.lineJetD = tok.ph == Ph::F ? (curStress_ ? kLineFJetD : kLineFJetDUnstr) : tok.ph == Ph::VV ? kLineVJetD : tok.ph == Ph::ZH && !inAffricate_ && kLineZhJetD >= 0 ? kLineZhJetD : post ? kLinePostJetD : 0.0f;
    if (post) e.lineJetMix = kLinePostJetMix, e.lineSlitScale = kLinePostSlitScale, e.lineJetSmoothMs = kLinePostStSmoothMs, e.lineJetFcMin = kLinePostFcMin, e.lineJetFcMax = kLinePostFcMax;
    e.lineFricLpr = tok.ph == Ph::ZH && !inAffricate_ && kLineZhLpr >= 0 ? kLineZhLpr : tok.ph == Ph::SH || tok.ph == Ph::ZH ? kLinePostLpr : tok.ph == Ph::F ? kLineFLpr : tok.ph == Ph::VV ? kLineVLpr : 0.0f;  // (not ʧ ʤ: at 3 they lost 39 -> 34, 26 -> 23 of 54, 56)
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
    // The h takes the next vowel's shape, as the cascade's formants do (hTakesVowel_): with no keys of its own the
    // line glided from the previous sound's last key to the vowel's first in one straight move, ~200 ms for
    // "mortal hands" (the l held to 70 % of its 175 ms, F2 1040 -> 1400 only by 70 ms into the æ); 11labs and
    // Kokoro leave the l 60 ms before the h and breathe the h on the vowel's shape. Jyrki 2026-11-09:
    // "mortallllhaaands", "puhuja makes fast transition between a and n [h and a], while 11labs makes a smooth natural transition".
    {
      static const float kLineHTakesVowel = param("LINE_H_TAKES_VOWEL", 1.0f);
      if (!have && tok.ph == Ph::H && kLineHTakesVowel > 0 && next && isVowel(*next)) have = lineShapeOf(*next, &sh);
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
      // (R agent 2026-11-10) the female onset ɹ sat at F2 ~1400 (Kokoro's female 900-1050): female-only pharynx / lip area x these,
      // constriction place set to this (-1: the row's); only before a vowel and not after one (onset and cluster ɹ)
      static const float kRFAph = param("LINE_R_F_APH", 1.0f), kRFLipA = param("LINE_R_F_LIPA", 1.0f), kRFX = param("LINE_R_F_X", -1.0f);
      if (female_ && next && isVowel(*next) && !isVowel(lastPh_)) {
        if (kRFAph != 1.0f) sh.aPharynx *= kRFAph;
        if (kRFLipA != 1.0f) sh.aLip *= kRFLipA;
        if (kRFX > 0) { sh.xc = kRFX; sh.sc = lineVowelStrength(sh.xc, sh.wc, sh.length); }
      }
    }
    if (have && tok.ph == Ph::O && next && *next == Ph::UH) {
      // ... and starts open-mid and central (Kokoro 530-600/1060-1290 at 20 %;
      // puhuja's o held 450/1070 and drifted up: "know" was heard as "no-ʉ").
      TractShape mid;
      if (lineShapeOf(Ph::AH, &mid)) sh = lineShapeMix(sh, mid, 0.5f);
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
    }
    if (have && tok.ph == Ph::U && lastPh_ == Ph::J && kLineUAfterJ > 0) {
      TractShape front;
      if (lineShapeOf(Ph::I, &front)) sh = lineShapeMix(sh, front, kLineUAfterJ);
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
          const float mix = open && kRColourOpenMix >= 0 ? kRColourOpenMix : frontVowel(tok.ph) && kRColourFrontMix >= 0 ? kRColourFrontMix : kRColourMix;
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
        // The ɑ's late key mixed toward ɔ (LINE_AA_OFF_MIX; 2026-11-10 AA agent: a back offglide against the coda's fronting).
        static const float kAaOffMix = param("LINE_AA_OFF_MIX", 0.0f);
        if (tok.ph == Ph::AA && kAaOffMix > 0) {
          TractShape o;
          if (lineShapeOf(Ph::AO, &o)) late = lineShapeMix(late, o, kAaOffMix);
        }
        // A coda l before a vowel or h across the word boundary ("mortal hands") leaves its shape at this share of
        // its length instead of 0.7 (LINE_CODA_L_KEY; the readers' l moves toward the vowel over its last 60 ms).
        static const float kLineCodaLKey = param("LINE_CODA_L_KEY", 0.7f);
        const bool codaLRelease = tok.ph == Ph::L && next && *next == Ph::H;  // (a vowel next: no word-boundary knowledge here, so only before h)
        lineKeys_.push_back({t0 + (codaLRelease ? kLineCodaLKey : tok.ph == Ph::W ? kLineWKeyTo : tok.ph == Ph::AA && kLineAaKeyTo >= 0 ? kLineAaKeyTo : hold ? 0.7 : 0.5) * dur, late});
        {
          // The planner's demands: the held middle; an r-coloured end of its own.
          const bool rColoured = late.ac != sh.ac || late.xc != sh.xc || late.aLip != sh.aLip;
          const float preR = isVowel(tok.ph) && next && *next == Ph::RR ? std::max(kPreRVowelW, frontVowel(tok.ph) ? kPreRFrontW : 0.0f) : 0.0f;  // see kPreRVowelW
          TractShape own;
          const bool borrowed = !isVowel(tok.ph) && !lineShapeOf(tok.ph, &own);  // f in the next vowel's shape
          const float ws = borrowed ? kShapeBorrowedW : 1.0f;
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
            const double a = owGlide ? kLineOwGlideFrom : eyGlide ? kLineEyGlideFrom :
                             ayGlide ? kLineAyGlideFrom : awGlide ? kLineAwGlideFrom :
                             chainEnd ? kShapeGlideEnd : 0.1, b = chainStart ? kShapeGlideStart : 0.9;
            addShapeDemand(t0 + std::min(a, b) * dur, t0 + std::max(a, b) * dur, sh, tok.ph, ws * kShapeGlideW);
          } else if (!rColoured) {
            // A phrase-initial j or l: the tongue in its place before the voice
            // (2026-11-04: the j's F2 rose from the pause's ə, 1550 Hz, and
            // reached only ~1750 by its end; Kokoro holds 2100-2200 from the start).
            const float preMs = !prevWasPause_ ? (prevWasPlosive_ && isVowel(tok.ph) ? kLineAfterStopLeadMs : 0.0f) : tok.ph == Ph::J ? kLineJPreShapeMs : tok.ph == Ph::L ? kLineLPreShapeMs : tok.ph == Ph::W ? kLineWPreShapeMs : tok.ph == Ph::RR ? kLineRPreShapeMs : 0.0f;
            const bool er = tok.ph == Ph::ER;
            addShapeDemand(preMs > 0 ? t0 - preMs * ms : t0 + (er ? kLineErFrom : tok.ph == Ph::RR && kLineRKeyFrom >= 0 ? kLineRKeyFrom : 0.3) * dur, t0 + (er ? kLineErTo : tok.ph == Ph::W ? kLineWKeyTo : tok.ph == Ph::RR && kLineRKeyTo >= 0 ? kLineRKeyTo : 0.7) * dur, sh, tok.ph, er ? ws * kLineErW : ws, er ? std::max(preR, kErPlaceW) : preR);
          }
          else {
            const bool open = tok.ph == Ph::AA || tok.ph == Ph::A;
            const double at = open && kRColourOpenAt >= 0 ? kRColourOpenAt : frontVowel(tok.ph) && kRColourFrontAt >= 0 ? kRColourFrontAt : kRColourAt;
            addShapeDemand(t0 + 0.3 * dur, t0 + std::max(0.3, at - 0.15) * dur, sh, tok.ph, ws, preR);
            addShapeDemand(t0 + at * dur, t0 + std::min(1.0, at + 0.15) * dur, late, tok.ph, ws);
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
    if ((tok.ph == Ph::SH || tok.ph == Ph::ZH) && shOn > 0) e.acMs = shOn;
    if ((lastPh_ == Ph::SH || lastPh_ == Ph::ZH) && shOff > 0) e.acMs = std::max(e.acMs, shOff);
    // s f θ into a voiced sound: the groove opens over LINE_FRIC_OFF_MS (0: the
    // 20 ms boundary). Kokoro's s in "save" fades over its last 30 ms before
    // the vowel starts; puhuja's hiss rose 12 dB in its last 10 ms with the
    // voice already on, a z (2026-11-05).
    static const float fricOff = param("LINE_FRIC_OFF_MS", 0.0f);
    if ((lastPh_ == Ph::S || lastPh_ == Ph::F || lastPh_ == Ph::TH) && fricOff > 0 && in.cls != Cls::Fricative && in.cls != Cls::Plosive) e.acMs = std::max(e.acMs, fricOff);
  }
  e.ag = aero.ag;
  if (in.cls == Cls::Vowel) e.ag *= tok.shape.press;  // voice quality by prosody
  e.agMs = tract_ == Tract::Line && in.cls == Cls::Fricative && aero.ag > kAgBreathy ? kLineAbductMs : kGlottalMs;
  e.aspGain = in.cls == Cls::H ? hAspirationGain_ * kHGain : kAspirationGain * kAspirationTiltGain;
  if (lineHAsp_ && in.cls != Cls::H) {  // after a voiceless h: the noise gain back
    e.lineAspMul = 1.0f;
    lineHAsp_ = false;
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
  // English nasalizes a vowel before a nasal through most of it: the velum
  // opens part way during the vowel ("man", "and", "on"); on the line the
  // port was shut until the nasal's own event (2026-11-02; set after the aero default, which had overwritten it).
  if (tract_ == Tract::Line && isVowel(tok.ph) && next && info(*next).cls == Cls::Nasal) {
    e.an = kAnNasal * kLineAnticipNasal;
    e.anMs = static_cast<float>(dur / ms);
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
      { static const float kThNoiseOn = param("LINE_TH_NOISE_ON_MS", -1.0f); if (tok.ph == Ph::TH && kThNoiseOn >= 0) e.noiseFadeMs = kThNoiseOn; }  // θ's own fade-in (< 0: 12 ms; lab 2026-11-09)
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
      const bool voicedCtx = prevVoiced_ || (tract_ == Tract::Line && (kLineHVoicedCtx >= 2 || (kLineHVoicedCtx >= 1 && !prevWasPause_)));  // (LINE_H_VOICED_CTX)
      bool voicedH = voicedCtx && next && isSonorant(*next) && !beforeJ;
      if (voicedH) {
        e.ag = voicedHAg_;
        e.aspGain *= voicedHNoise_;
        if (tract_ == Tract::Line && prevWasPause_) {  // (LINE_H_VOICED_CTX 2: the kick starts the folds, as a phrase-initial vowel's)
          if (kLineHPauseAg >= 0) e.ag = kLineHPauseAg;
          if (kLineOnsetKickAg > 0 && kLineOnsetKickMs > 0) {
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
      }
      if (tract_ == Tract::Line && !voicedH && !beforeJ) {
        e.ag = kLineHAg;
        e.agMs = kLineHRiseMs;
        e.lineAspMul = kLineHAspMul;
        lineHAsp_ = true;
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
    if (kLineAdductLeadMs > 0 && e.ag >= 0) {
      Event adduct{std::max(e.t - kLineAdductLeadMs * ms, phonemeStart_ + 1.0)};
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
      static const float kVEarlyBreath = param("LINE_V_EARLY_BREATH", 0.0f);
      const bool earlyBreath = in.cls == Cls::Nasal || in.cls == Cls::Liquid || in.cls == Cls::Glide || in.cls == Cls::Trill ||
                               (tok.ph == Ph::VV && kVEarlyBreath > 0);
      Event prePress{std::max(e.t - kLinePrePressMs * ms, earlyBreath ? std::min(inPauseFrom, e.t) : phonemeStart_ + 1)};
      prePress.press = 1.0f;
      prePress.pressMs = kLinePrePressMs;
      prePress.ac = prePress.ag = prePress.an = -1;
      prePress.speech = -1;
      prePress.aspGain = -1;
      prePress.expand = -1;
      prePress.inhaleMs = 0;
      prePress.pop = -1;
      // (not over a stop's push, LINE_PULSE: it starts from the same rise)
      if (prePress.t >= pulseEnd_) addEventAt(prePress);
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
        Event pre{std::max(e.t - kLineOnsetMs * ms, phonemeStart_ + 1)};
        pre.ag = kAgBreathy;
        pre.agMs = 10;
        addEvent(pre);
        // ... but within the vowel: an unaccented "It" (67 ms) before the t's
        // open glottis never reached voicing with the 80 ms adduction, so
        // "It cannot." lost its "It" (Jyrki 2026-11-05). At most
        // LINE_ONSET_SHARE of the sound's length.
        e.agMs = std::min(kLineOnsetMs, kLineOnsetShare * static_cast<float>(dur / fs_ * 1000.0));
        // ... and the string folds do not start from an open glottis on their
        // own: at 0.054 cm^2 with full pressure they sat still through the
        // whole "It" (silent, -56 dB), while an accented word's glottal onset
        // (pressed shut, then released) starts them within a cycle. A brief
        // near-closure (LINE_ONSET_KICK_AG for LINE_ONSET_KICK_MS) as the
        // adduction begins; 0: none.
        if (kLineOnsetKickAg > 0 && kLineOnsetKickMs > 0) {
          Event kick{e.t};
          kick.ag = kLineOnsetKickAg;
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
  }
  const bool isM = tok.ph == Ph::M;
  const float approachMs = isM ? kLineMApproachMs : tok.ph == Ph::NG ? kLineNgApproachMs : 0.0f;
  if (tract_ == Tract::Line && approachMs > 0 && prevWasVowel_ && !prevWasPlosive_) {
    Event ap{std::max(e.t - approachMs * ms, phonemeStart_ + 1)};
    ap.ac = isM ? kLineMApproachA : kLineNgApproachA;
    ap.acMs = static_cast<float>((e.t - ap.t) / ms);
    ap.lineX = linePlaceOf(tok.ph);
    addEventAt(ap);
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
  prevWasPlosive_ = false;
  prevWasNasal_ = in.cls == Cls::Nasal;
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
    planShapes(timeline_ - (hold ? kLinePausePlanHoldMs * 0.001 * fs_ : 0.0));
    planSentence();
  }
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
    case Ph::TH: { static const float s = param("LINE_TH_COART", 1.0f); return {{0.1f * s, 0.25f * s, 0.05f * s}, {0.05f * s, 0.35f * s, 0.25f * s}}; }  // x the θ's shares (lab 2026-11-09)
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
  const bool unreleased = next && (info(*next).cls == Cls::Plosive ||
                                   (tract_ == Tract::Line && pi.voiced && nextFric && tok.ph != Ph::JH && (nextVoicedFric ? kVstopVfricUnrel : kVstopFricUnrel) > 0.5f));  // (not the affricate's own release into its ʒ: inAffricate_ is set after this)
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
      lineKeys_.push_back({timeline_ + 0.5 * closure, sh});
      addShapeDemand(timeline_ + 0.3 * closure, timeline_ + 0.7 * closure, sh, tok.ph, kShapeBorrowedW);
      lastKeyShape_ = sh;
      haveLastKeyShape_ = true;
    }
  }
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
    const float apMs = velar ? kLineKgApproachMs : labial ? kLineBpApproachMs : 0.0f;
    if (tract_ == Tract::Line && apMs > 0 && prevWasVowel_ && !prevWasPlosive_ && !afterUnreleased) {
      Event ap{std::max(tClosed - apMs * ms, phonemeStart_ + 1)};
      if (ap.t < closeStart - ms) {
        ap.ac = velar ? kLineKgApproachA : kLineBpApproachA;
        ap.acMs = static_cast<float>((closeStart - ap.t) / ms);
        ap.lineX = linePlaceOf(tok.ph);
        addEventAt(ap);
      }
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
    addEvent(approach);
  }
  Event close{tClosed - sealMs * ms};
  // The line's constriction for this closure: its place, groove width and
  // rounding, and a voiced stop's soft, longer closure. (Plosives bypass the
  // generic event that carries these, so until 2026-11-03 every b, p, g, k
  // closed wherever the last consonant had left the tongue, mostly the
  // alveolar ridge: "abolish" was heard as "atolish".)
  close.lineX = linePlaceOf(tok.ph);
  {
    const bool post = tok.ph == Ph::CH || tok.ph == Ph::JH;
    close.lineHw = post ? kLinePostHw : 1.0f;
    close.lineLip = post ? kLinePostLip : 0.0f;
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
    close.lineExpand = tok.ph == Ph::JH ? kLineJhExpandFlow : 0.0f;
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
  rel.ac = nextIsPause ? kAcOpen : nextAero.ac;
  // On the line the release is also the next sound's own event (emit drops
  // the event of a sound after a plosive): it carries that sound's slit, place
  // and groove. Without them the f of "not forget" was made at the t's
  // alveolar place with the table's 0.1 cm^2: an s-like hiss 12 dB over the
  // vowels, 600 cm^3/s through what should be a near-sealed lip (2026-11-03).
  if (tract_ == Tract::Line && next && !nextIsPause && info(*next).cls == Cls::Fricative) {
    rel.ac = lineSlit(*next, rel.ac);
    rel.lineX = linePlaceOf(*next);
    const bool post = *next == Ph::SH || *next == Ph::ZH;
    rel.lineHw = post ? kLinePostHw : 1.0f;
    rel.lineLip = post ? kLinePostLip : 0.0f;
    if (post) rel.lineJetD = kLinePostJetD, rel.lineJetMix = kLinePostJetMix, rel.lineSlitScale = kLinePostSlitScale, rel.lineJetSmoothMs = kLinePostStSmoothMs, rel.lineJetFcMin = kLinePostFcMin, rel.lineJetFcMax = kLinePostFcMax, rel.lineFricLpr = kLinePostLpr;
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
  rel.acMs = pi.releaseMs * (tract_ == Tract::Line ? (tok.ph == Ph::D && kLineDReleaseScale >= 0 ? kLineDReleaseScale : (tok.ph == Ph::P && kLinePReleaseScale >= 0 ? kLinePReleaseScale : tok.ph == Ph::G && kLineGReleaseScale >= 0 ? kLineGReleaseScale : tok.ph == Ph::T && kLineTReleaseScale >= 0 ? kLineTReleaseScale : kLineReleaseScale) * (pi.voiced ? kLineVoicedReleaseScale : 1.0f)) : 1.0f);
  // LINE_FINAL_D_RELEASE_SCALE (lab knob 2026-11-10, D agent; < 0: off): a d before a pause opens over releaseMs x this
  // (Kokoro's final d: a soft -21..-24 dB release where puhuja's pops at -9).
  static const float kLineFinalDReleaseScale = param("LINE_FINAL_D_RELEASE_SCALE", -1.0f);
  if (tract_ == Tract::Line && tok.ph == Ph::D && nextIsPause && kLineFinalDReleaseScale >= 0) rel.acMs = pi.releaseMs * kLineFinalDReleaseScale;
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
  } else if (aspirate_ && !pi.voiced && !glottalised && next && lastPh_ != Ph::S &&
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
      rel.ag = tok.ph == Ph::P && kLinePAspAg >= 0 ? kLinePAspAg : kLineAspAg;
      rel.agMs = 5;
      // ... and the puff hisses: the turbulence of the full lung pressure
      // through the open glottis, far stronger than an h's (LINE_ASP alone
      // left it 30 dB under the readers' 3-5 kHz puff).
      rel.lineAspMul = tok.ph == Ph::P && kLinePStopAspMul >= 0 ? kLinePStopAspMul : tok.ph == Ph::T && kLineTStopAspMul >= 0 ? kLineTStopAspMul
                       : tok.ph == Ph::K && kLineKStopAspMul >= 0 ? kLineKStopAspMul : kLineStopAspMul;
      // The female's puffs ran 3 dB over her vowels where the male's were 5 dB under
      // ("power to abolish", 2026-11-05): her puff noise x LINE_PUFF_FEMALE.
      static const float kLinePuffFemale = param("LINE_PUFF_FEMALE", 0.25f);
      if (female_) rel.lineAspMul *= kLinePuffFemale;
      // A word-medial p t k puff x LINE_PUFF_MEDIAL (2026-11-06): Kokoro's medial p ("oppose", "support", -29..-52 dB
      // burst re the vowel) puffs ~10 dB under its word-initial ones ("poverty", "powers", -17..-22), whatever the
      // syllable's stress; LINE_VOT_HOLD_UNSTR shortens an unstressed one only.
      static const float kLinePuffMedial = param("LINE_PUFF_MEDIAL", 0.3f);
      if (!tok.wordStart) rel.lineAspMul *= kLinePuffMedial;
      // A p before l ɹ w j (LINE_P_LIQ_VOT x the puff's hold, LINE_P_LIQ_ASP x its noise; lab knobs 2026-11-10, P agent:
      // "price" -> "lice", "provide" -> "revived"; 1: as before).
      static const float kLinePLiqVot = param("LINE_P_LIQ_VOT", 1.0f), kLinePLiqAsp = param("LINE_P_LIQ_ASP", 3.0f);
      const bool pLiq = tok.ph == Ph::P && !isVowel(*next);
      if (pLiq) rel.lineAspMul *= kLinePLiqAsp;
      Event adductLine{release + (tok.ph == Ph::P && kLinePVotHoldMs >= 0 ? kLinePVotHoldMs : kLineVotHoldMs) * (curStress_ ? 1.0f : kLineVotHoldUnstressed) * (pLiq ? kLinePLiqVot : 1.0f) * ms};
      adductLine.ag = nextAero.ag;
      adductLine.agMs = kLineVotAdductMs;
      pendingKick_ = adductLine;
      // The hiss lasts while the folds close (cut at the adduction's start it
      // left a 10 ms silence before the voice).
      // (queued after the release, with the adduction, below)
      pendingAspEnd_ = Event{adductLine.t + kLineVotAdductMs * ms};
      pendingAspEnd_.lineAspMul = 1.0f;
      if (kLinePuffLpr > 0) {
        rel.linePuffLpr = tok.ph == Ph::P && kLinePPuffLpr > 0 ? kLinePPuffLpr : kLinePuffLpr;
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
    if (tract_ == Tract::Line && tok.ph == Ph::JH && kLineJhAg >= 0) rel.ag = kLineJhAg;  // (half-devoiced ʤ)
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
  // A final voiced stop before a pause: the breath stops this long before the release (ms; negative: after it;
  // <= -500: off), ramping down over
  // LINE_FINAL_VSTOP_CUT_RAMP_MS, so no voicing follows the release (Jyrki 2026-11-09: "God" "sounds a bit like
  // 'God-a', perhaps the pressure could stop 15 ms earlier"); the stored mouth pressure still gives a soft release.
  if (tract_ == Tract::Line && nextIsPause && pi.voiced) {
    static const float kLineFinalVstopCutMs = param("LINE_FINAL_VSTOP_CUT_MS", -1000.0f);
    static const float kLineFinalVstopCutRampMs = param("LINE_FINAL_VSTOP_CUT_RAMP_MS", 10.0f);
    if (kLineFinalVstopCutMs > -500.0f) {
      Event cut{std::max(release - kLineFinalVstopCutMs * ms, t_ + 1.0)};
      cut.press = 0.0f;
      cut.pressMs = kLineFinalVstopCutRampMs;
      addEventAt(cut);
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
      if (kLineFinalPressLeadMs > 0) {
        Event ease{std::max(release - kLineFinalPressLeadMs * ms, t_ + 1.0)};
        ease.press = kLineFinalPressFloor;
        ease.pressMs = kLineFinalPressLeadMs;
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
    const float finalAspMul = tok.ph == Ph::P && kLineFinalPAsp >= 0 ? kLineFinalPAsp : kLineFinalAspMul;
    const float finalAspMs = tok.ph == Ph::P && kLineFinalPAspMs >= 0 ? kLineFinalPAspMs : kLineFinalAspMs;
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
  static const float kLineGBurst = param("LINE_G_BURST", -1.0f);
  static const float kLineDBurst = param("LINE_D_BURST", -1.0f);  // (the same for the d; D agent 2026-11-10)
  if (vstopBurst) rel.lineFricMul = tok.ph == Ph::G && kLineGBurst >= 0 ? kLineGBurst : tok.ph == Ph::D && kLineDBurst >= 0 ? kLineDBurst : kLineVStopBurst;
  if (tract_ == Tract::Line && tok.ph == Ph::T && kLineTCloseFric != 1.0f && rel.lineFricMul < 0) rel.lineFricMul = next && *next == Ph::S ? kLineSFric : 1.0f;
  // The affricate's closure ran its slit noise at LINE_JH_CLOSE_FRIC (0.3) and
  // nothing reset it: the ʒ of ʤ hissed 15 dB under the word's s with 220
  // cm^3/s flowing through it ("the generosity" heard as "denerosity", Jyrki
  // 2026-11-07). The release restores the gain (LINE_JH_REL_FRIC).
  if (tract_ == Tract::Line && (tok.ph == Ph::JH || tok.ph == Ph::CH) && rel.lineFricMul < 0) {
    static const float kLineJhRelFric = param("LINE_JH_REL_FRIC", 5.0f);  // 2026-11-07, Jyrki: "pledge-l20vr5.wav is the best" (voiced ʤ, the hiss x5: -8..-15 dB re the s through the frication)
    rel.lineFricMul = kLineJhRelFric;
  }
  if (tract_ == Tract::Line && kLineCloseFric != 1.0f && rel.lineFricMul < 0 && tok.ph != Ph::T && tok.ph != Ph::JH && tok.ph != Ph::CH) rel.lineFricMul = 1.0f;  // (the release's burst keeps the full gain)
  // k's (and p's) release hiss x LINE_K_REL_FRIC / LINE_P_REL_FRIC while the
  // closure opens, back to 1 once it is open (2026-11-05, "It cannot.": the
  // k release peaked 3 dB over the following vowel across kennedy, Kokoro's
  // 7 dB under; the aspiration alone (LINE_K_STOP_ASP) took only 4 dB off).
  const float relFric = tok.ph == Ph::K ? kLineKRelFric : tok.ph == Ph::P ? kLinePRelFric : tok.ph == Ph::T ? kLineTRelFric : 1.0f;
  const bool relFricOn = tract_ == Tract::Line && relFric != 1.0f && !unreleased && !(next && info(*next).cls == Cls::Fricative);
  const bool nasalBurst = nasalVel && kNasalStopBurst != 1.0f && !relFricOn && !(next && info(*next).cls == Cls::Fricative);
  if (relFricOn) rel.lineFricMul = relFric;
  if (nasalBurst) rel.lineFricMul = kNasalStopBurst;  // (LINE_NASAL_STOP_BURST, restored below)
  // ʧ's ʃ phase: its emit() adds no airway event after the plosive (the release carries the targets), so the gain
  // rides on the release; the next sound's emit() sets its own (LINE_CH_FRIC).
  if (tract_ == Tract::Line && tok.ph == Ph::CH && kLineChFric != 1.0f) rel.lineFricMul = kLineChFric;
  if (closurePressEased_) {  // (LINE_CLOSURE_PRESS: the breath comes back with the release)
    static const float kLineClosurePressBackMs = param("LINE_CLOSURE_PRESS_BACK_MS", 15.0f);
    rel.press = 1.0f;
    rel.pressMs = kLineClosurePressBackMs;
    closurePressEased_ = false;
  }
  addEvent(rel);
  // k and g into a vowel in two steps (LINE_K_BURST_MS, 2026-11-04): the
  // closure opens first to a slit at the velum, its hiss the burst, and only
  // after LINE_K_BURST_MS on toward the vowel (with rel's own opening time).
  // (LINE_G_BURST_MS / LINE_G_BURST_A, lab knobs 2026-11-09, G agent; < 0: the k's)
  static const float kLineGBurstMs = param("LINE_G_BURST_MS", -1.0f), kLineGBurstA = param("LINE_G_BURST_A", -1.0f);
  const float kgBurstMs = tok.ph == Ph::G && kLineGBurstMs >= 0 ? kLineGBurstMs : kLineKBurstMs;
  const bool kBurst = tract_ == Tract::Line && kgBurstMs > 0 && (tok.ph == Ph::K || tok.ph == Ph::G) &&
                      !unreleased && next && isVowel(*next);
  if (kBurst) {
    Event slit{release};
    slit.ac = tok.ph == Ph::G && kLineGBurstA >= 0 ? kLineGBurstA : kLineKBurstA;
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
  if (kBurst) {
    Event open{release + kgBurstMs * ms};
    open.ac = rel.ac;
    open.acMs = rel.acMs;
    addEventAt(open);
  }
  if (dzRel) {  // (LINE_DZ_REL_FRIC: the z's own gain once the groove has formed)
    Event back{release + (rel.acMs + 10.0f) * ms};
    back.lineFricMul = dzBackFric;
    addEventAt(back);
  }
  if (vstopBurst || relFricOn || nasalBurst) {
    Event back{release + (rel.acMs + 10.0f) * ms};
    back.lineFricMul = 1.0f;
    addEventAt(back);
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
  static const float kLineKgRelShapeL = param("LINE_KG_REL_SHAPE_L", 0.0f), kLineKgRelMs = param("LINE_KG_REL_MS", 25.0f),
                     kLineKgRelAc = param("LINE_KG_REL_AC", 0.6f);
  const bool kgLiquid = next && (*next == Ph::L || *next == Ph::RR || *next == Ph::W || *next == Ph::J);
  if (tract_ == Tract::Line && (tok.ph == Ph::K || tok.ph == Ph::G) && !unreleased && next &&
      ((kLineKgRelShape > 0 && isVowel(*next)) || (kLineKgRelShapeL > 0 && kgLiquid))) {
    // The velar pinch after k g (LINE_KG_REL_SHAPE, 2026-11-04): the vowel's
    // first 25 ms are asked for its own shape with a narrow tongue hump at
    // the velum (0.6 cm^2, half-width 2 cm, at k's place), weighted by the
    // knob and only in the hump's channels (as for j), so F2 and F3 leave the
    // release together and part into the vowel.
    TractShape v;
    if (lineShapeOf(*next, &v)) {
      v.xc = linePlaceOf(tok.ph);
      v.ac = kLineKgRelAc;
      v.wc = 2.0f;
      v.sc = 1.0f;
      addShapeDemand(release, release + kLineKgRelMs * ms, v, Ph::J, isVowel(*next) ? kLineKgRelShape : kLineKgRelShapeL);
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
      Event close{release + (tok.ph == Ph::P && kLineFinalPAspMs >= 0 ? kLineFinalPAspMs : kLineFinalAspMs) * ms};
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
    burst.ac = kLineBurstArea * (tSoft ? softBurst : 1.0f);
    burst.acMs = 3;
    addEvent(burst);
    Event narrow{release + kLineBurstMs * ms};
    narrow.ac = tract_ == Tract::Line ? lineSlit(*next, nextAero.ac) : nextAero.ac;  // (the table's 0.1 cm^2 left the f of "not forget" leaking 380 cm^3/s)
    narrow.acMs = 8;
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

float Synth::tick() {
  // Start the articulator movements and noise changes that are due.
  while (!events_.empty() && events_.front().t <= t_) {
    const Event& e = events_.front();
    if (e.press >= 0) {
      press_.start(e.press, msToSamples(e.pressMs, fs_));
      pressTo_ = e.press;
    }
    if (e.aspGain >= 0) aspGainTarget_ = e.aspGain;
    if (e.lineX >= 0) lineXTarget_ = e.lineX;
    // The groove width and rounding step at once (the place glides): an
    // affricate's 30-60 ms of hiss never reached a followed target.
    if (e.lineLevel > 0) lineLevelTarget_ = e.lineLevel;
    if (e.lineAspMul >= 0) lineAspMul_ = e.lineAspMul;
    if (e.lineFricMul >= 0) lineFricMul_ = e.lineFricMul;
    if (e.lineFricMul >= 0 && std::getenv("PUHUJA_LINE_TRACE")) std::fprintf(stderr, "fricmul t %.3f mul %.3f\n", t_ / fs_, e.lineFricMul);
    if (e.lineFricFade >= 0) { lineFricFade_ = e.lineFricFade; line_.setFricFade(lineFricFade_); }
    if (e.lineSoft >= 0) lineSoft_ = e.lineSoft;
    if (e.lineExtra >= 0) lineExtra_ = e.lineExtra;
    if (e.lineExpand >= 0) lineExpand_ = e.lineExpand;
    if (e.lineHw >= 0) lineHw_ = lineHwTarget_ = e.lineHw;
    if (e.lineJetD >= 0) line_.setJetThickness(e.lineJetD, e.lineJetMix, e.lineSlitScale, e.lineJetSmoothMs, e.lineJetFcMin, e.lineJetFcMax);
    if (e.lineFricLpr >= 0) line_.setFricLpRatio(e.lineFricLpr);
    if (e.lineAspHp >= 0) {
      static const int tAspPoles = static_cast<int>(param("LINE_T_ASP_HP_POLES", 3.0f));
      line_.setAspHp(e.lineAspHp, tAspPoles);
    }
    if (e.linePuffLpr >= 0) line_.setPuffLpRatio(e.linePuffLpr);
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
    if (kLineXSnapArea > 0 && ac > kLineXSnapArea) lineX_ = lineXTarget_;
    else lineX_ += (lineXTarget_ - lineX_) * lineXFollow_;
    lineLevel_ += (lineLevelTarget_ - lineLevel_) * lineXFollow_;
    lineHw_ += (lineHwTarget_ - lineHw_) * lineXFollow_;
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
          epilarynx(a);
          epilarynx(b);
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
      epilarynx(sh);
      line_.setShape(sh, lineX_, ac, an);
    lineShapeDone:;
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
    line_.setBreath(kLineBreath > 0 ? kLineBreath * kLineAsp * lineAspMul_ * (1.0f - smoothstep(kAgVoice, kAgBreathy, ag)) : 0.0f);
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
    line_.setExpansion(voicedClosure ? expandFlow : vbarFlow, expand > 1.0f || vbarClosure);  // a voiced closure (lineSoft_ marks it); the line holds the cavity at LINE_EXPAND_HOLD of the lung pressure
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
      all(0.2f); w[LIPA] = w[LIPL] = 1.0f; w[XC] = w[AC] = w[SC] = kWBackW; return;
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
