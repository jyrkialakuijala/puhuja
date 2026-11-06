// Kokoro mode (Jyrki 2026-11-06): an external pitch and rhythm profile guiding
// the synthesis, for comparison. The synthesizer sees only ProsodyGuide through
// Synth::setProsodyGuide(); everything else lives here (ref/opt/prosody/KOKORO_MODE.md).
//
// A profile is a JSON file (version 1), produced by tools/prosody_from_ref.py
// (the labels layer) and tools/prosody_extract.py (words, tracks):
//   {"version": 1, "source": "...", "text": "...", "base_f0_hz": 85.4,
//    "labels": [{"sym": "æ", "ms": 143, "f0": [st, st]}, ...],   // in puhuja's -l label order
//    "words":  [{"text": "torch", "t": [s, s], "f0": [st x 5] | null, "level": dB, "pause_after": s}, ...],
//    "tracks": {"t0": 0, "hop": 0.005, "f0_st": [st | null, ...], "energy_db": [dB, ...]},
//    "phones": [{"sym": "æ", "t": [s, s]}, ...]}                 // the reference's phones (informational)
// Absent sections are absent. f0 in semitones re base_f0_hz; times in seconds
// of the reference audio.
#pragma once

#include <string>
#include <vector>

namespace puhuja {

struct ProsodyProfile {
  static constexpr float kNone = -1000.0f;  // "no value" for pitches, levels
  int version = 0;
  std::string source, text;
  float baseF0 = 0;
  struct Label { std::string sym; float ms = 0, pitch = kNone, pitchEnd = kNone; };
  struct Word { std::string text; double t0 = 0, t1 = 0; std::vector<float> f0; float level = kNone, pauseAfter = 0; };
  struct Tracks { double t0 = 0, hop = 0; std::vector<float> f0St, energyDb; bool empty() const { return f0St.empty() && energyDb.empty(); } };
  struct Phone { std::string sym; double t0 = 0, t1 = 0; };
  std::vector<Label> labels;
  std::vector<Word> words;
  Tracks tracks;
  std::vector<Phone> phones;
  // Reads the file; false with *err set on a syntax or schema problem.
  bool load(const std::string& path, std::string* err);
};

// The policy: which layers are on, and the answers the synthesizer asks for.
class ProsodyGuide {
 public:
  enum Layer { kRhythm = 1, kPitch = 2, kLevel = 4, kAll = 7 };
  ProsodyGuide(const ProsodyProfile& profile, int layers = kAll) : profile_(profile), layers_(layers) {}
  int layers() const { return layers_; }
  bool has(Layer l) const { return (layers_ & l) != 0; }
  const ProsodyProfile& profile() const { return profile_; }

  // The labels layer: the synthesizer walks its label sequence; each call
  // offers the next label's symbol (addLabel's spelling) and gets the planned
  // length (ms, 0: keep) and a vowel's two pitches (kNone: keep). Resyncs over
  // a few labels when the symbols drift; false when nothing matches.
  bool nextLabel(const std::string& sym, float* ms, float* pitch, float* pitchEnd);
  void reset() { labelAt_ = 0; warp_.clear(); warpAt_ = 0; }
  bool hasLabels() const { return !profile_.labels.empty(); }
  bool hasWords() const { return !profile_.words.empty() && !wordMap_.empty(); }
  bool hasTracks() const { return !profile_.tracks.empty(); }

  // The words layer: the input text's words (whitespace-split) are matched to
  // the profile's words by sequence alignment of their normalised spellings.
  void setInputWords(const std::vector<std::string>& words);
  // Called at a word's first token with the planned total of its tokens (ms):
  // the factor that makes it the reference's length (false: no reference word).
  bool wordScale(int inputWord, float plannedMs, float* scale) const;
  // Registers the word's span on the render clock (seconds) once its length is
  // final: the time warp the tracks are read through.
  void wordStart(int inputWord, double renderT0, double renderMs);
  // The tracks layer through the warp: the reference's f0 (semitones re its
  // base) and level (dB re its voiced median) at a render time; false where
  // the reference is unvoiced or the time is outside any registered word.
  bool f0At(double renderT, float* st);
  // perWord: the word's level (the words layer; Kokoro's reduction is per word and smooth), else the 5 ms track.
  bool levelAt(double renderT, float* db, bool perWord);
  // The reference's gap after an input word (ms; false: no reference word), and before its first word.
  bool pauseAfterMs(int inputWord, float* ms) const;
  float leadInMs() const;

 private:
  const ProsodyProfile& profile_;
  int layers_;
  size_t labelAt_ = 0;
  std::vector<int> wordMap_;  // input word -> profile word, -1 none
  struct Warp { double r0, r1, ref0, ref1; int word; };
  std::vector<Warp> warp_;
  size_t warpAt_ = 0;
  bool refTime(double renderT, double* refT, int* word = nullptr);
};

}  // namespace puhuja
