// Streaming front end: reads text from stdin (or the command line), writes
// mono float32 audio as a WAV file or as raw samples to stdout.
//
//   echo "aiaiai ouou" | ./puhuja -o out.wav
//   ./puhuja -r "aeiou" | aplay -f FLOAT_LE -r 48000 -c 1
//   ./puhuja -en -r "Hello, how are you?" | aplay -f FLOAT_LE -r 48000 -c 1
//
//   -en         read English text (default: Finnish spelling / IPA)
//   -g2p        print English pronunciations of the words on stdin
//   -g2p-raw    the same from the letter-to-sound rules alone
//   -m damped|spline  formant motion: damped target-following (default) or
//               a spline through the targets

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

#include "english.h"
#include "puhuja.h"

#if defined(__SSE__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace {

// -room: a small, dry room. A few early reflections (7-23 ms, -14..-22 dB)
// and a short, quiet tail (Schroeder: 4 damped combs, 2 all-passes, RT60
// ~0.25 s, -24 dB), so the voice sits in a space and pauses are not
// digital silence.
class Room {
 public:
  explicit Room(float fs) {
    const float tapMs[4] = {7, 11, 17, 23}, tapDb[4] = {-14, -17, -19, -22};
    for (int i = 0; i < 4; ++i) {
      tap_[i] = static_cast<int>(tapMs[i] * 0.001f * fs);
      tapGain_[i] = std::pow(10.0f, tapDb[i] / 20.0f);
    }
    const float combMs[4] = {29.7f, 37.1f, 41.1f, 43.7f};
    for (int i = 0; i < 4; ++i) {
      comb_[i].assign(static_cast<size_t>(combMs[i] * 0.001f * fs), 0.0f);
      fb_[i] = std::pow(10.0f, -3.0f * combMs[i] * 0.001f / kRt60);
    }
    // Four all-passes (two were too few: a grainy, sparse tail).
    const float apMs[4] = {5.0f, 3.1f, 1.7f, 0.9f};
    for (int i = 0; i < 4; ++i) ap_[i].assign(static_cast<size_t>(apMs[i] * 0.001f * fs), 0.0f);
    earlyLpK_ = 1.0f - std::exp(-2.0f * 3.14159265f * 5000.0f / fs);
    line_.assign(static_cast<size_t>(0.03f * fs), 0.0f);
  }
  float process(float x) {
    line_[pos_] = x;
    // Early reflections with alternating signs (all positive, they made
    // a comb ripple of 11-19 dB above 300 Hz), softened above ~5 kHz.
    float early = 0;
    for (int i = 0; i < 4; ++i)
      early += (i % 2 ? -1.0f : 1.0f) * tapGain_[i] * line_[(pos_ + line_.size() - tap_[i]) % line_.size()];
    earlyLp_ += earlyLpK_ * (early - earlyLp_);
    early = earlyLp_;
    pos_ = (pos_ + 1) % line_.size();
    float tail = 0;
    for (int i = 0; i < 4; ++i) {
      std::vector<float>& c = comb_[i];
      const float y = c[cp_[i]];
      lp_[i] += 0.4f * (y - lp_[i]);  // damping: the highs die faster
      c[cp_[i]] = x + fb_[i] * lp_[i];
      cp_[i] = (cp_[i] + 1) % c.size();
      tail += y;
    }
    for (int i = 0; i < 4; ++i) {
      std::vector<float>& a = ap_[i];
      const float g = i < 2 ? 0.5f : 0.6f;
      const float d = a[ap_pos_[i]];
      const float v = tail + g * d;
      a[ap_pos_[i]] = v;
      ap_pos_[i] = (ap_pos_[i] + 1) % a.size();
      tail = d - g * v;
    }
    return x + early + kTailGain * tail;
  }

 private:
  static constexpr float kRt60 = 0.25f, kTailGain = 0.063f;  // -24 dB
  std::vector<float> line_, comb_[4], ap_[4];
  size_t pos_ = 0, cp_[4] = {0, 0, 0, 0}, ap_pos_[4] = {0, 0, 0, 0};
  float earlyLp_ = 0, earlyLpK_ = 1;
  int tap_[4];
  float tapGain_[4], fb_[4], lp_[4] = {0, 0, 0, 0};
};

void usage(FILE* f) {
  fputs(
      "puhuja: a streaming formant and vocal-fold speech synthesizer.\n"
      "Reads text from the command line or stdin and writes mono float32 audio\n"
      "at 48 kHz.\n"
      "\n"
      "usage: puhuja [options] [text]\n"
      "       echo text | puhuja [options]\n"
      "\n"
      "  -en           read English text (default: Finnish spelling or IPA)\n"
      "  -o file       output WAV file (default out.wav)\n"
      "  -r            raw float32 samples to stdout instead of a WAV file\n"
      "  -female       female voice\n"
      "  -f0 Hz        base pitch (default 105 Hz male, 166 Hz female)\n"
      "  -rate x       English speaking rate, 0.9 = 10 % slower (default 0.95)\n"
      "  -room         a small room: early reflections and a short tail\n"
      "  -l file       write phoneme timings (start, end, symbol)\n"
      "  -words file   write word timings (start, end, word number from 1)\n"
      "  -w source     voice source: physical (default), grid (4x8 multi-mass folds), shaped, glottal, triangle, square\n"
      "  PUHUJA_PARAMS=file  research parameters (\"name value\" lines, see params.h); without it a\n"
      "                puhuja.params in the working directory or next to the binary is read (a running\n"
      "                tools/word_opt.py keeps its best state there)\n"
      "  -m motion     formant motion: damped/dampen (default) or spline\n"
      "  -leak x       part of the source that bypasses the formants (default 0.03)\n"
      "  -g2p          print English pronunciations of the words on stdin\n"
      "  -g2p-raw      the candidate before the stored corrections (letter rules, prefixes)\n"
      "  -chunk N      feed the text N bytes at a time (for testing streaming)\n"
      "  -h, --help    this help\n"
      "\n"
      "examples:\n"
      "  puhuja -en -o hello.wav \"Hello, how are you?\"\n"
      "  puhuja -en -r \"Hello.\" | aplay -f FLOAT_LE -r 48000 -c 1\n"
      "  puhuja \"Hyvää huomenta.\"\n",
      f);
}

void put32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
void put16(FILE* f, uint16_t v) { fwrite(&v, 2, 1, f); }

// IEEE float WAV header. Sizes are patched at the end if the file is seekable.
void writeWavHeader(FILE* f, uint32_t rate, uint32_t dataBytes) {
  fwrite("RIFF", 1, 4, f);
  put32(f, 36 + dataBytes);
  fwrite("WAVEfmt ", 1, 8, f);
  put32(f, 16);
  put16(f, 3);  // WAVE_FORMAT_IEEE_FLOAT
  put16(f, 1);  // mono
  put32(f, rate);
  put32(f, rate * 4);
  put16(f, 4);
  put16(f, 32);
  fwrite("data", 1, 4, f);
  put32(f, dataBytes);
}

}  // namespace

int main(int argc, char** argv) {
#if defined(__SSE__) || defined(_M_X64)
  // Flush denormals to zero: decaying filter states in long silences.
  _mm_setcsr(_mm_getcsr() | 0x8040);
#endif
  const char* outPath = "out.wav";
  bool raw = false;
  float f0 = 0;  // -f0 Hz: overrides the voice's base pitch
  bool female = false;  // -female: female voice preset
  const char* labelPath = nullptr;  // -l file: phoneme timings (start end symbol)
  const char* wordsPath = nullptr;  // -words file: word timings (start end number)
  float leak = 0.03f;    // -leak x: part of the source bypassing the formants
  // -w physical|triangle|square|glottal: voice source (default physical).
  puhuja::Carrier carrier = puhuja::Carrier::Physical;
  std::string text;
  bool english = false;
  bool useRoom = false;  // -room: early reflections and a short tail
  // -rate x: English speaking rate (0.9 = 10 % slower). The default is a
  // little slower than the voice-over reference it is fitted to.
  float rate = 0.95f;
  size_t chunk = 0;  // -chunk N: feed the text N bytes at a time (testing streaming)
  puhuja::FormantMotion motion = puhuja::FormantMotion::Damped;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-?") || !strcmp(argv[i], "--help") ||
        !strcmp(argv[i], "-help")) {
      usage(stdout);
      return 0;
    }
    if (argv[i][0] == '-' && argv[i][1] && !(argv[i][1] >= '0' && argv[i][1] <= '9') &&
        strcmp(argv[i], "-o") && strcmp(argv[i], "-r") && strcmp(argv[i], "-w") &&
        strcmp(argv[i], "-l") && strcmp(argv[i], "-words") && strcmp(argv[i], "-leak") && strcmp(argv[i], "-m") &&
        strcmp(argv[i], "-room") && strcmp(argv[i], "-rate") && strcmp(argv[i], "-en") &&
        strcmp(argv[i], "-g2p") && strcmp(argv[i], "-g2p-raw") && strcmp(argv[i], "-female") &&
        strcmp(argv[i], "-f0") && strcmp(argv[i], "-chunk")) {
      fprintf(stderr, "puhuja: unknown option %s\n\n", argv[i]);
      usage(stderr);
      return 2;
    }
    if (!strcmp(argv[i], "-o") && i + 1 < argc) {
      outPath = argv[++i];
    } else if (!strcmp(argv[i], "-r")) {
      raw = true;
    } else if (!strcmp(argv[i], "-w") && i + 1 < argc) {
      ++i;
      carrier = !strcmp(argv[i], "physical") ? puhuja::Carrier::Physical
                : !strcmp(argv[i], "glottal") ? puhuja::Carrier::Glottal
                : !strcmp(argv[i], "shaped") ? puhuja::Carrier::Shaped
                : !strcmp(argv[i], "grid") ? puhuja::Carrier::Grid
                : !strcmp(argv[i], "square") ? puhuja::Carrier::Square
                                             : puhuja::Carrier::Triangle;
    } else if (!strcmp(argv[i], "-words") && i + 1 < argc) {
      wordsPath = argv[++i];
    } else if (!strcmp(argv[i], "-l") && i + 1 < argc) {
      labelPath = argv[++i];
    } else if (!strcmp(argv[i], "-leak") && i + 1 < argc) {
      leak = atof(argv[++i]);
    } else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
      // -m damped|spline: how the formants move between targets.
      motion = !strcmp(argv[++i], "spline")    ? puhuja::FormantMotion::Spline
               : !strcmp(argv[i], "planned") ? puhuja::FormantMotion::Planned
                                            : puhuja::FormantMotion::Damped;
    } else if (!strcmp(argv[i], "-room")) {
      useRoom = true;
    } else if (!strcmp(argv[i], "-rate") && i + 1 < argc) {
      rate = static_cast<float>(atof(argv[++i]));
    } else if (!strcmp(argv[i], "-en")) {
      english = true;
    } else if (!strcmp(argv[i], "-chunk") && i + 1 < argc) {
      chunk = static_cast<size_t>(atoi(argv[++i]));  // testing: feed the text in pieces
    } else if (!strcmp(argv[i], "-g2p") || !strcmp(argv[i], "-g2p-raw")) {
      const bool raw = !strcmp(argv[i], "-g2p-raw");
      char line[256];
      while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        auto ph = raw ? puhuja::englishRules(line) : puhuja::englishPronounce(line);
        for (size_t k = 0; k < ph.size(); ++k) printf(k ? " %s" : "%s", ph[k].c_str());
        printf("\n");
      }
      return 0;
    } else if (!strcmp(argv[i], "-female")) {
      female = true;
    } else if (!strcmp(argv[i], "-f0") && i + 1 < argc) {
      f0 = atof(argv[++i]);
    } else {
      if (!text.empty()) text += ' ';
      text += argv[i];
    }
  }

  // Nothing to say and nothing piped in: show how to use it instead of
  // silently waiting for typed text.
  if (text.empty() && isatty(fileno(stdin))) {
    usage(stderr);
    return 2;
  }

  puhuja::Synth synth(48000);
  if (female) synth.setVoice(puhuja::VoicePreset::Female);
  if (f0 > 0) synth.setBaseF0(f0);
  synth.setCarrier(carrier);
  synth.setFormantMotion(motion);
  synth.setLeak(leak);
  FILE* out = raw ? stdout : fopen(outPath, "wb");
  if (!out) {
    perror(outPath);
    return 1;
  }
  if (!raw) writeWavHeader(out, 48000, 0xffffffffu - 36);

  uint64_t total = 0;
  std::unique_ptr<Room> room;
  if (useRoom) room = std::make_unique<Room>(48000.0f);
  float buf[1024];
  auto drain = [&] {
    size_t n;
    while ((n = synth.pull(buf, 1024)) > 0) {
      if (room)
        for (size_t k = 0; k < n; ++k) buf[k] = room->process(buf[k]);
      fwrite(buf, sizeof(float), n, out);
      total += n;
    }
    fflush(out);
  };

  puhuja::EnglishReader reader(synth);
  reader.setRate(rate);
  if (!english) {  // the reader set these up for English
    synth.setSpellingRules(true);
    synth.setFricationGain(1);
  }
  auto feed = [&](const char* s) {
    if (english) reader.push(s); else synth.pushText(s);
  };
  synth.pushText(" ");  // short lead-in silence
  if (!text.empty()) {
    feed(text.c_str());
    if (english) reader.flush();
    drain();
  } else {
    char line[4096];
    while (fgets(line, sizeof line, stdin)) {
      if (!english)  // the English reader uses line ends (headings, paragraphs)
        for (char* p = line; *p; ++p)
          if (*p == '\n') *p = ' ';
      if (chunk > 0) {
        for (size_t k = 0, n = strlen(line); k < n; k += chunk) {
          const std::string piece(line + k, std::min(chunk, n - k));
          feed(piece.c_str());
          drain();
        }
      } else {
        feed(line);
        drain();
      }
    }
    if (english) reader.flush();
  }
  synth.pushText(" ");  // let the last vowel fade out
  synth.finish();
  drain();
  if (room) {  // let the room's tail ring out
    for (int k = 0; k < 48000 / 3; ++k) {
      const float y = room->process(0.0f);
      fwrite(&y, sizeof(float), 1, out);
      ++total;
    }
  }

  if (labelPath) {
    FILE* lf = fopen(labelPath, "w");
    if (!lf) {
      perror(labelPath);
      return 1;
    }
    for (const auto& l : synth.labels())
      fprintf(lf, "%.4f\t%.4f\t%s\n", l.start, l.end, l.symbol.c_str());
    fclose(lf);
  }
  if (wordsPath) {
    FILE* wf = fopen(wordsPath, "w");
    if (!wf) {
      perror(wordsPath);
      return 1;
    }
    // A word runs from its first phoneme to the next word's first phoneme or
    // the next pause, whichever comes first; a consonant shared across the
    // boundary ("this century") is split in the middle.
    const auto labels = synth.labels();
    auto startOf = [&](size_t i) { return labels[i].wordJoin ? 0.5 * (labels[i].start + labels[i].end) : labels[i].start; };
    bool tagged = false;
    for (const auto& l : labels) tagged = tagged || l.word >= 0;
    if (tagged) {
      // English: the labels carry the input word's number; a word's span runs
      // from its first label to its last (a word puhuja splits, like a
      // spelled abbreviation, stays one span), pauses left out.
      int cur = -1;
      double a = 0, b = 0;
      auto flush = [&] { if (cur >= 0) fprintf(wf, "%.4f\t%.4f\t%d\n", a, b, cur); };
      for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i].word < 0) continue;
        if (labels[i].word != cur) {
          flush();
          cur = labels[i].word;
          a = startOf(i);
          b = labels[i].end;
        } else {
          b = labels[i].end;
        }
        if (i + 1 < labels.size() && labels[i + 1].wordJoin && labels[i + 1].word != cur) b = startOf(i + 1);
      }
      flush();
    } else {
      int n = 0;
      for (size_t i = 0; i < labels.size(); ++i) {
        if (!labels[i].wordStart) continue;
        size_t j = i + 1;
        while (j < labels.size() && !labels[j].wordStart && labels[j].symbol != "_") ++j;
        fprintf(wf, "%.4f\t%.4f\t%d\n", startOf(i), j < labels.size() ? startOf(j) : labels[i].end, ++n);
      }
    }
    fclose(wf);
  }

  if (!raw && fseek(out, 0, SEEK_SET) == 0)
    writeWavHeader(out, 48000, static_cast<uint32_t>(total * 4));
  if (!raw) fclose(out);
  return 0;
}
