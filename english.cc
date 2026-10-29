#include "english.h"

#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace puhuja {
namespace {

#include "english_data.inc"
#include "english_shape.inc"

static_assert(sizeof(kEnglishShape) / sizeof(kEnglishShape[0]) == static_cast<size_t>(Ph::Count));

constexpr int kLevels = sizeof(kEnglishLadder) / sizeof(kEnglishLadder[0]);
constexpr int kPad = 4;

// Both tables are front-coded: per entry, the length of the prefix shared
// with the previous key, the rest of the key (bytes 0x22-0x7f), the value
// (rules: tokens >= 0x80; exceptions: an edit script) and a 0 terminator.
using Table = std::unordered_map<std::string, std::string>;

Table decode(const unsigned char* data, size_t size) {
  Table t;
  std::string key;
  size_t i = 0;
  while (i < size) {
    key.resize(data[i++]);
    while (i < size && data[i] >= 0x22 && data[i] < 0x80) key += static_cast<char>(data[i++]);
    std::string value;
    while (i < size && data[i] != 0) value += static_cast<char>(data[i++]);
    ++i;
    t.emplace(key, std::move(value));
  }
  return t;
}

// Minimal raw-deflate decoder (RFC 1951) for the tables, after puff.c.
class Inflater {
 public:
  Inflater(const unsigned char* in, size_t size) : in_(in), size_(size) {}

  std::string run(size_t expected) {
    out_.reserve(expected);
    int last;
    do {
      last = bits(1);
      const int type = bits(2);
      if (type == 0) {
        stored();
      } else if (type == 1) {
        fixed();
      } else if (type == 2) {
        dynamic();
      } else {
        break;
      }
    } while (!last && pos_ <= size_);
    return out_;
  }

 private:
  struct Huffman {
    uint16_t count[16];
    uint16_t symbol[320];
  };

  int bits(int n) {
    int v = 0;
    for (int i = 0; i < n; ++i) {
      if (bitCount_ == 0) {
        bitBuf_ = pos_ < size_ ? in_[pos_] : 0;
        ++pos_;
        bitCount_ = 8;
      }
      v |= (bitBuf_ & 1) << i;
      bitBuf_ >>= 1;
      --bitCount_;
    }
    return v;
  }

  static void build(Huffman& h, const uint8_t* lengths, int n) {
    uint16_t offs[16];
    for (int i = 0; i < 16; ++i) h.count[i] = 0;
    for (int i = 0; i < n; ++i) ++h.count[lengths[i]];
    h.count[0] = 0;
    offs[1] = 0;
    for (int i = 1; i < 15; ++i) offs[i + 1] = offs[i] + h.count[i];
    for (int i = 0; i < n; ++i)
      if (lengths[i]) h.symbol[offs[lengths[i]]++] = static_cast<uint16_t>(i);
  }

  int decode(const Huffman& h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len < 16; ++len) {
      code |= bits(1);
      const int count = h.count[len];
      if (code - count < first) return h.symbol[index + (code - first)];
      index += count;
      first = (first + count) << 1;
      code <<= 1;
    }
    return -1;
  }

  void stored() {
    bitCount_ = 0;  // skip to a byte boundary
    if (pos_ + 4 > size_) return;
    const size_t len = in_[pos_] | (in_[pos_ + 1] << 8);
    pos_ += 4;
    for (size_t i = 0; i < len && pos_ < size_; ++i) out_ += static_cast<char>(in_[pos_++]);
  }

  void codes(const Huffman& lit, const Huffman& dist) {
    static const uint16_t kLenBase[] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                        31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static const uint8_t kLenExtra[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                        2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static const uint16_t kDistBase[] = {1,    2,    3,    4,    5,    7,     9,     13,    17,  25,
                                         33,   49,   65,   97,   129,  193,   257,   385,   513, 769,
                                         1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static const uint8_t kDistExtra[] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                         6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    for (;;) {
      int sym = decode(lit);
      if (sym < 0 || sym == 256 || pos_ > size_) return;
      if (sym < 256) {
        out_ += static_cast<char>(sym);
        continue;
      }
      sym -= 257;
      if (sym >= 29) return;
      const size_t len = kLenBase[sym] + bits(kLenExtra[sym]);
      const int d = decode(dist);
      if (d < 0 || d >= 30) return;
      const size_t back = kDistBase[d] + bits(kDistExtra[d]);
      if (back > out_.size()) return;
      for (size_t i = 0; i < len; ++i) out_ += out_[out_.size() - back];
    }
  }

  void fixed() {
    uint8_t lengths[320];
    int i = 0;
    for (; i < 144; ++i) lengths[i] = 8;
    for (; i < 256; ++i) lengths[i] = 9;
    for (; i < 280; ++i) lengths[i] = 7;
    for (; i < 288; ++i) lengths[i] = 8;
    Huffman lit, dist;
    build(lit, lengths, 288);
    for (i = 0; i < 30; ++i) lengths[i] = 5;
    build(dist, lengths, 30);
    codes(lit, dist);
  }

  void dynamic() {
    static const uint8_t kOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    const int nlen = bits(5) + 257, ndist = bits(5) + 1, ncode = bits(4) + 4;
    uint8_t lengths[320] = {};
    for (int i = 0; i < ncode; ++i) lengths[kOrder[i]] = static_cast<uint8_t>(bits(3));
    Huffman lencode, lit, dist;
    build(lencode, lengths, 19);
    int i = 0;
    while (i < nlen + ndist) {
      int sym = decode(lencode);
      if (sym < 0) return;
      if (sym < 16) {
        lengths[i++] = static_cast<uint8_t>(sym);
        continue;
      }
      uint8_t len = 0;
      int rep;
      if (sym == 16) {
        if (i == 0) return;
        len = lengths[i - 1];
        rep = 3 + bits(2);
      } else if (sym == 17) {
        rep = 3 + bits(3);
      } else {
        rep = 11 + bits(7);
      }
      while (rep-- && i < nlen + ndist) lengths[i++] = len;
    }
    build(lit, lengths, nlen);
    build(dist, lengths + nlen, ndist);
    codes(lit, dist);
  }

  const unsigned char* in_;
  size_t size_;
  size_t pos_ = 0;
  int bitBuf_ = 0, bitCount_ = 0;
  std::string out_;
};

Table load(const unsigned char* z, size_t zsize, size_t size) {
  const std::string data = Inflater(z, zsize).run(size);
  return decode(reinterpret_cast<const unsigned char*>(data.data()), data.size());
}

const Table& rules() {
  static const Table t = load(kEnglishRules, sizeof(kEnglishRules), kEnglishRulesSize);
  return t;
}

const Table& exceptions() {
  static const Table t =
      load(kEnglishExceptions, sizeof(kEnglishExceptions), kEnglishExceptionsSize);
  return t;
}

bool isVowelToken(uint8_t t) {
  const char* s = kEnglishTokens[t - 0x80];
  char c = s[strlen(s) - 1];
  return c >= '0' && c <= '2';
}

// Exactly one primary stress (as in tools/english_build.py fix_stress).
void fixStress(std::vector<uint8_t>& ph) {
  int first = -1, firstFull = -1, primaries = 0;
  for (size_t i = 0; i < ph.size(); ++i) {
    if (!isVowelToken(ph[i])) continue;
    const char* s = kEnglishTokens[ph[i] - 0x80];
    char st = s[strlen(s) - 1];
    if (st == '1') {
      if (primaries++) ph[i] += 1;  // "X1" -> "X2": the next token
      continue;
    }
    if (first < 0) first = static_cast<int>(i);
    if (firstFull < 0 && strcmp(s, "AH0") && strcmp(s, "IH0") && strcmp(s, "ER0"))
      firstFull = static_cast<int>(i);
  }
  if (primaries == 0) {
    int pick = firstFull >= 0 ? firstFull : first;
    if (pick >= 0) {
      const char* s = kEnglishTokens[ph[pick] - 0x80];
      ph[pick] += '1' - s[strlen(s) - 1];
    }
  }
}

// A prefix's stem has at least this many letters.
constexpr size_t kMinStem = 3;

std::vector<uint8_t> ruleTokens(const std::string& w) {
  const std::string padded = std::string(kPad, '#') + w + std::string(kPad, '#');
  const Table& t = rules();
  std::vector<uint8_t> out;
  std::string key;
  for (size_t i = 0; i < w.size(); ++i) {
    const int pi = static_cast<int>(i) + kPad;
    for (int lv = kLevels - 1; lv >= 0; --lv) {
      const int l = kEnglishLadder[lv][0], r = kEnglishLadder[lv][1];
      key.assign(1, static_cast<char>('1' + lv));
      key.append(padded, pi - l, l + 1 + r);
      auto it = t.find(key);
      if (it != t.end()) {
        for (unsigned char c : it->second) out.push_back(c);
        break;
      }
    }
  }
  fixStress(out);
  return out;
}

std::vector<uint8_t> applyEdit(const std::vector<uint8_t>& cand, const std::string& script) {
  std::vector<uint8_t> out;
  size_t i = 0;
  for (size_t k = 0; k < script.size(); ++k) {
    const uint8_t b = static_cast<uint8_t>(script[k]);
    if (b < 0x20) {
      for (int n = 0; n < b && i < cand.size(); ++n) out.push_back(cand[i++]);
    } else if (b == 0x20) {
      ++i;
    } else if (b == 0x21) {
      if (++k < script.size()) out.push_back(static_cast<uint8_t>(script[k]));
    } else {
      out.push_back(b);
      ++i;
    }
  }
  while (i < cand.size()) out.push_back(cand[i++]);
  return out;
}

std::string lower(std::string_view w) {
  std::string s(w);
  for (char& c : s)
    if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
  return s;
}

uint8_t tokenCode(const std::string& name) {
  for (int t = 0; t < static_cast<int>(sizeof(kEnglishTokens) / sizeof(kEnglishTokens[0])); ++t)
    if (name == kEnglishTokens[t]) return static_cast<uint8_t>(0x80 + t);
  return 0;
}

std::vector<uint8_t> pronounce(const std::string& w, int depth = 0);
bool suffixCandidate(const std::string& w, int depth, std::vector<uint8_t>& out);

// The candidate pronunciation the stored edits apply to: a known prefix
// (tools/english_build.py learns which ones combine reliably) before a
// stored stem reuses the stem's pronunciation ("undoing" = un + doing; by
// the letter rules alone it was "un-doy-ing"); otherwise the letter rules.
std::vector<uint8_t> candidate(const std::string& w, int depth = 0) {
  if (depth < 2) {
    for (size_t p = 0; p < kEnglishPrefixCount; ++p) {
      const std::string pre = kEnglishPrefixes[p][0];
      if (pre.empty() || w.size() < pre.size() + kMinStem || w.compare(0, pre.size(), pre) != 0)
        continue;
      const std::string stem = w.substr(pre.size());
      if (exceptions().find(stem) == exceptions().end()) continue;  // only real, stored stems
      std::vector<uint8_t> out;
      std::string tok;
      for (const char* s = kEnglishPrefixes[p][1];; ++s) {
        if (*s == ' ' || *s == 0) {
          if (!tok.empty()) out.push_back(tokenCode(tok));
          tok.clear();
          if (*s == 0) break;
        } else {
          tok += *s;
        }
      }
      const std::vector<uint8_t> st = pronounce(stem, depth + 1);
      out.insert(out.end(), st.begin(), st.end());
      fixStress(out);
      return out;
    }
  }
  if (depth < 2) {
    std::vector<uint8_t> out;
    if (suffixCandidate(w, depth, out)) return out;
  }
  return ruleTokens(w);
}

// The last consonant class of a pronunciation, for -s and -ed.
bool endsVoiceless(const std::vector<uint8_t>& t) {
  if (t.empty()) return false;
  const char* s = kEnglishTokens[t.back() - 0x80];
  return !strcmp(s, "P") || !strcmp(s, "T") || !strcmp(s, "K") || !strcmp(s, "F") || !strcmp(s, "TH") ||
         !strcmp(s, "S") || !strcmp(s, "SH") || !strcmp(s, "CH");
}
bool endsSibilant(const std::vector<uint8_t>& t) {
  if (t.empty()) return false;
  const char* s = kEnglishTokens[t.back() - 0x80];
  return !strcmp(s, "S") || !strcmp(s, "Z") || !strcmp(s, "SH") || !strcmp(s, "ZH") || !strcmp(s, "CH") ||
         !strcmp(s, "JH");
}
bool endsAlveolarStop(const std::vector<uint8_t>& t) {
  if (t.empty()) return false;
  const char* s = kEnglishTokens[t.back() - 0x80];
  return !strcmp(s, "T") || !strcmp(s, "D");
}

void appendArpa(std::vector<uint8_t>& out, const char* arpa) {
  std::string tok;
  for (const char* s = arpa;; ++s) {
    if (*s == ' ' || *s == 0) {
      if (!tok.empty()) out.push_back(tokenCode(tok));
      tok.clear();
      if (*s == 0) break;
    } else {
      tok += *s;
    }
  }
}

// A stored stem with a common suffix (tools/english_build.py keeps the
// suffixes that combine reliably): the stem's pronunciation and the suffix
// by rule (-s: s / z / ɪz, -ed: t / d / ɪd by the stem's last sound).
// Spelling: hop(e)+ing, run(n)+ing, cit(y->i)+es.
bool suffixCandidate(const std::string& w, int depth, std::vector<uint8_t>& out) {
  for (size_t k = 0; k < kEnglishSuffixCount; ++k) {
    const std::string suf = kEnglishSuffixes[k];
    if (suf.empty() || w.size() < suf.size() + kMinStem || w.compare(w.size() - suf.size(), suf.size(), suf) != 0)
      continue;
    const std::string base = w.substr(0, w.size() - suf.size());
    std::string stems[4] = {base, "", "", ""};
    const bool vowelSuffix = strchr("aeiou", suf[0]) != nullptr;
    if (vowelSuffix) stems[1] = base + "e";
    if (base.size() >= 2 && base.back() == base[base.size() - 2] && vowelSuffix)
      stems[2] = base.substr(0, base.size() - 1);
    if (!base.empty() && base.back() == 'i') stems[3] = base.substr(0, base.size() - 1) + "y";
    for (const std::string& st : stems) {
      if (st.size() < kMinStem || exceptions().find(st) == exceptions().end()) continue;
      out = pronounce(st, depth + 1);
      if (suf == "s" || suf == "es")
        appendArpa(out, endsSibilant(out) ? "IH0 Z" : endsVoiceless(out) ? "S" : "Z");
      else if (suf == "ed")
        appendArpa(out, endsAlveolarStop(out) ? "IH0 D" : endsVoiceless(out) ? "T" : "D");
      else
        appendArpa(out, kEnglishSuffixes[k] + suf.size() + 1);  // "ing\0IH0 NG": the ARPAbet after it
      fixStress(out);
      return true;
    }
  }
  return false;
}

std::vector<uint8_t> pronounce(const std::string& w, int depth) {
  std::vector<uint8_t> cand = candidate(w, depth);
  auto it = exceptions().find(w);
  return it == exceptions().end() ? cand : applyEdit(cand, it->second);
}

std::vector<std::string> names(const std::vector<uint8_t>& t) {
  std::vector<std::string> out;
  for (uint8_t c : t) out.push_back(kEnglishTokens[c - 0x80]);
  return out;
}

// One English sound as puhuja phonemes.
struct Sound {
  Ph ph[3];
  int n;
  bool longVowel;     // the first vowel is long (ː)
  bool backOffglide = false;  // aʊ, oʊ: the glide ends back and rounded
  const float* start = nullptr;  // own formants for the first vowel (aʊ, ɔɪ)
};

// End of the aʊ and oʊ glides: back and rounded, kept out of the tuning (the
// tuned ʊ of "good" is fronted, and a fronted glide makes "bow" sound "bli").
// F3 2700 (was 2400): goog.wav keeps F3 level through oʊ and l ("holds":
// ~3100-3200 female); at 2400 it sagged in the glide and jumped 400 Hz up at the l.
constexpr float kBackGlide[3] = {420, 1050, 2700};  // (380 900: too far back, voice-over 1030-1390)
constexpr float kAwGlide[3] = {450, 1300, 2400};  // the end of aʊ
constexpr float kGlideFallShare = 0.5f;  // a final diphthong's fall done by its glide's centre
// Start of ɔɪ ("boy"): open, back and rounded, so the two vowels stay distinct
// (a close start sounds like "bees").
constexpr float kOpenBackStart[3] = {560, 850, 2500};
// Start of aʊ ("now", "how"): open and central; the tuned English a comes
// from aɪ ("I", "my") and is too front, which makes "now" sound like "neh-oo".
constexpr float kOpenCentralStart[3] = {730, 1400, 2500};
// oʊ starts mid and a little central (Hillenbrand o: F1 ~500), so it glides
// clearly into ʊ: from the tuned o (F1 400, F2 1040), next to the glide's
// end, "holds" sounded like "hoolds".
// oʊ starts a little more open and back (goog.wav "holds": F1 falls steadily
// from ~800 female; from 520 F1 sat flat and then dropped in steps).
constexpr float kMidBackStart[3] = {560, 1100, 2650};

Sound soundOf(const char* tok) {
  std::string t(tok);
  const char st = t.back();
  const bool vowel = st >= '0' && st <= '2';
  const bool stressed = vowel && st != '0';
  if (vowel) t.pop_back();
  if (t == "AA") return {{Ph::AA}, 1, stressed};
  if (t == "AE") return {{Ph::AE}, 1, false};
  if (t == "AH") return {{stressed ? Ph::AH : Ph::Schwa}, 1, false};
  if (t == "AO") return {{Ph::AO}, 1, stressed};
  if (t == "AW") return {{Ph::A, Ph::UH}, 2, false, true, kOpenCentralStart};
  if (t == "AY") return {{Ph::A, Ph::IH}, 2, false};
  if (t == "EH") return {{Ph::EH}, 1, false};
  // Unstressed too: the r colour (low F2) starts in the vowel. With the
  // English ə (F2 ~1800) "power" glided down, up and down again (pow-yer).
  if (t == "ER") return {{Ph::ER, Ph::RR}, 2, stressed};
  if (t == "EY") return {{Ph::E, Ph::IH}, 2, false};
  if (t == "IH") return {{Ph::IH}, 1, false};
  if (t == "IY") return {{Ph::I}, 1, stressed};
  if (t == "OW") return {{Ph::O, Ph::UH}, 2, false, true, kMidBackStart};
  if (t == "OY") return {{Ph::AO, Ph::IH}, 2, false, false, kOpenBackStart};
  if (t == "UH") return {{Ph::UH}, 1, false};
  if (t == "UW") return {{Ph::U}, 1, stressed};
  static const struct {
    const char* name;
    Ph ph;
  } kCons[] = {{"B", Ph::B},   {"CH", Ph::CH}, {"D", Ph::D},   {"DH", Ph::DH}, {"F", Ph::F},
               {"G", Ph::G},   {"HH", Ph::H},  {"JH", Ph::JH}, {"K", Ph::K},   {"L", Ph::L},
               {"M", Ph::M},   {"N", Ph::N},   {"NG", Ph::NG}, {"P", Ph::P},   {"R", Ph::RR},
               {"S", Ph::S},   {"SH", Ph::SH}, {"T", Ph::T},   {"TH", Ph::TH}, {"V", Ph::VV},
               {"W", Ph::W},   {"Y", Ph::J},   {"Z", Ph::Z},   {"ZH", Ph::ZH}};
  for (const auto& c : kCons)
    if (t == c.name) return {{c.ph}, 1, false};
  return {{Ph::Stop}, 0, false};
}

// Consonant pairs that can start an English syllable (for placing ˈ).
bool onsetPair(const char* a, const char* b) {
  const std::string x(a), y(b);
  if (x == "S" && (y == "P" || y == "T" || y == "K" || y == "M" || y == "N" || y == "L" ||
                   y == "W"))
    return true;
  const bool obstruent = x == "P" || x == "B" || x == "T" || x == "D" || x == "K" ||
                         x == "G" || x == "F" || x == "TH" || x == "SH";
  return obstruent && (y == "L" || y == "R" || y == "W" || y == "Y");
}

const char* const kOnes[] = {"zero",    "one",     "two",       "three",    "four",
                             "five",    "six",     "seven",     "eight",    "nine",
                             "ten",     "eleven",  "twelve",    "thirteen", "fourteen",
                             "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
const char* const kTens[] = {"",      "",      "twenty",  "thirty", "forty",
                             "fifty", "sixty", "seventy", "eighty", "ninety"};

void spellBelow1000(unsigned n, std::vector<std::string>& out) {
  if (n >= 100) {
    out.push_back(kOnes[n / 100]);
    out.push_back("hundred");
    n %= 100;
    if (n) out.push_back("and");
  }
  if (n >= 20) {
    out.push_back(kTens[n / 10]);
    if (n % 10) out.push_back(kOnes[n % 10]);
  } else if (n || out.empty()) {
    out.push_back(kOnes[n]);
  }
}

}  // namespace

// The candidate (what the stored edits apply to): tools/english_build.py
// runs this through -g2p-raw.
std::vector<std::string> englishRules(std::string_view word) {
  return names(candidate(lower(word)));
}

std::vector<std::string> englishPronounce(std::string_view word) {
  return names(pronounce(lower(word)));
}

// Hiss of fricatives and bursts: an English voice-over reference
// (Voices.com demo reel) has sibilant peaks ~13 dB stronger relative to its
// vowels than puhuja's Piper-calibrated (Finnish) levels.
// The rate the timing was fitted at (main.cc's default -rate).
constexpr float kDefaultRate = 0.95f;
// Longest unfinished word held back between push() calls.
constexpr size_t kMaxHeldBack = 256;
constexpr float kEnglishFricationGain = 4.5f;
constexpr float kEnglishAspirationGain = 50.0f, kEnglishHGain = 20.0f;
constexpr float kEnglishVoicedHNoise = 0.3f;  // x the h gain, h between voiced sounds (3: in "the hand" ~12 dB louder than goog.wav)
constexpr float kEnglishVoicedHAg = 0.13f;  // cm^2, h between voiced sounds (goog.wav's stays voiced)

EnglishReader::EnglishReader(Synth& synth) : synth_(synth) {
  synth_.setSpellingRules(false);  // English phonemes come ready-made
  synth_.setFricationGain(kEnglishFricationGain);
  synth_.setDarkL(true);
  // (Formant ranges, setFormantRanges(true): even narrow F2/F3 bands cost
  // intelligibility, 13.5% -> 18.2% in the round trip; off.)
  synth_.setAspiration(true);
  // English h and aspiration are breathier than Finnish h. With the default
  // gain a phrase-initial h was 22 dB under its vowel (0.5-4 kHz; real ~12)
  // and "He" was heard as "They" (tools/roundtrip.py); aspiration x4 puts
  // the puff ~12 dB under the vowel.
  synth_.setAspirationGain(kEnglishAspirationGain, kEnglishHGain);
  // English h between voiced sounds ("mortal hands") is noisier than the
  // Finnish half-voiced h: its breath noise was hidden under the voicing.
  synth_.setVoicedHNoise(kEnglishVoicedHNoise, kEnglishVoicedHAg);
  synth_.setHTakesVowel(true);
}

namespace {

// Words that are normally unaccented in running speech.
bool isFunctionWord(const std::string& w) {
  static const char* const kWords[] = {
      "a", "an", "the", "and", "or", "but", "nor", "if", "of", "to", "in", "on", "at", "by",
      "for", "with", "from", "as", "into", "onto", "than", "that", "this", "these", "those",
      "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "have", "has", "had",
      "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
      "your", "his", "its", "our", "their", "can", "could", "will", "would", "shall", "should",
      "may", "might", "must", "so", "there", "then", "just", "i'm", "it's", "you're", "we're",
      "they're", "i'll", "i've", "let's", "any",
      // relative and question words starting a clause, long prepositions,
      // pronoun contractions (negatives like don't stay accented)
      "who", "whom", "whose", "which", "where", "when", "what", "while", "because", "about",
      "through", "over", "being", "having", "that's", "there's", "he's", "she's", "we'll",
      "they'll", "you'll", "he'll", "she'll", "we've", "they've", "you've", "i'd", "you'd",
      "he'd", "she'd", "we'd", "they'd"};
  for (const char* f : kWords)
    if (w == f) return true;
  return false;
}

// Intonation, in semitones from the base pitch.
// Baseline at the phrase start, below the voice's base pitch: accents rise
// above it and endings fall below, so the speaking pitch centres on the
// base (at +1.9 every variation went up: a 92 Hz voice spoke at ~126 Hz; now ~95).
constexpr float kPhraseStart = -4.5f;
constexpr float kDeclination = -0.3f;    // baseline change per syllable,
constexpr float kDeclinationMin = -2.0f, kDeclinationMax = -6.5f;  // in total over a phrase
constexpr float kParagraphStartLift = 0.67f;  // first sentence of a paragraph (st)
constexpr float kSentenceStep = 0.33f, kSentenceStepMax = 0.67f;  // later sentences: lower
constexpr float kAfterCommaStart = -1.0f;     // a phrase after a comma starts lower
constexpr float kPhraseVariation = 0.5f;     // +- deterministic variation of accent height
// Accent sizes by meaning (see sayPhrase).
const char* const kEmphatic[] = {"every", "only", "never", "all", "not", "no", "nothing", "none",
                                 "always", "first", "most", "very", "must", "don't", "can't", "cannot"};
constexpr float kEmphasisBoost = 2.0f;   // st
constexpr float kQuoteLift = 1.0f;       // st: quoted speech
constexpr float kEmphaticFloor = 0.8f;   // share of the first accent's height emphasis words reach
constexpr float kLetScale = 0.8f;        // phrase-initial imperative "let"
constexpr float kGivenScale = 0.7f;      // a word said within the last kRecentWords content words
constexpr float kPrenuclearScale = 0.8f; // first of two adjacent accented words
constexpr float kNuclearMin = 8.0f;      // the last accent is at least this high
constexpr float kEndingVariety = 0.7f;  // st: final lows vary by -1, 0, +1 times this
constexpr size_t kShortClosingSyllables = 6;  // a sentence's short closing phrase ...
constexpr float kClosingNucleusBelow = -0.5f;  // ... has its nucleus at most this far under the highest accent
constexpr float kNucleusDecline = -3.0f;  // st: the nucleus's baseline declines at most this much
constexpr size_t kRecentWords = 40;
constexpr float kRepeatLift = 1.5f, kRepeatLiftMax = 4.5f;  // st per repetition of a phrase
constexpr size_t kRecentPhrases = 12;
constexpr float kExclaimLift = 3.0f;  // st on the accents of an exclamation
constexpr float kExclaimDb = 2.0f;    // and louder
constexpr float kFirstAccent = 11.0f;      // first accent above the baseline
constexpr float kDownstep = 0.85f;         // each further accent: this much of the last
constexpr float kMinAccent = 5.5f;
constexpr float kPostAccent = 0.2f;       // syllable after an accent keeps this share
// Unaccented syllables between two accents: a single quick one sits on the
// line, a longer run sags into a valley (st per syllable of the run, capped).
constexpr float kInterAccentSagPerSyl = 2.0f, kInterAccentSagMax = 5.0f;
constexpr float kSecondary = 1.0f;        // secondary stress in a content word
constexpr float kFinalLow = -7.0f;        // end of a long or exclaimed statement
constexpr float kFinalLowShort = -5.0f;   // end of a short statement
constexpr float kLevelEnding = -5.0f;     // a level ending (near the baseline)
constexpr unsigned kLevelEndingEvery = 5;
constexpr size_t kLongSentenceSyllables = 12;
constexpr float kQuestionHigh = 6.0f;     // end of a question
constexpr float kContinuation = -3.0f;    // end before a comma (a little above the baseline)
// Pauses in 15 ms stops: about 200 ms after a comma (longer when the speaker
// breathes there); after a sentence 120 ms plus the breath (>= 180 ms).
constexpr int kCommaPauseStops = 13;     // 195 ms after a phrase of 5 syllables, growing:
constexpr int kCommaPauseMaxStops = 24;  // at most 360 ms
constexpr int kShortPhraseSyllables = 4; // a phrase this short gets the short pause:
constexpr int kShortCommaPauseStops = 8; // 120 ms
constexpr int kClausePauseStops = 18;    // after ; and : at least 270 ms
constexpr float kPauseVariation = 0.15f; // +- share
constexpr float kFinalLengthSentence = 1.4f, kFinalLengthPhrase = 1.15f;  // final lengthening
constexpr float kFinalLengthClause = 1.25f;  // before ; or :
constexpr float kFinalCodaSonorant = 3.5f;  // a phrase-final l m n ŋ coda
constexpr int kSentencePauseStops = 10;  // + the breath before the next sentence
constexpr int kDashPauseStops = 8;        // parenthesis: ~120 ms
constexpr int kThinkingPauseStops = 20;   // en/em dash: a pause for thought, ~300 ms (400 outranked sentence ends)
constexpr int kQuotePauseStops = 5;       // before a quotation: ~75 ms
constexpr float kSpeechTagDrop = -2.0f, kSpeechTagAccent = 0.4f;  // a speech tag: lower start (st), accents x
constexpr int kParagraphPauseStops = 25;  // empty line: ~375 ms more

}  // namespace

namespace {

// Segment durations after Klatt (1979): inherent and minimum duration (ms)
// per sound; the rules below scale the part above the minimum.
struct KlattDur {
  const char* name;
  float inh, min;
};
constexpr float kHBeforeJ = 2.5f;
constexpr KlattDur kKlatt[] = {
    {"AA", 240, 100}, {"AE", 230, 80}, {"AH", 140, 60}, {"AO", 240, 100}, {"AW", 260, 100},
    {"AY", 250, 150}, {"EH", 150, 70}, {"ER", 180, 80}, {"EY", 190, 100}, {"IH", 135, 40},
    {"IY", 155, 55},  {"OW", 220, 80}, {"OY", 280, 150}, {"UH", 160, 60}, {"UW", 210, 70},
    {"B", 85, 60},    {"CH", 70, 50},  {"D", 75, 50},   {"DH", 70, 45},  {"F", 100, 80},
    {"G", 80, 60},    {"HH", 80, 20},  {"JH", 70, 50},  {"K", 80, 60},   {"L", 80, 40},
    {"M", 70, 60},    {"N", 60, 50},   {"NG", 95, 60},  {"P", 90, 50},   {"R", 80, 30},
    {"S", 105, 60},   {"SH", 105, 80}, {"T", 75, 50},   {"TH", 90, 60},  {"V", 60, 40},
    {"W", 80, 60},    {"Y", 80, 40},   {"Z", 75, 40},   {"ZH", 70, 40}};

std::string baseName(uint8_t t) {
  std::string s = kEnglishTokens[t - 0x80];
  if (!s.empty() && s.back() >= '0' && s.back() <= '2') s.pop_back();
  return s;
}

const KlattDur& klatt(const std::string& base) {
  for (const auto& k : kKlatt)
    if (base == k.name) return k;
  return kKlatt[2];  // AH
}

bool isVoicelessPlosive(const std::string& b) { return b == "P" || b == "T" || b == "K" || b == "CH"; }
bool isVoicedPlosive(const std::string& b) { return b == "B" || b == "D" || b == "G" || b == "JH"; }
bool isVoicedFricative(const std::string& b) { return b == "V" || b == "DH" || b == "Z" || b == "ZH"; }
bool isNasal(const std::string& b) { return b == "M" || b == "N" || b == "NG"; }

}  // namespace

// Duration (ms) of every token of the phrase, by Klatt's rules (simplified).
// Tempo by phrase structure (duration multipliers).
constexpr int kFastPhraseSyllables = 15, kShortPhraseTempoSyllables = 6;
constexpr float kLongPhraseTempo = 0.94f, kShortPhraseTempo = 1.06f, kParagraphOpeningTempo = 1.06f;

// Stress timing (on top of Klatt's rules).
constexpr float kUnstressedVowel = 0.42f;      // Klatt: 0.5
constexpr float kAccentedVowelTiming = 1.2f, kUnstressedVowelTiming = 0.85f;  // stress timing
constexpr float kStressedOnset = 1.2f;         // consonants starting a stressed syllable
constexpr float kUnstressedConsonant = 0.85f;  // other consonants

std::vector<std::vector<float>> EnglishReader::durations() const {
  std::vector<std::vector<float>> out(phrase_.size());
  // Tempo by structure: long phrases run a little faster, short ones and
  // the opening of a paragraph a little slower.
  const int syllables = phraseSyllables();
  float structure = syllables > kFastPhraseSyllables ? kLongPhraseTempo
                    : syllables <= kShortPhraseTempoSyllables ? kShortPhraseTempo : 1.0f;
  if (sentenceInParagraph_ == 0 && phraseInSentence_ == 0) structure *= kParagraphOpeningTempo;
  // Last vowel of the phrase: its syllable is phrase-final.
  int lastWordWithVowel = -1;
  for (size_t w = 0; w < phrase_.size(); ++w)
    for (uint8_t t : phrase_[w].tokens)
      if (isVowelToken(t)) lastWordWithVowel = static_cast<int>(w);
  for (size_t w = 0; w < phrase_.size(); ++w) {
    const auto& tok = phrase_[w].tokens;
    int vowels = 0, lastVowel = -1;
    for (size_t i = 0; i < tok.size(); ++i)
      if (isVowelToken(tok[i])) {
        ++vowels;
        lastVowel = static_cast<int>(i);
      }
    out[w].resize(tok.size());
    for (size_t i = 0; i < tok.size(); ++i) {
      const std::string b = baseName(tok[i]);
      const KlattDur& k = klatt(b);
      const bool vowel = isVowelToken(tok[i]);
      const bool inLastSyllable = static_cast<int>(i) >= lastVowel;
      const bool phraseFinal = static_cast<int>(w) == lastWordWithVowel && inLastSyllable;
      float prc = 1, mn = k.min;
      if (vowel) {
        const char* name = kEnglishTokens[tok[i] - 0x80];
        // A function word ending the phrase is said in full ("little we can
        // do;", "yes, I can."): its stressed syllable is not shortened.
        const bool stressed = (phrase_[w].accent || phraseFinal) && name[strlen(name) - 1] != '0';
        if (!phraseFinal) prc *= 0.6f;
        if (!inLastSyllable) prc *= 0.85f;
        if (vowels > 1) prc *= 0.8f;
        if (!stressed) {
          prc *= kUnstressedVowel;
          mn *= kUnstressedVowel;
        }
        // Stress timing: accented vowels longer, unstressed ones shorter (the
        // syllables were too even: nPVI 61, the voice-overs about 76).
        if (phrase_[w].accent && name[strlen(name) - 1] == '1') prc *= kAccentedVowelTiming;
        else if (!stressed) prc *= kUnstressedVowelTiming;
        // The consonant after the vowel (same word) colours its length.
        if (i + 1 < tok.size() && !isVowelToken(tok[i + 1])) {
          const std::string n = baseName(tok[i + 1]);
          prc *= isVoicedFricative(n) ? 1.4f : isVoicedPlosive(n) ? 1.2f
               : isNasal(n) ? 0.9f : isVoicelessPlosive(n) ? 0.75f : 1.0f;
        }
      } else {
        if (i > 0) prc *= 0.85f;  // not word-initial
        const bool consBefore = i > 0 && !isVowelToken(tok[i - 1]);
        const bool consAfter = i + 1 < tok.size() && !isVowelToken(tok[i + 1]);
        if (consBefore) prc *= 0.7f;
        if (consAfter) prc *= 0.7f;
        // Stress timing: the consonants that start a stressed syllable are
        // longer, those of unstressed syllables shorter.
        size_t v = i + 1;
        while (v < tok.size() && !isVowelToken(tok[v])) ++v;
        const char* vn = v < tok.size() ? kEnglishTokens[tok[v] - 0x80] : nullptr;
        const bool onsetOfStressed = vn && phrase_[w].accent && vn[strlen(vn) - 1] == '1';
        prc *= onsetOfStressed ? kStressedOnset : kUnstressedConsonant;
        // h before j is one sound, the palatal hiss [ç] ("human"): not
        // shortened as a cluster, and long (goog.wav ~120 ms; here it was 51).
        if (b == "HH" && i + 1 < tok.size() && baseName(tok[i + 1]) == "Y") prc *= kHBeforeJ;
      }
      // Final lengthening by the strength of the boundary.
      if (phraseFinal)
        prc *= sentenceEnd_ ? kFinalLengthSentence
               : lastEnd_ == ';' || lastEnd_ == ':' ? kFinalLengthClause : kFinalLengthPhrase;
      // A final l, m, n or ŋ before the pause is held: the voice carries on
      // through it (goog.wav "well or ill,": the l ~150 ms, here it was 68).
      if (phraseFinal && !vowel && i > static_cast<size_t>(lastVowel) && (b == "L" || isNasal(b)))
        prc *= kFinalCodaSonorant;
      // Rate: vowels follow it, consonants less (a slow reader keeps the
      // consonants crisp); relative to the default rate, which stays as fitted.
      const float r = rate_ / kDefaultRate;
      const float rateDiv = kDefaultRate * (vowel ? r : std::sqrt(r));
      out[w][i] = (mn + (k.inh - mn) * prc) * kEnglishTempo * structure / rateDiv;
    }
  }
  return out;
}

// Splitting long phrases (see addWord).
// Planned lung volume (cm^3) below which a breath is taken between words.
constexpr float kEmergencyBreathAir = 450;
// ... and at a seam (a split before "and", a relative word, a preposition)
// below this: a natural place to breathe, before the air would run out in
// the middle of the next phrase (~3 s of speech).
constexpr float kSeamBreathAir = 900;
constexpr int kSplitSyllables = 22;  // (12, 16: orators hold longer phrases)
constexpr size_t kForceSplitWords = 25;
constexpr int kLightBreakStops = 4;  // ~60 ms
constexpr int kSymmetryPauseStops = 10;  // ~150 ms before the "and" of parallel halves

// Homographs read as verbs after a subject, a modal or "to" (the dictionary
// gives the other reading first): "who live" /lɪv/, "as I close" /z/.
struct Homograph {
  const char* word;
  const char* verb;  // ARPAbet of the verb reading
};
const Homograph kHomographs[] = {
    {"live", "L IH1 V"}, {"close", "K L OW1 Z"}, {"closes", "K L OW1 Z IH0 Z"},
    {"read", "R IY1 D"}, {"lead", "L IY1 D"}, {"use", "Y UW1 Z"}, {"wind", "W AY1 N D"},
    {"tear", "T EH1 R"}, {"excuse", "IH0 K S K Y UW1 Z"}, {"house", "HH AW1 Z"},
    {"abuse", "AH0 B Y UW1 Z"}, {"refuse", "R IH0 F Y UW1 Z"}};
const char* const kVerbCues[] = {
    "i", "you", "we", "they", "who", "to", "will", "would", "can", "could", "may", "might",
    "must", "shall", "should", "do", "don't", "did", "didn't", "not", "people", "men", "let's"};
// "read" after these is the past tense /rɛd/ (the dictionary's first reading).
const char* const kPastCues[] = {"have", "has", "had", "was", "were", "been", "be", "is", "are"};

std::vector<uint8_t> arpaTokens(const char* arpa);

// Names of the letters, for spelling.
const char* const kLetterNames[26] = {
    "EY1", "B IY1", "S IY1", "D IY1", "IY1", "EH1 F", "JH IY1", "EY1 CH", "AY1", "JH EY1",
    "K EY1", "EH1 L", "EH1 M", "EH1 N", "OW1", "P IY1", "K Y UW1", "AA1 R", "EH1 S", "T IY1",
    "Y UW1", "V IY1", "D AH1 B AH0 L Y UW0", "EH1 K S", "W AY1", "Z IY1"};

void EnglishReader::addWord(const std::string& raw) {
  size_t a = raw.find_first_not_of('\''), b = raw.find_last_not_of('\'');
  if (a == std::string::npos) return;
  const std::string w = raw.substr(a, b - a + 1);
  std::vector<uint8_t> t = pronounce(w);
  // Long phrases are split at a seam (before a conjunction, a relative word
  // or a preposition) once they have kSplitSyllables syllables, and at
  // kForceSplitWords words anyway: a light break with a continuation pitch.
  {
    // (Not "to", "for", "or": "intends | to remain" broke what belongs together.)
    static const char* const kSeams[] = {"and", "but", "so", "because", "that", "which",
                                         "who", "when", "where", "if", "while", "in", "on",
                                         "with", "from", "as", "than", "like"};
    int syllables = 0;
    for (const Word& pw : phrase_)
      for (uint8_t tok : pw.tokens) syllables += isVowelToken(tok);
    bool seam = false;
    for (const char* s : kSeams) seam = seam || w == s;
    // Never right after a function word ("the heirs of | that first").
    static const char* const kNoSplitAfter[] = {"of", "the", "a", "an", "to", "in", "for", "my",
                                                "your", "our", "their", "his", "her", "its", "and"};
    const std::string& before = phrase_.empty() ? std::string() : phrase_.back().text;
    for (const char* s : kNoSplitAfter) seam = seam && before != s;
    // A split before "and" / "or" waits for three more words: if the halves
    // are parallel it becomes the longer symmetry break (it won over it).
    if (deferredSplit_ > 0 && phrase_.size() >= deferredSplit_ + 3) {
      const size_t j = deferredSplit_;
      deferredSplit_ = 0;
      const bool sym = symmetricAt(j);
      std::vector<Word> tail(phrase_.begin() + static_cast<long>(j), phrase_.end());
      phrase_.resize(j);
      sayPhrase(',');
      pause(sym ? kSymmetryPauseStops : kLightBreakStops);
      if (synth_.plannedAir() < kSeamBreathAir) synth_.breathPoint(false);
      phrase_ = std::move(tail);
      syllables = 0;
      for (const Word& pw : phrase_)
        for (uint8_t tok : pw.tokens) syllables += isVowelToken(tok);
    }
    const bool andOr = w == "and" || w == "or";
    // A contrast ("come not from the generosity of the state | but from the
    // hand of God"): a light break before the "but" that answers a "not",
    // whatever the length. (The references pause before an unpunctuated
    // "but" only when it is written with a comma; this is the rhetorical
    // reading the user asked for.)
    bool contrast = false;
    size_t negation = 0;  // the "not" the "but" answers (the last one)
    if (w == "but" && seam && phrase_.size() >= 4)
      for (size_t i = 0; i < phrase_.size(); ++i)
        if (phrase_[i].text == "not" || phrase_[i].text == "never" || phrase_[i].text == "neither") {
          contrast = true;
          negation = i;
        }
    if (andOr && deferredSplit_ == 0 && seam && syllables >= kSplitSyllables && phrase_.size() >= 4) {
      deferredSplit_ = phrase_.size();
    } else if (deferredSplit_ == 0 && ((seam && syllables >= kSplitSyllables && phrase_.size() >= 4) ||
                                       contrast || phrase_.size() >= kForceSplitWords)) {
      // ... and the "not X" of "not X but Y" is set off before as well
      // ("come | not from the generosity of the state | but ..."), unless
      // it opens the phrase.
      if (contrast && negation >= 2) {
        std::vector<Word> rest(phrase_.begin() + static_cast<long>(negation), phrase_.end());
        phrase_.resize(negation);
        sayPhrase(',');
        pause(kLightBreakStops);
        phrase_ = std::move(rest);
      }
      sayPhrase(',');
      pause(kLightBreakStops);
      // A breath here only when the air is getting low: they belong at the
      // punctuation ("the survival | (breath) and the success").
      // (not at a contrast: the user asked for a small pause there)
      if (!contrast && synth_.plannedAir() < kSeamBreathAir) synth_.breathPoint(false);
    }
  }
  const std::string prev = phrase_.empty() ? lastWord_ : phrase_.back().text;
  for (const Homograph& h : kHomographs) {
    if (w != h.word) continue;
    bool verb = false;
    for (const char* c : kVerbCues) verb = verb || prev == c;
    if (w == "read")
      for (const char* c : kPastCues) verb = verb && prev != c;
    if (verb) t = arpaTokens(h.verb);
    // "wind": CMUdict's first reading is the verb; without a verb cue it is
    // the noun ("the wind blew").
    else if (w == "wind") t = arpaTokens("W IH1 N D");
  }
  lastWord_ = w;
  // Spelled letter by letter: words without a vowel ("bg", "zh") and short
  // all-capital words that are not in the dictionary (acronyms: "NSA").
  bool vowel = false;
  for (uint8_t tok : t) vowel = vowel || isVowelToken(tok);
  // (Not when it reads as a word: at least 4 letters, vowels and
  // consonants alternating in runs of at most two, "COVID".)
  bool pronounceable = w.size() >= 4;
  {
    int run = 0;
    bool lastV = false, anyV = false;
    for (size_t k = 0; k < w.size(); ++k) {
      const bool v = strchr("aeiou", w[k]) != nullptr;
      run = k > 0 && v == lastV ? run + 1 : 1;
      lastV = v;
      anyV = anyV || v;
      if (run > 2) pronounceable = false;
    }
    pronounceable = pronounceable && anyV;
  }
  const bool acronym = capsWord_ && w.size() >= 2 && w.size() <= 5 && !pronounceable &&
                       exceptions().find(w) == exceptions().end();
  if ((!vowel && w.size() <= 6) || acronym) {
    for (size_t k = 0; k < w.size(); ++k)
      if (w[k] >= 'a' && w[k] <= 'z')
        phrase_.push_back({arpaTokens(kLetterNames[w[k] - 'a']), k + 1 == w.size(), std::string(1, w[k])});
    return;
  }
  if (!t.empty()) phrase_.push_back({std::move(t), !isFunctionWord(w), w});
}

// Longest run of digits read one by one.
constexpr size_t kMaxDigits = 20;

// A word and its replacement (abbreviations, ordinals).
struct Abbreviation {
  const char* abbr;
  const char* full;
  bool bare = false;  // read in full even without a period ("mr smith")
};

void EnglishReader::addNumber(const std::string& digits) {
  std::vector<std::string> words;
  const bool point = pointDigits_;  // digits after a decimal point
  pointDigits_ = false;
  if (centsNext_) {  // $3.50: the cents ("fifty cents", "five cents")
    centsNext_ = false;
    const unsigned n = std::stoul(digits);
    spellBelow1000(n, words);
    for (const auto& w : words) addWord(w);
    if (centsPence_) addWord(n == 1 ? "penny" : "pence");
    else addWord(n == 1 ? "cent" : "cents");
    centsPence_ = false;
    numberJustSaid_ = false;
    return;
  }
  if (fractionNext_) {  // the denominator: "half", "quarters", "fifths"
    fractionNext_ = false;
    const unsigned d = std::stoul(digits.substr(0, 9));
    const bool many = numerator_ != 1;
    std::string w = d == 2 ? (many ? "halves" : "half") : d == 4 ? (many ? "quarters" : "quarter") : "";
    if (w.empty()) {
      ordinalNext_ = true;
      fractionPlural_ = many;
    } else {
      addWord(w);
      return;
    }
  }
  const unsigned year = digits.size() == 4 && digits[0] != '0' ? std::stoul(digits) : 0;
  if (point || digits.size() > 12 || (digits.size() > 1 && digits[0] == '0')) {
    // digit by digit, at most kMaxDigits of them
    for (size_t k = 0; k < digits.size() && k < kMaxDigits; ++k) words.push_back(kOnes[digits[k] - '0']);
  } else if (!ordinalNext_ && ((year >= 1100 && year <= 1999) || (year >= 2010 && year <= 2099))) {
    // years in pairs: 1963 "nineteen sixty three", 1900 "nineteen hundred",
    // 1905 "nineteen oh five"
    spellBelow1000(year / 100, words);
    const unsigned lo = year % 100;
    if (lo == 0) {
      words.push_back("hundred");
    } else {
      if (lo < 10) words.push_back("oh");
      spellBelow1000(lo, words);
    }
  } else {
    const unsigned long long n = std::stoull(digits);
    static const char* const kScale[] = {"", "thousand", "million", "billion"};
    unsigned long long div = 1000000000ull;
    for (int g = 3; g >= 0; --g, div /= 1000) {
      const unsigned part = static_cast<unsigned>((n / div) % 1000);
      if (part || (n == 0 && g == 0)) {
        spellBelow1000(part, words);
        if (g) words.push_back(kScale[g]);
      }
    }
  }
  if (ordinalNext_ && !words.empty()) {  // 21st: "twenty first"
    ordinalNext_ = false;
    std::string& w = words.back();
    static const Abbreviation kOrd[] = {{"one", "first"}, {"two", "second"}, {"three", "third"},
                                        {"five", "fifth"}, {"eight", "eighth"}, {"nine", "ninth"},
                                        {"twelve", "twelfth"}};
    bool done = false;
    for (const auto& o : kOrd)
      if (w == o.abbr) w = o.full, done = true;
    if (!done) w = w.back() == 'y' ? w.substr(0, w.size() - 1) + "ieth" : w + "th";
  }
  if (fractionPlural_ && !words.empty()) {  // 2/5: "two fifths"
    fractionPlural_ = false;
    words.back() += "s";
  }
  if (pluralNext_ && !words.empty()) {  // 1990s: "nineteen nineties"
    pluralNext_ = false;
    std::string& w = words.back();
    w = w.back() == 'y' ? w.substr(0, w.size() - 1) + "ies" : w + "s";
  }
  for (const auto& w : words) addWord(w);
  if (dollarNext_) {  // $ £ €
    static const char* const kCurrency[][2] = {{"dollar", "dollars"}, {"pound", "pounds"}, {"euro", "euros"}};
    addWord(kCurrency[dollarNext_ - 1][digits == "1" ? 0 : 1]);
    dollarNext_ = 0;
  }
  // For a unit or "am" / "pm" after the number ("5 km", "7 pm").
  numberJustSaid_ = true;
  numberWasOne_ = digits == "1" && !point;
}


// Token of an ARPAbet name ("AH0"), 0 if unknown.
uint8_t arpaToken(const char* name) {
  for (int t = 0; t < static_cast<int>(sizeof(kEnglishTokens) / sizeof(kEnglishTokens[0])); ++t)
    if (!strcmp(kEnglishTokens[t], name)) return static_cast<uint8_t>(0x80 + t);
  return 0;
}

// Foreign phrases quoted in English text (Kennedy's Berlin speech): a phrase
// with at least two words of one language is read with these
// pronunciations (approximated with English sounds: ich ~ ɪʃ, nach ~ nɑk).
struct ForeignWord {
  const char* word;
  const char* arpa;
};
const ForeignWord kGerman[] = {
    {"ich", "IH1 SH"}, {"bin", "B IH1 N"}, {"ein", "AY1 N"}, {"berliner", "B EH0 R L IY1 N ER0"},
    {"berlin", "B EH0 R L IY1 N"}, {"lasst", "L AA1 S T"}, {"lass", "L AA1 S"}, {"sie", "Z IY1"},
    {"nach", "N AA1 K"}, {"kommen", "K AO1 M AH0 N"}};
const ForeignWord kLatin[] = {
    {"civis", "K IY1 W IH0 S"}, {"romanus", "R OW0 M AA1 N UH0 S"}, {"sum", "S UH1 M"}};

std::vector<uint8_t> arpaTokens(const char* arpa) {
  std::vector<uint8_t> t;
  const std::string a = arpa;
  for (size_t p = 0; p < a.size();) {
    size_t q = a.find(' ', p);
    if (q == std::string::npos) q = a.size();
    if (uint8_t tok = arpaToken(a.substr(p, q - p).c_str())) t.push_back(tok);
    p = q + 1;
  }
  return t;
}

struct WeakForm {
  const char* word;
  const char* weak;
};

// The word after have/has/had makes it an auxiliary: a past participle
// (-ed, -en, or a common irregular one), "been", "got", "not" or an adverb
// standing between.
bool isAuxiliaryUse(const std::string& next) {
  static const char* const kAux[] = {
      "been", "got", "gotten", "not", "never", "always", "already", "just", "ever", "also", "only",
      "done", "gone", "seen", "made", "said", "come", "become", "run", "won", "told", "heard",
      "found", "left", "lost", "met", "paid", "put", "read", "sent", "spent", "stood", "thought",
      "brought", "bought", "caught", "taught", "held", "kept", "built", "felt", "had", "led",
      "meant", "set", "let", "cut", "hit", "known", "shown", "grown", "drawn", "flown", "thrown"};
  for (const char* a : kAux)
    if (next == a) return true;
  const size_t n = next.size();
  return n > 3 && (next.compare(n - 2, 2, "ed") == 0 || next.compare(n - 2, 2, "en") == 0);
}
const WeakForm kWeakForms[] = {
    {"to", "T AH0"}, {"for", "F ER0"}, {"and", "AH0 N D"}, {"can", "K AH0 N"},
    {"that", "DH AH0 T"}, {"from", "F R AH0 M"}, {"them", "DH AH0 M"}, {"was", "W AH0 Z"},
    {"at", "AH0 T"}, {"as", "AH0 Z"}, {"your", "Y ER0"}, {"of", "AH0 V"}, {"have", "HH AH0 V"},
    {"has", "HH AH0 Z"}, {"had", "HH AH0 D"}, {"are", "ER0"}, {"were", "W ER0"},
    {"but", "B AH0 T"}, {"or", "ER0"}, {"been", "B IH0 N"}, {"does", "D AH0 Z"},
    {"than", "DH AH0 N"}, {"could", "K UH0 D"}, {"would", "W UH0 D"}, {"should", "SH UH0 D"},
    {"just", "JH AH0 S T"}, {"us", "AH0 S"}, {"his", "IH0 Z"}, {"her", "ER0"}};

template <size_t N>
void applyForeign(std::vector<EnglishReader::Word>& phrase, const ForeignWord (&table)[N]) {
  int hits = 0;
  for (const auto& w : phrase)
    for (const auto& f : table) hits += w.text == f.word;
  if (hits < 2) return;
  for (auto& w : phrase)
    for (const auto& f : table)
      if (w.text == f.word) {
        w.tokens = arpaTokens(f.arpa);
        w.accent = true;
      }
}

// Loudness shape of a phrase (dB, vowels).
constexpr float kArcStartDb = 1.0f, kArcEndDb = -2.0f;  // (-3: with the end release too steep)
constexpr float kCreakBelow = -6.5f;  // st: statement endings this low get creaky voice (-5: nearly all)
constexpr float kPressedAccent = 0.9f, kLaxUnstressed = 1.08f, kLaxFinal = 1.12f;  // glottal opening
constexpr float kAccentDb = 1.5f, kFunctionWordDb = -1.5f;
constexpr float kSecondaryVowelLength = 0.8f, kInitialRLength = 1.0f;  // see sayPhrase
constexpr float kSyllabicNLength = 1.6f;
constexpr float kLinkGlideMs = 40, kLinkGlideLevel = 0.7f;  // w / j linking a vowel to the next word's
constexpr float kGlideBeforeVowel = 0.4f;  // a diphthong's glide before a vowel: share of that vowel
constexpr float kFlapMs = 22;  // a flap's closure  // the n of -tion (ʃn̩) carries the syllable
// Vowel lengthening (see sayPhrase): factor at 50 ms, change per ms, floor.
constexpr float kVowelLength50 = 1.46f, kVowelLengthSlope = -0.0029f, kVowelLengthMin = 1.1f;

// Parallel halves at word j ("and" / "or"): the words right after it repeat
// words from before it, and the half after starts with one ("all").
bool EnglishReader::symmetricAt(size_t j) const {
  if (j < 2 || j + 2 >= phrase_.size()) return false;
  if (phrase_[j].text != "and" && phrase_[j].text != "or") return false;
  int repeated = 0;
  for (size_t k = j + 1; k < phrase_.size() && k <= j + 3; ++k)
    for (size_t m = 0; m < j; ++m)
      if (phrase_[m].text == phrase_[k].text) {
        ++repeated;
        break;
      }
  bool startsAlike = false;
  for (size_t m = 0; m < j; ++m) startsAlike = startsAlike || phrase_[m].text == phrase_[j + 1].text;
  return repeated >= 2 && startsAlike;
}

void EnglishReader::sayPhrase(char end) {
  if (breathBeforeNext_ && !phrase_.empty()) {
    synth_.breathPoint(true);
    breathBeforeNext_ = false;
  }
  if (phrase_.empty()) return;
  numberJustSaid_ = false;
  // Parallel halves ("all forms of human poverty | and all forms of human
  // life"): when the words right after "and" / "or" repeat words from before
  // it, a short break before it sets up the expectation of symmetry.
  // A long-phrase split deferred at "and" / "or" (addWord) that turned out
  // not symmetric still takes its light break.
  size_t split = deferredSplit_;
  deferredSplit_ = 0;
  for (size_t j = 2; j + 2 < phrase_.size(); ++j) {
    if (!symmetricAt(j)) continue;
    split = 0;
    std::vector<Word> tail(phrase_.begin() + static_cast<long>(j), phrase_.end());
    phrase_.resize(j);
    sayPhrase(',');
    pause(kSymmetryPauseStops);
    phrase_ = std::move(tail);
    break;
  }
  if (split > 0 && split < phrase_.size()) {
    std::vector<Word> tail(phrase_.begin() + static_cast<long>(split), phrase_.end());
    phrase_.resize(split);
    sayPhrase(',');
    pause(kLightBreakStops);
    phrase_ = std::move(tail);
  }
  sentenceEnd_ = end == '.' || end == '!' || end == '?';
  lastEnd_ = end;
  if (phraseInSentence_ == 0) sentenceFirstWord_ = phrase_[0].text;
  // "nor" carries the negation ("neither ... nor"): accented. "can" answering
  // a "cannot" said shortly before is the contrast: accented, its full form.
  for (Word& pw : phrase_) {
    if (pw.text == "nor") pw.accent = true;
    if (pw.text == "can" && cannotSentences_ > 0) pw.accent = true;
    if (pw.text == "cannot" || pw.text == "can't") cannotSentences_ = 2;
  }
  applyForeign(phrase_, kGerman);
  applyForeign(phrase_, kLatin);
  // Weak forms: an unaccented function word inside a phrase is said with a
  // reduced vowel (to /tə/, for /fɚ/, can /kən/); at the end of a phrase
  // ("where are you from?") it keeps its full form.
  // (Not the phrase's first word: it leans on nothing before it, "His car",
  // "Are you" -> with ɪz, ɜɹ it was heard as "is", "or".)
  for (size_t w = 1; w + 1 < phrase_.size(); ++w) {
    if (phrase_[w].accent) continue;
    // have/has/had are weak only as auxiliaries ("have been", "has made");
    // as main verbs they keep the full form ("have a big dog": weak, it was
    // heard as "gonna be").
    const std::string& t = phrase_[w].text;
    if ((t == "have" || t == "has" || t == "had") && !isAuxiliaryUse(phrase_[w + 1].text)) continue;
    for (const WeakForm& f : kWeakForms)
      if (t == f.word) phrase_[w].tokens = arpaTokens(f.weak);
  }
  // Demonstrative "that" (after a preposition, before a content word: "to
  // that day") is accented; the conjunction and relative "that" are not.
  static const char* const kPrep[] = {"to", "on", "in", "at", "of", "for", "from", "with", "by",
                                      "about", "like", "into"};
  for (size_t w = 1; w + 1 < phrase_.size(); ++w) {
    if (phrase_[w].text != "that" || !phrase_[w + 1].accent) continue;
    for (const char* p : kPrep)
      if (phrase_[w - 1].text == p) phrase_[w].accent = true;
  }
  // Before a dental fricative (ð θ), s and z move to the teeth: "is that"
  // is said with a dental z (ɪð ðæt), "this thing" with a dental s. Only
  // after a vowel across words: after a consonant ("hands the") the s or z
  // is a plural or verb ending, and turning it into ð loses it.
  auto tokenOf = [](const char* name) {
    for (int t = 0; t < static_cast<int>(sizeof(kEnglishTokens) / sizeof(kEnglishTokens[0])); ++t)
      if (!strcmp(kEnglishTokens[t], name)) return static_cast<uint8_t>(0x80 + t);
    return static_cast<uint8_t>(0);
  };
  static const uint8_t kS = tokenOf("S"), kZ = tokenOf("Z"), kTH = tokenOf("TH"), kDH = tokenOf("DH");
  for (size_t w = 0; w < phrase_.size(); ++w) {
    auto& t = phrase_[w].tokens;
    for (size_t i = 0; i < t.size(); ++i) {
      if (t[i] != kS && t[i] != kZ) continue;
      const uint8_t next = i + 1 < t.size() ? t[i + 1]
                           : w + 1 < phrase_.size() && !phrase_[w + 1].tokens.empty()
                               ? phrase_[w + 1].tokens[0] : 0;
      const uint8_t prev = i > 0 ? t[i - 1]
                           : w > 0 && !phrase_[w - 1].tokens.empty()
                               ? phrase_[w - 1].tokens.back() : 0;
      // (Not after the r-coloured vowel of -er: "workers this" lost its plural.)
      const bool afterVowel = prev >= tokenOf("AA0") && prev != tokenOf("ER0") &&
                              prev != tokenOf("ER1") && prev != tokenOf("ER2");
      // Inside a word the cluster simplifies anyway ("sixth" sɪkθ).
      const bool inWord = i + 1 < t.size();
      if ((afterVowel || inWord) && (next == kDH || next == kTH)) t[i] = t[i] == kZ ? kDH : kTH;
    }
  }
  // -tion, -sion ("revolution" ʃən): the schwa drops and the n carries the
  // syllable (ʃn̩, as usual in American English); with the schwa's own
  // target between ʃ and n the formants travelled up, down and up again.
  // Before the syllables and durations are counted (dropped after them,
  // every later vowel got its predecessor's pitch: "powers." rose).
  {
    static const uint8_t kSH = tokenOf("SH"), kZH = tokenOf("ZH"), kAH0 = tokenOf("AH0"),
                         kN = tokenOf("N");
    for (auto& word : phrase_) {
      auto& t = word.tokens;
      const size_t n = t.size();
      if (n >= 4 && t[n - 1] == kN && t[n - 2] == kAH0 && (t[n - 3] == kSH || t[n - 3] == kZH)) {
        t.erase(t.end() - 2);
        word.syllabicN = true;
      }
    }
  }
  // Syllables: one per vowel token (a diphthong is one token).
  struct Syl {
    size_t word, token;
    int stress;  // 0, 1, 2 as in CMUdict
    bool accent;
    float pitch;
    float end = Synth::kNoPitch;  // a second pitch inside the vowel (final fall)
  };
  std::vector<Syl> syl;
  for (size_t w = 0; w < phrase_.size(); ++w)
    for (size_t i = 0; i < phrase_[w].tokens.size(); ++i)
      if (isVowelToken(phrase_[w].tokens[i])) {
        const char* s = kEnglishTokens[phrase_[w].tokens[i] - 0x80];
        const int st = s[strlen(s) - 1] - '0';
        syl.push_back({w, i, st, phrase_[w].accent && st == 1, 0});
      }
  int lastAccent = -1;
  for (size_t k = 0; k < syl.size(); ++k)
    if (syl[k].accent) lastAccent = static_cast<int>(k);
  // Phrase-to-phrase variation: the first sentence of a paragraph starts
  // higher, later ones lower; a phrase after a comma resets only partly;
  // and each phrase gets a small deterministic variation from its words
  // (the same text always sounds the same).
  unsigned hash = 2166136261u;
  for (const Word& pw : phrase_)
    for (char ch : pw.text) hash = (hash ^ static_cast<unsigned char>(ch)) * 16777619u;
  const float vary = (static_cast<float>(hash % 1001) / 500.0f - 1.0f) * kPhraseVariation;
  const float topic = sentenceInParagraph_ == 0 ? kParagraphStartLift
                      : -std::min(kSentenceStep * sentenceInParagraph_, kSentenceStepMax);
  // "not X, but Y": the "but" phrase restarts high, not lower like a
  // phrase after a comma.
  const bool butAfterNot = !phrase_.empty() && phrase_[0].text == "but" && prevPhraseHadNot_;
  // Quoted speech (dialogue) sits a little higher than the narration.
  // A speech tag after a quotation ("Hello," said Mum) is an aside: lower,
  // with small accents.
  const float startShift = topic + (phraseInSentence_ > 0 && !butAfterNot ? kAfterCommaStart : 0.0f) +
                           (inQuote_ ? kQuoteLift : 0.0f) + (speechTag_ ? kSpeechTagDrop : 0.0f);
  // A phrase parallel to the one before it (two or more of the same content
  // words: "ask not what your country ... ask what you can do for your
  // country") keeps its repeated words: they carry the contrast.
  int shared = 0;
  for (const Word& pw : phrase_)
    if (pw.accent && std::find(prevPhraseWords_.begin(), prevPhraseWords_.end(), pw.text) != prevPhraseWords_.end())
      ++shared;
  const bool parallel = shared >= 2;
  // Exclamations and repeated phrases build up: "Let them come to Berlin"
  // said again gets a wider range each time, instead of being treated as
  // known (given) material.
  std::string key;
  for (const Word& pw : phrase_) key += pw.text + " ";
  const int repeats = static_cast<int>(std::count(recentPhrases_.begin(), recentPhrases_.end(), key));
  recentPhrases_.push_back(key);
  if (recentPhrases_.size() > kRecentPhrases) recentPhrases_.pop_front();
  const float build = std::min(kRepeatLift * repeats, kRepeatLiftMax) + (end == '!' ? kExclaimLift : 0.0f);
  float accentHeight = (kFirstAccent + topic + vary + build) * (speechTag_ ? kSpeechTagAccent : 1.0f);
  const bool tag = speechTag_;
  speechTag_ = false;
  // Declination spread over the whole phrase (no floor, so long phrases do
  // not end in a flat drone): kDeclination per syllable, the total kept
  // between kDeclinationMin and kDeclinationMax.
  const float n = static_cast<float>(syl.size());
  const float decline = std::clamp(kDeclination * n, kDeclinationMax, kDeclinationMin);
  for (size_t k = 0; k < syl.size(); ++k) {
    const float base = kPhraseStart + startShift + decline * (n > 1 ? k / (n - 1) : 0.0f);
    float p = base;
    if (syl[k].accent) {
      // Accent size by meaning: emphatic words more; words said shortly
      // before (given) and the first of two adjacent content words less;
      // the last accent of the phrase (the nucleus) is not the smallest.
      const size_t w = syl[k].word;
      const std::string& text = phrase_[w].text;
      float scale = 1.0f;
      bool emphatic = false;
      for (const char* e : kEmphatic) emphatic = emphatic || text == e;
      if (!parallel && repeats == 0 && std::find(recent_.begin(), recent_.end(), text) != recent_.end())
        scale *= kGivenScale;
      if (w + 1 < phrase_.size() && phrase_[w + 1].accent && static_cast<int>(k) != lastAccent)
        scale *= kPrenuclearScale;
      // "Let them come to Berlin": the imperative "let" yields to the words
      // that carry the meaning.
      if (w == 0 && text == "let") scale *= kLetScale;
      const float height = static_cast<int>(k) == lastAccent && !tag
                               ? std::max(accentHeight, kNuclearMin + vary) : accentHeight;
      // The nucleus stands on a baseline declined by at most kNucleusDecline
      // (on the full decline a late climax was buried, "...to Berlin").
      const float accentBase = static_cast<int>(k) == lastAccent
                                   ? std::max(base, kPhraseStart + startShift + kNucleusDecline) : base;
      p = accentBase + height * scale;
      // Emphasis words reach most of a first accent's height, however far
      // the phrase has stepped down ("come not from": "not" was at +2 st).
      if (emphatic) p = std::max(p + kEmphasisBoost, base + kEmphaticFloor * (kFirstAccent + topic + build));
      accentHeight = std::max(accentHeight * kDownstep, kMinAccent);
    } else if (k > 0 && syl[k - 1].accent) {
      p = base + kPostAccent * (syl[k - 1].pitch - base);
    } else if (syl[k].stress == 2 && phrase_[syl[k].word].accent) {
      p = base + kSecondary;
    }
    syl[k].pitch = p;
  }
  // A short phrase closing a sentence ("... but from the hand of God.") is
  // its climax: the nucleus at least about as high as the sentence's
  // highest accent so far (it was stepped down below them all).
  const bool sentenceClose = end == '.' || end == '!';
  if (sentenceClose && phraseInSentence_ > 0 && lastAccent >= 0 && syl.size() <= kShortClosingSyllables &&
      sentenceMaxAccent_ > -100)
    syl[lastAccent].pitch = std::max(syl[lastAccent].pitch, sentenceMaxAccent_ + kClosingNucleusBelow);
  for (const Syl& s : syl)
    if (s.accent) sentenceMaxAccent_ = std::max(sentenceMaxAccent_, s.pitch);
  // Between two accents the unaccented syllables stay up on the line from
  // one accent to the next, sagging a little (the hat pattern), instead of
  // dropping to the baseline and climbing back for every content word.
  // Nine human readers: a function word between two accents sits 0 to
  // 0.3 st below the mean of its neighbours (p75 0.6-2.9), a vowel 3 st or
  // more below both neighbours in 5-15 % of cases; puhuja's was in 27 %, a
  // singsong on "that", "are", "the". With this (2 st per syllable of the
  // run, capped at 5): 2 %, steps between vowels median 1.4 st (humans
  // 1.2-2.2).
  for (size_t ka = 0; ka < syl.size(); ++ka) {
    if (!syl[ka].accent) continue;
    size_t kb = ka + 1;
    while (kb < syl.size() && !syl[kb].accent) ++kb;
    if (kb >= syl.size() || kb == ka + 1) continue;
    const float sag = std::min(kInterAccentSagPerSyl * static_cast<float>(kb - ka - 1), kInterAccentSagMax);
    for (size_t k = ka + 1; k < kb; ++k) {
      const float f = static_cast<float>(k - ka) / static_cast<float>(kb - ka);
      const float line = syl[ka].pitch + f * (syl[kb].pitch - syl[ka].pitch);
      syl[k].pitch = line - sag * std::sin(static_cast<float>(M_PI) * f);
    }
  }
  // Phrase ending: from the last accent the pitch moves to the end target.
  if (!syl.empty() && end != ' ') {
    // Endings: wh-questions fall like statements; a statement falls fully
    // when it is exclaimed or long, less when short, and about one in
    // kLevelEndingEvery short ones ends level (the voice-overs end 28-58 %
    // of statements level or rising).
    static const char* const kWh[] = {"what", "why", "how", "who", "where", "when", "which", "whose"};
    bool wh = false;
    for (const char* q : kWh) wh = wh || sentenceFirstWord_ == q;
    // An either/or question ("a pen or a pin?") falls like a statement.
    bool eitherOr = false;
    for (size_t w = 0; w + 1 < phrase_.size(); ++w) eitherOr = eitherOr || phrase_[w].text == "or";
    const bool question = end == '?' && !wh && !eitherOr;
    const bool final = end == '.' || end == '!' || end == '?' || end == ';' || end == ':';
    // Long or short by the whole sentence ("In the long history of the world,
    // ... maximum danger." ended with a short phrase's weaker fall).
    const size_t sentenceSyl = sentenceSyllables_ + syl.size();
    float statement = end == '!' || sentenceSyl >= kLongSentenceSyllables ? kFinalLow : kFinalLowShort;
    if (end == '.' && sentenceSyl < kLongSentenceSyllables && hash % kLevelEndingEvery == 0)
      statement = kLevelEnding;
    // Endings vary a little (from the words, so repeatable): every full
    // fall at the same depth sounded recited.
    if (statement < kLevelEnding) statement += static_cast<float>(static_cast<int>((hash >> 10) % 3) - 1) * kEndingVariety;
    const float target = question ? kQuestionHigh : final ? statement : kContinuation;
    const size_t last = syl.size() - 1;
    if (question && lastAccent >= 0) syl[lastAccent].pitch = -1.0f;  // low accent, then rise
    const size_t from = lastAccent >= 0 ? static_cast<size_t>(lastAccent) + 1 : last;
    // A statement's last vowel falls inside itself to the final low (goog.wav
    // falls ~6.5 st within the last word; with one pitch per vowel the
    // endings stayed level and sounded unfinished).
    const bool falling = !question && end != ',' && end != ' ';
    if (from > last) {
      // The accent is on the last syllable: it starts at the accent and falls
      // (questions and continuations meet the target halfway).
      // A continuation keeps the accent's peak and glides down to where it
      // used to sit (list items like "disease," lost their accent).
      if (falling) syl[last].end = target;
      else if (!question) syl[last].end = 0.5f * (syl[last].pitch + target);
      else syl[last].pitch = 0.5f * (syl[last].pitch + target);
    } else {
      for (size_t k = from; k <= last; ++k) {
        const float f = static_cast<float>(k - from + 1) / static_cast<float>(last - from + 1);
        syl[k].pitch += f * (target - syl[k].pitch);
      }
      // The last vowel glides into the low instead of sitting on it.
      if (falling && last > from) {
        syl[last].end = target;
        syl[last].pitch = 0.5f * (syl[last - 1].pitch + target);
      }
    }
  }

  const std::vector<std::vector<float>> dur = durations();
  // "the" before a vowel sound is "thee" (ði): "the heirs" with ðə ran into
  // one syllable, "theirs". The spelling does not tell (heir, hour, honest,
  // but not union): the next word's first sound does.
  {
    static const uint8_t kIY0 = tokenOf("IY0");
    for (size_t w = 0; w + 1 < phrase_.size(); ++w) {
      if (phrase_[w].text != "the" || phrase_[w + 1].tokens.empty()) continue;
      if (!isVowelToken(phrase_[w + 1].tokens[0])) continue;
      auto& t = phrase_[w].tokens;
      if (t.size() == 2 && t[0] == kDH) t[1] = kIY0;
    }
  }
  // American flap: t or d after a vowel (or r) and before an unstressed
  // vowel in the same word ("city", "better", "water"): a quick voiced tap
  // instead of a full, aspirated t.
  {
    static const uint8_t kT = tokenOf("T"), kD = tokenOf("D"), kR = tokenOf("R");
    auto unstressedVowel = [](uint8_t t) {
      if (!isVowelToken(t)) return false;
      const char* s = kEnglishTokens[t - 0x80];
      return s[strlen(s) - 1] == '0';
    };
    for (size_t w = 0; w < phrase_.size(); ++w) {
      auto& t = phrase_[w].tokens;
      for (size_t i = 1; i < t.size() && i < 32; ++i) {
        if (t[i] != kT && t[i] != kD) continue;
        if (!isVowelToken(t[i - 1]) && t[i - 1] != kR) continue;
        // Inside the word only: across a word boundary ("bet again") the
        // flap merged the words for listeners ("better than").
        if (i + 1 >= t.size() || !unstressedVowel(t[i + 1])) continue;
        t[i] = kD;
        phrase_[w].flaps |= 1u << i;
      }
    }
  }
  // Final devoicing: before a sentence-end pause a final b d g loses its
  // voicing and is heard as p t k ("world." ends like "worlt"); a voiced
  // release into silence sounded like an extra "də". After the durations:
  // the vowel before keeps its length, the cue that it is a d.
  if (end == '.' || end == '!' || end == '?' || end == ';' || end == ':') {
    for (size_t w = phrase_.size(); w-- > 0;) {
      auto& t = phrase_[w].tokens;
      if (t.empty()) continue;
      static const uint8_t kB = tokenOf("B"), kD = tokenOf("D"), kG = tokenOf("G");
      if (t.back() == kB) t.back() = tokenOf("P");
      else if (t.back() == kD) t.back() = tokenOf("T");
      else if (t.back() == kG) t.back() = tokenOf("K");
      break;
    }
  }
  size_t si = 0;
  for (size_t w = 0; w < phrase_.size(); ++w) {
    // No pause between words: connected speech runs words together (the
    // voice-over reference has no gap between words). Except when the air
    // runs low in a long stretch without punctuation: then a short pause
    // and a breath between words, before the voice would fade out.
    if (w > 0 && synth_.plannedAir() < kEmergencyBreathAir) {
      pause(4);
      synth_.breathPoint(false);
    }
    const std::vector<uint8_t>& tokens = phrase_[w].tokens;
    // Primary stress mark (content words only) before the stressed onset.
    int stressAt = -1;
    for (size_t i = 0; phrase_[w].accent && i < tokens.size(); ++i) {
      const char* s = kEnglishTokens[tokens[i] - 0x80];
      if (s[strlen(s) - 1] != '1') continue;
      const int k = static_cast<int>(i);
      int c = 0;
      while (k - c - 1 >= 0 && !isVowelToken(tokens[k - c - 1])) ++c;
      const bool initial = k - c == 0;
      int onset = initial ? c : std::min(c, 1);
      if (!initial && c >= 2 &&
          onsetPair(kEnglishTokens[tokens[k - 2] - 0x80], kEnglishTokens[tokens[k - 1] - 0x80]))
        onset = 2;
      stressAt = k - onset;
      break;
    }
    bool first = true;
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (static_cast<int>(i) == stressAt) synth_.pushText("\xcb\x88");  // ˈ
      // A vowel-initial accented word starting a phrase gets a glottal onset.
      if (i == 0 && w == 0 && phrase_[w].accent && isVowelToken(tokens[0])) synth_.glottalOnset();
      // The last vowel of a low statement ending turns creaky.
      if (isVowelToken(tokens[i]) && si + 1 == syl.size() && sentenceEnd_ && lastEnd_ != '?' &&
          syl.back().pitch < kCreakBelow)
        synth_.creakyEnd();
      const bool vowel = isVowelToken(tokens[i]);
      const float pitch = vowel && si < syl.size() ? syl[si].pitch : Synth::kNoPitch;
      const float pitchEnd = vowel && si < syl.size() ? syl[si].end : Synth::kNoPitch;
      if (vowel) ++si;
      Sound s = soundOf(kEnglishTokens[tokens[i] - 0x80]);
      // ɜ + ɹ before a consonant or a pause ("word", "first", "for the"):
      // one r-coloured vowel (its low F3 is the colour, /wɝːd/), no ɹ
      // segment: the separate ɹ, at twice the vowel's level, was heard as
      // a consonant r. Before a vowel it stays and links ("her eyes").
      if (s.ph[0] == Ph::ER && s.n == 2) {
        const uint8_t nt = i + 1 < tokens.size() ? tokens[i + 1]
                           : w + 1 < phrase_.size() && !phrase_[w + 1].tokens.empty() ? phrase_[w + 1].tokens[0] : 0;
        if (!(nt && isVowelToken(nt))) s.n = 1;
      }
      for (int k = 0; k < s.n; ++k) {
        // The second pitch on the sound's last vowel part (a diphthong's glide).
        const int lastPart = s.n > 1 && s.ph[s.n - 1] != Ph::RR ? s.n - 1 : 0;
        // A falling diphthong falls through its glide: the glide's centre is
        // already halfway down ("right.", "now.": the fall came late, all of
        // it after the glide's centre).
        const bool fallingGlide = vowel && k > 0 && k == lastPart && pitchEnd != Synth::kNoPitch;
        if (vowel && s.ph[k] != Ph::RR)  // ER = vowel + ɹ
          synth_.setPitch(fallingGlide ? pitch + kGlideFallShare * (pitchEnd - pitch) : pitch);
        if (vowel && k == lastPart && pitchEnd != Synth::kNoPitch) synth_.setPitchEnd(pitchEnd);
        // English variant of this phoneme (tools/tune_english.py).
        const auto& v = kEnglishShape[static_cast<int>(s.ph[k])];
        Synth::Shape shape;
        const bool glide = k == 1 && s.backOffglide;
        const float* start = k == 0 ? s.start : nullptr;
        // aʊ ends less far back than oʊ (voice-over: F2 ~1390 against ~1000).
        const float* glideEnd = s.start == kOpenCentralStart ? kAwGlide : kBackGlide;
        shape.f1 = glide ? glideEnd[0] : start ? start[0] : v.f1;
        shape.f2 = glide ? glideEnd[1] : start ? start[1] : v.f2;
        shape.f3 = glide ? glideEnd[2] : start ? start[2] : v.f3;
        // A diphthong's glide right before a vowel ("loyalty" ɔɪ-ə) falls
        // short, toward that vowel: reaching the full ɪ (F2 ~2000) between
        // ɔ and ə made the formants swing up and down again.
        if (k == 1 && s.n == 2 && s.ph[1] != Ph::RR && !s.backOffglide) {
          const uint8_t nt = i + 1 < tokens.size() ? tokens[i + 1]
                             : w + 1 < phrase_.size() && !phrase_[w + 1].tokens.empty() ? phrase_[w + 1].tokens[0] : 0;
          if (nt && isVowelToken(nt)) {
            const Sound ns = soundOf(kEnglishTokens[nt - 0x80]);
            const auto& nv = kEnglishShape[static_cast<int>(ns.ph[0])];
            const float own[3] = {shape.f1 > 0 ? shape.f1 : Synth::tableFormant(s.ph[1], 0), shape.f2 > 0 ? shape.f2 : Synth::tableFormant(s.ph[1], 1),
                                  shape.f3 > 0 ? shape.f3 : Synth::tableFormant(s.ph[1], 2)};
            const float nxt[3] = {nv.f1 > 0 ? nv.f1 : Synth::tableFormant(ns.ph[0], 0), nv.f2 > 0 ? nv.f2 : Synth::tableFormant(ns.ph[0], 1),
                                  nv.f3 > 0 ? nv.f3 : Synth::tableFormant(ns.ph[0], 2)};
            shape.f1 = own[0] + kGlideBeforeVowel * (nxt[0] - own[0]);
            shape.f2 = own[1] + kGlideBeforeVowel * (nxt[1] - own[1]);
            shape.f3 = own[2] + kGlideBeforeVowel * (nxt[2] - own[2]);
          }
        }
        // Duration (Klatt rules); diphthongs share it 60/40, ɜ+ɹ 80/20 (the
        // vowel is r-coloured itself; the ɹ is only its tail).
        const float firstShare = s.ph[0] == Ph::ER ? 0.8f : 0.6f;
        const float share = s.n == 1 ? 1.0f : k == 0 ? firstShare : 1.0f - firstShare;
        shape.ms = dur[w][i] * share * v.length;
        // Vowels as long as in the Google reading of kennedy.txt (goog.wav):
        // short vowels are lengthened most (x1.46 at 50 ms), long ones less
        // (x1.17 at 150 ms).
        if (vowel)
          shape.ms *= std::clamp(kVowelLength50 + kVowelLengthSlope * (shape.ms - 50.0f),
                                 kVowelLengthMin, kVowelLength50);
        // A secondary-stress vowel ("revolution") is shorter, a word-initial
        // ɹ before a vowel too: both sounded drawn out ("re-").
        if (vowel) {
          const char* sn = kEnglishTokens[tokens[i] - 0x80];
          if (sn[strlen(sn) - 1] == '2') shape.ms *= kSecondaryVowelLength;
        }
        if (i == 0 && s.ph[k] == Ph::RR && tokens.size() > 1 && isVowelToken(tokens[1]))
          shape.ms *= kInitialRLength;
        if (phrase_[w].syllabicN && i + 1 == tokens.size()) shape.ms *= kSyllabicNLength;
        if (i < 32 && (phrase_[w].flaps >> i & 1)) {
          shape.flap = true;
          shape.ms = kFlapMs;
        }
        // Plosives: Klatt's duration includes the release; this is the closure.
        if (isVoicelessPlosive(baseName(tokens[i])) || isVoicedPlosive(baseName(tokens[i])))
          shape.ms *= 0.75f;
        // The back offglide fades: about 3 dB weaker than a full vowel.
        shape.level = glide ? 0.7f * v.level : v.level;
        // Voice quality: accented vowels slightly pressed (brighter, more
        // closed phase), unstressed and phrase-final ones slightly lax.
        if (vowel) {
          const char* vq = kEnglishTokens[tokens[i] - 0x80];
          const bool accented = phrase_[w].accent && vq[strlen(vq) - 1] == '1';
          shape.press = accented ? kPressedAccent
                        : si == syl.size() ? kLaxFinal : kLaxUnstressed;
        }
        // Loudness: an arc over the phrase (louder at the start, softer
        // toward its end), accents louder, function words softer. Vowels
        // only (the consonants keep their calibrated levels).
        if (vowel) {
          const float frac = syl.size() > 1 ? static_cast<float>(si - 1) / (syl.size() - 1) : 0.0f;
          float db = kArcStartDb + (kArcEndDb - kArcStartDb) * frac;
          const char* vn = kEnglishTokens[tokens[i] - 0x80];
          if (sentenceEnd_ && lastEnd_ == '!') db += kExclaimDb;
          if (phrase_[w].accent && vn[strlen(vn) - 1] == '1') db += kAccentDb;
          else if (!phrase_[w].accent) db += kFunctionWordDb;
          shape.level *= std::pow(10.0f, db / 20.0f);
        }
        synth_.shapeNext(shape);
        synth_.push(s.ph[k], first);
        first = false;
        if (k == 0 && s.longVowel) synth_.lengthen();
      }
      // Linking: a word ending in a high vowel or a diphthong's glide runs
      // into a vowel-initial next word through a short w or j ("to assure"
      // tuː w‿əʃʊɹ, "the heirs" ði j‿ɛɹz). Without it the two vowels blurred
      // into one ("to assure" heard as "to show").
      if (vowel && i + 1 == tokens.size() && w + 1 < phrase_.size() &&
          !phrase_[w + 1].tokens.empty() && isVowelToken(phrase_[w + 1].tokens[0])) {
        const Ph last = s.ph[s.n - 1];
        const Ph link = last == Ph::U || last == Ph::UH ? Ph::W
                        : last == Ph::I || last == Ph::IH ? Ph::J : Ph::Stop;
        if (link != Ph::Stop) {
          Synth::Shape ls;
          ls.ms = kLinkGlideMs;
          ls.level = kLinkGlideLevel * kEnglishShape[static_cast<int>(link)].level;
          const auto& lv = kEnglishShape[static_cast<int>(link)];
          ls.f1 = lv.f1;
          ls.f2 = lv.f2;
          ls.f3 = lv.f3;
          synth_.shapeNext(ls);
          synth_.push(link, false);
        }
      }
    }
  }
  for (const Word& pw : phrase_)
    if (pw.accent) {
      recent_.push_back(pw.text);
      if (recent_.size() > kRecentWords) recent_.pop_front();
    }
  // Kept for the next phrase of the sentence (parallelism, "not ... but").
  prevPhraseWords_.clear();
  prevPhraseHadNot_ = false;
  for (const Word& pw : phrase_) {
    if (pw.accent) prevPhraseWords_.push_back(pw.text);
    prevPhraseHadNot_ = prevPhraseHadNot_ || pw.text == "not" || pw.text == "never";
  }
  sentenceSyllables_ += static_cast<size_t>(phraseSyllables());
  if (end == '.' || end == '!' || end == '?') {
    ++sentenceInParagraph_;
    phraseInSentence_ = 0;
    sentenceSyllables_ = 0;
    sentenceMaxAccent_ = -1000;
    if (cannotSentences_ > 0) --cannotSentences_;
    // (The accented words carry over one sentence: "Let us never negotiate
    // out of fear. But let us never fear to negotiate." is parallel too.)
  } else {
    ++phraseInSentence_;
  }
  phrase_.clear();
}

void EnglishReader::pause(int stops) {
  // Pauses follow the rate most (rate^2, relative to the default).
  const float r = rate_ / kDefaultRate;
  stops = static_cast<int>(stops / (kDefaultRate * r * r) + 0.5f);
  for (int i = 0; i < stops; ++i) synth_.push(Ph::Stop);
}

// Units after a number: as written, singular, plural.
const char* const kUnits[][3] = {
    {"km", "kilometer", "kilometers"}, {"cm", "centimeter", "centimeters"},
    {"mm", "millimeter", "millimeters"}, {"kg", "kilogram", "kilograms"}, {"mg", "milligram", "milligrams"},
    {"g", "gram", "grams"}, {"ml", "milliliter", "milliliters"}, {"ft", "foot", "feet"},
    {"lb", "pound", "pounds"}, {"lbs", "pound", "pounds"}, {"oz", "ounce", "ounces"},
    {"mph", "mile per hour", "miles per hour"}, {"kph", "kilometer per hour", "kilometers per hour"},
    {"h", "hour", "hours"}, {"hr", "hour", "hours"}, {"hrs", "hour", "hours"},
    {"min", "minute", "minutes"}, {"sec", "second", "seconds"}, {"ms", "millisecond", "milliseconds"},
    {"kb", "kilobyte", "kilobytes"}, {"mb", "megabyte", "megabytes"}, {"gb", "gigabyte", "gigabytes"},
    {"tb", "terabyte", "terabytes"}, {"hz", "hertz", "hertz"}, {"khz", "kilohertz", "kilohertz"},
    {"mhz", "megahertz", "megahertz"}, {"ghz", "gigahertz", "gigahertz"}, {"kw", "kilowatt", "kilowatts"},
    {"kwh", "kilowatt hour", "kilowatt hours"}};

void EnglishReader::endWord() {
  capsWord_ = !word_.empty() && wordCaps_ == static_cast<int>(word_.size());
  wordCaps_ = 0;
  if (!word_.empty()) {
    // After a number: "am" / "pm" spelled (9:00 am: "nine a m"), units
    // read out ("5 km": "five kilometers").
    const char* unit = nullptr;
    if (numberJustSaid_)
      for (const auto& u : kUnits)
        if (word_ == u[0]) unit = numberWasOne_ ? u[1] : u[2];
    if (numberJustSaid_ && (word_ == "am" || word_ == "pm")) {
      if (!phrase_.empty() && phrase_.back().text == "o'clock") phrase_.pop_back();
      for (char ch : word_)
        phrase_.push_back({arpaTokens(kLetterNames[ch - 'a']), ch == 'm', std::string(1, ch)});
    } else if (unit) {
      std::string u = unit;
      for (size_t a = 0, b; a < u.size(); a = b + 1) {
        b = u.find(' ', a);
        if (b == std::string::npos) b = u.size();
        addWord(u.substr(a, b - a));
      }
    } else {
      addWord(word_);
    }
    numberJustSaid_ = false;
  }
  capsWord_ = false;
  if (!digits_.empty()) addNumber(digits_);
  word_.clear();
  digits_.clear();
}

void EnglishReader::flush() {
  if (!pending_.empty()) {  // the held-back last word
    flushing_ = true;
    const std::string rest = pending_;
    pending_.clear();
    push(rest);
    flushing_ = false;
  }
  endWord();
  sayPhrase('.');
}

// Abbreviations read in full; single letters (initials, U.S.) stay letters.
// Those marked are read in full without the period too (the others are
// words or ambiguous: st street, gen, rep, mt).
const Abbreviation kAbbreviations[] = {
    {"mr", "mister", true}, {"mrs", "missus", true}, {"ms", "miz"}, {"dr", "doctor"}, {"prof", "professor"},
    {"st", "saint"}, {"jr", "junior", true}, {"sr", "senior", true}, {"vs", "versus", true}, {"gen", "general"},
    {"gov", "governor"}, {"sen", "senator"}, {"rep", "representative"}, {"mt", "mount"},
    {"etc", "et cetera", true}, {"approx", "approximately", true}, {"dept", "department", true}};

bool EnglishReader::expandAbbreviation(bool bare) {
  Word& last = phrase_.back();
  if (last.text.size() == 1) return true;  // an initial: keep reading
  for (const Abbreviation& a : kAbbreviations) {
    if (last.text != a.abbr || (bare && !a.bare)) continue;
    phrase_.pop_back();
    std::string full = a.full;
    size_t p = 0;
    while (p < full.size()) {
      size_t q = full.find(' ', p);
      if (q == std::string::npos) q = full.size();
      addWord(full.substr(p, q - p));
      p = q + 1;
    }
    return true;
  }
  return false;
}

// Latin-1 letters (second byte of the UTF-8 pair C3 xx, both cases) folded
// to ASCII, so that "Schöneberg" or "café" stay one word.
const char* foldLatin1(unsigned char b) {
  static const char* const kFold[64] = {
      "a", "a", "a", "a", "a", "a", "ae", "c", "e", "e", "e", "e", "i", "i", "i", "i",
      "d", "n", "o", "o", "o", "o", "o", "", "o", "u", "u", "u", "u", "y", "th", "ss"};
  if (b < 0x80 || b > 0xbf) return "";
  return kFold[(b - 0x80) & 0x1f];
}

int EnglishReader::phraseSyllables() const {
  int n = 0;
  for (const Word& w : phrase_)
    for (uint8_t t : w.tokens) n += isVowelToken(t);
  return n;
}

// Deterministic value in [-1, 1] from the phrase's words.
float EnglishReader::phraseHash() const {
  unsigned h = 2166136261u;
  for (const Word& w : phrase_)
    for (char ch : w.text) h = (h ^ static_cast<unsigned char>(ch)) * 16777619u;
  return static_cast<float>(h % 1001) / 500.0f - 1.0f;
}

// A closing quotation mark: when the quotation's last phrase was already
// said ("Hello," ...), a short pause sets it off from what follows, and the
// next phrase is a speech tag.
void EnglishReader::closeQuote() {
  inQuote_ = false;
  if (phrase_.empty() && word_.empty() && digits_.empty()) {
    pause(kQuotePauseStops);
    speechTag_ = true;
  }
}

void EnglishReader::push(std::string_view input) {
  // A UTF-8 sequence split between two calls is completed with the next.
  std::string joined;
  std::string_view text = input;
  if (!pending_.empty()) {
    joined = pending_ + std::string(input);
    text = joined;
    pending_.clear();
  }
  {
    size_t k = text.size(), back = 0;
    while (k > 0 && back < 3 && (static_cast<unsigned char>(text[k - 1]) & 0xc0) == 0x80) --k, ++back;
    if (k > 0) {
      const unsigned char lead = static_cast<unsigned char>(text[k - 1]);
      const size_t need = lead >= 0xf0 ? 3 : lead >= 0xe0 ? 2 : lead >= 0xc0 ? 1 : 0;
      if (need > back) {
        pending_ = std::string(text.substr(k - 1));
        text = text.substr(0, k - 1);
      }
    }
  }
  // The unfinished last word waits for the next call (or flush): the rules
  // below look one character ahead and back ("21st", "1,000", "$3.50",
  // "--"), and a word split between two calls was read in two pieces.
  if (!flushing_) {
    const size_t ws = text.find_last_of(" \t\n\r");
    const size_t keep = ws == std::string_view::npos ? 0 : ws + 1;
    if (keep < text.size() && text.size() - keep < kMaxHeldBack) {
      pending_ = std::string(text.substr(keep)) + pending_;
      text = text.substr(0, keep);
    }
  }
  for (size_t i = 0; i < text.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    // £ and €: currencies like $; °, °C, °F: degrees.
    if (c == 0xc2 && i + 1 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0xa3) {
      endWord();
      dollarNext_ = 2;
      ++i;
      continue;
    }
    if (c == 0xe2 && i + 2 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0x82 &&
        static_cast<unsigned char>(text[i + 2]) == 0xac) {
      endWord();
      dollarNext_ = 3;
      i += 2;
      continue;
    }
    if (c == 0xc2 && i + 1 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0xb0) {
      const bool one = digits_ == "1";
      endWord();
      numberJustSaid_ = false;
      addWord(one ? "degree" : "degrees");
      ++i;
      const char s = i + 1 < text.size() ? static_cast<char>(text[i + 1] | 0x20) : 0;
      const bool alone = i + 2 >= text.size() || !isalpha(static_cast<unsigned char>(text[i + 2]));
      if ((s == 'c' || s == 'f') && alone) {
        addWord(s == 'c' ? "celsius" : "fahrenheit");
        ++i;
      }
      continue;
    }
    // 10:30 "ten thirty", 9:05 "nine oh five", 9:00 "nine o'clock".
    if (c == ':' && !digits_.empty() && digits_.size() <= 2 && word_.empty() && i + 2 < text.size() &&
        isdigit(static_cast<unsigned char>(text[i + 1])) && isdigit(static_cast<unsigned char>(text[i + 2])) &&
        (i + 3 >= text.size() || !isdigit(static_cast<unsigned char>(text[i + 3])))) {
      endWord();
      const std::string mm(text.substr(i + 1, 2));
      i += 2;
      if (mm == "00") {
        addWord("o'clock");
      } else {
        if (mm[0] == '0') addWord("oh");
        digits_ = mm[0] == '0' ? mm.substr(1) : mm;
        endWord();
      }
      numberJustSaid_ = true;
      numberWasOne_ = false;
      continue;
    }
    // Web and mail addresses: "example.com" is "example dot com",
    // "info@example" "info at example" (a dot between words with no space,
    // before a small letter: "U.S." and sentence ends stay as they were).
    if (c == '.' && word_.size() >= 2 && i + 1 < text.size() && text[i + 1] >= 'a' && text[i + 1] <= 'z') {
      endWord();
      addWord("dot");
      continue;
    }
    if (c == '@' && !word_.empty() && i + 1 < text.size() && isalpha(static_cast<unsigned char>(text[i + 1]))) {
      endWord();
      addWord("at");
      continue;
    }
    if (c == 0xc3 && i + 1 < text.size()) {
      const char* f = foldLatin1(static_cast<unsigned char>(text[i + 1]));
      if (*f) {
        if (!digits_.empty()) endWord();
        word_ += f;
        ++i;
        continue;
      }
    }
    // Typographic punctuation (E2 80 xx): ’ is ', dashes are a pause for
    // thought, “ opens a quotation, … is a full stop.
    if (c == 0xe2 && i + 2 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0x80) {
      const unsigned char d = static_cast<unsigned char>(text[i + 2]);
      if (d == 0x99 || d == 0x98 || d == 0x93 || d == 0x94 || d == 0x9c || d == 0x9d || d == 0xa6) {
        c = d == 0x99 ? '\'' : d == 0x93 || d == 0x94 ? '~' : d == 0x9c ? '"' : d == 0xa6 ? '.' : ' ';
        if (d == 0x98) c = ' ';
        if (d == 0x9d) closeQuote();  // ” closes a quotation
        i += 2;
      }
    }
    if (c == '"') {  // straight quote: opening if it does not follow a word
      const bool opening = i == 0 || text[i - 1] == ' ' || text[i - 1] == '\n' || text[i - 1] == '(';
      if (!opening) closeQuote();  // a closing quote
      c = opening ? '"' : ' ';
    }
    if (c == '-' && i + 1 < text.size() && text[i + 1] == '-') {  // "--" is a dash
      c = '~';
      ++i;
    }
    if (c == '(' || c == ')') c = '|';
    if (c != '\n' && c != '\r' && c != ' ') newlines_ = 0;
    if (c > ' ' && c != '.' && c != '!' && c != '?' && c != '"' && c != '\'') sentenceEnded_ = false;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '\'' && !word_.empty())) {
      // 1990s, 80s: a decade, "nineteen nineties"
      if (!digits_.empty() && word_.empty() && (c == 's' || c == 'S') && digits_.back() == '0' &&
          (i + 1 >= text.size() || !isalpha(static_cast<unsigned char>(text[i + 1])))) {
        pluralNext_ = true;
        endWord();
        continue;
      }
      if (!digits_.empty() && word_.empty() && i + 1 < text.size()) {
        // 1st 2nd 3rd 4th: an ordinal
        std::string suf{static_cast<char>(c | 0x20), static_cast<char>(text[i + 1] | 0x20)};
        const bool end = i + 2 >= text.size() || !isalpha(static_cast<unsigned char>(text[i + 2]));
        if (end && (suf == "st" || suf == "nd" || suf == "rd" || suf == "th")) {
          ordinalNext_ = true;
          ++i;
          endWord();
          continue;
        }
      }
      if (!digits_.empty()) endWord();
      if (c >= 'A' && c <= 'Z') ++wordCaps_;
      word_ += static_cast<char>(c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c);
      continue;
    }
    if (c >= '0' && c <= '9') {
      if (!word_.empty()) endWord();
      digits_ += static_cast<char>(c);
      continue;
    }
    if (c == ',' && !digits_.empty() && i + 1 < text.size() && text[i + 1] >= '0' &&
        text[i + 1] <= '9')
      continue;  // 1,000
    // $3.50: "three dollars and fifty cents"
    if (c == '.' && !digits_.empty() && dollarNext_ && i + 2 < text.size() && isdigit(static_cast<unsigned char>(text[i + 1])) &&
        isdigit(static_cast<unsigned char>(text[i + 2])) &&
        (i + 3 >= text.size() || !isdigit(static_cast<unsigned char>(text[i + 3])))) {
      centsPence_ = dollarNext_ == 2;
      endWord();
      addWord("and");
      centsNext_ = true;
      continue;
    }
    // 1/2, 3/4: "one half", "three quarters"
    if (c == '/' && !digits_.empty() && i + 1 < text.size() && isdigit(static_cast<unsigned char>(text[i + 1]))) {
      numerator_ = std::stoul(digits_.substr(0, 9));
      endWord();
      fractionNext_ = true;
      continue;
    }
    // -5: "minus five" (a minus right before a number)
    if (c == '-' && digits_.empty() && word_.empty() && i + 1 < text.size() &&
        isdigit(static_cast<unsigned char>(text[i + 1])) && (i == 0 || text[i - 1] == ' ' || text[i - 1] == '(')) {
      addWord("minus");
      continue;
    }
    if (c == '.' && !digits_.empty() && i + 1 < text.size() && text[i + 1] >= '0' &&
        text[i + 1] <= '9') {
      endWord();  // 3.14: "three point one four"
      addWord("point");
      pointDigits_ = true;
      continue;
    }
    const bool afterWord = !word_.empty();
    endWord();
    if (c == '%') addWord("percent");
    if (c == '$') dollarNext_ = 1;
    // A full stop right after an abbreviation or a single letter ("Mr.",
    // "U.S.", "John F. Kennedy") does not end the sentence.
    if (c == '.' && afterWord && !phrase_.empty() && expandAbbreviation(false)) continue;
    if (c != '.' && afterWord && !phrase_.empty()) expandAbbreviation(true);
    // Repeated punctuation ("?!", "...", ",,") is one break: after a break
    // with nothing said since, further marks are ignored.
    const bool words = !phrase_.empty();
    switch (c) {
      case ',': case ';': case ':': {
        if (!words) break;
        // Comma pauses by phrase length: none after a short phrase
        // ("Today,"), longer after a long one, with a little variation.
        // ; and : end a clause: a falling end and a longer pause (they were
        // planned as commas). A short phrase still gets a short pause (lists,
        // "Hello," said ...); goog.wav's comma pauses: median 370 ms.
        const int syl = phraseSyllables();
        const float vary = 1.0f + kPauseVariation * phraseHash();
        const bool clause = c == ';' || c == ':';
        sayPhrase(clause ? static_cast<char>(c) : ',');
        // A short phrase after another comma phrase (a list item: "the
        // poverty, the disease,") gets the full comma pause: the run of
        // commas is the list's rhythm.
        int stops = syl > kShortPhraseSyllables
                        ? std::min(kCommaPauseStops + (syl - kShortPhraseSyllables) / 2, kCommaPauseMaxStops)
                        : commaRun_ > 0 ? kCommaPauseStops : kShortCommaPauseStops;
        if (clause) stops = std::max(stops, kClausePauseStops);
        commaRun_ = clause ? 0 : commaRun_ + 1;
        const int commaStops = static_cast<int>(stops * vary + 0.5f);
        maxCommaStops_ = std::max(maxCommaStops_, commaStops);
        pause(commaStops);
        synth_.breathPoint(false);  // only when the air is running low
        break;
      }
      case '|':  // parenthesis: a lighter break, no breath
        if (!words) break;
        sayPhrase(',');
        pause(kDashPauseStops);
        break;
      case '~':  // en or em dash ("--"): a pause for thought
        if (!words) break;
        sayPhrase(',');
        pause(kThinkingPauseStops);
        synth_.breathPoint(false);  // only when the air is running low
        break;
      case '"':  // opening quotation: set off by a short pause
        inQuote_ = true;
        if (!words) break;
        sayPhrase(',');
        pause(kQuotePauseStops);
        break;
      case '.': case '!': case '?':
        if (!words) break;
      {
        const float vary = 1.0f + kPauseVariation * phraseHash();
        sayPhrase(static_cast<char>(c));
        commaRun_ = 0;
        // Pause hierarchy: a sentence end is never a shorter break than a
        // comma inside the sentence.
        pause(std::max(static_cast<int>(kSentencePauseStops * vary + 0.5f), maxCommaStops_));
        maxCommaStops_ = 0;
        // A breath before every sentence (the user's choice; goog.wav's
        // sentence gaps: median 0.43 s), taken when the next one starts:
        // none after the last sentence.
        breathBeforeNext_ = true;
        sentenceEnded_ = true;
        break;
      }
      case '\n':
        // A line without a closing mark (a heading) ends a phrase; an empty
        // line ends a paragraph: a longer pause and a full breath.
        if (words) {
          sayPhrase(',');
          pause(kCommaPauseStops);
        }
        ++newlines_;
        // A line ending right after a sentence (one paragraph per line, as
        // in the Kennedy text): half the paragraph pause; an empty line the
        // rest.
        if (newlines_ == 1 && !words && sentenceEnded_) {
          pause(kParagraphPauseStops / 2);
          sentenceInParagraph_ = 0;
        }
        if (newlines_ == 2) {
          sentenceInParagraph_ = 0;
          pause(kParagraphPauseStops - kParagraphPauseStops / 2);
          synth_.breathPoint(true);
        }
        break;
      default:
        break;  // spaces and other symbols only separate words
    }
  }
}

}  // namespace puhuja
