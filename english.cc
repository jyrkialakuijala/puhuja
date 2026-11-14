#include "english.h"
#include "params.h"

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
constexpr float kAwGlide[3] = {470, 1120, 2400};  // the end of aʊ (450 1300: the references' aʊ offglide in "power", "around", "proud" has F2 820-1310, median ~1050, F1 ~480; puhuja sat at 1290 and "around" had no F2 travel, 2026-10-30)
// An unstressed ɜ+ɹ before a stressed back or open vowel ("around",
// "generosity", "generation") is the r itself: the references' ɜ there has
// F2 930-1230 and F3 1440-1650 (puhuja 1380/1690, with the ɹ only 20 ms);
// before i or ɪ ("century", "every") it stays schwa-like (F2 1240-1600).
// (EN_ER_ONSET_SHARE / EN_ER_ONSET_PULL: ER agent 2026-11-11 lab knobs, 0.5 / 0.5 as before.)
const float kErOnsetShare = param("EN_ER_ONSET_SHARE", 0.5f), kErOnsetPull = param("EN_ER_ONSET_PULL", 0.5f);
constexpr float kErOnsetFrontF2 = 1900.0f;
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
constexpr float kMidBackStart[3] = {500, 1100, 2650};  // (560: the oʊ onset's F1 was 590 against ElevenLabs' 437-497)

bool en0Stressed(uint8_t tok) {  // a primary-stressed vowel token
  const char* n = kEnglishTokens[tok - 0x80];
  return n[strlen(n) - 1] == '1';
}

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
  if (t == "OY") {
    // Lab (OY agent 2026-11-10; 0 = ɔ + ɪ as before): EN_OY_START 1 starts on o, 2 on ɑ; EN_OY_END 1 ends on i.
    static const int kOyStart = static_cast<int>(param("EN_OY_START", 0.0f)), kOyEnd = static_cast<int>(param("EN_OY_END", 0.0f));
    return {{kOyStart == 1 ? Ph::O : kOyStart == 2 ? Ph::AA : Ph::AO, kOyEnd == 1 ? Ph::I : Ph::IH}, 2, false, false, kOpenBackStart};
  }
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
constexpr size_t kMaxHeldSentence = 4000;  // ... and the longest sentence held back whole
constexpr float kEnglishFricationGain = 4.5f;
constexpr float kEnglishAspirationGain = 50.0f, kEnglishHGain = 20.0f;
const float kEnglishVoicedHNoise = param("kEnglishVoicedHNoise", 0.4f);  // x the h gain, h between voiced sounds (readers' medial h is mostly breathy voice, 10-13 dB under the vowel; 1.0 put puhuja's at -1 dB, INPUT.md 2026-10-31) (0.3 with the h gain 0.7: "the hand" was heard as "the land": the h had the l's length, level and harmonicity; 3 was 12 dB louder than goog.wav)
const float kEnglishVoicedHAg = param("kEnglishVoicedHAg", 0.17f);  // 0.25 until 2026-11-05: it stopped the folds, a 45 ms gap in "his"; Jyrki chose 0.15 (hih-Z1)  // (0.2 with the folds stopping above kAgBreathy: h-vowel -7.1 dB over kennedy.txt, 0.25: -8.7, readers -10..-13; round trip 8.6 -> 8.2 %, 2026-11-01)  // cm^2, h between voiced sounds (goog.wav's stays voiced)

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
// Loaded, frightening words ("a far more iron tyranny"): EN_EMOTIVE 1 accents them like the emphatic words;
// EN_EMOTIVE_DB / EN_EMOTIVE_LEN / EN_EMOTIVE_PITCH (st) make them louder, slower, darker (Jyrki 2026-11-10: the
// raised pitch alone sounded "like question for a recipe for a cake").
bool isEmotiveWord(const std::string& t) {
  static const char* const kWords[] = {"iron", "tyranny", "brutal", "terror", "evil", "deadly", "destruction",
                                       "slaughter", "oppression", "savage", "cruel", "doom", "annihilation"};
  for (const char* e : kWords)
    if (t == e) return true;
  return false;
}
// A content word repeated within this many words: the first +kRepeatRise
// st, the second at least kRepeatSecondAbove above it, falling kRepeatFall
// within its vowel, after a pause of kRepeatPauseStops (~15 ms each).
constexpr size_t kRepeatWithin = 5;
const float kRepeatRise = param("EN_REPEAT_RISE", 2.0f), kRepeatSecondAbove = param("EN_REPEAT_ABOVE", 1.0f), kRepeatFall = param("EN_REPEAT_FALL", 3.0f);
const int kRepeatPauseStops = static_cast<int>(param("EN_REPEAT_PAUSE_STOPS", 3));  // Jyrki 2026-11-10: "to help them help themselves" confusing
const float kRepeatSecondLen = param("EN_REPEAT_LEN", 1.0f);  // the repeated word's length x this
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
const float kFirstAccent = param("EN_FIRST_ACCENT", 11.0f);      // first accent above the baseline
const float kDownstep = param("EN_DOWNSTEP", 0.85f);         // each further accent: this much of the last
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
const int kCommaPauseStops = static_cast<int>(param("EN_COMMA_PAUSE_STOPS", 13));     // 195 ms after a phrase of 5 syllables, growing:
const int kCommaPauseMaxStops = static_cast<int>(param("EN_COMMA_PAUSE_MAX_STOPS", 24));  // at most 360 ms
constexpr int kShortPhraseSyllables = 4; // a phrase this short gets the short pause:
constexpr int kShortCommaPauseStops = 8; // 120 ms
constexpr int kClausePauseStops = 18;    // after ; and : at least 270 ms
constexpr float kPauseVariation = 0.15f; // +- share
const float kFinalLengthSentence = param("kFinalLengthSentence", 1.3f) /* 1.17: the sentence-final word was 1.18x a content word, goog 1.41, 11labs 1.21 */, kFinalLengthPhrase = param("kFinalLengthPhrase", 1.15f);  // final lengthening
constexpr float kFinalLengthClause = 1.25f;  // before ; or :
const float kFunctionWordVowel = param("kFunctionWordVowel", 0.8f);  // vowel length in a function word inside the phrase
constexpr float kFinalCodaSonorant = 3.5f;  // a phrase-final l m n ŋ coda
constexpr float kFinalCodaFricative = 2.7f;  // a phrase-final s ʃ f θ after a vowel or sonorant
const int kSentencePauseStops = static_cast<int>(param("EN_SENTENCE_PAUSE_STOPS", 10));  // + the breath before the next sentence
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
constexpr float kLotInh = 150, kLotMin = 70;  // the short ɑ of "poverty" (AA: 240, 100)
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
bool isApproximant(const std::string& b) { return b == "L" || b == "R" || b == "W" || b == "Y"; }
bool isSonorantCoda(const std::string& b) { return isNasal(b) || b == "L" || b == "R"; }

}  // namespace

// Duration (ms) of every token of the phrase, by Klatt's rules (simplified).
// Tempo by phrase structure (duration multipliers).
constexpr int kFastPhraseSyllables = 15, kShortPhraseTempoSyllables = 6;
constexpr float kLongPhraseTempo = 0.94f, kShortPhraseTempo = 1.06f, kParagraphOpeningTempo = 1.06f;

// Stress timing (on top of Klatt's rules).
const float kUnstressedVowel = param("EN_UNSTRESSED_VOWEL", 0.6f);  // (a knob since 2026-11-07: "Americans" ə 83/78/122 ms against Kokoro's 50/20/70)       // Klatt: 0.5 (0.42: function words 137 ms against goog.wav's 240 on the same text, content words 431 vs 360: runs of function words lurched)
constexpr float kAccentedVowelTiming = 1.2f, kUnstressedVowelTiming = 0.85f;  // stress timing
constexpr float kStressedOnset = 1.2f;         // consonants starting a stressed syllable
constexpr float kUnstressedConsonant = 0.85f;  // other consonants
// All consonants: against goog.wav's reading of the same text (its DTW
// alignment) puhuja's consonants were 20 % short across the board (median
// 58 vs 70 ms, s 86 vs 121, n 51 vs 65, t 45 vs 60), the vowels not (105 vs
// 99): function words, mostly consonants, ran at 137 ms against 240.
// A word-initial nasal before a vowel x this (EN_NASAL_ONSET_LEN; 2026-11-06, Jyrki: "in his mortal hands -- m in mortal
// becomes too weak or short"): Kokoro's word-initial m n run 80-90 ms ("mortal" 90, "Americans" 80, "nation" 80, "any"
// 110) against puhuja's 44-52; its medial ones (50-55 ms) match puhuja's.
const float kOnsetNasalLength = param("EN_NASAL_ONSET_LEN", 1.15f);
const float kEnglishConsonantLength = param("kEnglishConsonantLength", 0.82f) /* 0.70 from the timing search cut every consonant 30 %: nasal murmurs 8-35 ms against Kokoro 40-75, words 20 % short (2026-11-02) */;
// A sonorant cluster across a word boundary ("human life", "in love", "all
// right"): the readers hold the first consonant as the coda of its word and
// make the second a short glide into the vowel (ElevenLabs and Kokoro "human
// life": n 109-120 ms, l 10-40; puhuja had n 72, l 110; Jyrki: "l in the
// space of i, while n remains in the space of a", 2026-10-30).
constexpr float kClusterCodaLength = 1.5f, kClusterOnsetLength = 0.3f;
// l by position: word-initial before a vowel it ran 98 ms against the
// readers' 44-110 (median ~70), word-final after a vowel 77 against 80-115.
constexpr float kOnsetLLength = 0.75f, kCodaLLength = 1.2f;
constexpr float kUnaccentedCodaAffricate = 0.7f;  // ʧ ʤ closing an unaccented word (0.35: "for which our" ʧ 35 ms against the readers' 70, a tick after a long w; Jyrki: /wɪtʃ/ "ends with a ch sound", 2026-10-31)
constexpr float kVoicedHLength = 0.65f;  // h between voiced sounds
// f: kEnglishConsonantLength scales Klatt's 80 ms minimum too, so no f got
// past ~66 ms (Kennedy: median 59 ms, the readers' 100; "forget" 66 against
// 60-90; Jyrki 2026-11-03: "f can be longer in 'forget'").
const float kFLength = param("kFLength", 1.4f);

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
      const bool vowel = isVowelToken(tok[i]);
      const bool inLastSyllable = static_cast<int>(i) >= lastVowel;
      // ɑ before another syllable of its word is the short "lot" vowel
      // ("poverty", "body", "problem"); the long one stays at the word's end
      // ("spa", "car"). Jyrki 2026-11-03: "'poverty' not 'pooverty'".
      static constexpr KlattDur kLotVowel = {"AA", kLotInh, kLotMin};
      const KlattDur& k = b == "AA" && !inLastSyllable ? kLotVowel : klatt(b);
      const bool phraseFinal = static_cast<int>(w) == lastWordWithVowel && inLastSyllable;
      float prc = 1, mn = k.min;
      float whole = 1.0f;  // a factor on the whole duration, minimum included
      if (vowel) {
        const char* name = kEnglishTokens[tok[i] - 0x80];
        // A function word ending the phrase is said in full ("little we can
        // do;", "yes, I can."): its stressed syllable is not shortened.
        const bool stressed = (phrase_[w].accent || phraseFinal) && name[strlen(name) - 1] != '0';
        if (!phraseFinal) prc *= 0.6f;
        // (PROSODY agent11, lab knob; 1: as before) a phrase-final vowel x EN_FINAL_VOWEL_LEN: its last vowel ran 1.50x
        // the same vowel inside the phrase (both voices), the 11labs readers' 1.19 (1.12-1.25).
        { static const float kFinVowelLen = param("EN_FINAL_VOWEL_LEN", 1.0f); if (phraseFinal && kFinVowelLen != 1.0f) prc *= kFinVowelLen; }
        if (!inLastSyllable) prc *= 0.85f;
        if (vowels > 1) prc *= 0.8f;
        if (!stressed) {
          prc *= kUnstressedVowel;
          mn *= kUnstressedVowel;
        }
        // A function word inside the phrase is quicker still: in kennedy.txt
        // puhuja's function words ran 285 ms against the human readers' 240-250
        // and Kokoro's 112-125, while its content words were on the short side
        // (302 against 325-387), 2026-11-02.
        if (!phrase_[w].accent && !phraseFinal) prc *= kFunctionWordVowel;
        // Stress timing: accented vowels longer, unstressed ones shorter (the
        // syllables were too even: nPVI 61, the voice-overs about 76).
        if (phrase_[w].accent && name[strlen(name) - 1] == '1') prc *= kAccentedVowelTiming;
        else if (!stressed) prc *= kUnstressedVowelTiming;
        // The consonant after the vowel (same word) colours its length.
        if (i + 1 < tok.size() && !isVowelToken(tok[i + 1])) {
          const std::string n = baseName(tok[i + 1]);
          // (EN_PRE_ZH_VOWEL: before a ʒ the vowel x this instead of 1.4; "television", "decision": the stressed ɪ ran
          // 240-310 ms against a 50 ms ʒ, Jyrki 2026-11-09: "the vi could be shorter, the ci could be shorter")
          static const float kPreZhVowel = param("EN_PRE_ZH_VOWEL", 0.5f);
          // (EN_AA_PRE_VLESS: the ɑ before a voiceless plosive x this instead of 0.75; 2026-11-10 AA agent: "not, got, hot, stop")
          static const float kAaPreVless = param("EN_AA_PRE_VLESS", 0.75f);
          // (EN_PRE_T_VOWEL, T agent 10 round K, lab; < 0: as the other voiceless plosives) the vowel before a t x this.
          static const float kPreTVowel = param("EN_PRE_T_VOWEL", -1.0f);
          // (EN_PRE_Z_VOWEL: before a z x this instead of 1.4; lab 2026-11-11, the z heard as s: "his" ɪ 237 ms vs "hiss" 158)
          static const float kPreZVowel = param("EN_PRE_Z_VOWEL", 1.4f);
          // (EN_PRE_G_VOWEL, G agent 7 round H, lab; < 0: the voiced plosives' 1.2) the vowel before a g of the same word.
          static const float kPreGVowel = param("EN_PRE_G_VOWEL", -1.0f);
          // (EN_PRE_JH_VOWEL: before a ʤ x this instead of the voiced plosives' 1.2; JH agent round H: a final ʤ is heard as ʧ)
          static const float kPreJhVowel = param("EN_PRE_JH_VOWEL", 1.2f);
          // (EN_PRE_D_VOWEL: before a d x this instead of the voiced plosives' 1.2; < 0: 1.2. D agent9 round J: the d is
          // heard as t 519 times in the dictionary, "beads" -> beats, "trader" -> traitor: the vowel's length is the
          // listener's voicing cue. EN_PRE_D_VOWEL_VV: only before a d between vowels; EN_PRE_D_VOWEL_DZ: before a d + s/z.)
          static const float kPreDVowel = param("EN_PRE_D_VOWEL", -1.0f), kPreDVowelVv = param("EN_PRE_D_VOWEL_VV", -1.0f),
                             kPreDVowelDz = param("EN_PRE_D_VOWEL_DZ", -1.0f);
          if (n == "D") {
            const std::string nn = i + 2 < tok.size() ? baseName(tok[i + 2]) : "";
            const bool vv = i + 2 < tok.size() && isVowelToken(tok[i + 2]), dz = nn == "Z" || nn == "S";
            const float f = vv && kPreDVowelVv >= 0 ? kPreDVowelVv : dz && kPreDVowelDz >= 0 ? kPreDVowelDz : kPreDVowel;
            if (f >= 0) prc *= f / 1.2f;
          }
          prc *= n == "JH" ? kPreJhVowel : n == "ZH" ? kPreZhVowel : n == "Z" ? kPreZVowel : n == "G" && kPreGVowel >= 0 ? kPreGVowel : isVoicedFricative(n) ? 1.4f : isVoicedPlosive(n) ? 1.2f
               : isNasal(n) ? 0.9f : isVoicelessPlosive(n) ? (n == "T" && kPreTVowel >= 0 ? kPreTVowel : b == "AA" ? kAaPreVless : 0.75f) : 1.0f;
          // Z-FINAL agent13 round L (lab; 1: as before): the vowel before a nasal / voiced stop + the word's final z
          // x EN_PRE_NZ_VOWEL / EN_PRE_BZ_VOWEL ("pens" ɛ 209 ms = "pence" 209: the z's voicing cue never reached the
          // vowel across the n; Kokoro's voiced stretch of "pens" "means" "beds" is ~420 ms, puhuja's ~347).
          static const float kPreNzVowel = param("EN_PRE_NZ_VOWEL", 1.0f), kPreBzVowel = param("EN_PRE_BZ_VOWEL", 1.0f);
          if ((kPreNzVowel != 1.0f || kPreBzVowel != 1.0f) && i + 3 == tok.size() && baseName(tok[i + 2]) == "Z")
            prc *= isNasal(n) ? kPreNzVowel : isVoicedPlosive(n) ? kPreBzVowel : 1.0f;
        }
        // (T agent 10 round K, lab; 1: as before) the vowel before a word-final t (also t + s, and with an ɹ l n between:
        // "Cart", "Built", "Brent") x EN_PRE_FINAL_T_VOWEL: the final t is heard as d 238 times in the dictionary ("Gate"
        // -> good, "Mate" -> maid, "Cart" -> card). Kokoro's voiced stretch before a final t is 0.61-0.76 of the same word
        // with d (Mate 280 / Made 420 ms), puhuja's 0.80-0.91 (435 / 510): the clipping above acts on the vowel's
        // variable part only, and not at all across an ɹ l n.
        {
          static const float kPreFinalTVowel = param("EN_PRE_FINAL_T_VOWEL", 1.0f);
          if (kPreFinalTVowel != 1.0f) {
            size_t j = i + 1;
            if (j < tok.size() && (baseName(tok[j]) == "R" || baseName(tok[j]) == "L" || baseName(tok[j]) == "N")) ++j;
            if (j < tok.size() && baseName(tok[j]) == "T" && (j + 1 == tok.size() || (j + 2 == tok.size() && baseName(tok[j + 1]) == "S")))
              whole *= kPreFinalTVowel;
          }
        }
        // Z-FINAL-2 agent19 round Q (lab; 1: as before): the whole vowel before a word-final z (also across an ɹ l n m:
        // "his", "pens") x EN_PRE_FINAL_Z_WHOLE, before a word-final s x EN_PRE_FINAL_S_WHOLE. The voiced stretch is the
        // listener's z / s cue: Kokoro's z words run 1.18-1.22x its s words (adam 372 / 316 ms, bella 333 / 274, heart
        // 354 / 291), puhuja's 1.10 / 1.06 (398 / 361, 361 / 340): EN_PRE_Z_VOWEL acts on the vowel's variable part only.
        // Puhuja's s words are the long ones (Kokoro's z words 333-372 ms, puhuja's 361-398). Lab (263 final-z words,
        // two draws): Z_WHOLE 1.15 74 -> 69.5 errors, +2.2 % duration (the duration cheat); 1.25 87. S_WHOLE untested.
        {
          static const float kPreFinalZWhole = param("EN_PRE_FINAL_Z_WHOLE", 1.0f), kPreFinalSWhole = param("EN_PRE_FINAL_S_WHOLE", 1.0f);
          if (kPreFinalZWhole != 1.0f || kPreFinalSWhole != 1.0f) {
            size_t j = i + 1;
            if (j < tok.size() && (baseName(tok[j]) == "R" || baseName(tok[j]) == "L" || baseName(tok[j]) == "N" || baseName(tok[j]) == "M")) ++j;
            if (j + 1 == tok.size() && baseName(tok[j]) == "Z") whole *= kPreFinalZWhole;
            if (j + 1 == tok.size() && baseName(tok[j]) == "S") whole *= kPreFinalSWhole;
          }
        }
        // The vowel before a ŋ in the same word x EN_PRE_NG_VOWEL_LEN (lab knob 2026-11-11, NG agent 3; 1: as before): the
        // vowel carries the velar pinch (F2 and F3 converging into the ŋ), the cue that tells ŋ from n.
        if (i + 1 < tok.size() && baseName(tok[i + 1]) == "NG") {
          static const float kPreNgVowel = param("EN_PRE_NG_VOWEL_LEN", 1.0f);
          whole *= kPreNgVowel;
        }
        // ... and before a θ x EN_PRE_TH_VOWEL_LEN (TH agent7, round H; 1: as before): the clipping of a vowel before a
        // voiceless consonant ("author" -> "over": the long voiced ɔː told Whisper the next consonant was voiced).
        if (i + 1 < tok.size() && baseName(tok[i + 1]) == "TH") {
          static const float kPreThVowel = param("EN_PRE_TH_VOWEL_LEN", 1.0f);
          whole *= kPreThVowel;
        }
        // A word-final unstressed ə x this (EN_FINAL_SCHWA_LEN, lab knob 2026-11-11, AX agent; 1: as before): before a
        // pause it ran 125-140 ms, heard as u / ɝ in 70 % of the final cells.
        { static const float kFinalSchwaLen = param("EN_FINAL_SCHWA_LEN", 1.0f);
          if (b == "AH" && !stressed && i + 1 == tok.size()) whole *= kFinalSchwaLen; }
        // (round-J ə agent 9, lab knobs; 1: as before) a word-initial unstressed ə x EN_SCHWA_INIT_LEN ("about", "again":
        // 19 % wrong, heard ɪ or lost), one in its word's last syllable before a consonant x EN_SCHWA_LAST_LEN ("Nixon",
        // "Gibson": 11.4 % wrong in two-syllable words, heard ɪ; 124-134 ms phrase-finally), any other medial one x
        // EN_SCHWA_MID_LEN.
        { static const float kSwInitLen = param("EN_SCHWA_INIT_LEN", 1.0f), kSwLastLen = param("EN_SCHWA_LAST_LEN", 1.0f),
                             kSwMidLen = param("EN_SCHWA_MID_LEN", 1.0f);
          if (b == "AH" && !stressed && i + 1 < tok.size()) whole *= i == 0 ? kSwInitLen : inLastSyllable ? kSwLastLen : kSwMidLen; }
        // (AX-NASAL agent14 round M, lab knobs; 1: as before) an unstressed ə before a nasal of its word, not word-initial,
        // x EN_AXN_LEN (the word's first vowel after one consonant, "Commands": x EN_AXC_LEN instead; after t / d,
        // "Sutton": x EN_AXT_LEN instead; the nasal after it: EN_AXN_NN_LEN below). Ten Kokoro voices (k1k words, DTW):
        // the ə before n ran 118-137 ms here against 81-106 (male), 105-123 against 67-88 (female).
        { static const float kAxnLen = param("EN_AXN_LEN", 1.0f), kAxcLen = param("EN_AXC_LEN", 1.0f),
                             kAxtLen = param("EN_AXT_LEN", 1.0f);
          auto axAt = [&](size_t j) {
            if (j == 0 || j + 1 >= tok.size() || baseName(tok[j]) != "AH") return false;
            const char* nm = kEnglishTokens[tok[j] - 0x80];
            const std::string nb = baseName(tok[j + 1]);
            return nm[strlen(nm) - 1] == '0' && (nb == "M" || nb == "N" || nb == "NG");
          };
          if (axAt(i)) {
            const std::string pb = baseName(tok[i - 1]);
            whole *= i == 1 && !isVowelToken(tok[0]) ? kAxcLen : pb == "T" || pb == "D" ? kAxtLen : kAxnLen;
          }
        }
        // The ʊ's own length (EN_UH_LEN; 2026-11-09 UH agent: in isolated words puhuja's ʊ ran 220-260 ms, Kokoro's 150-280).
        { static const float kUhLen = param("EN_UH_LEN", 1.0f); if (b == "UH") whole *= kUhLen; }
        // ... and by its coda in the word (EN_UH_VL_LEN before p t k ʧ s ʃ f θ, EN_UH_L_LEN before l; round-H ʊ agent, lab;
        // 1: as before): Kokoro's ʊ runs 140-210 ms in "Put", "Took", "Push" and 280-320 in "Full", "Pull"; puhuja 226 / 250.
        { static const float kUhVlLen = param("EN_UH_VL_LEN", 1.0f), kUhLLen = param("EN_UH_L_LEN", 1.0f);
          if (b == "UH" && i + 1 < tok.size() && !isVowelToken(tok[i + 1])) {
            const std::string n = baseName(tok[i + 1]);
            static const float kUhRLen = param("EN_UH_R_LEN", 1.0f);  // ... and before ɹ ("sure", "tour")
            if (n == "L") whole *= kUhLLen;
            else if (n == "R") whole *= kUhRLen;
            else if (isVoicelessPlosive(n) || n == "CH" || n == "S" || n == "SH" || n == "F" || n == "TH") whole *= kUhVlLen;
          } }
        // The ɑ's own length (EN_AA_LEN; 2026-11-10 AA agent: a short ɑ is heard as ʌ / ə).
        { static const float kAaLen = param("EN_AA_LEN", 1.4f); if (b == "AA") whole *= kAaLen; }
        // ... and by its place (AA agent 10, round K, lab knobs; 1: as before): EN_AA_FINAL_LEN at the word's end (puhuja's
        // "Pa", "La", "Las" ran 335-425 ms, Kokoro's 175-235: a long open back vowel reads as ɔ, "La" -> law),
        // EN_AA_UNSTR_LEN unstressed (16 % wrong, heard ə), EN_AA_NAS_LEN before a nasal of its word ("On" -> john).
        { static const float kAaFinalLen = param("EN_AA_FINAL_LEN", 1.0f), kAaUnstrLen = param("EN_AA_UNSTR_LEN", 1.0f),
                             kAaNasLen = param("EN_AA_NAS_LEN", 1.0f);
          if (b == "AA") {
            if (i + 1 == tok.size()) whole *= kAaFinalLen;
            if (name[strlen(name) - 1] == '0') whole *= kAaUnstrLen;
            if (i + 1 < tok.size() && isNasal(baseName(tok[i + 1]))) whole *= kAaNasLen;
          } }
        // The ɔ's own length by what follows in its word (AO agent 8, round I, lab knobs; 1: as before): EN_AO_LEN not before
        // an ɹ, EN_AO_R_LEN before it, EN_AO_VL_LEN before a voiceless obstruent (p t k f θ s ʃ ʧ), EN_AO_L_LEN before an l,
        // EN_AO_FINAL_LEN at the word's end. The dictionary's ɔ: 8 % wrong before ɹ, 36 % elsewhere (oʊ 238 of 564).
        {
          static const float kAoLen = param("EN_AO_LEN", 1.0f), kAoRLen = param("EN_AO_R_LEN", 1.0f),
                             kAoVlLen = param("EN_AO_VL_LEN", 1.0f), kAoLLen = param("EN_AO_L_LEN", 1.0f),
                             kAoFinalLen = param("EN_AO_FINAL_LEN", 1.0f);
          if (b == "AO") {
            const std::string n = i + 1 < tok.size() ? baseName(tok[i + 1]) : "";
            if (n == "R") whole *= kAoRLen;
            else {
              whole *= kAoLen;
              if (n.empty()) whole *= kAoFinalLen;
              else if (n == "L") whole *= kAoLLen;
              else if (isVoicelessPlosive(n) || n == "CH" || n == "S" || n == "SH" || n == "F" || n == "TH") whole *= kAoVlLen;
            }
          }
        }
        // The ɛ's own length (EN_EH_LEN), x EN_EH_MONO_LEN more in a word of one vowel, x EN_EH_NASAL_LEN more before a
        // nasal of its word (round E/F EH agents 2026-11-11 lab knobs; 1 / 1 / 1: as before): ɛ heard as æ 784 times in
        // the dictionary (352 before a nasal: "send" -> sand), 38 % wrong in monosyllables; its connected-speech label ran
        // 139 ms against Kokoro's 72 and 11labs' 101, longer than puhuja's own æ words: an American æ length.
        { static const float kEhLen = param("EN_EH_LEN", 1.0f), kEhMonoLen = param("EN_EH_MONO_LEN", 1.0f),
                             kEhNasalLen = param("EN_EH_NASAL_LEN", 1.0f);
          if (b == "EH") {
            whole *= kEhLen;
            if (vowels == 1) whole *= kEhMonoLen;
            if (i + 1 < tok.size() && isNasal(baseName(tok[i + 1]))) whole *= kEhNasalLen;
            // (EN_EH_VPLOS_LEN: x this before a voiced plosive of its word, where the 1.2 lengthening made "bed", "beg"
            // an æ length: æ 100 of the ɛ errors before b d g.)
            static const float kEhVplosLen = param("EN_EH_VPLOS_LEN", 1.0f);
            if (i + 1 < tok.size() && isVoicedPlosive(baseName(tok[i + 1]))) whole *= kEhVplosLen;
            // (round I EH agent 2026-11-12 lab knobs; 1: as before) x EN_EH_R_LEN before an r, x EN_EH_L_LEN before an l
            // of its word: one-syllable ɛr 40 % wrong ("fares" -> fails, "pairs" -> pals), ɛl 19 % ("cells" -> sales).
            static const float kEhRLen = param("EN_EH_R_LEN", 1.0f), kEhLLen = param("EN_EH_L_LEN", 1.0f);
            if (i + 1 < tok.size() && baseName(tok[i + 1]) == "R") whole *= kEhRLen;
            if (i + 1 < tok.size() && baseName(tok[i + 1]) == "L") whole *= kEhLLen;
          } }
        // The stressed ʌ x EN_AH_LEN, in a one-syllable word x EN_AH_MONO_LEN as well (round-F ʌ agent 2026-11-11 lab knobs;
        // 1 / 1: as before; ʌ is the shortest stressed vowel, 140 ms base against ɑ's 240 and æ's 230).
        { static const float kAhLen = param("EN_AH_LEN", 1.0f), kAhMonoLen = param("EN_AH_MONO_LEN", 1.0f);
          if (b == "AH" && stressed && (kAhLen != 1.0f || kAhMonoLen != 1.0f)) {
            int vowels = 0;
            for (size_t q = 0; q < tok.size(); ++q) vowels += isVowelToken(tok[q]);
            whole *= kAhLen * (vowels == 1 ? kAhMonoLen : 1.0f);
          } }
        // ... x EN_AH_VLESS_LEN before a voiceless obstruent of its word, x EN_AH_INIT_LEN word-initial (round-I ʌ agent
        // 2026-11-12 lab knobs; 1 / 1: as before): ʌ is heard as ɑ in "nut", "shut", "duct" (ɑ is the long one), and the
        // word-initial ʌ ("other", "under", "us") 18 % wrong, heard ə ɑ.
        { static const float kAhVlessLen = param("EN_AH_VLESS_LEN", 1.0f), kAhInitLen = param("EN_AH_INIT_LEN", 1.0f);
          if (b == "AH" && stressed) {
            if (kAhVlessLen != 1.0f && i + 1 < tok.size() && !isVowelToken(tok[i + 1])) {
              const std::string n = baseName(tok[i + 1]);
              if (isVoicelessPlosive(n) || n == "S" || n == "SH" || n == "F" || n == "TH") whole *= kAhVlessLen;
            }
            if (i == 0) whole *= kAhInitLen;
          } }
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
        whole *= kEnglishConsonantLength;
        if (b == "F") whole *= kFLength;
        // Across the word boundary: a sonorant coda before an approximant
        // onset is held, the onset becomes a glide into its vowel
        // (kClusterCodaLength, kClusterOnsetLength).
        const bool wordInitial = i == 0, wordFinal = i + 1 == tok.size();
        const bool vowelAfter = i + 1 < tok.size() && isVowelToken(tok[i + 1]);
        const bool vowelBefore = i > 0 && isVowelToken(tok[i - 1]);
        bool clusterOnset = false;
        if (wordFinal && (isNasal(b) || b == "R") && w + 1 < phrase_.size()) {  // (l: kCodaLLength below is enough)
          const auto& nt = phrase_[w + 1].tokens;
          if (nt.size() >= 2 && isApproximant(baseName(nt[0])) && isVowelToken(nt[1])) whole *= kClusterCodaLength;
        }
        if (wordInitial && vowelAfter && isApproximant(b) && w > 0 && !phrase_[w - 1].tokens.empty() &&
            isSonorantCoda(baseName(phrase_[w - 1].tokens.back()))) {
          whole *= kClusterOnsetLength;
          clusterOnset = true;
        }
        // l after a plosive in the same word ("pledge", "please", "black"): 59 ms
        // here, Kokoro's "pledge," l 160 (Jyrki 2026-11-07: "'pledge' is
        // mispronounced, compare with kokoro"); x EN_STOP_L_LEN (1: unchanged).
        if (b == "L" && i > 0 && (isVoicelessPlosive(baseName(tok[i - 1])) || isVoicedPlosive(baseName(tok[i - 1])))) {
          static const float kStopLLen = param("EN_STOP_L_LEN", 1.7f);  // Jyrki: "pledge-l20vr5.wav is the best" (l 59 -> 118 ms)
          whole *= kStopLLen;
        }
        // h after a plosive across the word boundary ("at home"): 119 ms of
        // bright noise after the t, heard as an f ("at phone", Jyrki 2026-11-07);
        // Kokoro's t release carries a weak breath straight into the o. x EN_STOP_H_LEN.
        if (b == "HH" && i == 0 && w > 0 && !phrase_[w - 1].tokens.empty() &&
            (isVoicelessPlosive(baseName(phrase_[w - 1].tokens.back())) || isVoicedPlosive(baseName(phrase_[w - 1].tokens.back())))) {
          static const float kStopHLen = param("EN_STOP_H_LEN", 0.6f);  // Jyrki 2026-11-07: "let's make that the default then own60" (h after a stop 119 -> 71 ms)
          whole *= kStopHLen;
        }
        // w: x EN_W_LEN (1: unchanged). 72 ms here; after "by" its F2 reaches
        // only ~860 Hz from the ɪ's 1900 in that time (Kokoro's w 520-570).
        // ŋ x EN_NG_LEN (lab knob 2026-11-11, NG agent 3; 1: as before).
        if (b == "NG") {
          static const float kNgLen = param("EN_NG_LEN", 1.0f);
          whole *= kNgLen;
        }
        if (b == "W") {
          static const float kWLen = param("EN_W_LEN", 1.0f);
          whole *= kWLen;
          // (W agent round G, lab; 1: as before) a w after a plosive in its word x EN_STOP_W_LEN, after s x EN_S_W_LEN:
          // "twice" released into a 64 ms w (Kokoro's t to vowel ~135 ms), the puff took 36 of it.
          static const float kStopWLen = param("EN_STOP_W_LEN", 1.0f), kSWLen = param("EN_S_W_LEN", 1.0f);
          if (i > 0 && (isVoicelessPlosive(baseName(tok[i - 1])) || isVoicedPlosive(baseName(tok[i - 1])))) whole *= kStopWLen;
          if (i > 0 && baseName(tok[i - 1]) == "S") whole *= kSWLen;
        }
        // ɹ after a plosive in the same word ("proud", "tried", "great"): 53 ms
        // with F3 at 2100-2900 Hz, no r-colour, heard as an l ("proud" ->
        // "pollot", Jyrki 2026-11-07); Kokoro's ɹ 80 ms, F3 1800-2000. x EN_STOP_R_LEN.
        if (b == "R" && i > 0 && (isVoicelessPlosive(baseName(tok[i - 1])) || isVoicedPlosive(baseName(tok[i - 1])))) {
          static const float kStopRLen = param("EN_STOP_R_LEN", 1.7f);  // Jyrki: "proud-r15-cut.wav perhaps 17 would be ideal"
          whole *= kStopRLen;
        }
        // j after a plosive in the same word ("cute", "beauty"; Y agent round G, lab): x EN_STOP_J_LEN (1: unchanged).
        if (b == "Y" && i > 0 && (isVoicelessPlosive(baseName(tok[i - 1])) || isVoicedPlosive(baseName(tok[i - 1])))) {
          static const float kStopJLen = param("EN_STOP_J_LEN", 1.0f);
          whole *= kStopJLen;
        }
        if (isNasal(b) && wordInitial && vowelAfter) whole *= kOnsetNasalLength;  // (EN_NASAL_ONSET_LEN)
        // A word-final voiceless stop linked to a vowel-initial next word ("not out" as "no-tout"): x EN_LINK_STOP_LEN.
        // Jyrki 2026-11-10: "plan for 'notout' with a little longer t".
        if (wordFinal && vowelBefore && isVoicelessPlosive(b) && w + 1 < phrase_.size() && !phrase_[w + 1].tokens.empty() &&
            isVowelToken(phrase_[w + 1].tokens.front())) {
          static const float kLinkStopLen = param("EN_LINK_STOP_LEN", 1.3f);
          whole *= kLinkStopLen;
        }
        if (b == "L" && !clusterOnset) {
          // (EH-AE agent14 round M, lab; 0.75: as before) EN_ONSET_L_LEN: Kokoro's word-initial l is steady for ~60 ms
          // (am_adam, 16 words), puhuja's for ~40 (0.95: 60). Lab: 0.95 cuts the stop heard before the l 308 -> 273 of 680.
          static const float kOnsetLLen = param("EN_ONSET_L_LEN", kOnsetLLength);
          if (wordInitial && vowelAfter) whole *= kOnsetLLen;
          else if (wordFinal && vowelBefore && !phraseFinal) whole *= kCodaLLength;
        }
        // A coda affricate of an unaccented word ("which our"): quick, a
        // tap of hiss (goog.wav "for which our": ʧ 50 ms; here 142, as long
        // and loud as a stressed one, a hiss bump in a run of function words).
        if (!phrase_[w].accent && !phraseFinal && v >= tok.size() && (b == "CH" || b == "JH"))
          whole *= kUnaccentedCodaAffricate;  // (below the Klatt minimum, which prc cannot reach)
        // h before j is one sound, the palatal hiss [ç] ("human"): not
        // shortened as a cluster, and long (goog.wav ~120 ms; here it was 51).
        if (b == "HH" && i + 1 < tok.size() && baseName(tok[i + 1]) == "Y") prc *= kHBeforeJ;
        // h between voiced sounds ("the hand"): a short breathy dip (goog.wav
        // ~40 ms; at the full 92 ms with the l's level it was heard as "land").
        if (b == "HH" && i == 0 && w > 0 && !phrase_[w - 1].tokens.empty() && isVowelToken(phrase_[w - 1].tokens.back()) &&
            i + 1 < tok.size() && isVowelToken(tok[i + 1]))
          whole *= kVoicedHLength;
      }
      // Final lengthening by the strength of the boundary.
      if (phraseFinal)
        prc *= sentenceEnd_ ? kFinalLengthSentence
               : lastEnd_ == ';' || lastEnd_ == ':' ? kFinalLengthClause : kFinalLengthPhrase;
      // A final l, m, n or ŋ before the pause is held: the voice carries on
      // through it (goog.wav "well or ill,": the l ~150 ms, here it was 68).
      if (phraseFinal && !vowel && i > static_cast<size_t>(lastVowel) && (b == "L" || isNasal(b))) {
        // ... a final ŋ x EN_FINAL_NG_LEN on top (lab knob 2026-11-11, NG agent 3; 1: as before).
        static const float kFinalNgLen = param("EN_FINAL_NG_LEN", 1.0f);
        // ... a final m x EN_FINAL_M_LEN (round J, M agent; 1: as before): Kokoro's phrase-final m murmur runs 100-150 ms
        // ("Kim", "Blame", "Seems"), puhuja's 70-80.
        static const float kFinalMLen = param("EN_FINAL_M_LEN", 1.0f);
        // ... a final n x EN_FINAL_N_LEN (round J, N agent 9, lab knob; 1: as before).
        static const float kFinalNLen = param("EN_FINAL_N_LEN", 1.0f);
        prc *= kFinalCodaSonorant * (b == "NG" ? kFinalNgLen : b == "M" ? kFinalMLen : b == "N" ? kFinalNLen : 1.0f);
      }
      // A phrase-final voiceless fricative after a vowel or sonorant is held
      // too (goog/11labs "house.": the s 120-125 ms, here 60: "hou"; after a
      // plosive, "votes", theirs is 30-40 ms, so not then).
      if (phraseFinal && !vowel && i > 0 && i > static_cast<size_t>(lastVowel) && (b == "S" || b == "SH" || b == "F" || b == "TH") &&
          !isVoicelessPlosive(baseName(tok[i - 1])) && !isVoicedPlosive(baseName(tok[i - 1])))
        prc *= kFinalCodaFricative;
      // ... a phrase-final f x this on top (EN_FINAL_F_LEN; Jyrki 2026-11-08: "word ending f can be a little
      // shorter, perhaps 20 % shorter, like 'life' or 'belief'": 264 and 249 ms here).
      if (phraseFinal && !vowel && i > 0 && i > static_cast<size_t>(lastVowel) && b == "F" && i + 1 == tok.size()) {
        static const float kFinalFLen = param("EN_FINAL_F_LEN", 0.8f);
        whole *= kFinalFLen;
      }
      // ... and a phrase-final θ x this (EN_FINAL_TH_LEN; lab 2026-11-09: "month" 77 ms against Kokoro's ~80 but weak).
      if (phraseFinal && !vowel && i > 0 && i > static_cast<size_t>(lastVowel) && b == "TH" && i + 1 == tok.size()) {
        static const float kFinalThLen = param("EN_FINAL_TH_LEN", 1.0f);
        whole *= kFinalThLen;
      }
      // ... and a phrase-final p x this (EN_FINAL_P_LEN, lab knob 2026-11-10, P agent: final p 50 % errors; 1: as before).
      if (phraseFinal && !vowel && i > 0 && i > static_cast<size_t>(lastVowel) && b == "P" && i + 1 == tok.size()) {
        static const float kFinalPLen = param("EN_FINAL_P_LEN", 1.6f);
        whole *= kFinalPLen;
      }
      // ... and a phrase-final k (EN_FINAL_K_LEN, lab knob 2026-11-10, K agent 2: final k 31 % errors, heard as t d; 1: as before).
      if (phraseFinal && !vowel && i > 0 && i > static_cast<size_t>(lastVowel) && b == "K" && i + 1 == tok.size()) {
        static const float kFinalKLen = param("EN_FINAL_K_LEN", 1.0f);
        whole *= kFinalKLen;
      }
      // ... and a phrase-final d after a vowel ("Bad.", "Good."; EN_FINAL_D_LEN, lab knob 2026-11-10, D agent): 47 ms here,
      // Kokoro's closure 12-22 ms with a soft release (1: as before).
      if (phraseFinal && !vowel && i > 0 && i > static_cast<size_t>(lastVowel) && b == "D" && i + 1 == tok.size()) {
        static const float kFinalDLen = param("EN_FINAL_D_LEN", 1.0f);
        whole *= kFinalDLen;
      }
      // (N agent13 round L, lab knobs; 1: as before) a word-final n after an unstressed vowel x EN_FN_N_LEN, that vowel x
      // EN_FN_V_LEN: Kokoro's -in / -en ("Gavin", "Griffin") has a short vowel (~90 ms) and a long murmur (130-200 ms);
      // puhuja's vowel 123 ms, the n 71.
      {
        static const float kFnNLen = param("EN_FN_N_LEN", 1.0f), kFnVLen = param("EN_FN_V_LEN", 1.0f);
        auto unstressedAt = [&](size_t j) {
          const char* nm = kEnglishTokens[tok[j] - 0x80];
          return isVowelToken(tok[j]) && nm[strlen(nm) - 1] == '0';
        };
        if (b == "N" && i + 1 == tok.size() && i > 0 && unstressedAt(i - 1)) whole *= kFnNLen;
        if (vowel && i + 2 == tok.size() && baseName(tok[i + 1]) == "N" && unstressedAt(i)) whole *= kFnVLen;
      }
      // (D-AFTER-N agent15 round N, lab; 1: as before) the vowel before a nasal + its word's final d x EN_ND_V_LEN.
      {
        static const float kNdVLen = param("EN_ND_V_LEN", 1.0f);
        if (kNdVLen != 1.0f && vowel && i + 3 == tok.size() && isNasal(baseName(tok[i + 1])) && baseName(tok[i + 2]) == "D")
          whole *= kNdVLen;
        // ... and in the phrase's last word only (EN_ND_V_PF_LEN; the "and" / "end" inside a phrase keep their vowel).
        static const float kNdVPfLen = param("EN_ND_V_PF_LEN", 1.0f);
        if (kNdVPfLen != 1.0f && vowel && i + 3 == tok.size() && isNasal(baseName(tok[i + 1])) && baseName(tok[i + 2]) == "D" &&
            w + 1 == phrase_.size())
          whole *= kNdVPfLen;
      }
      // (AX-NASAL agent14 round M, lab knob; 1: as before) a nasal after an unstressed ə of its word x EN_AXN_NN_LEN
      // (Kokoro's murmur after a reduced vowel is long: "Gavin" 130-200 ms against puhuja's 71).
      { static const float kAxnNnLen = param("EN_AXN_NN_LEN", 1.0f);
        if (kAxnNnLen != 1.0f && !vowel && i > 1 && baseName(tok[i - 1]) == "AH") {
          const std::string b0 = baseName(tok[i]);
          const char* nm = kEnglishTokens[tok[i - 1] - 0x80];
          if ((b0 == "M" || b0 == "N" || b0 == "NG") && nm[strlen(nm) - 1] == '0') whole *= kAxnNnLen;
        } }
      // A phrase-final v after a vowel ("Have."): 50 ms, Kokoro holds ~150-200
      // of fading voiced frication (2026-11-04, EN_FINAL_V_LEN; 1 = today's).
      if (phraseFinal && !vowel && i > static_cast<size_t>(lastVowel) && b == "V") {
        static const float vLen = param("EN_FINAL_V_LEN", 1.0f);
        whole *= vLen;
      }
      // A phrase-final z after a vowel or sonorant x this (EN_FINAL_Z_LEN, lab knob 2026-11-11; 1: as before): 59 ms here,
      // and its slit holds ~70 ms more into the pause with the breath decaying ("his": ~140 ms of voiceless hiss at -6 dB
      // re the word's peak, an s); Kokoro's final z hisses 70-100 ms at -17..-20 dB.
      // Z-FINAL-2 agent19 round Q: since round O's sink (LINE_FZ_SINK_*) and LINE_FINAL_Z_FRIC 0.35 the hiss is at Kokoro's
      // level with ~30-40 ms of voice under it; 1.2: the hiss 136 / 138 ms (m / f; Kokoro 119-138), and the lab (263
      // final-z words, two independent draw pairs) 160 -> 149.5 errors, with LINE_FZ_SINK_N 2 142 (vowel + z cell 105 -> 86).
      if (phraseFinal && !vowel && i > 0 && i > static_cast<size_t>(lastVowel) && b == "Z" &&
          !isVoicelessPlosive(baseName(tok[i - 1])) && !isVoicedPlosive(baseName(tok[i - 1]))) {
        static const float kFinalZLen = param("EN_FINAL_Z_LEN", 1.2f);
        whole *= kFinalZLen;
      }
      // The voiceless fricatives' and stops' lengths x these (Jyrki 2026-11-05,
      // the blend exploration: Kokoro's Kennedy has s 80 ms, ʃ 110, t 60, k 70,
      // p 70 against puhuja's 70, 73, 37, 43, 49; 1: unchanged).
      // The first of two adjacent voiceless fricatives ("beliefs": f 115 ms + s
      // 63; Kokoro's f 27 + s 53, 2026-11-07, Jyrki: "the s sounds a bit odd")
      // x this (1: unchanged).
      if (!vowel && (b == "S" || b == "SH" || b == "F" || b == "TH") && i + 1 < tok.size()) {
        const std::string nb = baseName(tok[i + 1]);
        if (nb == "S" || nb == "SH" || nb == "F" || nb == "TH") {
          static const float kFricClusterLen = param("EN_FRIC_CLUSTER_LEN", 0.7f);
          whole *= kFricClusterLen;
        }
      }
      // A z after a consonant ("holds", "hands"): 47 ms here, Kokoro's >= 70
      // (Jyrki 2026-11-07: "the s/z in the end of 'holds' could be stronger").
      if (!vowel && b == "Z" && i > 0 && !isVowelToken(tok[i - 1])) {
        static const float kClusterZLen = param("EN_CLUSTER_Z_LEN", 1.35f);  // Jyrki 2026-11-07: "holds-Lag is best, but perhaps 15 % less of that", then "holds-Lag70.wav is better than 85": 1.5 and LINE_AG_Z 0.22 with 30 % of the step back
        whole *= kClusterZLen;
      }
      if (!vowel) {
        // A voiced stop before a fricative ("and foe", "and foo", "United there"):
        // 40 ms here, in which the seal never completed (the level stayed at
        // the vowel's) and the release popped into the f; Kokoro's d 70 ms with
        // a silent closure (Jyrki 2026-11-07: "end of and is very strange
        // blib"). x EN_VSTOP_FRIC_LEN (1: unchanged).
        if (isVoicedPlosive(b)) {
          std::string nb;
          if (i + 1 < tok.size()) nb = baseName(tok[i + 1]);
          else if (w + 1 < phrase_.size() && !phrase_[w + 1].tokens.empty()) nb = baseName(phrase_[w + 1].tokens[0]);
          if (nb == "S" || nb == "SH" || nb == "F" || nb == "TH" || isVoicedFricative(nb)) {
            static const float kVstopFricLen = param("EN_VSTOP_FRIC_LEN", 1.0f);
            whole *= kVstopFricLen;
          }
        }
        // A voiced stop opening a stressed syllable after an unstressed one in
        // the same word ("undoing", "about", "begin"): Kokoro's d in "undoing"
        // is 90 ms with a 40 ms dip, puhuja's 39 (Jyrki 2026-11-07: "a small
        // timing delay between un-doing"); x EN_STRESS_ONSET_VSTOP_LEN (1: unchanged).
        size_t pv = i;  // the previous vowel of the word, past any consonants ("un-d")
        while (pv > 0 && !isVowelToken(tok[pv - 1])) --pv;
        if (isVoicedPlosive(b) && pv > 0 && i + 1 < tok.size() && isVowelToken(tok[i + 1]) &&
            kEnglishTokens[tok[pv - 1] - 0x80][strlen(kEnglishTokens[tok[pv - 1] - 0x80]) - 1] == '0' &&
            kEnglishTokens[tok[i + 1] - 0x80][strlen(kEnglishTokens[tok[i + 1] - 0x80]) - 1] == '1') {
          static const float kStressOnsetVstopLen = param("EN_STRESS_ONSET_VSTOP_LEN", 1.0f);
          whole *= kStressOnsetVstopLen;
        }
        static const float vfricLen = param("EN_VFRIC_LEN", 1.4f) /* cost Whisper 898 composite 24.95/28.30 95d-all3 */, vstopLen = param("EN_VSTOP_LEN", 1.3f);
        if (b == "S" || b == "SH" || b == "F" || b == "TH") whole *= vfricLen;
        // A ʃ before a stressed vowel (SH agent 10, round K; 1: unchanged): EN_SH_STRESS_LEN, word-initial
        // EN_SH_INIT_LEN instead (< 0: the same). Puhuja's initial ʃ was 122-132 ms ("Shall", "Short"), Kokoro's 80-90.
        if (b == "SH" && i + 1 < tok.size() && isVowelToken(tok[i + 1])) {
          const char* nv = kEnglishTokens[tok[i + 1] - 0x80];
          const char st = nv[strlen(nv) - 1];
          static const float kShStressLen = param("EN_SH_STRESS_LEN", 1.0f), kShInitLen = param("EN_SH_INIT_LEN", -1.0f);
          if (st == '1' || st == '2') whole *= i == 0 && kShInitLen >= 0 ? kShInitLen : kShStressLen;
        }
        else if (isVoicelessPlosive(b)) whole *= vstopLen;
        else if (b == "D") {
          static const float kDLen = param("EN_D_LEN", 1.0f);  // (lab knob 2026-11-10, D agent)
          whole *= kDLen;
          // D agent9 round J lab knobs (1: as before): the d between a vowel and an unstressed vowel of its word
          // (EN_D_VV_LEN; "reader" -> creator, "trader" -> traitor: puhuja's closure 40 ms at -27 dB, Kokoro's tap
          // 25 ms at -17), a d before s z (EN_DZ_D_LEN), a d after a nasal or l at the word's end or before its s z
          // (EN_ND_D_LEN; "cleaned" -> clean, "finds" -> fines).
          static const float kDVvLen = param("EN_D_VV_LEN", 0.7f), kDzDLen = param("EN_DZ_D_LEN", 1.0f), kNdDLen = param("EN_ND_D_LEN", 1.0f);
          const std::string pb = i > 0 ? baseName(tok[i - 1]) : "", nb = i + 1 < tok.size() ? baseName(tok[i + 1]) : "";
          if (i > 0 && isVowelToken(tok[i - 1]) && i + 1 < tok.size() && isVowelToken(tok[i + 1]) &&
              kEnglishTokens[tok[i + 1] - 0x80][strlen(kEnglishTokens[tok[i + 1] - 0x80]) - 1] == '0') whole *= kDVvLen;
          if (nb == "Z" || nb == "S") whole *= kDzDLen;
          if ((isNasal(pb) || pb == "L") && (nb.empty() || nb == "Z" || nb == "S")) whole *= kNdDLen;
          // (INITIAL-D agent18 round P, lab knob; 1: as before) a word-initial d before a vowel (puhuja.cc LINE_DI_*).
          static const float kDiLen = param("EN_DI_LEN", 1.0f);
          if (i == 0 && i + 1 < tok.size() && isVowelToken(tok[i + 1])) whole *= kDiLen;
        }
        else if (b == "G") {  // the g's own length, initial / medial / final (G agent 7 round H, lab knobs; 1: unchanged)
          static const float kGLen = param("EN_G_LEN", 1.0f), kGInitLen = param("EN_G_INIT_LEN", 1.0f), kGFinalLen = param("EN_G_FINAL_LEN", 1.0f);
          whole *= kGLen * (i == 0 ? kGInitLen : i + 1 == tok.size() ? kGFinalLen : 1.0f);
        }
        // b d g before l ɹ (EN_VSTOP_LIQ_LEN) and the l ɹ after one (EN_LIQ_AFTER_VSTOP_LEN): the cluster's timing
        // (BDG agent 2026-11-11; dictionary: b d g before l/ɹ 44 % errors against 26 % overall; 1: unchanged).
        {
          static const float kVstopLiqLen = param("EN_VSTOP_LIQ_LEN", 1.0f), kLiqAfterVstopLen = param("EN_LIQ_AFTER_VSTOP_LEN", 1.0f);
          if (isVoicedPlosive(b) && i + 1 < tok.size()) {
            const std::string n = baseName(tok[i + 1]);
            if (n == "L" || n == "R") whole *= kVstopLiqLen;
          }
          if ((b == "L" || b == "R") && i > 0 && isVoicedPlosive(baseName(tok[i - 1]))) whole *= kLiqAfterVstopLen;
        }
        // ... and the dentals on top (EN_TH_LEN): puhuja's initial θ ran 90-115 ms against Kokoro's 55-65 (its s 130
        // against 70-80), as long and as loud as an s, and the dictionary hears 68 % of θ as s or z (2026-11-09).
        { static const float kZhLen = param("EN_ZH_LEN", 2.0f); if (b == "ZH") whole *= kZhLen; }
        { static const float kVLen = param("EN_V_LEN", 1.8f); if (b == "V") whole *= kVLen; }
        // The z's own length (EN_Z_LEN, all z) and between two vowels of a word (EN_MED_Z_LEN, "easy", "music"); lab
        // knobs 2026-11-11 (1: unchanged): the medial z ran 50-56 ms (Kokoro 70-80) and was heard as t (164 of the
        // dictionary's medial z errors).
        { static const float kZLen = param("EN_Z_LEN", 1.0f), kMedZLen = param("EN_MED_Z_LEN", 1.3f);
          if (b == "Z") whole *= kZLen;
          if (b == "Z" && i > 0 && isVowelToken(tok[i - 1]) && i + 1 < tok.size() && isVowelToken(tok[i + 1])) whole *= kMedZLen; }
        // (VFRIC-INITIAL agent17 round O, lab; 1: off) a word-initial v / z before a vowel x EN_VZ_INIT_LEN (puhuja's initial
        // v 95-100 ms with its pre-voice, z 65-70; Kokoro am_adam v 75, z 70).
        { static const float kVzInitLen = param("EN_VZ_INIT_LEN", 1.0f);
          if (kVzInitLen != 1.0f && i == 0 && (b == "V" || b == "Z") && i + 1 < tok.size() && isVowelToken(tok[i + 1])) whole *= kVzInitLen; }
        // D agent9 round J lab knobs (1: as before): a nasal or l before a d at its word's end or before the word's
        // final s z x EN_ND_N_LEN (Kokoro's n of "cleaned", "band" 80-140 ms, puhuja's 60: "cleaned" -> clean),
        // a z after a d x EN_DZ_Z_LEN ("beads" -> beats).
        { static const float kNdNLen = param("EN_ND_N_LEN", 1.0f), kDzZLen = param("EN_DZ_Z_LEN", 1.0f);
          if ((isNasal(b) || b == "L") && i + 1 < tok.size() && baseName(tok[i + 1]) == "D" &&
              (i + 2 == tok.size() || (i + 3 == tok.size() && (baseName(tok[i + 2]) == "Z" || baseName(tok[i + 2]) == "S"))))
            whole *= kNdNLen;
          // (D-AFTER-N agent15 round N, lab; 1: as before) the same for a nasal before the final d of its phrase's last
          // word only (EN_ND_N_PF_LEN; the "and" inside a phrase keeps its n): Kokoro's n there 100-140 ms, puhuja's 60.
          static const float kNdNPfLen = param("EN_ND_N_PF_LEN", 1.0f);
          if (kNdNPfLen != 1.0f && isNasal(b) && i + 2 == tok.size() && baseName(tok[i + 1]) == "D" && w + 1 == phrase_.size())
            whole *= kNdNPfLen;
          if (b == "Z" && i > 0 && baseName(tok[i - 1]) == "D") whole *= kDzZLen; }
        // Z-FINAL agent13 round L (lab; 1: as before): a nasal / l / voiced stop right before the word's final z
        // x EN_NZ_N_LEN / EN_BZ_B_LEN (the n of "pens" 63 ms = the n of "pence"; Kokoro's voiced stretch before
        // the hiss is ~70 ms longer than puhuja's after a nasal or a voiced stop), and the final z after a voiced
        // stop x EN_BZ_Z_LEN (EN_FINAL_Z_LEN leaves it alone). Lab (245 hard final-z words, both voices, two draws;
        // base 125 final-z errors, placebos 108.5 / 124): EN_NZ_N_LEN 1.5 103, EN_BZ_Z_LEN 0.8 100, both (0.65, 1.5) 83
        // with the total length unchanged (the z after b d g 73 -> 47 ms; 11labs's z after d in Kennedy 40-50 ms).
        { static const float kNzNLen = param("EN_NZ_N_LEN", 1.5f), kBzBLen = param("EN_BZ_B_LEN", 1.0f),
                             kBzZLen = param("EN_BZ_Z_LEN", 0.65f);
          if (i + 2 == tok.size() && baseName(tok[i + 1]) == "Z") {
            if (isNasal(b) || b == "L") whole *= kNzNLen;
            else if (isVoicedPlosive(b)) whole *= kBzBLen;
          }
          if (b == "Z" && i + 1 == tok.size() && i > 0 && isVoicedPlosive(baseName(tok[i - 1]))) whole *= kBzZLen; }
        { static const float kPLen = param("EN_P_LEN", 1.0f); if (b == "P") whole *= kPLen; }
        { static const float kKLen = param("EN_K_LEN", 1.0f); if (b == "K") whole *= kKLen; }
        // t's own (T agent 3, lab knobs 2026-11-11; 1: unchanged): its length, a phrase-final t's, a t before s ("its",
        // "rates": t heard as k 230 of 1956).
        if (b == "T") {
          static const float kTLen = param("EN_T_LEN", 1.0f), kFinalTLen = param("EN_FINAL_T_LEN", 1.0f), kTsLen = param("EN_T_S_LEN", 1.0f);
          whole *= kTLen;
          if (phraseFinal && i > static_cast<size_t>(lastVowel) && i + 1 == tok.size()) whole *= kFinalTLen;
          if (i + 1 < tok.size() && baseName(tok[i + 1]) == "S") whole *= kTsLen;
          // (T agent 10 round K, lab; 1: as before) a word-final t after p k ("Looked" -> look, "Jumped" -> jump: 23-29 %
          // deleted) x EN_T_AFTER_STOP_LEN.
          static const float kTAfterStopLen = param("EN_T_AFTER_STOP_LEN", 1.5f);
          if (i > 0 && i + 1 == tok.size() && (baseName(tok[i - 1]) == "K" || baseName(tok[i - 1]) == "P")) whole *= kTAfterStopLen;
        }
        // (T agent 10 round K, lab; 1: as before) an ɹ l n between a vowel and a word-final t x EN_PRE_T_SON_LEN (the
        // voicing cue of "Cart" -> card, "Built" -> build, "Brent" -> brand lies in the sonorant too).
        {
          static const float kPreTSonLen = param("EN_PRE_T_SON_LEN", 1.0f);
          if (kPreTSonLen != 1.0f && (b == "R" || b == "L" || b == "N") && i > 0 && isVowelToken(tok[i - 1]) && i + 1 < tok.size() &&
              baseName(tok[i + 1]) == "T" && (i + 2 == tok.size() || (i + 3 == tok.size() && baseName(tok[i + 2]) == "S")))
            whole *= kPreTSonLen;
        }  // the k's own length (K lab 2026-11-10; 1: unchanged)  // the p's own length (P lab 2026-11-10; 1: unchanged)  // the v's own length (lab 2026-11-09; 1: unchanged)  // the ʒ's own length (2026-11-09: 50 ms against Kokoro's ~100)
        { static const float kThLen = param("EN_TH_LEN", 0.6f), kDhLen = param("EN_DH_LEN", 0.85f); if (b == "TH") whole *= kThLen; else if (b == "DH") whole *= kDhLen; }
        // θ's length by context (TH agent7, round H; 1: unchanged): word-initial (EN_INIT_TH_LEN), between two vowels of a
        // word (EN_MED_TH_LEN: "author" -> "over", "nothing": V the commonest medial confusion, 45 % errors before an
        // unstressed vowel), before an unstressed vowel (EN_TH_UNSTR_LEN), before an s (EN_TH_S_LEN: "paths" -> "pass").
        if (b == "TH") {
          static const float kInitThLen = param("EN_INIT_TH_LEN", 1.0f), kMedThLen = param("EN_MED_TH_LEN", 1.0f),
                             kThUnstrLen = param("EN_TH_UNSTR_LEN", 1.0f), kThSLen = param("EN_TH_S_LEN", 1.0f);
          if (i == 0) whole *= kInitThLen;
          if (i > 0 && isVowelToken(tok[i - 1]) && i + 1 < tok.size() && isVowelToken(tok[i + 1])) whole *= kMedThLen;
          if (i + 1 < tok.size() && isVowelToken(tok[i + 1]) &&
              kEnglishTokens[tok[i + 1] - 0x80][strlen(kEnglishTokens[tok[i + 1] - 0x80]) - 1] == '0') whole *= kThUnstrLen;
          if (i + 1 < tok.size() && baseName(tok[i + 1]) == "S") whole *= kThSLen;
        }  // (ð separately: shortened with the θ it lost, 2026-11-09)
        // ð's length word-initially (EN_DH_INIT_LEN: the, this, that) and between vowels (EN_DH_MED_LEN: other, either); DH
        // agent7, round H (1: unchanged). Kokoro's ð in kennedy runs 50 ms (median), puhuja's 35.
        { static const float kDhInitLen = param("EN_DH_INIT_LEN", 1.0f), kDhMedLen = param("EN_DH_MED_LEN", 1.0f);
          if (b == "DH" && i == 0) whole *= kDhInitLen;
          if (b == "DH" && i > 0 && isVowelToken(tok[i - 1]) && i + 1 < tok.size() && isVowelToken(tok[i + 1])) whole *= kDhMedLen; }
        // ... and h's (EN_H_LEN; Jyrki 2026-11-06: "hand of God" sounds too close to "land of God", make h here
        // more clear -- consider noise shaping, making it longer, or slightly stronger). Kokoro's h runs ~60 ms
        // against puhuja's 41.
        static const float hLen = param("EN_H_LEN", 2.0f);
        if (b == "HH") whole *= hLen;
        // An h before j ("human", "huge") x EN_HJ_LEN, the j after it x EN_HJ_J_LEN (1: as before; HJ agent16 2026-11-14,
        // "human" heard "fjuman"): 11labs' eight Kennedy "human"s breathe 20-140 ms (mostly 50-80) of palatal hiss, then
        // 10-70 ms of j that is already the vowel's onset; puhuja's in-phrase h ran 189 ms and its j 55 ms on top.
        static const float kHjLen = param("EN_HJ_LEN", 0.6f), kHjJLen = param("EN_HJ_J_LEN", 1.0f);
        if (b == "HH" && i + 1 < tok.size() && baseName(tok[i + 1]) == "Y") whole *= kHjLen;
        if (b == "Y" && i > 0 && baseName(tok[i - 1]) == "HH") whole *= kHjJLen;
        // ... and f's on top of EN_VFRIC_LEN (EN_F_LEN; Jyrki 2026-11-06: the f of "forms" "slightly artificial": Kokoro's
        // 160 ms against puhuja's 94, with the l's voice tailing into it and a flat spectrum to 8 kHz).
        // Only an f starting a stressed syllable: Kokoro's typical f is short and dark, its stressed one long and bright.
        static const float fLen = param("EN_F_LEN", 1.15f), fLenUnstr = param("EN_F_LEN_UNSTR", 0.75f);
        if (b == "F") {
          size_t v = i + 1;
          while (v < tok.size() && !isVowelToken(tok[v])) ++v;
          const char* vn = v < tok.size() ? kEnglishTokens[tok[v] - 0x80] : nullptr;
          const bool fStressed = vn && phrase_[w].accent && vn[strlen(vn) - 1] == '1';
          whole *= fStressed ? fLen : fLenUnstr;
        }
      }
      // A vowel after a stop + l / r onset in its word ("black", "price", "great") x EN_CLUSTER_VOWEL_LEN (1: as before):
      // the cluster takes time the vowel gives back. Jyrki 2026-11-11: "black should be shorter, not much longer or
      // equal with cat and back" (black 395 ms: b 49, l 111 (EN_STOP_L_LEN 2, his "pledge" pick), æ 206; cat ~310).
      {
        static const float kClusterVowelLen = param("EN_CLUSTER_VOWEL_LEN", 0.76f);
        if (vowel && kClusterVowelLen != 1.0f && i >= 2) {
          const std::string l1 = baseName(tok[i - 1]), p0 = baseName(tok[i - 2]);
          if ((l1 == "L" || l1 == "R") && (isVoicelessPlosive(p0) || isVoicedPlosive(p0))) whole *= kClusterVowelLen;
        }
        // (W agent round G, lab; 1: as before) ... and after a stop / s + w ("twice", "swim") x EN_CLUSTER_W_VOWEL_LEN.
        static const float kClusterWVowelLen = param("EN_CLUSTER_W_VOWEL_LEN", 1.0f);
        if (vowel && kClusterWVowelLen != 1.0f && i >= 2 && baseName(tok[i - 1]) == "W") {
          const std::string p0 = baseName(tok[i - 2]);
          if (isVoicelessPlosive(p0) || isVoicedPlosive(p0) || p0 == "S") whole *= kClusterWVowelLen;
        }
      }
      // Rate: vowels follow it, consonants less (a slow reader keeps the
      // consonants crisp); relative to the default rate, which stays as fitted.
      const float r = rate_ / kDefaultRate;
      const float rateDiv = kDefaultRate * (vowel ? r : std::sqrt(r));
      out[w][i] = (mn + (k.inh - mn) * prc) * whole * kEnglishTempo * structure / rateDiv;
      if (phrase_[w].repeat == 2) out[w][i] *= kRepeatSecondLen;  // (EN_REPEAT_LEN)
      { static const float kEmoLen = param("EN_EMOTIVE_LEN", 1.25f); if (kEmoLen != 1.0f && isEmotiveWord(phrase_[w].text)) out[w][i] *= kEmoLen; }
      // The breath plan's tempo for this word (planBreaths), as the rate.
      const float g = w < wordTempo_.size() ? wordTempo_[w] : 1.0f;
      if (g != 1.0f) out[w][i] /= vowel ? g : std::sqrt(g);
      // An onset ɹ before a vowel is at least this long (ms): in "the same
      // revolutionary beliefs" every factor above left it 16 ms, the shortest
      // r of the paragraph (the others 39-71), and the tongue never reached
      // the r shape (F3 2500 Hz, Kokoro 1660 at 40 ms). Jyrki 2026-11-09:
      // "r in 'revolution' should be stronger ... longer or louder or more
      // active -- I don't know what I need".
      if (!vowel && baseName(tok[i]) == "R" && i + 1 < tok.size() && isVowelToken(tok[i + 1])) {
        static const float kOnsetRMinMs = param("EN_ONSET_R_MIN_MS", 90.0f);
        if (kOnsetRMinMs > 0) out[w][i] = std::max(out[w][i], kOnsetRMinMs);
      }
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
// ... and between words before a phrase opener below this.
constexpr float kGoodBreathAir = 700;
// Air a syllable takes (cm^3, from the calibrated plan: ~90) and the reserve
// kept beyond the rest of the phrase; no breath with this many words left.
constexpr float kAirPerSyllable = 90, kBreathMargin = 150;
constexpr float kCommaBreathBelow = 900;  // (the synth's own threshold: above it no breath is taken)
constexpr size_t kNoBreathLastWords = 2;
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

// Words that open a phrase: a breath may be taken before them.
bool isBreathOpener(const std::string& w) {
  static const char* const kOpeners[] = {"the", "a", "an", "this", "that", "these", "those", "his", "her",
                                         "their", "our", "my", "your", "its", "and", "but", "or", "nor",
                                         "so", "yet", "which", "who", "when", "where", "if", "because",
                                         "while", "as", "than", "to", "for", "from", "with", "in", "on",
                                         "at", "by", "into", "about", "through", "over", "under", "after",
                                         "before", "between", "against", "without"};
  for (const char* o : kOpeners)
    if (w == o) return true;
  return false;
}
// ... but not right after these (they lean on the next word).
bool isNoSplitAfter(const std::string& w) {
  // (A preposition is fine before it: "from | the hand of God" is where the
  // breath belongs; "of" and "to" are not.)
  static const char* const kAfter[] = {"of", "the", "a", "an", "to", "my", "your", "our",
                                       "their", "his", "her", "its", "and", "or", "but", "that", "which",
                                       "any", "every", "some", "no", "each", "all", "both", "such",
                                       "who", "when", "where", "if", "because", "while", "as", "than", "so"};
  for (const char* o : kAfter)
    if (w == o) return true;
  return false;
}

void EnglishReader::addWord(const std::string& raw) {
  size_t a = raw.find_first_not_of('\''), b = raw.find_last_not_of('\'');
  if (a == std::string::npos) return;
  std::string w = raw.substr(a, b - a + 1);
  // Respellings: "puhuja" is read as "puhuyah" (Jyrki 2026-11-10: "Puhuja will be pronounced better if puhuja tries to
  // pronounce 'puhuyah'"; the letter rules gave P UW1 HH UW2 Y AH0).
  if (w == "puhuja") w = "puhuyah";
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
      splitTail_ = tail;
      splitTail_.push_back({t, !isFunctionWord(w), w});  // the word being added belongs to the tail too
      sayPhrase(',');
      splitTail_.clear();
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
                                       contrast ||
                                       (phrase_.size() >= kForceSplitWords && !isNoSplitAfter(phrase_.back().text)))) {
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
  // Readings where CMUdict's first variant is not the usual American one.
  // ("because": Kokoro and most speakers say bɪˈkʌz; CMUdict's first is -ɔːz,
  // and the word scored 0.046 against the Kokoro voices where a human
  // reading scored 0.032, 2026-10-30.)
  // ("every": the dictionary's first reading has three syllables, ɛvəri;
  // speakers say ɛvri, its second reading. Jyrki, 2026-10-30: "I hear ɛvɛri".)
  static const Homograph kReadings[] = {{"because", "B IH0 K AH1 Z"}, {"every", "EH1 V R IY0"}};
  for (const Homograph& r : kReadings)
    if (w == r.word) t = arpaTokens(r.verb);
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
      if (w[k] >= 'a' && w[k] <= 'z') {
        phrase_.push_back({arpaTokens(kLetterNames[w[k] - 'a']), k + 1 == w.size(), std::string(1, w[k])});
        phrase_.back().source = sourceWord_;
      }
    return;
  }
  if (!t.empty()) {
    phrase_.push_back({std::move(t), !isFunctionWord(w), w});
    phrase_.back().source = sourceWord_;
  }
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
    // (his, her keep their h: the readers say "in his" with the h, 160-180 ms;
    // without it the word was heard as too fast, Jyrki 2026-11-02.)
    {"just", "JH AH0 S T"}, {"us", "AH0 S"}, {"his", "HH IH0 Z"}, {"her", "HH ER0"}};

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
const float kArcStartDb = param("EN_ARC_START_DB", 1.0f), kArcEndDb = param("EN_ARC_END_DB", -2.0f);  // (-3: with the end release too steep)
constexpr float kCreakBelow = -6.5f;  // st: statement endings this low get creaky voice (-5: nearly all)
constexpr float kPressedAccent = 0.9f, kLaxUnstressed = 1.08f, kLaxFinal = 1.12f;  // glottal opening
constexpr float kAccentDb = 1.5f, kFunctionWordDb = -1.5f;
const float kSecondaryDb = param("kSecondaryDb", -4.0f), kUnstressedDb = param("kUnstressedDb", 2.0f);
const float kSecondaryPreDb = param("kSecondaryPreDb", 3.0f);  // a secondary stress before the primary (see the level rule; Jyrki 2026-11-06: "gen-sp4 is the best, but probably sp3 would be better", then "sp3 is better"; kennedy Whisper 788 vs 798)
const float kSecondaryVowelLength = param("kSecondaryVowelLength", 1.362f) /* cost 0.0307218938484690 wobble */ /* 1.19 from the timing search made the secondary-stress syllable as long as the primary */, kInitialRLength = 1.0f;
// A word-initial ɹ before a vowel ("revolution", "republics", "rights") is a
// full r: deeper (F3 1550 as the male readings' 1450-1600; puhuja's reached
// 1730) and a little louder (Jyrki: "r should be pronounced much more when it
// is in the beginning of the word", 2026-10-30).
const float kInitialRF3 = param("kInitialRF3", 1614) /* cost 0.0294516433593330 n13-timing-w30 */, kInitialRLevel = param("kInitialRLevel", 0.6041f) /* cost 0.0294516433593330 n13-timing-w30 */;
// Diphthongs (aɪ aʊ oʊ eɪ ɔɪ) run shorter than the Klatt table gives: over
// kennedy.txt puhuja's aɪ was 227 ms against 135-160 in Kokoro and
// ElevenLabs, oʊ 201 against 110-169, eɪ 178 against 130-150; "alike" was
// heard as "too much effort, too active" (2026-10-30). 0.85: a quarter of
// the way, as with the other durations.
const float kDiphthongLength = param("kDiphthongLength", 0.8787f) /* cost 0.0307218938484690 wobble */;  // see sayPhrase (0.8: "generosity" ʤɛ 264 ms against 130-160 in the references, 2026-10-30)
constexpr float kSyllabicNLength = 1.6f;
constexpr float kSyllabicLLength = 1.8f;
const float kInnerGlide = param("EN_INNER_GLIDE", 1.0f);  // w / j inside a word (0 off, 1 aʊ, 2 every high vowel or glide)
// "to" before a vowel sound (see sayPhrase): 0 keeps the weak tə, 1 tʊ, 2 tu.
const float kToBeforeVowel = param("EN_TO_VOWEL", 0.0f);  // Jyrki 2026-11-05: in "power to abolish" tu (2) sounded "chu"; of tu / tʊ / tə renders the weak tə was best by ear
const float kToVowelLength = param("EN_TO_VOWEL_LEN", 0.6f);  // ... its vowel's length then
const float kToGlide = param("EN_TO_GLIDE", 1.0f);  // ... and its linking w (0: none)
const float kToWeakOnly = param("EN_TO_VOWEL_WEAK", 1.0f);  // 1: only before an unstressed vowel ("to abolish", not "to ask")
constexpr float kLinkGlideMs = 40, kLinkGlideLevel = 0.7f;  // w / j linking a vowel to the next word's
constexpr float kGlideBeforeVowel = 0.4f;  // a diphthong's glide before a vowel: share of that vowel
// EN_FLAP_SYLLABIC_L: only t d before a syllabic l ("mortal", "little"):
// Kokoro and 11labs tap it; the full t released into the l's narrow side
// opening left 25 ms of near silence after the burst (Jyrki 2026-11-05:
// "mortal, the t is weird").
const float kFlapSyllabicL = param("EN_FLAP_SYLLABIC_L", 1.0f);
const float kFlapping = param("kFlapping", 0.0f);  // 1: the American flap of t/d between vowels
const float kFlapMs = param("kFlapMs", 18.61) /* cost 0.0294516433593330 n13-timing-w30 */;  // a flap's closure  // the n of -tion (ʃn̩) carries the syllable
// Vowel lengthening (see sayPhrase): factor at 50 ms, change per ms, floor.
const float kVowelLength50 = param("kVowelLength50", 1.615f) /* cost 0.0307218938484690 wobble */, kVowelLengthSlope = -0.0029f, kVowelLengthMin = param("kVowelLengthMin", 0.773f) /* cost 0.0307218938484690 wobble */;

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

// --- Breath planning ------------------------------------------------------
// Jyrki (2026-11-01): "do the breath optimization iteratively the same way
// like the pitch and formant travel, to find a good compromise that can
// affect the formant travel, too -- and will lead to breath not running out
// on long statements accidentally."
//
// Before a phrase is spoken, the words of this phrase and of the rest of
// its sentence (the held-back text, restOfSentence) are costed: the air
// each takes (the synth's table per phoneme over its planned duration,
// times the drain the larynx has really shown, Synth::drainScale) and the
// kind of place before it: a punctuation or split boundary, a phrase
// opener after a word that may be split from, a plain word boundary, or
// none (after "of", "the"; the last words of the sentence). A dynamic
// program over the breath places minimises
//   sum of the places' costs
//   + rateW * duration * ln(tempo)^2 per breath group    (speaking faster
//     saves air but compresses the formant paths; this is the compromise)
//   + floorW * ((floor - air left at the group's end) / 100)^2 below the
//     floor, where the lung pressure starts to sag, and a wall above empty.
// Each group's tempo is the smallest that keeps its end above the floor,
// found by a few rounds of iteration (vowels follow the tempo fully,
// consonants by its square root, so the air is not a power of it). The
// plan is redone at every phrase with the air actually left, so a wrong
// estimate corrects itself within the sentence.
namespace {
float breathKnob(const char* name, float d) {
  const char* v = std::getenv(name);
  return v ? static_cast<float>(std::atof(v)) : param(name, d);
}
struct BreathWeights {
  float placeCost[4] = {breathKnob("BREATH_PUNCT", 0.3f), breathKnob("BREATH_SEAM", 1.0f), breathKnob("BREATH_PLAIN", 3.0f), 1e9f};
  float rateW = breathKnob("BREATH_RATE_W", 30.0f);     // cost of ln(tempo)^2 per second of speech
  float rateMax = breathKnob("BREATH_RATE_MAX", 1.15f);  // the fastest tempo the plan may ask for
  float floorMargin = breathKnob("BREATH_FLOOR_MARGIN", 100.0f);  // cm^3 kept above the sag point
  float floorW = breathKnob("BREATH_FLOOR_W", 2.0f);
  int rounds = static_cast<int>(breathKnob("BREATH_ROUNDS", 6));
  // (PROSODY agent11, lab knobs; 0: as before) a cost on long breath groups, BREATH_LEN_W x (seconds / BREATH_LEN_T)^2
  // per group: the 11labs readers breathe at 52-77 % of the commas (groups 3.5 s median), puhuja at 3-6 % (6.2 s).
  float lenW = breathKnob("BREATH_LEN_W", 0.3f);
  float lenT = breathKnob("BREATH_LEN_T", 3.5f);
  bool off = std::getenv("BREATH_OFF") != nullptr;
  bool debug = std::getenv("BREATH_DEBUG") != nullptr;
};
struct BreathItem {
  float airV, airC;  // air at tempo 1 (cm^3): the vowels' and the consonants'
  float dur;         // seconds at tempo 1
  int place;         // the place before it: 0 punctuation, 1 seam, 2 plain, 3 none
  std::string text;
};
}  // namespace

std::string EnglishReader::restOfSentence() const {
  if (lookPos_ + 1 >= lookahead_.size()) return std::string();
  const char at = lookahead_[lookPos_];
  if (at == '.' || at == '!' || at == '?' || at == '\n') return std::string();  // the phrase ends the sentence
  std::string_view rest(lookahead_);
  rest = rest.substr(lookPos_ + 1);
  size_t end = rest.size();
  for (size_t k = 0; k < rest.size(); ++k) {
    const char c = rest[k];
    if (c == '\n' || ((c == '.' || c == '!' || c == '?') && (k + 1 == rest.size() || rest[k + 1] == ' ' || rest[k + 1] == '\n' || rest[k + 1] == '\r' || rest[k + 1] == '\t'))) {
      end = k + 1;
      break;
    }
  }
  return std::string(rest.substr(0, end));
}

// (PROSODY agent11) the breath-group contour's steps.
namespace {
float bgStep(float a, float b, float x) {
  const float t = std::clamp((x - a) / std::max(b - a, 1e-3f), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}
float bgStartTo() { static const float v = param("EN_BG_START_TO", 0.4f); return v; }
float bgFinalFrom() { static const float v = param("EN_BG_FINAL_FROM", 0.7f); return v; }
}  // namespace

void EnglishReader::planBreaths() {
  static const BreathWeights W;
  const size_t n = phrase_.size();
  wordTempo_.assign(n, 1.0f);
  breathPlan_.clear();
  breathSnatch_.clear();
  if (W.off || n == 0) return;
  breathPlan_.assign(n, 0);
  breathSnatch_.assign(n, 0);
  std::vector<BreathItem> items;
  // This phrase: the fitted durations (the tempo is 1 here).
  const std::vector<std::vector<float>> dur = durations();
  for (size_t w = 0; w < n; ++w) {
    BreathItem it{0, 0, 0, 2, phrase_[w].text};
    const auto& tok = phrase_[w].tokens;
    for (size_t i = 0; i < tok.size() && i < dur[w].size(); ++i) {
      const Sound snd = soundOf(kEnglishTokens[tok[i] - 0x80]);
      const float air = dur[w][i] * 0.001f * Synth::airRate(snd.ph[0]) * synth_.airScale();
      (isVowelToken(tok[i]) ? it.airV : it.airC) += air;
      it.dur += dur[w][i] * 0.001f;
    }
    const std::string& before = w > 0 ? phrase_[w - 1].text : std::string();
    it.place = w == 0 ? 0 : isNoSplitAfter(before) ? 3 : isBreathOpener(phrase_[w].text) ? 1 : 2;
    items.push_back(it);
  }
  // The other half of a split phrase (spoken next, no longer in the text).
  for (size_t w = 0; w < splitTail_.size(); ++w) {
    BreathItem it{0, 0, 0, w == 0 ? 0 : 2, splitTail_[w].text};
    for (uint8_t t : splitTail_[w].tokens) {
      const char* name = kEnglishTokens[t - 0x80];
      const float ms = klatt(baseName(t)).inh * kEnglishTempo;
      (isVowelToken(t) ? it.airV : it.airC) += ms * 0.001f * Synth::airRate(soundOf(name).ph[0]) * synth_.airScale();
      it.dur += ms * 0.001f;
    }
    if (w > 0) it.place = isNoSplitAfter(splitTail_[w - 1].text) ? 3 : isBreathOpener(splitTail_[w].text) ? 1 : 2;
    items.push_back(it);
  }
  // The rest of the sentence from the text: words by white space, their
  // punctuation marking the boundaries, their pronunciation for the air.
  bool sawEnd = false;
  {
    std::string rest = restOfSentence();
    // Dashes (also the UTF-8 ones) and other marks beyond ASCII separate
    // words with a pause; a curly apostrophe is an apostrophe.
    for (size_t k = 0; k < rest.size(); ++k) {
      const unsigned char c = static_cast<unsigned char>(rest[k]);
      if (c == 0xe2 && k + 2 < rest.size() && static_cast<unsigned char>(rest[k + 1]) == 0x80 && static_cast<unsigned char>(rest[k + 2]) == 0x99) {
        rest.replace(k, 3, "'");
      } else if (c == '-' || c >= 0x80) {
        rest.replace(k, 1, " , ");
        k += 2;
      }
    }
    int nextPlace = 0;  // after this phrase's end: a boundary with a pause
    std::string prevText = !splitTail_.empty() ? splitTail_.back().text : n > 0 ? phrase_[n - 1].text : std::string();
    size_t i = 0;
    while (i < rest.size() && !sawEnd) {
      while (i < rest.size() && (rest[i] == ' ' || rest[i] == '\t' || rest[i] == '\r' || rest[i] == '\n')) ++i;
      size_t j = i;
      while (j < rest.size() && !(rest[j] == ' ' || rest[j] == '\t' || rest[j] == '\r' || rest[j] == '\n')) ++j;
      if (j == i) break;
      std::string chunk = rest.substr(i, j - i);
      i = j;
      // Trailing marks: a sentence end stops the sentence; , ; : and dashes
      // make the next boundary a punctuation one.
      bool punctAfter = false;
      while (!chunk.empty()) {
        const unsigned char c = static_cast<unsigned char>(chunk.back());
        if (c == '.' || c == '!' || c == '?') { sawEnd = true; chunk.pop_back(); continue; }
        if (c == ',' || c == ';' || c == ':' || c == '-' || c == '"' || c == ')' || c == '\'' || c >= 0x80) {
          if (c != '\'' && c != '"' && c != ')') punctAfter = true;
          chunk.pop_back();
          continue;
        }
        break;
      }
      size_t a = 0;
      while (a < chunk.size() && !((chunk[a] >= 'a' && chunk[a] <= 'z') || (chunk[a] >= 'A' && chunk[a] <= 'Z') || (chunk[a] >= '0' && chunk[a] <= '9'))) ++a;
      chunk = chunk.substr(a);
      if (chunk.empty()) { if (punctAfter) nextPlace = 0; continue; }
      std::string word;
      int digits = 0;
      for (char c : chunk) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        if ((c >= 'a' && c <= 'z') || c == '\'') word += c;
        if (c >= '0' && c <= '9') ++digits;
      }
      BreathItem it{0, 0, 0, 2, word};
      if (digits > 0) {  // a number: about three syllables a digit
        it.airV = 90.0f * 2.0f * digits;
        it.airC = 90.0f * 1.0f * digits;
        it.dur = 0.22f * 3 * digits;
      } else if (!word.empty()) {
        const std::vector<uint8_t> tok = pronounce(word);
        for (uint8_t t : tok) {
          const char* name = kEnglishTokens[t - 0x80];
          const Sound snd = soundOf(name);
          const float ms = klatt(baseName(t)).inh * kEnglishTempo;
          const float air = ms * 0.001f * Synth::airRate(snd.ph[0]) * synth_.airScale();
          (isVowelToken(t) ? it.airV : it.airC) += air;
          it.dur += ms * 0.001f;
        }
      }
      it.place = nextPlace == 0 ? 0 : isNoSplitAfter(prevText) ? 3 : isBreathOpener(word) ? 1 : 2;
      items.push_back(it);
      prevText = word;
      nextPlace = punctAfter ? 0 : 2;
    }
  }
  const size_t N = items.size();
  // No breath with only the last words of the sentence left.
  if (sawEnd)
    for (size_t k = N > kNoBreathLastWords ? N - kNoBreathLastWords : 0; k < N; ++k)
      if (k > 0) items[k].place = 3;
  const float scale = synth_.drainScale();
  for (BreathItem& it : items) { it.airV *= scale; it.airC *= scale; }

  // The dynamic program. best[i]: the least cost of speaking items i.. with
  // a breath group starting at i (full lungs, unless i == 0: the air left).
  const float V0 = synth_.plannedAir(), Vmax = synth_.lungMax();
  const float floor = synth_.lungFloor() + W.floorMargin, empty = synth_.lungEmpty() + 100.0f;
  const double kHuge = 1e8;
  std::vector<double> best(N + 1, 0.0);
  std::vector<int> nextBreath(N + 1, -1);
  std::vector<float> groupRate(N + 1, 1.0f);
  // Prefix sums for the group air and duration.
  std::vector<double> sumV(N + 1, 0), sumC(N + 1, 0), sumD(N + 1, 0);
  for (size_t k = 0; k < N; ++k) {
    sumV[k + 1] = sumV[k] + items[k].airV;
    sumC[k + 1] = sumC[k] + items[k].airC;
    sumD[k + 1] = sumD[k] + items[k].dur;
  }
  // (BREATH_LEN_W) the group running at the phrase's start began this long ago.
  const double elapsed0 = V0 >= Vmax - 1.0f ? 0.0 : groupElapsed_;
  auto groupCost = [&](size_t i, size_t j, float& rate) {
    const double aV = sumV[j] - sumV[i], aC = sumC[j] - sumC[i], d = sumD[j] - sumD[i];
    const double dLen = d + (i == 0 ? elapsed0 : 0.0);
    const double start = i == 0 ? V0 : Vmax;
    // The tempo: 1 if the air carries to the floor, else the smallest that
    // does (iterated: the consonants follow it only by its square root).
    double r = 1.0;
    for (int round = 0; round < W.rounds; ++round) {
      const double need = aV / r + aC / std::sqrt(r);
      if (start - need >= floor) break;
      const double avail = std::max(start - floor, 1.0);
      // Scale the air by avail/need, mostly through the vowels.
      const double rNew = r * need / avail;
      r = std::min(std::max(rNew, 1.0), static_cast<double>(W.rateMax));
      if (r >= W.rateMax) break;
    }
    rate = static_cast<float>(r);
    const double need = aV / r + aC / std::sqrt(r);
    const double end = start - need;
    double cost = W.rateW * d * std::log(r) * std::log(r);
    if (W.lenW > 0) cost += W.lenW * (dLen / W.lenT) * (dLen / W.lenT);
    if (end < floor) cost += W.floorW * ((floor - end) / 100.0) * ((floor - end) / 100.0);
    if (end < empty) cost += kHuge;
    return cost;
  };
  for (size_t i = N; i-- > 0;) {
    double bi = kHuge * 10;
    int bj = -1;
    float br = 1.0f;
    for (size_t j = i + 1; j <= N; ++j) {
      float r;
      double c = groupCost(i, j, r);
      if (j < N) {
        if (items[j].place == 3) continue;
        c += W.placeCost[items[j].place] + best[j];
      }
      if (c < bi) { bi = c; bj = j < N ? static_cast<int>(j) : -1; br = r; }
    }
    best[i] = bi;
    nextBreath[i] = bj;
    groupRate[i] = br;
  }
  // Read the plan off for this phrase's words. A group that starts at the
  // phrase's first word with air to spare needs no breath (i == 0 is the
  // air left, not a breath); a breath before word 0 is planned only when
  // it pays: compare with starting full.
  {
    float r0;
    const double here = groupCost(0, nextBreath[0] < 0 ? N : static_cast<size_t>(nextBreath[0]), r0);
    // Cost of breathing first: a punctuation place (the pause is there),
    // then the groups from a full start.
    const float savedV0 = V0;
    (void)savedV0;
    double breathFirst = W.placeCost[0];
    {
      // Recompute the first group from full lungs.
      std::vector<double> bestFull(best);
      size_t i = 0;
      double bi = kHuge * 10; int bj = -1; float br = 1.0f;
      for (size_t j = 1; j <= N; ++j) {
        const double aV = sumV[j] - sumV[i], aC = sumC[j] - sumC[i], d = sumD[j] - sumD[i];
        double r = 1.0;
        for (int round = 0; round < W.rounds; ++round) {
          const double need = aV / r + aC / std::sqrt(r);
          if (Vmax - need >= floor) break;
          r = std::min(std::max(r * need / std::max(Vmax - floor, 1.0f), 1.0), static_cast<double>(W.rateMax));
          if (r >= W.rateMax) break;
        }
        const double need = aV / r + aC / std::sqrt(r), end = Vmax - need;
        double c = W.rateW * d * std::log(r) * std::log(r);
        if (W.lenW > 0) c += W.lenW * (d / W.lenT) * (d / W.lenT);
        if (end < floor) c += W.floorW * ((floor - end) / 100.0) * ((floor - end) / 100.0);
        if (end < empty) c += kHuge;
        if (j < N) { if (items[j].place == 3) continue; c += W.placeCost[items[j].place] + bestFull[j]; }
        if (c < bi) { bi = c; bj = j < N ? static_cast<int>(j) : -1; br = static_cast<float>(r); }
      }
      breathFirst += bi;
      if (breathFirst < best[0]) {
        breathPlan_[0] = 1;
        nextBreath[0] = bj;
        groupRate[0] = br;
        best[0] = breathFirst;
      }
    }
    (void)here;
  }
  size_t start = 0;
  while (true) {
    const int nb = nextBreath[start];
    const size_t end = nb < 0 ? N : static_cast<size_t>(nb);
    for (size_t k = start; k < end && k < n; ++k) wordTempo_[k] = groupRate[start];
    if (nb < 0 || static_cast<size_t>(nb) >= N) break;
    if (static_cast<size_t>(nb) < n) {
      breathPlan_[nb] = 1;
      breathSnatch_[nb] = items[nb].place == 2;
    }
    start = static_cast<size_t>(nb);
  }
  // (PROSODY agent11) each word's planned air and its place in its breath group (the groups as planned, the air at
  // mid-word; a group that began in an earlier phrase began full, at a breath).
  {
    wordGroupPos_.assign(n, 0.0f);
    wordAirU_.assign(n, 1.0f);
    size_t s = 0;
    double air = V0;
    bool firstGroup = true;
    while (true) {
      const int nb = nextBreath[s];
      const size_t e = nb < 0 ? N : static_cast<size_t>(nb);
      if (!firstGroup || (s == 0 && breathPlan_[0])) air = Vmax;
      const double r = groupRate[s];
      const double gStart = Vmax;
      const double gEnd = air - ((sumV[e] - sumV[s]) / r + (sumC[e] - sumC[s]) / std::sqrt(r));
      for (size_t k = s; k < e && k < n; ++k) {
        const double need = items[k].airV / r + items[k].airC / std::sqrt(r);
        const double mid = air - 0.5 * need;
        wordGroupPos_[k] = static_cast<float>(std::clamp((gStart - mid) / std::max(gStart - gEnd, 1.0), 0.0, 1.0));
        wordAirU_[k] = static_cast<float>(std::clamp((mid - synth_.lungFloor()) / std::max(Vmax - synth_.lungFloor(), 1.0f), 0.0, 1.0));
        air -= need;
      }
      if (nb < 0 || e >= n) break;
      s = e;
      firstGroup = false;
    }
    // The running group's length after this phrase (BREATH_LEN_W).
    size_t lastBreath = 0;
    bool any = V0 >= Vmax - 1.0f;
    for (size_t k = 0; k < n; ++k) if (breathPlan_[k]) lastBreath = k, any = true;
    double el = any ? 0.0 : groupElapsed_;
    for (size_t k = lastBreath; k < n; ++k) el += items[k].dur;
    groupElapsed_ = static_cast<float>(el);
  }
  if (W.debug) {
    std::fprintf(stderr, "breath plan: %zu words of this phrase + %zu ahead, air left %.0f, scale %.2f, cost %.2f\n", n, N - n, V0, scale, best[0]);
    size_t s = 0;
    while (true) {
      const int nb = nextBreath[s];
      const size_t e = nb < 0 ? N : static_cast<size_t>(nb);
      std::fprintf(stderr, "  group %zu-%zu tempo %.3f%s:", s, e, groupRate[s], s == 0 && breathPlan_[0] ? " (breath first)" : "");
      for (size_t k = s; k < e; ++k) std::fprintf(stderr, " %s", items[k].text.c_str());
      std::fprintf(stderr, "\n");
      if (nb < 0) break;
      s = static_cast<size_t>(nb);
    }
  }
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
    splitTail_ = tail;
    sayPhrase(',');
    splitTail_.clear();
    pause(kSymmetryPauseStops);
    phrase_ = std::move(tail);
    break;
  }
  if (split > 0 && split < phrase_.size()) {
    std::vector<Word> tail(phrase_.begin() + static_cast<long>(split), phrase_.end());
    phrase_.resize(split);
    splitTail_ = tail;
    sayPhrase(',');
    splitTail_.clear();
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
  // "And" opening a phrase is said in full (CMUdict's first reading is the
  // weak ənd): "And yet the same" with ənd came back as "In the end",
  // "Then we have".
  if (!phrase_.empty() && phrase_[0].text == "and") phrase_[0].tokens = arpaTokens("AE1 N D");
  for (size_t w = 1; w + 1 < phrase_.size(); ++w) {
    if (phrase_[w].accent) continue;
    // have/has/had are weak only as auxiliaries ("have been", "has made");
    // as main verbs they keep the full form ("have a big dog": weak, it was
    // heard as "gonna be").
    const std::string& t = phrase_[w].text;
    if ((t == "have" || t == "has" || t == "had") && !isAuxiliaryUse(phrase_[w + 1].text)) continue;
    // "are" after a word ending in a vowel ("who are", "they are"): the
    // weak ɚ fused with the linking glide into a contraction, and Whisper
    // heard "who were rich", "we're right". The full form keeps the words.
    if (t == "are" && !phrase_[w - 1].tokens.empty() && isVowelToken(phrase_[w - 1].tokens.back())) continue;
    for (const WeakForm& f : kWeakForms)
      if (t == f.word) phrase_[w].tokens = arpaTokens(f.weak);
    // "for": the weak ɜ (F2 ~1370 Hz) was heard as "fur" ("beliefs for which", Jyrki 2026-11-06); the readers' reduced
    // "for" keeps the rounded back vowel under the r colour (Kokoro F2 865-1334, "beliefs for which" 1097): a short ɔ + ɹ.
    static const float kForWeakRounded = param("EN_FOR_ROUNDED", 1.0f);  // Jyrki 2026-11-06: "forwhich-ro.wav is good" (kennedy Whisper 856, unchanged)
    if (t == "for" && kForWeakRounded > 0.5f) phrase_[w].tokens = arpaTokens("F AO0 R");
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
  // Opposites contrasted in the phrase ("well or ill", "the rich and the
  // poor") are both emphasised; not when they form one term ("black and
  // white photography": joined by "and" and followed by a content word).
  static const char* const kOpposites[][2] = {
      {"black", "white"}, {"bright", "dark"}, {"light", "dark"}, {"poor", "rich"}, {"well", "ill"},
      {"good", "bad"}, {"good", "evil"}, {"big", "small"}, {"large", "small"}, {"high", "low"},
      {"hot", "cold"}, {"old", "new"}, {"old", "young"}, {"fast", "slow"}, {"strong", "weak"},
      {"war", "peace"}, {"friend", "foe"}, {"friend", "enemy"}, {"friends", "enemies"}, {"life", "death"},
      {"love", "hate"}, {"right", "wrong"}, {"left", "right"}, {"day", "night"}, {"up", "down"},
      {"more", "less"}, {"first", "last"}, {"win", "lose"}, {"open", "closed"}, {"full", "empty"},
      {"early", "late"}, {"near", "far"}, {"north", "south"}, {"east", "west"}, {"men", "women"},
      {"past", "future"}, {"truth", "lies"}, {"few", "many"}, {"before", "after"}, {"begin", "end"},
      {"beginning", "end"}, {"question", "answer"}, {"give", "take"}, {"buy", "sell"}, {"inside", "outside"},
      {"yes", "no"}, {"rise", "fall"}, {"victory", "defeat"}, {"success", "failure"}, {"wealth", "poverty"},
      {"hope", "fear"}, {"happy", "sad"}, {"joy", "sorrow"}, {"weak", "powerful"}, {"more", "fewer"}};
  for (size_t i = 0; i < phrase_.size(); ++i)
    for (size_t j = i + 1; j < phrase_.size(); ++j) {
      bool pair = false;
      for (const auto& o : kOpposites)
        pair = pair || (phrase_[i].text == o[0] && phrase_[j].text == o[1]) || (phrase_[i].text == o[1] && phrase_[j].text == o[0]);
      if (!pair) continue;
      const bool oneTerm = j == i + 2 && phrase_[i + 1].text == "and" && j + 1 < phrase_.size() && phrase_[j + 1].accent;
      if (oneTerm) continue;
      phrase_[i].contrast = phrase_[j].contrast = true;
      phrase_[i].accent = phrase_[j].accent = true;
    }
  // A content word said again within a few words ("to help them help
  // themselves") is said differently the second time: a short pause before
  // it, and it starts a little above the first and falls.
  for (size_t j = 1; j < phrase_.size(); ++j) {
    if (!phrase_[j].accent || phrase_[j].repeat) continue;
    for (size_t i = j - 1; i + kRepeatWithin >= j && i + 1 > 0; --i)
      if (phrase_[i].accent && phrase_[i].text == phrase_[j].text && !phrase_[i].repeat) {
        phrase_[i].repeat = 1;
        phrase_[j].repeat = 2;
        phrase_[j].repeatOf = i;
        break;
      }
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
      // (N agent13 round L, lab knobs; 0: off) a final unstressed -ɪn as -ən (EN_FN_IH_SCHWA 1; 2: as -ɛn; Kokoro's ɪ there sits
      // at F2 1500-1750, F3 2400-2600, between ɪ and ə), and -tən -dən -sən -zən with a syllabic n (EN_FN_SYLLABIC 1:
      // "Latin", "Aston", "Hansen"; as -ʃən above). N lab (228 final-n words + 50 -ing + 30 -m words, 2 voices, two
      // draws; base 56 n errors, placebo sd 5.3): EN_FN_IH_SCHWA 1 with LINE_FN_ANTICIP 0 34 (heard ŋ 40.5 -> 21), words
      // right 422 -> 422, words1000 1478 -> 1474; + LINE_FN_PRE_MIX 0.3 toward ɪ (row 9) 36, words 433, words1000 1494.
      // EN_FN_SYLLABIC 1: 96 (the n deleted).
      {
        static const float kFnIhSchwa = param("EN_FN_IH_SCHWA", 1.0f), kFnSyllabic = param("EN_FN_SYLLABIC", 0.0f);
        static const uint8_t kIH0 = tokenOf("IH0"), kT = tokenOf("T"), kD = tokenOf("D"), kS = tokenOf("S"), kZ = tokenOf("Z");
        const size_t m = t.size();
        // Only after another vowel (a word of 2+ syllables: Gavin, Griffin): the one-syllable function word "been"
        // (reduced to b IH0 n) stays bɪn (make test: "They have been here.").
        const bool earlierVowel = m >= 3 && std::any_of(t.begin(), t.end() - 2, [](uint8_t x) { return isVowelToken(x); });
        if (kFnIhSchwa > 0.5f && earlierVowel && t[m - 1] == kN && t[m - 2] == kIH0) t[m - 2] = kFnIhSchwa > 1.5f ? tokenOf("EH0") : kAH0;
        if (kFnSyllabic > 0.5f && !word.syllabicN && m >= 4 && t[m - 1] == kN && t[m - 2] == kAH0 &&
            (t[m - 3] == kT || t[m - 3] == kD || t[m - 3] == kS || t[m - 3] == kZ)) {
          t.erase(t.end() - 2);
          word.syllabicN = true;
        }
      }
      // -tle, -ple, -ble, -dle, -kle, -gle, -fle, -vle, -sle, -zle ("little",
      // "people", "trouble"): a syllabic l (l̩), no schwa (with it "little"
      // was "lit-tuhl", a schwa and an l of 90 ms each).
      static const uint8_t kL = tokenOf("L");
      if (n >= 4 && t[n - 1] == kL && t[n - 2] == kAH0 && !isVowelToken(t[n - 3]) && t[n - 3] != kL &&
          baseName(t[n - 3]) != "R" && baseName(t[n - 3]) != "N" && baseName(t[n - 3]) != "M" && baseName(t[n - 3]) != "NG") {
        t.erase(t.end() - 2);
        word.syllabicL = true;
      }
    }
  }
  // A word-initial ER0 before a vowel ("arise" ER0 AY1 Z) as AH0 R, the way Kokoro says it (əˈɹaɪz): the ə and the ɹ
  // then take their own durations, levels and shapes (EN_ER_INIT_AH 1; 0: as before; ER-INITIAL agent13 round L, see
  // EN_ER_INIT below). Before the syllables and durations are counted.
  {
    static const float kErInitAh = param("EN_ER_INIT_AH", 0.0f);
    if (kErInitAh > 0) {
      static const uint8_t kER0 = tokenOf("ER0"), kAH0 = tokenOf("AH0"), kR = tokenOf("R");
      for (auto& word : phrase_) {
        auto& t = word.tokens;
        if (t.size() > 1 && t[0] == kER0 && isVowelToken(t[1])) {
          t[0] = kAH0;
          t.insert(t.begin() + 1, kR);
          word.erInitAh = true;
        }
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
  std::vector<float> wordPitch(phrase_.size(), kPhraseStart);  // each word's accent pitch (for repeats)
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
      bool emphatic = phrase_[w].contrast;
      for (const char* e : kEmphatic) emphatic = emphatic || text == e;
      // Loaded, frightening words carry emphasis already as modifiers ("a far more iron tyranny": "iron" is not the
      // metal; Jyrki 2026-11-10: "it cannot be read without some emphasis already on 'iron'"). EN_EMOTIVE 1: on.
      {
        static const float kEmotive = param("EN_EMOTIVE", 0.0f);
        if (kEmotive > 0 && isEmotiveWord(text)) emphatic = true;
      }
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
      // A repeated word: the first rises, the second starts above it and falls.
      if (phrase_[w].repeat == 1) p += kRepeatRise;
      if (phrase_[w].repeat == 2) {
        // EN_REPEAT_LEVEL > 0: the whole second word sits this many semitones above the first one's level, held flat
        // (Jyrki 2026-11-10: "the whole 2nd 'help' could be a whole tone higher ... instead of just the beginning").
        static const float kRepeatLevel = param("EN_REPEAT_LEVEL", 0.0f);
        if (kRepeatLevel > 0) {
          p = wordPitch[phrase_[w].repeatOf] + kRepeatLevel;
          syl[k].end = p;
        } else {
          p = std::max(p, wordPitch[phrase_[w].repeatOf] + kRepeatSecondAbove);
          syl[k].end = p - kRepeatFall;
        }
      }
      { static const float kEmoPitch = param("EN_EMOTIVE_PITCH", -2.0f); if (kEmoPitch != 0 && isEmotiveWord(text)) p += kEmoPitch; }
      wordPitch[w] = p;
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
  // EN_REPEAT_WHOLE 1: every syllable of a repeated word (repeat 2) sits flat on the EN_REPEAT_LEVEL pedestal, not
  // only its stressed one (Jyrki 2026-11-10: "try it with a much longer word, let me listen if the whole word needs
  // to [be] higher").
  {
    static const float kRepeatWhole = param("EN_REPEAT_WHOLE", 0.0f), kRepeatLevelW = param("EN_REPEAT_LEVEL", 0.0f);
    // EN_REPEAT_FIRST_FALL > 0: the first word glides down this many semitones over its syllables, from its accent
    // pitch (Jyrki 2026-11-10: "the first repeating word could have a pitch that lowers by a semitone or so, and the
    // second time ... remain flat"); the second sits flat at the first's start + EN_REPEAT_LEVEL (may be 0).
    static const float kRepeatFirstFall = param("EN_REPEAT_FIRST_FALL", 0.0f);
    if (kRepeatWhole > 0 && (kRepeatLevelW > 0 || kRepeatFirstFall > 0)) {
      for (size_t i = 0; i < syl.size(); ++i) {
        Syl& s2 = syl[i];
        if (phrase_[s2.word].repeat == 2) { s2.pitch = wordPitch[phrase_[s2.word].repeatOf] + kRepeatLevelW; s2.end = s2.pitch; }
        else if (phrase_[s2.word].repeat == 1 && kRepeatFirstFall > 0) {
          size_t a0 = i, n = 0;
          while (a0 > 0 && syl[a0 - 1].word == s2.word) --a0;
          for (size_t j = a0; j < syl.size() && syl[j].word == s2.word; ++j) ++n;
          const float p0 = wordPitch[s2.word], k0 = static_cast<float>(i - a0);
          s2.pitch = p0 - kRepeatFirstFall * k0 / n;
          s2.end = p0 - kRepeatFirstFall * (k0 + 1) / n;
        } else if (phrase_[s2.word].repeat == 1 && !s2.accent) { s2.pitch += kRepeatRise; if (s2.end != Synth::kNoPitch) s2.end += kRepeatRise; }  // the first word moves as a whole too
      }
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

  planBreaths();  // where to breathe in this phrase, and its tempo (sets wordTempo_)
  // (PROSODY agent11, lab knobs; 0: as before) the pitch follows the breath group: EN_BG_ST st from the group's start
  // (+half) to its end (-half), by the planned air spent (11labs: 7.5 st from a group's first words to its last, a
  // 7.4 st reset after a breath; puhuja 3.6-5 and 2.9); EN_AIR_ST st by the lungs' planned fullness (full +half,
  // floor -half).
  {
    // ... and the shape: EN_BG_START_ST on the group's first part (fading out by EN_BG_START_TO of it), EN_BG_FINAL_ST
    // off its last part (from EN_BG_FINAL_FROM): the readers' 7.5 st sit in a high start and a final drop, the middle
    // declines only ~0.5 st/s.
    static const float kBgSt = param("EN_BG_ST", 0.0f), kAirSt = param("EN_AIR_ST", 0.0f),
                       kBgStartSt = param("EN_BG_START_ST", 3.5f), kBgFinSt = param("EN_BG_FINAL_ST", 2.5f);
    if ((kBgSt != 0 || kAirSt != 0 || kBgStartSt != 0 || kBgFinSt != 0) && !wordGroupPos_.empty())
      for (Syl& y : syl) {
        if (y.word >= wordGroupPos_.size()) continue;
        const float g = wordGroupPos_[y.word];
        const float d = kBgSt * (0.5f - g) + kAirSt * (wordAirU_[y.word] - 0.5f) +
                        kBgStartSt * (1.0f - bgStep(0.0f, bgStartTo(), g)) - kBgFinSt * bgStep(bgFinalFrom(), 1.0f, g);
        y.pitch += d;
        if (y.end != Synth::kNoPitch) y.end += d;
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
  // "to" before a vowel sound keeps a rounded vowel, tʊ / tu, and links into
  // the next word through a short w ("the power to abolish": goog, 11labs and
  // Kokoro start the vowel at F1 300-350, F2 850-1300 and rise into the ə;
  // the weak tə ran into abolish's ə as one flat 160 ms schwa).
  if (kToBeforeVowel > 0.5f) {
    static const uint8_t kT = tokenOf("T"), kAH0 = tokenOf("AH0"), kUH0 = tokenOf("UH0"),
                         kUW0 = tokenOf("UW0");
    for (size_t w = 0; w + 1 < phrase_.size(); ++w) {
      if (phrase_[w].text != "to" || phrase_[w + 1].tokens.empty()) continue;
      const uint8_t nt = phrase_[w + 1].tokens[0];
      if (!isVowelToken(nt)) continue;
      if (kToWeakOnly > 0.5f) {
        const char* nn = kEnglishTokens[nt - 0x80];
        if (nn[strlen(nn) - 1] != '0') continue;
      }
      auto& t = phrase_[w].tokens;
      if (t.size() == 2 && t[0] == kT && t[1] == kAH0) t[1] = kToBeforeVowel > 1.5f ? kUW0 : kUH0;
    }
  }
  // American flap: t or d after a vowel (or r) and before an unstressed
  // vowel in the same word ("city", "better", "water"): a quick voiced tap
  // instead of a full, aspirated t.
  // (Off by default since 2026-11-02: the readers who are understood make a
  // full t in "poverty", 65 ms of silence while the pressure builds and a
  // 30 ms puff at the release; the tap had neither, and the listener lost
  // the t. kFlapping 1 restores the American tap.)
  if (kFlapping > 0.5f || kFlapSyllabicL > 0.5f) {
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
        // ... or a syllabic l or n ("little", "button"): the tap is there too.
        const bool syllabic = i + 2 == t.size() && (phrase_[w].syllabicL || phrase_[w].syllabicN);
        if (kFlapping <= 0.5f && !(i + 2 == t.size() && phrase_[w].syllabicL)) continue;
        if (i + 1 >= t.size() || !(unstressedVowel(t[i + 1]) || syllabic)) continue;
        t[i] = kD;
        phrase_[w].flaps |= 1u << i;
      }
    }
  }
  // Final devoicing: before a sentence-end pause a final b d g loses its
  // voicing and is heard as p t k ("world." ends like "worlt"); a voiced
  // release into silence sounded like an extra "də". After the durations:
  // the vowel before keeps its length, the cue that it is a d.
  // EN_FINAL_DEVOICE 0 keeps the b d g (Jyrki 2026-11-09: the t of "the hand of God." is "too sharp, too t like, it
  // should be darker"; the dictionary hears 77 % of final voiced stops after a vowel as p t k).
  static const float kFinalDevoice = param("EN_FINAL_DEVOICE", 0.0f);
  if (kFinalDevoice > 0 && (end == '.' || end == '!' || end == '?' || end == ';' || end == ':')) {
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
    // ... taken early, before a word that opens a phrase (a determiner, a
    // conjunction, a relative), while the air is merely low: "from the hand
    // of God" breathed between "hand" and "of" and was heard as an emphasis
    // on "God"; a breath belongs before "the hand of God".
    const bool opener = w > 0 && isBreathOpener(phrase_[w].text) && !isNoSplitAfter(phrase_[w - 1].text);
    // A speaker anticipates: at an opener the breath is taken if the air
    // would not carry to the next opener (or to the phrase's end when there
    // is none), and never with only the last words left (nobody leaves one
    // word for the next breath).
    const bool lastWords = phrase_.size() - w <= kNoBreathLastWords;
    bool anticipate = false;
    if (opener && !lastWords) {
      size_t nextOpener = phrase_.size();
      for (size_t k = w + 1; k + kNoBreathLastWords < phrase_.size(); ++k)
        if (isBreathOpener(phrase_[k].text) && !isNoSplitAfter(phrase_[k - 1].text)) { nextOpener = k; break; }
      int syllablesToNext = 0, syllablesToEnd = 0;
      for (size_t k = w; k < phrase_.size(); ++k)
        for (uint8_t t : phrase_[k].tokens) {
          syllablesToEnd += isVowelToken(t);
          if (k < nextOpener) syllablesToNext += isVowelToken(t);
        }
      const float air = synth_.plannedAir();
      const bool fitsToEnd = air > syllablesToEnd * kAirPerSyllable + kBreathMargin;
      const bool reachesNext = nextOpener < phrase_.size() && air > syllablesToNext * kAirPerSyllable + kBreathMargin;
      anticipate = !fitsToEnd && !reachesNext && air <= kCommaBreathBelow;
    }
    if (phrase_[w].repeat == 2) pause(kRepeatPauseStops);  // "help them | help themselves"
    // A word-final voiceless stop before a vowel-initial word ("not | out"): a short pause (EN_STOP_VOWEL_PAUSE_STOPS,
    // ~15 ms each; 0: none), so the stop is released as a final one (the breath eased, LINE_FINAL_PRESS_*) and the vowel
    // starts afresh. Jyrki 2026-11-10: "between 'Not' and 'out' open the mouth and neutralize pressure in the lungs".
    {
      static const int kStopVowelPause = static_cast<int>(param("EN_STOP_VOWEL_PAUSE_STOPS", 0));
      if (kStopVowelPause > 0 && w > 0 && !phrase_[w].tokens.empty() && !phrase_[w - 1].tokens.empty() &&
          isVowelToken(phrase_[w].tokens.front()) && isVoicelessPlosive(baseName(phrase_[w - 1].tokens.back())))
        pause(kStopVowelPause);
    }
    // A word-final fricative straight into a word-initial h ("of half"): a short pause between them
    // (EN_FRIC_H_PAUSE_STOPS, ~15 ms each; 0: none). Jyrki 2026-11-10: "changing directly from f to h creates an
    // unnecessary event of sound color change activity".
    {
      static const int kFricHPause = static_cast<int>(param("EN_FRIC_H_PAUSE_STOPS", 0));
      if (kFricHPause > 0 && w > 0 && !phrase_[w].tokens.empty() && !phrase_[w - 1].tokens.empty() &&
          baseName(phrase_[w].tokens.front()) == "HH") {
        const std::string f = baseName(phrase_[w - 1].tokens.back());
        static const bool kFricHAffr = param("EN_FRIC_H_PAUSE_AFFR", 0.0f) > 0;  // (... and after ʧ ʤ: "torch has")
        if (f == "F" || f == "V" || f == "S" || f == "Z" || f == "SH" || f == "ZH" || f == "TH" || f == "DH" ||
            (kFricHAffr && (f == "CH" || f == "JH")))
          pause(kFricHPause);
      }
    }
    const bool allowed = w > 0 && !isNoSplitAfter(phrase_[w - 1].text);  // not "any | friend", "of | God"
    // The breath plan (planBreaths) says where; without a plan (BREATH_OFF)
    // the thresholds above decide. The emergency threshold stays in force.
    const bool planned = w < breathPlan_.size() && breathPlan_[w];
    const bool unplanned = !breathPlan_.empty();
    if (planned || (allowed && !lastWords &&
                    (synth_.plannedAir() < kEmergencyBreathAir ||
                     (!unplanned && ((opener && synth_.plannedAir() < kGoodBreathAir) || anticipate))))) {
      if (w > 0) pause(4);
      // snatched (the quick breath) only at a poor place
      synth_.breathPoint(false, false, planned ? breathSnatch_[w] != 0 : !opener, planned);
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
        // (EN_ER_KEEP_R 1: a stressed ɜ keeps its ɹ tail before a consonant too; ER agent 2026-11-11 lab knob, 0: off.)
        static const float kErKeepR = param("EN_ER_KEEP_R", 0.0f);
        const char* en = kEnglishTokens[tokens[i] - 0x80];
        if (!(nt && isVowelToken(nt)) && !(kErKeepR > 0 && en[strlen(en) - 1] == '1')) s.n = 1;
      }
      // A word-initial unstressed ɜ+ɹ before a vowel ("arise", "around", "arrest": CMUdict ER0 with no R) is ə + ɹ:
      // Kokoro opens with ~60 ms of a mid-central vowel at the word's peak level (male F1 470-570, F3 2050-2250), then
      // the ɹ (F1 300-370, F3 1450-1700, 4-6 dB down). Starting on the r-coloured ɜ, puhuja's first frame was already
      // the ɹ (F1 330, F3 1550) at the word's peak: 73 of 88 heard as "rise", "round", "range", "rest" (ER-INITIAL
      // agent13 round L). EN_ER_INIT 1: the first part is the English ə (2: ʌ, 3: ɪ); its share of the token
      // EN_ER_INIT_SHARE, the token's length x EN_ER_INIT_LEN, the ə's level + EN_ER_INIT_DB, the ɹ's + EN_ER_INIT_R_DB,
      // the shape is the line's (Shape::erctx, LINE_ERI_*). 0: as before (one ɜ).
      static const float kErInit = param("EN_ER_INIT", 2.0f), kErInitShare = param("EN_ER_INIT_SHARE", 0.5f),
                         kErInitLen = param("EN_ER_INIT_LEN", 1.05f), kErInitDb = param("EN_ER_INIT_DB", 1.0f),
                         kErInitRDb = param("EN_ER_INIT_R_DB", 0.0f), kErInitPitch = param("EN_ER_INIT_PITCH", 1.0f);
      bool erInit = false;
      if (kErInit > 0 && i == 0 && s.n == 2 && s.ph[0] == Ph::ER && s.ph[1] == Ph::RR && tokens.size() > 1 &&
          isVowelToken(tokens[1])) {
        const char* en = kEnglishTokens[tokens[i] - 0x80];
        if (en[strlen(en) - 1] == '0') {
          erInit = true;
          s.ph[0] = kErInit > 2.5f ? Ph::IH : kErInit > 1.5f ? Ph::AH : Ph::Schwa;
        }
      }
      for (int k = 0; k < s.n; ++k) {
        // The second pitch on the sound's last vowel part (a diphthong's glide).
        const int lastPart = s.n > 1 && s.ph[s.n - 1] != Ph::RR ? s.n - 1 : 0;
        // A falling diphthong falls through its glide: the glide's centre is
        // already halfway down ("right.", "now.": the fall came late, all of
        // it after the glide's centre).
        const bool fallingGlide = vowel && k > 0 && k == lastPart && pitchEnd != Synth::kNoPitch;
        // aɪ's open start: its pitch x EN_AY_A_PITCH, its level + EN_AY_A_DB, both parts' glottis x EN_AY_PRESS (AY agent
        // 2026-11-10; 1 / 0 / 1: as before).
        static const float kAyAPitch = param("EN_AY_A_PITCH", 1.0f), kAyADb = param("EN_AY_A_DB", 0.0f), kAyPress = param("EN_AY_PRESS", 1.0f);
        const bool ay = s.n == 2 && s.ph[0] == Ph::A && s.ph[1] == Ph::IH;
        // (EN_ER_INIT_PITCH: the initial ə of EN_ER_INIT / EN_ER_INIT_AH x this; 1: as before.)
        if (vowel && s.ph[k] != Ph::RR && kErInitPitch != 1.0f && (erInit || (phrase_[w].erInitAh && i == 0)))
          synth_.setPitch(pitch * kErInitPitch);
        else if (vowel && s.ph[k] != Ph::RR)  // ER = vowel + ɹ
          synth_.setPitch(fallingGlide ? pitch + kGlideFallShare * (pitchEnd - pitch) : ay && k == 0 && kAyAPitch != 1.0f ? pitch * kAyAPitch : pitch);
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
        // Unstressed ɜ+ɹ as the onset of the next syllable (see kErOnsetShare).
        bool erOnset = false;
        if (s.ph[0] == Ph::ER && s.n == 2) {
          const char* tname = kEnglishTokens[tokens[i] - 0x80];
          const uint8_t ntok = i + 1 < tokens.size() ? tokens[i + 1]
                               : w + 1 < phrase_.size() && !phrase_[w + 1].tokens.empty() ? phrase_[w + 1].tokens[0] : 0;
          if (tname[strlen(tname) - 1] == '0' && ntok && isVowelToken(ntok)) {
            const Sound ns = soundOf(kEnglishTokens[ntok - 0x80]);
            const auto& nv = kEnglishShape[static_cast<int>(ns.ph[0])];
            const float nf2 = ns.start ? ns.start[1] : nv.f2 > 0 ? nv.f2 : Synth::tableFormant(ns.ph[0], 1);
            erOnset = nf2 < kErOnsetFrontF2;
          }
          // (F2 only: the ɹ's F3 is not below the ɜ's own.)
          if (erOnset && k == 0) shape.f2 += kErOnsetPull * (Synth::tableFormant(Ph::RR, 1) - shape.f2);
        }
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
        // aʊ's own split (EN_AW_FIRST_SHARE): Jyrki 2026-11-05, "now" as "naauu",
        // the ʊ part too long (a 203, ʊ 171 ms with 0.6: the short-vowel
        // lengthening below favours the shorter half).
        static const float kAwFirstShare = param("EN_AW_FIRST_SHARE", 0.6f);
        // ɔɪ's own split and length (EN_OY_FIRST_SHARE, EN_OY_LEN; OY agent 2026-11-10, lab: 0.6 and 1 = as before).
        static const float kOyFirstShare = param("EN_OY_FIRST_SHARE", 0.6f), kOyLen = param("EN_OY_LEN", 1.0f);
        const bool oy = s.n == 2 && s.ph[0] == Ph::AO && s.ph[1] == Ph::IH;
        // aɪ's own split and length (EN_AY_FIRST_SHARE, < 0: 0.6; EN_AY_LEN x the whole; AY agent 2026-11-10: Kokoro
        // holds the open start longer and glides late).
        static const float kAyFirstShare = param("EN_AY_FIRST_SHARE", -1.0f), kAyLen = param("EN_AY_LEN", 1.0f);
        // eɪ's own split and length (EN_EY_FIRST_SHARE, EN_EY_LEN; a word-final eɪ x EN_EY_FINAL_LEN as well; EY agent
        // 2026-11-10 lab knobs, 0.6 / 1 / 1 as before).
        static const float kEyFirstShare = param("EN_EY_FIRST_SHARE", 0.6f), kEyLen = param("EN_EY_LEN", 1.0f), kEyFinalLen = param("EN_EY_FINAL_LEN", 1.0f);
        const bool ey = s.n == 2 && s.ph[0] == Ph::E && s.ph[1] == Ph::IH;
        // eɪ's context for the line (Shape::eyctx) and its length / split in a closed syllable (EN_EY_CLOSED_LEN,
        // EN_EY_CLOSED_SHARE, < 0: EN_EY_FIRST_SHARE) and word-initially (EN_EY_INIT_LEN). EY agent10 round K lab knobs
        // (1 / -1 / 1: as before): eɪ was heard as ɪ 280 times in 6980 (combo9), 147 of them before t n k m p.
        static const float kEyClosedLen = param("EN_EY_CLOSED_LEN", 1.0f), kEyClosedShare = param("EN_EY_CLOSED_SHARE", -1.0f),
                           kEyInitLen = param("EN_EY_INIT_LEN", 1.0f);
        const bool eyClosed = ey && i + 1 < tokens.size() && !isVowelToken(tokens[i + 1]);
        if (ey) shape.eyctx = (eyClosed ? 1 : 0) | (eyClosed && isNasal(baseName(tokens[i + 1])) ? 2 : 0) | (i == 0 ? 4 : 0);
        // ɜ+ɹ: the vowel's share (EN_ER_SHARE), the whole ɜ x EN_ER_LEN, a stressed one x EN_ER_LONG_LEN as well (ER agent
        // 2026-11-11 lab knobs; 0.8 / 1 / 1 as before).
        static const float kErShare = param("EN_ER_SHARE", 0.8f), kErLen = param("EN_ER_LEN", 1.0f), kErLongLen = param("EN_ER_LONG_LEN", 1.0f);
        const float firstShare = erInit ? kErInitShare : s.ph[0] == Ph::ER ? (erOnset ? kErOnsetShare : kErShare)
                                 : s.n == 2 && s.ph[0] == Ph::A && s.ph[1] == Ph::UH ? kAwFirstShare : oy ? kOyFirstShare : ey ? (eyClosed && kEyClosedShare >= 0 ? kEyClosedShare : kEyFirstShare)
                                 : s.n == 2 && s.ph[0] == Ph::A && s.ph[1] == Ph::IH && kAyFirstShare >= 0 ? kAyFirstShare : 0.6f;
        const float share = s.n == 1 ? 1.0f : k == 0 ? firstShare : 1.0f - firstShare;
        shape.ms = dur[w][i] * share * v.length;
        {  // the word's planned total (Kokoro mode's word rhythm, prosody.h)
          float total = 0;
          for (size_t q = 0; q < dur[w].size(); ++q) total += dur[w][q];
          shape.wordMs = total;
        }
        const bool toLinked = kToBeforeVowel > 0.5f && phrase_[w].text == "to" && i == 1 &&
                              (s.ph[0] == Ph::U || s.ph[0] == Ph::UH) && w + 1 < phrase_.size() &&
                              !phrase_[w + 1].tokens.empty() && isVowelToken(phrase_[w + 1].tokens[0]);
        if (toLinked) shape.ms *= kToVowelLength;
        if (s.n == 2 && s.ph[1] != Ph::RR) shape.ms *= kDiphthongLength;  // a true diphthong, not ɜ+ɹ
        if (oy) shape.ms *= kOyLen;
        if (s.n == 2 && s.ph[0] == Ph::A && s.ph[1] == Ph::IH && kAyLen != 1.0f) shape.ms *= kAyLen;
        // ... and in a closed syllable only (a consonant follows in the word: "time", "like"; EN_AY_CLOSED_LEN).
        static const float kAyClosedLen = param("EN_AY_CLOSED_LEN", 1.0f);
        if (s.n == 2 && s.ph[0] == Ph::A && s.ph[1] == Ph::IH && kAyClosedLen != 1.0f && i + 1 < tokens.size() &&
            !isVowelToken(tokens[i + 1]))
          shape.ms *= kAyClosedLen;
        if (ey) shape.ms *= kEyLen * (i + 1 == tokens.size() ? kEyFinalLen : 1.0f) * (eyClosed ? kEyClosedLen : 1.0f) *
                            (i == 0 ? kEyInitLen : 1.0f);
        // iː lab knobs (IY agent 2026-11-11; 1 = as before): every iː x EN_IY_LEN; an iː right before a vowel in the
        // same word ("media", "create", "area": 33 % of them wrong, heard as j, the next ə as u) x EN_IY_HIATUS_LEN;
        // the vowel right after such an iː x EN_IY_NEXT_LEN.
        {
          static const float kIyLen = param("EN_IY_LEN", 1.0f), kIyHiatusLen = param("EN_IY_HIATUS_LEN", 1.0f),
                             kIyNextLen = param("EN_IY_NEXT_LEN", 1.0f);
          if (baseName(tokens[i]) == "IY") {
            shape.ms *= kIyLen;
            if (i + 1 < tokens.size() && isVowelToken(tokens[i + 1])) shape.ms *= kIyHiatusLen;
          }
          if (vowel && i > 0 && baseName(tokens[i - 1]) == "IY") shape.ms *= kIyNextLen;
        }
        // The standalone ɪ x EN_IH_LEN, in a one-syllable word x EN_IH_MONO_LEN as well (IH agent 2026-11-11 lab knobs; 1 / 1:
        // as before; "fill" 234 ms, "big" 250 ms on the male voice, near "feel"'s iː 247: a tense-vowel length).
        static const float kIhLen = param("EN_IH_LEN", 1.0f), kIhMonoLen = param("EN_IH_MONO_LEN", 1.0f);
        if (s.n == 1 && s.ph[0] == Ph::IH && (kIhLen != 1.0f || kIhMonoLen != 1.0f)) {
          int vowels = 0;
          for (size_t q = 0; q < tokens.size(); ++q) vowels += isVowelToken(tokens[q]);
          shape.ms *= kIhLen * (vowels == 1 ? kIhMonoLen : 1.0f);
        }
        // The standalone ɪ's context for the line (Shape::ihctx) and the stressed ɪ's length beside l / n (IH-STRESSED
        // agent14 round M lab knobs; 1: as before). A stressed ɪ beside l / n was heard as eɪ / ɛ ("Mill" -> mail, "Lin"
        // -> plan): against five Kokoro voices per sex (words1k) its DTW-mapped span is 181 vs 121 ms before l (male),
        // 145 vs 112 before n / m, 140 vs 97 after l ɹ w before a nasal; elsewhere 177 vs 174. x EN_IHS_L_LEN before an
        // l, EN_IHS_NAS_LEN before n / m, EN_IHS_LPREV_LEN after l ɹ w, EN_IHS_NPREV_LEN after m n; the l / n / m after
        // a stressed ɪ x EN_IHS_CODA_LEN.
        {
          static const float kIhsLLen = param("EN_IHS_L_LEN", 0.75f), kIhsNasLen = param("EN_IHS_NAS_LEN", 0.75f),
                             kIhsLPrevLen = param("EN_IHS_LPREV_LEN", 1.0f), kIhsNPrevLen = param("EN_IHS_NPREV_LEN", 1.0f),
                             kIhsCodaLen = param("EN_IHS_CODA_LEN", 1.4f);
          auto stressedIh = [&](size_t j) {
            const char* nm = kEnglishTokens[tokens[j] - 0x80];
            return baseName(tokens[j]) == "IH" && (nm[strlen(nm) - 1] == '1' || nm[strlen(nm) - 1] == '2');
          };
          if (s.n == 1 && s.ph[0] == Ph::IH) {
            const std::string pb = i > 0 ? baseName(tokens[i - 1]) : "", nb = i + 1 < tokens.size() ? baseName(tokens[i + 1]) : "";
            int vowels = 0;
            for (size_t q = 0; q < tokens.size(); ++q) vowels += isVowelToken(tokens[q]);
            const bool st = stressedIh(i);
            shape.ihctx = (st ? 1 : 0) | (nb == "L" ? 2 : 0) | (nb == "N" || nb == "M" ? 4 : 0) |
                          (pb == "L" || pb == "R" || pb == "W" ? 8 : 0) | (pb == "M" || pb == "N" ? 16 : 0) | (vowels == 1 ? 32 : 0);
            if (st) {
              if (nb == "L") shape.ms *= kIhsLLen;
              if (nb == "N" || nb == "M") shape.ms *= kIhsNasLen;
              if (pb == "L" || pb == "R" || pb == "W") shape.ms *= kIhsLPrevLen;
              if (pb == "M" || pb == "N") shape.ms *= kIhsNPrevLen;
            }
          }
          const std::string cb = baseName(tokens[i]);
          if (i > 0 && (cb == "L" || cb == "N" || cb == "M") && stressedIh(i - 1)) {
            shape.ihctx = 64;
            shape.ms *= kIhsCodaLen;
          }
        }
        // The unstressed ɜ after aʊ ("power", "hour", "flower") x this (EN_AW_ER_LEN;
        // Jyrki 2026-11-05: "the rhythm of the words is a little off, first
        // syllable should be stressed, 2nd unstressed": its ɜ ran 115 ms at the
        // first syllable's level, Kokoro's and 11labs' r tail is ~80 ms and a
        // continuous glide out of the ʊ).
        static const float kAwErLen = param("EN_AW_ER_LEN", 1.0f);
        if (s.ph[0] == Ph::ER && i > 0 && baseName(tokens[i - 1]) == "AW") shape.ms *= kAwErLen;
        if (s.ph[0] == Ph::ER && kErLen != 1.0f) shape.ms *= kErLen;
        if (s.ph[0] == Ph::ER && kErLongLen != 1.0f && en0Stressed(tokens[i])) shape.ms *= kErLongLen;
        if (erInit) shape.ms *= kErInitLen;
        // (EN_ER_INIT_AH_LEN / _AH_R_LEN: the rewritten ə / ɹ x these; 1: as before.)
        static const float kErInitAhLen = param("EN_ER_INIT_AH_LEN", 1.0f), kErInitAhRLen = param("EN_ER_INIT_AH_R_LEN", 1.0f);
        if (phrase_[w].erInitAh && i < 2) shape.ms *= i == 0 ? kErInitAhLen : kErInitAhRLen;
        if (erInit) shape.erctx = static_cast<unsigned char>(k + 1);
        else if (phrase_[w].erInitAh && i < 2) shape.erctx = static_cast<unsigned char>(i + 1);
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
        const bool initialR = i == 0 && s.ph[k] == Ph::RR && tokens.size() > 1 && isVowelToken(tokens[1]);
        if (initialR) {
          shape.ms *= kInitialRLength;
          shape.f3 = kInitialRF3;
        }
        // A coda ɹ before a consonant or a pause ("dare not", "first", "heirs") x EN_CODA_R_LEN: at 50 ms after a 148 ms
        // vowel it ran into the n, "dare" heard as "don't" (Jyrki 2026-11-06; Kokoro's coda r 10-20 ms, its vowel 50-90).
        {
          static const float kCodaRLen = param("EN_CODA_R_LEN", 1.25f);
          const bool codaR = s.ph[k] == Ph::RR && s.n == 1 && i > 0 && isVowelToken(tokens[i - 1]) &&
                             !(i + 1 < tokens.size() && isVowelToken(tokens[i + 1])) &&
                             !(i + 1 == tokens.size() && w + 1 < phrase_.size() && !phrase_[w + 1].tokens.empty() && isVowelToken(phrase_[w + 1].tokens[0]));
          if (codaR) shape.ms *= kCodaRLen;
          // (R-L agent19 round Q, lab; 1: as before) a coda ɹ before a z ("fares", "tears": heard l 33 of 160, the dark
          // ɹ running into a voiced alveolar) x EN_RZ_LEN, the vowel before that ɹ x EN_RZ_VLEN.
          static const float kRzLen = param("EN_RZ_LEN", 1.0f), kRzVLen = param("EN_RZ_VLEN", 1.0f);
          if (codaR && kRzLen != 1.0f && i + 1 < tokens.size() && baseName(tokens[i + 1]) == "Z") shape.ms *= kRzLen;
          if (kRzVLen != 1.0f && vowel && s.n == 1 && i + 2 < tokens.size() && baseName(tokens[i + 1]) == "R" &&
              baseName(tokens[i + 2]) == "Z")
            shape.ms *= kRzVLen;
          // (R-POSTVOCALIC agent14 round M, lab; 1: as before) the vowel before a coda ɹ x EN_PRE_CODA_R_VLEN (with
          // EN_CODA_R_LEN: the r-coloured tail longer at the same word length; 11labs "dare", "share": ɹ 30-100 ms).
          static const float kPreCodaRVLen = param("EN_PRE_CODA_R_VLEN", 1.0f);
          if (kPreCodaRVLen != 1.0f && vowel && s.n == 1 && i + 1 < tokens.size() && baseName(tokens[i + 1]) == "R" &&
              !(i + 2 < tokens.size() && isVowelToken(tokens[i + 2])) &&
              !(i + 2 == tokens.size() && w + 1 < phrase_.size() && !phrase_[w + 1].tokens.empty() && isVowelToken(phrase_[w + 1].tokens[0])))
            shape.ms *= kPreCodaRVLen;
        }
        if (phrase_[w].syllabicN && i + 1 == tokens.size()) shape.ms *= kSyllabicNLength;
        if (phrase_[w].syllabicL && i + 1 == tokens.size()) shape.ms *= kSyllabicLLength;
        if (i < 32 && (phrase_[w].flaps >> i & 1)) {
          shape.flap = true;
          shape.ms = kFlapMs;
        }
        // The d's context for the line's d knobs (D agent9 round J): 1 between a vowel and an unstressed vowel of
        // its word, 2 before s z, 3 at the word's end after a nasal or l, 4 at the word's end after a vowel, 5 before
        // ɹ, 6 at the word's start before a vowel, 7 between l and an unstressed vowel ("builder"). EN_FLAP_D 1: a d in context 1 is a tap (kFlapMs, as kFlapping 1
        // but the t keeps its full closure: the t / d contrast is then tap against stop).
        // The g's context for the line's initial-g knobs (G-INITIAL agent14 round M): 1 at the word's start before a vowel.
        if (baseName(tokens[i]) == "G" && i == 0 && i + 1 < tokens.size() && isVowelToken(tokens[i + 1])) shape.gctx = 1;
        // (VELAR-L agent15 round N, lab; 1: as before) a word-initial k / g before l ɹ x EN_VL_LEN, the l / ɹ after it x
        // EN_VL_LIQ_LEN.
        {
          static const float kVlLen = param("EN_VL_LEN", 1.0f), kVlLiqLen = param("EN_VL_LIQ_LEN", 1.0f);
          auto velar = [&](size_t j) { const std::string b = baseName(tokens[j]); return b == "K" || b == "G"; };
          auto liquid = [&](size_t j) { const std::string b = baseName(tokens[j]); return b == "L" || b == "R"; };
          if (kVlLen != 1.0f && i == 0 && tokens.size() > 1 && velar(0) && liquid(1)) shape.ms *= kVlLen;
          if (kVlLiqLen != 1.0f && i == 1 && velar(0) && liquid(1)) shape.ms *= kVlLiqLen;
        }
        // (PL-INITIAL agent17 round O, lab; 1: as before) a word-initial p before l ɹ x EN_PL_LEN, the l / ɹ after it x
        // EN_PL_LIQ_LEN.
        {
          static const float kPlLen = param("EN_PL_LEN", 1.0f), kPlLiqLen = param("EN_PL_LIQ_LEN", 1.0f);
          auto liquid = [&](size_t j) { const std::string b = baseName(tokens[j]); return b == "L" || b == "R"; };
          const bool pl = tokens.size() > 1 && baseName(tokens[0]) == "P" && liquid(1);
          if (kPlLen != 1.0f && i == 0 && pl) shape.ms *= kPlLen;
          if (kPlLiqLen != 1.0f && i == 1 && pl) shape.ms *= kPlLiqLen;
        }
        // (PK-INITIAL agent19 round Q, lab; 1: as before) a word-initial p / k before a vowel x EN_PV_LEN / EN_KV_LEN.
        {
          static const float kPvLen = param("EN_PV_LEN", 1.0f), kKvLen = param("EN_KV_LEN", 1.0f);
          if (i == 0 && tokens.size() > 1 && isVowelToken(tokens[1])) {
            const std::string b0 = baseName(tokens[0]);
            if (b0 == "P" && kPvLen != 1.0f) shape.ms *= kPvLen;
            if (b0 == "K" && kKvLen != 1.0f) shape.ms *= kKvLen;
          }
        }
        if (baseName(tokens[i]) == "D") {
          static const float kFlapD = param("EN_FLAP_D", 0.0f), kFlapDMs = param("EN_FLAP_D_MS", -1.0f);
          const bool pv = i > 0 && isVowelToken(tokens[i - 1]), nv = i + 1 < tokens.size() && isVowelToken(tokens[i + 1]);
          const std::string pb = i > 0 ? baseName(tokens[i - 1]) : "", nb = i + 1 < tokens.size() ? baseName(tokens[i + 1]) : "";
          if (pv && nv && kEnglishTokens[tokens[i + 1] - 0x80][strlen(kEnglishTokens[tokens[i + 1] - 0x80]) - 1] == '0') shape.dctx = 1;
          else if (nb == "Z" || nb == "S") shape.dctx = 2;
          else if (i + 1 == tokens.size() && (isNasal(pb) || pb == "L")) shape.dctx = 3;
          else if (i + 1 == tokens.size() && pv) shape.dctx = 4;
          else if (nb == "R") shape.dctx = 5;
          else if (i == 0 && nv) shape.dctx = 6;
          else if (pb == "L" && nv && kEnglishTokens[tokens[i + 1] - 0x80][strlen(kEnglishTokens[tokens[i + 1] - 0x80]) - 1] == '0') shape.dctx = 7;
          if (shape.dctx == 1 && kFlapD > 0.5f && !shape.flap) {
            shape.flap = true;
            shape.ms = kFlapDMs >= 0 ? kFlapDMs : kFlapMs;
          }
        }
        // The final unstressed n's context for the line (Shape::nctx; N agent13 round L): 1 the unstressed vowel right
        // before the word's final n, 2 that n ("Gavin", "Linen", "Aspen": the -in / -en heard as -ing).
        {
          auto unstressedAt = [&](size_t j) {
            const char* nm = kEnglishTokens[tokens[j] - 0x80];
            return isVowelToken(tokens[j]) && nm[strlen(nm) - 1] == '0';
          };
          if (i + 1 == tokens.size() && i > 0 && baseName(tokens[i]) == "N" && unstressedAt(i - 1)) shape.nctx = 2;
          if (i + 2 == tokens.size() && baseName(tokens[i + 1]) == "N" && unstressedAt(i)) shape.nctx = 1;
        }
        // An unstressed ə before a nasal of its word (Shape::axctx; AX-NASAL agent14 round M): 1, | 2 the word's first
        // vowel after one consonant ("Commands", "Commit"), | 4 after t / d ("Sutton", "Weston"), | 8 after a vowel ("Lions").
        if (i > 0 && i + 1 < tokens.size() && baseName(tokens[i]) == "AH" &&
            kEnglishTokens[tokens[i] - 0x80][strlen(kEnglishTokens[tokens[i] - 0x80]) - 1] == '0') {
          const std::string nb = baseName(tokens[i + 1]), pb = baseName(tokens[i - 1]);
          if (nb == "M" || nb == "N" || nb == "NG") {
            shape.axctx = 1;
            if (i == 1 && !isVowelToken(tokens[0])) shape.axctx |= 2;
            if (pb == "T" || pb == "D") shape.axctx |= 4;
            if (isVowelToken(tokens[i - 1])) shape.axctx |= 8;
          }
        }
        // Plosives: Klatt's duration includes the release; this is the closure.
        if (isVoicelessPlosive(baseName(tokens[i])) || isVoicedPlosive(baseName(tokens[i])))
          shape.ms *= 0.75f;
        // ʤ x EN_JH_LEN (Jyrki 2026-11-06: the g of "generosity" "too k like", should be "more of z-like, more voiced":
        // its closure ran 92 ms against Kokoro's 70 ms voiced affricate).
        static const float kJhLen = param("EN_JH_LEN", 1.0f);
        if (baseName(tokens[i]) == "JH") shape.ms *= kJhLen;
        // The back offglide fades: about 3 dB weaker than a full vowel.
        shape.level = glide ? 0.7f * v.level : v.level;
        if (initialR) shape.level *= kInitialRLevel;
        // Voice quality: accented vowels slightly pressed (brighter, more
        // closed phase), unstressed and phrase-final ones slightly lax.
        if (vowel) {
          const char* vq = kEnglishTokens[tokens[i] - 0x80];
          const bool accented = phrase_[w].accent && vq[strlen(vq) - 1] == '1';
          shape.press = accented ? kPressedAccent
                        : si == syl.size() ? kLaxFinal : kLaxUnstressed;
        }
        if (ay && kAyPress != 1.0f) shape.press *= kAyPress;
        // The ɜ's voice quality x EN_ER_PRESS (ER agent 2026-11-11 lab knob, 1: as before).
        static const float kErPress = param("EN_ER_PRESS", 1.0f);
        if (vowel && s.ph[k] == Ph::ER && kErPress != 1.0f) shape.press *= kErPress;
        // (EN_ER_INIT_PRESS: the initial ə of EN_ER_INIT's voice quality x this; 1: as before.)
        static const float kErInitPress = param("EN_ER_INIT_PRESS", 1.0f);
        if (erInit && k == 0 && kErInitPress != 1.0f) shape.press *= kErInitPress;
        // The ɛ's voice quality x EN_EH_PRESS (round F EH agent 2026-11-11 lab knob, 1: as before): a closer ɛ (lower F1)
        // takes 0.5-0.7 dB off the 1-3 kHz share of the ɛ words, already 5-7 dB under Kokoro's and 11labs'.
        static const float kEhPress = param("EN_EH_PRESS", 1.0f);
        if (vowel && s.ph[k] == Ph::EH && kEhPress != 1.0f) shape.press *= kEhPress;
        // Loudness: an arc over the phrase (louder at the start, softer
        // toward its end), accents louder, function words softer. Vowels
        // only (the consonants keep their calibrated levels).
        if (vowel) {
          const float frac = syl.size() > 1 ? static_cast<float>(si - 1) / (syl.size() - 1) : 0.0f;
          float db = kArcStartDb + (kArcEndDb - kArcStartDb) * frac;
          const char* vn = kEnglishTokens[tokens[i] - 0x80];
          if (sentenceEnd_ && lastEnd_ == '!') db += kExclaimDb;
          // (PROSODY agent11, lab knobs; 0: as before) the level follows the breath group: EN_BG_DB dB from its start
          // (+half) to its end (-half), EN_AIR_DB by the planned fullness, and EN_BG_FINAL_DB more over the group's
          // last part (from EN_BG_FINAL_FROM of it): 11labs fall 6.3 dB from a group's first words to its last (the
          // last word 3.2 under the group's mean), puhuja 1.9 (1.2).
          {
            static const float kBgDb = param("EN_BG_DB", 0.0f), kAirDb = param("EN_AIR_DB", 0.0f),
                               kBgFinDb = param("EN_BG_FINAL_DB", 1.0f), kBgStartDb = param("EN_BG_START_DB", 0.8f);
            if ((kBgDb != 0 || kAirDb != 0 || kBgFinDb != 0 || kBgStartDb != 0) && w < wordGroupPos_.size()) {
              const float g = wordGroupPos_[w];
              db += kBgDb * (0.5f - g) + kAirDb * (wordAirU_[w] - 0.5f) +
                    kBgStartDb * (1.0f - bgStep(0.0f, bgStartTo(), g)) - kBgFinDb * bgStep(bgFinalFrom(), 1.0f, g);
            }
          }
          if (phrase_[w].accent && vn[strlen(vn) - 1] == '1') db += kAccentDb;
          else if (!phrase_[w].accent) db += kFunctionWordDb;
          { static const float kEmoDb = param("EN_EMOTIVE_DB", 1.5f); if (kEmoDb != 0 && isEmotiveWord(phrase_[w].text)) db += kEmoDb; }
          if (ay && k == 0) db += kAyADb;
          // (round I EH agent 2026-11-12 lab knob; 0: as before) the ɛ's level + EN_EH_DB.
          { static const float kEhDb = param("EN_EH_DB", 0.0f); if (kEhDb != 0 && s.ph[k] == Ph::EH) db += kEhDb; }
          // (round-J ə agent 9, lab knobs; 0: as before) a word-initial unstressed ə's level + EN_SCHWA_INIT_DB (106 of its
          // 351 dictionary errors are deletions: "allowed" -> loud, "asleep" -> sleep), an ə in its word's last syllable
          // before a consonant + EN_SCHWA_LAST_DB.
          { static const float kSwInitDb = param("EN_SCHWA_INIT_DB", 0.0f), kSwLastDb = param("EN_SCHWA_LAST_DB", 0.0f);
            if (s.ph[k] == Ph::Schwa && vn[strlen(vn) - 1] == '0' && baseName(tokens[i]) == "AH") {
              int lastV = -1;
              for (size_t j = 0; j < tokens.size(); ++j) if (isVowelToken(tokens[j])) lastV = static_cast<int>(j);
              if (i == 0 && i + 1 < tokens.size()) db += kSwInitDb;
              else if (static_cast<int>(i) == lastV && i + 1 < tokens.size()) db += kSwLastDb;
            } }
          // (AX-NASAL agent14 round M, lab knob; 0: as before) an ə before a nasal of its word (Shape::axctx) + EN_AXN_DB.
          { static const float kAxnDb = param("EN_AXN_DB", 0.0f); if (kAxnDb != 0 && s.ph[k] == Ph::Schwa && shape.axctx) db += kAxnDb; }
          // Secondary stress is a step down from the primary, not its equal:
          // Kokoro's "robot", "habitat", "antibody" end 5-15 dB under their
          // first syllable; here the last third was the loudest (2026-11-02).
          // An unstressed syllable, by contrast, was too weak (-18..-24 dB re
          // the word's peak against Kokoro's -13..-17).
          // ... but a secondary stress BEFORE the word's primary ("generosity", "revolutionary", "generation") is the
          // word's onset and Kokoro's loudest syllable there: its own level (kSecondaryPreDb; Jyrki 2026-11-06: "the
          // voiced g in generosity is missing": its ʤɛ sat 10-13 dB under the following unstressed ɜ).
          bool primaryLater = false;
          for (size_t j = i + 1; j < tokens.size(); ++j) {
            const char* tn = kEnglishTokens[tokens[j] - 0x80];
            if (isVowelToken(tokens[j]) && tn[strlen(tn) - 1] == '1') primaryLater = true;
          }
          if (vn[strlen(vn) - 1] == '2') db += primaryLater ? kSecondaryPreDb : kSecondaryDb;
          else if (vn[strlen(vn) - 1] == '0') db += kUnstressedDb;
          // ... and the ɜ of "power", "hour" a step under its aʊ (EN_AW_ER_DB; with
          // kUnstressedDb +2 it was the louder syllable, -11 dB re the phrase
          // against the aʊ's -13; Kokoro and 11labs: 2-5 dB under).
          static const float kAwErDb = param("EN_AW_ER_DB", 0.0f);
          if (i > 0 && baseName(tokens[i]) == "ER" && baseName(tokens[i - 1]) == "AW") db += kAwErDb;
          // The ɜ's level, dB (EN_ER_DB; a stressed one + EN_ER_LONG_DB; ER agent 2026-11-11 lab knobs, 0: as before).
          static const float kErDb = param("EN_ER_DB", 0.0f), kErLongDb = param("EN_ER_LONG_DB", 0.0f);
          if (baseName(tokens[i]) == "ER" && s.ph[k] == Ph::ER) db += kErDb + (vn[strlen(vn) - 1] == '1' ? kErLongDb : 0.0f);
          if (erInit) db += k == 0 ? kErInitDb : kErInitRDb;
          // (EN_ER_INIT_NEXT_DB: the vowel right after such an ə + ɹ + this, dB; 0: as before. Kokoro's stressed vowel
          // comes back to the ə's level after the ɹ's dip, puhuja's sat 5-6 dB under its onset.)
          static const float kErInitNextDb = param("EN_ER_INIT_NEXT_DB", 0.0f);
          if (kErInitNextDb != 0 && ((kErInit > 0 && i == 1 && tokens.size() > 1 && baseName(tokens[0]) == "ER" &&
                                      kEnglishTokens[tokens[0] - 0x80][2] == '0') ||
                                     (phrase_[w].erInitAh && i == 2)))
            db += kErInitNextDb;
          // The ʊ half of aʊ x this (EN_AW_UH_DB, dB; 2026-11-09 AW agent: Kokoro's aʊ fades 4-6 dB from the a to the ʊ end, puhuja's is flat).
          static const float kAwUhDb = param("EN_AW_UH_DB", 0.0f);
          if (s.n == 2 && s.ph[0] == Ph::A && s.ph[1] == Ph::UH && k == 1) db += kAwUhDb;
          // The standalone ʊ's level, dB (EN_UH_DB; round-H ʊ agent, lab; 0: as before).
          static const float kUhDb = param("EN_UH_DB", 0.0f);
          if (s.n == 1 && s.ph[0] == Ph::UH) db += kUhDb;
          // The ɔ and ɪ halves of ɔɪ, dB (EN_OY_AO_DB, EN_OY_IH_DB; OY agent 2026-11-10, lab: 0 = as before).
          static const float kOyAoDb = param("EN_OY_AO_DB", 0.0f), kOyIhDb = param("EN_OY_IH_DB", 0.0f);
          if (s.n == 2 && s.ph[0] == Ph::AO && s.ph[1] == Ph::IH) db += k == 0 ? kOyAoDb : kOyIhDb;
          // eɪ's halves x these (EN_EY_E_DB, EN_EY_I_DB, dB; EY agent 2026-11-10 lab knobs, 0: as before).
          static const float kEyEDb = param("EN_EY_E_DB", 0.0f), kEyIDb = param("EN_EY_I_DB", 0.0f);
          if (s.n == 2 && s.ph[0] == Ph::E && s.ph[1] == Ph::IH) db += k == 0 ? kEyEDb : kEyIDb;
          shape.level *= std::pow(10.0f, db / 20.0f);
          shape.prosody = std::pow(10.0f, db / 20.0f);
        }
        synth_.shapeNext(shape);
        if (first) synth_.setWord(phrase_[w].source), lastSourceWord_ = phrase_[w].source;
        synth_.push(s.ph[k], first);
        first = false;
        // A stressed ɑ before another syllable of the same word is the short
        // "lot" vowel (see the durations): "poverty", "body", "problem".
        bool shortLot = baseName(tokens[i]) == "AA";
        if (shortLot) {
          shortLot = false;
          for (size_t j = i + 1; j < tokens.size(); ++j) shortLot |= isVowelToken(tokens[j]);
        }
        if (k == 0 && s.longVowel && !shortLot) synth_.lengthen();
      }
      // Linking: a word ending in a high vowel or a diphthong's glide runs
      // into a vowel-initial next word through a short w or j ("to assure"
      // tuː w‿əʃʊɹ, "the heirs" ði j‿ɛɹz). Without it the two vowels blurred
      // into one ("to assure" heard as "to show").
      // ... and inside a word (EN_INNER_GLIDE, 2026-11-04): aʊ before the
      // r vowel or a vowel ("power", "flower", "hour": goog and 11labs dip F2
      // to 900-1150 Hz with F1 ~450-600 between the two, a w; puhuja's F2
      // sat flat at 1250), and a high vowel or glide before another vowel
      // ("going", "player" with EN_INNER_GLIDE 2).
      // ... and after uː only (EN_UW_INNER_GLIDE, round-I uː agent, lab; 0: off): a uː before a vowel inside a word
      // ("viewer" -> year, "newer" -> gear, "sewer" -> sir, "ruin"; 26 % wrong, 38 of them deleted) gets the w link.
      static const float kUwInnerGlide = param("EN_UW_INNER_GLIDE", 0.0f);
      const bool innerNext = vowel && i + 1 < tokens.size() &&
                             ((kInnerGlide >= 1 && baseName(tokens[i]) == "AW" &&
                               (isVowelToken(tokens[i + 1]) || baseName(tokens[i + 1]) == "R")) ||
                              (kInnerGlide >= 2 && isVowelToken(tokens[i + 1])) ||
                              (kUwInnerGlide > 0.5f && baseName(tokens[i]) == "UW" && isVowelToken(tokens[i + 1])));
      if ((vowel && i + 1 == tokens.size() && w + 1 < phrase_.size() &&
           !phrase_[w + 1].tokens.empty() && isVowelToken(phrase_[w + 1].tokens[0])) || innerNext) {
        const Ph last = s.ph[s.n - 1];
        const Ph link = last == Ph::U || last == Ph::UH ? Ph::W
                        : last == Ph::I || last == Ph::IH ? Ph::J : Ph::Stop;
        const bool toNoGlide = kToGlide < 0.5f && kToBeforeVowel > 0.5f && phrase_[w].text == "to";
        if (link != Ph::Stop && !toNoGlide) {
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
  synth_.planSentence();  // the formant and pitch paths of the phrase, as a whole
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
  if (!word_.empty()) ++sourceWord_;
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
      for (char ch : word_) {
        phrase_.push_back({arpaTokens(kLetterNames[ch - 'a']), ch == 'm', std::string(1, ch)});
        phrase_.back().source = sourceWord_;
      }
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
  // The text is read a sentence at a time: what follows the last sentence
  // end (. ! ? before white space, or a line end) waits for the next call,
  // so that each phrase is spoken knowing the rest of its sentence (the
  // breath planner, planBreaths). A very long stretch is read anyway.
  if (!flushing_) {
    size_t cut = 0;
    for (size_t k = text.size(); k-- > 0;) {
      const char c = text[k];
      if (c == '\n' || ((c == '.' || c == '!' || c == '?') && k + 1 < text.size() &&
                         (text[k + 1] == ' ' || text[k + 1] == '\t' || text[k + 1] == '\r' || text[k + 1] == '\n'))) {
        // PLAN_INVARIANT 1: the white space after the mark goes with it. Cut at the mark, the sentence's last word
        // ("life.") had no space after it and the rule below held it back too, so the breath planner's rest of
        // the sentence missed it: input fed in pieces (-chunk 7) breathed differently from whole lines (2026-11-11).
        static const bool kInvariant = param("PLAN_INVARIANT", 1.0f) > 0;
        cut = k + (kInvariant && c != '\n' ? 2 : 1);
        break;
      }
    }
    if (cut < text.size() && text.size() - cut < kMaxHeldSentence) {
      pending_ = std::string(text.substr(cut)) + pending_;
      text = text.substr(0, cut);
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
  lookahead_ = std::string(text);
  for (size_t i = 0; i < text.size(); ++i) {
    lookPos_ = i;
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
          synth_.breathPoint(true, true);  // a paragraph: the longest breath
        }
        break;
      default:
        break;  // spaces and other symbols only separate words
    }
  }
}

}  // namespace puhuja
