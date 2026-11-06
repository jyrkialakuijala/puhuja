#include "prosody.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>

namespace puhuja {
namespace {

// A small JSON reader: enough for the profile (objects, arrays, strings with
// the usual escapes, numbers, true/false/null). No dependency, no surprises.
struct Json {
  enum Type { kNull, kBool, kNumber, kString, kArray, kObject } type = kNull;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;
  const Json* get(const char* key) const {
    if (type != kObject) return nullptr;
    auto it = obj.find(key);
    return it == obj.end() ? nullptr : &it->second;
  }
  double number(double dflt) const { return type == kNumber ? num : dflt; }
};

class Parser {
 public:
  Parser(const std::string& s, std::string* err) : s_(s), err_(err) {}
  bool parse(Json* out) {
    ws();
    if (!value(out)) return false;
    ws();
    if (p_ != s_.size()) return fail("trailing characters");
    return true;
  }

 private:
  bool fail(const char* what) {
    if (err_ && err_->empty()) *err_ = std::string(what) + " at byte " + std::to_string(p_);
    return false;
  }
  void ws() { while (p_ < s_.size() && std::strchr(" \t\r\n", s_[p_])) ++p_; }
  bool value(Json* v) {
    if (p_ >= s_.size()) return fail("unexpected end");
    const char c = s_[p_];
    if (c == '{') return object(v);
    if (c == '[') return array(v);
    if (c == '"') { v->type = Json::kString; return string(&v->str); }
    if (c == 't' || c == 'f') {
      const bool t = c == 't';
      const char* w = t ? "true" : "false";
      if (s_.compare(p_, std::strlen(w), w) != 0) return fail("bad literal");
      p_ += std::strlen(w); v->type = Json::kBool; v->b = t; return true;
    }
    if (c == 'n') {
      if (s_.compare(p_, 4, "null") != 0) return fail("bad literal");
      p_ += 4; v->type = Json::kNull; return true;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
      char* end = nullptr;
      v->num = std::strtod(s_.c_str() + p_, &end);
      if (end == s_.c_str() + p_) return fail("bad number");
      p_ = static_cast<size_t>(end - s_.c_str()); v->type = Json::kNumber; return true;
    }
    return fail("unexpected character");
  }
  bool string(std::string* out) {
    ++p_;  // the opening quote
    while (p_ < s_.size()) {
      const char c = s_[p_++];
      if (c == '"') return true;
      if (c != '\\') { out->push_back(c); continue; }
      if (p_ >= s_.size()) break;
      const char e = s_[p_++];
      switch (e) {
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case '/': out->push_back('/'); break;
        case 'b': out->push_back('\b'); break;
        case 'f': out->push_back('\f'); break;
        case 'n': out->push_back('\n'); break;
        case 'r': out->push_back('\r'); break;
        case 't': out->push_back('\t'); break;
        case 'u': {
          if (p_ + 4 > s_.size()) return fail("bad \\u escape");
          unsigned cp = static_cast<unsigned>(std::strtoul(s_.substr(p_, 4).c_str(), nullptr, 16));
          p_ += 4;
          if (cp >= 0xD800 && cp <= 0xDBFF && p_ + 6 <= s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') {  // a surrogate pair
            const unsigned lo = static_cast<unsigned>(std::strtoul(s_.substr(p_ + 2, 4).c_str(), nullptr, 16));
            if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); p_ += 6; }
          }
          if (cp < 0x80) out->push_back(static_cast<char>(cp));
          else if (cp < 0x800) { out->push_back(static_cast<char>(0xC0 | (cp >> 6))); out->push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
          else if (cp < 0x10000) { out->push_back(static_cast<char>(0xE0 | (cp >> 12))); out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F))); out->push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
          else { out->push_back(static_cast<char>(0xF0 | (cp >> 18))); out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F))); out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F))); out->push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
          break;
        }
        default: return fail("bad escape");
      }
    }
    return fail("unterminated string");
  }
  bool array(Json* v) {
    v->type = Json::kArray;
    ++p_; ws();
    if (p_ < s_.size() && s_[p_] == ']') { ++p_; return true; }
    for (;;) {
      v->arr.emplace_back();
      ws();
      if (!value(&v->arr.back())) return false;
      ws();
      if (p_ >= s_.size()) return fail("unterminated array");
      if (s_[p_] == ',') { ++p_; continue; }
      if (s_[p_] == ']') { ++p_; return true; }
      return fail("expected , or ]");
    }
  }
  bool object(Json* v) {
    v->type = Json::kObject;
    ++p_; ws();
    if (p_ < s_.size() && s_[p_] == '}') { ++p_; return true; }
    for (;;) {
      ws();
      if (p_ >= s_.size() || s_[p_] != '"') return fail("expected a key");
      std::string key;
      if (!string(&key)) return false;
      ws();
      if (p_ >= s_.size() || s_[p_] != ':') return fail("expected :");
      ++p_; ws();
      if (!value(&v->obj[key])) return false;
      ws();
      if (p_ >= s_.size()) return fail("unterminated object");
      if (s_[p_] == ',') { ++p_; continue; }
      if (s_[p_] == '}') { ++p_; return true; }
      return fail("expected , or }");
    }
  }
  const std::string& s_;
  std::string* err_;
  size_t p_ = 0;
};

float numberOr(const Json* j, float dflt) { return j && j->type == Json::kNumber ? static_cast<float>(j->num) : dflt; }

}  // namespace

bool ProsodyProfile::load(const std::string& path, std::string* err) {
  std::string text;
  if (FILE* f = std::fopen(path.c_str(), "rb")) {
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    std::fclose(f);
  } else {
    if (err) *err = "cannot open " + path;
    return false;
  }
  Json root;
  std::string perr;
  if (!Parser(text, &perr).parse(&root)) {
    if (err) *err = path + ": " + perr;
    return false;
  }
  if (root.type != Json::kObject) { if (err) *err = path + ": not an object"; return false; }
  version = static_cast<int>(numberOr(root.get("version"), 0));
  if (version != 1) { if (err) *err = path + ": version " + std::to_string(version) + " (want 1)"; return false; }
  if (const Json* j = root.get("source")) source = j->str;
  if (const Json* j = root.get("text")) text = j->str, this->text = j->str;
  baseF0 = numberOr(root.get("base_f0_hz"), 0.0f);
  if (const Json* j = root.get("labels"); j && j->type == Json::kArray) {
    for (const Json& e : j->arr) {
      Label L;
      if (const Json* s = e.get("sym")) L.sym = s->str;
      L.ms = numberOr(e.get("ms"), 0.0f);
      if (const Json* f = e.get("f0"); f && f->type == Json::kArray) {
        if (f->arr.size() > 0) L.pitch = numberOr(&f->arr[0], kNone);
        if (f->arr.size() > 1) L.pitchEnd = numberOr(&f->arr[1], kNone);
      }
      labels.push_back(L);
    }
  }
  if (const Json* j = root.get("words"); j && j->type == Json::kArray) {
    for (const Json& e : j->arr) {
      Word w;
      if (const Json* s = e.get("text")) w.text = s->str;
      if (const Json* t = e.get("t"); t && t->type == Json::kArray && t->arr.size() == 2) { w.t0 = t->arr[0].number(0); w.t1 = t->arr[1].number(0); }
      if (const Json* f = e.get("f0"); f && f->type == Json::kArray) for (const Json& v : f->arr) w.f0.push_back(numberOr(&v, kNone));
      w.level = numberOr(e.get("level"), kNone);
      w.pauseAfter = numberOr(e.get("pause_after"), 0.0f);
      words.push_back(w);
    }
  }
  // Punctuation-only tokens (Kokoro's timestamps carry "." and "," as words of their own): their span and gap go to
  // the word before them, so a sentence's end keeps its pause.
  {
    std::vector<Word> kept;
    for (const Word& w : words) {
      bool lettered = false;
      for (unsigned char c : w.text) if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0x80) { lettered = true; break; }
      if (lettered || kept.empty()) kept.push_back(w);
      else kept.back().pauseAfter += static_cast<float>(w.t1 - w.t0) + w.pauseAfter + static_cast<float>(w.t0 - kept.back().t1);
    }
    words.swap(kept);
  }
  if (const Json* j = root.get("tracks"); j && j->type == Json::kObject) {
    tracks.t0 = numberOr(j->get("t0"), 0.0f);
    tracks.hop = numberOr(j->get("hop"), 0.0f);
    if (const Json* f = j->get("f0_st"); f && f->type == Json::kArray) for (const Json& v : f->arr) tracks.f0St.push_back(numberOr(&v, kNone));
    if (const Json* f = j->get("energy_db"); f && f->type == Json::kArray) for (const Json& v : f->arr) tracks.energyDb.push_back(numberOr(&v, kNone));
    if (tracks.hop <= 0 && !tracks.empty()) { if (err) *err = path + ": tracks without a hop"; return false; }
  }
  if (const Json* j = root.get("phones"); j && j->type == Json::kArray) {
    for (const Json& e : j->arr) {
      Phone p;
      if (const Json* s = e.get("sym")) p.sym = s->str;
      if (const Json* t = e.get("t"); t && t->type == Json::kArray && t->arr.size() == 2) { p.t0 = t->arr[0].number(0); p.t1 = t->arr[1].number(0); }
      phones.push_back(p);
    }
  }
  return true;
}

namespace {
std::string normWord(const std::string& w) {
  std::string o;
  for (unsigned char c : w) {
    if (c >= 'A' && c <= 'Z') o.push_back(static_cast<char>(c - 'A' + 'a'));
    else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c >= 0x80) o.push_back(static_cast<char>(c));
  }
  return o;
}
}  // namespace

void ProsodyGuide::setInputWords(const std::vector<std::string>& words) {
  // Longest common subsequence of the normalised spellings; unmatched input
  // words get -1 (puhuja's own timing).
  std::vector<std::string> a, b;
  for (const auto& w : words) a.push_back(normWord(w));
  for (const auto& w : profile_.words) b.push_back(normWord(w.text));
  const size_t n = a.size(), m = b.size();
  std::vector<std::vector<int>> L(n + 1, std::vector<int>(m + 1, 0));
  for (size_t i = n; i-- > 0;)
    for (size_t j = m; j-- > 0;)
      L[i][j] = a[i] == b[j] && !a[i].empty() ? L[i + 1][j + 1] + 1 : std::max(L[i + 1][j], L[i][j + 1]);
  wordMap_.assign(n, -1);
  for (size_t i = 0, j = 0; i < n && j < m;) {
    if (a[i] == b[j] && !a[i].empty()) { wordMap_[i] = static_cast<int>(j); ++i; ++j; }
    else if (L[i + 1][j] >= L[i][j + 1]) ++i;
    else ++j;
  }
}

bool ProsodyGuide::wordScale(int inputWord, float plannedMs, float* scale) const {
  *scale = 1.0f;
  if (!has(kRhythm) || inputWord < 0 || inputWord >= static_cast<int>(wordMap_.size()) || wordMap_[inputWord] < 0 || plannedMs <= 0) return false;
  const auto& w = profile_.words[wordMap_[inputWord]];
  const double refMs = (w.t1 - w.t0) * 1000.0;
  if (refMs <= 0) return false;
  *scale = static_cast<float>(std::min(2.5, std::max(0.4, refMs / plannedMs)));  // (a mismatched word must not explode)
  return true;
}

void ProsodyGuide::wordStart(int inputWord, double renderT0, double renderMs) {
  if (inputWord < 0 || inputWord >= static_cast<int>(wordMap_.size()) || wordMap_[inputWord] < 0 || renderMs <= 0) return;
  const auto& w = profile_.words[wordMap_[inputWord]];
  if (!warp_.empty() && warp_.back().r1 > renderT0) warp_.back().r1 = renderT0;  // the previous word ends where this one starts
  warp_.push_back({renderT0, renderT0 + renderMs * 0.001, w.t0, w.t1, wordMap_[inputWord]});
}

bool ProsodyGuide::refTime(double renderT, double* refT, int* word) {
  if (warp_.empty()) return false;
  while (warpAt_ + 1 < warp_.size() && renderT >= warp_[warpAt_ + 1].r0) ++warpAt_;
  while (warpAt_ > 0 && renderT < warp_[warpAt_].r0) --warpAt_;
  const Warp& w = warp_[warpAt_];
  if (renderT < w.r0 || renderT >= w.r1) return false;  // between words (a pause): the reference's own gap is not followed
  const double f = (renderT - w.r0) / (w.r1 - w.r0);
  *refT = w.ref0 + f * (w.ref1 - w.ref0);
  if (word) *word = w.word;
  return true;
}

bool ProsodyGuide::f0At(double renderT, float* st) {
  if (!has(kPitch) || profile_.tracks.f0St.empty()) return false;
  double t;
  if (!refTime(renderT, &t)) return false;
  const auto& T = profile_.tracks;
  const long k = std::lround((t - T.t0) / T.hop);
  if (k < 0 || k >= static_cast<long>(T.f0St.size()) || T.f0St[k] == ProsodyProfile::kNone) return false;
  *st = T.f0St[k];
  return true;
}

bool ProsodyGuide::levelAt(double renderT, float* db, bool perWord) {
  if (!has(kLevel)) return false;
  double t;
  int word = -1;
  if (!refTime(renderT, &t, &word)) return false;
  if (perWord) {
    if (word < 0 || profile_.words[word].level == ProsodyProfile::kNone) return false;
    *db = profile_.words[word].level;
    return true;
  }
  if (profile_.tracks.energyDb.empty()) return false;
  const auto& T = profile_.tracks;
  const long k = std::lround((t - T.t0) / T.hop);
  if (k < 0 || k >= static_cast<long>(T.energyDb.size()) || T.energyDb[k] == ProsodyProfile::kNone) return false;
  *db = T.energyDb[k];
  return true;
}

bool ProsodyGuide::pauseAfterMs(int inputWord, float* ms) const {
  if (!has(kRhythm) || inputWord < 0 || inputWord >= static_cast<int>(wordMap_.size()) || wordMap_[inputWord] < 0) return false;
  *ms = profile_.words[wordMap_[inputWord]].pauseAfter * 1000.0f;
  return true;
}

float ProsodyGuide::leadInMs() const {
  return has(kRhythm) && !profile_.words.empty() ? static_cast<float>(profile_.words[0].t0 * 1000.0) : -1.0f;
}

bool ProsodyGuide::nextLabel(const std::string& sym, float* ms, float* pitch, float* pitchEnd) {
  *ms = 0; *pitch = *pitchEnd = ProsodyProfile::kNone;
  const auto& L = profile_.labels;
  if (L.empty() || labelAt_ >= L.size()) return false;
  size_t k = labelAt_;
  while (k < L.size() && k < labelAt_ + 3 && L[k].sym != sym) ++k;  // resync over a few labels
  if (k >= L.size() || L[k].sym != sym) return false;
  labelAt_ = k + 1;
  if (has(kRhythm)) *ms = L[k].ms;
  if (has(kPitch)) { *pitch = L[k].pitch; *pitchEnd = L[k].pitchEnd; }
  return true;
}

}  // namespace puhuja
