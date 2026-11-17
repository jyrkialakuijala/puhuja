// Research parameters: a tunable constant is written
//   const float kName = param("kName", default);
// and an optimizer (tools/word_opt.py, tools/larynx_opt.py) overrides it
// through PUHUJA_PARAMS=file, a file of "name value" lines, or NAME=value
// in the environment. GRID_PARAMS is read the same way (the grid larynx's
// knobs and per-mass multipliers), so one file can carry both. With neither
// variable set, a file "puhuja.params" in the working directory or next to
// the binary is read: a running search keeps its best state there, so
// ./puhuja from the command line always speaks with the best parameters so
// far (Jyrki, 2026-10-30), and the file goes away when the result is
// written into the defaults. Read once.
#ifndef PUHUJA_PARAMS_H_
#define PUHUJA_PARAMS_H_

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>

#include "tuned_defaults.h"

inline std::string liveParamFile() {
  if (FILE* fp = std::fopen("puhuja.params", "r")) {
    std::fclose(fp);
    return "puhuja.params";
  }
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return "";
  buf[n] = 0;
  std::string p(buf);
  p = p.substr(0, p.rfind('/') + 1) + "puhuja.params";
  if (FILE* fp = std::fopen(p.c_str(), "r")) {
    std::fclose(fp);
    return p;
  }
  return "";
}

// The voice of this process for per-voice knob values ("NAME@f" / "NAME@m" in a params file or in tuned_defaults.h):
// female when the command line has -female (VOWDICT agent36, 2026-11-17). Knobs are read once, many at static
// initialisation before main(), so the command line is the one place the voice is known that early.
inline bool paramVoiceFemale() {
  static const bool female = [] {
    bool f = false;
    if (FILE* cf = std::fopen("/proc/self/cmdline", "rb")) {
      // The whole command line: a long text before -female must not hide it (a 4 KB buffer did).
      std::string buf;
      char chunk[4096];
      for (size_t n; (n = std::fread(chunk, 1, sizeof chunk, cf)) > 0;) buf.append(chunk, n);
      std::fclose(cf);
      const size_t got = buf.size();
      for (size_t i = 0; i < got;) {
        const char* a = buf.data() + i;
        size_t len = 0;
        while (i + len < got && a[len]) ++len;
        if (std::string(a, len) == "-female") f = true;
        i += len + 1;
      }
    }
    return f;
  }();
  return female;
}

inline const std::map<std::string, float>& paramFile() {
  static const std::map<std::string, float> m = [] {
    std::map<std::string, float> out;
    std::string files[2];
    int n = 0;
    for (const char* env : {"GRID_PARAMS", "PUHUJA_PARAMS"}) {
      if (const char* f = std::getenv(env)) files[n++] = f;
    }
    if (n == 0) files[n++] = liveParamFile();
    // (VOWDICT agent36, lab) "NAME@f value" / "NAME@m value": only for the female / male voice (the process's
    // command line has -female or not); these win over a plain "NAME value" line.
    std::map<std::string, float> sexed;
    const bool female = paramVoiceFemale();
    for (int i = 0; i < n; ++i) {
      if (files[i].empty()) continue;
      if (FILE* fp = std::fopen(files[i].c_str(), "r")) {
        char line[512], name[64];
        double v;
        while (std::fgets(line, sizeof line, fp)) {
          if (line[0] == '#') continue;  // "# cost 0.0391... stops-c32-mf": the optimizer's header
          if (std::sscanf(line, "%63s %lf", name, &v) == 2) {
            std::string s(name);
            const size_t at = s.size() > 2 ? s.size() - 2 : 0;
            if (at > 0 && s[at] == '@' && (s[at + 1] == 'f' || s[at + 1] == 'm')) {
              if ((s[at + 1] == 'f') == female) sexed[s.substr(0, at)] = static_cast<float>(v);
            } else {
              out[s] = static_cast<float>(v);
            }
          }
        }
        std::fclose(fp);
      }
    }
    for (const auto& kv : sexed) out[kv.first] = kv.second;
    return out;
  }();
  return m;
}

// The default in effect: tuned_defaults.h's "NAME@f" / "NAME@m" for this voice, else its "NAME", else the code's.
inline float tunedDefault(const char* name, float def) {
  const auto& td = tunedDefaults();
  auto t = td.find(std::string(name) + (paramVoiceFemale() ? "@f" : "@m"));
  if (t == td.end()) t = td.find(name);
  return t != td.end() ? t->second : def;
}

// PUHUJA_PARAM_LOG=file: append "NAME default" for each knob the first time it is read (tools/knob_search.py
// finds the knobs a text touches this way; the audio does not change).
inline void paramLog(const char* name, float def) {
  static FILE* f = [] { const char* p = std::getenv("PUHUJA_PARAM_LOG"); return p ? std::fopen(p, "a") : nullptr; }();
  if (!f) return;
  static std::set<std::string> seen;
  if (seen.insert(name).second) std::fprintf(f, "%s %.9g\n", name, def), std::fflush(f);
}

inline float param(const char* name, float def) {
  {
    paramLog(name, tunedDefault(name, def));  // the default in effect (tuned_defaults.h first)
  }
  const auto& pf = paramFile();
  const auto it = pf.find(name);
  if (it != pf.end()) return it->second;
  const char* v = std::getenv(name);
  if (v) return std::strtof(v, nullptr);
  return tunedDefault(name, def);
}

#endif  // PUHUJA_PARAMS_H_
