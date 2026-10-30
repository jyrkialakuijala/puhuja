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
#include <string>

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

inline const std::map<std::string, float>& paramFile() {
  static const std::map<std::string, float> m = [] {
    std::map<std::string, float> out;
    std::string files[2];
    int n = 0;
    for (const char* env : {"GRID_PARAMS", "PUHUJA_PARAMS"}) {
      if (const char* f = std::getenv(env)) files[n++] = f;
    }
    if (n == 0) files[n++] = liveParamFile();
    for (int i = 0; i < n; ++i) {
      if (files[i].empty()) continue;
      if (FILE* fp = std::fopen(files[i].c_str(), "r")) {
        char line[512], name[64];
        double v;
        while (std::fgets(line, sizeof line, fp)) {
          if (line[0] == '#') continue;  // "# cost 0.0391... stops-c32-mf": the optimizer's header
          if (std::sscanf(line, "%63s %lf", name, &v) == 2) out[name] = static_cast<float>(v);
        }
        std::fclose(fp);
      }
    }
    return out;
  }();
  return m;
}

inline float param(const char* name, float def) {
  const auto& pf = paramFile();
  const auto it = pf.find(name);
  if (it != pf.end()) return it->second;
  const char* v = std::getenv(name);
  return v ? std::strtof(v, nullptr) : def;
}

#endif  // PUHUJA_PARAMS_H_
