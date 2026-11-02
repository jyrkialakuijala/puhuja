// Fits the soft tract shape (tract.h) of every English vowel to its formant
// targets (english_shape.inc) on the transmission line (line.h):
//   linefit [out.txt] [--render dir]
// The formants are read off the line's own impulse response (a unit flow
// pulse at the glottis, 40 ms, the spectrum's lowest peaks), the shape is
// searched by Nelder-Mead over place, area and width of the constriction,
// the pharynx, mouth and lip areas. --render writes a sustained vowel per
// shape with the string folds for listening.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "line.h"
#include "string.h"
#include "tract.h"

using namespace puhuja;

namespace {
constexpr float kFs = 48000;
constexpr int kSections = 64;
#include "english_shape.inc"

// Formants of a shape: the lowest peaks of the line's impulse response.
std::vector<float> formants(const TractShape& sh, int want = 3) {
  TransmissionLine line;
  line.init(sh.length, kFs, kSections);
  line.setAreas(sh.sections(line.sections()));
  const int n = static_cast<int>(0.04f * kFs);
  std::vector<float> y(n);
  for (int i = 0; i < n; ++i) {
    float v = 0;
    for (int k = 0; k < TransmissionLine::kSubsteps; ++k) {
      line.substep(i == 0 && k == 0 ? 1.0f : 0.0f);
      v += line.lipFlow();  // the flow itself: no radiation tilt in the peaks
    }
    y[i] = v;
  }
  // Magnitude spectrum at 5 Hz steps, 100-5000 Hz (a Hann window).
  const float df = 5;
  const int nb = static_cast<int>(4900 / df) + 1;
  std::vector<float> mag(nb);
  std::vector<float> w(n);
  for (int i = 0; i < n; ++i) w[i] = 0.5f - 0.5f * std::cos(2 * 3.14159265f * i / (n - 1));
  for (int b = 0; b < nb; ++b) {
    const float f = 100 + b * df;
    const float wstep = 2 * 3.14159265f * f / kFs;
    float re = 0, im = 0, c = 1, s = 0;
    const float cs = std::cos(wstep), sn = std::sin(wstep);
    for (int i = 0; i < n; ++i) {
      const float v = y[i] * w[i];
      re += v * c;
      im += v * s;
      const float c2 = c * cs - s * sn;
      s = s * cs + c * sn;
      c = c2;
    }
    mag[b] = std::log(re * re + im * im + 1e-30f);
  }
  std::vector<float> out;
  for (int b = 2; b + 2 < nb && static_cast<int>(out.size()) < want; ++b) {
    if (mag[b] > mag[b - 1] && mag[b] >= mag[b + 1] && mag[b] > mag[b - 2] && mag[b] > mag[b + 2]) {
      // Parabolic interpolation of the peak.
      const float d = 0.5f * (mag[b - 1] - mag[b + 1]) / (mag[b - 1] - 2 * mag[b] + mag[b + 1] + 1e-9f);
      out.push_back(100 + (b + d) * df);
    }
  }
  while (static_cast<int>(out.size()) < want) out.push_back(0);
  return out;
}

// The search variables (log-scaled) <-> the shape.
constexpr int kDim = 9;
TractShape shapeOf(const double* x, const TractShape& base) {
  TractShape s = base;
  s.xc = std::clamp(static_cast<float>(x[0]), 0.15f, 0.95f);
  s.ac = std::exp(static_cast<float>(x[1]));
  s.wc = std::clamp(std::exp(static_cast<float>(x[2])), 0.8f, 6.0f);
  s.aLip = std::exp(static_cast<float>(x[3]));
  s.aPharynx = std::exp(static_cast<float>(x[4]));
  s.aOral = std::exp(static_cast<float>(x[5]));
  s.lLip = std::clamp(static_cast<float>(x[6]), 0.3f, 3.0f);
  s.aLarynx = std::clamp(std::exp(static_cast<float>(x[7])), 0.3f, 3.0f);
  s.length = std::clamp(static_cast<float>(x[8]), 15.5f, 19.0f);  // larynx height and lip protrusion
  return s;
}
double cost(const double* x, const TractShape& base, const float* target) {
  const TractShape s = shapeOf(x, base);
  const std::vector<float> f = formants(s);
  static const float kW[3] = {1.0f, 1.0f, 0.5f};
  double c = 0;
  for (int k = 0; k < 3; ++k) {
    if (f[k] <= 0) return 10.0;
    const double d = std::log(f[k] / target[k]);
    c += kW[k] * d * d;
  }
  // Keep the shape plausible: areas within 0.2-10 cm^2, mild preference for the defaults.
  for (int k = 1; k < 6; ++k) {
    const double v = x[k];
    if (k != 2) c += 0.001 * (v - std::log(3.0)) * (v - std::log(3.0));
    if (std::exp(v) < 0.15) c += 0.5 * (std::log(0.15) - v);
    if (std::exp(v) > 14) c += 0.5 * (v - std::log(14.0));
  }
  if (x[0] < 0.15 || x[0] > 0.95) c += 1.0;
  if (x[6] < 0.3 || x[6] > 3.0) c += 0.5;
  if (x[8] < 15.5 || x[8] > 19.0) c += 0.5;
  c += 0.01 * (x[8] - 17.5) * (x[8] - 17.5);  // the length stays near the voice's
  return c;
}

// A small Nelder-Mead.
std::vector<double> nelderMead(std::vector<double> x0, const std::vector<double>& step, int evals,
                               const TractShape& base, const float* target, double* bestCost) {
  const int n = static_cast<int>(x0.size());
  std::vector<std::vector<double>> pts(n + 1, x0);
  std::vector<double> fv(n + 1);
  for (int i = 1; i <= n; ++i) pts[i][i - 1] += step[i - 1];
  for (int i = 0; i <= n; ++i) fv[i] = cost(pts[i].data(), base, target);
  int used = n + 1;
  while (used < evals) {
    std::vector<int> idx(n + 1);
    for (int i = 0; i <= n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return fv[a] < fv[b]; });
    std::vector<std::vector<double>> P(n + 1);
    std::vector<double> F(n + 1);
    for (int i = 0; i <= n; ++i) { P[i] = pts[idx[i]]; F[i] = fv[idx[i]]; }
    pts = P; fv = F;
    std::vector<double> c(n, 0);
    for (int i = 0; i < n; ++i) for (int k = 0; k < n; ++k) c[k] += pts[i][k] / n;
    auto along = [&](double t) { std::vector<double> y(n); for (int k = 0; k < n; ++k) y[k] = c[k] + t * (pts[n][k] - c[k]); return y; };
    std::vector<double> xr = along(-1.0); double fr = cost(xr.data(), base, target); ++used;
    if (fr < fv[0]) {
      std::vector<double> xe = along(-2.0); double fe = cost(xe.data(), base, target); ++used;
      if (fe < fr) { pts[n] = xe; fv[n] = fe; } else { pts[n] = xr; fv[n] = fr; }
    } else if (fr < fv[n - 1]) {
      pts[n] = xr; fv[n] = fr;
    } else {
      std::vector<double> xc = along(fr < fv[n] ? -0.5 : 0.5); double fc = cost(xc.data(), base, target); ++used;
      if (fc < std::min(fr, fv[n])) { pts[n] = xc; fv[n] = fc; }
      else {  // shrink
        for (int i = 1; i <= n; ++i) {
          for (int k = 0; k < n; ++k) pts[i][k] = pts[0][k] + 0.5 * (pts[i][k] - pts[0][k]);
          fv[i] = cost(pts[i].data(), base, target); ++used;
        }
      }
    }
  }
  int b = 0;
  for (int i = 1; i <= n; ++i) if (fv[i] < fv[b]) b = i;
  *bestCost = fv[b];
  return pts[b];
}

void put32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
void put16(FILE* f, uint16_t v) { fwrite(&v, 2, 1, f); }

void render(const TractShape& sh, const char* path, float seconds, float f0) {
  TransmissionLine line;
  line.init(sh.length, kFs, kSections);
  line.setAreas(sh.sections(line.sections()));
  StringFolds folds;
  folds.reset();
  const float dt = 1.0f / (TransmissionLine::kSubsteps * kFs);
  const int total = static_cast<int>(seconds * kFs);
  std::vector<float> out(total);
  float p1 = 0, lp = 0;
  const float aa = 1.0f - std::exp(-2 * 3.14159f * 16000.0f * dt);
  for (int i = 0; i < total; ++i) {
    const float env = std::min(1.0f, i / (0.03f * kFs)) * std::min(1.0f, (total - i) / (0.05f * kFs));
    for (int k = 0; k < TransmissionLine::kSubsteps; ++k) {
      VocalFolds::Input in;
      in.ps = 7840.0f * env;
      in.p1 = p1;
      in.q = f0 / 118.5f;
      in.rest = 0.02f;
      p1 = line.substep(folds.step(dt, in));
      lp += aa * (line.radiated() - lp);
    }
    out[i] = lp;
  }
  float peak = 1e-9f;
  for (float v : out) peak = std::max(peak, std::fabs(v));
  FILE* f = std::fopen(path, "wb");
  const uint32_t bytes = total * 2;
  fwrite("RIFF", 1, 4, f); put32(f, 36 + bytes); fwrite("WAVEfmt ", 1, 8, f); put32(f, 16); put16(f, 1); put16(f, 1);
  put32(f, kFs); put32(f, kFs * 2); put16(f, 2); put16(f, 16); fwrite("data", 1, 4, f); put32(f, bytes);
  for (float v : out) put16(f, static_cast<int16_t>(std::lround(v / peak * 0.5f * 32767)));
  std::fclose(f);
}
}  // namespace

// --measure: read kLineVowels' shape knobs from puhuja.cc and print, per
// vowel, the formants the line gives for the current shape (for refreshing
// the f1-f3 labels in the table after a shape search).
int measure() {
  FILE* f = std::fopen("puhuja.cc", "r");
  if (!f) { std::fprintf(stderr, "run in the puhuja directory\n"); return 1; }
  std::string src;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) src.append(buf, n);
  std::fclose(f);
  const char* names[9] = {"XC", "AC", "WC", "ALIP", "APH", "AOR", "LLIP", "ALAR", "LEN"};
  for (int ph = 0; ph < 40; ++ph) {
    TractShape sh;
    float* fields[9] = {&sh.xc, &sh.ac, &sh.wc, &sh.aLip, &sh.aPharynx, &sh.aOral, &sh.lLip, &sh.aLarynx, &sh.length};
    bool found = true;
    for (int k = 0; k < 9 && found; ++k) {
      char key[64];
      std::snprintf(key, sizeof key, "param(\"LINE_V%d_%s\", ", ph, names[k]);
      const size_t at = src.find(key);
      if (at == std::string::npos) { found = false; break; }
      *fields[k] = static_cast<float>(std::atof(src.c_str() + at + std::strlen(key)));
    }
    if (!found) continue;
    const std::vector<float> fm = formants(sh);
    std::printf("%d %.1f %.1f %.1f\n", ph, fm[0], fm[1], fm[2]);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "--measure")) return measure();
  const char* outPath = argc > 1 && argv[1][0] != '-' ? argv[1] : "ref/line/vowel_tract.txt";
  const char* renderDir = nullptr;
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], "--render")) renderDir = argv[i + 1];
  FILE* out = std::fopen(outPath, "w");
  std::fprintf(out, "# symbol xc ac wc aLip aPharynx aOral lLip aLarynx length  F1 F2 F3 (line)  target F1 F2 F3\n");
  std::printf("%-4s %7s %7s %7s | %6s %6s %6s %6s %6s %6s %5s %5s %5s | %s\n", "ph", "F1", "F2", "F3", "xc", "ac", "wc", "aLip", "aPh", "aOr", "lLip", "aLar", "L", "target");
  const TractShape base;
  const int nPh = sizeof(kEnglishShape) / sizeof(kEnglishShape[0]);
  for (int i = 0; i < nPh && i < 17; ++i) {  // the vowels come first in Ph order
    const auto& v = kEnglishShape[i];
    if (v.f1 <= 0) continue;
    const float target[3] = {v.f1, v.f2, v.f3};
    // A few starts: front, central, back constrictions.
    double bestC = 1e9;
    std::vector<double> best;
    for (double xc0 : {0.3, 0.5, 0.7, 0.85}) {
      std::vector<double> x0 = {xc0, std::log(1.5), std::log(3.0), std::log(2.0), std::log(4.0), std::log(4.0), 1.0, 0.0, 17.5};
      std::vector<double> st = {0.1, 0.5, 0.3, 0.5, 0.4, 0.4, 0.5, 0.3, 0.6};
      double c;
      std::vector<double> x = nelderMead(x0, st, 500, base, target, &c);
      if (c < bestC) { bestC = c; best = x; }
    }
    const TractShape s = shapeOf(best.data(), base);
    const std::vector<float> f = formants(s);
    std::printf("%-4s %7.0f %7.0f %7.0f | %6.3f %6.2f %6.2f %6.2f %6.2f %6.2f %5.2f %5.2f %5.1f | %.0f %.0f %.0f  cost %.4f\n", v.symbol, f[0], f[1], f[2], s.xc, s.ac, s.wc, s.aLip, s.aPharynx, s.aOral, s.lLip, s.aLarynx, s.length, v.f1, v.f2, v.f3, bestC);
    std::fprintf(out, "%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f  %.0f %.0f %.0f  %.0f %.0f %.0f\n", v.symbol, s.xc, s.ac, s.wc, s.aLip, s.aPharynx, s.aOral, s.lLip, s.aLarynx, s.length, f[0], f[1], f[2], v.f1, v.f2, v.f3);
    if (renderDir) {
      std::string path = std::string(renderDir) + "/" + v.symbol + ".wav";
      render(s, path.c_str(), 0.7f, 110.0f);
    }
  }
  std::fclose(out);
  return 0;
}
