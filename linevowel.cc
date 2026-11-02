// Sustained vowels from the transmission line driven by the string folds:
//   linevowel uniform|a|i|u|schwa seconds out.wav [f0 Hz] [ps dyn/cm^2]
// A check of the line against tube theory (uniform 17.5 cm: 500, 1500, 2500 Hz).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "line.h"
#include "string.h"

using namespace puhuja;

namespace {
constexpr float kFs = 48000;
constexpr float kLength = 17.5f;
// Area functions (cm^2) by position (cm from the glottis): piecewise constant.
struct Seg { float to, area; };
std::vector<float> areas(const char* v, int n, float dx) {
  std::vector<Seg> segs;
  if (!strcmp(v, "uniform")) segs = {{kLength, 3.0f}};
  else if (!strcmp(v, "schwa")) segs = {{2, 1.0f}, {kLength, 4.0f}};
  else if (!strcmp(v, "a")) segs = {{2, 1.0f}, {9, 1.0f}, {kLength, 8.0f}};
  else if (!strcmp(v, "i")) segs = {{2, 1.0f}, {10, 8.0f}, {15, 0.8f}, {kLength, 3.0f}};
  else if (!strcmp(v, "u")) segs = {{2, 1.0f}, {7, 5.0f}, {9.5f, 0.8f}, {16, 5.0f}, {kLength, 0.5f}};
  else { std::fprintf(stderr, "unknown vowel %s\n", v); std::exit(1); }
  std::vector<float> a(n);
  for (int k = 0; k < n; ++k) {
    const float x = (k + 0.5f) * dx;
    float area = segs.back().area;
    for (const Seg& s : segs) if (x < s.to) { area = s.area; break; }
    a[k] = area;
  }
  return a;
}
void put32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
void put16(FILE* f, uint16_t v) { fwrite(&v, 2, 1, f); }
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "linevowel uniform|a|i|u|schwa seconds out.wav [f0] [ps]\n"); return 1; }
  const float seconds = std::atof(argv[2]);
  const float f0 = argc > 4 ? std::atof(argv[4]) : 110.0f;
  const float ps = argc > 5 ? std::atof(argv[5]) : 7840.0f;
  TransmissionLine line;
  const int sections = std::getenv("LINE_SECTIONS") ? std::atoi(std::getenv("LINE_SECTIONS")) : 0;
  line.init(kLength, kFs, sections);
  const std::vector<float> a = areas(argv[1], line.sections(), line.dx());
  line.setAreas(a);
  StringFolds folds;
  folds.reset();
  const float dt = 1.0f / (TransmissionLine::kSubsteps * kFs);
  // Tension: the string's natural f0 is STRING_F0 (118.5) at q = 1.
  const float q = f0 / 118.5f;
  const int total = static_cast<int>(seconds * kFs);
  std::vector<float> out(total);
  float p1 = 0, lp = 0;
  const float aa = 1.0f - std::exp(-2 * 3.14159f * 16000.0f * dt);  // a mild anti-alias before decimation
  for (int i = 0; i < total; ++i) {
    // Lung pressure rises over 30 ms.
    const float env = std::min(1.0f, i / (0.03f * kFs));
    float y = 0;
    for (int k = 0; k < TransmissionLine::kSubsteps; ++k) {
      VocalFolds::Input in;
      in.ps = ps * env;
      in.p1 = p1;
      in.q = q;
      in.rest = 0.02f;  // the voicing rest area (cm^2), as the larynx uses
      const float u = folds.step(dt, in);
      p1 = line.substep(u);
      lp += aa * (line.radiated() - lp);
      y = lp;
    }
    out[i] = y;
  }
  float peak = 1e-9f;
  for (float v : out) peak = std::max(peak, std::fabs(v));
  FILE* f = std::fopen(argv[3], "wb");
  const uint32_t bytes = total * 2;
  fwrite("RIFF", 1, 4, f); put32(f, 36 + bytes); fwrite("WAVEfmt ", 1, 8, f); put32(f, 16); put16(f, 1); put16(f, 1);
  put32(f, kFs); put32(f, kFs * 2); put16(f, 2); put16(f, 16); fwrite("data", 1, 4, f); put32(f, bytes);
  for (float v : out) put16(f, static_cast<int16_t>(std::lround(v / peak * 0.5f * 32767)));
  std::fclose(f);
  std::fprintf(stderr, "%s: %d sections of %.3f cm, peak %.3g, written %s\n", argv[1], line.sections(), line.dx(), peak, argv[3]);
  return 0;
}
