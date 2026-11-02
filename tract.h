// A soft articulatory area function for the transmission line (line.h).
//
// The tract is described continuously, in cm from the glottis: a larynx tube
// of area aLarynx and length lLarynx, a pharynx of area aPharynx blending
// into an oral cavity of area aOral around the velum, a tongue constriction
// at xc (as a share of the length) of area ac and half-width wc (a cosine^2
// hump in the log area, so the shape is smooth and the constriction's place,
// depth and length move continuously, Jyrki, 2026-11-01: "all things done
// softly with subpixel resolution"), and lips of area aLip over the last
// lLip cm. Each section's area is the average of the continuous function
// over the section's length (8 points), so the line sees no staircase as an
// articulator moves.
#pragma once

#include <vector>

namespace puhuja {

struct TractShape {
  float length = 17.5f;   // glottis to lips (cm)
  float lLarynx = 2.0f, aLarynx = 1.0f;
  float aPharynx = 4.0f, aOral = 4.0f;
  float xVelum = 0.55f;   // where the pharynx becomes the mouth (share of the length)
  float xc = 0.6f;        // the tongue constriction's place (share of the length)
  float ac = 4.0f;        // ... its area (cm^2)
  float wc = 3.0f;        // ... its half-width (cm)
  float lLip = 1.0f, aLip = 3.0f;
  // The continuous area (cm^2) at x cm from the glottis.
  float area(float x) const;
  // The sections' areas for a line of n sections over the length.
  std::vector<float> sections(int n) const;
};

}  // namespace puhuja
