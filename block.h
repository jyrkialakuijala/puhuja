// A multi-mass vocal fold pair as a 2 x 2 x 6 block of masses per fold
// (Jyrki's layout, 2026-10-31: each fold "a 2x2x8 structure" placed in
// AVX-512 registers, "if easier for dynamics, it can also be a 2x2x7 or
// 2x2x6, then the neighborhood around the two ends can be properly managed
// without juggling the values in different registers"): 2 in depth (the
// cover on the medial surface and the body behind it, as in Story & Titze's
// body-cover model), 2 layers from the inferior to the superior edge, and 8
// lanes along the cord of which the first and the last are the anchors (the
// anterior commissure and the vocal process of the arytenoid, fixed at zero)
// and the 6 between them move. The coordinate is the lateral displacement
// (x > 0 opens), so the slit width is the rest width plus the two covers'
// displacements, as in the two-mass model (folds.h): the lateral motion
// makes the opening.
//
// Lane layout: the two folds lie side by side in 16 lanes (lane = fold * 8 +
// segment), so each (depth, layer) plane of the pair is one row of 16
// floats, the width of an AVX-512 register; the block is 4 such rows for x
// and 4 for v. The anchors are zero lanes that never move, so the neighbour
// along the cord is the row shifted by one lane either way, with nothing to
// mask: at a fold's ends the shift brings in its own anchor, or the other
// fold's. The opposing mass is lane ^ 8. The step is loops over the 16 lanes
// with no data-dependent branches, for the compiler to vectorize.
//
// Per mass: the body masses hang on the wall by the body spring (stiffness
// ~ tension^2), the cover masses hang on the body mass of their layer by the
// cover spring; string-like coupling along the cord and shear coupling
// between the two layers; the air pressure on the covers' surfaces
// (Bernoulli along each segment's channel down to the narrowest layer, the
// tract pressure above it; the full subglottal pressure below a closure), a
// collision spring with extra damping when the opposing covers overlap. The
// flow is per segment channel with inertance, as in grid.cc; the heard flow
// goes through a rounded area, plus the sound of the moving surfaces.
// Units: CGS.
//
// Same interface as VocalFolds (folds.h) and GridFolds (grid.h).
#pragma once

#include "folds.h"

namespace puhuja {

class BlockFolds {
 public:
  static constexpr int kDepth = 2, kLayers = 2, kSegments = 8, kLanes = 2 * kSegments;
  static constexpr int kFirst = 1, kLast = kSegments - 2;  // the moving segments (the ends are anchors)
  static constexpr bool kOwnChink = true;  // the chink is the last moving segment's own rest gap
  void reset();
  float step(float dt, const VocalFolds::Input& in);
  float flow() const { return uSum_; }
  float area() const { return area_; }
  bool closedNow() const { return closedNow_; }
  float closeFraction() const { return closeFrac_; }
  // Debug: the lateral displacement of fold f, depth d (0 cover, 1 body),
  // layer j, segment i (cm), and which segments are open (bit i).
  float x(int f, int d, int j, int i) const { return x_[d][j][f * kSegments + i]; }
  unsigned openMask() const { return openMask_; }
  float segmentFlow(int i) const { return u_[i]; }

 private:
  // Displacement (cm, outward > 0) and velocity: [depth][layer][lane].
  alignas(64) float x_[kDepth][kLayers][kLanes] = {};
  alignas(64) float v_[kDepth][kLayers][kLanes] = {};
  float restW_[kSegments] = {};  // rest slit width per segment (cm)
  float u_[kSegments] = {}, uOut_[kSegments] = {};  // flow per segment channel, and through the rounded area
  float uSum_ = 0, uOutSum_ = 0;
  float surface_ = 0;  // volume velocity of the moving surfaces (cm^3/s)
  unsigned openMask_ = 0;
  float area_ = 0;  // the membranous part's open area (cm^2)
  bool wasOpen_ = false, closedNow_ = false;
  float closeFrac_ = 0;
};

}  // namespace puhuja
