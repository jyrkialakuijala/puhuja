// A multi-mass vocal fold pair: a 4 x 8 grid of masses per fold (Jyrki's
// layout, 2026-10-30), 4 layers from the inferior to the superior edge of
// the medial surface x 8 segments from the anterior commissure to the
// arytenoid, each mass moving up and down (the slit width follows from the
// vertical displacement through the wedge shape of the medial surface).
// 64 masses, 128 state variables.
//
// Per mass: a spring to the body (stiffness ~ tension^2, mass fixed, damping
// ratio constant: the physical tension scaling), string-like coupling to its
// neighbours along the cord (the ends are anchored, so the fundamental mode
// is a half sine along the length), shear coupling to the layers above and
// below (the lateral motion travels up the layers: the mucosal wave), the
// air pressure on its surface (Bernoulli along each segment's channel,
// separation at the narrowest layer; the full subglottal pressure on the
// layers below a closure), a collision spring with extra damping when the
// opposing masses touch, and a friction term between the layers of a pressed
// contact (the vertical sliding of the mucosal wave against the other cord,
// as a lateral-only grid sees it). The flow is one quantity through the sum
// of the segments' narrowest areas, with inertance, as in folds.cc; the
// heard flow goes through a rounded area. Units: CGS.
//
// Same interface as VocalFolds (folds.h), so the larynx driver can run
// either.
#pragma once

#include "folds.h"

namespace puhuja {

class GridFolds {
 public:
  static constexpr int kLayers = 4, kSegments = 8;
  static constexpr bool kOwnChink = true;  // the chink is the posterior segments' own rest gap
  void reset();
  float step(float dt, const VocalFolds::Input& in);
  float flow() const { return uSum_; }
  float area() const { return area_; }
  bool closedNow() const { return closedNow_; }
  float closeFraction() const { return closeFrac_; }
  // Debug: the vertical displacement of fold f, layer j, segment i (cm), and
  // which segments are open (bit i), for the zipper pattern of the closure.
  float z(int f, int j, int i) const { return z_[f][j][i]; }
  unsigned openMask() const { return openMask_; }
  float segmentFlow(int i) const { return u_[i]; }

 private:
  float gap(int j, int i) const;  // the slit width at layer j, segment i (cm)
  float z_[2][kLayers][kSegments] = {}, v_[2][kLayers][kSegments] = {};  // vertical displacement (cm, up > 0) and velocity
  float restW_[kSegments] = {};  // rest slit width per segment (cm)
  // Each segment is its own channel through the stack of layers, with its own
  // flow and inertance (parallel channels; Jyrki: the air through relates to
  // the size of each opening); flow() is their sum.
  float u_[kSegments] = {}, uOut_[kSegments] = {};
  float uSum_ = 0, uOutSum_ = 0;
  float surface_ = 0;  // volume velocity of the moving surfaces (cm^3/s)
  unsigned openMask_ = 0;
  float area_ = 0;  // the membranous part's open area (cm^2)
  bool wasOpen_ = false, closedNow_ = false;
  float closeFrac_ = 0;
};

}  // namespace puhuja
