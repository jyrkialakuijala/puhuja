// A single vocal fold against a mirror wall as two strings of 16 lanes
// (Jyrki, 2026-10-31: "a single fold model with two registers of 16 values,
// on top of each other, both operating as strings, but connected mildly --
// there is an assumption of perfect symmetry, so we can think that this model
// is working against a perfectly smooth wall").
//
// The two folds of a symmetric larynx move as mirror images, so one fold is
// enough: the glottal slit is the rest width plus twice this fold's lateral
// displacement, and the midline is a smooth wall the fold collides with. The
// fold is two layers from the inferior to the superior edge (the registers,
// one above the other), each a string of 16 lanes along the cord: lanes 0
// and 15 are the anchors (anterior commissure, vocal process), the 14 between
// them move. Each layer's row of 16 floats is one AVX-512 register, for x and
// for v. Each lane hangs on the wall by a weak spring and on its neighbours
// along the cord by the string tension, which carries most of the restoring
// force (the lowest string mode sets the natural frequency); the two layers
// are coupled mildly by shear. The air pressure acts on each layer's surface
// as in the block (Bernoulli down to the narrowest layer of each segment's
// channel, the tract pressure above it, the subglottal pressure under a
// closure); collision with the wall is a spring plus extra damping. The flow
// is per segment channel with inertance; the heard flow goes through a
// rounded area, plus the sound of the moving surface (doubled: the mirror
// fold moves too). The continuity regularizers of the block (neighbour mean
// of position, velocity and force; a closed lane shares its layer's mean
// velocity) are here as well. Lane loops run over all 16 lanes with no
// masking; the anchors' force, velocity and position are overwritten with
// zero after each stage. Units: CGS.
//
// Same interface as VocalFolds (folds.h), GridFolds and BlockFolds.
#pragma once

#include "folds.h"

namespace puhuja {

class StringFolds {
 public:
  static constexpr int kLayers = 2, kLanes = 16;
  static constexpr int kFirst = 1, kLast = kLanes - 2;  // the moving lanes (the ends are anchors)
  static constexpr bool kOwnChink = true;  // the chink is the last moving lane's own rest gap
  void reset();
  float step(float dt, const VocalFolds::Input& in);
  float flow() const { return uSum_; }
  float area() const { return area_; }
  bool closedNow() const { return closedNow_; }
  float closeFraction() const { return closeFrac_; }
  // Debug: the lateral displacement of layer j (0 inferior, 1 superior), lane l (cm).
  float x(int j, int l) const { return x_[j][l]; }
  unsigned openMask() const { return openMask_; }

 private:
  // Displacement (cm, outward > 0) and velocity: [layer][lane].
  alignas(64) float x_[kLayers][kLanes] = {};
  alignas(64) float v_[kLayers][kLanes] = {};
  alignas(64) float restW_[kLanes] = {};  // rest slit width per lane (cm; both folds together)
  alignas(64) float u_[kLanes] = {}, uOut_[kLanes] = {};  // flow per lane channel, and through the rounded area
  float uSum_ = 0, uOutSum_ = 0;
  alignas(64) float aHeard_[kLanes] = {};  // the heard channel area per lane, its closing slowed by kCloseMs
  float surface_ = 0;  // volume velocity of the moving surfaces (cm^3/s, both folds)
  unsigned openMask_ = 0;
  float area_ = 0;  // the membranous part's open area (cm^2)
  bool wasOpen_ = false, closedNow_ = false;
  float closeFrac_ = 0;
};

}  // namespace puhuja
