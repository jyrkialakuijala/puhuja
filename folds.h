// Physical voice source: a two-mass model of the vocal folds (Ishizaka &
// Flanagan 1972) with the airflow through the glottis, coupled to the vocal
// tract through the pressure the tract builds up above the folds.
//
// Each fold is a lower mass m1 and an upper mass m2 on springs (k1, k2),
// coupled by kc, moving sideways (x > 0 opens). The glottis is two slits,
// a_i = rest_i + 2 L x_i; the flow goes through the narrower one. The air
// in the tract has inertia (inertance), so the flow U lags the pressure:
//
//   L dU/dt = (Ps - P1) - rho/2 (U / a_min)^2       (a_min > 0, else U = 0)
//
// and pushes the lower masses apart (Bernoulli pressure in the glottis);
// when the folds touch, a stiff collision spring and extra damping act.
// Units: CGS (g, cm, s; pressures in dyn/cm^2, flow in cm^3/s).
#ifndef PUHUJA_FOLDS_H_
#define PUHUJA_FOLDS_H_

namespace puhuja {

class VocalFolds {
 public:
  static constexpr bool kOwnChink = false;
  struct Input {
    float ps;       // pressure below the glottis minus above it (static part)
    float p1;       // acoustic pressure just above the folds (tract feedback)
    float q;        // tension: scales all oscillation frequencies (1 = rest)
    float rest;     // rest area of each slit (cm^2); <= 0 pressed together
    float gap = 0;  // posterior chink (cm^2), for models that place it (the two-mass has it in rest)
  };

  void reset();
  // Advance by dt seconds; returns the glottal flow U (cm^3/s).
  float step(float dt, const Input& in);
  float flow() const { return u_; }
  float area() const;  // current open area (cm^2), 0 when closed
  // True once per cycle, at the moment the folds close.
  bool closedNow() const { return closedNow_; }
  // When closedNow(): where in the last step the opening reached zero
  // (0..1 of dt, interpolated between the substeps), for sub-sample timing.
  float closeFraction() const { return closeFrac_; }

 private:
  float x1_ = 0, v1_ = 0, x2_ = 0, v2_ = 0;
  float u_ = 0;
  float uOut_ = 0;  // the flow heard: through the rounded area (see step)
  float rest_ = 0.05f;
  bool wasOpen_ = false, closedNow_ = false;
  float closeFrac_ = 0;
};

}  // namespace puhuja

#endif  // PUHUJA_FOLDS_H_
