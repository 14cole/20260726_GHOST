# Feature-Signature Workflow — How-To

Add non-axisymmetric surface features (panel gaps, seals, door seams, steps) to
a Body-of-Revolution RCS baseline to build a reduced-order engineering
signature. You characterise a joint as a 2-D "delta", then place it on a
vehicle by supplying its perimeter coordinates. Reuse is conditional on the
same feature cross-section, local clean material/coating stack, frequency,
units, phase origin, polarization convention, and sufficiently similar local
curvature.

This guide takes a new user from nothing to a combined signature. A complete,
runnable example lives in [`demo_feature_signature.py`](demo_feature_signature.py);
every snippet below matches it. Run it first to confirm your install works:

```bash
python3 demo_feature_signature.py
```

It builds a cylinder, one panel-gap delta, two "doors" at different places, and
writes `demo_feature_out/vehicle_signature_{VV,HH,VH}.grim` in ~2 minutes.

---

## 1. The idea in one paragraph

The BoR solver gives you the **clean, un-featured body baseline** within its
supported formulation and discretization. A surface feature adds a
*differential* field: the joint's scattering **minus** the smooth skin it
replaces (the smooth skin is already in the body — never double-count it). You
compute that differential once from a 2-D cross-section of the joint (the
**delta**), then spread it along the feature's known 3-D perimeter on the
vehicle skin (**line expansion**) and add it to the body. Both solvers use a
calibrated phase mapping and the same `exp(+2jk·d·r)` two-way placement factor,
so vehicle coordinates supply the modeled geometric translation phase.

**Model status:** this is an engineering delta embedding, not a fully coupled
3-D Maxwell solve. It does not generate body-feature mutual coupling, multiple
scattering, creeping waves, or diffraction absent from the isolated component
models. Omitted terms can interfere constructively or destructively, so the
result is not generally an upper or lower bound.

**What it is good for:** seams, gaps, seals, steps, small doors — anything that
is *locally two-dimensional* (looks the same all along its length) and whose
perimeter you know. **What it is not:** volumetric features, cavities behind an
opening, fin–body corners. Those need an external 3-D pattern or a separately
labelled reduced-order term; the line sum alone does not represent them.

**Demonstrated anchor, not a universal error bar:** for the canonical PEC
cosine-squared groove on the gated cylinder, the converged 2-D/BoR comparison
supports **±20° of local-edge broadside**, with limits of **3.5 dB magnitude**
and **25° residual phase**. Other cross-sections, materials, curvature, and
electrical sizes need their own convergence/reference study.

There are two separate angular assumptions. The 2-D table is sampled at
`phi = atan2(d·n, d·b)`, where `phi=90°` is surface-normal incidence in the
plane across the seam. The model has no independent longitudinal/conical
coefficient for `d·t`, so dominant contributions must also have `d·t≈0`.
A large smooth closed loop often has stationary-phase points satisfying the
second condition, but that does not guarantee `phi≈90°`; corners, endpoints,
short loops, and abrupt coefficient changes can contribute outside the anchor.

---

## 2. Prerequisites

- The 2-D solver (`rcs_solver.py`) and BoR solver (`bor_solver.py` /
  `bor_dispatch.py`) working (run `tests/validate_mie.py` and
  `tests/validate_bor_phase1.py` once).
- `line_expand.py`, `feature_sum.py`, `grim_io.py` in the same folder.
- Confirm the feature machinery itself with its gates:
  ```bash
  cd tests && python3 validate_line_expansion.py && python3 validate_feature_sum.py
  ```
  Both must print `ALL GATES PASSED`. The first also re-measures the two
  calibration constants (see §5).

---

## 3. Workflow at a glance

```
  2-D solver                              BoR solver
  ┌───────────────┐                       ┌──────────────┐
  │ clean coupon  │  ─┐                    │  clean body  │
  │ featured coupon│ ─┤ make_delta_grim    │  (once)      │
  └───────────────┘   └──►  seam.grim ─┐   └──────┬───────┘
                          (reusable)   │          │
   perimeter.txt per location ─────────┼──────────┤
   (vehicle-frame coordinates)         ▼          ▼
                                    sum_features / export_*_grim
                                             │
                        ┌────────────────────┴────────────────────┐
              export_radar_grim                        export_signature_grim
        vehicle_radar.grim (az×el×freq×pol)     vehicle_{VV,HH,VH}.grim (aspect×roll)
           ← usual final deliverable                    ← body-frame diagnostic
```

Steps 1–2 are done once per joint design and compatible local host stack.
Steps 3–5 are per vehicle.

---

## 4. Step 1 — Draw and solve the 2-D coupon

You solve a small 2-D cross-section of the joint **twice**: the featured joint,
and an otherwise-identical smooth reference. Five rules make the delta correct;
violating them can corrupt both complex magnitude and phase.

1. **Outer face on the x-axis (y = 0), feature centred on x = 0.** The delta's
   phase reference must be the seam line on the outer skin — the same point the
   perimeter coordinates trace. Referencing to mid-panel injects a spurious
   angle-dependent phase.
2. **No sharp edges on the coupon.** A rectangular plank's end edges ring the
   joint field back and forth; that does *not* cancel in the difference. Use
   rounded end caps (a "capsule"), or absorb the ends. The demo's
   `capsule_coupon()` shows the capsule.
3. **Wide enough coupon.** Widen it until the delta stops changing over the
   angles you use (the demo uses 8 wavelengths; the gate checks 8 vs 12).
4. **The clean coupon must match the body the BoR solves.** Bare PEC ground
   plane → bare PEC coupon. MAGRAM-coated body → coat the coupon's ground plane
   the same way. Only then does the subtraction cancel everything but the joint.
5. **Keep the coupon termination physically continuous with the host.** Do not
   replace the rounded/absorbed host ends with detached TYPE-1 “floating”
   absorber sheets. In the coupon bake-off that termination retained roughly a
   `180°` VV phase error and only a `0–5°` usable sector. It is outside the
   calibrated embedding model even though the central groove itself is
   attached.

The independent line-expansion anchor is presently a PEC groove. A
dielectric-, IBC-, or coating-based delta can be solved by the supported 2-D
formulations, but should not inherit the PEC embedding error numbers without
its own higher-dimensional reference or convergence evidence.

Solve at **both polarizations** (`TM` and `TE`) and every frequency you will
combine at, over a cut-angle sweep where **90° = normal incidence** on the outer
face. Export each to `.grim`:

```python
from rcs_solver import solve_monostatic_rcs_2d
from grim_io import export_result_to_grim

phi = list(np.arange(0.0, 180.1, 2.5))          # full lit support; 90 = normal
clean, feat = [], []
for pol in ("TM", "TE"):
    r = solve_monostatic_rcs_2d(clean_snapshot,    [FREQ_GHZ], phi, pol, geometry_units="meters")
    clean.append(export_result_to_grim(r, "coupon_clean_" + pol)[0])
    r = solve_monostatic_rcs_2d(featured_snapshot, [FREQ_GHZ], phi, pol, geometry_units="meters")
    feat.append(export_result_to_grim(r, "coupon_feat_" + pol)[0])
```

Coefficient lookup is strict complex interpolation with no extrapolation and
no "missing means zero" behavior. Characterize every incidence angle a
placement can request—normally the full `[0°, 180°]` lit interval shown above.

> Draw the coupon in whatever tool you like (a `.geo`, the GUI, or a snapshot
> dict as in the demo). These five coupon-specific rules are in addition to
> the geometry, material, mesh, and convergence requirements in
> `GEOMETRY_GUIDE.md`.

## 5. Step 2 — Build the reusable delta `.grim`

```python
from feature_sum import make_delta_grim
delta = make_delta_grim(clean, feat, "panelgap.grim")     # coherent subtraction
```

`clean`/`feat` may each be a single file or a list (one per polarization, as
above). The result carries the complex `featured − clean` amplitude for every
polarization and frequency. It is vehicle-independent data, but physically
reusable only where the feature and its clean local host stack, units, phase
origin, polarization convention, and curvature assumptions remain compatible.

The **calibration constants** `PSI_VV_DEG` / `PSI_HH_DEG` in `line_expand.py`
map the stored 2-D TE/TM coefficients into the BoR far-field convention. They
are applied to the two local coefficients *before* their projected
contributions are added; a finite perimeter generally mixes both into an HH or
VV channel. The physical 2-D asymptotic amplitude is `A_phys = jB`; that factor
does not change scattering width but does matter to phase when importing an
external solve. These constants are measured jointly by the canonical ring
gate and apply to deltas made with this repository's 2-D solver. This fixes the
inter-solver convention, not the omitted embedding physics. Re-measure the
calibration before using another tool's amplitudes.

If a delta is formed in the external GRIM viewer, its derived file no longer
contains the solver's raw complex-field metadata. Attest it against one of the
actual source coupon files before using it:

```python
from feature_sum import tag_as_delta
tag_as_delta("viewer_delta.grim", source_2d_grim=clean)
```

Calling `tag_as_delta` without a verified source is rejected: the utility will
not guess the time convention, far-field normalization, or phase origin.

## 6. Step 3 — Solve the clean body (BoR)

Solve the un-featured body once, over the full aspect range you will look across
(0–180°), keeping **both** polarizations:

```python
from bor_solver import solve_bor
gen  = cylinder_generatrix()                      # (rho, z) polyline, +z end -> -z end
body = solve_bor(gen, FREQ_GHZ * 1e9, list(np.arange(0, 180.1, 5.0)),
                 formulation="cfie", cfie_alpha=0.5, workers=4)
```

Keep the `gen` array — the combiner derives the normal of its discretized,
piecewise-linear generatrix at every perimeter point. Draw the body **smooth
through the feature region** — no cutout in the generatrix. For a feature on a
non-BoR placement surface, provide and independently verify the corresponding
`normal_fn`.

> A real vehicle body is normally a `.geo` run through `bor_dispatch`; any result
> carrying `theta_deg`, `amp_vv`, `amp_hh` works. Complex body fields are
> interpolated in real/imaginary parts and out-of-support queries raise. Refine
> the aspect grid until interpolation no longer moves important phase or nulls.
> The demo uses a bare cylinder.

## 7. Step 4 — Write a perimeter file per feature location

One text file per place a feature appears, segmented head-to-tail, in the
**vehicle frame, metres** (the same frame as the body generatrix, axis = +z):

```
# x1 y1 z1  x2 y2 z2   — one straight segment per line, chained head-to-tail
0.050000 0.000000 0.075000  0.050000 0.000000 0.125000
0.050000 0.000000 0.125000  0.047553 0.015451 0.125000
...
```

- Points must lie **on the body surface** (the demo generates them at radius
  `A_BODY`). A polygonised curved outline sags slightly inside the skin; assess
  the two-way phase error `2k d·Δr` at the highest frequency and every relevant
  look. Even a projected position error of `λ/50` is about `14.4°` of phase.
- A loop whose last point returns to its first is a **closed door outline**; an
  open chain is a **seam**. Both are read the same way.
- The same delta can be placed at many locations — pass several placements.

The demo's `write_door()` generates rectangular door loops; adapt it or export
the coordinates from your CAD.

## 8. Step 5 — Combine and export

Quick numerical look (body alone vs body + features):

```python
from feature_sum import sum_features, directions_from_aspect_roll
from line_expand import dbsm

placements = [{"delta": delta, "perimeter": "doorA.txt"},
              {"delta": delta, "perimeter": "doorB.txt"}]
dirs, asp, roll = directions_from_aspect_roll([80, 90, 100], rolls_deg=[0, 90])
full = sum_features(body, placements, dirs, FREQ_GHZ, generatrix=gen)
print(dbsm(full["sigma_vv"]))          # per look direction
```

There are two ways to write the result, depending on the frame you want.

**Body-frame** (aspect × roll × frequency, 3 files) — the vehicle's own scattering:

```python
from feature_sum import export_signature_grim
export_signature_grim("vehicle_signature",
    bor_result=body, placements=placements, generatrix=gen,
    frequencies_ghz=[FREQ_GHZ],
    aspects_deg=np.arange(60, 120.1, 5),
    rolls_deg=np.arange(0, 180.1, 15),
    mode="coherent")
# -> vehicle_signature_VV.grim, _HH.grim, _VH.grim
```

**Radar-frame** (azimuth × elevation × frequency × polarization, ONE file) — the
usual final deliverable, i.e. what a radar measures with the vehicle at a given
attitude:

```python
from feature_sum import export_radar_grim
export_radar_grim("vehicle_radar",
    bor_result=body, placements=placements, generatrix=gen,
    frequencies_ghz=[FREQ_GHZ],
    azimuths_deg=np.arange(0, 360.1, 5),
    elevations_deg=np.arange(-30, 30.1, 5),
    axis_az_deg=0.0, axis_el_deg=0.0, roll_deg=0.0)   # vehicle attitude
# -> vehicle_radar.grim  (pol axis = [VV, HH, VH])
```

`axis_az/el_deg` point the vehicle axis in the earth frame; `roll_deg` rolls the
vehicle about its axis (this is what fixes where the side-mounted features sit
relative to the radar). The exporter rotates the full scattering matrix from the
vehicle's meridian basis into the radar's earth-vertical V/H basis — so VV/HH are
the radar's own antennas and VH is the radar-frame cross-pol of the modeled
composite, with the horizontal-axis V/H label swap handled for you. This is
always the **coherent field represented by the reduced-order model**; it does
not add cross-polar mechanisms absent from the component models.

**Combine modes:** `sum_features` returns the coherent complex amplitude in
every mode. Its `sigma_*` is the selected in-memory estimate, while
`coherent_sigma_*` is always the power of that field. Exporters never overload
the primary file schema: `rcs_power` always matches `rcs_amp`, and a noncoherent
estimate is written separately as `combination_estimate_power`.

| mode | what it does | when |
|------|--------------|------|
| `"coherent"` *(default)* | everything summed in amplitude | the deterministic modeled field; requires verified common amplitude, phase-origin, time-sign, polarization, and embedding conventions |
| `"hybrid"` | features summed coherently with each other, then power-added to the body | an explicit estimate when body–feature phase is uncertain; feature–feature phase still requires compatible conventions |
| `"envelope"` | all individual powers added | an explicit incoherent estimate; it is neither a deterministic field nor generally a bound |

## 9. Reading the output

**Radar-frame file** (`export_radar_grim`, the usual deliverable):
- **One file**, `polarizations` axis = `[VV, HH, VH]`, arrays shaped
  `[azimuth, elevation, frequency, pol]`.
- `azimuths`/`elevations` are the **radar's** earth-frame look angles; VV/HH are
  the radar's earth V/H antennas; VH is the modeled radar-frame cross-pol.
- `rcs_power` = σ (m², plot as dBsm); `rcs_amp_real/imag` = the complex
  radar-frame scattering (phase preserved).

**Body-frame files** (`export_signature_grim`, a diagnostic):
- **Three files**, one per channel `VV`/`HH`/`VH`.
- `azimuths` = roll about the axis, `elevations` = aspect from the nose
  (0 = nose-on) — the vehicle's own frame, no earth rotation.
- `rcs_power` = `4π|rcs_amp|²`, always.
- `combination_estimate_power` = the requested hybrid/envelope/coherent
  engineering estimate (present on these diagnostic exports).

Common to both:
- **`rcs_power`** holds the field-consistent physical σ conversion of the
  stored coherent modeled field (m², plot as dBsm), so
  `4π|amp|² == rcs_power` in every exported file. This schema identity does
  not turn the reduced-order composite into a coupled 3-D solution.
- If requested, **`combination_estimate_power`** holds a separately labelled
  statistical/engineering estimate that has no corresponding deterministic
  complex amplitude.
- **Demo-specific sanity pattern:** in the supplied example the body dominates
  near broadside and the features matter more off broadside. This is not a
  general rule: a large/resonant feature or a null in the body field can make a
  feature dominate legitimately.

## 9b. Wings and fins

A wing/fin is the same line expansion with three differences from a surface
feature, and it plugs into `sum_features`/the exporters as just another
placement:

1. **Full-object coefficient, not a delta.** The coefficient is the *whole* 2-D
   airfoil cross-section's amplitude (a wing is a scatterer in its own right,
   not a perturbation of the skin). Build it with `coefficients_from_2d` (a
   single 2-D solve, no subtraction), or load a plain 2-D `.grim` export with
   `load_coefficients_from_grim`.
2. **The line is the open span** (root → tip), not a closed loop. Same
   `x1 y1 z1 x2 y2 z2` perimeter file.
3. **It carries its own normal** — the airfoil face normal, not the body
   surface normal. Put it in the placement dict:

```python
from line_expand import coefficients_from_2d
wing_coef = coefficients_from_2d(airfoil_snapshot, FREQ_GHZ, np.arange(0, 180.1, 2))
placements += [{"delta": wing_coef, "perimeter": "wing_span.txt",
                "normal": (0.0, 1.0, 0.0)}]        # airfoil face normal
```

Everything downstream (combine modes, body-frame and radar-frame export) is
unchanged. A wing's normalization is anchored to the standard electrically
large PEC flat-plate physical-optics result `4πA²/λ²`, and the implemented
uniform straight aperture reproduces its analytic `sinc²` null locations. That
anchor does not validate arbitrary airfoils, materials, conical incidence,
twist/taper, root/tip diffraction, current redistribution, or body coupling.
The wing term is therefore a reduced-order extrusion rather than an exact
finite-wing Maxwell solution. The wing–body corner double-bounce is also absent
unless added explicitly (§9c). Gated in `tests/validate_wing.py`.

## 9c. Wing–body corner (dihedral double-bounce)

The line-expansion sum is **single-bounce** — body and wing each scatter in
isolation and their fields add. The wing *root* is a **double-bounce**
(body → wing → radar) that is in neither isolated solve, and near a right-angle
root it is often the **dominant** return. Add it as an explicit corner term:

```python
corner = {"fold": root_chord_line,          # (2,3) or (n,2,3): the fold/root line
          "n_wing": (0.0, 1.0, 0.0),         # outward wing face normal
          "n_body": (1.0, 0.0, 0.0),         # outward body normal at the root
          "face_width": 3 * lam}             # effective double-bounce height (m)
corner_only = sum_features(None, [], dirs, FREQ_GHZ, generatrix=gen,
                           corners=[corner])
# Add corner_only["sigma_*"] to the coherent vehicle sigma only as an explicitly
# labelled expected-power estimate unless internal_phase_deg is calibrated.
```

It is a **PO-level estimate**, not a rigorous solve: magnitude follows the
standard dihedral `8πa²b²/λ²` (fold length `b`, `face_width` `a`) with a
`sinc²` aperture along the fold and a broad `cos²` retroreflective lobe across
it. Its Jones transformation is analytic within that PO model (co-pol with a
V/H sign flip when the fold aligns to the radar basis, pure cross-pol at 45°);
the placement phase is geometric. What it does *not* know: the internal
double-bounce constant phase (`internal_phase_deg`, default 0). Passing an
uncalibrated corner beside a wing in one `sum_features` call would therefore
invent their interference; solve/export it alone and power-add only in a
clearly labelled estimate (the production step 3b/4 workflow does this).
`face_width` is the main knob
(how far the double bounce reaches up the wing / along the body); a curved body
usually reduces the idealized PO return, but this term is not a guaranteed
bound on the coupled vehicle response.

### Non-right dihedrals (canted / dihedral / anhedral roots)

A root that is not square is assigned a documented engineering heuristic, not
independently validated full-wave corner physics. From the outward
normals, `ε = asin(n_wing·n_body)` is the interior angle's departure from 90°
(`δ = |ε|`, the deviation from square; `ε > 0` = corner opened out). Two
reflections rotate a ray by `2α`, so the double bounce leaves the corner
**deflected `2δ`** off the incidence reversal instead of retroreflecting:

* **lobe deflection** — the retro lobe centre is rotated by `2ε` about the fold
  axis (`n_wing × n_body`, which lies along the fold and so fixes the handedness
  regardless of how you ordered the fold endpoints). *Both* bounce senses exist
  (body→wing and wing→body deflect oppositely), so the lobe is a **symmetric
  pair at ±2δ** and the perpendicular-plane `cos²` is measured to the nearer
  centre. That leaves a `cos²(2δ)` dip *on* the bisector — the direction that
  used to be the peak.
* **peak rolloff** — the peak is multiplied by `cos²(2ε)`: exactly 1 at `δ = 0`,
  monotone, and 0 at `δ = 45°`, where the faces have gone coplanar (or shut into
  a cusp) and no double bounce survives. `−0.5 dB` at 10°, `−2.3 dB` at 20°,
  `−6 dB` at 30°.
* **illumination** — unchanged (`d·n_wing > 0` and `d·n_body > 0`), which is
  why the sign of the cant still matters a little: opening the corner *widens*
  the lit wedge to `±(45° + δ/2)`, closing it narrows it to `±(45° − δ/2)`, so a
  closed corner clips its own deflected lobe sooner.
* `δ = 0` preserves the right-dihedral **magnitude** model (gated); the
  complex field retains signed-sinc π phase reversals between aperture
  sidelobes. Corners
  more than ~20° off square still report a warning in `res["warnings"]` —
  now worded *"dihedral N deg off square — deflected/attenuated estimate"*.

**Honest limit:** the `2δ` deflection is a *bistatic ray-geometry* statement (the
exit beam misses the radar by `2δ` at *every* look angle in the perpendicular
plane), so the rigorous monostatic answer is two shifted **plate** lobes whose
aperture mismatch attenuates the return roughly like `sinc²(k a sin 2δ)` — tens
of dB within a few degrees of square for an electrically large face, far sharper
than `cos²(2δ)`. The smooth `cos²` rolloff is a monotone screening heuristic,
not a guaranteed upper or lower bound. Read it as *"the modeled corner stops
pointing energy back at you and points it 2δ away"*, not as an exact
canted-dihedral pattern. `tests/validate_corner.py` checks implementation of
those formulas and limiting cases; it is not an independent Maxwell reference.

## 9d. Compact 3-D differential patterns

A cavity/inlet/hole pattern placed through `points=` is a **3-D differential**
artifact, not a 2-D seam coefficient and not a free-space full-object result.
For a `.grim` input the loader requires:

- `rcs_domain='delta'` (featured minus the identical clean patch);
- the full reciprocal Jones set `VV`, `HH`, and `VH`;
- preserved complex far-field amplitude `F`;
- `units["rcs_linear_quantity"] == "sigma_3d"` and
  `rcs_power = 4π|F|²`;
- the exact phase-origin, time-sign, amplitude, and cavity-frame tags returned
  by `point_pattern_convention_metadata()`;
- strictly ordered physical axes and an explicitly complete, continuous
  360-degree azimuth seam.

Missing cross-pol is not assumed to be zero, and a full-object pattern is
refused because it would double-count the smooth body skin. The external solve
must put its phase origin at the placement coordinate and use the cavity-frame
basis documented by `point_scatterer_amplitude`. A lit look outside the
pattern's elevation support raises; it is never silently replaced by zero.

## 9e. A delta library, and manufacturing tolerance around one seam

A seam design is a *family*, not a single delta: the same panel gap is solved at
several gap widths, coating thicknesses, and so on. `delta_library.py` indexes
that family **by filename** — the parameters live in the name and nowhere else:

```
library/seal/0.020bmag_0.060gap.grim        bmag = 0.020 m, gap = 0.060 m
library/panel_gap/0.020bmag_0.060gap.grim   same point, different seam TYPE
```

Rules that make the filesystem a trustworthy parameter key, all enforced by the
scan: tokens sorted by key, one decimal width per key library-wide (else `0.02`
and `0.020` are two names for one point), every file carrying the same
variables, the **seam type as the directory** (types are never interpolated
across), and `2rev` reserved for a re-solve of the same point. Values are metres,
like everything else post-solve. Files whose names don't parse are *reported*,
never silently skipped — a silent skip is how you analyse 5 of 8 designs and
don't notice.

> **The one cost of filename-only metadata:** a hand-edited filename is
> undetectable — nothing in the file cross-checks it. Renames change meaning.

Pin the configuration, then spread a tolerance around the door:

```python
from delta_library import DeltaLibrary, Range, tolerance_placements

lib = DeltaLibrary.from_dir("library/seal")
lib.summary()                                    # axes, ragged-grid flag, unindexed
lib.validate([3.0, 6.0])                         # every entry is a delta and covers both

fam = lib.select(bmag=0.020)                     # config -> a 1-D gap family
res = fam.resolve(gap=Range(0.030, 0.080), n=6)  # min/max over 6 arcs
# res = fam.resolve(gap=[0.035, 0.045, 0.060])   # or explicit widths
print(res.report)          # every off-grid snap, named: requested -> used
pl  = tolerance_placements(perimeter, res.entries)
out = sum_features(body, pl, dirs, 6.0, generatrix=gen, mode="hybrid")
```

`select` takes an exact value, a list, a `Range`/2-tuple, or a predicate (a
*list* means "these values", a *tuple* means "this range"). `resolve` turns a
tolerance request into one entry per arc and is deliberately strict:

* an off-grid request **snaps to the nearest node and says so** in
  `res.report` — don't drop that report;
* `off_grid="error"` refuses instead;
* a request **outside** the axis always raises. The library is never
  extrapolated. The ~0.17–1 λ behavior printed by
  `tests/validate_line_expansion_size.py` belongs only to its canonical PEC
  groove family and is not a universal seam-size limit;
* interpolation between nodes is **not** offered — it needs a magnitude and
  unwrapped-phase check between the bracketing nodes, and when the grid is too
  coarse the honest fix is to solve the intermediate coupon (a 2-D coupon is
  minutes).

`tolerance_placements` splits the perimeter into equal-arc-length arcs at
segment boundaries (so the closed-form per-segment phase integral is untouched —
the same delta on every arc reproduces the single-placement result exactly) and
orders them with `smooth_cycle`: **up the odd ranks, back down the even ranks**.
Every neighbour is then two ranks apart, which attains the smallest possible
largest adjacent jump on a closed loop, `max(a[i+2] − a[i])`, so the widest gap
sits beside the 2nd/3rd widest and never beside the tightest. That is both what a
real hinge/latch stack-up looks like (one side tight, the other loose) and what
keeps the expansion inside its own validity — it assumes the cross-section is
locally invariant along the line, so an abrupt coefficient step is a
discontinuity it does not model. Abrupt changes create additional junction
scattering. Treat an arc-length rule or a mean/"worst" isolated coefficient as
a screening approximation and validate it for the actual family.

**Screen before you agonise.** In the supplied demo, 0.4–2 mm gaps at 6 GHz are
deeply subwavelength and the widest isolated delta is strongest. Those
dimensions, levels, and dominance are demo-specific; coherent placement can
still make a weaker isolated delta important. Rank the deltas in isolation
with `trade_study.door_trade_study(lib.select(bmag=0.020).paths(), ...)`, then
check the assembled field.

**Report an ensemble, not one arrangement.** Arcs sum *coherently* and each has
its own placement phase, so which arc holds which gap changes the interference.
The true per-unit arrangement is unknown, so one arrangement is spuriously
precise. Because the expansion is linear you can precompute the `arc × delta`
amplitude matrix once and then enumerate *every* arrangement for free:

```python
from itertools import permutations
from line_expand import combine, dbsm
from delta_library import arc_slices

arcs = arc_slices(perimeter, len(res.entries))
A = {(i, j): sum_features(None, [{"delta": e.path, "perimeter": arc}], dirs, 6.0,
                          generatrix=gen)["feature_amps"][0]
     for i, arc in enumerate(arcs) for j, e in enumerate(res.entries)}
peaks = np.array([dbsm(combine(out["body_amp"], [A[i, p[i]] for i in range(len(arcs))],
                               mode="hybrid")["sigma_vv"])
                  for p in permutations(range(len(arcs)))])
nominal, lo, hi = peaks[0], peaks.min(0), peaks.max(0)
```

Deliver the smooth ordering as the nominal modeled build and the enumerated
min/max as the **reduced-order assignment range**, not a physical or
manufacturing bound. If that modeled range is small, report it with the
remaining model limitations. Gated in `tests/validate_delta_library.py`
(L0–L6).

## 10. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `segment intersection` / `near-zero length` solving the coupon | groove deeper than half the plank; overlapping closing point — check rule 1/2 |
| Delta huge, or signature dominated by the feature everywhere | clean coupon doesn't match the body background (rule 4), or you added body+feature instead of the differential |
| Feature phase drifts / lobes in wrong place | perimeter/frame/phase origin mismatch, points off the surface, insufficient complex-angle sampling, or use outside the feature's independently supported embedding range |
| `outside characterized support` | the 2-D coefficient table does not cover a lit requested angle; solve the missing angles (normally use 0–180°), do not extrapolate |
| `no frequency … in delta` | solve the coupon at the frequencies you combine at (§4) |
| `a perimeter segment is normal to the skin` | perimeter and generatrix are in different frames |
| Cross-pol (VH) identically zero | expected for a purely circumferential/axial feature viewed in a principal plane; appears off those cuts |
| Signature looks noisy vs aspect | sample aspect finely — interference between features separated by `D` oscillates at `2kD`; combine on a fine grid, never interpolate σ |

## 11. Conventions cheat-sheet

- Time `e^{+jωt}`; look directions are **coming-from** unit vectors.
- 2-D cut: **90° = normal incidence** on the outer face; TM = E along the seam
  (→ HH / φ-pol), TE = E across it (→ VV / θ-pol).
- BoR aspect: **0° = nose-on**, 90° = broadside, 180° = tail-on.
- Everything in **metres**, vehicle frame, axis = +z, phase origin `(0,0,0)`.
- Canonical PEC-groove anchor only: **±20°** of local-edge broadside, gated at
  **3.5 dB magnitude / 25° residual phase**. Establish bounds separately for
  other feature/material/curvature families.
- **Two data types, and only two.** In every `.grim`, `rcs_power` is the physical
  power quantity associated with the modeled field, while
  `rcs_amp_real/imag` follows the solver's complex convention rather than
  `sqrt(power)`:
  - **2-D**: the stored coefficient is the layer-potential bare-integral
    amplitude `B`, with physical asymptotic `A_phys = jB`.
    `σ₂d = |B|²/(4k) = |A_phys|²/(4k)`, a scattering *width* in metres,
    plotted as **dBke** (`10log10(k·σ₂d)`). A **delta is 2-D** —
    `rcs_domain='delta'` is a separate axis meaning "this is a difference",
    not a unit.
  - **3-D**: `σ = 4π|F|²` in m², plotted as **dBsm**.
- **Three angle spaces, and only one of them is radar azimuth.** They all land in
  a `.grim`'s `azimuths` slot, so read the tag, not the axis name:
  - **2-D cut angle** (coupons, deltas): local to the cross-section, 90° = normal
    to the outer face. Not a vehicle angle at all.
  - **BoR aspect** (`body.grim`; the `elevations_deg` argument of
    `solve_monostatic_rcs_bor`): the polar angle from the rotation axis,
    0 = nose-on … 180 = tail-on. A cone half-angle, not an azimuth.
  - **Radar azimuth/elevation** (`export_radar_grim`, `bor_az_el_grid`): the real
    earth-frame look, for a vehicle at a given attitude. The end goal.

  The general map is `aspect = arccos(d̂·â)` for look `d̂` and body axis `â`.
  Aspect **equals** radar azimuth (measured from the nose) only when the axis is
  **horizontal** *and* you stay in the **elevation-0 cut** — the waterline. With a
  horizontal axis off that cut, `aspect = arccos(cos el · cos Δaz)`, so a whole
  *cone* of (az, el) looks shares one aspect — which is exactly what lets a 1-D
  aspect sweep fill the 2-D radar grid. Tilt the axis and even the elevation-0
  identity is gone. Polarization is a separate matter: even where the angles
  coincide, the body's meridian V/H and the radar's V/H differ by a rotation
  (handled by the `M` matrix in the radar export).
