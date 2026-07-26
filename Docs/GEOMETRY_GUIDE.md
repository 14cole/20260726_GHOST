# Geometry File Guide (.geo) — 2D and BoR Solvers

Everything a geometry author needs to know: file format, TYPE semantics,
sign conventions, drawing directions, and the gotchas that produce silently
wrong answers in other tools but hard errors (or warnings) here.

---

## 1. File format

```
Title: my geometry
Segment: <name> <TYPE>
properties: <TYPE> <N> <IBC> <POS_MAT> <NEG_MAT>
x1 y1 x2 y2            # one straight primitive per line, 4 numbers
x1 y1 x2 y2            # primitives must chain head-to-tail within a segment
...
Segment: <next> <TYPE>
...
IBCS_Resistances:
<flag> <kind> <R_start> <X_start> <R_end> <X_end>    # inline impedance
<flag>                                               # flag > 50: mat.<flag> table
Dielectrics:
<flag> <eps'> <eps''> <mu'> <mu''>                   # inline material
<flag>                                               # flag > 50: mat.<flag> table
```

- `properties:` must have exactly **5 fields**. Blank values are written as
  concrete tokens on save (`0`), never as empty strings.
- Coordinates are saved with `repr()` full precision — a load/save cycle
  round-trips exactly. Do not hand-truncate coordinates of closed contours;
  endpoints that should meet must match to ~1e-9 of the model size.
- `mat.<flag>` table files are resolved **relative to the geometry file's
  own folder** — keep them next to their `.geo` when copying/transferring.
- Every non-comment `mat.<flag>` row is strict: impedance tables require
  exactly `f_GHz Re(Zs) Im(Zs)` and dielectric tables require exactly
  `f_GHz Re(eps) Im(eps) Re(mu) Im(mu)`. Values must be finite, frequencies
  positive and unique. Malformed rows and duplicate frequencies are errors.
- Table interpolation is linear in the complex components and is allowed only
  inside the characterized frequency interval (including its endpoints).
  Out-of-band requests raise; they are never endpoint-clamped or extrapolated.
- Units are chosen at solve time (`geometry_units="inches"` or `"meters"`),
  not in the file. The runners default to **inches**.

## 2. Segment TYPEs

| TYPE | Boundary            | pos_mat        | neg_mat | 2D | BoR |
|------|---------------------|----------------|---------|----|-----|
| 1    | resistive sheet     | —              | —       | ✓  | ✗ (rejected) |
| 2    | air \| PEC (opt. IBC via flag) | —   | —       | ✓  | ✓  |
| 3    | air \| dielectric   | the dielectric | —       | ✓  | ✓  |
| 4    | dielectric \| PEC   | the coating    | —       | ✓  | ✓  |
| 5    | dielectric \| dielectric | one side  | other   | ✓  | ✓ (layer\|layer) |

The check marks above are formulation ingredients, not permission to combine
them arbitrarily:

- In 2-D, TYPE 1 is supported for all-sheet scenes and for sheets plus a
  pure-PEC TYPE 2 body. TYPE 1 mixed with an IBC-coated body, dielectric body,
  or layered coating is rejected.
- A TYPE 5 dielectric/dielectric interface cannot be the only exterior
  boundary. The unbounded background is air, so the geometry must also contain
  the enclosing air-facing TYPE 2/3 boundary required by the selected layout.
- BoR TYPE 5 is accepted only in the explicitly listed layer, patch, and band
  layouts below; BoR TYPE 1 is rejected.

`N` (2nd property): panel/element density. `N > 0` = explicit count per
primitive; `N < 0` = |N| panels per wavelength; `0`/blank = auto density
(λ/20). For both 2-D automatic/negative-`N` meshing and BoR penetrable
meshing, wavelength means the shortest wavelength among the referenced
materials (air included), not unconditionally the free-space wavelength.
When `mesh_reference_ghz` is supplied for a sweep, it is an additional
mesh-control frequency, not permission to coarsen the mesh: both solvers use
the shortest referenced-material wavelength over the reference and every
requested solve frequency.

`IBC` (3rd property): 0 = none, otherwise an `IBCS_Resistances` flag, but
support is formulation-specific and fail-closed:

- 2-D supports it on TYPE 1 sheets, TYPE 2 conductor boundaries, and TYPE 4
  dielectric-to-conductor boundaries; TYPE 3/5 transmission-interface IBCs
  are rejected because their jump condition is not implemented.
- BoR supports it on TYPE 2 conductor boundaries only. TYPE 3/4/5 IBC flags
  are rejected rather than silently ignored or routed through the wrong
  formulation.

## 3. Sign conventions (get these wrong and the physics is wrong)

**Time convention is e^{+jωt}** throughout (outgoing waves ~ H₀⁽²⁾ / h_n⁽²⁾).
Consequences:

- **Lossy dielectrics have NEGATIVE imaginary parts**: Im(ε_r) ≤ 0 and
  Im(μ_r) ≤ 0. In a Dielectrics row, ε = 3 − 0.5j, μ = 1.5 − 0.8j is
  written:

  ```
  1 3 -0.5 1.5 -0.8
  ```

  A **positive** ε″/μ″ is an active (gain) medium and is rejected; active media
  do not have a supported opt-in. `mat.*` tables use the same convention.
  Singular ENZ/MNZ inputs (`|ε_r|` or `|μ_r| <= 1e-12`) are also rejected
  rather than replaced by air; they require a dedicated limiting formulation.
  A file-backed geometry resolves every `mat.N` table only beside its `.geo`
  (or in an explicitly supplied `material_base_dir`); changing the process
  working directory cannot substitute an unrelated table. Only a pathless
  programmatic snapshot uses the working directory as its documented default.

- **Surface impedance** Z_s = R + jX in ohms, Leontovich `E_tan = Z_s·J`
  with the normal pointing out of the conductor. Passive surfaces have
  **R ≥ 0**. Z_s = 0 is exactly PEC; Z_s = η₀ ≈ 376.73 Ω is the matched
  (Weston) case. Lossy Z_s (R > 0) also damps interior cavity resonances.
  Conservatively, an effectively lossless reactive impedance
  (`max|Re Z_s| < 10⁻³ max|Z_s|`) on a closed BoR body is rejected because the
  available IBC EFIE cannot reliably rule out interior resonances and an
  IBC-compatible resonance-free CFIE is not implemented. Such IBCs remain
  available on intentionally open surfaces.
  Negative resistance is active and is rejected in both inline and tabulated
  material definitions.

- **IBC tapers** (`kind` = `constant` | `linear` | `cosine` | `exp`)
  interpolate from (R_start, X_start) at the segment's **first drawn point**
  (arc s = 0) to (R_end, X_end) at its **last** (s = 1). Reversing a
  segment's endpoint order reverses its taper. `exp` interpolates in log
  space, so it needs nonzero endpoints (zeros are floored to a tiny value
  rather than erroring — use `linear`/`cosine` to taper to true PEC). For
  `constant`, the end values are placeholders (write 0).

## 4. Drawing directions — 2D solver

The 2D solver models an infinite cylinder's cross-section in the (x, y)
plane (z out of the page). σ is a 2D width (per length); .grim exports use
the knife-edge dBke normalization.

**The normal is LEFT of the direction of travel, and it must point into the
correct medium.** Wrong winding is a **hard preflight error** — the solver
never silently flips (a wrong winding would silently corrupt TE results,
which is why the auto-flip was removed):

- TYPE 2 / TYPE 3: normal INTO AIR. A body surrounded by unbounded air is
  drawn **CW**; a void (air pocket) nested inside a body is drawn **CCW**;
  parity alternates with nesting depth.
- TYPE 4 (coated core): normal INTO the pos_mat coating → drawn **CW** when
  the coating surrounds it.
- TYPE 5: the winding IS your pos/neg labeling choice (never flagged).
- Open chains of TYPE 2/3 segments must run head-to-tail with consistent
  air sides; start-to-start / end-to-end meetings are errors.
- An open TYPE 2 contour is supported only in TM. TE uses a closed-obstacle
  MFIE for this boundary and therefore rejects an open TYPE 2 contour instead
  of applying an invalid equation.

**Angles (2D, "coming-from")**: 0° = from the right (+x), +90° = from the
top (+y), increasing CCW. Polarization aliases in the elevation cut:
**TM ≡ HH, TE ≡ VV**.

The stored 2-D complex output is the layer-potential coefficient `B`, while
the physical asymptotic convention is `A_phys = jB`. Thus
`sigma_2d = |B|²/(4k) = |A_phys|²/(4k)`; the `j` matters to external phase
comparisons even though it does not change scattering width.

## 5. Drawing directions — BoR solver

Same file format; the drawing plane is reinterpreted: **x = ρ (distance
from the rotation axis, must be ≥ 0), y = z (the axis, vertical in the
drawing)**. You draw the half-profile (generatrix); the solver revolves it.

- A closed body's generatrix is an **open polyline with both endpoints ON
  the axis (x = 0)**, traversed **from the +z end (top/nose) to the −z end
  (bottom/tail)**. With that traversal the left-of-travel normal faces the
  exterior — same left-normal rule as 2D. Bottom-to-top traversal, axis
  crossings (x < 0), or off-axis endpoints are hard errors.
- Multi-segment bodies stitch strictly head-to-tail **as drawn** (a
  start-to-start meeting is an error, not auto-reversed — reversal would
  also flip any IBC taper).
- Junction endpoints (coating edges, layer patches) must **coincide
  exactly** across the segments that meet there.

Supported material layouts (anything else is a named error):

| Segments present            | Solved as                                   |
|-----------------------------|---------------------------------------------|
| all TYPE 2                  | PEC (CFIE) / IBC-EFIE, tapers + mat tables  |
| all TYPE 3 (one pos_mat)    | homogeneous penetrable body (PMCHWT)        |
| TYPE 3 + TYPE 4             | fully coated PEC                            |
| TYPE 2 + 3 + 4              | **partially** coated PEC (junction circles) |
| TYPE 3 + 5 + 4              | two-layer stack, or patch-on-coating        |
| TYPE 3 + 5 + 5 + … + 4      | N-layer full stacks (TYPE 5 pos→neg flags chain outer→inner) |
| TYPE 2 + 3 + 4 + 5 (or 3+4+5 with several TYPE-4 materials) | **banded** side-by-side coatings (e.g. alternating MAGRAM bands): TYPE 5 walls where band touches band, TYPE 3 edge walls where a band meets bare metal |

For a direct single-surface PEC solve, EFIE, CFIE, and MFIE are available;
CFIE/MFIE require a closed body and MFIE is primarily diagnostic. An
intentionally open shell is EFIE-only. A nonzero TYPE 2 surface impedance is
also EFIE-only. Homogeneous penetrable bodies are dispatched to PMCHWT; the
remaining penetrable cases are restricted to the explicit layouts above.

Banded layouts: adjacent bands must carry **different** dielectric flags
(same material touching = one band); each band needs its own TYPE 4 covered
run; TYPE 5 wall pos_mat = the band the wall's left-of-travel normal faces;
band walls must run the full coating thickness (equal band thickness at the
wall); bare pieces are PEC-only for now.  With LOSSLESS band materials,
narrow (~±1 % in frequency) discrete-resonance bumps of the fictitious band
cavity can appear at unlucky mesh/frequency combinations — real lossy
MAGRAMs damp them; if a lossless run looks suspicious, re-solve at
f × 1.01 (the jitter check).

TYPE 5 orientation: pos_mat = the **outer** layer (the side the left
normal faces), neg_mat = inner. The TYPE 4 core's pos_mat must equal the
innermost layer's flag.

**Partial-coating rule of thumb**: if a bare piece carries an IBC, taper
Z_s toward **zero at the coating junction** (the physical edge treatment).
An abrupt Z_s step at a junction is an ill-defined sheet-model limit — the
solver can return a finite field but warns when |Z_s| > 0.02·η₀ there because
the result does not mesh-converge at that edge. One validation case showed a
~0.5 dB mesh/discretization plateau; that observation is **not** an accuracy
bound.

**Angles (BoR)**: aspect measured **from +z** — 0° = nose-on (top of the
drawing), 90° = broadside, 180° = tail-on. [0°, 180°] is the complete
monostatic dataset (`expand_to_360` mirrors it exactly). Mapping to the 2D
convention: 2D angle = 90° − BoR aspect (mirror, not a rotation — don't
just add 90°). Polarizations are true 3-D: **VV = θ-pol** (E in the plane
containing the axis), **HH = φ-pol**. σ is 3-D RCS in m² (dBsm).

Radar-frame caution for az/el products: with a **horizontal** body axis,
the waterline meridian plane is horizontal, so radar-VV on the el = 0 cut
equals the solver's **HH** sweep (the `bor_az_el_grid` tool handles this;
don't wire labels by hand).

## 6. Other things worth knowing

- **cfie_alpha**: real and validated in the **BoR** solver (closed PEC
  bodies default to CFIE, α = 0.5). In the **2D** solver it has **no
  effect** on any supported geometry (the indirect single-layer paths are
  far-field-immune to interior resonances) — a warning says so if you set
  it. Leave it 0 in 2D configs.
- **Solve-quality diagnostics**: every sample carries `linear_residual`
  (direct solves are typically near machine precision). A small residual
  confirms the reported algebraic system was solved; it does not by itself
  prove mesh, modal, formulation, or reduced-order-model convergence. A cheap
  independent resonance check is to re-solve at f × 1.0005;
  integral-equation resonances are razor-thin while physical RCS is normally
  smooth on that scale.
- **BoR modal convergence**: results report `mode_converged`. If the modal cap
  is reached before two quiet increments, the solver raises without returning
  a field. Increase `n_modes` and repeat.
- **BoR close folds and low frequency**: self, adjacent, and second-neighbour
  elements receive refined near quadrature, but there is no general
  near-quadrature route for topologically distant surfaces in a re-entrant
  fold. Touching or
  overlapping folds are rejected. A close fold also raises if the modal FFT
  far table would require more samples than its safety cap. Dedicated
  low-frequency stabilization is not implemented, so electrically small
  cases require an explicit frequency/formulation/convergence study.
- **2-D boundary visualization**: `compute_boundary_densities` returns
  formulation-specific SLP/DLP layer densities, not physical electric or
  magnetic surface currents. `compute_surface_currents` is a compatibility
  alias and does not change that interpretation.
- **Warnings are load-bearing.** The solvers put preflight findings, dead
  knobs, junction cautions, memory-precision switches, and resonance guards
  into `result["metadata"]["warnings"]` (and the HPC runners print them
  into the SLURM logs). Read them.
- **Memory/scale (BoR)**: assembly auto-switches between validated table
  and streaming paths and single/double precision by memory estimate; a
  hard 32 GB gate refuses with guidance instead of swapping. Compile
  `bor_stream_kernel.c` **on the target machine** for the fast native
  sampler — the loader only accepts a build tagged for the platform it is
  running on (a Mac `.so` copied to a Linux cluster is correctly skipped),
  and prints a one-time notice with the exact compile line when it falls
  back to the bit-equivalent (~2-8x slower) NumPy sampler.
- **Mie/analytic gates**: `tests/` contains the validation batteries (2D
  cylinders, BoR spheres/coatings/junctions/streaming). Run them once on
  any new machine before committing node-hours.

## 7. Minimal examples

**2D — coated PEC cylinder (drawn CW = normals into air/coating):**

```
Title: coated cylinder
Segment: shell 3
properties: 3 0 0 1 0
# outer dielectric interface, CW circle of radius 0.06 (head-to-tail arcs)
...
Segment: core 4
properties: 4 0 0 1 0
# PEC core, CW circle of radius 0.04
...
IBCS_Resistances:
Dielectrics:
1 3 -0.5 1 0
```

**BoR — missile with RAM-coated nose (see demo_bor_rocket.geo for the
full working file):**

```
Title: missile, RAM nose cap
Segment: coat 3
properties: 3 0 0 1 0
# coating outer surface: axis apex (0, z_top+t) -> offset ogive -> edge -> junction on core
...
Segment: nose_covered 4
properties: 4 0 0 1 0
# core under the coating: axis tip (0, z_top) -> ogive -> junction
...
Segment: body 2
properties: 2 0 0 0 0
# bare PEC: junction -> shoulder -> cylinder -> boattail -> base -> axis (0, z_bottom)
...
IBCS_Resistances:
Dielectrics:
1 3 -1 1.5 -0.8
```

All three segments run top-to-bottom (+z → −z); the coating edge and the
two core pieces share the junction point exactly.
