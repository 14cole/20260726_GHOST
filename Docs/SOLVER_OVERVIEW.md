# RCS Solver — 2D BEM, Body-of-Revolution MoM, and feature/wing line expansion

Method-of-moments radar-cross-section solvers with material conditioning
(PEC / IBC / dielectric / coated), plus a line-expansion layer that adds
non-axisymmetric surface features (panel gaps, seams, doors) and wings/fins to
a Body-of-Revolution baseline to build a reduced-order composite signature.

## Start here

| Doc | What it covers |
|-----|----------------|
| [PHYSICAL_CORRECTNESS_AUDIT.md](PHYSICAL_CORRECTNESS_AUDIT.md) | Current physical evidence, corrections, supported claim boundary, unresolved gaps, and stale-artifact status |
| [GEOMETRY_GUIDE.md](GEOMETRY_GUIDE.md) | `.geo` file format, TYPE table, sign/winding/angle conventions, 2D & BoR drawing rules |
| [FEATURE_SUM_GUIDE.md](FEATURE_SUM_GUIDE.md) | **How-to for the feature/wing workflow** — coupon rules, all steps, exports, troubleshooting |
| [BOR_SOLVER_PLAN.md](BOR_SOLVER_PLAN.md) | BoR solver scope, formulations, phases, cost model |
| [BOR_CONVENTIONS.md](BOR_CONVENTIONS.md) | BoR far-field/pol conventions and analytic anchors |

This document describes the **library**.  For how to actually run a job, the
numbered steps in [../README.md](../README.md) are the workflow — they replaced
the older `Step_by_step/`, `Step_by_step_hpc/` and `Workflow_steps/` walkthroughs
that earlier versions of this file pointed at.

## The solvers

- **2D BEM** (`rcs_solver.py`) — infinite-cylinder cross-section RCS (σ is a 2D
  scattering width, dBke). Entry: `solve_monostatic_rcs_2d`.
- **BoR-MoM** (`bor_solver.py`, `bor_dispatch.py`) — axisymmetric 3-D RCS
  (dBsm). Entry: `solve_bor` / `bor_dispatch.solve_monostatic_rcs_bor`;
  radar-frame body product via `bor_az_el_grid`.
- **Shared I/O** — `.geo` via `geometry_io.py`, `.grim` (NPZ) via `grim_io.py`,
  analytic references in `mie_sphere.py` / `mie_reference.py`.

### Supported physics and fail-closed boundaries

“Arbitrary geometry” is limited to valid, non-self-intersecting geometries that
fit a supported formulation. The preflight intentionally rejects a model rather
than reinterpret or silently ignore unsupported material flags.

| Path | Supported scope | Explicit boundary / convergence requirement |
|------|-----------------|---------------------------------------------|
| 2-D | Closed PEC; all-sheet and sheet + pure-PEC scenes; supported TYPE 2/4 impedance cases; and air-facing TYPE 3/4/5 dielectric layouts described in the geometry guide. TM and TE are available where the selected integral equation applies. Both automatic meshing and negative `N` resolve the shortest wavelength among referenced materials. | A TYPE 5-only scene has no supported air exterior. TYPE 1 mixed with an IBC body, dielectric body, or layered coating raises. TYPE 3/5 transmission-interface IBC flags raise. TE on an open TYPE 2 contour raises because its closed-obstacle MFIE is invalid there. Active media, singular ENZ/MNZ inputs, invalid winding/topology, and iterative/FMM nonconvergence also raise. |
| Direct BoR | Single-surface PEC EFIE, CFIE, and MFIE; open PEC shells with EFIE. MFIE is primarily a diagnostic formulation. | A nonzero surface impedance is implemented only in EFIE; CFIE/MFIE + impedance raise. A closed body with effectively lossless reactive nonzero impedance (`max|Re Zs| < 10⁻³ max|Zs|`) also raises because IBC-EFIE cannot reliably exclude interior resonances. CFIE/MFIE require a closed, correctly oriented generatrix. |
| Dispatched BoR materials | Homogeneous dielectric PMCHWT and the explicitly enumerated coating/layer layouts in the geometry guide. | Unsupported TYPE/material/IBC combinations raise. Failure to converge before the azimuthal modal cap aborts without returning a field; increase `n_modes` and repeat. |

The BoR self, adjacent, and second-neighbour element interactions use refined
quadrature, preventing a smooth refined profile's local O(h) gaps from
inflating the modal FFT grid. More distant topological pairs still use the
modal FFT far path:
touching/overlapping folds raise, and a fold whose required FFT sampling
exceeds the safety cap raises rather than accepting a capped table. A general
nonadjacent close-fold near-quadrature route and a dedicated low-frequency
stabilization are not implemented. Mesh, modal, frequency-jitter, and
formulation-convergence studies remain necessary for geometries outside the
analytic gates.

`compute_boundary_densities` is a visualization/debug API. Its values are
formulation-specific SLP/DLP representation densities; they are not generally
physical electric or magnetic surface-current densities. The legacy
`compute_surface_currents` name is only a compatibility alias with the same
boundary-density semantics.

## Feature / wing line expansion

A joint or wing is characterised by a 2-D cross-section and can be reused only
with the same cross-section, clean local material/coating stack, frequency,
units, phase origin, polarization convention, and compatible curvature. Both
solvers share a calibrated far-field convention, so placement supplies the
modeled geometric translation phase; this does not establish omitted
body-feature coupling or embedding phase.

| File | Role |
|------|------|
| `line_expand.py` | the expansion: `seam_coefficients_from_2d` (differential feature), `coefficients_from_2d` (full-object wing), `expand_perimeter`, `combine`; jointly measured local-TE/TM calibration constants `PSI_VV_DEG` / `PSI_HH_DEG` |
| `feature_sum.py` | pipeline: `make_delta_grim` (coherent subtract → reusable delta `.grim`), `load_seam_from_grim` / `load_coefficients_from_grim`, `sum_features` (N placements, per-placement normals for wings, `corners=` for wing-body dihedral double-bounce), `corner_amplitude`, `export_signature_grim` (body-frame), `export_radar_grim` (**final product**: az × el × freq × pol `.grim`) |
| `demo_feature_signature.py` | runnable end-to-end example |
| `trade_study.py` | `door_trade_study(...)` — compare feature delta designs at one perimeter (isolated peak dBsm + lift over body); `python3 trade_study.py` runs a demo |
| `occluder.py` | self-contained STL body-shadowing: `Occluder.from_stl("body.stl")` → pass `occluder=` to `sum_features`/exporters/`door_trade_study` to mask features the body geometrically blocks. The heuristic default `bias` is for the calibration sweep; production step 3a requires the explicit value established with `0_calibrate_shadowing`, because an over-large bias can skip a real nearby blocker |
| `feature_sum.save_body_grim` / `load_body_grim` | the BoR body solve as a normal `.grim` (aspect × 1 × freq × [VV, HH], 3-D convention, amplitude preserved) — so the baseline opens in the viewer, and it doubles as the solve cache. **Its azimuth axis is the BoR aspect**, tagged in `units["azimuth_meaning"]` and `history` |
| `grim_compat.py` | bridge to the **GRIM_Revised_2** viewer/dataset tool (`grim_dataset.RcsGrid`): `to_grid` / `from_grid`, `field_amplitude` (the one conversion that matters — see below), `load_pattern_any` (read `.out` / `.ss` / PIO / theta-φ CSV-TXT as a point-scatterer pattern), `describe`. Non-`.grim` formats cannot encode the complete placement convention, so the caller must explicitly pass the verified `point_pattern_convention_metadata()` when importing them; untagged patterns are refused. |
| `grim_naming.py` | the production filename grammar — `SEAL-00-01_0.010gap_OPN.grim` (featured) / `_FRD` (clean) / no marker (delta), and `HH_2.000GHz_<variation>.grim` as it comes off the solver. `join_grims` folds the per-(pol, frequency) files into one per variation; `pair_variants` matches OPN to FRD so a whole study subtracts unattended |
| `delta_library.py` | filename-indexed delta library (`0.020bmag_0.060gap.grim`): `DeltaLibrary.from_dir` / `select` / `resolve` (min-max or explicit tolerance lists, off-grid snap policy), plus `arc_slices` / `smooth_cycle` / `tolerance_placements` to spread a tolerance around a door perimeter |

Key facts (see the guide for the full story):
- This is an **engineering delta embedding**, not a fully coupled 3-D Maxwell
  solve. It does not create body-feature mutual coupling, multiple scattering,
  creeping waves, or diffraction terms absent from the supplied models.
- Delta is a conditionally reusable, vehicle-independent artifact; features add
  as a *differential* (never body+feature — the smooth skin is already in the
  body).
- The sum is **single-bounce**; the wing-body **corner double-bounce** (often
  dominant at the root) is added separately via `corners=` (PO-level estimate,
  with an analytic polarization map inside that model). A **canted** root
  (not 90°) uses a screening heuristic that deflects the modeled lobe `2δ` off
  the bisector and rolls its peak off by `cos²(2δ)`; this is not a validated
  full-wave canted-corner pattern or a guaranteed bound.
- **Compact 3-D features** (a blind cavity) that can't be line-expanded are
  added via `points=`: a precomputed 3-D delta pattern (az/el/freq/pol grim
  from an **external 3-D MoM**) placed at one body coordinate —
  `point_scatterer_amplitude` (regular azimuth/elevation-grid interpolation of
  complex fields, orientation + pol rotation, single-point placement phase,
  shadow mask).
- Components use a binary high-frequency local-normal illumination mask,
  appropriate to a single-bounce geometric visibility test on a **convex**
  body. For a
  **non-convex** body, pass `occluder=Occluder.from_stl("body.stl")` to add
  geometric body-shadowing (binary ray occlusion against the clean STL). These
  masks omit partial illumination, shadow-boundary diffraction, creeping waves,
  and induced-current redistribution.
- **Validity**: the converged canonical PEC-groove anchor supports ±20° of
  local-edge broadside at limits of 3.5 dB magnitude and 25° residual phase.
  These numbers are not a universal material/cross-section error bar. The wing
  gate anchors only PEC flat-plate broadside normalization and the implemented
  uniform straight-aperture `sinc²` pattern—not arbitrary finite wings,
  materials, conical incidence, tips/roots, or body coupling.
- **Combine modes**: `coherent` is the deterministic complex field represented
  by the reduced-order model; `hybrid` and `envelope` are explicit
  engineering/statistical estimates. Exported `rcs_power` is always the
  physical power conversion of the stored coherent field, while a requested
  noncoherent estimate is stored separately as `combination_estimate_power`.

## Sharing files with GRIM_Revised_2 (the viewer / dataset tool)

Both projects read and write `.grim` (npz) and they **are** compatible:
`RcsGrid.load` reads every file this repo writes, and phase agrees to float32.
`grim_compat.py` is the bridge; `tests/validate_grim_compat.py` gates it (and
skips cleanly when that repo isn't present).

**There are exactly two data types**, and they follow the dimensionality of the
solve. `rcs_power` is the physical power quantity associated with the stored
modeled field, but this repo's `rcs_amp_real/imag` is not `sqrt(rcs_power)`:

| data type | stored complex amplitude | `rcs_power` holds | log unit | √power / \|amp\| |
|---|---|---|---|---|
| **2-D** | layer-potential coefficient `B`; `A_phys = jB` | σ₂d = \|B\|²/(4k) = \|A_phys\|²/(4k), a scattering *width* in metres | dBke | `1/(2√k)`, per frequency |
| **3-D** | physical far-field amplitude `F` | σ = 4π\|F\|², a cross-section in m² | dBsm | `√(4π) = 3.5449` |

2-D "RCS" is a width, so its length comes from `1/k` — which is why that factor
moves with frequency. 3-D is a cross-section and the `4π` is the isotropic
reference in the definition of σ. Both are set by the solver's own far-field
convention ([rcs_solver.py](../Backend/rcs_solver.py),
[bor_solver.py](../Backend/bor_solver.py)),
not by the file format — the `.grim` writer stores both encodings side by side
without converting. The global `j` between stored `B` and physical asymptotic
`A_phys` leaves 2-D width unchanged but must be applied when comparing complex
phase with an external convention; `.grim` metadata records this relation.

**A delta is a 2-D data type, not a third one.** `rcs_domain='delta'` is an
*orthogonal* axis: it says the samples are a difference (featured − clean), which
is what `sum_features` routes on, and it says nothing about units. (Deltas
written before this was true stored bare \|dA\|² and are tagged
`power_domain='delta_amp_sq'`; `amp_scale` still honours them, but their dBke
display reads `10log10(4k)` high — +24 dB at 3 GHz, +27 at 6, tilting 3 dB/octave
— so rebuild them with step 03/04.)

So everything `RcsGrid` does in the **power** domain (dBsm/dBke, crop, mirror,
join, statistics, plotting) is correct on these files as-is. But `RcsGrid.rcs` is
`sqrt(power)·e^{jφ}` — a *scaled* field: fine to add or subtract within one data
type (the constant cancels), wrong if you mix 2-D and 3-D. Use
`grim_compat.field_amplitude()` to get this repo's amplitude back, or
`grim_compat.describe(path)` to print which case a file is.

`RcsGrid.save` now carries unmodelled keys through, so a file can go
here → viewer → back without losing its complex amplitude or its `rcs_domain`
tag (`sum_features` **routes** on that tag, so a delta that came back labelled
`power_phase` would be silently misread as a wing coefficient). A grid that was
cropped or joined correctly drops the stale amplitude instead — reattach one with
`grim_compat.from_grid(grid, out, amp=...)`.

## Validation

Supported solver anchor cases are validated against analytic series; other
gates are cross-formulation or implementation-consistency checks. These
establish the named cases and numerical limits, not every arbitrary geometry
or a coupled non-axisymmetric vehicle. Run the feature suite:

```bash
cd tests && python3 run_feature_gates.py          # feature/wing suite
cd tests && python3 run_feature_gates.py --all     # + foundational solver batteries
cd tests && python3 run_feature_gates.py --fast     # skip the slow BoR sweeps
```

| Gate | Checks |
|------|--------|
| `validate_wing.py` | wing expansion == analytic flat plate `4πA²/λ²` (±0.03 dB), `sinc²` span pattern |
| `validate_corner.py` | implementation/self-consistency of the chosen PO corner estimate, polarization map, signed-sinc aperture, and canted-corner heuristic—not an external Maxwell reference |
| `validate_point_scatterer.py` | bookkeeping/transform checks for a supplied 3-D delta pattern: placement phase, shadow, polarization rotation, energy—not validation of the supplied pattern's physics |
| `validate_grim_compat.py` | interop with GRIM_Revised_2: RcsGrid reads all three grim flavours, the power↔amplitude table is predicted from tags, round-trip keeps amplitude *and* domain tags, a cropped grid drops its stale amplitude, external patterns (`.out`/`.ss`/PIO/θφ) place identically to a `.grim`. Skips if that repo is absent |
| `validate_join_datasets.py` | production filenames: grammar round-trip, join preserves every cell exactly, ragged/duplicate/mixed-unit or mixed-field-convention refusals, TM-vs-HH aliasing, OPN/FRD pairing, study id is categorical |
| `validate_delta_library.py` | filename-only delta library: canonical names, scan errors (mixed decimals, missing variable, ragged grid, unindexed files), select/resolve + off-grid snap reporting, `rev` tie-break, arc partition, optimal `smooth_cycle` ordering |
| `validate_line_expansion.py` | cross-formulation line model against converged BoR groove truth; jointly measures the local TE/TM coefficient phases and evaluates the actual production sum over the canonical ±20° / 3.5 dB / 25° envelope |
| `validate_feature_sum.py` | pipeline G0–G6: delta subtraction, null, superposition, placement phase, exporters, and wing integration; its G1 reuses the calibration geometry |
| `validate_full_workflow.py` | algebra/schema/full-chain integration: perimeter .txt → delta grim → body+feature+wing+corner → radar .grim; not absolute EM truth |
| `validate_line_expansion_band.py` | calibration behavior at the sampled 1, 6, and 18 GHz fixed-electrical-size anchors |
| `validate_line_expansion_size.py` | canonical PEC-groove behavior versus electrical size; any reported sweet spot is family-specific |
| `validate_line_expansion_coated.py` | applies the bare-PEC coefficient calibration to one lossy RAM-on-PEC groove anchor; not a universal coated-feature bound |
| `validate_coupon_bakeoff.py` | diagnostic comparison of three coupon terminations against the canonical ring truth; it does not turn the winning termination into a universal coupon model |

Foundational batteries (`--all`): 2D Mie, BoR Mie sphere, BoR phases 1–4
(PEC/CFIE/IBC/PMCHWT/coated/integration). Additional solver gates in `tests/`
cover junctions, multilayer, streaming, orientation, and preflight.

BoR solves inside the gates cache to hidden `.pkl` files after the first run.
Current caches are keyed by exact geometry/settings, solver/native-kernel
source hashes, and the Python/NumPy/SciPy/platform environment, so a source or
environment change forces a fresh reference solve.
Compile `bor_stream_kernel.c` **on the target machine** for the fast native
sampler (see GEOMETRY_GUIDE.md §6).
