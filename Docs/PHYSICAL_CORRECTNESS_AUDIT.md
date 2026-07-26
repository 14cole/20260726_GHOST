# Physical-correctness audit

Audit date: 2026-07-26

## Executive verdict

The supported 2-D and body-of-revolution formulations now have strong
canonical evidence against independent cylindrical and spherical Mie series.
Known unsupported combinations, nonconvergence, invalid topology, stale field
metadata, and out-of-support interpolation fail closed instead of silently
publishing a plausible-looking field.

This does **not** prove every arbitrary geometry or material assignment. The
claim supported by this audit is narrower:

- valid geometries within the formulations listed below;
- passive, nonsingular constitutive data using the documented `exp(+jωt)`
  signs;
- a mesh, modal cap, and frequency/aspect sampling shown converged for the
  case being solved; and
- output that passes the current residual, schema, convention, and provenance
  checks.

The feature workflow is a reduced-order isolated-delta embedding. Its
canonical PEC groove anchor is green, but it is not a coupled 3-D Maxwell
solution and its measured error envelope is not universal.

## Supported scope and fail-closed boundaries

| Path | Supported in this audit | Deliberately rejected or not established |
|---|---|---|
| 2-D PEC/IBC | Closed PEC; all-sheet and sheet + pure-PEC scenes; supported TYPE 2/4 impedance cases; TM and TE where their integral equations apply | TE on an open TYPE 2 contour; TYPE 1 mixed with an IBC/dielectric/layered body; IBC flags on TYPE 3/5 transmission interfaces; active impedance |
| 2-D dielectric | Air-facing TYPE 3/4/5 homogeneous, lossy, magnetic, coated, and layered layouts documented in the geometry guide | TYPE 5-only scenes with no air-facing boundary; undefined material IDs, active media, singular ENZ/MNZ limits, malformed or extrapolated material tables |
| Direct BoR | PEC EFIE/CFIE/MFIE; open PEC shell with EFIE; dissipative IBC with EFIE | IBC with CFIE/MFIE; closed, effectively lossless reactive nonzero IBC (`max|Re Zs| < 10⁻³ max|Zs|`), including a reactive bare section of a partial coating; CFIE/MFIE on an open or incorrectly oriented generatrix |
| Dispatched BoR materials | Homogeneous PMCHWT and the explicitly enumerated coated/multilayer layouts | Unimplemented interface/IBC combinations and arbitrary material junction graphs |
| Line-expanded features | Rounded, converged 2-D delta coupon or full-object coefficient placed on a known 3-D line, with strict angle/frequency support and explicit phase conventions | Detached TYPE-1 coupon terminations, coupled body-feature currents, multiple scattering, unprovided diffraction/creeping-wave terms, arbitrary conical dependence, or a universal material/geometry error bar |

The BoR code also lacks a dedicated low-frequency stabilization and a general
near-singular route for arbitrary nonadjacent close folds. Such a fold is
rejected when the required far-kernel sampling exceeds the safety cap.

## Independent physical anchors

### 2-D solver

- PEC, dielectric, lossy dielectric, coated PEC, and near-PEC impedance
  cylinders passed the cylindrical-Mie battery. The worst exact
  PEC/dielectric/coated error was `0.013 dB`; the finite `1 Ω` impedance versus
  the PEC limiting reference differed by `0.047 dB`.
- The dedicated complex-impedance cylinder comparison agreed to
  `1.084e-4` relative error for TM and `1.717e-5` for TE.
- The resistive-sheet cylinder comparison agreed to `8.951e-4` for TM and
  `5.043e-4` for TE.
- The interior-resonance battery passed all ten gates; worst far-field error
  was `0.0040 dB` with maximum reported linear residual `1.0e-12`.
- Circular-cylinder isotropy spread was `0.0000 dB` in the printed Mie cases.

### Body-of-revolution solver

- PEC spheres over `ka = 0.5, 1, 3, 6` agreed with spherical Mie to at worst
  `0.040 dB`.
- CFIE and MFIE sphere anchors were within `0.032 dB` and `0.014 dB`.
- The impedance-sphere anchor was within `0.023 dB`; the Weston matched
  impedance null was about `-59 dB` relative to PEC.
- Lossless, lossy, magnetic, and high-contrast dielectric spheres plus
  lossy/thin/magnetic/lossless coated PEC spheres were within `0.065 dB`.
- Smooth multilayer sphere anchors were within `0.056 dB`. A coarse
  fictitious material patch showed `0.143 dB` error and improved to
  `0.024 dB` on the doubled mesh.
- Dispatch spot checks against Mie were within `0.047 dB`; the adaptive
  frequency sweep's worst Mie error was `0.079 dB`.

These sphere/cylinder comparisons are the strongest evidence in the project.
Mesh, symmetry, formulation, native-versus-NumPy, streaming-versus-table,
serialization, and dispatch-versus-direct comparisons are important
implementation checks, but they are not independent Maxwell references.

## Feature-model anchor

The canonical cross-formulation geometry used for the feature model is a circumferential
cosine-squared PEC groove on a finite cylinder. Its axisymmetry lets the BoR
solver provide the 3-D complex featured-minus-smooth truth while an independent
2-D coupon supplies the line coefficient.

The local TE and TM phase mappings are calibrated **before** their projected
contributions are added. This matters because a finite ring mixes both local
coefficients into an HH or VV channel away from exact broadside. With
`PSI_TE/VV = -9.2°` and `PSI_TM/HH = 166.9°`, the current production expansion
over the declared ±20° sector has:

| Channel | Worst magnitude error | Worst residual phase |
|---|---:|---:|
| VV | `1.53 dB` | `14.3°` |
| HH | `1.97 dB` | `11.9°` |

The release envelope remains the more conservative `3.5 dB` magnitude and
`25°` residual phase. It applies only to this converged canonical PEC groove
family and local-edge sector. Material-specific and electrical-size sweeps
must be read as scope studies, not as universal guarantees.

The same ring supplies the phase calibration and its closure test. Therefore
the fitted phase offset is not an out-of-sample validation; the nonconstant
residual versus aspect, absolute magnitude, coupon-width convergence, and
separate frequency and electrical-size sweeps are the additional completed
evidence. The material-specific coated-transfer sweep did not complete in the
available validation window and is **inconclusive**, not a pass. The
`validate_feature_sum.py` G1 check deliberately reuses this geometry to prove
that serialization and the production API preserve the calibrated result.

The frequency scope sweep at `1`, `6`, and `18 GHz` produced essentially the
same normalized errors for the electrically scaled canonical family: VV
`1.54 dB / 14.3°` and HH `1.69 dB / 12.1°`. The fixed-frequency
electrical-size sweep remained inside its `2.5 dB / 25° / ±20°` study
criterion from groove widths `0.08λ` through `1.0λ`; the feature itself was
only `-52.3 dBsm` at the smallest size, where relative errors are least useful.
These are scope studies of one scaled PEC family, not proof of scale or
frequency invariance for arbitrary features.

The coupon-termination bake-off also supplies a negative result. Rounded
capsules at `8λ` and `12λ` were usable; the `12λ` reference gave the canonical
`1.53 dB / 14.3°` VV and `1.97 dB / 11.9°` HH result. Detached TYPE-1 absorber
sheets left about `178–179°` VV phase error and only a `0–5°` usable sector.
An on-body IBC taper failed the HH criterion at `8λ` and only narrowly passed
at `12λ`. Consequently the current supported recipe remains the converged,
rounded capsule; the alternative terminations are not release shortcuts.

The wing anchor establishes the PEC flat-plate broadside normalization
`4πA²/λ²` and the implemented uniform straight-aperture `sinc²` null pattern.
The corner and point-pattern tests primarily validate their encoded
reduced-order models and coordinate transforms, not independent full-wave
physics.

## Corrections made during the audit

1. Material/geometry parsing now rejects missing or duplicate five-field
   segment properties, inconsistent header/property TYPE declarations,
   disconnected primitives inside one segment, negative flags, malformed or
   duplicate material IDs, extrapolated tables, active data, singular ENZ/MNZ,
   and unsupported interfaces rather than substituting air or clamping a
   table. File-backed snapshots resolve `mat.N` only beside their source
   `.geo` unless an explicit material directory overrides it.
2. Automatic 2-D and BoR meshing uses the shortest wavelength among the
   actually referenced media, including dispersive table values.
3. FMM/GMRES, dense residual, BoR modal-truncation, far-kernel resolution, and
   quality failures abort without returning a field. Strict 2-D quality
   gating is the public-API and GUI default, is forwarded by the bistatic and
   fine-mesh GUI paths, and can be disabled only explicitly for diagnostics.
   Production 2-D solves require a normalized residual no larger than
   `1e-6` and actually compute the requested condition diagnostic.
4. Each production BoR modal solve checks the worst normalized residual over
   all right-hand-side columns (`<= 1e-8`) and a LAPACK 1-norm condition
   estimate (`<= 1e12`). Missing, nonfinite, or excessive diagnostics abort
   the solve; dispatch also requires modal, field/power, and finite-value
   consistency.
5. Geometry preflight rejects invalid winding, zero-length elements,
   self-intersections, close folds outside the quadrature capability, and
   formulation/topology mismatches.
6. BoR self, adjacent, and second-neighbour interactions use conservative
   refined quadrature; unsupported effectively lossless reactive closed-body
   IBC-EFIE is rejected because an interior-resonance-free formulation is
   unavailable.
7. Exact physical nulls remain zero in linear arrays and files; logarithmic
   floors are presentation-only.
8. Radar azimuth/elevation axes, BoR aspect support, on-axis polarization
   isotropy, and Jones-basis rotations are validated and do not silently clamp
   outside the solved body cut.
9. The line model now uses signed local polarization axes, strict
   frequency/angular support, validated perimeters/normals/directions, and the
   jointly calibrated coefficient-level phases described above.
10. Production raw coherent GRIM fields are stored in `float64`; the central
   writer and component rewrites enforce that dtype. This prevents subtraction
   of two large full-object fields from destroying a much smaller seam delta
   through on-disk `float32` quantization. Power and display phase may remain
   compact `float32`; the raw field is authoritative for coherent work.
11. GRIM writers/loaders enforce dimensional normalization, units, phase
    origin, amplitude convention, field domain, field/power/phase
    consistency, exact nulls, monotone axes, and explicit component
    `combine_role`. Joining files with different coherent-field conventions is
    refused.
12. Feature inputs fail closed by role: a body cannot be used as a seam, a
    seam cannot be used as a body, declared full-object and delta semantics
    cannot conflict, both physical TM/TE channels are mandatory, and singleton
    elevation plus exact convention metadata are required.
13. Local workflow caches and generic local/HPC unit runners now bind reuse to
    exact Python/native source, geometry/material/unit inputs, output bytes,
    and Python/platform/NumPy/SciPy/BLAS runtime fingerprints. Output
    attestations are checked before resume/collection. Missing, extra,
    modified, mixed-source, or legacy outputs are reported as stale rather
    than silently reused.
14. Long-running local solves re-fingerprint their inputs and implementation
    before committing a completed cache. HPC collectors copy into staging,
    semantically re-verify the staged field/attestation pairs, and write an
    atomic exact-inventory commit marker last. Step-3 component builders first
    mark their bundles in progress, then commit exact output inventories and
    hashes. Step 1c commits the delta library itself; step 4 discovers only
    verified component inventories and commits its combined products; step 5
    commits its exact ranking inputs and CSV.
15. GUI execution now honors the selected monostatic or bistatic mode on both
    base and refinement meshes. A 2-D dBke CSV export rejects BoR/dBsm data
    instead of silently relabeling it.

## Final validation ledger

The final post-correction run completed with:

- `188/188` backend unit/regression tests passing, including configured 1b/2b
  SLURM-manifest and worker-script generation;
- the complete 2-D Mie, resonance, mixed-material, winding, geometry-hole,
  GRIM interoperability, dataset-join, delta-library, feature-sum, and signed
  polarization gates passing;
- BoR phase-1 and phase-2 analytic/formulation gates passing;
- all banded-coating, material-junction, and multilayer BoR gates passing,
  including lossy, magnetic, IBC, partial-patch, and three-layer cases; and
- the canonical feature-sum result reproduced at `1.53 dB / 14.3°` for VV and
  `1.97 dB / 11.9°` for HH.

Two deliberately expensive stress studies remain **inconclusive** rather than
failed: the coated feature-transfer sweep did not finish after more than
87 minutes, and the `ka=10`, 400-element streaming stress did not finish in
the allotted run. Smaller table-versus-streaming comparisons agreed at roughly
`1e-15`, but that does not substitute for completion of the large stress case.

## What remains unproven

- Arbitrary 2-D bistatic patterns and general interacting/composite 2-D
  geometries have no external full-wave regression beyond the canonical
  analytic cases.
- Arbitrary BoR profiles are supported numerically, but smooth-sphere Mie,
  convergence, symmetry, and limiting cases cannot prove every sharp profile,
  close fold, material junction, or partial coating.
- Finite coating patches and triple junctions lack an independent external
  full-wave reference. Current evidence is Mie limiting cases plus mesh and
  same-solver consistency.
- The feature sum omits induced-current redistribution and body-feature or
  feature-feature multiple scattering. Its translation phase is correct inside
  the encoded isolated-scatterer model, not proof of coupled-system phase.
- The flat-coupon line coefficient has no longitudinal/conical `d·t`
  dependence. Corners, endpoints, coefficient discontinuities, and short or
  sharply curved loops can add physics outside the ring anchor.
- The canted-corner rolloff, binary illumination/occlusion masks, and
  phase-unknown power additions are explicitly engineering estimates.
- Passive dispersive values at sampled frequencies are checked, but the code
  does not prove that a user-supplied table is a globally causal material
  model.

## Existing workflow artifacts

The shipped numbered workflow currently contains 82 generated files and no
current provenance manifests. They predate the strict field schema and must
not be treated as validated release output:

- the existing body files use a legacy 19-aspect grid rather than the 85 exact
  aspects required by the current radar grid;
- the step-3 component and step-4 combined files fail current coherent
  field/power or convention checks;
- the wing output combines an old wing/corner power artifact instead of the
  current separate coherent-wing and power-role-corner files; and
- the cavity pattern lacks the current compact-pattern convention.

They were deliberately preserved rather than overwritten. The minimal active
local rebuild order is:

1. run `1a_solve_2d_local` and `2a_solve_body_local` with `FORCE=True`
   (independently or in parallel);
2. run `1c_build_deltas`; steps 3a and 5 now consume its committed `Deltas/`
   library directly, including `collection_manifest.json`;
3. replace or explicitly normalize the external cavity pattern;
4. run optional step 0 if shadowing is enabled;
5. run `3a_doors`, `3b_add_wing` with `FORCE=True`, and `3c_add_cavity`;
6. run `4_combine`; then run optional step 5 if the ranking report is wanted.

The checked-in numbered workflow now treats `1b` and `2b` as the production
defaults. Their submitters generate immutable configured drivers and frozen
geometry/material inputs; workers attest every unit; collectors stage,
semantically revalidate, and atomically commit the exact bundles. The local
`1a`/`2a` branches remain supported for development-sized cases by repointing
`INPUTS`/`BODY_DIR`.

## Release rule

For a new geometry/material case, “the solver returned numbers” is not a
physical acceptance criterion. Before treating an output as release-quality:

1. confirm the case lies inside a supported formulation;
2. run mesh, modal-cap, and where relevant frequency-jitter/formulation
   convergence studies;
3. require finite residual and condition diagnostics and no ignored warnings;
4. keep all material, phase, units, support, and provenance metadata;
5. validate a representative canonical or independent-reference case for any
   new material/interface family; and
6. for line-expanded features, establish a case-specific angular,
   electrical-size, and host-material envelope or use a coupled 3-D solver.
