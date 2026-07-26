# RCS workflow

Everything you edit lives in a numbered step folder. Everything you don't lives
in `Backend/`. There is nothing else to learn.

```
Claude21/
  Backend/                the solvers and libraries — you never open these
  Docs/                   the reference guides
  grid.py                 the output grid, declared ONCE for steps 3a/3b/3c
  GHOST_GUI.py            the geometry/solver GUI — a way in, not a step

  0_calibrate_shadowing/  before you ever set SHADOW = True          (optional)

  1a_solve_2d_local/      solve the 2-D cross-sections, on this machine
  1b_solve_2d_hpc/        the same solve, on a SLURM cluster    (pick 1a OR 1b)
  1c_build_deltas/        join + subtract  ->  the reusable delta library

  2a_solve_body_local/    solve the bare body, on this machine  (slow, once)
  2b_solve_body_hpc/      the same solve, on a SLURM cluster    (pick 2a OR 2b)

  3a_doors/               seams and doors, with a build tolerance
  3b_add_wing/            a wing/fin AND its root corner            (as many of
  3c_add_cavity/          a compact 3-D feature placed at a point   these as you
                                                                      need)
  4_combine/              body + every component  ->  the deliverable

  5_rank_designs/         which seam design actually matters?   (a side question)
```

A stage that can run either place comes in a **local** flavour (one `run.py`)
and an **HPC** flavour (`submit.py` → `status.py` → `collect.py`). The numbered
workflow defaults to the HPC outputs because the production 2-D libraries and
BoR bodies can exceed a workstation. Pick one path; both use the same solver
formulations and downstream schemas.

Every `run.py` opens with a header saying what it does, what it reads, what it
writes, and what its `KNOBS` block controls. Read that, edit the knobs, run it.

## The run

```bash
cd 1b_solve_2d_hpc
python3 submit.py
python3 status.py
python3 collect.py
cd ../1c_build_deltas && python3 run.py

cd ../2b_solve_body_hpc
python3 submit.py
python3 status.py
python3 collect.py

cd ../3a_doors   && python3 run.py          # uses 1c/Deltas directly
cd ../3b_add_wing   && python3 run.py        # only if you have a wing
cd ../3c_add_cavity && python3 run.py        # only if you have a cavity

cd ../4_combine     && python3 run.py        # instant — nothing is re-solved
```

For a local development-sized case, run `1a_solve_2d_local/run.py` and
`2a_solve_body_local/run.py`, then repoint `1c INPUTS` and the downstream
`BODY_DIR` knobs to those local outputs.

Each HPC branch uses the same three commands:

```bash
cd 1b_solve_2d_hpc              # or 2b_solve_body_hpc
python3 submit.py       # edit the KNOBS first; SUBMIT = True to actually sbatch
python3 status.py       # …wait…
python3 collect.py
```

The checked-in defaults already point to the cluster outputs:

| production input | knob | default |
|---|---|---|
| 1b 2-D collection | `1c_build_deltas/run.py` → `INPUTS` | `../1b_solve_2d_hpc/results` |
| 2b BoR collection | `BODY_DIR` in 3a, 3b, 3c, 4, 5 | `../2b_solve_body_hpc/Body` |
| step-1c delta library | `DATASETS_DIR` in 3a and 5 | `../1c_build_deltas/Deltas` |

**How closely the two flavours agree.** 1a/1b and 2a/2b use the same respective
solver calls; the cluster changes where work runs, not the formulation. Small
roundoff changes can remain across BLAS/thread configurations and look large in
dB at a deep null. Body caches are fingerprinted against the `.geo`, every
adjacent `mat.*` table, the units, Python/native solver source, and numerical
runtime. Cluster runs also freeze those exact inputs inside the run directory
and give every resumable unit a byte-level output attestation, so collection
cannot pair a changed field with a different profile or run state.
Collectors stage and revalidate the exact bytes before committing them.
Steps 1c, 3a/3b/3c, 4, and 5 also publish exact-inventory commit manifests;
downstream steps reject missing, stale, mixed, or modified bundles.

## Physics scope and hard limits

The evidence, corrections, unsupported cases, and stale-artifact status from
the full review are recorded in
[Docs/PHYSICAL_CORRECTNESS_AUDIT.md](Docs/PHYSICAL_CORRECTNESS_AUDIT.md).

The solvers are fail-closed at known formulation boundaries; “arbitrary
geometry” means a valid, non-self-intersecting geometry inside this supported
scope, not every Maxwell boundary-value problem.

- The 2-D solver covers closed PEC and supported IBC boundaries, resistive
  sheets, and the dielectric/interface layouts listed in the geometry guide.
  Automatic and negative-`N` meshing use the shortest wavelength among the
  referenced materials. An IBC on a TYPE 3 or TYPE 5 dielectric transmission
  interface is rejected, as is TE polarization on an open TYPE 2 contour.
  `compute_boundary_densities` returns integral-equation layer densities for
  inspection, not physical electric or magnetic surface currents.
- A stored 2-D complex amplitude is the layer-potential bare-integral
  coefficient `B`. The physical asymptotic convention is `A_phys = j B`;
  the global factor does not change `sigma_2d = |B|^2/(4k)`, but it matters
  when phases cross an external-tool boundary.
- The direct BoR solver covers PEC EFIE/CFIE/MFIE (MFIE is primarily
  diagnostic); nonzero surface impedance is supported only with EFIE.
  A closed body with an effectively lossless reactive nonzero impedance
  (`max|Re Zs| < 10⁻³ max|Zs|`) is rejected because the available IBC-EFIE path
  cannot reliably exclude interior-resonance contamination.
  `bor_dispatch` selects PMCHWT for its supported homogeneous dielectric and
  enumerated coating/layer layouts. An open shell is EFIE-only.
- BoR mode truncation reports `mode_converged`; hitting the modal cap aborts
  without publishing a field and requires a larger-cap convergence study.
  Nonadjacent touching
  folds are rejected, and a close fold is rejected if its far-kernel FFT
  sampling requirement exceeds the safety cap. There is no dedicated
  low-frequency stabilization or general close-fold near-singular quadrature.

The feature stages are a reduced-order engineering embedding of isolated 2-D
deltas (plus separately identified PO or external-pattern terms), not a fully
coupled 3-D solution. The primary `rcs_power` is always the power of the stored
coherent modeled field; phase-unknown power additions live only in the
separately labelled `combination_estimate_power`.

## What each step is for

**1a / 1b — the 2-D solve.** A seam or panel gap is characterised as a 2-D
cross-section and can be reused where the local cross-section, clean host
material/coating stack, frequency, phase origin, polarization convention, and
curvature assumptions remain compatible. Each design needs a matched pair
sharing one mesh and one angle grid:

```
geometries/FRD/<base>.geo     clean    (FRD = faired / smooth)
geometries/OPN/<base>.geo     featured (OPN = feature present)
```

Same file name in both folders — the folder *is* the role. Coupons are
individually cheap, so 1b is worth it when a whole design library makes the
**count** large.

**1c — the deltas.** Joins the solver's per-(polarization, frequency) files into
one per variation, then subtracts: `delta = featured − clean`. You add a
feature's **difference** to the body, never the featured coupon itself — the
smooth skin is already in the body solve, so adding the whole coupon counts it
twice. A delta keeps its modeled phase and is independent of a particular
vehicle file, but its physical reuse remains conditional on those same local
host and embedding assumptions.

**2a / 2b — the body.** The bare body of revolution. The only slow step, run
once per body and frequency set; everything afterwards reuses `Body/body.grim`.
It also writes `Body/body_profile.csv`, which gives every component its surface
normal. A BoR unit parallelizes *internally* across azimuthal modes and solves
all aspects in one call, so 2b's parallel grain is (frequency × polarization) —
the aspect sweep is nearly free either way.

**3a — doors and seams.** Places deltas around each perimeter and writes what
**that door alone** returns. A build **tolerance** is represented by cutting the
door into arcs, one dataset per arc. Because the arcs sum coherently, *which arc
holds which gap* changes the answer — so it also re-draws the arrangement at
random and reports the spread. That width is the uncertainty, not noise.

**3b — a wing, with its root corner estimate.** These are configured in one
step but exported separately. A wing is the same line expansion as a seam
except the coefficient is the
airfoil's *full* amplitude (nothing subtracted — a wing isn't a modification of
the skin), the line is an open root-to-tip span, and it carries its own face
normal. The **corner** is the wing-body double bounce, in neither isolated
solve, and near a right-angle root often the biggest return on the vehicle. The
fold and the body normal are *derived* from where the root sits, so they can't
be set wrong. Its PO magnitude is useful, but its internal phase is not known,
so it is written as a separate power-role estimate and never coherently
interfered with the wing.

**3c — a cavity.** Some features can't be line-expanded: a blind cavity is
compact, resonant and genuinely 3-D — a 2-D slice through it is a trench, not a
hole. So an external 3-D code solves it once as (featured − clean) in the
cavity's own frame, and this step places that pattern at one body coordinate.

**4 — the combination.** Body + every component, re-solving nothing. Separate
placement and summation are numerically equivalent inside the isolated,
single-bounce line-expansion model. The result is an engineering composite,
not an exact coupled 3-D Maxwell solve: body-feature mutual coupling, multiple
scattering, creeping waves, and diffraction beyond the supplied reduced-order
terms are not created by the linear sum.

**5 — rank the designs.** A *different question* from 3a, and not a link in the
chain. 3a puts your whole tolerance band on one door at once; step 5 puts one
design on the whole door at a time and compares designs against each other. Run
it to pick a design, then take that design through 3a and 4.

**0 — calibrate shadowing.** Run before you set `SHADOW = True` anywhere. On a
convex body the occluder must change **nothing**; anything it removes is shadow
acne, not blockage. The step sweeps the bias that lifts a point off the mesh and
tells you what value to trust on *your* mesh.

## Not every component may interfere with every other

Each component file carries a `combine_role` tag saying whether its **phase** can
be trusted against the others:

| role | who | step 4 does |
|---|---|---|
| `coherent` | doors, cavities, a bare wing | sums complex amplitudes |
| `power` | the separately exported PO root-corner estimate | includes it only in a labelled engineering estimate |

The corner's internal double-bounce phase is not tracked, so 3b writes it apart
from the wing. Letting the two interfere would invent a cross term even before
step 4 saw the role tag. Untagged files are refused because field metadata
cannot establish whether an engineering term's phase is trustworthy. Tagged
files must also pass the full current schema: sigma_3d/dBsm, the common vehicle
origin, exp(+jwt), radar-frame physical `F`, and `rcs_power = 4π|F|²`.

In the combined output, `rcs_amp` is the coherent modeled field (body +
coherent components) and `rcs_power` is always the field-consistent physical
conversion `4π|rcs_amp|²`.
`combination_estimate_power` is the separately labelled `MODE` estimate and is
the only place phase-unknown power-role terms appear.

## The frame — one rule, no attitude knobs

Draw everything in the CAD frame: **+y nose, +x right, +z up**. The vehicle is
**level, upright, nose at azimuth 0** — fixed in `Backend/frame.py`, not a knob.
So output angles are vehicle-relative and every component lands where it sits:

| component on | shows up at |
|---|---|
| right side (+x) | azimuth 270 |
| left side (−x) | azimuth 90 |
| top (+z) | elevation +90 |
| nose (+y) | azimuth 0 |

A door's face is radial, so it is always perpendicular to the nose — a side door
looks out 90° off the nose, never along it. A heading or bank angle is a rigid
rotation of the finished result and belongs in a scene-level step, not here.

The BoR solve can't be reoriented (it's axisymmetric about its own +z), so the
steps rotate your CAD coordinates into that frame for you. A body profile is
frame-free: it's `(rho, z)`.

## Four things that are checked, not assumed

**Peak |delta| should grow with the gap.** Step 1c prints it per variation. A
delta near zero means the featured and clean coupons were the same drawing.

**Components must lie on the skin.** Steps 3a and 3c compare distance from the
axis against the body profile and refuse if it's more than a millimetre off,
naming the likely cause (wrong frame, wrong `UNITS`, wrong body). A clean door
reads ~0.1 mm — the perimeter polyline's chord sag, not an error.

**A component you cannot see returns nothing, correctly.** Everything hides
itself when it faces away, so a wing looked at edge-on, a corner with one face
lit, or a cavity whose aperture points away all read −200 dBsm. That is right,
and it is the single most common reason something "doesn't work" — so 3b and 3c
print where each piece peaks and warn if a cavity *never* goes dark.

**Shadowing needs calibrating before you trust it.** Step 0, above. On the demo
mesh: 8.3 dB of pure artefact at a 0.02 mm bias, 0.00 dB at the mesh-scaled
default.

## GHOST — the GUI

```bash
python3 GHOST_GUI.py        # needs PySide6; the numbered steps do not
```

Draw and inspect a `.geo`, then solve it, without writing a script. A separate
way in to the **same** `Backend/` the numbered steps use. Use it to sanity-check
a drawing before committing a long sweep to 1a/1b or 2a/2b; use the numbered
steps for anything you want repeatable.

## Backend

`Backend/` holds the solvers (`rcs_solver.py` 2-D BEM, `bor_solver.py` /
`bor_dispatch.py` BoR-MoM), the I/O (`geometry_io.py`, `grim_io.py`,
`grim_naming.py`, `grim_compat.py`), the feature layer (`line_expand.py`,
`feature_sum.py`, `delta_library.py`, `occluder.py`, `trade_study.py`), the
frame convention (`frame.py`), the component-role rules (`components.py`), the
cluster drivers, and the validation suite in `Backend/tests/`. Every step puts
`Backend/` on its path with a relative lookup, so the whole folder can be moved
or copied anywhere.

```bash
cd Backend/tests && python3 run_feature_gates.py           # feature/wing suite
cd Backend/tests && python3 run_feature_gates.py --fast    # skip the slow sweeps
cd Backend/tests && python3 run_feature_gates.py --all     # + solver batteries
```

Nothing in the pipeline needs the GRIM_Revised_2 repo. `grim_compat.py` is an
optional bridge to its viewer/dataset tool and is never imported by a step; it
finds that repo via `$GRIM_REVISED_PATH` or as a nearby sibling directory.

## Docs

| Doc | What it covers |
|-----|----------------|
| [Docs/GEOMETRY_GUIDE.md](Docs/GEOMETRY_GUIDE.md) | `.geo` format, TYPE table, sign/winding/angle conventions, 2-D & BoR drawing rules |
| [Docs/FEATURE_SUM_GUIDE.md](Docs/FEATURE_SUM_GUIDE.md) | the feature/wing workflow in full — **coupon drawing rules are §4** |
| [Docs/BOR_SOLVER_PLAN.md](Docs/BOR_SOLVER_PLAN.md) | BoR solver scope, formulations, cost model |
| [Docs/BOR_CONVENTIONS.md](Docs/BOR_CONVENTIONS.md) | BoR far-field / polarization conventions and analytic anchors |
| [Docs/SOLVER_OVERVIEW.md](Docs/SOLVER_OVERVIEW.md) | the whole library, module by module, and the `.grim` data-type table |
