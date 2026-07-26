#!/usr/bin/env python3
"""
STEP 3a -- DOORS AND SEAMS
==========================

WHAT IT DOES
  Places your seam deltas around each door perimeter and writes out what THAT
  DOOR ALONE returns, over azimuth x elevation x frequency x polarization.  The
  body is not involved yet -- that is deliberate, so you can rank doors against
  each other before a big body return buries the differences.  Step 4 adds it.

  Each door is cut into as many arcs as there are matching datasets, one
  dataset per arc, ordered so the largest value never sits next to the
  smallest.  That is how a BUILD TOLERANCE is represented: not one nominal gap
  everywhere, but the spread you actually expect, distributed around the
  perimeter.

  WHICH ARC HOLDS WHICH GAP CHANGES THE ANSWER, because the arcs sum
  coherently.  The real per-unit arrangement is unknown, so SPREAD_TRIALS
  re-shuffles the arcs at random and reports the spread of the peak.  Treat
  that spread as the uncertainty on this door, not as noise.  (A uniform build
  -- one dataset -- has no spread by construction.)

  THIS IS NOT THE DESIGN TRADE STUDY.  Here every design in play goes onto ONE
  door at once, as a tolerance.  To rank designs AGAINST each other, one at a
  time on the same perimeter, use 5_rank_designs.

INPUTS
  DATASETS_DIR/<study>_<value><key>.grim
                         the committed delta library from step 1c
  DATASETS_DIR/collection_manifest.json
                         exact delta-library commit marker
  Tracks/<door>.txt                    one perimeter per door, as many as you
                                       like; one straight segment per line:
                                           x1 y1 z1 x2 y2 z2
  BODY_DIR/body_profile.csv            from step 2a or 2b -- gives the doors
                                       their surface normal
  <any>.stl                            only when SHADOW = True

OUTPUTS
  Output/<door>.grim                   the DOOR ALONE:
                                       azimuth x elevation x frequency x pol,
                                       tagged combine_role='coherent'
  Output/provenance_manifest.json      exact committed door-component names,
                                       bytes, inputs, grid, and workflow source
  arrangement_spread.csv               per door: the peak's spread over random
                                       arc arrangements (only when there is
                                       more than one dataset in play)

THE FRAME -- one rule, no attitude knobs
  Draw everything in the CAD frame: +y nose, +x right, +z up.  The vehicle is
  LEVEL, UPRIGHT, NOSE AT AZIMUTH 0 -- fixed in Backend/frame.py, not a knob --
  so the output angles are vehicle-relative and every door lands where it
  actually sits:

      right side (+x) -> azimuth 270      left side (-x) -> azimuth 90
      top       (+z) -> elevation +90     nose      (+y) -> azimuth 0

  A door's face is radial, so it is always perpendicular to the nose: a side
  door looks out 90 deg off the nose, never along it.  A heading or bank angle
  is a rigid rotation of the finished result and belongs in a scene-level step,
  not here.

KNOBS (below)
  TOLERANCES -- one entry per variable in your delta-library file names,
                either (min, max) or an explicit list
  STUDY, SHADOW, SHADOW_BIAS_M, SPREAD_TRIALS, UNITS, SKIN_TOL_M,
  SKIN_PHASE_TOL_DEG,
  DATASETS_DIR, BODY_DIR

  The output grid (frequencies, azimuths, elevations, polarizations) is NOT a
  knob here -- it is shared with 3b and 3c and lives in ../grid.py.

TWO THINGS THAT ARE CHECKED, NOT ASSUMED
  * DOORS MUST LIE ON THE SKIN.  A door floating off the surface still returns
    a plausible-looking answer, so the distance from the axis is compared
    against the body profile and refused if it is more than SKIN_TOL_M out.
    A clean door reads ~0.1 mm -- that is the perimeter polyline's chord sag,
    not an error.
  * SHADOWING DOES NOTHING ON A CONVEX BODY.  Doors already hide themselves
    when they face away; the STL only adds blocking by OTHER structure.  If you
    turn SHADOW on and see a loss on a convex hull, that is shadow acne -- see
    Docs/FEATURE_SUM_GUIDE.md on the bias calibration.

NEXT  3b_add_wing / 3c_add_cavity if you have them, then 4_combine

    python3 run.py
"""

import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

# ─── KNOBS ──────────────────────────────────────────────────────────────────
# one entry per variable in your delta-library file names.  Either
#     (min, max)        every dataset whose value falls in the band
#     [v1, v2, ...]     exactly these values
TOLERANCES = {
    "gap": (0.006, 0.017),
}

STUDY = ""                    # "" = DATASETS_DIR must hold exactly ONE study;
                              # else name it, e.g. "SEAL-00-01"
SHADOW = False                # block doors the body geometrically hides
SHADOW_BIAS_M = None          # REQUIRED when SHADOW=True: calibrated in step 0
UNITS = "meters"              # units the .txt and .stl are drawn in
SKIN_TOL_M = 1e-3             # how far off the skin a door may sit
SKIN_PHASE_TOL_DEG = 15.0     # max two-way placement-phase error at highest freq
SPREAD_TRIALS = 200           # random arc arrangements; 0 = skip the spread
SPREAD_FREQ_GHZ = 6.0         # the spread is measured at one frequency

DATASETS_DIR = os.path.join("..", "1c_build_deltas", "Deltas")
                              # use step 1c's committed library directly
BODY_DIR = os.path.join("..", "2b_solve_body_hpc", "Body")
                              # HPC production default; use 2a.../Body locally
# ────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(HERE))                # for ../grid.py
from grid import (FREQUENCIES_GHZ, AZIMUTHS_DEG,                      # noqa: E402
                  ELEVATIONS_DEG, POLARIZATIONS)
from frame import (AXIS_AZ_DEG, AXIS_EL_DEG, ROLL_DEG,                # noqa: E402
                   scale_for, to_axis_frame)
from components import keep_pols, tag_component                       # noqa: E402
from delta_library import DeltaLibrary, Range, tolerance_placements    # noqa: E402
from feature_sum import (export_radar_grim, sum_features,             # noqa: E402
                         directions_from_aspect_roll, _load_grim)
from feature_sum import verify_body_artifact_bundle                  # noqa: E402
from line_expand import (C0, read_perimeter_txt, dbsm,                 # noqa: E402
                         perimeter_surface_deviation)
from workflow_provenance import (backend_source_fingerprint,          # noqa: E402
                                 runtime_environment_fingerprint,
                                 sha256_file, stable_json_fingerprint,
                                 verify_artifact_manifest,
                                 write_artifact_in_progress,
                                 write_artifact_manifest)

SCALE = scale_for(UNITS)
OUTPUT_SCHEMA = "ghost.workflow.doors-output-provenance.v1"


def _here(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def _refuse_unexpected_grims(folder, expected_names):
    existing = {
        os.path.basename(path)
        for path in glob.glob(os.path.join(folder, "*.grim"))
    }
    stale = sorted(existing - set(expected_names))
    if stale:
        raise SystemExit(
            "Output/ contains unexpected stale door component(s):\n  "
            + "\n  ".join(stale)
            + "\nThey were preserved. Move them out of Output/ before "
              "building this exact track set."
        )


def _source_fingerprint():
    root = os.path.dirname(HERE)
    return backend_source_fingerprint(
        BACKEND,
        {
            "3a_doors/run.py": os.path.abspath(__file__),
            "grid.py": os.path.join(root, "grid.py"),
        },
    )


def _doors_provenance_payload(ds, tracks, entries, expected_names):
    dataset_paths = sorted(glob.glob(os.path.join(ds, "*.grim")))
    body_profile = os.path.join(_here(BODY_DIR), "body_profile.csv")
    stls = (
        sorted(glob.glob(os.path.join(HERE, "*.stl")))
        if SHADOW else []
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "datasets_sha256": {
            os.path.basename(path): sha256_file(path)
            for path in dataset_paths
        },
        "selected_datasets": sorted(
            os.path.basename(entry.path) for entry in entries
        ),
        "tracks_sha256": {
            os.path.basename(path): sha256_file(path) for path in tracks
        },
        "body_profile_sha256": sha256_file(body_profile),
        "shadow_mesh_sha256": {
            os.path.basename(path): sha256_file(path) for path in stls
        },
        "study": str(STUDY),
        "tolerances": TOLERANCES,
        "shadow": bool(SHADOW),
        "shadow_bias_m": (
            None if SHADOW_BIAS_M is None else float(SHADOW_BIAS_M)
        ),
        "units": str(UNITS).strip().lower(),
        "skin_tol_m": float(SKIN_TOL_M),
        "skin_phase_tol_deg": float(SKIN_PHASE_TOL_DEG),
        "spread_trials": int(SPREAD_TRIALS),
        "spread_frequency_ghz": float(SPREAD_FREQ_GHZ),
        "frequencies_ghz": [float(value) for value in FREQUENCIES_GHZ],
        "azimuths_deg": [float(value) for value in AZIMUTHS_DEG],
        "elevations_deg": [float(value) for value in ELEVATIONS_DEG],
        "polarizations": [str(value) for value in POLARIZATIONS],
        "axis_az_deg": float(AXIS_AZ_DEG),
        "axis_el_deg": float(AXIS_EL_DEG),
        "roll_deg": float(ROLL_DEG),
        "workflow_source_sha256": _source_fingerprint(),
        "runtime_environment_sha256":
            runtime_environment_fingerprint(),
        "expected_outputs": sorted(expected_names),
    }


def profile():
    body_dir = _here(BODY_DIR)
    p = os.path.join(body_dir, "body_profile.csv")
    if not os.path.exists(p):
        raise SystemExit(f"no {p} -- run 2a_solve_body_local (or 2b) first, "
                         f"or point the BODY_DIR knob at the body you want.")
    try:
        verify_body_artifact_bundle(body_dir)
    except ValueError as exc:
        raise SystemExit(f"uncommitted or changed body bundle: {exc}") from exc
    try:
        rows = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"invalid body profile {p}: expected strict numeric rho_m,z_m rows"
        ) from exc
    if rows.ndim != 2 or rows.shape[1] != 2 or len(rows) < 2 \
            or not np.all(np.isfinite(rows)):
        raise SystemExit(
            f"invalid body profile {p}: require at least two finite rho_m,z_m rows"
        )
    return rows


def pick_datasets(lib):
    """Apply TOLERANCES -> the datasets in play, one per arc."""
    keys = lib.keys()
    for k in keys:
        if k not in TOLERANCES:
            raise SystemExit(f"the file names carry '{k}' but TOLERANCES has no "
                             f"entry for it -- add one (you have "
                             f"{list(TOLERANCES)}).")
    # ONE study per spread.  Two studies in one folder is not a duplicate -- the
    # same gap from a different study is a different design -- so nothing
    # upstream objects, and the arcs would silently be drawn from both.  Catch
    # it here.
    studies = sorted({e.study for e in lib.entries})
    if STUDY:
        if STUDY not in studies:
            raise SystemExit(f"STUDY={STUDY!r} is not in {DATASETS_DIR}/ "
                             f"(have {studies}).")
        lib = DeltaLibrary([e for e in lib.entries if e.study == STUDY],
                           lib.decimals, lib.root, lib.unindexed)
    elif len(studies) > 1:
        raise SystemExit(
            f"{DATASETS_DIR}/ holds {len(studies)} studies {studies}.  A "
            f"tolerance spread must come from ONE study, or the arcs mix "
            f"designs and the answer changes silently.\n  Set "
            f"STUDY = \"one of those\", or split the folder.")
    print(f"Study     {studies[0] if not STUDY else STUDY}")
    sub = lib
    for key, spec in TOLERANCES.items():
        if key not in keys:
            raise SystemExit(f"TOLERANCES has '{key}' but the file names carry "
                             f"{keys}.")
        if isinstance(spec, tuple) and len(spec) == 2:
            want = Range(*spec)
        else:
            want = [float(v) for v in np.atleast_1d(spec)]
            have = lib.axes()[key]
            tol = 0.5 * 10.0 ** -int(lib.decimals.get(key, 6))
            gone = [v for v in want if not any(abs(v - h) <= tol for h in have)]
            if gone:
                print(f"  NOTE  {key}: no dataset for {gone} -- have {have}")
        try:
            sub = sub.select(**{key: want})
        except ValueError as exc:
            raise SystemExit(f"  {exc}")
    return sub.entries


def check_on_skin(per, gen, name):
    """Require every sampled perimeter chord to lie on the actual BoR skin."""
    off = perimeter_surface_deviation(per, gen, samples_per_segment=33)
    fmax_hz = float(np.max(FREQUENCIES_GHZ)) * 1e9
    lam_min = C0 / fmax_hz
    phase_limit_m = float(SKIN_PHASE_TOL_DEG) * lam_min / 720.0
    limit_m = min(float(SKIN_TOL_M), phase_limit_m)
    phase_deg = 720.0 * off / lam_min
    if off > limit_m:
        raise SystemExit(
            f"{name}: door is up to {off*1e3:.1f} mm off the body skin "
            f"along its sampled chords ({phase_deg:.1f} deg worst-case two-way "
            f"phase at {float(np.max(FREQUENCIES_GHZ)):g} GHz; allowed "
            f"{limit_m*1e3:.3f} mm / {SKIN_PHASE_TOL_DEG:g} deg).\n"
            f"  Check the CAD frame (+y nose, +x right, +z up), the UNITS knob, "
            f"BODY_DIR, and perimeter chord density.")
    return off


def arrangement_spread(per, entries, order_key, gen, occ, dirs):
    """Re-draw which arc holds which dataset and report the spread of the peak.

    The arcs sum COHERENTLY, so the arrangement is not bookkeeping -- it changes
    the answer.  Nobody knows the real per-unit arrangement, so the honest
    output is a RANGE, with the ordered case (widest never beside tightest)
    located inside it rather than presented as the answer.
    """
    rng = np.random.default_rng(20260725)  # fixed: a re-run reproduces the range
    peaks = []
    for _ in range(int(SPREAD_TRIALS)):
        pl = tolerance_placements(per, entries, order="random", rng=rng,
                                  kind="delta")
        r = sum_features(None, pl, dirs, float(SPREAD_FREQ_GHZ), generatrix=gen,
                         occluder=occ, mode="coherent")
        peaks.append(float(np.max(r["dbsm_vv"])))
    pl0 = tolerance_placements(per, entries, order_by=order_key, kind="delta")
    r0 = sum_features(None, pl0, dirs, float(SPREAD_FREQ_GHZ), generatrix=gen,
                      occluder=occ, mode="coherent")
    return np.asarray(peaks, float), float(np.max(r0["dbsm_vv"]))


def main():
    ds = _here(DATASETS_DIR)
    if not os.path.isdir(ds) or not glob.glob(os.path.join(ds, "*.grim")):
        raise SystemExit(
            f"no *.grim in {ds} -- run 1c_build_deltas and point "
            "DATASETS_DIR at its committed Deltas/ directory."
        )
    try:
        verify_artifact_manifest(
            ds,
            [
                os.path.basename(path)
                for path in glob.glob(os.path.join(ds, "*.grim"))
            ],
            exact_grim_set=True,
            expected_schema="ghost.workflow.delta-library.v1",
        )
    except ValueError as exc:
        raise SystemExit(
            f"uncommitted or changed delta library {ds}: {exc}"
        ) from exc
    lib = DeltaLibrary.from_dir(ds)
    if lib.unindexed:
        details = "; ".join(
            f"{os.path.basename(path)}: {why}"
            for path, why in lib.unindexed[:8]
        )
        raise SystemExit(
            "The delta library contains unindexed .grim files. Production "
            "generation "
            "does not silently ignore possible stale designs: " + details
        )
    print("Datasets  " + ", ".join(f"{k}={lib.axes()[k]}" for k in lib.keys()))
    entries = pick_datasets(lib)
    order_key = next(iter(TOLERANCES))
    print(f"In play   {len(entries)} dataset(s) = {len(entries)} arc(s) per door")

    body_profile_path = os.path.join(
        _here(BODY_DIR), "body_profile.csv"
    )
    body_profile_sha256_used = sha256_file(body_profile_path)
    gen = profile()
    if sha256_file(body_profile_path) != body_profile_sha256_used:
        raise SystemExit(
            "the body profile changed while step 3a was reading it; no "
            "component generation was started."
        )
    occ = None
    shadow_mesh_sha256_used = None
    if SHADOW:
        if SHADOW_BIAS_M is None:
            raise SystemExit(
                "SHADOW=True requires an explicit SHADOW_BIAS_M from "
                "0_calibrate_shadowing. The mesh-scaled default is useful for "
                "that calibration sweep but can hide real nearby blockers.")
        stls = sorted(glob.glob(os.path.join(HERE, "*.stl")))
        if len(stls) != 1:
            raise SystemExit(f"SHADOW=True needs exactly one .stl here, found "
                             f"{[os.path.basename(s) for s in stls]}.")
        from occluder import Occluder, read_stl                       # noqa: PLC0415
        shadow_mesh_sha256_used = sha256_file(stls[0])
        occ = Occluder(to_axis_frame(read_stl(stls[0])), scale=SCALE,
                       bias=float(SHADOW_BIAS_M))
        if sha256_file(stls[0]) != shadow_mesh_sha256_used:
            raise SystemExit(
                "the shadow mesh changed while step 3a was reading it; no "
                "component generation was started."
            )
        print(f"Shadowing {os.path.basename(stls[0])} ({len(occ.tris)} facets, "
              f"bias {occ.bias*1e3:g} mm)")

    tracks = sorted(glob.glob(os.path.join(HERE, "Tracks", "*.txt")))
    if not tracks:
        raise SystemExit("no Tracks/*.txt -- put at least one door perimeter "
                         "there.")
    output_dir = os.path.join(HERE, "Output")
    os.makedirs(output_dir, exist_ok=True)
    expected_names = sorted(
        os.path.splitext(os.path.basename(track))[0] + ".grim"
        for track in tracks
    )
    if len(set(expected_names)) != len(expected_names):
        raise SystemExit(
            "door track names collide after conversion to component names."
        )
    _refuse_unexpected_grims(output_dir, expected_names)
    provenance_payload = _doors_provenance_payload(
        ds, tracks, entries, expected_names
    )
    if (
        provenance_payload["body_profile_sha256"]
        != body_profile_sha256_used
        or (
            SHADOW
            and provenance_payload["shadow_mesh_sha256"].get(
                os.path.basename(stls[0])
            ) != shadow_mesh_sha256_used
        )
    ):
        raise SystemExit(
            "the body profile or shadow mesh changed before step 3a could "
            "freeze its provenance; no component generation was started."
        )
    provenance_sha256 = stable_json_fingerprint(provenance_payload)
    write_artifact_in_progress(
        output_dir,
        OUTPUT_SCHEMA,
        expected_names,
        dict(provenance_payload, run_sha256=provenance_sha256),
        manifest_name="provenance_manifest.json",
    )
    spread_rows = [("door", "n_arcs", "trials", "ordered_dBsm", "min_dBsm",
                    "median_dBsm", "max_dBsm", "spread_dB")]

    for track in tracks:
        name = os.path.splitext(os.path.basename(track))[0]
        per = to_axis_frame(read_perimeter_txt(track, scale=SCALE))
        off = check_on_skin(per, gen, os.path.basename(track))
        try:
            # kind="delta" selects the strict differential-field loader.  The
            # directory name is not trusted as a physical declaration: every
            # file must carry the current delta-domain, normalization, phase,
            # support, and complete TM/TE metadata.
            pl = tolerance_placements(per, entries, order_by=order_key,
                                      kind="delta")
        except ValueError as exc:
            raise SystemExit(f"{os.path.basename(track)}: {exc}")
        out = export_radar_grim(
            os.path.join(output_dir, name),
            bor_result=None,                # the DOOR ALONE; step 4 adds the body
            placements=pl, generatrix=gen, occluder=occ,
            frequencies_ghz=FREQUENCIES_GHZ,
            azimuths_deg=AZIMUTHS_DEG, elevations_deg=ELEVATIONS_DEG,
            axis_az_deg=AXIS_AZ_DEG, axis_el_deg=AXIS_EL_DEG,   # fixed by frame.py
            roll_deg=ROLL_DEG,
            history=f"step 3 {name} tolerances={TOLERANCES} shadow={bool(occ)}")
        keep_pols(out, POLARIZATIONS)
        # doors are line-expanded and share one phase reference, so their
        # amplitudes may be summed against every other coherent component
        tag_component(out, "coherent",
                      note="line expansion is linear; phase is trustworthy")

        g = _load_grim(out)
        pols = [str(x) for x in np.asarray(g["polarizations"]).ravel()]
        pk = {p: dbsm(np.max(np.asarray(g["rcs_power"], float)[:, :, :, i]))
              for i, p in enumerate(pols)}
        print(f"  {name:<20} {len(per)} segs / {len(pl)} arcs, {off*1e3:.2f} mm "
              f"off skin  peak "
              + "  ".join(f"{p} {v:+.1f}" for p, v in pk.items()) + " dBsm")

        # --- how much does the ARRANGEMENT of the arcs matter? ---------------
        if SPREAD_TRIALS and len(entries) > 1:
            dirs, _a, _r = directions_from_aspect_roll(
                np.arange(30.0, 150.1, 15.0), np.arange(0.0, 359.1, 15.0))
            pk_rand, pk_ord = arrangement_spread(per, entries, order_key, gen,
                                                 occ, dirs)
            lo, md, hi = (float(np.min(pk_rand)), float(np.median(pk_rand)),
                          float(np.max(pk_rand)))
            spread_rows.append((name, str(len(entries)), str(SPREAD_TRIALS),
                                f"{pk_ord:.2f}", f"{lo:.2f}", f"{md:.2f}",
                                f"{hi:.2f}", f"{hi - lo:.2f}"))
            print(f"  {'':<20} arrangement spread at {SPREAD_FREQ_GHZ:g} GHz: "
                  f"{lo:+.1f} .. {hi:+.1f} dBsm ({hi - lo:.1f} dB wide), "
                  f"ordered {pk_ord:+.1f}")
        elif SPREAD_TRIALS:
            print(f"  {'':<20} one dataset in play -- a uniform build has no "
                  f"arrangement spread")

    if len(spread_rows) > 1:
        with open(os.path.join(HERE, "arrangement_spread.csv"), "w",
                  newline="") as fh:
            csv.writer(fh).writerows(spread_rows)
        print("\n         wrote arrangement_spread.csv -- that width is the "
              "UNCERTAINTY on this\n         door, not noise.  Quoting the "
              "ordered number alone is spuriously precise.")

    _refuse_unexpected_grims(output_dir, expected_names)
    actual_names = {
        os.path.basename(path)
        for path in glob.glob(os.path.join(output_dir, "*.grim"))
    }
    if actual_names != set(expected_names):
        raise SystemExit(
            "Output/ does not contain the exact expected door component set "
            f"(expected {expected_names}, got {sorted(actual_names)}). "
            "Files were preserved but no complete manifest was written."
        )
    current_tracks = sorted(
        glob.glob(os.path.join(HERE, "Tracks", "*.txt"))
    )
    try:
        current_payload = _doors_provenance_payload(
            ds, current_tracks, entries, expected_names
        )
    except Exception as exc:
        raise SystemExit(
            "door inputs became unreadable during step 3a. Outputs were "
            "preserved but no complete provenance manifest was written."
        ) from exc
    if stable_json_fingerprint(current_payload) != provenance_sha256:
        raise SystemExit(
            "door datasets, tracks, body profile, mesh, grid, configuration, "
            "runtime, or workflow source changed during step 3a. Outputs were "
            "preserved but no complete provenance manifest was written."
        )
    write_artifact_manifest(
        output_dir,
        OUTPUT_SCHEMA,
        expected_names,
        dict(provenance_payload, run_sha256=provenance_sha256),
        manifest_name="provenance_manifest.json",
    )
    print("         wrote Output/provenance_manifest.json "
          "(exact component inventory and hashes)")

    print("\nNEXT    3b_add_wing / 3c_add_cavity if you have them, then 4_combine")


if __name__ == "__main__":
    main()
