#!/usr/bin/env python3
"""
STEP 5 -- RANK THE DESIGNS  (which seam design actually matters?)
================================================================

WHAT IT DOES
  Runs EVERY delta in your library through the SAME perimeter, one at a time,
  so the only thing that differs between runs is the seam itself.  Two numbers
  per design and frequency:

    isolated peak    how loud the seam is on its own.  This RANKS the designs.
    lift over body   the most it raises the total above the bare body, in dB.
                     This is the one that DECIDES anything.

  The second is what people skip and then argue about.  A seam 15 dB below the
  body never shows up no matter how it ranks against the other seams; two
  designs that differ by 6 dB in isolation can be indistinguishable once they
  are on the vehicle.

THIS IS NOT STEP 3a, AND IT IS NOT A LINK IN THE 1-2-3-4 CHAIN
  Step 3a puts your whole tolerance BAND on one door at once, cut into arcs --
  it answers "what does this door, built to this tolerance, do?"  This step
  puts ONE design on the whole door at a time and compares designs against each
  other -- it answers "which design should I even be building?"

  So it is a loop, not a line: rank designs here, pick one, then take it
  through 3a and 4.  Nothing downstream reads this step's output.

INPUTS
  DATASETS_DIR/*.grim          step 1c's committed delta library
  DATASETS_DIR/collection_manifest.json
                               exact delta-library commit marker
  TRACK                        one perimeter .txt -- the COMMON door every
                               design is tested on
  BODY_DIR/body.grim, body_profile.csv     from step 2a/2b

OUTPUTS
  trade_study.csv    one row per (design, frequency): peak VV/HH, the look it
                     peaks at, peak cross-pol, and the lift over the body
  printed table      the same thing, ranked
  trade_study_manifest.json
                     exact input/configuration and CSV commit marker

KNOBS (below)
  DATASETS_DIR, TRACK, STUDY, ASPECTS_DEG, ROLLS_DEG, MODE, UNITS, BODY_DIR

  This step sweeps ASPECT and ROLL rather than the az/el grid in ../grid.py:
  it is a comparison, not a component, so it never has to line up with 3a/3b/3c.

    python3 run.py
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

# ─── KNOBS ──────────────────────────────────────────────────────────────────
DATASETS_DIR = os.path.join("..", "1c_build_deltas", "Deltas")
TRACK = os.path.join("..", "3a_doors", "Tracks", "door_right.txt")

STUDY = ""                    # "" = the folder must hold exactly ONE study
FREQUENCIES_GHZ = [3.0, 6.0]
ASPECTS_DEG = np.arange(30.0, 150.1, 10.0)
ROLLS_DEG = [0.0]             # the door is on ONE side, so roll matters
# Coherent is the physical field-sum model. "hybrid" is available only as a
# separately interpreted engineering estimate when relative body/feature phase
# is intentionally treated as unknown.
MODE = "coherent"
UNITS = "meters"
BODY_DIR = os.path.join("..", "2b_solve_body_hpc", "Body")
                              # HPC production default; use 2a.../Body locally
# ────────────────────────────────────────────────────────────────────────────

from frame import scale_for, to_axis_frame                            # noqa: E402
from delta_library import DeltaLibrary                                # noqa: E402
from feature_sum import (load_body_grim,                             # noqa: E402
                         verify_body_artifact_bundle)
from line_expand import read_perimeter_txt                            # noqa: E402
from trade_study import door_trade_study                              # noqa: E402
from workflow_provenance import (backend_source_fingerprint,          # noqa: E402
                                 runtime_environment_fingerprint,
                                 sha256_file, stable_json_fingerprint,
                                 verify_artifact_manifest,
                                 write_artifact_in_progress,
                                 write_artifact_manifest)

SCALE = scale_for(UNITS)
OUTPUT_SCHEMA = "ghost.workflow.trade-study-provenance.v1"


def _here(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def _trade_provenance_payload(lib, track, body_dir):
    body_manifests = [
        os.path.join(body_dir, name)
        for name in ("cache_manifest.json", "collection_manifest.json")
        if os.path.isfile(os.path.join(body_dir, name))
    ]
    if len(body_manifests) != 1:
        raise ValueError(
            "the body directory must contain exactly one local/HPC commit "
            "manifest."
        )
    return {
        "schema": OUTPUT_SCHEMA,
        "datasets_sha256": {
            os.path.relpath(path, _here(DATASETS_DIR)): sha256_file(path)
            for path in sorted(lib.paths())
        },
        "track_sha256": sha256_file(track),
        "body_manifest_name": os.path.basename(body_manifests[0]),
        "body_manifest_sha256": sha256_file(body_manifests[0]),
        "study": str(STUDY),
        "frequencies_ghz": [float(value) for value in FREQUENCIES_GHZ],
        "aspects_deg": [float(value) for value in ASPECTS_DEG],
        "rolls_deg": [float(value) for value in ROLLS_DEG],
        "mode": str(MODE),
        "units": str(UNITS).strip().lower(),
        "workflow_source_sha256": backend_source_fingerprint(
            BACKEND,
            {"5_rank_designs/run.py": os.path.abspath(__file__)},
        ),
        "runtime_environment_sha256":
            runtime_environment_fingerprint(),
        "expected_outputs": ["trade_study.csv"],
    }


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
            "Datasets/ contains unindexed .grim files. Ranking does not "
            "silently ignore possible stale designs: " + details
        )

    studies = sorted({e.study for e in lib.entries})
    if STUDY:
        if STUDY not in studies:
            raise SystemExit(f"STUDY={STUDY!r} is not in {ds} (have {studies}).")
        lib = DeltaLibrary([e for e in lib.entries if e.study == STUDY],
                           lib.decimals, lib.root, lib.unindexed)
    elif len(studies) > 1:
        raise SystemExit(
            f"Datasets/ holds {len(studies)} studies {studies}.  Ranking across "
            f"two studies compares\n  different seams, not different versions of "
            f"one -- set STUDY to the one you mean.")
    try:
        lib.validate(FREQUENCIES_GHZ)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    track = _here(TRACK)
    if not os.path.exists(track):
        raise SystemExit(f"no {track} -- point the TRACK knob at the perimeter "
                         f"every design should be tested on.")
    body_dir = _here(BODY_DIR)
    body_grim = os.path.join(body_dir, "body.grim")
    prof_csv = os.path.join(body_dir, "body_profile.csv")
    for p in (body_grim, prof_csv):
        if not os.path.exists(p):
            raise SystemExit(f"no {p} -- run 2a_solve_body_local (or 2b) first.")
    try:
        verify_body_artifact_bundle(body_dir)
    except ValueError as exc:
        raise SystemExit(f"uncommitted or changed body bundle: {exc}") from exc

    gen = np.loadtxt(prof_csv, delimiter=",", skiprows=1)
    bodies = load_body_grim(body_grim)
    missing = [f for f in FREQUENCIES_GHZ
               if not any(abs(f - h) < 1e-6 for h in bodies)]
    if missing:
        raise SystemExit(f"the body was solved at {sorted(bodies)} GHz but you "
                         f"asked for {missing} -- re-run step 2 with those "
                         f"frequencies.")
    door = to_axis_frame(read_perimeter_txt(track, scale=SCALE))

    key = lib.keys()[0] if lib.keys() else None
    labels = None
    if key:
        labels = [f"{key} {e.params[key] * 1e3:g} mm" for e in lib.entries]
    print(f"STEP 5   {len(lib)} design(s) on {os.path.basename(track)}, study "
          f"{studies[0] if not STUDY else STUDY}")
    if key:
        print(f"         {key} = {lib.axes()[key]}")
    print(f"         {len(ASPECTS_DEG)} aspects x {len(ROLLS_DEG)} roll(s), "
          f"{list(FREQUENCIES_GHZ)} GHz, mode={MODE}\n")

    try:
        provenance_payload = _trade_provenance_payload(
            lib, track, body_dir
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    provenance_sha256 = stable_json_fingerprint(provenance_payload)
    write_artifact_in_progress(
        HERE,
        OUTPUT_SCHEMA,
        ["trade_study.csv"],
        dict(provenance_payload, run_sha256=provenance_sha256),
        manifest_name="trade_study_manifest.json",
    )
    door_trade_study(lib.paths(), door, gen, list(FREQUENCIES_GHZ),
                     aspects_deg=list(ASPECTS_DEG), rolls_deg=list(ROLLS_DEG),
                     body=bodies, mode=MODE, labels=labels,
                     csv_path=os.path.join(HERE, "trade_study.csv"))

    try:
        verify_body_artifact_bundle(body_dir)
        lib.validate(FREQUENCIES_GHZ)
        current_payload = _trade_provenance_payload(lib, track, body_dir)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "a ranking input changed or became invalid during step 5. The CSV "
            f"was preserved but no complete manifest was written: {exc}"
        ) from exc
    if stable_json_fingerprint(current_payload) != provenance_sha256:
        raise SystemExit(
            "a dataset, track, body, configuration, workflow source, or "
            "numerical runtime changed during step 5. The CSV was preserved "
            "but no complete manifest was written."
        )
    write_artifact_manifest(
        HERE,
        OUTPUT_SCHEMA,
        ["trade_study.csv"],
        dict(provenance_payload, run_sha256=provenance_sha256),
        manifest_name="trade_study_manifest.json",
    )

    print("\n         wrote trade_study.csv")
    print("         committed trade_study_manifest.json")
    print("\nCHECK    the isolated peak should RISE with the gap width.  If the "
          "LIFT column is\n         ~0 dB for every design, none of them matter "
          "on this body at these looks --\n         that is a real answer.  Try "
          "another aspect band, a higher frequency, or\n         accept that the "
          "seam is irrelevant here and stop optimising it.")
    print("NEXT     pick a design, then 3a_doors (tolerance spread) -> 4_combine")


if __name__ == "__main__":
    main()
