#!/usr/bin/env python3
"""
STEP 1c -- BUILD THE DELTAS  (join, then featured - clean)
==========================================================

WHAT IT DOES
  Two things, in this order, on whatever step 1a or 1b produced.

  1. JOIN.  The solver writes ONE FILE PER (polarization, frequency), with
     those LEADING the name:

         TM_3.000GHz_SEAL-00-01_0.010gap_OPN.grim  \\
         TE_3.000GHz_SEAL-00-01_0.010gap_OPN.grim   >--  SEAL-00-01_0.010gap_OPN
         TM_6.000GHz_SEAL-00-01_0.010gap_OPN.grim   /
         TE_6.000GHz_SEAL-00-01_0.010gap_OPN.grim  /

     That is not tidying.  The subtraction below requires every file handed to
     it to share ONE (angle, frequency) grid -- it merges polarizations, not
     frequencies -- so the split files CANNOT be subtracted as they come.
     TM/TE and HH/VV are both understood (TM -> HH, TE -> VV); the solver's own
     primaries are kept in the file.

  2. SUBTRACT.  Coherently, complex amplitude by complex amplitude:

         delta = featured (OPN) - clean (FRD)

     This is the heart of the method.  You add the seam's DIFFERENCE to the
     body, never the featured coupon itself -- the smooth skin is already in
     the body solve, so adding the whole coupon would count it twice.  Because
     it is a difference of two solves that share a phase reference, the result
     keeps its phase and can be placed anywhere on any vehicle.  A delta is
     REUSABLE and VEHICLE-INDEPENDENT: solve it once, use it forever.

     Pairing is automatic from the file names.  An FRD matches an OPN when the
     study is the same and every FRD parameter has the same value in the OPN.
     The OPN may carry additional feature-only variables, so one clean baseline
     can serve many featured cases.  The most-specific compatible FRD wins;
     equal-specificity ambiguity is refused. Anything unmatched is LISTED.

       SEAL-00-00_0.050bmag_FRD
         -> SEAL-00-00_0.050bmag_0.010het_0.020crv_OPN
         -> SEAL-00-00_0.050bmag_0.015het_0.025crv_OPN

     produces two deltas named from the full OPN parameter sets.

INPUTS
  INPUTS knob (below), default ../1b_solve_2d_hpc/results
                        <POL>_<FREQ>GHz_<base>_<ROLE>.grim, any number of them,
                        any number of variations mixed together

OUTPUTS
  joined/<base>_<ROLE>.grim   the intermediate, one per (variation, role);
                              kept because it is what you open when a delta
                              looks wrong
  Deltas/<base>.grim          THE PRODUCT -- one delta per variation, tagged
                              rcs_domain='delta', named so its parameters are
                              in the file name.
  Deltas/collection_manifest.json
                              exact committed delta-library inventory
  delta_summary.csv           peak |delta| per variation and polarization
  build_manifest.json         exact hashes of every solver input, builder
                              source state, joined file, delta, and summary.
                              Unexpected old .grim outputs are reported and
                              preserved rather than deleted.

KNOBS (below)
  INPUTS

NEXT  point 3a/5 DATASETS_DIR at this committed Deltas/ directory, then run
      2a_solve_body_local or 2b_solve_body_hpc (once per body), then
      3a_doors or 5_rank_designs.

WATCH OUT  clean and featured must come from the SAME angle/frequency grid and
  the same background -- same plank, same coating.  A grid mismatch raises
  here; a mismatched background is SILENT, so keep a variation's two coupons
  together and re-solve them together.

    python3 run.py
"""

import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

# ─── KNOBS ──────────────────────────────────────────────────────────────────
# where the solver's per-(pol, frequency) files are. HPC is the production
# default; point this at step 1a's results/ for a local solve.
INPUTS = os.path.join("..", "1b_solve_2d_hpc", "results")
# ────────────────────────────────────────────────────────────────────────────

from feature_sum import make_delta_grim, _load_grim                   # noqa: E402
from grim_naming import (group_solver_files, join_grims,              # noqa: E402
                         pair_variants, parse_variation)
from workflow_provenance import (backend_source_paths,                # noqa: E402
                                 runtime_environment_fingerprint,
                                 verify_artifact_manifest,
                                 write_artifact_in_progress,
                                 write_artifact_manifest)

BUILD_SCHEMA = "ghost.workflow.delta-build.v1"
DELTA_LIBRARY_SCHEMA = "ghost.workflow.delta-library.v1"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha256(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_fingerprint():
    root = os.path.dirname(HERE)
    paths = [os.path.abspath(__file__)]
    paths.extend(backend_source_paths(BACKEND))
    return _canonical_sha256([
        {"path": os.path.relpath(p, root), "sha256": _sha256_file(p)}
        for p in sorted(set(paths))
    ])


def _write_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def _refuse_unexpected_grims(folder, expected_names, label):
    existing = {
        os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.grim"))
    }
    stale = sorted(existing - set(expected_names))
    if stale:
        raise SystemExit(
            f"{label} contains unexpected stale .grim output(s):\n  "
            + "\n  ".join(stale)
            + "\nThey were not deleted. Move them out of this workflow "
              "directory and rerun.")


def _verified_input_state(src, paths):
    """Verify the upstream commit marker and snapshot every consumed byte."""

    try:
        manifest = verify_artifact_manifest(
            src,
            [os.path.basename(path) for path in paths],
            exact_grim_set=True,
        )
    except ValueError as exc:
        raise SystemExit(
            "solver INPUTS are not one complete byte-attested local/HPC "
            f"bundle: {exc}"
        ) from exc
    if manifest.get("schema") not in {
        "ghost.workflow.2d-local-cache.v1",
        "ghost.hpc.2d-collection.v1",
    }:
        raise SystemExit(
            "solver INPUTS have the wrong artifact role/schema "
            f"{manifest.get('schema')!r}; expected a completed local or HPC "
            "2-D solver bundle."
        )
    manifests = [
        os.path.join(src, name)
        for name in ("cache_manifest.json", "collection_manifest.json")
        if os.path.isfile(os.path.join(src, name))
    ]
    if len(manifests) != 1:
        raise SystemExit(
            "solver INPUTS do not have exactly one upstream commit marker."
        )
    input_sha256 = {
        os.path.basename(path): _sha256_file(path) for path in paths
    }
    if len(input_sha256) != len(paths):
        raise SystemExit(
            "INPUTS contains duplicate .grim basenames, so provenance would "
            "be ambiguous. Put one solver library in one directory."
        )
    return {
        "input_sha256": input_sha256,
        "input_manifest_name": os.path.basename(manifests[0]),
        "input_manifest_sha256": _sha256_file(manifests[0]),
    }


def _signature_payload(src, input_state, expected_joined, expected_deltas):
    return {
        "schema": BUILD_SCHEMA,
        "input_root": os.path.relpath(src, HERE),
        "input_sha256": dict(input_state["input_sha256"]),
        "input_manifest_name": input_state["input_manifest_name"],
        "input_manifest_sha256": input_state["input_manifest_sha256"],
        "builder_source_sha256": _source_fingerprint(),
        "runtime_environment_sha256": runtime_environment_fingerprint(),
        "expected_joined_outputs": sorted(expected_joined),
        "expected_delta_outputs": sorted(expected_deltas),
    }


def _require_unchanged_signature(initial_payload, current_payload):
    if _canonical_sha256(current_payload) != _canonical_sha256(initial_payload):
        raise SystemExit(
            "solver input bytes/manifest, builder source, or runtime changed "
            "while step 1c was running. Joined/delta outputs were preserved "
            "but no complete build manifest was written; rerun from one "
            "stable input/source state."
        )


def peak_amp(path):
    g = _load_grim(path)
    a = np.abs(g["rcs_amp_real"] + 1j * g["rcs_amp_imag"])
    pols = [str(p) for p in np.asarray(g["polarizations"]).ravel()]
    return {pols[j]: float(np.max(a[..., j])) for j in range(len(pols))}


def main():
    src = INPUTS if os.path.isabs(INPUTS) else os.path.join(HERE, INPUTS)
    src = os.path.normpath(src)
    paths = sorted(glob.glob(os.path.join(src, "*.grim")))
    if not paths:
        raise SystemExit(
            f"no *.grim in {src}\n"
            f"  Run 1a_solve_2d_local (or 1b_solve_2d_hpc + its collect.py) "
            f"first,\n  or point the INPUTS knob at wherever your solver files are.")
    input_state = _verified_input_state(src, paths)
    print(f"STEP 1c  reading {len(paths)} file(s) from "
          f"{os.path.relpath(src, HERE)}")

    # ---- 1. join ------------------------------------------------------------
    groups, unparsed = group_solver_files(paths)
    for p, why in unparsed:
        print(f"         NOT INDEXED  {os.path.basename(p)}: {why}")
    if unparsed:
        raise SystemExit(
            "refusing to build a partial delta library while input .grim "
            "files are not indexed. Move unrelated files out of INPUTS or "
            "correct their <POL>_<FREQ>GHz_<variation>.grim names.")
    if not groups:
        raise SystemExit("nothing parsed as <POL>_<FREQ>GHz_<variation>.grim.")

    joined_dir = os.path.join(HERE, "joined")
    expected_joined = sorted(f"{variation}.grim" for variation in groups)
    virtual_joined = [os.path.join(joined_dir, name)
                      for name in expected_joined]
    pairs, unmatched = pair_variants(virtual_joined)
    if not pairs:
        raise SystemExit(
            "nothing to subtract: no OPN has a compatible FRD baseline.\n"
            "  A compatible FRD has the same study and a parameter set that "
            "is a subset of the OPN.")
    expected_deltas = sorted(p["delta_name"] for p in pairs)
    out_dir = os.path.join(HERE, "Deltas")
    os.makedirs(joined_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    _refuse_unexpected_grims(joined_dir, expected_joined, "joined/")
    _refuse_unexpected_grims(out_dir, expected_deltas, "Deltas/")

    signature_payload = _signature_payload(
        src, input_state, expected_joined, expected_deltas
    )
    run_sha256 = _canonical_sha256(signature_payload)
    in_progress = dict(signature_payload)
    in_progress.update(run_sha256=run_sha256, status="in_progress")
    _write_json_atomic(
        os.path.join(HERE, "build_manifest.json"), in_progress
    )
    write_artifact_in_progress(
        out_dir,
        DELTA_LIBRARY_SCHEMA,
        expected_deltas,
        {
            "build_run_sha256": run_sha256,
            "input_manifest_sha256":
                input_state["input_manifest_sha256"],
            "builder_source_sha256":
                signature_payload["builder_source_sha256"],
            "runtime_environment_sha256":
                signature_payload["runtime_environment_sha256"],
        },
    )

    print(f"\nJOIN     {len(paths)} file(s) -> {len(groups)} variation(s)")
    roles = {}
    for variation in sorted(groups):
        recs = groups[variation]
        pols = sorted({r["pol_canon"] for r in recs})
        freqs = sorted({r["freq_ghz"] for r in recs})
        join_grims([r["path"] for r in recs],
                   os.path.join(joined_dir, f"{variation}.grim"),
                   history=(f"step 1c joined {len(recs)} solver file(s); "
                            f"build_input_sha256={run_sha256}"))
        _base, role = parse_variation(variation)
        roles[role] = roles.get(role, 0) + 1
        print(f"         {len(recs):>3} file(s)  {' '.join(pols):<8} "
              f"{' '.join(f'{f:g}' for f in freqs):<12} {variation}")
    print(f"         roles {roles}   (OPN = featured, FRD = clean)")

    # ---- 2. subtract --------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    pairs, unmatched = pair_variants(
        sorted(glob.glob(os.path.join(joined_dir, "*.grim"))))
    if not pairs:
        raise SystemExit(
            "nothing to subtract: no OPN has a compatible FRD baseline.\n"
            "  A compatible FRD has the same study and a parameter set that "
            "is a subset of the OPN.")
    print(f"\nSUBTRACT {len(pairs)} variation(s), OPN - FRD")
    for u in unmatched:
        print(f"         UNMATCHED  {os.path.basename(str(u['path'])):<44} "
              f"{u['reason']}")

    rows = [("variation", "peak_abs_delta_HH", "peak_abs_delta_VV")]
    for p in pairs:
        out = os.path.join(out_dir, p["delta_name"])
        make_delta_grim(p["clean"], p["featured"], out,
                        history=(f"step 1c featured - clean, {p['base']} "
                                 f"using baseline {p['clean_base']}; "
                                 f"build_input_sha256={run_sha256}"))
        pk = peak_amp(out)
        rows.append((p["base"], f"{pk.get('HH', float('nan')):.6e}",
                     f"{pk.get('VV', float('nan')):.6e}"))
        print(f"         {p['delta_name']:<36} peak |delta|  "
              f"HH(TM) {pk.get('HH', float('nan')):.3e}   "
              f"VV(TE) {pk.get('VV', float('nan')):.3e}")

    with open(os.path.join(HERE, "delta_summary.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print("         wrote delta_summary.csv")

    output_sha256 = {}
    for name in expected_joined:
        rel = os.path.join("joined", name)
        output_sha256[rel] = _sha256_file(os.path.join(HERE, rel))
    for name in expected_deltas:
        rel = os.path.join("Deltas", name)
        output_sha256[rel] = _sha256_file(os.path.join(HERE, rel))
    output_sha256["delta_summary.csv"] = _sha256_file(
        os.path.join(HERE, "delta_summary.csv"))
    _refuse_unexpected_grims(joined_dir, expected_joined, "joined/")
    _refuse_unexpected_grims(out_dir, expected_deltas, "Deltas/")
    current_paths = sorted(glob.glob(os.path.join(src, "*.grim")))
    try:
        current_state = _verified_input_state(src, current_paths)
        current_payload = _signature_payload(
            src, current_state, expected_joined, expected_deltas
        )
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            "solver inputs became unreadable while step 1c was running. "
            "Outputs were preserved but no complete manifest was written."
        ) from exc
    _require_unchanged_signature(signature_payload, current_payload)
    delta_manifest = write_artifact_manifest(
        out_dir,
        DELTA_LIBRARY_SCHEMA,
        expected_deltas,
        {
            "build_run_sha256": run_sha256,
            "input_manifest_sha256":
                input_state["input_manifest_sha256"],
            "builder_source_sha256":
                signature_payload["builder_source_sha256"],
            "runtime_environment_sha256":
                signature_payload["runtime_environment_sha256"],
        },
    )
    output_sha256["Deltas/collection_manifest.json"] = _sha256_file(
        delta_manifest
    )
    manifest = dict(signature_payload)
    manifest.update(
        run_sha256=run_sha256,
        expected_outputs=sorted(output_sha256),
        output_sha256=output_sha256,
        status="complete")
    _write_json_atomic(os.path.join(HERE, "build_manifest.json"), manifest)
    print("         wrote build_manifest.json (exact input/output fingerprints)")

    print("\nCHECK    peak |delta| should GROW with the gap width.  A delta that "
          "is ~0 means\n         the featured and clean coupons were effectively "
          "the same drawing.")
    print("\nNEXT     point 3a_doors/5_rank_designs DATASETS_DIR at this "
          "committed Deltas/ directory.")


if __name__ == "__main__":
    main()
