#!/usr/bin/env python3
"""
STEP 1b (part 2) -- COLLECT THE CLUSTER'S RESULTS
=================================================

WHAT IT DOES
  Copies the finished per-unit files out of the run directory into results/,
  which is the ONLY folder step 1c needs to know about.  After this, 1b's
  results/ and 1a's results/ are the same thing:

      results/TM_3.000GHz_SEAL-00-01_0.010gap_OPN.grim
      results/TE_3.000GHz_SEAL-00-01_0.010gap_OPN.grim
      results/TM_6.000GHz_SEAL-00-01_0.010gap_OPN.grim
      ...

  The join across polarization and frequency, and the OPN - FRD subtraction,
  both happen in step 1c -- so the cluster path and the laptop path converge
  before anything irreversible is done to the data.

INPUTS
  submitted.txt          written by submit.py (or pass a run dir as argv[1])
  runs/<run>/results/*.grim

OUTPUTS
  results/*.grim         one per (polarization, frequency, coupon)
  collected.csv          what was copied, and the (pol x frequency) grid found

KNOBS  none.

NEXT   1c_build_deltas

    python3 collect.py
"""

import csv
import glob
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

from hpc_common import (require_hpc_output_attestations,              # noqa: E402
                        require_hpc_run_provenance, run_status)
from grim_naming import group_solver_files, parse_variation           # noqa: E402
from workflow_provenance import (manifest_solve_spec_fingerprint,     # noqa: E402
                                 sha256_file, write_artifact_in_progress,
                                 write_artifact_manifest)


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        stamp = os.path.join(HERE, "submitted.txt")
        if not os.path.exists(stamp):
            raise SystemExit("no submitted.txt -- run submit.py first, or pass a "
                             "run directory as an argument.")
        run_dir = open(stamp).read().strip()

    st = run_status(run_dir)
    run_manifest_path = os.path.join(
        str(st["run_dir"]), "manifest.json"
    )
    manifest_hash_before_read = sha256_file(run_manifest_path)
    try:
        with open(run_manifest_path, encoding="utf-8") as stream:
            manifest_from_disk = json.load(stream)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"unreadable HPC run manifest: {exc}") from exc
    run_manifest_sha256 = sha256_file(run_manifest_path)
    if (
        manifest_hash_before_read != run_manifest_sha256
        or manifest_from_disk != st["manifest"]
    ):
        raise SystemExit(
            "the HPC run manifest changed while collection was starting; "
            "nothing was collected."
        )
    try:
        require_hpc_run_provenance(
            st["manifest"], "ghost.hpc.2d-run.v1"
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
    print(f"STEP 1b  collecting {os.path.basename(str(st['run_dir']))}")
    print(f"         {st['n_done']} of {st['n_units']} unit(s) done"
          + (f", {st['pending']} still pending" if st["pending"] else ", complete"))
    if st["pending"]:
        raise SystemExit(
            "refusing to collect an incomplete run. Mixing its partial files "
            "with an older results/ folder can fabricate an apparently complete "
            "delta from different solver inputs; wait for every unit.")
    if st["n_done"] == 0:
        raise SystemExit("nothing to collect yet -- check the queue (python3 "
                         "status.py) or runs/<run>/logs/ for failures.")
    try:
        require_hpc_output_attestations(st["run_dir"], st["manifest"])
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    src = sorted(glob.glob(os.path.join(str(st["run_dir"]), "results", "*.grim")))
    sidecars = [p + ".provenance.json" for p in src]
    missing_sidecars = [
        os.path.basename(p) for p in sidecars if not os.path.isfile(p)
    ]
    if missing_sidecars:
        raise SystemExit(
            "attested HPC results lost provenance sidecars before collection "
            f"(first {missing_sidecars[:5]})."
        )
    source_names = {os.path.basename(p) for p in src}
    stale = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(out_dir, "*.grim"))
        if os.path.basename(p) not in source_names)
    if stale:
        raise SystemExit(
            f"results/ contains {len(stale)} stale .grim file(s) not produced "
            f"by this run (first {stale[:5]}). Move/archive them before "
            "collecting; runs are never silently mixed.")
    expected_sidecars = {os.path.basename(p) for p in sidecars}
    stale_sidecars = sorted(
        os.path.basename(p)
        for p in glob.glob(
            os.path.join(out_dir, "*.grim.provenance.json")
        )
        if os.path.basename(p) not in expected_sidecars
    )
    if stale_sidecars:
        raise SystemExit(
            "results/ contains stale output-attestation sidecar(s) not "
            f"produced by this run (first {stale_sidecars[:5]}). "
            "Move/archive them before collecting."
        )
    if os.path.exists(os.path.join(out_dir, "cache_manifest.json")):
        raise SystemExit(
            "results/ contains a local cache_manifest.json. Refusing to mix "
            "a local cache with an HPC collection; move/archive it first."
        )

    groups, unparsed = group_solver_files(src)
    for p, why in unparsed:
        print(f"         NOT INDEXED  {os.path.basename(p)}: {why}")

    rows = [("variation", "role", "n_files", "polarizations", "frequencies_GHz")]
    roles = {}
    for variation in sorted(groups):
        recs = groups[variation]
        pols = sorted({r["pol_canon"] for r in recs})
        freqs = sorted({r["freq_ghz"] for r in recs})
        _base, role = parse_variation(variation)
        roles[role] = roles.get(role, 0) + 1
        rows.append((variation, role or "none", str(len(recs)), " ".join(pols),
                     " ".join(f"{f:g}" for f in freqs)))
        print(f"         {len(recs):>3} file(s)  {' '.join(pols):<8} "
              f"{' '.join(f'{f:g}' for f in freqs):<12} {variation}")

    stage = tempfile.mkdtemp(prefix=".collect-stage-", dir=HERE)
    try:
        stage_results = os.path.join(stage, "results")
        os.makedirs(stage_results)
        for path in src + sidecars:
            target = os.path.join(
                stage_results, os.path.basename(path)
            )
            shutil.copyfile(path, target)
            if sha256_file(path) != sha256_file(target):
                raise SystemExit(
                    f"staged copy changed bytes for "
                    f"{os.path.basename(path)}."
                )
        staged_csv = os.path.join(stage, "collected.csv")
        with open(staged_csv, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)

        # Validate the copied pair, not merely the source that existed before
        # staging. This closes the copy-time race and proves the exact bytes
        # that will be committed still satisfy every unit's semantic
        # attestation. Recheck frozen geometry/material inputs and the run
        # manifest at the same boundary.
        if sha256_file(run_manifest_path) != run_manifest_sha256:
            raise SystemExit(
                "the HPC run manifest changed during collection. Staged "
                "files were not committed."
            )
        try:
            require_hpc_run_provenance(
                st["manifest"], "ghost.hpc.2d-run.v1"
            )
            require_hpc_output_attestations(stage, st["manifest"])
        except (OSError, ValueError) as exc:
            raise SystemExit(
                f"staged HPC output/attestation verification failed: {exc}"
            ) from exc

        # Replace data first, then atomically write the commit marker. An
        # interrupted collection leaves a manifest/hash mismatch downstream.
        collection_provenance = {
            "run_id": str(st["manifest"]["run_id"]),
            "run_manifest_sha256": run_manifest_sha256,
            "run_solve_spec_sha256":
                manifest_solve_spec_fingerprint(st["manifest"]),
            "solver_source_sha256":
                str(st["manifest"]["solver_source_sha256"]),
            "runtime_environment_sha256":
                str(st["manifest"]["runtime_environment_sha256"]),
        }
        write_artifact_in_progress(
            out_dir,
            "ghost.hpc.2d-collection.v1",
            sorted(source_names | expected_sidecars),
            collection_provenance,
        )
        for name in sorted(source_names | expected_sidecars):
            os.replace(
                os.path.join(stage_results, name),
                os.path.join(out_dir, name),
            )
        os.replace(staged_csv, os.path.join(HERE, "collected.csv"))
        write_artifact_manifest(
            out_dir,
            "ghost.hpc.2d-collection.v1",
            sorted(source_names | expected_sidecars),
            collection_provenance,
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(f"\n         copied {len(src)} file(s) into results/, wrote collected.csv")
    print("         committed results/collection_manifest.json with exact "
          "data and attestation hashes")
    print(f"         {len(groups)} variation(s); roles {roles}  "
          f"(OPN = featured, FRD = clean)")
    if roles.get("OPN") != roles.get("FRD"):
        print("         NOTE the OPN and FRD counts differ -- step 1c pairs by "
              "base name and\n              LISTS whatever is unmatched rather "
              "than dropping it quietly.")
    print("\nCHECK    every variation should show BOTH polarizations and the SAME "
          "frequency\n         list.  A missing pol means half the delta is zero.")
    print("NEXT     1c_build_deltas  (point its INPUTS knob at this results/)")


if __name__ == "__main__":
    main()
