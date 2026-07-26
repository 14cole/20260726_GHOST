#!/usr/bin/env python3
"""
STEP 2b (part 2) -- COLLECT THE BODY SOLVE
==========================================

WHAT IT DOES
  Merges the cluster's per-unit files into the same two artefacts step 2a
  writes, in the same folder, so steps 3 and 4 cannot tell which path you took:

      Body/body.grim          ONE .grim: aspect x 1 x frequency x [VV, HH],
                              3-D convention, complex amplitude preserved.
                              Opens in the GRIM viewer like any other dataset.
      Body/body_profile.csv   the outer (rho, z) surface profile, in metres --
                              what gives the doors their surface normal.

  The driver writes one file per (polarization, frequency); the pipeline wants
  one artefact carrying both axes, so the pieces are put back together here.
  Aspects above 180 (the EXPAND_TO_360 mirror, if you turned it on) are dropped
  -- they are the same monostatic response reflected, and step 3 interpolates
  on 0..180.

  GEOMETRY_UNITS is read from the run's manifest, not re-declared here, so the
  profile cannot silently disagree with the solve.

INPUTS
  submitted.txt          written by submit.py (or pass a run dir as argv[1])
  runs/<run>/results/*.grim
  runs/<run>/inputs/     frozen .geo and mat.<flag> files actually solved

OUTPUTS
  Body/body.grim, Body/body_profile.csv, body_dbsm.csv

KNOBS  none.

CHECK  the dBsm numbers printed here should track a local solve of the same
       body and frequencies (step 2a).  Both paths use the same material-aware
       dispatch; small roundoff differences can remain across BLAS/thread
       configurations, especially when quoted in dB at a deep null.

NEXT   3_trade_study, with its BODY_DIR knob pointed at this folder's Body/

    python3 collect.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

from hpc_common import (bodies_from_units, read_unit_grims,           # noqa: E402
                        require_hpc_output_attestations,
                        require_hpc_run_provenance, run_status)
from feature_sum import (load_body_grim, outer_generatrix,             # noqa: E402
                         save_body_grim, geometry_input_fingerprint)
from geometry_io import parse_geometry, build_geometry_snapshot        # noqa: E402
from line_expand import dbsm                                          # noqa: E402
from workflow_provenance import (backend_source_fingerprint,          # noqa: E402
                                 manifest_solve_spec_fingerprint,
                                 runtime_environment_fingerprint,
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
            "the HPC run manifest changed while body collection was "
            "starting; nothing was collected."
        )
    try:
        require_hpc_run_provenance(
            st["manifest"], "ghost.hpc.bor-run.v1"
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
    print(f"STEP 2b  collecting {os.path.basename(str(st['run_dir']))}")
    print(f"         {st['n_done']} of {st['n_units']} unit(s) done"
          + (f", {st['pending']} still pending" if st["pending"] else ", complete"))
    if st["pending"]:
        raise SystemExit(
            "refusing to collect an incomplete body run; wait for every "
            "frequency/polarization unit so Body/body.grim cannot become a "
            "partial cache.")
    if st["n_done"] == 0:
        raise SystemExit("nothing to collect yet -- check the queue (python3 "
                         "status.py) or runs/<run>/logs/ for failures.")
    try:
        require_hpc_output_attestations(st["run_dir"], st["manifest"])
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))

    units = read_unit_grims(os.path.join(str(st["run_dir"]), "results"))
    try:
        bodies = bodies_from_units(units)
    except ValueError as exc:
        raise SystemExit(f"{exc}\n  (a missing polarization usually means the "
                         f"run is not finished -- python3 status.py)")
    freqs = sorted(bodies)
    print(f"         frequencies: {freqs} GHz;  "
          f"{len(bodies[freqs[0]]['theta_deg'])} aspects "
          f"{bodies[freqs[0]]['theta_deg'][0]:g}.."
          f"{bodies[freqs[0]]['theta_deg'][-1]:g} deg")

    # the units the solve actually ran in -- taken from the run, not re-declared
    units_str = st["manifest"].get("solver_config", {}).get("geometry_units")
    if not units_str:
        raise SystemExit("the run manifest has no solver_config.geometry_units "
                         "-- this run did not come from submit.py.")

    solved_geos = sorted({
        os.path.abspath(str(u.get("geometry", "")))
        for u in st["manifest"].get("units", [])
        if u.get("geometry")
    })
    if len(solved_geos) != 1 or not os.path.isfile(solved_geos[0]):
        raise SystemExit(
            "the run manifest must reference exactly one readable frozen "
            f"geometry; found {solved_geos}. Legacy runs without frozen inputs "
            "cannot safely rebuild body_profile.csv.")
    solved_geo = solved_geos[0]
    with open(solved_geo) as fh:
        snap = build_geometry_snapshot(*parse_geometry(fh.read()))
    gen = outer_generatrix(snap, units_str)
    input_hash = geometry_input_fingerprint(solved_geo, units_str)

    rows, hdr = [bodies[freqs[0]]["theta_deg"]], ["aspect_deg"]
    for f in freqs:
        for ch in ("vv", "hh"):
            rows.append(dbsm(4 * np.pi * np.abs(bodies[f][f"amp_{ch}"]) ** 2))
            hdr.append(f"{ch}_dBsm_{f:g}GHz")

    out_dir = os.path.join(HERE, "Body")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(os.path.join(out_dir, "cache_manifest.json")):
        raise SystemExit(
            "Body/ contains a local cache_manifest.json. Refusing to mix a "
            "local body cache with an HPC collection; move/archive it first."
        )
    unexpected_grims = sorted(
        name for name in os.listdir(out_dir)
        if name.lower().endswith(".grim") and name != "body.grim"
    )
    if unexpected_grims:
        raise SystemExit(
            "Body/ contains unexpected stale .grim file(s): "
            f"{unexpected_grims}. They were preserved; move/archive them "
            "before collection."
        )

    run_solve_sha256 = manifest_solve_spec_fingerprint(st["manifest"])
    collection_source_sha256 = backend_source_fingerprint(
        BACKEND,
        {"2b_solve_body_hpc/collect.py": os.path.abspath(__file__)},
    )
    collection_runtime_sha256 = runtime_environment_fingerprint()
    provenance = {
        "run_id": str(st["manifest"]["run_id"]),
        "run_manifest_sha256": run_manifest_sha256,
        "run_solve_spec_sha256": run_solve_sha256,
        "geometry_input_sha256": input_hash,
        "solver_source_sha256":
            str(st["manifest"]["solver_source_sha256"]),
        "solver_runtime_environment_sha256":
            str(st["manifest"]["runtime_environment_sha256"]),
        "collection_source_sha256": collection_source_sha256,
        "collection_runtime_environment_sha256":
            collection_runtime_sha256,
    }

    stage = tempfile.mkdtemp(prefix=".collect-stage-", dir=HERE)
    try:
        stage_body = os.path.join(stage, "Body")
        os.makedirs(stage_body)
        save_body_grim(
            bodies, os.path.join(stage_body, "body.grim"),
            source_path=solved_geo,
            geometry_input_sha256=input_hash,
            solver_source_sha256=
                str(st["manifest"]["solver_source_sha256"]),
            runtime_environment_sha256=
                str(st["manifest"]["runtime_environment_sha256"]),
            run_solve_spec_sha256=run_solve_sha256,
            collection_source_sha256=collection_source_sha256,
            history=f"step 2b HPC run "
                    f"{os.path.basename(str(st['run_dir']))}",
        )
        np.savetxt(
            os.path.join(stage_body, "body_profile.csv"),
            np.asarray(gen, float),
            delimiter=",",
            header="rho_m,z_m",
            comments="",
        )
        staged_dbsm = os.path.join(stage, "body_dbsm.csv")
        np.savetxt(
            staged_dbsm,
            np.column_stack(rows),
            delimiter=",",
            fmt="%.4f",
            header=",".join(hdr),
            comments="",
        )

        # Validate the exact derived bytes that will be committed, not only
        # the per-unit sources used to create them.
        try:
            staged_bodies = load_body_grim(
                os.path.join(stage_body, "body.grim")
            )
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(
                f"staged Body/body.grim failed strict physical validation: "
                f"{exc}"
            ) from exc
        if set(staged_bodies) != set(bodies):
            raise SystemExit(
                "staged Body/body.grim changed the solved frequency set."
            )
        for frequency in sorted(bodies):
            before = bodies[frequency]
            after = staged_bodies[frequency]
            for key in ("theta_deg", "amp_vv", "amp_hh"):
                if not np.array_equal(
                    np.asarray(before[key]), np.asarray(after[key])
                ):
                    raise SystemExit(
                        "staged Body/body.grim changed the exact "
                        f"{frequency:g} GHz {key} values."
                    )
        staged_profile = np.loadtxt(
            os.path.join(stage_body, "body_profile.csv"),
            delimiter=",",
            skiprows=1,
        )
        if (
            staged_profile.shape != np.asarray(gen, float).shape
            or not np.all(np.isfinite(staged_profile))
            or not np.array_equal(staged_profile, np.asarray(gen, float))
        ):
            raise SystemExit(
                "staged Body/body_profile.csv changed or contains nonfinite "
                "geometry."
            )

        # Recheck every frozen input and source after the derived product was
        # built, before committing any bytes.
        require_hpc_run_provenance(
            st["manifest"], "ghost.hpc.bor-run.v1"
        )
        require_hpc_output_attestations(st["run_dir"], st["manifest"])
        if (
            geometry_input_fingerprint(solved_geo, units_str) != input_hash
            or sha256_file(run_manifest_path) != run_manifest_sha256
            or backend_source_fingerprint(
                BACKEND,
                {
                    "2b_solve_body_hpc/collect.py":
                        os.path.abspath(__file__)
                },
            ) != collection_source_sha256
            or runtime_environment_fingerprint()
            != collection_runtime_sha256
        ):
            raise SystemExit(
                "an input, run manifest, collector source, or numerical "
                "runtime changed during body collection; no new bundle was "
                "committed."
            )

        # Data replacements happen before their atomic commit manifests. A
        # crash therefore leaves hashes that downstream stages reject.
        write_artifact_in_progress(
            out_dir,
            "ghost.hpc.body-collection.v1",
            ["body.grim", "body_profile.csv"],
            provenance,
        )
        write_artifact_in_progress(
            HERE,
            "ghost.hpc.body-collection-bundle.v1",
            [
                "Body/body.grim",
                "Body/body_profile.csv",
                "Body/collection_manifest.json",
                "body_dbsm.csv",
            ],
            provenance,
        )
        os.replace(
            os.path.join(stage_body, "body.grim"),
            os.path.join(out_dir, "body.grim"),
        )
        os.replace(
            os.path.join(stage_body, "body_profile.csv"),
            os.path.join(out_dir, "body_profile.csv"),
        )
        os.replace(staged_dbsm, os.path.join(HERE, "body_dbsm.csv"))
        write_artifact_manifest(
            out_dir,
            "ghost.hpc.body-collection.v1",
            ["body.grim", "body_profile.csv"],
            provenance,
        )
        write_artifact_manifest(
            HERE,
            "ghost.hpc.body-collection-bundle.v1",
            [
                "Body/body.grim",
                "Body/body_profile.csv",
                "Body/collection_manifest.json",
                "body_dbsm.csv",
            ],
            provenance,
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(f"         wrote and committed Body/body.grim and "
          f"Body/body_profile.csv  ({len(gen)} points, max radius "
          f"{float(np.max(np.asarray(gen)[:, 0])):.4f} m, "
          f"geometry drawn in {units_str})")
    print("         wrote body_dbsm.csv and collection_manifest.json")
    for i, f in enumerate(freqs):
        vv = rows[1 + 2 * i]
        print(f"         {f:g} GHz VV: nose-on {vv[0]:+7.1f}  broadside "
              f"{vv[len(vv)//2]:+7.1f}  tail-on {vv[-1]:+7.1f} dBsm")

    print("\nCHECK    these should match a local solve of the same body and "
          "frequencies\n         (step 2a) -- the cluster only changes WHERE it "
          "is computed.")
    print("NEXT     3_trade_study  (point its BODY_DIR knob at "
          "../2b_solve_body_hpc/Body)")


if __name__ == "__main__":
    main()
