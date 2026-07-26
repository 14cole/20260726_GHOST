#!/usr/bin/env python3
"""
STEP 1a -- 2-D MONOSTATIC SOLVE, ON THIS MACHINE
================================================

WHAT IT DOES
  Solves every 2-D cross-section ("coupon") in geometries/, in both
  polarizations, at every frequency, using all the cores you have.  A coupon is
  a slice through the joint: a flat plank with the feature cut into its outer
  face.  Each seam VARIATION needs a PAIR that shares one mesh and one angle
  grid, which is what makes the subtraction in step 1c cancel cleanly:

      geometries/FRD/<base>.geo    clean    (FRD = faired / smooth)
      geometries/OPN/<base>.geo    featured (OPN = feature present)

  Nothing is subtracted here.  The four coupon-drawing rules (outer face at
  y = 0, feature centred, plank much wider than the feature, rounded end caps)
  are in ../Docs/FEATURE_SUM_GUIDE.md section 4 -- they are what make step 1c
  legitimate.

  This is the same solve as step 1b, without a cluster.  Use 1a for a handful
  of coupons; use 1b when a whole design library makes the count large.

INPUTS
  geometries/FRD/<base>.geo        one clean cross-section per variation
  geometries/OPN/<base>.geo        the same plank WITH the feature

OUTPUTS
  results/<POL>_<FREQ>GHz_<base>_<ROLE>.grim
                                   one file per (polarization, frequency) --
                                   EXACTLY the names the HPC driver writes, so
                                   step 1c reads 1a and 1b output identically.

KNOBS (below)
  FREQUENCIES_GHZ, ANGLES_DEG (90 = normal incidence on the outer face),
  POLARIZATIONS, GEOMETRY_UNITS, WORKERS, FORCE

NEXT  1c_build_deltas

NOTE  Reuse is allowed only when results/cache_manifest.json proves that the
      exact geometry/material bytes, units, grids, polarizations, solver source,
      expected filenames, and output bytes still match.  A legacy, partial, or
      changed cache is preserved and rejected; set FORCE = True to rebuild all
      expected units consistently. Unexpected stale .grim files are never
      deleted automatically.

    python3 run.py
"""

import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set before NumPy/SciPy can initialize a BLAS runtime; pool workers then
# inherit the same one-thread policy recorded by this runner.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)

from workflow_provenance import (backend_source_paths,                # noqa: E402
                                 runtime_environment_fingerprint)

# ─── KNOBS ──────────────────────────────────────────────────────────────────
FREQUENCIES_GHZ = [3.0, 6.0]
ANGLES_DEG = np.arange(0.0, 180.1, 5.0)   # 2-D cut angles; 90 = onto the face
POLARIZATIONS = ["TM", "TE"]              # BOTH -- a delta needs both channels
                                          # (TM -> HH, TE -> VV downstream)
GEOMETRY_UNITS = "meters"                 # units the .geo files are drawn in
WORKERS = None                            # None = cpu_count() - 1
FORCE = False                             # True = re-solve even if it exists
# ────────────────────────────────────────────────────────────────────────────

ROLE_DIR = {"FRD": "FRD", "OPN": "OPN"}   # role marker <- which folder it is in
CACHE_SCHEMA = "ghost.workflow.2d-local-cache.v1"


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
    """Hash the runner plus every backend Python/native implementation file."""
    root = os.path.dirname(HERE)
    paths = [os.path.abspath(__file__)]
    paths.extend(backend_source_paths(BACKEND))
    records = [
        {"path": os.path.relpath(p, root), "sha256": _sha256_file(p)}
        for p in sorted(set(paths))
    ]
    return _canonical_sha256(records)


def _write_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def _read_manifest(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(
            f"{path} is missing or unreadable ({exc}). Existing solver files "
            "were preserved; set FORCE = True to rebuild the exact current "
            "output set.") from exc


def _cache_is_reusable(res_dir, expected_names, run_sha256):
    """Fail closed unless the manifest, exact file set, and bytes all agree."""
    manifest_path = os.path.join(res_dir, "cache_manifest.json")
    if os.path.isfile(os.path.join(res_dir, "collection_manifest.json")):
        raise SystemExit(
            "results/ contains an HPC collection_manifest.json. Refusing to "
            "mix local and collected solver bundles; move/archive the HPC "
            "bundle first."
        )
    existing = {
        os.path.basename(p)
        for p in glob.glob(os.path.join(res_dir, "*.grim"))
    }
    expected = set(expected_names)
    unexpected = sorted(existing - expected)
    if unexpected:
        raise SystemExit(
            "results/ contains unexpected stale .grim file(s) that step 1c "
            "would also ingest:\n  " + "\n  ".join(unexpected)
            + "\nThey were not deleted. Move them out of results/ before "
              "running this input set.")

    if FORCE:
        return False
    if not existing and not os.path.exists(manifest_path):
        return False
    if existing != expected:
        missing = sorted(expected - existing)
        raise SystemExit(
            "the 2-D cache is incomplete"
            + (f" (missing {missing[:8]})" if missing else "")
            + ". Existing files were preserved; set FORCE = True to rebuild "
              "all units from one consistent input state.")

    manifest = _read_manifest(manifest_path)
    if (manifest.get("schema") != CACHE_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("run_sha256") != run_sha256
            or set(manifest.get("expected_outputs", [])) != expected):
        raise SystemExit(
            "results/cache_manifest.json does not match the current geometry, "
            "material tables, units, frequency/angle grids, polarizations, or "
            "solver source. Existing files were preserved; set FORCE = True "
            "to rebuild.")
    recorded = manifest.get("output_sha256", {})
    bad = [
        name for name in sorted(expected)
        if recorded.get(name) != _sha256_file(os.path.join(res_dir, name))
    ]
    if bad:
        raise SystemExit(
            "cached 2-D output bytes do not match their manifest: "
            + ", ".join(bad[:8])
            + ". Existing files were preserved; set FORCE = True to rebuild.")
    return True


def _pin_blas(n=1):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


def _solve_unit(geo_path, role, pol, freq, out_path, provenance_sha256):
    """Pool-worker entry point: one (coupon, polarization, frequency)."""
    from geometry_io import parse_geometry, build_geometry_snapshot
    from rcs_solver import solve_monostatic_rcs_2d
    from grim_io import export_result_to_grim

    with open(geo_path) as fh:
        snap = build_geometry_snapshot(*parse_geometry(fh.read()))
    snap["source_path"] = geo_path
    result = solve_monostatic_rcs_2d(
        snap, [float(freq)], [float(a) for a in ANGLES_DEG], pol,
        geometry_units=GEOMETRY_UNITS,
        strict_quality_gate=True,
        compute_condition_number=True)
    written = export_result_to_grim(
        result, out_path[: -len(".grim")], source_path=geo_path,
        history=(f"step 1a local 2-D role={role} pol={pol} "
                 f"freq={freq}GHz; cache_input_sha256={provenance_sha256}"))
    return ("written", written[0] if written else out_path)


def discover():
    """geometries/{FRD,OPN}/*.geo -> [(path, base, role)], and the pairing report."""
    units, bases = [], {}
    for role in ("FRD", "OPN"):
        d = os.path.join(HERE, "geometries", role)
        if not os.path.isdir(d):
            print(f"  [warn] no geometries/{role}/ folder")
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.geo"))):
            base = os.path.basename(p)[: -len(".geo")]
            # tolerate a name that already carries its role marker
            if base.endswith(("_FRD", "_OPN")):
                base = base[:-4]
            units.append((p, base, role))
            bases.setdefault(base, set()).add(role)
    return units, bases


def _jobs_for_units(units, res_dir):
    jobs = []
    for geo, _base, role in units:
        base = os.path.basename(geo)[: -len(".geo")]
        if base.endswith(("_FRD", "_OPN")):
            base = base[:-4]
        for pol in POLARIZATIONS:
            for freq in FREQUENCIES_GHZ:
                out = os.path.join(
                    res_dir,
                    f"{pol}_{float(freq):.3f}GHz_{base}_{role}.grim",
                )
                jobs.append((geo, role, pol, float(freq), out))
    return jobs


def _signature_payload(units, expected_names):
    from feature_sum import geometry_input_fingerprint

    return {
        "schema": CACHE_SCHEMA,
        "geometry_units": str(GEOMETRY_UNITS).strip().lower(),
        "frequencies_ghz": [float(f) for f in FREQUENCIES_GHZ],
        "angles_deg": [float(a) for a in ANGLES_DEG],
        "polarizations": [str(p).strip().upper() for p in POLARIZATIONS],
        "geometry_inputs": {
            os.path.relpath(geo, HERE):
                geometry_input_fingerprint(geo, GEOMETRY_UNITS)
            for geo, _base, _role in units
        },
        "solver_source_sha256": _source_fingerprint(),
        "runtime_environment_sha256": runtime_environment_fingerprint(),
        "expected_outputs": sorted(expected_names),
    }


def _require_unchanged_signature(initial_payload, current_payload, label):
    if _canonical_sha256(current_payload) != _canonical_sha256(initial_payload):
        raise SystemExit(
            f"{label} changed while step 1a was running. Outputs were "
            "preserved but no complete manifest was written; rerun with "
            "FORCE = True from one stable input/source state."
        )


def main():
    units, bases = discover()
    if not units:
        raise SystemExit(
            "no geometries/FRD/*.geo or geometries/OPN/*.geo.\n"
            "  Put the clean cross-section of each variation in geometries/FRD/\n"
            "  and the featured one, SAME FILE NAME, in geometries/OPN/.")

    print(f"STEP 1a  {len(bases)} variation(s) at {FREQUENCIES_GHZ} GHz, "
          f"{len(ANGLES_DEG)} incidence angles, pols {POLARIZATIONS}")
    for base, roles in sorted(bases.items()):
        if roles != {"FRD", "OPN"}:
            print(f"         WARNING  {base} has only {sorted(roles)} -- step 1c "
                  f"needs BOTH to subtract")

    res_dir = os.path.join(HERE, "results")
    os.makedirs(res_dir, exist_ok=True)
    jobs = _jobs_for_units(units, res_dir)
    expected_names = [os.path.basename(j[-1]) for j in jobs]
    if len(set(expected_names)) != len(expected_names):
        raise SystemExit(
            "two requested solve units map to the same output filename. Check "
            "for duplicate polarization/frequency knobs or duplicate geometry "
            "base names.")

    signature_payload = _signature_payload(units, expected_names)
    run_sha256 = _canonical_sha256(signature_payload)
    if _cache_is_reusable(res_dir, expected_names, run_sha256):
        print(f"         cache verified: {len(expected_names)} output file(s), "
              "exact input/source manifest match")
        print("\nNEXT     1c_build_deltas  (joins pol/frequency, then OPN - FRD)")
        return

    in_progress = dict(signature_payload)
    in_progress.update(run_sha256=run_sha256, status="in_progress")
    _write_json_atomic(
        os.path.join(res_dir, "cache_manifest.json"), in_progress
    )

    cpu = os.cpu_count() or 1
    n_workers = max(1, min(int(WORKERS or max(1, cpu - 1)), len(jobs)))
    print(f"         {len(jobs)} unit(s) = {len(units)} coupon x "
          f"{len(POLARIZATIONS)} pol x {len(FREQUENCIES_GHZ)} freq, "
          f"{n_workers} of {cpu} cores\n")

    _pin_blas(1)
    t0 = time.time()
    n_done = n_skip = n_fail = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_pin_blas) as pool:
        futs = {
            pool.submit(_solve_unit, *j, run_sha256): j for j in jobs
        }
        for fut in as_completed(futs):
            geo, role, pol, f, out = futs[fut]
            tag = f"{pol} {f:6.3f}GHz {os.path.basename(out)[:-5]}"
            try:
                status, path = fut.result()
                n_skip += status == "skipped"
                n_done += status == "written"
                print(f"  [{n_done+n_skip+n_fail:3d}/{len(jobs)}] {status:7s}  "
                      f"{os.path.basename(path)}", flush=True)
            except Exception as exc:
                n_fail += 1
                print(f"  [{n_done+n_skip+n_fail:3d}/{len(jobs)}] FAILED   "
                      f"{tag}: {exc}", flush=True)

    print(f"\n         wrote {n_done}, skipped {n_skip}, failed {n_fail}  "
          f"({time.time()-t0:.1f} s)")
    print(f"         results/ now holds "
          f"{len(glob.glob(os.path.join(res_dir, '*.grim')))} file(s)")
    if n_fail:
        raise SystemExit("some units failed -- fix those before step 1c.")
    output_sha256 = {
        name: _sha256_file(os.path.join(res_dir, name))
        for name in sorted(expected_names)
    }
    current_units, _current_bases = discover()
    current_jobs = _jobs_for_units(current_units, res_dir)
    current_names = [os.path.basename(job[-1]) for job in current_jobs]
    try:
        current_payload = _signature_payload(current_units, current_names)
    except Exception as exc:
        raise SystemExit(
            "geometry/material inputs became unreadable while step 1a was "
            "running. Outputs were preserved but no complete manifest was "
            "written."
        ) from exc
    _require_unchanged_signature(
        signature_payload, current_payload, "solver inputs or source"
    )
    manifest = dict(signature_payload)
    manifest.update(
        run_sha256=run_sha256,
        output_sha256=output_sha256,
        status="complete")
    _write_json_atomic(os.path.join(res_dir, "cache_manifest.json"), manifest)
    print("         wrote results/cache_manifest.json")
    print("\nNEXT     1c_build_deltas  (joins pol/frequency, then OPN - FRD)")


if __name__ == "__main__":
    main()
