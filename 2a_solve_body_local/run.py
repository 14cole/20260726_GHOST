#!/usr/bin/env python3
"""
STEP 2a -- SOLVE THE BODY, ON THIS MACHINE
==========================================

WHAT IT DOES
  Solves the bare body of revolution -- the smooth vehicle with no features on
  it.  This is the slow step, and the only one that is: you run it ONCE per
  body and per frequency set, and every trade study afterwards reuses it.

  This is the same solve as step 2b, without a cluster.  Use 2a when one body
  at a handful of frequencies will finish while you wait; use 2b when the
  aspect sweep, the frequency list or the body size makes that unreasonable.

  Put exactly ONE body .geo in this folder.  Materials come from the .geo
  itself (segment TYPE tags + the IBCS_Resistances / Dielectrics sections),
  so there is nothing to configure here beyond the sweep.

INPUTS
  <one>.geo              your body of revolution, drawn in GEOMETRY_UNITS

OUTPUTS
  Body/body.grim         the solve: aspect x frequency x [VV, HH], dBsm.
                         Opens in the GRIM viewer like any other dataset, and
                         doubles as the cache only when cache_manifest.json
                         verifies every physical input, source, and output byte.
  Body/body_profile.csv  the outer (rho, z) surface profile, in metres.  Steps
                         3 and 4 read this to give the doors their surface
                         normal and to check they sit on the skin.  You never
                         edit it.
  Body/cache_manifest.json
                         stable provenance for the exact body/profile pair.

  The angle axis is the ASPECT from the nose (0 = nose-on, 180 = tail-on),
  which is all a body of revolution has.  Azimuth and elevation appear in
  steps 3 and 4.

KNOBS (below)
  FREQUENCIES_GHZ, ASPECT_STEP_DEG, GEOMETRY_UNITS, WORKERS, FORCE

  FREQUENCIES_GHZ must cover what you ask for in step 3 -- the body cannot be
  interpolated onto a frequency it was never solved at, and step 3 will say so.

NEXT  3_trade_study

    python3 run.py
"""

import glob
import hashlib
import json
import os
import sys

# BoR parallelism is explicit in WORKERS; keep nested BLAS threads at one and
# set the policy before NumPy/SciPy initialize.
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
ROOT = os.path.dirname(HERE)
BACKEND = os.path.join(ROOT, "Backend")
sys.path.insert(0, BACKEND)
sys.path.insert(0, ROOT)

# ─── KNOBS ──────────────────────────────────────────────────────────────────
FREQUENCIES_GHZ = [3.0, 6.0]
# None solves the EXACT aspects used by grid.py (recommended/physical default).
# A positive number requests a legacy uniform 0..180 sweep; step 4 will refuse
# it unless it also happens to contain every output-grid aspect exactly.
ASPECT_STEP_DEG = None
GEOMETRY_UNITS = "meters"     # units the .geo is drawn in
WORKERS = 4
FORCE = False                 # True = re-solve even if Body/body.grim exists
# ────────────────────────────────────────────────────────────────────────────

CACHE_SCHEMA = "ghost.workflow.body-local-cache.v1"

from grid import AZIMUTHS_DEG, ELEVATIONS_DEG                         # noqa: E402
from frame import AXIS_AZ_DEG, AXIS_EL_DEG                            # noqa: E402
from feature_sum import (solve_vehicle_body, save_body_grim,          # noqa: E402
                         load_body_grim, outer_generatrix,
                         radar_grid_aspects, geometry_input_fingerprint)
from geometry_io import parse_geometry, build_geometry_snapshot        # noqa: E402
from line_expand import dbsm                                          # noqa: E402
from workflow_provenance import (backend_source_paths,                # noqa: E402
                                 runtime_environment_fingerprint)


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
    paths = [os.path.abspath(__file__),
             os.path.join(ROOT, "grid.py"),
             os.path.join(BACKEND, "frame.py")]
    paths.extend(backend_source_paths(BACKEND))
    return _canonical_sha256([
        {"path": os.path.relpath(p, ROOT), "sha256": _sha256_file(p)}
        for p in sorted(set(paths))
    ])


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
            f"{path} is missing or unreadable ({exc}). Existing body "
            "artifacts were preserved; set FORCE = True to rebuild.") from exc


def _cache_is_reusable(out_dir, run_sha256):
    if os.path.isfile(os.path.join(out_dir, "collection_manifest.json")):
        raise SystemExit(
            "Body/ contains an HPC collection_manifest.json. Refusing to mix "
            "local and collected body bundles; move/archive the HPC bundle "
            "first."
        )
    expected_grims = {"body.grim"}
    existing_grims = {
        os.path.basename(p)
        for p in glob.glob(os.path.join(out_dir, "*.grim"))
    }
    unexpected = sorted(existing_grims - expected_grims)
    if unexpected:
        raise SystemExit(
            "Body/ contains unexpected stale .grim file(s):\n  "
            + "\n  ".join(unexpected)
            + "\nThey were not deleted. Move them out of Body/ before "
              "running this body input.")

    manifest_path = os.path.join(out_dir, "cache_manifest.json")
    expected_paths = {
        "body.grim": os.path.join(out_dir, "body.grim"),
        "body_profile.csv": os.path.join(out_dir, "body_profile.csv"),
    }
    any_artifact = (
        os.path.exists(manifest_path)
        or any(os.path.exists(p) for p in expected_paths.values())
    )
    if FORCE:
        return False
    if not any_artifact:
        return False
    missing = [name for name, path in expected_paths.items()
               if not os.path.isfile(path)]
    if missing:
        raise SystemExit(
            f"the body cache is incomplete (missing {missing}). Existing "
            "artifacts were preserved; set FORCE = True to rebuild.")
    manifest = _read_manifest(manifest_path)
    if (manifest.get("schema") != CACHE_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("run_sha256") != run_sha256
            or set(manifest.get("expected_outputs", []))
            != set(expected_paths)):
        raise SystemExit(
            "Body/cache_manifest.json does not match the current geometry, "
            "material tables, units, frequency/aspect grid, frame, solver "
            "knobs, or source code. Existing artifacts were preserved; set "
            "FORCE = True to rebuild.")
    recorded = manifest.get("output_sha256", {})
    bad = [
        name for name, path in expected_paths.items()
        if recorded.get(name) != _sha256_file(path)
    ]
    if bad:
        raise SystemExit(
            "body cache bytes do not match their manifest: "
            + ", ".join(bad)
            + ". Existing artifacts were preserved; set FORCE = True to "
              "rebuild.")
    return True


def _signature_payload(geo, input_hash, aspects, required_aspects):
    return {
        "schema": CACHE_SCHEMA,
        "geometry_input_sha256": input_hash,
        "geometry_units": str(GEOMETRY_UNITS).strip().lower(),
        "frequencies_ghz": sorted(float(f) for f in FREQUENCIES_GHZ),
        "aspects_deg": [float(a) for a in aspects],
        "required_radar_aspects_deg": [
            float(a) for a in required_aspects
        ],
        "axis_az_deg": float(AXIS_AZ_DEG),
        "axis_el_deg": float(AXIS_EL_DEG),
        "cfie_alpha": 0.5,
        "workers": int(WORKERS),
        "solver_source_sha256": _source_fingerprint(),
        "runtime_environment_sha256": runtime_environment_fingerprint(),
        "expected_outputs": ["body.grim", "body_profile.csv"],
    }


def _require_unchanged_signature(initial_payload, current_payload):
    if _canonical_sha256(current_payload) != _canonical_sha256(initial_payload):
        raise SystemExit(
            "body geometry/material inputs, grids, runtime, or solver source "
            "changed while step 2a was running. Body artifacts were preserved "
            "but no complete manifest was written; rerun with FORCE = True "
            "from one stable input/source state."
        )


def main():
    geos = sorted(glob.glob(os.path.join(HERE, "*.geo")))
    if len(geos) != 1:
        raise SystemExit(f"put exactly one body .geo in {HERE} -- found "
                         f"{[os.path.basename(g) for g in geos]}.")
    geo = geos[0]
    input_hash = geometry_input_fingerprint(geo, GEOMETRY_UNITS)
    with open(geo) as fh:
        snap = build_geometry_snapshot(*parse_geometry(fh.read()))
    if geometry_input_fingerprint(geo, GEOMETRY_UNITS) != input_hash:
        raise SystemExit(
            "body geometry/material inputs changed while they were being "
            "parsed; no solve was started."
        )
    gen = outer_generatrix(snap, GEOMETRY_UNITS)
    out_dir = os.path.join(HERE, "Body")
    os.makedirs(out_dir, exist_ok=True)
    grim = os.path.join(out_dir, "body.grim")
    required_aspects = radar_grid_aspects(
        AZIMUTHS_DEG, ELEVATIONS_DEG, AXIS_AZ_DEG, AXIS_EL_DEG)
    if ASPECT_STEP_DEG is None:
        aspects = required_aspects
    else:
        step = float(ASPECT_STEP_DEG)
        if not np.isfinite(step) or step <= 0.0:
            raise SystemExit("ASPECT_STEP_DEG must be None or a positive number.")
        aspects = np.arange(0.0, 180.0 + 0.5 * step, step)

    missing_requested_aspects = [
        float(q) for q in required_aspects
        if not np.any(np.isclose(aspects, q, rtol=0.0, atol=1e-9))
    ]
    if missing_requested_aspects:
        raise SystemExit(
            "the requested body aspect sweep does not contain every exact "
            f"aspect required by grid.py ({len(missing_requested_aspects)} "
            f"missing; first {missing_requested_aspects[:5]}). Use "
            "ASPECT_STEP_DEG = None for the production grid.")

    signature_payload = _signature_payload(
        geo, input_hash, aspects, required_aspects
    )
    run_sha256 = _canonical_sha256(signature_payload)

    if _cache_is_reusable(out_dir, run_sha256):
        with np.load(grim, allow_pickle=False) as cached:
            cached_hash = (
                str(np.asarray(cached["geometry_input_sha256"]).ravel()[0])
                if "geometry_input_sha256" in cached.files else "")
        if cached_hash != input_hash:
            why = ("has no geometry fingerprint" if not cached_hash else
                   "was solved from different geometry, material tables, or units")
            raise SystemExit(
                f"Body/body.grim {why}. Set FORCE = True and re-solve; a stale "
                "body field is never paired with the current profile.")
        have = load_body_grim(grim)
        if not np.array_equal(
                np.asarray(sorted(have), dtype=float),
                np.asarray(sorted(float(f) for f in FREQUENCIES_GHZ),
                           dtype=float)):
            raise SystemExit(
                f"Body/body.grim has frequency grid {sorted(have)} GHz but the "
                f"current exact grid is "
                f"{sorted(float(f) for f in FREQUENCIES_GHZ)}. Set FORCE = "
                "True to rebuild.")
        have_aspects = np.asarray(
            have[sorted(have)[0]]["theta_deg"], dtype=float)
        if not np.array_equal(have_aspects, np.asarray(aspects, dtype=float)):
            raise SystemExit(
                "Body/body.grim does not contain the exact current aspect "
                "grid. Set FORCE = True and re-solve; coarse complex "
                "interpolation is not accepted for production.")
        print(f"STEP 2a  reusing Body/body.grim ({len(have)} frequencies, "
              f"{len(have[sorted(have)[0]]['theta_deg'])} aspects)")
        bodies = have
    else:
        in_progress = dict(signature_payload)
        in_progress.update(run_sha256=run_sha256, status="in_progress")
        _write_json_atomic(
            os.path.join(out_dir, "cache_manifest.json"), in_progress
        )
        print(f"STEP 2a  solving {os.path.basename(geo)} at {FREQUENCIES_GHZ} GHz, "
              f"{len(aspects)} aspects (this is the slow step) ...")
        bodies, _gen = solve_vehicle_body(geo, FREQUENCIES_GHZ, list(aspects),
                                          geometry_units=GEOMETRY_UNITS,
                                          cfie_alpha=0.5, workers=WORKERS)
        save_body_grim(
            bodies, grim, source_path=os.path.abspath(geo),
            geometry_input_sha256=input_hash,
            history=(f"step 2 from {os.path.basename(geo)}; "
                     f"cache_input_sha256={run_sha256}"))
        print("        wrote Body/body.grim")

        profile_path = os.path.join(out_dir, "body_profile.csv")
        np.savetxt(profile_path, np.asarray(gen, float), delimiter=",",
                   header="rho_m,z_m", comments="")
        output_sha256 = {
            "body.grim": _sha256_file(grim),
            "body_profile.csv": _sha256_file(profile_path),
        }
        current_geos = sorted(glob.glob(os.path.join(HERE, "*.geo")))
        if current_geos != [geo]:
            raise SystemExit(
                "the body geometry file set changed while step 2a was "
                "running. Body artifacts were preserved but no complete "
                "manifest was written."
            )
        try:
            current_input_hash = geometry_input_fingerprint(
                geo, GEOMETRY_UNITS
            )
            current_payload = _signature_payload(
                geo, current_input_hash, aspects, required_aspects
            )
        except Exception as exc:
            raise SystemExit(
                "body geometry/material inputs became unreadable while step "
                "2a was running. Body artifacts were preserved but no "
                "complete manifest was written."
            ) from exc
        _require_unchanged_signature(signature_payload, current_payload)
        manifest = dict(signature_payload)
        manifest.update(
            run_sha256=run_sha256,
            output_sha256=output_sha256,
            status="complete")
        _write_json_atomic(
            os.path.join(out_dir, "cache_manifest.json"), manifest)
        print("        wrote Body/body_profile.csv and cache_manifest.json "
              f"({len(gen)} profile points, max radius "
              f"{float(np.max(np.asarray(gen)[:, 0])):.4f} m)")

    for f in sorted(bodies):
        vv = dbsm(4 * np.pi * np.abs(np.asarray(bodies[f]["amp_vv"])) ** 2)
        print(f"        {f:g} GHz VV: nose-on {vv[0]:+7.1f}  broadside "
              f"{vv[len(vv)//2]:+7.1f}  tail-on {vv[-1]:+7.1f} dBsm")
    print("\nNEXT    3_trade_study")


if __name__ == "__main__":
    main()
