#!/usr/bin/env python3
"""
STEP 3b -- ADD A WING / FIN, WITH ITS ROOT CORNER
=================================================

WHAT IT DOES
  A wing and the corner it forms with the body are one physical assembly, so
  they are handled in one step. Solving the wing without its root is an
  incomplete reference, not a guaranteed lower bound: omitted terms can
  interfere either way. Near a right-angle root the corner estimate is often
  the largest individual contribution.

  THE WING is the same line expansion as a seam, with three differences:
    1. the coefficient is the airfoil's FULL 2-D amplitude, not a
       featured-minus-clean difference.  Nothing is subtracted: a wing is not a
       modification of the body skin, it is its own object;
    2. the line is the OPEN span, root to tip, not a closed loop;
    3. it carries its OWN normal -- the airfoil face -- not the body's surface
       normal.  Get this wrong and the wing faces the wrong way.
  A straight untwisted wing matches the model's 2-D extrusion assumption along
  its interior span. Finite tip/root diffraction, body coupling, and higher
  multiple scattering remain outside this reduced-order model.

  THE CORNER is a double bounce -- body, then wing, then back to the radar --
  which is in neither isolated solve:
    magnitude    the dihedral peak 8*pi*a^2*b^2/lambda^2, b = the fold (root)
                 length, a = FACE_WIDTH, the effective height the double bounce
                 reaches up the wing.  FACE_WIDTH is the main knob and it is a
                 judgement call.
    pattern      narrow along the fold (sinc^2), broad across it -- the
                 retroreflector behaviour that makes corners so bright.
    canted root  not square?  The lobe deflects 2*CANT off the bisector and the
                 peak rolls off as cos^2(2*CANT).  CANT = 0 is the exact
                 right-dihedral result.
    pol          exact: co-pol when the fold lines up with the radar V/H, pure
                 cross-pol at 45 degrees.
  It is a PO-LEVEL ESTIMATE, deliberately on the high side, and its internal
  double-bounce phase is NOT tracked.

INPUTS
  wing_airfoil.geo       the 2-D airfoil cross-section (a full object, not a
                         featured/clean pair)
  wing_span.txt          two rows, root then tip: x,y,z in UNITS, CAD frame
  BODY_DIR/body_profile.csv    from step 2a/2b -- the body normal at the root

OUTPUTS
  wing_<pol>.grim        the airfoil solve, cached only with an exact verified
                         wing_cache_manifest.json (use FORCE to rebuild)
  Output/<name>.grim     the wing alone, with its coherent complex field
  Output/<name>_corner_estimate.grim
                         the PO corner alone, tagged combine_role='power'.
                         Its unknown internal phase is never allowed to create
                         a fictitious wing-corner interference term.
  wing_dbsm.csv          wing / corner / statistical estimate over the grid
  Output/provenance_manifest.json
                         exact hashes/configuration for the component outputs

KNOBS (below)
  NAME, WING_NORMAL, ADD_CORNER, FOLD_LENGTH_M, FACE_WIDTH_M, CANT_DEG,
  ANGLES_DEG, UNITS, BODY_DIR, FORCE, ROOT_SKIN_TOL_M,
  ROOT_PHASE_TOL_DEG  (N_BODY is derived unless you override it)

  The output grid lives in ../grid.py, shared with 3a and 3c.

WHY THE WING GOES SILENT
  A component you cannot see returns nothing, CORRECTLY.  Looked at edge-on a
  wing reads -200 dBsm, and a corner with only one face lit reads -200 dBsm.
  That is right, and it is also the single most common reason someone thinks
  this step "doesn't work".  The summary below prints where each one peaks so
  you can see it is being lit at all.

NEXT  4_combine

    python3 run.py
"""

import csv
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.dirname(HERE))                # for ../grid.py

# ─── KNOBS ──────────────────────────────────────────────────────────────────
NAME = "fin_dorsal"           # what the component file is called
WING_NORMAL = (1.0, 0.0, 0.0)  # the airfoil FACE normal, CAD frame
                               # (this fin stands up, so its face looks right)
UNITS = "meters"               # units the .geo and .txt are drawn in

ADD_CORNER = True              # the wing-body root double bounce
FOLD_LENGTH_M = 0.20           # b: the root/fold length that is actually welded
                               # (here: the root chord = the airfoil chord)
FACE_WIDTH_M = 0.05            # a: how far up the wing the double bounce reaches
CANT_DEG = 0.0                 # how far off square the root is (0 = right angle)
N_BODY = None                  # None = DERIVE the body normal at the root from
                               # the actual BoR profile at that point.  Only
                               # override it if your root
                               # is not on a surface of revolution.
ROOT_SKIN_TOL_M = 2e-3         # also limited by ROOT_PHASE_TOL_DEG below
ROOT_PHASE_TOL_DEG = 15.0      # max two-way root-placement phase error

ANGLES_DEG = np.arange(0.0, 180.1, 5.0)   # 2-D cut angles for the airfoil solve
BODY_DIR = os.path.join("..", "2b_solve_body_hpc", "Body")
                              # HPC production default; use 2a.../Body locally
FORCE = False                  # True = re-solve the airfoil
# ────────────────────────────────────────────────────────────────────────────

from grid import (FREQUENCIES_GHZ, AZIMUTHS_DEG,                      # noqa: E402
                  ELEVATIONS_DEG, POLARIZATIONS)
from frame import (AXIS_AZ_DEG, AXIS_EL_DEG, ROLL_DEG,                # noqa: E402
                   scale_for, to_axis_frame)
from components import keep_pols, tag_component                       # noqa: E402
from feature_sum import (export_radar_grim, sum_features,             # noqa: E402
                         directions_from_aspect_roll,
                         surface_of_revolution_distance, _load_grim,
                         geometry_input_fingerprint,
                         verify_body_artifact_bundle)
from geometry_io import parse_geometry, build_geometry_snapshot        # noqa: E402
from grim_io import export_result_to_grim                             # noqa: E402
from line_expand import dbsm, surface_of_revolution_normal            # noqa: E402
from rcs_solver import solve_monostatic_rcs_2d                        # noqa: E402
from workflow_provenance import (backend_source_paths,                # noqa: E402
                                 runtime_environment_fingerprint,
                                 write_artifact_in_progress)

C0 = 299_792_458.0
SCALE = scale_for(UNITS)
AIRFOIL_CACHE_SCHEMA = "ghost.workflow.wing-airfoil-cache.v1"
OUTPUT_SCHEMA = "ghost.workflow.wing-output-provenance.v1"


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
    paths = [os.path.abspath(__file__),
             os.path.join(root, "grid.py"),
             os.path.join(BACKEND, "frame.py")]
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


def _read_manifest(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(
            f"{path} is missing or unreadable ({exc}). Existing wing solver "
            "artifacts were preserved; set FORCE = True to rebuild.") from exc


def _refuse_unexpected_grims(folder, expected_names, label, pattern="*.grim"):
    existing = {
        os.path.basename(p) for p in glob.glob(os.path.join(folder, pattern))
    }
    stale = sorted(existing - set(expected_names))
    if stale:
        raise SystemExit(
            f"{label} contains unexpected stale .grim output(s):\n  "
            + "\n  ".join(stale)
            + "\nThey were not deleted. Move them out of this workflow "
              "directory and rerun.")


def _here(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def _profile_path():
    return os.path.join(_here(BODY_DIR), "body_profile.csv")


def solve_airfoil():
    """The airfoil's FULL 2-D amplitude, both polarizations, every frequency.

    Not a delta: a wing is its own object, so nothing is subtracted.  One file
    per polarization, each spanning all frequencies -- which is the shape
    load_coefficients_from_grim wants.
    """
    geos = sorted(glob.glob(os.path.join(HERE, "*_airfoil.geo")))
    if len(geos) != 1:
        raise SystemExit(f"put exactly one <name>_airfoil.geo here -- found "
                         f"{[os.path.basename(g) for g in geos]}.")
    out = [os.path.join(HERE, f"wing_{p}.grim") for p in ("TM", "TE")]
    expected_names = [os.path.basename(p) for p in out]
    _refuse_unexpected_grims(
        HERE, expected_names, "3b_add_wing/", pattern="wing_*.grim")
    signature_payload = {
        "schema": AIRFOIL_CACHE_SCHEMA,
        "geometry_input_sha256":
            geometry_input_fingerprint(geos[0], UNITS),
        "geometry_units": str(UNITS).strip().lower(),
        "frequencies_ghz": sorted(float(f) for f in FREQUENCIES_GHZ),
        "angles_deg": [float(a) for a in ANGLES_DEG],
        "polarizations": ["TM", "TE"],
        "solver_source_sha256": _source_fingerprint(),
        "runtime_environment_sha256": runtime_environment_fingerprint(),
        "expected_outputs": expected_names,
    }
    run_sha256 = _canonical_sha256(signature_payload)
    manifest_path = os.path.join(HERE, "wing_cache_manifest.json")
    any_cache = os.path.exists(manifest_path) or any(
        os.path.exists(p) for p in out)
    if not FORCE and any_cache:
        missing = [os.path.basename(p) for p in out if not os.path.isfile(p)]
        if missing:
            raise SystemExit(
                f"the wing airfoil cache is incomplete (missing {missing}). "
                "Existing files were preserved; set FORCE = True to rebuild.")
        manifest = _read_manifest(manifest_path)
        if (manifest.get("schema") != AIRFOIL_CACHE_SCHEMA
                or manifest.get("status") != "complete"
                or manifest.get("run_sha256") != run_sha256
                or set(manifest.get("expected_outputs", []))
                != set(expected_names)):
            raise SystemExit(
                "wing_cache_manifest.json does not match the current airfoil "
                "geometry/material tables, units, frequency/angle grid, or "
                "solver source. Existing files were preserved; set FORCE = "
                "True to rebuild.")
        recorded = manifest.get("output_sha256", {})
        bad = [
            os.path.basename(p) for p in out
            if recorded.get(os.path.basename(p)) != _sha256_file(p)
        ]
        if bad:
            raise SystemExit(
                "wing airfoil cache bytes do not match their manifest: "
                + ", ".join(bad)
                + ". Existing files were preserved; set FORCE = True to "
                  "rebuild.")
        print("         reusing wing_TM.grim / wing_TE.grim "
              "(exact manifest verified)")
        return out

    in_progress = dict(signature_payload)
    in_progress.update(run_sha256=run_sha256, status="in_progress")
    _write_json_atomic(manifest_path, in_progress)
    with open(geos[0]) as fh:
        snap = build_geometry_snapshot(*parse_geometry(fh.read()))
    snap["source_path"] = os.path.abspath(geos[0])
    n = len(snap["segments"][0]["point_pairs"])
    print(f"         solving {os.path.basename(geos[0])} ({n} segments, both "
          f"pols, {len(FREQUENCIES_GHZ)} freq) ...")
    paths = []
    for pol in ("TM", "TE"):
        r = solve_monostatic_rcs_2d(snap, list(FREQUENCIES_GHZ),
                                    list(ANGLES_DEG), pol,
                                    geometry_units=UNITS,
                                    strict_quality_gate=True,
                                    compute_condition_number=True)
        paths.append(export_result_to_grim(
            r, os.path.join(HERE, f"wing_{pol}"),
            source_path=os.path.abspath(geos[0]),
            history=(f"step 3b wing airfoil pol={pol}; "
                     f"cache_input_sha256={run_sha256}"))[0])
    if (
        geometry_input_fingerprint(geos[0], UNITS)
        != signature_payload["geometry_input_sha256"]
        or _source_fingerprint()
        != signature_payload["solver_source_sha256"]
        or runtime_environment_fingerprint()
        != signature_payload["runtime_environment_sha256"]
    ):
        raise SystemExit(
            "the airfoil geometry/material input, solver source, or numerical "
            "runtime changed during the wing solve. Outputs were preserved "
            "but no complete cache manifest was written; rerun with FORCE = "
            "True from one stable state.")
    manifest = dict(signature_payload)
    manifest.update(
        run_sha256=run_sha256,
        output_sha256={
            os.path.basename(p): _sha256_file(p) for p in paths
        },
        status="complete")
    _write_json_atomic(manifest_path, manifest)
    print("         wrote wing_cache_manifest.json")
    return paths


def profile():
    p = _profile_path()
    if not os.path.exists(p):
        raise SystemExit(f"no {p} -- run 2a_solve_body_local (or 2b) first, "
                         f"or point the BODY_DIR knob at the body you want.")
    try:
        verify_body_artifact_bundle(_here(BODY_DIR))
    except ValueError as exc:
        raise SystemExit(f"uncommitted or changed body bundle: {exc}") from exc
    return np.loadtxt(p, delimiter=",", skiprows=1)


def _output_provenance_payload(wing, span_txt, expected_component_names):
    airfoil_manifest_path = os.path.join(HERE, "wing_cache_manifest.json")
    airfoil_manifest = _read_manifest(airfoil_manifest_path)
    return {
        "schema": OUTPUT_SCHEMA,
        "name": str(NAME),
        "airfoil_cache_manifest_sha256":
            _sha256_file(airfoil_manifest_path),
        "airfoil_cache_run_sha256":
            str(airfoil_manifest.get("run_sha256", "")),
        "airfoil_geometry_input_sha256":
            str(airfoil_manifest.get("geometry_input_sha256", "")),
        "airfoil_cache_outputs_sha256": {
            os.path.basename(p): _sha256_file(p) for p in wing
        },
        "wing_span_sha256": _sha256_file(span_txt),
        "body_profile_sha256": _sha256_file(_profile_path()),
        "units": str(UNITS).strip().lower(),
        "wing_normal_cad": [float(x) for x in WING_NORMAL],
        "add_corner": bool(ADD_CORNER),
        "fold_length_m": float(FOLD_LENGTH_M),
        "face_width_m": float(FACE_WIDTH_M),
        "cant_deg": float(CANT_DEG),
        "n_body_override_cad": (
            None if N_BODY is None else [float(x) for x in N_BODY]),
        "root_skin_tol_m": float(ROOT_SKIN_TOL_M),
        "root_phase_tol_deg": float(ROOT_PHASE_TOL_DEG),
        "frequencies_ghz": [float(x) for x in FREQUENCIES_GHZ],
        "angles_deg": [float(x) for x in ANGLES_DEG],
        "azimuths_deg": [float(x) for x in AZIMUTHS_DEG],
        "elevations_deg": [float(x) for x in ELEVATIONS_DEG],
        "polarizations": [str(x) for x in POLARIZATIONS],
        "axis_az_deg": float(AXIS_AZ_DEG),
        "axis_el_deg": float(AXIS_EL_DEG),
        "roll_deg": float(ROLL_DEG),
        "workflow_source_sha256": _source_fingerprint(),
        "runtime_environment_sha256": runtime_environment_fingerprint(),
        "expected_outputs": sorted(expected_component_names),
    }


def main():
    print(f"STEP 3b  wing assembly '{NAME}' at {list(FREQUENCIES_GHZ)} GHz")
    output_dir = os.path.join(HERE, "Output")
    os.makedirs(output_dir, exist_ok=True)
    expected_component_names = {f"{NAME}.grim"}
    if ADD_CORNER:
        expected_component_names.add(f"{NAME}_corner_estimate.grim")
    _refuse_unexpected_grims(
        output_dir, expected_component_names, "Output/")

    wing = solve_airfoil()
    gen = profile()

    span_txt = os.path.join(HERE, "wing_span.txt")
    if not os.path.exists(span_txt):
        raise SystemExit("no wing_span.txt -- two rows, root then tip: x,y,z "
                         f"in {UNITS}, CAD frame (+y nose, +x right, +z up).")
    span_cad = np.loadtxt(span_txt, delimiter=",", skiprows=1) * SCALE
    if np.shape(span_cad) != (2, 3):
        raise SystemExit(f"wing_span.txt must be exactly two rows (root, tip) "
                         f"of x,y,z -- got shape {np.shape(span_cad)}.")
    span = to_axis_frame(span_cad)[None, :, :]        # (1, 2, 3), solver frame
    n_wing = tuple(to_axis_frame(np.asarray(WING_NORMAL, float)))
    L = float(np.linalg.norm(span[0, 1] - span[0, 0]))
    print(f"         span {L:.3f} m root->tip, face normal {WING_NORMAL} (CAD)")

    place = [{"delta": wing, "perimeter": span, "normal": n_wing}]

    corners = []
    if ADD_CORNER:
        # The fold is the line where the wing is actually welded to the body: it
        # starts at the ROOT and runs along the body axis (the root chord).
        # The body normal comes from the nearest segment of the actual BoR
        root = np.asarray(span[0, 0], float)          # solver frame
        axis = np.array([0.0, 0.0, 1.0])              # solver z = the body axis
        fold = np.array([root - axis * FOLD_LENGTH_M / 2,
                         root + axis * FOLD_LENGTH_M / 2])
        radial = np.array([root[0], root[1], 0.0])
        if np.linalg.norm(radial) < 1e-12:
            raise SystemExit("the wing root sits ON the axis, so there is no "
                             "body normal there -- move the root onto the skin, "
                             "or set ADD_CORNER = False.")
        nb = surface_of_revolution_normal(gen)(root[None, :])[0]
        if N_BODY is not None:
            nb = to_axis_frame(np.asarray(N_BODY, float))
            nb = nb / np.linalg.norm(nb)
        rho_root = float(np.linalg.norm(radial))
        off = float(surface_of_revolution_distance(gen, root[None, :])[0])
        lam_min = C0 / (float(np.max(FREQUENCIES_GHZ)) * 1e9)
        phase_limit_m = ROOT_PHASE_TOL_DEG * lam_min / 720.0
        root_limit_m = min(ROOT_SKIN_TOL_M, phase_limit_m)
        if off > root_limit_m:
            raise SystemExit(
                f"the wing root is {off*1e3:.1f} mm off the nearest body-skin "
                f"segment ({720.0*off/lam_min:.1f} deg worst-case two-way "
                f"phase at {float(np.max(FREQUENCIES_GHZ)):g} GHz; allowed "
                f"{root_limit_m*1e3:.3f} mm / {ROOT_PHASE_TOL_DEG:g} deg).\n"
                "  The corner assumes a root on the skin; check UNITS, the CAD "
                "frame, BODY_DIR, and wing_span.txt.")
        g = math.radians(CANT_DEG)
        nw = np.asarray(n_wing, float)
        if CANT_DEG:
            # rotate the wing face off square, in the plane spanned by the two
            # normals -- the cant is a property of the JOINT, not of the wing
            nw = math.cos(g) * nw + math.sin(g) * nb
            nw /= np.linalg.norm(nw)
        corners = [{"fold": fold, "n_wing": tuple(nw), "n_body": tuple(nb),
                    "face_width": FACE_WIDTH_M}]
        lam = C0 / (float(FREQUENCIES_GHZ[-1]) * 1e9)
        sig0 = (8 * math.pi * FACE_WIDTH_M ** 2 * FOLD_LENGTH_M ** 2 / lam ** 2)
        print(f"         corner: fold b = {FOLD_LENGTH_M:.3f} m "
              f"({FOLD_LENGTH_M/lam:.1f} lam), face a = {FACE_WIDTH_M:.3f} m, "
              f"cant {CANT_DEG:g} deg")
        print(f"                 root at rho = {rho_root:.4f} m, body normal "
              f"{np.round(nb, 3).tolist()} (solver frame)")
        print(f"                 textbook ceiling 8*pi*a^2*b^2/lam^2 = "
              f"{dbsm(sig0):+.1f} dBsm at {FREQUENCIES_GHZ[-1]:g} GHz")
    else:
        print("         corner: OFF -- the isolated wing is an incomplete "
              "reference near the root")

    provenance_payload = _output_provenance_payload(
        wing, span_txt, expected_component_names)
    provenance_sha256 = _canonical_sha256(provenance_payload)
    write_artifact_in_progress(
        output_dir,
        OUTPUT_SCHEMA,
        sorted(expected_component_names),
        dict(provenance_payload, run_sha256=provenance_sha256),
        manifest_name="provenance_manifest.json",
    )

    # ---- the component file step 4 reads -----------------------------------
    out = export_radar_grim(
        os.path.join(output_dir, NAME),
        bor_result=None,                   # wing alone; step 4 adds the body
        placements=place, generatrix=gen, corners=(),
        frequencies_ghz=FREQUENCIES_GHZ,
        azimuths_deg=AZIMUTHS_DEG, elevations_deg=ELEVATIONS_DEG,
        axis_az_deg=AXIS_AZ_DEG, axis_el_deg=AXIS_EL_DEG,   # fixed by frame.py
        roll_deg=ROLL_DEG,
        history=(f"step 3b coherent wing '{NAME}'; "
                 f"provenance_sha256={provenance_sha256}"))
    keep_pols(out, POLARIZATIONS)
    tag_component(out, "coherent", note="wing only; line expansion is linear")

    g = _load_grim(out)
    pols = [str(x) for x in np.asarray(g["polarizations"]).ravel()]
    power = np.asarray(g["rcs_power"], float)
    pk = {p: dbsm(np.max(power[:, :, :, i])) for i, p in enumerate(pols)}
    print(f"\n         wrote Output/{NAME}.grim  peak "
          + "  ".join(f"{p} {v:+.1f}" for p, v in pk.items())
          + " dBsm   [coherent wing]")

    produced_grims = [out]
    if corners:
        corner_out = export_radar_grim(
            os.path.join(output_dir, f"{NAME}_corner_estimate"),
            bor_result=None, placements=[], generatrix=gen, corners=corners,
            frequencies_ghz=FREQUENCIES_GHZ,
            azimuths_deg=AZIMUTHS_DEG, elevations_deg=ELEVATIONS_DEG,
            axis_az_deg=AXIS_AZ_DEG, axis_el_deg=AXIS_EL_DEG,
            roll_deg=ROLL_DEG,
            history=(f"step 3b PO corner estimate for '{NAME}'; "
                     "internal double-bounce phase uncalibrated; "
                     f"provenance_sha256={provenance_sha256}"))
        keep_pols(corner_out, POLARIZATIONS)
        tag_component(
            corner_out, "power",
            note=("PO corner-only statistical estimate; internal double-bounce "
                  "phase is uncalibrated and must not be coherently summed"))
        cg = _load_grim(corner_out)
        cp = np.asarray(cg["rcs_power"], float)
        cpk = {p: dbsm(np.max(cp[:, :, :, i])) for i, p in enumerate(pols)}
        print(f"         wrote Output/{NAME}_corner_estimate.grim  peak "
              + "  ".join(f"{p} {v:+.1f}" for p, v in cpk.items())
              + " dBsm   [power-only estimate]")
        produced_grims.append(corner_out)

    # ---- what did each piece contribute? -----------------------------------
    dirs, asp, rol = directions_from_aspect_roll(np.arange(30.0, 150.1, 10.0),
                                                [0.0, 45.0, 90.0])
    f0 = float(FREQUENCIES_GHZ[-1])
    kw = dict(generatrix=gen, mode="coherent")
    only_wing = sum_features(None, place, dirs, f0, **kw)
    only_corner = (sum_features(None, [], dirs, f0, corners=corners, **kw)
                   if corners else None)
    for w in (only_corner or {}).get("warnings", []):
        print("         WARNING:", w)
    est_vv = only_wing["sigma_vv"] + (
        only_corner["sigma_vv"] if only_corner is not None else 0.0)
    est_vh = only_wing["sigma_vh"] + (
        only_corner["sigma_vh"] if only_corner is not None else 0.0)
    est_dbsm_vv, est_dbsm_vh = dbsm(est_vv), dbsm(est_vh)

    rows = [("aspect_deg", "roll_deg", "wing_vv_dBsm", "corner_vv_dBsm",
             "expected_power_vv_dBsm", "expected_power_crosspol_vh_dBsm")]
    cz = (only_corner["dbsm_vv"] if only_corner is not None
          else np.full_like(only_wing["dbsm_vv"], -200.0))
    for i in range(len(asp)):
        rows.append((f"{asp[i]:g}", f"{rol[i]:g}",
                     f"{only_wing['dbsm_vv'][i]:.3f}", f"{cz[i]:.3f}",
                     f"{est_dbsm_vv[i]:.3f}", f"{est_dbsm_vh[i]:.3f}"))
    summary_path = os.path.join(HERE, "wing_dbsm.csv")
    with open(summary_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"         wrote wing_dbsm.csv  (at {f0:g} GHz)")

    final_component_names = {
        os.path.basename(p)
        for p in glob.glob(os.path.join(output_dir, "*.grim"))
    }
    if final_component_names != expected_component_names:
        raise SystemExit(
            "Output/ does not contain the exact expected component set after "
            f"generation (expected {sorted(expected_component_names)}, got "
            f"{sorted(final_component_names)}). Files were preserved.")
    output_sha256 = {
        os.path.basename(path): _sha256_file(path)
        for path in produced_grims
    }
    output_sha256["../wing_dbsm.csv"] = _sha256_file(summary_path)
    try:
        current_payload = _output_provenance_payload(
            wing, span_txt, expected_component_names
        )
    except Exception as exc:
        raise SystemExit(
            "a wing input became unreadable while step 3b was running. "
            "Outputs were preserved but no complete provenance manifest was "
            "written."
        ) from exc
    if _canonical_sha256(current_payload) != provenance_sha256:
        raise SystemExit(
            "a wing input, body profile, grid, workflow source, or numerical "
            "runtime changed while step 3b was running. Outputs were preserved "
            "but no complete provenance manifest was written; rerun from one "
            "stable state.")
    manifest = dict(provenance_payload)
    manifest.update(
        run_sha256=provenance_sha256,
        output_sha256=output_sha256,
        status="complete")
    _write_json_atomic(
        os.path.join(output_dir, "provenance_manifest.json"), manifest)
    print("         wrote Output/provenance_manifest.json")

    print(f"\n         roll   wing peak   corner peak   expected-power estimate  (dBsm, "
          f"{f0:g} GHz)")
    for r in sorted(set(rol)):
        m = rol == r
        w_pk, c_pk = float(np.max(only_wing["dbsm_vv"][m])), float(np.max(cz[m]))
        a_pk = float(np.max(est_dbsm_vv[m]))
        note = ""
        if w_pk < -199:
            note = "   <- wing EDGE-ON (silent, and correct)"
        elif corners and c_pk < -199:
            note = "   <- corner has only one face lit (silent, and correct)"
        print(f"         {r:4.0f}   {w_pk:+9.1f}   {c_pk:+11.1f}   "
              f"{a_pk:+13.1f}{note}")

    print("\nCHECK    face-on, a flat-plate wing peaks broadside to the span and "
          "falls off as\n         sinc^2.  Edge-on it is -200 dBsm.  A corner "
          "needs BOTH faces lit, so it\n         peaks on the bisector of "
          "n_wing and n_body.  Silent everywhere usually\n         means "
          "WING_NORMAL or N_BODY points the wrong way, not that it is small.\n"
          "         The assembly column is an expected-power estimate, not a "
          "phase-resolved field.")
    print("NEXT     4_combine")


if __name__ == "__main__":
    main()
