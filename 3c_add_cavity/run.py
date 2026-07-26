#!/usr/bin/env python3
"""
STEP 3c -- ADD COMPACT 3-D FEATURES (cavities, fasteners, holes)
================================================================

WHAT IT DOES
  Some features cannot be line-expanded.  A blind cavity, a hole, an inlet lip
  is compact, resonant and genuinely 3-D -- a 2-D slice through it is a TRENCH,
  not a hole -- and sitting off-axis it is not axisymmetric either.  So it is
  handled the other way round:

      somebody solves it ONCE in a full 3-D code as (featured - clean), stores
      the complex difference as an az/el/frequency/pol pattern in the FEATURE's
      own frame, and this step puts that pattern at one or many body coordinates.

  Per look direction it rotates the look into the cavity frame, interpolates
  the stored pattern, rotates the polarizations back, applies the placement
  phase for where the cavity sits, and hides it when the aperture faces away.

  THE CONVENTION YOUR 3-D CODE MUST MATCH -- get this wrong and the answer is
  silently rotated, not obviously broken:
    * az/el are CAVITY-FRAME spherical angles with +z the aperture's outward
      normal;
    * VV = theta-pol and HH = phi-pol about that normal;
    * phase origin AT the cavity and exp(+jwt); placement then applies the
      two-way exp(+2jk d.r) translation phase;
    * the pattern explicitly carries the convention metadata emitted by
      Backend.feature_sum.point_pattern_convention_metadata().

  WHAT IS DROPPED: body-cavity mutual coupling.  Fine for a cavity recessed in
  a large smooth area; weak next to a strong edge.

INPUTS
  <name>.grim                  the 3-D delta pattern from your external solver,
                               in the feature's own frame.  The one shipped here
                               is FABRICATED for the demo -- replace it.  It
                               must carry VV/HH/VH, raw complex F,
                               rcs_power=4*pi*|F|^2, sigma_3d units, and
                               rcs_domain='delta', the explicit convention
                               metadata, a complete continuous 360-degree
                               azimuth seam, and elevation support for every
                               requested lit look; incomplete or inconsistent
                               patterns are refused.
  <type>.csv                   optional fastener placement table for one pattern:
                                  x,y,z
                               or x,y,z,nx,ny,nz
                               or x,y,z,nx,ny,nz,rx,ry,rz
                               Values are in the CAD frame and UNITS below.
  BODY_DIR/body_profile.csv    from step 2a/2b, to check every feature is on skin

OUTPUTS
  Output/<name>.grim           all configured compact features coherently summed,
                               az x el x freq x pol, combine_role='coherent'
  Output/provenance_manifest.json
                               exact committed component name, bytes, pattern,
                               body profile, grid, and workflow source
  <name>_dbsm.csv              aggregate feature response over a diagnostic
                               aspect and roll sweep (legacy mode retains
                               cavity_dbsm.csv)

KNOBS (below)
  FASTENER_TYPES, NAME, PATTERN, LOCATION_CAD, APERTURE_NORMAL, ROLL_REF,
  UNITS, BODY_DIR, SKIN_TOL_M, SKIN_PHASE_TOL_DEG, NORMAL_TOL_DEG

  The output grid lives in ../grid.py, shared with 3a and 3b.

  Leave FASTENER_TYPES empty for the original single-cavity mode.  Otherwise,
  each entry pairs one reusable pattern with one CSV of placement coordinates.
  Coordinates and vectors use the CAD frame: +y nose, +x right, +z up. Missing
  normals are derived from the BoR skin. Supplied normals are checked against it.

CHECK  the cavity MUST go silent (-200 dBsm) once its aperture faces away.  The
  sweep below looks from both sides for exactly that reason: a placement that
  never goes dark has its aperture normal wrong, and will read plausibly
  everywhere while being wrong everywhere.

NEXT  4_combine

    python3 run.py
"""

import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "Backend")
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.dirname(HERE))                # for ../grid.py

# ─── KNOBS ──────────────────────────────────────────────────────────────────
NAME = "cavity_stbd"          # what the component file is called
PATTERN = ""                  # "" = the only *.grim in this folder; else a name

LOCATION_CAD = (0.030, 0.020, 0.0)   # where it sits, CAD frame, in UNITS
APERTURE_NORMAL = None        # None = derive from the actual BoR skin profile
ROLL_REF = (0.0, 1.0, 0.0)    # CAD direction fixing the cavity frame's az zero
                              # -- must match how your 3-D solve was set up

# Multi-fastener mode. One external 3-D delta pattern is reused at every row in
# its coordinate CSV. Leave this empty to retain the single-cavity knobs above.
#
# FASTENER_TYPES = [
#     {
#         "name": "flush_rivet",
#         "pattern": os.path.join("Patterns", "flush_rivet.grim"),
#         "coordinates": os.path.join("Fasteners", "flush_rivet.csv"),
#         "roll_ref": (0.0, 1.0, 0.0),  # default CAD-frame azimuth-zero vector
#     },
#     {
#         "name": "bolt_head",
#         "pattern": os.path.join("Patterns", "bolt_head.grim"),
#         "coordinates": os.path.join("Fasteners", "bolt_head.csv"),
#         "roll_ref": (0.0, 1.0, 0.0),
#     },
# ]
FASTENER_TYPES = []

UNITS = "meters"
SKIN_TOL_M = 2e-3             # how far off the skin the cavity may sit
SKIN_PHASE_TOL_DEG = 15.0     # max two-way placement-phase error
NORMAL_TOL_DEG = 15.0          # supplied normal vs derived skin normal
BODY_DIR = os.path.join("..", "2b_solve_body_hpc", "Body")
                              # HPC production default; use 2a.../Body locally
# ────────────────────────────────────────────────────────────────────────────

from grid import (FREQUENCIES_GHZ, AZIMUTHS_DEG,                      # noqa: E402
                  ELEVATIONS_DEG, POLARIZATIONS)
from frame import (AXIS_AZ_DEG, AXIS_EL_DEG, ROLL_DEG,                # noqa: E402
                   scale_for, to_axis_frame)
from components import keep_pols, tag_component                       # noqa: E402
from feature_sum import (export_radar_grim, sum_features,             # noqa: E402
                         directions_from_aspect_roll,
                         surface_of_revolution_distance, _load_grim,
                         prepare_point_pattern,
                         verify_body_artifact_bundle)
from line_expand import dbsm, surface_of_revolution_normal            # noqa: E402
from workflow_provenance import (backend_source_fingerprint,          # noqa: E402
                                 runtime_environment_fingerprint,
                                 sha256_file, stable_json_fingerprint,
                                 write_artifact_in_progress,
                                 write_artifact_manifest)

SCALE = scale_for(UNITS)
OUTPUT_SCHEMA = "ghost.workflow.cavity-output-provenance.v1"


def _here(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def _normalized_header(value):
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


_COLUMN_ALIASES = {
    "x": ("x",),
    "y": ("y",),
    "z": ("z",),
    "nx": ("nx", "xnormal", "normalx"),
    "ny": ("ny", "ynormal", "normaly"),
    "nz": ("nz", "znormal", "normalz"),
    "rx": ("rx", "xroll", "rollx", "xref", "refx"),
    "ry": ("ry", "yroll", "rolly", "yref", "refy"),
    "rz": ("rz", "zroll", "rollz", "zref", "refz"),
}


def read_placement_csv(path):
    """Read CAD-frame x/y/z plus optional normal and roll-reference vectors."""
    source = _here(path)
    try:
        with open(source, newline="", encoding="utf-8-sig") as stream:
            raw_rows = [
                (line_number, [cell.strip() for cell in row])
                for line_number, row in enumerate(csv.reader(stream), 1)
                if row and any(cell.strip() for cell in row)
                and not row[0].lstrip().startswith("#")
            ]
    except OSError as exc:
        raise SystemExit(f"cannot read fastener coordinates {source}: {exc}") from exc
    if not raw_rows:
        raise SystemExit(f"{source}: placement CSV is empty.")

    first_number, first = raw_rows[0]
    try:
        [float(value) for value in first]
        header = None
    except ValueError:
        header = first
        raw_rows = raw_rows[1:]
    if not raw_rows:
        raise SystemExit(f"{source}: placement CSV has no data rows.")

    if header is None:
        width = len(first)
        if width not in (3, 6, 9):
            raise SystemExit(
                f"{source}:{first_number}: headerless rows require 3, 6, or "
                f"9 columns; found {width}."
            )
        keys = ("x", "y", "z", "nx", "ny", "nz", "rx", "ry", "rz")[:width]
        indices = {key: index for index, key in enumerate(keys)}
    else:
        normalized = [_normalized_header(value) for value in header]
        indices = {}
        for key, aliases in _COLUMN_ALIASES.items():
            matches = [
                index for index, value in enumerate(normalized)
                if value in aliases
            ]
            if len(matches) > 1:
                raise SystemExit(
                    f"{source}: header provides {key!r} more than once."
                )
            if matches:
                indices[key] = matches[0]
        missing_xyz = [key for key in ("x", "y", "z") if key not in indices]
        if missing_xyz:
            raise SystemExit(
                f"{source}: header is missing coordinate columns {missing_xyz}."
            )
        for group, label in (
            (("nx", "ny", "nz"), "normal"),
            (("rx", "ry", "rz"), "roll-reference"),
        ):
            present = [key in indices for key in group]
            if any(present) and not all(present):
                raise SystemExit(
                    f"{source}: provide all three {label} columns or none."
                )

    placements = []
    for line_number, row in raw_rows:
        if max(indices.values()) >= len(row):
            raise SystemExit(
                f"{source}:{line_number}: row has too few columns."
            )
        try:
            values = {
                key: float(row[index]) for key, index in indices.items()
            }
        except ValueError as exc:
            raise SystemExit(
                f"{source}:{line_number}: placement values must be numeric."
            ) from exc
        if not np.all(np.isfinite(list(values.values()))):
            raise SystemExit(
                f"{source}:{line_number}: placement contains NaN or infinity."
            )
        entry = {
            "line": line_number,
            "location_cad": tuple(values[key] for key in ("x", "y", "z")),
            "normal_cad": (
                tuple(values[key] for key in ("nx", "ny", "nz"))
                if "nx" in values else None
            ),
            "roll_ref_cad": (
                tuple(values[key] for key in ("rx", "ry", "rz"))
                if "rx" in values else None
            ),
        }
        placements.append(entry)
    return source, placements


def _unit_vector(value, label):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise SystemExit(f"{label} must be one finite 3-vector.")
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1e-12:
        raise SystemExit(f"{label} cannot be zero.")
    return vector / magnitude


def _skin_limit_m():
    skin_tolerance = float(SKIN_TOL_M)
    phase_tolerance = float(SKIN_PHASE_TOL_DEG)
    normal_tolerance = float(NORMAL_TOL_DEG)
    if (
        not np.isfinite(skin_tolerance)
        or skin_tolerance < 0.0
        or not np.isfinite(phase_tolerance)
        or phase_tolerance < 0.0
        or not np.isfinite(normal_tolerance)
        or normal_tolerance < 0.0
        or normal_tolerance > 90.0
    ):
        raise SystemExit(
            "SKIN_TOL_M and SKIN_PHASE_TOL_DEG must be finite and "
            "non-negative; NORMAL_TOL_DEG must be finite and between 0 and 90."
        )
    lam_min = 299_792_458.0 / (float(np.max(FREQUENCIES_GHZ)) * 1e9)
    phase_limit_m = phase_tolerance * lam_min / 720.0
    return min(skin_tolerance, phase_limit_m), lam_min


def _validated_point(gen, pattern, location_cad, normal_cad, roll_ref_cad, label):
    loc = to_axis_frame(np.asarray(location_cad, float) * SCALE)
    off = float(surface_of_revolution_distance(gen, loc[None, :])[0])
    if not np.isfinite(off):
        raise SystemExit(f"{label}: body-skin distance is nonfinite.")
    skin_limit_m, lam_min = _skin_limit_m()
    if off > skin_limit_m:
        raise SystemExit(
            f"{label}: coordinate is {off*1e3:.3f} mm off the body skin "
            f"({720.0*off/lam_min:.1f} deg worst-case two-way phase at "
            f"{float(np.max(FREQUENCIES_GHZ)):g} GHz; allowed "
            f"{skin_limit_m*1e3:.3f} mm / {SKIN_PHASE_TOL_DEG:g} deg)."
        )
    derived = _unit_vector(
        surface_of_revolution_normal(gen)(loc[None, :])[0],
        f"{label} derived skin normal",
    )
    if normal_cad is None:
        normal = derived
        normal_source = "derived"
    else:
        normal = _unit_vector(
            to_axis_frame(np.asarray(normal_cad, float)),
            f"{label} normal",
        )
        cosine = float(np.clip(normal @ derived, -1.0, 1.0))
        difference_deg = float(np.degrees(np.arccos(cosine)))
        normal_tolerance = float(NORMAL_TOL_DEG)
        if difference_deg > normal_tolerance:
            raise SystemExit(
                f"{label}: supplied normal is {difference_deg:.2f} deg from "
                f"the outward BoR skin normal; allowed {NORMAL_TOL_DEG:g} deg."
            )
        normal_source = "supplied"
    roll = _unit_vector(
        to_axis_frame(np.asarray(roll_ref_cad, float)),
        f"{label} roll_ref",
    )
    if np.linalg.norm(roll - (roll @ normal) * normal) <= 1e-9:
        raise SystemExit(f"{label}: roll_ref is parallel to the surface normal.")
    return {
        "pattern": pattern,
        "location": tuple(loc),
        "aperture_normal": tuple(normal),
        "roll_ref": tuple(roll),
    }, {
        "location_cad": [float(value) for value in location_cad],
        "normal_cad": (
            None if normal_cad is None
            else [float(value) for value in normal_cad]
        ),
        "normal_source": normal_source,
        "roll_ref_cad": [float(value) for value in roll_ref_cad],
        "skin_offset_m": off,
    }


def build_points(gen):
    """Return placed point dictionaries and provenance source records."""
    if not FASTENER_TYPES:
        pattern = find_pattern()
        point, placement = _validated_point(
            gen, pattern, LOCATION_CAD, APERTURE_NORMAL, ROLL_REF,
            "single compact feature",
        )
        return [point], [{
            "name": str(NAME),
            "pattern": pattern,
            "coordinates": None,
            "placements": [placement],
        }], True

    if not isinstance(FASTENER_TYPES, (list, tuple)):
        raise SystemExit("FASTENER_TYPES must be a list of configuration mappings.")
    points = []
    sources = []
    names = set()
    occupied = {}
    for type_index, config in enumerate(FASTENER_TYPES):
        if not isinstance(config, dict):
            raise SystemExit(f"FASTENER_TYPES[{type_index}] must be a mapping.")
        name = str(config.get("name", "")).strip()
        if not name or name in names:
            raise SystemExit(
                f"FASTENER_TYPES[{type_index}] needs a unique non-empty name."
            )
        names.add(name)
        pattern_value = str(config.get("pattern", "")).strip()
        coordinates_value = str(config.get("coordinates", "")).strip()
        if not pattern_value or not coordinates_value:
            raise SystemExit(
                f"fastener type {name!r} requires pattern and coordinates."
            )
        pattern = _here(pattern_value)
        if not os.path.isfile(pattern):
            raise SystemExit(f"fastener type {name!r}: no pattern {pattern}.")
        coordinate_path, rows = read_placement_csv(coordinates_value)
        default_roll = config.get("roll_ref", ROLL_REF)
        records = []
        for row in rows:
            label = f"{name} {os.path.basename(coordinate_path)}:{row['line']}"
            roll_ref = (
                row["roll_ref_cad"]
                if row["roll_ref_cad"] is not None else default_roll
            )
            point, record = _validated_point(
                gen, pattern, row["location_cad"], row["normal_cad"],
                roll_ref, label,
            )
            key = tuple(np.round(np.asarray(point["location"], float), 12))
            if key in occupied:
                raise SystemExit(
                    f"{label}: duplicate coordinate already used by {occupied[key]}."
                )
            occupied[key] = label
            points.append(point)
            records.append(dict(record, csv_line=int(row["line"])))
        sources.append({
            "name": name,
            "pattern": pattern,
            "coordinates": coordinate_path,
            "default_roll_ref_cad": [float(value) for value in default_roll],
            "placements": records,
        })
    if not points:
        raise SystemExit("FASTENER_TYPES did not provide any placements.")
    return points, sources, False


def _source_fingerprint():
    root = os.path.dirname(HERE)
    return backend_source_fingerprint(
        BACKEND,
        {
            "3c_add_cavity/run.py": os.path.abspath(__file__),
            "grid.py": os.path.join(root, "grid.py"),
        },
    )


def _compact_feature_provenance_payload(sources, expected_names):
    patterns = {
        os.path.relpath(source["pattern"], HERE):
            sha256_file(source["pattern"])
        for source in sources
    }
    coordinates = {
        os.path.relpath(source["coordinates"], HERE):
            sha256_file(source["coordinates"])
        for source in sources if source["coordinates"] is not None
    }
    return {
        "schema": OUTPUT_SCHEMA,
        "name": str(NAME),
        "mode": "fastener_catalog" if FASTENER_TYPES else "single",
        "patterns_sha256": patterns,
        "coordinate_tables_sha256": coordinates,
        "fastener_types": [
            {
                "name": source["name"],
                "pattern": os.path.relpath(source["pattern"], HERE),
                "coordinates": (
                    None if source["coordinates"] is None
                    else os.path.relpath(source["coordinates"], HERE)
                ),
                "default_roll_ref_cad": source.get("default_roll_ref_cad"),
                "placement_count": len(source["placements"]),
                "derived_normal_count": sum(
                    entry["normal_source"] == "derived"
                    for entry in source["placements"]
                ),
                "supplied_normal_count": sum(
                    entry["normal_source"] == "supplied"
                    for entry in source["placements"]
                ),
                "max_skin_offset_m": max(
                    entry["skin_offset_m"] for entry in source["placements"]
                ),
            }
            for source in sources
        ],
        "placement_count": sum(len(source["placements"]) for source in sources),
        "body_profile_sha256": sha256_file(
            os.path.join(_here(BODY_DIR), "body_profile.csv")
        ),
        "units": str(UNITS).strip().lower(),
        "skin_tol_m": float(SKIN_TOL_M),
        "skin_phase_tol_deg": float(SKIN_PHASE_TOL_DEG),
        "normal_tol_deg": float(NORMAL_TOL_DEG),
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


def _cavity_provenance_payload(pattern, expected_names):
    """Backward-compatible helper retained for focused provenance tests."""
    return {
        "schema": OUTPUT_SCHEMA,
        "name": str(NAME),
        "pattern_name": os.path.basename(pattern),
        "pattern_sha256": sha256_file(pattern),
        "body_profile_sha256": sha256_file(
            os.path.join(_here(BODY_DIR), "body_profile.csv")
        ),
        "location_cad": [float(value) for value in LOCATION_CAD],
        "aperture_normal_cad": (
            None if APERTURE_NORMAL is None
            else [float(value) for value in APERTURE_NORMAL]
        ),
        "roll_ref_cad": [float(value) for value in ROLL_REF],
        "units": str(UNITS).strip().lower(),
        "skin_tol_m": float(SKIN_TOL_M),
        "skin_phase_tol_deg": float(SKIN_PHASE_TOL_DEG),
        "normal_tol_deg": float(NORMAL_TOL_DEG),
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


def _refuse_unexpected_grims(output_dir, expected_names):
    actual = {
        os.path.basename(path)
        for path in glob.glob(os.path.join(output_dir, "*.grim"))
    }
    stale = sorted(actual - set(expected_names))
    if stale:
        raise SystemExit(
            "Output/ contains unexpected stale compact-feature component(s):\n  "
            + "\n  ".join(stale)
            + "\nThey were preserved. Move them out of Output/ before "
              "building this exact cavity output."
        )


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
    return np.loadtxt(p, delimiter=",", skiprows=1)


def find_pattern():
    if PATTERN:
        p = os.path.join(HERE, PATTERN)
        if not os.path.exists(p):
            raise SystemExit(f"no {PATTERN} here.")
        return p
    cands = sorted(glob.glob(os.path.join(HERE, "*.grim")))
    if len(cands) != 1:
        raise SystemExit(
            f"put exactly one cavity pattern .grim in {HERE}, or name it with "
            f"the PATTERN knob -- found {[os.path.basename(c) for c in cands]}.")
    return cands[0]


def main():
    gen = profile()
    points, sources, legacy_single = build_points(gen)
    prepared_patterns = {}
    for pattern in sorted({point["pattern"] for point in points}):
        try:
            prepared_patterns[pattern] = prepare_point_pattern(pattern)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid compact-feature pattern {pattern}: {exc}") from exc
    solver_points = [
        dict(point, pattern=prepared_patterns[point["pattern"]])
        for point in points
    ]
    output_dir = os.path.join(HERE, "Output")
    os.makedirs(output_dir, exist_ok=True)
    expected_names = [f"{NAME}.grim"]
    _refuse_unexpected_grims(output_dir, expected_names)

    print(f"STEP 3c  compact component '{NAME}': {len(points)} placement(s), "
          f"{len(sources)} type(s)")
    for source in sources:
        offsets = [entry["skin_offset_m"] for entry in source["placements"]]
        print(
            f"         {source['name']:<20} {len(offsets):>7} placement(s)  "
            f"pattern={os.path.basename(source['pattern'])}  "
            f"max skin offset={max(offsets)*1e3:.3f} mm"
        )

    provenance_payload = _compact_feature_provenance_payload(
        sources, expected_names
    )
    provenance_sha256 = stable_json_fingerprint(provenance_payload)
    write_artifact_in_progress(
        output_dir,
        OUTPUT_SCHEMA,
        expected_names,
        dict(provenance_payload, run_sha256=provenance_sha256),
        manifest_name="provenance_manifest.json",
    )
    out = export_radar_grim(
        os.path.join(output_dir, NAME),
        bor_result=None,                # compact features alone; step 4 adds body
        placements=[], points=solver_points, generatrix=gen,
        frequencies_ghz=FREQUENCIES_GHZ,
        azimuths_deg=AZIMUTHS_DEG, elevations_deg=ELEVATIONS_DEG,
        axis_az_deg=AXIS_AZ_DEG, axis_el_deg=AXIS_EL_DEG,   # fixed by frame.py
        roll_deg=ROLL_DEG,
        history=(
            f"step 3c compact component '{NAME}': {len(points)} placements "
            f"across {[source['name'] for source in sources]}"
        ))
    keep_pols(out, POLARIZATIONS)
    # a placed 3-D pattern keeps its own phase and the placement phase, so it
    # sums against the other components exactly like a line-expanded feature
    tag_component(out, "coherent",
                  note=(
                      f"{len(points)} placed 3-D compact feature(s); individual "
                      "rotation and two-way placement phase are tracked"
                  ))

    g = _load_grim(out)
    pols = [str(x) for x in np.asarray(g["polarizations"]).ravel()]
    power = np.asarray(g["rcs_power"], float)
    pk = {p: dbsm(np.max(power[:, :, :, i])) for i, p in enumerate(pols)}
    print(f"\n         wrote Output/{NAME}.grim  peak "
          + "  ".join(f"{p} {v:+.1f}" for p, v in pk.items()) + " dBsm")

    # ---- aggregate diagnostic over both sides -------------------------------
    dirs, asp, rol = directions_from_aspect_roll(np.arange(30.0, 150.1, 20.0),
                                                 [0.0, 90.0, 180.0, 270.0])
    f0 = float(FREQUENCIES_GHZ[-1])
    alone = sum_features(None, [], dirs, f0, generatrix=gen, points=solver_points,
                         mode="coherent")
    rows = [("aspect_deg", "roll_deg", "feature_vv_dBsm", "feature_vh_dBsm")]
    for i in range(len(asp)):
        rows.append((f"{asp[i]:g}", f"{rol[i]:g}",
                     f"{alone['dbsm_vv'][i]:.3f}", f"{alone['dbsm_vh'][i]:.3f}"))
    diagnostic_name = "cavity_dbsm.csv" if legacy_single else f"{NAME}_dbsm.csv"
    with open(os.path.join(HERE, diagnostic_name), "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"         wrote {diagnostic_name}  (aggregate at {f0:g} GHz)")

    print(f"\n         roll   compact-feature peak VV   (dBsm, {f0:g} GHz)")
    dark = []
    for r in sorted(set(rol)):
        m = rol == r
        v = float(np.max(alone["dbsm_vv"][m]))
        hidden = v <= -199.0
        dark.append(hidden)
        print(f"         {r:4.0f}   {v:+13.1f}"
              + ("   <- all apertures face away" if hidden
                 else ""))
    if legacy_single and not any(dark):
        print("\n         WARNING  the cavity never goes dark over a full roll "
              "sweep.  A cavity in\n                  the skin can only be seen "
              "from its own side -- check "
              "APERTURE_NORMAL\n                  and the CAD frame before "
              "trusting this.")

    _refuse_unexpected_grims(output_dir, expected_names)
    actual_names = {
        os.path.basename(path)
        for path in glob.glob(os.path.join(output_dir, "*.grim"))
    }
    if actual_names != set(expected_names):
        raise SystemExit(
            "Output/ does not contain the exact expected cavity component set. "
            "Files were preserved but no complete provenance manifest was "
            "written."
        )
    try:
        _current_points, current_sources, _legacy = build_points(gen)
        current_payload = _compact_feature_provenance_payload(
            current_sources, expected_names
        )
    except Exception as exc:
        raise SystemExit(
            "compact-feature inputs became unreadable during "
            "step 3c. Outputs were preserved but no complete provenance "
            "manifest was written."
        ) from exc
    if stable_json_fingerprint(current_payload) != provenance_sha256:
        raise SystemExit(
            "pattern, coordinate table, body profile, grid, configuration, "
            "runtime, or "
            "workflow source changed during step 3c. Outputs were preserved "
            "but no complete provenance manifest was written."
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

    print("\nNEXT     4_combine")


if __name__ == "__main__":
    main()
