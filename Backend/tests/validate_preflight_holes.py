"""Validate the four geometry-preflight fixes."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

from validation_2d_helpers import make_circle_segment
from rcs_solver import (
    validate_geometry_snapshot_for_solver, _build_panels, _build_linear_mesh,
    solve_monostatic_rcs_2d,
)
import mie_reference as mie

ok = True

def expect(label, snap, should_raise, needle="", scale=1.0):
    global ok
    try:
        validate_geometry_snapshot_for_solver(snap, base_dir=".", meters_scale=scale)
        raised, msg = False, ""
    except ValueError as e:
        raised, msg = True, str(e)
    good = raised == should_raise and (needle in msg if should_raise and needle else True)
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {label}: raised={raised}"
          + (f" [{msg[:90]}...]" if raised else ""))


def box_snap(gap):
    # Final edge descends the left side but stops `gap` short of the start
    # corner, leaving a slit along the boundary path (not touching any edge).
    pairs = [
        {"x1": 0, "y1": 0, "x2": 1, "y2": 0},
        {"x1": 1, "y1": 0, "x2": 1, "y2": 1},
        {"x1": 1, "y1": 1, "x2": 0, "y2": 1},
        {"x1": 0, "y1": 1, "x2": 0, "y2": gap},
    ]
    return {"title": "box", "segments": [{
        "name": "box", "seg_type": "2",
        "properties": ["2", "0", "0", "0", "0"],
        "point_pairs": list(reversed([{"x1": p["x2"], "y1": p["y2"], "x2": p["x1"], "y2": p["y1"]} for p in pairs])),
    }], "ibcs": [], "dielectrics": []}

# CW box (correct winding: reversed CCW chain). gap=0 exact -> pass.
expect("closed box, exact endpoints", box_snap(0.0), False)
# 1e-7 m gap: previously silent, now a crack error.
expect("box with 1e-7 m crack", box_snap(1e-7), True, "crack")
# 5e-10 m gap: below snap tolerance, merges fine -> pass.
expect("box with 5e-10 m gap (snaps)", box_snap(5e-10), False)
# Large intentional gap (aperture): not a crack (bigger than drawing tol).
expect("box with 0.01 m aperture", box_snap(0.01), False)

# Duplicate primitive.
snap = {"title": "d", "segments": [{
    "name": "s", "seg_type": "2", "properties": ["2", "0", "0", "0", "0"],
    "point_pairs": [
        {"x1": 0, "y1": 0, "x2": 1, "y2": 0},
        {"x1": 0, "y1": 0, "x2": 1, "y2": 0},
    ]}], "ibcs": [], "dielectrics": []}
expect("exact duplicate primitive", snap, True, "Duplicate")

# Reversed duplicate across segments.
snap = {"title": "d", "segments": [
    {"name": "a", "seg_type": "2", "properties": ["2", "0", "0", "0", "0"],
     "point_pairs": [{"x1": 0, "y1": 0, "x2": 1, "y2": 0}]},
    {"name": "b", "seg_type": "2", "properties": ["2", "0", "0", "0", "0"],
     "point_pairs": [{"x1": 1, "y1": 0, "x2": 0, "y2": 0}]},
], "ibcs": [], "dielectrics": []}
expect("reversed duplicate across segments", snap, True, "Duplicate")

# Collinear overlap from a shared endpoint.
snap = {"title": "c", "segments": [
    {"name": "a", "seg_type": "2", "properties": ["2", "0", "0", "0", "0"],
     "point_pairs": [{"x1": 0, "y1": 0, "x2": 1, "y2": 0}]},
    {"name": "b", "seg_type": "2", "properties": ["2", "0", "0", "0", "0"],
     "point_pairs": [{"x1": 0, "y1": 0, "x2": 2, "y2": 0}]},
], "ibcs": [], "dielectrics": []}
expect("collinear overlap via shared endpoint", snap, True, "Collinear")

# Backtracking chain (B->C doubles back over A->B).
snap = {"title": "c", "segments": [{
    "name": "s", "seg_type": "2", "properties": ["2", "0", "0", "0", "0"],
    "point_pairs": [
        {"x1": 0, "y1": 0, "x2": 1, "y2": 0},
        {"x1": 1, "y1": 0, "x2": 0.5, "y2": 0},
    ]}], "ibcs": [], "dielectrics": []}
expect("backtracking collinear chain", snap, True, "Collinear")

# Straight multi-primitive strip (collinear but non-overlapping): fine.
snap = {"title": "s", "segments": [{
    "name": "strip", "seg_type": "2", "properties": ["2", "0", "0", "0", "0"],
    "point_pairs": [
        {"x1": 0.0, "y1": 0, "x2": 0.25, "y2": 0},
        {"x1": 0.25, "y1": 0, "x2": 0.5, "y2": 0},
        {"x1": 0.5, "y1": 0, "x2": 0.75, "y2": 0},
        {"x1": 0.75, "y1": 0, "x2": 1.0, "y2": 0},
    ]}], "ibcs": [], "dielectrics": []}
expect("straight collinear strip (legal)", snap, False)

# TYPE fallback agreement: empty properties + seg_type '1' -> validator and
# builder must agree (both TYPE 1 now).
snap = {"title": "t", "segments": [{
    "name": "s", "seg_type": "1", "properties": [],
    "point_pairs": [{"x1": 0, "y1": 0, "x2": 1, "y2": 0}],
}], "ibcs": [["1", "constant", "100", "0", "0", "0"]], "dielectrics": []}
panels = _build_panels(snap, 1.0, 0.1)
good = panels[0].seg_type == 1
ok &= good
print(f"{'PASS' if good else 'FAIL'}  empty-props TYPE fallback: built seg_type={panels[0].seg_type} (validator sees 1)")

# Missing N -> auto density (a 10-wavelength line must not be 1 panel).
snap = {"title": "t", "segments": [{
    "name": "s", "seg_type": "2", "properties": ["2"],
    "point_pairs": [{"x1": 0, "y1": 0, "x2": 1.0, "y2": 0}],
}], "ibcs": [], "dielectrics": []}
panels = _build_panels(snap, 1.0, 0.1)   # lambda = 0.1 m -> 10 wavelengths
good = len(panels) >= 100
ok &= good
print(f"{'PASS' if good else 'FAIL'}  missing-N 10-lambda line: panels={len(panels)} (auto density)")

# Explicit N=1 still honored.
snap["segments"][0]["properties"] = ["2", "1"]
panels = _build_panels(snap, 1.0, 0.1)
good = len(panels) == 1
ok &= good
print(f"{'PASS' if good else 'FAIL'}  explicit N=1 honored: panels={len(panels)}")

# End-to-end regression: PEC cylinder still solves and matches Mie.
snap = {"title": "p", "segments": [make_circle_segment("c", 2, 0.05, 128, cw=True)],
        "ibcs": [], "dielectrics": []}
out = solve_monostatic_rcs_2d(snap, [3.0], [0.0], "TE", geometry_units="meters")
got = out["samples"][0]["rcs_db"]
ref = 10 * math.log10(mie.sigma_pec_cylinder(0.05, 3e9, "TE"))
good = abs(got - ref) < 0.05
ok &= good
print(f"{'PASS' if good else 'FAIL'}  end-to-end TE PEC: {got:.3f} vs {ref:.3f} dB")

print("ALL GATES PASSED" if ok else "GATES FAILED")
sys.exit(0 if ok else 1)
