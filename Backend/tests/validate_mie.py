"""End-to-end validation of the 2D RCS solver against analytic Mie series."""
import math
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rcs_solver import solve_monostatic_rcs_2d
import mie_reference as mie

MAX_ERROR_DB = 0.20
MAX_ISOTROPY_SPREAD_DB = 0.05
_fails = []


def make_circle_segment(name, seg_type, r, n_prim, ibc="0", pos_mat="0", neg_mat="0",
                        cw=True, cx=0.0, cy=0.0):
    """Circle as n_prim straight primitives. cw=True -> outward normal under
    the project's left-right/up-normal convention."""
    pts = []
    for i in range(n_prim + 1):
        th = 2.0 * math.pi * i / n_prim
        if cw:
            th = -th
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    pairs = [{"x1": pts[i][0], "y1": pts[i][1], "x2": pts[i + 1][0], "y2": pts[i + 1][1]}
             for i in range(n_prim)]
    return {
        "name": name,
        "seg_type": str(seg_type),
        "properties": [str(seg_type), "0", ibc, pos_mat, neg_mat],
        "point_pairs": pairs,
    }


def run_case(label, snapshot, pol, freq_ghz, sigma_ref_m,
             expect_rejection=False):
    try:
        out = solve_monostatic_rcs_2d(
            geometry_snapshot=snapshot,
            frequencies_ghz=[freq_ghz],
            elevations_deg=[0.0, 37.0],
            polarization=pol,
            geometry_units="meters",
        )
    except Exception as exc:
        if expect_rejection:
            print(f"{label:42s} {pol}  PASS: rejected invalid winding "
                  f"({type(exc).__name__})")
            return
        print(f"{label:42s} {pol}  FAILED: {type(exc).__name__}: {exc}")
        _fails.append(f"{label} {pol}: solver exception")
        return
    if expect_rejection:
        print(f"{label:42s} {pol}  FAILED: invalid winding was accepted")
        _fails.append(f"{label} {pol}: invalid winding was accepted")
        return
    s0 = out["samples"][0]["rcs_linear"]
    s1 = out["samples"][1]["rcs_linear"]
    db0 = 10 * math.log10(s0)
    db1 = 10 * math.log10(s1)
    ref_db = 10 * math.log10(sigma_ref_m)
    form = out["metadata"]["formulation"]
    angle_spread = abs(db0 - db1)
    print(f"{label:42s} {pol}  solver={db0:8.3f} dB  ref={ref_db:8.3f} dB  "
          f"err={db0 - ref_db:+7.3f} dB  angle-spread={angle_spread:.4f} dB")
    print(f"    [{form}]")
    if abs(db0 - ref_db) > MAX_ERROR_DB:
        _fails.append(
            f"{label} {pol}: {db0-ref_db:+.3f} dB error exceeds "
            f"{MAX_ERROR_DB:.2f} dB")
    if angle_spread > MAX_ISOTROPY_SPREAD_DB:
        _fails.append(
            f"{label} {pol}: {angle_spread:.3f} dB angular spread exceeds "
            f"{MAX_ISOTROPY_SPREAD_DB:.2f} dB")
    warn = out["metadata"].get("warnings") or []
    for w in warn[:3]:
        print(f"    warn: {w}")


FREQ = 3.0  # GHz
FREQ_HZ = FREQ * 1e9
NPRIM = 128

# --- 1. PEC cylinder, r = 0.05 m (ka ~ pi) ---
r_pec = 0.05
ref_tm = mie.sigma_pec_cylinder(r_pec, FREQ_HZ, "TM")
ref_te = mie.sigma_pec_cylinder(r_pec, FREQ_HZ, "TE")
snap_pec = {"title": "pec", "segments": [make_circle_segment("c", 2, r_pec, NPRIM)],
            "ibcs": [], "dielectrics": []}
run_case("PEC cyl r=0.05m cw", snap_pec, "TM", FREQ, ref_tm)
run_case("PEC cyl r=0.05m cw", snap_pec, "TE", FREQ, ref_te)

# same but drawn CCW (wrong winding) — does the solver handle/normalize it?
snap_pec_ccw = {"title": "pec", "segments": [make_circle_segment("c", 2, r_pec, NPRIM, cw=False)],
                "ibcs": [], "dielectrics": []}
run_case("PEC cyl r=0.05m CCW-drawn", snap_pec_ccw, "TM", FREQ, ref_tm,
         expect_rejection=True)
run_case("PEC cyl r=0.05m CCW-drawn", snap_pec_ccw, "TE", FREQ, ref_te,
         expect_rejection=True)

# --- 2. Dielectric cylinder eps=4 ---
r_d = 0.05
ref_dtm = mie.sigma_dielectric_cylinder(r_d, 4.0, 1.0, FREQ_HZ, "TM")
ref_dte = mie.sigma_dielectric_cylinder(r_d, 4.0, 1.0, FREQ_HZ, "TE")
snap_d = {"title": "diel", "segments": [make_circle_segment("c", 3, r_d, NPRIM, pos_mat="1")],
          "ibcs": [], "dielectrics": [["1", "4", "0", "1", "0"]]}
run_case("Diel cyl eps=4 r=0.05m", snap_d, "TM", FREQ, ref_dtm)
run_case("Diel cyl eps=4 r=0.05m", snap_d, "TE", FREQ, ref_dte)

# --- 3. Lossy dielectric eps = 4 - 1j ---
ref_ltm = mie.sigma_dielectric_cylinder(r_d, 4.0 - 1.0j, 1.0, FREQ_HZ, "TM")
ref_lte = mie.sigma_dielectric_cylinder(r_d, 4.0 - 1.0j, 1.0, FREQ_HZ, "TE")
snap_l = {"title": "lossy", "segments": [make_circle_segment("c", 3, r_d, NPRIM, pos_mat="1")],
          "ibcs": [], "dielectrics": [["1", "4", "-1", "1", "0"]]}
run_case("Lossy diel eps=4-1j r=0.05m", snap_l, "TM", FREQ, ref_ltm)
run_case("Lossy diel eps=4-1j r=0.05m", snap_l, "TE", FREQ, ref_lte)

# --- 4. Coated PEC: PEC r=0.04 with eps=3 coat to r=0.06 ---
a_in, a_out = 0.04, 0.06
ref_ctm = mie.sigma_coated_pec_cylinder(a_in, a_out, 3.0, 1.0, FREQ_HZ, "TM")
ref_cte = mie.sigma_coated_pec_cylinder(a_in, a_out, 3.0, 1.0, FREQ_HZ, "TE")
snap_c = {"title": "coated",
          "segments": [
              make_circle_segment("outer", 3, a_out, NPRIM, pos_mat="1"),
              make_circle_segment("inner", 4, a_in, NPRIM, pos_mat="1"),
          ],
          "ibcs": [], "dielectrics": [["1", "3", "0", "1", "0"]]}
run_case("Coated PEC a=0.04/0.06 eps=3", snap_c, "TM", FREQ, ref_ctm)
run_case("Coated PEC a=0.04/0.06 eps=3", snap_c, "TE", FREQ, ref_cte)

# --- 5. IBC cylinder with small Zs ~ PEC limit (TM) ---
snap_ibc = {"title": "ibc",
            "segments": [make_circle_segment("c", 2, r_pec, NPRIM, ibc="1")],
            "ibcs": [["1", "constant", "1", "0", "0", "0"]],
            "dielectrics": []}
run_case("IBC cyl Zs=1ohm (~PEC) r=0.05m", snap_ibc, "TM", FREQ, ref_tm)
run_case("IBC cyl Zs=1ohm (~PEC) r=0.05m", snap_ibc, "TE", FREQ, ref_te)

print("\n" + "=" * 74)
if _fails:
    print(f"{len(_fails)} FAILED")
    for failure in _fails:
        print("  FAILED:", failure)
else:
    print("ALL GATES PASSED")
print("=" * 74)
sys.exit(1 if _fails else 0)
