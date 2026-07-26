"""Second-round validation: mixed materials, TYPE 5, degenerate limits."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rcs_solver import solve_monostatic_rcs_2d
import mie_reference as mie
from validation_2d_helpers import make_circle_segment, check_mie_case

FREQ = 3.0
FREQ_HZ = FREQ * 1e9
NPRIM = 128
ok = True

# --- A. TYPE 5 sanity: layered cylinder with SAME eps in both layers == homogeneous ---
# outer r=0.05 TYPE 3 (air/diel, pos_mat=1), inner r=0.03 TYPE 5 (diel1/diel2, pos_mat=2 neg_mat=1)
# with eps1 = eps2 = 4 the answer must equal the homogeneous eps=4 Mie result.
ref = mie.sigma_dielectric_cylinder(0.05, 4.0, 1.0, FREQ_HZ, "TM")
ref_te = mie.sigma_dielectric_cylinder(0.05, 4.0, 1.0, FREQ_HZ, "TE")
# TYPE 5: normal from neg_mat into pos_mat. Inner circle: pos_mat=2 is the core
# (inside), neg_mat=1 the shell (outside) -> normal points inward -> CCW winding.
snap_t5 = {"title": "t5",
           "segments": [
               make_circle_segment("outer", 3, 0.05, NPRIM, pos_mat="1"),
               make_circle_segment("inner", 5, 0.03, NPRIM, pos_mat="2", neg_mat="1", cw=False),
           ],
           "ibcs": [],
           "dielectrics": [["1", "4", "0", "1", "0"], ["2", "4", "0", "1", "0"]]}
ok &= check_mie_case(
    "TYPE5 layered eps 4|4 == homog eps4", snap_t5, "TM", FREQ, ref)
ok &= check_mie_case(
    "TYPE5 layered eps 4|4 == homog eps4", snap_t5, "TE", FREQ, ref_te)

# --- B. Coated PEC with eps=1 coat == bare PEC of inner radius ---
ref_p = mie.sigma_pec_cylinder(0.04, FREQ_HZ, "TM")
ref_p_te = mie.sigma_pec_cylinder(0.04, FREQ_HZ, "TE")
snap_air = {"title": "aircoat",
            "segments": [
                make_circle_segment("outer", 3, 0.06, NPRIM, pos_mat="1"),
                make_circle_segment("inner", 4, 0.04, NPRIM, pos_mat="1"),
            ],
            "ibcs": [], "dielectrics": [["1", "1", "0", "1", "0"]]}
ok &= check_mie_case(
    "Coated PEC eps=1 coat == bare PEC", snap_air, "TM", FREQ, ref_p)
ok &= check_mie_case(
    "Coated PEC eps=1 coat == bare PEC", snap_air, "TE", FREQ, ref_p_te)

# --- C. Mixed scene: PEC body + separate dielectric body (dispatch check) ---
# No analytic reference; just verify it runs, report formulation label + warnings.
snap_mix = {"title": "mix",
            "segments": [
                make_circle_segment("pec", 2, 0.04, 96, cx=-0.12),
                make_circle_segment("diel", 3, 0.04, 96, pos_mat="1", cx=0.12),
            ],
            "ibcs": [], "dielectrics": [["1", "4", "0", "1", "0"]]}
for pol in ("TM", "TE"):
    try:
        out = solve_monostatic_rcs_2d(snap_mix, [FREQ], [0.0, 90.0], pol,
                                      geometry_units="meters")
        s = out["samples"]
        case_ok = (all(np.isfinite(float(row["rcs_linear"]))
                       and float(row["rcs_linear"]) >= 0.0 for row in s)
                   and np.isfinite(float(
                       out["metadata"]["residual_norm_max"])))
        ok &= case_ok
        print(f"{'PASS' if case_ok else 'FAIL'}  Mixed PEC+diel bodies {pol}: "
              f"rcs@0={s[0]['rcs_db']:.3f} dB "
              f"rcs@90={s[1]['rcs_db']:.3f} dB")
        print(f"    [{out['metadata']['formulation']}] residual_max={out['metadata']['residual_norm_max']:.2e}")
        for w in (out["metadata"].get("warnings") or [])[:4]:
            print(f"    warn: {w}")
    except Exception as exc:
        ok = False
        print(f"FAIL  Mixed PEC+diel bodies {pol}: "
              f"{type(exc).__name__}: {exc}")

# --- D. Mixed scene: PEC body + IBC body ---
snap_mix2 = {"title": "mix2",
             "segments": [
                 make_circle_segment("pec", 2, 0.04, 96, cx=-0.12),
                 make_circle_segment("ibc", 2, 0.04, 96, ibc="1", cx=0.12),
             ],
             "ibcs": [["1", "constant", "200", "100", "0", "0"]], "dielectrics": []}
for pol in ("TM", "TE"):
    try:
        out = solve_monostatic_rcs_2d(snap_mix2, [FREQ], [0.0], pol,
                                      geometry_units="meters")
        s = out["samples"]
        case_ok = (np.isfinite(float(s[0]["rcs_linear"]))
                   and float(s[0]["rcs_linear"]) >= 0.0
                   and np.isfinite(float(
                       out["metadata"]["residual_norm_max"])))
        ok &= case_ok
        print(f"{'PASS' if case_ok else 'FAIL'}  Mixed PEC+IBC bodies {pol}: "
              f"rcs@0={s[0]['rcs_db']:.3f} dB "
              f"[{out['metadata']['formulation']}] residual={out['metadata']['residual_norm_max']:.2e}")
    except Exception as exc:
        ok = False
        print(f"FAIL  Mixed PEC+IBC bodies {pol}: "
              f"{type(exc).__name__}: {exc}")

# --- E. Open PEC strip (sheet-free open arc, TYPE 2) — plate of width 0.1 m ---
# Broadside backscatter of a strip: sigma_2d ~ k*w^2 (physical optics) for TM at normal incidence.
w = 0.1
k0 = 2 * math.pi * FREQ_HZ / 299792458.0
po = k0 * w * w  # PO broadside estimate sigma = k w^2
snap_strip = {"title": "strip",
              "segments": [{
                  "name": "strip", "seg_type": "2",
                  "properties": ["2", "0", "0", "0", "0"],
                  "point_pairs": [{"x1": -w / 2, "y1": 0.0, "x2": w / 2, "y2": 0.0}],
              }],
              "ibcs": [], "dielectrics": []}
for pol in ("TM",):
    try:
        out = solve_monostatic_rcs_2d(snap_strip, [FREQ], [90.0], pol,
                                      geometry_units="meters")
        s = out["samples"][0]
        case_ok = (np.isfinite(float(s["rcs_linear"]))
                   and float(s["rcs_linear"]) >= 0.0)
        ok &= case_ok
        print(f"{'PASS' if case_ok else 'FAIL'}  "
              f"PEC strip w=0.1m broadside {pol}: {s['rcs_db']:.3f} dB "
              f"(PO est {10*math.log10(po):.3f} dB) [{out['metadata']['formulation']}]")
    except Exception as exc:
        ok = False
        print(f"FAIL  PEC strip broadside {pol}: "
              f"{type(exc).__name__}: {exc}")

try:
    solve_monostatic_rcs_2d(
        snap_strip, [FREQ], [90.0], "TE", geometry_units="meters")
except ValueError as exc:
    te_rejected = "open TYPE 2" in str(exc)
    ok &= te_rejected
    print(f"{'PASS' if te_rejected else 'FAIL'}  "
          "open TYPE 2 TE follows the documented rejection boundary")
except Exception as exc:  # noqa: BLE001
    ok = False
    print(f"FAIL  open TYPE 2 TE raised {type(exc).__name__}: {exc}")
else:
    ok = False
    print("FAIL  open TYPE 2 TE was accepted by a closed-obstacle formulation")

print("ALL GATES PASSED" if ok else "GATES FAILED")
sys.exit(0 if ok else 1)
