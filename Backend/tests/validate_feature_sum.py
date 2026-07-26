#!/usr/bin/env python3
"""
End-to-end gate for the delta-grim + multi-placement pipeline (feature_sum.py).

Exercises exactly the workflow the user drives:
  2D coupon solve (clean & featured) -> export .grim -> make_delta_grim ->
  load_seam_from_grim -> sum_features on a vehicle BoR body.

Gates
  G0  make_delta_grim(clean, featured) reproduces the in-memory coherent
      subtraction through the authoritative float64 raw-field path.
  G1  a delta grim expanded around the groove ring reproduces the BoR ground
      truth to the SAME accuracy the ring gate established (the calibrated
      psi_VV/psi_HH now baked in -> bounded residual phase, <3.5 dB magnitude,
      broadside +/- 20 deg).
  G2  NULL: featured == clean -> delta == 0 -> vehicle signature == bare body
      to machine precision.
  G3  placement / superposition: two copies of one feature summed in one call
      == the coherent sum of two separate single-feature calls; and translating
      a feature changes only its phase, not its standalone sigma.

Reuses the ring gate's cached BoR ground truth when present.

Run from tests/:  python3 validate_feature_sum.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, "..")

import validate_line_expansion as RG          # noqa: E402  (geometry + cached BoR)
from feature_sum import (_attitude, _direction, directions_from_aspect_roll,   # noqa: E402
                         export_radar_grim, export_signature_grim,
                         load_seam_from_grim, make_delta_grim, sum_features)
from line_expand import coefficients_from_2d                                   # noqa: E402
from line_expand import (PSI_HH_DEG, PSI_VV_DEG,                            # noqa: E402
                         VALIDITY_HALF_ANGLE_DEG, SeamCoefficients,
                         expand_perimeter, surface_of_revolution_normal)
from rcs_solver import solve_monostatic_rcs_2d                               # noqa: E402
from grim_io import export_result_to_grim                                    # noqa: E402

TMP = "_feature_sum_tmp"
os.makedirs(TMP, exist_ok=True)
_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


FREQ = RG.FREQ_GHZ
# Match the ring gate's coupon exactly (width + full angle range) so G1 tests
# ONLY the grim round-trip + calibration, not coupon-width/range convergence
# (which is the ring gate's job).
COUPON_LAM = 12.0
PHI = np.arange(0.0, 180.1, 2.5)
ASPECTS = RG.ASPECTS


def _solve_and_export(grooved, stem):
    """Solve the capsule coupon (both pols) and export one grim per pol."""
    snap = RG.coupon(COUPON_LAM, grooved)
    paths = []
    for pol in ("TM", "TE"):
        res = solve_monostatic_rcs_2d(snap, [FREQ], list(PHI), pol,
                                      geometry_units="meters")
        p = export_result_to_grim(res, os.path.join(TMP, f"{stem}_{pol}"),
                                  history=f"coupon {stem} {pol}")[0]
        paths.append(p)
    return paths


print("=" * 74)
print("Feature-sum pipeline gate")
print("=" * 74)

print("\nG0. make_delta_grim round-trips the coherent subtraction")
clean = _solve_and_export(False, "clean")
feat = _solve_and_export(True, "feat")
delta_path = make_delta_grim(clean, feat, os.path.join(TMP, "seam.grim"))
seam = load_seam_from_grim(delta_path, FREQ)

# in-memory reference subtraction
ref = {}
for pol, attr in (("TM", "dA_tm"), ("TE", "dA_te")):
    ca = solve_monostatic_rcs_2d(RG.coupon(COUPON_LAM, False), [FREQ], list(PHI), pol,
                                 geometry_units="meters")["samples"]
    fa = solve_monostatic_rcs_2d(RG.coupon(COUPON_LAM, True), [FREQ], list(PHI), pol,
                                 geometry_units="meters")["samples"]
    ca = {round(s["theta_inc_deg"], 4): complex(s["rcs_amp_real"], s["rcs_amp_imag"]) for s in ca}
    fa = {round(s["theta_inc_deg"], 4): complex(s["rcs_amp_real"], s["rcs_amp_imag"]) for s in fa}
    ref[attr] = np.array([fa[round(p, 4)] - ca[round(p, 4)] for p in PHI])
scale = max(np.max(np.abs(ref["dA_tm"])), np.max(np.abs(ref["dA_te"])), 1e-30)
e_tm = np.max(np.abs(seam.dA_tm - ref["dA_tm"])) / scale
e_te = np.max(np.abs(seam.dA_te - ref["dA_te"])) / scale
gate("delta grim == in-memory subtraction (TM & TE)", max(e_tm, e_te) < 1e-3,
     f"(TM {e_tm:.1e}, TE {e_te:.1e} of peak)")

print("\nG1. expanded delta grim reproduces the BoR groove truth (calibrated)")
# A standalone run may not have the ring gate's cache yet. Build it instead
# of continuing with ``None`` and failing for a test-harness reason.
bor = RG.bor if RG.bor is not None else RG.load_cached_bor(verbose=True)
dF = {"F_vv": np.asarray(bor["groove"]["amp_vv"]) - np.asarray(bor["smooth"]["amp_vv"]),
      "F_hh": np.asarray(bor["groove"]["amp_hh"]) - np.asarray(bor["smooth"]["amp_hh"])}
per = RG.ring_perimeter(RG.A_BODY, RG.Z0)
nfn = surface_of_revolution_normal(RG.cylinder_gen(False))
th = np.radians(ASPECTS)
dirs = np.column_stack([np.sin(th), np.zeros_like(th), np.cos(th)])
exp = expand_perimeter(per, seam, nfn, dirs,
                       psi_tm_deg=PSI_HH_DEG, psi_te_deg=PSI_VV_DEG)
# Both bounds are the measured envelope of the independent, converged ring
# benchmark.  They describe this reduced-order PEC-groove anchor, not a
# universal guarantee for every material/cross-section.
mag_core = np.abs(ASPECTS - 90.0) <= VALIDITY_HALF_ANGLE_DEG
ph_core = mag_core
for ch, name in (("F_vv", "VV"), ("F_hh", "HH")):
    rm = dF[ch][mag_core] / exp[ch][mag_core]
    rp = dF[ch][ph_core] / exp[ch][ph_core]
    mag = np.abs(20 * np.log10(np.abs(rm)))
    ph = np.abs(np.degrees(np.angle(rp)))           # calibrated -> should be ~0
    gate(f"{name} magnitude within 3.5 dB over broadside "
         f"+/-{VALIDITY_HALF_ANGLE_DEG:g}",
         float(mag.max()) < 3.5,
         f"(worst {mag.max():.2f} dB)")
    gate(f"{name} residual phase < 25 deg over broadside "
         f"+/-{VALIDITY_HALF_ANGLE_DEG:g}",
         float(ph.max()) < 25.0,
         f"(worst {ph.max():.1f} deg)")

print("\nG2. NULL: featured==clean -> zero delta -> signature == bare body")
null_delta = make_delta_grim(clean, clean, os.path.join(TMP, "null.grim"))
null_seam = load_seam_from_grim(null_delta, FREQ)
body_only = sum_features(bor["smooth"], [], dirs, FREQ,
                         generatrix=RG.cylinder_gen(False), mode="coherent")
with_null = sum_features(bor["smooth"],
                         [{"delta": null_seam, "perimeter": per}],
                         dirs, FREQ, generatrix=RG.cylinder_gen(False), mode="coherent")
dmax = float(np.max(np.abs(with_null["sigma_vv"] - body_only["sigma_vv"])
                    + np.abs(with_null["sigma_hh"] - body_only["sigma_hh"])))
gate("null feature leaves body signature unchanged", dmax < 1e-20,
     f"(max sigma drift {dmax:.1e} m^2)")

print("\nG3. placement / superposition")
one = sum_features(None, [{"delta": seam, "perimeter": per}], dirs, FREQ,
                   generatrix=RG.cylinder_gen(False), mode="coherent")
two_sep_vv = 2.0 * one["feature_amps"][0]["F_vv"]      # two identical copies
two_in_one = sum_features(None,
                          [{"delta": seam, "perimeter": per},
                           {"delta": seam, "perimeter": per}],
                          dirs, FREQ, generatrix=RG.cylinder_gen(False), mode="coherent")
err = float(np.max(np.abs(two_in_one["feature_amps"][0]["F_vv"]
                          + two_in_one["feature_amps"][1]["F_vv"] - two_sep_vv)))
gate("two placements in one call == sum of separate calls", err < 1e-18,
     f"(max amp diff {err:.1e})")

# translate one copy along +z by 40 mm: standalone sigma invariant, phase shifts
per_shift = per.copy()
per_shift[..., 2] += 0.040
shifted = sum_features(None, [{"delta": seam, "perimeter": per_shift}], dirs, FREQ,
                       generatrix=RG.cylinder_gen(False), mode="coherent")
d_sig = float(np.max(np.abs(shifted["sigma_vv"] - one["sigma_vv"])))
k = 2.0 * math.pi * FREQ * 1e9 / 299792458.0
pred = 2.0 * k * 0.040 * np.cos(th)          # predicted two-way phase shift vs axis
got = np.angle(shifted["feature_amps"][0]["F_vv"] / one["feature_amps"][0]["F_vv"])
dphi = np.abs(np.angle(np.exp(1j * (got - pred))))
gate("translation leaves standalone sigma invariant", d_sig < 1e-6 * (one["sigma_vv"].max() + 1e-30),
     f"(max sigma drift {d_sig:.1e} m^2)")
_c = np.abs(ASPECTS - 90.0) <= VALIDITY_HALF_ANGLE_DEG
gate("translation phase shift matches 2k*dz*cos(theta)", float(np.nanmax(dphi[_c])) < 0.05,
     f"(worst {np.nanmax(dphi[_c]):.3f} rad over core)")

print("\nG4. export_signature_grim writes per-channel .grim and round-trips")
asp_grid = np.arange(70.0, 110.1, 10.0)
roll_grid = np.array([0.0, 90.0])
paths = export_signature_grim(
    os.path.join(TMP, "vehicle"), bor_result=bor["smooth"],
    placements=[{"delta": delta_path, "perimeter": per}],
    generatrix=RG.cylinder_gen(False), frequencies_ghz=[FREQ],
    aspects_deg=asp_grid, rolls_deg=roll_grid, mode="hybrid")
gate("writes one .grim per channel (VV, HH, VH)", len(paths) == 3,
     f"({[os.path.basename(p) for p in paths]})")

# reload the VV file: the primary power must match its stored coherent field,
# while the explicitly requested hybrid estimate lives under a separate key
g = np.load(paths[0], allow_pickle=False)
az, el = g["azimuths"], g["elevations"]
grim_ok = (list(az) == list(roll_grid)) and (list(el) == list(asp_grid))
gate("axes are body-frame (azimuth=roll, elevation=aspect)", grim_ok,
     f"(az={list(az)}, el={list(el)})")
# recompute VV sigma on the same grid and compare (float32 tolerance)
dirs4, _, _ = directions_from_aspect_roll(asp_grid, roll_grid)
recompute = sum_features(bor["smooth"], [{"delta": delta_path, "perimeter": per}],
                         dirs4, FREQ, generatrix=RG.cylinder_gen(False), mode="hybrid")
sig_grid = recompute["sigma_vv"].reshape(len(asp_grid), len(roll_grid)).T[..., None, None]
coh_grid = recompute["coherent_sigma_vv"].reshape(
    len(asp_grid), len(roll_grid)).T[..., None, None]
rel = np.max(np.abs(g["rcs_power"] - coh_grid.astype(np.float32))) / (
    coh_grid.max() + 1e-30)
rel_est = np.max(np.abs(g["combination_estimate_power"]
                        - sig_grid.astype(np.float32))) / (
                            sig_grid.max() + 1e-30)
gate("reloaded rcs_power matches coherent complex field", float(rel) < 1e-4,
     f"(max rel {rel:.1e})")
gate("hybrid estimate persists under its explicit estimate key",
     float(rel_est) < 1e-4, f"(max rel {rel_est:.1e})")
g.close()

print("\nG5. export_radar_grim: (az, el, freq, pol) radar-frame .grim")
az_g = np.array([0.0, 30.0, 60.0, 90.0])
el_g = np.array([-20.0, 0.0, 20.0])
radar_placements = [{"delta": delta_path, "perimeter": per}]
# (a) horizontal axis: rotation is nontrivial -> check Frobenius invariance
rpath = export_radar_grim(os.path.join(TMP, "radar"),
                          bor_result=None, placements=radar_placements,
                          generatrix=RG.cylinder_gen(False), frequencies_ghz=[FREQ],
                          azimuths_deg=az_g, elevations_deg=el_g,
                          axis_az_deg=0.0, axis_el_deg=0.0, roll_deg=0.0)
gr = np.load(rpath, allow_pickle=False)
gate("single .grim with az/el/freq/pol axes", str(gr["azimuths"].shape[0]) == "4"
     and list(gr["polarizations"]) == ["VV", "HH", "VH"]
     and gr["rcs_power"].shape == (4, 3, 1, 3),
     f"(shape {gr['rcs_power'].shape}, pols {list(gr['polarizations'])})")

# recompute vehicle-frame S at the same looks and compare Frobenius norms
R, ax = _attitude(0.0, 0.0, 0.0)
d_e = np.array([_direction(a, e) for a in az_g for e in el_g])
d_v = d_e @ R
veh = sum_features(None, radar_placements, d_v, FREQ,
                   generatrix=RG.cylinder_gen(False), mode="coherent")
fro_veh = (np.abs(veh["amp_vv"]) ** 2 + np.abs(veh["amp_hh"]) ** 2
           + 2 * np.abs(veh["amp_vh"]) ** 2)
ar = gr["rcs_amp_real"] + 1j * gr["rcs_amp_imag"]      # [az, el, f, pol]
ar = ar.reshape(len(az_g) * len(el_g), 3)              # rows match d_e order
fro_rad = (np.abs(ar[:, 0]) ** 2 + np.abs(ar[:, 1]) ** 2 + 2 * np.abs(ar[:, 2]) ** 2)
rel = np.max(np.abs(fro_rad - fro_veh)) / (fro_veh.max() + 1e-30)
gate("polarimetric power invariant under basis rotation (||S||_F)", float(rel) < 1e-4,
     f"(max rel {rel:.1e})")
gr.close()

# (b) vertical axis -> rotation is identity -> radar channels == vehicle channels
rpath2 = export_radar_grim(os.path.join(TMP, "radar_vert"),
                           bor_result=None, placements=radar_placements,
                           generatrix=RG.cylinder_gen(False), frequencies_ghz=[FREQ],
                           azimuths_deg=az_g, elevations_deg=el_g,
                           axis_az_deg=0.0, axis_el_deg=90.0, roll_deg=0.0)
gv = np.load(rpath2, allow_pickle=False)
Rv, _ = _attitude(0.0, 90.0, 0.0)
d_v2 = np.array([_direction(a, e) for a in az_g for e in el_g]) @ Rv
veh2 = sum_features(None, radar_placements, d_v2, FREQ,
                    generatrix=RG.cylinder_gen(False), mode="coherent")
av = (gv["rcs_amp_real"] + 1j * gv["rcs_amp_imag"]).reshape(len(az_g) * len(el_g), 3)
d_vv = np.max(np.abs(av[:, 0] - veh2["amp_vv"]))
d_vh = np.max(np.abs(av[:, 2] - veh2["amp_vh"]))
scale = np.abs(veh2["amp_vv"]).max() + 1e-30
gate("vertical axis: radar channels == vehicle channels (M=I)",
     float(max(d_vv, d_vh)) / scale < 1e-5,
     f"(VV {d_vv/scale:.1e}, VH {d_vh/scale:.1e})")
gv.close()

print("\nG6. wing as a placement (own normal) added to the body")
# a flat-plate 'wing': full-object 2D strip coefficient, expanded along a span
# line, carrying its OWN face normal (not the body surface normal)
LAMW = 299792458.0 / (FREQ * 1e9)
c_w, L_w, t_w = 4 * LAMW, 8 * LAMW, LAMW / 50.0
dsw = LAMW / 25.0


def _plate_snapshot():
    nx = int(math.ceil(c_w / dsw))
    xs = np.linspace(-c_w / 2, c_w / 2, nx + 1)
    ys = np.linspace(t_w / 2, -t_w / 2, 3)
    pts = [(x, t_w / 2) for x in xs] + [(c_w / 2, y) for y in ys[1:]]
    pts += [(x, -t_w / 2) for x in xs[::-1][1:]] + [(-c_w / 2, y) for y in ys[::-1][1:-1]]
    pairs = [{"x1": pts[i][0], "y1": pts[i][1],
              "x2": pts[(i + 1) % len(pts)][0], "y2": pts[(i + 1) % len(pts)][1]}
             for i in range(len(pts))]
    return {"title": "wing", "segments": [{"name": "w", "seg_type": "2",
            "properties": ["2", "0", "0", "0", "0"], "point_pairs": pairs}],
            "ibcs": [], "dielectrics": []}


wing_coef = coefficients_from_2d(_plate_snapshot(), FREQ,
                                 np.arange(40.0, 140.1, 2.0), geometry_units="meters")
# span line along +z, offset off-axis so the wing sticks out of the body
span_line = np.array([[[0.10, 0.0, -L_w / 2], [0.10, 0.0, L_w / 2]]])
wing = {"delta": wing_coef, "perimeter": span_line, "normal": (0.0, 1.0, 0.0)}
# broadside look on the wing face (+y); at this look the thin cylinder body is
# side-on but the wing presents its full area -> wing dominates
d_wing = np.array([[0.0, 1.0, 0.0]])
res_w = sum_features(bor["smooth"], [wing], d_wing, FREQ,
                     generatrix=RG.cylinder_gen(False), mode="coherent")
sig_po = 4 * math.pi * (c_w * L_w) ** 2 / LAMW ** 2
# the wing's standalone contribution must equal the analytic plate (psi-free mag)
wing_only = 4 * math.pi * np.abs(res_w["feature_amps"][0]["F_vv"][0]) ** 2
gate("wing placement expands to the analytic plate 4piA^2/lam^2",
     abs(10 * math.log10(wing_only / sig_po)) < 1.0,
     f"({10 * math.log10(wing_only / sig_po):+.2f} dB vs PO, "
     f"wing {10 * math.log10(wing_only):+.1f} dBsm)")
# a wing REQUIRES its own normal: the body surface normal is the wrong frame
# for a span line (here the body normal is degenerate w.r.t. the span) -> raises
raised = False
try:
    sum_features(bor["smooth"], [{"delta": wing_coef, "perimeter": span_line}],
                 d_wing, FREQ, generatrix=RG.cylinder_gen(False), mode="coherent")
except ValueError:
    raised = True
gate("wing needs its own normal (body normal is the wrong frame)", raised,
     "(body-normal path raises; per-placement 'normal' is required)")

print(f"\n{'=' * 74}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED")
for f in _fails:
    print(f"   FAILED: {f}")
print("=" * 74)
sys.exit(1 if _fails else 0)
