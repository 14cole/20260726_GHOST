#!/usr/bin/env python3
"""
Full-workflow integration gate — the whole chain the user actually runs:

  perimeter .txt (door coords)  +  2D clean/feature -> make_delta_grim
  +  wing (2D airfoil coefficient)  +  wing-body corner
  +  point scatterer (precomputed 3-D delta pattern, e.g. a cavity)
  ->  assembled on a MULTI-FREQUENCY BoR body
  ->  ONE radar .grim (azimuth x elevation x frequency x [VV,HH,VH]).

Exercises all FOUR component types at once.

This does NOT re-check absolute accuracy (that is pinned per-piece: feature vs
BoR truth, wing vs analytic plate, corner vs dihedral formula).  It checks the
assembled pipeline is INTERNALLY CONSISTENT:

  T0  perimeter .txt round-trips through read_perimeter_txt (the real input).
  T1  export_radar_grim writes ONE grim with the 4 axes and it reloads.
  T2  SUPERPOSITION: full(body+door+wing+corner) == body-only + components-only
      (the whole pipeline, incl. the radar-frame rotation, is linear).
  T3  LIMITING: a NULL feature (featured==clean) leaves the body untouched.
  T4  RECIPROCITY/ENERGY: polarimetric power ||S||_F is conserved from the
      vehicle meridian basis into the radar V/H basis for the full mix.
  T5  MULTI-FREQUENCY: the body is frequency-dependent and the two frequency
      slices genuinely differ.

Reuses the ring-gate cylinder + cached bodies; solves small 2D coupons/plate.

Run from tests/:  python3 validate_full_workflow.py
"""

import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "..")

import validate_line_expansion as RG                                    # noqa: E402
from bor_solver import solve_bor                                        # noqa: E402
from grim_io import export_result_to_grim                              # noqa: E402
from rcs_solver import solve_monostatic_rcs_2d                          # noqa: E402
from feature_sum import (_attitude, _direction, export_radar_grim,      # noqa: E402
                         make_delta_grim,
                         point_pattern_convention_metadata, sum_features)
from line_expand import read_perimeter_txt                             # noqa: E402
from validation_cache import bor_solver_sources, cache_path            # noqa: E402

TMP = "_full_wf_tmp"
os.makedirs(TMP, exist_ok=True)
FREQS = [3.0, 6.0]
LAM6 = RG.LAM
A = RG.A_BODY
_fails = []


def gate(label, ok, note=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {note}")
    if not ok:
        _fails.append(label)


# ── multi-frequency BoR body (same cylinder, solved at each frequency) ────────
def body_dict():
    backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    body_aspects = np.arange(0.0, 180.1, 10.0)
    key = cache_path(
        "full_wf_body",
        {
            "frequencies_ghz": FREQS,
            "aspects_deg": body_aspects,
            "n_modes": 22,
            "formulation": "cfie",
            "cfie_alpha": 0.5,
            "generatrix": RG.cylinder_gen(False),
        },
        bor_solver_sources(backend),
    )
    if os.path.exists(key):
        with open(key, "rb") as fh:
            return pickle.load(fh)
    gen = RG.cylinder_gen(False)
    asp = list(body_aspects)
    bodies = {}
    for f in FREQS:
        print(f"   solving body at {f} GHz ...", flush=True)
        bodies[f] = solve_bor(gen, f * 1e9, asp, formulation="cfie", cfie_alpha=0.5,
                              n_modes=22, workers=4)
    with open(key, "wb") as fh:
        pickle.dump(bodies, fh)
    return bodies


# ── small 2D pieces ──────────────────────────────────────────────────────────
def delta_grim(null=False):
    """A panel-gap delta .grim (multi-pol, multi-freq).  null -> featured==clean."""
    phi = list(np.arange(0.0, 180.1, 5.0))
    clean, feat = [], []
    for pol in ("TM", "TE"):
        rc = solve_monostatic_rcs_2d(RG.coupon(6.0, False), FREQS, phi, pol, geometry_units="meters")
        clean.append(export_result_to_grim(rc, os.path.join(TMP, f"c_{pol}"))[0])
        src = RG.coupon(6.0, False) if null else RG.coupon(6.0, True)
        rf = solve_monostatic_rcs_2d(src, FREQS, phi, pol, geometry_units="meters")
        feat.append(export_result_to_grim(rf, os.path.join(TMP, f"f_{pol}"))[0])
    return make_delta_grim(clean, feat, os.path.join(TMP, f"delta{'_null' if null else ''}.grim"))


def wing_grims():
    """Full-object airfoil (flat plate) coefficient as per-pol multi-freq grims."""
    c, t = 4 * LAM6, LAM6 / 50.0
    ds = LAM6 / 25.0
    nx = int(math.ceil(c / ds))
    xs = np.linspace(-c / 2, c / 2, nx + 1)
    ys = np.linspace(t / 2, -t / 2, 3)
    pts = [(x, t / 2) for x in xs] + [(c / 2, y) for y in ys[1:]]
    pts += [(x, -t / 2) for x in xs[::-1][1:]] + [(-c / 2, y) for y in ys[::-1][1:-1]]
    pp = [{"x1": pts[i][0], "y1": pts[i][1],
           "x2": pts[(i + 1) % len(pts)][0], "y2": pts[(i + 1) % len(pts)][1]}
          for i in range(len(pts))]
    snap = {"title": "wing", "segments": [{"name": "w", "seg_type": "2",
            "properties": ["2", "0", "0", "0", "0"], "point_pairs": pp}],
            "ibcs": [], "dielectrics": []}
    phi = list(np.arange(0.0, 180.1, 5.0))
    out = []
    for pol in ("TM", "TE"):
        r = solve_monostatic_rcs_2d(snap, FREQS, phi, pol, geometry_units="meters")
        out.append(export_result_to_grim(r, os.path.join(TMP, f"wing_{pol}"))[0])
    return out


def door_txt(path, z0, phi0_deg, dz=0.05, arc=0.05, n=6):
    dphi = arc / A
    p0, p1 = math.radians(phi0_deg) - dphi / 2, math.radians(phi0_deg) + dphi / 2
    zl, zh = z0 - dz / 2, z0 + dz / 2
    def P(ph, z): return (A * math.cos(ph), A * math.sin(ph), z)
    c = [P(p0, z) for z in np.linspace(zl, zh, n + 1)]
    c += [P(ph, zh) for ph in np.linspace(p0, p1, n + 1)[1:]]
    c += [P(p1, z) for z in np.linspace(zh, zl, n + 1)[1:]]
    c += [P(ph, zl) for ph in np.linspace(p1, p0, n + 1)[1:]]
    with open(path, "w") as fh:
        for i in range(len(c) - 1):
            fh.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (*c[i], *c[i + 1]))
    return path


# ── assemble ─────────────────────────────────────────────────────────────────
print("=" * 74)
print("Full-workflow gate — body + feature + wing + corner + point -> radar .grim")
print("=" * 74)

bodies = body_dict()
gen = RG.cylinder_gen(False)
print("\nBuilding 2D pieces (delta grim, wing grims) ...", flush=True)
delta = delta_grim()
wings = wing_grims()

door = door_txt(os.path.join(TMP, "door.txt"), z0=+0.06, phi0_deg=0.0)
segs = read_perimeter_txt(door)
gate("T0 perimeter .txt round-trips (head-to-tail, closed loop)",
     segs.shape[1:] == (2, 3) and len(segs) >= 4
     and np.linalg.norm(segs[-1, 1] - segs[0, 0]) < 1e-6 * A,
     f"({len(segs)} segs, closed)")

span = np.array([[[0.11, 0.0, -4 * LAM6], [0.11, 0.0, 4 * LAM6]]])
placements = [
    {"delta": delta, "perimeter": door},                       # feature via delta .grim path
    {"delta": wings, "perimeter": span, "normal": (0.0, 1.0, 0.0)},  # wing via per-pol grim list
]
corner = {"fold": np.array([[0.11, 0.0, -4 * LAM6], [0.11, 0.0, 4 * LAM6]]),
          "n_wing": (0.0, 1.0, 0.0), "n_body": (1.0, 0.0, 0.0), "face_width": 3 * LAM6}

# a POINT scatterer (e.g. a blind cavity from external 3-D MoM): synthetic
# multi-freq delta pattern, aperture facing +x on the body side, placed off-axis
az_p = np.arange(0.0, 360.1, 20.0)
el_p = np.arange(0.0, 90.1, 15.0)
amp_p = np.zeros((len(az_p), len(el_p), len(FREQS), 3), complex)
for fi, ff in enumerate(FREQS):
    amp_p[:, :, fi, 0] = 0.03 * ff / 6.0                        # VV, freq-dependent
    amp_p[:, :, fi, 1] = -0.02 * ff / 6.0                       # HH
    amp_p[:, :, fi, 2] = 0.005j                                 # a little cross-pol
cavity = {"pattern": {"azimuths": az_p, "elevations": el_p, "frequencies": FREQS,
                      "polarizations": ["VV", "HH", "VH"], "amp": amp_p,
                      **point_pattern_convention_metadata()},
          "location": (A, 0.0, -0.03), "aperture_normal": (1.0, 0.0, 0.0),
          "roll_ref": (0.0, 0.0, 1.0)}
points = [cavity]

AZ = np.arange(0.0, 360.1, 30.0)
EL = np.arange(-20.0, 20.1, 20.0)
kw = dict(generatrix=gen, frequencies_ghz=FREQS, azimuths_deg=AZ, elevations_deg=EL,
          axis_az_deg=0.0, axis_el_deg=0.0)

print("\nExporting assembled radar .grim (body + feature + wing + corner + point) ...", flush=True)
p_full = export_radar_grim(os.path.join(TMP, "full"), bor_result=bodies,
                           placements=placements, corners=[corner], points=points, **kw)
p_body = export_radar_grim(os.path.join(TMP, "body"), bor_result=bodies,
                           placements=[], corners=[], **kw)
p_comp = export_radar_grim(os.path.join(TMP, "comp"), bor_result=None,
                           placements=placements, corners=[corner], points=points, **kw)


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


gf, gb, gc = load(p_full), load(p_body), load(p_comp)

# T1 axes + shape
ok = (list(gf["azimuths"]) == list(AZ) and list(gf["elevations"]) == list(EL)
      and list(gf["frequencies"]) == FREQS and list(gf["polarizations"]) == ["VV", "HH", "VH"]
      and gf["rcs_power"].shape == (len(AZ), len(EL), len(FREQS), 3))
gate("T1 one .grim with axes az x el x freq x pol", ok,
     f"(shape {gf['rcs_power'].shape}, pols {list(gf['polarizations'])}, freqs {list(gf['frequencies'])})")

# T2 superposition: full == body + components (complex amplitude, all axes)
Af = gf["rcs_amp_real"] + 1j * gf["rcs_amp_imag"]
Ab = gb["rcs_amp_real"] + 1j * gb["rcs_amp_imag"]
Ac = gc["rcs_amp_real"] + 1j * gc["rcs_amp_imag"]
scale = np.abs(Af).max() + 1e-30
sup = float(np.max(np.abs(Af - (Ab + Ac)))) / scale
gate("T2 superposition: full == body-only + components-only", sup < 1e-6,
     f"(max rel {sup:.1e})")

# T3 null feature leaves the body untouched (wing + corner + point still present)
p_null = export_radar_grim(os.path.join(TMP, "null"), bor_result=bodies,
                           placements=[{"delta": delta_grim(null=True), "perimeter": door},
                                       {"delta": wings, "perimeter": span, "normal": (0, 1, 0)}],
                           corners=[corner], points=points, **kw)
An = load(p_null)
An = An["rcs_amp_real"] + 1j * An["rcs_amp_imag"]
# null feature + same wing + corner + point: == the same run with the door removed
p_wc = export_radar_grim(os.path.join(TMP, "wc"), bor_result=bodies,
                         placements=[{"delta": wings, "perimeter": span, "normal": (0, 1, 0)}],
                         corners=[corner], points=points, **kw)
Awc = load(p_wc); Awc = Awc["rcs_amp_real"] + 1j * Awc["rcs_amp_imag"]
gate("T3 null feature contributes nothing (null-door run == wing+corner+point run)",
     float(np.max(np.abs(An - Awc))) / scale < 1e-6,
     f"(max rel {float(np.max(np.abs(An - Awc)))/scale:.1e})")

# T4 Frobenius energy conserved vehicle -> radar for the full mix (sample looks, 6 GHz)
R, _ = _attitude(0.0, 0.0, 0.0)
d_e = np.array([_direction(a, e) for a in AZ for e in EL])
d_v = d_e @ R
veh = sum_features(bodies[6.0], placements, d_v, 6.0, generatrix=gen, mode="coherent",
                   corners=[corner], points=points)
fro_v = np.abs(veh["amp_vv"]) ** 2 + np.abs(veh["amp_hh"]) ** 2 + 2 * np.abs(veh["amp_vh"]) ** 2
S = Af[:, :, 1, :].reshape(len(AZ) * len(EL), 3)      # freq index 1 == 6 GHz
fro_r = np.abs(S[:, 0]) ** 2 + np.abs(S[:, 1]) ** 2 + 2 * np.abs(S[:, 2]) ** 2
gate("T4 polarimetric power ||S||_F conserved vehicle->radar (full mix)",
     float(np.max(np.abs(fro_r - fro_v))) / (fro_v.max() + 1e-30) < 1e-4,
     f"(max rel {float(np.max(np.abs(fro_r - fro_v)))/(fro_v.max()+1e-30):.1e})")

# T5 the two frequency slices genuinely differ (body + features are freq-dependent)
d01 = float(np.max(np.abs(Af[:, :, 0, :] - Af[:, :, 1, :]))) / scale
gate("T5 multi-frequency: 3 GHz and 6 GHz slices differ", d01 > 0.05,
     f"(max rel diff {d01:.2f})")

print(f"\n{'=' * 74}")
print("ALL GATES PASSED" if not _fails else f"{len(_fails)} FAILED")
for f in _fails:
    print(f"   FAILED: {f}")
print("=" * 74)
import shutil
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if _fails else 0)
