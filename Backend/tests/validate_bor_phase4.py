"""Phase-4 gate battery: end-to-end BoR integration.

The headline gate from BOR_SOLVER_PLAN.md: an ogive-nose + cylinder + base
rocket profile, dispatched from .geo text through bor_dispatch, swept over
the FULL 0-180 deg aspect range in both polarizations, with mode- and
mesh-convergence demonstrated.  Plus: every material path of the dispatch
layer against the Mie references, taper mirror symmetry through the .geo
IBCS machinery, .grim export round-trip with dBsm units, and the adaptive
frequency sweep.
"""
import math
import os
import sys
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from geometry_io import parse_geometry, build_geometry_snapshot
from bor_dispatch import solve_monostatic_rcs_bor, solve_adaptive_frequency_sweep_bor
from grim_io import export_result_to_grim
from solver_quality import evaluate_mesh_convergence, scale_snapshot_panel_density
import mie_sphere as M

warnings.filterwarnings("ignore", category=RuntimeWarning)
ok = True
WORKERS = max(1, (os.cpu_count() or 2) - 1)


def gate(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {label} {detail}")


def geo_lines(pts):
    return [f"{float(p0[0])!r} {float(p0[1])!r} {float(p1[0])!r} {float(p1[1])!r}"
            for p0, p1 in zip(pts[:-1], pts[1:])]


def snap_of(text):
    title, segs, ibcs, diels = parse_geometry(text)
    return build_geometry_snapshot(title, segs, ibcs, diels)


def sphere_pts(a, n, z0=0.0):
    th = np.linspace(0.0, math.pi, n + 1)
    return np.column_stack([a * np.sin(th), z0 + a * np.cos(th)])


def solve(snap, freqs, aspects, pol, **kw):
    kw.setdefault("workers", WORKERS)
    return solve_monostatic_rcs_bor(snap, freqs, aspects, pol,
                                    geometry_units="meters", **kw)


def db_curve(result):
    return np.array([s["rcs_db"] for s in
                     sorted(result["samples"], key=lambda s: s["theta_inc_deg"])])


# ═════════════════════════════════════════════════════════════════════════════
# 1. Rocket profile: tangent-ogive nose + cylinder + flat base (3 segments,
#    exercises multi-segment stitching), full 0-180 aspect sweep, both pols.
# ═════════════════════════════════════════════════════════════════════════════

A_R = 0.2          # body radius (m)
L_NOSE = 0.6       # nose length
L_CYL = 1.2        # cylinder length
FREQ_GHZ = 0.6

# tangent ogive: circular arc from the tip (0, z_tip) to a tangent meeting
# with the cylinder at (A_R, z_sh); arc center at (A_R - R_og, z_sh).
z_tip = L_NOSE + L_CYL / 2.0
z_sh = L_CYL / 2.0
R_og = (A_R ** 2 + L_NOSE ** 2) / (2.0 * A_R)
phi = np.linspace(math.asin(L_NOSE / R_og), 0.0, 25)
nose = np.column_stack([(A_R - R_og) + R_og * np.cos(phi),
                        z_sh + R_og * np.sin(phi)])
nose[0] = [0.0, z_tip]
nose[-1] = [A_R, z_sh]
cyl = np.array([[A_R, z_sh], [A_R, -L_CYL / 2.0]])
base = np.array([[A_R, -L_CYL / 2.0], [0.0, -L_CYL / 2.0]])

rocket_geo = ("Title: rocket\n"
              "Segment: nose 2\nproperties: 2 0 0 0 0\n" + "\n".join(geo_lines(nose)) +
              "\nSegment: body 2\nproperties: 2 0 0 0 0\n" + "\n".join(geo_lines(cyl)) +
              "\nSegment: base 2\nproperties: 2 0 0 0 0\n" + "\n".join(geo_lines(base)) +
              "\nIBCS_Resistances:\nDielectrics:\n")
rocket = snap_of(rocket_geo)

aspects = [float(a) for a in range(0, 181, 5)]
res_vv = solve(rocket, [FREQ_GHZ], aspects, "VV")
res_hh = solve(rocket, [FREQ_GHZ], aspects, "HH")
meta = res_vv["metadata"]["per_frequency"][0]
print(f"      rocket sweep: {meta['n_unknowns']} unknowns, "
      f"{meta['modes_used']} modes, formulation {res_vv['metadata']['formulation']}")

vv = db_curve(res_vv)
hh = db_curve(res_hh)
gate("rocket full 0-180 sweep finite", np.all(np.isfinite(vv)) and np.all(np.isfinite(hh)),
     f"(VV span {vv.min():.1f}..{vv.max():.1f} dBsm)")
gate("rocket nose-on VV == HH (BoR symmetry)", abs(vv[0] - hh[0]) < 1e-3,
     f"(diff {abs(vv[0] - hh[0]):.2e} dB)")
gate("rocket tail-on VV == HH", abs(vv[-1] - hh[-1]) < 1e-3,
     f"(diff {abs(vv[-1] - hh[-1]):.2e} dB)")
i_bs = aspects.index(90.0)
gate("rocket broadside is the sweep max (long cylinder)",
     vv[i_bs] == vv.max() and hh[i_bs] == hh.max(),
     f"(VV broadside {vv[i_bs]:.1f} dBsm)")

# ── mode convergence: auto truncation stopped early AND adding headroom
#    does not move the answer ────────────────────────────────────────────────
sub = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
base_sub = solve(rocket, [FREQ_GHZ], sub, "VV")
more = solve(rocket, [FREQ_GHZ], sub, "VV",
             n_modes=meta["modes_used"] + 6)
d_mode = np.max(np.abs(db_curve(base_sub) - db_curve(more)))
gate("rocket mode convergence (+6 modes < 0.05 dB)", d_mode < 0.05,
     f"(delta {d_mode:.4f} dB)")

# ── mesh convergence: 1.5x element density through the shared gate ──────────
fine = scale_snapshot_panel_density(rocket, 1.5)
res_vv_fine = solve(fine, [FREQ_GHZ], aspects, "VV")
mc = evaluate_mesh_convergence(res_vv, res_vv_fine,
                               rms_limit_db=0.15, max_abs_limit_db=0.5)
gate("rocket mesh convergence (1.5x, RMS <= 0.15 dB)", mc["passed"],
     f"(rms {mc['rms_db']:.3f} dB, max {mc['max_abs_db']:.3f} dB)")

# ═════════════════════════════════════════════════════════════════════════════
# 2. Dispatch material paths vs Mie (sphere family through .geo text).
# ═════════════════════════════════════════════════════════════════════════════

a = 0.1
f_ghz = 2.0 * M.C0 / (2 * math.pi * a) / 1e9
fq = f_ghz * 1e9
sph48 = "\n".join(geo_lines(sphere_pts(a, 48)))

cases = [
    ("PEC sphere (CFIE)",
     "Title: t\nSegment: s 2\nproperties: 2 0 0 0 0\n" + sph48 +
     "\nIBCS_Resistances:\nDielectrics:\n",
     M.sigma_pec_sphere(a, fq)),
    ("IBC sphere (inline constant)",
     "Title: t\nSegment: s 2\nproperties: 2 0 1 0 0\n" + sph48 +
     "\nIBCS_Resistances:\n1 constant 150 -80 0 0\nDielectrics:\n",
     M.sigma_impedance_sphere(a, fq, 150 - 80j)),
    ("dielectric sphere (PMCHWT)",
     "Title: t\nSegment: s 3\nproperties: 3 0 0 1 0\n" + sph48 +
     "\nIBCS_Resistances:\nDielectrics:\n1 2.5 -1.2 1 0\n",
     M.sigma_dielectric_sphere(a, 2.5 - 1.2j, 1.0, fq)),
    ("coated PEC sphere (multi-region)",
     "Title: t\nSegment: outer 3\nproperties: 3 0 0 1 0\n" + sph48 +
     "\nSegment: core 4\nproperties: 4 0 0 1 0\n" +
     "\n".join(geo_lines(sphere_pts(0.06, 32))) +
     "\nIBCS_Resistances:\nDielectrics:\n1 3 -0.5 1 0\n",
     M.sigma_coated_pec_sphere(0.06, a, 3 - 0.5j, 1.0, fq)),
]
for label, geo, ref in cases:
    out = solve(snap_of(geo), [f_ghz], [0.0, 60.0], "VV")
    w = max(abs(s["rcs_db"] - 10 * math.log10(ref)) for s in out["samples"])
    gate(f"dispatch {label} <= 0.1 dB", w < 0.1, f"(worst {w:.3f})")

# tapered IBC through the .geo IBCS table: mirror symmetry on a sphere
taper_ab = ("Title: t\nSegment: s 2\nproperties: 2 0 1 0 0\n" + sph48 +
            "\nIBCS_Resistances:\n1 linear 50 0 400 0\nDielectrics:\n")
taper_ba = ("Title: t\nSegment: s 2\nproperties: 2 0 1 0 0\n" + sph48 +
            "\nIBCS_Resistances:\n1 linear 400 0 50 0\nDielectrics:\n")
oa = solve(snap_of(taper_ab), [f_ghz], [35.0], "VV")
ob = solve(snap_of(taper_ba), [f_ghz], [145.0], "VV")
d = abs(oa["samples"][0]["rcs_db"] - ob["samples"][0]["rcs_db"])
gate("dispatch tapered IBC mirror symmetry", d < 1e-6, f"({d:.2e} dB)")

# ═════════════════════════════════════════════════════════════════════════════
# 3. .grim export round-trip with dBsm units.
# ═════════════════════════════════════════════════════════════════════════════

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "rocket_vv.grim")
    written = export_result_to_grim(res_vv, path, history="phase4 gate")
    data = np.load(written[0], allow_pickle=False)
    import json
    units = json.loads(str(data["units"]))
    gate(".grim units say dBsm / sigma_3d",
         units.get("rcs_log_unit") == "dBsm" and
         units.get("rcs_linear_quantity") == "sigma_3d")
    az = np.asarray(data["azimuths"], dtype=float)
    power = np.asarray(data["rcs_power"], dtype=float)[:, 0, 0, 0]
    by_th = {float(s["theta_scat_deg"]): float(s["rcs_linear"])
             for s in res_vv["samples"]}
    match = all(abs(power[i] - by_th[float(az[i])]) <=
                1e-6 * max(by_th[float(az[i])], 1e-30) for i in range(len(az)))
    gate(".grim power grid round-trips the samples", match,
         f"({len(az)} aspects)")
    gate(".grim polarization label VV", str(data["polarizations"][0]) == "VV")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Adaptive frequency sweep (small PEC sphere through the resonance region).
# ═════════════════════════════════════════════════════════════════════════════

sph_geo = ("Title: t\nSegment: s 2\nproperties: 2 0 0 0 0\n" +
           "\n".join(geo_lines(sphere_pts(a, 32))) +
           "\nIBCS_Resistances:\nDielectrics:\n")
f_lo = 0.8 * M.C0 / (2 * math.pi * a) / 1e9
f_hi = 2.4 * M.C0 / (2 * math.pi * a) / 1e9
ad = solve_adaptive_frequency_sweep_bor(
    snap_of(sph_geo), f_lo, f_hi, [0.0], "VV",
    initial_points=5, max_refinements=2, db_threshold=0.5,
    geometry_units="meters", workers=WORKERS)
n_final = ad["metadata"]["final_point_count"]
freqs_seen = sorted({s["frequency_ghz"] for s in ad["samples"]})
gate("adaptive sweep refined the resonance region",
     ad["metadata"]["refinement_count"] >= 1 and n_final > 5,
     f"({n_final} freqs, {ad['metadata']['refinement_count']} refinements)")
gate("adaptive sweep: no duplicate frequencies",
     len(freqs_seen) == len(ad["samples"]))
mie_db = [10 * math.log10(M.sigma_pec_sphere(a, f * 1e9)) for f in freqs_seen]
sol_db = [s["rcs_db"] for s in sorted(ad["samples"], key=lambda s: s["frequency_ghz"])]
w = max(abs(m - s) for m, s in zip(mie_db, sol_db))
gate("adaptive sweep tracks Mie curve <= 0.1 dB", w < 0.1, f"(worst {w:.3f})")

# expand_to_360: exact axisymmetric mirror (sigma(360-th) = sigma(th)),
# seam directions 0/360 and 180 not duplicated.
r360 = solve_monostatic_rcs_bor(snap_of(sph_geo), [2.0 * M.C0 / (2 * math.pi * a) / 1e9],
                                [0.0, 45.0, 90.0, 135.0, 180.0], "VV",
                                geometry_units="meters", workers=WORKERS,
                                expand_to_360=True)
angs = sorted({s["theta_inc_deg"] for s in r360["samples"]})
by = {s["theta_inc_deg"]: s for s in r360["samples"]}
gate("expand_to_360 grid (no seam duplicates)",
     angs == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
gate("expand_to_360 mirror exact (power + amplitude)",
     all(by[360.0 - th]["rcs_linear"] == by[th]["rcs_linear"] and
         by[360.0 - th]["rcs_amp_imag"] == by[th]["rcs_amp_imag"]
         for th in (45.0, 90.0, 135.0)))

# radar-frame (az, el) polarimetric grid from the rocket's two co-pol sweeps
from bor_dispatch import bor_az_el_grid
from grim_io import save_bor_az_el_grim

az_g = [float(x) for x in range(0, 181, 15)]
el_g = [-60.0, -30.0, 0.0, 30.0, 60.0]
grid = bor_az_el_grid(res_vv, res_hh, az_g, el_g)
lin_v = {s["theta_inc_deg"]: s["rcs_linear"] for s in res_vv["samples"]}
lin_h = {s["theta_inc_deg"]: s["rcs_linear"] for s in res_hh["samples"]}
j0 = el_g.index(0.0)
# horizontal axis: the waterline meridian plane is horizontal, so radar-VV
# on el=0 equals the SOLVER's HH sweep (and vice versa)
w = max(max(abs(grid["sigma"]["VV"][i, j0, 0] / lin_h[a] - 1),
            abs(grid["sigma"]["HH"][i, j0, 0] / lin_v[a] - 1))
        for i, a in enumerate(az_g) if 0 < a < 180)
gate("az/el grid: waterline == solver sweep (V/H swap, horizontal axis)",
     w < 1e-10, f"(rel {w:.1e})")
gate("az/el grid: waterline and az=0 cuts have zero cross-pol",
     np.max(grid["sigma"]["VH"][:, j0, 0]) == 0.0 and
     np.max(grid["sigma"]["VH"][az_g.index(0.0), :, 0]) == 0.0)
# polarimetric Frobenius invariant at every grid point
th_s = np.asarray(sorted(lin_v))
amp_v = {s["theta_inc_deg"]: complex(s["rcs_amp_real"], s["rcs_amp_imag"])
         for s in res_vv["samples"]}
amp_h = {s["theta_inc_deg"]: complex(s["rcs_amp_real"], s["rcs_amp_imag"])
         for s in res_hh["samples"]}
thq = grid["theta_map_deg"].ravel()
Fv = (np.interp(thq, th_s, [amp_v[t].real for t in th_s]) +
      1j * np.interp(thq, th_s, [amp_v[t].imag for t in th_s]))
Fh = (np.interp(thq, th_s, [amp_h[t].real for t in th_s]) +
      1j * np.interp(thq, th_s, [amp_h[t].imag for t in th_s]))
lhs = (np.abs(grid["amp"]["VV"][..., 0].ravel()) ** 2 +
       np.abs(grid["amp"]["HH"][..., 0].ravel()) ** 2 +
       2 * np.abs(grid["amp"]["VH"][..., 0].ravel()) ** 2)
inv = np.max(np.abs(lhs / (np.abs(Fv) ** 2 + np.abs(Fh) ** 2) - 1))
gate("az/el grid: polarimetric invariant", inv < 1e-12, f"(rel {inv:.1e})")
# vertical axis: zero tilt everywhere, labels coincide, no cross-pol
gv = bor_az_el_grid(res_vv, res_hh, [0.0, 90.0], [-45.0, 0.0, 45.0],
                    axis_el_deg=90.0)
gate("az/el grid: vertical axis has zero cross-pol",
     np.max(gv["sigma"]["VH"]) <= 1e-20 * np.max(gv["sigma"]["VV"]))
paths = save_bor_az_el_grim(grid, os.path.join(tempfile.mkdtemp(), "azel.grim"))
d0 = np.load(paths[0])
gate("az/el .grim export (real elevation axis, 3 channels)",
     len(paths) == 3 and d0["rcs_power"].shape == (len(az_g), len(el_g), 1, 1))

print("ALL PHASE-4 GATES PASS" if ok else "PHASE-4 GATES FAILED")
