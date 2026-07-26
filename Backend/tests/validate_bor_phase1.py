"""Phase-1 gate battery: BoR PEC EFIE vs analytic/cross-solver references."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bor_solver import solve_bor_pec, sphere_generatrix, cylinder_generatrix
import mie_sphere as M

ok = True

def gate(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {label} {detail}")

# 1. PEC sphere vs Mie: aspects x pols x sizes, target 0.1 dB.
a = 0.1
for ka, n in ((0.5, 24), (1.0, 32), (3.0, 48), (6.0, 96)):
    f = ka * M.C0 / (2 * math.pi * a)
    ref = M.sigma_pec_sphere(a, f)
    out = solve_bor_pec(sphere_generatrix(a, n), f, [0.0, 30.0, 45.0, 90.0])
    errs = [abs(10 * math.log10(s / ref)) for s in out["sigma_vv"] + out["sigma_hh"]]
    gate(f"sphere ka={ka} all aspects/pols <= 0.1 dB", max(errs) < 0.1,
         f"(worst {max(errs):.3f} dB, modes {out['modes_used']})")

# 2. Cross-solver: broadside closed cylinder vs 2D x 2L^2/lambda (strip approx tol 0.15 dB).
from rcs_solver import solve_monostatic_rcs_2d


def make_circle_segment(name, seg_type, radius, n_prim):
    """Clockwise polygon used by the 2-D strip cross-check.

    Keep this local: importing the executable validate_mie.py battery would run
    that entire suite and exit before the remaining Phase-1 gates.
    """
    angles = -2.0 * math.pi * np.arange(n_prim + 1) / n_prim
    points = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles)])
    pairs = [
        {"x1": points[i, 0], "y1": points[i, 1],
         "x2": points[i + 1, 0], "y2": points[i + 1, 1]}
        for i in range(n_prim)
    ]
    return {
        "name": name,
        "seg_type": str(seg_type),
        "properties": [str(seg_type), "0", "0", "0", "0"],
        "point_pairs": pairs,
    }


a_c, L, f2 = 0.05, 1.0, 3.0e9
lam = M.C0 / f2
snap = {"title": "c", "segments": [make_circle_segment("c", 2, a_c, 128)],
        "ibcs": [], "dielectrics": []}
out = solve_bor_pec(cylinder_generatrix(a_c, L, 8, 100), f2, [90.0])
for pol2d, key in (("TM", "sigma_vv"), ("TE", "sigma_hh")):
    s2d = solve_monostatic_rcs_2d(snap, [3.0], [0.0], pol2d, geometry_units="meters")["samples"][0]["rcs_linear"]
    strip = 2.0 * L ** 2 / lam * s2d
    d = 10 * math.log10(out[key][0] / strip)
    gate(f"cylinder broadside {pol2d} vs 2D strip", abs(d) < 0.15, f"(diff {d:+.3f} dB)")

# 3. Cone-sphere: nose-on VV == HH (exact symmetry), mesh convergence, physics.
def cone_sphere(a, half_deg, n_cone, n_sph):
    al = math.radians(half_deg)
    apex = np.array([0.0, a / math.sin(al)])
    T = np.array([a * math.cos(al), a * math.sin(al)])
    cone = apex[None, :] + np.linspace(0, 1, n_cone + 1)[:, None] * (T - apex)[None, :]
    th = np.linspace(math.acos(math.sin(al)), math.pi, n_sph + 1)[1:]
    return np.vstack([cone, np.column_stack([a * np.sin(th), a * np.cos(th)])])

f3 = 3.0 * M.C0 / (2 * math.pi * a)
o1 = solve_bor_pec(cone_sphere(a, 15.0, 36, 24), f3, [0.0, 180.0])
o2 = solve_bor_pec(cone_sphere(a, 15.0, 72, 48), f3, [0.0, 180.0])
gate("cone-sphere nose-on VV == HH",
     abs(10 * math.log10(o2["sigma_vv"][0] / o2["sigma_hh"][0])) < 1e-3)
conv = abs(10 * math.log10(o1["sigma_vv"][0] / o2["sigma_vv"][0]))
gate("cone-sphere nose-on mesh-converged", conv < 0.1, f"(delta {conv:.3f} dB)")
# Rear-on (viewing the sphere end): dominated by the sphere specular,
# modulated by the cone shadow-boundary contribution -> within a few dB
# of the isolated Mie sphere.
rear_vs_mie = abs(10 * math.log10(o2["sigma_vv"][1] / M.sigma_pec_sphere(a, f3)))
gate("cone-sphere rear-on ~ isolated Mie sphere", rear_vs_mie < 4.0,
     f"(diff {rear_vs_mie:.1f} dB)")

print("ALL PHASE-1 GATES PASS" if ok else "PHASE-1 GATES FAILED")
