"""Multi-layer coating gate battery (phase 6): two-layer stacks and coating
patches terminating on an underlying coating (dielectric triple junctions).

Anchors: a new multilayer Mie reference (validated against its own
degenerate limits), the bit-level agreement of the generic multi-region
assembler with the phase-3 solver, exact eps=1 / same-material limits for
the patch, and the usual symmetry/convergence gates.
"""
import math
import sys
import warnings

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from bor_solver import (solve_bor_coated_pec, solve_bor_coated2_pec,
                        solve_bor_coated_n_pec, solve_bor_coating_patch,
                        sphere_generatrix)
from geometry_io import parse_geometry, build_geometry_snapshot
from bor_dispatch import solve_monostatic_rcs_bor
import mie_sphere as M

warnings.filterwarnings("ignore", category=RuntimeWarning)
ok = True


def gate(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {label} {detail}")


def arc(r, th0, th1, n):
    th = np.linspace(th0, th1, n + 1)
    return np.column_stack([r * np.sin(th), r * np.cos(th)])


B, R1, T2 = 0.06, 0.08, 0.02
F = 2.0 * M.C0 / (2 * math.pi * 0.1)
ASPECTS = [0.0, 45.0, 90.0, 135.0, 180.0]
EPS_IN = 4.0 - 1.0j

# 1. Multilayer Mie reference degenerate anchors.
s1 = M.sigma_coated_pec_sphere(B, 0.1, 3.0 - 0.5j, 1.0, F)
s2 = M.sigma_multilayer_pec_sphere([B, 0.1], [3.0 - 0.5j], [1.0], F)
d1 = abs(10 * math.log10(s1 / s2))
s3 = M.sigma_multilayer_pec_sphere([B, 0.08, 0.1],
                                   [3.0 - 0.5j, 3.0 - 0.5j], [1.0, 1.0], F)
d2 = abs(10 * math.log10(s1 / s3))
s4 = M.sigma_multilayer_pec_sphere([B, 0.08], [2.5 - 1.0j], [1.0], F)
s5 = M.sigma_multilayer_pec_sphere([B, 0.08, 0.1], [2.5 - 1.0j, 1.0],
                                   [1.0, 1.0], F)
d3 = abs(10 * math.log10(s4 / s5))
gate("multilayer Mie degenerate anchors <= 1e-10 dB", max(d1, d2, d3) < 1e-10,
     f"(1-layer {d1:.1e}, split {d2:.1e}, eps=1 outer {d3:.1e})")

# 2. Two-layer coated spheres vs multilayer Mie.
for label, e_i, u_i, e_o, u_o in (
    ("lossy dielectric stack", 4.0 - 1.0j, 1.0, 2.0 - 0.3j, 1.0),
    ("magnetic RAM stack", 1.8 - 0.3j, 1.6 - 0.9j, 2.0 - 0.3j, 1.0),
    ("lossless stack", 4.0, 1.0, 2.25, 1.0),
):
    ref = M.sigma_multilayer_pec_sphere([B, R1, 0.1], [e_i, e_o], [u_i, u_o], F)
    out = solve_bor_coated2_pec(sphere_generatrix(0.1, 48),
                                sphere_generatrix(R1, 40),
                                sphere_generatrix(B, 32), F, [0.0, 45.0, 90.0],
                                e_i, u_i, e_o, u_o)
    w = max(abs(10 * math.log10(s / ref))
            for s in out["sigma_vv"] + out["sigma_hh"])
    gate(f"2-layer sphere {label} <= 0.1 dB", w < 0.1, f"(worst {w:.3f})")

# ── coating patch (dielectric triple junction) ───────────────────────────────
THC = math.radians(60.0)


def patch_geo(scale=1):
    n_c, n_b, n_e = 8 * scale, 16 * scale, 3 * scale
    mid_cov = arc(R1, 0.0, THC, n_c)
    mid_bare = arc(R1, THC, math.pi, n_b)
    outer = arc(R1 + T2, 0.0, THC, n_c)
    A = np.array([R1 * math.sin(THC), R1 * math.cos(THC)])
    edge = np.column_stack([np.linspace(outer[-1][0], A[0], n_e + 1),
                            np.linspace(outer[-1][1], A[1], n_e + 1)])
    return np.vstack([outer, edge[1:]]), mid_cov, mid_bare, sphere_generatrix(B, 28 * scale)


def worst_vs(out, ref_scalar):
    return max(abs(10 * math.log10(s / ref_scalar))
               for s in out["sigma_vv"] + out["sigma_hh"])


# 3. eps_patch = 1: fictitious patch == single-layer Mie, and converging.
ref = M.sigma_coated_pec_sphere(B, R1, EPS_IN, 1.0, F)
errs = {}
for scale in (1, 2):
    patch, mc, mb, core = patch_geo(scale)
    out = solve_bor_coating_patch(patch, mc, [mb], core, F, ASPECTS,
                                  EPS_IN, 1.0, 1.0, 1.0)
    errs[scale] = worst_vs(out, ref)
gate("eps_patch=1 == single-layer Mie <= 0.2 dB", errs[1] < 0.2,
     f"(worst {errs[1]:.3f})")
gate("eps_patch=1 mesh-converging", errs[2] < errs[1],
     f"({errs[1]:.3f} -> {errs[2]:.3f} dB at 2x)")

# 4. eps_patch = eps_in: same material == bumped-profile single-region solve.
patch, mc, mb, core = patch_geo(2)
bumped = np.vstack([patch, mb[1:]])
ref2 = solve_bor_coated_pec(bumped, core, F, ASPECTS, EPS_IN, 1.0)
out = solve_bor_coating_patch(patch, mc, [mb], core, F, ASPECTS,
                              EPS_IN, 1.0, EPS_IN, 1.0)
w = max(abs(10 * math.log10(a / b_)) for a, b_ in
        zip(out["sigma_vv"] + out["sigma_hh"],
            ref2["sigma_vv"] + ref2["sigma_hh"]))
gate("eps_patch=eps_in == bumped single-region <= 0.2 dB", w < 0.2,
     f"(worst {w:.3f})")

# 5. Full-coverage patch (no junctions) == two-layer solver, bit-consistent.
e_o = 2.0 - 0.3j
o1 = solve_bor_coated2_pec(sphere_generatrix(0.1, 40), sphere_generatrix(R1, 32),
                           sphere_generatrix(B, 28), F, [0.0, 60.0],
                           EPS_IN, 1.0, e_o, 1.0)
o2 = solve_bor_coating_patch(sphere_generatrix(0.1, 40), sphere_generatrix(R1, 32),
                             [], sphere_generatrix(B, 28), F, [0.0, 60.0],
                             EPS_IN, 1.0, e_o, 1.0)
d = max(abs(np.array(o1["sigma_vv"]) / np.array(o2["sigma_vv"]) - 1).max(),
        abs(np.array(o1["sigma_hh"]) / np.array(o2["sigma_hh"]) - 1).max())
gate("full-coverage patch == 2-layer solver", d < 1e-10, f"(rel {d:.1e})")

# 6. Mirror symmetry with a lossy magnetic patch.
def mirror(p):
    q = np.asarray(p).copy()
    q[:, 1] = -q[:, 1]
    return q[::-1]

em, um = 2.5 - 0.8j, 1.4 - 0.6j
patch, mc, mb, core = patch_geo(1)
oa = solve_bor_coating_patch(patch, mc, [mb], core, F, [35.0],
                             EPS_IN, 1.0, em, um)
ob = solve_bor_coating_patch(mirror(patch), mirror(mc), [mirror(mb)],
                             mirror(core), F, [145.0], EPS_IN, 1.0, em, um)
d = max(abs(10 * math.log10(oa["sigma_vv"][0] / ob["sigma_vv"][0])),
        abs(10 * math.log10(oa["sigma_hh"][0] / ob["sigma_hh"][0])))
gate("mirror symmetry (lossy magnetic patch)", d < 1e-6, f"({d:.2e} dB)")

# 7. RAM patch mesh convergence.
outs = []
for scale in (1, 2):
    patch, mc, mb, core = patch_geo(scale)
    outs.append(solve_bor_coating_patch(patch, mc, [mb], core, F, ASPECTS,
                                        EPS_IN, 1.0, em, um))
dd = max(abs(10 * math.log10(a / b_)) for a, b_ in
         zip(outs[0]["sigma_vv"] + outs[0]["sigma_hh"],
             outs[1]["sigma_vv"] + outs[1]["sigma_hh"]))
gate("RAM patch mesh convergence (2x) <= 0.2 dB", dd < 0.2, f"(worst {dd:.3f})")

# 8. Dispatch: TYPE 3+5+4 full two-layer and patch layouts.
def geo_lines(pts):
    return [f"{float(p0[0])!r} {float(p0[1])!r} {float(p1[0])!r} {float(p1[1])!r}"
            for p0, p1 in zip(pts[:-1], pts[1:])]

e_i, e_o = 4.0 - 1.0j, 2.0 - 0.3j
ref = M.sigma_multilayer_pec_sphere([B, R1, 0.1], [e_i, e_o], [1.0, 1.0], F)
geo = ("Title: twolayer\n"
       "Segment: outer 3\nproperties: 3 0 0 2 0\n" +
       "\n".join(geo_lines(sphere_generatrix(0.1, 48))) +
       "\nSegment: mid 5\nproperties: 5 0 0 2 1\n" +
       "\n".join(geo_lines(sphere_generatrix(R1, 40))) +
       "\nSegment: core 4\nproperties: 4 0 0 1 0\n" +
       "\n".join(geo_lines(sphere_generatrix(B, 32))) +
       "\nIBCS_Resistances:\nDielectrics:\n1 4 -1 1 0\n2 2 -0.3 1 0\n")
title, segs, ibcs, diels = parse_geometry(geo)
snap = build_geometry_snapshot(title, segs, ibcs, diels)
res = solve_monostatic_rcs_bor(snap, [F / 1e9], [0.0, 90.0], "VV",
                               geometry_units="meters")
w = max(abs(s["rcs_db"] - 10 * math.log10(ref)) for s in res["samples"])
gate("dispatch TYPE 3+5+4 two-layer == multilayer Mie <= 0.1 dB", w < 0.1,
     f"(worst {w:.3f}, {res['metadata']['formulation']})")

patch, mc, mb, core = patch_geo(1)
ref = M.sigma_coated_pec_sphere(B, R1, EPS_IN, 1.0, F)
geo = ("Title: patch\n"
       "Segment: patch 3\nproperties: 3 0 0 2 0\n" + "\n".join(geo_lines(patch)) +
       "\nSegment: mid 5\nproperties: 5 0 0 2 1\n" + "\n".join(geo_lines(mc)) +
       "\nSegment: exposed 3\nproperties: 3 0 0 1 0\n" + "\n".join(geo_lines(mb)) +
       "\nSegment: core 4\nproperties: 4 0 0 1 0\n" + "\n".join(geo_lines(core)) +
       "\nIBCS_Resistances:\nDielectrics:\n1 4 -1 1 0\n2 1 0 1 0\n")
title, segs, ibcs, diels = parse_geometry(geo)
snap = build_geometry_snapshot(title, segs, ibcs, diels)
res = solve_monostatic_rcs_bor(snap, [F / 1e9], [0.0, 90.0], "VV",
                               geometry_units="meters")
w = max(abs(s["rcs_db"] - 10 * math.log10(ref)) for s in res["samples"])
gate("dispatch TYPE 3+5+4 patch (eps=1) == single-layer Mie <= 0.2 dB",
     w < 0.2, f"(worst {w:.3f}, {res['metadata']['formulation']})")

# ── N-layer stacks ───────────────────────────────────────────────────────────

# 9. N=2 via the N-layer entry == the dedicated two-layer solver.
o1 = solve_bor_coated_n_pec([sphere_generatrix(0.1, 40), sphere_generatrix(R1, 32)],
                            sphere_generatrix(B, 28), F, [0.0, 60.0],
                            [EPS_IN, e_o], [1.0, 1.0])
o2 = solve_bor_coated2_pec(sphere_generatrix(0.1, 40), sphere_generatrix(R1, 32),
                           sphere_generatrix(B, 28), F, [0.0, 60.0],
                           EPS_IN, 1.0, e_o, 1.0)
d = max(abs(np.array(o1["sigma_vv"]) / np.array(o2["sigma_vv"]) - 1).max(),
        abs(np.array(o1["sigma_hh"]) / np.array(o2["sigma_hh"]) - 1).max())
gate("N-layer entry (N=2) == two-layer solver", d < 1e-10, f"(rel {d:.1e})")

# 10. Three-layer magnetic lossy stack vs multilayer Mie.
radii = [0.05, 0.065, 0.08, 0.1]
eps3 = [5.0 - 1.5j, 3.0 - 0.6j, 1.8 - 0.2j]
mu3 = [1.0, 1.3 - 0.4j, 1.0]
ref = M.sigma_multilayer_pec_sphere(radii, eps3, mu3, F)
out = solve_bor_coated_n_pec([sphere_generatrix(0.1, 48), sphere_generatrix(0.08, 44),
                              sphere_generatrix(0.065, 40)],
                             sphere_generatrix(0.05, 32), F, [0.0, 45.0, 90.0],
                             eps3, mu3)
w = max(abs(10 * math.log10(s / ref)) for s in out["sigma_vv"] + out["sigma_hh"])
gate("3-layer magnetic stack vs multilayer Mie <= 0.1 dB", w < 0.1,
     f"(worst {w:.3f})")

# 11. Dispatch: three-layer .geo (two TYPE 5 flag pairs).
geo = ("Title: threelayer\n"
       "Segment: outer 3\nproperties: 3 0 0 3 0\n" +
       "\n".join(geo_lines(sphere_generatrix(0.1, 48))) +
       "\nSegment: mid23 5\nproperties: 5 0 0 3 2\n" +
       "\n".join(geo_lines(sphere_generatrix(0.08, 44))) +
       "\nSegment: mid12 5\nproperties: 5 0 0 2 1\n" +
       "\n".join(geo_lines(sphere_generatrix(0.065, 40))) +
       "\nSegment: core 4\nproperties: 4 0 0 1 0\n" +
       "\n".join(geo_lines(sphere_generatrix(0.05, 32))) +
       "\nIBCS_Resistances:\nDielectrics:\n"
       "1 5 -1.5 1 0\n2 3 -0.6 1.3 -0.4\n3 1.8 -0.2 1 0\n")
title, segs, ibcs, diels = parse_geometry(geo)
snap = build_geometry_snapshot(title, segs, ibcs, diels)
res = solve_monostatic_rcs_bor(snap, [F / 1e9], [0.0, 90.0], "VV",
                               geometry_units="meters")
w = max(abs(s["rcs_db"] - 10 * math.log10(ref)) for s in res["samples"])
gate("dispatch 3-layer TYPE 3+5+5+4 vs multilayer Mie <= 0.1 dB", w < 0.1,
     f"(worst {w:.3f}, {res['metadata']['formulation']})")

print("ALL MULTILAYER GATES PASS" if ok else "MULTILAYER GATES FAILED")
