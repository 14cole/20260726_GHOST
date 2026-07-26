"""Junction gate battery: partial coatings terminating on the PEC surface.

There is no analytic reference for a partially coated sphere, so the gates
are exact limits and symmetries — the same discipline as the other phases:

  * eps=1 coating of ANY shape must scatter exactly like the bare PEC body
    (this is the gate that caught the junction row-orientation bug: every
    row block must carry the same region-equation sign or the Q^T fold at
    the junction sums the layer equation wrongly — it failed at +9.4 dB);
  * full coverage (no junctions) must equal the phase-3 coated solver;
  * a shrinking bare cap must converge to the fully-coated Mie answer;
  * mirror symmetry must hold exactly for a lossy magnetic cap;
  * mesh refinement must converge (junction wedge behavior is only
    approximated by the tied nodal bases, so this is checked explicitly).
"""
import math
import sys
import warnings

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from bor_solver import (solve_bor_partial_coating, solve_bor_coated_pec,
                        solve_bor, sphere_generatrix)
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


B, T = 0.1, 0.03
F = 2.0 * M.C0 / (2 * math.pi * B)          # core ka = 2
ASPECTS = [0.0, 45.0, 90.0, 135.0, 180.0]


def cap_geometry(thc, n_cov, n_bare, n_edge=4):
    """Nose-cap coating of coverage angle thc: uniform thickness T with a
    blunt radial edge terminating on the core at the junction."""
    covered = arc(B, 0.0, thc, n_cov)
    bare = arc(B, thc, math.pi, n_bare)
    outer = arc(B + T, 0.0, thc, n_cov)
    A = np.array([B * math.sin(thc), B * math.cos(thc)])
    E = outer[-1]
    edge = np.column_stack([np.linspace(E[0], A[0], n_edge + 1),
                            np.linspace(E[1], A[1], n_edge + 1)])
    return np.vstack([outer, edge[1:]]), covered, [bare]


def worst_db(out, ref):
    return max(abs(10 * math.log10(s / ref))
               for s in out["sigma_vv"] + out["sigma_hh"])


# 1. Full coverage (no junctions) == phase-3 coated solver, bit-consistent.
o1 = solve_bor_coated_pec(sphere_generatrix(B + T, 40), sphere_generatrix(B, 28),
                          F, [0.0, 60.0], 3.0 - 0.5j)
o2 = solve_bor_partial_coating(sphere_generatrix(B + T, 40), sphere_generatrix(B, 28),
                               [], F, [0.0, 60.0], 3.0 - 0.5j)
d = max(abs(np.array(o1["sigma_vv"]) / np.array(o2["sigma_vv"]) - 1).max(),
        abs(np.array(o1["sigma_hh"]) / np.array(o2["sigma_hh"]) - 1).max())
gate("full coverage == phase-3 coated solver", d < 1e-10, f"(rel {d:.1e})")

# 2. eps=1 caps == bare PEC Mie, and mesh-converging.
pec = M.sigma_pec_sphere(B, F)
errs = {}
for thc_deg, scale in ((60.0, 1), (60.0, 2), (120.0, 1)):
    iface, cov, bares = cap_geometry(math.radians(thc_deg),
                                     8 * scale, 16 * scale, 4 * scale)
    out = solve_bor_partial_coating(iface, cov, bares, F, ASPECTS, 1.0, 1.0)
    errs[(thc_deg, scale)] = worst_db(out, pec)
gate("eps=1 cap 60 deg == PEC Mie <= 0.15 dB", errs[(60.0, 1)] < 0.15,
     f"(worst {errs[(60.0, 1)]:.3f})")
gate("eps=1 cap 60 deg mesh-converging", errs[(60.0, 2)] < errs[(60.0, 1)],
     f"({errs[(60.0, 1)]:.3f} -> {errs[(60.0, 2)]:.3f} dB at 2x)")
gate("eps=1 cap 120 deg == PEC Mie <= 0.3 dB", errs[(120.0, 1)] < 0.3,
     f"(worst {errs[(120.0, 1)]:.3f})")

# 3. eps=1 BAND coating (two junctions, two bare pieces).
th1, th2 = math.radians(50), math.radians(120)
covered = arc(B, th1, th2, 12)
bare_top = arc(B, 0.0, th1, 8)
bare_bot = arc(B, th2, math.pi, 10)
A1 = np.array([B * math.sin(th1), B * math.cos(th1)])
O1 = np.array([(B + T) * math.sin(th1), (B + T) * math.cos(th1)])
A2 = np.array([B * math.sin(th2), B * math.cos(th2)])
O2 = np.array([(B + T) * math.sin(th2), (B + T) * math.cos(th2)])
e1 = np.column_stack([np.linspace(A1[0], O1[0], 5), np.linspace(A1[1], O1[1], 5)])
e2 = np.column_stack([np.linspace(O2[0], A2[0], 5), np.linspace(O2[1], A2[1], 5)])
iface_band = np.vstack([e1, arc(B + T, th1, th2, 12)[1:], e2[1:]])
out = solve_bor_partial_coating(iface_band, covered, [bare_top, bare_bot],
                                F, ASPECTS, 1.0, 1.0)
w = worst_db(out, pec)
gate("eps=1 band (2 junctions) == PEC Mie <= 0.25 dB", w < 0.25,
     f"(worst {w:.3f}, junctions {out['n_junctions']})")

# 4. Shrinking bare cap converges to the fully-coated Mie answer.
eps, mu = 3.0 - 0.5j, 1.0
coated_ref = M.sigma_coated_pec_sphere(B, B + T, eps, mu, F)
diffs = {}
for thc_deg in (150.0, 172.5):
    thc = math.radians(thc_deg)
    n_cov = int(28 * thc_deg / 180)
    n_bare = max(4, int(28 * (180 - thc_deg) / 180) + 2)
    iface, cov, bares = cap_geometry(thc, n_cov, n_bare, 5)
    out = solve_bor_partial_coating(iface, cov, bares, F, [0.0], eps, mu)
    diffs[thc_deg] = abs(10 * math.log10(out["sigma_vv"][0] / coated_ref))
gate("shrinking bare cap -> coated Mie",
     diffs[172.5] < diffs[150.0] and diffs[172.5] < 0.05,
     f"(150deg {diffs[150.0]:.3f} dB -> 172.5deg {diffs[172.5]:.3f} dB)")

# 5. Mirror symmetry: lossy magnetic nose cap at theta == tail cap at 180-theta.
def mirror(pts):
    q = np.asarray(pts).copy()
    q[:, 1] = -q[:, 1]
    return q[::-1]

eps_m, mu_m = 3.0 - 1.0j, 1.5 - 0.5j
iface, cov, bares = cap_geometry(math.radians(60), 10, 20)
oa = solve_bor_partial_coating(iface, cov, bares, F, [35.0], eps_m, mu_m)
ob = solve_bor_partial_coating(mirror(iface), mirror(cov), [mirror(bares[0])],
                               F, [145.0], eps_m, mu_m)
d = max(abs(10 * math.log10(oa["sigma_vv"][0] / ob["sigma_vv"][0])),
        abs(10 * math.log10(oa["sigma_hh"][0] / ob["sigma_hh"][0])))
gate("mirror symmetry (lossy magnetic cap)", d < 1e-6, f"({d:.2e} dB)")

# 6. RAM cap mesh convergence (real coating, both pols over the sweep).
outs = []
for scale in (1, 2):
    iface, cov, bares = cap_geometry(math.radians(90), 10 * scale,
                                     10 * scale, 4 * scale)
    outs.append(solve_bor_partial_coating(iface, cov, bares, F, ASPECTS,
                                          1.8 - 0.3j, 1.6 - 0.9j))
dd = max(abs(10 * math.log10(a / b)) for a, b in
         zip(outs[0]["sigma_vv"] + outs[0]["sigma_hh"],
             outs[1]["sigma_vv"] + outs[1]["sigma_hh"]))
gate("RAM cap mesh convergence (2x) <= 0.2 dB", dd < 0.2, f"(worst {dd:.3f})")

# 7. Dispatch: TYPE 2+3+4 .geo snapshot routes to the junction solver.
def geo_lines(pts):
    return [f"{float(p0[0])!r} {float(p0[1])!r} {float(p1[0])!r} {float(p1[1])!r}"
            for p0, p1 in zip(pts[:-1], pts[1:])]

iface, cov, bares = cap_geometry(math.radians(60), 10, 20)
geo = ("Title: partial\n"
       "Segment: coat 3\nproperties: 3 0 0 1 0\n" + "\n".join(geo_lines(iface)) +
       "\nSegment: under 4\nproperties: 4 0 0 1 0\n" + "\n".join(geo_lines(cov)) +
       "\nSegment: bare 2\nproperties: 2 0 0 0 0\n" + "\n".join(geo_lines(bares[0])) +
       "\nIBCS_Resistances:\nDielectrics:\n1 1 0 1 0\n")
title, segs, ibcs, diels = parse_geometry(geo)
snap = build_geometry_snapshot(title, segs, ibcs, diels)
res = solve_monostatic_rcs_bor(snap, [F / 1e9], [0.0, 90.0], "VV",
                               geometry_units="meters")
w = max(abs(s["rcs_db"] - 10 * math.log10(pec)) for s in res["samples"])
gate("dispatch TYPE 2+3+4 eps=1 cap == PEC Mie <= 0.2 dB", w < 0.2,
     f"(worst {w:.3f}, {res['metadata']['formulation']})")

# ── IBC on the bare pieces ───────────────────────────────────────────────────
# Exact cross-check: with an eps=1 (fictitious) coating, the partially coated
# body IS a plain [PEC-under-cap + impedance-elsewhere] sphere — solvable by
# the validated single-surface mixed PEC/IBC path on the same core mesh.

THC = math.radians(60.0)
N_COV, N_BARE = 10, 20


def ibc_pieces(scale=1):
    iface, cov, bares = cap_geometry(THC, N_COV * scale, N_BARE * scale,
                                     4 * scale)
    return iface, cov, bares[0]


def zs_taper(n):
    """Cosine taper 0 -> 300 ohm from the junction toward the tail (the
    physical edge treatment: Z_s -> 0 at the coating junction)."""
    s = (np.arange(n) + 0.5) / n
    return (300.0 * 0.5 * (1 - np.cos(math.pi * s))).astype(complex)


def single_surface_ref(zs_bare_fn, scale):
    n_cov, n_bare = N_COV * scale, N_BARE * scale
    covered = arc(B, 0.0, THC, n_cov)
    bare = arc(B, THC, math.pi, n_bare)
    core = np.vstack([covered, bare[1:]])
    zs = np.concatenate([np.zeros(n_cov, dtype=complex), zs_bare_fn(n_bare)])
    return solve_bor(core, F, ASPECTS, formulation="efie", zs=zs)


# 8. Z_s = 0 arrays take the identical PEC-bare path.
iface, cov, bare = ibc_pieces()
o_pec = solve_bor_partial_coating(iface, cov, [bare], F, [45.0], 3.0 - 0.5j)
o_zs0 = solve_bor_partial_coating(iface, cov, [bare], F, [45.0], 3.0 - 0.5j,
                                  bare_zs=[np.zeros(N_BARE, dtype=complex)])
gate("bare_zs = 0 == PEC bare path",
     o_pec["sigma_vv"][0] == o_zs0["sigma_vv"][0])

# 9. eps=1 cap + tapered-to-zero IBC bare == single-surface mixed solve.
ref = single_surface_ref(zs_taper, 3)
rv = np.array(ref["sigma_vv"] + ref["sigma_hh"])
out = solve_bor_partial_coating(iface, cov, [bare], F, ASPECTS, 1.0, 1.0,
                                bare_zs=[zs_taper(N_BARE)])
w = max(abs(10 * np.log10(np.array(out["sigma_vv"] + out["sigma_hh"]) / rv)))
gate("eps=1 cap + tapered IBC == mixed-impedance sphere <= 0.2 dB",
     w < 0.2, f"(worst {w:.3f})")
gate("tapered junction Z_s raises no warning", not out["warnings"])

# 10. Abrupt Z_s at the junction: documented ill-defined sheet-model limit —
#     loose accuracy gate plus the warning that flags it.
ref = single_surface_ref(lambda n: np.full(n, 150 - 80j), 3)
rv = np.array(ref["sigma_vv"] + ref["sigma_hh"])
out = solve_bor_partial_coating(iface, cov, [bare], F, ASPECTS, 1.0, 1.0,
                                bare_zs=[np.full(N_BARE, 150 - 80j)])
w = max(abs(10 * np.log10(np.array(out["sigma_vv"] + out["sigma_hh"]) / rv)))
gate("eps=1 cap + abrupt IBC within 0.6 dB (known limit)", w < 0.6,
     f"(worst {w:.3f})")
gate("abrupt junction Z_s warns", bool(out["warnings"]))

# 11. Mirror symmetry with tapered IBC bare + lossy magnetic cap.
iface, cov, bare = ibc_pieces()
zs = zs_taper(N_BARE)
oa = solve_bor_partial_coating(iface, cov, [bare], F, [35.0], eps_m, mu_m,
                               bare_zs=[zs])
ob = solve_bor_partial_coating(mirror(iface), mirror(cov), [mirror(bare)],
                               F, [145.0], eps_m, mu_m, bare_zs=[zs[::-1].copy()])
d = max(abs(10 * math.log10(oa["sigma_vv"][0] / ob["sigma_vv"][0])),
        abs(10 * math.log10(oa["sigma_hh"][0] / ob["sigma_hh"][0])))
gate("mirror symmetry (tapered IBC + lossy cap)", d < 1e-6, f"({d:.2e} dB)")

# 12. Dispatch: TYPE 2 with an inline IBC taper + TYPE 3 + TYPE 4.
iface, cov, bare = ibc_pieces()
geo = ("Title: partial-ibc\n"
       "Segment: coat 3\nproperties: 3 0 0 1 0\n" + "\n".join(geo_lines(iface)) +
       "\nSegment: under 4\nproperties: 4 0 0 1 0\n" + "\n".join(geo_lines(cov)) +
       "\nSegment: bare 2\nproperties: 2 0 1 0 0\n" + "\n".join(geo_lines(bare)) +
       "\nIBCS_Resistances:\n1 cosine 0 0 300 0\nDielectrics:\n1 1 0 1 0\n")
title, segs, ibcs, diels = parse_geometry(geo)
snap = build_geometry_snapshot(title, segs, ibcs, diels)
res = solve_monostatic_rcs_bor(snap, [F / 1e9], ASPECTS, "VV",
                               geometry_units="meters")
ref = single_surface_ref(
    lambda n: np.array([150.0 * (1 - math.cos(math.pi * (i + 0.5) / n))
                        for i in range(n)], dtype=complex), 3)
w = max(abs(s["rcs_db"] - 10 * math.log10(r))
        for s, r in zip(res["samples"], ref["sigma_vv"]))
gate("dispatch TYPE 2(IBC taper)+3+4 == mixed sphere <= 0.25 dB", w < 0.25,
     f"(worst {w:.3f}, {res['metadata']['formulation']})")

# ── conductor junctions in the GENERIC multi-region assembler ────────────────
# The sigma/traversal tie rule plus the M_t = 0 conductor-line mask must
# reproduce the dedicated phase-5 solver exactly, and unlock configurations
# it cannot express (multiple independent coating patches on one body).
from bor_solver import _MultiRegionBor, _solve_multiregion

iface, cov, bares = cap_geometry(math.radians(60), 10, 20, 4)
eps_g, mu_g = 3.0 - 1.0j, 1.5 - 0.5j
ref = solve_bor_partial_coating(iface, cov, bares, F, ASPECTS, eps_g, mu_g)
sys_ = _MultiRegionBor(
    surfaces=[(iface, False), (cov, True), (bares[0], True)],
    regions=[{"medium": None, "bounds": [(0, +1), (2, +1)], "exterior": True},
             {"medium": (eps_g, mu_g), "bounds": [(0, -1), (1, +1)]}],
    freq_hz=F)
out = _solve_multiregion(sys_, F, ASPECTS, None, 1e-6, 1, None, None, "t", {})
d = max(abs(np.array(out["sigma_vv"] + out["sigma_hh"]) /
            np.array(ref["sigma_vv"] + ref["sigma_hh"]) - 1))
gate("generic assembler == phase-5 partial-coating solver", d < 1e-10,
     f"(rel {d:.1e})")

# two independent eps=1 patches (nose + tail caps, PEC band between):
# a config the dedicated solver cannot express — must equal bare PEC Mie.
T2 = 0.025
th1g, th2g = math.radians(50.0), math.radians(130.0)
cov_n = arc(B, 0.0, th1g, 8)
out_n = arc(B + T2, 0.0, th1g, 8)
A1g = np.array([B * math.sin(th1g), B * math.cos(th1g)])
e1g = np.column_stack([np.linspace(out_n[-1][0], A1g[0], 4),
                       np.linspace(out_n[-1][1], A1g[1], 4)])
if_n = np.vstack([out_n, e1g[1:]])
cov_t = arc(B, th2g, math.pi, 8)
out_t = arc(B + T2, th2g, math.pi, 8)
A2g = np.array([B * math.sin(th2g), B * math.cos(th2g)])
e2g = np.column_stack([np.linspace(A2g[0], out_t[0][0], 4),
                       np.linspace(A2g[1], out_t[0][1], 4)])
if_t = np.vstack([e2g, out_t[1:]])
bare_mid = arc(B, th1g, th2g, 14)
sys_ = _MultiRegionBor(
    surfaces=[(if_n, False), (if_t, False), (cov_n, True), (cov_t, True),
              (bare_mid, True)],
    regions=[{"medium": None, "exterior": True,
              "bounds": [(0, +1), (1, +1), (4, +1)]},
             {"medium": (1.0, 1.0), "bounds": [(0, -1), (2, +1)]},
             {"medium": (1.0, 1.0), "bounds": [(1, -1), (3, +1)]}],
    freq_hz=F)
out = _solve_multiregion(sys_, F, ASPECTS, None, 1e-6, 1, None, None, "t", {})
w = worst_db(out, pec)
gate("two independent eps=1 patches == PEC Mie <= 0.2 dB", w < 0.2,
     f"(worst {w:.3f}, junctions {out['n_junctions']})")

print("ALL JUNCTION GATES PASS" if ok else "JUNCTION GATES FAILED")
