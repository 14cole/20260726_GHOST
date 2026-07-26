"""Banded (side-by-side) coating gates — phase 9.

The user configuration that drove this: a PEC rocket with alternating
MAGRAM bands, i.e. TYPE 5 walls where band touches band, TYPE 4 covered
core under each band, and bare TYPE 2 elsewhere ({2,3,4,5} mixed).  Each
band is its own region in the generic multi-region assembler; walls are
interfaces between two coating regions; wall-core meetings are conductor
junctions and wall-outer meetings are dielectric triple junctions.

Known artifact (documented, not a bug): with LOSSLESS bands the fictitious
band cavity has discrete interior resonances; a mesh whose detuned
eigenfrequency lands on the solve frequency shows a narrow ~0.2 dB bump
(verified frequency-jitter-localized).  Real lossy MAGRAMs damp it.  Gates
below sit at jitter-checked mesh/frequency combinations.
"""
import math
import sys
import warnings

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from bor_solver import _MultiRegionBor, _solve_multiregion
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
F = 2.0 * M.C0 / (2 * math.pi * B)
THW = math.pi / 2
ASPECTS = [0.0, 45.0, 90.0, 135.0, 180.0]
PEC = M.sigma_pec_sphere(B, F)


def solve_two_band(epsA, muA, epsB, muB, scale):
    n_arc, n_w = 12 * scale, 5 * scale
    outA = arc(B + T, 0.0, THW, n_arc)
    outB = arc(B + T, THW, math.pi, n_arc)
    covA = arc(B, 0.0, THW, n_arc)
    covB = arc(B, THW, math.pi, n_arc)
    wall = np.column_stack([np.linspace(B, B + T, n_w), np.zeros(n_w)])
    sys_ = _MultiRegionBor(
        surfaces=[(outA, False), (outB, False), (wall, False),
                  (covA, True), (covB, True)],
        regions=[{"medium": None, "exterior": True,
                  "bounds": [(0, +1), (1, +1)]},
                 {"medium": (epsA, muA), "bounds": [(0, -1), (2, -1), (3, +1)]},
                 {"medium": (epsB, muB), "bounds": [(1, -1), (2, +1), (4, +1)]}],
        freq_hz=F)
    return _solve_multiregion(sys_, F, ASPECTS, None, 1e-6, 1, None, None,
                              "banded", {})


def worst(out, ref):
    return max(abs(10 * math.log10(s / ref))
               for s in out["sigma_vv"] + out["sigma_hh"])


# 1. Two eps=1 bands with a TYPE-5 wall == bare PEC Mie, mesh-converging.
w1 = worst(solve_two_band(1.0, 1.0, 1.0, 1.0, 1), PEC)
w2 = worst(solve_two_band(1.0, 1.0, 1.0, 1.0, 2), PEC)
gate("two eps=1 bands + wall == PEC Mie <= 0.2 dB", w1 < 0.2, f"(worst {w1:.3f})")
gate("eps=1 bands mesh-converging", w2 < w1, f"({w1:.3f} -> {w2:.3f} at 2x)")

# 2. Same magram in both bands (fictitious wall) == full single-coating Mie.
eA, uA = 1.8 - 0.3j, 1.6 - 0.9j
ref = M.sigma_coated_pec_sphere(B, B + T, eA, uA, F)
w = worst(solve_two_band(eA, uA, eA, uA, 2), ref)
gate("same-magram bands (fictitious wall) == coated Mie <= 0.1 dB", w < 0.1,
     f"(worst {w:.3f})")

# 3. Band swap == mirror (different lossy magrams).
eB, uB = 3.0 - 1.0j, 1.2 - 0.4j
oa = solve_two_band(eA, uA, eB, uB, 2)
ob = solve_two_band(eB, uB, eA, uA, 2)
d = max(abs(10 * math.log10(x / y)) for x, y in
        zip(oa["sigma_vv"] + oa["sigma_hh"],
            list(reversed(ob["sigma_vv"])) + list(reversed(ob["sigma_hh"]))))
gate("band-swap mirror (discretization-limited) <= 2e-2 dB", d < 2e-2,
     f"({d:.1e})")

# 4. Dispatch: the full {2,3,4,5} banded layout (bands + bare PEC section).
def geo_lines(pts):
    return "\n".join(f"{float(p[0])!r} {float(p[1])!r} "
                     f"{float(q[0])!r} {float(q[1])!r}"
                     for p, q in zip(pts[:-1], pts[1:]))


TH1, TH2 = math.radians(60), math.radians(120)
covA = arc(B, 0, TH1, 8)
covB = arc(B, TH1, TH2, 8)
bare = arc(B, TH2, math.pi, 10)
outA = arc(B + T, 0, TH1, 8)
outB = arc(B + T, TH1, TH2, 8)
wAB = np.column_stack([np.linspace(B * math.sin(TH1), (B + T) * math.sin(TH1), 4),
                       np.linspace(B * math.cos(TH1), (B + T) * math.cos(TH1), 4)])
edgeB = np.column_stack([np.linspace((B + T) * math.sin(TH2), B * math.sin(TH2), 4),
                         np.linspace((B + T) * math.cos(TH2), B * math.cos(TH2), 4)])


def banded_geo(d1, d2, N=-60):
    return ("Title: banded\n"
            f"Segment: outA 3\nproperties: 3 {N} 0 1 0\n" + geo_lines(outA) +
            f"\nSegment: outB 3\nproperties: 3 {N} 0 2 0\n" + geo_lines(outB) +
            f"\nSegment: wallAB 5\nproperties: 5 {N} 0 1 2\n" + geo_lines(wAB) +
            f"\nSegment: edgeB 3\nproperties: 3 {N} 0 2 0\n" + geo_lines(edgeB) +
            f"\nSegment: covA 4\nproperties: 4 {N} 0 1 0\n" + geo_lines(covA) +
            f"\nSegment: covB 4\nproperties: 4 {N} 0 2 0\n" + geo_lines(covB) +
            f"\nSegment: bare 2\nproperties: 2 {N} 0 0 0\n" + geo_lines(bare) +
            f"\nIBCS_Resistances:\nDielectrics:\n{d1}\n{d2}\n")


t_, s_, i_, d_ = parse_geometry(banded_geo("1 1 0 1 0", "2 1 0 1 0"))
snap = build_geometry_snapshot(t_, s_, i_, d_)
r = solve_monostatic_rcs_bor(snap, [F / 1e9], ASPECTS, "VV",
                             geometry_units="meters", workers=8)
w = max(abs(s["rcs_db"] - 10 * math.log10(PEC)) for s in r["samples"])
gate("dispatch {2,3,4,5} banded eps=1 == PEC Mie <= 0.15 dB", w < 0.15,
     f"(worst {w:.3f}, {r['metadata']['formulation']})")

t_, s_, i_, d_ = parse_geometry(banded_geo("1 1.8 -0.3 1.6 -0.9",
                                           "2 3 -1 1.2 -0.4"))
snap = build_geometry_snapshot(t_, s_, i_, d_)
r = solve_monostatic_rcs_bor(snap, [F / 1e9], ASPECTS, "VV",
                             geometry_units="meters", workers=8)
res = r["metadata"]["per_frequency"][0]["linear_residual"]
gate("dispatch two-magram banded solves cleanly",
     all(np.isfinite(s["rcs_db"]) for s in r["samples"]) and res < 1e-10,
     f"(residual {res:.1e})")

print("ALL BANDED GATES PASS" if ok else "BANDED GATES FAILED")
